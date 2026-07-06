"""
run_he2he.py - Direct HE-to-HE registration pipeline (no LLM agent).

Uses H-channel, grayscale, and feature matching with automatic flip detection.
Pipeline: Stage1 (Similarity) → Stage2 (Boundary BSpline) → Stage3 (Tile Feature)

Key: For same-stain HE-HE, NCC (normalized cross-correlation) is used as the
primary metric to select the best orientation, since NMI can give false positives
when shape overlaps but internal structures don't match.

Usage:
    conda run -n pathml python run_he2he.py \
        --fixed HE2HE/Xenium5K_HE.tif \
        --moving HE2HE/CosMX_HE.tif HE2HE/VisiumHD_HE.tif \
        --output_dir HE2HE/output
"""
import argparse
import cv2
import json
import numpy as np
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _compute_ncc_after_transform(fixed_id, moving_id, transform_id):
    """Compute NCC between fixed and warped moving (grayscale).
    For same-stain HE-HE, NCC is the most reliable alignment metric."""
    import SimpleITK as sitk
    from registration.preprocess import get_image
    from registration.global_reg import get_transform, _numpy_to_sitk

    fixed_arr = get_image(fixed_id)
    moving_arr = get_image(moving_id)
    tx = get_transform(transform_id)

    # Convert to grayscale if needed
    if fixed_arr.ndim == 3:
        fixed_gray = cv2.cvtColor(fixed_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        fixed_gray = fixed_arr.astype(np.float32)
    if moving_arr.ndim == 3:
        moving_gray = cv2.cvtColor(moving_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        moving_gray = moving_arr.astype(np.float32)

    fixed_sitk = _numpy_to_sitk(fixed_gray, fixed_id)
    moving_sitk = _numpy_to_sitk(moving_gray, moving_id)

    resampled = sitk.Resample(moving_sitk, fixed_sitk, tx, sitk.sitkLinear, 0.0,
                               moving_sitk.GetPixelID())
    warped = sitk.GetArrayFromImage(resampled).astype(np.float32)

    # Compute NCC only in overlap region
    mask = (fixed_gray > 10) & (warped > 10)
    if mask.sum() < 100:
        return -1.0

    a = fixed_gray[mask]
    b = warped[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom < 1e-10:
        return 0.0
    return float(np.sum(a * b) / denom)


def _create_flipped_images(moving_id, moving_gray_id, moving_h_id, flip_code, flip_name):
    """Create flipped versions of moving images (RGB, gray, H).
    Returns (flip_rgb_id, flip_gray_id, flip_h_id)."""
    from registration.preprocess import get_image, store_image, get_meta, _meta_store

    result = {}
    for src_id in [moving_id, moving_gray_id, moving_h_id]:
        flip_id = f"{src_id}__{flip_name}"
        arr = get_image(src_id)
        store_image(flip_id, cv2.flip(arr, flip_code))
        meta = get_meta(src_id)
        if meta:
            _meta_store[flip_id] = dict(meta)
        result[src_id] = flip_id

    return result[moving_id], result[moving_gray_id], result[moving_h_id]


def _try_all_methods(fixed_gray_id, fixed_h_id, moving_gray_id, moving_h_id,
                     fixed_id, moving_id, label, log):
    """Try feature, similarity(gray), similarity(H), shape on given image pair.
    Returns list of (transform_id, nmi) for all successful attempts."""
    from registration.global_reg import (
        run_similarity_registration, run_shape_similarity,
        run_feature_global_registration,
    )

    results = []

    # Feature-based (AKAZE/ORB) on grayscale
    try:
        fid, fnmi = run_feature_global_registration(fixed_gray_id, moving_gray_id)
        if fid is not None:
            log(f'    [{label}] feature(gray)={fnmi:.4f}')
            results.append((fid, fnmi))
    except Exception as e:
        log(f'    [{label}] feature(gray) failed: {e}')

    # Feature-based on H-channel
    try:
        fid2, fnmi2 = run_feature_global_registration(fixed_h_id, moving_h_id)
        if fid2 is not None:
            log(f'    [{label}] feature(H)={fnmi2:.4f}')
            results.append((fid2, fnmi2))
    except Exception as e:
        log(f'    [{label}] feature(H) failed: {e}')

    # Similarity on grayscale
    try:
        sid, snmi = run_similarity_registration(fixed_gray_id, moving_gray_id)
        log(f'    [{label}] sim(gray)={snmi:.4f}')
        results.append((sid, snmi))
    except Exception as e:
        log(f'    [{label}] sim(gray) failed: {e}')

    # Similarity on H-channel
    try:
        sid2, snmi2 = run_similarity_registration(fixed_h_id, moving_h_id)
        log(f'    [{label}] sim(H)={snmi2:.4f}')
        results.append((sid2, snmi2))
    except Exception as e:
        log(f'    [{label}] sim(H) failed: {e}')

    # Shape similarity on RGB
    try:
        shid, shnmi = run_shape_similarity(fixed_id, moving_id)
        log(f'    [{label}] shape={shnmi:.4f}')
        results.append((shid, shnmi))
    except Exception as e:
        log(f'    [{label}] shape failed: {e}')

    return results


def register_one(fixed_path, moving_path, out_dir, log):
    from registration.preprocess import (
        load_wsi_thumbnail, to_grayscale, extract_hematoxylin,
        get_image, store_image,
        _image_store, _meta_store,
    )
    from registration.global_reg import (
        run_similarity_registration, run_shape_similarity,
        run_feature_global_registration,
        _transform_store, get_transform,
    )
    from registration.nonrigid_reg import run_boundary_correction, run_tile_feature_registration
    from registration.quality import assess_quality, compute_feature_metrics

    # Load images (auto fallback for plain TIFF)
    fixed_id, _ = load_wsi_thumbnail(fixed_path, level=3)
    moving_id, _ = load_wsi_thumbnail(moving_path, level=3)
    log(f'  fixed:  {fixed_id}')
    log(f'  moving: {moving_id}')

    # Extract grayscale and H-channel
    fixed_gray_id, _ = to_grayscale(fixed_id)
    moving_gray_id, _ = to_grayscale(moving_id)
    fixed_h_id, _ = extract_hematoxylin(fixed_id)
    moving_h_id, _ = extract_hematoxylin(moving_id)

    # ── Stage 1: Global Similarity ────────────────────────────────────────
    # Try all orientations (original + 3 flips) × all methods,
    # then select best by NCC (not NMI) to avoid false positives.
    all_candidates = []  # (transform_id, nmi, ncc, flip_name, moving_rgb_id)

    orientations = [
        (None, 'original', moving_id, moving_gray_id, moving_h_id),
    ]
    # Create flipped versions
    for flip_code, flip_name in [(1, 'hflip'), (0, 'vflip'), (-1, 'hvflip')]:
        f_rgb, f_gray, f_h = _create_flipped_images(
            moving_id, moving_gray_id, moving_h_id, flip_code, flip_name)
        orientations.append((flip_code, flip_name, f_rgb, f_gray, f_h))

    for _, orient_name, m_rgb, m_gray, m_h in orientations:
        log(f'  [{orient_name}]')
        results = _try_all_methods(
            fixed_gray_id, fixed_h_id, m_gray, m_h,
            fixed_id, m_rgb, orient_name, log)
        for tid, nmi in results:
            # Compute NCC for this candidate
            ncc = _compute_ncc_after_transform(fixed_id, m_rgb, tid)
            all_candidates.append((tid, nmi, ncc, orient_name, m_rgb))
            log(f'      → NCC={ncc:.4f}')

    # Select best by NCC (primary), with NMI as tiebreaker
    if not all_candidates:
        raise RuntimeError('All registration attempts failed')

    all_candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
    best_tid, best_nmi_s1, best_ncc, best_flip, best_moving_rgb = all_candidates[0]
    sim_id = best_tid
    nmi1 = best_nmi_s1
    used_flip = best_flip if best_flip != 'original' else None

    log(f'  Stage1 best: NMI={nmi1:.4f} NCC={best_ncc:.4f} orient={best_flip}')

    # Log top 3 for debugging
    for i, (tid, nmi, ncc, flip, _) in enumerate(all_candidates[:3]):
        log(f'    #{i+1}: {flip} NMI={nmi:.4f} NCC={ncc:.4f}')

    # Determine effective moving IDs for Stage2/3
    if used_flip:
        moving_id_eff = f"{moving_id}__{used_flip}"
        moving_gray_id_eff = f"{moving_gray_id}__{used_flip}"
    else:
        moving_id_eff = moving_id
        moving_gray_id_eff = moving_gray_id

    # ── Stage 2: Boundary Correction (RGB) ────────────────────────────────
    STAGE2_BW_LIST = [60, 100, 120]
    stage2_id = sim_id
    stage2_nmi = nmi1
    nmi2 = nmi1

    for bw in STAGE2_BW_LIST:
        try:
            bc_id_try, nmi2_try = run_boundary_correction(
                fixed_id, moving_id_eff, sim_id, boundary_width_px=bw)
            log(f'  Stage2 bw={bw} NMI={nmi2_try:.4f}')
            if nmi2_try > stage2_nmi:
                stage2_id = bc_id_try
                stage2_nmi = nmi2_try
                nmi2 = nmi2_try
                break
        except Exception as e:
            log(f'  Stage2 bw={bw} failed: {e}')
            continue

    if stage2_id is sim_id:
        log(f'  Stage2 all attempts degraded NMI, using Stage1')

    # ── Stage 3: Tile Feature Registration ────────────────────────────────
    tf_id, nmi3, ncc = run_tile_feature_registration(fixed_id, moving_id_eff, stage2_id)
    log(f'  Stage3={nmi3:.4f}  NCC={ncc:.4f}')
    if nmi3 > stage2_nmi:
        best_id = tf_id
        best_stage = 3
        best_nmi = nmi3
    else:
        best_id = stage2_id
        best_stage = 2 if stage2_id is not sim_id else 1
        best_nmi = stage2_nmi

    # ── Save transform ────────────────────────────────────────────────────
    import SimpleITK as sitk
    tx = get_transform(best_id)
    if isinstance(tx, sitk.CompositeTransform):
        tx.FlattenTransform()
    sitk.WriteTransform(tx, os.path.join(out_dir, 'transform.h5'))
    try:
        sitk.WriteTransform(tx, os.path.join(out_dir, 'transform.tfm'))
    except Exception:
        pass

    json.dump(
        {
            'fixed_path': fixed_path,
            'moving_path': moving_path,
            'mode': 'he-he',
            'flip': used_flip,
        },
        open(os.path.join(out_dir, 'transform_meta.json'), 'w'),
        indent=2,
    )

    # ── Quality assessment ────────────────────────────────────────────────
    assess_quality(fixed_gray_id, moving_gray_id_eff, best_id, out_dir)

    try:
        feat = compute_feature_metrics(fixed_id, moving_id_eff, best_id)
    except Exception:
        feat = {'h_ncc': 0.0, 'akaze_inliers': 0, 'edge_dice': 0.0}

    passed = best_nmi >= 1.10
    log(f'  Best=Stage{best_stage} NMI={best_nmi:.4f} passed={passed}')
    log(f'  h_ncc={feat["h_ncc"]:.4f}  akaze_inliers={feat["akaze_inliers"]}  edge_dice={feat["edge_dice"]:.4f}')

    return dict(
        fixed=fixed_path,
        moving=moving_path,
        flip=used_flip,
        nmi1=round(nmi1, 4),
        nmi2=round(nmi2, 4),
        nmi3=round(nmi3, 4),
        avg_tile_ncc=round(ncc, 4),
        best_stage=best_stage,
        best_nmi=round(best_nmi, 4),
        passed=passed,
        h_ncc=feat['h_ncc'],
        akaze_inliers=feat['akaze_inliers'],
        edge_dice=feat['edge_dice'],
    )


def main():
    parser = argparse.ArgumentParser(description='HE-to-HE registration')
    parser.add_argument('--fixed', required=True, help='Fixed/reference HE image')
    parser.add_argument('--moving', nargs='+', required=True,
                        help='One or more moving HE images to register to fixed')
    parser.add_argument('--output_dir', default='HE2HE/output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'run.log')
    log_file = open(log_path, 'w', buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f'HE-to-HE Registration')
    log(f'Fixed: {args.fixed}')
    log(f'Moving: {args.moving}')
    log(f'Output: {args.output_dir}')

    summary = []
    for i, moving in enumerate(args.moving, 1):
        from registration.preprocess import _image_store, _meta_store
        from registration.global_reg import _transform_store

        name = Path(moving).stem
        out_dir = os.path.join(args.output_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        log(f'\n=== [{i}/{len(args.moving)}] {name} ===')

        # Clear stores between runs
        _image_store.clear()
        _transform_store.clear()
        _meta_store.clear()

        try:
            row = register_one(args.fixed, moving, out_dir, log)
            summary.append(row)
        except Exception as e:
            tb = traceback.format_exc()
            log(f'  ERROR: {e}\n{tb}')
            summary.append(dict(moving=moving, error=str(e)))
            with open(os.path.join(out_dir, 'error.log'), 'w') as f:
                f.write(tb)

        with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    log('\n=== SUMMARY ===')
    for r in summary:
        if 'error' in r:
            log(f"  ERROR  {r.get('moving', '?')}")
        else:
            flag = 'PASS' if r['passed'] else 'FAIL'
            log(f"  {flag}  {Path(r['moving']).stem:20s}  NMI={r['best_nmi']}  "
                f"h_ncc={r.get('h_ncc','?')}  akaze={r.get('akaze_inliers','?')}  "
                f"edge_dice={r.get('edge_dice','?')}")

    passed = [r for r in summary if r.get('passed')]
    log(f'\nPassed: {len(passed)}/{len(summary)}')
    log_file.close()


if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    main()
