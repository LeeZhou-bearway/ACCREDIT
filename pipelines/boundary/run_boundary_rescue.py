"""
run_boundary_rescue.py - Rescue Xenium boundary-to-HE registrations with low global dice.

Why ORB fails: filled_image is a sparse binary mask — no ORB keypoints, 4 inliers
from 25K matches is meaningless. The boundary and HE coordinate systems are often
rotated by ~90°.

Core idea (boundary-specific, completely standalone):
  - HE nuclear channel: blue channel of HE (hematoxylin stains nuclei blue)
    → captures WHERE cells are in HE space
  - Boundary density map: Gaussian-blur the filled contour mask
    → captures WHERE cells are in Xenium parquet space
  Both encode the same biological feature (cell positions) in their respective
  coordinate systems. NCC + phase correlation on these is far more robust than
  ORB on a binary mask.

Pipeline:
  1. HE → blue channel (nuclear signal) at WORK_SIZE.
  2. filled_image.png → Gaussian-blurred density map at WORK_SIZE.
  3. Build tissue masks for scoring (from both sources).
  4. NCC rotation scan (0-355° coarse → fine ±12°) on density maps → top candidates.
  5. Phase correlation for translation, for {NCC_best, 0°, 90°, 180°, 270°}.
  6. ECC refinement on best candidate (nuclear channel vs warped density).
  7. Save affine_M_rescue.npy + update result.json if global dice improves.

Does NOT import or call run_xenium_reg.py. Completely standalone.

Usage:
    python run_boundary_rescue.py                    # rescue global_dice < 0.5
    python run_boundary_rescue.py --threshold 0.4
    python run_boundary_rescue.py --dataset Xenium_FFPE_Human_Breast_Cancer_Rep1
"""
import argparse
import json
import os
import time
import traceback

import cv2
import numpy as np
import tifffile

XENIUM_BASE = '/home/zhoule/workspace/Image_Reg/benchmark_data/DAPI/Xenium'
WORK_SIZE   = 2000   # working resolution
SCAN_SIZE   = 256    # NCC scan resolution
DENSITY_SIGMA_RATIO = 0.02   # Gaussian sigma = SCAN_SIZE * this (≈5px at 256)


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_he_work(he_path, work_size=WORK_SIZE, log=print):
    """Load HE at work_size. Returns (he_rgb_work, scale)."""
    log('  Loading HE...')
    he = tifffile.imread(he_path)
    if he.ndim == 3 and he.shape[0] in (1, 3, 4) and he.shape[0] < min(he.shape[1:]):
        he = np.moveaxis(he, 0, -1)
    h, w = he.shape[:2]
    scale = min(1.0, work_size / max(h, w))
    he_w = cv2.resize(he, (int(w * scale), int(h * scale)))
    log(f'  HE: {he.shape} → {he_w.shape} (scale={scale:.4f})')
    return he_w, scale


def load_filled_work(filled_path, work_size=WORK_SIZE, log=print):
    """Load cached filled_image.png at work_size. Returns (filled_work, scale).
    Uses PIL to avoid cv2 pixel-count limit on large images."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None   # disable decompression bomb check for large tissue images
    log('  Loading filled_image.png...')
    if not os.path.exists(filled_path):
        raise FileNotFoundError(f'filled_image.png not found: {filled_path}')
    img = Image.open(filled_path).convert('L')
    w, h = img.size          # PIL: (width, height)
    scale = min(1.0, work_size / max(h, w))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img_small = img.resize((new_w, new_h), Image.LANCZOS)
    filled_w = np.array(img_small)
    log(f'  Filled: ({h},{w}) → {filled_w.shape} (scale={scale:.4f})')
    return filled_w, scale


# ─── Feature extraction ───────────────────────────────────────────────────────

def he_nuclear_channel(he_rgb):
    """
    Extract nuclear signal from H&E: hematoxylin stains nuclei blue.
    Blue channel is dominant for nuclei; subtracting red suppresses eosin (pink/red).
    Returns uint8 image where high values = high nuclear density.
    """
    b = he_rgb[:, :, 2].astype(np.float32)
    r = he_rgb[:, :, 0].astype(np.float32)
    nuclear = np.clip(b.astype(np.float32) - 0.5 * r, 0, 255).astype(np.uint8)
    return nuclear


def boundary_density_map(filled_work, sigma_ratio=DENSITY_SIGMA_RATIO):
    """
    Convert sparse binary boundary mask to smooth cell-density map.
    Gaussian blur spreads each cell's signal so it's matchable across scales.
    sigma ≈ 2% of SCAN_SIZE ≈ 5px at 256px (covers a cell neighborhood).
    """
    sigma = max(1.0, filled_work.shape[0] * sigma_ratio)
    blurred = cv2.GaussianBlur(filled_work.astype(np.float32), (0, 0), sigma)
    mn, mx = blurred.min(), blurred.max()
    if mx == mn:
        return np.zeros_like(filled_work, dtype=np.uint8)
    return ((blurred - mn) / (mx - mn) * 255).astype(np.uint8)


# ─── Tissue masks (for scoring) ───────────────────────────────────────────────

def make_he_tissue_mask(he_rgb):
    """Tissue mask from HE: invert gray → Otsu → dilate."""
    gray = cv2.cvtColor(he_rgb, cv2.COLOR_RGB2GRAY) if he_rgb.ndim == 3 else he_rgb
    _, mask = cv2.threshold((255 - gray).astype(np.uint8), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return mask


def make_filled_tissue_mask(filled_work):
    """Tissue mask from boundary fill: Otsu on density → dilate."""
    density = boundary_density_map(filled_work, sigma_ratio=0.05)  # coarser for tissue
    _, mask = cv2.threshold(density, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
    return mask


# ─── NCC rotation scan ────────────────────────────────────────────────────────

def ncc_at_angle(ref_s, mov_s, angle, cx, cy, flip_y=False):
    """NCC between ref and mov rotated by angle (both at SCAN_SIZE).
    If flip_y=True, flip mov vertically before rotating."""
    m = mov_s
    if flip_y:
        m = np.flipud(m).copy()
    if angle != 0:
        R = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        m = cv2.warpAffine(m, R, (m.shape[1], m.shape[0]))
    ref_n = ref_s.astype(np.float64)
    m_n   = m.astype(np.float64)
    ref_n = (ref_n - ref_n.mean()) / (ref_n.std() + 1e-8)
    m_n   = (m_n   - m_n.mean())   / (m_n.std()   + 1e-8)
    return float((ref_n * m_n).mean())


def estimate_rotation(he_feat, filled_feat, coarse_step=5, fine_range=12,
                      fine_step=1, log=print):
    """
    Two-stage NCC rotation scan on nuclear-channel vs boundary-density features.
    Returns (best_angle_deg, best_ncc_score).
    """
    sz = SCAN_SIZE
    ref_s = cv2.resize(he_feat,     (sz, sz)).astype(np.float32)
    mov_s = cv2.resize(filled_feat, (sz, sz)).astype(np.float32)
    cx, cy = sz / 2.0, sz / 2.0

    coarse_angles = list(range(0, 360, coarse_step))
    coarse_scores = [ncc_at_angle(ref_s, mov_s, a, cx, cy) for a in coarse_angles]

    sorted_idx = np.argsort(coarse_scores)[::-1]
    top3 = [coarse_angles[i] for i in sorted_idx[:3]]
    log(f'  Coarse top-3: {top3}  scores={[round(coarse_scores[i],4) for i in sorted_idx[:3]]}')

    best_angle, best_score = top3[0], -1.0
    for cand in top3:
        fine_angles = list(range(cand - fine_range, cand + fine_range + 1, fine_step))
        fine_scores = [ncc_at_angle(ref_s, mov_s, a, cx, cy) for a in fine_angles]
        idx = int(np.argmax(fine_scores))
        if fine_scores[idx] > best_score:
            best_score = fine_scores[idx]
            best_angle = fine_angles[idx]

    log(f'  Best rotation: {best_angle}°  (NCC={best_score:.4f})')
    return best_angle, best_score


# ─── Phase correlation ────────────────────────────────────────────────────────

def phase_align(he_feat, filled_feat, angle_deg, log=print):
    """
    Rotate filled_feat by angle_deg (at filled size), resize to he size,
    then phase-correlate to find translation.
    Returns M (2×3, filled_work_coords → he_work_coords).
    """
    h, w   = he_feat.shape[:2]
    fh, fw = filled_feat.shape[:2]
    cx_f, cy_f = fw / 2.0, fh / 2.0

    R_mat       = cv2.getRotationMatrix2D((cx_f, cy_f), angle_deg, 1.0)
    filled_rot  = cv2.warpAffine(filled_feat, R_mat, (fw, fh))
    filled_he   = cv2.resize(filled_rot, (w, h))

    (dx, dy), response = cv2.phaseCorrelate(
        he_feat.astype(np.float32),
        filled_he.astype(np.float32)
    )
    log(f'  Phase: dx={dx:.1f}, dy={dy:.1f}, response={response:.4f}')

    scale_x = w / fw
    scale_y = h / fh
    R_3x3 = np.vstack([R_mat, [0, 0, 1]])
    S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float64)
    T = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]], dtype=np.float64)
    M_3x3 = T @ S @ R_3x3
    return M_3x3[:2, :], filled_he


# ─── Scoring ─────────────────────────────────────────────────────────────────

def quick_dice(a, b):
    a = (a > 0)
    b = (b > 0)
    return float(2.0 * np.sum(a & b) / (np.sum(a) + np.sum(b) + 1e-10))


def score_M(he_tissue, filled_tissue_work, M, he_shape):
    """Warp filled_tissue_work with M → dice vs he_tissue (both at work size)."""
    h, w = he_shape[:2]
    warped = cv2.warpAffine(filled_tissue_work.astype(np.float32), M, (w, h))
    return quick_dice(he_tissue, warped)


# ─── Main rescue ──────────────────────────────────────────────────────────────

def rescue_one(dataset_name, threshold=0.5, log=print):
    ds_dir      = os.path.join(XENIUM_BASE, dataset_name)
    out_dir     = os.path.join(ds_dir, 'output')
    result_path = os.path.join(out_dir, 'result.json')
    filled_path = os.path.join(out_dir, 'filled_image.png')

    if not os.path.exists(result_path):
        log('  No result.json — skipping')
        return None, None
    if not os.path.exists(filled_path):
        log('  No filled_image.png — skipping')
        return None, None

    with open(result_path) as f:
        result = json.load(f)

    orig_dice = result.get('quality_global', {}).get('dice', 0.0)
    log(f'  Current global dice: {orig_dice:.4f}')

    if orig_dice >= threshold:
        log(f'  >= {threshold}, skipping')
        return orig_dice, orig_dice

    he_path = result.get('he_path', '')
    if not os.path.exists(he_path):
        he_path = os.path.join(ds_dir, 'he_image.ome.tif')
    if not os.path.exists(he_path):
        log(f'  HE not found')
        return None, None

    # ── Load ──────────────────────────────────────────────────────────────────
    he_work,     he_scale     = load_he_work(he_path, WORK_SIZE, log)
    filled_work, filled_scale = load_filled_work(filled_path, WORK_SIZE, log)

    # ── Feature extraction ────────────────────────────────────────────────────
    # Nuclear channel from HE (where nuclei are in HE space)
    he_nuclear  = he_nuclear_channel(he_work)
    # Cell density from filled boundary mask (where cells are in Xenium space)
    filled_density = boundary_density_map(filled_work)

    # Tissue masks for global dice scoring
    he_tissue     = make_he_tissue_mask(he_work)
    filled_tissue = make_filled_tissue_mask(filled_work)

    he_cov = np.mean(he_tissue > 0) * 100
    fi_cov = np.mean(filled_tissue > 0) * 100
    log(f'  Coverage: HE={he_cov:.1f}%, Filled={fi_cov:.1f}%')

    # ── Step 1: NCC rotation scan on nuclear vs density ───────────────────────
    log('  Step 1: NCC rotation scan (nuclear channel vs boundary density)...')
    best_angle, ncc_score = estimate_rotation(he_nuclear, filled_density, log=log)
    candidates = sorted(set([best_angle, 0, 90, 180, 270]))
    log(f'  Candidates to test: {candidates}')

    best_dice    = orig_dice
    best_M_work  = None
    best_cand    = None
    best_filled_he = None   # keep the warped filled at best angle for ECC

    for angle_deg in candidates:
        # ── Step 2: Phase correlation ─────────────────────────────────────────
        M_phase, filled_he = phase_align(he_nuclear, filled_density, angle_deg, log=log)

        # Score against tissue masks
        d = score_M(he_tissue, filled_tissue, M_phase, he_tissue.shape)
        log(f'  Phase angle={angle_deg}°: global_dice={d:.4f}')

        if d > best_dice:
            best_dice   = d
            best_M_work = M_phase
            best_cand   = angle_deg
            best_filled_he = filled_he

    # ── Step 3: Fine angle search around best candidate ───────────────────────
    # ECC is unreliable for binary/sparse masks; instead do a fine ±10° search
    # using phase correlation to refine the rotation estimate.
    if best_M_work is not None:
        log(f'  Step 3: Fine angle search ±10° around {best_cand}°...')
        fine_angles = [a for a in range(best_cand - 10, best_cand + 11, 2)
                       if a not in candidates]
        for angle_deg in fine_angles:
            M_fine, _ = phase_align(he_nuclear, filled_density, angle_deg, log=log)
            d_fine = score_M(he_tissue, filled_tissue, M_fine, he_tissue.shape)
            if d_fine > best_dice:
                best_dice   = d_fine
                best_M_work = M_fine
                best_cand   = angle_deg
        log(f'  After fine search: best_angle={best_cand}°, global_dice={best_dice:.4f}')

    if best_M_work is None or best_dice <= orig_dice:
        log(f'  No improvement ({orig_dice:.4f} → {best_dice:.4f})')
        return orig_dice, orig_dice

    # ── Scale M_work (filled_work→he_work) → M_full (filled_full→he_full) ────
    ratio  = filled_scale / he_scale
    M_full = best_M_work.copy().astype(np.float64)
    M_full[0, 0] *= ratio;  M_full[0, 1] *= ratio
    M_full[1, 0] *= ratio;  M_full[1, 1] *= ratio
    M_full[0, 2] /= he_scale
    M_full[1, 2] /= he_scale

    theta = np.degrees(np.arctan2(M_full[1, 0], M_full[0, 0]))
    log(f'  Improved: {orig_dice:.4f} → {best_dice:.4f} (angle~{theta:.1f}°, candidate={best_cand}°)')

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, 'affine_M_rescue.npy'), M_full)

    result['orig_global_dice']          = round(orig_dice, 4)
    result['affine_M']                  = M_full.tolist()
    result['method']                    = f'rescue_rot{best_cand}'
    result['rescue_angle_deg']          = round(theta, 2)
    result['quality_global']['dice']    = round(best_dice, 4)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    return orig_dice, best_dice


# ─── Path-explicit entry point (used by run_pipeline.py) ─────────────────────

def rescue_from_paths(he_path, filled_path, out_dir, current_dice=0.0, log=print):
    """Boundary rescue with explicit paths. Returns (orig_dice, new_dice).

    Same logic as rescue_one() but accepts paths directly instead of dataset_name.
    Used by run_pipeline.py to chain rescue steps without hardcoded directory structure.
    """
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, 'result.json')
    orig_dice   = current_dice

    if not os.path.exists(filled_path):
        log(f'  No filled_image.png at {filled_path} — skipping boundary rescue')
        return orig_dice, orig_dice

    he_work,     he_scale     = load_he_work(he_path, WORK_SIZE, log)
    filled_work, filled_scale = load_filled_work(filled_path, WORK_SIZE, log)

    he_nuclear     = he_nuclear_channel(he_work)
    filled_density = boundary_density_map(filled_work)
    he_tissue      = make_he_tissue_mask(he_work)
    filled_tissue  = make_filled_tissue_mask(filled_work)

    best_angle, _ = estimate_rotation(he_nuclear, filled_density, log=log)
    candidates = sorted(set([best_angle, 0, 90, 180, 270]))
    log(f'  Candidates: {candidates}')

    best_dice = orig_dice; best_M_work = None; best_cand = None

    for angle_deg in candidates:
        M_phase, _ = phase_align(he_nuclear, filled_density, angle_deg, log=log)
        d = score_M(he_tissue, filled_tissue, M_phase, he_tissue.shape)
        log(f'  Phase angle={angle_deg}°: global_dice={d:.4f}')
        if d > best_dice:
            best_dice = d; best_M_work = M_phase; best_cand = angle_deg

    if best_M_work is not None:
        fine_angles = [a for a in range(best_cand - 10, best_cand + 11, 2)
                       if a not in candidates]
        for angle_deg in fine_angles:
            M_fine, _ = phase_align(he_nuclear, filled_density, angle_deg, log=log)
            d_fine = score_M(he_tissue, filled_tissue, M_fine, he_tissue.shape)
            if d_fine > best_dice:
                best_dice = d_fine; best_M_work = M_fine; best_cand = angle_deg
        log(f'  After fine search: best_angle={best_cand}°, global_dice={best_dice:.4f}')

    if best_M_work is None or best_dice <= orig_dice:
        log(f'  No improvement ({orig_dice:.4f} → {best_dice:.4f})')
        return orig_dice, orig_dice

    ratio  = filled_scale / he_scale
    M_full = best_M_work.copy().astype(np.float64)
    M_full[0, 0] *= ratio;  M_full[0, 1] *= ratio
    M_full[1, 0] *= ratio;  M_full[1, 1] *= ratio
    M_full[0, 2] /= he_scale;  M_full[1, 2] /= he_scale
    theta = np.degrees(np.arctan2(M_full[1, 0], M_full[0, 0]))
    log(f'  Improved: {orig_dice:.4f} → {best_dice:.4f} (angle~{theta:.1f}°, candidate={best_cand}°)')

    np.save(os.path.join(out_dir, 'affine_M_rescue.npy'), M_full)

    result = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
    result.setdefault('quality_global', {})['dice'] = round(best_dice, 4)
    result['orig_global_dice'] = round(orig_dice, 4)
    result['affine_M']         = M_full.tolist()
    result['method']           = f'rescue_boundary_rot{best_cand}'
    result['rescue_angle_deg'] = round(theta, 2)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    return orig_dice, best_dice


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Rescue if global dice < threshold (default 0.5)')
    parser.add_argument('--dataset', default=None)
    args = parser.parse_args()

    datasets = sorted([
        d for d in os.listdir(XENIUM_BASE)
        if os.path.isdir(os.path.join(XENIUM_BASE, d))
    ])
    if args.dataset:
        datasets = [args.dataset]

    log_path  = os.path.join(os.path.dirname(XENIUM_BASE), 'benchmark_boundary_rescue.log')
    log_file  = open(log_path, 'w', buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    results = []
    for ds in datasets:
        result_path = os.path.join(XENIUM_BASE, ds, 'output', 'result.json')
        if not os.path.exists(result_path):
            log(f'Skip {ds}: no result.json')
            continue

        with open(result_path) as f:
            r = json.load(f)
        dice = r.get('quality_global', {}).get('dice', 0.0)

        if dice >= args.threshold:
            log(f'Skip {ds}: global_dice={dice:.4f} >= {args.threshold}')
            continue

        log(f'\n{"="*60}')
        log(f'Rescuing: {ds}  (global_dice={dice:.4f})')
        log(f'{"="*60}')
        t0 = time.time()
        try:
            orig, new = rescue_one(ds, threshold=args.threshold, log=log)
            results.append({'dataset': ds, 'orig_dice': orig, 'new_dice': new})
            log(f'  Done in {time.time()-t0:.0f}s')
        except Exception as e:
            log(f'ERROR: {e}\n{traceback.format_exc()}')
            results.append({'dataset': ds, 'error': str(e)})

    log(f'\n{"="*60}')
    log('RESCUE SUMMARY')
    log(f'{"="*60}')
    for r in results:
        name = (r['dataset'].replace('Xenium_', '').replace('_FFPE', '').replace('_FF', ''))
        if 'error' in r:
            log(f'  {name:<45} ERROR')
        elif r['orig_dice'] is None:
            log(f'  {name:<45} SKIPPED')
        else:
            flag = '↑' if r['new_dice'] > r['orig_dice'] else '='
            log(f'  {name:<45} {r["orig_dice"]:.4f} → {r["new_dice"]:.4f} {flag}')
    log_file.close()


if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    main()
