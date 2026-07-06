"""
run_codex_rescue.py - Rescue CODEX→HE registrations with low dice.

CODEX-specific rescue (protein marker channel). Considerations:
  - Multi-channel image: uses the best_channel recorded in result.json
  - Partial FOV is common: rotation guard prevents wrong rotations when
    Step1 already found the correct orientation (dice >= 0.40)

Rotation guard:
  - current_dice >= 0.40: Step1 orientation is likely correct (partial FOV case).
    Only attempt rotation=0° (translation-only correction via phase correlation).
  - current_dice  < 0.40: Step1 clearly failed. Full NCC rotation scan (0-355°).

Pipeline:
  1. Load HE and CODEX best channel at WORK_SIZE.
  2. Build tissue masks for scoring.
  3. NCC rotation scan (full or 0°-only depending on rotation guard).
  4. Phase correlation for translation at each candidate angle.
  5. ECC refinement on best candidate.
  6. Save affine_M_rescue.npy + update result.json if dice improves.

Usage (standalone):
    python run_codex_rescue.py --he HE.tif --codex CODEX.tif --out output/ --best_ch 9
"""
import argparse
import json
import os
import traceback

import cv2
import numpy as np
import tifffile

WORK_SIZE        = 4000
ECC_SIZE         = 2000
COARSE_SCAN_SIZE = 256
ROTATION_GUARD_THRESHOLD = 0.40  # if Step1 dice >= this, skip rotation scan


# ─── Loading ──────────────────────────────────────────────────────────────────

def norm8(arr):
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def load_he_work(he_path, work_size=WORK_SIZE, log=print):
    log('  Loading HE...')
    he = tifffile.imread(he_path)
    if he.ndim == 3 and he.shape[0] in (1, 3, 4) and he.shape[0] < min(he.shape[1:]):
        he = np.moveaxis(he, 0, -1)
    h, w = he.shape[:2]
    scale = min(1.0, work_size / max(h, w))
    he_w = cv2.resize(he, (int(w * scale), int(h * scale)))
    log(f'  HE: {he.shape} → {he_w.shape} (scale={scale:.4f})')
    return he_w, scale


def load_codex_channel(codex_path, best_ch, work_size=WORK_SIZE, log=print):
    """Load a single CODEX channel at work_size. Returns (channel_uint8, scale)."""
    log(f'  Loading CODEX ch={best_ch}...')
    with tifffile.TiffFile(codex_path) as t:
        shape = t.series[0].shape

    # CHW: shape[0] is channels
    if len(shape) == 3 and shape[0] < shape[1] and shape[0] < shape[2]:
        ch_h, ch_w = shape[1], shape[2]
        page_idx = best_ch
    else:
        ch_h, ch_w = shape[-2], shape[-1]
        page_idx = 0

    scale = min(1.0, work_size / max(ch_h, ch_w))
    out_h, out_w = int(ch_h * scale), int(ch_w * scale)

    with tifffile.TiffFile(codex_path) as t:
        page = t.pages[page_idx].asarray().astype(np.float32)

    ch_norm = norm8(cv2.resize(page, (out_w, out_h)))
    log(f'  CODEX ch{best_ch}: {shape} → {ch_norm.shape} (scale={scale:.4f})')
    return ch_norm, scale


# ─── Tissue masks ─────────────────────────────────────────────────────────────

def make_he_tissue_mask(he_rgb):
    gray = cv2.cvtColor(he_rgb, cv2.COLOR_RGB2GRAY) if he_rgb.ndim == 3 else he_rgb
    inv = (255 - gray).astype(np.uint8)
    _, mask = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return mask


def make_codex_tissue_mask(codex_norm):
    _, mask = cv2.threshold(codex_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
    return mask


# ─── Rotation scan ────────────────────────────────────────────────────────────

def ncc_at_angle(he_s, codex_s, angle, cx, cy):
    if angle != 0:
        R = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        m = cv2.warpAffine(codex_s, R, (codex_s.shape[1], codex_s.shape[0]))
    else:
        m = codex_s
    he_n = he_s.astype(np.float64)
    m_n  = m.astype(np.float64)
    he_n = (he_n - he_n.mean()) / (he_n.std() + 1e-8)
    m_n  = (m_n  - m_n.mean())  / (m_n.std()  + 1e-8)
    return float((he_n * m_n).mean())


def estimate_rotation(he_tissue, codex_tissue, coarse_step=5, fine_range=12, fine_step=1, log=print):
    size = COARSE_SCAN_SIZE
    he_s    = cv2.resize(he_tissue,    (size, size)).astype(np.float32)
    codex_s = cv2.resize(codex_tissue, (size, size)).astype(np.float32)
    cx, cy  = size / 2.0, size / 2.0

    coarse_angles = list(range(0, 360, coarse_step))
    coarse_scores = [ncc_at_angle(he_s, codex_s, a, cx, cy) for a in coarse_angles]

    sorted_idx = np.argsort(coarse_scores)[::-1]
    top3 = [coarse_angles[i] for i in sorted_idx[:3]]
    log(f'  Coarse top-3: {top3} scores={[round(coarse_scores[i],4) for i in sorted_idx[:3]]}')

    best_angle, best_score = top3[0], -1.0
    for cand in top3:
        fine_angles = list(range(cand - fine_range, cand + fine_range + 1, fine_step))
        fine_scores = [ncc_at_angle(he_s, codex_s, a, cx, cy) for a in fine_angles]
        idx = int(np.argmax(fine_scores))
        if fine_scores[idx] > best_score:
            best_score  = fine_scores[idx]
            best_angle  = fine_angles[idx]

    log(f'  Best rotation: {best_angle}° (NCC={best_score:.4f})')
    return best_angle, best_score


# ─── Phase correlation + ECC ──────────────────────────────────────────────────

def phase_align(he_gray, codex_gray, angle_deg, log=print):
    h,  w  = he_gray.shape[:2]
    mh, mw = codex_gray.shape[:2]
    cx_m, cy_m = mw / 2.0, mh / 2.0

    R_mat    = cv2.getRotationMatrix2D((cx_m, cy_m), angle_deg, 1.0)
    codex_rot = cv2.warpAffine(codex_gray, R_mat, (mw, mh))
    codex_he  = cv2.resize(codex_rot, (w, h))

    (dx, dy), response = cv2.phaseCorrelate(
        he_gray.astype(np.float32),
        codex_he.astype(np.float32)
    )
    log(f'  Phase correlation: dx={dx:.1f}, dy={dy:.1f}, response={response:.4f}')

    scale_x = w / mw
    scale_y = h / mh
    R_3x3 = np.vstack([R_mat, [0, 0, 1]])
    S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float64)
    T = np.array([[1, 0, -dx],     [0, 1, -dy],     [0, 0, 1]], dtype=np.float64)
    M_3x3 = T @ S @ R_3x3
    return M_3x3[:2, :], codex_he


def ecc_refine(he_gray, codex_gray, M_init, scale=0.5, log=print):
    h, w = he_gray.shape[:2]
    sh, sw = int(h * scale), int(w * scale)
    he_s    = cv2.resize(he_gray,    (sw, sh)).astype(np.float32)
    codex_s = cv2.resize(codex_gray, (sw, sh)).astype(np.float32)
    M_s = M_init.astype(np.float64).copy()
    M_s[0, 2] *= scale;  M_s[1, 2] *= scale
    try:
        wm = M_s.copy().astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
        _, M_out = cv2.findTransformECC(he_s, codex_s, wm, cv2.MOTION_AFFINE,
                                        criteria, inputMask=None, gaussFiltSize=5)
        M_full = M_out.astype(np.float64)
        M_full[0, 2] /= scale;  M_full[1, 2] /= scale
        return M_full
    except Exception as e:
        log(f'  ECC failed: {e}, using M_init')
        return M_init


def quick_dice(a, b):
    a, b = (a > 0), (b > 0)
    return 2.0 * np.sum(a & b) / (np.sum(a) + np.sum(b) + 1e-10)


def score_M(he_tissue, codex_tissue, M, he_shape):
    h, w = he_shape[:2]
    warped = cv2.warpAffine(codex_tissue, M, (w, h))
    return quick_dice(he_tissue, warped)


# ─── Path-explicit entry point (used by run_codex_pipeline.py) ────────────────

def rescue_from_paths(he_path, codex_path, out_dir, best_ch=0, current_dice=0.0, log=print):
    """CODEX rescue with explicit paths. Returns (orig_dice, new_dice).

    Rotation guard: if current_dice >= ROTATION_GUARD_THRESHOLD (0.40),
    only attempts translation correction (rotation=0°), preserving Step1 orientation.
    Full rotation scan is only used when Step1 clearly failed (dice < 0.40).
    """
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, 'result.json')
    orig_dice   = current_dice

    he_work,    he_scale    = load_he_work(he_path, WORK_SIZE, log)
    codex_work, codex_scale = load_codex_channel(codex_path, best_ch, WORK_SIZE, log)

    he_tissue    = make_he_tissue_mask(he_work)
    codex_tissue = make_codex_tissue_mask(codex_work)
    he_gray = cv2.cvtColor(he_work, cv2.COLOR_RGB2GRAY) if he_work.ndim == 3 else he_work
    he_inv  = (255 - he_gray).astype(np.uint8)

    if orig_dice >= ROTATION_GUARD_THRESHOLD:
        log(f'  dice={orig_dice:.4f} >= {ROTATION_GUARD_THRESHOLD}: '
            f'translation-only rescue (rotation guard active)')
        candidates = [0]
    else:
        log('  Coarse rotation scan (0-355° NCC)...')
        best_angle, _ = estimate_rotation(he_tissue, codex_tissue, log=log)
        candidates = sorted(set([best_angle, 0, 90, 180, 270]))

    log(f'  Candidates: {candidates}')
    best_dice = orig_dice;  best_M_work = None;  best_candidate = None

    for angle_deg in candidates:
        M_phase, codex_he = phase_align(he_inv, codex_work, angle_deg, log=log)
        d = score_M(he_tissue, codex_tissue, M_phase, he_tissue.shape)
        log(f'  Phase angle={angle_deg}°: dice={d:.4f}')
        if d > best_dice:
            best_dice = d;  best_M_work = M_phase;  best_candidate = angle_deg

    if best_M_work is not None:
        codex_he_best = cv2.resize(codex_work, (he_inv.shape[1], he_inv.shape[0]))
        M_ecc = ecc_refine(he_inv, codex_he_best, best_M_work, scale=0.5, log=log)
        d_ecc = score_M(he_tissue, codex_tissue, M_ecc, he_tissue.shape)
        log(f'  ECC dice: {d_ecc:.4f}')
        if d_ecc > best_dice:
            best_dice = d_ecc;  best_M_work = M_ecc

    if best_M_work is None or best_dice <= orig_dice:
        log(f'  No improvement ({orig_dice:.4f} → {best_dice:.4f})')
        return orig_dice, orig_dice

    ratio  = codex_scale / he_scale
    M_full = best_M_work.copy().astype(np.float64)
    M_full[0, 0] *= ratio;  M_full[0, 1] *= ratio
    M_full[1, 0] *= ratio;  M_full[1, 1] *= ratio
    M_full[0, 2] /= he_scale;  M_full[1, 2] /= he_scale
    theta = np.degrees(np.arctan2(M_full[1, 0], M_full[0, 0]))
    log(f'  Improved: {orig_dice:.4f} → {best_dice:.4f} '
        f'(angle~{theta:.1f}°, candidate={best_candidate}°)')

    np.save(os.path.join(out_dir, 'affine_M_rescue.npy'), M_full)

    # ── Compute multi-metric quality ──────────────────────────────────────
    try:
        from quality_metrics import compute_quality_metrics
        warped_mask  = cv2.warpAffine(codex_tissue, best_M_work, (he_tissue.shape[1], he_tissue.shape[0]))
        warped_norm  = cv2.warpAffine(codex_work,   best_M_work, (he_inv.shape[1], he_inv.shape[0]))
        qm = compute_quality_metrics(he_tissue, warped_mask, he_inv, warped_norm)
        log(f'  Quality: dice={qm["tissue_dice"]:.4f} bnd_dice={qm["boundary_dice"]:.4f} '
            f'ncc={qm["ncc"]:.4f} hd={qm["hausdorff_mean"]:.1f}px '
            f'inlier_ratio={qm["inlier_ratio"]:.4f} ({qm["n_inliers"]}/{qm["n_matches"]})')
    except Exception as e:
        log(f'  Multi-metric scoring failed: {e}')
        qm = {'tissue_dice': round(best_dice, 4)}

    existing = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
    existing['quality_metrics'] = qm
    existing.setdefault('quality_global', {})['dice'] = round(best_dice, 4)
    existing['M_full']      = M_full.tolist()
    existing['method']      = f'rescue_codex_rot{best_candidate}'
    existing['angle_deg']   = round(theta, 2)
    existing['orig_dice']   = round(orig_dice, 4)
    existing['best_channel'] = best_ch
    with open(result_path, 'w') as f:
        json.dump(existing, f, indent=2)

    return orig_dice, best_dice


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CODEX→HE rescue registration')
    parser.add_argument('--he',      required=True)
    parser.add_argument('--codex',   required=True)
    parser.add_argument('--out',     required=True)
    parser.add_argument('--best_ch', type=int, default=0)
    parser.add_argument('--current_dice', type=float, default=0.0)
    args = parser.parse_args()

    orig, new = rescue_from_paths(
        he_path=args.he, codex_path=args.codex, out_dir=args.out,
        best_ch=args.best_ch, current_dice=args.current_dice)
    print(f'Result: {orig:.4f} → {new:.4f}')


if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    main()
