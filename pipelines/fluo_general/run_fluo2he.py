"""
run_fluo2he.py - Register fluorescence images (DAPI, CODEX, etc.) to HE.

Strategy:
  1. For each fluorescence input, auto-select the best matching channel
     by template matching all channels against HE (inverted grayscale).
  2. If fluo/HE resolution ratio is close to 1:1 (0.8–1.2), use ORB
     feature matching on binary masks (fast, accurate for same-resolution).
  3. Otherwise, use SimpleITK Similarity2D + NMI optimization
     (handles scale differences robustly).
  4. Refine with ECC if improvement detected.

Usage:
    # Full-FOV (default): fluo covers entire HE
    conda run -n pathml python run_fluo2he.py \
        --he DAPI-CODEX-HE/Xenium5K/Xenium5K_HE.tif \
        --fluo DAPI-CODEX-HE/Xenium5K/DAPI.tif \
               DAPI-CODEX-HE/Xenium5K/CODEX.tif \
        --output_dir DAPI-CODEX-HE/Xenium5K/output

    # ROI mode: fluo covers only part of HE
    conda run -n pathml python run_fluo2he.py \
        --he DAPI-CODEX-HE/VisiumHD/HE.tif \
        --fluo DAPI-CODEX-HE/VisiumHD/CODEX.tif \
        --output_dir DAPI-CODEX-HE/VisiumHD/output --roi
"""
import argparse
import os
import sys
import time
import json
import numpy as np
import cv2
import tifffile
from skimage.color import rgb2hed
import SimpleITK as sitk


# ── Utilities ──────────────────────────────────────────────────────────────

def norm8(img):
    """Normalize image to uint8 with percentile stretching."""
    f = img.astype(np.float64)
    if np.any(f > 0):
        p1, p99 = np.percentile(f[f > 0], [1, 99])
    else:
        p1, p99 = 0, 1
    return np.clip((f - p1) / (p99 - p1 + 1e-10) * 255, 0, 255).astype(np.uint8)


def quick_dice(a, b):
    """Fast Dice coefficient on binary masks."""
    a_pos = a > 0
    b_pos = b > 0
    inter = np.sum(a_pos & b_pos)
    total = np.sum(a_pos) + np.sum(b_pos)
    return 2.0 * inter / (total + 1e-10)


def he_to_inv_gray(he_img):
    """HE → inverted grayscale (tissue bright, like fluorescence)."""
    gray = cv2.cvtColor(he_img, cv2.COLOR_RGB2GRAY)
    return (255 - gray)


def he_to_h_channel(he_img):
    """HE → H-channel (hematoxylin, nuclei bright)."""
    hed = rgb2hed(he_img)
    h = hed[:, :, 0]
    return ((h - h.min()) / (h.max() - h.min() + 1e-10) * 255).astype(np.uint8)


# ── Channel Selection ──────────────────────────────────────────────────────

def select_best_channel(fluo_img, he_inv, log=print):
    """Auto-select the best matching channel from multi-channel fluorescence.

    Uses template matching (TM_CCOEFF_NORMED) at 0.25x scale to score each
    channel against HE inverted grayscale. Returns (best_channel_id, best_score,
    channel_image, all_scores).
    """
    if fluo_img.ndim == 2:
        score = _channel_tm_score(fluo_img, he_inv)
        log(f'  Single channel, TM score: {score:.4f}')
        return 0, score, fluo_img, {0: score}

    n_channels = fluo_img.shape[0]
    log(f'  Scoring {n_channels} channels against HE...')

    s = 0.25
    pad = 100
    he_s = cv2.resize(he_inv, None, fx=s, fy=s)
    he_padded = cv2.copyMakeBorder(he_s, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=0)

    scores = {}
    for i in range(n_channels):
        ch = norm8(fluo_img[i])
        ch_s = cv2.resize(ch, None, fx=s, fy=s)
        result = cv2.matchTemplate(he_padded, ch_s, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        scores[i] = round(max_val, 4)

    best_ch = max(scores, key=scores.get)
    best_score = scores[best_ch]
    log(f'  Channel scores: {scores}')
    log(f'  Best channel: ch{best_ch} (score={best_score:.4f})')

    return best_ch, best_score, fluo_img[best_ch], scores


def _channel_tm_score(ch_img, he_inv, s=0.25):
    pad = 100
    he_s = cv2.resize(he_inv, None, fx=s, fy=s)
    he_padded = cv2.copyMakeBorder(he_s, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=0)
    ch_s = cv2.resize(norm8(ch_img), None, fx=s, fy=s)
    result = cv2.matchTemplate(he_padded, ch_s, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val


# ── ORB Registration (same-resolution) ────────────────────────────────────

def register_orb(he_ref, fluo_norm, log=print):
    """ORB feature matching on binary masks. Returns (M, dice, info_dict).
    M maps fluo pixels → he_ref pixels (at the input resolution).
    """
    _, he_bin = cv2.threshold(he_ref, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fl_bin = cv2.threshold(fluo_norm, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    orb = cv2.ORB_create(nfeatures=100000)
    kp1, des1 = orb.detectAndCompute(he_bin, None)
    kp2, des2 = orb.detectAndCompute(fl_bin, None)

    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        log('  ORB: insufficient features')
        return None, 0, {}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC,
                                              ransacReprojThreshold=5.0)
    n_inliers = int(inliers.sum()) if inliers is not None else 0

    if M is None or n_inliers < 10:
        log(f'  ORB: only {n_inliers} inliers')
        return None, 0, {}

    warped = cv2.warpAffine(fl_bin, M, (he_bin.shape[1], he_bin.shape[0]))
    dice = quick_dice(he_bin, warped)

    theta = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
    info = {'method': 'ORB', 'inliers': n_inliers, 'matches': len(matches),
            'angle_deg': round(theta, 2), 'scale': round(scale, 4)}
    log(f'  ORB: {n_inliers} inliers / {len(matches)} matches, '
        f'angle={theta:.2f}°, scale={scale:.4f}, dice={dice:.4f}')

    return M, dice, info


# ── SimpleITK NMI Registration (cross-resolution) ─────────────────────────

def register_sitk_nmi(he_ref, fluo_norm, log=print, work_scale=0.2):
    """SimpleITK Similarity2D + Mattes MI. Returns (M, dice, info_dict).
    he_ref and fluo_norm should be at the SAME resolution (resize fluo to
    match he before calling). M maps fluo pixels → he pixels at that resolution.
    """
    s = work_scale
    ref_s = cv2.resize(he_ref, None, fx=s, fy=s)
    mov_s = cv2.resize(fluo_norm, None, fx=s, fy=s)

    fixed = sitk.GetImageFromArray(ref_s.astype(np.float64))
    moving = sitk.GetImageFromArray(mov_s.astype(np.float64))

    initial_tf = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.3)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=0.001, numberOfIterations=1000)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(initial_tf, inPlace=False)

    t0 = time.time()
    final_tf = reg.Execute(fixed, moving)
    dt = time.time() - t0

    params = final_tf.GetParameters()  # [scale, angle, tx, ty]
    center = final_tf.GetFixedParameters()  # [cx, cy]

    scale_est = params[0]
    angle_rad = params[1]
    theta = np.degrees(angle_rad)
    # Scale transform back to full resolution
    tx, ty = params[2] / s, params[3] / s
    cx, cy = center[0] / s, center[1] / s

    log(f'  NMI: scale={scale_est:.4f}, angle={theta:.3f}°, '
        f'tx={tx:.1f}, ty={ty:.1f}, metric={reg.GetMetricValue():.4f}, time={dt:.1f}s')

    # Build M for warpAffine: maps fluo → he (at input resolution)
    # SimpleITK transform maps fixed→moving, we need the inverse
    inv_scale = 1.0 / scale_est
    cos_inv = inv_scale * np.cos(-angle_rad)
    sin_inv = inv_scale * np.sin(-angle_rad)
    t_vec = np.array([tx, ty])
    c_vec = np.array([cx, cy])

    M = np.array([
        [cos_inv, -sin_inv,
         -cos_inv * (c_vec[0] + t_vec[0]) + sin_inv * (c_vec[1] + t_vec[1]) + c_vec[0]],
        [sin_inv, cos_inv,
         -sin_inv * (c_vec[0] + t_vec[0]) - cos_inv * (c_vec[1] + t_vec[1]) + c_vec[1]]
    ], dtype=np.float64)

    # Dice check
    _, he_bin = cv2.threshold(he_ref, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fl_bin = cv2.threshold(fluo_norm, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    warped = cv2.warpAffine(fl_bin, M, (he_bin.shape[1], he_bin.shape[0]))
    dice = quick_dice(he_bin, warped)
    log(f'  NMI dice: {dice:.4f}')

    info = {'method': 'SimpleITK_NMI_Similarity2D',
            'sitk_scale': round(scale_est, 6),
            'angle_deg': round(theta, 4),
            'tx': round(tx, 1), 'ty': round(ty, 1),
            'nmi_metric': round(reg.GetMetricValue(), 4)}

    return M, dice, info


# ── ECC Refinement ─────────────────────────────────────────────────────────

def ecc_refine(he_ref, fluo_norm, M_init, log=print, scale=0.25):
    """ECC refinement at reduced scale. Returns (M_refined, dice)."""
    h, w = he_ref.shape[:2]
    sh, sw = int(h * scale), int(w * scale)
    ref_s = cv2.resize(he_ref, (sw, sh)).astype(np.float32)

    M_s = M_init.copy().astype(np.float32)
    M_s[0, 2] *= scale
    M_s[1, 2] *= scale

    mov_warped = cv2.warpAffine(
        cv2.resize(fluo_norm, (sw, sh)).astype(np.float32),
        M_s, (sw, sh))

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
    try:
        residual = np.eye(2, 3, dtype=np.float32)
        _, residual = cv2.findTransformECC(ref_s, mov_warped, residual,
                                            cv2.MOTION_EUCLIDEAN, criteria)
        M_refined = M_init.copy()
        M_refined[0, 2] += residual[0, 2] / scale
        M_refined[1, 2] += residual[1, 2] / scale
        return M_refined
    except cv2.error as e:
        log(f'  ECC refinement failed: {e}')
        return M_init


# ── Main Pipeline ──────────────────────────────────────────────────────────

def register_one(he_img, fluo_path, out_dir, log=print):
    """Register one fluorescence image to HE.

    Args:
        he_img: HE image array (pre-loaded to avoid repeated I/O)
        fluo_path: path to fluorescence TIFF
    Returns result dict with transform, quality metrics, etc.
    """
    os.makedirs(out_dir, exist_ok=True)
    fluo_name = os.path.splitext(os.path.basename(fluo_path))[0]

    log(f'\n{"="*60}')
    log(f'Registering {fluo_name} → HE')
    log(f'  HE: {he_img.shape}')

    t0 = time.time()
    fluo_raw = tifffile.imread(fluo_path)
    log(f'  {fluo_name}: {fluo_raw.shape} loaded in {time.time()-t0:.1f}s')

    # 2. Prepare HE representations at fluo resolution
    if fluo_raw.ndim == 3 and fluo_raw.shape[0] < fluo_raw.shape[1]:
        # Multi-channel: (C, H, W)
        fluo_h, fluo_w = fluo_raw.shape[1], fluo_raw.shape[2]
    else:
        # Single channel: (H, W) or (H, W, C)
        fluo_h, fluo_w = fluo_raw.shape[0], fluo_raw.shape[1]

    he_h, he_w = he_img.shape[0], he_img.shape[1]
    res_ratio = (he_w / fluo_w + he_h / fluo_h) / 2.0
    log(f'  HE: {he_w}x{he_h}, Fluo: {fluo_w}x{fluo_h}, ratio={res_ratio:.4f}')

    # Work at a common reduced resolution to avoid OOM
    # Target: ~4000px on the long side
    work_long = 4000
    work_scale_he = work_long / max(he_w, he_h)
    work_scale_fluo = work_long / max(fluo_w, fluo_h)
    ws_h, ws_w = int(he_h * work_scale_he), int(he_w * work_scale_he)
    fs_h, fs_w = int(fluo_h * work_scale_fluo), int(fluo_w * work_scale_fluo)
    # Make both the same size (match to HE work size)
    log(f'  Work scale: HE {work_scale_he:.4f} ({ws_w}x{ws_h}), '
        f'Fluo {work_scale_fluo:.4f} ({fs_w}x{fs_h})')

    he_work = cv2.resize(he_img, (ws_w, ws_h))
    he_inv_work = he_to_inv_gray(he_work)
    he_h_ch_work = he_to_h_channel(he_work)

    # 3. Auto-select best channel (at work resolution)
    t0 = time.time()
    # For channel selection, resize fluo channels to HE work size
    if fluo_raw.ndim == 3 and fluo_raw.shape[0] < fluo_raw.shape[1]:
        # Multi-channel: create resized version for selection
        n_ch = fluo_raw.shape[0]
        fluo_work_all = np.stack([
            cv2.resize(fluo_raw[i], (ws_w, ws_h)) for i in range(n_ch)
        ])
        best_ch, tm_score, _, all_scores = select_best_channel(
            fluo_work_all, he_inv_work, log=log)
        fluo_ch_work = cv2.resize(fluo_raw[best_ch], (ws_w, ws_h))
        del fluo_work_all
    else:
        fluo_ch_work = cv2.resize(fluo_raw if fluo_raw.ndim == 2 else fluo_raw[:,:,0],
                                   (ws_w, ws_h))
        best_ch = 0
        tm_score = _channel_tm_score(fluo_ch_work, he_inv_work)
        all_scores = {0: round(tm_score, 4)}
        log(f'  Single channel, TM score: {tm_score:.4f}')

    fluo_norm_work = norm8(fluo_ch_work)
    log(f'  Channel selection time: {time.time()-t0:.1f}s')
    del fluo_raw  # free memory

    # 4. Choose registration method based on resolution ratio
    same_resolution = 0.8 < res_ratio < 1.2
    log(f'  Strategy: {"ORB (same resolution)" if same_resolution else "SimpleITK NMI (cross resolution)"}')

    # All registration happens at work resolution (ws_w × ws_h)
    t0 = time.time()
    if same_resolution:
        M_work, dice, info = register_orb(he_h_ch_work, fluo_norm_work, log=log)
        if M_work is None:
            log('  ORB failed, falling back to NMI...')
            M_work, dice, info = register_sitk_nmi(he_inv_work, fluo_norm_work, log=log)
    else:
        M_work, dice, info = register_sitk_nmi(he_inv_work, fluo_norm_work, log=log)

    if M_work is None:
        log(f'  Registration FAILED for {fluo_name}')
        return {'name': fluo_name, 'status': 'failed'}

    dt = time.time() - t0
    log(f'  Registration time: {dt:.1f}s')

    # 5. ECC refinement (at work resolution)
    log('  ECC refinement...')
    M_ecc = ecc_refine(he_inv_work, fluo_norm_work, M_work, log=log, scale=0.5)
    _, he_bin = cv2.threshold(he_inv_work, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fl_bin = cv2.threshold(fluo_norm_work, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    warped_ecc = cv2.warpAffine(fl_bin, M_ecc, (he_bin.shape[1], he_bin.shape[0]))
    dice_ecc = quick_dice(he_bin, warped_ecc)
    if dice_ecc > dice:
        log(f'  ECC improved dice: {dice:.4f} → {dice_ecc:.4f}')
        M_work = M_ecc
        dice = dice_ecc
    else:
        log(f'  ECC no improvement ({dice_ecc:.4f} vs {dice:.4f}), keeping original')

    # 6. Scale M_work to full coordinates: fluo_full_px → he_full_px
    # M_work maps fluo_work_px → he_work_px
    # fluo_work_px = fluo_full_px * work_scale_fluo
    # he_work_px = he_full_px * work_scale_he
    # So: he_full = (1/work_scale_he) * M_work * work_scale_fluo * fluo_full
    M_full = M_work.copy().astype(np.float64)
    scale_ratio = work_scale_fluo / work_scale_he
    M_full[0, 0] *= scale_ratio
    M_full[0, 1] *= scale_ratio
    M_full[1, 0] *= scale_ratio
    M_full[1, 1] *= scale_ratio
    M_full[0, 2] /= work_scale_he
    M_full[1, 2] /= work_scale_he

    # 7. Save overlay (at work resolution — already small)
    log('  Saving overlay...')
    warped_vis = cv2.warpAffine(fluo_norm_work, M_work,
                                 (ws_w, ws_h))
    overlay = he_work.copy()
    mask = warped_vis > 30
    overlay[mask] = (overlay[mask] * 0.6 +
                     np.array([0, 255, 0], dtype=np.uint8) * 0.4).astype(np.uint8)
    overlay_path = os.path.join(out_dir, f'overlay_{fluo_name}.png')
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    log(f'  Saved {overlay_path}')

    # 8. Save transform
    np.save(os.path.join(out_dir, f'affine_{fluo_name}.npy'), M_full)

    # ── Multi-metric quality scoring ────────────────────────────────────
    try:
        from quality_metrics import compute_quality_metrics
        he_gray_work = cv2.cvtColor(he_work, cv2.COLOR_RGB2GRAY) if he_work.ndim == 3 else he_work
        he_inv_work = (255 - he_gray_work).astype(np.uint8)
        warped_vis_q = cv2.warpAffine(fluo_norm_work, M_work, (ws_w, ws_h))
        qm = compute_quality_metrics(he_bin, warped_ecc if dice_ecc > dice else
                                     cv2.warpAffine(fl_bin, M_work, (he_bin.shape[1], he_bin.shape[0])),
                                     he_inv_work, warped_vis_q)
        log(f'  Quality: dice={qm["tissue_dice"]:.4f} bnd_dice={qm["boundary_dice"]:.4f} '
            f'ncc={qm["ncc"]:.4f} hd={qm["hausdorff_mean"]:.1f}px '
            f'inlier_ratio={qm["inlier_ratio"]:.4f} ({qm["n_inliers"]}/{qm["n_matches"]})')
    except Exception as e:
        log(f'  Multi-metric scoring skipped: {e}')
        qm = {'tissue_dice': round(dice, 4)}

    result = {
        'name': fluo_name,
        'fluo_path': fluo_path,
        'fluo_size': [fluo_w, fluo_h],
        'he_size': [he_w, he_h],
        'resolution_ratio': round(res_ratio, 4),
        'best_channel': int(best_ch),
        'tm_score': round(tm_score, 4),
        'channel_scores': {str(k): v for k, v in all_scores.items()},
        'dice': round(dice, 4),
        'quality_metrics': qm,
        'M_full': M_full.tolist(),
        **info,
    }
    log(f'  Final dice: {dice:.4f}')
    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Register fluorescence images (DAPI, CODEX, etc.) to HE')
    parser.add_argument('--he', required=True, help='Path to HE image (TIFF)')
    parser.add_argument('--fluo', nargs='+', required=True,
                        help='Path(s) to fluorescence image(s) (TIFF)')
    parser.add_argument('--output_dir', default='output',
                        help='Output directory')
    parser.add_argument('--roi', action='store_true',
                        help='Use ROI pipeline (fluo covers only part of HE)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = open(os.path.join(args.output_dir, 'run.log'), 'w', buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    mode = 'ROI' if args.roi else 'Full-FOV'
    log(f'Fluorescence → HE Registration ({mode})')
    log(f'HE: {args.he}')
    log(f'Fluorescence inputs: {args.fluo}')
    log(f'Output: {args.output_dir}')

    log('Loading HE image...')
    t0 = time.time()
    he_img = tifffile.imread(args.he)
    log(f'  HE: {he_img.shape} loaded in {time.time()-t0:.1f}s')

    # Select pipeline
    if args.roi:
        from run_fluo2he_roi import register_one_roi as reg_func
    else:
        reg_func = register_one

    results = {}
    for fluo_path in args.fluo:
        try:
            result = reg_func(he_img, fluo_path, args.output_dir, log=log)
            results[result['name']] = result
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log(f'ERROR processing {fluo_path}: {e}\n{tb}')
            name = os.path.splitext(os.path.basename(fluo_path))[0]
            results[name] = {'status': 'error', 'error': str(e)}

    # Save combined results
    with open(os.path.join(args.output_dir, 'result.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    log(f'\n{"="*60}')
    log('Summary')
    log(f'{"="*60}')
    for name, r in results.items():
        if 'dice' in r:
            log(f'  {name}: dice={r["dice"]}, method={r.get("method","")}, '
                f'ch={r.get("best_channel","")}, angle={r.get("angle_deg","")}°')
        else:
            log(f'  {name}: {r.get("status", "unknown")}')

    log_file.close()


if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    main()
