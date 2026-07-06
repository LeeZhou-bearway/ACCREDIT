"""
run_fluo2he_roi.py - Register fluorescence ROI images to full HE.

For cases where the fluorescence image covers only a PARTIAL region of the HE,
or has a large resolution difference. Uses NMI-based multi-scale search to
find the best (channel, pixel_ratio) combination, then fine-registers with
SimpleITK Similarity2D.

Usage:
    conda run -n pathml python run_fluo2he_roi.py \
        --he DAPI-CODEX-HE/VisiumHD/HE.tif \
        --fluo DAPI-CODEX-HE/VisiumHD/CODEX.tif \
        --output_dir DAPI-CODEX-HE/VisiumHD/output
"""
import argparse
import os
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
    a_pos = a > 0
    b_pos = b > 0
    inter = np.sum(a_pos & b_pos)
    total = np.sum(a_pos) + np.sum(b_pos)
    return 2.0 * inter / (total + 1e-10)


def he_to_inv_gray(he_img):
    gray = cv2.cvtColor(he_img, cv2.COLOR_RGB2GRAY)
    return (255 - gray)


# ── NMI Registration Core ─────────────────────────────────────────────────

def run_nmi_registration(he_inv, fluo_norm, log=print):
    """Run SimpleITK Similarity2D + NMI on pre-sized images.

    Both inputs should already be at the desired work resolution.
    Returns (M, metric, params_dict) where M maps fluo_px → he_px at input resolution.
    """
    fixed = sitk.GetImageFromArray(he_inv.astype(np.float64))
    moving = sitk.GetImageFromArray(fluo_norm.astype(np.float64))

    initial_tf = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.3)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=0.001, numberOfIterations=500)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(initial_tf, inPlace=False)

    final_tf = reg.Execute(fixed, moving)
    metric = reg.GetMetricValue()

    params = final_tf.GetParameters()   # [scale, angle, tx, ty]
    center = final_tf.GetFixedParameters()  # [cx, cy]

    scale_est = params[0]
    angle_rad = params[1]
    tx, ty = params[2], params[3]
    cx, cy = center[0], center[1]

    # Build warpAffine M: maps fluo → he at input resolution
    inv_scale = 1.0 / scale_est
    cos_inv = inv_scale * np.cos(-angle_rad)
    sin_inv = inv_scale * np.sin(-angle_rad)

    M = np.array([
        [cos_inv, -sin_inv,
         -cos_inv * (cx + tx) + sin_inv * (cy + ty) + cx],
        [sin_inv, cos_inv,
         -sin_inv * (cx + tx) - cos_inv * (cy + ty) + cy]
    ], dtype=np.float64)

    return M, metric, {
        'sitk_scale': scale_est,
        'angle_deg': np.degrees(angle_rad),
        'tx': tx, 'ty': ty,
        'cx': cx, 'cy': cy,
        'nmi_metric': metric,
    }


# ── Channel + Scale Selection ─────────────────────────────────────────────

def select_channel_and_scale(fluo_raw, he_inv_work, he_full_w, he_full_h, log=print):
    """Select best (channel, pixel_ratio) by NMI metric scan.

    Pre-scales each fluo channel to various pixel_ratios, runs quick NMI,
    picks the combination with the best (most negative) metric.

    Args:
        fluo_raw: raw fluo (2D or CxHxW)
        he_inv_work: HE inverted grayscale at work resolution (~2000px)
        he_full_w, he_full_h: full HE pixel dimensions
    Returns:
        (best_ch, best_pr, best_metric, channel_img, all_scores)
    """
    he_wh, he_ww = he_inv_work.shape[:2]

    if fluo_raw.ndim == 2:
        channels = {0: fluo_raw}
        fluo_h, fluo_w = fluo_raw.shape
    else:
        channels = {i: fluo_raw[i] for i in range(fluo_raw.shape[0])}
        fluo_h, fluo_w = fluo_raw.shape[1], fluo_raw.shape[2]

    # Work scale for HE
    work_scale_he = he_ww / he_full_w  # how he_inv_work relates to full HE

    # Candidate pixel ratios (HE_full_px / fluo_full_px)
    # At work scale, fluo should be resized to: fluo_dim * pixel_ratio * work_scale_he
    ratios = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    # HE binary mask for QCS evaluation
    _, he_bin = cv2.threshold(he_inv_work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        from quality_metrics import compute_quality_metrics_v2
        use_qcs = True
    except ImportError:
        use_qcs = False

    log(f'  Scanning {len(channels)} channels × {len(ratios)} ratios (select by {"QCS" if use_qcs else "NMI"})...')
    best_qcs = -1
    best_metric = 0
    best_result = None
    all_scores = {}

    for chi, ch_raw in channels.items():
        ch_norm = norm8(ch_raw)
        ch_best_qcs = -1

        for pr in ratios:
            # Resize fluo to HE work coordinate system
            fw = int(fluo_w * pr * work_scale_he)
            fh = int(fluo_h * pr * work_scale_he)

            # Skip if fluo would be way larger than HE work image
            if fw > he_ww * 2 or fh > he_wh * 2:
                continue
            if fw < 30 or fh < 30:
                continue

            ch_resized = cv2.resize(ch_norm, (fw, fh))

            try:
                M_cand, metric, _ = run_nmi_registration(he_inv_work, ch_resized, log=lambda x: None)
            except Exception:
                continue

            # Evaluate with QCS instead of NMI metric
            if use_qcs and M_cand is not None:
                _, fl_bin = cv2.threshold(ch_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                warped_mask = cv2.warpAffine(fl_bin, M_cand, (he_ww, he_wh))
                warped_norm = cv2.warpAffine(ch_resized, M_cand, (he_ww, he_wh))
                try:
                    qm = compute_quality_metrics_v2(he_bin, warped_mask, he_inv_work, warped_norm, eval_px=4000)
                    qcs_val = qm['qcs']
                except Exception:
                    qcs_val = -1
            else:
                qcs_val = -metric if metric < 0 else 0  # fallback to NMI

            if qcs_val > best_qcs:
                best_qcs = qcs_val
                best_metric = metric
                best_result = (chi, pr, metric)

            if qcs_val > ch_best_qcs:
                ch_best_qcs = qcs_val

        all_scores[chi] = round(ch_best_qcs, 4)

    if best_result is None:
        log('  No valid registration found')
        return None, None, None, None, all_scores

    best_ch, best_pr, _ = best_result
    log(f'  Channel scores (best QCS): {all_scores}')
    log(f'  Best: ch{best_ch}, pixel_ratio={best_pr:.1f}, QCS={best_qcs:.4f}, NMI={best_metric:.4f}')

    return best_ch, best_pr, best_metric, channels[best_ch], all_scores


# ── ECC Refinement ─────────────────────────────────────────────────────────

def ecc_refine(he_ref, fluo_norm, M_init, log=print, scale=0.25):
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

def register_one_roi(he_img, fluo_path, out_dir, force_channel=None, log=print):
    """Register a fluorescence ROI to full HE.

    Strategy:
    1. Scan (channel, pixel_ratio) with NMI to find best combination
    2. Fine NMI registration at higher resolution
    3. Compose transform: pre-scale + NMI → full coordinates
    """
    os.makedirs(out_dir, exist_ok=True)
    fluo_name = os.path.splitext(os.path.basename(fluo_path))[0]

    log(f'\n{"="*60}')
    log(f'Registering {fluo_name} (ROI) → HE')
    log(f'  HE: {he_img.shape}')

    t0 = time.time()
    fluo_raw = tifffile.imread(fluo_path)
    log(f'  {fluo_name}: {fluo_raw.shape} loaded in {time.time()-t0:.1f}s')

    he_h, he_w = he_img.shape[0], he_img.shape[1]
    if fluo_raw.ndim == 3 and fluo_raw.shape[0] < fluo_raw.shape[1]:
        fluo_h, fluo_w = fluo_raw.shape[1], fluo_raw.shape[2]
    else:
        fluo_h, fluo_w = fluo_raw.shape[0], fluo_raw.shape[1]

    area_ratio = (fluo_w * fluo_h) / (he_w * he_h)
    log(f'  HE: {he_w}x{he_h}, Fluo: {fluo_w}x{fluo_h}, area_ratio={area_ratio:.3f}')

    # Phase 1: Prepare HE at scan resolution (~2000px for fast NMI scan)
    scan_long = 2000
    scan_scale_he = scan_long / max(he_w, he_h)
    scan_w, scan_h = int(he_w * scan_scale_he), int(he_h * scan_scale_he)
    he_scan = cv2.resize(he_img, (scan_w, scan_h))
    he_inv_scan = he_to_inv_gray(he_scan)

    # Phase 2: Channel + scale selection via NMI scan
    if force_channel is not None:
        log(f'  Phase 1: Forced channel={force_channel}, scanning pixel_ratios only...')
        t0 = time.time()
        # Extract forced channel, scan only pixel_ratios
        if fluo_raw.ndim == 2:
            forced_ch_img = fluo_raw
        else:
            forced_ch_img = fluo_raw[force_channel]
        # Run scale scan on forced channel only
        forced_raw = forced_ch_img[np.newaxis, ...] if fluo_raw.ndim == 3 else forced_ch_img
        best_ch_f, best_pr, best_metric, _, all_scores = \
            select_channel_and_scale(forced_raw, he_inv_scan, he_w, he_h, log=log)
        best_ch = force_channel
        fluo_ch = forced_ch_img
        del fluo_raw
        log(f'  Scan time: {time.time()-t0:.1f}s')
    else:
        log('  Phase 1: Channel + scale selection (NMI scan)...')
        t0 = time.time()
        best_ch, best_pr, best_metric, fluo_ch, all_scores = \
            select_channel_and_scale(fluo_raw, he_inv_scan, he_w, he_h, log=log)
        del fluo_raw  # free memory
        log(f'  Scan time: {time.time()-t0:.1f}s')

    if best_ch is None:
        log(f'  FAILED: could not find valid registration for {fluo_name}')
        return {'name': fluo_name, 'status': 'failed'}

    # Phase 3: Fine registration at higher resolution
    log(f'  Phase 2: Fine NMI at higher resolution...')
    work_long = 2000
    work_scale_he = work_long / max(he_w, he_h)
    ws_w, ws_h = int(he_w * work_scale_he), int(he_h * work_scale_he)
    he_work = cv2.resize(he_img, (ws_w, ws_h))
    he_inv_work = he_to_inv_gray(he_work)

    # Pre-scale fluo to HE coordinate system
    fluo_norm = norm8(fluo_ch)
    fw = int(fluo_w * best_pr * work_scale_he)
    fh = int(fluo_h * best_pr * work_scale_he)
    fluo_prescaled = cv2.resize(fluo_norm, (fw, fh))
    log(f'  HE work: {ws_w}x{ws_h}, Fluo prescaled: {fw}x{fh}')

    t0 = time.time()
    M_work, metric, nmi_info = run_nmi_registration(he_inv_work, fluo_prescaled, log=log)
    dt = time.time() - t0

    theta = nmi_info['angle_deg']
    scale_est = nmi_info['sitk_scale']
    log(f'  NMI: scale={scale_est:.4f}, angle={theta:.3f}°, '
        f'metric={metric:.4f}, time={dt:.1f}s')

    # Dice at work resolution
    _, he_bin = cv2.threshold(he_inv_work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fl_bin = cv2.threshold(fluo_prescaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    warped = cv2.warpAffine(fl_bin, M_work, (ws_w, ws_h))
    dice = quick_dice(he_bin, warped)
    log(f'  Dice: {dice:.4f}')

    # Phase 4: ECC refinement
    log('  ECC refinement...')
    M_ecc = ecc_refine(he_inv_work, fluo_prescaled, M_work, log=log, scale=0.5)
    warped_ecc = cv2.warpAffine(fl_bin, M_ecc, (ws_w, ws_h))
    dice_ecc = quick_dice(he_bin, warped_ecc)
    if dice_ecc > dice:
        log(f'  ECC improved dice: {dice:.4f} → {dice_ecc:.4f}')
        M_work = M_ecc
        dice = dice_ecc
    else:
        log(f'  ECC no improvement ({dice_ecc:.4f} vs {dice:.4f})')

    # Phase 5: Compose final transform (fluo_full_px → he_full_px)
    # Chain: fluo_full → (pre-scale by best_pr * work_scale_he) → fluo_prescaled_work
    #        → (M_work) → he_work → (scale back by 1/work_scale_he) → he_full
    pre_scale = best_pr * work_scale_he  # fluo_full → prescaled_work

    S_pre = np.array([
        [pre_scale, 0, 0],
        [0, pre_scale, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    M_work_3x3 = np.vstack([M_work, [0, 0, 1]]).astype(np.float64)

    S_back = np.array([
        [1.0 / work_scale_he, 0, 0],
        [0, 1.0 / work_scale_he, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    M_full_3x3 = S_back @ M_work_3x3 @ S_pre
    M_full = M_full_3x3[:2, :]

    log(f'  M_full composed: fluo_full_px → he_full_px')

    # Phase 6: Save outputs
    log('  Saving outputs...')

    # Overlay: fluo on HE at work resolution
    warped_vis = cv2.warpAffine(fluo_prescaled, M_work, (ws_w, ws_h))
    overlay = he_work.copy()
    mask = warped_vis > 30
    overlay[mask] = (overlay[mask] * 0.6 +
                     np.array([0, 255, 0], dtype=np.uint8) * 0.4).astype(np.uint8)

    overlay_path = os.path.join(out_dir, f'overlay_{fluo_name}.png')
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    log(f'  Saved {overlay_path}')

    # Save transform
    np.save(os.path.join(out_dir, f'affine_{fluo_name}.npy'), M_full)

    # QCS evaluation
    try:
        from quality_metrics import compute_quality_metrics_v2
        he_mask = cv2.threshold(he_inv_work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        warped_mask = cv2.warpAffine(fl_bin, M_work, (ws_w, ws_h))
        warped_norm = cv2.warpAffine(fluo_prescaled, M_work, (ws_w, ws_h))
        qcs_metrics = compute_quality_metrics_v2(he_mask, warped_mask, he_inv_work, warped_norm, eval_px=4000)
        log(f'  QCS={qcs_metrics["qcs"]:.3f}  td={qcs_metrics["tissue_dice"]:.3f}  '
            f'bd={qcs_metrics["boundary_dice"]:.3f}  ncc={qcs_metrics["ncc"]:.3f}  '
            f'hd={qcs_metrics["hausdorff_mean"]:.1f}')
    except Exception as e:
        log(f'  QCS evaluation failed: {e}')
        qcs_metrics = {}

    result = {
        'name': fluo_name,
        'fluo_path': fluo_path,
        'fluo_size': [fluo_w, fluo_h],
        'he_size': [he_w, he_h],
        'area_ratio': round(area_ratio, 4),
        'best_channel': int(best_ch),
        'pixel_ratio': round(best_pr, 4),
        'channel_scores': {str(k): v for k, v in all_scores.items()},
        'dice': round(dice, 4),
        'quality_metrics': qcs_metrics,
        'qcs': qcs_metrics.get('qcs', None),
        'method': 'NMI_prescale_Similarity2D',
        'angle_deg': round(theta, 4),
        'sitk_scale': round(scale_est, 6),
        'nmi_metric': round(metric, 4),
        'M_full': M_full.tolist(),
    }
    log(f'  Final dice: {dice:.4f}  QCS: {qcs_metrics.get("qcs", "N/A")}')
    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Register fluorescence ROI images to full HE')
    parser.add_argument('--he', required=True, help='Path to HE image (TIFF)')
    parser.add_argument('--fluo', nargs='+', required=True,
                        help='Path(s) to fluorescence image(s) (TIFF)')
    parser.add_argument('--output_dir', default='output',
                        help='Output directory')
    parser.add_argument('--enhance', type=int, default=0, choices=[0, 1, 2],
                        help='Enhancement level: 0=none, 1=CODEX gamma(0.5), '
                             '2=CODEX gamma(0.5) + HE CLAHE')
    parser.add_argument('--channel', type=int, default=None,
                        help='Force specific channel (e.g. 0 for nuclear). '
                             'Default: auto-select by NMI scan')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = open(os.path.join(args.output_dir, 'run.log'), 'w', buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    log('Fluorescence ROI → HE Registration')
    log(f'HE: {args.he}')
    log(f'Fluorescence inputs: {args.fluo}')
    log(f'Output: {args.output_dir}')
    if args.enhance:
        log(f'Enhancement level: {args.enhance}')

    log('Loading HE image...')
    t0 = time.time()
    he_img = tifffile.imread(args.he)
    log(f'  HE: {he_img.shape} loaded in {time.time()-t0:.1f}s')

    # HE enhancement (level 2)
    if args.enhance >= 2:
        log('  Applying HE CLAHE enhancement...')
        if he_img.ndim == 3 and he_img.shape[-1] == 3:
            lab = cv2.cvtColor(he_img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            he_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        log('  HE CLAHE done')

    # CODEX gamma enhancement: preprocess fluo files
    enhanced_fluo_paths = args.fluo
    if args.enhance >= 1:
        import tempfile
        enhanced_fluo_paths = []
        for fp in args.fluo:
            log(f'  Applying gamma=0.5 to {os.path.basename(fp)}...')
            raw = tifffile.imread(fp)
            # Apply gamma per channel/page
            if raw.ndim == 2:
                raw_f = raw.astype(np.float32)
                p99 = np.percentile(raw_f[raw_f > 0], 99) if np.any(raw_f > 0) else 1
                raw_gamma = np.clip(np.power(raw_f / p99, 0.5) * p99, 0, p99).astype(raw.dtype)
            else:
                # Multi-page: apply gamma to each channel
                raw_gamma = raw.copy()
                for ch in range(raw.shape[0] if raw.ndim == 3 else 1):
                    plane = raw[ch] if raw.ndim == 3 else raw
                    pf = plane.astype(np.float32)
                    p99 = np.percentile(pf[pf > 0], 99) if np.any(pf > 0) else 1
                    gamma_plane = np.clip(np.power(pf / p99, 0.5) * p99, 0, p99).astype(plane.dtype)
                    if raw.ndim == 3:
                        raw_gamma[ch] = gamma_plane
                    else:
                        raw_gamma = gamma_plane
            # Save enhanced to temp file
            enh_path = os.path.join(args.output_dir, f'enhanced_{os.path.basename(fp)}')
            tifffile.imwrite(enh_path, raw_gamma)
            enhanced_fluo_paths.append(enh_path)
            del raw, raw_gamma
            log(f'  Saved enhanced: {enh_path}')

    results = {}
    for fluo_path in enhanced_fluo_paths:
        try:
            result = register_one_roi(he_img, fluo_path, args.output_dir,
                                     force_channel=args.channel, log=log)
            results[result['name']] = result
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log(f'ERROR processing {fluo_path}: {e}\n{tb}')
            name = os.path.splitext(os.path.basename(fluo_path))[0]
            results[name] = {'status': 'error', 'error': str(e)}

    with open(os.path.join(args.output_dir, 'result.json'), 'w') as f:
        json.dump(results, f, indent=2)

    log(f'\n{"="*60}')
    log('Summary')
    log(f'{"="*60}')
    for name, r in results.items():
        if 'dice' in r:
            log(f'  {name}: dice={r["dice"]}, ch={r.get("best_channel","")}, '
                f'pixel_ratio={r.get("pixel_ratio","")}, '
                f'angle={r.get("angle_deg","")}°')
        else:
            log(f'  {name}: {r.get("status", "unknown")}')

    log_file.close()


if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    main()
