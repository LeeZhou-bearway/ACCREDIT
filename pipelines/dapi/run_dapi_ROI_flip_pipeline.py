"""
run_dapi_ROI_flip_pipeline.py — DAPI→HE registration for partial FOV (ROI) cases.

For cases where DAPI covers only a partial region of the HE, with possible
flip/rotation. Combines flip detection with multi-scale NMI search.

Pipeline steps:
  Step 1  Orientation detection
          4 flips × 4 rotations × multiple pixel_ratios, evaluate QCS.

  Step 2  Fine NMI alignment
          SimpleITK Similarity2D on the best (flip, rotation, pixel_ratio).

  Step 3  NCC-guided angle fine-tune (±90°, 1° step)

  Step 4  ECC refinement

  Step 5  BSpline non-rigid refinement

Usage:
  python run_dapi_ROI_flip_pipeline.py \\
      --he data/he_image.ome.tif \\
      --fluo data/morphology.ome.tif \\
      --output_dir output/
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_accredit_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_accredit_root, 'quality'))

from quality_metrics import (tissue_dice, boundary_dice, hausdorff_distance,
                              ncc, compute_quality_metrics_v2)

SCAN_SIZE = 2000   # HE work resolution for scanning
WORK_SIZE = 2000   # work resolution for fine registration
PIXEL_RATIOS = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5]
DS_TARGET = 4000   # downsample DAPI so max dim ≤ this before flip/rot

ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

FLIPS = ['none', 'flipud', 'fliplr', 'flip_both']


# ── Helpers ──────────────────────────────────────────────────────────────────

def to_hwc(img):
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < min(img.shape[1:]):
        return np.moveaxis(img, 0, -1)
    return img


def norm8(arr):
    f = arr.astype(np.float32)
    pos = f[f > 0]
    if len(pos) < 100:
        return np.zeros_like(arr, dtype=np.uint8)
    p1, p99 = np.percentile(pos, [1, 99])
    return np.clip((f - p1) / (p99 - p1 + 1e-10) * 255, 0, 255).astype(np.uint8)


def make_tissue_mask(gray_uint8):
    _, mask = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return mask


def apply_flip(img, flip_name):
    if flip_name == 'none': return img
    elif flip_name == 'flipud': return np.flipud(img).copy()
    elif flip_name == 'fliplr': return np.fliplr(img).copy()
    elif flip_name == 'flip_both': return np.flipud(np.fliplr(img)).copy()
    return img


def apply_transform(img, flip_name, rot_deg):
    out = apply_flip(img, flip_name)
    rot_code = ROTATIONS[rot_deg]
    if rot_code is not None:
        out = cv2.rotate(out, rot_code)
    return out


def compute_qcs(he_mask, he_inv, warped_mask, warped_norm):
    return compute_quality_metrics_v2(he_mask, warped_mask, he_inv, warped_norm, eval_px=4000)


# ── NMI Registration ────────────────────────────────────────────────────────

def run_nmi(he_inv, fluo_norm, log=print):
    """SimpleITK Similarity2D + NMI. Both at same work resolution.
    fluo_norm may be smaller than he_inv (ROI case).
    Returns (M, metric, info) where M maps fluo_norm px → he_inv px."""
    import SimpleITK as sitk

    fixed = sitk.GetImageFromArray(he_inv.astype(np.float64))
    moving = sitk.GetImageFromArray(fluo_norm.astype(np.float64))

    best_M = None
    best_metric = 0

    for init_type, init_name in [
        (sitk.CenteredTransformInitializerFilter.MOMENTS, 'MOMENTS'),
        (sitk.CenteredTransformInitializerFilter.GEOMETRY, 'GEOMETRY'),
    ]:
        try:
            initial_tf = sitk.CenteredTransformInitializer(
                fixed, moving, sitk.Similarity2DTransform(), init_type)

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

            if metric < best_metric:
                best_metric = metric
                params = final_tf.GetParameters()
                center = final_tf.GetFixedParameters()
                scale_est = params[0]
                angle_rad = params[1]
                tx, ty = params[2], params[3]
                cx, cy = center[0], center[1]

                inv_scale = 1.0 / scale_est
                cos_inv = inv_scale * np.cos(-angle_rad)
                sin_inv = inv_scale * np.sin(-angle_rad)

                best_M = np.array([
                    [cos_inv, -sin_inv,
                     -cos_inv * (cx + tx) + sin_inv * (cy + ty) + cx],
                    [sin_inv, cos_inv,
                     -sin_inv * (cx + tx) - cos_inv * (cy + ty) + cy]
                ], dtype=np.float64)
        except Exception:
            pass

    return best_M, best_metric


# ── Core pipeline ────────────────────────────────────────────────────────────

def register_one_roi(he_path, fluo_path, out_dir, enhance=False, log=print):
    """ROI+flip DAPI→HE registration.

    Args:
        enhance: If True, apply gamma enhancement (γ=0.5) to DAPI before matching.
                 Use for weak DAPI signal samples (e.g. Mouse_Brain, Skin, humanLung).
    """
    import tifffile

    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    # ── Load HE ──────────────────────────────────────────────────────────────
    log('Loading images...')
    he_full = to_hwc(tifffile.imread(he_path))[..., :3]
    he_h, he_w = he_full.shape[:2]

    scan_scale = SCAN_SIZE / max(he_w, he_h)
    scan_w, scan_h = int(he_w * scan_scale), int(he_h * scan_scale)
    he_scan = cv2.resize(he_full, (scan_w, scan_h))
    he_inv_scan = (255 - cv2.cvtColor(he_scan, cv2.COLOR_RGB2GRAY)).astype(np.uint8)
    he_mask_scan = make_tissue_mask(he_inv_scan)
    del he_full

    log(f'  HE: {he_w}x{he_h} → {scan_w}x{scan_h}')

    # ── Load DAPI (all channels for selection) ───────────────────────────────
    with tifffile.TiffFile(fluo_path) as t:
        n_pages = len(t.pages)
        fl_page0 = t.pages[0].asarray()
    fl_h, fl_w = fl_page0.shape[:2]
    del fl_page0
    ds = max(1, max(he_w, he_h) // DS_TARGET)
    ds_fl_w, ds_fl_h = fl_w // ds, fl_h // ds
    log(f'  DAPI: {fl_w}x{fl_h}, {n_pages} channels (ds={ds} → {ds_fl_w}x{ds_fl_h})')

    # ── Step 1: Scan flip × rotation × pixel_ratio × channel ────────────────
    log('\n' + '=' * 60)
    log('STEP 1: Orientation + scale scan')
    log(f'  {len(FLIPS)} flips × {len(ROTATIONS)} rotations × {len(PIXEL_RATIOS)} ratios')
    log('=' * 60)

    best_candidates = []

    with tifffile.TiffFile(fluo_path) as t:
        # Select best channel by tissue coverage (use small thumbnail)
        ch_scores = []
        for ch in range(min(n_pages, 20)):
            page = t.pages[ch].asarray().astype(np.float32)
            page_small = norm8(cv2.resize(page, (fl_w // 10, fl_h // 10)))
            del page
            mask_small = make_tissue_mask(page_small)
            coverage = np.sum(mask_small > 0) / mask_small.size
            ch_scores.append((coverage, ch))
        ch_scores.sort(reverse=True)
        top_channels = [ch for _, ch in ch_scores[:1]]  # Use only best channel
        log(f'  Best channel by coverage: {top_channels[0]}')

        for ch in top_channels:
            # Downsample before flip/rot to avoid OOM on large DAPI
            page_full = t.pages[ch].asarray().astype(np.float32)
            page = norm8(cv2.resize(page_full, (ds_fl_w, ds_fl_h)))
            del page_full
            if enhance:
                page = np.clip(np.power(page / 255.0, 0.5) * 255, 0, 255).astype(np.uint8)
                log(f'  Applied gamma=0.5 enhancement to ch{ch}')

            for flip_name in FLIPS:
                for rot_deg in ROTATIONS:
                    transformed = apply_transform(page, flip_name, rot_deg)
                    th, tw = transformed.shape[:2]

                    for pr in PIXEL_RATIOS:
                        # Resize fluo to HE scan coordinate system
                        # page is at ds scale, so multiply by ds to get full-res equivalent
                        fw_sc = int(tw * ds * pr * scan_scale)
                        fh_sc = int(th * ds * pr * scan_scale)

                        if fw_sc < 30 or fh_sc < 30:
                            continue
                        if fw_sc > scan_w * 2 or fh_sc > scan_h * 2:
                            continue

                        fl_prescaled = cv2.resize(transformed, (fw_sc, fh_sc))

                        # Quick NMI to find position
                        M_quick, metric = run_nmi(he_inv_scan, fl_prescaled, log=lambda x: None)
                        if M_quick is None:
                            continue

                        # Evaluate with tissue dice
                        fl_mask = make_tissue_mask(fl_prescaled)
                        warped_mask = cv2.warpAffine(fl_mask, M_quick, (scan_w, scan_h))
                        td = tissue_dice(he_mask_scan, warped_mask)

                        if td > 0.3:
                            best_candidates.append({
                                'ch': ch, 'flip': flip_name, 'rot': rot_deg,
                                'pr': pr, 'metric': metric, 'td': td,
                                'M_scan': M_quick,
                            })

    best_candidates.sort(key=lambda x: x['metric'])  # most negative = best NMI
    log(f'\n  {len(best_candidates)} valid candidates')

    if not best_candidates:
        log('  FAILED: no valid candidates')
        return {'status': 'failed'}

    # Show top 5
    for i, c in enumerate(best_candidates[:5]):
        log(f'  #{i+1}: ch{c["ch"]} {c["flip"]} {c["rot"]}° pr={c["pr"]:.1f} '
            f'metric={c["metric"]:.4f} td={c["td"]:.3f}')

    winner = best_candidates[0]
    log(f'\n  Winner: ch{winner["ch"]} {winner["flip"]} {winner["rot"]}° '
        f'pr={winner["pr"]:.1f} metric={winner["metric"]:.4f}')

    # ── Step 2: Fine NMI at higher resolution ────────────────────────────────
    log('\n' + '=' * 60)
    log('STEP 2: Fine NMI alignment')
    log('=' * 60)

    work_scale = WORK_SIZE / max(he_w, he_h)
    ws_w, ws_h = int(he_w * work_scale), int(he_h * work_scale)
    # Reuse HE from Step 1 (SCAN_SIZE == WORK_SIZE, no need to reload)
    he_work = he_scan
    he_inv_work = he_inv_scan
    he_mask_work = he_mask_scan

    with tifffile.TiffFile(fluo_path) as t:
        fl_full = t.pages[winner['ch']].asarray().astype(np.float32)
    fl_raw = norm8(cv2.resize(fl_full, (ds_fl_w, ds_fl_h)))
    del fl_full
    if enhance:
        fl_raw = np.clip(np.power(fl_raw / 255.0, 0.5) * 255, 0, 255).astype(np.uint8)
    fl_transformed = apply_transform(fl_raw, winner['flip'], winner['rot'])
    del fl_raw
    th, tw = fl_transformed.shape[:2]

    # Prescale to HE work coordinates (multiply by ds for full-res equivalent)
    fw_work = int(tw * ds * winner['pr'] * work_scale)
    fh_work = int(th * ds * winner['pr'] * work_scale)
    fl_prescaled = cv2.resize(fl_transformed, (fw_work, fh_work))
    fl_mask_prescaled = make_tissue_mask(fl_prescaled)
    log(f'  HE work: {ws_w}x{ws_h}, DAPI prescaled: {fw_work}x{fh_work}')

    M_work, metric = run_nmi(he_inv_work, fl_prescaled, log=log)
    if M_work is None:
        log('  Fine NMI failed')
        return {'status': 'failed'}

    warped_mask = cv2.warpAffine(fl_mask_prescaled, M_work, (ws_w, ws_h))
    warped_fl = cv2.warpAffine(fl_prescaled, M_work, (ws_w, ws_h))
    met = compute_qcs(he_mask_work, he_inv_work, warped_mask, warped_fl)
    log(f'  NMI: metric={metric:.4f} QCS={met["qcs"]:.3f} td={met["tissue_dice"]:.3f} '
        f'bd={met["boundary_dice"]:.3f} ncc={met["ncc"]:.3f}')
    best_M = M_work
    best_met = met

    # ── Step 3: NCC angle fine-tune ──────────────────────────────────────────
    log('\n' + '=' * 60)
    log('STEP 3: NCC angle fine-tune (±90°)')
    log('=' * 60)

    bcx, bcy = ws_w / 2.0, ws_h / 2.0
    best_ncc_val = best_met.get('ncc', -1)
    best_finetune_M = best_M.copy()

    for da in range(-90, 91):
        R_corr = cv2.getRotationMatrix2D((bcx, bcy), da, 1.0)
        R_3 = np.vstack([R_corr, [0, 0, 1]])
        M_3 = np.vstack([best_M, [0, 0, 1]])
        M_rot = (R_3 @ M_3)[:2, :]

        wf = cv2.warpAffine(fl_prescaled, M_rot, (ws_w, ws_h))
        wm = cv2.warpAffine(fl_mask_prescaled, M_rot, (ws_w, ws_h))
        overlap = ((he_mask_work > 0) & (wm > 0)).astype(np.uint8) * 255
        nc = ncc(he_inv_work, wf, mask=overlap)
        if nc > best_ncc_val:
            best_ncc_val = nc
            best_finetune_M = M_rot.copy()

    # Translation
    M_angle = best_finetune_M.copy()
    for dx in range(-15, 16, 3):
        for dy in range(-15, 16, 3):
            Mt = M_angle.copy()
            Mt[0, 2] += dx; Mt[1, 2] += dy
            wf = cv2.warpAffine(fl_prescaled, Mt, (ws_w, ws_h))
            wm = cv2.warpAffine(fl_mask_prescaled, Mt, (ws_w, ws_h))
            overlap = ((he_mask_work > 0) & (wm > 0)).astype(np.uint8) * 255
            nc = ncc(he_inv_work, wf, mask=overlap)
            if nc > best_ncc_val:
                best_ncc_val = nc
                best_finetune_M = Mt.copy()

    best_M = best_finetune_M
    wm = cv2.warpAffine(fl_mask_prescaled, best_M, (ws_w, ws_h))
    wf = cv2.warpAffine(fl_prescaled, best_M, (ws_w, ws_h))
    best_met = compute_qcs(he_mask_work, he_inv_work, wm, wf)
    log(f'  After fine-tune: QCS={best_met["qcs"]:.3f} ncc={best_met["ncc"]:.3f}')

    # ── Step 4: ECC refinement ───────────────────────────────────────────────
    log('\n' + '=' * 60)
    log('STEP 4: ECC refinement')
    log('=' * 60)

    try:
        ecc_sc = 0.5
        ref = cv2.resize(he_inv_work, None, fx=ecc_sc, fy=ecc_sc).astype(np.float32)
        warped = cv2.warpAffine(fl_prescaled, best_M, (ws_w, ws_h))
        mov = cv2.resize(warped, None, fx=ecc_sc, fy=ecc_sc).astype(np.float32)

        M_ecc = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
        _, M_out = cv2.findTransformECC(ref, mov, M_ecc, cv2.MOTION_EUCLIDEAN,
                                        crit, inputMask=None, gaussFiltSize=5)
        M_c = M_out.astype(np.float64)
        M_c[0, 2] /= ecc_sc; M_c[1, 2] /= ecc_sc
        M_refined = (np.vstack([M_c, [0, 0, 1]]) @ np.vstack([best_M, [0, 0, 1]]))[:2, :]

        wm = cv2.warpAffine(fl_mask_prescaled, M_refined, (ws_w, ws_h))
        wf = cv2.warpAffine(fl_prescaled, M_refined, (ws_w, ws_h))
        met_ecc = compute_qcs(he_mask_work, he_inv_work, wm, wf)
        log(f'  ECC: QCS={met_ecc["qcs"]:.3f} ncc={met_ecc["ncc"]:.3f}')
        if met_ecc['qcs'] > best_met['qcs']:
            best_M = M_refined
            best_met = met_ecc
            log(f'  ECC improved!')
        else:
            log(f'  ECC did not improve')
    except Exception as e:
        log(f'  ECC failed: {e}')

    # ── Step 5: BSpline ──────────────────────────────────────────────────────
    log('\n' + '=' * 60)
    log('STEP 5: BSpline non-rigid refinement')
    log('=' * 60)

    warped_affine = cv2.warpAffine(fl_prescaled, best_M, (ws_w, ws_h))
    warped_mask_affine = cv2.warpAffine(fl_mask_prescaled, best_M, (ws_w, ws_h))
    warped_for_overlay = warped_affine

    try:
        import SimpleITK as sitk

        bsp_scale = 0.5
        ref_bsp = cv2.resize(he_inv_work, None, fx=bsp_scale, fy=bsp_scale).astype(np.float64)
        mov_bsp = cv2.resize(warped_affine, None, fx=bsp_scale, fy=bsp_scale).astype(np.float64)

        fixed_sitk = sitk.GetImageFromArray(ref_bsp)
        moving_sitk = sitk.GetImageFromArray(mov_bsp)

        mesh_size = 8
        bspline_tf = sitk.BSplineTransformInitializer(fixed_sitk, [mesh_size] * 2)

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.REGULAR)
        reg.SetMetricSamplingPercentage(0.2, seed=42)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetInitialTransform(bspline_tf, inPlace=False)
        reg.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5, numberOfIterations=200,
            maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=500,
            costFunctionConvergenceFactor=1e7)
        reg.SetShrinkFactorsPerLevel([2, 1])
        reg.SetSmoothingSigmasPerLevel([1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        bspline_result = reg.Execute(fixed_sitk, moving_sitk)
        log(f'  BSpline metric={reg.GetMetricValue():.4f}')

        disp_filter = sitk.TransformToDisplacementFieldFilter()
        disp_filter.SetReferenceImage(fixed_sitk)
        disp_field = disp_filter.Execute(bspline_result)
        disp_arr = sitk.GetArrayFromImage(disp_field)

        dx_full = cv2.resize(disp_arr[:, :, 0], (ws_w, ws_h)) / bsp_scale
        dy_full = cv2.resize(disp_arr[:, :, 1], (ws_w, ws_h)) / bsp_scale

        map_y, map_x = np.mgrid[0:ws_h, 0:ws_w].astype(np.float32)
        warped_bsp = cv2.remap(warped_affine, (map_x + dx_full).astype(np.float32),
                               (map_y + dy_full).astype(np.float32), cv2.INTER_LINEAR)
        mask_bsp = cv2.remap(warped_mask_affine, (map_x + dx_full).astype(np.float32),
                             (map_y + dy_full).astype(np.float32), cv2.INTER_LINEAR)

        met_bsp = compute_qcs(he_mask_work, he_inv_work, mask_bsp, warped_bsp)
        log(f'  BSpline: QCS={met_bsp["qcs"]:.3f} ncc={met_bsp["ncc"]:.3f}')
        if met_bsp['ncc'] > best_met.get('ncc', 0):
            best_met = met_bsp
            # Keep warped_for_overlay = warped_affine (no BSpline in overlay, consistent with M_full)
            np.savez(os.path.join(out_dir, 'displacement_field.npz'), dx=dx_full, dy=dy_full)
            log(f'  BSpline improved!')
        else:
            log(f'  BSpline did not improve')
    except Exception as e:
        import traceback
        log(f'  BSpline failed: {e}\n{traceback.format_exc()}')

    # ── Build M_full ─────────────────────────────────────────────────────────
    # Chain: fluo_full → F → R_rot → S_down(1/ds) → S_pre(pr*work_scale*ds) → M_work → S_back(1/work_scale) → he_full
    # Note: S_down @ S_pre = pr*work_scale (ds cancels), so M_full is resolution-independent
    flip_name = winner['flip']
    rot_deg = winner['rot']
    pr = winner['pr']

    # Flip matrix
    if flip_name == 'none':
        F = np.eye(3, dtype=np.float64)
    elif flip_name == 'flipud':
        F = np.array([[1, 0, 0], [0, -1, fl_h - 1], [0, 0, 1]], dtype=np.float64)
    elif flip_name == 'fliplr':
        F = np.array([[-1, 0, fl_w - 1], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    else:
        F = np.array([[-1, 0, fl_w - 1], [0, -1, fl_h - 1], [0, 0, 1]], dtype=np.float64)

    if rot_deg == 0:
        R_rot = np.eye(3, dtype=np.float64)
        out_w, out_h = fl_w, fl_h
    elif rot_deg == 90:
        # CW90: new_x = fl_h-1-y, new_y = x
        R_rot = np.array([[0, -1, fl_h - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        out_w, out_h = fl_h, fl_w
    elif rot_deg == 180:
        R_rot = np.array([[-1, 0, fl_w - 1], [0, -1, fl_h - 1], [0, 0, 1]], dtype=np.float64)
        out_w, out_h = fl_w, fl_h
    elif rot_deg == 270:
        # CCW90: new_x = y, new_y = fl_w-1-x
        R_rot = np.array([[0, 1, 0], [-1, 0, fl_w - 1], [0, 0, 1]], dtype=np.float64)
        out_w, out_h = fl_h, fl_w

    S_down = np.array([[1.0/ds, 0, 0], [0, 1.0/ds, 0], [0, 0, 1]], dtype=np.float64)
    pre_scale = pr * work_scale * ds  # maps ds-space → work-space
    S_pre = np.array([[pre_scale, 0, 0], [0, pre_scale, 0], [0, 0, 1]], dtype=np.float64)
    S_back = np.array([[1/work_scale, 0, 0], [0, 1/work_scale, 0], [0, 0, 1]], dtype=np.float64)
    M_work_3x3 = np.vstack([best_M, [0, 0, 1]])

    M_full_3x3 = S_back @ M_work_3x3 @ S_pre @ S_down @ R_rot @ F
    M_full = M_full_3x3[:2, :]

    theta = np.degrees(np.arctan2(M_full[1, 0], M_full[0, 0]))
    scale_f = np.sqrt(M_full[0, 0]**2 + M_full[1, 0]**2)

    log(f'\n{"=" * 60}')
    log(f'FINAL: ch{winner["ch"]} {flip_name} {rot_deg}° pr={pr:.1f}  QCS={best_met["qcs"]:.3f}')
    log(f'  angle={theta:.1f}°, scale={scale_f:.4f}')
    log(f'  metrics: {best_met}')
    log(f'  time={time.time() - t0:.0f}s')
    log('=' * 60)

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, 'affine_M_final.npy'), M_full)
    np.save(os.path.join(out_dir, 'best_M_work.npy'), best_M)

    result = {
        'he_path': he_path, 'fluo_path': fluo_path,
        'best_channel': winner['ch'], 'pixel_ratio': pr,
        'flip': flip_name, 'rotation': rot_deg,
        'method': f'ROI_flip_{flip_name}_rot{rot_deg}_pr{pr}',
        'M_full': M_full.tolist(),
        'best_M_work': best_M.tolist(),
        'work_scale': work_scale,
        'fl_w': fl_w, 'fl_h': fl_h,
        'he_w': he_w, 'he_h': he_h,
        'quality_metrics': best_met,
        'dice': best_met['tissue_dice'],
        'qcs': best_met['qcs'],
    }
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    # Overlay
    ov = he_work.copy()
    mask = warped_for_overlay > 30
    ov[mask] = (ov[mask] * 0.6 + np.array([0, 255, 0], dtype=np.uint8) * 0.4).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, 'overlay_dapi_roi.png'),
                cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    return result


# ── Contour → DAPI → HE pipeline ────────────────────────────────────────────

def transform_contours_via_dapi(dapi_result, parquet_path, he_path, out_dir,
                                 cell_parquet_path=None, log=print):
    """Transform parquet contours to HE space using DAPI's registration result.

    Chain: parquet_um → DAPI_px (×dapi_sf) → flip+rot → prescale → best_M → scale_up → HE_px

    Args:
        dapi_result: dict from register_one_roi (must contain best_M_work, flip, rotation, etc.)
        parquet_path: path to nucleus_boundaries.parquet
        he_path: path to HE OME-TIFF (only for dimensions and overlay)
        out_dir: output directory
        cell_parquet_path: optional cell_boundaries.parquet
    """
    import pandas as pd
    import tifffile
    import json as _json

    os.makedirs(out_dir, exist_ok=True)

    best_M = np.array(dapi_result['best_M_work'])
    work_scale = dapi_result['work_scale']
    pr = dapi_result['pixel_ratio']
    fl_w = dapi_result['fl_w']
    fl_h = dapi_result['fl_h']
    flip_name = dapi_result['flip']
    rot_deg = dapi_result['rotation']
    dapi_sf = 1.0 / 0.2125  # Xenium standard morphology pixel size

    with tifffile.TiffFile(he_path) as t:
        he_h, he_w = t.pages[0].shape[:2]

    log(f'  Contour→DAPI→HE: flip={flip_name}, rot={rot_deg}°, pr={pr}')
    log(f'  DAPI: {fl_w}x{fl_h}, HE: {he_w}x{he_h}')

    def _transform_parquet(pq_path, label_name='nucleus'):
        """Transform one parquet file and return (coords_he, grouped_df, n_contours)."""
        df = pd.read_parquet(pq_path)
        n = len(df['cell_id'].unique())
        log(f'  Loading {label_name}: {len(df)} vertices, {n} contours')

        if 'label_id' not in df.columns:
            df['label_id'] = df['cell_id']

        # Step 1: parquet physical → DAPI pixels
        x = df['vertex_x'].values * dapi_sf
        y = df['vertex_y'].values * dapi_sf

        # Step 2: apply flip
        if flip_name == 'flipud':
            y = (fl_h - 1) - y
        elif flip_name == 'fliplr':
            x = (fl_w - 1) - x
        elif flip_name == 'flip_both':
            x = (fl_w - 1) - x
            y = (fl_h - 1) - y

        # Step 3: apply rotation
        if rot_deg == 90:
            # CW90: new_x = fl_h-1-y, new_y = x
            x, y = (fl_h - 1) - y, x.copy()
        elif rot_deg == 180:
            x = (fl_w - 1) - x
            y = (fl_h - 1) - y
        elif rot_deg == 270:
            # CCW90: new_x = y, new_y = fl_w-1-x
            x, y = y.copy(), (fl_w - 1) - x

        # Step 4: prescale + best_M + scale_up
        ps = pr * work_scale
        x_ps, y_ps = x * ps, y * ps
        x_he = (best_M[0, 0] * x_ps + best_M[0, 1] * y_ps + best_M[0, 2]) / work_scale
        y_he = (best_M[1, 0] * x_ps + best_M[1, 1] * y_ps + best_M[1, 2]) / work_scale

        coords_he = np.column_stack([x_he, y_he])
        log(f'  {label_name} → HE: x=[{x_he.min():.0f},{x_he.max():.0f}], y=[{y_he.min():.0f},{y_he.max():.0f}]')

        return coords_he, df.groupby('label_id'), n

    # ── Nucleus boundaries ──
    coords_nuc, grouped_nuc, n_nuc = _transform_parquet(parquet_path, 'nucleus')

    # Save nucleus GeoJSON
    log('  Saving nucleus GeoJSON...')
    t0 = time.time()
    features = []
    for lid, grp in grouped_nuc:
        contour = coords_nuc[grp.index].tolist()
        if len(contour) >= 3:
            contour.append(contour[0])  # close polygon
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [contour]},
                "properties": {"label_id": int(lid)}
            })
    with open(os.path.join(out_dir, 'nucleus_boundaries.geojson'), 'w') as f:
        _json.dump({"type": "FeatureCollection", "features": features}, f)
    log(f'  {len(features)} nuclei, time={time.time()-t0:.1f}s')

    # ── Cell boundaries (optional) ──
    n_cell = 0
    if cell_parquet_path and os.path.exists(cell_parquet_path):
        coords_cell, grouped_cell, n_cell = _transform_parquet(cell_parquet_path, 'cell')

        log('  Saving cell GeoJSON...')
        t0 = time.time()
        features_cell = []
        for lid, grp in grouped_cell:
            contour = coords_cell[grp.index].tolist()
            if len(contour) >= 3:
                contour.append(contour[0])
                features_cell.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [contour]},
                    "properties": {"label_id": int(lid)}
                })
        with open(os.path.join(out_dir, 'cell_boundaries.geojson'), 'w') as f:
            _json.dump({"type": "FeatureCollection", "features": features_cell}, f)
        log(f'  {len(features_cell)} cells, time={time.time()-t0:.1f}s')

    # ── Overlay (downsampled) ──
    log('  Creating overlay...')
    ov_scale = 4000 / max(he_h, he_w)
    he_img = tifffile.imread(he_path)
    if he_img.ndim == 3 and he_img.shape[0] in (1, 3, 4) and he_img.shape[0] < min(he_img.shape[1:]):
        he_img = np.moveaxis(he_img, 0, -1)
    he_small = cv2.resize(he_img[:, :, :3], (int(he_w * ov_scale), int(he_h * ov_scale)))
    del he_img

    for lid, grp in grouped_nuc:
        c = (coords_nuc[grp.index] * ov_scale).astype(np.int32)
        if len(c) >= 3:
            cv2.drawContours(he_small, [c], -1, (0, 255, 255), 1)

    cv2.imwrite(os.path.join(out_dir, 'overlay_contours.png'),
                cv2.cvtColor(he_small, cv2.COLOR_RGB2BGR))
    log(f'  Overlay saved')

    # ── Result JSON ──
    contour_result = {
        'method': 'contour_via_dapi',
        'n_nuclei': n_nuc,
        'n_cells': n_cell,
        'dapi_flip': flip_name,
        'dapi_rotation': rot_deg,
        'dapi_pixel_ratio': pr,
        'dapi_qcs': dapi_result.get('qcs'),
    }
    with open(os.path.join(out_dir, 'result_contours.json'), 'w') as f:
        _json.dump(contour_result, f, indent=2)

    log(f'  Done: {n_nuc} nuclei, {n_cell} cells')
    return contour_result


# ── Batch ────────────────────────────────────────────────────────────────────

XENIUM_BASE = '/home/zhoule/workspace/Image_Reg/benchmark_data/DAPI/Xenium'


def run_single(he_path, fluo_path, out_dir, enhance=False, parquet=None,
               cell_parquet=None, log=print):
    """Run pipeline on a single sample with auto-detected parquet."""
    os.makedirs(out_dir, exist_ok=True)

    if enhance:
        log('Enhancement mode: gamma=0.5 applied to DAPI')

    result = register_one_roi(he_path, fluo_path, out_dir,
                               enhance=enhance, log=log)

    # Auto-detect parquet files
    fluo_dir = os.path.dirname(fluo_path)
    if not parquet:
        candidate = os.path.join(fluo_dir, 'nucleus_boundaries.parquet')
        if os.path.exists(candidate):
            parquet = candidate
    if not cell_parquet:
        candidate = os.path.join(fluo_dir, 'cell_boundaries.parquet')
        if os.path.exists(candidate):
            cell_parquet = candidate

    # Transform contours if parquet available
    if parquet and os.path.exists(parquet) and 'best_M_work' in result:
        log('\n' + '=' * 60)
        log('STEP 6: Contour → DAPI → HE')
        log('=' * 60)
        contour_out = os.path.join(out_dir, 'contours')
        transform_contours_via_dapi(
            result, parquet, he_path, contour_out,
            cell_parquet_path=cell_parquet, log=log)

    return result


def run_batch(log=print):
    """Run pipeline on all Xenium samples."""
    import gc

    samples = sorted([d for d in os.listdir(XENIUM_BASE)
                      if os.path.isdir(os.path.join(XENIUM_BASE, d))
                      and d.startswith('Xenium')])

    log(f'Found {len(samples)} samples\n')
    all_results = {}

    for name in samples:
        ds_dir = os.path.join(XENIUM_BASE, name)
        he_path = os.path.join(ds_dir, 'he_image.ome.tif')
        fluo_path = os.path.join(ds_dir, 'morphology.ome.tif')
        out_dir = os.path.join(ds_dir, 'output_dapi_roi_flip')

        if not os.path.exists(he_path) or not os.path.exists(fluo_path):
            log(f'\n{"="*60}\n{name}: SKIP (missing files)\n{"="*60}')
            continue

        # Skip if already has M_full
        existing = os.path.join(out_dir, 'result.json')
        if os.path.exists(existing):
            try:
                import json as _json
                r = _json.load(open(existing))
                if 'M_full' in r and 'best_M_work' in r and 'work_scale' in r:
                    log(f'\n{"="*60}\n{name}: SKIP (already has M_full, QCS={r["qcs"]:.4f})\n{"="*60}')
                    all_results[name] = r
                    continue
            except Exception:
                pass

        log(f'\n{"="*60}')
        log(f'{name}')
        log(f'{"="*60}')

        try:
            result = run_single(he_path, fluo_path, out_dir, log=log)
            all_results[name] = result
        except Exception as e:
            import traceback
            log(f'  ERROR: {e}\n{traceback.format_exc()}')

        gc.collect()

    # Summary
    log(f'\n\n{"="*60}')
    log('SUMMARY')
    log(f'{"="*60}')
    for name, r in sorted(all_results.items()):
        qcs = r.get('qcs', 'N/A')
        log(f'  {name}: QCS={qcs}')

    return all_results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DAPI→HE ROI+flip registration pipeline')
    parser.add_argument('--batch', action='store_true', default=True,
                        help='Run all Xenium samples (default)')
    parser.add_argument('--he', default=None, help='Single sample: HE path')
    parser.add_argument('--fluo', default=None, help='Single sample: DAPI path')
    parser.add_argument('--output_dir', default='output',
                        help='Single sample: output directory')
    parser.add_argument('--parquet', default=None,
                        help='nucleus_boundaries.parquet (auto-detect if adjacent to --fluo)')
    parser.add_argument('--cell_parquet', default=None,
                        help='cell_boundaries.parquet (auto-detect if adjacent to --fluo)')
    parser.add_argument('--enhance', action='store_true',
                        help='Apply gamma enhancement (γ=0.5) to DAPI')
    args = parser.parse_args()

    # Single sample mode if --he is provided
    if args.he:
        if not args.fluo:
            parser.error('--fluo is required when --he is specified')
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = open(os.path.join(args.output_dir, 'dapi_roi_flip.log'), 'w', buffering=1)

        def log(msg):
            print(msg, flush=True)
            log_file.write(msg + '\n')
            log_file.flush()

        run_single(args.he, args.fluo, args.output_dir,
                    enhance=args.enhance, parquet=args.parquet,
                    cell_parquet=args.cell_parquet, log=log)
        log_file.close()
    else:
        # Batch mode (default)
        log_path = os.path.join(XENIUM_BASE, 'batch_roi_flip.log')
        log_file = open(log_path, 'w', buffering=1)

        def log(msg):
            print(msg, flush=True)
            log_file.write(msg + '\n')
            log_file.flush()

        run_batch(log=log)
        log_file.close()
        print(f'\nBatch log: {log_path}')


if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    main()
