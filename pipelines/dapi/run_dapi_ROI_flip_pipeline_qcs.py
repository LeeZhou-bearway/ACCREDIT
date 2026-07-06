"""
run_dapi_ROI_flip_pipeline_qcs.py — QCS-guided DAPI→HE registration.

Precision variant of run_dapi_ROI_flip_pipeline.py. All candidate selection
and refinement decisions are guided by QCS (composite quality score) instead
of individual NMI/NCC metrics.

Differences from the original:
  Step 1  Candidates sorted by full QCS (TD+BD+HD+IR+NMI) instead of NMI metric
  Step 3  Angle fine-tune scored by QCS-fast (TD+BD+HD, no ORB) instead of NCC
  Step 4  Same as original (already QCS-guided)
  Step 5  BSpline accept/reject by QCS instead of NCC

Trade-off: slower than original (~2-5x for Step 1 due to full QCS per candidate),
but provides holistic quality guidance throughout.

Usage:
  python run_dapi_ROI_flip_pipeline_qcs.py \\
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
sys.path.insert(0, '/home/zhoule/workspace/Image_Reg')

from quality_metrics import (tissue_dice, boundary_dice, hausdorff_distance,
                              ncc, compute_quality_metrics_v2, composite_quality_score,
                              compute_nmi as compute_nmi_metric,
                              orb_inlier_ratio)

from run_dapi_ROI_flip_pipeline import (
    SCAN_SIZE, WORK_SIZE, PIXEL_RATIOS, DS_TARGET,
    ROTATIONS, FLIPS,
    to_hwc, norm8, make_tissue_mask, apply_flip, apply_transform,
    run_nmi,
)


# ── QCS scoring helpers ─────────────────────────────────────────────────────

def compute_qcs_full(he_mask, he_inv, warped_mask, warped_norm):
    """Full QCS: all 5 metrics (TD+BD+HD+IR+NMI). ~50-200ms per call."""
    return compute_quality_metrics_v2(he_mask, warped_mask, he_inv, warped_norm, eval_px=4000)


def compute_qcs_fast(he_mask, warped_mask):
    """Fast QCS: TD+BD+HD only, no ORB or NMI. ~5-10ms per call.

    Used for inner loops (Step 3 angle scan) where full QCS is too slow.
    Returns (qcs_fast, metrics_dict).
    """
    td = tissue_dice(he_mask, warped_mask)
    bd = boundary_dice(he_mask, warped_mask)
    hd_mean, hd_95 = hausdorff_distance(he_mask, warped_mask)

    s_td = max(0.0, min(1.0, td))
    s_bd = max(0.0, min(1.0, bd))
    s_hd = max(0.0, min(1.0, 1.0 - hd_mean / 500.0))

    # Renormalize weights (0.30+0.30+0.15 = 0.75)
    qcs_fast = (0.30 / 0.75) * s_td + (0.30 / 0.75) * s_bd + (0.15 / 0.75) * s_hd

    return round(qcs_fast, 4), {
        'tissue_dice': round(td, 4),
        'boundary_dice': round(bd, 4),
        'hausdorff_mean': round(hd_mean, 2),
        'hausdorff_95': round(hd_95, 2),
        'qcs_fast': round(qcs_fast, 4),
    }


# ── Core pipeline ────────────────────────────────────────────────────────────

def register_one_roi(he_path, fluo_path, out_dir, enhance=False, log=print):
    """QCS-guided ROI+flip DAPI→HE registration.

    Same structure as the original, but all scoring decisions use QCS.
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

    # ── Load DAPI ────────────────────────────────────────────────────────────
    with tifffile.TiffFile(fluo_path) as t:
        n_pages = len(t.pages)
        fl_page0 = t.pages[0].asarray()
    fl_h, fl_w = fl_page0.shape[:2]
    del fl_page0
    ds = max(1, max(he_w, he_h) // DS_TARGET)
    ds_fl_w, ds_fl_h = fl_w // ds, fl_h // ds
    log(f'  DAPI: {fl_w}x{fl_h}, {n_pages} channels (ds={ds} → {ds_fl_w}x{ds_fl_h})')

    # ── Step 1: Scan flip × rotation × pixel_ratio (QCS-guided) ─────────────
    log('\n' + '=' * 60)
    log('STEP 1: Orientation + scale scan  [QCS-guided]')
    log(f'  {len(FLIPS)} flips × {len(ROTATIONS)} rotations × {len(PIXEL_RATIOS)} ratios')
    log('=' * 60)

    best_candidates = []
    n_evaluated = 0

    with tifffile.TiffFile(fluo_path) as t:
        ch_scores = []
        for ch in range(min(n_pages, 20)):
            page = t.pages[ch].asarray().astype(np.float32)
            page_small = norm8(cv2.resize(page, (fl_w // 10, fl_h // 10)))
            del page
            mask_small = make_tissue_mask(page_small)
            coverage = np.sum(mask_small > 0) / mask_small.size
            ch_scores.append((coverage, ch))
        ch_scores.sort(reverse=True)
        top_channels = [ch for _, ch in ch_scores[:1]]
        log(f'  Best channel by coverage: {top_channels[0]}')

        for ch in top_channels:
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
                        fw_sc = int(tw * ds * pr * scan_scale)
                        fh_sc = int(th * ds * pr * scan_scale)

                        if fw_sc < 30 or fh_sc < 30:
                            continue
                        if fw_sc > scan_w * 2 or fh_sc > scan_h * 2:
                            continue

                        fl_prescaled = cv2.resize(transformed, (fw_sc, fh_sc))

                        # NMI for initial positioning (optimizer internal)
                        M_quick, metric = run_nmi(he_inv_scan, fl_prescaled, log=lambda x: None)
                        if M_quick is None:
                            continue

                        # Pre-filter by tissue dice
                        fl_mask = make_tissue_mask(fl_prescaled)
                        warped_mask = cv2.warpAffine(fl_mask, M_quick, (scan_w, scan_h))
                        td = tissue_dice(he_mask_scan, warped_mask)

                        if td > 0.3:
                            # Full QCS for candidate ranking
                            warped_fluo = cv2.warpAffine(fl_prescaled, M_quick, (scan_w, scan_h))
                            qcs_met = compute_qcs_full(he_mask_scan, he_inv_scan,
                                                       warped_mask, warped_fluo)
                            n_evaluated += 1
                            best_candidates.append({
                                'ch': ch, 'flip': flip_name, 'rot': rot_deg,
                                'pr': pr, 'metric': metric, 'td': td,
                                'qcs': qcs_met['qcs'],
                                'qcs_metrics': qcs_met,
                                'M_scan': M_quick,
                            })

    # Sort by QCS (highest = best) instead of NMI (most negative = best)
    best_candidates.sort(key=lambda x: -x['qcs'])
    log(f'\n  {len(best_candidates)} valid candidates ({n_evaluated} QCS evaluations)')

    if not best_candidates:
        log('  FAILED: no valid candidates')
        return {'status': 'failed'}

    for i, c in enumerate(best_candidates[:5]):
        log(f'  #{i+1}: ch{c["ch"]} {c["flip"]} {c["rot"]}° pr={c["pr"]:.1f} '
            f'QCS={c["qcs"]:.4f} (td={c["qcs_metrics"]["tissue_dice"]:.3f} '
            f'bd={c["qcs_metrics"]["boundary_dice"]:.3f} '
            f'nmi={c["qcs_metrics"].get("nmi", "?")}) NMI_metric={c["metric"]:.4f}')

    winner = best_candidates[0]
    log(f'\n  Winner: ch{winner["ch"]} {winner["flip"]} {winner["rot"]}° '
        f'pr={winner["pr"]:.1f} QCS={winner["qcs"]:.4f}')

    # ── Step 2: Fine NMI at higher resolution ────────────────────────────────
    log('\n' + '=' * 60)
    log('STEP 2: Fine NMI alignment')
    log('=' * 60)

    work_scale = WORK_SIZE / max(he_w, he_h)
    ws_w, ws_h = int(he_w * work_scale), int(he_h * work_scale)
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
    met = compute_qcs_full(he_mask_work, he_inv_work, warped_mask, warped_fl)
    log(f'  NMI: QCS={met["qcs"]:.4f} td={met["tissue_dice"]:.3f} '
        f'bd={met["boundary_dice"]:.3f} ncc={met["ncc"]:.3f} nmi={met.get("nmi","?")}')
    best_M = M_work
    best_met = met

    # ── Step 3: QCS-guided angle fine-tune (±90°) ───────────────────────────
    log('\n' + '=' * 60)
    log('STEP 3: QCS-guided angle fine-tune (±90°)')
    log('=' * 60)

    bcx, bcy = ws_w / 2.0, ws_h / 2.0
    best_qcs_fast_val = compute_qcs_fast(
        he_mask_work,
        cv2.warpAffine(fl_mask_prescaled, best_M, (ws_w, ws_h))
    )[0]
    best_finetune_M = best_M.copy()

    # Angle scan
    for da in range(-90, 91):
        R_corr = cv2.getRotationMatrix2D((bcx, bcy), da, 1.0)
        R_3 = np.vstack([R_corr, [0, 0, 1]])
        M_3 = np.vstack([best_M, [0, 0, 1]])
        M_rot = (R_3 @ M_3)[:2, :]

        wm = cv2.warpAffine(fl_mask_prescaled, M_rot, (ws_w, ws_h))
        qf, _ = compute_qcs_fast(he_mask_work, wm)
        if qf > best_qcs_fast_val:
            best_qcs_fast_val = qf
            best_finetune_M = M_rot.copy()

    # Translation scan
    M_angle = best_finetune_M.copy()
    for dx in range(-15, 16, 3):
        for dy in range(-15, 16, 3):
            Mt = M_angle.copy()
            Mt[0, 2] += dx
            Mt[1, 2] += dy
            wm = cv2.warpAffine(fl_mask_prescaled, Mt, (ws_w, ws_h))
            qf, _ = compute_qcs_fast(he_mask_work, wm)
            if qf > best_qcs_fast_val:
                best_qcs_fast_val = qf
                best_finetune_M = Mt.copy()

    best_M = best_finetune_M
    wm = cv2.warpAffine(fl_mask_prescaled, best_M, (ws_w, ws_h))
    wf = cv2.warpAffine(fl_prescaled, best_M, (ws_w, ws_h))
    best_met = compute_qcs_full(he_mask_work, he_inv_work, wm, wf)
    log(f'  After fine-tune: QCS={best_met["qcs"]:.4f} qcs_fast={best_qcs_fast_val:.4f} '
        f'ncc={best_met["ncc"]:.3f}')

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
        M_c[0, 2] /= ecc_sc
        M_c[1, 2] /= ecc_sc
        M_refined = (np.vstack([M_c, [0, 0, 1]]) @ np.vstack([best_M, [0, 0, 1]]))[:2, :]

        wm = cv2.warpAffine(fl_mask_prescaled, M_refined, (ws_w, ws_h))
        wf = cv2.warpAffine(fl_prescaled, M_refined, (ws_w, ws_h))
        met_ecc = compute_qcs_full(he_mask_work, he_inv_work, wm, wf)
        log(f'  ECC: QCS={met_ecc["qcs"]:.4f} ncc={met_ecc["ncc"]:.3f}')
        if met_ecc['qcs'] > best_met['qcs']:
            best_M = M_refined
            best_met = met_ecc
            log(f'  ECC improved!')
        else:
            log(f'  ECC did not improve')
    except Exception as e:
        log(f'  ECC failed: {e}')

    # ── Step 5: BSpline non-rigid refinement (QCS-guided accept) ────────────
    log('\n' + '=' * 60)
    log('STEP 5: BSpline non-rigid refinement  [QCS-guided accept]')
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

        met_bsp = compute_qcs_full(he_mask_work, he_inv_work, mask_bsp, warped_bsp)
        log(f'  BSpline: QCS={met_bsp["qcs"]:.4f} ncc={met_bsp["ncc"]:.3f}')
        if met_bsp['qcs'] > best_met['qcs']:
            best_met = met_bsp
            np.savez(os.path.join(out_dir, 'displacement_field.npz'), dx=dx_full, dy=dy_full)
            log(f'  BSpline improved!')
        else:
            log(f'  BSpline did not improve')
    except Exception as e:
        import traceback
        log(f'  BSpline failed: {e}\n{traceback.format_exc()}')

    # ── Build M_full ─────────────────────────────────────────────────────────
    flip_name = winner['flip']
    rot_deg = winner['rot']
    pr = winner['pr']

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
    elif rot_deg == 90:
        R_rot = np.array([[0, -1, fl_h - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    elif rot_deg == 180:
        R_rot = np.array([[-1, 0, fl_w - 1], [0, -1, fl_h - 1], [0, 0, 1]], dtype=np.float64)
    elif rot_deg == 270:
        R_rot = np.array([[0, 1, 0], [-1, 0, fl_w - 1], [0, 0, 1]], dtype=np.float64)

    S_down = np.array([[1.0/ds, 0, 0], [0, 1.0/ds, 0], [0, 0, 1]], dtype=np.float64)
    pre_scale = pr * work_scale * ds
    S_pre = np.array([[pre_scale, 0, 0], [0, pre_scale, 0], [0, 0, 1]], dtype=np.float64)
    S_back = np.array([[1/work_scale, 0, 0], [0, 1/work_scale, 0], [0, 0, 1]], dtype=np.float64)
    M_work_3x3 = np.vstack([best_M, [0, 0, 1]])

    M_full_3x3 = S_back @ M_work_3x3 @ S_pre @ S_down @ R_rot @ F
    M_full = M_full_3x3[:2, :]

    theta = np.degrees(np.arctan2(M_full[1, 0], M_full[0, 0]))
    scale_f = np.sqrt(M_full[0, 0]**2 + M_full[1, 0]**2)

    log(f'\n{"=" * 60}')
    log(f'FINAL: ch{winner["ch"]} {flip_name} {rot_deg}° pr={pr:.1f}  QCS={best_met["qcs"]:.4f}')
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
        'method': f'ROI_flip_QCS_{flip_name}_rot{rot_deg}_pr{pr}',
        'scoring': 'qcs',
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


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='QCS-guided DAPI→HE ROI+flip registration (precision variant)')
    parser.add_argument('--he', required=True, help='HE OME-TIFF path')
    parser.add_argument('--fluo', required=True, help='DAPI/morphology image path')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--enhance', action='store_true',
                        help='Apply gamma=0.5 enhancement')
    args = parser.parse_args()

    result = register_one_roi(args.he, args.fluo, args.output_dir,
                              enhance=args.enhance)
    if result:
        print(f'\nQCS={result.get("qcs", "N/A")}')
        print(json.dumps(result, indent=2, default=str)[:500])


if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    main()
