"""
Re-register problematic CODEX samples using nuclear channel (ch0) directly.
Uses multi-init NMI + NCC rotation scan + phase correlation + ECC.
"""
import json
import os
import sys
import numpy as np
import cv2
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CODEX_BASE = '/home/zhoule/workspace/Image_Reg/benchmark_data/CODEX/glioma_spatialomics'
WORK_SIZE = 4000

# Samples needing re-registration
# All samples except the already-good ones (MGH258_v6=0.74, ZH1019_INF=0.92)
SAMPLES = [
    'ZH1007_INF', 'ZH1007_NEC_B', 'ZH1019_T1_A',
    'ZH811_INF_v6', 'ZH811_T1_v6',
    'ZH916_INF', 'ZH916_T1_B',
    'MGH258_v6',  # try this too
]


def norm8(arr):
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def to_hwc(img):
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < min(img.shape[1:]):
        img = np.moveaxis(img, 0, -1)
    return img


def quick_dice(a, b):
    a, b = (a > 0), (b > 0)
    return 2.0 * np.sum(a & b) / (np.sum(a) + np.sum(b) + 1e-10)


def register_with_nuclear_ch(he_path, codex_path, out_dir, log=print):
    """Register CODEX to HE using nuclear channel (ch0) with multi-init NMI + rescue."""
    import SimpleITK as sitk
    import math

    os.makedirs(out_dir, exist_ok=True)

    # Load HE
    log('  Loading HE...')
    he = to_hwc(tifffile.imread(he_path))[..., :3]
    he_h, he_w = he.shape[:2]
    he_scale = min(1.0, WORK_SIZE / max(he_h, he_w))
    ws_h, ws_w = int(he_h * he_scale), int(he_w * he_scale)
    he_work = cv2.resize(he, (ws_w, ws_h))
    log(f'  HE: {he_w}x{he_h} → {ws_w}x{ws_h}')

    # HE inverted gray for NMI
    he_gray = cv2.cvtColor(he_work, cv2.COLOR_RGB2GRAY)
    he_inv = (255 - he_gray).astype(np.uint8)

    # HE tissue mask
    _, he_mask = cv2.threshold(he_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    he_mask = cv2.morphologyEx(he_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    he_mask = cv2.morphologyEx(he_mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # Load CODEX nuclear channel (ch0)
    log('  Loading CODEX nuclear ch0...')
    with tifffile.TiffFile(codex_path) as t:
        shape = t.series[0].shape
        nuc = t.pages[0].asarray().astype(np.float32)
    fl_h, fl_w = nuc.shape[:2]
    fl_scale = min(1.0, WORK_SIZE / max(fl_h, fl_w))
    fl_ws_h, fl_ws_w = int(fl_h * fl_scale), int(fl_w * fl_scale)
    nuc_work = norm8(cv2.resize(nuc, (fl_ws_w, fl_ws_h)))
    log(f'  CODEX nuclear ch0: {fl_w}x{fl_h} → {fl_ws_w}x{fl_ws_h}')
    del nuc

    # Nuclear channel tissue mask
    _, nuc_mask = cv2.threshold(nuc_work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nuc_mask = cv2.morphologyEx(nuc_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    nuc_mask = cv2.morphologyEx(nuc_mask, cv2.MORPH_DILATE, kernel, iterations=2)

    # ── Method 1: Multi-init NMI ──────────────────────────────────────────
    log('  Running multi-init NMI...')
    s = 0.25
    ref_s = cv2.resize(he_inv, None, fx=s, fy=s).astype(np.float64)
    mov_s = cv2.resize(nuc_work, None, fx=s, fy=s).astype(np.float64)

    fixed = sitk.GetImageFromArray(ref_s)
    moving = sitk.GetImageFromArray(mov_s)

    def _run_one(init_tf):
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
        reg.SetInitialTransform(init_tf, inPlace=False)
        final_tf = reg.Execute(fixed, moving)
        metric = reg.GetMetricValue()
        return final_tf, metric

    init_mom = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS)
    init_geo = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    candidates = [init_mom, init_geo]
    for angle_offset in [math.pi/2, -math.pi/2, math.pi]:
        t = sitk.Similarity2DTransform()
        t.SetParameters(init_mom.GetParameters())
        params = list(t.GetParameters())
        params[0] = params[0] + angle_offset
        t.SetParameters(params)
        candidates.append(t)

    best_M_work = None
    best_dice = 0.0
    best_method = ''

    for i, cand in enumerate(candidates):
        try:
            tf, metric = _run_one(cand)
            params = tf.GetParameters()
            center = tf.GetFixedParameters()
            scale_est = params[0]
            angle_rad = params[1]
            tx, ty = params[2] / s, params[3] / s
            cx, cy = center[0] / s, center[1] / s

            inv_scale = 1.0 / scale_est
            cos_inv = inv_scale * np.cos(-angle_rad)
            sin_inv = inv_scale * np.sin(-angle_rad)
            c_vec = np.array([cx, cy])
            t_vec = np.array([tx, ty])
            M = np.array([
                [cos_inv, -sin_inv,
                 -cos_inv * (c_vec[0] + t_vec[0]) + sin_inv * (c_vec[1] + t_vec[1]) + c_vec[0]],
                [sin_inv, cos_inv,
                 -sin_inv * (c_vec[0] + t_vec[0]) - cos_inv * (c_vec[1] + t_vec[1]) + c_vec[1]]
            ], dtype=np.float64)

            warped = cv2.warpAffine(nuc_mask, M, (ws_w, ws_h))
            dice = quick_dice(he_mask, warped)
            theta = np.degrees(angle_rad)
            log(f'  NMI init {i}: metric={metric:.4f}, angle={theta:.1f}°, '
                f'scale={scale_est:.4f}, dice={dice:.4f}')
            if dice > best_dice:
                best_dice = dice
                best_M_work = M
                best_method = f'NMI_nuclear_init{i}'
        except Exception as e:
            log(f'  NMI init {i}: failed ({e})')

    # ── Method 2: NCC rotation scan + phase correlation ───────────────────
    log('  Running NCC rotation scan on tissue masks...')
    scan_size = 256
    he_s = cv2.resize(he_mask, (scan_size, scan_size)).astype(np.float32)
    nuc_s = cv2.resize(nuc_mask, (scan_size, scan_size)).astype(np.float32)
    cx_s, cy_s = scan_size / 2.0, scan_size / 2.0

    coarse_scores = []
    for angle in range(0, 360, 5):
        if angle != 0:
            R = cv2.getRotationMatrix2D((cx_s, cy_s), angle, 1.0)
            m = cv2.warpAffine(nuc_s, R, (scan_size, scan_size))
        else:
            m = nuc_s
        he_n = (he_s - he_s.mean()) / (he_s.std() + 1e-8)
        m_n = (m - m.mean()) / (m.std() + 1e-8)
        ncc = float((he_n * m_n).mean())
        coarse_scores.append((angle, ncc))

    coarse_scores.sort(key=lambda x: -x[1])
    top3 = [a for a, _ in coarse_scores[:3]]
    log(f'  NCC top-3: {top3} scores={[round(s,4) for _,s in coarse_scores[:3]]}')

    # Fine scan around top candidates
    fine_best_angle, fine_best_score = 0, -1
    for cand_a in top3:
        for angle in range(cand_a - 10, cand_a + 11):
            if angle != 0:
                R = cv2.getRotationMatrix2D((cx_s, cy_s), angle, 1.0)
                m = cv2.warpAffine(nuc_s, R, (scan_size, scan_size))
            else:
                m = nuc_s
            he_n = (he_s - he_s.mean()) / (he_s.std() + 1e-8)
            m_n = (m - m.mean()) / (m.std() + 1e-8)
            ncc = float((he_n * m_n).mean())
            if ncc > fine_best_score:
                fine_best_score = ncc
                fine_best_angle = angle

    log(f'  Fine best: {fine_best_angle}° (NCC={fine_best_score:.4f})')

    # Phase correlation at best angles
    for angle_deg in sorted(set([fine_best_angle, 0, 90, 180, 270])):
        mh, mw = nuc_work.shape[:2]
        cx_m, cy_m = mw / 2.0, mh / 2.0
        R_mat = cv2.getRotationMatrix2D((cx_m, cy_m), angle_deg, 1.0)
        nuc_rot = cv2.warpAffine(nuc_work, R_mat, (mw, mh))
        nuc_he = cv2.resize(nuc_rot, (ws_w, ws_h))

        (dx, dy), response = cv2.phaseCorrelate(
            he_inv.astype(np.float32), nuc_he.astype(np.float32))

        scale_x = ws_w / mw
        scale_y = ws_h / mh
        R_3x3 = np.vstack([R_mat, [0, 0, 1]])
        S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float64)
        T = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]], dtype=np.float64)
        M_phase = (T @ S @ R_3x3)[:2, :]

        warped = cv2.warpAffine(nuc_mask, M_phase, (ws_w, ws_h))
        dice = quick_dice(he_mask, warped)
        log(f'  Phase {angle_deg}°: dx={dx:.1f}, dy={dy:.1f}, dice={dice:.4f}')

        if dice > best_dice:
            best_dice = dice
            best_M_work = M_phase
            best_method = f'phase_nuc_rot{angle_deg}'

    # ECC refinement on best
    if best_M_work is not None:
        log('  ECC refinement...')
        ecc_scale = 0.5
        he_ecc = cv2.resize(he_inv, None, fx=ecc_scale, fy=ecc_scale).astype(np.float32)
        nuc_ecc = cv2.resize(nuc_work, (ws_w, ws_h))
        nuc_ecc = cv2.resize(nuc_ecc, None, fx=ecc_scale, fy=ecc_scale).astype(np.float32)
        M_ecc = best_M_work.copy().astype(np.float32)
        M_ecc[0, 2] *= ecc_scale
        M_ecc[1, 2] *= ecc_scale
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
            _, M_out = cv2.findTransformECC(he_ecc, nuc_ecc, M_ecc, cv2.MOTION_AFFINE,
                                            criteria, inputMask=None, gaussFiltSize=5)
            M_refined = M_out.astype(np.float64)
            M_refined[0, 2] /= ecc_scale
            M_refined[1, 2] /= ecc_scale
            warped = cv2.warpAffine(nuc_mask, M_refined, (ws_w, ws_h))
            dice_ecc = quick_dice(he_mask, warped)
            log(f'  ECC: dice={dice_ecc:.4f}')
            if dice_ecc > best_dice:
                best_dice = dice_ecc
                best_M_work = M_refined
                best_method += '_ecc'
        except Exception as e:
            log(f'  ECC failed: {e}')

    if best_M_work is None:
        log('  FAILED - no valid registration')
        return None, 0.0

    # Convert M_work to M_full
    ratio = fl_scale / he_scale
    M_full = best_M_work.copy().astype(np.float64)
    M_full[0, 0] *= ratio
    M_full[0, 1] *= ratio
    M_full[1, 0] *= ratio
    M_full[1, 1] *= ratio
    M_full[0, 2] /= he_scale
    M_full[1, 2] /= he_scale

    theta = np.degrees(np.arctan2(M_full[1, 0], M_full[0, 0]))
    scale_f = np.sqrt(M_full[0, 0]**2 + M_full[1, 0]**2)
    log(f'  Best: dice={best_dice:.4f}, method={best_method}, '
        f'angle={theta:.1f}°, scale={scale_f:.4f}')

    return M_full, best_dice, best_method


def main():
    for name in SAMPLES:
        codex_path = os.path.join(CODEX_BASE, 'CODEX', f'{name}.ome.tif')
        he_path = os.path.join(CODEX_BASE, 'HE', 'HE', f'{name}_HE.ome.tif')
        out_dir = os.path.join(CODEX_BASE, 'output', name)
        result_path = os.path.join(out_dir, 'result.json')

        r = json.load(open(result_path))
        old_qg = r.get('quality_global', {}).get('dice', r.get('dice', 0))

        print(f'\n{"="*60}')
        print(f'{name}: current dice={old_qg:.4f}')
        print(f'{"="*60}')

        result = register_with_nuclear_ch(he_path, codex_path, out_dir, log=print)
        if result is None:
            print(f'  No improvement')
            continue

        M_full, new_dice, method = result

        if new_dice > old_qg:
            print(f'  IMPROVED: {old_qg:.4f} -> {new_dice:.4f}')
            # Update result.json
            r['M_full'] = M_full.tolist()
            r['quality_global'] = r.get('quality_global', {})
            r['quality_global']['dice'] = round(new_dice, 4)
            r['method'] = method
            r['best_channel'] = 0  # nuclear ch0
            with open(result_path, 'w') as f:
                json.dump(r, f, indent=2)
            np.save(os.path.join(out_dir, 'affine_M_rescue.npy'), M_full)

            # Update pipeline summary
            sp = os.path.join(out_dir, 'pipeline_summary.json')
            if os.path.exists(sp):
                s = json.load(open(sp))
                s['final_dice'] = round(new_dice, 4)
                s['final_stage'] = 'nuclear_rerescue'
                with open(sp, 'w') as f:
                    json.dump(s, f, indent=2)
        else:
            print(f'  No improvement ({old_qg:.4f} >= {new_dice:.4f})')


if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    main()
