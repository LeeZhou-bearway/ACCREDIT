"""DAPI-boundary partial-match pipeline (SENIOR PLUS).

Flow:
  STEP 1: Initial M via ACCREDIT register_one_roi
    - Tries ROI+flip (no enhance) → QCS
    - Tries ROI+flip (gamma=0.5 enhance) → QCS
    - Picks best → M_full (includes BSpline if it improved)

  STEP 2: Boundary chain
    Apply M_init to parquet vertices → run boundary pipeline

  STEP 3: DAPI warp
    scipy.affine_transform DAPI MIP → HE_small space using M_init

  STEP 4: DAPI rescue
    Binary ORB + multi-scale phase correlation refine

  STEP 5: Via contour fallback (if step 4 fails)

Usage:
  python run_senior_plus_pipeline.py \\
      --he data/he_image.ome.tif \\
      --dapi data/dapi_mip.tif \\
      --parquet data/nucleus_boundaries.parquet \\
      --output_dir output/
"""
import os, sys, json, argparse, time
import numpy as np
import cv2
import pandas as pd
import tifffile
from PIL import Image
from scipy.ndimage import affine_transform
import xml.etree.ElementTree as ET
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dapi'))


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_he_mpp(he_path):
    with tifffile.TiffFile(he_path) as t:
        if t.ome_metadata:
            root = ET.fromstring(t.ome_metadata)
            ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
            return float(root.find('.//ome:Pixels', ns).get('PhysicalSizeX', '0.2125'))
    return 0.2125


def make_binary(gray, reverse=False):
    g = (255 - gray).astype(np.uint8) if reverse else gray
    blur = cv2.GaussianBlur(g, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)


def orb_align(image1, image2, nfeatures=100000, scale_factor=0.2, log=print):
    if scale_factor != 1:
        im1 = cv2.resize(image1, (int(image1.shape[1] * scale_factor), int(image1.shape[0] * scale_factor)))
        im2 = cv2.resize(image2, (int(image2.shape[1] * scale_factor), int(image2.shape[0] * scale_factor)))
    else:
        im1, im2 = image1, image2

    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    log(f'  kp1={len(kp1) if kp1 else 0}, kp2={len(kp2) if kp2 else 0}')
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    log(f'  matches={len(matches)}')
    if len(matches) < 4:
        return None, 0, 0

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(dst, src)
    if M is None:
        return None, 0, 0
    M[:, 2] = M[:, 2] / scale_factor
    inliers = int(mask.sum()) if mask is not None else 0
    return M, inliers, len(matches)


def quick_dice(a, b):
    am = a > 0
    bm = b > 0
    return 2 * (am & bm).sum() / (am.sum() + bm.sum() + 1e-10)


def refine_translation(he_bin, mov_bin, mov_gray, M, scales=(0.1, 0.25, 0.5), log=print):
    best_M = M.copy()
    H, W = he_bin.shape
    best_warped_bin = cv2.warpAffine(mov_bin, best_M, (W, H))
    best_dice = quick_dice(he_bin, best_warped_bin)
    log(f'  init dice={best_dice:.4f}')
    for s in scales:
        ref_s = cv2.resize(he_bin, None, fx=s, fy=s)
        warped = cv2.warpAffine(mov_gray, best_M, (W, H))
        mov_s = cv2.resize(warped, (ref_s.shape[1], ref_s.shape[0]))
        shift, _ = cv2.phaseCorrelate(ref_s.astype(np.float64), mov_s.astype(np.float64))
        dx, dy = shift[0] / s, shift[1] / s
        if abs(dx) > 500 or abs(dy) > 500:
            log(f'  s={s}: shift ({dx:.1f}, {dy:.1f}) too large, skip')
            continue
        trial = best_M.copy()
        trial[0, 2] += dx
        trial[1, 2] += dy
        trial_bin = cv2.warpAffine(mov_bin, trial, (W, H))
        trial_dice = quick_dice(he_bin, trial_bin)
        if trial_dice > best_dice:
            log(f'  s={s}: shift=({dx:.1f}, {dy:.1f}) dice {best_dice:.4f} -> {trial_dice:.4f}')
            best_M = trial
            best_dice = trial_dice
        else:
            log(f'  s={s}: dice {trial_dice:.4f} no improvement')
    return best_M, best_dice


# ── STEP 1 ───────────────────────────────────────────────────────────────────

def step1_unified_agent(he_path, dapi_path, out_dir, enhance=False, log=print):
    from run_dapi_ROI_flip_pipeline import register_one_roi

    label = 'ROI flip + enhance + BSpline' if enhance else 'ROI flip + BSpline'
    log('=' * 60)
    log(f'STEP 1: Initial M via ACCREDIT ({label})')
    log('=' * 60)

    roi_out = os.path.join(out_dir, 'step1_init', 'roi_flip')
    result = register_one_roi(he_path, dapi_path, roi_out, enhance=enhance, log=log)
    if result is None:
        log('  FAILED')
        return None

    qcs = result.get('qcs', 0)
    log(f'\n  {label} QCS: {qcs:.4f}')

    with open(os.path.join(out_dir, 'step1_init', 'result.json'), 'w') as f:
        json.dump(result, f, indent=2, default=float)
    return result


# ── STEP 2 ───────────────────────────────────────────────────────────────────

def step2_boundary_chain(he_path, parquet_path, M_init, out_dir, log=print):
    log('\n' + '=' * 60)
    log('STEP 2: Boundary chain (warp parquet -> boundary pipeline)')
    log('=' * 60)
    step_dir = os.path.join(out_dir, 'step2_boundary')
    os.makedirs(step_dir, exist_ok=True)

    mpp_he = get_he_mpp(he_path)
    mpp_dapi = 0.2125
    log(f'  HE mpp = {mpp_he:.4f}, DAPI mpp = {mpp_dapi}')

    log('  Loading parquet...')
    df = pd.read_parquet(parquet_path)
    log(f'  {len(df)} vertices, {df["cell_id"].nunique()} cells')

    xs = df['vertex_x'].values / mpp_dapi
    ys = df['vertex_y'].values / mpp_dapi
    pts = np.vstack([xs, ys, np.ones_like(xs)])
    he_pts = M_init @ pts
    df = df.copy()
    df['vertex_x'] = he_pts[0] * mpp_he
    df['vertex_y'] = he_pts[1] * mpp_he

    warped_parq = os.path.join(step_dir, 'warped_nuclei.parquet')
    df.to_parquet(warped_parq, index=False)
    log(f'  Wrote {warped_parq}')

    boundary_dir = os.path.join(os.path.dirname(__file__), '..', 'boundary')
    sys.path.insert(0, boundary_dir)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from run_boundary_pipeline import run_boundary_pipeline
    log('  Running boundary pipeline...')
    boundary_out = os.path.join(step_dir, 'boundary')
    os.makedirs(boundary_out, exist_ok=True)
    result = run_boundary_pipeline(he_path, warped_parq, boundary_out, log=log)
    log(f'  -> boundary done, dice={result.get("dice","?")}')
    return result, boundary_out


# ── STEP 3 ───────────────────────────────────────────────────────────────────

def step3_warp_dapi(he_path, dapi_path, M_init, out_dir, he_level=2, log=print):
    log('\n' + '=' * 60)
    log(f'STEP 3: Warp DAPI to HE_small (level {he_level})')
    log('=' * 60)

    with tifffile.TiffFile(he_path) as t:
        levels = list(t.series[0].levels)
        full_h, full_w = levels[0].shape[:2]
        small_arr = levels[he_level].asarray()
    if small_arr.ndim == 3 and small_arr.shape[0] in (3, 4):
        small_arr = np.moveaxis(small_arr, 0, -1)
    sh_h, sh_w = small_arr.shape[:2]
    s_small = sh_w / full_w
    log(f'  HE full: {full_w}x{full_h}, small: {sh_w}x{sh_h}, s_small={s_small:.4f}')

    log('  Loading DAPI MIP...')
    dapi = tifffile.imread(dapi_path)
    if dapi.ndim == 3:
        dapi = dapi.max(axis=0)
    mn, mx = float(dapi.min()), float(dapi.max())
    dapi = ((dapi.astype(np.float32) - mn) / (mx - mn + 1e-10) * 255).astype(np.uint8)
    dapi = np.clip(np.power(dapi / 255.0, 0.8) * 255, 0, 255).astype(np.uint8)
    log(f'  DAPI: {dapi.shape} (normalized + gamma 0.8)')

    S_small = np.diag([s_small, s_small, 1.0])
    H = (S_small @ np.vstack([M_init, [0, 0, 1]]))[:2]

    H3 = np.vstack([H, [0, 0, 1]])
    H_inv = np.linalg.inv(H3)
    mat_yx = np.array([[H_inv[1, 1], H_inv[1, 0]],
                       [H_inv[0, 1], H_inv[0, 0]]])
    off_yx = np.array([H_inv[1, 2], H_inv[0, 2]])

    log('  scipy.affine_transform...')
    warped = affine_transform(dapi, mat_yx, offset=off_yx,
                              output_shape=(sh_h, sh_w),
                              order=1, mode='constant', cval=0)
    warped = warped.astype(np.uint8)
    log(f'  warped DAPI: {warped.shape}, cov={(warped>30).sum()/warped.size:.1%}')

    out_dapi = os.path.join(out_dir, 'step3_warped_dapi.tif')
    out_he = os.path.join(out_dir, 'step3_he_small.tif')
    tifffile.imwrite(out_dapi, warped, bigtiff=True, tile=(512, 512), compression='zstd')
    tifffile.imwrite(out_he, small_arr, bigtiff=True, tile=(512, 512), compression='zstd')
    log(f'  saved {out_dapi}')
    return out_dapi, out_he, s_small


# ── STEP 4 ───────────────────────────────────────────────────────────────────

def step4_dapi_rescue(warped_dapi_path, he_small_path, out_dir, log=print):
    log('\n' + '=' * 60)
    log('STEP 4: DAPI rescue (binary ORB + phase correlation)')
    log('=' * 60)
    step_dir = os.path.join(out_dir, 'step4_dapi_rescue')
    os.makedirs(step_dir, exist_ok=True)

    dapi = tifffile.imread(warped_dapi_path)
    he = tifffile.imread(he_small_path)
    he_g = cv2.cvtColor(he, cv2.COLOR_RGB2GRAY) if he.ndim == 3 else he
    he_inv = (255 - he_g).astype(np.uint8)
    H, W = he_g.shape

    he_bin = make_binary(he_g, reverse=True)
    dapi_bin = make_binary(dapi, reverse=False)
    log(f'  HE binary: {he_bin.sum()/he_bin.size:.1%}, DAPI binary: {dapi_bin.sum()/dapi_bin.size:.1%}')

    log('  ORB align (binary masks, scale 0.2)...')
    M, inliers, n_matches = orb_align(he_bin, dapi_bin, log=log)
    if M is None:
        log('  ORB FAILED')
        return None
    theta = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
    log(f'  ORB: {inliers}/{n_matches} inliers, angle={theta:.2f} deg, scale={scale:.4f}')

    log('  Phase correlation refine...')
    M_ref, final_dice = refine_translation(he_bin, dapi_bin, dapi, M, log=log)

    np.save(os.path.join(step_dir, 'M_dapi_refined.npy'), M_ref)
    dapi_final = cv2.warpAffine(dapi, M_ref, (W, H))

    def ncc(a, b):
        am = a.astype(np.float32) - a.mean()
        bm = b.astype(np.float32) - b.mean()
        return float((am * bm).sum() / (np.sqrt((am**2).sum()) * np.sqrt((bm**2).sum()) + 1e-10))

    ncc_before = ncc(he_inv, dapi)
    ncc_after = ncc(he_inv, dapi_final)
    log(f'  NCC before={ncc_before:.4f}  after={ncc_after:.4f}')

    ov = he.copy() if he.ndim == 3 else cv2.cvtColor(he_g, cv2.COLOR_GRAY2BGR)
    m = dapi_final > 30
    ov[m] = (ov[m].astype(np.float32) * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
    cv2.putText(ov, f'rescue ncc={ncc_after:.3f}', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
    Image.fromarray(ov).save(os.path.join(step_dir, 'overlay.png'),
                              optimize=False, compress_level=1)
    return {'M_refined': M_ref.tolist(), 'dice': float(final_dice),
            'ncc_before': ncc_before, 'ncc_after': ncc_after,
            'inliers': inliers, 'matches': n_matches}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='DAPI-boundary partial-match pipeline (SENIOR PLUS)')
    p.add_argument('--he', required=True, help='HE OME-TIFF path')
    p.add_argument('--dapi', required=True, help='DAPI MIP TIFF path')
    p.add_argument('--parquet', required=True, help='nucleus_boundaries.parquet path')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--he_level', type=int, default=2, help='HE pyramid level for warped DAPI step')
    p.add_argument('--enhance', action='store_true', help='Apply gamma=0.5 enhancement in step 1')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = open(os.path.join(args.output_dir, 'pipeline.log'), 'w', buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_file.write(str(msg) + '\n')
        log_file.flush()

    t0 = time.time()
    log(f'\n{"="*60}\nSENIOR PLUS PIPELINE\n{"="*60}')
    log(f'HE:      {args.he}')
    log(f'DAPI:    {args.dapi}')
    log(f'parquet: {args.parquet}')
    log(f'output:  {args.output_dir}\n')

    # STEP 1
    init_res = step1_unified_agent(args.he, args.dapi, args.output_dir, enhance=args.enhance, log=log)
    if init_res is None:
        log('PIPELINE ABORTED at STEP 1')
        return 1
    M_init = np.array(init_res['M_full'], dtype=np.float64)

    # STEP 2
    boundary_res, boundary_dir = step2_boundary_chain(args.he, args.parquet, M_init, args.output_dir, log=log)

    # STEP 3
    warped_dapi, he_small, _ = step3_warp_dapi(args.he, args.dapi, M_init, args.output_dir,
                                                he_level=args.he_level, log=log)

    # STEP 4
    rescue_res = step4_dapi_rescue(warped_dapi, he_small, args.output_dir, log=log)

    # STEP 5: Via contour fallback if step 4 failed
    step5_res = None
    step4_ok = (rescue_res is not None and
                rescue_res.get('ncc_after', 0) > rescue_res.get('ncc_before', 0) and
                rescue_res.get('dice', 0) > 0.5)
    if not step4_ok and boundary_dir is not None:
        log('\n' + '=' * 60)
        log('STEP 5: DAPI via contour fallback (step4 failed)')
        log('=' * 60)
        from run_dapi_via_contour import register_dapi_via_contour
        step5_dir = os.path.join(args.output_dir, 'step5_via_contour')
        step5_res = register_dapi_via_contour(
            he_path=args.he,
            fluo_path=args.dapi,
            boundary_dir=boundary_dir,
            output_dir=step5_dir,
            log=log,
        )
        if step5_res:
            log(f'  Step 5 via contour done.')
        else:
            log(f'  Step 5 via contour FAILED.')
    elif step4_ok:
        log(f'\nStep 4 rescue OK (dice={rescue_res["dice"]:.4f}). Skip step 5.')

    summary = {
        'pipeline': 'senior_plus',
        'time_s': round(time.time() - t0, 1),
        'step1_method': init_res.get('method'),
        'step1_qcs': init_res.get('qcs'),
        'step1_M_full': M_init.tolist(),
        'step2_boundary_dice': boundary_res.get('dice') if boundary_res else None,
        'step4_rescue': rescue_res,
        'step4_ok': step4_ok,
        'step5_via_contour': step5_res,
    }
    with open(os.path.join(args.output_dir, 'pipeline_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    log(f'\n{"="*60}\nDONE in {summary["time_s"]}s\n{"="*60}')
    log_file.close()
    return 0


if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    sys.exit(main())
