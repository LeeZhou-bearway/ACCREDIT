"""
run_st2he_test.py - ST spot plot to HE image registration via hole landmark matching.

Transform model: he_x = sx * st_x + tx, he_y = sy * st_y + ty (4 parameters).

Strategy:
1. Detect holes (gaps without spots/tissue) in both ST and HE images
2. Characterize holes by centroid, area, shape descriptors
3. RANSAC: match hole centroids to find consistent 4-param transform
4. Refine with multi-scale NMI optimization
5. Try all 4 orientations, select best by inlier count + NMI
"""
import argparse
import cv2
import json
import numpy as np
import os
from scipy import optimize


def extract_st_info(img_bgr):
    """Extract red dot mask and holes from ST spot plot.
    Returns: holes list, binary dot mask (cropped), crop bbox.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask1 = (hsv[:, :, 0] < 10) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 100)
    mask2 = (hsv[:, :, 0] > 170) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 100)
    red_mask = (mask1 | mask2).astype(np.uint8) * 255

    coords = np.where(red_mask > 0)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    margin = 5
    y_min = max(0, y_min - margin)
    y_max = min(red_mask.shape[0], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(red_mask.shape[1], x_max + margin)

    cropped = red_mask[y_min:y_max, x_min:x_max]

    # Dilate to connect nearby dots into solid region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    solid = cv2.dilate(cropped, kernel, iterations=2)

    # Find interior holes by flood-filling exterior
    inverted = cv2.bitwise_not(solid)
    mask_flood = np.zeros((inverted.shape[0] + 2, inverted.shape[1] + 2), np.uint8)
    cv2.floodFill(inverted, mask_flood, (0, 0), 0)
    interior = inverted  # only internal holes remain

    holes = _find_holes(interior, min_area=80)
    return holes, cropped, (y_min, y_max, x_min, x_max)


def extract_he_info(img_bgr):
    """Extract tissue mask and holes from HE image.
    Returns: holes list, tissue mask.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find holes: white regions inside tissue
    inverted = cv2.bitwise_not(mask)
    mask_flood = np.zeros((inverted.shape[0] + 2, inverted.shape[1] + 2), np.uint8)
    cv2.floodFill(inverted, mask_flood, (0, 0), 0)
    interior = inverted

    holes = _find_holes(interior, min_area=30)
    return holes, mask


def _find_holes(binary_mask, min_area=50):
    """Find contour-based holes from a binary mask."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        rect = cv2.minAreaRect(c)
        w_r, h_r = rect[1]
        if min(w_r, h_r) < 1:
            continue
        aspect = max(w_r, h_r) / min(w_r, h_r)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1)
        perim = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / max(perim ** 2, 1)

        holes.append({
            'cx': cx, 'cy': cy,
            'area': area,
            'aspect': aspect,
            'solidity': solidity,
            'circularity': circularity,
        })
    holes.sort(key=lambda h: h['area'], reverse=True)
    return holes


def _find_holes_from_dots(dot_mask):
    """Extract holes from an already-cropped binary dot mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    solid = cv2.dilate(dot_mask, kernel, iterations=2)
    inverted = cv2.bitwise_not(solid)
    mask_flood = np.zeros((inverted.shape[0] + 2, inverted.shape[1] + 2), np.uint8)
    cv2.floodFill(inverted, mask_flood, (0, 0), 0)
    return _find_holes(inverted, min_area=80)


def compute_nmi(img1, img2, mask=None):
    """Compute NMI between two images."""
    if mask is not None:
        a = img1[mask > 0].ravel()
        b = img2[mask > 0].ravel()
    else:
        a = img1.ravel()
        b = img2.ravel()
    if len(a) < 100:
        return 1.0
    bins = 64
    jh = np.histogram2d(a.astype(float), b.astype(float), bins=bins,
                        range=[[0, 255], [0, 255]])[0]
    jh = jh / jh.sum()
    jh_nz = jh[jh > 0]
    h_fm = -np.sum(jh_nz * np.log2(jh_nz))
    hf = jh.sum(axis=1); hf = hf[hf > 0]; hf = hf / hf.sum()
    h_f = -np.sum(hf * np.log2(hf))
    hm = jh.sum(axis=0); hm = hm[hm > 0]; hm = hm / hm.sum()
    h_m = -np.sum(hm * np.log2(hm))
    return (h_f + h_m) / h_fm if h_fm > 0 else 1.0


def solve_transform(st_pts, he_pts):
    """Solve for sx, sy, tx, ty from matched point pairs.
    he_x = sx * st_x + tx,  he_y = sy * st_y + ty
    """
    if len(st_pts) < 2:
        return None
    st_pts = np.array(st_pts)
    he_pts = np.array(he_pts)
    A_x = np.column_stack([st_pts[:, 0], np.ones(len(st_pts))])
    A_y = np.column_stack([st_pts[:, 1], np.ones(len(st_pts))])
    res_x = np.linalg.lstsq(A_x, he_pts[:, 0], rcond=None)
    res_y = np.linalg.lstsq(A_y, he_pts[:, 1], rcond=None)
    sx, tx = res_x[0]
    sy, ty = res_y[0]
    return sx, sy, tx, ty


def match_holes_ransac(st_holes, he_holes, log, n_top=15, n_iter=2000):
    """Match holes using RANSAC on the 4-param transform model.
    Pick 2 random (st, he) hole pairs, compute transform, count inliers.
    """
    st_top = st_holes[:n_top]
    he_top = he_holes[:min(n_top * 3, len(he_holes))]

    if len(st_top) < 2 or len(he_top) < 2:
        return None, 0, None

    st_pts = np.array([(h['cx'], h['cy']) for h in st_top])
    he_pts = np.array([(h['cx'], h['cy']) for h in he_top])
    st_areas = np.array([h['area'] for h in st_top])
    he_areas = np.array([h['area'] for h in he_top])

    best_inliers = 0
    best_params = None
    best_matches = None

    rng = np.random.RandomState(42)

    for _ in range(n_iter):
        si = rng.choice(len(st_top), 2, replace=False)
        hi = rng.choice(len(he_top), 2, replace=False)

        params = solve_transform(st_pts[si], he_pts[hi])
        if params is None:
            continue
        sx, sy, tx, ty = params

        if sx < 0.3 or sx > 3.0 or sy < 0.3 or sy > 3.0:
            continue
        if max(sx, sy) / min(sx, sy) > 2.0:
            continue

        st_transformed = st_pts * np.array([sx, sy]) + np.array([tx, ty])
        inlier_count = 0
        matches = []
        area_scale = sx * sy

        for j in range(len(st_top)):
            dists = np.sqrt(np.sum((he_pts - st_transformed[j]) ** 2, axis=1))
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]
            if min_dist < 25:
                expected_area = st_areas[j] * area_scale
                actual_area = he_areas[min_idx]
                area_ratio = actual_area / max(expected_area, 1)
                if 0.2 < area_ratio < 5.0:
                    inlier_count += 1
                    matches.append((j, min_idx))

        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_params = params
            best_matches = matches

    if best_params is not None and best_inliers >= 2:
        st_matched = np.array([st_pts[m[0]] for m in best_matches])
        he_matched = np.array([he_pts[m[1]] for m in best_matches])
        refined = solve_transform(st_matched, he_matched)
        if refined is not None:
            best_params = refined

    log(f'    RANSAC: {best_inliers} inliers, params={best_params}')
    return best_params, best_inliers, best_matches


def refine_nmi(he_gray, st_density, init_params, log, sigma=5):
    """Refine 4-param transform using NMI optimization."""
    sx0, sy0, tx0, ty0 = init_params
    h_he, w_he = he_gray.shape

    he_blur = cv2.GaussianBlur(he_gray.astype(np.float32), (0, 0), sigma)
    he_inv = 255.0 - he_blur
    he_inv = (he_inv / max(he_inv.max(), 1) * 255).astype(np.uint8)

    st_blur = cv2.GaussianBlur(st_density.astype(np.float32), (0, 0), sigma)
    st_blur = (st_blur / max(st_blur.max(), 1) * 255).astype(np.uint8)

    def objective(p):
        sx, sy, tx, ty = p
        if sx < 0.1 or sy < 0.1:
            return 0.0
        M = np.array([[sx, 0, tx], [0, sy, ty]], dtype=np.float64)
        warped = cv2.warpAffine(st_blur, M, (w_he, h_he),
                                flags=cv2.INTER_LINEAR, borderValue=0)
        valid = warped > 5
        if valid.sum() < 200:
            return 0.0
        return -compute_nmi(he_inv[valid], warped[valid])

    result = optimize.minimize(
        objective, [sx0, sy0, tx0, ty0],
        method='Nelder-Mead',
        options={'xatol': 0.5, 'fatol': 1e-5, 'maxiter': 2000, 'adaptive': True}
    )
    nmi = -result.fun
    log(f'    NMI refine (sigma={sigma}): {nmi:.4f} '
        f'params=({result.x[0]:.4f}, {result.x[1]:.4f}, {result.x[2]:.1f}, {result.x[3]:.1f})')
    return tuple(result.x), nmi


def try_orientation(orient_name, st_holes, st_density, st_dots,
                    he_holes, he_gray, he_mask, log):
    """Try one orientation: RANSAC hole matching + NMI refinement."""
    result = match_holes_ransac(st_holes, he_holes, log)
    if result[0] is None:
        log(f'    [{orient_name}] RANSAC failed')
        return None, -1.0, 0

    params, n_inliers, matches = result
    if n_inliers < 2:
        log(f'    [{orient_name}] Too few inliers: {n_inliers}')
        return None, -1.0, 0

    # Multi-scale NMI refinement
    for sigma in [12, 6, 3]:
        params, nmi = refine_nmi(he_gray, st_density, params, log, sigma=sigma)

    log(f'  [{orient_name}] final NMI={nmi:.4f} inliers={n_inliers} '
        f'params=({params[0]:.4f}, {params[1]:.4f}, {params[2]:.1f}, {params[3]:.1f})')
    return params, nmi, n_inliers


def register_st2he(st_path, he_path, out_dir, log):
    """Main registration function. Returns result dict."""
    os.makedirs(out_dir, exist_ok=True)

    st_bgr = cv2.imread(st_path)
    he_bgr = cv2.imread(he_path)
    if st_bgr is None:
        raise FileNotFoundError(f'Cannot read ST image: {st_path}')
    if he_bgr is None:
        raise FileNotFoundError(f'Cannot read HE image: {he_path}')
    log(f'  ST: {st_bgr.shape}, HE: {he_bgr.shape}')

    # ── Extract features ──────────────────────────────────────────────────
    st_holes, st_dots, st_bbox = extract_st_info(st_bgr)
    he_holes, he_mask = extract_he_info(he_bgr)
    he_gray = cv2.cvtColor(he_bgr, cv2.COLOR_BGR2GRAY)
    log(f'  ST holes: {len(st_holes)}, HE holes: {len(he_holes)}')

    st_density = cv2.GaussianBlur(st_dots.astype(np.float32), (0, 0), sigmaX=8)
    st_density = (st_density / max(st_density.max(), 1) * 255).astype(np.uint8)

    # Debug visualizations
    st_vis = cv2.cvtColor(st_dots, cv2.COLOR_GRAY2BGR)
    for i, h in enumerate(st_holes[:20]):
        cv2.circle(st_vis, (int(h['cx']), int(h['cy'])), 5, (0, 255, 0), -1)
        cv2.putText(st_vis, f"{i}:{int(h['area'])}", (int(h['cx']) + 5, int(h['cy'])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(out_dir, 'st_holes.png'), st_vis)

    he_vis = he_bgr.copy()
    for i, h in enumerate(he_holes[:30]):
        cv2.circle(he_vis, (int(h['cx']), int(h['cy'])), 5, (0, 255, 0), -1)
        cv2.putText(he_vis, f"{i}:{int(h['area'])}", (int(h['cx']) + 5, int(h['cy'])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(out_dir, 'he_holes.png'), he_vis)

    # ── Try all 4 orientations ────────────────────────────────────────────
    best_params, best_nmi, best_orient = None, -1.0, None
    best_inliers = 0
    best_dots = None

    orientations = [('original', st_holes, st_density, st_dots)]
    for flip_name, flip_code in [('hflip', 1), ('vflip', 0), ('hvflip', -1)]:
        flipped_dots = cv2.flip(st_dots, flip_code)
        flipped_density = cv2.flip(st_density, flip_code)
        flipped_holes = _find_holes_from_dots(flipped_dots)
        orientations.append((flip_name, flipped_holes, flipped_density, flipped_dots))

    for orient_name, orient_holes, orient_density, orient_dots in orientations:
        log(f'  --- Trying {orient_name} ---')
        params, nmi, n_inliers = try_orientation(
            orient_name, orient_holes, orient_density, orient_dots,
            he_holes, he_gray, he_mask, log)
        if params is not None and (nmi > best_nmi or
                                    (nmi > best_nmi - 0.01 and n_inliers > best_inliers)):
            best_params = params
            best_nmi = nmi
            best_orient = orient_name
            best_inliers = n_inliers
            best_dots = orient_dots

    if best_params is None:
        log('  Registration FAILED: no valid transform found')
        return {'st': st_path, 'he': he_path, 'error': 'no valid transform'}

    sx, sy, tx, ty = best_params
    log(f'  === Best: orient={best_orient} NMI={best_nmi:.4f} inliers={best_inliers} ===')
    log(f'      sx={sx:.5f} sy={sy:.5f} tx={tx:.1f} ty={ty:.1f}')

    # ── Visualization ─────────────────────────────────────────────────────
    M_fwd = np.array([[sx, 0, tx], [0, sy, ty]], dtype=np.float64)
    h_he, w_he = he_bgr.shape[:2]

    warped_dots = cv2.warpAffine(best_dots, M_fwd, (w_he, h_he),
                                  flags=cv2.INTER_NEAREST, borderValue=0)
    dot_mask = warped_dots > 128

    # Semi-transparent salmon overlay
    overlay = he_bgr.copy().astype(np.float32)
    salmon = np.array([120, 128, 250], dtype=np.float32)
    overlay[dot_mask] = 0.55 * salmon + 0.45 * overlay[dot_mask]
    cv2.imwrite(os.path.join(out_dir, 'overlay.png'), overlay.astype(np.uint8))

    # Red overlay
    overlay_red = he_bgr.copy()
    overlay_red[dot_mask, 0] = 100
    overlay_red[dot_mask, 1] = 100
    overlay_red[dot_mask, 2] = 255
    cv2.imwrite(os.path.join(out_dir, 'overlay_red.png'), overlay_red)

    # Save result
    result = {
        'st': st_path, 'he': he_path,
        'orient': best_orient,
        'flip': best_orient if best_orient != 'original' else None,
        'nmi': round(best_nmi, 4),
        'inliers': best_inliers,
        'sx': round(sx, 5), 'sy': round(sy, 5),
        'tx': round(tx, 2), 'ty': round(ty, 2),
        'st_bbox': [int(x) for x in st_bbox],
    }
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description='ST spot plot to HE registration')
    parser.add_argument('--st', required=True, help='ST spot plot image (red dots on white)')
    parser.add_argument('--he', required=True, help='HE tissue image')
    parser.add_argument('--output_dir', default='ST2HE/output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = open(os.path.join(args.output_dir, 'run.log'), 'w', buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    log('ST-to-HE Registration (Hole Landmark Matching)')
    log(f'ST: {args.st}')
    log(f'HE: {args.he}')
    log(f'Output: {args.output_dir}')

    result = register_st2he(args.st, args.he, args.output_dir, log)

    if 'error' in result:
        log(f'\nFAILED: {result["error"]}')
    else:
        log(f'\nSUCCESS: orient={result["orient"]} NMI={result["nmi"]} '
            f'inliers={result["inliers"]}')
        log(f'Transform: sx={result["sx"]} sy={result["sy"]} '
            f'tx={result["tx"]} ty={result["ty"]}')

    log_file.close()


if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    main()
