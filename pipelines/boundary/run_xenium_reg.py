"""
run_xenium_reg.py - Xenium nucleus boundary to HE image registration.

Pipeline:
  1. Load nucleus_boundaries.parquet → pixel coords
  2. Render filled contour binary mask
  3. HE → enhanced grayscale binary (contrast + weighted RGB + invert + Otsu)
  4. Global ORB alignment (1/5 scale, 100K features)
  4.5. Multi-scale phase correlation refinement of global translation
  5. Per-block template-matching refinement (translation-only, smoothed displacement field)
  6. Quality assessment + output

Usage:
    conda run -n pathml python run_xenium_reg.py \
        --he Xenium_offical/Breast1/Human_Breast_Biomarkers_S1_Top_he_image.ome.tif \
        --parquet Xenium_offical/Breast1/nucleus_boundaries.parquet \
        --output_dir Xenium_offical/Breast1/output
"""
import argparse
import cv2
import json
import numpy as np
import os
import sys
import time
import traceback
from pathlib import Path
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Image conversion ──────────────────────────────────────────────────────

def myrgb2gray(image, reverse=False, binary=False):
    """Enhanced grayscale: contrast boost + weighted RGB + optional invert + Otsu.

    Adaptive: if tissue fraction after Otsu is too low (<5%), retries with lower
    alpha values to handle pale-stained tissue (e.g. Kidney).
    """
    for alpha in [1.4, 1.0, 0.8]:
        img = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        # Weighted RGB (emphasize blue channel for H&E)
        gray = (0.3 * img[:, :, 2] + 0.1 * img[:, :, 1] + 0.6 * img[:, :, 0]).astype(np.uint8)
        if reverse:
            gray = 255 - gray
        if binary:
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            tissue_frac = np.count_nonzero(gray) / gray.size
            if tissue_frac >= 0.05:
                return gray
            # Too little tissue detected, try lower alpha
        else:
            return gray
    # Fallback: return last attempt even if low tissue
    return gray


# ── Feature alignment ─────────────────────────────────────────────────────

def orb_align(image1, image2, nfeatures=100000, scale_factor=0.2):
    """ORB feature matching: align image2 → image1. Returns 2×3 affine matrix."""
    orb = cv2.ORB_create(nfeatures=nfeatures)
    if scale_factor == 1:
        im1, im2 = image1, image2
    else:
        im1 = cv2.resize(image1, (int(image1.shape[1] * scale_factor), int(image1.shape[0] * scale_factor)))
        im2 = cv2.resize(image2, (int(image2.shape[1] * scale_factor), int(image2.shape[0] * scale_factor)))

    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise RuntimeError("Not enough ORB features")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    if len(matches) < 4:
        raise RuntimeError(f"Only {len(matches)} ORB matches")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts)
    if M is None:
        raise RuntimeError("estimateAffinePartial2D failed")

    M[:, 2] = M[:, 2] / scale_factor
    inliers = int(mask.sum()) if mask is not None else 0
    return M, inliers, len(matches)


def akaze_align(image1, image2, scale_factor=1, **kwargs):
    """AKAZE feature matching: align image2 → image1. Returns 2×3 affine matrix."""
    if scale_factor != 1:
        im1 = cv2.resize(image1, (int(image1.shape[1] * scale_factor), int(image1.shape[0] * scale_factor)))
        im2 = cv2.resize(image2, (int(image2.shape[1] * scale_factor), int(image2.shape[0] * scale_factor)))
    else:
        im1, im2 = image1, image2

    akaze = cv2.AKAZE_create(**kwargs)
    kp1, des1 = akaze.detectAndCompute(im1, None)
    kp2, des2 = akaze.detectAndCompute(im2, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise RuntimeError("Not enough AKAZE features")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    if len(matches) < 4:
        raise RuntimeError(f"Only {len(matches)} AKAZE matches")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts)
    if M is None:
        raise RuntimeError("estimateAffinePartial2D failed")

    M[:, 2] = M[:, 2] / scale_factor
    inliers = int(mask.sum()) if mask is not None else 0
    return M, inliers, len(matches)


# ── Global translation refinement ─────────────────────────────────────────

def refine_global_translation(he_binary, filled_image, M, warp_fn,
                               scales=(0.1, 0.25, 0.5), log=print):
    """Multi-scale phase correlation to refine the translation in M.
    After ORB gets rotation/scale right, this fixes residual translation error."""
    best_M = M.copy()
    best_dice = _quick_dice(he_binary, warp_fn(filled_image, best_M, he_binary.shape[:2]))
    log(f'    Refine init dice={best_dice:.4f}')

    for s in scales:
        ref_s = cv2.resize(he_binary, None, fx=s, fy=s)
        warped = warp_fn(filled_image, best_M, he_binary.shape[:2])
        mov_s = cv2.resize(warped, (ref_s.shape[1], ref_s.shape[0]))

        shift, response = cv2.phaseCorrelate(
            ref_s.astype(np.float64), mov_s.astype(np.float64))

        dx, dy = shift[0] / s, shift[1] / s
        if abs(dx) > 500 or abs(dy) > 500:
            log(f'    scale={s}: shift too large ({dx:.1f}, {dy:.1f}), skip')
            continue

        trial_M = best_M.copy()
        trial_M[0, 2] += dx
        trial_M[1, 2] += dy
        trial_warped = warp_fn(filled_image, trial_M, he_binary.shape[:2])
        trial_dice = _quick_dice(he_binary, trial_warped)

        if trial_dice > best_dice:
            log(f'    scale={s}: shift=({dx:.1f}, {dy:.1f}), dice {best_dice:.4f}→{trial_dice:.4f} ✓')
            best_M = trial_M
            best_dice = trial_dice
        else:
            log(f'    scale={s}: shift=({dx:.1f}, {dy:.1f}), dice {trial_dice:.4f} no improvement')

    # Also try ECC refinement at medium scale for sub-pixel accuracy
    try:
        s = 0.25
        ref_s = cv2.resize(he_binary, None, fx=s, fy=s).astype(np.float32)
        warped = warp_fn(filled_image, best_M, he_binary.shape[:2])
        mov_s = cv2.resize(warped, (ref_s.shape[1], ref_s.shape[0])).astype(np.float32)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
        _, warp_matrix = cv2.findTransformECC(
            ref_s, mov_s, warp_matrix, cv2.MOTION_TRANSLATION, criteria)

        dx_ecc = float(warp_matrix[0, 2]) / s
        dy_ecc = float(warp_matrix[1, 2]) / s
        if abs(dx_ecc) < 500 and abs(dy_ecc) < 500:
            trial_M = best_M.copy()
            trial_M[0, 2] += dx_ecc
            trial_M[1, 2] += dy_ecc
            trial_warped = warp_fn(filled_image, trial_M, he_binary.shape[:2])
            trial_dice = _quick_dice(he_binary, trial_warped)
            if trial_dice > best_dice:
                log(f'    ECC: shift=({dx_ecc:.1f}, {dy_ecc:.1f}), dice {best_dice:.4f}→{trial_dice:.4f} ✓')
                best_M = trial_M
                best_dice = trial_dice
            else:
                log(f'    ECC: shift=({dx_ecc:.1f}, {dy_ecc:.1f}), dice {trial_dice:.4f} no improvement')
    except cv2.error as e:
        log(f'    ECC failed: {e}')

    return best_M, best_dice


def _quick_dice(a, b):
    """Fast Dice on binary masks."""
    a_pos = a > 0
    b_pos = b > 0
    inter = np.sum(a_pos & b_pos)
    total = np.sum(a_pos) + np.sum(b_pos)
    return 2.0 * inter / (total + 1e-10)


def _quick_iou(a, b):
    """Fast IoU on binary masks."""
    a_pos = a > 0
    b_pos = b > 0
    inter = np.sum(a_pos & b_pos)
    union = np.sum(a_pos | b_pos)
    return float(inter) / (union + 1e-10)


def _quick_metric(a, b, metric='dice'):
    """Compute metric on binary masks. metric: 'dice' or 'iou'."""
    if metric == 'iou':
        return _quick_iou(a, b)
    return _quick_dice(a, b)


# ── Warp ───────────────────────────────────────────────────────────────────

def custom_warp_affine(image, M, output_shape, block_size=5000):
    """Block-by-block affine warp using inverse mapping. Out-of-bounds → 0."""
    h, w = output_shape
    out = np.zeros((h, w), dtype=image.dtype)
    M_inv = np.linalg.pinv(np.vstack([M, [0, 0, 1]]))[:2, :]

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            bh = min(block_size, h - y)
            bw = min(block_size, w - x)
            xv, yv = np.meshgrid(np.arange(x, x + bw), np.arange(y, y + bh))
            coords = np.stack([xv, yv, np.ones_like(xv)], axis=-1).reshape(-1, 3).T
            src = M_inv @ coords
            sx = np.round(src[0]).astype(int).reshape(bh, bw)
            sy = np.round(src[1]).astype(int).reshape(bh, bw)
            mask = (sx >= 0) & (sx < image.shape[1]) & (sy >= 0) & (sy < image.shape[0])
            out[y:y+bh, x:x+bw][mask] = image[sy[mask], sx[mask]]
    return out


def transform_contours(contours, M):
    """Apply affine M to list of contour arrays. Returns list of int arrays."""
    result = []
    for c in contours:
        hom = np.hstack([c, np.ones((c.shape[0], 1))])
        t = (M @ hom.T).T[:, :2]
        result.append(np.round(t).astype(int))
    return result


# ── Block-based refinement (v2: template matching + displacement smoothing) ──

def _estimate_block_displacement(ref_region, template, search_radius=50):
    """Estimate (dx, dy) displacement of template within ref_region using
    template matching with phase correlation fallback.
    Returns (dx, dy, confidence)."""
    if template.shape[0] < 10 or template.shape[1] < 10:
        return 0.0, 0.0, 0.0

    # Template matching: ref_region should be padded search area
    if (ref_region.shape[0] > template.shape[0] and
            ref_region.shape[1] > template.shape[1]):
        result = cv2.matchTemplate(ref_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        # The expected location of template within ref_region is (search_radius, search_radius)
        # if ref_region was padded by search_radius on each side
        dx = max_loc[0] - search_radius
        dy = max_loc[1] - search_radius
        return float(dx), float(dy), float(max_val)

    # Fallback: phase correlation (requires same-size inputs)
    b1 = ref_region[:template.shape[0], :template.shape[1]].astype(np.float64)
    b2 = template.astype(np.float64)
    if b1.shape != b2.shape or b1.size == 0:
        return 0.0, 0.0, 0.0
    (dx, dy), response = cv2.phaseCorrelate(b1, b2)
    return float(dx), float(dy), float(response)


def _fill_nan_neighbors(grid, max_iter=20):
    """Iteratively fill NaN values with mean of valid 8-neighbors."""
    filled = grid.copy()
    for _ in range(max_iter):
        nans = np.isnan(filled)
        if not nans.any():
            break
        rows, cols = np.where(nans)
        for r, c in zip(rows, cols):
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < filled.shape[0] and 0 <= nc < filled.shape[1]:
                        if not np.isnan(filled[nr, nc]):
                            neighbors.append(filled[nr, nc])
            if neighbors:
                filled[r, c] = np.mean(neighbors)
    return np.nan_to_num(filled, nan=0.0)


def align_contours_by_blocks(image1, image2, df, grouped_df,
                              num_blocks_split=32, min_contours=20,
                              search_radius=50, max_displacement=80,
                              smooth_sigma=1.5, conf_threshold=0.15,
                              metric='dice', log=print):
    """Per-block translation refinement using template matching.

    Improvements over v1 (AKAZE-based):
    - Template matching is more robust on binary masks than feature matching
    - Translation-only model (no per-block rotation/scale)
    - Displacement field smoothing interpolates failed blocks from neighbors
    - Quality gating: only apply correction if local Dice improves
    """
    # ── Phase A: Grid setup (group contours into blocks) ──
    log('  Grouping contours into blocks...')
    x_min, y_min = df[['vertex_x_p_trans', 'vertex_y_p_trans']].min()
    x_max, y_max = df[['vertex_x_p_trans', 'vertex_y_p_trans']].max()
    x_min, y_min = x_min - 1, y_min - 1
    x_max, y_max = x_max + 1, y_max + 1
    grid_w = (x_max - x_min) / num_blocks_split
    grid_h = (y_max - y_min) / num_blocks_split

    grid_contours = {}
    for label, group in tqdm(grouped_df, desc='  grouping'):
        contour = group[['vertex_x_p_trans', 'vertex_y_p_trans']].values
        center = np.mean(contour, axis=0).astype(int)
        gx = min(int((center[0] - x_min) / grid_w), num_blocks_split - 1)
        gy = min(int((center[1] - y_min) / grid_h), num_blocks_split - 1)
        grid_contours.setdefault((gx, gy), []).append((label, contour))

    # Merge small blocks into neighbors
    final = {k: v[:] for k, v in grid_contours.items()}
    for gk, contours_in in list(grid_contours.items()):
        if len(contours_in) >= min_contours:
            continue
        for item in contours_in:
            center = np.mean(item[1], axis=0)
            best_key, best_dist = None, float('inf')
            for ddx, ddy in [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]:
                nk = (gk[0]+ddx, gk[1]+ddy)
                if nk in final and len(final[nk]) >= min_contours:
                    nc = np.array([nk[0]*grid_w + grid_w/2 + x_min,
                                   nk[1]*grid_h + grid_h/2 + y_min])
                    d = np.linalg.norm(center - nc)
                    if d < best_dist:
                        best_dist, best_key = d, nk
            if best_key is not None:
                final[best_key].append(item)
        if gk in final:
            del final[gk]

    # ── Phase B: Per-block displacement estimation via template matching ──
    log(f'  Estimating displacements for {len(final)} blocks (search_radius={search_radius})...')
    block_meta = {}  # key -> (dx, dy, confidence, bbox)
    pad = search_radius

    for key, grouped in tqdm(final.items(), desc='  template match'):
        tmp = np.vstack([c[1] for c in grouped])
        xmin_b, ymin_b = tmp.min(axis=0) - 2
        xmax_b, ymax_b = tmp.max(axis=0) + 2
        xmin_b = max(0, int(xmin_b))
        ymin_b = max(0, int(ymin_b))
        xmax_b = min(image1.shape[1], int(xmax_b))
        ymax_b = min(image1.shape[0], int(ymax_b))

        template = image2[ymin_b:ymax_b, xmin_b:xmax_b]
        if template.size == 0 or template.shape[0] < 10 or template.shape[1] < 10:
            block_meta[key] = (0.0, 0.0, 0.0, (xmin_b, ymin_b, xmax_b, ymax_b))
            continue

        # Padded search region from reference image
        sy0 = max(0, ymin_b - pad)
        sx0 = max(0, xmin_b - pad)
        sy1 = min(image1.shape[0], ymax_b + pad)
        sx1 = min(image1.shape[1], xmax_b + pad)
        ref_region = image1[sy0:sy1, sx0:sx1]

        # Actual padding achieved (may be less at image boundaries)
        actual_pad_x = xmin_b - sx0
        actual_pad_y = ymin_b - sy0

        # Template matching
        if (ref_region.shape[0] > template.shape[0] and
                ref_region.shape[1] > template.shape[1]):
            result = cv2.matchTemplate(ref_region, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            dx = float(max_loc[0] - actual_pad_x)
            dy = float(max_loc[1] - actual_pad_y)
            conf = float(max_val)
        else:
            # Fallback: phase correlation (same-size blocks)
            b1 = image1[ymin_b:ymax_b, xmin_b:xmax_b].astype(np.float64)
            b2 = template.astype(np.float64)
            if b1.shape == b2.shape and b1.size > 0:
                (dx, dy), conf = cv2.phaseCorrelate(b1, b2)
                dx, dy, conf = float(dx), float(dy), float(conf)
            else:
                dx, dy, conf = 0.0, 0.0, 0.0

        # Reject unreasonable displacements
        if abs(dx) > max_displacement or abs(dy) > max_displacement:
            dx, dy, conf = 0.0, 0.0, 0.0

        block_meta[key] = (dx, dy, conf, (xmin_b, ymin_b, xmax_b, ymax_b))

    # ── Phase C: Build and smooth displacement field ──
    log('  Smoothing displacement field...')
    all_keys = list(block_meta.keys())
    gx_vals = [k[0] for k in all_keys]
    gy_vals = [k[1] for k in all_keys]
    gx_min, gx_max = min(gx_vals), max(gx_vals)
    gy_min, gy_max = min(gy_vals), max(gy_vals)
    grid_rows = gy_max - gy_min + 1
    grid_cols = gx_max - gx_min + 1

    dx_grid = np.full((grid_rows, grid_cols), np.nan)
    dy_grid = np.full((grid_rows, grid_cols), np.nan)
    conf_grid = np.zeros((grid_rows, grid_cols))

    n_confident = 0
    for key, (dx, dy, conf, _) in block_meta.items():
        gi = key[0] - gx_min
        gj = key[1] - gy_min
        if conf > conf_threshold:
            dx_grid[gj, gi] = dx
            dy_grid[gj, gi] = dy
            conf_grid[gj, gi] = conf
            n_confident += 1

    log(f'    {n_confident}/{len(block_meta)} blocks above confidence threshold ({conf_threshold})')

    # Fill NaN from neighbors, then smooth
    dx_filled = _fill_nan_neighbors(dx_grid)
    dy_filled = _fill_nan_neighbors(dy_grid)
    dx_smooth = gaussian_filter(dx_filled, sigma=smooth_sigma)
    dy_smooth = gaussian_filter(dy_filled, sigma=smooth_sigma)

    median_dx = np.nanmedian(dx_grid[~np.isnan(dx_grid)]) if n_confident > 0 else 0
    median_dy = np.nanmedian(dy_grid[~np.isnan(dy_grid)]) if n_confident > 0 else 0
    log(f'    Median displacement: ({median_dx:.1f}, {median_dy:.1f})')

    # ── Phase D: Quality-gated application to contours ──
    log('  Applying displacements with quality gating...')
    all_transformed = {}
    applied_blocks = 0
    gated_blocks = 0

    for key, grouped in tqdm(final.items(), desc='  applying'):
        gi = key[0] - gx_min
        gj = key[1] - gy_min
        dx = dx_smooth[gj, gi]
        dy = dy_smooth[gj, gi]

        # Quality gate: check if displacement improves local Dice
        bbox = block_meta[key][3]
        xmin_b, ymin_b, xmax_b, ymax_b = bbox
        block_ref = image1[ymin_b:ymax_b, xmin_b:xmax_b]
        block_mov = image2[ymin_b:ymax_b, xmin_b:xmax_b]

        if block_ref.size > 0 and block_mov.size > 0:
            score_before = _quick_metric(block_ref, block_mov, metric)
            # Simulate shifted block: shift the crop window
            sy0 = max(0, int(ymin_b - dy))
            sx0 = max(0, int(xmin_b - dx))
            sy1 = min(image2.shape[0], int(ymax_b - dy))
            sx1 = min(image2.shape[1], int(xmax_b - dx))
            bh = min(ymax_b - ymin_b, sy1 - sy0)
            bw = min(xmax_b - xmin_b, sx1 - sx0)
            if bh > 0 and bw > 0:
                block_shifted = image2[sy0:sy0+bh, sx0:sx0+bw]
                block_ref_crop = block_ref[:bh, :bw]
                score_after = _quick_metric(block_ref_crop, block_shifted, metric)
            else:
                score_after = 0.0

            if score_after <= score_before - 0.02:
                # Displacement makes it worse → don't apply
                dx, dy = 0.0, 0.0
                gated_blocks += 1
            else:
                applied_blocks += 1
        else:
            applied_blocks += 1

        offset = np.array([dx, dy])
        for label_id, contour in grouped:
            t_global = np.round(contour + offset).astype(int)
            cell_id = grouped_df.get_group(label_id)['cell_id'].iloc[0]
            all_transformed[label_id] = {
                'cell_id': cell_id,
                'contour': t_global,
            }

    log(f'  Block refinement done: {len(all_transformed)} contours, '
        f'{applied_blocks} blocks applied, {gated_blocks} gated (rejected)')

    # Return displacement field info for reuse on other parquet files
    disp_info = {
        'dx_smooth': dx_smooth, 'dy_smooth': dy_smooth,
        'gx_min': gx_min, 'gy_min': gy_min,
        'x_min': x_min, 'y_min': y_min,
        'grid_w': grid_w, 'grid_h': grid_h,
        'num_blocks': num_blocks_split,
    }
    return all_transformed, disp_info


# ── Wide-search retry for low-dice blocks ─────────────────────────────────

def retry_low_dice_blocks(image1, image2, all_transformed, grouped_df,
                           dice_threshold=0.3, search_radius=120,
                           num_blocks=32, min_tissue_frac=0.15,
                           metric='dice', log=print):
    """Re-run template matching with larger search_radius for blocks with low quality.

    Only retries blocks where HE tissue fraction is significant (not pure boundary),
    i.e. blocks that could genuinely benefit from a wider search.
    """
    # ── Group transformed contours into blocks ──
    label_ids = list(all_transformed.keys())
    centers = []
    for lid in label_ids:
        c = all_transformed[lid]['contour']
        centers.append(np.mean(c, axis=0))
    centers = np.array(centers)

    x_min_c, y_min_c = centers.min(axis=0) - 1
    x_max_c, y_max_c = centers.max(axis=0) + 1
    grid_w = (x_max_c - x_min_c) / num_blocks
    grid_h = (y_max_c - y_min_c) / num_blocks

    block_groups = {}
    for i, lid in enumerate(label_ids):
        cx, cy = centers[i]
        gx = min(int((cx - x_min_c) / grid_w), num_blocks - 1)
        gy = min(int((cy - y_min_c) / grid_h), num_blocks - 1)
        block_groups.setdefault((gx, gy), []).append(lid)

    # ── Identify low-dice blocks worth retrying ──
    retry_blocks = []
    for key, lids in block_groups.items():
        if len(lids) < 5:
            continue
        pts = np.vstack([all_transformed[lid]['contour'] for lid in lids])
        xmin_b, ymin_b = pts.min(axis=0) - 2
        xmax_b, ymax_b = pts.max(axis=0) + 2
        xmin_b = max(0, int(xmin_b))
        ymin_b = max(0, int(ymin_b))
        xmax_b = min(image1.shape[1], int(xmax_b))
        ymax_b = min(image1.shape[0], int(ymax_b))

        ref_crop = image1[ymin_b:ymax_b, xmin_b:xmax_b]
        mov_crop = image2[ymin_b:ymax_b, xmin_b:xmax_b]
        if ref_crop.size == 0:
            continue

        score = _quick_metric(ref_crop, mov_crop, metric)
        he_frac = np.sum(ref_crop > 0) / (ref_crop.size + 1e-10)

        if score < dice_threshold and he_frac >= min_tissue_frac:
            retry_blocks.append({
                'key': key, 'lids': lids,
                'bbox': (xmin_b, ymin_b, xmax_b, ymax_b),
                'score_before': score,
                'he_frac': he_frac,
            })

    log(f'    {len(retry_blocks)} interior blocks with {metric} < {dice_threshold} to retry '
        f'(search_radius={search_radius})')
    if not retry_blocks:
        return all_transformed, 0

    # ── Wide template matching for each block ──
    improved = 0
    pad = search_radius

    for rb in tqdm(retry_blocks, desc='  wide search'):
        xmin_b, ymin_b, xmax_b, ymax_b = rb['bbox']
        template = image2[ymin_b:ymax_b, xmin_b:xmax_b]
        if template.size == 0 or template.shape[0] < 10 or template.shape[1] < 10:
            continue

        # Padded search region
        sy0 = max(0, ymin_b - pad)
        sx0 = max(0, xmin_b - pad)
        sy1 = min(image1.shape[0], ymax_b + pad)
        sx1 = min(image1.shape[1], xmax_b + pad)
        ref_region = image1[sy0:sy1, sx0:sx1]

        actual_pad_x = xmin_b - sx0
        actual_pad_y = ymin_b - sy0

        if (ref_region.shape[0] <= template.shape[0] or
                ref_region.shape[1] <= template.shape[1]):
            continue

        result = cv2.matchTemplate(ref_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        dx = float(max_loc[0] - actual_pad_x)
        dy = float(max_loc[1] - actual_pad_y)

        if abs(dx) > search_radius or abs(dy) > search_radius:
            continue

        # Check if this improves Dice
        bh = ymax_b - ymin_b
        bw = xmax_b - xmin_b
        new_y0 = max(0, int(ymin_b - dy))
        new_x0 = max(0, int(xmin_b - dx))
        new_y1 = min(image2.shape[0], int(ymax_b - dy))
        new_x1 = min(image2.shape[1], int(xmax_b - dx))
        crop_h = min(bh, new_y1 - new_y0)
        crop_w = min(bw, new_x1 - new_x0)
        if crop_h <= 0 or crop_w <= 0:
            continue

        ref_crop = image1[ymin_b:ymin_b+crop_h, xmin_b:xmin_b+crop_w]
        shifted = image2[new_y0:new_y0+crop_h, new_x0:new_x0+crop_w]
        score_after = _quick_metric(ref_crop, shifted, metric)

        if score_after > rb['score_before'] + 0.02:
            offset = np.array([dx, dy])
            for lid in rb['lids']:
                all_transformed[lid]['contour'] = np.round(
                    all_transformed[lid]['contour'] + offset).astype(int)
            improved += 1
            if improved <= 10:
                log(f'      block {rb["key"]}: {metric} {rb["score_before"]:.3f}→{score_after:.3f} '
                    f'(dx={dx:.0f}, dy={dy:.0f}), he_frac={rb["he_frac"]:.2f}')

    log(f'    Wide search: {improved}/{len(retry_blocks)} blocks improved')
    return all_transformed, improved


# ── Failed block recalibration (brute-force rotation + translation search) ──

def recalibrate_failed_blocks(he_binary, all_transformed, grouped_df,
                               dice_threshold=0.3, search_range=60,
                               search_step=5, angle_range=3.0, angle_step=0.5,
                               block_pad=50, metric='dice', log=print):
    """Re-calibrate blocks with low quality by brute-force (dx, dy, theta) search.

    For each block below dice_threshold:
    1. Render local contour mask from current positions
    2. Try grid of (dx, dy, theta) combinations
    3. Keep the combo that maximizes local metric with HE binary
    4. Apply winning transform to all contours in the block
    """
    # ── Step 1: Identify blocks and compute per-block quality ──
    # Re-group transformed contours into spatial blocks
    log(f'  Recalibrating blocks with {metric} < {dice_threshold}...')
    all_contour_pts = []
    label_ids = list(all_transformed.keys())
    for lid in label_ids:
        c = all_transformed[lid]['contour']
        center = np.mean(c, axis=0)
        all_contour_pts.append(center)
    all_contour_pts = np.array(all_contour_pts)

    x_min_c, y_min_c = all_contour_pts.min(axis=0) - 1
    x_max_c, y_max_c = all_contour_pts.max(axis=0) + 1

    # Use ~32 blocks (same grid as main refinement)
    num_blocks = 32
    grid_w = (x_max_c - x_min_c) / num_blocks
    grid_h = (y_max_c - y_min_c) / num_blocks

    block_groups = {}  # (gx,gy) -> [label_id, ...]
    for i, lid in enumerate(label_ids):
        cx, cy = all_contour_pts[i]
        gx = min(int((cx - x_min_c) / grid_w), num_blocks - 1)
        gy = min(int((cy - y_min_c) / grid_h), num_blocks - 1)
        block_groups.setdefault((gx, gy), []).append(lid)

    # Compute per-block Dice
    failed_blocks = []
    for key, lids in block_groups.items():
        if len(lids) < 5:
            continue
        # Get bounding box of this block's contours
        pts = np.vstack([all_transformed[lid]['contour'] for lid in lids])
        xmin_b, ymin_b = pts.min(axis=0) - 2
        xmax_b, ymax_b = pts.max(axis=0) + 2
        xmin_b = max(0, int(xmin_b))
        ymin_b = max(0, int(ymin_b))
        xmax_b = min(he_binary.shape[1], int(xmax_b))
        ymax_b = min(he_binary.shape[0], int(ymax_b))

        # Skip if bounding box is invalid (contours outside image)
        if xmax_b <= xmin_b or ymax_b <= ymin_b:
            continue

        # Render local contour mask
        local_mask = np.zeros((ymax_b - ymin_b, xmax_b - xmin_b), dtype=np.uint8)
        for lid in lids:
            c = all_transformed[lid]['contour'] - np.array([xmin_b, ymin_b])
            cv2.fillPoly(local_mask, [c], 255)

        ref_crop = he_binary[ymin_b:ymax_b, xmin_b:xmax_b]
        if ref_crop.size == 0 or local_mask.size == 0:
            continue

        score = _quick_metric(ref_crop, local_mask, metric)
        if score < dice_threshold:
            failed_blocks.append({
                'key': key, 'lids': lids,
                'bbox': (xmin_b, ymin_b, xmax_b, ymax_b),
                'score_before': score,
                'local_mask': local_mask,
            })

    log(f'    {len(failed_blocks)} blocks below {metric} threshold ({dice_threshold})')
    if not failed_blocks:
        return all_transformed, 0

    # ── Step 2: Coarse-to-fine search for each failed block ──
    recalibrated = 0
    for bi, fb in enumerate(tqdm(failed_blocks, desc='  recalibrating')):
        xmin_b, ymin_b, xmax_b, ymax_b = fb['bbox']
        local_mask = fb['local_mask']
        bh, bw = local_mask.shape

        # Expand reference region to allow for search displacement
        ext = search_range + 10
        ey0 = max(0, ymin_b - ext)
        ex0 = max(0, xmin_b - ext)
        ey1 = min(he_binary.shape[0], ymax_b + ext)
        ex1 = min(he_binary.shape[1], xmax_b + ext)
        ref_ext = he_binary[ey0:ey1, ex0:ex1]

        off_y = ymin_b - ey0
        off_x = xmin_b - ex0

        best_score = fb['score_before']
        best_dx, best_dy, best_theta = 0.0, 0.0, 0.0
        cy_m, cx_m = bh / 2.0, bw / 2.0

        # --- Pass 1: Coarse search at 1/4 scale ---
        ds = 4
        ref_ext_s = cv2.resize(ref_ext, (ref_ext.shape[1]//ds, ref_ext.shape[0]//ds),
                                interpolation=cv2.INTER_NEAREST)
        local_mask_s = cv2.resize(local_mask, (bw//ds, bh//ds),
                                   interpolation=cv2.INTER_NEAREST)
        bh_s, bw_s = local_mask_s.shape
        off_y_s, off_x_s = off_y // ds, off_x // ds
        cy_ms, cx_ms = bh_s / 2.0, bw_s / 2.0

        coarse_step = max(search_step, 4)  # in downscaled pixels
        coarse_angles = np.arange(-angle_range, angle_range + 1.0, 1.0)
        coarse_dx = np.arange(-search_range//ds, search_range//ds + coarse_step, coarse_step)
        coarse_dy = np.arange(-search_range//ds, search_range//ds + coarse_step, coarse_step)

        for theta in coarse_angles:
            if abs(theta) < 0.01:
                rotated = local_mask_s
            else:
                rm = cv2.getRotationMatrix2D((cx_ms, cy_ms), theta, 1.0)
                rotated = cv2.warpAffine(local_mask_s, rm, (bw_s, bh_s),
                                          flags=cv2.INTER_NEAREST)
            for dx_s in coarse_dx:
                for dy_s in coarse_dy:
                    ry = int(off_y_s + dy_s)
                    rx = int(off_x_s + dx_s)
                    if ry < 0 or rx < 0:
                        continue
                    if ry + bh_s > ref_ext_s.shape[0] or rx + bw_s > ref_ext_s.shape[1]:
                        continue
                    ref_crop = ref_ext_s[ry:ry+bh_s, rx:rx+bw_s]
                    s = _quick_metric(ref_crop, rotated, metric)
                    if s > best_score:
                        best_score = s
                        best_dx = dx_s * ds
                        best_dy = dy_s * ds
                        best_theta = theta

        # --- Pass 2: Fine search around coarse best at full scale ---
        fine_radius = ds * coarse_step  # search around coarse best
        fine_dx = np.arange(best_dx - fine_radius, best_dx + fine_radius + 1, search_step)
        fine_dy = np.arange(best_dy - fine_radius, best_dy + fine_radius + 1, search_step)
        fine_angles = np.arange(best_theta - 1.0, best_theta + 1.0 + angle_step, angle_step)

        for theta in fine_angles:
            if abs(theta) < 0.01:
                rotated_mask = local_mask
            else:
                rot_mat = cv2.getRotationMatrix2D((cx_m, cy_m), theta, 1.0)
                rotated_mask = cv2.warpAffine(local_mask, rot_mat, (bw, bh),
                                               flags=cv2.INTER_NEAREST)
            for dx in fine_dx:
                for dy in fine_dy:
                    ry = int(off_y + dy)
                    rx = int(off_x + dx)
                    if ry < 0 or rx < 0:
                        continue
                    if ry + bh > ref_ext.shape[0] or rx + bw > ref_ext.shape[1]:
                        continue
                    ref_crop = ref_ext[ry:ry+bh, rx:rx+bw]
                    s = _quick_metric(ref_crop, rotated_mask, metric)
                    if s > best_score:
                        best_score = s
                        best_dx, best_dy, best_theta = dx, dy, theta

        if best_score > fb['score_before'] + 0.02:
            # Apply the winning transform
            offset = np.array([best_dx, best_dy])
            if abs(best_theta) > 0.01:
                # Rotation around block center
                cx_global = (xmin_b + xmax_b) / 2.0
                cy_global = (ymin_b + ymax_b) / 2.0
                rad = np.radians(best_theta)
                cos_t, sin_t = np.cos(rad), np.sin(rad)
                for lid in fb['lids']:
                    c = all_transformed[lid]['contour'].astype(np.float64)
                    c_centered = c - np.array([cx_global, cy_global])
                    rotated = np.column_stack([
                        c_centered[:, 0] * cos_t - c_centered[:, 1] * sin_t,
                        c_centered[:, 0] * sin_t + c_centered[:, 1] * cos_t,
                    ])
                    c_new = rotated + np.array([cx_global, cy_global]) + offset
                    all_transformed[lid]['contour'] = np.round(c_new).astype(int)
            else:
                for lid in fb['lids']:
                    all_transformed[lid]['contour'] = np.round(
                        all_transformed[lid]['contour'] + offset).astype(int)

            recalibrated += 1
            if bi < 10:
                log(f'    block {fb["key"]}: {metric} {fb["score_before"]:.3f}→{best_score:.3f} '
                    f'(dx={best_dx:.0f}, dy={best_dy:.0f}, θ={best_theta:.1f}°)')

    log(f'  Recalibration done: {recalibrated}/{len(failed_blocks)} blocks improved')
    return all_transformed, recalibrated


# ── Quality metrics ────────────────────────────────────────────────────────

def compute_quality(he_binary, filled_warped, he_gray, filled_warped_gray, log=print):
    """Compute alignment quality metrics."""
    # 1. Dice coefficient (binary overlap)
    intersection = np.sum((he_binary > 0) & (filled_warped > 0))
    union = np.sum(he_binary > 0) + np.sum(filled_warped > 0)
    dice = 2.0 * intersection / (union + 1e-10)

    # 2. IoU
    union_area = np.sum((he_binary > 0) | (filled_warped > 0))
    iou = intersection / (union_area + 1e-10)

    # 3. NCC on overlap region
    mask = (he_binary > 0) & (filled_warped > 0)
    if mask.sum() > 100:
        a = he_gray[mask].astype(np.float64)
        b = filled_warped_gray[mask].astype(np.float64)
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
        ncc = float(np.sum(a * b) / (denom + 1e-10))
    else:
        ncc = 0.0

    # 4. Coverage: fraction of HE tissue covered by warped contours
    he_tissue = np.sum(he_binary > 0)
    coverage = intersection / (he_tissue + 1e-10)

    return {
        'dice': round(dice, 4),
        'iou': round(iou, 4),
        'ncc': round(ncc, 4),
        'coverage': round(coverage, 4),
    }


def compute_interior_quality(he_binary, contour_mask, num_blocks=32,
                              min_tissue_frac=0.25, log=print):
    """Compute Dice and IoU on interior blocks where both HE and contours are dense.
    Excludes boundary/sparse blocks that drag down the global metric."""
    h, w = he_binary.shape
    bh = h // num_blocks
    bw = w // num_blocks

    block_dices = []
    block_ious = []
    for gy in range(num_blocks):
        for gx in range(num_blocks):
            y0, y1 = gy * bh, min((gy + 1) * bh, h)
            x0, x1 = gx * bw, min((gx + 1) * bw, w)
            ref_crop = he_binary[y0:y1, x0:x1]
            mov_crop = contour_mask[y0:y1, x0:x1]
            area = ref_crop.size
            if area == 0:
                continue
            he_frac = np.sum(ref_crop > 0) / area
            cont_frac = np.sum(mov_crop > 0) / area
            if he_frac < min_tissue_frac or cont_frac < 0.02:
                continue
            block_dices.append(_quick_dice(ref_crop, mov_crop))
            block_ious.append(_quick_iou(ref_crop, mov_crop))

    if not block_dices:
        return {'interior_dice': 0.0, 'interior_dice_median': 0.0,
                'interior_dice_min': 0.0, 'interior_dice_max': 0.0,
                'interior_iou': 0.0, 'interior_iou_median': 0.0, 'interior_blocks': 0}

    return {
        'interior_dice': round(float(np.mean(block_dices)), 4),
        'interior_dice_median': round(float(np.median(block_dices)), 4),
        'interior_dice_min': round(float(np.min(block_dices)), 4),
        'interior_dice_max': round(float(np.max(block_dices)), 4),
        'interior_iou': round(float(np.mean(block_ious)), 4),
        'interior_iou_median': round(float(np.median(block_ious)), 4),
        'interior_iou_min': round(float(np.min(block_ious)), 4),
        'interior_iou_max': round(float(np.max(block_ious)), 4),
        'interior_blocks': len(block_dices),
    }


# ── GeoJSON export ────────────────────────────────────────────────────────

def contours_to_geojson(all_transformed, out_path, label='nucleus', log=print):
    """Export transformed contours as QuPath-compatible GeoJSON.

    Format follows the FeatureCollection schema used by QuPath/Xenium:
    - Each cell is a Feature with Polygon geometry
    - Coordinates are in HE pixel space [x, y]
    - Properties include cell_id and objectType
    """
    import uuid

    features = []
    for label_id, data in all_transformed.items():
        contour = data['contour']
        cell_id = str(data.get('cell_id', label_id))

        # GeoJSON Polygon: list of rings, first ring is exterior
        # Close the polygon (first == last point)
        coords = contour.tolist()
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        feature = {
            'type': 'Feature',
            'id': str(uuid.uuid4()),
            'geometry': {
                'type': 'Polygon',
                'coordinates': [coords],
            },
            'properties': {
                'objectType': 'detection',
                'name': cell_id,
                'classification': {
                    'name': label,
                    'color': [255, 255, 0] if label == 'nucleus' else [0, 255, 0],
                },
            },
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features,
    }

    with open(out_path, 'w') as f:
        json.dump(geojson, f)

    log(f'  Saved {len(features)} {label} contours to {out_path}')
    return out_path


# ── Apply existing transform to another parquet ──────────────────────────

def apply_transform_to_parquet(parquet_path, M, disp_info, all_nuc_transformed,
                                scale_factor, he_shape, out_dir, image_he=None,
                                label='cell', log=print):
    """Apply the same global M + block displacement field to a different parquet.

    Uses the displacement field computed from nucleus registration to transform
    cell boundaries (or any other parquet with the same coordinate system).
    """
    import pandas as pd

    log(f'  Loading {label} boundaries...')
    df = pd.read_parquet(parquet_path)
    n_cells = len(df['cell_id'].unique())
    log(f'  {len(df)} vertices, {n_cells} cells')

    # Convert physical coords to pixel coords
    df[['vertex_x_p', 'vertex_y_p']] = (df[['vertex_x', 'vertex_y']] * scale_factor).astype(np.int32)
    # Ensure label_id exists (some Xenium versions only have cell_id)
    if 'label_id' not in df.columns:
        df['label_id'] = df['cell_id']
    grouped_df = df.groupby('label_id')

    # Apply global affine M
    log(f'  Applying global transform to {label} contours...')
    transformed_coords = transform_contours([df[['vertex_x_p', 'vertex_y_p']].values], M)[0]
    df[['vertex_x_p_trans', 'vertex_y_p_trans']] = transformed_coords

    # Apply block displacement field
    log(f'  Applying block displacements to {label} contours...')
    dx_smooth = disp_info['dx_smooth']
    dy_smooth = disp_info['dy_smooth']
    gx_min = disp_info['gx_min']
    gy_min = disp_info['gy_min']
    x_min = disp_info['x_min']
    y_min = disp_info['y_min']
    grid_w = disp_info['grid_w']
    grid_h = disp_info['grid_h']
    num_blocks = disp_info['num_blocks']
    grid_rows, grid_cols = dx_smooth.shape

    all_transformed = {}
    for label_id, group in tqdm(grouped_df, desc=f'  {label} transform'):
        contour = group[['vertex_x_p_trans', 'vertex_y_p_trans']].values
        center = np.mean(contour, axis=0)
        gx = min(int((center[0] - x_min) / grid_w), num_blocks - 1)
        gy = min(int((center[1] - y_min) / grid_h), num_blocks - 1)

        # Look up displacement from smoothed field
        gi = gx - gx_min
        gj = gy - gy_min
        if 0 <= gi < grid_cols and 0 <= gj < grid_rows:
            dx = dx_smooth[gj, gi]
            dy = dy_smooth[gj, gi]
        else:
            dx, dy = 0.0, 0.0

        offset = np.array([dx, dy])
        t_contour = np.round(contour + offset).astype(int)
        cell_id = group['cell_id'].iloc[0]
        all_transformed[label_id] = {'cell_id': cell_id, 'contour': t_contour}

    log(f'  {label}: {len(all_transformed)} contours transformed')

    # Also apply any per-contour corrections from wide-search/recalibration
    # by matching nucleus block shifts to cell contours in the same spatial region
    # (The displacement field already captures this since it was smoothed)

    # Save overlay
    if image_he is not None:
        log(f'  Saving {label} overlay...')
        overlay = image_he.copy()
        for lid, data in all_transformed.items():
            cv2.drawContours(overlay, [data['contour']], -1, (0, 255, 0), 2)
        ds = 4
        overlay_small = cv2.resize(overlay, (overlay.shape[1] // ds, overlay.shape[0] // ds))
        cv2.imwrite(os.path.join(out_dir, f'overlay_{label}.png'), overlay_small)

    return all_transformed, n_cells


# ── Main pipeline ──────────────────────────────────────────────────────────

def register_one(he_path, parquet_path, out_dir, cell_parquet_path=None,
                 metric='dice', refine=False, log=print):
    import pandas as pd
    from skimage import io as skio
    import tifffile

    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load parquet ───────────────────────────────────────────────────
    log('  Loading nucleus boundaries...')
    df = pd.read_parquet(parquet_path)
    n_cells = len(df['cell_id'].unique())
    log(f'  {len(df)} vertices, {n_cells} cells')

    # Get HE pixel size from OME metadata
    import xml.etree.ElementTree as ET
    with tifffile.TiffFile(he_path) as t:
        if t.ome_metadata:
            root = ET.fromstring(t.ome_metadata)
            ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
            pixels = root.find('.//ome:Pixels', ns)
            mpp = float(pixels.get('PhysicalSizeX', '0.2125'))
        else:
            mpp = 0.2125
    scale_factor = 1.0 / mpp
    log(f'  HE mpp={mpp:.4f} µm/px, scale_factor={scale_factor:.2f}')

    # Convert physical coords to pixel coords
    df[['vertex_x_p', 'vertex_y_p']] = (df[['vertex_x', 'vertex_y']] * scale_factor).astype(np.int32)
    # Ensure label_id exists (some Xenium versions only have cell_id)
    if 'label_id' not in df.columns:
        df['label_id'] = df['cell_id']
    grouped_df = df.groupby('label_id')

    # ── 2. Render filled contour image ────────────────────────────────────
    filled_path = os.path.join(out_dir, 'filled_image.png')
    if os.path.exists(filled_path):
        log('  Loading cached filled_image...')
        try:
            filled_image = cv2.imread(filled_path, cv2.IMREAD_GRAYSCALE)
            if filled_image is None:
                raise ValueError('cv2.imread returned None')
        except Exception:
            from PIL import Image as _PIL_Image
            _PIL_Image.MAX_IMAGE_PIXELS = None
            filled_image = np.array(_PIL_Image.open(filled_path).convert('L'))
    else:
        log('  Rendering filled contours...')
        img_h = df['vertex_y_p'].max() + 1
        img_w = df['vertex_x_p'].max() + 1
        filled_image = np.zeros((img_h, img_w), dtype=np.uint8)
        for label, group in tqdm(grouped_df, desc='  filling'):
            contour = group[['vertex_x_p', 'vertex_y_p']].values
            cv2.fillPoly(filled_image, [contour], color=255)
        cv2.imwrite(filled_path, filled_image)
    log(f'  filled_image: {filled_image.shape}')

    # ── 3. Load HE and convert to binary ──────────────────────────────────
    log('  Loading HE image...')
    image_he = skio.imread(he_path)
    log(f'  HE shape: {image_he.shape}')

    he_binary = myrgb2gray(image_he, reverse=True, binary=True)
    he_gray = myrgb2gray(image_he, reverse=True, binary=False)
    log(f'  HE binary: {he_binary.shape}')

    # ── 4. Global ORB alignment ───────────────────────────────────────────
    log('  Global ORB alignment...')
    t0 = time.time()
    M, inliers, n_matches = orb_align(he_binary, filled_image, nfeatures=100000, scale_factor=0.2)
    dt = time.time() - t0
    theta = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
    log(f'  ORB: {inliers} inliers / {n_matches} matches, '
        f'angle={theta:.2f}°, scale={scale:.4f}, time={dt:.1f}s')
    log(f'  M = {M.tolist()}')

    # Warp filled image
    log('  Warping filled image...')
    aligned_image = custom_warp_affine(filled_image, M, he_binary.shape[:2])

    # ── 4.5. Quality after global alignment ───────────────────────────────
    aligned_gray = aligned_image  # already grayscale
    q_global = compute_quality(he_binary, aligned_image, he_gray, aligned_gray, log)
    log(f'  Global quality: dice={q_global["dice"]}, iou={q_global["iou"]}, '
        f'ncc={q_global["ncc"]}, coverage={q_global["coverage"]}')

    # ── 4.6. Optional: Refine global translation via phase correlation + ECC
    if refine:
        log('  [Refine] Refining global translation...')
        t0 = time.time()
        M_refined, refined_dice = refine_global_translation(
            he_binary, filled_image, M, custom_warp_affine, log=log)
        dt = time.time() - t0
        log(f'  Translation refinement: {dt:.1f}s')

        if refined_dice > q_global['dice']:
            log(f'  Using refined M (dice {q_global["dice"]}→{refined_dice:.4f})')
            M = M_refined
            aligned_image = custom_warp_affine(filled_image, M, he_binary.shape[:2])
            q_global = compute_quality(he_binary, aligned_image, he_gray, aligned_image, log)
            log(f'  Refined global quality: dice={q_global["dice"]}, iou={q_global["iou"]}, '
                f'ncc={q_global["ncc"]}, coverage={q_global["coverage"]}')
        else:
            log('  No improvement from translation refinement, keeping original M')
    else:
        log('  Skipping translation refinement (use --refine to enable)')

    # Transform contour coords with global M
    log('  Transforming contour coordinates...')
    transformed_coords = transform_contours([df[['vertex_x_p', 'vertex_y_p']].values], M)[0]
    df[['vertex_x_p_trans', 'vertex_y_p_trans']] = transformed_coords

    # ── 5. Block-based template matching refinement ─────────────────────────
    log('  Block-based template matching refinement...')
    t0 = time.time()
    all_transformed, disp_info = align_contours_by_blocks(
        he_binary, aligned_image, df, grouped_df,
        num_blocks_split=32, min_contours=20,
        search_radius=50, max_displacement=80,
        smooth_sigma=1.5, conf_threshold=0.15, metric=metric, log=log)
    dt = time.time() - t0
    log(f'  Block refinement time: {dt:.1f}s')

    # ── 5.5. Quality after block refinement ───────────────────────────────
    refined_image = np.zeros_like(he_binary)
    for label_id, data in all_transformed.items():
        cv2.fillPoly(refined_image, [data['contour']], color=255)
    q_refined = compute_quality(he_binary, refined_image, he_gray, refined_image, log)
    log(f'  Refined quality: dice={q_refined["dice"]}, iou={q_refined["iou"]}, '
        f'ncc={q_refined["ncc"]}, coverage={q_refined["coverage"]}')

    # Interior Dice (primary metric — excludes boundary/sparse blocks)
    q_interior = compute_interior_quality(he_binary, refined_image, log=log)
    log(f'  Interior quality: dice={q_interior["interior_dice"]} (median={q_interior["interior_dice_median"]}), '
        f'iou={q_interior["interior_iou"]} (median={q_interior["interior_iou_median"]}), '
        f'blocks={q_interior["interior_blocks"]}')

    # ── 5.6. Wide-search retry for low-dice blocks ─────────────────────────
    log('  Wide-search retry for low-dice blocks...')
    t0 = time.time()
    all_transformed, n_wide = retry_low_dice_blocks(
        he_binary, aligned_image, all_transformed, grouped_df,
        dice_threshold=0.3, search_radius=120,
        min_tissue_frac=0.15, metric=metric, log=log)
    dt = time.time() - t0
    log(f'  Wide-search time: {dt:.1f}s')

    # ── 5.7. Brute-force recalibration for remaining failed blocks ────────
    log('  Recalibrating remaining failed blocks...')
    t0 = time.time()
    all_transformed, n_recal = recalibrate_failed_blocks(
        he_binary, all_transformed, grouped_df,
        dice_threshold=0.3, search_range=120, search_step=5,
        angle_range=3.0, angle_step=0.5, metric=metric, log=log)
    dt = time.time() - t0
    log(f'  Recalibration time: {dt:.1f}s')

    if n_wide > 0 or n_recal > 0:
        refined_image = np.zeros_like(he_binary)
        for label_id, data in all_transformed.items():
            cv2.fillPoly(refined_image, [data['contour']], color=255)
        q_refined = compute_quality(he_binary, refined_image, he_gray, refined_image, log)
        q_interior = compute_interior_quality(he_binary, refined_image, log=log)
        log(f'  After refinement: dice={q_refined["dice"]}, '
            f'interior_dice={q_interior["interior_dice"]}, '
            f'median={q_interior["interior_dice_median"]}')

    # ── 6. Save outputs ──────────────────────────────────────────────────
    # Overlay: contours on HE
    log('  Saving overlay...')
    overlay = image_he.copy()
    for label_id, data in all_transformed.items():
        cv2.drawContours(overlay, [data['contour']], -1, (0, 255, 255), 2)

    # Save a downscaled overlay for quick viewing
    ds = 4
    overlay_small = cv2.resize(overlay, (overlay.shape[1] // ds, overlay.shape[0] // ds))
    cv2.imwrite(os.path.join(out_dir, 'overlay_contours.png'), overlay_small)

    # Save block comparison
    block_pos = (min(he_binary.shape[1] // 2, 25000), min(he_binary.shape[0] // 2, 25000))
    block_size = 1000
    y0, x0 = block_pos[1], block_pos[0]
    y1, x1 = min(y0 + block_size, overlay.shape[0]), min(x0 + block_size, overlay.shape[1])
    block_overlay = overlay[y0:y1, x0:x1]
    cv2.imwrite(os.path.join(out_dir, 'block_detail.png'), block_overlay)

    # Save transform
    np.save(os.path.join(out_dir, 'affine_M.npy'), M)

    # ── 6.5. Export GeoJSON ──────────────────────────────────────────────
    log('  Exporting nucleus GeoJSON...')
    t0 = time.time()
    contours_to_geojson(all_transformed,
                        os.path.join(out_dir, 'nucleus_boundaries.geojson'),
                        label='nucleus', log=log)
    log(f'  GeoJSON export time: {time.time() - t0:.1f}s')

    # ── 7. Cell boundaries (reuse nucleus transform) ─────────────────────
    cell_result = {}
    if cell_parquet_path:
        log('')
        log('=== Cell Boundary Registration (reusing nucleus transform) ===')
        t0 = time.time()
        cell_transformed, n_cell = apply_transform_to_parquet(
            cell_parquet_path, M, disp_info, all_transformed,
            scale_factor, he_binary.shape, out_dir,
            image_he=image_he, label='cell', log=log)
        dt = time.time() - t0
        log(f'  Cell boundary transform time: {dt:.1f}s')

        # Quality assessment for cell boundaries
        cell_mask = np.zeros_like(he_binary)
        for lid, data in cell_transformed.items():
            cv2.fillPoly(cell_mask, [data['contour']], color=255)
        q_cell = compute_quality(he_binary, cell_mask, he_gray, cell_mask, log)
        q_cell_interior = compute_interior_quality(he_binary, cell_mask, log=log)
        log(f'  Cell quality: dice={q_cell["dice"]}, iou={q_cell["iou"]}, coverage={q_cell["coverage"]}')
        log(f'  Cell interior: dice={q_cell_interior["interior_dice"]} '
            f'(median={q_cell_interior["interior_dice_median"]}), '
            f'iou={q_cell_interior["interior_iou"]} '
            f'(median={q_cell_interior["interior_iou_median"]})')

        # Export cell GeoJSON
        log('  Exporting cell GeoJSON...')
        t0 = time.time()
        contours_to_geojson(cell_transformed,
                            os.path.join(out_dir, 'cell_boundaries.geojson'),
                            label='cell', log=log)
        log(f'  Cell GeoJSON export time: {time.time() - t0:.1f}s')

        cell_result = {
            'cell_parquet_path': cell_parquet_path,
            'n_cell_contours': n_cell,
            'n_cell_transformed': len(cell_transformed),
            'cell_quality': q_cell,
            'cell_quality_interior': q_cell_interior,
        }

    # Save results
    result = {
        'he_path': he_path,
        'parquet_path': parquet_path,
        'mpp': mpp,
        'he_shape': list(image_he.shape),
        'n_cells': n_cells,
        'orb_inliers': inliers,
        'orb_matches': n_matches,
        'angle_deg': round(theta, 2),
        'scale': round(scale, 4),
        'affine_M': M.tolist(),
        'quality_global': q_global,
        'quality_refined': q_refined,
        'quality_interior': q_interior,
        'metric': metric,
        'n_transformed_contours': len(all_transformed),
        **cell_result,
    }
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    # Save displacement field for reproducibility
    disp_save = {k: v for k, v in disp_info.items()}
    np.savez(os.path.join(out_dir, 'disp_info.npz'), **disp_save)

    log(f'  Done. Results saved to {out_dir}')
    return result


def main():
    parser = argparse.ArgumentParser(description='Xenium nucleus boundary to HE registration')
    parser.add_argument('--he', required=True, help='Path to HE OME-TIFF')
    parser.add_argument('--parquet', required=True, help='Path to nucleus_boundaries.parquet')
    parser.add_argument('--cell-parquet', default=None,
                        help='Path to cell_boundaries.parquet (optional, reuses nucleus transform)')
    parser.add_argument('--output_dir', default='Xenium_offical/output')
    parser.add_argument('--metric', choices=['dice', 'iou'], default='dice',
                        help='Quality metric for gating: dice (default) or iou')
    parser.add_argument('--refine', action='store_true',
                        help='Enable refine steps (wide-search retry + brute-force recalibration) for higher accuracy')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = open(os.path.join(args.output_dir, 'run.log'), 'w', buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    log('Xenium Registration')
    log(f'HE: {args.he}')
    log(f'Nucleus parquet: {args.parquet}')
    if args.cell_parquet:
        log(f'Cell parquet: {args.cell_parquet}')
    log(f'Output: {args.output_dir}')

    try:
        result = register_one(args.he, args.parquet, args.output_dir,
                              cell_parquet_path=args.cell_parquet,
                              metric=args.metric, refine=args.refine, log=log)
        qi = result['quality_interior']
        log('\n=== SUMMARY ===')
        log(f'  Cells: {result["n_cells"]}')
        log(f'  ORB: {result["orb_inliers"]} inliers / {result["orb_matches"]} matches')
        log(f'  Angle: {result["angle_deg"]}°, Scale: {result["scale"]}')
        log(f'  Global:  dice={result["quality_global"]["dice"]}  '
            f'iou={result["quality_global"]["iou"]}  '
            f'coverage={result["quality_global"]["coverage"]}')
        log(f'  Refined: dice={result["quality_refined"]["dice"]}  '
            f'iou={result["quality_refined"]["iou"]}  '
            f'coverage={result["quality_refined"]["coverage"]}')
        log(f'  Interior: dice={qi["interior_dice"]} (median={qi["interior_dice_median"]})  '
            f'iou={qi["interior_iou"]} (median={qi["interior_iou_median"]})  '
            f'blocks={qi["interior_blocks"]}')
        log(f'  Metric used for gating: {args.metric}')
        if 'cell_quality_interior' in result:
            qci = result['cell_quality_interior']
            log(f'  --- Cell Boundaries ---')
            log(f'  Cell contours: {result["n_cell_transformed"]}')
            log(f'  Cell: dice={result["cell_quality"]["dice"]}  '
                f'iou={result["cell_quality"]["iou"]}  '
                f'coverage={result["cell_quality"]["coverage"]}')
            log(f'  Cell interior: dice={qci["interior_dice"]} (median={qci["interior_dice_median"]})  '
                f'iou={qci["interior_iou"]} (median={qci["interior_iou_median"]})')
    except Exception as e:
        tb = traceback.format_exc()
        log(f'ERROR: {e}\n{tb}')

    log_file.close()


if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    main()
