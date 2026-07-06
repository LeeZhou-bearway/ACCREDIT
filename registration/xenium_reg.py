"""
xenium_reg.py - Xenium nucleus boundary contours to HE image registration.

Pipeline: contour rendering → feature-based global alignment → block-based refinement.
All functions are designed to be called by the LLM agent via tools.
"""
import cv2
import json
import numpy as np
import os
import time
from tqdm import tqdm


# ── In-memory stores ──────────────────────────────────────────────────────

_contour_store: dict = {}     # dataset_id -> {df, grouped_df, mpp, scale_factor, ...}
_image_store: dict = {}       # image_id -> np.ndarray
_transform_store: dict = {}   # transform_id -> 2x3 affine matrix (np.ndarray)


def get_image(image_id: str) -> np.ndarray:
    if image_id not in _image_store:
        raise KeyError(f"image_id '{image_id}' not found")
    return _image_store[image_id]


def get_transform(transform_id: str) -> np.ndarray:
    if transform_id not in _transform_store:
        raise KeyError(f"transform_id '{transform_id}' not found")
    return _transform_store[transform_id]


# ── Load & preprocess ─────────────────────────────────────────────────────

def load_xenium_data(he_path: str, parquet_path: str) -> dict:
    """Load HE OME-TIFF and nucleus_boundaries.parquet.
    Returns dataset info dict with dataset_id."""
    import pandas as pd
    import tifffile
    import xml.etree.ElementTree as ET

    # Get mpp from OME metadata
    with tifffile.TiffFile(he_path) as t:
        page = t.pages[0]
        he_shape = page.shape
        mpp = 0.2125  # default
        if t.ome_metadata:
            root = ET.fromstring(t.ome_metadata)
            ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
            pixels = root.find('.//ome:Pixels', ns)
            if pixels is not None:
                mpp = float(pixels.get('PhysicalSizeX', '0.2125'))

    scale_factor = 1.0 / mpp

    # Load parquet
    df = pd.read_parquet(parquet_path)
    n_cells = len(df['cell_id'].unique())

    # Convert physical coords to pixel coords
    df[['vertex_x_p', 'vertex_y_p']] = (df[['vertex_x', 'vertex_y']] * scale_factor).astype(np.int32)
    grouped_df = df.groupby('label_id')

    dataset_id = os.path.basename(os.path.dirname(he_path))
    _contour_store[dataset_id] = {
        'df': df,
        'grouped_df': grouped_df,
        'mpp': mpp,
        'scale_factor': scale_factor,
        'he_path': he_path,
        'he_shape': he_shape,
        'n_cells': n_cells,
    }

    return {
        'dataset_id': dataset_id,
        'he_shape': list(he_shape),
        'mpp': round(mpp, 4),
        'scale_factor': round(scale_factor, 2),
        'n_cells': n_cells,
        'n_vertices': len(df),
        'contour_x_range': [float(df['vertex_x_p'].min()), float(df['vertex_x_p'].max())],
        'contour_y_range': [float(df['vertex_y_p'].min()), float(df['vertex_y_p'].max())],
    }


def render_contour_mask(dataset_id: str, output_dir: str = None) -> dict:
    """Render all nucleus contours as filled binary mask."""
    data = _contour_store[dataset_id]
    df = data['df']
    grouped_df = data['grouped_df']

    # Check cache
    if output_dir:
        cache_path = os.path.join(output_dir, 'filled_image.png')
        if os.path.exists(cache_path):
            filled = cv2.imread(cache_path, cv2.IMREAD_GRAYSCALE)
            image_id = f"{dataset_id}__contour_mask"
            _image_store[image_id] = filled
            return {'image_id': image_id, 'shape': list(filled.shape), 'cached': True}

    img_h = int(df['vertex_y_p'].max()) + 1
    img_w = int(df['vertex_x_p'].max()) + 1
    filled = np.zeros((img_h, img_w), dtype=np.uint8)

    for label, group in tqdm(grouped_df, desc='rendering contours'):
        contour = group[['vertex_x_p', 'vertex_y_p']].values
        cv2.fillPoly(filled, [contour], color=255)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, 'filled_image.png'), filled)

    image_id = f"{dataset_id}__contour_mask"
    _image_store[image_id] = filled
    return {'image_id': image_id, 'shape': list(filled.shape), 'cached': False}


def convert_he_binary(dataset_id: str, alpha: float = 1.4,
                       r_weight: float = 0.3, g_weight: float = 0.1, b_weight: float = 0.6) -> dict:
    """Load HE image and convert to enhanced binary (contrast + weighted RGB + invert + Otsu).
    Parameters are adjustable by the agent."""
    from skimage import io as skio

    data = _contour_store[dataset_id]
    he_path = data['he_path']

    image_he = skio.imread(he_path)
    he_rgb_id = f"{dataset_id}__he_rgb"
    _image_store[he_rgb_id] = image_he

    # Enhanced grayscale
    enhanced = cv2.convertScaleAbs(image_he, alpha=alpha, beta=0)
    gray = (r_weight * enhanced[:, :, 2] + g_weight * enhanced[:, :, 1] + b_weight * enhanced[:, :, 0]).astype(np.uint8)

    # Inverted grayscale (for NCC quality)
    gray_inv = 255 - gray
    gray_inv_id = f"{dataset_id}__he_gray_inv"
    _image_store[gray_inv_id] = gray_inv

    # Binary (Otsu)
    binary = 255 - gray
    _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_id = f"{dataset_id}__he_binary"
    _image_store[binary_id] = binary

    return {
        'he_rgb_id': he_rgb_id,
        'he_binary_id': binary_id,
        'he_gray_inv_id': gray_inv_id,
        'shape': list(image_he.shape),
        'alpha': alpha,
        'rgb_weights': [r_weight, g_weight, b_weight],
    }


# ── Global alignment ──────────────────────────────────────────────────────

def run_feature_alignment(ref_id: str, moving_id: str,
                           method: str = 'orb',
                           nfeatures: int = 100000,
                           scale_factor: float = 0.2) -> dict:
    """Global feature-based alignment. Agent can try different methods/params.
    Methods: 'orb', 'akaze', 'sift'."""
    ref = get_image(ref_id)
    moving = get_image(moving_id)

    if scale_factor != 1:
        ref_s = cv2.resize(ref, (int(ref.shape[1] * scale_factor), int(ref.shape[0] * scale_factor)))
        mov_s = cv2.resize(moving, (int(moving.shape[1] * scale_factor), int(moving.shape[0] * scale_factor)))
    else:
        ref_s, mov_s = ref, moving

    if method == 'orb':
        detector = cv2.ORB_create(nfeatures=nfeatures)
        norm = cv2.NORM_HAMMING
    elif method == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif method == 'sift':
        detector = cv2.SIFT_create(nfeatures=nfeatures)
        norm = cv2.NORM_L2
    else:
        raise ValueError(f"Unknown method: {method}")

    kp1, des1 = detector.detectAndCompute(ref_s, None)
    kp2, des2 = detector.detectAndCompute(mov_s, None)

    if des1 is None or des2 is None:
        return {'error': f'No features detected with {method}', 'transform_id': None}
    if len(kp1) < 4 or len(kp2) < 4:
        return {'error': f'Too few features: {len(kp1)}/{len(kp2)}', 'transform_id': None}

    if method == 'sift':
        bf = cv2.BFMatcher(norm)
        raw_matches = bf.knnMatch(des1, des2, k=2)
        matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
    else:
        bf = cv2.BFMatcher(norm, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

    if len(matches) < 4:
        return {'error': f'Too few matches: {len(matches)}', 'transform_id': None}

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts)

    if M is None:
        return {'error': 'estimateAffinePartial2D failed', 'transform_id': None}

    M[:, 2] = M[:, 2] / scale_factor
    inliers = int(mask.sum()) if mask is not None else 0
    theta = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    scale = float(np.sqrt(M[0, 0]**2 + M[1, 0]**2))

    transform_id = f"global_{method}_{ref_id}__{moving_id}"
    _transform_store[transform_id] = M

    return {
        'transform_id': transform_id,
        'method': method,
        'nfeatures': nfeatures,
        'scale_factor': scale_factor,
        'keypoints_ref': len(kp1),
        'keypoints_moving': len(kp2),
        'total_matches': len(matches),
        'inliers': inliers,
        'angle_deg': round(theta, 2),
        'scale': round(scale, 4),
    }


# ── Warp & transform contours ─────────────────────────────────────────────

def warp_contour_mask(moving_id: str, transform_id: str, ref_id: str) -> dict:
    """Warp contour mask image using affine transform. Returns warped image_id."""
    moving = get_image(moving_id)
    M = get_transform(transform_id)
    ref = get_image(ref_id)

    h, w = ref.shape[:2]
    out = np.zeros((h, w), dtype=moving.dtype)
    M_inv = np.linalg.pinv(np.vstack([M, [0, 0, 1]]))[:2, :]

    block_size = 5000
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            bh = min(block_size, h - y)
            bw = min(block_size, w - x)
            xv, yv = np.meshgrid(np.arange(x, x + bw), np.arange(y, y + bh))
            coords = np.stack([xv, yv, np.ones_like(xv)], axis=-1).reshape(-1, 3).T
            src = M_inv @ coords
            sx = np.round(src[0]).astype(int).reshape(bh, bw)
            sy = np.round(src[1]).astype(int).reshape(bh, bw)
            mask = (sx >= 0) & (sx < moving.shape[1]) & (sy >= 0) & (sy < moving.shape[0])
            out[y:y+bh, x:x+bw][mask] = moving[sy[mask], sx[mask]]

    warped_id = f"{moving_id}__warped"
    _image_store[warped_id] = out
    return {'image_id': warped_id, 'shape': list(out.shape)}


def apply_global_transform(dataset_id: str, transform_id: str) -> dict:
    """Apply global affine transform to all contour coordinates."""
    data = _contour_store[dataset_id]
    df = data['df']
    M = get_transform(transform_id)

    coords = df[['vertex_x_p', 'vertex_y_p']].values
    hom = np.hstack([coords, np.ones((coords.shape[0], 1))])
    t = (M @ hom.T).T[:, :2]
    df[['vertex_x_p_trans', 'vertex_y_p_trans']] = np.round(t).astype(int)

    return {
        'n_vertices_transformed': len(df),
        'trans_x_range': [int(df['vertex_x_p_trans'].min()), int(df['vertex_x_p_trans'].max())],
        'trans_y_range': [int(df['vertex_y_p_trans'].min()), int(df['vertex_y_p_trans'].max())],
    }


# ── Block-based refinement ─────────────────────────────────────────────────

def run_block_refinement(dataset_id: str, ref_id: str, warped_id: str,
                          num_blocks: int = 32, min_contours: int = 20,
                          method: str = 'akaze') -> dict:
    """Per-block feature refinement. Agent can adjust num_blocks and method."""
    data = _contour_store[dataset_id]
    df = data['df']
    grouped_df = data['grouped_df']
    ref = get_image(ref_id)
    warped = get_image(warped_id)

    # Group contours into grid blocks
    x_min, y_min = df[['vertex_x_p_trans', 'vertex_y_p_trans']].min()
    x_max, y_max = df[['vertex_x_p_trans', 'vertex_y_p_trans']].max()
    x_min, y_min = x_min - 1, y_min - 1
    x_max, y_max = x_max + 1, y_max + 1
    grid_size = [(x_max - x_min) / num_blocks, (y_max - y_min) / num_blocks]

    grid_contours = {}
    for label, group in tqdm(grouped_df, desc='grouping'):
        contour = group[['vertex_x_p_trans', 'vertex_y_p_trans']].values
        center = np.mean(contour, axis=0).astype(int)
        gx = int((center[0] - x_min) / grid_size[0])
        gy = int((center[1] - y_min) / grid_size[1])
        grid_contours.setdefault((gx, gy), []).append((label, contour))

    # Merge small blocks
    final = {k: v[:] for k, v in grid_contours.items()}
    for gk, items in list(grid_contours.items()):
        if len(items) >= min_contours:
            continue
        for item in items:
            center = np.mean(item[1], axis=0)
            best_key, best_dist = None, float('inf')
            for dx, dy in [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]:
                nk = (gk[0]+dx, gk[1]+dy)
                if nk in final and len(final[nk]) >= min_contours:
                    nc = np.array([nk[0]*grid_size[0]+grid_size[0]/2+x_min,
                                   nk[1]*grid_size[1]+grid_size[1]/2+y_min])
                    d = np.linalg.norm(center - nc)
                    if d < best_dist:
                        best_dist, best_key = d, nk
            if best_key is not None:
                final[best_key].append(item)
        if gk in final:
            del final[gk]

    # Per-block alignment
    all_transformed = {}
    failed_blocks = 0
    total_blocks = len(final)

    for key, grouped in tqdm(final.items(), desc='block align'):
        tmp = np.vstack([c[1] for c in grouped])
        xmin_b, ymin_b = tmp.min(axis=0) - 2
        xmax_b, ymax_b = tmp.max(axis=0) + 2
        xmin_b = max(0, xmin_b)
        ymin_b = max(0, ymin_b)
        xmax_b = min(ref.shape[1], xmax_b)
        ymax_b = min(ref.shape[0], ymax_b)

        block1 = ref[ymin_b:ymax_b, xmin_b:xmax_b]
        block2 = warped[ymin_b:ymax_b, xmin_b:xmax_b]

        if block1.size == 0 or block2.size == 0:
            failed_blocks += 1
            for label_id, contour in grouped:
                cell_id = grouped_df.get_group(label_id)['cell_id'].iloc[0]
                all_transformed[label_id] = {'cell_id': cell_id, 'contour': contour}
            continue

        try:
            if method == 'akaze':
                akaze = cv2.AKAZE_create()
                kp1, des1 = akaze.detectAndCompute(block1, None)
                kp2, des2 = akaze.detectAndCompute(block2, None)
            elif method == 'orb':
                orb = cv2.ORB_create(nfeatures=5000)
                kp1, des1 = orb.detectAndCompute(block1, None)
                kp2, des2 = orb.detectAndCompute(block2, None)
            else:
                raise ValueError(f"Unknown method: {method}")

            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                raise RuntimeError("Not enough features")

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
            if len(matches) < 4:
                raise RuntimeError("Not enough matches")

            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M_small, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts)
            if M_small is None:
                raise RuntimeError("Affine estimation failed")

            theta = np.degrees(np.arctan2(M_small[1, 0], M_small[0, 0]))
            if abs(theta) > 5:
                # Try upright descriptor
                akaze_up = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT)
                kp1u, des1u = akaze_up.detectAndCompute(block1, None)
                kp2u, des2u = akaze_up.detectAndCompute(block2, None)
                if des1u is not None and des2u is not None and len(kp1u) >= 4 and len(kp2u) >= 4:
                    matches_u = sorted(bf.match(des1u, des2u), key=lambda x: x.distance)
                    if len(matches_u) >= 4:
                        src_u = np.float32([kp1u[m.queryIdx].pt for m in matches_u]).reshape(-1, 1, 2)
                        dst_u = np.float32([kp2u[m.trainIdx].pt for m in matches_u]).reshape(-1, 1, 2)
                        M_small, _ = cv2.estimateAffinePartial2D(dst_u, src_u)
                        if M_small is not None:
                            theta = np.degrees(np.arctan2(M_small[1, 0], M_small[0, 0]))
                if M_small is None or abs(theta) > 5:
                    raise RuntimeError(f"Block rotation too large: {theta:.1f}°")

        except Exception:
            failed_blocks += 1
            for label_id, contour in grouped:
                cell_id = grouped_df.get_group(label_id)['cell_id'].iloc[0]
                all_transformed[label_id] = {'cell_id': cell_id, 'contour': contour}
            continue

        offset = np.array([xmin_b, ymin_b])
        for label_id, contour in grouped:
            local = contour - offset
            hom = np.hstack([local, np.ones((local.shape[0], 1))])
            t_local = (M_small @ hom.T).T[:, :2]
            t_global = np.round(t_local + offset).astype(int)
            cell_id = grouped_df.get_group(label_id)['cell_id'].iloc[0]
            all_transformed[label_id] = {'cell_id': cell_id, 'contour': t_global}

    # Store result
    result_id = f"{dataset_id}__refined"
    _contour_store[result_id] = all_transformed

    return {
        'result_id': result_id,
        'total_blocks': total_blocks,
        'failed_blocks': failed_blocks,
        'n_contours': len(all_transformed),
        'method': method,
        'num_blocks': num_blocks,
    }


# ── Quality assessment ─────────────────────────────────────────────────────

def assess_xenium_quality(dataset_id: str, ref_binary_id: str,
                           ref_gray_id: str = None,
                           output_dir: str = None) -> dict:
    """Compute alignment quality: Dice, IoU, NCC, coverage.
    Can be called after global alignment or after block refinement."""
    ref_binary = get_image(ref_binary_id)

    # Render current contours as mask
    refined_key = f"{dataset_id}__refined"
    if refined_key in _contour_store:
        all_contours = _contour_store[refined_key]
        contour_mask = np.zeros_like(ref_binary)
        for label_id, data in all_contours.items():
            cv2.fillPoly(contour_mask, [data['contour']], color=255)
        stage = 'refined'
    else:
        # Use globally warped mask
        warped_id = f"{dataset_id}__contour_mask__warped"
        if warped_id in _image_store:
            contour_mask = get_image(warped_id)
            stage = 'global'
        else:
            return {'error': 'No warped contours found. Run alignment first.'}

    # Dice
    intersection = np.sum((ref_binary > 0) & (contour_mask > 0))
    union_sum = np.sum(ref_binary > 0) + np.sum(contour_mask > 0)
    dice = 2.0 * intersection / (union_sum + 1e-10)

    # IoU
    union_area = np.sum((ref_binary > 0) | (contour_mask > 0))
    iou = intersection / (union_area + 1e-10)

    # Coverage
    he_tissue = np.sum(ref_binary > 0)
    coverage = intersection / (he_tissue + 1e-10)

    # NCC (if gray reference available)
    ncc = 0.0
    if ref_gray_id and ref_gray_id in _image_store:
        ref_gray = get_image(ref_gray_id)
        mask = (ref_binary > 0) & (contour_mask > 0)
        if mask.sum() > 100:
            a = ref_gray[mask].astype(np.float64)
            b = contour_mask[mask].astype(np.float64)
            a -= a.mean()
            b -= b.mean()
            denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
            ncc = float(np.sum(a * b) / (denom + 1e-10))

    result = {
        'stage': stage,
        'dice': round(float(dice), 4),
        'iou': round(float(iou), 4),
        'ncc': round(float(ncc), 4),
        'coverage': round(float(coverage), 4),
    }

    # Save overlay if output_dir given
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        he_rgb_id = f"{dataset_id}__he_rgb"
        if he_rgb_id in _image_store:
            he_rgb = _image_store[he_rgb_id]
            # Downscale overlay
            ds = max(1, max(he_rgb.shape[:2]) // 4000)
            small_he = cv2.resize(he_rgb, (he_rgb.shape[1] // ds, he_rgb.shape[0] // ds))
            small_mask = cv2.resize(contour_mask, (contour_mask.shape[1] // ds, contour_mask.shape[0] // ds))

            overlay = small_he.copy()
            overlay[small_mask > 0] = (overlay[small_mask > 0] * 0.6 + np.array([0, 255, 255]) * 0.4).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f'overlay_{stage}.png'), overlay)
            result['overlay_path'] = os.path.join(output_dir, f'overlay_{stage}.png')

        # Save result.json
        with open(os.path.join(output_dir, 'quality.json'), 'w') as f:
            json.dump(result, f, indent=2)

    return result


def save_outputs(dataset_id: str, output_dir: str) -> dict:
    """Save final transformed contours and affine matrix."""
    os.makedirs(output_dir, exist_ok=True)

    refined_key = f"{dataset_id}__refined"
    if refined_key in _contour_store:
        import pickle
        contours = _contour_store[refined_key]
        with open(os.path.join(output_dir, 'contours.pkl'), 'wb') as f:
            pickle.dump(contours, f)

    # Save all transforms
    for tid, M in _transform_store.items():
        if dataset_id in tid:
            np.save(os.path.join(output_dir, f'{tid}.npy'), M)

    return {
        'output_dir': output_dir,
        'n_contours': len(_contour_store.get(refined_key, {})),
    }
