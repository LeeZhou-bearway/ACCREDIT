"""
quality.py - NMI computation + checkerboard overlay visualization.
"""
import numpy as np
import SimpleITK as sitk
import cv2
import os

from registration.preprocess import get_image, get_meta
from registration.global_reg import get_transform, _numpy_to_sitk


def compute_feature_metrics(
    fixed_id: str,
    moving_id: str,
    transform_id: str,
) -> dict:
    """Compute three feature-based similarity metrics after registration.

    All computed on H-channel (hematoxylin) within the tissue mask:
      h_ncc       : global Pearson NCC on tissue pixels [-1, 1]
      akaze_inliers: AKAZE RANSAC inlier count
      edge_dice   : Canny edge map Dice coefficient [0, 1]

    Returns dict with keys h_ncc, akaze_inliers, edge_dice.
    """
    from registration.preprocess import _image_store
    from skimage.color import rgb2hed

    def _get_h(image_id: str) -> np.ndarray:
        """Return H-channel: reuse cached __H if available, else extract."""
        cached = _image_store.get(image_id + "__H")
        if cached is not None:
            return cached.astype(np.uint8) if cached.dtype != np.uint8 else cached
        arr = get_image(image_id)
        if arr.ndim == 3:
            hed = rgb2hed(arr.astype(np.float64) / 255.0)
            h = hed[:, :, 0]
            p_lo, p_hi = float(np.percentile(h, 1)), float(np.percentile(h, 99))
            h = np.clip((h - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0)
            return (h * 255).astype(np.uint8)
        return arr

    transform = get_transform(transform_id)

    fixed_h = _get_h(fixed_id)
    moving_h = _get_h(moving_id)

    # Warp moving H-channel into fixed space
    f_sitk = _numpy_to_sitk(fixed_h.astype(np.float32), fixed_id)
    m_sitk = _numpy_to_sitk(moving_h.astype(np.float32), moving_id)
    warped_sitk = sitk.Resample(m_sitk, f_sitk, transform,
                                sitk.sitkLinear, 0.0, m_sitk.GetPixelID())
    warped_h = np.clip(sitk.GetArrayFromImage(warped_sitk), 0, 255).astype(np.uint8)

    # Tissue mask from fixed H-channel (non-background region)
    _, tissue = cv2.threshold(fixed_h, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, close_k)

    # 1. H-channel global NCC (Pearson) on tissue pixels
    f_px = fixed_h[tissue > 0].astype(np.float32)
    m_px = warped_h[tissue > 0].astype(np.float32)
    h_ncc = 0.0
    if len(f_px) > 100:
        f_z = f_px - f_px.mean()
        m_z = m_px - m_px.mean()
        denom = float(np.sqrt(np.sum(f_z ** 2) * np.sum(m_z ** 2)))
        if denom > 1e-10:
            h_ncc = float(np.sum(f_z * m_z) / denom)

    # 2. AKAZE inlier count (feature match after registration)
    akaze = cv2.AKAZE_create()
    kp1, d1 = akaze.detectAndCompute(fixed_h, tissue)
    kp2, d2 = akaze.detectAndCompute(warped_h, tissue)
    akaze_inliers = 0
    if (d1 is not None and d2 is not None
            and len(kp1) >= 10 and len(kp2) >= 10):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d1, d2)
        if len(matches) >= 10:
            src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if mask is not None:
                akaze_inliers = int(mask.sum())

    # 3. Canny edge Dice coefficient (tissue region only)
    fixed_edges = cv2.bitwise_and(cv2.Canny(fixed_h, 50, 150), tissue)
    warped_edges = cv2.bitwise_and(cv2.Canny(warped_h, 50, 150), tissue)
    intersection = int(np.logical_and(fixed_edges > 0, warped_edges > 0).sum())
    total = int((fixed_edges > 0).sum()) + int((warped_edges > 0).sum())
    edge_dice = float(2 * intersection / total) if total > 0 else 0.0

    return {
        'h_ncc': round(h_ncc, 4),
        'akaze_inliers': akaze_inliers,
        'edge_dice': round(edge_dice, 4),
    }


def compute_nmi(fixed_id: str, moving_id: str, transform_id: str) -> float:
    """
    Resample moving onto fixed using transform, compute NMI.
    Uses joint histogram entropy: NMI = (H(f) + H(m)) / H(f,m).
    """
    fixed_arr = get_image(fixed_id)
    moving_arr = get_image(moving_id)

    def to_gray(arr):
        if arr.ndim == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return arr

    fixed_gray = to_gray(fixed_arr).astype(np.float32)
    moving_gray = to_gray(moving_arr).astype(np.float32)

    fixed_sitk = _numpy_to_sitk(fixed_gray, fixed_id)
    moving_sitk = _numpy_to_sitk(moving_gray, moving_id)

    transform = get_transform(transform_id)

    resampled = sitk.Resample(
        moving_sitk, fixed_sitk, transform, sitk.sitkLinear, 0.0,
        moving_sitk.GetPixelID()
    )
    resampled_arr = sitk.GetArrayFromImage(resampled).astype(np.float32)

    nmi = _normalized_mutual_information(fixed_gray, resampled_arr)
    return float(nmi)


def _normalized_mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Compute NMI from two grayscale float arrays."""
    # Flatten and ignore background (zero) pixels
    mask = (a > 0) & (b > 0)
    if mask.sum() < 100:
        return 1.0

    a_flat = a[mask].ravel()
    b_flat = b[mask].ravel()

    # Joint histogram
    hist_2d, _, _ = np.histogram2d(a_flat, b_flat, bins=bins)
    hist_2d = hist_2d / (hist_2d.sum() + 1e-10)

    # Marginals
    p_a = hist_2d.sum(axis=1)
    p_b = hist_2d.sum(axis=0)

    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p + 1e-10))

    H_a = entropy(p_a)
    H_b = entropy(p_b)
    H_ab = entropy(hist_2d.ravel())

    if H_ab < 1e-10:
        return 0.0
    nmi = (H_a + H_b) / H_ab
    return float(nmi)


def _warp_image(moving_arr: np.ndarray, fixed_arr: np.ndarray,
                transform, moving_id: str, fixed_id: str,
                fill: float = 255.0) -> np.ndarray:
    """Warp moving array into fixed space using the given transform."""
    def to_gray(a):
        return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a

    fixed_g = to_gray(fixed_arr).astype(np.float32)
    moving_g = to_gray(moving_arr).astype(np.float32)

    f_sitk = _numpy_to_sitk(fixed_g, fixed_id)
    m_sitk = _numpy_to_sitk(moving_g, moving_id)

    warped = sitk.Resample(m_sitk, f_sitk, transform,
                           sitk.sitkLinear, fill, m_sitk.GetPixelID())
    return sitk.GetArrayFromImage(warped).astype(np.float32)


def _warp_rgb(moving_rgb: np.ndarray, fixed_arr: np.ndarray,
              transform, moving_id: str, fixed_id: str) -> np.ndarray:
    """Warp each channel of an RGB image into fixed space."""
    def to_gray(a):
        return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a

    fixed_g = to_gray(fixed_arr).astype(np.float32)
    f_sitk = _numpy_to_sitk(fixed_g, fixed_id)
    H, W = fixed_g.shape
    result = np.zeros((H, W, 3), dtype=np.uint8)

    for c in range(3):
        chan = moving_rgb[:, :, c].astype(np.float32)
        m_sitk = _numpy_to_sitk(chan, moving_id)
        warped = sitk.Resample(m_sitk, f_sitk, transform,
                               sitk.sitkLinear, 255.0, m_sitk.GetPixelID())
        result[:, :, c] = np.clip(sitk.GetArrayFromImage(warped), 0, 255).astype(np.uint8)
    return result


def _ihc_to_red(ihc_gray: np.ndarray) -> np.ndarray:
    """
    Re-color a grayscale IHC image so tissue appears red and background stays white.

    Convention (original IHC grayscale): background ≈ 255 (bright), tissue < 200.
    Output BGR: background → (255,255,255), tissue → (low, low, 255) i.e. red.
    """
    g = ihc_gray.astype(np.float32)
    # Clip to tissue range: stretch contrast using 1st-99th percentile
    p_lo = float(np.percentile(g, 1))
    p_hi = float(np.percentile(g, 99))
    g_norm = np.clip((g - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0)

    # g_norm ≈ 1 → background (white), g_norm ≈ 0 → dense tissue (red)
    R = np.ones_like(g_norm) * 255
    G = g_norm * 255
    B = g_norm * 255
    # Stack as BGR for OpenCV
    return np.stack([B, G, R], axis=-1).astype(np.uint8)


def _add_label(img: np.ndarray, text: str, color_bgr=(30, 30, 30)) -> np.ndarray:
    """Burn a text label into the top-left corner of an image (in-place copy)."""
    out = img.copy()
    H = out.shape[0]
    scale = max(0.6, H / 2000.0)
    thick = max(1, int(H / 1200))
    cv2.putText(out, text, (12, int(40 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                thick + 2, cv2.LINE_AA)
    cv2.putText(out, text, (12, int(40 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr,
                thick, cv2.LINE_AA)
    return out


def generate_checkerboard_overlay(
    fixed_id: str,
    moving_id: str,
    transform_id: str,
    output_path: str,
    grid_size: int = 50,
) -> str:
    """Checkerboard overlay: alternating tiles from HE and warped IHC (grayscale)."""
    fixed_arr = get_image(fixed_id)
    moving_arr = get_image(moving_id)
    transform = get_transform(transform_id)

    warped = _warp_image(moving_arr, fixed_arr, transform, moving_id, fixed_id)

    def norm255(img):
        mn, mx = img.min(), img.max()
        return ((img - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8) if mx > mn else np.zeros_like(img, dtype=np.uint8)

    def to_gray(a):
        return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a

    fixed_u8 = norm255(to_gray(fixed_arr).astype(np.float32))
    warped_u8 = norm255(warped)

    H, W = fixed_u8.shape
    checkerboard = np.zeros((H, W), dtype=np.uint8)
    for gy in range(0, H, grid_size):
        for gx in range(0, W, grid_size):
            y1 = min(gy + grid_size, H)
            x1 = min(gx + grid_size, W)
            if ((gy // grid_size) + (gx // grid_size)) % 2 == 0:
                checkerboard[gy:y1, gx:x1] = fixed_u8[gy:y1, gx:x1]
            else:
                checkerboard[gy:y1, gx:x1] = warped_u8[gy:y1, gx:x1]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, checkerboard)
    return output_path


def generate_color_comparison(
    fixed_id: str,
    moving_id: str,
    transform_id: str,
    output_dir: str = "output",
    max_side: int = 2048,
) -> dict:
    """
    Generate two colored comparison images:

    1. color_overlay_<id>.png  – HE in grayscale + IHC aligned in red, 50% blend.
    2. comparison_<id>.png     – side-by-side: [HE] | [IHC-red] | [overlay].

    IHC coloring: tissue → red, background (white) → white.
    Uses original RGB images when available (strips __H/__gray suffix).
    Returns dict with keys 'color_overlay_path' and 'side_by_side_path'.
    """
    transform = get_transform(transform_id)

    # ---- Retrieve images ----
    # Try to use original RGB (strip channel suffix)
    def base_id(iid):
        for suf in ("__H", "__gray"):
            if iid.endswith(suf):
                return iid[: -len(suf)]
        return iid

    from registration.preprocess import _image_store  # noqa: F401

    he_base = base_id(fixed_id)
    ihc_base = base_id(moving_id)

    he_rgb = _image_store.get(he_base)   # may be None
    ihc_rgb = _image_store.get(ihc_base)

    # Fallback to stored (possibly grayscale) image
    he_display = he_rgb if he_rgb is not None else get_image(fixed_id)
    ihc_src = ihc_rgb if ihc_rgb is not None else get_image(moving_id)

    # ---- HE panel (original RGB → BGR for OpenCV) ----
    if he_display.ndim == 3:
        he_bgr = cv2.cvtColor(he_display, cv2.COLOR_RGB2BGR)
    else:
        he_bgr = cv2.cvtColor(he_display, cv2.COLOR_GRAY2BGR)

    # ---- Warp IHC to HE space ----
    if ihc_src.ndim == 3:
        warped_ihc_rgb = _warp_rgb(ihc_src, he_display, transform, ihc_base, he_base)
        # Convert to grayscale for red-coloring (tissue=dark, bg=white)
        warped_ihc_gray = cv2.cvtColor(warped_ihc_rgb, cv2.COLOR_RGB2GRAY)
    else:
        warped_ihc_gray_f = _warp_image(ihc_src, he_display, transform,
                                        moving_id, fixed_id, fill=255.0)
        warped_ihc_gray = np.clip(warped_ihc_gray_f, 0, 255).astype(np.uint8)

    # ---- IHC panel: red coloring (tissue=red, background=white) ----
    ihc_red_bgr = _ihc_to_red(warped_ihc_gray)

    # ---- Apply HE tissue mask: IHC red only shown within HE tissue boundary ----
    he_gray_u8 = cv2.cvtColor(he_bgr, cv2.COLOR_BGR2GRAY)
    _, tissue_mask = cv2.threshold(he_gray_u8, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
    ihc_red_bgr[tissue_mask == 0] = 255  # non-tissue areas → white

    # ---- Overlay panel: HE grayscale + IHC red, 50/50 blend ----
    he_gray_bgr = cv2.cvtColor(cv2.cvtColor(he_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    overlay_bgr = cv2.addWeighted(he_gray_bgr, 0.5, ihc_red_bgr, 0.5, 0)

    # ---- Scale panels to max_side ----
    H, W = he_bgr.shape[:2]
    if max(H, W) > max_side:
        scale = max_side / max(H, W)
        new_w, new_h = int(W * scale), int(H * scale)
        he_bgr = cv2.resize(he_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ihc_red_bgr = cv2.resize(ihc_red_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        overlay_bgr = cv2.resize(overlay_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ---- Add labels ----
    he_labeled = _add_label(he_bgr, "HE (reference)", (30, 30, 30))
    ihc_labeled = _add_label(ihc_red_bgr, "IHC P40 aligned", (0, 0, 200))
    overlay_labeled = _add_label(overlay_bgr, "Overlay (HE gray + IHC red)", (0, 60, 180))

    # ---- Save outputs ----
    os.makedirs(output_dir, exist_ok=True)
    safe_id = transform_id.replace("/", "_").replace(".", "_")[:55]

    color_overlay_path = os.path.join(output_dir, f"color_overlay_{safe_id}.png")
    side_by_side_path = os.path.join(output_dir, f"comparison_{safe_id}.png")

    cv2.imwrite(color_overlay_path, overlay_bgr)

    # Add a thin white separator between panels
    sep = np.full((he_labeled.shape[0], 4, 3), 200, dtype=np.uint8)
    side_by_side = np.concatenate([he_labeled, sep, ihc_labeled, sep, overlay_labeled], axis=1)
    cv2.imwrite(side_by_side_path, side_by_side)

    return {
        "color_overlay_path": color_overlay_path,
        "side_by_side_path": side_by_side_path,
    }


def assess_quality(
    fixed_id: str,
    moving_id: str,
    transform_id: str,
    output_dir: str = "output",
) -> dict:
    """
    Full quality assessment: compute NMI + generate all overlay visualizations.
    Returns a dict with nmi, overlay_path, color_overlay_path, side_by_side_path.
    """
    nmi = compute_nmi(fixed_id, moving_id, transform_id)

    os.makedirs(output_dir, exist_ok=True)
    safe_id = transform_id.replace("/", "_").replace(".", "_")[:60]
    overlay_path = os.path.join(output_dir, f"overlay_{safe_id}.png")
    generate_checkerboard_overlay(fixed_id, moving_id, transform_id, overlay_path)

    color_paths = generate_color_comparison(fixed_id, moving_id, transform_id, output_dir)

    # Proper NMI = (H_f + H_m) / H_joint, range [1.0, ~1.5+]
    # Thresholds tuned for HE-IHC P40 cross-modal adjacent-section registration.
    thresholds = {
        "nonrigid": 1.07,
        "affine": 1.06,
        "similarity": 1.05,
        "rigid": 1.05,
    }

    stage = "rigid"
    # Check most specific first (nonrigid before rigid to avoid substring match)
    for key in ("nonrigid", "affine", "similarity", "rigid"):
        if key in transform_id:
            stage = key
            break

    threshold = thresholds.get(stage, 0.4)
    passed = nmi >= threshold

    report = {
        "nmi": nmi,
        "threshold": threshold,
        "passed": passed,
        "stage": stage,
        "overlay_path": overlay_path,
        "color_overlay_path": color_paths["color_overlay_path"],
        "side_by_side_path": color_paths["side_by_side_path"],
        "transform_id": transform_id,
    }
    return report
