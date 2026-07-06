"""
nonrigid_reg.py - Stage 2/3: BSpline and feature-based non-rigid registration.
"""
import math
import numpy as np
import SimpleITK as sitk
import cv2

from registration.preprocess import get_image, store_image
from registration.global_reg import get_transform, _transform_store, _compute_proper_nmi, _numpy_to_sitk


def run_nonrigid_registration(
    fixed_id: str,
    moving_id: str,
    init_transform_id: str,
    mesh_size: int = 8,
    num_iterations: int = 100,
) -> tuple[str, float]:
    """
    Stage 3: BSpline non-rigid registration at medium resolution.
    Applies init_transform first, then runs BSpline on the pre-aligned image.
    Returns (transform_id, nmi_score).
    """
    fixed_arr = get_image(fixed_id)
    moving_arr = get_image(moving_id)

    def to_gray(arr):
        if arr.ndim == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return arr

    fixed_gray = to_gray(fixed_arr)
    moving_gray = to_gray(moving_arr)

    fixed_sitk = _numpy_to_sitk(fixed_gray, fixed_id)
    moving_sitk = _numpy_to_sitk(moving_gray, moving_id)

    init_transform = get_transform(init_transform_id)
    pre_aligned = sitk.Resample(
        moving_sitk, fixed_sitk, init_transform, sitk.sitkLinear, 0.0,
        moving_sitk.GetPixelID()
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(0.15, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)

    transform_domain_mesh_size = [mesh_size] * fixed_sitk.GetDimension()
    bspline_transform = sitk.BSplineTransformInitializer(
        fixed_sitk, transform_domain_mesh_size
    )

    registration.SetInitialTransform(bspline_transform, inPlace=False)
    registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=num_iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=500,
        costFunctionConvergenceFactor=1e7,
    )
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    bspline_result = registration.Execute(fixed_sitk, pre_aligned)

    composite = sitk.CompositeTransform(2)
    composite.AddTransform(bspline_result)
    composite.AddTransform(init_transform)

    nmi = _compute_proper_nmi(fixed_gray, moving_gray, composite,
                               fixed_id=fixed_id, moving_id=moving_id)

    transform_id = f"nonrigid_{fixed_id}__{moving_id}"
    _transform_store[transform_id] = composite
    return transform_id, float(nmi)


def run_boundary_correction(
    fixed_id: str,
    moving_id: str,
    init_transform_id: str,
    boundary_width_px: int = 60,
    mesh_size: int = 8,
    num_iterations: int = 200,
) -> tuple[str, float]:
    """
    Two-pass strategy: H-channel Similarity gives good interior alignment;
    this function adds a grayscale BSpline correction focused on the tissue
    boundary zone (ring of `boundary_width_px` around the tissue edge).
    """
    from registration.preprocess import get_image, get_meta
    fixed_arr = get_image(fixed_id)
    moving_arr = get_image(moving_id)

    def to_gray(arr):
        if arr.ndim == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return arr

    fixed_gray = to_gray(fixed_arr).astype(np.float32)
    moving_gray = to_gray(moving_arr).astype(np.float32)

    fixed_u8 = np.clip(fixed_gray, 0, 255).astype(np.uint8)
    _, tissue_mask = cv2.threshold(fixed_u8, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, close_k)

    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (boundary_width_px * 2 + 1,
                                         boundary_width_px * 2 + 1))
    interior_mask = cv2.erode(tissue_mask, erode_k)
    boundary_mask = cv2.subtract(tissue_mask, interior_mask)

    fixed_sitk  = _numpy_to_sitk(fixed_gray,  fixed_id)
    moving_sitk = _numpy_to_sitk(moving_gray, moving_id)

    init_transform = get_transform(init_transform_id)
    pre_aligned = sitk.Resample(
        moving_sitk, fixed_sitk, init_transform,
        sitk.sitkLinear, 0.0, moving_sitk.GetPixelID()
    )

    mask_sitk = sitk.GetImageFromArray(boundary_mask)
    mask_sitk.CopyInformation(fixed_sitk)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(0.5, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetMetricFixedMask(mask_sitk)

    bspline_transform = sitk.BSplineTransformInitializer(
        fixed_sitk, [mesh_size] * 2
    )
    registration.SetInitialTransform(bspline_transform, inPlace=False)
    registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=num_iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7,
    )
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    bspline_result = registration.Execute(fixed_sitk, pre_aligned)

    composite = sitk.CompositeTransform(2)
    composite.AddTransform(bspline_result)
    composite.AddTransform(init_transform)

    nmi = _compute_proper_nmi(fixed_gray, moving_gray, composite,
                               fixed_id=fixed_id, moving_id=moving_id)

    transform_id = f"boundary_corrected_{fixed_id}__{moving_id}"
    _transform_store[transform_id] = composite
    return transform_id, float(nmi)


def run_masked_nonrigid_registration(
    fixed_id: str,
    moving_id: str,
    init_transform_id: str,
    mask_right_fraction: float = 0.4,
    mesh_size: int = 4,
    num_iterations: int = 150,
) -> tuple[str, float]:
    """BSpline non-rigid registration restricted to a right-side mask region."""
    fixed_arr = get_image(fixed_id)
    moving_arr = get_image(moving_id)

    def to_gray(arr):
        if arr.ndim == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return arr

    fixed_gray = to_gray(fixed_arr)
    moving_gray = to_gray(moving_arr)

    fixed_sitk = _numpy_to_sitk(fixed_gray, fixed_id)
    moving_sitk = _numpy_to_sitk(moving_gray, moving_id)

    init_transform = get_transform(init_transform_id)
    pre_aligned = sitk.Resample(
        moving_sitk, fixed_sitk, init_transform, sitk.sitkLinear, 0.0,
        moving_sitk.GetPixelID()
    )

    H, W = fixed_gray.shape
    x_start = int(W * (1.0 - mask_right_fraction))
    mask_arr = np.zeros((H, W), dtype=np.uint8)
    mask_arr[:, x_start:] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mask_arr = cv2.dilate(mask_arr, kernel)
    mask_sitk = sitk.GetImageFromArray(mask_arr)
    mask_sitk.CopyInformation(fixed_sitk)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(0.3, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetMetricFixedMask(mask_sitk)

    bspline_transform = sitk.BSplineTransformInitializer(
        fixed_sitk, [mesh_size] * 2
    )
    registration.SetInitialTransform(bspline_transform, inPlace=False)
    registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=num_iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=500,
        costFunctionConvergenceFactor=1e7,
    )
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    bspline_result = registration.Execute(fixed_sitk, pre_aligned)

    composite = sitk.CompositeTransform(2)
    composite.AddTransform(bspline_result)
    composite.AddTransform(init_transform)

    nmi = _compute_proper_nmi(fixed_gray, moving_gray, composite,
                               fixed_id=fixed_id, moving_id=moving_id)

    transform_id = f"nonrigid_masked_{fixed_id}__{moving_id}"
    _transform_store[transform_id] = composite
    return transform_id, float(nmi)


def run_tile_based_nonrigid(
    fixed_id: str,
    moving_id: str,
    init_transform_id: str,
    mesh_size: int = 8,
    tile_size: int = 2048,
    overlap: int = 256,
) -> tuple[str, float]:
    """Tile-based BSpline non-rigid registration for large images."""
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

    init_transform = get_transform(init_transform_id)
    pre_aligned = sitk.Resample(
        moving_sitk, fixed_sitk, init_transform, sitk.sitkLinear, 0.0,
        moving_sitk.GetPixelID()
    )
    pre_aligned_arr = sitk.GetArrayFromImage(pre_aligned)

    H, W = fixed_gray.shape
    disp_x = np.zeros((H, W), dtype=np.float32)
    disp_y = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    y_starts = list(range(0, H, tile_size - overlap))
    x_starts = list(range(0, W, tile_size - overlap))

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)

            fixed_tile = fixed_gray[y0:y1, x0:x1]
            moving_tile = pre_aligned_arr[y0:y1, x0:x1]

            if fixed_tile.shape[0] < 32 or fixed_tile.shape[1] < 32:
                continue

            sx, sy = fixed_sitk.GetSpacing()
            fixed_tile_sitk = sitk.GetImageFromArray(fixed_tile)
            fixed_tile_sitk.SetSpacing([sx, sy])
            fixed_tile_sitk.SetOrigin([x0 * sx, y0 * sy])
            moving_tile_sitk = sitk.GetImageFromArray(moving_tile)
            moving_tile_sitk.SetSpacing([sx, sy])
            moving_tile_sitk.SetOrigin([x0 * sx, y0 * sy])

            try:
                reg = sitk.ImageRegistrationMethod()
                reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
                reg.SetMetricSamplingStrategy(reg.REGULAR)
                reg.SetMetricSamplingPercentage(0.2, seed=42)
                reg.SetInterpolator(sitk.sitkLinear)

                mesh = [max(2, mesh_size // 2)] * 2
                bspline = sitk.BSplineTransformInitializer(fixed_tile_sitk, mesh)
                reg.SetInitialTransform(bspline, inPlace=False)
                reg.SetOptimizerAsLBFGSB(
                    gradientConvergenceTolerance=1e-5,
                    numberOfIterations=50,
                    maximumNumberOfCorrections=5,
                    maximumNumberOfFunctionEvaluations=200,
                    costFunctionConvergenceFactor=1e7,
                )
                tile_transform = reg.Execute(fixed_tile_sitk, moving_tile_sitk)

                disp_filter = sitk.TransformToDisplacementFieldFilter()
                disp_filter.SetReferenceImage(fixed_tile_sitk)
                disp_field = disp_filter.Execute(tile_transform)
                disp_arr = sitk.GetArrayFromImage(disp_field)

                tile_h = y1 - y0
                tile_w = x1 - x0
                wy = np.hanning(tile_h).reshape(-1, 1)
                wx = np.hanning(tile_w).reshape(1, -1)
                w = wy * wx

                disp_x[y0:y1, x0:x1] += disp_arr[:, :, 0] * w
                disp_y[y0:y1, x0:x1] += disp_arr[:, :, 1] * w
                weight_map[y0:y1, x0:x1] += w

            except Exception:
                pass

    valid = weight_map > 0
    disp_x[valid] /= weight_map[valid]
    disp_y[valid] /= weight_map[valid]

    disp_combined = np.stack([disp_x, disp_y], axis=-1)
    disp_sitk = sitk.GetImageFromArray(disp_combined.astype(np.float64), isVector=True)
    disp_sitk.SetOrigin(fixed_sitk.GetOrigin())
    disp_sitk.SetSpacing(fixed_sitk.GetSpacing())

    # Combine with init_transform via composite
    disp_transform = sitk.DisplacementFieldTransform(disp_sitk)

    composite = sitk.CompositeTransform(2)
    composite.AddTransform(disp_transform)
    composite.AddTransform(init_transform)

    nmi = _compute_proper_nmi(fixed_gray, moving_gray, composite,
                               fixed_id=fixed_id, moving_id=moving_id)

    transform_id = f"nonrigid_{fixed_id}__{moving_id}"
    _transform_store[transform_id] = composite
    return transform_id, float(nmi)


# ---------------------------------------------------------------------------
# Stage 3: Tile-based MI translation registration (H-channel, cross-modal)
# ---------------------------------------------------------------------------

def _tile_ncc(tile1: np.ndarray, tile2: np.ndarray) -> float:
    """Normalized cross-correlation between two tiles. Range [-1, 1]."""
    t1 = tile1.astype(np.float32) - tile1.mean()
    t2 = tile2.astype(np.float32) - tile2.mean()
    denom = np.sqrt(np.sum(t1 ** 2) * np.sum(t2 ** 2))
    if denom < 1e-10:
        return 0.0
    return float(np.sum(t1 * t2) / denom)


def _mi_translate_tile(
    tile_fixed: np.ndarray, tile_moving: np.ndarray,
    sx: float, sy: float, y0: int, x0: int,
    max_disp_px: float = 30.0,
    num_iterations: int = 50,
) -> "tuple[float, float] | None":
    """MI-based translation-only registration for a single tile (H-channel).

    Returns (dx_mm, dy_mm) in fixed→moving convention, or None if failed/diverged.
    """
    f_sitk = sitk.GetImageFromArray(tile_fixed.astype(np.float32))
    f_sitk.SetSpacing([sx, sy])
    f_sitk.SetOrigin([x0 * sx, y0 * sy])

    m_sitk = sitk.GetImageFromArray(tile_moving.astype(np.float32))
    m_sitk.SetSpacing([sx, sy])
    m_sitk.SetOrigin([x0 * sx, y0 * sy])

    try:
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        reg.SetMetricSamplingStrategy(reg.REGULAR)
        reg.SetMetricSamplingPercentage(0.5, seed=42)
        reg.SetInterpolator(sitk.sitkLinear)

        tx = sitk.TranslationTransform(2)
        reg.SetInitialTransform(tx, inPlace=False)
        reg.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-4,
            numberOfIterations=num_iterations,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=100,
            costFunctionConvergenceFactor=1e7,
        )
        reg.SetShrinkFactorsPerLevel([1])
        reg.SetSmoothingSigmasPerLevel([0])

        result = reg.Execute(f_sitk, m_sitk)
        dx_mm, dy_mm = float(result.GetParameters()[0]), float(result.GetParameters()[1])

        max_disp_mm = max_disp_px * max(sx, sy)
        if abs(dx_mm) > max_disp_mm or abs(dy_mm) > max_disp_mm:
            return None

        return dx_mm, dy_mm
    except Exception:
        return None


def _akaze_align_tile(
    img1: np.ndarray, img2: np.ndarray,
    upright: bool = False, min_matches: int = 10,
) -> "np.ndarray | None":
    """Estimate 2×3 affine mapping img2 keypoints → img1 keypoints via AKAZE."""
    descriptor_type = (
        cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT if upright
        else cv2.AKAZE_DESCRIPTOR_MLDB
    )
    akaze = cv2.AKAZE_create(descriptor_type=descriptor_type)
    kp1, d1 = akaze.detectAndCompute(img1, None)
    kp2, d2 = akaze.detectAndCompute(img2, None)
    if d1 is None or d2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda x: x.distance)
    if len(matches) < min_matches:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(
        dst, src, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    return M


def _orb_align_tile(
    img1: np.ndarray, img2: np.ndarray,
    nfeatures: int = 50000, min_matches: int = 10,
) -> "np.ndarray | None":
    """Estimate 2×3 affine mapping img2 keypoints → img1 keypoints via ORB."""
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, d1 = orb.detectAndCompute(img1, None)
    kp2, d2 = orb.detectAndCompute(img2, None)
    if d1 is None or d2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda x: x.distance)
    if len(matches) < min_matches:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(
        dst, src, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    return M


def run_tile_feature_registration(
    fixed_id: str,
    moving_id: str,
    init_transform_id: str,
    tile_size: int = 512,
    overlap: int = 64,
    max_disp_px: float = 30.0,
) -> tuple[str, float, float]:
    """Stage 3: Tile-based MI translation registration using H-channel.

    H-channel (hematoxylin) is present in both HE and IHC P40, making it
    cross-modal compatible. Each tissue tile runs a translation-only SimpleITK
    MI registration. NCC gate: only apply displacement if it genuinely improves
    local alignment. Hanning-blended across overlapping tiles.

    Returns (transform_id, global_nmi, avg_tile_ncc).
    """
    from registration.preprocess import extract_hematoxylin

    fixed_arr = get_image(fixed_id)
    moving_arr = get_image(moving_id)

    def to_gray(arr):
        if arr.ndim == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return arr

    # H-channel for tile MI registration (cross-modal compatible)
    fixed_h_id, fixed_h_arr = extract_hematoxylin(fixed_id)
    moving_h_id, moving_h_arr = extract_hematoxylin(moving_id)
    fixed_h_f32 = fixed_h_arr.astype(np.float32)
    moving_h_f32 = moving_h_arr.astype(np.float32)

    # Grayscale for tissue mask and NCC evaluation
    fixed_gray_u8 = to_gray(fixed_arr)
    fixed_gray_f32 = fixed_gray_u8.astype(np.float32)
    moving_gray_f32 = to_gray(moving_arr).astype(np.float32)

    fixed_sitk = _numpy_to_sitk(fixed_gray_f32, fixed_id)
    moving_sitk = _numpy_to_sitk(moving_gray_f32, moving_id)
    fixed_h_sitk = _numpy_to_sitk(fixed_h_f32, fixed_id)
    moving_h_sitk = _numpy_to_sitk(moving_h_f32, moving_id)
    sx, sy = fixed_sitk.GetSpacing()

    init_transform = get_transform(init_transform_id)
    pre_aligned_h_sitk = sitk.Resample(
        moving_h_sitk, fixed_h_sitk, init_transform,
        sitk.sitkLinear, 0.0, moving_h_sitk.GetPixelID(),
    )
    pre_aligned_h = sitk.GetArrayFromImage(pre_aligned_h_sitk).astype(np.float32)

    H, W = fixed_gray_u8.shape
    disp_x = np.zeros((H, W), dtype=np.float32)
    disp_y = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    _, tissue_mask = cv2.threshold(
        fixed_gray_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    stride = tile_size - overlap
    y_starts = list(range(0, H, stride))
    x_starts = list(range(0, W, stride))

    tiles_total = tiles_success = tiles_gated = 0

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            if y1 - y0 < 64 or x1 - x0 < 64:
                continue
            if np.mean(tissue_mask[y0:y1, x0:x1] > 0) < 0.20:
                continue

            tiles_total += 1
            fixed_tile_h = fixed_h_f32[y0:y1, x0:x1]
            moving_tile_h = pre_aligned_h[y0:y1, x0:x1]

            result = _mi_translate_tile(
                fixed_tile_h, moving_tile_h, sx, sy, y0, x0,
                max_disp_px=max_disp_px,
            )
            if result is None:
                continue

            dx_mm, dy_mm = result
            dx_px, dy_px = dx_mm / sx, dy_mm / sy

            # NCC gate: only apply if displacement genuinely improves alignment
            ncc_before = _tile_ncc(fixed_tile_h, moving_tile_h)
            M_shift = np.float32([[1, 0, -dx_px], [0, 1, -dy_px]])
            tile_h_u8 = np.clip(moving_tile_h, 0, 255).astype(np.uint8)
            translated = cv2.warpAffine(
                tile_h_u8, M_shift,
                (moving_tile_h.shape[1], moving_tile_h.shape[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            ncc_after = _tile_ncc(fixed_tile_h, translated.astype(np.float32))

            if ncc_after <= ncc_before:
                tiles_gated += 1
                continue

            tiles_success += 1
            tile_h, tile_w = y1 - y0, x1 - x0
            w = (np.hanning(tile_h).reshape(-1, 1) *
                 np.hanning(tile_w).reshape(1, -1)).astype(np.float32)

            disp_x[y0:y1, x0:x1] += dx_mm * w
            disp_y[y0:y1, x0:x1] += dy_mm * w
            weight_map[y0:y1, x0:x1] += w

    print(f"[tile_feature] {tiles_success}/{tiles_total} applied, "
          f"{tiles_gated} gated (already aligned)")

    valid = weight_map > 0
    disp_x[valid] /= weight_map[valid]
    disp_y[valid] /= weight_map[valid]

    disp_combined = np.stack([disp_x, disp_y], axis=-1)
    disp_sitk = sitk.GetImageFromArray(
        disp_combined.astype(np.float64), isVector=True
    )
    disp_sitk.SetOrigin(fixed_sitk.GetOrigin())
    disp_sitk.SetSpacing(fixed_sitk.GetSpacing())

    disp_transform = sitk.DisplacementFieldTransform(disp_sitk)
    composite = sitk.CompositeTransform(2)
    composite.AddTransform(disp_transform)
    composite.AddTransform(init_transform)

    nmi = _compute_proper_nmi(
        fixed_gray_f32, moving_gray_f32, composite,
        fixed_id=fixed_id, moving_id=moving_id,
    )

    # Per-tile NCC after full registration
    registered_sitk = sitk.Resample(
        moving_sitk, fixed_sitk, composite,
        sitk.sitkLinear, 0.0, moving_sitk.GetPixelID(),
    )
    registered_u8 = np.clip(
        sitk.GetArrayFromImage(registered_sitk), 0, 255
    ).astype(np.uint8)

    tile_nccs = []
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            if y1 - y0 < 64 or x1 - x0 < 64:
                continue
            if np.mean(tissue_mask[y0:y1, x0:x1] > 0) < 0.20:
                continue
            ncc = _tile_ncc(
                fixed_gray_u8[y0:y1, x0:x1],
                registered_u8[y0:y1, x0:x1],
            )
            tile_nccs.append(ncc)

    avg_tile_ncc = float(np.mean(tile_nccs)) if tile_nccs else 0.0
    print(f"[tile_feature] global_NMI={nmi:.4f}  avg_tile_NCC={avg_tile_ncc:.4f} "
          f"({len(tile_nccs)} tiles)")

    transform_id = f"tile_feature_{fixed_id}__{moving_id}"
    _transform_store[transform_id] = composite
    return transform_id, float(nmi), avg_tile_ncc
