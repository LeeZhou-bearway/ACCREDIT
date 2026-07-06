"""
global_reg.py - Stage 1 (rigid) and Stage 2 (affine) registration using SimpleITK.
"""
import numpy as np
import SimpleITK as sitk
from registration.preprocess import get_image, store_image, get_meta


# Global transform store (transform_id -> SimpleITK transform)
_transform_store: dict[str, sitk.Transform] = {}


def _numpy_to_sitk(arr: np.ndarray, image_id: str = "") -> sitk.Image:
    """Convert a 2D numpy uint8 array to a SimpleITK Image with physical spacing."""
    import cv2
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    sitk_img = sitk.GetImageFromArray(arr.astype(np.float32))
    # Set physical spacing from slide metadata (mm/pixel)
    meta = get_meta(image_id) if image_id else {}
    sx = meta.get("spacing_x_mm", 0.001)  # default 1 μm/pixel in mm
    sy = meta.get("spacing_y_mm", 0.001)
    sitk_img.SetSpacing([sx, sy])
    return sitk_img


def get_transform(transform_id: str) -> sitk.Transform:
    if transform_id not in _transform_store:
        raise KeyError(f"transform_id '{transform_id}' not found.")
    return _transform_store[transform_id]


def store_transform(transform_id: str, transform: sitk.Transform) -> str:
    _transform_store[transform_id] = transform
    return transform_id


def run_rigid_registration(
    fixed_id: str,
    moving_id: str,
    sampling_pct: float = 0.1,
    num_iterations: int = 200,
) -> tuple[str, float]:
    """
    Stage 1: Rigid registration (translation + rotation) at thumbnail resolution.
    Returns (transform_id, nmi_score).  NMI is computed via proper joint-entropy
    formula and ranges ~1.0-1.5 (higher = better alignment).
    """
    fixed_arr = get_image(fixed_id)
    moving_arr = get_image(moving_id)

    fixed_sitk = _numpy_to_sitk(fixed_arr, fixed_id)
    moving_sitk = _numpy_to_sitk(moving_arr, moving_id)

    registration = sitk.ImageRegistrationMethod()

    # Metric: Mattes Mutual Information
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(sampling_pct, seed=42)

    # Interpolator
    registration.SetInterpolator(sitk.sitkLinear)

    # Optimizer: LBFGSB
    registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=num_iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7,
    )

    # Initial transform: CenteredTransformInitializer (rigid)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_sitk,
        moving_sitk,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)

    # Multi-resolution
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = registration.Execute(fixed_sitk, moving_sitk)

    # Compute proper NMI (joint-entropy formula, range ~1.0-1.5)
    nmi = _compute_proper_nmi(fixed_arr, moving_arr, final_transform,
                               fixed_id=fixed_id, moving_id=moving_id)

    transform_id = f"rigid_{fixed_id}__{moving_id}"
    _transform_store[transform_id] = final_transform
    return transform_id, float(nmi)


def run_similarity_registration(
    fixed_id: str,
    moving_id: str,
    sampling_pct: float = 0.1,
    num_iterations: int = 300,
) -> tuple[str, float]:
    """
    Stage 1: Similarity registration (uniform scale + rotation + translation).
    Robust multi-init strategy: tries MOMENTS, GEOMETRY, and ±90°/180° rotation
    offsets, returns the best NMI result. Handles cases where tissue orientation
    differs significantly between HE and IHC scanner placements.
    Returns (transform_id, nmi_score).
    """
    import math
    fixed_arr = get_image(fixed_id)
    moving_arr = get_image(moving_id)

    fixed_sitk  = _numpy_to_sitk(fixed_arr, fixed_id)
    moving_sitk = _numpy_to_sitk(moving_arr, moving_id)

    def _run_one(init_transform):
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.REGULAR)
        reg.SetMetricSamplingPercentage(sampling_pct, seed=42)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=num_iterations,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1000,
            costFunctionConvergenceFactor=1e7,
        )
        reg.SetInitialTransform(init_transform, inPlace=False)
        # Multi-resolution pyramid: [4,2,1] is stable for partial tissue.
        # Backup option: [5,4,3,2,1] with sigmas [4,3,2,1,0] (slightly less stable).
        # Avoid [8,4,2,1]: causes "All samples map outside" for partial tissue.
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        t = reg.Execute(fixed_sitk, moving_sitk)
        nmi = _compute_proper_nmi(fixed_arr, moving_arr, t,
                                   fixed_id=fixed_id, moving_id=moving_id)
        return t, float(nmi)

    # --- Build candidate initializations ---
    # 1) MOMENTS
    init_moments = sitk.CenteredTransformInitializer(
        fixed_sitk, moving_sitk, sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    # 2) GEOMETRY
    init_geometry = sitk.CenteredTransformInitializer(
        fixed_sitk, moving_sitk, sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    # 3-5) MOMENTS base + ±90°, 180° rotation offsets
    rotation_offsets = [math.pi / 2, -math.pi / 2, math.pi]
    candidates = [init_moments, init_geometry]
    for angle in rotation_offsets:
        t = sitk.Similarity2DTransform()
        t.SetParameters(init_moments.GetParameters())   # copy moments init
        params = list(t.GetParameters())
        params[0] = params[0] + angle                  # params[0] = angle
        t.SetParameters(params)
        candidates.append(t)

    best_transform, best_nmi = None, -1.0
    for i, cand in enumerate(candidates):
        try:
            t, nmi = _run_one(cand)
            print(f"  [sim init {i}] NMI={nmi:.4f}")
            if nmi > best_nmi:
                best_nmi, best_transform = nmi, t
        except Exception as e:
            print(f"  [sim init {i}] failed: {e}")

    transform_id = f"similarity_{fixed_id}__{moving_id}"
    _transform_store[transform_id] = best_transform
    return transform_id, best_nmi


def run_shape_similarity(
    fixed_id: str,
    moving_id: str,
    sampling_pct: float = 0.1,
    num_iterations: int = 300,
) -> tuple[str, float]:
    """
    Stage 1 (fallback): Similarity registration initialized from tissue shape (binary mask).
    Use when H-channel intensity inversion causes standard intensity-based init to fail
    (e.g., lymphoid follicles bright in HE but dark in IHC P40).

    Computes Otsu binary tissue masks from fixed/moving images, then initializes
    Similarity2DTransform from mask MOMENTS + GEOMETRY + ±45°/±90°/180° offsets (7 candidates).
    Optimization uses same H-channel images + MattesMI as run_similarity_registration.
    Returns (transform_id, nmi_score).
    """
    import math
    import cv2

    fixed_arr = get_image(fixed_id)
    moving_arr = get_image(moving_id)

    fixed_sitk  = _numpy_to_sitk(fixed_arr, fixed_id)
    moving_sitk = _numpy_to_sitk(moving_arr, moving_id)

    def _make_mask_sitk(arr, image_id):
        """Otsu binarize → invert if background bright → morph close → sitk Image."""
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr.copy()
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if binary.mean() > 127:   # background is bright → invert so tissue=255
            binary = 255 - binary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        mask_sitk = sitk.GetImageFromArray(binary.astype(np.float32))
        meta = get_meta(image_id) if image_id else {}
        sx = meta.get("spacing_x_mm", 0.001)
        sy = meta.get("spacing_y_mm", 0.001)
        mask_sitk.SetSpacing([sx, sy])
        return mask_sitk

    fixed_mask  = _make_mask_sitk(fixed_arr, fixed_id)
    moving_mask = _make_mask_sitk(moving_arr, moving_id)

    def _run_one(init_transform):
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.REGULAR)
        reg.SetMetricSamplingPercentage(sampling_pct, seed=42)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=num_iterations,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1000,
            costFunctionConvergenceFactor=1e7,
        )
        reg.SetInitialTransform(init_transform, inPlace=False)
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        t = reg.Execute(fixed_sitk, moving_sitk)
        nmi = _compute_proper_nmi(fixed_arr, moving_arr, t,
                                   fixed_id=fixed_id, moving_id=moving_id)
        return t, float(nmi)

    # Build candidates: MOMENTS, GEOMETRY, rotations, + flips
    init_moments = sitk.CenteredTransformInitializer(
        fixed_mask, moving_mask, sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    init_geometry = sitk.CenteredTransformInitializer(
        fixed_mask, moving_mask, sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    rotation_offsets = [math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2, math.pi]
    candidates = [init_moments, init_geometry]
    for angle in rotation_offsets:
        t = sitk.Similarity2DTransform()
        t.SetParameters(init_moments.GetParameters())
        params = list(t.GetParameters())
        params[0] = params[0] + angle
        t.SetParameters(params)
        candidates.append(t)

    best_transform, best_nmi = None, -1.0
    for i, cand in enumerate(candidates):
        try:
            t, nmi = _run_one(cand)
            print(f"  [shape_sim init {i}] NMI={nmi:.4f}")
            if nmi > best_nmi:
                best_nmi, best_transform = nmi, t
        except Exception as e:
            print(f"  [shape_sim init {i}] failed: {e}")

    transform_id = f"shape_similarity_{fixed_id}__{moving_id}"
    _transform_store[transform_id] = best_transform
    return transform_id, best_nmi


def run_affine_registration(
    fixed_id: str,
    moving_id: str,
    init_transform_id: str,
    num_iterations: int = 200,
) -> tuple[str, float]:
    """
    Stage 2: Affine registration (scale + shear) starting from rigid transform.
    Returns (transform_id, nmi_score).
    """
    fixed_arr = get_image(fixed_id)
    moving_arr = get_image(moving_id)

    fixed_sitk = _numpy_to_sitk(fixed_arr, fixed_id)
    moving_sitk = _numpy_to_sitk(moving_arr, moving_id)

    init_transform = get_transform(init_transform_id)

    # Pre-apply the rigid transform, then run affine from identity on pre-aligned images.
    # This avoids issues with CompositeTransform parameter extraction.
    pre_aligned = sitk.Resample(
        moving_sitk, fixed_sitk, init_transform, sitk.sitkLinear, 0.0,
        moving_sitk.GetPixelID()
    )

    affine_init = sitk.CenteredTransformInitializer(
        fixed_sitk,
        pre_aligned,
        sitk.AffineTransform(2),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(0.15, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=num_iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7,
    )
    registration.SetInitialTransform(affine_init, inPlace=False)
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    affine_result = registration.Execute(fixed_sitk, pre_aligned)

    # Compose: affine_result first (residual correction in HE space),
    # then init_transform (maps HE → IHC).
    # CompositeTransform applies in add order, so:
    # composite(x) = init_transform(affine_result(x))
    composite = sitk.CompositeTransform(2)
    composite.AddTransform(affine_result)
    composite.AddTransform(init_transform)

    nmi = _compute_proper_nmi(fixed_arr, moving_arr, composite,
                               fixed_id=fixed_id, moving_id=moving_id)

    transform_id = f"affine_{fixed_id}__{moving_id}"
    _transform_store[transform_id] = composite
    return transform_id, float(nmi)


def apply_transform_to_image(
    fixed_id: str,
    moving_id: str,
    transform_id: str,
) -> tuple[str, np.ndarray]:
    """Apply a stored transform to a moving image, resample to fixed space."""
    fixed_arr = get_image(fixed_id)
    moving_arr = get_image(moving_id)

    fixed_sitk = _numpy_to_sitk(fixed_arr, fixed_id)
    moving_sitk = _numpy_to_sitk(moving_arr, moving_id)
    transform = get_transform(transform_id)

    resampled = sitk.Resample(
        moving_sitk,
        fixed_sitk,
        transform,
        sitk.sitkLinear,
        0.0,
        moving_sitk.GetPixelID(),
    )
    result_arr = sitk.GetArrayFromImage(resampled).astype(np.uint8)
    new_id = moving_id + f"__warped_{transform_id}"
    store_image(new_id, result_arr)
    return new_id, result_arr


def _compute_proper_nmi(
    fixed_arr: np.ndarray,
    moving_arr: np.ndarray,
    transform: sitk.Transform,
    bins: int = 64,
    fixed_id: str = "",
    moving_id: str = "",
) -> float:
    """
    Compute proper NMI = (H(f) + H(m)) / H(f,m) after applying transform.
    Returns value in range [1.0, inf) — higher means better alignment.
    Typical range for histology: 1.05-1.40.
    """
    import cv2

    def to_gray(arr):
        if arr.ndim == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return arr

    fixed_gray = to_gray(fixed_arr).astype(np.float32)
    moving_gray = to_gray(moving_arr).astype(np.float32)

    fixed_sitk = _numpy_to_sitk(fixed_gray, fixed_id)
    moving_sitk = _numpy_to_sitk(moving_gray, moving_id)

    resampled = sitk.Resample(
        moving_sitk, fixed_sitk, transform, sitk.sitkLinear, 0.0,
        moving_sitk.GetPixelID()
    )
    warped_arr = sitk.GetArrayFromImage(resampled).astype(np.float32)

    # Mask out background (zero) pixels
    mask = (fixed_gray > 0) & (warped_arr > 0)
    if mask.sum() < 100:
        return 1.0

    a_flat = fixed_gray[mask].ravel()
    b_flat = warped_arr[mask].ravel()

    hist_2d, _, _ = np.histogram2d(a_flat, b_flat, bins=bins)
    hist_2d = hist_2d / (hist_2d.sum() + 1e-10)

    p_a = hist_2d.sum(axis=1)
    p_b = hist_2d.sum(axis=0)

    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p + 1e-10))

    H_a = entropy(p_a)
    H_b = entropy(p_b)
    H_ab = entropy(hist_2d.ravel())

    if H_ab < 1e-10:
        return 1.0
    return float((H_a + H_b) / H_ab)


def _compute_nmi_sitk(
    fixed: sitk.Image,
    moving: sitk.Image,
    transform: sitk.Transform,
) -> float:
    """Legacy: Mattes MI proxy. Use _compute_proper_nmi instead."""
    resampled = sitk.Resample(
        moving, fixed, transform, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    metric_filter = sitk.ImageRegistrationMethod()
    metric_filter.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    metric_filter.SetMetricSamplingStrategy(metric_filter.REGULAR)
    metric_filter.SetMetricSamplingPercentage(0.2, seed=42)
    metric_filter.SetInterpolator(sitk.sitkNearestNeighbor)
    identity = sitk.TranslationTransform(2)
    identity.SetOffset([0, 0])
    metric_filter.SetInitialTransform(identity, inPlace=False)
    metric_filter.SetOptimizerAsGradientDescent(learningRate=0.0, numberOfIterations=1)
    try:
        metric_filter.Execute(fixed, resampled)
        value = metric_filter.GetMetricValue()
        return -float(value)
    except Exception:
        return 0.0


def compose_transforms(t1_id: str, t2_id: str) -> tuple[str, sitk.Transform]:
    """Compose two transforms: first apply t1 then t2."""
    t1 = get_transform(t1_id)
    t2 = get_transform(t2_id)
    composite = sitk.CompositeTransform(2)
    composite.AddTransform(t1)
    composite.AddTransform(t2)
    new_id = f"composite_{t1_id}__{t2_id}"
    _transform_store[new_id] = composite
    return new_id, composite


def run_feature_global_registration(
    fixed_id: str,
    moving_id: str,
    sampling_pct: float = 0.1,
    num_iterations: int = 300,
    downsample: float = 0.5,
) -> tuple[str, float]:
    """
    Stage 1 (feature): AKAZE/ORB keypoint matching at downsampled resolution for
    global initialization, then refined with MI optimization.
    Uses H-channel or grayscale + CLAHE for cross-modal robustness.
    Returns (transform_id, nmi_score) or (None, -1.0) if feature matching fails.
    """
    import math
    import cv2

    fixed_arr = get_image(fixed_id)
    moving_arr = get_image(moving_id)

    fixed_sitk = _numpy_to_sitk(fixed_arr, fixed_id)
    moving_sitk = _numpy_to_sitk(moving_arr, moving_id)

    meta_f = get_meta(fixed_id) if fixed_id else {}
    meta_m = get_meta(moving_id) if moving_id else {}
    sx_m = meta_m.get("spacing_x_mm", 0.001)
    sy_m = meta_m.get("spacing_y_mm", 0.001)

    def _prep(arr):
        g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr.copy()
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(g.astype(np.uint8))

    fh, fw = fixed_arr.shape[:2]
    mh, mw = moving_arr.shape[:2]
    f_small = cv2.resize(_prep(fixed_arr), (int(fw * downsample), int(fh * downsample)))
    m_small = cv2.resize(_prep(moving_arr), (int(mw * downsample), int(mh * downsample)))

    M_px = None
    for detector in [cv2.AKAZE_create(), cv2.ORB_create(nfeatures=3000)]:
        try:
            kp1, des1 = detector.detectAndCompute(f_small, None)
            kp2, des2 = detector.detectAndCompute(m_small, None)
            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                continue
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
            if len(matches) < 4:
                continue
            top = matches[:max(4, len(matches) // 2)]
            pts_f = np.float32([kp1[m.queryIdx].pt for m in top])
            pts_m = np.float32([kp2[m.trainIdx].pt for m in top])
            M, inliers = cv2.estimateAffinePartial2D(
                pts_m, pts_f, method=cv2.RANSAC, ransacReprojThreshold=8.0
            )
            if M is not None and inliers is not None and int(inliers.sum()) >= 4:
                M_px = M
                print(f"  [feature init] {int(inliers.sum())} inliers from "
                      f"{detector.__class__.__name__}")
                break
        except Exception:
            continue

    if M_px is None:
        print("  [feature init] no keypoint matches found")
        return None, -1.0

    # Scale translation back to full-resolution pixels
    M_px[0, 2] /= downsample
    M_px[1, 2] /= downsample

    # M_px maps moving_px → fixed_px; invert to get fixed_px → moving_px
    M_inv = cv2.invertAffineTransform(M_px)

    # Extract similarity params (rotation angle, scale, translation in moving mm)
    angle = math.atan2(float(M_inv[1, 0]), float(M_inv[0, 0]))
    scale = math.sqrt(float(M_inv[0, 0]) ** 2 + float(M_inv[1, 0]) ** 2)

    # Sanity check: HE/IHC of same tissue should have scale in [0.7, 1.4]
    if not (0.7 <= scale <= 1.4):
        print(f"  [feature init] rejected: scale={scale:.3f} out of range [0.7, 1.4]")
        return None, -1.0

    tx_mm = float(M_inv[0, 2]) * sx_m
    ty_mm = float(M_inv[1, 2]) * sy_m

    # Similarity2DTransform with zero center: p_moving = scale*R(angle)*p_fixed + t
    sim = sitk.Similarity2DTransform()
    sim.SetCenter([0.0, 0.0])
    sim.SetAngle(angle)
    sim.SetScale(scale)
    sim.SetTranslation([tx_mm, ty_mm])

    def _run_one(init_t):
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.REGULAR)
        reg.SetMetricSamplingPercentage(sampling_pct, seed=42)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=num_iterations,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1000,
            costFunctionConvergenceFactor=1e7,
        )
        reg.SetInitialTransform(init_t, inPlace=False)
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        t = reg.Execute(fixed_sitk, moving_sitk)
        nmi = _compute_proper_nmi(fixed_arr, moving_arr, t,
                                   fixed_id=fixed_id, moving_id=moving_id)
        return t, float(nmi)

    try:
        t, nmi = _run_one(sim)
        print(f"  [feature init] NMI={nmi:.4f} "
              f"(angle={math.degrees(angle):.1f}° scale={scale:.3f})")
        transform_id = f"feature_{fixed_id}__{moving_id}"
        _transform_store[transform_id] = t
        return transform_id, nmi
    except Exception as e:
        print(f"  [feature init] MI refinement failed: {e}")
        return None, -1.0
