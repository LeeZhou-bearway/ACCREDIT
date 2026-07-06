"""
quality_metrics.py — Multi-metric registration quality scoring.

Computes 5 discriminating metrics between warped fluorescence and HE:
  1. Tissue Dice        — binary mask overlap (coarse, original metric)
  2. Boundary Dice      — dice on tissue boundary band only (sensitive to shifts)
  3. NCC                — normalized cross-correlation (pixel-level, most discriminating)
  4. Hausdorff Distance — contour distance in pixels (mean + 95th percentile)
  5. Inlier Ratio       — ORB feature match inlier rate (feature-level quality)

Usage:
    from quality_metrics import compute_quality_metrics, score_M_multi

    # Evaluate a registration result (all metrics)
    metrics = compute_quality_metrics(he_tissue_mask, warped_mask, he_inv, warped_fluo)

    # Quick scoring during pipeline (returns single combined score)
    score = score_M_multi(he_tissue, fluo_tissue, he_inv, fluo_norm, M, he_shape)
"""
import cv2
import numpy as np


# ── 1. Tissue Dice ────────────────────────────────────────────────────────────

def tissue_dice(mask_a, mask_b):
    """Binary mask overlap. Classic metric, but insensitive to small shifts."""
    a, b = mask_a > 0, mask_b > 0
    return 2.0 * np.sum(a & b) / (np.sum(a) + np.sum(b) + 1e-10)


# ── 2. Boundary Dice ─────────────────────────────────────────────────────────

def boundary_dice(mask_a, mask_b, band_width=8):
    """Dice on boundary band only. More sensitive to translation/rotation errors.

    Extracts a strip of width `band_width` pixels around each mask boundary,
    then computes dice within that region only.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (band_width * 2 + 1, band_width * 2 + 1))
    a_bin = (mask_a > 0).astype(np.uint8)
    b_bin = (mask_b > 0).astype(np.uint8)

    a_band = cv2.dilate(a_bin, kernel) - cv2.erode(a_bin, kernel)
    b_band = cv2.dilate(b_bin, kernel) - cv2.erode(b_bin, kernel)

    band_region = (a_band > 0) | (b_band > 0)
    if np.sum(band_region) == 0:
        return 0.0

    a_in = (mask_a > 0) & band_region
    b_in = (mask_b > 0) & band_region
    return 2.0 * np.sum(a_in & b_in) / (np.sum(a_in) + np.sum(b_in) + 1e-10)


# ── 3. NCC ────────────────────────────────────────────────────────────────────

def ncc(img_a, img_b, mask=None):
    """Normalized cross-correlation. Most discriminating metric.

    Computes pixel-level correlation between HE inverted gray and warped fluo.
    Optionally restricted to overlap region via mask.
    """
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)

    if mask is not None:
        m = mask > 0
        if np.sum(m) == 0:
            return 0.0
        a, b = a[m], b[m]
    else:
        a, b = a.ravel(), b.ravel()

    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2))
    if denom < 1e-10:
        return 0.0
    return float(np.sum(a * b) / denom)


# ── 4. Hausdorff Distance ────────────────────────────────────────────────────

def hausdorff_distance(mask_a, mask_b, max_pts=2000):
    """Mean and 95th percentile contour distance in pixels.

    Lower = better. Directly measures how far tissue boundaries are
    from each other after registration.
    """
    contours_a, _ = cv2.findContours((mask_a > 0).astype(np.uint8),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_b, _ = cv2.findContours((mask_b > 0).astype(np.uint8),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours_a or not contours_b:
        return 999.0, 999.0

    pts_a = max(contours_a, key=len).reshape(-1, 2).astype(np.float32)
    pts_b = max(contours_b, key=len).reshape(-1, 2).astype(np.float32)

    # Subsample for speed
    for arr_name in ['pts_a', 'pts_b']:
        arr = locals()[arr_name]
        if len(arr) > max_pts:
            idx = np.linspace(0, len(arr) - 1, max_pts, dtype=int)
            if arr_name == 'pts_a':
                pts_a = arr[idx]
            else:
                pts_b = arr[idx]

    # Vectorized distance computation
    dists_ab = np.min(np.sqrt(np.sum(
        (pts_a[:, None, :] - pts_b[None, :, :]) ** 2, axis=2)), axis=1)
    dists_ba = np.min(np.sqrt(np.sum(
        (pts_b[:, None, :] - pts_a[None, :, :]) ** 2, axis=2)), axis=1)

    all_dists = np.concatenate([dists_ab, dists_ba])
    return float(np.mean(all_dists)), float(np.percentile(all_dists, 95))


# ── 5. ORB Inlier Ratio ──────────────────────────────────────────────────────

def orb_inlier_ratio(img_a, img_b, n_features=5000, reproj_thresh=5.0):
    """Feature matching inlier rate after RANSAC.

    Measures geometric consistency of local features between HE and warped fluo.
    Returns (inlier_ratio, n_inliers, n_matches).
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(img_a, None)
    kp2, des2 = orb.detectAndCompute(img_b, None)

    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        return 0.0, 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
    n_matches = len(matches)

    if n_matches < 4:
        return 0.0, 0, n_matches

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    _, inliers = cv2.estimateAffinePartial2D(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=reproj_thresh)

    if inliers is None:
        return 0.0, 0, n_matches

    n_inliers = int(inliers.sum())
    return n_inliers / n_matches, n_inliers, n_matches


# ── Combined API ──────────────────────────────────────────────────────────────

def compute_quality_metrics(he_mask, warped_mask, he_inv, warped_fluo):
    """Compute all 5 metrics at once.

    Args:
        he_mask:      HE tissue binary mask (uint8, 0/255)
        warped_mask:  Warped fluo tissue mask (uint8, 0/255)
        he_inv:       HE inverted grayscale (uint8)
        warped_fluo:  Warped fluo normalized (uint8)

    Returns:
        dict with all metric values.
    """
    overlap = ((he_mask > 0) & (warped_mask > 0)).astype(np.uint8) * 255

    td = tissue_dice(he_mask, warped_mask)
    bd = boundary_dice(he_mask, warped_mask)
    nc = ncc(he_inv, warped_fluo, mask=overlap)
    hd_mean, hd_95 = hausdorff_distance(he_mask, warped_mask)
    ir, n_inliers, n_matches = orb_inlier_ratio(he_inv, warped_fluo)

    # Coverage metrics
    total_px = warped_mask.size
    warped_pos = warped_mask > 0
    coverage = warped_pos.sum() / total_px if total_px > 0 else 0
    he_bg = he_mask == 0
    bg_total = he_bg.sum()
    bg_coverage = (warped_pos & he_bg).sum() / bg_total if bg_total > 0 else 0

    return {
        'tissue_dice':    round(td, 4),
        'boundary_dice':  round(bd, 4),
        'ncc':            round(nc, 4),
        'hausdorff_mean': round(hd_mean, 2),
        'hausdorff_95':   round(hd_95, 2),
        'inlier_ratio':   round(ir, 4),
        'n_inliers':      n_inliers,
        'n_matches':      n_matches,
        'coverage':       round(coverage, 4),
        'bg_coverage':    round(bg_coverage, 4),
    }


# ── Composite Quality Score (QCS) ────────────────────────────────────────────
#
# QCS = (0.30×TD + 0.30×BD + 0.15×HD + 0.10×IR + 0.15×NMI) × cov × bg
#
# 5 metrics, each normalized to [0, 1] then weighted:
#   tissue_dice    0.30  — overall overlap, robust but less sensitive to small errors
#   boundary_dice  0.30  — dice on boundary band, sensitive to alignment shifts
#   hausdorff      0.15  — geometric distance (HD_MAX scales with eval resolution)
#   inlier_ratio   0.10  — ORB feature match quality, noisy for cross-modal
#   NMI            0.15  — normalized mutual information, cross-modality metric
#
# HD_MAX = 500px at eval_px=4000. Scales proportionally for other resolutions.
# NMI normalized: (nmi - 1.0) / 0.15 maps [1.0, 1.15] → [0, 1].
#
# Evaluation resolution is modality-specific:
#   DAPI/CODEX/fluorescence: eval_px=4000 (standard Otsu + close + dilate mask)
#   IHC-HE (brightfield):   eval_px=512  (fixed Otsu excluding saturated pixels)

WEIGHT_TISSUE_DICE   = 0.30
WEIGHT_BOUNDARY_DICE = 0.30
WEIGHT_HAUSDORFF     = 0.15
WEIGHT_INLIER_RATIO  = 0.10
WEIGHT_NMI           = 0.15

HD_MAX = 500.0   # pixel distance at which hausdorff score → 0 (at 4000px eval)
IR_MAX = 0.15    # inlier ratio at which score → 1
NMI_MIN = 1.0    # NMI lower bound (no mutual info)
NMI_RANGE = 0.15 # NMI normalization: [1.0, 1.15] → [0, 1]
COV_MIN = 0.20   # tissue coverage below this gets linearly penalized
COV_FAIL = 0.05  # tissue coverage below this = failed registration


def compute_nmi(img_a, img_b, bins=64):
    """Normalized Mutual Information between two grayscale images.

    Standard cross-modality metric. Returns NMI >= 1.0 (1.0 = no mutual info).
    """
    mask = (img_a > 0) & (img_b > 0)
    if mask.sum() < 100:
        return 1.0
    af = img_a[mask].ravel().astype(np.float64)
    bf = img_b[mask].ravel().astype(np.float64)
    h, _, _ = np.histogram2d(af, bf, bins=bins)
    h = h / (h.sum() + 1e-10)
    pa, pb = h.sum(1), h.sum(0)
    def ent(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p + 1e-10))
    Ha, Hb, Hab = ent(pa), ent(pb), ent(h.ravel())
    return float((Ha + Hb) / Hab) if Hab > 1e-10 else 1.0


def composite_quality_score(metrics, eval_px=4000):
    """Composite Quality Score (QCS).

    QCS = (0.30×TD + 0.30×BD + 0.15×HD + 0.10×IR + 0.15×NMI) × cov × bg
    """
    td = metrics.get('tissue_dice', 0) or 0
    bd = metrics.get('boundary_dice', 0) or 0
    hd = metrics.get('hausdorff_mean', HD_MAX) or HD_MAX
    ir = metrics.get('inlier_ratio', 0) or 0
    nmi = metrics.get('nmi', NMI_MIN) or NMI_MIN
    cov = metrics.get('coverage', 1.0)
    if cov is None: cov = 1.0
    bg = metrics.get('bg_coverage', 0.0) or 0.0

    hd_max = HD_MAX * eval_px / 4000.0

    s_td  = max(0.0, min(1.0, td))
    s_bd  = max(0.0, min(1.0, bd))
    s_hd  = max(0.0, min(1.0, 1.0 - hd / hd_max))
    s_ir  = max(0.0, min(1.0, ir / IR_MAX))
    s_nmi = max(0.0, min(1.0, (nmi - NMI_MIN) / NMI_RANGE))

    qcs = (WEIGHT_TISSUE_DICE   * s_td +
           WEIGHT_BOUNDARY_DICE * s_bd +
           WEIGHT_HAUSDORFF     * s_hd +
           WEIGHT_INLIER_RATIO  * s_ir +
           WEIGHT_NMI           * s_nmi)

    cov_factor = min(1.0, cov / COV_MIN)
    bg_penalty = max(0.0, 1.0 - bg)
    qcs *= cov_factor * bg_penalty

    return round(qcs, 4)


def compute_quality_metrics_v2(he_mask, warped_mask, he_inv, warped_fluo, eval_px=4000):
    """Compute all metrics + NMI + QCS.

    Returns dict with 9 metrics: td, bd, ncc, hd, ir, nmi, coverage, bg_coverage, qcs.
    """
    metrics = compute_quality_metrics(he_mask, warped_mask, he_inv, warped_fluo)
    metrics['nmi'] = round(compute_nmi(he_inv, warped_fluo), 4)
    metrics['qcs'] = composite_quality_score(metrics, eval_px=eval_px)
    return metrics


# ── CyCIF-specific QCS ─────────────────────────────────────────────────────
#
# For full-FOV modalities (CyCIF, CODEX whole-slide) where both images should
# cover roughly the same tissue area. Key differences from default QCS:
#   - Higher coverage threshold (COV_MIN_CYCIF=0.70 vs 0.20)
#   - Area ratio penalty: warped_area / he_area should be ~1.0
#   - Reweighted: more NCC, less IR/NMI (cross-modal feature matching unreliable)

CYCIF_COV_MIN  = 0.70
CYCIF_COV_FAIL = 0.20
CYCIF_NCC_MAX  = 0.40   # cross-modal NCC ceiling: NCC=0.4 → score 1.0

CYCIF_WEIGHT_TISSUE_DICE   = 0.25
CYCIF_WEIGHT_BOUNDARY_DICE = 0.25
CYCIF_WEIGHT_NCC           = 0.20
CYCIF_WEIGHT_HAUSDORFF     = 0.20
CYCIF_WEIGHT_NMI           = 0.10


def composite_quality_score_cycif(metrics, eval_px=4000):
    """QCS for full-FOV CyCIF/CODEX registration.

    Stricter coverage penalty and area-ratio penalty vs default QCS.
    NCC normalized to [0, NCC_MAX] since cross-modal NCC is inherently low.
    """
    td = metrics.get('tissue_dice', 0) or 0
    bd = metrics.get('boundary_dice', 0) or 0
    nc = metrics.get('ncc', 0) or 0
    hd = metrics.get('hausdorff_mean', HD_MAX) or HD_MAX
    nmi = metrics.get('nmi', NMI_MIN) or NMI_MIN
    cov = metrics.get('coverage', 1.0)
    if cov is None: cov = 1.0
    bg = metrics.get('bg_coverage', 0.0) or 0.0

    hd_max = HD_MAX * eval_px / 4000.0

    s_td  = max(0.0, min(1.0, td))
    s_bd  = max(0.0, min(1.0, bd))
    s_ncc = max(0.0, min(1.0, nc / CYCIF_NCC_MAX))
    s_hd  = max(0.0, min(1.0, 1.0 - hd / hd_max))
    s_nmi = max(0.0, min(1.0, (nmi - NMI_MIN) / NMI_RANGE))

    qcs = (CYCIF_WEIGHT_TISSUE_DICE   * s_td +
           CYCIF_WEIGHT_BOUNDARY_DICE  * s_bd +
           CYCIF_WEIGHT_NCC            * s_ncc +
           CYCIF_WEIGHT_HAUSDORFF      * s_hd +
           CYCIF_WEIGHT_NMI            * s_nmi)

    # Coverage penalty: stricter threshold for full-FOV
    if cov < CYCIF_COV_FAIL:
        cov_factor = 0.0
    elif cov < CYCIF_COV_MIN:
        cov_factor = (cov - CYCIF_COV_FAIL) / (CYCIF_COV_MIN - CYCIF_COV_FAIL)
    else:
        cov_factor = 1.0

    # Background coverage penalty (same as default)
    bg_penalty = max(0.0, 1.0 - bg)

    # Area ratio penalty: Gaussian-shaped, gentle near 1.0
    # ratio=1.0 → 1.0, ratio=1.2 → ~0.98, ratio=2.0 → ~0.72
    he_area = metrics.get('_he_tissue_area', 0)
    warped_area = metrics.get('_warped_tissue_area', 0)
    if he_area > 0 and warped_area > 0:
        ratio = warped_area / he_area
        log_ratio = np.log(ratio + 1e-10)
        area_penalty = np.exp(-0.5 * log_ratio ** 2)
    else:
        area_penalty = 1.0

    qcs *= cov_factor * bg_penalty * area_penalty

    return round(qcs, 4)


def compute_quality_metrics_cycif(he_mask, warped_mask, he_inv, warped_fluo, eval_px=4000):
    """Compute all metrics + CyCIF-specific QCS.

    Same metrics as v2, but uses composite_quality_score_cycif for scoring.
    """
    metrics = compute_quality_metrics(he_mask, warped_mask, he_inv, warped_fluo)
    metrics['nmi'] = round(compute_nmi(he_inv, warped_fluo), 4)

    # Add tissue area counts for area ratio penalty
    metrics['_he_tissue_area'] = int((he_mask > 0).sum())
    metrics['_warped_tissue_area'] = int((warped_mask > 0).sum())
    area_ratio = metrics['_warped_tissue_area'] / max(metrics['_he_tissue_area'], 1)
    metrics['area_ratio'] = round(area_ratio, 4)

    metrics['qcs'] = composite_quality_score_cycif(metrics, eval_px=eval_px)
    return metrics


# ── Fast scoring for pipeline candidate selection ─────────────────────────────

def score_M_fast(he_tissue, fluo_tissue, he_inv, fluo_norm, M, he_shape):
    """Fast composite scoring for a candidate M (no ORB — too slow for inner loop).

    Uses tissue_dice + boundary_dice + hausdorff only (3 metrics, ~10ms).
    Returns (qcs_fast, metrics_dict).
    """
    h, w = he_shape[:2]
    warped_mask = cv2.warpAffine(fluo_tissue, M, (w, h))

    td = tissue_dice(he_tissue, warped_mask)
    bd = boundary_dice(he_tissue, warped_mask)
    hd_mean, hd_95 = hausdorff_distance(he_tissue, warped_mask)

    s_td = max(0.0, min(1.0, td))
    s_bd = max(0.0, min(1.0, bd))
    s_hd = max(0.0, min(1.0, 1.0 - hd_mean / HD_MAX))

    # Reweight without IR and NMI (renormalize: 0.30+0.30+0.15 = 0.75)
    w_td = WEIGHT_TISSUE_DICE / 0.75
    w_bd = WEIGHT_BOUNDARY_DICE / 0.75
    w_hd = WEIGHT_HAUSDORFF / 0.75
    qcs_fast = w_td * s_td + w_bd * s_bd + w_hd * s_hd

    metrics = {
        'tissue_dice':    round(td, 4),
        'boundary_dice':  round(bd, 4),
        'hausdorff_mean': round(hd_mean, 2),
        'hausdorff_95':   round(hd_95, 2),
        'qcs_fast':       round(qcs_fast, 4),
    }
    return round(qcs_fast, 4), metrics
