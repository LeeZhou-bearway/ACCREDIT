"""QCS v2 Formula — Final version for paper.

Weights: TD=0.30  BD=0.30  HD=0.15  IR=0.10  NMI=0.15
  (changed from v1: BD 0.45→0.30, added NMI 0.15)

Metrics (9 total):
  1. tissue_dice      — Dice of warped DAPI mask vs HE tissue mask
  2. boundary_dice    — Dice of boundary contours
  3. ncc              — Normalized Cross-Correlation (stored but NOT in QCS)
  4. hausdorff_mean   — Mean Hausdorff distance (pixels)
  5. inlier_ratio     — ORB feature match inlier ratio
  6. nmi              — Normalized Mutual Information (cross-modal)
  7. coverage         — Warped DAPI mask area / total area
  8. bg_coverage      — Warped DAPI on HE background / background area
  9. qcs              — Composite quality score

QCS = base_score × cov_factor × bg_penalty

base_score = 0.30 × s_td + 0.30 × s_bd + 0.15 × s_hd + 0.10 × s_ir + 0.15 × s_nmi

Normalization:
  s_td  = clamp(tissue_dice, 0, 1)
  s_bd  = clamp(boundary_dice, 0, 1)
  s_hd  = clamp(1 - hausdorff_mean / HD_MAX, 0, 1)
  s_ir  = clamp(inlier_ratio / 0.15, 0, 1)
  s_nmi = clamp((nmi - 1.0) / 0.15, 0, 1)    # NMI ∈ [1.0, 1.15] → [0, 1]

Penalties:
  cov_factor = min(1, coverage / 0.20)
  bg_penalty = max(0, 1 - bg_coverage)

HD_MAX scaling:
  HD_MAX = 500 at 4000px eval. Scale proportionally: HD_MAX = 500 × eval_px / 4000

Paper uses 4000px eval resolution (15 samples, excluding Pancreas_Cancer and Pancreas).
"""

# Constants
WEIGHT_TD  = 0.30
WEIGHT_BD  = 0.30
WEIGHT_HD  = 0.15
WEIGHT_IR  = 0.10
WEIGHT_NMI = 0.15

HD_MAX_REF = 500.0   # at 4000px
IR_MAX     = 0.15
NMI_MIN    = 1.0     # NMI lower bound (no mutual info)
NMI_RANGE  = 0.15    # NMI normalization range: [1.0, 1.15] → [0, 1]
COV_MIN    = 0.20
EVAL_PX_REF = 4000


def compute_qcs_v2(metrics, eval_px=4000):
    """Compute QCS v2 from metrics dict.

    Args:
        metrics: dict with keys tissue_dice, boundary_dice, hausdorff_mean,
                 inlier_ratio, nmi, coverage, bg_coverage
        eval_px: evaluation resolution (max dimension in pixels)

    Returns:
        float QCS score in [0, 1]
    """
    td  = metrics.get('tissue_dice', 0) or 0
    bd  = metrics.get('boundary_dice', 0) or 0
    hd  = metrics.get('hausdorff_mean', 9999) or 9999
    ir  = metrics.get('inlier_ratio', 0) or 0
    nmi = metrics.get('nmi', NMI_MIN) or NMI_MIN
    cov = metrics.get('coverage', 1.0)
    if cov is None: cov = 1.0
    bg  = metrics.get('bg_coverage', 0.0) or 0.0

    hd_max = HD_MAX_REF * eval_px / EVAL_PX_REF

    s_td  = max(0.0, min(1.0, td))
    s_bd  = max(0.0, min(1.0, bd))
    s_hd  = max(0.0, min(1.0, 1.0 - hd / hd_max))
    s_ir  = max(0.0, min(1.0, ir / IR_MAX))
    s_nmi = max(0.0, min(1.0, (nmi - NMI_MIN) / NMI_RANGE))

    qcs = (WEIGHT_TD * s_td + WEIGHT_BD * s_bd + WEIGHT_HD * s_hd +
           WEIGHT_IR * s_ir + WEIGHT_NMI * s_nmi)

    cov_factor = min(1.0, cov / COV_MIN)
    bg_penalty = max(0.0, 1.0 - bg)
    qcs *= cov_factor * bg_penalty

    return round(qcs, 4)


def compute_nmi(a, b, bins=64):
    """Compute Normalized Mutual Information between two grayscale images.

    Args:
        a, b: uint8 numpy arrays (same shape)

    Returns:
        float NMI value. 1.0 = no mutual info, higher = more shared info.
    """
    import numpy as np
    mask = (a > 0) & (b > 0)
    if mask.sum() < 100:
        return 1.0
    af = a[mask].ravel().astype(np.float64)
    bf = b[mask].ravel().astype(np.float64)
    h, _, _ = np.histogram2d(af, bf, bins=bins)
    h = h / (h.sum() + 1e-10)
    pa, pb = h.sum(1), h.sum(0)

    def ent(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p + 1e-10))

    Ha, Hb, Hab = ent(pa), ent(pb), ent(h.ravel())
    return float((Ha + Hb) / Hab) if Hab > 1e-10 else 1.0
