"""ACCREDIT IHC-HE registration — improved pipeline (2026 release).

Two entry points:
  1. DIRECT MODE  — register from raw HE and IHC SVS files
        python run_he_ihc.py --he HE.svs --ihc IHC.svs --output_dir OUT/

  2. RESCUE MODE  — refine an existing warped IHC (from baseline ACCREDIT or any source)
        python run_he_ihc.py --he HE.svs --warped_ihc WARPED.png --output_dir OUT/

See README.md for details.
"""
import argparse, os, sys, time, json, glob
import numpy as np
import cv2
import torch as tc
import torch.nn.functional as F
import openslide
import SimpleITK as sitk
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dhr_nonrigid import nonrigid_register, preprocess as dhr_pre, warp as dhr_warp_tensor
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

device = 'cuda' if tc.cuda.is_available() else 'cpu'


# ── Utilities ────────────────────────────────────────────────────────────────
def to_gray(im):
    if im.ndim == 3 and im.shape[2] >= 3:
        return cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2GRAY)
    return im


def norm8(a):
    f = a.astype(np.float32); pos = f[f > 0]
    if len(pos) < 100: return np.zeros_like(a, np.uint8)
    p1, p99 = np.percentile(pos, [1, 99])
    return np.clip((f - p1) / (p99 - p1 + 1e-10) * 255, 0, 255).astype(np.uint8)


def load_svs(svs, target_max=4096):
    try:
        s = openslide.OpenSlide(svs)
        best_lv = 0
        for i, dims in enumerate(s.level_dimensions):
            if max(dims) <= target_max:
                best_lv = i; break
        lw, lh = s.level_dimensions[best_lv]
        img = np.array(s.read_region((0, 0), best_lv, (lw, lh)).convert('RGB'))
        s.close()
        return img
    except Exception:
        import tifffile
        img = tifffile.imread(svs)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if max(img.shape[:2]) > target_max:
            scale = target_max / max(img.shape[:2])
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img


# ── LightGlue (loaded lazily) ────────────────────────────────────────────────
_sp = None; _lg = None
def _init_lg():
    global _sp, _lg
    if _sp is None:
        _sp = SuperPoint(max_num_keypoints=8192).eval().to(device)
        _lg = LightGlue(features='superpoint').eval().to(device)


def lg_match(he_gray, ihc_gray, target_max=1024):
    """SuperPoint + LightGlue matching; returns keypoints in original image coords."""
    _init_lg()
    def prep(a): return cv2.createCLAHE(2.0, (8, 8)).apply(255 - a)
    def to_t(a):
        H, W = a.shape; s = target_max / max(H, W)
        if s < 1.0:
            a = cv2.resize(a, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        return tc.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0).to(device), s
    he_t, s_he = to_t(prep(he_gray))
    ihc_t, s_ihc = to_t(prep(ihc_gray))
    with tc.no_grad():
        f0 = _sp.extract(he_t); f1 = _sp.extract(ihc_t)
        m01 = _lg({'image0': f0, 'image1': f1})
    f0, f1, m01 = [rbd(x) for x in [f0, f1, m01]]
    matches = m01['matches']
    return (f0['keypoints'][matches[..., 0]].cpu().numpy() / s_he,
            f1['keypoints'][matches[..., 1]].cpu().numpy() / s_ihc)


def lg_affine_step(he_gray, ihc_gray):
    """Apply LightGlue + RANSAC similarity transform."""
    Hh, Wh = he_gray.shape
    pts0, pts1 = lg_match(he_gray, ihc_gray)
    if len(pts0) < 10: return ihc_gray
    M, _ = cv2.estimateAffinePartial2D(
        pts1.astype(np.float32), pts0.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if M is None: return ihc_gray
    return cv2.warpAffine(ihc_gray, M, (Wh, Hh), flags=cv2.INTER_LINEAR, borderValue=255)


# ── Nonrigid backends ───────────────────────────────────────────────────────
def apply_dhr_nr(ihc_gray, he_gray, target_max=2048, **kw):
    src, _ = dhr_pre(np.stack([ihc_gray] * 3, -1), target_max=target_max, device=device)
    tgt, _ = dhr_pre(np.stack([he_gray] * 3, -1), target_max=target_max, device=device)
    Hs, Ws = src.shape[-2:]; Ht, Wt = tgt.shape[-2:]
    Hm, Wm = max(Hs, Ht), max(Ws, Wt)
    if (Hs, Ws) != (Hm, Wm): src = F.pad(src, (0, Wm - Ws, 0, Hm - Hs))
    if (Ht, Wt) != (Hm, Wm): tgt = F.pad(tgt, (0, Wm - Wt, 0, Hm - Ht))
    df = nonrigid_register(src, tgt, device=device, echo=False, **kw)
    Hh, Wh = he_gray.shape
    df_full = F.interpolate(df.permute(0, 3, 1, 2), size=(Hh, Wh),
                             mode='bicubic', align_corners=False).permute(0, 2, 3, 1).contiguous()
    src_t = tc.from_numpy(ihc_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    return (dhr_warp_tensor(src_t, df_full)[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def apply_farneback(ihc_gray, he_gray, winsize=21, iterations=10, levels=5, poly_n=7):
    he_p = cv2.createCLAHE(2.0, (8, 8)).apply(255 - he_gray)
    ihc_p = cv2.createCLAHE(2.0, (8, 8)).apply(255 - ihc_gray)
    flow = cv2.calcOpticalFlowFarneback(he_p, ihc_p, None,
                                         pyr_scale=0.5, levels=levels, winsize=winsize,
                                         iterations=iterations, poly_n=poly_n, poly_sigma=1.5, flags=0)
    H, W = ihc_gray.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(ihc_gray, (xx + flow[..., 0]).astype(np.float32),
                      (yy + flow[..., 1]).astype(np.float32),
                      cv2.INTER_LINEAR, borderValue=255)


def apply_demons(ihc_gray, he_gray, iterations=300, smoothing=1.5):
    fixed = sitk.GetImageFromArray((255 - he_gray).astype(np.float32))
    moving = sitk.GetImageFromArray((255 - ihc_gray).astype(np.float32))
    m = sitk.HistogramMatchingImageFilter()
    m.SetNumberOfHistogramLevels(256); m.SetNumberOfMatchPoints(7); m.ThresholdAtMeanIntensityOn()
    moving = m.Execute(moving, fixed)
    demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(iterations)
    demons.SetStandardDeviations(smoothing)
    df = demons.Execute(fixed, moving)
    df_t = sitk.DisplacementFieldTransform(df)
    warped = sitk.Resample(sitk.GetImageFromArray(ihc_gray.astype(np.float32)),
                            fixed, df_t, sitk.sitkLinear, 255.0)
    return sitk.GetArrayFromImage(warped).clip(0, 255).astype(np.uint8)


# ── Recipes (alternating LG-affine + nonrigid variants) ─────────────────────
def recipe_iter(he, ihc, n_iter=4):
    aligned = ihc.copy()
    for _ in range(n_iter):
        aligned = lg_affine_step(he, aligned)
        aligned = apply_dhr_nr(aligned, he)
    return aligned


def recipe_dhr_far(he, ihc, n_dhr=3, n_far=2):
    aligned = ihc.copy()
    for _ in range(n_dhr):
        aligned = lg_affine_step(he, aligned)
        aligned = apply_dhr_nr(aligned, he)
    for _ in range(n_far):
        aligned = lg_affine_step(he, aligned)
        aligned = apply_farneback(aligned, he)
    return aligned


def recipe_dhr_demons(he, ihc, n_dhr=3, demons_iter=300, demons_smooth=1.5):
    aligned = ihc.copy()
    for _ in range(n_dhr):
        aligned = lg_affine_step(he, aligned)
        aligned = apply_dhr_nr(aligned, he)
    aligned = lg_affine_step(he, aligned)
    aligned = apply_demons(aligned, he, iterations=demons_iter, smoothing=demons_smooth)
    return aligned


RECIPES = {
    'iter4':          lambda he, ihc: recipe_iter(he, ihc, n_iter=4),
    'iter6':          lambda he, ihc: recipe_iter(he, ihc, n_iter=6),
    'dhr3_far2':      lambda he, ihc: recipe_dhr_far(he, ihc, n_dhr=3, n_far=2),
    'dhr4_far2':      lambda he, ihc: recipe_dhr_far(he, ihc, n_dhr=4, n_far=2),
    'dhr6_far4':      lambda he, ihc: recipe_dhr_far(he, ihc, n_dhr=6, n_far=4),
    'dhr3_demons300': lambda he, ihc: recipe_dhr_demons(he, ihc, n_dhr=3, demons_iter=300, demons_smooth=1.5),
}


# ── Quality scoring (optional) ───────────────────────────────────────────────
def _otsu_excluding_saturated(g8):
    """Otsu on pixels with 0 < value < 255 (fixes norm8 saturation spike bug)."""
    valid = (g8 > 0) & (g8 < 255)
    if valid.sum() < 100:
        return 127
    hist = np.bincount(g8[valid], minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 127
    levels = np.arange(256, dtype=np.float64)
    sum_total = (levels * hist).sum()
    sum_b, w_b, var_max, threshold = 0.0, 0.0, -1.0, 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t
    return threshold


def _make_mask_ihc(g, min_area_frac=0.0001):
    """Tissue mask for brightfield IHC (fixed Otsu + percentile fallback + area filter)."""
    g8 = g.astype(np.uint8)
    thr = _otsu_excluding_saturated(g8)
    m = (g8 > thr).astype(np.uint8) * 255
    if (m > 0).mean() < 0.05:
        pos = g8[(g8 > 5) & (g8 < 255)]
        if len(pos) > 100:
            m = (g8 >= int(np.percentile(pos, 50))).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    n, lbls, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return (m > 0).astype(np.uint8) * 255
    min_area = int(min_area_frac * m.size)
    keep = np.zeros_like(m, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep |= (lbls == i)
    return keep.astype(np.uint8) * 255


def compute_qcs(he_gray, warped_gray, eval_px=512):
    """Compute QCS v2 @ 512px with fixed Otsu mask for IHC-HE."""
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                        '..', '..', 'quality')))
        from quality_metrics import compute_quality_metrics_v2
    except ImportError:
        return None

    H, W = he_gray.shape
    s = eval_px / max(H, W)
    if s < 1.0:
        he_gray = cv2.resize(he_gray, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        warped_gray = cv2.resize(warped_gray, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    he_inv = norm8(255 - he_gray)
    warp_n = norm8(255 - warped_gray)
    return compute_quality_metrics_v2(_make_mask_ihc(he_inv), _make_mask_ihc(warp_n),
                                      he_inv, warp_n, eval_px=max(he_gray.shape))


# ── Main entry ──────────────────────────────────────────────────────────────
def register(he_gray, ihc_gray, recipe='auto', eval_px=512, verbose=True):
    """Register IHC to HE. Returns (warped_ihc, recipe_name, metrics_dict).

    Args:
      he_gray, ihc_gray: 2D uint8 numpy arrays (brightfield: bg=bright, tissue=dark)
      recipe: recipe name or 'auto' to try all and pick highest QCS
      eval_px: resolution for QCS scoring (matches VALIS convention at 512)
    """
    if recipe != 'auto':
        if recipe not in RECIPES:
            raise ValueError(f"Unknown recipe '{recipe}'. Options: {list(RECIPES.keys())}")
        warped = RECIPES[recipe](he_gray, ihc_gray)
        m = compute_qcs(he_gray, warped, eval_px=eval_px)
        return warped, recipe, m

    # auto: try all, pick best
    best = (-1, None, None, None)
    all_metrics = {}
    for name, fn in RECIPES.items():
        try:
            warped = fn(he_gray, ihc_gray)
            m = compute_qcs(he_gray, warped, eval_px=eval_px)
            if m is not None:
                all_metrics[name] = {'qcs': float(m['qcs'])}
                if m['qcs'] > best[0]:
                    best = (m['qcs'], warped, name, m)
            if verbose:
                qcs_val = m['qcs'] if m else float('nan')
                print(f'  {name:18}: qcs={qcs_val:.3f}')
        except Exception as e:
            if verbose: print(f'  {name:18}: FAILED ({e})')
    return best[1], best[2], best[3]


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description='ACCREDIT HE-IHC improved pipeline')
    p.add_argument('--he', required=True, help='HE SVS file path')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--ihc', help='Raw IHC SVS (direct mode)')
    g.add_argument('--warped_ihc', help='Pre-warped IHC PNG/TIF (rescue mode — refine existing)')
    p.add_argument('--output_dir', required=True, help='Output directory')
    p.add_argument('--recipe', default='auto',
                   help=f'Recipe name or "auto" (try all). Options: {list(RECIPES.keys())}')
    p.add_argument('--target_max', type=int, default=4096,
                   help='Working resolution (SVS pyramid level selection)')
    p.add_argument('--eval_px', type=int, default=512,
                   help='QCS evaluation resolution (512 matches VALIS)')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # Load HE
    print(f'[HE] Loading {args.he} at target_max={args.target_max}...')
    he_rgb = load_svs(args.he, target_max=args.target_max)
    he_gray = to_gray(he_rgb)
    print(f'[HE] shape={he_gray.shape}')

    # Load IHC or warped_ihc
    if args.ihc:
        print(f'[IHC] Loading {args.ihc} (direct mode)...')
        ihc_rgb = load_svs(args.ihc, target_max=args.target_max)
        ihc_gray = to_gray(ihc_rgb)
    else:
        print(f'[IHC] Loading pre-warped {args.warped_ihc} (rescue mode)...')
        ihc_raw = np.array(Image.open(args.warped_ihc))
        # Rescue mode: input may be pre-inverted (tissue=bright) from ACCREDIT overlays
        ihc_gray = to_gray(ihc_raw)
        if ihc_gray.mean() < 128:
            # Looks like already-inverted overlay — invert back to brightfield
            print('[IHC] Detected inverted overlay — re-inverting to brightfield convention')
            ihc_gray = 255 - ihc_gray
        # Resize to HE shape
        if ihc_gray.shape != he_gray.shape:
            ihc_gray = cv2.resize(ihc_gray, (he_gray.shape[1], he_gray.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
    print(f'[IHC] shape={ihc_gray.shape}')

    # Register
    print(f'\n[REGISTER] Running recipe={args.recipe} ...')
    warped, chosen_recipe, metrics = register(he_gray, ihc_gray, recipe=args.recipe,
                                                 eval_px=args.eval_px)
    elapsed = time.time() - t0

    # Save
    out_png = os.path.join(args.output_dir, 'warped_ihc.png')
    Image.fromarray(warped).save(out_png)
    print(f'\n[OUT] warped IHC → {out_png}')

    # Metrics
    result = {
        'mode': 'rescue' if args.warped_ihc else 'direct',
        'recipe_used': chosen_recipe,
        'elapsed_seconds': round(elapsed, 1),
        'he_path': args.he,
        'ihc_path': args.ihc or args.warped_ihc,
        'output_png': out_png,
    }
    if metrics is not None:
        result['qcs_metrics'] = {k: float(v) for k, v in metrics.items()
                                  if isinstance(v, (int, float))}
    with open(os.path.join(args.output_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n[DONE] recipe={chosen_recipe}  time={elapsed:.1f}s')
    if metrics:
        print(f'       QCS@{args.eval_px}={metrics["qcs"]:.3f}  '
              f'TD={metrics["tissue_dice"]:.3f}  BD={metrics["boundary_dice"]:.3f}')


if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    main()
