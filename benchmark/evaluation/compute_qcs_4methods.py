"""Compute QCS v2 @ 4000px for DHR, VoltRon, SimpleITK, Palom (16 samples).
VALIS and ACCREDIT already computed separately.
"""
import sys, os, gc, json, csv
import numpy as np
import cv2
import tifffile
from PIL import Image as PILImage
PILImage.MAX_IMAGE_PIXELS = None

sys.path.insert(0, '/home/zhoule/workspace/Image_Reg')
from quality_metrics import (tissue_dice, boundary_dice, hausdorff_distance,
                              ncc, orb_inlier_ratio)

BENCHMARK = '/home/zhoule/workspace/Image_Reg/benchmark_data/DAPI/Xenium'
V2 = '/home/zhoule/workspace/Image_Reg/benchmark_v2'
OUT_DIR = '/home/zhoule/workspace/ACCREDIT/paper_figures/qcs_final_reference'
QCS_PX = 4000
HD_MAX = 500.0
IR_MAX = 0.15
NMI_MIN = 1.0
NMI_RANGE = 0.15

SAMPLES = {
    'Breast_Cancer':      'Xenium_Prime_Breast_Cancer_FFPE',
    'Cervical_Cancer':    'Xenium_Prime_Cervical_Cancer_FFPE',
    'Lung_Cancer_Prime':  'Xenium_Prime_Human_Lung_Cancer_FFPE',
    'Ovary':              'Xenium_Prime_Human_Ovary_FF',
    'Mouse_Brain':        'Xenium_Prime_Mouse_Brain_Coronal_FF',
    'BoneMarrow':         'Xenium_V1_hBoneMarrow_ALL',
    'Liver':              'Xenium_V1_hLiver_nondiseased_FFPE',
    'Colorectal_Cancer':  'Xenium_V1_Human_Colorectal_Cancer_FFPE',
    'Kidney':             'Xenium_V1_Human_Kidney_FFPE_Protein',
    'Pancreas':           'Xenium_V1_human_Pancreas_FFPE',
    'Skin':               'Xenium_Prime_Human_Skin_FFPE',
    'Heart':              'Xenium_V1_hHeart_nondiseased_FFPE',
    'humanLung':          'Xenium_V1_humanLung_Cancer_FFPE',
    'Prostate':           'Xenium_Prime_Human_Prostate_FFPE',
    'Rep1':               'Xenium_FFPE_Human_Breast_Cancer_Rep1',
    'Rep2':               'Xenium_FFPE_Human_Breast_Cancer_Rep2',
}


def norm8(a):
    f = a.astype(np.float32)
    pos = f[f > 0]
    if len(pos) < 100: return np.zeros_like(a, dtype=np.uint8)
    p1, p99 = np.percentile(pos, [1, 99])
    return np.clip((f - p1) / (p99 - p1 + 1e-10) * 255, 0, 255).astype(np.uint8)


def downsample_to(img, target_px):
    h, w = img.shape[:2]
    scale = target_px / max(h, w)
    if scale >= 1.0: return img
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)


def make_mask(g, min_area_frac=0.0001):
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_lbl <= 1: return m > 0
    min_area = int(min_area_frac * m.size)
    keep = np.zeros_like(m, dtype=bool)
    for i in range(1, n_lbl):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep |= (lbls == i)
    return keep


def compute_nmi(a, b, bins=64):
    mask = (a > 0) & (b > 0)
    if mask.sum() < 100: return 1.0
    af = a[mask].ravel().astype(np.float64)
    bf = b[mask].ravel().astype(np.float64)
    h, _, _ = np.histogram2d(af, bf, bins=bins)
    h = h / (h.sum() + 1e-10)
    pa, pb = h.sum(1), h.sum(0)
    def ent(p): p = p[p > 0]; return -np.sum(p * np.log(p + 1e-10))
    Ha, Hb, Hab = ent(pa), ent(pb), ent(h.ravel())
    return float((Ha + Hb) / Hab) if Hab > 1e-10 else 1.0


def compute_metrics(he_mask_b, warped_mask_b, he_inv, warped_fluo):
    ys, xs = np.where(warped_mask_b)
    if len(ys) < 100: return None
    y1, y2, x1, x2 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    hm = (he_mask_b[y1:y2, x1:x2].astype(np.uint8))*255
    wm = (warped_mask_b[y1:y2, x1:x2].astype(np.uint8))*255
    hi = he_inv[y1:y2, x1:x2]; wf = warped_fluo[y1:y2, x1:x2]
    overlap = ((hm>0)&(wm>0)).astype(np.uint8)*255
    td = tissue_dice(hm, wm); bd = boundary_dice(hm, wm)
    nc = ncc(hi, wf, mask=overlap)
    hd_mean, _ = hausdorff_distance(hm, wm)
    ir, _, _ = orb_inlier_ratio(hi, wf)
    nmi_val = compute_nmi(hi, wf)
    cov = warped_mask_b.sum() / warped_mask_b.size
    bg = (warped_mask_b & ~he_mask_b).sum() / max(1, (~he_mask_b).sum())
    s_td=max(0,min(1,td)); s_bd=max(0,min(1,bd))
    s_hd=max(0,min(1,1-hd_mean/HD_MAX)); s_ir=max(0,min(1,ir/IR_MAX))
    s_nmi=max(0,min(1,(nmi_val-NMI_MIN)/NMI_RANGE))
    qcs=0.30*s_td+0.30*s_bd+0.15*s_hd+0.10*s_ir+0.15*s_nmi
    qcs*=min(1,cov/0.20)*max(0,1-bg)
    return {'tissue_dice':round(float(td),4),'boundary_dice':round(float(bd),4),
            'ncc':round(float(nc),4),'hausdorff_mean':round(float(hd_mean),2),
            'inlier_ratio':round(float(ir),4),'nmi':round(float(nmi_val),4),
            'coverage':round(float(cov),4),'bg_coverage':round(float(bg),4),'qcs':round(qcs,4)}


def get_warped_2000px(method, sample_dn):
    """Load 2000px warped output for DHR/VoltRon/SimpleITK."""
    if method == 'dhr':
        p = f'{V2}/runs/dhr/{sample_dn}/warped_source.png'
        if not os.path.exists(p): return None
        w = np.array(PILImage.open(p))
        if w.ndim == 3: w = cv2.cvtColor(w, cv2.COLOR_RGB2GRAY)
        w[w == 255] = 0
        return w
    elif method == 'voltron':
        p = f'{V2}/runs/voltron/{sample_dn}/warped_dapi.png'
        if not os.path.exists(p): return None
        w = np.array(PILImage.open(p))
        if w.ndim == 3: w = cv2.cvtColor(w, cv2.COLOR_RGB2GRAY)
        return w
    elif method == 'simpleitk':
        p = f'{V2}/runs/simpleitk/{sample_dn}/warped_work.png'
        if not os.path.exists(p): return None
        w = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        return w
    return None


def get_warped_palom(sample_dn, he_eval_w, he_eval_h):
    """Palom: warp dapi_mip with M_full at eval resolution."""
    sd = f'{BENCHMARK}/{sample_dn}'
    p = f'{V2}/runs/palom/{sample_dn}/M_full.npy'
    if not os.path.exists(p): return None
    M = np.load(p)
    if M.shape == (2,3): M = np.vstack([M,[0,0,1]])
    dapi = np.array(PILImage.open(f'{sd}/voltron_input/dapi_mip.png'))
    if dapi.ndim == 3: dapi = cv2.cvtColor(dapi, cv2.COLOR_RGB2GRAY)
    with tifffile.TiffFile(f'{sd}/he_image.ome.tif') as t:
        he_fh, he_fw = t.pages[0].shape[:2]
    with tifffile.TiffFile(f'{sd}/morphology.ome.tif') as t:
        fl_fh, fl_fw = t.pages[0].shape[:2]
    s_d = dapi.shape[1] / fl_fw
    s_h = he_eval_w / he_fw
    H = np.diag([s_h,s_h,1.0]) @ M @ np.diag([1/s_d,1/s_d,1.0])
    return cv2.warpAffine(dapi, H[:2].astype(np.float64), (he_eval_w, he_eval_h))


def main():
    METHODS = ['dhr', 'voltron', 'simpleitk', 'palom']
    results = {}

    for key, dn in SAMPLES.items():
        sd = f'{BENCHMARK}/{dn}'
        print(f'\n=== {key} ===', flush=True)
        try:
            # Load HE for eval
            he = tifffile.imread(f'{sd}/he_image.ome.tif')
            if he.ndim == 3 and he.shape[0] in (3,4): he = np.moveaxis(he, 0, -1)
            he = he[..., :3]
            he_fh, he_fw = he.shape[:2]
            he_gray = cv2.cvtColor(he, cv2.COLOR_RGB2GRAY)
            he_inv = (255 - he_gray).astype(np.uint8)
            he_inv_ds = downsample_to(he_inv, QCS_PX)
            he_mask_b = make_mask(he_inv_ds)
            eval_h, eval_w = he_inv_ds.shape[:2]
            del he, he_gray, he_inv
            gc.collect()

            sample_results = {}
            for method in METHODS:
                print(f'  {method}...', flush=True, end=' ')
                if method == 'palom':
                    warped = get_warped_palom(dn, eval_w, eval_h)
                else:
                    warped_raw = get_warped_2000px(method, dn)
                    if warped_raw is not None:
                        warped = cv2.resize(warped_raw, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        warped = None

                if warped is None:
                    print('N/A', flush=True)
                    sample_results[method] = None
                    continue

                w_mask = make_mask(warped)
                met = compute_metrics(he_mask_b, w_mask, he_inv_ds, warped)
                if met:
                    print(f'QCS={met["qcs"]:.4f}', flush=True)
                else:
                    print('FAIL', flush=True)
                sample_results[method] = met
                del warped

            results[key] = sample_results
            del he_inv_ds, he_mask_b
            gc.collect()
        except Exception as e:
            import traceback
            print(f'  ERROR: {e}', flush=True)
            traceback.print_exc()
            results[key] = {'error': str(e)}

    # Save
    with open(f'{OUT_DIR}/qcs_4methods_4000px.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f'\n{"sample":<20}', end='')
    for m in ['dhr','voltron','simpleitk','palom']:
        print(f' {m:>10}', end='')
    print()
    print('-'*65)
    for key in SAMPLES:
        r = results.get(key, {})
        print(f'{key:<20}', end='')
        for m in ['dhr','voltron','simpleitk','palom']:
            met = r.get(m)
            if met:
                print(f' {met["qcs"]:>10.4f}', end='')
            else:
                print(f' {"N/A":>10}', end='')
        print()

    print(f'\nSaved: {OUT_DIR}/qcs_4methods_4000px.json')


if __name__ == '__main__':
    main()
