"""
run_dapi_via_contour.py - Register DAPI to HE via boundary contour as intermediate.

For samples where direct DAPI-HE registration fails (e.g. partial coverage,
weak DAPI signal), this pipeline leverages the boundary pipeline's existing
contour-to-HE alignment:

  1. DAPI → contour space: resize DAPI to match filled_image dimensions
     (both come from same Xenium instrument, coordinate systems align via resize)
  2. contour → HE: use boundary pipeline's affine_M.npy (ORB-based alignment)

Requirements:
  - Boundary pipeline must have been run first (output/filled_image.png, output/affine_M.npy)
  - morphology_focus or morphology.ome.tif available

Usage:
    conda run -n clam_latest python run_dapi_via_contour.py \
        --he data/he_image.ome.tif \
        --fluo data/morphology_focus/morphology_focus_0001.ome.tif \
        --boundary_dir data/output \
        --output_dir data/output_dapi_via_contour

    # Or with auto-detect (looks for files in sample directory):
    conda run -n clam_latest python run_dapi_via_contour.py \
        --sample_dir benchmark_data/DAPI/Xenium/Xenium_Prime_Human_Prostate_FFPE
"""
import argparse
import cv2
import json
import numpy as np
import os
import sys
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def norm8(a):
    f = a.astype(np.float32)
    pos = f[f > 0]
    if len(pos) < 100:
        return np.zeros_like(a, dtype=np.uint8)
    p1, p99 = np.percentile(pos, [1, 99])
    return np.clip((f - p1) / (p99 - p1 + 1e-10) * 255, 0, 255).astype(np.uint8)


def register_dapi_via_contour(he_path, fluo_path, boundary_dir, output_dir,
                               affine_name='affine_M.npy', display_size=4000, log=print):
    """
    Register DAPI to HE using boundary contour alignment as bridge.

    Args:
        he_path: path to HE OME-TIFF
        fluo_path: path to DAPI/morphology TIFF (single channel or multi-page)
        boundary_dir: directory containing filled_image.png and affine_M.npy
        output_dir: output directory
        affine_name: which affine matrix to use (default: affine_M.npy, NOT affine_M_final.npy)
        display_size: max dimension for display overlay
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load boundary pipeline results ──
    filled_path = os.path.join(boundary_dir, 'filled_image.png')
    affine_path = os.path.join(boundary_dir, affine_name)

    if not os.path.exists(filled_path):
        log(f'ERROR: {filled_path} not found. Run boundary pipeline first.')
        return None
    if not os.path.exists(affine_path):
        log(f'ERROR: {affine_path} not found. Run boundary pipeline first.')
        return None

    filled = cv2.imread(filled_path, cv2.IMREAD_GRAYSCALE)
    fh, fw = filled.shape[:2]
    M = np.load(affine_path)
    log(f'Filled image: {fw}x{fh}')
    log(f'Affine M ({affine_name}): angle={np.degrees(np.arctan2(M[1,0], M[0,0])):.2f}°, '
        f'scale={np.sqrt(M[0,0]**2 + M[1,0]**2):.4f}')

    # ── Load DAPI ──
    log(f'Loading DAPI: {fluo_path}')
    with tifffile.TiffFile(fluo_path) as t:
        # If multi-page, find best channel (highest signal)
        if len(t.pages) > 1:
            best_ch, best_mean = 0, 0
            for ch in range(len(t.pages)):
                pg = t.pages[ch].asarray()
                m = pg.mean()
                if m > best_mean:
                    best_mean = m
                    best_ch = ch
            log(f'  Multi-page ({len(t.pages)} channels), best ch={best_ch}')
            fl = t.pages[best_ch].asarray()
        else:
            fl = t.pages[0].asarray()
    fl_h, fl_w = fl.shape[:2]
    log(f'  DAPI: {fl_w}x{fl_h}')

    # ── Step 1: DAPI → contour space (resize) ──
    log(f'Step 1: Resize DAPI ({fl_w}x{fl_h}) to contour space ({fw}x{fh})')
    dapi_in_contour = cv2.resize(norm8(fl.astype(np.float32)), (fw, fh))
    del fl

    # ── Step 2: contour → HE (warpAffine with M) ──
    with tifffile.TiffFile(he_path) as t:
        he_h, he_w = t.pages[0].shape[:2]
    log(f'Step 2: Warp to HE space ({he_w}x{he_h}) using {affine_name}')

    # Work at display resolution
    scale = display_size / max(he_h, he_w)
    he_out_w, he_out_h = int(he_w * scale), int(he_h * scale)

    dapi_sc = cv2.resize(dapi_in_contour, (int(fw * scale), int(fh * scale)))
    del dapi_in_contour

    M_sc = M.copy()
    M_sc[:, 2] *= scale

    warped = cv2.warpAffine(dapi_sc, M_sc, (he_out_w, he_out_h))
    log(f'  Warped: {he_out_w}x{he_out_h}, coverage={((warped>30).sum()/warped.size*100):.1f}%')

    # ── Overlay ──
    log('Creating overlay...')
    he_img = cv2.resize(
        tifffile.imread(he_path)[..., :3] if tifffile.imread(he_path).ndim == 3
        else cv2.cvtColor(tifffile.imread(he_path), cv2.COLOR_GRAY2RGB),
        (he_out_w, he_out_h)
    )

    ov = he_img.copy()
    mask_ov = warped > 30
    ov[mask_ov] = (ov[mask_ov] * 0.6 + np.array([0, 255, 0], dtype=np.uint8) * 0.4).astype(np.uint8)
    overlay_path = os.path.join(output_dir, 'overlay_dapi.png')
    cv2.imwrite(overlay_path, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    result = {
        'method': 'dapi_via_contour',
        'fluo_path': fluo_path,
        'he_path': he_path,
        'affine_used': affine_name,
        'filled_size': [fw, fh],
        'dapi_size': [fl_w, fl_h],
        'he_size': [he_w, he_h],
    }
    with open(os.path.join(output_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    log(f'Saved: {overlay_path}')
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Register DAPI to HE via boundary contour alignment')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sample_dir', help='Sample directory (auto-detect files)')
    group.add_argument('--he', help='HE image path')

    parser.add_argument('--fluo', help='DAPI/morphology path')
    parser.add_argument('--boundary_dir', help='Boundary pipeline output directory')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--affine', default='affine_M.npy',
                        help='Affine matrix filename (default: affine_M.npy)')
    args = parser.parse_args()

    if args.sample_dir:
        sd = args.sample_dir
        he = os.path.join(sd, 'he_image.ome.tif')
        boundary_dir = os.path.join(sd, 'output')
        output_dir = args.output_dir or os.path.join(sd, 'output_dapi_via_contour')

        # Find DAPI: prefer morphology_focus, fallback to morphology.ome.tif
        focus_dir = os.path.join(sd, 'morphology_focus')
        if os.path.isdir(focus_dir):
            # Find best channel in morphology_focus
            focus_files = sorted([f for f in os.listdir(focus_dir) if f.endswith('.ome.tif')])
            if focus_files:
                fluo = os.path.join(focus_dir, focus_files[1] if len(focus_files) > 1 else focus_files[0])
            else:
                fluo = os.path.join(sd, 'morphology.ome.tif')
        else:
            fluo = os.path.join(sd, 'morphology.ome.tif')
    else:
        he = args.he
        fluo = args.fluo
        boundary_dir = args.boundary_dir
        output_dir = args.output_dir or 'output_dapi_via_contour'

    print(f'HE: {he}')
    print(f'DAPI: {fluo}')
    print(f'Boundary: {boundary_dir}')
    print(f'Output: {output_dir}')
    print()

    register_dapi_via_contour(he, fluo, boundary_dir, output_dir,
                               affine_name=args.affine)


if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    main()
