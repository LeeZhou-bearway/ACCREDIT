#!/usr/bin/env python3
"""
run_unified_agent.py - ACCREDIT: Agent-based Cell-resolved Cross-modal
Registration Engine with Dynamic Iterative Tuning.

Unified entry point for all registration pipelines.

Supports 6 modalities:
  1. boundary  — Xenium parquet contours → HE
  2. dapi      — DAPI/morphology fluorescence → HE
  3. codex     — Multi-channel CODEX protein → HE
  4. he-ihc    — IHC adjacent section → HE
  5. he-he     — HE cross-platform → HE
  6. st-he     — Spatial transcriptomics → HE

The agent auto-detects modality from input files, or accepts explicit --mode.

Usage:
    # Auto-detect modality
    python run_unified_agent.py \
        --he data/he_image.ome.tif \
        --moving data/morphology.ome.tif \
        --output_dir output/

    # Explicit modality
    python run_unified_agent.py \
        --he data/he_image.ome.tif \
        --moving data/nucleus_boundaries.parquet \
        --mode boundary \
        --output_dir output/

    # Sample directory (auto-detect all available data)
    python run_unified_agent.py \
        --sample_dir benchmark_data/DAPI/Xenium/Xenium_Prime_Breast_Cancer_FFPE \
        --output_dir output/

    # CODEX with explicit mode
    python run_unified_agent.py \
        --he data/HE.tif \
        --moving data/CODEX.ome.tif \
        --mode codex \
        --output_dir output/
"""
import argparse
import json
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
for _sd in ['pipelines/dapi', 'pipelines/codex', 'pipelines/boundary',
            'pipelines/st_he', 'pipelines/fluo_general',
            'pipelines/he_ihc',
            'quality', 'registration']:
    _p = os.path.join(_PROJECT_ROOT, _sd)
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Stain Detection (content-based) ─────────────────────────────────────────

def _load_thumbnail(path, target_px=1024):
    """Load a small RGB thumbnail from SVS/TIFF for stain analysis."""
    try:
        import openslide
        s = openslide.OpenSlide(path)
        w0, h0 = s.level_dimensions[0]
        scale = target_px / max(w0, h0)
        tw, th = int(w0 * scale), int(h0 * scale)
        thumb = s.get_thumbnail((tw, th))
        s.close()
        import numpy as np
        return np.array(thumb.convert('RGB'))
    except Exception:
        pass
    try:
        import tifffile, cv2
        import numpy as np
        img = tifffile.imread(path)
        if img.ndim == 3 and img.shape[0] in (3, 4):
            img = np.moveaxis(img, 0, -1)
        img = img[..., :3]
        scale = target_px / max(img.shape[:2])
        return cv2.resize(img, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    except Exception:
        return None


def analyze_image(path):
    """Analyze image content to determine its type.

    Loads a thumbnail and characterizes the image by:
      - Color space: brightfield (RGB tissue) vs fluorescence (dark background)
      - Stain type:  HE (Eosin-dominant) vs IHC (DAB-dominant) for brightfield
      - Channel count: single vs multi-channel for fluorescence

    Returns dict:
      {
        'imaging': 'brightfield' | 'fluorescence',
        'stain':   'he' | 'ihc' | None,
        'n_channels': int,
        'thumbnail': np.ndarray or None,
      }
    """
    import numpy as np

    result = {'imaging': None, 'stain': None, 'n_channels': 1, 'thumbnail': None}

    # Channel count from TIFF metadata
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff', '.ome.tif'):
        try:
            import tifffile
            with tifffile.TiffFile(path) as t:
                result['n_channels'] = len(t.pages)
        except Exception:
            pass

    # Load thumbnail for content analysis
    thumb = _load_thumbnail(path, target_px=512)
    if thumb is None:
        return result
    result['thumbnail'] = thumb

    gray = np.mean(thumb, axis=2)
    tissue_mask = gray < 220
    dark_mask = gray < 30

    if tissue_mask.sum() < 500:
        return result

    dark_ratio = dark_mask.sum() / tissue_mask.sum()

    # Fluorescence: mostly dark background with sparse bright signal
    if dark_ratio > 0.6:
        result['imaging'] = 'fluorescence'
        return result

    # Brightfield: distinguish HE vs IHC via HED color deconvolution
    result['imaging'] = 'brightfield'
    from skimage.color import rgb2hed
    hed = rgb2hed(np.clip(thumb.astype(np.float64) / 255, 1e-10, 1.0))

    eosin_signal = hed[:, :, 1][tissue_mask]
    dab_signal = hed[:, :, 2][tissue_mask]

    eosin_strong = np.mean(eosin_signal > 0.1)
    dab_strong = np.mean(dab_signal > 0.1)

    if dab_strong > 0.05 and dab_strong > eosin_strong * 0.3:
        result['stain'] = 'ihc'
    else:
        result['stain'] = 'he'

    return result


# ── Modality Detection ───────────────────────────────────────────────────────

def detect_modality(he_path, moving_path, sample_dir=None):
    """Auto-detect registration modality from input files.

    Returns: (modality, he_path, moving_path, extra_args)
    """
    extra = {}

    if sample_dir:
        he_path = he_path or os.path.join(sample_dir, 'he_image.ome.tif')
        # Check what data is available
        has_parquet = os.path.exists(os.path.join(sample_dir, 'nucleus_boundaries.parquet'))
        has_morphology = os.path.exists(os.path.join(sample_dir, 'morphology.ome.tif'))
        has_focus = os.path.isdir(os.path.join(sample_dir, 'morphology_focus'))
        has_boundary_result = os.path.exists(os.path.join(sample_dir, 'output', 'affine_M.npy'))

        if has_parquet and not moving_path:
            moving_path = os.path.join(sample_dir, 'nucleus_boundaries.parquet')
        if has_morphology and not moving_path:
            moving_path = os.path.join(sample_dir, 'morphology.ome.tif')

        extra['has_parquet'] = has_parquet
        extra['has_morphology'] = has_morphology
        extra['has_focus'] = has_focus
        extra['has_boundary_result'] = has_boundary_result
        extra['sample_dir'] = sample_dir

    if not moving_path:
        raise ValueError('No moving image specified and could not auto-detect')

    ext = os.path.splitext(moving_path)[1].lower()
    basename = os.path.basename(moving_path).lower()

    # ── Non-image formats (detected by extension alone) ──
    if ext == '.parquet':
        return 'boundary', he_path, moving_path, extra

    if ext == '.png' and ('spot' in basename or 'st_' in basename):
        return 'st-he', he_path, moving_path, extra

    # ── Image formats: read data and analyze content ──
    info = analyze_image(moving_path)
    extra['image_info'] = info

    # Filename hints for IHC (fast path before content analysis)
    ihc_keywords = ['p40', 'ihc', 'ki67', 'cd3', 'cd8', 'cd20', 'cd68',
                    'ck', 'pdl1', 'pd-l1', 'her2', 'er_', 'pr_', 'dab',
                    'chromogen', 'sox10', 'panck', 'vimentin', 'sma']
    name_hints_ihc = any(kw in basename for kw in ihc_keywords)

    # Brightfield images (SVS, TIFF, etc.)
    if info['imaging'] == 'brightfield':
        if name_hints_ihc or info['stain'] == 'ihc':
            return 'he-ihc', he_path, moving_path, extra
        if ext in ('.svs',):
            return 'he-he', he_path, moving_path, extra
        # Brightfield TIFF — likely HE or IHC
        if info['stain'] == 'ihc':
            return 'he-ihc', he_path, moving_path, extra
        return 'he-he', he_path, moving_path, extra

    # Fluorescence images
    if info['imaging'] == 'fluorescence':
        n_ch = info['n_channels']
        if 'codex' in basename:
            return 'codex', he_path, moving_path, extra
        if n_ch > 5:
            return 'codex', he_path, moving_path, extra
        return 'dapi', he_path, moving_path, extra

    # Fallback: use extension + filename heuristics
    if ext in ('.svs',):
        if name_hints_ihc:
            return 'he-ihc', he_path, moving_path, extra
        return 'he-he', he_path, moving_path, extra

    if ext in ('.tif', '.tiff'):
        return 'dapi', he_path, moving_path, extra

    return 'dapi', he_path, moving_path, extra


# ── Pipeline Runners ─────────────────────────────────────────────────────────

def run_boundary(he_path, moving_path, output_dir, extra, log=print):
    """Run boundary pipeline (parquet → HE)."""
    log('Pipeline: Boundary (Xenium Parquet → HE)')
    from run_boundary_pipeline import run_boundary_pipeline
    return run_boundary_pipeline(
        he_path, moving_path, output_dir,
        cell_parquet_path=extra.get('cell_parquet_path'),
        log=log)


def run_dapi(he_path, moving_path, output_dir, extra, log=print):
    """Run DAPI pipeline with automatic method selection.

    Strategy:
    1. If boundary result exists → try contour chain first (fastest, most accurate)
    2. ROI+flip pipeline (run_dapi_ROI_flip_pipeline.py)
    3. If QCS < threshold → try with enhancement
    4. Select best by QCS
    """
    log('Pipeline: DAPI (Fluorescence → HE)')

    sample_dir = extra.get('sample_dir', os.path.dirname(moving_path))
    has_boundary = extra.get('has_boundary_result', False)
    results = []

    # ── Method 1: Contour chain (if boundary result available) ──
    if has_boundary:
        log('\n--- Method 1: Contour Chain (via boundary M) ---')
        try:
            from run_dapi_via_contour import register_dapi_via_contour
            boundary_dir = os.path.join(sample_dir, 'output')
            contour_out = os.path.join(output_dir, 'contour_chain')
            result = register_dapi_via_contour(
                he_path, moving_path, boundary_dir, contour_out, log=log)
            if result and result.get('qcs', 0) > 0:
                results.append(('contour_chain', result.get('qcs', 0), contour_out, result))
                log(f'  Contour chain QCS: {result.get("qcs", 0):.4f}')
        except Exception as e:
            log(f'  Contour chain failed: {e}')

    # ── Method 2: ROI+flip pipeline ──
    log('\n--- Method 2: ROI+flip (run_dapi_ROI_flip_pipeline) ---')
    try:
        from run_dapi_ROI_flip_pipeline import register_one_roi
        roi_out = os.path.join(output_dir, 'roi_flip')
        result = register_one_roi(he_path, moving_path, roi_out, log=log)
        if result and result.get('qcs', 0) > 0:
            results.append(('roi_flip', result.get('qcs', 0), roi_out, result))
            log(f'  ROI+flip QCS: {result.get("qcs", 0):.4f}')

            # ── Method 2b: Enhancement rescue if QCS low ──
            if result.get('qcs', 0) < 0.65:
                log('\n--- Method 2b: ROI+flip + enhancement ---')
                enh_out = os.path.join(output_dir, 'roi_flip_enh')
                result_enh = register_one_roi(he_path, moving_path, enh_out,
                                               enhance=True, log=log)
                if result_enh and result_enh.get('qcs', 0) > 0:
                    results.append(('roi_flip_enh', result_enh.get('qcs', 0),
                                   enh_out, result_enh))
                    log(f'  ROI+flip+enh QCS: {result_enh.get("qcs", 0):.4f}')
    except Exception as e:
        log(f'  ROI+flip failed: {e}')

    # ── Select best ──
    if not results:
        log('ERROR: All DAPI methods failed')
        return None

    results.sort(key=lambda x: -x[1])
    best_name, best_qcs, best_dir, best_result = results[0]
    log(f'\nBest: {best_name} (QCS={best_qcs:.4f})')

    # Copy best overlay to output root
    import shutil
    for f in ['overlay_dapi.png', 'overlay_dapi_roi.png', 'result.json']:
        src = os.path.join(best_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, f))

    return best_result


def run_codex(he_path, moving_path, output_dir, extra, log=print):
    """Run CODEX pipeline (multi-channel protein → HE, QCS-based channel selection)."""
    log('Pipeline: CODEX (Multi-channel Protein → HE)')
    from run_fluo2he_roi import register_one_roi as register_codex_roi
    import tifffile
    he_img = tifffile.imread(he_path)
    return register_codex_roi(he_img, moving_path, output_dir, log=log)


def run_he_ihc(he_path, moving_path, output_dir, extra, log=print):
    """Run HE-IHC pipeline (LightGlue + DHR nonrigid + Demons)."""
    log('Pipeline: HE-IHC (LightGlue + DHR + Demons)')
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'pipelines/he_ihc'))
    from run_unified_he_ihc import run_unified_he_ihc
    return run_unified_he_ihc(he_path, moving_path, output_dir, log=log)


def run_he_he(he_path, moving_path, output_dir, extra, log=print):
    """Run HE-HE pipeline (Feature + Similarity + BSpline + Tile)."""
    log('Pipeline: HE-HE (Cross-platform)')
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'pipelines/he_he'))
    from run_he2he import register_one
    return register_one(he_path, moving_path, output_dir, log=log)


def run_st_he(he_path, moving_path, output_dir, extra, log=print):
    """Run ST-HE pipeline."""
    log('Pipeline: ST-HE (Spatial Transcriptomics)')
    # ST pipeline is standalone
    import subprocess
    result = subprocess.run([
        sys.executable, 'run_st2he_test.py',
        '--st', moving_path, '--he', he_path,
        '--output_dir', output_dir
    ], capture_output=True, text=True)
    log(result.stdout)
    if result.returncode != 0:
        log(f'ST-HE failed: {result.stderr}')
    rp = os.path.join(output_dir, 'result.json')
    if os.path.exists(rp):
        return json.load(open(rp))
    return None


# ── ROI+flip helper (not modifying existing scripts) ─────────────────────────


# ── Pipeline Router ──────────────────────────────────────────────────────────

def run_senior(he_path, moving_path, output_dir, extra, log=print):
    """Senior pipeline for hard partial-match Xenium samples.

    Uses senior_plus (ROI flip + BSpline + NCC angle fine-tune + ECC)
    for Step 1, then boundary chain + DAPI rescue.

    Requires both DAPI MIP and parquet — pass parquet via extra['parquet_path']
    or auto-detect adjacent to DAPI.
    """
    parquet_path = extra.get('parquet_path')
    if not parquet_path:
        cand = os.path.join(os.path.dirname(moving_path), 'nucleus_boundaries.parquet')
        if os.path.exists(cand):
            parquet_path = cand
    if not parquet_path or not os.path.exists(parquet_path):
        log('  ERROR: senior pipeline needs parquet (set extra[parquet_path])')
        return None

    sys.path.insert(0, os.path.join(_ACCREDIT_ROOT, 'pipelines/senior'))
    from run_senior_plus_pipeline import (
        step1_unified_agent, step2_boundary_chain, step3_warp_dapi, step4_dapi_rescue)
    import numpy as np

    init_res = step1_unified_agent(he_path, moving_path, output_dir,
                                    enhance=extra.get('enhance', False), log=log)
    if init_res is None:
        return None
    M_init = np.array(init_res['M_full'], dtype=np.float64)
    boundary_res, _ = step2_boundary_chain(he_path, parquet_path, M_init, output_dir, log=log)
    warped_dapi, he_small, _ = step3_warp_dapi(
        he_path, moving_path, M_init, output_dir,
        he_level=extra.get('he_level', 2), log=log)
    rescue_res = step4_dapi_rescue(warped_dapi, he_small, output_dir, log=log)
    return {
        'qcs': init_res.get('qcs'),
        'dice': boundary_res.get('dice') if boundary_res else None,
        'step1_init_M': M_init.tolist(),
        'step4_rescue': rescue_res,
    }


_ACCREDIT_ROOT = os.path.dirname(os.path.abspath(__file__))


PIPELINE_MAP = {
    'boundary': run_boundary,
    'dapi': run_dapi,
    'codex': run_codex,
    'he-ihc': run_he_ihc,
    'he-he': run_he_he,
    'st-he': run_st_he,
    'senior': run_senior,
}


MODALITY_DESCRIPTIONS = {
    'boundary': 'Boundary (Xenium parquet contours → HE)',
    'dapi':     'DAPI (fluorescence morphology → HE)',
    'codex':    'CODEX (multi-channel protein fluorescence → HE)',
    'he-ihc':   'HE-IHC (IHC adjacent section → HE, LightGlue + DHR + Demons)',
    'he-he':    'HE-HE (cross-platform / cross-scanner HE → HE)',
    'st-he':    'ST-HE (spatial transcriptomics → HE)',
    'senior':   'Senior (hard partial-match Xenium, multi-step)',
}


def report_detection(he_path, moving_path, modality, extra, log=print):
    """Report modality detection result to user with analysis details."""
    info = extra.get('image_info', {})
    desc = MODALITY_DESCRIPTIONS.get(modality, modality)

    log('=' * 60)
    log('ACCREDIT — Modality Detection Report')
    log('=' * 60)
    log(f'  Fixed (HE):  {he_path}')
    log(f'  Moving:      {moving_path}')
    log('')
    log(f'  Detected modality:  {modality}')
    log(f'  Description:        {desc}')
    log('')
    if info:
        log('  Image analysis:')
        if info.get('imaging'):
            log(f'    Imaging type:   {info["imaging"]}')
        if info.get('stain'):
            log(f'    Stain type:     {info["stain"].upper()}')
        if info.get('n_channels', 1) > 1:
            log(f'    Channels:       {info["n_channels"]}')
    log('=' * 60)


def run_unified(he_path, moving_path, output_dir, mode=None, sample_dir=None,
                extra=None, log=print, interactive=True):
    """
    Unified registration entry point.

    Auto-detects modality if mode is not specified.
    When interactive=True (default), reports detection and asks user to confirm
    before running registration. Set interactive=False for batch/programmatic use.
    extra: dict with pipeline-specific params (cell_parquet_path, enhance, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()
    log_lines = []

    def _log(msg):
        log_lines.append(msg)
        log(msg)

    caller_extra = extra or {}

    # Detect modality
    if mode:
        modality = mode
        extra = {'sample_dir': sample_dir} if sample_dir else {}
    else:
        modality, he_path, moving_path, extra = detect_modality(
            he_path, moving_path, sample_dir)

    extra.update(caller_extra)

    # ── Report detection and ask for confirmation ──
    report_detection(he_path, moving_path, modality, extra, log=_log)

    if interactive:
        _log('')
        _log(f'Available modalities: {", ".join(PIPELINE_MAP.keys())}')
        _log('')
        reply = input('Proceed with this modality? '
                      '[Y/n/modality name to override]: ').strip()
        if reply.lower() in ('n', 'no', 'q', 'quit'):
            _log('Aborted by user.')
            return None
        if reply and reply.lower() not in ('y', 'yes', ''):
            if reply in PIPELINE_MAP:
                _log(f'User override: {modality} → {reply}')
                modality = reply
            else:
                _log(f'Unknown modality "{reply}". Aborting.')
                return None

    _log('')
    _log(f'Starting registration: {modality}')
    _log(f'  Output: {output_dir}')
    _log('')

    # Run built-in pipeline
    pipeline_fn = PIPELINE_MAP.get(modality)
    if pipeline_fn is None:
        _log(f'ERROR: Unknown modality "{modality}"')
        return None

    try:
        result = pipeline_fn(he_path, moving_path, output_dir, extra, log=_log)
    except Exception as e:
        import traceback
        _log(f'ERROR: {e}')
        _log(traceback.format_exc())
        result = None

    builtin_qcs = result.get('qcs', 0) if result else 0

    # Run plugins (if any registered for this modality)
    try:
        from plugins import get_plugins_for_modality
        plugins = get_plugins_for_modality(modality)
        if plugins:
            _log(f'\n--- Running {len(plugins)} plugin(s) ---')
        for mod, info in plugins:
            pname = info['name']
            _log(f'  Plugin: {pname} (v{info.get("version","?")})')
            try:
                plugin_out = os.path.join(output_dir, f'plugin_{pname.replace(" ", "_").lower()}')
                plugin_result = mod.register(
                    he_path=he_path, moving_path=moving_path,
                    output_dir=plugin_out, modality=modality,
                    log=_log, **extra)
                if plugin_result and plugin_result.get('qcs', 0) > builtin_qcs:
                    _log(f'  Plugin {pname} QCS={plugin_result["qcs"]:.4f} > built-in {builtin_qcs:.4f} → adopted')
                    result = plugin_result
                    builtin_qcs = plugin_result['qcs']
                    # Copy best to output root
                    import shutil
                    for f in ['overlay_dapi.png', 'result.json']:
                        src = os.path.join(plugin_out, f)
                        if os.path.exists(src):
                            shutil.copy2(src, os.path.join(output_dir, f))
                elif plugin_result:
                    _log(f'  Plugin {pname} QCS={plugin_result.get("qcs",0):.4f} <= built-in → skipped')
                else:
                    _log(f'  Plugin {pname} returned None')
            except Exception as e:
                _log(f'  Plugin {pname} failed: {e}')
    except ImportError:
        pass  # No plugins directory

    # ── Agent Rescue Gate ────────────────────────────────────────────────────
    RESCUE_THRESHOLD = 0.65
    # Only rescue modalities that don't already use a full agent
    RESCUE_MODALITIES = {'dapi', 'codex', 'boundary', 'st-he'}

    if (modality in RESCUE_MODALITIES and
            builtin_qcs < RESCUE_THRESHOLD and builtin_qcs > 0):
        _log(f'\nQCS={builtin_qcs:.4f} < {RESCUE_THRESHOLD}: invoking rescue agent...')
        try:
            from agent.agent import run_rescue_agent
            rescue_dir = os.path.join(output_dir, 'rescue')
            rescue_result = run_rescue_agent(
                mode=modality,
                he_path=he_path,
                moving_path=moving_path,
                output_dir=rescue_dir,
                initial_result=result,
                extra=extra,
                log=_log,
            )
            rescue_qcs = rescue_result.get('qcs', 0) if rescue_result else 0
            if rescue_qcs > builtin_qcs:
                _log(f'Rescue improved QCS: {builtin_qcs:.4f} → {rescue_qcs:.4f}')
                result = rescue_result
                builtin_qcs = rescue_qcs
            else:
                _log(f'Rescue did not improve. Keeping QCS={builtin_qcs:.4f}')
        except Exception as e:
            _log(f'Rescue agent failed: {e}')

    # ── User Plugin Gate ────────────────────────────────────────────────────
    # After all automatic methods, check if user has a custom plugin result.
    # If the user manually ran a custom method and saved result to output_dir/user_plugin/,
    # evaluate it and adopt if better.
    user_plugin_dir = os.path.join(output_dir, 'user_plugin')
    user_result_path = os.path.join(user_plugin_dir, 'result.json')
    if os.path.exists(user_result_path):
        try:
            with open(user_result_path) as f:
                user_result = json.load(f)
            user_qcs = user_result.get('qcs', 0)
            if user_qcs > builtin_qcs:
                _log(f'\nUser plugin QCS={user_qcs:.4f} > current {builtin_qcs:.4f} → adopted!')
                result = user_result
                builtin_qcs = user_qcs
                # Learn from user's method: save to learned_strategies
                _learn_user_strategy(modality, user_result, output_dir, _log)
            else:
                _log(f'\nUser plugin QCS={user_qcs:.4f} <= current {builtin_qcs:.4f} → skipped')
        except Exception as e:
            _log(f'\nUser plugin read failed: {e}')

    # Summary
    elapsed = time.time() - t_start
    _log('')
    _log('=' * 60)
    if result:
        qcs = result.get('qcs', result.get('quality_metrics', {}).get('qcs', 'N/A'))
        _log(f'DONE: {modality} pipeline, QCS={qcs}, time={elapsed:.0f}s')
    else:
        _log(f'FAILED: {modality} pipeline, time={elapsed:.0f}s')
    _log('=' * 60)

    # Save log
    with open(os.path.join(output_dir, 'unified_agent.log'), 'w') as f:
        f.write('\n'.join(log_lines))

    return result


def _learn_user_strategy(modality, user_result, output_dir, log=print):
    """Learn from a successful user-provided strategy.

    When the user's custom method beats all automatic methods, record it
    so the rescue agent can try it on future similar cases.

    Saved to ACCREDIT/agent/learned_strategies.json (accumulates over time).
    """
    strategy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'agent', 'learned_strategies.json')
    # Load existing
    strategies = []
    if os.path.exists(strategy_file):
        try:
            with open(strategy_file) as f:
                strategies = json.load(f)
        except Exception:
            strategies = []

    # Record new strategy
    import datetime
    entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'modality': modality,
        'qcs': user_result.get('qcs', 0),
        'method': user_result.get('method', 'user_custom'),
        'description': user_result.get('description', ''),
        'parameters': {k: v for k, v in user_result.items()
                      if k in ('flip', 'rotation', 'pixel_ratio', 'best_channel',
                               'enhance', 'gamma', 'scale')},
        'output_dir': output_dir,
    }
    strategies.append(entry)

    with open(strategy_file, 'w') as f:
        json.dump(strategies, f, indent=2)

    log(f'  Learned user strategy: {entry["method"]} (QCS={entry["qcs"]:.4f})')
    log(f'  Saved to {strategy_file} ({len(strategies)} strategies total)')


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='ACCREDIT: Agent-based Cell-resolved Cross-modal Registration Engine with Dynamic Iterative Tuning')
    parser.add_argument('--he', '--fixed', nargs='+', dest='he',
                        help='HE image path(s). Multiple paths for batch mode.')
    parser.add_argument('--moving', nargs='+',
                        help='Moving image path(s) (DAPI/CODEX/IHC/parquet/etc)')
    parser.add_argument('--sample_dir', help='Sample directory (auto-detect files)')
    parser.add_argument('--mode', choices=list(PIPELINE_MAP.keys()),
                        help='Force modality (default: auto-detect)')
    parser.add_argument('--output_dir', default='output', help='Output directory')
    # Boundary-specific
    parser.add_argument('--parquet', nargs='+', default=None,
                        help='Nucleus boundaries parquet(s) (shortcut for boundary mode)')
    parser.add_argument('--cell-parquet', nargs='+', default=None,
                        help='Cell boundaries parquet(s) (optional, auto-detected if omitted)')
    parser.add_argument('--metric', choices=['dice', 'iou'], default='dice',
                        help='Quality metric for boundary gating (default: dice)')
    parser.add_argument('--refine', action='store_true',
                        help='Enable refine steps for boundary pipeline')
    # Senior-specific
    parser.add_argument('--enhance', action='store_true',
                        help='Apply gamma enhancement (senior/dapi)')
    parser.add_argument('--he_level', type=int, default=2,
                        help='HE pyramid level for senior pipeline (default: 2)')
    # LLM agent
    parser.add_argument('--api-key', default=None,
                        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip modality confirmation (non-interactive)')
    args = parser.parse_args()

    if not args.he and not args.sample_dir:
        parser.error('Either --he or --sample_dir is required')

    if args.api_key:
        os.environ['ANTHROPIC_API_KEY'] = args.api_key

    # --parquet implies boundary mode and serves as --moving
    if args.parquet and not args.moving:
        args.moving = args.parquet
        if not args.mode:
            args.mode = 'boundary'

    # ── Single vs batch ──
    he_list = args.he or [None]
    moving_list = args.moving or [None]
    cell_parquets = args.cell_parquet or []

    if len(he_list) == 1 and len(moving_list) == 1:
        # Single mode
        extra = {}
        if cell_parquets:
            extra['cell_parquet_path'] = cell_parquets[0]
        if args.enhance:
            extra['enhance'] = True
        if args.he_level != 2:
            extra['he_level'] = args.he_level
        if args.parquet and len(args.parquet) == 1:
            extra['parquet_path'] = args.parquet[0]

        result = run_unified(
            he_path=he_list[0],
            moving_path=moving_list[0],
            output_dir=args.output_dir,
            mode=args.mode,
            sample_dir=args.sample_dir,
            extra=extra,
            interactive=not args.yes,
        )
        if result:
            print(f'\nResult: {json.dumps(result, indent=2, default=str)[:500]}')
    else:
        # Batch mode
        if len(he_list) != len(moving_list):
            parser.error('--he and --moving must have the same number of arguments in batch mode')
        if cell_parquets and len(cell_parquets) != len(he_list):
            parser.error('--cell-parquet must match --he count')

        # Auto-detect cell parquets if not provided
        if not cell_parquets and args.mode == 'boundary':
            for mv in moving_list:
                cell_path = os.path.join(os.path.dirname(mv), 'cell_boundaries.parquet')
                cell_parquets.append(cell_path if os.path.exists(cell_path) else None)

        results = {}
        for i, (he, mv) in enumerate(zip(he_list, moving_list)):
            dataset_name = os.path.basename(os.path.dirname(he))
            out_dir = os.path.join(args.output_dir, dataset_name) if len(he_list) > 1 else args.output_dir

            extra = {}
            if cell_parquets and i < len(cell_parquets) and cell_parquets[i]:
                extra['cell_parquet_path'] = cell_parquets[i]
            if args.enhance:
                extra['enhance'] = True

            print(f'\n{"="*60}')
            print(f'[{i+1}/{len(he_list)}] {dataset_name}')
            print(f'{"="*60}')

            result = run_unified(
                he_path=he, moving_path=mv,
                output_dir=out_dir, mode=args.mode, extra=extra,
                interactive=False,
            )
            results[dataset_name] = result

        # Batch summary
        print(f'\n{"="*60}')
        print('Batch Summary')
        print(f'{"="*60}')
        for name, r in results.items():
            if r:
                qcs = r.get('qcs', r.get('quality_metrics', {}).get('qcs', 'N/A'))
                print(f'  {name}: QCS={qcs}')
            else:
                print(f'  {name}: FAILED')


if __name__ == '__main__':
    main()
