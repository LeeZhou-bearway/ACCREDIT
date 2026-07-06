"""
agent/rescue_tools.py — Rescue agent tool definitions for all modalities.

Each tool is a thin wrapper calling existing pipeline functions.
No new registration logic — just exposes pipeline functions as agent tools.
"""
import json
import os
import sys
import numpy as np

# Add project directories to path (relative to this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_subdirs = ['pipelines/dapi', 'pipelines/codex', 'pipelines/boundary',
            'pipelines/st_he', 'pipelines/fluo_general',
            'quality', 'registration']
for _sd in [PROJECT_ROOT] + [os.path.join(PROJECT_ROOT, s) for s in _subdirs]:
    if _sd not in sys.path:
        sys.path.insert(0, _sd)


# ── Tool JSON Schemas (Claude API format) ────────────────────────────────────

COMMON_TOOLS = [
    {
        "name": "read_result_json",
        "description": "Read result.json and QCS metrics from a pipeline output directory. "
                       "Returns all metrics, method used, and file paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "output_dir": {"type": "string", "description": "Path to pipeline output directory"}
            },
            "required": ["output_dir"]
        }
    },
    {
        "name": "read_qcs_breakdown",
        "description": "Compute full QCS breakdown (tissue_dice, boundary_dice, hausdorff, "
                       "inlier_ratio, composite QCS) from HE and warped fluo images. "
                       "Use this to diagnose which metric is low.",
        "input_schema": {
            "type": "object",
            "properties": {
                "he_path": {"type": "string", "description": "Path to HE image"},
                "overlay_dir": {"type": "string", "description": "Directory containing overlay and result"}
            },
            "required": ["he_path", "overlay_dir"]
        }
    },
    {
        "name": "compare_results",
        "description": "Compare two result directories by QCS. Returns which is better and by how much.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dir_a": {"type": "string", "description": "First result directory"},
                "dir_b": {"type": "string", "description": "Second result directory"}
            },
            "required": ["dir_a", "dir_b"]
        }
    },
    {
        "name": "finish_rescue",
        "description": "MANDATORY last step. Write final rescue result.json with original and rescue QCS. "
                       "Copies best overlay to output root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "output_dir": {"type": "string", "description": "Rescue output directory"},
                "best_result_dir": {"type": "string", "description": "Directory with best result"},
                "original_qcs": {"type": "number", "description": "QCS from original pipeline"},
                "rescue_qcs": {"type": "number", "description": "QCS from best rescue attempt"},
                "rescue_method": {"type": "string", "description": "Method that achieved best rescue QCS"}
            },
            "required": ["output_dir", "best_result_dir", "original_qcs", "rescue_qcs", "rescue_method"]
        }
    },
]

DAPI_TOOLS = [
    {
        "name": "run_dapi_roi_flip",
        "description": "Run DAPI ROI+flip pipeline. Searches 4 flips × 4 rotations × 6 pixel_ratios. "
                       "Includes NMI + NCC angle tune + ECC + BSpline. "
                       "Set enhance=true for weak DAPI signal (gamma=0.5).",
        "input_schema": {
            "type": "object",
            "properties": {
                "he_path": {"type": "string"},
                "fluo_path": {"type": "string"},
                "output_dir": {"type": "string"},
                "enhance": {"type": "boolean", "default": False,
                           "description": "Apply gamma=0.5 enhancement for weak signal"}
            },
            "required": ["he_path", "fluo_path", "output_dir"]
        }
    },
    {
        "name": "run_dapi_contour_chain",
        "description": "Register DAPI to HE via boundary contour alignment. "
                       "Requires boundary pipeline output (filled_image.png + affine_M.npy). "
                       "Best for samples with very weak DAPI signal.",
        "input_schema": {
            "type": "object",
            "properties": {
                "he_path": {"type": "string"},
                "fluo_path": {"type": "string"},
                "boundary_dir": {"type": "string", "description": "Directory with boundary pipeline output"},
                "output_dir": {"type": "string"}
            },
            "required": ["he_path", "fluo_path", "boundary_dir", "output_dir"]
        }
    },
]

CODEX_TOOLS = [
    {
        "name": "run_codex_registration",
        "description": "Run CODEX ROI registration with QCS-based channel selection. "
                       "Optionally force a specific channel. "
                       "Enhancement: 0=none, 1=CODEX gamma, 2=CODEX gamma + HE CLAHE.",
        "input_schema": {
            "type": "object",
            "properties": {
                "he_path": {"type": "string"},
                "codex_path": {"type": "string"},
                "output_dir": {"type": "string"},
                "force_channel": {"type": "integer", "description": "Force specific channel (e.g. 0 for nuclear)"},
                "enhance": {"type": "integer", "default": 0, "description": "0=none, 1=CODEX gamma, 2=+HE CLAHE"}
            },
            "required": ["he_path", "codex_path", "output_dir"]
        }
    },
    {
        "name": "run_codex_rescue_rotation",
        "description": "Run CODEX rotation rescue: NCC rotation scan (0-355°) + phase correlation + ECC. "
                       "Use when the image might be rotated.",
        "input_schema": {
            "type": "object",
            "properties": {
                "he_path": {"type": "string"},
                "codex_path": {"type": "string"},
                "output_dir": {"type": "string"},
                "best_channel": {"type": "integer", "description": "Channel to use for rescue"}
            },
            "required": ["he_path", "codex_path", "output_dir"]
        }
    },
    {
        "name": "list_codex_channels",
        "description": "Scan CODEX TIFF and report channel count and tissue coverage per channel. "
                       "Helps decide which channel to try.",
        "input_schema": {
            "type": "object",
            "properties": {
                "codex_path": {"type": "string"}
            },
            "required": ["codex_path"]
        }
    },
]

BOUNDARY_TOOLS = [
    {
        "name": "run_boundary_registration",
        "description": "Run full boundary pipeline: ORB feature matching + block refinement + auto rescue. "
                       "For Xenium parquet contours → HE.",
        "input_schema": {
            "type": "object",
            "properties": {
                "he_path": {"type": "string"},
                "parquet_path": {"type": "string"},
                "output_dir": {"type": "string"},
                "cell_parquet_path": {"type": "string", "description": "Optional cell_boundaries.parquet"}
            },
            "required": ["he_path", "parquet_path", "output_dir"]
        }
    },
    {
        "name": "run_boundary_rescue_ncc",
        "description": "Run boundary NCC rescue: HE nuclear channel vs cell density NCC rotation scan. "
                       "Use when ORB feature matching failed (few inliers).",
        "input_schema": {
            "type": "object",
            "properties": {
                "he_path": {"type": "string"},
                "filled_image_path": {"type": "string", "description": "Path to filled_image.png"},
                "output_dir": {"type": "string"},
                "current_dice": {"type": "number", "description": "Current global dice"}
            },
            "required": ["he_path", "filled_image_path", "output_dir"]
        }
    },
]

ST_TOOLS = [
    {
        "name": "run_st_registration",
        "description": "Run ST→HE registration via hole landmark matching + RANSAC + NMI.",
        "input_schema": {
            "type": "object",
            "properties": {
                "he_path": {"type": "string"},
                "st_path": {"type": "string"},
                "output_dir": {"type": "string"}
            },
            "required": ["he_path", "st_path", "output_dir"]
        }
    },
]

# All rescue tools combined
RESCUE_TOOLS = COMMON_TOOLS + DAPI_TOOLS + CODEX_TOOLS + BOUNDARY_TOOLS + ST_TOOLS

# Modality → tool name mapping
MODALITY_RESCUE_TOOLS = {
    'dapi':     {t['name'] for t in DAPI_TOOLS},
    'codex':    {t['name'] for t in CODEX_TOOLS},
    'boundary': {t['name'] for t in BOUNDARY_TOOLS},
    'st-he':    {t['name'] for t in ST_TOOLS},
    'he-ihc':   set(),
    'he-he':    set(),
}

COMMON_TOOL_NAMES = {t['name'] for t in COMMON_TOOLS}


def get_tools_for_modality(modality):
    """Get tool schemas for a specific modality (common + modality-specific)."""
    modality_names = MODALITY_RESCUE_TOOLS.get(modality, set()) | COMMON_TOOL_NAMES
    return [t for t in RESCUE_TOOLS if t['name'] in modality_names]


# ── Tool Implementations ─────────────────────────────────────────────────────

def _read_result_json(output_dir):
    """Read result.json from output directory."""
    candidates = ['result.json', 'result_contours.json', 'pipeline_summary.json']
    for fname in candidates:
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            # Flatten nested result (run_fluo2he_roi wraps in {name: {...}})
            if isinstance(data, dict) and len(data) == 1:
                key = list(data.keys())[0]
                if isinstance(data[key], dict) and 'dice' in data[key]:
                    data = data[key]
            return json.dumps({
                'status': 'ok',
                'file': fname,
                'data': data,
                'qcs': data.get('qcs', data.get('quality_metrics', {}).get('qcs', None)),
            }, default=str)
    return json.dumps({'status': 'error', 'message': f'No result.json found in {output_dir}'})


def _read_qcs_breakdown(he_path, overlay_dir):
    """Read QCS breakdown from existing result or recompute."""
    result_path = os.path.join(overlay_dir, 'result.json')
    if os.path.exists(result_path):
        with open(result_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and len(data) == 1:
            key = list(data.keys())[0]
            if isinstance(data[key], dict):
                data = data[key]
        qm = data.get('quality_metrics', {})
        if qm:
            return json.dumps({'status': 'ok', 'metrics': qm}, default=str)
    return json.dumps({'status': 'error', 'message': 'No quality metrics found'})


def _compare_results(dir_a, dir_b):
    """Compare two results by QCS."""
    def get_qcs(d):
        try:
            r = json.loads(_read_result_json(d))
            return r.get('qcs', 0) or 0
        except:
            return 0
    qcs_a = get_qcs(dir_a)
    qcs_b = get_qcs(dir_b)
    winner = 'a' if qcs_a >= qcs_b else 'b'
    return json.dumps({
        'qcs_a': qcs_a, 'qcs_b': qcs_b,
        'winner': winner,
        'winner_dir': dir_a if winner == 'a' else dir_b,
        'improvement': abs(qcs_a - qcs_b),
    })


def _finish_rescue(output_dir, best_result_dir, original_qcs, rescue_qcs, rescue_method):
    """Write final rescue result."""
    import shutil
    os.makedirs(output_dir, exist_ok=True)

    # Copy best result files to output root
    for fname in os.listdir(best_result_dir):
        src = os.path.join(best_result_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    # Write rescue summary
    summary = {
        'rescue_status': 'improved' if rescue_qcs > original_qcs else 'no_improvement',
        'original_qcs': original_qcs,
        'rescue_qcs': rescue_qcs,
        'rescue_method': rescue_method,
        'best_result_dir': best_result_dir,
    }
    with open(os.path.join(output_dir, 'rescue_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return json.dumps(summary)


def _run_dapi_roi_flip(he_path, fluo_path, output_dir, enhance=False):
    """Run DAPI ROI+flip pipeline."""
    from run_dapi_ROI_flip_pipeline import register_one_roi
    os.makedirs(output_dir, exist_ok=True)
    result = register_one_roi(he_path, fluo_path, output_dir, enhance=enhance)
    return json.dumps({
        'status': 'ok',
        'qcs': result.get('qcs', 0),
        'method': result.get('method', ''),
        'output_dir': output_dir,
    }, default=str)


def _run_dapi_contour_chain(he_path, fluo_path, boundary_dir, output_dir):
    """Run DAPI via contour chain."""
    from run_dapi_via_contour import register_dapi_via_contour
    os.makedirs(output_dir, exist_ok=True)
    result = register_dapi_via_contour(he_path, fluo_path, boundary_dir, output_dir)
    return json.dumps({
        'status': 'ok' if result else 'failed',
        'result': result,
        'output_dir': output_dir,
    }, default=str)


def _run_codex_registration(he_path, codex_path, output_dir, force_channel=None, enhance=0):
    """Run CODEX ROI registration."""
    import tifffile
    # Build CLI args and run via subprocess to handle enhance properly
    import subprocess
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, 'pipelines/codex/run_fluo2he_roi.py'),
        '--he', he_path,
        '--fluo', codex_path,
        '--output_dir', output_dir,
    ]
    if force_channel is not None:
        cmd += ['--channel', str(force_channel)]
    if enhance:
        cmd += ['--enhance', str(enhance)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Read result
    result_path = os.path.join(output_dir, 'result.json')
    if os.path.exists(result_path):
        with open(result_path) as f:
            data = json.load(f)
        key = list(data.keys())[0] if isinstance(data, dict) and len(data) == 1 else None
        if key:
            data = data[key]
        return json.dumps({
            'status': 'ok',
            'qcs': data.get('qcs', 0),
            'dice': data.get('dice', 0),
            'best_channel': data.get('best_channel', None),
            'output_dir': output_dir,
        }, default=str)
    return json.dumps({'status': 'failed', 'stderr': result.stderr[:500]})


def _run_codex_rescue_rotation(he_path, codex_path, output_dir, best_channel=0):
    """Run CODEX rotation rescue."""
    from run_codex_rescue import rescue_from_paths
    os.makedirs(output_dir, exist_ok=True)
    result = rescue_from_paths(he_path, codex_path, output_dir,
                                best_ch=best_channel, current_dice=0)
    return json.dumps({
        'status': 'ok',
        'result': result,
        'output_dir': output_dir,
    }, default=str)


def _list_codex_channels(codex_path):
    """List CODEX channels with coverage info."""
    import tifffile
    import cv2
    with tifffile.TiffFile(codex_path) as t:
        n_pages = len(t.pages)
        channels = []
        for ch in range(min(n_pages, 20)):
            page = t.pages[ch].asarray()
            h, w = page.shape[:2]
            # Quick coverage at 1/10 scale
            small = cv2.resize(page.astype(np.float32), (w // 10, h // 10))
            coverage = np.sum(small > np.percentile(small, 50)) / small.size
            channels.append({'channel': ch, 'coverage': round(float(coverage), 3)})
    channels.sort(key=lambda x: -x['coverage'])
    return json.dumps({
        'n_channels': n_pages,
        'size': [w, h],
        'channels': channels,
    })


def _run_boundary_registration(he_path, parquet_path, output_dir, cell_parquet_path=None):
    """Run boundary pipeline."""
    from run_boundary_pipeline import run_boundary_pipeline
    os.makedirs(output_dir, exist_ok=True)
    result = run_boundary_pipeline(he_path, parquet_path, output_dir,
                                    cell_parquet_path=cell_parquet_path)
    dice = result.get('quality_global', {}).get('dice', 0)
    return json.dumps({
        'status': 'ok',
        'global_dice': dice,
        'output_dir': output_dir,
    }, default=str)


def _run_boundary_rescue_ncc(he_path, filled_image_path, output_dir, current_dice=0):
    """Run boundary NCC rescue."""
    from run_boundary_rescue import rescue_from_paths
    os.makedirs(output_dir, exist_ok=True)
    _, new_dice = rescue_from_paths(he_path, filled_image_path, output_dir,
                                     current_dice=current_dice)
    return json.dumps({
        'status': 'ok',
        'new_dice': new_dice,
        'output_dir': output_dir,
    })


def _run_st_registration(he_path, st_path, output_dir):
    """Run ST→HE registration."""
    import subprocess
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, 'pipelines/st_he/run_st2he_test.py'),
        '--he', he_path, '--st', st_path, '--output_dir', output_dir
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    result_path = os.path.join(output_dir, 'result.json')
    if os.path.exists(result_path):
        with open(result_path) as f:
            data = json.load(f)
        return json.dumps({'status': 'ok', 'result': data, 'output_dir': output_dir}, default=str)
    return json.dumps({'status': 'failed', 'stderr': result.stderr[:500]})


# ── Tool Dispatcher ──────────────────────────────────────────────────────────

_TOOL_DISPATCH = {
    'read_result_json': lambda inputs: _read_result_json(inputs['output_dir']),
    'read_qcs_breakdown': lambda inputs: _read_qcs_breakdown(inputs['he_path'], inputs['overlay_dir']),
    'compare_results': lambda inputs: _compare_results(inputs['dir_a'], inputs['dir_b']),
    'finish_rescue': lambda inputs: _finish_rescue(
        inputs['output_dir'], inputs['best_result_dir'],
        inputs['original_qcs'], inputs['rescue_qcs'], inputs['rescue_method']),
    'run_dapi_roi_flip': lambda inputs: _run_dapi_roi_flip(
        inputs['he_path'], inputs['fluo_path'], inputs['output_dir'],
        enhance=inputs.get('enhance', False)),
    'run_dapi_contour_chain': lambda inputs: _run_dapi_contour_chain(
        inputs['he_path'], inputs['fluo_path'], inputs['boundary_dir'], inputs['output_dir']),
    'run_codex_registration': lambda inputs: _run_codex_registration(
        inputs['he_path'], inputs['codex_path'], inputs['output_dir'],
        force_channel=inputs.get('force_channel'), enhance=inputs.get('enhance', 0)),
    'run_codex_rescue_rotation': lambda inputs: _run_codex_rescue_rotation(
        inputs['he_path'], inputs['codex_path'], inputs['output_dir'],
        best_channel=inputs.get('best_channel', 0)),
    'list_codex_channels': lambda inputs: _list_codex_channels(inputs['codex_path']),
    'run_boundary_registration': lambda inputs: _run_boundary_registration(
        inputs['he_path'], inputs['parquet_path'], inputs['output_dir'],
        cell_parquet_path=inputs.get('cell_parquet_path')),
    'run_boundary_rescue_ncc': lambda inputs: _run_boundary_rescue_ncc(
        inputs['he_path'], inputs['filled_image_path'], inputs['output_dir'],
        current_dice=inputs.get('current_dice', 0)),
    'run_st_registration': lambda inputs: _run_st_registration(
        inputs['he_path'], inputs['st_path'], inputs['output_dir']),
}


def execute_rescue_tool(tool_name, tool_input):
    """Execute a rescue tool by name. Returns JSON string."""
    if tool_name not in _TOOL_DISPATCH:
        return json.dumps({'status': 'error', 'message': f'Unknown tool: {tool_name}'})
    try:
        return _TOOL_DISPATCH[tool_name](tool_input)
    except Exception as e:
        import traceback
        return json.dumps({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()[:500],
        })
