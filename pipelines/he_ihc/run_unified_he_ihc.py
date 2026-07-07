"""Unified HE-IHC pipeline (LightGlue + nonrigid refinement + Demons + optional LLM Agent rescue).

Flow:
  Stage 1 — nr3_demons300: LightGlue → 3× nonrigid refinement → 300-iter Demons (30-60s).
  Stage 2 — LLM Agent rescue (optional, off by default):
            If QCS < AGENT_THRESHOLD, invoke Claude-based agent for autonomous
            diagnosis and re-registration. Enabled via --agent flag or
            ACCREDIT_ENABLE_AGENT=1.

The pipeline produces:
  - warped_ihc.png         — best warped IHC (canonical)
  - color_overlay.png      — ACCREDIT-style red overlay (HE gray + IHC red)
  - result.json            — full cascade history + metrics
  - summary.json           — backwards-compat schema (nmi_final, qcs, passed, …)
"""
import os, sys, time, json, glob, traceback, shutil
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_ACCREDIT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ACCREDIT_ROOT)
sys.path.insert(0, _HERE)

RECIPE = 'nr3_demons300'
AGENT_THRESHOLD = float(os.environ.get('ACCREDIT_AGENT_THRESHOLD', '0.60'))


# ── Helpers ────────────────────────────────────────────────────────────────
def _to_gray(im):
    import cv2
    if im.ndim == 3 and im.shape[2] >= 3:
        return cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2GRAY)
    return im


def _run_registration(he_path, ihc_path, output_dir, log=print):
    """Run nr3_demons300 recipe."""
    from run_he_ihc import load_svs, to_gray, register
    t0 = time.time()
    he_gray = to_gray(load_svs(he_path))
    ihc_gray = to_gray(load_svs(ihc_path))
    try:
        warped, chosen, metrics = register(he_gray, ihc_gray, recipe=RECIPE, verbose=False)
    except Exception as e:
        log(f'[REGISTER] FAILED: {type(e).__name__}: {e}')
        traceback.print_exc()
        return {'stage': 'failed', 'qcs_512': 0.0, 'error': str(e),
                'elapsed_seconds': round(time.time() - t0, 1)}
    out_png = os.path.join(output_dir, 'warped_ihc.png')
    Image.fromarray(warped).save(out_png)
    elapsed = time.time() - t0
    qcs_val = float(metrics['qcs']) if metrics else 0.0
    log(f'[REGISTER] recipe={chosen}  QCS@512={qcs_val:.3f}  ({elapsed:.1f}s)')
    return {
        'stage': 'nr3_demons300', 'recipe': chosen, 'qcs_512': qcs_val,
        'metrics': {k: float(v) for k, v in (metrics or {}).items()
                    if isinstance(v, (int, float))},
        'warped_path': out_png, 'elapsed_seconds': round(elapsed, 1),
    }


def _find_agent_warped(agent_out_dir):
    candidates = (
        glob.glob(os.path.join(agent_out_dir, '**/IHC_warped*.tif'), recursive=True)
        + glob.glob(os.path.join(agent_out_dir, '**/warped*.png'), recursive=True)
        + glob.glob(os.path.join(agent_out_dir, '**/warped*.tif'), recursive=True)
        + glob.glob(os.path.join(agent_out_dir, '**/overlay_*.png'), recursive=True)
    )
    candidates = [c for c in candidates if 'color_overlay' not in c and 'comparison' not in c]
    return candidates[0] if candidates else None


def _qcs_on_warped(he_path, warped_path, eval_px=512):
    import cv2
    from run_he_ihc import load_svs, compute_qcs, to_gray
    he_gray = to_gray(load_svs(he_path))
    warped = np.array(Image.open(warped_path)) if warped_path.endswith(('.png', '.jpg')) else None
    if warped is None:
        try:
            import tifffile
            warped = tifffile.imread(warped_path)
        except Exception:
            warped = np.array(Image.open(warped_path))
    warped = _to_gray(warped)
    if warped.shape != he_gray.shape:
        warped = cv2.resize(warped, (he_gray.shape[1], he_gray.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
    return compute_qcs(he_gray, warped, eval_px=eval_px)


def _run_agent(he_path, ihc_path, output_dir, log=print):
    """Run LLM agent rescue. Returns dict with qcs + warped path."""
    log('─' * 60)
    log('Stage 2: LLM Agent rescue')
    log('─' * 60)
    t0 = time.time()
    agent_out = os.path.join(output_dir, 'agent_output')
    os.makedirs(agent_out, exist_ok=True)
    try:
        from agent.agent import run_agent
        agent_result = run_agent(he_path, ihc_path, agent_out, mode='he-ihc')
    except Exception as e:
        log(f'[AGENT] FAILED: {type(e).__name__}: {e}')
        traceback.print_exc()
        return {'stage': 'agent', 'qcs_512': 0.0, 'error': str(e),
                'elapsed_seconds': round(time.time() - t0, 1)}
    elapsed = time.time() - t0

    warped_path = _find_agent_warped(agent_out)
    qcs_val = 0.0
    metrics = {}
    if warped_path and os.path.isfile(warped_path):
        try:
            m = _qcs_on_warped(he_path, warped_path)
            qcs_val = float(m['qcs'])
            metrics = {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}
        except Exception as e:
            log(f'[AGENT] QCS eval failed: {e}')
    log(f'[AGENT] QCS@512={qcs_val:.3f}  ({elapsed:.1f}s)')
    return {
        'stage': 'agent', 'qcs_512': qcs_val, 'metrics': metrics,
        'warped_path': warped_path, 'agent_result': agent_result,
        'elapsed_seconds': round(elapsed, 1),
    }


def _generate_color_overlay(he_path, warped_png, output_dir, log=print):
    import cv2
    from run_he_ihc import load_svs, to_gray
    he_gray = to_gray(load_svs(he_path))
    warped = np.array(Image.open(warped_png))
    if warped.ndim == 3:
        warped = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    if warped.shape != he_gray.shape:
        warped = cv2.resize(warped, (he_gray.shape[1], he_gray.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
    g = warped.astype(np.float32)
    p_lo, p_hi = np.percentile(g, [1, 99])
    gn = np.clip((g - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0)
    ihc_red = np.stack([np.full_like(gn, 255), gn * 255, gn * 255], axis=-1).astype(np.uint8)
    _, mask = cv2.threshold(he_gray.astype(np.uint8), 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    ihc_red[mask == 0] = 255
    he_rgb_gray = np.stack([he_gray] * 3, axis=-1).astype(np.uint8)
    overlay = cv2.addWeighted(he_rgb_gray, 0.5, ihc_red, 0.5, 0)
    out = os.path.join(output_dir, 'color_overlay.png')
    Image.fromarray(overlay).save(out)
    log(f'[OVERLAY] color overlay → {out}')
    return out


# ── Main entry ─────────────────────────────────────────────────────────────
def run_unified_he_ihc(he_path, ihc_path, output_dir, log=print,
                        enable_agent=False):
    """HE-IHC registration with optional LLM agent rescue.

    Args:
        he_path: HE SVS path
        ihc_path: IHC SVS path (or second HE for HE-HE mode)
        output_dir: output directory
        enable_agent: if True, invoke LLM agent when QCS < AGENT_THRESHOLD
    """
    os.makedirs(output_dir, exist_ok=True)
    total_t0 = time.time()
    history = []
    candidates = []

    # ── Stage 1: nr3_demons300 ────────────────────────────────────────────
    log('=' * 60)
    log(f'Stage 1 — {RECIPE}')
    log('=' * 60)

    reg = _run_registration(he_path, ihc_path, output_dir, log=log)
    history.append(reg)
    if 'error' not in reg:
        candidates.append(reg)

    # ── Stage 2: LLM Agent rescue (optional) ──────────────────────────────
    enable = enable_agent or os.environ.get('ACCREDIT_ENABLE_AGENT', '0') == '1'
    if enable and reg.get('qcs_512', 0) < AGENT_THRESHOLD:
        log(f'\nQCS={reg["qcs_512"]:.3f} < {AGENT_THRESHOLD}: invoking LLM agent...')
        agent = _run_agent(he_path, ihc_path, output_dir, log=log)
        history.append(agent)
        if 'error' not in agent and agent.get('qcs_512', 0) > 0:
            candidates.append(agent)

    # ── Pick best ─────────────────────────────────────────────────────────
    if not candidates:
        log('All stages failed')
        best = {'stage': 'failed', 'qcs_512': 0.0, 'warped_path': None, 'metrics': {}}
    else:
        best = max(candidates, key=lambda r: r.get('qcs_512', 0))
        log(f'Best: {best["stage"]}  QCS={best["qcs_512"]:.3f}')

    # Copy best warped to canonical location + generate overlay
    if best and best.get('warped_path') and os.path.isfile(best['warped_path']):
        canonical = os.path.join(output_dir, 'warped_ihc.png')
        if os.path.abspath(best['warped_path']) != os.path.abspath(canonical):
            shutil.copy(best['warped_path'], canonical)
        best['warped_path_canonical'] = canonical
        try:
            overlay = _generate_color_overlay(he_path, canonical, output_dir, log=log)
            best['color_overlay'] = overlay
        except Exception as e:
            log(f'[OVERLAY] FAILED: {type(e).__name__}: {e}')

    total_elapsed = time.time() - total_t0
    result = {
        'final_stage': best.get('stage', 'unknown'),
        'final_qcs_512': best.get('qcs_512', 0.0),
        'final_recipe': best.get('recipe', RECIPE),
        'final_metrics': best.get('metrics', {}),
        'warped_path': best.get('warped_path_canonical', best.get('warped_path')),
        'color_overlay': best.get('color_overlay'),
        'total_elapsed_seconds': round(total_elapsed, 1),
        'agent_enabled': enable,
        'stage_history': history,
    }
    with open(os.path.join(output_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2, default=float)

    summary = {
        'nmi_final': best.get('metrics', {}).get('nmi', 0),
        'avg_tile_ncc': best.get('metrics', {}).get('ncc', 0),
        'qcs': result['final_qcs_512'],
        'passed': result['final_qcs_512'] >= 0.60,
        'stage_used': result['final_stage'],
        'recipe': RECIPE,
        'elapsed_seconds': round(total_elapsed, 1),
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    log('=' * 60)
    log(f'DONE  stage={result["final_stage"]}  '
        f'QCS={result["final_qcs_512"]:.3f}  ({total_elapsed:.1f}s total)')
    log('=' * 60)
    return result


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    import argparse
    p = argparse.ArgumentParser(
        description='HE-IHC registration (LightGlue + nonrigid + Demons, optional LLM agent)')
    p.add_argument('--he', required=True)
    p.add_argument('--ihc', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--agent', action='store_true',
                   help='Enable LLM agent rescue when QCS < threshold')
    args = p.parse_args()
    run_unified_he_ihc(args.he, args.ihc, args.output_dir,
                        enable_agent=args.agent)
