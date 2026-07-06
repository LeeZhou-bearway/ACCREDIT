"""
run_codex_pipeline.py — CODEX→HE registration pipeline with multi-metric QCS.

Decision flow (QCS = composite quality score from quality_metrics.py):
  Step 1  CODEX→HE registration (best protein marker channel)
          run_fluo2he.register_one()
          → QCS >= 0.60 → done

  Step 2  Rescue on protein marker channel  (if QCS < 0.60)
          run_codex_rescue.rescue_from_paths()
          Rotation scan + phase correlation + ECC on the same channel.
          → QCS >= 0.55 → done

  Step 3  Nuclear ch0 NMI rescue  (if QCS < 0.55 after Step 2)
          Multi-init SimpleITK NMI on nuclear channel (ch0).
          Nuclear channel has full tissue coverage → better for global alignment.
          → QCS >= 0.45 → done

  Step 4  ROI registration  (if QCS < 0.35 — partial FOV / large scale)
          run_fluo2he_roi.register_one_roi()
          NMI-based channel+scale search.
          → QCS >= 0.45 → done

  Step 5  DAPI-HE fallback  (if QCS < 0.60 after all CODEX steps)
          run_dapi_ROI_flip_pipeline.register_one_roi()
          Exhaustive orientation scan (4 flips × 4 rotations × 6 ratios)
          + NMI + NCC fine-tune + ECC + BSpline.
          Keeps result only if QCS improves.

  Step 6  Give up  (QCS < 0.25)
          Writes user_guidance.txt with manual intervention suggestions.

QCS thresholds:
  QCS_DONE       = 0.60  → confident, no further steps
  QCS_RESCUE     = 0.60  → trigger protein-channel rescue
  QCS_NUCLEAR    = 0.55  → trigger nuclear ch0 NMI rescue
  QCS_ROI        = 0.35  → trigger ROI registration
  QCS_DAPI       = 0.60  → trigger DAPI-HE fallback
  QCS_GIVEUP     = 0.25  → give up

Outputs:
  <out_dir>/result.json           — final result with quality_metrics + qcs
  <out_dir>/pipeline_summary.json — stage, QCS, all thresholds
  <out_dir>/pipeline.log          — full decision log
"""
import argparse
import json
import os
import time

# ── QCS-based decision thresholds ─────────────────────────────────────────────
QCS_DONE    = 0.60  # after any step, QCS >= this → satisfied
QCS_RESCUE  = 0.60  # after step1, QCS < this → try protein-channel rescue
QCS_NUCLEAR = 0.55  # after step2, QCS < this → try nuclear ch0 NMI rescue
QCS_ROI     = 0.35  # after step3, QCS < this → try ROI
QCS_DAPI    = 0.60  # after step4, QCS < this → try DAPI-HE fallback
QCS_GIVEUP  = 0.25  # after all steps, QCS < this → ask user

OVERLAY_SIZE = 2000  # long-side target for output overlay


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_overlay(he_path, codex_path, M_full, out_dir, best_ch=0, log=print):
    """Generate overlay image using best protein channel at OVERLAY_SIZE resolution."""
    import cv2
    import numpy as np
    import tifffile

    def _to_hwc(img):
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < min(img.shape[1:]):
            img = np.moveaxis(img, 0, -1)
        return img

    def _norm8(arr):
        f = arr.astype(np.float32)
        pos = f[f > 0]
        if len(pos) > 100:
            p1, p99 = np.percentile(pos, [1, 99])
            return np.clip((f - p1) / (p99 - p1 + 1e-10) * 255, 0, 255).astype(np.uint8)
        return np.zeros_like(arr, dtype=np.uint8)

    he = _to_hwc(tifffile.imread(he_path))[..., :3]
    he_h, he_w = he.shape[:2]
    he_scale = min(1.0, OVERLAY_SIZE / max(he_h, he_w))
    out_w, out_h = int(he_w * he_scale), int(he_h * he_scale)
    he_out = cv2.resize(he, (out_w, out_h), interpolation=cv2.INTER_AREA)
    del he

    with tifffile.TiffFile(codex_path) as t:
        fl = t.pages[best_ch].asarray().astype(np.float32)
    fl_h, fl_w = fl.shape[:2]
    fl_scale = min(1.0, 8000 / max(fl_h, fl_w))
    if fl_scale < 1.0:
        fl = cv2.resize(fl, (int(fl_w * fl_scale), int(fl_h * fl_scale)))
    fl_norm = _norm8(fl)
    del fl

    M_out = M_full.copy()
    M_out[0, 0] *= he_scale / fl_scale;  M_out[0, 1] *= he_scale / fl_scale
    M_out[1, 0] *= he_scale / fl_scale;  M_out[1, 1] *= he_scale / fl_scale
    M_out[0, 2] *= he_scale;             M_out[1, 2] *= he_scale

    warped = cv2.warpAffine(fl_norm, M_out, (out_w, out_h), flags=cv2.INTER_LINEAR)

    # Green overlay — fixed alpha, same style as run_fluo2he
    overlay = he_out.copy()
    mask = warped > 30
    overlay[mask] = (overlay[mask] * 0.6 +
                     np.array([0, 255, 0], dtype=np.uint8) * 0.4).astype(np.uint8)

    out_path = os.path.join(out_dir, 'overlay_final.png')
    cv2.imwrite(out_path, cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
    log(f'  Overlay saved: {out_path}')
    return out_path


def _compute_qcs(he_path, codex_path, M_full, out_dir, best_ch=0, log=print):
    """Compute full quality metrics + QCS for a given M_full.

    Loads images at work size, warps nuclear ch0, computes all metrics.
    Returns (qcs, metrics_dict).
    """
    import cv2
    import numpy as np
    import tifffile
    from quality_metrics import compute_quality_metrics_v2

    WORK_SIZE = 4000

    def _to_hwc(img):
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < min(img.shape[1:]):
            img = np.moveaxis(img, 0, -1)
        return img

    def _norm8(arr):
        f = arr.astype(np.float32)
        pos = f[f > 0]
        if len(pos) > 100:
            p1, p99 = np.percentile(pos, [1, 99])
            return np.clip((f - p1) / (p99 - p1 + 1e-10) * 255, 0, 255).astype(np.uint8)
        return np.zeros_like(arr, dtype=np.uint8)

    # Load HE
    he = _to_hwc(tifffile.imread(he_path))[..., :3]
    he_h, he_w = he.shape[:2]
    he_scale = min(1.0, WORK_SIZE / max(he_h, he_w))
    ws_w, ws_h = int(he_w * he_scale), int(he_h * he_scale)
    he_work = cv2.resize(he, (ws_w, ws_h))
    del he

    he_gray = cv2.cvtColor(he_work, cv2.COLOR_RGB2GRAY)
    he_inv = (255 - he_gray).astype(np.uint8)
    _, he_mask = cv2.threshold(he_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    he_mask = cv2.morphologyEx(he_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    he_mask = cv2.morphologyEx(he_mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # Load nuclear channel (ch0) for quality evaluation
    with tifffile.TiffFile(codex_path) as t:
        nuc = t.pages[0].asarray().astype(np.float32)
    fl_h, fl_w = nuc.shape[:2]
    fl_scale = min(1.0, WORK_SIZE / max(fl_h, fl_w))
    nuc_work = _norm8(cv2.resize(nuc, (int(fl_w * fl_scale), int(fl_h * fl_scale))))
    del nuc

    _, nuc_mask = cv2.threshold(nuc_work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nuc_mask = cv2.morphologyEx(nuc_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    nuc_mask = cv2.morphologyEx(nuc_mask, cv2.MORPH_DILATE, kernel, iterations=2)

    # Scale M_full to work coords
    ratio = fl_scale / he_scale
    M_work = M_full.copy()
    M_work[0, 0] *= ratio;  M_work[0, 1] *= ratio
    M_work[1, 0] *= ratio;  M_work[1, 1] *= ratio
    M_work[0, 2] *= he_scale;  M_work[1, 2] *= he_scale

    warped_mask = cv2.warpAffine(nuc_mask, M_work, (ws_w, ws_h))
    warped_norm = cv2.warpAffine(nuc_work, M_work, (ws_w, ws_h))

    metrics = compute_quality_metrics_v2(he_mask, warped_mask, he_inv, warped_norm, eval_px=4000)
    return metrics['qcs'], metrics


def _save_pipeline_summary(out_dir, stage, qcs, metrics, decision_log=None):
    summary = {
        'final_stage':   stage,
        'final_qcs':     qcs,
        'quality_metrics': metrics,
        'thresholds': {
            'qcs_done':   QCS_DONE,
            'qcs_rescue': QCS_RESCUE,
            'qcs_nuclear': QCS_NUCLEAR,
            'qcs_roi':    QCS_ROI,
            'qcs_dapi':   QCS_DAPI,
            'qcs_giveup': QCS_GIVEUP,
        },
    }
    with open(os.path.join(out_dir, 'pipeline_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    if decision_log is not None:
        with open(os.path.join(out_dir, 'pipeline.log'), 'w') as f:
            f.write('\n'.join(decision_log))


def _write_user_guidance(out_dir, codex_path, qcs, metrics, log):
    guidance = f"""Registration Quality: POOR  (QCS = {qcs:.3f})
All automatic fallback steps have been attempted.

Quality metrics:
  Tissue Dice:     {metrics.get('tissue_dice', 'N/A')}
  Boundary Dice:   {metrics.get('boundary_dice', 'N/A')}
  Hausdorff Mean:  {metrics.get('hausdorff_mean', 'N/A')} px
  Inlier Ratio:    {metrics.get('inlier_ratio', 'N/A')}
  Composite QCS:   {qcs}

═══════════════════════════════════════════════════════════
Suggested manual interventions (try in order):
═══════════════════════════════════════════════════════════

1. ORIENTATION CHECK
   Inspect overlay files in {out_dir}
   Action: Try pre-rotating the CODEX image (0°/90°/180°/270°).

2. SCALE MISMATCH
   Compare HE and CODEX image dimensions (see result.json).
   Action: python run_fluo2he_roi.py --he <he> --fluo {codex_path} --output_dir <out>

3. CHANNEL SELECTION
   CODEX panels may have 40+ channels; try nuclear channel (ch0) manually.

4. PARTIAL OVERLAP / TISSUE COVERAGE
   Action: python run_fluo2he_roi.py --he <he> --fluo {codex_path} --output_dir <out>

═══════════════════════════════════════════════════════════
"""
    path = os.path.join(out_dir, 'user_guidance.txt')
    with open(path, 'w') as f:
        f.write(guidance)
    log(f'  Guidance written to: {path}')


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_codex_pipeline(he_path, codex_path, out_dir, log=print):
    """Run the full CODEX→HE registration pipeline with QCS-based decisions."""
    import numpy as np
    import tifffile
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    dlog_lines = []
    def dlog(msg):
        dlog_lines.append(msg)
        log(msg)

    result_path = os.path.join(out_dir, 'result.json')

    # ── Step 1: CODEX→HE registration (protein marker channel) ───────────
    dlog('=' * 60)
    dlog('STEP 1: CODEX→HE registration (auto channel selection)')
    dlog('=' * 60)

    from run_fluo2he import register_one as fluo2he_register
    dlog('  Loading HE image...')
    he_img = tifffile.imread(he_path)
    result = fluo2he_register(he_img, codex_path, out_dir, log=log)
    del he_img

    dice = result.get('dice', 0.0)
    M_full = np.array(result['M_full'])
    best_ch = result.get('best_channel', 0)
    overlay_ch = best_ch  # keep Step 1's best protein channel for overlay

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Compute QCS for Step 1 result
    qcs_step1, metrics_step1 = _compute_qcs(he_path, codex_path, M_full, out_dir, log=log)
    dlog(f'  → dice={dice:.4f}  QCS={qcs_step1:.4f}  '
         f'(bnd_dice={metrics_step1["boundary_dice"]:.3f} hd={metrics_step1["hausdorff_mean"]:.0f}px '
         f'ir={metrics_step1["inlier_ratio"]:.4f})')

    # Track best result across all steps
    best_qcs, best_metrics, best_M, best_stage = qcs_step1, metrics_step1, M_full.copy(), 'step1'

    # ── Step 2: Nuclear ch0 NMI rescue (always run — most effective method)
    dlog('')
    dlog('=' * 60)
    dlog(f'STEP 2: Nuclear ch0 NMI rescue  (always run)')
    dlog('=' * 60)

    try:
        from run_codex_nuclear_rescue import register_with_nuclear_ch
        nuc_result = register_with_nuclear_ch(he_path, codex_path, out_dir, log=log)

        if nuc_result is not None:
            M_nuc, dice_nuc, method_nuc = nuc_result
            qcs_nuc, metrics_nuc = _compute_qcs(he_path, codex_path, M_nuc, out_dir, log=log)
            dlog(f'  Nuclear ch0: QCS={qcs_nuc:.4f}  '
                 f'(bnd_dice={metrics_nuc["boundary_dice"]:.3f} '
                 f'hd={metrics_nuc["hausdorff_mean"]:.0f}px)')

            # Nuclear ch0 result always takes priority when available
            dlog(f'  Nuclear ch0 takes priority: QCS {best_qcs:.4f} → {qcs_nuc:.4f}')
            best_qcs, best_metrics = qcs_nuc, metrics_nuc
            best_M, best_stage = M_nuc.copy(), 'step2_nuclear'
            result['M_full'] = M_nuc.tolist()
            result['method'] = method_nuc
            result['best_channel'] = 0
            np.save(os.path.join(out_dir, 'affine_M_rescue.npy'), M_nuc)
        else:
            dlog('  Nuclear ch0 rescue returned None')
    except Exception as e:
        import traceback
        dlog(f'  Step 2 ERROR: {e}\n{traceback.format_exc()}')

    # ── Step 3: Protein-channel rescue (if still not good enough) ────────
    if best_qcs < QCS_DONE:
        dlog('')
        dlog('=' * 60)
        dlog(f'STEP 3: Protein-channel rescue  (QCS={best_qcs:.4f} < {QCS_DONE})')
        dlog('=' * 60)

        from run_codex_rescue import rescue_from_paths as codex_rescue
        _, dice3 = codex_rescue(he_path, codex_path, out_dir, best_ch=best_ch,
                                current_dice=dice, log=log)

        # Reload result (rescue may have updated M_full)
        result_reloaded = json.load(open(result_path))
        M_rescue = np.array(result_reloaded['M_full'])
        qcs_rescue, metrics_rescue = _compute_qcs(he_path, codex_path, M_rescue, out_dir, log=log)
        dlog(f'  Protein rescue: QCS={qcs_rescue:.4f}')

        if qcs_rescue > best_qcs:
            dlog(f'  Protein rescue improved: QCS {best_qcs:.4f} → {qcs_rescue:.4f}')
            best_qcs, best_metrics = qcs_rescue, metrics_rescue
            best_M, best_stage = M_rescue.copy(), 'step3_rescue'
            result['M_full'] = M_rescue.tolist()
            result['method'] = result_reloaded.get('method', 'rescue')
        else:
            dlog(f'  Protein rescue no improvement ({qcs_rescue:.4f} <= {best_qcs:.4f})')
    else:
        dlog(f'\n  Step 3 skipped — QCS={best_qcs:.4f} >= {QCS_DONE}')

    # Check if we're good enough to stop before ROI + DAPI fallback
    if best_qcs >= QCS_DAPI:
        result['quality_metrics'] = best_metrics
        result.setdefault('quality_global', {})['dice'] = round(best_metrics.get('tissue_dice', 0), 4)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        _save_overlay(he_path, codex_path, best_M, out_dir, best_ch=overlay_ch, log=dlog)
        _save_pipeline_summary(out_dir, best_stage, best_qcs, best_metrics, dlog_lines)
        dlog(f'\n  Done at {best_stage} (QCS={best_qcs:.4f} >= {QCS_DAPI})')
        return result

    # ── Step 4: ROI registration (only for very low QCS) ──────────────
    if best_qcs < QCS_ROI:
        dlog('')
        dlog('=' * 60)
        dlog(f'STEP 4: ROI registration  (QCS={best_qcs:.4f} < {QCS_ROI})')
        dlog('=' * 60)

        try:
            from run_fluo2he_roi import register_one_roi
            dlog('  Loading HE image for ROI registration...')
            he_img = tifffile.imread(he_path)
            roi_result = register_one_roi(he_img, codex_path, out_dir, log=log)
            del he_img

            M_roi = np.array(roi_result['M_full'])
            qcs_roi, metrics_roi = _compute_qcs(he_path, codex_path, M_roi, out_dir, log=log)
            dlog(f'  ROI: QCS={qcs_roi:.4f}')

            if qcs_roi > best_qcs:
                best_qcs, best_metrics = qcs_roi, metrics_roi
                best_M, best_stage = M_roi.copy(), 'step4_roi'
                result.update(roi_result)
                result['quality_metrics'] = best_metrics
                result.setdefault('quality_global', {})['dice'] = round(best_metrics.get('tissue_dice', 0), 4)
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
        except Exception as e:
            import traceback
            dlog(f'  Step 4 ERROR: {e}\n{traceback.format_exc()}')
    else:
        dlog(f'\n  Step 4 skipped — QCS={best_qcs:.4f} >= {QCS_ROI}')

    # ── Step 5: DAPI-HE fallback ───────────────────────────────────────
    if best_qcs < QCS_DAPI:
        dlog('')
        dlog('=' * 60)
        dlog(f'STEP 5: DAPI-HE fallback  (QCS={best_qcs:.4f} < {QCS_DAPI})')
        dlog('=' * 60)

        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dapi'))
            from run_dapi_ROI_flip_pipeline import register_one_roi as dapi_register

            dapi_out = os.path.join(out_dir, 'dapi_fallback')
            dapi_result = dapi_register(
                he_path=he_path,
                fluo_path=codex_path,
                out_dir=dapi_out,
                log=log,
            )

            if dapi_result and dapi_result.get('status') != 'failed':
                M_dapi = np.array(dapi_result['M_full'])
                qcs_dapi, metrics_dapi = _compute_qcs(
                    he_path, codex_path, M_dapi, out_dir, log=log)
                dlog(f'  DAPI fallback: QCS={qcs_dapi:.4f}  '
                     f'(td={metrics_dapi["tissue_dice"]:.3f} '
                     f'bd={metrics_dapi["boundary_dice"]:.3f})')

                if qcs_dapi > best_qcs:
                    dlog(f'  DAPI fallback improved: QCS {best_qcs:.4f} → {qcs_dapi:.4f}')
                    best_qcs, best_metrics = qcs_dapi, metrics_dapi
                    best_M, best_stage = M_dapi.copy(), 'step5_dapi'
                    result['M_full'] = M_dapi.tolist()
                    result['method'] = dapi_result.get('method', 'dapi_fallback')
                    result['dapi_fallback'] = {
                        'flip': dapi_result.get('flip'),
                        'rotation': dapi_result.get('rotation'),
                        'pixel_ratio': dapi_result.get('pixel_ratio'),
                        'best_channel': dapi_result.get('best_channel'),
                    }
                    np.save(os.path.join(out_dir, 'affine_M_dapi_fallback.npy'), M_dapi)
                else:
                    dlog(f'  DAPI fallback no improvement ({qcs_dapi:.4f} <= {best_qcs:.4f})')
            else:
                dlog('  DAPI fallback returned no valid result')
        except Exception as e:
            import traceback
            dlog(f'  Step 5 DAPI ERROR: {e}\n{traceback.format_exc()}')
    else:
        dlog(f'\n  Step 5 skipped — QCS={best_qcs:.4f} >= {QCS_DAPI}')

    # ── Step 6: Final decision ───────────────────────────────────────────
    dlog('')
    dlog('=' * 60)
    if best_qcs < QCS_GIVEUP:
        dlog(f'STEP 6: All steps failed  (QCS={best_qcs:.4f} < {QCS_GIVEUP})')
        _write_user_guidance(out_dir, codex_path, best_qcs, best_metrics, dlog)
        _save_pipeline_summary(out_dir, 'failed', best_qcs, best_metrics, dlog_lines)
    else:
        dlog(f'Pipeline complete (QCS={best_qcs:.4f})')
        result['quality_metrics'] = best_metrics
        result.setdefault('quality_global', {})['dice'] = round(best_metrics.get('tissue_dice', 0), 4)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        _save_pipeline_summary(out_dir, 'partial', best_qcs, best_metrics, dlog_lines)

    # ── Final overlay with best M ──────────────────────────────────────
    try:
        import numpy as np
        M_final = np.array(result['M_full'])
        _save_overlay(he_path, codex_path, M_final, out_dir, best_ch=overlay_ch, log=dlog)
    except Exception as e:
        dlog(f'  Overlay generation failed: {e}')

    dlog(f'\nTotal time: {time.time() - t_start:.0f}s')
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Automated CODEX→HE registration pipeline (QCS-based)')
    parser.add_argument('--he',    required=True, help='HE image path')
    parser.add_argument('--codex', required=True, nargs='+',
                        help='CODEX image path(s)')
    parser.add_argument('--output_dir', default='output', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = open(os.path.join(args.output_dir, 'run.log'), 'w', buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    log('CODEX→HE Registration Pipeline (QCS-based)')
    log(f'  HE:     {args.he}')
    log(f'  CODEX:  {args.codex}')
    log(f'  Output: {args.output_dir}')
    log(f'  QCS thresholds: done={QCS_DONE}, rescue={QCS_RESCUE}, '
        f'nuclear={QCS_NUCLEAR}, roi={QCS_ROI}, giveup={QCS_GIVEUP}')
    log('')

    try:
        all_results = {}
        for codex_path in args.codex:
            codex_name = os.path.splitext(os.path.basename(codex_path))[0]
            codex_out  = os.path.join(args.output_dir, codex_name) \
                         if len(args.codex) > 1 else args.output_dir

            log(f'\n{"="*60}')
            log(f'Processing: {codex_name}')
            log(f'{"="*60}')

            result = run_codex_pipeline(
                he_path=args.he,
                codex_path=codex_path,
                out_dir=codex_out,
                log=log,
            )
            all_results[codex_name] = result

        log(f'\n{"="*60}')
        log('PIPELINE COMPLETE — SUMMARY')
        log(f'{"="*60}')
        for name, r in all_results.items():
            qm = r.get('quality_metrics', {})
            qcs = qm.get('qcs', 'N/A')
            stage = 'unknown'
            summary_path = os.path.join(args.output_dir, name, 'pipeline_summary.json') \
                           if len(args.codex) > 1 \
                           else os.path.join(args.output_dir, 'pipeline_summary.json')
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    stage = json.load(f).get('final_stage', 'unknown')
            log(f'  {name}: QCS={qcs}  dice={r.get("dice", "N/A")}  stage={stage}')

    except Exception as e:
        import traceback
        log(f'PIPELINE ERROR: {e}\n{traceback.format_exc()}')
    finally:
        log_file.close()


if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    main()
