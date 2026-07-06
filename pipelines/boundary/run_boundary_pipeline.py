"""
run_boundary_pipeline.py — Xenium boundary→HE registration pipeline with automatic fallback.

This is the Xenium-specific pipeline. It uses cell/nucleus boundary parquets as
registration features. DAPI image registration (run_dapi_rescue.py, run_fluo2he.py)
is a separate pipeline.

Decision flow (automatic, no user input until Step 4):

  Step 1  Boundary registration
          register_one() from run_xenium_reg
          Source: cell/nucleus boundaries (parquet) rendered as density mask vs HE

  Step 2  Rescue  (if global_dice < THRESHOLD_RESCUE = 0.50)
          run_boundary_rescue — NCC rotation scan: HE nuclear channel vs cell density map
          Uses the blue channel of HE (hematoxylin) to find where nuclei are,
          and Gaussian-blurred boundary mask to find where cells are in Xenium space.

  Step 3  Ask user (if global_dice still < THRESHOLD_GIVEUP = 0.20 after rescue)
          Writes diagnosis + suggestions to <out_dir>/user_guidance.txt
          Note: no ROI step — boundary pipeline has no fluorescence image,
          only parquet contours. ROI fallback belongs to run_dapi_pipeline.py.

Quality thresholds (global dice):
  THRESHOLD_GOOD    = 0.50  → done, no further steps
  THRESHOLD_RESCUE  = 0.50  → try boundary rescue
  THRESHOLD_GIVEUP  = 0.20  → ask user

Outputs:
  <out_dir>/result.json           — final registration result
  <out_dir>/pipeline_summary.json — which stage succeeded + final quality
  <out_dir>/pipeline.log          — full decision log
  <out_dir>/user_guidance.txt     — (Step 4 only) diagnosis + manual suggestions

Usage:
  python run_pipeline.py \\
      --he      data/he_image.ome.tif \\
      --parquet data/nucleus_boundaries.parquet \\
      --morph   data/morphology.ome.tif \\
      --output_dir output/

  # morph is auto-detected if morphology.ome.tif lives next to the parquet file
  python run_pipeline.py \\
      --he      data/he_image.ome.tif \\
      --parquet data/nucleus_boundaries.parquet \\
      --output_dir output/
"""
import argparse
import json
import os
import time

# ── Decision thresholds ────────────────────────────────────────────────────────
THRESHOLD_GOOD   = 0.50  # global_dice >= this → done
THRESHOLD_RESCUE = 0.50  # global_dice <  this → try boundary rescue
THRESHOLD_GIVEUP = 0.20  # after rescue, still < this → ask user


# ── Helpers ───────────────────────────────────────────────────────────────────

def _global_dice(result):
    return result.get('quality_global', {}).get('dice', 0.0)

def _interior_dice(result):
    return result.get('quality_interior', {}).get('interior_dice', 0.0)

def _reload_result(result, out_dir):
    """Refresh result dict from disk after rescue scripts update result.json."""
    path = os.path.join(out_dir, 'result.json')
    if os.path.exists(path):
        with open(path) as f:
            result.update(json.load(f))

def _save_pipeline_summary(out_dir, stage, global_dice, interior_dice=None, decision_log=None):
    summary = {
        'final_stage':       stage,
        'final_global_dice': round(global_dice, 4),
        'thresholds': {
            'good':    THRESHOLD_GOOD,
            'rescue':  THRESHOLD_RESCUE,
            'giveup':  THRESHOLD_GIVEUP,
        },
    }
    if interior_dice is not None:
        summary['final_interior_dice'] = round(interior_dice, 4)

    with open(os.path.join(out_dir, 'pipeline_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    if decision_log is not None:
        with open(os.path.join(out_dir, 'pipeline.log'), 'w') as f:
            f.write('\n'.join(decision_log))

def _write_user_guidance(out_dir, final_dice, log):
    guidance = f"""Registration Quality: POOR  (global_dice = {final_dice:.3f})
All automatic fallback steps have been attempted and failed to reach a satisfactory result.

═══════════════════════════════════════════════════════════
Suggested manual interventions (try in order):
═══════════════════════════════════════════════════════════

1. ORIENTATION CHECK
   Inspect overlay_contours.png in {out_dir}
   - Are the cell boundaries in roughly the right part of the HE image?
   - If completely off: the HE and DAPI may be mirrored or in a very different orientation.
   Action: Try pre-rotating or flipping the input images before re-running.

2. COVERAGE CHECK — does DAPI cover only PART of the HE?
   Look at filled_image.png vs the HE dimensions in result.json.
   - If the DAPI footprint is much smaller than the HE: use ROI mode explicitly.
   Action: python run_fluo2he_roi.py --he <he> --fluo morphology.ome.tif --output_dir <out>

3. TISSUE QUALITY
   - Is filled_image.png empty or very sparse? → Nucleus boundary segmentation may have failed.
   - Is the DAPI image very dim or saturated? → Try a different Z layer.
   Action: inspect the morphology.ome.tif Z layers manually and extract the best MIP.

4. MANUAL COARSE ALIGNMENT
   If you know the approximate rotation angle between HE and DAPI:
   Action: run run_boundary_rescue.py or run_dapi_rescue.py with a specific --dataset
           after manually pre-rotating the input PNG.

5. PROVIDE INITIAL TRANSFORM
   If you have a rough homography from a visual inspection:
   Action: save it as affine_M_init.npy in {out_dir} and re-run with --init_M flag
           (coming in a future version of this pipeline).

═══════════════════════════════════════════════════════════
Diagnostic files:
  {out_dir}/filled_image.png       — cell boundary mask (Xenium/DAPI space)
  {out_dir}/overlay_contours.png   — current best alignment overlay on HE
  {out_dir}/result.json            — quality metrics and transform matrix
  {out_dir}/pipeline.log           — full decision log showing what was tried
═══════════════════════════════════════════════════════════
"""
    path = os.path.join(out_dir, 'user_guidance.txt')
    with open(path, 'w') as f:
        f.write(guidance)

    log('\n' + '!' * 60)
    log('REGISTRATION FAILED — MANUAL INTERVENTION REQUIRED')
    log('!' * 60)
    log(f'  Final global_dice = {final_dice:.4f}  (threshold = {THRESHOLD_GIVEUP})')
    log(f'  Guidance written to: {path}')
    log('  See user_guidance.txt for suggested next steps.')


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_boundary_pipeline(he_path, parquet_path, out_dir,
                          cell_parquet_path=None, log=print):
    """
    Run the full Fluo→HE registration pipeline with automatic fallback.
    Returns the final result dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    dlog_lines = []
    def dlog(msg):
        dlog_lines.append(msg)
        log(msg)

    # ── Step 1: Boundary registration ─────────────────────────────────────────
    dlog('=' * 60)
    dlog('STEP 1: Boundary registration')
    dlog('=' * 60)

    from run_xenium_reg import register_one
    result = register_one(
        he_path=he_path,
        parquet_path=parquet_path,
        out_dir=out_dir,
        cell_parquet_path=cell_parquet_path,
        metric='dice',
        log=log,
    )

    gd = _global_dice(result)
    id_ = _interior_dice(result)
    dlog(f'  → global_dice={gd:.4f}  interior_dice={id_:.4f}')

    if gd >= THRESHOLD_GOOD:
        dlog(f'  ✓ GOOD (>= {THRESHOLD_GOOD}). Pipeline done at Step 1.')
        _save_pipeline_summary(out_dir, 'step1', gd, id_, dlog_lines)
        return result

    # ── Step 2: Rescue ────────────────────────────────────────────────────────
    dlog('')
    dlog('=' * 60)
    dlog(f'STEP 2: Rescue  (global_dice={gd:.4f} < {THRESHOLD_RESCUE})')
    dlog('=' * 60)

    filled_path = os.path.join(out_dir, 'filled_image.png')

    # Step 2a — Boundary rescue
    dlog('  Step 2a: Boundary rescue (nuclear-channel NCC vs cell-density NCC)...')
    from run_boundary_rescue import rescue_from_paths as boundary_rescue
    _, gd = boundary_rescue(he_path, filled_path, out_dir, current_dice=gd, log=log)
    dlog(f'  → global_dice={gd:.4f}')

    if gd >= THRESHOLD_GOOD:
        dlog(f'  ✓ GOOD after boundary rescue. Pipeline done at Step 2.')
        _reload_result(result, out_dir)
        _save_pipeline_summary(out_dir, 'step2', gd, None, dlog_lines)
        return result

    # ── Step 3: Ask user ──────────────────────────────────────────────────────
    # No ROI step: boundary pipeline has no fluorescence image, only parquet contours.
    # ROI fallback (run_fluo2he_roi) belongs to run_dapi_pipeline.py.
    dlog('')
    dlog('=' * 60)
    if gd < THRESHOLD_GIVEUP:
        dlog(f'STEP 3: All automatic steps failed  (global_dice={gd:.4f} < {THRESHOLD_GIVEUP})')
        _write_user_guidance(out_dir, gd, dlog)
        _save_pipeline_summary(out_dir, 'failed', gd, None, dlog_lines)
    else:
        dlog(f'Pipeline complete (global_dice={gd:.4f}, below ideal but above give-up threshold)')
        dlog(f'  Registration may be usable — inspect overlay_contours.png')
        _save_pipeline_summary(out_dir, 'partial', gd, None, dlog_lines)

    dlog(f'\nTotal time: {time.time() - t_start:.0f}s')
    _reload_result(result, out_dir)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Automated Fluo→HE registration pipeline with fallback')
    parser.add_argument('--he',      required=True, help='HE OME-TIFF path')
    parser.add_argument('--parquet', required=True, help='nucleus_boundaries.parquet path')
    parser.add_argument('--cell-parquet', default=None,
                        help='cell_boundaries.parquet (auto-detected if adjacent to parquet)')
    parser.add_argument('--output_dir', default='output', help='Output directory')
    args = parser.parse_args()

    # Auto-detect cell_boundaries.parquet
    cell_parquet = args.cell_parquet
    if not cell_parquet:
        candidate = os.path.join(os.path.dirname(args.parquet), 'cell_boundaries.parquet')
        if os.path.exists(candidate):
            cell_parquet = candidate

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = open(os.path.join(args.output_dir, 'run.log'), 'w', buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    log('Fluo→HE Registration Pipeline')
    log(f'  HE:      {args.he}')
    log(f'  Parquet: {args.parquet}')
    log(f'  Cell:    {cell_parquet or "not provided"}')
    log(f'  Output:  {args.output_dir}')
    log(f'  Thresholds: good={THRESHOLD_GOOD}, rescue={THRESHOLD_RESCUE}, '
        f'giveup={THRESHOLD_GIVEUP}')
    log(f'  Note: DAPI pipeline (run_dapi_pipeline.py) is separate.')
    log('')

    try:
        result = run_boundary_pipeline(
            he_path=args.he,
            parquet_path=args.parquet,
            out_dir=args.output_dir,
            cell_parquet_path=cell_parquet,
            log=log,
        )
        qg  = result.get('quality_global', {})
        qi  = result.get('quality_interior', {})
        stage = json.load(open(os.path.join(args.output_dir, 'pipeline_summary.json')))\
                    .get('final_stage', 'unknown') \
                if os.path.exists(os.path.join(args.output_dir, 'pipeline_summary.json')) \
                else 'unknown'
        log(f'\n{"="*60}')
        log('PIPELINE COMPLETE')
        log(f'  final_stage   = {stage}')
        log(f'  global_dice   = {qg.get("dice", "N/A")}')
        log(f'  interior_dice = {qi.get("interior_dice", "N/A")}')
        log(f'{"="*60}')
    except Exception as e:
        import traceback
        log(f'PIPELINE ERROR: {e}\n{traceback.format_exc()}')
    finally:
        log_file.close()


if __name__ == '__main__':
    import warnings; warnings.warn('Direct pipeline execution is for development only. Use run_unified_agent.py as the unified entry point.', stacklevel=1)
    main()
