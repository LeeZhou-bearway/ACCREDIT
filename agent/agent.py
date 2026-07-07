"""
agent.py - LLM Agent main loop driven by Claude API (anthropic SDK).
"""
import json
import os
import anthropic

from agent.tools import TOOLS, execute_tool


SYSTEM_PROMPT_HE_IHC = """You are an expert computational pathology image registration agent. Your task is to align an IHC (P40 immunohistochemistry) whole-slide image to an H&E (hematoxylin & eosin) reference image from adjacent 5µm tissue sections. The target accuracy is cell-level (~10µm).

## Strategy C — Preferred Primary Path (use this FIRST)

For most HE-IHC cases, the fastest and highest-quality approach is the improved
deterministic pipeline (LightGlue + nonrigid + Demons). It achieves mean QCS 0.886
on the PASC benchmark, beating VALIS (0.707) on all 12 pairs.

**Preferred cascade**:
1. Call `run_improved_fast(he_path, ihc_path, output_dir)` with recipe='nr3_demons300' (default).
   - Takes 30-60 seconds on GPU. Returns QCS@512.
   - If QCS ≥ 0.70 → call `finish_registration(nmi_final=QCS, passed=True, ...)` and stop.
   - If QCS < 0.70 → go to step 2.
2. Call `run_improved_auto(he_path, ihc_path, output_dir)` (tries all 6 recipes, 2-4 min).
   - If QCS ≥ 0.60 → finish_registration and stop.
   - If QCS < 0.60 → go to step 3 (classical fallback).
3. Fall back to the Classical Tool Chain below (hematoxylin channel + BSpline).
   After classical tools produce a warped output, you may call `run_improved_rescue`
   to refine it with LightGlue + nonrigid + Demons — this often recovers borderline cases.

Track `qcs_512` across attempts and always keep the highest-scoring result.

## Classical Tool Chain (fallback, used only if Strategy C fails)

### Setup
1. Use `get_slide_info` to inspect both slides.
2. Load thumbnails: HE level 3, IHC level 2 (both ~8µm/px).
3. Extract hematoxylin channel (`extract_hematoxylin`) for BOTH images — use these H-channel IDs for Stage 1.

## Registration Strategy (Max 3 attempts per stage)

### Stage 1: Global Similarity (H-channel images)
Attempt 1: `run_similarity_registration` (H-channel, multi-init with rotations)
  NMI > 1.05 → go to Stage 2  |  NMI ≤ 1.05 → Attempt 2

Attempt 2: `run_shape_similarity` (tissue mask initialization — use when H-channel has intensity
  inversion, e.g. lymphoid follicles bright in HE but dark in IHC)
  NMI > 1.05 → go to Stage 2  |  NMI ≤ 1.05 → Attempt 3

Attempt 3: `run_similarity_registration` with sampling_pct=0.2
  NMI > 1.05 → go to Stage 2
  NMI ≤ 1.05 after all 3 attempts → SKIP: call finish_registration(status="skipped", ...) and stop

Always use the BEST NMI transform from Stage 1 for Stage 2.

### Stage 2: Boundary Correction (RGB images, NOT H-channel)
Use the SAME level 3/2 image_ids WITHOUT the __H suffix (original RGB IDs).

Attempt 1: `run_boundary_correction(boundary_width_px=60, mesh_size=8)`
  NMI improved AND > 1.10 → done (proceed to Completion)
  NMI did NOT improve over Stage 1 → revert to Stage 1 transform, skip further attempts
  NMI improved but < 1.10 → Attempt 2

Attempt 2: `run_boundary_correction(boundary_width_px=100, mesh_size=12)`
  Better than Attempt 1 → keep; still < 1.10 → Attempt 3

Attempt 3: `run_boundary_correction(boundary_width_px=120, mesh_size=16)`
  Keep highest NMI across all Stage 2 attempts.
  If all Stage 2 NMI < Stage 1 NMI: revert to Stage 1 transform.

### Stage 3: Interior Feature Matching (tile-based AKAZE/ORB)
Run after best Stage 2 to refine cellular-level alignment (~10-30px residuals) inside the tissue.
Pass RGB image IDs (NOT __H) and BEST Stage 2 transform_id as init_transform_id.

`run_tile_feature_registration(tile_size=512, overlap=64)`
  Returns: transform_id, nmi (global), avg_tile_ncc (local, range [-1,1]; higher=better).
  NMI improved over Stage 2 → keep Stage 3 transform.
  NMI not improved → revert to Stage 2 transform (do NOT retry).

Note: Cross-modal feature matching (HE↔IHC) — some tiles will use identity fallback (normal).
Use avg_tile_ncc as a secondary local quality indicator alongside global NMI.

### Completion
1. Call `assess_quality` with the BEST transform (Stage 3 if improved, else Stage 2).
   Pass output_dir = the directory from the task.
2. Call `finish_registration` with final status and NMI values — THIS IS MANDATORY.
   Use nmi_final = best NMI achieved (Stage 3 or Stage 2).

## Quality Thresholds
NMI = (H_fixed + H_moving) / H_joint — higher is better.
- Stage 1: NMI > 1.05 (typical ~1.07)
- Stage 2: NMI > 1.10 (typical ~1.12)
- Stage 3: NMI improved over Stage 2; avg_tile_ncc > 0.5 indicates good local alignment

## Important Notes
- image_id and transform_id values are returned by tools — always use exact returned IDs.
- For `run_boundary_correction`: pass RGB image IDs (e.g. `HE/slide.svs__level3`), NOT __H channel IDs.
- Do NOT use `run_affine_registration` at high resolution — causes boundary artifacts.
- Do NOT use global `run_nonrigid_registration` on the full image — causes overfitting.
- HE and IHC slides have ~1.5% physical size difference — `run_similarity_registration` corrects this.
"""

SYSTEM_PROMPT_HE_HE = """You are an expert computational pathology image registration agent. Your task is to align two H&E (hematoxylin & eosin) whole-slide images from adjacent serial sections. The target accuracy is cell-level (~10µm).

## Key Differences from HE-IHC Registration
- Both images use the SAME staining (H&E), so intensity distributions are very similar.
- Grayscale or H-channel both work well for all stages — no cross-modal issues.
- NCC (normalized cross-correlation) is a strong metric alongside MI for same-stain registration.
- Tissue morphology is very similar between adjacent sections, but there will be small deformations
  from sectioning artifacts (folding, stretching, compression).

## Setup
1. Use `get_slide_info` to inspect both slides.
2. Load thumbnails: both at level 3 (~8µm/px).
3. Convert to grayscale (`to_grayscale`) for BOTH images — use these grayscale IDs for Stage 1.
   (Both images are H&E stained, so grayscale works well directly — no need for hematoxylin deconvolution.)

## Registration Strategy (Max 3 attempts per stage)

### Stage 1: Global Similarity (grayscale images)
Attempt 1: `run_similarity_registration` (grayscale, multi-init with rotations)
  NMI > 1.10 → go to Stage 2  |  NMI ≤ 1.10 → Attempt 2

Attempt 2: `run_shape_similarity` (tissue mask initialization)
  NMI > 1.10 → go to Stage 2  |  NMI ≤ 1.10 → Attempt 3

Attempt 3: `run_similarity_registration` with sampling_pct=0.2
  NMI > 1.10 → go to Stage 2
  NMI ≤ 1.10 after all 3 attempts → SKIP: call finish_registration(status="skipped", ...) and stop

Note: HE-HE Stage 1 NMI threshold is HIGHER (1.10) than HE-IHC (1.05) because same-stain
registration should achieve higher similarity scores.
Always use the BEST NMI transform from Stage 1 for Stage 2.

### Stage 2: Boundary Correction (RGB images, NOT H-channel)
Use the SAME level 3 image_ids WITHOUT the __H suffix (original RGB IDs).

Attempt 1: `run_boundary_correction(boundary_width_px=60, mesh_size=8)`
  NMI improved AND > 1.15 → done (proceed to Completion)
  NMI did NOT improve over Stage 1 → revert to Stage 1 transform, skip further attempts
  NMI improved but < 1.15 → Attempt 2

Attempt 2: `run_boundary_correction(boundary_width_px=100, mesh_size=12)`
  Better than Attempt 1 → keep; still < 1.15 → Attempt 3

Attempt 3: `run_boundary_correction(boundary_width_px=120, mesh_size=16)`
  Keep highest NMI across all Stage 2 attempts.
  If all Stage 2 NMI < Stage 1 NMI: revert to Stage 1 transform.

### Stage 3: Interior Feature Matching (tile-based AKAZE/ORB)
Run after best Stage 2 to refine cellular-level alignment.
Pass RGB image IDs (NOT __H) and BEST Stage 2 transform_id as init_transform_id.

`run_tile_feature_registration(tile_size=512, overlap=64)`
  Returns: transform_id, nmi (global), avg_tile_ncc (local).
  For HE-HE, expect higher avg_tile_ncc (> 0.3) than HE-IHC because same staining.
  NMI improved over Stage 2 → keep Stage 3 transform.
  NMI not improved → revert to Stage 2 transform (do NOT retry).

### Completion
1. Call `assess_quality` with the BEST transform (Stage 3 if improved, else Stage 2).
   Pass output_dir = the directory from the task.
2. Call `finish_registration` with final status and NMI values — THIS IS MANDATORY.
   Use nmi_final = best NMI achieved (Stage 3 or Stage 2).

## Quality Thresholds (HE-HE — higher than HE-IHC)
NMI = (H_fixed + H_moving) / H_joint — higher is better.
- Stage 1: NMI > 1.10 (typical ~1.15 for same-stain)
- Stage 2: NMI > 1.15 (typical ~1.20)
- Stage 3: NMI improved over Stage 2; avg_tile_ncc > 0.3 indicates good local alignment

## Important Notes
- image_id and transform_id values are returned by tools — always use exact returned IDs.
- For `run_boundary_correction`: pass RGB image IDs (e.g. `HE/slide.svs__level3`), NOT __H channel IDs.
- HE-HE same-stain registration typically achieves HIGHER NMI and NCC than cross-modal HE-IHC.
- Adjacent HE sections have ~1-2% physical size difference — `run_similarity_registration` corrects this.
"""


SYSTEM_PROMPT_XENIUM = """You are an expert computational pathology image registration agent. Your task is to align Xenium platform nucleus boundary contours (from parquet) to the corresponding HE tissue image. The contours define nuclear boundaries in physical coordinates; the HE image provides the tissue context.

## Key Concepts
- Xenium outputs nucleus_boundaries.parquet with physical (µm) coordinates.
- The HE OME-TIFF has known mpp (microns-per-pixel), so physical→pixel conversion = 1/mpp.
- After coordinate conversion, contours are rendered as a filled binary mask (the "moving" image).
- The HE image is converted to an enhanced binary mask (the "reference").
- Feature-based alignment matches these two binary representations.
- Block-based refinement corrects local misalignments.

## Adaptive Strategy
You have adjustable parameters at each stage. If quality is poor, TRY DIFFERENT PARAMETERS before giving up.

### Stage 1: Data Loading & Preprocessing
1. `load_xenium_data` — load HE and parquet, get dataset_id.
2. `render_contour_mask` — render contours as binary mask.
3. `convert_he_binary` — convert HE to binary. Adjustable:
   - alpha (contrast): default 1.4, try 1.0-2.0
   - r/g/b_weight: default 0.3/0.1/0.6, adjust for different tissue staining

### Stage 2: Global Feature Alignment (max 3 attempts)
Use `run_feature_alignment` to align contour mask to HE binary.

Attempt 1: ORB with nfeatures=100000, scale_factor=0.2
  → Check inliers (>50 is good) and angle (~0° expected). If poor → Attempt 2.

Attempt 2: AKAZE with scale_factor=0.2
  → AKAZE is more robust to scale/rotation. If still poor → Attempt 3.

Attempt 3: Try different scale_factor (0.1, 0.3, 0.5) or nfeatures (50000, 200000).

After best global alignment:
- `warp_contour_mask` to create warped mask.
- `assess_xenium_quality` to check Dice/IoU.
  → Dice > 0.6: proceed to block refinement
  → Dice < 0.4: global alignment likely wrong, retry with different method/params
  → Dice 0.4-0.6: marginal, try block refinement but may need to retry global

### Stage 3: Block Refinement
1. `apply_global_transform` — transform contour coordinates.
2. `run_block_refinement` — per-block AKAZE/ORB refinement. Adjustable:
   - num_blocks: default 32, try 16 (coarser) or 64 (finer)
   - method: 'akaze' (default) or 'orb'
   - min_contours: default 20, lower for sparse regions
3. `assess_xenium_quality` again to verify improvement.

### Stage 4: Completion
1. `save_xenium_outputs` — save final contours and transforms.
2. `finish_xenium_registration` — record result.json. THIS IS MANDATORY.

## Quality Metrics
- **Dice**: overlap of binary masks. >0.7 good, >0.8 excellent.
- **IoU**: intersection over union. >0.5 good.
- **NCC**: normalized cross-correlation on grayscale. >0.3 good.
- **Coverage**: fraction of HE tissue covered by contours.

## Decision Rules
- Always compare metrics BEFORE and AFTER each stage.
- If block refinement WORSENS Dice, revert to global-only result.
- If multiple global methods succeed, pick the one with highest Dice.
- Log your reasoning for parameter choices.

## Important Notes
- Contour rendering is slow (~minutes for 400K+ cells). It's cached to disk — call render_contour_mask with output_dir to enable caching.
- HE images are large (2-3 GB). Loading is slow but one-time.
- Block refinement with too many blocks (>64) may have many failed blocks — use fewer blocks first.
- image_id and transform_id values are returned by tools — always use exact returned IDs.
"""


def run_agent(fixed_path: str, moving_path: str, output_dir: str,
              mode: str = "he-ihc") -> dict:
    """
    Run the LLM registration agent.

    Args:
        fixed_path: Path to fixed/reference image (HE SVS/TIFF or Xenium HE OME-TIFF).
        moving_path: Path to moving image (IHC/HE SVS or nucleus_boundaries.parquet).
        output_dir: Directory for quality overlays and result.json.
        mode: 'he-ihc', 'he-he', or 'xenium'.

    Returns:
        dict with registration results (read from result.json).
    """
    import agent.tools as _tools_mod
    _tools_mod._current_output_dir = output_dir

    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    if mode == "xenium":
        system_prompt = SYSTEM_PROMPT_XENIUM
    elif mode == "he-he":
        system_prompt = SYSTEM_PROMPT_HE_HE
    else:
        system_prompt = SYSTEM_PROMPT_HE_IHC

    # Select only relevant tools for each mode
    if mode == "xenium":
        xenium_tool_names = {
            "load_xenium_data", "render_contour_mask", "convert_he_binary",
            "run_feature_alignment", "warp_contour_mask", "apply_global_transform",
            "run_block_refinement", "assess_xenium_quality", "save_xenium_outputs",
            "finish_xenium_registration",
        }
        active_tools = [t for t in TOOLS if t["name"] in xenium_tool_names]
    else:
        xenium_tool_names = {
            "load_xenium_data", "render_contour_mask", "convert_he_binary",
            "run_feature_alignment", "warp_contour_mask", "apply_global_transform",
            "run_block_refinement", "assess_xenium_quality", "save_xenium_outputs",
            "finish_xenium_registration",
        }
        active_tools = [t for t in TOOLS if t["name"] not in xenium_tool_names]

    if mode == "xenium":
        initial_message = (
            f"Please register Xenium nucleus boundary contours to the HE tissue image.\n\n"
            f"HE image: {fixed_path}\n"
            f"Parquet (nucleus_boundaries): {moving_path}\n"
            f"Output directory: {output_dir}\n\n"
            f"Follow the adaptive Xenium registration strategy from the system prompt. "
            f"Pass output_dir='{output_dir}' to render_contour_mask, assess_xenium_quality, "
            f"and save_xenium_outputs. "
            f"You MUST call finish_xenium_registration as the final step."
        )
    elif mode == "he-he":
        initial_message = (
            f"Please register the moving HE image to the fixed HE reference image.\n\n"
            f"Fixed HE (reference): {fixed_path}\n"
            f"Moving HE: {moving_path}\n"
            f"Mode: HE-HE (same-stain adjacent section registration)\n"
            f"Output directory (for overlays and result.json): {output_dir}\n\n"
            f"Follow the HE-HE registration strategy from the system prompt. "
            f"Use grayscale (to_grayscale) for Stage 1 — NOT hematoxylin. "
            f"Pass output_dir='{output_dir}' to assess_quality. "
            f"You MUST call finish_registration as the final step."
        )
    else:
        initial_message = (
            f"Please register the IHC image to the HE reference image.\n\n"
            f"HE (reference/fixed): {fixed_path}\n"
            f"IHC (moving): {moving_path}\n"
            f"Mode: HE-IHC (cross-modal registration)\n"
            f"Output directory (for overlays and result.json): {output_dir}\n\n"
            f"Follow the adaptive registration strategy from the system prompt. "
            f"Use hematoxylin channel for Stage 1. "
            f"Pass output_dir='{output_dir}' to assess_quality. "
            f"You MUST call finish_registration as the final step."
        )

    messages = [{"role": "user", "content": initial_message}]

    mode_label = {"xenium": "Xenium", "he-he": "HE-HE"}.get(mode, "HE-IHC")
    print(f"\n{'='*60}")
    print(f"{mode_label} Registration Agent")
    print(f"Fixed:  {fixed_path}")
    print(f"Moving: {moving_path}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")

    max_turns = 50
    turn = 0

    while turn < max_turns:
        turn += 1
        print(f"\n[Agent] Turn {turn}/{max_turns}")

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=system_prompt,
            tools=active_tools,
            messages=messages,
        )

        print(f"[Agent] stop_reason={response.stop_reason}")

        # Collect tool calls and text from response
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                if block.text.strip():
                    print(f"[Agent] {block.text.strip()}")
            elif block.type == "tool_use":
                tool_calls.append(block)
                print(f"[Agent] -> tool_use: {block.name}({json.dumps(block.input)[:200]})")

        # Add assistant message to history
        messages.append({"role": "assistant", "content": response.content})

        # If no tool calls, agent is done
        if response.stop_reason == "end_turn" or not tool_calls:
            print("\n[Agent] Registration pipeline complete.")
            break

        # Execute all tool calls and collect results
        tool_results = []
        for tool_call in tool_calls:
            print(f"[Tool] Executing {tool_call.name} ...")
            result_str = execute_tool(tool_call.name, tool_call.input)
            result_data = json.loads(result_str)

            print(f"[Tool] Result: {json.dumps(result_data)[:300]}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result_str,
            })

        messages.append({"role": "user", "content": tool_results})

    if turn >= max_turns:
        print(f"\n[Agent] Warning: reached maximum turns ({max_turns}).")

    print("\n[Agent] Session ended.")

    # Read and return result.json written by finish_registration
    result_path = os.path.join(output_dir, "result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    return {"status": "error", "nmi_stage1": 0.0, "nmi_final": 0.0,
            "stage1_method": "unknown", "passed": False}


# ── Rescue Agent ─────────────────────────────────────────────────────────────

def run_rescue_agent(mode, he_path, moving_path, output_dir,
                     initial_result, extra=None, log=print):
    """Run rescue agent when standard pipeline produces low QCS.

    Uses modality-specific tools and prompts to diagnose and rescue.
    Does NOT replace run_agent() — this is a separate rescue mechanism.

    Args:
        mode: modality string ('dapi', 'codex', 'boundary', 'st-he')
        he_path: path to HE image
        moving_path: path to moving image (DAPI/CODEX/parquet/ST)
        output_dir: rescue output directory
        initial_result: result dict from standard pipeline
        extra: additional context (sample_dir, etc.)
        log: logging function

    Returns:
        Best result dict (original or rescued)
    """
    from agent.rescue_tools import get_tools_for_modality, execute_rescue_tool
    from agent.rescue_prompts import get_rescue_prompt

    os.makedirs(output_dir, exist_ok=True)

    client = anthropic.Anthropic()

    # Get modality-specific prompt and tools
    system_prompt = get_rescue_prompt(mode)
    active_tools = get_tools_for_modality(mode)

    # Build initial message with context
    qcs = initial_result.get('qcs', 0) if initial_result else 0
    qm = initial_result.get('quality_metrics', {}) if initial_result else {}

    initial_message = (
        f"The standard {mode} pipeline produced QCS={qcs:.4f}.\n\n"
        f"HE: {he_path}\n"
        f"Moving: {moving_path}\n"
        f"Rescue output: {output_dir}\n\n"
        f"Quality metrics breakdown:\n"
        f"  tissue_dice:   {qm.get('tissue_dice', 'N/A')}\n"
        f"  boundary_dice: {qm.get('boundary_dice', 'N/A')}\n"
        f"  ncc:           {qm.get('ncc', 'N/A')}\n"
        f"  hausdorff:     {qm.get('hausdorff_mean', 'N/A')}\n"
        f"  inlier_ratio:  {qm.get('inlier_ratio', 'N/A')}\n\n"
        f"Method used: {initial_result.get('method', 'unknown')}\n"
    )

    if extra:
        if extra.get('has_boundary_result'):
            initial_message += "\nNote: Boundary pipeline output is available (can use contour chain).\n"
        if extra.get('sample_dir'):
            initial_message += f"Sample directory: {extra['sample_dir']}\n"

    initial_message += (
        f"\nPlease diagnose the issue and attempt rescue strategies.\n"
        f"You MUST call finish_rescue as the final step."
    )

    messages = [{"role": "user", "content": initial_message}]

    log(f'\n{"="*60}')
    log(f'Rescue Agent ({mode})')
    log(f'  Original QCS: {qcs:.4f}')
    log(f'  HE: {he_path}')
    log(f'  Moving: {moving_path}')
    log(f'{"="*60}\n')

    max_turns = 30
    turn = 0
    best_result = initial_result

    while turn < max_turns:
        turn += 1
        log(f'[Rescue] Turn {turn}/{max_turns}')

        try:
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                system=system_prompt,
                tools=active_tools,
                messages=messages,
            )
        except Exception as e:
            log(f'[Rescue] API error: {e}')
            break

        log(f'[Rescue] stop_reason={response.stop_reason}')

        tool_calls = []
        for block in response.content:
            if block.type == "text":
                if block.text.strip():
                    log(f'[Rescue] {block.text.strip()[:200]}')
            elif block.type == "tool_use":
                tool_calls.append(block)
                log(f'[Rescue] -> tool: {block.name}({json.dumps(block.input)[:150]})')

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn" or not tool_calls:
            log('[Rescue] Agent finished.')
            break

        tool_results = []
        for tool_call in tool_calls:
            log(f'[Rescue] Executing {tool_call.name} ...')
            result_str = execute_rescue_tool(tool_call.name, tool_call.input)
            result_data = json.loads(result_str)
            log(f'[Rescue] Result: {json.dumps(result_data)[:200]}')

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result_str,
            })

        messages.append({"role": "user", "content": tool_results})

    if turn >= max_turns:
        log(f'[Rescue] Warning: reached max turns ({max_turns})')

    log('[Rescue] Session ended.')

    # Read rescue summary
    summary_path = os.path.join(output_dir, 'rescue_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        if summary.get('rescue_qcs', 0) > qcs:
            # Read the best result
            best_dir = summary.get('best_result_dir', output_dir)
            result_path = os.path.join(best_dir, 'result.json')
            if os.path.exists(result_path):
                with open(result_path) as f:
                    best_result = json.load(f)
                best_result['rescue_summary'] = summary
                log(f'[Rescue] Improved: {qcs:.4f} → {summary["rescue_qcs"]:.4f}')
                return best_result

    log(f'[Rescue] No improvement. Keeping original QCS={qcs:.4f}')
    return initial_result
