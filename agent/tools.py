"""
tools.py - Claude tool_use definitions (JSON schema) + execution functions.
"""
import json
import traceback
import numpy as np
import os

# Injected by run_agent() before each run — used by finish_registration
_current_output_dir: str = "output"

# --- Tool JSON schemas for Claude API ---

TOOLS = [
    {
        "name": "load_wsi_thumbnail",
        "description": (
            "Load a WSI (whole slide image) at a given OpenSlide pyramid level and "
            "cache it in memory. Level 0 is full resolution; higher levels are progressively "
            "downsampled (typically 2x per level). Returns image_id and image shape."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the SVS/TIFF slide file.",
                },
                "level": {
                    "type": "integer",
                    "description": "OpenSlide level index. 0=full resolution. Use high levels (e.g. 4-6) for thumbnails.",
                },
            },
            "required": ["path", "level"],
        },
    },
    {
        "name": "get_slide_info",
        "description": (
            "Return metadata about a WSI: number of levels, dimensions at each level, "
            "downsampling factors, and microns-per-pixel (mpp)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the SVS/TIFF slide file."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "extract_hematoxylin",
        "description": (
            "Apply rgb2hed colour deconvolution to a cached RGB image and extract the "
            "H (hematoxylin / haematoxylin) channel as a uint8 grayscale image. "
            "This channel is present in both HE (blue-purple nuclei) and IHC P40 "
            "(blue nuclear counterstain), making it ideal for cross-modal registration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "description": "ID of a previously loaded RGB image."},
            },
            "required": ["image_id"],
        },
    },
    {
        "name": "to_grayscale",
        "description": (
            "Convert a cached RGB image to grayscale (luminance-weighted). "
            "Use as fallback strategy B when hematoxylin extraction does not yield "
            "sufficient contrast for registration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "description": "ID of a previously loaded RGB image."},
            },
            "required": ["image_id"],
        },
    },
    {
        "name": "run_similarity_registration",
        "description": (
            "Stage 1 (preferred): Similarity registration (uniform scale + rotation + translation). "
            "Adds 1 scale DOF over rigid registration, correcting the ~1.5% physical size difference "
            "between HE and IHC slides that pure rigid registration cannot handle. "
            "Use this instead of run_rigid_registration as the first step. "
            "Returns transform_id and NMI score. Target NMI > 1.05."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fixed_id": {"type": "string", "description": "image_id of the fixed (reference) image."},
                "moving_id": {"type": "string", "description": "image_id of the moving image to register."},
                "sampling_pct": {
                    "type": "number",
                    "description": "Fraction of pixels to sample for metric (0.05-0.3). Default 0.1.",
                },
            },
            "required": ["fixed_id", "moving_id"],
        },
    },
    {
        "name": "run_shape_similarity",
        "description": (
            "Stage 1 (fallback): Similarity registration initialized from tissue SHAPE (binary mask). "
            "Use this when run_similarity_registration fails (NMI ≤ 1.05), especially when "
            "H-channel has intensity inversion (e.g., lymphoid follicles bright in HE but dark in IHC P40). "
            "Computes Otsu tissue masks from fixed/moving images and initializes from mask MOMENTS + GEOMETRY "
            "+ ±45°/±90°/180° rotation offsets (7 candidates). "
            "Optimization uses same H-channel images + MattesMI as run_similarity_registration. "
            "Returns transform_id and NMI score. Target NMI > 1.05."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fixed_id": {"type": "string", "description": "image_id of the fixed H-channel image."},
                "moving_id": {"type": "string", "description": "image_id of the moving H-channel image."},
                "sampling_pct": {
                    "type": "number",
                    "description": "Fraction of pixels to sample for metric (0.05-0.3). Default 0.1.",
                },
            },
            "required": ["fixed_id", "moving_id"],
        },
    },
    {
        "name": "finish_registration",
        "description": (
            "MANDATORY LAST STEP: Record the final registration result and write it to result.json. "
            "Call this after assess_quality, with the final status and NMI values. "
            "status='success' if registration passed, status='skipped' if all attempts failed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["success", "skipped"],
                    "description": "'success' if registration achieved NMI > 1.05, 'skipped' if all attempts failed.",
                },
                "nmi_stage1": {
                    "type": "number",
                    "description": "Best NMI achieved in Stage 1 (global similarity).",
                },
                "nmi_final": {
                    "type": "number",
                    "description": "Final NMI after Stage 2 boundary correction (or same as nmi_stage1 if Stage 2 not run).",
                },
                "stage1_method": {
                    "type": "string",
                    "description": "Method used for Stage 1: 'similarity', 'shape_similarity', etc.",
                },
                "passed": {
                    "type": "boolean",
                    "description": "True if final NMI >= 1.10 (Stage 2 threshold).",
                },
            },
            "required": ["status", "nmi_stage1", "nmi_final", "stage1_method", "passed"],
        },
    },
    {
        "name": "run_rigid_registration",
        "description": (
            "Stage 1: Perform rigid registration (translation + rotation) between two "
            "cached grayscale images using SimpleITK MattesMutualInformation metric and "
            "LBFGSB optimiser with multi-resolution pyramid. "
            "Returns transform_id and NMI score. Target NMI > 0.3."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fixed_id": {"type": "string", "description": "image_id of the fixed (reference) image."},
                "moving_id": {"type": "string", "description": "image_id of the moving image to register."},
                "sampling_pct": {
                    "type": "number",
                    "description": "Fraction of pixels to sample for metric (0.05-0.3). Default 0.1.",
                },
            },
            "required": ["fixed_id", "moving_id"],
        },
    },
    {
        "name": "run_affine_registration",
        "description": (
            "Stage 2: Refine registration with an affine transform (includes scale + shear) "
            "starting from a previous rigid transform. Use higher-resolution images than Stage 1. "
            "Returns transform_id and NMI score. Target NMI > 0.4."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fixed_id": {"type": "string", "description": "image_id of the fixed image (higher res than Stage 1)."},
                "moving_id": {"type": "string", "description": "image_id of the moving image (higher res than Stage 1)."},
                "init_transform_id": {
                    "type": "string",
                    "description": "transform_id from Stage 1 rigid registration.",
                },
            },
            "required": ["fixed_id", "moving_id", "init_transform_id"],
        },
    },
    {
        "name": "run_nonrigid_registration",
        "description": (
            "Stage 3: Global coarse BSpline to correct interior tissue deformations not fixed by boundary correction. "
            "Pass the RGB image IDs (NOT __H channel) — the function converts to grayscale internally. "
            "SAFE usage: mesh_size=4 (coarse, global low-freq correction). "
            "RISKY: mesh_size>=8 at thumbnail resolution may overfit and cause white artifacts — avoid. "
            "Pass the best Stage 2 transform_id as init_transform_id. "
            "Keep result only if NMI improves over Stage 2; revert otherwise. "
            "Returns transform_id and NMI score. Target NMI > Stage 2 NMI."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fixed_id": {"type": "string", "description": "image_id of the fixed image."},
                "moving_id": {"type": "string", "description": "image_id of the moving image."},
                "init_transform_id": {
                    "type": "string",
                    "description": "transform_id from Stage 2 affine registration.",
                },
                "mesh_size": {
                    "type": "integer",
                    "description": "BSpline mesh grid size per dimension (default 8). Try 4-16.",
                },
            },
            "required": ["fixed_id", "moving_id", "init_transform_id"],
        },
    },
    {
        "name": "run_boundary_correction",
        "description": (
            "Stage 2 (preferred after Similarity): BSpline correction focused only on the "
            "tissue boundary zone using GRAYSCALE images. "
            "Computes an Otsu tissue mask on the HE image, erodes it to find the interior, "
            "and uses the boundary ring (tissue - interior) as SetMetricFixedMask so the "
            "optimizer only fits pixels at the tissue edge. "
            "This fixes boundary misalignment left by the H-channel Similarity without "
            "disturbing the well-aligned interior. "
            "IMPORTANT: pass the original RGB image IDs (not __H channel IDs) so the "
            "function can use full grayscale. "
            "Returns transform_id and NMI score. Target NMI > 1.10."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fixed_id": {"type": "string", "description": "image_id of the fixed RGB image (NOT __H channel)."},
                "moving_id": {"type": "string", "description": "image_id of the moving RGB image (NOT __H channel)."},
                "init_transform_id": {"type": "string", "description": "transform_id from Stage 1 Similarity."},
                "boundary_width_px": {
                    "type": "integer",
                    "description": "Width of boundary ring in pixels (default 60).",
                },
                "mesh_size": {
                    "type": "integer",
                    "description": "BSpline mesh grid size (default 8).",
                },
            },
            "required": ["fixed_id", "moving_id", "init_transform_id"],
        },
    },
    {
        "name": "run_tile_nonrigid_registration",
        "description": (
            "Stage 3 (alternative): Tile-based BSpline non-rigid registration. "
            "Splits the image into overlapping 2048×2048 tiles and runs BSpline independently "
            "on each tile, then blends displacement fields with Hanning weights. "
            "Better than global BSpline when different regions need different deformations "
            "(e.g., right side misaligned while left side is already good). "
            "Returns transform_id and NMI score."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fixed_id": {"type": "string"},
                "moving_id": {"type": "string"},
                "init_transform_id": {"type": "string"},
                "mesh_size": {"type": "integer", "description": "BSpline mesh per tile (default 4)."},
                "tile_size": {"type": "integer", "description": "Tile size in pixels (default 2048)."},
            },
            "required": ["fixed_id", "moving_id", "init_transform_id"],
        },
    },
    {
        "name": "run_tile_feature_registration",
        "description": (
            "Stage 3 (preferred after boundary correction): Tile-based AKAZE/ORB feature matching "
            "for cellular-level residual alignment (~10-30px corrections). "
            "Splits image into overlapping 512×512 tiles; for each tissue tile, matches keypoints "
            "between fixed and pre-aligned moving, extracts per-tile translation corrections, "
            "and blends displacement fields with Hanning windows. "
            "Tiles where feature matching fails or rotation exceeds 5° use identity displacement (safe fallback). "
            "After optimization, computes per-tile NCC between fixed and registered tiles and reports "
            "the average (avg_tile_ncc, range [-1,1]) as a local alignment quality metric. "
            "IMPORTANT: pass RGB image IDs (NOT __H channel) and best Stage 2 transform_id as init_transform_id. "
            "NMI improved over Stage 2 → keep; NMI not improved → revert to Stage 2 (do NOT retry). "
            "Returns transform_id, global NMI, and avg_tile_ncc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fixed_id": {
                    "type": "string",
                    "description": "image_id of the fixed RGB image (NOT __H channel).",
                },
                "moving_id": {
                    "type": "string",
                    "description": "image_id of the moving RGB image (NOT __H channel).",
                },
                "init_transform_id": {
                    "type": "string",
                    "description": "transform_id from best Stage 2 result.",
                },
                "tile_size": {
                    "type": "integer",
                    "description": "Tile size in pixels (default 512).",
                },
                "overlap": {
                    "type": "integer",
                    "description": "Overlap between adjacent tiles in pixels (default 64).",
                },
            },
            "required": ["fixed_id", "moving_id", "init_transform_id"],
        },
    },
    {
        "name": "assess_quality",
        "description": (
            "Evaluate registration quality: compute NMI and generate a checkerboard overlay PNG. "
            "Returns a report dict with nmi, threshold, passed (bool), and overlay_path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fixed_id": {"type": "string", "description": "image_id of the fixed image."},
                "moving_id": {"type": "string", "description": "image_id of the moving image."},
                "transform_id": {"type": "string", "description": "transform_id to evaluate."},
                "output_dir": {
                    "type": "string",
                    "description": "Directory to save overlay PNG. Default: 'output'.",
                },
            },
            "required": ["fixed_id", "moving_id", "transform_id"],
        },
    },
    {
        "name": "apply_full_warp",
        "description": (
            "Apply the final combined transform to the full-resolution IHC image and write "
            "an OME-TIFF pyramid output. This is the last step - produces the aligned IHC slide."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ihc_path": {"type": "string", "description": "Path to IHC SVS file."},
                "transform_id": {"type": "string", "description": "Final transform_id (from Stage 3 or best available)."},
                "he_path": {"type": "string", "description": "Path to HE SVS file (reference space)."},
                "output_path": {"type": "string", "description": "Output OME-TIFF path."},
            },
            "required": ["ihc_path", "transform_id", "he_path", "output_path"],
        },
    },
    # ── Xenium-specific tools ──────────────────────────────────────────────
    {
        "name": "load_xenium_data",
        "description": (
            "Load Xenium HE OME-TIFF and nucleus_boundaries.parquet. "
            "Extracts mpp from OME metadata, converts physical coordinates to pixel coordinates. "
            "Returns dataset_id, image shape, mpp, scale_factor, cell/vertex counts, and coordinate ranges."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "he_path": {"type": "string", "description": "Path to HE OME-TIFF image."},
                "parquet_path": {"type": "string", "description": "Path to nucleus_boundaries.parquet."},
            },
            "required": ["he_path", "parquet_path"],
        },
    },
    {
        "name": "render_contour_mask",
        "description": (
            "Render all nucleus contours as a filled binary mask image. "
            "This creates the 'moving' image for registration. "
            "Caches result to disk (filled_image.png) to avoid re-rendering. "
            "Returns image_id and shape."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "dataset_id from load_xenium_data."},
                "output_dir": {"type": "string", "description": "Directory to cache rendered mask."},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "convert_he_binary",
        "description": (
            "Convert HE image to enhanced binary mask for registration. "
            "Pipeline: contrast enhancement (alpha) → weighted RGB to gray → invert → Otsu threshold. "
            "Agent can adjust alpha (contrast), r_weight/g_weight/b_weight (RGB mixing) to optimize "
            "the binary representation for different tissue types. "
            "Returns he_rgb_id (full color), he_binary_id (binary mask), he_gray_inv_id (inverted gray)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "dataset_id from load_xenium_data."},
                "alpha": {
                    "type": "number",
                    "description": "Contrast enhancement factor (default 1.4). Higher = more contrast.",
                },
                "r_weight": {
                    "type": "number",
                    "description": "Red channel weight for grayscale conversion (default 0.3).",
                },
                "g_weight": {
                    "type": "number",
                    "description": "Green channel weight for grayscale conversion (default 0.1).",
                },
                "b_weight": {
                    "type": "number",
                    "description": "Blue channel weight for grayscale conversion (default 0.6).",
                },
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "run_feature_alignment",
        "description": (
            "Global feature-based alignment between reference (HE binary) and moving (contour mask). "
            "Methods: 'orb' (fast, adjustable nfeatures), 'akaze' (rotation-invariant), 'sift' (scale-invariant). "
            "Agent should try different methods and parameters if initial alignment is poor. "
            "scale_factor controls downsampling for feature detection (lower = faster but less precise). "
            "Returns transform_id, inlier count, rotation angle, and scale."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ref_id": {"type": "string", "description": "image_id of reference (HE binary mask)."},
                "moving_id": {"type": "string", "description": "image_id of moving (contour mask)."},
                "method": {
                    "type": "string",
                    "enum": ["orb", "akaze", "sift"],
                    "description": "Feature detection method (default 'orb').",
                },
                "nfeatures": {
                    "type": "integer",
                    "description": "Max features for ORB/SIFT (default 100000). Try 50000-200000.",
                },
                "scale_factor": {
                    "type": "number",
                    "description": "Downsampling for feature detection (default 0.2). Try 0.1-0.5.",
                },
            },
            "required": ["ref_id", "moving_id"],
        },
    },
    {
        "name": "warp_contour_mask",
        "description": (
            "Warp the contour mask image using a global affine transform. "
            "Uses block-by-block inverse warp for memory efficiency. "
            "Returns warped image_id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "moving_id": {"type": "string", "description": "image_id of contour mask to warp."},
                "transform_id": {"type": "string", "description": "transform_id from run_feature_alignment."},
                "ref_id": {"type": "string", "description": "image_id of reference image (defines output size)."},
            },
            "required": ["moving_id", "transform_id", "ref_id"],
        },
    },
    {
        "name": "apply_global_transform",
        "description": (
            "Apply global affine transform to all contour vertex coordinates. "
            "Creates vertex_x_p_trans and vertex_y_p_trans columns. "
            "Must be called before run_block_refinement."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "dataset_id from load_xenium_data."},
                "transform_id": {"type": "string", "description": "transform_id from run_feature_alignment."},
            },
            "required": ["dataset_id", "transform_id"],
        },
    },
    {
        "name": "run_block_refinement",
        "description": (
            "Per-block feature-based refinement of contour alignment. "
            "Splits tissue into a grid of blocks, runs local AKAZE/ORB alignment per block, "
            "and refines each contour's position. Blocks with too few contours are merged with neighbors. "
            "Agent can adjust num_blocks (grid density) and method. "
            "Higher num_blocks = finer local corrections but more blocks may fail. "
            "Returns total/failed block counts and refined contour count."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "dataset_id from load_xenium_data."},
                "ref_id": {"type": "string", "description": "image_id of reference binary image."},
                "warped_id": {"type": "string", "description": "image_id of warped contour mask."},
                "num_blocks": {
                    "type": "integer",
                    "description": "Grid subdivision per axis (default 32). Try 16-64.",
                },
                "min_contours": {
                    "type": "integer",
                    "description": "Min contours per block before merging (default 20).",
                },
                "method": {
                    "type": "string",
                    "enum": ["akaze", "orb"],
                    "description": "Feature method for block alignment (default 'akaze').",
                },
            },
            "required": ["dataset_id", "ref_id", "warped_id"],
        },
    },
    {
        "name": "assess_xenium_quality",
        "description": (
            "Compute alignment quality metrics: Dice coefficient, IoU, NCC, and coverage. "
            "Can be called after global alignment or after block refinement to compare improvements. "
            "Generates an overlay image showing contours on HE tissue. "
            "Dice > 0.7 is good; > 0.8 is excellent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "dataset_id from load_xenium_data."},
                "ref_binary_id": {"type": "string", "description": "image_id of HE binary mask."},
                "ref_gray_id": {
                    "type": "string",
                    "description": "image_id of HE inverted grayscale (for NCC computation).",
                },
                "output_dir": {"type": "string", "description": "Directory to save overlay and quality.json."},
            },
            "required": ["dataset_id", "ref_binary_id"],
        },
    },
    {
        "name": "save_xenium_outputs",
        "description": (
            "Save final transformed contours (pickle) and affine matrices (npy) to output directory. "
            "Call this as the last step after the best alignment is achieved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "dataset_id from load_xenium_data."},
                "output_dir": {"type": "string", "description": "Directory to save outputs."},
            },
            "required": ["dataset_id", "output_dir"],
        },
    },
    {
        "name": "finish_xenium_registration",
        "description": (
            "MANDATORY LAST STEP: Record the final Xenium registration result to result.json. "
            "Call this after assess_xenium_quality with final metrics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["success", "failed"],
                    "description": "'success' if alignment achieved acceptable Dice, 'failed' if all attempts failed.",
                },
                "dataset_id": {"type": "string", "description": "dataset_id."},
                "dice": {"type": "number", "description": "Final Dice coefficient."},
                "iou": {"type": "number", "description": "Final IoU."},
                "ncc": {"type": "number", "description": "Final NCC."},
                "coverage": {"type": "number", "description": "Final coverage."},
                "global_method": {"type": "string", "description": "Method used for global alignment."},
                "block_method": {"type": "string", "description": "Method used for block refinement (if run)."},
                "output_dir": {"type": "string", "description": "Output directory."},
            },
            "required": ["status", "dataset_id", "dice", "output_dir"],
        },
    },

    # ── Strategy C deterministic pipelines (HE-IHC improved, 2026) ──
    {
        "name": "run_improved_fast",
        "description": (
            "HE-IHC strategy-C fast pipeline (LightGlue + nonrigid + SimpleITK Demons). "
            "Registers raw HE/IHC SVS in one shot (30-60s on GPU). Returns QCS@512 and "
            "warped IHC path. Use this as the DEFAULT first tool for new HE-IHC pairs "
            "— it wins 9/12 cases in benchmarks. If QCS ≥ 0.70, call finish_registration; "
            "otherwise try run_improved_auto."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "he_path": {"type": "string", "description": "Path to HE SVS/TIFF."},
                "ihc_path": {"type": "string", "description": "Path to IHC SVS/TIFF."},
                "output_dir": {"type": "string", "description": "Output directory."},
                "recipe": {
                    "type": "string",
                    "description": "Recipe: 'nr3_demons300' (default, best general), "
                                   "'iter4' (fastest), 'nr6_far4' (hard cases).",
                },
            },
            "required": ["he_path", "ihc_path", "output_dir"],
        },
    },
    {
        "name": "run_improved_auto",
        "description": (
            "HE-IHC strategy-C auto pipeline — tries all 6 recipes (iter4, iter6, "
            "nr3_far2, nr4_far2, nr6_far4, nr3_demons300) and returns the best "
            "QCS@512. Slower (2-4 min) but most likely to recover hard cases. Use "
            "after run_improved_fast if QCS was < 0.70."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "he_path": {"type": "string"},
                "ihc_path": {"type": "string"},
                "output_dir": {"type": "string"},
            },
            "required": ["he_path", "ihc_path", "output_dir"],
        },
    },
    {
        "name": "run_improved_rescue",
        "description": (
            "Rescue mode — refine an existing warped IHC (e.g. produced by classical "
            "tool chain: run_similarity_registration + run_boundary_correction, or a "
            "prior pipeline output) using LightGlue affine refinement + nonrigid + Demons. "
            "Useful when classical tools produced a reasonable alignment but local fit "
            "needs improvement."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "he_path": {"type": "string"},
                "warped_ihc_path": {"type": "string",
                                     "description": "Path to existing warped IHC PNG/TIF."},
                "output_dir": {"type": "string"},
                "recipe": {"type": "string",
                           "description": "Default 'nr3_demons300'."},
            },
            "required": ["he_path", "warped_ihc_path", "output_dir"],
        },
    },
]


# --- Execution functions ---

def execute_tool(name: str, inputs: dict) -> str:
    """Dispatch tool call to the appropriate function. Returns JSON string."""
    try:
        if name == "load_wsi_thumbnail":
            return _tool_load_wsi_thumbnail(**inputs)
        elif name == "run_similarity_registration":
            return _tool_run_similarity_registration(**inputs)
        elif name == "run_shape_similarity":
            return _tool_run_shape_similarity(**inputs)
        elif name == "finish_registration":
            return _tool_finish_registration(**inputs)
        elif name == "get_slide_info":
            return _tool_get_slide_info(**inputs)
        elif name == "extract_hematoxylin":
            return _tool_extract_hematoxylin(**inputs)
        elif name == "to_grayscale":
            return _tool_to_grayscale(**inputs)
        elif name == "run_rigid_registration":
            return _tool_run_rigid_registration(**inputs)
        elif name == "run_affine_registration":
            return _tool_run_affine_registration(**inputs)
        elif name == "run_nonrigid_registration":
            return _tool_run_nonrigid_registration(**inputs)
        elif name == "run_boundary_correction":
            return _tool_run_boundary_correction(**inputs)
        elif name == "run_tile_nonrigid_registration":
            return _tool_run_tile_nonrigid_registration(**inputs)
        elif name == "run_tile_feature_registration":
            return _tool_run_tile_feature_registration(**inputs)
        elif name == "assess_quality":
            return _tool_assess_quality(**inputs)
        elif name == "apply_full_warp":
            return _tool_apply_full_warp(**inputs)
        elif name == "load_xenium_data":
            return _tool_load_xenium_data(**inputs)
        elif name == "render_contour_mask":
            return _tool_render_contour_mask(**inputs)
        elif name == "convert_he_binary":
            return _tool_convert_he_binary(**inputs)
        elif name == "run_feature_alignment":
            return _tool_run_feature_alignment(**inputs)
        elif name == "warp_contour_mask":
            return _tool_warp_contour_mask(**inputs)
        elif name == "apply_global_transform":
            return _tool_apply_global_transform(**inputs)
        elif name == "run_block_refinement":
            return _tool_run_block_refinement(**inputs)
        elif name == "assess_xenium_quality":
            return _tool_assess_xenium_quality(**inputs)
        elif name == "save_xenium_outputs":
            return _tool_save_xenium_outputs(**inputs)
        elif name == "finish_xenium_registration":
            return _tool_finish_xenium_registration(**inputs)
        elif name == "run_improved_fast":
            return _tool_run_improved_fast(**inputs)
        elif name == "run_improved_auto":
            return _tool_run_improved_auto(**inputs)
        elif name == "run_improved_rescue":
            return _tool_run_improved_rescue(**inputs)
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


def _tool_load_wsi_thumbnail(path: str, level: int) -> str:
    from registration.preprocess import load_wsi_thumbnail, get_slide_info
    image_id, img = load_wsi_thumbnail(path, level)
    return json.dumps({
        "image_id": image_id,
        "shape": list(img.shape),
        "dtype": str(img.dtype),
        "level_loaded": level,
    })


def _tool_get_slide_info(path: str) -> str:
    from registration.preprocess import get_slide_info
    info = get_slide_info(path)
    # Convert tuples to lists for JSON serialisation
    info["level_dimensions"] = [list(d) for d in info["level_dimensions"]]
    info["level_downsamples"] = list(info["level_downsamples"])
    return json.dumps(info)


def _tool_extract_hematoxylin(image_id: str) -> str:
    from registration.preprocess import extract_hematoxylin
    new_id, h_img = extract_hematoxylin(image_id)
    return json.dumps({
        "image_id": new_id,
        "shape": list(h_img.shape),
        "dtype": str(h_img.dtype),
        "min": float(h_img.min()),
        "max": float(h_img.max()),
        "mean": float(h_img.mean()),
    })


def _tool_to_grayscale(image_id: str) -> str:
    from registration.preprocess import to_grayscale
    new_id, gray = to_grayscale(image_id)
    return json.dumps({
        "image_id": new_id,
        "shape": list(gray.shape),
        "dtype": str(gray.dtype),
    })


def _tool_run_similarity_registration(
    fixed_id: str, moving_id: str, sampling_pct: float = 0.1
) -> str:
    from registration.global_reg import run_similarity_registration
    print(f"[tool] Stage 1 similarity registration: {fixed_id} <- {moving_id} (sampling={sampling_pct})")
    transform_id, nmi = run_similarity_registration(fixed_id, moving_id, sampling_pct)
    passed = nmi >= 1.05
    print(f"[tool] Similarity NMI={nmi:.4f}, passed={passed}")
    return json.dumps({
        "transform_id": transform_id,
        "nmi": nmi,
        "threshold": 1.05,
        "passed": passed,
        "stage": "similarity",
    })


def _tool_run_shape_similarity(
    fixed_id: str, moving_id: str, sampling_pct: float = 0.1
) -> str:
    from registration.global_reg import run_shape_similarity
    print(f"[tool] Stage 1 shape-similarity registration: {fixed_id} <- {moving_id} (sampling={sampling_pct})")
    transform_id, nmi = run_shape_similarity(fixed_id, moving_id, sampling_pct)
    passed = nmi >= 1.05
    print(f"[tool] Shape-similarity NMI={nmi:.4f}, passed={passed}")
    return json.dumps({
        "transform_id": transform_id,
        "nmi": nmi,
        "threshold": 1.05,
        "passed": passed,
        "stage": "shape_similarity",
    })


def _tool_finish_registration(
    status: str, nmi_stage1: float, nmi_final: float, stage1_method: str, passed: bool
) -> str:
    import agent.tools as _tools_mod
    output_dir = _tools_mod._current_output_dir
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "status": status,
        "nmi_stage1": round(float(nmi_stage1), 4),
        "nmi_final": round(float(nmi_final), 4),
        "stage1_method": stage1_method,
        "passed": bool(passed),
    }
    result_path = os.path.join(output_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[tool] finish_registration: status={status}, nmi_stage1={nmi_stage1:.4f}, "
          f"nmi_final={nmi_final:.4f}, passed={passed} -> {result_path}")
    return json.dumps({"written": result_path, **result})


def _tool_run_rigid_registration(
    fixed_id: str, moving_id: str, sampling_pct: float = 0.1
) -> str:
    from registration.global_reg import run_rigid_registration
    print(f"[tool] Stage 1 rigid registration: {fixed_id} <- {moving_id} (sampling={sampling_pct})")
    transform_id, nmi = run_rigid_registration(fixed_id, moving_id, sampling_pct)
    passed = nmi >= 1.05  # Stage 1 threshold
    print(f"[tool] Rigid NMI={nmi:.4f}, passed={passed}")
    return json.dumps({
        "transform_id": transform_id,
        "nmi": nmi,
        "threshold": 1.05,
        "passed": passed,
        "stage": "rigid",
    })


def _tool_run_affine_registration(
    fixed_id: str, moving_id: str, init_transform_id: str
) -> str:
    from registration.global_reg import run_affine_registration
    print(f"[tool] Stage 2 affine registration: {fixed_id} <- {moving_id}")
    transform_id, nmi = run_affine_registration(fixed_id, moving_id, init_transform_id)
    passed = nmi >= 1.06  # Stage 2 threshold
    print(f"[tool] Affine NMI={nmi:.4f}, passed={passed}")
    return json.dumps({
        "transform_id": transform_id,
        "nmi": nmi,
        "threshold": 1.06,
        "passed": passed,
        "stage": "affine",
    })


def _tool_run_nonrigid_registration(
    fixed_id: str, moving_id: str, init_transform_id: str, mesh_size: int = 8
) -> str:
    from registration.nonrigid_reg import run_nonrigid_registration
    print(f"[tool] Stage 3 nonrigid registration: mesh_size={mesh_size}")
    transform_id, nmi = run_nonrigid_registration(
        fixed_id, moving_id, init_transform_id, mesh_size=mesh_size
    )
    passed = nmi >= 1.07  # Stage 3 threshold
    print(f"[tool] Nonrigid NMI={nmi:.4f}, passed={passed}")
    return json.dumps({
        "transform_id": transform_id,
        "nmi": nmi,
        "threshold": 1.07,
        "passed": passed,
        "stage": "nonrigid",
    })


def _tool_run_boundary_correction(
    fixed_id: str, moving_id: str, init_transform_id: str,
    boundary_width_px: int = 60, mesh_size: int = 8
) -> str:
    from registration.nonrigid_reg import run_boundary_correction
    print(f"[tool] Boundary correction: boundary={boundary_width_px}px mesh={mesh_size}")
    transform_id, nmi = run_boundary_correction(
        fixed_id, moving_id, init_transform_id,
        boundary_width_px=boundary_width_px, mesh_size=mesh_size
    )
    passed = nmi >= 1.10
    print(f"[tool] Boundary correction NMI={nmi:.4f}, passed={passed}")
    return json.dumps({
        "transform_id": transform_id,
        "nmi": nmi,
        "threshold": 1.10,
        "passed": passed,
        "stage": "boundary_correction",
    })


def _tool_run_tile_nonrigid_registration(
    fixed_id: str, moving_id: str, init_transform_id: str,
    mesh_size: int = 4, tile_size: int = 2048
) -> str:
    from registration.nonrigid_reg import run_tile_based_nonrigid
    print(f"[tool] Tile-based nonrigid: mesh={mesh_size}, tile={tile_size}")
    transform_id, nmi = run_tile_based_nonrigid(
        fixed_id, moving_id, init_transform_id, mesh_size=mesh_size, tile_size=tile_size
    )
    passed = nmi >= 1.07
    print(f"[tool] Tile nonrigid NMI={nmi:.4f}, passed={passed}")
    return json.dumps({
        "transform_id": transform_id,
        "nmi": nmi,
        "threshold": 1.07,
        "passed": passed,
        "stage": "nonrigid",
    })


def _tool_run_tile_feature_registration(
    fixed_id: str, moving_id: str, init_transform_id: str,
    tile_size: int = 512, overlap: int = 64,
) -> str:
    from registration.nonrigid_reg import run_tile_feature_registration
    print(f"[tool] Tile-feature registration: tile={tile_size}px overlap={overlap}px")
    transform_id, nmi, avg_tile_ncc = run_tile_feature_registration(
        fixed_id, moving_id, init_transform_id,
        tile_size=tile_size, overlap=overlap,
    )
    passed = nmi >= 1.10
    print(f"[tool] Tile-feature NMI={nmi:.4f}, avg_tile_NCC={avg_tile_ncc:.4f}, passed={passed}")
    return json.dumps({
        "transform_id": transform_id,
        "nmi": nmi,
        "avg_tile_ncc": avg_tile_ncc,
        "threshold": 1.10,
        "passed": passed,
        "stage": "tile_feature",
    })


def _tool_assess_quality(
    fixed_id: str, moving_id: str, transform_id: str, output_dir: str = "output"
) -> str:
    from registration.quality import assess_quality
    report = assess_quality(fixed_id, moving_id, transform_id, output_dir)
    # Convert numpy types to plain python for JSON
    return json.dumps({k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                           float(v) if isinstance(v, (float, np.floating)) else v)
                       for k, v in report.items()})


def _tool_apply_full_warp(
    ihc_path: str, transform_id: str, he_path: str, output_path: str
) -> str:
    from registration.warp import apply_full_warp
    print(f"[tool] Applying full-resolution warp -> {output_path}")
    result = apply_full_warp(ihc_path, transform_id, he_path, output_path)
    size_mb = os.path.getsize(result) / 1e6 if os.path.exists(result) else 0
    return json.dumps({
        "output_path": result,
        "size_mb": round(size_mb, 1),
        "status": "success",
    })


# --- Xenium tool implementations ---

def _tool_load_xenium_data(he_path: str, parquet_path: str) -> str:
    from registration.xenium_reg import load_xenium_data
    print(f"[tool] Loading Xenium data: HE={he_path}, parquet={parquet_path}")
    result = load_xenium_data(he_path, parquet_path)
    print(f"[tool] Dataset '{result['dataset_id']}': {result['n_cells']} cells, "
          f"HE shape={result['he_shape']}, mpp={result['mpp']}")
    return json.dumps(result)


def _tool_render_contour_mask(dataset_id: str, output_dir: str = None) -> str:
    from registration.xenium_reg import render_contour_mask
    print(f"[tool] Rendering contour mask for {dataset_id}")
    result = render_contour_mask(dataset_id, output_dir)
    print(f"[tool] Contour mask: shape={result['shape']}, cached={result.get('cached', False)}")
    return json.dumps(result)


def _tool_convert_he_binary(dataset_id: str, alpha: float = 1.4,
                             r_weight: float = 0.3, g_weight: float = 0.1,
                             b_weight: float = 0.6) -> str:
    from registration.xenium_reg import convert_he_binary
    print(f"[tool] Converting HE to binary: alpha={alpha}, weights=[{r_weight},{g_weight},{b_weight}]")
    result = convert_he_binary(dataset_id, alpha=alpha, r_weight=r_weight,
                                g_weight=g_weight, b_weight=b_weight)
    print(f"[tool] HE binary: shape={result['shape']}")
    return json.dumps(result)


def _tool_run_feature_alignment(ref_id: str, moving_id: str,
                                 method: str = 'orb', nfeatures: int = 100000,
                                 scale_factor: float = 0.2) -> str:
    from registration.xenium_reg import run_feature_alignment
    print(f"[tool] Feature alignment: method={method}, nfeatures={nfeatures}, scale={scale_factor}")
    result = run_feature_alignment(ref_id, moving_id, method=method,
                                    nfeatures=nfeatures, scale_factor=scale_factor)
    if result.get('transform_id'):
        print(f"[tool] Alignment: inliers={result['inliers']}, "
              f"angle={result['angle_deg']}°, scale={result['scale']}")
    else:
        print(f"[tool] Alignment failed: {result.get('error')}")
    return json.dumps(result)


def _tool_warp_contour_mask(moving_id: str, transform_id: str, ref_id: str) -> str:
    from registration.xenium_reg import warp_contour_mask
    print(f"[tool] Warping contour mask with {transform_id}")
    result = warp_contour_mask(moving_id, transform_id, ref_id)
    print(f"[tool] Warped mask: shape={result['shape']}")
    return json.dumps(result)


def _tool_apply_global_transform(dataset_id: str, transform_id: str) -> str:
    from registration.xenium_reg import apply_global_transform
    print(f"[tool] Applying global transform to contour coordinates")
    result = apply_global_transform(dataset_id, transform_id)
    print(f"[tool] Transformed {result['n_vertices_transformed']} vertices")
    return json.dumps(result)


def _tool_run_block_refinement(dataset_id: str, ref_id: str, warped_id: str,
                                num_blocks: int = 32, min_contours: int = 20,
                                method: str = 'akaze') -> str:
    from registration.xenium_reg import run_block_refinement
    print(f"[tool] Block refinement: num_blocks={num_blocks}, method={method}")
    result = run_block_refinement(dataset_id, ref_id, warped_id,
                                   num_blocks=num_blocks, min_contours=min_contours,
                                   method=method)
    print(f"[tool] Blocks: {result['total_blocks']} total, {result['failed_blocks']} failed, "
          f"{result['n_contours']} contours refined")
    return json.dumps(result)


def _tool_assess_xenium_quality(dataset_id: str, ref_binary_id: str,
                                 ref_gray_id: str = None, output_dir: str = None) -> str:
    from registration.xenium_reg import assess_xenium_quality
    print(f"[tool] Assessing Xenium quality for {dataset_id}")
    result = assess_xenium_quality(dataset_id, ref_binary_id,
                                    ref_gray_id=ref_gray_id, output_dir=output_dir)
    if 'error' not in result:
        print(f"[tool] Quality: Dice={result['dice']}, IoU={result['iou']}, "
              f"NCC={result['ncc']}, coverage={result['coverage']}")
    else:
        print(f"[tool] Quality error: {result['error']}")
    return json.dumps(result)


def _tool_save_xenium_outputs(dataset_id: str, output_dir: str) -> str:
    from registration.xenium_reg import save_outputs
    print(f"[tool] Saving Xenium outputs to {output_dir}")
    result = save_outputs(dataset_id, output_dir)
    print(f"[tool] Saved {result['n_contours']} contours")
    return json.dumps(result)


def _tool_finish_xenium_registration(status: str, dataset_id: str, dice: float,
                                      output_dir: str, iou: float = 0.0,
                                      ncc: float = 0.0, coverage: float = 0.0,
                                      global_method: str = '', block_method: str = '') -> str:
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "status": status,
        "dataset_id": dataset_id,
        "dice": round(float(dice), 4),
        "iou": round(float(iou), 4),
        "ncc": round(float(ncc), 4),
        "coverage": round(float(coverage), 4),
        "global_method": global_method,
        "block_method": block_method,
    }
    result_path = os.path.join(output_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[tool] finish_xenium_registration: status={status}, dice={dice:.4f} -> {result_path}")
    return json.dumps({"written": result_path, **result})


# ── Strategy C deterministic pipeline tools (HE-IHC improved, 2026) ──

def _load_improved_pipeline():
    """Lazy import of pipelines/he_ihc/run_he_ihc.py — only when agent calls it."""
    import sys
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PIPELINE_DIR = os.path.abspath(os.path.join(_HERE, '..', 'pipelines', 'he_ihc'))
    if _PIPELINE_DIR not in sys.path:
        sys.path.insert(0, _PIPELINE_DIR)
    import run_he_ihc as pipe
    return pipe


def _tool_run_improved_fast(he_path: str, ihc_path: str, output_dir: str,
                             recipe: str = 'nr3_demons300') -> str:
    """Fast deterministic HE-IHC: LightGlue + nonrigid + Demons (single recipe).
    30-60s on GPU. Returns QCS + warped path for the agent to inspect."""
    import time
    from PIL import Image as _PIL
    _PIL.MAX_IMAGE_PIXELS = None
    pipe = _load_improved_pipeline()
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    he_rgb = pipe.load_svs(he_path)
    ihc_rgb = pipe.load_svs(ihc_path)
    he_gray = pipe.to_gray(he_rgb); ihc_gray = pipe.to_gray(ihc_rgb)
    warped, chosen_recipe, metrics = pipe.register(
        he_gray, ihc_gray, recipe=recipe, verbose=False)
    out_png = os.path.join(output_dir, f'improved_fast_warped.png')
    _PIL.fromarray(warped).save(out_png)
    elapsed = time.time() - t0
    qcs = float(metrics['qcs']) if metrics else 0.0
    print(f"[tool] run_improved_fast: recipe={chosen_recipe}, QCS@512={qcs:.3f}, "
          f"elapsed={elapsed:.1f}s")
    return json.dumps({
        "recipe_used": chosen_recipe,
        "qcs_512": qcs,
        "warped_path": out_png,
        "metrics": {k: float(v) for k, v in (metrics or {}).items()
                    if isinstance(v, (int, float))},
        "elapsed_seconds": round(elapsed, 1),
        "he_shape": list(he_gray.shape),
        "ihc_shape": list(ihc_gray.shape),
    })


def _tool_run_improved_auto(he_path: str, ihc_path: str, output_dir: str) -> str:
    """Auto mode — try all 6 recipes, pick best QCS. 2-4 min on GPU."""
    import time
    from PIL import Image as _PIL
    _PIL.MAX_IMAGE_PIXELS = None
    pipe = _load_improved_pipeline()
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    he_gray = pipe.to_gray(pipe.load_svs(he_path))
    ihc_gray = pipe.to_gray(pipe.load_svs(ihc_path))
    warped, chosen, metrics = pipe.register(he_gray, ihc_gray,
                                               recipe='auto', verbose=False)
    out_png = os.path.join(output_dir, f'improved_auto_warped.png')
    _PIL.fromarray(warped).save(out_png)
    elapsed = time.time() - t0
    qcs = float(metrics['qcs']) if metrics else 0.0
    print(f"[tool] run_improved_auto: recipe={chosen}, QCS@512={qcs:.3f}, "
          f"elapsed={elapsed:.1f}s")
    return json.dumps({
        "recipe_used": chosen,
        "qcs_512": qcs,
        "warped_path": out_png,
        "metrics": {k: float(v) for k, v in (metrics or {}).items()
                    if isinstance(v, (int, float))},
        "elapsed_seconds": round(elapsed, 1),
    })


def _tool_run_improved_rescue(he_path: str, warped_ihc_path: str, output_dir: str,
                               recipe: str = 'nr3_demons300') -> str:
    """Rescue: refine an existing warped IHC (e.g. produced by classical tool chain)."""
    import time
    import cv2 as _cv2
    from PIL import Image as _PIL
    _PIL.MAX_IMAGE_PIXELS = None
    pipe = _load_improved_pipeline()
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    he_gray = pipe.to_gray(pipe.load_svs(he_path))
    warped_raw = np.array(_PIL.open(warped_ihc_path))
    if warped_raw.ndim == 3:
        warped_raw = _cv2.cvtColor(warped_raw, _cv2.COLOR_RGB2GRAY)
    # Detect pre-inverted overlay
    if warped_raw.mean() < 128:
        warped_raw = 255 - warped_raw
    if warped_raw.shape != he_gray.shape:
        warped_raw = _cv2.resize(warped_raw, (he_gray.shape[1], he_gray.shape[0]),
                                   interpolation=_cv2.INTER_LINEAR)
    warped, chosen, metrics = pipe.register(he_gray, warped_raw,
                                               recipe=recipe, verbose=False)
    out_png = os.path.join(output_dir, f'improved_rescue_warped.png')
    _PIL.fromarray(warped).save(out_png)
    elapsed = time.time() - t0
    qcs = float(metrics['qcs']) if metrics else 0.0
    print(f"[tool] run_improved_rescue: recipe={chosen}, QCS@512={qcs:.3f}, "
          f"elapsed={elapsed:.1f}s")
    return json.dumps({
        "recipe_used": chosen,
        "qcs_512": qcs,
        "warped_path": out_png,
        "metrics": {k: float(v) for k, v in (metrics or {}).items()
                    if isinstance(v, (int, float))},
        "elapsed_seconds": round(elapsed, 1),
    })
