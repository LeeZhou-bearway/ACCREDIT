#!/bin/bash
# ACCREDIT Command-Line Usage Examples
# =====================================

# ── 1. Auto-detect modality (recommended) ──────────────────────────────────
# ACCREDIT analyzes image content to detect the modality automatically,
# reports the result, and waits for user confirmation before proceeding.
python run_unified_agent.py \
    --he data/HE_image.svs \
    --moving data/moving_image.svs \
    --output_dir output/

# ── 2. Explicit modality ───────────────────────────────────────────────────
# Force a specific modality with --mode (skips content-based detection).
python run_unified_agent.py \
    --he data/HE_image.tif \
    --moving data/morphology.ome.tif \
    --mode dapi \
    --output_dir output/

# ── 3. Boundary mode with parquet ──────────────────────────────────────────
# Use --parquet as shortcut for boundary registration (Xenium data).
python run_unified_agent.py \
    --he data/he_image.ome.tif \
    --parquet data/nucleus_boundaries.parquet \
    --output_dir output/

# ── 4. Batch mode ─────────────────────────────────────────────────────────
# Pass multiple HE/moving pairs. Batch mode skips interactive confirmation.
python run_unified_agent.py \
    --he data/HE_1.svs data/HE_2.svs \
    --moving data/IHC_1.svs data/IHC_2.svs \
    --output_dir output/

# ── 5. Skip confirmation with -y flag ──────────────────────────────────────
# Use -y (or --yes) for non-interactive single-pair runs.
python run_unified_agent.py \
    --he data/HE.svs --moving data/IHC.svs \
    --output_dir output/ -y

# ── 6. With LLM rescue agent ──────────────────────────────────────────────
# Set Anthropic API key to enable automatic rescue when QCS is low.
export ANTHROPIC_API_KEY=your-key-here
python run_unified_agent.py \
    --he data/HE.svs --moving data/IHC.svs \
    --output_dir output/

# ── 7. Sample directory (Xenium) ──────────────────────────────────────────
# Point to a Xenium sample directory; ACCREDIT auto-detects available files.
python run_unified_agent.py \
    --sample_dir data/Xenium_Prime_Breast_Cancer_FFPE \
    --output_dir output/
