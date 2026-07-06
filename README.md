# ACCREDIT

**Agent-based Cell-resolved Cross-modal Registration Engine with Dynamic Iterative Tuning**

ACCREDIT is a unified image registration framework for computational pathology that automatically aligns multi-modal whole-slide images to H&E references. It supports six imaging modalities through a single entry point, with content-based modality detection, composite quality scoring, and an LLM-powered rescue agent for difficult cases.

<p align="center">
  <img src="assets/pipeline_overview.png" alt="ACCREDIT Pipeline Overview" width="100%">
</p>

## Supported Modalities

| Modality | Input | Method |
|----------|-------|--------|
| **Boundary** | Xenium nucleus parquet → HE | Contour-mask affine + block refinement |
| **DAPI** | DAPI/morphology fluorescence → HE | ROI flip search + phase correlation |
| **CODEX** | Multi-channel protein fluorescence → HE | QCS-based channel selection + ROI registration |
| **HE-IHC** | IHC adjacent section → HE | LightGlue + DHR non-rigid + Demons |
| **HE-HE** | Cross-platform HE → HE | Feature matching + Similarity + BSpline |
| **ST-HE** | Spatial transcriptomics → HE | Spot-pattern alignment |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/LeeZhou-bearway/ACCREDIT.git
cd ACCREDIT

# Create conda environment
conda create -n accredit python=3.10
conda activate accredit

# Install dependencies
pip install numpy opencv-python tifffile SimpleITK scikit-image openslide-python pillow
pip install torch torchvision   # for LightGlue (HE-IHC pipeline)
```

### Usage

ACCREDIT uses a single unified entry point. The system first analyzes image content to detect the modality, reports the result, and waits for user confirmation before proceeding.

```bash
# Auto-detect modality (recommended)
python run_unified_agent.py \
    --he data/HE_image.svs \
    --moving data/moving_image.svs \
    --output_dir output/

# Explicit modality
python run_unified_agent.py \
    --he data/HE_image.tif \
    --moving data/morphology.ome.tif \
    --mode dapi \
    --output_dir output/

# Boundary mode with parquet
python run_unified_agent.py \
    --he data/he_image.ome.tif \
    --parquet data/nucleus_boundaries.parquet \
    --output_dir output/

# Batch mode (skips interactive confirmation)
python run_unified_agent.py \
    --he data/HE_1.svs data/HE_2.svs \
    --moving data/IHC_1.svs data/IHC_2.svs \
    --output_dir output/

# Skip confirmation with -y flag
python run_unified_agent.py \
    --he data/HE.svs --moving data/IHC.svs \
    --output_dir output/ -y
```

### Interactive Modality Confirmation

When run in single-image mode, ACCREDIT reports its detection and asks for confirmation:

```
============================================================
ACCREDIT — Modality Detection Report
============================================================
  Fixed (HE):  data/HE_image.svs
  Moving:      data/sample_P40.svs

  Detected modality:  he-ihc
  Description:        HE-IHC (IHC adjacent section → HE, LightGlue + DHR + Demons)

  Image analysis:
    Imaging type:   brightfield
    Stain type:     IHC
============================================================

Available modalities: boundary, dapi, codex, he-ihc, he-he, st-he, senior

Proceed with this modality? [Y/n/modality name to override]:
```

You can confirm (`Y`), abort (`n`), or override by typing a modality name (e.g., `he-he`).

## Modality Detection

ACCREDIT detects image modality through a multi-layer analysis pipeline:

1. **File format**: `.parquet` → boundary, `.png` with spot keywords → ST-HE
2. **Image content analysis** (performed at load time for all image formats):
   - **Brightfield vs fluorescence**: dark-background ratio distinguishes fluorescence from brightfield
   - **HE vs IHC**: HED color deconvolution compares DAB (brown) and Eosin (pink) channel intensities
   - **Channel count**: >5 channels in fluorescence → CODEX, otherwise → DAPI
3. **Filename hints**: common IHC markers (P40, Ki67, CD3, CD8, PD-L1, HER2, CK, etc.) provide fast-path detection

## Quality Composite Score (QCS)

ACCREDIT evaluates registration quality using a composite metric:

```
QCS = base_score × coverage_factor × background_penalty

base_score = 0.30 × tissue_dice
           + 0.30 × boundary_dice
           + 0.15 × (1 - hausdorff / HD_MAX)
           + 0.10 × inlier_ratio
           + 0.15 × NMI
```

| Component | Weight | Description |
|-----------|--------|-------------|
| Tissue Dice | 0.30 | Overlap of tissue masks |
| Boundary Dice | 0.30 | Overlap of boundary contours |
| Hausdorff Distance | 0.15 | Mean boundary distance (inverted) |
| Inlier Ratio | 0.10 | ORB feature match inlier ratio |
| NMI | 0.15 | Normalized Mutual Information |

Penalties: low tissue coverage and warped signal on background regions reduce the final score.

## Benchmark

Evaluated on 16 Xenium DAPI→HE samples at 4000px resolution (QCS):

| Method | Mean QCS | Samples |
|--------|----------|---------|
| **ACCREDIT** | **0.7673** | 16 |
| VALIS | 0.6687 | 16 |
| DeepHistReg | 0.5909 | 16 |
| SimpleITK | 0.3451 | 14 |
| VoltRon | 0.2521 | 15 |
| PALOM | 0.1632 | 14 |

Full per-sample and per-metric results are available in `benchmark/evaluation/`. See `benchmark/BENCHMARK_METHODS.md` for details on how competing methods were installed and evaluated.

## Project Structure

```
ACCREDIT/
├── run_unified_agent.py          # Unified entry point (modality detection → confirmation → registration)
│
├── agent/                        # LLM-powered rescue agent
│   ├── agent.py                  #   Main agent loop (Claude API)
│   ├── tools.py                  #   Registration tools exposed to LLM
│   ├── rescue_tools.py           #   Rescue-specific tool implementations
│   └── rescue_prompts.py         #   Modality-specific rescue prompts
│
├── registration/                 # Shared registration modules
│   ├── preprocess.py             #   Image loading, grayscale, H-channel extraction
│   ├── global_reg.py             #   Feature matching, similarity registration
│   ├── nonrigid_reg.py           #   BSpline, tile-feature non-rigid registration
│   ├── quality.py                #   Registration quality assessment
│   ├── warp.py                   #   Image warping utilities
│   └── xenium_reg.py             #   Xenium-specific registration
│
├── quality/                      # Quality metrics
│   └── quality_metrics.py        #   QCS computation (tissue dice, boundary dice, etc.)
│
├── pipelines/                    # Modality-specific pipelines
│   ├── boundary/                 #   Xenium parquet contours → HE
│   ├── dapi/                     #   DAPI fluorescence → HE
│   ├── codex/                    #   CODEX multi-channel → HE
│   ├── he_ihc/                   #   IHC adjacent section → HE (LightGlue + DHR + Demons)
│   ├── he_he/                    #   Cross-platform HE → HE
│   ├── fluo_general/             #   General fluorescence utilities
│   ├── senior/                   #   Hard partial-match samples (multi-step)
│   └── st_he/                    #   Spatial transcriptomics → HE
│
└── benchmark/                    # Benchmark evaluation
    ├── BENCHMARK_METHODS.md      #   Competing method setup documentation
    └── evaluation/               #   QCS scripts and per-method metric tables
```

## LLM Rescue Agent

When the initial registration produces a QCS below the threshold (default 0.65), ACCREDIT can optionally invoke an LLM-powered rescue agent (requires Anthropic API key):

```bash
export ANTHROPIC_API_KEY=your-key-here
python run_unified_agent.py --he HE.svs --moving IHC.svs --output_dir output/
```

The rescue agent diagnoses registration failures and iteratively applies alternative strategies (enhancement, rotation search, channel selection, etc.) until quality improves.

## Citation

Paper in preparation. If you use ACCREDIT, please star this repository and check back for citation details.

**Authors**: Le Zhou, Ren Tao — Oregon Health & Science University (OHSU)

## License

This project is released under the [MIT License](LICENSE).
