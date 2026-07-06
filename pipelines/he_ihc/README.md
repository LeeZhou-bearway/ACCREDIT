# ACCREDIT HE-IHC Registration Pipeline (2026)

## Introduction

ACCREDIT's HE-IHC pipeline registers immunohistochemistry (IHC) whole-slide images
to adjacent-section H&E references. It is designed around a core principle:
**fast deterministic methods handle the common case, and an adaptive LLM agent
handles the hard cases** — never paying for LLM reasoning when simpler tools
suffice, but always having the agent's flexibility available when needed.

The pipeline integrates three best-in-class components:
1. **SuperPoint + LightGlue** — pretrained cross-modal feature matching (MagicLeap
   + CVG weights, no fine-tuning needed)
2. **DHR-style instance-optimization nonrigid** — per-pixel displacement field
   optimized by Adam with local-NCC cost and diffusion regularization
   (reimplemented from [DeeperHistReg](https://github.com/MWod/DeeperHistReg) in
   pure PyTorch)
3. **SimpleITK Fast Symmetric Forces Demons** — final polish step for cross-modal
   registration

On a 12-pair PASC HE-P40-IHC benchmark, this pipeline achieves **mean QCS 0.886**
(at 512-pixel evaluation), outperforming VALIS (0.707), DHR (0.623), and four
other baselines on **all 12 pairs** with a mean margin of +0.180.

---

## Architecture — Strategy C Cascade

The pipeline is a three-stage escalating cascade. Each stage has a QCS threshold;
if it passes, later stages are skipped (saving time and API cost).

```
Input: raw HE.svs + raw IHC.svs
             │
             ▼
  ┌────────────────────────────────────┐
  │  STAGE 1 — FAST                    │  Single recipe: dhr3_demons300
  │  LightGlue + DHR-NR + Demons       │  (LG×3 + DHR-NR×3 + Demons 300 iter)
  │  ≈ 30–60 s on GPU                  │
  └────────────────┬───────────────────┘
                   │
       QCS@512 ≥ 0.70 ?  ─── yes ──▶ DONE  (~90% of pairs end here)
                   │
                   no
                   ▼
  ┌────────────────────────────────────┐
  │  STAGE 2 — AUTO                    │  Try all 6 recipes,
  │  6 recipes, pick highest QCS       │  keep the best QCS.
  │  ≈ 2–4 min on GPU                  │
  └────────────────┬───────────────────┘
                   │
       QCS@512 ≥ 0.60 ?  ─── yes ──▶ DONE
                   │
                   no
                   ▼
  ┌────────────────────────────────────┐
  │  STAGE 3 — LLM AGENT (rescue)      │  Claude agent with full ACCREDIT
  │  Adaptive tool orchestration       │  tool suite, including the
  │  + refine-agent-output             │  improved deterministic tools.
  │  ≈ minutes, Anthropic API          │  Refines its own output with DHR+Demons.
  └────────────────┬───────────────────┘
                   │
                   ▼
         pick max(QCS) over all stages
         generate color overlay + save
```

### Why Strategy C

| Stage | Purpose | When it runs |
|---|---|---|
| **FAST** | Default path — wins 9/12 on benchmark with single recipe | Always first |
| **AUTO** | Coverage for hard-but-tractable pairs | If fast < threshold |
| **AGENT** | Safety net for unusual cases (rare stains, extreme rotations) | If auto < threshold |

This cascade gives:
- **Best speed**: most cases finish in 30-60 s, no API cost
- **Best quality**: mean QCS 0.886 (beats every other method on our benchmark)
- **Best safety**: agent available as last-resort rescue for hard cases
- **Adaptive**: if you push a stubborn pair, the system escalates automatically

---

## Agent ↔ Strategy C Integration

The integration goes both directions:

1. **Deterministic wrapper calls the agent as last rescue** (`run_unified_he_ihc.py`).
2. **The agent itself has access to Strategy C tools** (in `agent/tools.py`):
   - `run_improved_fast` — call from the agent to run stage 1
   - `run_improved_auto` — call from the agent to run stage 2
   - `run_improved_rescue` — refine an existing warped IHC

The agent's system prompt has been updated to prefer `run_improved_fast` as its
first action, keeping the classical hematoxylin+BSpline tools as fallback.

This means that even if you invoke the agent directly (`ACCREDIT_HEIHC_MODE=agent`),
it will try the fast deterministic pipeline first, and only fall back to classical
tools if that fails.

---

## Usage

### Default — unified cascade
```bash
python run_unified_agent.py \
    --mode he-ihc \
    --he HE.svs --moving IHC.svs \
    --output_dir output/
```
This runs Strategy C (fast → auto → agent-as-rescue). Best for batch processing
and production.

### Direct pipeline call
```bash
python pipelines/he_ihc/run_unified_he_ihc.py \
    --he HE.svs --ihc IHC.svs --output_dir out/ \
    [--fast_pass 0.70] [--auto_pass 0.60] \
    [--no_agent] [--always_all]
```

### Single recipe (skip cascade)
```bash
python pipelines/he_ihc/run_he_ihc.py \
    --he HE.svs --ihc IHC.svs --output_dir out/ \
    --recipe dhr3_demons300
```
Recipes: `iter4`, `iter6`, `dhr3_far2`, `dhr4_far2`, `dhr6_far4`, `dhr3_demons300`, `auto`.

### Legacy agent-only mode
```bash
ACCREDIT_HEIHC_MODE=agent python run_unified_agent.py --mode he-ihc ...
```
Forces the old LLM-agent path (still uses the improved tools internally).

---

## Configuration

### Environment variables
| Variable | Default | Purpose |
|---|---|---|
| `ACCREDIT_FAST_PASS` | `0.70` | QCS threshold to skip stages 2+3 |
| `ACCREDIT_AUTO_PASS` | `0.60` | QCS threshold to skip stage 3 |
| `ACCREDIT_FAST_RECIPE` | `dhr3_demons300` | Recipe for stage 1 |
| `ACCREDIT_SKIP_AGENT` | `0` | Set `1` to disable stage 3 entirely |
| `ACCREDIT_ALWAYS_ALL` | `0` | Set `1` to run all 3 stages and keep best |
| `ACCREDIT_HEIHC_MODE` | `unified` | Set `agent` for legacy agent-only mode |
| `ANTHROPIC_API_KEY` | — | Required for stage 3 (LLM agent) |

### Output files
```
output/
├── warped_ihc.png                     ← canonical best warped IHC (use this)
├── color_overlay.png                  ← ACCREDIT-style red overlay on HE
├── result.json                        ← full cascade history, QCS per stage
├── summary.json                       ← backwards-compat schema for legacy tools
├── warped_ihc_fast.png                ← stage 1 output (if it ran)
├── warped_ihc_auto.png                ← stage 2 output (if it ran)
├── warped_ihc_agent_rescue.png        ← stage 3 rescue output (if it ran)
└── agent_output/                      ← raw agent artifacts (only if stage 3 ran)
    ├── overlay_*.png
    └── transform_final.h5
```

---

## Recipes

All recipes alternate LightGlue-affine refinement with a nonrigid backend:

| Recipe | Composition | Wins (12 pairs) | Time |
|---|---|---:|---:|
| `iter4`          | 4× (LG + DHR-NR) | 1 | ~15-30 s |
| `iter6`          | 6× (LG + DHR-NR) | 0 | ~25-45 s |
| `dhr3_far2`      | 3× (LG + DHR-NR) → 2× (LG + Farneback) | 0 | ~20-40 s |
| `dhr4_far2`      | 4× (LG + DHR-NR) → 2× (LG + Farneback) | 1 | ~25-45 s |
| `dhr6_far4`      | 6× (LG + DHR-NR) → 4× (LG + Farneback) | 1 | ~40-60 s |
| **`dhr3_demons300`** | **3× (LG + DHR-NR) → LG + Demons(300)** | **9** | **~25-45 s** |
| `auto`           | Run all 6; return max QCS | — | ~2-4 min |

`dhr3_demons300` is the default — wins 75% of single-recipe benchmarks.

---

## Python API

```python
from pipelines.he_ihc.run_unified_he_ihc import run_unified_he_ihc

result = run_unified_he_ihc(
    he_path='HE.svs',
    ihc_path='IHC.svs',
    output_dir='out/',
    disable_agent=False,
)
print(result['final_qcs_512'], result['final_stage'], result['warped_path'])
```

Low-level components:
```python
from pipelines.he_ihc.run_he_ihc import (
    load_svs, to_gray, register,                    # high-level
    lg_affine_step, apply_dhr_nr,                   # building blocks
    apply_farneback, apply_demons,
)

he = to_gray(load_svs('HE.svs'))
ihc = to_gray(load_svs('IHC.svs'))

# Manual composition
aligned = ihc
for _ in range(3):
    aligned = lg_affine_step(he, aligned)
    aligned = apply_dhr_nr(aligned, he)
aligned = apply_demons(aligned, he, iterations=300)
```

---

## Dependencies

```
python >= 3.10
torch >= 2.0
torchvision
lightglue              # pip install git+https://github.com/cvg/LightGlue.git
openslide-python >= 1.3
SimpleITK >= 2.3
opencv-python >= 4.7
tifffile >= 2023.3
Pillow >= 10.0
numpy >= 1.24
anthropic              # only needed for stage 3 (LLM agent)
```

GPU recommended but not required.

---

## Benchmark (January 2026)

| Method | Mean QCS@512 | Wins | Runtime |
|---|---:|---:|---:|
| **ACCREDIT (this pipeline)** | **0.886** | **12/12** | 30-60 s (common case) |
| VALIS | 0.707 | 0/12 | ~30 s |
| DHR | 0.623 | 0/12 | ~30 s GPU |
| SimpleITK baseline | 0.244 | 0/12 | ~50 s |
| VoltRon | 0.090 | 0/12 | ~15 s |
| Palom | 0.013 | 0/12 | ~15 s |

Full per-pair breakdown: `/home/lab-server/workspace/IHC-HE/qcs_accredit_FINAL.json`.

---

## Files

```
pipelines/he_ihc/
├── __init__.py
├── README.md                          ← this file (user documentation)
├── METHODS.md                         ← paper-ready methods document
├── run_he_ihc.py                      ← single-recipe + Python API
├── run_unified_he_ihc.py              ← Strategy C cascade (fast → auto → agent)
└── dhr_nonrigid.py                    ← pure-PyTorch DHR nonrigid implementation
```

Related files in the broader ACCREDIT codebase:
```
agent/
├── agent.py                           ← LLM agent (updated prompt)
└── tools.py                           ← adds run_improved_{fast,auto,rescue}
run_unified_agent.py                   ← routes --mode=he-ihc to Strategy C
```

---

## References

1. Wodzinski, M. et al. *DeeperHistReg: Robust Whole Slide Images Registration
   Framework*. arXiv:2404.14434 (2024).
2. Lindenberger, P., Sarlin, P.-E., Pollefeys, M. *LightGlue: Local Feature
   Matching at Light Speed*. ICCV 2023.
3. DeTone, D., Malisiewicz, T., Rabinovich, A. *SuperPoint: Self-Supervised
   Interest Point Detection and Description*. CVPRW 2018.
4. Gatenbee, C. D. et al. *VALIS: Virtual Alignment of pathoLogy Image Series*.
   Nature Communications 14, 4502 (2023).
5. Vercauteren, T. et al. *Symmetric Log-Domain Diffeomorphic Registration:
   A Demons-based Approach*. MICCAI 2008.
