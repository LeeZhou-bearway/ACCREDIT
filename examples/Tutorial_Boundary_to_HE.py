"""
ACCREDIT Tutorial: Cell Boundary-to-HE Registration
====================================================

This tutorial demonstrates how to align Xenium cell boundary coordinates
(parquet format) to a matched H&E whole-slide image.

This pipeline replaces the manual 10x Xenium Explorer GUI workflow with
an automated, batch-capable, QCS-gated alternative.

The pipeline:
  1. Converts parquet boundary coordinates to a binary contour mask
  2. Performs global affine alignment via feature matching
  3. Applies local block refinement
  4. Transforms all boundary polygons into H&E image space
  5. Evaluates quality using an overlap Dice coefficient (internal QC gate)

Note on metrics:
  `result['dice']` is a *filled-mask overlap* Dice (the warped cell-boundary
  mask vs the H&E tissue mask). It is the pipeline's internal QC / gating
  score and is intentionally loose, so it reads higher than a boundary-level
  metric. It is NOT the cell-boundary ("cell Dice") number reported in the
  paper (Fig. 3c), which is a stricter contour/cell-level metric computed by a
  separate evaluation (ACCREDIT vs the 10x alignment, cross-checked with
  CellViT). Do not compare `result['dice']` directly against the paper's cell
  Dice — they are different metrics.
"""
import os
import sys

ACCREDIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ACCREDIT_ROOT)


# ── Example 1: Single sample ──────────────────────────────────────────────

def example_single():
    """Register boundary parquet to HE for a single Xenium sample."""
    from run_unified_agent import run_unified

    result = run_unified(
        he_path='data/Xenium_Prime_Breast_Cancer_FFPE/he_image.ome.tif',
        moving_path='data/Xenium_Prime_Breast_Cancer_FFPE/nucleus_boundaries.parquet',
        output_dir='output/boundary_breast',
    )

    if result:
        dice = result.get('dice', 'N/A')
        # Overlap Dice = internal QC gate, not the paper's cell Dice (see module docstring).
        print(f"Overlap Dice (QC): {dice}")


# ── Example 2: With cell boundaries ───────────────────────────────────────

def example_with_cell_parquet():
    """Include optional cell boundaries for finer evaluation."""
    from run_unified_agent import run_unified

    result = run_unified(
        he_path='data/sample/he_image.ome.tif',
        moving_path='data/sample/nucleus_boundaries.parquet',
        output_dir='output/boundary_with_cells',
        extra={
            'cell_parquet_path': 'data/sample/cell_boundaries.parquet',
        },
        interactive=False,
    )

    if result:
        print(f"Overlap Dice (QC): {result.get('dice', 'N/A')}")


# ── Example 3: Batch processing ───────────────────────────────────────────

def example_batch():
    """Batch register multiple Xenium samples."""
    from run_unified_agent import run_unified

    samples = [
        'Xenium_Prime_Breast_Cancer_FFPE',
        'Xenium_Prime_Cervical_Cancer_FFPE',
        'Xenium_Prime_Human_Lung_Cancer_FFPE',
    ]

    base = 'data'
    results = {}

    for sample in samples:
        sample_dir = os.path.join(base, sample)
        he = os.path.join(sample_dir, 'he_image.ome.tif')
        parquet = os.path.join(sample_dir, 'nucleus_boundaries.parquet')

        if not os.path.exists(he) or not os.path.exists(parquet):
            print(f"Skipping {sample}: files not found")
            continue

        result = run_unified(
            he_path=he,
            moving_path=parquet,
            output_dir=f'output/batch_boundary/{sample}',
            interactive=False,
        )
        results[sample] = result

    print("\nBatch Summary:")
    for name, r in results.items():
        dice = r.get('dice', 'FAILED') if r else 'FAILED'
        print(f"  {name}: Overlap Dice (QC)={dice}")


if __name__ == '__main__':
    print("ACCREDIT Boundary-to-HE Tutorial")
    print("=" * 40)
    print("Select an example:")
    print("  1. Single sample registration")
    print("  2. With cell boundaries parquet")
    print("  3. Batch processing")

    choice = input("\nEnter choice (1-3): ").strip()
    examples = {'1': example_single, '2': example_with_cell_parquet,
                '3': example_batch}
    fn = examples.get(choice)
    if fn:
        fn()
    else:
        print("Invalid choice.")
