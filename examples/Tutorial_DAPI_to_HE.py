"""
ACCREDIT Tutorial: DAPI-to-HE Registration
===========================================

This tutorial demonstrates how to register a DAPI fluorescence morphology
image to a matched H&E whole-slide image using ACCREDIT's unified pipeline.

The pipeline automatically:
  1. Analyzes the input image to detect modality (fluorescence vs brightfield)
  2. Reports detection result and waits for user confirmation
  3. Runs the DAPI-to-HE registration with automatic method selection
  4. Evaluates registration quality using QCS
  5. Optionally invokes rescue agent if QCS is below threshold
"""
import os
import sys

# Add ACCREDIT root to path
ACCREDIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ACCREDIT_ROOT)


# ── Example 1: Auto-detect modality (interactive) ──────────────────────────

def example_auto_detect():
    """Run with automatic modality detection and user confirmation."""
    from run_unified_agent import run_unified

    result = run_unified(
        he_path='data/he_image.ome.tif',
        moving_path='data/morphology.ome.tif',
        output_dir='output/dapi_example',
    )

    if result:
        print(f"Registration successful!")
        print(f"  QCS: {result.get('qcs', 'N/A')}")
    else:
        print("Registration failed or was aborted by user.")


# ── Example 2: Explicit modality (non-interactive) ─────────────────────────

def example_explicit_mode():
    """Run with explicit modality, skip confirmation."""
    from run_unified_agent import run_unified

    result = run_unified(
        he_path='data/he_image.ome.tif',
        moving_path='data/morphology.ome.tif',
        output_dir='output/dapi_explicit',
        mode='dapi',
        interactive=False,
    )

    if result:
        qcs = result.get('qcs', 0)
        print(f"QCS: {qcs:.4f}")


# ── Example 3: Using sample directory ──────────────────────────────────────

def example_sample_dir():
    """Auto-detect files from a Xenium sample directory structure."""
    from run_unified_agent import run_unified

    # Xenium sample directories typically contain:
    #   he_image.ome.tif
    #   morphology.ome.tif
    #   nucleus_boundaries.parquet
    result = run_unified(
        he_path=None,
        moving_path=None,
        output_dir='output/xenium_auto',
        sample_dir='data/Xenium_Prime_Breast_Cancer_FFPE',
    )

    if result:
        print(f"QCS: {result.get('qcs', 'N/A')}")


# ── Example 4: Batch registration ─────────────────────────────────────────

def example_batch():
    """Register multiple pairs in batch mode (non-interactive)."""
    from run_unified_agent import run_unified

    pairs = [
        ('data/HE_1.tif', 'data/DAPI_1.tif'),
        ('data/HE_2.tif', 'data/DAPI_2.tif'),
        ('data/HE_3.tif', 'data/DAPI_3.tif'),
    ]

    results = {}
    for he, moving in pairs:
        name = os.path.splitext(os.path.basename(he))[0]
        out = f'output/batch/{name}'
        result = run_unified(
            he_path=he,
            moving_path=moving,
            output_dir=out,
            mode='dapi',
            interactive=False,
        )
        results[name] = result

    # Summary
    print("\nBatch Summary:")
    for name, r in results.items():
        qcs = r.get('qcs', 'FAILED') if r else 'FAILED'
        print(f"  {name}: QCS={qcs}")


if __name__ == '__main__':
    print("ACCREDIT DAPI-to-HE Tutorial")
    print("=" * 40)
    print("Select an example:")
    print("  1. Auto-detect modality (interactive)")
    print("  2. Explicit DAPI mode (non-interactive)")
    print("  3. Sample directory auto-detect")
    print("  4. Batch registration")

    choice = input("\nEnter choice (1-4): ").strip()
    examples = {'1': example_auto_detect, '2': example_explicit_mode,
                '3': example_sample_dir, '4': example_batch}
    fn = examples.get(choice)
    if fn:
        fn()
    else:
        print("Invalid choice.")
