"""
ACCREDIT Tutorial: IHC-to-HE Registration
==========================================

This tutorial demonstrates how to register an IHC-stained adjacent section
to a matched H&E whole-slide image.

ACCREDIT automatically detects IHC staining through:
  1. Filename keywords (P40, Ki67, CD3, CD8, etc.)
  2. Image content analysis (DAB brown chromogen detection via HED deconvolution)

The HE-IHC pipeline uses:
  - LightGlue feature matching for global alignment
  - Iterative nonrigid refinement for local deformation correction
  - Demons registration for final polish
"""
import os
import sys

ACCREDIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ACCREDIT_ROOT)


# ── Example 1: IHC auto-detected from filename ────────────────────────────

def example_filename_detection():
    """P40 in filename triggers automatic IHC detection."""
    from run_unified_agent import run_unified

    result = run_unified(
        he_path='data/PASC_03_01.svs',
        moving_path='data/PASC_03_01_P40.svs',
        output_dir='output/ihc_p40',
    )

    if result:
        print(f"QCS: {result.get('qcs', 'N/A')}")


# ── Example 2: IHC detected from image content ───────────────────────────

def example_content_detection():
    """Even without IHC keywords in filename, DAB staining is detected."""
    from run_unified_agent import run_unified

    # ACCREDIT loads a thumbnail, applies HED deconvolution,
    # and detects DAB-dominant staining → routes to he-ihc pipeline
    result = run_unified(
        he_path='data/sample_HE.svs',
        moving_path='data/sample_stained.svs',
        output_dir='output/ihc_content',
    )

    if result:
        print(f"QCS: {result.get('qcs', 'N/A')}")


# ── Example 3: Direct pipeline call (advanced) ───────────────────────────

def example_direct_pipeline():
    """Call the HE-IHC pipeline directly with specific parameters."""
    sys.path.insert(0, os.path.join(ACCREDIT_ROOT, 'pipelines/he_ihc'))
    from run_unified_he_ihc import run_unified_he_ihc

    result = run_unified_he_ihc(
        he_path='data/PASC_03_01.svs',
        ihc_path='data/PASC_03_01_P40.svs',
        output_dir='output/ihc_direct',
        enable_agent=False,
    )

    if result:
        print(f"Best stage: {result.get('stage')}")
        print(f"QCS: {result.get('qcs_512', 'N/A')}")


if __name__ == '__main__':
    print("ACCREDIT IHC-to-HE Tutorial")
    print("=" * 40)
    print("Select an example:")
    print("  1. IHC auto-detected from filename (P40)")
    print("  2. IHC detected from image content (DAB)")
    print("  3. Direct pipeline call (advanced)")

    choice = input("\nEnter choice (1-3): ").strip()
    examples = {'1': example_filename_detection,
                '2': example_content_detection,
                '3': example_direct_pipeline}
    fn = examples.get(choice)
    if fn:
        fn()
    else:
        print("Invalid choice.")
