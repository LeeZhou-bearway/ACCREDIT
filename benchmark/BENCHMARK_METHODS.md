# Benchmark Competing Methods

This document describes how each competing registration method was installed and applied for the DAPI→HE benchmark comparison (16 Xenium samples, QCS v2 @ 4000px).

## 1. VALIS

- **Source**: Docker Hub — `cdgatenbee/valis-wsi`
- **GitHub**: https://github.com/MathOnco/valis
- **Paper**: Gatenbee et al., *Nature Communications*, 2023
- **Installation**:
  ```bash
  docker pull cdgatenbee/valis-wsi
  ```
- **Usage**:
  1. Pre-compute full-res DAPI MIP outside Docker (Docker image lacks `imagecodecs` for JPEG2000):
     ```bash
     python precompute_mip_for_valis.py <sample_dir> <output.tif>
     ```
  2. Run VALIS Docker — alignment at 2000px working resolution, full-res warp output:
     ```bash
     docker run --rm \
         -v /data:/data \
         cdgatenbee/valis-wsi \
         python3 run_valis_fullres_warp.py \
             --dapi_mip <dapi_mip.npy> --he <he_small.png> \
             --fullres_mip <fullres_mip.npy> --output_dir <out>
     ```
- **Notes**: VALIS performs rigid + non-rigid registration internally. Output is a warped DAPI image at full resolution.

## 2. DeepHistReg (DHR)

- **Source**: Docker Hub — `marwod/deeperhistreg`
- **GitHub**: https://github.com/MWod/DeeperHistReg
- **Paper**: Wodzinski & Muller, *IEEE TMI*, 2024
- **Installation**:
  ```bash
  docker pull marwod/deeperhistreg
  ```
- **Usage**:
  ```bash
  docker run --rm --gpus all \
      -v /data:/data:ro \
      -v /output:/output \
      marwod/deeperhistreg \
          --srcp <dapi_mip.png> --trgp <he_small.png> \
          --out <output_dir> \
          --params nonrigid_gpu_params.json \
          --exp <sample_name> --sdf --temp <tmp_dir>
  ```
- **Notes**: Requires GPU (`--gpus all`). Uses non-rigid registration with GPU acceleration. Outputs a `displacement_field.mha` displacement field. A separate apply step warps the full-res DAPI through the displacement field.

## 3. PALOM

- **Source**: PyPI / GitHub
- **GitHub**: https://github.com/Yu-AnChen/palom
- **Paper**: Chen et al., associated with CyCIF / MCMICRO pipeline
- **Installation**:
  ```bash
  pip install palom
  ```
- **Usage** (Python API):
  ```python
  import palom

  reader_he = palom.reader.OmePyramidReader(he_path)
  reader_fl = palom.reader.OmePyramidReader(fluo_path)

  aligner = palom.align.Aligner(
      ref_reader=reader_he,
      moving_reader=reader_fl,
      ref_channel=0,
      moving_channel=0,
  )
  aligner.coarse_register_affine()
  aligner.compute_shifts()
  ```
- **Notes**: Affine-only registration. Outputs an affine matrix `M_palom.npy`. No non-rigid refinement.

## 4. SimpleITK (Baseline)

- **Source**: PyPI
- **GitHub**: https://github.com/SimpleITK/SimpleITK
- **Documentation**: https://simpleitk.readthedocs.io/
- **Installation**:
  ```bash
  pip install SimpleITK
  ```
- **Usage** (Python API):
  ```python
  import SimpleITK as sitk

  fixed = sitk.GetImageFromArray(he_inverted_gray)
  moving = sitk.GetImageFromArray(dapi_gray)

  reg = sitk.ImageRegistrationMethod()
  reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
  reg.SetInitialTransform(sitk.CenteredTransformInitializer(
      fixed, moving, sitk.Similarity2DTransform(),
      sitk.CenteredTransformInitializerFilter.MOMENTS))
  reg.SetInterpolator(sitk.sitkLinear)
  result_transform = reg.Execute(fixed, moving)
  ```
- **Notes**: Baseline method using Similarity2D transform with Mattes Mutual Information. Affine-only, no non-rigid refinement.

## 5. VoltRon

- **Source**: R package via GitHub
- **GitHub**: https://github.com/BIMSBbioinfo/VoltRon
- **Paper**: Manukyan et al., *Bioinformatics*, 2024
- **Installation** (R):
  ```r
  devtools::install_github("BIMSBbioinfo/VoltRon")
  ```
- **Usage** (R):
  ```r
  library(VoltRon)

  he_obj <- importImageData(he_path, sample_name = "HE",
                            channels = "HE", is.RGB = TRUE)
  fl_obj <- importImageData(dapi_path, sample_name = "DAPI",
                            channels = "DAPI")

  reg_result <- registerSpatialData(fl_obj, he_obj)
  ```
- **Notes**: R-based spatial data analysis toolkit. Uses internal registration routines. Invoked via `Rscript run_voltron_single.R <dapi> <he> <output_dir> <dataset_name>`.

## Evaluation

All methods were evaluated using the same QCS v2 formula at 4000px resolution. See `benchmark/evaluation/qcs_formula_v2.py` for the exact formula and `qcs_all_6methods_4000px.json` for the full results.
