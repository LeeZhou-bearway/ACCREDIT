"""
preprocess.py - OpenSlide multi-resolution loading + rgb2hed H-channel extraction
"""
import numpy as np
import openslide
from skimage.color import rgb2hed
import cv2


# Global image store (image_id -> numpy array)
_image_store: dict[str, np.ndarray] = {}

# Metadata store: image_id -> {mpp_x, mpp_y, downsample, level, path}
_meta_store: dict[str, dict] = {}


def load_wsi_thumbnail(path: str, level: int) -> tuple[str, np.ndarray]:
    """
    Load a WSI at a given OpenSlide level and store in memory.
    Falls back to load_tiff_downsampled for plain TIFF files.
    Returns (image_id, rgb_array).
    """
    try:
        slide = openslide.OpenSlide(path)
    except (openslide.OpenSlideUnsupportedFormatError, openslide.OpenSlideError):
        # Plain TIFF fallback
        return load_tiff_downsampled(path, target_mpp=8.0)

    n_levels = slide.level_count
    level = min(level, n_levels - 1)
    dims = slide.level_dimensions[level]
    region = slide.read_region((0, 0), level, dims)
    img = np.array(region.convert("RGB"))

    mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
    mpp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, 0.25))
    downsample = float(slide.level_downsamples[level])
    slide.close()

    image_id = f"{path}__level{level}"
    _image_store[image_id] = img
    _meta_store[image_id] = {
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
        "downsample": downsample,
        "level": level,
        "path": path,
        # Physical pixel size in mm (for SimpleITK spacing)
        "spacing_x_mm": mpp_x * downsample / 1000.0,
        "spacing_y_mm": mpp_y * downsample / 1000.0,
    }
    return image_id, img


def load_wsi_at_mpp(path: str, target_mpp: float = 8.0) -> tuple[str, np.ndarray, int]:
    """
    Load a WSI at the pyramid level whose resolution is closest to target_mpp µm/px.
    Returns (image_id, rgb_array, level_index).
    """
    slide = openslide.OpenSlide(path)
    base_mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
    best_level, best_diff = 0, float('inf')
    for lvl in range(slide.level_count):
        diff = abs(base_mpp * slide.level_downsamples[lvl] - target_mpp)
        if diff < best_diff:
            best_diff, best_level = diff, lvl
    slide.close()
    image_id, img = load_wsi_thumbnail(path, level=best_level)
    return image_id, img, best_level


def best_matched_levels(he_path: str, ihc_path: str,
                        target_mpp: float = 8.0) -> tuple[int, int]:
    """
    Find HE and IHC levels whose resolutions are closest to each other AND near target_mpp.
    Strategy: pick the IHC level closest to target_mpp, then pick the HE level
    closest to that IHC level's actual mpp. This ensures the two registration
    thumbnails have the most similar pixel spacing, even when pyramid structures differ.
    Returns (he_level, ihc_level).
    """
    def _get_mpps(path):
        slide = openslide.OpenSlide(path)
        base = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
        mpps = [base * d for d in slide.level_downsamples]
        slide.close()
        return mpps

    he_mpps  = _get_mpps(he_path)
    ihc_mpps = _get_mpps(ihc_path)

    # Pick IHC level closest to target_mpp
    ihc_level = int(min(range(len(ihc_mpps)), key=lambda i: abs(ihc_mpps[i] - target_mpp)))
    ihc_chosen_mpp = ihc_mpps[ihc_level]

    # Pick HE level closest to the chosen IHC mpp
    he_level = int(min(range(len(he_mpps)), key=lambda i: abs(he_mpps[i] - ihc_chosen_mpp)))

    return he_level, ihc_level


def get_image(image_id: str) -> np.ndarray:
    """Retrieve a stored image by its ID."""
    if image_id not in _image_store:
        raise KeyError(f"image_id '{image_id}' not found. Call load_wsi_thumbnail first.")
    return _image_store[image_id]


def store_image(image_id: str, img: np.ndarray) -> str:
    """Store an image under a given ID and return the ID."""
    _image_store[image_id] = img
    return image_id


def get_meta(image_id: str) -> dict:
    """Return metadata for an image. Walk up the ID chain (strip suffixes) to find it."""
    if image_id in _meta_store:
        return _meta_store[image_id]
    # Strip derived suffixes like __H, __gray, __warped_...
    base = image_id
    for suffix in ("__H", "__gray"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            if base in _meta_store:
                return _meta_store[base]
    return {}


def extract_hematoxylin(image_id: str) -> tuple[str, np.ndarray]:
    """
    Decompose RGB image via rgb2hed and return the H (hematoxylin) channel.
    H channel: index 0 in HED decomposition.
    Returns (new_image_id, h_channel_uint8).
    """
    img = get_image(image_id)
    # rgb2hed expects float64 in [0,1]
    img_float = img.astype(np.float64) / 255.0
    hed = rgb2hed(img_float)
    h_channel = hed[:, :, 0]

    # Percentile-based normalization: robust to outliers across resolution levels.
    p_low = float(np.percentile(h_channel, 1))
    p_high = float(np.percentile(h_channel, 99))
    if p_high - p_low < 1e-8:
        p_low, p_high = float(h_channel.min()), float(h_channel.max())
    h_norm = (h_channel - p_low) / (p_high - p_low + 1e-8)
    h_norm = np.clip(h_norm, 0.0, 1.0)
    h_uint8 = (h_norm * 255).astype(np.uint8)

    new_id = image_id + "__H"
    _image_store[new_id] = h_uint8
    return new_id, h_uint8


def to_grayscale(image_id: str) -> tuple[str, np.ndarray]:
    """
    Convert RGB image to grayscale (fallback strategy B).
    Returns (new_image_id, gray_uint8).
    """
    img = get_image(image_id)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    new_id = image_id + "__gray"
    _image_store[new_id] = gray
    return new_id, gray


def get_slide_info(path: str) -> dict:
    """Return basic slide info: level count, dimensions, mpp."""
    try:
        slide = openslide.OpenSlide(path)
        info = {
            "level_count": slide.level_count,
            "level_dimensions": slide.level_dimensions,
            "level_downsamples": slide.level_downsamples,
            "mpp_x": float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0)),
            "mpp_y": float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, 0)),
        }
        slide.close()
        return info
    except openslide.OpenSlideUnsupportedFormatError:
        return _get_tiff_info(path)


def _get_tiff_info(path: str) -> dict:
    """Fallback: read plain TIFF metadata via tifffile."""
    import tifffile
    with tifffile.TiffFile(path) as t:
        page = t.pages[0]
        h, w = page.shape[:2]
        mpp = 0.0
        # Try to read resolution: ResolutionUnit 2=inch, 3=cm
        res_unit_tag = page.tags.get('ResolutionUnit')
        xres_tag = page.tags.get('XResolution')
        if xres_tag and res_unit_tag:
            res_unit = res_unit_tag.value  # 1=no unit, 2=inch, 3=cm
            xres = xres_tag.value
            if isinstance(xres, tuple) and len(xres) == 2 and xres[1] != 0:
                ppu = xres[0] / xres[1]  # pixels per unit
                if res_unit == 2 and ppu > 1:
                    # pixels per inch → µm per pixel
                    mpp = 25400.0 / ppu
                elif res_unit == 3 and ppu > 1:
                    # pixels per cm → µm per pixel
                    mpp = 10000.0 / ppu
    return {
        "level_count": 1,
        "level_dimensions": [(w, h)],
        "level_downsamples": [1.0],
        "mpp_x": mpp,
        "mpp_y": mpp,
        "format": "plain_tiff",
        "full_width": w,
        "full_height": h,
    }


def load_tiff_downsampled(path: str, target_mpp: float = 8.0) -> tuple[str, np.ndarray]:
    """
    Load a plain TIFF file and downsample to approximately target_mpp resolution.
    For images without mpp info, downsample to keep longest edge ~4000px.
    Returns (image_id, rgb_array).
    """
    import tifffile
    info = _get_tiff_info(path)
    full_w, full_h = info["full_width"], info["full_height"]
    mpp = info["mpp_x"]

    if mpp > 0:
        downsample = target_mpp / mpp
    else:
        # No mpp info: downsample so longest edge ~4000px
        longest = max(full_w, full_h)
        downsample = max(1.0, longest / 4000.0)

    downsample = max(1.0, downsample)
    new_w = max(1, int(full_w / downsample))
    new_h = max(1, int(full_h / downsample))

    print(f"[preprocess] Loading {path}: {full_w}x{full_h} → {new_w}x{new_h} "
          f"(ds={downsample:.1f}x, mpp={mpp:.3f})")

    img_full = tifffile.imread(path)
    if img_full.ndim == 2:
        img_full = cv2.cvtColor(img_full, cv2.COLOR_GRAY2RGB)
    elif img_full.shape[2] == 4:
        img_full = img_full[:, :, :3]  # drop alpha

    img = cv2.resize(img_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
    del img_full  # free memory

    actual_mpp = mpp * downsample if mpp > 0 else 0.0
    image_id = f"{path}__ds{downsample:.0f}"
    _image_store[image_id] = img
    _meta_store[image_id] = {
        "mpp_x": mpp if mpp > 0 else 0.25,
        "mpp_y": mpp if mpp > 0 else 0.25,
        "downsample": downsample,
        "level": 0,
        "path": path,
        "spacing_x_mm": actual_mpp / 1000.0 if actual_mpp > 0 else 0.25 * downsample / 1000.0,
        "spacing_y_mm": actual_mpp / 1000.0 if actual_mpp > 0 else 0.25 * downsample / 1000.0,
    }
    return image_id, img
