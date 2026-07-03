# %% [markdown]
# # Cinematic YouTube Shorts from Real Astronomical Images
#
# This notebook downloads an **authoritative public astronomical dataset** and turns it into a **vertical 1080×1920 cinematic short** suitable for YouTube Shorts (≤ 60 seconds).
#
# ## What this notebook does
#
# - Downloads either:
#   - **SDSS FITS imaging** for a famous galaxy target (default: **M51 / Whirlpool Galaxy**), or
#   - a **historic Hubble public-release image** (default alternative: **Hubble Ultra Deep Field 2004** raster).
# - Loads FITS data and preserves **WCS** when available.
# - Builds a scientifically respectful visual image using:
#   - percentile clipping,
#   - arcsinh / Lupton RGB stretching,
#   - optional mild denoising,
#   - colour grading that is kept **separate** from the scientific base composite.
# - Renders a vertical cinematic sequence with:
#   - animated zooms,
#   - pans,
#   - parallax-style layered motion,
#   - particles and overlays,
#   - text captions,
#   - fade transitions.
# - Exports:
#   - a raw vertical MP4,
#   - an optional subtitled MP4,
#   - an optional audio-muxed MP4,
#   - an `.srt` subtitle file.
#
# ## Scientific fidelity note
#
# This notebook is designed for **visual storytelling, not photometry**. It keeps the scientific source image as the foundation, while putting stylistic effects in **separate overlay layers**. The goal is to remain visually compelling **without inventing scientific structure inside the data layer**.

# %%
# If needed, install dependencies from inside the notebook.
# Recommended:
#   pip install numpy scipy pillow matplotlib astropy astroquery reproject scikit-image imageio imageio-ffmpeg tqdm
#
# Conda alternative:
#   conda install -c conda-forge numpy scipy pillow matplotlib astropy astroquery reproject scikit-image imageio imageio-ffmpeg tqdm

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
from astroquery.sdss import SDSS
import astropy.units as u
from reproject import reproject_interp
from scipy.ndimage import gaussian_filter
from skimage import exposure
from skimage.restoration import denoise_wavelet
from tqdm.auto import tqdm

plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["image.origin"] = "lower"

print("Imports loaded.")

# %% [markdown]
# ## Image representation and processing choices
#
# **FITS** is the standard astronomy file format because it stores both pixel values and metadata. In imaging workflows, the most useful metadata often includes calibration information and a **World Coordinate System** (**WCS**), which maps pixels to sky coordinates. In this notebook:
#
# - For **SDSS FITS** data, WCS is preserved and used when we crop and align bands.
# - For **public-release RGB rasters** such as the historical HUDF image, the notebook treats the file as a display image, not as a calibrated multi-band measurement product.
#
# ### Why the processing pipeline is split into two layers
#
# The notebook makes a clear distinction between:
#
# 1. **Scientific base image**
#    - band alignment,
#    - crop selection,
#    - contrast stretch,
#    - mild denoising,
#    - RGB assembly.
#
# 2. **Presentation layer**
#    - cinematic camera motion,
#    - text,
#    - parallax separation,
#    - particles,
#    - vignettes,
#    - soundtrack and subtitles.
#
# That separation helps preserve scientific integrity while still producing a short-form social video that feels polished and cinematic.
#
# ### Why Lupton RGB and arcsinh-like stretching are useful
#
# Astronomical image data usually span a wide dynamic range. A linear stretch often hides faint structure or clips bright cores. This is why the notebook uses a Lupton-style RGB synthesis for SDSS `g/r/i` data and conservative percentile clipping. This is a good compromise between readability and fidelity for outreach-style videos.

# %%
OUTPUT_ROOT = Path("astro_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
FRAME_DIR = OUTPUT_ROOT / "frames"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)
DATA_ROOT.mkdir(exist_ok=True, parents=True)
FRAME_DIR.mkdir(exist_ok=True, parents=True)
PREVIEW_DIR.mkdir(exist_ok=True, parents=True)

CONFIG = {
    # Choose from: "sdss_m51", "hubble_udf_2004"
    "dataset_key": "sdss_m51",

    # Final delivery
    "video_width": 1080,
    "video_height": 1920,
    "fps": 24,
    "duration_s": 58,  # keep below 60s to leave platform margin
    "output_basename": "astronomy_short_vertical",

    # Scientific-image options
    "cutout_size_arcmin": 7.0,
    "denoise": True,
    "denoise_sigma": 0.02,

    # Lupton / RGB synthesis for FITS mode
    "lupton_Q": 10,
    "lupton_stretch": 0.6,

    # Visual style
    "vignette_strength": 0.18,
    "saturation_boost": 1.12,
    "contrast_boost": 1.10,
    "gamma": 0.95,
    "particle_count": 180,

    # Text / captions
    "title_text": "A Famous Galaxy, Reimagined from Real Survey Data",
    "sdss_data_release": 17,
    "subtitle_text": "Real astronomical pixels • cinematic motion • vertical 1080×1920",
    "credit_prefix": "Data credit: ",

    # Optional external assets
    "audio_path": None,         # e.g. "audio/ambient_track.mp3"
    "burn_subtitles": False,    # requires ffmpeg with libass
    "write_subtitle_sidecar": True,
}

DATASETS = {
    "sdss_m51": {
        "kind": "sdss_fits",
        "title": "Whirlpool Galaxy from SDSS imaging",
        "target_name": "M51",
        "coord": SkyCoord("13h29m52.7s +47d11m43s", frame="icrs"),
        "query_radius": "2 arcmin",
        "cutout_size_arcmin": 7.0,
        "bands": ("i", "r", "g"),
        "credit": "Sloan Digital Sky Survey",
        "notes": "Uses official SDSS FITS data through astroquery.sdss and preserves WCS.",
    },
    "hubble_udf_2004": {
        "kind": "rgb_raster_url",
        "title": "Hubble Ultra Deep Field 2004 public-release image",
        "url": "https://cdn.esahubble.org/archives/images/large/heic0406a.jpg",
        "local_name": "heic0406a_hudf.jpg",
        "credit": "NASA, ESA, S. Beckwith (STScI), and the HUDF Team",
        "notes": "Historic ESA/Hubble public-release raster. Beautiful for cinematic use, but unlike FITS mode it does not carry WCS in the image file itself.",
    },
}

dataset = DATASETS[CONFIG["dataset_key"]]

print("Selected dataset:", CONFIG["dataset_key"])
print("Dataset title:", dataset["title"])

# %%
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def lerp(a, b, t):
    return a + (b - a) * t

def smoothstep(t):
    t = clamp(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

def ease_in_out_sine(t):
    t = clamp(t, 0.0, 1.0)
    return -(math.cos(math.pi * t) - 1.0) / 2.0

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def download_file(url: str, local_path: Path):
    import urllib.request

    ensure_parent(local_path)
    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"Using cached file: {local_path}")
        return local_path

    print(f"Downloading\n  {url}\n-> {local_path}")
    urllib.request.urlretrieve(url, local_path)
    return local_path

def save_image(arr: np.ndarray, path: Path):
    ensure_parent(path)
    Image.fromarray(arr).save(path)

def np_to_pil_rgba(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] == 3:
        return Image.fromarray(arr, mode="RGB").convert("RGBA")
    if arr.shape[-1] == 4:
        return Image.fromarray(arr, mode="RGBA")
    raise ValueError(f"Unsupported image shape: {arr.shape}")

def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)

def gamma_correct_uint8(arr: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if gamma == 1.0:
        return arr
    x = np.clip(arr.astype(np.float32) / 255.0, 0, 1)
    x = np.power(x, gamma)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)

def find_ffmpeg() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

def get_font(size: int = 48, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

def draw_multiline_text(
    image: Image.Image,
    lines: List[str],
    xy: Tuple[int, int],
    font_size: int = 48,
    fill=(255, 255, 255, 255),
    anchor: str = "la",
    line_spacing: int = 8,
    stroke_width: int = 2,
    stroke_fill=(0, 0, 0, 220),
    bold: bool = False,
):
    draw = ImageDraw.Draw(image)
    font = get_font(font_size, bold=bold)
    x, y = xy
    for line in lines:
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
            anchor=anchor,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        bbox = draw.textbbox((x, y), line, font=font, anchor=anchor, stroke_width=stroke_width)
        y += (bbox[3] - bbox[1]) + line_spacing
    return image

def make_vignette(width: int, height: int, strength: float = 0.2) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width / 2.0, height / 2.0
    nx = (xx - cx) / (width / 2.0)
    ny = (yy - cy) / (height / 2.0)
    rr = np.sqrt(nx**2 + ny**2)
    vignette = 1.0 - strength * np.clip(rr**1.8, 0, 1)
    vignette = np.clip(vignette, 0, 1)
    return vignette.astype(np.float32)

def apply_basic_grade(arr: np.ndarray, contrast: float, saturation: float, gamma: float) -> np.ndarray:
    img = Image.fromarray(arr)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    out = np.array(img)
    out = gamma_correct_uint8(out, gamma=gamma)
    return out

def float01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - mn) / (mx - mn), 0, 1)

def robust_single_channel(arr: np.ndarray, low=0.5, high=99.8, asinh_factor=8.0) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), copy=False)
    vmin, vmax = np.percentile(arr, [low, high])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    x = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    x = np.arcsinh(asinh_factor * x) / np.arcsinh(asinh_factor)
    return np.clip(x, 0, 1)

def best_data_hdu(hdul: fits.HDUList):
    for hdu in hdul:
        if getattr(hdu, "data", None) is not None and getattr(hdu.data, "ndim", 0) >= 2:
            return hdu
    raise ValueError("No image HDU with data found.")

def pad_or_crop_cover(img: Image.Image, out_size: Tuple[int, int], center_xy=(0.5, 0.5)) -> Image.Image:
    out_w, out_h = out_size
    in_w, in_h = img.size
    scale = max(out_w / in_w, out_h / in_h)
    new_w = max(1, int(round(in_w * scale)))
    new_h = max(1, int(round(in_h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    cx = int(round(center_xy[0] * new_w))
    cy = int(round(center_xy[1] * new_h))
    left = int(np.clip(cx - out_w // 2, 0, max(0, new_w - out_w)))
    top = int(np.clip(cy - out_h // 2, 0, max(0, new_h - out_h)))
    return resized.crop((left, top, left + out_w, top + out_h))

def transform_layer(
    img_rgba: Image.Image,
    out_size: Tuple[int, int],
    zoom: float = 1.0,
    center_xy: Tuple[float, float] = (0.5, 0.5),
    angle_deg: float = 0.0,
) -> Image.Image:
    w, h = img_rgba.size
    scaled_w = max(2, int(round(w * zoom)))
    scaled_h = max(2, int(round(h * zoom)))
    scaled = img_rgba.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
    if abs(angle_deg) > 1e-6:
        scaled = scaled.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=True)

    cw = int(round(center_xy[0] * scaled.size[0]))
    ch = int(round(center_xy[1] * scaled.size[1]))
    left = int(np.clip(cw - out_size[0] // 2, 0, max(0, scaled.size[0] - out_size[0])))
    top = int(np.clip(ch - out_size[1] // 2, 0, max(0, scaled.size[1] - out_size[1])))
    cropped = scaled.crop((left, top, left + out_size[0], top + out_size[1]))

    if cropped.size != out_size:
        canvas = Image.new("RGBA", out_size, (0, 0, 0, 255))
        canvas.alpha_composite(cropped, dest=((out_size[0] - cropped.size[0]) // 2, (out_size[1] - cropped.size[1]) // 2))
        cropped = canvas

    return cropped

def compute_focus_point(rgb: np.ndarray) -> Tuple[float, float]:
    lum = (
        0.2126 * rgb[..., 0].astype(np.float32)
        + 0.7152 * rgb[..., 1].astype(np.float32)
        + 0.0722 * rgb[..., 2].astype(np.float32)
    )
    h, w = lum.shape
    y0, y1 = int(h * 0.15), int(h * 0.85)
    x0, x1 = int(w * 0.15), int(w * 0.85)
    cropped = lum[y0:y1, x0:x1]
    threshold = np.percentile(cropped, 99.5)
    mask = np.where(cropped >= threshold, cropped, 0.0)
    if np.sum(mask) <= 0:
        return (0.5, 0.5)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    x = float(np.sum(xx * mask) / np.sum(mask)) / w
    y = float(np.sum(yy * mask) / np.sum(mask)) / h
    return (x, y)

def build_parallax_layers(rgb: np.ndarray) -> Dict[str, Image.Image]:
    rgb_f = rgb.astype(np.float32) / 255.0
    lum = (
        0.2126 * rgb_f[..., 0]
        + 0.7152 * rgb_f[..., 1]
        + 0.0722 * rgb_f[..., 2]
    )
    lum_soft = gaussian_filter(lum, sigma=10)
    detail = np.clip(lum - gaussian_filter(lum, sigma=3), 0, None)
    detail_n = float01(detail)
    bright_n = float01(lum_soft)

    bg = gaussian_filter(rgb_f, sigma=(14, 14, 0))
    bg = np.clip(bg * 0.92, 0, 1)

    mid_alpha = np.clip((bright_n - 0.08) / 0.55, 0, 1)
    fg_alpha = np.clip((detail_n - 0.08) / 0.70, 0, 1) * np.clip((bright_n - 0.15) / 0.75, 0, 1)
    star_alpha = np.clip((lum - np.percentile(lum, 99.7)) * 120, 0, 1)

    bg_rgba = np.dstack([bg, np.ones_like(lum)])
    mid_rgba = np.dstack([rgb_f, mid_alpha])
    fg_rgba = np.dstack([rgb_f, fg_alpha])
    star_rgba = np.dstack([np.ones_like(rgb_f), star_alpha])

    return {
        "background": np_to_pil_rgba(np.clip(bg_rgba * 255, 0, 255).astype(np.uint8)),
        "midground": np_to_pil_rgba(np.clip(mid_rgba * 255, 0, 255).astype(np.uint8)),
        "foreground": np_to_pil_rgba(np.clip(fg_rgba * 255, 0, 255).astype(np.uint8)),
        "stars": np_to_pil_rgba(np.clip(star_rgba * 255, 0, 255).astype(np.uint8)),
    }

def generate_particles(width: int, height: int, count: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    particles = []
    for _ in range(count):
        particles.append({
            "x": rng.uniform(0, width),
            "y": rng.uniform(0, height),
            "r": rng.uniform(0.7, 2.4),
            "a": rng.uniform(40, 130),
            "speed": rng.uniform(10, 45),
            "phase": rng.uniform(0, 2 * np.pi),
        })
    return particles

def render_particle_overlay(width: int, height: int, particles, t: float) -> Image.Image:
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for p in particles:
        x = (p["x"] + 20 * math.sin(0.3 * t + p["phase"])) % width
        y = (p["y"] - p["speed"] * t) % height
        alpha = int(np.clip(p["a"] * (0.55 + 0.45 * math.sin(1.3 * t + p["phase"])), 10, 180))
        bbox = (x - p["r"], y - p["r"], x + p["r"], y + p["r"])
        draw.ellipse(bbox, fill=(255, 255, 255, alpha))
    return overlay

print("Helper functions ready.")

# %% [markdown]
# ## Quick scientific preview
#
# The next cells preview the downloaded data before any cinematic treatment.
#
# For **SDSS FITS mode**, the notebook:
# - crops an equal sky region from each band,
# - reprojects all bands onto a shared WCS,
# - converts `i/r/g` into an outreach-style RGB image.
#
# For **HUDF raster mode**, the notebook previews the historic archival public-release image directly.

# %%
def load_sdss_dataset(ds: Dict, config: Dict):
    coord = ds["coord"]
    matches = SDSS.query_region(
        coord,
        radius=ds["query_radius"],
        photoobj_fields=["ra", "dec", "run", "rerun", "camcol", "field"],
        data_release=config.get("sdss_data_release", 17),
    )
    if matches is None or len(matches) == 0:
        raise RuntimeError("No SDSS matches found near the requested target.")

    match_coords = SkyCoord(matches["ra"], matches["dec"], unit="deg", frame="icrs")
    nearest_idx = int(np.argmin(coord.separation(match_coords).arcsec))
    nearest = matches[nearest_idx:nearest_idx + 1]

    band_images = {}
    cutout_size = config.get("cutout_size_arcmin", ds.get("cutout_size_arcmin", 7.0)) * u.arcmin

    for band in ds["bands"]:
        hdul_list = SDSS.get_images(matches=nearest, band=band, data_release=config.get("sdss_data_release", 17))
        if not hdul_list:
            raise RuntimeError(f"Could not retrieve SDSS band {band}.")
        hdu = best_data_hdu(hdul_list[0])
        data = np.asarray(hdu.data, dtype=np.float32)
        header = hdu.header
        wcs = WCS(header)

        cutout = Cutout2D(data, position=coord, size=cutout_size, wcs=wcs, mode="partial", fill_value=np.nan)
        band_images[band] = {
            "data": cutout.data.astype(np.float32),
            "wcs": cutout.wcs,
            "header": cutout.wcs.to_header(),
        }

    # Reproject all bands to r-band WCS for safety.
    ref_band = "r" if "r" in band_images else list(band_images.keys())[0]
    ref_header = band_images[ref_band]["header"]
    ref_shape = band_images[ref_band]["data"].shape

    aligned = {}
    for band, info in band_images.items():
        hdu = fits.PrimaryHDU(info["data"], header=info["header"])
        arr, footprint = reproject_interp(hdu, ref_header, shape_out=ref_shape)
        arr = np.nan_to_num(arr, nan=np.nanmedian(arr[np.isfinite(arr)]) if np.isfinite(arr).any() else 0.0)
        aligned[band] = arr.astype(np.float32)

    # Optional mild denoising on the scientific base arrays.
    if config.get("denoise", True):
        for band in list(aligned.keys()):
            arr = robust_single_channel(aligned[band], low=0.1, high=99.95, asinh_factor=6.0)
            arr = denoise_wavelet(arr, sigma=config.get("denoise_sigma", 0.02), rescale_sigma=True)
            aligned[band] = arr.astype(np.float32)
    else:
        for band in list(aligned.keys()):
            aligned[band] = robust_single_channel(aligned[band], low=0.1, high=99.95, asinh_factor=6.0)

    # Lupton RGB expects red, green, blue arrays.
    r_arr = aligned.get("i", aligned.get("r"))
    g_arr = aligned.get("r", aligned.get("g"))
    b_arr = aligned.get("g", aligned.get("r"))
    rgb = make_lupton_rgb(
        r_arr,
        g_arr,
        b_arr,
        Q=config.get("lupton_Q", 10),
        stretch=config.get("lupton_stretch", 0.6)
    )
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Small tone harmonisation for vertical video use.
    rgb = exposure.match_histograms(rgb, rgb, channel_axis=-1)  # no-op, keeps function in pipeline explicit

    return {
        "rgb": rgb,
        "wcs": WCS(ref_header),
        "bands": aligned,
        "credit": ds["credit"],
        "title": ds["title"],
        "notes": ds["notes"],
        "source_kind": ds["kind"],
        "coord": coord,
    }

def load_rgb_raster_url_dataset(ds: Dict, config: Dict):
    local_path = DATA_ROOT / ds["local_name"]
    download_file(ds["url"], local_path)
    img = Image.open(local_path).convert("RGB")
    rgb = np.array(img)
    return {
        "rgb": rgb.astype(np.uint8),
        "wcs": None,
        "bands": None,
        "credit": ds["credit"],
        "title": ds["title"],
        "notes": ds["notes"],
        "source_kind": ds["kind"],
        "coord": None,
    }

def load_dataset(ds: Dict, config: Dict):
    if ds["kind"] == "sdss_fits":
        return load_sdss_dataset(ds, config)
    if ds["kind"] == "rgb_raster_url":
        return load_rgb_raster_url_dataset(ds, config)
    raise ValueError(f"Unsupported dataset kind: {ds['kind']}")

dataset_payload = load_dataset(dataset, CONFIG)
base_rgb = dataset_payload["rgb"]
print("Loaded image shape:", base_rgb.shape)
print("Data credit:", dataset_payload["credit"])
print("Source notes:", dataset_payload["notes"])

# %%
def show_preview(payload):
    rgb = payload["rgb"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(rgb, origin="lower")
    ax.set_title(payload["title"])
    ax.set_axis_off()
    plt.show()

show_preview(dataset_payload)

# %%
if dataset_payload["bands"] is not None:
    band_order = ["g", "r", "i"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for i, band in enumerate(band_order):
        if band in dataset_payload["bands"]:
            axes[i].imshow(dataset_payload["bands"][band], cmap="gray", origin="lower")
            axes[i].set_title(f"{band}-band")
            axes[i].set_axis_off()
        else:
            axes[i].axis("off")
    axes[3].imshow(dataset_payload["rgb"], origin="lower")
    axes[3].set_title("RGB composite")
    axes[3].set_axis_off()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Aesthetic choices and scientific fidelity
#
# This notebook deliberately separates *data treatment* from *style treatment*.
#
# ### Kept close to the source data
# - band alignment and cropping,
# - WCS-aware reprojected cutouts in FITS mode,
# - conservative percentile clipping,
# - Lupton RGB synthesis for `i/r/g`,
# - only mild denoising.
#
# ### Added only in the presentation layer
# - Ken Burns motion,
# - parallax-like depth cues,
# - particles,
# - subtitles,
# - title cards,
# - vignette and mild grade.
#
# That means viewers get a cinematic experience, but the base astronomical structure still comes from real survey or archive pixels.

# %%
SHOT_PLAN = [
    {
        "name": "intro_wide",
        "start": 0.0,
        "end": 6.0,
        "zoom_start": 1.02,
        "zoom_end": 1.18,
        "center_start": (0.50, 0.52),
        "center_end": (0.50, 0.50),
        "angle_start": -0.6,
        "angle_end": -0.2,
        "caption": "Real public astronomy data",
    },
    {
        "name": "approach",
        "start": 6.0,
        "end": 22.0,
        "zoom_start": 1.18,
        "zoom_end": 1.55,
        "center_start": (0.50, 0.50),
        "center_end": (0.54, 0.48),
        "angle_start": -0.2,
        "angle_end": 0.35,
        "caption": "Percentile scaling • mild denoising • colour-preserving RGB",
    },
    {
        "name": "pan_detail",
        "start": 22.0,
        "end": 38.0,
        "zoom_start": 1.55,
        "zoom_end": 2.10,
        "center_start": (0.54, 0.48),
        "center_end": (0.48, 0.52),
        "angle_start": 0.35,
        "angle_end": -0.25,
        "caption": "Parallax and particles are overlays, not invented data",
    },
    {
        "name": "deep_focus",
        "start": 38.0,
        "end": 52.0,
        "zoom_start": 2.10,
        "zoom_end": 2.80,
        "center_start": None,  # filled in after focus point computation
        "center_end": None,    # filled in after focus point computation
        "angle_start": -0.25,
        "angle_end": 0.10,
        "caption": "WCS-respecting crop in FITS mode",
    },
    {
        "name": "outro",
        "start": 52.0,
        "end": CONFIG["duration_s"],
        "zoom_start": 2.80,
        "zoom_end": 2.35,
        "center_start": None,  # filled in after focus point computation
        "center_end": (0.50, 0.50),
        "angle_start": 0.10,
        "angle_end": 0.00,
        "caption": "Built for vertical 1080×1920 storytelling",
    },
]

focus_point = compute_focus_point(base_rgb)
SHOT_PLAN[3]["center_start"] = (0.50, 0.50)
SHOT_PLAN[3]["center_end"] = focus_point
SHOT_PLAN[4]["center_start"] = focus_point

CAPTIONS = [
    (0.6, 4.8, "Real public astronomical data"),
    (6.5, 18.0, "Loaded from official archives or survey services"),
    (18.2, 30.0, "Bands aligned • stretched • colour mapped"),
    (30.2, 43.5, "Cinematic motion added in separate presentation layers"),
    (43.8, 55.0, "Vertical framing for YouTube Shorts"),
    (55.2, CONFIG["duration_s"] - 0.3, "Credit the archive and keep the science honest"),
]

def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(captions, path: Path):
    ensure_parent(path)
    lines = []
    for idx, (start, end, text) in enumerate(captions, start=1):
        lines.append(str(idx))
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

subtitle_path = OUTPUT_ROOT / f'{CONFIG["output_basename"]}.srt'
if CONFIG.get("write_subtitle_sidecar", True):
    write_srt(CAPTIONS, subtitle_path)
    print("Subtitle sidecar written to:", subtitle_path)

# %%
OUT_SIZE = (CONFIG["video_width"], CONFIG["video_height"])
BASE_PIL = np_to_pil_rgba(base_rgb)
LAYERS = build_parallax_layers(base_rgb)
PARTICLES = generate_particles(OUT_SIZE[0], OUT_SIZE[1], CONFIG["particle_count"], seed=7)
VIGNETTE = make_vignette(OUT_SIZE[0], OUT_SIZE[1], CONFIG["vignette_strength"])

def get_shot_at_time(t: float):
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]

def interpolated_shot_state(t: float):
    shot = get_shot_at_time(t)
    local_t = (t - shot["start"]) / max(shot["end"] - shot["start"], 1e-6)
    e = ease_in_out_sine(local_t)
    center = (
        lerp(shot["center_start"][0], shot["center_end"][0], e),
        lerp(shot["center_start"][1], shot["center_end"][1], e),
    )
    zoom = lerp(shot["zoom_start"], shot["zoom_end"], e)
    angle = lerp(shot["angle_start"], shot["angle_end"], e)
    return shot, center, zoom, angle, local_t

def caption_at_time(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None

def add_cinematic_text(frame: Image.Image, t: float):
    title_alpha = int(255 * (smoothstep((t - 0.2) / 1.4) * (1 - smoothstep((t - 4.2) / 1.0))))
    sub_alpha = int(220 * (smoothstep((t - 0.8) / 1.6) * (1 - smoothstep((t - 4.8) / 1.0))))

    if title_alpha > 4:
        draw_multiline_text(
            frame,
            [CONFIG["title_text"]],
            (70, 120),
            font_size=60,
            fill=(255, 255, 255, title_alpha),
            bold=True,
            stroke_width=3,
            stroke_fill=(0, 0, 0, min(220, title_alpha)),
        )

    if sub_alpha > 4:
        draw_multiline_text(
            frame,
            [CONFIG["subtitle_text"]],
            (72, 210),
            font_size=28,
            fill=(225, 235, 255, sub_alpha),
            stroke_width=2,
            stroke_fill=(0, 0, 0, min(190, sub_alpha)),
        )

    live_caption = caption_at_time(t)
    if live_caption:
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        y0 = frame.size[1] - 230
        draw.rounded_rectangle((48, y0, frame.size[0] - 48, y0 + 110), radius=24, fill=(0, 0, 0, 110))
        frame.alpha_composite(overlay)
        draw_multiline_text(
            frame,
            [live_caption],
            (70, y0 + 24),
            font_size=34,
            fill=(255, 255, 255, 245),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 180),
        )

    outro_alpha = int(210 * smoothstep((t - (CONFIG["duration_s"] - 3.0)) / 1.2))
    if outro_alpha > 4:
        draw_multiline_text(
            frame,
            [f'{CONFIG["credit_prefix"]}{dataset_payload["credit"]}'],
            (70, frame.size[1] - 120),
            font_size=24,
            fill=(230, 230, 230, outro_alpha),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 180),
        )
    return frame

def render_frame(t: float) -> np.ndarray:
    _, center, zoom, angle, _ = interpolated_shot_state(t)

    # Slightly different motion per layer to create a parallax feel.
    bg_center = (
        clamp(center[0] - 0.015 * math.sin(0.35 * t), 0.0, 1.0),
        clamp(center[1] - 0.010 * math.cos(0.27 * t), 0.0, 1.0),
    )
    mg_center = center
    fg_center = (
        clamp(center[0] + 0.010 * math.sin(0.43 * t), 0.0, 1.0),
        clamp(center[1] + 0.012 * math.cos(0.31 * t), 0.0, 1.0),
    )

    bg = transform_layer(LAYERS["background"], OUT_SIZE, zoom=max(1.0, zoom * 0.96), center_xy=bg_center, angle_deg=angle * 0.6)
    mg = transform_layer(LAYERS["midground"], OUT_SIZE, zoom=zoom, center_xy=mg_center, angle_deg=angle)
    fg = transform_layer(LAYERS["foreground"], OUT_SIZE, zoom=zoom * 1.03, center_xy=fg_center, angle_deg=angle * 1.18)
    stars = transform_layer(LAYERS["stars"], OUT_SIZE, zoom=zoom * 1.06, center_xy=fg_center, angle_deg=angle * 1.2)

    frame = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 255))
    frame.alpha_composite(bg)
    frame.alpha_composite(mg)
    frame.alpha_composite(fg)
    # Keep stars subtle
    stars_mod = stars.copy()
    frame.alpha_composite(stars_mod)

    particles = render_particle_overlay(OUT_SIZE[0], OUT_SIZE[1], PARTICLES, t)
    frame.alpha_composite(particles)

    frame = add_cinematic_text(frame, t)

    frame_np = np.array(frame.convert("RGB"))
    frame_np = apply_basic_grade(
        frame_np,
        contrast=CONFIG["contrast_boost"],
        saturation=CONFIG["saturation_boost"],
        gamma=CONFIG["gamma"],
    )

    frame_np = np.clip(frame_np.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)

    # Fade in / fade out.
    fade_in = smoothstep(t / 1.2)
    fade_out = 1.0 - smoothstep((t - (CONFIG["duration_s"] - 1.4)) / 1.2)
    frame_np = np.clip(frame_np.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)
    return frame_np

# Preview a few frames without rendering the whole movie yet.
preview_times = [1.0, 12.0, 28.0, 46.0, CONFIG["duration_s"] - 1.0]
preview_arrays = [render_frame(t) for t in preview_times]

fig, axes = plt.subplots(1, len(preview_arrays), figsize=(20, 10))
for ax, arr, t in zip(axes, preview_arrays, preview_times):
    ax.imshow(arr)
    ax.set_title(f"{t:.0f}s")
    ax.set_axis_off()
plt.tight_layout()
plt.show()

# %%
RAW_VIDEO_PATH = OUTPUT_ROOT / f'{CONFIG["output_basename"]}_raw.mp4'
SUBBED_VIDEO_PATH = OUTPUT_ROOT / f'{CONFIG["output_basename"]}_subbed.mp4'
AUDIO_VIDEO_PATH = OUTPUT_ROOT / f'{CONFIG["output_basename"]}_with_audio.mp4'
FINAL_VIDEO_PATH = OUTPUT_ROOT / f'{CONFIG["output_basename"]}_final.mp4'

nframes = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
times = np.arange(nframes) / CONFIG["fps"]

print(f"Rendering {nframes} frames to {RAW_VIDEO_PATH} ...")
with iio.get_writer(
    RAW_VIDEO_PATH,
    fps=CONFIG["fps"],
    codec="libx264",
    quality=8,
    pixelformat="yuv420p",
    macro_block_size=None,
) as writer:
    for idx, t in enumerate(tqdm(times)):
        frame = render_frame(float(t))
        writer.append_data(frame)

print("Raw video written:", RAW_VIDEO_PATH)

# %%
def run_ffmpeg(cmd: List[str]):
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

ffmpeg = find_ffmpeg()
print("FFmpeg detected:", ffmpeg)

final_candidate = RAW_VIDEO_PATH

if CONFIG.get("burn_subtitles", False) and ffmpeg and subtitle_path.exists():
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(RAW_VIDEO_PATH),
        "-vf", f"subtitles={subtitle_path}:force_style=Fontname=DejaVu Sans,Fontsize=22,Outline=1.2,BorderStyle=3,MarginV=90",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(SUBBED_VIDEO_PATH),
    ]
    run_ffmpeg(cmd)
    final_candidate = SUBBED_VIDEO_PATH
    print("Subtitled video written:", SUBBED_VIDEO_PATH)

audio_path = CONFIG.get("audio_path")
if audio_path and ffmpeg and Path(audio_path).exists():
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(final_candidate),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(AUDIO_VIDEO_PATH),
    ]
    run_ffmpeg(cmd)
    final_candidate = AUDIO_VIDEO_PATH
    print("Audio-muxed video written:", AUDIO_VIDEO_PATH)
elif audio_path:
    print("Requested audio_path was not found. Skipping audio mux.")

# Copy the best available deliverable to a simple final filename.
if final_candidate.exists():
    shutil.copyfile(final_candidate, FINAL_VIDEO_PATH)
    print("Final deliverable:", FINAL_VIDEO_PATH)
else:
    print("No final candidate was produced.")

# %%
print("Output directory:", OUTPUT_ROOT.resolve())
for path in sorted(OUTPUT_ROOT.glob("*")):
    print("-", path.name)

