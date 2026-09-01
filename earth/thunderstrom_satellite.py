from __future__ import annotations

"""
A Thunderstorm Growing From Satellite Data — cinematic YouTube Short renderer

Creates a vertical 1080x1920 YouTube Short from a short sequence of geostationary
infrared satellite images. The default live source is NOAA GOES-19 (GOES-East)
ABI Band 13, the 10.3 µm "clean" longwave infrared window.

The script downloads the latest full-disk JPEG sequence published by NOAA/NESDIS,
automatically searches the sequence for a region with strong cloud-field change,
and turns that crop into a cinematic storm-growth time lapse.



LIVE SOURCE
-----------
NOAA/NESDIS/STAR GOES-19 ABI Full Disk Band 13 image directory:
    https://cdn.star.nesdis.noaa.gov/GOES19/ABI/FD/13/

NOAA Band 13 information / loop page:
    https://www.goes.noaa.gov/fulldisk_band.php?band=13&length=12&sat=G19

As of 2026, GOES-19 is NOAA's operational GOES-East satellite.

LOCAL IMAGE MODE
----------------
You can bypass downloading and use your own chronological satellite-image folder:

    SATELLITE_IMAGE_DIR=/path/to/frames python a_thunderstorm_growing_from_satellite_data.py

Supported image extensions: .jpg .jpeg .png .webp .tif .tiff
Files are sorted by filename. NOAA-style timestamps in filenames are detected when
possible; otherwise frames are assigned 10-minute intervals.

MANUAL STORM CROP
-----------------
By default the script automatically selects a storm-growth region. Override it with
normalized full-image coordinates x0,y0,x1,y1 in the 0..1 range:

    STORM_CROP=0.42,0.31,0.67,0.56 python a_thunderstorm_growing_from_satellite_data.py

TUNING
------
    THUNDERSTORM_SHORT_QUICK=1     540x960, 6 fps, 12 s preview
    THUNDERSTORM_SHORT_OFFLINE=1   deterministic synthetic satellite-style fallback
    GOES_SATELLITE=G19             NOAA satellite directory token
    GOES_REGION=FD                 NOAA region token (default full disk)
    GOES_BAND=13                   ABI band token
    GOES_LOOP_IMAGES=12            number of source frames to use
    GOES_IMAGE_SIZE=1808x1808      source JPEG resolution published by NOAA

Recommended install
-------------------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm

Outputs
-------
- final vertical MP4 with generated atmospheric/thunder audio when ffmpeg exists
- silent MP4 fallback
- SRT subtitle sidecar
- preview PNG frames
- CSV of frame-by-frame growth proxy metrics
- JSON summary and source notes
- cached NOAA source imagery when live download succeeds

Primary references
------------------
- NOAA GOES-19 operational GOES-East announcement:
  https://www.nesdis.noaa.gov/news/noaas-goes-19-now-operational-goes-east-providing-critical-new-data-forecasters
- NOAA GOES-19 Band 13 loop / channel description:
  https://www.goes.noaa.gov/fulldisk_band.php?band=13&length=12&sat=G19
- NOAA/NESDIS/STAR GOES imagery archive/CDN:
  https://cdn.star.nesdis.noaa.gov/GOES19/ABI/FD/13/
"""

import io
import json
import math
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("THUNDERSTORM_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("THUNDERSTORM_SHORT_OFFLINE", "0") == "1"
LOCAL_IMAGE_DIR = os.environ.get("SATELLITE_IMAGE_DIR", "").strip()
MANUAL_CROP_TEXT = os.environ.get("STORM_CROP", "").strip()

GOES_SATELLITE = os.environ.get("GOES_SATELLITE", "G19").strip().upper() or "G19"
GOES_REGION = os.environ.get("GOES_REGION", "FD").strip().upper() or "FD"
GOES_BAND = os.environ.get("GOES_BAND", "13").strip() or "13"
GOES_LOOP_IMAGES = max(6, int(os.environ.get("GOES_LOOP_IMAGES", "12")))
GOES_IMAGE_SIZE = os.environ.get("GOES_IMAGE_SIZE", "1808x1808").strip() or "1808x1808"

OUTPUT_ROOT = Path("a_thunderstorm_growing_from_satellite_data_output")
DATA_ROOT = OUTPUT_ROOT / "data"
CACHE_ROOT = DATA_ROOT / "cache"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
SOURCE_FRAME_DIR = CACHE_ROOT / "satellite_frames"
for directory in [OUTPUT_ROOT, DATA_ROOT, CACHE_ROOT, PREVIEW_DIR, SOURCE_FRAME_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12.0 if QUICK_MODE else 58.0,
    "output_basename": "a_thunderstorm_growing_from_satellite_data",
    "title": "A THUNDERSTORM GROWING",
    "title_2": "FROM SATELLITE DATA",
    "subtitle": f"NOAA {GOES_SATELLITE} // ABI BAND {GOES_BAND} // INFRARED",
    "sat_panel_left": 24 if QUICK_MODE else 48,
    "sat_panel_top": 145 if QUICK_MODE else 290,
    "sat_panel_right": 516 if QUICK_MODE else 1032,
    "sat_panel_bottom": 785 if QUICK_MODE else 1570,
    "background_particles": 100 if QUICK_MODE else 260,
    "dust_particles": 70 if QUICK_MODE else 170,
    "contrast": 1.12,
    "saturation": 0.96,
    "vignette": 0.38,
    "soundtrack_sample_rate": 22050 if QUICK_MODE else 44100,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)

COLORS = {
 
}

FULL_SHOT_PLAN = [

]

FULL_CAPTIONS = [
]

if QUICK_MODE:
    _scale = float(CONFIG["duration_s"]) / 58.0
    SHOT_PLAN = [
        {"name": shot["name"], "start": shot["start"] * _scale, "end": shot["end"] * _scale}
        for shot in FULL_SHOT_PLAN
    ]
    CAPTIONS = [(a * _scale, b * _scale, text) for a, b, text in FULL_CAPTIONS]
else:
    SHOT_PLAN = FULL_SHOT_PLAN
    CAPTIONS = FULL_CAPTIONS

NOAA_CDN_DIR = f"https://cdn.star.nesdis.noaa.gov/GOES{GOES_SATELLITE.removeprefix('G')}/ABI/{GOES_REGION}/{GOES_BAND}/"
# NOAA also accepts /GOES19/... rather than /GOESG19/.
NOAA_CDN_DIR = f"https://cdn.star.nesdis.noaa.gov/GOES{GOES_SATELLITE.lstrip('G')}/ABI/{GOES_REGION}/{GOES_BAND}/"
NOAA_INFO_URL = (
    f"https://www.goes.noaa.gov/fulldisk_band.php?band={urllib.parse.quote(GOES_BAND)}"
    f"&length={GOES_LOOP_IMAGES}&sat={urllib.parse.quote(GOES_SATELLITE)}"
)
NOAA_GOES19_OPS_URL = "https://www.nesdis.noaa.gov/news/noaas-goes-19-now-operational-goes-east-providing-critical-new-data-forecasters"


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass
class SatelliteFrame:
    index: int
    time_utc: datetime
    image: Image.Image
    source_name: str
    source_url: str = ""
    synthetic: bool = False
    cold_proxy: float = 0.0
    cold_area_proxy: float = 0.0
    change_proxy: float = 0.0


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(value: float) -> float:
    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


def smootherstep(value: float) -> float:
    x = clamp(value)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def shot_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - float(shot["start"])) / max(float(shot["end"] - shot["start"]), 1e-9))


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000.0))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for index, (start, end, text) in enumerate(captions, start=1):
        lines.extend([str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def get_font(size: int, bold: bool = False, condensed: bool = False):
    candidates: List[str] = []
    if condensed and bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
            "DejaVuSansCondensed-Bold.ttf",
        ])
    if condensed:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
            "DejaVuSansCondensed.ttf",
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ])
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int,
    fill: Tuple[int, int, int, int],
    bold: bool = False,
    condensed: bool = False,
    anchor: str = "la",
    stroke: int = 2,
) -> None:
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold=bold, condensed=condensed),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(230, fill[3])),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill: Tuple[int, int, int, int],
    line_spacing: int,
) -> None:
    draw = ImageDraw.Draw(image)
    font = get_font(size)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 225))
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += bbox[3] - bbox[1] + line_spacing


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.72, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


def request_bytes(url: str, timeout: int = 45) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ThunderstormSatelliteShort/1.0; educational visualization)",
            "Accept": "text/html,image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_noaa_timestamp(name: str) -> Optional[datetime]:
    # NOAA filenames commonly begin YYYYJJJHHMM, e.g. 20262411610_...
    match = re.search(r"(?<!\d)(20\d{2})(\d{3})(\d{2})(\d{2})(?!\d)", name)
    if not match:
        return None
    try:
        year = int(match.group(1))
        julian_day = int(match.group(2))
        hour = int(match.group(3))
        minute = int(match.group(4))
        return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
            days=julian_day - 1, hours=hour, minutes=minute
        )
    except Exception:
        return None


def normalized_manual_crop() -> Optional[Tuple[float, float, float, float]]:
    if not MANUAL_CROP_TEXT:
        return None
    try:
        values = [float(item.strip()) for item in MANUAL_CROP_TEXT.split(",")]
        if len(values) != 4:
            raise ValueError
        x0, y0, x1, y1 = values
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError
        return x0, y0, x1, y1
    except Exception as exc:
        raise ValueError("STORM_CROP must be normalized x0,y0,x1,y1 values in the 0..1 range") from exc


# -----------------------------------------------------------------------------
# Source loading
# -----------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def load_local_frames(directory: Path) -> List[SatelliteFrame]:
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if len(paths) < 2:
        raise RuntimeError(f"Need at least two images in {directory}")
    if len(paths) > GOES_LOOP_IMAGES:
        paths = paths[-GOES_LOOP_IMAGES:]
    base_time = datetime.now(timezone.utc) - timedelta(minutes=10 * (len(paths) - 1))
    frames: List[SatelliteFrame] = []
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        timestamp = parse_noaa_timestamp(path.name) or base_time + timedelta(minutes=10 * index)
        frames.append(
            SatelliteFrame(
                index=index,
                time_utc=timestamp,
                image=image,
                source_name=path.name,
                source_url=str(path.resolve()),
                synthetic=False,
            )
        )
    return frames


def discover_noaa_images() -> List[Tuple[datetime, str, str]]:
    html = request_bytes(NOAA_CDN_DIR, timeout=45).decode("utf-8", errors="ignore")
    # Directory indexes expose timestamped files like:
    # 20262271300_GOES19-ABI-FD-13-1808x1808.jpg
    sat_token = f"GOES{GOES_SATELLITE.lstrip('G')}"
    pattern = re.compile(
        rf'href=["\']([^"\']*?(20\d{{2}}\d{{3}}\d{{4}})_{re.escape(sat_token)}-ABI-{re.escape(GOES_REGION)}-'
        rf'{re.escape(GOES_BAND)}-{re.escape(GOES_IMAGE_SIZE)}\.jpg)["\']',
        re.IGNORECASE,
    )
    found: Dict[str, Tuple[datetime, str, str]] = {}
    for href, stamp in pattern.findall(html):
        time_utc = parse_noaa_timestamp(stamp)
        if time_utc is None:
            continue
        absolute = urllib.parse.urljoin(NOAA_CDN_DIR, href)
        found[absolute] = (time_utc, absolute, Path(urllib.parse.urlparse(absolute).path).name)
    if not found:
        # Broader fallback for indexes with altered markup.
        filename_pattern = re.compile(
            rf'(20\d{{9}}_{re.escape(sat_token)}-ABI-{re.escape(GOES_REGION)}-'
            rf'{re.escape(GOES_BAND)}-{re.escape(GOES_IMAGE_SIZE)}\.jpg)',
            re.IGNORECASE,
        )
        for filename in filename_pattern.findall(html):
            time_utc = parse_noaa_timestamp(filename)
            if time_utc is not None:
                absolute = urllib.parse.urljoin(NOAA_CDN_DIR, filename)
                found[absolute] = (time_utc, absolute, filename)
    return sorted(found.values(), key=lambda item: item[0])[-GOES_LOOP_IMAGES:]


def fetch_noaa_frames() -> List[SatelliteFrame]:
    discovered = discover_noaa_images()
    if len(discovered) < 2:
        raise RuntimeError("NOAA CDN index did not expose enough timestamped Band 13 images")
    frames: List[SatelliteFrame] = []
    for index, (time_utc, url, filename) in enumerate(discovered):
        cache_path = SOURCE_FRAME_DIR / filename
        if not cache_path.exists() or cache_path.stat().st_size < 20_000:
            cache_path.write_bytes(request_bytes(url, timeout=60))
        image = Image.open(cache_path).convert("RGB")
        frames.append(
            SatelliteFrame(
                index=index,
                time_utc=time_utc,
                image=image,
                source_name=filename,
                source_url=url,
                synthetic=False,
            )
        )
    return frames


def fractal_noise(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    base = np.zeros((height, width), dtype=np.float32)
    for scale, amplitude in [(16, 0.55), (32, 0.28), (64, 0.17)]:
        gh = max(2, height // scale)
        gw = max(2, width // scale)
        grid = (rng.random((gh, gw)) * 255).astype(np.uint8)
        layer = Image.fromarray(grid, "L").resize((width, height), Image.Resampling.BICUBIC)
        base += np.asarray(layer, dtype=np.float32) / 255.0 * amplitude
    base -= float(base.min())
    denom = max(float(base.max()), 1e-6)
    return base / denom


def make_synthetic_frames(count: int) -> List[SatelliteFrame]:
    # Deterministic satellite-style sequence useful for layout/testing only.
    count = max(6, count)
    size = 900 if not QUICK_MODE else 540
    yy, xx = np.mgrid[0:size, 0:size]
    rng = np.random.default_rng(7319)
    background_texture = fractal_noise(size, size, rng)
    start = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=10 * (count - 1))
    frames: List[SatelliteFrame] = []

    bg_clouds = [
        (0.22, 0.28, 0.10, 0.05, 0.27),
        (0.73, 0.22, 0.15, 0.07, 0.22),
        (0.16, 0.72, 0.13, 0.08, 0.18),
        (0.78, 0.76, 0.12, 0.06, 0.23),
        (0.48, 0.12, 0.20, 0.035, 0.16),
    ]

    for index in range(count):
        p = index / max(count - 1, 1)
        image = np.zeros((size, size, 3), dtype=np.float32)
        image[..., 0] = 11 + background_texture * 12
        image[..., 1] = 18 + background_texture * 15
        image[..., 2] = 30 + background_texture * 20

        # Static / drifting background cloud bands.
        cloud_field = np.zeros((size, size), dtype=np.float32)
        for order, (cx, cy, sx, sy, strength) in enumerate(bg_clouds):
            dcx = cx + 0.035 * p * math.sin(order * 1.7 + 0.4)
            dcy = cy + 0.015 * p * math.cos(order * 1.2)
            g = np.exp(-0.5 * (((xx / size - dcx) / sx) ** 2 + ((yy / size - dcy) / sy) ** 2))
            cloud_field += g * strength

        # Developing thunderstorm: core grows vertically/cools, anvil expands.
        cx = 0.54 + 0.015 * math.sin(p * math.pi)
        cy = 0.52 - 0.018 * p
        core_sigma = lerp(0.028, 0.075, smoothstep(p))
        core = np.exp(-0.5 * (((xx / size - cx) / core_sigma) ** 2 + ((yy / size - cy) / (core_sigma * 0.82)) ** 2))
        tower = np.exp(-0.5 * (((xx / size - (cx - 0.035 * p)) / (0.050 + 0.025 * p)) ** 2 + ((yy / size - (cy + 0.020)) / (0.038 + 0.030 * p)) ** 2))
        anvil_sigma_x = lerp(0.055, 0.190, smootherstep(max(0.0, (p - 0.22) / 0.78)))
        anvil_sigma_y = lerp(0.030, 0.082, smootherstep(max(0.0, (p - 0.22) / 0.78)))
        anvil = np.exp(-0.5 * (((xx / size - (cx + 0.035 * p)) / anvil_sigma_x) ** 2 + ((yy / size - (cy - 0.025 * p)) / anvil_sigma_y) ** 2))
        overshoot = np.exp(-0.5 * (((xx / size - (cx - 0.015)) / 0.018) ** 2 + ((yy / size - (cy - 0.010)) / 0.016) ** 2))
        storm = np.clip(core * (0.45 + 0.70 * p) + tower * (0.25 + 0.52 * p) + anvil * (0.10 + 0.70 * p) + overshoot * max(0.0, (p - 0.6)) * 0.95, 0, 1.65)

        # Infrared-like enhancement: brighter/cooler storm tops and slight cyan cast.
        combined = np.clip(cloud_field + storm, 0.0, 1.0)
        image[..., 0] += combined * 178
        image[..., 1] += combined * 194
        image[..., 2] += combined * 218
        cold = np.clip((storm - 0.52) / 0.48, 0.0, 1.0)
        image[..., 0] -= cold * 28
        image[..., 1] += cold * 18
        image[..., 2] += cold * 35

        # Fine cold-top texture.
        texture = fractal_noise(size, size, np.random.default_rng(9000 + index))
        image += (texture[..., None] - 0.5) * combined[..., None] * 26
        arr = np.clip(image, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(0.45 if QUICK_MODE else 0.65))
        frames.append(
            SatelliteFrame(
                index=index,
                time_utc=start + timedelta(minutes=10 * index),
                image=pil,
                source_name=f"synthetic_goes19_band13_{index:02d}.png",
                synthetic=True,
            )
        )
    return frames


def load_frames() -> Tuple[List[SatelliteFrame], str, List[str]]:
    notes: List[str] = []
    if LOCAL_IMAGE_DIR:
        path = Path(LOCAL_IMAGE_DIR).expanduser()
        frames = load_local_frames(path)
        notes.append(f"Loaded {len(frames)} chronological images from SATELLITE_IMAGE_DIR")
        return frames, "local_satellite_image_sequence", notes

    if not OFFLINE_MODE:
        try:
            frames = fetch_noaa_frames()
            notes.append(f"Downloaded {len(frames)} NOAA {GOES_SATELLITE} ABI Band {GOES_BAND} frames")
            return frames, "noaa_nesdis_star_goes_jpeg", notes
        except Exception as exc:
            notes.append(f"Live NOAA imagery unavailable: {exc}")

    notes.append("Using deterministic synthetic satellite-style frames for preview/layout testing")
    return make_synthetic_frames(GOES_LOOP_IMAGES), "synthetic_satellite_style_fixture", notes


# -----------------------------------------------------------------------------
# Storm crop detection and metrics
# -----------------------------------------------------------------------------

def image_luminance(image: Image.Image, size: int = 360) -> np.ndarray:
    gray = image.convert("L")
    gray.thumbnail((size, size), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    x = (size - gray.width) // 2
    y = (size - gray.height) // 2
    canvas.paste(gray, (x, y))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def smooth_array(array: np.ndarray, radius: float) -> np.ndarray:
    im = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), "L")
    return np.asarray(im.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


def detect_growth_crop(frames: Sequence[SatelliteFrame]) -> Tuple[Tuple[float, float, float, float], Dict[str, Any]]:
    manual = normalized_manual_crop()
    if manual is not None:
        return manual, {"method": "manual_normalized_crop", "crop": list(manual)}

    stacks = np.stack([image_luminance(frame.image, 360) for frame in frames], axis=0)
    first = np.mean(stacks[: max(1, len(stacks) // 4)], axis=0)
    last = np.mean(stacks[-max(1, len(stacks) // 4) :], axis=0)
    temporal_std = np.std(stacks, axis=0)
    positive_change = np.maximum(last - first, 0.0)
    absolute_change = np.abs(last - first)

    # Cold-looking/high cloud in NOAA enhanced IR is often bright, but because this
    # script also supports arbitrary rendered IR palettes we do not treat brightness
    # as calibrated temperature. We combine change + final brightness + variability.
    final_high = np.maximum(last - np.percentile(last, 64), 0.0)
    score = 1.35 * positive_change + 0.65 * absolute_change + 0.75 * temporal_std + 0.35 * final_high
    score = smooth_array(score / max(float(score.max()), 1e-6), 12.0)

    h, w = score.shape
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx - w / 2.0) / (w / 2.0)
    ny = (yy - h / 2.0) / (h / 2.0)
    earth_core = (nx * nx + ny * ny) <= 0.72**2
    # Exclude near-space/limb pixels and edge artifacts.
    score = np.where(earth_core, score, 0.0)
    cy, cx = np.unravel_index(int(np.argmax(score)), score.shape)

    # Slightly generous square crop around the growth center.
    half = 0.155
    cxn = cx / max(w - 1, 1)
    cyn = cy / max(h - 1, 1)
    x0 = clamp(cxn - half, 0.0, 1.0)
    x1 = clamp(cxn + half, 0.0, 1.0)
    y0 = clamp(cyn - half, 0.0, 1.0)
    y1 = clamp(cyn + half, 0.0, 1.0)

    # Keep a stable crop size only when clipping against an image edge actually
    # shortened the requested window. A small tolerance avoids floating-point
    # roundoff turning an already-correct crop inside out.
    target_span = half * 2.0
    if x1 - x0 < target_span * 0.98:
        if x0 <= 1e-6:
            x1 = min(1.0, target_span)
        elif x1 >= 1.0 - 1e-6:
            x0 = max(0.0, 1.0 - target_span)
    if y1 - y0 < target_span * 0.98:
        if y0 <= 1e-6:
            y1 = min(1.0, target_span)
        elif y1 >= 1.0 - 1e-6:
            y0 = max(0.0, 1.0 - target_span)

    return (x0, y0, x1, y1), {
        "method": "automatic_change_plus_cold_cloud_proxy",
        "analysis_center_xy": [float(cxn), float(cyn)],
        "peak_score": float(np.max(score)),
        "crop": [float(x0), float(y0), float(x1), float(y1)],
    }


def crop_image(image: Image.Image, crop: Tuple[float, float, float, float]) -> Image.Image:
    x0, y0, x1, y1 = crop
    w, h = image.size
    box = (
        int(round(x0 * w)),
        int(round(y0 * h)),
        max(int(round(x0 * w)) + 2, int(round(x1 * w))),
        max(int(round(y0 * h)) + 2, int(round(y1 * h))),
    )
    return image.crop(box)


def compute_frame_metrics(frames: Sequence[SatelliteFrame], crop: Tuple[float, float, float, float]) -> Dict[str, Any]:
    crops: List[np.ndarray] = []
    for frame in frames:
        patch = crop_image(frame.image, crop).convert("L").resize((256, 256), Image.Resampling.BILINEAR)
        crops.append(np.asarray(patch, dtype=np.float32) / 255.0)
    stack = np.stack(crops, axis=0)
    global_threshold = float(np.percentile(stack, 82.0))
    first = stack[0]
    rows: List[Dict[str, Any]] = []
    cold_values: List[float] = []
    area_values: List[float] = []
    change_values: List[float] = []

    for index, (frame, arr) in enumerate(zip(frames, stack)):
        top = arr[arr >= np.percentile(arr, 92.0)]
        cold_proxy = float(np.mean(top)) if top.size else float(np.mean(arr))
        cold_area = float(np.mean(arr >= global_threshold))
        change = float(np.mean(np.abs(arr - first)))
        frame.cold_proxy = cold_proxy
        frame.cold_area_proxy = cold_area
        frame.change_proxy = change
        cold_values.append(cold_proxy)
        area_values.append(cold_area)
        change_values.append(change)
        rows.append(
            {
                "frame_index": index,
                "time_utc": frame.time_utc.isoformat(),
                "source_name": frame.source_name,
                "cold_top_proxy": cold_proxy,
                "cold_cloud_area_proxy": cold_area,
                "change_from_first_proxy": change,
                "synthetic": frame.synthetic,
            }
        )

    def norm(values: Sequence[float]) -> List[float]:
        arr = np.asarray(values, dtype=float)
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi - lo < 1e-9:
            return [0.5 for _ in arr]
        return [float((v - lo) / (hi - lo)) for v in arr]

    return {
        "rows": rows,
        "cold_norm": norm(cold_values),
        "area_norm": norm(area_values),
        "change_norm": norm(change_values),
        "global_brightness_threshold": global_threshold,
        "represented_minutes": max((frames[-1].time_utc - frames[0].time_utc).total_seconds() / 60.0, 0.0),
    }


# -----------------------------------------------------------------------------
# Scene rendering
# -----------------------------------------------------------------------------

class ThunderstormScene:
    def __init__(
        self,
        frames: List[SatelliteFrame],
        crop: Tuple[float, float, float, float],
        metrics: Dict[str, Any],
        source: str,
    ):
        self.frames = frames
        self.crop = crop
        self.metrics = metrics
        self.source = source
        self.synthetic = all(frame.synthetic for frame in frames)
        self.panel_box = (
            int(CONFIG["sat_panel_left"]),
            int(CONFIG["sat_panel_top"]),
            int(CONFIG["sat_panel_right"]),
            int(CONFIG["sat_panel_bottom"]),
        )
        self.panel_w = self.panel_box[2] - self.panel_box[0]
        self.panel_h = self.panel_box[3] - self.panel_box[1]
        self.crops = [self._prepare_crop(frame.image) for frame in frames]
        self.crops_gray = [np.asarray(im.convert("L"), dtype=np.float32) / 255.0 for im in self.crops]
        self.particles = self._make_particles(int(CONFIG["background_particles"]), seed=92)
        self.dust = self._make_particles(int(CONFIG["dust_particles"]), seed=143)
        self.start_time = frames[0].time_utc
        self.end_time = frames[-1].time_utc
        self.represented_minutes = max((self.end_time - self.start_time).total_seconds() / 60.0, 10.0)

    @staticmethod
    def _make_particles(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.35, 1.6 if QUICK_MODE else 2.3)),
                "a": float(rng.uniform(8, 54)),
                "phase": float(rng.uniform(0, math.tau)),
                "speed": float(rng.uniform(1.0, 8.0)),
            }
            for _ in range(count)
        ]

    def _prepare_crop(self, image: Image.Image) -> Image.Image:
        patch = crop_image(image, self.crop).convert("RGB")
        # Fill the cinematic panel while preserving aspect ratio.
        scale = max(self.panel_w / patch.width, self.panel_h / patch.height)
        target = (max(2, int(round(patch.width * scale))), max(2, int(round(patch.height * scale))))
        patch = patch.resize(target, Image.Resampling.LANCZOS)
        left = max(0, (patch.width - self.panel_w) // 2)
        top = max(0, (patch.height - self.panel_h) // 2)
        patch = patch.crop((left, top, left + self.panel_w, top + self.panel_h))
        patch = ImageEnhance.Contrast(patch).enhance(1.10)
        patch = ImageEnhance.Sharpness(patch).enhance(1.12)
        return patch

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
        top = np.array(COLORS["navy"], dtype=float)
        bottom = np.array(COLORS["black"], dtype=float)
        for y in range(OUT_H):
            u = y / max(OUT_H - 1, 1)
            rgb = (top * (1.0 - u) + bottom * u).astype(np.uint8)
            arr[y, :, :3] = rgb
            arr[y, :, 3] = 255
        image = Image.fromarray(arr, "RGBA")
        draw = ImageDraw.Draw(image)
        for particle in self.particles:
            x = (particle["x"] + math.sin(t * 0.12 + particle["phase"]) * 8.0) % OUT_W
            y = (particle["y"] + t * particle["speed"] * 0.08) % OUT_H
            alpha = int(particle["a"] * (0.45 + 0.55 * math.sin(t * 0.6 + particle["phase"]) ** 2))
            r = particle["r"]
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(173, 215, 245, alpha))
        return image

    def draw_panel_frame(self, image: Image.Image) -> None:
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        x0, y0, x1, y1 = self.panel_box
        radius = 15 if QUICK_MODE else 30
        draw.rounded_rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), radius=radius, fill=(3, 8, 18, 235), outline=COLORS["grid"] + (90,), width=1)
        image.alpha_composite(overlay)

    def paste_satellite(self, image: Image.Image, index_float: float, opacity: float = 1.0, zoom: float = 1.0) -> int:
        n = len(self.crops)
        position = clamp(index_float, 0.0, 1.0) * (n - 1)
        i0 = int(math.floor(position))
        i1 = min(i0 + 1, n - 1)
        frac = position - i0
        if i0 == i1 or frac < 0.02:
            patch = self.crops[i0].copy()
        else:
            patch = Image.blend(self.crops[i0], self.crops[i1], frac)

        if zoom > 1.0001:
            nw = int(round(self.panel_w * zoom))
            nh = int(round(self.panel_h * zoom))
            patch = patch.resize((nw, nh), Image.Resampling.BICUBIC)
            left = max(0, (nw - self.panel_w) // 2)
            top = max(0, (nh - self.panel_h) // 2)
            patch = patch.crop((left, top, left + self.panel_w, top + self.panel_h))

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        rgba = patch.convert("RGBA")
        if opacity < 0.999:
            alpha = rgba.getchannel("A").point(lambda v: int(v * clamp(opacity)))
            rgba.putalpha(alpha)
        layer.alpha_composite(rgba, dest=(self.panel_box[0], self.panel_box[1]))
        image.alpha_composite(layer)
        return min(i0 if frac < 0.5 else i1, n - 1)

    def draw_scan_grid(self, image: Image.Image, alpha: int = 34) -> None:
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        x0, y0, x1, y1 = self.panel_box
        for frac in [0.25, 0.5, 0.75]:
            x = int(lerp(x0, x1, frac))
            y = int(lerp(y0, y1, frac))
            draw.line((x, y0, x, y1), fill=COLORS["grid"] + (alpha,), width=1)
            draw.line((x0, y, x1, y), fill=COLORS["grid"] + (alpha,), width=1)
        draw.line((x0, (y0 + y1) // 2, x1, (y0 + y1) // 2), fill=COLORS["cyan"] + (alpha // 2,), width=1)
        image.alpha_composite(overlay)

    def draw_target_box(self, image: Image.Image, t: float, strength: float = 1.0) -> None:
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        x0, y0, x1, y1 = self.panel_box
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        size = min(self.panel_w, self.panel_h) * (0.31 + 0.015 * math.sin(t * 1.8))
        left, top, right, bottom = cx - size, cy - size, cx + size, cy + size
        color = COLORS["cyan"] + (int(175 * clamp(strength)),)
        length = size * 0.25
        width = 1 if QUICK_MODE else 2
        for a, b, c, d in [
            (left, top, left + length, top), (left, top, left, top + length),
            (right, top, right - length, top), (right, top, right, top + length),
            (left, bottom, left + length, bottom), (left, bottom, left, bottom - length),
            (right, bottom, right - length, bottom), (right, bottom, right, bottom - length),
        ]:
            draw.line((a, b, c, d), fill=color, width=width)
        image.alpha_composite(overlay)

    def draw_contours(self, image: Image.Image, frame_index: int, alpha: int = 150, expansion: float = 1.0) -> None:
        arr = self.crops_gray[frame_index]
        thresholds = [np.percentile(arr, 76), np.percentile(arr, 86), np.percentile(arr, 94)]
        overlay = Image.new("RGBA", (self.panel_w, self.panel_h), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        colors = [COLORS["blue"], COLORS["cyan"], COLORS["white"]]
        for threshold, color in zip(thresholds, colors):
            mask = Image.fromarray((arr >= threshold).astype(np.uint8) * 255, "L")
            mask = mask.filter(ImageFilter.GaussianBlur(2 if QUICK_MODE else 4))
            m = np.asarray(mask, dtype=np.uint8)
            # Edge approximation via neighbor differences.
            edge = np.zeros_like(m)
            edge[1:, :] = np.maximum(edge[1:, :], np.abs(m[1:, :].astype(int) - m[:-1, :].astype(int)).astype(np.uint8))
            edge[:, 1:] = np.maximum(edge[:, 1:], np.abs(m[:, 1:].astype(int) - m[:, :-1].astype(int)).astype(np.uint8))
            edge_img = Image.fromarray(edge, "L").point(lambda v: int(alpha * expansion) if v > 28 else 0)
            color_layer = Image.new("RGBA", (self.panel_w, self.panel_h), color + (0,))
            color_layer.putalpha(edge_img)
            overlay.alpha_composite(color_layer)
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        layer.alpha_composite(overlay, dest=(self.panel_box[0], self.panel_box[1]))
        image.alpha_composite(layer)

    def draw_time_hud(self, image: Image.Image, frame_index: int, label: str = "SATELLITE FRAME") -> None:
        frame = self.frames[frame_index]
        x0, y0, x1, y1 = self.panel_box
        pad = 12 if QUICK_MODE else 24
        box_h = 58 if QUICK_MODE else 116
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            (x0 + pad, y1 - box_h - pad, x1 - pad, y1 - pad),
            radius=10 if QUICK_MODE else 20,
            fill=(2, 7, 15, 182),
            outline=COLORS["grid"] + (68,),
            width=1,
        )
        image.alpha_composite(overlay)
        draw_text(
            image,
            label,
            (x0 + pad * 2, y1 - box_h + (3 if QUICK_MODE else 7)),
            size=9 if QUICK_MODE else 18,
            fill=COLORS["cyan"] + (225,),
            bold=True,
            condensed=True,
            stroke=1,
        )
        draw_text(
            image,
            frame.time_utc.strftime("%d %b %Y  %H:%M UTC").upper(),
            (x0 + pad * 2, y1 - (26 if QUICK_MODE else 50)),
            size=11 if QUICK_MODE else 22,
            fill=COLORS["white"] + (245,),
            bold=True,
            condensed=True,
            stroke=1,
        )

    def draw_metric_hud(self, image: Image.Image, frame_index: int, mode: str) -> None:
        cold = float(self.metrics["cold_norm"][frame_index])
        area = float(self.metrics["area_norm"][frame_index])
        change = float(self.metrics["change_norm"][frame_index])
        left = 34 if QUICK_MODE else 68
        top = 800 if QUICK_MODE else 1605
        width = OUT_W - 2 * left
        height = 74 if QUICK_MODE else 148
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            (left, top, left + width, top + height),
            radius=14 if QUICK_MODE else 28,
            fill=(2, 7, 15, 188),
            outline=COLORS["grid"] + (65,),
            width=1,
        )
        image.alpha_composite(overlay)

        if mode == "cold":
            label = "IR COLD-TOP PROXY"
            value = cold
            color = COLORS["cyan"]
        elif mode == "area":
            label = "COLD CLOUD AREA PROXY"
            value = area
            color = COLORS["violet"]
        else:
            label = "CHANGE FROM FIRST FRAME"
            value = change
            color = COLORS["gold"]

        draw_text(
            image,
            label,
            (left + (14 if QUICK_MODE else 28), top + (12 if QUICK_MODE else 23)),
            size=9 if QUICK_MODE else 18,
            fill=color + (235,),
            bold=True,
            condensed=True,
            stroke=1,
        )
        bar_left = left + (14 if QUICK_MODE else 28)
        bar_right = left + width - (14 if QUICK_MODE else 28)
        bar_y = top + (48 if QUICK_MODE else 96)
        bar_h = 7 if QUICK_MODE else 14
        ImageDraw.Draw(image).rounded_rectangle((bar_left, bar_y, bar_right, bar_y + bar_h), radius=bar_h // 2, fill=(45, 63, 81, 220))
        fill_right = int(lerp(bar_left, bar_right, value))
        if fill_right > bar_left:
            ImageDraw.Draw(image).rounded_rectangle((bar_left, bar_y, fill_right, bar_y + bar_h), radius=bar_h // 2, fill=color + (245,))

    def draw_acquisition(self, image: Image.Image, t: float, shot: Dict[str, Any]) -> None:
        progress = smoothstep(shot_progress(t, shot))
        self.draw_panel_frame(image)
        idx = self.paste_satellite(image, 0.0, opacity=0.35 + 0.65 * progress, zoom=1.04 - 0.02 * progress)
        self.draw_scan_grid(image, alpha=int(20 + 35 * progress))
        self.draw_target_box(image, t, strength=progress)
        self.draw_time_hud(image, idx, label="FIRST FRAME")
        draw_text(
            image,
            f"{len(self.frames)} IR FRAMES",
            (OUT_W // 2, int(OUT_H * 0.82)),
            size=17 if QUICK_MODE else 34,
            fill=COLORS["white"] + (245,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            f"{self.represented_minutes:.0f} MINUTES OF WEATHER",
            (OUT_W // 2, int(OUT_H * 0.855)),
            size=10 if QUICK_MODE else 20,
            fill=COLORS["cyan"] + (225,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_growth_loop(self, image: Image.Image, t: float, shot: Dict[str, Any]) -> None:
        progress = smootherstep(shot_progress(t, shot))
        self.draw_panel_frame(image)
        idx = self.paste_satellite(image, progress, opacity=1.0, zoom=1.015 + 0.035 * progress)
        self.draw_scan_grid(image, alpha=26)
        self.draw_target_box(image, t, strength=0.85)
        self.draw_time_hud(image, idx, label="TRACKED REGION")
        self.draw_metric_hud(image, idx, "change")

        # Brief white flash when rapid change is strongest.
        change = float(self.metrics["change_norm"][idx])
        if change > 0.74:
            pulse = ((math.sin(t * 7.0) + 1.0) * 0.5) ** 7
            if pulse > 0.18:
                overlay = Image.new("RGBA", OUT_SIZE, (225, 245, 255, int(18 * pulse * change)))
                image.alpha_composite(overlay)

    def draw_cold_tops(self, image: Image.Image, t: float, shot: Dict[str, Any]) -> None:
        progress = smoothstep(shot_progress(t, shot))
        pos = 0.38 + 0.62 * progress
        self.draw_panel_frame(image)
        idx = self.paste_satellite(image, pos, opacity=1.0, zoom=1.07 + 0.08 * progress)
        self.draw_contours(image, idx, alpha=int(85 + 75 * progress))
        self.draw_time_hud(image, idx, label="10.3 µm INFRARED")
        self.draw_metric_hud(image, idx, "cold")
        draw_text(
            image,
            "COLDER TOPS → HIGHER CLOUDS",
            (OUT_W // 2, int(OUT_H * 0.79)),
            size=14 if QUICK_MODE else 28,
            fill=COLORS["cyan"] + (240,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_anvil(self, image: Image.Image, t: float, shot: Dict[str, Any]) -> None:
        progress = smootherstep(shot_progress(t, shot))
        pos = 0.58 + 0.42 * progress
        self.draw_panel_frame(image)
        idx = self.paste_satellite(image, pos, opacity=1.0, zoom=1.10 - 0.035 * progress)
        self.draw_contours(image, idx, alpha=125, expansion=0.9)
        self.draw_time_hud(image, idx, label="EXPANDING CLOUD SHIELD")
        self.draw_metric_hud(image, idx, "area")
        # Outward arrows around the center suggest anvil spread without implying wind vectors.
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cx = (self.panel_box[0] + self.panel_box[2]) / 2
        cy = (self.panel_box[1] + self.panel_box[3]) / 2
        r0 = min(self.panel_w, self.panel_h) * 0.12
        r1 = min(self.panel_w, self.panel_h) * (0.22 + 0.04 * progress)
        for angle in np.linspace(0, math.tau, 8, endpoint=False):
            x0 = cx + math.cos(angle) * r0
            y0 = cy + math.sin(angle) * r0
            x1 = cx + math.cos(angle) * r1
            y1 = cy + math.sin(angle) * r1
            draw.line((x0, y0, x1, y1), fill=COLORS["violet"] + (85,), width=1 if QUICK_MODE else 2)
        image.alpha_composite(overlay)

    def draw_rapid_timelapse(self, image: Image.Image, t: float, shot: Dict[str, Any]) -> None:
        progress = shot_progress(t, shot)
        # Accelerated sequence with slight hold on final frame.
        local = clamp(progress / 0.86)
        pos = smootherstep(local)
        self.draw_panel_frame(image)
        idx = self.paste_satellite(image, pos, opacity=1.0, zoom=1.06)
        self.draw_time_hud(image, idx, label="TIME COMPRESSED")
        self.draw_contours(image, idx, alpha=72)
        draw_text(
            image,
            f"{self.represented_minutes:.0f} MINUTES → {float(shot['end'] - shot['start']):.0f} SECONDS",
            (OUT_W // 2, int(OUT_H * 0.82)),
            size=17 if QUICK_MODE else 34,
            fill=COLORS["white"] + (250,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            "WATCH THE CLOUD SHIELD EXPAND",
            (OUT_W // 2, int(OUT_H * 0.86)),
            size=10 if QUICK_MODE else 20,
            fill=COLORS["gold"] + (235,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_before_after(self, image: Image.Image, t: float, shot: Dict[str, Any]) -> None:
        progress = smoothstep(shot_progress(t, shot))
        # Full panel split: first frame left, final frame right.
        self.draw_panel_frame(image)
        left_crop = self.crops[0]
        right_crop = self.crops[-1]
        split = self.panel_w // 2
        composite = Image.new("RGB", (self.panel_w, self.panel_h), (0, 0, 0))
        composite.paste(left_crop.crop((0, 0, split, self.panel_h)), (0, 0))
        composite.paste(right_crop.crop((split, 0, self.panel_w, self.panel_h)), (split, 0))
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        layer.alpha_composite(composite.convert("RGBA"), dest=(self.panel_box[0], self.panel_box[1]))
        image.alpha_composite(layer)
        draw = ImageDraw.Draw(image)
        x = self.panel_box[0] + split
        draw.line((x, self.panel_box[1], x, self.panel_box[3]), fill=COLORS["white"] + (220,), width=1 if QUICK_MODE else 3)
        draw_text(
            image,
            "BEFORE",
            (self.panel_box[0] + split // 2, self.panel_box[1] + (24 if QUICK_MODE else 48)),
            size=12 if QUICK_MODE else 24,
            fill=COLORS["white"] + (235,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            "AFTER",
            (self.panel_box[0] + split + split // 2, self.panel_box[1] + (24 if QUICK_MODE else 48)),
            size=12 if QUICK_MODE else 24,
            fill=COLORS["cyan"] + (245,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            "THE STORM BUILT UPWARD — THEN SPREAD OUT",
            (OUT_W // 2, int(OUT_H * 0.82)),
            size=13 if QUICK_MODE else 26,
            fill=COLORS["white"] + (int(245 * progress),),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_top_titles(self, image: Image.Image, t: float, shot_name: str) -> None:
        intro = smoothstep(t / (1.6 if not QUICK_MODE else 0.7))
        alpha = int(255 * intro)
        draw_text(
            image,
            str(CONFIG["title"]),
            (OUT_W // 2, 42 if QUICK_MODE else 84),
            size=22 if QUICK_MODE else 44,
            fill=COLORS["white"] + (alpha,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=2,
        )
        draw_text(
            image,
            str(CONFIG["title_2"]),
            (OUT_W // 2, 74 if QUICK_MODE else 148),
            size=18 if QUICK_MODE else 36,
            fill=COLORS["cyan"] + (alpha,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=2,
        )
        label_map = {
            "acquisition": "ACQUIRE",
            "growth_loop": "STORM GROWTH",
            "cold_tops": "CLOUD-TOP COOLING",
            "anvil": "ANVIL EXPANSION",
            "rapid_timelapse": "TIME LAPSE",
            "before_after": "BEFORE / AFTER",
        }
        draw_text(
            image,
            label_map.get(shot_name, shot_name.upper()),
            (OUT_W // 2, 105 if QUICK_MODE else 210),
            size=9 if QUICK_MODE else 18,
            fill=COLORS["muted"] + (215,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_source_hud(self, image: Image.Image) -> None:
        source_text = (
            "SYNTHETIC SATELLITE-STYLE PREVIEW"
            if self.synthetic
            else f"NOAA {GOES_SATELLITE} // ABI BAND {GOES_BAND} // {GOES_IMAGE_SIZE} JPG"
        )
        draw_text(
            image,
            source_text,
            (OUT_W // 2, OUT_H - (42 if QUICK_MODE else 84)),
            size=8 if QUICK_MODE else 16,
            fill=(COLORS["gold"] if self.synthetic else COLORS["muted"]) + (215,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )
        if not self.synthetic:
            draw_text(
                image,
                "INFORMATIONAL VISUALIZATION — NOT FOR OPERATIONAL WEATHER DECISIONS",
                (OUT_W // 2, OUT_H - (25 if QUICK_MODE else 50)),
                size=6 if QUICK_MODE else 12,
                fill=COLORS["muted"] + (155,),
                bold=True,
                condensed=True,
                anchor="ma",
                stroke=1,
            )

    def draw_caption(self, image: Image.Image, t: float) -> None:
        text = caption_at(t)
        if not text:
            return
        box_left = 34 if QUICK_MODE else 68
        box_right = OUT_W - box_left
        box_top = int(OUT_H * 0.885)
        box_bottom = int(OUT_H * 0.957)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            (box_left, box_top, box_right, box_bottom),
            radius=10 if QUICK_MODE else 20,
            fill=(1, 5, 12, 205),
            outline=COLORS["grid"] + (55,),
            width=1,
        )
        image.alpha_composite(overlay)
        draw_wrapped_text(
            image,
            text,
            (box_left + (12 if QUICK_MODE else 24), box_top + (10 if QUICK_MODE else 19)),
            max_width=(box_right - box_left) - (24 if QUICK_MODE else 48),
            size=8 if QUICK_MODE else 16,
            fill=COLORS["white"] + (238,),
            line_spacing=2 if QUICK_MODE else 4,
        )

    def draw_film_texture(self, image: Image.Image, t: float) -> None:
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for particle in self.dust:
            pulse = 0.5 + 0.5 * math.sin(t * 1.3 + particle["phase"])
            if pulse < 0.62:
                continue
            x = (particle["x"] + t * particle["speed"] * 0.38) % OUT_W
            y = (particle["y"] + math.sin(t * 0.6 + particle["phase"]) * 4.0) % OUT_H
            length = (5 if QUICK_MODE else 10) + particle["r"] * 4
            draw.line((x, y, x + length, y), fill=COLORS["cyan"] + (int(12 * pulse),), width=1)
        offset = int((t * 37) % 8)
        for y in range(offset, OUT_H, 8):
            draw.line((0, y, OUT_W, y), fill=(119, 165, 203, 7), width=1)
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        name = str(shot["name"])
        image = self.background(t)

        if name == "acquisition":
            self.draw_acquisition(image, t, shot)
        elif name == "growth_loop":
            self.draw_growth_loop(image, t, shot)
        elif name == "cold_tops":
            self.draw_cold_tops(image, t, shot)
        elif name == "anvil":
            self.draw_anvil(image, t, shot)
        elif name == "rapid_timelapse":
            self.draw_rapid_timelapse(image, t, shot)
        else:
            self.draw_before_after(image, t, shot)

        self.draw_top_titles(image, t, name)
        self.draw_source_hud(image)
        self.draw_caption(image, t)
        self.draw_film_texture(image, t)

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        arr *= VIGNETTE[..., None]
        # Tiny deterministic grain for film texture.
        rng = np.random.default_rng(int(t * 1000) + 177)
        grain = rng.normal(0.0, 2.2 if QUICK_MODE else 3.0, arr.shape[:2]).astype(np.float32)
        arr += grain[..., None]
        return np.clip(arr, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Data products
# -----------------------------------------------------------------------------

def save_data_products(
    frames: Sequence[SatelliteFrame],
    crop: Tuple[float, float, float, float],
    crop_info: Dict[str, Any],
    metrics: Dict[str, Any],
    source: str,
    notes: Sequence[str],
) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "satellite_storm_growth_metrics.csv"
    summary_path = DATA_ROOT / "satellite_storm_growth_summary.json"
    pd.DataFrame(metrics["rows"]).to_csv(csv_path, index=False)

    summary = {
        "title": CONFIG["title"] + " " + CONFIG["title_2"],
        "data_source": source,
        "satellite": GOES_SATELLITE,
        "instrument": "Advanced Baseline Imager (ABI)",
        "band": GOES_BAND,
        "band_description": "10.3 µm clean longwave infrared window when GOES_BAND=13",
        "frame_count": len(frames),
        "first_frame_utc": frames[0].time_utc.isoformat(),
        "last_frame_utc": frames[-1].time_utc.isoformat(),
        "represented_minutes": metrics["represented_minutes"],
        "crop_normalized": list(crop),
        "crop_selection": crop_info,
        "all_frames_synthetic": all(frame.synthetic for frame in frames),
        "important_caveat": (
            "Cold-top and cloud-area values are visual proxies computed from rendered imagery. "
            "They are not calibrated brightness temperatures, cloud-top heights, or severe-weather diagnostics."
        ),
        "source_urls": {
            "noaa_cdn_directory": NOAA_CDN_DIR,
            "noaa_band_info": NOAA_INFO_URL,
            "noaa_goes19_operational": NOAA_GOES19_OPS_URL,
        },
        "notes": list(notes),
        "frames": [
            {
                "index": frame.index,
                "time_utc": frame.time_utc.isoformat(),
                "source_name": frame.source_name,
                "source_url": frame.source_url,
                "synthetic": frame.synthetic,
                "cold_top_proxy": frame.cold_proxy,
                "cold_cloud_area_proxy": frame.cold_area_proxy,
                "change_from_first_proxy": frame.change_proxy,
            }
            for frame in frames
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return csv_path, summary_path


# -----------------------------------------------------------------------------
# Soundtrack and video rendering
# -----------------------------------------------------------------------------

def gaussian_envelope(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((times - center) / max(width, 1e-6)) ** 2)


def generate_ambient_soundtrack(path: Path, metrics: Dict[str, Any]) -> Path:
    sample_rate = int(CONFIG["soundtrack_sample_rate"])
    duration = float(CONFIG["duration_s"])
    count = int(round(sample_rate * duration))
    times = np.arange(count, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(61913)

    audio = np.zeros(count, dtype=np.float64)
    # Low atmosphere / distant thunder bed.
    audio += 0.085 * np.sin(math.tau * 27.0 * times + 0.25 * np.sin(math.tau * 0.055 * times))
    audio += 0.050 * np.sin(math.tau * 41.0 * times + 1.7)
    audio += 0.018 * np.sin(math.tau * 67.0 * times + 0.4 * np.sin(math.tau * 0.10 * times))

    controls = rng.normal(0.0, 1.0, max(10, int(duration * 5)))
    slow_noise = np.interp(times, np.linspace(0.0, duration, len(controls)), controls)
    audio += 0.023 * slow_noise

    # Pulses through the growth sequence.
    growth = next(shot for shot in SHOT_PLAN if shot["name"] == "growth_loop")
    changes = np.asarray(metrics["change_norm"], dtype=float)
    for index, strength in enumerate(changes):
        fraction = index / max(len(changes) - 1, 1)
        center = lerp(float(growth["start"]), float(growth["end"]), fraction)
        env = gaussian_envelope(times, center, 0.09 if QUICK_MODE else 0.15)
        audio += env * (0.022 + 0.040 * strength) * np.sin(math.tau * (110 + 80 * strength) * times)

    # Three cinematic thunder swells.
    for center, strength in [
        (float(SHOT_PLAN[2]["start"]) + 1.2 * (duration / 58.0), 0.18),
        (float(SHOT_PLAN[3]["start"]) + 1.8 * (duration / 58.0), 0.22),
        (float(SHOT_PLAN[4]["start"]) + 2.0 * (duration / 58.0), 0.28),
    ]:
        env = gaussian_envelope(times, center, 0.35 if QUICK_MODE else 0.65)
        rumble = np.sin(math.tau * 34.0 * times) + 0.5 * np.sin(math.tau * 51.0 * times + 0.6)
        audio += strength * env * rumble

    intro_x = np.clip(times / max(1.4, duration * 0.08), 0.0, 1.0)
    outro_x = np.clip((times - (duration - 1.2)) / 1.0, 0.0, 1.0)
    intro = intro_x * intro_x * (3.0 - 2.0 * intro_x)
    outro = 1.0 - outro_x * outro_x * (3.0 - 2.0 * outro_x)
    audio *= intro * outro

    peak = max(float(np.max(np.abs(audio))), 1e-9)
    audio = np.clip(audio / peak * 0.86, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


def find_ffmpeg() -> Optional[str]:
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return shutil.which("ffmpeg")


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    command = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def render_video(scene: ThunderstormScene, metrics: Dict[str, Any]) -> Path:
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    raw_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_silent.mp4"
    final_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    audio_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_ambient.wav"
    frame_count = int(round(float(CONFIG["duration_s"]) * int(CONFIG["fps"])))
    times = np.arange(frame_count) / int(CONFIG["fps"])

    print("Subtitle sidecar:", srt_path.resolve())
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(
        raw_video,
        fps=int(CONFIG["fps"]),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times, desc="Rendering thunderstorm satellite short"):
            writer.append_data(scene.render_frame(float(t)))

    generate_ambient_soundtrack(audio_path, metrics)
    if mux_audio(raw_video, audio_path, final_video):
        print("Final video with audio:", final_video.resolve())
        return final_video
    shutil.copyfile(raw_video, final_video)
    print("ffmpeg audio mux unavailable; copied silent video to:", final_video.resolve())
    return final_video


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("Title:", CONFIG["title"], CONFIG["title_2"])
    print("Preferred live source:", NOAA_CDN_DIR)
    print("Loading satellite sequence ...")
    frames, source, notes = load_frames()
    print("Source:", source)
    print("Frames:", len(frames))
    print("Frame window:", frames[0].time_utc.isoformat(), "to", frames[-1].time_utc.isoformat())

    print("Selecting storm-growth crop ...")
    crop, crop_info = detect_growth_crop(frames)
    print("Crop:", crop)
    print("Crop method:", crop_info.get("method"))

    metrics = compute_frame_metrics(frames, crop)
    csv_path, summary_path = save_data_products(frames, crop, crop_info, metrics, source, notes)
    print("Metrics CSV:", csv_path.resolve())
    print("Summary JSON:", summary_path.resolve())
    for note in notes:
        print("Data note:", note)

    scene = ThunderstormScene(frames, crop, metrics, source)
    preview_times = [
        1.0,
        min(10.0, float(CONFIG["duration_s"]) * 0.20),
        min(24.0, float(CONFIG["duration_s"]) * 0.43),
        min(34.0, float(CONFIG["duration_s"]) * 0.60),
        min(44.0, float(CONFIG["duration_s"]) * 0.78),
        float(CONFIG["duration_s"]) - 0.7,
    ]
    for preview_time in tqdm(preview_times, desc="Preview frames"):
        frame = scene.render_frame(float(preview_time))
        Image.fromarray(frame).save(PREVIEW_DIR / f"preview_{int(preview_time):02d}s.png")

    final = render_video(scene, metrics)
    print("Final:", final.resolve())
    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-", path.name)


if __name__ == "__main__":
    main()
