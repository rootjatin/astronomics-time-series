from __future__ import annotations

"""
The Eye of a Hurricane Forms in Real Data — cinematic YouTube Short renderer

A vertical 1080x1920 YouTube Short built around Hurricane Milton on 7 October
2024. The default live path downloads NOAA GOES-16 ABI Band 13 (10.3 µm clean
longwave infrared) Cloud and Moisture Imagery files from NOAA's public AWS S3
archive, crops each frame around Milton's NHC center position, and derives a
simple eye/eyewall thermal-contrast metric from the brightness-temperature data.

The story is intentionally narrow: watch a small ragged eye become a sharply
resolved warm eye surrounded by very cold eyewall cloud tops while NHC wind and
pressure observations intensify.

REAL-DATA PATH
--------------
The script tries to download historical GOES-16 ABI-L2-CMIPF Band 13 files for
several UTC times on 2024-10-07. It uses the geostationary projection metadata
inside each NetCDF/HDF5 file to crop around the NHC storm center.

The displayed IR values are brightness temperatures from the CMI variable.
The eye metric is descriptive, not an official NOAA/NHC product:
- find the warmest candidate inside ~40 km of the NHC center
- compare median temperature in a small eye disk to cold-cloud temperature in a
  surrounding eyewall annulus
- report the resulting warm-eye / cold-eyewall contrast in kelvin

If the NOAA archive cannot be reached, h5py/pyproj are missing, or files cannot
be decoded, the script falls back to a deterministic satellite-style IR
reconstruction driven by the real NHC wind/pressure timeline. The fallback is
prominently labeled SYNTHETIC IR RECONSTRUCTION on screen and in metadata.

NHC EYE-FORMATION ANCHORS
-------------------------
NHC Discussion 8 (0900 UTC 7 Oct) reported a central dense overcast with cloud
tops colder than -80 C and a small ragged eye.
NHC Discussion 10 (1500 UTC) said satellite images showed a small eye within
very cold central cloud cover and that the eye was becoming better defined;
Mexican radar showed a small closed eye and intense eyewall.
NHC Discussion 11 (2100 UTC) said the small eye had become even more distinct.


Recommended install
-------------------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm h5py pyproj

Outputs
-------
- final vertical MP4 with generated atmospheric audio when ffmpeg is available
- silent MP4 fallback
- SRT subtitle sidecar
- preview PNG frames
- CSV containing the NHC timeline and derived eye metrics
- JSON summary/source notes
- cached NOAA GOES-16 Band 13 NetCDF files when live retrieval succeeds

"""

import json
import math
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
import wave
import xml.etree.ElementTree as ET
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
    import h5py
except Exception:
    h5py = None

try:
    import pyproj
except Exception:
    pyproj = None

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("EYE_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("EYE_SHORT_OFFLINE", "0") == "1"
NO_AUDIO = os.environ.get("EYE_SHORT_NO_AUDIO", "0") == "1"
LOCAL_IMAGE_DIR = os.environ.get("EYE_SATELLITE_DIR", "").strip()
MAX_LIVE_FRAMES = max(3, int(os.environ.get("EYE_MAX_FRAMES", "4" if QUICK_MODE else "7")))

OUTPUT_ROOT = Path("the_eye_of_a_hurricane_forms_in_real_data_output")
DATA_ROOT = OUTPUT_ROOT / "data"
CACHE_ROOT = DATA_ROOT / "goes16_cache"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in [OUTPUT_ROOT, DATA_ROOT, CACHE_ROOT, PREVIEW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12.0 if QUICK_MODE else 58.0,
    "output_basename": "the_eye_of_a_hurricane_forms_in_real_data",
    "title": "THE EYE OF A HURRICANE",
    "title_2": "FORMS IN REAL DATA",
    "subtitle": "HURRICANE MILTON // 07 OCT 2024 // GOES-16 + NHC",
    "panel_left": 20 if QUICK_MODE else 40,
    "panel_top": 132 if QUICK_MODE else 264,
    "panel_right": 520 if QUICK_MODE else 1040,
    "panel_bottom": 690 if QUICK_MODE else 1380,
    "soundtrack_sample_rate": 22050 if QUICK_MODE else 44100,
    "crop_radius_px": 130 if QUICK_MODE else 210,
    "grain_strength": 4.0,
    "vignette": 0.38,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)



if QUICK_MODE:
    scale = float(CONFIG["duration_s"]) / 58.0
    SHOT_PLAN = [
        {"name": s["name"], "start": s["start"] * scale, "end": s["end"] * scale}
        for s in FULL_SHOT_PLAN
    ]
    CAPTIONS = [(a * scale, b * scale, text) for a, b, text in FULL_CAPTIONS]
else:
    SHOT_PLAN = FULL_SHOT_PLAN
    CAPTIONS = FULL_CAPTIONS


# -----------------------------------------------------------------------------
# Real NHC timeline
# -----------------------------------------------------------------------------

UTC = timezone.utc

@dataclass(frozen=True)
class Fix:
    time_utc: datetime
    lat: float
    lon: float
    wind_mph: float
    pressure_mb: float
    note: str

    @property
    def category(self) -> int:
        return saffir_simpson_category(self.wind_mph)


FIXES: List[Fix] = [
    Fix(datetime(2024, 10, 7, 3, 0, tzinfo=UTC), 22.4, -93.1, 90, 977, "NHC Advisory 7"),
    Fix(datetime(2024, 10, 7, 9, 0, tzinfo=UTC), 22.1, -92.6, 100, 972, "NHC Discussion 8: small ragged eye"),
    Fix(datetime(2024, 10, 7, 12, 0, tzinfo=UTC), 21.8, -92.2, 125, 945, "NHC Special Advisory 9"),
    Fix(datetime(2024, 10, 7, 15, 0, tzinfo=UTC), 21.7, -91.7, 155, 933, "NHC Discussion 10: eye becoming better defined"),
    Fix(datetime(2024, 10, 7, 15, 55, tzinfo=UTC), 21.7, -91.6, 160, 925, "NHC Category 5 update"),
    Fix(datetime(2024, 10, 7, 18, 0, tzinfo=UTC), 21.7, -91.2, 175, 914, "Interpolated between NHC fixes"),
    Fix(datetime(2024, 10, 7, 21, 0, tzinfo=UTC), 21.8, -90.8, 180, 905, "NHC Discussion 11: eye even more distinct"),
]

SATELLITE_TARGET_TIMES = [
    datetime(2024, 10, 7, 9, 0, tzinfo=UTC),
    datetime(2024, 10, 7, 10, 30, tzinfo=UTC),
    datetime(2024, 10, 7, 12, 0, tzinfo=UTC),
    datetime(2024, 10, 7, 13, 30, tzinfo=UTC),
    datetime(2024, 10, 7, 15, 0, tzinfo=UTC),
    datetime(2024, 10, 7, 18, 0, tzinfo=UTC),
    datetime(2024, 10, 7, 21, 0, tzinfo=UTC),
]

SOURCE_URLS = {
    "nhc_archive": "https://www.nhc.noaa.gov/archive/2024/MILTON.shtml",
    "nhc_discussion_8": "https://www.nhc.noaa.gov/archive/2024/al14/al142024.discus.008.shtml",
    "nhc_discussion_10": "https://www.nhc.noaa.gov/archive/2024/al14/al142024.discus.010.shtml",
    "nhc_discussion_11": "https://www.nhc.noaa.gov/archive/2024/al14/al142024.discus.011.shtml",
    "goes16_s3": "https://noaa-goes16.s3.amazonaws.com/",
    "noaa_milton_feature": "https://www.nesdis.noaa.gov/news/hurricane-milton-eyes-florida",
}


# -----------------------------------------------------------------------------
# Helpers
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


def saffir_simpson_category(wind_mph: float) -> int:
    if wind_mph < 74:
        return 0
    if wind_mph < 96:
        return 1
    if wind_mph < 111:
        return 2
    if wind_mph < 130:
        return 3
    if wind_mph < 157:
        return 4
    return 5


def category_label(wind_mph: float) -> str:
    cat = saffir_simpson_category(wind_mph)
    return "TROPICAL STORM" if cat == 0 else f"CATEGORY {cat}"


def category_color(cat: int) -> Tuple[int, int, int]:
    return {
        0: COLORS["cyan"],
        1: COLORS["green"],
        2: COLORS["yellow"],
        3: COLORS["orange"],
        4: COLORS["red"],
        5: COLORS["magenta"],
    }.get(cat, COLORS["white"])


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if float(shot["start"]) <= t < float(shot["end"]):
            return shot
    return SHOT_PLAN[-1]


def shot_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - float(shot["start"])) / max(float(shot["end"]) - float(shot["start"]), 1e-9))


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
        candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf", "DejaVuSansCondensed-Bold.ttf"]
    if condensed:
        candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf", "DejaVuSansCondensed.ttf"]
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
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
):
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold=bold, condensed=condensed),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(fill[3], 230)),
    )


def draw_wrapped(image: Image.Image, text: str, xy: Tuple[int, int], max_width: int, size: int, fill: Tuple[int, int, int, int], spacing: int = 5):
    draw = ImageDraw.Draw(image)
    font = get_font(size)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=1)
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
        draw.text((x, y), line, font=font, fill=fill, stroke_width=1, stroke_fill=(0, 0, 0, 220))
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=1)
        y += bbox[3] - bbox[1] + spacing


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


def interpolate_fix(time_utc: datetime) -> Fix:
    if time_utc <= FIXES[0].time_utc:
        return FIXES[0]
    if time_utc >= FIXES[-1].time_utc:
        return FIXES[-1]
    for a, b in zip(FIXES[:-1], FIXES[1:]):
        if a.time_utc <= time_utc <= b.time_utc:
            span = max((b.time_utc - a.time_utc).total_seconds(), 1.0)
            u = (time_utc - a.time_utc).total_seconds() / span
            return Fix(
                time_utc=time_utc,
                lat=lerp(a.lat, b.lat, u),
                lon=lerp(a.lon, b.lon, u),
                wind_mph=lerp(a.wind_mph, b.wind_mph, u),
                pressure_mb=lerp(a.pressure_mb, b.pressure_mb, u),
                note="Interpolated NHC timeline",
            )
    return FIXES[-1]


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.75, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# NOAA GOES-16 historical Band 13 loader
# -----------------------------------------------------------------------------

@dataclass
class SatelliteFrame:
    time_utc: datetime
    image: Image.Image
    fix: Fix
    source: str
    eye_temp_k: float
    eyewall_temp_k: float
    contrast_k: float
    eye_diameter_km: float
    raw_path: Optional[str] = None


def request_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; HurricaneEyeShort/1.0; educational visualization)",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def goes_prefix_for_time(t: datetime) -> str:
    doy = int(t.strftime("%j"))
    return f"ABI-L2-CMIPF/{t.year}/{doy:03d}/{t.hour:02d}/"


def parse_scan_start_from_key(key: str) -> Optional[datetime]:
    match = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", key)
    if not match:
        return None
    year, doy, hh, mm, ss = map(int, match.groups())
    return datetime.strptime(f"{year}-{doy:03d} {hh:02d}:{mm:02d}:{ss:02d}", "%Y-%j %H:%M:%S").replace(tzinfo=UTC)


def list_goes_band13_keys(t: datetime) -> List[str]:
    prefix = goes_prefix_for_time(t)
    query = urllib.parse.urlencode({"list-type": "2", "prefix": prefix})
    payload = request_bytes(f"https://noaa-goes16.s3.amazonaws.com/?{query}", timeout=45)
    root = ET.fromstring(payload)
    keys: List[str] = []
    for element in root.iter():
        if element.tag.endswith("Key") and element.text:
            key = element.text
            if "CMIPF" in key and "C13" in key and key.endswith(".nc"):
                keys.append(key)
    return keys


def choose_nearest_key(t: datetime) -> Optional[Tuple[str, datetime]]:
    keys = list_goes_band13_keys(t)
    candidates: List[Tuple[float, str, datetime]] = []
    for key in keys:
        scan = parse_scan_start_from_key(key)
        if scan is None:
            continue
        candidates.append((abs((scan - t).total_seconds()), key, scan))
    if not candidates:
        return None
    _, key, scan = min(candidates, key=lambda item: item[0])
    return key, scan


def download_goes_file(key: str) -> Path:
    target = CACHE_ROOT / Path(key).name
    if target.exists() and target.stat().st_size > 100_000:
        return target
    url = "https://noaa-goes16.s3.amazonaws.com/" + urllib.parse.quote(key, safe="/_.-")
    target.write_bytes(request_bytes(url, timeout=120))
    return target


def read_scaled_dataset(handle, name: str) -> np.ndarray:
    ds = handle[name]
    data = np.asarray(ds[...])
    fill = ds.attrs.get("_FillValue")
    data = data.astype(np.float64)
    if fill is not None:
        data[data == float(np.asarray(fill).reshape(-1)[0])] = np.nan
    scale = float(np.asarray(ds.attrs.get("scale_factor", 1.0)).reshape(-1)[0])
    offset = float(np.asarray(ds.attrs.get("add_offset", 0.0)).reshape(-1)[0])
    return data * scale + offset


def extract_goes_crop(path: Path, fix: Fix) -> Tuple[np.ndarray, float, float, float, float]:
    if h5py is None or pyproj is None:
        raise RuntimeError("Live GOES decoding requires h5py and pyproj")
    with h5py.File(path, "r") as handle:
        cmi = handle["CMI"]
        x = read_scaled_dataset(handle, "x")
        y = read_scaled_dataset(handle, "y")
        proj_ds = handle["goes_imager_projection"]
        attrs = proj_ds.attrs
        h = float(np.asarray(attrs["perspective_point_height"]).reshape(-1)[0])
        lon0 = float(np.asarray(attrs["longitude_of_projection_origin"]).reshape(-1)[0])
        a = float(np.asarray(attrs["semi_major_axis"]).reshape(-1)[0])
        b = float(np.asarray(attrs["semi_minor_axis"]).reshape(-1)[0])
        sweep = attrs.get("sweep_angle_axis", b"x")
        if isinstance(sweep, bytes):
            sweep = sweep.decode("ascii", "ignore")
        crs = pyproj.CRS.from_proj4(f"+proj=geos +h={h} +lon_0={lon0} +sweep={sweep} +a={a} +b={b} +units=m +no_defs")
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        target_x_m, target_y_m = transformer.transform(fix.lon, fix.lat)
        x_m = x * h
        y_m = y * h
        ix = int(np.nanargmin(np.abs(x_m - target_x_m)))
        iy = int(np.nanargmin(np.abs(y_m - target_y_m)))
        half = int(CONFIG["crop_radius_px"])
        x0, x1 = max(0, ix - half), min(len(x), ix + half + 1)
        y0, y1 = max(0, iy - half), min(len(y), iy + half + 1)
        raw = np.asarray(cmi[y0:y1, x0:x1]).astype(np.float64)
        fill = cmi.attrs.get("_FillValue")
        if fill is not None:
            raw[raw == float(np.asarray(fill).reshape(-1)[0])] = np.nan
        raw = raw * float(np.asarray(cmi.attrs.get("scale_factor", 1.0)).reshape(-1)[0]) + float(np.asarray(cmi.attrs.get("add_offset", 0.0)).reshape(-1)[0])
        crop_x_m = x_m[x0:x1]
        crop_y_m = y_m[y0:y1]
        # Approximate horizontal pixel scale around center in projected space.
        dx_km = float(np.nanmedian(np.abs(np.diff(crop_x_m)))) / 1000.0 if len(crop_x_m) > 1 else 2.0
        dy_km = float(np.nanmedian(np.abs(np.diff(crop_y_m)))) / 1000.0 if len(crop_y_m) > 1 else 2.0
        return raw, dx_km, dy_km, float(ix - x0), float(iy - y0)


def eye_metrics(bt: np.ndarray, dx_km: float, dy_km: float, center_x: float, center_y: float) -> Tuple[float, float, float, float, Tuple[float, float]]:
    h, w = bt.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist_from_nhc = np.sqrt(((xx - center_x) * dx_km) ** 2 + ((yy - center_y) * dy_km) ** 2)
    candidate = np.where(dist_from_nhc <= 45.0, bt, np.nan)
    if not np.isfinite(candidate).any():
        return float("nan"), float("nan"), float("nan"), float("nan"), (center_x, center_y)

    # Warm eye candidate: use a small smoothed field to avoid one noisy warm pixel.
    finite = np.where(np.isfinite(bt), bt, np.nanmedian(bt))
    small = Image.fromarray(np.clip((finite - 180.0) / 130.0 * 255.0, 0, 255).astype(np.uint8))
    smoothed = np.asarray(small.filter(ImageFilter.GaussianBlur(radius=max(1.0, 5.0 / max(dx_km, 1e-3))))).astype(float)
    masked = np.where(dist_from_nhc <= 45.0, smoothed, -1e9)
    eye_y, eye_x = np.unravel_index(int(np.argmax(masked)), masked.shape)

    dist = np.sqrt(((xx - eye_x) * dx_km) ** 2 + ((yy - eye_y) * dy_km) ** 2)
    eye_values = bt[(dist <= 12.0) & np.isfinite(bt)]
    ring_values = bt[(dist >= 22.0) & (dist <= 60.0) & np.isfinite(bt)]
    if len(eye_values) < 5 or len(ring_values) < 20:
        return float("nan"), float("nan"), float("nan"), float("nan"), (float(eye_x), float(eye_y))
    eye_temp = float(np.nanmedian(eye_values))
    eyewall_temp = float(np.nanpercentile(ring_values, 20))
    contrast = eye_temp - eyewall_temp

    # Rough diameter: warm contiguous-looking area around candidate using a thermal threshold.
    threshold = eyewall_temp + 0.58 * max(contrast, 0.0)
    eye_mask = (dist <= 35.0) & np.isfinite(bt) & (bt >= threshold)
    area_km2 = float(np.sum(eye_mask)) * dx_km * dy_km
    diameter = 2.0 * math.sqrt(max(area_km2, 0.0) / math.pi) if area_km2 > 0 else float("nan")
    return eye_temp, eyewall_temp, contrast, diameter, (float(eye_x), float(eye_y))


def ir_palette(bt: np.ndarray) -> np.ndarray:
    """Cinematic enhancement of 180–310 K Band-13 brightness temperatures."""
    t = np.nan_to_num(bt, nan=300.0)
    stops = [
        (180.0, (255, 255, 255)),
        (195.0, (238, 95, 255)),
        (210.0, (92, 110, 255)),
        (225.0, (60, 219, 255)),
        (240.0, (204, 235, 245)),
        (255.0, (124, 142, 154)),
        (270.0, (62, 72, 80)),
        (285.0, (27, 31, 36)),
        (310.0, (4, 6, 9)),
    ]
    out = np.zeros(t.shape + (3,), dtype=np.float64)
    for (ta, ca), (tb, cb) in zip(stops[:-1], stops[1:]):
        mask = (t >= ta) & (t <= tb)
        u = np.clip((t - ta) / max(tb - ta, 1e-9), 0.0, 1.0)
        for ch in range(3):
            out[..., ch] = np.where(mask, ca[ch] * (1.0 - u) + cb[ch] * u, out[..., ch])
    out[t < stops[0][0]] = stops[0][1]
    out[t > stops[-1][0]] = stops[-1][1]
    return np.clip(out, 0, 255).astype(np.uint8)


def load_real_goes_frames() -> Tuple[List[SatelliteFrame], List[str]]:
    notes: List[str] = []
    if OFFLINE_MODE:
        return [], ["Offline mode requested with EYE_SHORT_OFFLINE=1"]
    if h5py is None or pyproj is None:
        return [], ["h5py or pyproj unavailable; cannot decode GOES-16 NetCDF files"]

    targets = SATELLITE_TARGET_TIMES
    if len(targets) > MAX_LIVE_FRAMES:
        indices = np.linspace(0, len(targets) - 1, MAX_LIVE_FRAMES).round().astype(int)
        targets = [targets[i] for i in sorted(set(indices))]

    frames: List[SatelliteFrame] = []
    for target in targets:
        try:
            chosen = choose_nearest_key(target)
            if chosen is None:
                notes.append(f"No Band-13 file found near {target.isoformat()}")
                continue
            key, scan_time = chosen
            path = download_goes_file(key)
            fix = interpolate_fix(scan_time)
            bt, dx_km, dy_km, center_x, center_y = extract_goes_crop(path, fix)
            eye_t, wall_t, contrast, diameter, _ = eye_metrics(bt, dx_km, dy_km, center_x, center_y)
            rgb = ir_palette(bt)
            image = Image.fromarray(rgb, "RGB").resize((720, 720), Image.Resampling.BICUBIC)
            frames.append(
                SatelliteFrame(
                    time_utc=scan_time,
                    image=image,
                    fix=fix,
                    source="NOAA GOES-16 ABI Band 13 CMI",
                    eye_temp_k=eye_t,
                    eyewall_temp_k=wall_t,
                    contrast_k=contrast,
                    eye_diameter_km=diameter,
                    raw_path=str(path),
                )
            )
        except Exception as exc:
            notes.append(f"GOES frame failed near {target.isoformat()}: {exc}")
    frames.sort(key=lambda f: f.time_utc)
    if frames:
        notes.append(f"Loaded {len(frames)} historical NOAA GOES-16 ABI Band 13 frames")
    return frames, notes


# -----------------------------------------------------------------------------
# Optional local-image path and synthetic fallback
# -----------------------------------------------------------------------------

def load_local_images() -> List[Image.Image]:
    if not LOCAL_IMAGE_DIR:
        return []
    root = Path(LOCAL_IMAGE_DIR).expanduser()
    if not root.is_dir():
        return []
    paths = sorted([p for p in root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
    images: List[Image.Image] = []
    for path in paths:
        try:
            images.append(Image.open(path).convert("RGB"))
        except Exception:
            pass
    return images


def synthetic_bt_field(size: int, progress: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cx = size * (0.50 + 0.010 * math.sin(progress * math.tau))
    cy = size * (0.50 + 0.008 * math.cos(progress * math.tau * 0.8))
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy) / size
    theta = np.arctan2(dy, dx)

    eye_radius = lerp(0.010, 0.047, smoothstep((progress - 0.20) / 0.70))
    wall_radius = eye_radius + lerp(0.045, 0.030, progress)
    wall_width = lerp(0.028, 0.018, progress)
    base = 286.0 - 28.0 * np.exp(-r / 0.32)
    eyewall = np.exp(-0.5 * ((r - wall_radius) / max(wall_width, 0.005)) ** 2)
    spiral = np.sin(theta * 5.0 + r * 48.0 - progress * 5.0)
    bands = np.exp(-r / 0.30) * (0.5 + 0.5 * spiral)
    bt = base - (50.0 + 18.0 * progress) * eyewall - 17.0 * bands

    eye = np.exp(-0.5 * (r / max(eye_radius, 0.002)) ** 4)
    bt += (8.0 + 29.0 * progress) * eye
    # Early raggedness partially fills the eye with cold fragments.
    ragged = max(0.0, 1.0 - progress * 1.5)
    bt -= ragged * 22.0 * eye * (0.5 + 0.5 * np.sin(theta * 3.0 + 8.0 * r))

    noise = rng.normal(0.0, 1.7, (size, size))
    smooth_noise = np.asarray(Image.fromarray(np.clip((noise + 5) * 20, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4))).astype(float)
    smooth_noise = (smooth_noise - np.mean(smooth_noise)) / max(np.std(smooth_noise), 1e-6)
    bt += 2.6 * smooth_noise
    return np.clip(bt, 178.0, 305.0)


def make_fallback_frames(count: int = 7) -> List[SatelliteFrame]:
    frames: List[SatelliteFrame] = []
    for i in range(count):
        u = i / max(count - 1, 1)
        t = SATELLITE_TARGET_TIMES[0] + (SATELLITE_TARGET_TIMES[-1] - SATELLITE_TARGET_TIMES[0]) * u
        fix = interpolate_fix(t)
        bt = synthetic_bt_field(420 if not QUICK_MODE else 280, u, seed=20241007 + i)
        size = bt.shape[0]
        dx = 1.6
        eye_t, wall_t, contrast, diameter, _ = eye_metrics(bt, dx, dx, size / 2, size / 2)
        frames.append(
            SatelliteFrame(
                time_utc=t,
                image=Image.fromarray(ir_palette(bt), "RGB").resize((720, 720), Image.Resampling.BICUBIC),
                fix=fix,
                source="SYNTHETIC IR RECONSTRUCTION",
                eye_temp_k=eye_t,
                eyewall_temp_k=wall_t,
                contrast_k=contrast,
                eye_diameter_km=diameter,
            )
        )
    return frames


def build_frames() -> Tuple[List[SatelliteFrame], str, List[str]]:
    notes: List[str] = []
    local = load_local_images()
    if local:
        frames: List[SatelliteFrame] = []
        for i, image in enumerate(local):
            u = i / max(len(local) - 1, 1)
            t = SATELLITE_TARGET_TIMES[0] + (SATELLITE_TARGET_TIMES[-1] - SATELLITE_TARGET_TIMES[0]) * u
            fix = interpolate_fix(t)
            frames.append(SatelliteFrame(t, image.resize((720, 720), Image.Resampling.BICUBIC), fix, "USER-SUPPLIED SATELLITE IMAGERY", float("nan"), float("nan"), float("nan"), float("nan")))
        return frames, "user_supplied_images", [f"Loaded {len(frames)} local satellite images"]

    live, live_notes = load_real_goes_frames()
    notes.extend(live_notes)
    if len(live) >= 3:
        return live, "noaa_goes16_abi_band13", notes

    notes.append("Using deterministic IR reconstruction fallback driven by NHC intensity data")
    return make_fallback_frames(5 if QUICK_MODE else 7), "synthetic_ir_reconstruction", notes


# -----------------------------------------------------------------------------
# Saved data products
# -----------------------------------------------------------------------------

def save_data_products(frames: Sequence[SatelliteFrame], source: str, notes: Sequence[str]) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "milton_eye_formation_timeline.csv"
    summary_path = DATA_ROOT / "eye_formation_summary.json"
    rows: List[Dict[str, Any]] = []
    for frame in frames:
        rows.append({
            "time_utc": frame.time_utc.isoformat(),
            "latitude": frame.fix.lat,
            "longitude": frame.fix.lon,
            "wind_mph": frame.fix.wind_mph,
            "pressure_mb": frame.fix.pressure_mb,
            "category": frame.fix.category,
            "frame_source": frame.source,
            "eye_temp_k": frame.eye_temp_k,
            "eyewall_temp_k_p20": frame.eyewall_temp_k,
            "eye_eyewall_contrast_k": frame.contrast_k,
            "estimated_warm_eye_diameter_km": frame.eye_diameter_km,
            "raw_goes_file": frame.raw_path,
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summary = {
        "title": "The Eye of a Hurricane Forms in Real Data",
        "storm": "Hurricane Milton",
        "date": "2024-10-07",
        "data_mode": source,
        "frame_count": len(frames),
        "scientific_note": (
            "When NOAA GOES files are available, imagery is ABI Band 13 Cloud and Moisture Imagery brightness temperature. "
            "The eye contrast/diameter values are descriptive calculations made by this script, not official NHC products."
        ),
        "fallback_warning": source == "synthetic_ir_reconstruction",
        "notes": list(notes),
        "source_urls": SOURCE_URLS,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return csv_path, summary_path


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class EyeScene:
    def __init__(self, frames: List[SatelliteFrame], source: str):
        self.frames = frames
        self.source = source
        rng = np.random.default_rng(1407)
        self.stars = [
            (float(rng.uniform(0, OUT_W)), float(rng.uniform(0, OUT_H)), float(rng.uniform(0.4, 1.7)), float(rng.uniform(10, 55)))
            for _ in range(120 if QUICK_MODE else 300)
        ]

    def frame_pair(self, u: float) -> Tuple[SatelliteFrame, SatelliteFrame, float]:
        u = clamp(u)
        pos = u * max(len(self.frames) - 1, 1)
        i = min(int(math.floor(pos)), len(self.frames) - 1)
        j = min(i + 1, len(self.frames) - 1)
        return self.frames[i], self.frames[j], pos - i

    def blended_satellite(self, u: float) -> Tuple[Image.Image, SatelliteFrame, float]:
        a, b, frac = self.frame_pair(u)
        if a is b or frac <= 0.001:
            return a.image.copy(), a, frac
        return Image.blend(a.image, b.image, smoothstep(frac)), b, frac

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, COLORS["black"] + (255,))
        draw = ImageDraw.Draw(image)
        for x, y, r, a in self.stars:
            pulse = 0.45 + 0.55 * math.sin(t * 0.45 + x * 0.013) ** 2
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(150, 194, 218, int(a * pulse)))
        return image

    def draw_panel(self, image: Image.Image, t: float, u: float, zoom: float = 1.0, show_target: bool = True) -> SatelliteFrame:
        sat, frame, _ = self.blended_satellite(u)
        left, top, right, bottom = [int(CONFIG[k]) for k in ("panel_left", "panel_top", "panel_right", "panel_bottom")]
        pw, ph = right-left, bottom-top
        sat = sat.convert("RGBA")
        if zoom > 1.001:
            nw, nh = int(sat.width * zoom), int(sat.height * zoom)
            sat = sat.resize((nw, nh), Image.Resampling.BICUBIC)
            x0 = max(0, (nw - 720)//2)
            y0 = max(0, (nh - 720)//2)
            sat = sat.crop((x0, y0, x0+720, y0+720))
        sat = sat.resize((pw, ph), Image.Resampling.LANCZOS)
        layer = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        layer.alpha_composite(sat, (left, top))
        image.alpha_composite(layer)
        d = ImageDraw.Draw(image)
        d.rounded_rectangle((left, top, right, bottom), radius=14 if QUICK_MODE else 28, outline=COLORS["grid"]+(125,), width=2 if not QUICK_MODE else 1)
        # crosshair indicates NHC center around which the crop was built
        if show_target:
            cx, cy = (left+right)//2, (top+bottom)//2
            rr = 12 if QUICK_MODE else 24
            d.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), outline=COLORS["yellow"]+(150,), width=1 if QUICK_MODE else 2)
            d.line((cx-rr*2, cy, cx-rr//2, cy), fill=COLORS["yellow"]+(105,), width=1)
            d.line((cx+rr//2, cy, cx+rr*2, cy), fill=COLORS["yellow"]+(105,), width=1)
            d.line((cx, cy-rr*2, cx, cy-rr//2), fill=COLORS["yellow"]+(105,), width=1)
            d.line((cx, cy+rr//2, cx, cy+rr*2), fill=COLORS["yellow"]+(105,), width=1)
        return frame

    def draw_hud(self, image: Image.Image, frame: SatelliteFrame, show_metric: bool = True):
        left = int(CONFIG["panel_left"])
        right = int(CONFIG["panel_right"])
        top = int(CONFIG["panel_bottom"]) + (16 if QUICK_MODE else 30)
        bottom = top + (112 if QUICK_MODE else 220)
        overlay = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((left, top, right, bottom), radius=14 if QUICK_MODE else 28, fill=(3,9,18,220), outline=COLORS["grid"]+(72,), width=1)
        image.alpha_composite(overlay)
        scale = OUT_W / 1080.0
        draw_text(image, frame.time_utc.strftime("%H:%M UTC // %d OCT 2024"), (left+int(22*scale), top+int(20*scale)), int(26*scale), COLORS["white"]+(245,), bold=True, condensed=True, stroke=1)
        draw_text(image, f"{frame.fix.wind_mph:.0f} MPH", (left+int(22*scale), top+int(67*scale)), int(38*scale), category_color(frame.fix.category)+(255,), bold=True, condensed=True, stroke=1)
        draw_text(image, f"{frame.fix.pressure_mb:.0f} MB", (left+int(214*scale), top+int(73*scale)), int(24*scale), COLORS["muted"]+(235,), bold=True, condensed=True, stroke=1)
        draw_text(image, category_label(frame.fix.wind_mph), (right-int(22*scale), top+int(29*scale)), int(22*scale), category_color(frame.fix.category)+(250,), bold=True, condensed=True, anchor="ra", stroke=1)
        if show_metric and np.isfinite(frame.contrast_k):
            draw_text(image, f"EYE / EYEWALL ΔT  +{frame.contrast_k:.0f} K", (right-int(22*scale), top+int(77*scale)), int(20*scale), COLORS["cyan"]+(245,), bold=True, condensed=True, anchor="ra", stroke=1)

    def draw_opening(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        frame = self.draw_panel(image, t, p*0.12, zoom=1.04+0.06*p)
        self.draw_hud(image, frame, show_metric=False)
        draw_text(image, "WATCH THE WARM CENTER", (OUT_W//2, int(OUT_H*0.86)), 17 if QUICK_MODE else 34, COLORS["white"]+(245,), bold=True, condensed=True, anchor="ma", stroke=1)
        draw_text(image, "EMERGE FROM THE COLD CLOUD SHIELD", (OUT_W//2, int(OUT_H*0.90)), 11 if QUICK_MODE else 22, COLORS["cyan"]+(235,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_ragged(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smootherstep(shot_progress(t, shot))
        frame = self.draw_panel(image, t, lerp(0.02,0.26,p), zoom=1.10)
        self.draw_hud(image, frame)
        draw_text(image, "SMALL, RAGGED EYE", (OUT_W//2, int(OUT_H*0.83)), 19 if QUICK_MODE else 38, COLORS["yellow"]+(255,), bold=True, condensed=True, anchor="ma", stroke=1)
        draw_text(image, "NHC DISCUSSION 8", (OUT_W//2, int(OUT_H*0.875)), 10 if QUICK_MODE else 20, COLORS["muted"]+(230,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_eyewall(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smootherstep(shot_progress(t, shot))
        frame = self.draw_panel(image, t, lerp(0.24,0.50,p), zoom=1.16)
        self.draw_hud(image, frame)
        # ring annotation
        left, top, right, bottom = [int(CONFIG[k]) for k in ("panel_left","panel_top","panel_right","panel_bottom")]
        cx, cy = (left+right)//2, (top+bottom)//2
        d=ImageDraw.Draw(image)
        rr=int((78 if QUICK_MODE else 155)*(0.95+0.04*math.sin(t*2)))
        d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr), outline=COLORS["cyan"]+(155,), width=2 if not QUICK_MODE else 1)
        draw_text(image, "COLD EYEWALL WRAPS AROUND THE CENTER", (OUT_W//2, int(OUT_H*0.84)), 13 if QUICK_MODE else 26, COLORS["white"]+(245,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_eye_opens(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smootherstep(shot_progress(t, shot))
        frame = self.draw_panel(image, t, lerp(0.48,0.72,p), zoom=1.22)
        self.draw_hud(image, frame)
        if np.isfinite(frame.eye_temp_k) and np.isfinite(frame.eyewall_temp_k):
            draw_text(image, f"EYE  {frame.eye_temp_k-273.15:+.0f}°C", (int(OUT_W*0.25), int(OUT_H*0.84)), 14 if QUICK_MODE else 28, COLORS["yellow"]+(250,), bold=True, condensed=True, anchor="ma", stroke=1)
            draw_text(image, f"EYEWALL  {frame.eyewall_temp_k-273.15:+.0f}°C", (int(OUT_W*0.75), int(OUT_H*0.84)), 14 if QUICK_MODE else 28, COLORS["cyan"]+(250,), bold=True, condensed=True, anchor="ma", stroke=1)
        draw_text(image, "THE WARM OPENING BECOMES CLEANER", (OUT_W//2, int(OUT_H*0.895)), 11 if QUICK_MODE else 23, COLORS["white"]+(245,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_defined(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        frame = self.draw_panel(image, t, lerp(0.70,1.0,p), zoom=1.27-0.05*p)
        self.draw_hud(image, frame)
        draw_text(image, "EYE BECOMING BETTER DEFINED", (OUT_W//2, int(OUT_H*0.835)), 16 if QUICK_MODE else 32, COLORS["magenta"]+(255,), bold=True, condensed=True, anchor="ma", stroke=1)
        draw_text(image, "NHC DISCUSSION 10 → 11", (OUT_W//2, int(OUT_H*0.88)), 10 if QUICK_MODE else 20, COLORS["muted"]+(230,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_compare(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        a = self.frames[0].image.resize((int(OUT_W*0.43), int(OUT_W*0.43)), Image.Resampling.LANCZOS)
        b = self.frames[-1].image.resize((int(OUT_W*0.43), int(OUT_W*0.43)), Image.Resampling.LANCZOS)
        y = int(OUT_H*0.34)
        x1 = int(OUT_W*0.04)
        x2 = OUT_W-int(OUT_W*0.04)-b.width
        image.alpha_composite(a.convert("RGBA"), (x1,y))
        image.alpha_composite(b.convert("RGBA"), (x2,y))
        d=ImageDraw.Draw(image)
        d.rounded_rectangle((x1,y,x1+a.width,y+a.height), radius=12 if QUICK_MODE else 24, outline=COLORS["grid"]+(130,), width=1)
        d.rounded_rectangle((x2,y,x2+b.width,y+b.height), radius=12 if QUICK_MODE else 24, outline=COLORS["magenta"]+(180,), width=2 if not QUICK_MODE else 1)
        draw_text(image, "09:00 UTC", (x1+a.width//2, y+a.height+int(25*OUT_W/1080)), 12 if QUICK_MODE else 24, COLORS["muted"]+(240,), bold=True, condensed=True, anchor="ma", stroke=1)
        draw_text(image, "LATER", (x2+b.width//2, y+b.height+int(25*OUT_W/1080)), 12 if QUICK_MODE else 24, COLORS["magenta"]+(250,), bold=True, condensed=True, anchor="ma", stroke=1)
        draw_text(image, "A HURRICANE EYE\nTAKES SHAPE", (OUT_W//2, int(OUT_H*0.76)), 30 if QUICK_MODE else 60, COLORS["white"]+(255,), bold=True, condensed=True, anchor="ma", stroke=2)

    def draw_titles(self, image: Image.Image, name: str):
        draw_text(image, str(CONFIG["title"]), (OUT_W//2, int(OUT_H*0.045)), 20 if QUICK_MODE else 40, COLORS["white"]+(250,), bold=True, condensed=True, anchor="ma", stroke=1)
        draw_text(image, str(CONFIG["title_2"]), (OUT_W//2, int(OUT_H*0.083)), 18 if QUICK_MODE else 36, COLORS["cyan"]+(250,), bold=True, condensed=True, anchor="ma", stroke=1)
        draw_text(image, str(CONFIG["subtitle"]), (OUT_W//2, int(OUT_H*0.113)), 8 if QUICK_MODE else 16, COLORS["muted"]+(220,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_source(self, image: Image.Image):
        if self.source == "noaa_goes16_abi_band13":
            text = "REAL DATA // NOAA GOES-16 ABI BAND 13 + NHC"
            color = COLORS["green"]
        elif self.source == "user_supplied_images":
            text = "USER-SUPPLIED SATELLITE IMAGERY + NHC"
            color = COLORS["yellow"]
        else:
            text = "SYNTHETIC IR RECONSTRUCTION // NHC DATA-DRIVEN"
            color = COLORS["orange"]
        draw_text(image, text, (OUT_W//2, int(OUT_H*0.965)), 8 if QUICK_MODE else 16, color+(230,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        x = int(OUT_W*0.06)
        y = int(OUT_H*0.905)
        w = int(OUT_W*0.88)
        overlay=Image.new("RGBA",OUT_SIZE,(0,0,0,0))
        od=ImageDraw.Draw(overlay)
        od.rounded_rectangle((x,y,w+x,int(OUT_H*0.955)), radius=10 if QUICK_MODE else 20, fill=(0,2,6,175))
        image.alpha_composite(overlay)
        draw_wrapped(image, text, (x+int(12*OUT_W/1080), y+int(10*OUT_W/1080)), w-int(24*OUT_W/1080), 8 if QUICK_MODE else 16, COLORS["white"]+(245,), spacing=2 if QUICK_MODE else 4)

    def texture(self, image: Image.Image, t: float) -> np.ndarray:
        arr = np.asarray(image.convert("RGB")).astype(np.float32)
        arr *= VIGNETTE[...,None]
        rng = np.random.default_rng(int(t*1000)+730)
        grain = rng.normal(0.0, float(CONFIG["grain_strength"]), arr.shape[:2])[...,None]
        arr = np.clip(arr+grain,0,255).astype(np.uint8)
        return arr

    def render_frame(self, t: float) -> np.ndarray:
        image = self.background(t)
        shot = get_shot(t)
        name = str(shot["name"])
        if name == "opening": self.draw_opening(image,t,shot)
        elif name == "ragged": self.draw_ragged(image,t,shot)
        elif name == "eyewall": self.draw_eyewall(image,t,shot)
        elif name == "eye_opens": self.draw_eye_opens(image,t,shot)
        elif name == "defined": self.draw_defined(image,t,shot)
        else: self.draw_compare(image,t,shot)
        self.draw_titles(image,name)
        if name != "compare": self.draw_caption(image,t)
        self.draw_source(image)
        return self.texture(image,t)


# -----------------------------------------------------------------------------
# Soundtrack / render
# -----------------------------------------------------------------------------

def gaussian(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5*((times-center)/max(width,1e-6))**2)


def generate_soundtrack(path: Path) -> Path:
    sr = int(CONFIG["soundtrack_sample_rate"])
    duration = float(CONFIG["duration_s"])
    n = int(round(sr*duration))
    times = np.arange(n,dtype=np.float64)/sr
    rng=np.random.default_rng(20241007)
    audio = 0.055*np.sin(math.tau*29.0*times+0.4*np.sin(math.tau*0.06*times))
    audio += 0.032*np.sin(math.tau*43.0*times+1.0)
    controls=rng.normal(0,1,max(8,int(duration*4)))
    audio += 0.020*np.interp(times,np.linspace(0,duration,len(controls)),controls)
    # Intensification pulses / eye reveal swell.
    for shot_name, amp, freq in [("ragged",0.08,80),("eyewall",0.10,96),("eye_opens",0.15,122),("defined",0.18,145)]:
        shot=next(s for s in SHOT_PLAN if s["name"]==shot_name)
        center=(float(shot["start"])+float(shot["end"]))/2
        env=gaussian(times,center,max(0.4,(float(shot["end"])-float(shot["start"]))*0.28))
        audio += amp*env*np.sin(math.tau*freq*times)
    reveal=next(s for s in SHOT_PLAN if s["name"]=="defined")
    center=float(reveal["start"])+0.22*(float(reveal["end"])-float(reveal["start"]))
    audio += 0.22*gaussian(times,center,0.35 if not QUICK_MODE else 0.12)*np.sin(math.tau*210*times)
    intro=np.clip(times/max(1.4,duration*0.08),0,1)
    outro=1-np.clip((times-(duration-1.2))/1.0,0,1)
    audio*=intro*outro
    peak=max(float(np.max(np.abs(audio))),1e-9)
    pcm=(np.clip(audio/peak*0.88,-1,1)*32767).astype(np.int16)
    with wave.open(str(path),"wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(sr); handle.writeframes(pcm.tobytes())
    return path


def find_ffmpeg() -> Optional[str]:
    if imageio_ffmpeg is not None:
        try: return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception: pass
    return shutil.which("ffmpeg")


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    ffmpeg=find_ffmpeg()
    if not ffmpeg: return False
    cmd=[ffmpeg,"-y","-i",str(video_path),"-i",str(audio_path),"-c:v","copy","-c:a","aac","-b:a","192k","-shortest",str(output_path)]
    try:
        subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return output_path.exists() and output_path.stat().st_size>0
    except Exception:
        return False


def render_video(scene: EyeScene) -> Path:
    srt_path=OUTPUT_ROOT/f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS,srt_path)
    raw=OUTPUT_ROOT/f"{CONFIG['output_basename']}_silent.mp4"
    final=OUTPUT_ROOT/f"{CONFIG['output_basename']}_final.mp4"
    audio=OUTPUT_ROOT/f"{CONFIG['output_basename']}_ambient.wav"
    frame_count=int(round(float(CONFIG["duration_s"])*int(CONFIG["fps"])))
    times=np.arange(frame_count)/int(CONFIG["fps"])
    print("Subtitle sidecar:",srt_path.resolve())
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(raw,fps=int(CONFIG["fps"]),codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=None) as writer:
        for t in tqdm(times,desc="Rendering hurricane-eye short"):
            writer.append_data(scene.render_frame(float(t)))
    if not NO_AUDIO:
        generate_soundtrack(audio)
        if mux_audio(raw,audio,final):
            print("Final video with audio:",final.resolve()); return final
    shutil.copyfile(raw,final)
    print("Final video (silent fallback):",final.resolve())
    return final

