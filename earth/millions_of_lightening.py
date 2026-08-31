from __future__ import annotations

"""
Millions of Lightning Strikes in Seconds — cinematic YouTube Short renderer

Creates a vertical 1080x1920 data-driven / climatology-driven YouTube Short that
compresses a represented lightning window (default: one global day) into ~58 seconds.
At the classic NASA/OTD estimate of about 44 total lightning flashes per second,
one day corresponds to ~3.8 million flashes worldwide.

IMPORTANT SCIENTIFIC FRAMING
----------------------------
The requested title uses the word "strikes", but satellite lightning climatologies
measure TOTAL lightning flashes, including intracloud lightning. The video therefore
uses "strikes" as a dramatic title while captions and metadata say "flashes" where
scientific precision matters.

Default mode
------------
The script does NOT pretend to contain millions of individually observed detections.
Instead, it creates a deterministic representative sample whose geographic weights are
inspired by NASA LIS/OTD lightning climatology findings (strong land preference,
tropical maxima, Lake Maracaibo / Congo / East African Rift / South & Southeast Asia,
Central America, Florida, etc.). Each rendered sample point represents many flashes.
The count HUD scales to the estimated total flashes in the represented time window.



Recommended install
-------------------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm pyshp

Outputs
-------
- final vertical MP4 with generated thunder/electric ambient audio when ffmpeg exists
- silent MP4 fallback
- SRT subtitle sidecar
- preview PNG frames
- CSV export of representative / imported flash points
- JSON summary and source notes
- cached Natural Earth land geometry


"""

import json
import math
import os
import shutil
import subprocess
import urllib.request
import wave
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import shapefile  # pyshp
except Exception:
    shapefile = None

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("LIGHTNING_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("LIGHTNING_SHORT_OFFLINE", "0") == "1"
CSV_OVERRIDE = os.environ.get("LIGHTNING_CSV", "").strip()

REPRESENT_SECONDS = max(60.0, float(os.environ.get("LIGHTNING_REPRESENT_SECONDS", "86400")))
GLOBAL_FLASH_RATE = max(1.0, float(os.environ.get("LIGHTNING_GLOBAL_FLASH_RATE", "44")))
DEFAULT_SAMPLE_COUNT = 18000 if QUICK_MODE else 90000
SAMPLE_COUNT = max(2000, int(os.environ.get("LIGHTNING_SAMPLE_COUNT", str(DEFAULT_SAMPLE_COUNT))))

NOW_UTC = datetime.now(timezone.utc)
WINDOW_END = NOW_UTC
WINDOW_START = WINDOW_END - timedelta(seconds=REPRESENT_SECONDS)
WINDOW_LABEL = (
    "24 HOURS OF GLOBAL LIGHTNING"
    if abs(REPRESENT_SECONDS - 86400.0) < 1.0
    else f"{REPRESENT_SECONDS / 3600.0:.1f} HOURS OF GLOBAL LIGHTNING"
)

OUTPUT_ROOT = Path("millions_of_lightning_strikes_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
CACHE_ROOT = DATA_ROOT / "cache"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in [OUTPUT_ROOT, DATA_ROOT, CACHE_ROOT, PREVIEW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {

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

NATURAL_EARTH_LAND_URL = "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip"
NASA_GLOBAL_RATE_URL = "https://ntrs.nasa.gov/search.jsp?R=20020051098"
NASA_MARACAIBO_URL = "https://www.nasa.gov/missions/trmm/earths-new-lightning-capital-revealed/"
NASA_CLIMATOLOGY_URL = "https://www.earthdata.nasa.gov/data/catalog/ghrc-daac-lisvhrdc-1"
NASA_NEW_MAP_URL = "https://science.nasa.gov/earth/earth-observatory/a-new-look-at-earths-lightning-149301/"


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FlashPoint:
    flash_id: str
    time_utc: datetime
    latitude: float
    longitude: float
    intensity: float
    cluster: str
    weight: float = 1.0

    @property
    def time_fraction(self) -> float:
        span = max((WINDOW_END - WINDOW_START).total_seconds(), 1.0)
        return clamp((self.time_utc - WINDOW_START).total_seconds() / span)


@dataclass(frozen=True)
class Hotspot:
    name: str
    latitude: float
    longitude: float
    sigma_lon: float
    sigma_lat: float
    weight: float
    intensity_boost: float = 1.0


# A compact, qualitative hotspot mixture designed to resemble broad LIS/OTD patterns.
# It is NOT a reproduction of the NASA gridded dataset.
HOTSPOTS: List[Hotspot] = [
    Hotspot("Lake Maracaibo", 9.75, -71.65, 2.5, 2.2, 0.075, 1.50),
    Hotspot("Congo Basin", -1.5, 27.0, 10.0, 7.0, 0.125, 1.35),
    Hotspot("Lake Kivu / Rift", -1.8, 29.1, 5.0, 5.0, 0.060, 1.35),
    Hotspot("Lake Victoria", -1.0, 33.0, 6.5, 5.5, 0.055, 1.20),
    Hotspot("Cameroon Line", 6.0, 10.0, 7.0, 5.0, 0.045, 1.15),
    Hotspot("Guinea Coast", 7.0, -2.0, 12.0, 6.0, 0.040, 1.05),
    Hotspot("Amazon North", -3.0, -62.0, 16.0, 8.0, 0.070, 1.05),
    Hotspot("Colombia / Andes", 5.0, -75.0, 7.0, 6.0, 0.065, 1.25),
    Hotspot("Central America", 12.0, -87.0, 9.0, 5.5, 0.045, 1.10),
    Hotspot("Florida / Gulf", 27.0, -82.0, 8.0, 5.5, 0.035, 1.05),
    Hotspot("Mississippi Valley", 34.0, -90.0, 12.0, 7.0, 0.028, 1.00),
    Hotspot("Brahmaputra / NE India", 26.0, 91.0, 8.0, 5.0, 0.055, 1.25),
    Hotspot("Himalayan Foothills", 29.0, 80.0, 13.0, 6.0, 0.045, 1.10),
    Hotspot("Bay of Bengal / Myanmar", 18.0, 96.0, 10.0, 7.0, 0.035, 1.05),
    Hotspot("Indochina", 16.0, 105.0, 12.0, 7.0, 0.040, 1.05),
    Hotspot("Indonesia", -3.0, 115.0, 18.0, 7.0, 0.065, 1.10),
    Hotspot("Papua New Guinea", -5.0, 145.0, 11.0, 6.0, 0.035, 1.10),
    Hotspot("Northern Australia", -15.0, 132.0, 14.0, 7.0, 0.028, 1.00),
    Hotspot("Madagascar / Mozambique", -18.0, 42.0, 11.0, 8.0, 0.025, 1.00),
    Hotspot("South Africa", -25.0, 27.0, 10.0, 7.0, 0.020, 0.95),
    Hotspot("Argentina / Paraguay", -27.0, -59.0, 13.0, 8.0, 0.030, 1.05),
    Hotspot("Southeast Brazil", -20.0, -47.0, 12.0, 8.0, 0.028, 1.00),
    # Oceanic/background storm corridors intentionally get less weight.
    Hotspot("West Pacific Ocean", 12.0, 145.0, 20.0, 10.0, 0.018, 0.80),
    Hotspot("Indian Ocean", -8.0, 78.0, 22.0, 11.0, 0.015, 0.75),
    Hotspot("Atlantic ITCZ", 5.0, -30.0, 22.0, 8.0, 0.014, 0.75),
]


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
):
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold=bold, condensed=condensed),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(235, fill[3])),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill: Tuple[int, int, int, int],
    line_spacing: int,
):
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
    return np.clip(1.0 - strength * radius**1.75, 0.0, 1.0).astype(np.float32)


def apply_grade(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array)
    image = ImageEnhance.Contrast(image).enhance(float(CONFIG["contrast"]))
    image = ImageEnhance.Color(image).enhance(float(CONFIG["saturation"]))
    return np.asarray(image)


def request_bytes(url: str, timeout: int = 45) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; LightningShort/1.0; educational visualization)",
            "Accept": "application/zip,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def alpha_composite_with_opacity(base: Image.Image, layer: Image.Image, opacity: float) -> None:
    opacity = clamp(opacity)
    if opacity <= 0:
        return
    if opacity < 0.999:
        layer = layer.copy()
        alpha = layer.getchannel("A").point(lambda value: int(value * opacity))
        layer.putalpha(alpha)
    base.alpha_composite(layer)


def zoom_and_shift(layer: Image.Image, zoom: float, dx: float, dy: float) -> Image.Image:
    zoom = max(1.0, float(zoom))
    new_size = (int(round(OUT_W * zoom)), int(round(OUT_H * zoom)))
    resized = layer.resize(new_size, Image.Resampling.BICUBIC)
    left = int((new_size[0] - OUT_W) / 2.0 - dx)
    top = int((new_size[1] - OUT_H) / 2.0 - dy)
    left = max(0, min(left, new_size[0] - OUT_W))
    top = max(0, min(top, new_size[1] - OUT_H))
    return resized.crop((left, top, left + OUT_W, top + OUT_H))


def estimate_total_flashes() -> int:
    return int(round(GLOBAL_FLASH_RATE * REPRESENT_SECONDS))


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Lightning data loading
# -----------------------------------------------------------------------------

def _pick_column(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lowered = {str(column).strip().lower(): str(column) for column in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    for column in columns:
        low = str(column).strip().lower()
        for alias in aliases:
            if alias in low:
                return str(column)
    return None


def load_csv_flashes(path: Path) -> Tuple[List[FlashPoint], Dict[str, Any]]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("LIGHTNING_CSV contains no rows")

    lat_col = _pick_column(frame.columns, ["latitude", "lat"])
    lon_col = _pick_column(frame.columns, ["longitude", "lon", "lng"])
    time_col = _pick_column(frame.columns, ["timestamp", "datetime", "time", "date"])
    intensity_col = _pick_column(frame.columns, ["energy", "power", "intensity", "amplitude"])
    if lat_col is None or lon_col is None:
        raise ValueError("LIGHTNING_CSV must contain latitude/lat and longitude/lon columns")

    work = frame.copy()
    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
    work = work[np.isfinite(work[lat_col]) & np.isfinite(work[lon_col])]
    work = work[(work[lat_col] >= -90) & (work[lat_col] <= 90)]
    if work.empty:
        raise ValueError("LIGHTNING_CSV has no valid latitude/longitude rows")

    if time_col is not None:
        parsed = pd.to_datetime(work[time_col], errors="coerce", utc=True)
        valid = parsed.notna()
        work = work.loc[valid].copy()
        parsed = parsed.loc[valid]
        if work.empty:
            time_col = None
        else:
            order = np.argsort(parsed.astype("int64").to_numpy())
            work = work.iloc[order].reset_index(drop=True)
            parsed = parsed.iloc[order].reset_index(drop=True)
            raw_min = parsed.iloc[0].to_pydatetime()
            raw_max = parsed.iloc[-1].to_pydatetime()
            raw_span = max((raw_max - raw_min).total_seconds(), 1.0)
    else:
        parsed = None
        raw_min = WINDOW_START
        raw_span = REPRESENT_SECONDS

    if intensity_col is not None:
        intensity_values = pd.to_numeric(work[intensity_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(intensity_values)
        if finite.any():
            lo, hi = np.nanpercentile(intensity_values[finite], [5, 95])
            denom = max(hi - lo, 1e-9)
            normalized = np.clip((intensity_values - lo) / denom, 0.0, 1.0)
            normalized[~finite] = 0.45
            normalized = 0.25 + normalized * 0.75
        else:
            normalized = np.full(len(work), 0.55)
    else:
        normalized = np.full(len(work), 0.55)

    # Downsample huge files deterministically while preserving chronology.
    max_points = SAMPLE_COUNT
    if len(work) > max_points:
        indices = np.linspace(0, len(work) - 1, max_points).astype(int)
        work = work.iloc[indices].reset_index(drop=True)
        normalized = normalized[indices]
        if parsed is not None:
            parsed = parsed.iloc[indices].reset_index(drop=True)

    total_rows = len(frame)
    retained = len(work)
    weight = max(total_rows / max(retained, 1), 1.0)
    flashes: List[FlashPoint] = []
    for idx, row in work.iterrows():
        if parsed is not None:
            fraction = clamp((parsed.iloc[idx].to_pydatetime() - raw_min).total_seconds() / raw_span)
        else:
            fraction = idx / max(retained - 1, 1)
        t = WINDOW_START + timedelta(seconds=REPRESENT_SECONDS * fraction)
        flashes.append(
            FlashPoint(
                flash_id=f"csv_{idx:07d}",
                time_utc=t,
                latitude=float(row[lat_col]),
                longitude=((float(row[lon_col]) + 180.0) % 360.0) - 180.0,
                intensity=float(normalized[idx]),
                cluster="CSV detection",
                weight=weight,
            )
        )

    metadata = {
        "csv_path": str(path.resolve()),
        "input_rows": int(total_rows),
        "retained_rows": int(retained),
        "representative_weight": float(weight),
        "latitude_column": lat_col,
        "longitude_column": lon_col,
        "time_column": time_col,
        "intensity_column": intensity_col,
    }
    return flashes, metadata


def make_climatology_sample() -> Tuple[List[FlashPoint], Dict[str, Any]]:
    # Stable seed makes previews reproducible but varies slightly with represented hours.
    seed = 441996 + int(round(REPRESENT_SECONDS / 3600.0)) * 17
    rng = np.random.default_rng(seed)
    weights = np.array([spot.weight for spot in HOTSPOTS], dtype=float)
    weights /= weights.sum()

    sample_count = SAMPLE_COUNT
    estimated_total = estimate_total_flashes()
    representative_weight = estimated_total / max(sample_count, 1)
    choices = rng.choice(len(HOTSPOTS), size=sample_count, p=weights)

    # Time distribution: not a literal diurnal climatology; just enough storm burstiness
    # to make the time-compression visual feel organic.
    base_times = np.sort(rng.random(sample_count))
    jitter = 0.010 * np.sin(np.linspace(0, math.tau * 9.0, sample_count))
    fractions = np.clip(base_times + jitter, 0.0, 1.0)
    fractions.sort()

    flashes: List[FlashPoint] = []
    for idx in range(sample_count):
        spot = HOTSPOTS[int(choices[idx])]
        lon = float(rng.normal(spot.longitude, spot.sigma_lon))
        lat = float(rng.normal(spot.latitude, spot.sigma_lat))

        # Occasional storm-anvil spread elongates some clusters east-west.
        if rng.random() < 0.22:
            lon += float(rng.normal(0.0, spot.sigma_lon * 0.75))
        lon = ((lon + 180.0) % 360.0) - 180.0
        lat = float(np.clip(lat, -58.0, 72.0))

        intensity = float(np.clip(rng.beta(1.6, 3.8) * spot.intensity_boost + 0.12, 0.12, 1.0))
        t = WINDOW_START + timedelta(seconds=REPRESENT_SECONDS * float(fractions[idx]))
        flashes.append(
            FlashPoint(
                flash_id=f"model_{idx:07d}",
                time_utc=t,
                latitude=lat,
                longitude=lon,
                intensity=intensity,
                cluster=spot.name,
                weight=representative_weight,
            )
        )

    metadata = {
        "estimated_total_flashes": int(estimated_total),
        "assumed_global_flash_rate_per_second": float(GLOBAL_FLASH_RATE),
        "represented_seconds": float(REPRESENT_SECONDS),
        "sample_points": int(sample_count),
        "representative_flashes_per_sample": float(representative_weight),
        "model_note": (
            "Deterministic representative sample inspired by broad NASA LIS/OTD climatology patterns. "
            "It is not a direct download or reconstruction of the NASA gridded dataset."
        ),
    }
    return flashes, metadata


def load_flashes() -> Tuple[List[FlashPoint], str, List[str], Dict[str, Any]]:
    notes: List[str] = []
    if CSV_OVERRIDE:
        path = Path(CSV_OVERRIDE).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"LIGHTNING_CSV does not exist: {path}")
        flashes, metadata = load_csv_flashes(path)
        notes.append("Loaded user-provided lightning detections from CSV")
        notes.append("Rendered point count may be downsampled for performance; representative weight is recorded")
        return flashes, "user_csv_lightning_detections", notes, metadata

    flashes, metadata = make_climatology_sample()
    notes.append("No LIGHTNING_CSV supplied; using deterministic climatology-driven representative sample")
    notes.append("Global total is estimated with the configured flashes-per-second rate; sample points are weighted")
    notes.append("The model is suitable for visualization, not event-level scientific analysis")
    return flashes, "nasa_climatology_inspired_model", notes, metadata


# -----------------------------------------------------------------------------
# Natural Earth land geometry
# -----------------------------------------------------------------------------

BUILTIN_LAND_POLYGONS: List[List[Tuple[float, float]]] = [
    [(-168, 72), (-140, 70), (-124, 55), (-126, 42), (-115, 30), (-101, 20), (-83, 8), (-77, 18), (-82, 25), (-80, 32), (-66, 46), (-52, 56), (-75, 72), (-168, 72)],
    [(-82, 12), (-70, 12), (-53, 4), (-35, -8), (-42, -25), (-58, -55), (-72, -50), (-80, -20), (-82, 12)],
    [(-17, 37), (5, 36), (34, 31), (50, 12), (42, -12), (33, -35), (18, -35), (6, -12), (-15, 12), (-17, 37)],
    [(-10, 72), (40, 72), (78, 66), (110, 56), (145, 50), (180, 64), (180, 8), (140, 6), (122, 22), (105, 5), (78, 8), (52, 28), (28, 38), (10, 45), (-10, 58), (-10, 72)],
    [(112, -10), (154, -12), (154, -39), (138, -45), (116, -35), (112, -10)],
    [(-74, 59), (-44, 83), (-18, 72), (-35, 59), (-74, 59)],
    [(-180, -62), (-120, -70), (-60, -65), (0, -72), (60, -66), (120, -72), (180, -62), (180, -90), (-180, -90), (-180, -62)],
    [(44, -12), (51, -14), (49, -26), (44, -25), (44, -12)],
    [(166, -34), (178, -38), (174, -47), (166, -46), (166, -34)],
    [(95, 5), (141, 6), (151, -10), (130, -12), (108, -7), (95, 5)],
]


def load_land_polygons() -> Tuple[List[List[Tuple[float, float]]], str, List[str]]:
    notes: List[str] = []
    if OFFLINE_MODE or shapefile is None:
        if OFFLINE_MODE:
            notes.append("Offline mode requested; using coarse built-in land polygons")
        elif shapefile is None:
            notes.append("pyshp unavailable; using coarse built-in land polygons")
        return BUILTIN_LAND_POLYGONS, "built_in_coarse_land", notes

    zip_path = CACHE_ROOT / "ne_110m_land.zip"
    extract_dir = CACHE_ROOT / "ne_110m_land"
    shp_path = extract_dir / "ne_110m_land.shp"
    try:
        if not shp_path.exists():
            if not zip_path.exists() or zip_path.stat().st_size < 10_000:
                zip_path.write_bytes(request_bytes(NATURAL_EARTH_LAND_URL, timeout=45))
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_dir)
        reader = shapefile.Reader(str(shp_path))
        polygons: List[List[Tuple[float, float]]] = []
        for shape in reader.shapes():
            points = shape.points
            parts = list(shape.parts) + [len(points)]
            for start, end in zip(parts[:-1], parts[1:]):
                polygon = [(float(lon), float(lat)) for lon, lat in points[start:end]]
                if len(polygon) >= 3:
                    polygons.append(polygon)
        notes.append("Loaded Natural Earth 110m land polygons")
        return polygons, "natural_earth_110m_land", notes
    except Exception as exc:
        notes.append(f"Natural Earth fallback: {exc}")
        return BUILTIN_LAND_POLYGONS, "built_in_coarse_land", notes


def unwrapped_relative_longitudes(polygon: Sequence[Tuple[float, float]], center_lon: float) -> List[float]:
    values: List[float] = []
    previous: Optional[float] = None
    for lon, _ in polygon:
        value = ((lon - center_lon + 180.0) % 360.0) - 180.0
        if previous is not None:
            while value - previous > 180.0:
                value -= 360.0
            while value - previous < -180.0:
                value += 360.0
        values.append(value)
        previous = value
    return values


# -----------------------------------------------------------------------------
# Summaries and saved products
# -----------------------------------------------------------------------------

def flash_summary(
    flashes: Sequence[FlashPoint],
    source: str,
    land_source: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    total_weighted = int(round(sum(point.weight for point in flashes)))
    if source == "nasa_climatology_inspired_model":
        total_weighted = estimate_total_flashes()
    intensities = np.array([point.intensity for point in flashes], dtype=float)
    cluster_counts: Dict[str, float] = {}
    for point in flashes:
        cluster_counts[point.cluster] = cluster_counts.get(point.cluster, 0.0) + point.weight
    top_clusters = sorted(cluster_counts.items(), key=lambda item: item[1], reverse=True)[:8]
    return {
        "title": CONFIG["title"] + " " + CONFIG["title_2"],
        "data_source": source,
        "land_source": land_source,
        "window_start_utc": WINDOW_START.isoformat(),
        "window_end_utc": WINDOW_END.isoformat(),
        "represented_seconds": REPRESENT_SECONDS,
        "represented_hours": REPRESENT_SECONDS / 3600.0,
        "sample_points": len(flashes),
        "weighted_flash_count": total_weighted,
        "configured_global_flash_rate_per_second": GLOBAL_FLASH_RATE,
        "mean_sample_intensity": float(np.mean(intensities)) if len(intensities) else None,
        "top_model_clusters_by_weight": [{"name": name, "weighted_flashes": value} for name, value in top_clusters],
        "metadata": metadata,
        "scientific_caveat": (
            "In model mode, rendered points are a climatology-inspired representative sample, not individual observations. "
            "The dramatic title uses 'strikes', while the reference rate and satellite climatologies describe total lightning flashes."
        ),
    }


def save_data_products(
    flashes: Sequence[FlashPoint],
    summary: Dict[str, Any],
    notes: Sequence[str],
) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "lightning_representative_points.csv"
    summary_path = DATA_ROOT / "lightning_summary.json"
    pd.DataFrame(
        [
            {
                "flash_id": point.flash_id,
                "time_utc": point.time_utc.isoformat(),
                "latitude": point.latitude,
                "longitude": point.longitude,
                "intensity": point.intensity,
                "cluster": point.cluster,
                "representative_weight": point.weight,
            }
            for point in flashes
        ]
    ).to_csv(csv_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "notes": list(notes),
                "source_urls": {
                    "nasa_global_rate": NASA_GLOBAL_RATE_URL,
                    "nasa_lis_otd_climatology": NASA_CLIMATOLOGY_URL,
                    "nasa_maracaibo": NASA_MARACAIBO_URL,
                    "nasa_new_lightning_map": NASA_NEW_MAP_URL,
                    "natural_earth_land": NATURAL_EARTH_LAND_URL,
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return csv_path, summary_path


# -----------------------------------------------------------------------------
# Visual helpers
# -----------------------------------------------------------------------------

def intensity_color(intensity: float, alpha: int = 230) -> Tuple[int, int, int, int]:
    x = clamp(intensity)
    if x > 0.78:
        c = COLORS["white"]
    elif x > 0.52:
        c = COLORS["gold"]
    elif x > 0.30:
        c = COLORS["electric"]
    else:
        c = COLORS["violet"]
    return c + (alpha,)


def marker_radius(intensity: float) -> float:
    scale = OUT_W / 1080.0
    return (1.25 + 3.5 * clamp(intensity) ** 1.7) * scale


def compact_count(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class LightningScene:
    def __init__(
        self,
        flashes: List[FlashPoint],
        land_polygons: List[List[Tuple[float, float]]],
        summary: Dict[str, Any],
    ):
        self.flashes = sorted(flashes, key=lambda point: point.time_utc)
        self.land_polygons = land_polygons
        self.summary = summary
        self.total_weighted = float(summary["weighted_flash_count"])
        self.event_fractions = np.array([point.time_fraction for point in self.flashes], dtype=float)
        self.weights = np.array([point.weight for point in self.flashes], dtype=float)
        self.cumulative_weights = np.cumsum(self.weights)
        self.particles = self._make_particles(int(CONFIG["background_particles"]), seed=220)
        self.dust = self._make_particles(int(CONFIG["dust_particles"]), seed=884)
        self.hotspot_points = {
            "maracaibo": (-71.65, 9.75),
            "congo": (27.0, -1.5),
            "india": (91.0, 26.0),
        }
        self.base_maps = {
            "world": self._render_static_map(0.0),
            "americas": self._render_static_map(-72.0),
            "africa": self._render_static_map(24.0),
        }
        self.all_layers = {
            "world": self._make_flash_layer(self.flashes, 0.0, alpha_scale=0.80),
            "americas": self._make_flash_layer(self.flashes, -72.0, alpha_scale=0.80),
            "africa": self._make_flash_layer(self.flashes, 24.0, alpha_scale=0.80),
        }
        self.timeline_layers = self._build_timeline_layers()

    @staticmethod
    def _make_particles(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.35, 1.6 if QUICK_MODE else 2.2)),
                "a": float(rng.uniform(10, 72)),
                "phase": float(rng.uniform(0, math.tau)),
                "speed": float(rng.uniform(2.0, 15.0)),
            }
            for _ in range(count)
        ]

    def project(self, lon: float, lat: float, center_lon: float = 0.0) -> Tuple[float, float]:
        relative = ((lon - center_lon + 180.0) % 360.0) - 180.0
        x0 = float(CONFIG["map_margin_x"])
        x1 = OUT_W - float(CONFIG["map_margin_x"])
        y0 = float(CONFIG["map_top"])
        y1 = float(CONFIG["map_bottom"])
        x = x0 + (relative + 180.0) / 360.0 * (x1 - x0)
        lat_clamped = float(np.clip(lat, CONFIG["map_lat_min"], CONFIG["map_lat_max"]))
        y = y0 + (float(CONFIG["map_lat_max"]) - lat_clamped) / (
            float(CONFIG["map_lat_max"]) - float(CONFIG["map_lat_min"])
        ) * (y1 - y0)
        return x, y

    def _render_static_map(self, center_lon: float) -> Image.Image:
        array = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
        top = np.array(COLORS["sky_top"], dtype=float)
        bottom = np.array(COLORS["sky_bottom"], dtype=float)
        for y in range(OUT_H):
            u = y / max(OUT_H - 1, 1)
            rgb = (top * (1.0 - u) + bottom * u).astype(np.uint8)
            array[y, :, :3] = rgb
            array[y, :, 3] = 255
        image = Image.fromarray(array, "RGBA")
        draw = ImageDraw.Draw(image)
        x0 = int(CONFIG["map_margin_x"])
        x1 = OUT_W - x0
        y0 = int(CONFIG["map_top"])
        y1 = int(CONFIG["map_bottom"])

        for lat in range(-60, 76, 15):
            _, y = self.project(center_lon, float(lat), center_lon)
            draw.line((x0, y, x1, y), fill=COLORS["grid"] + (25,), width=1)
        for relative_lon in range(-150, 181, 30):
            lon = center_lon + relative_lon
            x, _ = self.project(lon, 0.0, center_lon)
            draw.line((x, y0, x, y1), fill=COLORS["grid"] + (22,), width=1)

        for polygon in self.land_polygons:
            relative_lons = unwrapped_relative_longitudes(polygon, center_lon)
            lats = [lat for _, lat in polygon]
            for shift in (-360.0, 0.0, 360.0):
                points: List[Tuple[float, float]] = []
                for relative, lat in zip(relative_lons, lats):
                    x = x0 + (relative + shift + 180.0) / 360.0 * (x1 - x0)
                    lat_clamped = float(np.clip(lat, CONFIG["map_lat_min"], CONFIG["map_lat_max"]))
                    y = y0 + (float(CONFIG["map_lat_max"]) - lat_clamped) / (
                        float(CONFIG["map_lat_max"]) - float(CONFIG["map_lat_min"])
                    ) * (y1 - y0)
                    points.append((x, y))
                if points and max(p[0] for p in points) >= x0 - 50 and min(p[0] for p in points) <= x1 + 50:
                    draw.polygon(points, fill=COLORS["land"] + (255,), outline=COLORS["land_edge"] + (105,))

        _, eq_y = self.project(center_lon, 0.0, center_lon)
        draw.line((x0, eq_y, x1, eq_y), fill=COLORS["grid"] + (52,), width=1)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=14 if QUICK_MODE else 28, outline=COLORS["grid"] + (66,), width=1)

        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for cx, cy, color in [
            (OUT_W * 0.23, OUT_H * 0.37, COLORS["violet"]),
            (OUT_W * 0.77, OUT_H * 0.49, COLORS["electric"]),
        ]:
            for radius, alpha in [
                (OUT_W * 0.38, 8),
                (OUT_W * 0.23, 12),
                (OUT_W * 0.12, 18),
            ]:
                hd.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color + (alpha,))
        haze = haze.filter(ImageFilter.GaussianBlur(55 if not QUICK_MODE else 28))
        image.alpha_composite(haze)
        return image

    def _make_flash_layer(
        self,
        flashes: Iterable[FlashPoint],
        center_lon: float,
        alpha_scale: float = 1.0,
    ) -> Image.Image:
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        core = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        cd = ImageDraw.Draw(core)
        # Prebuilt layers intentionally use tiny markers; density does the storytelling.
        for point in flashes:
            x, y = self.project(point.longitude, point.latitude, center_lon)
            radius = max(0.7 * OUT_W / 1080.0, marker_radius(point.intensity) * 0.65)
            color = intensity_color(point.intensity, int(105 * clamp(alpha_scale)))
            glow_radius = max(radius * 2.0, 1.8 * OUT_W / 1080.0)
            gd.ellipse((x - glow_radius, y - glow_radius, x + glow_radius, y + glow_radius), fill=color)
            core_color = intensity_color(point.intensity, int(175 * clamp(alpha_scale)))
            cd.ellipse((x - radius, y - radius, x + radius, y + radius), fill=core_color)
        glow = glow.filter(ImageFilter.GaussianBlur(4 if QUICK_MODE else 8))
        glow.alpha_composite(core)
        return glow

    def _build_timeline_layers(self) -> List[Image.Image]:
        buckets = int(CONFIG["timeline_buckets"])
        layers: List[Image.Image] = []
        for bucket in tqdm(range(1, buckets + 1), desc="Building lightning timeline layers", leave=False):
            fraction = bucket / buckets
            index = int(np.searchsorted(self.event_fractions, fraction, side="right"))
            layers.append(self._make_flash_layer(self.flashes[:index], 0.0, alpha_scale=0.82))
        return layers

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, COLORS["dark"] + (255,))
        draw = ImageDraw.Draw(image)
        for particle in self.particles:
            x = (particle["x"] + math.sin(t * 0.19 + particle["phase"]) * 11.0) % OUT_W
            y = (particle["y"] + t * particle["speed"] * 0.14) % OUT_H
            alpha = int(particle["a"] * (0.45 + 0.55 * math.sin(t * 0.9 + particle["phase"]) ** 2))
            radius = particle["r"]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(190, 220, 255, alpha))
        return image

    def compose_map(self, key: str, flash_layer: Optional[Image.Image] = None) -> Image.Image:
        layer = self.base_maps[key].copy()
        if flash_layer is not None:
            layer.alpha_composite(flash_layer)
        return layer

    def draw_flash_pulse(
        self,
        image: Image.Image,
        point: FlashPoint,
        center_lon: float,
        age: float,
        strength: float = 1.0,
    ):
        age = clamp(age)
        x, y = self.project(point.longitude, point.latitude, center_lon)
        radius = marker_radius(point.intensity)
        color = intensity_color(point.intensity, 255)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        flash = (1.0 - age) ** 2 * strength
        glow_r = radius * (2.0 + age * 5.5)
        draw.ellipse((x - glow_r, y - glow_r, x + glow_r, y + glow_r), fill=color[:3] + (int(135 * flash),))
        ring_r = radius * (1.4 + age * 6.0)
        draw.ellipse(
            (x - ring_r, y - ring_r, x + ring_r, y + ring_r),
            outline=color[:3] + (int(190 * (1.0 - age) * strength),),
            width=max(1, int(2 * OUT_W / 1080.0)),
        )
        core_r = radius * (0.85 + flash * 0.9)
        draw.ellipse((x - core_r, y - core_r, x + core_r, y + core_r), fill=color[:3] + (int(245 * strength),))
        overlay = overlay.filter(ImageFilter.GaussianBlur(1 if QUICK_MODE else 2))
        image.alpha_composite(overlay)

    def draw_lightning_bolt(self, image: Image.Image, x: float, top: float, bottom: float, seed: int, alpha: int = 240):
        rng = np.random.default_rng(seed)
        points: List[Tuple[float, float]] = [(x, top)]
        segments = 11 if QUICK_MODE else 17
        for i in range(1, segments):
            u = i / segments
            xx = x + float(rng.normal(0.0, OUT_W * 0.018)) * (0.35 + u)
            yy = lerp(top, bottom, u)
            points.append((xx, yy))
        points.append((x + float(rng.normal(0.0, OUT_W * 0.015)), bottom))

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.line(points, fill=COLORS["electric"] + (int(alpha * 0.55),), width=max(2, int(8 * OUT_W / 1080.0)))
        glow = glow.filter(ImageFilter.GaussianBlur(8 if not QUICK_MODE else 4))
        image.alpha_composite(glow)
        draw = ImageDraw.Draw(image)
        draw.line(points, fill=COLORS["white"] + (alpha,), width=max(1, int(3 * OUT_W / 1080.0)))

    def weighted_count_at_fraction(self, fraction: float) -> float:
        if not len(self.flashes):
            return 0.0
        idx = int(np.searchsorted(self.event_fractions, clamp(fraction), side="right"))
        if idx <= 0:
            return 0.0
        value = float(self.cumulative_weights[min(idx - 1, len(self.cumulative_weights) - 1)])
        if self.summary["data_source"] == "nasa_climatology_inspired_model":
            return self.total_weighted * clamp(fraction)
        return value

    def draw_ignition(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        progress = shot_progress(t, shot)
        map_layer = zoom_and_shift(self.compose_map("world"), 1.035 + 0.025 * progress, 0.0, -8.0 * progress)
        alpha_composite_with_opacity(image, map_layer, smoothstep(progress * 1.8))

        if 0.20 < progress < 0.96:
            flash_phase = (progress * 5.0) % 1.0
            if flash_phase < 0.18:
                overlay = Image.new("RGBA", OUT_SIZE, (255, 255, 255, int(55 * (1 - flash_phase / 0.18))))
                image.alpha_composite(overlay)
        if progress > 0.18:
            self.draw_lightning_bolt(
                image,
                OUT_W * 0.50,
                OUT_H * 0.23,
                OUT_H * 0.62,
                seed=int(progress * 100) + 77,
                alpha=int(245 * smoothstep((progress - 0.18) / 0.20)),
            )

        draw_text(
            image,
            "EARTH FLASHES ~44 TIMES / SECOND",
            (OUT_W // 2, int(OUT_H * 0.73)),
            size=15 if QUICK_MODE else 31,
            fill=COLORS["white"] + (238,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            "AVERAGE TOTAL LIGHTNING // NASA OTD STUDY",
            (OUT_W // 2, int(OUT_H * 0.775)),
            size=8 if QUICK_MODE else 16,
            fill=COLORS["muted"] + (225,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_global_sweep(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        progress = smootherstep(shot_progress(t, shot))
        bucket_count = len(self.timeline_layers)
        bucket = min(int(progress * bucket_count), bucket_count - 1)
        map_layer = self.compose_map("world", self.timeline_layers[bucket])
        map_layer = zoom_and_shift(map_layer, 1.02 + 0.035 * math.sin(progress * math.pi), math.sin(t * 0.22) * 7.0, -5.0)
        image.alpha_composite(map_layer)

        window = 0.013 if QUICK_MODE else 0.0065
        lo = max(0, int(np.searchsorted(self.event_fractions, progress - window, side="left")))
        hi = min(len(self.flashes), int(np.searchsorted(self.event_fractions, progress + 0.002, side="right")))
        active = self.flashes[lo:hi]
        step = max(1, len(active) // (22 if QUICK_MODE else 62))
        for point in active[::step]:
            age = clamp((progress - point.time_fraction) / max(window, 1e-6))
            if progress >= point.time_fraction:
                self.draw_flash_pulse(image, point, 0.0, age, strength=0.86)

        represented = self.weighted_count_at_fraction(progress)
        represented_time = WINDOW_START + (WINDOW_END - WINDOW_START) * progress
        left = 28 if QUICK_MODE else 56
        top = int(OUT_H * 0.665)
        right = OUT_W - left
        bottom = top + (100 if QUICK_MODE else 198)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rounded_rectangle((left, top, right, bottom), radius=14 if QUICK_MODE else 28, fill=(2, 6, 15, 178), outline=COLORS["grid"] + (72,), width=1)
        image.alpha_composite(panel)
        draw_text(
            image,
            represented_time.strftime("%H:%M:%S UTC"),
            (left + (16 if QUICK_MODE else 30), top + (15 if QUICK_MODE else 28)),
            size=12 if QUICK_MODE else 24,
            fill=COLORS["muted"] + (230,),
            bold=True,
            condensed=True,
            stroke=1,
        )
        draw_text(
            image,
            f"{int(represented):,}",
            (left + (16 if QUICK_MODE else 30), top + (43 if QUICK_MODE else 83)),
            size=28 if QUICK_MODE else 56,
            fill=COLORS["white"] + (255,),
            bold=True,
            condensed=True,
            stroke=1,
        )
        draw_text(
            image,
            "FLASHES REPRESENTED",
            (right - (16 if QUICK_MODE else 30), top + (56 if QUICK_MODE else 110)),
            size=10 if QUICK_MODE else 20,
            fill=COLORS["electric"] + (235,),
            bold=True,
            condensed=True,
            anchor="ra",
            stroke=1,
        )

    def draw_hotspots(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        progress = shot_progress(t, shot)
        map_layer = self.compose_map("world", self.all_layers["world"])
        map_layer = zoom_and_shift(map_layer, 1.04, math.sin(t * 0.18) * 5.0, -6.0)
        alpha_composite_with_opacity(image, map_layer, 0.92)

        labels = [
            ("LAKE MARACAIBO", -71.65, 9.75),
            ("CONGO BASIN", 27.0, -1.5),
            ("EAST AFRICAN RIFT", 29.0, -2.0),
            ("NE INDIA", 91.0, 26.0),
            ("INDONESIA", 115.0, -3.0),
        ]
        reveal = int(math.ceil(progress * len(labels)))
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for idx, (name, lon, lat) in enumerate(labels[:reveal]):
            x, y = self.project(lon, lat, 0.0)
            r = (8 if QUICK_MODE else 16) + (idx % 2) * (2 if QUICK_MODE else 4)
            od.ellipse((x - r, y - r, x + r, y + r), outline=COLORS["gold"] + (210,), width=max(1, int(2 * OUT_W / 1080.0)))
            line_end_x = x + (48 if QUICK_MODE else 92) * (-1 if x > OUT_W * 0.72 else 1)
            od.line((x, y, line_end_x, y - (18 if QUICK_MODE else 34)), fill=COLORS["gold"] + (170,), width=1)
            anchor = "ra" if line_end_x < x else "la"
            tx = line_end_x - (4 if anchor == "ra" else -4)
            draw_text(
                overlay,
                name,
                (int(tx), int(y - (20 if QUICK_MODE else 38))),
                size=7 if QUICK_MODE else 14,
                fill=COLORS["white"] + (238,),
                bold=True,
                condensed=True,
                anchor=anchor,
                stroke=1,
            )
        image.alpha_composite(overlay)

        draw_text(
            image,
            "LIGHTNING HAS FAVORITE PLACES",
            (OUT_W // 2, int(OUT_H * 0.70)),
            size=17 if QUICK_MODE else 34,
            fill=COLORS["white"] + (248,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            "TROPICAL LAND + MOUNTAINS + LAKES + CONVERGENCE",
            (OUT_W // 2, int(OUT_H * 0.75)),
            size=8 if QUICK_MODE else 16,
            fill=COLORS["electric"] + (230,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_land_ocean(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        progress = smoothstep(shot_progress(t, shot))
        map_layer = self.compose_map("world", self.all_layers["world"])
        map_layer = zoom_and_shift(map_layer, 1.03 + 0.025 * progress, 0.0, -4.0)
        alpha_composite_with_opacity(image, map_layer, 0.78)

        y = int(OUT_H * 0.70)
        x0 = int(OUT_W * 0.13)
        x1 = int(OUT_W * 0.87)
        bar_h = 18 if QUICK_MODE else 36
        # The 82/18 split comes from the classic OTD global distribution paper.
        land_share = 0.82
        fill_x = int(lerp(x0, x1, land_share * progress))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((x0, y, x1, y + bar_h), radius=bar_h // 2, fill=(8, 17, 28, 220), outline=COLORS["grid"] + (95,), width=1)
        if fill_x > x0:
            draw.rounded_rectangle((x0, y, fill_x, y + bar_h), radius=bar_h // 2, fill=COLORS["electric"] + (220,))

        draw_text(
            image,
            "~82% OVER LAND",
            (x0, y - (16 if QUICK_MODE else 31)),
            size=14 if QUICK_MODE else 28,
            fill=COLORS["white"] + (248,),
            bold=True,
            condensed=True,
            stroke=1,
        )
        draw_text(
            image,
            "~18% OVER OCEAN",
            (x1, y - (16 if QUICK_MODE else 31)),
            size=12 if QUICK_MODE else 24,
            fill=COLORS["muted"] + (230,),
            bold=True,
            condensed=True,
            anchor="ra",
            stroke=1,
        )
        draw_text(
            image,
            "CLASSIC OTD GLOBAL DISTRIBUTION",
            (OUT_W // 2, y + (42 if QUICK_MODE else 82)),
            size=8 if QUICK_MODE else 16,
            fill=COLORS["muted"] + (210,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_maracaibo(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        progress = shot_progress(t, shot)
        map_layer = self.compose_map("americas", self.all_layers["americas"])
        # Pan so northern South America dominates the middle of frame.
        map_layer = zoom_and_shift(map_layer, 1.30 + 0.10 * smoothstep(progress), -OUT_W * 0.12, OUT_H * 0.03)
        alpha_composite_with_opacity(image, map_layer, 0.94)

        x, y = self.project(-71.65, 9.75, -72.0)
        # Apply approximate same zoom shift visually with a simpler dramatic target ring.
        target_x = int(OUT_W * 0.50)
        target_y = int(OUT_H * 0.43)
        pulse = 0.5 + 0.5 * math.sin(t * 5.0)
        r = (28 if QUICK_MODE else 56) + pulse * (9 if QUICK_MODE else 18)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.ellipse((target_x - r, target_y - r, target_x + r, target_y + r), outline=COLORS["gold"] + (220,), width=max(1, int(3 * OUT_W / 1080.0)))
        glow_r = r * 1.25
        od.ellipse((target_x - glow_r, target_y - glow_r, target_x + glow_r, target_y + glow_r), outline=COLORS["electric"] + (70,), width=max(1, int(8 * OUT_W / 1080.0)))
        image.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(1 if QUICK_MODE else 2)))

        if progress > 0.2:
            self.draw_lightning_bolt(image, target_x, OUT_H * 0.21, target_y, seed=1900 + int(t * 6), alpha=245)

        draw_text(
            image,
            "LAKE MARACAIBO",
            (OUT_W // 2, int(OUT_H * 0.69)),
            size=22 if QUICK_MODE else 44,
            fill=COLORS["white"] + (255,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=2,
        )
        draw_text(
            image,
            "VENEZUELA // SATELLITE LIGHTNING HOTSPOT",
            (OUT_W // 2, int(OUT_H * 0.75)),
            size=8 if QUICK_MODE else 17,
            fill=COLORS["gold"] + (235,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_all_flashes(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        progress = smoothstep(shot_progress(t, shot))
        map_layer = self.compose_map("world", self.all_layers["world"])
        map_layer = zoom_and_shift(map_layer, 1.09 - 0.05 * progress, 0.0, -8.0 * progress)
        alpha_composite_with_opacity(image, map_layer, 0.82 + 0.18 * progress)

        total = int(round(self.total_weighted))
        draw_text(
            image,
            f"{total:,}",
            (OUT_W // 2, int(OUT_H * 0.66)),
            size=57 if QUICK_MODE else 116,
            fill=COLORS["white"] + (255,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=2,
        )
        draw_text(
            image,
            "LIGHTNING FLASHES REPRESENTED",
            (OUT_W // 2, int(OUT_H * 0.745)),
            size=15 if QUICK_MODE else 30,
            fill=COLORS["electric"] + (240,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            f"{WINDOW_LABEL} // ~{GLOBAL_FLASH_RATE:.0f} FLASHES / SECOND",
            (OUT_W // 2, int(OUT_H * 0.79)),
            size=8 if QUICK_MODE else 16,
            fill=COLORS["muted"] + (225,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_top_titles(self, image: Image.Image, t: float, name: str):
        intro = clamp(t / (1.0 if QUICK_MODE else 1.8))
        alpha = int(245 * smoothstep(intro))
        draw_text(
            image,
            CONFIG["title"],
            (OUT_W // 2, int(OUT_H * 0.052)),
            size=22 if QUICK_MODE else 44,
            fill=COLORS["white"] + (alpha,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=2,
        )
        draw_text(
            image,
            CONFIG["title_2"],
            (OUT_W // 2, int(OUT_H * 0.094)),
            size=28 if QUICK_MODE else 56,
            fill=COLORS["electric"] + (alpha,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=2,
        )
        if name != "ignition":
            draw_text(
                image,
                CONFIG["subtitle"],
                (OUT_W // 2, int(OUT_H * 0.127)),
                size=7 if QUICK_MODE else 14,
                fill=COLORS["muted"] + (220,),
                bold=True,
                condensed=True,
                anchor="ma",
                stroke=1,
            )

    def draw_source_hud(self, image: Image.Image):
        source = self.summary["data_source"]
        if source == "nasa_climatology_inspired_model":
            text = "MODEL // NASA LIS/OTD-INSPIRED DISTRIBUTION // 44 FLASHES/S REFERENCE"
        else:
            text = "USER LIGHTNING CSV // RENDERED CHRONOLOGICALLY"
        draw_text(
            image,
            text,
            (OUT_W // 2, int(OUT_H * 0.845)),
            size=6 if QUICK_MODE else 12,
            fill=COLORS["muted"] + (205,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        left = int(OUT_W * 0.08)
        top = int(OUT_H * 0.865)
        right = int(OUT_W * 0.92)
        bottom = int(OUT_H * 0.972)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rounded_rectangle((left, top, right, bottom), radius=14 if QUICK_MODE else 28, fill=(1, 4, 10, 190), outline=COLORS["grid"] + (65,), width=1)
        image.alpha_composite(panel)
        draw_wrapped_text(
            image,
            text,
            (left + (14 if QUICK_MODE else 28), top + (11 if QUICK_MODE else 21)),
            max_width=(right - left) - (28 if QUICK_MODE else 56),
            size=8 if QUICK_MODE else 16,
            fill=COLORS["white"] + (240,),
            line_spacing=2 if QUICK_MODE else 4,
        )

    def draw_film_texture(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for particle in self.dust:
            pulse = 0.5 + 0.5 * math.sin(t * 1.6 + particle["phase"])
            if pulse < 0.60:
                continue
            x = (particle["x"] + t * particle["speed"] * 0.40) % OUT_W
            y = (particle["y"] + math.sin(t * 0.8 + particle["phase"]) * 5.0) % OUT_H
            length = (5 if QUICK_MODE else 11) + particle["r"] * 5
            draw.line((x, y, x + length, y), fill=COLORS["electric"] + (int(16 * pulse),), width=1)
        offset = int((t * 47) % 8)
        for y in range(offset, OUT_H, 8):
            draw.line((0, y, OUT_W, y), fill=(120, 180, 230, 8), width=1)
        scan_y = int((t * 155) % (OUT_H + 180)) - 90
        draw.rectangle((0, scan_y, OUT_W, scan_y + (32 if QUICK_MODE else 64)), fill=(90, 220, 255, 5))
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        name = str(shot["name"])
        image = self.background(t)

        if name == "ignition":
            self.draw_ignition(image, t, shot)
        elif name == "global_sweep":
            self.draw_global_sweep(image, t, shot)
        elif name == "hotspots":
            self.draw_hotspots(image, t, shot)
        elif name == "land_ocean":
            self.draw_land_ocean(image, t, shot)
        elif name == "maracaibo":
            self.draw_maracaibo(image, t, shot)
        else:
            self.draw_all_flashes(image, t, shot)

        self.draw_top_titles(image, t, name)
        self.draw_source_hud(image)
        self.draw_caption(image, t)
        self.draw_film_texture(image, t)

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        graded = apply_grade(rgb).astype(np.float32)
        graded *= VIGNETTE[:, :, None]

        # Subtle deterministic grain changes with time but avoids heavy flicker in quick mode.
        rng = np.random.default_rng(1000 + int(round(t * int(CONFIG["fps"]))))
        grain = rng.normal(0.0, float(CONFIG["grain_strength"]), graded.shape[:2]).astype(np.float32)
        graded += grain[:, :, None]
        return np.clip(graded, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Soundtrack and rendering
# -----------------------------------------------------------------------------

def gaussian_envelope(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((times - center) / max(width, 1e-6)) ** 2)


def generate_ambient_soundtrack(path: Path) -> Path:
    sample_rate = int(CONFIG["soundtrack_sample_rate"])
    duration = float(CONFIG["duration_s"])
    count = int(round(sample_rate * duration))
    times = np.arange(count, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(440044)

    audio = np.zeros(count, dtype=np.float64)
    # Low thunder bed.
    audio += 0.085 * np.sin(math.tau * 28.0 * times + 0.6 * np.sin(math.tau * 0.055 * times))
    audio += 0.050 * np.sin(math.tau * 41.0 * times + 1.3)
    audio += 0.022 * np.sin(math.tau * 67.0 * times + 0.4 * np.sin(math.tau * 0.11 * times))

    controls = rng.normal(0.0, 1.0, max(10, int(duration * 5)))
    slow_noise = np.interp(times, np.linspace(0.0, duration, len(controls)), controls)
    audio += 0.025 * slow_noise

    # Electric crackles during the global sweep.
    sweep = next(shot for shot in SHOT_PLAN if shot["name"] == "global_sweep")
    crackle_count = 18 if QUICK_MODE else 54
    for center in np.linspace(float(sweep["start"]) + 0.15, float(sweep["end"]) - 0.15, crackle_count):
        center += float(rng.uniform(-0.10, 0.10))
        env = gaussian_envelope(times, center, 0.018 if QUICK_MODE else 0.025)
        burst = rng.normal(0.0, 1.0, count)
        audio += 0.025 * env * burst

    # Thunder swells at shot transitions.
    for shot in SHOT_PLAN[1:]:
        center = float(shot["start"]) + 0.25
        env = gaussian_envelope(times, center, 0.55 if QUICK_MODE else 0.9)
        rumble = np.sin(math.tau * (18.0 + 4.0 * np.sin(math.tau * 0.05 * times)) * times)
        audio += 0.085 * env * rumble

    # Maracaibo crack + tail.
    mar = next(shot for shot in SHOT_PLAN if shot["name"] == "maracaibo")
    for offset in [0.45, 2.1, 4.0] if not QUICK_MODE else [0.35, 1.0]:
        center = float(mar["start"]) + offset
        if center >= duration:
            continue
        env = np.exp(-np.maximum(times - center, 0.0) / 0.65) * (times >= center)
        click_env = gaussian_envelope(times, center, 0.012)
        audio += 0.13 * click_env * rng.normal(0.0, 1.0, count)
        audio += 0.045 * env * np.sin(math.tau * 24.0 * times)

    intro_x = np.clip(times / max(1.5, duration * 0.08), 0.0, 1.0)
    outro_x = np.clip((times - (duration - 1.3)) / 1.1, 0.0, 1.0)
    intro = intro_x * intro_x * (3.0 - 2.0 * intro_x)
    outro = 1.0 - outro_x * outro_x * (3.0 - 2.0 * outro_x)
    audio *= intro * outro

    peak = max(float(np.max(np.abs(audio))), 1e-9)
    audio = np.clip(audio / peak * 0.88, -1.0, 1.0)
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


def render_video(scene: LightningScene) -> Path:
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    raw_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_silent.mp4"
    final_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    audio_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_thunder.wav"
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
        for t in tqdm(times, desc="Rendering lightning short"):
            writer.append_data(scene.render_frame(float(t)))

    generate_ambient_soundtrack(audio_path)
    if mux_audio(raw_video, audio_path, final_video):
        print("Final video with audio:", final_video.resolve())
        return final_video
    shutil.copyfile(raw_video, final_video)
    print("ffmpeg audio mux unavailable; copied silent video to:", final_video.resolve())
    return final_video


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


