from __future__ import annotations

"""
THE ATMOSPHERE IS ALWAYS MOVING — cinematic YouTube Shorts renderer

Creates a vertical 1080x1920 data-driven short that reveals horizontal wind at
four pressure levels: 1000, 700, 500, and 250 hPa. The intent is to show that
"the wind" is not one layer: the atmosphere is moving differently from near the
surface through the mid-troposphere and into the upper-level jet-stream region.

The production pattern mirrors the supplied Shorts renderers:
- archived public data first
- cached data second
- deterministic synthetic fixture third
- quick-preview mode
- preview PNGs
- CSV / JSON data products
- SRT captions
- generated ambient soundtrack
- final MP4 when ffmpeg is available

Pressure levels shown:
- 1000 hPa : near-surface / lower troposphere
- 700 hPa  : lower-mid troposphere (~3 km guide)
- 500 hPa  : mid troposphere (~5.5 km guide)
- 250 hPa  : upper troposphere / jet-stream region (~10–11 km guide)

Pressure surfaces rise and fall with weather, temperature, and terrain, so the
altitude labels are intentionally approximate guides rather than fixed heights.
The moving tracer dots follow display-space integrations through the daily-mean
wind field. They are cinematic flow cues, not literal air-parcel trajectories.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm scipy

Quick preview render
--------------------
    ATMOSPHERE_QUICK=1 python the_atmosphere_is_always_moving.py

Force offline fixture mode
--------------------------
    ATMOSPHERE_OFFLINE=1 python the_atmosphere_is_always_moving.py

Choose another archived date
----------------------------
    ATMOSPHERE_DATE=2025-12-31 python the_atmosphere_is_always_moving.py

Outputs
-------
- final vertical MP4 with generated ambient audio when ffmpeg is available
- silent MP4 fallback
- SRT subtitle sidecar
- preview PNG frames
- CSV with per-pressure-level wind statistics
- JSON summary and source notes
- cached NOAA U/V NetCDF subsets

Sources
-------
- NOAA PSL NCEP/NCAR Reanalysis:
  https://psl.noaa.gov/data/reanalysis/reanalysis.shtml
- NOAA PSL THREDDS daily pressure-level files:
  https://psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis/Dailies/pressure/catalog.html
"""

import json
import math
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from scipy.io import netcdf_file
from tqdm.auto import tqdm

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("ATMOSPHERE_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("ATMOSPHERE_OFFLINE", "0") == "1"
DEFAULT_DATE = date(2026, 3, 17)
PRESSURE_LEVELS = [1000, 700, 500, 250]

LEVEL_META: Dict[int, Dict[str, Any]] = {
    1000: {
        "name": "LOWER ATMOSPHERE",
        "short": "NEAR SURFACE",
        "altitude": "~0–0.2 KM GUIDE",
        "accent": (69, 222, 186),
        "seed_lats": [-60, -45, -30, -15, 0, 15, 30, 45, 60],
    },
    700: {
        "name": "LOWER-MID TROPOSPHERE",
        "short": "~3 KM",
        "altitude": "~3 KM GUIDE",
        "accent": (62, 197, 242),
        "seed_lats": [-65, -50, -35, -20, 20, 35, 50, 65],
    },
    500: {
        "name": "MID TROPOSPHERE",
        "short": "~5.5 KM",
        "altitude": "~5.5 KM GUIDE",
        "accent": (160, 144, 255),
        "seed_lats": [-65, -50, -35, -20, 20, 35, 50, 65],
    },
    250: {
        "name": "UPPER TROPOSPHERE",
        "short": "JET LEVEL",
        "altitude": "~10–11 KM GUIDE",
        "accent": (255, 191, 78),
        "seed_lats": [-70, -60, -50, -40, -30, 30, 40, 50, 60, 70],
    },
}


def parse_target_date() -> date:
    raw = os.environ.get("ATMOSPHERE_DATE", "").strip()
    if not raw:
        return DEFAULT_DATE
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("ATMOSPHERE_DATE must be YYYY-MM-DD, e.g. 2025-12-31") from exc


TARGET_DATE = parse_target_date()

OUTPUT_ROOT = Path("the_atmosphere_is_always_moving_output")
DATA_ROOT = OUTPUT_ROOT / "data"
CACHE_ROOT = DATA_ROOT / "cache"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, CACHE_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12.0 if QUICK_MODE else 58.0,
    "title": "THE ATMOSPHERE IS ALWAYS MOVING",
    "subtitle": "GLOBAL DAILY-MEAN WIND // 1000 · 700 · 500 · 250 HPA",
    "output_basename": "the_atmosphere_is_always_moving",
    "map_margin_x": 20 if QUICK_MODE else 40,
    "map_top": 140 if QUICK_MODE else 280,
    "map_bottom": 675 if QUICK_MODE else 1350,
    "stream_seed_spacing_lon": 36 if QUICK_MODE else 22,
    "stream_step_count": 10 if QUICK_MODE else 18,
    "tracer_count_per_level": 26 if QUICK_MODE else 72,
    "star_count": 80 if QUICK_MODE else 190,
    "dust_count": 40 if QUICK_MODE else 110,
    "grain_strength": 3.0,
    "contrast": 1.10,
    "saturation": 1.08,
    "vignette": 0.34,
    "sample_rate": 22050 if QUICK_MODE else 44100,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0

COLORS = {

}

FULL_SHOT_PLAN = [

]

FULL_CAPTIONS = [

]

if QUICK_MODE:
    k = float(CONFIG["duration_s"]) / 58.0
    SHOT_PLAN = [{**shot, "start": shot["start"] * k, "end": shot["end"] * k} for shot in FULL_SHOT_PLAN]
    CAPTIONS = [(a * k, b * k, text) for a, b, text in FULL_CAPTIONS]
else:
    SHOT_PLAN = FULL_SHOT_PLAN
    CAPTIONS = FULL_CAPTIONS


# -----------------------------------------------------------------------------
# Data model and helpers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AtmosphereDataset:
    when: date
    lat: np.ndarray
    lon: np.ndarray
    u: Dict[int, np.ndarray]
    v: Dict[int, np.ndarray]
    source: str
    notes: List[str]

    def speed(self, level: int) -> np.ndarray:
        return np.hypot(self.u[level], self.v[level])


@dataclass(frozen=True)
class LevelStats:
    level_hpa: int
    max_ms: float
    max_lat: float
    max_lon: float
    mean_ms: float
    p90_ms: float
    area_gt_20_fraction: float


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
        if float(shot["start"]) <= t < float(shot["end"]):
            return shot
    return SHOT_PLAN[-1]


def shot_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - float(shot["start"])) / max(float(shot["end"] - shot["start"]), 1e-9))


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
    for index, (start, end, text) in enumerate(captions, 1):
        lines.extend([str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def get_font(size: int, bold: bool = False, condensed: bool = False):
    candidates: List[str] = []
    if condensed and bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
            "DejaVuSansCondensed-Bold.ttf",
        ]
    if condensed:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
            "DejaVuSansCondensed.ttf",
        ]
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
        stroke_fill=(0, 0, 0, min(235, fill[3])),
    )


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.75, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# NOAA PSL data loading
# -----------------------------------------------------------------------------

def ncss_url(var: str, year: int, when: date, level_hpa: int) -> str:
    base = (
        "https://psl.noaa.gov/thredds/ncss/grid/"
        f"Datasets/ncep.reanalysis/Dailies/pressure/{var}.{year}.nc"
    )
    params = {
        "var": var,
        "north": "90",
        "south": "-90",
        "west": "0",
        "east": "357.5",
        "horizStride": "1",
        "time_start": f"{when.isoformat()}T00:00:00Z",
        "time_end": f"{when.isoformat()}T00:00:00Z",
        "timeStride": "1",
        "vertCoord": str(int(level_hpa)),
        "accept": "netcdf3",
    }
    return base + "?" + urllib.parse.urlencode(params)


def request_bytes(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; AtmosphereMovingShort/1.0; educational visualization)",
            "Accept": "application/x-netcdf,application/octet-stream,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def load_netcdf3(path: Path, var_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with netcdf_file(str(path), "r", mmap=False) as nc:
        if var_name not in nc.variables:
            raise RuntimeError(f"{path.name} is missing variable {var_name}")
        lat_name = "lat" if "lat" in nc.variables else "latitude"
        lon_name = "lon" if "lon" in nc.variables else "longitude"
        lat = np.asarray(nc.variables[lat_name][:], dtype=np.float32).copy()
        lon = np.asarray(nc.variables[lon_name][:], dtype=np.float32).copy()
        data = np.asarray(nc.variables[var_name][:], dtype=np.float32).copy()
        fill = getattr(nc.variables[var_name], "_FillValue", None)
        missing = getattr(nc.variables[var_name], "missing_value", None)

    data = np.squeeze(data)
    if data.ndim == 3:
        data = data[0]
    if data.ndim != 2:
        raise RuntimeError(f"Expected {var_name} to resolve to [lat,lon], got shape {data.shape}")
    for bad in (fill, missing):
        if bad is not None:
            try:
                data[np.isclose(data, float(bad))] = np.nan
            except Exception:
                pass
    data[np.abs(data) > 500] = np.nan
    return data.astype(np.float32), lat, lon


def standardize_grid(
    data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_wrapped = ((lon.astype(float) + 180.0) % 360.0) - 180.0
    lon_order = np.argsort(lon_wrapped)
    lon_sorted = lon_wrapped[lon_order].astype(np.float32)
    data = data[..., lon_order]

    lat = lat.astype(np.float32)
    if len(lat) >= 2 and lat[0] > lat[-1]:
        lat = lat[::-1].copy()
        data = data[::-1, :]
    return data.astype(np.float32), lat, lon_sorted


def download_level(var: str, when: date, level_hpa: int, notes: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = CACHE_ROOT / f"{var}_{level_hpa}hpa_{when.isoformat()}.nc"
    url = ncss_url(var, when.year, when, level_hpa)
    try:
        payload = request_bytes(url, timeout=120)
        if len(payload) < 4_000:
            raise RuntimeError(f"NOAA subset response was unexpectedly small ({len(payload)} bytes)")
        path.write_bytes(payload)
        notes.append(f"Downloaded NOAA PSL {var} at {level_hpa} hPa for {when.isoformat()}")
    except Exception as exc:
        notes.append(f"Live NOAA {var} {level_hpa} hPa download failed: {exc}")
        if not (path.exists() and path.stat().st_size > 4_000):
            raise RuntimeError(f"No live or cached {var} data available for {level_hpa} hPa") from exc
        notes.append(f"Using cached NOAA {var} {level_hpa} hPa subset")

    data, lat, lon = load_netcdf3(path, var)
    return standardize_grid(data, lat, lon)


def download_atmosphere(when: date) -> AtmosphereDataset:
    notes: List[str] = []
    u: Dict[int, np.ndarray] = {}
    v: Dict[int, np.ndarray] = {}
    reference_lat: Optional[np.ndarray] = None
    reference_lon: Optional[np.ndarray] = None

    for level in PRESSURE_LEVELS:
        uu, lat_u, lon_u = download_level("uwnd", when, level, notes)
        vv, lat_v, lon_v = download_level("vwnd", when, level, notes)
        if uu.shape != vv.shape:
            raise RuntimeError(f"U/V shapes differ at {level} hPa: {uu.shape} vs {vv.shape}")
        if not (np.allclose(lat_u, lat_v) and np.allclose(lon_u, lon_v)):
            raise RuntimeError(f"U/V coordinate grids differ at {level} hPa")
        if reference_lat is None:
            reference_lat, reference_lon = lat_u, lon_u
        elif not (np.allclose(reference_lat, lat_u) and np.allclose(reference_lon, lon_u)):
            raise RuntimeError(f"Coordinate grid differs between pressure levels at {level} hPa")
        u[level] = uu
        v[level] = vv

    assert reference_lat is not None and reference_lon is not None
    notes.append("Pressure-to-altitude labels are approximate guides; pressure surfaces are not fixed geometric heights")
    notes.append("Animated tracers are display-space flow cues through daily-mean wind, not parcel trajectories")
    return AtmosphereDataset(
        when=when,
        lat=reference_lat,
        lon=reference_lon,
        u=u,
        v=v,
        source="noaa_psl_ncep_ncar_reanalysis1_daily_pressure_wind",
        notes=notes,
    )


def make_synthetic_atmosphere(when: date) -> AtmosphereDataset:
    lat = np.arange(-90.0, 90.01, 2.5, dtype=np.float32)
    lon = np.arange(-180.0, 180.0, 2.5, dtype=np.float32)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    u: Dict[int, np.ndarray] = {}
    v: Dict[int, np.ndarray] = {}

    # Deterministic large-scale flow fields. Each pressure level is intentionally
    # different so offline previews communicate atmospheric depth rather than
    # recycling the exact same pattern four times.
    for level in PRESSURE_LEVELS:
        phase = {1000: 0.1, 700: 0.8, 500: 1.45, 250: 2.1}[level]
        if level == 1000:
            trades = 10.0 * np.exp(-0.5 * (lat2d / 19.0) ** 2)
            westerlies = 13.0 * (
                np.exp(-0.5 * ((lat2d - 45.0) / 15.0) ** 2)
                + np.exp(-0.5 * ((lat2d + 45.0) / 15.0) ** 2)
            )
            uu = -trades + westerlies + 5.0 * np.sin(np.deg2rad(lon2d * 1.3) + phase) * np.cos(np.deg2rad(lat2d))
            vv = 5.0 * np.sin(np.deg2rad(lat2d * 3.0) - np.deg2rad(lon2d * 0.7) + phase)
        elif level == 700:
            belt = 21.0 * (
                np.exp(-0.5 * ((lat2d - 42.0) / 16.0) ** 2)
                + np.exp(-0.5 * ((lat2d + 42.0) / 16.0) ** 2)
            )
            uu = 5.0 + belt + 8.0 * np.sin(np.deg2rad(lon2d * 1.6) + phase) * np.cos(np.deg2rad(lat2d * 0.9))
            vv = 9.0 * np.cos(np.deg2rad(lon2d * 1.3) - phase) * np.sin(np.deg2rad(lat2d * 1.5))
        elif level == 500:
            north_axis = 48.0 + 9.0 * np.sin(np.deg2rad(lon2d * 1.3) + phase)
            south_axis = -48.0 + 8.0 * np.sin(np.deg2rad(lon2d * 1.2) - phase)
            waves = np.exp(-0.5 * ((lat2d - north_axis) / 13.0) ** 2) + np.exp(-0.5 * ((lat2d - south_axis) / 13.0) ** 2)
            uu = 8.0 + 31.0 * waves
            vv = 14.0 * np.sin(np.deg2rad(lon2d * 1.5) + phase) * np.sin(np.deg2rad(lat2d * 1.7))
        else:
            north_axis = 50.0 + 10.0 * np.sin(np.deg2rad(lon2d * 1.5) + phase) + 3.0 * np.sin(np.deg2rad(lon2d * 3.0) - phase)
            south_axis = -50.0 + 9.0 * np.sin(np.deg2rad(lon2d * 1.3) - phase)
            north_jet = np.exp(-0.5 * ((lat2d - north_axis) / 8.0) ** 2)
            south_jet = np.exp(-0.5 * ((lat2d - south_axis) / 8.0) ** 2)
            uu = 8.0 + 58.0 * north_jet + 52.0 * south_jet
            uu += 10.0 * north_jet * np.sin(np.deg2rad(lon2d * 2.0) + phase)
            vv = 15.0 * north_jet * np.cos(np.deg2rad(lon2d * 1.5) + phase)
            vv -= 14.0 * south_jet * np.cos(np.deg2rad(lon2d * 1.3) - phase)
        u[level] = uu.astype(np.float32)
        v[level] = vv.astype(np.float32)

    return AtmosphereDataset(
        when=when,
        lat=lat,
        lon=lon,
        u=u,
        v=v,
        source="synthetic_procedural_fixture",
        notes=[
            "Using deterministic synthetic multi-level wind fixture; not observational data",
            "Pressure-to-altitude labels are approximate guides",
            "Animated tracers are cinematic flow cues, not parcel trajectories",
        ],
    )


def load_atmosphere() -> AtmosphereDataset:
    if OFFLINE_MODE:
        return make_synthetic_atmosphere(TARGET_DATE)
    try:
        return download_atmosphere(TARGET_DATE)
    except Exception as exc:
        fixture = make_synthetic_atmosphere(TARGET_DATE)
        return AtmosphereDataset(
            when=fixture.when,
            lat=fixture.lat,
            lon=fixture.lon,
            u=fixture.u,
            v=fixture.v,
            source=fixture.source,
            notes=[f"Archived NOAA load failed: {exc}"] + fixture.notes,
        )


# -----------------------------------------------------------------------------
# Statistics and data products
# -----------------------------------------------------------------------------

def compute_level_stats(data: AtmosphereDataset) -> List[LevelStats]:
    stats: List[LevelStats] = []
    for level in PRESSURE_LEVELS:
        speed = data.speed(level)
        safe = np.where(np.isfinite(speed), speed, -np.inf)
        flat = int(np.argmax(safe))
        iy, ix = np.unravel_index(flat, speed.shape)
        stats.append(
            LevelStats(
                level_hpa=level,
                max_ms=float(np.nanmax(speed)),
                max_lat=float(data.lat[iy]),
                max_lon=float(data.lon[ix]),
                mean_ms=float(np.nanmean(speed)),
                p90_ms=float(np.nanpercentile(speed, 90.0)),
                area_gt_20_fraction=float(np.nanmean(speed >= 20.0)),
            )
        )
    return stats


def save_data_products(data: AtmosphereDataset, stats: Sequence[LevelStats]) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "atmosphere_level_stats.csv"
    summary_path = DATA_ROOT / "atmosphere_summary.json"
    rows = []
    for s in stats:
        meta = LEVEL_META[s.level_hpa]
        rows.append(
            {
                "date": data.when.isoformat(),
                "pressure_hpa": s.level_hpa,
                "layer_name": meta["name"],
                "approx_altitude_guide": meta["altitude"],
                "global_max_ms": s.max_ms,
                "global_max_kmh": s.max_ms * 3.6,
                "global_max_lat": s.max_lat,
                "global_max_lon": s.max_lon,
                "global_mean_ms": s.mean_ms,
                "p90_ms": s.p90_ms,
                "grid_fraction_at_or_above_20_ms": s.area_gt_20_fraction,
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    summary = {
        "title": CONFIG["title"],
        "date": data.when.isoformat(),
        "pressure_levels_hpa": PRESSURE_LEVELS,
        "data_source": data.source,
        "source_notes": data.notes,
        "levels": rows,
        "source_urls": {
            "noaa_reanalysis": "https://psl.noaa.gov/data/reanalysis/reanalysis.shtml",
            "thredds_catalog": "https://psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis/Dailies/pressure/catalog.html",
        },
        "scientific_note": (
            "The fields are horizontal daily-mean winds on pressure surfaces. Pressure surfaces are not fixed heights, "
            "and the animated tracers are display-space flow cues rather than air-parcel trajectories."
        ),
        "fallback_warning": "synthetic_procedural_fixture is preview data, not observational data",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, summary_path


# -----------------------------------------------------------------------------
# Map geometry and color helpers
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


def speed_color_array(speed: np.ndarray, level: int) -> np.ndarray:
    accent = np.array(LEVEL_META[level]["accent"], dtype=float)
    dark = np.array([4, 16, 28], dtype=float)
    white = np.array([246, 244, 225], dtype=float)
    magenta = np.array(COLORS["magenta"], dtype=float)
    scale_max = {1000: 34.0, 700: 45.0, 500: 60.0, 250: 95.0}[level]
    x = np.nan_to_num(speed, nan=0.0, posinf=scale_max, neginf=0.0)
    n = np.clip(x / scale_max, 0.0, 1.0)
    rgb = np.empty(x.shape + (3,), dtype=np.float32)
    low = n <= 0.58
    t1 = np.where(low, n / 0.58, 0.0)[..., None]
    rgb[:] = dark
    rgb[low] = (dark * (1.0 - t1[low]) + accent * t1[low])
    high = ~low
    t2 = np.where(high, (n - 0.58) / 0.42, 0.0)[..., None]
    target = white if level != 250 else magenta
    rgb[high] = accent * (1.0 - t2[high]) + target * t2[high]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def speed_color(speed_ms: float, level: int, alpha: int = 255) -> Tuple[int, int, int, int]:
    arr = speed_color_array(np.array([[float(speed_ms)]], dtype=float), level)[0, 0]
    return int(arr[0]), int(arr[1]), int(arr[2]), int(alpha)


def unwrapped_relative_longitudes(polygon: Sequence[Tuple[float, float]]) -> List[float]:
    values: List[float] = []
    previous: Optional[float] = None
    for lon, _ in polygon:
        value = ((lon + 180.0) % 360.0) - 180.0
        if previous is not None:
            while value - previous > 180:
                value -= 360
            while value - previous < -180:
                value += 360
        values.append(value)
        previous = value
    return values


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class AtmosphereScene:
    def __init__(self, data: AtmosphereDataset, stats: Sequence[LevelStats]):
        self.data = data
        self.stats = {s.level_hpa: s for s in stats}
        self.particles = self._make_particles(int(CONFIG["star_count"]), 77)
        self.dust = self._make_particles(int(CONFIG["dust_count"]), 137)
        self.static_map = self._render_static_map()
        self.level_layers: Dict[int, Image.Image] = {}
        self.paths: Dict[int, List[List[Tuple[float, float]]]] = {}
        for level in tqdm(PRESSURE_LEVELS, desc="Building atmospheric layers", leave=False):
            self.paths[level] = self._build_stream_paths(level)
            self.level_layers[level] = self._render_level_layer(level)
        self.blended_layer = self._render_blended_layer()

    @staticmethod
    def _make_particles(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.4, 1.8 if QUICK_MODE else 2.2)),
                "a": float(rng.uniform(10, 52)),
                "phase": float(rng.uniform(0, math.tau)),
                "speed": float(rng.uniform(0.5, 5.5)),
            }
            for _ in range(count)
        ]

    def project(self, lon: float, lat: float) -> Tuple[float, float]:
        x0 = float(CONFIG["map_margin_x"])
        x1 = OUT_W - x0
        y0 = float(CONFIG["map_top"])
        y1 = float(CONFIG["map_bottom"])
        x = x0 + ((lon + 180.0) / 360.0) * (x1 - x0)
        y = y0 + (90.0 - float(np.clip(lat, -90.0, 90.0))) / 180.0 * (y1 - y0)
        return x, y

    def _render_static_map(self) -> Image.Image:
        arr = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
        top = np.array(COLORS["bg_top"], dtype=float)
        bottom = np.array(COLORS["bg_bottom"], dtype=float)
        for y in range(OUT_H):
            u = y / max(OUT_H - 1, 1)
            rgb = (top * (1.0 - u) + bottom * u).astype(np.uint8)
            arr[y, :, :3] = rgb
            arr[y, :, 3] = 255
        image = Image.fromarray(arr, "RGBA")
        draw = ImageDraw.Draw(image)
        x0, x1 = int(CONFIG["map_margin_x"]), OUT_W - int(CONFIG["map_margin_x"])
        y0, y1 = int(CONFIG["map_top"]), int(CONFIG["map_bottom"])

        for lat in range(-60, 61, 20):
            _, y = self.project(0, lat)
            draw.line((x0, y, x1, y), fill=COLORS["grid"] + (24,), width=1)
        for lon in range(-150, 181, 30):
            x, _ = self.project(lon, 0)
            draw.line((x, y0, x, y1), fill=COLORS["grid"] + (20,), width=1)

        for polygon in BUILTIN_LAND_POLYGONS:
            rel = unwrapped_relative_longitudes(polygon)
            lats = [lat for _, lat in polygon]
            for shift in (-360.0, 0.0, 360.0):
                pts = [self.project(relative + shift, lat) for relative, lat in zip(rel, lats)]
                if pts and max(p[0] for p in pts) >= x0 - 50 and min(p[0] for p in pts) <= x1 + 50:
                    draw.polygon(pts, fill=COLORS["land"] + (245,), outline=COLORS["land_edge"] + (95,))

        draw.rounded_rectangle((x0, y0, x1, y1), radius=max(10, int(26 * SCALE)), outline=COLORS["grid"] + (65,), width=1)
        _, eq_y = self.project(0, 0)
        draw.line((x0, eq_y, x1, eq_y), fill=COLORS["grid"] + (38,), width=1)
        return image

    def _interp_field(self, field: np.ndarray, lon: float, lat: float) -> float:
        lon = ((lon + 180.0) % 360.0) - 180.0
        lat = float(np.clip(lat, self.data.lat[0], self.data.lat[-1]))
        x = np.searchsorted(self.data.lon, lon) - 1
        y = np.searchsorted(self.data.lat, lat) - 1
        x = int(np.clip(x, 0, len(self.data.lon) - 2))
        y = int(np.clip(y, 0, len(self.data.lat) - 2))
        x1, x2 = float(self.data.lon[x]), float(self.data.lon[x + 1])
        y1, y2 = float(self.data.lat[y]), float(self.data.lat[y + 1])
        tx = 0.0 if x2 == x1 else (lon - x1) / (x2 - x1)
        ty = 0.0 if y2 == y1 else (lat - y1) / (y2 - y1)
        a = field[y, x] * (1 - tx) + field[y, x + 1] * tx
        b = field[y + 1, x] * (1 - tx) + field[y + 1, x + 1] * tx
        return float(a * (1 - ty) + b * ty)

    def _streamline(self, level: int, lon0: float, lat0: float) -> Tuple[List[Tuple[float, float]], List[float]]:
        u = self.data.u[level]
        v = self.data.v[level]
        points: List[Tuple[float, float]] = []
        speeds: List[float] = []
        lon, lat = float(lon0), float(lat0)
        steps = int(CONFIG["stream_step_count"])
        min_speed = {1000: 2.5, 700: 4.0, 500: 5.0, 250: 8.0}[level]
        lon_gain = {1000: 0.11, 700: 0.09, 500: 0.078, 250: 0.066}[level]
        lat_gain = {1000: 0.070, 700: 0.058, 500: 0.047, 250: 0.038}[level]
        if QUICK_MODE:
            lon_gain *= 1.25
            lat_gain *= 1.25

        for _ in range(steps):
            uu = self._interp_field(u, lon, lat)
            vv = self._interp_field(v, lon, lat)
            spd = math.hypot(uu, vv)
            if not np.isfinite(spd) or spd < min_speed:
                break
            points.append(self.project(lon, lat))
            speeds.append(spd)
            coslat = max(0.25, math.cos(math.radians(lat)))
            lon += uu * lon_gain / coslat
            lat += vv * lat_gain
            lon = ((lon + 180.0) % 360.0) - 180.0
            if abs(lat) > 83:
                break
        return points, speeds

    def _build_stream_paths(self, level: int) -> List[List[Tuple[float, float]]]:
        paths: List[List[Tuple[float, float]]] = []
        spacing = int(CONFIG["stream_seed_spacing_lon"])
        for lat0 in LEVEL_META[level]["seed_lats"]:
            for lon0 in range(-180, 180, spacing):
                pts, speeds = self._streamline(level, float(lon0), float(lat0))
                if len(pts) < 3:
                    continue
                if float(np.mean(speeds)) < {1000: 4, 700: 7, 500: 10, 250: 20}[level]:
                    continue
                # Remove paths with a major dateline chord from cinematic tracer use.
                clean: List[Tuple[float, float]] = [pts[0]]
                for p1, p2 in zip(pts[:-1], pts[1:]):
                    if abs(p2[0] - p1[0]) > (OUT_W - 2 * int(CONFIG["map_margin_x"])) * 0.5:
                        break
                    clean.append(p2)
                if len(clean) >= 3:
                    paths.append(clean)
        return paths

    def _render_level_layer(self, level: int) -> Image.Image:
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        x0, x1 = int(CONFIG["map_margin_x"]), OUT_W - int(CONFIG["map_margin_x"])
        y0, y1 = int(CONFIG["map_top"]), int(CONFIG["map_bottom"])
        map_w, map_h = x1 - x0, y1 - y0

        speed = self.data.speed(level)
        rgb = speed_color_array(np.flipud(speed), level)
        threshold = {1000: 2.0, 700: 4.0, 500: 6.0, 250: 12.0}[level]
        span = {1000: 24.0, 700: 34.0, 500: 45.0, 250: 70.0}[level]
        alpha = np.clip((np.flipud(speed) - threshold) / span * 165.0, 0, 165).astype(np.uint8)
        rgba = np.dstack([rgb, alpha])
        heat = Image.fromarray(rgba, "RGBA").resize((map_w, map_h), Image.Resampling.BILINEAR)
        heat = heat.filter(ImageFilter.GaussianBlur(max(1, int(3.2 * SCALE))))
        layer.alpha_composite(heat, (x0, y0))

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        core = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        cd = ImageDraw.Draw(core)
        accent = tuple(LEVEL_META[level]["accent"])

        for path in self.paths[level]:
            for p1, p2 in zip(path[:-1], path[1:]):
                gd.line((*p1, *p2), fill=accent + (48,), width=max(2, int(7 * SCALE)))
                cd.line((*p1, *p2), fill=accent + (160,), width=max(1, int(1.8 * SCALE)))

        glow = glow.filter(ImageFilter.GaussianBlur(max(1, int(5 * SCALE))))
        layer.alpha_composite(glow)
        layer.alpha_composite(core)
        return layer

    def _render_blended_layer(self) -> Image.Image:
        blend = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        for level, opacity in [(1000, 0.35), (700, 0.36), (500, 0.40), (250, 0.48)]:
            layer = self.level_layers[level].copy()
            a = layer.getchannel("A").point(lambda x, o=opacity: int(x * o))
            layer.putalpha(a)
            blend.alpha_composite(layer)
        return blend

    def background(self, t: float) -> Image.Image:
        image = self.static_map.copy()
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for p in self.particles:
            pulse = 0.45 + 0.55 * math.sin(t * 0.65 + p["phase"]) ** 2
            x = (p["x"] + t * p["speed"] * 0.15) % OUT_W
            y = p["y"]
            r = p["r"] * SCALE
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(190, 225, 237, int(p["a"] * pulse)))
        image.alpha_composite(overlay)
        return image

    def compose_map(self, layer: Image.Image, opacity: float = 1.0) -> Image.Image:
        result = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        copy = layer.copy()
        if opacity < 0.999:
            copy.putalpha(copy.getchannel("A").point(lambda x: int(x * opacity)))
        result.alpha_composite(copy)
        return result

    def draw_tracers(self, image: Image.Image, level: int, t: float, opacity: float = 1.0, size_boost: float = 1.0):
        paths = self.paths[level]
        if not paths:
            return
        count = min(int(CONFIG["tracer_count_per_level"]), len(paths))
        step = max(1, len(paths) // count)
        selected = paths[::step][:count]
        accent = tuple(LEVEL_META[level]["accent"])
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        core = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        cd = ImageDraw.Draw(core)
        speed_factor = {1000: 0.16, 700: 0.20, 500: 0.24, 250: 0.31}[level]
        for i, path in enumerate(selected):
            if len(path) < 2:
                continue
            phase = (t * speed_factor + i * 0.137 + level * 0.0007) % 1.0
            pos = phase * (len(path) - 1)
            j = min(int(pos), len(path) - 2)
            f = pos - j
            x = lerp(path[j][0], path[j + 1][0], f)
            y = lerp(path[j][1], path[j + 1][1], f)
            rr = max(1.0, (2.4 if QUICK_MODE else 4.4) * SCALE * size_boost)
            gd.ellipse((x - rr * 3.0, y - rr * 3.0, x + rr * 3.0, y + rr * 3.0), fill=accent + (int(58 * opacity),))
            cd.ellipse((x - rr, y - rr, x + rr, y + rr), fill=accent + (int(245 * opacity),))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(1, int(4.0 * SCALE)))))
        image.alpha_composite(core)

    def draw_opening(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smootherstep(shot_progress(t, shot))
        veil = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 90))
        image.alpha_composite(veil)

        # Four luminous atmospheric ribbons make the opening feel dimensional.
        ribbon = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        rd = ImageDraw.Draw(ribbon)
        center_y = OUT_H * 0.46
        for i, level in enumerate(PRESSURE_LEVELS):
            accent = tuple(LEVEL_META[level]["accent"])
            y = center_y + (i - 1.5) * 72 * SCALE
            amp = (28 + i * 7) * SCALE
            pts = []
            for x in np.linspace(-40 * SCALE, OUT_W + 40 * SCALE, 80):
                yy = y + amp * math.sin(x / (150 * SCALE + 1e-6) + t * (0.45 + i * 0.08) + i)
                pts.append((x, yy))
            rd.line(pts, fill=accent + (int(70 + 35 * p),), width=max(2, int((11 - i) * SCALE)))
        ribbon = ribbon.filter(ImageFilter.GaussianBlur(max(2, int(7 * SCALE))))
        image.alpha_composite(ribbon)

        alpha = int(255 * smoothstep(min(1.0, p * 1.5)))
        draw_text(image, "THE ATMOSPHERE", (OUT_W // 2, int(OUT_H * 0.28)), 36 if QUICK_MODE else 72,
                  COLORS["white"] + (alpha,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, "IS ALWAYS MOVING", (OUT_W // 2, int(OUT_H * 0.37)), 39 if QUICK_MODE else 78,
                  COLORS["white"] + (alpha,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, "1000 · 700 · 500 · 250 HPA", (OUT_W // 2, int(OUT_H * 0.59)), 13 if QUICK_MODE else 26,
                  COLORS["muted"] + (int(225 * p),), bold=True, condensed=True, anchor="ma", stroke=1)
        draw_text(image, "ONE PLANET // MANY MOVING LAYERS", (OUT_W // 2, int(OUT_H * 0.64)), 10 if QUICK_MODE else 20,
                  COLORS["cyan"] + (int(230 * p),), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_level_scene(self, image: Image.Image, t: float, shot: Dict[str, Any], level: int):
        p = smoothstep(shot_progress(t, shot))
        image.alpha_composite(self.compose_map(self.level_layers[level], opacity=0.78 + 0.20 * p))
        self.draw_tracers(image, level, t, opacity=0.95, size_boost=1.1)
        stat = self.stats[level]
        accent = tuple(LEVEL_META[level]["accent"])

        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        top = int(1425 * SCALE)
        pd.rounded_rectangle(
            (int(48 * SCALE), top, OUT_W - int(48 * SCALE), top + int(325 * SCALE)),
            radius=max(12, int(28 * SCALE)),
            fill=(2, 7, 14, 207),
            outline=accent + (90,),
            width=1,
        )
        image.alpha_composite(panel)
        draw_text(image, f"{level} HPA", (int(78 * SCALE), top + int(58 * SCALE)), 35 if QUICK_MODE else 70,
                  accent + (255,), bold=True, condensed=True, stroke=2)
        draw_text(image, LEVEL_META[level]["name"], (int(80 * SCALE), top + int(128 * SCALE)), 13 if QUICK_MODE else 26,
                  COLORS["white"] + (245,), bold=True, condensed=True, stroke=1)
        draw_text(image, LEVEL_META[level]["altitude"], (int(80 * SCALE), top + int(170 * SCALE)), 9 if QUICK_MODE else 18,
                  COLORS["muted"] + (225,), bold=True, condensed=True, stroke=1)
        draw_text(image, f"MAX {stat.max_ms:.0f} M/S  //  MEAN {stat.mean_ms:.0f} M/S", (int(80 * SCALE), top + int(238 * SCALE)), 13 if QUICK_MODE else 26,
                  speed_color(stat.max_ms, level, 250), bold=True, condensed=True, stroke=1)
        draw_text(image, data_date_label(self.data.when), (int(80 * SCALE), top + int(286 * SCALE)), 9 if QUICK_MODE else 18,
                  COLORS["muted"] + (210,), bold=True, condensed=True, stroke=1)

    def draw_stack(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        image.alpha_composite(Image.new("RGBA", OUT_SIZE, (0, 0, 0, 90)))
        draw_text(image, "THE SAME ATMOSPHERE", (OUT_W // 2, int(145 * SCALE)), 17 if QUICK_MODE else 34,
                  COLORS["muted"] + (240,), bold=True, condensed=True, anchor="ma", stroke=1)
        draw_text(image, "FOUR MOVING LEVELS", (OUT_W // 2, int(210 * SCALE)), 29 if QUICK_MODE else 58,
                  COLORS["white"] + (255,), bold=True, condensed=True, anchor="ma", stroke=2)

        source_box = (int(CONFIG["map_margin_x"]), int(CONFIG["map_top"]), OUT_W - int(CONFIG["map_margin_x"]), int(CONFIG["map_bottom"]))
        crop_w = source_box[2] - source_box[0]
        crop_h = source_box[3] - source_box[1]
        left = int(50 * SCALE)
        right = OUT_W - int(50 * SCALE)
        panel_w = right - left
        mini_h = int(260 * SCALE)
        gap = int(22 * SCALE)
        start_y = int(335 * SCALE)

        for i, level in enumerate(reversed(PRESSURE_LEVELS)):
            y = start_y + i * (mini_h + gap)
            base_crop = self.static_map.crop(source_box).resize((panel_w, mini_h), Image.Resampling.BILINEAR)
            layer_crop = self.level_layers[level].crop(source_box).resize((panel_w, mini_h), Image.Resampling.BILINEAR)
            panel = Image.new("RGBA", (panel_w, mini_h), (0, 0, 0, 255))
            panel.alpha_composite(base_crop)
            panel.alpha_composite(layer_crop)
            mask = Image.new("RGBA", (panel_w, mini_h), (0, 0, 0, 0))
            md = ImageDraw.Draw(mask)
            accent = tuple(LEVEL_META[level]["accent"])
            md.rounded_rectangle((1, 1, panel_w - 2, mini_h - 2), radius=max(10, int(20 * SCALE)), outline=accent + (115,), width=max(1, int(2 * SCALE)))
            panel.alpha_composite(mask)
            image.alpha_composite(panel, (left, y))
            draw_text(image, f"{level} HPA", (left + int(20 * SCALE), y + int(30 * SCALE)), 11 if QUICK_MODE else 22,
                      accent + (245,), bold=True, condensed=True, stroke=1)
            draw_text(image, LEVEL_META[level]["short"], (right - int(20 * SCALE), y + int(30 * SCALE)), 9 if QUICK_MODE else 18,
                      COLORS["muted"] + (220,), bold=True, condensed=True, anchor="ra", stroke=1)

        draw_text(image, "DIFFERENT SPEEDS // DIFFERENT DIRECTIONS // SAME TIME", (OUT_W // 2, int(1810 * SCALE)), 10 if QUICK_MODE else 20,
                  COLORS["cyan"] + (int(225 * p),), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_all_layers(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        image.alpha_composite(self.compose_map(self.blended_layer, opacity=0.78 + 0.2 * p))
        for level in PRESSURE_LEVELS:
            self.draw_tracers(image, level, t, opacity=0.78, size_boost=0.9)
        top = int(1450 * SCALE)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rounded_rectangle((int(48 * SCALE), top, OUT_W - int(48 * SCALE), top + int(300 * SCALE)),
                             radius=max(12, int(28 * SCALE)), fill=(2, 7, 14, 205), outline=COLORS["grid"] + (78,), width=1)
        image.alpha_composite(panel)
        draw_text(image, "ATMOSPHERIC DEPTH", (OUT_W // 2, top + int(55 * SCALE)), 15 if QUICK_MODE else 30,
                  COLORS["muted"] + (235,), bold=True, condensed=True, anchor="ma", stroke=1)
        draw_text(image, "NOT ONE WIND FIELD", (OUT_W // 2, top + int(135 * SCALE)), 28 if QUICK_MODE else 56,
                  COLORS["white"] + (255,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, "A DEEP FLUID WITH MOTION AT EVERY LEVEL", (OUT_W // 2, top + int(225 * SCALE)), 10 if QUICK_MODE else 20,
                  COLORS["cyan"] + (245,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_finale(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        image.alpha_composite(self.compose_map(self.blended_layer))
        for level in PRESSURE_LEVELS:
            self.draw_tracers(image, level, t, opacity=0.65, size_boost=0.85)
        image.alpha_composite(Image.new("RGBA", OUT_SIZE, (0, 0, 0, int(75 * p))))
        draw_text(image, "ONE PLANET", (OUT_W // 2, int(OUT_H * 0.70)), 35 if QUICK_MODE else 70,
                  COLORS["white"] + (255,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, "MANY LAYERS", (OUT_W // 2, int(OUT_H * 0.78)), 31 if QUICK_MODE else 62,
                  COLORS["white"] + (255,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, "ALWAYS MOVING", (OUT_W // 2, int(OUT_H * 0.87)), 24 if QUICK_MODE else 48,
                  COLORS["yellow"] + (255,), bold=True, condensed=True, anchor="ma", stroke=2)

    def draw_header(self, image: Image.Image, name: str):
        if name in {"opening", "stack"}:
            return
        draw_text(image, "THE ATMOSPHERE", (int(48 * SCALE), int(70 * SCALE)), 18 if not QUICK_MODE else 9,
                  COLORS["white"] + (245,), bold=True, condensed=True, stroke=1)
        draw_text(image, "IS ALWAYS MOVING", (int(48 * SCALE), int(115 * SCALE)), 34 if not QUICK_MODE else 17,
                  COLORS["white"] + (245,), bold=True, condensed=True, stroke=2)
        draw_text(image, CONFIG["subtitle"], (int(50 * SCALE), int(170 * SCALE)), 12 if not QUICK_MODE else 6,
                  COLORS["muted"] + (220,), bold=True, condensed=True, stroke=1)

    def draw_layer_key(self, image: Image.Image, name: str):
        if name not in {"all_layers", "finale"}:
            return
        x = int(60 * SCALE)
        y = int(1395 * SCALE)
        for level in PRESSURE_LEVELS:
            accent = tuple(LEVEL_META[level]["accent"])
            r = max(2, int(5 * SCALE))
            ImageDraw.Draw(image).ellipse((x - r, y - r, x + r, y + r), fill=accent + (235,))
            draw_text(image, f"{level}", (x + int(15 * SCALE), y), 9 if not QUICK_MODE else 5,
                      COLORS["muted"] + (215,), bold=True, condensed=True, anchor="lm", stroke=1)
            x += int(210 * SCALE)
        draw_text(image, "HPA", (x - int(12 * SCALE), y), 9 if not QUICK_MODE else 5,
                  COLORS["muted"] + (180,), bold=True, condensed=True, anchor="lm", stroke=1)

    def draw_source_hud(self, image: Image.Image):
        if self.data.source.startswith("synthetic"):
            text = "SYNTHETIC PREVIEW // NOT OBSERVATIONAL DATA"
        else:
            text = "NOAA PSL // NCEP-NCAR REANALYSIS 1 // ARCHIVED DAILY PRESSURE-LEVEL WIND"
        draw_text(image, text, (int(48 * SCALE), OUT_H - int(45 * SCALE)), 9 if not QUICK_MODE else 5,
                  COLORS["muted"] + (185,), bold=True, condensed=True, stroke=1)

    def draw_film_texture(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for p in self.dust:
            pulse = 0.5 + 0.5 * math.sin(t * 1.0 + p["phase"])
            if pulse < 0.69:
                continue
            x = (p["x"] + t * p["speed"] * 0.3) % OUT_W
            y = (p["y"] + math.sin(t * 0.45 + p["phase"]) * 4.0) % OUT_H
            draw.line((x, y, x + (5 if QUICK_MODE else 10) + p["r"] * 3, y), fill=COLORS["cyan"] + (int(9 * pulse),), width=1)
        offset = int((t * 41) % 8)
        for y in range(offset, OUT_H, 8):
            draw.line((0, y, OUT_W, y), fill=(120, 170, 190, 5), width=1)
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        name = str(shot["name"])
        image = self.background(t)
        if name == "opening":
            self.draw_opening(image, t, shot)
        elif name == "surface":
            self.draw_level_scene(image, t, shot, 1000)
        elif name == "lower_mid":
            self.draw_level_scene(image, t, shot, 700)
        elif name == "mid":
            self.draw_level_scene(image, t, shot, 500)
        elif name == "upper":
            self.draw_level_scene(image, t, shot, 250)
        elif name == "stack":
            self.draw_stack(image, t, shot)
        elif name == "all_layers":
            self.draw_all_layers(image, t, shot)
        else:
            self.draw_finale(image, t, shot)

        self.draw_header(image, name)
        self.draw_layer_key(image, name)
        self.draw_source_hud(image)
        self.draw_film_texture(image, t)

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        arr *= VIGNETTE[..., None]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img = ImageEnhance.Contrast(img).enhance(float(CONFIG["contrast"]))
        img = ImageEnhance.Color(img).enhance(float(CONFIG["saturation"]))
        arr = np.asarray(img, dtype=np.int16)
        rng = np.random.default_rng(int(t * 1000) + 551)
        grain = rng.normal(0.0, float(CONFIG["grain_strength"]), arr.shape[:2])[:, :, None]
        return np.clip(arr + grain, 0, 255).astype(np.uint8)


def data_date_label(when: date) -> str:
    return when.strftime("%d %B %Y").upper() + " // ARCHIVED DAILY MEAN"


# -----------------------------------------------------------------------------
# Audio and video
# -----------------------------------------------------------------------------

def gaussian_envelope(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((times - center) / max(width, 1e-6)) ** 2)


def generate_ambient_soundtrack(path: Path) -> Path:
    sr = int(CONFIG["sample_rate"])
    duration = float(CONFIG["duration_s"])
    n = int(round(sr * duration))
    times = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(90210)

    # Slow sub-bed + airy filtered noise. No external audio dependency.
    audio = 0.035 * np.sin(math.tau * 43.0 * times)
    audio += 0.022 * np.sin(math.tau * 64.5 * times + 0.7)
    audio += 0.011 * np.sin(math.tau * 96.0 * times + 1.3)

    noise = rng.normal(0.0, 1.0, n)
    kernel_len = max(8, int(sr * 0.035))
    kernel = np.ones(kernel_len, dtype=np.float64) / kernel_len
    airy = np.convolve(noise, kernel, mode="same")
    airy /= max(np.max(np.abs(airy)), 1e-9)
    air_env = 0.55 + 0.45 * np.sin(math.tau * 0.055 * times + 0.4) ** 2
    audio += 0.025 * airy * air_env

    # Shot-transition swells.
    transition_times = [5.8, 15.5, 25.0, 34.5, 44.0, 51.5, 56.0]
    scale = duration / 58.0
    for i, center in enumerate([x * scale for x in transition_times]):
        env = gaussian_envelope(times, center, 0.55 * scale + 0.18)
        tone = np.sin(math.tau * (115 + i * 17) * times + i * 0.6)
        audio += 0.025 * env * tone

    # Upper-layer shimmer enters late.
    shimmer_env = 1.0 / (1.0 + np.exp(-(times - duration * 0.55) * 0.7))
    audio += 0.008 * shimmer_env * np.sin(math.tau * 480.0 * times + 0.5 * np.sin(math.tau * 0.18 * times))

    fade = max(1, int(sr * 0.65))
    audio[:fade] *= np.linspace(0.0, 1.0, fade)
    audio[-fade:] *= np.linspace(1.0, 0.0, fade)
    audio = np.tanh(audio * 1.45)
    audio /= max(np.max(np.abs(audio)), 1e-9)
    pcm = np.int16(np.clip(audio * 0.72, -1.0, 1.0) * 32767)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def find_ffmpeg() -> Optional[str]:
    direct = shutil.which("ffmpeg")
    if direct:
        return direct
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    return None


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    command = [
        ffmpeg, "-y", "-i", str(video_path), "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest",
        "-movflags", "+faststart", str(output_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def render_video(scene: AtmosphereScene) -> Path:
    basename = str(CONFIG["output_basename"])
    raw_video = OUTPUT_ROOT / f"{basename}_silent.mp4"
    final_video = OUTPUT_ROOT / f"{basename}.mp4"
    audio_path = OUTPUT_ROOT / f"{basename}_ambient.wav"
    srt_path = OUTPUT_ROOT / f"{basename}.srt"
    write_srt(CAPTIONS, srt_path)

    fps = int(CONFIG["fps"])
    duration = float(CONFIG["duration_s"])
    frame_count = max(1, int(round(fps * duration)))
    times = np.arange(frame_count, dtype=float) / fps
    with iio.get_writer(
        raw_video,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times, desc="Rendering atmosphere short"):
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

