from __future__ import annotations

"""
A YEAR OF GLOBAL WIND IN 60 SECONDS — cinematic YouTube Shorts renderer

Creates a vertical 1080x1920 data-driven short that compresses one complete
calendar year of global daily-mean wind into about one minute.

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



Default year: 2025, chosen as a complete archived calendar year in the source
used by the supplied jet-stream renderer.

The moving streamline graphics are cinematic flow cues through daily-mean wind.
They are not literal air-parcel trajectories or minute-by-minute forecasts.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm scipy

Quick preview render
--------------------
    GLOBAL_WIND_QUICK=1 python a_year_of_global_wind_in_60_seconds.py

Force offline fixture mode
--------------------------
    GLOBAL_WIND_OFFLINE=1 python a_year_of_global_wind_in_60_seconds.py

Choose another archived year
----------------------------
    GLOBAL_WIND_YEAR=2024 python a_year_of_global_wind_in_60_seconds.py

Outputs
-------
- final vertical MP4 with generated ambient audio when ffmpeg is available
- silent MP4 fallback
- SRT subtitle sidecar
- preview PNG frames
- CSV with one row of wind statistics per day
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
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

QUICK_MODE = os.environ.get("GLOBAL_WIND_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("GLOBAL_WIND_OFFLINE", "0") == "1"
PRESSURE_HPA = 850.0
DEFAULT_YEAR = 2025


def parse_year() -> int:
    raw = os.environ.get("GLOBAL_WIND_YEAR", "").strip()
    if not raw:
        return DEFAULT_YEAR
    try:
        year = int(raw)
    except ValueError as exc:
        raise ValueError("GLOBAL_WIND_YEAR must be a four-digit year, e.g. 2024") from exc
    if year < 1948 or year > 2100:
        raise ValueError("GLOBAL_WIND_YEAR is outside the supported archival range")
    return year


TARGET_YEAR = parse_year()
START_DATE = date(TARGET_YEAR, 1, 1)
END_DATE = date(TARGET_YEAR, 12, 31)

OUTPUT_ROOT = Path("a_year_of_global_wind_in_60_seconds_output")
DATA_ROOT = OUTPUT_ROOT / "data"
CACHE_ROOT = DATA_ROOT / "cache"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, CACHE_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {

}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0

COLORS = {

}

FULL_SHOT_PLAN = [
    {"name": "opening", "start": 0.0, "end": 5.5},
    {"name": "year", "start": 5.5, "end": 41.5},
    {"name": "seasons", "start": 41.5, "end": 48.0},
    {"name": "strongest_day", "start": 48.0, "end": 52.8},
    {"name": "annual_mean", "start": 52.8, "end": 56.0},
    {"name": "finale", "start": 56.0, "end": 58.0},
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
class WindYearDataset:
    dates: List[date]
    lat: np.ndarray
    lon: np.ndarray
    u: np.ndarray  # day, lat, lon
    v: np.ndarray  # day, lat, lon
    source: str
    notes: List[str]

    @property
    def speed(self) -> np.ndarray:
        return np.hypot(self.u, self.v)


@dataclass(frozen=True)
class DayStats:
    when: date
    global_max_ms: float
    global_max_lat: float
    global_max_lon: float
    north_midlat_mean_ms: float
    tropics_mean_ms: float
    south_midlat_mean_ms: float
    global_mean_ms: float
    p90_ms: float


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

def ncss_url(var: str, year: int) -> str:
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
        "time_start": f"{START_DATE.isoformat()}T00:00:00Z",
        "time_end": f"{END_DATE.isoformat()}T00:00:00Z",
        "timeStride": "1",
        "vertCoord": str(int(PRESSURE_HPA)),
        "accept": "netcdf3",
    }
    return base + "?" + urllib.parse.urlencode(params)


def request_bytes(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; GlobalWindYearShort/1.0; educational visualization)",
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
    if data.ndim != 3:
        raise RuntimeError(f"Expected {var_name} to resolve to [time,lat,lon], got shape {data.shape}")
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
        data = data[..., ::-1, :]
    return data.astype(np.float32), lat, lon_sorted


def download_year() -> WindYearDataset:
    notes: List[str] = []

    def obtain(var: str) -> Path:
        path = CACHE_ROOT / f"{var}_{int(PRESSURE_HPA)}hpa_{TARGET_YEAR}.nc"
        try:
            payload = request_bytes(ncss_url(var, TARGET_YEAR), timeout=240)
            if len(payload) < 100_000:
                raise RuntimeError(f"NOAA subset response was unexpectedly small ({len(payload)} bytes)")
            path.write_bytes(payload)
            notes.append(f"Downloaded NOAA PSL {var} {int(PRESSURE_HPA)} hPa subset for {TARGET_YEAR}")
            return path
        except Exception as exc:
            notes.append(f"Live NOAA {var} download failed: {exc}")
            if path.exists() and path.stat().st_size > 100_000:
                notes.append(f"Using cached NOAA {var} subset")
                return path
            raise RuntimeError(f"No live or cached {var} data available") from exc

    u_path = obtain("uwnd")
    v_path = obtain("vwnd")
    u, lat_u, lon_u = load_netcdf3(u_path, "uwnd")
    v, lat_v, lon_v = load_netcdf3(v_path, "vwnd")
    u, lat, lon = standardize_grid(u, lat_u, lon_u)
    v, lat2, lon2 = standardize_grid(v, lat_v, lon_v)

    if u.shape != v.shape:
        raise RuntimeError(f"U/V shapes differ: {u.shape} vs {v.shape}")
    if not (np.allclose(lat, lat2) and np.allclose(lon, lon2)):
        raise RuntimeError("U/V coordinate grids differ")

    expected_dates = [START_DATE + timedelta(days=i) for i in range((END_DATE - START_DATE).days + 1)]
    if u.shape[0] < len(expected_dates):
        raise RuntimeError(f"NOAA returned only {u.shape[0]} daily slices; need {len(expected_dates)}")
    if u.shape[0] > len(expected_dates):
        u = u[: len(expected_dates)]
        v = v[: len(expected_dates)]

    notes.append("850 hPa is a pressure surface; its geometric height varies in space and time")
    notes.append("Streamlines are cinematic direction cues through daily-mean wind, not parcel trajectories")
    return WindYearDataset(
        dates=expected_dates,
        lat=lat,
        lon=lon,
        u=u,
        v=v,
        source="noaa_psl_ncep_ncar_reanalysis1_850hpa_daily",
        notes=notes,
    )


def make_synthetic_year() -> WindYearDataset:
    dates = [START_DATE + timedelta(days=i) for i in range((END_DATE - START_DATE).days + 1)]
    lat = np.arange(-90.0, 90.01, 2.5, dtype=np.float32)
    lon = np.arange(-180.0, 180.0, 2.5, dtype=np.float32)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    u_days: List[np.ndarray] = []
    v_days: List[np.ndarray] = []

    for i, _ in enumerate(dates):
        phase = i / len(dates) * math.tau
        # Seasonal storm-track migration and trade-wind modulation.
        north_axis = 45.0 + 7.0 * math.sin(phase - 0.7) + 5.0 * np.sin(np.deg2rad(lon2d * 1.35) + phase * 1.2)
        south_axis = -45.0 + 7.0 * math.sin(phase + 0.7) + 5.0 * np.sin(np.deg2rad(lon2d * 1.25) - phase * 1.1)
        north_mid = np.exp(-0.5 * ((lat2d - north_axis) / 13.0) ** 2)
        south_mid = np.exp(-0.5 * ((lat2d - south_axis) / 13.0) ** 2)
        trades_n = np.exp(-0.5 * ((lat2d - 18.0) / 12.0) ** 2)
        trades_s = np.exp(-0.5 * ((lat2d + 18.0) / 12.0) ** 2)

        north_strength = 19.0 + 6.0 * math.cos(phase)
        south_strength = 19.0 - 6.0 * math.cos(phase)
        uu = 3.0 + north_strength * north_mid + south_strength * south_mid
        uu -= (8.5 + 2.0 * math.sin(phase)) * trades_n
        uu -= (8.5 - 2.0 * math.sin(phase)) * trades_s
        uu += 6.0 * np.sin(np.deg2rad(lon2d * 1.7) + phase * 1.6) * (north_mid + south_mid)

        vv = 7.5 * north_mid * np.cos(np.deg2rad(lon2d * 1.35) + phase * 1.2)
        vv -= 7.0 * south_mid * np.cos(np.deg2rad(lon2d * 1.25) - phase * 1.1)
        vv += 2.8 * np.sin(np.deg2rad(lat2d * 3.0) + phase * 0.7)

        # A few moving synoptic pulses make the synthetic year feel less periodic.
        for offset, sign in [(0.08, 1.0), (0.31, -1.0), (0.57, 1.0), (0.82, -1.0)]:
            center_phase = (phase - offset * math.tau + math.pi) % math.tau - math.pi
            pulse = math.exp(-0.5 * (center_phase / 0.22) ** 2)
            lon_center = ((offset * 720.0 + i * 2.8) % 360.0) - 180.0
            dlon = ((lon2d - lon_center + 180.0) % 360.0) - 180.0
            vortex = np.exp(-0.5 * ((dlon / 28.0) ** 2 + ((lat2d - sign * 42.0) / 15.0) ** 2))
            uu += pulse * 12.0 * vortex
            vv += pulse * 9.0 * vortex * np.sign(dlon)

        u_days.append(uu.astype(np.float32))
        v_days.append(vv.astype(np.float32))

    return WindYearDataset(
        dates=dates,
        lat=lat,
        lon=lon,
        u=np.stack(u_days),
        v=np.stack(v_days),
        source="synthetic_procedural_fixture",
        notes=[
            "Using deterministic synthetic annual wind fixture; not observational data",
            "850 hPa altitude is approximate and varies",
            "Streamlines are cinematic flow cues, not parcel trajectories",
        ],
    )


def load_winds() -> WindYearDataset:
    if OFFLINE_MODE:
        return make_synthetic_year()
    try:
        return download_year()
    except Exception as exc:
        fixture = make_synthetic_year()
        return WindYearDataset(
            fixture.dates,
            fixture.lat,
            fixture.lon,
            fixture.u,
            fixture.v,
            fixture.source,
            [f"Archived NOAA load failed: {exc}"] + fixture.notes,
        )


# -----------------------------------------------------------------------------
# Statistics and data products
# -----------------------------------------------------------------------------

def compute_day_stats(data: WindYearDataset) -> List[DayStats]:
    speed = data.speed
    lat2d = np.repeat(data.lat[:, None], len(data.lon), axis=1)
    north_mid = (lat2d >= 30) & (lat2d <= 60)
    tropics = np.abs(lat2d) <= 25
    south_mid = (lat2d <= -30) & (lat2d >= -60)
    stats: List[DayStats] = []

    for i, when in enumerate(data.dates):
        s = speed[i]
        safe = np.where(np.isfinite(s), s, -np.inf)
        flat = int(np.argmax(safe))
        iy, ix = np.unravel_index(flat, s.shape)
        stats.append(
            DayStats(
                when=when,
                global_max_ms=float(np.nanmax(s)),
                global_max_lat=float(data.lat[iy]),
                global_max_lon=float(data.lon[ix]),
                north_midlat_mean_ms=float(np.nanmean(np.where(north_mid, s, np.nan))),
                tropics_mean_ms=float(np.nanmean(np.where(tropics, s, np.nan))),
                south_midlat_mean_ms=float(np.nanmean(np.where(south_mid, s, np.nan))),
                global_mean_ms=float(np.nanmean(s)),
                p90_ms=float(np.nanpercentile(s, 90.0)),
            )
        )
    return stats


def season_indices(dates: Sequence[date], season: str) -> List[int]:
    months = {
        "DJF": {12, 1, 2},
        "MAM": {3, 4, 5},
        "JJA": {6, 7, 8},
        "SON": {9, 10, 11},
    }[season]
    return [i for i, d in enumerate(dates) if d.month in months]


def save_data_products(data: WindYearDataset, stats: Sequence[DayStats]) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "global_wind_daily_stats.csv"
    summary_path = DATA_ROOT / "global_wind_summary.json"
    rows = [
        {
            "date": s.when.isoformat(),
            "global_max_ms": s.global_max_ms,
            "global_max_kmh": s.global_max_ms * 3.6,
            "global_max_lat": s.global_max_lat,
            "global_max_lon": s.global_max_lon,
            "north_midlat_mean_ms": s.north_midlat_mean_ms,
            "tropics_mean_ms": s.tropics_mean_ms,
            "south_midlat_mean_ms": s.south_midlat_mean_ms,
            "global_mean_ms": s.global_mean_ms,
            "p90_ms": s.p90_ms,
        }
        for s in stats
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    strongest = max(stats, key=lambda s: s.global_max_ms)
    monthly_means: Dict[str, float] = {}
    for month in range(1, 13):
        values = [s.global_mean_ms for s in stats if s.when.month == month]
        monthly_means[f"{month:02d}"] = float(np.mean(values)) if values else float("nan")

    summary = {
        "title": CONFIG["title"],
        "year": TARGET_YEAR,
        "pressure_hpa": PRESSURE_HPA,
        "days_loaded": len(data.dates),
        "start_date": data.dates[0].isoformat(),
        "end_date": data.dates[-1].isoformat(),
        "data_source": data.source,
        "source_notes": data.notes,
        "strongest_day": {
            "date": strongest.when.isoformat(),
            "max_wind_ms": strongest.global_max_ms,
            "max_wind_kmh": strongest.global_max_ms * 3.6,
            "latitude": strongest.global_max_lat,
            "longitude": strongest.global_max_lon,
        },
        "monthly_global_mean_ms": monthly_means,
        "annual_global_mean_ms": float(np.mean([s.global_mean_ms for s in stats])),
        "source_urls": {
            "noaa_reanalysis": "https://psl.noaa.gov/data/reanalysis/reanalysis.shtml",
            "thredds_catalog": "https://psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis/Dailies/pressure/catalog.html",
        },
        "scientific_note": (
            "The displayed field is daily-mean horizontal wind at 850 hPa. The pressure surface is not a fixed altitude, "
            "and the cinematic streamlines are not literal parcel trajectories."
        ),
        "fallback_warning": "synthetic_procedural_fixture is preview data, not observational data",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, summary_path


# -----------------------------------------------------------------------------
# Map geometry and colors
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


def speed_color_array(speed: np.ndarray) -> np.ndarray:
    stops = np.array([0.0, 6.0, 12.0, 18.0, 25.0, 35.0, 50.0, 70.0], dtype=float)
    colors = np.array([
        [3, 15, 31],
        [17, 61, 84],
        [31, 128, 149],
        [52, 203, 209],
        [246, 211, 91],
        [255, 129, 75],
        [237, 91, 207],
        [247, 244, 250],
    ], dtype=float)
    x = np.nan_to_num(speed, nan=0.0, posinf=70.0, neginf=0.0)
    rgb = np.empty(x.shape + (3,), dtype=np.uint8)
    for c in range(3):
        rgb[..., c] = np.clip(np.interp(x.ravel(), stops, colors[:, c]).reshape(x.shape), 0, 255).astype(np.uint8)
    return rgb


def speed_color(speed_ms: float, alpha: int = 255) -> Tuple[int, int, int, int]:
    arr = speed_color_array(np.array([[float(speed_ms)]], dtype=float))[0, 0]
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


def month_name(month: int) -> str:
    return ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"][month - 1]


def season_for_month(month: int) -> Tuple[str, Tuple[int, int, int]]:
    if month in {12, 1, 2}:
        return "DJF // BOREAL WINTER", COLORS["winter"]
    if month in {3, 4, 5}:
        return "MAM // BOREAL SPRING", COLORS["spring"]
    if month in {6, 7, 8}:
        return "JJA // BOREAL SUMMER", COLORS["summer"]
    return "SON // BOREAL AUTUMN", COLORS["autumn"]


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class GlobalWindYearScene:
    def __init__(self, data: WindYearDataset, stats: Sequence[DayStats]):
        self.data = data
        self.stats = list(stats)
        self.speed = data.speed
        self.strongest_index = int(np.argmax([s.global_max_ms for s in stats]))
        self.particles = self._make_particles(int(CONFIG["star_count"]), 291)
        self.dust = self._make_particles(int(CONFIG["dust_count"]), 711)
        self.static_map = self._render_static_map()
        self.layer_cache: "OrderedDict[int, Image.Image]" = OrderedDict()
        self.annual_mean_layer = self._render_mean_layer(np.nanmean(self.speed, axis=0), label="annual")
        self.season_layers: Dict[str, Image.Image] = {}
        for season in ("DJF", "MAM", "JJA", "SON"):
            idx = season_indices(self.data.dates, season)
            mean_speed = np.nanmean(self.speed[idx], axis=0)
            self.season_layers[season] = self._render_mean_layer(mean_speed, label=season)

    @staticmethod
    def _make_particles(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.4, 1.8 if QUICK_MODE else 2.2)),
                "a": float(rng.uniform(10, 46)),
                "phase": float(rng.uniform(0, math.tau)),
                "speed": float(rng.uniform(0.6, 5.0)),
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
            q = y / max(OUT_H - 1, 1)
            rgb = (top * (1.0 - q) + bottom * q).astype(np.uint8)
            arr[y, :, :3] = rgb
            arr[y, :, 3] = 255
        image = Image.fromarray(arr, "RGBA")
        draw = ImageDraw.Draw(image)
        x0, x1 = int(CONFIG["map_margin_x"]), OUT_W - int(CONFIG["map_margin_x"])
        y0, y1 = int(CONFIG["map_top"]), int(CONFIG["map_bottom"])

        for lat in range(-60, 61, 20):
            _, y = self.project(0, lat)
            draw.line((x0, y, x1, y), fill=COLORS["grid"] + (23,), width=1)
        for lon in range(-150, 181, 30):
            x, _ = self.project(lon, 0)
            draw.line((x, y0, x, y1), fill=COLORS["grid"] + (20,), width=1)

        for polygon in BUILTIN_LAND_POLYGONS:
            rel = unwrapped_relative_longitudes(polygon)
            lats = [lat for _, lat in polygon]
            for shift in (-360.0, 0.0, 360.0):
                pts = [self.project(relative + shift, lat) for relative, lat in zip(rel, lats)]
                if pts and max(p[0] for p in pts) >= x0 - 50 and min(p[0] for p in pts) <= x1 + 50:
                    draw.polygon(pts, fill=COLORS["land"] + (245,), outline=COLORS["land_edge"] + (92,))

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

    def _streamline(self, day_index: int, lon0: float, lat0: float) -> Tuple[List[Tuple[float, float]], List[float]]:
        u = self.data.u[day_index]
        v = self.data.v[day_index]
        points: List[Tuple[float, float]] = []
        speeds: List[float] = []
        lon, lat = float(lon0), float(lat0)
        for _ in range(int(CONFIG["stream_step_count"])):
            uu = self._interp_field(u, lon, lat)
            vv = self._interp_field(v, lon, lat)
            spd = math.hypot(uu, vv)
            if not np.isfinite(spd) or spd < 5.0:
                break
            points.append(self.project(lon, lat))
            speeds.append(spd)
            coslat = max(0.28, math.cos(math.radians(lat)))
            lon += uu * (0.14 if QUICK_MODE else 0.105) / coslat
            lat += vv * (0.075 if QUICK_MODE else 0.055)
            lon = ((lon + 180.0) % 360.0) - 180.0
            if abs(lat) > 83:
                break
        return points, speeds

    def _render_day_layer(self, day_index: int) -> Image.Image:
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        x0, x1 = int(CONFIG["map_margin_x"]), OUT_W - int(CONFIG["map_margin_x"])
        y0, y1 = int(CONFIG["map_top"]), int(CONFIG["map_bottom"])
        map_w, map_h = x1 - x0, y1 - y0

        speed = self.speed[day_index]
        rgb = speed_color_array(np.flipud(speed))
        alpha = np.clip((np.flipud(speed) - 4.0) / 28.0 * 178.0, 0, 178).astype(np.uint8)
        heat = Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA").resize((map_w, map_h), Image.Resampling.BILINEAR)
        heat = heat.filter(ImageFilter.GaussianBlur(max(1, int(3.2 * SCALE))))
        layer.alpha_composite(heat, (x0, y0))

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        core = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        cd = ImageDraw.Draw(core)

        seed_lats = [-62, -50, -38, -26, -14, 14, 26, 38, 50, 62]
        spacing = int(CONFIG["stream_seed_spacing_lon"])
        for lat0 in seed_lats:
            for lon0 in range(-180, 180, spacing):
                pts, speeds = self._streamline(day_index, float(lon0), float(lat0))
                if len(pts) < 2:
                    continue
                mean_speed = float(np.mean(speeds))
                if mean_speed < 8.0:
                    continue
                color = speed_color(mean_speed, 215)
                for p1, p2 in zip(pts[:-1], pts[1:]):
                    if abs(p2[0] - p1[0]) > map_w * 0.5:
                        continue
                    gd.line((*p1, *p2), fill=color[:3] + (62,), width=max(2, int(6 * SCALE)))
                    cd.line((*p1, *p2), fill=color, width=max(1, int(2.0 * SCALE)))

        glow = glow.filter(ImageFilter.GaussianBlur(max(1, int(4.5 * SCALE))))
        layer.alpha_composite(glow)
        layer.alpha_composite(core)
        return layer

    def get_day_layer(self, day_index: int) -> Image.Image:
        day_index = int(np.clip(day_index, 0, len(self.data.dates) - 1))
        if day_index in self.layer_cache:
            layer = self.layer_cache.pop(day_index)
            self.layer_cache[day_index] = layer
            return layer
        layer = self._render_day_layer(day_index)
        self.layer_cache[day_index] = layer
        while len(self.layer_cache) > int(CONFIG["layer_cache_size"]):
            self.layer_cache.popitem(last=False)
        return layer

    def _render_mean_layer(self, mean_speed: np.ndarray, label: str) -> Image.Image:
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        x0, x1 = int(CONFIG["map_margin_x"]), OUT_W - int(CONFIG["map_margin_x"])
        y0, y1 = int(CONFIG["map_top"]), int(CONFIG["map_bottom"])
        map_w, map_h = x1 - x0, y1 - y0
        rgb = speed_color_array(np.flipud(mean_speed))
        alpha = np.clip((np.flipud(mean_speed) - 4.0) / 24.0 * 205.0, 0, 205).astype(np.uint8)
        heat = Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA").resize((map_w, map_h), Image.Resampling.BILINEAR)
        layer.alpha_composite(heat, (x0, y0))
        return layer

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, COLORS["dark"] + (255,))
        draw = ImageDraw.Draw(image)
        for p in self.particles:
            x = (p["x"] + math.sin(t * 0.13 + p["phase"]) * 7.0) % OUT_W
            y = (p["y"] + t * p["speed"] * 0.07) % OUT_H
            a = int(p["a"] * (0.55 + 0.45 * math.sin(t * 0.6 + p["phase"]) ** 2))
            r = p["r"]
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(177, 218, 232, a))
        return image

    def compose_map(self, layer: Optional[Image.Image] = None, opacity: float = 1.0) -> Image.Image:
        base = self.static_map.copy()
        if layer is not None:
            if opacity < 0.999:
                layer = layer.copy()
                alpha = layer.getchannel("A").point(lambda v: int(v * opacity))
                layer.putalpha(alpha)
            base.alpha_composite(layer)
        return base

    def day_blend(self, progress: float) -> Tuple[Image.Image, int, int, float]:
        pos = clamp(progress) * (len(self.data.dates) - 1)
        i0 = int(math.floor(pos))
        i1 = min(i0 + 1, len(self.data.dates) - 1)
        f = smoothstep(pos - i0)
        layer0 = self.get_day_layer(i0)
        if i0 == i1:
            return layer0, i0, i1, f
        layer1 = self.get_day_layer(i1)
        return Image.blend(layer0, layer1, f), i0, i1, f

    def draw_opening(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        idx = min(int((len(self.data.dates) - 1) * (0.08 + 0.10 * p)), len(self.data.dates) - 1)
        image.alpha_composite(self.compose_map(self.get_day_layer(idx), opacity=0.45 + 0.42 * p))
        draw_text(image, "A YEAR OF", (OUT_W // 2, int(OUT_H * 0.68)), 27 if QUICK_MODE else 54,
                  COLORS["muted"] + (245,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, "GLOBAL WIND", (OUT_W // 2, int(OUT_H * 0.765)), 46 if QUICK_MODE else 92,
                  COLORS["white"] + (255,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, "IN 60 SECONDS", (OUT_W // 2, int(OUT_H * 0.85)), 25 if QUICK_MODE else 50,
                  COLORS["white"] + (250,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, str(TARGET_YEAR), (OUT_W // 2, int(OUT_H * 0.915)), 12 if QUICK_MODE else 24,
                  COLORS["moderate"] + (240,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_year(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smootherstep(shot_progress(t, shot))
        layer, i0, i1, f = self.day_blend(p)
        image.alpha_composite(self.compose_map(layer))
        idx = i0 if f < 0.5 else i1
        stat = self.stats[idx]
        when = stat.when
        season_text, season_color = season_for_month(when.month)

        left = int(48 * SCALE)
        top = int(1400 * SCALE)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rounded_rectangle((left, top, OUT_W - left, top + int(330 * SCALE)),
                             radius=max(12, int(28 * SCALE)), fill=(2, 7, 14, 207), outline=COLORS["grid"] + (80,), width=1)
        image.alpha_composite(panel)

        draw_text(image, f"DAY {idx + 1:03d} / {len(self.data.dates)}", (left + int(28 * SCALE), top + int(34 * SCALE)),
                  14 if QUICK_MODE else 28, COLORS["moderate"] + (250,), bold=True, condensed=True, stroke=1)
        draw_text(image, when.strftime("%d %B %Y").upper(), (left + int(28 * SCALE), top + int(92 * SCALE)),
                  26 if QUICK_MODE else 52, COLORS["white"] + (255,), bold=True, condensed=True, stroke=2)
        draw_text(image, season_text, (left + int(28 * SCALE), top + int(155 * SCALE)),
                  10 if QUICK_MODE else 20, season_color + (240,), bold=True, condensed=True, stroke=1)
        draw_text(image, f"PEAK 850 HPA WIND  {stat.global_max_ms:.0f} M/S  //  {stat.global_max_ms * 3.6:.0f} KM/H",
                  (left + int(28 * SCALE), top + int(205 * SCALE)), 10 if QUICK_MODE else 20,
                  speed_color(stat.global_max_ms, 245), bold=True, condensed=True, stroke=1)
        draw_text(image, f"GLOBAL MEAN  {stat.global_mean_ms:.1f} M/S   //   90TH PERCENTILE  {stat.p90_ms:.1f} M/S",
                  (left + int(28 * SCALE), top + int(250 * SCALE)), 9 if QUICK_MODE else 18,
                  COLORS["muted"] + (225,), bold=True, condensed=True, stroke=1)

        # 12-month timeline.
        draw = ImageDraw.Draw(image)
        x0 = left + int(28 * SCALE)
        x1 = OUT_W - left - int(28 * SCALE)
        bar_y = top + int(298 * SCALE)
        draw.rounded_rectangle((x0, bar_y, x1, bar_y + max(2, int(7 * SCALE))), radius=max(1, int(4 * SCALE)),
                               fill=COLORS["grid"] + (62,))
        frac = idx / max(len(self.data.dates) - 1, 1)
        px = lerp(x0, x1, frac)
        draw.rounded_rectangle((x0, bar_y, px, bar_y + max(2, int(7 * SCALE))), radius=max(1, int(4 * SCALE)),
                               fill=season_color + (225,))
        for month in range(1, 13):
            month_day = (date(TARGET_YEAR, month, 1) - START_DATE).days
            mx = lerp(x0, x1, month_day / max(len(self.data.dates) - 1, 1))
            draw.line((mx, bar_y - int(4 * SCALE), mx, bar_y + int(12 * SCALE)), fill=COLORS["muted"] + (85,), width=1)
            if month in {1, 3, 5, 7, 9, 11}:
                draw_text(image, month_name(month), (int(mx), bar_y + int(23 * SCALE)), 6 if QUICK_MODE else 12,
                          COLORS["muted"] + (185,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_seasons(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = shot_progress(t, shot)
        seasons = ["DJF", "MAM", "JJA", "SON"]
        idx = min(int(p * 4), 3)
        season = seasons[idx]
        image.alpha_composite(self.compose_map(self.season_layers[season], opacity=0.96))
        labels = {
            "DJF": ("DEC–JAN–FEB", "BOREAL WINTER / AUSTRAL SUMMER", COLORS["winter"]),
            "MAM": ("MAR–APR–MAY", "BOREAL SPRING / AUSTRAL AUTUMN", COLORS["spring"]),
            "JJA": ("JUN–JUL–AUG", "BOREAL SUMMER / AUSTRAL WINTER", COLORS["summer"]),
            "SON": ("SEP–OCT–NOV", "BOREAL AUTUMN / AUSTRAL SPRING", COLORS["autumn"]),
        }
        months, subtitle, accent = labels[season]
        top = int(1450 * SCALE)
        draw_text(image, "SEASONAL MEAN", (OUT_W // 2, top), 15 if QUICK_MODE else 30,
                  COLORS["muted"] + (235,), bold=True, condensed=True, anchor="ma", stroke=1)
        draw_text(image, months, (OUT_W // 2, top + int(74 * SCALE)), 34 if QUICK_MODE else 68,
                  COLORS["white"] + (255,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, subtitle, (OUT_W // 2, top + int(150 * SCALE)), 11 if QUICK_MODE else 22,
                  accent + (245,), bold=True, condensed=True, anchor="ma", stroke=1)
        draw_text(image, "THE PLANET'S WIND BELTS SHIFT WITH THE SEASONS", (OUT_W // 2, top + int(215 * SCALE)),
                  9 if QUICK_MODE else 18, COLORS["muted"] + (220,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_strongest_day(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        stat = self.stats[self.strongest_index]
        image.alpha_composite(self.compose_map(self.get_day_layer(self.strongest_index)))
        x, y = self.project(stat.global_max_lon, stat.global_max_lat)
        pulse = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(pulse)
        rr = (12 + 8 * math.sin(t * 5.0) ** 2) * SCALE
        c = speed_color(stat.global_max_ms, 255)
        pd.ellipse((x - rr, y - rr, x + rr, y + rr), outline=c, width=max(1, int(3 * SCALE)))
        pd.ellipse((x - rr * 2.0, y - rr * 2.0, x + rr * 2.0, y + rr * 2.0), outline=c[:3] + (88,), width=max(1, int(2 * SCALE)))
        image.alpha_composite(pulse.filter(ImageFilter.GaussianBlur(max(1, int(1.2 * SCALE)))))

        draw_text(image, "STRONGEST DAY OF THE YEAR", (int(56 * SCALE), int(1450 * SCALE)), 13 if QUICK_MODE else 26,
                  COLORS["muted"] + (235,), bold=True, condensed=True, stroke=1)
        draw_text(image, stat.when.strftime("%d %B %Y").upper(), (int(56 * SCALE), int(1520 * SCALE)), 31 if QUICK_MODE else 62,
                  COLORS["white"] + (255,), bold=True, condensed=True, stroke=2)
        draw_text(image, f"{stat.global_max_ms:.0f} M/S  //  {stat.global_max_ms * 3.6:.0f} KM/H", (int(56 * SCALE), int(1610 * SCALE)),
                  22 if QUICK_MODE else 44, c, bold=True, condensed=True, stroke=2)
        draw_text(image, f"DAILY-MEAN WIND AT {int(PRESSURE_HPA)} HPA", (int(56 * SCALE), int(1686 * SCALE)), 10 if QUICK_MODE else 20,
                  COLORS["muted"] + (225,), bold=True, condensed=True, stroke=1)

    def draw_annual_mean(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        image.alpha_composite(self.compose_map(self.annual_mean_layer, opacity=0.74 + 0.24 * p))
        top = int(1450 * SCALE)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rounded_rectangle((int(48 * SCALE), top, OUT_W - int(48 * SCALE), top + int(270 * SCALE)),
                             radius=max(12, int(28 * SCALE)), fill=(2, 7, 14, 200), outline=COLORS["grid"] + (74,), width=1)
        image.alpha_composite(panel)
        draw_text(image, f"{TARGET_YEAR} ANNUAL MEAN", (OUT_W // 2, top + int(55 * SCALE)), 16 if QUICK_MODE else 32,
                  COLORS["muted"] + (240,), bold=True, condensed=True, anchor="ma", stroke=1)
        draw_text(image, "THE PERSISTENT WIND CORRIDORS", (OUT_W // 2, top + int(135 * SCALE)), 24 if QUICK_MODE else 48,
                  COLORS["white"] + (255,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, "BRIGHTER ZONES = STRONGER MEAN 850 HPA WIND", (OUT_W // 2, top + int(215 * SCALE)), 9 if QUICK_MODE else 18,
                  COLORS["muted"] + (220,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_finale(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        image.alpha_composite(self.compose_map(self.annual_mean_layer))
        image.alpha_composite(Image.new("RGBA", OUT_SIZE, (0, 0, 0, int(64 * p))))
        draw_text(image, "365 DAYS", (OUT_W // 2, int(OUT_H * 0.72)), 38 if QUICK_MODE else 76,
                  COLORS["white"] + (255,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, "ONE ATMOSPHERE", (OUT_W // 2, int(OUT_H * 0.80)), 27 if QUICK_MODE else 54,
                  COLORS["white"] + (255,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, "NEVER STILL", (OUT_W // 2, int(OUT_H * 0.875)), 17 if QUICK_MODE else 34,
                  COLORS["moderate"] + (245,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_header(self, image: Image.Image, name: str):
        if name == "opening":
            return
        draw_text(image, "A YEAR OF GLOBAL WIND", (int(48 * SCALE), int(72 * SCALE)), 20 if not QUICK_MODE else 10,
                  COLORS["white"] + (245,), bold=True, condensed=True, stroke=1)
        draw_text(image, "IN 60 SECONDS", (int(48 * SCALE), int(116 * SCALE)), 36 if not QUICK_MODE else 18,
                  COLORS["white"] + (245,), bold=True, condensed=True, stroke=2)
        draw_text(image, CONFIG["subtitle"], (int(50 * SCALE), int(170 * SCALE)), 13 if not QUICK_MODE else 7,
                  COLORS["muted"] + (220,), bold=True, condensed=True, stroke=1)

    def draw_legend(self, image: Image.Image, name: str):
        if name not in {"year", "strongest_day", "annual_mean", "finale"}:
            return
        items = [("8", 8), ("15", 15), ("25", 25), ("35", 35), ("50+ M/S", 50)]
        x = int(55 * SCALE)
        y = int(1365 * SCALE)
        for label, speed in items:
            c = speed_color(float(speed), 235)
            r = max(2, int(5 * SCALE))
            ImageDraw.Draw(image).ellipse((x - r, y - r, x + r, y + r), fill=c)
            draw_text(image, label, (x + int(15 * SCALE), y), 10 if not QUICK_MODE else 5,
                      COLORS["muted"] + (215,), bold=True, condensed=True, anchor="lm", stroke=1)
            x += int(190 * SCALE)

    def draw_source_hud(self, image: Image.Image):
        if self.data.source.startswith("synthetic"):
            text = "SYNTHETIC PREVIEW // NOT OBSERVATIONAL DATA"
        else:
            text = f"NOAA PSL // NCEP-NCAR REANALYSIS 1 // DAILY {int(PRESSURE_HPA)} HPA WIND // {TARGET_YEAR}"
        draw_text(image, text, (int(48 * SCALE), OUT_H - int(46 * SCALE)), 10 if not QUICK_MODE else 5,
                  COLORS["muted"] + (185,), bold=True, condensed=True, stroke=1)

    def draw_film_texture(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for p in self.dust:
            pulse = 0.5 + 0.5 * math.sin(t * 1.0 + p["phase"])
            if pulse < 0.69:
                continue
            x = (p["x"] + t * p["speed"] * 0.28) % OUT_W
            y = (p["y"] + math.sin(t * 0.45 + p["phase"]) * 4.0) % OUT_H
            draw.line((x, y, x + (5 if QUICK_MODE else 10) + p["r"] * 3, y),
                      fill=COLORS["moderate"] + (int(10 * pulse),), width=1)
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
        elif name == "year":
            self.draw_year(image, t, shot)
        elif name == "seasons":
            self.draw_seasons(image, t, shot)
        elif name == "strongest_day":
            self.draw_strongest_day(image, t, shot)
        elif name == "annual_mean":
            self.draw_annual_mean(image, t, shot)
        else:
            self.draw_finale(image, t, shot)

        self.draw_header(image, name)
        self.draw_legend(image, name)
        self.draw_source_hud(image)
        self.draw_film_texture(image, t)

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        arr *= VIGNETTE[..., None]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img = ImageEnhance.Contrast(img).enhance(float(CONFIG["contrast"]))
        img = ImageEnhance.Color(img).enhance(float(CONFIG["saturation"]))
        arr = np.asarray(img, dtype=np.int16)
        rng = np.random.default_rng(int(t * 1000) + 913)
        grain = rng.normal(0.0, float(CONFIG["grain_strength"]), arr.shape[:2])[:, :, None]
        return np.clip(arr + grain, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Audio and video
# -----------------------------------------------------------------------------

def gaussian_envelope(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((times - center) / max(width, 1e-6)) ** 2)


def generate_ambient_soundtrack(path: Path) -> Path:
    sample_rate = int(CONFIG["sample_rate"])
    duration = float(CONFIG["duration_s"])
    count = int(round(sample_rate * duration))
    times = np.arange(count, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(365850)
    audio = np.zeros(count, dtype=np.float64)
    audio += 0.060 * np.sin(math.tau * 35.0 * times + 0.6 * np.sin(math.tau * 0.06 * times))
    audio += 0.038 * np.sin(math.tau * 57.0 * times + 1.0)
    audio += 0.014 * np.sin(math.tau * 114.0 * times + 0.5 * np.sin(math.tau * 0.11 * times))
    controls = rng.normal(0.0, 1.0, max(8, int(duration * 5)))
    slow_noise = np.interp(times, np.linspace(0, duration, len(controls)), controls)
    audio += 0.016 * slow_noise

    sweep = next(s for s in SHOT_PLAN if s["name"] == "year")
    for fraction in np.linspace(0.02, 0.98, 18 if not QUICK_MODE else 6):
        center = lerp(float(sweep["start"]), float(sweep["end"]), float(fraction))
        env = gaussian_envelope(times, center, 0.07 if not QUICK_MODE else 0.10)
        note = 190 + 210 * fraction
        audio += env * (0.020 * np.sin(math.tau * note * times))

    for scene_name, strength in [("seasons", 0.08), ("strongest_day", 0.12), ("annual_mean", 0.10), ("finale", 0.15)]:
        shot = next(s for s in SHOT_PLAN if s["name"] == scene_name)
        center = float(shot["start"]) + 0.30 * (float(shot["end"]) - float(shot["start"]))
        env = gaussian_envelope(times, center, 0.68 if not QUICK_MODE else 0.22)
        audio += env * strength * np.sin(math.tau * 44.0 * times)

    intro_x = np.clip(times / max(1.5, duration * 0.07), 0, 1)
    outro_x = np.clip((times - (duration - 1.1)) / 0.9, 0, 1)
    intro = intro_x * intro_x * (3 - 2 * intro_x)
    outro = 1 - outro_x * outro_x * (3 - 2 * outro_x)
    audio *= intro * outro
    peak = max(float(np.max(np.abs(audio))), 1e-9)
    pcm = (np.clip(audio / peak * 0.88, -1, 1) * 32767).astype(np.int16)
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
        ffmpeg, "-y", "-i", str(video_path), "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def render_video(scene: GlobalWindYearScene) -> Path:
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    raw_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_silent.mp4"
    final_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    audio_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_ambient.wav"
    frame_count = int(round(float(CONFIG["duration_s"]) * int(CONFIG["fps"])))
    times = np.arange(frame_count) / int(CONFIG["fps"])
    print("Subtitle sidecar:", srt_path.resolve())
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(raw_video, fps=int(CONFIG["fps"]), codec="libx264", quality=8,
                        pixelformat="yuv420p", macro_block_size=None) as writer:
        for t in tqdm(times, desc="Rendering annual global-wind short"):
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


