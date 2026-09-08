from __future__ import annotations

"""
EARTH'S TEMPERATURE ANOMALY SINCE 1880 — cinematic YouTube Shorts renderer

Creates a vertical 1080x1920 data-driven short following Earth's global surface-
temperature anomaly from 1880 through the latest complete year in NASA GISTEMP.
The map and the global anomaly curve advance together so the viewer sees both the
spatial pattern and the changing planet-wide mean.

Production pattern matches the earlier Shorts renderers:
- NASA GISTEMP download first
- cached NetCDF second
- deterministic synthetic fixture third
- quick-preview mode
- preview PNGs
- CSV / JSON exports
- SRT captions
- generated ambient soundtrack
- final MP4 when ffmpeg is available

REAL DATA
---------
Primary source: NASA GISS GISTEMP v4 Land-Ocean Temperature Index, GHCN v4 +
ERSST v5, 1200 km smoothing, regular 2°x2° NetCDF grid. Monthly temperature
anomalies are relative to the 1951-1980 mean. This renderer averages complete
calendar years into annual maps and computes an area-weighted global anomaly.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm scipy

Quick preview
-------------
    TEMP_ANOMALY_QUICK=1 python earths_temperature_anomaly_since_1880.py

Force offline fixture
---------------------
    TEMP_ANOMALY_OFFLINE=1 python earths_temperature_anomaly_since_1880.py

Choose ending year
------------------
    TEMP_ANOMALY_END_YEAR=2024 python earths_temperature_anomaly_since_1880.py


"""

import gzip
import io
import json
import math
import os
import shutil
import subprocess
import urllib.request
import wave
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

QUICK_MODE = os.environ.get("TEMP_ANOMALY_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("TEMP_ANOMALY_OFFLINE", "0") == "1"
START_YEAR = 1880
DEFAULT_END_YEAR = 2025
END_YEAR = int(os.environ.get("TEMP_ANOMALY_END_YEAR", str(DEFAULT_END_YEAR)))
END_YEAR = max(START_YEAR + 10, min(END_YEAR, 2100))
BASELINE_TEXT = "1951-1980"

OUTPUT_ROOT = Path("earths_temperature_anomaly_since_1880_output")
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
    "title": "EARTH'S TEMPERATURE ANOMALY SINCE 1880",
    "subtitle": "NASA GISTEMP V4 // GLOBAL SURFACE ANOMALY VS 1951-1980",
    "output_basename": "earths_temperature_anomaly_since_1880",
    "map_margin_x": 22 if QUICK_MODE else 44,
    "map_top": 150 if QUICK_MODE else 300,
    "map_bottom": 660 if QUICK_MODE else 1320,
    "grain_strength": 3.0,
    "contrast": 1.08,
    "saturation": 1.07,
    "vignette": 0.34,
    "sample_rate": 22050 if QUICK_MODE else 44100,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0

COLORS = {
    "bg_top": (4, 7, 15),
    "bg_bottom": (1, 2, 7),
    "white": (247, 249, 250),
    "muted": (161, 181, 190),
    "grid": (96, 118, 128),
    "cold2": (34, 69, 145),
    "cold1": (61, 147, 207),
    "near": (104, 178, 190),
    "neutral": (38, 43, 48),
    "warm1": (240, 196, 91),
    "warm2": (244, 126, 71),
    "hot": (219, 62, 68),
    "extreme": (190, 70, 148),
}

FULL_SHOT_PLAN = [
    {"name": "opening", "start": 0.0, "end": 5.5},
    {"name": "history", "start": 5.5, "end": 35.5},
    {"name": "landmarks", "start": 35.5, "end": 43.5},
    {"name": "curve", "start": 43.5, "end": 51.5},
    {"name": "recent", "start": 51.5, "end": 56.0},
    {"name": "finale", "start": 56.0, "end": 58.0},
]

FULL_CAPTIONS = [
    (0.4, 5.2, "This is Earth's temperature anomaly record, beginning in 1880."),
    (5.8, 35.1, "Each year is compared with NASA GISTEMP's 1951 to 1980 average. The map and the global mean move forward together."),
    (35.8, 43.1, "Pause at a few points in the record and the long-term shift becomes easier to see."),
    (43.8, 51.1, "The year-to-year line rises and falls, but the long-term direction is clear."),
    (51.8, 55.7, "Recent years sit near the warmest end of the modern instrumental record."),
    (56.1, 57.8, "From 1880 to now, Earth's temperature anomaly has moved decisively upward."),
]

if QUICK_MODE:
    k = float(CONFIG["duration_s"]) / 58.0
    SHOT_PLAN = [{**s, "start": s["start"] * k, "end": s["end"] * k} for s in FULL_SHOT_PLAN]
    CAPTIONS = [(a * k, b * k, text) for a, b, text in FULL_CAPTIONS]
else:
    SHOT_PLAN = FULL_SHOT_PLAN
    CAPTIONS = FULL_CAPTIONS

GISTEMP_GZ_URL = "https://data.giss.nasa.gov/pub/gistemp/gistemp1200_GHCNv4_ERSSTv5.nc.gz"


# -----------------------------------------------------------------------------
# Helpers / models
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TemperatureDataset:
    years: np.ndarray             # [year]
    lat: np.ndarray               # [lat]
    lon: np.ndarray               # [lon]
    annual: np.ndarray            # [year, lat, lon] deg C anomaly
    source: str
    notes: List[str]


@dataclass(frozen=True)
class YearStats:
    year: int
    global_mean_c: float
    north_mean_c: float
    tropics_mean_c: float
    south_mean_c: float
    warm_fraction: float


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(x: float) -> float:
    x = clamp(x)
    return x * x * (3.0 - 2.0 * x)


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if float(shot["start"]) <= t < float(shot["end"]):
            return shot
    return SHOT_PLAN[-1]


def shot_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - float(shot["start"])) / max(float(shot["end"] - shot["start"]), 1e-9))


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
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(image: Image.Image, text: str, xy: Tuple[int, int], size: int,
              fill: Tuple[int, int, int, int], bold: bool = False,
              condensed: bool = False, anchor: str = "la", stroke: int = 2):
    ImageDraw.Draw(image).text(
        xy, text, font=get_font(size, bold, condensed), fill=fill, anchor=anchor,
        stroke_width=stroke, stroke_fill=(0, 0, 0, min(235, fill[3]))
    )


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for i, (a, b, text) in enumerate(captions, 1):
        lines.extend([str(i), f"{format_srt_time(a)} --> {format_srt_time(b)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    r = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * r**1.75, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# NASA GISTEMP loading
# -----------------------------------------------------------------------------

def request_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; TemperatureAnomalyShort/1.0; educational visualization)",
        "Accept": "application/gzip,application/octet-stream,*/*",
    })
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def decode_temp_variable(var) -> np.ndarray:
    raw = np.asarray(var[:])
    fill = getattr(var, "_FillValue", None)
    missing = getattr(var, "missing_value", None)
    arr = raw.astype(np.float32)
    bad_mask = np.zeros(arr.shape, dtype=bool)
    for bad in (fill, missing):
        if bad is not None:
            try:
                bad_mask |= raw == np.asarray(bad).astype(raw.dtype)
            except Exception:
                pass
    scale = float(getattr(var, "scale_factor", 1.0))
    offset = float(getattr(var, "add_offset", 0.0))
    arr = arr * scale + offset
    arr[bad_mask] = np.nan
    arr[np.abs(arr) > 30.0] = np.nan
    return arr


def parse_netcdf(path: Path) -> TemperatureDataset:
    with netcdf_file(str(path), "r", mmap=False) as nc:
        if "tempanomaly" not in nc.variables:
            raise RuntimeError("GISTEMP NetCDF is missing tempanomaly")
        lat = np.asarray(nc.variables["lat"][:], dtype=np.float32).copy()
        lon = np.asarray(nc.variables["lon"][:], dtype=np.float32).copy()
        time_values = np.asarray(nc.variables["time"][:], dtype=np.float64).copy()
        var = nc.variables["tempanomaly"]
        monthly = decode_temp_variable(var)
        dims = [str(x).lower() for x in getattr(var, "dimensions", ())]

    if dims and all(name in dims for name in ("time", "lat", "lon")):
        monthly = np.transpose(monthly, (dims.index("time"), dims.index("lat"), dims.index("lon")))
    elif monthly.shape[0] != len(time_values):
        axes = list(monthly.shape)
        try:
            t_axis = axes.index(len(time_values))
            la_axis = axes.index(len(lat))
            lo_axis = axes.index(len(lon))
            monthly = np.transpose(monthly, (t_axis, la_axis, lo_axis))
        except Exception as exc:
            raise RuntimeError(f"Could not resolve GISTEMP variable dimensions: {monthly.shape}") from exc

    if lat[0] > lat[-1]:
        lat = lat[::-1].copy()
        monthly = monthly[:, ::-1, :]
    if lon.min() >= 0 and lon.max() > 180:
        wrapped = ((lon + 180.0) % 360.0) - 180.0
        order = np.argsort(wrapped)
        lon = wrapped[order].astype(np.float32)
        monthly = monthly[:, :, order]

    origin = datetime(1800, 1, 1)
    dates = [(origin + timedelta(days=float(v))).date() for v in time_values]
    year_to_indices: Dict[int, List[int]] = {}
    for i, d in enumerate(dates):
        year_to_indices.setdefault(d.year, []).append(i)

    years: List[int] = []
    annual: List[np.ndarray] = []
    for year in sorted(year_to_indices):
        if year < START_YEAR or year > END_YEAR:
            continue
        ids = year_to_indices[year]
        if len(ids) < 12:
            continue
        years.append(year)
        annual.append(np.nanmean(monthly[ids[:12]], axis=0).astype(np.float32))
    if len(years) < 20:
        raise RuntimeError("Too few complete annual maps parsed from GISTEMP")
    return TemperatureDataset(
        years=np.asarray(years, dtype=np.int32), lat=lat, lon=lon,
        annual=np.stack(annual).astype(np.float32), source="nasa_giss_gistemp_v4_live_or_cached",
        notes=["Annual maps are means of complete monthly GISTEMP v4 fields", "Anomalies are relative to 1951-1980"],
    )


def make_synthetic_dataset() -> TemperatureDataset:
    years = np.arange(START_YEAR, END_YEAR + 1, dtype=np.int32)
    lat = np.arange(-89.0, 90.0, 4.0 if QUICK_MODE else 2.0, dtype=np.float32)
    lon = np.arange(-179.0, 180.0, 4.0 if QUICK_MODE else 2.0, dtype=np.float32)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    rng = np.random.default_rng(18802025)
    maps: List[np.ndarray] = []
    noise = rng.normal(0.0, 0.08, size=(len(lat), len(lon)))
    for year in years:
        x = year - 1880
        recent = max(0, year - 1970)
        if year <= 1970:
            trend = -0.22 + 0.0019 * x
        else:
            trend = -0.049 + 0.015 * recent + 0.0002 * recent * recent
        polar = (np.abs(lat2d) / 90.0) ** 1.8
        north_boost = np.clip(lat2d / 90.0, 0, 1) ** 2
        spatial = trend * (0.74 + 0.38 * polar + 0.38 * north_boost)
        pacific = 0.22 * math.sin((year - 1880) * 0.83) * np.exp(-((lat2d / 17.0) ** 2)) * np.cos(np.deg2rad(lon2d + 155))
        waves = 0.08 * np.sin(np.deg2rad(lon2d * 2.0 + lat2d * 1.4 + year * 1.7))
        maps.append((spatial + pacific + waves + noise * (0.65 + 0.35 * math.sin(year * 0.13))).astype(np.float32))
    return TemperatureDataset(
        years=years, lat=lat, lon=lon, annual=np.stack(maps),
        source="synthetic_procedural_fixture",
        notes=["Deterministic synthetic warming fixture for preview/timing only", "Not observational data"],
    )


def load_temperature_data() -> TemperatureDataset:
    cache_nc = CACHE_ROOT / "gistemp1200_GHCNv4_ERSSTv5.nc"
    cache_gz = CACHE_ROOT / "gistemp1200_GHCNv4_ERSSTv5.nc.gz"
    notes: List[str] = []
    if OFFLINE_MODE:
        return make_synthetic_dataset()

    try:
        payload = request_bytes(GISTEMP_GZ_URL)
        if len(payload) < 1_000_000:
            raise RuntimeError(f"GISTEMP download unexpectedly small ({len(payload)} bytes)")
        cache_gz.write_bytes(payload)
        cache_nc.write_bytes(gzip.decompress(payload))
        ds = parse_netcdf(cache_nc)
        return TemperatureDataset(ds.years, ds.lat, ds.lon, ds.annual, "nasa_giss_gistemp_v4_live", ["Downloaded current NASA GISTEMP v4 grid"] + ds.notes)
    except Exception as exc:
        notes.append(f"Live GISTEMP download failed: {exc}")

    if cache_nc.exists() and cache_nc.stat().st_size > 1_000_000:
        try:
            ds = parse_netcdf(cache_nc)
            return TemperatureDataset(ds.years, ds.lat, ds.lon, ds.annual, "nasa_giss_gistemp_v4_cached", notes + ["Loaded cached NASA GISTEMP NetCDF"] + ds.notes)
        except Exception as exc:
            notes.append(f"Cached GISTEMP parse failed: {exc}")

    fixture = make_synthetic_dataset()
    return TemperatureDataset(fixture.years, fixture.lat, fixture.lon, fixture.annual, fixture.source, notes + fixture.notes)


# -----------------------------------------------------------------------------
# Statistics / exports
# -----------------------------------------------------------------------------

def weighted_mean(field: np.ndarray, lat: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    weights = np.cos(np.deg2rad(lat.astype(float)))[:, None] * np.ones((1, field.shape[1]))
    valid = np.isfinite(field)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        return float("nan")
    return float(np.sum(field[valid] * weights[valid]) / np.sum(weights[valid]))


def compute_stats(data: TemperatureDataset) -> List[YearStats]:
    lat2d = np.repeat(data.lat[:, None], len(data.lon), axis=1)
    stats: List[YearStats] = []
    for i, year in enumerate(data.years):
        field = data.annual[i]
        valid = np.isfinite(field)
        stats.append(YearStats(
            year=int(year),
            global_mean_c=weighted_mean(field, data.lat),
            north_mean_c=weighted_mean(field, data.lat, lat2d >= 60),
            tropics_mean_c=weighted_mean(field, data.lat, np.abs(lat2d) <= 30),
            south_mean_c=weighted_mean(field, data.lat, lat2d <= -60),
            warm_fraction=float(np.mean(field[valid] > 0.0)) if np.any(valid) else float("nan"),
        ))
    return stats


def save_data_products(data: TemperatureDataset, stats: Sequence[YearStats]) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "annual_temperature_stats.csv"
    json_path = DATA_ROOT / "temperature_anomaly_summary.json"
    pd.DataFrame([{
        "year": s.year,
        "global_mean_anomaly_c": s.global_mean_c,
        "north_of_60n_mean_c": s.north_mean_c,
        "tropics_30s_30n_mean_c": s.tropics_mean_c,
        "south_of_60s_mean_c": s.south_mean_c,
        "fraction_of_valid_grid_cells_above_baseline": s.warm_fraction,
    } for s in stats]).to_csv(csv_path, index=False)
    recent = stats[-1]
    summary = {
        "title": CONFIG["title"],
        "data_source": data.source,
        "start_year": int(data.years[0]),
        "end_year": int(data.years[-1]),
        "grid": {"lat_cells": len(data.lat), "lon_cells": len(data.lon)},
        "baseline": BASELINE_TEXT,
        "latest_global_mean_anomaly_c": recent.global_mean_c,
        "latest_warm_grid_fraction": recent.warm_fraction,
        "source_notes": data.notes,
        "source_urls": {"gistemp": "https://data.giss.nasa.gov/gistemp/", "netcdf": GISTEMP_GZ_URL},
        "visualization_note": "Pixel colors show annual gridded temperature anomalies, not absolute temperature.",
        "fallback_warning": "synthetic_procedural_fixture is preview data, not observational data",
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path


# -----------------------------------------------------------------------------
# Color / simple map geometry
# -----------------------------------------------------------------------------

COLOR_STOPS = [
    (-4.0, COLORS["cold2"]), (-1.5, COLORS["cold1"]), (-0.4, COLORS["near"]),
    (0.0, COLORS["neutral"]), (0.5, COLORS["warm1"]), (1.0, COLORS["warm2"]),
    (2.0, COLORS["hot"]), (4.0, COLORS["extreme"]),
]


def anomaly_to_rgb(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.zeros(vals.shape + (3,), dtype=np.float32)
    valid = np.isfinite(vals)
    out[:] = np.array((10, 13, 18), dtype=float)
    for i in range(len(COLOR_STOPS) - 1):
        a, ca = COLOR_STOPS[i]
        b, cb = COLOR_STOPS[i + 1]
        mask = valid & (vals >= a) & (vals <= b)
        if not np.any(mask):
            continue
        u = np.clip((vals[mask] - a) / (b - a), 0, 1)[:, None]
        out[mask] = np.array(ca) * (1 - u) + np.array(cb) * u
    out[valid & (vals < COLOR_STOPS[0][0])] = COLOR_STOPS[0][1]
    out[valid & (vals > COLOR_STOPS[-1][0])] = COLOR_STOPS[-1][1]
    return np.clip(out, 0, 255).astype(np.uint8)


BUILTIN_LAND_POLYGONS: List[List[Tuple[float, float]]] = [
    [(-168,72),(-140,70),(-124,55),(-126,42),(-115,30),(-101,20),(-83,8),(-77,18),(-82,25),(-80,32),(-66,46),(-52,56),(-75,72),(-168,72)],
    [(-82,12),(-70,12),(-53,4),(-35,-8),(-42,-25),(-58,-55),(-72,-50),(-80,-20),(-82,12)],
    [(-17,37),(5,36),(34,31),(50,12),(42,-12),(33,-35),(18,-35),(6,-12),(-15,12),(-17,37)],
    [(-10,72),(40,72),(78,66),(110,56),(145,50),(180,64),(180,8),(140,6),(122,22),(105,5),(78,8),(52,28),(28,38),(10,45),(-10,58),(-10,72)],
    [(112,-10),(154,-12),(154,-39),(138,-45),(116,-35),(112,-10)],
    [(-74,59),(-44,83),(-18,72),(-35,59),(-74,59)],
    [(-180,-62),(-120,-70),(-60,-65),(0,-72),(60,-66),(120,-72),(180,-62)],
]


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class TemperatureAnomalyScene:
    def __init__(self, data: TemperatureDataset, stats: Sequence[YearStats]):
        self.data = data
        self.stats = list(stats)
        self.year_lookup = {int(y): i for i, y in enumerate(data.years)}
        self.map_box = (
            int(CONFIG["map_margin_x"]), int(CONFIG["map_top"]),
            OUT_W - int(CONFIG["map_margin_x"]), int(CONFIG["map_bottom"]),
        )
        self.rng = np.random.default_rng(18802025)
        self.stars = [
            (float(self.rng.uniform(0, OUT_W)), float(self.rng.uniform(0, OUT_H)),
             float(self.rng.uniform(0.4, 1.8)), float(self.rng.uniform(8, 38)))
            for _ in range(110 if not QUICK_MODE else 40)
        ]
        self.annual_layers = [
            self._render_custom_field(data.annual[i])
            for i in tqdm(range(len(data.years)), desc="Building annual anomaly maps", leave=False)
        ]
        vals = np.array([s.global_mean_c for s in self.stats], dtype=float)
        finite = vals[np.isfinite(vals)]
        self.chart_min = min(-0.7, float(np.nanmin(finite)) - 0.12) if len(finite) else -0.7
        self.chart_max = max(1.5, float(np.nanmax(finite)) + 0.12) if len(finite) else 1.5

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
        top = np.array(COLORS["bg_top"], dtype=float)
        bottom = np.array(COLORS["bg_bottom"], dtype=float)
        for y in range(OUT_H):
            u = y / max(OUT_H - 1, 1)
            arr[y, :, :3] = (top * (1-u) + bottom * u).astype(np.uint8)
            arr[y, :, 3] = 255
        im = Image.fromarray(arr, "RGBA")
        dr = ImageDraw.Draw(im)
        for x, y, r, a in self.stars:
            pulse = 0.5 + 0.5 * math.sin(t * 0.55 + x * 0.012 + y * 0.008) ** 2
            rr = r * SCALE
            dr.ellipse((x-rr, y-rr, x+rr, y+rr), fill=(210, 224, 230, int(a * pulse)))
        return im

    def project(self, lon: float, lat: float, box: Optional[Tuple[int,int,int,int]] = None) -> Tuple[float,float]:
        x0, y0, x1, y1 = box or self.map_box
        return x0 + ((lon + 180) / 360) * (x1-x0), y0 + ((90-lat) / 180) * (y1-y0)

    def _render_custom_field(self, field: np.ndarray) -> Image.Image:
        rgb = anomaly_to_rgb(np.flipud(field))
        native = Image.fromarray(rgb, "RGB").convert("RGBA")
        x0, y0, x1, y1 = self.map_box
        return native.resize((x1-x0, y1-y0), Image.Resampling.NEAREST)

    def draw_map_frame(self, image: Image.Image, layer: Image.Image, alpha: int = 255,
                       box: Optional[Tuple[int,int,int,int]] = None):
        box = box or self.map_box
        x0, y0, x1, y1 = box
        lay = layer.copy() if box == self.map_box else layer.resize((x1-x0, y1-y0), Image.Resampling.NEAREST)
        if alpha < 255:
            lay.putalpha(alpha)
        image.alpha_composite(lay, (x0, y0))
        dr = ImageDraw.Draw(image)
        for lon in range(-180, 181, 60):
            x, _ = self.project(lon, 0, box)
            dr.line((x, y0, x, y1), fill=COLORS["grid"]+(38,), width=max(1, int(SCALE)))
        for lat in range(-60, 61, 30):
            _, y = self.project(0, lat, box)
            dr.line((x0, y, x1, y), fill=COLORS["grid"]+(38,), width=max(1, int(SCALE)))
        for poly in BUILTIN_LAND_POLYGONS:
            pts = [self.project(lon, lat, box) for lon, lat in poly]
            if len(pts) >= 2:
                dr.line(pts, fill=(235, 242, 245, 95), width=max(1, int(1.2*SCALE)), joint="curve")
        dr.rectangle(box, outline=(185, 208, 215, 100), width=max(1, int(2*SCALE)))

    def draw_header(self, image: Image.Image, scene_name: str):
        draw_text(image, CONFIG["title"], (int(48*SCALE), int(72*SCALE)), 31 if not QUICK_MODE else 15,
                  COLORS["white"]+(245,), True, True, stroke=2)
        if scene_name != "opening":
            draw_text(image, CONFIG["subtitle"], (int(50*SCALE), int(122*SCALE)), 12 if not QUICK_MODE else 6,
                      COLORS["muted"]+(205,), True, True, stroke=1)

    def draw_legend(self, image: Image.Image, y: int):
        x0 = int(90*SCALE); x1 = OUT_W-int(90*SCALE); h = max(8, int(20*SCALE))
        vals = np.linspace(-2.0, 2.5, max(2, x1-x0))[None, :]
        strip = Image.fromarray(anomaly_to_rgb(vals), "RGB").resize((x1-x0, h)).convert("RGBA")
        image.alpha_composite(strip, (x0, y))
        draw_text(image, "COOLER", (x0, y+int(39*SCALE)), 11 if not QUICK_MODE else 5,
                  COLORS["muted"]+(215,), True, True, "la", 1)
        draw_text(image, "BASELINE", ((x0+x1)//2, y+int(39*SCALE)), 11 if not QUICK_MODE else 5,
                  COLORS["muted"]+(215,), True, True, "ma", 1)
        draw_text(image, "WARMER  °C", (x1, y+int(39*SCALE)), 11 if not QUICK_MODE else 5,
                  COLORS["muted"]+(215,), True, True, "ra", 1)

    def draw_curve(self, image: Image.Image, upto_idx: int, box: Tuple[int,int,int,int],
                   emphasize: bool = False, label_current: bool = True):
        x0, y0, x1, y1 = box
        dr = ImageDraw.Draw(image)
        dr.rounded_rectangle(box, radius=max(4, int(14*SCALE)), fill=(10, 15, 22, 215),
                             outline=(140, 166, 176, 70), width=max(1, int(2*SCALE)))
        left = x0 + int(42*SCALE); right = x1 - int(22*SCALE)
        top = y0 + int(24*SCALE); bottom = y1 - int(38*SCALE)

        def x_for(i: int) -> float:
            return lerp(left, right, i / max(len(self.stats)-1, 1))
        def y_for(v: float) -> float:
            return lerp(bottom, top, (v-self.chart_min) / max(self.chart_max-self.chart_min, 1e-9))

        zero_y = y_for(0.0)
        dr.line((left, zero_y, right, zero_y), fill=(210, 220, 224, 75), width=max(1, int(2*SCALE)))
        for val in (-0.5, 0.5, 1.0, 1.5):
            if self.chart_min <= val <= self.chart_max:
                yy = y_for(val)
                dr.line((left, yy, right, yy), fill=(130, 150, 160, 28), width=1)
        pts = []
        for i in range(min(upto_idx, len(self.stats)-1)+1):
            v = self.stats[i].global_mean_c
            if np.isfinite(v): pts.append((x_for(i), y_for(v)))
        if len(pts) >= 2:
            glow = Image.new("RGBA", OUT_SIZE, (0,0,0,0)); gd = ImageDraw.Draw(glow)
            gd.line(pts, fill=COLORS["warm2"]+(80,), width=max(2, int((10 if emphasize else 7)*SCALE)), joint="curve")
            glow = glow.filter(ImageFilter.GaussianBlur(max(1, int(6*SCALE))))
            image.alpha_composite(glow)
            dr = ImageDraw.Draw(image)
            dr.line(pts, fill=COLORS["warm1"]+(245,), width=max(1, int((4 if emphasize else 3)*SCALE)), joint="curve")
        if upto_idx >= 0:
            cx = x_for(upto_idx); cy = y_for(self.stats[upto_idx].global_mean_c)
            rr = max(3, int((8 if emphasize else 6)*SCALE))
            dr.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=COLORS["white"]+(245,), outline=COLORS["warm2"]+(255,), width=max(1,int(2*SCALE)))
            if label_current:
                sign = "+" if self.stats[upto_idx].global_mean_c >= 0 else ""
                draw_text(image, f"{int(self.data.years[upto_idx])}  {sign}{self.stats[upto_idx].global_mean_c:.2f}°C",
                          (int(cx), int(cy-int(24*SCALE))), 12 if not QUICK_MODE else 6,
                          COLORS["white"]+(245,), True, True, "ma", 1)
        draw_text(image, str(int(self.data.years[0])), (left, y1-int(12*SCALE)), 10 if not QUICK_MODE else 5,
                  COLORS["muted"]+(210,), True, True, "la", 1)
        draw_text(image, str(int(self.data.years[-1])), (right, y1-int(12*SCALE)), 10 if not QUICK_MODE else 5,
                  COLORS["muted"]+(210,), True, True, "ra", 1)
        draw_text(image, "0°C", (left-int(10*SCALE), int(zero_y)), 10 if not QUICK_MODE else 5,
                  COLORS["muted"]+(205,), True, True, "ra", 1)

    def draw_meter(self, image: Image.Image, value: float, x: int, y0: int, y1: int):
        dr = ImageDraw.Draw(image)
        w = max(8, int(30*SCALE)); center = x
        dr.rounded_rectangle((center-w//2, y0, center+w//2, y1), radius=w//2,
                             fill=(22, 28, 35, 230), outline=(195, 212, 218, 110), width=max(1,int(2*SCALE)))
        lo, hi = self.chart_min, self.chart_max
        frac = clamp((value-lo)/max(hi-lo, 1e-9))
        yy = int(lerp(y1, y0, frac))
        fill_col = COLORS["warm2"] if value >= 0 else COLORS["cold1"]
        dr.rounded_rectangle((center-w//2+int(4*SCALE), yy, center+w//2-int(4*SCALE), y1-int(4*SCALE)),
                             radius=max(2,w//3), fill=fill_col+(220,))
        zero_y = int(lerp(y1, y0, clamp((0-lo)/max(hi-lo,1e-9))))
        dr.line((center-w, zero_y, center+w, zero_y), fill=COLORS["white"]+(150,), width=max(1,int(2*SCALE)))

    def draw_opening(self, image: Image.Image, t: float, shot: Dict[str,Any]):
        p = smoothstep(shot_progress(t, shot))
        split = OUT_W//2
        left_box = (int(42*SCALE), int(330*SCALE), split-int(9*SCALE), int(1220*SCALE))
        right_box = (split+int(9*SCALE), int(330*SCALE), OUT_W-int(42*SCALE), int(1220*SCALE))
        self.draw_map_frame(image, self.annual_layers[0], alpha=int(120+120*p), box=left_box)
        self.draw_map_frame(image, self.annual_layers[-1], alpha=int(70+170*p), box=right_box)
        draw_text(image, "1880", ((left_box[0]+left_box[2])//2, int(1280*SCALE)), 30 if not QUICK_MODE else 15,
                  COLORS["white"]+(245,), True, True, "ma", 2)
        draw_text(image, str(int(self.data.years[-1])), ((right_box[0]+right_box[2])//2, int(1280*SCALE)), 30 if not QUICK_MODE else 15,
                  COLORS["white"]+(245,), True, True, "ma", 2)
        first = self.stats[0]; last = self.stats[-1]
        draw_text(image, f"{first.global_mean_c:+.2f}°C  →  {last.global_mean_c:+.2f}°C",
                  (OUT_W//2, int(1450*SCALE)), 42 if not QUICK_MODE else 21,
                  COLORS["warm1"]+(250,), True, True, "ma", 2)
        draw_text(image, "GLOBAL SURFACE TEMPERATURE ANOMALY", (OUT_W//2, int(1525*SCALE)), 16 if not QUICK_MODE else 8,
                  COLORS["muted"]+(225,), True, True, "ma", 1)
        self.draw_legend(image, int(1630*SCALE))

    def draw_history(self, image: Image.Image, t: float, shot: Dict[str,Any]):
        p = smoothstep(shot_progress(t, shot))
        idx = min(len(self.data.years)-1, int(round(p*(len(self.data.years)-1))))
        self.draw_map_frame(image, self.annual_layers[idx])
        stat = self.stats[idx]
        draw_text(image, str(int(self.data.years[idx])), (int(72*SCALE), int(1390*SCALE)), 72 if not QUICK_MODE else 36,
                  COLORS["white"]+(250,), True, True, "la", 3)
        sign = "+" if stat.global_mean_c >= 0 else ""
        draw_text(image, f"{sign}{stat.global_mean_c:.2f}°C", (OUT_W-int(92*SCALE), int(1400*SCALE)), 43 if not QUICK_MODE else 21,
                  COLORS["warm1"]+(250,), True, True, "ra", 2)
        self.draw_meter(image, stat.global_mean_c, OUT_W-int(55*SCALE), int(1490*SCALE), int(1700*SCALE))
        self.draw_curve(image, idx, (int(70*SCALE), int(1480*SCALE), OUT_W-int(95*SCALE), int(1740*SCALE)))
        self.draw_legend(image, int(1780*SCALE))

    def draw_landmarks(self, image: Image.Image, t: float, shot: Dict[str,Any]):
        candidates = [1880, 1940, 1980, 2000, int(self.data.years[-1])]
        years = [int(min(self.data.years, key=lambda y: abs(int(y)-c))) for c in candidates]
        p = smoothstep(shot_progress(t, shot)); reveal = max(1, min(len(years), 1+int(p*len(years))))
        margin = int(52*SCALE); top = int(330*SCALE); row_h = int(236*SCALE)
        for j, year in enumerate(years[:reveal]):
            idx = self.year_lookup[year]; st = self.stats[idx]
            y = top + j*row_h
            box = (margin, y, margin+int(330*SCALE), y+int(195*SCALE))
            self.draw_map_frame(image, self.annual_layers[idx], box=box)
            draw_text(image, str(year), (margin+int(365*SCALE), y+int(48*SCALE)), 30 if not QUICK_MODE else 15,
                      COLORS["white"]+(250,), True, True, "la", 2)
            draw_text(image, f"{st.global_mean_c:+.2f}°C", (OUT_W-margin, y+int(48*SCALE)), 25 if not QUICK_MODE else 12,
                      COLORS["warm1"]+(245,), True, True, "ra", 2)
            dr = ImageDraw.Draw(image)
            x0 = margin+int(365*SCALE); x1 = OUT_W-margin; yy = y+int(105*SCALE)
            dr.line((x0, yy, x1, yy), fill=(130,150,160,65), width=max(1,int(2*SCALE)))
            zero = lerp(x0, x1, clamp((0-self.chart_min)/(self.chart_max-self.chart_min)))
            dr.line((zero, yy-int(12*SCALE), zero, yy+int(12*SCALE)), fill=COLORS["white"]+(110,), width=max(1,int(2*SCALE)))
            marker = lerp(x0, x1, clamp((st.global_mean_c-self.chart_min)/(self.chart_max-self.chart_min)))
            rr=max(3,int(7*SCALE)); dr.ellipse((marker-rr,yy-rr,marker+rr,yy+rr),fill=COLORS["warm2"]+(245,))
        draw_text(image, "SAME BASELINE. DIFFERENT MOMENTS.", (OUT_W//2, int(1590*SCALE)), 20 if not QUICK_MODE else 10,
                  COLORS["muted"]+(225,), True, True, "ma", 1)

    def draw_curve_scene(self, image: Image.Image, t: float, shot: Dict[str,Any]):
        p = smoothstep(shot_progress(t, shot))
        upto = min(len(self.stats)-1, int(round(p*(len(self.stats)-1))))
        self.draw_curve(image, upto, (int(58*SCALE), int(400*SCALE), OUT_W-int(58*SCALE), int(1320*SCALE)), emphasize=True)
        idx = upto
        inset = (int(650*SCALE), int(1390*SCALE), OUT_W-int(60*SCALE), int(1710*SCALE))
        self.draw_map_frame(image, self.annual_layers[idx], box=inset)
        draw_text(image, "GLOBAL MEAN ANOMALY", (int(70*SCALE), int(1430*SCALE)), 22 if not QUICK_MODE else 11,
                  COLORS["white"]+(245,), True, True, "la", 2)
        draw_text(image, "YEAR-TO-YEAR VARIABILITY", (int(70*SCALE), int(1495*SCALE)), 13 if not QUICK_MODE else 6,
                  COLORS["muted"]+(220,), True, True, "la", 1)
        draw_text(image, "LONG-TERM RISE", (int(70*SCALE), int(1550*SCALE)), 20 if not QUICK_MODE else 10,
                  COLORS["warm1"]+(245,), True, True, "la", 2)

    def draw_recent(self, image: Image.Image, t: float, shot: Dict[str,Any]):
        recent_count = min(12, len(self.data.years))
        start = len(self.data.years)-recent_count
        p = smoothstep(shot_progress(t, shot)); idx = start + min(recent_count-1, int(p*recent_count))
        self.draw_map_frame(image, self.annual_layers[idx])
        draw_text(image, "RECENT YEARS", (OUT_W//2, int(1420*SCALE)), 26 if not QUICK_MODE else 13,
                  COLORS["white"]+(245,), True, True, "ma", 2)
        draw_text(image, str(int(self.data.years[idx])), (OUT_W//2, int(1510*SCALE)), 64 if not QUICK_MODE else 32,
                  COLORS["white"]+(250,), True, True, "ma", 3)
        draw_text(image, f"{self.stats[idx].global_mean_c:+.2f}°C", (OUT_W//2, int(1595*SCALE)), 31 if not QUICK_MODE else 15,
                  COLORS["warm1"]+(250,), True, True, "ma", 2)
        self.draw_curve(image, idx, (int(110*SCALE), int(1660*SCALE), OUT_W-int(110*SCALE), int(1815*SCALE)), label_current=False)

    def draw_finale(self, image: Image.Image, t: float, shot: Dict[str,Any]):
        p = smoothstep(shot_progress(t, shot))
        self.draw_map_frame(image, self.annual_layers[-1], alpha=int(160+95*p))
        last = self.stats[-1]
        draw_text(image, f"1880 → {int(self.data.years[-1])}", (OUT_W//2, int(1450*SCALE)), 56 if not QUICK_MODE else 28,
                  COLORS["white"]+(250,), True, True, "ma", 3)
        draw_text(image, f"LATEST COMPLETE YEAR  {last.global_mean_c:+.2f}°C", (OUT_W//2, int(1545*SCALE)), 21 if not QUICK_MODE else 10,
                  COLORS["warm1"]+(250,), True, True, "ma", 2)
        draw_text(image, "EARTH'S TEMPERATURE ANOMALY", (OUT_W//2, int(1620*SCALE)), 18 if not QUICK_MODE else 9,
                  COLORS["muted"]+(225,), True, True, "ma", 1)
        self.draw_curve(image, len(self.stats)-1, (int(110*SCALE), int(1690*SCALE), OUT_W-int(110*SCALE), int(1830*SCALE)), label_current=False)

    def draw_source_hud(self, image: Image.Image):
        text = "SYNTHETIC PREVIEW // NOT OBSERVATIONAL DATA" if self.data.source.startswith("synthetic") else "NASA GISS / GISTEMP V4 // GHCN V4 + ERSST V5"
        draw_text(image, text, (int(48*SCALE), OUT_H-int(42*SCALE)), 10 if not QUICK_MODE else 5,
                  COLORS["muted"]+(185,), True, True, "la", 1)

    def draw_film_texture(self, image: Image.Image, t: float):
        ov = Image.new("RGBA", OUT_SIZE, (0,0,0,0)); dr = ImageDraw.Draw(ov)
        off = int((t*43) % 9)
        for y in range(off, OUT_H, 9):
            dr.line((0, y, OUT_W, y), fill=(140,160,170,5), width=1)
        image.alpha_composite(ov)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t); name = str(shot["name"]); image = self.background(t)
        if name == "opening": self.draw_opening(image, t, shot)
        elif name == "history": self.draw_history(image, t, shot)
        elif name == "landmarks": self.draw_landmarks(image, t, shot)
        elif name == "curve": self.draw_curve_scene(image, t, shot)
        elif name == "recent": self.draw_recent(image, t, shot)
        else: self.draw_finale(image, t, shot)
        self.draw_header(image, name); self.draw_source_hud(image); self.draw_film_texture(image, t)
        arr = np.asarray(image.convert("RGB"), dtype=np.float32); arr *= VIGNETTE[...,None]; arr = np.clip(arr,0,255).astype(np.uint8)
        img = ImageEnhance.Contrast(Image.fromarray(arr)).enhance(float(CONFIG["contrast"])); img = ImageEnhance.Color(img).enhance(float(CONFIG["saturation"]))
        arr = np.asarray(img, dtype=np.int16); rng = np.random.default_rng(int(t*1000)+1880); grain = rng.normal(0.0,float(CONFIG["grain_strength"]),arr.shape[:2])[:,:,None]
        return np.clip(arr+grain,0,255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Audio / video
# -----------------------------------------------------------------------------

def gaussian_envelope(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5*((times-center)/max(width,1e-6))**2)


def generate_ambient_soundtrack(path: Path) -> Path:
    sr = int(CONFIG["sample_rate"]); dur = float(CONFIG["duration_s"]); n = int(round(sr*dur)); times = np.arange(n,dtype=np.float64)/sr
    rng = np.random.default_rng(18802025); audio = np.zeros(n,dtype=np.float64)
    audio += 0.050*np.sin(math.tau*31*times + 0.8*np.sin(math.tau*0.042*times))
    audio += 0.031*np.sin(math.tau*47*times + 1.1)
    controls = rng.normal(0,1,max(8,int(dur*4))); audio += 0.013*np.interp(times,np.linspace(0,dur,len(controls)),controls)
    sweep = next(s for s in SHOT_PLAN if s["name"]=="history")
    for frac in np.linspace(0.03,0.97,22 if not QUICK_MODE else 6):
        center = lerp(float(sweep["start"]),float(sweep["end"]),float(frac)); env = gaussian_envelope(times,center,0.055 if not QUICK_MODE else 0.10)
        audio += env*0.016*np.sin(math.tau*(145+285*frac)*times)
    for scene,strength in [("landmarks",0.07),("curve",0.11),("recent",0.10),("finale",0.15)]:
        sh = next(s for s in SHOT_PLAN if s["name"]==scene); c = float(sh["start"])+0.35*(float(sh["end"])-float(sh["start"]))
        audio += gaussian_envelope(times,c,0.62 if not QUICK_MODE else 0.20)*strength*np.sin(math.tau*42*times)
    intro = np.clip(times/max(1.5,dur*0.07),0,1); intro = intro*intro*(3-2*intro)
    outrox = np.clip((times-(dur-1.1))/0.9,0,1); outro = 1-outrox*outrox*(3-2*outrox); audio *= intro*outro
    peak = max(float(np.max(np.abs(audio))),1e-9); pcm = (np.clip(audio/peak*0.88,-1,1)*32767).astype(np.int16)
    with wave.open(str(path),"wb") as h:
        h.setnchannels(1); h.setsampwidth(2); h.setframerate(sr); h.writeframes(pcm.tobytes())
    return path


def find_ffmpeg()->Optional[str]:
    if imageio_ffmpeg is not None:
        try: return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception: pass
    return shutil.which("ffmpeg")


def mux_audio(video_path:Path,audio_path:Path,output_path:Path)->bool:
    ffmpeg=find_ffmpeg()
    if not ffmpeg: return False
    cmd=[ffmpeg,"-y","-i",str(video_path),"-i",str(audio_path),"-c:v","copy","-c:a","aac","-b:a","192k","-shortest",str(output_path)]
    try:
        subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return output_path.exists() and output_path.stat().st_size>0
    except Exception: return False


def render_video(scene:TemperatureAnomalyScene)->Path:
    srt=OUTPUT_ROOT/f"{CONFIG['output_basename']}.srt"; write_srt(CAPTIONS,srt)
    raw=OUTPUT_ROOT/f"{CONFIG['output_basename']}_silent.mp4"; final=OUTPUT_ROOT/f"{CONFIG['output_basename']}_final.mp4"; audio=OUTPUT_ROOT/f"{CONFIG['output_basename']}_ambient.wav"
    frames=int(round(float(CONFIG["duration_s"])*int(CONFIG["fps"]))); times=np.arange(frames)/int(CONFIG["fps"])
    print(f"Rendering {frames:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(raw,fps=int(CONFIG["fps"]),codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=None) as writer:
        for t in tqdm(times,desc="Rendering temperature-anomaly short"): writer.append_data(scene.render_frame(float(t)))
    generate_ambient_soundtrack(audio)
    if mux_audio(raw,audio,final): print("Final video with audio:",final.resolve()); return final
    shutil.copyfile(raw,final); print("ffmpeg audio mux unavailable; copied silent video to:",final.resolve()); return final


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


