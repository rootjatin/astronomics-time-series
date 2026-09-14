from __future__ import annotations

"""
THE PLANET IS GETTING WARMER PIXEL BY PIXEL — cinematic YouTube Shorts renderer

Creates a vertical 1080x1920 data-driven short showing global surface-temperature
anomalies changing across NASA GISTEMP's regular 2°x2° grid from 1880 through 2025.

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
ERSST v5, 1200 km smoothing, regular 2°x2° NetCDF grid. The file contains monthly
surface-temperature anomalies relative to the 1951-1980 mean. This renderer
averages complete calendar years into annual maps.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm scipy

Quick preview
-------------
    WARMING_PIXELS_QUICK=1 python the_planet_is_getting_warmer_pixel_by_pixel.py

Force offline fixture
---------------------
    WARMING_PIXELS_OFFLINE=1 python the_planet_is_getting_warmer_pixel_by_pixel.py

Choose ending year
------------------
    WARMING_PIXELS_END_YEAR=2024 python the_planet_is_getting_warmer_pixel_by_pixel.py

Outputs
-------
- final vertical MP4 with generated ambient audio when ffmpeg is available
- silent MP4 fallback
- SRT subtitle sidecar
- preview PNG frames
- CSV with annual global / hemispheric anomaly statistics
- JSON summary and source notes
- cached NASA GISTEMP NetCDF


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

QUICK_MODE = os.environ.get("WARMING_PIXELS_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("WARMING_PIXELS_OFFLINE", "0") == "1"
START_YEAR = 1880
DEFAULT_END_YEAR = 2025
END_YEAR = int(os.environ.get("WARMING_PIXELS_END_YEAR", str(DEFAULT_END_YEAR)))
END_YEAR = max(START_YEAR + 10, min(END_YEAR, 2100))
BASELINE_TEXT = "1951-1980"

OUTPUT_ROOT = Path("the_planet_is_getting_warmer_pixel_by_pixel_output")
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

]

FULL_CAPTIONS = [

]

if QUICK_MODE:
    k = float(CONFIG["duration_s"]) / 58.0
    SHOT_PLAN = [{**s, "start": s["start"] * k, "end": s["end"] * k} for s in FULL_SHOT_PLAN]
    CAPTIONS = [(a * k, b * k, text) for a, b, text in FULL_CAPTIONS]
else:
    SHOT_PLAN = FULL_SHOT_PLAN
    CAPTIONS = FULL_CAPTIONS



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
        "User-Agent": "Mozilla/5.0 (compatible; WarmingPixelsShort/1.0; educational visualization)",
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
    json_path = DATA_ROOT / "warming_pixels_summary.json"
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

class WarmingPixelsScene:
    def __init__(self, data: TemperatureDataset, stats: Sequence[YearStats]):
        self.data = data
        self.stats = list(stats)
        self.year_lookup = {int(y): i for i, y in enumerate(data.years)}
        self.map_box = (
            int(CONFIG["map_margin_x"]), int(CONFIG["map_top"]),
            OUT_W - int(CONFIG["map_margin_x"]), int(CONFIG["map_bottom"]),
        )
        self.rng = np.random.default_rng(90210)
        self.stars = [(float(self.rng.uniform(0, OUT_W)), float(self.rng.uniform(0, OUT_H)), float(self.rng.uniform(0.4, 1.8)), float(self.rng.uniform(8, 38))) for _ in range(120 if not QUICK_MODE else 45)]
        self.annual_layers = [self._render_field_layer(i) for i in tqdm(range(len(data.years)), desc="Building annual temperature maps", leave=False)]
        early_ids = [i for i, y in enumerate(data.years) if 1880 <= y <= 1900]
        recent_ids = [i for i, y in enumerate(data.years) if max(int(data.years[-1]) - 10, 2015) <= y <= int(data.years[-1])]
        self.early_mean = np.nanmean(data.annual[early_ids], axis=0)
        self.recent_mean = np.nanmean(data.annual[recent_ids], axis=0)
        self.diff_layer = self._render_custom_field(self.recent_mean - self.early_mean, difference=True)

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
            pulse = 0.5 + 0.5 * math.sin(t * 0.6 + x * 0.013 + y * 0.008) ** 2
            rr = r * SCALE
            dr.ellipse((x-rr, y-rr, x+rr, y+rr), fill=(210, 224, 230, int(a * pulse)))
        return im

    def project(self, lon: float, lat: float, box: Optional[Tuple[int,int,int,int]] = None) -> Tuple[float,float]:
        x0, y0, x1, y1 = box or self.map_box
        return x0 + ((lon + 180) / 360) * (x1-x0), y0 + ((90-lat) / 180) * (y1-y0)

    def _render_custom_field(self, field: np.ndarray, difference: bool = False) -> Image.Image:
        rgb = anomaly_to_rgb(np.flipud(field))
        if difference:
            rgb = anomaly_to_rgb(np.flipud(np.clip(field, -4, 4)))
        native = Image.fromarray(rgb, "RGB").convert("RGBA")
        x0,y0,x1,y1 = self.map_box
        return native.resize((x1-x0, y1-y0), Image.Resampling.NEAREST)

    def _render_field_layer(self, index: int) -> Image.Image:
        return self._render_custom_field(self.data.annual[index])

    def draw_map_frame(self, image: Image.Image, layer: Image.Image, alpha: int = 255, box: Optional[Tuple[int,int,int,int]] = None):
        box = box or self.map_box
        x0,y0,x1,y1 = box
        if box == self.map_box:
            lay = layer.copy()
        else:
            lay = layer.resize((x1-x0, y1-y0), Image.Resampling.NEAREST)
        if alpha < 255:
            lay.putalpha(alpha)
        image.alpha_composite(lay, (x0,y0))
        dr = ImageDraw.Draw(image)
        for lon in range(-180,181,60):
            x,_ = self.project(lon,0,box)
            dr.line((x,y0,x,y1), fill=COLORS["grid"]+(40,), width=max(1,int(SCALE)))
        for lat in range(-60,61,30):
            _,y = self.project(0,lat,box)
            dr.line((x0,y,x1,y), fill=COLORS["grid"]+(40,), width=max(1,int(SCALE)))
        for poly in BUILTIN_LAND_POLYGONS:
            pts = [self.project(lon,lat,box) for lon,lat in poly]
            if len(pts) >= 2:
                dr.line(pts, fill=(235,242,245,95), width=max(1,int(1.2*SCALE)), joint="curve")
        dr.rectangle(box, outline=(185,208,215,100), width=max(1,int(2*SCALE)))

    def draw_header(self, image: Image.Image, scene_name: str):
        draw_text(image, CONFIG["title"], (int(48*SCALE), int(72*SCALE)), 31 if not QUICK_MODE else 15,
                  COLORS["white"]+(245,), True, True, stroke=2)
        if scene_name != "opening":
            draw_text(image, CONFIG["subtitle"], (int(50*SCALE), int(122*SCALE)), 12 if not QUICK_MODE else 6,
                      COLORS["muted"]+(205,), True, True, stroke=1)

    def draw_legend(self, image: Image.Image, y: Optional[int] = None):
        if y is None:
            y = int(1390*SCALE)
        x0 = int(85*SCALE); x1 = OUT_W-int(85*SCALE); h = max(8,int(22*SCALE))
        vals = np.linspace(-2.0, 2.5, max(2, x1-x0))[None,:]
        strip = Image.fromarray(anomaly_to_rgb(vals), "RGB").resize((x1-x0,h)).convert("RGBA")
        image.alpha_composite(strip,(x0,y))
        draw_text(image, "COOLER", (x0,y+int(42*SCALE)), 12 if not QUICK_MODE else 6, COLORS["muted"]+(220,), True, True, "la",1)
        draw_text(image, "0", ((x0+x1)//2,y+int(42*SCALE)), 12 if not QUICK_MODE else 6, COLORS["muted"]+(220,), True, True, "ma",1)
        draw_text(image, "WARMER  °C", (x1,y+int(42*SCALE)), 12 if not QUICK_MODE else 6, COLORS["muted"]+(220,), True, True, "ra",1)

    def draw_opening(self, image: Image.Image, t: float, shot: Dict[str,Any]):
        p = smoothstep(shot_progress(t,shot))
        last = self.annual_layers[-1]
        self.draw_map_frame(image,last,alpha=int(50+150*p))
        overlay = Image.new("RGBA",OUT_SIZE,(0,0,0,0)); dr=ImageDraw.Draw(overlay)
        x0,y0,x1,y1=self.map_box
        cell = max(12,int((28-16*p)*SCALE))
        for yy in range(y0,y1,cell):
            for xx in range(x0,x1,cell):
                if self.rng.random() < 0.25 + 0.55*p:
                    dr.rectangle((xx,yy,min(xx+cell-2,x1),min(yy+cell-2,y1)), outline=(255,180,95,int(20+80*p)), width=1)
        image.alpha_composite(overlay)
        draw_text(image,"1880 → 2025",(OUT_W//2,int(1495*SCALE)),54 if not QUICK_MODE else 27,COLORS["white"]+(245,),True,True,"ma",2)
        draw_text(image,"ONE GRID CELL AT A TIME",(OUT_W//2,int(1570*SCALE)),17 if not QUICK_MODE else 8,COLORS["muted"]+(225,),True,True,"ma",1)

    def draw_timeline(self, image: Image.Image, t: float, shot: Dict[str,Any]):
        p = smoothstep(shot_progress(t,shot))
        pos = p * (len(self.data.years)-1)
        i0=int(np.floor(pos)); i1=min(i0+1,len(self.data.years)-1); f=pos-i0
        if f < 0.5:
            idx=i0
        else:
            idx=i1
        self.draw_map_frame(image,self.annual_layers[idx])
        year=int(self.data.years[idx]); stat=self.stats[idx]
        draw_text(image,str(year),(OUT_W//2,int(1458*SCALE)),82 if not QUICK_MODE else 40,COLORS["white"]+(250,),True,True,"ma",3)
        sign="+" if stat.global_mean_c>=0 else ""
        draw_text(image,f"GLOBAL  {sign}{stat.global_mean_c:.2f}°C",(OUT_W//2,int(1565*SCALE)),23 if not QUICK_MODE else 11,COLORS["warm1"]+(245,),True,True,"ma",2)
        x0=int(80*SCALE); x1=OUT_W-int(80*SCALE); y=int(1642*SCALE)
        ImageDraw.Draw(image).line((x0,y,x1,y),fill=(150,170,178,100),width=max(1,int(3*SCALE)))
        xx=int(lerp(x0,x1,idx/max(len(self.data.years)-1,1)))
        ImageDraw.Draw(image).ellipse((xx-int(8*SCALE),y-int(8*SCALE),xx+int(8*SCALE),y+int(8*SCALE)),fill=COLORS["warm2"]+(240,))
        draw_text(image,str(int(self.data.years[0])),(x0,y+int(36*SCALE)),11 if not QUICK_MODE else 6,COLORS["muted"]+(210,),True,True,"la",1)
        draw_text(image,str(int(self.data.years[-1])),(x1,y+int(36*SCALE)),11 if not QUICK_MODE else 6,COLORS["muted"]+(210,),True,True,"ra",1)
        self.draw_legend(image,int(1720*SCALE))

    def draw_decades(self, image: Image.Image, t: float, shot: Dict[str,Any]):
        candidates=[1880,1940,1980,int(self.data.years[-1])]
        years=[min(self.data.years,key=lambda y:abs(int(y)-c)) for c in candidates]
        gap=int(18*SCALE); margin=int(42*SCALE); w=(OUT_W-2*margin-gap)//2; h=int(410*SCALE)
        boxes=[(margin,int(300*SCALE),margin+w,int(300*SCALE)+h),(margin+w+gap,int(300*SCALE),OUT_W-margin,int(300*SCALE)+h),
               (margin,int(760*SCALE),margin+w,int(760*SCALE)+h),(margin+w+gap,int(760*SCALE),OUT_W-margin,int(760*SCALE)+h)]
        for y,box in zip(years,boxes):
            idx=self.year_lookup[int(y)]
            self.draw_map_frame(image,self.annual_layers[idx],box=box)
            s=self.stats[idx]; sign="+" if s.global_mean_c>=0 else ""
            draw_text(image,f"{int(y)}",(box[0]+int(10*SCALE),box[1]+int(24*SCALE)),18 if not QUICK_MODE else 9,COLORS["white"]+(245,),True,True,"la",1)
            draw_text(image,f"{sign}{s.global_mean_c:.2f}°C",(box[2]-int(10*SCALE),box[1]+int(24*SCALE)),15 if not QUICK_MODE else 7,COLORS["white"]+(230,),True,True,"ra",1)
        draw_text(image,"SAME SCALE. DIFFERENT DECADES.",(OUT_W//2,int(1280*SCALE)),24 if not QUICK_MODE else 12,COLORS["white"]+(245,),True,True,"ma",2)
        self.draw_legend(image,int(1370*SCALE))

    def draw_difference(self, image: Image.Image, t: float, shot: Dict[str,Any]):
        self.draw_map_frame(image,self.diff_layer)
        diff=self.recent_mean-self.early_mean
        global_diff=weighted_mean(diff,self.data.lat)
        draw_text(image,"RECENT MINUS 1880–1900",(OUT_W//2,int(1450*SCALE)),28 if not QUICK_MODE else 14,COLORS["white"]+(250,),True,True,"ma",2)
        draw_text(image,f"AREA-WEIGHTED CHANGE  +{global_diff:.2f}°C",(OUT_W//2,int(1512*SCALE)),19 if not QUICK_MODE else 9,COLORS["warm1"]+(245,),True,True,"ma",2)
        draw_text(image,"COLOR = LOCAL CHANGE, NOT ABSOLUTE TEMPERATURE",(OUT_W//2,int(1570*SCALE)),11 if not QUICK_MODE else 6,COLORS["muted"]+(215,),True,True,"ma",1)
        self.draw_legend(image,int(1650*SCALE))

    def draw_latitude(self, image: Image.Image, t: float, shot: Dict[str,Any]):
        early_stats=[s for s in self.stats if 1880<=s.year<=1900]
        recent_stats=[s for s in self.stats if s.year>=int(self.data.years[-1])-10]
        bands=[("60°N–90°N",np.mean([s.north_mean_c for s in recent_stats])-np.mean([s.north_mean_c for s in early_stats])),
               ("30°S–30°N",np.mean([s.tropics_mean_c for s in recent_stats])-np.mean([s.tropics_mean_c for s in early_stats])),
               ("60°S–90°S",np.mean([s.south_mean_c for s in recent_stats])-np.mean([s.south_mean_c for s in early_stats]))]
        draw_text(image,"WARMING BY LATITUDE",(int(70*SCALE),int(330*SCALE)),30 if not QUICK_MODE else 15,COLORS["white"]+(250,),True,True,"la",2)
        maxv=max(v for _,v in bands)*1.12
        for j,(label,val) in enumerate(bands):
            y=int((530+j*290)*SCALE); x0=int(80*SCALE); x1=OUT_W-int(80*SCALE)
            dr=ImageDraw.Draw(image)
            dr.rounded_rectangle((x0,y,x1,y+int(90*SCALE)),radius=int(14*SCALE),fill=(20,27,34,210),outline=(120,145,155,80),width=max(1,int(2*SCALE)))
            bw=int((x1-x0)*clamp(val/max(maxv,0.1)))
            dr.rounded_rectangle((x0,y,x0+bw,y+int(90*SCALE)),radius=int(14*SCALE),fill=COLORS["warm2"]+(190,))
            draw_text(image,label,(x0,y-int(24*SCALE)),18 if not QUICK_MODE else 9,COLORS["muted"]+(230,),True,True,"la",1)
            draw_text(image,f"+{val:.2f}°C",(x1,y+int(46*SCALE)),26 if not QUICK_MODE else 13,COLORS["white"]+(250,),True,True,"ra",2)
        draw_text(image,"RECENT ~10 YEARS MINUS 1880–1900",(OUT_W//2,int(1485*SCALE)),14 if not QUICK_MODE else 7,COLORS["muted"]+(220,),True,True,"ma",1)

    def draw_finale(self, image: Image.Image, t: float, shot: Dict[str,Any]):
        p=smoothstep(shot_progress(t,shot)); self.draw_map_frame(image,self.annual_layers[-1],alpha=int(170+85*p))
        last=self.stats[-1]; sign="+" if last.global_mean_c>=0 else ""
        draw_text(image,str(int(self.data.years[-1])),(OUT_W//2,int(1450*SCALE)),84 if not QUICK_MODE else 42,COLORS["white"]+(250,),True,True,"ma",3)
        draw_text(image,f"{sign}{last.global_mean_c:.2f}°C GLOBAL ANOMALY",(OUT_W//2,int(1555*SCALE)),24 if not QUICK_MODE else 12,COLORS["warm1"]+(250,),True,True,"ma",2)
        draw_text(image,"THE SIGNAL IS IN THE PIXELS",(OUT_W//2,int(1630*SCALE)),18 if not QUICK_MODE else 9,COLORS["white"]+(230,),True,True,"ma",1)

    def draw_source_hud(self,image:Image.Image):
        text="SYNTHETIC PREVIEW // NOT OBSERVATIONAL DATA" if self.data.source.startswith("synthetic") else "NASA GISS / GISTEMP V4 // GHCN V4 + ERSST V5"
        draw_text(image,text,(int(48*SCALE),OUT_H-int(42*SCALE)),10 if not QUICK_MODE else 5,COLORS["muted"]+(185,),True,True,"la",1)

    def draw_film_texture(self,image:Image.Image,t:float):
        ov=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); dr=ImageDraw.Draw(ov)
        off=int((t*43)%9)
        for y in range(off,OUT_H,9):
            dr.line((0,y,OUT_W,y),fill=(140,160,170,5),width=1)
        image.alpha_composite(ov)

    def render_frame(self,t:float)->np.ndarray:
        shot=get_shot(t); name=str(shot["name"]); image=self.background(t)
        if name=="opening": self.draw_opening(image,t,shot)
        elif name=="timeline": self.draw_timeline(image,t,shot)
        elif name=="decades": self.draw_decades(image,t,shot)
        elif name=="difference": self.draw_difference(image,t,shot)
        elif name=="latitude": self.draw_latitude(image,t,shot)
        else: self.draw_finale(image,t,shot)
        self.draw_header(image,name); self.draw_source_hud(image); self.draw_film_texture(image,t)
        arr=np.asarray(image.convert("RGB"),dtype=np.float32); arr*=VIGNETTE[...,None]; arr=np.clip(arr,0,255).astype(np.uint8)
        img=ImageEnhance.Contrast(Image.fromarray(arr)).enhance(float(CONFIG["contrast"])); img=ImageEnhance.Color(img).enhance(float(CONFIG["saturation"]))
        arr=np.asarray(img,dtype=np.int16); rng=np.random.default_rng(int(t*1000)+2048); grain=rng.normal(0.0,float(CONFIG["grain_strength"]),arr.shape[:2])[:,:,None]
        return np.clip(arr+grain,0,255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Audio / video
# -----------------------------------------------------------------------------

def gaussian_envelope(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5*((times-center)/max(width,1e-6))**2)


def generate_ambient_soundtrack(path: Path) -> Path:
    sr=int(CONFIG["sample_rate"]); dur=float(CONFIG["duration_s"]); n=int(round(sr*dur)); times=np.arange(n,dtype=np.float64)/sr
    rng=np.random.default_rng(18802025); audio=np.zeros(n,dtype=np.float64)
    audio+=0.052*np.sin(math.tau*32*times+0.7*np.sin(math.tau*0.045*times)); audio+=0.034*np.sin(math.tau*49*times+1.2)
    controls=rng.normal(0,1,max(8,int(dur*4))); audio+=0.014*np.interp(times,np.linspace(0,dur,len(controls)),controls)
    sweep=next(s for s in SHOT_PLAN if s["name"]=="timeline")
    for frac in np.linspace(0.03,0.97,20 if not QUICK_MODE else 6):
        center=lerp(float(sweep["start"]),float(sweep["end"]),float(frac)); env=gaussian_envelope(times,center,0.055 if not QUICK_MODE else 0.10)
        audio+=env*0.018*np.sin(math.tau*(160+260*frac)*times)
    for scene,strength in [("decades",0.07),("difference",0.11),("latitude",0.09),("finale",0.14)]:
        sh=next(s for s in SHOT_PLAN if s["name"]==scene); c=float(sh["start"])+0.35*(float(sh["end"])-float(sh["start"]))
        audio+=gaussian_envelope(times,c,0.65 if not QUICK_MODE else 0.20)*strength*np.sin(math.tau*43*times)
    intro=np.clip(times/max(1.5,dur*0.07),0,1); intro=intro*intro*(3-2*intro); outrox=np.clip((times-(dur-1.1))/0.9,0,1); outro=1-outrox*outrox*(3-2*outrox); audio*=intro*outro
    peak=max(float(np.max(np.abs(audio))),1e-9); pcm=(np.clip(audio/peak*0.88,-1,1)*32767).astype(np.int16)
    with wave.open(str(path),"wb") as h: h.setnchannels(1); h.setsampwidth(2); h.setframerate(sr); h.writeframes(pcm.tobytes())
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


def render_video(scene:WarmingPixelsScene)->Path:
    srt=OUTPUT_ROOT/f"{CONFIG['output_basename']}.srt"; write_srt(CAPTIONS,srt)
    raw=OUTPUT_ROOT/f"{CONFIG['output_basename']}_silent.mp4"; final=OUTPUT_ROOT/f"{CONFIG['output_basename']}_final.mp4"; audio=OUTPUT_ROOT/f"{CONFIG['output_basename']}_ambient.wav"
    frames=int(round(float(CONFIG["duration_s"])*int(CONFIG["fps"]))); times=np.arange(frames)/int(CONFIG["fps"])
    print(f"Rendering {frames:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(raw,fps=int(CONFIG["fps"]),codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=None) as writer:
        for t in tqdm(times,desc="Rendering warming-pixels short"): writer.append_data(scene.render_frame(float(t)))
    generate_ambient_soundtrack(audio)
    if mux_audio(raw,audio,final): print("Final video with audio:",final.resolve()); return final
    shutil.copyfile(raw,final); print("ffmpeg audio mux unavailable; copied silent video to:",final.resolve()); return final

def main():
    print("Title:",CONFIG["title"]); print("Loading NASA GISTEMP annual grid ...")
    data=load_temperature_data(); stats=compute_stats(data); csv_path,json_path=save_data_products(data,stats)
    print("Data source:",data.source); print("Years:",int(data.years[0]),"to",int(data.years[-1])); print("Grid:",len(data.lat),"lat x",len(data.lon),"lon")
    print("Latest global anomaly:",f"{stats[-1].global_mean_c:+.2f}°C"); print("CSV:",csv_path.resolve()); print("Summary:",json_path.resolve())
    for note in data.notes: print("Data note:",note)
    scene=WarmingPixelsScene(data,stats)
    preview_times=[1.3,min(12.0,float(CONFIG["duration_s"])*0.25),min(36.0,float(CONFIG["duration_s"])*0.62),min(45.0,float(CONFIG["duration_s"])*0.77),min(52.0,float(CONFIG["duration_s"])*0.89),float(CONFIG["duration_s"])-0.7]
    for pt in tqdm(preview_times,desc="Preview frames"):
        Image.fromarray(scene.render_frame(float(pt))).save(PREVIEW_DIR/f"preview_{int(pt):02d}s.png")
    render_video(scene)
    print("Output directory:",OUTPUT_ROOT.resolve())

