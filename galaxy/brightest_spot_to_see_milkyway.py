from __future__ import annotations

"""
result : https://www.youtube.com/shorts/59glkiJZ888
Real data video generation script 
Where on Earth Does the Milky Way Shine Brightest? — ultra-cinematic, full-frame
YouTube Shorts renderer

Creates a borderless vertical science film in 1080x1920, with an optional true
4K vertical mode at 2160x3840. The animation combines an orbit-to-ground map
dive, satellite night-light imagery, climate proxies, a transparent scoring
model, and a procedural drone-style desert flyover beneath a luminous Milky Way.

Question answered
-----------------
The script does not claim to measure the intrinsic brightness of the Milky Way.
Instead, it ranks a curated international shortlist by an explicit visibility
proxy built from:

- local artificial-light brightness sampled from NASA Black Marble imagery
- clear-sky proxy from sunshine duration divided by daylight duration
- annual precipitation as a broad dryness proxy
- elevation
- maximum altitude of the Galactic Centre, whose declination is approximated
  as -29 degrees for this cinematic calculation

A site near latitude -29 degrees places the Galactic Centre almost overhead at
culmination. That geometric advantage is combined with darkness and atmosphere
proxies. The result is an illustrative shortlist ranking, not a professional
site survey or an exhaustive search of every point on Earth.

Real-data sources
-----------------
- NASA Earth at Night / Black Marble 2016 grayscale map:
  https://science.nasa.gov/earth/earth-observatory/earth-at-night/maps/
- Open-Meteo Historical Weather API using ERA5 / ERA5-Land reanalysis:
  https://open-meteo.com/en/docs/historical-weather-api


modes
---------------
Standard full-screen vertical:
    python where_on_earth_milky_way_shines_brightest_cinematic_short.py

True 4K vertical, 2160x3840:
    MILKY_WAY_SHORT_4K=1 python where_on_earth_milky_way_shines_brightest_cinematic_short.py

Fast validation preview:
    MILKY_WAY_SHORT_QUICK=1 python where_on_earth_milky_way_shines_brightest_cinematic_short.py

Force deterministic offline preview data:
    MILKY_WAY_SHORT_OFFLINE=1 python where_on_earth_milky_way_shines_brightest_cinematic_short.py

Use local data:
    MILKY_WAY_DATA_PATH=/path/to/site_metrics.csv \
    MILKY_WAY_NIGHTLIGHT_PATH=/path/to/BlackMarble_2016_01deg_gray.jpg \
        python where_on_earth_milky_way_shines_brightest_cinematic_short.py
"""

import json
import math
import os
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("MILKY_WAY_SHORT_QUICK", "0") == "1"
FOUR_K_MODE = os.environ.get("MILKY_WAY_SHORT_4K", "0") == "1" and not QUICK_MODE
OFFLINE_MODE = os.environ.get("MILKY_WAY_SHORT_OFFLINE", "0") == "1"
LOCAL_DATA_PATH = os.environ.get("MILKY_WAY_DATA_PATH", "").strip()
LOCAL_NIGHTLIGHT_PATH = os.environ.get("MILKY_WAY_NIGHTLIGHT_PATH", "").strip()

OUTPUT_ROOT = Path("where_on_earth_milky_way_shines_brightest_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

BLACK_MARBLE_URL = (
    "https://assets.science.nasa.gov/content/dam/science/esd/eo/images/"
    "imagerecords/144000/144897/BlackMarble_2016_01deg_gray.jpg"
)
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else (2160 if FOUR_K_MODE else 1080),
    "video_height": 960 if QUICK_MODE else (3840 if FOUR_K_MODE else 1920),
    "fps": 6 if QUICK_MODE else (30 if FOUR_K_MODE else 24),
    "duration_s": 12 if QUICK_MODE else 60,
    "output_basename": "where_on_earth_does_the_milky_way_shine_brightest",
    "title": "WHERE ON EARTH DOES THE MILKY WAY SHINE BRIGHTEST?",
    "subtitle": "satellite darkness + climate + galactic geometry // cinematic data film",
    "data_timeout_s": 60,
    "climate_start": "2015-01-01",
    "climate_end": "2024-12-31",
    "climate_batch_size": 12,
    "nightlight_sample_radius_deg": 0.55,
    "candidate_marker_limit": 24,
    "galactic_centre_dec_deg": -29.0,
    "ranking_limit": 6,
    "background_stars": 280 if QUICK_MODE else 650,
    "hud_noise": 30 if QUICK_MODE else 72,
    "contrast": 1.10,
    "saturation": 1.08,
    "vignette": 0.27,
    "score_weights": {
        "darkness_score": 0.30,
        "clear_score": 0.25,
        "dryness_score": 0.16,
        "elevation_score": 0.14,
        "galactic_core_geometry_score": 0.15,
    },
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)

COLORS = {
    "ice": (153, 226, 255),
    "cyan": (72, 232, 255),
    "blue": (70, 125, 255),
    "violet": (183, 107, 255),
    "gold": (255, 194, 86),
    "rose": (255, 93, 150),
    "green": (100, 243, 178),
    "orange": (255, 140, 73),
    "white": (246, 251, 255),
    "muted": (157, 205, 226),
    "dark": (3, 7, 16),
    "earth": (40, 82, 113),
    "sand": (189, 129, 78),
}

# Curated locations rather than a claim of exhaustively testing every land pixel.
# Coordinates and elevation hints are public reference values rounded for a
# cinematic shortlist. Live elevation returned by Open-Meteo replaces the hint.
CANDIDATE_SITES: List[Dict[str, Any]] = [
    {"name": "Chajnantor Plateau", "region": "Chile", "lat": -23.029, "lon": -67.755, "elevation_hint_m": 5050, "kind": "observatory"},
    {"name": "Cerro Paranal", "region": "Chile", "lat": -24.627, "lon": -70.404, "elevation_hint_m": 2635, "kind": "observatory"},
    {"name": "Cerro Armazones", "region": "Chile", "lat": -24.589, "lon": -70.191, "elevation_hint_m": 3046, "kind": "observatory"},
    {"name": "La Silla", "region": "Chile", "lat": -29.258, "lon": -70.734, "elevation_hint_m": 2400, "kind": "observatory"},
    {"name": "Cerro Pachon", "region": "Chile", "lat": -30.240, "lon": -70.736, "elevation_hint_m": 2715, "kind": "observatory"},
    {"name": "Puna de Atacama", "region": "Argentina", "lat": -24.590, "lon": -67.400, "elevation_hint_m": 3500, "kind": "dark-sky"},
    {"name": "Mauna Kea", "region": "Hawaii, USA", "lat": 19.821, "lon": -155.468, "elevation_hint_m": 4205, "kind": "observatory"},
    {"name": "Roque de los Muchachos", "region": "La Palma, Spain", "lat": 28.762, "lon": -17.879, "elevation_hint_m": 2396, "kind": "observatory"},
    {"name": "Teide Observatory", "region": "Tenerife, Spain", "lat": 28.300, "lon": -16.511, "elevation_hint_m": 2390, "kind": "observatory"},
    {"name": "Hanle", "region": "Ladakh, India", "lat": 32.779, "lon": 78.964, "elevation_hint_m": 4500, "kind": "observatory"},
    {"name": "Ali Observatory", "region": "Tibet, China", "lat": 32.319, "lon": 80.026, "elevation_hint_m": 5100, "kind": "observatory"},
    {"name": "San Pedro Martir", "region": "Baja California, Mexico", "lat": 31.044, "lon": -115.464, "elevation_hint_m": 2830, "kind": "observatory"},
    {"name": "Kitt Peak", "region": "Arizona, USA", "lat": 31.958, "lon": -111.598, "elevation_hint_m": 2096, "kind": "observatory"},
    {"name": "Mount Graham", "region": "Arizona, USA", "lat": 32.701, "lon": -109.892, "elevation_hint_m": 3191, "kind": "observatory"},
    {"name": "Siding Spring", "region": "New South Wales, Australia", "lat": -31.273, "lon": 149.071, "elevation_hint_m": 1165, "kind": "observatory"},
    {"name": "Murchison", "region": "Western Australia", "lat": -26.704, "lon": 116.670, "elevation_hint_m": 377, "kind": "radio-quiet"},
    {"name": "Sutherland", "region": "South Africa", "lat": -32.379, "lon": 20.811, "elevation_hint_m": 1798, "kind": "observatory"},
    {"name": "Gamsberg", "region": "Namibia", "lat": -23.340, "lon": 16.230, "elevation_hint_m": 2347, "kind": "dark-sky"},
    {"name": "NamibRand", "region": "Namibia", "lat": -25.000, "lon": 16.000, "elevation_hint_m": 1100, "kind": "dark-sky"},
    {"name": "Aoraki Mackenzie", "region": "New Zealand", "lat": -44.000, "lon": 170.470, "elevation_hint_m": 1029, "kind": "dark-sky"},
    {"name": "Jebel Toubkal", "region": "Morocco", "lat": 31.059, "lon": -7.916, "elevation_hint_m": 4167, "kind": "high-altitude"},
    {"name": "Mount Kenya", "region": "Kenya", "lat": -0.152, "lon": 37.308, "elevation_hint_m": 5199, "kind": "equatorial"},
    {"name": "Chimborazo", "region": "Ecuador", "lat": -1.469, "lon": -78.817, "elevation_hint_m": 6263, "kind": "equatorial"},
    {"name": "Quito Equator", "region": "Ecuador", "lat": 0.000, "lon": -78.455, "elevation_hint_m": 2850, "kind": "equatorial"},
]

FULL_CAPTIONS = [
    (0.4, 7.0, "The Milky Way is always there. But from most cities, artificial light erases its faint structure before your eyes can see it."),
    (7.1, 15.0, "Geometry matters too. The Galactic Centre sits near minus twenty-nine degrees declination, so southern subtropical latitudes can lift its brightest core almost overhead."),
    (15.1, 23.8, "Now scan Earth at night. The brightest continents are the worst places to chase the galaxy; the darkest remote plateaus survive the first cut."),
    (23.9, 33.5, "Then add a decade of sunshine and precipitation proxies, elevation, and local satellite night-light brightness for a curated global shortlist."),
    (33.6, 44.5, "Every site receives a transparent visibility score. Darkness dominates, but clear air, dryness, altitude, and Galactic Centre geometry all change the ranking."),
    (44.6, 54.0, "The camera leaves orbit and dives toward the highest-ranked landscape, where the galactic core climbs high above a dry, dark horizon."),
    (54.1, 59.5, "This is not the single absolute brightest point on Earth. It is the strongest Milky Way visibility proxy among the locations tested in this run."),
]

SHOT_PLAN_FULL = [
    {"name": "intro", "start": 0.0, "end": 7.2},
    {"name": "coverage", "start": 7.2, "end": 15.4},
    {"name": "darkness", "start": 15.4, "end": 24.2},
    {"name": "climate", "start": 24.2, "end": 34.0},
    {"name": "ranking", "start": 34.0, "end": 45.0},
    {"name": "flyover", "start": 45.0, "end": 54.2},
    {"name": "finale", "start": 54.2, "end": 60.0},
]

if QUICK_MODE:
    _scale = float(CONFIG["duration_s"]) / 60.0
    CAPTIONS = [(a * _scale, b * _scale, text) for a, b, text in FULL_CAPTIONS]
    SHOT_PLAN = [
        {"name": shot["name"], "start": shot["start"] * _scale, "end": shot["end"] * _scale}
        for shot in SHOT_PLAN_FULL
    ]
else:
    CAPTIONS = FULL_CAPTIONS
    SHOT_PLAN = SHOT_PLAN_FULL


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(value: float) -> float:
    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


def get_font(size: int, bold: bool = False):
    candidates = [
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
    size: int = 28,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    stroke: int = 2,
    anchor: str = "la",
):
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold=bold),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(220, fill[3] if len(fill) > 3 else 220)),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int = 28,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    line_spacing: int = 6,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        box = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)
        if box[2] - box[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 220))
        box = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += (box[3] - box[1]) + line_spacing


def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 170):
    overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        box,
        radius=24 if not QUICK_MODE else 12,
        fill=(2, 7, 18, alpha),
        outline=(100, 200, 235, 70),
        width=1,
    )
    image.alpha_composite(overlay)


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.8, 0.0, 1.0).astype(np.float32)


def apply_grade(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array)
    image = ImageEnhance.Contrast(image).enhance(float(CONFIG["contrast"]))
    image = ImageEnhance.Color(image).enhance(float(CONFIG["saturation"]))
    return np.asarray(image)


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


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except Exception:
        return default


def latitude_sky_fraction(latitude_deg: float) -> float:
    """Ideal fraction of the celestial sphere that rises above a flat horizon."""
    return (1.0 + math.cos(math.radians(abs(float(latitude_deg))))) / 2.0


def lonlat_to_map_xy(lon: float, lat: float, width: int, height: int) -> Tuple[float, float]:
    x = ((float(lon) + 180.0) % 360.0) / 360.0 * width
    y = (90.0 - float(lat)) / 180.0 * height
    return x, y


def percentile_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    fill_value = float(numeric.median()) if numeric.notna().any() else 0.0
    ranks = numeric.fillna(fill_value).rank(method="average", pct=True)
    return ranks if higher_is_better else 1.0 - ranks + 1.0 / max(len(ranks), 1)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def download_file(url: str, path: Path, timeout_s: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "milky-way-visibility-short-renderer/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return path


def synthetic_nightlight_map(width: int = 1800, height: int = 900) -> Image.Image:
    """Deterministic layout-only fixture, not a geographic data product."""
    rng = np.random.default_rng(2016)
    base = np.zeros((height, width), dtype=np.float32)
    # A few broad, city-like clusters to make the preview read as a night map.
    centers = [
        (-75, 40, 0.95), (-0.1, 51.5, 0.90), (10, 50, 0.86), (77, 28, 0.90),
        (121, 31, 0.95), (139, 35, 0.95), (-46, -23, 0.83), (151, -33, 0.76),
        (31, 30, 0.78), (28, -26, 0.74),
    ]
    yy, xx = np.mgrid[0:height, 0:width]
    for lon, lat, amplitude in centers:
        cx, cy = lonlat_to_map_xy(lon, lat, width, height)
        sx = rng.uniform(10, 28)
        sy = rng.uniform(6, 18)
        base += amplitude * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2.0)
    base += rng.random((height, width), dtype=np.float32) * 0.015
    return Image.fromarray(np.clip(base * 255.0, 0, 255).astype(np.uint8), mode="L")


def load_nightlight_map() -> Tuple[Image.Image, str, List[str]]:
    notes: List[str] = []
    if LOCAL_NIGHTLIGHT_PATH:
        try:
            image = Image.open(Path(LOCAL_NIGHTLIGHT_PATH).expanduser()).convert("L")
            notes.append(f"Loaded MILKY_WAY_NIGHTLIGHT_PATH={LOCAL_NIGHTLIGHT_PATH}")
            return image, "local_nasa_compatible_nightlight_map", notes
        except Exception as exc:
            notes.append(f"Local night-light image failed: {exc}")
    if OFFLINE_MODE:
        notes.append("Offline mode requested; using synthetic night-light fixture")
        return synthetic_nightlight_map(), "synthetic_nightlight_fixture", notes
    cache_path = DATA_ROOT / "BlackMarble_2016_01deg_gray.jpg"
    try:
        if not cache_path.exists() or cache_path.stat().st_size < 100_000:
            download_file(BLACK_MARBLE_URL, cache_path, float(CONFIG["data_timeout_s"]))
        return Image.open(cache_path).convert("L"), "nasa_black_marble_2016", notes
    except Exception as exc:
        notes.append(f"NASA Black Marble fallback: {exc}")
        return synthetic_nightlight_map(), "synthetic_nightlight_fixture", notes


def batch_items(items: Sequence[Dict[str, Any]], size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def fetch_open_meteo_batch(sites: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    params = {
        "latitude": ",".join(f"{site['lat']:.5f}" for site in sites),
        "longitude": ",".join(f"{site['lon']:.5f}" for site in sites),
        "start_date": CONFIG["climate_start"],
        "end_date": CONFIG["climate_end"],
        "daily": "precipitation_sum,sunshine_duration,daylight_duration",
        "timezone": "UTC",
    }
    url = OPEN_METEO_ARCHIVE_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "milky-way-visibility-short-renderer/1.0"})
    with urllib.request.urlopen(request, timeout=float(CONFIG["data_timeout_s"])) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or len(payload) != len(sites):
        raise RuntimeError(f"Unexpected Open-Meteo response shape: expected {len(sites)}, received {type(payload).__name__}")
    return payload


def summarize_open_meteo(site: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    daily = response.get("daily") or {}
    precip = pd.to_numeric(pd.Series(daily.get("precipitation_sum", [])), errors="coerce")
    sunshine = pd.to_numeric(pd.Series(daily.get("sunshine_duration", [])), errors="coerce")
    daylight = pd.to_numeric(pd.Series(daily.get("daylight_duration", [])), errors="coerce")
    sunshine_fraction = (sunshine / daylight.replace(0, np.nan)).clip(0.0, 1.0)
    dates = pd.to_datetime(pd.Series(daily.get("time", [])), errors="coerce")
    years = max(int(dates.dt.year.nunique()) if dates.notna().any() else 0, 1)
    return {
        **site,
        "mean_clear_sunshine_fraction": float(sunshine_fraction.mean()) if sunshine_fraction.notna().any() else float("nan"),
        "cloudiness_proxy_pct": float((1.0 - sunshine_fraction.mean()) * 100.0) if sunshine_fraction.notna().any() else float("nan"),
        "annual_precipitation_mm": float(precip.sum() / years) if precip.notna().any() else float("nan"),
        "mean_daily_sunshine_h": float(sunshine.mean() / 3600.0) if sunshine.notna().any() else float("nan"),
        "elevation_m": safe_float(response.get("elevation"), safe_float(site.get("elevation_hint_m"), 0.0)),
        "climate_days": int(sunshine_fraction.notna().sum()),
        "climate_source": "open_meteo_era5_historical",
    }


def fixture_site_metrics() -> pd.DataFrame:
    """Deterministic preview values; intentionally labelled as non-observational."""
    rng = np.random.default_rng(202408)
    rows: List[Dict[str, Any]] = []
    for site in CANDIDATE_SITES:
        lat = abs(float(site["lat"]))
        desert_bonus = 1.0 if site["region"] in {"Chile", "Argentina", "Namibia", "Ladakh, India", "Tibet, China"} else 0.0
        cloud = np.clip(48 - desert_bonus * 25 + lat * 0.15 + rng.normal(0, 6), 7, 78)
        precip = np.clip(900 - desert_bonus * 760 + lat * 4 + rng.normal(0, 120), 8, 2200)
        sunshine = np.clip(8.2 + desert_bonus * 2.2 - cloud * 0.035 + rng.normal(0, 0.5), 2.5, 12.5)
        rows.append({
            **site,
            "mean_clear_sunshine_fraction": float(1.0 - cloud / 100.0),
            "cloudiness_proxy_pct": float(cloud),
            "annual_precipitation_mm": float(precip),
            "mean_daily_sunshine_h": float(sunshine),
            "elevation_m": float(site["elevation_hint_m"]),
            "climate_days": 3650,
            "climate_source": "deterministic_preview_fixture",
        })
    return pd.DataFrame(rows)


def load_climate_metrics() -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    notes: List[str] = []
    request_urls: List[str] = []
    if LOCAL_DATA_PATH:
        try:
            frame = pd.read_csv(Path(LOCAL_DATA_PATH).expanduser())
            if "mean_clear_sunshine_fraction" not in frame.columns:
                if "cloudiness_proxy_pct" in frame.columns:
                    frame["mean_clear_sunshine_fraction"] = 1.0 - pd.to_numeric(frame["cloudiness_proxy_pct"], errors="coerce") / 100.0
                elif "mean_cloud_cover_pct" in frame.columns:
                    frame["cloudiness_proxy_pct"] = pd.to_numeric(frame["mean_cloud_cover_pct"], errors="coerce")
                    frame["mean_clear_sunshine_fraction"] = 1.0 - frame["cloudiness_proxy_pct"] / 100.0
            if "cloudiness_proxy_pct" not in frame.columns and "mean_clear_sunshine_fraction" in frame.columns:
                frame["cloudiness_proxy_pct"] = (1.0 - pd.to_numeric(frame["mean_clear_sunshine_fraction"], errors="coerce")) * 100.0
            required = {"name", "region", "lat", "lon", "mean_clear_sunshine_fraction", "annual_precipitation_mm", "elevation_m"}
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            frame["climate_source"] = frame.get("climate_source", "local_candidate_metrics_csv")
            notes.append(f"Loaded MILKY_WAY_DATA_PATH={LOCAL_DATA_PATH}")
            return frame, "local_candidate_metrics_csv", notes, request_urls
        except Exception as exc:
            notes.append(f"Local candidate CSV failed: {exc}")
    if OFFLINE_MODE:
        notes.append("Offline mode requested; using deterministic climate fixture")
        return fixture_site_metrics(), "deterministic_preview_fixture", notes, request_urls
    rows: List[Dict[str, Any]] = []
    try:
        for group in batch_items(CANDIDATE_SITES, int(CONFIG["climate_batch_size"])):
            params = {
                "latitude": ",".join(f"{site['lat']:.5f}" for site in group),
                "longitude": ",".join(f"{site['lon']:.5f}" for site in group),
                "start_date": CONFIG["climate_start"],
                "end_date": CONFIG["climate_end"],
                "daily": "precipitation_sum,sunshine_duration,daylight_duration",
                "timezone": "UTC",
            }
            request_urls.append(OPEN_METEO_ARCHIVE_URL + "?" + urllib.parse.urlencode(params))
            responses = fetch_open_meteo_batch(group)
            rows.extend(summarize_open_meteo(site, response) for site, response in zip(group, responses))
        frame = pd.DataFrame(rows)
        if len(frame) != len(CANDIDATE_SITES):
            raise RuntimeError(f"Only {len(frame)} of {len(CANDIDATE_SITES)} sites were returned")
        return frame, "open_meteo_era5_historical", notes, request_urls
    except Exception as exc:
        notes.append(f"Open-Meteo fallback: {exc}")
        return fixture_site_metrics(), "deterministic_preview_fixture", notes, request_urls


def sample_nightlight(image: Image.Image, lon: float, lat: float, radius_deg: float) -> float:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    height, width = array.shape
    cx, cy = lonlat_to_map_xy(lon, lat, width, height)
    rx = max(1, int(round(radius_deg / 360.0 * width)))
    ry = max(1, int(round(radius_deg / 180.0 * height)))
    xs = (np.arange(int(cx) - rx, int(cx) + rx + 1) % width).astype(int)
    ys = np.clip(np.arange(int(cy) - ry, int(cy) + ry + 1), 0, height - 1).astype(int)
    patch = array[np.ix_(ys, xs)]
    # Upper quantile is sensitive to nearby artificial-light sources without
    # making one saturated map pixel dominate the score.
    return float(np.percentile(patch, 80)) if patch.size else 0.0


def score_sites(frame: pd.DataFrame, night_map: Image.Image) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = frame.copy().reset_index(drop=True)
    out["nightlight_brightness_0_255"] = [
        sample_nightlight(
            night_map,
            float(row.lon),
            float(row.lat),
            float(CONFIG["nightlight_sample_radius_deg"]),
        )
        for row in out.itertuples()
    ]

    # Approximate maximum Galactic Centre altitude at upper culmination.
    # The Galactic Centre declination is fixed here at about -29 degrees.
    gc_dec = float(CONFIG["galactic_centre_dec_deg"])
    out["galactic_core_max_altitude_deg"] = np.clip(
        90.0 - np.abs(pd.to_numeric(out["lat"], errors="coerce") - gc_dec),
        0.0,
        90.0,
    )
    out["galactic_core_geometry_score"] = np.sin(
        np.deg2rad(out["galactic_core_max_altitude_deg"])
    ).clip(0.0, 1.0)

    out["clear_score"] = np.clip(
        pd.to_numeric(out["mean_clear_sunshine_fraction"], errors="coerce").fillna(0.5),
        0.0,
        1.0,
    )
    light = np.clip(
        pd.to_numeric(out["nightlight_brightness_0_255"], errors="coerce").fillna(0.0),
        0.0,
        255.0,
    )
    out["darkness_score"] = 1.0 - np.log1p(light) / math.log1p(255.0)
    precip = np.clip(
        pd.to_numeric(out["annual_precipitation_mm"], errors="coerce").fillna(1500.0),
        0.0,
        None,
    )
    out["dryness_score"] = 1.0 / (1.0 + precip / 250.0)
    elevation = pd.to_numeric(out["elevation_m"], errors="coerce").fillna(
        pd.to_numeric(out.get("elevation_hint_m", 0.0), errors="coerce").fillna(0.0)
    )
    out["elevation_score"] = np.clip((elevation - 250.0) / 4750.0, 0.0, 1.0)

    weights = CONFIG["score_weights"]
    out["milky_way_score"] = sum(float(weight) * out[column] for column, weight in weights.items())
    out["milky_way_score_100"] = out["milky_way_score"] * 100.0
    out = out.sort_values(["milky_way_score", "elevation_m"], ascending=[False, False]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    winner = out.iloc[0]
    summary = {
        "winner_name": str(winner["name"]),
        "winner_region": str(winner["region"]),
        "winner_lat": float(winner["lat"]),
        "winner_lon": float(winner["lon"]),
        "winner_elevation_m": float(winner["elevation_m"]),
        "winner_score_100": float(winner["milky_way_score_100"]),
        "winner_galactic_core_max_altitude_deg": float(winner["galactic_core_max_altitude_deg"]),
        "winner_clear_sunshine_fraction": float(winner["mean_clear_sunshine_fraction"]),
        "winner_cloudiness_proxy_pct": float(winner["cloudiness_proxy_pct"]),
        "winner_annual_precipitation_mm": float(winner["annual_precipitation_mm"]),
        "winner_darkness_score": float(winner["darkness_score"]),
        "candidate_count": int(len(out)),
        "galactic_centre_declination_deg": gc_dec,
        "score_weights": weights,
        "score_warning": "Illustrative shortlist visibility proxy; not a calibrated Milky Way surface-brightness measurement, professional site survey, or exhaustive global search.",
        "geometry_note": "Maximum Galactic Centre altitude approximated as 90 - abs(latitude - declination), with declination fixed near -29 degrees.",
    }
    return out, summary

def save_data_products(
    scores: pd.DataFrame,
    summary: Dict[str, Any],
    climate_source: str,
    night_source: str,
    notes: List[str],
    request_urls: List[str],
) -> Tuple[Path, Path]:
    ranking_path = DATA_ROOT / "milky_way_visibility_candidate_ranking.csv"
    summary_path = DATA_ROOT / "milky_way_visibility_summary.json"
    scores.to_csv(ranking_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "climate_source": climate_source,
                "nightlight_source": night_source,
                "climate_period": [CONFIG["climate_start"], CONFIG["climate_end"]],
                "notes": notes,
                "request_urls": request_urls,
                "source_urls": {
                    "nasa_black_marble_maps": "https://science.nasa.gov/earth/earth-observatory/earth-at-night/maps/",
                    "nasa_black_marble_asset": BLACK_MARBLE_URL,
                    "open_meteo_historical_weather": "https://open-meteo.com/en/docs/historical-weather-api",
                    "era5": "https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5",
                },
                "method": {
                    "nightlight": "80th percentile grayscale brightness within a local radius around each candidate",
                    "clear_score": "mean sunshine duration divided by daylight duration (daily climatological proxy)",
                    "dryness_score": "1 / (1 + annual precipitation / 250 mm)",
                    "elevation_score": "clipped linear score from 250 m to 5000 m",
                    "galactic_core_geometry_score": "sin(maximum Galactic Centre altitude), using declination approximately -29 degrees",
                    "milky_way_score": "weighted sum of darkness, clear-sky, dryness, elevation, and Galactic Centre geometry proxies",
                },
                "fixture_warning": "Any source containing 'fixture' is deterministic preview data, not observational data.",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return ranking_path, summary_path


def create_scientific_plots(scores: pd.DataFrame):
    top = scores.head(12).sort_values("milky_way_score_100", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["name"], top["milky_way_score_100"])
    ax.set_title("Illustrative Milky Way visibility proxy among candidate sites")
    ax.set_xlabel("Milky Way visibility proxy (0–100)")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "candidate_milky_way_visibility_ranking.png", dpi=170)
    plt.close(fig)

    latitudes = np.linspace(-90, 90, 721)
    gc_altitude = np.clip(90.0 - np.abs(latitudes - float(CONFIG["galactic_centre_dec_deg"])), 0.0, 90.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(latitudes, gc_altitude)
    ax.set_title("Maximum Galactic Centre altitude by latitude")
    ax.set_xlabel("Latitude (degrees)")
    ax.set_ylabel("Maximum altitude above horizon (degrees)")
    ax.set_ylim(0, 94)
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "galactic_centre_altitude_by_latitude.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    scatter = ax.scatter(
        scores["cloudiness_proxy_pct"],
        scores["elevation_m"],
        c=scores["milky_way_score_100"],
        s=35 + scores["darkness_score"] * 90,
        alpha=0.8,
    )
    for _, row in scores.head(5).iterrows():
        ax.annotate(row["name"], (row["cloudiness_proxy_pct"], row["elevation_m"]), fontsize=8)
    ax.set_title("Candidate climate and elevation proxies")
    ax.set_xlabel("Cloudiness proxy from sunshine/daylight (%)")
    ax.set_ylabel("Elevation (m)")
    fig.colorbar(scatter, ax=ax, label="Composite score")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "candidate_proxy_scatter.png", dpi=170)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

@dataclass
class Star:
    x: float
    y: float
    radius: float
    alpha: float
    phase: float


class MilkyWayScene:
    def __init__(
        self,
        scores: pd.DataFrame,
        summary: Dict[str, Any],
        night_map: Image.Image,
        climate_source: str,
        night_source: str,
    ):
        self.scores = scores.copy().reset_index(drop=True)
        self.summary = summary
        self.climate_source = climate_source
        self.night_source = night_source
        self.winner = self.scores.iloc[0]
        self.top = self.scores.head(int(CONFIG["ranking_limit"])).copy().reset_index(drop=True)
        self.night_map = night_map.convert("L")
        self.map_preview = self._make_map_preview(self.night_map)
        self.stars = self._make_stars(int(CONFIG["background_stars"]), 202408)
        self.hud = self._make_hud(int(CONFIG["hud_noise"]), 202409)
        self.terrain = self._make_terrain(seed=int(abs(self.winner["lat"] * 1000 + self.winner["lon"] * 100)))
        self.milky_way_overlay = self._make_milky_way_overlay(seed=202610)
        self.globe_cache: Dict[int, Image.Image] = {}

    @staticmethod
    def _make_stars(count: int, seed: int) -> List[Star]:
        rng = np.random.default_rng(seed)
        return [
            Star(
                x=float(rng.uniform(0, OUT_W)),
                y=float(rng.uniform(0, OUT_H)),
                radius=float(rng.uniform(0.35, 2.2) * OUT_W / 1080),
                alpha=float(rng.uniform(22, 125)),
                phase=float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(count)
        ]

    @staticmethod
    def _make_hud(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "length": float(rng.uniform(12, 95) * OUT_W / 1080),
                "alpha": float(rng.uniform(8, 45)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            }
            for _ in range(count)
        ]

    @staticmethod
    def _make_terrain(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed % (2**32 - 1))
        width, depth = 220, 180
        base = rng.normal(0, 1, (depth, width)).astype(np.float32)
        image = Image.fromarray(np.uint8(np.clip((base - base.min()) / max(float(np.ptp(base)), 1e-6) * 255, 0, 255)))
        for radius in (18, 10, 5, 2):
            image = image.filter(ImageFilter.GaussianBlur(radius))
            arr = np.asarray(image, dtype=np.float32) / 255.0
            base += arr * (radius / 7.0)
        yy, xx = np.mgrid[0:depth, 0:width]
        ridges = 0.8 * np.sin(xx / 18.0 + np.sin(yy / 25.0)) + 0.5 * np.cos((xx + yy) / 31.0)
        base = base + ridges.astype(np.float32)
        base -= base.min()
        base /= max(float(base.max()), 1e-6)
        return base

    @staticmethod
    def _make_milky_way_overlay(seed: int) -> Image.Image:
        """Pre-render a dense star field and luminous galactic band for drone shots."""
        rng = np.random.default_rng(seed)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        sky_h = int(OUT_H * 0.46)

        # Fine stars, with a few larger diffraction-like points.
        star_count = 420 if QUICK_MODE else (1800 if FOUR_K_MODE else 900)
        for index in range(star_count):
            x = float(rng.uniform(0, OUT_W))
            y = float(rng.uniform(0, sky_h))
            r = float(rng.choice([0.45, 0.65, 0.9, 1.25, 1.8], p=[0.30, 0.28, 0.23, 0.14, 0.05])) * OUT_W / 1080
            alpha = int(rng.uniform(55, 220))
            tint = COLORS["ice"] if index % 7 else COLORS["gold"]
            draw.ellipse((x-r, y-r, x+r, y+r), fill=tint + (alpha,))
            if r > 1.3 * OUT_W / 1080:
                draw.line((x-r*3.0, y, x+r*3.0, y), fill=tint + (int(alpha*0.34),), width=1)
                draw.line((x, y-r*3.0, x, y+r*3.0), fill=tint + (int(alpha*0.34),), width=1)

        # Build the Milky Way as a diagonal, granular band rather than one flat arc.
        band_w = int(OUT_W * 1.65)
        band_h = int(OUT_H * 0.22)
        band = Image.new("RGBA", (band_w, band_h), (0, 0, 0, 0))
        bd = ImageDraw.Draw(band)
        center_y = band_h * 0.52
        for width_factor, alpha in [(0.48, 16), (0.30, 25), (0.18, 38), (0.09, 56)]:
            half = band_h * width_factor
            bd.rounded_rectangle(
                (0, center_y-half, band_w, center_y+half),
                radius=max(2, int(half)),
                fill=COLORS["ice"] + (alpha,),
            )
        for index in range(950 if not QUICK_MODE else 320):
            x = float(rng.uniform(0, band_w))
            distance = abs(float(rng.normal(0.0, band_h * 0.16)))
            y = center_y + (distance if rng.random() > 0.5 else -distance)
            radius = float(rng.uniform(0.4, 2.5)) * OUT_W / 1080
            alpha = int(rng.uniform(12, 95) * (1.0 - min(distance / (band_h * 0.48), 0.9)))
            colour = COLORS["white"] if index % 4 else COLORS["gold"]
            bd.ellipse((x-radius, y-radius, x+radius, y+radius), fill=colour + (alpha,))

        # Luminous core and dust-lane asymmetry.
        core_x = int(band_w * 0.62)
        for radius, alpha in [
            (int(band_h * 0.42), 24),
            (int(band_h * 0.26), 44),
            (int(band_h * 0.12), 75),
        ]:
            bd.ellipse((core_x-radius, center_y-radius, core_x+radius, center_y+radius), fill=COLORS["gold"] + (alpha,))
        bd.rounded_rectangle((0, center_y-band_h*0.035, band_w, center_y+band_h*0.018), radius=6, fill=(8, 7, 18, 50))

        band = band.filter(ImageFilter.GaussianBlur(max(4, int(18 * OUT_W / 1080))))
        band = band.rotate(-14, resample=Image.Resampling.BICUBIC, expand=True)
        overlay.alpha_composite(band, (-int(OUT_W * 0.28), -int(OUT_H * 0.015)))

        # Additional core bloom where the band meets the horizon-facing frame.
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gx, gy = int(OUT_W * 0.64), int(OUT_H * 0.20)
        for radius, alpha in [
            (int(OUT_W * 0.16), 14),
            (int(OUT_W * 0.095), 25),
            (int(OUT_W * 0.045), 42),
        ]:
            gd.ellipse((gx-radius, gy-radius, gx+radius, gy+radius), fill=COLORS["gold"] + (alpha,))
        overlay.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(5, int(28 * OUT_W / 1080)))))
        return overlay

    @staticmethod
    def _make_map_preview(night_map: Image.Image) -> Image.Image:
        image = night_map.resize((1600, 800), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        rgb = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(array**0.65 * 255, 0, 255)
        rgb[..., 1] = np.clip(array**0.75 * 220, 0, 255)
        rgb[..., 2] = np.clip(35 + array**0.55 * 200, 0, 255)
        return Image.fromarray(rgb, mode="RGB").convert("RGBA")

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, (2, 6, 15, 255))
        draw = ImageDraw.Draw(image)
        for star in self.stars:
            alpha = int(star.alpha * (0.72 + 0.28 * math.sin(t * 1.45 + star.phase)))
            r = star.radius
            draw.ellipse((star.x - r, star.y - r, star.x + r, star.y + r), fill=COLORS["white"] + (alpha,))
        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for cx, cy, color in [
            (OUT_W * 0.20, OUT_H * 0.25, (20, 44, 130)),
            (OUT_W * 0.82, OUT_H * 0.42, (74, 24, 120)),
            (OUT_W * 0.52, OUT_H * 0.78, (10, 82, 116)),
        ]:
            radius = 300 * OUT_W / 1080
            hd.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color + (25,))
        image.alpha_composite(haze.filter(ImageFilter.GaussianBlur(70 if not QUICK_MODE else 35)))
        return image

    def draw_globe(self, image: Image.Image, center: Tuple[int, int], radius: int, center_lon: float, alpha: int = 255):
        # Cache by coarse longitude to reduce full-render CPU cost.
        key = int(round(center_lon / 4.0) * 4)
        if key not in self.globe_cache:
            size = radius * 2
            yy, xx = np.mgrid[0:size, 0:size]
            nx = (xx - radius + 0.5) / radius
            ny = (yy - radius + 0.5) / radius
            r2 = nx * nx + ny * ny
            mask = r2 <= 1.0
            z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
            lon = np.rad2deg(np.arctan2(nx, z)) + key
            lat = np.rad2deg(np.arcsin(np.clip(-ny, -1.0, 1.0)))
            source = np.asarray(self.night_map, dtype=np.float32)
            h, w = source.shape
            sx = (((lon + 180.0) % 360.0) / 360.0 * w).astype(int) % w
            sy = np.clip(((90.0 - lat) / 180.0 * h).astype(int), 0, h - 1)
            light = source[sy, sx] / 255.0
            shade = np.clip(0.24 + 0.80 * z + 0.18 * nx, 0.0, 1.0)
            rim = np.clip((1.0 - z) ** 1.7, 0.0, 1.0)
            rgb = np.zeros((size, size, 4), dtype=np.uint8)
            rgb[..., 0] = np.clip(8 + 85 * shade + 205 * light, 0, 255)
            rgb[..., 1] = np.clip(18 + 105 * shade + 205 * light, 0, 255)
            rgb[..., 2] = np.clip(38 + 150 * shade + 120 * light + 75 * rim, 0, 255)
            rgb[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
            globe = Image.fromarray(rgb, mode="RGBA")
            glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow)
            gd.ellipse((5, 5, size - 5, size - 5), outline=COLORS["cyan"] + (110,), width=max(1, int(radius * 0.018)))
            globe.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(2, int(radius * 0.03)))))
            globe.alpha_composite(glow)
            self.globe_cache[key] = globe
        globe = self.globe_cache[key]
        if alpha < 255:
            globe = globe.copy()
            globe.putalpha(np.asarray(globe.getchannel("A"), dtype=np.float32).clip(0, 255).astype(np.uint8))
        image.alpha_composite(globe, (center[0] - radius, center[1] - radius))

    def draw_world_map(self, image: Image.Image, box: Tuple[int, int, int, int], zoom: float = 1.0, focus: Optional[Tuple[float, float]] = None, alpha: int = 235):
        x0, y0, x1, y1 = box
        target_w, target_h = x1 - x0, y1 - y0
        source = self.map_preview
        if focus is None or zoom <= 1.01:
            crop = source
        else:
            lon, lat = focus
            sx, sy = lonlat_to_map_xy(lon, lat, source.width, source.height)
            crop_w = max(40, int(source.width / zoom))
            crop_h = max(20, int(source.height / zoom))
            # Horizontal wrap for the equirectangular map.
            rolled = source.transform(
                source.size,
                Image.Transform.AFFINE,
                (1, 0, source.width / 2 - sx, 0, 1, 0),
                resample=Image.Resampling.BILINEAR,
            )
            cy = int(np.clip(sy, crop_h / 2, source.height - crop_h / 2))
            crop = rolled.crop((source.width // 2 - crop_w // 2, cy - crop_h // 2, source.width // 2 + crop_w // 2, cy + crop_h // 2))
        resized = crop.resize((target_w, target_h), Image.Resampling.LANCZOS).convert("RGBA")
        resized.putalpha(alpha)
        mask = Image.new("L", (target_w, target_h), 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle((0, 0, target_w - 1, target_h - 1), radius=int(target_h * 0.10), fill=255)
        resized.putalpha(Image.fromarray(np.minimum(np.asarray(resized.getchannel("A")), np.asarray(mask)).astype(np.uint8)))
        image.alpha_composite(resized, (x0, y0))
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle(box, radius=int(target_h * 0.10), outline=COLORS["cyan"] + (75,), width=2 if not QUICK_MODE else 1)
        image.alpha_composite(overlay)

    def draw_map_markers(self, image: Image.Image, box: Tuple[int, int, int, int], reveal: float = 1.0, show_labels: bool = False):
        x0, y0, x1, y1 = box
        draw = ImageDraw.Draw(image)
        limit = min(len(self.scores), int(CONFIG["candidate_marker_limit"]))
        count = int(round(limit * clamp(reveal)))
        max_score = max(float(self.scores["milky_way_score_100"].max()), 1.0)
        for index, row in self.scores.head(limit).iterrows():
            if index >= count:
                break
            x = x0 + ((float(row["lon"]) + 180.0) % 360.0) / 360.0 * (x1 - x0)
            y = y0 + (90.0 - float(row["lat"])) / 180.0 * (y1 - y0)
            weight = float(row["milky_way_score_100"]) / max_score
            colour = COLORS["gold"] if int(row["rank"]) == 1 else tuple(
                int(round(lerp(COLORS["blue"][c], COLORS["rose"][c], weight))) for c in range(3)
            )
            radius = (4 + 5 * weight) * OUT_W / 1080
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour + (240,), outline=COLORS["white"] + (220,), width=1)
            if show_labels and int(row["rank"]) <= 5:
                draw_text(image, f"{int(row['rank'])}. {row['name']}", (int(x + 10), int(y - 8)), size=13 if not QUICK_MODE else 6, fill=colour + (235,), bold=True, stroke=1)

    def draw_intro(self, image: Image.Image, t: float):
        shot = get_shot(t)
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        radius = int((230 + 35 * local) * OUT_W / 1080)
        self.draw_globe(image, (OUT_W // 2, int(OUT_H * 0.39)), radius, center_lon=-35 + t * 14)
        panel(image, (int(OUT_W * 0.08), int(OUT_H * 0.69), int(OUT_W * 0.92), int(OUT_H * 0.82)), 165)
        draw_text(image, "ONE GALAXY. ONE PLANET. ONE BEST WINDOW.", (OUT_W // 2, int(OUT_H * 0.735)), size=24 if not QUICK_MODE else 12, fill=COLORS["gold"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "where darkness, atmosphere, altitude, and geometry align", (OUT_W // 2, int(OUT_H * 0.782)), size=17 if not QUICK_MODE else 8, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)

    def draw_coverage(self, image: Image.Image, t: float):
        shot = get_shot(t)
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        radius = int(238 * OUT_W / 1080)
        center = (OUT_W // 2, int(OUT_H * 0.39))
        self.draw_globe(image, center, radius, center_lon=15 + t * 8)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        # Latitude rings projected as ellipses; equator pulses.
        for lat in (-60, -30, 0, 30, 60):
            y = center[1] - radius * math.sin(math.radians(lat))
            ring_w = radius * 2 * math.cos(math.radians(lat))
            alpha = int(70 + (150 * local if lat == -30 else 0))
            colour = COLORS["gold"] if lat == -30 else COLORS["cyan"]
            draw.ellipse((center[0] - ring_w / 2, y - radius * 0.07, center[0] + ring_w / 2, y + radius * 0.07), outline=colour + (alpha,), width=3 if lat == -30 else 1)
        image.alpha_composite(overlay)
        panel(image, (int(OUT_W * 0.08), int(OUT_H * 0.69), int(OUT_W * 0.92), int(OUT_H * 0.835)), 172)
        draw_text(image, "THE GALACTIC CORE HAS A GEOMETRIC SWEET SPOT", (OUT_W // 2, int(OUT_H * 0.728)), size=24 if not QUICK_MODE else 12, fill=COLORS["gold"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "maximum core altitude = 90° − |latitude − (−29°)|", (OUT_W // 2, int(OUT_H * 0.772)), size=17 if not QUICK_MODE else 8, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)
        draw_text(image, f"near 29° south, the bright core can pass almost overhead", (OUT_W // 2, int(OUT_H * 0.808)), size=16 if not QUICK_MODE else 8, fill=COLORS["cyan"] + (220,), anchor="ma", stroke=1)

    def draw_darkness(self, image: Image.Image, t: float):
        shot = get_shot(t)
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        box = (int(OUT_W * 0.055), int(OUT_H * 0.22), int(OUT_W * 0.945), int(OUT_H * 0.68))
        self.draw_world_map(image, box, alpha=245)
        self.draw_map_markers(image, box, reveal=min(local * 1.3, 1.0), show_labels=False)
        panel(image, (int(OUT_W * 0.08), int(OUT_H * 0.715), int(OUT_W * 0.92), int(OUT_H * 0.835)), 174)
        draw_text(image, "CITY LIGHTS ERASE THE GALAXY", (OUT_W // 2, int(OUT_H * 0.752)), size=24 if not QUICK_MODE else 12, fill=COLORS["rose"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "NASA Black Marble reveals which candidate horizons remain truly dark", (OUT_W // 2, int(OUT_H * 0.797)), size=15 if not QUICK_MODE else 7, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)

    def draw_climate(self, image: Image.Image, t: float):
        shot = get_shot(t)
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        box = (int(OUT_W * 0.055), int(OUT_H * 0.20), int(OUT_W * 0.945), int(OUT_H * 0.67))
        self.draw_world_map(image, box, alpha=210)
        self.draw_map_markers(image, box, reveal=1.0, show_labels=True)
        # Animated cloud wisps as a data-theme transition, not a weather map.
        cloud = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        cd = ImageDraw.Draw(cloud)
        rng = np.random.default_rng(86)
        for index in range(28 if not QUICK_MODE else 14):
            x = (rng.uniform(0, OUT_W) + t * (8 + index % 4) * OUT_W / 1080) % OUT_W
            y = rng.uniform(OUT_H * 0.22, OUT_H * 0.65)
            radius = rng.uniform(22, 65) * OUT_W / 1080
            cd.ellipse((x - radius, y - radius * 0.3, x + radius, y + radius * 0.3), fill=COLORS["ice"] + (int(8 + 28 * local),))
        image.alpha_composite(cloud.filter(ImageFilter.GaussianBlur(18 if not QUICK_MODE else 9)))
        panel(image, (int(OUT_W * 0.07), int(OUT_H * 0.705), int(OUT_W * 0.93), int(OUT_H * 0.845)), 180)
        draw_text(image, "NOW TEST THE ATMOSPHERE", (OUT_W // 2, int(OUT_H * 0.743)), size=23 if not QUICK_MODE else 11, fill=COLORS["cyan"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "clear-sky fraction • precipitation • elevation • night lights • core altitude", (OUT_W // 2, int(OUT_H * 0.787)), size=16 if not QUICK_MODE else 8, fill=COLORS["white"] + (228,), anchor="ma", stroke=1)
        draw_text(image, f"candidate shortlist // {len(self.scores)} locations", (OUT_W // 2, int(OUT_H * 0.819)), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (215,), anchor="ma", stroke=1)

    def draw_ranking(self, image: Image.Image, t: float):
        shot = get_shot(t)
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        x0, x1 = int(OUT_W * 0.08), int(OUT_W * 0.92)
        y0 = int(OUT_H * 0.23)
        panel(image, (x0, y0, x1, int(OUT_H * 0.74)), 190)
        draw_text(image, "MILKY WAY VISIBILITY RANKING", (OUT_W // 2, int(OUT_H * 0.19)), size=24 if not QUICK_MODE else 12, fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=1)
        draw = ImageDraw.Draw(image)
        row_h = int(OUT_H * 0.073)
        max_score = max(float(self.top["milky_way_score_100"].max()), 1.0)
        reveal_count = max(1, int(math.ceil(len(self.top) * local)))
        for index, row in self.top.iterrows():
            if index >= reveal_count:
                break
            y = y0 + int(OUT_H * 0.055) + index * row_h
            rank_colour = COLORS["gold"] if index == 0 else COLORS["cyan"]
            draw_text(image, f"{index + 1}", (x0 + int(OUT_W * 0.045), y), size=22 if not QUICK_MODE else 11, fill=rank_colour + (245,), bold=True, anchor="ma", stroke=1)
            draw_text(image, str(row["name"]), (x0 + int(OUT_W * 0.095), y - int(OUT_H * 0.012)), size=18 if not QUICK_MODE else 9, fill=COLORS["white"] + (240,), bold=True, stroke=1)
            draw_text(image, str(row["region"]), (x0 + int(OUT_W * 0.095), y + int(OUT_H * 0.012)), size=13 if not QUICK_MODE else 6, fill=COLORS["muted"] + (210,), stroke=1)
            bar_x0 = x0 + int(OUT_W * 0.48)
            bar_x1 = x1 - int(OUT_W * 0.08)
            bar_y0 = y - int(OUT_H * 0.012)
            bar_y1 = y + int(OUT_H * 0.012)
            draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y1), radius=max(2, int((bar_y1 - bar_y0) / 2)), fill=(20, 42, 64, 230))
            fill_x = int(lerp(bar_x0, bar_x1, float(row["milky_way_score_100"]) / max_score))
            draw.rounded_rectangle((bar_x0, bar_y0, fill_x, bar_y1), radius=max(2, int((bar_y1 - bar_y0) / 2)), fill=rank_colour + (220,))
            draw_text(image, f"{row['milky_way_score_100']:.1f}", (bar_x1 + int(OUT_W * 0.02), y), size=15 if not QUICK_MODE else 7, fill=COLORS["white"] + (230,), bold=True, anchor="mm", stroke=1)
        panel(image, (int(OUT_W * 0.07), int(OUT_H * 0.77), int(OUT_W * 0.93), int(OUT_H * 0.855)), 164)
        draw_text(image, "illustrative visibility proxy—not a professional site survey", (OUT_W // 2, int(OUT_H * 0.812)), size=15 if not QUICK_MODE else 7, fill=COLORS["gold"] + (230,), bold=True, anchor="ma", stroke=1)

    def draw_drone_terrain(self, image: Image.Image, t: float, local: float):
        horizon = int(OUT_H * 0.43)
        draw = ImageDraw.Draw(image)
        # Sky gradient.
        for y in range(0, horizon):
            p = y / max(horizon, 1)
            colour = (
                int(lerp(4, 22, p)),
                int(lerp(8, 36, p)),
                int(lerp(22, 62, p)),
                255,
            )
            draw.line((0, y, OUT_W, y), fill=colour)
        # Bring back a dense real-night aesthetic after the gradient overwrites the space background.
        image.alpha_composite(self.milky_way_overlay)

        # A faint horizon airglow separates the sky from the terrain.
        airglow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        ad = ImageDraw.Draw(airglow)
        ad.rectangle((0, horizon-int(OUT_H*0.035), OUT_W, horizon+int(OUT_H*0.018)), fill=COLORS["cyan"] + (22,))
        image.alpha_composite(airglow.filter(ImageFilter.GaussianBlur(max(4, int(18 * OUT_W / 1080)))))

        # Perspective terrain strips sampled from a deterministic height field.
        depth, width = self.terrain.shape
        camera_shift = (t * 16.0) % depth
        for screen_y in range(horizon, OUT_H):
            p = (screen_y - horizon) / max(OUT_H - horizon, 1)
            world_depth = int((camera_shift + (1.0 - p) ** 2 * (depth - 2)) % depth)
            visible_width = int(lerp(width * 0.18, width * 0.95, p))
            center_x = width / 2 + math.sin(t * 0.35) * width * 0.10
            xs = np.linspace(center_x - visible_width / 2, center_x + visible_width / 2, OUT_W)
            xs = np.clip(xs.astype(int), 0, width - 1)
            heights = self.terrain[world_depth, xs]
            shade = np.clip(0.30 + 0.78 * heights + 0.18 * np.gradient(heights), 0.0, 1.0)
            base_r = 45 + 125 * p
            base_g = 34 + 75 * p
            base_b = 32 + 45 * p
            row = np.zeros((OUT_W, 4), dtype=np.uint8)
            row[:, 0] = np.clip(base_r * (0.65 + 0.55 * shade), 0, 255)
            row[:, 1] = np.clip(base_g * (0.62 + 0.48 * shade), 0, 255)
            row[:, 2] = np.clip(base_b * (0.72 + 0.35 * shade), 0, 255)
            row[:, 3] = 255
            line = Image.fromarray(row.reshape(1, OUT_W, 4), mode="RGBA")
            image.alpha_composite(line, (0, screen_y))
        # A distant ridge and moving foreground shadows.
        ridge = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        rd = ImageDraw.Draw(ridge)
        points = [(0, horizon + 35 * OUT_H / 1920)]
        for x in range(0, OUT_W + 1, max(8, int(20 * OUT_W / 1080))):
            y = horizon + (26 + 22 * math.sin(x / max(OUT_W, 1) * 8 + t * 0.15)) * OUT_H / 1920
            points.append((x, y))
        points.extend([(OUT_W, OUT_H), (0, OUT_H)])
        rd.polygon(points, fill=(18, 19, 28, 170))
        image.alpha_composite(ridge.filter(ImageFilter.GaussianBlur(2 if not QUICK_MODE else 1)))

    def draw_flyover(self, image: Image.Image, t: float):
        shot = get_shot(t)
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        if local < 0.42:
            box = (int(OUT_W * 0.04), int(OUT_H * 0.17), int(OUT_W * 0.96), int(OUT_H * 0.71))
            zoom = lerp(1.0, 8.0, smoothstep(local / 0.42))
            self.draw_world_map(
                image,
                box,
                zoom=zoom,
                focus=(float(self.winner["lon"]), float(self.winner["lat"])),
                alpha=245,
            )
            x = (box[0] + box[2]) // 2
            y = (box[1] + box[3]) // 2
            draw = ImageDraw.Draw(image)
            pulse = 8 + 12 * math.sin(t * 7) ** 2
            draw.ellipse((x - pulse, y - pulse, x + pulse, y + pulse), outline=COLORS["gold"] + (240,), width=3 if not QUICK_MODE else 1)
        else:
            terrain_local = smoothstep((local - 0.42) / 0.58)
            self.draw_drone_terrain(image, t, terrain_local)
            # Milky-Way-like haze.
            haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            hd = ImageDraw.Draw(haze)
            hd.arc((-OUT_W * 0.35, -OUT_H * 0.05, OUT_W * 1.35, OUT_H * 0.72), 200, 342, fill=COLORS["ice"] + (92,), width=max(18, int(72 * OUT_W / 1080)))
            image.alpha_composite(haze.filter(ImageFilter.GaussianBlur(34 if not QUICK_MODE else 17)))
        panel(image, (int(OUT_W * 0.07), int(OUT_H * 0.73), int(OUT_W * 0.93), int(OUT_H * 0.865)), 182)
        draw_text(image, "DIVING TOWARD THE GALACTIC SWEET SPOT", (OUT_W // 2, int(OUT_H * 0.765)), size=22 if not QUICK_MODE else 11, fill=COLORS["gold"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, f"{self.winner['name']} // {self.winner['region']}", (OUT_W // 2, int(OUT_H * 0.805)), size=20 if not QUICK_MODE else 10, fill=COLORS["white"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(image, f"{self.winner['lat']:+.3f}°, {self.winner['lon']:+.3f}°  •  {self.winner['elevation_m']:,.0f} m", (OUT_W // 2, int(OUT_H * 0.839)), size=15 if not QUICK_MODE else 7, fill=COLORS["cyan"] + (220,), anchor="ma", stroke=1)

    def draw_finale(self, image: Image.Image, t: float):
        shot = get_shot(t)
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_drone_terrain(image, t, local)
        panel(image, (int(OUT_W * 0.055), int(OUT_H * 0.49), int(OUT_W * 0.945), int(OUT_H * 0.82)), 198)
        draw_text(image, "GALACTIC CORE AT CULMINATION", (OUT_W // 2, int(OUT_H * 0.535)), size=20 if not QUICK_MODE else 10, fill=COLORS["muted"] + (235,), bold=True, anchor="ma", stroke=1)
        draw_text(image, f"{self.winner['galactic_core_max_altitude_deg']:.1f}° ABOVE THE HORIZON", (OUT_W // 2, int(OUT_H * 0.575)), size=31 if not QUICK_MODE else 15, fill=COLORS["gold"] + (250,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "BRIGHTEST MILKY WAY PROXY IN THIS RUN", (OUT_W // 2, int(OUT_H * 0.635)), size=20 if not QUICK_MODE else 10, fill=COLORS["muted"] + (235,), bold=True, anchor="ma", stroke=1)
        draw_text(image, str(self.winner["name"]).upper(), (OUT_W // 2, int(OUT_H * 0.682)), size=29 if not QUICK_MODE else 14, fill=COLORS["white"] + (250,), bold=True, anchor="ma", stroke=1)
        draw_text(image, f"score {self.winner['milky_way_score_100']:.1f}/100  •  core altitude {self.winner['galactic_core_max_altitude_deg']:.1f}°", (OUT_W // 2, int(OUT_H * 0.729)), size=17 if not QUICK_MODE else 8, fill=COLORS["cyan"] + (235,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "shortlist visibility proxy // not a global absolute measurement", (OUT_W // 2, int(OUT_H * 0.775)), size=14 if not QUICK_MODE else 7, fill=COLORS["gold"] + (225,), anchor="ma", stroke=1)

    def draw_titles(self, image: Image.Image, t: float, shot_name: str):
        intro_end = SHOT_PLAN[0]["end"] * 0.88
        alpha = int(255 * smoothstep((t - 0.15) / (0.8 if not QUICK_MODE else 0.16)) * (1.0 - smoothstep((t - intro_end) / (0.7 if not QUICK_MODE else 0.14))))
        if alpha > 4:
            draw_text(image, "WHERE ON EARTH DOES", (52 if not QUICK_MODE else 26, 80 if not QUICK_MODE else 40), size=36 if not QUICK_MODE else 18, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(image, "THE MILKY WAY", (52 if not QUICK_MODE else 26, 126 if not QUICK_MODE else 63), size=36 if not QUICK_MODE else 18, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(image, "SHINE BRIGHTEST?", (52 if not QUICK_MODE else 26, 172 if not QUICK_MODE else 86), size=36 if not QUICK_MODE else 18, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(image, CONFIG["subtitle"], (54 if not QUICK_MODE else 27, 220 if not QUICK_MODE else 110), size=18 if not QUICK_MODE else 9, fill=COLORS["cyan"] + (min(alpha, 230),), bold=True)
        labels = {
            "intro": "BEGIN IN ORBIT",
            "coverage": "GALACTIC GEOMETRY // THE CORE CLIMBS HIGHER SOUTH",
            "darkness": "SATELLITE DARKNESS // CITY LIGHTS DISAPPEAR",
            "climate": "ATMOSPHERE // CLEAR, DRY, HIGH",
            "ranking": "TRANSPARENT MILKY WAY VISIBILITY SCORE",
            "flyover": "ORBIT-TO-DESERT CAMERA DIVE",
            "finale": "THE GALAXY FILLS THE FRAME",
        }
        if t > SHOT_PLAN[0]["end"] * 0.70:
            draw_text(image, labels[shot_name], (52 if not QUICK_MODE else 26, 54 if not QUICK_MODE else 27), size=17 if not QUICK_MODE else 8, fill=COLORS["muted"] + (205,), bold=True, stroke=1)

    def draw_source_hud(self, image: Image.Image):
        live = "fixture" not in self.climate_source and "synthetic" not in self.night_source
        colour = COLORS["cyan"] if live else COLORS["gold"]
        label = "REAL-DATA MODE" if live else "PREVIEW FIXTURE MODE"
        draw_text(image, label, (OUT_W - (42 if not QUICK_MODE else 21), 64 if not QUICK_MODE else 32), size=15 if not QUICK_MODE else 7, fill=colour + (235,), bold=True, anchor="ra", stroke=1)
        draw_text(image, f"CANDIDATES // {len(self.scores)}", (OUT_W - (42 if not QUICK_MODE else 21), 92 if not QUICK_MODE else 46), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (205,), anchor="ra", stroke=1)
        draw_text(image, f"MILKY WAY WINNER // {self.winner['name'].upper()}", (OUT_W - (42 if not QUICK_MODE else 21), 118 if not QUICK_MODE else 59), size=13 if not QUICK_MODE else 6, fill=COLORS["muted"] + (195,), anchor="ra", stroke=1)

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        y0 = OUT_H - (240 if not QUICK_MODE else 120)
        panel(image, (42 if not QUICK_MODE else 21, y0, OUT_W - (42 if not QUICK_MODE else 21), y0 + (122 if not QUICK_MODE else 61)), 178)
        draw_wrapped_text(
            image,
            text,
            (66 if not QUICK_MODE else 33, y0 + (26 if not QUICK_MODE else 13)),
            OUT_W - (132 if not QUICK_MODE else 66),
            size=27 if not QUICK_MODE else 13,
            fill=COLORS["white"] + (245,),
        )

    def draw_hud_noise(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for item in self.hud:
            pulse = 0.5 + 0.5 * math.sin(t * 1.9 + item["phase"])
            if pulse < 0.76:
                continue
            y = (item["y"] + t * 8.0) % OUT_H
            draw.line((item["x"], y, item["x"] + item["length"], y), fill=COLORS["cyan"] + (int(item["alpha"] * pulse),), width=1)
        offset = int((t * 41) % 7)
        for y in range(offset, OUT_H, 7):
            draw.line((0, y, OUT_W, y), fill=(120, 200, 240, 9), width=1)
        scan_y = int((t * 170) % (OUT_H + 220)) - 110
        draw.rectangle((0, scan_y, OUT_W, scan_y + (48 if not QUICK_MODE else 24)), fill=(80, 210, 240, 8))
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        name = shot["name"]
        image = self.background(t)
        if name == "intro":
            self.draw_intro(image, t)
        elif name == "coverage":
            self.draw_coverage(image, t)
        elif name == "darkness":
            self.draw_darkness(image, t)
        elif name == "climate":
            self.draw_climate(image, t)
        elif name == "ranking":
            self.draw_ranking(image, t)
        elif name == "flyover":
            self.draw_flyover(image, t)
        else:
            self.draw_finale(image, t)
        self.draw_source_hud(image)
        self.draw_titles(image, t, name)
        self.draw_caption(image, t)
        self.draw_hud_noise(image, t)
        array = apply_grade(np.asarray(image.convert("RGB")))
        array = np.clip(array.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / (0.9 if not QUICK_MODE else 0.18))
        fade_out = 1.0 - smoothstep((t - (float(CONFIG["duration_s"]) - (1.0 if not QUICK_MODE else 0.2))) / (0.9 if not QUICK_MODE else 0.18))
        return np.clip(array.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def render_video(scene: MilkyWayScene) -> Path:
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    print("Subtitle sidecar:", srt_path.resolve())
    raw_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    final_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    frame_count = int(round(float(CONFIG["duration_s"]) * int(CONFIG["fps"])))
    times = np.arange(frame_count) / int(CONFIG["fps"])
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(
        raw_video,
        fps=int(CONFIG["fps"]),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times, desc="Rendering Milky Way cinematic short"):
            writer.append_data(scene.render_frame(float(t)))
    shutil.copyfile(raw_video, final_video)
    print("Final video:", final_video.resolve())
    return final_video

def main():
    print("Loading NASA night-light map ...")
    night_map, night_source, night_notes = load_nightlight_map()
    print("Loading historical climate proxies ...")
    climate, climate_source, climate_notes, request_urls = load_climate_metrics()
    notes = night_notes + climate_notes
    print("Scoring Milky Way viewing candidates ...")
    scores, summary = score_sites(climate, night_map)
    ranking_path, summary_path = save_data_products(
        scores,
        summary,
        climate_source,
        night_source,
        notes,
        request_urls,
    )
    create_scientific_plots(scores)

    winner = scores.iloc[0]
    print("Climate source:", climate_source)
    print("Night-light source:", night_source)
    print("Candidate sites:", len(scores))
    print("Galactic Centre geometry uses declination approximately -29 degrees")
    print("Top Milky Way visibility candidate:", winner["name"], "//", winner["region"])
    print("Milky Way proxy score:", f"{winner['milky_way_score_100']:.2f}/100")
    print("Clear-sunshine fraction:", f"{winner['mean_clear_sunshine_fraction'] * 100.0:.1f}%")
    print("Annual precipitation proxy:", f"{winner['annual_precipitation_mm']:.1f} mm")
    print("Elevation:", f"{winner['elevation_m']:.0f} m")
    print("Galactic Centre maximum altitude:", f"{winner['galactic_core_max_altitude_deg']:.1f} degrees")
    for note in notes:
        print("Data note:", note)
    print("Ranking data:", ranking_path.resolve())
    print("Summary:", summary_path.resolve())

    scene = MilkyWayScene(scores, summary, night_map, climate_source, night_source)
    preview_times = [
        1.0,
        min(10.0, float(CONFIG["duration_s"]) * 0.20),
        min(20.0, float(CONFIG["duration_s"]) * 0.37),
        min(31.0, float(CONFIG["duration_s"]) * 0.56),
        min(40.0, float(CONFIG["duration_s"]) * 0.72),
        min(49.0, float(CONFIG["duration_s"]) * 0.86),
        float(CONFIG["duration_s"]) - 0.7,
    ]
    for preview_time in tqdm(preview_times, desc="Preview frames"):
        Image.fromarray(scene.render_frame(float(preview_time))).save(PREVIEW_DIR / f"preview_{int(preview_time):02d}s.png")
    render_video(scene)
    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-", path.name)


if __name__ == "__main__":
    main()


