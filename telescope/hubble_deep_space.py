from __future__ import annotations

"""
Result : https://www.youtube.com/shorts/IySP411Um2M
Hubble's Deepest Observations on One Sky Map — cinematic YouTube Short renderer

Creates a vertical 1080x1920 astronomy short from public Hubble Space Telescope
observation metadata in the Mikulski Archive for Space Telescopes (MAST). The
animation places Hubble image observations at their real sky coordinates, sums
exposure time in small equal-angle cells as a transparent depth proxy, labels
canonical Hubble deep fields, and plays the archive chronologically.

Preferred live source
---------------------
MAST observation service:
    https://mast.stsci.edu/api/v0/invoke

The renderer queries Mast.Caom.Filtered for public HST science image records
and uses archive-level metadata including:

- observation identifier
- target coordinates (ICRS right ascension and declination)
- observation start time (MJD)
- exposure time in seconds
- instrument, filter, target, and proposal metadata

Only records with exposure time at or above the configurable minimum are
requested by default. The returned observations are binned on the sky, and the
sum of metadata exposure times in each cell is visualized as a depth proxy.

Science notes
-------------
- Hubble is a pointed observatory, not an all-sky survey telescope.
- "Deep" can mean different things: total exposure, filter coverage, detector
  sensitivity, background, angular resolution, and processing all matter.
- This script uses cumulative archive exposure time per sky cell as a practical
  visual proxy. It is not a limiting-magnitude map or precision sensitivity map.
- MAST observation rows can represent products at different calibration levels;
  the script deduplicates primarily by archive observation ID before summing.
- The canonical field markers are reference positions, not true field outlines.
- Archive holdings and reprocessing can change, so totals may differ by run date.

Offline behaviour
-----------------
If MAST is unreachable, the script uses a clearly labelled deterministic
fixture containing clustered deep fields, repeated exposures, several Hubble
instrument generations, and a mission timeline. The fixture is for preview and
layout validation only and is not observational data.

Recommended install
-------------------
    pip install numpy pandas matplotlib pillow imageio imageio-ffmpeg tqdm

Quick preview render
--------------------
    HUBBLE_DEEP_SHORT_QUICK=1 python hubbles_deepest_observations_one_sky_map_short.py

Force offline fixture mode
--------------------------
    HUBBLE_DEEP_SHORT_OFFLINE=1 python hubbles_deepest_observations_one_sky_map_short.py

Use a previously downloaded CSV
-------------------------------
    HUBBLE_DEEP_DATA_PATH=/path/to/hst_mast_observations.csv \
        python hubbles_deepest_observations_one_sky_map_short.py

Primary references
------------------
- MAST API tutorial:
  https://mast.stsci.edu/api/v0/MastApiTutorial.html
- MAST CAOM field descriptions:
  https://mast.stsci.edu/api/v0/_c_a_o_mfields.html
- NASA Hubble deep-fields overview:
  https://science.nasa.gov/mission/hubble/science/universe-uncovered/hubble-deep-fields/
- MAST Hubble Frontier Fields archive:
  https://archive.stsci.edu/prepds/frontier/
"""

import json
import math
import os
import shutil
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("HUBBLE_DEEP_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("HUBBLE_DEEP_SHORT_OFFLINE", "0") == "1"
LOCAL_DATA_PATH = os.environ.get("HUBBLE_DEEP_DATA_PATH", "").strip()

OUTPUT_ROOT = Path("hubbles_deepest_observations_one_sky_map_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MAST_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "hubbles_deepest_observations_on_one_sky_map",
    "title": "HUBBLE'S DEEPEST OBSERVATIONS ON ONE SKY MAP",
    "subtitle": "public HST images // cumulative exposure-time depth proxy",
    "data_timeout_s": 45,
    "mast_page_size": 1200 if QUICK_MODE else 5000,
    "mast_max_pages": 5 if QUICK_MODE else 50,
    "max_archive_rows": 6000 if QUICK_MODE else 180000,
    "min_exposure_s": 700.0,
    "max_render_points": 2400 if QUICK_MODE else 10500,
    "fixture_rows": 3600 if QUICK_MODE else 18000,
    "background_stars": 280 if QUICK_MODE else 520,
    "hud_noise": 34 if QUICK_MODE else 68,
    "contrast": 1.08,
    "saturation": 1.06,
    "vignette": 0.25,
    "depth_cell_lon_deg": 1.5,
    "depth_cell_lat_deg": 1.5,
    "top_depth_cells": 100 if QUICK_MODE else 240,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
SQRT2 = math.sqrt(2.0)
MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)

COLORS = {
    "ice": (146, 224, 255),
    "cyan": (76, 229, 255),
    "blue": (72, 131, 255),
    "violet": (185, 110, 255),
    "gold": (255, 193, 89),
    "rose": (255, 99, 157),
    "green": (105, 242, 179),
    "orange": (255, 145, 77),
    "white": (245, 250, 255),
    "muted": (157, 203, 226),
    "dark": (3, 7, 17),
}

INSTRUMENT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "ACS": COLORS["cyan"],
    "WFC3": COLORS["violet"],
    "WFPC2": COLORS["gold"],
    "STIS": COLORS["rose"],
    "NICMOS": COLORS["green"],
    "OTHER": COLORS["ice"],
}

# Reference centres. Markers are intentionally points, not survey footprints.
DEEP_FIELDS: List[Dict[str, Any]] = [
    {"name": "HDF-N", "ra": 189.201083, "dec": 62.217219, "kind": "LEGACY"},
    {"name": "HDF-S", "ra": 338.229167, "dec": -60.552778, "kind": "LEGACY"},
    {"name": "HUDF / XDF", "ra": 53.160417, "dec": -27.783333, "kind": "ULTRA"},
    {"name": "GOODS-N", "ra": 189.229125, "dec": 62.237500, "kind": "SURVEY"},
    {"name": "GOODS-S", "ra": 53.125000, "dec": -27.805556, "kind": "SURVEY"},
    {"name": "COSMOS", "ra": 150.116667, "dec": 2.205833, "kind": "SURVEY"},
    {"name": "A2744", "ra": 3.586250, "dec": -30.400278, "kind": "FRONTIER"},
    {"name": "MACS0416", "ra": 64.038125, "dec": -24.067500, "kind": "FRONTIER"},
    {"name": "MACS0717", "ra": 109.391667, "dec": 37.755556, "kind": "FRONTIER"},
    {"name": "MACS1149", "ra": 177.398750, "dec": 22.398611, "kind": "FRONTIER"},
    {"name": "AS1063", "ra": 342.183750, "dec": -44.530833, "kind": "FRONTIER"},
    {"name": "A370", "ra": 39.970833, "dec": -1.576667, "kind": "FRONTIER"},
]

FULL_CAPTIONS = [
    (0.5, 7.2, "Hubble built its deepest views by returning to tiny, carefully chosen patches of sky for hours, days, and sometimes years."),
    (7.3, 17.2, "Every point here is a public Hubble science image record from MAST, placed at its real position on the sky."),
    (17.3, 27.2, "Now add the exposure time in each patch. The brightest cells are where Hubble collected the most light in this archive sample."),
    (27.3, 38.8, "The famous fields appear as pinpricks: HDF North and South, GOODS, the Ultra Deep Field, COSMOS, and the Frontier Fields."),
    (38.9, 49.6, "Play the observations in time and the deep sky accumulates—from early WFPC2 images to ACS, WFC3, and decades of repeat visits."),
    (49.7, 57.4, "A few tiny windows received extraordinary attention. Together they became Hubble's deepest observations on one sky map."),
]
if QUICK_MODE:
    _caption_scale = float(CONFIG["duration_s"]) / 58.0
    CAPTIONS = [(a * _caption_scale, b * _caption_scale, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.8 if not QUICK_MODE else 1.65},
    {"name": "all_sky", "start": 7.8 if not QUICK_MODE else 1.65, "end": 18.5 if not QUICK_MODE else 3.85},
    {"name": "depth", "start": 18.5 if not QUICK_MODE else 3.85, "end": 29.0 if not QUICK_MODE else 6.05},
    {"name": "fields", "start": 29.0 if not QUICK_MODE else 6.05, "end": 40.5 if not QUICK_MODE else 8.4},
    {"name": "timeline", "start": 40.5 if not QUICK_MODE else 8.4, "end": 50.5 if not QUICK_MODE else 10.45},
    {"name": "finale", "start": 50.5 if not QUICK_MODE else 10.45, "end": float(CONFIG["duration_s"])},
]


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


def evenly_subsample(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy().reset_index(drop=True)
    indices = np.linspace(0, len(frame) - 1, maximum).astype(int)
    return frame.iloc[indices].copy().reset_index(drop=True)


def safe_text(value: Any, default: str = "UNKNOWN") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "-999"}:
        return default
    return text


def mjd_to_datetime(value: float) -> Optional[datetime]:
    try:
        if not np.isfinite(float(value)):
            return None
        return MJD_EPOCH + timedelta(days=float(value))
    except Exception:
        return None


def datetime_to_mjd(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (value.astimezone(timezone.utc) - MJD_EPOCH).total_seconds() / 86400.0


def format_date_from_mjd(value: float) -> str:
    converted = mjd_to_datetime(value)
    return converted.strftime("%Y-%m-%d") if converted else "UNKNOWN"


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Coordinate transforms and projection
# -----------------------------------------------------------------------------

EQ_TO_GAL = np.array(
    [
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [0.4941094279, -0.4448296300, 0.7469822445],
        [-0.8676661490, -0.1980763734, 0.4559837762],
    ],
    dtype=float,
)


def equatorial_to_galactic(ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    xyz_eq = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    xyz_gal = xyz_eq @ EQ_TO_GAL.T
    lon = np.mod(np.rad2deg(np.arctan2(xyz_gal[:, 1], xyz_gal[:, 0])), 360.0)
    lat = np.rad2deg(np.arcsin(np.clip(xyz_gal[:, 2], -1.0, 1.0)))
    return lon, lat


def hammer_project(lon_deg: np.ndarray, lat_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon = np.deg2rad(((np.asarray(lon_deg, dtype=float) + 180.0) % 360.0) - 180.0)
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    denominator = np.sqrt(np.maximum(1.0 + np.cos(lat) * np.cos(lon / 2.0), 1e-12))
    x = -2.0 * SQRT2 * np.cos(lat) * np.sin(lon / 2.0) / denominator
    y = SQRT2 * np.sin(lat) / denominator
    return x, y


# -----------------------------------------------------------------------------
# MAST data loading
# -----------------------------------------------------------------------------

def mast_query(request_object: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    payload = urllib.parse.urlencode({"request": json.dumps(request_object)}).encode("utf-8")
    request = urllib.request.Request(
        MAST_INVOKE_URL,
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text/plain",
            "User-Agent": "hubble-deep-map-short-renderer/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        content = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(content)
    status = safe_text(parsed.get("status"), "")
    if status.upper() not in {"COMPLETE", "COMPLETED"}:
        raise RuntimeError(f"MAST query status was {status!r}: {safe_text(parsed.get('msg'), '')[:240]}")
    return parsed


def mast_filters() -> List[Dict[str, Any]]:
    return [
        {"paramName": "obs_collection", "values": ["HST"]},
        {"paramName": "intentType", "values": ["science"]},
        {"paramName": "dataRights", "values": ["PUBLIC"]},
        {"paramName": "dataproduct_type", "values": ["IMAGE"]},
        {"paramName": "t_exptime", "values": [{"min": float(CONFIG["min_exposure_s"]), "max": 1.0e9}]},
    ]


def fetch_mast_observations() -> Tuple[pd.DataFrame, List[str], List[Dict[str, Any]]]:
    notes: List[str] = []
    requests_used: List[Dict[str, Any]] = []
    pieces: List[pd.DataFrame] = []
    page_size = int(CONFIG["mast_page_size"])
    max_pages = int(CONFIG["mast_max_pages"])
    max_rows = int(CONFIG["max_archive_rows"])

    for page in range(1, max_pages + 1):
        request_object = {
            "service": "Mast.Caom.Filtered",
            "format": "json",
            "params": {"columns": "*", "filters": mast_filters()},
            "pagesize": page_size,
            "page": page,
            "removenullcolumns": False,
        }
        requests_used.append(request_object)
        try:
            result = mast_query(request_object, timeout_s=float(CONFIG["data_timeout_s"]))
        except Exception as exc:
            if page == 1:
                raise
            notes.append(f"MAST page {page} failed after earlier pages succeeded: {exc}")
            break
        rows = result.get("data") or []
        if not rows:
            break
        pieces.append(pd.DataFrame(rows))
        current_rows = sum(len(piece) for piece in pieces)
        paging = result.get("paging") or {}
        pages_filtered = int(paging.get("pagesFiltered") or 0)
        if current_rows >= max_rows:
            notes.append(f"Live query capped at {max_rows:,} rows before normalization")
            break
        if len(rows) < page_size or (pages_filtered and page >= pages_filtered):
            break

    if not pieces:
        raise RuntimeError("MAST returned no public HST science image rows")
    combined = pd.concat(pieces, ignore_index=True)
    if len(combined) > max_rows:
        combined = combined.iloc[:max_rows].copy()
    return combined, notes, requests_used


def normalize_instrument(value: Any) -> str:
    text = safe_text(value).upper().replace("-", "").replace("_", "")
    if text.startswith("ACS"):
        return "ACS"
    if text.startswith("WFC3"):
        return "WFC3"
    if text.startswith("WFPC2") or text.startswith("WFPC"):
        return "WFPC2"
    if text.startswith("STIS"):
        return "STIS"
    if text.startswith("NICMOS") or text.startswith("NIC"):
        return "NICMOS"
    return "OTHER"


def normalize_observations(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    frame = raw.copy().rename(columns={column: column.strip() for column in raw.columns})
    lookup = {column.lower(): column for column in frame.columns}

    def column(candidates: Sequence[str]) -> Optional[str]:
        for candidate in candidates:
            found = lookup.get(candidate.lower())
            if found is not None:
                return found
        return None

    ra_col = column(["s_ra", "ra", "ra_deg"])
    dec_col = column(["s_dec", "dec", "dec_deg"])
    if ra_col is None or dec_col is None:
        raise RuntimeError("Observation table must contain s_ra/s_dec or compatible RA/Dec columns")

    result = pd.DataFrame({
        "s_ra": pd.to_numeric(frame[ra_col], errors="coerce"),
        "s_dec": pd.to_numeric(frame[dec_col], errors="coerce"),
    })
    mappings = {
        "obsid": ["obsid", "obsID"],
        "obs_id": ["obs_id", "observation_id"],
        "instrument_name": ["instrument_name", "instrument"],
        "filters": ["filters", "filter"],
        "target_name": ["target_name", "target"],
        "t_min": ["t_min", "mjd", "start_mjd"],
        "t_max": ["t_max", "end_mjd"],
        "t_exptime": ["t_exptime", "exposure_time", "exptime"],
        "proposal_id": ["proposal_id", "program", "program_id"],
        "proposal_pi": ["proposal_pi", "pi"],
        "obs_title": ["obs_title", "title"],
        "dataproduct_type": ["dataproduct_type", "product_type"],
        "calib_level": ["calib_level", "calibration_level"],
        "dataRights": ["datarights", "data_rights"],
        "intentType": ["intenttype", "intent_type"],
        "s_region": ["s_region", "region"],
    }
    for canonical, candidates in mappings.items():
        source_column = column(candidates)
        result[canonical] = frame[source_column] if source_column is not None else ""

    for numeric in ("t_min", "t_max", "t_exptime", "calib_level"):
        result[numeric] = pd.to_numeric(result[numeric], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["s_ra", "s_dec", "t_exptime"])
    result = result[
        result["s_ra"].between(0.0, 360.0)
        & result["s_dec"].between(-90.0, 90.0)
        & (result["t_exptime"] > 0)
    ].copy()

    if source == "mast_public_hst_image_observations":
        rights = result["dataRights"].astype(str).str.upper()
        intent = result["intentType"].astype(str).str.lower()
        product = result["dataproduct_type"].astype(str).str.upper()
        result = result[
            (rights.eq("PUBLIC") | rights.eq(""))
            & (intent.eq("science") | intent.eq(""))
            & (product.eq("IMAGE") | product.eq(""))
        ].copy()

    obsid = result["obsid"].astype(str).str.strip()
    obs_id = result["obs_id"].astype(str).str.strip()
    fallback_key = (
        result["s_ra"].round(7).astype(str) + ":" + result["s_dec"].round(7).astype(str)
        + ":" + result["t_min"].round(6).astype(str) + ":" + result["instrument_name"].astype(str)
    )
    invalid_obsid = obsid.isin({"", "nan", "None", "null", "-999"})
    invalid_obs_id = obs_id.isin({"", "nan", "None", "null", "-999"})
    result["_archive_key"] = np.where(
        ~invalid_obsid,
        "obsid:" + obsid,
        np.where(~invalid_obs_id, "obs_id:" + obs_id, "coord:" + fallback_key),
    )
    result = result.drop_duplicates("_archive_key").drop(columns="_archive_key").reset_index(drop=True)
    result["instrument_group"] = result["instrument_name"].map(normalize_instrument)
    result["target_name"] = result["target_name"].map(lambda value: safe_text(value, "UNNAMED TARGET"))
    result["proposal_id"] = result["proposal_id"].map(lambda value: safe_text(value, "UNKNOWN"))
    result["data_source"] = source
    return result


def load_local_observations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return normalize_observations(pd.read_csv(path), "local_mast_compatible_csv")


# -----------------------------------------------------------------------------
# Deterministic offline fixture
# -----------------------------------------------------------------------------

def fallback_observations() -> Tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(19901204)
    target = int(CONFIG["fixture_rows"])
    rows: List[Dict[str, Any]] = []
    instruments = np.array(["ACS/WFC", "WFC3/IR", "WFPC2/WFC", "STIS/CCD", "NICMOS/NIC3"])
    probabilities = np.array([0.34, 0.32, 0.18, 0.10, 0.06])
    start = datetime(1990, 5, 20, tzinfo=timezone.utc)
    end = datetime(2026, 7, 1, tzinfo=timezone.utc)
    mission_days = (end - start).total_seconds() / 86400.0

    # Named fields get dense repeated visits with long-tailed exposures.
    allocations = [0.10, 0.07, 0.18, 0.10, 0.10, 0.08] + [0.035] * 6
    allocation_total = sum(allocations)
    for field, fraction in zip(DEEP_FIELDS, allocations):
        count = max(30, int(target * fraction / allocation_total * 0.72))
        for _ in range(count):
            instrument = str(rng.choice(instruments, p=probabilities))
            jitter = rng.normal(0.0, 0.04 if field["kind"] != "SURVEY" else 0.12, 2)
            exposure = float(np.exp(rng.normal(math.log(1550 if field["kind"] == "FRONTIER" else 2200), 0.75)))
            date = start + timedelta(days=float(rng.uniform(0.12, 1.0) * mission_days))
            rows.append({
                "obsid": len(rows) + 1,
                "obs_id": f"fixture_{len(rows)+1:06d}",
                "s_ra": (field["ra"] + jitter[0] / max(math.cos(math.radians(field["dec"])), 0.25)) % 360.0,
                "s_dec": float(np.clip(field["dec"] + jitter[1], -89.8, 89.8)),
                "t_min": datetime_to_mjd(date),
                "t_max": datetime_to_mjd(date + timedelta(seconds=exposure)),
                "t_exptime": exposure,
                "instrument_name": instrument,
                "filters": "F606W;F814W" if "ACS" in instrument else "F125W;F160W",
                "target_name": field["name"],
                "proposal_id": str(rng.integers(5000, 18000)),
                "proposal_pi": "FIXTURE",
                "obs_title": f"Synthetic preview near {field['name']}",
                "dataproduct_type": "IMAGE",
                "calib_level": 2,
                "dataRights": "PUBLIC",
                "intentType": "science",
                "s_region": "",
            })

    # Broad pointed-observatory background, concentrated away from the Galactic plane.
    remaining = max(0, target - len(rows))
    for _ in range(remaining):
        instrument = str(rng.choice(instruments, p=probabilities))
        ra = float(rng.uniform(0, 360))
        dec = float(np.degrees(np.arcsin(rng.uniform(-1, 1))))
        exposure = float(np.exp(rng.normal(math.log(1050), 0.72)))
        date = start + timedelta(days=float(rng.uniform(0.0, 1.0) * mission_days))
        rows.append({
            "obsid": len(rows) + 1,
            "obs_id": f"fixture_{len(rows)+1:06d}",
            "s_ra": ra,
            "s_dec": dec,
            "t_min": datetime_to_mjd(date),
            "t_max": datetime_to_mjd(date + timedelta(seconds=exposure)),
            "t_exptime": exposure,
            "instrument_name": instrument,
            "filters": "F606W",
            "target_name": f"FIXTURE-TARGET-{rng.integers(1, 900):03d}",
            "proposal_id": str(rng.integers(5000, 18000)),
            "proposal_pi": "FIXTURE",
            "obs_title": "Synthetic HST pointing",
            "dataproduct_type": "IMAGE",
            "calib_level": 2,
            "dataRights": "PUBLIC",
            "intentType": "science",
            "s_region": "",
        })
    frame = normalize_observations(pd.DataFrame(rows), "offline_hubble_deep_fixture")
    return frame, "offline_hubble_deep_fixture"


def load_all_data() -> Tuple[pd.DataFrame, str, List[str], List[Dict[str, Any]]]:
    notes: List[str] = []
    requests_used: List[Dict[str, Any]] = []
    if LOCAL_DATA_PATH:
        try:
            frame = load_local_observations(Path(LOCAL_DATA_PATH).expanduser())
            notes.append(f"Loaded HUBBLE_DEEP_DATA_PATH={LOCAL_DATA_PATH}")
            return frame, "local_mast_compatible_csv", notes, requests_used
        except Exception as exc:
            notes.append(f"Local catalogue failed: {exc}")
    if OFFLINE_MODE:
        notes.append("Offline mode requested with HUBBLE_DEEP_SHORT_OFFLINE=1")
        frame, source = fallback_observations()
        return frame, source, notes, requests_used
    try:
        raw, live_notes, requests_used = fetch_mast_observations()
        notes.extend(live_notes)
        frame = normalize_observations(raw, "mast_public_hst_image_observations")
        if len(frame) < 250:
            raise RuntimeError(f"Only {len(frame)} valid HST image observation records returned")
        return frame, "mast_public_hst_image_observations", notes, requests_used
    except Exception as exc:
        notes.append(f"MAST fallback: {exc}")
        frame, source = fallback_observations()
        return frame, source, notes, requests_used


# -----------------------------------------------------------------------------
# Analysis and data products
# -----------------------------------------------------------------------------

def prepare_observations(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    out = frame.copy().reset_index(drop=True)
    gal_lon, gal_lat = equatorial_to_galactic(out["s_ra"].to_numpy(float), out["s_dec"].to_numpy(float))
    out["galactic_lon_deg"] = gal_lon
    out["galactic_lat_deg"] = gal_lat
    hx, hy = hammer_project(gal_lon, gal_lat)
    out["hammer_x"] = hx
    out["hammer_y"] = hy

    valid_time = np.isfinite(out["t_min"].to_numpy(float))
    if valid_time.any():
        time_low = float(np.nanmin(out.loc[valid_time, "t_min"]))
        time_high = float(np.nanmax(out.loc[valid_time, "t_min"]))
    else:
        time_low, time_high = 0.0, 1.0
    out["time_norm"] = np.where(
        valid_time,
        np.clip((out["t_min"] - time_low) / max(time_high - time_low, 1e-9), 0.0, 1.0),
        0.5,
    )

    lon_cell = float(CONFIG["depth_cell_lon_deg"])
    lat_cell = float(CONFIG["depth_cell_lat_deg"])
    out["depth_lon_bin"] = np.floor(out["galactic_lon_deg"] / lon_cell).astype(int)
    out["depth_lat_bin"] = np.floor((out["galactic_lat_deg"] + 90.0) / lat_cell).astype(int)
    cells = (
        out.groupby(["depth_lon_bin", "depth_lat_bin"], as_index=False)
        .agg(
            observation_records=("obs_id", "size"),
            total_exposure_s=("t_exptime", "sum"),
            median_exposure_s=("t_exptime", "median"),
            first_mjd=("t_min", "min"),
            last_mjd=("t_min", "max"),
        )
        .sort_values("total_exposure_s", ascending=False)
        .reset_index(drop=True)
    )
    cells["galactic_lon_center_deg"] = (cells["depth_lon_bin"] + 0.5) * lon_cell
    cells["galactic_lat_center_deg"] = (cells["depth_lat_bin"] + 0.5) * lat_cell - 90.0
    cx, cy = hammer_project(cells["galactic_lon_center_deg"], cells["galactic_lat_center_deg"])
    cells["hammer_x"] = cx
    cells["hammer_y"] = cy
    max_depth = max(float(cells["total_exposure_s"].max()), 1.0)
    cells["depth_norm"] = np.log1p(cells["total_exposure_s"]) / math.log1p(max_depth)

    out = out.merge(
        cells[["depth_lon_bin", "depth_lat_bin", "total_exposure_s", "depth_norm"]].rename(
            columns={"total_exposure_s": "cell_total_exposure_s"}
        ),
        on=["depth_lon_bin", "depth_lat_bin"],
        how="left",
    )
    out = out.sort_values(["t_min", "instrument_group", "obs_id"], na_position="last").reset_index(drop=True)

    fields = pd.DataFrame(DEEP_FIELDS)
    flon, flat = equatorial_to_galactic(fields["ra"].to_numpy(float), fields["dec"].to_numpy(float))
    fields["galactic_lon_deg"] = flon
    fields["galactic_lat_deg"] = flat
    fx, fy = hammer_project(flon, flat)
    fields["hammer_x"] = fx
    fields["hammer_y"] = fy
    # Summarise archive records within a generous radius for an informative field label.
    field_rows: List[Dict[str, Any]] = []
    ra_rad = np.deg2rad(out["s_ra"].to_numpy(float))
    dec_rad = np.deg2rad(out["s_dec"].to_numpy(float))
    for _, row in fields.iterrows():
        fra = math.radians(float(row["ra"]))
        fdec = math.radians(float(row["dec"]))
        cosine = np.sin(dec_rad) * math.sin(fdec) + np.cos(dec_rad) * math.cos(fdec) * np.cos(ra_rad - fra)
        separation = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
        radius = 0.35 if row["kind"] in {"SURVEY", "FRONTIER"} else 0.18
        near = out[separation <= radius]
        item = row.to_dict()
        item["search_radius_deg"] = radius
        item["observation_records"] = int(len(near))
        item["total_exposure_s"] = float(np.nansum(near["t_exptime"]))
        field_rows.append(item)
    fields = pd.DataFrame(field_rows).sort_values("total_exposure_s", ascending=False).reset_index(drop=True)

    instrument_counts = out["instrument_group"].value_counts().to_dict()
    valid_dates = out.loc[np.isfinite(out["t_min"]), "t_min"]
    total_exposure_s = float(np.nansum(np.clip(out["t_exptime"].to_numpy(float), 0.0, None)))
    summary = {
        "source": str(out["data_source"].iloc[0]),
        "is_live_observational_data": str(out["data_source"].iloc[0]) in {
            "mast_public_hst_image_observations", "local_mast_compatible_csv"
        },
        "observation_records": int(len(out)),
        "unique_targets": int(out["target_name"].nunique()),
        "unique_programmes": int(out["proposal_id"].nunique()),
        "occupied_depth_cells": int(len(cells)),
        "total_exposure_hours_metadata_sum": total_exposure_s / 3600.0,
        "deepest_cell_exposure_hours": float(cells.iloc[0]["total_exposure_s"] / 3600.0) if len(cells) else 0.0,
        "deepest_cell_records": int(cells.iloc[0]["observation_records"]) if len(cells) else 0,
        "date_start_mjd": float(valid_dates.min()) if len(valid_dates) else None,
        "date_end_mjd": float(valid_dates.max()) if len(valid_dates) else None,
        "date_start": format_date_from_mjd(float(valid_dates.min())) if len(valid_dates) else "UNKNOWN",
        "date_end": format_date_from_mjd(float(valid_dates.max())) if len(valid_dates) else "UNKNOWN",
        "instrument_counts": {str(key): int(value) for key, value in instrument_counts.items()},
        "canonical_fields": fields.to_dict(orient="records"),
        "projection": "Hammer projection in Galactic longitude and latitude",
        "depth_proxy": f"Sum of t_exptime metadata in {lon_cell:g}° × {lat_cell:g}° Galactic-coordinate cells",
        "query_minimum_exposure_s": float(CONFIG["min_exposure_s"]),
        "warning": "Exposure-time sum is a depth proxy, not a limiting-magnitude or sensitivity map.",
    }
    return out, cells, fields, summary


def save_data_products(
    frame: pd.DataFrame,
    cells: pd.DataFrame,
    fields: pd.DataFrame,
    summary: Dict[str, Any],
    notes: List[str],
    requests_used: List[Dict[str, Any]],
) -> Tuple[Path, Path]:
    catalogue_path = DATA_ROOT / "hst_public_deep_image_observations.csv"
    cells_path = DATA_ROOT / "hst_exposure_depth_cells.csv"
    fields_path = DATA_ROOT / "hst_canonical_deep_fields.csv"
    summary_path = DATA_ROOT / "hubble_deep_sky_map_summary.json"
    frame.to_csv(catalogue_path, index=False)
    cells.to_csv(cells_path, index=False)
    fields.to_csv(fields_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "notes": notes,
                "mast_endpoint": MAST_INVOKE_URL,
                "mast_request_template": {
                    "service": "Mast.Caom.Filtered",
                    "format": "json",
                    "params": {"columns": "*", "filters": mast_filters()},
                },
                "requests_used": requests_used,
                "fallback_warning": "offline_hubble_deep_fixture is deterministic synthetic preview data, not observational data",
                "science_warning": "Cumulative metadata exposure time is only a visual depth proxy; detector, filter, sky background, overlap geometry, calibration, and processing also affect real depth.",
                "source_urls": {
                    "mast_api_tutorial": "https://mast.stsci.edu/api/v0/MastApiTutorial.html",
                    "mast_caom_fields": "https://mast.stsci.edu/api/v0/_c_a_o_mfields.html",
                    "nasa_hubble_deep_fields": "https://science.nasa.gov/mission/hubble/science/universe-uncovered/hubble-deep-fields/",
                    "nasa_hudf": "https://science.nasa.gov/asset/hubble/hubble-ultra-deep-field/",
                    "nasa_xdf": "https://science.nasa.gov/asset/hubble/hubble-extreme-deep-field-xdf/",
                    "mast_frontier_fields": "https://archive.stsci.edu/prepds/frontier/",
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return catalogue_path, summary_path


def create_scientific_plots(frame: pd.DataFrame, cells: pd.DataFrame, fields: pd.DataFrame, summary: Dict[str, Any]):
    sample = evenly_subsample(frame, 18000)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    scatter = ax.scatter(sample["hammer_x"], sample["hammer_y"], c=np.log10(sample["t_exptime"]), s=2, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Public HST science image records used by the depth map")
    ax.set_xlabel("Hammer x")
    ax.set_ylabel("Hammer y")
    fig.colorbar(scatter, ax=ax, label="log10 exposure time (s)")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "hst_all_sky_image_records.png", dpi=170)
    plt.close(fig)

    top = cells.head(2000)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    scatter = ax.scatter(top["hammer_x"], top["hammer_y"], c=np.log10(top["total_exposure_s"]), s=8, alpha=0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("HST cumulative exposure-time depth proxy")
    ax.set_xlabel("Hammer x")
    ax.set_ylabel("Hammer y")
    fig.colorbar(scatter, ax=ax, label="log10 cumulative exposure (s)")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "hst_exposure_depth_proxy.png", dpi=170)
    plt.close(fig)

    dated = frame[np.isfinite(frame["t_min"])].sort_values("t_min")
    if len(dated):
        fig, ax = plt.subplots(figsize=(10, 5.2))
        dates = [mjd_to_datetime(value) for value in dated["t_min"].to_numpy(float)]
        cumulative_hours = np.cumsum(dated["t_exptime"].to_numpy(float)) / 3600.0
        ax.plot(dates, cumulative_hours)
        ax.set_title("Cumulative exposure time in the queried HST sample")
        ax.set_xlabel("Observation date")
        ax.set_ylabel("Cumulative exposure (hours)")
        plt.tight_layout()
        plt.savefig(PREVIEW_DIR / "hst_cumulative_exposure_timeline.png", dpi=170)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    field_plot = fields.sort_values("total_exposure_s", ascending=True)
    ax.barh(field_plot["name"], field_plot["total_exposure_s"] / 3600.0)
    ax.set_title("Exposure-time proxy near canonical Hubble deep fields")
    ax.set_xlabel("Summed exposure time within reference radius (hours)")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "hst_named_deep_field_exposure.png", dpi=170)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class HubbleDeepSkyScene:
    def __init__(self, frame: pd.DataFrame, cells: pd.DataFrame, fields: pd.DataFrame, summary: Dict[str, Any]):
        render = frame.copy()
        if len(render) > int(CONFIG["max_render_points"]):
            # Preserve chronology and the long-exposure tail.
            deep_half = render.nlargest(int(CONFIG["max_render_points"] * 0.48), "t_exptime")
            time_half = evenly_subsample(render.sort_values("t_min", na_position="last"), int(CONFIG["max_render_points"] * 0.62))
            render = pd.concat([deep_half, time_half], ignore_index=True).drop_duplicates("obs_id")
            render = evenly_subsample(render.sort_values("t_min", na_position="last"), int(CONFIG["max_render_points"]))
        self.frame = render.reset_index(drop=True)
        self.cells = cells.head(int(CONFIG["top_depth_cells"])).copy().reset_index(drop=True)
        self.fields = fields.copy().reset_index(drop=True)
        self.summary = summary
        self.stars = self._make_stars(int(CONFIG["background_stars"]), seed=1990)
        self.hud = self._make_hud(int(CONFIG["hud_noise"]), seed=2026)
        self.map_box = (int(OUT_W * 0.065), int(OUT_H * 0.22), int(OUT_W * 0.935), int(OUT_H * 0.68))
        self.point_xy = self._to_screen(self.frame["hammer_x"].to_numpy(float), self.frame["hammer_y"].to_numpy(float))
        self.cell_xy = self._to_screen(self.cells["hammer_x"].to_numpy(float), self.cells["hammer_y"].to_numpy(float))
        self.field_xy = self._to_screen(self.fields["hammer_x"].to_numpy(float), self.fields["hammer_y"].to_numpy(float))
        self.instrument = self.frame["instrument_group"].astype(str).to_numpy()
        self.time_norm = self.frame["time_norm"].to_numpy(float)
        self.exposure_norm = np.log1p(self.frame["t_exptime"].to_numpy(float))
        self.exposure_norm /= max(float(np.nanmax(self.exposure_norm)), 1e-9)
        self.order_all = np.arange(len(self.frame))
        self.order_time = np.argsort(self.time_norm)
        self.order_exposure = np.argsort(self.exposure_norm)
        self.max_cell_exposure = max(float(self.cells["total_exposure_s"].max()) if len(self.cells) else 1.0, 1.0)

    @staticmethod
    def _make_stars(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [{"x": float(rng.uniform(0, OUT_W)), "y": float(rng.uniform(0, OUT_H)), "r": float(rng.uniform(0.3, 2.0)), "a": float(rng.uniform(18, 105)), "phase": float(rng.uniform(0, 2 * math.pi))} for _ in range(count)]

    @staticmethod
    def _make_hud(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [{"x": float(rng.uniform(0, OUT_W)), "y": float(rng.uniform(0, OUT_H)), "length": float(rng.uniform(10, 95)), "a": float(rng.uniform(8, 42)), "phase": float(rng.uniform(0, 2 * math.pi))} for _ in range(count)]

    def _to_screen(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.map_box
        sx = x0 + (np.asarray(x) + 2.0 * SQRT2) / (4.0 * SQRT2) * (x1 - x0)
        sy = y1 - (np.asarray(y) + SQRT2) / (2.0 * SQRT2) * (y1 - y0)
        return np.column_stack([sx, sy])

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, (2, 6, 16, 255))
        draw = ImageDraw.Draw(image)
        for star in self.stars:
            alpha = int(star["a"] * (0.72 + 0.28 * math.sin(t * 1.35 + star["phase"])))
            r = star["r"]
            draw.ellipse((star["x"] - r, star["y"] - r, star["x"] + r, star["y"] + r), fill=(220, 235, 255, alpha))
        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for cx, cy, color in [
            (OUT_W * 0.18, OUT_H * 0.28, (30, 48, 145)),
            (OUT_W * 0.82, OUT_H * 0.40, (76, 24, 126)),
            (OUT_W * 0.52, OUT_H * 0.76, (8, 82, 124)),
        ]:
            for radius, alpha in [(430 * OUT_W / 1080, 14), (280 * OUT_W / 1080, 22), (165 * OUT_W / 1080, 30)]:
                hd.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color + (alpha,))
        image.alpha_composite(haze.filter(ImageFilter.GaussianBlur(62 if not QUICK_MODE else 31)))
        return image

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 170):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(box, radius=24 if not QUICK_MODE else 12, fill=(2, 7, 18, alpha), outline=(100, 200, 235, 64), width=1)
        image.alpha_composite(overlay)

    def draw_hubble(self, image: Image.Image, center: Tuple[float, float], scale: float, phase: float, alpha: int = 225):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cx, cy = center
        tilt = 0.08 * math.sin(phase * 0.7)
        # Simple cylindrical telescope body and solar arrays.
        body_w, body_h = 2.6 * scale, 0.78 * scale
        draw.rounded_rectangle((cx - body_w / 2, cy - body_h / 2, cx + body_w / 2, cy + body_h / 2), radius=int(scale * 0.18), fill=COLORS["ice"] + (int(alpha * 0.22),), outline=COLORS["ice"] + (alpha,), width=max(1, int(scale * 0.06)))
        draw.ellipse((cx - body_w / 2 - scale * 0.18, cy - body_h / 2, cx - body_w / 2 + scale * 0.35, cy + body_h / 2), outline=COLORS["gold"] + (alpha,), width=max(1, int(scale * 0.08)))
        for side in (-1, 1):
            x0 = cx + side * body_w * 0.55
            draw.rectangle((x0 - scale * 0.85, cy - scale * 1.15, x0 + scale * 0.85, cy - scale * 0.62), fill=COLORS["blue"] + (int(alpha * 0.40),), outline=COLORS["cyan"] + (int(alpha * 0.8),))
            for grid in range(1, 4):
                gx = x0 - scale * 0.85 + grid * scale * 0.425
                draw.line((gx, cy - scale * 1.15, gx, cy - scale * 0.62), fill=COLORS["cyan"] + (90,), width=1)
        draw.line((cx - scale * 0.8, cy + body_h / 2, cx - scale * 1.15, cy + scale * 1.25), fill=COLORS["gold"] + (alpha,), width=max(1, int(scale * 0.08)))
        draw.line((cx + scale * 0.8, cy + body_h / 2, cx + scale * 1.15, cy + scale * 1.25), fill=COLORS["gold"] + (alpha,), width=max(1, int(scale * 0.08)))
        image.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(12 if not QUICK_MODE else 6)))
        image.alpha_composite(overlay)

    def draw_sky_grid(self, image: Image.Image, alpha: int = 80, label: bool = True):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        x0, y0, x1, y1 = self.map_box
        draw.rounded_rectangle(self.map_box, radius=int((y1 - y0) * 0.47), fill=(1, 7, 18, 154), outline=COLORS["cyan"] + (80,), width=2 if not QUICK_MODE else 1)
        for lat in (-60, -30, 0, 30, 60):
            lon_values = np.linspace(-180, 180, 240)
            px, py = hammer_project(lon_values % 360.0, np.full_like(lon_values, lat))
            screen = self._to_screen(px, py)
            draw.line([tuple(point) for point in screen], fill=COLORS["muted"] + (alpha if lat else min(150, alpha + 65),), width=1)
        for lon in range(-150, 180, 30):
            lat_values = np.linspace(-89.5, 89.5, 220)
            px, py = hammer_project(np.full_like(lat_values, lon % 360.0), lat_values)
            screen = self._to_screen(px, py)
            draw.line([tuple(point) for point in screen], fill=COLORS["muted"] + (alpha,), width=1)
        plane = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(plane)
        lon_values = np.linspace(-180, 180, 300)
        px, py = hammer_project(lon_values % 360.0, np.zeros_like(lon_values))
        pd.line([tuple(point) for point in self._to_screen(px, py)], fill=COLORS["gold"] + (66,), width=8 if not QUICK_MODE else 4)
        overlay.alpha_composite(plane.filter(ImageFilter.GaussianBlur(8 if not QUICK_MODE else 4)))
        image.alpha_composite(overlay)
        if label:
            draw_text(image, "GALACTIC COORDINATES // MILKY WAY PLANE", (x0 + (18 if not QUICK_MODE else 9), y1 + (28 if not QUICK_MODE else 14)), size=15 if not QUICK_MODE else 7, fill=COLORS["muted"] + (190,), bold=True, stroke=1)

    def draw_points(self, image: Image.Image, indices: np.ndarray, reveal: float = 1.0, mode: str = "ice", alpha: int = 205, size_boost: float = 1.0):
        count = int(round(len(indices) * clamp(reveal)))
        if count <= 0:
            return
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw, gd = ImageDraw.Draw(overlay), ImageDraw.Draw(glow)
        base_radius = (2.0 if not QUICK_MODE else 1.0) * size_boost
        for index in indices[:count]:
            index = int(index)
            x, y = self.point_xy[index]
            if mode == "instrument":
                colour = INSTRUMENT_COLORS.get(self.instrument[index], COLORS["ice"])
            elif mode == "exposure":
                value = float(self.exposure_norm[index])
                colour = tuple(int(round(lerp(COLORS["blue"][c], COLORS["rose"][c], value))) for c in range(3))
            else:
                colour = COLORS["ice"]
            radius = base_radius * (0.8 + 0.9 * float(self.exposure_norm[index]))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour + (alpha,))
            if index % (80 if not QUICK_MODE else 36) == 0:
                gr = radius * 4.0
                gd.ellipse((x - gr, y - gr, x + gr, y + gr), fill=colour + (36,))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(7 if not QUICK_MODE else 3)))
        image.alpha_composite(overlay)

    def draw_depth_cells(self, image: Image.Image, local: float, labels: bool = False):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for index, row in self.cells.iterrows():
            x, y = self.cell_xy[index]
            weight = math.log1p(float(row["total_exposure_s"])) / math.log1p(self.max_cell_exposure)
            pulse = 0.65 + 0.35 * math.sin(local * 7.0 + index * 0.73)
            radius = (6 + 31 * weight * (0.65 + 0.35 * local)) * OUT_W / 1080
            colour = tuple(int(round(lerp(COLORS["violet"][c], COLORS["rose"][c], weight))) for c in range(3))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=colour + (int(60 + 150 * weight * pulse),), width=max(1, int(2.0 * OUT_W / 1080)))
            if index < 8:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=COLORS["white"] + (220,))
        image.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(0.4 if not QUICK_MODE else 0.2)))

    def draw_field_markers(self, image: Image.Image, reveal: float = 1.0):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        count = int(round(len(self.fields) * clamp(reveal)))
        for index in range(count):
            row = self.fields.iloc[index]
            x, y = self.field_xy[index]
            colour = COLORS["rose"] if row["kind"] == "ULTRA" else COLORS["gold"] if row["kind"] == "FRONTIER" else COLORS["cyan"]
            radius = (7 if not QUICK_MODE else 3.5) + (2 if row["kind"] == "ULTRA" else 0)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour + (245,), outline=COLORS["white"] + (220,), width=1)
            # Alternate label direction to reduce overlap.
            anchor = "la" if (index % 2 == 0 or x < OUT_W * 0.35) else "ra"
            offset = 13 if anchor == "la" else -13
            draw_text(overlay, str(row["name"]), (int(x + offset), int(y - 10)), size=14 if not QUICK_MODE else 7, fill=colour + (235,), bold=True, anchor=anchor, stroke=1)
        image.alpha_composite(overlay)

    def draw_intro(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[0]
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        center = (OUT_W * 0.50, OUT_H * 0.39)
        self.draw_hubble(image, center, 70 * OUT_W / 1080, phase=t, alpha=int(225 * (0.45 + 0.55 * local)))
        draw = ImageDraw.Draw(image)
        rng = np.random.default_rng(1990)
        for index in range(150 if not QUICK_MODE else 65):
            angle = rng.uniform(0, 2 * math.pi) + t * (0.10 + 0.03 * (index % 5))
            target_r = rng.uniform(120, 420) * OUT_W / 1080
            radius = lerp(target_r * 1.7, target_r, local)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + 0.66 * radius * math.sin(angle)
            r = rng.uniform(0.8, 2.3) * OUT_W / 1080
            draw.ellipse((x-r, y-r, x+r, y+r), fill=COLORS["cyan"] + (int(35 + 145 * local * rng.uniform(0.3, 1.0)),))
        self.panel(image, (int(OUT_W * 0.10), int(OUT_H * 0.68), int(OUT_W * 0.90), int(OUT_H * 0.80)), alpha=158)
        draw_text(image, "DEPTH COMES FROM COLLECTING MORE LIGHT", (OUT_W // 2, int(OUT_H * 0.716)), size=22 if not QUICK_MODE else 11, fill=COLORS["gold"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "repeat visits • long exposures • carefully chosen windows", (OUT_W // 2, int(OUT_H * 0.755)), size=17 if not QUICK_MODE else 8, fill=COLORS["white"] + (220,), anchor="ma", stroke=1)

    def draw_all_sky(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "all_sky")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_sky_grid(image, alpha=45, label=True)
        self.draw_points(image, self.order_all, reveal=min(local * 1.18, 1.0), mode="ice", alpha=200)
        self.panel(image, (int(OUT_W * 0.09), int(OUT_H * 0.71), int(OUT_W * 0.91), int(OUT_H * 0.82)), alpha=154)
        draw_text(image, "PUBLIC HUBBLE IMAGE RECORDS ON THE REAL SKY", (OUT_W // 2, int(OUT_H * 0.748)), size=21 if not QUICK_MODE else 10, fill=COLORS["cyan"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(image, f"{self.summary['observation_records']:,} deduplicated records above {CONFIG['min_exposure_s']:.0f} seconds", (OUT_W // 2, int(OUT_H * 0.785)), size=17 if not QUICK_MODE else 8, fill=COLORS["white"] + (220,), anchor="ma", stroke=1)

    def draw_depth(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "depth")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_sky_grid(image, alpha=32, label=False)
        self.draw_points(image, self.order_exposure, reveal=1.0, mode="exposure", alpha=135, size_boost=0.92)
        self.draw_depth_cells(image, local)
        self.panel(image, (int(OUT_W * 0.08), int(OUT_H * 0.70), int(OUT_W * 0.92), int(OUT_H * 0.83)), alpha=160)
        draw_text(image, "SUM THE EXPOSURE TIME IN EACH PATCH", (OUT_W // 2, int(OUT_H * 0.742)), size=22 if not QUICK_MODE else 11, fill=COLORS["rose"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "brighter cells are a depth proxy—not a limiting-magnitude map", (OUT_W // 2, int(OUT_H * 0.783)), size=16 if not QUICK_MODE else 8, fill=COLORS["white"] + (220,), anchor="ma", stroke=1)

    def draw_fields(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "fields")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_sky_grid(image, alpha=30, label=False)
        self.draw_points(image, self.order_all, reveal=1.0, mode="exposure", alpha=105, size_boost=0.85)
        self.draw_depth_cells(image, local)
        self.draw_field_markers(image, reveal=min(1.0, local * 1.3))
        self.panel(image, (int(OUT_W * 0.08), int(OUT_H * 0.705), int(OUT_W * 0.92), int(OUT_H * 0.835)), alpha=168)
        draw_text(image, "THE DEEP FIELDS ARE TINY PINPRICKS", (OUT_W // 2, int(OUT_H * 0.742)), size=22 if not QUICK_MODE else 11, fill=COLORS["gold"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "HDF • GOODS • HUDF/XDF • COSMOS • six Frontier Fields", (OUT_W // 2, int(OUT_H * 0.786)), size=16 if not QUICK_MODE else 8, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)

    def draw_timeline(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "timeline")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_sky_grid(image, alpha=34, label=False)
        visible_count = int(np.count_nonzero(self.time_norm[self.order_time] <= local))
        self.draw_points(image, self.order_time, reveal=visible_count / max(len(self.order_time), 1), mode="instrument", alpha=215)
        self.draw_field_markers(image, reveal=1.0)
        start_mjd, end_mjd = self.summary.get("date_start_mjd"), self.summary.get("date_end_mjd")
        date_label = format_date_from_mjd(lerp(float(start_mjd), float(end_mjd), local)) if start_mjd is not None and end_mjd is not None else "OBSERVATION DATE UNKNOWN"
        x0, _, x1, _ = self.map_box
        bar_y = int(OUT_H * 0.715)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((x0, bar_y, x1, bar_y + (18 if not QUICK_MODE else 9)), radius=9 if not QUICK_MODE else 4, fill=(12, 32, 58, 210), outline=COLORS["cyan"] + (65,))
        draw.rounded_rectangle((x0, bar_y, int(lerp(x0, x1, local)), bar_y + (18 if not QUICK_MODE else 9)), radius=9 if not QUICK_MODE else 4, fill=COLORS["cyan"] + (205,))
        draw_text(image, date_label, (OUT_W // 2, int(OUT_H * 0.755)), size=31 if not QUICK_MODE else 15, fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, f"archive records revealed // {visible_count:,} of {len(self.order_time):,} rendered", (OUT_W // 2, int(OUT_H * 0.795)), size=16 if not QUICK_MODE else 8, fill=COLORS["muted"] + (215,), anchor="ma", stroke=1)

    def draw_finale(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "finale")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_sky_grid(image, alpha=28, label=False)
        self.draw_points(image, self.order_all, reveal=1.0, mode="instrument", alpha=200, size_boost=0.95)
        self.draw_depth_cells(image, local)
        self.draw_field_markers(image, reveal=1.0)
        self.panel(image, (int(OUT_W * 0.07), int(OUT_H * 0.60), int(OUT_W * 0.93), int(OUT_H * 0.83)), alpha=187)
        draw_text(image, "HUBBLE'S DEEPEST OBSERVATIONS", (OUT_W // 2, int(OUT_H * 0.642)), size=29 if not QUICK_MODE else 14, fill=COLORS["white"] + (248,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "ON ONE SKY MAP", (OUT_W // 2, int(OUT_H * 0.684)), size=29 if not QUICK_MODE else 14, fill=COLORS["white"] + (248,), bold=True, anchor="ma", stroke=1)
        draw_text(image, f"{self.summary['observation_records']:,} records  •  {self.summary['total_exposure_hours_metadata_sum']:,.0f} exposure-hours", (OUT_W // 2, int(OUT_H * 0.735)), size=18 if not QUICK_MODE else 9, fill=COLORS["gold"] + (238,), bold=True, anchor="ma", stroke=1)
        draw_text(image, f"{self.summary['date_start']}  →  {self.summary['date_end']}", (OUT_W // 2, int(OUT_H * 0.774)), size=17 if not QUICK_MODE else 8, fill=COLORS["cyan"] + (230,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "depth proxy = summed archive exposure time per sky cell", (OUT_W // 2, int(OUT_H * 0.807)), size=15 if not QUICK_MODE else 7, fill=COLORS["white"] + (215,), anchor="ma", stroke=1)

    def draw_source_hud(self, image: Image.Image):
        live = bool(self.summary["is_live_observational_data"])
        label = "SOURCE // MAST PUBLIC HST IMAGES" if live else "PREVIEW SOURCE // SYNTHETIC FIXTURE"
        colour = COLORS["cyan"] if live else COLORS["gold"]
        draw_text(image, label, (OUT_W - (46 if not QUICK_MODE else 23), 70 if not QUICK_MODE else 35), size=16 if not QUICK_MODE else 8, fill=colour + (235,), bold=True, anchor="ra", stroke=1)
        draw_text(image, f"RECORDS // {self.summary['observation_records']:,}", (OUT_W - (46 if not QUICK_MODE else 23), 100 if not QUICK_MODE else 50), size=15 if not QUICK_MODE else 7, fill=COLORS["muted"] + (205,), anchor="ra", stroke=1)
        draw_text(image, f"EXPOSURE // {self.summary['total_exposure_hours_metadata_sum']:,.0f} h", (OUT_W - (46 if not QUICK_MODE else 23), 127 if not QUICK_MODE else 63), size=15 if not QUICK_MODE else 7, fill=COLORS["muted"] + (195,), anchor="ra", stroke=1)

    def draw_titles(self, image: Image.Image, t: float, shot_name: str):
        intro_end = 6.7 if not QUICK_MODE else 1.4
        alpha = int(255 * smoothstep((t - 0.2) / 0.8) * (1.0 - smoothstep((t - intro_end) / 0.65)))
        if alpha > 4:
            draw_text(image, "HUBBLE'S DEEPEST", (54 if not QUICK_MODE else 27, 88 if not QUICK_MODE else 43), size=39 if not QUICK_MODE else 19, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(image, "OBSERVATIONS", (54 if not QUICK_MODE else 27, 136 if not QUICK_MODE else 67), size=39 if not QUICK_MODE else 19, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(image, "ON ONE SKY MAP", (54 if not QUICK_MODE else 27, 184 if not QUICK_MODE else 91), size=39 if not QUICK_MODE else 19, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(image, CONFIG["subtitle"], (56 if not QUICK_MODE else 28, 236 if not QUICK_MODE else 118), size=20 if not QUICK_MODE else 10, fill=COLORS["cyan"] + (min(alpha, 230),), bold=True)
        labels = {
            "intro": "HOW HUBBLE BUILDS A DEEP VIEW",
            "all_sky": "PUBLIC HST IMAGE RECORDS ACROSS THE SKY",
            "depth": "CUMULATIVE EXPOSURE TIME REVEALS THE HOTSPOTS",
            "fields": "THE CANONICAL DEEP FIELDS",
            "timeline": "THREE DECADES OF OBSERVATIONS ACCUMULATE",
            "finale": "A DEPTH-WEIGHTED MAP OF HUBBLE'S ARCHIVE",
        }
        if t > (5.1 if not QUICK_MODE else 1.2):
            draw_text(image, labels[shot_name], (54 if not QUICK_MODE else 27, 60 if not QUICK_MODE else 30), size=18 if not QUICK_MODE else 9, fill=COLORS["muted"] + (205,), bold=True, stroke=1)

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        y0 = OUT_H - (244 if not QUICK_MODE else 124)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((44 if not QUICK_MODE else 22, y0, OUT_W - (44 if not QUICK_MODE else 22), y0 + (124 if not QUICK_MODE else 66)), radius=24 if not QUICK_MODE else 12, fill=(2, 6, 15, 176), outline=(80, 190, 228, 66), width=1)
        image.alpha_composite(panel)
        draw_wrapped_text(image, text, (68 if not QUICK_MODE else 34, y0 + (28 if not QUICK_MODE else 14)), OUT_W - (136 if not QUICK_MODE else 68), size=29 if not QUICK_MODE else 14, fill=COLORS["white"] + (245,))

    def draw_hud_noise(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for item in self.hud:
            pulse = 0.5 + 0.5 * math.sin(t * 1.9 + item["phase"])
            if pulse < 0.74:
                continue
            y = (item["y"] + t * 9.0) % OUT_H
            draw.line((item["x"], y, item["x"] + item["length"], y), fill=COLORS["cyan"] + (int(item["a"] * pulse),), width=1)
        offset = int((t * 39) % 7)
        for y in range(offset, OUT_H, 7):
            draw.line((0, y, OUT_W, y), fill=(120, 200, 240, 10), width=1)
        scan_y = int((t * 164) % (OUT_H + 220)) - 110
        draw.rectangle((0, scan_y, OUT_W, scan_y + (48 if not QUICK_MODE else 24)), fill=(80, 210, 240, 8))
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        name = shot["name"]
        image = self.background(t)
        if name == "intro":
            self.draw_intro(image, t)
        elif name == "all_sky":
            self.draw_all_sky(image, t)
        elif name == "depth":
            self.draw_depth(image, t)
        elif name == "fields":
            self.draw_fields(image, t)
        elif name == "timeline":
            self.draw_timeline(image, t)
        elif name == "finale":
            self.draw_finale(image, t)
        self.draw_source_hud(image)
        self.draw_titles(image, t, name)
        self.draw_caption(image, t)
        self.draw_hud_noise(image, t)
        array = apply_grade(np.asarray(image.convert("RGB")))
        array = np.clip(array.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / 0.9)
        fade_out = 1.0 - smoothstep((t - (float(CONFIG["duration_s"]) - 1.1)) / 1.0)
        return np.clip(array.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def render_video(scene: HubbleDeepSkyScene) -> Path:
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    print("Subtitle sidecar:", srt_path.resolve())
    raw_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    final_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    frame_count = int(round(float(CONFIG["duration_s"]) * int(CONFIG["fps"])))
    times = np.arange(frame_count) / int(CONFIG["fps"])
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(raw_video, fps=int(CONFIG["fps"]), codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None) as writer:
        for t in tqdm(times, desc="Rendering Hubble deep-map short"):
            writer.append_data(scene.render_frame(float(t)))
    shutil.copyfile(raw_video, final_video)
    print("Final video:", final_video.resolve())
    return final_video


def main():
    print("Loading public Hubble image observation metadata ...")
    frame, source, notes, requests_used = load_all_data()
    print("Transforming coordinates and constructing exposure-time depth cells ...")
    frame, cells, fields, summary = prepare_observations(frame)
    catalogue_path, summary_path = save_data_products(frame, cells, fields, summary, notes, requests_used)
    create_scientific_plots(frame, cells, fields, summary)

    print("Data source:", source)
    print("Observation records:", f"{summary['observation_records']:,}")
    print("Unique target names:", f"{summary['unique_targets']:,}")
    print("Unique programmes:", f"{summary['unique_programmes']:,}")
    print("Exposure-time metadata sum:", f"{summary['total_exposure_hours_metadata_sum']:,.1f} hours")
    print("Observation span:", summary["date_start"], "to", summary["date_end"])
    for instrument, count in summary["instrument_counts"].items():
        print(f"Instrument {instrument}: {count:,}")
    for note in notes:
        print("Data note:", note)
    print("Data:", catalogue_path.resolve())
    print("Summary:", summary_path.resolve())

    scene = HubbleDeepSkyScene(frame, cells, fields, summary)
    preview_times = [
        1.0,
        min(10.0, float(CONFIG["duration_s"]) * 0.20),
        min(22.0, float(CONFIG["duration_s"]) * 0.39),
        min(34.0, float(CONFIG["duration_s"]) * 0.60),
        min(45.0, float(CONFIG["duration_s"]) * 0.79),
        float(CONFIG["duration_s"]) - 1.0,
    ]
    for preview_time in tqdm(preview_times, desc="Preview frames"):
        Image.fromarray(scene.render_frame(float(preview_time))).save(PREVIEW_DIR / f"preview_{int(preview_time):02d}s.png")
    render_video(scene)
    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-", path.name)


if __name__ == "__main__":
    main()
