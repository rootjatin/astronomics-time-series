from __future__ import annotations

"""
Where the James Webb Telescope Looked — cinematic YouTube Short renderer

Creates a vertical 1080x1920 astronomy short from public James Webb Space
Telescope observation metadata in the Mikulski Archive for Space Telescopes
(MAST). The animation turns archive coordinates into an all-sky map, reveals
where Webb returned repeatedly, plays the observations in time, and colours the
pointings by science instrument.

Preferred live source
---------------------
MAST observation service:
    https://mast.stsci.edu/api/v0/invoke

The renderer queries Mast.Caom.Filtered for public JWST science observations
and uses archive-level metadata including:

- observation identifier
- target coordinates (ICRS right ascension and declination)
- observation start time (MJD)
- science instrument
- target and programme metadata
- exposure time when available

The sky coordinates are transformed to Galactic longitude and latitude and
shown with a Hammer all-sky projection. Each rendered dot represents one
public MAST observation record after deduplication by archive observation ID.
Repeated dots and bright cells indicate repeated archive records near the same
part of the sky; they are not a direct measurement of equal-area survey depth.

Science notes
-------------
- JWST is a pointed observatory, not an all-sky survey telescope.
- The map is intentionally uneven because observing programmes select targets.
- Dense regions can represent deep fields, mosaics, monitoring, spectroscopy,
  calibration-adjacent science records, or repeated visits.
- Point size is cinematic, not the true instrument field of view.
- Archive content grows and can be reprocessed, so a new run may return a
  different total than an earlier render.
- This is an archive visualisation, not an exposure-map or completeness study.

Offline behaviour
-----------------
If MAST is unreachable, the script uses a clearly labelled deterministic
fixture that mimics clustered pointings, repeated visits, multiple instruments,
and a mission timeline. The fixture is for preview/layout validation only and
is not observational data.

Recommended install
-------------------
    pip install numpy pandas matplotlib pillow imageio imageio-ffmpeg tqdm

Quick preview render
--------------------
    JWST_LOOKED_SHORT_QUICK=1 python where_the_james_webb_telescope_looked_short.py

Force offline fixture mode
--------------------------
    JWST_LOOKED_SHORT_OFFLINE=1 python where_the_james_webb_telescope_looked_short.py

Use a previously downloaded CSV
-------------------------------
    JWST_LOOKED_DATA_PATH=/path/to/jwst_mast_observations.csv \
        python where_the_james_webb_telescope_looked_short.py

Primary references
------------------
- MAST API tutorial:
  https://mast.stsci.edu/api/v0/MastApiTutorial.html
- MAST API guidance for JWST:
  https://outerspace.stsci.edu/spaces/MASTDOCS/pages/113771434/Using+MAST+APIs
- JWST archive manual:
  https://outerspace.stsci.edu/spaces/MASTDOCS/pages/113771318/JWST+Archive+Manual
"""

import io
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

QUICK_MODE = os.environ.get("JWST_LOOKED_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("JWST_LOOKED_SHORT_OFFLINE", "0") == "1"
LOCAL_DATA_PATH = os.environ.get("JWST_LOOKED_DATA_PATH", "").strip()

OUTPUT_ROOT = Path("where_james_webb_telescope_looked_short_output")
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
    "output_basename": "where_the_james_webb_telescope_looked",
    "title": "WHERE THE JAMES WEBB TELESCOPE LOOKED",
    "subtitle": "public JWST observations // MAST archive // real sky coordinates",
    "data_timeout_s": 45,
    "mast_page_size": 1500 if QUICK_MODE else 5000,
    "mast_max_pages": 4 if QUICK_MODE else 30,
    "max_archive_rows": 6000 if QUICK_MODE else 50000,
    "max_render_points": 2200 if QUICK_MODE else 9000,
    "fixture_rows": 3200 if QUICK_MODE else 15000,
    "background_stars": 280 if QUICK_MODE else 520,
    "hud_noise": 32 if QUICK_MODE else 66,
    "contrast": 1.08,
    "saturation": 1.06,
    "vignette": 0.25,
    "density_cell_lon_deg": 6.0,
    "density_cell_lat_deg": 6.0,
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
    "NIRCAM": COLORS["cyan"],
    "NIRSPEC": COLORS["violet"],
    "MIRI": COLORS["rose"],
    "NIRISS": COLORS["gold"],
    "FGS": COLORS["green"],
    "OTHER": COLORS["ice"],
}

FULL_CAPTIONS = [
    (0.5, 7.2, "James Webb does not scan the whole sky. It turns toward carefully selected targets, one programme at a time."),
    (7.3, 17.2, "Every dot here is a public JWST science observation record from MAST, placed at its real sky coordinates."),
    (17.3, 27.2, "The pattern is deliberately uneven. Bright knots are places Webb revisited for deep fields, mosaics, spectra, or monitoring."),
    (27.3, 38.8, "Play the archive in time and Webb's footprint grows: target after target, visit after visit, across the infrared sky."),
    (38.9, 49.6, "Colour separates the instruments: NIRCam, NIRSpec, MIRI, NIRISS, and the fine-guidance system's science records."),
    (49.7, 57.4, "This is where the James Webb Space Telescope looked: not a finished atlas, but a living map that keeps growing."),
]
if QUICK_MODE:
    _caption_scale = float(CONFIG["duration_s"]) / 58.0
    CAPTIONS = [(a * _caption_scale, b * _caption_scale, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.8 if not QUICK_MODE else 1.65},
    {"name": "all_sky", "start": 7.8 if not QUICK_MODE else 1.65, "end": 18.5 if not QUICK_MODE else 3.85},
    {"name": "density", "start": 18.5 if not QUICK_MODE else 3.85, "end": 29.0 if not QUICK_MODE else 6.05},
    {"name": "timeline", "start": 29.0 if not QUICK_MODE else 6.05, "end": 40.5 if not QUICK_MODE else 8.4},
    {"name": "instruments", "start": 40.5 if not QUICK_MODE else 8.4, "end": 50.5 if not QUICK_MODE else 10.45},
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
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-999"}:
        return default
    return text


def mjd_to_datetime(value: float) -> Optional[datetime]:
    try:
        if not np.isfinite(value):
            return None
        return MJD_EPOCH + timedelta(days=float(value))
    except Exception:
        return None


def datetime_to_mjd(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (value - MJD_EPOCH).total_seconds() / 86400.0


def format_date_from_mjd(value: float) -> str:
    dt = mjd_to_datetime(value)
    return dt.strftime("%d %b %Y").upper() if dt else "DATE UNKNOWN"


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Coordinate transforms and projection
# -----------------------------------------------------------------------------

# IAU J2000 rotation from equatorial Cartesian coordinates to Galactic.
EQ_TO_GAL = np.array(
    [
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [0.4941094279, -0.4448296300, 0.7469822445],
        [-0.8676661490, -0.1980763734, 0.4559837762],
    ],
    dtype=float,
)
GAL_TO_EQ = EQ_TO_GAL.T


def equatorial_to_galactic(ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    xyz_eq = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    xyz_gal = xyz_eq @ EQ_TO_GAL.T
    lon = np.mod(np.rad2deg(np.arctan2(xyz_gal[:, 1], xyz_gal[:, 0])), 360.0)
    lat = np.rad2deg(np.arcsin(np.clip(xyz_gal[:, 2], -1.0, 1.0)))
    return lon, lat


def galactic_to_equatorial(lon_deg: np.ndarray, lat_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    xyz_gal = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    xyz_eq = xyz_gal @ GAL_TO_EQ.T
    ra = np.mod(np.rad2deg(np.arctan2(xyz_eq[:, 1], xyz_eq[:, 0])), 360.0)
    dec = np.rad2deg(np.arcsin(np.clip(xyz_eq[:, 2], -1.0, 1.0)))
    return ra, dec


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
            "User-Agent": "jwst-looked-short-renderer/1.0",
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
        {"paramName": "obs_collection", "values": ["JWST"]},
        {"paramName": "intentType", "values": ["science"]},
        {"paramName": "dataRights", "values": ["PUBLIC"]},
    ]


def fetch_mast_observations() -> Tuple[pd.DataFrame, List[str], List[Dict[str, Any]]]:
    notes: List[str] = []
    requests_used: List[Dict[str, Any]] = []
    page_size = int(CONFIG["mast_page_size"])
    max_pages = int(CONFIG["mast_max_pages"])
    max_rows = int(CONFIG["max_archive_rows"])
    pieces: List[pd.DataFrame] = []

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
        raise RuntimeError("MAST returned no public JWST science observation rows")
    combined = pd.concat(pieces, ignore_index=True)
    if len(combined) > max_rows:
        combined = combined.iloc[:max_rows].copy()
    return combined, notes, requests_used


def normalize_instrument(value: Any) -> str:
    text = safe_text(value).upper().replace("-", "").replace("_", "")
    if text.startswith("NIRCAM"):
        return "NIRCAM"
    if text.startswith("NIRSPEC"):
        return "NIRSPEC"
    if text.startswith("MIRI"):
        return "MIRI"
    if text.startswith("NIRISS"):
        return "NIRISS"
    if text.startswith("FGS"):
        return "FGS"
    return "OTHER"


def normalize_observations(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    frame = raw.copy()
    frame = frame.rename(columns={column: column.strip() for column in frame.columns})
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

    result = pd.DataFrame(
        {
            "s_ra": pd.to_numeric(frame[ra_col], errors="coerce"),
            "s_dec": pd.to_numeric(frame[dec_col], errors="coerce"),
        }
    )

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
        if source_column is not None:
            result[canonical] = frame[source_column]

    for numeric in ("t_min", "t_max", "t_exptime", "calib_level"):
        if numeric not in result.columns:
            result[numeric] = np.nan
        result[numeric] = pd.to_numeric(result[numeric], errors="coerce")

    for text_col in (
        "obsid",
        "obs_id",
        "instrument_name",
        "filters",
        "target_name",
        "proposal_id",
        "proposal_pi",
        "obs_title",
        "dataproduct_type",
        "dataRights",
        "intentType",
        "s_region",
    ):
        if text_col not in result.columns:
            result[text_col] = ""

    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["s_ra", "s_dec"])
    result = result[result["s_ra"].between(0.0, 360.0) & result["s_dec"].between(-90.0, 90.0)].copy()

    if source == "mast_public_jwst_observations":
        rights = result["dataRights"].astype(str).str.upper()
        intent = result["intentType"].astype(str).str.lower()
        result = result[(rights.eq("PUBLIC") | rights.eq("")) & (intent.eq("science") | intent.eq(""))].copy()

    obsid = result["obsid"].astype(str).str.strip()
    obs_id = result["obs_id"].astype(str).str.strip()
    fallback_key = (
        result["s_ra"].round(7).astype(str)
        + ":"
        + result["s_dec"].round(7).astype(str)
        + ":"
        + result["t_min"].round(6).astype(str)
        + ":"
        + result["instrument_name"].astype(str)
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
    rng = np.random.default_rng(25122021)
    count = int(CONFIG["fixture_rows"])

    centers = np.array(
        [
            [96.0, -60.0],
            [224.0, -54.0],
            [126.0, 54.0],
            [28.0, -31.0],
            [278.0, 32.0],
            [5.0, 25.0],
            [45.0, 2.0],
            [332.0, -18.0],
            [188.0, 42.0],
            [250.0, 4.0],
            [305.0, -43.0],
            [150.0, -7.0],
        ],
        dtype=float,
    )
    weights = np.array([0.14, 0.13, 0.11, 0.10, 0.09, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04])
    weights /= weights.sum()
    assignments = rng.choice(len(centers), size=int(count * 0.82), p=weights)
    widths = rng.uniform(0.25, 2.4, len(centers))
    lon_cluster = centers[assignments, 0] + rng.normal(0.0, widths[assignments])
    lat_cluster = centers[assignments, 1] + rng.normal(0.0, widths[assignments] * 0.72)

    field_count = count - len(lon_cluster)
    lon_field = rng.uniform(0.0, 360.0, field_count)
    sin_lat = rng.uniform(-1.0, 1.0, field_count)
    lat_field = np.rad2deg(np.arcsin(sin_lat))
    keep = rng.random(field_count) > 0.40 * np.exp(-(lat_field / 12.0) ** 2)
    lon_field = lon_field[keep]
    lat_field = lat_field[keep]
    while len(lon_field) + len(lon_cluster) < count:
        extra = count - len(lon_field) - len(lon_cluster)
        lon_field = np.concatenate([lon_field, rng.uniform(0.0, 360.0, extra)])
        lat_field = np.concatenate([lat_field, np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, extra)))])

    gal_lon = np.mod(np.concatenate([lon_cluster, lon_field[: count - len(lon_cluster)]]), 360.0)
    gal_lat = np.clip(np.concatenate([lat_cluster, lat_field[: count - len(lat_cluster)]]), -89.8, 89.8)
    ra, dec = galactic_to_equatorial(gal_lon, gal_lat)

    instruments = rng.choice(
        ["NIRCAM/IMAGE", "NIRSPEC/MOS", "MIRI/IMAGE", "NIRISS/SOSS", "FGS/IMAGE"],
        size=count,
        p=[0.47, 0.25, 0.16, 0.08, 0.04],
    )
    start_mjd = datetime_to_mjd(datetime(2022, 7, 12, tzinfo=timezone.utc))
    end_mjd = datetime_to_mjd(datetime(2026, 6, 30, tzinfo=timezone.utc))
    base_time = np.sort(rng.uniform(start_mjd, end_mjd, count))
    programme_bursts = rng.normal(0.0, 8.0, count)
    t_min = np.clip(base_time + programme_bursts, start_mjd, end_mjd)
    exposure = np.clip(rng.lognormal(mean=7.2, sigma=1.15, size=count), 20.0, 160000.0)

    target_labels = [
        "DEEP FIELD ALPHA",
        "DEEP FIELD BETA",
        "GALAXY MOSAIC",
        "EXOPLANET HOST",
        "STAR FORMING REGION",
        "DISTANT GALAXY",
        "SOLAR SYSTEM TARGET",
        "TRANSIENT FIELD",
        "CLUSTER LENS",
        "SPECTROSCOPY FIELD",
        "PROTOSTAR",
        "BROWN DWARF",
    ]
    target_name = np.array([target_labels[index] for index in assignments], dtype=object)
    if len(target_name) < count:
        target_name = np.concatenate(
            [target_name, rng.choice(target_labels, size=count - len(target_name), replace=True)]
        )

    raw = pd.DataFrame(
        {
            "obsid": np.arange(90_000_000, 90_000_000 + count, dtype=np.int64),
            "obs_id": [f"fixture-jwst-{index:05d}" for index in range(count)],
            "s_ra": ra,
            "s_dec": dec,
            "instrument_name": instruments,
            "filters": rng.choice(["F200W", "F444W", "PRISM", "F770W", "CLEAR"], size=count),
            "target_name": target_name[:count],
            "t_min": t_min,
            "t_max": t_min + exposure / 86400.0,
            "t_exptime": exposure,
            "proposal_id": rng.integers(1000, 7000, size=count),
            "proposal_pi": "SYNTHETIC FIXTURE",
            "obs_title": "Synthetic preview observation",
            "dataproduct_type": rng.choice(["image", "spectrum", "cube"], size=count, p=[0.65, 0.27, 0.08]),
            "calib_level": rng.choice([2, 3], size=count, p=[0.38, 0.62]),
            "dataRights": "PUBLIC",
            "intentType": "science",
            "s_region": "",
        }
    )
    return normalize_observations(raw, "offline_jwst_pointing_fixture"), "offline_jwst_pointing_fixture"


def load_all_data() -> Tuple[pd.DataFrame, str, List[str], List[Dict[str, Any]]]:
    notes: List[str] = []
    requests_used: List[Dict[str, Any]] = []

    if LOCAL_DATA_PATH:
        try:
            frame = load_local_observations(Path(LOCAL_DATA_PATH).expanduser())
            notes.append(f"Loaded JWST_LOOKED_DATA_PATH={LOCAL_DATA_PATH}")
            return frame, "local_mast_compatible_csv", notes, requests_used
        except Exception as exc:
            notes.append(f"Local catalogue failed: {exc}")

    if OFFLINE_MODE:
        notes.append("Offline mode requested with JWST_LOOKED_SHORT_OFFLINE=1")
        frame, source = fallback_observations()
        return frame, source, notes, requests_used

    try:
        raw, live_notes, requests_used = fetch_mast_observations()
        notes.extend(live_notes)
        frame = normalize_observations(raw, "mast_public_jwst_observations")
        if len(frame) < 100:
            raise RuntimeError(f"Only {len(frame)} valid public JWST observation records returned")
        return frame, "mast_public_jwst_observations", notes, requests_used
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
    hammer_x, hammer_y = hammer_project(gal_lon, gal_lat)
    out["hammer_x"] = hammer_x
    out["hammer_y"] = hammer_y

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

    lon_cell = float(CONFIG["density_cell_lon_deg"])
    lat_cell = float(CONFIG["density_cell_lat_deg"])
    out["density_lon_bin"] = np.floor(out["galactic_lon_deg"] / lon_cell).astype(int)
    out["density_lat_bin"] = np.floor((out["galactic_lat_deg"] + 90.0) / lat_cell).astype(int)
    density = (
        out.groupby(["density_lon_bin", "density_lat_bin"], as_index=False)
        .agg(
            observation_records=("obs_id", "size"),
            median_mjd=("t_min", "median"),
            total_exposure_s=("t_exptime", "sum"),
        )
        .sort_values("observation_records", ascending=False)
        .reset_index(drop=True)
    )
    density["galactic_lon_center_deg"] = (density["density_lon_bin"] + 0.5) * lon_cell
    density["galactic_lat_center_deg"] = (density["density_lat_bin"] + 0.5) * lat_cell - 90.0
    dx, dy = hammer_project(density["galactic_lon_center_deg"], density["galactic_lat_center_deg"])
    density["hammer_x"] = dx
    density["hammer_y"] = dy

    out = out.merge(
        density[["density_lon_bin", "density_lat_bin", "observation_records"]].rename(
            columns={"observation_records": "cell_record_count"}
        ),
        on=["density_lon_bin", "density_lat_bin"],
        how="left",
    )
    out["density_norm"] = np.log1p(out["cell_record_count"]) / max(
        float(np.log1p(out["cell_record_count"].max())), 1e-9
    )
    out = out.sort_values(["t_min", "instrument_group", "obs_id"], na_position="last").reset_index(drop=True)

    target_counts = (
        out.groupby("target_name", as_index=False)
        .agg(
            observation_records=("obs_id", "size"),
            galactic_lon_deg=("galactic_lon_deg", "median"),
            galactic_lat_deg=("galactic_lat_deg", "median"),
            first_mjd=("t_min", "min"),
            last_mjd=("t_min", "max"),
        )
        .sort_values("observation_records", ascending=False)
        .reset_index(drop=True)
    )

    instrument_counts = out["instrument_group"].value_counts().to_dict()
    valid_dates = out.loc[np.isfinite(out["t_min"]), "t_min"]
    total_exposure_s = float(np.nansum(np.clip(out["t_exptime"].to_numpy(float), 0.0, None)))
    repeated_records = int(out.loc[out["cell_record_count"] > 1].shape[0])
    summary = {
        "source": str(out["data_source"].iloc[0]),
        "is_live_observational_data": str(out["data_source"].iloc[0]) in {
            "mast_public_jwst_observations",
            "local_mast_compatible_csv",
        },
        "observation_records": int(len(out)),
        "unique_targets": int(out["target_name"].nunique()),
        "unique_programmes": int(out["proposal_id"].nunique()),
        "occupied_sky_cells": int(len(density)),
        "repeated_record_fraction": float(repeated_records / max(len(out), 1)),
        "total_exposure_hours_metadata_sum": total_exposure_s / 3600.0,
        "date_start_mjd": float(valid_dates.min()) if len(valid_dates) else None,
        "date_end_mjd": float(valid_dates.max()) if len(valid_dates) else None,
        "date_start": format_date_from_mjd(float(valid_dates.min())) if len(valid_dates) else "UNKNOWN",
        "date_end": format_date_from_mjd(float(valid_dates.max())) if len(valid_dates) else "UNKNOWN",
        "instrument_counts": {str(key): int(value) for key, value in instrument_counts.items()},
        "top_targets": target_counts.head(12).to_dict(orient="records"),
        "projection": "Hammer projection in Galactic longitude and latitude",
        "record_definition": "One deduplicated public MAST observation row, keyed primarily by obsid",
        "warning": "Point sizes are cinematic and do not represent true instrument footprints or uniform survey depth.",
    }
    return out, density, target_counts, summary


def save_data_products(
    frame: pd.DataFrame,
    density: pd.DataFrame,
    targets: pd.DataFrame,
    summary: Dict[str, Any],
    notes: List[str],
    requests_used: List[Dict[str, Any]],
) -> Tuple[Path, Path]:
    catalogue_path = DATA_ROOT / "jwst_public_observation_map.csv"
    density_path = DATA_ROOT / "jwst_sky_density_cells.csv"
    targets_path = DATA_ROOT / "jwst_target_record_counts.csv"
    summary_path = DATA_ROOT / "jwst_where_it_looked_summary.json"
    frame.to_csv(catalogue_path, index=False)
    density.to_csv(density_path, index=False)
    targets.to_csv(targets_path, index=False)
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
                "fallback_warning": "offline_jwst_pointing_fixture is deterministic synthetic preview data, not observational data",
                "science_warning": "Archive record density is not the same as exposure depth, sky coverage, or scientific completeness.",
                "source_urls": {
                    "mast_api_tutorial": "https://mast.stsci.edu/api/v0/MastApiTutorial.html",
                    "mast_jwst_api_guidance": "https://outerspace.stsci.edu/spaces/MASTDOCS/pages/113771434/Using+MAST+APIs",
                    "jwst_archive_manual": "https://outerspace.stsci.edu/spaces/MASTDOCS/pages/113771318/JWST+Archive+Manual",
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return catalogue_path, summary_path


def create_scientific_plots(frame: pd.DataFrame, density: pd.DataFrame, summary: Dict[str, Any]):
    sample = evenly_subsample(frame, 18000)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.scatter(sample["hammer_x"], sample["hammer_y"], s=2, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Public JWST science observation records in Galactic coordinates")
    ax.set_xlabel("Hammer x")
    ax.set_ylabel("Hammer y")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "jwst_all_sky_pointings.png", dpi=170)
    plt.close(fig)

    dated = frame[np.isfinite(frame["t_min"])].sort_values("t_min").copy()
    if len(dated):
        fig, ax = plt.subplots(figsize=(10, 5.2))
        dates = [mjd_to_datetime(value) for value in dated["t_min"].to_numpy(float)]
        ax.plot(dates, np.arange(1, len(dated) + 1))
        ax.set_title("Cumulative public JWST observation records")
        ax.set_xlabel("Observation date")
        ax.set_ylabel("Cumulative records")
        plt.tight_layout()
        plt.savefig(PREVIEW_DIR / "jwst_observation_timeline.png", dpi=170)
        plt.close(fig)

    order = [name for name in INSTRUMENT_COLORS if name in summary["instrument_counts"]]
    counts = [summary["instrument_counts"][name] for name in order]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(order, counts)
    ax.set_title("JWST observation records by instrument")
    ax.set_xlabel("Instrument")
    ax.set_ylabel("Records")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "jwst_instrument_counts.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.hist(density["observation_records"], bins=40)
    ax.set_yscale("log")
    ax.set_title("Observation-record density per 6° × 6° Galactic cell")
    ax.set_xlabel("Records in cell")
    ax.set_ylabel("Cells")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "jwst_density_cell_histogram.png", dpi=170)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class JWSTLookedScene:
    def __init__(self, frame: pd.DataFrame, density: pd.DataFrame, targets: pd.DataFrame, summary: Dict[str, Any]):
        render = frame.copy()
        if len(render) > int(CONFIG["max_render_points"]):
            # Preserve the timeline and instrument mix better than a simple head().
            sampled_parts: List[pd.DataFrame] = []
            render_limit = int(CONFIG["max_render_points"])
            for _, part in render.groupby("instrument_group", sort=False):
                allocation = max(40, int(round(render_limit * len(part) / len(frame))))
                sampled_parts.append(
                    evenly_subsample(part.sort_values("t_min", na_position="last"), allocation)
                )
            render = pd.concat(sampled_parts, ignore_index=True)
            render = evenly_subsample(
                render.sort_values("t_min", na_position="last"),
                render_limit,
            )
        self.frame = render.reset_index(drop=True)
        self.density = density.head(80 if not QUICK_MODE else 32).copy().reset_index(drop=True)
        self.targets = targets.head(10).copy().reset_index(drop=True)
        self.summary = summary
        self.stars = self._make_stars(int(CONFIG["background_stars"]), seed=1225)
        self.hud = self._make_hud(int(CONFIG["hud_noise"]), seed=2021)
        self.map_box = (
            int(OUT_W * 0.065),
            int(OUT_H * 0.22),
            int(OUT_W * 0.935),
            int(OUT_H * 0.68),
        )
        self.point_xy = self._to_screen(self.frame["hammer_x"].to_numpy(float), self.frame["hammer_y"].to_numpy(float))
        self.density_xy = self._to_screen(self.density["hammer_x"].to_numpy(float), self.density["hammer_y"].to_numpy(float))
        self.instrument = self.frame["instrument_group"].astype(str).to_numpy()
        self.time_norm = self.frame["time_norm"].to_numpy(float)
        self.density_norm = self.frame["density_norm"].to_numpy(float)
        self.order_all = np.arange(len(self.frame))
        self.order_density = np.argsort(self.density_norm)
        self.order_time = np.argsort(self.time_norm)
        self.max_cell_records = max(int(self.density["observation_records"].max()) if len(self.density) else 1, 1)

    @staticmethod
    def _make_stars(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.3, 2.0)),
                "a": float(rng.uniform(18, 105)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            }
            for _ in range(count)
        ]

    @staticmethod
    def _make_hud(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "length": float(rng.uniform(10, 95)),
                "a": float(rng.uniform(8, 42)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            }
            for _ in range(count)
        ]

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
            draw.ellipse(
                (star["x"] - r, star["y"] - r, star["x"] + r, star["y"] + r),
                fill=(220, 235, 255, alpha),
            )
        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        clouds = [
            (OUT_W * 0.18, OUT_H * 0.28, (30, 48, 145)),
            (OUT_W * 0.82, OUT_H * 0.40, (76, 24, 126)),
            (OUT_W * 0.52, OUT_H * 0.76, (8, 82, 124)),
        ]
        for cx, cy, color in clouds:
            for radius, alpha in [(430 * OUT_W / 1080, 14), (280 * OUT_W / 1080, 22), (165 * OUT_W / 1080, 30)]:
                hd.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color + (alpha,))
        haze = haze.filter(ImageFilter.GaussianBlur(62 if not QUICK_MODE else 31))
        image.alpha_composite(haze)
        return image

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 170):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            box,
            radius=24 if not QUICK_MODE else 12,
            fill=(2, 7, 18, alpha),
            outline=(100, 200, 235, 64),
            width=1,
        )
        image.alpha_composite(overlay)

    def draw_mirror(self, image: Image.Image, center: Tuple[float, float], scale: float, phase: float, alpha: int = 220):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cx, cy = center
        positions = [
            (0, 0),
            (-1, 0), (1, 0),
            (-0.5, -0.87), (0.5, -0.87),
            (-0.5, 0.87), (0.5, 0.87),
            (-2, 0), (2, 0),
            (-1.5, -0.87), (1.5, -0.87),
            (-1.5, 0.87), (1.5, 0.87),
            (-1, -1.74), (0, -1.74), (1, -1.74),
            (-1, 1.74), (0, 1.74), (1, 1.74),
        ]
        radius = scale * 0.49
        for index, (gx, gy) in enumerate(positions):
            x = cx + gx * scale * 0.86
            y = cy + gy * scale
            points = []
            rotation = phase * 0.04 + (index % 3) * 0.02
            for corner in range(6):
                angle = math.pi / 6.0 + corner * math.pi / 3.0 + rotation
                points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
            pulse = 0.78 + 0.22 * math.sin(phase * 1.9 + index * 0.8)
            fill_alpha = int(alpha * 0.20 * pulse)
            outline_alpha = int(alpha * (0.65 + 0.25 * pulse))
            draw.polygon(points, fill=COLORS["gold"] + (fill_alpha,), outline=COLORS["gold"] + (outline_alpha,))
        glow = overlay.filter(ImageFilter.GaussianBlur(18 if not QUICK_MODE else 9))
        image.alpha_composite(glow)
        image.alpha_composite(overlay)

    def draw_sky_grid(self, image: Image.Image, alpha: int = 80, label: bool = True):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        x0, y0, x1, y1 = self.map_box
        draw.rounded_rectangle(
            self.map_box,
            radius=int((y1 - y0) * 0.47),
            fill=(1, 7, 18, 154),
            outline=COLORS["cyan"] + (80,),
            width=2 if not QUICK_MODE else 1,
        )

        for lat in (-60, -30, 0, 30, 60):
            lon_values = np.linspace(-180, 180, 240)
            lat_values = np.full_like(lon_values, lat)
            px, py = hammer_project(lon_values % 360.0, lat_values)
            screen = self._to_screen(px, py)
            draw.line([tuple(point) for point in screen], fill=COLORS["muted"] + (alpha if lat else min(150, alpha + 65),), width=1)

        for lon in range(-150, 180, 30):
            lat_values = np.linspace(-89.5, 89.5, 220)
            lon_values = np.full_like(lat_values, lon % 360.0)
            px, py = hammer_project(lon_values, lat_values)
            screen = self._to_screen(px, py)
            draw.line([tuple(point) for point in screen], fill=COLORS["muted"] + (alpha,), width=1)

        # A broader glow marks the Milky Way plane, b = 0°.
        plane = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(plane)
        lon_values = np.linspace(-180, 180, 300)
        px, py = hammer_project(lon_values % 360.0, np.zeros_like(lon_values))
        screen = self._to_screen(px, py)
        pd.line([tuple(point) for point in screen], fill=COLORS["gold"] + (72,), width=8 if not QUICK_MODE else 4)
        plane = plane.filter(ImageFilter.GaussianBlur(8 if not QUICK_MODE else 4))
        overlay.alpha_composite(plane)

        image.alpha_composite(overlay)
        if label:
            draw_text(
                image,
                "GALACTIC COORDINATES // MILKY WAY PLANE",
                (x0 + (18 if not QUICK_MODE else 9), y1 + (28 if not QUICK_MODE else 14)),
                size=15 if not QUICK_MODE else 7,
                fill=COLORS["muted"] + (190,),
                bold=True,
                stroke=1,
            )

    def draw_points(
        self,
        image: Image.Image,
        indices: np.ndarray,
        reveal: float = 1.0,
        mode: str = "ice",
        alpha: int = 210,
        size_boost: float = 1.0,
    ):
        count = int(round(len(indices) * clamp(reveal)))
        if count <= 0:
            return
        selected = indices[:count]
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        base_radius = (2.0 if not QUICK_MODE else 1.0) * size_boost
        for index in selected:
            x, y = self.point_xy[int(index)]
            if mode == "instrument":
                colour = INSTRUMENT_COLORS.get(self.instrument[int(index)], COLORS["ice"])
            elif mode == "density":
                value = float(self.density_norm[int(index)])
                colour = tuple(
                    int(round(lerp(COLORS["blue"][channel], COLORS["rose"][channel], value)))
                    for channel in range(3)
                )
            else:
                colour = COLORS["ice"]
            radius = base_radius * (0.82 + 0.55 * float(self.density_norm[int(index)]))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour + (alpha,))
            if int(index) % (85 if not QUICK_MODE else 38) == 0:
                gr = radius * 4.2
                gd.ellipse((x - gr, y - gr, x + gr, y + gr), fill=colour + (38,))
        glow = glow.filter(ImageFilter.GaussianBlur(7 if not QUICK_MODE else 3))
        image.alpha_composite(glow)
        image.alpha_composite(overlay)

    def draw_hotspots(self, image: Image.Image, local: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for index, row in self.density.iterrows():
            x, y = self.density_xy[index]
            weight = math.log1p(float(row["observation_records"])) / math.log1p(self.max_cell_records)
            pulse = 0.65 + 0.35 * math.sin(local * 8.0 + index * 0.77)
            radius = (10 + 35 * weight * (0.65 + 0.35 * local)) * OUT_W / 1080
            color = tuple(
                int(round(lerp(COLORS["violet"][channel], COLORS["rose"][channel], weight)))
                for channel in range(3)
            )
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=color + (int(70 + 120 * weight * pulse),),
                width=max(1, int(2.0 * OUT_W / 1080)),
            )
            if index < 7:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=COLORS["white"] + (220,))
        overlay = overlay.filter(ImageFilter.GaussianBlur(0.45 if not QUICK_MODE else 0.25))
        image.alpha_composite(overlay)

    def draw_intro(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[0]
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        center = (OUT_W * 0.50, OUT_H * 0.39)
        self.draw_mirror(image, center, 62 * OUT_W / 1080, phase=t, alpha=int(230 * (0.45 + 0.55 * local)))
        draw = ImageDraw.Draw(image)
        rng = np.random.default_rng(20211225)
        points = 160 if not QUICK_MODE else 70
        for index in range(points):
            angle = rng.uniform(0, 2 * math.pi) + t * (0.12 + 0.04 * (index % 5))
            target_r = rng.uniform(115, 410) * OUT_W / 1080
            radius = lerp(target_r * 1.8, target_r, local)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + 0.68 * radius * math.sin(angle)
            a = int(35 + 145 * local * rng.uniform(0.3, 1.0))
            r = rng.uniform(0.8, 2.4) * OUT_W / 1080
            draw.ellipse((x - r, y - r, x + r, y + r), fill=COLORS["cyan"] + (a,))
        self.panel(image, (int(OUT_W * 0.10), int(OUT_H * 0.68), int(OUT_W * 0.90), int(OUT_H * 0.80)), alpha=158)
        draw_text(
            image,
            "A POINTED OBSERVATORY, NOT AN ALL-SKY CAMERA",
            (OUT_W // 2, int(OUT_H * 0.716)),
            size=22 if not QUICK_MODE else 11,
            fill=COLORS["gold"] + (240,),
            bold=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            "one target • one programme • one direction at a time",
            (OUT_W // 2, int(OUT_H * 0.755)),
            size=17 if not QUICK_MODE else 8,
            fill=COLORS["white"] + (220,),
            anchor="ma",
            stroke=1,
        )

    def draw_all_sky(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "all_sky")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_sky_grid(image, alpha=45, label=True)
        self.draw_points(image, self.order_all, reveal=min(local * 1.18, 1.0), mode="ice", alpha=205)
        self.panel(image, (int(OUT_W * 0.09), int(OUT_H * 0.71), int(OUT_W * 0.91), int(OUT_H * 0.82)), alpha=154)
        draw_text(
            image,
            "THE PUBLIC JWST ARCHIVE, PLACED ON THE REAL SKY",
            (OUT_W // 2, int(OUT_H * 0.748)),
            size=21 if not QUICK_MODE else 10,
            fill=COLORS["cyan"] + (240,),
            bold=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            f"{self.summary['observation_records']:,} deduplicated observation records",
            (OUT_W // 2, int(OUT_H * 0.785)),
            size=17 if not QUICK_MODE else 8,
            fill=COLORS["white"] + (220,),
            anchor="ma",
            stroke=1,
        )

    def draw_density(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "density")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_sky_grid(image, alpha=34, label=False)
        self.draw_points(image, self.order_density, reveal=1.0, mode="density", alpha=150, size_boost=0.95)
        self.draw_hotspots(image, local)
        self.panel(image, (int(OUT_W * 0.08), int(OUT_H * 0.70), int(OUT_W * 0.92), int(OUT_H * 0.83)), alpha=160)
        draw_text(
            image,
            "WEBB RETURNED TO SOME PATCHES AGAIN AND AGAIN",
            (OUT_W // 2, int(OUT_H * 0.742)),
            size=22 if not QUICK_MODE else 11,
            fill=COLORS["rose"] + (240,),
            bold=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            "record density can mean deep fields, mosaics, spectra, or monitoring",
            (OUT_W // 2, int(OUT_H * 0.783)),
            size=16 if not QUICK_MODE else 8,
            fill=COLORS["white"] + (220,),
            anchor="ma",
            stroke=1,
        )

    def draw_timeline(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "timeline")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_sky_grid(image, alpha=38, label=False)
        visible_mask = self.time_norm[self.order_time] <= local
        visible_count = int(np.count_nonzero(visible_mask))
        self.draw_points(image, self.order_time, reveal=visible_count / max(len(self.order_time), 1), mode="instrument", alpha=215)

        start_mjd = self.summary.get("date_start_mjd")
        end_mjd = self.summary.get("date_end_mjd")
        if start_mjd is not None and end_mjd is not None:
            current_mjd = lerp(float(start_mjd), float(end_mjd), local)
            date_label = format_date_from_mjd(current_mjd)
        else:
            date_label = "OBSERVATION DATE UNKNOWN"

        x0, _, x1, _ = self.map_box
        bar_y = int(OUT_H * 0.715)
        ImageDraw.Draw(image).rounded_rectangle(
            (x0, bar_y, x1, bar_y + (18 if not QUICK_MODE else 9)),
            radius=9 if not QUICK_MODE else 4,
            fill=(12, 32, 58, 210),
            outline=COLORS["cyan"] + (65,),
        )
        ImageDraw.Draw(image).rounded_rectangle(
            (x0, bar_y, int(lerp(x0, x1, local)), bar_y + (18 if not QUICK_MODE else 9)),
            radius=9 if not QUICK_MODE else 4,
            fill=COLORS["cyan"] + (205,),
        )
        draw_text(
            image,
            date_label,
            (OUT_W // 2, int(OUT_H * 0.755)),
            size=31 if not QUICK_MODE else 15,
            fill=COLORS["white"] + (245,),
            bold=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            f"archive records revealed // {visible_count:,} of {len(self.order_time):,} rendered",
            (OUT_W // 2, int(OUT_H * 0.795)),
            size=16 if not QUICK_MODE else 8,
            fill=COLORS["muted"] + (215,),
            anchor="ma",
            stroke=1,
        )

    def draw_instruments(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "instruments")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_sky_grid(image, alpha=32, label=False)
        self.draw_points(image, self.order_all, reveal=1.0, mode="instrument", alpha=215, size_boost=1.02)

        panel_x0 = int(OUT_W * 0.12)
        panel_x1 = int(OUT_W * 0.88)
        panel_y0 = int(OUT_H * 0.70)
        panel_y1 = int(OUT_H * 0.84)
        self.panel(image, (panel_x0, panel_y0, panel_x1, panel_y1), alpha=166)
        names = [name for name in ("NIRCAM", "NIRSPEC", "MIRI", "NIRISS", "FGS") if name in self.summary["instrument_counts"]]
        if "OTHER" in self.summary["instrument_counts"]:
            names.append("OTHER")
        total = max(sum(self.summary["instrument_counts"].values()), 1)
        gap = (panel_x1 - panel_x0 - 40 * OUT_W / 1080) / max(len(names), 1)
        for index, name in enumerate(names):
            x = panel_x0 + 20 * OUT_W / 1080 + gap * (index + 0.5)
            y = panel_y0 + 44 * OUT_H / 1920
            radius = (10 + 4 * math.sin(local * math.pi + index)) * OUT_W / 1080
            colour = INSTRUMENT_COLORS.get(name, COLORS["ice"])
            ImageDraw.Draw(image).ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour + (235,))
            count = int(self.summary["instrument_counts"].get(name, 0))
            draw_text(image, name, (int(x), int(y + 30 * OUT_H / 1920)), size=15 if not QUICK_MODE else 7, fill=COLORS["white"] + (230,), bold=True, anchor="ma", stroke=1)
            draw_text(image, f"{100.0 * count / total:.0f}%", (int(x), int(y + 58 * OUT_H / 1920)), size=14 if not QUICK_MODE else 7, fill=colour + (225,), bold=True, anchor="ma", stroke=1)
        draw_text(
            image,
            "ONE OBSERVATORY // DIFFERENT INFRARED INSTRUMENTS",
            (OUT_W // 2, int(OUT_H * 0.825)),
            size=18 if not QUICK_MODE else 9,
            fill=COLORS["muted"] + (215,),
            bold=True,
            anchor="ma",
            stroke=1,
        )

    def draw_finale(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "finale")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_sky_grid(image, alpha=30, label=False)
        self.draw_points(image, self.order_all, reveal=1.0, mode="instrument", alpha=220, size_boost=1.05)
        self.draw_hotspots(image, local)
        self.panel(image, (int(OUT_W * 0.08), int(OUT_H * 0.61), int(OUT_W * 0.92), int(OUT_H * 0.82)), alpha=184)
        draw_text(
            image,
            "WHERE THE JAMES WEBB TELESCOPE LOOKED",
            (OUT_W // 2, int(OUT_H * 0.652)),
            size=29 if not QUICK_MODE else 14,
            fill=COLORS["white"] + (248,),
            bold=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            f"{self.summary['observation_records']:,} records  •  {self.summary['unique_targets']:,} target names",
            (OUT_W // 2, int(OUT_H * 0.707)),
            size=19 if not QUICK_MODE else 9,
            fill=COLORS["gold"] + (238,),
            bold=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            f"{self.summary['date_start']}  →  {self.summary['date_end']}",
            (OUT_W // 2, int(OUT_H * 0.752)),
            size=18 if not QUICK_MODE else 9,
            fill=COLORS["cyan"] + (230,),
            bold=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            "a living archive map • new observations keep changing it",
            (OUT_W // 2, int(OUT_H * 0.790)),
            size=16 if not QUICK_MODE else 8,
            fill=COLORS["white"] + (220,),
            anchor="ma",
            stroke=1,
        )

    def draw_source_hud(self, image: Image.Image):
        live = bool(self.summary["is_live_observational_data"])
        label = "SOURCE // MAST PUBLIC JWST OBSERVATIONS" if live else "PREVIEW SOURCE // SYNTHETIC FIXTURE"
        colour = COLORS["cyan"] if live else COLORS["gold"]
        draw_text(
            image,
            label,
            (OUT_W - (46 if not QUICK_MODE else 23), 70 if not QUICK_MODE else 35),
            size=16 if not QUICK_MODE else 8,
            fill=colour + (235,),
            bold=True,
            anchor="ra",
            stroke=1,
        )
        draw_text(
            image,
            f"RECORDS // {self.summary['observation_records']:,}",
            (OUT_W - (46 if not QUICK_MODE else 23), 100 if not QUICK_MODE else 50),
            size=15 if not QUICK_MODE else 7,
            fill=COLORS["muted"] + (205,),
            anchor="ra",
            stroke=1,
        )
        draw_text(
            image,
            f"PROGRAMMES // {self.summary['unique_programmes']:,}",
            (OUT_W - (46 if not QUICK_MODE else 23), 127 if not QUICK_MODE else 63),
            size=15 if not QUICK_MODE else 7,
            fill=COLORS["muted"] + (195,),
            anchor="ra",
            stroke=1,
        )

    def draw_titles(self, image: Image.Image, t: float, shot_name: str):
        intro_end = 6.7 if not QUICK_MODE else 1.4
        alpha = int(255 * smoothstep((t - 0.2) / 0.8) * (1.0 - smoothstep((t - intro_end) / 0.65)))
        if alpha > 4:
            draw_text(
                image,
                "WHERE THE JAMES WEBB",
                (54 if not QUICK_MODE else 27, 88 if not QUICK_MODE else 43),
                size=39 if not QUICK_MODE else 19,
                fill=COLORS["white"] + (alpha,),
                bold=True,
            )
            draw_text(
                image,
                "TELESCOPE LOOKED",
                (54 if not QUICK_MODE else 27, 136 if not QUICK_MODE else 67),
                size=39 if not QUICK_MODE else 19,
                fill=COLORS["white"] + (alpha,),
                bold=True,
            )
            draw_text(
                image,
                CONFIG["subtitle"],
                (56 if not QUICK_MODE else 28, 188 if not QUICK_MODE else 94),
                size=20 if not QUICK_MODE else 10,
                fill=COLORS["cyan"] + (min(alpha, 230),),
                bold=True,
            )

        labels = {
            "intro": "A TELESCOPE THAT CHOOSES DIRECTIONS",
            "all_sky": "THE PUBLIC ARCHIVE AS AN ALL-SKY MAP",
            "density": "THE PLACES WEBB VISITED REPEATEDLY",
            "timeline": "THE OBSERVING FOOTPRINT GROWS IN TIME",
            "instruments": "FIVE INSTRUMENT GROUPS, ONE INFRARED OBSERVATORY",
            "finale": "A LIVING MAP OF WHERE WEBB LOOKED",
        }
        if t > (5.1 if not QUICK_MODE else 1.2):
            draw_text(
                image,
                labels[shot_name],
                (54 if not QUICK_MODE else 27, 60 if not QUICK_MODE else 30),
                size=18 if not QUICK_MODE else 9,
                fill=COLORS["muted"] + (205,),
                bold=True,
                stroke=1,
            )

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        y0 = OUT_H - (244 if not QUICK_MODE else 124)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle(
            (44 if not QUICK_MODE else 22, y0, OUT_W - (44 if not QUICK_MODE else 22), y0 + (124 if not QUICK_MODE else 66)),
            radius=24 if not QUICK_MODE else 12,
            fill=(2, 6, 15, 176),
            outline=(80, 190, 228, 66),
            width=1,
        )
        image.alpha_composite(panel)
        draw_wrapped_text(
            image,
            text,
            (68 if not QUICK_MODE else 34, y0 + (28 if not QUICK_MODE else 14)),
            OUT_W - (136 if not QUICK_MODE else 68),
            size=29 if not QUICK_MODE else 14,
            fill=COLORS["white"] + (245,),
        )

    def draw_hud_noise(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for item in self.hud:
            pulse = 0.5 + 0.5 * math.sin(t * 1.9 + item["phase"])
            if pulse < 0.74:
                continue
            y = (item["y"] + t * 9.0) % OUT_H
            draw.line(
                (item["x"], y, item["x"] + item["length"], y),
                fill=COLORS["cyan"] + (int(item["a"] * pulse),),
                width=1,
            )
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
        elif name == "density":
            self.draw_density(image, t)
        elif name == "timeline":
            self.draw_timeline(image, t)
        elif name == "instruments":
            self.draw_instruments(image, t)
        elif name == "finale":
            self.draw_finale(image, t)
        self.draw_source_hud(image)
        self.draw_titles(image, t, name)
        self.draw_caption(image, t)
        self.draw_hud_noise(image, t)
        array = np.asarray(image.convert("RGB"))
        array = apply_grade(array)
        array = np.clip(array.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / 0.9)
        fade_out = 1.0 - smoothstep((t - (float(CONFIG["duration_s"]) - 1.1)) / 1.0)
        return np.clip(array.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def render_video(scene: JWSTLookedScene) -> Path:
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
        for t in tqdm(times, desc="Rendering JWST archive short"):
            writer.append_data(scene.render_frame(float(t)))
    shutil.copyfile(raw_video, final_video)
    print("Final video:", final_video.resolve())
    return final_video


def main():
    print("Loading public JWST observation metadata ...")
    frame, source, notes, requests_used = load_all_data()
    print("Transforming coordinates and analysing archive density ...")
    frame, density, targets, summary = prepare_observations(frame)
    catalogue_path, summary_path = save_data_products(frame, density, targets, summary, notes, requests_used)
    create_scientific_plots(frame, density, summary)

    print("Data source:", source)
    print("Observation records:", f"{summary['observation_records']:,}")
    print("Unique target names:", f"{summary['unique_targets']:,}")
    print("Unique programmes:", f"{summary['unique_programmes']:,}")
    print("Observation span:", summary["date_start"], "to", summary["date_end"])
    for instrument, count in summary["instrument_counts"].items():
        print(f"Instrument {instrument}: {count:,}")
    for note in notes:
        print("Data note:", note)
    print("Data:", catalogue_path.resolve())
    print("Summary:", summary_path.resolve())

    scene = JWSTLookedScene(frame, density, targets, summary)
    preview_times = [
        1.0,
        min(10.0, float(CONFIG["duration_s"]) * 0.20),
        min(22.0, float(CONFIG["duration_s"]) * 0.39),
        min(34.0, float(CONFIG["duration_s"]) * 0.60),
        min(45.0, float(CONFIG["duration_s"]) * 0.79),
        float(CONFIG["duration_s"]) - 1.0,
    ]
    for preview_time in tqdm(preview_times, desc="Preview frames"):
        Image.fromarray(scene.render_frame(float(preview_time))).save(
            PREVIEW_DIR / f"preview_{int(preview_time):02d}s.png"
        )
    render_video(scene)
    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-", path.name)


if __name__ == "__main__":
    main()
