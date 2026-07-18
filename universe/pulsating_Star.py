from __future__ import annotations

"""
A Pulsating Star Beating Like a Cosmic Heart
============================================

A cinematic vertical YouTube Short renderer built from a real TESS light curve
retrieved from the Mikulski Archive for Space Telescopes (MAST) using
Lightkurve.

Default target
--------------
RR Lyrae (RR Lyr), the prototype of the RR Lyrae class of pulsating variable
stars. The script automatically chooses an available TESS sector unless you
request a specific one.

What the video shows
--------------------
- A real TESS brightness time series downloaded from MAST when the script runs.
- The repeating rise and fall of the star across an observing sector.
- A period estimate measured directly from the downloaded light curve.
- A phase-folded pulse profile that stacks many cycles into one "heartbeat."
- An animated stellar disk that expands and contracts in step with the measured
  brightness pattern.
- Cadence, observation span, point count, pulse amplitude, period, and the
  number of cycles covered by TESS.
- A target-pixel stamp and pipeline aperture when a TESS Target Pixel File is
  available.
  
Official sources and tools
--------------------------
TESS data at MAST:
    https://archive.stsci.edu/missions-and-data/tess


Honesty / interpretation rules
------------------------------
- TESS records brightness measurements, not a resolved movie of the stellar
  surface. The expanding star graphic is a visual metaphor synchronized to the
  light curve.
- The period is estimated from the downloaded data. It is not substituted from
  a catalog value.
- A phase-folded curve combines measurements from many cycles; it does not show
  one single uninterrupted pulse.
- Pixel colors are contrast-stretched for visibility and are not true color.
- Different TESS products, sectors, detrending choices, and data gaps can shift
  the measured amplitude or period slightly.

Offline fallback
----------------
If MAST or Lightkurve is unavailable, the script uses a clearly labeled,
deterministic RR-Lyrae-like layout fixture. The fixture is not real TESS
photometry and is never labeled as live data.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg requests tqdm scipy lightkurve

Run final quality
-----------------
    python a_pulsating_star_beating_like_a_cosmic_heart_short.py

Run quick preview
-----------------
    PULSATING_STAR_SHORT_QUICK=1 python a_pulsating_star_beating_like_a_cosmic_heart_short.py

Choose another pulsating star
-----------------------------
    PULSATING_STAR_TARGET="Delta Cep" python a_pulsating_star_beating_like_a_cosmic_heart_short.py

Choose a TESS sector
--------------------
    PULSATING_STAR_SECTOR=40 python a_pulsating_star_beating_like_a_cosmic_heart_short.py

Force offline layout testing
----------------------------
    PULSATING_STAR_SHORT_OFFLINE=1 PULSATING_STAR_SHORT_QUICK=1 \
        python a_pulsating_star_beating_like_a_cosmic_heart_short.py
"""

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import requests
except Exception:
    requests = None

try:
    import lightkurve as lk
except Exception:
    lk = None

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None


# =============================================================================
# Configuration
# =============================================================================

QUICK_MODE = os.environ.get("PULSATING_STAR_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("PULSATING_STAR_SHORT_OFFLINE", "0") == "1"
REFRESH = os.environ.get("PULSATING_STAR_SHORT_REFRESH", "0") == "1"
FETCH_PIXELS = os.environ.get("PULSATING_STAR_FETCH_PIXELS", "1") == "1"

TARGET_QUERY = os.environ.get("PULSATING_STAR_TARGET", "RR Lyr")
SECTOR_TEXT = os.environ.get("PULSATING_STAR_SECTOR", "any").strip()
REQUESTED_SECTOR = int(SECTOR_TEXT) if SECTOR_TEXT and SECTOR_TEXT.lower() not in {"any", "latest", "none"} else None

OUTPUT_ROOT = Path("a_pulsating_star_beating_like_a_cosmic_heart_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
CACHE_ROOT = OUTPUT_ROOT / "cache"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT, CACHE_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "width": 540 if QUICK_MODE else 1080,
    "height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "basename": "a_pulsating_star_beating_like_a_cosmic_heart",
    "title": "A PULSATING STAR",
    "subtitle": "BEATING LIKE A COSMIC HEART • REAL TESS DATA",
    "stars": 580 if QUICK_MODE else 1050,
    "max_render_points": 4200 if QUICK_MODE else 11000,
    "max_saved_points": 100000,
    "period_min_days": 0.12,
    "period_max_days": 2.0,
    "cache_hours": 72,
}

W = CONFIG["width"]
H = CONFIG["height"]
SIZE = (W, H)
SCALE = W / 1080.0

COLORS = {
    "bg": (4, 8, 16),
    "white": (246, 249, 255),
    "muted": (150, 198, 222),
    "cyan": (92, 223, 255),
    "gold": (255, 197, 92),
    "violet": (201, 116, 255),
    "green": (104, 255, 181),
    "red": (255, 115, 125),
    "orange": (255, 160, 84),
    "star": (255, 235, 202),
}

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.0 if not QUICK_MODE else 1.8},
    {"name": "curve", "start": 7.0 if not QUICK_MODE else 1.8, "end": 20.0 if not QUICK_MODE else 4.4},
    {"name": "events", "start": 20.0 if not QUICK_MODE else 4.4, "end": 32.0 if not QUICK_MODE else 6.8},
    {"name": "period", "start": 32.0 if not QUICK_MODE else 6.8, "end": 44.0 if not QUICK_MODE else 9.1},
    {"name": "stats", "start": 44.0 if not QUICK_MODE else 9.1, "end": 53.0 if not QUICK_MODE else 10.9},
    {"name": "outro", "start": 53.0 if not QUICK_MODE else 10.9, "end": CONFIG["duration_s"]},
]

CAPTION_TEXTS = [
    "This star repeatedly swells brighter and fades again, creating a rhythm that looks like a cosmic heartbeat.",
    "TESS measured the star point by point. Every rise and fall in this curve is a change in recorded brightness.",
    "Zooming into the pulse reveals a steep brightening and a longer fading branch, repeated cycle after cycle.",
    "Folding the observations on the strongest measured period stacks many pulses into one repeating heartbeat.",
    "The period, pulse amplitude, cadence, observation span, and number of cycles are all measured from this light curve.",
    "The star animation is a visual metaphor. TESS measured unresolved brightness, not the star's physical surface.",
]

if QUICK_MODE:
    CAPTIONS = [
        (0.2, 1.8, CAPTION_TEXTS[0]),
        (1.8, 4.4, CAPTION_TEXTS[1]),
        (4.4, 6.8, CAPTION_TEXTS[2]),
        (6.8, 9.1, CAPTION_TEXTS[3]),
        (9.1, 10.9, CAPTION_TEXTS[4]),
        (10.9, 11.8, CAPTION_TEXTS[5]),
    ]
else:
    CAPTIONS = [
        (0.4, 6.9, CAPTION_TEXTS[0]),
        (7.0, 20.0, CAPTION_TEXTS[1]),
        (20.1, 32.0, CAPTION_TEXTS[2]),
        (32.1, 44.0, CAPTION_TEXTS[3]),
        (44.1, 53.0, CAPTION_TEXTS[4]),
        (53.1, 57.7, CAPTION_TEXTS[5]),
    ]


# =============================================================================
# Data model
# =============================================================================

@dataclass
class TESSMetadata:
    target_query: str
    object_name: str
    tic_id: str
    sector: int
    camera: int
    ccd: int
    author: str
    exposure_s: float
    flux_origin: str
    ra_deg: float
    dec_deg: float
    source_kind: str
    data_status: str
    offline_fixture: bool
    fetched_at_utc: str
    source_url: str
    note: str = ""


@dataclass
class LightCurveAnalysis:
    time_day: np.ndarray
    flux_norm: np.ndarray
    rel_flux_pct: np.ndarray
    trend_norm: np.ndarray
    residual_pct: np.ndarray
    cadence_s: float
    span_days: float
    point_count: int
    robust_sigma_pct: float
    variability_range_pct: float
    strongest_brightening_index: int
    strongest_dimming_index: int
    strongest_brightening_pct: float
    strongest_dimming_pct: float
    candidate_event_count: int
    dominant_period_days: float
    folded_phase: np.ndarray
    folded_flux_pct: np.ndarray
    pixel_image: Optional[np.ndarray]
    aperture_mask: Optional[np.ndarray]


# =============================================================================
# Generic utilities
# =============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_float(value, default=np.nan) -> float:
    try:
        if hasattr(value, "value"):
            value = value.value
        return float(value)
    except Exception:
        return float(default)


def safe_int(value, default=0) -> int:
    try:
        if hasattr(value, "value"):
            value = value.value
        return int(float(value))
    except Exception:
        return int(default)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def ease_in_out_sine(t: float) -> float:
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


def get_shot(t: float) -> Dict:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


def clip_text(text: str, n: int = 34) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(7, int(size)))
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int = 28,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    anchor: str = "la",
    stroke: int = 2,
):
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold),
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
    font = get_font(size, bold)
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for word in words:
        test = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), test, font=font, stroke_width=2)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 220))
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += bbox[3] - bbox[1] + line_spacing


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(path: Path):
    lines = []
    for i, (start, end, text) in enumerate(CAPTIONS, start=1):
        lines.extend([str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_vignette(width: int, height: int, strength: float = 0.25) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1 - strength * radius**1.8, 0, 1).astype(np.float32)


VIGNETTE = make_vignette(W, H)


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    center = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - center))
    return float(1.4826 * mad)


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    window = max(5, int(window))
    if window % 2 == 0:
        window += 1
    window = min(window, max(5, len(values) // 2 * 2 - 1))
    series = pd.Series(values)
    trend = series.rolling(window, center=True, min_periods=max(3, window // 5)).median()
    trend = trend.interpolate(limit_direction="both").bfill().ffill()
    arr = trend.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        arr = np.where(np.isfinite(arr), arr, np.nanmedian(values))
    return arr


def select_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, max_points).astype(int)


# =============================================================================
# Live TESS retrieval
# =============================================================================


def table_value(table, row_index: int, column: str, default=None):
    try:
        value = table[column][row_index]
        if hasattr(value, "mask") and bool(value.mask):
            return default
        if hasattr(value, "value"):
            return value.value
        return value
    except Exception:
        return default


def choose_search_index(search_result, prefer_exptime_s: float = 120.0) -> int:
    table = search_result.table
    scores = []
    for idx in range(len(search_result)):
        author = str(table_value(table, idx, "author", "")).upper()
        exptime = safe_float(table_value(table, idx, "exptime", np.nan))
        sector = safe_int(table_value(table, idx, "sequence_number", 0))
        author_score = 0 if author == "SPOC" else (1 if "SPOC" in author else 3)
        exposure_score = abs(math.log(max(exptime, 1.0) / max(prefer_exptime_s, 1.0))) if np.isfinite(exptime) else 9.0
        sector_score = -sector
        scores.append((author_score, exposure_score, sector_score, idx))
    return min(scores)[-1] if scores else 0


def fetch_target_pixel_stamp(target: str, sector: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if lk is None or not FETCH_PIXELS:
        return None, None
    try:
        search = lk.search_targetpixelfile(target, mission="TESS", sector=sector)
        if len(search) == 0:
            return None, None
        idx = choose_search_index(search, prefer_exptime_s=120.0)
        tpf = search[idx].download(download_dir=str(CACHE_ROOT))
        cube = np.asarray(tpf.flux.value, dtype=float)
        if cube.ndim != 3:
            return None, None
        if cube.shape[0] > 5000:
            sample = np.linspace(0, cube.shape[0] - 1, 5000).astype(int)
            cube = cube[sample]
        image = np.nanmedian(cube, axis=0)
        mask = None
        try:
            mask = np.asarray(tpf.pipeline_mask, dtype=bool)
        except Exception:
            mask = None
        return image, mask
    except Exception:
        return None, None


def fetch_live_tess_data() -> Tuple[np.ndarray, np.ndarray, TESSMetadata, Optional[np.ndarray], Optional[np.ndarray]]:
    if lk is None:
        raise RuntimeError("lightkurve is not installed")

    kwargs = {"mission": "TESS"}
    if REQUESTED_SECTOR is not None:
        kwargs["sector"] = REQUESTED_SECTOR
    search = lk.search_lightcurve(TARGET_QUERY, **kwargs)
    if len(search) == 0:
        raise RuntimeError(f"No TESS light curve found for {TARGET_QUERY!r}")

    idx = choose_search_index(search, prefer_exptime_s=120.0)
    row_table = search.table
    selected_author = str(table_value(row_table, idx, "author", "unknown"))
    selected_sector = safe_int(table_value(row_table, idx, "sequence_number", REQUESTED_SECTOR or 0))
    selected_exptime = safe_float(table_value(row_table, idx, "exptime", np.nan))

    lc = search[idx].download(download_dir=str(CACHE_ROOT), quality_bitmask="default")
    if lc is None:
        raise RuntimeError("Lightkurve returned no downloaded light curve")

    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    good = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    time = time[good]
    flux = flux[good]
    if len(time) < 100:
        raise RuntimeError("Downloaded light curve contains too few valid points")

    order = np.argsort(time)
    time = time[order]
    flux = flux[order]
    time_day = time - np.nanmin(time)

    meta = dict(getattr(lc, "meta", {}) or {})
    object_name = str(meta.get("OBJECT") or meta.get("LABEL") or TARGET_QUERY)
    tic_id = str(meta.get("TICID") or meta.get("TARGETID") or "")
    sector = safe_int(meta.get("SECTOR"), selected_sector)
    camera = safe_int(meta.get("CAMERA"), 0)
    ccd = safe_int(meta.get("CCD"), 0)
    exposure_s = safe_float(meta.get("EXPOSURE"), selected_exptime)
    if not np.isfinite(exposure_s) or exposure_s <= 0:
        exposure_s = float(np.nanmedian(np.diff(time_day)) * 86400.0)
    flux_origin = str(meta.get("FLUX_ORIGIN") or getattr(lc, "flux_column", "flux"))
    ra_deg = safe_float(meta.get("RA_OBJ", meta.get("RA")))
    dec_deg = safe_float(meta.get("DEC_OBJ", meta.get("DEC")))

    pixel_image, aperture_mask = fetch_target_pixel_stamp(TARGET_QUERY, sector)

    metadata = TESSMetadata(
        target_query=TARGET_QUERY,
        object_name=object_name,
        tic_id=tic_id,
        sector=sector,
        camera=camera,
        ccd=ccd,
        author=selected_author,
        exposure_s=exposure_s,
        flux_origin=flux_origin,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        source_kind="TESS light curve downloaded from MAST via Lightkurve",
        data_status="live",
        offline_fixture=False,
        fetched_at_utc=iso_z(utc_now()),
        source_url="https://archive.stsci.edu/missions-and-data/tess",
        note="The pulse period and amplitude are descriptive measurements from the selected TESS light curve.",
    )
    return time_day, flux, metadata, pixel_image, aperture_mask


# =============================================================================
# Offline fixture
# =============================================================================


def make_offline_fixture() -> Tuple[np.ndarray, np.ndarray, TESSMetadata, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(77119)
    cadence_s = 120.0
    period_days = 0.5668
    time_day = np.arange(0, 26.8, cadence_s / 86400.0)
    phase = (time_day / period_days) % 1.0

    # A deterministic RR-Lyrae-like asymmetric waveform. The fundamental is
    # intentionally dominant so the same period-finding code used on live data
    # recovers the injected pulse period instead of a harmonic.
    pulse = (
        0.145 * np.sin(2 * np.pi * phase + 0.25)
        + 0.050 * np.sin(4 * np.pi * phase + 1.15)
        + 0.022 * np.sin(6 * np.pi * phase + 1.85)
        + 0.010 * np.sin(8 * np.pi * phase + 2.25)
    )
    amplitude_modulation = 1.0 + 0.025 * np.sin(2 * np.pi * time_day / 18.0)
    slow_baseline = 0.0025 * np.sin(2 * np.pi * time_day / 8.2)
    noise = rng.normal(0, 0.0018, len(time_day))
    flux = 1.0 + amplitude_modulation * pulse + slow_baseline + noise

    yy, xx = np.mgrid[-5:6, -5:6]
    pixel = 2600 * np.exp(-((xx - 0.15) ** 2 + (yy + 0.10) ** 2) / (2 * 1.18 ** 2))
    pixel += 210 * np.exp(-((xx + 3.0) ** 2 + (yy - 2.4) ** 2) / (2 * 0.78 ** 2))
    pixel += rng.normal(12, 3, pixel.shape)
    mask = ((xx - 0.15) ** 2 + (yy + 0.10) ** 2) <= 2.55 ** 2

    metadata = TESSMetadata(
        target_query=TARGET_QUERY,
        object_name="RR Lyrae",
        tic_id="offline fixture",
        sector=REQUESTED_SECTOR or 40,
        camera=2,
        ccd=3,
        author="fixture",
        exposure_s=cadence_s,
        flux_origin="synthetic RR-Lyrae-like fixture flux",
        ra_deg=np.nan,
        dec_deg=np.nan,
        source_kind="offline deterministic pulsating-star layout fixture",
        data_status="offline-fixture",
        offline_fixture=True,
        fetched_at_utc=iso_z(utc_now()),
        source_url="https://archive.stsci.edu/missions-and-data/tess",
        note="This fallback resembles an RR Lyrae pulse but is not real TESS photometry.",
    )
    return time_day, flux, metadata, pixel, mask


# =============================================================================
# Analysis
# =============================================================================


def estimate_dominant_period(time_day: np.ndarray, flux_norm: np.ndarray) -> float:
    if len(time_day) < 100:
        return np.nan
    span = float(np.nanmax(time_day) - np.nanmin(time_day))
    if span <= CONFIG["period_min_days"] * 3:
        return np.nan

    dt = float(np.nanmedian(np.diff(time_day)))
    if not np.isfinite(dt) or dt <= 0:
        return np.nan
    grid_count = min(24000, max(1024, int(span / dt)))
    grid = np.linspace(time_day.min(), time_day.max(), grid_count)
    values = np.interp(grid, time_day, flux_norm)

    # Replace sharp outliers using a short local baseline, but preserve the
    # multi-day stellar modulation that we are trying to measure.
    short_window = max(21, int(0.18 / max(grid[1] - grid[0], 1e-8)))
    local_baseline = rolling_median(values, short_window)
    local_residual = values - local_baseline
    sigma = robust_sigma(local_residual)
    if np.isfinite(sigma) and sigma > 0:
        outliers = np.abs(local_residual) > 6.0 * sigma
        values = values.copy()
        values[outliers] = local_baseline[outliers]

    # Remove only a linear drift; subtracting a short rolling trend would erase
    # the star's repeating modulation and bias the period toward short cadences.
    centered_grid = grid - np.nanmean(grid)
    coeff = np.polyfit(centered_grid, values, 1)
    values = values - np.polyval(coeff, centered_grid)
    values -= np.nanmean(values)
    window = np.hanning(len(values))
    n_fft = 1 << int(math.ceil(math.log2(max(2048, len(values) * 8))))
    spectrum = np.fft.rfft(values * window, n=n_fft)
    power = np.abs(spectrum) ** 2
    freq = np.fft.rfftfreq(n_fft, d=(grid[1] - grid[0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = 1.0 / freq
    max_period = min(CONFIG["period_max_days"], span / 2.0)
    valid = (periods >= CONFIG["period_min_days"]) & (periods <= max_period) & np.isfinite(power)
    if not np.any(valid):
        return np.nan
    idxs = np.flatnonzero(valid)
    best = idxs[int(np.nanargmax(power[valid]))]
    return float(periods[best])


def make_folded_curve(time_day: np.ndarray, flux_norm: np.ndarray, period_days: float) -> Tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(period_days) or period_days <= 0:
        return np.array([]), np.array([])
    phase = (time_day % period_days) / period_days
    bins = np.linspace(0, 1, 65)
    centers = 0.5 * (bins[:-1] + bins[1:])
    medians = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        if np.any(mask):
            medians[i] = np.nanmedian((flux_norm[mask] - 1.0) * 100.0)
    good = np.isfinite(medians)
    return centers[good], medians[good]


def detect_events(residual_pct: np.ndarray, sigma_pct: float) -> Tuple[int, int, int]:
    if len(residual_pct) == 0:
        return 0, 0, 0
    bright_idx = int(np.nanargmax(residual_pct))
    dim_idx = int(np.nanargmin(residual_pct))
    threshold = max(0.15, 5.0 * sigma_pct) if np.isfinite(sigma_pct) else 0.5
    if find_peaks is not None:
        peaks, _ = find_peaks(residual_pct, height=threshold, distance=max(2, len(residual_pct) // 3000))
        event_count = int(len(peaks))
    else:
        local = (residual_pct[1:-1] > residual_pct[:-2]) & (residual_pct[1:-1] >= residual_pct[2:])
        event_count = int(np.sum(local & (residual_pct[1:-1] > threshold)))
    return bright_idx, dim_idx, event_count


def analyze_light_curve(
    time_day: np.ndarray,
    flux: np.ndarray,
    pixel_image: Optional[np.ndarray],
    aperture_mask: Optional[np.ndarray],
) -> LightCurveAnalysis:
    time_day = np.asarray(time_day, dtype=float)
    flux = np.asarray(flux, dtype=float)
    good = np.isfinite(time_day) & np.isfinite(flux) & (flux > 0)
    time_day = time_day[good]
    flux = flux[good]
    order = np.argsort(time_day)
    time_day = time_day[order]
    flux = flux[order]

    median_flux = float(np.nanmedian(flux))
    flux_norm = flux / median_flux
    cadence_days = float(np.nanmedian(np.diff(time_day)))
    cadence_s = cadence_days * 86400.0
    trend_window = int(max(31, min(2401, 0.55 / max(cadence_days, 1e-6))))
    trend_norm = rolling_median(flux_norm, trend_window)
    trend_norm = np.where(np.abs(trend_norm) > 1e-8, trend_norm, 1.0)
    rel_flux_pct = (flux_norm - 1.0) * 100.0
    residual_pct = (flux_norm / trend_norm - 1.0) * 100.0
    sigma_pct = robust_sigma(residual_pct)
    bright_idx, dim_idx, event_count = detect_events(residual_pct, sigma_pct)
    variability_range = float(np.nanpercentile(rel_flux_pct, 95) - np.nanpercentile(rel_flux_pct, 5))
    dominant_period = estimate_dominant_period(time_day, flux_norm)
    folded_phase, folded_flux_pct = make_folded_curve(time_day, flux_norm, dominant_period)

    return LightCurveAnalysis(
        time_day=time_day,
        flux_norm=flux_norm,
        rel_flux_pct=rel_flux_pct,
        trend_norm=trend_norm,
        residual_pct=residual_pct,
        cadence_s=cadence_s,
        span_days=float(time_day[-1] - time_day[0]),
        point_count=len(time_day),
        robust_sigma_pct=sigma_pct,
        variability_range_pct=variability_range,
        strongest_brightening_index=bright_idx,
        strongest_dimming_index=dim_idx,
        strongest_brightening_pct=float(residual_pct[bright_idx]),
        strongest_dimming_pct=float(residual_pct[dim_idx]),
        candidate_event_count=event_count,
        dominant_period_days=dominant_period,
        folded_phase=folded_phase,
        folded_flux_pct=folded_flux_pct,
        pixel_image=pixel_image,
        aperture_mask=aperture_mask,
    )


def collect_data() -> Tuple[TESSMetadata, LightCurveAnalysis, Dict]:
    errors = {}
    if OFFLINE_MODE:
        time_day, flux, metadata, pixel, mask = make_offline_fixture()
    else:
        try:
            time_day, flux, metadata, pixel, mask = fetch_live_tess_data()
        except Exception as exc:
            errors["live_fetch"] = str(exc)
            time_day, flux, metadata, pixel, mask = make_offline_fixture()

    analysis = analyze_light_curve(time_day, flux, pixel, mask)
    summary = {
        "generated_at_utc": iso_z(utc_now()),
        "target_query": metadata.target_query,
        "object_name": metadata.object_name,
        "tic_id": metadata.tic_id,
        "sector": metadata.sector,
        "camera": metadata.camera,
        "ccd": metadata.ccd,
        "data_status": metadata.data_status,
        "offline_fixture": metadata.offline_fixture,
        "point_count": analysis.point_count,
        "cadence_s": analysis.cadence_s,
        "span_days": analysis.span_days,
        "variability_range_pct": analysis.variability_range_pct,
        "strongest_brightening_pct": analysis.strongest_brightening_pct,
        "strongest_dimming_pct": analysis.strongest_dimming_pct,
        "candidate_event_count": analysis.candidate_event_count,
        "dominant_period_days": analysis.dominant_period_days,
        "cycles_observed": analysis.span_days / analysis.dominant_period_days if np.isfinite(analysis.dominant_period_days) and analysis.dominant_period_days > 0 else np.nan,
        "pixel_stamp_available": analysis.pixel_image is not None,
        "errors": errors,
        "warning": "The period and pulse profile are measured from the selected light curve; the expanding-star animation is illustrative.",
    }
    return metadata, analysis, summary


def save_data(metadata: TESSMetadata, analysis: LightCurveAnalysis, summary: Dict) -> Tuple[Path, Path]:
    n = min(analysis.point_count, CONFIG["max_saved_points"])
    indices = select_indices(analysis.point_count, n)
    df = pd.DataFrame({
        "time_since_start_days": analysis.time_day[indices],
        "normalized_flux": analysis.flux_norm[indices],
        "relative_flux_percent": analysis.rel_flux_pct[indices],
        "trend_normalized": analysis.trend_norm[indices],
        "detrended_residual_percent": analysis.residual_pct[indices],
    })
    csv_path = DATA_ROOT / "pulsating_star_tess_light_curve.csv"
    json_path = DATA_ROOT / "pulsating_star_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "summary": summary,
        "metadata": asdict(metadata),
        "folded_curve": {
            "phase": analysis.folded_phase.tolist(),
            "relative_flux_percent": analysis.folded_flux_pct.tolist(),
        },
    }, indent=2), encoding="utf-8")
    if analysis.pixel_image is not None:
        np.save(DATA_ROOT / "tess_pixel_stamp.npy", analysis.pixel_image)
    if analysis.aperture_mask is not None:
        np.save(DATA_ROOT / "tess_aperture_mask.npy", analysis.aperture_mask)
    return csv_path, json_path


# =============================================================================
# Plot helpers
# =============================================================================


def map_series(
    x_values: np.ndarray,
    y_values: np.ndarray,
    box: Tuple[int, int, int, int],
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    x0, y0, x1, y1 = box
    if x_range is None:
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    else:
        xmin, xmax = x_range
    if y_range is None:
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    else:
        ymin, ymax = y_range
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    px = x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)
    py = y1 - (y - ymin) / (ymax - ymin) * (y1 - y0)
    return np.vstack([px, py]).T


def draw_plot_grid(img: Image.Image, box: Tuple[int, int, int, int], x_ticks: int = 5, y_ticks: int = 5):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    for i in range(x_ticks + 1):
        x = lerp(x0, x1, i / x_ticks)
        d.line((x, y0, x, y1), fill=(90, 150, 180, 35), width=1)
    for i in range(y_ticks + 1):
        y = lerp(y0, y1, i / y_ticks)
        d.line((x0, y, x1, y), fill=(90, 150, 180, 35), width=1)
    d.rectangle((x0, y0, x1, y1), outline=(90, 180, 210, 80), width=1)


# =============================================================================
# Scene renderer
# =============================================================================

class PulsatingStarScene:
    def __init__(self, metadata: TESSMetadata, analysis: LightCurveAnalysis, summary: Dict):
        self.metadata = metadata
        self.analysis = analysis
        self.summary = summary
        self.stars = self._make_stars(CONFIG["stars"], seed=6271)
        self.render_indices = select_indices(analysis.point_count, CONFIG["max_render_points"])
        self.render_time = analysis.time_day[self.render_indices]
        self.render_flux = analysis.rel_flux_pct[self.render_indices]
        low, high = np.nanpercentile(self.render_flux, [1, 99])
        margin = max(0.1, (high - low) * 0.12)
        self.full_y_range = (float(low - margin), float(high + margin))
        self.curve_box = (int(W * 0.07), int(H * 0.22), int(W * 0.93), int(H * 0.69))
        self.zoom_box_top = (int(W * 0.07), int(H * 0.20), int(W * 0.93), int(H * 0.44))
        self.zoom_box_bottom = (int(W * 0.07), int(H * 0.49), int(W * 0.58), int(H * 0.72))
        self.pixel_box = (int(W * 0.63), int(H * 0.49), int(W * 0.93), int(H * 0.72))

    @staticmethod
    def _make_stars(n: int, seed: int):
        rng = np.random.default_rng(seed)
        return [
            (float(rng.uniform(0, W)), float(rng.uniform(0, H)), float(rng.uniform(.4, 2.0) * SCALE),
             int(rng.integers(25, 145)), float(rng.uniform(0, math.tau)))
            for _ in range(n)
        ]

    def background(self, t: float) -> Image.Image:
        img = Image.new("RGBA", SIZE, COLORS["bg"] + (255,))
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        clouds = [
            (W * 0.18, H * 0.31, (16, 70, 125)),
            (W * 0.72, H * 0.25, (90, 35, 110)),
            (W * 0.48, H * 0.76, (15, 65, 92)),
        ]
        for cx, cy, color in clouds:
            for radius, alpha in [(W * 0.46, 13), (W * 0.30, 23), (W * 0.18, 32)]:
                gd.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=color + (alpha,))
        glow = glow.filter(ImageFilter.GaussianBlur(65 if not QUICK_MODE else 32))
        img.alpha_composite(glow)

        d = ImageDraw.Draw(img)
        for x, y, r, a, phase in self.stars:
            alpha = int(a * (0.72 + 0.28 * math.sin(1.6 * t + phase)))
            d.ellipse((x-r, y-r, x+r, y+r), fill=(214, 228, 255, alpha))
        return img

    def draw_title(self, img: Image.Image, t: float):
        alpha = int(255 * smoothstep((t - 0.15) / 0.8) * (1 - smoothstep((t - (6.4 if not QUICK_MODE else 1.55)) / 0.8)))
        if alpha > 4:
            draw_text(img, CONFIG["title"], (56 if not QUICK_MODE else 28, 90 if not QUICK_MODE else 45),
                      size=42 if not QUICK_MODE else 19, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(img, CONFIG["subtitle"], (58 if not QUICK_MODE else 30, 151 if not QUICK_MODE else 76),
                      size=22 if not QUICK_MODE else 10, fill=COLORS["cyan"] + (min(alpha, 230),), bold=True)
        shot_titles = {
            "intro": "A STAR TURNED INTO A TIME SERIES",
            "curve": "ONE TESS SECTOR OF BRIGHTNESS DATA",
            "events": "INSIDE ONE STELLAR PULSE",
            "period": "THE PHASE-FOLDED COSMIC HEARTBEAT",
            "stats": "THE HEARTBEAT BY THE NUMBERS",
            "outro": "ONE STAR • THOUSANDS OF PULSES",
        }
        if t > (5.0 if not QUICK_MODE else 1.25):
            draw_text(img, shot_titles[get_shot(t)["name"]], (56 if not QUICK_MODE else 28, 61 if not QUICK_MODE else 30),
                      size=19 if not QUICK_MODE else 9, fill=COLORS["muted"] + (210,), bold=True, stroke=1)

    def draw_source_hud(self, img: Image.Image):
        status = "OFFLINE FIXTURE" if self.metadata.offline_fixture else ("CACHE" if self.metadata.data_status == "cache" else "MAST")
        label = f"TESS DATA // {status}"
        draw_text(img, label, (W - (48 if not QUICK_MODE else 24), 72 if not QUICK_MODE else 36),
                  size=17 if not QUICK_MODE else 8, fill=COLORS["cyan"] + (220,), bold=True, anchor="ra", stroke=1)
        details = f"SECTOR {self.metadata.sector} • {self.analysis.cadence_s:.0f}s"
        draw_text(img, details, (W - (48 if not QUICK_MODE else 24), 102 if not QUICK_MODE else 51),
                  size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (195,), anchor="ra", stroke=1)

    def draw_caption(self, img: Image.Image, t: float):
        caption = caption_at(t)
        if not caption:
            return
        y0 = H - (244 if not QUICK_MODE else 124)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((44 if not QUICK_MODE else 22, y0, W-(44 if not QUICK_MODE else 22), y0+(124 if not QUICK_MODE else 66)),
                             radius=24 if not QUICK_MODE else 12, fill=(2, 6, 14, 172), outline=(80, 185, 220, 65), width=1)
        img.alpha_composite(overlay)
        draw_wrapped_text(img, caption, (68 if not QUICK_MODE else 34, y0+(28 if not QUICK_MODE else 14)),
                          W-(136 if not QUICK_MODE else 68), size=29 if not QUICK_MODE else 14,
                          fill=COLORS["white"] + (245,))

    def draw_hud_noise(self, img: Image.Image, t: float):
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        offset = int((t * 39) % 7)
        for y in range(offset, H, 7):
            od.line((0, y, W, y), fill=(120, 200, 240, 10), width=1)
        scan_y = int((t * 165) % (H + 220)) - 110
        od.rectangle((0, scan_y, W, scan_y + (48 if not QUICK_MODE else 24)), fill=(90, 210, 240, 7))
        img.alpha_composite(overlay)

    def draw_intro(self, img: Image.Image, t: float):
        cx, cy = W * 0.5, H * 0.40
        period = self.analysis.dominant_period_days
        shot_duration = max(1e-6, SHOT_PLAN[0]["end"] - SHOT_PLAN[0]["start"])
        pulse_phase = ((t - SHOT_PLAN[0]["start"]) / shot_duration * 3.0) % 1.0

        if len(self.analysis.folded_phase) > 3:
            order = np.argsort(self.analysis.folded_phase)
            ph = self.analysis.folded_phase[order]
            fl = self.analysis.folded_flux_pct[order]
            extended_ph = np.concatenate([ph - 1.0, ph, ph + 1.0])
            extended_fl = np.tile(fl, 3)
            brightness = float(np.interp(pulse_phase, extended_ph, extended_fl))
            low, high = np.nanpercentile(fl, [2, 98])
            normalized = clamp((brightness - low) / max(high - low, 1e-6))
        else:
            data_index = int(pulse_phase * (self.analysis.point_count - 1))
            brightness = float(self.analysis.rel_flux_pct[data_index])
            normalized = clamp(0.5 + brightness / max(1.0, self.analysis.variability_range_pct * 1.5))

        radius = (78 + 34 * normalized) * SCALE
        warmth = normalized
        star_color = (
            int(lerp(255, 255, warmth)),
            int(lerp(208, 244, warmth)),
            int(lerp(170, 220, warmth)),
        )

        star_glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(star_glow)
        for mul, alpha in [(3.6, 18), (2.5, 32), (1.7, 62)]:
            rr = radius * mul
            gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=star_color + (alpha,))
        star_glow = star_glow.filter(ImageFilter.GaussianBlur(34 if not QUICK_MODE else 17))
        img.alpha_composite(star_glow)
        d = ImageDraw.Draw(img)
        d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=star_color + (255,), outline=(255, 255, 255, 225), width=2)

        # A stylized heartbeat trace synchronized to the phase-folded data.
        trace_box = (int(W * 0.10), int(H * 0.58), int(W * 0.90), int(H * 0.69))
        draw_plot_grid(img, trace_box, 6, 2)
        if len(self.analysis.folded_phase) > 3:
            x_phase = np.concatenate([self.analysis.folded_phase, self.analysis.folded_phase + 1.0])
            y_flux = np.concatenate([self.analysis.folded_flux_pct, self.analysis.folded_flux_pct])
            lo, hi = np.nanpercentile(y_flux, [2, 98])
            margin = max(0.1, (hi - lo) * 0.18)
            mapped = map_series(x_phase, y_flux, trace_box, x_range=(0, 2), y_range=(lo-margin, hi+margin))
            d.line([tuple(p) for p in mapped], fill=COLORS["cyan"] + (225,), width=max(2, int(3*SCALE)))
            play_x = lerp(trace_box[0], trace_box[2], pulse_phase / 2.0)
            d.line((play_x, trace_box[1], play_x, trace_box[3]), fill=COLORS["gold"] + (210,), width=2)

        draw_text(img, clip_text(self.metadata.object_name, 32).upper(), (W // 2, int(H * 0.73)),
                  size=30 if not QUICK_MODE else 14, fill=COLORS["white"] + (240,), bold=True, anchor="ma", stroke=1)
        period_text = f"PULSE PERIOD {period:.4f} DAYS" if np.isfinite(period) else "PULSE PERIOD NOT RESOLVED"
        draw_text(img, period_text, (W // 2, int(H * 0.78)),
                  size=22 if not QUICK_MODE else 10, fill=COLORS["gold"] + (235,), bold=True, anchor="ma", stroke=1)
        draw_text(img, f"PHASE {pulse_phase:.2f} • BRIGHTNESS {brightness:+.2f}%", (W // 2, int(H * 0.82)),
                  size=18 if not QUICK_MODE else 8, fill=COLORS["muted"] + (220,), bold=True, anchor="ma", stroke=1)

    def draw_full_curve(self, img: Image.Image, t: float):
        x0, y0, x1, y1 = self.curve_box
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0, y0, x1, y1), radius=28 if not QUICK_MODE else 14,
                             fill=(2, 6, 14, 178), outline=(88, 185, 220, 78), width=2)
        img.alpha_composite(overlay)
        plot_box = (x0 + 38, y0 + 58, x1 - 22, y1 - 42)
        draw_plot_grid(img, plot_box)
        draw_text(img, "RELATIVE FLUX (%)", (x0 + 18, y0 + 18), size=18 if not QUICK_MODE else 8,
                  fill=COLORS["cyan"] + (220,), bold=True, stroke=1)
        draw_text(img, f"{self.analysis.span_days:.2f} DAYS", (x1 - 18, y0 + 18), size=17 if not QUICK_MODE else 8,
                  fill=COLORS["muted"] + (220,), bold=True, anchor="ra", stroke=1)

        points = map_series(self.render_time, self.render_flux, plot_box, y_range=self.full_y_range)
        reveal = smoothstep((t - SHOT_PLAN[1]["start"]) / max(1e-6, SHOT_PLAN[1]["end"] - SHOT_PLAN[1]["start"]))
        count = max(2, int(len(points) * reveal))
        d = ImageDraw.Draw(img)
        d.line([tuple(p) for p in points[:count]], fill=COLORS["cyan"] + (220,), width=max(1, int(2 * SCALE)))

        scan_x = lerp(plot_box[0], plot_box[2], reveal)
        d.line((scan_x, plot_box[1], scan_x, plot_box[3]), fill=COLORS["gold"] + (160,), width=2)

        for idx, label, color in [
            (self.analysis.strongest_brightening_index, "BRIGHTEST", COLORS["gold"]),
            (self.analysis.strongest_dimming_index, "FAINTEST", COLORS["violet"]),
        ]:
            day = self.analysis.time_day[idx]
            value = self.analysis.rel_flux_pct[idx]
            if day <= self.render_time[min(count - 1, len(self.render_time) - 1)]:
                p = map_series(np.array([day]), np.array([value]), plot_box,
                               x_range=(self.render_time.min(), self.render_time.max()), y_range=self.full_y_range)[0]
                r = 7 * SCALE
                d.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r), fill=color + (240,), outline=(255, 255, 255, 200), width=1)
                draw_text(img, label, (int(p[0]), int(p[1] - 18 * SCALE)), size=14 if not QUICK_MODE else 7,
                          fill=color + (235,), bold=True, anchor="ma", stroke=1)

        draw_text(img, "TIME SINCE SECTOR START →", (plot_box[2], plot_box[3] + 18), size=14 if not QUICK_MODE else 7,
                  fill=COLORS["muted"] + (210,), bold=True, anchor="ra", stroke=1)

    def event_window(self, center_index: int, half_width_days: float) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], Tuple[float, float]]:
        center_time = self.analysis.time_day[center_index]
        mask = np.abs(self.analysis.time_day - center_time) <= half_width_days
        x = self.analysis.time_day[mask]
        y = self.analysis.residual_pct[mask]
        if len(x) < 10:
            lo = max(0, center_index - 50)
            hi = min(self.analysis.point_count, center_index + 51)
            x = self.analysis.time_day[lo:hi]
            y = self.analysis.residual_pct[lo:hi]
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = np.nanpercentile(y, [1, 99])
        margin = max(0.05, (ymax - ymin) * 0.18)
        return x, y, (xmin, xmax), (float(ymin - margin), float(ymax + margin))

    def draw_event_panel(self, img: Image.Image, box, center_index: int, title: str, color: Tuple[int, int, int], half_width_days: float):
        x0, y0, x1, y1 = box
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0, y0, x1, y1), radius=24 if not QUICK_MODE else 12,
                             fill=(3, 8, 17, 184), outline=color + (85,), width=2)
        img.alpha_composite(overlay)
        plot_box = (x0 + 34, y0 + 48, x1 - 18, y1 - 30)
        draw_plot_grid(img, plot_box, 4, 4)
        x, y, xr, yr = self.event_window(center_index, half_width_days)
        mapped = map_series(x, y, plot_box, x_range=xr, y_range=yr)
        d = ImageDraw.Draw(img)
        d.line([tuple(p) for p in mapped], fill=color + (230,), width=max(1, int(2 * SCALE)))
        center_time = self.analysis.time_day[center_index]
        center_val = self.analysis.residual_pct[center_index]
        p = map_series(np.array([center_time]), np.array([center_val]), plot_box, x_range=xr, y_range=yr)[0]
        r = 7 * SCALE
        d.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r), fill=color + (245,), outline=(255, 255, 255, 200), width=1)
        draw_text(img, title, (x0 + 16, y0 + 14), size=18 if not QUICK_MODE else 8,
                  fill=color + (240,), bold=True, stroke=1)
        draw_text(img, f"{center_val:+.2f}%", (x1 - 16, y0 + 14), size=18 if not QUICK_MODE else 8,
                  fill=COLORS["white"] + (235,), bold=True, anchor="ra", stroke=1)

    def draw_pixel_stamp(self, img: Image.Image, box):
        x0, y0, x1, y1 = box
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0, y0, x1, y1), radius=24 if not QUICK_MODE else 12,
                             fill=(3, 8, 17, 184), outline=COLORS["cyan"] + (85,), width=2)
        img.alpha_composite(overlay)
        draw_text(img, "TESS PIXEL STAMP", (x0 + 12, y0 + 12), size=15 if not QUICK_MODE else 7,
                  fill=COLORS["cyan"] + (230,), bold=True, stroke=1)
        image = self.analysis.pixel_image
        if image is None or image.ndim != 2:
            draw_text(img, "NOT AVAILABLE", ((x0+x1)//2, (y0+y1)//2), size=17 if not QUICK_MODE else 8,
                      fill=COLORS["muted"] + (220,), bold=True, anchor="ma", stroke=1)
            return
        finite = image[np.isfinite(image)]
        if len(finite) == 0:
            return
        lo, hi = np.nanpercentile(finite, [5, 99.5])
        norm = np.clip((image - lo) / max(hi - lo, 1e-8), 0, 1)
        norm = np.sqrt(norm)
        rows, cols = image.shape
        pad = 12
        px0, py0, px1, py1 = x0 + pad, y0 + 42, x1 - pad, y1 - pad
        cell_w = (px1 - px0) / cols
        cell_h = (py1 - py0) / rows
        d = ImageDraw.Draw(img)
        mask = self.analysis.aperture_mask
        for r in range(rows):
            for c in range(cols):
                v = norm[r, c]
                color = (
                    int(lerp(12, 255, v)),
                    int(lerp(32, 220, v)),
                    int(lerp(70, 145, v)),
                    235,
                )
                xx0 = px0 + c * cell_w
                yy0 = py0 + r * cell_h
                xx1 = px0 + (c + 1) * cell_w
                yy1 = py0 + (r + 1) * cell_h
                d.rectangle((xx0, yy0, xx1, yy1), fill=color, outline=(5, 10, 18, 180), width=1)
                if mask is not None and mask.shape == image.shape and bool(mask[r, c]):
                    d.rectangle((xx0+1, yy0+1, xx1-1, yy1-1), outline=COLORS["green"] + (235,), width=max(1, int(2*SCALE)))

    def draw_events(self, img: Image.Image):
        self.draw_event_panel(img, self.zoom_box_top, self.analysis.strongest_brightening_index,
                              "BRIGHTEST PART OF THE PULSE", COLORS["gold"], half_width_days=max(0.08, self.analysis.dominant_period_days * 0.55 if np.isfinite(self.analysis.dominant_period_days) else 0.22))
        self.draw_event_panel(img, self.zoom_box_bottom, self.analysis.strongest_dimming_index,
                              "FAINTEST PART OF THE PULSE", COLORS["violet"], half_width_days=max(0.08, self.analysis.dominant_period_days * 0.55 if np.isfinite(self.analysis.dominant_period_days) else 0.30))
        self.draw_pixel_stamp(img, self.pixel_box)
        draw_wrapped_text(img, "The two panels center on the brightest and faintest parts of the repeating pulse. The pixel stamp shows where TESS collected the light.",
                          (int(W * 0.08), int(H * 0.76)), int(W * 0.84), size=18 if not QUICK_MODE else 8,
                          fill=COLORS["muted"] + (225,))

    def draw_period(self, img: Image.Image):
        x0, y0, x1, y1 = int(W * 0.08), int(H * 0.22), int(W * 0.92), int(H * 0.70)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0, y0, x1, y1), radius=28 if not QUICK_MODE else 14,
                             fill=(2, 6, 14, 178), outline=(88, 185, 220, 78), width=2)
        img.alpha_composite(overlay)
        period = self.analysis.dominant_period_days
        draw_text(img, "MEASURED PULSE PERIOD", (x0 + 18, y0 + 18), size=18 if not QUICK_MODE else 8,
                  fill=COLORS["cyan"] + (220,), bold=True, stroke=1)
        draw_text(img, f"{period:.3f} DAYS" if np.isfinite(period) else "NOT RESOLVED", (x1 - 18, y0 + 18),
                  size=25 if not QUICK_MODE else 11, fill=COLORS["gold"] + (240,), bold=True, anchor="ra", stroke=1)

        plot_box = (x0 + 44, y0 + 92, x1 - 28, y1 - 46)
        draw_plot_grid(img, plot_box, 4, 5)
        phase = self.analysis.folded_phase
        flux = self.analysis.folded_flux_pct
        if len(phase) > 2:
            low, high = np.nanpercentile(flux, [2, 98])
            margin = max(0.1, (high - low) * 0.2)
            mapped = map_series(phase, flux, plot_box, x_range=(0, 1), y_range=(low-margin, high+margin))
            d = ImageDraw.Draw(img)
            d.line([tuple(p) for p in mapped], fill=COLORS["green"] + (230,), width=max(2, int(3*SCALE)))
            for p in mapped[::max(1, len(mapped)//25)]:
                r = 3 * SCALE
                d.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r), fill=COLORS["green"] + (225,))
        draw_text(img, "PHASE 0", (plot_box[0], plot_box[3] + 18), size=14 if not QUICK_MODE else 7,
                  fill=COLORS["muted"] + (210,), stroke=1)
        draw_text(img, "PHASE 1", (plot_box[2], plot_box[3] + 18), size=14 if not QUICK_MODE else 7,
                  fill=COLORS["muted"] + (210,), anchor="ra", stroke=1)
        draw_wrapped_text(img, "The light curve is folded on the strongest measured period. Stacking many cycles reveals the repeating pulse shape more clearly.",
                          (x0 + 18, y1 + 22), x1 - x0 - 36, size=18 if not QUICK_MODE else 8,
                          fill=COLORS["muted"] + (225,))

    def stat_line(self, img: Image.Image, left: str, right: str, x: int, y: int, width: int, color: Tuple[int, int, int]):
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x, y, x + width, y + int(72 * SCALE)), radius=22 if not QUICK_MODE else 10,
                             fill=(3, 8, 17, 185), outline=color + (80,), width=2)
        img.alpha_composite(overlay)
        draw_text(img, left, (x + 18, y + 18), size=21 if not QUICK_MODE else 10,
                  fill=color + (240,), bold=True, stroke=1)
        draw_text(img, right, (x + width - 18, y + 18), size=21 if not QUICK_MODE else 10,
                  fill=COLORS["white"] + (235,), bold=True, anchor="ra", stroke=1)

    def draw_stats(self, img: Image.Image):
        x = int(W * 0.08)
        y = int(H * 0.21)
        width = int(W * 0.84)
        gap = int(16 * SCALE)
        values = [
            ("Target", clip_text(self.metadata.object_name, 24), COLORS["cyan"]),
            ("TESS sector", str(self.metadata.sector), COLORS["green"]),
            ("Cadence", f"{self.analysis.cadence_s:.1f} s", COLORS["gold"]),
            ("Observation span", f"{self.analysis.span_days:.2f} d", COLORS["violet"]),
            ("Valid measurements", f"{self.analysis.point_count:,}", COLORS["orange"]),
            ("Pulse period", f"{self.analysis.dominant_period_days:.4f} d" if np.isfinite(self.analysis.dominant_period_days) else "n/a", COLORS["red"]),
            ("90% pulse range", f"{self.analysis.variability_range_pct:.2f}%", COLORS["muted"]),
        ]
        row_h = int(76 * SCALE)
        for idx, (left, right, color) in enumerate(values):
            self.stat_line(img, left, right, x, y + idx * (row_h + gap), width, color)

    def draw_outro(self, img: Image.Image):
        x0, y0, x1, y1 = int(W * 0.08), int(H * 0.25), int(W * 0.92), int(H * 0.65)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0, y0, x1, y1), radius=28 if not QUICK_MODE else 14,
                             fill=(2, 6, 14, 178), outline=(88, 185, 220, 78), width=2)
        img.alpha_composite(overlay)
        draw_text(img, clip_text(self.metadata.object_name, 32).upper(), (W // 2, y0 + 64),
                  size=35 if not QUICK_MODE else 16, fill=COLORS["gold"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(img, f"{self.analysis.point_count:,} MEASUREMENTS • {self.analysis.span_days / self.analysis.dominant_period_days:.1f} PULSES" if np.isfinite(self.analysis.dominant_period_days) else f"{self.analysis.point_count:,} BRIGHTNESS MEASUREMENTS", (W // 2, y0 + 125),
                  size=24 if not QUICK_MODE else 11, fill=COLORS["cyan"] + (230,), bold=True, anchor="ma", stroke=1)
        draw_text(img, f"SECTOR {self.metadata.sector} • {self.analysis.span_days:.2f} DAYS • {self.analysis.cadence_s:.0f}s CADENCE", (W // 2, y0 + 170),
                  size=18 if not QUICK_MODE else 8, fill=COLORS["muted"] + (220,), bold=True, anchor="ma", stroke=1)
        status_text = "REAL TESS LIGHT CURVE FROM MAST" if not self.metadata.offline_fixture else "OFFLINE LAYOUT FIXTURE • NOT REAL PHOTOMETRY"
        draw_text(img, status_text, (W // 2, y0 + 226), size=19 if not QUICK_MODE else 9,
                  fill=(COLORS["green"] if not self.metadata.offline_fixture else COLORS["red"]) + (235,), bold=True, anchor="ma", stroke=1)
        draw_wrapped_text(img, "Run the script on another pulsating star or TESS sector and a different cosmic heartbeat will appear.",
                          (x0 + 28, y0 + 275), x1 - x0 - 56, size=20 if not QUICK_MODE else 9,
                          fill=COLORS["white"] + (230,), bold=True)

    def render_frame(self, t: float) -> np.ndarray:
        img = self.background(t)
        self.draw_title(img, t)
        self.draw_source_hud(img)
        shot = get_shot(t)["name"]
        if shot == "intro":
            self.draw_intro(img, t)
        elif shot == "curve":
            self.draw_full_curve(img, t)
        elif shot == "events":
            self.draw_events(img)
        elif shot == "period":
            self.draw_period(img)
        elif shot == "stats":
            self.draw_stats(img)
        else:
            self.draw_outro(img)

        self.draw_caption(img, t)
        self.draw_hud_noise(img, t)

        arr = np.array(img.convert("RGB"))
        graded = Image.fromarray(arr)
        graded = ImageEnhance.Contrast(graded).enhance(1.08)
        graded = ImageEnhance.Color(graded).enhance(1.06)
        arr = np.array(graded)
        arr = np.clip(arr.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / 0.9)
        fade_out = 1 - smoothstep((t - (CONFIG["duration_s"] - 1.1)) / 1.0)
        return np.clip(arr.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)


# =============================================================================
# Output
# =============================================================================


def render_video(scene: PulsatingStarScene) -> Path:
    raw_path = OUTPUT_ROOT / f"{CONFIG['basename']}_raw.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['basename']}_final.mp4"
    write_srt(OUTPUT_ROOT / f"{CONFIG['basename']}.srt")
    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    with iio.get_writer(raw_path, fps=CONFIG["fps"], codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None) as writer:
        for frame_index in tqdm(range(frame_count), desc="Rendering pulsating star short"):
            writer.append_data(scene.render_frame(frame_index / CONFIG["fps"]))
    shutil.copyfile(raw_path, final_path)
    return final_path


def make_contact_sheet(paths: Sequence[Path], out_path: Path):
    thumbs = []
    for path in paths[:6]:
        image = Image.open(path).convert("RGB").resize((270, 480))
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 120, 38), fill=(0, 0, 0))
        draw.text((18, 13), path.stem.replace("preview_", ""), fill=(255, 255, 255))
        thumbs.append(image)
    sheet = Image.new("RGB", (600, 1520), (8, 11, 18))
    for index, thumb in enumerate(thumbs):
        row, col = divmod(index, 2)
        sheet.paste(thumb, (20 + col * 290, 20 + row * 500))
    sheet.save(out_path, quality=92)


def main():
    print("Collecting pulsating-star light-curve data ...")
    metadata, analysis, summary = collect_data()
    csv_path, json_path = save_data(metadata, analysis, summary)
    print("Data:", csv_path.resolve())
    print("Summary:", json_path.resolve())

    scene = PulsatingStarScene(metadata, analysis, summary)
    preview_times = [1.0, min(11.0, CONFIG["duration_s"] * 0.22), min(25.0, CONFIG["duration_s"] * 0.43), min(37.0, CONFIG["duration_s"] * 0.65), min(47.0, CONFIG["duration_s"] * 0.83), CONFIG["duration_s"] - 1]
    preview_paths = []
    for t in tqdm(preview_times, desc="Preview frames"):
        path = PREVIEW_ROOT / f"preview_{int(t):02d}s.png"
        Image.fromarray(scene.render_frame(float(t))).save(path)
        preview_paths.append(path)
    make_contact_sheet(preview_paths, PREVIEW_ROOT / "a_pulsating_star_beating_like_a_cosmic_heart_contact_sheet.jpg")
    video_path = render_video(scene)
    print("Video:", video_path.resolve())
    print("Source status:", summary)


if __name__ == "__main__":
    main()
