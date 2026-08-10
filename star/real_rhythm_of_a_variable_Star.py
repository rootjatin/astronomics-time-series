from __future__ import annotations

"""
The Real Rhythm of a Variable Star — cinematic YouTube Short renderer

Creates a vertical 1080x1920 astronomy short that reveals the repeating pulse
hidden inside a variable-star light curve. The featured target is RR Lyrae, the
prototype of the RR Lyrae class.

Preferred live sources
----------------------
1. NASA TESS light-curve products downloaded through Lightkurve / MAST.
2. AAVSO VSX object metadata for the catalog period and classification.

The renderer tells this science story:
- A variable star is measured as a sequence of brightness samples: a light curve.
- RR Lyrae's raw time series can look complicated, especially across gaps and
  slow amplitude modulation.
- A period search tests many candidate cycle lengths.
- Folding every measurement by the best period reveals the repeating pulse.
- RR Lyrae brightens quickly and fades more slowly during a radial pulsation.
- The cycle is about 0.567 day, or roughly 13 hours 36 minutes.
- RR Lyrae also shows the Blazhko effect, so real cycles are not perfectly cloned.

Offline behavior
----------------
If TESS and/or VSX cannot be reached, the script uses a deterministic, clearly
labeled RR-Lyrae-like fixture for preview and layout validation. The fallback is
not observational data and must not be presented as such.

Recommended install
-------------------
    pip install numpy pandas matplotlib pillow imageio imageio-ffmpeg tqdm \
        requests astropy lightkurve

Quick preview render
--------------------
    VARIABLE_STAR_SHORT_QUICK=1 python the_real_rhythm_of_a_variable_star_short.py

Force offline fixture mode
--------------------------
    VARIABLE_STAR_SHORT_OFFLINE=1 python the_real_rhythm_of_a_variable_star_short.py

Primary references used when designing the script
--------------------------------------------------
- NASA TESS mission: https://science.nasa.gov/mission/tess/
- MAST TESS archive: https://archive.stsci.edu/missions-and-data/tess
- AAVSO VSX: https://vsx.aavso.org/
- AAVSO RR Lyrae overview: https://www.aavso.org/vsots_rrlyr
"""

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
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
    from astropy.timeseries import LombScargle
except Exception:
    LombScargle = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("VARIABLE_STAR_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("VARIABLE_STAR_SHORT_OFFLINE", "0") == "1"
OUTPUT_ROOT = Path("variable_star_rhythm_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
CACHE_DIR = DATA_ROOT / "mast_cache"
for directory in [OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "the_real_rhythm_of_a_variable_star",
    "title": "THE REAL RHYTHM OF A VARIABLE STAR",
    "subtitle": "RR Lyrae // raw light curve → period → folded pulse",
    "target_name": "RR Lyr",
    "vsx_ident": "RR Lyr",
    "fallback_period_days": 0.56686776,
    "period_min_days": 0.35,
    "period_max_days": 0.85,
    "period_trial_count": 1800 if QUICK_MODE else 5200,
    "max_analysis_points": 2600 if QUICK_MODE else 7000,
    "max_live_sectors": 1 if QUICK_MODE else 3,
    "background_stars": 280,
    "hud_noise": 48,
    "contrast": 1.08,
    "saturation": 1.07,
    "vignette": 0.25,
    "vsx_urls": [
        "https://vsx.aavso.org/index.php",
        "https://www.aavso.org/vsx/index.php",
    ],
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)

COLORS = {
    "cyan": (92, 225, 255),
    "gold": (255, 190, 88),
    "rose": (255, 105, 155),
    "violet": (178, 126, 255),
    "white": (244, 248, 253),
    "muted": (158, 203, 224),
    "dark": (3, 7, 15),
}

FULL_CAPTIONS = [
    (0.5, 6.8, "This star is not flickering at random. Its light carries a repeating clock."),
    (6.9, 15.5, "TESS measures the star again and again, building a light curve from thousands of brightness samples."),
    (15.6, 25.0, "In ordinary time, gaps, noise, and changing pulse strength can hide the pattern."),
    (25.1, 35.0, "A period search tests thousands of possible cycle lengths and asks which one lines the data up best."),
    (35.1, 46.0, "Fold the observations by that period, and the scattered measurements collapse into one repeating pulse."),
    (46.1, 57.3, "RR Lyrae expands and contracts radially—brightening fast, fading slowly, and never repeating with machine-perfect precision."),
]
if QUICK_MODE:
    _scale = CONFIG["duration_s"] / 58.0
    CAPTIONS = [(a * _scale, b * _scale, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 8.0 if not QUICK_MODE else 2.0},
    {"name": "raw_curve", "start": 8.0 if not QUICK_MODE else 2.0, "end": 21.0 if not QUICK_MODE else 4.6},
    {"name": "period_search", "start": 21.0 if not QUICK_MODE else 4.6, "end": 33.5 if not QUICK_MODE else 7.2},
    {"name": "phase_fold", "start": 33.5 if not QUICK_MODE else 7.2, "end": 47.0 if not QUICK_MODE else 9.8},
    {"name": "pulsation", "start": 47.0 if not QUICK_MODE else 9.8, "end": CONFIG["duration_s"]},
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
    draw = ImageDraw.Draw(image)
    draw.text(
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
    image = ImageEnhance.Contrast(image).enhance(CONFIG["contrast"])
    image = ImageEnhance.Color(image).enhance(CONFIG["saturation"])
    return np.array(image)


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
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


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        return number if np.isfinite(number) else None
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(value))
    if not match:
        return None
    try:
        number = float(match.group(0))
    except Exception:
        return None
    return number if np.isfinite(number) else None


def recursive_find(data: Any, wanted_keys: Sequence[str]) -> Any:
    wanted = {key.lower().replace("_", "").replace(" ", "") for key in wanted_keys}
    if isinstance(data, dict):
        for key, value in data.items():
            normalized = str(key).lower().replace("_", "").replace(" ", "")
            if normalized in wanted and value not in (None, "", "--"):
                return value
        for value in data.values():
            found = recursive_find(value, wanted_keys)
            if found not in (None, "", "--"):
                return found
    elif isinstance(data, list):
        for value in data:
            found = recursive_find(value, wanted_keys)
            if found not in (None, "", "--"):
                return found
    return None


VIGNETTE = make_vignette(OUT_W, OUT_H, CONFIG["vignette"])


# -----------------------------------------------------------------------------
# Live metadata and light-curve loading
# -----------------------------------------------------------------------------

def fetch_vsx_metadata() -> Tuple[Dict[str, Any], str]:
    if requests is None:
        raise RuntimeError("requests is not installed")

    errors = []
    for base_url in CONFIG["vsx_urls"]:
        try:
            response = requests.get(
                base_url,
                params={"view": "api.object", "ident": CONFIG["vsx_ident"], "format": "json"},
                timeout=35,
                headers={"User-Agent": "variable-star-short/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
            period = safe_float(recursive_find(payload, ["period", "period_days", "Period"] ))
            epoch = safe_float(recursive_find(payload, ["epoch", "Epoch"] ))
            var_type = recursive_find(payload, ["vartype", "var_type", "type", "variability_type"])
            name = recursive_find(payload, ["name", "display_name", "ident", "designation"])
            return {
                "raw": payload,
                "catalog_period_days": period,
                "epoch_jd": epoch,
                "variable_type": str(var_type) if var_type else "RRAB / pulsating variable",
                "catalog_name": str(name) if name else CONFIG["target_name"],
            }, "aavso_vsx"
        except Exception as exc:
            errors.append(f"{base_url}: {exc}")
    raise RuntimeError("; ".join(errors))


def fetch_tess_lightcurve() -> Tuple[pd.DataFrame, str]:
    if lk is None:
        raise RuntimeError("lightkurve is not installed")

    search = lk.search_lightcurve(CONFIG["target_name"], mission="TESS")
    if len(search) == 0:
        raise RuntimeError(f"No TESS light curves found for {CONFIG['target_name']}")

    selected = search[: int(CONFIG["max_live_sectors"])]
    collection = selected.download_all(download_dir=str(CACHE_DIR))
    if collection is None or len(collection) == 0:
        raise RuntimeError("TESS search returned products but none downloaded")

    pieces = []
    for lightcurve in collection:
        try:
            cleaned = lightcurve.remove_nans().normalize()
            time_values = np.asarray(cleaned.time.value, dtype=float)
            flux_values = np.asarray(cleaned.flux.value, dtype=float)
            mask = np.isfinite(time_values) & np.isfinite(flux_values)
            time_values = time_values[mask]
            flux_values = flux_values[mask]
            if len(time_values) < 50:
                continue
            pieces.append(pd.DataFrame({"time_native": time_values, "relative_flux": flux_values}))
        except Exception:
            continue

    if not pieces:
        raise RuntimeError("Downloaded TESS products contained no usable flux rows")

    df = pd.concat(pieces, ignore_index=True).dropna().sort_values("time_native")
    df = df.drop_duplicates("time_native").reset_index(drop=True)
    df["time_days"] = df["time_native"] - float(df["time_native"].min())

    # Robustly remove only extreme instrumental outliers. Do not flatten the
    # light curve because flattening can erase the stellar pulsation itself.
    median = float(np.nanmedian(df["relative_flux"]))
    mad = float(np.nanmedian(np.abs(df["relative_flux"] - median)))
    if mad > 0:
        robust_sigma = 1.4826 * mad
        df = df[np.abs(df["relative_flux"] - median) < 9.0 * robust_sigma].copy()

    df["brightness_percent"] = (df["relative_flux"] - 1.0) * 100.0
    df["data_source"] = "tess_lightkurve_mast"
    return df.reset_index(drop=True), "tess_lightkurve_mast"


# -----------------------------------------------------------------------------
# Offline fallback model
# -----------------------------------------------------------------------------

def rr_lyrae_profile(phase: np.ndarray) -> np.ndarray:
    """Asymmetric RRab-like brightness profile, normalized around zero."""
    phase = np.mod(np.asarray(phase, dtype=float), 1.0)

    # Fourier series chosen to produce a fast rise and slower decline. It is a
    # visual/analysis fixture, not a fitted physical model of RR Lyrae.
    profile = (
        0.70 * np.sin(2.0 * math.pi * (phase - 0.78))
        + 0.31 * np.sin(4.0 * math.pi * (phase - 0.82))
        + 0.18 * np.sin(6.0 * math.pi * (phase - 0.84))
        + 0.08 * np.sin(8.0 * math.pi * (phase - 0.86))
    )
    profile -= float(np.mean(profile))
    peak = max(float(np.max(np.abs(profile))), 1e-9)
    return profile / peak


def fallback_rr_lyrae_lightcurve() -> Tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(4417)
    period = float(CONFIG["fallback_period_days"])

    # Ten-minute cadence across 28 days, with realistic gaps.
    times = np.arange(0.0, 28.0, 10.0 / (24.0 * 60.0))
    keep = np.ones_like(times, dtype=bool)
    for start, length in [(3.2, 0.21), (8.1, 0.48), (14.5, 0.17), (21.0, 0.72), (26.1, 0.22)]:
        keep &= ~((times >= start) & (times <= start + length))
    times = times[keep]

    phase = np.mod(times / period, 1.0)
    pulse = rr_lyrae_profile(phase)

    # The prototype RR Lyrae is a Blazhko star. Add slow amplitude/phase
    # modulation so folded live-looking cycles are not perfectly identical.
    blazhko = 1.0 + 0.15 * np.sin(2.0 * math.pi * times / 39.0 + 0.6)
    phase_wobble = 0.015 * np.sin(2.0 * math.pi * times / 39.0)
    pulse_wobbled = rr_lyrae_profile(phase + phase_wobble)
    flux = 1.0 + 0.115 * blazhko * pulse_wobbled
    flux += 0.0025 * np.sin(2.0 * math.pi * times / 6.7)
    flux += rng.normal(0.0, 0.0045, size=len(times))

    df = pd.DataFrame({
        "time_native": times,
        "time_days": times,
        "relative_flux": flux,
        "brightness_percent": (flux - 1.0) * 100.0,
        "data_source": "offline_rr_lyrae_fixture",
    })
    return df, "offline_rr_lyrae_fixture"


def load_all_data() -> Tuple[pd.DataFrame, Dict[str, Any], str, List[str]]:
    notes: List[str] = []
    metadata: Dict[str, Any] = {
        "catalog_period_days": float(CONFIG["fallback_period_days"]),
        "epoch_jd": None,
        "variable_type": "RRAB / pulsating variable",
        "catalog_name": CONFIG["target_name"],
        "metadata_source": "built_in_reference",
    }

    if OFFLINE_MODE:
        notes.append("Offline mode requested with VARIABLE_STAR_SHORT_OFFLINE=1")
        df, source = fallback_rr_lyrae_lightcurve()
        return df, metadata, source, notes

    try:
        live_metadata, metadata_source = fetch_vsx_metadata()
        metadata.update(live_metadata)
        metadata["metadata_source"] = metadata_source
    except Exception as exc:
        notes.append(f"VSX metadata fallback: {exc}")

    try:
        df, source = fetch_tess_lightcurve()
    except Exception as exc:
        notes.append(f"TESS light-curve fallback: {exc}")
        df, source = fallback_rr_lyrae_lightcurve()

    return df, metadata, source, notes


# -----------------------------------------------------------------------------
# Period analysis and phase folding
# -----------------------------------------------------------------------------

def evenly_subsample(df: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(df) <= maximum:
        return df.copy().reset_index(drop=True)
    indices = np.linspace(0, len(df) - 1, maximum).astype(int)
    return df.iloc[indices].copy().reset_index(drop=True)


def phase_dispersion_score(times: np.ndarray, values: np.ndarray, period: float, bins: int = 42) -> float:
    phase = np.mod(times / period, 1.0)
    bin_index = np.minimum((phase * bins).astype(int), bins - 1)
    counts = np.bincount(bin_index, minlength=bins).astype(float)
    sums = np.bincount(bin_index, weights=values, minlength=bins)
    sums_sq = np.bincount(bin_index, weights=values * values, minlength=bins)
    valid = counts >= 3
    if valid.sum() < bins * 0.35:
        return float("inf")
    within = sums_sq[valid] - (sums[valid] * sums[valid] / counts[valid])
    total_variance = float(np.sum((values - np.mean(values)) ** 2))
    if total_variance <= 0:
        return float("inf")
    return float(np.sum(within) / total_variance)


def estimate_period(
    df: pd.DataFrame,
    catalog_period: Optional[float],
) -> Tuple[float, pd.DataFrame, str]:
    sample = evenly_subsample(df, int(CONFIG["max_analysis_points"]))
    times = sample["time_days"].to_numpy(float)
    values = sample["relative_flux"].to_numpy(float)
    values = values - float(np.mean(values))

    p_min = float(CONFIG["period_min_days"])
    p_max = float(CONFIG["period_max_days"])
    if catalog_period and p_min <= catalog_period <= p_max:
        p_min = max(p_min, catalog_period * 0.72)
        p_max = min(p_max, catalog_period * 1.28)

    periods = np.linspace(p_min, p_max, int(CONFIG["period_trial_count"]))
    method = "phase_dispersion_minimization"

    if LombScargle is not None:
        frequencies = 1.0 / periods
        try:
            ls_power = LombScargle(times, values).power(frequencies)
            # Lomb-Scargle can favor a harmonic for asymmetric RRab pulses.
            # Re-rank its strongest candidates using phase dispersion.
            order = np.argsort(ls_power)[::-1]
            candidates: List[float] = []
            for index in order[:30]:
                base = float(periods[index])
                for candidate in [base, base * 2.0, base / 2.0]:
                    if p_min <= candidate <= p_max:
                        candidates.append(candidate)
            if catalog_period and p_min <= catalog_period <= p_max:
                candidates.append(float(catalog_period))
            candidates = sorted(set(round(value, 8) for value in candidates))
            candidate_scores = [phase_dispersion_score(times, values, period) for period in candidates]
            best_period = float(candidates[int(np.argmin(candidate_scores))])

            # For the visible chart, combine normalized LS power with inverse
            # phase dispersion so the selected peak is intuitive.
            score_periods = periods[:: max(1, len(periods) // 900)]
            pdm = np.array([phase_dispersion_score(times, values, p) for p in score_periods])
            pdm_quality = 1.0 - (pdm - np.nanmin(pdm)) / max(np.nanmax(pdm) - np.nanmin(pdm), 1e-9)
            interp_power = np.interp(score_periods, periods, ls_power)
            interp_power = (interp_power - np.min(interp_power)) / max(np.max(interp_power) - np.min(interp_power), 1e-9)
            combined = 0.55 * interp_power + 0.45 * pdm_quality
            periodogram = pd.DataFrame({
                "period_days": score_periods,
                "search_power": combined,
                "lomb_scargle_power": interp_power,
                "phase_coherence": pdm_quality,
            })
            method = "lomb_scargle_plus_phase_dispersion"
            return best_period, periodogram, method
        except Exception:
            pass

    scores = np.array([phase_dispersion_score(times, values, period) for period in periods])
    finite = np.isfinite(scores)
    if not finite.any():
        fallback = float(catalog_period or CONFIG["fallback_period_days"])
        periodogram = pd.DataFrame({"period_days": periods, "search_power": np.zeros_like(periods)})
        return fallback, periodogram, method

    finite_scores = scores[finite]
    quality = np.zeros_like(scores)
    score_range = max(float(np.max(finite_scores) - np.min(finite_scores)), 1e-9)
    quality[finite] = 1.0 - (scores[finite] - np.min(finite_scores)) / score_range
    best_period = float(periods[int(np.nanargmax(quality))])
    periodogram = pd.DataFrame({"period_days": periods, "search_power": quality})
    return best_period, periodogram, method


def phase_fold(df: pd.DataFrame, period_days: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    folded = df.copy()
    folded["phase"] = np.mod(folded["time_days"].to_numpy(float) / period_days, 1.0)
    folded = folded.sort_values("phase").reset_index(drop=True)

    bin_count = 72
    bin_index = np.minimum((folded["phase"].to_numpy() * bin_count).astype(int), bin_count - 1)
    folded["phase_bin"] = bin_index
    profile = (
        folded.groupby("phase_bin")
        .agg(phase=("phase", "median"), relative_flux=("relative_flux", "median"), samples=("relative_flux", "size"))
        .reset_index(drop=True)
    )
    profile["brightness_percent"] = (profile["relative_flux"] - 1.0) * 100.0
    return folded, profile


def summarize(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    source: str,
    period_days: float,
    profile: pd.DataFrame,
    period_method: str,
) -> Dict[str, Any]:
    flux = df["relative_flux"].to_numpy(float)
    low, high = np.nanpercentile(flux, [2.0, 98.0])
    amplitude_percent = float((high - low) * 100.0)
    catalog_period = safe_float(metadata.get("catalog_period_days"))
    period_error_seconds = None
    if catalog_period:
        period_error_seconds = float(abs(period_days - catalog_period) * 86400.0)

    max_row = profile.iloc[int(np.argmax(profile["relative_flux"].to_numpy(float)))]
    return {
        "target": metadata.get("catalog_name", CONFIG["target_name"]),
        "variable_type": metadata.get("variable_type", "RRAB"),
        "lightcurve_source": source,
        "metadata_source": metadata.get("metadata_source"),
        "rows": int(len(df)),
        "time_span_days": float(df["time_days"].max() - df["time_days"].min()),
        "detected_period_days": float(period_days),
        "detected_period_hours": float(period_days * 24.0),
        "catalog_period_days": catalog_period,
        "period_error_seconds": period_error_seconds,
        "period_method": period_method,
        "robust_peak_to_peak_percent": amplitude_percent,
        "phase_of_maximum": float(max_row["phase"]),
        "approx_cycles_observed": float((df["time_days"].max() - df["time_days"].min()) / period_days),
    }


def save_data_products(
    df: pd.DataFrame,
    folded: pd.DataFrame,
    profile: pd.DataFrame,
    periodogram: pd.DataFrame,
    summary: Dict[str, Any],
    metadata: Dict[str, Any],
    notes: List[str],
) -> Tuple[Path, Path]:
    lightcurve_path = DATA_ROOT / "rr_lyrae_lightcurve.csv"
    folded_path = DATA_ROOT / "rr_lyrae_phase_folded.csv"
    profile_path = DATA_ROOT / "rr_lyrae_phase_profile.csv"
    periodogram_path = DATA_ROOT / "rr_lyrae_period_search.csv"
    summary_path = DATA_ROOT / "rr_lyrae_summary.json"

    df.to_csv(lightcurve_path, index=False)
    folded.to_csv(folded_path, index=False)
    profile.to_csv(profile_path, index=False)
    periodogram.to_csv(periodogram_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "metadata": {key: value for key, value in metadata.items() if key != "raw"},
                "notes": notes,
                "fallback_warning": "offline_rr_lyrae_fixture is a deterministic visual fixture, not observational data",
                "source_urls": {
                    "tess": "https://science.nasa.gov/mission/tess/",
                    "mast": "https://archive.stsci.edu/missions-and-data/tess",
                    "vsx": "https://vsx.aavso.org/",
                    "rr_lyrae_background": "https://www.aavso.org/vsots_rrlyr",
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return lightcurve_path, summary_path


def create_scientific_plots(
    df: pd.DataFrame,
    folded: pd.DataFrame,
    profile: pd.DataFrame,
    periodogram: pd.DataFrame,
    period_days: float,
):
    sample = evenly_subsample(df, 6000)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(sample["time_days"], sample["brightness_percent"], linewidth=0.7)
    ax.set_title("RR Lyrae brightness through time")
    ax.set_xlabel("Time since first sample (days)")
    ax.set_ylabel("Relative brightness (%)")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "raw_light_curve.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(periodogram["period_days"], periodogram["search_power"], linewidth=1.1)
    ax.axvline(period_days, linewidth=1.0)
    ax.set_title("Period search")
    ax.set_xlabel("Trial period (days)")
    ax.set_ylabel("Relative coherence")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "period_search.png", dpi=170)
    plt.close(fig)

    folded_sample = evenly_subsample(folded, 7000)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(folded_sample["phase"], folded_sample["brightness_percent"], s=4, alpha=0.25)
    ax.plot(profile["phase"], profile["brightness_percent"], linewidth=1.8)
    ax.set_title(f"Phase-folded pulse: period = {period_days:.6f} days")
    ax.set_xlabel("Pulsation phase")
    ax.set_ylabel("Relative brightness (%)")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "phase_folded_light_curve.png", dpi=170)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class VariableStarScene:
    def __init__(
        self,
        df: pd.DataFrame,
        folded: pd.DataFrame,
        profile: pd.DataFrame,
        periodogram: pd.DataFrame,
        summary: Dict[str, Any],
    ):
        self.df = df.copy().reset_index(drop=True)
        self.folded = folded.copy().reset_index(drop=True)
        self.profile = profile.copy().reset_index(drop=True)
        self.periodogram = periodogram.copy().reset_index(drop=True)
        self.summary = summary
        self.stars = self._make_stars(CONFIG["background_stars"], seed=35)
        self.hud = self._make_hud(CONFIG["hud_noise"], seed=91)

        self.raw_display = self._prepare_raw_display()
        self.folded_display = evenly_subsample(self.folded, 2300 if not QUICK_MODE else 800)
        self.period_display = evenly_subsample(self.periodogram, 900 if not QUICK_MODE else 420)

        flux_values = self.df["brightness_percent"].to_numpy(float)
        self.flux_low, self.flux_high = np.nanpercentile(flux_values, [1.0, 99.0])
        padding = max((self.flux_high - self.flux_low) * 0.12, 0.5)
        self.flux_low -= padding
        self.flux_high += padding

    @staticmethod
    def _make_stars(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.5, 2.0)),
                "a": float(rng.uniform(18, 100)),
                "phase": float(rng.uniform(0, 2.0 * math.pi)),
                "drift": float(rng.uniform(-7, 7)),
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
                "length": float(rng.uniform(10, 88)),
                "a": float(rng.uniform(8, 42)),
                "phase": float(rng.uniform(0, 2.0 * math.pi)),
            }
            for _ in range(count)
        ]

    def _prepare_raw_display(self) -> pd.DataFrame:
        span = float(self.df["time_days"].max() - self.df["time_days"].min())
        desired = min(span, 8.0)
        start = float(self.df["time_days"].min())
        subset = self.df[self.df["time_days"] <= start + desired].copy()
        if len(subset) < 200:
            subset = self.df.copy()
        return evenly_subsample(subset, 2600 if not QUICK_MODE else 900)

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, (2, 6, 14, 255))
        draw = ImageDraw.Draw(image)
        for star in self.stars:
            x = (star["x"] + star["drift"] * t * 0.04) % OUT_W
            y = (star["y"] + star["drift"] * t * 0.018) % OUT_H
            alpha = int(star["a"] * (0.72 + 0.28 * math.sin(t * 1.65 + star["phase"])))
            radius = star["r"]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(218, 230, 255, alpha))

        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        haze_draw = ImageDraw.Draw(haze)
        clouds = [
            (OUT_W * 0.20, OUT_H * 0.29, (65, 30, 120)),
            (OUT_W * 0.78, OUT_H * 0.42, (14, 75, 125)),
            (OUT_W * 0.52, OUT_H * 0.78, (75, 28, 54)),
        ]
        for cx, cy, color in clouds:
            for radius, alpha in [
                (430 * OUT_W / 1080.0, 16),
                (290 * OUT_W / 1080.0, 24),
                (175 * OUT_W / 1080.0, 34),
            ]:
                haze_draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color + (alpha,))
        haze = haze.filter(ImageFilter.GaussianBlur(64 if not QUICK_MODE else 32))
        image.alpha_composite(haze)
        return image

    def profile_value(self, phase: float) -> float:
        phases = self.profile["phase"].to_numpy(float)
        values = self.profile["relative_flux"].to_numpy(float)
        if len(phases) < 2:
            return 1.0
        extended_phase = np.concatenate([phases - 1.0, phases, phases + 1.0])
        extended_values = np.tile(values, 3)
        return float(np.interp(phase, extended_phase, extended_values))

    def draw_pulsing_star(
        self,
        image: Image.Image,
        center: Tuple[float, float],
        base_radius: float,
        phase: float,
        t: float,
        show_wave: bool = True,
    ):
        cx, cy = center
        flux = self.profile_value(phase)
        profile_flux = self.profile["relative_flux"].to_numpy(float)
        minimum = float(np.min(profile_flux))
        maximum = float(np.max(profile_flux))
        normalized = (flux - minimum) / max(maximum - minimum, 1e-9)

        # Brightest phase is slightly smaller/hotter in this explanatory visual.
        radius = base_radius * lerp(1.08, 0.94, normalized)
        heat = normalized
        star_color = (
            int(lerp(255, 255, heat)),
            int(lerp(138, 222, heat)),
            int(lerp(62, 150, heat)),
        )

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for scale, alpha in [(1.75, 16), (1.42, 36), (1.18, 80)]:
            rr = radius * scale
            gd.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=star_color + (alpha,))
        glow = glow.filter(ImageFilter.GaussianBlur(22 if not QUICK_MODE else 11))
        image.alpha_composite(glow)

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for index in range(24, 0, -1):
            frac = index / 24.0
            rr = radius * frac
            edge = 0.45 + 0.55 * frac
            color = (
                255,
                int(star_color[1] * edge + 42 * (1.0 - edge)),
                int(star_color[2] * edge + 18 * (1.0 - edge)),
                255,
            )
            draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=color)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(255, 242, 195, 170), width=2)

        rng = np.random.default_rng(int(t * 14.0) + 220)
        for _ in range(22):
            angle = float(rng.uniform(0, 2.0 * math.pi))
            radial = radius * math.sqrt(float(rng.uniform(0.05, 0.88)))
            x = cx + math.cos(angle) * radial
            y = cy + math.sin(angle) * radial
            spot = float(rng.uniform(radius * 0.008, radius * 0.021))
            draw.ellipse((x - spot, y - spot, x + spot, y + spot), fill=(255, 235, 175, 45))

        if show_wave:
            wave_radius = radius * (1.08 + 0.18 * ((t * 0.75) % 1.0))
            draw.ellipse(
                (cx - wave_radius, cy - wave_radius, cx + wave_radius, cy + wave_radius),
                outline=COLORS["cyan"] + (int(150 * (1.0 - ((t * 0.75) % 1.0))),),
                width=3,
            )
        image.alpha_composite(layer)

    def draw_intro(self, image: Image.Image, t: float):
        local_end = SHOT_PLAN[0]["end"]
        progress = smoothstep(t / max(local_end - 0.3, 0.1))
        cycles = 3.4 if not QUICK_MODE else 1.6
        phase = (progress * cycles) % 1.0
        center = (OUT_W * 0.5, OUT_H * 0.39)
        self.draw_pulsing_star(image, center, 190 * OUT_W / 1080.0, phase, t)

        hours = float(self.summary["detected_period_hours"])
        draw_text(
            image,
            f"{hours:.1f} HOURS",
            (OUT_W // 2, int(OUT_H * 0.67)),
            size=48 if not QUICK_MODE else 23,
            fill=COLORS["gold"] + (242,),
            bold=True,
            anchor="ma",
        )
        draw_text(
            image,
            "ONE STELLAR PULSE",
            (OUT_W // 2, int(OUT_H * 0.72)),
            size=24 if not QUICK_MODE else 12,
            fill=COLORS["white"] + (225,),
            bold=True,
            anchor="ma",
            stroke=1,
        )

    def chart_panel(self, image: Image.Image, bounds: Tuple[int, int, int, int]):
        x0, y0, x1, y1 = bounds
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            bounds,
            radius=24 if not QUICK_MODE else 12,
            fill=(3, 7, 15, 174),
            outline=(90, 185, 218, 68),
            width=1,
        )
        image.alpha_composite(overlay)

    def map_flux_y(self, values: np.ndarray, y0: float, y1: float) -> np.ndarray:
        return y0 + (self.flux_high - values) / max(self.flux_high - self.flux_low, 1e-9) * (y1 - y0)

    def draw_raw_curve(self, image: Image.Image, t: float):
        x0, x1 = int(OUT_W * 0.07), int(OUT_W * 0.93)
        y0, y1 = int(OUT_H * 0.24), int(OUT_H * 0.76)
        self.chart_panel(image, (x0, y0, x1, y1))
        draw = ImageDraw.Draw(image)

        inset_x0, inset_x1 = x0 + int(30 * OUT_W / 1080), x1 - int(20 * OUT_W / 1080)
        inset_y0, inset_y1 = y0 + int(80 * OUT_W / 1080), y1 - int(54 * OUT_W / 1080)
        for fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
            yy = lerp(inset_y0, inset_y1, fraction)
            draw.line((inset_x0, yy, inset_x1, yy), fill=(100, 175, 205, 36), width=1)

        local_start = SHOT_PLAN[1]["start"]
        reveal = smoothstep((t - local_start) / max(SHOT_PLAN[1]["end"] - local_start - 1.0, 0.1))
        count = max(2, int(len(self.raw_display) * reveal))
        subset = self.raw_display.iloc[:count]
        times = subset["time_days"].to_numpy(float)
        values = subset["brightness_percent"].to_numpy(float)
        t_min = float(self.raw_display["time_days"].min())
        t_max = float(self.raw_display["time_days"].max())
        xs = inset_x0 + (times - t_min) / max(t_max - t_min, 1e-9) * (inset_x1 - inset_x0)
        ys = self.map_flux_y(values, inset_y0, inset_y1)
        if len(xs) >= 2:
            draw.line(list(zip(xs, ys)), fill=COLORS["cyan"] + (225,), width=3 if not QUICK_MODE else 2)

        if len(xs):
            scan_x = float(xs[-1])
            draw.line((scan_x, inset_y0, scan_x, inset_y1), fill=COLORS["gold"] + (130,), width=2)
            draw.ellipse((scan_x - 7, ys[-1] - 7, scan_x + 7, ys[-1] + 7), fill=COLORS["gold"] + (245,))

        draw_text(image, "RAW TESS LIGHT CURVE", (x0 + 22, y0 + 22), size=23 if not QUICK_MODE else 11,
                  fill=COLORS["cyan"] + (235,), bold=True, stroke=1)
        draw_text(image, "time →", (x1 - 24, y1 - 18), size=16 if not QUICK_MODE else 8,
                  fill=COLORS["muted"] + (190,), anchor="ra", stroke=1)
        draw_text(image, "relative brightness", (x0 + 18, y0 + 58 if not QUICK_MODE else y0 + 29),
                  size=15 if not QUICK_MODE else 7, fill=COLORS["muted"] + (190,), stroke=1)

        gap_count = int(np.sum(np.diff(self.raw_display["time_days"].to_numpy(float)) > 0.05))
        draw_text(image, f"SAMPLES // {len(self.df):,}", (x0 + 22, y1 + (32 if not QUICK_MODE else 16)),
                  size=18 if not QUICK_MODE else 9, fill=COLORS["white"] + (220,), bold=True, stroke=1)
        draw_text(image, f"VISIBLE GAPS // {gap_count}", (x1 - 22, y1 + (32 if not QUICK_MODE else 16)),
                  size=18 if not QUICK_MODE else 9, fill=COLORS["gold"] + (220,), bold=True, anchor="ra", stroke=1)

    def draw_period_search(self, image: Image.Image, t: float):
        x0, x1 = int(OUT_W * 0.07), int(OUT_W * 0.93)
        y0, y1 = int(OUT_H * 0.24), int(OUT_H * 0.76)
        self.chart_panel(image, (x0, y0, x1, y1))
        draw = ImageDraw.Draw(image)

        ix0, ix1 = x0 + int(32 * OUT_W / 1080), x1 - int(22 * OUT_W / 1080)
        iy0, iy1 = y0 + int(82 * OUT_W / 1080), y1 - int(58 * OUT_W / 1080)
        periods = self.period_display["period_days"].to_numpy(float)
        powers = self.period_display["search_power"].to_numpy(float)
        p_min, p_max = float(np.min(periods)), float(np.max(periods))
        power_min, power_max = float(np.min(powers)), float(np.max(powers))

        local_start = SHOT_PLAN[2]["start"]
        reveal = smoothstep((t - local_start) / max(SHOT_PLAN[2]["end"] - local_start - 1.0, 0.1))
        count = max(2, int(len(periods) * reveal))
        px = ix0 + (periods[:count] - p_min) / max(p_max - p_min, 1e-9) * (ix1 - ix0)
        py = iy1 - (powers[:count] - power_min) / max(power_max - power_min, 1e-9) * (iy1 - iy0)

        for fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
            yy = lerp(iy0, iy1, fraction)
            draw.line((ix0, yy, ix1, yy), fill=(100, 175, 205, 34), width=1)
        if len(px) >= 2:
            draw.line(list(zip(px, py)), fill=COLORS["violet"] + (230,), width=3 if not QUICK_MODE else 2)

        best = float(self.summary["detected_period_days"])
        best_x = ix0 + (best - p_min) / max(p_max - p_min, 1e-9) * (ix1 - ix0)
        pulse = 0.6 + 0.4 * math.sin(t * 4.0)
        draw.line((best_x, iy0, best_x, iy1), fill=COLORS["gold"] + (int(140 + 80 * pulse),), width=3)
        draw.ellipse((best_x - 9, iy0 + 8, best_x + 9, iy0 + 26), fill=COLORS["gold"] + (240,))

        draw_text(image, "PERIOD SEARCH // THOUSANDS OF TRIAL CLOCKS", (x0 + 22, y0 + 22),
                  size=22 if not QUICK_MODE else 10, fill=COLORS["violet"] + (240,), bold=True, stroke=1)
        draw_text(image, "trial period (days) →", (x1 - 22, y1 - 20), size=15 if not QUICK_MODE else 7,
                  fill=COLORS["muted"] + (190,), anchor="ra", stroke=1)
        draw_text(image, "phase coherence", (x0 + 18, y0 + 58 if not QUICK_MODE else y0 + 29),
                  size=15 if not QUICK_MODE else 7, fill=COLORS["muted"] + (190,), stroke=1)

        draw_text(image, "BEST-FIT PERIOD", (OUT_W // 2, int(OUT_H * 0.79)), size=20 if not QUICK_MODE else 10,
                  fill=COLORS["muted"] + (210,), bold=True, anchor="ma", stroke=1)
        draw_text(image, f"{best:.6f} DAYS", (OUT_W // 2, int(OUT_H * 0.83)), size=38 if not QUICK_MODE else 18,
                  fill=COLORS["gold"] + (245,), bold=True, anchor="ma", stroke=1)

    def draw_phase_fold(self, image: Image.Image, t: float):
        x0, x1 = int(OUT_W * 0.07), int(OUT_W * 0.93)
        y0, y1 = int(OUT_H * 0.23), int(OUT_H * 0.77)
        self.chart_panel(image, (x0, y0, x1, y1))
        draw = ImageDraw.Draw(image)
        ix0, ix1 = x0 + int(34 * OUT_W / 1080), x1 - int(22 * OUT_W / 1080)
        iy0, iy1 = y0 + int(84 * OUT_W / 1080), y1 - int(58 * OUT_W / 1080)

        local_start = SHOT_PLAN[3]["start"]
        reveal = smoothstep((t - local_start) / max(SHOT_PLAN[3]["end"] - local_start - 1.0, 0.1))
        count = max(10, int(len(self.folded_display) * reveal))
        subset = self.folded_display.iloc[:count]
        phases = subset["phase"].to_numpy(float)
        values = subset["brightness_percent"].to_numpy(float)
        xs = ix0 + phases * (ix1 - ix0)
        ys = self.map_flux_y(values, iy0, iy1)

        for fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
            xx = lerp(ix0, ix1, fraction)
            draw.line((xx, iy0, xx, iy1), fill=(100, 175, 205, 30), width=1)
        for x, y in zip(xs, ys):
            radius = 3 if not QUICK_MODE else 2
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=COLORS["cyan"] + (75,))

        profile_x = ix0 + self.profile["phase"].to_numpy(float) * (ix1 - ix0)
        profile_y = self.map_flux_y(self.profile["brightness_percent"].to_numpy(float), iy0, iy1)
        profile_reveal = max(2, int(len(profile_x) * smoothstep((reveal - 0.25) / 0.75)))
        if profile_reveal >= 2:
            draw.line(list(zip(profile_x[:profile_reveal], profile_y[:profile_reveal])),
                      fill=COLORS["gold"] + (245,), width=5 if not QUICK_MODE else 3)

        scan_phase = (t * 0.32) % 1.0
        scan_x = ix0 + scan_phase * (ix1 - ix0)
        draw.line((scan_x, iy0, scan_x, iy1), fill=COLORS["rose"] + (120,), width=2)

        draw_text(image, "PHASE FOLD // EVERY CYCLE STACKED TOGETHER", (x0 + 22, y0 + 22),
                  size=22 if not QUICK_MODE else 10, fill=COLORS["cyan"] + (240,), bold=True, stroke=1)
        draw_text(image, "0.0", (ix0, y1 - 18), size=15 if not QUICK_MODE else 7,
                  fill=COLORS["muted"] + (195,), anchor="ma", stroke=1)
        draw_text(image, "0.5", ((ix0 + ix1) // 2, y1 - 18), size=15 if not QUICK_MODE else 7,
                  fill=COLORS["muted"] + (195,), anchor="ma", stroke=1)
        draw_text(image, "1.0", (ix1, y1 - 18), size=15 if not QUICK_MODE else 7,
                  fill=COLORS["muted"] + (195,), anchor="ma", stroke=1)
        draw_text(image, "pulsation phase →", (x1 - 22, y1 + (28 if not QUICK_MODE else 14)),
                  size=17 if not QUICK_MODE else 8, fill=COLORS["muted"] + (205,), anchor="ra", stroke=1)

        draw_text(image, "FAST RISE", (int(ix0 + 0.78 * (ix1 - ix0)), iy0 + 20),
                  size=18 if not QUICK_MODE else 9, fill=COLORS["rose"] + (235,), bold=True, stroke=1)
        draw_text(image, "SLOWER FADE", (int(ix0 + 0.18 * (ix1 - ix0)), iy1 - 22),
                  size=18 if not QUICK_MODE else 9, fill=COLORS["gold"] + (230,), bold=True, stroke=1)

    def draw_pulsation(self, image: Image.Image, t: float):
        local_start = SHOT_PLAN[4]["start"]
        duration = max(SHOT_PLAN[4]["end"] - local_start, 0.1)
        phase = ((t - local_start) / duration * (3.2 if not QUICK_MODE else 1.2)) % 1.0
        center = (OUT_W * 0.5, OUT_H * 0.35)
        self.draw_pulsing_star(image, center, 168 * OUT_W / 1080.0, phase, t)

        profile_flux = self.profile["relative_flux"].to_numpy(float)
        current_flux = self.profile_value(phase)
        brightness = (current_flux - float(np.min(profile_flux))) / max(float(np.max(profile_flux) - np.min(profile_flux)), 1e-9)
        radius_state = lerp(1.08, 0.94, brightness)

        draw_text(image, f"PHASE // {phase:0.2f}", (OUT_W // 2, int(OUT_H * 0.59)),
                  size=22 if not QUICK_MODE else 11, fill=COLORS["cyan"] + (230,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "HOTTER + BRIGHTER" if brightness > 0.62 else "COOLER + FAINTER",
                  (OUT_W // 2, int(OUT_H * 0.63)), size=24 if not QUICK_MODE else 12,
                  fill=(COLORS["gold"] if brightness > 0.62 else COLORS["rose"]) + (235,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "SMALLER RADIUS" if radius_state < 1.0 else "LARGER RADIUS",
                  (OUT_W // 2, int(OUT_H * 0.67)), size=19 if not QUICK_MODE else 9,
                  fill=COLORS["white"] + (215,), bold=True, anchor="ma", stroke=1)

        x0, y0 = int(OUT_W * 0.10), int(OUT_H * 0.72)
        width, height = int(OUT_W * 0.80), 178 if not QUICK_MODE else 90
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle((x0, y0, x0 + width, y0 + height), radius=24 if not QUICK_MODE else 12,
                               fill=(3, 7, 15, 176), outline=(90, 185, 218, 65), width=1)
        image.alpha_composite(overlay)
        draw_text(image, "THE REAL RHYTHM", (x0 + 22, y0 + 18), size=23 if not QUICK_MODE else 11,
                  fill=COLORS["gold"] + (240,), bold=True, stroke=1)
        lines = [
            "radial pulsation—not an eclipse",
            "rapid brightening, slower fading",
            "Blazhko modulation changes real cycles",
        ]
        yy = y0 + (58 if not QUICK_MODE else 29)
        for line in lines:
            draw_text(image, f"• {line}", (x0 + 24, yy), size=20 if not QUICK_MODE else 10,
                      fill=COLORS["white"] + (225,), bold=True, stroke=1)
            yy += 34 if not QUICK_MODE else 17

    def draw_source_hud(self, image: Image.Image):
        source = self.summary["lightcurve_source"]
        if source == "tess_lightkurve_mast":
            source_label = "SOURCE // NASA TESS via MAST"
            color = COLORS["cyan"]
        else:
            source_label = "PREVIEW SOURCE // OFFLINE FIXTURE"
            color = COLORS["gold"]
        draw_text(image, source_label, (OUT_W - (48 if not QUICK_MODE else 24), 72 if not QUICK_MODE else 36),
                  size=18 if not QUICK_MODE else 9, fill=color + (235,), bold=True, anchor="ra", stroke=1)
        draw_text(image, f"TARGET // {self.summary['target']}",
                  (OUT_W - (48 if not QUICK_MODE else 24), 104 if not QUICK_MODE else 52),
                  size=16 if not QUICK_MODE else 8, fill=COLORS["muted"] + (205,), anchor="ra", stroke=1)
        draw_text(image, f"ROWS // {self.summary['rows']:,}",
                  (OUT_W - (48 if not QUICK_MODE else 24), 132 if not QUICK_MODE else 66),
                  size=16 if not QUICK_MODE else 8, fill=COLORS["muted"] + (190,), anchor="ra", stroke=1)

    def draw_titles(self, image: Image.Image, t: float, shot_name: str):
        alpha = int(255 * smoothstep((t - 0.2) / 0.8) * (1.0 - smoothstep((t - (6.7 if not QUICK_MODE else 1.7)) / 0.7)))
        if alpha > 4:
            draw_text(image, "THE REAL RHYTHM OF", (56 if not QUICK_MODE else 28, 88 if not QUICK_MODE else 43),
                      size=42 if not QUICK_MODE else 19, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(image, "A VARIABLE STAR", (56 if not QUICK_MODE else 28, 136 if not QUICK_MODE else 67),
                      size=42 if not QUICK_MODE else 19, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(image, CONFIG["subtitle"], (58 if not QUICK_MODE else 30, 188 if not QUICK_MODE else 94),
                      size=21 if not QUICK_MODE else 10, fill=COLORS["cyan"] + (min(alpha, 230),), bold=True)

        labels = {
            "intro": "RR LYRAE // A STAR WITH A CLOCK INSIDE",
            "raw_curve": "OBSERVATIONS // BRIGHTNESS THROUGH TIME",
            "period_search": "SIGNAL SEARCH // FINDING THE HIDDEN PERIOD",
            "phase_fold": "PHASE SPACE // THE REPEATING SHAPE APPEARS",
            "pulsation": "RADIAL PULSATION // A REAL STELLAR HEARTBEAT",
        }
        if t > (5.2 if not QUICK_MODE else 1.4):
            draw_text(image, labels[shot_name], (56 if not QUICK_MODE else 28, 62 if not QUICK_MODE else 31),
                      size=19 if not QUICK_MODE else 9, fill=COLORS["muted"] + (205,), bold=True, stroke=1)

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
            fill=(2, 6, 14, 172),
            outline=(70, 180, 220, 64),
            width=1,
        )
        image.alpha_composite(panel)
        draw_wrapped_text(
            image,
            text,
            (68 if not QUICK_MODE else 34, y0 + (28 if not QUICK_MODE else 14)),
            OUT_W - (136 if not QUICK_MODE else 68),
            size=30 if not QUICK_MODE else 14,
            fill=COLORS["white"] + (245,),
        )

    def draw_hud_noise(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for item in self.hud:
            pulse = 0.5 + 0.5 * math.sin(t * 1.9 + item["phase"])
            if pulse < 0.73:
                continue
            y = (item["y"] + t * 9.0) % OUT_H
            draw.line((item["x"], y, item["x"] + item["length"], y),
                      fill=COLORS["cyan"] + (int(item["a"] * pulse),), width=1)
        offset = int((t * 39) % 7)
        for y in range(offset, OUT_H, 7):
            draw.line((0, y, OUT_W, y), fill=(120, 200, 240, 11), width=1)
        scan_y = int((t * 164) % (OUT_H + 220)) - 110
        draw.rectangle((0, scan_y, OUT_W, scan_y + (48 if not QUICK_MODE else 24)), fill=(80, 210, 240, 8))
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        image = self.background(t)
        name = shot["name"]

        if name == "intro":
            self.draw_intro(image, t)
        elif name == "raw_curve":
            self.draw_raw_curve(image, t)
        elif name == "period_search":
            self.draw_period_search(image, t)
        elif name == "phase_fold":
            self.draw_phase_fold(image, t)
        elif name == "pulsation":
            self.draw_pulsation(image, t)

        self.draw_source_hud(image)
        self.draw_titles(image, t, name)
        self.draw_caption(image, t)
        self.draw_hud_noise(image, t)

        array = np.array(image.convert("RGB"))
        array = apply_grade(array)
        array = np.clip(array.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / 0.9)
        fade_out = 1.0 - smoothstep((t - (CONFIG["duration_s"] - 1.1)) / 1.0)
        return np.clip(array.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def render_video(scene: VariableStarScene) -> Path:
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    print("Subtitle sidecar:", srt_path.resolve())

    raw_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    final_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    times = np.arange(frame_count) / CONFIG["fps"]
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(
        raw_video,
        fps=CONFIG["fps"],
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times, desc="Rendering variable-star short"):
            writer.append_data(scene.render_frame(float(t)))
    shutil.copyfile(raw_video, final_video)
    print("Final video:", final_video.resolve())
    return final_video


def main():
    print("Loading RR Lyrae metadata and light curve ...")
    df, metadata, source, notes = load_all_data()
    catalog_period = safe_float(metadata.get("catalog_period_days"))

    print("Estimating pulsation period ...")
    period_days, periodogram, period_method = estimate_period(df, catalog_period)
    folded, profile = phase_fold(df, period_days)
    summary = summarize(df, metadata, source, period_days, profile, period_method)

    lightcurve_path, summary_path = save_data_products(
        df, folded, profile, periodogram, summary, metadata, notes
    )
    create_scientific_plots(df, folded, profile, periodogram, period_days)

    print("Light-curve source:", source)
    print("Detected period:", f"{period_days:.8f} days / {period_days * 24.0:.4f} hours")
    if catalog_period:
        print("VSX/catalog period:", f"{catalog_period:.8f} days")
    for note in notes:
        print("Data note:", note)
    print("Data:", lightcurve_path.resolve())
    print("Summary:", summary_path.resolve())

    scene = VariableStarScene(df, folded, profile, periodogram, summary)
    preview_times = [
        1.0,
        min(11.0, CONFIG["duration_s"] * 0.22),
        min(25.0, CONFIG["duration_s"] * 0.44),
        min(38.0, CONFIG["duration_s"] * 0.66),
        min(50.0, CONFIG["duration_s"] * 0.86),
        CONFIG["duration_s"] - 1.0,
    ]
    for preview_time in tqdm(preview_times, desc="Preview frames"):
        frame = scene.render_frame(float(preview_time))
        Image.fromarray(frame).save(PREVIEW_DIR / f"preview_{int(preview_time):02d}s.png")

    render_video(scene)
    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-", path.name)


if __name__ == "__main__":
    main()
