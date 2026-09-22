output = 'https://youtube.com/shorts/XCShPhMNAP0?feature=share'
from __future__ import annotations

"""
A White Dwarf Pulsing in Real Data — cinematic YouTube Short renderer

Creates a vertical 1080x1920 astronomy short that reveals minute-scale
oscillations in real Kepler photometry of KIC 4552982 (WD J1916+3938), a
hydrogen-atmosphere pulsating white dwarf / ZZ Ceti star.

Preferred live source
---------------------
NASA Kepler short-cadence light curves downloaded from MAST through Lightkurve.
The script searches KIC 4552982, selects short-cadence products when available,
normalizes the flux, separates slow trends/outbursts from minute-scale pulses,
and measures the strongest pulsation periods.

Science story
-------------
- A white dwarf is the compact remnant left by a low- or intermediate-mass star.
- KIC 4552982 is a ZZ Ceti: a hydrogen-atmosphere white dwarf with non-radial
  gravity-mode oscillations.
- Kepler short cadence resolves brightness changes on timescales of minutes.
- The raw light curve also contains slow instrumental trends and genuine
  hours-long outbursts, so the pulsations are isolated with a long-window trend.
- A Fourier/Lomb-Scargle spectrum reveals several modes rather than one perfect
  heartbeat. Published observations report pulsation periods around 800–1450 s.
- Those modes let astronomers use asteroseismology to probe the star's otherwise
  hidden interior and rotation.

Offline behavior
----------------
If MAST cannot be reached, or Lightkurve is unavailable, the script uses a
clearly labeled deterministic fixture containing several ZZ-Ceti-like modes,
noise, data gaps, slow drift, and hours-long outbursts. The fallback is for
preview/layout validation only and is not observational data.

Recommended install
-------------------
    pip install numpy pandas matplotlib pillow imageio imageio-ffmpeg tqdm \
        astropy lightkurve

Quick preview render
--------------------
    WHITE_DWARF_SHORT_QUICK=1 python a_white_dwarf_pulsing_in_real_data_short.py

Force offline fixture mode
--------------------------
    WHITE_DWARF_SHORT_OFFLINE=1 python a_white_dwarf_pulsing_in_real_data_short.py


"""

import json
import math
import os
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

QUICK_MODE = os.environ.get("WHITE_DWARF_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("WHITE_DWARF_SHORT_OFFLINE", "0") == "1"
OUTPUT_ROOT = Path("white_dwarf_pulsing_real_data_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
CACHE_DIR = DATA_ROOT / "mast_cache"
for directory in [OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)



FULL_CAPTIONS = [
]
if QUICK_MODE:
    _scale = CONFIG["duration_s"] / 58.0
    CAPTIONS = [(a * _scale, b * _scale, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [

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
    image = ImageEnhance.Contrast(image).enhance(float(CONFIG["contrast"]))
    image = ImageEnhance.Color(image).enhance(float(CONFIG["saturation"]))
    return np.array(image)


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


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    return max(1.4826 * mad, 1e-9)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------

def choose_short_cadence_products(search: Any, maximum: int) -> Any:
    """Prefer products with exposure times below roughly two minutes."""
    try:
        table = search.table
        exptime = np.asarray(table["exptime"], dtype=float)
        indices = np.flatnonzero(exptime <= 120.0)
        if len(indices):
            return search[indices[:maximum]]
    except Exception:
        pass
    return search[:maximum]


def detrend_lightcurve(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values("time_days").reset_index(drop=True)
    times = out["time_days"].to_numpy(float)
    flux = out["relative_flux"].to_numpy(float)
    if len(times) < 5:
        raise RuntimeError("Not enough light-curve samples")

    cadence_days = float(np.nanmedian(np.diff(times)))
    cadence_days = max(cadence_days, 1.0 / (24.0 * 60.0 * 60.0))
    window = int(round((float(CONFIG["trend_hours"]) / 24.0) / cadence_days))
    window = max(31, window)
    if window % 2 == 0:
        window += 1

    trend = pd.Series(flux).rolling(window=window, center=True, min_periods=max(7, window // 8)).median()
    trend = trend.interpolate(limit_direction="both").to_numpy(float)
    bad = ~np.isfinite(trend) | (trend <= 0)
    if np.any(bad):
        trend[bad] = float(np.nanmedian(flux))

    pulse_percent = (flux / trend - 1.0) * 100.0
    sigma = robust_sigma(pulse_percent)
    analysis_limit = max(2.0, 12.0 * sigma)
    out["trend_flux"] = trend
    out["raw_brightness_percent"] = (flux - 1.0) * 100.0
    out["pulse_percent"] = pulse_percent
    out["analysis_ok"] = np.abs(pulse_percent - np.nanmedian(pulse_percent)) <= analysis_limit
    return out


def fetch_kepler_lightcurve() -> Tuple[pd.DataFrame, str, List[str]]:
    if lk is None:
        raise RuntimeError("lightkurve is not installed")

    notes: List[str] = []
    search = lk.search_lightcurve(CONFIG["target_name"], mission=CONFIG["mission"])
    if len(search) == 0:
        raise RuntimeError(f"No Kepler light curves found for {CONFIG['target_name']}")

    selected = choose_short_cadence_products(search, int(CONFIG["max_live_products"]))
    collection = selected.download_all(download_dir=str(CACHE_DIR))
    if collection is None or len(collection) == 0:
        raise RuntimeError("Kepler search returned products but none downloaded")

    pieces: List[pd.DataFrame] = []
    for lightcurve in collection:
        try:
            cleaned = lightcurve.remove_nans().remove_outliers(sigma=12.0).normalize()
            time_values = np.asarray(cleaned.time.value, dtype=float)
            flux_values = np.asarray(cleaned.flux.value, dtype=float)
            mask = np.isfinite(time_values) & np.isfinite(flux_values) & (flux_values > 0)
            time_values = time_values[mask]
            flux_values = flux_values[mask]
            if len(time_values) < 500:
                continue
            pieces.append(pd.DataFrame({"time_native": time_values, "relative_flux": flux_values}))
        except Exception as exc:
            notes.append(f"Skipped one downloaded product: {exc}")

    if not pieces:
        raise RuntimeError("Downloaded Kepler products contained no usable flux rows")

    frame = pd.concat(pieces, ignore_index=True).dropna().sort_values("time_native")
    frame = frame.drop_duplicates("time_native").reset_index(drop=True)
    frame["time_days"] = frame["time_native"] - float(frame["time_native"].min())
    frame["data_source"] = "kepler_short_cadence_mast"
    frame = detrend_lightcurve(frame)
    return frame, "kepler_short_cadence_mast", notes


# -----------------------------------------------------------------------------
# Offline fixture
# -----------------------------------------------------------------------------

def gaussian_event(times: np.ndarray, center: float, width: float, height_percent: float) -> np.ndarray:
    return height_percent * np.exp(-0.5 * ((times - center) / width) ** 2)


def fallback_white_dwarf_lightcurve() -> Tuple[pd.DataFrame, str]:
    """Deterministic multi-mode fixture for offline rendering only."""
    rng = np.random.default_rng(4552982)
    cadence_days = 58.85 / 86400.0
    times = np.arange(0.0, 12.0, cadence_days)

    keep = np.ones_like(times, dtype=bool)
    for start, length in [(1.8, 0.06), (4.1, 0.14), (7.35, 0.04), (9.7, 0.10)]:
        keep &= ~((times >= start) & (times <= start + length))
    times = times[keep]

    mode_periods_s = np.array([828.0, 910.0, 1014.0, 1129.0, 1248.0, 1387.0])
    mode_amplitudes_pct = np.array([0.19, 0.10, 0.16, 0.22, 0.13, 0.09])
    mode_phases = np.array([0.3, 1.8, 4.2, 2.3, 5.0, 0.9])

    time_seconds = times * 86400.0
    pulse_percent = np.zeros_like(times)
    for period, amplitude, phase in zip(mode_periods_s, mode_amplitudes_pct, mode_phases):
        slow_modulation = 1.0 + 0.22 * np.sin(2.0 * math.pi * times / (2.1 + period / 3000.0) + phase)
        pulse_percent += amplitude * slow_modulation * np.sin(2.0 * math.pi * time_seconds / period + phase)

    # Nonlinear combination terms create a less perfectly sinusoidal pulse train.
    pulse_percent += 0.055 * np.sin(2.0 * math.pi * time_seconds / 564.0 + 1.2)
    pulse_percent += rng.normal(0.0, 0.045, size=len(times))

    drift_percent = 0.16 * np.sin(2.0 * math.pi * times / 4.8) + 0.05 * (times - times.mean()) / max(float(np.ptp(times)), 1e-9)
    outburst_percent = np.zeros_like(times)
    for center, width, height in [(2.45, 0.13, 4.8), (5.85, 0.19, 8.5), (9.05, 0.11, 3.6), (11.15, 0.16, 6.2)]:
        outburst_percent += gaussian_event(times, center, width, height)

    raw_percent = pulse_percent + drift_percent + outburst_percent
    relative_flux = 1.0 + raw_percent / 100.0
    frame = pd.DataFrame({
        "time_native": times,
        "time_days": times,
        "relative_flux": relative_flux,
        "data_source": "offline_multimode_white_dwarf_fixture",
    })
    frame = detrend_lightcurve(frame)
    return frame, "offline_multimode_white_dwarf_fixture"


def load_all_data() -> Tuple[pd.DataFrame, str, List[str]]:
    notes: List[str] = []
    if OFFLINE_MODE:
        notes.append("Offline mode requested with WHITE_DWARF_SHORT_OFFLINE=1")
        frame, source = fallback_white_dwarf_lightcurve()
        return frame, source, notes

    try:
        frame, source, live_notes = fetch_kepler_lightcurve()
        notes.extend(live_notes)
        return frame, source, notes
    except Exception as exc:
        notes.append(f"Kepler/MAST fallback: {exc}")
        frame, source = fallback_white_dwarf_lightcurve()
        return frame, source, notes


# -----------------------------------------------------------------------------
# Pulsation analysis
# -----------------------------------------------------------------------------

def estimate_spectrum(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, float]], str]:
    analysis = frame[frame["analysis_ok"]].copy()
    analysis = evenly_subsample(analysis, int(CONFIG["max_analysis_points"]))
    times_s = analysis["time_days"].to_numpy(float) * 86400.0
    values = analysis["pulse_percent"].to_numpy(float)
    values = values - float(np.nanmean(values))

    periods_s = np.linspace(
        float(CONFIG["min_period_seconds"]),
        float(CONFIG["max_period_seconds"]),
        int(CONFIG["spectrum_trials"]),
    )
    frequencies_hz = 1.0 / periods_s
    method = "lomb_scargle"

    if LombScargle is not None:
        try:
            power = LombScargle(times_s, values).power(frequencies_hz)
        except Exception:
            power = np.zeros_like(periods_s)
            method = "interpolated_fft"
    else:
        power = np.zeros_like(periods_s)
        method = "interpolated_fft"

    if method == "interpolated_fft" or not np.isfinite(power).any() or float(np.nanmax(power)) <= 0:
        order = np.argsort(times_s)
        times_s = times_s[order]
        values = values[order]
        cadence = max(float(np.nanmedian(np.diff(times_s))), 1.0)
        grid = np.arange(times_s.min(), times_s.max(), cadence)
        interpolated = np.interp(grid, times_s, values)
        interpolated -= np.mean(interpolated)
        fft = np.fft.rfft(interpolated)
        fft_freq = np.fft.rfftfreq(len(interpolated), d=cadence)
        fft_power = np.abs(fft) ** 2
        valid = fft_freq > 0
        interp_freq = frequencies_hz[::-1]
        interp_power = np.interp(interp_freq, fft_freq[valid], fft_power[valid], left=0.0, right=0.0)
        power = interp_power[::-1]

    power = np.asarray(power, dtype=float)
    power -= float(np.nanmin(power))
    peak = max(float(np.nanmax(power)), 1e-12)
    power /= peak

    spectrum = pd.DataFrame({
        "period_seconds": periods_s,
        "frequency_microhz": frequencies_hz * 1e6,
        "relative_power": power,
    }).sort_values("period_seconds").reset_index(drop=True)

    # Select strong peaks while enforcing a separation in period.
    ranked = np.argsort(power)[::-1]
    modes: List[Dict[str, float]] = []
    for index in ranked:
        period = float(periods_s[index])
        if any(abs(period - mode["period_seconds"]) < 35.0 for mode in modes):
            continue
        modes.append({
            "period_seconds": period,
            "period_minutes": period / 60.0,
            "frequency_microhz": float(frequencies_hz[index] * 1e6),
            "relative_power": float(power[index]),
        })
        if len(modes) >= 8:
            break
    modes.sort(key=lambda item: item["relative_power"], reverse=True)
    return spectrum, modes, method


def phase_fold(frame: pd.DataFrame, period_seconds: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    folded = frame[frame["analysis_ok"]].copy()
    folded["phase"] = np.mod(folded["time_days"].to_numpy(float) * 86400.0 / period_seconds, 1.0)
    folded = folded.sort_values("phase").reset_index(drop=True)
    bin_count = 64
    folded["phase_bin"] = np.minimum((folded["phase"].to_numpy(float) * bin_count).astype(int), bin_count - 1)
    profile = (
        folded.groupby("phase_bin")
        .agg(phase=("phase", "median"), pulse_percent=("pulse_percent", "median"), samples=("pulse_percent", "size"))
        .reset_index(drop=True)
    )
    return folded, profile


def choose_zoom_window(frame: pd.DataFrame, hours: float = 5.0) -> pd.DataFrame:
    span_days = hours / 24.0
    if frame["time_days"].max() - frame["time_days"].min() <= span_days:
        return frame.copy()

    # Find a low-outburst section with strong minute-scale variance.
    times = frame["time_days"].to_numpy(float)
    candidates = np.linspace(times.min(), times.max() - span_days, 80)
    best_start = candidates[0]
    best_score = -np.inf
    for start in candidates:
        part = frame[(frame["time_days"] >= start) & (frame["time_days"] <= start + span_days)]
        if len(part) < 80:
            continue
        raw_level = abs(float(np.nanmedian(part["raw_brightness_percent"])))
        variability = float(np.nanstd(part["pulse_percent"]))
        score = variability - 0.08 * raw_level
        if score > best_score:
            best_score = score
            best_start = start
    return frame[(frame["time_days"] >= best_start) & (frame["time_days"] <= best_start + span_days)].copy()


def summarize(
    frame: pd.DataFrame,
    source: str,
    spectrum: pd.DataFrame,
    modes: List[Dict[str, float]],
    spectrum_method: str,
) -> Dict[str, Any]:
    strongest = modes[0]
    pulse_low, pulse_high = np.nanpercentile(frame.loc[frame["analysis_ok"], "pulse_percent"], [1.0, 99.0])
    raw_low, raw_high = np.nanpercentile(frame["raw_brightness_percent"], [1.0, 99.0])
    return {
        "target": CONFIG["target_name"],
        "alias": CONFIG["target_alias"],
        "class": "DAV / ZZ Ceti pulsating hydrogen-atmosphere white dwarf",
        "lightcurve_source": source,
        "rows": int(len(frame)),
        "time_span_days": float(frame["time_days"].max() - frame["time_days"].min()),
        "median_cadence_seconds": float(np.nanmedian(np.diff(frame["time_days"])) * 86400.0),
        "spectrum_method": spectrum_method,
        "strongest_period_seconds": float(strongest["period_seconds"]),
        "strongest_period_minutes": float(strongest["period_minutes"]),
        "strongest_frequency_microhz": float(strongest["frequency_microhz"]),
        "detected_modes": modes,
        "robust_pulsation_peak_to_peak_percent": float(pulse_high - pulse_low),
        "robust_raw_peak_to_peak_percent": float(raw_high - raw_low),
        "published_period_range_seconds": [800, 1450],
    }


def save_data_products(
    frame: pd.DataFrame,
    spectrum: pd.DataFrame,
    folded: pd.DataFrame,
    profile: pd.DataFrame,
    summary: Dict[str, Any],
    notes: List[str],
) -> Tuple[Path, Path]:
    lightcurve_path = DATA_ROOT / "kic4552982_lightcurve.csv"
    spectrum_path = DATA_ROOT / "kic4552982_pulsation_spectrum.csv"
    folded_path = DATA_ROOT / "kic4552982_strongest_mode_folded.csv"
    profile_path = DATA_ROOT / "kic4552982_strongest_mode_profile.csv"
    summary_path = DATA_ROOT / "kic4552982_summary.json"

    frame.to_csv(lightcurve_path, index=False)
    spectrum.to_csv(spectrum_path, index=False)
    folded.to_csv(folded_path, index=False)
    profile.to_csv(profile_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "notes": notes,
                "fallback_warning": "offline_multimode_white_dwarf_fixture is deterministic synthetic preview data, not observational data",
                "source_urls": {
                    "mast_kepler": "https://archive.stsci.edu/missions-and-data/kepler",
                    "lightkurve": "https://lightkurve.github.io/lightkurve/",
                    "discovery_paper": "https://arxiv.org/abs/1109.6023",
                    "kepler_analysis_paper": "https://arxiv.org/abs/1506.07878",
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return lightcurve_path, summary_path


def create_scientific_plots(
    frame: pd.DataFrame,
    spectrum: pd.DataFrame,
    folded: pd.DataFrame,
    profile: pd.DataFrame,
    summary: Dict[str, Any],
):
    raw = evenly_subsample(frame, 8000)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(raw["time_days"], raw["raw_brightness_percent"], linewidth=0.65)
    ax.set_title("KIC 4552982 Kepler light curve")
    ax.set_xlabel("Time since first sample (days)")
    ax.set_ylabel("Relative brightness (%)")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "raw_kepler_lightcurve.png", dpi=170)
    plt.close(fig)

    zoom = choose_zoom_window(frame, 5.0)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot((zoom["time_days"] - zoom["time_days"].min()) * 24.0, zoom["pulse_percent"], linewidth=0.8)
    ax.set_title("Minute-scale white-dwarf pulsations")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Detrended brightness (%)")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "pulsation_zoom.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(spectrum["period_seconds"], spectrum["relative_power"], linewidth=1.0)
    ax.axvline(summary["strongest_period_seconds"], linewidth=1.0)
    ax.set_title("White-dwarf pulsation spectrum")
    ax.set_xlabel("Period (seconds)")
    ax.set_ylabel("Relative power")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "pulsation_spectrum.png", dpi=170)
    plt.close(fig)

    folded_sample = evenly_subsample(folded, 7000)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(folded_sample["phase"], folded_sample["pulse_percent"], s=3, alpha=0.18)
    ax.plot(profile["phase"], profile["pulse_percent"], linewidth=1.8)
    ax.set_title(f"Strongest mode folded at {summary['strongest_period_seconds']:.1f} s")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Detrended brightness (%)")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "strongest_mode_folded.png", dpi=170)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class WhiteDwarfScene:
    def __init__(
        self,
        frame: pd.DataFrame,
        spectrum: pd.DataFrame,
        folded: pd.DataFrame,
        profile: pd.DataFrame,
        summary: Dict[str, Any],
    ):
        self.frame = frame.copy().reset_index(drop=True)
        self.spectrum = spectrum.copy().reset_index(drop=True)
        self.folded = folded.copy().reset_index(drop=True)
        self.profile = profile.copy().reset_index(drop=True)
        self.summary = summary
        self.stars = self._make_stars(int(CONFIG["background_stars"]), seed=73)
        self.hud = self._make_hud(int(CONFIG["hud_noise"]), seed=151)

        self.raw_display = evenly_subsample(self.frame, 2400 if not QUICK_MODE else 850)
        self.zoom_display = evenly_subsample(choose_zoom_window(self.frame, 5.0), 1500 if not QUICK_MODE else 600)
        self.spectrum_display = evenly_subsample(self.spectrum, 1200 if not QUICK_MODE else 520)
        self.folded_display = evenly_subsample(self.folded, 2600 if not QUICK_MODE else 850)

        raw_values = self.frame["raw_brightness_percent"].to_numpy(float)
        self.raw_low, self.raw_high = np.nanpercentile(raw_values, [1.0, 99.0])
        raw_pad = max((self.raw_high - self.raw_low) * 0.12, 0.3)
        self.raw_low -= raw_pad
        self.raw_high += raw_pad

        pulse_values = self.frame.loc[self.frame["analysis_ok"], "pulse_percent"].to_numpy(float)
        self.pulse_low, self.pulse_high = np.nanpercentile(pulse_values, [1.0, 99.0])
        pulse_pad = max((self.pulse_high - self.pulse_low) * 0.16, 0.08)
        self.pulse_low -= pulse_pad
        self.pulse_high += pulse_pad

    @staticmethod
    def _make_stars(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.4, 2.1)),
                "a": float(rng.uniform(18, 105)),
                "phase": float(rng.uniform(0, 2.0 * math.pi)),
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
                "phase": float(rng.uniform(0, 2.0 * math.pi)),
            }
            for _ in range(count)
        ]

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, (2, 6, 15, 255))
        draw = ImageDraw.Draw(image)
        for star in self.stars:
            alpha = int(star["a"] * (0.72 + 0.28 * math.sin(t * 1.5 + star["phase"])))
            radius = star["r"]
            draw.ellipse(
                (star["x"] - radius, star["y"] - radius, star["x"] + radius, star["y"] + radius),
                fill=(220, 235, 255, alpha),
            )

        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        haze_draw = ImageDraw.Draw(haze)
        clouds = [
            (OUT_W * 0.22, OUT_H * 0.28, (30, 50, 145)),
            (OUT_W * 0.76, OUT_H * 0.38, (55, 22, 110)),
            (OUT_W * 0.55, OUT_H * 0.78, (8, 78, 120)),
        ]
        for cx, cy, color in clouds:
            for radius, alpha in [(420 * OUT_W / 1080.0, 16), (280 * OUT_W / 1080.0, 24), (170 * OUT_W / 1080.0, 32)]:
                haze_draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color + (alpha,))
        haze = haze.filter(ImageFilter.GaussianBlur(62 if not QUICK_MODE else 31))
        image.alpha_composite(haze)
        return image

    def draw_white_dwarf(
        self,
        image: Image.Image,
        center: Tuple[float, float],
        radius: float,
        phase: float,
        mode_mix: float = 1.0,
        cutaway: bool = False,
    ):
        cx, cy = center
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for scale, alpha in [(1.8, 17), (1.45, 34), (1.18, 74)]:
            rr = radius * scale
            gd.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=COLORS["ice"] + (alpha,))
        glow = glow.filter(ImageFilter.GaussianBlur(22 if not QUICK_MODE else 11))
        image.alpha_composite(glow)

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for step in range(28, 0, -1):
            frac = step / 28.0
            rr = radius * frac
            base = int(170 + 72 * frac)
            draw.ellipse(
                (cx - rr, cy - rr, cx + rr, cy + rr),
                fill=(max(base - 30, 0), min(base + 12, 255), 255, 255),
            )
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(242, 251, 255, 190), width=2)

        # Surface brightness pattern from several non-radial modes.
        clip = Image.new("L", OUT_SIZE, 0)
        ImageDraw.Draw(clip).ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
        pattern = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(pattern)
        band_count = 9
        for band in range(-band_count, band_count + 1):
            y = cy + band * radius / band_count
            local = max(0.0, 1.0 - ((y - cy) / radius) ** 2)
            half_width = radius * math.sqrt(local)
            wave = math.sin(phase * 2.0 * math.pi + band * 0.85)
            wave += 0.55 * mode_mix * math.sin(phase * 2.0 * math.pi * 1.73 - band * 0.48)
            alpha = int(32 + 45 * abs(wave))
            color = COLORS["violet"] if wave < 0 else COLORS["cyan"]
            pd.rectangle((cx - half_width, y - radius / band_count / 2, cx + half_width, y + radius / band_count / 2), fill=color + (alpha,))
        pattern.putalpha(Image.composite(pattern.getchannel("A"), Image.new("L", OUT_SIZE, 0), clip))
        layer.alpha_composite(pattern)

        if cutaway:
            draw.pieslice((cx - radius, cy - radius, cx + radius, cy + radius), start=270, end=90, fill=(6, 13, 30, 238))
            for frac, color in [(0.76, COLORS["blue"]), (0.53, COLORS["violet"]), (0.28, COLORS["gold"])]:
                rr = radius * frac
                draw.arc((cx - rr, cy - rr, cx + rr, cy + rr), start=270, end=90, fill=color + (190,), width=3 if not QUICK_MODE else 2)

        image.alpha_composite(layer)

    @staticmethod
    def _panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 168):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(box, radius=24 if not QUICK_MODE else 12, fill=(2, 7, 17, alpha), outline=(100, 200, 235, 62), width=1)
        image.alpha_composite(overlay)

    @staticmethod
    def _map_series(
        x_values: np.ndarray,
        y_values: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        y_low: float,
        y_high: float,
    ) -> List[Tuple[float, float]]:
        if len(x_values) == 0:
            return []
        x_min = float(np.nanmin(x_values))
        x_max = float(np.nanmax(x_values))
        x_span = max(x_max - x_min, 1e-9)
        y_span = max(y_high - y_low, 1e-9)
        xs = x0 + (x_values - x_min) / x_span * (x1 - x0)
        ys = y1 - (y_values - y_low) / y_span * (y1 - y0)
        return list(zip(xs.tolist(), ys.tolist()))

    @staticmethod
    def _draw_segmented_series(
        draw: ImageDraw.ImageDraw,
        x_values: np.ndarray,
        y_values: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        y_low: float,
        y_high: float,
        fill: Tuple[int, int, int, int],
        width: int,
        gap_factor: float = 7.0,
    ) -> List[Tuple[float, float]]:
        """Draw a time series without connecting across missing-data gaps."""
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[finite]
        y_values = y_values[finite]
        if len(x_values) < 2:
            return []
        x_min = float(np.nanmin(x_values))
        x_max = float(np.nanmax(x_values))
        x_span = max(x_max - x_min, 1e-9)
        y_span = max(y_high - y_low, 1e-9)
        xs = x0 + (x_values - x_min) / x_span * (x1 - x0)
        ys = y1 - (y_values - y_low) / y_span * (y1 - y0)
        points = list(zip(xs.tolist(), ys.tolist()))

        positive_steps = np.diff(x_values)
        positive_steps = positive_steps[positive_steps > 0]
        typical_step = float(np.nanmedian(positive_steps)) if len(positive_steps) else x_span
        gap_limit = max(typical_step * gap_factor, x_span / max(len(x_values), 2) * gap_factor)
        split_after = set((np.flatnonzero(np.diff(x_values) > gap_limit) + 1).tolist())
        start = 0
        for index in range(1, len(points) + 1):
            if index in split_after or index == len(points):
                segment = points[start:index]
                if len(segment) > 1:
                    draw.line(segment, fill=fill, width=width)
                start = index
        return points

    def draw_intro(self, image: Image.Image, t: float):
        center = (OUT_W * 0.5, OUT_H * 0.37)
        radius = 170 * OUT_W / 1080.0
        local = t / max(SHOT_PLAN[0]["end"], 1e-6)
        phase = local * 9.0
        self.draw_white_dwarf(image, center, radius, phase, mode_mix=1.0)

        y = int(OUT_H * 0.67)
        x0 = int(OUT_W * 0.11)
        x1 = int(OUT_W * 0.89)
        points = []
        for i in range(180):
            u = i / 179.0
            wave = math.sin(u * math.pi * 12.0 + t * 4.0) + 0.45 * math.sin(u * math.pi * 19.0 - t * 2.2)
            points.append((lerp(x0, x1, u), y + wave * 28 * OUT_W / 1080.0))
        ImageDraw.Draw(image).line(points, fill=COLORS["cyan"] + (220,), width=3 if not QUICK_MODE else 2)
        draw_text(image, "PULSES MEASURED IN MINUTES", (OUT_W // 2, int(OUT_H * 0.73)), size=30 if not QUICK_MODE else 14,
                  fill=COLORS["gold"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "not a smooth, single heartbeat", (OUT_W // 2, int(OUT_H * 0.77)), size=20 if not QUICK_MODE else 10,
                  fill=COLORS["white"] + (220,), anchor="ma", stroke=1)

    def draw_real_data(self, image: Image.Image, t: float):
        x0, x1 = int(OUT_W * 0.07), int(OUT_W * 0.93)
        y0, y1 = int(OUT_H * 0.25), int(OUT_H * 0.76)
        self._panel(image, (x0, y0, x1, y1))
        draw = ImageDraw.Draw(image)
        zero_y = y1 - (0.0 - self.raw_low) / max(self.raw_high - self.raw_low, 1e-9) * (y1 - y0)
        draw.line((x0 + 22, zero_y, x1 - 18, zero_y), fill=(220, 235, 245, 55), width=1)

        shot = next(item for item in SHOT_PLAN if item["name"] == "real_data")
        reveal = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"] - 1.0, 1e-6))
        count = max(2, int(len(self.raw_display) * reveal))
        part = self.raw_display.iloc[:count]
        points = self._draw_segmented_series(
            draw,
            part["time_days"].to_numpy(float), part["raw_brightness_percent"].to_numpy(float),
            x0 + 22, y0 + 60, x1 - 18, y1 - 35, self.raw_low, self.raw_high,
            COLORS["ice"] + (220,), 2 if not QUICK_MODE else 1,
        )

        draw_text(image, "REAL KEPLER PHOTOMETRY", (x0 + 22, y0 + (18 if not QUICK_MODE else 10)), size=22 if not QUICK_MODE else 11,
                  fill=COLORS["cyan"] + (235,), bold=True, stroke=1)
        draw_text(image, "days of flux // pulsations + trends + outbursts", (x0 + 22, y0 + (50 if not QUICK_MODE else 29)),
                  size=16 if not QUICK_MODE else 8, fill=COLORS["muted"] + (205,), stroke=1)
        draw_text(image, "TIME →", (x1 - 18, y1 - 14), size=15 if not QUICK_MODE else 7,
                  fill=COLORS["muted"] + (190,), anchor="ra", stroke=1)
        draw_text(image, "relative flux", (x0 + 8, y0 + 72), size=14 if not QUICK_MODE else 7,
                  fill=COLORS["muted"] + (185,), stroke=1)

    def draw_zoom(self, image: Image.Image, t: float):
        x0, x1 = int(OUT_W * 0.07), int(OUT_W * 0.93)
        y0, y1 = int(OUT_H * 0.25), int(OUT_H * 0.75)
        self._panel(image, (x0, y0, x1, y1))
        draw = ImageDraw.Draw(image)
        zero_y = y1 - (0.0 - self.pulse_low) / max(self.pulse_high - self.pulse_low, 1e-9) * (y1 - y0)
        draw.line((x0 + 22, zero_y, x1 - 18, zero_y), fill=(230, 238, 245, 75), width=1)

        shot = next(item for item in SHOT_PLAN if item["name"] == "zoom")
        reveal = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"] - 0.8, 1e-6))
        count = max(2, int(len(self.zoom_display) * reveal))
        part = self.zoom_display.iloc[:count]
        hours = (part["time_days"] - float(self.zoom_display["time_days"].min())) * 24.0
        points = self._draw_segmented_series(
            draw,
            hours.to_numpy(float), part["pulse_percent"].to_numpy(float),
            x0 + 22, y0 + 66, x1 - 18, y1 - 34, self.pulse_low, self.pulse_high,
            COLORS["cyan"] + (235,), 3 if not QUICK_MODE else 2,
        )
        for point in points[:: max(1, len(points) // 90)]:
            rr = 3 if not QUICK_MODE else 1.5
            draw.ellipse((point[0] - rr, point[1] - rr, point[0] + rr, point[1] + rr), fill=COLORS["white"] + (170,))

        period = float(self.summary["strongest_period_seconds"])
        draw_text(image, "ZOOM // DETRENDED PULSATIONS", (x0 + 22, y0 + (18 if not QUICK_MODE else 10)), size=22 if not QUICK_MODE else 11,
                  fill=COLORS["cyan"] + (235,), bold=True, stroke=1)
        draw_text(image, f"strongest detected mode ≈ {period:.0f} s  ({period / 60.0:.1f} min)",
                  (x0 + 22, y0 + (50 if not QUICK_MODE else 29)), size=17 if not QUICK_MODE else 8,
                  fill=COLORS["gold"] + (235,), bold=True, stroke=1)
        draw_text(image, "HOURS →", (x1 - 18, y1 - 14), size=15 if not QUICK_MODE else 7,
                  fill=COLORS["muted"] + (190,), anchor="ra", stroke=1)

    def draw_spectrum(self, image: Image.Image, t: float):
        x0, x1 = int(OUT_W * 0.07), int(OUT_W * 0.93)
        y0, y1 = int(OUT_H * 0.25), int(OUT_H * 0.76)
        self._panel(image, (x0, y0, x1, y1))
        draw = ImageDraw.Draw(image)

        shot = next(item for item in SHOT_PLAN if item["name"] == "spectrum")
        reveal = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"] - 1.0, 1e-6))
        count = max(2, int(len(self.spectrum_display) * reveal))
        part = self.spectrum_display.iloc[:count]
        points = self._map_series(
            part["period_seconds"].to_numpy(float), part["relative_power"].to_numpy(float),
            x0 + 22, y0 + 64, x1 - 18, y1 - 34, 0.0, 1.05,
        )
        if len(points) > 1:
            draw.line(points, fill=COLORS["violet"] + (235,), width=3 if not QUICK_MODE else 2)

        p_min = float(CONFIG["min_period_seconds"])
        p_max = float(CONFIG["max_period_seconds"])
        for rank, mode in enumerate(self.summary["detected_modes"][:5], start=1):
            period = float(mode["period_seconds"])
            x = x0 + 22 + (period - p_min) / (p_max - p_min) * (x1 - x0 - 40)
            top = y1 - 34 - float(mode["relative_power"]) / 1.05 * (y1 - y0 - 98)
            draw.line((x, top - 10, x, y1 - 34), fill=COLORS["gold"] + (80,), width=1)
            draw_text(image, str(rank), (int(x), int(top - 17)), size=16 if not QUICK_MODE else 8,
                      fill=COLORS["gold"] + (235,), bold=True, anchor="ma", stroke=1)

        draw_text(image, "PULSATION SPECTRUM", (x0 + 22, y0 + (18 if not QUICK_MODE else 10)), size=23 if not QUICK_MODE else 11,
                  fill=COLORS["violet"] + (235,), bold=True, stroke=1)
        draw_text(image, "many peaks = many simultaneous gravity modes", (x0 + 22, y0 + (50 if not QUICK_MODE else 29)),
                  size=17 if not QUICK_MODE else 8, fill=COLORS["white"] + (215,), stroke=1)
        draw_text(image, "PERIOD (SECONDS) →", (x1 - 18, y1 - 14), size=15 if not QUICK_MODE else 7,
                  fill=COLORS["muted"] + (190,), anchor="ra", stroke=1)

    def draw_fold(self, image: Image.Image, t: float):
        x0, x1 = int(OUT_W * 0.07), int(OUT_W * 0.93)
        y0, y1 = int(OUT_H * 0.25), int(OUT_H * 0.76)
        self._panel(image, (x0, y0, x1, y1))
        draw = ImageDraw.Draw(image)

        shot = next(item for item in SHOT_PLAN if item["name"] == "fold")
        reveal = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"] - 0.8, 1e-6))
        visible = max(2, int(len(self.folded_display) * reveal))
        part = self.folded_display.iloc[:visible]

        x_span = x1 - x0 - 40
        y_span = y1 - y0 - 102
        for _, row in part.iterrows():
            x = x0 + 22 + float(row["phase"]) * x_span
            y = y1 - 34 - (float(row["pulse_percent"]) - self.pulse_low) / max(self.pulse_high - self.pulse_low, 1e-9) * y_span
            rr = 2.2 if not QUICK_MODE else 1.1
            draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=COLORS["ice"] + (65,))

        profile_points = []
        for _, row in self.profile.iterrows():
            x = x0 + 22 + float(row["phase"]) * x_span
            y = y1 - 34 - (float(row["pulse_percent"]) - self.pulse_low) / max(self.pulse_high - self.pulse_low, 1e-9) * y_span
            profile_points.append((x, y))
        if len(profile_points) > 1:
            draw.line(profile_points, fill=COLORS["gold"] + (240,), width=4 if not QUICK_MODE else 2)

        draw_text(image, "ONE MODE, PHASE-FOLDED", (x0 + 22, y0 + (18 if not QUICK_MODE else 10)), size=23 if not QUICK_MODE else 11,
                  fill=COLORS["gold"] + (240,), bold=True, stroke=1)
        draw_text(image, "scatter remains because other modes are still present", (x0 + 22, y0 + (50 if not QUICK_MODE else 29)),
                  size=17 if not QUICK_MODE else 8, fill=COLORS["white"] + (215,), stroke=1)
        draw_text(image, "0", (x0 + 22, y1 - 14), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (190,), anchor="ma", stroke=1)
        draw_text(image, "PHASE", (OUT_W // 2, y1 - 14), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (190,), anchor="ma", stroke=1)
        draw_text(image, "1", (x1 - 18, y1 - 14), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (190,), anchor="ma", stroke=1)

    def draw_interior(self, image: Image.Image, t: float):
        center = (OUT_W * 0.40, OUT_H * 0.39)
        radius = 165 * OUT_W / 1080.0
        phase = (t - next(item for item in SHOT_PLAN if item["name"] == "interior")["start"]) * 1.8
        self.draw_white_dwarf(image, center, radius, phase, mode_mix=1.0, cutaway=True)

        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cx, cy = center
        for index in range(6):
            angle = phase * 0.7 + index * math.pi / 3.0
            points = []
            for step in range(45):
                u = step / 44.0
                rr = radius * (0.18 + 0.72 * u)
                theta = angle + 0.55 * math.sin(u * math.pi * 4.0 + phase + index)
                points.append((cx + rr * math.cos(theta), cy + rr * math.sin(theta)))
            draw.line(points, fill=COLORS["cyan"] + (110,), width=2)
        image.alpha_composite(overlay)

        x0 = int(OUT_W * 0.08)
        y0 = int(OUT_H * 0.61)
        self._panel(image, (x0, y0, int(OUT_W * 0.92), int(OUT_H * 0.80)), alpha=176)
        draw_text(image, "WHAT THE MODES REVEAL", (x0 + 22, y0 + 20), size=24 if not QUICK_MODE else 12,
                  fill=COLORS["cyan"] + (240,), bold=True, stroke=1)
        lines = [
            "• layered interior structure",
            "• rotation from split frequencies",
            "• cooling and convection physics",
        ]
        yy = y0 + (66 if not QUICK_MODE else 33)
        for line in lines:
            draw_text(image, line, (x0 + 24, yy), size=21 if not QUICK_MODE else 10,
                      fill=COLORS["white"] + (230,), bold=True, stroke=1)
            yy += 37 if not QUICK_MODE else 18

    def draw_source_hud(self, image: Image.Image):
        live = self.summary["lightcurve_source"] == "kepler_short_cadence_mast"
        label = "SOURCE // NASA KEPLER / MAST" if live else "PREVIEW SOURCE // SYNTHETIC FIXTURE"
        color = COLORS["cyan"] if live else COLORS["gold"]
        draw_text(image, label, (OUT_W - (48 if not QUICK_MODE else 24), 72 if not QUICK_MODE else 36),
                  size=18 if not QUICK_MODE else 9, fill=color + (235,), bold=True, anchor="ra", stroke=1)
        draw_text(image, f"TARGET // {CONFIG['target_name']}", (OUT_W - (48 if not QUICK_MODE else 24), 104 if not QUICK_MODE else 52),
                  size=16 if not QUICK_MODE else 8, fill=COLORS["muted"] + (205,), anchor="ra", stroke=1)
        draw_text(image, f"CADENCE // {self.summary['median_cadence_seconds']:.1f} s", (OUT_W - (48 if not QUICK_MODE else 24), 132 if not QUICK_MODE else 66),
                  size=16 if not QUICK_MODE else 8, fill=COLORS["muted"] + (195,), anchor="ra", stroke=1)

    def draw_titles(self, image: Image.Image, t: float, shot_name: str):
        alpha = int(255 * smoothstep((t - 0.2) / 0.8) * (1.0 - smoothstep((t - (6.7 if not QUICK_MODE else 1.4)) / 0.65)))
        if alpha > 4:
            draw_text(image, "A WHITE DWARF PULSING", (56 if not QUICK_MODE else 28, 88 if not QUICK_MODE else 43),
                      size=42 if not QUICK_MODE else 19, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(image, "IN REAL DATA", (56 if not QUICK_MODE else 28, 136 if not QUICK_MODE else 67),
                      size=42 if not QUICK_MODE else 19, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(image, CONFIG["subtitle"], (58 if not QUICK_MODE else 30, 188 if not QUICK_MODE else 94),
                      size=22 if not QUICK_MODE else 10, fill=COLORS["cyan"] + (min(alpha, 230),), bold=True)

        labels = {
            "intro": "A COMPACT REMNANT // STILL OSCILLATING",
            "real_data": "THE RAW RECORD // PULSES HIDDEN IN FLUX",
            "zoom": "FIVE-HOUR ZOOM // MINUTE-SCALE VARIABILITY",
            "spectrum": "FREQUENCY SPACE // SEPARATING THE MODES",
            "fold": "PHASE SPACE // ONE MODE AT A TIME",
            "interior": "WHITE-DWARF ASTEROSEISMOLOGY",
        }
        if t > (5.1 if not QUICK_MODE else 1.2):
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
            fill=(2, 6, 15, 174),
            outline=(80, 190, 228, 65),
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
        elif name == "real_data":
            self.draw_real_data(image, t)
        elif name == "zoom":
            self.draw_zoom(image, t)
        elif name == "spectrum":
            self.draw_spectrum(image, t)
        elif name == "fold":
            self.draw_fold(image, t)
        elif name == "interior":
            self.draw_interior(image, t)

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

def render_video(scene: WhiteDwarfScene) -> Path:
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
        for t in tqdm(times, desc="Rendering white-dwarf short"):
            writer.append_data(scene.render_frame(float(t)))
    shutil.copyfile(raw_video, final_video)
    print("Final video:", final_video.resolve())
    return final_video



