# %% [markdown]
# # Cinematic YouTube Short from Real MAST/TESS Stellar-Flare Data
#
# Theme: **A STAR SUDDENLY GOT BRIGHTER**
#
# Data source:
# - TESS light-curve products archived at MAST.
# - Optional TESS target-pixel products archived at MAST.
#
# The script:
# - searches MAST through Lightkurve for real TESS observations of a flare star;
# - downloads several official SPOC light curves;
# - scans those observations for a strong multi-cadence positive brightness spike;
# - automatically chooses the strongest flare-like event found in the scanned data;
# - estimates peak brightening, robust signal-to-noise, duration above 3 sigma,
#   and approximate equivalent duration;
# - optionally downloads the matching TESS target-pixel file and replays the
#   actual detector stamp around the event;
# - creates scientific preview plots;
# - renders a 1080x1920 vertical YouTube Short;
# - writes an SRT subtitle sidecar;
# - writes title, description, and combined YouTube metadata text files;
# - exports raw and final MP4 files.
#
# Scientific fidelity:
# Timestamps and brightness values come from archived TESS products.
# The flare candidate is selected automatically using a robust positive-residual
# score after subtracting a rolling-median baseline. This is a science-
# communication detector, not a published flare-catalog pipeline and not an
# independent astrophysical classification.
#
# Camera motion, glow, scan lines, labels, and color grading are cinematic
# encodings. When target-pixel data are available, the detector stamp is real
# TESS pixel data rendered with a fixed display scale.

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import lightkurve as lk
except ImportError as exc:
    raise RuntimeError(
        "This script requires Lightkurve. Install with:\n"
        "pip install lightkurve astropy imageio imageio-ffmpeg matplotlib "
        "numpy pandas pillow tqdm"
    ) from exc

plt.rcParams["figure.figsize"] = (11, 6)
plt.rcParams["axes.grid"] = True

print("Imports loaded.")

# %% [markdown]
# ## Configuration
#
# `AD Leo` is a famous nearby flare star and is a useful default for this Short.
# The script does not hard-code a flare time. It scans downloaded TESS light
# curves and chooses the strongest multi-cadence positive brightness event it
# finds in the products inspected.
#
# For a quick rendering test, set:
#
# ```python
# CONFIG["fps"] = 12
# CONFIG["duration_s"] = 12
# CONFIG["max_lightcurves_to_scan"] = 3
# ```

OUTPUT_ROOT = Path("mast_tess_stellar_flare_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"

for p in [OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR]:
    p.mkdir(parents=True, exist_ok=True)

CONFIG = {
    # Video
    "video_width": 1080,
    "video_height": 1920,
    "fps": 24,
    "duration_s": 58,
    "output_basename": "mast_tess_stellar_flare_short",

    # MAST / TESS search
    "target": "AD Leo",
    "mission": "TESS",
    "author": "SPOC",
    "max_lightcurves_to_scan": 12,

    # Flare-candidate detector
    "baseline_window_days": 0.35,
    "event_smooth_cadences": 3,
    "minimum_event_sigma": 5.0,
    "flare_window_before_days": 0.18,
    "flare_window_after_days": 0.42,
    "duration_threshold_sigma": 3.0,

    # Rendering
    "background_star_count": 850,
    "vignette_strength": 0.30,
    "contrast_boost": 1.12,
    "saturation_boost": 1.08,

    # Text
    "title_text": "A STAR SUDDENLY GOT BRIGHTER",
    "subtitle_text": "A real stellar flare candidate in TESS data from MAST",
    "credit_text": "Data: NASA TESS products archived at MAST",
    "scientific_note": (
        "Brightness values are archived TESS measurements. "
        "The highlighted event is an automated flare-like candidate, not a published classification."
    ),

    # Optional audio / subtitle burn-in
    "audio_path": None,
    "burn_subtitles": False,
    "write_subtitle_sidecar": True,
}

print("Configuration ready.")

# %% [markdown]
# ## Light-curve preparation and robust flare-like event scoring


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return np.nan

    med = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - med)))
    sigma = 1.4826 * mad

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(values))

    return sigma



def odd_window_from_days(time_days: np.ndarray, desired_days: float) -> int:
    time_days = np.asarray(time_days, dtype=float)
    dt = np.diff(time_days)
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(dt) == 0:
        return 101

    cadence_days = float(np.nanmedian(dt))
    window = int(round(desired_days / max(cadence_days, 1e-8)))
    window = int(np.clip(window, 31, 1501))

    if window % 2 == 0:
        window += 1

    return window



def select_science_flux(lc):
    """Prefer PDCSAP_FLUX when the downloaded LightCurve contains it."""
    colnames = {str(c).lower(): str(c) for c in lc.colnames}

    if "pdcsap_flux" in colnames:
        return lc.select_flux(colnames["pdcsap_flux"]), "PDCSAP_FLUX"

    if "sap_flux" in colnames:
        return lc.select_flux(colnames["sap_flux"]), "SAP_FLUX"

    return lc, "FLUX"



def lightcurve_arrays(lc) -> Tuple[np.ndarray, np.ndarray]:
    time_days = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)

    good = np.isfinite(time_days) & np.isfinite(flux)
    time_days = time_days[good]
    flux = flux[good]

    order = np.argsort(time_days)
    time_days = time_days[order]
    flux = flux[order]

    median_flux = float(np.nanmedian(flux))
    if not np.isfinite(median_flux) or median_flux == 0:
        raise RuntimeError("Downloaded light curve has an invalid median flux.")

    normalized_flux = flux / median_flux
    return time_days, normalized_flux



def analyze_lightcurve(lc, source_label: str) -> Dict:
    selected_lc, flux_column = select_science_flux(lc)
    time_days, flux = lightcurve_arrays(selected_lc)

    if len(time_days) < 100:
        raise RuntimeError("Too few finite light-curve samples for event scoring.")

    baseline_window = odd_window_from_days(
        time_days,
        float(CONFIG["baseline_window_days"]),
    )

    series = pd.Series(flux)
    min_periods = max(11, baseline_window // 5)
    baseline = (
        series.rolling(
            baseline_window,
            center=True,
            min_periods=min_periods,
        )
        .median()
        .interpolate(limit_direction="both")
        .to_numpy(float)
    )

    finite_baseline = np.isfinite(baseline) & (baseline != 0)
    residual = np.full_like(flux, np.nan, dtype=float)
    residual[finite_baseline] = (
        flux[finite_baseline] / baseline[finite_baseline]
    ) - 1.0

    noise_sigma = robust_sigma(residual)
    if not np.isfinite(noise_sigma) or noise_sigma <= 0:
        raise RuntimeError("Could not estimate robust light-curve noise.")

    smooth_cadences = max(1, int(CONFIG["event_smooth_cadences"]))
    smoothed_residual = (
        pd.Series(residual)
        .rolling(
            smooth_cadences,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy(float)
    )

    peak_index = int(np.nanargmax(smoothed_residual))
    score_sigma = float(smoothed_residual[peak_index] / noise_sigma)

    sector = lc.meta.get("SECTOR", lc.meta.get("CAMPAIGN", "unknown"))
    try:
        sector = int(sector)
    except Exception:
        sector = str(sector)

    return {
        "source_label": source_label,
        "sector": sector,
        "flux_column": flux_column,
        "time": time_days,
        "flux": flux,
        "baseline": baseline,
        "residual": residual,
        "smoothed_residual": smoothed_residual,
        "noise_sigma": float(noise_sigma),
        "score_sigma": score_sigma,
        "peak_index": peak_index,
        "peak_time": float(time_days[peak_index]),
        "lc": lc,
    }


# %% [markdown]
# ## Search MAST and choose the strongest event found


def find_tess_lightcurves(config: Dict):
    print(
        f"Searching MAST for {config['mission']} light curves of "
        f"{config['target']} ..."
    )

    search = lk.search_lightcurve(
        config["target"],
        mission=config["mission"],
        author=config["author"],
        exptime="short",
    )

    if len(search) == 0:
        print("No short-cadence SPOC result found; retrying without cadence filter.")
        search = lk.search_lightcurve(
            config["target"],
            mission=config["mission"],
            author=config["author"],
        )

    if len(search) == 0:
        print("No SPOC result found; retrying with any available author.")
        search = lk.search_lightcurve(
            config["target"],
            mission=config["mission"],
        )

    if len(search) == 0:
        raise RuntimeError(
            f"No {config['mission']} light curves were found for {config['target']}."
        )

    print(search)
    return search



def scan_for_strongest_event(search, config: Dict) -> Tuple[Dict, pd.DataFrame]:
    scan_count = min(
        len(search),
        int(config["max_lightcurves_to_scan"]),
    )

    analyses: List[Dict] = []
    summary_rows: List[Dict] = []

    for i in tqdm(range(scan_count), desc="Scanning TESS light curves"):
        source_label = f"MAST search row {i}"

        try:
            lc = search[i].download(download_dir=str(DATA_ROOT))
            if lc is None:
                raise RuntimeError("Download returned None.")

            analysis = analyze_lightcurve(lc, source_label)
            analyses.append(analysis)

            summary_rows.append({
                "search_row": i,
                "sector": analysis["sector"],
                "flux_column": analysis["flux_column"],
                "sample_count": len(analysis["time"]),
                "robust_noise_fraction": analysis["noise_sigma"],
                "strongest_positive_score_sigma": analysis["score_sigma"],
                "candidate_time_btjd": analysis["peak_time"],
            })

        except Exception as exc:
            print(f"Skipping search row {i}: {exc}")
            summary_rows.append({
                "search_row": i,
                "sector": "error",
                "flux_column": "",
                "sample_count": 0,
                "robust_noise_fraction": np.nan,
                "strongest_positive_score_sigma": np.nan,
                "candidate_time_btjd": np.nan,
                "error": str(exc),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = DATA_ROOT / "flare_scan_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("Saved scan summary:", summary_path.resolve())

    if not analyses:
        raise RuntimeError("None of the downloaded TESS light curves could be analyzed.")

    analyses.sort(key=lambda item: item["score_sigma"], reverse=True)
    best = analyses[0]

    if best["score_sigma"] < float(config["minimum_event_sigma"]):
        print(
            "WARNING: strongest positive event is below the configured minimum "
            f"of {config['minimum_event_sigma']:.1f} sigma."
        )
        print(
            "The script will continue for visualization, but the metadata will "
            "describe it cautiously as a brightness-spike candidate."
        )

    return best, summary_df


SEARCH_RESULTS = find_tess_lightcurves(CONFIG)
EVENT, SCAN_SUMMARY = scan_for_strongest_event(SEARCH_RESULTS, CONFIG)

TIME_DAYS = EVENT["time"]
FLUX = EVENT["flux"]
BASELINE = EVENT["baseline"]
RESIDUAL = EVENT["residual"]
SMOOTHED_RESIDUAL = EVENT["smoothed_residual"]
NOISE_SIGMA = float(EVENT["noise_sigma"])
PEAK_INDEX = int(EVENT["peak_index"])
PEAK_TIME = float(EVENT["peak_time"])
SECTOR = EVENT["sector"]
FLUX_COLUMN = str(EVENT["flux_column"])
EVENT_SCORE_SIGMA = float(EVENT["score_sigma"])

print()
print("Selected strongest positive event:")
print("Target:", CONFIG["target"])
print("Sector:", SECTOR)
print("Flux column:", FLUX_COLUMN)
print("Candidate time [BTJD]:", f"{PEAK_TIME:.6f}")
print("Robust positive score [sigma]:", f"{EVENT_SCORE_SIGMA:.2f}")

# %% [markdown]
# ## Measure the selected brightness-spike candidate

FLARE_T0 = PEAK_TIME - float(CONFIG["flare_window_before_days"])
FLARE_T1 = PEAK_TIME + float(CONFIG["flare_window_after_days"])

FLARE_MASK = (
    (TIME_DAYS >= FLARE_T0)
    & (TIME_DAYS <= FLARE_T1)
    & np.isfinite(RESIDUAL)
)

if FLARE_MASK.sum() < 10:
    raise RuntimeError("Selected event window contains too few samples.")

FLARE_TIME = TIME_DAYS[FLARE_MASK]
FLARE_FLUX = FLUX[FLARE_MASK]
FLARE_BASELINE = BASELINE[FLARE_MASK]
FLARE_RESIDUAL = RESIDUAL[FLARE_MASK]

PEAK_RESIDUAL = float(np.nanmax(FLARE_RESIDUAL))
PEAK_BRIGHTENING_PERCENT = 100.0 * PEAK_RESIDUAL
PEAK_BRIGHTENING_PPM = 1_000_000.0 * PEAK_RESIDUAL

DURATION_THRESHOLD = float(CONFIG["duration_threshold_sigma"]) * NOISE_SIGMA
ABOVE = FLARE_RESIDUAL >= DURATION_THRESHOLD

if ABOVE.any():
    above_times = FLARE_TIME[ABOVE]
    DURATION_MINUTES = float(
        (above_times.max() - above_times.min()) * 24.0 * 60.0
    )
else:
    DURATION_MINUTES = 0.0

positive_residual = np.clip(FLARE_RESIDUAL, 0.0, None)
EQUIVALENT_DURATION_SECONDS = float(
    np.trapz(positive_residual, FLARE_TIME) * 86400.0
)

EVENT_METRICS = {
    "target": CONFIG["target"],
    "sector": SECTOR,
    "flux_column": FLUX_COLUMN,
    "candidate_time_btjd": PEAK_TIME,
    "robust_score_sigma": EVENT_SCORE_SIGMA,
    "peak_brightening_percent": PEAK_BRIGHTENING_PERCENT,
    "peak_brightening_ppm": PEAK_BRIGHTENING_PPM,
    "duration_above_3sigma_minutes": DURATION_MINUTES,
    "approx_equivalent_duration_seconds": EQUIVALENT_DURATION_SECONDS,
    "robust_noise_fraction": NOISE_SIGMA,
}

pd.DataFrame([EVENT_METRICS]).to_csv(
    DATA_ROOT / "selected_flare_candidate_metrics.csv",
    index=False,
)

print()
print("Measured candidate properties:")
for key, value in EVENT_METRICS.items():
    print(f"- {key}: {value}")

# %% [markdown]
# ## Optional: download the matching TESS target-pixel file
#
# The target-pixel data are not required for the Short. When a matching MAST
# product is available, the renderer uses the actual TESS detector stamp around
# the brightness-spike candidate. Otherwise the video falls back to the real
# light curve only.


def try_download_target_pixel_file(config: Dict, sector) -> Optional[object]:
    try:
        kwargs = {
            "target": config["target"],
            "mission": config["mission"],
            "author": config["author"],
        }

        if isinstance(sector, int):
            kwargs["sector"] = sector

        search = lk.search_targetpixelfile(**kwargs)

        if len(search) == 0:
            kwargs.pop("author", None)
            search = lk.search_targetpixelfile(**kwargs)

        if len(search) == 0:
            print("No matching target-pixel file found. Continuing without pixel replay.")
            return None

        print("Target-pixel search results:")
        print(search)

        tpf = search[0].download(download_dir=str(DATA_ROOT))
        return tpf

    except Exception as exc:
        print("Target-pixel download unavailable; continuing without it.")
        print("Reason:", exc)
        return None


TPF = try_download_target_pixel_file(CONFIG, SECTOR)

PIXEL_TIME = None
PIXEL_CUBE = None
PIXEL_VMIN = None
PIXEL_VMAX = None

if TPF is not None:
    try:
        pixel_time_all = np.asarray(TPF.time.value, dtype=float)
        pixel_cube_all = np.asarray(TPF.flux.value, dtype=float)

        pixel_mask = (
            np.isfinite(pixel_time_all)
            & (pixel_time_all >= FLARE_T0)
            & (pixel_time_all <= FLARE_T1)
        )

        pixel_time = pixel_time_all[pixel_mask]
        pixel_cube = pixel_cube_all[pixel_mask]

        if len(pixel_time) >= 5 and pixel_cube.ndim == 3:
            finite_pixels = pixel_cube[np.isfinite(pixel_cube)]

            if len(finite_pixels):
                PIXEL_TIME = pixel_time
                PIXEL_CUBE = pixel_cube
                PIXEL_VMIN = float(np.nanpercentile(finite_pixels, 5))
                PIXEL_VMAX = float(np.nanpercentile(finite_pixels, 99.7))

                print("Pixel replay enabled.")
                print("Pixel cube shape:", PIXEL_CUBE.shape)
            else:
                print("Pixel data contained no finite values in the event window.")
        else:
            print("Target-pixel event window was too small for replay.")

    except Exception as exc:
        print("Could not prepare target-pixel replay:", exc)

# %% [markdown]
# ## Scientific preview plots

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(TIME_DAYS, FLUX, linewidth=0.65)
ax.axvline(PEAK_TIME, linewidth=1.5, linestyle="--")
ax.set_title(
    f"TESS light curve of {CONFIG['target']} — selected positive event in Sector {SECTOR}"
)
ax.set_xlabel("TESS time [BTJD]")
ax.set_ylabel("Normalized flux")
plt.tight_layout()
plt.savefig(PREVIEW_DIR / "full_tess_light_curve.png", dpi=180)
plt.show()

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(FLARE_TIME, FLARE_FLUX, linewidth=1.1, label="normalized flux")
ax.plot(FLARE_TIME, FLARE_BASELINE, linewidth=1.0, label="rolling-median baseline")
ax.axvline(PEAK_TIME, linewidth=1.3, linestyle="--", label="selected peak")
ax.set_title("Zoom on automatically selected flare-like brightness spike")
ax.set_xlabel("TESS time [BTJD]")
ax.set_ylabel("Normalized flux")
ax.legend()
plt.tight_layout()
plt.savefig(PREVIEW_DIR / "selected_flare_candidate_zoom.png", dpi=180)
plt.show()

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(FLARE_TIME, FLARE_RESIDUAL * 100.0, linewidth=1.0)
ax.axhline(
    float(CONFIG["duration_threshold_sigma"]) * NOISE_SIGMA * 100.0,
    linewidth=1.2,
    linestyle="--",
)
ax.set_title("Brightness residual above rolling baseline")
ax.set_xlabel("TESS time [BTJD]")
ax.set_ylabel("Relative brightening [%]")
plt.tight_layout()
plt.savefig(PREVIEW_DIR / "flare_brightness_residual.png", dpi=180)
plt.show()

if PIXEL_CUBE is not None:
    nearest = int(np.argmin(np.abs(PIXEL_TIME - PEAK_TIME)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        PIXEL_CUBE[nearest],
        origin="lower",
        vmin=PIXEL_VMIN,
        vmax=PIXEL_VMAX,
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_title("Actual TESS target-pixel stamp near selected event peak")
    ax.set_xlabel("Detector column")
    ax.set_ylabel("Detector row")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "tess_pixel_stamp_peak.png", dpi=180)
    plt.show()

# %% [markdown]
# ## Rendering helpers

OUT_SIZE = (
    int(CONFIG["video_width"]),
    int(CONFIG["video_height"]),
)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))



def lerp(a, b, t):
    return a + (b - a) * t



def smoothstep(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)



def ease_in_out_sine(t):
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1) / 2



def find_ffmpeg() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return found

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None



def get_font(size: int, bold: bool = False):
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue

    return ImageFont.load_default()



def draw_text(
    img: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int = 36,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    stroke: int = 2,
    anchor: str = "la",
):
    draw = ImageDraw.Draw(img)
    font = get_font(size, bold=bold)
    draw.text(
        xy,
        str(text),
        font=font,
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, 220),
    )



def draw_wrapped_text(
    img: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int = 30,
    fill=(255, 255, 255, 245),
    bold: bool = False,
):
    draw = ImageDraw.Draw(img)
    font = get_font(size, bold=bold)

    words = str(text).split()
    lines: List[str] = []
    current = ""

    for word in words:
        candidate = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)

        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    x, y = xy

    for line in lines:
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
            stroke_width=2,
            stroke_fill=(0, 0, 0, 220),
        )
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += bbox[3] - bbox[1] + 8



def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    rr = np.sqrt(nx**2 + ny**2)

    return np.clip(
        1 - strength * rr**1.65,
        0,
        1,
    ).astype(np.float32)



def apply_grade(arr: np.ndarray) -> np.ndarray:
    img = Image.fromarray(arr)
    img = ImageEnhance.Contrast(img).enhance(float(CONFIG["contrast_boost"]))
    img = ImageEnhance.Color(img).enhance(float(CONFIG["saturation_boost"]))
    return np.array(img)


VIGNETTE = make_vignette(
    OUT_SIZE[0],
    OUT_SIZE[1],
    float(CONFIG["vignette_strength"]),
)

# %% [markdown]
# ## Space background


def generate_starfield(count: int, width: int, height: int, seed: int = 73):
    rng = np.random.default_rng(seed)
    stars = []

    for _ in range(count):
        stars.append({
            "x": float(rng.uniform(0, width)),
            "y": float(rng.uniform(0, height)),
            "r": float(rng.uniform(0.45, 2.3)),
            "alpha": int(rng.integers(35, 170)),
            "phase": float(rng.uniform(0, 2 * math.pi)),
            "drift": float(rng.uniform(-12, 12)),
        })

    return stars


STARS = generate_starfield(
    int(CONFIG["background_star_count"]),
    OUT_SIZE[0],
    OUT_SIZE[1],
)



def render_background(width: int, height: int, t: float) -> Image.Image:
    img = Image.new("RGBA", (width, height), (2, 3, 12, 255))

    nebula = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    nd = ImageDraw.Draw(nebula)

    cx = width * (0.62 + 0.025 * math.sin(t * 0.09))
    cy = height * (0.36 + 0.02 * math.cos(t * 0.07))

    for radius, alpha in [
        (720, 14),
        (520, 20),
        (330, 25),
        (180, 28),
    ]:
        nd.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=(80, 35, 110, alpha),
        )

    nebula = nebula.filter(ImageFilter.GaussianBlur(80))
    img.alpha_composite(nebula)

    draw = ImageDraw.Draw(img)

    for star in STARS:
        x = (star["x"] + star["drift"] * t * 0.22) % width
        y = (star["y"] + star["drift"] * t * 0.05) % height
        twinkle = 0.76 + 0.24 * math.sin(t * 1.2 + star["phase"])
        alpha = int(star["alpha"] * twinkle)
        radius = star["r"]

        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(225, 235, 255, alpha),
        )

    return img

# %% [markdown]
# ## Plot geometry

PLOT_RECT = (70, 365, 1010, 1220)



def plot_map(
    x: np.ndarray,
    y: np.ndarray,
    rect: Tuple[int, int, int, int],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = rect

    px = x0 + (
        (np.asarray(x, dtype=float) - xlim[0])
        / max(xlim[1] - xlim[0], 1e-12)
    ) * (x1 - x0)

    py = y1 - (
        (np.asarray(y, dtype=float) - ylim[0])
        / max(ylim[1] - ylim[0], 1e-12)
    ) * (y1 - y0)

    return px, py


FULL_XLIM = (
    float(np.nanmin(TIME_DAYS)),
    float(np.nanmax(TIME_DAYS)),
)

full_flux_lo, full_flux_hi = np.nanpercentile(FLUX, [0.5, 99.8])
full_pad = max((full_flux_hi - full_flux_lo) * 0.12, 0.002)
FULL_YLIM = (
    float(full_flux_lo - full_pad),
    float(full_flux_hi + full_pad),
)

flare_flux_lo = min(
    float(np.nanpercentile(FLARE_FLUX, 1)),
    float(np.nanpercentile(FLARE_BASELINE, 1)),
)
flare_flux_hi = max(
    float(np.nanpercentile(FLARE_FLUX, 99.9)),
    float(np.nanpercentile(FLARE_BASELINE, 99.9)),
)
flare_pad = max((flare_flux_hi - flare_flux_lo) * 0.16, 0.001)
FLARE_XLIM = (float(FLARE_TIME.min()), float(FLARE_TIME.max()))
FLARE_YLIM = (
    flare_flux_lo - flare_pad,
    flare_flux_hi + flare_pad,
)

# %% [markdown]
# ## Real-light-curve drawing


def draw_plot_shell(
    canvas: Image.Image,
    title: str,
    subtitle: str,
    t: float,
):
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = PLOT_RECT

    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=30,
        fill=(2, 7, 18, 210),
        outline=(145, 180, 255, 105),
        width=2,
    )

    # Moving vertical scan line.
    scan_x = x0 + ((t * 0.085) % 1.0) * (x1 - x0)
    draw.line(
        (scan_x, y0 + 70, scan_x, y1 - 45),
        fill=(175, 210, 255, 38),
        width=2,
    )

    canvas.alpha_composite(overlay)

    draw_text(
        canvas,
        title,
        (x0 + 28, y0 + 28),
        size=31,
        fill=(255, 255, 255, 245),
        bold=True,
    )

    draw_text(
        canvas,
        subtitle,
        (x0 + 30, y0 + 72),
        size=20,
        fill=(180, 215, 250, 210),
        stroke=1,
    )



def draw_full_light_curve(canvas: Image.Image, t: float):
    draw_plot_shell(
        canvas,
        f"REAL TESS LIGHT CURVE // {CONFIG['target']}",
        f"Sector {SECTOR}  •  {FLUX_COLUMN}  •  brightness vs. time",
        t,
    )

    progress = smoothstep((t - 4.0) / 13.0)
    visible_count = max(2, int(progress * len(TIME_DAYS)))

    x = TIME_DAYS[:visible_count]
    y = FLUX[:visible_count]
    px, py = plot_map(x, y, PLOT_RECT, FULL_XLIM, FULL_YLIM)

    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    line = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ld = ImageDraw.Draw(line)

    points = list(zip(px.astype(float), py.astype(float)))

    if len(points) >= 2:
        gd.line(points, fill=(100, 205, 255, 65), width=8)
        ld.line(points, fill=(150, 225, 255, 225), width=2)

    glow = glow.filter(ImageFilter.GaussianBlur(5))
    canvas.alpha_composite(glow)
    canvas.alpha_composite(line)

    if t >= 12.0:
        peak_px, peak_py = plot_map(
            np.array([PEAK_TIME]),
            np.array([FLUX[PEAK_INDEX]]),
            PLOT_RECT,
            FULL_XLIM,
            FULL_YLIM,
        )

        pulse = 1.0 + 0.25 * math.sin(t * 4.5)
        radius = 15 * pulse

        marker = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        md = ImageDraw.Draw(marker)
        md.ellipse(
            (
                peak_px[0] - radius,
                peak_py[0] - radius,
                peak_px[0] + radius,
                peak_py[0] + radius,
            ),
            outline=(255, 155, 100, 235),
            width=4,
        )
        canvas.alpha_composite(marker)

        draw_text(
            canvas,
            "BRIGHTNESS SPIKE",
            (int(peak_px[0]) + 20, int(peak_py[0]) - 18),
            size=24,
            fill=(255, 185, 130, 245),
            bold=True,
        )



def draw_detection_view(canvas: Image.Image, t: float):
    draw_plot_shell(
        canvas,
        "REMOVE THE SLOW BASELINE",
        "A rolling median tracks the star's slower brightness changes",
        t,
    )

    x = TIME_DAYS
    px_flux, py_flux = plot_map(x, FLUX, PLOT_RECT, FULL_XLIM, FULL_YLIM)
    px_base, py_base = plot_map(x, BASELINE, PLOT_RECT, FULL_XLIM, FULL_YLIM)

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    draw.line(
        list(zip(px_flux.astype(float), py_flux.astype(float))),
        fill=(105, 190, 235, 125),
        width=2,
    )

    reveal = smoothstep((t - 17.0) / 5.0)
    baseline_count = max(2, int(reveal * len(px_base)))

    draw.line(
        list(
            zip(
                px_base[:baseline_count].astype(float),
                py_base[:baseline_count].astype(float),
            )
        ),
        fill=(255, 225, 125, 235),
        width=4,
    )

    canvas.alpha_composite(layer)

    draw_text(
        canvas,
        "YELLOW = LOCAL BASELINE",
        (100, 1135),
        size=22,
        fill=(255, 225, 135, 230),
        bold=True,
        stroke=1,
    )

    if t >= 22.0:
        draw_text(
            canvas,
            f"STRONGEST MULTI-CADENCE SPIKE: {EVENT_SCORE_SIGMA:.1f}σ",
            (100, 1088),
            size=27,
            fill=(255, 170, 120, 245),
            bold=True,
        )



def draw_flare_zoom(canvas: Image.Image, t: float):
    draw_plot_shell(
        canvas,
        "ZOOM INTO THE EVENT",
        "The star rises above its local baseline, then fades",
        t,
    )

    px_flux, py_flux = plot_map(
        FLARE_TIME,
        FLARE_FLUX,
        PLOT_RECT,
        FLARE_XLIM,
        FLARE_YLIM,
    )

    px_base, py_base = plot_map(
        FLARE_TIME,
        FLARE_BASELINE,
        PLOT_RECT,
        FLARE_XLIM,
        FLARE_YLIM,
    )

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    draw.line(
        list(zip(px_base.astype(float), py_base.astype(float))),
        fill=(255, 215, 115, 150),
        width=3,
    )

    reveal = smoothstep((t - 27.0) / 7.0)
    count = max(2, int(reveal * len(px_flux)))

    points = list(
        zip(
            px_flux[:count].astype(float),
            py_flux[:count].astype(float),
        )
    )

    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.line(points, fill=(255, 95, 105, 80), width=12)
    glow = glow.filter(ImageFilter.GaussianBlur(7))

    draw.line(points, fill=(255, 155, 125, 245), width=4)

    canvas.alpha_composite(glow)
    canvas.alpha_composite(layer)

    marker_x, marker_y = plot_map(
        np.array([PEAK_TIME]),
        np.array([np.nanmax(FLARE_FLUX)]),
        PLOT_RECT,
        FLARE_XLIM,
        FLARE_YLIM,
    )

    draw_text(
        canvas,
        f"+{PEAK_BRIGHTENING_PERCENT:.2f}%",
        (int(marker_x[0]), int(marker_y[0]) - 32),
        size=38,
        fill=(255, 205, 160, 250),
        bold=True,
        anchor="ma",
    )

    draw_text(
        canvas,
        f"≈ {DURATION_MINUTES:.1f} min above {CONFIG['duration_threshold_sigma']:.0f}σ",
        (100, 1124),
        size=25,
        fill=(210, 225, 250, 235),
        bold=True,
        stroke=1,
    )


# %% [markdown]
# ## Actual target-pixel stamp rendering


def pixel_frame_at_video_time(t: float) -> Optional[np.ndarray]:
    if PIXEL_CUBE is None or PIXEL_TIME is None:
        return None

    progress = smoothstep((t - 38.0) / 10.0)
    target_time = lerp(float(PIXEL_TIME.min()), float(PIXEL_TIME.max()), progress)
    index = int(np.argmin(np.abs(PIXEL_TIME - target_time)))
    return np.asarray(PIXEL_CUBE[index], dtype=float)



def colorize_pixel_frame(frame: np.ndarray) -> Image.Image:
    frame = np.asarray(frame, dtype=float)
    scaled = np.clip(
        (frame - PIXEL_VMIN) / max(PIXEL_VMAX - PIXEL_VMIN, 1e-12),
        0,
        1,
    )

    # Compact magma-like cinematic mapping without depending on a GUI backend.
    r = np.clip(255 * (0.20 + 1.15 * scaled), 0, 255)
    g = np.clip(255 * (0.02 + 0.65 * scaled**1.6), 0, 255)
    b = np.clip(255 * (0.18 + 0.40 * (1 - scaled) + 0.30 * scaled**3), 0, 255)

    rgb = np.dstack([r, g, b]).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")

    target_w = 700
    target_h = 700
    return image.resize(
        (target_w, target_h),
        resample=Image.Resampling.NEAREST,
    )



def draw_pixel_replay(canvas: Image.Image, t: float):
    x0, y0, x1, y1 = PLOT_RECT

    if PIXEL_CUBE is None:
        draw_plot_shell(
            canvas,
            "THE DETECTOR SAW A BRIGHTNESS SURGE",
            "No matching target-pixel replay was available; the light curve remains real TESS data",
            t,
        )
        draw_flare_zoom(canvas, 34.0)
        return

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=30,
        fill=(2, 7, 18, 218),
        outline=(145, 180, 255, 105),
        width=2,
    )

    canvas.alpha_composite(overlay)

    draw_text(
        canvas,
        "ACTUAL TESS TARGET PIXELS",
        (x0 + 28, y0 + 28),
        size=31,
        fill=(255, 255, 255, 245),
        bold=True,
    )

    draw_text(
        canvas,
        "Same archived observation • fixed display scale through the replay",
        (x0 + 30, y0 + 72),
        size=20,
        fill=(180, 215, 250, 210),
        stroke=1,
    )

    frame = pixel_frame_at_video_time(t)
    if frame is None:
        return

    pixel_img = colorize_pixel_frame(frame)
    px = (canvas.size[0] - pixel_img.size[0]) // 2
    py = y0 + 120

    canvas.alpha_composite(pixel_img.convert("RGBA"), (px, py))

    # Pixel grid for honest detector-stamp styling.
    ph, pw = frame.shape
    grid = ImageDraw.Draw(canvas)

    for col in range(pw + 1):
        gx = px + col * pixel_img.size[0] / max(pw, 1)
        grid.line(
            (gx, py, gx, py + pixel_img.size[1]),
            fill=(255, 255, 255, 38),
            width=1,
        )

    for row in range(ph + 1):
        gy = py + row * pixel_img.size[1] / max(ph, 1)
        grid.line(
            (px, gy, px + pixel_img.size[0], gy),
            fill=(255, 255, 255, 38),
            width=1,
        )

    progress = smoothstep((t - 38.0) / 10.0)
    replay_time = lerp(float(PIXEL_TIME.min()), float(PIXEL_TIME.max()), progress)
    residual_at_time = float(
        np.interp(replay_time, FLARE_TIME, FLARE_RESIDUAL)
    )

    draw_text(
        canvas,
        f"BRIGHTENING ABOVE BASELINE: {100 * residual_at_time:+.2f}%",
        (canvas.size[0] // 2, y1 - 62),
        size=27,
        fill=(255, 210, 165, 245),
        bold=True,
        anchor="ma",
    )


# %% [markdown]
# ## Captions and text layers

CAPTIONS = [
    (
        0.5,
        6.0,
        "This is not an artist's animation of a flare. The brightness measurements come from TESS data archived at MAST.",
    ),
    (
        6.3,
        16.0,
        "Most of the light curve changes slowly. Then the star suddenly becomes brighter.",
    ),
    (
        16.3,
        26.0,
        "A rolling baseline removes slower variation. The strongest multi-cadence positive spike rises far above the local noise.",
    ),
    (
        26.3,
        37.0,
        "Zoom in: the light climbs rapidly above the baseline and then fades. That shape is flare-like.",
    ),
    (
        37.3,
        48.5,
        "When target pixels are available, these are the actual TESS detector measurements replayed around the same event.",
    ),
    (
        48.8,
        57.3,
        "A stellar flare is a sudden release of magnetic energy. In a light curve, the star can announce it as a brief surge in brightness.",
    ),
]



def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None



def draw_metric_cards(canvas: Image.Image, t: float):
    alpha = int(230 * smoothstep((t - 28.0) / 4.0))

    if alpha <= 5:
        return

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    cards = [
        ("PEAK", f"+{PEAK_BRIGHTENING_PERCENT:.2f}%"),
        ("SCORE", f"{EVENT_SCORE_SIGMA:.1f}σ"),
        ("3σ SPAN", f"{DURATION_MINUTES:.1f} min"),
    ]

    x = 65
    y = 1265
    card_w = 290
    card_h = 122

    for label, value in cards:
        draw.rounded_rectangle(
            (x, y, x + card_w, y + card_h),
            radius=24,
            fill=(0, 5, 16, int(alpha * 0.72)),
            outline=(160, 200, 255, int(alpha * 0.35)),
            width=1,
        )

        draw_text(
            overlay,
            label,
            (x + 20, y + 18),
            size=18,
            fill=(170, 205, 240, alpha),
            bold=True,
            stroke=1,
        )

        draw_text(
            overlay,
            value,
            (x + 20, y + 52),
            size=31,
            fill=(255, 230, 200, alpha),
            bold=True,
        )

        x += card_w + 35

    canvas.alpha_composite(overlay)



def add_text_layers(canvas: Image.Image, t: float):
    title_alpha = int(
        255
        * smoothstep((t - 0.25) / 1.0)
        * (1 - smoothstep((t - 5.5) / 1.0))
    )

    if title_alpha > 5:
        draw_text(
            canvas,
            CONFIG["title_text"],
            (54, 105),
            size=52,
            fill=(255, 255, 255, title_alpha),
            bold=True,
        )

        draw_wrapped_text(
            canvas,
            CONFIG["subtitle_text"],
            (58, 175),
            max_width=920,
            size=26,
            fill=(185, 215, 255, title_alpha),
        )

    cap = caption_at(t)

    if cap:
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        y0 = canvas.size[1] - 250

        draw.rounded_rectangle(
            (42, y0, canvas.size[0] - 42, y0 + 142),
            radius=26,
            fill=(0, 4, 14, 178),
        )

        canvas.alpha_composite(overlay)

        draw_wrapped_text(
            canvas,
            cap,
            (66, y0 + 24),
            max_width=canvas.size[0] - 132,
            size=28,
            fill=(255, 255, 255, 245),
            bold=True,
        )

    note_alpha = int(220 * smoothstep((t - 49.0) / 2.5))

    if note_alpha > 5:
        draw_text(
            canvas,
            CONFIG["credit_text"],
            (58, canvas.size[1] - 78),
            size=17,
            fill=(205, 225, 245, note_alpha),
            stroke=1,
        )

        draw_wrapped_text(
            canvas,
            CONFIG["scientific_note"],
            (58, canvas.size[1] - 49),
            max_width=960,
            size=14,
            fill=(165, 195, 220, note_alpha),
        )


# %% [markdown]
# ## Frame render


def render_frame(t: float) -> np.ndarray:
    canvas = render_background(OUT_SIZE[0], OUT_SIZE[1], t)

    if t < 16.0:
        draw_full_light_curve(canvas, t)
    elif t < 26.0:
        draw_detection_view(canvas, t)
    elif t < 37.5:
        draw_flare_zoom(canvas, t)
    elif t < 49.0:
        draw_pixel_replay(canvas, t)
    else:
        draw_flare_zoom(canvas, 34.0)

    draw_metric_cards(canvas, t)
    add_text_layers(canvas, t)

    arr = np.array(canvas.convert("RGB"))
    arr = apply_grade(arr)

    arr = np.clip(
        arr.astype(np.float32) * VIGNETTE[..., None],
        0,
        255,
    ).astype(np.uint8)

    fade_in = smoothstep(t / 0.9)
    fade_out = 1.0 - smoothstep(
        (
            t
            - (
                float(CONFIG["duration_s"])
                - 1.15
            )
        )
        / 1.0
    )

    arr = np.clip(
        arr.astype(np.float32) * fade_in * fade_out,
        0,
        255,
    ).astype(np.uint8)

    return arr


print("Stellar-flare renderer ready.")

# %% [markdown]
# ## Preview frames

preview_times = [
    1.0,
    10.0,
    21.0,
    32.0,
    43.0,
    float(CONFIG["duration_s"]) - 1.0,
]

preview_arrays = []

for t in tqdm(preview_times, desc="Preview frames"):
    arr = render_frame(float(t))
    preview_arrays.append(arr)

    Image.fromarray(arr).save(
        PREVIEW_DIR / f"preview_{int(t):02d}s.png"
    )

fig, axes = plt.subplots(
    1,
    len(preview_arrays),
    figsize=(19, 10),
)

for ax, image, t in zip(axes, preview_arrays, preview_times):
    ax.imshow(image)
    ax.set_title(f"{t:.0f}s")
    ax.set_axis_off()

plt.tight_layout()
plt.savefig(PREVIEW_DIR / "preview_grid.png", dpi=170)
plt.show()

print("Preview images written to:", PREVIEW_DIR.resolve())

# %% [markdown]
# ## Subtitle sidecar


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))

    hours = ms // 3_600_000
    ms %= 3_600_000

    minutes = ms // 60_000
    ms %= 60_000

    secs = ms // 1000
    ms %= 1000

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{ms:03d}"
    )



def write_srt(captions, path: Path):
    lines = []

    for index, (start, end, text) in enumerate(captions, start=1):
        lines.append(str(index))
        lines.append(
            f"{format_srt_time(start)} --> "
            f"{format_srt_time(end)}"
        )
        lines.append(text)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


SRT_PATH = OUTPUT_ROOT / (CONFIG["output_basename"] + ".srt")

if CONFIG.get("write_subtitle_sidecar", True):
    write_srt(CAPTIONS, SRT_PATH)
    print("Subtitle sidecar:", SRT_PATH.resolve())

# %% [markdown]
# ## Title and YouTube description files

SAFE_SECTOR = str(SECTOR)

TITLE_TEXT = (
    f"TESS Caught a Star Suddenly Getting Brighter | {CONFIG['target']} #Shorts"
)

DESCRIPTION_TEXT = f"""A real brightness-spike candidate found automatically in TESS data archived at MAST.

This Python visualization searches TESS light curves for {CONFIG['target']}, scans multiple downloaded observations, removes slower brightness variation with a rolling-median baseline, and selects the strongest multi-cadence positive residual in the data inspected.

Measured by this script from the selected archived light curve:

• TESS sector: {SAFE_SECTOR}
• Flux column: {FLUX_COLUMN}
• Candidate time: {PEAK_TIME:.6f} BTJD
• Robust positive-event score: {EVENT_SCORE_SIGMA:.2f} sigma
• Peak brightening above the local baseline: {PEAK_BRIGHTENING_PERCENT:.3f}% ({PEAK_BRIGHTENING_PPM:,.0f} ppm)
• Approximate time span above {CONFIG['duration_threshold_sigma']:.0f} sigma: {DURATION_MINUTES:.2f} minutes
• Approximate positive-residual equivalent duration: {EQUIVALENT_DURATION_SECONDS:.2f} seconds

Concept:
A stellar flare can appear in a light curve as a sudden rise in measured brightness followed by a decay. Stellar flares are associated with rapid releases of magnetic energy. The Short uses the actual archived TESS time-series measurements and, when a matching target-pixel product is available, replays the real detector stamp around the selected event.

Scientific caution:
The event highlighted here is an automated flare-like brightness candidate selected for science communication. A positive spike in one processing workflow is not, by itself, a published astrophysical classification. Instrumental artifacts, contamination, and data-processing choices must be checked before making a scientific claim.

Data source: NASA TESS products archived at the Mikulski Archive for Space Telescopes (MAST).
Analysis and animation: Python + Lightkurve + NumPy + Pandas + Pillow + ImageIO.

#TESS #MAST #StellarFlare #Astronomy #Space #NASA #DataVisualization #Python #ScienceShorts #LightCurve #Astrophysics #Shorts
""".strip()

TITLE_PATH = OUTPUT_ROOT / (CONFIG["output_basename"] + "_title.txt")
DESCRIPTION_PATH = OUTPUT_ROOT / (
    CONFIG["output_basename"] + "_description.txt"
)
METADATA_PATH = OUTPUT_ROOT / (
    CONFIG["output_basename"] + "_youtube_metadata.txt"
)

TITLE_PATH.write_text(TITLE_TEXT + "\n", encoding="utf-8")
DESCRIPTION_PATH.write_text(DESCRIPTION_TEXT + "\n", encoding="utf-8")
METADATA_PATH.write_text(
    "TITLE\n=====\n"
    + TITLE_TEXT
    + "\n\nDESCRIPTION\n===========\n"
    + DESCRIPTION_TEXT
    + "\n",
    encoding="utf-8",
)

print("Title file:", TITLE_PATH.resolve())
print("Description file:", DESCRIPTION_PATH.resolve())
print("YouTube metadata file:", METADATA_PATH.resolve())

# %% [markdown]
# ## Render the full vertical MP4

RAW_VIDEO_PATH = OUTPUT_ROOT / (
    CONFIG["output_basename"] + "_raw.mp4"
)

SUBBED_VIDEO_PATH = OUTPUT_ROOT / (
    CONFIG["output_basename"] + "_subbed.mp4"
)

AUDIO_VIDEO_PATH = OUTPUT_ROOT / (
    CONFIG["output_basename"] + "_with_audio.mp4"
)

FINAL_VIDEO_PATH = OUTPUT_ROOT / (
    CONFIG["output_basename"] + "_final.mp4"
)

nframes = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
times = np.arange(nframes) / CONFIG["fps"]

print(
    f"Rendering {nframes:,} frames at "
    f"{CONFIG['video_width']}x{CONFIG['video_height']} ..."
)

with iio.get_writer(
    RAW_VIDEO_PATH,
    fps=CONFIG["fps"],
    codec="libx264",
    quality=8,
    pixelformat="yuv420p",
    macro_block_size=None,
) as writer:
    for t in tqdm(times, desc="Rendering video"):
        writer.append_data(render_frame(float(t)))

print("Raw video:", RAW_VIDEO_PATH.resolve())

# %% [markdown]
# ## Optional subtitles and audio


def run_ffmpeg(cmd: List[str]):
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


ffmpeg = find_ffmpeg()
print("FFmpeg detected:", ffmpeg)

final_candidate = RAW_VIDEO_PATH

if (
    CONFIG.get("burn_subtitles", False)
    and ffmpeg
    and SRT_PATH.exists()
):
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(final_candidate),
        "-vf",
        (
            f"subtitles={SRT_PATH}:"
            "force_style="
            "Fontname=DejaVu Sans,"
            "Fontsize=22,"
            "Outline=1.2,"
            "BorderStyle=3,"
            "MarginV=90"
        ),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(SUBBED_VIDEO_PATH),
    ]

    run_ffmpeg(command)
    final_candidate = SUBBED_VIDEO_PATH

audio_path = CONFIG.get("audio_path")

if audio_path and Path(audio_path).exists() and ffmpeg:
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(final_candidate),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(AUDIO_VIDEO_PATH),
    ]

    run_ffmpeg(command)
    final_candidate = AUDIO_VIDEO_PATH

elif audio_path:
    print(
        "audio_path was set, but the audio file or FFmpeg was unavailable."
    )

if final_candidate.exists():
    shutil.copyfile(final_candidate, FINAL_VIDEO_PATH)
    print("Final video:", FINAL_VIDEO_PATH.resolve())

print("Output directory:", OUTPUT_ROOT.resolve())

for path in sorted(OUTPUT_ROOT.glob("*")):
    print("-", path.name)

# %% [markdown]
# # Suggested narration
#
# > This is not an artist's animation of a flare.
# > These brightness measurements came from TESS data archived at MAST.
# > Most of the star changes slowly.
# > Then its measured brightness suddenly rises.
# > Remove the slower baseline, and one multi-cadence spike stands far above the local noise.
# > Zoom in.
# > The light climbs rapidly, reaches a peak, and fades.
# > That shape is flare-like.
# > When target pixels are available, the detector itself can be replayed around the same event.
# > Stellar flares are sudden releases of magnetic energy.
# > In a light curve, a star can announce one with a brief surge in brightness.
#
# Suggested hook:
#
# > A SPACE TELESCOPE WATCHED THIS STAR GET BRIGHTER IN REAL TIME.
