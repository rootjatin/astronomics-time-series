from __future__ import annotations

"""
The Universe Expands in Real Redshift Data — cinematic YouTube Short renderer

Creates a vertical 1080x1920 astronomy short from the public Pantheon+SH0ES
Type Ia supernova distance–redshift catalogue. The animation reveals the
observational pattern behind cosmic expansion: more distant supernovae are,
on average, measured at larger cosmological redshift.

Preferred live source
---------------------
Pantheon+SH0ES public data release:
    Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat

The script downloads the official whitespace-delimited table, keeps valid
spectroscopically confirmed Type Ia supernova measurements, combines repeated
light-curve entries by supernova identifier, and derives:

- Hubble-diagram coordinates: redshift z_HD versus distance modulus.
- Luminosity distance in megaparsecs from the released distance modulus.
- A descriptive low-redshift slope using v ≈ cz and a line through the origin.
- Redshift-bin medians and quantiles for a robust visual trend.
- The scale factor at emission, a = 1 / (1 + z), for selected real objects.


Science story
-------------
- Redshift stretches observed wavelengths: lambda_observed = (1 + z) lambda_rest.
- Type Ia supernovae provide standardized relative-distance indicators.
- In the real Pantheon+ catalogue, larger distance and larger redshift track
  one another, revealing an expanding universe.
- The simple v ≈ cz relation is used only in the nearby, low-redshift panel.
- At larger redshift the relationship curves, so the full Hubble diagram is
  compared with a reference flat-Lambda-CDM distance-redshift curve.
- The fitted nearby slope shown in the animation is descriptive, not a new or
  independent precision measurement of the Hubble constant.

Offline behavior
----------------
If the data release cannot be reached, the script uses a clearly labelled,
deterministic synthetic fixture with a similar redshift range and scatter.
The fixture is only for preview/layout validation and is not observational data.

Recommended install
-------------------
    pip install numpy pandas matplotlib pillow imageio imageio-ffmpeg tqdm

Quick preview render
--------------------
    REDSHIFT_SHORT_QUICK=1 python the_universe_expands_in_real_redshift_data_short.py

Force offline fixture mode
--------------------------
    REDSHIFT_SHORT_OFFLINE=1 python the_universe_expands_in_real_redshift_data_short.py

Primary references
------------------
- Pantheon+SH0ES data release:
  https://github.com/PantheonPlusSH0ES/DataRelease
- Pantheon+ full dataset paper:
  https://arxiv.org/abs/2112.03863
- Pantheon+ cosmological constraints:
  https://arxiv.org/abs/2202.04077
"""

import io
import json
import math
import os
import shutil
import urllib.request
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

QUICK_MODE = os.environ.get("REDSHIFT_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("REDSHIFT_SHORT_OFFLINE", "0") == "1"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = Path("universe_expands_real_redshift_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in [OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

PANTHEON_URL = (
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
    "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "the_universe_expands_in_real_redshift_data",
    "title": "THE UNIVERSE EXPANDS IN REAL REDSHIFT DATA",
    "subtitle": "Pantheon+ // Type Ia supernovae // distance versus redshift",
    "data_timeout_s": 30,
    "reference_h0": 70.0,
    "reference_omega_m": 0.3,
    "local_z_min": 0.0233,
    "local_z_max": 0.08,
    "max_plot_points": 1543,
    "background_stars": 340,
    "hud_noise": 54,
    "contrast": 1.08,
    "saturation": 1.05,
    "vignette": 0.24,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
C_KMS = 299_792.458

COLORS = {
    "ice": (146, 224, 255),
    "cyan": (78, 226, 255),
    "blue": (78, 132, 255),
    "violet": (185, 112, 255),
    "gold": (255, 193, 91),
    "rose": (255, 102, 162),
    "white": (245, 249, 255),
    "muted": (157, 203, 226),
    "dark": (3, 7, 17),
}

FULL_CAPTIONS = [
    (0.5, 7.2, "Astronomers measure redshift when the universe stretches light toward longer wavelengths."),
    (7.3, 17.2, "These are real Type Ia supernova measurements from Pantheon+: redshift on one axis, distance on the other."),
    (17.3, 27.2, "Nearby, redshift behaves almost like a recession speed. The farther the supernova, the faster its host recedes."),
    (27.3, 38.8, "Across the full catalogue, the trend bends. At high redshift, cosmic history and geometry shape the Hubble diagram."),
    (38.9, 49.6, "Each redshift also tells us the universe's scale when that light was emitted: a equals one over one plus z."),
    (49.7, 57.4, "One catalogue, more than fifteen hundred supernovae, and the expansion of the universe written into their light."),
]
if QUICK_MODE:
    _caption_scale = float(CONFIG["duration_s"]) / 58.0
    CAPTIONS = [(a * _caption_scale, b * _caption_scale, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.8 if not QUICK_MODE else 1.65},
    {"name": "catalogue", "start": 7.8 if not QUICK_MODE else 1.65, "end": 18.5 if not QUICK_MODE else 3.85},
    {"name": "local_hubble", "start": 18.5 if not QUICK_MODE else 3.85, "end": 29.0 if not QUICK_MODE else 6.05},
    {"name": "full_diagram", "start": 29.0 if not QUICK_MODE else 6.05, "end": 40.5 if not QUICK_MODE else 8.4},
    {"name": "scale_factor", "start": 40.5 if not QUICK_MODE else 8.4, "end": 50.5 if not QUICK_MODE else 10.45},
    {"name": "finale", "start": 50.5 if not QUICK_MODE else 10.45, "end": float(CONFIG["duration_s"])},
]


# -----------------------------------------------------------------------------
# Helpers
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


def robust_limits(values: np.ndarray, low_q: float = 1.0, high_q: float = 99.0, pad: float = 0.08) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    low, high = np.nanpercentile(values, [low_q, high_q])
    span = max(float(high - low), 1e-9)
    return float(low - pad * span), float(high + pad * span)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Real data
# -----------------------------------------------------------------------------

def read_pantheon_live() -> pd.DataFrame:
    """Load an explicit/local cached table first, otherwise download and cache it."""
    candidates: List[Path] = []
    explicit = os.environ.get("PANTHEON_DATA_PATH", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([
        DATA_ROOT / "PantheonPlusSH0ES.dat",
        SCRIPT_DIR / "PantheonPlusSH0ES.dat",
        SCRIPT_DIR / "Pantheon+SH0ES.dat",
    ])
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 10_000:
            return pd.read_csv(candidate, sep=r"\s+")

    request = urllib.request.Request(PANTHEON_URL, headers={"User-Agent": "redshift-short-renderer/1.0"})
    with urllib.request.urlopen(request, timeout=int(CONFIG["data_timeout_s"])) as response:
        payload = response.read()
    if len(payload) < 10_000:
        raise RuntimeError("Pantheon+ response was unexpectedly small")
    cache_path = DATA_ROOT / "PantheonPlusSH0ES.dat"
    cache_path.write_bytes(payload)
    return pd.read_csv(io.BytesIO(payload), sep=r"\s+")


def prepare_supernovae(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    required = ["CID", "zHD", "zHDERR", "MU_SH0ES", "MU_SH0ES_ERR_DIAG", "RA", "DEC", "IS_CALIBRATOR"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise RuntimeError(f"Pantheon+ table is missing expected columns: {missing}")

    frame = raw[required].copy()
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([-999, -9], np.nan)
    frame = frame[np.isfinite(frame["zHD"]) & np.isfinite(frame["MU_SH0ES"]) & (frame["zHD"] > 0)].copy()

    # Pantheon+ contains 1701 light curves for fewer distinct supernovae.
    # Combine repeated entries by identifier for a clean one-point-per-object visual.
    grouped = (
        frame.groupby("CID", as_index=False)
        .agg(
            redshift=("zHD", "median"),
            redshift_error=("zHDERR", "median"),
            distance_modulus=("MU_SH0ES", "median"),
            distance_modulus_error=("MU_SH0ES_ERR_DIAG", "median"),
            ra_deg=("RA", "median"),
            dec_deg=("DEC", "median"),
            is_calibrator=("IS_CALIBRATOR", "max"),
            light_curve_count=("CID", "size"),
        )
        .sort_values("redshift")
        .reset_index(drop=True)
    )
    grouped["luminosity_distance_mpc"] = 10.0 ** ((grouped["distance_modulus"] - 25.0) / 5.0)
    grouped["approx_velocity_kms"] = C_KMS * grouped["redshift"]
    grouped["scale_factor_at_emission"] = 1.0 / (1.0 + grouped["redshift"])
    grouped["wavelength_stretch"] = 1.0 + grouped["redshift"]
    grouped["data_source"] = source
    return grouped


def fallback_supernovae() -> pd.DataFrame:
    """Deterministic Pantheon-like fixture for offline rendering only."""
    rng = np.random.default_rng(1701)
    count = 1543
    low = np.exp(rng.uniform(np.log(0.0012), np.log(0.12), size=int(count * 0.58)))
    middle = np.exp(rng.uniform(np.log(0.12), np.log(0.72), size=int(count * 0.31)))
    high = np.exp(rng.uniform(np.log(0.72), np.log(2.27), size=count - len(low) - len(middle)))
    redshift = np.sort(np.concatenate([low, middle, high]))
    reference_distance = luminosity_distance_flat_lcdm(redshift, h0=70.0, omega_m=0.3)
    mu = 5.0 * np.log10(np.maximum(reference_distance, 1e-8)) + 25.0
    scatter = rng.normal(0.0, 0.13 + 0.05 * np.sqrt(redshift), size=count)
    mu += scatter
    distance = 10.0 ** ((mu - 25.0) / 5.0)
    frame = pd.DataFrame({
        "CID": [f"fixture_{index:04d}" for index in range(count)],
        "redshift": redshift,
        "redshift_error": np.maximum(0.00002, 0.0005 * redshift),
        "distance_modulus": mu,
        "distance_modulus_error": 0.13 + 0.05 * np.sqrt(redshift),
        "ra_deg": rng.uniform(0, 360, size=count),
        "dec_deg": np.degrees(np.arcsin(rng.uniform(-1, 1, size=count))),
        "is_calibrator": np.zeros(count),
        "light_curve_count": np.ones(count, dtype=int),
        "luminosity_distance_mpc": distance,
        "approx_velocity_kms": C_KMS * redshift,
        "scale_factor_at_emission": 1.0 / (1.0 + redshift),
        "wavelength_stretch": 1.0 + redshift,
        "data_source": "offline_pantheon_like_fixture",
    })
    return frame


def load_all_data() -> Tuple[pd.DataFrame, str, List[str], int]:
    notes: List[str] = []
    if OFFLINE_MODE:
        notes.append("Offline mode requested with REDSHIFT_SHORT_OFFLINE=1")
        frame = fallback_supernovae()
        return frame, "offline_pantheon_like_fixture", notes, len(frame)
    try:
        raw = read_pantheon_live()
        frame = prepare_supernovae(raw, "pantheon_plus_shoes_public_release")
        return frame, "pantheon_plus_shoes_public_release", notes, int(len(raw))
    except Exception as exc:
        notes.append(f"Pantheon+ download fallback: {exc}")
        frame = fallback_supernovae()
        return frame, "offline_pantheon_like_fixture", notes, len(frame)


# -----------------------------------------------------------------------------
# Cosmology and analysis
# -----------------------------------------------------------------------------

def luminosity_distance_flat_lcdm(z: np.ndarray, h0: float = 70.0, omega_m: float = 0.3) -> np.ndarray:
    """Numerical flat-Lambda-CDM luminosity distance, sufficient for plotting."""
    z = np.asarray(z, dtype=float)
    output = np.zeros_like(z)
    omega_l = 1.0 - omega_m
    for index, value in enumerate(z):
        if value <= 0:
            output[index] = 0.0
            continue
        steps = max(180, int(220 * value))
        grid = np.linspace(0.0, value, steps)
        ez = np.sqrt(omega_m * (1.0 + grid) ** 3 + omega_l)
        comoving_mpc = (C_KMS / h0) * np.trapezoid(1.0 / ez, grid) if hasattr(np, "trapezoid") else np.trapz(1.0 / ez, grid)
        output[index] = (1.0 + value) * comoving_mpc
    return output


def fit_local_hubble_slope(frame: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    local = frame[
        (frame["redshift"] >= float(CONFIG["local_z_min"]))
        & (frame["redshift"] <= float(CONFIG["local_z_max"]))
        & (frame["is_calibrator"].fillna(0) == 0)
    ].copy()
    if len(local) < 20:
        local = frame[(frame["redshift"] >= 0.01) & (frame["redshift"] <= 0.1)].copy()
    distance = local["luminosity_distance_mpc"].to_numpy(float)
    velocity = local["approx_velocity_kms"].to_numpy(float)
    finite = np.isfinite(distance) & np.isfinite(velocity) & (distance > 0)
    distance = distance[finite]
    velocity = velocity[finite]
    if len(distance) < 2:
        return float("nan"), local
    slope = float(np.sum(distance * velocity) / np.sum(distance * distance))
    return slope, local


def redshift_bins(frame: pd.DataFrame, bin_count: int = 18) -> pd.DataFrame:
    valid = frame[(frame["redshift"] > 0) & np.isfinite(frame["distance_modulus"])].copy()
    log_edges = np.linspace(np.log10(valid["redshift"].min()), np.log10(valid["redshift"].max()), bin_count + 1)
    valid["bin"] = np.clip(np.digitize(np.log10(valid["redshift"]), log_edges) - 1, 0, bin_count - 1)
    binned = (
        valid.groupby("bin")
        .agg(
            redshift=("redshift", "median"),
            distance_modulus=("distance_modulus", "median"),
            distance_low=("distance_modulus", lambda values: np.nanpercentile(values, 16)),
            distance_high=("distance_modulus", lambda values: np.nanpercentile(values, 84)),
            samples=("CID", "size"),
        )
        .reset_index(drop=True)
    )
    return binned


def choose_representative_objects(frame: pd.DataFrame) -> pd.DataFrame:
    targets = [0.03, 0.10, 0.30, 0.70, 1.20, 2.0]
    rows = []
    used: set[str] = set()
    for target in targets:
        order = np.argsort(np.abs(frame["redshift"].to_numpy(float) - target))
        for index in order:
            row = frame.iloc[int(index)]
            cid = str(row["CID"])
            if cid not in used:
                rows.append(row)
                used.add(cid)
                break
    return pd.DataFrame(rows).reset_index(drop=True)


def summarize(frame: pd.DataFrame, source: str, raw_rows: int, local_slope: float, local: pd.DataFrame) -> Dict[str, Any]:
    real = source == "pantheon_plus_shoes_public_release"
    return {
        "source": source,
        "is_live_observational_data": bool(real),
        "released_light_curve_rows": int(raw_rows),
        "distinct_supernovae_rendered": int(len(frame)),
        "redshift_min": float(frame["redshift"].min()),
        "redshift_max": float(frame["redshift"].max()),
        "distance_modulus_min": float(frame["distance_modulus"].min()),
        "distance_modulus_max": float(frame["distance_modulus"].max()),
        "max_wavelength_stretch": float(frame["wavelength_stretch"].max()),
        "min_emission_scale_factor": float(frame["scale_factor_at_emission"].min()),
        "descriptive_local_slope_km_s_mpc": float(local_slope),
        "local_slope_redshift_range": [float(CONFIG["local_z_min"]), float(CONFIG["local_z_max"])],
        "local_slope_sample_count": int(len(local)),
        "local_slope_warning": (
            "Descriptive fit using released, calibrated distances and v≈cz; "
            "not an independent precision H0 measurement."
        ),
        "reference_curve": {
            "model": "flat Lambda-CDM",
            "H0_km_s_Mpc": float(CONFIG["reference_h0"]),
            "Omega_m": float(CONFIG["reference_omega_m"]),
            "purpose": "visual reference only",
        },
    }


def save_data_products(
    frame: pd.DataFrame,
    binned: pd.DataFrame,
    representatives: pd.DataFrame,
    summary: Dict[str, Any],
    notes: List[str],
) -> Tuple[Path, Path]:
    catalogue_path = DATA_ROOT / "pantheon_plus_distinct_supernovae.csv"
    bins_path = DATA_ROOT / "pantheon_plus_redshift_bins.csv"
    reps_path = DATA_ROOT / "pantheon_plus_representative_objects.csv"
    summary_path = DATA_ROOT / "redshift_short_summary.json"
    frame.to_csv(catalogue_path, index=False)
    binned.to_csv(bins_path, index=False)
    representatives.to_csv(reps_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "notes": notes,
                "fallback_warning": "offline_pantheon_like_fixture is synthetic preview data, not observational data",
                "source_urls": {
                    "pantheon_plus_data_release": "https://github.com/PantheonPlusSH0ES/DataRelease",
                    "pantheon_plus_dataset_paper": "https://arxiv.org/abs/2112.03863",
                    "pantheon_plus_cosmology_paper": "https://arxiv.org/abs/2202.04077",
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return catalogue_path, summary_path


def create_scientific_plots(
    frame: pd.DataFrame,
    binned: pd.DataFrame,
    local: pd.DataFrame,
    summary: Dict[str, Any],
):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sample = evenly_subsample(frame, 3000)
    ax.scatter(sample["redshift"], sample["distance_modulus"], s=5, alpha=0.3)
    ax.set_xscale("log")
    ax.set_title("Pantheon+ Type Ia supernova Hubble diagram")
    ax.set_xlabel("Hubble-diagram redshift z")
    ax.set_ylabel("Distance modulus")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "pantheon_plus_hubble_diagram.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.scatter(local["luminosity_distance_mpc"], local["approx_velocity_kms"], s=7, alpha=0.35)
    max_distance = float(local["luminosity_distance_mpc"].quantile(0.99))
    x = np.linspace(0.0, max_distance, 200)
    ax.plot(x, summary["descriptive_local_slope_km_s_mpc"] * x, linewidth=1.5)
    ax.set_title("Nearby descriptive Hubble relation")
    ax.set_xlabel("Luminosity distance (Mpc)")
    ax.set_ylabel("Approximate recession velocity cz (km/s)")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "nearby_hubble_relation.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.errorbar(
        binned["redshift"],
        binned["distance_modulus"],
        yerr=[binned["distance_modulus"] - binned["distance_low"], binned["distance_high"] - binned["distance_modulus"]],
        fmt="o-",
        linewidth=1.0,
        markersize=4,
    )
    ax.set_xscale("log")
    ax.set_title("Binned Pantheon+ distance-redshift trend")
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Distance modulus")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "binned_redshift_trend.png", dpi=170)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class RedshiftScene:
    def __init__(
        self,
        frame: pd.DataFrame,
        binned: pd.DataFrame,
        representatives: pd.DataFrame,
        local: pd.DataFrame,
        summary: Dict[str, Any],
    ):
        self.frame = frame.copy().reset_index(drop=True)
        self.binned = binned.copy().reset_index(drop=True)
        self.representatives = representatives.copy().reset_index(drop=True)
        self.local = local.copy().reset_index(drop=True)
        self.summary = summary
        self.stars = self._make_stars(int(CONFIG["background_stars"]), 58)
        self.hud = self._make_hud(int(CONFIG["hud_noise"]), 91)
        self.catalogue_display = evenly_subsample(self.frame.sample(frac=1.0, random_state=42), int(CONFIG["max_plot_points"]))
        self.catalogue_display = self.catalogue_display.sort_values("redshift").reset_index(drop=True)
        self.local_display = evenly_subsample(self.local.sample(frac=1.0, random_state=17), 720 if not QUICK_MODE else 260)
        self.z_min = max(float(self.frame["redshift"].min()), 1e-4)
        self.z_max = float(self.frame["redshift"].max())
        self.mu_low, self.mu_high = robust_limits(self.frame["distance_modulus"].to_numpy(float), 0.5, 99.5, 0.04)
        self.local_distance_high = float(self.local["luminosity_distance_mpc"].quantile(0.995))
        self.local_velocity_high = float(self.local["approx_velocity_kms"].quantile(0.995))
        curve_z = np.geomspace(max(self.z_min, 0.001), self.z_max, 360)
        curve_distance = luminosity_distance_flat_lcdm(
            curve_z,
            h0=float(CONFIG["reference_h0"]),
            omega_m=float(CONFIG["reference_omega_m"]),
        )
        self.reference_curve = pd.DataFrame({
            "redshift": curve_z,
            "distance_modulus": 5.0 * np.log10(np.maximum(curve_distance, 1e-8)) + 25.0,
        })

    @staticmethod
    def _make_stars(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.35, 2.2)),
                "a": float(rng.uniform(18, 108)),
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

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, (2, 6, 16, 255))
        draw = ImageDraw.Draw(image)
        for star in self.stars:
            alpha = int(star["a"] * (0.72 + 0.28 * math.sin(t * 1.4 + star["phase"])))
            r = star["r"]
            draw.ellipse((star["x"] - r, star["y"] - r, star["x"] + r, star["y"] + r), fill=(220, 235, 255, alpha))
        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        clouds = [
            (OUT_W * 0.22, OUT_H * 0.30, (24, 48, 145)),
            (OUT_W * 0.78, OUT_H * 0.38, (65, 20, 118)),
            (OUT_W * 0.52, OUT_H * 0.79, (8, 78, 123)),
        ]
        for cx, cy, color in clouds:
            for radius, alpha in [(420 * OUT_W / 1080, 16), (280 * OUT_W / 1080, 24), (170 * OUT_W / 1080, 32)]:
                hd.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color + (alpha,))
        haze = haze.filter(ImageFilter.GaussianBlur(62 if not QUICK_MODE else 31))
        image.alpha_composite(haze)
        return image

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 170):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(box, radius=24 if not QUICK_MODE else 12, fill=(2, 7, 18, alpha), outline=(100, 200, 235, 64), width=1)
        image.alpha_composite(overlay)

    def map_log_z(self, z: float, x0: float, x1: float) -> float:
        low = math.log10(self.z_min)
        high = math.log10(self.z_max)
        return x0 + (math.log10(max(z, self.z_min)) - low) / max(high - low, 1e-9) * (x1 - x0)

    def map_mu(self, mu: float, y0: float, y1: float) -> float:
        return y1 - (mu - self.mu_low) / max(self.mu_high - self.mu_low, 1e-9) * (y1 - y0)

    def draw_expanding_grid(self, image: Image.Image, center: Tuple[float, float], phase: float, alpha: int = 120):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cx, cy = center
        base = 32 * OUT_W / 1080
        for ring in range(1, 9):
            radius = base * (ring + phase % 1.0)
            fade = int(alpha * (1.0 - ring / 10.0))
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=COLORS["cyan"] + (fade,), width=2 if not QUICK_MODE else 1)
        for angle_index in range(12):
            angle = 2 * math.pi * angle_index / 12.0 + phase * 0.08
            radius = base * 9.2
            draw.line((cx, cy, cx + radius * math.cos(angle), cy + radius * math.sin(angle)), fill=COLORS["blue"] + (42,), width=1)
        image.alpha_composite(overlay)

    def draw_intro(self, image: Image.Image, t: float):
        center = (OUT_W * 0.5, OUT_H * 0.37)
        local = t / max(SHOT_PLAN[0]["end"], 1e-6)
        self.draw_expanding_grid(image, center, local * 2.2, alpha=155)
        draw = ImageDraw.Draw(image)
        rest_x = int(OUT_W * 0.27)
        obs_x = int(OUT_W * 0.72)
        y0 = int(OUT_H * 0.60)
        y1 = int(OUT_H * 0.72)
        draw.line((OUT_W * 0.13, y1, OUT_W * 0.87, y1), fill=COLORS["muted"] + (100,), width=2)
        rest_shift = 0.04 * math.sin(t * 2.0)
        draw.rectangle((rest_x - 5, y0, rest_x + 5, y1), fill=COLORS["cyan"] + (235,))
        moving_x = int(lerp(rest_x + 15, obs_x, smoothstep(local)))
        draw.rectangle((moving_x - 6, y0 - 12, moving_x + 6, y1), fill=COLORS["rose"] + (240,))
        draw_text(image, "REST", (rest_x, y1 + (18 if not QUICK_MODE else 9)), size=16 if not QUICK_MODE else 8, fill=COLORS["cyan"] + (220,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "OBSERVED", (moving_x, y1 + (18 if not QUICK_MODE else 9)), size=16 if not QUICK_MODE else 8, fill=COLORS["rose"] + (220,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "λOBS = (1 + z) λREST", (OUT_W // 2, int(OUT_H * 0.79)), size=28 if not QUICK_MODE else 13, fill=COLORS["gold"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "space expands // wavelengths stretch", (OUT_W // 2, int(OUT_H * 0.83)), size=19 if not QUICK_MODE else 9, fill=COLORS["white"] + (215,), anchor="ma", stroke=1)

    def draw_catalogue(self, image: Image.Image, t: float):
        x0, x1 = int(OUT_W * 0.07), int(OUT_W * 0.93)
        y0, y1 = int(OUT_H * 0.24), int(OUT_H * 0.77)
        self.panel(image, (x0, y0, x1, y1))
        draw = ImageDraw.Draw(image)
        shot = next(item for item in SHOT_PLAN if item["name"] == "catalogue")
        reveal = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"] - 0.8, 1e-6))
        count = max(2, int(len(self.catalogue_display) * reveal))
        plot_x0, plot_x1 = x0 + 30, x1 - 20
        plot_y0, plot_y1 = y0 + 78, y1 - 38
        for _, row in self.catalogue_display.iloc[:count].iterrows():
            x = self.map_log_z(float(row["redshift"]), plot_x0, plot_x1)
            y = self.map_mu(float(row["distance_modulus"]), plot_y0, plot_y1)
            r = 2.4 if not QUICK_MODE else 1.2
            draw.ellipse((x - r, y - r, x + r, y + r), fill=COLORS["ice"] + (95,))
        draw_text(image, "REAL PANTHEON+ HUBBLE DIAGRAM", (x0 + 22, y0 + (18 if not QUICK_MODE else 10)), size=22 if not QUICK_MODE else 11, fill=COLORS["cyan"] + (238,), bold=True, stroke=1)
        draw_text(image, f"{self.summary['distinct_supernovae_rendered']:,} distinct Type Ia supernovae", (x0 + 22, y0 + (50 if not QUICK_MODE else 29)), size=17 if not QUICK_MODE else 8, fill=COLORS["white"] + (215,), stroke=1)
        draw_text(image, "REDSHIFT z  →  (LOG SCALE)", (x1 - 18, y1 - 15), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (190,), anchor="ra", stroke=1)
        draw_text(image, "distance modulus", (x0 + 10, y0 + 80), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (190,), stroke=1)

    def draw_local_hubble(self, image: Image.Image, t: float):
        x0, x1 = int(OUT_W * 0.07), int(OUT_W * 0.93)
        y0, y1 = int(OUT_H * 0.24), int(OUT_H * 0.77)
        self.panel(image, (x0, y0, x1, y1))
        draw = ImageDraw.Draw(image)
        shot = next(item for item in SHOT_PLAN if item["name"] == "local_hubble")
        reveal = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"] - 0.8, 1e-6))
        plot_x0, plot_x1 = x0 + 30, x1 - 20
        plot_y0, plot_y1 = y0 + 80, y1 - 38
        visible = max(2, int(len(self.local_display) * reveal))
        for _, row in self.local_display.iloc[:visible].iterrows():
            x = plot_x0 + float(row["luminosity_distance_mpc"]) / max(self.local_distance_high, 1e-9) * (plot_x1 - plot_x0)
            y = plot_y1 - float(row["approx_velocity_kms"]) / max(self.local_velocity_high, 1e-9) * (plot_y1 - plot_y0)
            r = 2.8 if not QUICK_MODE else 1.3
            draw.ellipse((x - r, y - r, x + r, y + r), fill=COLORS["cyan"] + (105,))
        slope = float(self.summary["descriptive_local_slope_km_s_mpc"])
        line_x = np.linspace(0, self.local_distance_high, 90)
        points = []
        for distance in line_x:
            velocity = slope * distance
            x = plot_x0 + distance / max(self.local_distance_high, 1e-9) * (plot_x1 - plot_x0)
            y = plot_y1 - velocity / max(self.local_velocity_high, 1e-9) * (plot_y1 - plot_y0)
            points.append((x, y))
        draw.line(points, fill=COLORS["gold"] + (240,), width=4 if not QUICK_MODE else 2)
        draw_text(image, "NEARBY // THE HUBBLE RELATION", (x0 + 22, y0 + (18 if not QUICK_MODE else 10)), size=22 if not QUICK_MODE else 11, fill=COLORS["gold"] + (238,), bold=True, stroke=1)
        draw_text(image, f"descriptive slope ≈ {slope:.0f} km/s/Mpc", (x0 + 22, y0 + (50 if not QUICK_MODE else 29)), size=18 if not QUICK_MODE else 9, fill=COLORS["white"] + (225,), bold=True, stroke=1)
        draw_text(image, "DISTANCE (Mpc) →", (x1 - 18, y1 - 15), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (190,), anchor="ra", stroke=1)
        draw_text(image, "cz (km/s)", (x0 + 10, y0 + 80), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (190,), stroke=1)
        draw_text(image, "visual fit only — not an independent precision H₀ result", (OUT_W // 2, int(OUT_H * 0.82)), size=16 if not QUICK_MODE else 8, fill=COLORS["muted"] + (205,), anchor="ma", stroke=1)

    def draw_full_diagram(self, image: Image.Image, t: float):
        x0, x1 = int(OUT_W * 0.07), int(OUT_W * 0.93)
        y0, y1 = int(OUT_H * 0.24), int(OUT_H * 0.77)
        self.panel(image, (x0, y0, x1, y1))
        draw = ImageDraw.Draw(image)
        shot = next(item for item in SHOT_PLAN if item["name"] == "full_diagram")
        reveal = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"] - 0.9, 1e-6))
        plot_x0, plot_x1 = x0 + 30, x1 - 20
        plot_y0, plot_y1 = y0 + 80, y1 - 38
        points = []
        curve_count = max(2, int(len(self.reference_curve) * reveal))
        for _, row in self.reference_curve.iloc[:curve_count].iterrows():
            points.append((self.map_log_z(float(row["redshift"]), plot_x0, plot_x1), self.map_mu(float(row["distance_modulus"]), plot_y0, plot_y1)))
        if len(points) > 1:
            draw.line(points, fill=COLORS["violet"] + (220,), width=3 if not QUICK_MODE else 2)
        bin_count = max(1, int(len(self.binned) * reveal))
        for _, row in self.binned.iloc[:bin_count].iterrows():
            x = self.map_log_z(float(row["redshift"]), plot_x0, plot_x1)
            y = self.map_mu(float(row["distance_modulus"]), plot_y0, plot_y1)
            y_low = self.map_mu(float(row["distance_low"]), plot_y0, plot_y1)
            y_high = self.map_mu(float(row["distance_high"]), plot_y0, plot_y1)
            draw.line((x, y_high, x, y_low), fill=COLORS["cyan"] + (125,), width=2 if not QUICK_MODE else 1)
            r = 5 if not QUICK_MODE else 2.5
            draw.ellipse((x - r, y - r, x + r, y + r), fill=COLORS["gold"] + (235,))
        draw_text(image, "THE FULL DISTANCE–REDSHIFT CURVE", (x0 + 22, y0 + (18 if not QUICK_MODE else 10)), size=22 if not QUICK_MODE else 11, fill=COLORS["violet"] + (238,), bold=True, stroke=1)
        draw_text(image, "gold = real redshift-bin medians // violet = reference ΛCDM", (x0 + 22, y0 + (50 if not QUICK_MODE else 29)), size=16 if not QUICK_MODE else 8, fill=COLORS["white"] + (215,), stroke=1)
        draw_text(image, "REDSHIFT z →", (x1 - 18, y1 - 15), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (190,), anchor="ra", stroke=1)

    def draw_scale_factor(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "scale_factor")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-6))
        self.draw_expanding_grid(image, (OUT_W * 0.50, OUT_H * 0.42), local * 2.5, alpha=130)
        x0, x1 = int(OUT_W * 0.08), int(OUT_W * 0.92)
        y0, y1 = int(OUT_H * 0.58), int(OUT_H * 0.80)
        self.panel(image, (x0, y0, x1, y1), alpha=180)
        draw = ImageDraw.Draw(image)
        visible = max(1, int(len(self.representatives) * clamp(local * 1.15)))
        row_height = (y1 - y0 - 58) / max(len(self.representatives), 1)
        for index, (_, row) in enumerate(self.representatives.iloc[:visible].iterrows()):
            y = y0 + 52 + row_height * index
            z = float(row["redshift"])
            a = float(row["scale_factor_at_emission"])
            bar_x0 = x0 + 170 * OUT_W / 1080
            bar_x1 = x1 - 28
            draw.line((bar_x0, y, bar_x1, y), fill=COLORS["muted"] + (55,), width=7 if not QUICK_MODE else 3)
            draw.line((bar_x0, y, bar_x0 + a * (bar_x1 - bar_x0), y), fill=COLORS["cyan"] + (220,), width=7 if not QUICK_MODE else 3)
            draw_text(image, f"z={z:.2f}", (x0 + 22, int(y)), size=16 if not QUICK_MODE else 8, fill=COLORS["gold"] + (235,), bold=True, anchor="lm", stroke=1)
            draw_text(image, f"a={a:.2f}", (x1 - 24, int(y)), size=15 if not QUICK_MODE else 7, fill=COLORS["white"] + (220,), anchor="rm", stroke=1)
        draw_text(image, "SCALE FACTOR WHEN THE LIGHT LEFT", (x0 + 22, y0 + (18 if not QUICK_MODE else 10)), size=21 if not QUICK_MODE else 10, fill=COLORS["cyan"] + (238,), bold=True, stroke=1)
        draw_text(image, "a = 1 / (1 + z)", (OUT_W // 2, int(OUT_H * 0.51)), size=34 if not QUICK_MODE else 16, fill=COLORS["gold"] + (245,), bold=True, anchor="ma", stroke=1)

    def draw_finale(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "finale")
        local = (t - shot["start"]) / max(shot["end"] - shot["start"], 1e-6)
        center = (OUT_W * 0.5, OUT_H * 0.40)
        self.draw_expanding_grid(image, center, local * 3.0, alpha=170)
        draw = ImageDraw.Draw(image)
        rng = np.random.default_rng(22)
        sample = evenly_subsample(self.frame, 150 if not QUICK_MODE else 70)
        for index, (_, row) in enumerate(sample.iterrows()):
            angle = (float(row["ra_deg"]) if np.isfinite(row["ra_deg"]) else rng.uniform(0, 360)) * math.pi / 180.0
            z_norm = math.log1p(float(row["redshift"])) / math.log1p(self.z_max)
            radius = (65 + 245 * z_norm * (0.75 + 0.25 * local)) * OUT_W / 1080
            x = center[0] + radius * math.cos(angle)
            y = center[1] + 0.72 * radius * math.sin(angle)
            r = 2.2 if not QUICK_MODE else 1.1
            draw.ellipse((x - r, y - r, x + r, y + r), fill=COLORS["white"] + (90,))
        self.panel(image, (int(OUT_W * 0.09), int(OUT_H * 0.63), int(OUT_W * 0.91), int(OUT_H * 0.81)), alpha=178)
        draw_text(image, f"{self.summary['released_light_curve_rows']:,}", (OUT_W // 2, int(OUT_H * 0.665)), size=54 if not QUICK_MODE else 25, fill=COLORS["gold"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "released supernova light curves", (OUT_W // 2, int(OUT_H * 0.715)), size=22 if not QUICK_MODE else 10, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)
        draw_text(image, f"redshift range  {self.summary['redshift_min']:.3f}  →  {self.summary['redshift_max']:.2f}", (OUT_W // 2, int(OUT_H * 0.758)), size=19 if not QUICK_MODE else 9, fill=COLORS["cyan"] + (230,), bold=True, anchor="ma", stroke=1)

    def draw_source_hud(self, image: Image.Image):
        live = bool(self.summary["is_live_observational_data"])
        label = "SOURCE // PANTHEON+ PUBLIC RELEASE" if live else "PREVIEW SOURCE // SYNTHETIC FIXTURE"
        color = COLORS["cyan"] if live else COLORS["gold"]
        draw_text(image, label, (OUT_W - (48 if not QUICK_MODE else 24), 72 if not QUICK_MODE else 36), size=18 if not QUICK_MODE else 9, fill=color + (235,), bold=True, anchor="ra", stroke=1)
        draw_text(image, f"OBJECTS // {self.summary['distinct_supernovae_rendered']:,}", (OUT_W - (48 if not QUICK_MODE else 24), 104 if not QUICK_MODE else 52), size=16 if not QUICK_MODE else 8, fill=COLORS["muted"] + (205,), anchor="ra", stroke=1)
        draw_text(image, f"MAX z // {self.summary['redshift_max']:.2f}", (OUT_W - (48 if not QUICK_MODE else 24), 132 if not QUICK_MODE else 66), size=16 if not QUICK_MODE else 8, fill=COLORS["muted"] + (195,), anchor="ra", stroke=1)

    def draw_titles(self, image: Image.Image, t: float, shot_name: str):
        alpha = int(255 * smoothstep((t - 0.2) / 0.8) * (1.0 - smoothstep((t - (6.7 if not QUICK_MODE else 1.4)) / 0.65)))
        if alpha > 4:
            draw_text(image, "THE UNIVERSE EXPANDS", (56 if not QUICK_MODE else 28, 88 if not QUICK_MODE else 43), size=40 if not QUICK_MODE else 18, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(image, "IN REAL REDSHIFT DATA", (56 if not QUICK_MODE else 28, 136 if not QUICK_MODE else 67), size=40 if not QUICK_MODE else 18, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(image, CONFIG["subtitle"], (58 if not QUICK_MODE else 30, 188 if not QUICK_MODE else 94), size=21 if not QUICK_MODE else 10, fill=COLORS["cyan"] + (min(alpha, 230),), bold=True)
        labels = {
            "intro": "LIGHT STRETCHED BY EXPANDING SPACE",
            "catalogue": "THE REAL CATALOGUE // DISTANCE VERSUS REDSHIFT",
            "local_hubble": "THE NEARBY UNIVERSE // ALMOST LINEAR",
            "full_diagram": "THE DEEP UNIVERSE // THE CURVE EMERGES",
            "scale_factor": "RED SHIFTED LIGHT // AN EARLIER SCALE FACTOR",
            "finale": "THE EXPANSION WRITTEN INTO SUPERNOVA LIGHT",
        }
        if t > (5.1 if not QUICK_MODE else 1.2):
            draw_text(image, labels[shot_name], (56 if not QUICK_MODE else 28, 62 if not QUICK_MODE else 31), size=19 if not QUICK_MODE else 9, fill=COLORS["muted"] + (205,), bold=True, stroke=1)

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        y0 = OUT_H - (244 if not QUICK_MODE else 124)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((44 if not QUICK_MODE else 22, y0, OUT_W - (44 if not QUICK_MODE else 22), y0 + (124 if not QUICK_MODE else 66)), radius=24 if not QUICK_MODE else 12, fill=(2, 6, 15, 176), outline=(80, 190, 228, 66), width=1)
        image.alpha_composite(panel)
        draw_wrapped_text(image, text, (68 if not QUICK_MODE else 34, y0 + (28 if not QUICK_MODE else 14)), OUT_W - (136 if not QUICK_MODE else 68), size=30 if not QUICK_MODE else 14, fill=COLORS["white"] + (245,))

    def draw_hud_noise(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for item in self.hud:
            pulse = 0.5 + 0.5 * math.sin(t * 1.9 + item["phase"])
            if pulse < 0.73:
                continue
            y = (item["y"] + t * 9.0) % OUT_H
            draw.line((item["x"], y, item["x"] + item["length"], y), fill=COLORS["cyan"] + (int(item["a"] * pulse),), width=1)
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
        elif name == "catalogue":
            self.draw_catalogue(image, t)
        elif name == "local_hubble":
            self.draw_local_hubble(image, t)
        elif name == "full_diagram":
            self.draw_full_diagram(image, t)
        elif name == "scale_factor":
            self.draw_scale_factor(image, t)
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

def render_video(scene: RedshiftScene) -> Path:
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
        for t in tqdm(times, desc="Rendering redshift short"):
            writer.append_data(scene.render_frame(float(t)))
    shutil.copyfile(raw_video, final_video)
    print("Final video:", final_video.resolve())
    return final_video


def main():
    print("Loading Pantheon+ distance-redshift data ...")
    frame, source, notes, raw_rows = load_all_data()
    print("Analysing the Hubble diagram ...")
    local_slope, local = fit_local_hubble_slope(frame)
    binned = redshift_bins(frame)
    representatives = choose_representative_objects(frame)
    summary = summarize(frame, source, raw_rows, local_slope, local)
    catalogue_path, summary_path = save_data_products(frame, binned, representatives, summary, notes)
    create_scientific_plots(frame, binned, local, summary)

    print("Data source:", source)
    print("Released light-curve rows:", f"{summary['released_light_curve_rows']:,}")
    print("Distinct supernovae rendered:", f"{summary['distinct_supernovae_rendered']:,}")
    print("Redshift range:", f"{summary['redshift_min']:.5f} to {summary['redshift_max']:.5f}")
    print("Descriptive nearby slope:", f"{local_slope:.2f} km/s/Mpc")
    for note in notes:
        print("Data note:", note)
    print("Data:", catalogue_path.resolve())
    print("Summary:", summary_path.resolve())

    scene = RedshiftScene(frame, binned, representatives, local, summary)
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
