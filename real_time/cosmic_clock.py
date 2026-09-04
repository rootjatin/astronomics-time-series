from __future__ import annotations

"""
Radio Waves — Time Analysis of a Pulsar
=======================================
Cinematic vertical YouTube Short renderer showing how astronomers turn noisy
radio-telescope data into a pulsar timing measurement.


Default numerical demonstration
-------------------------------
The synthetic observation is parameterized after the bright pulsar PSR B0329+54:
    period ~ 0.7145197 s
    dispersion measure ~ 26.7641 pc cm^-3

The dynamic spectrum, noise, individual pulse amplitudes, timing residuals, and
soundtrack are SIMULATED for explanation. They are not raw telescope data.

Optional real/processed dynamic-spectrum CSV
--------------------------------------------
Set PULSAR_RADIO_CSV to a CSV containing:
    time_s, frequency_mhz, intensity
The script pivots those rows into a dynamic spectrum. For best results use a
regular time/frequency grid. The educational overlays still use the configured
period/DM unless you override them with:
    PULSAR_PERIOD_S=...
    PULSAR_DM=...

Quick preview
-------------
    PULSAR_RADIO_SHORT_QUICK=1 python radio_waves_time_analysis_of_a_pulsar.py

Full render
-----------
    python radio_waves_time_analysis_of_a_pulsar.py

Recommended install
-------------------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm

Primary references
------------------
- NASA Science — Pulsars:
  https://science.nasa.gov/mission/hubble/science/science-behind-the-discoveries/hubble-pulsars/
- CSIRO / ATNF Pulsar Catalogue documentation:
  https://www.atnf.csiro.au/research/pulsar/psrcat/psrcat_help.html
- CSIRO Parkes Pulsar Timing Array data products:
  https://data.csiro.au/collection/csiro:41824

"""

import json
import math
import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("PULSAR_RADIO_SHORT_QUICK", "0") == "1"
NO_AUDIO = os.environ.get("PULSAR_RADIO_SHORT_NO_AUDIO", "0") == "1"
RADIO_CSV = os.environ.get("PULSAR_RADIO_CSV", "").strip()

PERIOD_S = float(os.environ.get("PULSAR_PERIOD_S", "0.7145197"))
DM = float(os.environ.get("PULSAR_DM", "26.7641"))

OUTPUT_ROOT = Path("radio_waves_time_analysis_of_a_pulsar_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12.0 if QUICK_MODE else 58.0,
    "output_basename": "radio_waves_time_analysis_of_a_pulsar",
    "title_1": "RADIO WAVES",
    "title_2": "TIME ANALYSIS OF A PULSAR",
    "subtitle": "NOISE → DISPERSION → PERIOD → FOLDED PULSE",
    "soundtrack_sample_rate": 22050 if QUICK_MODE else 44100,
    "grain_strength": 4.1,
    "vignette": 0.50,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
S = OUT_W / 1080.0

COLORS = {
    "black": (1, 2, 7),
    "space": (3, 6, 15),
    "panel": (5, 13, 25),
    "deep": (8, 20, 43),
    "cyan": (82, 240, 255),
    "blue": (65, 154, 255),
    "violet": (151, 96, 255),
    "magenta": (245, 89, 255),
    "amber": (255, 193, 86),
    "red": (255, 77, 94),
    "green": (95, 255, 176),
    "white": (248, 251, 255),
    "muted": (158, 185, 211),
    "grid": (73, 107, 145),
}

FULL_SHOT_PLAN = [
    {"name": "cold_open", "start": 0.0, "end": 6.5},
    {"name": "dispersion", "start": 6.5, "end": 17.0},
    {"name": "dedispersion", "start": 17.0, "end": 27.5},
    {"name": "period_search", "start": 27.5, "end": 38.5},
    {"name": "folding", "start": 38.5, "end": 49.5},
    {"name": "timing", "start": 49.5, "end": 56.0},
    {"name": "finale", "start": 56.0, "end": 58.0},
]

FULL_CAPTIONS = [
    (0.4, 6.2, "A radio telescope can receive a pulsar as a weak repeating signal buried inside noise."),
    (6.8, 16.7, "Interstellar electrons delay lower radio frequencies more than higher ones, stretching one pulse into a curved sweep across the band."),
    (17.2, 27.1, "Astronomers dedisperse the data: each frequency channel is shifted by the delay expected from the pulsar's dispersion measure."),
    (27.8, 38.1, "Now the pulses line up. A period search finds the repetition rate: about 0.71452 seconds for this demonstration."),
    (38.8, 49.1, "Fold hundreds or thousands of rotations on that period, and random noise averages down while the pulse profile adds coherently."),
    (49.8, 55.7, "The exact pulse arrival times can then be compared with a timing model, turning a faint radio source into a remarkably precise clock."),
    (56.1, 57.9, "Radio noise in. A cosmic clock out."),
]

if QUICK_MODE:
    scale = float(CONFIG["duration_s"]) / 58.0
    SHOT_PLAN = [
        {"name": shot["name"], "start": shot["start"] * scale, "end": shot["end"] * scale}
        for shot in FULL_SHOT_PLAN
    ]
    CAPTIONS = [(a * scale, b * scale, text) for a, b, text in FULL_CAPTIONS]
else:
    SHOT_PLAN = FULL_SHOT_PLAN
    CAPTIONS = FULL_CAPTIONS

SOURCE_URLS = {
    "nasa_pulsars": "https://science.nasa.gov/mission/hubble/science/science-behind-the-discoveries/hubble-pulsars/",
    "atnf_catalogue": "https://www.atnf.csiro.au/research/pulsar/psrcat/psrcat_help.html",
    "parkes_timing_data": "https://data.csiro.au/collection/csiro:41824",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(value: float) -> float:
    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


def smootherstep(value: float) -> float:
    x = clamp(value)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def shot_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - float(shot["start"])) / max(float(shot["end"] - shot["start"]), 1e-9))


def get_font(size: int, bold: bool = False, condensed: bool = False):
    candidates: List[str] = []
    if condensed and bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
            "DejaVuSansCondensed-Bold.ttf",
        ]
    if condensed:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
            "DejaVuSansCondensed.ttf",
        ]
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(8, int(size)))
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int,
    fill: Tuple[int, int, int, int],
    bold: bool = False,
    condensed: bool = False,
    anchor: str = "la",
    stroke: int = 2,
) -> None:
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold=bold, condensed=condensed),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(230, fill[3])),
    )


def draw_wrapped(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill: Tuple[int, int, int, int],
    line_spacing: int = 8,
) -> None:
    draw = ImageDraw.Draw(image)
    font = get_font(size)
    words = text.split()
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
        draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 230))
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += (bbox[3] - bbox[1]) + line_spacing


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    hh = ms // 3_600_000
    ms %= 3_600_000
    mm = ms // 60_000
    ms %= 60_000
    ss = ms // 1000
    ms %= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for index, (start, end, text) in enumerate(captions, start=1):
        lines += [str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.7, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


def dispersion_delay_s(freq_mhz: np.ndarray, dm: float, ref_mhz: float) -> np.ndarray:
    # Cold-plasma delay relative to a reference frequency.
    # 4.148808e6 gives milliseconds when frequency is in MHz.
    delay_ms = 4.148808e6 * dm * (freq_mhz ** -2 - ref_mhz ** -2)
    return delay_ms / 1000.0


# -----------------------------------------------------------------------------
# Radio data model
# -----------------------------------------------------------------------------

def simulate_dynamic_spectrum() -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(32954)
    duration = 24.0
    dt = 0.004 if not QUICK_MODE else 0.008
    times = np.arange(0.0, duration, dt)
    freqs = np.linspace(400.0, 800.0, 64 if not QUICK_MODE else 40)
    ref = float(freqs.max())
    delays = dispersion_delay_s(freqs, DM, ref)

    data = rng.normal(0.0, 0.75, (len(freqs), len(times)))
    # Slow receiver bandpass + scintillation-like structure.
    band = 0.75 + 0.25 * np.sin(np.linspace(0.0, 2.2 * math.pi, len(freqs)) + 0.8)
    scint = 0.78 + 0.22 * np.sin(np.linspace(0.0, 5.1 * math.pi, len(freqs)) + 1.5) ** 2
    amp_by_freq = band * scint

    pulse_centers = np.arange(0.75, duration + PERIOD_S, PERIOD_S)
    pulse_jitter = rng.normal(0.0, 0.0045, len(pulse_centers))
    pulse_amp = np.clip(rng.lognormal(mean=0.05, sigma=0.28, size=len(pulse_centers)), 0.55, 2.1)
    width = 0.020

    for fi, delay in enumerate(delays):
        profile = np.zeros_like(times)
        for center, jitter, amp in zip(pulse_centers, pulse_jitter, pulse_amp):
            c = center + jitter + delay
            profile += amp * np.exp(-0.5 * ((times - c) / width) ** 2)
            # Weak trailing component makes the folded profile more interesting.
            profile += 0.28 * amp * np.exp(-0.5 * ((times - (c + 0.048)) / (width * 1.25)) ** 2)
        data[fi] += 3.0 * amp_by_freq[fi] * profile

    # Occasional broadband impulsive RFI, deliberately not periodic.
    for _ in range(8):
        c = float(rng.uniform(0.2, duration - 0.2))
        w = float(rng.uniform(0.002, 0.010))
        data += rng.uniform(0.4, 1.0) * np.exp(-0.5 * ((times - c) / w) ** 2)[None, :]

    return times, freqs, data.astype(np.float32), "synthetic_psr_b0329_like"


def load_csv_dynamic_spectrum(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    df = pd.read_csv(path)
    columns = {c.lower().strip(): c for c in df.columns}
    required = ["time_s", "frequency_mhz", "intensity"]
    if not all(name in columns for name in required):
        raise ValueError("CSV must contain time_s, frequency_mhz, intensity")
    tcol, fcol, icol = (columns[name] for name in required)
    clean = df[[tcol, fcol, icol]].dropna().copy()
    clean[tcol] = clean[tcol].astype(float)
    clean[fcol] = clean[fcol].astype(float)
    clean[icol] = clean[icol].astype(float)
    pivot = clean.pivot_table(index=fcol, columns=tcol, values=icol, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    times = pivot.columns.to_numpy(dtype=float)
    freqs = pivot.index.to_numpy(dtype=float)
    data = pivot.to_numpy(dtype=float)
    # Fill sparse holes with row medians so visualization remains stable.
    for i in range(data.shape[0]):
        row = data[i]
        med = np.nanmedian(row) if np.any(np.isfinite(row)) else 0.0
        row[~np.isfinite(row)] = med
    return times, freqs, data.astype(np.float32), "user_csv"


def load_radio_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, List[str]]:
    notes: List[str] = []
    if RADIO_CSV:
        try:
            times, freqs, data, source = load_csv_dynamic_spectrum(RADIO_CSV)
            notes.append(f"Loaded dynamic spectrum from {RADIO_CSV}")
            return times, freqs, data, source, notes
        except Exception as exc:
            notes.append(f"Could not load PULSAR_RADIO_CSV; using synthetic demonstration: {exc}")
    times, freqs, data, source = simulate_dynamic_spectrum()
    notes.append("Using deterministic synthetic radio data parameterized after PSR B0329+54")
    return times, freqs, data, source, notes


def dedisperse(times: np.ndarray, freqs: np.ndarray, data: np.ndarray, dm: float) -> np.ndarray:
    if len(times) < 2:
        return data.copy()
    dt = float(np.median(np.diff(times)))
    ref = float(np.max(freqs))
    delays = dispersion_delay_s(freqs, dm, ref)
    out = np.zeros_like(data)
    for i, delay in enumerate(delays):
        shift = int(round(delay / dt))
        if shift <= 0:
            out[i] = data[i]
        elif shift < data.shape[1]:
            out[i, :-shift] = data[i, shift:]
            out[i, -shift:] = np.nanmedian(data[i])
        else:
            out[i] = np.nanmedian(data[i])
    return out


def normalize_time_series(data: np.ndarray) -> np.ndarray:
    series = np.mean(data, axis=0)
    series = series - np.median(series)
    scale = np.percentile(np.abs(series), 98)
    if scale <= 1e-9:
        scale = 1.0
    return np.clip(series / scale, -1.3, 1.8)


def period_search(times: np.ndarray, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    dt = float(np.median(np.diff(times)))
    centered = series - np.mean(series)
    window = np.hanning(len(centered))
    spec = np.fft.rfft(centered * window)
    freqs_hz = np.fft.rfftfreq(len(centered), d=dt)
    power = np.abs(spec) ** 2
    mask = (freqs_hz > 0.7) & (freqs_hz < 3.0)
    if not np.any(mask):
        return np.array([PERIOD_S]), np.array([1.0]), PERIOD_S
    f = freqs_hz[mask]
    p = power[mask]
    # Fundamental can be weaker than a harmonic, so score candidate periods by
    # folding coherence around a tight grid centered on the demonstration period.
    grid = np.linspace(max(0.45, PERIOD_S - 0.12), PERIOD_S + 0.12, 220)
    coherence = []
    for trial in grid:
        phase = (times % trial) / trial
        bins = np.linspace(0.0, 1.0, 33)
        means = []
        for a, b in zip(bins[:-1], bins[1:]):
            sel = (phase >= a) & (phase < b)
            means.append(float(np.mean(series[sel])) if np.any(sel) else 0.0)
        means_arr = np.asarray(means)
        coherence.append(float(np.var(means_arr)))
    coherence_arr = np.asarray(coherence)
    best_period = float(grid[int(np.argmax(coherence_arr))])
    return grid, coherence_arr, best_period


def folded_profile(times: np.ndarray, series: np.ndarray, period: float, bins: int = 96) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = (times % period) / period
    edges = np.linspace(0.0, 1.0, bins + 1)
    profile = np.zeros(bins)
    counts = np.zeros(bins)
    for i in range(bins):
        mask = (phase >= edges[i]) & (phase < edges[i + 1])
        if np.any(mask):
            profile[i] = float(np.mean(series[mask]))
            counts[i] = int(np.sum(mask))
    x = (edges[:-1] + edges[1:]) / 2.0
    profile -= np.min(profile)
    if np.max(profile) > 1e-9:
        profile /= np.max(profile)
    return x, profile, counts


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def spectral_color(v: float) -> Tuple[int, int, int]:
    v = clamp(v)
    stops = [
        (0.00, COLORS["black"]),
        (0.18, (11, 26, 64)),
        (0.40, COLORS["violet"]),
        (0.62, COLORS["magenta"]),
        (0.80, COLORS["amber"]),
        (1.00, COLORS["white"]),
    ]
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if a <= v <= b:
            t = (v - a) / max(b - a, 1e-9)
            return tuple(int(round(lerp(x, y, t))) for x, y in zip(ca, cb))
    return stops[-1][1]


def dynamic_spectrum_image(data: np.ndarray, width: int, height: int) -> Image.Image:
    lo, hi = np.percentile(data, [5.0, 99.4])
    norm = np.clip((data - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    norm = np.sqrt(norm)
    lut = np.array([spectral_color(i / 255.0) for i in range(256)], dtype=np.uint8)
    indices = np.clip((norm * 255).astype(np.int32), 0, 255)
    rgb = lut[indices]
    rgb = rgb[::-1, :, :]  # high frequency at top
    image = Image.fromarray(rgb, "RGB").convert("RGBA")
    return image.resize((width, height), Image.Resampling.BILINEAR)


def rounded_panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 205, outline_alpha: int = 85) -> None:
    overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius = max(8, int(24 * S))
    draw.rounded_rectangle(box, radius=radius, fill=COLORS["panel"] + (alpha,), outline=COLORS["grid"] + (outline_alpha,), width=max(1, int(2 * S)))
    image.alpha_composite(overlay)


def draw_axes(image: Image.Image, box: Tuple[int, int, int, int], xlabel: str = "TIME", ylabel: str = "") -> None:
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(image)
    draw.line((x0, y1, x1, y1), fill=COLORS["grid"] + (130,), width=max(1, int(2 * S)))
    draw.line((x0, y0, x0, y1), fill=COLORS["grid"] + (130,), width=max(1, int(2 * S)))
    if xlabel:
        draw_text(image, xlabel, ((x0 + x1) // 2, y1 + int(35 * S)), max(10, int(19 * S)), COLORS["muted"] + (220,), bold=True, condensed=True, anchor="ma", stroke=1)
    if ylabel:
        draw_text(image, ylabel, (x0 + int(10 * S), y0 + int(10 * S)), max(9, int(17 * S)), COLORS["muted"] + (220,), bold=True, condensed=True, stroke=1)


def draw_series(image: Image.Image, box: Tuple[int, int, int, int], values: np.ndarray, color: Tuple[int, int, int], reveal: float = 1.0, baseline: float = 0.55, scale_y: float = 0.42) -> None:
    x0, y0, x1, y1 = box
    n = max(2, int(len(values) * clamp(reveal)))
    subset = values[:n]
    if len(subset) < 2:
        return
    points = []
    for i, v in enumerate(subset):
        x = lerp(x0, x1, i / max(len(values) - 1, 1))
        y = lerp(y0, y1, baseline - float(v) * scale_y)
        points.append((x, y))
    glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.line(points, fill=color + (90,), width=max(2, int(8 * S)))
    glow = glow.filter(ImageFilter.GaussianBlur(max(2, int(7 * S))))
    image.alpha_composite(glow)
    ImageDraw.Draw(image).line(points, fill=color + (245,), width=max(1, int(3 * S)))


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------

class RadioPulsarScene:
    def __init__(self, times: np.ndarray, freqs: np.ndarray, raw: np.ndarray, source: str):
        self.times = times
        self.freqs = freqs
        self.raw = raw
        self.source = source
        self.dedispersed = dedisperse(times, freqs, raw, DM)
        self.raw_series = normalize_time_series(raw)
        self.dedisp_series = normalize_time_series(self.dedispersed)
        self.period_grid, self.period_score, self.best_period = period_search(times, self.dedisp_series)
        self.phase, self.profile, self.profile_counts = folded_profile(times, self.dedisp_series, self.best_period)
        self.raw_spec = dynamic_spectrum_image(raw, int(900 * S), int(560 * S))
        self.dedisp_spec = dynamic_spectrum_image(self.dedispersed, int(900 * S), int(560 * S))
        self.rng = np.random.default_rng(8128)
        self.stars = [(float(self.rng.uniform(0, OUT_W)), float(self.rng.uniform(0, OUT_H)), float(self.rng.uniform(0.4, 1.8)), float(self.rng.uniform(12, 85))) for _ in range(180 if not QUICK_MODE else 90)]
        self.residual_x = np.linspace(0.0, 1.0, 26)
        self.residual_us = self.rng.normal(0.0, 24.0, len(self.residual_x)) + 7.0 * np.sin(np.linspace(0, 2.2 * math.pi, len(self.residual_x)))

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
        top = np.array(COLORS["space"], dtype=float)
        bottom = np.array(COLORS["black"], dtype=float)
        for y in range(OUT_H):
            u = y / max(OUT_H - 1, 1)
            rgb = (top * (1.0 - u) + bottom * u).astype(np.uint8)
            arr[y, :, :3] = rgb
            arr[y, :, 3] = 255
        image = Image.fromarray(arr, "RGBA")
        draw = ImageDraw.Draw(image)
        for x, y, r, a in self.stars:
            twinkle = 0.45 + 0.55 * math.sin(t * 0.6 + x * 0.013 + y * 0.007) ** 2
            rr = max(0.3, r * S)
            draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=(190, 220, 255, int(a * twinkle)))
        return image

    def top_titles(self, image: Image.Image, t: float, name: str) -> None:
        fade = smoothstep(t / max(1.2, float(CONFIG["duration_s"]) * 0.04))
        draw_text(image, CONFIG["title_1"], (int(64 * S), int(78 * S)), max(13, int(25 * S)), COLORS["cyan"] + (int(240 * fade),), bold=True, condensed=True, stroke=1)
        draw_text(image, CONFIG["title_2"], (int(64 * S), int(120 * S)), max(20, int(43 * S)), COLORS["white"] + (int(250 * fade),), bold=True, condensed=True, stroke=2)
        draw_text(image, CONFIG["subtitle"], (int(64 * S), int(177 * S)), max(9, int(18 * S)), COLORS["muted"] + (int(215 * fade),), bold=True, condensed=True, stroke=1)
        labels = {
            "cold_open": "01 / RAW RADIO TIME SERIES",
            "dispersion": "02 / FREQUENCY-DEPENDENT DELAY",
            "dedispersion": "03 / DEDISPERSE",
            "period_search": "04 / FIND THE PERIOD",
            "folding": "05 / FOLD THE ROTATIONS",
            "timing": "06 / TIME OF ARRIVAL",
            "finale": "RADIO NOISE → COSMIC CLOCK",
        }
        draw_text(image, labels.get(name, ""), (OUT_W - int(62 * S), int(88 * S)), max(9, int(17 * S)), COLORS["amber"] + (220,), bold=True, condensed=True, anchor="ra", stroke=1)

    def source_hud(self, image: Image.Image) -> None:
        label = "USER RADIO CSV" if self.source == "user_csv" else "SIMULATED RADIO DATA // PSR B0329+54 PARAMETERS"
        draw_text(image, label, (int(58 * S), OUT_H - int(55 * S)), max(8, int(14 * S)), COLORS["muted"] + (180,), bold=True, condensed=True, stroke=1)

    def caption(self, image: Image.Image, t: float) -> None:
        text = None
        for a, b, value in CAPTIONS:
            if a <= t < b:
                text = value
                break
        if not text:
            return
        box = (int(55 * S), OUT_H - int(330 * S), OUT_W - int(55 * S), OUT_H - int(95 * S))
        rounded_panel(image, box, alpha=185, outline_alpha=55)
        draw_wrapped(image, text, (box[0] + int(28 * S), box[1] + int(26 * S)), box[2] - box[0] - int(56 * S), max(13, int(25 * S)), COLORS["white"] + (245,), line_spacing=max(3, int(8 * S)))

    def cold_open(self, image: Image.Image, t: float, shot: Dict[str, Any]) -> None:
        p = shot_progress(t, shot)
        box = (int(85 * S), int(340 * S), OUT_W - int(85 * S), int(1040 * S))
        rounded_panel(image, box, alpha=180)
        draw_text(image, "WHAT DOES A PULSAR LOOK LIKE TO A RADIO TELESCOPE?", (OUT_W // 2, int(300 * S)), max(13, int(25 * S)), COLORS["white"] + (240,), bold=True, condensed=True, anchor="ma", stroke=1)
        graph = (box[0] + int(40 * S), box[1] + int(90 * S), box[2] - int(40 * S), box[3] - int(100 * S))
        draw_axes(image, graph, "TIME", "RADIO INTENSITY")
        # Start with almost pure noise, then reveal periodic spikes hidden within it.
        signal = self.raw_series.copy()
        if len(signal) > 1800:
            signal = signal[:1800]
        mix = smoothstep((p - 0.30) / 0.55)
        noise = np.sin(np.linspace(0, 71, len(signal))) * 0.07
        show = signal * (0.35 + 0.65 * mix) + noise * (1.0 - mix)
        draw_series(image, graph, show, COLORS["cyan"], reveal=min(1.0, p * 1.35), baseline=0.56, scale_y=0.28)
        draw_text(image, "IT LOOKS LIKE NOISE.", (OUT_W // 2, int(1120 * S)), max(22, int(49 * S)), COLORS["white"] + (250,), bold=True, condensed=True, anchor="ma")
        if p > 0.62:
            draw_text(image, "BUT SOMETHING REPEATS.", (OUT_W // 2, int(1190 * S)), max(14, int(28 * S)), COLORS["amber"] + (int(255 * smoothstep((p - .62)/.2)),), bold=True, condensed=True, anchor="ma")

    def draw_spectrum_panel(self, image: Image.Image, spectrum: Image.Image, title: str, sub: str, reveal: float = 1.0) -> Tuple[int, int, int, int]:
        box = (int(88 * S), int(315 * S), OUT_W - int(88 * S), int(1085 * S))
        rounded_panel(image, box, alpha=195)
        draw_text(image, title, (box[0] + int(32 * S), box[1] + int(30 * S)), max(15, int(29 * S)), COLORS["white"] + (245,), bold=True, condensed=True, stroke=1)
        draw_text(image, sub, (box[0] + int(32 * S), box[1] + int(72 * S)), max(9, int(17 * S)), COLORS["muted"] + (220,), bold=True, condensed=True, stroke=1)
        sx = box[0] + int(38 * S)
        sy = box[1] + int(125 * S)
        sw = box[2] - box[0] - int(76 * S)
        sh = int(545 * S)
        crop_w = max(2, int(spectrum.width * clamp(reveal)))
        revealed = spectrum.crop((0, 0, crop_w, spectrum.height)).resize((max(2, int(sw * clamp(reveal))), sh), Image.Resampling.BILINEAR)
        image.alpha_composite(revealed, (sx, sy))
        draw = ImageDraw.Draw(image)
        draw.rectangle((sx, sy, sx + sw, sy + sh), outline=COLORS["grid"] + (135,), width=max(1, int(2*S)))
        draw_text(image, "800 MHz", (sx - int(16*S), sy + int(12*S)), max(8, int(14*S)), COLORS["muted"] + (200,), anchor="ra", stroke=1)
        draw_text(image, "400 MHz", (sx - int(16*S), sy + sh - int(4*S)), max(8, int(14*S)), COLORS["muted"] + (200,), anchor="ra", stroke=1)
        draw_text(image, "TIME →", (sx + sw, sy + sh + int(36*S)), max(8, int(15*S)), COLORS["muted"] + (210,), bold=True, condensed=True, anchor="ra", stroke=1)
        return (sx, sy, sx + sw, sy + sh)

    def dispersion(self, image: Image.Image, t: float, shot: Dict[str, Any]) -> None:
        p = smootherstep(shot_progress(t, shot))
        spec_box = self.draw_spectrum_panel(image, self.raw_spec, "ONE PULSE ARRIVES AT DIFFERENT TIMES", f"DISPERSION MEASURE  {DM:.2f} pc cm⁻³", reveal=min(1.0, p * 1.25))
        x0, y0, x1, y1 = spec_box
        # Overlay schematic delay curve for one pulse.
        if p > 0.2:
            freqs = np.linspace(float(self.freqs.max()), float(self.freqs.min()), 80)
            delays = dispersion_delay_s(freqs, DM, float(self.freqs.max()))
            delays /= max(float(np.max(delays)), 1e-9)
            points = []
            for f, d in zip(freqs, delays):
                yy = lerp(y0, y1, (float(self.freqs.max()) - f) / max(float(self.freqs.max() - self.freqs.min()), 1e-9))
                xx = lerp(x0 + 0.20*(x1-x0), x0 + 0.46*(x1-x0), d)
                points.append((xx, yy))
            ImageDraw.Draw(image).line(points, fill=COLORS["white"] + (215,), width=max(2, int(4*S)))
            draw_text(image, "LOWER FREQUENCY = LATER ARRIVAL", (OUT_W//2, int(1145*S)), max(14, int(28*S)), COLORS["amber"] + (245,), bold=True, condensed=True, anchor="ma")
        if p > 0.65:
            delay = float(np.max(dispersion_delay_s(np.array([self.freqs.min()]), DM, float(self.freqs.max()))))
            draw_text(image, f"~{delay*1000:.0f} ms DELAY ACROSS 400–800 MHz", (OUT_W//2, int(1210*S)), max(12, int(23*S)), COLORS["white"] + (235,), bold=True, condensed=True, anchor="ma")

    def dedispersion(self, image: Image.Image, t: float, shot: Dict[str, Any]) -> None:
        p = smootherstep(shot_progress(t, shot))
        blend = Image.blend(self.raw_spec, self.dedisp_spec, p)
        self.draw_spectrum_panel(image, blend, "SHIFT EACH FREQUENCY CHANNEL", "DEDISPERSION REMOVES THE PLASMA DELAY", reveal=1.0)
        # Channel shift HUD.
        hud = (int(145*S), int(1115*S), OUT_W-int(145*S), int(1305*S))
        rounded_panel(image, hud, alpha=178)
        draw_text(image, "APPLY  Δt ∝ DM × ν⁻²", (OUT_W//2, hud[1]+int(42*S)), max(18, int(34*S)), COLORS["cyan"] + (250,), bold=True, condensed=True, anchor="ma")
        draw_text(image, "THE CURVE STRAIGHTENS INTO ONE VERTICAL PULSE", (OUT_W//2, hud[1]+int(105*S)), max(11, int(21*S)), COLORS["white"] + (235,), bold=True, condensed=True, anchor="ma")

    def period_search_scene(self, image: Image.Image, t: float, shot: Dict[str, Any]) -> None:
        p = smootherstep(shot_progress(t, shot))
        top_box = (int(88*S), int(315*S), OUT_W-int(88*S), int(760*S))
        bottom_box = (int(88*S), int(815*S), OUT_W-int(88*S), int(1245*S))
        rounded_panel(image, top_box, alpha=192)
        rounded_panel(image, bottom_box, alpha=192)
        draw_text(image, "DEDISPERSED RADIO INTENSITY", (top_box[0]+int(28*S), top_box[1]+int(25*S)), max(13, int(25*S)), COLORS["white"]+(245,), bold=True, condensed=True, stroke=1)
        g1 = (top_box[0]+int(35*S), top_box[1]+int(90*S), top_box[2]-int(35*S), top_box[3]-int(40*S))
        draw_series(image, g1, self.dedisp_series[:min(1600,len(self.dedisp_series))], COLORS["cyan"], reveal=min(1.0,p*1.35), baseline=.60, scale_y=.28)
        draw_text(image, "SEARCH TRIAL PERIODS", (bottom_box[0]+int(28*S), bottom_box[1]+int(24*S)), max(13, int(25*S)), COLORS["white"]+(245,), bold=True, condensed=True, stroke=1)
        g2 = (bottom_box[0]+int(45*S), bottom_box[1]+int(95*S), bottom_box[2]-int(40*S), bottom_box[3]-int(65*S))
        scores = self.period_score - np.min(self.period_score)
        if np.max(scores)>0: scores=scores/np.max(scores)
        draw_series(image, g2, scores, COLORS["amber"], reveal=min(1.0,p*1.3), baseline=.92, scale_y=.75)
        if p>0.56:
            idx=int(np.argmax(scores)); x=lerp(g2[0],g2[2],idx/max(len(scores)-1,1))
            d=ImageDraw.Draw(image); d.line((x,g2[1],x,g2[3]),fill=COLORS["white"]+(190,),width=max(1,int(2*S)))
            draw_text(image, f"P = {self.best_period:.6f} s", (OUT_W//2, int(1290*S)), max(22,int(46*S)), COLORS["amber"]+(255,), bold=True, condensed=True, anchor="ma")
            draw_text(image, f"ROTATION RATE ≈ {1.0/self.best_period:.3f} Hz", (OUT_W//2, int(1360*S)), max(11,int(22*S)), COLORS["white"]+(230,), bold=True, condensed=True, anchor="ma")

    def folding_scene(self, image: Image.Image, t: float, shot: Dict[str, Any]) -> None:
        p = smootherstep(shot_progress(t, shot))
        box = (int(90*S), int(315*S), OUT_W-int(90*S), int(1260*S))
        rounded_panel(image, box, alpha=192)
        draw_text(image, "FOLD MANY ROTATIONS ON THE SAME PERIOD", (OUT_W//2, box[1]+int(45*S)), max(14,int(27*S)), COLORS["white"]+(245,), bold=True, condensed=True, anchor="ma")
        # Stack several noisy pulse rows that progressively converge to an average.
        rows = 9 if not QUICK_MODE else 7
        left=box[0]+int(48*S); right=box[2]-int(48*S); top=box[1]+int(125*S); row_h=int(62*S)
        rng=np.random.default_rng(97)
        base=np.interp(np.linspace(0,1,160), self.phase, self.profile)
        for r in range(rows):
            reveal=clamp(p*1.35-r/(rows*1.1))
            if reveal<=0: continue
            noise=rng.normal(0,0.24,160)
            individual=np.clip(base*(0.6+0.6*rng.random())+noise,-.35,1.35)
            y0=top+r*row_h; g=(left,y0,right,y0+int(48*S))
            draw_series(image,g,individual,COLORS["blue"],reveal=reveal,baseline=.72,scale_y=.38)
        avg_box=(left,top+rows*row_h+int(35*S),right,top+rows*row_h+int(230*S))
        ImageDraw.Draw(image).rounded_rectangle(avg_box,radius=max(8,int(18*S)),outline=COLORS["amber"]+(130,),width=max(1,int(2*S)),fill=(4,10,22,120))
        draw_series(image,avg_box,base,COLORS["amber"],reveal=clamp((p-.35)/.5),baseline=.88,scale_y=.70)
        draw_text(image,"AVERAGE PULSE PROFILE",(OUT_W//2,avg_box[3]+int(40*S)),max(12,int(23*S)),COLORS["amber"]+(245,),bold=True,condensed=True,anchor="ma")
        if p>.72:
            rotations=int(round((self.times[-1]-self.times[0])/self.best_period))
            draw_text(image,f"{rotations} ROTATIONS IN THIS SHORT DEMO",(OUT_W//2,int(1390*S)),max(10,int(20*S)),COLORS["muted"]+(220,),bold=True,condensed=True,anchor="ma")

    def timing_scene(self, image: Image.Image, t: float, shot: Dict[str, Any]) -> None:
        p = smootherstep(shot_progress(t, shot))
        box=(int(92*S),int(330*S),OUT_W-int(92*S),int(1175*S))
        rounded_panel(image,box,alpha=194)
        draw_text(image,"PULSE TIMES OF ARRIVAL",(OUT_W//2,box[1]+int(48*S)),max(18,int(35*S)),COLORS["white"]+(245,),bold=True,condensed=True,anchor="ma")
        draw_text(image,"OBSERVED − TIMING MODEL",(OUT_W//2,box[1]+int(98*S)),max(10,int(19*S)),COLORS["muted"]+(220,),bold=True,condensed=True,anchor="ma")
        g=(box[0]+int(60*S),box[1]+int(175*S),box[2]-int(48*S),box[3]-int(100*S))
        d=ImageDraw.Draw(image)
        ymid=(g[1]+g[3])/2
        for frac in np.linspace(0,1,5):
            yy=lerp(g[1],g[3],frac); d.line((g[0],yy,g[2],yy),fill=COLORS["grid"]+(48,),width=1)
        d.line((g[0],ymid,g[2],ymid),fill=COLORS["white"]+(120,),width=max(1,int(2*S)))
        n=max(1,int(len(self.residual_x)*min(1,p*1.2)))
        for xfrac,res in zip(self.residual_x[:n],self.residual_us[:n]):
            x=lerp(g[0],g[2],float(xfrac)); y=ymid-res/75.0*(g[3]-g[1])/2
            rr=max(2,int(6*S)); d.ellipse((x-rr,y-rr,x+rr,y+rr),fill=COLORS["cyan"]+(235,),outline=COLORS["white"]+(120,))
        draw_text(image,"EARLIER",(g[0],g[1]-int(30*S)),max(8,int(15*S)),COLORS["muted"]+(205,),bold=True,condensed=True,stroke=1)
        draw_text(image,"LATER",(g[0],g[3]+int(18*S)),max(8,int(15*S)),COLORS["muted"]+(205,),bold=True,condensed=True,stroke=1)
        if p>.55:
            draw_text(image,"THE PULSE BECOMES A CLOCK TICK",(OUT_W//2,int(1260*S)),max(18,int(35*S)),COLORS["amber"]+(255,),bold=True,condensed=True,anchor="ma")

    def finale(self, image: Image.Image, t: float, shot: Dict[str, Any]) -> None:
        p=smoothstep(shot_progress(t,shot))
        # Large pulse expanding from center.
        cx,cy=OUT_W//2,int(760*S)
        overlay=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); d=ImageDraw.Draw(overlay)
        for k in range(5):
            r=(35+130*k+180*p)*S
            alpha=int((85-12*k)*(1-p*0.5))
            d.ellipse((cx-r,cy-r,cx+r,cy+r),outline=COLORS["cyan"]+(max(0,alpha),),width=max(1,int(3*S)))
        overlay=overlay.filter(ImageFilter.GaussianBlur(max(1,int(3*S)))); image.alpha_composite(overlay)
        draw_text(image,"RADIO NOISE",(OUT_W//2,int(540*S)),max(24,int(54*S)),COLORS["muted"]+(225,),bold=True,condensed=True,anchor="ma")
        draw_text(image,"↓",(OUT_W//2,int(625*S)),max(25,int(58*S)),COLORS["cyan"]+(245,),bold=True,anchor="ma")
        draw_text(image,"COSMIC CLOCK",(OUT_W//2,int(880*S)),max(30,int(70*S)),COLORS["white"]+(255,),bold=True,condensed=True,anchor="ma")
        draw_text(image,f"PERIOD  {self.best_period:.6f} s",(OUT_W//2,int(980*S)),max(14,int(28*S)),COLORS["amber"]+(245,),bold=True,condensed=True,anchor="ma")

    def film_texture(self, image: Image.Image, t: float) -> None:
        arr=np.asarray(image.convert("RGB"),dtype=np.float32)
        rng=np.random.default_rng(int(t*1000)+123)
        noise=rng.normal(0.0,float(CONFIG["grain_strength"]),arr.shape[:2])[:,:,None]
        arr=np.clip(arr+noise,0,255)
        arr*=VIGNETTE[:,:,None]
        out=Image.fromarray(arr.astype(np.uint8),"RGB").convert("RGBA")
        image.paste(out)
        overlay=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); d=ImageDraw.Draw(overlay)
        scan=int((t*190)%(OUT_H+140))-70
        d.rectangle((0,scan,OUT_W,scan+int(38*S)),fill=COLORS["cyan"]+(5,))
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot=get_shot(t); name=str(shot["name"])
        image=self.background(t)
        if name=="cold_open": self.cold_open(image,t,shot)
        elif name=="dispersion": self.dispersion(image,t,shot)
        elif name=="dedispersion": self.dedispersion(image,t,shot)
        elif name=="period_search": self.period_search_scene(image,t,shot)
        elif name=="folding": self.folding_scene(image,t,shot)
        elif name=="timing": self.timing_scene(image,t,shot)
        else: self.finale(image,t,shot)
        self.top_titles(image,t,name)
        if name!="finale": self.caption(image,t)
        self.source_hud(image)
        self.film_texture(image,t)
        return np.asarray(image.convert("RGB"))


# -----------------------------------------------------------------------------
# Data products, soundtrack, video
# -----------------------------------------------------------------------------

def save_data_products(scene: RadioPulsarScene, notes: Sequence[str]) -> Tuple[Path, Path]:
    csv_path=DATA_ROOT/"pulsar_radio_time_analysis.csv"
    pd.DataFrame({
        "time_s": scene.times,
        "raw_band_average": scene.raw_series,
        "dedispersed_band_average": scene.dedisp_series,
        "phase_at_best_period": (scene.times % scene.best_period)/scene.best_period,
    }).to_csv(csv_path,index=False)
    summary_path=DATA_ROOT/"summary.json"
    summary={
        "title": CONFIG["title_2"],
        "data_source": scene.source,
        "configured_period_s": PERIOD_S,
        "configured_dm_pc_cm3": DM,
        "best_period_s_from_demo_search": scene.best_period,
        "frequency_min_mhz": float(np.min(scene.freqs)),
        "frequency_max_mhz": float(np.max(scene.freqs)),
        "samples": int(len(scene.times)),
        "channels": int(len(scene.freqs)),
        "notes": list(notes),
        "scientific_caveat": "Synthetic mode demonstrates pulsar time-analysis concepts; it is not raw telescope data. User CSV mode visualizes supplied samples but does not replace a full professional pulsar timing pipeline.",
        "source_urls": SOURCE_URLS,
    }
    summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return csv_path,summary_path


def gaussian_envelope(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5*((times-center)/max(width,1e-6))**2)


def generate_soundtrack(path: Path, scene: RadioPulsarScene) -> Path:
    sr=int(CONFIG["soundtrack_sample_rate"]); duration=float(CONFIG["duration_s"])
    n=int(round(sr*duration)); t=np.arange(n,dtype=np.float64)/sr
    rng=np.random.default_rng(901)
    audio=0.035*rng.normal(0,1,n)
    audio+=0.065*np.sin(math.tau*36*t+0.5*np.sin(math.tau*.08*t))
    audio+=0.035*np.sin(math.tau*57*t+1.1)
    # Radio chirps during dispersion scene.
    disp=next(s for s in SHOT_PLAN if s["name"]=="dispersion")
    for frac in np.linspace(.12,.90,6 if not QUICK_MODE else 4):
        c=lerp(float(disp["start"]),float(disp["end"]),float(frac))
        env=gaussian_envelope(t,c,.18 if not QUICK_MODE else .10)
        chirp=np.sin(math.tau*(380*t-22*t*t/duration)+frac*3)
        audio+=.035*env*chirp
    # Repeating pulsar ticks after dedispersion.
    start=next(s for s in SHOT_PLAN if s["name"]=="period_search")["start"]
    end=next(s for s in SHOT_PLAN if s["name"]=="timing")["end"]
    tick_period=max(.08, scene.best_period*(float(CONFIG["duration_s"])/58.0))
    c=float(start)
    while c<float(end):
        env=gaussian_envelope(t,c,.014 if not QUICK_MODE else .010)
        audio+=.105*env*np.sin(math.tau*820*t)
        c+=tick_period
    # Reveal hit.
    final=next(s for s in SHOT_PLAN if s["name"]=="finale")
    c=float(final["start"])+.20*(float(final["end"])-float(final["start"]))
    env=gaussian_envelope(t,c,.32)
    audio+=.12*env*np.sin(math.tau*62*t)
    intro=np.clip(t/max(1.0,duration*.06),0,1); outro=np.clip((duration-t)/max(.8,duration*.035),0,1)
    audio*=np.minimum(intro,outro)
    peak=max(float(np.max(np.abs(audio))),1e-9); audio=np.clip(audio/peak*.88,-1,1)
    pcm=(audio*32767).astype(np.int16)
    with wave.open(str(path),"wb") as h:
        h.setnchannels(1); h.setsampwidth(2); h.setframerate(sr); h.writeframes(pcm.tobytes())
    return path


def find_ffmpeg() -> Optional[str]:
    if imageio_ffmpeg is not None:
        try: return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception: pass
    return shutil.which("ffmpeg")


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    ffmpeg=find_ffmpeg()
    if not ffmpeg: return False
    cmd=[ffmpeg,"-y","-i",str(video_path),"-i",str(audio_path),"-c:v","copy","-c:a","aac","-b:a","192k","-shortest",str(output_path)]
    try:
        subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return output_path.exists() and output_path.stat().st_size>0
    except Exception:
        return False


def render_video(scene: RadioPulsarScene) -> Path:
    srt=OUTPUT_ROOT/f"{CONFIG['output_basename']}.srt"; write_srt(CAPTIONS,srt)
    raw=OUTPUT_ROOT/f"{CONFIG['output_basename']}_silent.mp4"
    final=OUTPUT_ROOT/f"{CONFIG['output_basename']}_final.mp4"
    audio=OUTPUT_ROOT/f"{CONFIG['output_basename']}_cinematic.wav"
    frames=int(round(float(CONFIG["duration_s"])*int(CONFIG["fps"])))
    times=np.arange(frames)/int(CONFIG["fps"])
    print("Subtitle sidecar:",srt.resolve())
    print(f"Rendering {frames:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(raw,fps=int(CONFIG["fps"]),codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=None) as writer:
        for tt in tqdm(times,desc="Rendering pulsar radio timing short"):
            writer.append_data(scene.render_frame(float(tt)))
    if NO_AUDIO:
        shutil.copyfile(raw,final); return final
    generate_soundtrack(audio,scene)
    if mux_audio(raw,audio,final): return final
    shutil.copyfile(raw,final); return final



