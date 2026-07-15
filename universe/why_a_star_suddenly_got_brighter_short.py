# # A Star Suddenly Got Brighter — V462 Lupi Data-Driven YouTube Short
#
# Creates a vertical 1080x1920 cinematic astronomy Short about Nova V462 Lupi.
#
# Real published observations compiled in this renderer:
# - GOTO pre-eruption and early-rise L-band photometry,
# - ASAS-SN discovery photometry,
# - published visual/V-band rise and peak summaries.
#
# Story:
# A faint binary system in Lupus brightened rapidly in June 2025 and became
# visible to the unaided eye. The event was a classical nova: gas accumulated
# on a white dwarf until a runaway thermonuclear reaction erupted on its surface.
# The white dwarf was not destroyed. A nova is not a supernova.
#
# Scientific notes:
# - Published points use different optical passbands (GOTO L, Sloan g, V/visual).
#   The plotted sequence preserves the reported values and labels every band.
# - Lines between observations are cinematic interpolation, not additional data.
# - The binary, gas stream, ejecta, star field and camera are explanatory artwork.
# - The sky coordinates and listed observation values are data-driven.
#
# Install:
#     pip install numpy pandas pillow imageio imageio-ffmpeg tqdm
#
# Render final video:
#     python a_star_suddenly_got_brighter_short.py
#
# Quick test:
#     NOVA_SHORT_QUICK=1 python a_star_suddenly_got_brighter_short.py

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# %% [markdown]
# ## Configuration

OUTPUT_ROOT = Path("star_suddenly_brighter_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict = {
    "video_width": 1080,
    "video_height": 1920,
    "fps": 24,
    "duration_s": 58,
    "output_basename": "a_star_suddenly_got_brighter_v462_lupi",

    # Target system.
    "target_name": "V462 LUPI",
    "target_alias": "NOVA LUPI 2025 // ASASSN-25cm // AT 2025nlr",
    "ra_j2000": "15h 08m 03.27s",
    "dec_j2000": "-40° 08′ 29.6″",
    "constellation": "LUPUS",

    # Published peak used for relative visual scaling.
    "peak_magnitude": 5.5,
    "quiescent_reference_magnitude": 19.2,

    # Procedural rendering.
    "star_count": 470,
    "dust_cloud_count": 16,
    "binary_particle_count": 270,
    "ejecta_particle_count": 440,
    "contrast_boost": 1.12,
    "saturation_boost": 1.12,
    "vignette_strength": 0.28,

    # Text.
    "title_text": "A STAR SUDDENLY\nGOT BRIGHTER",
    "subtitle_text": "The real 2025 outburst of V462 Lupi",
    "credit_text": (
        "Published photometry: GOTO, ASAS-SN, AAVSO-linked reports and ATel #17240"
    ),
    "scientific_note": (
        "Mixed optical passbands are shown as reported. Connecting curves are "
        "interpolation. Binary geometry, ejecta and sky texture are explanatory artwork."
    ),

    # Optional finishing.
    "audio_path": None,
    "burn_subtitles": False,
    "write_subtitle_sidecar": True,
}

if os.getenv("NOVA_SHORT_QUICK") == "1":
    CONFIG.update({
        "video_width": 540,
        "video_height": 960,
        "fps": 6,
        "duration_s": 12,
        "output_basename": "a_star_suddenly_got_brighter_quick",
        "star_count": 260,
        "binary_particle_count": 130,
        "ejecta_particle_count": 220,
    })

OUT_SIZE = (int(CONFIG["video_width"]), int(CONFIG["video_height"]))
SX = OUT_SIZE[0] / 1080.0
SY = OUT_SIZE[1] / 1920.0
S = min(SX, SY)


# %% [markdown]
# ## Compiled published photometry

# Each row is either a reported measurement or a clearly marked reference value.
# The quiescent row represents the typical GOTO pre-eruption level rather than a
# measurement from one exact night. Different passbands must not be treated as a
# precision single-filter light curve; the renderer labels every band.
PUBLISHED_PHOTOMETRY = [
    {
        "date_utc": "2024-01-01T00:00:00Z",
        "magnitude": 19.20,
        "uncertainty": np.nan,
        "band": "GOTO L",
        "kind": "quiescent reference",
        "source": "GOTO / ATel 17237",
        "note": "Representative pre-eruption level from numerous 2023-2025 epochs.",
    },
    {
        "date_utc": "2025-05-23T08:52:45Z",
        "magnitude": 17.98,
        "uncertainty": 0.05,
        "band": "GOTO L",
        "kind": "early rise",
        "source": "GOTO / ATel 17237",
        "note": "Difference-imaging detection before the ASAS-SN discovery.",
    },
    {
        "date_utc": "2025-06-05T10:09:09Z",
        "magnitude": 17.74,
        "uncertainty": 0.14,
        "band": "GOTO L",
        "kind": "early rise",
        "source": "GOTO / ATel 17237",
        "note": "Last reported GOTO point before the source saturated.",
    },
    {
        "date_utc": "2025-06-12T20:52:48Z",
        "magnitude": 8.70,
        "uncertainty": np.nan,
        "band": "Sloan g",
        "kind": "discovery",
        "source": "ASAS-SN / ATel 17237",
        "note": "ASAS-SN discovery measurement.",
    },
    {
        "date_utc": "2025-06-18T12:00:00Z",
        "magnitude": 5.70,
        "uncertainty": np.nan,
        "band": "visual",
        "kind": "naked-eye threshold",
        "source": "AAVSO-linked observing reports",
        "note": "Reported near the unaided-eye visibility threshold under dark skies.",
    },
    {
        "date_utc": "2025-06-20T12:00:00Z",
        "magnitude": 5.50,
        "uncertainty": np.nan,
        "band": "V",
        "kind": "published peak",
        "source": "ATel 17240",
        "note": "Published peak after a smooth rise lasting more than six days.",
    },
    {
        "date_utc": "2025-06-21T18:00:00Z",
        "magnitude": 5.60,
        "uncertainty": np.nan,
        "band": "visual",
        "kind": "near peak",
        "source": "AAVSO-linked observing summary",
        "note": "Near-peak visual estimate reported the following day.",
    },
]


def load_photometry() -> pd.DataFrame:
    frame = pd.DataFrame(PUBLISHED_PHOTOMETRY)
    frame["date_utc"] = pd.to_datetime(frame["date_utc"], utc=True)
    frame["magnitude"] = pd.to_numeric(frame["magnitude"], errors="coerce")
    frame["uncertainty"] = pd.to_numeric(frame["uncertainty"], errors="coerce")
    frame = frame.dropna(subset=["date_utc", "magnitude"]).sort_values("date_utc")
    frame = frame.reset_index(drop=True)

    peak_mag = float(frame["magnitude"].min())
    frame["relative_flux_to_peak"] = 10.0 ** (-0.4 * (frame["magnitude"] - peak_mag))
    frame["brightness_vs_quiescent_reference"] = 10.0 ** (
        0.4 * (float(CONFIG["quiescent_reference_magnitude"]) - frame["magnitude"])
    )

    path = DATA_ROOT / "v462_lupi_published_photometry.csv"
    save = frame.copy()
    save["date_utc"] = save["date_utc"].astype(str)
    save.to_csv(path, index=False)
    return frame


# %% [markdown]
# ## Math and image helpers


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t: float) -> float:
    t = clamp(t)
    return t * t * t * (t * (t * 6 - 15) + 10)


def ease_in_out_sine(t: float) -> float:
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


def deterministic_unit(text: str, salt: str = "") -> float:
    digest = hashlib.sha256(f"{text}|{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def find_ffmpeg() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def get_font(size: int, bold: bool = False):
    size = max(8, int(size * S))
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
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
    xy: Tuple[float, float],
    size: int = 40,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    stroke: int = 2,
    anchor: str = "la",
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
    draw.text(
        (xy[0] * SX, xy[1] * SY),
        text,
        font=font,
        fill=fill,
        anchor=anchor,
        stroke_width=max(1, int(stroke * S)),
        stroke_fill=(0, 0, 0, min(fill[3] if len(fill) > 3 else 255, 225)),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[float, float],
    max_width: float,
    size: int = 30,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    line_spacing: int = 7,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
    max_width_px = max_width * SX
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=max(1, int(2*S)))
        if bbox[2] - bbox[0] <= max_width_px:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    x = xy[0] * SX
    y = xy[1] * SY
    for line in lines:
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
            stroke_width=max(1, int(2 * S)),
            stroke_fill=(0, 0, 0, 220),
        )
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=max(1, int(2*S)))
        y += (bbox[3] - bbox[1]) + line_spacing * SY


def rounded_panel(
    canvas: Image.Image,
    box: Tuple[float, float, float, float],
    fill=(2, 6, 16, 176),
    outline=(100, 170, 255, 90),
    radius: int = 22,
    width: int = 1,
):
    overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scaled = (
        box[0] * SX,
        box[1] * SY,
        box[2] * SX,
        box[3] * SY,
    )
    draw.rounded_rectangle(
        scaled,
        radius=max(2, int(radius * S)),
        fill=fill,
        outline=outline,
        width=max(1, int(width * S)),
    )
    canvas.alpha_composite(overlay)


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.8, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_SIZE[0], OUT_SIZE[1], float(CONFIG["vignette_strength"]))


def apply_grade(arr: np.ndarray) -> np.ndarray:
    image = Image.fromarray(arr)
    image = ImageEnhance.Contrast(image).enhance(float(CONFIG["contrast_boost"]))
    image = ImageEnhance.Color(image).enhance(float(CONFIG["saturation_boost"]))
    return np.array(image)


def magnitude_flux_ratio(delta_magnitude: float) -> float:
    return 10.0 ** (0.4 * float(delta_magnitude))


def format_flux_ratio(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} MILLION×"
    if value >= 1_000:
        return f"{value / 1_000:.0f} THOUSAND×"
    return f"{value:.0f}×"


# %% [markdown]
# ## Timeline and subtitles

CAPTIONS = [
    (0.5, 6.0, "For years, this point of light was far too faint to see."),
    (6.2, 16.6, "Then V462 Lupi climbed from near magnitude nineteen to naked-eye brightness."),
    (17.0, 27.8, "It was not a new star. It was a close binary hiding a white dwarf."),
    (28.1, 39.8, "Gas piled onto the white dwarf until hydrogen fusion ran away across its surface."),
    (40.1, 49.3, "The blast brightened the system, but it did not destroy the white dwarf."),
    (49.6, 57.3, "That is a nova. A supernova is a different, far more destructive event."),
]

SHOT_PLAN = [
    {
        "name": "faint",
        "start": 0.0,
        "end": 6.5,
        "camera_zoom_start": 0.82,
        "camera_zoom_end": 1.06,
        "target_x_start": 650,
        "target_x_end": 540,
        "target_y_start": 1030,
        "target_y_end": 880,
        "caption": "LUPUS // PRE-ERUPTION FIELD",
    },
    {
        "name": "rise",
        "start": 6.5,
        "end": 17.0,
        "camera_zoom_start": 1.06,
        "camera_zoom_end": 1.26,
        "target_x_start": 540,
        "target_x_end": 610,
        "target_y_start": 880,
        "target_y_end": 805,
        "caption": "PUBLISHED PHOTOMETRY // RAPID RISE",
    },
    {
        "name": "binary",
        "start": 17.0,
        "end": 28.0,
        "camera_zoom_start": 1.0,
        "camera_zoom_end": 1.16,
        "target_x_start": 540,
        "target_x_end": 515,
        "target_y_start": 960,
        "target_y_end": 915,
        "caption": "CLASSICAL NOVA // BINARY SYSTEM",
    },
    {
        "name": "runaway",
        "start": 28.0,
        "end": 40.0,
        "camera_zoom_start": 1.16,
        "camera_zoom_end": 1.34,
        "target_x_start": 515,
        "target_x_end": 565,
        "target_y_start": 915,
        "target_y_end": 890,
        "caption": "THERMONUCLEAR RUNAWAY // WHITE-DWARF SURFACE",
    },
    {
        "name": "aftermath",
        "start": 40.0,
        "end": 50.0,
        "camera_zoom_start": 1.34,
        "camera_zoom_end": 0.93,
        "target_x_start": 565,
        "target_x_end": 540,
        "target_y_start": 890,
        "target_y_end": 950,
        "caption": "EJECTA EXPANSION // SYSTEM SURVIVES",
    },
    {
        "name": "outro",
        "start": 50.0,
        "end": 58.0,
        "camera_zoom_start": 0.93,
        "camera_zoom_end": 0.72,
        "target_x_start": 540,
        "target_x_end": 540,
        "target_y_start": 950,
        "target_y_end": 1010,
        "caption": "NOVA ≠ SUPERNOVA",
    },
]


def time_scale(t: float) -> float:
    if float(CONFIG["duration_s"]) == 58:
        return t
    return t * 58.0 / float(CONFIG["duration_s"])


def get_shot(t: float) -> Dict:
    tt = time_scale(t)
    for shot in SHOT_PLAN:
        if shot["start"] <= tt < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def shot_state(t: float):
    tt = time_scale(t)
    shot = get_shot(t)
    u = (tt - shot["start"]) / max(shot["end"] - shot["start"], 1e-6)
    e = ease_in_out_sine(u)
    zoom = lerp(shot["camera_zoom_start"], shot["camera_zoom_end"], e)
    x = lerp(shot["target_x_start"], shot["target_x_end"], e)
    y = lerp(shot["target_y_start"], shot["target_y_end"], e)
    return shot, x, y, zoom


def caption_at(t: float) -> Optional[str]:
    tt = time_scale(t)
    for start, end, text in CAPTIONS:
        if start <= tt < end:
            return text
    return None


# %% [markdown]
# ## Scene renderer

class NovaScene:
    def __init__(self, photometry: pd.DataFrame):
        self.photometry = photometry.copy().reset_index(drop=True)
        self.dates = self.photometry["date_utc"].to_numpy()
        self.magnitudes = self.photometry["magnitude"].to_numpy(float)
        self.bands = self.photometry["band"].astype(str).to_numpy(object)
        self.kinds = self.photometry["kind"].astype(str).to_numpy(object)

        self.plot_start = pd.Timestamp("2025-05-20T00:00:00Z")
        self.plot_end = pd.Timestamp("2025-06-23T00:00:00Z")
        self.plot_span_s = (self.plot_end - self.plot_start).total_seconds()
        self.point_fraction = np.array([
            (pd.Timestamp(value) - self.plot_start).total_seconds() / self.plot_span_s
            for value in self.photometry["date_utc"]
        ], dtype=float)

        # Plot only the event-era points. The quiescent reference appears as a
        # separate baseline marker before the plotted 2025 time window.
        self.event_mask = self.photometry["date_utc"] >= self.plot_start
        self.event_indices = np.where(self.event_mask.to_numpy())[0]

        self.stars = self._make_stars(int(CONFIG["star_count"]), seed=311)
        self.dust = self._make_dust(int(CONFIG["dust_cloud_count"]), seed=204)
        self.stream_particles = self._make_particles(
            int(CONFIG["binary_particle_count"]), seed=810
        )
        self.ejecta_particles = self._make_particles(
            int(CONFIG["ejecta_particle_count"]), seed=221
        )

        delta_mag = float(CONFIG["quiescent_reference_magnitude"]) - float(CONFIG["peak_magnitude"])
        self.illustrative_ratio = magnitude_flux_ratio(delta_mag)

    @staticmethod
    def _make_stars(count: int, seed: int):
        rng = np.random.default_rng(seed)
        result = []
        for index in range(count):
            x = float(rng.uniform(-0.08, 1.08))
            y = float(rng.uniform(-0.04, 1.04))
            # Dense Milky-Way-like diagonal band.
            band_y = 0.27 + 0.58 * x
            band_weight = math.exp(-((y - band_y) / 0.15) ** 2)
            size = float(rng.uniform(0.45, 1.75) * (1.0 + 0.85 * band_weight))
            alpha = int(rng.integers(35, 165) * (0.72 + 0.55 * band_weight))
            tint = rng.choice([0, 1, 2, 3], p=[0.58, 0.18, 0.16, 0.08])
            result.append({
                "x": x,
                "y": y,
                "r": size,
                "a": min(alpha, 230),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "drift": float(rng.uniform(-0.003, 0.003)),
                "tint": int(tint),
                "id": index,
            })
        return result

    @staticmethod
    def _make_dust(count: int, seed: int):
        rng = np.random.default_rng(seed)
        clouds = []
        for _ in range(count):
            clouds.append({
                "x": float(rng.uniform(0.05, 0.95)),
                "y": float(rng.uniform(0.10, 0.92)),
                "rx": float(rng.uniform(0.10, 0.31)),
                "ry": float(rng.uniform(0.045, 0.15)),
                "angle": float(rng.uniform(-0.7, 0.7)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "warm": bool(rng.random() < 0.45),
            })
        return clouds

    @staticmethod
    def _make_particles(count: int, seed: int):
        rng = np.random.default_rng(seed)
        particles = []
        for index in range(count):
            particles.append({
                "u": float(rng.random()),
                "v": float(rng.random()),
                "angle": float(rng.uniform(0, 2 * math.pi)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "size": float(rng.uniform(0.7, 2.5)),
                "speed": float(rng.uniform(0.55, 1.45)),
                "id": index,
            })
        return particles

    def observation_progress(self, t: float) -> float:
        tt = time_scale(t)
        # The photometry cursor is featured from the end of intro through the rise.
        return smootherstep((tt - 3.8) / 18.8)

    def interpolated_magnitude(self, t: float) -> float:
        progress = self.observation_progress(t)
        # Map across event-era data, preserving the long pre-discovery phase but
        # accelerating the final rise for readability.
        event_fractions = np.clip(self.point_fraction[self.event_indices], 0.0, 1.0)
        event_mags = self.magnitudes[self.event_indices]
        if progress <= event_fractions[0]:
            # Interpolate from quiescent reference into the first event point.
            u = progress / max(event_fractions[0], 1e-6)
            return lerp(float(CONFIG["quiescent_reference_magnitude"]), event_mags[0], smoothstep(u))
        return float(np.interp(progress, event_fractions, event_mags))

    def render_background(self, t: float) -> Image.Image:
        canvas = Image.new("RGBA", OUT_SIZE, (1, 2, 9, 255))

        # Layered Milky Way dust and gas.
        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for cloud in self.dust:
            pulse = 0.86 + 0.14 * math.sin(t * 0.12 + cloud["phase"])
            cx = cloud["x"] * OUT_SIZE[0]
            cy = cloud["y"] * OUT_SIZE[1]
            rx = cloud["rx"] * OUT_SIZE[0]
            ry = cloud["ry"] * OUT_SIZE[1]
            if cloud["warm"]:
                color = (85, 36, 75, int(22 * pulse))
            else:
                color = (18, 74, 115, int(24 * pulse))
            hd.ellipse((cx-rx, cy-ry, cx+rx, cy+ry), fill=color)
        haze = haze.filter(ImageFilter.GaussianBlur(max(10, int(75 * S))))
        canvas.alpha_composite(haze)

        draw = ImageDraw.Draw(canvas)
        palette = [
            (220, 232, 255),
            (150, 195, 255),
            (255, 218, 165),
            (255, 175, 140),
        ]
        for star in self.stars:
            x = ((star["x"] + star["drift"] * t) % 1.16 - 0.08) * OUT_SIZE[0]
            y = star["y"] * OUT_SIZE[1]
            twinkle = 0.72 + 0.28 * math.sin(t * 1.23 + star["phase"])
            alpha = int(star["a"] * twinkle)
            radius = star["r"] * S
            color = palette[star["tint"]]
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(*color, alpha))
            if radius > 1.4 * S:
                draw.line((x-radius*3.2, y, x+radius*3.2, y), fill=(*color, int(alpha*0.22)), width=1)
                draw.line((x, y-radius*3.2, x, y+radius*3.2), fill=(*color, int(alpha*0.18)), width=1)

        return canvas

    def draw_target_star(self, canvas: Image.Image, t: float, x: float, y: float, zoom: float):
        tt = time_scale(t)
        magnitude = self.interpolated_magnitude(t)
        peak = float(CONFIG["peak_magnitude"])
        quiescent = float(CONFIG["quiescent_reference_magnitude"])
        normalized = clamp((quiescent - magnitude) / max(quiescent - peak, 1e-6))
        intensity = normalized ** 1.25

        x *= SX
        y *= SY
        radius = (2.1 + 10.5 * intensity + 4.0 * intensity**2) * S * zoom
        halo_radius = (24 + 165 * intensity**1.5) * S * zoom

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for factor, alpha in [(1.0, 42), (0.62, 72), (0.30, 135)]:
            r = halo_radius * factor
            gd.ellipse((x-r, y-r, x+r, y+r), fill=(115, 170, 255, int(alpha * (0.2 + 0.8*intensity))))
        glow = glow.filter(ImageFilter.GaussianBlur(max(4, int(24 * S))))
        canvas.alpha_composite(glow)

        sharp = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sharp)
        sd.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(255, 250, 232, 255))
        spike = (18 + 84 * intensity) * S * zoom
        sd.line((x-spike, y, x+spike, y), fill=(185, 215, 255, int(80 + 145*intensity)), width=max(1, int(2*S)))
        sd.line((x, y-spike, x, y+spike), fill=(185, 215, 255, int(70 + 135*intensity)), width=max(1, int(2*S)))
        diag = spike * 0.52
        sd.line((x-diag, y-diag, x+diag, y+diag), fill=(215, 230, 255, int(48 + 92*intensity)), width=1)
        sd.line((x-diag, y+diag, x+diag, y-diag), fill=(215, 230, 255, int(48 + 92*intensity)), width=1)
        canvas.alpha_composite(sharp)

        # Target reticle and factual callout.
        reticle_alpha = int(225 * smoothstep((tt - 0.8) / 1.2) * (1.0 - smoothstep((tt - 17.0) / 2.5)))
        if reticle_alpha > 3:
            overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            rr = (46 + 10 * math.sin(t*2.0)) * S
            od.arc((x-rr, y-rr, x+rr, y+rr), 5, 105, fill=(105, 225, 255, reticle_alpha), width=max(1, int(2*S)))
            od.arc((x-rr, y-rr, x+rr, y+rr), 185, 285, fill=(105, 225, 255, reticle_alpha), width=max(1, int(2*S)))
            od.line((x-rr-12*S, y, x-rr+5*S, y), fill=(105,225,255,reticle_alpha), width=max(1,int(2*S)))
            od.line((x+rr-5*S, y, x+rr+12*S, y), fill=(105,225,255,reticle_alpha), width=max(1,int(2*S)))
            canvas.alpha_composite(overlay)

        if 5.0 <= tt < 17.2:
            panel_x = 55 if x / SX > 560 else 575
            rounded_panel(canvas, (panel_x, 235, panel_x+450, 425), fill=(2,6,17,184), outline=(75,190,240,105), radius=20)
            draw_text(canvas, "LIVE BRIGHTNESS REPLAY", (panel_x+22, 254), size=19, fill=(120,225,250,235), bold=True, stroke=1)
            draw_text(canvas, f"MAGNITUDE {magnitude:4.1f}", (panel_x+22, 294), size=34, fill=(248,252,255,250), bold=True)
            draw_text(canvas, "smaller number = brighter", (panel_x+22, 344), size=19, fill=(185,210,232,230), stroke=1)
            relative = 10.0 ** (0.4 * (quiescent - magnitude))
            draw_text(canvas, f"~{format_flux_ratio(relative)} VS QUIESCENT", (panel_x+22, 377), size=18, fill=(255,190,105,235), bold=True, stroke=1)

    def draw_light_curve(self, canvas: Image.Image, t: float):
        tt = time_scale(t)
        alpha = int(240 * smoothstep((tt - 4.2) / 2.0) * (1.0 - smoothstep((tt - 24.0) / 4.0)))
        if alpha <= 3:
            return

        x0, x1 = 72, 1008
        y0, y1 = 1230, 1575
        rounded_panel(canvas, (42, 1140, 1038, 1660), fill=(1,5,15,178), outline=(75,175,225,92), radius=24)
        draw_text(canvas, "PUBLISHED OPTICAL MEASUREMENTS", (70, 1170), size=23, fill=(185,230,248,alpha), bold=True, stroke=1)
        draw_text(canvas, "Magnitude axis is inverted", (1005, 1173), size=17, fill=(170,200,220,alpha), stroke=1, anchor="ra")

        d = ImageDraw.Draw(canvas)
        x0p, x1p = x0*SX, x1*SX
        y0p, y1p = y0*SY, y1*SY

        min_mag, max_mag = 5.0, 20.0
        for mag in [5, 8, 11, 14, 17, 20]:
            yy = y0p + (mag-min_mag)/(max_mag-min_mag)*(y1p-y0p)
            d.line((x0p, yy, x1p, yy), fill=(100,155,190,int(alpha*0.22)), width=1)
            draw_text(canvas, f"{mag}", (55, yy/SY), size=16, fill=(155,195,218,alpha), stroke=1, anchor="ra")

        # Baseline marker for quiescence.
        qmag = float(CONFIG["quiescent_reference_magnitude"])
        qy = y0p + (qmag-min_mag)/(max_mag-min_mag)*(y1p-y0p)
        d.line((x0p, qy, x1p, qy), fill=(150,120,210,int(alpha*0.34)), width=max(1,int(2*S)))
        draw_text(canvas, "GOTO QUIESCENT ~19.2", (92, qy/SY-24), size=16, fill=(190,165,235,alpha), bold=True, stroke=1)

        # Plot observed points only as the cursor reaches them.
        progress = self.observation_progress(t)
        plotted: List[Tuple[float,float,int]] = []
        for index in self.event_indices:
            frac = clamp(self.point_fraction[index])
            xx = x0p + frac * (x1p-x0p)
            mag = self.magnitudes[index]
            yy = y0p + (mag-min_mag)/(max_mag-min_mag)*(y1p-y0p)
            if frac <= progress + 0.015:
                plotted.append((xx, yy, index))

        if plotted:
            # Connecting line is explicitly editorial interpolation.
            points = [(x0p, qy)] + [(item[0], item[1]) for item in plotted]
            line_layer = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
            ld = ImageDraw.Draw(line_layer)
            ld.line(points, fill=(80,210,255,int(alpha*0.42)), width=max(1,int(3*S)), joint="curve")
            line_layer = line_layer.filter(ImageFilter.GaussianBlur(max(1,int(2*S))))
            canvas.alpha_composite(line_layer)
            d = ImageDraw.Draw(canvas)
            d.line(points, fill=(135,235,255,alpha), width=max(1,int(2*S)), joint="curve")

            band_colors = {
                "GOTO L": (165,135,255),
                "Sloan g": (80,210,255),
                "visual": (255,215,130),
                "V": (255,155,95),
            }
            for xx, yy, index in plotted:
                color = band_colors.get(str(self.bands[index]), (230,240,255))
                rr = 7*S
                d.ellipse((xx-rr, yy-rr, xx+rr, yy+rr), fill=(*color,alpha), outline=(255,255,255,alpha), width=max(1,int(1*S)))

        cursor_x = x0p + progress*(x1p-x0p)
        d.line((cursor_x, y0p-12*S, cursor_x, y1p+12*S), fill=(255,180,90,alpha), width=max(1,int(2*S)))

        draw_text(canvas, "MAY 20", (x0, 1595), size=16, fill=(160,195,215,alpha), stroke=1)
        draw_text(canvas, "JUN 12", (710, 1595), size=16, fill=(160,195,215,alpha), stroke=1, anchor="ma")
        draw_text(canvas, "JUN 23", (x1, 1595), size=16, fill=(160,195,215,alpha), stroke=1, anchor="ra")
        draw_text(canvas, "points = observations // line = interpolation", (72, 1630), size=16, fill=(150,185,210,alpha), stroke=1)

    def draw_radial_star(self, canvas: Image.Image, center: Tuple[float,float], radius: float, colors: Sequence[Tuple[int,int,int]], glow_scale: float = 1.5):
        cx, cy = center[0]*SX, center[1]*SY
        r = radius*S
        glow = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        gr = r*glow_scale
        gd.ellipse((cx-gr,cy-gr,cx+gr,cy+gr), fill=(*colors[-1],85))
        glow = glow.filter(ImageFilter.GaussianBlur(max(3,int(r*0.45))))
        canvas.alpha_composite(glow)

        layer = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        ld = ImageDraw.Draw(layer)
        steps = 34
        for i in range(steps, 0, -1):
            u = i/steps
            rr = r*u
            # Blend outer to inner.
            c0, c1 = colors[-1], colors[0]
            mix = 1.0-u
            col = tuple(int(lerp(c0[j], c1[j], mix)) for j in range(3))
            ld.ellipse((cx-rr,cy-rr,cx+rr,cy+rr), fill=(*col,255))
        canvas.alpha_composite(layer)

    def draw_binary(self, canvas: Image.Image, t: float, x: float, y: float, zoom: float):
        tt = time_scale(t)
        alpha = smoothstep((tt-15.6)/2.4) * (1.0-smoothstep((tt-50.0)/6.0))
        if alpha <= 0.01:
            return

        donor = (x-225*zoom, y+20*zoom)
        white_dwarf = (x+205*zoom, y-30*zoom)
        donor_radius = 128*zoom
        wd_radius = 34*zoom

        # Orbit guide.
        overlay = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        od = ImageDraw.Draw(overlay)
        od.ellipse(
            ((x-315*zoom)*SX, (y-175*zoom)*SY, (x+315*zoom)*SX, (y+175*zoom)*SY),
            outline=(90,175,220,int(58*alpha)), width=max(1,int(2*S))
        )
        canvas.alpha_composite(overlay)

        self.draw_radial_star(canvas, donor, donor_radius, [(255,225,165),(200,75,45),(80,12,12)], glow_scale=1.28)

        # Donor surface texture.
        surface = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        sd = ImageDraw.Draw(surface)
        dcx,dcy=donor[0]*SX,donor[1]*SY
        dr=donor_radius*S
        for i in range(32):
            ang = deterministic_unit(str(i),"donor-a")*2*math.pi + t*0.035
            radial = math.sqrt(deterministic_unit(str(i),"donor-r"))*dr*0.82
            px = dcx+math.cos(ang)*radial
            py = dcy+math.sin(ang)*radial
            rr=(2+7*deterministic_unit(str(i),"donor-s"))*S
            sd.ellipse((px-rr,py-rr,px+rr,py+rr),fill=(85,20,15,int(50*alpha)))
        canvas.alpha_composite(surface)

        # Accretion disk behind and in front of the white dwarf.
        disk = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        dd = ImageDraw.Draw(disk)
        wx,wy=white_dwarf[0]*SX,white_dwarf[1]*SY
        for band in range(18,0,-1):
            u=band/18
            rx=(105*zoom*u)*SX
            ry=(27*zoom*u)*SY
            col=(int(lerp(255,115,u)),int(lerp(245,105,u)),int(lerp(220,255,u)),int((30+125*(1-u))*alpha))
            dd.ellipse((wx-rx,wy-ry,wx+rx,wy+ry),outline=col,width=max(1,int(3*S)))
        disk= disk.filter(ImageFilter.GaussianBlur(max(1,int(1.3*S))))
        canvas.alpha_composite(disk)

        self.draw_radial_star(canvas, white_dwarf, wd_radius, [(255,255,255),(165,220,255),(35,85,180)], glow_scale=2.6)

        # Gas stream from donor to disk.
        stream = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        std = ImageDraw.Draw(stream)
        p0=np.array([donor[0]+donor_radius*0.72, donor[1]-donor_radius*0.05],float)
        p1=np.array([x-20*zoom,y-155*zoom],float)
        p2=np.array([white_dwarf[0]-70*zoom,white_dwarf[1]],float)
        path=[]
        for k in range(70):
            u=k/69
            point=(1-u)**2*p0+2*(1-u)*u*p1+u**2*p2
            path.append((point[0]*SX,point[1]*SY))
        std.line(path,fill=(135,210,255,int(82*alpha)),width=max(1,int(14*S*zoom)))
        stream=stream.filter(ImageFilter.GaussianBlur(max(2,int(7*S))))
        canvas.alpha_composite(stream)

        sharp=Image.new("RGBA",OUT_SIZE,(0,0,0,0))
        shd=ImageDraw.Draw(sharp)
        shd.line(path,fill=(205,235,255,int(155*alpha)),width=max(1,int(3*S*zoom)))
        for part in self.stream_particles:
            u=(part["u"]+t*0.15*part["speed"])%1.0
            point=(1-u)**2*p0+2*(1-u)*u*p1+u**2*p2
            jitter=(part["v"]-0.5)*14*zoom
            px=(point[0]+math.sin(part["phase"]+u*9)*jitter)*SX
            py=(point[1]+math.cos(part["phase"]+u*9)*jitter)*SY
            rr=part["size"]*S
            shd.ellipse((px-rr,py-rr,px+rr,py+rr),fill=(230,245,255,int(120*alpha)))
        canvas.alpha_composite(sharp)

        # Labels.
        label_alpha=int(235*alpha)
        draw_text(canvas,"DONOR STAR",(donor[0]-30,donor[1]+165*zoom),size=20,fill=(255,205,150,label_alpha),bold=True,stroke=1,anchor="ma")
        draw_text(canvas,"WHITE DWARF",(white_dwarf[0],white_dwarf[1]+105*zoom),size=20,fill=(145,220,255,label_alpha),bold=True,stroke=1,anchor="ma")
        draw_text(canvas,"gas transfer",(x-20,y-165*zoom),size=18,fill=(185,220,240,label_alpha),stroke=1,anchor="ma")

        # Thermonuclear ignition and shells.
        ignition = smoothstep((tt-28.0)/4.2)
        if ignition > 0.01:
            self.draw_nova_eruption(canvas,t,white_dwarf,zoom,ignition)

    def draw_nova_eruption(self, canvas: Image.Image, t: float, white_dwarf: Tuple[float,float], zoom: float, ignition: float):
        tt=time_scale(t)
        cx,cy=white_dwarf[0]*SX,white_dwarf[1]*SY
        # Flash envelope peaks around 34 seconds then settles.
        flash_rise=smoothstep((tt-28.2)/4.0)
        flash_decay=1.0-0.55*smoothstep((tt-36.0)/10.0)
        flash=flash_rise*flash_decay

        glow=Image.new("RGBA",OUT_SIZE,(0,0,0,0))
        gd=ImageDraw.Draw(glow)
        for radius,alpha in [(250,18),(160,34),(90,85),(48,170)]:
            rr=radius*S*zoom*(0.55+0.75*flash)
            gd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=(110,190,255,int(alpha*flash)))
        glow=glow.filter(ImageFilter.GaussianBlur(max(5,int(28*S))))
        canvas.alpha_composite(glow)

        # Expanding ejecta rings.
        expansion=smoothstep((tt-32.0)/12.0)
        if expansion>0:
            ring_layer=Image.new("RGBA",OUT_SIZE,(0,0,0,0))
            rd=ImageDraw.Draw(ring_layer)
            for ring_i in range(4):
                phase=clamp(expansion-ring_i*0.10)
                rr=(55+330*phase)*S*zoom
                a=int(145*(1-phase)*ignition)
                rd.ellipse((cx-rr,cy-rr*0.72,cx+rr,cy+rr*0.72),outline=(115,205,255,a),width=max(1,int((4-ring_i*0.5)*S)))
            ring_layer=ring_layer.filter(ImageFilter.GaussianBlur(max(1,int(2*S))))
            canvas.alpha_composite(ring_layer)

            particles=Image.new("RGBA",OUT_SIZE,(0,0,0,0))
            pd=ImageDraw.Draw(particles)
            for part in self.ejecta_particles:
                local=clamp(expansion*1.25-part["u"]*0.28)
                if local<=0:
                    continue
                angle=part["angle"]+0.12*math.sin(t*0.8+part["phase"])
                radius=(45+420*local*part["speed"])*S*zoom
                flatten=0.70+0.20*math.sin(part["phase"])
                px=cx+math.cos(angle)*radius
                py=cy+math.sin(angle)*radius*flatten
                rr=part["size"]*S*(0.7+0.8*local)
                a=int(180*(1-local)*ignition)
                color=(150,220,255,a) if part["v"]>0.35 else (255,190,125,a)
                pd.ellipse((px-rr,py-rr,px+rr,py+rr),fill=color)
            canvas.alpha_composite(particles)

        # Surface runaway label.
        label_alpha=int(245*smoothstep((tt-28.0)/2.5)*(1.0-smoothstep((tt-41.0)/4.0)))
        if label_alpha>3:
            rounded_panel(canvas,(70,250,530,430),fill=(4,8,20,172),outline=(110,200,255,90),radius=20)
            draw_text(canvas,"THERMONUCLEAR RUNAWAY",(92,272),size=23,fill=(130,220,255,label_alpha),bold=True,stroke=1)
            draw_wrapped_text(canvas,"Accumulated hydrogen ignites across the white-dwarf surface.",(92,318),400,size=25,fill=(242,248,255,label_alpha),bold=False)

    def draw_fact_panels(self, canvas: Image.Image, t: float):
        tt=time_scale(t)

        if 8.0 <= tt < 18.0:
            alpha=int(235*smoothstep((tt-8)/1.8)*(1.0-smoothstep((tt-17)/1.0)))
            rounded_panel(canvas,(55,470,510,665),fill=(2,7,18,174),outline=(255,165,95,95),radius=20)
            draw_text(canvas,"THE OBSERVED JUMP",(78,492),size=20,fill=(255,190,110,alpha),bold=True,stroke=1)
            draw_text(canvas,"~19.2  →  5.5",(78,532),size=41,fill=(248,252,255,alpha),bold=True)
            draw_text(canvas,"about 13.7 magnitudes",(78,590),size=21,fill=(190,220,238,alpha),stroke=1)
            draw_text(canvas,f"illustrative flux ratio: {format_flux_ratio(self.illustrative_ratio)}",(78,625),size=18,fill=(150,215,245,alpha),bold=True,stroke=1)

        if 40.0 <= tt < 51.0:
            alpha=int(240*smoothstep((tt-40)/2.0)*(1.0-smoothstep((tt-50)/2.0)))
            rounded_panel(canvas,(55,235,545,515),fill=(2,6,17,182),outline=(90,190,235,100),radius=22)
            draw_text(canvas,"WHAT CHANGED?",(78,258),size=22,fill=(125,220,250,alpha),bold=True,stroke=1)
            rows=[
                ("SYSTEM","survived"),
                ("WHITE DWARF","not destroyed"),
                ("OUTER GAS","ejected"),
                ("OPTICAL LIGHT","temporarily amplified"),
            ]
            for i,(left,right) in enumerate(rows):
                yy=306+i*49
                draw_text(canvas,left,(78,yy),size=18,fill=(175,205,225,alpha),bold=True,stroke=1)
                draw_text(canvas,right,(515,yy),size=18,fill=(245,248,255,alpha),stroke=1,anchor="ra")

        if tt >= 49.0:
            alpha=int(245*smoothstep((tt-49)/2.2))
            rounded_panel(canvas,(70,255,1010,600),fill=(1,4,13,186),outline=(255,170,90,100),radius=24)
            draw_text(canvas,"NOVA",(270,300),size=46,fill=(120,220,255,alpha),bold=True,anchor="ma")
            draw_text(canvas,"SUPERNOVA",(800,300),size=46,fill=(255,175,110,alpha),bold=True,anchor="ma")
            draw_wrapped_text(canvas,"Surface eruption in an accreting binary. The white dwarf can remain.",(105,380),340,size=24,fill=(235,245,255,alpha))
            draw_wrapped_text(canvas,"Catastrophic stellar destruction or core collapse. A different event.",(635,380),330,size=24,fill=(255,238,225,alpha))
            midx=540*SX
            d=ImageDraw.Draw(canvas)
            d.line((midx,340*SY,midx,555*SY),fill=(130,165,195,int(alpha*0.5)),width=max(1,int(2*S)))

    def draw_coordinate_hud(self, canvas: Image.Image, t: float):
        tt=time_scale(t)
        alpha=int(220*smoothstep((tt-0.2)/1.0))
        draw_text(canvas,CONFIG["target_name"],(54,66),size=26,fill=(242,248,255,alpha),bold=True)
        draw_text(canvas,CONFIG["target_alias"],(56,107),size=17,fill=(120,215,240,alpha),bold=True,stroke=1)
        draw_text(canvas,f"RA  {CONFIG['ra_j2000']}",(1025,68),size=18,fill=(160,205,225,alpha),stroke=1,anchor="ra")
        draw_text(canvas,f"DEC {CONFIG['dec_j2000']}",(1025,99),size=18,fill=(160,205,225,alpha),stroke=1,anchor="ra")
        draw_text(canvas,f"CONSTELLATION // {CONFIG['constellation']}",(1025,130),size=17,fill=(150,195,218,alpha),stroke=1,anchor="ra")

    def draw_chapter_bar(self, canvas: Image.Image, t: float):
        tt=time_scale(t)
        shot=get_shot(t)
        x0,x1,y=70,1010,1735
        d=ImageDraw.Draw(canvas)
        d.line((x0*SX,y*SY,x1*SX,y*SY),fill=(90,165,205,145),width=max(1,int(2*S)))
        fraction=clamp(tt/58.0)
        d.line((x0*SX,y*SY,(x0+(x1-x0)*fraction)*SX,y*SY),fill=(255,180,90,235),width=max(1,int(3*S)))
        draw_text(canvas,shot["caption"],(70,1688),size=19,fill=(155,215,235,215),bold=True,stroke=1)
        draw_text(canvas,f"{tt:04.1f}s",(1010,1688),size=17,fill=(160,198,220,205),stroke=1,anchor="ra")

    def add_titles_and_caption(self, canvas: Image.Image, t: float):
        tt=time_scale(t)
        title_alpha=int(255*smoothstep((tt-0.15)/1.1)*(1.0-smoothstep((tt-5.8)/1.0)))
        if title_alpha>3:
            draw_text(canvas,CONFIG["title_text"],(55,235),size=57,fill=(246,250,255,title_alpha),bold=True)
            draw_text(canvas,CONFIG["subtitle_text"],(59,390),size=25,fill=(115,225,250,min(title_alpha,230)),bold=True,stroke=1)

        caption=caption_at(t)
        if caption:
            rounded_panel(canvas,(45,1770,1035,1886),fill=(1,4,12,178),outline=(65,170,220,65),radius=22)
            draw_wrapped_text(canvas,caption,(70,1792),940,size=29,fill=(245,249,255,245))

        if tt>52.0:
            alpha=int(215*smoothstep((tt-52)/2.0))
            draw_text(canvas,CONFIG["credit_text"],(60,1635),size=16,fill=(205,222,238,alpha),stroke=1)
            draw_wrapped_text(canvas,CONFIG["scientific_note"],(60,1661),955,size=15,fill=(175,202,225,alpha))

    def draw_scanlines(self, canvas: Image.Image, t: float):
        overlay=Image.new("RGBA",OUT_SIZE,(0,0,0,0))
        d=ImageDraw.Draw(overlay)
        offset=int((t*37)%7)
        for yy in range(offset,OUT_SIZE[1],7):
            d.line((0,yy,OUT_SIZE[0],yy),fill=(100,180,230,10),width=1)
        scan_y=int((t*132)%(OUT_SIZE[1]+260))-130
        d.rectangle((0,scan_y,OUT_SIZE[0],scan_y+45*S),fill=(100,210,245,8))
        canvas.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot,x,y,zoom=shot_state(t)
        canvas=self.render_background(t)

        if shot["name"] in {"faint","rise"}:
            self.draw_target_star(canvas,t,x,y,zoom)
            self.draw_light_curve(canvas,t)
        else:
            self.draw_binary(canvas,t,x,y,zoom)

        self.draw_fact_panels(canvas,t)
        self.draw_coordinate_hud(canvas,t)
        self.draw_chapter_bar(canvas,t)
        self.add_titles_and_caption(canvas,t)
        self.draw_scanlines(canvas,t)

        arr=np.array(canvas.convert("RGB"))
        arr=apply_grade(arr)
        arr=np.clip(arr.astype(np.float32)*VIGNETTE[...,None],0,255).astype(np.uint8)

        tt=time_scale(t)
        fade_in=smoothstep(tt/0.8)
        fade_out=1.0-smoothstep((tt-56.8)/1.0)
        arr=np.clip(arr.astype(np.float32)*fade_in*fade_out,0,255).astype(np.uint8)
        return arr


# %% [markdown]
# ## Data previews


def create_data_preview(photometry: pd.DataFrame):
    # PIL chart keeps matplotlib optional and matches the video visual language.
    width,height=1400,820
    image=Image.new("RGB",(width,height),(4,8,18))
    d=ImageDraw.Draw(image)
    font=get_font(30,bold=True)
    small=get_font(21,bold=False)
    d.text((60,38),"V462 Lupi — compiled published optical measurements",font=font,fill=(235,245,255))
    d.text((60,82),"Different passbands are preserved and labeled; magnitude axis is inverted.",font=small,fill=(165,205,228))

    x0,x1=110,1330
    y0,y1=155,700
    min_mag,max_mag=5,20
    event=photometry[photometry["date_utc"]>=pd.Timestamp("2025-05-20T00:00:00Z")].copy()
    start=pd.Timestamp("2025-05-20T00:00:00Z")
    end=pd.Timestamp("2025-06-23T00:00:00Z")
    span=(end-start).total_seconds()

    for mag in [5,8,11,14,17,20]:
        yy=y0+(mag-min_mag)/(max_mag-min_mag)*(y1-y0)
        d.line((x0,yy,x1,yy),fill=(50,90,120),width=1)
        d.text((52,yy-12),str(mag),font=small,fill=(175,205,225))

    pts=[]
    colors={"GOTO L":(180,145,255),"Sloan g":(75,215,255),"visual":(255,220,135),"V":(255,155,95)}
    for _,row in event.iterrows():
        frac=(row["date_utc"]-start).total_seconds()/span
        xx=x0+frac*(x1-x0)
        yy=y0+(row["magnitude"]-min_mag)/(max_mag-min_mag)*(y1-y0)
        pts.append((xx,yy))
    if pts:
        d.line(pts,fill=(120,225,255),width=4)
    for (_,row),(xx,yy) in zip(event.iterrows(),pts):
        color=colors.get(row["band"],(240,240,255))
        d.ellipse((xx-8,yy-8,xx+8,yy+8),fill=color,outline=(255,255,255),width=2)
        d.text((xx+12,yy-27),f"{row['magnitude']:.2f} {row['band']}",font=small,fill=color)

    d.text((x0,y1+35),"2025-05-20",font=small,fill=(175,205,225))
    d.text((x1-130,y1+35),"2025-06-23",font=small,fill=(175,205,225))
    path=PREVIEW_DIR/"v462_lupi_compiled_light_curve.png"
    image.save(path)
    return path


# %% [markdown]
# ## Subtitles and video output


def format_srt_time(seconds: float) -> str:
    ms=int(round(seconds*1000))
    hours=ms//3_600_000
    ms%=3_600_000
    minutes=ms//60_000
    ms%=60_000
    secs=ms//1000
    ms%=1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(captions, path: Path):
    scale=float(CONFIG["duration_s"])/58.0
    lines=[]
    for i,(start,end,text) in enumerate(captions,1):
        lines.append(str(i))
        lines.append(f"{format_srt_time(start*scale)} --> {format_srt_time(end*scale)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines),encoding="utf-8")
    return path


def run_ffmpeg(command: List[str]):
    subprocess.run(command,check=True)


def render_video(scene: NovaScene):
    raw=OUTPUT_ROOT/f"{CONFIG['output_basename']}_raw.mp4"
    final=OUTPUT_ROOT/f"{CONFIG['output_basename']}_final.mp4"
    subbed=OUTPUT_ROOT/f"{CONFIG['output_basename']}_subbed.mp4"
    audio_out=OUTPUT_ROOT/f"{CONFIG['output_basename']}_with_audio.mp4"
    srt=OUTPUT_ROOT/f"{CONFIG['output_basename']}.srt"

    if CONFIG.get("write_subtitle_sidecar",True):
        write_srt(CAPTIONS,srt)

    frame_count=int(round(float(CONFIG["duration_s"])*int(CONFIG["fps"])))
    times=np.arange(frame_count)/float(CONFIG["fps"])
    with iio.get_writer(
        raw,
        fps=int(CONFIG["fps"]),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times,desc="Rendering V462 Lupi nova short"):
            writer.append_data(scene.render_frame(float(t)))

    candidate=raw
    ffmpeg=find_ffmpeg()
    if CONFIG.get("burn_subtitles",False) and ffmpeg and srt.exists():
        command=[
            ffmpeg,"-y","-i",str(candidate),
            "-vf",f"subtitles={srt}:force_style=Fontname=DejaVu Sans,Fontsize=22,Outline=1.2,BorderStyle=3,MarginV=90",
            "-c:v","libx264","-pix_fmt","yuv420p","-c:a","copy",str(subbed),
        ]
        run_ffmpeg(command)
        candidate=subbed

    audio_path=CONFIG.get("audio_path")
    if audio_path and Path(audio_path).exists() and ffmpeg:
        command=[
            ffmpeg,"-y","-i",str(candidate),"-i",str(audio_path),
            "-c:v","copy","-c:a","aac","-b:a","192k","-shortest",str(audio_out),
        ]
        run_ffmpeg(command)
        candidate=audio_out

    shutil.copyfile(candidate,final)
    return final


# %% [markdown]
# ## Main pipeline


def main():
    print("Starting V462 Lupi nova Short pipeline ...")
    photometry=load_photometry()
    print(photometry[["date_utc","magnitude","band","kind"]].to_string(index=False))
    print("Peak magnitude in compiled table:",photometry["magnitude"].min())
    print("Illustrative quiescent-to-peak flux ratio:",magnitude_flux_ratio(
        float(CONFIG["quiescent_reference_magnitude"])-float(CONFIG["peak_magnitude"])
    ))

    create_data_preview(photometry)
    scene=NovaScene(photometry)

    # Fixed story-time previews, scaled for quick mode.
    story_times=[1.2,9.0,21.0,33.5,44.0,55.0]
    scale=float(CONFIG["duration_s"])/58.0
    for story_t in tqdm(story_times,desc="Preview frames"):
        t=story_t*scale
        frame=scene.render_frame(t)
        Image.fromarray(frame).save(PREVIEW_DIR/f"preview_{int(story_t):02d}s.png")

    final=render_video(scene)
    print("Final video:",final.resolve())
    print("Data CSV:",(DATA_ROOT/"v462_lupi_published_photometry.csv").resolve())
    print("Preview directory:",PREVIEW_DIR.resolve())


if __name__=="__main__":
    main()


# ## Suggested narration
#
# For years, this point of light was almost invisible.
# Then, in June 2025, V462 Lupi suddenly surged.
# ASAS-SN found it at magnitude 8.7.
# Eight days later it reached about magnitude 5.5 — bright enough for dark-sky
# observers to see without a telescope.
# But this was not a brand-new star, and it was not a supernova.
# A white dwarf had been stealing gas from a companion.
# When enough hydrogen accumulated, fusion ran away across the white dwarf's surface.
# The outer gas blasted into space. The binary survived.
# That temporary stellar flare-up is called a nova.
#
# Suggested title:
# A Star Suddenly Got Brighter — Then Astronomers Realized Why
#
# Suggested caption:
# V462 Lupi rose from a faint pre-eruption binary to a naked-eye nova in June 2025.
# Published measurements use several optical passbands, shown exactly as labeled.
# The connecting light-curve line and binary artwork are explanatory visualization.
#
# #Astronomy #Nova #V462Lupi #Space #Stars #Python #DataVisualization
