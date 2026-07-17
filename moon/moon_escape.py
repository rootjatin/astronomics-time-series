from __future__ import annotations

"""
The Moon Is Slowly "Escaping" Earth — cinematic YouTube Short renderer

This script creates a vertical 1080×1920 astronomy short explaining the measured
lunar recession caused by tidal interactions in the Earth–Moon system.

Scientific grounding:
- Mean Earth–Moon distance: ~384,400 km
- Present-day lunar recession from lunar laser ranging: ~3.82 cm/year
- Apollo-era retroreflectors allow direct ranging using laser pulses
- The Moon is moving outward, but it is NOT currently escaping Earth's gravity
- The recession rate is a present-day measured average and should not be
  back-extrapolated linearly over all geological time

Recommended install:
    pip install numpy pandas matplotlib pillow imageio imageio-ffmpeg tqdm

Quick test render:
    MOON_ESCAPE_SHORT_QUICK=1 python the_moon_is_slowly_escaping_earth_short.py
"""

import math
import os
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


# %% Configuration

OUTPUT_ROOT = Path("moon_escape_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"

for directory in [OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

QUICK_MODE = os.environ.get("MOON_ESCAPE_SHORT_QUICK", "0") == "1"

CONFIG = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "the_moon_is_slowly_escaping_earth",

    # Modern Earth–Moon system reference values
    "mean_earth_moon_distance_km": 384_400.0,
    "lunar_recession_cm_per_year": 3.82,
    "earth_radius_km": 6_371.0,
    "moon_radius_km": 1_737.4,
    "earth_mu_km3_s2": 398_600.4418,
    "mean_lunar_orbital_speed_km_s": 1.022,
    "day_length_increase_ms_per_century": 1.7,

    # Rendering controls
    "contrast_boost": 1.10,
    "saturation_boost": 1.08,
    "vignette_strength": 0.25,
    "background_particle_count": 360,
    "hud_noise_count": 60,

    # Text
    "title_text": "THE MOON IS SLOWLY ESCAPING EARTH",
    "subtitle_text": "Tidal friction, laser ranging, and a widening orbit",
    "credit_text": "Reference values: lunar laser ranging and standard Earth-Moon system constants",
    "scientific_note": (
        '"Escaping" is headline shorthand. The Moon is receding by about 3.82 cm per year, '
        'but it remains gravitationally bound to Earth. The present recession rate should not '
        'be extended linearly across all of geologic history.'
    ),
}

OUT_SIZE = (CONFIG["video_width"], CONFIG["video_height"])


# %% Captions and shot plan

CAPTIONS = [
    (0.5, 6.5, "The Moon is measurably drifting away from Earth."),
    (6.6, 14.5, "Laser pulses bounced off lunar reflectors show a widening orbit of about 3.82 centimeters per year."),
    (14.6, 25.0, "Earth's tidal bulges rotate slightly ahead of the Moon and transfer angular momentum outward."),
    (25.1, 36.5, "That slows Earth's spin and pushes the Moon into a higher orbit."),
    (36.6, 48.5, "Over one million years, today's rate would add only about 38 kilometers to the distance."),
    (48.6, 57.4, "So the Moon is receding—not breaking free into space."),
]

SHOT_PLAN = [
    {
        "name": "intro",
        "start": 0.0,
        "end": 8.0 if not QUICK_MODE else 2.0,
        "earth_x_start": 540 if not QUICK_MODE else 270,
        "earth_x_end": 520 if not QUICK_MODE else 258,
        "earth_y_start": 1130 if not QUICK_MODE else 565,
        "earth_y_end": 1095 if not QUICK_MODE else 548,
        "zoom_start": 0.96,
        "zoom_end": 1.04,
        "caption": "EARTH-MOON SYSTEM // PRESENT DAY",
    },
    {
        "name": "laser",
        "start": 8.0 if not QUICK_MODE else 2.0,
        "end": 17.0 if not QUICK_MODE else 4.0,
        "earth_x_start": 520 if not QUICK_MODE else 258,
        "earth_x_end": 505 if not QUICK_MODE else 248,
        "earth_y_start": 1095 if not QUICK_MODE else 548,
        "earth_y_end": 1060 if not QUICK_MODE else 530,
        "zoom_start": 1.04,
        "zoom_end": 1.08,
        "caption": "LUNAR LASER RANGING // DIRECT MEASUREMENT",
    },
    {
        "name": "tides",
        "start": 17.0 if not QUICK_MODE else 4.0,
        "end": 31.0 if not QUICK_MODE else 7.0,
        "earth_x_start": 505 if not QUICK_MODE else 248,
        "earth_x_end": 560 if not QUICK_MODE else 276,
        "earth_y_start": 1060 if not QUICK_MODE else 530,
        "earth_y_end": 1080 if not QUICK_MODE else 542,
        "zoom_start": 1.08,
        "zoom_end": 1.14,
        "caption": "TIDAL FRICTION // ANGULAR MOMENTUM TRANSFER",
    },
    {
        "name": "scale",
        "start": 31.0 if not QUICK_MODE else 7.0,
        "end": 47.0 if not QUICK_MODE else 9.5,
        "earth_x_start": 560 if not QUICK_MODE else 276,
        "earth_x_end": 565 if not QUICK_MODE else 278,
        "earth_y_start": 1080 if not QUICK_MODE else 542,
        "earth_y_end": 1040 if not QUICK_MODE else 520,
        "zoom_start": 1.14,
        "zoom_end": 0.96,
        "caption": "HOW FAST IS THAT? // SCALE COMPARISON",
    },
    {
        "name": "finale",
        "start": 47.0 if not QUICK_MODE else 9.5,
        "end": CONFIG["duration_s"],
        "earth_x_start": 565 if not QUICK_MODE else 278,
        "earth_x_end": 540 if not QUICK_MODE else 270,
        "earth_y_start": 1040 if not QUICK_MODE else 520,
        "earth_y_end": 1110 if not QUICK_MODE else 556,
        "zoom_start": 0.96,
        "zoom_end": 0.84,
        "caption": "RECEDING, NOT UNBOUND // EARTH STILL HOLDS THE MOON",
    },
]


# %% Utilities

def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(t):
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def ease_in_out_sine(t):
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


def get_shot(t: float) -> Dict:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def shot_state(t: float):
    shot = get_shot(t)
    duration = max(shot["end"] - shot["start"], 1e-6)
    u = (t - shot["start"]) / duration
    e = ease_in_out_sine(u)
    earth_x = lerp(shot["earth_x_start"], shot["earth_x_end"], e)
    earth_y = lerp(shot["earth_y_start"], shot["earth_y_end"], e)
    zoom = lerp(shot["zoom_start"], shot["zoom_end"], e)
    return shot, earth_x, earth_y, zoom


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cx = width / 2.0
    cy = height / 2.0
    nx = (xx - cx) / (width / 2.0)
    ny = (yy - cy) / (height / 2.0)
    radius = np.sqrt(nx**2 + ny**2)
    return np.clip(1.0 - strength * radius**1.8, 0.0, 1.0).astype(np.float32)


def apply_grade(arr: np.ndarray) -> np.ndarray:
    image = Image.fromarray(arr)
    image = ImageEnhance.Contrast(image).enhance(CONFIG["contrast_boost"])
    image = ImageEnhance.Color(image).enhance(CONFIG["saturation_boost"])
    return np.array(image)


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
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
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
    xy: Tuple[int, int],
    size: int = 42,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    stroke: int = 2,
    anchor: str = "la",
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
    draw.text(
        xy,
        text,
        font=font,
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(fill[3] if len(fill) > 3 else 255, 220)),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int = 30,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    line_spacing: int = 8,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
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
        y += (bbox[3] - bbox[1]) + line_spacing


def orbital_escape_speed_km_s(radius_km: float) -> float:
    mu = float(CONFIG["earth_mu_km3_s2"])
    return math.sqrt(2.0 * mu / radius_km)


def format_distance_km(value: float) -> str:
    return f"{value:,.0f} km"


def format_cm_per_year(value: float) -> str:
    return f"{value:.2f} cm/yr"


def format_km(value: float) -> str:
    return f"{value:,.1f} km"


VIGNETTE = make_vignette(OUT_SIZE[0], OUT_SIZE[1], CONFIG["vignette_strength"])


# %% Reference data

def create_reference_data() -> pd.DataFrame:
    rate_cm_per_year = float(CONFIG["lunar_recession_cm_per_year"])
    base_km = float(CONFIG["mean_earth_moon_distance_km"])
    years = [0, 1, 10, 100, 1_000, 10_000, 1_000_000, 10_000_000, 100_000_000]

    rows = []
    for year in years:
        added_cm = rate_cm_per_year * year
        added_m = added_cm / 100.0
        added_km = added_cm / 100_000.0
        rows.append({
            "years_after_present": int(year),
            "added_distance_cm": float(added_cm),
            "added_distance_m": float(added_m),
            "added_distance_km": float(added_km),
            "approx_mean_distance_km_if_linear": float(base_km + added_km),
            "note": "Uses present-day measured recession rate as a simple linear illustration.",
        })

    df = pd.DataFrame(rows)
    csv_path = DATA_ROOT / "moon_recession_reference_data.csv"
    df.to_csv(csv_path, index=False)
    print("Reference data written:", csv_path.resolve())
    return df


def create_scientific_previews(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    subset = df[df["years_after_present"] <= 100_000_000].copy()
    x = subset["years_after_present"].to_numpy(float)
    y = subset["added_distance_km"].to_numpy(float)
    ax.plot(x, y, marker="o")
    ax.set_xscale("log")
    ax.set_title("Linear illustration of lunar recession at 3.82 cm/year")
    ax.set_xlabel("Years after present")
    ax.set_ylabel("Extra Earth-Moon distance [km]")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "recession_linear_illustration.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sample_years = np.linspace(0, 100_000_000, 200)
    day_ms = (sample_years / 100.0) * float(CONFIG["day_length_increase_ms_per_century"])
    ax.plot(sample_years, day_ms)
    ax.set_title("Illustrative Earth day-length increase")
    ax.set_xlabel("Years after present")
    ax.set_ylabel("Extra day length [milliseconds]")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "day_length_illustration.png", dpi=170)
    plt.close(fig)

    print("Scientific preview plots written to:", PREVIEW_DIR.resolve())


# %% Subtitle helpers

def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(captions, path: Path):
    lines = []
    for index, (start, end, text) in enumerate(captions, start=1):
        lines.append(str(index))
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# %% Scene

class MoonEscapeScene:
    def __init__(self, ref_df: pd.DataFrame):
        self.ref_df = ref_df.copy()
        self.particles = self._make_particles(int(CONFIG["background_particle_count"]), seed=42)
        self.hud_noise = self._make_hud_noise(int(CONFIG["hud_noise_count"]), seed=101)

    @staticmethod
    def _make_particles(count: int, seed: int):
        rng = np.random.default_rng(seed)
        particles = []
        for _ in range(count):
            particles.append({
                "x": float(rng.uniform(0, OUT_SIZE[0])),
                "y": float(rng.uniform(0, OUT_SIZE[1])),
                "r": float(rng.uniform(0.5, 2.0)),
                "a": int(rng.integers(18, 105)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "drift": float(rng.uniform(-12, 12)),
            })
        return particles

    @staticmethod
    def _make_hud_noise(count: int, seed: int):
        rng = np.random.default_rng(seed)
        noise = []
        for _ in range(count):
            noise.append({
                "x": float(rng.uniform(0, OUT_SIZE[0])),
                "y": float(rng.uniform(0, OUT_SIZE[1])),
                "length": float(rng.uniform(10, 90)),
                "alpha": int(rng.integers(10, 55)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            })
        return noise

    def render_background(self, t: float) -> Image.Image:
        canvas = Image.new("RGBA", OUT_SIZE, (2, 6, 14, 255))
        draw = ImageDraw.Draw(canvas)

        for particle in self.particles:
            x = (particle["x"] + particle["drift"] * 0.07 * t) % OUT_SIZE[0]
            y = (particle["y"] + particle["drift"] * 0.02 * t) % OUT_SIZE[1]
            twinkle = 0.65 + 0.35 * math.sin(t * 1.4 + particle["phase"])
            alpha = int(particle["a"] * twinkle)
            r = particle["r"]
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(215, 230, 255, alpha))

        # faint nebula / space glow
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        centers = [
            (OUT_SIZE[0]*0.52, OUT_SIZE[1]*0.55, (20, 90, 140)),
            (OUT_SIZE[0]*0.18, OUT_SIZE[1]*0.28, (75, 25, 110)),
            (OUT_SIZE[0]*0.86, OUT_SIZE[1]*0.73, (12, 45, 115)),
        ]
        for cx, cy, color in centers:
            for radius, alpha in [(660, 16), (430, 22), (250, 30)]:
                gd.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(color[0], color[1], color[2], alpha))
        glow = glow.filter(ImageFilter.GaussianBlur(85))
        canvas.alpha_composite(glow)
        return canvas

    def moon_state(self, t: float, earth_x: float, earth_y: float, zoom: float):
        # Exaggerated orbital motion and outward drift to make the idea readable.
        fraction = t / max(CONFIG["duration_s"], 1e-6)
        drift_px = lerp(0.0, 86.0 * (OUT_SIZE[0] / 1080.0), smoothstep((t - 20.0) / 28.0))
        base_r = 340.0 * zoom * (OUT_SIZE[0] / 1080.0)
        radius = base_r + drift_px
        angle = -0.18 + fraction * 2.35 * math.pi
        moon_x = earth_x + math.cos(angle) * radius
        moon_y = earth_y + math.sin(angle) * radius * 0.58
        return moon_x, moon_y, radius, angle

    def draw_earth(self, canvas: Image.Image, earth_x: float, earth_y: float, zoom: float, t: float, show_tides: bool):
        earth_r = 210 * zoom * (OUT_SIZE[0] / 1080.0)
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for scale, alpha in [(1.35, 18), (1.18, 28), (1.07, 55)]:
            r = earth_r * scale
            gd.ellipse((earth_x-r, earth_y-r, earth_x+r, earth_y+r), fill=(30, 140, 255, alpha))
        glow = glow.filter(ImageFilter.GaussianBlur(28))
        canvas.alpha_composite(glow)

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        if show_tides:
            # Tidal bulges slightly ahead of the Moon / rotation direction
            bulge_angle = t * 0.35 + math.radians(18)
            bulge_scale_x = 1.09
            bulge_scale_y = 0.94
            temp = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            td = ImageDraw.Draw(temp)
            td.ellipse(
                (earth_x-earth_r*bulge_scale_x, earth_y-earth_r*bulge_scale_y,
                 earth_x+earth_r*bulge_scale_x, earth_y+earth_r*bulge_scale_y),
                fill=(18, 72, 138, 220),
                outline=(75, 215, 255, 120),
                width=max(1, int(2*zoom)),
            )
            temp = temp.rotate(math.degrees(bulge_angle), center=(earth_x, earth_y), resample=Image.Resampling.BICUBIC)
            canvas.alpha_composite(temp)

        draw.ellipse(
            (earth_x-earth_r, earth_y-earth_r, earth_x+earth_r, earth_y+earth_r),
            fill=(8, 39, 92, 255), outline=(92, 220, 255, 240), width=max(2, int(3*zoom)),
        )

        # night side shadow
        shadow_offset = earth_r * 0.42
        draw.ellipse(
            (earth_x-earth_r+shadow_offset, earth_y-earth_r*1.02,
             earth_x+earth_r*1.18+shadow_offset, earth_y+earth_r*1.02),
            fill=(1, 5, 17, 208),
        )

        for lat in [-60, -30, 0, 30, 60]:
            y = earth_y + math.sin(math.radians(lat)) * earth_r * 0.8
            band_width = math.cos(math.radians(lat)) * earth_r * 1.7
            draw.arc((earth_x-band_width/2, y-earth_r*0.16, earth_x+band_width/2, y+earth_r*0.16),
                     0, 180, fill=(55, 155, 225, 86), width=max(1, int(2*zoom)))

        rotation = (t * 8.0) % 180.0
        for offset in [-70, -35, 0, 35, 70]:
            x = earth_x + math.sin(math.radians(rotation + offset)) * earth_r * 0.57
            squash = 0.30 + 0.65 * abs(math.cos(math.radians(rotation + offset)))
            draw.ellipse((x-earth_r*0.15*squash, earth_y-earth_r*0.86,
                          x+earth_r*0.15*squash, earth_y+earth_r*0.86),
                         outline=(52, 160, 222, 55), width=1)

        canvas.alpha_composite(layer)

    def draw_moon(self, canvas: Image.Image, moon_x: float, moon_y: float, zoom: float, t: float):
        moon_r = 52 * zoom * (OUT_SIZE[0] / 1080.0)
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for scale, alpha in [(1.7, 9), (1.4, 17), (1.15, 28)]:
            r = moon_r * scale
            gd.ellipse((moon_x-r, moon_y-r, moon_x+r, moon_y+r), fill=(190, 200, 220, alpha))
        glow = glow.filter(ImageFilter.GaussianBlur(12))
        canvas.alpha_composite(glow)

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.ellipse((moon_x-moon_r, moon_y-moon_r, moon_x+moon_r, moon_y+moon_r), fill=(176, 180, 190, 255), outline=(235, 235, 240, 180), width=1)

        # stylized craters
        crater_specs = [(-0.25, -0.18, 0.14), (0.15, -0.08, 0.10), (-0.10, 0.20, 0.12), (0.24, 0.17, 0.07)]
        wobble = math.sin(t * 0.7) * 0.06
        for cx, cy, rr in crater_specs:
            x = moon_x + moon_r * (cx + wobble * 0.15)
            y = moon_y + moon_r * cy
            r = moon_r * rr
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(150, 154, 164, 210))
        canvas.alpha_composite(layer)

    def draw_orbit_guides(self, canvas: Image.Image, earth_x: float, earth_y: float, zoom: float, radius: float, show_outward_arrow: bool):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        orbit_r_x = radius
        orbit_r_y = radius * 0.58
        draw.ellipse((earth_x-orbit_r_x, earth_y-orbit_r_y, earth_x+orbit_r_x, earth_y+orbit_r_y),
                     outline=(95, 210, 245, 90), width=2)
        draw.ellipse((earth_x-(orbit_r_x+40), earth_y-(orbit_r_y+24), earth_x+(orbit_r_x+40), earth_y+(orbit_r_y+24)),
                     outline=(255, 185, 80, 45), width=1)
        if show_outward_arrow:
            draw.line((earth_x+orbit_r_x*0.82, earth_y-20, earth_x+orbit_r_x+52, earth_y-58), fill=(255, 180, 80, 170), width=4)
            draw.polygon([
                (earth_x+orbit_r_x+52, earth_y-58),
                (earth_x+orbit_r_x+38, earth_y-63),
                (earth_x+orbit_r_x+46, earth_y-46),
            ], fill=(255, 180, 80, 170))
            draw_text(overlay, "higher orbit", (int(earth_x+orbit_r_x+60), int(earth_y-72)), size=20 if not QUICK_MODE else 11, fill=(255, 195, 95, 235), bold=True, stroke=1)
        canvas.alpha_composite(overlay)

    def draw_laser(self, canvas: Image.Image, earth_x: float, earth_y: float, moon_x: float, moon_y: float, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        station_x = earth_x - 38
        station_y = earth_y - 175 * (OUT_SIZE[0] / 1080.0)
        draw.rectangle((station_x-10, station_y-12, station_x+14, station_y+12), fill=(110, 220, 255, 180))

        pulse = (t * 1.8) % 1.0
        for offset in [0.0, 0.45]:
            p = (pulse + offset) % 1.0
            x = lerp(station_x, moon_x, p)
            y = lerp(station_y, moon_y, p)
            draw.line((station_x, station_y, moon_x, moon_y), fill=(95, 220, 245, 50), width=2)
            draw.ellipse((x-10, y-10, x+10, y+10), fill=(130, 240, 255, 160))

        draw_text(overlay, "laser", (int(station_x - 8), int(station_y - 28)), size=19 if not QUICK_MODE else 10, fill=(130, 240, 255, 225), bold=True, stroke=1)
        draw_text(overlay, "retroreflector", (int(moon_x + 18), int(moon_y - 8)), size=19 if not QUICK_MODE else 10, fill=(230, 235, 245, 225), bold=True, stroke=1)
        canvas.alpha_composite(overlay)

    def draw_mechanism(self, canvas: Image.Image, earth_x: float, earth_y: float, moon_x: float, moon_y: float, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Earth rotation arrow
        r = 295 * (OUT_SIZE[0] / 1080.0)
        bbox = (earth_x-r, earth_y-r, earth_x+r, earth_y+r)
        draw.arc(bbox, start=208, end=330, fill=(95, 225, 245, 170), width=5)
        ang = math.radians(330)
        tip_x = earth_x + math.cos(ang) * r
        tip_y = earth_y + math.sin(ang) * r
        draw.polygon([(tip_x, tip_y), (tip_x-20, tip_y-6), (tip_x-7, tip_y-21)], fill=(95, 225, 245, 190))
        draw_text(overlay, "Earth rotates faster", (int(earth_x - r*0.78), int(earth_y + r*0.64)), size=24 if not QUICK_MODE else 12, fill=(110, 230, 245, 235), bold=True, stroke=1)

        # Tidal bulge label and torque arrow
        draw.line((earth_x+120, earth_y-180, moon_x-40, moon_y-26), fill=(255, 180, 90, 170), width=4)
        draw.polygon([(moon_x-40, moon_y-26), (moon_x-56, moon_y-35), (moon_x-45, moon_y-10)], fill=(255, 180, 90, 190))
        draw_text(overlay, "tidal tug adds orbital energy", (int(earth_x+40), int(earth_y-230)), size=23 if not QUICK_MODE else 12, fill=(255, 200, 110, 235), bold=True, stroke=1)
        canvas.alpha_composite(overlay)

    def draw_scale_panel(self, canvas: Image.Image, t: float):
        alpha = int(235 * smoothstep((t - 30.5) / 3.0))
        if alpha <= 4:
            return
        x0 = 55 if not QUICK_MODE else 28
        y0 = 300 if not QUICK_MODE else 152
        width = 460 if not QUICK_MODE else 230
        row_h = 52 if not QUICK_MODE else 26

        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((x0-18, y0-52, x0+width, y0+row_h*4+28), radius=24 if not QUICK_MODE else 12,
                               fill=(1, 5, 14, 176), outline=(255, 175, 90, 96), width=1)
        canvas.alpha_composite(panel)

        draw_text(canvas, "IF TODAY'S RATE STAYED CONSTANT", (x0, y0-36), size=23 if not QUICK_MODE else 12,
                  fill=(255, 195, 105, alpha), bold=True, stroke=1)

        rows = [
            ("1 year", "3.82 cm farther"),
            ("100 years", "3.82 m farther"),
            ("1 million years", "38.2 km farther"),
            ("100 million years", "3,820 km farther"),
        ]
        for idx, (left, right) in enumerate(rows):
            y = y0 + idx * row_h
            draw_text(canvas, left, (x0, y), size=22 if not QUICK_MODE else 11, fill=(235, 245, 255, alpha), bold=True, stroke=1)
            draw_text(canvas, right, (x0 + width - 16, y), size=22 if not QUICK_MODE else 11,
                      fill=(110, 230, 255, alpha), bold=True, stroke=1, anchor="ra")

    def draw_status_panel(self, canvas: Image.Image, t: float):
        alpha = int(235 * smoothstep((t - 44.0) / 4.0))
        if alpha <= 4:
            return
        x0 = OUT_SIZE[0] - (470 if not QUICK_MODE else 236)
        y0 = 260 if not QUICK_MODE else 132
        width = 420 if not QUICK_MODE else 212

        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((x0-16, y0-48, x0+width, y0+190 if not QUICK_MODE else y0+100),
                               radius=22 if not QUICK_MODE else 11,
                               fill=(2, 6, 15, 175), outline=(80, 200, 230, 90), width=1)
        canvas.alpha_composite(panel)

        escape_speed = orbital_escape_speed_km_s(float(CONFIG["mean_earth_moon_distance_km"]))
        orbital_speed = float(CONFIG["mean_lunar_orbital_speed_km_s"])
        rows = [
            ("mean distance", format_distance_km(CONFIG["mean_earth_moon_distance_km"])),
            ("recession", format_cm_per_year(CONFIG["lunar_recession_cm_per_year"])),
            ("Moon orbital speed", f"{orbital_speed:.3f} km/s"),
            ("local escape speed", f"{escape_speed:.3f} km/s"),
        ]

        draw_text(canvas, "STILL GRAVITATIONALLY BOUND", (x0, y0-32), size=22 if not QUICK_MODE else 11,
                  fill=(110, 235, 245, alpha), bold=True, stroke=1)
        step = 36 if not QUICK_MODE else 18
        for i, (left, right) in enumerate(rows):
            y = y0 + i * step
            draw_text(canvas, left, (x0, y), size=20 if not QUICK_MODE else 10, fill=(220, 234, 245, alpha), stroke=1)
            draw_text(canvas, right, (x0 + width - 10, y), size=20 if not QUICK_MODE else 10, fill=(255, 205, 110, alpha), bold=True, stroke=1, anchor="ra")

        bound_text = "Orbit is widening, not unbinding"
        draw_wrapped_text(canvas, bound_text, (x0, y0 + (150 if not QUICK_MODE else 76)), width-16,
                          size=19 if not QUICK_MODE else 10, fill=(245, 250, 255, alpha), bold=True)

    def draw_timeline(self, canvas: Image.Image, t: float):
        x0 = 76 if not QUICK_MODE else 40
        x1 = OUT_SIZE[0] - (76 if not QUICK_MODE else 40)
        y = OUT_SIZE[1] - (320 if not QUICK_MODE else 160)
        width = x1 - x0
        d = ImageDraw.Draw(canvas)
        d.line((x0, y, x1, y), fill=(90, 195, 225, 220), width=2)
        chapter_times = [s["start"] for s in SHOT_PLAN] + [CONFIG["duration_s"]]
        chapter_names = [s["name"].upper() for s in SHOT_PLAN]
        for idx, name in enumerate(chapter_names):
            frac = chapter_times[idx] / CONFIG["duration_s"]
            x = x0 + frac * width
            d.line((x, y-12, x, y+12), fill=(80, 205, 235, 100), width=1)
            draw_text(canvas, name, (int(x), y+24), size=15 if not QUICK_MODE else 8, fill=(150, 205, 225, 210), stroke=1, anchor="ma")
        fraction = t / max(CONFIG["duration_s"], 1e-6)
        cursor_x = x0 + fraction * width
        d.line((cursor_x, y-36, cursor_x, y+36), fill=(255, 180, 85, 245), width=3)
        draw_text(canvas, "STORY TIMELINE", (x0, y-64), size=20 if not QUICK_MODE else 10, fill=(165, 220, 240, 220), bold=True, stroke=1)

    def draw_corner_hud(self, canvas: Image.Image, t: float):
        draw_text(canvas, f"DRIFT // {CONFIG['lunar_recession_cm_per_year']:.2f} CM/YR", (OUT_SIZE[0]-55 if not QUICK_MODE else OUT_SIZE[0]-28, 83 if not QUICK_MODE else 42),
                  size=22 if not QUICK_MODE else 11, fill=(105, 230, 245, 220), bold=True, stroke=1, anchor="ra")
        draw_text(canvas, "MOON LASER RANGE // DIRECT", (OUT_SIZE[0]-55 if not QUICK_MODE else OUT_SIZE[0]-28, 118 if not QUICK_MODE else 60),
                  size=18 if not QUICK_MODE else 9, fill=(150, 205, 225, 200), stroke=1, anchor="ra")
        sweep = int((t * 17.0) % 100)
        draw_text(canvas, f"SCAN // {sweep:02d}", (OUT_SIZE[0]-55 if not QUICK_MODE else OUT_SIZE[0]-28, 149 if not QUICK_MODE else 76),
                  size=18 if not QUICK_MODE else 9, fill=(150, 205, 225, 190), stroke=1, anchor="ra")

    def draw_text_layers(self, canvas: Image.Image, t: float, shot: Dict):
        title_alpha = int(255 * smoothstep((t - 0.25) / 1.0) * (1.0 - smoothstep((t - ((7.0 if not QUICK_MODE else 1.8))) / 0.9)))
        if title_alpha > 4:
            draw_text(canvas, CONFIG["title_text"], (54 if not QUICK_MODE else 28, 102 if not QUICK_MODE else 52), size=48 if not QUICK_MODE else 22,
                      fill=(245, 250, 255, title_alpha), bold=True)
            draw_text(canvas, CONFIG["subtitle_text"], (58 if not QUICK_MODE else 30, 167 if not QUICK_MODE else 84), size=24 if not QUICK_MODE else 11,
                      fill=(100, 230, 245, min(title_alpha, 225)), bold=True)

        if t > (5.2 if not QUICK_MODE else 1.6):
            draw_text(canvas, shot["caption"], (54 if not QUICK_MODE else 28, 65 if not QUICK_MODE else 32), size=21 if not QUICK_MODE else 10,
                      fill=(140, 215, 235, 205), bold=True, stroke=1)

        caption = caption_at(t)
        if caption:
            panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            draw = ImageDraw.Draw(panel)
            y0 = OUT_SIZE[1] - (246 if not QUICK_MODE else 126)
            draw.rounded_rectangle((45 if not QUICK_MODE else 22, y0, OUT_SIZE[0] - (45 if not QUICK_MODE else 22), y0 + (126 if not QUICK_MODE else 68)),
                                   radius=24 if not QUICK_MODE else 12,
                                   fill=(1, 4, 12, 170), outline=(55, 180, 220, 70), width=1)
            canvas.alpha_composite(panel)
            draw_wrapped_text(canvas, caption, (70 if not QUICK_MODE else 34, y0 + (28 if not QUICK_MODE else 14)),
                              max_width=OUT_SIZE[0] - (140 if not QUICK_MODE else 68),
                              size=30 if not QUICK_MODE else 14, fill=(245, 250, 255, 245))

        note_alpha = int(220 * smoothstep((t - (50.0 if not QUICK_MODE else 10.0)) / 3.0))
        if note_alpha > 4:
            draw_wrapped_text(canvas, CONFIG["credit_text"], (65 if not QUICK_MODE else 32, OUT_SIZE[1] - (112 if not QUICK_MODE else 58)),
                              max_width=OUT_SIZE[0] - (140 if not QUICK_MODE else 70), size=18 if not QUICK_MODE else 9,
                              fill=(220, 232, 245, note_alpha))
            draw_wrapped_text(canvas, CONFIG["scientific_note"], (65 if not QUICK_MODE else 32, OUT_SIZE[1] - (80 if not QUICK_MODE else 40)),
                              max_width=OUT_SIZE[0] - (140 if not QUICK_MODE else 70), size=16 if not QUICK_MODE else 8,
                              fill=(190, 210, 232, note_alpha))

    def draw_hud_noise(self, canvas: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for item in self.hud_noise:
            pulse = 0.5 + 0.5 * math.sin(t * 1.9 + item["phase"])
            if pulse < 0.72:
                continue
            x = item["x"]
            y = (item["y"] + t * 12.0) % OUT_SIZE[1]
            draw.line((x, y, x + item["length"], y), fill=(90, 210, 240, int(item["alpha"] * pulse)), width=1)
        offset = int((t * 41) % 7)
        for y in range(offset, OUT_SIZE[1], 7):
            draw.line((0, y, OUT_SIZE[0], y), fill=(120, 205, 245, 13), width=1)
        scan_y = int((t * 158) % (OUT_SIZE[1] + 260)) - 130
        draw.rectangle((0, scan_y, OUT_SIZE[0], scan_y + 54), fill=(80, 210, 240, 9))
        canvas.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot, earth_x, earth_y, zoom = shot_state(t)
        canvas = self.render_background(t)

        moon_x, moon_y, orbit_radius, moon_angle = self.moon_state(t, earth_x, earth_y, zoom)

        show_tides = shot["name"] in {"tides", "scale", "finale"}
        self.draw_earth(canvas, earth_x, earth_y, zoom, t, show_tides=show_tides)
        self.draw_orbit_guides(canvas, earth_x, earth_y, zoom, orbit_radius, show_outward_arrow=shot["name"] in {"tides", "scale", "finale"})
        self.draw_moon(canvas, moon_x, moon_y, zoom, t)

        if shot["name"] == "laser":
            self.draw_laser(canvas, earth_x, earth_y, moon_x, moon_y, t)
        if shot["name"] in {"tides", "scale", "finale"}:
            self.draw_mechanism(canvas, earth_x, earth_y, moon_x, moon_y, t)

        self.draw_scale_panel(canvas, t)
        self.draw_status_panel(canvas, t)
        self.draw_timeline(canvas, t)
        self.draw_corner_hud(canvas, t)
        self.draw_text_layers(canvas, t, shot)
        self.draw_hud_noise(canvas, t)

        arr = np.array(canvas.convert("RGB"))
        arr = apply_grade(arr)
        arr = np.clip(arr.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)

        fade_in = smoothstep(t / 0.9)
        fade_out = 1.0 - smoothstep((t - (CONFIG["duration_s"] - 1.1)) / 1.0)
        arr = np.clip(arr.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)
        return arr


# %% Video render

def run_ffmpeg(command: List[str]):
    print("Running:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def render_video(scene: MoonEscapeScene):
    raw_video_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    final_video_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    print("Subtitle sidecar written:", srt_path.resolve())

    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    times = np.arange(frame_count) / CONFIG["fps"]
    print(f"Rendering {frame_count:,} frames at {CONFIG['video_width']}×{CONFIG['video_height']} ...")

    with iio.get_writer(raw_video_path, fps=CONFIG["fps"], codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None) as writer:
        for t in tqdm(times, desc="Rendering moon recession short"):
            writer.append_data(scene.render_frame(float(t)))

    print("Raw video written:", raw_video_path.resolve())
    shutil.copyfile(raw_video_path, final_video_path)
    print("Final video:", final_video_path.resolve())
    return final_video_path


# %% Main

def main():
    print("Starting Moon recession short pipeline ...")
    ref_df = create_reference_data()
    create_scientific_previews(ref_df)
    scene = MoonEscapeScene(ref_df)

    preview_times = [1.0, min(10.0, CONFIG["duration_s"] * 0.2), min(20.0, CONFIG["duration_s"] * 0.4), min(33.0, CONFIG["duration_s"] * 0.58), min(46.0, CONFIG["duration_s"] * 0.8), CONFIG["duration_s"] - 1.0]
    for preview_time in tqdm(preview_times, desc="Preview frames"):
        frame = scene.render_frame(float(preview_time))
        Image.fromarray(frame).save(PREVIEW_DIR / f"preview_{int(preview_time):02d}s.png")
    print("Preview frames written to:", PREVIEW_DIR.resolve())

    render_video(scene)

    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-", path.name)


if __name__ == "__main__":
    main()
