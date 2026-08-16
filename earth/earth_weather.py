from __future__ import annotations

"""
Earth, Weather, Climate & Oceans — One Connected System
Cinematic vertical YouTube Shorts renderer.

Creates a 1080x1920 science short (optional 4K / quick preview) using only
procedural visuals and sound. No external images or copyrighted music needed.

Scientific framing
------------------
- Weather: local atmospheric conditions over short timescales.
- Climate: expected patterns measured over much longer periods.
- Ocean: covers about 71% of Earth and absorbs around 90% of the excess heat
  associated with planetary warming.

Sources used for the narrative framing:
- NASA: Weather vs. climate
  https://science.nasa.gov/climate-change/faq/whats-the-difference-between-weather-and-climate/
- NOAA: One global ocean covers about 71% of Earth
  https://oceanservice.noaa.gov/news/june17/30days.html
- NASA: Ocean and climate change / ocean heat uptake
  https://science.nasa.gov/earth/explore/the-ocean-and-climate-change/

Usage
-----
Full render:
    python earth_weather_climate_oceans_cinematic_short.py

Quick preview:
    EARTH_SYSTEM_SHORT_QUICK=1 python earth_weather_climate_oceans_cinematic_short.py

4K vertical:
    EARTH_SYSTEM_SHORT_4K=1 python earth_weather_climate_oceans_cinematic_short.py
"""

import json
import math
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("EARTH_SYSTEM_SHORT_QUICK", "0") == "1"
FOUR_K_MODE = os.environ.get("EARTH_SYSTEM_SHORT_4K", "0") == "1" and not QUICK_MODE

OUTPUT_ROOT = Path("earth_weather_climate_oceans_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else (2160 if FOUR_K_MODE else 1080),
    "video_height": 960 if QUICK_MODE else (3840 if FOUR_K_MODE else 1920),
    "fps": 8 if QUICK_MODE else 24,
    "duration_s": 13 if QUICK_MODE else 52,
    "audio_rate": 44100,
    "title": "EARTH, WEATHER, CLIMATE & OCEANS",
    "subtitle": "one connected planetary system",
    "output_basename": "earth_weather_climate_oceans_one_connected_system",
    "stars": 180 if QUICK_MODE else 720,
    "vignette": 0.24,
    "contrast": 1.08,
    "saturation": 1.06,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
FPS = int(CONFIG["fps"])
DURATION = float(CONFIG["duration_s"])
FRAME_COUNT = int(round(FPS * DURATION))

COLORS = {
    "white": (245, 249, 250),
    "muted": (177, 204, 214),
    "cyan": (88, 220, 255),
    "blue": (45, 116, 232),
    "deep_blue": (5, 35, 86),
    "green": (83, 204, 133),
    "gold": (255, 190, 76),
    "orange": (255, 126, 62),
    "red": (255, 84, 78),
    "dark": (2, 7, 16),
}

FULL_CAPTIONS = [
    (0.2, 3.8, "Earth does not have separate weather, climate, and ocean systems. It has one connected engine."),
    (4.3, 7.9, "Weather is local and fast — clouds, rain, wind, heat, and storms changing from hour to hour."),
    (10.2, 13.8, "Climate is the long-term pattern: what conditions a place usually experiences over many years."),
    (16.0, 19.6, "About seventy-one percent of Earth's surface is ocean, so water dominates the planet's energy flow."),
    (23.0, 26.6, "The ocean absorbs around ninety percent of the excess heat from planetary warming."),
    (29.2, 32.8, "Currents move that heat across the globe, shaping temperatures, rainfall, and entire weather systems."),
    (36.0, 39.6, "Evaporation feeds clouds. Winds drive waves. Ice changes reflectivity. Every part pushes on the others."),
    (43.2, 47.0, "To understand Earth's weather or climate, you have to follow the heat — especially through the ocean."),
]

SHOT_PLAN_FULL = [
    ("intro", 0.0, 4.2),
    ("weather", 4.2, 10.0),
    ("climate", 10.0, 16.0),
    ("ocean_share", 16.0, 22.5),
    ("ocean_heat", 22.5, 29.0),
    ("currents", 29.0, 35.5),
    ("water_cycle", 35.5, 42.5),
    ("finale", 42.5, 52.0),
]

if QUICK_MODE:
    scale = DURATION / 52.0
    CAPTIONS = [(a * scale, b * scale, text) for a, b, text in FULL_CAPTIONS]
    SHOT_PLAN = [(name, a * scale, b * scale) for name, a, b in SHOT_PLAN_FULL]
else:
    CAPTIONS = FULL_CAPTIONS
    SHOT_PLAN = SHOT_PLAN_FULL

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


def get_shot(t: float) -> Tuple[str, float, float]:
    for shot in SHOT_PLAN:
        if shot[1] <= t < shot[2]:
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
            pass
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int,
    fill,
    bold: bool = False,
    anchor: str = "la",
    stroke: int = 2,
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
    box: Tuple[int, int, int, int],
    size: int,
    fill,
    bold: bool = False,
):
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bb = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)
        if bb[2] - bb[0] <= x1 - x0:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    y = y0
    for line in lines:
        draw.text(
            (x0, y), line, font=font, fill=fill,
            stroke_width=2, stroke_fill=(0, 0, 0, 210),
        )
        bb = draw.textbbox((x0, y), line, font=font, stroke_width=2)
        y += bb[3] - bb[1] + max(4, int(size * 0.18))
        if y > y1:
            break


def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 130):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius = max(8, int((box[3] - box[1]) * 0.16))
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=(2, 9, 22, alpha),
        outline=(90, 210, 240, 55),
        width=1,
    )
    image.alpha_composite(overlay)


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius ** 1.8, 0.0, 1.0).astype(np.float32)


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000.0))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path):
    lines: List[str] = []
    for index, (start, end, text) in enumerate(captions, start=1):
        lines.extend([
            str(index),
            f"{format_srt_time(start)} --> {format_srt_time(end)}",
            text,
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


@dataclass
class Star:
    x: float
    y: float
    radius: float
    alpha: float
    phase: float


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------


class EarthSystemScene:
    def __init__(self):
        self.stars = self._make_stars(int(CONFIG["stars"]), seed=2026)
        self.globe_cache: Dict[Tuple[int, int], Image.Image] = {}
        self.cloud_noise = self._make_noise(512, 256, seed=91)
        self.ocean_noise = self._make_noise(512, 256, seed=44)
        self.continent_noise = self._make_noise(512, 256, seed=18)
        self.current_paths = self._make_current_paths()

    @staticmethod
    def _make_stars(count: int, seed: int) -> List[Star]:
        rng = np.random.default_rng(seed)
        stars: List[Star] = []
        for _ in range(count):
            stars.append(
                Star(
                    x=float(rng.uniform(0, OUT_W)),
                    y=float(rng.uniform(0, OUT_H)),
                    radius=float(rng.uniform(0.3, 2.0) * OUT_W / 1080),
                    alpha=float(rng.uniform(16, 120)),
                    phase=float(rng.uniform(0, 2 * math.pi)),
                )
            )
        return stars

    @staticmethod
    def _make_noise(width: int, height: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        source = rng.random((height, width), dtype=np.float32)
        image = Image.fromarray(np.uint8(source * 255))
        accum = np.zeros((height, width), dtype=np.float32)
        for blur, weight in [(28, 0.50), (14, 0.28), (6, 0.15), (2, 0.07)]:
            layer = np.asarray(image.filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32) / 255.0
            accum += layer * weight
        accum -= accum.min()
        accum /= max(float(accum.max()), 1e-6)
        return accum

    @staticmethod
    def _make_current_paths() -> List[List[Tuple[float, float]]]:
        return [
            [(0.08, 0.50), (0.20, 0.43), (0.31, 0.39), (0.42, 0.43), (0.53, 0.50)],
            [(0.53, 0.50), (0.63, 0.58), (0.75, 0.60), (0.86, 0.54), (0.93, 0.46)],
            [(0.18, 0.66), (0.31, 0.73), (0.45, 0.70), (0.57, 0.64)],
            [(0.57, 0.34), (0.70, 0.29), (0.83, 0.34), (0.93, 0.43)],
        ]

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", (OUT_W, OUT_H), COLORS["dark"] + (255,))
        draw = ImageDraw.Draw(image)

        for y in range(OUT_H):
            p = y / max(OUT_H - 1, 1)
            color = (
                int(lerp(2, 4, p)),
                int(lerp(7, 15, p)),
                int(lerp(16, 30, p)),
                255,
            )
            draw.line((0, y, OUT_W, y), fill=color)

        for star in self.stars:
            alpha = int(star.alpha * (0.74 + 0.26 * math.sin(star.phase + t * 1.2)))
            r = star.radius
            draw.ellipse(
                (star.x - r, star.y - r, star.x + r, star.y + r),
                fill=COLORS["white"] + (alpha,),
            )

        haze = Image.new("RGBA", image.size, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for cx, cy, color in [
            (OUT_W * 0.18, OUT_H * 0.28, (10, 65, 120)),
            (OUT_W * 0.80, OUT_H * 0.45, (20, 95, 120)),
            (OUT_W * 0.50, OUT_H * 0.78, (8, 58, 95)),
        ]:
            radius = OUT_W * 0.32
            hd.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=color + (18,),
            )
        image.alpha_composite(haze.filter(ImageFilter.GaussianBlur(max(20, int(60 * OUT_W / 1080)))))
        return image

    def _earth_texture(self, size: int, rotation_deg: float, cloud_shift: float, heat: float = 0.0) -> Image.Image:
        radius = size // 2
        yy, xx = np.mgrid[0:size, 0:size]
        nx = (xx - radius + 0.5) / radius
        ny = (yy - radius + 0.5) / radius
        r2 = nx * nx + ny * ny
        mask = r2 <= 1.0
        z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

        lon = np.arctan2(nx, z) / (2 * math.pi) + 0.5 + rotation_deg / 360.0
        lat = np.arcsin(np.clip(-ny, -1.0, 1.0)) / math.pi + 0.5
        tw = self.continent_noise.shape[1]
        th = self.continent_noise.shape[0]
        tx = (lon * tw).astype(int) % tw
        ty = np.clip((lat * th).astype(int), 0, th - 1)

        continents_raw = self.continent_noise[ty, tx]
        # Latitude weighting creates recognisable broad land masses rather than random speckle.
        land = (continents_raw + 0.11 * np.sin((lon * 7 + lat * 3) * math.pi) > 0.57) & mask
        ice = ((np.abs(ny) > 0.80) & mask)

        ocean_tex = self.ocean_noise[ty, tx]
        cloud_tx = ((lon + cloud_shift) * tw).astype(int) % tw
        clouds_raw = self.cloud_noise[ty, cloud_tx]
        clouds = np.clip((clouds_raw - 0.54) * 3.0, 0.0, 1.0) * mask

        light = np.clip(0.18 + 0.86 * (0.56 * nx - 0.18 * ny + 0.80 * z), 0.0, 1.0)
        rim = np.clip((1.0 - z) ** 1.65, 0.0, 1.0)

        rgb = np.zeros((size, size, 4), dtype=np.float32)
        # Ocean
        rgb[..., 0] = 5 + 14 * ocean_tex + 35 * light
        rgb[..., 1] = 32 + 70 * ocean_tex + 85 * light
        rgb[..., 2] = 72 + 95 * ocean_tex + 115 * light
        # Land
        rgb[..., 0][land] = 38 + 60 * light[land] + 35 * heat
        rgb[..., 1][land] = 78 + 90 * light[land] - 15 * heat
        rgb[..., 2][land] = 48 + 40 * light[land] - 25 * heat
        # Arid regions
        dry = land & (np.sin((lon * 11 - lat * 5) * math.pi) > 0.45)
        rgb[..., 0][dry] += 45
        rgb[..., 1][dry] += 25
        rgb[..., 2][dry] -= 5
        # Ice
        rgb[..., 0][ice] = 220
        rgb[..., 1][ice] = 237
        rgb[..., 2][ice] = 246
        # Clouds
        for channel in range(3):
            rgb[..., channel] = rgb[..., channel] * (1.0 - clouds * 0.58) + 250 * clouds * 0.58
        # Atmosphere
        rgb[..., 1] += 34 * rim
        rgb[..., 2] += 72 * rim
        rgb[..., 3] = np.where(mask, 255, 0)

        return Image.fromarray(np.uint8(np.clip(rgb, 0, 255)), mode="RGBA")

    def draw_earth(
        self,
        image: Image.Image,
        center: Tuple[int, int],
        radius: int,
        t: float,
        heat: float = 0.0,
        atmosphere_alpha: int = 150,
    ):
        globe = self._earth_texture(
            radius * 2,
            rotation_deg=t * 5.0,
            cloud_shift=t * 0.003,
            heat=heat,
        )
        image.alpha_composite(globe, (center[0] - radius, center[1] - radius))

        glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for expansion, alpha in [(0.02, atmosphere_alpha), (0.07, atmosphere_alpha // 3), (0.13, atmosphere_alpha // 7)]:
            r = radius * (1 + expansion)
            gd.ellipse(
                (center[0] - r, center[1] - r, center[0] + r, center[1] + r),
                outline=COLORS["cyan"] + (alpha,),
                width=max(1, int(radius * 0.016)),
            )
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(3, int(radius * 0.04)))))

    def draw_storm(self, image: Image.Image, center: Tuple[int, int], radius: int, t: float, strength: float):
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx, cy = center
        turns = 3.6
        points: List[Tuple[float, float]] = []
        for i in range(180):
            a = i / 179.0
            angle = a * math.pi * 2 * turns + t * 0.7
            rr = radius * (1.0 - a) * (0.92 + 0.08 * math.sin(a * 13))
            x = cx + math.cos(angle) * rr
            y = cy + math.sin(angle) * rr * 0.55
            points.append((x, y))
        draw.line(points, fill=COLORS["white"] + (int(180 * strength),), width=max(2, int(8 * OUT_W / 1080)))
        for j in range(16):
            angle = j / 16 * math.pi * 2 + t * 0.5
            rr = radius * (0.15 + 0.75 * (j / 16))
            x = cx + math.cos(angle) * rr
            y = cy + math.sin(angle) * rr * 0.55
            d = radius * 0.16
            draw.ellipse((x - d, y - d * 0.45, x + d, y + d * 0.45), fill=COLORS["white"] + (int(24 * strength),))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(4, int(12 * OUT_W / 1080)))))

    def draw_climate_band(self, image: Image.Image, box: Tuple[int, int, int, int], progress: float):
        x0, y0, x1, y1 = box
        draw = ImageDraw.Draw(image)
        panel(image, box, 115)
        mid = (y0 + y1) // 2
        draw.line((x0 + 25, mid, x1 - 25, mid), fill=(180, 225, 240, 65), width=1)

        # Thirty-year ribbon; deliberately not a fake measured dataset.
        rng = np.random.default_rng(300)
        values = []
        for i in range(90):
            seasonal = math.sin(i * 0.48) * 0.20
            slow = math.sin(i * 0.09) * 0.11
            noise = float(rng.normal(0, 0.08))
            values.append(seasonal + slow + noise)

        max_count = max(2, int(len(values) * clamp(progress)))
        pts: List[Tuple[int, int]] = []
        for i, value in enumerate(values[:max_count]):
            x = int(lerp(x0 + 25, x1 - 25, i / (len(values) - 1)))
            y = int(mid - value * (y1 - y0) * 0.75)
            pts.append((x, y))
        if len(pts) > 1:
            glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow)
            gd.line(pts, fill=COLORS["cyan"] + (90,), width=max(5, int(13 * OUT_W / 1080)))
            image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4, int(8 * OUT_W / 1080)))))
            draw.line(pts, fill=COLORS["white"] + (225,), width=max(2, int(4 * OUT_W / 1080)))

        draw_text(image, "SHORT EVENTS", (x0 + 30, y1 - 25), max(9, int(15 * OUT_W / 1080)), COLORS["cyan"] + (220,), anchor="ls", stroke=1)
        draw_text(image, "LONG-TERM PATTERN", (x1 - 30, y1 - 25), max(9, int(15 * OUT_W / 1080)), COLORS["gold"] + (220,), anchor="rs", stroke=1)

    def draw_ocean_share(self, image: Image.Image, center: Tuple[int, int], radius: int, progress: float):
        draw = ImageDraw.Draw(image)
        # Thin ring, not a big graph.
        start = -90
        total = 360 * 0.71 * progress
        box = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
        draw.arc(box, start=start, end=start + total, fill=COLORS["cyan"] + (245,), width=max(5, int(18 * OUT_W / 1080)))
        draw.arc(box, start=start + total, end=start + 360, fill=(160, 180, 180, 70), width=max(5, int(18 * OUT_W / 1080)))
        draw_text(image, "71%", center, max(20, int(84 * OUT_W / 1080)), COLORS["white"] + (245,), bold=True, anchor="mm", stroke=2)
        draw_text(image, "OCEAN", (center[0], center[1] + int(radius * 0.35)), max(10, int(24 * OUT_W / 1080)), COLORS["cyan"] + (240,), bold=True, anchor="ma", stroke=1)

    def draw_heat_meter(self, image: Image.Image, box: Tuple[int, int, int, int], progress: float):
        x0, y0, x1, y1 = box
        panel(image, box, 120)
        draw = ImageDraw.Draw(image)
        width = x1 - x0
        fill_x = int(lerp(x0 + 25, x1 - 25, 0.90 * progress))
        bar_y0 = y0 + int((y1 - y0) * 0.52)
        bar_y1 = y0 + int((y1 - y0) * 0.68)
        draw.rounded_rectangle((x0 + 25, bar_y0, x1 - 25, bar_y1), radius=max(4, int((bar_y1 - bar_y0) / 2)), fill=(20, 56, 80, 210))
        if fill_x > x0 + 25:
            draw.rounded_rectangle((x0 + 25, bar_y0, fill_x, bar_y1), radius=max(4, int((bar_y1 - bar_y0) / 2)), fill=COLORS["orange"] + (235,))
        draw_text(image, "~90% OF EXCESS HEAT", ((x0 + x1) // 2, y0 + int((y1 - y0) * 0.28)), max(12, int(28 * OUT_W / 1080)), COLORS["white"] + (240,), bold=True, anchor="mm", stroke=1)
        draw_text(image, "absorbed by the ocean", ((x0 + x1) // 2, y0 + int((y1 - y0) * 0.82)), max(10, int(20 * OUT_W / 1080)), COLORS["gold"] + (225,), anchor="mm", stroke=1)

    def draw_currents(self, image: Image.Image, box: Tuple[int, int, int, int], progress: float, t: float):
        x0, y0, x1, y1 = box
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for path_index, path in enumerate(self.current_paths):
            pts = [(lerp(x0, x1, x), lerp(y0, y1, y)) for x, y in path]
            count = max(2, int(len(pts) * clamp(progress * 1.5)))
            pts = pts[:count]
            color = COLORS["orange"] if path_index % 2 == 0 else COLORS["cyan"]
            if len(pts) > 1:
                draw.line(pts, fill=color + (190,), width=max(2, int(5 * OUT_W / 1080)), joint="curve")
                # moving particle
                segment = (t * 0.18 + path_index * 0.21) % 1.0
                p = int(segment * (len(pts) - 1))
                px, py = pts[p]
                d = max(3, int(7 * OUT_W / 1080))
                draw.ellipse((px - d, py - d, px + d, py + d), fill=COLORS["white"] + (230,), outline=color + (245,), width=1)
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1, int(2 * OUT_W / 1080)))))

    def draw_water_cycle(self, image: Image.Image, t: float, progress: float):
        horizon = int(OUT_H * 0.63)
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        # Ocean surface
        for y in range(horizon, OUT_H):
            p = (y - horizon) / max(OUT_H - horizon, 1)
            color = (
                int(lerp(8, 2, p)),
                int(lerp(70, 30, p)),
                int(lerp(130, 74, p)),
                255,
            )
            draw.line((0, y, OUT_W, y), fill=color)
        for i in range(18):
            y = horizon + i * int(OUT_H * 0.012)
            offset = math.sin(t * 1.7 + i) * OUT_W * 0.02
            draw.arc((-OUT_W * 0.1 + offset, y, OUT_W * 0.45 + offset, y + OUT_H * 0.035), 190, 350, fill=COLORS["cyan"] + (75,), width=2)
            draw.arc((OUT_W * 0.4 - offset, y, OUT_W * 1.1 - offset, y + OUT_H * 0.035), 190, 350, fill=COLORS["cyan"] + (60,), width=2)

        # Clouds
        cloud_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        cd = ImageDraw.Draw(cloud_layer)
        for j in range(8):
            cx = OUT_W * (0.15 + 0.10 * j) + math.sin(t * 0.25 + j) * OUT_W * 0.03
            cy = OUT_H * (0.30 + 0.025 * math.sin(j * 1.3))
            rx = OUT_W * (0.06 + 0.01 * (j % 3))
            ry = rx * 0.36
            cd.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=COLORS["white"] + (70,))
        cloud_layer = cloud_layer.filter(ImageFilter.GaussianBlur(max(5, int(15 * OUT_W / 1080))))
        layer.alpha_composite(cloud_layer)

        # Evaporation arrows and rain
        arrow_alpha = int(200 * progress)
        for j in range(5):
            x = int(OUT_W * (0.20 + j * 0.14))
            y0 = horizon - int(OUT_H * 0.02)
            y1 = int(OUT_H * 0.42)
            draw.line((x, y0, x, y1), fill=COLORS["cyan"] + (arrow_alpha,), width=max(2, int(3 * OUT_W / 1080)))
            draw.polygon([(x, y1 - 12), (x - 7, y1 + 3), (x + 7, y1 + 3)], fill=COLORS["cyan"] + (arrow_alpha,))
        for j in range(14):
            x = int(OUT_W * (0.12 + (j / 13) * 0.76))
            y = int(OUT_H * (0.42 + 0.02 * math.sin(j)))
            length = int(OUT_H * 0.04)
            draw.line((x, y, x - 4, y + length), fill=COLORS["white"] + (int(150 * progress),), width=1)

        image.alpha_composite(layer)

    def draw_caption(self, image: Image.Image, caption: str, t: float):
        start = end = 0.0
        for a, b, text in CAPTIONS:
            if text == caption and a <= t < b:
                start, end = a, b
                break
        alpha = int(230 * min(clamp((t - start) / 0.32), clamp((end - t) / 0.45)))
        if alpha <= 0:
            return
        box = (int(OUT_W * 0.08), int(OUT_H * 0.74), int(OUT_W * 0.92), int(OUT_H * 0.84))
        panel(image, box, min(105, alpha // 2))
        draw_wrapped_text(
            image,
            caption,
            (box[0] + 24, box[1] + 18, box[2] - 24, box[3] - 16),
            max(12, int(29 * OUT_W / 1080)),
            COLORS["white"] + (alpha,),
        )

    def frame(self, t: float) -> np.ndarray:
        shot, t0, t1 = get_shot(t)
        local = smoothstep((t - t0) / max(t1 - t0, 1e-9))
        image = self.background(t)

        if shot == "intro":
            self.draw_earth(image, (OUT_W // 2, int(OUT_H * 0.42)), int(OUT_W * 0.30), t)
            draw_text(image, "EARTH", (OUT_W // 2, int(OUT_H * 0.095)), max(22, int(78 * OUT_W / 1080)), COLORS["white"] + (245,), bold=True, anchor="ma")
            draw_text(image, "WEATHER • CLIMATE • OCEANS", (OUT_W // 2, int(OUT_H * 0.145)), max(12, int(30 * OUT_W / 1080)), COLORS["cyan"] + (235,), bold=True, anchor="ma", stroke=1)
            draw_text(image, "ONE CONNECTED SYSTEM", (OUT_W // 2, int(OUT_H * 0.69)), max(12, int(27 * OUT_W / 1080)), COLORS["gold"] + (240,), bold=True, anchor="ma", stroke=1)

        elif shot == "weather":
            self.draw_earth(image, (OUT_W // 2, int(OUT_H * 0.36)), int(OUT_W * 0.31), t)
            storm_center = (int(OUT_W * 0.58), int(OUT_H * 0.34))
            self.draw_storm(image, storm_center, int(OUT_W * 0.12), t, local)
            draw_text(image, "WEATHER", (OUT_W // 2, int(OUT_H * 0.66)), max(18, int(54 * OUT_W / 1080)), COLORS["white"] + (240,), bold=True, anchor="ma")
            draw_text(image, "LOCAL • FAST • CHANGING", (OUT_W // 2, int(OUT_H * 0.71)), max(10, int(22 * OUT_W / 1080)), COLORS["cyan"] + (230,), anchor="ma", stroke=1)

        elif shot == "climate":
            self.draw_earth(image, (OUT_W // 2, int(OUT_H * 0.29)), int(OUT_W * 0.23), t)
            draw_text(image, "CLIMATE", (OUT_W // 2, int(OUT_H * 0.50)), max(18, int(52 * OUT_W / 1080)), COLORS["white"] + (240,), bold=True, anchor="ma")
            self.draw_climate_band(image, (int(OUT_W * 0.09), int(OUT_H * 0.56), int(OUT_W * 0.91), int(OUT_H * 0.71)), local)

        elif shot == "ocean_share":
            self.draw_earth(image, (OUT_W // 2, int(OUT_H * 0.35)), int(OUT_W * 0.30), t)
            self.draw_ocean_share(image, (OUT_W // 2, int(OUT_H * 0.69)), int(OUT_W * 0.14), local)

        elif shot == "ocean_heat":
            heat = local * 0.8
            self.draw_earth(image, (OUT_W // 2, int(OUT_H * 0.30)), int(OUT_W * 0.25), t, heat=heat)
            self.draw_heat_meter(image, (int(OUT_W * 0.10), int(OUT_H * 0.56), int(OUT_W * 0.90), int(OUT_H * 0.70)), local)

        elif shot == "currents":
            self.draw_earth(image, (OUT_W // 2, int(OUT_H * 0.39)), int(OUT_W * 0.32), t)
            box = (int(OUT_W * 0.15), int(OUT_H * 0.18), int(OUT_W * 0.85), int(OUT_H * 0.60))
            self.draw_currents(image, box, local, t)
            draw_text(image, "OCEAN CURRENTS MOVE HEAT", (OUT_W // 2, int(OUT_H * 0.68)), max(13, int(31 * OUT_W / 1080)), COLORS["gold"] + (235,), bold=True, anchor="ma", stroke=1)

        elif shot == "water_cycle":
            self.draw_water_cycle(image, t, local)
            draw_text(image, "OCEAN → AIR → CLOUD → RAIN", (OUT_W // 2, int(OUT_H * 0.14)), max(14, int(35 * OUT_W / 1080)), COLORS["white"] + (240,), bold=True, anchor="ma")
            draw_text(image, "THE WATER CYCLE LINKS EVERYTHING", (OUT_W // 2, int(OUT_H * 0.19)), max(10, int(22 * OUT_W / 1080)), COLORS["cyan"] + (225,), anchor="ma", stroke=1)

        elif shot == "finale":
            self.draw_earth(image, (OUT_W // 2, int(OUT_H * 0.37)), int(OUT_W * 0.31), t)
            self.draw_currents(image, (int(OUT_W * 0.18), int(OUT_H * 0.19), int(OUT_W * 0.82), int(OUT_H * 0.56)), 1.0, t)
            draw_text(image, "FOLLOW THE HEAT", (OUT_W // 2, int(OUT_H * 0.69)), max(19, int(57 * OUT_W / 1080)), COLORS["white"] + (245,), bold=True, anchor="ma")
            draw_text(image, "especially through the ocean", (OUT_W // 2, int(OUT_H * 0.735)), max(11, int(25 * OUT_W / 1080)), COLORS["gold"] + (235,), anchor="ma", stroke=1)
            draw_text(image, "EARTH SYSTEM SCIENCE", (OUT_W // 2, int(OUT_H * 0.90)), max(10, int(22 * OUT_W / 1080)), COLORS["cyan"] + (215,), bold=True, anchor="ma", stroke=1)

        caption = caption_at(t)
        if caption:
            self.draw_caption(image, caption, t)

        # Tiny source note only; unobtrusive and always below the safe content area.
        draw_text(image, "NASA • NOAA • procedural visualization", (int(OUT_W * 0.03), int(OUT_H * 0.985)), max(7, int(12 * OUT_W / 1080)), COLORS["muted"] + (100,), anchor="ls", stroke=1)

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        arr = np.clip(arr * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        graded = Image.fromarray(arr)
        graded = ImageEnhance.Contrast(graded).enhance(float(CONFIG["contrast"]))
        graded = ImageEnhance.Color(graded).enhance(float(CONFIG["saturation"]))
        return np.asarray(graded, dtype=np.uint8)


# -----------------------------------------------------------------------------
# Audio
# -----------------------------------------------------------------------------


def fade_envelope(length: int, attack: float, release: float) -> np.ndarray:
    env = np.ones(length, dtype=np.float32)
    a = int(length * attack)
    r = int(length * release)
    if a > 0:
        env[:a] = np.linspace(0, 1, a, endpoint=False)
    if r > 0:
        env[-r:] = np.minimum(env[-r:], np.linspace(1, 0, r, endpoint=True))
    return env


def make_audio(path: Path, duration_s: float):
    sample_rate = int(CONFIG["audio_rate"])
    count = int(duration_s * sample_rate)
    t = np.arange(count, dtype=np.float32) / sample_rate
    rng = np.random.default_rng(17)

    drone = 0.13 * np.sin(2 * math.pi * 48.0 * t)
    drone += 0.08 * np.sin(2 * math.pi * 72.0 * t + 0.4)
    ocean = 0.028 * rng.normal(0, 1, count).astype(np.float32)
    ocean = np.convolve(ocean, np.ones(180, dtype=np.float32) / 180, mode="same")
    shimmer = 0.025 * np.sin(2 * math.pi * 360.0 * t) * (0.5 + 0.5 * np.sin(2 * math.pi * 0.08 * t))
    bed = drone + ocean + shimmer

    for index, (_, start, _) in enumerate(SHOT_PLAN[1:], start=1):
        i0 = int(start * sample_rate)
        length = min(int(0.65 * sample_rate), count - i0)
        if length <= 0:
            continue
        tt = np.arange(length, dtype=np.float32) / sample_rate
        hit = 0.13 * np.sin(2 * math.pi * (92 + index * 8) * tt) * np.exp(-tt * 5.2)
        whoosh = 0.035 * rng.normal(0, 1, length).astype(np.float32) * np.exp(-tt * 3.0)
        bed[i0:i0 + length] += hit + whoosh

    # Water-cycle droplets.
    for shot_name, start, end in SHOT_PLAN:
        if shot_name == "water_cycle":
            for pulse_t in np.arange(start + 0.3, end, 0.55):
                i0 = int(pulse_t * sample_rate)
                length = min(int(0.12 * sample_rate), count - i0)
                if length <= 0:
                    continue
                tt = np.arange(length, dtype=np.float32) / sample_rate
                pulse = 0.055 * np.sin(2 * math.pi * 520 * tt) * fade_envelope(length, 0.04, 0.80)
                bed[i0:i0 + length] += pulse
            break

    finale_start = int(max(0, duration_s - 7.0) * sample_rate)
    tt = np.arange(count - finale_start, dtype=np.float32) / sample_rate
    bed[finale_start:] += 0.045 * np.sin(2 * math.pi * 540.0 * tt) * np.exp(-tt * 0.35)

    bed /= max(float(np.max(np.abs(bed))), 1e-6)
    bed *= 0.72
    left = bed * (0.97 + 0.03 * np.sin(2 * math.pi * 0.025 * t))
    right = bed * (0.97 + 0.03 * np.cos(2 * math.pi * 0.025 * t + 0.7))
    stereo = np.stack([left, right], axis=1)
    pcm = np.int16(np.clip(stereo, -1, 1) * 32767)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


# -----------------------------------------------------------------------------
# Render / package
# -----------------------------------------------------------------------------


def render_video(scene: EarthSystemScene, output_path: Path):
    with iio.get_writer(
        str(output_path),
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    ) as writer:
        for frame_index in range(FRAME_COUNT):
            writer.append_data(scene.frame(frame_index / FPS))


def mux_audio(video_path: Path, audio_path: Path, final_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copy2(video_path, final_path)
        return False
    command = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(final_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        shutil.copy2(video_path, final_path)
        return False


def save_contact_sheet(scene: EarthSystemScene, path: Path):
    times = [
        (a + b) / 2 for _, a, b in SHOT_PLAN
    ]
    frames = [Image.fromarray(scene.frame(t)).resize((270, 480), Image.Resampling.LANCZOS) for t in times]
    sheet = Image.new("RGB", (270 * 4, 480 * 2), (0, 0, 0))
    for i, frame in enumerate(frames[:8]):
        sheet.paste(frame, ((i % 4) * 270, (i // 4) * 480))
    sheet.save(path, quality=92)


def main():
    scene = EarthSystemScene()
    silent_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_silent.mp4"
    audio_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_soundtrack.wav"
    final_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}.mp4"
    subtitles = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    summary_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_summary.json"
    contact_sheet = PREVIEW_DIR / f"{CONFIG['output_basename']}_contact_sheet.jpg"

    print(f"Rendering {FRAME_COUNT} frames at {OUT_W}x{OUT_H} ...")
    render_video(scene, silent_video)
    print("Generating original stereo soundtrack ...")
    make_audio(audio_path, DURATION)
    print("Muxing audio ...")
    audio_muxed = mux_audio(silent_video, audio_path, final_video)
    write_srt(CAPTIONS, subtitles)
    save_contact_sheet(scene, contact_sheet)

    summary = {
        "title": CONFIG["title"],
        "subtitle": CONFIG["subtitle"],
        "duration_seconds": DURATION,
        "resolution": [OUT_W, OUT_H],
        "fps": FPS,
        "audio_muxed": audio_muxed,
        "facts_used": {
            "ocean_surface_share_percent": 71,
            "ocean_excess_heat_share_approx_percent": 90,
            "weather": "local atmospheric conditions on short timescales",
            "climate": "long-term expected patterns",
        },
        "visualization_note": "All visuals are procedural cinematic illustrations, not satellite imagery or calibrated climate maps.",
        "sources": [
            "https://science.nasa.gov/climate-change/faq/whats-the-difference-between-weather-and-climate/",
            "https://oceanservice.noaa.gov/news/june17/30days.html",
            "https://science.nasa.gov/earth/explore/the-ocean-and-climate-change/",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Final video : {final_video.resolve()}")
    print(f"Subtitles   : {subtitles.resolve()}")
    print(f"Contact sheet: {contact_sheet.resolve()}")
    print(f"Summary     : {summary_path.resolve()}")


if __name__ == "__main__":
    main()
