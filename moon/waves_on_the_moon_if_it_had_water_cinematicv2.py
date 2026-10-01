from __future__ import annotations

"""
Waves on the Moon — If It Had Water — cinematic YouTube Short renderer (enhanced)

A more cinematic, more graphical version of the Moon-wave short. The physics
framing stays narrow and grounded: if liquid water existed on the Moon, could
waves form under real lunar gravity and the Moon's near-vacuum environment?

Key framing
-----------
- Liquid water is assumed to exist, but an Earth-like atmosphere is NOT added.
- The Moon still has gravity (~1.62 m/s^2), so gravity waves are physically possible.
- The Moon has only a tenuous exosphere, so ordinary wind-driven ocean swell would be
  essentially absent.
- Waves could still be launched by disturbances such as meteoroid impacts,
  seafloor displacement, or mass movement.
- For deep-water gravity waves at the same wavelength, speed scales as sqrt(g),
  so the lunar speed is ~0.41 of the Earth value and the period is ~2.46x longer.

Visual style
------------
Cinematic and eye-catching, with stronger graphics, camera moves, glow, richer
water shading, and more simulation-like wave motion than the earlier version.
Still diagrammatic; not a full CFD fluid simulation.

Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    MOON_WAVES_SHORT_QUICK=1 python waves_on_the_moon_if_it_had_water_cinematic.py

Full render
-----------
    python waves_on_the_moon_if_it_had_water_cinematic.py

4K vertical
-----------
    MOON_WAVES_SHORT_4K=1 python waves_on_the_moon_if_it_had_water_cinematic.py
"""

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("MOON_WAVES_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("MOON_WAVES_SHORT_4K", "0") == "1" and not QUICK_MODE

OUT_W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
OUT_H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 8 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SCALE = OUT_W / 1080.0
OUT_SIZE = (OUT_W, OUT_H)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "waves_on_the_moon_if_it_had_water_cinematic_v2_output"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "WAVES ON THE MOON — IF IT HAD WATER",
    "subtitle": "more cinematic // lunar gravity // no wind-generated swell",
    "output_basename": "waves_on_the_moon_if_it_had_water_cinematic_v2",
    "contrast": 1.10,
    "saturation": 1.12,
    "vignette": 0.31,
}

COLORS = {
    "space": (3, 8, 20),
    "space2": (8, 18, 42),
    "white": (244, 248, 255),
    "muted": (165, 190, 220),
    "cyan": (71, 228, 255),
    "blue": (56, 141, 255),
    "deep_blue": (3, 42, 106),
    "water": (13, 90, 176),
    "water2": (48, 166, 220),
    "water3": (8, 58, 132),
    "foam": (227, 247, 255),
    "moon": (165, 166, 171),
    "moon_dark": (71, 73, 80),
    "rock": (104, 100, 98),
    "rock2": (154, 149, 141),
    "rock3": (72, 70, 76),
    "gold": (255, 203, 86),
    "orange": (255, 141, 69),
    "red": (255, 90, 105),
    "violet": (185, 128, 255),
    "green": (109, 242, 176),
    "earth_ocean": (45, 116, 201),
    "earth_land": (82, 156, 102),
}

G_EARTH = 9.81
G_MOON = 1.62
SPEED_RATIO = math.sqrt(G_MOON / G_EARTH)
PERIOD_RATIO = math.sqrt(G_EARTH / G_MOON)

FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.35, 7.4, "If liquid water lay on the Moon, could it make waves? Yes. Lunar gravity can still pull a disturbed surface back toward level, so gravity waves are possible."),
    (7.5, 16.9, "But the Moon has almost no atmosphere. That means no ordinary wind-driven ocean swell — the endless wave field we usually associate with Earthly seas."),
    (17.0, 27.0, "A meteoroid impact would be a different story. It would shove water aside, launch circular waves, and send energy racing across the basin."),
    (27.1, 37.8, "A moonquake or sudden movement of the basin floor could also displace water and produce broad, long-period waves — more like a tsunami-style pulse than choppy surf."),
    (37.9, 48.1, "Because lunar gravity is only 1.62 meters per second squared, waves of the same wavelength would travel at about forty-one percent of the Earth speed and oscillate more slowly."),
    (48.2, 57.4, "And if one of those waves reached shallow water, it could still steepen and break on a lunar shore. So yes: waves are possible — just not normal wind-made surf."),
]

if QUICK_MODE:
    factor = DURATION / 58.0
    CAPTIONS = [(a * factor, b * factor, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "hook", "start": 0.0, "end": 7.8 if not QUICK_MODE else 1.75},
    {"name": "no_wind", "start": 7.8 if not QUICK_MODE else 1.75, "end": 17.2 if not QUICK_MODE else 3.95},
    {"name": "impact", "start": 17.2 if not QUICK_MODE else 3.95, "end": 27.5 if not QUICK_MODE else 6.2},
    {"name": "moonquake", "start": 27.5 if not QUICK_MODE else 6.2, "end": 38.5 if not QUICK_MODE else 8.7},
    {"name": "gravity", "start": 38.5 if not QUICK_MODE else 8.7, "end": 49.0 if not QUICK_MODE else 11.0},
    {"name": "shore", "start": 49.0 if not QUICK_MODE else 11.0, "end": DURATION},
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(value: float) -> float:
    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


def ease_out_cubic(x: float) -> float:
    x = clamp(x)
    return 1.0 - (1.0 - x) ** 3


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def caption_at(t: float) -> Optional[str]:
    """Return a short, readable page of the active caption.

    Long narration is split across its original time window so only about
    15-20 words are visible at once; the full text remains in the SRT file.
    """
    for start, end, text in CAPTIONS:
        if start <= t < end:
            words = str(text).split()
            if not words:
                return None
            target_words = 18
            page_count = max(1, (len(words) + target_words - 1) // target_words)
            base = len(words) // page_count
            extra = len(words) % page_count
            progress = (t - start) / max(end - start, 1e-9)
            page = min(page_count - 1, max(0, int(progress * page_count)))
            first = page * base + min(page, extra)
            count = base + (1 if page < extra else 0)
            return " ".join(words[first:first + count])
    return None


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(7, int(size * SCALE)))
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(image: Image.Image, value: str, xy: Tuple[int, int], size: int,
              fill=(255, 255, 255, 255), bold: bool = False,
              anchor: str = "la", stroke: int = 2):
    ImageDraw.Draw(image).text(
        xy,
        value,
        font=get_font(size, bold),
        fill=fill,
        anchor=anchor,
        stroke_width=max(1, int(stroke * SCALE)),
        stroke_fill=(0, 0, 0, 220),
    )


def draw_wrapped_text(image: Image.Image, value: str, xy: Tuple[int, int],
                      max_width: int, size: int, fill=(255, 255, 255, 245),
                      bold: bool = False, spacing: int = 6):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold)
    words = value.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        box = draw.textbbox((0, 0), candidate, font=font, stroke_width=max(1, int(2 * SCALE)))
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
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
            stroke_width=max(1, int(2 * SCALE)),
            stroke_fill=(0, 0, 0, 220),
        )
        box = draw.textbbox((x, y), line, font=font, stroke_width=max(1, int(2 * SCALE)))
        y += (box[3] - box[1]) + int(spacing * SCALE)


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    rr = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * rr ** 1.85, 0.0, 1.0).astype(np.float32)


def apply_grade(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array)
    image = ImageEnhance.Contrast(image).enhance(float(CONFIG["contrast"]))
    image = ImageEnhance.Color(image).enhance(float(CONFIG["saturation"]))
    return np.asarray(image)


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for i, (start, end, value) in enumerate(captions, start=1):
        lines.extend([str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", value, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


@dataclass(frozen=True)
class Star:
    x: float
    y: float
    r: float
    alpha: int
    phase: float


@dataclass(frozen=True)
class Spray:
    angle: float
    speed: float
    life: float
    size: float


class MoonWaveScene:
    def __init__(self):
        rng = np.random.default_rng(20260818)
        self.stars = [
            Star(
                x=float(rng.uniform(0, OUT_W)),
                y=float(rng.uniform(0, OUT_H * 0.58)),
                r=float(rng.uniform(0.45, 1.9) * SCALE),
                alpha=int(rng.uniform(35, 140)),
                phase=float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(180 if QUICK_MODE else 480)
        ]
        self.spray = [
            Spray(
                angle=float(rng.uniform(-1.15, -0.25)),
                speed=float(rng.uniform(0.45, 1.2)),
                life=float(rng.uniform(0.55, 1.0)),
                size=float(rng.uniform(2.0, 8.5) * SCALE),
            )
            for _ in range(100 if QUICK_MODE else 260)
        ]

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 138):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            box,
            radius=max(8, int(26 * SCALE)),
            fill=(2, 10, 24, alpha),
            outline=COLORS["cyan"] + (36,),
            width=max(1, int(2 * SCALE)),
        )
        image.alpha_composite(overlay)

    def background(self, t: float, horizon_shift: float = 0.0) -> Image.Image:
        arr = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        yy = np.linspace(0, 1, OUT_H)[:, None]
        arr[..., 0] = np.clip(3 + 22 * yy, 0, 255)
        arr[..., 1] = np.clip(9 + 42 * yy, 0, 255)
        arr[..., 2] = np.clip(24 + 82 * yy, 0, 255)
        image = Image.fromarray(arr, "RGB").convert("RGBA")
        draw = ImageDraw.Draw(image)
        for item in self.stars:
            a = int(item.alpha * (0.8 + 0.2 * math.sin(t * 1.15 + item.phase)))
            draw.ellipse((item.x - item.r, item.y - item.r, item.x + item.r, item.y + item.r), fill=COLORS["white"] + (a,))
        # soft blue horizon haze
        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hz = ImageDraw.Draw(haze)
        hy = int(OUT_H * (0.60 + horizon_shift))
        hz.rectangle((0, hy - int(80 * SCALE), OUT_W, hy + int(140 * SCALE)), fill=COLORS["blue"] + (22,))
        image.alpha_composite(haze.filter(ImageFilter.GaussianBlur(max(6, int(45 * SCALE)))))
        return image

    def draw_earth(self, image: Image.Image, t: float, x_frac: float = 0.84, y_frac: float = 0.18, radius_frac: float = 0.11):
        cx = int(OUT_W * x_frac)
        cy = int(OUT_H * y_frac)
        r = int(OUT_W * radius_frac)
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for extra, a in [(0.34, 16), (0.20, 28), (0.08, 38)]:
            rr = int(r * (1.0 + extra))
            gd.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=COLORS["cyan"] + (a,))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4, int(20 * SCALE)))))

        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=COLORS["earth_ocean"] + (255,))
        continents = [
            [(0.12, -0.72), (0.42, -0.62), (0.48, -0.20), (0.21, -0.06), (-0.02, -0.18), (0.00, -0.52)],
            [(-0.42, -0.18), (-0.22, -0.34), (0.00, -0.10), (-0.02, 0.18), (-0.26, 0.32), (-0.42, 0.12)],
            [(0.20, 0.18), (0.42, 0.26), (0.50, 0.54), (0.22, 0.68), (0.06, 0.48)],
        ]
        rot = 0.16 * math.sin(t * 0.18)
        for poly in continents:
            pts = []
            for px, py in poly:
                x = px * math.cos(rot) - py * math.sin(rot)
                y = px * math.sin(rot) + py * math.cos(rot)
                pts.append((cx + x * r, cy + y * r))
            d.polygon(pts, fill=COLORS["earth_land"] + (240,))
        d.pieslice((cx-r, cy-r, cx+r, cy+r), 88, 272, fill=(0, 0, 0, 110))
        image.alpha_composite(overlay)

    def draw_hud(self, image: Image.Image, t: float):
        draw_text(image, "PHYSICS VISUALIZATION // LIQUID WATER ASSUMED", (OUT_W - int(44 * SCALE), int(58 * SCALE)), 12 if not QUICK_MODE else 6, COLORS["gold"] + (224,), True, "ra", 1)
        draw_text(image, "MOON'S ACTUAL GRAVITY // NEAR-VACUUM CONDITIONS", (OUT_W - int(44 * SCALE), int(84 * SCALE)), 10 if not QUICK_MODE else 5, COLORS["muted"] + (188,), False, "ra", 1)

    def title(self, image: Image.Image, t: float):
        if t < (5.7 if not QUICK_MODE else 1.3):
            fade = smoothstep(t / (0.85 if not QUICK_MODE else 0.18))
            draw_text(image, "WAVES", (OUT_W // 2, int(OUT_H * 0.068)), 34 if not QUICK_MODE else 17, COLORS["white"] + (int(240 * fade),), True, "ma", 2)
            draw_text(image, "ON THE MOON", (OUT_W // 2, int(OUT_H * 0.108)), 42 if not QUICK_MODE else 21, COLORS["cyan"] + (int(246 * fade),), True, "ma", 2)
            draw_text(image, "IF IT HAD WATER", (OUT_W // 2, int(OUT_H * 0.145)), 26 if not QUICK_MODE else 13, COLORS["gold"] + (int(235 * fade),), True, "ma", 2)

    def caption(self, image: Image.Image, t: float):
        cap = caption_at(t)
        if not cap:
            return
        y0 = OUT_H - int(188 * SCALE)
        self.panel(image, (int(58 * SCALE), y0, OUT_W - int(58 * SCALE), y0 + int(116 * SCALE)), 118)
        draw_wrapped_text(image, cap, (int(82 * SCALE), y0 + int(17 * SCALE)), OUT_W - int(164 * SCALE), 30)

    def label(self, image: Image.Image, textv: str):
        draw_text(image, textv, (int(48 * SCALE), int(57 * SCALE)), 14 if not QUICK_MODE else 7, COLORS["muted"] + (198,), True, "la", 1)

    def terrain_profile(self, x_values: np.ndarray, t: float, base: float = 0.72) -> np.ndarray:
        u = x_values / max(OUT_W - 1, 1)
        y = (
            base
            + 0.020 * np.sin(2.5 * np.pi * u + 0.4)
            + 0.026 * np.sin(6.0 * np.pi * u + 1.2)
            + 0.010 * np.sin(12.0 * np.pi * u + 0.3)
        )
        # central bay dip for water body
        y -= 0.055 * np.exp(-((u - 0.52) / 0.18) ** 2)
        return y * OUT_H

    def wave_line(self, x_values: np.ndarray, t: float, y_base: float, calm: float = 0.6,
                  impact_center: Optional[float] = None, impact_strength: float = 0.0,
                  quake_strength: float = 0.0, shoal: float = 0.0) -> np.ndarray:
        x = x_values.astype(np.float32)
        y = np.full_like(x, y_base, dtype=np.float32)
        x_norm = x / max(OUT_W - 1, 1)
        # base long waves (subdued calm sea)
        y += (10 + 4 * calm) * SCALE * np.sin(2 * np.pi * (x_norm * 1.3 - t * (0.030 + 0.005 * calm)) + 0.2)
        y += (4 + 2.5 * calm) * SCALE * np.sin(2 * np.pi * (x_norm * 3.2 - t * 0.056) + 1.1)
        y += 2.6 * SCALE * np.sin(2 * np.pi * (x_norm * 6.8 - t * 0.090) + 0.4)

        if impact_center is not None and impact_strength > 0.0:
            dist = np.abs(x - impact_center)
            envelope = np.exp(-(dist / (OUT_W * 0.22)) ** 1.55)
            ring = np.sin((dist / (OUT_W * 0.028)) - t * 0.64)
            y += impact_strength * envelope * ring * 22 * SCALE

        if quake_strength > 0.0:
            # broad long-wave pulse from left to right, tsunami-like
            front = (x_norm - (0.18 + 0.06 * t))
            pulse = np.exp(-(front / 0.10) ** 2) - 0.5 * np.exp(-((front - 0.12) / 0.13) ** 2)
            y += quake_strength * pulse * 40 * SCALE

        if shoal > 0.0:
            shore_gain = 1.0 + 2.7 * shoal * np.clip((x_norm - 0.67) / 0.33, 0, 1) ** 1.7
            y = y_base + (y - y_base) * shore_gain
            y -= shoal * np.clip((x_norm - 0.78) / 0.22, 0, 1) ** 1.5 * 18 * SCALE

        return y

    def draw_water_scene(self, image: Image.Image, t: float, water_level: float = 0.67, calm: float = 0.7,
                         impact_center: Optional[float] = None, impact_strength: float = 0.0,
                         quake_strength: float = 0.0, shoal: float = 0.0,
                         with_terrain: bool = True, shoreline_right: bool = False):
        x_values = np.linspace(0, OUT_W - 1, 240)
        terrain = self.terrain_profile(x_values, t, base=0.69 if shoreline_right else 0.72)
        surface = self.wave_line(
            x_values, t, OUT_H * water_level, calm=calm,
            impact_center=impact_center, impact_strength=impact_strength,
            quake_strength=quake_strength, shoal=shoal,
        )

        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)

        # terrain silhouette / shore
        if with_terrain:
            terrain_pts = [(float(x_values[0]), OUT_H)]
            terrain_pts += [(float(x), float(y)) for x, y in zip(x_values, terrain)]
            terrain_pts += [(float(x_values[-1]), OUT_H)]
            d.polygon(terrain_pts, fill=COLORS["rock3"] + (255,))
            # highlight ridge
            d.line([(float(x), float(y)) for x, y in zip(x_values, terrain)], fill=COLORS["rock2"] + (70,), width=max(1, int(3 * SCALE)))

        # luminous sub-surface band
        water_poly = [(0.0, OUT_H)] + [(float(x), float(y)) for x, y in zip(x_values, surface)] + [(float(OUT_W), OUT_H)]
        d.polygon(water_poly, fill=COLORS["water"] + (232,))
        # deeper band
        d.polygon([(0.0, OUT_H)] + [(float(x), float(min(OUT_H, y + 120 * SCALE))) for x, y in zip(x_values, surface)] + [(float(OUT_W), OUT_H)], fill=COLORS["water3"] + (90,))
        image.alpha_composite(overlay)

        # water vertical gradient and soft light shafts
        arr = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
        yy = np.linspace(0, 1, OUT_H)[:, None]
        arr[..., 0] = np.clip(3 + 25 * yy, 0, 255)
        arr[..., 1] = np.clip(55 + 110 * yy, 0, 255)
        arr[..., 2] = np.clip(120 + 80 * yy, 0, 255)
        arr[..., 3] = np.clip((yy ** 0.9) * 115, 0, 255)
        gradient = Image.fromarray(arr, "RGBA")
        mask = Image.new("L", OUT_SIZE, 0)
        ImageDraw.Draw(mask).polygon(water_poly, fill=255)
        gradient.putalpha(mask)
        image.alpha_composite(gradient)

        # bright crest line
        line_layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        ld = ImageDraw.Draw(line_layer)
        pts = [(float(x), float(y)) for x, y in zip(x_values, surface)]
        ld.line(pts, fill=COLORS["foam"] + (210,), width=max(2, int(3 * SCALE)))
        ld.line(pts, fill=COLORS["cyan"] + (120,), width=max(6, int(11 * SCALE)))

        # secondary wave traces for texture
        for offset, alpha, width in [(18, 80, 2), (38, 55, 2), (58, 30, 1)]:
            ld.line([(x, y + offset * SCALE) for x, y in pts], fill=COLORS["foam"] + (alpha,), width=max(1, int(width * SCALE)))
        image.alpha_composite(line_layer.filter(ImageFilter.GaussianBlur(max(1, int(3 * SCALE)))))
        image.alpha_composite(line_layer)

        # caustic style horizontal streaks in water
        caustics = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        cd = ImageDraw.Draw(caustics)
        for j in range(10 if QUICK_MODE else 18):
            frac = j / max((9 if QUICK_MODE else 17), 1)
            yy_line = lerp(surface.min() + 28 * SCALE, OUT_H - 50 * SCALE, frac)
            phase = t * 0.9 + j * 0.8
            pts2 = []
            for x in np.linspace(0, OUT_W - 1, 90):
                pts2.append((float(x), float(yy_line + 5 * SCALE * math.sin(x * 0.017 + phase))))
            cd.line(pts2, fill=COLORS["foam"] + (28 if j % 2 else 18,), width=max(1, int(2 * SCALE)))
        caustics.putalpha(mask)
        image.alpha_composite(caustics)

        # foam plume for breaking shore
        if shoal > 0.0 and shoreline_right:
            x0 = OUT_W * 0.84
            y0 = float(np.interp(x0, x_values, surface))
            crest = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            cd2 = ImageDraw.Draw(crest)
            arch = []
            for k in range(18):
                u = k / 17.0
                px = x0 + (u - 0.1) * 82 * SCALE
                py = y0 - 12 * SCALE - math.sin(u * math.pi) * (35 + 30 * shoal) * SCALE
                arch.append((px, py))
            cd2.line(arch, fill=COLORS["foam"] + (235,), width=max(2, int(7 * SCALE)))
            for px, py in arch[4:14]:
                cd2.ellipse((px - 4 * SCALE, py - 4 * SCALE, px + 4 * SCALE, py + 4 * SCALE), fill=COLORS["foam"] + (180,))
            image.alpha_composite(crest.filter(ImageFilter.GaussianBlur(max(1, int(2 * SCALE)))))
            image.alpha_composite(crest)

    def draw_impact_event(self, image: Image.Image, t_local: float):
        # bright impact flash and spray above water
        progress = clamp(t_local)
        x0 = OUT_W * 0.52
        water_y = OUT_H * 0.66
        flash = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        fd = ImageDraw.Draw(flash)
        meteor_x = lerp(OUT_W * 0.30, x0, progress)
        meteor_y = lerp(OUT_H * 0.22, water_y - 18 * SCALE, progress)
        fd.line((meteor_x - 120 * SCALE, meteor_y - 90 * SCALE, meteor_x, meteor_y), fill=COLORS["orange"] + (190,), width=max(2, int(4 * SCALE)))
        fd.line((meteor_x - 65 * SCALE, meteor_y - 50 * SCALE, meteor_x, meteor_y), fill=COLORS["gold"] + (220,), width=max(2, int(8 * SCALE)))
        fd.ellipse((meteor_x - 10 * SCALE, meteor_y - 10 * SCALE, meteor_x + 10 * SCALE, meteor_y + 10 * SCALE), fill=COLORS["white"] + (255,))
        if progress > 0.38:
            p2 = smoothstep((progress - 0.38) / 0.62)
            burst_r = (18 + 80 * p2) * SCALE
            fd.ellipse((x0 - burst_r, water_y - burst_r, x0 + burst_r, water_y + burst_r), outline=COLORS["gold"] + (int(210 * (1 - 0.35 * p2)),), width=max(1, int(6 * SCALE)))
            fd.ellipse((x0 - burst_r * 0.55, water_y - burst_r * 0.55, x0 + burst_r * 0.55, water_y + burst_r * 0.55), fill=COLORS["white"] + (int(95 * (1 - p2 * 0.5)),))
            for particle in self.spray:
                u = p2 / max(particle.life, 1e-6)
                if u <= 1.0:
                    px = x0 + math.cos(particle.angle) * particle.speed * u * 150 * SCALE
                    py = water_y + math.sin(particle.angle) * particle.speed * u * 180 * SCALE + 120 * SCALE * u * u
                    rr = particle.size * (1.0 - 0.55 * u)
                    fd.ellipse((px - rr, py - rr, px + rr, py + rr), fill=COLORS["foam"] + (130,))
        image.alpha_composite(flash.filter(ImageFilter.GaussianBlur(max(2, int(5 * SCALE)))))
        image.alpha_composite(flash)

    def draw_no_wind_graphic(self, image: Image.Image, t: float):
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        cy = int(OUT_H * 0.34)
        x0 = int(OUT_W * 0.16)
        for k in range(4):
            xx = x0 + int(k * 105 * SCALE)
            d.line((xx, cy, xx + 58 * SCALE, cy), fill=COLORS["muted"] + (120,), width=max(1, int(3 * SCALE)))
            d.polygon([(xx + 58 * SCALE, cy), (xx + 45 * SCALE, cy - 7 * SCALE), (xx + 45 * SCALE, cy + 7 * SCALE)], fill=COLORS["muted"] + (120,))
        d.line((int(OUT_W * 0.28), int(OUT_H * 0.26), int(OUT_W * 0.70), int(OUT_H * 0.45)), fill=COLORS["red"] + (235,), width=max(2, int(5 * SCALE)))
        d.line((int(OUT_W * 0.70), int(OUT_H * 0.26), int(OUT_W * 0.28), int(OUT_H * 0.45)), fill=COLORS["red"] + (235,), width=max(2, int(5 * SCALE)))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1, int(1 * SCALE)))))
        image.alpha_composite(layer)

    def draw_moonquake_cutaway(self, image: Image.Image, t: float):
        # lower cutaway panel showing uplift beneath water
        box = (int(OUT_W * 0.06), int(OUT_H * 0.58), int(OUT_W * 0.94), int(OUT_H * 0.84))
        self.panel(image, box, 150)
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        x0, y0, x1, y1 = box
        mid = y0 + int((y1 - y0) * 0.46)
        d.rectangle((x0 + 18 * SCALE, y0 + 18 * SCALE, x1 - 18 * SCALE, mid), fill=COLORS["water2"] + (90,))
        floor_pts = [
            (x0 + 20 * SCALE, y1 - 28 * SCALE),
            (x0 + 160 * SCALE, y1 - 90 * SCALE),
            (x0 + 320 * SCALE, y1 - 58 * SCALE),
            (x0 + 500 * SCALE, y1 - 128 * SCALE),
            (x1 - 20 * SCALE, y1 - 48 * SCALE),
            (x1 - 20 * SCALE, y1 - 18 * SCALE),
            (x0 + 20 * SCALE, y1 - 18 * SCALE),
        ]
        d.polygon(floor_pts, fill=COLORS["rock"] + (220,))
        uplift = 16 * SCALE * math.sin(t * 3.2)
        fault_x = x0 + 505 * SCALE
        d.line((fault_x, y1 - 20 * SCALE, fault_x - 70 * SCALE, y1 - 120 * SCALE), fill=COLORS["gold"] + (215,), width=max(2, int(5 * SCALE)))
        d.polygon([
            (fault_x - 120 * SCALE, mid + 8 * SCALE),
            (fault_x - 20 * SCALE, mid + 8 * SCALE),
            (fault_x - 20 * SCALE, mid - 35 * SCALE - uplift),
            (fault_x - 120 * SCALE, mid - 12 * SCALE),
        ], fill=COLORS["water2"] + (130,))
        d.line((x0 + 60 * SCALE, mid + 10 * SCALE, x1 - 60 * SCALE, mid + 10 * SCALE), fill=COLORS["foam"] + (205,), width=max(2, int(3 * SCALE)))
        draw_text(layer, "SEAFLOOR DISPLACEMENT", (int((x0 + x1) / 2), y0 + int(32 * SCALE)), 14 if not QUICK_MODE else 7, COLORS["gold"] + (228,), True, "ma", 1)
        draw_text(layer, "long-wave source", (int((x0 + x1) / 2), y0 + int(56 * SCALE)), 11 if not QUICK_MODE else 6, COLORS["muted"] + (205,), False, "ma", 1)
        image.alpha_composite(layer)

    def draw_gravity_compare(self, image: Image.Image, t: float):
        # two comparison panels with animated wave trains
        top_box = (int(OUT_W * 0.10), int(OUT_H * 0.22), int(OUT_W * 0.90), int(OUT_H * 0.45))
        bot_box = (int(OUT_W * 0.10), int(OUT_H * 0.51), int(OUT_W * 0.90), int(OUT_H * 0.74))
        for box in (top_box, bot_box):
            self.panel(image, box, 155)

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for label, box, col, speed, period_mult in [
            ("EARTH  g ≈ 9.81 m/s²", top_box, COLORS["white"], 1.0, 1.0),
            ("MOON  g ≈ 1.62 m/s²", bot_box, COLORS["cyan"], SPEED_RATIO, PERIOD_RATIO),
        ]:
            x0, y0, x1, y1 = box
            draw_text(layer, label, (x0 + int(24 * SCALE), y0 + int(22 * SCALE)), 15 if not QUICK_MODE else 8, col + (230,), True, "la", 1)
            draw_text(layer, "same wavelength", (x1 - int(24 * SCALE), y0 + int(22 * SCALE)), 11 if not QUICK_MODE else 6, COLORS["muted"] + (190,), False, "ra", 1)
            baseline = y0 + int((y1 - y0) * 0.56)
            pts = []
            xs = np.linspace(x0 + 22 * SCALE, x1 - 22 * SCALE, 180)
            for xx in xs:
                xn = (xx - xs[0]) / max(xs[-1] - xs[0], 1)
                yy = baseline + math.sin(xn * 4.2 * math.pi - t * 2.6 * speed) * (18 * SCALE)
                pts.append((float(xx), float(yy)))
            d.line(pts, fill=col + (235,), width=max(2, int(3 * SCALE)))
            for px, py in pts[::12]:
                d.ellipse((px - 1.5 * SCALE, py - 1.5 * SCALE, px + 1.5 * SCALE, py + 1.5 * SCALE), fill=col + (200,))
            draw_text(layer, f"speed × {speed:.2f}" if speed < 1.0 else "speed × 1.00", (x0 + int(24 * SCALE), y1 - int(30 * SCALE)), 11 if not QUICK_MODE else 6, COLORS["gold"] + (224,), True, "la", 1)
            draw_text(layer, f"period × {period_mult:.2f}", (x1 - int(24 * SCALE), y1 - int(30 * SCALE)), 11 if not QUICK_MODE else 6, COLORS["muted"] + (208,), False, "ra", 1)
        image.alpha_composite(layer)

    # -----------------------------------------------------------------
    # Individual scenes
    # -----------------------------------------------------------------

    def hook(self, image: Image.Image, t: float):
        local = smoothstep((t - SHOT_PLAN[0]["start"]) / max(SHOT_PLAN[0]["end"] - SHOT_PLAN[0]["start"], 1e-9))
        self.draw_earth(image, t, x_frac=0.86, y_frac=0.18, radius_frac=0.11)
        self.draw_water_scene(image, t * 1.05, water_level=0.66, calm=0.75, with_terrain=True)
        self.panel(image, (int(OUT_W * 0.20), int(OUT_H * 0.63), int(OUT_W * 0.80), int(OUT_H * 0.72)), 132)
        draw_text(image, "COULD LUNAR WATER MAKE WAVES?", (OUT_W // 2, int(OUT_H * 0.652)), 20 if not QUICK_MODE else 10, COLORS["white"] + (240,), True, "ma", 1)
        draw_text(image, "yes — gravity still pulls the surface back toward level", (OUT_W // 2, int(OUT_H * 0.688)), 13 if not QUICK_MODE else 7, COLORS["cyan"] + (224,), False, "ma", 1)
        draw_text(image, "gravity ≈ 1.62 m/s²", (OUT_W // 2, int(OUT_H * 0.716)), 14 if not QUICK_MODE else 7, COLORS["gold"] + (224,), True, "ma", 1)
        self.label(image, "1 // THE QUESTION")

    def no_wind(self, image: Image.Image, t: float):
        self.draw_earth(image, t, x_frac=0.84, y_frac=0.18, radius_frac=0.10)
        self.draw_water_scene(image, t * 0.8, water_level=0.68, calm=0.18, with_terrain=True)
        self.draw_no_wind_graphic(image, t)
        self.panel(image, (int(OUT_W * 0.20), int(OUT_H * 0.64), int(OUT_W * 0.80), int(OUT_H * 0.73)), 132)
        draw_text(image, "WIND WAVES: NO", (OUT_W // 2, int(OUT_H * 0.662)), 20 if not QUICK_MODE else 10, COLORS["red"] + (240,), True, "ma", 1)
        draw_text(image, "the Moon has only a tenuous exosphere", (OUT_W // 2, int(OUT_H * 0.697)), 13 if not QUICK_MODE else 7, COLORS["white"] + (216,), False, "ma", 1)
        draw_text(image, "little to no air → no ordinary swell", (OUT_W // 2, int(OUT_H * 0.721)), 13 if not QUICK_MODE else 7, COLORS["muted"] + (210,), False, "ma", 1)
        self.label(image, "2 // REMOVE THE WIND")

    def impact(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[2]
        local = clamp((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        impact_center = OUT_W * 0.52
        impact_strength = ease_out_cubic(local)
        self.draw_earth(image, t, x_frac=0.86, y_frac=0.18, radius_frac=0.10)
        self.draw_water_scene(image, t * 0.9, water_level=0.67, calm=0.18, impact_center=impact_center, impact_strength=impact_strength, with_terrain=True)
        self.draw_impact_event(image, local)
        self.panel(image, (int(OUT_W * 0.19), int(OUT_H * 0.63), int(OUT_W * 0.81), int(OUT_H * 0.73)), 132)
        draw_text(image, "IMPACT WAVES: YES", (OUT_W // 2, int(OUT_H * 0.652)), 20 if not QUICK_MODE else 10, COLORS["green"] + (240,), True, "ma", 1)
        draw_text(image, "a disturbance displaces the surface", (OUT_W // 2, int(OUT_H * 0.689)), 13 if not QUICK_MODE else 7, COLORS["white"] + (216,), False, "ma", 1)
        draw_text(image, "gravity drives circular waves outward", (OUT_W // 2, int(OUT_H * 0.714)), 14 if not QUICK_MODE else 7, COLORS["cyan"] + (224,), True, "ma", 1)
        self.label(image, "3 // IMPACT CREATES A WAVE")

    def moonquake(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[3]
        local = clamp((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_water_scene(image, t * 0.85, water_level=0.54, calm=0.10, quake_strength=0.9 * smoothstep(local), with_terrain=False)
        self.draw_moonquake_cutaway(image, t)
        self.panel(image, (int(OUT_W * 0.16), int(OUT_H * 0.17), int(OUT_W * 0.84), int(OUT_H * 0.26)), 128)
        draw_text(image, "MOONQUAKE / BASIN MOTION", (OUT_W // 2, int(OUT_H * 0.205)), 20 if not QUICK_MODE else 10, COLORS["white"] + (240,), True, "ma", 1)
        draw_text(image, "seafloor motion can launch long-period waves", (OUT_W // 2, int(OUT_H * 0.236)), 13 if not QUICK_MODE else 7, COLORS["cyan"] + (220,), False, "ma", 1)
        draw_text(image, "more like a tsunami-style pulse", (OUT_W // 2, int(OUT_H * 0.255)), 13 if not QUICK_MODE else 7, COLORS["gold"] + (220,), False, "ma", 1)
        self.label(image, "4 // A SECOND WAVE SOURCE")

    def gravity(self, image: Image.Image, t: float):
        self.draw_gravity_compare(image, t)
        self.panel(image, (int(OUT_W * 0.15), int(OUT_H * 0.82), int(OUT_W * 0.85), int(OUT_H * 0.89)), 130)
        draw_text(image, f"c_MOON ≈ {SPEED_RATIO:.2f} × c_EARTH", (OUT_W // 2, int(OUT_H * 0.842)), 18 if not QUICK_MODE else 9, COLORS["gold"] + (236,), True, "ma", 1)
        draw_text(image, f"period ≈ {PERIOD_RATIO:.2f} × longer", (OUT_W // 2, int(OUT_H * 0.872)), 13 if not QUICK_MODE else 7, COLORS["white"] + (216,), False, "ma", 1)
        self.label(image, "5 // LUNAR GRAVITY CHANGES THE MOTION")

    def shore(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[5]
        local = clamp((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_earth(image, t, x_frac=0.86, y_frac=0.16, radius_frac=0.10)
        self.draw_water_scene(image, t * 0.72, water_level=0.67, calm=0.28, shoal=smoothstep(local), with_terrain=True, shoreline_right=True)
        self.panel(image, (int(OUT_W * 0.20), int(OUT_H * 0.69), int(OUT_W * 0.80), int(OUT_H * 0.77)), 130)
        draw_text(image, "YES — WAVES COULD FORM", (OUT_W // 2, int(OUT_H * 0.716)), 20 if not QUICK_MODE else 10, COLORS["green"] + (240,), True, "ma", 1)
        draw_text(image, "but not ordinary wind-driven lunar surf", (OUT_W // 2, int(OUT_H * 0.748)), 13 if not QUICK_MODE else 7, COLORS["white"] + (216,), False, "ma", 1)
        self.label(image, "6 // A WAVE CAN STILL BREAK")

    def render_frame(self, t: float) -> np.ndarray:
        image = self.background(t)
        shot_name = get_shot(t)["name"]
        if shot_name == "hook":
            self.hook(image, t)
        elif shot_name == "no_wind":
            self.no_wind(image, t)
        elif shot_name == "impact":
            self.impact(image, t)
        elif shot_name == "moonquake":
            self.moonquake(image, t)
        elif shot_name == "gravity":
            self.gravity(image, t)
        else:
            self.shore(image, t)

        self.draw_hud(image, t)
        self.title(image, t)
        self.caption(image, t)

        arr = np.asarray(image.convert("RGB"))
        arr = apply_grade(arr)
        arr = np.clip(arr.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / (0.9 if not QUICK_MODE else 0.2))
        fade_out = 1.0 - smoothstep((t - (DURATION - (1.1 if not QUICK_MODE else 0.25))) / (1.0 if not QUICK_MODE else 0.2))
        return np.clip(arr.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def save_summary() -> Path:
    obj = {
        "title": CONFIG["title"],
        "format": f"{OUT_W}x{OUT_H} vertical MP4",
        "fps": FPS,
        "duration_s": DURATION,
        "quick_mode": QUICK_MODE,
        "four_k": FOUR_K,
        "facts": [
            "Moon gravity ≈ 1.62 m/s².",
            "Moon lacks an appreciable atmosphere; only a tenuous exosphere.",
            "Wind-driven ocean swell would be essentially absent.",
            f"For the same wavelength, lunar deep-water wave speed ≈ {SPEED_RATIO:.2f} of Earth.",
            f"For the same wavelength, lunar period ≈ {PERIOD_RATIO:.2f} times Earth's.",
            "Plausible wave sources include impacts and seafloor displacement.",
        ],
        "captions": [{"start": a, "end": b, "text": text} for a, b, text in CAPTIONS],
        "shots": SHOT_PLAN,
    }
    path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_summary.json"
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


def save_preview_frames(scene: MoonWaveScene) -> List[Path]:
    paths: List[Path] = []
    for i, shot in enumerate(SHOT_PLAN, start=1):
        t = (shot["start"] + shot["end"]) / 2.0
        frame = scene.render_frame(t)
        path = PREVIEW_DIR / f"scene_{i:02d}.png"
        Image.fromarray(frame).save(path)
        paths.append(path)
    return paths


def make_contact_sheet(image_paths: Sequence[Path], out_path: Path) -> Path:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    if not images:
        raise ValueError("No images for contact sheet")
    thumb_w, thumb_h = images[0].size
    cols = 2
    rows = math.ceil(len(images) / cols)
    margin = int(20 * SCALE)
    label_h = int(34 * SCALE)
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * margin, rows * (thumb_h + label_h) + (rows + 1) * margin), (5, 8, 15))
    draw = ImageDraw.Draw(sheet)
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + label_h + margin)
        sheet.paste(image, (x, y))
        draw_text(sheet, f"Scene {idx + 1}", (x + int(6 * SCALE), y + thumb_h + int(8 * SCALE)), 16 if not QUICK_MODE else 8, COLORS["white"] + (255,), False, "la", 1)
    sheet.save(out_path, quality=94)
    return out_path


def render_video(scene: MoonWaveScene, out_path: Path) -> Path:
    tmp_path = out_path.with_suffix(".tmp.mp4")
    total_frames = int(round(DURATION * FPS))
    with iio.get_writer(
        tmp_path,
        fps=FPS,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "18" if not QUICK_MODE else "22", "-movflags", "+faststart"],
    ) as writer:
        for i in tqdm(range(total_frames), desc="Rendering", unit="frame"):
            t = i / FPS
            writer.append_data(scene.render_frame(t))
    shutil.move(str(tmp_path), str(out_path))
    return out_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


# =============================================================================
# YouTube title / description sidecar
# =============================================================================

YOUTUBE_TITLE = 'What If the Moon Had Oceans and Waves? 🌕🌊 #rootjatin'
YOUTUBE_DESCRIPTION = 'A hypothetical cinematic thought experiment imagining waves and open water on the Moon. The real Moon does not have surface oceans like Earth, so the water, wave behavior, shoreline, atmosphere-like appearance, and cinematic scene are illustrative rather than a reconstruction of lunar conditions. The video is designed to visualize the idea, not claim that these oceans exist.'
YOUTUBE_HASHTAGS = '#rootjatin #Moon #WhatIf #Space #Astronomy #Ocean #Simulation #Science'

def write_youtube_metadata_txt() -> Path:
    cfg = globals().get("CONFIG") or globals().get("CFG") or {}
    basename = (
        cfg.get("basename")
        or cfg.get("output_basename")
        or cfg.get("file_stem")
        or Path(__file__).stem
    )
    root = globals().get("OUTPUT_ROOT") or globals().get("ROOT") or Path(".")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{basename}_title_description.txt"
    path.write_text(
        "TITLE\n" + YOUTUBE_TITLE +
        "\n\nDESCRIPTION\n" + YOUTUBE_DESCRIPTION +
        "\n\nHASHTAGS\n" + YOUTUBE_HASHTAGS + "\n",
        encoding="utf-8",
    )
    return path

def main():
    scene = MoonWaveScene()
    summary_path = save_summary()
    srt_path = write_srt(CAPTIONS, OUTPUT_ROOT / f"{CONFIG['output_basename']}_subtitles.srt")
    preview_paths = save_preview_frames(scene)
    contact_sheet_path = make_contact_sheet(preview_paths, OUTPUT_ROOT / f"{CONFIG['output_basename']}_contact_sheet.jpg")
    video_path = render_video(scene, OUTPUT_ROOT / f"{CONFIG['output_basename']}.mp4")
    print(f"Saved summary: {summary_path}")
    print(f"Saved subtitles: {srt_path}")
    print(f"Saved previews: {[str(p) for p in preview_paths]}")
    print(f"Saved contact sheet: {contact_sheet_path}")
    print(f"Saved video: {video_path}")
    metadata_txt = write_youtube_metadata_txt()
    print("Title/description TXT:", metadata_txt.resolve())


if __name__ == "__main__":
    main()
