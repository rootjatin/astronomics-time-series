
from __future__ import annotations

"""
How Fast Are Galaxies Moving Away From Us? — YouTube Shorts renderer

A vertical, no-Matplotlib astronomy animation explaining the
Hubble-Lemaître law:

    recession speed ≈ H0 × distance

For visual storytelling this script uses a rounded reference value of
70 km/s/Mpc. The exact present expansion rate is an active research topic,
with different methods giving somewhat different values.

The animation also explains:
- nearby galaxies can ignore the overall trend because local gravity dominates;
- cosmological redshift comes from expanding space;
- sufficiently distant galaxies can have recession rates greater than c
  without locally travelling through space faster than light.

Install:
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview:
    GALAXY_SPEED_SHORT_QUICK=1 python3 how_fast_are_galaxies_moving_away.py

Full render:
    python3 how_fast_are_galaxies_moving_away.py


"""

import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SCRIPT_BUILD = "GALAXY-RECESSION-NO-MATPLOTLIB-2026-07-25"
QUICK_MODE = os.environ.get("GALAXY_SPEED_SHORT_QUICK", "0") == "1"

OUTPUT_ROOT = Path("how_fast_are_galaxies_moving_away_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
DATA_DIR = OUTPUT_ROOT / "data"
for directory in (OUTPUT_ROOT, PREVIEW_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

YOUTUBE_TITLE = "How Fast Are Galaxies Moving Away From Us? 🌌 #Shorts"

YOUTUBE_DESCRIPTION = """The farther a galaxy is, the faster cosmic expansion carries it away.

Using a rounded Hubble constant of 70 kilometres per second per megaparsec:

• 1 megaparsec → about 70 km/s
• 10 megaparsecs → about 700 km/s
• 100 megaparsecs → about 7,000 km/s

This is the Hubble–Lemaître law. Nearby galaxies can break the pattern because
local gravity may be stronger than cosmic expansion. At enormous distances,
recession rates can even exceed the speed of light because space itself is
expanding—not because galaxies are locally flying through space faster than light.

The exact value of the Hubble constant is still being debated, so 70 km/s/Mpc
is used here as a simple visual reference.

Created with Python using NumPy, Pillow, ImageIO and FFmpeg.

#Space #Astronomy #Galaxies #Universe #Cosmology #Python #Science #Shorts
"""

CONFIG = {
    "video_width": 720 if QUICK_MODE else 1080,
    "video_height": 1280 if QUICK_MODE else 1920,
    "fps": 8 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "how_fast_are_galaxies_moving_away",
    "hubble_reference": 70.0,
    "background_stars": 150 if QUICK_MODE else 340,
    "galaxy_count": 70 if QUICK_MODE else 180,
    "contrast": 1.15,
    "saturation": 1.10,
    "brightness": 1.025,
    "sharpness": 1.30,
    "vignette": 0.17,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0
UI_SCALE = max(0.78, SCALE * 1.18)


def px(value: float) -> int:
    return max(1, int(round(value * SCALE)))


def ui(value: float, minimum: int = 12) -> int:
    return max(minimum, int(round(value * UI_SCALE)))


FULL_CAPTIONS = [
    (0.4, 7.2, "Most distant galaxies are receding because the space between galaxies is expanding."),
    (7.3, 17.8, "The farther away a galaxy is, the faster its recession usually appears."),
    (17.9, 29.0, "Using seventy kilometres per second per megaparsec, one megaparsec means about seventy kilometres per second."),
    (29.1, 40.0, "At one hundred megaparsecs, the simple Hubble-law estimate becomes about seven thousand kilometres per second."),
    (40.1, 50.0, "Nearby galaxies can move differently because local gravity may overpower the overall expansion."),
    (50.1, 57.3, "At extreme distances, expanding space can produce recession rates greater than light without breaking relativity."),
]

if QUICK_MODE:
    factor = CONFIG["duration_s"] / 58.0
    CAPTIONS = [(a * factor, b * factor, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 8.0 if not QUICK_MODE else 2.0},
    {"name": "law", "start": 8.0 if not QUICK_MODE else 2.0, "end": 24.0 if not QUICK_MODE else 5.1},
    {"name": "examples", "start": 24.0 if not QUICK_MODE else 5.1, "end": 39.5 if not QUICK_MODE else 8.3},
    {"name": "local", "start": 39.5 if not QUICK_MODE else 8.3, "end": 50.5 if not QUICK_MODE else 10.4},
    {"name": "outro", "start": 50.5 if not QUICK_MODE else 10.4, "end": CONFIG["duration_s"]},
]

DISTANCE_EXAMPLES = [
    {"distance_mpc": 1, "distance_label": "1 Mpc", "ly_label": "3.26 million light-years"},
    {"distance_mpc": 10, "distance_label": "10 Mpc", "ly_label": "32.6 million light-years"},
    {"distance_mpc": 50, "distance_label": "50 Mpc", "ly_label": "163 million light-years"},
    {"distance_mpc": 100, "distance_label": "100 Mpc", "ly_label": "326 million light-years"},
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(value: float) -> float:
    value = clamp(value)
    return value * value * (3.0 - 2.0 * value)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


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


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(captions: Iterable[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for index, (start, end, text) in enumerate(captions, 1):
        lines.extend([
            str(index),
            f"{format_srt_time(start)} --> {format_srt_time(end)}",
            text,
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf" if bold
        else "/usr/share/fonts/opentype/inter/InterDisplay-Regular.otf",
        "/usr/share/fonts/truetype/lato/Lato-Heavy.ttf" if bold
        else "/usr/share/fonts/truetype/lato/Lato-Medium.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(6, int(size)))
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
) -> None:
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold)
    alpha = fill[3] if len(fill) > 3 else 255
    effective_stroke = max(int(stroke), max(1, int(round(size * 0.045))))
    shadow = max(1, int(round(size * 0.055)))
    x, y = xy
    draw.text(
        (x + shadow, y + shadow),
        text,
        font=font,
        fill=(0, 0, 0, min(210, alpha)),
        anchor=anchor,
        stroke_width=effective_stroke + 1,
        stroke_fill=(0, 0, 0, min(230, alpha)),
    )
    draw.text(
        xy,
        text,
        font=font,
        fill=fill,
        anchor=anchor,
        stroke_width=effective_stroke,
        stroke_fill=(0, 0, 0, min(245, alpha)),
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
) -> None:
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold)
    stroke_width = max(2, int(round(size * 0.045)))
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bb = draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke_width)
        if bb[2] - bb[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    x, y = xy
    for line in lines:
        shadow = max(1, int(round(size * 0.05)))
        draw.text(
            (x + shadow, y + shadow),
            line,
            font=font,
            fill=(0, 0, 0, 205),
            stroke_width=stroke_width + 1,
            stroke_fill=(0, 0, 0, 225),
        )
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 245),
        )
        bb = draw.textbbox((x, y), line, font=font, stroke_width=stroke_width)
        y += bb[3] - bb[1] + max(line_spacing, int(size * 0.20))


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius ** 1.8, 0.0, 1.0).astype(np.float32)


def apply_grade(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array)
    image = ImageEnhance.Brightness(image).enhance(CONFIG["brightness"])
    image = ImageEnhance.Contrast(image).enhance(CONFIG["contrast"])
    image = ImageEnhance.Color(image).enhance(CONFIG["saturation"])
    image = ImageEnhance.Sharpness(image).enhance(CONFIG["sharpness"])
    image = image.filter(
        ImageFilter.UnsharpMask(radius=max(1, px(1.2)), percent=110, threshold=3)
    )
    return np.array(image)


VIGNETTE = make_vignette(OUT_W, OUT_H, CONFIG["vignette"])


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------

class GalaxyExpansionScene:
    def __init__(self) -> None:
        rng = np.random.default_rng(260726)
        self.space_stars = [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.45, 2.0) * max(0.8, SCALE)),
                "alpha": int(rng.integers(20, 115)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            }
            for _ in range(int(CONFIG["background_stars"]))
        ]

        # Galaxies distributed around a central observer in normalized coordinates.
        radii = np.sqrt(rng.uniform(0.08, 1.0, int(CONFIG["galaxy_count"])))
        angles = rng.uniform(0, 2 * math.pi, int(CONFIG["galaxy_count"]))
        self.galaxies = []
        for i, (r, a) in enumerate(zip(radii, angles)):
            self.galaxies.append({
                "r": float(r),
                "angle": float(a),
                "size": float(rng.uniform(7, 20)),
                "tilt": float(rng.uniform(0.25, 0.75)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "warm": bool(i % 5 == 0),
            })

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, (3, 7, 20, 255))
        draw = ImageDraw.Draw(image)
        for star in self.space_stars:
            pulse = 0.72 + 0.28 * math.sin(t * 1.6 + star["phase"])
            r = star["r"]
            draw.ellipse(
                (star["x"] - r, star["y"] - r, star["x"] + r, star["y"] + r),
                fill=(215, 232, 255, int(star["alpha"] * pulse)),
            )

        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for cx, cy, colour in [
            (OUT_W * 0.23, OUT_H * 0.26, (45, 31, 120)),
            (OUT_W * 0.78, OUT_H * 0.42, (15, 86, 126)),
            (OUT_W * 0.50, OUT_H * 0.77, (86, 34, 70)),
        ]:
            for radius, alpha in [(430 * SCALE, 13), (280 * SCALE, 21), (170 * SCALE, 29)]:
                hd.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    fill=colour + (alpha,),
                )
        haze = haze.filter(ImageFilter.GaussianBlur(max(14, int(64 * SCALE))))
        image.alpha_composite(haze)
        return image

    def draw_galaxy(
        self,
        image: Image.Image,
        center: Tuple[float, float],
        radius: float,
        tilt: float,
        angle: float,
        warm: bool = False,
        alpha: int = 240,
    ) -> None:
        size = max(3.0, radius)
        local_size = int(max(48, size * 7))
        layer = Image.new("RGBA", (local_size, local_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx = cy = local_size / 2.0

        colour = (255, 205, 145) if warm else (155, 205, 255)
        for mult, a in [(2.8, 12), (2.0, 23), (1.35, 52)]:
            rx = size * mult
            ry = rx * tilt
            draw.ellipse(
                (cx - rx, cy - ry, cx + rx, cy + ry),
                fill=colour + (min(alpha, a),),
            )

        # Spiral-like curved arms rendered as point streams.
        for arm in range(2):
            points = []
            for q in np.linspace(0.15, 1.0, 44):
                theta = arm * math.pi + q * 4.7
                rr = size * q
                x = cx + math.cos(theta) * rr
                y = cy + math.sin(theta) * rr * tilt
                points.append((x, y))
            draw.line(points, fill=colour + (min(alpha, 155),), width=max(1, int(size * 0.22)))

        core_r = max(1.5, size * 0.28)
        draw.ellipse(
            (cx - core_r, cy - core_r * tilt,
             cx + core_r, cy + core_r * tilt),
            fill=(255, 246, 218, min(alpha, 245)),
        )

        layer = layer.rotate(math.degrees(angle), resample=Image.Resampling.BICUBIC, expand=True)
        x = int(center[0] - layer.width / 2)
        y = int(center[1] - layer.height / 2)
        image.alpha_composite(layer, (x, y))

    def draw_observer(self, image: Image.Image, center: Tuple[float, float]) -> None:
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cx, cy = center
        for rr, alpha in [(42, 15), (28, 30), (17, 65)]:
            r = rr * SCALE
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(105, 232, 248, alpha))
        r = 9 * SCALE
        draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            fill=(255, 231, 125, 255),
            outline=(255, 255, 255, 235),
            width=max(1, px(2)),
        )
        image.alpha_composite(overlay)
        draw_text(image, "US", (int(cx), int(cy + px(31))), size=ui(17),
                  fill=(245, 249, 255, 230), bold=True, anchor="ma", stroke=2)

    def draw_expanding_field(
        self,
        image: Image.Image,
        t: float,
        expansion: float,
        show_arrows: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        cx, cy = OUT_W * 0.50, OUT_H * 0.42
        max_radius = min(OUT_W * 0.47, OUT_H * 0.31)
        galaxies = self.galaxies if limit is None else self.galaxies[:limit]

        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        ordered = sorted(galaxies, key=lambda g: g["r"], reverse=True)

        for galaxy in ordered:
            radius_norm = galaxy["r"] * expansion
            x = cx + math.cos(galaxy["angle"]) * radius_norm * max_radius
            y = cy + math.sin(galaxy["angle"]) * radius_norm * max_radius * 0.85
            size = galaxy["size"] * SCALE * (0.70 + galaxy["r"] * 0.75)

            self.draw_galaxy(
                image,
                (x, y),
                size,
                galaxy["tilt"],
                galaxy["phase"] + t * 0.04,
                galaxy["warm"],
                alpha=int(150 + 95 * galaxy["r"]),
            )

            if show_arrows and galaxy["r"] > 0.25:
                angle = galaxy["angle"]
                speed_factor = galaxy["r"]
                arrow_len = (20 + 70 * speed_factor) * SCALE
                sx = x + math.cos(angle) * size * 0.9
                sy = y + math.sin(angle) * size * 0.9
                ex = sx + math.cos(angle) * arrow_len
                ey = sy + math.sin(angle) * arrow_len
                draw.line(
                    (sx, sy, ex, ey),
                    fill=(255, 176, 90, int(85 + 145 * speed_factor)),
                    width=max(2, px(3.5)),
                )
                ah = px(10)
                draw.polygon(
                    [
                        (ex, ey),
                        (
                            ex - ah * math.cos(angle - 0.48),
                            ey - ah * math.sin(angle - 0.48),
                        ),
                        (
                            ex - ah * math.cos(angle + 0.48),
                            ey - ah * math.sin(angle + 0.48),
                        ),
                    ],
                    fill=(255, 176, 90, int(110 + 130 * speed_factor)),
                )

        image.alpha_composite(overlay)
        self.draw_observer(image, (cx, cy))

    def draw_intro(self, image: Image.Image, t: float) -> None:
        end = 8.0 if not QUICK_MODE else 2.0
        progress = smoothstep(t / max(end, 0.01))
        self.draw_expanding_field(
            image,
            t,
            expansion=lerp(0.40, 0.95, progress),
            show_arrows=True,
            limit=46 if QUICK_MODE else 90,
        )
        draw_text(
            image,
            "GALAXIES ARE",
            (OUT_W // 2, int(OUT_H * 0.66)),
            size=ui(58),
            fill=(245, 249, 255, 248),
            bold=True,
            anchor="ma",
            stroke=4,
        )
        draw_text(
            image,
            "MOVING AWAY",
            (OUT_W // 2, int(OUT_H * 0.72)),
            size=ui(70),
            fill=(110, 232, 248, 250),
            bold=True,
            anchor="ma",
            stroke=4,
        )
        draw_text(
            image,
            "but not all at the same speed",
            (OUT_W // 2, int(OUT_H * 0.78)),
            size=ui(27),
            fill=(255, 191, 103, 235),
            bold=True,
            anchor="ma",
            stroke=2,
        )

    def draw_hubble_graph(
        self,
        image: Image.Image,
        progress: float,
        highlight_distance: float,
    ) -> None:
        x0, y0 = px(95), int(OUT_H * 0.25)
        x1, y1 = OUT_W - px(72), int(OUT_H * 0.68)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        draw.rounded_rectangle(
            (x0 - px(34), y0 - px(42), x1 + px(24), y1 + px(50)),
            radius=max(12, px(24)),
            fill=(2, 8, 20, 226),
            outline=(95, 215, 242, 150),
            width=max(1, px(2)),
        )

        draw.line((x0, y1, x1, y1), fill=(215, 232, 245, 150), width=max(2, px(3)))
        draw.line((x0, y0, x0, y1), fill=(215, 232, 245, 150), width=max(2, px(3)))

        # Grid and tick labels.
        for value in (0, 25, 50, 75, 100):
            q = value / 100.0
            x = lerp(x0, x1, q)
            draw.line((x, y0, x, y1), fill=(130, 180, 210, 30), width=1)
            draw_text(
                overlay,
                str(value),
                (int(x), y1 + px(13)),
                size=ui(15),
                fill=(220, 232, 244, 210),
                anchor="ma",
                stroke=1,
            )

        for value in (0, 1750, 3500, 5250, 7000):
            q = value / 7000.0
            y = lerp(y1, y0, q)
            draw.line((x0, y, x1, y), fill=(130, 180, 210, 30), width=1)
            draw_text(
                overlay,
                f"{value:,}",
                (x0 - px(14), int(y)),
                size=ui(14),
                fill=(220, 232, 244, 205),
                anchor="rm",
                stroke=1,
            )

        line_end_x = lerp(x0, x1, smoothstep(progress))
        line_end_y = lerp(y1, y0, smoothstep(progress))
        draw.line(
            (x0, y1, line_end_x, line_end_y),
            fill=(110, 232, 248, 245),
            width=max(3, px(6)),
        )

        hd = clamp(highlight_distance / 100.0)
        hx = lerp(x0, x1, hd)
        hy = lerp(y1, y0, hd)
        if progress > hd * 0.75:
            for rr, alpha in [(16, 24), (10, 48)]:
                r = px(rr)
                draw.ellipse((hx - r, hy - r, hx + r, hy + r), fill=(255, 178, 92, alpha))
            r = px(6)
            draw.ellipse(
                (hx - r, hy - r, hx + r, hy + r),
                fill=(255, 185, 95, 255),
                outline=(255, 255, 255, 230),
                width=max(1, px(2)),
            )

        image.alpha_composite(overlay)

        draw_text(
            image,
            "DISTANCE (Mpc) →",
            (int((x0 + x1) / 2), y1 + px(48)),
            size=ui(18),
            fill=(235, 243, 252, 225),
            bold=True,
            anchor="ma",
            stroke=2,
        )
        draw_text(
            image,
            "RECESSION SPEED (km/s)",
            (x0 - px(48), y0 - px(22)),
            size=ui(16),
            fill=(235, 243, 252, 225),
            bold=True,
            stroke=2,
        )

    def draw_law(self, image: Image.Image, t: float) -> None:
        start = 8.0 if not QUICK_MODE else 2.0
        duration = 16.0 if not QUICK_MODE else 3.1
        frac = clamp((t - start) / max(duration, 0.01))
        highlight = lerp(1, 100, smoothstep(frac))
        self.draw_hubble_graph(image, progress=frac * 1.15, highlight_distance=highlight)

        draw_text(
            image,
            "HUBBLE–LEMAÎTRE LAW",
            (OUT_W // 2, px(86)),
            size=ui(31),
            fill=(110, 232, 248, 245),
            bold=True,
            anchor="ma",
            stroke=3,
        )
        draw_text(
            image,
            "v ≈ H₀ × d",
            (OUT_W // 2, int(OUT_H * 0.74)),
            size=ui(58),
            fill=(255, 193, 102, 248),
            bold=True,
            anchor="ma",
            stroke=4,
        )
        draw_text(
            image,
            "farther galaxy  =  faster recession",
            (OUT_W // 2, int(OUT_H * 0.80)),
            size=ui(25),
            fill=(245, 249, 255, 235),
            bold=True,
            anchor="ma",
            stroke=2,
        )

    def draw_speed_card(self, image: Image.Image, example: Dict, t: float) -> None:
        distance = float(example["distance_mpc"])
        speed = CONFIG["hubble_reference"] * distance

        cx, cy = OUT_W * 0.50, OUT_H * 0.39
        orbit = min(OUT_W * 0.36, px(355))
        angle = -0.25 + t * 0.10
        gx = cx + math.cos(angle) * orbit
        gy = cy + math.sin(angle) * orbit * 0.42

        self.draw_observer(image, (cx - orbit * 0.72, cy))
        self.draw_galaxy(
            image,
            (gx, gy),
            33 * SCALE,
            0.48,
            angle,
            warm=True,
            alpha=245,
        )

        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        start_x = cx - orbit * 0.60
        end_x = gx - px(45)
        y = cy
        draw.line(
            (start_x, y, end_x, gy),
            fill=(110, 232, 248, 140),
            width=max(2, px(4)),
        )
        segments = 10
        for i in range(segments):
            q = i / (segments - 1)
            x = lerp(start_x, end_x, q)
            yy = lerp(y, gy, q)
            rr = px(3 if i % 2 else 5)
            draw.ellipse((x - rr, yy - rr, x + rr, yy + rr), fill=(110, 232, 248, 150))
        image.alpha_composite(overlay)

        draw_text(
            image,
            example["distance_label"],
            (OUT_W // 2, int(OUT_H * 0.58)),
            size=ui(52),
            fill=(110, 232, 248, 248),
            bold=True,
            anchor="ma",
            stroke=4,
        )
        draw_text(
            image,
            example["ly_label"],
            (OUT_W // 2, int(OUT_H * 0.63)),
            size=ui(21),
            fill=(235, 243, 252, 220),
            bold=True,
            anchor="ma",
            stroke=2,
        )
        draw_text(
            image,
            f"≈ {speed:,.0f} km/s",
            (OUT_W // 2, int(OUT_H * 0.71)),
            size=ui(66),
            fill=(255, 187, 96, 250),
            bold=True,
            anchor="ma",
            stroke=4,
        )
        draw_text(
            image,
            "using H₀ = 70 km/s/Mpc",
            (OUT_W // 2, int(OUT_H * 0.77)),
            size=ui(20),
            fill=(245, 249, 255, 225),
            bold=True,
            anchor="ma",
            stroke=2,
        )

    def draw_examples(self, image: Image.Image, t: float) -> None:
        start = 24.0 if not QUICK_MODE else 5.1
        duration = 15.5 if not QUICK_MODE else 3.2
        frac = clamp((t - start) / max(duration, 0.01))
        index = min(len(DISTANCE_EXAMPLES) - 1, int(frac * len(DISTANCE_EXAMPLES)))
        example = DISTANCE_EXAMPLES[index]
        self.draw_speed_card(image, example, t)

        draw_text(
            image,
            f"EXAMPLE {index + 1} / {len(DISTANCE_EXAMPLES)}",
            (OUT_W // 2, px(82)),
            size=ui(20),
            fill=(150, 210, 230, 215),
            bold=True,
            anchor="ma",
            stroke=2,
        )

    def draw_local_exception(self, image: Image.Image, t: float) -> None:
        start = 39.5 if not QUICK_MODE else 8.3
        duration = 11.0 if not QUICK_MODE else 2.1
        frac = clamp((t - start) / max(duration, 0.01))

        cx1, cy1 = OUT_W * 0.30, OUT_H * 0.40
        cx2, cy2 = OUT_W * 0.70, OUT_H * 0.40
        approach = lerp(0, px(58), smoothstep(frac))

        self.draw_galaxy(
            image,
            (cx1 + approach, cy1),
            42 * SCALE,
            0.48,
            -0.45 + t * 0.03,
            warm=False,
            alpha=245,
        )
        self.draw_galaxy(
            image,
            (cx2 - approach, cy2),
            52 * SCALE,
            0.42,
            0.36 - t * 0.02,
            warm=True,
            alpha=245,
        )

        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        mid = OUT_W * 0.50
        y = OUT_H * 0.40
        draw.line(
            (cx1 + approach + px(65), y, mid - px(16), y),
            fill=(255, 183, 94, 210),
            width=max(2, px(4)),
        )
        draw.line(
            (cx2 - approach - px(65), y, mid + px(16), y),
            fill=(255, 183, 94, 210),
            width=max(2, px(4)),
        )
        image.alpha_composite(overlay)

        draw_text(
            image,
            "NEARBY EXCEPTION",
            (OUT_W // 2, px(88)),
            size=ui(29),
            fill=(255, 190, 100, 245),
            bold=True,
            anchor="ma",
            stroke=3,
        )
        draw_text(
            image,
            "LOCAL GRAVITY CAN WIN",
            (OUT_W // 2, int(OUT_H * 0.62)),
            size=ui(48),
            fill=(110, 232, 248, 248),
            bold=True,
            anchor="ma",
            stroke=4,
        )
        draw_text(
            image,
            "Some neighbouring galaxies approach or orbit each other",
            (OUT_W // 2, int(OUT_H * 0.69)),
            size=ui(23),
            fill=(245, 249, 255, 232),
            bold=True,
            anchor="ma",
            stroke=2,
        )
        draw_text(
            image,
            "Cosmic expansion dominates only across larger scales",
            (OUT_W // 2, int(OUT_H * 0.75)),
            size=ui(21),
            fill=(255, 194, 108, 225),
            bold=True,
            anchor="ma",
            stroke=2,
        )

    def draw_outro(self, image: Image.Image, t: float) -> None:
        start = 50.5 if not QUICK_MODE else 10.4
        frac = clamp((t - start) / max(CONFIG["duration_s"] - start, 0.01))
        self.draw_expanding_field(
            image,
            t,
            expansion=lerp(0.82, 1.16, smoothstep(frac)),
            show_arrows=True,
            limit=50 if QUICK_MODE else 120,
        )

        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        x0, y0 = px(58), int(OUT_H * 0.60)
        x1, y1 = OUT_W - px(58), int(OUT_H * 0.81)
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=max(12, px(26)),
            fill=(2, 8, 20, 232),
            outline=(95, 215, 242, 160),
            width=max(1, px(2)),
        )
        image.alpha_composite(panel)

        draw_text(
            image,
            "CAN RECESSION EXCEED LIGHT SPEED?",
            (OUT_W // 2, y0 + px(35)),
            size=ui(30),
            fill=(255, 190, 100, 245),
            bold=True,
            anchor="ma",
            stroke=3,
        )
        draw_text(
            image,
            "YES—AT EXTREME DISTANCES",
            (OUT_W // 2, y0 + px(82)),
            size=ui(39),
            fill=(110, 232, 248, 248),
            bold=True,
            anchor="ma",
            stroke=3,
        )
        draw_text(
            image,
            "Space expands; galaxies are not locally breaking the speed limit.",
            (OUT_W // 2, y0 + px(127)),
            size=ui(20),
            fill=(245, 249, 255, 230),
            bold=True,
            anchor="ma",
            stroke=2,
        )
        draw_text(
            image,
            "For very large distances, a full cosmological model replaces v = H₀d.",
            (OUT_W // 2, y0 + px(162)),
            size=ui(17),
            fill=(255, 198, 112, 218),
            bold=True,
            anchor="ma",
            stroke=2,
        )

    def draw_source_hud(self, image: Image.Image) -> None:
        draw_text(
            image,
            "H₀ = 70 km/s/Mpc  •  ROUNDED VISUAL REFERENCE",
            (OUT_W - px(34), px(40)),
            size=ui(13),
            fill=(110, 232, 248, 225),
            bold=True,
            anchor="ra",
            stroke=2,
        )

    def draw_section_label(self, image: Image.Image, shot_name: str, t: float) -> None:
        labels = {
            "intro": "THE EXPANDING UNIVERSE",
            "law": "DISTANCE SETS THE COSMIC FLOW",
            "examples": "SIMPLE LOW-REDSHIFT ESTIMATES",
            "local": "GRAVITY CREATES EXCEPTIONS",
            "outro": "EXPANDING SPACE IS NOT ORDINARY MOTION",
        }
        if t > (5.2 if not QUICK_MODE else 1.4):
            draw_text(
                image,
                labels[shot_name],
                (px(52), px(62)),
                size=ui(17),
                fill=(150, 210, 230, 205),
                bold=True,
                stroke=2,
            )

    def draw_caption(self, image: Image.Image, t: float) -> None:
        text = caption_at(t)
        if not text:
            return
        panel_h = px(196)
        y0 = OUT_H - px(320)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        bounds = (px(34), y0, OUT_W - px(34), y0 + panel_h)
        draw.rounded_rectangle(
            bounds,
            radius=max(12, px(26)),
            fill=(1, 7, 18, 232),
            outline=(90, 210, 242, 155),
            width=max(1, px(2)),
        )
        draw.rectangle(
            (px(34), y0 + px(18), px(42), y0 + panel_h - px(18)),
            fill=(95, 225, 248, 235),
        )
        image.alpha_composite(panel)
        draw_wrapped_text(
            image,
            text,
            (px(64), y0 + px(29)),
            OUT_W - px(128),
            size=ui(34),
            fill=(250, 252, 255, 255),
            bold=True,
            line_spacing=px(8),
        )

    def draw_hud_noise(self, image: Image.Image, t: float) -> None:
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        offset = int((t * 39) % 7)
        for y in range(offset, OUT_H, 7):
            draw.line((0, y, OUT_W, y), fill=(120, 200, 240, 4), width=1)
        scan_y = int((t * 160) % (OUT_H + 220)) - 110
        draw.rectangle((0, scan_y, OUT_W, scan_y + px(46)), fill=(80, 210, 240, 4))
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        image = self.background(t)
        name = shot["name"]

        if name == "intro":
            self.draw_intro(image, t)
        elif name == "law":
            self.draw_law(image, t)
        elif name == "examples":
            self.draw_examples(image, t)
        elif name == "local":
            self.draw_local_exception(image, t)
        else:
            self.draw_outro(image, t)

        self.draw_source_hud(image)
        self.draw_section_label(image, name, t)
        self.draw_caption(image, t)
        self.draw_hud_noise(image, t)

        array = np.array(image.convert("RGB"))
        array = apply_grade(array)
        array = np.clip(array.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / 0.85)
        fade_out = 1.0 - smoothstep((t - (CONFIG["duration_s"] - 1.0)) / 0.9)
        return np.clip(array.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def save_metadata() -> Tuple[Path, Path]:
    metadata_path = OUTPUT_ROOT / "youtube_title_and_description.txt"
    metadata_path.write_text(
        f"TITLE\n{YOUTUBE_TITLE}\n\nDESCRIPTION\n{YOUTUBE_DESCRIPTION.strip()}\n",
        encoding="utf-8",
    )

    examples_path = DATA_DIR / "hubble_law_examples.json"
    examples_path.write_text(
        json.dumps(
            {
                "rounded_hubble_reference_km_s_mpc": CONFIG["hubble_reference"],
                "formula": "recession_speed_km_s ≈ H0 × distance_mpc",
                "scope_note": (
                    "Useful as a low-redshift visual approximation. Very distant "
                    "objects require a full cosmological model."
                ),
                "measurement_note": (
                    "The exact Hubble constant remains under active study; "
                    "70 km/s/Mpc is used as a rounded storytelling value."
                ),
                "examples": [
                    {
                        **example,
                        "speed_km_s": CONFIG["hubble_reference"] * example["distance_mpc"],
                    }
                    for example in DISTANCE_EXAMPLES
                ],
                "references": [
                    "NASA Science — Hubble Cosmological Redshift",
                    "NASA Science — Hubble Constant and Tension",
                    "ESA — Galaxies and the Expanding Universe",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata_path, examples_path


def render_video(scene: GalaxyExpansionScene) -> Tuple[Path, Path]:
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)

    raw_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    times = np.arange(frame_count) / CONFIG["fps"]

    print(
        f"Rendering {frame_count:,} frames at "
        f"{OUT_W}x{OUT_H} and {CONFIG['fps']} fps ..."
    )
    with iio.get_writer(
        raw_path,
        fps=CONFIG["fps"],
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=[
            "-crf", "15",
            "-preset", "slow",
            "-profile:v", "high",
            "-level", "4.2",
            "-movflags", "+faststart",
        ],
    ) as writer:
        for t in tqdm(times, desc="Rendering galaxy-recession short"):
            writer.append_data(scene.render_frame(float(t)))

    shutil.copyfile(raw_path, final_path)
    return final_path, srt_path


def main() -> None:
    print("=" * 72)
    print("Running:", Path(__file__).resolve())
    print("Build:", SCRIPT_BUILD)
    print("Renderer: Pillow/NumPy only — Matplotlib and Axes3D are not used")
    print("=" * 72)

    metadata_path, examples_path = save_metadata()
    print("YouTube metadata:", metadata_path.resolve())
    print("Hubble-law examples:", examples_path.resolve())

    scene = GalaxyExpansionScene()

    preview_times = [
        1.0,
        min(10.0, CONFIG["duration_s"] * 0.22),
        min(26.0, CONFIG["duration_s"] * 0.45),
        min(35.0, CONFIG["duration_s"] * 0.62),
        min(44.0, CONFIG["duration_s"] * 0.80),
        CONFIG["duration_s"] - 1.0,
    ]
    for pt in tqdm(preview_times, desc="Creating preview frames"):
        Image.fromarray(scene.render_frame(float(pt))).save(
            PREVIEW_DIR / f"preview_{int(pt):02d}s.png"
        )

    final_path, srt_path = render_video(scene)
    print("Final video:", final_path.resolve())
    print("Subtitles:", srt_path.resolve())
    print("Output directory:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
