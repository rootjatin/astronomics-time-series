from __future__ import annotations

"""
Mariana Trench — cinematic underwater YouTube Short renderer

Creates a vertical 1080x1920 science/cinematic short showing a stylized descent
into the Mariana Trench. The animation is atmospheric and diagrammatic rather
than a bathymetric survey. It focuses on mood, scale, darkness, pressure, and
key facts about the trench.



Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    MARIANA_SHORT_QUICK=1 python3 mariana_trench.py

Full render
-----------
    python mariana_trench_cinematic_short.py

4K vertical
-----------
    MARIANA_SHORT_4K=1 python mariana_trench_cinematic_short.py
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

QUICK_MODE = os.environ.get("MARIANA_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("MARIANA_SHORT_4K", "0") == "1" and not QUICK_MODE

OUT_W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
OUT_H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 8 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 56.0
SCALE = OUT_W / 1080.0
OUT_SIZE = (OUT_W, OUT_H)

OUTPUT_ROOT = Path("mariana_trench_cinematic_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    
    },
}

COLORS = {

}

FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.4, 7.0, "Far out in the western Pacific lies the Mariana Trench — the deepest known part of Earth's ocean."),
    (7.1, 15.5, "As sunlight fades, the water turns cold, blue, and then almost completely black."),
    (15.6, 25.3, "The trench is nearly 11 kilometers deep. Challenger Deep sits at the bottom of this immense scar in the seafloor."),
    (25.4, 35.6, "Pressure rises enormously with depth — more than a thousand times the pressure we feel at sea level."),
    (35.7, 46.2, "Life still exists here. Tiny drifting particles, faint bioluminescent glows, and specialized deep-sea organisms endure the darkness."),
    (46.3, 55.2, "The Mariana Trench is not just deep. It is a world of crushing pressure, silence, and extreme isolation."),
]

if QUICK_MODE:
    factor = DURATION / 56.0
    CAPTIONS = [(a * factor, b * factor, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "surface", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "descent", "start": 8.0 if not QUICK_MODE else 1.8, "end": 17.0 if not QUICK_MODE else 3.8},
    {"name": "deep_ocean", "start": 17.0 if not QUICK_MODE else 3.8, "end": 26.5 if not QUICK_MODE else 5.9},
    {"name": "trench_walls", "start": 26.5 if not QUICK_MODE else 5.9, "end": 36.8 if not QUICK_MODE else 8.0},
    {"name": "challenger_deep", "start": 36.8 if not QUICK_MODE else 8.0, "end": 47.2 if not QUICK_MODE else 10.3},
    {"name": "finale", "start": 47.2 if not QUICK_MODE else 10.3, "end": DURATION},
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
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(7, int(size * SCALE)))
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
        stroke_width=max(1, int(stroke * SCALE)),
        stroke_fill=(0, 0, 0, min(225, fill[3] if len(fill) > 3 else 225)),
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
    anchor_center: bool = False,
):
    draw = ImageDraw.Draw(image)
    fnt = get_font(size, bold=bold)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        box = draw.textbbox((0, 0), candidate, font=fnt, stroke_width=max(1, int(2 * SCALE)))
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
        bbox = draw.textbbox((0, 0), line, font=fnt, stroke_width=max(1, int(2 * SCALE)))
        width = bbox[2] - bbox[0]
        tx = x - width // 2 if anchor_center else x
        draw.text(
            (tx, y),
            line,
            font=fnt,
            fill=fill,
            stroke_width=max(1, int(2 * SCALE)),
            stroke_fill=(0, 0, 0, 220),
        )
        y += (bbox[3] - bbox[1]) + int(line_spacing * SCALE)


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.7, 0.0, 1.0).astype(np.float32)


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


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Renderer
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Particle:
    x: float
    y: float
    size: float
    speed: float
    alpha: float
    drift: float
    phase: float


class MarianaTrenchScene:
    def __init__(self):
        rng = np.random.default_rng(20260817)
        self.particles = [
            Particle(
                x=float(rng.uniform(0, OUT_W)),
                y=float(rng.uniform(0, OUT_H)),
                size=float(rng.uniform(1.0, 4.5) * SCALE),
                speed=float(rng.uniform(8.0, 28.0) * SCALE),
                alpha=float(rng.uniform(16, 72)),
                drift=float(rng.uniform(-18.0, 18.0) * SCALE),
                phase=float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(140 if QUICK_MODE else 420)
        ]
        self.glow_creatures = [
            {
                "x": float(rng.uniform(0.15, 0.85)),
                "y": float(rng.uniform(0.25, 0.80)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "size": float(rng.uniform(7.0, 18.0) * SCALE),
                "color": rng.choice(["cyan", "green", "violet", "ice"]),
            }
            for _ in range(10 if QUICK_MODE else 26)
        ]
        self.hud_lines = [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "length": float(rng.uniform(16, 120) * SCALE),
                "a": float(rng.uniform(10, 36)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            }
            for _ in range(35 if QUICK_MODE else 88)
        ]

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 168):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            box,
            radius=max(10, int(24 * SCALE)),
            fill=(2, 8, 16, alpha),
            outline=COLORS["cyan"] + (52,),
            width=max(1, int(2 * SCALE)),
        )
        image.alpha_composite(overlay)

    def background(self, t: float, depth_factor: float) -> Image.Image:
        depth_factor = clamp(depth_factor)
        arr = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        yy = np.linspace(0, 1, OUT_H)[:, None]
        top_r = lerp(12, 3, depth_factor)
        top_g = lerp(92, 13, depth_factor)
        top_b = lerp(168, 28, depth_factor)
        bottom_r = lerp(2, 1, depth_factor)
        bottom_g = lerp(25, 4, depth_factor)
        bottom_b = lerp(52, 12, depth_factor)
        arr[..., 0] = np.clip(top_r * (1 - yy) + bottom_r * yy, 0, 255)
        arr[..., 1] = np.clip(top_g * (1 - yy) + bottom_g * yy, 0, 255)
        arr[..., 2] = np.clip(top_b * (1 - yy) + bottom_b * yy, 0, 255)
        image = Image.fromarray(arr, "RGB").convert("RGBA")

        # God rays near surface fade with depth.
        if depth_factor < 0.42:
            overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            strength = 1.0 - depth_factor / 0.42
            for k in range(6 if QUICK_MODE else 12):
                x = int((0.08 + k * 0.10) * OUT_W)
                top = -int(90 * SCALE)
                bottom = int(OUT_H * 0.72)
                width = int((36 + 8 * math.sin(t * 0.9 + k)) * SCALE)
                poly = [(x - width, top), (x + width, top), (x + int(250 * SCALE), bottom), (x + int(160 * SCALE), bottom)]
                draw.polygon(poly, fill=COLORS["ice"] + (int(12 * strength),))
            overlay = overlay.filter(ImageFilter.GaussianBlur(max(4, int(18 * SCALE))))
            image.alpha_composite(overlay)

        # Top shimmer.
        if depth_factor < 0.25:
            draw = ImageDraw.Draw(image)
            wave_y = int(OUT_H * 0.08)
            for i in range(22):
                y = wave_y + int(math.sin(t * 1.8 + i * 0.45) * 6 * SCALE)
                draw.line((0, y + i * int(2 * SCALE), OUT_W, y + i * int(2 * SCALE)), fill=COLORS["ice"] + (10,))
        return image

    def draw_particles(self, image: Image.Image, t: float, density: float = 1.0):
        density = clamp(density, 0.0, 1.5)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for p in self.particles:
            y = (p.y + t * p.speed) % OUT_H
            x = p.x + math.sin(t * 0.6 + p.phase) * p.drift
            rr = p.size
            a = int(p.alpha * density * (0.75 + 0.25 * math.sin(t * 1.7 + p.phase)))
            draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=COLORS["white"] + (a,))
        overlay = overlay.filter(ImageFilter.GaussianBlur(max(1, int(2 * SCALE))))
        image.alpha_composite(overlay)

    def draw_submersible(self, image: Image.Image, t: float, y_frac: float = 0.55, alpha: int = 255):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cx = int(OUT_W * 0.38 + math.sin(t * 0.9) * 18 * SCALE)
        cy = int(OUT_H * y_frac + math.sin(t * 1.4) * 10 * SCALE)
        body_w = int(168 * SCALE)
        body_h = int(62 * SCALE)
        draw.rounded_rectangle((cx - body_w // 2, cy - body_h // 2, cx + body_w // 2, cy + body_h // 2), radius=int(30 * SCALE), fill=(210, 217, 225, alpha), outline=(255, 255, 255, min(255, alpha)), width=max(1, int(2 * SCALE)))
        draw.ellipse((cx - int(28 * SCALE), cy - int(22 * SCALE), cx + int(28 * SCALE), cy + int(22 * SCALE)), fill=(48, 91, 122, alpha), outline=(220, 247, 255, alpha), width=max(1, int(2 * SCALE)))
        draw.rectangle((cx + int(54 * SCALE), cy - int(6 * SCALE), cx + int(120 * SCALE), cy + int(6 * SCALE)), fill=(185, 195, 205, alpha))
        light_x = cx + int(124 * SCALE)
        light_y = cy
        draw.ellipse((light_x - int(8 * SCALE), light_y - int(8 * SCALE), light_x + int(8 * SCALE), light_y + int(8 * SCALE)), fill=COLORS["gold"] + (alpha,))
        image.alpha_composite(overlay)

        beam = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        bd = ImageDraw.Draw(beam)
        bd.polygon([
            (light_x, light_y - int(14 * SCALE)),
            (light_x, light_y + int(14 * SCALE)),
            (light_x + int(360 * SCALE), light_y + int(112 * SCALE)),
            (light_x + int(360 * SCALE), light_y - int(112 * SCALE)),
        ], fill=COLORS["gold"] + (26,))
        beam = beam.filter(ImageFilter.GaussianBlur(max(4, int(14 * SCALE))))
        image.alpha_composite(beam)

    def draw_depth_scale(self, image: Image.Image, depth_text: str, progress: float):
        x = int(OUT_W * 0.11)
        y0 = int(OUT_H * 0.24)
        y1 = int(OUT_H * 0.76)
        draw = ImageDraw.Draw(image)
        draw.line((x, y0, x, y1), fill=COLORS["muted"] + (120,), width=max(2, int(3 * SCALE)))
        for idx, label in enumerate(["0 m", "1 km", "4 km", "8 km", "11 km"]):
            py = int(lerp(y0, y1, idx / 4))
            draw.line((x - int(10 * SCALE), py, x + int(10 * SCALE), py), fill=COLORS["white"] + (170,), width=max(1, int(2 * SCALE)))
            draw_text(image, label, (x - int(18 * SCALE), py), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (210,), anchor="ra", stroke=1)
        dot_y = int(lerp(y0, y1, clamp(progress)))
        draw.ellipse((x - int(9 * SCALE), dot_y - int(9 * SCALE), x + int(9 * SCALE), dot_y + int(9 * SCALE)), fill=COLORS["gold"] + (240,))
        draw_text(image, depth_text, (x + int(24 * SCALE), dot_y - int(18 * SCALE)), size=17 if not QUICK_MODE else 8, fill=COLORS["gold"] + (235,), bold=True, stroke=1)

    def draw_open_ocean_silhouette(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cx = int(OUT_W * 0.73)
        cy = int(OUT_H * 0.47 + math.sin(t * 1.5) * 16 * SCALE)
        pts = [
            (cx - int(110 * SCALE), cy),
            (cx - int(40 * SCALE), cy - int(28 * SCALE)),
            (cx + int(52 * SCALE), cy - int(14 * SCALE)),
            (cx + int(118 * SCALE), cy),
            (cx + int(52 * SCALE), cy + int(16 * SCALE)),
            (cx - int(46 * SCALE), cy + int(36 * SCALE)),
        ]
        draw.polygon(pts, fill=(12, 25, 40, 220))
        draw.polygon([(cx - int(20 * SCALE), cy - int(14 * SCALE)), (cx + int(25 * SCALE), cy - int(65 * SCALE)), (cx + int(40 * SCALE), cy - int(8 * SCALE))], fill=(12, 25, 40, 220))
        image.alpha_composite(overlay)

    def terrain_profile(self, image: Image.Image, t: float, openness: float = 0.0, trench: float = 0.0, floor: float = 0.82):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        top_y = int(OUT_H * floor)

        left_points: List[Tuple[int, int]] = [(0, OUT_H), (0, int(OUT_H * 0.52))]
        right_points: List[Tuple[int, int]] = [(OUT_W, OUT_H), (OUT_W, int(OUT_H * 0.52))]

        samples = 28 if QUICK_MODE else 56
        for i in range(samples + 1):
            p = i / samples
            x = int(p * OUT_W * 0.47)
            y = int(lerp(OUT_H * (0.52 - 0.10 * openness), top_y, p**(0.75 + 0.20 * trench)))
            y += int(math.sin(t * 0.25 + i * 0.5) * 7 * SCALE)
            y += int((0.5 - abs(p - 0.6)) * 45 * trench * SCALE)
            left_points.append((x, y))

        for i in range(samples + 1):
            p = i / samples
            x = int(OUT_W - p * OUT_W * 0.47)
            y = int(lerp(OUT_H * (0.54 - 0.10 * openness), top_y, p**(0.75 + 0.20 * trench)))
            y += int(math.cos(t * 0.27 + i * 0.55) * 7 * SCALE)
            y += int((0.5 - abs(p - 0.6)) * 45 * trench * SCALE)
            right_points.append((x, y))

        left_points.append((0, OUT_H))
        right_points.append((OUT_W, OUT_H))
        draw.polygon(left_points, fill=COLORS["rock"] + (250,))
        draw.polygon(right_points, fill=COLORS["silt"] + (245,))

        # Center floor or narrow trench bottom.
        floor_width = int(lerp(OUT_W * 0.62, OUT_W * 0.16, trench))
        cx0 = OUT_W // 2 - floor_width // 2
        cx1 = OUT_W // 2 + floor_width // 2
        draw.rectangle((cx0, top_y - int(5 * SCALE), cx1, OUT_H), fill=(52, 55, 66, 240))

        # Highlight edges.
        glow = overlay.filter(ImageFilter.GaussianBlur(max(3, int(8 * SCALE))))
        image.alpha_composite(glow)
        image.alpha_composite(overlay)

    def draw_bioluminescence(self, image: Image.Image, t: float, strength: float = 1.0):
        strength = clamp(strength)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for item in self.glow_creatures:
            x = int(item["x"] * OUT_W + math.sin(t * 0.6 + item["phase"]) * 26 * SCALE)
            y = int(item["y"] * OUT_H + math.cos(t * 0.8 + item["phase"]) * 18 * SCALE)
            rr = item["size"] * (0.85 + 0.25 * math.sin(t * 1.7 + item["phase"]))
            a = int(120 * strength * (0.55 + 0.45 * math.sin(t * 1.9 + item["phase"])))
            draw.ellipse((x - rr, y - rr * 0.65, x + rr, y + rr * 0.65), fill=COLORS[item["color"]] + (max(12, a),))
        overlay = overlay.filter(ImageFilter.GaussianBlur(max(4, int(12 * SCALE))))
        image.alpha_composite(overlay)

    def draw_title(self, image: Image.Image, t: float):
        intro_end = 5.0 if not QUICK_MODE else 1.1
        if t < intro_end:
            fade = smoothstep(t / (0.8 if not QUICK_MODE else 0.18))
            draw_text(image, "WATCH THE", (OUT_W // 2, int(OUT_H * 0.072)), size=31 if not QUICK_MODE else 15, fill=COLORS["white"] + (int(245 * fade),), bold=True, anchor="ma", stroke=2)
            draw_text(image, "MARIANA TRENCH", (OUT_W // 2, int(OUT_H * 0.109)), size=48 if not QUICK_MODE else 24, fill=COLORS["cyan"] + (int(250 * fade),), bold=True, anchor="ma", stroke=2)
            draw_text(image, "DESCEND INTO THE DEEPEST OCEAN", (OUT_W // 2, int(OUT_H * 0.150)), size=21 if not QUICK_MODE else 10, fill=COLORS["gold"] + (int(240 * fade),), bold=True, anchor="ma", stroke=2)

    def draw_corner_label(self, image: Image.Image, label: str):
        draw_text(image, label, (54 if not QUICK_MODE else 27, 58 if not QUICK_MODE else 29), size=18 if not QUICK_MODE else 9, fill=COLORS["muted"] + (210,), bold=True, stroke=1)

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        y0 = OUT_H - (246 if not QUICK_MODE else 124)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle(
            (46 if not QUICK_MODE else 23, y0, OUT_W - (46 if not QUICK_MODE else 23), y0 + (132 if not QUICK_MODE else 68)),
            radius=24 if not QUICK_MODE else 12,
            fill=(2, 7, 15, 180),
            outline=COLORS["cyan"] + (62,),
            width=max(1, int(SCALE)),
        )
        image.alpha_composite(panel)
        draw_wrapped_text(
            image,
            text,
            (68 if not QUICK_MODE else 34, y0 + (28 if not QUICK_MODE else 14)),
            OUT_W - (136 if not QUICK_MODE else 68),
            size=28 if not QUICK_MODE else 14,
            fill=COLORS["white"] + (245,),
        )

    def draw_hud(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for item in self.hud_lines:
            pulse = 0.5 + 0.5 * math.sin(t * 1.8 + item["phase"])
            if pulse < 0.72:
                continue
            y = (item["y"] + t * 9.0) % OUT_H
            draw.line((item["x"], y, item["x"] + item["length"], y), fill=COLORS["cyan"] + (int(item["a"] * pulse),), width=1)
        offset = int((t * 41) % 9)
        for y in range(offset, OUT_H, 9):
            draw.line((0, y, OUT_W, y), fill=(110, 190, 220, 7), width=1)
        image.alpha_composite(overlay)

    def draw_source_hud(self, image: Image.Image):
        draw_text(image, "CINEMATIC SCIENCE SHORT", (OUT_W - (48 if not QUICK_MODE else 24), 72 if not QUICK_MODE else 36), size=15 if not QUICK_MODE else 7, fill=COLORS["gold"] + (235,), bold=True, anchor="ra", stroke=1)
        draw_text(image, "VISUALIZATION // NOT TO SCALE", (OUT_W - (48 if not QUICK_MODE else 24), 101 if not QUICK_MODE else 51), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (205,), anchor="ra", stroke=1)
        draw_text(image, "MARIANA TRENCH // PACIFIC OCEAN", (OUT_W - (48 if not QUICK_MODE else 24), 128 if not QUICK_MODE else 64), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (190,), anchor="ra", stroke=1)

    # Scenes
    def scene_surface(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[0]
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_open_ocean_silhouette(image, t)
        self.draw_particles(image, t, density=0.32)
        self.draw_submersible(image, t, y_frac=0.58 - 0.06 * local)
        self.panel(image, (int(OUT_W * 0.08), int(OUT_H * 0.67), int(OUT_W * 0.92), int(OUT_H * 0.81)), alpha=164)
        draw_text(image, "THE DEEPEST KNOWN OCEAN TRENCH ON EARTH", (OUT_W // 2, int(OUT_H * 0.719)), size=23 if not QUICK_MODE else 11, fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "far below the surface of the western Pacific", (OUT_W // 2, int(OUT_H * 0.764)), size=17 if not QUICK_MODE else 8, fill=COLORS["ice"] + (225,), anchor="ma", stroke=1)
        self.draw_corner_label(image, "1 // SURFACE DESCENT")

    def scene_descent(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[1]
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_particles(image, t, density=0.48)
        self.draw_submersible(image, t, y_frac=0.52 + 0.10 * local)
        self.draw_depth_scale(image, ["300 m", "1.2 km", "3.4 km"][min(2, int(local * 2.99))], progress=0.08 + local * 0.34)
        self.panel(image, (int(OUT_W * 0.09), int(OUT_H * 0.67), int(OUT_W * 0.91), int(OUT_H * 0.81)), alpha=166)
        draw_text(image, "SUNLIGHT DISAPPEARS FAST", (OUT_W // 2, int(OUT_H * 0.719)), size=24 if not QUICK_MODE else 12, fill=COLORS["cyan"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "blue fades to darkness as the descent continues", (OUT_W // 2, int(OUT_H * 0.765)), size=16 if not QUICK_MODE else 8, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)
        self.draw_corner_label(image, "2 // TWILIGHT TO MIDNIGHT")

    def scene_deep_ocean(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[2]
        local = smoothstep((t - shot["start"] ) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_particles(image, t, density=0.72)
        self.draw_bioluminescence(image, t, strength=0.45 + 0.45 * local)
        self.draw_submersible(image, t, y_frac=0.59)
        self.panel(image, (int(OUT_W * 0.09), int(OUT_H * 0.67), int(OUT_W * 0.91), int(OUT_H * 0.82)), alpha=170)
        draw_text(image, CONFIG["facts"]["depth_text"].upper(), (OUT_W // 2, int(OUT_H * 0.714)), size=24 if not QUICK_MODE else 12, fill=COLORS["gold"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "Challenger Deep lies near the bottom of the trench", (OUT_W // 2, int(OUT_H * 0.758)), size=16 if not QUICK_MODE else 8, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)
        draw_text(image, "tiny flashes of life still appear in the dark", (OUT_W // 2, int(OUT_H * 0.796)), size=15 if not QUICK_MODE else 7, fill=COLORS["green"] + (220,), anchor="ma", stroke=1)
        self.draw_corner_label(image, "3 // OPEN ABYSS")

    def scene_trench_walls(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[3]
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.terrain_profile(image, t, openness=0.10, trench=0.65 * local + 0.25, floor=0.80)
        self.draw_particles(image, t, density=0.84)
        self.draw_submersible(image, t, y_frac=0.54)
        self.panel(image, (int(OUT_W * 0.07), int(OUT_H * 0.67), int(OUT_W * 0.93), int(OUT_H * 0.83)), alpha=174)
        draw_text(image, "THE SEAFLOOR FALLS INTO A GIANT NARROW TROUGH", (OUT_W // 2, int(OUT_H * 0.713)), size=21 if not QUICK_MODE else 10, fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "steep trench walls descend into extreme darkness", (OUT_W // 2, int(OUT_H * 0.758)), size=16 if not QUICK_MODE else 8, fill=COLORS["muted"] + (225,), anchor="ma", stroke=1)
        draw_text(image, CONFIG["facts"]["pressure_text"], (OUT_W // 2, int(OUT_H * 0.797)), size=15 if not QUICK_MODE else 7, fill=COLORS["gold"] + (225,), anchor="ma", stroke=1)
        self.draw_corner_label(image, "4 // TRENCH WALLS")

    def scene_challenger_deep(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[4]
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.terrain_profile(image, t, openness=0.0, trench=1.0, floor=0.83)
        self.draw_particles(image, t, density=0.94)
        self.draw_bioluminescence(image, t, strength=0.65)
        self.draw_submersible(image, t, y_frac=0.50, alpha=245)

        marker_x = int(OUT_W * 0.71)
        marker_y = int(OUT_H * 0.63)
        draw = ImageDraw.Draw(image)
        draw.line((marker_x - int(140 * SCALE), marker_y - int(40 * SCALE), marker_x, marker_y), fill=COLORS["cyan"] + (180,), width=max(1, int(2 * SCALE)))
        draw.ellipse((marker_x - int(6 * SCALE), marker_y - int(6 * SCALE), marker_x + int(6 * SCALE), marker_y + int(6 * SCALE)), fill=COLORS["cyan"] + (245,))
        draw_text(image, "CHALLENGER DEEP", (marker_x - int(150 * SCALE), marker_y - int(56 * SCALE)), size=18 if not QUICK_MODE else 9, fill=COLORS["cyan"] + (245,), bold=True, anchor="ra", stroke=1)
        draw_text(image, "DEEPEST KNOWN POINT", (marker_x - int(150 * SCALE), marker_y - int(28 * SCALE)), size=14 if not QUICK_MODE else 7, fill=COLORS["white"] + (220,), anchor="ra", stroke=1)

        self.panel(image, (int(OUT_W * 0.08), int(OUT_H * 0.68), int(OUT_W * 0.92), int(OUT_H * 0.84)), alpha=186)
        draw_text(image, "DOWN HERE THERE IS NO SUNLIGHT", (OUT_W // 2, int(OUT_H * 0.719)), size=24 if not QUICK_MODE else 12, fill=COLORS["white"] + (246,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "only the submersible lights and faint bioluminescence break the darkness", (OUT_W // 2, int(OUT_H * 0.765)), size=15 if not QUICK_MODE else 7, fill=COLORS["ice"] + (225,), anchor="ma", stroke=1)
        draw_text(image, "immense pressure defines this environment", (OUT_W // 2, int(OUT_H * 0.804)), size=16 if not QUICK_MODE else 8, fill=COLORS["gold"] + (228,), anchor="ma", stroke=1)
        self.draw_corner_label(image, "5 // CHALLENGER DEEP")

    def scene_finale(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[5]
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.terrain_profile(image, t, openness=0.0, trench=1.0, floor=0.84)
        self.draw_particles(image, t, density=1.0)
        self.draw_bioluminescence(image, t, strength=0.78)
        self.draw_submersible(image, t, y_frac=0.47)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle((0, 0, OUT_W, OUT_H), fill=(0, 0, 0, int(40 + 40 * local)))
        image.alpha_composite(overlay)
        self.panel(image, (int(OUT_W * 0.07), int(OUT_H * 0.60), int(OUT_W * 0.93), int(OUT_H * 0.84)), alpha=196)
        draw_text(image, "MARIANA TRENCH", (OUT_W // 2, int(OUT_H * 0.655)), size=30 if not QUICK_MODE else 15, fill=COLORS["cyan"] + (248,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "THE DEEPEST KNOWN PART OF THE OCEAN", (OUT_W // 2, int(OUT_H * 0.705)), size=20 if not QUICK_MODE else 10, fill=COLORS["white"] + (238,), bold=True, anchor="ma", stroke=1)
        draw_text(image, CONFIG["facts"]["depth_text"], (OUT_W // 2, int(OUT_H * 0.753)), size=18 if not QUICK_MODE else 9, fill=COLORS["gold"] + (235,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "darkness • cold • crushing pressure", (OUT_W // 2, int(OUT_H * 0.795)), size=17 if not QUICK_MODE else 8, fill=COLORS["ice"] + (225,), anchor="ma", stroke=1)
        self.draw_corner_label(image, "6 // THE ABYSS")

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        name = shot["name"]
        depth_lookup = {
            "surface": 0.05,
            "descent": 0.28,
            "deep_ocean": 0.60,
            "trench_walls": 0.82,
            "challenger_deep": 0.95,
            "finale": 0.98,
        }
        image = self.background(t, depth_lookup[name])

        if name == "surface":
            self.scene_surface(image, t)
        elif name == "descent":
            self.scene_descent(image, t)
        elif name == "deep_ocean":
            self.scene_deep_ocean(image, t)
        elif name == "trench_walls":
            self.scene_trench_walls(image, t)
        elif name == "challenger_deep":
            self.scene_challenger_deep(image, t)
        elif name == "finale":
            self.scene_finale(image, t)

        self.draw_source_hud(image)
        self.draw_title(image, t)
        self.draw_caption(image, t)
        self.draw_hud(image, t)

        array = np.asarray(image.convert("RGB"))
        array = apply_grade(array)
        array = np.clip(array.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / (0.9 if not QUICK_MODE else 0.22))
        fade_out = 1.0 - smoothstep((t - (DURATION - (1.15 if not QUICK_MODE else 0.25))) / (1.0 if not QUICK_MODE else 0.20))
        return np.clip(array.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def save_summary() -> Path:
    summary = {
        "title": CONFIG["title"],
        "format": f"{OUT_W}x{OUT_H} vertical MP4",
        "fps": FPS,
        "duration_s": DURATION,
        "quick_mode": QUICK_MODE,
        "four_k": FOUR_K,
        "key_points": [
            "The Mariana Trench lies in the western Pacific Ocean.",
            "Challenger Deep is the deepest known point in the trench.",
            "Depth is approximately 11 km.",
            "Sunlight fades away quickly with depth.",
            "Pressure becomes extreme in the deep trench.",
            "The deep ocean can still host specialized life and bioluminescence.",
        ],
        "visual_warning": "This animation is cinematic and diagrammatic, not a measured bathymetric reconstruction.",
    }
    path = OUTPUT_ROOT / "science_and_render_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def render_video(scene: MarianaTrenchScene) -> Path:
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    print("Subtitle sidecar:", srt_path.resolve())

    raw_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    final_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    frame_count = int(round(DURATION * FPS))
    times = np.arange(frame_count) / FPS
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")

    with iio.get_writer(
        raw_video,
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times, desc="Rendering Mariana Trench short"):
            writer.append_data(scene.render_frame(float(t)))

    shutil.copyfile(raw_video, final_video)
    print("Final video:", final_video.resolve())
    return final_video



