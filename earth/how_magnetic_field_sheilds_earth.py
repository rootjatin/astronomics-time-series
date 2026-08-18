from __future__ import annotations

"""
How Earth's Magnetic Field Shields Earth — cinematic YouTube Short renderer

Creates a vertical 1080x1920 science short explaining how Earth's magnetic
field interacts with the solar wind and forms the magnetosphere.


No external data or internet connection is required. All particles, stars and
field-line motion are deterministic so repeated renders look the same.

Recommended install
-------------------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview render
--------------------
    EARTH_MAGNETIC_SHORT_QUICK=1 python how_earths_magnetic_field_shields_earth_short.py

Full render
-----------
    python how_earths_magnetic_field_shields_earth_short.py

Outputs
-------
- MP4 video
- SRT subtitles
- PNG preview frames
- a small JSON science/production summary
"""

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("EARTH_MAGNETIC_SHORT_QUICK", "0") == "1"

OUTPUT_ROOT = Path("how_earths_magnetic_field_shields_earth_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "how_earths_magnetic_field_shields_earth",
    "title": "HOW EARTH'S MAGNETIC FIELD SHIELDS EARTH",
    "subtitle": "solar wind // magnetosphere // aurora",
    "background_stars": 180 if QUICK_MODE else 420,
    "solar_particles": 90 if QUICK_MODE else 240,
    "contrast": 1.08,
    "saturation": 1.08,
    "vignette": 0.25,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0

COLORS = {
    "space": (2, 6, 18),
    "white": (245, 250, 255),
    "muted": (155, 203, 226),
    "cyan": (74, 226, 255),
    "blue": (76, 128, 255),
    "violet": (178, 108, 255),
    "gold": (255, 194, 90),
    "orange": (255, 136, 67),
    "red": (255, 83, 111),
    "green": (94, 243, 174),
    "earth_blue": (26, 104, 196),
    "ocean": (15, 69, 149),
    "land": (74, 174, 116),
    "night": (3, 20, 48),
}
FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.4, 7.3, "The Sun constantly blows charged particles through space. This stream is called the solar wind."),
    (7.4, 17.2, "Earth behaves like a giant magnet. Motion in its liquid outer core generates a magnetic field that reaches far into space."),
    (17.3, 28.3, "Charged solar-wind particles are pushed onto curved paths by the magnetic field, so much of the flow is diverted around Earth."),
    (28.4, 39.2, "That interaction creates the magnetosphere: compressed on the Sun-facing side and stretched into a long magnetotail behind Earth."),
    (39.3, 49.4, "The shield is not perfect. Some particles enter and can be guided toward the poles, where they help create auroras high in the atmosphere."),
    (49.5, 57.5, "So Earth's magnetic field is not a force field. Together with the atmosphere, it reduces our direct exposure to the solar wind."),
]

if QUICK_MODE:
    _scale = float(CONFIG["duration_s"]) / 58.0
    CAPTIONS = [(a * _scale, b * _scale, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "solar_wind", "start": 0.0, "end": 7.8 if not QUICK_MODE else 1.65},
    {"name": "dipole", "start": 7.8 if not QUICK_MODE else 1.65, "end": 18.2 if not QUICK_MODE else 3.78},
    {"name": "deflection", "start": 18.2 if not QUICK_MODE else 3.78, "end": 29.2 if not QUICK_MODE else 6.08},
    {"name": "magnetosphere", "start": 29.2 if not QUICK_MODE else 6.08, "end": 40.5 if not QUICK_MODE else 8.42},
    {"name": "aurora", "start": 40.5 if not QUICK_MODE else 8.42, "end": 50.6 if not QUICK_MODE else 10.48},
    {"name": "finale", "start": 50.6 if not QUICK_MODE else 10.48, "end": float(CONFIG["duration_s"])},
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
            return ImageFont.truetype(candidate, size=max(7, int(size)))
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
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
            stroke_width=2,
            stroke_fill=(0, 0, 0, 220),
        )
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


def cubic_bezier(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    count: int = 120,
) -> List[Tuple[float, float]]:
    values = np.linspace(0.0, 1.0, count)
    out: List[Tuple[float, float]] = []
    for u in values:
        v = 1.0 - u
        x = v**3 * p0[0] + 3 * v * v * u * p1[0] + 3 * v * u * u * p2[0] + u**3 * p3[0]
        y = v**3 * p0[1] + 3 * v * v * u * p1[1] + 3 * v * u * u * p2[1] + u**3 * p3[1]
        out.append((x, y))
    return out


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class EarthMagneticFieldScene:
    def __init__(self):
        self.earth_center = (int(OUT_W * 0.57), int(OUT_H * 0.42))
        self.earth_radius = int(150 * SCALE)
        self.stars = self._make_stars(int(CONFIG["background_stars"]), seed=20260422)
        self.particles = self._make_particles(int(CONFIG["solar_particles"]), seed=1859)
        self.hud = self._make_hud(58 if not QUICK_MODE else 26, seed=1969)

    @staticmethod
    def _make_stars(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.4, 2.0) * SCALE),
                "a": float(rng.uniform(22, 120)),
                "phase": float(rng.uniform(0.0, 2 * math.pi)),
            }
            for _ in range(count)
        ]

    @staticmethod
    def _make_particles(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "y": float(rng.uniform(OUT_H * 0.18, OUT_H * 0.68)),
                "phase": float(rng.uniform(0.0, 1.0)),
                "speed": float(rng.uniform(0.75, 1.35)),
                "size": float(rng.uniform(1.5, 4.2) * SCALE),
                "charge": float(rng.choice([-1.0, 1.0])),
                "alpha": float(rng.uniform(120, 245)),
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
                "length": float(rng.uniform(10, 95) * SCALE),
                "a": float(rng.uniform(8, 40)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            }
            for _ in range(count)
        ]

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, COLORS["space"] + (255,))
        draw = ImageDraw.Draw(image)

        for star in self.stars:
            alpha = int(star["a"] * (0.72 + 0.28 * math.sin(t * 1.4 + star["phase"])))
            r = star["r"]
            draw.ellipse(
                (star["x"] - r, star["y"] - r, star["x"] + r, star["y"] + r),
                fill=COLORS["white"] + (alpha,),
            )

        # Soft solar glow from off-screen left.
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        sun_x = -int(120 * SCALE)
        sun_y = int(OUT_H * 0.41)
        for radius, alpha in [
            (600 * SCALE, 11),
            (410 * SCALE, 17),
            (250 * SCALE, 28),
            (125 * SCALE, 80),
        ]:
            gd.ellipse((sun_x - radius, sun_y - radius, sun_x + radius, sun_y + radius), fill=COLORS["orange"] + (alpha,))
        glow = glow.filter(ImageFilter.GaussianBlur(max(12, int(45 * SCALE))))
        image.alpha_composite(glow)
        return image

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 170):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            box,
            radius=max(8, int(24 * SCALE)),
            fill=(2, 7, 18, alpha),
            outline=COLORS["cyan"] + (64,),
            width=max(1, int(2 * SCALE)),
        )
        image.alpha_composite(overlay)

    def draw_sun_edge(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cx = -int(105 * SCALE)
        cy = int(OUT_H * 0.41)
        radius = int(205 * SCALE)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 169, 69, 238))
        for k in range(11):
            angle = t * 0.15 + k * (2 * math.pi / 11)
            r1 = radius * 0.88
            r2 = radius * (1.08 + 0.05 * math.sin(t * 1.7 + k))
            x1 = cx + r1 * math.cos(angle)
            y1 = cy + r1 * math.sin(angle)
            x2 = cx + r2 * math.cos(angle)
            y2 = cy + r2 * math.sin(angle)
            draw.line((x1, y1, x2, y2), fill=COLORS["gold"] + (110,), width=max(1, int(3 * SCALE)))
        glow = overlay.filter(ImageFilter.GaussianBlur(max(8, int(28 * SCALE))))
        image.alpha_composite(glow)
        image.alpha_composite(overlay)

    def draw_earth(self, image: Image.Image, t: float, alpha: int = 255, atmosphere: bool = True):
        cx, cy = self.earth_center
        r = self.earth_radius

        if atmosphere:
            glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow)
            for extra, a in [(38, 16), (24, 28), (12, 60)]:
                rr = r + int(extra * SCALE)
                gd.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=COLORS["cyan"] + (a,))
            glow = glow.filter(ImageFilter.GaussianBlur(max(4, int(16 * SCALE))))
            image.alpha_composite(glow)

        earth = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(earth)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=COLORS["earth_blue"] + (alpha,))

        # Simple stylized continents, clipped visually by drawing them inside the globe.
        land_alpha = min(alpha, 235)
        continents = [
            [(-0.55, -0.45), (-0.20, -0.58), (0.02, -0.37), (-0.08, -0.12), (-0.34, -0.04), (-0.48, -0.20)],
            [(0.10, -0.10), (0.42, -0.24), (0.60, -0.02), (0.45, 0.18), (0.18, 0.28), (0.02, 0.10)],
            [(-0.18, 0.25), (0.08, 0.18), (0.20, 0.43), (-0.01, 0.69), (-0.25, 0.53), (-0.32, 0.34)],
        ]
        rotation = 0.05 * math.sin(t * 0.4)
        for poly in continents:
            points = []
            for px, py in poly:
                x = px * math.cos(rotation) - py * math.sin(rotation)
                y = px * math.sin(rotation) + py * math.cos(rotation)
                points.append((cx + x * r, cy + y * r))
            draw.polygon(points, fill=COLORS["land"] + (land_alpha,))

        # Night-side shading on the right.
        shade = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shade)
        sd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0, 0))
        sd.pieslice((cx - r, cy - r, cx + r, cy + r), start=270, end=90, fill=COLORS["night"] + (145,))
        shade = shade.filter(ImageFilter.GaussianBlur(max(1, int(3 * SCALE))))
        earth.alpha_composite(shade)

        # Sun-facing rim.
        draw.arc((cx - r, cy - r, cx + r, cy + r), start=92, end=268, fill=COLORS["cyan"] + (200,), width=max(1, int(4 * SCALE)))
        image.alpha_composite(earth)

    def magnetic_poles(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        cx, cy = self.earth_center
        r = self.earth_radius * 0.92
        tilt = math.radians(11.0)
        north = (cx + r * math.sin(tilt), cy - r * math.cos(tilt))
        south = (cx - r * math.sin(tilt), cy + r * math.cos(tilt))
        return north, south

    def draw_field_lines(self, image: Image.Image, reveal: float = 1.0, distorted: float = 0.0, alpha: int = 180):
        reveal = clamp(reveal)
        distorted = clamp(distorted)
        north, south = self.magnetic_poles()
        cx, cy = self.earth_center
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Six loops per side, with the Sun-facing side compressed and night side stretched.
        for side in (-1, 1):
            for index in range(6):
                frac = (index + 1) / 6.0
                if side < 0:
                    reach = lerp(220, 430, frac) * SCALE
                    reach *= lerp(1.0, 0.64, distorted)
                else:
                    reach = lerp(220, 430, frac) * SCALE
                    reach *= lerp(1.0, 1.85, distorted)

                x_control = cx + side * reach
                vertical = lerp(35, 145, frac) * SCALE
                p1 = (x_control, north[1] - vertical)
                p2 = (x_control, south[1] + vertical)
                curve = cubic_bezier(north, p1, p2, south, count=130)
                count = max(2, int(len(curve) * reveal))
                curve = curve[:count]
                line_alpha = int(alpha * (0.48 + 0.52 * frac))
                draw.line(curve, fill=COLORS["cyan"] + (line_alpha,), width=max(1, int((1.4 + 1.2 * frac) * SCALE)))

                # Moving bead shows field-line direction rather than particle flow.
                if reveal > 0.65 and len(curve) > 8:
                    bead_index = int(((0.18 * index + reveal * 0.65) % 1.0) * (len(curve) - 1))
                    bx, by = curve[bead_index]
                    rr = (2.0 + frac * 2.2) * SCALE
                    draw.ellipse((bx - rr, by - rr, bx + rr, by + rr), fill=COLORS["white"] + (210,))

        glow = overlay.filter(ImageFilter.GaussianBlur(max(2, int(7 * SCALE))))
        image.alpha_composite(glow)
        image.alpha_composite(overlay)

    def draw_magnetic_axis(self, image: Image.Image, alpha: int = 180):
        north, south = self.magnetic_poles()
        cx, cy = self.earth_center
        dx = north[0] - cx
        dy = north[1] - cy
        length = self.earth_radius * 1.65
        norm = max(math.hypot(dx, dy), 1e-9)
        ux, uy = dx / norm, dy / norm
        p1 = (cx + ux * length, cy + uy * length)
        p2 = (cx - ux * length, cy - uy * length)
        ImageDraw.Draw(image).line((p1, p2), fill=COLORS["gold"] + (alpha,), width=max(1, int(2 * SCALE)))
        draw_text(image, "MAGNETIC AXIS", (int(p1[0] + 12 * SCALE), int(p1[1])), size=14 if not QUICK_MODE else 7, fill=COLORS["gold"] + (220,), bold=True, stroke=1)

    def draw_solar_wind(self, image: Image.Image, t: float, deflect: float = 0.0, intensity: float = 1.0):
        cx, cy = self.earth_center
        r = self.earth_radius
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        deflect = clamp(deflect)

        for index, particle in enumerate(self.particles):
            travel = ((particle["phase"] + t * 0.12 * particle["speed"]) % 1.0)
            x = lerp(-35 * SCALE, OUT_W + 60 * SCALE, travel)
            y0 = particle["y"]
            y = y0

            # Diagrammatic deflection: particles approaching the magnetosphere are
            # smoothly moved above/below Earth instead of crossing the globe.
            if deflect > 0.0:
                dx = x - cx
                dy = y0 - cy
                approach = math.exp(-((dx + 70 * SCALE) / max(1.0, 260 * SCALE)) ** 2)
                vertical_nearness = math.exp(-(dy / max(1.0, 220 * SCALE)) ** 2)
                direction = -1.0 if dy < 0 else 1.0
                if abs(dy) < 15 * SCALE:
                    direction = -1.0 if index % 2 == 0 else 1.0
                y += direction * deflect * approach * vertical_nearness * (165 * SCALE)

                # Avoid drawing particles inside Earth.
                if math.hypot(x - cx, y - cy) < r * 1.12:
                    continue

            colour = COLORS["cyan"] if particle["charge"] > 0 else COLORS["violet"]
            rr = particle["size"]
            a = int(particle["alpha"] * intensity)
            draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=colour + (a,))
            tail = max(6, int(18 * SCALE * particle["speed"]))
            draw.line((x - tail, y, x - rr, y), fill=colour + (max(18, a // 3),), width=max(1, int(2 * SCALE)))

        glow = overlay.filter(ImageFilter.GaussianBlur(max(1, int(3 * SCALE))))
        image.alpha_composite(glow)
        image.alpha_composite(overlay)

    def draw_magnetopause(self, image: Image.Image, pulse: float = 0.0, alpha: int = 170):
        cx, cy = self.earth_center
        nose_x = cx - 250 * SCALE
        tail_x = cx + 470 * SCALE
        upper_nose = (nose_x, cy)
        upper_tail = (tail_x, cy - 165 * SCALE)
        lower_tail = (tail_x, cy + 165 * SCALE)

        upper = cubic_bezier(
            upper_nose,
            (cx - 165 * SCALE, cy - 275 * SCALE),
            (cx + 150 * SCALE, cy - 260 * SCALE),
            upper_tail,
            count=130,
        )
        lower = cubic_bezier(
            upper_nose,
            (cx - 165 * SCALE, cy + 275 * SCALE),
            (cx + 150 * SCALE, cy + 260 * SCALE),
            lower_tail,
            count=130,
        )

        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width = max(1, int(3 * SCALE))
        a = int(alpha * (0.85 + 0.15 * math.sin(pulse * 2.0)))
        draw.line(upper, fill=COLORS["violet"] + (a,), width=width)
        draw.line(lower, fill=COLORS["violet"] + (a,), width=width)
        draw.line((upper_tail, (OUT_W + 40 * SCALE, cy - 120 * SCALE)), fill=COLORS["violet"] + (a // 2,), width=width)
        draw.line((lower_tail, (OUT_W + 40 * SCALE, cy + 120 * SCALE)), fill=COLORS["violet"] + (a // 2,), width=width)
        glow = overlay.filter(ImageFilter.GaussianBlur(max(2, int(8 * SCALE))))
        image.alpha_composite(glow)
        image.alpha_composite(overlay)

    def draw_aurora(self, image: Image.Image, strength: float = 1.0):
        strength = clamp(strength)
        north, south = self.magnetic_poles()
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        r = self.earth_radius

        for pole, flip in ((north, -1), (south, 1)):
            for band in range(5):
                width = r * (0.32 + 0.05 * band)
                height = r * (0.10 + 0.025 * band)
                x0 = pole[0] - width / 2
                x1 = pole[0] + width / 2
                y0 = pole[1] + flip * (8 + band * 5) * SCALE - height / 2
                y1 = y0 + height
                colour = COLORS["green"] if band % 2 == 0 else COLORS["cyan"]
                a = int((120 - band * 12) * strength)
                draw.arc((x0, y0, x1, y1), start=180, end=360, fill=colour + (a,), width=max(1, int((5 - band * 0.5) * SCALE)))

        glow = overlay.filter(ImageFilter.GaussianBlur(max(3, int(12 * SCALE))))
        image.alpha_composite(glow)
        image.alpha_composite(overlay)

    def draw_particle_funnels(self, image: Image.Image, t: float, strength: float):
        strength = clamp(strength)
        north, south = self.magnetic_poles()
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        rng = np.random.default_rng(77)
        for index in range(44 if not QUICK_MODE else 20):
            pole = north if index % 2 == 0 else south
            start_x = self.earth_center[0] + rng.uniform(80, 340) * SCALE
            start_y = self.earth_center[1] + rng.uniform(-260, 260) * SCALE
            u = (rng.uniform(0.0, 1.0) + t * rng.uniform(0.12, 0.24)) % 1.0
            # Quadratic-style interpolation pulled toward a pole.
            mid = (self.earth_center[0] + 120 * SCALE, (start_y + pole[1]) / 2)
            v = 1.0 - u
            x = v * v * start_x + 2 * v * u * mid[0] + u * u * pole[0]
            y = v * v * start_y + 2 * v * u * mid[1] + u * u * pole[1]
            rr = rng.uniform(1.2, 3.0) * SCALE
            colour = COLORS["violet"] if index % 3 else COLORS["cyan"]
            draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=colour + (int(190 * strength),))
        overlay = overlay.filter(ImageFilter.GaussianBlur(max(1, int(2 * SCALE))))
        image.alpha_composite(overlay)

    def draw_solar_wind_arrows(self, image: Image.Image, alpha: int = 180):
        draw = ImageDraw.Draw(image)
        for i in range(5):
            y = int(OUT_H * (0.27 + i * 0.07))
            x0 = int(60 * SCALE)
            x1 = int(230 * SCALE)
            draw.line((x0, y, x1, y), fill=COLORS["orange"] + (alpha,), width=max(1, int(3 * SCALE)))
            draw.polygon(
                [(x1, y), (x1 - 16 * SCALE, y - 8 * SCALE), (x1 - 16 * SCALE, y + 8 * SCALE)],
                fill=COLORS["orange"] + (alpha,),
            )

    def draw_solar_wind_scene(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[0]
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_sun_edge(image, t)
        self.draw_solar_wind(image, t, deflect=0.0, intensity=0.95)
        self.draw_earth(image, t, atmosphere=True)
        self.draw_solar_wind_arrows(image, alpha=int(80 + 110 * local))
        self.panel(image, (int(OUT_W * 0.10), int(OUT_H * 0.69), int(OUT_W * 0.90), int(OUT_H * 0.81)), alpha=164)
        draw_text(image, "THE SUN BLOWS CHARGED PARTICLES THROUGH SPACE", (OUT_W // 2, int(OUT_H * 0.732)), size=22 if not QUICK_MODE else 11, fill=COLORS["orange"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "this moving plasma is the solar wind", (OUT_W // 2, int(OUT_H * 0.775)), size=17 if not QUICK_MODE else 8, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)

    def draw_dipole_scene(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "dipole")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_sun_edge(image, t)
        self.draw_field_lines(image, reveal=min(1.0, local * 1.25), distorted=0.0, alpha=175)
        self.draw_earth(image, t)
        self.draw_magnetic_axis(image, alpha=int(190 * local))
        self.panel(image, (int(OUT_W * 0.09), int(OUT_H * 0.69), int(OUT_W * 0.91), int(OUT_H * 0.82)), alpha=166)
        draw_text(image, "EARTH GENERATES A PLANET-SCALE MAGNETIC FIELD", (OUT_W // 2, int(OUT_H * 0.731)), size=22 if not QUICK_MODE else 11, fill=COLORS["cyan"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "its source is the geodynamo in the liquid outer core", (OUT_W // 2, int(OUT_H * 0.775)), size=16 if not QUICK_MODE else 8, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)

    def draw_deflection_scene(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "deflection")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_sun_edge(image, t)
        self.draw_field_lines(image, reveal=1.0, distorted=0.55, alpha=155)
        self.draw_solar_wind(image, t, deflect=local, intensity=1.0)
        self.draw_earth(image, t)
        self.panel(image, (int(OUT_W * 0.09), int(OUT_H * 0.69), int(OUT_W * 0.91), int(OUT_H * 0.82)), alpha=166)
        draw_text(image, "CHARGED PARTICLES ARE DEFLECTED AROUND EARTH", (OUT_W // 2, int(OUT_H * 0.731)), size=22 if not QUICK_MODE else 11, fill=COLORS["violet"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "magnetic forces bend their paths instead of acting like a solid wall", (OUT_W // 2, int(OUT_H * 0.775)), size=15 if not QUICK_MODE else 7, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)

    def draw_magnetosphere_scene(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "magnetosphere")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_sun_edge(image, t)
        self.draw_solar_wind(image, t, deflect=1.0, intensity=0.88)
        self.draw_field_lines(image, reveal=1.0, distorted=local, alpha=145)
        self.draw_magnetopause(image, pulse=t, alpha=int(100 + 90 * local))
        self.draw_earth(image, t)
        draw_text(image, "COMPRESSED DAYSIDE", (int(OUT_W * 0.17), int(OUT_H * 0.29)), size=15 if not QUICK_MODE else 7, fill=COLORS["orange"] + (225,), bold=True, stroke=1)
        draw_text(image, "MAGNETOTAIL", (int(OUT_W * 0.78), int(OUT_H * 0.31)), size=15 if not QUICK_MODE else 7, fill=COLORS["violet"] + (225,), bold=True, anchor="ma", stroke=1)
        self.panel(image, (int(OUT_W * 0.09), int(OUT_H * 0.69), int(OUT_W * 0.91), int(OUT_H * 0.82)), alpha=166)
        draw_text(image, "THE MAGNETOSPHERE IS SHAPED BY THE SOLAR WIND", (OUT_W // 2, int(OUT_H * 0.731)), size=21 if not QUICK_MODE else 10, fill=COLORS["cyan"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "squashed toward the Sun • stretched far behind Earth", (OUT_W // 2, int(OUT_H * 0.775)), size=16 if not QUICK_MODE else 8, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)

    def draw_aurora_scene(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "aurora")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_field_lines(image, reveal=1.0, distorted=0.85, alpha=125)
        self.draw_magnetopause(image, pulse=t, alpha=115)
        self.draw_particle_funnels(image, t, strength=local)
        self.draw_earth(image, t)
        self.draw_aurora(image, strength=local)
        self.panel(image, (int(OUT_W * 0.09), int(OUT_H * 0.69), int(OUT_W * 0.91), int(OUT_H * 0.83)), alpha=172)
        draw_text(image, "SOME PARTICLES STILL GET IN", (OUT_W // 2, int(OUT_H * 0.727)), size=23 if not QUICK_MODE else 11, fill=COLORS["green"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "near the poles they can trigger auroras high in the atmosphere", (OUT_W // 2, int(OUT_H * 0.771)), size=15 if not QUICK_MODE else 7, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)
        draw_text(image, "SHIELD ≠ PERFECT BARRIER", (OUT_W // 2, int(OUT_H * 0.807)), size=15 if not QUICK_MODE else 7, fill=COLORS["gold"] + (225,), bold=True, anchor="ma", stroke=1)

    def draw_finale_scene(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "finale")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_sun_edge(image, t)
        self.draw_solar_wind(image, t, deflect=1.0, intensity=0.65)
        self.draw_field_lines(image, reveal=1.0, distorted=0.9, alpha=145)
        self.draw_magnetopause(image, pulse=t, alpha=140)
        self.draw_earth(image, t)
        self.draw_aurora(image, strength=0.65 + 0.35 * local)
        self.panel(image, (int(OUT_W * 0.07), int(OUT_H * 0.62), int(OUT_W * 0.93), int(OUT_H * 0.84)), alpha=188)
        draw_text(image, "EARTH'S MAGNETIC FIELD IS A SHIELD — NOT A WALL", (OUT_W // 2, int(OUT_H * 0.665)), size=24 if not QUICK_MODE else 12, fill=COLORS["white"] + (248,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "MAGNETOSPHERE", (OUT_W // 2, int(OUT_H * 0.716)), size=20 if not QUICK_MODE else 10, fill=COLORS["cyan"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "diverts much of the charged solar wind", (OUT_W // 2, int(OUT_H * 0.749)), size=16 if not QUICK_MODE else 8, fill=COLORS["white"] + (220,), anchor="ma", stroke=1)
        draw_text(image, "+  ATMOSPHERE", (OUT_W // 2, int(OUT_H * 0.786)), size=20 if not QUICK_MODE else 10, fill=COLORS["gold"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "provides another, different layer of protection", (OUT_W // 2, int(OUT_H * 0.818)), size=15 if not QUICK_MODE else 7, fill=COLORS["white"] + (220,), anchor="ma", stroke=1)

    def draw_titles(self, image: Image.Image, t: float, shot_name: str):
        intro_end = 5.4 if not QUICK_MODE else 1.05
        if t < intro_end:
            fade = smoothstep(t / (0.7 if not QUICK_MODE else 0.15))
            draw_text(image, "HOW EARTH'S", (OUT_W // 2, int(OUT_H * 0.075)), size=36 if not QUICK_MODE else 18, fill=COLORS["white"] + (int(245 * fade),), bold=True, anchor="ma", stroke=2)
            draw_text(image, "MAGNETIC FIELD", (OUT_W // 2, int(OUT_H * 0.108)), size=48 if not QUICK_MODE else 24, fill=COLORS["cyan"] + (int(250 * fade),), bold=True, anchor="ma", stroke=2)
            draw_text(image, "SHIELDS EARTH", (OUT_W // 2, int(OUT_H * 0.149)), size=40 if not QUICK_MODE else 20, fill=COLORS["gold"] + (int(245 * fade),), bold=True, anchor="ma", stroke=2)

        labels = {
            "solar_wind": "1 // THE SOLAR WIND",
            "dipole": "2 // EARTH'S MAGNETIC FIELD",
            "deflection": "3 // CHARGED-PARTICLE DEFLECTION",
            "magnetosphere": "4 // THE MAGNETOSPHERE",
            "aurora": "5 // SOME PARTICLES REACH THE POLES",
            "finale": "6 // SHIELD, NOT FORCE FIELD",
        }
        if t > (5.0 if not QUICK_MODE else 1.0):
            draw_text(image, labels[shot_name], (54 if not QUICK_MODE else 27, 60 if not QUICK_MODE else 30), size=18 if not QUICK_MODE else 9, fill=COLORS["muted"] + (205,), bold=True, stroke=1)

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        y0 = OUT_H - (250 if not QUICK_MODE else 126)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle(
            (44 if not QUICK_MODE else 22, y0, OUT_W - (44 if not QUICK_MODE else 22), y0 + (130 if not QUICK_MODE else 68)),
            radius=24 if not QUICK_MODE else 12,
            fill=(2, 6, 15, 178),
            outline=COLORS["cyan"] + (66,),
            width=1,
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

    def draw_source_hud(self, image: Image.Image):
        draw_text(image, "SCIENCE DIAGRAM // NOT TO SCALE", (OUT_W - (46 if not QUICK_MODE else 23), 72 if not QUICK_MODE else 36), size=15 if not QUICK_MODE else 7, fill=COLORS["gold"] + (235,), bold=True, anchor="ra", stroke=1)
        draw_text(image, "SOLAR WIND → CHARGED PARTICLES", (OUT_W - (46 if not QUICK_MODE else 23), 101 if not QUICK_MODE else 51), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (205,), anchor="ra", stroke=1)
        draw_text(image, "FIELD LINES → DIAGRAMMATIC", (OUT_W - (46 if not QUICK_MODE else 23), 128 if not QUICK_MODE else 64), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (190,), anchor="ra", stroke=1)

    def draw_hud_noise(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for item in self.hud:
            pulse = 0.5 + 0.5 * math.sin(t * 1.9 + item["phase"])
            if pulse < 0.76:
                continue
            y = (item["y"] + t * 8.0) % OUT_H
            draw.line((item["x"], y, item["x"] + item["length"], y), fill=COLORS["cyan"] + (int(item["a"] * pulse),), width=1)
        offset = int((t * 39) % 7)
        for y in range(offset, OUT_H, 7):
            draw.line((0, y, OUT_W, y), fill=(120, 200, 240, 9), width=1)
        scan_y = int((t * 164) % (OUT_H + 220)) - 110
        draw.rectangle((0, scan_y, OUT_W, scan_y + (48 if not QUICK_MODE else 24)), fill=(80, 210, 240, 7))
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        name = shot["name"]
        image = self.background(t)

        if name == "solar_wind":
            self.draw_solar_wind_scene(image, t)
        elif name == "dipole":
            self.draw_dipole_scene(image, t)
        elif name == "deflection":
            self.draw_deflection_scene(image, t)
        elif name == "magnetosphere":
            self.draw_magnetosphere_scene(image, t)
        elif name == "aurora":
            self.draw_aurora_scene(image, t)
        elif name == "finale":
            self.draw_finale_scene(image, t)

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
# Output helpers
# -----------------------------------------------------------------------------

def save_summary() -> Path:
    summary = {
        "title": CONFIG["title"],
        "format": f"{OUT_W}x{OUT_H} vertical MP4",
        "fps": CONFIG["fps"],
        "duration_s": CONFIG["duration_s"],
        "quick_mode": QUICK_MODE,
        "science_points": [
            "The solar wind is plasma containing charged particles from the Sun.",
            "Earth's geodynamo generates a large-scale magnetic field.",
            "The solar wind and magnetic field interact to form the magnetosphere.",
            "The dayside magnetosphere is compressed and the nightside forms a magnetotail.",
            "Some particles enter and can be guided toward polar regions, contributing to auroras.",
            "The magnetic field does not block all radiation; the atmosphere provides different protection.",
        ],
        "visual_warning": "All sizes, particle paths, field-line shapes, and timescales are diagrammatic and not to scale.",
    }
    path = OUTPUT_ROOT / "science_and_render_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def render_video(scene: EarthMagneticFieldScene) -> Path:
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
        for t in tqdm(times, desc="Rendering Earth magnetic-field short"):
            writer.append_data(scene.render_frame(float(t)))

    shutil.copyfile(raw_video, final_video)
    print("Final video:", final_video.resolve())
    return final_video


def main():
    print("Preparing Earth magnetic-field YouTube Short ...")
    print("Mode:", "QUICK" if QUICK_MODE else "FULL")
    print("Canvas:", f"{OUT_W}x{OUT_H}")
    print("FPS:", CONFIG["fps"])
    print("Duration:", CONFIG["duration_s"], "seconds")

    scene = EarthMagneticFieldScene()
    summary_path = save_summary()

    preview_times = [
        1.0,
        min(10.5, float(CONFIG["duration_s"]) * 0.20),
        min(22.5, float(CONFIG["duration_s"]) * 0.39),
        min(34.0, float(CONFIG["duration_s"]) * 0.60),
        min(45.0, float(CONFIG["duration_s"]) * 0.79),
        float(CONFIG["duration_s"]) - 1.0,
    ]
    for preview_time in tqdm(preview_times, desc="Preview frames"):
        Image.fromarray(scene.render_frame(float(preview_time))).save(
            PREVIEW_DIR / f"preview_{int(preview_time):02d}s.png"
        )

    print("Summary:", summary_path.resolve())
    render_video(scene)
    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-", path.name)


if __name__ == "__main__":
    main()
