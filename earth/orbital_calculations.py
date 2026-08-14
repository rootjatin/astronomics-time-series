from __future__ import annotations

"""
Calculating & Simulating Earth's Orbit Around the Sun — YouTube Short renderer

Creates a vertical 1080x1920 educational science Short that shows how to
calculate and numerically simulate gravitational orbits.

The video explains six ideas:

1. Newton's law of gravitation provides the force between two masses.
2. For a circular orbit, gravity supplies the centripetal acceleration, giving
       v = sqrt(G (M + m) / r)
   and
       T = 2*pi*sqrt(r^3 / (G (M + m))).
3. Earth's orbit is slightly elliptical, so the vis-viva equation
       v^2 = G (M + m) * (2/r - 1/a)
   gives the changing orbital speed.
4. When BOTH bodies are massive, neither is truly fixed: both orbit their
   shared center of mass (the barycenter).
5. A numerical integrator can advance positions and velocities step by step
   using Newtonian gravity. This renderer uses velocity-Verlet integration.
6. The same framework applies to planets, moons, binary stars, and many other
   two-body systems. Close to compact objects or at relativistic speeds,
   general relativity is required instead of a purely Newtonian model.

All calculations are deterministic and require no internet connection.
The Earth-Sun constants used here are standard approximate SI values.

Recommended install
-------------------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview render
--------------------
    ORBIT_SHORT_QUICK=1 python calculating_and_simulating_earths_orbit_short.py

Full render
-----------
    python calculating_and_simulating_earths_orbit_short.py

Outputs
-------
- MP4 video
- SRT subtitles
- PNG preview frames
- CSV with an Earth-Sun numerical orbit simulation
- JSON with the calculation summary and science notes
"""

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration and physical constants
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("ORBIT_SHORT_QUICK", "0") == "1"

OUTPUT_ROOT = Path("calculating_and_simulating_earths_orbit_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
DATA_DIR = OUTPUT_ROOT / "data"
for directory in (OUTPUT_ROOT, PREVIEW_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 8 if QUICK_MODE else 30,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "calculating_and_simulating_earths_orbit_cinematic_v2",
    "title": "EARTH IN MOTION",
    "subtitle": "A cinematic calculation of orbital mechanics",
    "background_stars": 260 if QUICK_MODE else 850,
    "contrast": 1.12,
    "saturation": 0.96,
    "vignette": 0.36,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0

# SI constants / approximate body data.
G = 6.67430e-11                  # m^3 kg^-1 s^-2
M_SUN = 1.98847e30               # kg
M_EARTH = 5.9722e24              # kg
AU = 1.495978707e11              # m
EARTH_E = 0.0167086              # orbital eccentricity (approximate)
DAY = 86400.0
YEAR = 365.256363004 * DAY
R_SUN = 6.957e8                   # m
R_EARTH = 6.371e6                 # m

MU_ES = G * (M_SUN + M_EARTH)
V_CIRC = math.sqrt(MU_ES / AU)
T_CIRC = 2.0 * math.pi * math.sqrt(AU**3 / MU_ES)
R_PERI = AU * (1.0 - EARTH_E)
R_APH = AU * (1.0 + EARTH_E)
V_PERI = math.sqrt(MU_ES * (2.0 / R_PERI - 1.0 / AU))
V_APH = math.sqrt(MU_ES * (2.0 / R_APH - 1.0 / AU))
ESCAPE_AT_1AU = math.sqrt(2.0 * MU_ES / AU)
SUN_BARY_RADIUS = AU * M_EARTH / (M_SUN + M_EARTH)
EARTH_BARY_RADIUS = AU * M_SUN / (M_SUN + M_EARTH)

COLORS = {
    "space": (3, 7, 18),
    "white": (247, 251, 255),
    "muted": (153, 198, 220),
    "cyan": (74, 226, 255),
    "blue": (82, 132, 255),
    "violet": (184, 112, 255),
    "gold": (255, 196, 82),
    "orange": (255, 137, 62),
    "red": (255, 83, 106),
    "green": (105, 235, 168),
    "earth": (74, 160, 255),
    "earth_land": (99, 216, 145),
    "sun": (255, 190, 63),
}

FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.4, 7.2, "An orbit is a continuous fall. Gravity pulls Earth inward, while Earth's sideways speed keeps it missing the Sun."),
    (7.3, 17.0, "For a circular orbit, gravity must supply the centripetal acceleration. That gives a first speed estimate."),
    (17.1, 27.2, "At one astronomical unit, Earth moves at about 29.8 kilometers per second and takes about 365.26 days to go around the Sun."),
    (27.3, 38.4, "Earth's real orbit is an ellipse with eccentricity 0.0167. The vis-viva equation tells you how speed changes with distance."),
    (38.5, 49.4, "When both bodies are massive, both move. They orbit a shared center of mass called the barycenter."),
    (49.5, 57.5, "A numerical integrator advances position and velocity step by step. That is how we simulate the orbit."),
]

if QUICK_MODE:
    _scale_t = float(CONFIG["duration_s"]) / 58.0
    CAPTIONS = [(a * _scale_t, b * _scale_t, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "fall", "start": 0.0, "end": 7.8 if not QUICK_MODE else 1.62},
    {"name": "circular", "start": 7.8 if not QUICK_MODE else 1.62, "end": 18.4 if not QUICK_MODE else 3.82},
    {"name": "numbers", "start": 18.4 if not QUICK_MODE else 3.82, "end": 28.2 if not QUICK_MODE else 5.85},
    {"name": "ellipse", "start": 28.2 if not QUICK_MODE else 5.85, "end": 39.2 if not QUICK_MODE else 8.13},
    {"name": "barycenter", "start": 39.2 if not QUICK_MODE else 8.13, "end": 49.8 if not QUICK_MODE else 10.34},
    {"name": "simulation", "start": 49.8 if not QUICK_MODE else 10.34, "end": float(CONFIG["duration_s"])},
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
            return ImageFont.truetype(candidate, size=max(7, int(size)))
        except Exception:
            pass
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
        stroke_width=max(0, int(stroke)),
        stroke_fill=(0, 0, 0, min(230, fill[3] if len(fill) > 3 else 230)),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    line_spacing: int = 6,
    anchor: str = "la",
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
        if anchor == "ma":
            draw.text((x, y), line, font=font, fill=fill, anchor="ma", stroke_width=2, stroke_fill=(0, 0, 0, 225))
            box = draw.textbbox((x, y), line, font=font, anchor="ma", stroke_width=2)
        else:
            draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 225))
            box = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += (box[3] - box[1]) + line_spacing


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    ms %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for i, (start, end, text) in enumerate(captions, start=1):
        lines.extend([str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.8, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


def apply_grade(rgb: np.ndarray) -> np.ndarray:
    image = Image.fromarray(rgb)
    image = ImageEnhance.Contrast(image).enhance(float(CONFIG["contrast"]))
    image = ImageEnhance.Color(image).enhance(float(CONFIG["saturation"]))
    array = np.asarray(image).astype(np.float32)
    array *= VIGNETTE[:, :, None]
    return np.clip(array, 0, 255).astype(np.uint8)


def arrow(draw: ImageDraw.ImageDraw, start: Tuple[float, float], end: Tuple[float, float], fill, width: int):
    draw.line([start, end], fill=fill, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    head = max(7 * SCALE, width * 2.7)
    for sign in (-1, 1):
        a = angle + math.pi + sign * math.pi / 6.0
        p = (end[0] + head * math.cos(a), end[1] + head * math.sin(a))
        draw.line([end, p], fill=fill, width=width)


# -----------------------------------------------------------------------------
# Newtonian two-body simulator
# -----------------------------------------------------------------------------

def accelerations(r1: np.ndarray, r2: np.ndarray, m1: float, m2: float) -> Tuple[np.ndarray, np.ndarray]:
    delta = r2 - r1
    dist2 = float(np.dot(delta, delta))
    dist = math.sqrt(max(dist2, 1e-30))
    inv_r3 = 1.0 / (dist2 * dist)
    a1 = G * m2 * delta * inv_r3
    a2 = -G * m1 * delta * inv_r3
    return a1, a2


def simulate_two_body(
    m1: float,
    m2: float,
    r1_0: np.ndarray,
    r2_0: np.ndarray,
    v1_0: np.ndarray,
    v2_0: np.ndarray,
    dt: float,
    steps: int,
) -> Dict[str, np.ndarray]:
    """Velocity-Verlet integration for two Newtonian point masses."""
    r1 = np.asarray(r1_0, dtype=float).copy()
    r2 = np.asarray(r2_0, dtype=float).copy()
    v1 = np.asarray(v1_0, dtype=float).copy()
    v2 = np.asarray(v2_0, dtype=float).copy()

    out_r1 = np.empty((steps + 1, 2), dtype=float)
    out_r2 = np.empty((steps + 1, 2), dtype=float)
    out_v1 = np.empty((steps + 1, 2), dtype=float)
    out_v2 = np.empty((steps + 1, 2), dtype=float)
    out_t = np.arange(steps + 1, dtype=float) * dt

    out_r1[0], out_r2[0], out_v1[0], out_v2[0] = r1, r2, v1, v2
    a1, a2 = accelerations(r1, r2, m1, m2)

    for i in range(1, steps + 1):
        r1_new = r1 + v1 * dt + 0.5 * a1 * dt * dt
        r2_new = r2 + v2 * dt + 0.5 * a2 * dt * dt
        a1_new, a2_new = accelerations(r1_new, r2_new, m1, m2)
        v1_new = v1 + 0.5 * (a1 + a1_new) * dt
        v2_new = v2 + 0.5 * (a2 + a2_new) * dt

        r1, r2, v1, v2, a1, a2 = r1_new, r2_new, v1_new, v2_new, a1_new, a2_new
        out_r1[i], out_r2[i], out_v1[i], out_v2[i] = r1, r2, v1, v2

    return {"t": out_t, "r1": out_r1, "r2": out_r2, "v1": out_v1, "v2": out_v2}


def earth_sun_initial_conditions() -> Dict[str, np.ndarray]:
    """Elliptical Earth-Sun state at perihelion in the barycentric frame."""
    separation = R_PERI
    v_rel = V_PERI
    r_sun = separation * M_EARTH / (M_SUN + M_EARTH)
    r_earth = separation * M_SUN / (M_SUN + M_EARTH)
    v_sun = v_rel * M_EARTH / (M_SUN + M_EARTH)
    v_earth = v_rel * M_SUN / (M_SUN + M_EARTH)
    return {
        "r1": np.array([-r_sun, 0.0]),
        "r2": np.array([r_earth, 0.0]),
        "v1": np.array([0.0, -v_sun]),
        "v2": np.array([0.0, v_earth]),
    }


def make_earth_sun_simulation() -> Dict[str, np.ndarray]:
    state = earth_sun_initial_conditions()
    dt = 0.5 * DAY
    steps = int(round(YEAR / dt))
    return simulate_two_body(M_SUN, M_EARTH, state["r1"], state["r2"], state["v1"], state["v2"], dt, steps)


def make_binary_demo() -> Dict[str, np.ndarray]:
    # Fictional binary stars: 1.0 and 0.65 solar masses, 1.2 AU apart.
    m1 = 1.0 * M_SUN
    m2 = 0.65 * M_SUN
    separation = 1.2 * AU
    omega = math.sqrt(G * (m1 + m2) / separation**3)
    r1_mag = separation * m2 / (m1 + m2)
    r2_mag = separation * m1 / (m1 + m2)
    state = {
        "r1": np.array([-r1_mag, 0.0]),
        "r2": np.array([r2_mag, 0.0]),
        "v1": np.array([0.0, -omega * r1_mag]),
        "v2": np.array([0.0, omega * r2_mag]),
    }
    period = 2.0 * math.pi / omega
    dt = period / 700.0
    return simulate_two_body(m1, m2, state["r1"], state["r2"], state["v1"], state["v2"], dt, 700)


EARTH_SIM = make_earth_sun_simulation()
BINARY_SIM = make_binary_demo()


def save_orbit_csv(path: Path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "time_days",
            "sun_x_m", "sun_y_m",
            "earth_x_m", "earth_y_m",
            "earth_sun_distance_m",
            "earth_speed_m_s",
        ])
        for i in range(len(EARTH_SIM["t"])):
            r1 = EARTH_SIM["r1"][i]
            r2 = EARTH_SIM["r2"][i]
            v2 = EARTH_SIM["v2"][i]
            writer.writerow([
                EARTH_SIM["t"][i] / DAY,
                r1[0], r1[1], r2[0], r2[1],
                float(np.linalg.norm(r2 - r1)),
                float(np.linalg.norm(v2)),
            ])


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class OrbitScene:
    """Cinematic renderer: restrained documentary graphics over simulated motion."""

    def __init__(self):
        rng = np.random.default_rng(20260812)
        self.stars = []
        for _ in range(int(CONFIG["background_stars"])):
            depth = float(rng.uniform(0.15, 1.0))
            self.stars.append(
                (
                    float(rng.uniform(-0.08 * OUT_W, 1.08 * OUT_W)),
                    float(rng.uniform(-0.06 * OUT_H, 1.06 * OUT_H)),
                    float(rng.uniform(0.28, 1.55) * max(SCALE, 0.58) * (0.55 + depth)),
                    int(rng.uniform(28, 150) * (0.5 + 0.5 * depth)),
                    float(rng.uniform(0, 2 * math.pi)),
                    depth,
                )
            )
        self.dust = [
            (
                float(rng.uniform(0, OUT_W)),
                float(rng.uniform(0, OUT_H)),
                float(rng.uniform(0.4, 1.0)),
                int(rng.uniform(8, 28)),
            )
            for _ in range(95 if QUICK_MODE else 240)
        ]

    # ------------------------------------------------------------------
    # Cinematic primitives
    # ------------------------------------------------------------------

    def camera(self, t: float) -> Tuple[float, float, float]:
        """Slow virtual dolly / drift. Returns x, y and zoom perturbations."""
        return (
            10.0 * SCALE * math.sin(t * 0.19),
            16.0 * SCALE * math.sin(t * 0.13 + 1.3),
            1.0 + 0.018 * math.sin(t * 0.11),
        )

    def background(self, t: float) -> Image.Image:
        # Deep navy-black gradient instead of a flat presentation background.
        yy = np.linspace(0.0, 1.0, OUT_H, dtype=np.float32)[:, None]
        top = np.array([2.0, 5.0, 12.0], dtype=np.float32)
        bottom = np.array([0.0, 2.0, 7.0], dtype=np.float32)
        rgb = top[None, None, :] * (1.0 - yy[:, :, None]) + bottom[None, None, :] * yy[:, :, None]
        rgb = np.repeat(rgb, OUT_W, axis=1)
        image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

        dx, dy, _ = self.camera(t)
        draw = ImageDraw.Draw(image)
        for x, y, r, a, phase, depth in self.stars:
            sx = x + dx * depth
            sy = y + dy * depth
            alpha = int(a * (0.83 + 0.17 * math.sin(0.55 * t + phase)))
            col = (214, 227, 242, alpha) if depth < 0.72 else (236, 241, 248, alpha)
            draw.ellipse((sx-r, sy-r, sx+r, sy+r), fill=col)

        # Fine film-like dust. Barely visible, but gives depth on moving shots.
        for x, y, depth, alpha in self.dust:
            sx = (x + 5.5 * t * depth) % OUT_W
            sy = y + 7.0 * SCALE * math.sin(t * 0.17 + x * 0.01)
            rr = max(0.3, 0.55 * SCALE * depth)
            draw.ellipse((sx-rr, sy-rr, sx+rr, sy+rr), fill=(158, 178, 200, alpha))

        # Large, low-opacity nebular gradients.
        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        hd.ellipse((-0.32*OUT_W, 0.12*OUT_H, 0.68*OUT_W, 0.96*OUT_H), fill=(21, 50, 91, 28))
        hd.ellipse((0.42*OUT_W, -0.15*OUT_H, 1.36*OUT_W, 0.66*OUT_H), fill=(70, 42, 91, 20))
        haze = haze.filter(ImageFilter.GaussianBlur(max(34, int(150*SCALE))))
        image.alpha_composite(haze)
        return image

    def soft_line(self, image: Image.Image, points, fill, width: float = 2.0, glow: float = 6.0):
        glow_layer = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        gd = ImageDraw.Draw(glow_layer)
        gd.line(points, fill=fill[:-1] + (max(8, int(fill[-1]*0.32)),), width=max(1, int(width*SCALE*3)))
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(max(2, int(glow*SCALE))))
        image.alpha_composite(glow_layer)
        ImageDraw.Draw(image).line(points, fill=fill, width=max(1, int(width*SCALE)))

    def label(self, image: Image.Image, text: str, xy: Tuple[int,int], size: int = 14, alpha: int = 205, anchor: str = "la", bold: bool = True):
        draw_text(image, text, xy, size=max(10, int((size+2)*SCALE)), fill=(215,226,236,alpha), bold=bold, stroke=2, anchor=anchor)

    def equation(self, image: Image.Image, text: str, xy: Tuple[int,int], size: int = 27, alpha: int = 245, anchor: str = "la", accent: bool = False):
        color = (240, 245, 250, alpha) if not accent else (207, 232, 248, alpha)
        draw_text(image, text, xy, size=max(12, int((size+4)*SCALE)), fill=color, bold=False, stroke=2, anchor=anchor)

    def glass_panel(self, image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 112):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rounded_rectangle(box, radius=max(10, int(20*SCALE)), fill=(4, 10, 20, alpha), outline=(170, 192, 214, min(60, alpha//2)), width=max(1, int(SCALE)))
        overlay = overlay.filter(ImageFilter.GaussianBlur(max(1, int(2*SCALE))))
        image.alpha_composite(overlay)

    def draw_sun(self, image: Image.Image, cx: float, cy: float, radius: float, pulse: float = 0.0):
        # Layered bloom gives a photographic rather than icon-like Sun.
        bloom = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        bd = ImageDraw.Draw(bloom)
        for mult, alpha in [(5.8, 8), (4.0, 14), (2.8, 28), (1.85, 56), (1.25, 98)]:
            rr = radius * mult * (1.0 + 0.012*math.sin(pulse*0.7))
            bd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=(255,171,70,alpha))
        bloom = bloom.filter(ImageFilter.GaussianBlur(max(5, int(radius*0.88))))
        image.alpha_composite(bloom)

        rr = max(8, int(radius))
        size = rr*2 + 4
        y, x = np.mgrid[-rr-2:rr+2, -rr-2:rr+2]
        rad = np.sqrt(x*x+y*y) / max(rr,1)
        mask = rad <= 1.0
        shade = np.clip(1.0 - rad, 0, 1)
        texture = 0.94 + 0.05*np.sin(x*0.19 + y*0.07 + pulse*0.5) * np.sin(y*0.13 - pulse*0.25)
        arr = np.zeros((size,size,4),dtype=np.uint8)
        arr[...,0] = np.clip(238 + 17*shade,0,255).astype(np.uint8)
        arr[...,1] = np.clip((126 + 105*shade)*texture,0,255).astype(np.uint8)
        arr[...,2] = np.clip((36 + 54*shade)*texture,0,255).astype(np.uint8)
        arr[...,3] = (mask*255).astype(np.uint8)
        orb = Image.fromarray(arr,"RGBA").filter(ImageFilter.GaussianBlur(max(0.2,0.7*SCALE)))
        image.alpha_composite(orb,(int(cx-size/2),int(cy-size/2)))

        # restrained horizontal lens flare
        flare = Image.new("RGBA",OUT_SIZE,(0,0,0,0)); fd=ImageDraw.Draw(flare)
        fd.line((cx-radius*2.3,cy,cx+radius*2.3,cy),fill=(255,202,137,72),width=max(1,int(1*SCALE)))
        for off, rmult, alpha in [(2.2,0.14,45),(3.7,0.09,27),(-2.9,0.06,18)]:
            fx=cx+radius*off
            fr=max(2,radius*rmult)
            fd.ellipse((fx-fr,cy-fr,fx+fr,cy+fr),outline=(255,190,125,alpha),width=max(1,int(SCALE)))
        flare=flare.filter(ImageFilter.GaussianBlur(max(0.6,1.7*SCALE))); image.alpha_composite(flare)

    def draw_earth(self, image: Image.Image, cx: float, cy: float, radius: float, phase: float = 0.0):
        # Procedural shaded sphere with atmosphere and terminator.
        rr=max(8,int(radius)); size=rr*2+8
        yy,xx=np.mgrid[-rr-4:rr+4,-rr-4:rr+4]
        nx=xx/max(rr,1); ny=yy/max(rr,1)
        r2=nx*nx+ny*ny
        mask=r2<=1.0
        nz=np.sqrt(np.clip(1.0-r2,0,1))
        # fixed light vector from upper-left
        lx,ly,lz=-0.58,-0.22,0.79
        light=np.clip(nx*lx+ny*ly+nz*lz,0,1)
        limb=np.clip(nz,0,1)
        ocean=np.zeros((size,size,4),dtype=np.uint8)
        ocean[...,0]=np.clip(7+20*light,0,255)
        ocean[...,1]=np.clip(28+73*light,0,255)
        ocean[...,2]=np.clip(54+118*light,0,255)
        # Soft pseudo-land texture from trigonometric fields; avoids cartoon blobs.
        lon=np.arctan2(nx,nz+1e-9)+phase*0.10
        lat=np.arcsin(np.clip(ny,-1,1))
        field=(np.sin(2.1*lon+0.7*np.sin(3.0*lat))+0.65*np.cos(3.3*lat-1.1*lon)+0.35*np.sin(5.2*lon+2.3*lat))
        land=(field>0.72)&mask
        land_light=np.clip(0.24+0.76*light,0,1)
        ocean[...,0]=np.where(land,np.clip(28+70*land_light,0,255),ocean[...,0])
        ocean[...,1]=np.where(land,np.clip(55+92*land_light,0,255),ocean[...,1])
        ocean[...,2]=np.where(land,np.clip(38+53*land_light,0,255),ocean[...,2])
        # Cloud bands
        clouds=(np.sin(8.0*lat+1.5*np.sin(2.5*lon+phase*0.08))>0.82)&mask&(light>0.16)
        for c in range(3):
            ocean[...,c]=np.where(clouds,np.clip(ocean[...,c]*0.48+150*light+40,0,255),ocean[...,c])
        ocean[...,3]=(mask*255).astype(np.uint8)
        sphere=Image.fromarray(ocean,"RGBA")
        image.alpha_composite(sphere,(int(cx-size/2),int(cy-size/2)))

        atmos=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); ad=ImageDraw.Draw(atmos)
        ad.ellipse((cx-radius*1.05,cy-radius*1.05,cx+radius*1.05,cy+radius*1.05),outline=(97,181,240,130),width=max(1,int(2*SCALE)))
        atmos=atmos.filter(ImageFilter.GaussianBlur(max(0.8,2.2*SCALE))); image.alpha_composite(atmos)

    def cinematic_title(self, image: Image.Image, eyebrow: str, title: str, subtitle: str = "", alpha: int = 255):
        x=int(OUT_W*0.075)
        self.label(image,eyebrow.upper(),(x,int(OUT_H*0.080)),size=12,alpha=min(alpha,190),anchor="la",bold=True)
        draw_text(image,title,(x,int(OUT_H*0.118)),size=max(15,int(39*SCALE)),fill=(244,247,250,alpha),bold=True,stroke=1,anchor="la")
        if subtitle:
            draw_text(image,subtitle,(x,int(OUT_H*0.168)),size=max(9,int(15*SCALE)),fill=(158,177,195,min(alpha,210)),bold=False,stroke=1,anchor="la")

    def science_hud(self, image: Image.Image):
        # tiny documentary footer rather than a persistent UI panel
        y=int(OUT_H*0.972)
        self.label(image,"NEWTONIAN TWO-BODY MODEL",(int(OUT_W*0.055),y),size=10,alpha=125,anchor="ls",bold=True)
        self.label(image,"VISUAL SCALE EXAGGERATED",(int(OUT_W*0.945),y),size=10,alpha=125,anchor="rs",bold=True)

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        box = (int(OUT_W*0.05), int(OUT_H*0.80), int(OUT_W*0.95), int(OUT_H*0.94))
        self.glass_panel(image, box, alpha=148)
        draw_wrapped_text(
            image,
            text,
            (int(OUT_W*0.085), int(OUT_H*0.842)),
            max_width=int(OUT_W*0.80),
            size=max(13, int(24*SCALE)),
            fill=(242, 246, 250, 248),
            bold=False,
            line_spacing=max(4, int(9*SCALE)),
            anchor="la",
        )

    # ------------------------------------------------------------------
    # Scenes
    # ------------------------------------------------------------------

    def scene_fall(self, image: Image.Image, t: float, local: float):
        self.cinematic_title(image,"ORBITAL MECHANICS","EARTH IS FALLING","It simply keeps missing the Sun.")
        dx,dy,zoom=self.camera(t)
        cx,cy=OUT_W*0.50+dx,OUT_H*0.46+dy
        self.draw_sun(image,cx,cy,62*SCALE*zoom,t)
        orbit_r=305*SCALE*zoom
        angle=-1.15+local*1.75
        pts=[]
        for a in np.linspace(-2.9,2.9,220):
            pts.append((cx+orbit_r*math.cos(a),cy+orbit_r*0.37*math.sin(a)))
        self.soft_line(image,pts,(136,166,189,76),width=1.0,glow=3.0)
        ex=cx+orbit_r*math.cos(angle); ey=cy+orbit_r*0.37*math.sin(angle)
        self.draw_earth(image,ex,ey,18*SCALE*zoom,t)

        d=ImageDraw.Draw(image)
        gx,gy=lerp(ex,cx,0.30),lerp(ey,cy,0.30)
        arrow(d,(ex,ey),(gx,gy),(236,186,117,200),max(1,int(2.2*SCALE)))
        tx,ty=-math.sin(angle),0.37*math.cos(angle); n=max(math.hypot(tx,ty),1e-6); tx/=n; ty/=n
        vx,vy=ex+tx*92*SCALE,ey+ty*92*SCALE
        arrow(d,(ex,ey),(vx,vy),(173,212,236,210),max(1,int(2.2*SCALE)))
        self.label(image,"GRAVITY",(int(gx),int(gy-18*SCALE)),size=11,alpha=160,anchor="ma")
        self.label(image,"VELOCITY",(int(vx),int(vy-18*SCALE)),size=11,alpha=160,anchor="ma")

        self.glass_panel(image, (int(OUT_W*0.055), int(OUT_H*0.67), int(OUT_W*0.57), int(OUT_H*0.78)), alpha=120)
        self.equation(image,"F = G M m / r²",(int(OUT_W*0.085),int(OUT_H*0.708)),size=36,anchor="la")
        self.label(image,"GRAVITY CURVES A STRAIGHT-LINE MOTION INTO AN ORBIT.",(int(OUT_W*0.085),int(OUT_H*0.752)),size=12,alpha=170,anchor="la")

    def scene_circular(self, image: Image.Image, t: float, local: float):
        self.cinematic_title(image,"THE FIRST CALCULATION","HOW FAST MUST EARTH MOVE?","For a circular orbit, gravity supplies the centripetal acceleration.")
        # Equation arrives as a sparse derivation on the left.
        x=int(OUT_W*0.075)
        fade=smoothstep(local*1.8)
        self.glass_panel(image, (int(OUT_W*0.055), int(OUT_H*0.215), int(OUT_W*0.62), int(OUT_H*0.455)), alpha=118)
        self.equation(image,"GMm / r² = mv² / r",(x,int(OUT_H*0.255)),size=29,alpha=int(235*fade))
        self.label(image,"cancel m and one r",(x,int(OUT_H*0.307)),size=12,alpha=int(150*fade),bold=False)
        self.equation(image,"v = √[G(M+m)/r]",(x,int(OUT_H*0.355)),size=34,alpha=int(250*fade),accent=True)
        self.equation(image,"T = 2π √[r³/G(M+m)]",(x,int(OUT_H*0.413)),size=26,alpha=int(220*fade))

        cx,cy=OUT_W*0.55,OUT_H*0.625
        rr=260*SCALE
        pts=[]
        for a in np.linspace(0,2*math.pi,260):
            pts.append((cx+rr*math.cos(a),cy+rr*0.38*math.sin(a)))
        self.soft_line(image,pts,(123,169,201,92),width=1.1,glow=3.0)
        self.draw_sun(image,cx,cy,47*SCALE,t)
        a=2*math.pi*local-math.pi/2
        ex=cx+rr*math.cos(a); ey=cy+rr*0.38*math.sin(a)
        self.draw_earth(image,ex,ey,17*SCALE,t)
        self.label(image,"r = orbital separation",(int(cx),int(cy+rr*0.38+42*SCALE)),size=11,alpha=145,anchor="ma",bold=False)

    def scene_numbers(self, image: Image.Image, t: float, local: float):
        self.cinematic_title(image,"EARTH + SUN","PUT IN THE REAL NUMBERS","Standard SI values. No fitted animation speed.")
        x=int(OUT_W*0.075)
        labels=[
            ("G", "6.67430 × 10⁻¹¹  m³ kg⁻¹ s⁻²"),
            ("M☉", "1.98847 × 10³⁰  kg"),
            ("M⊕", "5.9722 × 10²⁴  kg"),
            ("r", "1 AU = 1.495978707 × 10¹¹  m"),
        ]
        y=0.245
        for i,(k,v) in enumerate(labels):
            a=int(215*smoothstep(local*2.0-i*0.12))
            self.label(image,k,(x,int(OUT_H*y)),size=11,alpha=a,anchor="la")
            draw_text(image,v,(int(OUT_W*0.19),int(OUT_H*y)),size=max(9,int(15*SCALE)),fill=(215,224,233,a),bold=False,stroke=1,anchor="la")
            y+=0.058

        # Result dominates like a cinematic data reveal.
        reveal=smoothstep((local-0.28)/0.45)
        self.label(image,"CIRCULAR ORBIT SPEED AT 1 AU",(x,int(OUT_H*0.545)),size=11,alpha=int(150*reveal),anchor="la")
        draw_text(image,f"{V_CIRC/1000.0:,.2f}",(x,int(OUT_H*0.615)),size=max(24,int(68*SCALE)),fill=(243,246,249,int(255*reveal)),bold=True,stroke=1,anchor="la")
        draw_text(image,"km/s",(int(OUT_W*0.46),int(OUT_H*0.615)),size=max(12,int(22*SCALE)),fill=(150,170,189,int(220*reveal)),bold=False,stroke=1,anchor="lm")
        self.label(image,"ORBITAL PERIOD",(x,int(OUT_H*0.690)),size=11,alpha=int(145*reveal),anchor="la")
        draw_text(image,f"{T_CIRC/DAY:,.2f} days",(x,int(OUT_H*0.735)),size=max(16,int(31*SCALE)),fill=(221,228,235,int(240*reveal)),bold=True,stroke=1,anchor="la")

        # Small Sun/Earth scale cue on right.
        sx,sy=OUT_W*0.79,OUT_H*0.63
        self.draw_sun(image,sx,sy,52*SCALE,t)
        self.draw_earth(image,sx+116*SCALE,sy-46*SCALE,12*SCALE,t)
        self.soft_line(image,[(sx+58*SCALE,sy-5*SCALE),(sx+101*SCALE,sy-34*SCALE)],(154,178,197,75),width=1,glow=2)

    def scene_ellipse(self, image: Image.Image, t: float, local: float):
        self.cinematic_title(image,"REAL ORBITS","DISTANCE CHANGES. SPEED CHANGES.","The vis-viva equation handles an ellipse.")
        cx,cy=OUT_W*0.54,OUT_H*0.47
        a=325*SCALE
        e_real=EARTH_E
        b=a*math.sqrt(1.0-e_real**2)
        focus=e_real*a
        sun_x=cx-focus
        pts=[(cx+a*math.cos(q),cy+b*math.sin(q)) for q in np.linspace(0,2*math.pi,420)]
        self.soft_line(image,pts,(140,176,201,110),width=1.4,glow=4)
        self.draw_sun(image,sun_x,cy,45*SCALE,t)
        theta=2*math.pi*local-math.pi
        ex=cx+a*math.cos(theta); ey=cy+b*math.sin(theta)
        self.draw_earth(image,ex,ey,16*SCALE,t)
        self.label(image,"Because e is only 0.0167, the real ellipse looks almost circular.",(int(OUT_W*0.54), int(OUT_H*0.585)), size=11, alpha=150, anchor="ma", bold=False)

        # velocity trail behind Earth
        trail=[]
        for dq in np.linspace(-0.32,0,44):
            q=theta+dq
            trail.append((cx+a*math.cos(q),cy+b*math.sin(q)))
        self.soft_line(image,trail,(176,215,237,125),width=2.2,glow=8)

        x=int(OUT_W*0.075)
        self.glass_panel(image, (int(OUT_W*0.055), int(OUT_H*0.655), int(OUT_W*0.66), int(OUT_H*0.82)), alpha=118)
        self.equation(image,"v² = G(M+m) [2/r − 1/a]",(x,int(OUT_H*0.690)),size=28,anchor="la")
        self.label(image,f"REAL ECCENTRICITY  e = {EARTH_E:.7f}",(x,int(OUT_H*0.736)),size=12,alpha=185)
        self.label(image,f"PERIHELION   {V_PERI/1000.0:.2f} km/s",(x,int(OUT_H*0.772)),size=12,alpha=190)
        self.label(image,f"APHELION     {V_APH/1000.0:.2f} km/s",(x,int(OUT_H*0.804)),size=12,alpha=190)

    def scene_barycenter(self, image: Image.Image, t: float, local: float):
        self.cinematic_title(image,"TWO HEAVY BODIES","NOTHING IS TRULY FIXED","Both bodies orbit a shared center of mass.")
        center=np.array([OUT_W*0.52,OUT_H*0.48])
        idx=int(local*(len(BINARY_SIM["t"])-1))
        r1=BINARY_SIM["r1"]; r2=BINARY_SIM["r2"]
        scale=300*SCALE/max(np.max(np.linalg.norm(r2,axis=1)),np.max(np.linalg.norm(r1,axis=1)))
        pts1=[tuple(center+p*scale) for p in r1[::5]]
        pts2=[tuple(center+p*scale) for p in r2[::5]]
        self.soft_line(image,pts1,(211,169,113,82),width=1.1,glow=4)
        self.soft_line(image,pts2,(137,186,218,82),width=1.1,glow=4)
        p1=center+r1[idx]*scale; p2=center+r2[idx]*scale
        self.draw_sun(image,float(p1[0]),float(p1[1]),41*SCALE,t)

        # blue-white second star
        glow=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); gd=ImageDraw.Draw(glow); r=31*SCALE
        for mult,a in [(3.3,16),(2.1,30),(1.3,62)]:
            rr=r*mult; gd.ellipse((p2[0]-rr,p2[1]-rr,p2[0]+rr,p2[1]+rr),fill=(145,190,238,a))
        glow=glow.filter(ImageFilter.GaussianBlur(max(4,int(11*SCALE)))); image.alpha_composite(glow)
        d=ImageDraw.Draw(image); d.ellipse((p2[0]-r,p2[1]-r,p2[0]+r,p2[1]+r),fill=(188,216,243,255))

        br=4*SCALE
        d.ellipse((center[0]-br,center[1]-br,center[0]+br,center[1]+br),fill=(240,243,247,225))
        self.label(image,"BARYCENTER",(int(center[0]),int(center[1]-20*SCALE)),size=10,alpha=165,anchor="ma")
        self.equation(image,"m₁r₁ = m₂r₂",(int(OUT_W*0.075),int(OUT_H*0.700)),size=28)
        self.label(image,"THE SAME MECHANICS DESCRIBE BINARY STARS",(int(OUT_W*0.075),int(OUT_H*0.755)),size=11,alpha=155)

    def scene_simulation(self, image: Image.Image, t: float, local: float):
        self.cinematic_title(image,"NUMERICAL ORBIT","CALCULATE. ADVANCE. REPEAT.","The trajectory below is generated by velocity-Verlet integration.")
        x=int(OUT_W*0.075)
        code=[
            "a = gravity(r₁, r₂)",
            "r′ = r + vΔt + ½aΔt²",
            "a′ = gravity(r₁′, r₂′)",
            "v′ = v + ½(a + a′)Δt",
        ]
        self.glass_panel(image, (int(OUT_W*0.055), int(OUT_H*0.19), int(OUT_W*0.50), int(OUT_H*0.42)), alpha=112)
        for i,line in enumerate(code):
            self.equation(image,line,(x,int(OUT_H*(0.225+i*0.045))),size=15,alpha=185 if i in (0,2) else 160)

        center=np.array([OUT_W*0.58,OUT_H*0.61])
        r1=EARTH_SIM["r1"]; r2=EARTH_SIM["r2"]
        scale=270*SCALE/AU
        idx=int(local*(len(EARTH_SIM["t"])-1))
        # full guide in near-dark, simulated history luminous
        full=[tuple(center+p*scale) for p in r2[::4]]
        self.soft_line(image,full,(112,145,168,38),width=1,glow=2)
        hist=[tuple(center+p*scale) for p in r2[:idx+1:3]]
        if len(hist)>1:
            self.soft_line(image,hist,(148,195,224,155),width=2.0,glow=7)
        p1=center+r1[idx]*scale; p2=center+r2[idx]*scale
        self.draw_sun(image,float(p1[0]),float(p1[1]),42*SCALE,t)
        self.draw_earth(image,float(p2[0]),float(p2[1]),16*SCALE,t)

        day=EARTH_SIM["t"][idx]/DAY
        dist=np.linalg.norm(r2[idx]-r1[idx])/AU
        speed=np.linalg.norm(EARTH_SIM["v2"][idx])/1000.0
        self.label(image,"SIMULATION STATE",(x,int(OUT_H*0.695)),size=10,alpha=130)
        draw_text(image,f"DAY {day:05.1f}",(x,int(OUT_H*0.735)),size=max(13,int(24*SCALE)),fill=(232,237,242,235),bold=True,stroke=1,anchor="la")
        self.label(image,f"r = {dist:.4f} AU     v = {speed:.2f} km/s",(x,int(OUT_H*0.780)),size=11,alpha=170)

    def render(self, t: float) -> np.ndarray:
        image=self.background(t)
        shot=get_shot(t)
        local=smoothstep((t-shot["start"])/max(shot["end"]-shot["start"],1e-9))
        if shot["name"]=="fall": self.scene_fall(image,t,local)
        elif shot["name"]=="circular": self.scene_circular(image,t,local)
        elif shot["name"]=="numbers": self.scene_numbers(image,t,local)
        elif shot["name"]=="ellipse": self.scene_ellipse(image,t,local)
        elif shot["name"]=="barycenter": self.scene_barycenter(image,t,local)
        else: self.scene_simulation(image,t,local)
        self.draw_caption(image,t)
        self.science_hud(image)
        return apply_grade(np.asarray(image.convert("RGB")))


# -----------------------------------------------------------------------------
# Production outputs
# -----------------------------------------------------------------------------

def save_summary(path: Path):
    summary = {
        "title": CONFIG["title"],
        "model": "Newtonian two-body gravity",
        "integrator": "velocity-Verlet",
        "constants": {
            "G_m3_kg_s2": G,
            "sun_mass_kg": M_SUN,
            "earth_mass_kg": M_EARTH,
            "AU_m": AU,
            "earth_eccentricity_approx": EARTH_E,
        },
        "calculated_values": {
            "mu_earth_sun_m3_s2": MU_ES,
            "circular_speed_at_1AU_km_s": V_CIRC/1000.0,
            "circular_period_days": T_CIRC/DAY,
            "earth_perihelion_speed_km_s": V_PERI/1000.0,
            "earth_aphelion_speed_km_s": V_APH/1000.0,
            "escape_speed_at_1AU_km_s": ESCAPE_AT_1AU/1000.0,
            "sun_earth_barycenter_offset_from_sun_center_km": SUN_BARY_RADIUS/1000.0,
        },
        "equations": {
            "gravity": "F = G m1 m2 / r^2",
            "circular_orbit_speed": "v = sqrt(G(m1+m2)/r)",
            "orbital_period": "T = 2*pi*sqrt(r^3/[G(m1+m2)])",
            "vis_viva": "v^2 = G(m1+m2)*(2/r - 1/a)",
            "barycenter": "m1*r1 = m2*r2, with r1+r2=a",
        },
        "science_notes": [
            "The Earth-Sun numerical simulation includes motion of both bodies around their barycenter.",
            "Body sizes and orbital distances are not drawn to scale.",
            "The teaching scene uses Earth's real orbital eccentricity, e = 0.0167086, so the ellipse appears only slightly different from a circle.",
            "Newtonian gravity is an excellent approximation for many planetary and stellar two-body systems, but strong-gravity or relativistic cases require general relativity.",
        ],
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def make_previews(scene: OrbitScene):
    times = [0.6, 2.2, 4.3, 6.6, 8.7, 11.2] if QUICK_MODE else [3.0, 12.0, 22.0, 33.0, 44.0, 55.0]
    for t in times:
        arr=scene.render(t)
        Image.fromarray(arr).save(PREVIEW_DIR / f"preview_{t:g}s.png")


def render_video(scene: OrbitScene, path: Path):
    fps=int(CONFIG["fps"])
    duration=float(CONFIG["duration_s"])
    frames=int(round(fps*duration))
    writer=iio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8 if QUICK_MODE else 9,
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags","+faststart","-crf","15","-preset","slow"],
    )
    try:
        for i in tqdm(range(frames), desc="Rendering orbit Short"):
            t=i/fps
            writer.append_data(scene.render(t))
    finally:
        writer.close()


def main():
    scene=OrbitScene()
    basename=str(CONFIG["output_basename"])
    mp4_path=OUTPUT_ROOT / f"{basename}_final.mp4"
    srt_path=OUTPUT_ROOT / f"{basename}.srt"
    csv_path=DATA_DIR / "earth_sun_velocity_verlet_orbit.csv"
    json_path=DATA_DIR / "orbit_calculation_summary.json"

    write_srt(CAPTIONS,srt_path)
    save_orbit_csv(csv_path)
    save_summary(json_path)
    make_previews(scene)
    render_video(scene,mp4_path)

    print("\nRender complete")
    print(f"Video:    {mp4_path}")
    print(f"Subtitles:{srt_path}")
    print(f"Orbit CSV:{csv_path}")
    print(f"Summary:  {json_path}")


if __name__ == "__main__":
    main()
