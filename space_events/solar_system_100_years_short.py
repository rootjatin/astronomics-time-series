# %% [markdown]
# # Watch the Solar System Move for 100 Years — JPL Data-Driven YouTube Short
#
# This script creates a vertical 1080×1920 cinematic astronomy short showing
# heliocentric planetary motion from 1950-01-01 through 2050-01-01.
#
# DATA SOURCE
# -----------
# NASA/JPL Solar System Dynamics, "Approximate Positions of the Planets":
# https://ssd.jpl.nasa.gov/planets/approx_pos.html
#
# The source publishes Keplerian elements and century-rates fitted for the
# interval 1800 AD through 2050 AD. This renderer uses the published Table 1
# coefficients exactly and follows JPL's stated equations:
#   1) update a, e, I, L, longitude of perihelion, and longitude of node,
#   2) form mean anomaly,
#   3) solve Kepler's equation iteratively,
#   4) compute orbital-plane x'/y',
#   5) rotate into J2000 ecliptic heliocentric x/y/z coordinates.
#
# The chosen 1950→2050 window stays inside the source table's stated validity
# interval. JPL describes these formulae as LOWER-ACCURACY approximations; for
# high-precision work, JPL recommends Horizons. This video is a century-scale
# educational visualization, not a spacecraft-navigation product.
#
# IMPORTANT DISPLAY NOTES
# -----------------------
# - Planet coordinates and orbital geometry are data-driven from the JPL table.
# - Earth is represented by the Earth-Moon barycenter because that is the body
#   supplied in JPL Table 1.
# - Planet marker radii are intentionally enlarged; physical planet diameters are
#   not rendered to scale.
# - The camera zoom changes between shots, but coordinates remain in AU and the
#   scene projection remains linear inside each shot.
# - Stars, nebula haze, glints, HUD graphics, bloom, and camera drift are cinematic.
#
# Recommended install:
#
#     pip install numpy pandas matplotlib pillow imageio imageio-ffmpeg tqdm
#
# Quick test render:
#
#     CONFIG["fps"] = 12
#     CONFIG["duration_s"] = 14
#     CONFIG["video_width"] = 540
#     CONFIG["video_height"] = 960

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# %% [markdown]
# ## Configuration

OUTPUT_ROOT = Path("solar_system_100_years_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"

for directory in [OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CONFIG = {
    # Delivery
    "video_width": 1080,
    "video_height": 1920,
    "fps": 24,
    "duration_s": 58,
    "output_basename": "watch_solar_system_move_100_years",

    # Century playback: entirely inside JPL Table 1 validity (1800–2050)
    "start_date": "1950-01-01",
    "end_date": "2050-01-01",

    # Motion / path sampling
    "orbit_sample_count": 6200,
    "position_archive_step_days": 30,

    # Scene
    "background_particle_count": 540,
    "hud_noise_count": 72,
    "vignette_strength": 0.27,
    "contrast_boost": 1.13,
    "saturation_boost": 1.08,

    # Titles
    "title_text": "WATCH THE SOLAR SYSTEM MOVE",
    "title_line_2": "FOR 100 YEARS",
    "subtitle_text": "1950 → 2050 // NASA/JPL orbital elements",
    "credit_text": "Orbital elements + rates: NASA/JPL Solar System Dynamics",
    "scientific_note": (
        "JPL lower-accuracy planetary-position formulae, Table 1 (1800–2050). "
        "Earth marker = Earth-Moon barycenter. Planet sizes enlarged for visibility."
    ),

    # Optional finishing
    "audio_path": None,
    "burn_subtitles": False,
    "write_subtitle_sidecar": True,
}

OUT_SIZE = (CONFIG["video_width"], CONFIG["video_height"])


# %% [markdown]
# ## JPL Table 1: Keplerian elements and rates, valid 1800 AD–2050 AD
#
# Column order:
#   a [AU], e, I [deg], L [deg], long.peri [deg], long.node [deg]
# Rates are per Julian century.

@dataclass(frozen=True)
class PlanetModel:
    name: str
    label: str
    elements: Tuple[float, float, float, float, float, float]
    rates: Tuple[float, float, float, float, float, float]
    color: Tuple[int, int, int]
    marker_radius: float


PLANETS: Tuple[PlanetModel, ...] = (
    PlanetModel(
        "Mercury", "MERCURY",
        (0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593),
        (0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081),
        (190, 198, 207), 5.0,
    ),
    PlanetModel(
        "Venus", "VENUS",
        (0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255),
        (0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418),
        (255, 190, 104), 6.5,
    ),
    PlanetModel(
        "Earth", "EARTH / MOON BARYCENTER",
        (1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0),
        (0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0),
        (87, 191, 255), 7.0,
    ),
    PlanetModel(
        "Mars", "MARS",
        (1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891),
        (0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343),
        (255, 100, 72), 6.5,
    ),
    PlanetModel(
        "Jupiter", "JUPITER",
        (5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909),
        (-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106),
        (232, 189, 145), 10.5,
    ),
    PlanetModel(
        "Saturn", "SATURN",
        (9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448),
        (-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794),
        (246, 218, 145), 9.5,
    ),
    PlanetModel(
        "Uranus", "URANUS",
        (19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503),
        (-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589),
        (130, 232, 242), 8.5,
    ),
    PlanetModel(
        "Neptune", "NEPTUNE",
        (30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574),
        (0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664),
        (93, 128, 255), 8.5,
    ),
)

PLANET_BY_NAME = {planet.name: planet for planet in PLANETS}


# %% [markdown]
# ## Numeric helpers and JPL position equations

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def ease_in_out_sine(t: float) -> float:
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


def parse_utc(date_text: str) -> datetime:
    return datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def datetime_to_jd(dt: datetime) -> float:
    """Calendar timestamp to Julian date using the Unix epoch relation.

    JPL's approximate formulae specify JDTDB. For this century-scale visual,
    UTC-labeled midnight is used as the calendar sampling label; the small
    UTC↔TDB offset is negligible at the spatial and temporal resolution shown.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / 86400.0 + 2440587.5


def jd_to_datetime(jd: float) -> datetime:
    return datetime.fromtimestamp((jd - 2440587.5) * 86400.0, tz=timezone.utc)


def solve_kepler_degrees(mean_anomaly_deg: np.ndarray, eccentricity: np.ndarray) -> np.ndarray:
    """Solve JPL's degree-form Kepler equation to 1e-10 degrees."""
    M = np.asarray(mean_anomaly_deg, dtype=np.float64)
    e = np.asarray(eccentricity, dtype=np.float64)
    e_star = np.degrees(1.0) * e

    E = M + e_star * np.sin(np.radians(M))

    for _ in range(18):
        delta_M = M - (E - e_star * np.sin(np.radians(E)))
        delta_E = delta_M / (1.0 - e * np.cos(np.radians(E)))
        E = E + delta_E
        if float(np.nanmax(np.abs(delta_E))) <= 1e-10:
            break

    return E


def jpl_heliocentric_xyz(planet: PlanetModel, jd: np.ndarray | float) -> np.ndarray:
    """Return J2000-ecliptic heliocentric x/y/z in AU.

    Implements the equations published by NASA/JPL Solar System Dynamics for
    Table 1 approximate planetary positions, valid 1800 AD through 2050 AD.
    """
    jd_arr = np.atleast_1d(np.asarray(jd, dtype=np.float64))
    T = (jd_arr - 2451545.0) / 36525.0

    base = np.asarray(planet.elements, dtype=np.float64)
    rates = np.asarray(planet.rates, dtype=np.float64)
    values = base[None, :] + T[:, None] * rates[None, :]

    a = values[:, 0]
    e = values[:, 1]
    I = values[:, 2]
    L = values[:, 3]
    long_peri = values[:, 4]
    long_node = values[:, 5]

    omega = long_peri - long_node
    M = L - long_peri
    M = (M + 180.0) % 360.0 - 180.0
    E = solve_kepler_degrees(M, e)

    E_rad = np.radians(E)
    x_prime = a * (np.cos(E_rad) - e)
    y_prime = a * np.sqrt(np.maximum(0.0, 1.0 - e * e)) * np.sin(E_rad)

    omega_r = np.radians(omega)
    node_r = np.radians(long_node)
    inc_r = np.radians(I)

    cos_w = np.cos(omega_r)
    sin_w = np.sin(omega_r)
    cos_O = np.cos(node_r)
    sin_O = np.sin(node_r)
    cos_I = np.cos(inc_r)
    sin_I = np.sin(inc_r)

    x = (
        (cos_w * cos_O - sin_w * sin_O * cos_I) * x_prime
        + (-sin_w * cos_O - cos_w * sin_O * cos_I) * y_prime
    )
    y = (
        (cos_w * sin_O + sin_w * cos_O * cos_I) * x_prime
        + (-sin_w * sin_O + cos_w * cos_O * cos_I) * y_prime
    )
    z = (sin_w * sin_I) * x_prime + (cos_w * sin_I) * y_prime

    xyz = np.column_stack([x, y, z])
    if np.ndim(jd) == 0:
        return xyz[0]
    return xyz


def find_ffmpeg() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# %% [markdown]
# ## Text and image helpers

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
        stroke_fill=(0, 0, 0, min(fill[3] if len(fill) > 3 else 255, 225)),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int = 31,
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
            (x, y), line, font=font, fill=fill,
            stroke_width=2, stroke_fill=(0, 0, 0, 220),
        )
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += (bbox[3] - bbox[1]) + line_spacing


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cx = width / 2.0
    cy = height / 2.0
    nx = (xx - cx) / (width / 2.0)
    ny = (yy - cy) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.85, 0.0, 1.0).astype(np.float32)


def apply_grade(arr: np.ndarray) -> np.ndarray:
    image = Image.fromarray(arr)
    image = ImageEnhance.Contrast(image).enhance(CONFIG["contrast_boost"])
    image = ImageEnhance.Color(image).enhance(CONFIG["saturation_boost"])
    return np.asarray(image)


VIGNETTE = make_vignette(OUT_SIZE[0], OUT_SIZE[1], CONFIG["vignette_strength"])


# %% [markdown]
# ## Cinematic timeline

CAPTIONS = [
    (0.5, 6.0, "Eight planets. One century. The positions come from NASA/JPL orbital elements."),
    (6.4, 17.5, "The inner worlds redraw their orbits again and again while the year counter accelerates."),
    (18.0, 30.5, "Pull back: Jupiter and Saturn turn slowly while the rocky planets blur around the Sun."),
    (31.0, 44.5, "Farther out, a century is not enough for Neptune to complete one full revolution."),
    (45.0, 51.8, "The glowing ghosts mark the planets at the start of 1950."),
    (52.0, 57.2, "1950 to 2050. Same Sun. A completely different planetary arrangement."),
]

SHOT_PLAN = [
    {
        "name": "intro",
        "start": 0.0,
        "end": 6.0,
        "view_start": 2.2,
        "view_end": 2.2,
        "tilt_start": 67.0,
        "tilt_end": 63.0,
        "yaw_start": -15.0,
        "yaw_end": -4.0,
        "trail_years": 1.2,
        "caption": "JPL ELEMENTS // HELIOCENTRIC J2000 ECLIPTIC",
    },
    {
        "name": "inner",
        "start": 6.0,
        "end": 18.0,
        "view_start": 2.15,
        "view_end": 2.55,
        "tilt_start": 63.0,
        "tilt_end": 58.0,
        "yaw_start": -4.0,
        "yaw_end": 15.0,
        "trail_years": 1.4,
        "caption": "INNER SYSTEM // TRUE AU COORDINATES",
    },
    {
        "name": "giants",
        "start": 18.0,
        "end": 31.0,
        "view_start": 2.55,
        "view_end": 11.5,
        "tilt_start": 58.0,
        "tilt_end": 66.0,
        "yaw_start": 15.0,
        "yaw_end": 42.0,
        "trail_years": 14.0,
        "caption": "GAS GIANTS // CAMERA PULLBACK",
    },
    {
        "name": "outer",
        "start": 31.0,
        "end": 45.0,
        "view_start": 11.5,
        "view_end": 33.5,
        "tilt_start": 66.0,
        "tilt_end": 72.0,
        "yaw_start": 42.0,
        "yaw_end": 78.0,
        "trail_years": 55.0,
        "caption": "OUTER SYSTEM // ONE CENTURY IN MOTION",
    },
    {
        "name": "compare",
        "start": 45.0,
        "end": 52.0,
        "view_start": 33.5,
        "view_end": 33.5,
        "tilt_start": 72.0,
        "tilt_end": 64.0,
        "yaw_start": 78.0,
        "yaw_end": 102.0,
        "trail_years": 100.0,
        "caption": "1950 GHOST POSITIONS // 2050 LIVE SYSTEM",
    },
    {
        "name": "outro",
        "start": 52.0,
        "end": CONFIG["duration_s"],
        "view_start": 33.5,
        "view_end": 36.0,
        "tilt_start": 64.0,
        "tilt_end": 70.0,
        "yaw_start": 102.0,
        "yaw_end": 124.0,
        "trail_years": 100.0,
        "caption": "100 YEARS // SOLAR SYSTEM GEOMETRY CHANGED",
    },
]


def get_shot(t: float) -> Dict:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def shot_state(t: float) -> Tuple[Dict, float, float, float]:
    shot = get_shot(t)
    duration = max(shot["end"] - shot["start"], 1e-6)
    u = clamp((t - shot["start"]) / duration)
    e = ease_in_out_sine(u)
    view = lerp(shot["view_start"], shot["view_end"], e)
    tilt = lerp(shot["tilt_start"], shot["tilt_end"], e)
    yaw = lerp(shot["yaw_start"], shot["yaw_end"], e)
    return shot, view, tilt, yaw


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


# %% [markdown]
# ## Solar System renderer

class SolarSystemCenturyScene:
    def __init__(self):
        self.start_dt = parse_utc(CONFIG["start_date"])
        self.end_dt = parse_utc(CONFIG["end_date"])
        self.start_jd = datetime_to_jd(self.start_dt)
        self.end_jd = datetime_to_jd(self.end_dt)
        self.timeline_days = self.end_jd - self.start_jd

        if self.timeline_days <= 0:
            raise RuntimeError("The century playback range is invalid.")

        self.path_jd = np.linspace(
            self.start_jd,
            self.end_jd,
            int(CONFIG["orbit_sample_count"]),
            dtype=np.float64,
        )

        self.paths: Dict[str, np.ndarray] = {
            planet.name: jpl_heliocentric_xyz(planet, self.path_jd)
            for planet in PLANETS
        }
        self.start_positions = {
            planet.name: jpl_heliocentric_xyz(planet, self.start_jd)
            for planet in PLANETS
        }
        self.end_positions = {
            planet.name: jpl_heliocentric_xyz(planet, self.end_jd)
            for planet in PLANETS
        }

        self.particles = self._make_particles(CONFIG["background_particle_count"], seed=73)
        self.hud_noise = self._make_hud_noise(CONFIG["hud_noise_count"], seed=133)

    @staticmethod
    def _make_particles(count: int, seed: int):
        rng = np.random.default_rng(seed)
        particles = []
        for _ in range(count):
            particles.append({
                "x": float(rng.uniform(0, OUT_SIZE[0])),
                "y": float(rng.uniform(0, OUT_SIZE[1])),
                "r": float(rng.uniform(0.4, 2.1)),
                "a": int(rng.integers(24, 140)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "drift": float(rng.uniform(-15, 15)),
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
                "length": float(rng.uniform(8, 100)),
                "alpha": int(rng.integers(10, 48)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            })
        return noise

    def century_fraction(self, t: float) -> float:
        playback_start = 2.3
        playback_end = CONFIG["duration_s"] - 3.1
        return smoothstep((t - playback_start) / max(playback_end - playback_start, 1e-6))

    def current_jd(self, t: float) -> float:
        return self.start_jd + self.century_fraction(t) * self.timeline_days

    def current_datetime(self, t: float) -> datetime:
        return jd_to_datetime(self.current_jd(t))

    def current_positions(self, t: float) -> Dict[str, np.ndarray]:
        jd = self.current_jd(t)
        return {planet.name: jpl_heliocentric_xyz(planet, jd) for planet in PLANETS}

    def camera_project(
        self,
        xyz: np.ndarray,
        view_radius_au: float,
        tilt_deg: float,
        yaw_deg: float,
        center: Tuple[float, float] = (540.0, 920.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        points = np.asarray(xyz, dtype=np.float64)
        single = points.ndim == 1
        points = np.atleast_2d(points)

        yaw = math.radians(yaw_deg)
        tilt = math.radians(tilt_deg)

        cy = math.cos(yaw)
        sy = math.sin(yaw)
        ct = math.cos(tilt)
        st = math.sin(tilt)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        x1 = cy * x - sy * y
        y1 = sy * x + cy * y
        z1 = z

        y2 = ct * y1 - st * z1
        depth = st * y1 + ct * z1

        usable_radius = min(OUT_SIZE[0] * 0.44, OUT_SIZE[1] * 0.38)
        px_per_au = usable_radius / max(view_radius_au, 0.2)

        sx = center[0] + x1 * px_per_au
        sy_screen = center[1] + y2 * px_per_au
        projected = np.column_stack([sx, sy_screen])

        if single:
            return projected[0], np.asarray(depth[0])
        return projected, depth

    def render_background(self, t: float) -> Image.Image:
        canvas = Image.new("RGBA", OUT_SIZE, (1, 3, 10, 255))
        draw = ImageDraw.Draw(canvas)

        for particle in self.particles:
            x = (particle["x"] + particle["drift"] * 0.035 * t) % OUT_SIZE[0]
            y = (particle["y"] + particle["drift"] * 0.010 * t) % OUT_SIZE[1]
            twinkle = 0.70 + 0.30 * math.sin(t * 1.2 + particle["phase"])
            alpha = int(particle["a"] * twinkle)
            radius = particle["r"]
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(205, 226, 255, alpha),
            )

        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        haze_specs = [
            (180, 530, 520, (32, 48, 118, 27)),
            (830, 780, 610, (10, 118, 140, 23)),
            (610, 1420, 540, (92, 30, 105, 20)),
        ]
        for cx, cy, radius, color in haze_specs:
            hd.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
        haze = haze.filter(ImageFilter.GaussianBlur(105))
        canvas.alpha_composite(haze)
        return canvas

    def draw_sun(self, canvas: Image.Image, t: float):
        cx, cy = 540, 920
        pulse = 1.0 + 0.035 * math.sin(t * 2.0)
        core_radius = 11.5 * pulse

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for radius, alpha in [(120, 15), (75, 25), (42, 48), (24, 85)]:
            gd.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=(255, 180, 60, alpha),
            )
        glow = glow.filter(ImageFilter.GaussianBlur(24))
        canvas.alpha_composite(glow)

        d = ImageDraw.Draw(canvas)
        d.ellipse(
            (cx - core_radius, cy - core_radius, cx + core_radius, cy + core_radius),
            fill=(255, 244, 190, 255),
            outline=(255, 210, 90, 255),
            width=2,
        )
        for ray_index in range(12):
            angle = t * 0.25 + ray_index * math.pi / 6.0
            r0 = 17 + 3 * math.sin(t * 1.7 + ray_index)
            r1 = 26 + 6 * math.sin(t * 1.2 + ray_index * 0.7)
            d.line(
                (
                    cx + math.cos(angle) * r0,
                    cy + math.sin(angle) * r0,
                    cx + math.cos(angle) * r1,
                    cy + math.sin(angle) * r1,
                ),
                fill=(255, 205, 90, 145),
                width=2,
            )

    def draw_orbit_paths(
        self,
        canvas: Image.Image,
        t: float,
        shot: Dict,
        view_radius: float,
        tilt: float,
        yaw: float,
    ):
        current_fraction = self.century_fraction(t)
        trail_fraction = min(1.0, float(shot["trail_years"]) / 100.0)
        end_idx = int(round(current_fraction * (len(self.path_jd) - 1)))
        start_idx = max(0, int(round((current_fraction - trail_fraction) * (len(self.path_jd) - 1))))

        sharp = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sharp)

        for planet in PLANETS:
            path = self.paths[planet.name]
            segment = path[start_idx : end_idx + 1]
            if len(segment) < 2:
                continue

            projected, _ = self.camera_project(segment, view_radius, tilt, yaw)
            mask = (
                (projected[:, 0] > -120)
                & (projected[:, 0] < OUT_SIZE[0] + 120)
                & (projected[:, 1] > -120)
                & (projected[:, 1] < OUT_SIZE[1] + 120)
            )

            # Split visible runs to avoid long jumps through clipped regions.
            run: List[Tuple[float, float]] = []
            color = planet.color
            alpha = 90 if shot["name"] in {"intro", "inner"} else 70

            for point, visible in zip(projected, mask):
                if visible:
                    run.append((float(point[0]), float(point[1])))
                else:
                    if len(run) >= 2:
                        sd.line(run, fill=(*color, alpha), width=2)
                    run = []
            if len(run) >= 2:
                sd.line(run, fill=(*color, alpha), width=2)

        canvas.alpha_composite(sharp)

    def draw_start_ghosts(
        self,
        canvas: Image.Image,
        t: float,
        view_radius: float,
        tilt: float,
        yaw: float,
    ):
        alpha = int(205 * smoothstep((t - 43.8) / 2.2))
        if alpha <= 3:
            return

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for planet in PLANETS:
            screen, _ = self.camera_project(
                self.start_positions[planet.name], view_radius, tilt, yaw
            )
            x, y = float(screen[0]), float(screen[1])
            if not (-50 <= x <= OUT_SIZE[0] + 50 and -50 <= y <= OUT_SIZE[1] + 50):
                continue
            r = planet.marker_radius + 7
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                outline=(*planet.color, alpha),
                width=2,
            )
            draw.line((x - r - 7, y, x - r + 2, y), fill=(*planet.color, alpha), width=1)
            draw.line((x + r - 2, y, x + r + 7, y), fill=(*planet.color, alpha), width=1)
        canvas.alpha_composite(layer)

    def draw_planets(
        self,
        canvas: Image.Image,
        t: float,
        shot: Dict,
        view_radius: float,
        tilt: float,
        yaw: float,
    ):
        positions = self.current_positions(t)
        screen_records = []

        for planet in PLANETS:
            screen, depth = self.camera_project(positions[planet.name], view_radius, tilt, yaw)
            screen_records.append((float(depth), planet, screen, positions[planet.name]))

        # Far-to-near draw order.
        screen_records.sort(key=lambda row: row[0])

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        sharp = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sharp)

        for _, planet, screen, xyz in screen_records:
            x, y = float(screen[0]), float(screen[1])
            if not (-100 <= x <= OUT_SIZE[0] + 100 and -100 <= y <= OUT_SIZE[1] + 100):
                continue

            # Marker sizes are enlarged intentionally for visibility.
            shot_scale = 1.16 if shot["name"] in {"intro", "inner"} else 1.0
            radius = planet.marker_radius * shot_scale
            pulse = 1.0 + 0.08 * math.sin(t * 3.0 + len(planet.name))
            radius *= pulse

            # Direction-of-motion streak: 9 calendar days earlier.
            prev_xyz = jpl_heliocentric_xyz(planet, self.current_jd(t) - 9.0)
            prev_screen, _ = self.camera_project(prev_xyz, view_radius, tilt, yaw)
            motion = screen - prev_screen
            motion_len = float(np.linalg.norm(motion))
            if motion_len > 0.01:
                direction = motion / motion_len
                streak = min(62.0, 8.0 + motion_len * 3.3)
                tail = screen - direction * streak
                gd.line(
                    (float(tail[0]), float(tail[1]), x, y),
                    fill=(*planet.color, 72), width=max(4, int(radius * 1.2)),
                )
                sd.line(
                    (float(tail[0]), float(tail[1]), x, y),
                    fill=(*planet.color, 155), width=max(1, int(radius * 0.34)),
                )

            for glow_r, alpha in [(radius * 4.0, 28), (radius * 2.5, 55)]:
                gd.ellipse(
                    (x - glow_r, y - glow_r, x + glow_r, y + glow_r),
                    fill=(*planet.color, alpha),
                )

            sd.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(*planet.color, 255),
                outline=(245, 250, 255, 225),
                width=1,
            )
            sd.ellipse(
                (x - radius * 0.34, y - radius * 0.34, x + radius * 0.10, y + radius * 0.10),
                fill=(255, 255, 255, 155),
            )

            if planet.name == "Saturn":
                ring_w = radius * 2.15
                ring_h = radius * 0.62
                sd.ellipse(
                    (x - ring_w, y - ring_h, x + ring_w, y + ring_h),
                    outline=(245, 220, 160, 210), width=2,
                )

        glow = glow.filter(ImageFilter.GaussianBlur(11))
        canvas.alpha_composite(glow)
        canvas.alpha_composite(sharp)

        self.draw_labels(canvas, t, shot, screen_records)

    def draw_labels(self, canvas: Image.Image, t: float, shot: Dict, screen_records):
        # Keep label density readable as the camera pulls back.
        if shot["name"] in {"intro", "inner"}:
            allowed = {"Mercury", "Venus", "Earth", "Mars"}
        elif shot["name"] == "giants":
            allowed = {"Earth", "Mars", "Jupiter", "Saturn"}
        elif shot["name"] == "outer":
            allowed = {"Jupiter", "Saturn", "Uranus", "Neptune"}
        else:
            allowed = {"Earth", "Jupiter", "Saturn", "Uranus", "Neptune"}

        alpha = 215
        for _, planet, screen, xyz in screen_records:
            if planet.name not in allowed:
                continue
            x, y = float(screen[0]), float(screen[1])
            if not (35 <= x <= OUT_SIZE[0] - 35 and 180 <= y <= OUT_SIZE[1] - 370):
                continue

            r_au = float(np.linalg.norm(xyz))
            label = planet.name.upper()
            if planet.name == "Earth":
                label = "EARTH*"

            draw_text(
                canvas, label, (int(x + 15), int(y - 10)),
                size=17 if shot["name"] in {"outer", "compare", "outro"} else 19,
                fill=(*planet.color, alpha), bold=True, stroke=1,
            )
            if shot["name"] in {"outer", "compare", "outro"}:
                draw_text(
                    canvas, f"{r_au:.2f} AU", (int(x + 15), int(y + 13)),
                    size=14, fill=(175, 205, 225, 175), stroke=1,
                )

    def draw_planet_speed_board(self, canvas: Image.Image, t: float, shot: Dict):
        alpha = int(
            225
            * smoothstep((t - 7.2) / 2.5)
            * (1.0 - smoothstep((t - 29.2) / 2.8))
        )
        if alpha <= 3:
            return

        jd = self.current_jd(t)
        selected = [PLANET_BY_NAME[n] for n in ["Mercury", "Earth", "Mars", "Jupiter", "Saturn"]]
        x0, y0 = 55, 250
        width = 485
        row_h = 51

        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rounded_rectangle(
            (x0 - 18, y0 - 52, x0 + width, y0 + row_h * len(selected) + 26),
            radius=22, fill=(2, 6, 16, 170), outline=(75, 190, 225, 85), width=1,
        )
        canvas.alpha_composite(panel)
        draw_text(
            canvas, "HELIOCENTRIC MOTION // 9-DAY SCREEN VECTOR",
            (x0, y0 - 37), size=18, fill=(155, 220, 240, alpha), bold=True, stroke=1,
        )

        speeds = []
        for planet in selected:
            p0 = jpl_heliocentric_xyz(planet, jd - 0.5)
            p1 = jpl_heliocentric_xyz(planet, jd + 0.5)
            au_per_day = float(np.linalg.norm(p1 - p0))
            km_s = au_per_day * 149_597_870.7 / 86400.0
            speeds.append(km_s)

        max_speed = max(speeds)
        for index, (planet, speed) in enumerate(zip(selected, speeds)):
            y = y0 + index * row_h
            draw_text(
                canvas, planet.name.upper(), (x0, y + 7),
                size=18, fill=(*planet.color, alpha), bold=True, stroke=1,
            )
            bar_x = x0 + 132
            bar_w = 220 * speed / max_speed
            d = ImageDraw.Draw(canvas)
            d.rounded_rectangle(
                (bar_x, y + 15, bar_x + 220, y + 26),
                radius=5, fill=(35, 70, 92, int(alpha * 0.58)),
            )
            d.rounded_rectangle(
                (bar_x, y + 15, bar_x + bar_w, y + 26),
                radius=5, fill=(*planet.color, int(alpha * 0.88)),
            )
            draw_text(
                canvas, f"{speed:4.1f} km/s", (x0 + 465, y + 7),
                size=17, fill=(220, 235, 245, alpha), bold=True, stroke=1, anchor="ra",
            )

    def draw_outer_completion_panel(self, canvas: Image.Image, t: float):
        alpha = int(
            220
            * smoothstep((t - 31.8) / 3.0)
            * (1.0 - smoothstep((t - 46.5) / 3.0))
        )
        if alpha <= 3:
            return

        x0, y0 = 55, 240
        width = 515
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rounded_rectangle(
            (x0 - 18, y0 - 50, x0 + width, y0 + 238),
            radius=22, fill=(2, 6, 16, 170), outline=(100, 165, 255, 85), width=1,
        )
        canvas.alpha_composite(panel)

        draw_text(
            canvas, "CENTURY ORBIT COVERAGE",
            (x0, y0 - 35), size=20, fill=(175, 220, 250, alpha), bold=True, stroke=1,
        )

        # Estimate orbital revolutions from unwrapped mean longitude change over 100 years.
        rows = [PLANET_BY_NAME[n] for n in ["Jupiter", "Saturn", "Uranus", "Neptune"]]
        for idx, planet in enumerate(rows):
            revolutions = abs(planet.rates[3]) / 360.0
            y = y0 + idx * 54
            draw_text(
                canvas, planet.name.upper(), (x0, y + 5),
                size=19, fill=(*planet.color, alpha), bold=True, stroke=1,
            )
            d = ImageDraw.Draw(canvas)
            bar_x = x0 + 137
            track_w = 245
            d.rounded_rectangle(
                (bar_x, y + 14, bar_x + track_w, y + 28),
                radius=6, fill=(35, 65, 90, int(alpha * 0.55)),
            )
            fill_fraction = min(1.0, revolutions)
            d.rounded_rectangle(
                (bar_x, y + 14, bar_x + track_w * fill_fraction, y + 28),
                radius=6, fill=(*planet.color, int(alpha * 0.9)),
            )
            draw_text(
                canvas, f"{revolutions:.2f} turns", (x0 + 495, y + 5),
                size=17, fill=(220, 235, 245, alpha), bold=True, stroke=1, anchor="ra",
            )

    def draw_timeline(self, canvas: Image.Image, t: float):
        x0 = 75
        x1 = OUT_SIZE[0] - 75
        y = OUT_SIZE[1] - 320
        width = x1 - x0
        fraction = self.century_fraction(t)
        cursor_x = x0 + fraction * width
        current_dt = self.current_datetime(t)
        years_elapsed = fraction * 100.0

        d = ImageDraw.Draw(canvas)
        d.line((x0, y, x1, y), fill=(85, 190, 225, 225), width=2)

        for year in range(1950, 2051, 10):
            tick_fraction = (year - 1950) / 100.0
            x = x0 + tick_fraction * width
            tick_h = 22 if year % 25 == 0 else 13
            d.line((x, y - tick_h, x, y + tick_h), fill=(90, 200, 230, 125), width=1)
            if year in {1950, 1975, 2000, 2025, 2050}:
                draw_text(
                    canvas, str(year), (int(x), y + 33), size=15,
                    fill=(165, 205, 225, 195), stroke=1, anchor="ma",
                )

        d.line((cursor_x, y - 40, cursor_x, y + 40), fill=(255, 180, 82, 255), width=3)
        d.ellipse((cursor_x - 6, y - 6, cursor_x + 6, y + 6), fill=(255, 235, 190, 255))

        draw_text(
            canvas, "100-YEAR PLANETARY PLAYBACK", (x0, y - 70),
            size=20, fill=(165, 220, 240, 225), bold=True, stroke=1,
        )
        draw_text(
            canvas, current_dt.strftime("%Y-%m-%d"), (x1, y - 72),
            size=31, fill=(245, 250, 255, 250), bold=True, stroke=2, anchor="ra",
        )
        draw_text(
            canvas, f"+{years_elapsed:05.1f} YEARS", (x1, y + 48),
            size=18, fill=(255, 185, 90, 220), bold=True, stroke=1, anchor="ra",
        )

    def draw_corner_hud(self, canvas: Image.Image, t: float, shot: Dict, view_radius: float):
        if t < 5.4:
            return
        draw_text(
            canvas, f"VIEW // ±{view_radius:04.1f} AU", (OUT_SIZE[0] - 55, 82),
            size=20, fill=(105, 225, 245, 220), bold=True, stroke=1, anchor="ra",
        )
        draw_text(
            canvas, "FRAME // J2000 ECLIPTIC", (OUT_SIZE[0] - 55, 116),
            size=18, fill=(150, 205, 225, 205), bold=True, stroke=1, anchor="ra",
        )
        draw_text(
            canvas, f"SHOT // {shot['name'].upper()}", (OUT_SIZE[0] - 55, 148),
            size=17, fill=(150, 205, 225, 185), stroke=1, anchor="ra",
        )

    def add_text_layers(self, canvas: Image.Image, t: float, shot: Dict):
        title_alpha = int(
            255 * smoothstep((t - 0.18) / 0.9)
            * (1.0 - smoothstep((t - 5.35) / 0.75))
        )
        if title_alpha > 3:
            draw_text(
                canvas, CONFIG["title_text"], (54, 100),
                size=44, fill=(245, 250, 255, title_alpha), bold=True,
            )
            draw_text(
                canvas, CONFIG["title_line_2"], (54, 158),
                size=61, fill=(255, 190, 82, title_alpha), bold=True,
            )
            draw_text(
                canvas, CONFIG["subtitle_text"], (58, 236),
                size=23, fill=(105, 225, 245, min(title_alpha, 230)), bold=True,
            )

        if t > 5.4:
            draw_text(
                canvas, shot["caption"], (54, 65),
                size=20, fill=(140, 215, 235, 205), bold=True, stroke=1,
            )

        caption = caption_at(t)
        if caption:
            panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            pd = ImageDraw.Draw(panel)
            y0 = OUT_SIZE[1] - 244
            pd.rounded_rectangle(
                (45, y0, OUT_SIZE[0] - 45, y0 + 126),
                radius=24, fill=(1, 4, 12, 172), outline=(55, 180, 220, 72), width=1,
            )
            canvas.alpha_composite(panel)
            draw_wrapped_text(
                canvas, caption, (70, y0 + 27),
                max_width=OUT_SIZE[0] - 140, size=29, fill=(245, 250, 255, 245),
            )

        note_alpha = int(220 * smoothstep((t - 51.0) / 2.8))
        if note_alpha > 3:
            draw_wrapped_text(
                canvas, CONFIG["credit_text"], (65, OUT_SIZE[1] - 111),
                max_width=940, size=18, fill=(220, 232, 245, note_alpha),
            )
            draw_wrapped_text(
                canvas, CONFIG["scientific_note"], (65, OUT_SIZE[1] - 79),
                max_width=940, size=16, fill=(190, 210, 232, note_alpha),
            )

    def draw_hud_noise(self, canvas: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for item in self.hud_noise:
            pulse = 0.5 + 0.5 * math.sin(t * 1.8 + item["phase"])
            if pulse < 0.74:
                continue
            x = item["x"]
            y = (item["y"] + t * 10.0) % OUT_SIZE[1]
            draw.line(
                (x, y, x + item["length"], y),
                fill=(90, 205, 235, int(item["alpha"] * pulse)), width=1,
            )

        offset = int((t * 37) % 8)
        for y in range(offset, OUT_SIZE[1], 8):
            draw.line((0, y, OUT_SIZE[0], y), fill=(115, 200, 235, 11), width=1)

        scan_y = int((t * 150) % (OUT_SIZE[1] + 250)) - 125
        draw.rectangle((0, scan_y, OUT_SIZE[0], scan_y + 48), fill=(80, 205, 235, 8))
        canvas.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot, view_radius, tilt, yaw = shot_state(t)
        canvas = self.render_background(t)

        self.draw_orbit_paths(canvas, t, shot, view_radius, tilt, yaw)
        self.draw_start_ghosts(canvas, t, view_radius, tilt, yaw)
        self.draw_sun(canvas, t)
        self.draw_planets(canvas, t, shot, view_radius, tilt, yaw)
        self.draw_planet_speed_board(canvas, t, shot)
        self.draw_outer_completion_panel(canvas, t)
        self.draw_timeline(canvas, t)
        self.draw_corner_hud(canvas, t, shot, view_radius)
        self.add_text_layers(canvas, t, shot)
        self.draw_hud_noise(canvas, t)

        arr = np.asarray(canvas.convert("RGB"))
        arr = apply_grade(arr)
        arr = np.clip(arr.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)

        fade_in = smoothstep(t / 0.8)
        fade_out = 1.0 - smoothstep((t - (CONFIG["duration_s"] - 1.0)) / 0.9)
        arr = np.clip(arr.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)
        return arr


# %% [markdown]
# ## Derived position archive and scientific previews

def write_position_archive(scene: SolarSystemCenturyScene) -> Path:
    step = int(CONFIG["position_archive_step_days"])
    sample_jd = np.arange(scene.start_jd, scene.end_jd + 0.1, step, dtype=np.float64)
    rows = []
    for planet in PLANETS:
        xyz = jpl_heliocentric_xyz(planet, sample_jd)
        radius = np.linalg.norm(xyz, axis=1)
        for jd, position, r_au in zip(sample_jd, xyz, radius):
            rows.append({
                "date_utc_label": jd_to_datetime(float(jd)).strftime("%Y-%m-%d"),
                "julian_date": float(jd),
                "body": planet.name,
                "x_au_j2000_ecliptic": float(position[0]),
                "y_au_j2000_ecliptic": float(position[1]),
                "z_au_j2000_ecliptic": float(position[2]),
                "heliocentric_distance_au": float(r_au),
                "source_model": "NASA/JPL SSD approximate planetary positions Table 1",
            })

    path = DATA_ROOT / "jpl_planet_positions_1950_2050_30day.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("Derived JPL position archive written:", path.resolve())
    return path


def create_scientific_previews(scene: SolarSystemCenturyScene):
    fig, ax = plt.subplots(figsize=(9, 9))
    for planet in PLANETS:
        path = scene.paths[planet.name]
        ax.plot(path[:, 0], path[:, 1], linewidth=1.0, label=planet.name)
    ax.scatter([0], [0], s=80, marker="*", label="Sun")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("JPL approximate heliocentric planetary paths, 1950–2050")
    ax.set_xlabel("J2000 ecliptic x [AU]")
    ax.set_ylabel("J2000 ecliptic y [AU]")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "century_orbital_paths_xy.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    sample_dates = [jd_to_datetime(float(jd)).replace(tzinfo=None) for jd in scene.path_jd]
    for planet in PLANETS:
        path = scene.paths[planet.name]
        radius = np.linalg.norm(path, axis=1)
        ax.plot(sample_dates, radius, linewidth=1.0, label=planet.name)
    ax.set_title("Heliocentric distance through the 100-year playback")
    ax.set_xlabel("Date")
    ax.set_ylabel("Heliocentric distance [AU]")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "heliocentric_distance_1950_2050.png", dpi=170)
    plt.close(fig)

    print("Scientific preview plots written to:", PREVIEW_DIR.resolve())


# %% [markdown]
# ## Subtitle sidecar

def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path):
    lines = []
    for index, (start, end, text) in enumerate(captions, start=1):
        lines.append(str(index))
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# %% [markdown]
# ## Video render and optional FFmpeg finishing

def run_ffmpeg(command: List[str]):
    print("Running:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def render_video(scene: SolarSystemCenturyScene):
    raw_video_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    subbed_video_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_subbed.mp4"
    audio_video_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_with_audio.mp4"
    final_video_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"

    if CONFIG.get("write_subtitle_sidecar", True):
        write_srt(CAPTIONS, srt_path)
        print("Subtitle sidecar written:", srt_path.resolve())

    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    times = np.arange(frame_count) / CONFIG["fps"]

    print(
        f"Rendering {frame_count:,} frames at "
        f"{CONFIG['video_width']}×{CONFIG['video_height']} ..."
    )

    with iio.get_writer(
        raw_video_path,
        fps=CONFIG["fps"],
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times, desc="Rendering 100-year Solar System short"):
            writer.append_data(scene.render_frame(float(t)))

    print("Raw video written:", raw_video_path.resolve())

    ffmpeg = find_ffmpeg()
    print("FFmpeg detected:", ffmpeg)
    final_candidate = raw_video_path

    if CONFIG.get("burn_subtitles", False) and ffmpeg and srt_path.exists():
        command = [
            ffmpeg, "-y", "-i", str(final_candidate),
            "-vf",
            (
                f"subtitles={srt_path}:"
                "force_style=Fontname=DejaVu Sans,"
                "Fontsize=22,Outline=1.2,BorderStyle=3,MarginV=90"
            ),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
            str(subbed_video_path),
        ]
        run_ffmpeg(command)
        final_candidate = subbed_video_path
        print("Subtitled video written:", subbed_video_path.resolve())

    audio_path = CONFIG.get("audio_path")
    if audio_path and Path(audio_path).exists() and ffmpeg:
        command = [
            ffmpeg, "-y", "-i", str(final_candidate), "-i", str(audio_path),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            str(audio_video_path),
        ]
        run_ffmpeg(command)
        final_candidate = audio_video_path
        print("Audio-muxed video written:", audio_video_path.resolve())
    elif audio_path:
        print("audio_path was set, but the file was not found or FFmpeg was unavailable.")

    if final_candidate.exists():
        shutil.copyfile(final_candidate, final_video_path)
        print("Final video:", final_video_path.resolve())

    return final_video_path


# %% [markdown]
# ## Main pipeline

def main():
    print("Starting 100-year Solar System pipeline ...")
    print("Source model: NASA/JPL SSD approximate planetary positions, Table 1")
    print("Playback:", CONFIG["start_date"], "to", CONFIG["end_date"])

    scene = SolarSystemCenturyScene()

    write_position_archive(scene)
    create_scientific_previews(scene)

    preview_times = [1.2, 10.5, 23.0, 37.0, 48.0, CONFIG["duration_s"] - 1.2]
    for preview_time in tqdm(preview_times, desc="Preview frames"):
        frame = scene.render_frame(float(preview_time))
        Image.fromarray(frame).save(
            PREVIEW_DIR / f"preview_{int(round(preview_time)):02d}s.png"
        )

    print("Preview frames written to:", PREVIEW_DIR.resolve())
    render_video(scene)

    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-", path.name)


if __name__ == "__main__":
    main()


# %% [markdown]
# ## Suggested narration
#
# Eight planets. One century.
# These positions are calculated from NASA/JPL's published planetary orbital
# elements and their century-rates.
#
# Start the clock in 1950.
# Mercury tears around the Sun. Venus follows. Earth repeats its year while Mars
# traces a slower, more eccentric path.
#
# Pull back.
# Jupiter and Saturn move on a completely different timescale.
# Pull back again and Uranus and Neptune barely seem to turn.
# In one hundred years, Neptune does not finish a full revolution.
#
# The ghost rings mark where every planet started in 1950.
# By 2050, the entire arrangement has changed.
#
# The Solar System is not a diagram.
# It is a machine in continuous motion.
#
# Suggested YouTube Shorts caption:
#
# Watch 100 years of planetary motion compressed into one minute.
# Positions are calculated from NASA/JPL Solar System Dynamics' published
# Keplerian elements and century-rates for the 1800–2050 interval.
#
# Earth is represented by the Earth-Moon barycenter in the JPL table. Planet
# marker sizes are enlarged for visibility.
#
# #Astronomy #SolarSystem #NASA #JPL #SpaceData #Python #YouTubeShorts
