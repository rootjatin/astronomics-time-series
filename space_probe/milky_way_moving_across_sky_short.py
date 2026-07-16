# %% [markdown]
# # The Milky Way Is Moving Across the Sky — Real-Sky YouTube Short
#
# Creates a vertical 1080x1920 cinematic astronomy Short showing the apparent
# overnight motion of the Milky Way from Hanle, Ladakh, India.
#
#
# What is explanatory artwork:
# - the density of unresolved stars inside the Milky Way band,
# - dust-lane opacity and nebular glow,
# - mountain silhouettes, observatory foreground, bloom, haze and HUD effects.
#
# Scientific point:
# The Milky Way does not physically sweep around Earth in a few hours. The
# apparent east-to-west motion is diurnal motion caused by Earth's rotation.
#
# Default observing site:
# Indian Astronomical Observatory, Hanle
# Latitude  : 32 deg 46 min 46 sec N
# Longitude : 78 deg 57 min 51 sec E
# Elevation : about 4500 m
#
# Install:
#     pip install numpy pandas pillow imageio imageio-ffmpeg tqdm
#
# Final render:
#     python milky_way_moving_across_sky_short.py
#
# Quick test:
#     MILKY_WAY_SHORT_QUICK=1 python milky_way_moving_across_sky_short.py

from __future__ import annotations

import csv
import hashlib
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# %% [markdown]
# ## Configuration

OUTPUT_ROOT = Path("milky_way_moving_across_sky_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

CONFIG: Dict = {
    "video_width": 1080,
    "video_height": 1920,
    "fps": 24,
    "duration_s": 58,
    "output_basename": "the_milky_way_is_moving_across_the_sky",

    # Indian Astronomical Observatory, Hanle.
    "site_name": "INDIAN ASTRONOMICAL OBSERVATORY // HANLE",
    "latitude_deg": 32.779444,      # 32°46′46″ N
    "longitude_deg": 78.964167,     # 78°57′51″ E, east-positive
    "elevation_m": 4500,
    "timezone_label": "IST",

    # New-moon-night observing sequence.
    "night_start_local": "2026-07-14T20:30:00+05:30",
    "night_end_local": "2026-07-15T04:30:00+05:30",

    # Celestial rendering.
    "milky_way_star_count": 2850,
    "field_star_count": 900,
    "dust_knot_count": 190,
    "meteor_count": 9,
    "horizon_radius_px": 760,
    "dome_center_x": 540,
    "dome_center_y": 790,
    "south_camera_azimuth_deg": 180.0,
    "horizontal_fov_deg": 228.0,
    "min_altitude_deg": -3.0,

    # Image finishing.
    "contrast_boost": 1.13,
    "saturation_boost": 1.10,
    "vignette_strength": 0.31,
    "scanline_spacing": 8,

    # Text.
    "title_text": "THE MILKY WAY IS\nMOVING ACROSS THE SKY",
    "subtitle_text": "One real night compressed into 58 seconds",
    "credit_text": (
        "Sky geometry: J2000 Galactic coordinates + sidereal-time alt/az transform"
    ),
    "scientific_note": (
        "The apparent motion is caused by Earth's rotation. Star-density texture, "
        "dust contrast and landscape are explanatory artwork."
    ),

    # Optional finishing.
    "audio_path": None,
    "burn_subtitles": False,
    "write_subtitle_sidecar": True,
}

if os.getenv("MILKY_WAY_SHORT_QUICK") == "1":
    CONFIG.update({
        "video_width": 540,
        "video_height": 960,
        "fps": 6,
        "duration_s": 12,
        "output_basename": "the_milky_way_is_moving_quick_preview",
        "milky_way_star_count": 1450,
        "field_star_count": 430,
        "dust_knot_count": 90,
        "meteor_count": 5,
        "horizon_radius_px": 380,
        "dome_center_x": 270,
        "dome_center_y": 395,
    })

BASE_STORY_DURATION_S = 58.0

OUT_SIZE = (int(CONFIG["video_width"]), int(CONFIG["video_height"]))
SX = OUT_SIZE[0] / 1080.0
SY = OUT_SIZE[1] / 1920.0
S = min(SX, SY)


# %% [markdown]
# ## Real celestial anchors

# Coordinates are J2000/ICRS. The Galactic Center uses the radio position of
# Sagittarius A*. Magnitudes are approximate visual magnitudes used only to size
# the markers; stellar positions drive the sky motion.
BRIGHT_STARS = [
    # name, RA degrees, Dec degrees, V magnitude, label priority
    ("Sgr A* // Galactic Center", 266.41683, -29.00781, 8.0, 100),
    ("Antares", 247.35192, -26.43200, 0.96, 85),
    ("Shaula", 263.40217, -37.10382, 1.62, 70),
    ("Kaus Australis", 276.04300, -34.38462, 1.79, 70),
    ("Nunki", 283.81633, -26.29672, 2.05, 55),
    ("Altair", 297.69583, 8.86832, 0.77, 78),
    ("Vega", 279.23473, 38.78369, 0.03, 78),
    ("Deneb", 310.35798, 45.28034, 1.25, 74),
    ("Arcturus", 213.91530, 19.18241, -0.05, 60),
    ("Spica", 201.29825, -11.16132, 0.98, 58),
    ("Rasalhague", 263.73363, 12.56004, 2.08, 45),
    ("Sabik", 257.59450, -15.72491, 2.43, 42),
    ("Atria", 252.16623, -69.02772, 1.91, 38),
    ("Peacock", 306.41188, -56.73509, 1.94, 38),
]

# Standard J2000 rotation matrix from equatorial Cartesian coordinates to
# Galactic Cartesian coordinates. Transpose converts Galactic -> equatorial.
EQ_TO_GAL = np.array([
    [-0.0548755604, -0.8734370902, -0.4838350155],
    [ 0.4941094279, -0.4448296300,  0.7469822445],
    [-0.8676661490, -0.1980763734,  0.4559837762],
], dtype=float)
GAL_TO_EQ = EQ_TO_GAL.T


# %% [markdown]
# ## Captions and shot plan

CAPTIONS = [
    (0.6, 6.0, "The Milky Way seems to sweep across the sky in a single night."),
    (6.2, 15.5, "But the galaxy is not circling Earth. Earth is rotating beneath the stars."),
    (15.8, 27.0, "Fixed celestial coordinates become changing altitude and azimuth at Hanle."),
    (27.3, 39.5, "The bright Galactic Center follows a real calculated track toward the west."),
    (39.8, 49.5, "Long exposures turn the same rotation into arcs around the celestial pole."),
    (49.8, 57.2, "Eight hours later, the horizon has rotated beneath a nearly fixed galaxy."),
]

SHOT_PLAN = [
    {
        "name": "reveal",
        "start": 0.0,
        "end": 7.0,
        "time_start": 0.00,
        "time_end": 0.05,
        "camera_roll_start": -1.2,
        "camera_roll_end": 0.0,
        "band_gain": 1.00,
    },
    {
        "name": "motion",
        "start": 7.0,
        "end": 18.0,
        "time_start": 0.05,
        "time_end": 0.31,
        "camera_roll_start": 0.0,
        "camera_roll_end": 0.8,
        "band_gain": 1.05,
    },
    {
        "name": "coordinates",
        "start": 18.0,
        "end": 30.0,
        "time_start": 0.31,
        "time_end": 0.56,
        "camera_roll_start": 0.8,
        "camera_roll_end": -0.5,
        "band_gain": 1.02,
    },
    {
        "name": "galactic_center",
        "start": 30.0,
        "end": 41.0,
        "time_start": 0.56,
        "time_end": 0.76,
        "camera_roll_start": -0.5,
        "camera_roll_end": 0.3,
        "band_gain": 1.18,
    },
    {
        "name": "trails",
        "start": 41.0,
        "end": 50.0,
        "time_start": 0.76,
        "time_end": 0.91,
        "camera_roll_start": 0.3,
        "camera_roll_end": 0.0,
        "band_gain": 0.92,
    },
    {
        "name": "comparison",
        "start": 50.0,
        "end": 58.0,
        "time_start": 0.91,
        "time_end": 1.00,
        "camera_roll_start": 0.0,
        "camera_roll_end": 0.0,
        "band_gain": 1.08,
    },
]


# %% [markdown]
# ## General helpers


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t: float) -> float:
    t = clamp(t)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def ease_in_out_sine(t: float) -> float:
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


def deterministic_unit(text: str, salt: str = "") -> float:
    digest = hashlib.sha256(f"{text}|{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def wrap_degrees(value: float) -> float:
    return value % 360.0


def wrap_signed_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


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
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold else
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
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
    xy: Tuple[float, float],
    size: int = 42,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    stroke: int = 2,
    anchor: str = "la",
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
    stroke_scaled = max(0, int(stroke * S))
    draw.text(
        (int(xy[0]), int(xy[1])),
        text,
        font=font,
        fill=fill,
        anchor=anchor,
        stroke_width=stroke_scaled,
        stroke_fill=(0, 0, 0, min(fill[3] if len(fill) > 3 else 255, 225)),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[float, float],
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
    scaled_max_width = int(max_width * SX)
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=max(1, int(2 * S)))
        if bbox[2] - bbox[0] <= scaled_max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = int(xy[0]), int(xy[1])
    for line in lines:
        draw.text(
            (x, y), line, font=font, fill=fill,
            stroke_width=max(1, int(2 * S)), stroke_fill=(0, 0, 0, 220),
        )
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=max(1, int(2 * S)))
        y += (bbox[3] - bbox[1]) + max(2, int(line_spacing * S))


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cx = width / 2.0
    cy = height / 2.0
    nx = (xx - cx) / max(width / 2.0, 1.0)
    ny = (yy - cy) / max(height / 2.0, 1.0)
    radius = np.sqrt(nx**2 + ny**2)
    return np.clip(1.0 - strength * radius**1.8, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_SIZE[0], OUT_SIZE[1], float(CONFIG["vignette_strength"]))


# %% [markdown]
# ## Astronomical calculations


def parse_iso_datetime(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError("Configured date-time must include a UTC offset.")
    return dt


NIGHT_START_LOCAL = parse_iso_datetime(str(CONFIG["night_start_local"]))
NIGHT_END_LOCAL = parse_iso_datetime(str(CONFIG["night_end_local"]))
NIGHT_START_UTC = NIGHT_START_LOCAL.astimezone(UTC)
NIGHT_END_UTC = NIGHT_END_LOCAL.astimezone(UTC)
NIGHT_DURATION_SECONDS = (NIGHT_END_UTC - NIGHT_START_UTC).total_seconds()


def julian_date(dt: datetime) -> float:
    """UTC-like calendar date to Julian Date, adequate for visualization."""
    dt = dt.astimezone(UTC)
    year = dt.year
    month = dt.month
    day_fraction = (
        dt.day
        + (dt.hour + (dt.minute + (dt.second + dt.microsecond / 1e6) / 60.0) / 60.0) / 24.0
    )
    if month <= 2:
        year -= 1
        month += 12
    a = math.floor(year / 100)
    b = 2 - a + math.floor(a / 4)
    return (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day_fraction + b - 1524.5
    )


def greenwich_mean_sidereal_time_deg(dt: datetime) -> float:
    """Approximate GMST in degrees using a standard J2000 polynomial."""
    jd = julian_date(dt)
    t = (jd - 2451545.0) / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - (t * t * t) / 38710000.0
    )
    return wrap_degrees(gmst)


def local_sidereal_time_deg(dt: datetime, longitude_deg: float) -> float:
    return wrap_degrees(greenwich_mean_sidereal_time_deg(dt) + longitude_deg)


def radec_to_altaz(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    dt: datetime,
    latitude_deg: float,
    longitude_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert J2000-like RA/Dec to geometric altitude/azimuth.

    This educational renderer ignores precession, nutation, aberration,
    atmospheric refraction and polar motion. That is sufficient for the
    apparent multi-hour motion shown here but not precision astrometry.
    """
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    lat = math.radians(latitude_deg)
    lst = math.radians(local_sidereal_time_deg(dt, longitude_deg))
    hour_angle = (lst - ra + math.pi) % (2.0 * math.pi) - math.pi

    sin_alt = np.sin(dec) * math.sin(lat) + np.cos(dec) * math.cos(lat) * np.cos(hour_angle)
    alt = np.arcsin(np.clip(sin_alt, -1.0, 1.0))

    # Azimuth measured north=0, east=90.
    y = -np.sin(hour_angle) * np.cos(dec)
    x = np.sin(dec) * math.cos(lat) - np.cos(dec) * math.sin(lat) * np.cos(hour_angle)
    az = np.arctan2(y, x) % (2.0 * math.pi)
    return np.rad2deg(alt), np.rad2deg(az)


def galactic_to_radec(l_deg: np.ndarray, b_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    l = np.deg2rad(np.asarray(l_deg, dtype=float))
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    gal = np.stack([
        np.cos(b) * np.cos(l),
        np.cos(b) * np.sin(l),
        np.sin(b),
    ], axis=0)
    eq = GAL_TO_EQ @ gal
    x, y, z = eq[0], eq[1], eq[2]
    ra = np.arctan2(y, x) % (2.0 * math.pi)
    dec = np.arcsin(np.clip(z, -1.0, 1.0))
    return np.rad2deg(ra), np.rad2deg(dec)


def sky_project(
    alt_deg: np.ndarray,
    az_deg: np.ndarray,
    roll_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project the local sky onto a south-facing cinematic dome."""
    alt = np.asarray(alt_deg, dtype=float)
    az = np.asarray(az_deg, dtype=float)
    rel_az = ((az - float(CONFIG["south_camera_azimuth_deg"]) + 180.0) % 360.0) - 180.0

    visible = (
        (alt >= float(CONFIG["min_altitude_deg"]))
        & (np.abs(rel_az) <= float(CONFIG["horizontal_fov_deg"]) / 2.0)
    )

    zenith_distance = np.clip(90.0 - alt, 0.0, 96.0)
    r = float(CONFIG["horizon_radius_px"]) * (zenith_distance / 90.0) ** 0.92
    theta = np.deg2rad(rel_az + roll_deg)

    x = float(CONFIG["dome_center_x"]) + r * np.sin(theta)
    y = float(CONFIG["dome_center_y"]) + 0.72 * r * np.cos(theta)
    return x, y, visible


def format_sidereal_hours(degrees: float) -> str:
    total_hours = wrap_degrees(degrees) / 15.0
    hours = int(total_hours)
    minutes = int((total_hours - hours) * 60.0)
    return f"{hours:02d}h {minutes:02d}m"


# %% [markdown]
# ## Deterministic sky catalog


@dataclass
class SkyParticleCatalog:
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    magnitude: np.ndarray
    temperature: np.ndarray
    phase: np.ndarray
    kind: np.ndarray
    galactic_longitude: np.ndarray
    galactic_latitude: np.ndarray


def make_sky_catalog() -> SkyParticleCatalog:
    rng = np.random.default_rng(20260714)

    n_band = int(CONFIG["milky_way_star_count"])
    n_field = int(CONFIG["field_star_count"])

    # More unresolved stars are placed toward Galactic longitude zero, where the
    # inner Milky Way appears richest. Sampling remains explanatory artwork; the
    # Galactic coordinate plane itself is real.
    mix = rng.uniform(0.0, 1.0, n_band)
    longitude_uniform = rng.uniform(0.0, 360.0, n_band)
    longitude_center = rng.normal(0.0, 38.0, n_band) % 360.0
    l_band = np.where(mix < 0.48, longitude_center, longitude_uniform)

    center_distance = np.minimum(l_band, 360.0 - l_band)
    width = 2.0 + 5.0 * np.clip(center_distance / 180.0, 0.0, 1.0)
    b_band = rng.normal(0.0, width)
    b_band = np.clip(b_band, -15.0, 15.0)

    ra_band, dec_band = galactic_to_radec(l_band, b_band)
    mag_band = np.clip(rng.normal(5.8, 1.35, n_band), 1.4, 9.0)
    temp_band = rng.uniform(0.0, 1.0, n_band)

    # General field stars uniformly distributed on the celestial sphere.
    ra_field = rng.uniform(0.0, 360.0, n_field)
    dec_field = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n_field)))
    mag_field = np.clip(rng.normal(5.2, 1.25, n_field), 0.7, 8.5)
    temp_field = rng.uniform(0.0, 1.0, n_field)
    l_field = np.full(n_field, np.nan)
    b_field = np.full(n_field, np.nan)

    ra = np.concatenate([ra_band, ra_field])
    dec = np.concatenate([dec_band, dec_field])
    mag = np.concatenate([mag_band, mag_field])
    temp = np.concatenate([temp_band, temp_field])
    phase = rng.uniform(0.0, 2.0 * math.pi, len(ra))
    kind = np.concatenate([
        np.ones(n_band, dtype=np.int8),
        np.zeros(n_field, dtype=np.int8),
    ])
    l_all = np.concatenate([l_band, l_field])
    b_all = np.concatenate([b_band, b_field])

    return SkyParticleCatalog(
        ra_deg=ra,
        dec_deg=dec,
        magnitude=mag,
        temperature=temp,
        phase=phase,
        kind=kind,
        galactic_longitude=l_all,
        galactic_latitude=b_all,
    )


# %% [markdown]
# ## Derived real-sky archive


def create_position_archive() -> Path:
    rows: List[Dict] = []
    objects = BRIGHT_STARS
    step_minutes = 10
    dt = NIGHT_START_UTC
    while dt <= NIGHT_END_UTC:
        ra = np.array([item[1] for item in objects], dtype=float)
        dec = np.array([item[2] for item in objects], dtype=float)
        alt, az = radec_to_altaz(
            ra, dec, dt,
            float(CONFIG["latitude_deg"]),
            float(CONFIG["longitude_deg"]),
        )
        local = dt.astimezone(IST)
        lst_deg = local_sidereal_time_deg(dt, float(CONFIG["longitude_deg"]))
        for index, item in enumerate(objects):
            rows.append({
                "time_local_iso": local.isoformat(),
                "time_utc_iso": dt.isoformat(),
                "object": item[0],
                "ra_j2000_deg": item[1],
                "dec_j2000_deg": item[2],
                "altitude_deg_geometric": float(alt[index]),
                "azimuth_deg_north_through_east": float(az[index]),
                "local_mean_sidereal_time_deg": float(lst_deg),
                "local_mean_sidereal_time_hours": float(lst_deg / 15.0),
                "site_latitude_deg": float(CONFIG["latitude_deg"]),
                "site_longitude_deg_east": float(CONFIG["longitude_deg"]),
            })
        dt += timedelta(minutes=step_minutes)

    frame = pd.DataFrame(rows)
    path = DATA_ROOT / "hanle_sky_positions_2026-07-14_new_moon_night.csv"
    frame.to_csv(path, index=False)
    return path


# %% [markdown]
# ## Cinematic scene


class MilkyWaySkyScene:
    def __init__(self):
        self.catalog = make_sky_catalog()
        self.bright_names = np.array([item[0] for item in BRIGHT_STARS], dtype=object)
        self.bright_ra = np.array([item[1] for item in BRIGHT_STARS], dtype=float)
        self.bright_dec = np.array([item[2] for item in BRIGHT_STARS], dtype=float)
        self.bright_mag = np.array([item[3] for item in BRIGHT_STARS], dtype=float)
        self.bright_priority = np.array([item[4] for item in BRIGHT_STARS], dtype=int)

        # The first anchor is Sagittarius A* / Galactic Center.
        self.gc_ra = float(self.bright_ra[0])
        self.gc_dec = float(self.bright_dec[0])

        # Galactic equator ridge and nearby rails for band rendering.
        self.ridge_l = np.linspace(0.0, 360.0, 721)
        self.ridge_b = np.zeros_like(self.ridge_l)
        self.ridge_ra, self.ridge_dec = galactic_to_radec(self.ridge_l, self.ridge_b)
        self.rail_offsets = [-10.0, -6.0, -3.0, 0.0, 3.0, 6.0, 10.0]
        self.rails: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        for offset in self.rail_offsets:
            self.rails[offset] = galactic_to_radec(self.ridge_l, np.full_like(self.ridge_l, offset))

        self.dust_knots = self._make_dust_knots(int(CONFIG["dust_knot_count"]))
        self.meteors = self._make_meteors(int(CONFIG["meteor_count"]))
        self.landscape = self._make_landscape()

    @staticmethod
    def _make_dust_knots(count: int) -> List[Dict]:
        rng = np.random.default_rng(1991)
        knots: List[Dict] = []
        for index in range(count):
            # Concentrate dark material around the inner-galaxy half.
            if index < count * 0.65:
                l = float(rng.normal(0.0, 50.0) % 360.0)
            else:
                l = float(rng.uniform(0.0, 360.0))
            b = float(rng.normal(0.0, 2.7))
            ra, dec = galactic_to_radec(np.array([l]), np.array([b]))
            knots.append({
                "ra": float(ra[0]),
                "dec": float(dec[0]),
                "radius": float(rng.uniform(10.0, 45.0) * S),
                "alpha": int(rng.integers(14, 48)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            })
        return knots

    @staticmethod
    def _make_meteors(count: int) -> List[Dict]:
        rng = np.random.default_rng(8282)
        meteors: List[Dict] = []
        for index in range(count):
            start = float(rng.uniform(5.0, BASE_STORY_DURATION_S - 4.0))
            meteors.append({
                "start": start,
                "duration": float(rng.uniform(0.28, 0.62)),
                "x": float(rng.uniform(170.0, 920.0) * SX),
                "y": float(rng.uniform(220.0, 880.0) * SY),
                "length": float(rng.uniform(85.0, 190.0) * S),
                "angle": float(rng.uniform(-0.65, -0.25)),
            })
        return meteors

    @staticmethod
    def _make_landscape() -> Image.Image:
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        width, height = OUT_SIZE
        rng = np.random.default_rng(122)

        # Distant cold-desert mountain chain.
        points = [(0, int(1400 * SY))]
        x = -80.0 * SX
        while x < width + 120 * SX:
            peak_x = x + rng.uniform(65.0, 145.0) * SX
            peak_y = rng.uniform(1230.0, 1435.0) * SY
            points.append((int(peak_x), int(peak_y)))
            x = peak_x
        points.extend([(width, height), (0, height)])
        draw.polygon(points, fill=(6, 8, 14, 255))

        # Slightly nearer ridge.
        points2 = [(0, int(1490 * SY))]
        x = -50.0 * SX
        while x < width + 100 * SX:
            peak_x = x + rng.uniform(90.0, 190.0) * SX
            peak_y = rng.uniform(1390.0, 1535.0) * SY
            points2.append((int(peak_x), int(peak_y)))
            x = peak_x
        points2.extend([(width, height), (0, height)])
        draw.polygon(points2, fill=(3, 5, 9, 255))

        # Observatory dome silhouette.
        dome_cx = int(770 * SX)
        dome_base_y = int(1570 * SY)
        dome_r = int(92 * S)
        draw.rectangle(
            (dome_cx - int(88 * SX), dome_base_y, dome_cx + int(88 * SX), int(1790 * SY)),
            fill=(2, 4, 8, 255),
        )
        draw.pieslice(
            (dome_cx - dome_r, dome_base_y - dome_r, dome_cx + dome_r, dome_base_y + dome_r),
            180, 360, fill=(3, 5, 10, 255),
        )
        slit = int(16 * S)
        draw.rectangle(
            (dome_cx - slit, dome_base_y - dome_r, dome_cx + slit, dome_base_y + int(8 * SY)),
            fill=(7, 12, 20, 255),
        )

        # Telescope annex and red safety lights.
        draw.rectangle(
            (int(660 * SX), int(1640 * SY), int(865 * SX), int(1790 * SY)),
            fill=(2, 4, 8, 255),
        )
        for lx in [692, 835]:
            draw.ellipse(
                (int((lx - 3) * SX), int(1697 * SY), int((lx + 3) * SX), int(1703 * SY)),
                fill=(240, 35, 35, 150),
            )

        # Foreground ground plane.
        draw.rectangle((0, int(1760 * SY), width, height), fill=(1, 2, 5, 255))
        return layer

    def story_time(self, render_time: float) -> float:
        return float(render_time) * BASE_STORY_DURATION_S / max(float(CONFIG["duration_s"]), 1e-6)

    def get_shot(self, t: float) -> Dict:
        for shot in SHOT_PLAN:
            if shot["start"] <= t < shot["end"]:
                return shot
        return SHOT_PLAN[-1]

    def shot_state(self, t: float) -> Tuple[Dict, float, float, float]:
        shot = self.get_shot(t)
        duration = max(float(shot["end"] - shot["start"]), 1e-6)
        u = clamp((t - float(shot["start"])) / duration)
        e = ease_in_out_sine(u)
        time_fraction = lerp(float(shot["time_start"]), float(shot["time_end"]), e)
        roll = lerp(float(shot["camera_roll_start"]), float(shot["camera_roll_end"]), e)
        gain = float(shot["band_gain"])
        return shot, time_fraction, roll, gain

    def sky_datetime(self, time_fraction: float) -> datetime:
        return NIGHT_START_UTC + timedelta(seconds=NIGHT_DURATION_SECONDS * clamp(time_fraction))

    def render_background(self, t: float, time_fraction: float) -> Image.Image:
        width, height = OUT_SIZE
        yy = np.arange(height, dtype=np.float32)[:, None]
        top = np.array([1.0, 3.0, 12.0], dtype=np.float32)
        bottom = np.array([10.0, 16.0, 31.0], dtype=np.float32)
        blend = np.clip(yy / max(height * 0.82, 1.0), 0.0, 1.0) ** 1.35
        arr = top[None, None, :] * (1.0 - blend[:, :, None]) + bottom[None, None, :] * blend[:, :, None]
        arr = np.repeat(arr, width, axis=1)

        # Slight predawn blue near the end of the simulated night.
        dawn = smoothstep((time_fraction - 0.91) / 0.09)
        if dawn > 0:
            dawn_color = np.array([16.0, 28.0, 46.0], dtype=np.float32)
            horizon_weight = np.clip((yy - height * 0.42) / max(height * 0.48, 1.0), 0.0, 1.0)
            arr = arr * (1.0 - dawn * 0.52 * horizon_weight[:, :, None]) + dawn_color * (dawn * 0.52 * horizon_weight[:, :, None])

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

    def draw_milky_way_band(
        self,
        canvas: Image.Image,
        dt: datetime,
        roll: float,
        gain: float,
        t: float,
    ):
        # Broad glow rails establish the real galactic-equator geometry.
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for offset in self.rail_offsets:
            ra, dec = self.rails[offset]
            alt, az = radec_to_altaz(
                ra, dec, dt,
                float(CONFIG["latitude_deg"]),
                float(CONFIG["longitude_deg"]),
            )
            x, y, visible = sky_project(alt, az, roll)
            points: List[Tuple[int, int]] = []
            previous_valid = False
            width = max(1, int((10.0 - abs(offset) * 0.45) * S))
            alpha = int((13 + (10.0 - abs(offset)) * 3.2) * gain)
            for index in range(len(x)):
                valid = bool(visible[index]) and -120 <= x[index] <= OUT_SIZE[0] + 120 and -140 <= y[index] <= OUT_SIZE[1] + 140
                if valid:
                    point = (int(x[index]), int(y[index]))
                    if previous_valid and points:
                        gd.line((points[-1], point), fill=(112, 132, 185, alpha), width=width)
                    points.append(point)
                else:
                    points = []
                previous_valid = valid

        glow = glow.filter(ImageFilter.GaussianBlur(max(5, int(23 * S))))
        canvas.alpha_composite(glow)

        # Unresolved stars following fixed celestial coordinates.
        alt, az = radec_to_altaz(
            self.catalog.ra_deg,
            self.catalog.dec_deg,
            dt,
            float(CONFIG["latitude_deg"]),
            float(CONFIG["longitude_deg"]),
        )
        x, y, visible = sky_project(alt, az, roll)

        star_glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        sg = ImageDraw.Draw(star_glow)
        star_core = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        sc = ImageDraw.Draw(star_core)

        dawn = smoothstep((self.current_time_fraction - 0.91) / 0.09)
        for index in np.where(visible)[0]:
            px = float(x[index])
            py = float(y[index])
            if px < -12 or px > OUT_SIZE[0] + 12 or py < -12 or py > OUT_SIZE[1] + 12:
                continue
            mag = float(self.catalog.magnitude[index])
            is_band = int(self.catalog.kind[index]) == 1
            twinkle = 0.72 + 0.28 * math.sin(t * 1.7 + float(self.catalog.phase[index]))
            base_alpha = 225.0 * 10.0 ** (-0.17 * (mag - 1.0))
            if is_band:
                base_alpha *= 1.38 * gain
            base_alpha *= (1.0 - 0.60 * dawn)
            alpha = int(np.clip(base_alpha * twinkle, 7, 235))
            radius = np.clip((2.25 - 0.19 * mag) * S, 0.45 * S, 2.5 * S)

            temp = float(self.catalog.temperature[index])
            if temp < 0.32:
                color = (170, 205, 255)
            elif temp > 0.78:
                color = (255, 215, 170)
            else:
                color = (225, 235, 255)

            if radius > 1.0 * S and alpha > 55:
                gr = radius * 3.8
                sg.ellipse((px - gr, py - gr, px + gr, py + gr), fill=(*color, int(alpha * 0.14)))
            sc.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(*color, alpha))

        star_glow = star_glow.filter(ImageFilter.GaussianBlur(max(1, int(4 * S))))
        canvas.alpha_composite(star_glow)
        canvas.alpha_composite(star_core)

        # Dust knots subtract brightness along the real galactic plane.
        dust_layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        dd = ImageDraw.Draw(dust_layer)
        for knot in self.dust_knots:
            alt_k, az_k = radec_to_altaz(
                np.array([knot["ra"]]), np.array([knot["dec"]]), dt,
                float(CONFIG["latitude_deg"]), float(CONFIG["longitude_deg"]),
            )
            kx, ky, kv = sky_project(alt_k, az_k, roll)
            if not bool(kv[0]):
                continue
            radius = float(knot["radius"]) * (0.88 + 0.12 * math.sin(t * 0.12 + float(knot["phase"])))
            alpha = int(float(knot["alpha"]) * gain)
            dd.ellipse(
                (float(kx[0]) - radius, float(ky[0]) - radius * 0.55,
                 float(kx[0]) + radius, float(ky[0]) + radius * 0.55),
                fill=(0, 1, 6, alpha),
            )
        dust_layer = dust_layer.filter(ImageFilter.GaussianBlur(max(4, int(11 * S))))
        canvas.alpha_composite(dust_layer)

    def draw_named_stars(self, canvas: Image.Image, dt: datetime, roll: float, t: float):
        alt, az = radec_to_altaz(
            self.bright_ra, self.bright_dec, dt,
            float(CONFIG["latitude_deg"]), float(CONFIG["longitude_deg"]),
        )
        x, y, visible = sky_project(alt, az, roll)

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        sharp = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sharp)

        shot = self.current_shot["name"]
        for index in np.where(visible)[0]:
            px = float(x[index])
            py = float(y[index])
            if not (-40 <= px <= OUT_SIZE[0] + 40 and -40 <= py <= OUT_SIZE[1] + 40):
                continue
            name = str(self.bright_names[index])
            mag = float(self.bright_mag[index])
            is_gc = index == 0
            if is_gc:
                radius = 5.8 * S
                color = (255, 188, 90)
                pulse = 0.78 + 0.22 * math.sin(t * 3.0)
            else:
                radius = np.clip((5.0 - mag) * 0.75 * S, 1.3 * S, 4.3 * S)
                color = (210, 228, 255)
                pulse = 0.92 + 0.08 * math.sin(t * 2.0 + index)

            gr = radius * (6.0 if is_gc else 4.0)
            gd.ellipse((px - gr, py - gr, px + gr, py + gr), fill=(*color, int(55 * pulse)))
            sd.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(*color, int(245 * pulse)))

            show_label = (
                is_gc
                or (shot in {"coordinates", "galactic_center"} and int(self.bright_priority[index]) >= 70)
            )
            if show_label and py < 1450 * SY:
                label = "GALACTIC CENTER" if is_gc else name.upper()
                draw_text(
                    sharp, label, (px + 13 * S, py - 12 * S), size=17 if not is_gc else 20,
                    fill=(255, 205, 120, 240) if is_gc else (190, 220, 245, 215),
                    bold=True, stroke=1,
                )

        glow = glow.filter(ImageFilter.GaussianBlur(max(2, int(8 * S))))
        canvas.alpha_composite(glow)
        canvas.alpha_composite(sharp)

    def draw_galactic_center_track(self, canvas: Image.Image, dt: datetime, roll: float, t: float):
        if self.current_shot["name"] not in {"galactic_center", "trails", "comparison"}:
            return

        # Real track from the beginning to the end of the configured night.
        samples = 100
        datetimes = [
            NIGHT_START_UTC + timedelta(seconds=NIGHT_DURATION_SECONDS * index / (samples - 1))
            for index in range(samples)
        ]
        altitudes = []
        azimuths = []
        for sample_dt in datetimes:
            alt, az = radec_to_altaz(
                np.array([self.gc_ra]), np.array([self.gc_dec]), sample_dt,
                float(CONFIG["latitude_deg"]), float(CONFIG["longitude_deg"]),
            )
            altitudes.append(float(alt[0]))
            azimuths.append(float(az[0]))
        x, y, visible = sky_project(np.array(altitudes), np.array(azimuths), roll)

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        valid_points: List[Tuple[int, int]] = []
        for index in range(samples):
            if bool(visible[index]) and 0 <= x[index] <= OUT_SIZE[0] and 0 <= y[index] <= OUT_SIZE[1]:
                point = (int(x[index]), int(y[index]))
                if valid_points:
                    draw.line((valid_points[-1], point), fill=(255, 174, 80, 100), width=max(1, int(2 * S)))
                valid_points.append(point)
            else:
                valid_points = []

        # Current-track cursor.
        alt_now, az_now = radec_to_altaz(
            np.array([self.gc_ra]), np.array([self.gc_dec]), dt,
            float(CONFIG["latitude_deg"]), float(CONFIG["longitude_deg"]),
        )
        cx, cy, cv = sky_project(alt_now, az_now, roll)
        if bool(cv[0]):
            radius = (24.0 + 5.0 * math.sin(t * 4.0)) * S
            draw.arc((cx[0] - radius, cy[0] - radius, cx[0] + radius, cy[0] + radius),
                     20, 150, fill=(255, 190, 90, 230), width=max(1, int(2 * S)))
            draw.arc((cx[0] - radius, cy[0] - radius, cx[0] + radius, cy[0] + radius),
                     200, 330, fill=(255, 190, 90, 230), width=max(1, int(2 * S)))
        canvas.alpha_composite(layer)

    def draw_star_trails(self, canvas: Image.Image, dt: datetime, roll: float, t: float):
        alpha_envelope = smoothstep((t - 40.0) / 2.0) * (1.0 - smoothstep((t - 51.0) / 2.0))
        if alpha_envelope <= 0.01:
            return

        # Trails use a deterministic bright subset of the fixed celestial catalog.
        subset = np.argsort(self.catalog.magnitude)[:180]
        time_offsets = np.linspace(-42.0, 42.0, 13)  # minutes around current time
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        for star_index in subset:
            points: List[Tuple[int, int]] = []
            for offset_minutes in time_offsets:
                sample_dt = dt + timedelta(minutes=float(offset_minutes))
                alt, az = radec_to_altaz(
                    np.array([self.catalog.ra_deg[star_index]]),
                    np.array([self.catalog.dec_deg[star_index]]),
                    sample_dt,
                    float(CONFIG["latitude_deg"]),
                    float(CONFIG["longitude_deg"]),
                )
                x, y, visible = sky_project(alt, az, roll)
                if bool(visible[0]) and -10 <= x[0] <= OUT_SIZE[0] + 10 and -10 <= y[0] <= OUT_SIZE[1] + 10:
                    points.append((int(x[0]), int(y[0])))
            if len(points) >= 2:
                alpha = int(42 * alpha_envelope)
                draw.line(points, fill=(150, 195, 240, alpha), width=max(1, int(1 * S)))
        canvas.alpha_composite(layer)

    def draw_meteors(self, canvas: Image.Image, t: float):
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        sharp = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sharp)
        for meteor in self.meteors:
            local = (t - float(meteor["start"])) / float(meteor["duration"])
            if not (0.0 <= local <= 1.0):
                continue
            envelope = math.sin(math.pi * local)
            x = float(meteor["x"]) + 55 * S * local
            y = float(meteor["y"]) + 22 * S * local
            length = float(meteor["length"]) * envelope
            angle = float(meteor["angle"])
            tx = x - math.cos(angle) * length
            ty = y - math.sin(angle) * length
            gd.line((tx, ty, x, y), fill=(180, 220, 255, int(90 * envelope)), width=max(2, int(10 * S)))
            sd.line((tx, ty, x, y), fill=(240, 248, 255, int(220 * envelope)), width=max(1, int(2 * S)))
        glow = glow.filter(ImageFilter.GaussianBlur(max(2, int(6 * S))))
        canvas.alpha_composite(glow)
        canvas.alpha_composite(sharp)

    def draw_horizon_and_compass(self, canvas: Image.Image, t: float):
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx = float(CONFIG["dome_center_x"])
        cy = float(CONFIG["dome_center_y"])
        radius = float(CONFIG["horizon_radius_px"])

        # Horizon arc in the south-facing projection.
        points = []
        for relative_az in np.linspace(-114.0, 114.0, 220):
            theta = math.radians(relative_az)
            points.append((cx + radius * math.sin(theta), cy + 0.72 * radius * math.cos(theta)))
        draw.line(points, fill=(90, 135, 170, 75), width=max(1, int(2 * S)))

        labels = [(-90.0, "E"), (0.0, "S"), (90.0, "W")]
        for rel_az, label in labels:
            theta = math.radians(rel_az)
            x = cx + radius * math.sin(theta)
            y = cy + 0.72 * radius * math.cos(theta)
            draw_text(layer, label, (x, y + 18 * S), size=20, fill=(155, 195, 220, 190), bold=True, anchor="ma", stroke=1)

        canvas.alpha_composite(layer)

    def draw_motion_explanation(self, canvas: Image.Image, dt: datetime, t: float):
        shot = self.current_shot["name"]
        if shot not in {"motion", "coordinates"}:
            return
        alpha = int(230 * smoothstep((t - 6.5) / 2.0) * (1.0 - smoothstep((t - 30.0) / 2.0)))
        if alpha <= 4:
            return

        x0 = int(55 * SX)
        y0 = int(255 * SY)
        width = int(460 * SX)
        height = int(245 * SY)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((x0, y0, x0 + width, y0 + height), radius=max(8, int(22 * S)),
                               fill=(2, 6, 17, 174), outline=(90, 185, 225, 90), width=max(1, int(2 * S)))
        canvas.alpha_composite(panel)

        draw_text(canvas, "WHY THE SKY MOVES", (x0 + 22 * S, y0 + 20 * S), size=21,
                  fill=(130, 225, 245, alpha), bold=True, stroke=1)
        draw_text(canvas, "EARTH ROTATES EASTWARD", (x0 + 22 * S, y0 + 61 * S), size=24,
                  fill=(245, 250, 255, alpha), bold=True, stroke=1)
        draw_text(canvas, "THE STARS APPEAR TO DRIFT WEST", (x0 + 22 * S, y0 + 101 * S), size=20,
                  fill=(215, 228, 242, alpha), bold=True, stroke=1)

        elapsed_hours = (dt - NIGHT_START_UTC).total_seconds() / 3600.0
        approximate_rotation = elapsed_hours * 15.041
        draw_text(canvas, f"ELAPSED // {elapsed_hours:4.1f} h", (x0 + 22 * S, y0 + 153 * S), size=20,
                  fill=(255, 188, 92, alpha), bold=True, stroke=1)
        draw_text(canvas, f"SIDEREAL ROTATION // {approximate_rotation:5.1f}°", (x0 + 22 * S, y0 + 189 * S), size=19,
                  fill=(255, 188, 92, alpha), bold=True, stroke=1)

    def draw_coordinate_panel(self, canvas: Image.Image, dt: datetime, t: float):
        if self.current_shot["name"] not in {"coordinates", "galactic_center", "comparison"}:
            return

        alt, az = radec_to_altaz(
            np.array([self.gc_ra]), np.array([self.gc_dec]), dt,
            float(CONFIG["latitude_deg"]), float(CONFIG["longitude_deg"]),
        )
        lst = local_sidereal_time_deg(dt, float(CONFIG["longitude_deg"]))
        local = dt.astimezone(IST)

        x0 = int(570 * SX)
        y0 = int(255 * SY)
        width = int(455 * SX)
        height = int(285 * SY)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((x0, y0, x0 + width, y0 + height), radius=max(8, int(22 * S)),
                               fill=(2, 6, 16, 180), outline=(255, 169, 74, 105), width=max(1, int(2 * S)))
        canvas.alpha_composite(panel)

        draw_text(canvas, "GALACTIC CENTER TRACK", (x0 + 22 * S, y0 + 20 * S), size=21,
                  fill=(255, 195, 103, 245), bold=True, stroke=1)
        draw_text(canvas, local.strftime("%Y-%m-%d  %H:%M IST"), (x0 + 22 * S, y0 + 61 * S), size=20,
                  fill=(238, 245, 255, 235), bold=True, stroke=1)
        draw_text(canvas, f"ALTITUDE // {float(alt[0]):+05.1f}°", (x0 + 22 * S, y0 + 108 * S), size=23,
                  fill=(130, 225, 245, 245), bold=True, stroke=1)
        draw_text(canvas, f"AZIMUTH  // {float(az[0]):05.1f}°", (x0 + 22 * S, y0 + 151 * S), size=23,
                  fill=(130, 225, 245, 245), bold=True, stroke=1)
        draw_text(canvas, f"LOCAL SIDEREAL // {format_sidereal_hours(lst)}", (x0 + 22 * S, y0 + 199 * S), size=19,
                  fill=(190, 215, 235, 225), bold=True, stroke=1)
        draw_text(canvas, "RA 17h45m40s  DEC −29°00′28″", (x0 + 22 * S, y0 + 239 * S), size=17,
                  fill=(180, 202, 225, 215), stroke=1)

    def draw_start_end_ghosts(self, canvas: Image.Image, roll: float, t: float):
        alpha = int(225 * smoothstep((t - 49.5) / 2.0))
        if alpha <= 4:
            return
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        for dt, label, color in [
            (NIGHT_START_UTC, "20:30 START", (80, 210, 255)),
            (NIGHT_END_UTC, "04:30 END", (255, 175, 85)),
        ]:
            alt, az = radec_to_altaz(
                np.array([self.gc_ra]), np.array([self.gc_dec]), dt,
                float(CONFIG["latitude_deg"]), float(CONFIG["longitude_deg"]),
            )
            display_alt = alt.copy()
            below_horizon = float(alt[0]) < 0.0
            if below_horizon:
                display_alt[0] = 0.0
            x, y, visible = sky_project(display_alt, az, roll)
            if not bool(visible[0]):
                continue
            radius = 34 * S
            draw.ellipse((x[0] - radius, y[0] - radius, x[0] + radius, y[0] + radius),
                         outline=(*color, alpha), width=max(1, int(3 * S)))
            final_label = f"{label} // BELOW HORIZON" if below_horizon else label
            label_y = y[0] - radius - 17 * S if not below_horizon else y[0] - radius - 35 * S
            draw_text(layer, final_label, (x[0], label_y), size=17 if below_horizon else 18,
                      fill=(*color, alpha), bold=True, anchor="ma", stroke=1)
            if below_horizon:
                draw.line((x[0], y[0] + radius, x[0], y[0] + radius + 25 * S),
                          fill=(*color, int(alpha * 0.75)), width=max(1, int(2 * S)))
        canvas.alpha_composite(layer)

    def draw_timeline(self, canvas: Image.Image, dt: datetime):
        x0 = int(70 * SX)
        x1 = int(1010 * SX)
        y = int(1590 * SY)
        width = x1 - x0
        fraction = clamp((dt - NIGHT_START_UTC).total_seconds() / NIGHT_DURATION_SECONDS)
        cursor_x = x0 + fraction * width

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.line((x0, y, x1, y), fill=(95, 180, 215, 185), width=max(1, int(2 * S)))

        for index in range(9):
            x = x0 + index * width / 8.0
            draw.line((x, y - 8 * S, x, y + 8 * S), fill=(125, 190, 218, 150), width=max(1, int(1 * S)))
        draw.line((cursor_x, y - 24 * S, cursor_x, y + 24 * S), fill=(255, 177, 80, 245), width=max(1, int(3 * S)))

        local = dt.astimezone(IST)
        draw_text(layer, "ONE NIGHT AT HANLE", (x0, y - 60 * S), size=20,
                  fill=(175, 218, 235, 220), bold=True, stroke=1)
        draw_text(layer, local.strftime("%H:%M IST"), (x1, y - 63 * S), size=29,
                  fill=(245, 250, 255, 245), bold=True, anchor="ra", stroke=2)
        draw_text(layer, "20:30", (x0, y + 27 * S), size=17,
                  fill=(160, 195, 215, 210), stroke=1)
        draw_text(layer, "04:30", (x1, y + 27 * S), size=17,
                  fill=(160, 195, 215, 210), anchor="ra", stroke=1)
        canvas.alpha_composite(layer)

    def draw_title_and_captions(self, canvas: Image.Image, t: float, dt: datetime):
        title_alpha = int(255 * smoothstep((t - 0.3) / 1.0) * (1.0 - smoothstep((t - 6.0) / 1.0)))
        if title_alpha > 4:
            draw_text(canvas, CONFIG["title_text"], (55 * SX, 94 * SY), size=47,
                      fill=(245, 250, 255, title_alpha), bold=True, stroke=2)
            draw_text(canvas, CONFIG["subtitle_text"], (59 * SX, 226 * SY), size=24,
                      fill=(105, 225, 245, min(225, title_alpha)), bold=True, stroke=1)

        if t > 6.0:
            shot_labels = {
                "motion": "DIURNAL MOTION // EARTH ROTATION",
                "coordinates": "REAL SKY COORDINATES // HANLE",
                "galactic_center": "SAGITTARIUS A* // REAL TRACK",
                "trails": "LONG-EXPOSURE VIEW // SIDEREAL ARCS",
                "comparison": "START POSITION // END POSITION",
            }
            label = shot_labels.get(self.current_shot["name"], "REAL-SKY SIMULATION")
            draw_text(canvas, label, (55 * SX, 70 * SY), size=20,
                      fill=(145, 215, 235, 210), bold=True, stroke=1)

        caption = None
        for start, end, text in CAPTIONS:
            if start <= t < end:
                caption = text
                break
        if caption:
            y0 = int(1680 * SY)
            panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            draw = ImageDraw.Draw(panel)
            draw.rounded_rectangle((45 * SX, y0, 1035 * SX, y0 + 135 * SY),
                                   radius=max(8, int(24 * S)), fill=(1, 4, 13, 182),
                                   outline=(65, 170, 210, 75), width=max(1, int(1 * S)))
            canvas.alpha_composite(panel)
            draw_wrapped_text(canvas, caption, (72 * SX, y0 + 28 * SY), 925,
                              size=29, fill=(245, 250, 255, 245))

        note_alpha = int(220 * smoothstep((t - 51.0) / 3.0))
        if note_alpha > 4:
            draw_text(canvas, CONFIG["credit_text"], (60 * SX, 1843 * SY), size=16,
                      fill=(210, 225, 240, note_alpha), stroke=1)
            draw_wrapped_text(canvas, CONFIG["scientific_note"], (60 * SX, 1870 * SY), 955,
                              size=15, fill=(180, 202, 224, note_alpha), line_spacing=4)

    def draw_scanlines(self, canvas: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        spacing = max(4, int(float(CONFIG["scanline_spacing"]) * S))
        offset = int((t * 31.0) % spacing)
        for y in range(offset, OUT_SIZE[1], spacing):
            draw.line((0, y, OUT_SIZE[0], y), fill=(105, 180, 220, 10), width=1)
        scan_y = int((t * 134.0) % (OUT_SIZE[1] + 220 * S)) - 110 * S
        draw.rectangle((0, scan_y, OUT_SIZE[0], scan_y + 42 * S), fill=(90, 190, 225, 7))
        canvas.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        story_t = self.story_time(t)
        shot, time_fraction, roll, gain = self.shot_state(story_t)
        self.current_shot = shot
        self.current_time_fraction = time_fraction
        dt = self.sky_datetime(time_fraction)

        canvas = self.render_background(story_t, time_fraction)
        self.draw_milky_way_band(canvas, dt, roll, gain, story_t)
        self.draw_star_trails(canvas, dt, roll, story_t)
        self.draw_named_stars(canvas, dt, roll, story_t)
        self.draw_galactic_center_track(canvas, dt, roll, story_t)
        self.draw_start_end_ghosts(canvas, roll, story_t)
        self.draw_meteors(canvas, story_t)
        self.draw_horizon_and_compass(canvas, story_t)
        canvas.alpha_composite(self.landscape)
        self.draw_motion_explanation(canvas, dt, story_t)
        self.draw_coordinate_panel(canvas, dt, story_t)
        self.draw_timeline(canvas, dt)
        self.draw_title_and_captions(canvas, story_t, dt)
        self.draw_scanlines(canvas, story_t)

        image = canvas.convert("RGB")
        image = ImageEnhance.Contrast(image).enhance(float(CONFIG["contrast_boost"]))
        image = ImageEnhance.Color(image).enhance(float(CONFIG["saturation_boost"]))
        arr = np.array(image, dtype=np.uint8)
        arr = np.clip(arr.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)

        fade_in = smoothstep(t / 0.9)
        fade_out = 1.0 - smoothstep((t - (float(CONFIG["duration_s"]) - 1.1)) / 1.0)
        arr = np.clip(arr.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)
        return arr


# %% [markdown]
# ## Subtitle sidecar


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
        lines.extend([
            str(index),
            f"{format_srt_time(start)} --> {format_srt_time(end)}",
            text,
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# %% [markdown]
# ## Video render and optional finishing


def run_ffmpeg(command: List[str]):
    print("Running:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def render_video(scene: MilkyWaySkyScene) -> Path:
    raw_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    subbed_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_subbed.mp4"
    audio_path_out = OUTPUT_ROOT / f"{CONFIG['output_basename']}_with_audio.mp4"
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"

    if CONFIG.get("write_subtitle_sidecar", True):
        write_srt(CAPTIONS, srt_path)
        print("Subtitle sidecar:", srt_path.resolve())

    frame_count = int(round(float(CONFIG["duration_s"]) * int(CONFIG["fps"])))
    times = np.arange(frame_count, dtype=float) / float(CONFIG["fps"])
    print(f"Rendering {frame_count:,} frames at {OUT_SIZE[0]}x{OUT_SIZE[1]} ...")

    with iio.get_writer(
        raw_path,
        fps=int(CONFIG["fps"]),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times, desc="Rendering Milky Way sky motion"):
            writer.append_data(scene.render_frame(float(t)))

    ffmpeg = find_ffmpeg()
    final_candidate = raw_path

    if CONFIG.get("burn_subtitles", False) and ffmpeg and srt_path.exists():
        command = [
            ffmpeg, "-y", "-i", str(final_candidate),
            "-vf", (
                f"subtitles={srt_path}:"
                "force_style=Fontname=DejaVu Sans,Fontsize=22,Outline=1.2,"
                "BorderStyle=3,MarginV=90"
            ),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
            str(subbed_path),
        ]
        run_ffmpeg(command)
        final_candidate = subbed_path

    configured_audio = CONFIG.get("audio_path")
    if configured_audio and Path(str(configured_audio)).exists() and ffmpeg:
        command = [
            ffmpeg, "-y", "-i", str(final_candidate), "-i", str(configured_audio),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            str(audio_path_out),
        ]
        run_ffmpeg(command)
        final_candidate = audio_path_out

    shutil.copyfile(final_candidate, final_path)
    print("Final video:", final_path.resolve())
    return final_path


# %% [markdown]
# ## Main pipeline


def main():
    print("Starting Milky Way real-sky renderer ...")
    print("Site:", CONFIG["site_name"])
    print("Night:", NIGHT_START_LOCAL.isoformat(), "to", NIGHT_END_LOCAL.isoformat())

    archive_path = create_position_archive()
    print("Derived real-sky archive:", archive_path.resolve())

    scene = MilkyWaySkyScene()

    preview_times = [1.0, 10.0, 22.0, 34.0, 45.0, float(CONFIG["duration_s"]) - 1.0]
    for preview_time in tqdm(preview_times, desc="Full-resolution preview frames"):
        frame = scene.render_frame(float(preview_time))
        Image.fromarray(frame).save(PREVIEW_DIR / f"preview_{int(preview_time):02d}s.png")
    print("Preview frames:", PREVIEW_DIR.resolve())

    render_video(scene)
    print("Output directory:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()


# %% [markdown]
# ## Suggested narration
#
# The Milky Way is moving across the sky.
# At least, that is what it looks like from the ground.
# This is one real night at Hanle, compressed into less than a minute.
# The stars keep almost the same celestial coordinates.
# But Earth's rotation continuously changes their altitude and azimuth.
# The Galactic Center rises, arcs across the southern sky, and moves west.
# In a long exposure, the same rotation becomes curved star trails.
# The galaxy did not circle us overnight.
# We rotated beneath it.
#
# Suggested Shorts caption:
#
# One new-moon night at Hanle, Ladakh, simulated from fixed J2000 celestial
# coordinates and local sidereal time. The Milky Way's apparent motion is Earth's
# rotation made visible.
#
# #MilkyWay #NightSky #Astrophotography #Astronomy #Space #Python #Hanle
