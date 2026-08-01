from __future__ import annotations

"""
ONE LIGHT-DAY: VOYAGER'S LAST CONVERSATION
===========================================
Result : https://www.youtube.com/shorts/-icqeoWHAkQ
SCIENTIFIC FOUNDATION
---------------------
At runtime, the script asks NASA/JPL Horizons for Voyager 1 vectors relative to
both the Sun and Earth at the selected epoch. The Earth-relative range is used
to calculate one-way radio light time. If Horizons is unavailable, a clearly
labelled approximation is generated from NASA's published one-light-day
milestone date and a representative Voyager 1 cruise rate.

The following are data-driven:
    - selected epoch
    - Voyager 1 Earth-relative and heliocentric range
    - one-way light time

The following are artistic visualisations:
    - camera paths and compressed chronology
    - relative object sizes
    - heliosphere/heliopause geometry
    - star, nebula, plasma, and galaxy fields
    - soundtrack and radio-wave graphics

A self-contained cinematic micro-documentary renderer written in Python.
The default output is a 60-second vertical film (1080x1920, 24 fps) designed
for YouTube Shorts, Reels, and TikTok. Wide and square masters are supported.

Official references:
    NASA/JPL Horizons API: https://ssd-api.jpl.nasa.gov/doc/horizons.html
    NASA Voyager mission: https://science.nasa.gov/mission/voyager/
    Voyager current status: https://science.nasa.gov/mission/voyager/where-are-voyager-1-and-voyager-2-now/

INSTALL
-------
    pip install numpy pillow imageio imageio-ffmpeg requests tqdm

FULL RENDER
-----------
    python the_last_signal_one_light_day_cinematic.py

FAST VALIDATION RENDER
----------------------
    COSMIC_QUICK=1 COSMIC_OFFLINE=1 python the_last_signal_one_light_day_cinematic.py

PREVIEWS ONLY
-------------
    COSMIC_PREVIEW_ONLY=1 python the_last_signal_one_light_day_cinematic.py

FORMATS
-------
    COSMIC_FORMAT=vertical  # 1080x1920, default
    COSMIC_FORMAT=wide      # 1920x1080
    COSMIC_FORMAT=square    # 1080x1080

OPTIONAL EXTERNAL AUDIO
-----------------------
    COSMIC_MUSIC=/path/to/music.wav python the_last_signal_one_light_day_cinematic.py
    COSMIC_VOICEOVER=/path/to/voiceover.wav python the_last_signal_one_light_day_cinematic.py

The renderer always creates a procedural temp score. When external music or a
voice-over is supplied, FFmpeg mixes those sources into the final master.
"""

import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry


# =============================================================================
# Configuration
# =============================================================================

QUICK_MODE = os.getenv("COSMIC_QUICK", "0") == "1"
PREVIEW_ONLY = os.getenv("COSMIC_PREVIEW_ONLY", "0") == "1"
OFFLINE_MODE = os.getenv("COSMIC_OFFLINE", "0") == "1"
FORCE_REFRESH = os.getenv("COSMIC_REFRESH", "0") == "1"
FORMAT = os.getenv("COSMIC_FORMAT", "vertical").strip().lower()
EPOCH_TEXT = os.getenv("COSMIC_EPOCH", "2026-07-31").strip()
EXTERNAL_MUSIC = os.getenv("COSMIC_MUSIC", "").strip() or None
EXTERNAL_VOICEOVER = os.getenv("COSMIC_VOICEOVER", "").strip() or None
BURN_CAPTIONS = os.getenv("COSMIC_BURN_CAPTIONS", "1") == "1"

if FORMAT not in {"vertical", "wide", "square"}:
    raise ValueError("COSMIC_FORMAT must be vertical, wide, or square")

if QUICK_MODE:
    FORMAT_SIZES = {
        "vertical": (360, 640),
        "wide": (640, 360),
        "square": (512, 512),
    }
else:
    FORMAT_SIZES = {
        "vertical": (1080, 1920),
        "wide": (1920, 1080),
        "square": (1080, 1080),
    }

WIDTH, HEIGHT = FORMAT_SIZES[FORMAT]
OUT_SIZE = (WIDTH, HEIGHT)
FPS = max(8, int(os.getenv("COSMIC_FPS", "12" if QUICK_MODE else "24")))
DEFAULT_DURATION = 12.0 if QUICK_MODE else 60.0
DURATION = max(8.0, float(os.getenv("COSMIC_DURATION", str(DEFAULT_DURATION))))
RENDER_SCALE = float(os.getenv("COSMIC_RENDER_SCALE", "0.72" if not QUICK_MODE else "0.78"))
RENDER_SCALE = min(1.0, max(0.45, RENDER_SCALE))
RW, RH = max(240, int(WIDTH * RENDER_SCALE)), max(240, int(HEIGHT * RENDER_SCALE))
RS = RW / WIDTH
FS = WIDTH / (1080.0 if FORMAT != "wide" else 1920.0)

OUTPUT_ROOT = Path(os.getenv("COSMIC_OUTPUT_DIR", "one_light_day_output"))
DATA_DIR = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
AUDIO_DIR = OUTPUT_ROOT / "audio"
for directory in (OUTPUT_ROOT, DATA_DIR, PREVIEW_DIR, AUDIO_DIR):
    directory.mkdir(parents=True, exist_ok=True)

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
LIGHT_MIN_PER_AU = 8.316746397
AU_KM = 149_597_870.7
VOYAGER_1_ID = "-31"

# Narrative beats. The overlapping edges create cinematic dissolves.
SCENES = [
    (0.000, 0.085, "signal"),
    (0.065, 0.205, "earth"),
    (0.185, 0.355, "jupiter"),
    (0.335, 0.535, "heliopause"),
    (0.510, 0.825, "voyager"),
    (0.800, 1.000, "deepfield"),
]

VOICEOVER = [
    (0.00, 4.60, "Listen."),
    (4.60, 11.80, "This pulse is part of a conversation stretched across the Solar System."),
    (11.80, 18.80, "Voyager 1 left Earth in 1977."),
    (18.80, 27.00, "Jupiter bent its path and sent it toward the outer dark."),
    (27.00, 37.80, "In 2012, it crossed the heliopause—the changing frontier where the solar wind meets interstellar space."),
    (37.80, 50.80, "Today, a one-way message takes {light_hours:.2f} hours to reach it. The reply must cross the same distance again."),
    (50.80, 60.00, "A machine from another century is still calling home."),
]


# =============================================================================
# Data acquisition
# =============================================================================


@dataclass
class VoyagerState:
    epoch: str
    earth_range_au: float
    sun_range_au: float
    earth_light_hours: float
    heliocentric_speed_kms: float
    xyz_sun_au: Tuple[float, float, float]
    source: str


@dataclass
class RenderConfig:
    title: str
    format: str
    width: int
    height: int
    fps: int
    duration_s: float
    render_scale: float
    epoch: str
    burn_captions: bool


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


def fade_window(x: float, start: float, end: float, feather: float = 0.14) -> float:
    if end <= start:
        return 0.0
    u = (x - start) / (end - start)
    return clamp(min(smoothstep(u / feather), 1.0 - smoothstep((u - (1.0 - feather)) / feather)))


def parse_epoch(text: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"COSMIC_EPOCH must look like YYYY-MM-DD, got {text!r}")


def build_retry_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.25,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_connections=3, pool_maxsize=3)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "one-light-day-cinematic-renderer/1.0 educational"})
    return session


def parse_horizons_vector(text: str) -> Tuple[np.ndarray, np.ndarray]:
    if "$$SOE" not in text or "$$EOE" not in text:
        raise RuntimeError("Horizons response has no ephemeris block")
    block = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    row = next(line.strip() for line in block.splitlines() if line.strip())
    numbers: List[float] = []
    for cell in next(csv.reader([row])):
        try:
            numbers.append(float(cell.strip()))
        except ValueError:
            continue
    if len(numbers) < 7:
        raise RuntimeError(f"Could not parse Horizons row: {row}")
    return np.asarray(numbers[1:4], dtype=float), np.asarray(numbers[4:7], dtype=float)


def fetch_vector(command: str, center: str, epoch: datetime, cache_name: str) -> Tuple[np.ndarray, np.ndarray]:
    cache_path = DATA_DIR / cache_name
    if cache_path.exists() and cache_path.stat().st_size > 100 and not FORCE_REFRESH:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        params = {
            "format": "json",
            "COMMAND": command,
            "OBJ_DATA": "NO",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "VECTORS",
            "CENTER": center,
            "START_TIME": epoch.strftime("%Y-%m-%d"),
            "STOP_TIME": (epoch + timedelta(days=1)).strftime("%Y-%m-%d"),
            "STEP_SIZE": "1 d",
            "OUT_UNITS": "AU-D",
            "REF_PLANE": "ECLIPTIC",
            "REF_SYSTEM": "ICRF",
            "VEC_TABLE": "2",
            "VEC_CORR": "NONE",
            "CSV_FORMAT": "YES",
        }
        response = build_retry_session().get(HORIZONS_URL, params=params, timeout=(10, 45))
        response.raise_for_status()
        payload = response.json()
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return parse_horizons_vector(str(payload.get("result", "")))


def fallback_voyager_state(epoch: datetime) -> VoyagerState:
    # NASA stated that Voyager 1 would reach one light-day from Earth on
    # 2026-11-18. This fallback propagates from that date using a representative
    # heliocentric cruise rate. It is explicitly labelled approximate.
    one_light_day_au = 1440.0 / LIGHT_MIN_PER_AU
    milestone = datetime(2026, 11, 18, tzinfo=timezone.utc)
    years = (epoch - milestone).total_seconds() / (365.25 * 86400.0)
    cruise_au_per_year = 3.58
    earth_range = one_light_day_au + cruise_au_per_year * years
    sun_range = earth_range - 0.35
    direction = np.asarray([0.18, -0.89, 0.42], dtype=float)
    direction /= np.linalg.norm(direction)
    xyz = direction * sun_range
    speed_kms = cruise_au_per_year * AU_KM / (365.25 * 86400.0)
    return VoyagerState(
        epoch=epoch.strftime("%Y-%m-%d"),
        earth_range_au=float(earth_range),
        sun_range_au=float(sun_range),
        earth_light_hours=float(earth_range * LIGHT_MIN_PER_AU / 60.0),
        heliocentric_speed_kms=float(speed_kms),
        xyz_sun_au=tuple(float(v) for v in xyz),
        source="approximate fallback propagated from NASA one-light-day milestone",
    )


def load_voyager_state() -> VoyagerState:
    epoch = parse_epoch(EPOCH_TEXT)
    if OFFLINE_MODE:
        return fallback_voyager_state(epoch)
    try:
        xyz_sun, vel_sun = fetch_vector(
            VOYAGER_1_ID,
            "500@10",
            epoch,
            f"voyager1_sun_{epoch.strftime('%Y%m%d')}.json",
        )
        xyz_earth, _ = fetch_vector(
            VOYAGER_1_ID,
            "500@399",
            epoch,
            f"voyager1_earth_{epoch.strftime('%Y%m%d')}.json",
        )
        earth_range = float(np.linalg.norm(xyz_earth))
        sun_range = float(np.linalg.norm(xyz_sun))
        speed_kms = float(np.linalg.norm(vel_sun) * AU_KM / 86400.0)
        return VoyagerState(
            epoch=epoch.strftime("%Y-%m-%d"),
            earth_range_au=earth_range,
            sun_range_au=sun_range,
            earth_light_hours=earth_range * LIGHT_MIN_PER_AU / 60.0,
            heliocentric_speed_kms=speed_kms,
            xyz_sun_au=tuple(float(v) for v in xyz_sun),
            source="NASA/JPL Horizons live vectors",
        )
    except Exception as error:
        print(f"Horizons unavailable; using labelled fallback: {error}")
        return fallback_voyager_state(epoch)


# =============================================================================
# Drawing helpers
# =============================================================================


def get_font(size: int, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont:
    size = max(10, int(size))
    candidates: List[str] = []
    if serif:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf",
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ])
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[float, float],
    size: int,
    fill: Tuple[int, int, int, int] = (245, 248, 255, 255),
    bold: bool = False,
    serif: bool = False,
    anchor: str = "la",
    stroke: int = 1,
) -> None:
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold=bold, serif=serif),
        fill=fill,
        anchor=anchor,
        stroke_width=max(0, int(stroke)),
        stroke_fill=(0, 0, 0, min(230, fill[3])),
    )


def alpha_at(base: Image.Image, overlay: Image.Image, x: float, y: float) -> None:
    base.alpha_composite(overlay.convert("RGBA"), dest=(int(x), int(y)))


def normalize(vector: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vector))
    return vector / n if n > 1e-9 else np.zeros_like(vector)


def fractal_noise(width: int, height: int, seed: int, octaves: Sequence[int] = (5, 12, 28, 64)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.zeros((height, width), dtype=np.float32)
    total = 0.0
    for index, cells in enumerate(octaves):
        sw = max(2, int(cells * width / max(width, height)))
        sh = max(2, int(cells))
        raw = rng.random((sh, sw), dtype=np.float32)
        layer = np.asarray(
            Image.fromarray(np.uint8(raw * 255), "L").resize((width, height), Image.Resampling.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        weight = 0.57 ** index
        result += layer * weight
        total += weight
    result /= max(total, 1e-9)
    result -= result.min()
    result /= max(float(result.max()), 1e-8)
    return result


def make_planet_texture(kind: str, width: int = 1024, height: int = 512) -> np.ndarray:
    seed = int(hashlib.sha256(kind.encode("utf-8")).hexdigest()[:8], 16)
    n1 = fractal_noise(width, height, seed)
    n2 = fractal_noise(width, height, seed + 41, (7, 18, 41, 91))
    yy, xx = np.mgrid[0:height, 0:width]
    lat = (yy / max(height - 1, 1) - 0.5) * math.pi
    lon = (xx / max(width - 1, 1) - 0.5) * 2.0 * math.pi

    if kind == "earth":
        continents = (0.62 * n1 + 0.24 * np.sin(2.2 * lon + 3.2 * n2) * np.cos(1.5 * lat) + 0.14 * n2) > 0.54
        ice = np.clip((np.abs(lat) - 1.13) / 0.28, 0.0, 1.0)
        ocean = np.stack((10 + 24 * n2, 37 + 64 * n1, 88 + 94 * n2), axis=-1)
        green = np.stack((42 + 78 * n2, 76 + 82 * n1, 35 + 45 * n2), axis=-1)
        desert = np.stack((145 + 70 * n1, 105 + 68 * n2, 48 + 35 * n1), axis=-1)
        dry = np.clip((n2 - 0.52) * 4.0, 0.0, 1.0)[..., None]
        land = green * (1.0 - dry) + desert * dry
        rgb = np.where(continents[..., None], land, ocean)
        rgb = rgb * (1.0 - ice[..., None]) + 242.0 * ice[..., None]
        clouds = np.clip((fractal_noise(width, height, seed + 99, (8, 21, 49, 105)) - 0.58) * 4.0, 0.0, 1.0)
        lights = np.clip((n2 - 0.72) * 5.0, 0.0, 1.0) * continents
    elif kind == "jupiter":
        bands = 0.5 + 0.5 * np.sin(24.0 * lat + 2.3 * n1 + 0.7 * np.sin(5.0 * lon))
        turbulence = np.clip(0.58 * bands + 0.42 * n2, 0.0, 1.0)
        rgb = np.stack((132 + 112 * turbulence, 82 + 124 * bands, 52 + 108 * n1), axis=-1)
        spot = np.exp(-(((lon - 0.82) / 0.34) ** 2 + ((lat + 0.36) / 0.13) ** 2))
        rgb = rgb * (1.0 - 0.70 * spot[..., None]) + np.array([222, 78, 42])[None, None, :] * 0.70 * spot[..., None]
        clouds = np.clip((n2 - 0.81) * 3.0, 0.0, 0.22)
        lights = np.zeros_like(n1)
    else:
        rgb = np.stack((70 + 120 * n1, 65 + 100 * n2, 90 + 105 * n1), axis=-1)
        clouds = np.zeros_like(n1)
        lights = np.zeros_like(n1)

    return np.dstack((np.clip(rgb, 0, 255).astype(np.uint8), np.uint8(clouds * 255), np.uint8(lights * 255)))


# =============================================================================
# Cinematic renderer
# =============================================================================


class CinematicRenderer:
    def __init__(self, state: VoyagerState):
        self.state = state
        self.rng = np.random.default_rng(771_911)
        self.earth_texture = make_planet_texture("earth", 1024 if QUICK_MODE else 1536, 512 if QUICK_MODE else 768)
        self.jupiter_texture = make_planet_texture("jupiter", 1024 if QUICK_MODE else 1536, 512 if QUICK_MODE else 768)
        self.star_plate = self._make_star_plate()
        self.nebula_plate = self._make_nebula_plate()
        self.deep_field = self._make_deep_field()
        self.vignette = self._make_vignette()
        self.grain = self.rng.normal(0.0, 1.0, (HEIGHT, WIDTH)).astype(np.float32)

    def _make_star_plate(self) -> Image.Image:
        pad_x, pad_y = int(RW * 0.10), int(RH * 0.10)
        image = Image.new("RGBA", (RW + 2 * pad_x, RH + 2 * pad_y), (0, 1, 7, 255))
        draw = ImageDraw.Draw(image)
        count = 1100 if QUICK_MODE else int(4200 * max(0.65, RENDER_SCALE))
        for _ in range(count):
            x = self.rng.uniform(0, image.width)
            y = self.rng.uniform(0, image.height)
            lum = self.rng.lognormal(-0.45, 0.90)
            r = clamp(0.35 + lum * 0.42, 0.35, 2.2) * max(0.65, RENDER_SCALE)
            temp = self.rng.uniform(0.0, 1.0)
            colour = (int(155 + 100 * temp), int(174 + 70 * temp), int(220 + 35 * temp), int(clamp(55 + lum * 66, 30, 235)))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=colour)
        glow = image.filter(ImageFilter.GaussianBlur(max(0.4, 0.9 * RENDER_SCALE)))
        glow.putalpha(glow.getchannel("A").point(lambda p: p // 3))
        image.alpha_composite(glow)
        return image

    def _make_nebula_plate(self) -> Image.Image:
        sw, sh = max(180, RW // 4), max(180, RH // 4)
        n1 = fractal_noise(sw, sh, 8801, (4, 9, 22, 51))
        n2 = fractal_noise(sw, sh, 8802, (5, 13, 31, 72))
        yy, xx = np.mgrid[0:sh, 0:sw]
        cx, cy = sw * 0.35, sh * 0.56
        radial = np.exp(-(((xx - cx) / (sw * 0.44)) ** 2 + ((yy - cy) / (sh * 0.50)) ** 2))
        cyan = np.clip((n1 - 0.48) * 2.4, 0, 1) * radial
        violet = np.clip((n2 - 0.52) * 2.4, 0, 1) * np.roll(radial, sw // 4, axis=1)
        rgba = np.zeros((sh, sw, 4), dtype=np.uint8)
        rgba[..., 0] = np.uint8(np.clip(20 * cyan + 62 * violet, 0, 255))
        rgba[..., 1] = np.uint8(np.clip(70 * cyan + 22 * violet, 0, 255))
        rgba[..., 2] = np.uint8(np.clip(125 * cyan + 110 * violet, 0, 255))
        rgba[..., 3] = np.uint8(np.clip(62 * cyan + 58 * violet, 0, 115))
        return Image.fromarray(rgba, "RGBA").resize((RW, RH), Image.Resampling.BICUBIC).filter(
            ImageFilter.GaussianBlur(max(5, int(18 * RENDER_SCALE)))
        )

    def _make_deep_field(self) -> Image.Image:
        image = Image.new("RGBA", (RW, RH), (0, 0, 4, 255))
        draw = ImageDraw.Draw(image)
        count = 850 if QUICK_MODE else 3300
        for _ in range(count):
            x = self.rng.uniform(0.02, 0.98) * RW
            y = self.rng.uniform(0.02, 0.98) * RH
            size = self.rng.lognormal(-0.55, 0.80) * RENDER_SCALE
            angle = self.rng.uniform(0, math.pi)
            brightness = self.rng.lognormal(-0.45, 0.70)
            hue = self.rng.uniform(0.0, 1.0)
            colour = (
                int(120 + 110 * hue),
                int(112 + 105 * (1.0 - abs(hue - 0.45))),
                int(155 + 92 * (1.0 - hue)),
                int(clamp(28 + brightness * 47, 12, 150)),
            )
            rx = max(0.5, size * 2.6)
            ry = max(0.35, size * 0.72)
            # Pillow has no rotated ellipse primitive. Small elliptical strokes still
            # read as distant galaxies once glow and movement are applied.
            draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=colour)
            if brightness > 2.0:
                draw.ellipse((x - 0.5, y - 0.5, x + 0.5, y + 0.5), fill=(245, 236, 220, 185))
        blurred = image.filter(ImageFilter.GaussianBlur(max(0.25, 0.65 * RENDER_SCALE)))
        image = Image.blend(image, blurred, 0.38)
        return image

    def _make_vignette(self) -> np.ndarray:
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
        nx = (xx - WIDTH / 2.0) / (WIDTH / 2.0)
        ny = (yy - HEIGHT / 2.0) / (HEIGHT / 2.0)
        radius = np.sqrt(nx * nx + ny * ny)
        return np.clip(1.0 - 0.48 * radius ** 1.7, 0.34, 1.0).astype(np.float32)

    def _canvas(self, t: float, nebula: float = 0.0, star_brightness: float = 1.0) -> Image.Image:
        pad_x = (self.star_plate.width - RW) // 2
        pad_y = (self.star_plate.height - RH) // 2
        dx = int(math.sin(t * 0.043) * pad_x * 0.82)
        dy = int(math.cos(t * 0.037) * pad_y * 0.72)
        crop = self.star_plate.crop((pad_x + dx, pad_y + dy, pad_x + dx + RW, pad_y + dy + RH)).copy()
        if star_brightness != 1.0:
            crop = ImageEnhance.Brightness(crop).enhance(star_brightness)
        if nebula > 0.001:
            layer = self.nebula_plate.copy()
            layer.putalpha(layer.getchannel("A").point(lambda p: int(p * nebula)))
            crop.alpha_composite(layer)
        return crop

    @lru_cache(maxsize=160)
    def _planet_patch_cached(self, kind: str, radius: int, rotation_key: int, light_key: int) -> Image.Image:
        texture = self.earth_texture if kind == "earth" else self.jupiter_texture
        rotation = rotation_key / 180.0 * math.pi
        light_angle = light_key / 180.0 * math.pi
        radius = max(8, int(radius))
        size = radius * 2 + 8
        yy, xx = np.mgrid[0:size, 0:size]
        nx = (xx - size / 2 + 0.5) / radius
        ny = (yy - size / 2 + 0.5) / radius
        rr = nx * nx + ny * ny
        inside = rr <= 1.0
        z = np.sqrt(np.maximum(1.0 - rr, 0.0))
        lon = np.arctan2(nx, z) + rotation
        lat = np.arcsin(np.clip(-ny, -1.0, 1.0))
        th, tw = texture.shape[:2]
        tx = ((lon / (2.0 * math.pi) + 0.5) % 1.0 * (tw - 1)).astype(np.int32)
        ty = np.clip((lat / math.pi + 0.5) * (th - 1), 0, th - 1).astype(np.int32)
        sampled = texture[ty, tx]
        rgb = sampled[..., :3].astype(np.float32)
        cloud = sampled[..., 3].astype(np.float32) / 255.0
        city = sampled[..., 4].astype(np.float32) / 255.0

        light = normalize(np.array([math.cos(light_angle), -0.17, math.sin(light_angle)]))
        ndotl = nx * light[0] + ny * light[1] + z * light[2]
        illum = 0.045 + 0.955 * np.clip((ndotl + 0.08) / 1.08, 0.0, 1.0)
        limb = np.clip(z, 0.0, 1.0) ** 0.27
        rgb *= illum[..., None] * (0.70 + 0.30 * limb[..., None])
        spec = np.clip(ndotl, 0.0, 1.0) ** 32
        rgb += spec[..., None] * (34.0 if kind == "earth" else 16.0)

        if kind == "earth":
            # Night-side city-light texture. It is deliberately subtle.
            night = np.clip(0.26 - illum, 0.0, 0.26) / 0.26
            rgb += city[..., None] * night[..., None] * np.array([255.0, 151.0, 48.0])[None, None, :] * 0.52
            cloud_lit = np.clip(illum * (0.82 + 0.18 * z), 0.0, 1.0)
            rgb = rgb * (1.0 - 0.66 * cloud[..., None]) + 252.0 * cloud[..., None] * cloud_lit[..., None]

        edge = np.clip((1.0 - np.sqrt(np.maximum(rr, 0.0))) / 0.085, 0.0, 1.0)
        rim = (1.0 - edge) * inside
        atmosphere = np.array([73.0, 170.0, 255.0]) if kind == "earth" else np.array([255.0, 196.0, 126.0])
        rgb += atmosphere[None, None, :] * rim[..., None] * (0.82 if kind == "earth" else 0.16)
        alpha = np.where(inside, 255.0, np.clip((1.06 - np.sqrt(rr)) / 0.06 * 255.0, 0.0, 255.0))

        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        rgba[..., 3] = np.uint8(alpha)
        return Image.fromarray(rgba, "RGBA")

    def _draw_planet(self, canvas: Image.Image, kind: str, center: Tuple[float, float], radius: float, t: float, light_angle: float) -> None:
        radius_i = max(8, int(radius))
        rotation_key = int(round((t * (12.0 if kind == "jupiter" else 4.0)) % 360.0 / 2.0) * 2)
        light_key = int(round(math.degrees(light_angle) / 3.0) * 3)
        patch = self._planet_patch_cached(kind, radius_i, rotation_key, light_key).copy()
        glow = patch.filter(ImageFilter.GaussianBlur(max(2, int(radius * 0.10))))
        glow.putalpha(glow.getchannel("A").point(lambda p: int(p * (0.24 if kind == "earth" else 0.12))))
        x = center[0] - patch.width / 2
        y = center[1] - patch.height / 2
        alpha_at(canvas, glow, x, y)
        alpha_at(canvas, patch, x, y)

    def _draw_sun(self, canvas: Image.Image, center: Tuple[float, float], radius: float, alpha: float = 1.0) -> None:
        layer = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx, cy = center
        for mul, a in ((6.0, 8), (4.0, 16), (2.5, 34), (1.55, 80)):
            r = radius * mul
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 205, 95, int(a * alpha)))
        layer = layer.filter(ImageFilter.GaussianBlur(max(3, int(radius * 0.82))))
        canvas.alpha_composite(layer)
        core = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(core)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 239, 188, int(255 * alpha)))
        draw.ellipse((cx - radius * 0.55, cy - radius * 0.55, cx + radius * 0.55, cy + radius * 0.55), fill=(255, 255, 244, int(240 * alpha)))
        canvas.alpha_composite(core)

    def _draw_voyager(self, canvas: Image.Image, center: Tuple[float, float], scale: float, angle: float, alpha: float = 1.0) -> None:
        size = max(64, int(scale * 5.8))
        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx = cy = size / 2
        s = scale
        a = int(255 * clamp(alpha))

        # Dish, bus, booms, and RTGs. The silhouette is designed for immediate
        # recognition at phone size rather than engineering-diagram completeness.
        dish = (cx - 1.18 * s, cy - 0.62 * s, cx + 1.18 * s, cy + 0.72 * s)
        draw.pieslice(dish, 184, 356, fill=(206, 211, 217, a), outline=(248, 250, 252, a))
        draw.arc(dish, 184, 356, fill=(255, 255, 255, a), width=max(1, int(0.07 * s)))
        draw.line((cx, cy + 0.05 * s, cx, cy + 0.90 * s), fill=(190, 198, 208, a), width=max(1, int(0.10 * s)))
        draw.rounded_rectangle((cx - 0.36 * s, cy + 0.72 * s, cx + 0.36 * s, cy + 1.23 * s), radius=max(2, int(0.08 * s)), fill=(88, 99, 114, a), outline=(220, 228, 238, a), width=max(1, int(0.05 * s)))
        draw.line((cx + 0.28 * s, cy + 1.00 * s, cx + 1.86 * s, cy + 1.40 * s), fill=(175, 184, 196, a), width=max(1, int(0.085 * s)))
        for index in range(3):
            x0 = cx + (1.15 + 0.30 * index) * s
            y0 = cy + (1.20 + 0.08 * index) * s
            draw.rectangle((x0, y0, x0 + 0.24 * s, y0 + 0.38 * s), fill=(47, 50, 57, a), outline=(168, 176, 188, a))
        draw.line((cx - 0.12 * s, cy + 0.96 * s, cx - 1.90 * s, cy + 1.72 * s), fill=(164, 177, 194, a), width=max(1, int(0.055 * s)))
        draw.ellipse((cx - 2.02 * s, cy + 1.60 * s, cx - 1.78 * s, cy + 1.84 * s), fill=(215, 224, 235, a))
        draw.line((cx - 0.06 * s, cy + 0.78 * s, cx - 0.62 * s, cy - 1.26 * s), fill=(132, 151, 176, a), width=max(1, int(0.045 * s)))
        layer = layer.rotate(math.degrees(angle), resample=Image.Resampling.BICUBIC, expand=False)
        glow = layer.filter(ImageFilter.GaussianBlur(max(1, int(0.11 * s))))
        glow.putalpha(glow.getchannel("A").point(lambda p: int(p * 0.24)))
        x = center[0] - size / 2
        y = center[1] - size / 2
        alpha_at(canvas, glow, x, y)
        alpha_at(canvas, layer, x, y)

    def _draw_radio_arcs(self, canvas: Image.Image, origin: Tuple[float, float], t: float, strength: float, direction: str = "left") -> None:
        layer = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        ox, oy = origin
        for index in range(10):
            phase = (t * 0.18 + index / 10.0) % 1.0
            radius = phase * max(RW, RH) * 0.72
            alpha = int(90 * (1.0 - phase) * strength)
            box = (ox - radius, oy - radius, ox + radius, oy + radius)
            if direction == "left":
                draw.arc(box, 138, 222, fill=(104, 202, 251, alpha), width=max(1, int(2 * RENDER_SCALE)))
            else:
                draw.arc(box, -42, 42, fill=(104, 202, 251, alpha), width=max(1, int(2 * RENDER_SCALE)))
        canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(0.2, 0.55 * RENDER_SCALE))))

    def _draw_warp_stars(self, canvas: Image.Image, amount: float, t: float) -> None:
        if amount <= 0.01:
            return
        layer = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        rng = np.random.default_rng(91_000 + int(t * 4))
        cx, cy = RW * 0.50, RH * 0.50
        for _ in range(430 if not QUICK_MODE else 130):
            angle = rng.uniform(0, 2 * math.pi)
            radius = rng.uniform(0.08, 0.74) * math.hypot(RW, RH)
            x = cx + math.cos(angle) * radius
            y = cy + math.sin(angle) * radius
            length = amount * rng.uniform(8.0, 46.0) * RENDER_SCALE
            x0 = x - math.cos(angle) * length
            y0 = y - math.sin(angle) * length
            alpha = int(rng.uniform(18, 115) * amount)
            draw.line((x0, y0, x, y), fill=(155, 205, 249, alpha), width=max(1, int(rng.uniform(0.7, 1.8) * RENDER_SCALE)))
        canvas.alpha_composite(layer)

    # ------------------------------------------------------------------ scenes

    def _scene_signal(self, u: float, t: float) -> Image.Image:
        canvas = self._canvas(t, nebula=0.0, star_brightness=0.38)
        veil = Image.new("RGBA", (RW, RH), (0, 0, 6, 175))
        canvas.alpha_composite(veil)
        layer = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx = RW * 0.50
        cy = RH * (0.48 if FORMAT != "wide" else 0.50)
        pulse = 0.58 + 0.42 * math.sin(t * 12.0) ** 2
        r = lerp(2.0, 6.0, smootherstep(u)) * RENDER_SCALE
        for mul, a in ((8.0, 12), (5.0, 28), (2.4, 75)):
            rr = r * mul
            draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(115, 207, 255, int(a * pulse)))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(240, 251, 255, int(245 * pulse)))
        layer = layer.filter(ImageFilter.GaussianBlur(max(0.4, 1.0 * RENDER_SCALE)))
        canvas.alpha_composite(layer)
        self._draw_radio_arcs(canvas, (cx, cy), t, strength=smoothstep(u / 0.50), direction="left")
        self._draw_radio_arcs(canvas, (cx, cy), t + 0.22, strength=smoothstep(u / 0.50), direction="right")
        return canvas

    def _scene_earth(self, u: float, t: float) -> Image.Image:
        canvas = self._canvas(t, nebula=0.05, star_brightness=0.78)
        if FORMAT == "vertical":
            radius = lerp(RW * 0.22, RW * 0.62, smootherstep(u))
            center = (lerp(RW * 0.73, RW * 0.54, smootherstep(u)), lerp(RH * 0.74, RH * 1.02, smootherstep(u)))
        else:
            radius = lerp(RH * 0.25, RH * 0.62, smootherstep(u))
            center = (lerp(RW * 0.78, RW * 0.64, u), lerp(RH * 0.65, RH * 0.92, u))
        self._draw_planet(canvas, "earth", center, radius, t, light_angle=0.72)
        # A clean signal line visually connects home to the next act.
        overlay = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        start = (center[0] - radius * 0.60, center[1] - radius * 0.50)
        end = (RW * 0.18, RH * 0.18)
        q = smootherstep(u)
        x = lerp(start[0], end[0], q)
        y = lerp(start[1], end[1], q)
        draw.line((start[0], start[1], x, y), fill=(110, 205, 251, 94), width=max(1, int(2 * RENDER_SCALE)))
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(219, 245, 255, 180))
        canvas.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(max(0.2, 0.45 * RENDER_SCALE))))
        return canvas

    def _scene_jupiter(self, u: float, t: float) -> Image.Image:
        canvas = self._canvas(t, nebula=0.10, star_brightness=0.82)
        if FORMAT == "vertical":
            radius = lerp(RW * 0.26, RW * 0.58, math.sin(math.pi * clamp(u)) ** 0.75)
            cx = lerp(-radius * 0.25, RW + radius * 0.30, smootherstep(u))
            cy = RH * (0.48 + 0.09 * math.sin(u * math.pi))
        else:
            radius = lerp(RH * 0.31, RH * 0.58, math.sin(math.pi * clamp(u)) ** 0.75)
            cx = lerp(-radius * 0.25, RW + radius * 0.30, smootherstep(u))
            cy = RH * 0.56
        self._draw_planet(canvas, "jupiter", (cx, cy), radius, t, light_angle=0.62)
        overlay = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        points: List[Tuple[float, float]] = []
        for i in range(100):
            q = i / 99.0
            x = lerp(RW * 0.08, RW * 0.92, q)
            y = RH * (0.77 - 0.30 * math.sin(q * math.pi) - 0.10 * math.sin(q * 2.0 * math.pi + 0.6))
            points.append((x, y))
        visible_count = max(2, int(len(points) * smootherstep(u)))
        draw.line(points[:visible_count], fill=(105, 199, 249, 105), width=max(1, int(2.2 * RENDER_SCALE)))
        for marker in (0.22, 0.50, 0.78):
            index = min(visible_count - 1, int(marker * (len(points) - 1)))
            if index > 0:
                mx, my = points[index]
                draw.ellipse((mx - 2.4, my - 2.4, mx + 2.4, my + 2.4), fill=(218, 244, 255, 150))
        canvas.alpha_composite(overlay)
        probe_index = min(visible_count - 1, int(0.62 * (visible_count - 1)))
        px, py = points[max(0, probe_index)]
        self._draw_voyager(canvas, (px, py), max(6, RW * 0.012), angle=-0.45, alpha=0.95)
        self._draw_warp_stars(canvas, amount=0.17 * math.sin(math.pi * u), t=t)
        return canvas

    def _scene_heliopause(self, u: float, t: float) -> Image.Image:
        canvas = self._canvas(t, nebula=0.18 + 0.16 * u, star_brightness=0.88)
        sun_center = (RW * 0.50, RH * 0.52)
        self._draw_sun(canvas, sun_center, lerp(RW * 0.045, RW * 0.010, smootherstep(u)), alpha=1.0 - 0.30 * u)
        layer = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        max_r = math.hypot(RW, RH)
        for index in range(20):
            phase = index / 19.0
            r = (0.08 + phase * 0.92) * max_r * (0.44 + 0.62 * smootherstep(u))
            wobble = 1.0 + 0.028 * math.sin(index * 1.7 + t * 0.55)
            box = (sun_center[0] - r * wobble, sun_center[1] - r * 0.66, sun_center[0] + r * wobble, sun_center[1] + r * 0.66)
            a = int((34 - index * 1.15) * (0.48 + 0.52 * math.sin(index + t * 0.9) ** 2))
            draw.ellipse(box, outline=(55, 145, 237, max(4, a)), width=max(1, int((1.0 + phase * 1.6) * RENDER_SCALE)))
        # Flowing magnetic-field ribbons communicate that the boundary is a region.
        for band in range(12):
            points = []
            base_y = RH * (0.25 + band * 0.045)
            for i in range(90):
                q = i / 89.0
                x = RW * (0.42 + q * 0.72)
                y = base_y + math.sin(i * 0.20 + band * 0.7 + t * 0.65) * (4 + band * 0.7) * RENDER_SCALE
                points.append((x, y))
            draw.line(points, fill=(75, 158, 238, 10 + band), width=max(1, int(1.5 * RENDER_SCALE)))
        canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(0.4, 1.0 * RENDER_SCALE))))

        px = lerp(RW * 0.34, RW * 0.78, smootherstep(u))
        py = lerp(RH * 0.64, RH * 0.36, smootherstep(u))
        self._draw_voyager(canvas, (px, py), lerp(RW * 0.014, RW * 0.024, u), angle=-0.35, alpha=1.0)
        boundary_flash = math.exp(-((u - 0.58) / 0.08) ** 2)
        if boundary_flash > 0.01:
            flash = Image.new("RGBA", (RW, RH), (76, 169, 245, int(44 * boundary_flash)))
            canvas.alpha_composite(flash)
        self._draw_warp_stars(canvas, amount=0.45 * smoothstep((u - 0.68) / 0.30), t=t)
        return canvas

    def _scene_voyager(self, u: float, t: float) -> Image.Image:
        canvas = self._canvas(t, nebula=0.34, star_brightness=0.90)
        self._draw_sun(canvas, (RW * 0.16, RH * 0.22), lerp(RW * 0.014, RW * 0.004, u), alpha=0.82)
        if FORMAT == "vertical":
            center = (lerp(RW * 0.68, RW * 0.55, smootherstep(u)), lerp(RH * 0.58, RH * 0.46, smootherstep(u)))
            scale = lerp(RW * 0.075, RW * 0.145, smootherstep(min(u / 0.72, 1.0)))
        else:
            center = (lerp(RW * 0.72, RW * 0.62, u), RH * 0.53)
            scale = lerp(RH * 0.10, RH * 0.19, smootherstep(min(u / 0.72, 1.0)))
        self._draw_voyager(canvas, center, scale, angle=-0.26 + 0.08 * math.sin(t * 0.22), alpha=1.0)
        self._draw_radio_arcs(canvas, center, t, strength=0.92, direction="left")

        # A geometric distance line makes the abstract delay visually legible.
        overlay = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        y = RH * 0.84
        x0, x1 = RW * 0.10, RW * 0.90
        draw.line((x0, y, x1, y), fill=(104, 178, 226, 32), width=max(1, int(1.5 * RENDER_SCALE)))
        progress = smootherstep(u)
        draw.line((x0, y, lerp(x0, x1, progress), y), fill=(107, 201, 251, 135), width=max(1, int(2.5 * RENDER_SCALE)))
        for marker in np.linspace(0.0, 1.0, 7):
            x = lerp(x0, x1, marker)
            draw.line((x, y - 3 * RENDER_SCALE, x, y + 3 * RENDER_SCALE), fill=(168, 204, 230, 56), width=max(1, int(RENDER_SCALE)))
        canvas.alpha_composite(overlay)
        return canvas

    def _scene_deepfield(self, u: float, t: float) -> Image.Image:
        # A slow zoom into a deep field, then a return to one isolated point.
        zoom = lerp(1.03, 1.20, smootherstep(min(u / 0.72, 1.0)))
        new_w = int(RW * zoom)
        new_h = int(RH * zoom)
        plate = self.deep_field.resize((new_w, new_h), Image.Resampling.BICUBIC)
        left = max(0, (new_w - RW) // 2 + int(math.sin(t * 0.035) * RW * 0.02))
        top = max(0, (new_h - RH) // 2 + int(math.cos(t * 0.031) * RH * 0.02))
        canvas = plate.crop((left, top, left + RW, top + RH)).copy()
        neb = self.nebula_plate.copy()
        neb.putalpha(neb.getchannel("A").point(lambda p: int(p * (0.18 + 0.36 * u))))
        canvas.alpha_composite(neb)

        # Final loop point. It gradually becomes the same visual motif as the opening.
        loop = smoothstep((u - 0.74) / 0.26)
        if loop > 0.0:
            veil = Image.new("RGBA", (RW, RH), (0, 0, 5, int(215 * loop)))
            canvas.alpha_composite(veil)
            layer = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            cx, cy = RW * 0.50, RH * 0.48
            r = (2.4 + 2.0 * math.sin(t * 8.0) ** 2) * RENDER_SCALE
            for mul, a in ((8.0, 12), (4.0, 32), (1.0, 240)):
                rr = r * mul
                draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(190, 232, 255, int(a * loop)))
            canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(0.35, 0.9 * RENDER_SCALE))))
        return canvas

    def _render_scene(self, name: str, u: float, t: float) -> Image.Image:
        return {
            "signal": self._scene_signal,
            "earth": self._scene_earth,
            "jupiter": self._scene_jupiter,
            "heliopause": self._scene_heliopause,
            "voyager": self._scene_voyager,
            "deepfield": self._scene_deepfield,
        }[name](u, t)

    def _scene_composite(self, fraction: float, t: float) -> Image.Image:
        active: List[Tuple[Image.Image, float]] = []
        for start, end, name in SCENES:
            if start <= fraction <= end:
                u = clamp((fraction - start) / max(end - start, 1e-9))
                active.append((self._render_scene(name, u, t), fade_window(fraction, start, end, 0.12)))
        if not active:
            start, end, name = SCENES[-1]
            return self._render_scene(name, 1.0, t)
        base, base_weight = active[0]
        cumulative = base_weight
        for image, weight in active[1:]:
            mix = weight / max(cumulative + weight, 1e-9)
            base = Image.blend(base, image, clamp(mix))
            cumulative += weight
        return base

    def _caption_for_time(self, t: float) -> Optional[str]:
        for start, end, text in VOICEOVER:
            scaled_start = start / 60.0 * DURATION
            scaled_end = end / 60.0 * DURATION
            if scaled_start <= t < scaled_end:
                return text.format(light_hours=self.state.earth_light_hours)
        return None

    def _draw_interface(self, image: Image.Image, t: float) -> None:
        fraction = clamp(t / max(DURATION, 1e-9))
        draw = ImageDraw.Draw(image)

        # A compact title reveal. It never delays the opening hook.
        title_alpha = int(235 * fade_window(fraction, 0.014, 0.135, 0.24))
        if title_alpha > 0:
            if FORMAT == "vertical":
                draw_text(image, "ONE LIGHT-DAY", (WIDTH * 0.50, HEIGHT * 0.16), int(58 * FS), (244, 248, 255, title_alpha), True, True, "mm", 2)
                draw_text(image, "VOYAGER'S LAST CONVERSATION", (WIDTH * 0.50, HEIGHT * 0.205), int(18 * FS), (112, 193, 246, int(title_alpha * 0.92)), True, False, "mm", 1)
            else:
                draw_text(image, "ONE LIGHT-DAY", (WIDTH * 0.08, HEIGHT * 0.18), int(58 * FS), (244, 248, 255, title_alpha), True, True, "la", 2)
                draw_text(image, "VOYAGER'S LAST CONVERSATION", (WIDTH * 0.082, HEIGHT * 0.26), int(17 * FS), (112, 193, 246, int(title_alpha * 0.92)), True)

        # Central factual reveal.
        fact_alpha = int(245 * fade_window(fraction, 0.555, 0.825, 0.18))
        if fact_alpha > 0:
            if FORMAT == "vertical":
                y = HEIGHT * 0.70
                draw_text(image, f"{self.state.earth_light_hours:.2f} HOURS", (WIDTH * 0.50, y), int(52 * FS), (246, 250, 255, fact_alpha), True, True, "mm", 2)
                draw_text(image, "ONE-WAY SIGNAL TIME", (WIDTH * 0.50, y + 48 * FS), int(16 * FS), (118, 192, 241, int(fact_alpha * 0.94)), True, False, "mm", 1)
                draw_text(image, f"{self.state.earth_range_au:.2f} AU FROM EARTH  •  EPOCH {self.state.epoch}", (WIDTH * 0.50, y + 82 * FS), int(12 * FS), (180, 205, 226, int(fact_alpha * 0.80)), False, False, "mm", 1)
            else:
                y = HEIGHT * 0.73
                draw_text(image, f"{self.state.earth_light_hours:.2f} HOURS", (WIDTH * 0.08, y), int(48 * FS), (246, 250, 255, fact_alpha), True, True, "la", 2)
                draw_text(image, "ONE-WAY SIGNAL TIME", (WIDTH * 0.082, y + 55 * FS), int(15 * FS), (118, 192, 241, int(fact_alpha * 0.94)), True)

        # Retention-focused micro-facts, each kept to one concise line.
        micro_facts = [
            (0.215, 0.315, "JUPITER CHANGED VOYAGER'S SPEED AND DIRECTION"),
            (0.355, 0.505, "THE HELIOPAUSE IS A CHANGING REGION — NOT A WALL"),
            (0.835, 0.955, "A MACHINE FROM 1977 IS STILL CALLING HOME"),
        ]
        for start, end, text in micro_facts:
            alpha = int(220 * fade_window(fraction, start, end, 0.18))
            if alpha <= 0:
                continue
            if FORMAT == "vertical":
                draw_text(image, text, (WIDTH * 0.50, HEIGHT * 0.88), int(14 * FS), (205, 229, 246, alpha), True, False, "mm", 1)
            else:
                draw_text(image, text, (WIDTH * 0.50, HEIGHT * 0.90), int(14 * FS), (205, 229, 246, alpha), True, False, "mm", 1)

        # Burned captions are optional; an SRT is always created.
        if BURN_CAPTIONS:
            caption = self._caption_for_time(t)
            if caption:
                max_width = WIDTH * (0.84 if FORMAT == "vertical" else 0.70)
                font_size = int((26 if FORMAT == "vertical" else 24) * FS)
                font = get_font(font_size, bold=False)
                # Manual wrapping keeps subtitles stable and readable.
                words = caption.split()
                lines: List[str] = []
                current = ""
                for word in words:
                    candidate = word if not current else current + " " + word
                    box = draw.textbbox((0, 0), candidate, font=font, stroke_width=max(1, int(FS)))
                    if box[2] - box[0] <= max_width:
                        current = candidate
                    else:
                        lines.append(current)
                        current = word
                if current:
                    lines.append(current)
                line_height = int(font_size * 1.30)
                block_h = line_height * len(lines)
                bottom_margin = HEIGHT * (0.105 if FORMAT == "vertical" else 0.075)
                y0 = HEIGHT - bottom_margin - block_h
                box_pad = int(16 * FS)
                draw.rounded_rectangle(
                    (WIDTH * 0.08, y0 - box_pad, WIDTH * 0.92, y0 + block_h + box_pad),
                    radius=max(8, int(14 * FS)),
                    fill=(0, 2, 8, 132),
                )
                for index, line in enumerate(lines):
                    draw_text(image, line, (WIDTH * 0.50, y0 + index * line_height), font_size, (245, 248, 252, 246), False, False, "ma", 1)

        # Minimal progress line.
        x0, x1 = WIDTH * 0.08, WIDTH * 0.92
        y = HEIGHT * 0.965
        draw.line((x0, y, x1, y), fill=(120, 164, 202, 34), width=max(1, int(FS)))
        draw.line((x0, y, lerp(x0, x1, fraction), y), fill=(104, 195, 246, 118), width=max(1, int(2 * FS)))

    def _grade(self, image: Image.Image, t: float) -> Image.Image:
        image = ImageEnhance.Contrast(image.convert("RGB")).enhance(1.13)
        image = ImageEnhance.Color(image).enhance(1.07)
        array = np.asarray(image, dtype=np.float32)
        array *= self.vignette[..., None]
        # Deterministic moving film grain. Very low amplitude survives compression
        # without turning the image into noisy synthetic-looking footage.
        offset_x = int(t * 17) % WIDTH
        grain = np.roll(self.grain, offset_x, axis=1)
        array += grain[..., None] * (1.35 if not QUICK_MODE else 0.75)
        array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array, "RGB")

    def render_frame(self, t: float) -> np.ndarray:
        fraction = clamp(t / max(DURATION, 1e-9))
        internal = self._scene_composite(fraction, t)
        final = internal.resize(OUT_SIZE, Image.Resampling.LANCZOS).convert("RGBA")
        self._draw_interface(final, t)
        return np.asarray(self._grade(final, t), dtype=np.uint8)


# =============================================================================
# Audio, captions, metadata, and output
# =============================================================================


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000.0))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(path: Path, state: VoyagerState) -> Path:
    blocks: List[str] = []
    for index, (start, end, text) in enumerate(VOICEOVER, 1):
        start_scaled = start / 60.0 * DURATION
        end_scaled = end / 60.0 * DURATION
        blocks.extend([
            str(index),
            f"{format_srt_time(start_scaled)} --> {format_srt_time(end_scaled)}",
            text.format(light_hours=state.earth_light_hours),
            "",
        ])
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def add_event(signal: np.ndarray, sample_rate: int, time_s: float, frequency: float, amplitude: float, decay: float, pan: float = 0.0) -> None:
    start = max(0, int(time_s * sample_rate))
    if start >= len(signal):
        return
    n = len(signal) - start
    age = np.arange(n, dtype=np.float64) / sample_rate
    env = np.exp(-age / max(decay, 1e-6))
    sweep = frequency * (1.0 + 1.2 * np.exp(-age * 3.2))
    phase = 2.0 * math.pi * sweep * age
    mono = amplitude * env * (np.sin(phase) + 0.32 * np.sin(phase * 0.51 + 0.7))
    left_gain = math.sqrt((1.0 - clamp(pan, -1.0, 1.0)) * 0.5)
    right_gain = math.sqrt((1.0 + clamp(pan, -1.0, 1.0)) * 0.5)
    signal[start:, 0] += mono * left_gain
    signal[start:, 1] += mono * right_gain


def render_procedural_score(path: Path, duration: float, sample_rate: int = 48_000) -> Path:
    print("Generating cinematic temp score...")
    n = int(round(duration * sample_rate))
    times = np.arange(n, dtype=np.float64) / sample_rate
    fraction = times / max(duration, 1e-9)
    signal = np.zeros((n, 2), dtype=np.float64)

    # Slow harmonic field. It moves through timbre and register, not busy melody.
    root = 36.71 * (1.0 + 0.018 * np.sin(2.0 * math.pi * times / 31.0))
    ratios_a = np.asarray([1.0, 1.5, 2.0, 2.5])
    ratios_b = np.asarray([1.0, 4.0 / 3.0, 1.78, 2.25])
    morph = np.clip((fraction - 0.45) / 0.36, 0.0, 1.0)
    for index in range(len(ratios_a)):
        ratio = ratios_a[index] * (1.0 - morph) + ratios_b[index] * morph
        freq = root * ratio * (1.0 + 0.0025 * np.sin(times / (7.0 + index * 3.5) + index))
        phase = 2.0 * math.pi * freq * times
        amp = 0.058 / (1.0 + index * 0.72)
        signal[:, 0] += amp * np.sin(phase + index * 0.37)
        signal[:, 1] += amp * np.sin(phase * (1.0 + (index - 1.5) * 0.00030) + index * 0.61)

    # Low-frequency pulse, restrained enough to preserve narration space.
    pulse_env = np.clip(0.5 + 0.5 * np.sin(2.0 * math.pi * times / 10.5 - math.pi / 2.0), 0.0, 1.0) ** 3.0
    sub = 0.048 * np.sin(2.0 * math.pi * 29.0 * times) * pulse_env
    signal[:, 0] += sub
    signal[:, 1] += sub * 0.96

    # Wide air bed built by interpolating sparse controls—smooth, cheap, and mono-safe.
    rng = np.random.default_rng(822_019)
    control_count = max(32, int(duration * 18))
    xp = np.linspace(0, n - 1, control_count)
    air_l = np.interp(np.arange(n), xp, rng.normal(0.0, 1.0, control_count))
    air_r = np.interp(np.arange(n), xp, rng.normal(0.0, 1.0, control_count))
    warp = np.sin(math.pi * np.clip((fraction - 0.32) / 0.34, 0.0, 1.0)) ** 2
    air_amp = 0.012 + 0.020 * warp
    signal[:, 0] += air_l * air_amp
    signal[:, 1] += air_r * air_amp

    # Narrative identity events.
    events = [
        (0.00, 98.0, 0.24, 2.8, 0.0),
        (0.19 * duration, 54.0, 0.20, 3.2, -0.35),
        (0.34 * duration, 42.0, 0.24, 3.8, 0.36),
        (0.52 * duration, 72.0, 0.16, 4.4, -0.15),
        (0.81 * duration, 146.0, 0.12, 6.8, 0.22),
    ]
    for event in events:
        add_event(signal, sample_rate, *event)

    # Heliopause contrast: spectral near-silence without an abrupt digital mute.
    centre = 0.46 * duration
    width = 1.6 if duration >= 40 else 0.55
    dip = 1.0 - 0.66 * np.exp(-((times - centre) / width) ** 2)
    signal *= dip[:, None]

    # Simple multi-tap ambience using shifted copies.
    dry = signal.copy()
    for delay_s, gain, cross in ((0.19, 0.12, 0.03), (0.37, 0.08, 0.04), (0.61, 0.055, 0.05)):
        shift = int(delay_s * sample_rate)
        if shift < n:
            signal[shift:, 0] += dry[:-shift, 0] * gain + dry[:-shift, 1] * cross
            signal[shift:, 1] += dry[:-shift, 1] * gain + dry[:-shift, 0] * cross

    intro = np.sin(np.pi * 0.5 * np.clip(times / max(1.8, duration * 0.03), 0.0, 1.0))
    outro = np.sin(np.pi * 0.5 * np.clip((duration - times) / max(2.5, duration * 0.04), 0.0, 1.0))
    signal *= (intro * outro)[:, None]
    signal = np.tanh(signal * 1.20) * 0.82
    pcm = np.int16(np.clip(signal, -1.0, 1.0) * 32767)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return path


def find_ffmpeg() -> Optional[str]:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def run_ffmpeg(command: Sequence[str]) -> None:
    print("Running:", " ".join(str(x) for x in command))
    subprocess.run(list(command), check=True)


def render_previews(renderer: CinematicRenderer) -> List[Path]:
    fractions = [0.025, 0.115, 0.265, 0.435, 0.635, 0.865, 0.975]
    paths: List[Path] = []
    for index, fraction in enumerate(tqdm(fractions, desc="Preview frames"), 1):
        frame = renderer.render_frame(fraction * DURATION)
        path = PREVIEW_DIR / f"preview_{index:02d}_{fraction:0.3f}.jpg"
        Image.fromarray(frame).save(path, quality=94)
        paths.append(path)
    return paths


def create_contact_sheet(paths: Sequence[Path]) -> Optional[Path]:
    if not paths:
        return None
    images = [Image.open(path).convert("RGB") for path in paths]
    thumb_w = 260 if FORMAT == "vertical" else 380
    thumb_h = int(thumb_w * HEIGHT / WIDTH)
    columns = 4 if FORMAT == "vertical" else 3
    margin = 18
    rows = int(math.ceil(len(images) / columns))
    sheet = Image.new("RGB", (columns * thumb_w + (columns + 1) * margin, rows * thumb_h + (rows + 1) * margin), (2, 3, 10))
    for index, image in enumerate(images):
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = margin + (index % columns) * (thumb_w + margin)
        y = margin + (index // columns) * (thumb_h + margin)
        sheet.paste(thumb, (x, y))
    path = PREVIEW_DIR / "one_light_day_contact_sheet.jpg"
    sheet.save(path, quality=94)
    return path


def create_thumbnail(renderer: CinematicRenderer, state: VoyagerState) -> Path:
    frame = Image.fromarray(renderer.render_frame(DURATION * 0.665)).convert("RGBA")
    veil = Image.new("RGBA", OUT_SIZE, (0, 0, 3, 35))
    frame.alpha_composite(veil)
    if FORMAT == "vertical":
        draw_text(frame, "ALMOST", (WIDTH * 0.50, HEIGHT * 0.14), int(34 * FS), (240, 246, 252, 242), True, False, "mm", 2)
        draw_text(frame, "1 LIGHT-DAY", (WIDTH * 0.50, HEIGHT * 0.20), int(66 * FS), (255, 255, 255, 252), True, True, "mm", 2)
        draw_text(frame, "VOYAGER 1", (WIDTH * 0.50, HEIGHT * 0.255), int(20 * FS), (104, 196, 249, 242), True, False, "mm", 1)
    else:
        draw_text(frame, "ALMOST 1 LIGHT-DAY", (WIDTH * 0.07, HEIGHT * 0.18), int(64 * FS), (255, 255, 255, 252), True, True, "la", 2)
        draw_text(frame, "VOYAGER 1", (WIDTH * 0.075, HEIGHT * 0.30), int(20 * FS), (104, 196, 249, 242), True)
    path = OUTPUT_ROOT / f"thumbnail_{FORMAT}.jpg"
    frame.convert("RGB").save(path, quality=96)
    return path


def write_metadata(state: VoyagerState) -> Tuple[Path, Path, Path]:
    title = "VOYAGER IS ALMOST ONE LIGHT-DAY AWAY | Cinematic Space Documentary"
    if state.earth_light_hours >= 24.0:
        title = "VOYAGER IS ONE LIGHT-DAY AWAY | Cinematic Space Documentary"

    description = f"""A signal to Voyager 1 now takes {state.earth_light_hours:.2f} hours to cross the darkness one way.

ONE LIGHT-DAY is a cinematic micro-documentary following Voyager from Earth, past Jupiter's gravity assist, across the heliopause, and into interstellar space.

Data epoch: {state.epoch}
Voyager 1 distance from Earth: {state.earth_range_au:.3f} AU
Voyager 1 distance from the Sun: {state.sun_range_au:.3f} AU
One-way light time: {state.earth_light_hours:.3f} hours
Heliocentric speed: {state.heliocentric_speed_kms:.2f} km/s
Data source: {state.source}

Scientific note:
Voyager ranges and signal time are data-driven for the stated epoch where NASA/JPL Horizons was available. Camera paths, compressed chronology, relative sizes, heliopause shape, colour, particles, galaxies, and sound are artistic visualisations.

Official sources:
NASA/JPL Horizons API
NASA Voyager mission and current-status pages

#Voyager #Space #NASA #Interstellar #SpaceDocumentary
"""

    metadata_path = OUTPUT_ROOT / "youtube_title_description_tags.txt"
    metadata_path.write_text(
        "TITLE\n" + title + "\n\nDESCRIPTION\n" + description + "\nTAGS\n" +
        "Voyager 1, Voyager one light-day, heliopause, interstellar space, NASA Voyager, "
        "cinematic space film, space documentary, astronomy short, deep space, Solar System, NASA JPL\n",
        encoding="utf-8",
    )

    config = RenderConfig(
        title="ONE LIGHT-DAY: VOYAGER'S LAST CONVERSATION",
        format=FORMAT,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        duration_s=DURATION,
        render_scale=RENDER_SCALE,
        epoch=state.epoch,
        burn_captions=BURN_CAPTIONS,
    )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "render": asdict(config),
        "voyager_state": asdict(state),
        "scientific_disclosure": (
            "Voyager distance and light-time are data-driven where Horizons was available. "
            "Camera motion, compressed chronology, object scale, heliopause geometry, colour, "
            "particles, galaxy field and soundtrack are artistic visualisations."
        ),
        "official_sources": {
            "horizons": HORIZONS_URL,
            "voyager": "https://science.nasa.gov/mission/voyager/",
            "current_status": "https://science.nasa.gov/mission/voyager/where-are-voyager-1-and-voyager-2-now/",
        },
    }
    manifest_path = DATA_DIR / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    narration_path = OUTPUT_ROOT / "voiceover_script.txt"
    narration_path.write_text(
        "\n".join(text.format(light_hours=state.earth_light_hours) for _, _, text in VOICEOVER),
        encoding="utf-8",
    )
    return metadata_path, manifest_path, narration_path


def render_video(renderer: CinematicRenderer, state: VoyagerState) -> Path:
    basename = f"one_light_day_{FORMAT}_{int(DURATION)}s"
    silent_path = OUTPUT_ROOT / f"{basename}_silent.mp4"
    final_path = OUTPUT_ROOT / f"{basename}_final.mp4"
    score_path = AUDIO_DIR / f"{basename}_temp_score.wav"
    srt_path = OUTPUT_ROOT / f"{basename}.srt"
    write_srt(srt_path, state)

    frame_count = int(round(DURATION * FPS))
    times = np.arange(frame_count, dtype=float) / FPS
    print(f"Rendering {frame_count:,} frames at {WIDTH}x{HEIGHT}, {FPS} fps...")
    with iio.get_writer(
        silent_path,
        fps=FPS,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=["-crf", "17", "-preset", "medium", "-movflags", "+faststart"],
    ) as writer:
        for t in tqdm(times, desc="Rendering ONE LIGHT-DAY"):
            writer.append_data(renderer.render_frame(float(t)))

    render_procedural_score(score_path, DURATION)
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("FFmpeg was not found; the silent render is being copied as the final file.")
        shutil.copyfile(silent_path, final_path)
        return final_path

    music_path = Path(EXTERNAL_MUSIC).expanduser() if EXTERNAL_MUSIC else score_path
    voice_path = Path(EXTERNAL_VOICEOVER).expanduser() if EXTERNAL_VOICEOVER else None
    if not music_path.exists():
        print(f"External music not found: {music_path}; using procedural temp score")
        music_path = score_path
    if voice_path is not None and not voice_path.exists():
        print(f"Voice-over not found: {voice_path}; continuing without voice-over")
        voice_path = None

    if voice_path:
        # Music is lowered under speech; loudness is kept conservative for mobile.
        filter_complex = (
            "[1:a]volume=0.60,atrim=0:" + str(DURATION) + "[music];"
            "[2:a]volume=1.15,atrim=0:" + str(DURATION) + "[voice];"
            "[music][voice]amix=inputs=2:duration=longest:dropout_transition=2,"
            "alimiter=limit=0.92[aout]"
        )
        command = [
            ffmpeg, "-y", "-i", str(silent_path), "-i", str(music_path), "-i", str(voice_path),
            "-filter_complex", filter_complex,
            "-map", "0:v:0", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
            "-shortest", "-movflags", "+faststart", str(final_path),
        ]
    else:
        command = [
            ffmpeg, "-y", "-i", str(silent_path), "-i", str(music_path),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
            "-shortest", "-movflags", "+faststart", str(final_path),
        ]
    run_ffmpeg(command)
    return final_path


def main() -> None:
    print("Starting ONE LIGHT-DAY: VOYAGER'S LAST CONVERSATION")
    print("Format:", FORMAT)
    print("Output:", f"{WIDTH}x{HEIGHT} at {FPS} fps for {DURATION:.1f} seconds")
    print("Internal render:", f"{RW}x{RH} ({RENDER_SCALE:.2f} scale)")
    print("Epoch:", EPOCH_TEXT)
    print("Quick mode:", QUICK_MODE)
    print("Offline mode:", OFFLINE_MODE)

    state = load_voyager_state()
    print("Voyager data source:", state.source)
    print(f"Earth range: {state.earth_range_au:.3f} AU")
    print(f"One-way light time: {state.earth_light_hours:.3f} hours")
    print(f"Heliocentric speed: {state.heliocentric_speed_kms:.2f} km/s")

    renderer = CinematicRenderer(state)
    metadata_path, manifest_path, narration_path = write_metadata(state)
    print("Metadata:", metadata_path.resolve())
    print("Manifest:", manifest_path.resolve())
    print("Voice-over script:", narration_path.resolve())

    previews = render_previews(renderer)
    contact_sheet = create_contact_sheet(previews)
    thumbnail = create_thumbnail(renderer, state)
    if contact_sheet:
        print("Contact sheet:", contact_sheet.resolve())
    print("Thumbnail:", thumbnail.resolve())

    if PREVIEW_ONLY:
        print("Preview-only mode complete.")
    else:
        final_path = render_video(renderer, state)
        print("Final film:", final_path.resolve())

    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if path.is_file():
            print("-", path.relative_to(OUTPUT_ROOT))


if __name__ == "__main__":
    main()
