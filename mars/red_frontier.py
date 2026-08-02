# red frontier shorts creation
from __future__ import annotations

"""
RED FRONTIER: MARS BEFORE FOOTSTEPS
===================================

A long-form cinematic space documentary renderer written in Python.
The default output is a 3.5-minute widescreen film about Mars: its ancient
water, colossal landscapes, robot explorers, and the possibility of future
human arrival.

The style goal is atmospheric, premium, and emotionally paced rather than a
fast explainer. The film is divided into eight cinematic acts:
    1. The red light in the dark
    2. Earth looks toward Mars
    3. Arrival in orbit
    4. When Mars had water
    5. Olympus Mons and Valles Marineris
    6. Dust, rovers, and the blue sunset
    7. The first human footsteps (speculative)
    8. The waiting frontier

LOCAL ASSET SUPPORT
-------------------
The renderer looks for an img folder beside the script, or at COSMIC_IMG_DIR.
It can use files such as:
    img/Mars.jpg
    img/Earth.jpg
    img/Jupiter.jpg
    img/Saturn.jpg
    img/Sun.jpg
    img/Stars.jpg
    img/Saturn ring.png

If these assets are not found, the script falls back to fully procedural
planet textures and backgrounds.


SCIENTIFIC HONESTY
------------------
This film is a scientific visualisation with cinematic storytelling.
The following are grounded in established Mars science:
    - Mars is colder and drier than Earth today
    - Ancient Mars shows evidence of river valleys, lakes, and long-lived water
    - Olympus Mons is the Solar System's largest volcano
    - Valles Marineris is one of the largest canyon systems known
    - Viking, Sojourner, Spirit, Opportunity, Curiosity, Perseverance,
      and Ingenuity are real milestones in Mars exploration

The following are artistic:
    - camera motion, colour grading, clouds, dust density, and scene timing
    - visual reconstruction of ancient Mars
    - speculative human habitat scenes
    - soundtrack and sound design

INSTALL
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

FULL RENDER
-----------
    python red_frontier_mars_cinematic.py

FAST TEST
---------
    COSMIC_QUICK=1 python red_frontier_mars_cinematic.py

PREVIEWS ONLY
-------------
    COSMIC_PREVIEW_ONLY=1 python red_frontier_mars_cinematic.py

SELECT FORMAT / DURATION / IMG FOLDER
-------------------------------------
    COSMIC_FORMAT=wide python red_frontier_mars_cinematic.py
    COSMIC_DURATION=210 python red_frontier_mars_cinematic.py
    COSMIC_IMG_DIR=/path/to/img python red_frontier_mars_cinematic.py

OPTIONAL EXTERNAL AUDIO
-----------------------
    COSMIC_MUSIC=/path/to/music.wav python red_frontier_mars_cinematic.py
    COSMIC_VOICEOVER=/path/to/voiceover.wav python red_frontier_mars_cinematic.py
"""

import hashlib
import math
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from tqdm.auto import tqdm

# =============================================================================
# Configuration
# =============================================================================

QUICK_MODE = os.getenv("COSMIC_QUICK", "0") == "1"
PREVIEW_ONLY = os.getenv("COSMIC_PREVIEW_ONLY", "0") == "1"
FORMAT = os.getenv("COSMIC_FORMAT", "wide").strip().lower()
if FORMAT not in {"wide", "vertical", "square"}:
    raise ValueError("COSMIC_FORMAT must be wide, vertical, or square")

if QUICK_MODE:
    SIZES = {
        "wide": (960, 540),
        "vertical": (540, 960),
        "square": (720, 720),
    }
else:
    SIZES = {
        "wide": (1920, 1080),
        "vertical": (1080, 1920),
        "square": (1080, 1080),
    }

WIDTH, HEIGHT = SIZES[FORMAT]
OUT_SIZE = (WIDTH, HEIGHT)
FPS = max(8, int(os.getenv("COSMIC_FPS", "12" if QUICK_MODE else "24")))
DEFAULT_DURATION = 18.0 if QUICK_MODE else 210.0
DURATION = max(12.0, float(os.getenv("COSMIC_DURATION", str(DEFAULT_DURATION))))
FS = WIDTH / (1920.0 if FORMAT == "wide" else 1080.0)
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = Path(os.getenv("COSMIC_IMG_DIR", str(SCRIPT_DIR / "img"))).expanduser().resolve()
EXTERNAL_MUSIC = os.getenv("COSMIC_MUSIC", "").strip() or None
EXTERNAL_VOICEOVER = os.getenv("COSMIC_VOICEOVER", "").strip() or None
BURN_CAPTIONS = os.getenv("COSMIC_BURN_CAPTIONS", "1") == "1"

OUTPUT_ROOT = Path(os.getenv("COSMIC_OUTPUT_DIR", "red_frontier_output"))
DATA_DIR = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
AUDIO_DIR = OUTPUT_ROOT / "audio"
for directory in (OUTPUT_ROOT, DATA_DIR, PREVIEW_DIR, AUDIO_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SCENES = [
    (0.000, 0.090, "signal"),
    (0.070, 0.200, "earth_to_mars"),
    (0.180, 0.340, "orbit"),
    (0.320, 0.500, "ancient"),
    (0.480, 0.660, "monuments"),
    (0.640, 0.800, "rovers"),
    (0.780, 0.930, "future"),
    (0.910, 1.000, "finale"),
]

CHAPTERS = [
    (0.00, "A red light in the dark"),
    (0.10, "Earth looks outward"),
    (0.21, "Arrival at Mars"),
    (0.34, "When Mars had water"),
    (0.50, "The giant monuments"),
    (0.66, "Rovers under a blue sunset"),
    (0.80, "Before footsteps"),
    (0.93, "The waiting frontier"),
]

VOICEOVER = [
    (0.0, 9.0, "Mars begins as a red point of light. But it is not just a point. It is a world with a memory."),
    (9.0, 24.0, "Seen from Earth, it drifts through the darkness like an ember. Up close, it is a desert planet, cold, dry, and magnificent."),
    (24.0, 43.0, "Yet Mars was not always the silent wasteland we know today. Across its surface, ancient valleys and sediments preserve the story of flowing water."),
    (43.0, 64.0, "There may have been rivers. Lakes. Perhaps even a northern sea. If life ever had a second chance beyond Earth, Mars may once have been one of its best opportunities."),
    (64.0, 94.0, "This world carries extremes: Olympus Mons, the largest volcano in the Solar System, and Valles Marineris, a canyon system vast enough to humble continents."),
    (94.0, 123.0, "Dust storms can wrap the planet. Sunsets glow blue in the thin atmosphere. And on this alien ground, our machines have become our first field geologists."),
    (123.0, 157.0, "Viking. Sojourner. Spirit and Opportunity. Curiosity. Perseverance. Even Ingenuity, the first aircraft to fly on another world. Each mission has turned a red dot into a real place."),
    (157.0, 188.0, "The next chapter has not happened yet. Human footprints remain imaginary. Habitats remain sketches. But Mars keeps teaching us how such a future might begin."),
    (188.0, 210.0, "A world of lost water. A laboratory of survival. A frontier waiting for footsteps. Mars is not finished. It is waiting."),
]

FACT_LINES = [
    (0.12, "Mars: the fourth planet from the Sun"),
    (0.30, "Ancient Mars shows strong evidence of past water"),
    (0.50, "Olympus Mons: the Solar System's largest volcano"),
    (0.57, "Valles Marineris: one of the largest canyon systems known"),
    (0.70, "Mars sunsets can appear blue"),
    (0.83, "No human has yet set foot on Mars"),
]

YOUTUBE_TITLE = "RED FRONTIER: Mars Before Footsteps | A Cinematic Space Documentary"

# =============================================================================
# General helpers
# =============================================================================


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


def fade_window(x: float, start: float, end: float, feather: float = 0.10) -> float:
    if end <= start:
        return 0.0
    u = (x - start) / (end - start)
    a = smoothstep(u / max(feather, 1e-6))
    b = 1.0 - smoothstep((u - (1.0 - feather)) / max(feather, 1e-6))
    return clamp(min(a, b))


def deterministic_unit(text: str, salt: str = "") -> float:
    digest = hashlib.sha256(f"{text}|{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000.0))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def get_font(size: int, bold: bool = False, serif: bool = False):
    size = max(8, int(size))
    names: List[str] = []
    if serif:
        names += [
            "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ]
    names += [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[float, float],
    size: int,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    serif: bool = False,
    anchor: str = "la",
    stroke: int = 1,
) -> None:
    draw = ImageDraw.Draw(image)
    draw.text(
        xy,
        text,
        font=get_font(max(8, int(size * FS)), bold=bold, serif=serif),
        fill=fill,
        anchor=anchor,
        stroke_width=max(0, int(stroke * FS)),
        stroke_fill=(0, 0, 0, 210),
    )


def draw_multiline_block(image: Image.Image, text: str, xy: Tuple[int, int], max_width: int, size: int, fill=(255, 255, 255, 235), line_spacing: int = 8) -> int:
    draw = ImageDraw.Draw(image)
    font = get_font(max(8, int(size * FS)))
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        box = draw.textbbox((0, 0), candidate, font=font)
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
        draw.text((x, y), line, font=font, fill=fill, stroke_width=max(0, int(FS)), stroke_fill=(0, 0, 0, 210))
        box = draw.textbbox((x, y), line, font=font)
        y += box[3] - box[1] + int(line_spacing * FS)
    return y


def alpha_composite_at(base: Image.Image, overlay: Image.Image, xy: Tuple[int, int]) -> None:
    if overlay.mode != "RGBA":
        overlay = overlay.convert("RGBA")
    base.alpha_composite(overlay, dest=(int(xy[0]), int(xy[1])))


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


def fractal_noise(width: int, height: int, seed: int, octaves: Sequence[int] = (8, 18, 42, 90)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.zeros((height, width), dtype=np.float32)
    total = 0.0
    for index, cells in enumerate(octaves):
        small_w = max(2, int(cells * width / max(width, height)))
        small_h = max(2, cells)
        noise = rng.random((small_h, small_w), dtype=np.float32)
        image = Image.fromarray(np.uint8(noise * 255), "L").resize((width, height), Image.Resampling.BICUBIC)
        layer = np.asarray(image, dtype=np.float32) / 255.0
        weight = 0.56 ** index
        result += layer * weight
        total += weight
    result /= max(total, 1e-9)
    result = (result - result.min()) / max(float(result.max() - result.min()), 1e-8)
    return result


def cover_image(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    source = ImageOps.exif_transpose(image).convert("RGB")
    scale = max(target_w / source.width, target_h / source.height)
    resized = source.resize((max(1, int(round(source.width * scale))), max(1, int(round(source.height * scale)))), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - target_w) // 2)
    top = max(0, (resized.height - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h)).convert("RGBA")


def find_asset(folder: Path, candidates: Iterable[str]) -> Optional[Path]:
    if not folder.exists():
        return None
    mapping: Dict[str, Path] = {}
    for path in folder.iterdir():
        mapping[path.name.lower()] = path
    for name in candidates:
        path = mapping.get(name.lower())
        if path and path.is_file():
            return path
    return None


# =============================================================================
# Procedural textures and renderer
# =============================================================================


def make_texture(kind: str, width: int = 1200, height: int = 600) -> np.ndarray:
    seed = int(hashlib.sha256(kind.encode("utf-8")).hexdigest()[:8], 16)
    n1 = fractal_noise(width, height, seed)
    n2 = fractal_noise(width, height, seed + 17, (5, 13, 31, 71))
    yy, xx = np.mgrid[0:height, 0:width]
    lat = (yy / max(height - 1, 1) - 0.5) * math.pi
    lon = (xx / max(width - 1, 1) - 0.5) * 2.0 * math.pi

    if kind == "mars":
        highlands = np.clip(0.58 * n1 + 0.42 * np.sin(lon * 1.7 + 2.5 * n2) * np.cos(lat * 1.3), 0.0, 1.0)
        dust = np.clip(0.5 + 0.5 * np.sin(5.0 * lon + 6.0 * n1) * np.cos(3.0 * lat), 0.0, 1.0)
        polar = np.clip((np.abs(lat) - 1.18) / 0.18, 0.0, 1.0)
        base = np.stack((112 + 120 * highlands, 42 + 78 * dust, 20 + 42 * n2), axis=-1)
        base = base * (1.0 - polar[..., None]) + np.array([228, 218, 206], dtype=np.float32)[None, None, :] * polar[..., None]
        cloud = np.clip((fractal_noise(width, height, seed + 121, (7, 17, 39, 88)) - 0.74) * 4.0, 0.0, 0.35)
    elif kind == "earth":
        continents = (0.66 * n1 + 0.34 * np.sin(lon * 2.1 + 2.4 * n2) * np.cos(lat * 1.7)) > 0.55
        ice = np.clip((np.abs(lat) - 1.08) / 0.34, 0.0, 1.0)
        ocean = np.stack((16 + 18 * n2, 49 + 48 * n1, 96 + 78 * n1), axis=-1)
        land = np.stack((68 + 76 * n2, 84 + 78 * n1, 42 + 42 * n2), axis=-1)
        desert = np.stack((133 + 80 * n2, 104 + 65 * n1, 48 + 35 * n1), axis=-1)
        dry = np.clip((n2 - 0.52) * 4.0, 0.0, 1.0)[..., None]
        land_mix = land * (1.0 - dry) + desert * dry
        base = np.where(continents[..., None], land_mix, ocean)
        base = base * (1.0 - ice[..., None]) + 245.0 * ice[..., None]
        cloud = np.clip((fractal_noise(width, height, seed + 99, (7, 17, 39, 88)) - 0.58) * 4.0, 0.0, 1.0)
    elif kind == "jupiter":
        bands = 0.5 + 0.5 * np.sin(25.0 * lat + 2.8 * n1)
        turbulence = np.clip(0.52 * bands + 0.48 * n2, 0.0, 1.0)
        base = np.stack((132 + 110 * turbulence, 76 + 135 * bands, 47 + 120 * n1), axis=-1)
        spot = np.exp(-(((lon - 0.85) / 0.34) ** 2 + ((lat + 0.37) / 0.13) ** 2))
        base = base * (1.0 - 0.68 * spot[..., None]) + np.array([220, 88, 45])[None, None, :] * 0.68 * spot[..., None]
        cloud = np.clip((n2 - 0.83) * 3.0, 0.0, 0.25)
    elif kind == "saturn":
        bands = 0.5 + 0.5 * np.sin(31.0 * lat + 2.8 * n1)
        turbulence = np.clip(0.52 * bands + 0.48 * n2, 0.0, 1.0)
        base = np.stack((166 + 82 * turbulence, 135 + 88 * bands, 89 + 94 * n1), axis=-1)
        cloud = np.clip((n2 - 0.83) * 3.0, 0.0, 0.25)
    elif kind == "sun":
        grains = 0.5 + 0.5 * np.sin(lon * 12.0 + 6.0 * n1) * np.cos(lat * 8.0)
        base = np.stack((215 + 40 * n1, 140 + 95 * grains, 45 + 25 * n2), axis=-1)
        cloud = np.clip((n2 - 0.62) * 4.5, 0.0, 0.55)
    else:
        base = np.stack((70 + 125 * n1, 70 + 110 * n2, 90 + 80 * n1), axis=-1)
        cloud = np.zeros_like(n1)

    base = np.clip(base, 0, 255).astype(np.uint8)
    return np.dstack((base, np.uint8(np.clip(cloud, 0, 1) * 255)))


@dataclass
class StarField:
    xy: np.ndarray
    depth: np.ndarray
    luminosity: np.ndarray
    temperature: np.ndarray
    phase: np.ndarray


class MarsRenderer:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(7_102_008)
        self.local_maps: Dict[str, np.ndarray] = {}
        self.ring_assets: Dict[str, Image.Image] = {}
        self.star_background: Optional[Image.Image] = None
        self.textures = {
            "mars": make_texture("mars", 900 if QUICK_MODE else 1440, 450 if QUICK_MODE else 720),
            "earth": make_texture("earth", 900 if QUICK_MODE else 1440, 450 if QUICK_MODE else 720),
            "jupiter": make_texture("jupiter", 900 if QUICK_MODE else 1440, 450 if QUICK_MODE else 720),
            "saturn": make_texture("saturn", 900 if QUICK_MODE else 1440, 450 if QUICK_MODE else 720),
            "sun": make_texture("sun", 900 if QUICK_MODE else 1440, 450 if QUICK_MODE else 720),
        }
        self._load_local_assets()
        star_count = 4200 if QUICK_MODE else (18000 if FORMAT == "wide" else 14000)
        self.stars = self._build_stars(star_count)
        self.dust = self._build_dust(800 if QUICK_MODE else 4200)
        self.nebula = self._build_nebula()
        self.vignette = self._build_vignette()
        self.thumbnail_frame: Optional[np.ndarray] = None

    def _load_local_assets(self) -> None:
        asset_candidates = {
            "Mars": ["Mars.jpg", "mars.jpg", "Mars.png", "mars.png"],
            "Earth": ["Earth.jpg", "earth.jpg", "Earth.png", "earth.png"],
            "Jupiter": ["Jupiter.jpg", "jupiter.jpg", "Jupiter.png", "jupiter.png"],
            "Saturn": ["Saturn.jpg", "saturn.jpg", "Saturn.png", "saturn.png"],
            "Sun": ["Sun.jpg", "sun.jpg", "Sun.png", "sun.png"],
            "Stars": ["Stars.jpg", "stars.jpg", "Stars.png", "stars.png"],
            "SaturnRing": ["Saturn ring.png", "saturn ring.png", "SaturnRing.png", "saturnring.png"],
        }
        found = {name: find_asset(IMG_DIR, names) for name, names in asset_candidates.items()}
        for key in ["Mars", "Earth", "Jupiter", "Saturn", "Sun"]:
            path = found[key]
            if not path:
                continue
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGBA")
            aspect = image.width / max(image.height, 1)
            target_w = 1440 if not QUICK_MODE else 900
            target_h = target_w // 2
            if aspect >= 1.55:
                mapped = image.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
                rgb = np.asarray(mapped, dtype=np.uint8)
                texture_map = np.dstack((rgb, np.zeros((target_h, target_w), dtype=np.uint8)))
            else:
                texture_map = self._photo_to_equirectangular_map(image, target_w, target_h)
            self.local_maps[key.lower()] = texture_map
        if found["Stars"]:
            with Image.open(found["Stars"]) as opened:
                self.star_background = ImageEnhance.Brightness(cover_image(opened, (int(WIDTH * 1.08), int(HEIGHT * 1.08)))).enhance(0.72)
        if found["SaturnRing"]:
            with Image.open(found["SaturnRing"]) as opened:
                self.ring_assets["saturn"] = ImageOps.exif_transpose(opened).convert("RGBA")

    def asset_manifest(self) -> Dict[str, object]:
        return {
            "img_dir": str(IMG_DIR),
            "local_maps": sorted(self.local_maps),
            "has_stars_background": self.star_background is not None,
            "ring_assets": sorted(self.ring_assets),
        }

    def _detect_disc_geometry(self, image: Image.Image) -> Tuple[np.ndarray, float, float, float]:
        source = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.asarray(source, dtype=np.uint8)
        work = rgb.astype(np.float32)
        height, width = work.shape[:2]
        patch = max(2, min(width, height) // 18)
        corners = np.concatenate([
            work[:patch, :patch].reshape(-1, 3),
            work[:patch, -patch:].reshape(-1, 3),
            work[-patch:, :patch].reshape(-1, 3),
            work[-patch:, -patch:].reshape(-1, 3),
        ], axis=0)
        background = np.median(corners, axis=0)
        difference = np.linalg.norm(work - background[None, None, :], axis=2)
        yy, xx = np.mgrid[0:height, 0:width]
        cx0, cy0 = (width - 1) / 2.0, (height - 1) / 2.0
        radius0 = np.sqrt((xx - cx0) ** 2 + (yy - cy0) ** 2)
        max_radius = min(width, height) * 0.50
        central = radius0 <= max_radius
        threshold = max(16.0, float(np.percentile(difference[central], 50)) * 0.65)
        candidate = (difference > threshold) & central
        if candidate.sum() > 0.015 * central.sum():
            xs = xx[candidate].astype(np.float32)
            ys = yy[candidate].astype(np.float32)
            cx = float(np.median(xs))
            cy = float(np.median(ys))
            detected = float(np.percentile(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2), 98.0))
        else:
            cx, cy, detected = cx0, cy0, max_radius
        detected = clamp(detected, min(width, height) * 0.30, max_radius)
        return rgb, float(cx), float(cy), float(detected)

    def _photo_to_equirectangular_map(self, image: Image.Image, target_w: int, target_h: int) -> np.ndarray:
        rgb, cx, cy, radius = self._detect_disc_geometry(image)
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:target_h, 0:target_w]
        lon = (xx / max(target_w - 1, 1) - 0.5) * (2.0 * math.pi)
        lat = (yy / max(target_h - 1, 1) - 0.5) * math.pi
        map_x = np.sin(lon) * np.cos(lat)
        map_y = -np.sin(lat)
        sx = np.clip(np.rint(cx + radius * map_x), 0, w - 1).astype(np.int32)
        sy = np.clip(np.rint(cy + radius * map_y), 0, h - 1).astype(np.int32)
        sampled = rgb[sy, sx]
        sampled = np.asarray(sampled, dtype=np.uint8)
        seam = ((sampled[:, 0, :].astype(np.float32) + sampled[:, -1, :].astype(np.float32)) * 0.5).astype(np.uint8)
        sampled[:, 0, :] = seam
        sampled[:, -1, :] = seam
        cloud = np.zeros((target_h, target_w), dtype=np.uint8)
        return np.dstack((sampled, cloud))

    def _build_stars(self, count: int) -> StarField:
        xy = self.rng.uniform(-1.0, 1.0, (count, 2)).astype(np.float32)
        depth = np.exp(self.rng.uniform(math.log(0.25), math.log(8.0), count)).astype(np.float32)
        luminosity = self.rng.lognormal(-0.35, 0.95, count).astype(np.float32)
        temperature = self.rng.uniform(0.0, 1.0, count).astype(np.float32)
        phase = self.rng.uniform(0.0, 2.0 * math.pi, count).astype(np.float32)
        return StarField(xy, depth, luminosity, temperature, phase)

    def _build_dust(self, count: int) -> np.ndarray:
        xyz = self.rng.normal(0.0, 1.0, (count, 3))
        xyz[:, 2] = self.rng.uniform(0.2, 5.0, count)
        xyz[:, :2] *= xyz[:, 2:3]
        return xyz.astype(np.float32)

    def _build_nebula(self) -> Image.Image:
        small_w, small_h = max(260, WIDTH // 5), max(160, HEIGHT // 5)
        n1 = fractal_noise(small_w, small_h, 5501, (4, 9, 22, 55))
        n2 = fractal_noise(small_w, small_h, 5502, (5, 12, 30, 67))
        yy, xx = np.mgrid[0:small_h, 0:small_w]
        radial = np.exp(-(((xx / small_w - 0.48) / 0.46) ** 2 + ((yy / small_h - 0.54) / 0.50) ** 2))
        orange = np.clip((n1 - 0.52) * 2.0, 0, 1) * radial
        violet = np.clip((n2 - 0.50) * 2.2, 0, 1) * np.roll(radial, small_w // 5, axis=1)
        rgba = np.zeros((small_h, small_w, 4), dtype=np.uint8)
        rgba[..., 0] = np.uint8(np.clip(84 * orange + 52 * violet, 0, 255))
        rgba[..., 1] = np.uint8(np.clip(28 * orange + 16 * violet, 0, 255))
        rgba[..., 2] = np.uint8(np.clip(18 * orange + 98 * violet, 0, 255))
        rgba[..., 3] = np.uint8(np.clip(72 * orange + 48 * violet, 0, 140))
        return Image.fromarray(rgba, "RGBA").resize(OUT_SIZE, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(max(8, int(28 * FS))))

    def _build_vignette(self) -> np.ndarray:
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
        nx = (xx - WIDTH / 2) / (WIDTH / 2)
        ny = (yy - HEIGHT / 2) / (HEIGHT / 2)
        radius = np.sqrt(nx * nx + ny * ny)
        return np.clip(1.0 - 0.44 * radius ** 1.8, 0.0, 1.0).astype(np.float32)

    def _base_canvas(self, t: float, nebula_amount: float = 0.25, warm: float = 0.0) -> Image.Image:
        if self.star_background is not None:
            background = self.star_background
            max_x = max(0, background.width - WIDTH)
            max_y = max(0, background.height - HEIGHT)
            x = int((0.5 + 0.5 * math.sin(t * 0.0038)) * max_x)
            y = int((0.5 + 0.5 * math.cos(t * 0.0031)) * max_y)
            canvas = background.crop((x, y, x + WIDTH, y + HEIGHT)).copy()
            veil = Image.new("RGBA", OUT_SIZE, (2 + int(18 * warm), 3 + int(6 * warm), 10, 82))
            canvas.alpha_composite(veil)
        else:
            canvas = Image.new("RGBA", OUT_SIZE, (2 + int(18 * warm), 3 + int(6 * warm), 10, 255))
        if nebula_amount > 0:
            neb = self.nebula.copy()
            neb.putalpha(neb.getchannel("A").point(lambda p: int(p * nebula_amount)))
            canvas.alpha_composite(neb)
        return canvas

    def _draw_starfield(self, canvas: Image.Image, t: float, warp: float = 0.0, drift: Tuple[float, float] = (0.0, 0.0), brightness: float = 1.0) -> None:
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        xy = self.stars.xy.copy()
        depth = self.stars.depth
        xy[:, 0] += drift[0] / depth + 0.0025 * np.sin(t * 0.013 + self.stars.phase) / depth
        xy[:, 1] += drift[1] / depth + 0.0021 * np.cos(t * 0.011 + self.stars.phase) / depth
        sx = (0.5 + 0.5 * xy[:, 0]) * WIDTH
        sy = (0.5 + 0.5 * xy[:, 1]) * HEIGHT
        valid = (sx >= -100) & (sx <= WIDTH + 100) & (sy >= -100) & (sy <= HEIGHT + 100)
        order = np.where(valid)[0]
        for i in order:
            lum = float(self.stars.luminosity[i] * brightness / (0.42 + 0.22 * self.stars.depth[i]))
            twinkle = 0.82 + 0.18 * math.sin(t * (0.7 + 0.12 * self.stars.temperature[i]) + self.stars.phase[i])
            alpha = int(clamp(lum * twinkle * 145.0, 6.0, 245.0))
            temp = float(self.stars.temperature[i])
            color = (int(160 + 95 * temp), int(170 + 70 * temp), int(215 + 30 * temp), alpha)
            x, y = float(sx[i]), float(sy[i])
            radius = clamp(0.35 + lum * 0.7, 0.45, 2.6) * FS
            if warp > 0.01:
                vx = x - WIDTH * 0.5
                vy = y - HEIGHT * 0.5
                length = warp * (16.0 + 84.0 / max(float(self.stars.depth[i]), 0.2)) * FS
                norm = math.hypot(vx, vy) + 1e-6
                x0 = x - vx / norm * length
                y0 = y - vy / norm * length
                draw.line((x0, y0, x, y), fill=color, width=max(1, int(radius)))
            else:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        if warp < 0.1:
            glow = layer.filter(ImageFilter.GaussianBlur(max(0.4, 1.0 * FS)))
            glow.putalpha(glow.getchannel("A").point(lambda p: p // 3))
            canvas.alpha_composite(glow)
        canvas.alpha_composite(layer)

    def _draw_dust(self, canvas: Image.Image, t: float, speed: float, opacity: float = 0.7, tint=(176, 121, 82)) -> None:
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        xyz = self.dust.copy()
        z = (xyz[:, 2] - speed * t * 0.06) % 4.8 + 0.2
        x = WIDTH * 0.5 + xyz[:, 0] / z * WIDTH * 0.16
        y = HEIGHT * 0.5 + xyz[:, 1] / z * WIDTH * 0.16
        valid = (x >= 0) & (x < WIDTH) & (y >= 0) & (y < HEIGHT)
        for px, py, pz in zip(x[valid], y[valid], z[valid]):
            a = int(clamp(opacity * 70.0 / pz, 5.0, 125.0))
            r = clamp(1.4 / pz, 0.3, 2.4) * FS
            draw.ellipse((px - r, py - r, px + r, py + r), fill=(*tint, a))
        canvas.alpha_composite(layer)

    def _planet_patch(
        self,
        kind: str,
        radius_px: int,
        rotation: float,
        light_angle: float,
        atmosphere: Tuple[int, int, int] = (110, 190, 255),
        atmosphere_strength: float = 0.55,
        phase_softness: float = 0.14,
        texture_override: Optional[np.ndarray] = None,
        emissive: bool = False,
        tint: Optional[Tuple[float, float, float]] = None,
    ) -> Image.Image:
        radius_px = max(8, int(radius_px))
        size = radius_px * 2 + 8
        yy, xx = np.mgrid[0:size, 0:size]
        nx = (xx - size / 2 + 0.5) / radius_px
        ny = (yy - size / 2 + 0.5) / radius_px
        rr = nx * nx + ny * ny
        inside = rr <= 1.0
        z = np.sqrt(np.maximum(1.0 - rr, 0.0))
        lon = np.arctan2(nx, z) + rotation
        lat = np.arcsin(np.clip(-ny, -1.0, 1.0))
        texture = texture_override if texture_override is not None else self.textures[kind]
        th, tw = texture.shape[:2]
        tx = ((lon / (2.0 * math.pi) + 0.5) % 1.0 * (tw - 1)).astype(np.int32)
        ty = np.clip((lat / math.pi + 0.5) * (th - 1), 0, th - 1).astype(np.int32)
        sampled = texture[ty, tx]
        rgb = sampled[..., :3].astype(np.float32)
        if tint is not None:
            rgb = rgb * np.asarray(tint, dtype=np.float32)[None, None, :]
        cloud = sampled[..., 3].astype(np.float32) / 255.0
        light = normalize(np.array([math.cos(light_angle), -0.22, math.sin(light_angle)]))
        normal_dot = nx * light[0] + ny * light[1] + z * light[2]
        if emissive:
            illumination = np.ones_like(normal_dot)
            specular = np.zeros_like(normal_dot)
        else:
            illumination = 0.10 + 0.90 * np.clip((normal_dot + phase_softness) / (1.0 + phase_softness), 0.0, 1.0)
            specular = np.clip(normal_dot, 0.0, 1.0) ** 24
        limb = np.clip(z, 0.0, 1.0) ** 0.30
        rgb = rgb * illumination[..., None] * (0.75 + 0.25 * limb[..., None])
        rgb += specular[..., None] * 34.0
        if cloud.max() > 0:
            cloud_light = np.clip(illumination * (0.78 + 0.22 * z), 0.0, 1.0)
            rgb = rgb * (1.0 - 0.72 * cloud[..., None]) + 250.0 * cloud[..., None] * cloud_light[..., None]
        atmosphere_mask = np.clip((1.0 - np.sqrt(np.maximum(rr, 0.0))) / 0.12, 0.0, 1.0)
        rim = (1.0 - atmosphere_mask) * inside * atmosphere_strength
        rgb += np.asarray(atmosphere, dtype=np.float32)[None, None, :] * rim[..., None] * 0.72
        alpha = np.where(inside, 255.0, np.clip((1.08 - np.sqrt(rr)) / 0.08 * 255.0, 0.0, 255.0))
        array = np.zeros((size, size, 4), dtype=np.uint8)
        array[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        array[..., 3] = np.uint8(alpha)
        return Image.fromarray(array, "RGBA")

    def _draw_planet(self, canvas: Image.Image, center: Tuple[float, float], radius: float, body_name: str, t: float, light_angle: float = 0.7, alpha: float = 1.0, rings: bool = False, ancient: bool = False) -> None:
        radius = max(3.0, radius)
        lower = body_name.lower()
        local_map = self.local_maps.get(lower)
        atmosphere = {
            "mars": (218, 155, 108),
            "earth": (120, 210, 255),
            "jupiter": (250, 218, 172),
            "saturn": (248, 225, 174),
        }.get(lower, (120, 210, 255))
        tint = None
        if ancient and lower == "mars":
            tint = (0.90, 1.03, 1.10)
        patch = self._planet_patch(
            lower if lower in self.textures else "mars",
            int(radius),
            rotation=t * (0.009 if lower in {"mars", "earth"} else 0.006),
            light_angle=light_angle,
            atmosphere=atmosphere,
            atmosphere_strength=0.75 if lower in {"earth", "mars"} else 0.25,
            texture_override=local_map,
            tint=tint,
        )
        if ancient and lower == "mars":
            patch = self._apply_ancient_mars_overlay(patch)
        if alpha < 0.999:
            patch.putalpha(patch.getchannel("A").point(lambda p: int(p * alpha)))
        glow = patch.filter(ImageFilter.GaussianBlur(max(2, int(radius * 0.10))))
        glow.putalpha(glow.getchannel("A").point(lambda p: int(p * 0.20)))
        x = int(center[0] - patch.width / 2)
        y = int(center[1] - patch.height / 2)
        alpha_composite_at(canvas, glow, (x, y))
        if rings and lower == "saturn":
            ring_layers = self._ring_layers(radius, alpha)
            if ring_layers is not None:
                back, front = ring_layers
                alpha_composite_at(canvas, back, (int(center[0] - back.width / 2), int(center[1] - back.height / 2)))
                alpha_composite_at(canvas, patch, (x, y))
                alpha_composite_at(canvas, front, (int(center[0] - front.width / 2), int(center[1] - front.height / 2)))
                return
        alpha_composite_at(canvas, patch, (x, y))

    def _apply_ancient_mars_overlay(self, patch: Image.Image) -> Image.Image:
        w, h = patch.size
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        # Subtle blue-green ancient-water zone in the northern hemisphere.
        draw.ellipse((int(0.06 * w), int(0.33 * h), int(0.88 * w), int(0.90 * h)), fill=(58, 110, 136, 86))
        for i in range(5):
            draw.arc((int(0.12 * w), int((0.26 + i * 0.07) * h), int(0.83 * w), int((0.80 + i * 0.05) * h)), 210, 330, fill=(180, 220, 236, 70), width=max(1, int(2 * FS)))
        combined = Image.alpha_composite(patch.convert("RGBA"), overlay)
        return combined

    @lru_cache(maxsize=36)
    def _cached_ring_layers(self, radius_quantized: int) -> Tuple[Image.Image, Image.Image]:
        ring = self.ring_assets["saturn"]
        radius = max(8, int(radius_quantized))
        target_w = max(16, int(radius * 4.95))
        target_h = max(8, int(target_w * ring.height / max(ring.width, 1)))
        resized = ring.resize((target_w, target_h), Image.Resampling.LANCZOS)
        alpha = np.asarray(resized.getchannel("A"), dtype=np.float32)
        yy = np.arange(target_h, dtype=np.float32)[:, None]
        centre = (target_h - 1) / 2.0
        softness = max(2.0, target_h * 0.035)
        front_factor = np.clip((yy - centre + softness) / (2.0 * softness), 0.0, 1.0)
        back_factor = 1.0 - front_factor
        back = resized.copy()
        front = resized.copy()
        back.putalpha(Image.fromarray(np.uint8(alpha * back_factor), "L"))
        front.putalpha(Image.fromarray(np.uint8(alpha * front_factor), "L"))
        return back, front

    def _ring_layers(self, radius: float, alpha: float) -> Optional[Tuple[Image.Image, Image.Image]]:
        if "saturn" not in self.ring_assets:
            return None
        quantized = max(8, int(round(radius / 3.0) * 3))
        back, front = self._cached_ring_layers(quantized)
        back = back.copy()
        front = front.copy()
        if alpha < 0.999:
            back.putalpha(back.getchannel("A").point(lambda p: int(p * alpha)))
            front.putalpha(front.getchannel("A").point(lambda p: int(p * alpha)))
        return back, front

    def _draw_sun(self, canvas: Image.Image, center: Tuple[float, float], radius: float, alpha: float = 1.0) -> None:
        cx, cy = center
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(glow)
        colour = (255, 214, 102)
        for mul, a in [(4.6, 8), (3.2, 14), (2.2, 25), (1.5, 52)]:
            r = radius * mul
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*colour, int(a * alpha)))
        glow = glow.filter(ImageFilter.GaussianBlur(max(4, int(radius * 0.55))))
        canvas.alpha_composite(glow)
        patch = self._planet_patch(
            "sun",
            int(radius),
            rotation=0.012 * radius,
            light_angle=0.0,
            atmosphere=(255, 210, 95),
            atmosphere_strength=0.08,
            texture_override=self.local_maps.get("sun"),
            emissive=True,
        )
        if alpha < 0.999:
            patch.putalpha(patch.getchannel("A").point(lambda p: int(p * alpha)))
        alpha_composite_at(canvas, patch, (int(cx - patch.width / 2), int(cy - patch.height / 2)))

    def _draw_vista_surface(self, canvas: Image.Image, t: float, horizon_y: float, ancient: bool = False, blue_sunset: bool = False) -> None:
        ground = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(ground)
        y0 = int(horizon_y)
        for y in range(y0, HEIGHT):
            q = (y - y0) / max(HEIGHT - y0, 1)
            r = int(lerp(96 if not ancient else 82, 44 if not ancient else 58, q))
            g = int(lerp(54 if not ancient else 74, 22 if not ancient else 48, q))
            b = int(lerp(28 if not ancient else 62, 12 if not ancient else 35, q))
            draw.line((0, y, WIDTH, y), fill=(r, g, b, 255))
        # Dune silhouette.
        points = []
        for i in range(0, WIDTH + 40, max(18, int(26 * FS))):
            x = i
            y = horizon_y + 18 * FS * math.sin(i * 0.0065 + t * 0.22) + 52 * FS * deterministic_unit(f"ridge{i}", "a")
            points.append((x, y))
        points += [(WIDTH, HEIGHT), (0, HEIGHT)]
        draw.polygon(points, fill=(76, 40, 22, 255) if not ancient else (60, 56, 48, 255))
        canvas.alpha_composite(ground)
        # Atmospheric glow.
        sky = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(sky)
        for i in range(36):
            q = i / 35.0
            yy = horizon_y - q * HEIGHT * 0.42
            alpha = int(lerp(30, 0, q))
            colour = (176, 102, 76, alpha)
            if ancient:
                colour = (110, 152, 180, alpha)
            if blue_sunset:
                colour = (66, 120, 208, alpha)
            draw.rectangle((0, yy - 8, WIDTH, yy + 8), fill=colour)
        canvas.alpha_composite(sky.filter(ImageFilter.GaussianBlur(max(8, int(18 * FS)))))

    def _draw_radio_arcs(self, canvas: Image.Image, center: Tuple[float, float], phase: float, color=(95, 188, 255)) -> None:
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx, cy = center
        for i in range(6):
            q = (phase + i / 6.0) % 1.0
            r = q * max(WIDTH, HEIGHT) * 0.7
            a = int(75 * (1.0 - q))
            draw.arc((cx - r, cy - r, cx + r, cy + r), 142, 218, fill=(*color, a), width=max(1, int(2 * FS)))
        canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(0.2, 0.45 * FS))))

    def _draw_orbit_line(self, canvas: Image.Image, bbox: Tuple[float, float, float, float], start=210, end=345, colour=(120, 170, 230, 42)) -> None:
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).arc(bbox, start, end, fill=colour, width=max(1, int(2 * FS)))
        canvas.alpha_composite(overlay)

    def _draw_rocket_silhouette(self, canvas: Image.Image, center: Tuple[float, float], scale_px: float, angle: float = 0.0, alpha: float = 1.0) -> None:
        size = max(64, int(scale_px * 5.0))
        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx = cy = size / 2
        s = scale_px
        a = int(255 * clamp(alpha))
        body = [(cx, cy - 1.55 * s), (cx + 0.34 * s, cy + 0.88 * s), (cx - 0.34 * s, cy + 0.88 * s)]
        draw.polygon(body, fill=(200, 204, 212, a), outline=(245, 248, 251, a))
        draw.ellipse((cx - 0.18 * s, cy - 0.70 * s, cx + 0.18 * s, cy - 0.34 * s), fill=(72, 130, 210, a))
        draw.polygon([(cx - 0.34 * s, cy + 0.25 * s), (cx - 0.75 * s, cy + 0.78 * s), (cx - 0.28 * s, cy + 0.72 * s)], fill=(170, 56, 42, a))
        draw.polygon([(cx + 0.34 * s, cy + 0.25 * s), (cx + 0.75 * s, cy + 0.78 * s), (cx + 0.28 * s, cy + 0.72 * s)], fill=(170, 56, 42, a))
        draw.polygon([(cx - 0.16 * s, cy + 0.90 * s), (cx - 0.40 * s, cy + 1.22 * s), (cx, cy + 1.05 * s)], fill=(184, 188, 197, a))
        draw.polygon([(cx + 0.16 * s, cy + 0.90 * s), (cx + 0.40 * s, cy + 1.22 * s), (cx, cy + 1.05 * s)], fill=(184, 188, 197, a))
        flame_len = (0.45 + 0.25 * math.sin(scale_px * 0.3)) * s
        draw.polygon([(cx - 0.12 * s, cy + 0.90 * s), (cx, cy + 0.90 * s + flame_len), (cx + 0.12 * s, cy + 0.90 * s)], fill=(255, 188, 68, int(180 * alpha)))
        layer = layer.rotate(math.degrees(angle), resample=Image.Resampling.BICUBIC, expand=False)
        alpha_composite_at(canvas, layer, (int(center[0] - size / 2), int(center[1] - size / 2)))

    def _draw_rover(self, canvas: Image.Image, center: Tuple[float, float], scale_px: float, alpha: float = 1.0) -> None:
        size = max(110, int(scale_px * 6.2))
        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx = size * 0.48
        cy = size * 0.60
        s = scale_px
        a = int(255 * clamp(alpha))
        draw.rounded_rectangle((cx - 0.95 * s, cy - 0.38 * s, cx + 0.18 * s, cy + 0.26 * s), radius=max(2, int(0.12 * s)), fill=(176, 174, 170, a), outline=(245, 244, 241, a))
        draw.line((cx - 0.20 * s, cy - 0.10 * s, cx + 0.62 * s, cy - 0.70 * s), fill=(188, 190, 196, a), width=max(1, int(0.08 * s)))
        draw.ellipse((cx + 0.52 * s, cy - 0.86 * s, cx + 0.82 * s, cy - 0.56 * s), fill=(205, 208, 213, a))
        draw.line((cx - 0.85 * s, cy + 0.30 * s, cx - 1.10 * s, cy + 0.80 * s), fill=(155, 155, 160, a), width=max(1, int(0.06 * s)))
        draw.line((cx - 0.25 * s, cy + 0.30 * s, cx - 0.05 * s, cy + 0.82 * s), fill=(155, 155, 160, a), width=max(1, int(0.06 * s)))
        draw.line((cx + 0.12 * s, cy + 0.30 * s, cx + 0.36 * s, cy + 0.82 * s), fill=(155, 155, 160, a), width=max(1, int(0.06 * s)))
        for wx, wy in [(cx - 1.12 * s, cy + 0.88 * s), (cx - 0.03 * s, cy + 0.88 * s), (cx + 0.39 * s, cy + 0.90 * s)]:
            draw.ellipse((wx - 0.25 * s, wy - 0.25 * s, wx + 0.25 * s, wy + 0.25 * s), outline=(70, 70, 74, a), width=max(1, int(0.08 * s)), fill=(30, 26, 24, a))
        draw.rectangle((cx - 0.55 * s, cy - 0.92 * s, cx + 0.15 * s, cy - 0.76 * s), fill=(58, 98, 165, a))
        alpha_composite_at(canvas, layer, (int(center[0] - size / 2), int(center[1] - size / 2)))

    def _draw_astronaut(self, canvas: Image.Image, base_xy: Tuple[float, float], scale_px: float, alpha: float = 1.0) -> None:
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        x, y = base_xy
        s = scale_px
        a = int(255 * clamp(alpha))
        draw.ellipse((x - 0.22 * s, y - 1.90 * s, x + 0.22 * s, y - 1.46 * s), fill=(210, 214, 220, a), outline=(255, 255, 255, a))
        draw.rounded_rectangle((x - 0.30 * s, y - 1.44 * s, x + 0.30 * s, y - 0.52 * s), radius=max(2, int(0.10 * s)), fill=(190, 194, 200, a), outline=(252, 252, 252, a))
        draw.rectangle((x - 0.18 * s, y - 1.34 * s, x + 0.18 * s, y - 1.18 * s), fill=(56, 96, 170, a))
        draw.line((x - 0.18 * s, y - 0.52 * s, x - 0.26 * s, y), fill=(185, 188, 194, a), width=max(1, int(0.10 * s)))
        draw.line((x + 0.18 * s, y - 0.52 * s, x + 0.26 * s, y), fill=(185, 188, 194, a), width=max(1, int(0.10 * s)))
        draw.line((x - 0.28 * s, y - 1.20 * s, x - 0.56 * s, y - 0.74 * s), fill=(185, 188, 194, a), width=max(1, int(0.08 * s)))
        draw.line((x + 0.28 * s, y - 1.20 * s, x + 0.60 * s, y - 0.98 * s), fill=(185, 188, 194, a), width=max(1, int(0.08 * s)))
        draw.line((x + 0.60 * s, y - 0.98 * s, x + 1.02 * s, y - 1.78 * s), fill=(172, 182, 192, int(0.7 * a)), width=max(1, int(0.04 * s)))
        draw.ellipse((x + 0.98 * s - 0.06 * s, y - 1.78 * s - 0.06 * s, x + 0.98 * s + 0.06 * s, y - 1.78 * s + 0.06 * s), fill=(255, 255, 255, int(0.8 * a)))
        canvas.alpha_composite(layer)

    def _draw_habitat(self, canvas: Image.Image, anchor: Tuple[float, float], scale_px: float, alpha: float = 1.0) -> None:
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        x, y = anchor
        s = scale_px
        a = int(255 * clamp(alpha))
        draw.rounded_rectangle((x, y - 0.52 * s, x + 1.9 * s, y), radius=max(2, int(0.14 * s)), fill=(156, 154, 150, a), outline=(240, 240, 240, a))
        draw.rounded_rectangle((x + 2.2 * s, y - 0.42 * s, x + 3.0 * s, y), radius=max(2, int(0.12 * s)), fill=(166, 162, 158, a), outline=(240, 240, 240, a))
        draw.rectangle((x + 0.20 * s, y - 0.38 * s, x + 0.72 * s, y - 0.12 * s), fill=(74, 118, 190, a))
        draw.rectangle((x + 0.92 * s, y - 0.38 * s, x + 1.44 * s, y - 0.12 * s), fill=(74, 118, 190, a))
        draw.line((x + 3.0 * s, y - 0.22 * s, x + 3.58 * s, y - 1.22 * s), fill=(184, 184, 190, a), width=max(1, int(0.05 * s)))
        draw.ellipse((x + 3.58 * s - 0.24 * s, y - 1.22 * s - 0.24 * s, x + 3.58 * s + 0.24 * s, y - 1.22 * s + 0.24 * s), outline=(220, 220, 224, a), width=max(1, int(0.05 * s)))
        canvas.alpha_composite(layer)

    def _draw_film_grain(self, image: Image.Image, strength: float = 0.06) -> Image.Image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        noise = self.rng.normal(0.0, 255.0 * strength, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB").convert("RGBA")

    def _apply_grade(self, canvas: Image.Image, exposure: float = 1.0, contrast: float = 1.08, saturation: float = 1.08, grain: float = 0.045) -> Image.Image:
        rgb = canvas.convert("RGB")
        if exposure != 1.0:
            rgb = ImageEnhance.Brightness(rgb).enhance(exposure)
        rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
        rgb = ImageEnhance.Color(rgb).enhance(saturation)
        array = np.asarray(rgb, dtype=np.float32)
        array *= self.vignette[..., None]
        array = np.clip(array, 0, 255).astype(np.uint8)
        graded = Image.fromarray(array, "RGB").convert("RGBA")
        if grain > 0:
            graded = self._draw_film_grain(graded, strength=grain)
        return graded

    # ----------------------------------------------------------------------
    # Scene renderers
    # ----------------------------------------------------------------------

    def _scene_signal(self, u: float, t: float) -> Image.Image:
        canvas = self._base_canvas(t, nebula_amount=0.08)
        self._draw_starfield(canvas, t, brightness=0.88)
        center = (WIDTH * 0.51, HEIGHT * 0.48)
        pulse = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(pulse)
        r = lerp(4, 26, smootherstep(u)) * FS
        draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), fill=(255, 124, 88, 210))
        pulse = pulse.filter(ImageFilter.GaussianBlur(max(2, int(8 * FS))))
        canvas.alpha_composite(pulse)
        self._draw_radio_arcs(canvas, center, (u * 1.2) % 1.0, color=(235, 120, 88))
        if u > 0.52:
            mars_r = lerp(40, 180 if FORMAT == "wide" else 145, smoothstep((u - 0.52) / 0.48)) * FS
            self._draw_planet(canvas, center, mars_r, "Mars", t, light_angle=0.55)
        return self._apply_grade(canvas, exposure=1.02, contrast=1.12, saturation=1.14, grain=0.035)

    def _scene_earth_to_mars(self, u: float, t: float) -> Image.Image:
        canvas = self._base_canvas(t, nebula_amount=0.12)
        self._draw_starfield(canvas, t, drift=(0.016 * u, -0.008 * u), brightness=0.95)
        earth_r = lerp(170, 250, 1.0 - u) * FS if FORMAT == "wide" else lerp(150, 210, 1.0 - u) * FS
        earth_x = lerp(WIDTH * 0.28, WIDTH * 0.13, smootherstep(u))
        earth_y = lerp(HEIGHT * 0.60, HEIGHT * 0.66, smootherstep(u))
        self._draw_planet(canvas, (earth_x, earth_y), earth_r, "Earth", t, light_angle=0.95)
        mars_r = lerp(26, 72, smootherstep(u)) * FS
        mars_x = lerp(WIDTH * 0.74, WIDTH * 0.82, smootherstep(u))
        mars_y = lerp(HEIGHT * 0.34, HEIGHT * 0.26, smootherstep(u))
        self._draw_planet(canvas, (mars_x, mars_y), mars_r, "Mars", t, light_angle=0.65)
        sx = WIDTH * 0.10
        sy = HEIGHT * 0.18
        self._draw_sun(canvas, (sx, sy), 14 * FS, alpha=0.82)
        self._draw_orbit_line(canvas, (WIDTH * 0.04, HEIGHT * 0.18, WIDTH * 0.56, HEIGHT * 1.10), 198, 333, (105, 174, 225, 38))
        self._draw_orbit_line(canvas, (WIDTH * 0.20, HEIGHT * -0.12, WIDTH * 1.10, HEIGHT * 0.82), 206, 320, (236, 141, 96, 40))
        # transfer path
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        pts = []
        for i in range(40):
            q = i / 39.0
            x = lerp(earth_x + earth_r * 0.9, mars_x - mars_r * 0.7, q)
            y = lerp(earth_y - earth_r * 0.2, mars_y + mars_r * 0.5, q) - math.sin(q * math.pi) * 110 * FS
            pts.append((x, y))
        draw.line(pts, fill=(246, 188, 130, 82), width=max(1, int(2 * FS)))
        alpha_composite_at(canvas, overlay.filter(ImageFilter.GaussianBlur(max(0.2, 0.55 * FS))), (0, 0))
        return self._apply_grade(canvas, exposure=1.02, contrast=1.10, saturation=1.10, grain=0.04)

    def _scene_orbit(self, u: float, t: float) -> Image.Image:
        canvas = self._base_canvas(t, nebula_amount=0.10, warm=0.15)
        self._draw_starfield(canvas, t, drift=(-0.012 * u, 0.0), brightness=0.92)
        mars_r = lerp(240 if FORMAT == "wide" else 210, 420 if FORMAT == "wide" else 330, smoothstep(u)) * FS
        mars_x = lerp(WIDTH * 0.84, WIDTH * 0.60, smootherstep(u))
        mars_y = lerp(HEIGHT * 0.60, HEIGHT * 0.54, smootherstep(u))
        self._draw_planet(canvas, (mars_x, mars_y), mars_r, "Mars", t, light_angle=0.52)
        # tiny transfer ship
        ship_x = lerp(WIDTH * 0.25, WIDTH * 0.42, smootherstep(u))
        ship_y = lerp(HEIGHT * 0.73, HEIGHT * 0.60, smootherstep(u))
        self._draw_rocket_silhouette(canvas, (ship_x, ship_y), lerp(16, 24, u) * FS, angle=-0.50, alpha=0.96)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for i in range(7):
            q = i / 6.0
            a = int(64 * (1 - q))
            draw.arc((ship_x - q * 330 * FS, ship_y - q * 330 * FS, ship_x + q * 330 * FS, ship_y + q * 330 * FS), 130, 220, fill=(105, 188, 248, a), width=max(1, int(2 * FS)))
        canvas.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(max(0.2, 0.45 * FS))))
        return self._apply_grade(canvas, exposure=1.02, contrast=1.13, saturation=1.12, grain=0.042)

    def _scene_ancient(self, u: float, t: float) -> Image.Image:
        canvas = self._base_canvas(t, nebula_amount=0.18, warm=0.05)
        self._draw_starfield(canvas, t, brightness=0.88)
        self._draw_planet(canvas, (WIDTH * 0.70, HEIGHT * 0.42), lerp(240, 360, smoothstep(u)) * FS, "Mars", t, light_angle=0.62, ancient=True)
        # Surface insert panel feel using ancient watery horizon
        self._draw_vista_surface(canvas, t, HEIGHT * lerp(0.82, 0.74, smoothstep(u)), ancient=True)
        mist = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(mist)
        for i in range(30):
            y = HEIGHT * (0.32 + i * 0.012)
            draw.rectangle((0, y, WIDTH, y + 8), fill=(124, 184, 210, max(0, 24 - i)))
        canvas.alpha_composite(mist.filter(ImageFilter.GaussianBlur(max(10, int(24 * FS)))))
        return self._apply_grade(canvas, exposure=1.04, contrast=1.09, saturation=1.05, grain=0.04)

    def _scene_monuments(self, u: float, t: float) -> Image.Image:
        canvas = self._base_canvas(t, nebula_amount=0.12, warm=0.18)
        self._draw_starfield(canvas, t, drift=(0.006, 0.0), brightness=0.88)
        self._draw_planet(canvas, (WIDTH * 0.57, HEIGHT * 0.53), lerp(360, 520, smoothstep(u)) * FS, "Mars", t, light_angle=0.42)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        # Stylised scan lines and markers.
        marker_alpha = int(200 * fade_window(u, 0.08, 0.94, 0.18))
        pts = [
            (WIDTH * 0.54, HEIGHT * 0.40, "OLYMPUS MONS", WIDTH * 0.22, HEIGHT * 0.20),
            (WIDTH * 0.69, HEIGHT * 0.59, "VALLES MARINERIS", WIDTH * 0.86, HEIGHT * 0.80),
        ]
        for mx, my, label, tx, ty in pts:
            rr = 7 * FS
            draw.ellipse((mx - rr, my - rr, mx + rr, my + rr), outline=(110, 198, 248, marker_alpha), width=max(1, int(2 * FS)))
            draw.line((mx, my, tx, ty), fill=(110, 198, 248, int(marker_alpha * 0.72)), width=max(1, int(2 * FS)))
            draw.rounded_rectangle((tx - 10 * FS, ty - 18 * FS, tx + 220 * FS, ty + 14 * FS), radius=max(2, int(6 * FS)), fill=(3, 10, 18, int(marker_alpha * 0.62)), outline=(110, 198, 248, int(marker_alpha * 0.55)), width=max(1, int(FS)))
            draw_text(overlay, label, (tx + 4 * FS, ty), 15, (232, 238, 248, marker_alpha), True)
        for i in range(9):
            y = HEIGHT * (0.18 + i * 0.08)
            draw.line((WIDTH * 0.04, y, WIDTH * 0.96, y), fill=(98, 168, 220, 10), width=max(1, int(FS)))
        canvas.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(max(0.2, 0.35 * FS))))
        return self._apply_grade(canvas, exposure=1.02, contrast=1.14, saturation=1.10, grain=0.042)

    def _scene_rovers(self, u: float, t: float) -> Image.Image:
        canvas = self._base_canvas(t, nebula_amount=0.08, warm=0.10)
        horizon = lerp(HEIGHT * 0.63, HEIGHT * 0.70, smoothstep(u))
        self._draw_vista_surface(canvas, t, horizon, ancient=False, blue_sunset=True)
        self._draw_starfield(canvas, t, brightness=0.38)
        # small Mars in sky
        self._draw_planet(canvas, (WIDTH * 0.80, HEIGHT * 0.22), lerp(48, 80, smoothstep(u)) * FS, "Mars", t, light_angle=0.82)
        self._draw_rover(canvas, (WIDTH * 0.38, HEIGHT * 0.75), lerp(34, 46, u) * FS, alpha=0.98)
        self._draw_rover(canvas, (WIDTH * 0.62, HEIGHT * 0.80), lerp(26, 38, u) * FS, alpha=0.78)
        dust_overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(dust_overlay)
        for i in range(14):
            x0 = WIDTH * (-0.10 + i * 0.10)
            y0 = horizon - 70 * FS + i * 8 * FS
            x1 = x0 + WIDTH * 0.36
            y1 = y0 + 22 * FS
            draw.ellipse((x0, y0, x1, y1), fill=(176, 120, 82, 12))
        canvas.alpha_composite(dust_overlay.filter(ImageFilter.GaussianBlur(max(10, int(20 * FS)))))
        self._draw_dust(canvas, t, speed=0.22, opacity=0.24, tint=(176, 120, 82))
        return self._apply_grade(canvas, exposure=1.03, contrast=1.10, saturation=1.02, grain=0.045)

    def _scene_future(self, u: float, t: float) -> Image.Image:
        canvas = self._base_canvas(t, nebula_amount=0.05, warm=0.18)
        horizon = lerp(HEIGHT * 0.70, HEIGHT * 0.76, smoothstep(u))
        self._draw_vista_surface(canvas, t, horizon, ancient=False, blue_sunset=False)
        sky = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(sky)
        # Two faint moons / lights
        draw.ellipse((WIDTH * 0.72 - 10 * FS, HEIGHT * 0.22 - 10 * FS, WIDTH * 0.72 + 10 * FS, HEIGHT * 0.22 + 10 * FS), fill=(220, 215, 210, 150))
        draw.ellipse((WIDTH * 0.82 - 6 * FS, HEIGHT * 0.26 - 6 * FS, WIDTH * 0.82 + 6 * FS, HEIGHT * 0.26 + 6 * FS), fill=(208, 202, 194, 110))
        canvas.alpha_composite(sky.filter(ImageFilter.GaussianBlur(max(1, int(2 * FS)))))
        self._draw_habitat(canvas, (WIDTH * 0.56, horizon + 8 * FS), 54 * FS, alpha=0.88)
        self._draw_astronaut(canvas, (WIDTH * 0.42, horizon + 16 * FS), 54 * FS, alpha=0.98)
        self._draw_astronaut(canvas, (WIDTH * 0.30, horizon + 26 * FS), 38 * FS, alpha=0.74)
        # launch plume far distance
        launch = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(launch)
        lx = WIDTH * 0.18
        ly = horizon + 10 * FS
        draw.polygon([(lx, ly), (lx + 8 * FS, ly - 70 * FS), (lx + 16 * FS, ly)], fill=(255, 186, 92, 90))
        draw.line((lx + 8 * FS, ly - 70 * FS, lx + 8 * FS, ly - 116 * FS), fill=(238, 238, 238, 140), width=max(1, int(2 * FS)))
        canvas.alpha_composite(launch.filter(ImageFilter.GaussianBlur(max(8, int(14 * FS)))))
        return self._apply_grade(canvas, exposure=1.03, contrast=1.11, saturation=1.04, grain=0.042)

    def _scene_finale(self, u: float, t: float) -> Image.Image:
        canvas = self._base_canvas(t, nebula_amount=0.26, warm=0.12)
        self._draw_starfield(canvas, t, warp=0.08 * smoothstep(u), drift=(0.004 * u, -0.002 * u), brightness=0.86)
        mars_r = lerp(210 if FORMAT == "wide" else 180, 42 if FORMAT == "wide" else 34, smootherstep(u)) * FS
        mars_x = lerp(WIDTH * 0.62, WIDTH * 0.50, smootherstep(u))
        mars_y = lerp(HEIGHT * 0.48, HEIGHT * 0.45, smootherstep(u))
        self._draw_planet(canvas, (mars_x, mars_y), mars_r, "Mars", t, light_angle=0.52)
        self._draw_radio_arcs(canvas, (mars_x, mars_y), (u * 1.3) % 1.0, color=(235, 136, 86))
        # distant planets as tiny jewels for the sense of a larger solar system
        self._draw_planet(canvas, (WIDTH * 0.19, HEIGHT * 0.26), 14 * FS, "Jupiter", t, light_angle=0.7)
        self._draw_planet(canvas, (WIDTH * 0.24, HEIGHT * 0.72), 11 * FS, "Saturn", t, light_angle=0.7, rings=True)
        # tiny final red point to loop back
        if u > 0.72:
            overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            q = smoothstep((u - 0.72) / 0.28)
            x = lerp(mars_x, WIDTH * 0.50, q)
            y = lerp(mars_y, HEIGHT * 0.48, q)
            r = lerp(mars_r, 5 * FS, q)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 118, 84, 210))
            canvas.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(max(1, int(6 * FS)))))
        return self._apply_grade(canvas, exposure=1.03, contrast=1.13, saturation=1.10, grain=0.04)

    def _scene_for_fraction(self, fraction: float, t: float) -> Image.Image:
        mapping = {
            "signal": self._scene_signal,
            "earth_to_mars": self._scene_earth_to_mars,
            "orbit": self._scene_orbit,
            "ancient": self._scene_ancient,
            "monuments": self._scene_monuments,
            "rovers": self._scene_rovers,
            "future": self._scene_future,
            "finale": self._scene_finale,
        }
        active = []
        for start, end, name in SCENES:
            if start <= fraction <= end:
                active.append((start, end, mapping[name]))
        if not active:
            active = [(SCENES[-1][0], SCENES[-1][1], mapping[SCENES[-1][2]])]
        images: List[Tuple[Image.Image, float]] = []
        for start, end, function in active:
            u = clamp((fraction - start) / max(end - start, 1e-8))
            weight = fade_window(fraction, start, end, 0.08)
            images.append((function(u, t), weight))
        if len(images) == 1:
            return images[0][0]
        base = images[0][0]
        cumulative = images[0][1]
        for image, weight in images[1:]:
            mix = weight / max(cumulative + weight, 1e-8)
            base = Image.blend(base, image, clamp(mix))
            cumulative += weight
        return base

    def _draw_interface(self, canvas: Image.Image, t: float) -> None:
        fraction = clamp(t / max(DURATION, 1e-8))
        intro = min(16.0, DURATION * 0.08)
        if t < intro:
            fade_in = smoothstep(t / max(3.5, intro * 0.22))
            fade_out = 1.0 - smoothstep((t - intro * 0.72) / max(intro * 0.28, 1e-8))
            alpha = int(255 * fade_in * fade_out)
            draw_text(canvas, "RED FRONTIER", (WIDTH * 0.50, HEIGHT * 0.44), 84, (246, 248, 253, alpha), True, True, "mm", 2)
            draw_text(canvas, "MARS BEFORE FOOTSTEPS", (WIDTH * 0.50, HEIGHT * 0.53), 28, (238, 136, 96, int(alpha * 0.92)), True, False, "mm")
            draw_text(canvas, "A CINEMATIC SPACE DOCUMENTARY", (WIDTH * 0.50, HEIGHT * 0.60), 15, (190, 207, 228, int(alpha * 0.76)), True, False, "mm")

        # chapter label
        chapter_title = CHAPTERS[-1][1]
        chapter_progress = 1.0
        for index in range(len(CHAPTERS) - 1):
            start, title = CHAPTERS[index]
            end = CHAPTERS[index + 1][0]
            if start <= fraction < end:
                chapter_title = title
                chapter_progress = (fraction - start) / max(end - start, 1e-8)
                break
        chapter_alpha = int(205 * (1.0 - smoothstep((chapter_progress - 0.02) / 0.16)))
        if fraction > 0.03 and chapter_alpha > 0:
            draw_text(canvas, chapter_title.upper(), (68 * FS, 88 * FS), 20, (232, 239, 249, chapter_alpha), True)

        # fact callouts
        for frac, text in FACT_LINES:
            a = int(220 * fade_window(fraction, frac - 0.018, frac + 0.055, 0.30))
            if a > 0:
                w = min(int(WIDTH * 0.84), int(860 * FS))
                h = int(54 * FS)
                x = int((WIDTH - w) / 2)
                y = int(HEIGHT * 0.86)
                bar = Image.new("RGBA", (w, h), (4, 10, 16, int(a * 0.70)))
                dr = ImageDraw.Draw(bar)
                dr.rounded_rectangle((0, 0, w - 1, h - 1), radius=max(8, int(12 * FS)), fill=(4, 10, 16, int(a * 0.70)), outline=(236, 136, 96, int(a * 0.60)), width=max(1, int(2 * FS)))
                alpha_composite_at(canvas, bar, (x, y))
                draw_text(canvas, text, (WIDTH * 0.50, y + h / 2), 16, (240, 243, 248, a), True, False, "mm")

        # progress line
        if 0.08 < fraction < 0.985:
            overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            x0, x1, y = WIDTH * 0.08, WIDTH * 0.92, HEIGHT * 0.945
            draw.line((x0, y, x1, y), fill=(125, 170, 210, 35), width=max(1, int(FS)))
            draw.line((x0, y, lerp(x0, x1, fraction), y), fill=(238, 136, 96, 135), width=max(1, int(2 * FS)))
            px = lerp(x0, x1, fraction)
            draw.ellipse((px - 3 * FS, y - 3 * FS, px + 3 * FS, y + 3 * FS), fill=(252, 235, 210, 185))
            canvas.alpha_composite(overlay)

        if t >= DURATION * 0.972:
            a = int(255 * smoothstep((t - DURATION * 0.972) / max(DURATION * 0.028, 1e-8)))
            veil = Image.new("RGBA", OUT_SIZE, (0, 0, 4, int(135 * a / 255)))
            canvas.alpha_composite(veil)
            draw_text(canvas, "TEXTURES: LOCAL IMG FOLDER  •  VISUALS: ORIGINAL CINEMATIC RENDER", (WIDTH * 0.50, HEIGHT * 0.66), 13, (204, 212, 224, int(a * 0.90)), True, False, "mm")
            draw_text(canvas, "SCIENCE + STORY + SPECULATION", (WIDTH * 0.50, HEIGHT * 0.72), 18, (236, 136, 96, int(a * 0.92)), True, False, "mm")

    def render_frame(self, t: float) -> np.ndarray:
        fraction = clamp(t / max(DURATION, 1e-8))
        canvas = self._scene_for_fraction(fraction, t)
        self._draw_interface(canvas, t)
        return np.asarray(canvas.convert("RGB"), dtype=np.uint8)


# =============================================================================
# Audio generation and metadata
# =============================================================================


@dataclass
class AudioEvent:
    time_s: float
    frequency_hz: float
    amplitude: float
    decay_s: float
    pan: float
    kind: str = "chime"



def make_audio_events(duration: float) -> List[AudioEvent]:
    events: List[AudioEvent] = []
    scene_freqs = [72.0, 92.0, 110.0, 86.0, 132.0, 98.0, 146.0, 74.0]
    for idx, (start, _, _) in enumerate(SCENES):
        when = start * duration + min(2.0, duration * 0.01)
        events.append(AudioEvent(when, scene_freqs[idx], 0.18, 4.5 + idx * 0.45, -0.6 + idx * 0.18, "impact" if idx in {0, 3, 6} else "chime"))
        events.append(AudioEvent(when + 1.2, scene_freqs[idx] * 1.5, 0.07, 6.0, 0.4 - idx * 0.08, "chime"))
    return events


def _event_signal(event: AudioEvent, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    age = times - event.time_s
    active = age >= 0.0
    envelope = np.where(active, np.exp(-age / max(event.decay_s, 1e-6)), 0.0)
    if event.kind == "impact":
        sweep = event.frequency_hz * (1.0 + 1.8 * np.exp(-np.maximum(age, 0.0) * 2.7))
        phase = 2.0 * math.pi * sweep * np.maximum(age, 0.0)
        mono = np.sin(phase) * envelope + 0.35 * np.sin(phase * 0.503) * envelope
    else:
        phase = 2.0 * math.pi * event.frequency_hz * np.maximum(age, 0.0)
        mono = (np.sin(phase) + 0.42 * np.sin(phase * 2.01 + 0.8) + 0.18 * np.sin(phase * 3.99 + 1.7)) * envelope
    mono *= event.amplitude
    left_gain = math.sqrt((1.0 - clamp(event.pan, -1.0, 1.0)) * 0.5)
    right_gain = math.sqrt((1.0 + clamp(event.pan, -1.0, 1.0)) * 0.5)
    return mono * left_gain, mono * right_gain


def render_ambient_audio(path: Path, duration: float, sample_rate: int = 48_000) -> Path:
    rng = np.random.default_rng(77_209)
    events = make_audio_events(duration)
    chunk_seconds = 2.0
    chunk_size = int(sample_rate * chunk_seconds)
    frame_count = int(round(duration * sample_rate))
    written = 0
    delay_lengths = [int(sample_rate * 0.19), int(sample_rate * 0.37), int(sample_rate * 0.61)]
    delay_l = [np.zeros(length, dtype=np.float32) for length in delay_lengths]
    delay_r = [np.zeros(length, dtype=np.float32) for length in delay_lengths]
    delay_pos = [0 for _ in delay_lengths]

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        while written < frame_count:
            n = min(chunk_size, frame_count - written)
            times = (written + np.arange(n, dtype=np.float64)) / sample_rate
            fraction = times / max(duration, 1e-8)

            left = np.zeros(n, dtype=np.float64)
            right = np.zeros(n, dtype=np.float64)
            root = 36.70 * (1.0 + 0.03 * np.sin(2 * math.pi * times / 89.0))
            ratios = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
            for index, ratio in enumerate(ratios):
                wobble = 1.0 + 0.0035 * np.sin(2 * math.pi * times / (31.0 + index * 17.0) + index)
                freq = root * ratio * wobble
                phase = 2.0 * math.pi * freq * times
                amp = 0.052 / (1.0 + index * 0.65)
                left += amp * np.sin(phase + index * 0.37)
                right += amp * np.sin(phase * (1.0 + (index - 2) * 0.00035) + index * 0.61)

            # Low drone opens out during the future act.
            future_env = np.sin(np.pi * np.clip((fraction - 0.72) / 0.18, 0.0, 1.0)) ** 2
            sub = (0.060 + 0.016 * future_env) * np.sin(2.0 * math.pi * 31.0 * times)
            left += sub
            right += sub * 0.97

            control_count = max(3, int(math.ceil(n / 480)) + 2)
            controls_l = rng.normal(0.0, 1.0, control_count)
            controls_r = rng.normal(0.0, 1.0, control_count)
            xp = np.linspace(0, n - 1, control_count)
            noise_l = np.interp(np.arange(n), xp, controls_l)
            noise_r = np.interp(np.arange(n), xp, controls_r)
            air_amp = 0.013 + 0.020 * np.sin(np.pi * np.clip((fraction - 0.46) / 0.34, 0.0, 1.0)) ** 2
            left += air_amp * noise_l
            right += air_amp * noise_r

            # Radio identity motif.
            pulse_env = np.clip(np.sin(2 * math.pi * times / 5.2), 0, 1) ** 8
            telemetry = 0.015 * np.sin(2 * math.pi * (470.0 + 24.0 * np.sin(times / 9.0)) * times) * pulse_env
            left += telemetry * 0.8
            right += telemetry * 1.0

            for event in events:
                if event.time_s <= times[-1] and event.time_s + event.decay_s * 8.0 >= times[0]:
                    ev_l, ev_r = _event_signal(event, times)
                    left += ev_l
                    right += ev_r

            intro = np.clip(times / max(4.0, duration * 0.015), 0.0, 1.0)
            outro = np.clip((duration - times) / max(6.0, duration * 0.020), 0.0, 1.0)
            master_env = np.sin(np.pi * 0.5 * intro) * np.sin(np.pi * 0.5 * outro)
            left *= master_env
            right *= master_env

            dry_l = left.astype(np.float32)
            dry_r = right.astype(np.float32)
            wet_l = np.zeros(n, dtype=np.float32)
            wet_r = np.zeros(n, dtype=np.float32)
            feedbacks = [0.18, 0.12, 0.085]
            crosses = [0.035, 0.045, 0.055]
            for tap, (buf_l, buf_r, feedback, cross) in enumerate(zip(delay_l, delay_r, feedbacks, crosses)):
                pos = delay_pos[tap]
                length = len(buf_l)
                for i in range(n):
                    dl = buf_l[pos]
                    dr = buf_r[pos]
                    wet_l[i] += dl
                    wet_r[i] += dr
                    buf_l[pos] = dry_l[i] + feedback * dl + cross * dr
                    buf_r[pos] = dry_r[i] + feedback * dr + cross * dl
                    pos += 1
                    if pos == length:
                        pos = 0
                delay_pos[tap] = pos
            left = dry_l + 0.18 * wet_l
            right = dry_r + 0.18 * wet_r
            left = np.tanh(left * 1.15) * 0.82
            right = np.tanh(right * 1.15) * 0.82
            stereo = np.column_stack((left, right))
            pcm = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)
            wav.writeframes(pcm.tobytes())
            written += n
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
    subprocess.run(list(command), check=True)


def write_chapter_srt(path: Path) -> Path:
    lines: List[str] = []
    for index, (fraction, title) in enumerate(CHAPTERS, 1):
        start = fraction * DURATION
        end_fraction = CHAPTERS[index][0] if index < len(CHAPTERS) else 1.0
        end = min(end_fraction * DURATION, start + max(7.0, DURATION * 0.05))
        lines += [str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", title, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_narration_srt(path: Path) -> Path:
    lines: List[str] = []
    for index, (start, end, text) in enumerate(VOICEOVER, 1):
        lines += [str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_metadata(renderer: MarsRenderer) -> Tuple[Path, Path]:
    title_description = OUTPUT_ROOT / "youtube_title_and_description.txt"
    description = f"""TITLE
{YOUTUBE_TITLE}

DESCRIPTION
A cinematic journey to Mars — from a distant red light to a world of lost water,
giants of stone, robot explorers, and the dream of future human footsteps.

This long-form film was rendered in Python using local planet textures when
available. If you placed files such as img/Mars.jpg, img/Earth.jpg, img/Jupiter.jpg,
and img/Saturn.jpg beside the script, those textures were wrapped onto rendered
spheres instead of being shown as flat images.

Scientific basis:
- Ancient Mars preserves strong evidence of past water activity.
- Olympus Mons is the largest volcano in the Solar System.
- Valles Marineris is one of the largest canyon systems known.
- Viking, Sojourner, Spirit, Opportunity, Curiosity, Perseverance, and Ingenuity
  are real milestones in Mars exploration.

Artistic note:
Camera paths, timing, colour, atmosphere, dust behaviour, and future habitat scenes
are cinematic visualisations.

Best experienced in a dark room with headphones.
"""
    title_description.write_text(description, encoding="utf-8")
    narration_path = DATA_DIR / "narration_script.txt"
    narration_path.write_text("\n".join(f"[{a:06.2f}-{b:06.2f}] {text}" for a, b, text in VOICEOVER), encoding="utf-8")
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "title": YOUTUBE_TITLE,
        "format": FORMAT,
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "duration_seconds": DURATION,
        "quick_mode": QUICK_MODE,
        "preview_only": PREVIEW_ONLY,
        "assets": renderer.asset_manifest(),
        "chapters": CHAPTERS,
        "narration": VOICEOVER,
        "facts": FACT_LINES,
        "science_note": "Mars past-water evidence, Olympus Mons, Valles Marineris, and real mission names are factual; camera paths and future-human scenes are artistic.",
    }
    manifest_path = DATA_DIR / "render_manifest.json"
    import json

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return title_description, manifest_path


def render_previews(renderer: MarsRenderer) -> List[Path]:
    fractions = [0.035, 0.13, 0.27, 0.42, 0.56, 0.72, 0.86, 0.97]
    paths: List[Path] = []
    for index, fraction in enumerate(tqdm(fractions, desc="Preview frames"), 1):
        t = fraction * DURATION
        frame = renderer.render_frame(t)
        path = PREVIEW_DIR / f"preview_{index:02d}_{t:07.2f}s.png"
        Image.fromarray(frame).save(path)
        paths.append(path)
        if index == 4:
            renderer.thumbnail_frame = frame
    return paths


def create_contact_sheet(paths: Sequence[Path]) -> Optional[Path]:
    if not paths:
        return None
    images = [Image.open(path).convert("RGB") for path in paths]
    thumb_w = 420 if WIDTH >= 1920 else 260
    thumb_h = int(thumb_w * HEIGHT / WIDTH)
    margin = 22
    columns = 3
    rows = int(math.ceil(min(len(images), 9) / columns))
    sheet = Image.new("RGB", (thumb_w * columns + margin * (columns + 1), thumb_h * rows + margin * (rows + 1)), (2, 3, 10))
    for index, image in enumerate(images[:9]):
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = margin + (index % columns) * (thumb_w + margin)
        y = margin + (index // columns) * (thumb_h + margin)
        sheet.paste(thumb, (x, y))
    path = PREVIEW_DIR / "red_frontier_contact_sheet.jpg"
    sheet.save(path, quality=94)
    return path


def create_thumbnail(renderer: MarsRenderer) -> Optional[Path]:
    if renderer.thumbnail_frame is None:
        return None
    image = Image.fromarray(renderer.thumbnail_frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((WIDTH * 0.05, HEIGHT * 0.07, WIDTH * 0.58, HEIGHT * 0.30), radius=max(12, int(18 * FS)), fill=(0, 0, 0, 104))
    image.alpha_composite(overlay)
    draw_text(image, "RED FRONTIER", (WIDTH * 0.08, HEIGHT * 0.14), 54, (248, 248, 252, 255), True, True)
    draw_text(image, "MARS BEFORE FOOTSTEPS", (WIDTH * 0.08, HEIGHT * 0.22), 22, (244, 148, 108, 255), True)
    path = OUTPUT_ROOT / f"thumbnail_{FORMAT}.jpg"
    image.convert("RGB").save(path, quality=95)
    return path


def render_video(renderer: MarsRenderer) -> Path:
    basename = f"red_frontier_{FORMAT}_{int(DURATION)}s"
    raw_path = OUTPUT_ROOT / f"{basename}_silent.mp4"
    final_path = OUTPUT_ROOT / f"{basename}_final.mp4"
    audio_path = AUDIO_DIR / f"{basename}_soundtrack.wav"
    chapter_srt = OUTPUT_ROOT / f"{basename}_chapters.srt"
    narration_srt = OUTPUT_ROOT / f"{basename}_narration.srt"
    write_chapter_srt(chapter_srt)
    write_narration_srt(narration_srt)

    frame_count = int(round(DURATION * FPS))
    times = np.arange(frame_count, dtype=float) / FPS
    with iio.get_writer(
        raw_path,
        fps=FPS,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=["-crf", "17", "-preset", "medium", "-movflags", "+faststart"],
    ) as writer:
        for idx, t in enumerate(tqdm(times, desc="Rendering RED FRONTIER")):
            frame = renderer.render_frame(float(t))
            writer.append_data(frame)
            if renderer.thumbnail_frame is None and idx == int(frame_count * 0.42):
                renderer.thumbnail_frame = frame

    temp_score = render_ambient_audio(audio_path, DURATION, 48_000)
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        shutil.copyfile(raw_path, final_path)
        return final_path

    # Optional external music mix.
    music_path = EXTERNAL_MUSIC if EXTERNAL_MUSIC and Path(EXTERNAL_MUSIC).exists() else None
    voice_path = EXTERNAL_VOICEOVER if EXTERNAL_VOICEOVER and Path(EXTERNAL_VOICEOVER).exists() else None

    mix_inputs = ["-i", str(raw_path), "-i", str(temp_score)]
    input_count = 2
    if music_path:
        mix_inputs += ["-i", str(music_path)]
        input_count += 1
    if voice_path:
        mix_inputs += ["-i", str(voice_path)]
        input_count += 1

    if input_count == 2:
        run_ffmpeg([ffmpeg, "-y", *mix_inputs, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", str(final_path)])
        return final_path

    amix_parts = []
    audio_labels = []
    # base score
    amix_parts.append("[1:a]volume=1.0[a1]")
    audio_labels.append("[a1]")
    if music_path:
        amix_parts.append(f"[{2 if not voice_path else 2}:a]volume=0.42[a2]")
        audio_labels.append("[a2]")
    if voice_path:
        voice_index = 3 if music_path else 2
        amix_parts.append(f"[{voice_index}:a]volume=1.25[a3]")
        audio_labels.append("[a3]")
    amix_parts.append("".join(audio_labels) + f"amix=inputs={len(audio_labels)}:duration=longest:normalize=0[mix]")
    filter_complex = ";".join(amix_parts)
    run_ffmpeg([ffmpeg, "-y", *mix_inputs, "-filter_complex", filter_complex, "-map", "0:v:0", "-map", "[mix]", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", str(final_path)])
    return final_path


def main() -> None:
    print("Starting RED FRONTIER: MARS BEFORE FOOTSTEPS")
    print("Format:", FORMAT)
    print("Quick mode:", QUICK_MODE)
    print("Preview only:", PREVIEW_ONLY)
    print("Duration:", DURATION)
    print("Local img folder:", IMG_DIR)
    renderer = MarsRenderer()
    title_desc, manifest = write_metadata(renderer)
    print("Metadata:", title_desc.resolve())
    print("Manifest:", manifest.resolve())
    preview_paths = render_previews(renderer)
    contact_sheet = create_contact_sheet(preview_paths)
    thumb = create_thumbnail(renderer)
    if contact_sheet:
        print("Contact sheet:", contact_sheet.resolve())
    if thumb:
        print("Thumbnail:", thumb.resolve())
    if not PREVIEW_ONLY:
        final_path = render_video(renderer)
        print("Final film:", final_path.resolve())
    else:
        print("Preview-only mode complete; no final movie encoded.")
    print("Output directory:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
