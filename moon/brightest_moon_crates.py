from __future__ import annotations

"""
The Moon's Brightest Craters — And Why They Outshine the Surface
The Moon's Brightest Craters — And Why They Outshine the Surface

A vertical, cinematic YouTube Shorts renderer with minimal text, procedural
lunar visuals, a procedural stereo soundtrack, and a small real-metadata table
for prominent high-albedo / rayed lunar craters.



The code avoids presenting a made-up global albedo ranking. Aristarchus is the
headline feature; the others are shown as prominent bright/rayed examples.

Official source pages used for framing and metadata are written to the output
summary and README. The lunar surface imagery in this renderer is procedural,
not a replacement for calibrated LRO reflectance products.

Usage
-----
Standard vertical render:
    python the_moons_brightest_craters_cinematic_short.py



"""

import csv
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

QUICK_MODE = os.environ.get("MOON_CRATERS_SHORT_QUICK", "0") == "1"
FOUR_K_MODE = os.environ.get("MOON_CRATERS_SHORT_4K", "0") == "1" and not QUICK_MODE

OUTPUT_ROOT = Path("the_moons_brightest_craters_output")
DATA_DIR = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_DIR, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
   
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
FPS = int(CONFIG["fps"])
DURATION = float(CONFIG["duration_s"])
FRAME_COUNT = int(round(FPS * DURATION))

COLORS = {
    "white": (245, 247, 244),
    "muted": (177, 188, 197),
    "ice": (171, 222, 255),
    "gold": (245, 191, 94),
    "cyan": (88, 218, 244),
    "violet": (164, 120, 230),
    "dark": (3, 5, 12),
    "moon_dark": (45, 47, 51),
    "moon_mid": (119, 121, 119),
    "moon_light": (205, 208, 201),
}

# Coordinates use positive east longitude. West longitudes are negative.
# Ages are approximate and intentionally left blank where this short does not
# need a robust single-value age claim.
CRATERS: List[Dict[str, Any]] = [
    {
        "name": "Aristarchus",
        "lat": 23.7,
        "lon": -47.4,
        "diameter_km": 40.0,
        "age_myr": None,
        "role": "headline",
        "note": "One of the brightest features on the Moon; high-reflectance rays and exposed materials.",
        "source": "https://science.nasa.gov/resource/aristarchus-crater-2/",
    },
    {
        "name": "Tycho",
        "lat": -43.3,
        "lon": -11.4,
        "diameter_km": 85.0,
        "age_myr": 110.0,
        "role": "major_ray_crater",
        "note": "A young, prominent crater with bright rays extending across much of the nearside.",
        "source": "https://science.nasa.gov/resource/tycho-crater-on-the-moon-labeled/",
    },
    {
        "name": "Copernicus",
        "lat": 9.7,
        "lon": -20.1,
        "diameter_km": 93.0,
        "age_myr": 800.0,
        "role": "major_ray_crater",
        "note": "A large nearside complex crater with an extensive light-colored ejecta-ray system.",
        "source": "https://apod.nasa.gov/apod/ap010809.html",
    },
    {
        "name": "Kepler",
        "lat": 8.1,
        "lon": -38.0,
        "diameter_km": 32.0,
        "age_myr": None,
        "role": "ray_crater",
        "note": "Bright rays cross the darker basaltic terrain of Oceanus Procellarum.",
        "source": "https://science.nasa.gov/image-article/apod-2023-december-7-orion-and-the-ocean-of-storms/",
    },
    {
        "name": "Proclus",
        "lat": 16.1,
        "lon": 46.8,
        "diameter_km": 28.0,
        "age_myr": None,
        "role": "ray_crater",
        "note": "A conspicuous bright crater with an asymmetric ray pattern near Mare Crisium.",
        "source": "https://science.nasa.gov/moon/lunar-craters/",
    },
]

FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.0, 3.5, "Some lunar craters look almost white against the Moon's ancient surface."),
    (4.0, 7.5, "But they are not glowing. They are reflecting more sunlight."),
    (9.0, 12.8, "Aristarchus is one of the brightest features on the entire Moon."),
    (15.0, 18.8, "Tycho throws brilliant ejecta rays across much of the lunar nearside."),
    (21.0, 24.8, "Copernicus, Kepler, and Proclus reveal the same impact signature."),
    (27.0, 30.8, "The impact excavates fresh rock and sprays immature material over older terrain."),
    (33.0, 36.8, "Solar wind and micrometeoroids slowly weather exposed soil, making old surfaces darker."),
    (39.0, 42.8, "Composition and Sun angle matter too — brightness is not only about crater age."),
    (45.0, 49.0, "Bright rays are temporary geological fingerprints of comparatively recent impacts."),
]

SHOT_PLAN_FULL: List[Tuple[str, float, float]] = [
    ("intro", 0.0, 4.0),
    ("not_glowing", 4.0, 8.0),
    ("aristarchus", 8.0, 14.0),
    ("tycho", 14.0, 20.0),
    ("crater_gallery", 20.0, 26.0),
    ("fresh_ejecta", 26.0, 32.0),
    ("weathering", 32.0, 38.0),
    ("geometry", 38.0, 44.0),
    ("finale", 44.0, 52.0),
]

if QUICK_MODE:
    scale = DURATION / 52.0
    CAPTIONS = [(a * scale, b * scale, text) for a, b, text in FULL_CAPTIONS]
    SHOT_PLAN = [(name, a * scale, b * scale) for name, a, b in SHOT_PLAN_FULL]
else:
    CAPTIONS = FULL_CAPTIONS
    SHOT_PLAN = SHOT_PLAN_FULL


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
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)
        if bbox[2] - bbox[0] <= x1 - x0:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    y = y0
    for line in lines:
        draw.text((x0, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 220))
        bbox = draw.textbbox((x0, y), line, font=font, stroke_width=2)
        y += (bbox[3] - bbox[1]) + max(4, int(size * 0.20))
        if y > y1:
            break


def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 140):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius = max(8, int((box[3] - box[1]) * 0.18))
    draw.rounded_rectangle(box, radius=radius, fill=(2, 5, 13, alpha), outline=COLORS["ice"] + (42,), width=1)
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
        lines.extend([str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
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
# Data products
# -----------------------------------------------------------------------------


def save_crater_csv(path: Path):
    fields = ["name", "lat", "lon", "diameter_km", "age_myr", "role", "note", "source"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for crater in CRATERS:
            writer.writerow(crater)


# -----------------------------------------------------------------------------
# Renderer
# -----------------------------------------------------------------------------


class LunarCraterScene:
    def __init__(self):
        self.stars = self._make_stars(int(CONFIG["stars"]), seed=2077)
        self.moon_texture = self._make_moon_texture(seed=1969)
        self.closeup_textures = {
            crater["name"]: self._make_crater_closeup(crater, seed=1000 + index * 71)
            for index, crater in enumerate(CRATERS)
        }

    @staticmethod
    def _make_stars(count: int, seed: int) -> List[Star]:
        rng = np.random.default_rng(seed)
        stars: List[Star] = []
        for _ in range(count):
            stars.append(
                Star(
                    x=float(rng.uniform(0, OUT_W)),
                    y=float(rng.uniform(0, OUT_H)),
                    radius=float(rng.uniform(0.25, 2.0) * OUT_W / 1080),
                    alpha=float(rng.uniform(18, 135)),
                    phase=float(rng.uniform(0, 2 * math.pi)),
                )
            )
        return stars

    @staticmethod
    def _make_moon_texture(seed: int, size: int = 1152) -> Image.Image:
        rng = np.random.default_rng(seed)
        base = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
        image = Image.fromarray(np.uint8(np.clip((base - base.min()) / max(float(np.ptp(base)), 1e-6) * 255, 0, 255)))
        multi = np.zeros((size, size), dtype=np.float32)
        for blur, weight in [(46, 0.45), (22, 0.35), (9, 0.20), (3, 0.08)]:
            blurred = image.filter(ImageFilter.GaussianBlur(blur))
            multi += np.asarray(blurred, dtype=np.float32) / 255.0 * weight
        multi -= multi.min()
        multi /= max(float(multi.max()), 1e-6)

        yy, xx = np.mgrid[0:size, 0:size]
        # Procedural dark maria. These are artistic placements, not a calibrated map.
        maria = np.ones((size, size), dtype=np.float32)
        for cx, cy, rx, ry, depth in [
            (0.64, 0.37, 0.20, 0.14, 0.38),
            (0.43, 0.38, 0.17, 0.13, 0.31),
            (0.60, 0.57, 0.18, 0.12, 0.28),
            (0.32, 0.54, 0.12, 0.10, 0.24),
            (0.74, 0.49, 0.11, 0.09, 0.22),
        ]:
            ellipse = ((xx / size - cx) / rx) ** 2 + ((yy / size - cy) / ry) ** 2
            maria -= np.exp(-ellipse * 2.2) * depth
        texture = np.clip(0.40 + multi * 0.50, 0.0, 1.0) * np.clip(maria, 0.46, 1.0)

        # Fine craterlets.
        for _ in range(620 if not QUICK_MODE else 230):
            cx = int(rng.uniform(0, size))
            cy = int(rng.uniform(0, size))
            radius = int(rng.uniform(1.0, 8.0))
            y0 = max(0, cy - radius * 2)
            y1 = min(size, cy + radius * 2 + 1)
            x0 = max(0, cx - radius * 2)
            x1 = min(size, cx + radius * 2 + 1)
            sy, sx = np.mgrid[y0:y1, x0:x1]
            dist = np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2) / max(radius, 1)
            depression = -0.16 * np.exp(-((dist - 0.75) / 0.45) ** 2)
            rim = 0.10 * np.exp(-((dist - 1.05) / 0.25) ** 2)
            texture[y0:y1, x0:x1] += depression + rim

        texture = np.clip(texture, 0.0, 1.0)
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(texture * 224, 0, 255)
        rgb[..., 1] = np.clip(texture * 226, 0, 255)
        rgb[..., 2] = np.clip(texture * 219, 0, 255)
        return Image.fromarray(rgb, mode="RGB")

    @staticmethod
    def _make_crater_closeup(crater: Dict[str, Any], seed: int, size: int = 1000) -> Image.Image:
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:size, 0:size]
        cx, cy = size * 0.5, size * 0.51
        radius = size * (0.18 if crater["name"] != "Tycho" else 0.20)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        angle = np.arctan2(yy - cy, xx - cx)

        noise = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
        noise_img = Image.fromarray(np.uint8(np.clip((noise - noise.min()) / max(float(np.ptp(noise)), 1e-6) * 255, 0, 255)))
        terrain = np.asarray(noise_img.filter(ImageFilter.GaussianBlur(10)), dtype=np.float32) / 255.0
        terrain = 0.35 + 0.45 * terrain

        # Circular crater depression, bright rim, terraces, and central peak.
        terrain -= 0.34 * np.exp(-((dist / radius) / 0.72) ** 4)
        terrain += 0.35 * np.exp(-((dist - radius) / (radius * 0.10)) ** 2)
        terrain += 0.10 * np.cos(dist / max(radius * 0.10, 1.0) * math.pi) * np.exp(-((dist / radius - 0.70) / 0.28) ** 2)
        terrain += 0.22 * np.exp(-((dist / (radius * 0.18)) ** 2))

        # Rays. Aristarchus and Proclus receive more asymmetric systems.
        ray_count = {"Aristarchus": 13, "Tycho": 18, "Copernicus": 14, "Kepler": 11, "Proclus": 9}.get(crater["name"], 10)
        asym = 0.46 if crater["name"] == "Proclus" else 0.0
        for index in range(ray_count):
            theta = 2 * math.pi * index / ray_count + rng.uniform(-0.10, 0.10)
            if asym and math.cos(theta) < -0.15:
                continue
            angular = np.angle(np.exp(1j * (angle - theta)))
            width = rng.uniform(0.018, 0.055)
            radial_gate = 1.0 / (1.0 + np.exp(-(dist - radius * 0.82) / max(radius * 0.05, 1.0)))
            falloff = np.exp(-dist / (radius * rng.uniform(2.0, 3.6)))
            terrain += np.exp(-(angular / width) ** 2) * falloff * radial_gate * rng.uniform(0.14, 0.30)

        # Dark impact-melt-like patches.
        terrain -= 0.09 * np.exp(-((dist / (radius * 0.50)) ** 2)) * (0.5 + 0.5 * np.sin(angle * 5.0 + 0.7))
        terrain = np.clip(terrain, 0.0, 1.0)

        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        tint = 1.03 if crater["name"] == "Aristarchus" else 1.0
        rgb[..., 0] = np.clip(terrain * 235 * tint, 0, 255)
        rgb[..., 1] = np.clip(terrain * 238 * tint, 0, 255)
        rgb[..., 2] = np.clip(terrain * 232 * tint, 0, 255)
        return Image.fromarray(rgb, mode="RGB")

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", (OUT_W, OUT_H), COLORS["dark"] + (255,))
        draw = ImageDraw.Draw(image)
        for y in range(OUT_H):
            p = y / max(OUT_H - 1, 1)
            draw.line((0, y, OUT_W, y), fill=(int(lerp(2, 8, p)), int(lerp(4, 9, p)), int(lerp(12, 20, p)), 255))
        for star in self.stars:
            alpha = int(star.alpha * (0.72 + 0.28 * math.sin(t * 1.25 + star.phase)))
            r = star.radius
            draw.ellipse((star.x - r, star.y - r, star.x + r, star.y + r), fill=COLORS["white"] + (alpha,))
        return image

    @staticmethod
    def lonlat_to_disc(lon: float, lat: float, center: Tuple[int, int], radius: float) -> Optional[Tuple[float, float]]:
        lon_r = math.radians(lon)
        lat_r = math.radians(lat)
        # Orthographic view centered at 0°N, 0°E.
        visible = math.cos(lat_r) * math.cos(lon_r)
        if visible <= 0:
            return None
        x = center[0] + radius * math.cos(lat_r) * math.sin(lon_r)
        y = center[1] - radius * math.sin(lat_r)
        return x, y

    def draw_moon(
        self,
        image: Image.Image,
        center: Tuple[int, int],
        radius: int,
        phase: float = 0.92,
        rotation_deg: float = 0.0,
        show_craters: bool = True,
        ray_strength: float = 1.0,
    ):
        size = radius * 2
        texture = self.moon_texture.resize((size, size), Image.Resampling.LANCZOS)
        if rotation_deg:
            texture = texture.rotate(rotation_deg, resample=Image.Resampling.BICUBIC)
        arr = np.asarray(texture, dtype=np.float32) / 255.0
        yy, xx = np.mgrid[0:size, 0:size]
        nx = (xx - radius + 0.5) / radius
        ny = (yy - radius + 0.5) / radius
        r2 = nx * nx + ny * ny
        mask = r2 <= 1.0
        z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

        # phase=1 means near-full. Light source shifts slightly for cinematic relief.
        sx = lerp(-0.46, -0.08, phase)
        sz = math.sqrt(max(0.0, 1.0 - sx * sx))
        illumination = np.clip(nx * sx + z * sz, 0.0, 1.0)
        limb = np.clip(0.36 + 0.78 * z, 0.0, 1.0)
        shade = illumination * limb
        rgb = np.zeros((size, size, 4), dtype=np.uint8)
        rgb[..., :3] = np.clip(arr * shade[..., None] * 255.0, 0, 255).astype(np.uint8)
        rgb[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
        moon = Image.fromarray(rgb, mode="RGBA")

        # Draw highlighted crater/ray systems in disc coordinates.
        if show_craters:
            overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            local_center = (radius, radius)
            for crater in CRATERS:
                pos = self.lonlat_to_disc(crater["lon"], crater["lat"], local_center, radius * 0.94)
                if pos is None:
                    continue
                x, y = pos
                base = max(3.0, crater["diameter_km"] / 12.5 * radius / 280.0)
                emphasis = 1.25 if crater["name"] == "Aristarchus" else 1.0
                d = base * emphasis
                # bright core/rim
                od.ellipse((x - d, y - d, x + d, y + d), fill=COLORS["white"] + (235,), outline=COLORS["ice"] + (220,), width=max(1, int(radius * 0.006)))
                # ray systems
                ray_count = {"Aristarchus": 12, "Tycho": 18, "Copernicus": 13, "Kepler": 9, "Proclus": 7}[crater["name"]]
                ray_len = radius * ({"Aristarchus": 0.20, "Tycho": 0.63, "Copernicus": 0.34, "Kepler": 0.21, "Proclus": 0.22}[crater["name"]])
                for index in range(ray_count):
                    angle = 2 * math.pi * index / ray_count + (0.22 if crater["name"] == "Proclus" else 0.0)
                    if crater["name"] == "Proclus" and math.cos(angle) < -0.3:
                        continue
                    end_x = x + math.cos(angle) * ray_len * (0.72 + 0.28 * math.sin(index * 1.8 + 0.5))
                    end_y = y + math.sin(angle) * ray_len * (0.72 + 0.28 * math.cos(index * 1.5))
                    od.line((x, y, end_x, end_y), fill=COLORS["white"] + (int(72 * ray_strength),), width=max(1, int(radius * 0.004)))
            glow = overlay.filter(ImageFilter.GaussianBlur(max(2, int(radius * 0.025))))
            moon.alpha_composite(glow)
            moon.alpha_composite(overlay)

        # limb glow
        glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse((3, 3, size - 3, size - 3), outline=COLORS["ice"] + (78,), width=max(1, int(radius * 0.018)))
        moon.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(3, int(radius * 0.035)))))
        image.alpha_composite(moon, (center[0] - radius, center[1] - radius))

    def draw_crater_closeup(
        self,
        image: Image.Image,
        crater_name: str,
        box: Tuple[int, int, int, int],
        zoom: float = 1.0,
        pan: Tuple[float, float] = (0.0, 0.0),
        brightness: float = 1.0,
    ):
        x0, y0, x1, y1 = box
        source = self.closeup_textures[crater_name]
        crop_w = max(100, int(source.width / zoom))
        crop_h = max(100, int(source.height / zoom * ((y1 - y0) / max(x1 - x0, 1))))
        cx = int(source.width * (0.5 + pan[0]))
        cy = int(source.height * (0.5 + pan[1]))
        cx = int(np.clip(cx, crop_w // 2, source.width - crop_w // 2))
        cy = int(np.clip(cy, crop_h // 2, source.height - crop_h // 2))
        crop = source.crop((cx - crop_w // 2, cy - crop_h // 2, cx + crop_w // 2, cy + crop_h // 2))
        crop = crop.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
        crop = ImageEnhance.Brightness(crop).enhance(brightness).convert("RGBA")

        mask = Image.new("L", crop.size, 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle((0, 0, crop.width - 1, crop.height - 1), radius=int(min(crop.size) * 0.05), fill=255)
        crop.putalpha(mask)
        image.alpha_composite(crop, (x0, y0))

        frame = Image.new("RGBA", image.size, (0, 0, 0, 0))
        fd = ImageDraw.Draw(frame)
        fd.rounded_rectangle(box, radius=int(min(x1 - x0, y1 - y0) * 0.05), outline=COLORS["ice"] + (72,), width=max(1, int(OUT_W / 700)))
        image.alpha_composite(frame)

    def draw_crater_label(self, image: Image.Image, crater: Dict[str, Any], y: int, descriptor: str):
        draw_text(image, crater["name"].upper(), (OUT_W // 2, y), size=max(18, int(58 * OUT_W / 1080)), fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=2)
        draw_text(image, descriptor, (OUT_W // 2, y + int(OUT_H * 0.036)), size=max(10, int(25 * OUT_W / 1080)), fill=COLORS["gold"] + (235,), anchor="ma", stroke=1)

    def draw_impact_sequence(self, image: Image.Image, local: float):
        horizon = int(OUT_H * 0.66)
        # lunar ground
        draw = ImageDraw.Draw(image)
        for y in range(horizon, OUT_H):
            p = (y - horizon) / max(OUT_H - horizon, 1)
            value = int(lerp(54, 20, p))
            draw.line((0, y, OUT_W, y), fill=(value, value, value + 2, 255))

        impact_x = int(OUT_W * 0.50)
        impact_y = horizon + int(OUT_H * 0.055)
        flash = smoothstep(min(local * 2.0, 1.0)) * (1.0 - smoothstep(max((local - 0.40) / 0.60, 0.0)))
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        for radius, alpha in [
            (int(OUT_W * 0.20), int(35 * flash)),
            (int(OUT_W * 0.11), int(80 * flash)),
            (int(OUT_W * 0.045), int(180 * flash)),
        ]:
            ld.ellipse((impact_x - radius, impact_y - radius, impact_x + radius, impact_y + radius), fill=COLORS["white"] + (alpha,))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(4, int(20 * OUT_W / 1080)))))

        # ejecta rays / ballistic arcs
        ejecta = Image.new("RGBA", image.size, (0, 0, 0, 0))
        ed = ImageDraw.Draw(ejecta)
        ray_reveal = smoothstep((local - 0.18) / 0.72)
        for i in range(22):
            angle = math.radians(200 + i * (140 / 21))
            length = OUT_W * (0.18 + 0.34 * abs(math.sin(i * 0.73))) * ray_reveal
            ex = impact_x + math.cos(angle) * length
            ey = impact_y + math.sin(angle) * length * 0.68
            ed.line((impact_x, impact_y, ex, ey), fill=COLORS["white"] + (90,), width=max(1, int(OUT_W / 650)))
            d = max(1, int(3 * OUT_W / 1080))
            ed.ellipse((ex - d, ey - d, ex + d, ey + d), fill=COLORS["gold"] + (170,))
        image.alpha_composite(ejecta.filter(ImageFilter.GaussianBlur(max(1, int(2 * OUT_W / 1080)))))

        # final crater ring
        crater_r = int(OUT_W * 0.095 * ray_reveal)
        draw.ellipse((impact_x - crater_r, impact_y - crater_r * 0.30, impact_x + crater_r, impact_y + crater_r * 0.30), fill=(25, 25, 27, 255), outline=COLORS["white"] + (160,), width=max(1, int(4 * OUT_W / 1080)))

    def draw_weathering_comparison(self, image: Image.Image, local: float):
        y0, y1 = int(OUT_H * 0.26), int(OUT_H * 0.72)
        left_box = (int(OUT_W * 0.06), y0, int(OUT_W * 0.48), y1)
        right_box = (int(OUT_W * 0.52), y0, int(OUT_W * 0.94), y1)
        self.draw_crater_closeup(image, "Tycho", left_box, zoom=1.45, brightness=1.15)
        self.draw_crater_closeup(image, "Tycho", right_box, zoom=1.45, brightness=lerp(1.10, 0.58, local))

        # add reddening/darkening to weathered side
        tint = Image.new("RGBA", image.size, (0, 0, 0, 0))
        td = ImageDraw.Draw(tint)
        td.rounded_rectangle(right_box, radius=int((right_box[2] - right_box[0]) * 0.05), fill=(70, 40, 26, int(70 * local)))
        image.alpha_composite(tint)

        draw_text(image, "FRESH / IMMATURE", ((left_box[0] + left_box[2]) // 2, int(OUT_H * 0.75)), size=max(10, int(23 * OUT_W / 1080)), fill=COLORS["ice"] + (235,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "SPACE-WEATHERED", ((right_box[0] + right_box[2]) // 2, int(OUT_H * 0.75)), size=max(10, int(23 * OUT_W / 1080)), fill=COLORS["gold"] + (235,), bold=True, anchor="ma", stroke=1)

        # particle streaks above the weathered side
        particle = Image.new("RGBA", image.size, (0, 0, 0, 0))
        pd = ImageDraw.Draw(particle)
        rng = np.random.default_rng(500)
        for i in range(36 if not QUICK_MODE else 18):
            x = rng.uniform(right_box[0], right_box[2])
            y = rng.uniform(y0 - OUT_H * 0.16, y0 + OUT_H * 0.08)
            length = rng.uniform(14, 55) * OUT_W / 1080
            pd.line((x, y, x - length * 0.55, y + length), fill=COLORS["gold"] + (int(40 + 80 * local),), width=1)
        image.alpha_composite(particle.filter(ImageFilter.GaussianBlur(1)))

    def draw_phase_geometry(self, image: Image.Image, local: float):
        phases = [0.58, 0.76, 0.96]
        centers = [(int(OUT_W * 0.20), int(OUT_H * 0.45)), (int(OUT_W * 0.50), int(OUT_H * 0.45)), (int(OUT_W * 0.80), int(OUT_H * 0.45))]
        radius = int(OUT_W * 0.14)
        for index, (phase, center) in enumerate(zip(phases, centers)):
            reveal = clamp(local * 1.4 - index * 0.20)
            if reveal <= 0:
                continue
            self.draw_moon(image, center, radius, phase=phase, rotation_deg=index * 2.0, show_craters=True, ray_strength=reveal)
        draw_text(image, "SUN ANGLE CHANGES APPARENT CONTRAST", (OUT_W // 2, int(OUT_H * 0.67)), size=max(12, int(31 * OUT_W / 1080)), fill=COLORS["white"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "rays often stand out most near the full Moon", (OUT_W // 2, int(OUT_H * 0.715)), size=max(10, int(23 * OUT_W / 1080)), fill=COLORS["gold"] + (225,), anchor="ma", stroke=1)

    def draw_caption(self, image: Image.Image, caption: str, t: float):
        start, end = 0.0, 0.0
        for a, b, text in CAPTIONS:
            if text == caption and a <= t < b:
                start, end = a, b
                break
        fade_in = clamp((t - start) / max(0.08 if QUICK_MODE else 0.34, 1e-6))
        fade_out = clamp((end - t) / max(0.10 if QUICK_MODE else 0.44, 1e-6))
        alpha = int(225 * min(fade_in, fade_out, 1.0))
        if alpha <= 0:
            return
        box = (int(OUT_W * 0.07), int(OUT_H * 0.76), int(OUT_W * 0.93), int(OUT_H * 0.86))
        panel(image, box, alpha=min(105, alpha // 2))
        draw_wrapped_text(
            image,
            caption,
            (box[0] + int(OUT_W * 0.025), box[1] + int(OUT_H * 0.012), box[2] - int(OUT_W * 0.025), box[3] - int(OUT_H * 0.010)),
            size=max(12, int(30 * OUT_W / 1080)),
            fill=COLORS["white"] + (alpha,),
        )

    def frame(self, t: float) -> np.ndarray:
        shot, start, end = get_shot(t)
        local = smoothstep((t - start) / max(end - start, 1e-9))
        image = self.background(t)

        if shot == "intro":
            self.draw_moon(image, (OUT_W // 2, int(OUT_H * 0.43)), int(OUT_W * lerp(0.27, 0.31, local)), phase=0.94, rotation_deg=-2 + 4 * local, show_craters=True, ray_strength=0.85)
            draw_text(image, "THE MOON'S BRIGHTEST", (OUT_W // 2, int(OUT_H * 0.105)), size=max(18, int(67 * OUT_W / 1080)), fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=2)
            draw_text(image, "CRATERS", (OUT_W // 2, int(OUT_H * 0.155)), size=max(20, int(82 * OUT_W / 1080)), fill=COLORS["gold"] + (245,), bold=True, anchor="ma", stroke=2)

        elif shot == "not_glowing":
            phase = lerp(0.60, 0.97, local)
            self.draw_moon(image, (OUT_W // 2, int(OUT_H * 0.43)), int(OUT_W * 0.31), phase=phase, rotation_deg=2.0, show_craters=True, ray_strength=local)
            draw_text(image, "REFLECTED SUNLIGHT", (OUT_W // 2, int(OUT_H * 0.17)), size=max(14, int(40 * OUT_W / 1080)), fill=COLORS["ice"] + (240,), bold=True, anchor="ma", stroke=1)
            draw_text(image, "not lunar luminescence", (OUT_W // 2, int(OUT_H * 0.21)), size=max(10, int(24 * OUT_W / 1080)), fill=COLORS["muted"] + (220,), anchor="ma", stroke=1)

        elif shot == "aristarchus":
            self.draw_crater_closeup(image, "Aristarchus", (int(OUT_W * 0.05), int(OUT_H * 0.18), int(OUT_W * 0.95), int(OUT_H * 0.72)), zoom=lerp(1.25, 1.72, local), pan=(lerp(-0.03, 0.02, local), 0.0), brightness=1.10)
            crater = CRATERS[0]
            self.draw_crater_label(image, crater, int(OUT_H * 0.105), "~40 km • one of the Moon's brightest features")

        elif shot == "tycho":
            self.draw_crater_closeup(image, "Tycho", (int(OUT_W * 0.05), int(OUT_H * 0.18), int(OUT_W * 0.95), int(OUT_H * 0.72)), zoom=lerp(1.08, 1.50, local), pan=(0.0, lerp(-0.02, 0.02, local)), brightness=1.04)
            crater = CRATERS[1]
            self.draw_crater_label(image, crater, int(OUT_H * 0.105), "~85 km • ~110 million years • vast ray system")

        elif shot == "crater_gallery":
            names = ["Copernicus", "Kepler", "Proclus"]
            boxes = [
                (int(OUT_W * 0.06), int(OUT_H * 0.22), int(OUT_W * 0.48), int(OUT_H * 0.51)),
                (int(OUT_W * 0.52), int(OUT_H * 0.22), int(OUT_W * 0.94), int(OUT_H * 0.51)),
                (int(OUT_W * 0.29), int(OUT_H * 0.54), int(OUT_W * 0.71), int(OUT_H * 0.80)),
            ]
            for index, (name, box) in enumerate(zip(names, boxes)):
                reveal = clamp(local * 1.6 - index * 0.22)
                if reveal <= 0:
                    continue
                self.draw_crater_closeup(image, name, box, zoom=1.45, brightness=0.96 + 0.08 * reveal)
                draw_text(image, name.upper(), ((box[0] + box[2]) // 2, box[3] - int(OUT_H * 0.025)), size=max(10, int(24 * OUT_W / 1080)), fill=COLORS["white"] + (235,), bold=True, anchor="mm", stroke=1)
            draw_text(image, "PROMINENT HIGH-ALBEDO RAY CRATERS", (OUT_W // 2, int(OUT_H * 0.135)), size=max(12, int(34 * OUT_W / 1080)), fill=COLORS["gold"] + (240,), bold=True, anchor="ma", stroke=1)

        elif shot == "fresh_ejecta":
            self.draw_impact_sequence(image, local)
            draw_text(image, "FRESH EJECTA", (OUT_W // 2, int(OUT_H * 0.17)), size=max(15, int(46 * OUT_W / 1080)), fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=1)
            draw_text(image, "newly exposed material reflects strongly", (OUT_W // 2, int(OUT_H * 0.215)), size=max(10, int(23 * OUT_W / 1080)), fill=COLORS["gold"] + (225,), anchor="ma", stroke=1)

        elif shot == "weathering":
            self.draw_weathering_comparison(image, local)
            draw_text(image, "SPACE WEATHERING", (OUT_W // 2, int(OUT_H * 0.15)), size=max(15, int(44 * OUT_W / 1080)), fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=1)
            draw_text(image, "solar wind + micrometeoroids", (OUT_W // 2, int(OUT_H * 0.195)), size=max(10, int(23 * OUT_W / 1080)), fill=COLORS["gold"] + (225,), anchor="ma", stroke=1)

        elif shot == "geometry":
            self.draw_phase_geometry(image, local)

        elif shot == "finale":
            self.draw_moon(image, (OUT_W // 2, int(OUT_H * 0.42)), int(OUT_W * 0.32), phase=0.98, rotation_deg=-1.0, show_craters=True, ray_strength=1.0)
            draw_text(image, "BRIGHTNESS IS A GEOLOGICAL CLOCK", (OUT_W // 2, int(OUT_H * 0.13)), size=max(13, int(39 * OUT_W / 1080)), fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=1)
            draw_text(image, "freshness • composition • illumination", (OUT_W // 2, int(OUT_H * 0.18)), size=max(10, int(24 * OUT_W / 1080)), fill=COLORS["gold"] + (230,), anchor="ma", stroke=1)
            draw_text(image, "ARISTARCHUS • TYCHO • COPERNICUS • KEPLER • PROCLUS", (OUT_W // 2, int(OUT_H * 0.71)), size=max(8, int(18 * OUT_W / 1080)), fill=COLORS["ice"] + (205,), anchor="ma", stroke=1)

        caption = caption_at(t)
        if caption:
            self.draw_caption(image, caption, t)

        draw_text(image, "NASA / LRO science framing • procedural visualization", (int(OUT_W * 0.025), int(OUT_H * 0.988)), size=max(7, int(12 * OUT_W / 1080)), fill=COLORS["muted"] + (110,), anchor="ls", stroke=1)

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        arr = np.clip(arr * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        graded = Image.fromarray(arr)
        graded = ImageEnhance.Contrast(graded).enhance(float(CONFIG["contrast"]))
        graded = ImageEnhance.Color(graded).enhance(float(CONFIG["saturation"]))
        return np.asarray(graded)


# -----------------------------------------------------------------------------
# Procedural audio
# -----------------------------------------------------------------------------


def make_envelope(length: int, attack: float = 0.08, release: float = 0.60) -> np.ndarray:
    env = np.ones(length, dtype=np.float32)
    a = int(length * attack)
    r = int(length * release)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
    if r > 0:
        env[-r:] = np.minimum(env[-r:], np.linspace(1.0, 0.0, r, endpoint=True))
    return env


def create_soundtrack(path: Path, duration_s: float):
    sample_rate = int(CONFIG["audio_rate"])
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    rng = np.random.default_rng(1969)

    # restrained documentary bed
    audio = (
        0.15 * np.sin(2 * math.pi * 48.0 * t)
        + 0.08 * np.sin(2 * math.pi * 72.0 * t + 0.4)
        + 0.035 * np.sin(2 * math.pi * 144.0 * t + 0.9)
        + 0.014 * rng.normal(0.0, 1.0, n).astype(np.float32)
    )

    # transition impacts and whooshes
    for shot_name, start, _ in SHOT_PLAN[1:]:
        i0 = int(start * sample_rate)
        length = min(int(0.70 * sample_rate), n - i0)
        if length <= 0:
            continue
        tt = np.arange(length, dtype=np.float32) / sample_rate
        hit = 0.15 * np.sin(2 * math.pi * 92.0 * tt) * np.exp(-tt * 5.2)
        whoosh = 0.025 * rng.normal(0.0, 1.0, length).astype(np.float32) * np.exp(-tt * 3.0)
        if shot_name == "fresh_ejecta":
            hit += 0.12 * np.sin(2 * math.pi * 45.0 * tt) * np.exp(-tt * 2.7)
        audio[i0:i0 + length] += hit + whoosh

    # crystalline accents on bright crater reveals
    for start in [SHOT_PLAN[2][1], SHOT_PLAN[3][1], SHOT_PLAN[4][1]]:
        i0 = int(start * sample_rate)
        length = min(int(1.3 * sample_rate), n - i0)
        tt = np.arange(length, dtype=np.float32) / sample_rate
        chime = (
            0.055 * np.sin(2 * math.pi * 523.25 * tt)
            + 0.040 * np.sin(2 * math.pi * 659.25 * tt)
            + 0.026 * np.sin(2 * math.pi * 783.99 * tt)
        ) * np.exp(-tt * 1.6)
        audio[i0:i0 + length] += chime

    # finale shimmer
    i0 = int(max(0.0, duration_s - (1.4 if QUICK_MODE else 5.5)) * sample_rate)
    tt = np.arange(n - i0, dtype=np.float32) / sample_rate
    audio[i0:] += 0.045 * np.sin(2 * math.pi * 440.0 * tt) * np.exp(-tt * 0.42)

    audio /= max(float(np.max(np.abs(audio))), 1e-6)
    audio *= 0.72
    left = audio * (0.97 + 0.03 * np.sin(2 * math.pi * 0.035 * t))
    right = audio * (0.97 + 0.03 * np.cos(2 * math.pi * 0.035 * t + 0.8))
    stereo = np.stack([left, right], axis=1)
    pcm = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def render_video(scene: LunarCraterScene, output_path: Path):
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
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(final_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        shutil.copy2(video_path, final_path)
        return False


def save_contact_sheet(scene: LunarCraterScene, path: Path):
    samples = [
        SHOT_PLAN[0][1] + (SHOT_PLAN[0][2] - SHOT_PLAN[0][1]) * 0.55,
        SHOT_PLAN[2][1] + (SHOT_PLAN[2][2] - SHOT_PLAN[2][1]) * 0.55,
        SHOT_PLAN[3][1] + (SHOT_PLAN[3][2] - SHOT_PLAN[3][1]) * 0.55,
        SHOT_PLAN[4][1] + (SHOT_PLAN[4][2] - SHOT_PLAN[4][1]) * 0.62,
        SHOT_PLAN[5][1] + (SHOT_PLAN[5][2] - SHOT_PLAN[5][1]) * 0.60,
        SHOT_PLAN[6][1] + (SHOT_PLAN[6][2] - SHOT_PLAN[6][1]) * 0.62,
        SHOT_PLAN[7][1] + (SHOT_PLAN[7][2] - SHOT_PLAN[7][1]) * 0.58,
        SHOT_PLAN[8][1] + (SHOT_PLAN[8][2] - SHOT_PLAN[8][1]) * 0.60,
    ]
    thumbs: List[Image.Image] = []
    for sample_t in samples:
        frame = Image.fromarray(scene.frame(sample_t))
        thumbs.append(frame.resize((270, 480), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (270 * 4, 480 * 2), (5, 7, 12))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 4) * 270, (index // 4) * 480))
    sheet.save(path, quality=92)


def write_summary(path: Path, audio_muxed: bool):
    payload = {
        "title": CONFIG["title"],
        "subtitle": CONFIG["subtitle"],
        "featured_craters": CRATERS,
        "scientific_notes": [
            "The featured craters reflect sunlight; they are not self-luminous.",
            "Aristarchus is presented as one of the Moon's brightest features, not as a calibrated universal albedo ranking against every pixel on the Moon.",
            "Bright ray systems are associated with comparatively fresh ejecta and exposed immature material.",
            "Space weathering by solar-wind irradiation and micrometeoroid impacts changes lunar regolith optical properties over time.",
            "Composition, roughness, illumination angle, viewing angle, and camera processing can also alter apparent brightness.",
            "Procedural visuals are illustrative and are not calibrated LRO reflectance maps.",
        ],
        "official_sources": [
            "https://science.nasa.gov/resource/aristarchus-crater-2/",
            "https://science.nasa.gov/resource/tycho-crater-on-the-moon-labeled/",
            "https://svs.gsfc.nasa.gov/4220/",
            "https://apod.nasa.gov/apod/ap010809.html",
            "https://science.nasa.gov/moon/moonlight/",
            "https://science.nasa.gov/moon/lunar-craters/",
        ],
        "audio_muxed": audio_muxed,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

from __future__ import annotations

"""
The Moon's Brightest Craters — And Why They Outshine the Surface

A vertical, cinematic YouTube Shorts renderer with minimal text, procedural
lunar visuals, a procedural stereo soundtrack, and a small real-metadata table
for prominent high-albedo / rayed lunar craters.

Scientific framing
------------------
These craters are not self-luminous. They look bright because they reflect more
sunlight than much of the surrounding lunar surface. Fresh impacts expose and
spread comparatively immature material. Over time, solar-wind irradiation and
micrometeoroid bombardment alter and generally darken/redden exposed regolith —
a family of processes called space weathering. Composition and illumination /
viewing geometry also affect apparent brightness.

Featured examples
-----------------
- Aristarchus: one of the Moon's brightest features; ~40 km diameter
- Tycho: ~85 km diameter; ~110 million years old; enormous bright ray system
- Copernicus: ~93 km diameter; prominent bright rays
- Kepler: ~32 km diameter; bright rays over dark Oceanus Procellarum
- Proclus: ~28 km diameter; conspicuous asymmetric bright rays

The code avoids presenting a made-up global albedo ranking. Aristarchus is the
headline feature; the others are shown as prominent bright/rayed examples.

Official source pages used for framing and metadata are written to the output
summary and README. The lunar surface imagery in this renderer is procedural,
not a replacement for calibrated LRO reflectance products.

Usage
-----
Standard vertical render:
    python the_moons_brightest_craters_cinematic_short.py

Fast validation preview:
    MOON_CRATERS_SHORT_QUICK=1 python the_moons_brightest_craters_cinematic_short.py

4K vertical:
    MOON_CRATERS_SHORT_4K=1 python the_moons_brightest_craters_cinematic_short.py
"""

import csv
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

QUICK_MODE = os.environ.get("MOON_CRATERS_SHORT_QUICK", "0") == "1"
FOUR_K_MODE = os.environ.get("MOON_CRATERS_SHORT_4K", "0") == "1" and not QUICK_MODE

OUTPUT_ROOT = Path("the_moons_brightest_craters_output")
DATA_DIR = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_DIR, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else (2160 if FOUR_K_MODE else 1080),
    "video_height": 960 if QUICK_MODE else (3840 if FOUR_K_MODE else 1920),
    "fps": 8 if QUICK_MODE else 24,
    "duration_s": 13 if QUICK_MODE else 52,
    "audio_rate": 44100,
    "title": "THE MOON'S BRIGHTEST CRATERS",
    "subtitle": "Why fresh impacts outshine an ancient surface",
    "output_basename": "the_moons_brightest_craters",
    "stars": 160 if QUICK_MODE else 650,
    "contrast": 1.10,
    "saturation": 0.92,
    "vignette": 0.28,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
FPS = int(CONFIG["fps"])
DURATION = float(CONFIG["duration_s"])
FRAME_COUNT = int(round(FPS * DURATION))

COLORS = {
    "white": (245, 247, 244),
    "muted": (177, 188, 197),
    "ice": (171, 222, 255),
    "gold": (245, 191, 94),
    "cyan": (88, 218, 244),
    "violet": (164, 120, 230),
    "dark": (3, 5, 12),
    "moon_dark": (45, 47, 51),
    "moon_mid": (119, 121, 119),
    "moon_light": (205, 208, 201),
}

# Coordinates use positive east longitude. West longitudes are negative.
# Ages are approximate and intentionally left blank where this short does not
# need a robust single-value age claim.
CRATERS: List[Dict[str, Any]] = [
    {
        "name": "Aristarchus",
        "lat": 23.7,
        "lon": -47.4,
        "diameter_km": 40.0,
        "age_myr": None,
        "role": "headline",
        "note": "One of the brightest features on the Moon; high-reflectance rays and exposed materials.",
        "source": "https://science.nasa.gov/resource/aristarchus-crater-2/",
    },
    {
        "name": "Tycho",
        "lat": -43.3,
        "lon": -11.4,
        "diameter_km": 85.0,
        "age_myr": 110.0,
        "role": "major_ray_crater",
        "note": "A young, prominent crater with bright rays extending across much of the nearside.",
        "source": "https://science.nasa.gov/resource/tycho-crater-on-the-moon-labeled/",
    },
    {
        "name": "Copernicus",
        "lat": 9.7,
        "lon": -20.1,
        "diameter_km": 93.0,
        "age_myr": 800.0,
        "role": "major_ray_crater",
        "note": "A large nearside complex crater with an extensive light-colored ejecta-ray system.",
        "source": "https://apod.nasa.gov/apod/ap010809.html",
    },
    {
        "name": "Kepler",
        "lat": 8.1,
        "lon": -38.0,
        "diameter_km": 32.0,
        "age_myr": None,
        "role": "ray_crater",
        "note": "Bright rays cross the darker basaltic terrain of Oceanus Procellarum.",
        "source": "https://science.nasa.gov/image-article/apod-2023-december-7-orion-and-the-ocean-of-storms/",
    },
    {
        "name": "Proclus",
        "lat": 16.1,
        "lon": 46.8,
        "diameter_km": 28.0,
        "age_myr": None,
        "role": "ray_crater",
        "note": "A conspicuous bright crater with an asymmetric ray pattern near Mare Crisium.",
        "source": "https://science.nasa.gov/moon/lunar-craters/",
    },
]

FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.0, 3.5, "Some lunar craters look almost white against the Moon's ancient surface."),
    (4.0, 7.5, "But they are not glowing. They are reflecting more sunlight."),
    (9.0, 12.8, "Aristarchus is one of the brightest features on the entire Moon."),
    (15.0, 18.8, "Tycho throws brilliant ejecta rays across much of the lunar nearside."),
    (21.0, 24.8, "Copernicus, Kepler, and Proclus reveal the same impact signature."),
    (27.0, 30.8, "The impact excavates fresh rock and sprays immature material over older terrain."),
    (33.0, 36.8, "Solar wind and micrometeoroids slowly weather exposed soil, making old surfaces darker."),
    (39.0, 42.8, "Composition and Sun angle matter too — brightness is not only about crater age."),
    (45.0, 49.0, "Bright rays are temporary geological fingerprints of comparatively recent impacts."),
]

SHOT_PLAN_FULL: List[Tuple[str, float, float]] = [
    ("intro", 0.0, 4.0),
    ("not_glowing", 4.0, 8.0),
    ("aristarchus", 8.0, 14.0),
    ("tycho", 14.0, 20.0),
    ("crater_gallery", 20.0, 26.0),
    ("fresh_ejecta", 26.0, 32.0),
    ("weathering", 32.0, 38.0),
    ("geometry", 38.0, 44.0),
    ("finale", 44.0, 52.0),
]

if QUICK_MODE:
    scale = DURATION / 52.0
    CAPTIONS = [(a * scale, b * scale, text) for a, b, text in FULL_CAPTIONS]
    SHOT_PLAN = [(name, a * scale, b * scale) for name, a, b in SHOT_PLAN_FULL]
else:
    CAPTIONS = FULL_CAPTIONS
    SHOT_PLAN = SHOT_PLAN_FULL


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
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)
        if bbox[2] - bbox[0] <= x1 - x0:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    y = y0
    for line in lines:
        draw.text((x0, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 220))
        bbox = draw.textbbox((x0, y), line, font=font, stroke_width=2)
        y += (bbox[3] - bbox[1]) + max(4, int(size * 0.20))
        if y > y1:
            break


def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 140):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius = max(8, int((box[3] - box[1]) * 0.18))
    draw.rounded_rectangle(box, radius=radius, fill=(2, 5, 13, alpha), outline=COLORS["ice"] + (42,), width=1)
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
        lines.extend([str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
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
# Data products
# -----------------------------------------------------------------------------


def save_crater_csv(path: Path):
    fields = ["name", "lat", "lon", "diameter_km", "age_myr", "role", "note", "source"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for crater in CRATERS:
            writer.writerow(crater)


# -----------------------------------------------------------------------------
# Renderer
# -----------------------------------------------------------------------------


class LunarCraterScene:
    def __init__(self):
        self.stars = self._make_stars(int(CONFIG["stars"]), seed=2077)
        self.moon_texture = self._make_moon_texture(seed=1969)
        self.closeup_textures = {
            crater["name"]: self._make_crater_closeup(crater, seed=1000 + index * 71)
            for index, crater in enumerate(CRATERS)
        }

    @staticmethod
    def _make_stars(count: int, seed: int) -> List[Star]:
        rng = np.random.default_rng(seed)
        stars: List[Star] = []
        for _ in range(count):
            stars.append(
                Star(
                    x=float(rng.uniform(0, OUT_W)),
                    y=float(rng.uniform(0, OUT_H)),
                    radius=float(rng.uniform(0.25, 2.0) * OUT_W / 1080),
                    alpha=float(rng.uniform(18, 135)),
                    phase=float(rng.uniform(0, 2 * math.pi)),
                )
            )
        return stars

    @staticmethod
    def _make_moon_texture(seed: int, size: int = 1152) -> Image.Image:
        rng = np.random.default_rng(seed)
        base = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
        image = Image.fromarray(np.uint8(np.clip((base - base.min()) / max(float(np.ptp(base)), 1e-6) * 255, 0, 255)))
        multi = np.zeros((size, size), dtype=np.float32)
        for blur, weight in [(46, 0.45), (22, 0.35), (9, 0.20), (3, 0.08)]:
            blurred = image.filter(ImageFilter.GaussianBlur(blur))
            multi += np.asarray(blurred, dtype=np.float32) / 255.0 * weight
        multi -= multi.min()
        multi /= max(float(multi.max()), 1e-6)

        yy, xx = np.mgrid[0:size, 0:size]
        # Procedural dark maria. These are artistic placements, not a calibrated map.
        maria = np.ones((size, size), dtype=np.float32)
        for cx, cy, rx, ry, depth in [
            (0.64, 0.37, 0.20, 0.14, 0.38),
            (0.43, 0.38, 0.17, 0.13, 0.31),
            (0.60, 0.57, 0.18, 0.12, 0.28),
            (0.32, 0.54, 0.12, 0.10, 0.24),
            (0.74, 0.49, 0.11, 0.09, 0.22),
        ]:
            ellipse = ((xx / size - cx) / rx) ** 2 + ((yy / size - cy) / ry) ** 2
            maria -= np.exp(-ellipse * 2.2) * depth
        texture = np.clip(0.40 + multi * 0.50, 0.0, 1.0) * np.clip(maria, 0.46, 1.0)

        # Fine craterlets.
        for _ in range(620 if not QUICK_MODE else 230):
            cx = int(rng.uniform(0, size))
            cy = int(rng.uniform(0, size))
            radius = int(rng.uniform(1.0, 8.0))
            y0 = max(0, cy - radius * 2)
            y1 = min(size, cy + radius * 2 + 1)
            x0 = max(0, cx - radius * 2)
            x1 = min(size, cx + radius * 2 + 1)
            sy, sx = np.mgrid[y0:y1, x0:x1]
            dist = np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2) / max(radius, 1)
            depression = -0.16 * np.exp(-((dist - 0.75) / 0.45) ** 2)
            rim = 0.10 * np.exp(-((dist - 1.05) / 0.25) ** 2)
            texture[y0:y1, x0:x1] += depression + rim

        texture = np.clip(texture, 0.0, 1.0)
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(texture * 224, 0, 255)
        rgb[..., 1] = np.clip(texture * 226, 0, 255)
        rgb[..., 2] = np.clip(texture * 219, 0, 255)
        return Image.fromarray(rgb, mode="RGB")

    @staticmethod
    def _make_crater_closeup(crater: Dict[str, Any], seed: int, size: int = 1000) -> Image.Image:
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:size, 0:size]
        cx, cy = size * 0.5, size * 0.51
        radius = size * (0.18 if crater["name"] != "Tycho" else 0.20)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        angle = np.arctan2(yy - cy, xx - cx)

        noise = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
        noise_img = Image.fromarray(np.uint8(np.clip((noise - noise.min()) / max(float(np.ptp(noise)), 1e-6) * 255, 0, 255)))
        terrain = np.asarray(noise_img.filter(ImageFilter.GaussianBlur(10)), dtype=np.float32) / 255.0
        terrain = 0.35 + 0.45 * terrain

        # Circular crater depression, bright rim, terraces, and central peak.
        terrain -= 0.34 * np.exp(-((dist / radius) / 0.72) ** 4)
        terrain += 0.35 * np.exp(-((dist - radius) / (radius * 0.10)) ** 2)
        terrain += 0.10 * np.cos(dist / max(radius * 0.10, 1.0) * math.pi) * np.exp(-((dist / radius - 0.70) / 0.28) ** 2)
        terrain += 0.22 * np.exp(-((dist / (radius * 0.18)) ** 2))

        # Rays. Aristarchus and Proclus receive more asymmetric systems.
        ray_count = {"Aristarchus": 13, "Tycho": 18, "Copernicus": 14, "Kepler": 11, "Proclus": 9}.get(crater["name"], 10)
        asym = 0.46 if crater["name"] == "Proclus" else 0.0
        for index in range(ray_count):
            theta = 2 * math.pi * index / ray_count + rng.uniform(-0.10, 0.10)
            if asym and math.cos(theta) < -0.15:
                continue
            angular = np.angle(np.exp(1j * (angle - theta)))
            width = rng.uniform(0.018, 0.055)
            radial_gate = 1.0 / (1.0 + np.exp(-(dist - radius * 0.82) / max(radius * 0.05, 1.0)))
            falloff = np.exp(-dist / (radius * rng.uniform(2.0, 3.6)))
            terrain += np.exp(-(angular / width) ** 2) * falloff * radial_gate * rng.uniform(0.14, 0.30)

        # Dark impact-melt-like patches.
        terrain -= 0.09 * np.exp(-((dist / (radius * 0.50)) ** 2)) * (0.5 + 0.5 * np.sin(angle * 5.0 + 0.7))
        terrain = np.clip(terrain, 0.0, 1.0)

        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        tint = 1.03 if crater["name"] == "Aristarchus" else 1.0
        rgb[..., 0] = np.clip(terrain * 235 * tint, 0, 255)
        rgb[..., 1] = np.clip(terrain * 238 * tint, 0, 255)
        rgb[..., 2] = np.clip(terrain * 232 * tint, 0, 255)
        return Image.fromarray(rgb, mode="RGB")

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", (OUT_W, OUT_H), COLORS["dark"] + (255,))
        draw = ImageDraw.Draw(image)
        for y in range(OUT_H):
            p = y / max(OUT_H - 1, 1)
            draw.line((0, y, OUT_W, y), fill=(int(lerp(2, 8, p)), int(lerp(4, 9, p)), int(lerp(12, 20, p)), 255))
        for star in self.stars:
            alpha = int(star.alpha * (0.72 + 0.28 * math.sin(t * 1.25 + star.phase)))
            r = star.radius
            draw.ellipse((star.x - r, star.y - r, star.x + r, star.y + r), fill=COLORS["white"] + (alpha,))
        return image

    @staticmethod
    def lonlat_to_disc(lon: float, lat: float, center: Tuple[int, int], radius: float) -> Optional[Tuple[float, float]]:
        lon_r = math.radians(lon)
        lat_r = math.radians(lat)
        # Orthographic view centered at 0°N, 0°E.
        visible = math.cos(lat_r) * math.cos(lon_r)
        if visible <= 0:
            return None
        x = center[0] + radius * math.cos(lat_r) * math.sin(lon_r)
        y = center[1] - radius * math.sin(lat_r)
        return x, y

    def draw_moon(
        self,
        image: Image.Image,
        center: Tuple[int, int],
        radius: int,
        phase: float = 0.92,
        rotation_deg: float = 0.0,
        show_craters: bool = True,
        ray_strength: float = 1.0,
    ):
        size = radius * 2
        texture = self.moon_texture.resize((size, size), Image.Resampling.LANCZOS)
        if rotation_deg:
            texture = texture.rotate(rotation_deg, resample=Image.Resampling.BICUBIC)
        arr = np.asarray(texture, dtype=np.float32) / 255.0
        yy, xx = np.mgrid[0:size, 0:size]
        nx = (xx - radius + 0.5) / radius
        ny = (yy - radius + 0.5) / radius
        r2 = nx * nx + ny * ny
        mask = r2 <= 1.0
        z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

        # phase=1 means near-full. Light source shifts slightly for cinematic relief.
        sx = lerp(-0.46, -0.08, phase)
        sz = math.sqrt(max(0.0, 1.0 - sx * sx))
        illumination = np.clip(nx * sx + z * sz, 0.0, 1.0)
        limb = np.clip(0.36 + 0.78 * z, 0.0, 1.0)
        shade = illumination * limb
        rgb = np.zeros((size, size, 4), dtype=np.uint8)
        rgb[..., :3] = np.clip(arr * shade[..., None] * 255.0, 0, 255).astype(np.uint8)
        rgb[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
        moon = Image.fromarray(rgb, mode="RGBA")

        # Draw highlighted crater/ray systems in disc coordinates.
        if show_craters:
            overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            local_center = (radius, radius)
            for crater in CRATERS:
                pos = self.lonlat_to_disc(crater["lon"], crater["lat"], local_center, radius * 0.94)
                if pos is None:
                    continue
                x, y = pos
                base = max(3.0, crater["diameter_km"] / 12.5 * radius / 280.0)
                emphasis = 1.25 if crater["name"] == "Aristarchus" else 1.0
                d = base * emphasis
                # bright core/rim
                od.ellipse((x - d, y - d, x + d, y + d), fill=COLORS["white"] + (235,), outline=COLORS["ice"] + (220,), width=max(1, int(radius * 0.006)))
                # ray systems
                ray_count = {"Aristarchus": 12, "Tycho": 18, "Copernicus": 13, "Kepler": 9, "Proclus": 7}[crater["name"]]
                ray_len = radius * ({"Aristarchus": 0.20, "Tycho": 0.63, "Copernicus": 0.34, "Kepler": 0.21, "Proclus": 0.22}[crater["name"]])
                for index in range(ray_count):
                    angle = 2 * math.pi * index / ray_count + (0.22 if crater["name"] == "Proclus" else 0.0)
                    if crater["name"] == "Proclus" and math.cos(angle) < -0.3:
                        continue
                    end_x = x + math.cos(angle) * ray_len * (0.72 + 0.28 * math.sin(index * 1.8 + 0.5))
                    end_y = y + math.sin(angle) * ray_len * (0.72 + 0.28 * math.cos(index * 1.5))
                    od.line((x, y, end_x, end_y), fill=COLORS["white"] + (int(72 * ray_strength),), width=max(1, int(radius * 0.004)))
            glow = overlay.filter(ImageFilter.GaussianBlur(max(2, int(radius * 0.025))))
            moon.alpha_composite(glow)
            moon.alpha_composite(overlay)

        # limb glow
        glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse((3, 3, size - 3, size - 3), outline=COLORS["ice"] + (78,), width=max(1, int(radius * 0.018)))
        moon.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(3, int(radius * 0.035)))))
        image.alpha_composite(moon, (center[0] - radius, center[1] - radius))

    def draw_crater_closeup(
        self,
        image: Image.Image,
        crater_name: str,
        box: Tuple[int, int, int, int],
        zoom: float = 1.0,
        pan: Tuple[float, float] = (0.0, 0.0),
        brightness: float = 1.0,
    ):
        x0, y0, x1, y1 = box
        source = self.closeup_textures[crater_name]
        crop_w = max(100, int(source.width / zoom))
        crop_h = max(100, int(source.height / zoom * ((y1 - y0) / max(x1 - x0, 1))))
        cx = int(source.width * (0.5 + pan[0]))
        cy = int(source.height * (0.5 + pan[1]))
        cx = int(np.clip(cx, crop_w // 2, source.width - crop_w // 2))
        cy = int(np.clip(cy, crop_h // 2, source.height - crop_h // 2))
        crop = source.crop((cx - crop_w // 2, cy - crop_h // 2, cx + crop_w // 2, cy + crop_h // 2))
        crop = crop.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
        crop = ImageEnhance.Brightness(crop).enhance(brightness).convert("RGBA")

        mask = Image.new("L", crop.size, 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle((0, 0, crop.width - 1, crop.height - 1), radius=int(min(crop.size) * 0.05), fill=255)
        crop.putalpha(mask)
        image.alpha_composite(crop, (x0, y0))

        frame = Image.new("RGBA", image.size, (0, 0, 0, 0))
        fd = ImageDraw.Draw(frame)
        fd.rounded_rectangle(box, radius=int(min(x1 - x0, y1 - y0) * 0.05), outline=COLORS["ice"] + (72,), width=max(1, int(OUT_W / 700)))
        image.alpha_composite(frame)

    def draw_crater_label(self, image: Image.Image, crater: Dict[str, Any], y: int, descriptor: str):
        draw_text(image, crater["name"].upper(), (OUT_W // 2, y), size=max(18, int(58 * OUT_W / 1080)), fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=2)
        draw_text(image, descriptor, (OUT_W // 2, y + int(OUT_H * 0.036)), size=max(10, int(25 * OUT_W / 1080)), fill=COLORS["gold"] + (235,), anchor="ma", stroke=1)

    def draw_impact_sequence(self, image: Image.Image, local: float):
        horizon = int(OUT_H * 0.66)
        # lunar ground
        draw = ImageDraw.Draw(image)
        for y in range(horizon, OUT_H):
            p = (y - horizon) / max(OUT_H - horizon, 1)
            value = int(lerp(54, 20, p))
            draw.line((0, y, OUT_W, y), fill=(value, value, value + 2, 255))

        impact_x = int(OUT_W * 0.50)
        impact_y = horizon + int(OUT_H * 0.055)
        flash = smoothstep(min(local * 2.0, 1.0)) * (1.0 - smoothstep(max((local - 0.40) / 0.60, 0.0)))
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        for radius, alpha in [
            (int(OUT_W * 0.20), int(35 * flash)),
            (int(OUT_W * 0.11), int(80 * flash)),
            (int(OUT_W * 0.045), int(180 * flash)),
        ]:
            ld.ellipse((impact_x - radius, impact_y - radius, impact_x + radius, impact_y + radius), fill=COLORS["white"] + (alpha,))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(4, int(20 * OUT_W / 1080)))))

        # ejecta rays / ballistic arcs
        ejecta = Image.new("RGBA", image.size, (0, 0, 0, 0))
        ed = ImageDraw.Draw(ejecta)
        ray_reveal = smoothstep((local - 0.18) / 0.72)
        for i in range(22):
            angle = math.radians(200 + i * (140 / 21))
            length = OUT_W * (0.18 + 0.34 * abs(math.sin(i * 0.73))) * ray_reveal
            ex = impact_x + math.cos(angle) * length
            ey = impact_y + math.sin(angle) * length * 0.68
            ed.line((impact_x, impact_y, ex, ey), fill=COLORS["white"] + (90,), width=max(1, int(OUT_W / 650)))
            d = max(1, int(3 * OUT_W / 1080))
            ed.ellipse((ex - d, ey - d, ex + d, ey + d), fill=COLORS["gold"] + (170,))
        image.alpha_composite(ejecta.filter(ImageFilter.GaussianBlur(max(1, int(2 * OUT_W / 1080)))))

        # final crater ring
        crater_r = int(OUT_W * 0.095 * ray_reveal)
        draw.ellipse((impact_x - crater_r, impact_y - crater_r * 0.30, impact_x + crater_r, impact_y + crater_r * 0.30), fill=(25, 25, 27, 255), outline=COLORS["white"] + (160,), width=max(1, int(4 * OUT_W / 1080)))

    def draw_weathering_comparison(self, image: Image.Image, local: float):
        y0, y1 = int(OUT_H * 0.26), int(OUT_H * 0.72)
        left_box = (int(OUT_W * 0.06), y0, int(OUT_W * 0.48), y1)
        right_box = (int(OUT_W * 0.52), y0, int(OUT_W * 0.94), y1)
        self.draw_crater_closeup(image, "Tycho", left_box, zoom=1.45, brightness=1.15)
        self.draw_crater_closeup(image, "Tycho", right_box, zoom=1.45, brightness=lerp(1.10, 0.58, local))

        # add reddening/darkening to weathered side
        tint = Image.new("RGBA", image.size, (0, 0, 0, 0))
        td = ImageDraw.Draw(tint)
        td.rounded_rectangle(right_box, radius=int((right_box[2] - right_box[0]) * 0.05), fill=(70, 40, 26, int(70 * local)))
        image.alpha_composite(tint)

        draw_text(image, "FRESH / IMMATURE", ((left_box[0] + left_box[2]) // 2, int(OUT_H * 0.75)), size=max(10, int(23 * OUT_W / 1080)), fill=COLORS["ice"] + (235,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "SPACE-WEATHERED", ((right_box[0] + right_box[2]) // 2, int(OUT_H * 0.75)), size=max(10, int(23 * OUT_W / 1080)), fill=COLORS["gold"] + (235,), bold=True, anchor="ma", stroke=1)

        # particle streaks above the weathered side
        particle = Image.new("RGBA", image.size, (0, 0, 0, 0))
        pd = ImageDraw.Draw(particle)
        rng = np.random.default_rng(500)
        for i in range(36 if not QUICK_MODE else 18):
            x = rng.uniform(right_box[0], right_box[2])
            y = rng.uniform(y0 - OUT_H * 0.16, y0 + OUT_H * 0.08)
            length = rng.uniform(14, 55) * OUT_W / 1080
            pd.line((x, y, x - length * 0.55, y + length), fill=COLORS["gold"] + (int(40 + 80 * local),), width=1)
        image.alpha_composite(particle.filter(ImageFilter.GaussianBlur(1)))

    def draw_phase_geometry(self, image: Image.Image, local: float):
        phases = [0.58, 0.76, 0.96]
        centers = [(int(OUT_W * 0.20), int(OUT_H * 0.45)), (int(OUT_W * 0.50), int(OUT_H * 0.45)), (int(OUT_W * 0.80), int(OUT_H * 0.45))]
        radius = int(OUT_W * 0.14)
        for index, (phase, center) in enumerate(zip(phases, centers)):
            reveal = clamp(local * 1.4 - index * 0.20)
            if reveal <= 0:
                continue
            self.draw_moon(image, center, radius, phase=phase, rotation_deg=index * 2.0, show_craters=True, ray_strength=reveal)
        draw_text(image, "SUN ANGLE CHANGES APPARENT CONTRAST", (OUT_W // 2, int(OUT_H * 0.67)), size=max(12, int(31 * OUT_W / 1080)), fill=COLORS["white"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "rays often stand out most near the full Moon", (OUT_W // 2, int(OUT_H * 0.715)), size=max(10, int(23 * OUT_W / 1080)), fill=COLORS["gold"] + (225,), anchor="ma", stroke=1)

    def draw_caption(self, image: Image.Image, caption: str, t: float):
        start, end = 0.0, 0.0
        for a, b, text in CAPTIONS:
            if text == caption and a <= t < b:
                start, end = a, b
                break
        fade_in = clamp((t - start) / max(0.08 if QUICK_MODE else 0.34, 1e-6))
        fade_out = clamp((end - t) / max(0.10 if QUICK_MODE else 0.44, 1e-6))
        alpha = int(225 * min(fade_in, fade_out, 1.0))
        if alpha <= 0:
            return
        box = (int(OUT_W * 0.07), int(OUT_H * 0.76), int(OUT_W * 0.93), int(OUT_H * 0.86))
        panel(image, box, alpha=min(105, alpha // 2))
        draw_wrapped_text(
            image,
            caption,
            (box[0] + int(OUT_W * 0.025), box[1] + int(OUT_H * 0.012), box[2] - int(OUT_W * 0.025), box[3] - int(OUT_H * 0.010)),
            size=max(12, int(30 * OUT_W / 1080)),
            fill=COLORS["white"] + (alpha,),
        )

    def frame(self, t: float) -> np.ndarray:
        shot, start, end = get_shot(t)
        local = smoothstep((t - start) / max(end - start, 1e-9))
        image = self.background(t)

        if shot == "intro":
            self.draw_moon(image, (OUT_W // 2, int(OUT_H * 0.43)), int(OUT_W * lerp(0.27, 0.31, local)), phase=0.94, rotation_deg=-2 + 4 * local, show_craters=True, ray_strength=0.85)
            draw_text(image, "THE MOON'S BRIGHTEST", (OUT_W // 2, int(OUT_H * 0.105)), size=max(18, int(67 * OUT_W / 1080)), fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=2)
            draw_text(image, "CRATERS", (OUT_W // 2, int(OUT_H * 0.155)), size=max(20, int(82 * OUT_W / 1080)), fill=COLORS["gold"] + (245,), bold=True, anchor="ma", stroke=2)

        elif shot == "not_glowing":
            phase = lerp(0.60, 0.97, local)
            self.draw_moon(image, (OUT_W // 2, int(OUT_H * 0.43)), int(OUT_W * 0.31), phase=phase, rotation_deg=2.0, show_craters=True, ray_strength=local)
            draw_text(image, "REFLECTED SUNLIGHT", (OUT_W // 2, int(OUT_H * 0.17)), size=max(14, int(40 * OUT_W / 1080)), fill=COLORS["ice"] + (240,), bold=True, anchor="ma", stroke=1)
            draw_text(image, "not lunar luminescence", (OUT_W // 2, int(OUT_H * 0.21)), size=max(10, int(24 * OUT_W / 1080)), fill=COLORS["muted"] + (220,), anchor="ma", stroke=1)

        elif shot == "aristarchus":
            self.draw_crater_closeup(image, "Aristarchus", (int(OUT_W * 0.05), int(OUT_H * 0.18), int(OUT_W * 0.95), int(OUT_H * 0.72)), zoom=lerp(1.25, 1.72, local), pan=(lerp(-0.03, 0.02, local), 0.0), brightness=1.10)
            crater = CRATERS[0]
            self.draw_crater_label(image, crater, int(OUT_H * 0.105), "~40 km • one of the Moon's brightest features")

        elif shot == "tycho":
            self.draw_crater_closeup(image, "Tycho", (int(OUT_W * 0.05), int(OUT_H * 0.18), int(OUT_W * 0.95), int(OUT_H * 0.72)), zoom=lerp(1.08, 1.50, local), pan=(0.0, lerp(-0.02, 0.02, local)), brightness=1.04)
            crater = CRATERS[1]
            self.draw_crater_label(image, crater, int(OUT_H * 0.105), "~85 km • ~110 million years • vast ray system")

        elif shot == "crater_gallery":
            names = ["Copernicus", "Kepler", "Proclus"]
            boxes = [
                (int(OUT_W * 0.06), int(OUT_H * 0.22), int(OUT_W * 0.48), int(OUT_H * 0.51)),
                (int(OUT_W * 0.52), int(OUT_H * 0.22), int(OUT_W * 0.94), int(OUT_H * 0.51)),
                (int(OUT_W * 0.29), int(OUT_H * 0.54), int(OUT_W * 0.71), int(OUT_H * 0.80)),
            ]
            for index, (name, box) in enumerate(zip(names, boxes)):
                reveal = clamp(local * 1.6 - index * 0.22)
                if reveal <= 0:
                    continue
                self.draw_crater_closeup(image, name, box, zoom=1.45, brightness=0.96 + 0.08 * reveal)
                draw_text(image, name.upper(), ((box[0] + box[2]) // 2, box[3] - int(OUT_H * 0.025)), size=max(10, int(24 * OUT_W / 1080)), fill=COLORS["white"] + (235,), bold=True, anchor="mm", stroke=1)
            draw_text(image, "PROMINENT HIGH-ALBEDO RAY CRATERS", (OUT_W // 2, int(OUT_H * 0.135)), size=max(12, int(34 * OUT_W / 1080)), fill=COLORS["gold"] + (240,), bold=True, anchor="ma", stroke=1)

        elif shot == "fresh_ejecta":
            self.draw_impact_sequence(image, local)
            draw_text(image, "FRESH EJECTA", (OUT_W // 2, int(OUT_H * 0.17)), size=max(15, int(46 * OUT_W / 1080)), fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=1)
            draw_text(image, "newly exposed material reflects strongly", (OUT_W // 2, int(OUT_H * 0.215)), size=max(10, int(23 * OUT_W / 1080)), fill=COLORS["gold"] + (225,), anchor="ma", stroke=1)

        elif shot == "weathering":
            self.draw_weathering_comparison(image, local)
            draw_text(image, "SPACE WEATHERING", (OUT_W // 2, int(OUT_H * 0.15)), size=max(15, int(44 * OUT_W / 1080)), fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=1)
            draw_text(image, "solar wind + micrometeoroids", (OUT_W // 2, int(OUT_H * 0.195)), size=max(10, int(23 * OUT_W / 1080)), fill=COLORS["gold"] + (225,), anchor="ma", stroke=1)

        elif shot == "geometry":
            self.draw_phase_geometry(image, local)

        elif shot == "finale":
            self.draw_moon(image, (OUT_W // 2, int(OUT_H * 0.42)), int(OUT_W * 0.32), phase=0.98, rotation_deg=-1.0, show_craters=True, ray_strength=1.0)
            draw_text(image, "BRIGHTNESS IS A GEOLOGICAL CLOCK", (OUT_W // 2, int(OUT_H * 0.13)), size=max(13, int(39 * OUT_W / 1080)), fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=1)
            draw_text(image, "freshness • composition • illumination", (OUT_W // 2, int(OUT_H * 0.18)), size=max(10, int(24 * OUT_W / 1080)), fill=COLORS["gold"] + (230,), anchor="ma", stroke=1)
            draw_text(image, "ARISTARCHUS • TYCHO • COPERNICUS • KEPLER • PROCLUS", (OUT_W // 2, int(OUT_H * 0.71)), size=max(8, int(18 * OUT_W / 1080)), fill=COLORS["ice"] + (205,), anchor="ma", stroke=1)

        caption = caption_at(t)
        if caption:
            self.draw_caption(image, caption, t)

        draw_text(image, "NASA / LRO science framing • procedural visualization", (int(OUT_W * 0.025), int(OUT_H * 0.988)), size=max(7, int(12 * OUT_W / 1080)), fill=COLORS["muted"] + (110,), anchor="ls", stroke=1)

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        arr = np.clip(arr * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        graded = Image.fromarray(arr)
        graded = ImageEnhance.Contrast(graded).enhance(float(CONFIG["contrast"]))
        graded = ImageEnhance.Color(graded).enhance(float(CONFIG["saturation"]))
        return np.asarray(graded)


# -----------------------------------------------------------------------------
# Procedural audio
# -----------------------------------------------------------------------------


def make_envelope(length: int, attack: float = 0.08, release: float = 0.60) -> np.ndarray:
    env = np.ones(length, dtype=np.float32)
    a = int(length * attack)
    r = int(length * release)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
    if r > 0:
        env[-r:] = np.minimum(env[-r:], np.linspace(1.0, 0.0, r, endpoint=True))
    return env


def create_soundtrack(path: Path, duration_s: float):
    sample_rate = int(CONFIG["audio_rate"])
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    rng = np.random.default_rng(1969)

    # restrained documentary bed
    audio = (
        0.15 * np.sin(2 * math.pi * 48.0 * t)
        + 0.08 * np.sin(2 * math.pi * 72.0 * t + 0.4)
        + 0.035 * np.sin(2 * math.pi * 144.0 * t + 0.9)
        + 0.014 * rng.normal(0.0, 1.0, n).astype(np.float32)
    )

    # transition impacts and whooshes
    for shot_name, start, _ in SHOT_PLAN[1:]:
        i0 = int(start * sample_rate)
        length = min(int(0.70 * sample_rate), n - i0)
        if length <= 0:
            continue
        tt = np.arange(length, dtype=np.float32) / sample_rate
        hit = 0.15 * np.sin(2 * math.pi * 92.0 * tt) * np.exp(-tt * 5.2)
        whoosh = 0.025 * rng.normal(0.0, 1.0, length).astype(np.float32) * np.exp(-tt * 3.0)
        if shot_name == "fresh_ejecta":
            hit += 0.12 * np.sin(2 * math.pi * 45.0 * tt) * np.exp(-tt * 2.7)
        audio[i0:i0 + length] += hit + whoosh

    # crystalline accents on bright crater reveals
    for start in [SHOT_PLAN[2][1], SHOT_PLAN[3][1], SHOT_PLAN[4][1]]:
        i0 = int(start * sample_rate)
        length = min(int(1.3 * sample_rate), n - i0)
        tt = np.arange(length, dtype=np.float32) / sample_rate
        chime = (
            0.055 * np.sin(2 * math.pi * 523.25 * tt)
            + 0.040 * np.sin(2 * math.pi * 659.25 * tt)
            + 0.026 * np.sin(2 * math.pi * 783.99 * tt)
        ) * np.exp(-tt * 1.6)
        audio[i0:i0 + length] += chime

    # finale shimmer
    i0 = int(max(0.0, duration_s - (1.4 if QUICK_MODE else 5.5)) * sample_rate)
    tt = np.arange(n - i0, dtype=np.float32) / sample_rate
    audio[i0:] += 0.045 * np.sin(2 * math.pi * 440.0 * tt) * np.exp(-tt * 0.42)

    audio /= max(float(np.max(np.abs(audio))), 1e-6)
    audio *= 0.72
    left = audio * (0.97 + 0.03 * np.sin(2 * math.pi * 0.035 * t))
    right = audio * (0.97 + 0.03 * np.cos(2 * math.pi * 0.035 * t + 0.8))
    stereo = np.stack([left, right], axis=1)
    pcm = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def render_video(scene: LunarCraterScene, output_path: Path):
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
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(final_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        shutil.copy2(video_path, final_path)
        return False


def save_contact_sheet(scene: LunarCraterScene, path: Path):
    samples = [
        SHOT_PLAN[0][1] + (SHOT_PLAN[0][2] - SHOT_PLAN[0][1]) * 0.55,
        SHOT_PLAN[2][1] + (SHOT_PLAN[2][2] - SHOT_PLAN[2][1]) * 0.55,
        SHOT_PLAN[3][1] + (SHOT_PLAN[3][2] - SHOT_PLAN[3][1]) * 0.55,
        SHOT_PLAN[4][1] + (SHOT_PLAN[4][2] - SHOT_PLAN[4][1]) * 0.62,
        SHOT_PLAN[5][1] + (SHOT_PLAN[5][2] - SHOT_PLAN[5][1]) * 0.60,
        SHOT_PLAN[6][1] + (SHOT_PLAN[6][2] - SHOT_PLAN[6][1]) * 0.62,
        SHOT_PLAN[7][1] + (SHOT_PLAN[7][2] - SHOT_PLAN[7][1]) * 0.58,
        SHOT_PLAN[8][1] + (SHOT_PLAN[8][2] - SHOT_PLAN[8][1]) * 0.60,
    ]
    thumbs: List[Image.Image] = []
    for sample_t in samples:
        frame = Image.fromarray(scene.frame(sample_t))
        thumbs.append(frame.resize((270, 480), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (270 * 4, 480 * 2), (5, 7, 12))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 4) * 270, (index // 4) * 480))
    sheet.save(path, quality=92)


def write_summary(path: Path, audio_muxed: bool):
    payload = {
        "title": CONFIG["title"],
        "subtitle": CONFIG["subtitle"],
        "featured_craters": CRATERS,
        "scientific_notes": [
            "The featured craters reflect sunlight; they are not self-luminous.",
            "Aristarchus is presented as one of the Moon's brightest features, not as a calibrated universal albedo ranking against every pixel on the Moon.",
            "Bright ray systems are associated with comparatively fresh ejecta and exposed immature material.",
            "Space weathering by solar-wind irradiation and micrometeoroid impacts changes lunar regolith optical properties over time.",
            "Composition, roughness, illumination angle, viewing angle, and camera processing can also alter apparent brightness.",
            "Procedural visuals are illustrative and are not calibrated LRO reflectance maps.",
        ],
        "official_sources": [
            "https://science.nasa.gov/resource/aristarchus-crater-2/",
            "https://science.nasa.gov/resource/tycho-crater-on-the-moon-labeled/",
            "https://svs.gsfc.nasa.gov/4220/",
            "https://apod.nasa.gov/apod/ap010809.html",
            "https://science.nasa.gov/moon/moonlight/",
            "https://science.nasa.gov/moon/lunar-craters/",
        ],
        "audio_muxed": audio_muxed,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    scene = LunarCraterScene()

    crater_csv = DATA_DIR / "featured_bright_lunar_craters.csv"
    silent_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_silent.mp4"
    audio_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_soundtrack.wav"
    final_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}.mp4"
    subtitles_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    summary_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_summary.json"
    contact_sheet_path = PREVIEW_DIR / f"{CONFIG['output_basename']}_contact_sheet.jpg"

    save_crater_csv(crater_csv)
    write_srt(CAPTIONS, subtitles_path)
    save_contact_sheet(scene, contact_sheet_path)
    render_video(scene, silent_video)
    create_soundtrack(audio_path, DURATION)
    audio_muxed = mux_audio(silent_video, audio_path, final_video)
    write_summary(summary_path, audio_muxed)

    print("Render complete")
    print(f"Video: {final_video.resolve()}")
    print(f"CSV: {crater_csv.resolve()}")
    print(f"SRT: {subtitles_path.resolve()}")
    print(f"Contact sheet: {contact_sheet_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")



if __name__ == "__main__":
    main()
