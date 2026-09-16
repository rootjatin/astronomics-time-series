from __future__ import annotations

"""
output : https://youtube.com/shorts/1g675ltD_kE?feature=share

The Strongest Winds Ever Measured in a Storm — cinematic YouTube Short renderer

A vertical 1080x1920 cinematic data-story built around the WMO-verified world
record surface wind gust measured during Tropical Cyclone Olivia at Barrow
Island, Western Australia, on 10 April 1996.

CORE FACTS USED
---------------
- Record gust: 113.2 m/s = 220 kt = 253 mph = 408 km/h
- Time: 1055 UTC, 10 April 1996
- Location: Barrow Island, Western Australia
- Instrument: heavy-duty three-cup Synchrotac anemometer, 10 m above ground
- Olivia was assessed as an Australian Category 4 cyclone with estimated mean
  maximum winds near peak intensity of about 195 km/h.
- Other measured gusts during Olivia included 267 km/h at Varanus Island and
  257 km/h at Mardie Station.

IMPORTANT FRAMING
-----------------
"Strongest winds ever measured in a storm" is used as the YouTube title hook.
The on-screen scientific label is more precise: WMO WORLD RECORD SURFACE GUST
(3-SECOND), measured during a tropical cyclone.

The moving cyclone cloud field, wind streaks, debris, gauge sweep, and short
wind-trace around the record are CINEMATIC RECONSTRUCTIONS. They are not raw
1996 satellite pixels or a recovered second-by-second station time series.
The verified peak value, time, location, and instrument are observational.

Quick preview
-------------
    WIND_RECORD_SHORT_QUICK=1 python the_strongest_winds_ever_measured_in_a_storm.py

Skip audio
----------
    WIND_RECORD_SHORT_NO_AUDIO=1 python the_strongest_winds_ever_measured_in_a_storm.py

Recommended install
-------------------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm

Outputs
-------
- final vertical MP4 with generated cinematic wind/rumble soundtrack
- silent MP4 fallback
- SRT subtitle sidecar
- preview PNG frames
- CSV of verified observation values used in the story
- JSON summary and source notes

Primary references
------------------
- WMO Archive of Weather and Climate Extremes — Maximum Surface Wind Gust:
  https://wmo.int/asu-map?map=Wind_028
- WMO records table (maximum gust: 113.2 m/s / 253 mph / 220 kt):
  https://wmo.int/sites/default/files/2024-07/Table_Records_02Jul2024.pdf
- Australian Bureau of Meteorology — Severe Tropical Cyclone Olivia:
  https://www.bom.gov.au/cyclone/history/olivia.shtml
- Bureau of Meteorology — Tropical Cyclone Olivia:
  https://www.bom.gov.au/cyclone/history/wa/olivia.shtml
"""

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
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("WIND_RECORD_SHORT_QUICK", "0") == "1"
NO_AUDIO = os.environ.get("WIND_RECORD_SHORT_NO_AUDIO", "0") == "1"

OUTPUT_ROOT = Path("the_strongest_winds_ever_measured_in_a_storm_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12.0 if QUICK_MODE else 58.0,
    "output_basename": "the_strongest_winds_ever_measured_in_a_storm",
    "title_1": "THE STRONGEST WINDS",
    "title_2": "EVER MEASURED IN A STORM",
    "subtitle": "TROPICAL CYCLONE OLIVIA // BARROW ISLAND // 10 APR 1996",
    "soundtrack_sample_rate": 22050 if QUICK_MODE else 44100,
    "grain_strength": 4.8,
    "vignette": 0.46,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
S = OUT_W / 1080.0

COLORS = {
    "black": (1, 2, 5),
    "navy": (3, 8, 17),
    "storm": (16, 24, 34),
    "cloud": (176, 196, 205),
    "cloud_hi": (236, 244, 246),
    "grid": (82, 121, 143),
    "white": (248, 250, 252),
    "muted": (159, 185, 199),
    "cyan": (102, 233, 255),
    "amber": (255, 189, 77),
    "orange": (255, 129, 63),
    "red": (255, 66, 66),
    "deep_red": (112, 19, 23),
}

RECORD = {
    "gust_mps": 113.2,
    "gust_kmh": 408.0,
    "gust_mph": 253.0,
    "gust_kt": 220.0,
    "time_utc": "1996-04-10T10:55:00Z",
    "location": "Barrow Island, Western Australia",
    "latitude": -20.8167,
    "longitude": 115.3833,
    "instrument": "Heavy-duty three-cup Synchrotac anemometer",
    "instrument_height_m": 10.0,
    "station_elevation_m": 64.0,
    "formal_wmo_review": "Verified by WMO review; record accepted in 2011 and documented in 2012",
}

OBSERVATIONS = [
    {"site": "Barrow Island", "gust_kmh": 408, "gust_mph": 253.0, "note": "WMO world-record 3-second gust"},
    {"site": "Varanus Island", "gust_kmh": 267, "gust_mph": 165.9, "note": "Extreme gust measured during Olivia"},
    {"site": "Mardie Station", "gust_kmh": 257, "gust_mph": 159.7, "note": "Extreme gust measured during Olivia"},
]

FULL_SHOT_PLAN = [
    {"name": "cold_open", "start": 0.0, "end": 7.0},
    {"name": "approach", "start": 7.0, "end": 19.0},
    {"name": "eyewall", "start": 19.0, "end": 31.5},
    {"name": "record", "start": 31.5, "end": 43.5},
    {"name": "context", "start": 43.5, "end": 53.5},
    {"name": "final", "start": 53.5, "end": 58.0},
]

FULL_CAPTIONS = [
    (0.4, 6.7, "On a small island off Western Australia, one weather station recorded a wind gust so violent it would become a world record."),
    (7.2, 18.6, "Tropical Cyclone Olivia was crossing the Pilbara region on April 10, 1996. The edge of the eye swept across Barrow Island."),
    (19.3, 31.0, "As the eyewall arrived, the station's heavy-duty anemometer measured a series of extraordinary gusts. Then the trace surged beyond anything previously verified."),
    (31.8, 43.0, "At 10:55 UTC, the instrument measured 113.2 meters per second — 253 miles per hour, or 408 kilometers per hour."),
    (43.8, 53.0, "That number was a brief three-second gust, not Olivia's sustained wind. The cyclone itself was assessed at Australian Category 4 intensity."),
    (53.7, 57.8, "The measurement was later formally verified by the World Meteorological Organization: the world-record surface wind gust."),
]

if QUICK_MODE:
    scale = float(CONFIG["duration_s"]) / 58.0
    SHOT_PLAN = [{"name": s["name"], "start": s["start"] * scale, "end": s["end"] * scale} for s in FULL_SHOT_PLAN]
    CAPTIONS = [(a * scale, b * scale, text) for a, b, text in FULL_CAPTIONS]
else:
    SHOT_PLAN = FULL_SHOT_PLAN
    CAPTIONS = FULL_CAPTIONS

SOURCE_URLS = {
    "wmo_record": "https://wmo.int/asu-map?map=Wind_028",
    "wmo_records_table": "https://wmo.int/sites/default/files/2024-07/Table_Records_02Jul2024.pdf",
    "bom_olivia": "https://www.bom.gov.au/cyclone/history/olivia.shtml",
    "bom_olivia_wa": "https://www.bom.gov.au/cyclone/history/wa/olivia.shtml",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(value: float) -> float:
    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


def smootherstep(value: float) -> float:
    x = clamp(value)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def shot_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - float(shot["start"])) / max(float(shot["end"] - shot["start"]), 1e-9))


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
    for i, (start, end, text) in enumerate(captions, 1):
        lines.extend([str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


def get_font(size: int, bold: bool = False, condensed: bool = False):
    candidates: List[str] = []
    if condensed and bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
            "DejaVuSansCondensed-Bold.ttf",
        ]
    elif condensed:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
            "DejaVuSansCondensed.ttf",
        ]
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(6, int(size)))
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int,
    fill: Tuple[int, int, int, int],
    bold: bool = False,
    condensed: bool = False,
    anchor: str = "la",
    stroke: int = 2,
):
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold=bold, condensed=condensed),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(235, fill[3])),
    )


def draw_wrapped(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill: Tuple[int, int, int, int],
    line_spacing: int,
    max_lines: int = 4,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size)
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
    lines = lines[:max_lines]
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 230))
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += (bbox[3] - bbox[1]) + line_spacing


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    r = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * r**1.75, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Data products
# -----------------------------------------------------------------------------

def save_data_products() -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "olivia_verified_wind_observations.csv"
    json_path = DATA_ROOT / "olivia_wind_record_summary.json"
    pd.DataFrame(OBSERVATIONS).to_csv(csv_path, index=False)
    payload = {
        "title": "The Strongest Winds Ever Measured in a Storm",
        "precise_claim": "WMO world-record maximum surface wind gust (3-second) measured during a tropical cyclone",
        "record": RECORD,
        "cyclone": {
            "name": "Severe Tropical Cyclone Olivia",
            "date_range": "5-12 April 1996",
            "australian_category_near_landfall": 4,
            "estimated_mean_maximum_wind_kmh": 195,
            "note": "The 408 km/h gust is not representative of Olivia's sustained/mean cyclone intensity.",
        },
        "observations": OBSERVATIONS,
        "visualization_note": (
            "Cyclone clouds, motion, wind streaks, debris, instrument sweep, and the animated short-duration wind trace "
            "are cinematic reconstructions. The peak gust value/time/location/instrument and listed site gusts are observational."
        ),
        "source_urls": SOURCE_URLS,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


# -----------------------------------------------------------------------------
# Cinematic scene
# -----------------------------------------------------------------------------

@dataclass
class Particle:
    x: float
    y: float
    length: float
    speed: float
    alpha: float
    phase: float


class WindRecordScene:
    def __init__(self):
        self.rng = np.random.default_rng(19960410)
        self.wind_particles = self._make_particles(115 if QUICK_MODE else 320)
        self.debris = self._make_debris(38 if QUICK_MODE else 110)
        self.stars = self._make_stars(55 if QUICK_MODE else 150)
        self.cloud_texture = self._make_cloud_texture()

    def _make_particles(self, count: int) -> List[Particle]:
        return [
            Particle(
                x=float(self.rng.uniform(-OUT_W * 0.2, OUT_W)),
                y=float(self.rng.uniform(0, OUT_H)),
                length=float(self.rng.uniform(20, 100) * S),
                speed=float(self.rng.uniform(120, 420) * S),
                alpha=float(self.rng.uniform(20, 115)),
                phase=float(self.rng.uniform(0, math.tau)),
            )
            for _ in range(count)
        ]

    def _make_debris(self, count: int) -> List[Particle]:
        return [
            Particle(
                x=float(self.rng.uniform(-OUT_W, OUT_W)),
                y=float(self.rng.uniform(OUT_H * 0.26, OUT_H * 0.88)),
                length=float(self.rng.uniform(3, 20) * S),
                speed=float(self.rng.uniform(160, 560) * S),
                alpha=float(self.rng.uniform(45, 150)),
                phase=float(self.rng.uniform(0, math.tau)),
            )
            for _ in range(count)
        ]

    def _make_stars(self, count: int) -> List[Tuple[float, float, float, float]]:
        return [
            (
                float(self.rng.uniform(0, OUT_W)),
                float(self.rng.uniform(0, OUT_H * 0.48)),
                float(self.rng.uniform(0.4, 1.5) * S),
                float(self.rng.uniform(20, 80)),
            )
            for _ in range(count)
        ]

    def _make_cloud_texture(self) -> Image.Image:
        # Deterministic multi-scale noise texture, reused each frame for speed.
        h = max(120, OUT_H // 3)
        w = max(120, OUT_W // 2)
        base = self.rng.normal(0.5, 0.18, (h, w)).astype(np.float32)
        img = Image.fromarray(np.uint8(np.clip(base, 0, 1) * 255), "L")
        img = img.resize(OUT_SIZE, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(max(6, int(20 * S))))
        return img

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
        top = np.array(COLORS["navy"], dtype=float)
        bot = np.array(COLORS["black"], dtype=float)
        for y in range(OUT_H):
            u = y / max(OUT_H - 1, 1)
            rgb = (top * (1 - u) + bot * u).astype(np.uint8)
            arr[y, :, :3] = rgb
            arr[y, :, 3] = 255
        image = Image.fromarray(arr, "RGBA")
        d = ImageDraw.Draw(image)
        for x, y, r, a in self.stars:
            pulse = 0.55 + 0.45 * math.sin(t * 0.55 + x * 0.01) ** 2
            d.ellipse((x-r, y-r, x+r, y+r), fill=(188, 217, 228, int(a * pulse)))
        return image

    def draw_cyclone(self, image: Image.Image, t: float, intensity: float, zoom: float = 1.0, center_y: float = 0.49):
        intensity = clamp(intensity)
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        cx = OUT_W * (0.52 + 0.025 * math.sin(t * 0.12))
        cy = OUT_H * center_y
        base_radius = OUT_W * (0.34 + 0.18 * intensity) * zoom
        rotation = -t * (0.28 + 0.58 * intensity)

        # Broad cloud canopy made of spiral ellipses.
        cloud_count = 52 if QUICK_MODE else 125
        for i in range(cloud_count):
            frac = i / max(cloud_count - 1, 1)
            arm = i % 5
            theta = rotation + frac * math.tau * (2.1 + intensity * 1.2) + arm * math.tau / 5
            radius = base_radius * (0.16 + 0.92 * frac)
            radius *= 1.0 + 0.08 * math.sin(i * 1.87 + t * 0.7)
            x = cx + math.cos(theta) * radius
            y = cy + math.sin(theta) * radius * 0.72
            w = base_radius * (0.22 - 0.105 * frac) * (0.8 + 0.45 * intensity)
            h = w * (0.55 + 0.20 * math.sin(theta * 2.0) ** 2)
            a = int(28 + 78 * intensity + 40 * (1 - frac))
            c = int(118 + 105 * (1 - frac) + 18 * intensity)
            ld.ellipse((x-w, y-h, x+w, y+h), fill=(c, c+5 if c < 250 else 250, min(255, c+12), min(170, a)))

        # Dense eyewall ring.
        eye_r = base_radius * lerp(0.17, 0.075, intensity)
        ring_r = eye_r * (2.5 + 0.35 * intensity)
        for j in range(28 if QUICK_MODE else 70):
            theta = rotation * 2.1 + j / (28 if QUICK_MODE else 70) * math.tau
            rr = ring_r * (0.88 + 0.14 * math.sin(j * 2.6 + t * 1.8))
            x = cx + math.cos(theta) * rr
            y = cy + math.sin(theta) * rr * 0.82
            blob = max(5 * S, eye_r * 0.46)
            alpha = int(90 + 120 * intensity)
            ld.ellipse((x-blob, y-blob*0.7, x+blob, y+blob*0.7), fill=COLORS["cloud_hi"] + (alpha,))

        # Eye darkness.
        ld.ellipse((cx-eye_r, cy-eye_r*0.8, cx+eye_r, cy+eye_r*0.8), fill=(2, 5, 10, int(175 + 60 * intensity)))
        ld.ellipse((cx-eye_r*0.58, cy-eye_r*0.46, cx+eye_r*0.58, cy+eye_r*0.46), fill=(9, 15, 23, 230))

        layer = layer.filter(ImageFilter.GaussianBlur(max(1, int(2.5 * S))))
        image.alpha_composite(layer)

        # Cloud texture masking for a richer satellite feel.
        texture = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        tex_alpha = self.cloud_texture.point(lambda v: int(v * 0.14 * intensity))
        texture.paste((212, 225, 230, 0), (0, 0, OUT_W, OUT_H))
        texture.putalpha(tex_alpha)
        image.alpha_composite(texture)

    def draw_island_station(self, image: Image.Image, t: float, storm: float, instrument_focus: float = 0.0):
        d = ImageDraw.Draw(image)
        horizon = int(OUT_H * 0.73)

        # Ocean / island silhouette.
        d.rectangle((0, horizon, OUT_W, OUT_H), fill=(2, 6, 10, 255))
        pts = [
            (0, int(OUT_H*0.79)),
            (int(OUT_W*0.17), int(OUT_H*0.765)),
            (int(OUT_W*0.35), int(OUT_H*0.78)),
            (int(OUT_W*0.56), int(OUT_H*0.75)),
            (int(OUT_W*0.78), int(OUT_H*0.77)),
            (OUT_W, int(OUT_H*0.745)),
            (OUT_W, OUT_H), (0, OUT_H)
        ]
        d.polygon(pts, fill=(8, 13, 14, 255))

        # Small anemometer tower silhouette — visual metaphor anchored to real instrument type.
        tx = int(OUT_W * lerp(0.72, 0.54, instrument_focus))
        base_y = int(OUT_H * lerp(0.79, 0.77, instrument_focus))
        tower_h = int(OUT_H * lerp(0.15, 0.22, instrument_focus))
        top_y = base_y - tower_h
        width = max(2, int(3 * S))
        d.line((tx, base_y, tx, top_y), fill=(182, 198, 198, 230), width=width)
        # guy wires
        d.line((tx, top_y+int(20*S), tx-int(45*S), base_y), fill=(90, 110, 111, 150), width=1)
        d.line((tx, top_y+int(20*S), tx+int(45*S), base_y), fill=(90, 110, 111, 150), width=1)
        hub_y = top_y
        arm = int(28 * S * (1.0 + 0.8 * instrument_focus))
        d.line((tx-arm, hub_y, tx+arm, hub_y), fill=(217, 226, 224, 240), width=max(1, int(2*S)))
        spin = t * (7 + 32 * storm)
        for k in range(3):
            theta = spin + k * math.tau / 3
            cx = tx + math.cos(theta) * arm * 0.65
            cy = hub_y + math.sin(theta) * arm * 0.28
            r = max(3, int(7 * S * (1 + instrument_focus)))
            d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(224, 232, 230, 238))
        d.ellipse((tx-int(5*S), hub_y-int(5*S), tx+int(5*S), hub_y+int(5*S)), fill=(245, 248, 245, 255))

        if instrument_focus > 0.35:
            draw_text(image, "SYNCHROTAC ANEMOMETER", (int(OUT_W*0.08), int(OUT_H*0.80)), 22 if not QUICK_MODE else 11,
                      COLORS["white"] + (235,), bold=True, condensed=True, stroke=1)
            draw_text(image, "10 M ABOVE GROUND // AUTOMATIC STATION", (int(OUT_W*0.08), int(OUT_H*0.835)), 15 if not QUICK_MODE else 8,
                      COLORS["muted"] + (220,), bold=True, condensed=True, stroke=1)

    def draw_wind_streaks(self, image: Image.Image, t: float, strength: float, chaos: float = 0.0):
        strength = clamp(strength)
        if strength <= 0.02:
            return
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for p in self.wind_particles:
            x = (p.x + t * p.speed * (0.30 + 1.8 * strength)) % (OUT_W + OUT_W * 0.4) - OUT_W * 0.2
            wobble = math.sin(t * 3.1 + p.phase) * (8 + 34 * chaos) * S
            y = (p.y + wobble + t * 7 * strength) % OUT_H
            length = p.length * (0.35 + 1.8 * strength)
            alpha = int(p.alpha * strength * (0.55 + 0.45 * math.sin(t * 2.7 + p.phase) ** 2))
            d.line((x, y, x+length, y + math.sin(p.phase) * 4*S), fill=(190, 224, 232, alpha), width=max(1, int((1 + strength) * S)))
        overlay = overlay.filter(ImageFilter.GaussianBlur(max(0.3, 0.8 * S)))
        image.alpha_composite(overlay)

    def draw_debris(self, image: Image.Image, t: float, strength: float):
        strength = clamp(strength)
        if strength < 0.35:
            return
        d = ImageDraw.Draw(image)
        for p in self.debris:
            x = (p.x + t * p.speed * (0.5 + strength * 1.7)) % (OUT_W * 2) - OUT_W * 0.4
            y = p.y + math.sin(t * 4.2 + p.phase) * 45 * S * strength
            length = p.length * (0.6 + 1.6 * strength)
            alpha = int(p.alpha * (strength - 0.25))
            d.line((x, y, x+length, y-math.sin(p.phase)*length*0.45), fill=(171, 160, 135, alpha), width=max(1, int(2*S)))

    def draw_lightning_flash(self, image: Image.Image, t: float, amount: float):
        # Cinematic punctuation only; Olivia-specific lightning is not claimed.
        amount = clamp(amount)
        if amount <= 0:
            return
        overlay = Image.new("RGBA", OUT_SIZE, (210, 231, 240, int(90 * amount)))
        image.alpha_composite(overlay)

    def draw_gauge(self, image: Image.Image, value_mph: float, reveal: float, dramatic: bool = False):
        reveal = clamp(reveal)
        cx = OUT_W // 2
        cy = int(OUT_H * (0.59 if dramatic else 0.63))
        radius = int(OUT_W * (0.37 if dramatic else 0.31))
        d = ImageDraw.Draw(image)

        # plate
        d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(3, 7, 12, 222), outline=(112, 145, 155, 125), width=max(1, int(3*S)))
        inner = int(radius * 0.82)
        d.ellipse((cx-inner, cy-inner, cx+inner, cy+inner), outline=(64, 91, 101, 115), width=max(1, int(2*S)))

        start_angle = math.radians(215)
        end_angle = math.radians(-35)
        max_val = 260.0
        for val in range(0, 261, 20):
            u = val / max_val
            a = start_angle + (end_angle - start_angle) * u
            r0 = radius * (0.74 if val % 40 else 0.70)
            r1 = radius * 0.84
            x0, y0 = cx + math.cos(a)*r0, cy + math.sin(a)*r0
            x1, y1 = cx + math.cos(a)*r1, cy + math.sin(a)*r1
            tick_color = COLORS["red"] + (230,) if val >= 200 else COLORS["muted"] + (160,)
            d.line((x0, y0, x1, y1), fill=tick_color, width=max(1, int(2*S)))
            if val % 40 == 0:
                tx = cx + math.cos(a) * radius * 0.60
                ty = cy + math.sin(a) * radius * 0.60
                draw_text(image, str(val), (int(tx), int(ty)), 17 if not QUICK_MODE else 9, COLORS["muted"] + (210,),
                          bold=True, condensed=True, anchor="mm", stroke=1)

        actual = clamp(value_mph / max_val)
        a = start_angle + (end_angle - start_angle) * actual
        nx = cx + math.cos(a) * radius * 0.67
        ny = cy + math.sin(a) * radius * 0.67
        needle_color = COLORS["red"] if value_mph >= 200 else COLORS["amber"]
        d.line((cx, cy, nx, ny), fill=needle_color + (255,), width=max(2, int(7*S)))
        d.ellipse((cx-int(10*S), cy-int(10*S), cx+int(10*S), cy+int(10*S)), fill=COLORS["white"] + (255,))
        draw_text(image, f"{value_mph:03.0f}", (cx, cy+int(radius*0.36)), 76 if not QUICK_MODE else 38,
                  COLORS["white"] + (255,), bold=True, condensed=True, anchor="mm", stroke=2)
        draw_text(image, "MPH GUST", (cx, cy+int(radius*0.55)), 22 if not QUICK_MODE else 11,
                  COLORS["muted"] + (230,), bold=True, condensed=True, anchor="mm", stroke=1)

    def draw_trace(self, image: Image.Image, t: float, progress: float, peak: float):
        # This is explicitly labeled as an illustrative reconstruction.
        left = int(OUT_W * 0.09)
        right = int(OUT_W * 0.91)
        top = int(OUT_H * 0.69)
        bottom = int(OUT_H * 0.83)
        d = ImageDraw.Draw(image)
        d.rounded_rectangle((left, top, right, bottom), radius=int(18*S), fill=(2, 6, 11, 205), outline=COLORS["grid"] + (65,), width=1)
        draw_text(image, "ILLUSTRATIVE GUST TRACE — PEAK VALUE IS VERIFIED", (left+int(18*S), top+int(16*S)),
                  14 if not QUICK_MODE else 7, COLORS["muted"] + (230,), bold=True, condensed=True, stroke=1)
        x0 = left + int(18*S)
        x1 = right - int(18*S)
        y0 = top + int(42*S)
        y1 = bottom - int(17*S)
        pts = []
        n = 170
        for i in range(n):
            u = i/(n-1)
            # baseline roughness + dramatic narrow verified peak
            base = 0.28 + 0.08*math.sin(u*math.tau*7 + 0.9) + 0.045*math.sin(u*math.tau*19)
            spike = math.exp(-0.5*((u-0.72)/0.025)**2) * (0.74 + 0.12*math.sin(u*math.tau*45))
            v = clamp(base + spike)
            if u > progress:
                v *= 0.05
            x = lerp(x0, x1, u)
            y = lerp(y1, y0, v)
            pts.append((x,y))
        d.line(pts, fill=COLORS["cyan"] + (215,), width=max(1, int(3*S)))
        peak_x = lerp(x0, x1, 0.72)
        if progress > 0.70:
            d.line((peak_x, y0, peak_x, y1), fill=COLORS["red"] + (130,), width=1)
            draw_text(image, "253", (int(peak_x), y0-int(8*S)), 18 if not QUICK_MODE else 9,
                      COLORS["red"] + (255,), bold=True, condensed=True, anchor="ms", stroke=1)

    def draw_header(self, image: Image.Image, t: float, shot_name: str):
        # Fade title after cold open; keep source ribbon.
        if shot_name == "cold_open":
            p = smoothstep(clamp(t / max(SHOT_PLAN[0]["end"] if False else float(SHOT_PLAN[0]["end"]), 1e-9)))
            alpha = int(255 * min(1.0, p*1.8))
            draw_text(image, CONFIG["title_1"], (OUT_W//2, int(OUT_H*0.11)), 45 if not QUICK_MODE else 23,
                      COLORS["white"] + (alpha,), bold=True, condensed=True, anchor="mm", stroke=2)
            draw_text(image, CONFIG["title_2"], (OUT_W//2, int(OUT_H*0.16)), 42 if not QUICK_MODE else 21,
                      COLORS["white"] + (alpha,), bold=True, condensed=True, anchor="mm", stroke=2)
        else:
            draw_text(image, "CYCLONE OLIVIA // 10 APR 1996", (int(OUT_W*0.06), int(OUT_H*0.055)),
                      18 if not QUICK_MODE else 9, COLORS["muted"] + (225,), bold=True, condensed=True, stroke=1)

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        x0 = int(OUT_W*0.055)
        x1 = int(OUT_W*0.945)
        y0 = int(OUT_H*0.865)
        y1 = int(OUT_H*0.955)
        pd.rounded_rectangle((x0,y0,x1,y1), radius=int(18*S), fill=(0,3,8,175), outline=(90,130,145,50), width=1)
        image.alpha_composite(panel)
        draw_wrapped(image, text, (x0+int(22*S), y0+int(18*S)), x1-x0-int(44*S),
                     21 if not QUICK_MODE else 10, COLORS["white"] + (238,), 7 if not QUICK_MODE else 3, max_lines=3)

    def draw_cold_open(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smootherstep(shot_progress(t, shot))
        self.draw_cyclone(image, t, 0.35 + 0.28*p, zoom=1.17, center_y=0.52)
        self.draw_island_station(image, t, 0.20+0.25*p, instrument_focus=0.0)
        self.draw_wind_streaks(image, t, 0.08+0.20*p)
        if p > 0.58:
            draw_text(image, "BARROW ISLAND", (OUT_W//2, int(OUT_H*0.59)), 26 if not QUICK_MODE else 13,
                      COLORS["white"] + (220,), bold=True, condensed=True, anchor="mm", stroke=1)
            draw_text(image, "WESTERN AUSTRALIA", (OUT_W//2, int(OUT_H*0.625)), 15 if not QUICK_MODE else 8,
                      COLORS["muted"] + (210,), bold=True, condensed=True, anchor="mm", stroke=1)

    def draw_approach(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smootherstep(shot_progress(t, shot))
        self.draw_cyclone(image, t, 0.55+0.28*p, zoom=1.07+0.08*p, center_y=0.48)
        self.draw_island_station(image, t, 0.35+0.40*p, instrument_focus=0.05)
        self.draw_wind_streaks(image, t, 0.28+0.40*p, chaos=0.15)
        self.draw_debris(image, t, 0.30+0.30*p)
        draw_text(image, "THE EYEWALL IS APPROACHING", (OUT_W//2, int(OUT_H*0.71)), 28 if not QUICK_MODE else 14,
                  COLORS["white"] + (235,), bold=True, condensed=True, anchor="mm", stroke=2)
        draw_text(image, "EDGE OF THE EYE PASSES BARROW ISLAND", (OUT_W//2, int(OUT_H*0.75)), 15 if not QUICK_MODE else 8,
                  COLORS["muted"] + (220,), bold=True, condensed=True, anchor="mm", stroke=1)

    def draw_eyewall(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smootherstep(shot_progress(t, shot))
        self.draw_cyclone(image, t, 0.82+0.17*p, zoom=1.28, center_y=0.44)
        self.draw_island_station(image, t, 0.8+0.2*p, instrument_focus=0.55+0.35*p)
        self.draw_wind_streaks(image, t, 0.70+0.30*p, chaos=0.55+0.45*p)
        self.draw_debris(image, t, 0.78+0.22*p)

        # Dynamic camera shake via post-composition is handled in render_frame.
        value = lerp(105, 220, smoothstep(p)) + 8*math.sin(t*4.0)*p
        value = min(value, 232.0)
        self.draw_gauge(image, value, p, dramatic=False)
        if p > 0.55:
            draw_text(image, "HEAVY-DUTY AUTOMATIC ANEMOMETER", (OUT_W//2, int(OUT_H*0.80)), 15 if not QUICK_MODE else 8,
                      COLORS["muted"] + (230,), bold=True, condensed=True, anchor="mm", stroke=1)

    def draw_record(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = shot_progress(t, shot)
        # First 35% builds, then record hits with a micro-blackout / flash.
        hit = smoothstep(clamp((p-0.32)/0.12))
        value = lerp(228, RECORD["gust_mph"], hit)
        self.draw_cyclone(image, t, 0.98, zoom=1.36, center_y=0.42)
        self.draw_wind_streaks(image, t, 0.95, chaos=1.0)
        self.draw_debris(image, t, 1.0)
        self.draw_gauge(image, value, p, dramatic=True)
        self.draw_trace(image, t, clamp(p*1.08), RECORD["gust_mph"])

        # Sudden exposure blast at the record hit.
        pulse = math.exp(-0.5*((p-0.42)/0.025)**2)
        self.draw_lightning_flash(image, t, pulse)

        if p > 0.43:
            a = smoothstep((p-0.43)/0.14)
            draw_text(image, "253 MPH", (OUT_W//2, int(OUT_H*0.19)), 95 if not QUICK_MODE else 47,
                      COLORS["white"] + (int(255*a),), bold=True, condensed=True, anchor="mm", stroke=3)
            draw_text(image, "408 KM/H  //  113.2 M/S  //  220 KT", (OUT_W//2, int(OUT_H*0.255)),
                      22 if not QUICK_MODE else 11, COLORS["red"] + (int(255*a),), bold=True, condensed=True, anchor="mm", stroke=1)
            draw_text(image, "10:55 UTC // 10 APRIL 1996", (OUT_W//2, int(OUT_H*0.292)),
                      17 if not QUICK_MODE else 9, COLORS["muted"] + (int(235*a),), bold=True, condensed=True, anchor="mm", stroke=1)

    def draw_context(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smootherstep(shot_progress(t, shot))
        self.draw_cyclone(image, t*0.55, 0.72, zoom=1.15, center_y=0.44)
        self.draw_wind_streaks(image, t, 0.38*(1-p)+0.15, chaos=0.2)

        # Elegant comparison bars: short gust vs estimated mean max wind.
        panel = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        pd = ImageDraw.Draw(panel)
        x0, x1 = int(OUT_W*0.075), int(OUT_W*0.925)
        y0, y1 = int(OUT_H*0.36), int(OUT_H*0.72)
        pd.rounded_rectangle((x0,y0,x1,y1), radius=int(24*S), fill=(2,6,12,210), outline=COLORS["grid"]+(75,), width=1)
        image.alpha_composite(panel)
        draw_text(image, "WHAT 253 MPH ACTUALLY MEANS", (x0+int(28*S), y0+int(34*S)), 29 if not QUICK_MODE else 15,
                  COLORS["white"]+(250,), bold=True, condensed=True, stroke=1)
        draw_text(image, "A BRIEF 3-SECOND GUST — NOT SUSTAINED WIND", (x0+int(28*S), y0+int(78*S)), 16 if not QUICK_MODE else 8,
                  COLORS["muted"]+(235,), bold=True, condensed=True, stroke=1)

        bar_left = x0+int(42*S)
        bar_right = x1-int(42*S)
        bar_w = bar_right-bar_left
        # mean wind ~121 mph
        y_mean = y0+int(155*S)
        y_gust = y0+int(240*S)
        for y, label, mph, col in [
            (y_mean, "OLIVIA ESTIMATED MEAN MAX WIND", 121.0, COLORS["cyan"]),
            (y_gust, "BARROW ISLAND RECORD GUST", 253.0, COLORS["red"]),
        ]:
            draw_text(image, label, (bar_left, y-int(30*S)), 15 if not QUICK_MODE else 8, COLORS["muted"]+(225,), bold=True, condensed=True, stroke=1)
            pd = ImageDraw.Draw(image)
            pd.rounded_rectangle((bar_left,y,bar_right,y+int(25*S)), radius=int(12*S), fill=(23,34,42,220))
            frac = clamp(mph/260.0) * smoothstep(clamp(p*1.25))
            pd.rounded_rectangle((bar_left,y,bar_left+int(bar_w*frac),y+int(25*S)), radius=int(12*S), fill=col+(235,))
            draw_text(image, f"{mph:.0f} MPH", (bar_right, y-int(3*S)), 22 if not QUICK_MODE else 11,
                      col+(255,), bold=True, condensed=True, anchor="rs", stroke=1)

        draw_text(image, "AUSTRALIAN CATEGORY 4 CYCLONE", (OUT_W//2, int(OUT_H*0.77)), 21 if not QUICK_MODE else 11,
                  COLORS["white"]+(235,), bold=True, condensed=True, anchor="mm", stroke=1)

    def draw_final(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smootherstep(shot_progress(t, shot))
        self.draw_cyclone(image, t*0.25, 0.38*(1-p), zoom=1.35, center_y=0.50)
        draw_text(image, "WORLD RECORD", (OUT_W//2, int(OUT_H*0.26)), 34 if not QUICK_MODE else 17,
                  COLORS["red"]+(255,), bold=True, condensed=True, anchor="mm", stroke=1)
        draw_text(image, "253 MPH", (OUT_W//2, int(OUT_H*0.38)), 120 if not QUICK_MODE else 60,
                  COLORS["white"]+(255,), bold=True, condensed=True, anchor="mm", stroke=3)
        draw_text(image, "MAXIMUM SURFACE WIND GUST", (OUT_W//2, int(OUT_H*0.465)), 28 if not QUICK_MODE else 14,
                  COLORS["white"]+(245,), bold=True, condensed=True, anchor="mm", stroke=1)
        draw_text(image, "FORMALLY VERIFIED BY THE WMO", (OUT_W//2, int(OUT_H*0.515)), 19 if not QUICK_MODE else 10,
                  COLORS["muted"]+(230,), bold=True, condensed=True, anchor="mm", stroke=1)
        draw_text(image, "BARROW ISLAND // CYCLONE OLIVIA // 10 APR 1996", (OUT_W//2, int(OUT_H*0.59)), 17 if not QUICK_MODE else 9,
                  COLORS["muted"]+(220,), bold=True, condensed=True, anchor="mm", stroke=1)
        # Thin red record line
        d = ImageDraw.Draw(image)
        width = int(OUT_W*0.62*smoothstep(p*1.6))
        d.line((OUT_W//2-width//2, int(OUT_H*0.64), OUT_W//2+width//2, int(OUT_H*0.64)), fill=COLORS["red"]+(220,), width=max(1,int(3*S)))

    def film_finish(self, image: Image.Image, t: float, shot_name: str) -> Image.Image:
        # Camera shake peaks in eyewall/record sequence.
        shake_strength = 0.0
        if shot_name == "eyewall":
            shake_strength = 8.0 * S
        elif shot_name == "record":
            shake_strength = 11.0 * S
        if shake_strength > 0:
            dx = int(math.sin(t*23.0)*shake_strength + math.sin(t*41.0)*shake_strength*0.35)
            dy = int(math.cos(t*29.0)*shake_strength*0.5)
            shifted = Image.new("RGBA", OUT_SIZE, COLORS["black"]+(255,))
            shifted.alpha_composite(image, (dx,dy))
            image = shifted

        arr = np.asarray(image.convert("RGB")).astype(np.float32)
        arr *= VIGNETTE[..., None]
        # deterministic per-frame grain
        rng = np.random.default_rng(int(t*1000)+881)
        grain = rng.normal(0.0, float(CONFIG["grain_strength"]), arr.shape[:2])[...,None]
        arr = np.clip(arr + grain, 0, 255)
        # Slight contrast lift.
        out = Image.fromarray(arr.astype(np.uint8), "RGB")
        out = ImageEnhance.Contrast(out).enhance(1.08)
        return out

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        name = str(shot["name"])
        image = self.background(t)

        if name == "cold_open":
            self.draw_cold_open(image, t, shot)
        elif name == "approach":
            self.draw_approach(image, t, shot)
        elif name == "eyewall":
            self.draw_eyewall(image, t, shot)
        elif name == "record":
            self.draw_record(image, t, shot)
        elif name == "context":
            self.draw_context(image, t, shot)
        else:
            self.draw_final(image, t, shot)

        self.draw_header(image, t, name)
        self.draw_caption(image, t)

        # Always-visible source ribbon above captions.
        draw_text(image, "WMO RECORD // BOM CYCLONE OLIVIA", (int(OUT_W*0.055), int(OUT_H*0.845)),
                  13 if not QUICK_MODE else 7, COLORS["muted"]+(185,), bold=True, condensed=True, stroke=1)

        finished = self.film_finish(image, t, name)
        return np.asarray(finished)


# -----------------------------------------------------------------------------
# Soundtrack / encoding
# -----------------------------------------------------------------------------

def gaussian(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((times-center)/max(width,1e-6))**2)


def generate_soundtrack(path: Path) -> Path:
    sr = int(CONFIG["soundtrack_sample_rate"])
    duration = float(CONFIG["duration_s"])
    count = int(round(sr*duration))
    times = np.arange(count, dtype=np.float64)/sr
    rng = np.random.default_rng(408220)

    audio = np.zeros(count, dtype=np.float64)
    # Sub-bass bed / distant turbine feel.
    audio += 0.065*np.sin(math.tau*28.0*times + 0.35*np.sin(math.tau*0.08*times))
    audio += 0.035*np.sin(math.tau*41.5*times + 1.1)

    # Wind noise envelope that intensifies through eyewall and record.
    controls = rng.normal(0,1,max(12,int(duration*9)))
    wind_noise = np.interp(times, np.linspace(0,duration,len(controls)), controls)
    white = rng.normal(0,1,count)
    # Smooth high-noise into a broad roar.
    kernel_n = max(8, int(sr*0.006))
    kernel = np.ones(kernel_n)/kernel_n
    roar = np.convolve(white, kernel, mode="same")
    env = np.zeros(count)
    for shot in SHOT_PLAN:
        mask = (times>=shot["start"]) & (times<shot["end"])
        if shot["name"] == "cold_open": level=0.05
        elif shot["name"] == "approach": level=0.13
        elif shot["name"] == "eyewall": level=0.27
        elif shot["name"] == "record": level=0.34
        elif shot["name"] == "context": level=0.10
        else: level=0.04
        env[mask] = level
    audio += roar*env*2.1 + wind_noise*env*0.035

    # Low pulse like a trailer heartbeat.
    for sec in np.arange(4.0, duration*0.56, 1.45 if not QUICK_MODE else 0.36):
        sec = sec if not QUICK_MODE else sec
        audio += 0.11*gaussian(times, sec, 0.07 if not QUICK_MODE else 0.025)*np.sin(math.tau*58*times)

    # Record-impact hit around 40% into record shot.
    record_shot = next(s for s in SHOT_PLAN if s["name"]=="record")
    hit = lerp(record_shot["start"], record_shot["end"], 0.42)
    impact = gaussian(times, hit, 0.11 if not QUICK_MODE else 0.035)
    audio += impact*(0.42*np.sin(math.tau*48*times) + 0.18*rng.normal(0,1,count))
    # metallic after-ring
    tail = np.maximum(times-hit,0)
    ring = (times>=hit)*np.exp(-tail/1.7)*np.sin(math.tau*310*times)
    audio += 0.08*ring

    # Final low cinematic swell.
    final_shot = next(s for s in SHOT_PLAN if s["name"]=="final")
    final_center = (final_shot["start"]+final_shot["end"])/2
    audio += 0.10*gaussian(times, final_center, max(0.35,(final_shot["end"]-final_shot["start"])*0.55))*np.sin(math.tau*52*times)

    intro = smoothstep_array(np.clip(times/max(0.7,duration*0.025),0,1))
    outro_x = np.clip((times-(duration-1.0))/0.9,0,1)
    outro = 1.0-smoothstep_array(outro_x)
    audio *= intro*outro
    peak = max(float(np.max(np.abs(audio))), 1e-9)
    audio = np.clip(audio/peak*0.90,-1,1)
    pcm = (audio*32767).astype(np.int16)
    with wave.open(str(path),"wb") as h:
        h.setnchannels(1); h.setsampwidth(2); h.setframerate(sr); h.writeframes(pcm.tobytes())
    return path


def smoothstep_array(x: np.ndarray) -> np.ndarray:
    x = np.clip(x,0.0,1.0)
    return x*x*(3.0-2.0*x)


def find_ffmpeg() -> Optional[str]:
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return shutil.which("ffmpeg")


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    cmd = [ffmpeg,"-y","-i",str(video_path),"-i",str(audio_path),"-c:v","copy","-c:a","aac","-b:a","192k","-shortest",str(output_path)]
    try:
        subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return output_path.exists() and output_path.stat().st_size>0
    except Exception:
        return False


def render_video(scene: WindRecordScene) -> Path:
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    raw_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_silent.mp4"
    final_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    audio_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_cinematic.wav"
    frame_count = int(round(float(CONFIG["duration_s"])*int(CONFIG["fps"])))
    times = np.arange(frame_count)/int(CONFIG["fps"])

    print("Rendering", frame_count, "frames at", OUT_W, "x", OUT_H)
    with iio.get_writer(raw_video,fps=int(CONFIG["fps"]),codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=None) as writer:
        for t in tqdm(times,desc="Rendering wind-record short"):
            writer.append_data(scene.render_frame(float(t)))

    if NO_AUDIO:
        shutil.copyfile(raw_video, final_video)
        return final_video
    generate_soundtrack(audio_path)
    if mux_audio(raw_video,audio_path,final_video):
        return final_video
    shutil.copyfile(raw_video,final_video)
    return final_video


def main():
    csv_path, summary_path = save_data_products()
    print("Story: Tropical Cyclone Olivia / Barrow Island")
    print("Verified record gust: 113.2 m/s = 253 mph = 408 km/h")
    print("CSV:", csv_path.resolve())
    print("Summary:", summary_path.resolve())

    scene = WindRecordScene()
    preview_times = [
        min(float(CONFIG["duration_s"])-0.5, x) for x in (
            1.0,
            float(CONFIG["duration_s"])*0.20,
            float(CONFIG["duration_s"])*0.41,
            float(CONFIG["duration_s"])*0.61,
            float(CONFIG["duration_s"])*0.80,
            float(CONFIG["duration_s"])-0.7,
        )
    ]
    for pt in tqdm(preview_times,desc="Preview frames"):
        Image.fromarray(scene.render_frame(float(pt))).save(PREVIEW_DIR/f"preview_{int(pt):02d}s.png")

    final = render_video(scene)
    print("Final video:", final.resolve())
    print("Output directory:", OUTPUT_ROOT.resolve())

if __name__ == "__main__":
    main()

