from __future__ import annotations

"""
Dust Storms Covered Mars
========================

Cinematic vertical YouTube Shorts renderer about the 2018 Mars
planet-encircling dust event. The visuals are procedural; the dates, scale,
and Opportunity opacity value are based on NASA/JPL reporting.

The wording "covered Mars" is cinematic. NASA more precisely calls the event
"planet-encircling" and notes that such storms do not literally cover every
square metre of the planet.

Real timeline used
------------------
- 2018-05-30: MRO detected the storm.
- 2018-06-06: the storm exceeded 18 million km² and Opportunity shifted to
  minimal operations.
- 2018-06-10: Opportunity's last signal; measured opacity tau was about 10.8.
- 2018-06-12: the storm blanketed about one quarter of Mars.
- 2018-06-19: officially classified as planet-encircling.
- Early July: peak global intensity.
- 2018-07-23: more dust was falling than being raised; decay phase.
- 2018-09-20: skies over Opportunity's location had substantially cleared.


Modes
-----
Standard 52-second render:
    python dust_storms_covered_mars_cinematic_short.py

Fast 13-second validation preview:
    MARS_DUST_SHORT_QUICK=1 python dust_storms_covered_mars_cinematic_short.py

True 4K vertical:
    MARS_DUST_SHORT_4K=1 python dust_storms_covered_mars_cinematic_short.py

Disable soundtrack:
    MARS_DUST_SHORT_SOUND=0 python dust_storms_covered_mars_cinematic_short.py
"""

import csv
import json
import math
import os
import subprocess
import wave
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK = os.environ.get("MARS_DUST_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("MARS_DUST_SHORT_4K", "0") == "1" and not QUICK
SOUND = os.environ.get("MARS_DUST_SHORT_SOUND", "1") != "0"

W = 540 if QUICK else (2160 if FOUR_K else 1080)
H = 960 if QUICK else (3840 if FOUR_K else 1920)
FPS = 8 if QUICK else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK else 52.0
SIZE = (W, H)
SCALE = W / 1080.0

OUTPUT = Path("dust_storms_covered_mars_output")
DATA_DIR = OUTPUT / "data"
PREVIEW_DIR = OUTPUT / "previews"
for directory in (OUTPUT, DATA_DIR, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

BASENAME = "dust_storms_covered_mars"

COLORS = {
    "white": (248, 245, 236),
    "muted": (191, 187, 179),
    "gold": (255, 193, 84),
    "orange": (246, 117, 49),
    "red": (204, 65, 36),
    "dust": (164, 82, 42),
    "dust_light": (222, 132, 72),
    "cyan": (96, 211, 230),
    "blue": (60, 112, 171),
    "deep": (3, 5, 10),
    "night": (7, 9, 14),
    "panel": (5, 7, 12),
}

SOURCE_URLS = {
    "nasa_svs_timeline": "https://svs.gsfc.nasa.gov/30983/",
    "jpl_opportunity_storm": "https://www.jpl.nasa.gov/news/opportunity-hunkers-down-during-dust-storm/",
    "jpl_global_event": "https://www.jpl.nasa.gov/news/martian-dust-storm-grows-global-curiosity-captures-photos-of-thickening-haze/",
    "nasa_mcs_animation": "https://science.nasa.gov/photojournal/mars-climate-sounder-studies-2018-dust-storm/",
    "nasa_opportunity": "https://science.nasa.gov/mission/mer-opportunity/",
    "nasa_after_storm": "https://science.nasa.gov/resource/opportunity-after-the-dust-storm/",
}


@dataclass(frozen=True)
class StormEvent:
    date_iso: str
    day_index: int
    label: str
    fact: str
    extent_fraction: float
    tau: float | None = None


# extent_fraction is used only to stage the procedural animation. Values tied
# to direct NASA wording are 0.25 on June 12 and 1.0 for planet-encircling.
# Other values are conservative visual interpolation and are explicitly marked
# as such in the metadata file.
EVENTS: List[StormEvent] = [
    StormEvent("2018-05-30", 0, "FIRST DETECTION", "MRO detects a regional storm", 0.04),
    StormEvent("2018-06-06", 7, "RAPID EXPANSION", "More than 18 million km²", 0.16),
    StormEvent("2018-06-10", 11, "OPPORTUNITY GOES QUIET", "Last signal • tau ≈ 10.8", 0.22, 10.8),
    StormEvent("2018-06-12", 13, "ONE QUARTER OF MARS", "Storm blankets about 25%", 0.25),
    StormEvent("2018-06-19", 20, "PLANET-ENCIRCLING", "Dust haze wraps around Mars", 1.0),
    StormEvent("2018-07-05", 36, "PEAK INTENSITY", "Global intensity peaks in early July", 1.0),
    StormEvent("2018-07-23", 54, "DECAY BEGINS", "More dust falls than rises", 0.82),
    StormEvent("2018-09-20", 113, "SKIES CLEAR", "Storm substantially cleared locally", 0.16),
]

# Every narrative caption lasts no more than 4 seconds, followed by clean visual
# breathing room. This is intentionally sparse for mobile readability.
FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.45, 3.85, "In 2018, a regional storm began spreading across Mars."),
    (7.00, 10.50, "Within thirteen days, dust blanketed about one quarter of the planet."),
    (15.10, 18.70, "At Opportunity, atmospheric opacity climbed to tau 10.8."),
    (23.10, 26.60, "Sunlight collapsed. The solar-powered rover sent its final signal."),
    (31.00, 34.70, "By June 19, NASA classified the storm as planet-encircling."),
    (39.00, 42.50, "Orbiters watched the dust rise, heat the atmosphere, and circle Mars."),
    (47.00, 50.50, "The skies cleared. Opportunity never called home again."),
]

SHOT_PLAN_FULL: List[Tuple[str, float, float]] = [
    ("intro", 0.0, 6.5),
    ("growth", 6.5, 14.2),
    ("opacity", 14.2, 22.0),
    ("silence", 22.0, 30.0),
    ("global", 30.0, 38.0),
    ("orbiters", 38.0, 46.0),
    ("finale", 46.0, 52.0),
]

if QUICK:
    q = DURATION / 52.0
    CAPTIONS = [(a * q, b * q, text) for a, b, text in FULL_CAPTIONS]
    SHOT_PLAN = [(name, a * q, b * q) for name, a, b in SHOT_PLAN_FULL]
else:
    CAPTIONS = FULL_CAPTIONS
    SHOT_PLAN = SHOT_PLAN_FULL


# -----------------------------------------------------------------------------
# Data products
# -----------------------------------------------------------------------------


def write_data_products() -> Tuple[Path, Path]:
    csv_path = DATA_DIR / "mars_2018_dust_storm_timeline.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["date", "day_index", "label", "fact", "extent_fraction_for_animation", "tau"],
        )
        writer.writeheader()
        for event in EVENTS:
            writer.writerow(
                {
                    "date": event.date_iso,
                    "day_index": event.day_index,
                    "label": event.label,
                    "fact": event.fact,
                    "extent_fraction_for_animation": event.extent_fraction,
                    "tau": "" if event.tau is None else event.tau,
                }
            )

    json_path = DATA_DIR / "mars_2018_dust_storm_method_and_sources.json"
    payload = {
        "title": "Dust Storms Covered Mars",
        "scope": "2018 Mars planet-encircling dust event, with Opportunity as the surface story",
        "terminology_note": (
            "NASA uses planet-encircling because these events do not literally cover every square metre. "
            "The title is cinematic shorthand."
        ),
        "verified_facts": {
            "detected": "2018-05-30",
            "quarter_planet": "2018-06-12",
            "planet_encircling": "2018-06-19",
            "opportunity_last_signal": "2018-06-10",
            "opportunity_tau": 10.8,
            "decay_phase": "2018-07-23",
            "local_clearing_image": "2018-09-20",
        },
        "visualization_note": (
            "The procedural dust extent is a cinematic interpolation between verified timeline anchors. "
            "It is not a pixel-by-pixel scientific reconstruction."
        ),
        "sources": SOURCE_URLS,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def smoothstep(x: float) -> float:
    x = clamp(x)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def shot_at(t: float) -> Tuple[str, float, float]:
    for shot in SHOT_PLAN:
        if shot[1] <= t < shot[2]:
            return shot
    return SHOT_PLAN[-1]


def caption_at(t: float) -> Tuple[str, float] | None:
    for start, end, value in CAPTIONS:
        if start <= t < end:
            fade = min(clamp((t - start) / 0.35), clamp((end - t) / 0.35))
            return value, smoothstep(fade)
    return None


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, max(7, int(size * SCALE)))
        except Exception:
            pass
    return ImageFont.load_default()


def text(
    image: Image.Image,
    value: str,
    xy: Tuple[float, float],
    size: int,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    anchor: str = "la",
    stroke: int = 2,
) -> None:
    ImageDraw.Draw(image).text(
        xy,
        value,
        font=font(size, bold),
        fill=fill,
        anchor=anchor,
        stroke_width=max(0, int(stroke * SCALE)),
        stroke_fill=(0, 0, 0, min(220, fill[3] if len(fill) > 3 else 220)),
    )


def wrapped(
    image: Image.Image,
    value: str,
    box: Tuple[int, int, int, int],
    size: int,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    spacing: int = 6,
    align: str = "center",
) -> None:
    draw = ImageDraw.Draw(image)
    fnt = font(size, bold)
    x0, y0, x1, y1 = box
    max_width = x1 - x0
    words = value.split()
    lines: List[str] = []
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        bounds = draw.textbbox((0, 0), candidate, font=fnt, stroke_width=max(1, int(SCALE)))
        if bounds[2] - bounds[0] <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    heights = []
    for ln in lines:
        b = draw.textbbox((0, 0), ln, font=fnt, stroke_width=max(1, int(SCALE)))
        heights.append(b[3] - b[1])
    total_h = sum(heights) + max(0, len(lines) - 1) * int(spacing * SCALE)
    y = y0 + max(0, (y1 - y0 - total_h) // 2)
    for ln, h in zip(lines, heights):
        if align == "center":
            x, anchor = (x0 + x1) // 2, "ma"
        elif align == "right":
            x, anchor = x1, "ra"
        else:
            x, anchor = x0, "la"
        draw.text(
            (x, y),
            ln,
            font=fnt,
            fill=fill,
            anchor=anchor,
            stroke_width=max(1, int(2 * SCALE)),
            stroke_fill=(0, 0, 0, 220),
        )
        y += h + int(spacing * SCALE)


def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 165, radius: int = 24) -> None:
    layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle(
        box,
        radius=max(4, int(radius * SCALE)),
        fill=COLORS["panel"] + (alpha,),
        outline=(255, 177, 92, 60),
        width=max(1, int(SCALE)),
    )
    image.alpha_composite(layer)


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hh = ms // 3_600_000
    ms %= 3_600_000
    mm = ms // 60_000
    ms %= 60_000
    ss = ms // 1000
    ms %= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def write_srt(path: Path) -> Path:
    lines: List[str] = []
    for index, (start, end, value) in enumerate(CAPTIONS, 1):
        lines.extend([str(index), f"{srt_time(start)} --> {srt_time(end)}", value, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_vignette() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    nx = (xx - W / 2) / (W / 2)
    ny = (yy - H / 2) / (H / 2)
    r = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - 0.27 * r**1.75, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette()


@dataclass
class Star:
    x: float
    y: float
    radius: float
    alpha: float
    phase: float


# -----------------------------------------------------------------------------
# Renderer
# -----------------------------------------------------------------------------


class Renderer:
    def __init__(self) -> None:
        rng = np.random.default_rng(2018)
        self.stars = [
            Star(
                float(rng.uniform(0, W)),
                float(rng.uniform(0, H * 0.74)),
                float(rng.uniform(0.3, 1.8) * SCALE),
                float(rng.uniform(20, 125)),
                float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(260 if QUICK else 620)
        ]
        self.terrain = self.make_terrain(20180610)
        self.planet_texture = self.make_planet_texture(20180530)
        self.dust_texture = self.make_dust_texture(20180619)
        self.rover = self.make_rover_silhouette()

    @staticmethod
    def make_terrain(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        depth, width = 170, 220
        base = rng.normal(0, 1, (depth, width)).astype(np.float32)
        image = Image.fromarray(np.uint8((base - base.min()) / max(float(np.ptp(base)), 1e-6) * 255))
        accum = np.zeros_like(base)
        for blur, weight in ((24, 1.25), (11, 0.75), (4, 0.38), (1, 0.14)):
            blurred = image.filter(ImageFilter.GaussianBlur(blur))
            accum += np.asarray(blurred, dtype=np.float32) / 255.0 * weight
        yy, xx = np.mgrid[0:depth, 0:width]
        accum += 0.25 * np.sin(xx / 17.0 + yy / 29.0)
        accum += 0.14 * np.cos((xx - yy) / 23.0)
        accum -= accum.min()
        accum /= max(float(accum.max()), 1e-6)
        return accum

    @staticmethod
    def make_planet_texture(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        size = 700 if QUICK else 1300
        texture = rng.normal(0, 1, (size, size)).astype(np.float32)
        image = Image.fromarray(np.uint8((texture - texture.min()) / np.ptp(texture) * 255))
        layers = []
        for blur in (70, 30, 12, 4):
            layers.append(np.asarray(image.filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32) / 255.0)
        out = 0.48 * layers[0] + 0.27 * layers[1] + 0.18 * layers[2] + 0.07 * layers[3]
        yy, xx = np.mgrid[0:size, 0:size]
        out += 0.12 * np.sin(xx / 55.0 + np.sin(yy / 83.0))
        out -= out.min()
        out /= max(float(out.max()), 1e-6)
        return out

    @staticmethod
    def make_dust_texture(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        h, w = 640, 1280
        base = rng.random((h, w), dtype=np.float32)
        image = Image.fromarray(np.uint8(base * 255))
        broad = np.asarray(image.filter(ImageFilter.GaussianBlur(55)), dtype=np.float32) / 255.0
        medium = np.asarray(image.filter(ImageFilter.GaussianBlur(18)), dtype=np.float32) / 255.0
        fine = np.asarray(image.filter(ImageFilter.GaussianBlur(5)), dtype=np.float32) / 255.0
        out = 0.55 * broad + 0.31 * medium + 0.14 * fine
        out -= out.min()
        out /= max(float(out.max()), 1e-6)
        return out

    @staticmethod
    def make_rover_silhouette() -> Image.Image:
        rw, rh = int(520 * SCALE), int(330 * SCALE)
        img = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        dark = (18, 16, 14, 255)
        metal = (40, 34, 28, 255)
        # body and deck
        d.rounded_rectangle((rw*.25, rh*.36, rw*.72, rh*.62), radius=max(2, int(12*SCALE)), fill=dark)
        d.polygon([(rw*.22,rh*.39),(rw*.32,rh*.28),(rw*.67,rh*.29),(rw*.77,rh*.42)], fill=metal)
        # mast
        d.rectangle((rw*.49,rh*.12,rw*.53,rh*.36), fill=dark)
        d.rounded_rectangle((rw*.42,rh*.05,rw*.62,rh*.15), radius=max(2,int(8*SCALE)), fill=metal)
        d.ellipse((rw*.58,rh*.075,rw*.61,rh*.105), fill=(5,5,5,255))
        # arm
        d.line((rw*.67,rh*.47,rw*.84,rh*.64), fill=dark, width=max(2,int(10*SCALE)))
        d.line((rw*.84,rh*.64,rw*.91,rh*.59), fill=dark, width=max(2,int(8*SCALE)))
        # suspension
        d.line((rw*.31,rh*.56,rw*.18,rh*.73), fill=dark, width=max(2,int(8*SCALE)))
        d.line((rw*.66,rh*.56,rw*.80,rh*.73), fill=dark, width=max(2,int(8*SCALE)))
        # six wheels
        for cx, cy, rr in [(rw*.17,rh*.74,rh*.12),(rw*.34,rh*.77,rh*.11),(rw*.48,rh*.77,rh*.11),(rw*.64,rh*.77,rh*.11),(rw*.80,rh*.74,rh*.12)]:
            d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr), fill=(11,11,10,255), outline=(58,48,37,255), width=max(1,int(4*SCALE)))
            d.ellipse((cx-rr*.42,cy-rr*.42,cx+rr*.42,cy+rr*.42), outline=(90,72,51,210), width=max(1,int(2*SCALE)))
        return img.filter(ImageFilter.GaussianBlur(max(0, int(.35*SCALE))))

    def base(self, t: float) -> Image.Image:
        image = Image.new("RGBA", SIZE, COLORS["deep"] + (255,))
        d = ImageDraw.Draw(image)
        for star in self.stars:
            twinkle = 0.74 + 0.26 * math.sin(t * 1.25 + star.phase)
            alpha = int(star.alpha * twinkle)
            r = star.radius
            d.ellipse((star.x-r, star.y-r, star.x+r, star.y+r), fill=COLORS["white"] + (alpha,))
        return image

    def planet(self, image: Image.Image, t: float, dust: float, scale: float = 1.0, y_frac: float = 0.43) -> None:
        radius = int(390 * SCALE * scale)
        diameter = radius * 2
        yy, xx = np.mgrid[0:diameter, 0:diameter]
        nx = (xx - radius + .5) / radius
        ny = (yy - radius + .5) / radius
        rr = nx*nx + ny*ny
        mask = rr <= 1.0
        z = np.sqrt(np.clip(1.0 - rr, 0.0, 1.0))
        # moving texture coordinates simulate rotation
        tex = self.planet_texture
        th, tw = tex.shape
        tx = ((np.arctan2(nx, z) / (2*math.pi) + .5 + t*.012) * tw).astype(int) % tw
        ty = np.clip(((np.arcsin(np.clip(-ny,-1,1))/math.pi + .5) * th).astype(int), 0, th-1)
        terrain = tex[ty, tx]
        light = np.clip(.22 + .86*z + .24*nx, 0, 1)
        base_r = 88 + terrain*125
        base_g = 28 + terrain*55
        base_b = 18 + terrain*32
        # dust veil suppresses contrast and pushes toward ochre
        avg = (base_r + base_g + base_b) / 3
        base_r = lerp(base_r, 205 + avg*.08, dust*.72)
        base_g = lerp(base_g, 103 + avg*.04, dust*.72)
        base_b = lerp(base_b, 49 + avg*.02, dust*.72)
        rim = np.clip((1-z)**2.1, 0, 1)
        rgba = np.zeros((diameter,diameter,4),dtype=np.uint8)
        rgba[...,0] = np.clip(base_r*light + rim*90,0,255)
        rgba[...,1] = np.clip(base_g*light + rim*50,0,255)
        rgba[...,2] = np.clip(base_b*light + rim*24,0,255)
        rgba[...,3] = np.where(mask,255,0)
        globe = Image.fromarray(rgba,"RGBA")
        # moving dust plumes inside the disc
        cloud = Image.new("RGBA", (diameter,diameter), (0,0,0,0))
        cd = ImageDraw.Draw(cloud)
        if dust > .02:
            rng = np.random.default_rng(71)
            count = 24 if QUICK else 54
            for i in range(count):
                phase = i/count*2*math.pi + t*.09
                cx = radius + math.cos(phase*1.7+i)*radius*rng.uniform(.05,.70)
                cy = radius + math.sin(phase+i*.3)*radius*rng.uniform(.05,.66)
                rx = radius*rng.uniform(.10,.28)
                ry = radius*rng.uniform(.025,.08)
                alpha = int((18 + rng.uniform(0,24))*dust)
                cd.ellipse((cx-rx,cy-ry,cx+rx,cy+ry), fill=COLORS["dust_light"]+(alpha,))
            cloud = cloud.filter(ImageFilter.GaussianBlur(max(2,int(26*SCALE))))
            circular = Image.new("L",(diameter,diameter),0)
            ImageDraw.Draw(circular).ellipse((0,0,diameter-1,diameter-1),fill=255)
            cloud.putalpha(Image.fromarray(np.minimum(np.asarray(cloud.getchannel("A")),np.asarray(circular)).astype(np.uint8)))
            globe.alpha_composite(cloud)
        # atmospheric halo
        halo = Image.new("RGBA", (diameter+80,diameter+80), (0,0,0,0))
        hd = ImageDraw.Draw(halo)
        hd.ellipse((40,40,diameter+40,diameter+40), outline=COLORS["orange"]+(100+int(70*dust),), width=max(2,int(7*SCALE)))
        halo = halo.filter(ImageFilter.GaussianBlur(max(4,int(16*SCALE))))
        cx, cy = W//2, int(H*y_frac)
        image.alpha_composite(halo,(cx-radius-40,cy-radius-40))
        image.alpha_composite(globe,(cx-radius,cy-radius))

    def terrain_scene(self, image: Image.Image, t: float, dust: float, darkness: float = 0.0, rover: bool = True) -> None:
        horizon = int(H*.43)
        d = ImageDraw.Draw(image)
        # Sky: dust makes it amber but darker near Opportunity at peak.
        for y in range(horizon):
            p = y/max(horizon-1,1)
            r = lerp(7, 78+80*dust, p)
            g = lerp(8, 28+42*dust, p)
            b = lerp(13, 17+11*dust, p)
            dim = 1.0 - darkness*.72
            d.line((0,y,W,y),fill=(int(r*dim),int(g*dim),int(b*dim),255))
        # small dim sun
        sun_x, sun_y = int(W*.78), int(H*.25)
        sun_r = int(24*SCALE*(1-.45*dust))
        glow = Image.new("RGBA",SIZE,(0,0,0,0)); gd=ImageDraw.Draw(glow)
        for mult,alpha in ((6,8),(3,16),(1,115)):
            rr=max(1,int(sun_r*mult))
            gd.ellipse((sun_x-rr,sun_y-rr,sun_x+rr,sun_y+rr),fill=COLORS["gold"]+(int(alpha*(1-darkness*.82)),))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(3,int(18*SCALE)))))
        # Perspective terrain strips.
        terrain = self.terrain
        rows = terrain.shape[0]
        for py in range(horizon,H):
            q=(py-horizon)/max(H-horizon-1,1)
            src_y=min(rows-1,int((q**1.75)*(rows-1)))
            row=terrain[src_y]
            sample=np.interp(np.linspace(0,len(row)-1,W),np.arange(len(row)),row)
            ridge=np.clip(sample + .10*math.sin(t*.12+q*7),0,1)
            rr=lerp(48,142,q)+ridge*64
            gg=lerp(24,64,q)+ridge*29
            bb=lerp(16,35,q)+ridge*15
            dim=1-darkness*.58
            rgb=np.stack([rr*dim,gg*dim,bb*dim],axis=1).clip(0,255).astype(np.uint8)
            Image.fromarray(rgb.reshape(1,W,3),"RGB").convert("RGBA")
            d.line([(x,py) for x in range(W)],fill=None)
            # Pillow cannot draw varying-color line, paste one-row strip.
            image.paste(Image.fromarray(rgb.reshape(1,W,3),"RGB"),(0,py))
        # far ridges
        ridge_layer=Image.new("RGBA",SIZE,(0,0,0,0)); rd=ImageDraw.Draw(ridge_layer)
        pts=[(0,horizon+int(40*SCALE))]
        rng=np.random.default_rng(44)
        for x in np.linspace(0,W,34):
            yy=horizon-int((35+65*rng.random()+34*math.sin(x/W*8))*SCALE)
            pts.append((int(x),int(yy)))
        pts.extend([(W,horizon+int(90*SCALE)),(0,horizon+int(90*SCALE))])
        rd.polygon(pts,fill=(50,24,16,255))
        image.alpha_composite(ridge_layer)
        # airborne dust streaks
        dust_layer=Image.new("RGBA",SIZE,(0,0,0,0)); dd=ImageDraw.Draw(dust_layer)
        rng=np.random.default_rng(2018)
        count=int((35 if QUICK else 105)*dust)
        for i in range(count):
            x=(rng.uniform(-W*.2,W)+t*(26+i%9)*SCALE)%(W*1.2)-W*.1
            y=rng.uniform(H*.20,H*.86)
            length=rng.uniform(25,150)*SCALE
            dd.line((x,y,x+length,y-rng.uniform(0,12)*SCALE),fill=COLORS["dust_light"]+(int(rng.uniform(7,40)*dust),),width=max(1,int(rng.uniform(1,4)*SCALE)))
        image.alpha_composite(dust_layer.filter(ImageFilter.GaussianBlur(max(1,int(2*SCALE)))))
        if rover:
            rw,rh=self.rover.size
            image.alpha_composite(self.rover,(W//2-rw//2,int(H*.69)-rh//2))

    def date_card(self, image: Image.Image, title: str, subtitle: str, date_text: str, alpha: float = 1.0) -> None:
        a=int(245*clamp(alpha))
        panel(image,(int(W*.10),int(H*.18),int(W*.90),int(H*.35)),int(170*alpha),22)
        text(image,date_text,(W//2,int(H*.215)),16,COLORS["gold"]+(a,),True,"ma",1)
        text(image,title,(W//2,int(H*.264)),31,COLORS["white"]+(a,),True,"ma",1)
        text(image,subtitle,(W//2,int(H*.318)),14,COLORS["muted"]+(int(225*alpha),),False,"ma",1)

    def intro(self,image:Image.Image,t:float,local:float) -> None:
        self.planet(image,t,dust=.03+.05*local,scale=1.02,y_frac=.49)
        # Title only for first four seconds, then it disappears.
        shot_start=SHOT_PLAN[0][1]
        elapsed=t-shot_start
        if elapsed < (4.0 if not QUICK else 1.0):
            alpha=smoothstep(min(elapsed/.45,1))*smoothstep(min(((4.0 if not QUICK else 1.0)-elapsed)/.45,1))
            text(image,"DUST STORMS",(W//2,int(H*.13)),55,COLORS["white"]+(int(250*alpha),),True,"ma",2)
            text(image,"COVERED MARS",(W//2,int(H*.195)),55,COLORS["orange"]+(int(250*alpha),),True,"ma",2)
            text(image,"THE 2018 PLANET-ENCIRCLING EVENT",(W//2,int(H*.245)),15,COLORS["muted"]+(int(220*alpha),),True,"ma",1)

    def growth(self,image:Image.Image,t:float,local:float) -> None:
        dust=.08+.48*local
        self.planet(image,t,dust=dust,scale=1.02,y_frac=.50)
        # A single orbit-like progress rail, not a graph.
        d=ImageDraw.Draw(image)
        cx,cy=W//2,int(H*.50); r=int(420*SCALE)
        d.arc((cx-r,cy-r,cx+r,cy+r),195,345,fill=(255,190,90,70),width=max(1,int(3*SCALE)))
        head=195+150*local
        hx=cx+r*math.cos(math.radians(head)); hy=cy+r*math.sin(math.radians(head))
        rr=int(8*SCALE)
        d.ellipse((hx-rr,hy-rr,hx+rr,hy+rr),fill=COLORS["gold"]+(245,))
        # Date card is deliberately visible only for 3.3 seconds.
        local_t=t-SHOT_PLAN[1][1]
        if local_t<3.3*(DURATION/52 if QUICK else 1):
            duration=3.3*(DURATION/52 if QUICK else 1)
            alpha=smoothstep(min(local_t/.35,1))*smoothstep(min((duration-local_t)/.35,1))
            self.date_card(image,"ONE QUARTER OF MARS","The storm expanded in only thirteen days","MAY 30 → JUNE 12, 2018",alpha)

    def opacity(self,image:Image.Image,t:float,local:float) -> None:
        darkness=.18+.72*local
        self.terrain_scene(image,t,dust=.68,darkness=darkness,rover=True)
        # Sunlight meter is a physical metaphor, not a chart.
        x0,x1=int(W*.16),int(W*.84); y=int(H*.20)
        d=ImageDraw.Draw(image)
        d.rounded_rectangle((x0,y,x1,y+int(20*SCALE)),radius=int(10*SCALE),fill=(255,255,255,30))
        remaining=max(.02,1-local*.95)
        d.rounded_rectangle((x0,y,int(lerp(x0,x1,remaining)),y+int(20*SCALE)),radius=int(10*SCALE),fill=COLORS["gold"]+(210,))
        text(image,"DIRECT SUNLIGHT",(x0,y-int(18*SCALE)),12,COLORS["muted"]+(210,),True,"la",1)
        local_t=t-SHOT_PLAN[2][1]
        duration=3.6*(DURATION/52 if QUICK else 1)
        if local_t<duration:
            alpha=smoothstep(min(local_t/.35,1))*smoothstep(min((duration-local_t)/.35,1))
            self.date_card(image,"TAU 10.8","Typical local opacity was about 0.5","OPPORTUNITY • JUNE 10",alpha)

    def silence(self,image:Image.Image,t:float,local:float) -> None:
        self.terrain_scene(image,t,dust=.95,darkness=.85,rover=True)
        # Radio pulse collapses and disappears.
        layer=Image.new("RGBA",SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        cx,cy=W//2,int(H*.62)
        pulse=max(0.0,1-local*1.25)
        for k in range(4):
            rr=int((70+k*45+local*180)*SCALE)
            d.arc((cx-rr,cy-rr,cx+rr,cy+rr),205,335,fill=COLORS["cyan"]+(int(100*pulse/(k+1)),),width=max(1,int(3*SCALE)))
        image.alpha_composite(layer)
        local_t=t-SHOT_PLAN[3][1]
        duration=3.5*(DURATION/52 if QUICK else 1)
        if local_t<duration:
            alpha=smoothstep(min(local_t/.35,1))*smoothstep(min((duration-local_t)/.35,1))
            self.date_card(image,"FINAL SIGNAL","Solar power fell as the sky darkened","JUNE 10, 2018",alpha)

    def global_event(self,image:Image.Image,t:float,local:float) -> None:
        self.planet(image,t,dust=.83+.15*local,scale=1.08,y_frac=.50)
        # Orbital scanning rings, sparse and cinematic.
        layer=Image.new("RGBA",SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        cx,cy=W//2,int(H*.50)
        for i in range(3):
            rx=int((430+i*34)*SCALE); ry=int((132+i*12)*SCALE)
            start=(t*28+i*95)%360
            d.arc((cx-rx,cy-ry,cx+rx,cy+ry),start,start+135,fill=COLORS["cyan"]+(75-i*15,),width=max(1,int(2*SCALE)))
        image.alpha_composite(layer)
        local_t=t-SHOT_PLAN[4][1]
        duration=3.6*(DURATION/52 if QUICK else 1)
        if local_t<duration:
            alpha=smoothstep(min(local_t/.35,1))*smoothstep(min((duration-local_t)/.35,1))
            self.date_card(image,"PLANET-ENCIRCLING","Not literally every square metre — but a global-scale haze","JUNE 19, 2018",alpha)

    def orbiters(self,image:Image.Image,t:float,local:float) -> None:
        self.planet(image,t,dust=.92-.18*local,scale=.88,y_frac=.49)
        # Simple MRO silhouette and scanning fan.
        layer=Image.new("RGBA",SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        sx=int(lerp(W*.12,W*.80,local)); sy=int(H*.24+math.sin(local*math.pi)*H*.05)
        d.rectangle((sx-int(28*SCALE),sy-int(10*SCALE),sx+int(28*SCALE),sy+int(10*SCALE)),fill=(205,205,198,230))
        d.rectangle((sx-int(92*SCALE),sy-int(24*SCALE),sx-int(32*SCALE),sy+int(24*SCALE)),fill=COLORS["blue"]+(220,))
        d.rectangle((sx+int(32*SCALE),sy-int(24*SCALE),sx+int(92*SCALE),sy+int(24*SCALE)),fill=COLORS["blue"]+(220,))
        d.line((sx,sy+int(10*SCALE),W//2,int(H*.50)),fill=COLORS["cyan"]+(60,),width=max(1,int(2*SCALE)))
        image.alpha_composite(layer)
        local_t=t-SHOT_PLAN[5][1]
        duration=3.4*(DURATION/52 if QUICK else 1)
        if local_t<duration:
            alpha=smoothstep(min(local_t/.35,1))*smoothstep(min((duration-local_t)/.35,1))
            self.date_card(image,"MARS UNDER WATCH","MRO, Odyssey, MAVEN and Curiosity studied the storm","JUNE → SEPTEMBER 2018",alpha)

    def finale(self,image:Image.Image,t:float,local:float) -> None:
        self.terrain_scene(image,t,dust=.48*(1-local),darkness=.25*(1-local),rover=True)
        # Long silent visual beat; only the final source line appears briefly.
        local_t=t-SHOT_PLAN[6][1]
        if local_t<3.6*(DURATION/52 if QUICK else 1):
            duration=3.6*(DURATION/52 if QUICK else 1)
            alpha=smoothstep(min(local_t/.35,1))*smoothstep(min((duration-local_t)/.35,1))
            self.date_card(image,"THE SKY RETURNED","Opportunity remained silent after nearly fifteen years of exploration","SEPTEMBER 2018",alpha)
        # Minimal final mark in last 1.5 seconds.
        remaining=SHOT_PLAN[6][2]-t
        if remaining<1.7*(DURATION/52 if QUICK else 1):
            fade=smoothstep(clamp((1.7*(DURATION/52 if QUICK else 1)-remaining)/.45))
            text(image,"NASA/JPL • 2018 MARS DUST EVENT",(W//2,int(H*.84)),12,COLORS["muted"]+(int(220*fade),),True,"ma",1)

    def hud(self,image:Image.Image,t:float) -> None:
        d=ImageDraw.Draw(image)
        m=int(30*SCALE); l=int(42*SCALE)
        for x,y,sx,sy in ((m,m,1,1),(W-m,m,-1,1),(m,H-m,1,-1),(W-m,H-m,-1,-1)):
            d.line((x,y,x+sx*l,y),fill=COLORS["gold"]+(65,),width=max(1,int(SCALE)))
            d.line((x,y,x,y+sy*l),fill=COLORS["gold"]+(65,),width=max(1,int(SCALE)))
        cap=caption_at(t)
        if cap:
            value,alpha=cap
            panel(image,(int(W*.07),int(H*.865),int(W*.93),int(H*.955)),int(150*alpha),18)
            wrapped(image,value,(int(W*.11),int(H*.875),int(W*.89),int(H*.944)),15,COLORS["white"]+(int(245*alpha),),False,4,"center")

    def render(self,t:float) -> np.ndarray:
        image=self.base(t)
        name,a,b=shot_at(t)
        local=smoothstep((t-a)/max(b-a,1e-9))
        if name=="intro": self.intro(image,t,local)
        elif name=="growth": self.growth(image,t,local)
        elif name=="opacity": self.opacity(image,t,local)
        elif name=="silence": self.silence(image,t,local)
        elif name=="global": self.global_event(image,t,local)
        elif name=="orbiters": self.orbiters(image,t,local)
        else: self.finale(image,t,local)
        self.hud(image,t)
        arr=np.asarray(image.convert("RGB"),dtype=np.float32)
        arr*=VIGNETTE[...,None]
        arr=np.clip(arr,0,255).astype(np.uint8)
        graded=ImageEnhance.Contrast(Image.fromarray(arr)).enhance(1.08)
        graded=ImageEnhance.Color(graded).enhance(1.06)
        return np.asarray(graded)


# -----------------------------------------------------------------------------
# Original procedural soundtrack
# -----------------------------------------------------------------------------


def envelope(n:int,attack:float=.03,release:float=.12) -> np.ndarray:
    x=np.ones(n,np.float32)
    a=max(1,int(n*attack)); r=max(1,int(n*release))
    x[:a]=np.linspace(0,1,a); x[-r:]=np.linspace(1,0,r)
    return x


def add_tone(track:np.ndarray,sr:int,start:float,dur:float,freq:float,amp:float,pan:float=0.0,kind:str="sine") -> None:
    i0=max(0,int(start*sr)); n=min(len(track)-i0,int(dur*sr))
    if n<=0:return
    tt=np.arange(n,dtype=np.float32)/sr
    if kind=="triangle":
        wavef=(2/np.pi)*np.arcsin(np.sin(2*np.pi*freq*tt))
    else:
        wavef=np.sin(2*np.pi*freq*tt)
    wavef*=envelope(n)
    left=math.sqrt((1-pan)/2); right=math.sqrt((1+pan)/2)
    track[i0:i0+n,0]+=wavef*amp*left; track[i0:i0+n,1]+=wavef*amp*right


def add_noise(track:np.ndarray,sr:int,start:float,dur:float,amp:float,pan:float=0.0,seed:int=0,lowpass:int=70) -> None:
    i0=max(0,int(start*sr)); n=min(len(track)-i0,int(dur*sr))
    if n<=0:return
    rng=np.random.default_rng(seed)
    noise=rng.normal(0,1,n).astype(np.float32)
    kernel=np.ones(max(2,lowpass),np.float32)/max(2,lowpass)
    noise=np.convolve(noise,kernel,mode="same")
    noise/=max(float(np.max(np.abs(noise))),1e-6); noise*=envelope(n,.08,.20)
    left=math.sqrt((1-pan)/2); right=math.sqrt((1+pan)/2)
    track[i0:i0+n,0]+=noise*amp*left; track[i0:i0+n,1]+=noise*amp*right


def add_whoosh(track:np.ndarray,sr:int,start:float,dur:float,amp:float,seed:int) -> None:
    i0=max(0,int(start*sr)); n=min(len(track)-i0,int(dur*sr))
    if n<=0:return
    rng=np.random.default_rng(seed)
    noise=rng.normal(0,1,n).astype(np.float32)
    # Rising intensity and narrowing moving average creates a dust-front sweep.
    out=np.zeros(n,np.float32)
    chunks=24
    for c in range(chunks):
        a=int(c*n/chunks); b=int((c+1)*n/chunks)
        k=max(2,int(130-(c/chunks)*118))
        kernel=np.ones(k,np.float32)/k
        part=np.convolve(noise[a:b],kernel,mode="same")
        out[a:b]=part
    out/=max(float(np.max(np.abs(out))),1e-6)
    out*=np.sin(np.linspace(0,math.pi,n))**1.4
    track[i0:i0+n,0]+=out*amp*.72
    track[i0:i0+n,1]+=out*amp


def make_soundtrack(path:Path) -> Path:
    sr=44100; n=int(DURATION*sr); track=np.zeros((n,2),np.float32)
    # Low cinematic bed.
    for f,a,p in ((38,.12,-.18),(57,.085,.15),(76,.045,0)):
        add_tone(track,sr,0,DURATION,f,a,p)
    # Constant Martian wind texture.
    add_noise(track,sr,0,DURATION,.058,0,2018,180)
    add_noise(track,sr,0,DURATION,.032,-.3,2019,30)
    # Storm builds progressively.
    add_whoosh(track,sr,SHOT_PLAN[1][1],SHOT_PLAN[4][2]-SHOT_PLAN[1][1],.12,77)
    # Scene impacts.
    for idx,(_,start,_) in enumerate(SHOT_PLAN[1:],1):
        add_tone(track,sr,start,.48,48+idx*7,.14,0,"triangle")
        add_noise(track,sr,max(0,start-.28),.58,.10,.45 if idx%2 else -.45,330+idx,9)
    # Radio pings that stop during Opportunity silence.
    silence_start=next(s[1] for s in SHOT_PLAN if s[0]=="silence")
    tt=2.0*(DURATION/52 if QUICK else 1)
    step=2.25*(DURATION/52 if QUICK else 1)
    while tt<silence_start:
        add_tone(track,sr,tt,.12*(DURATION/52 if QUICK else 1),690,.045,.35,"sine")
        add_tone(track,sr,tt+.035*(DURATION/52 if QUICK else 1),.10*(DURATION/52 if QUICK else 1),920,.028,-.35,"sine")
        tt+=step
    # Final restrained chord.
    finale=SHOT_PLAN[-1][1]
    for i,f in enumerate((110,165,220,330)):
        add_tone(track,sr,finale+i*.08*(DURATION/52 if QUICK else 1),DURATION-finale-.1*i,f,.035,(-.5+i/3))
    peak=max(float(np.max(np.abs(track))),1e-6)
    track=np.tanh(track/(peak*.76))*0.88
    pcm=np.int16(np.clip(track,-1,1)*32767)
    with wave.open(str(path),"wb") as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(sr); wav.writeframes(pcm.tobytes())
    return path


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------


def mux_audio(video:Path,audio:Path,output:Path) -> Path:
    cmd=["ffmpeg","-y","-loglevel","error","-i",str(video),"-i",str(audio),"-c:v","copy","-c:a","aac","-b:a","192k","-shortest","-movflags","+faststart",str(output)]
    subprocess.run(cmd,check=True)
    return output


def contact_sheet(video:Path,path:Path) -> Path:
    reader=iio.get_reader(video)
    frames=[]
    total_frames=max(1,int(DURATION*FPS))
    for p in np.linspace(.05,.94,6):
        index=min(total_frames-1,int(p*total_frames))
        frame=reader.get_data(index)
        frames.append(Image.fromarray(frame).resize((270,480),Image.Resampling.LANCZOS))
    reader.close()
    sheet=Image.new("RGB",(270*3,480*2),(8,8,10))
    for i,frame in enumerate(frames):
        sheet.paste(frame,((i%3)*270,(i//3)*480))
    sheet.save(path,quality=92)
    return path



