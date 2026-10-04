from __future__ import annotations

"""
What If Earth Orbited Two Suns?
===============================

A cinematic vertical YouTube Short renderer about a physically plausible
version of a "two suns" Earth: a circumbinary planet orbiting around both
members of a close binary star system.

Scientific framing
------------------
- Circumbinary planets are real. NASA's Kepler mission confirmed Kepler-16b,
  the first planet known to definitively orbit two stars.
- This video does NOT turn our present Sun into two stars. It uses a clearly
  labeled hypothetical binary designed to make the orbital geometry easy to
  understand.
- Illustrative binary used here:
      primary star:      1.00 solar mass, 1.00 solar luminosity
      secondary star:    0.30 solar mass, 0.01 solar luminosity
      star separation:   0.15 AU, circular
      Earth orbit:       1.00 AU around the pair's barycenter, circular
- In that simplified setup the two stars orbit each other every ~18.61 days.
  Earth completes one circumbinary orbit in ~320.35 days because the total
  central mass is 1.30 solar masses.
- The stars move around their shared center of mass. The Sun-like primary moves
  about 0.0346 AU from the barycenter; the red dwarf moves about 0.1154 AU.
- Because the bright primary shifts back and forth relative to Earth, the total
  received flux in this toy system varies roughly from 0.95 to 1.08 times the
  modern solar constant even though Earth's barycentric radius stays at 1 AU.
  Exact climate response is not modeled.
- "Two suns" does not automatically mean an uninhabitable planet. NASA has
  discussed habitable zones around binary systems and modeled Earth-sized
  circumbinary worlds that could retain liquid water under suitable conditions.
- All orbital diagrams and sky views are schematic, not an exact simulation of
  any observed exoplanet system.

Primary references
------------------
NASA/JPL — Kepler-16b, first confirmed circumbinary planet:
    https://www.jpl.nasa.gov/news/nasas-kepler-discovery-confirms-first-planet-orbiting-two-stars/
NASA Science — Kepler-16b overview:
    https://science.nasa.gov/exoplanets/other-stars-other-worlds/kepler-16-b-almost-a-real-life-tatooine/
NASA Science — Earth-sized "Tatooine" planets could be habitable:
    https://science.nasa.gov/universe/exoplanets/earth-sized-tatooine-planets-could-be-habitable/
NASA — Habitable zone of two suns / Kepler-47:
    https://www.nasa.gov/image-article/orbiting-habitable-zone-of-two-suns/

Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    TWO_SUNS_SHORT_QUICK=1 python what_if_earth_orbited_two_suns.py

Full 1080x1920 render
---------------------
    python what_if_earth_orbited_two_suns.py

4K vertical render
------------------
    TWO_SUNS_SHORT_4K=1 python what_if_earth_orbited_two_suns.py
"""

import csv
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# =============================================================================
# Configuration
# =============================================================================

QUICK_MODE = os.environ.get("TWO_SUNS_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("TWO_SUNS_SHORT_4K", "0") == "1" and not QUICK_MODE

W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 6 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SIZE = (W, H)
SCALE = W / 1080.0

OUTPUT_ROOT = Path("what_if_earth_orbited_two_suns_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "WHAT IF EARTH ORBITED TWO SUNS?",
    "subtitle": "A circumbinary Earth under a constantly changing double sky",
    "basename": "what_if_earth_orbited_two_suns",
    "contrast": 1.10,
    "saturation": 1.07,
    "vignette": 0.27,
}

COLORS = {
    "space": (2, 5, 16),
    "space2": (8, 18, 38),
    "white": (248, 251, 255),
    "muted": (166, 193, 216),
    "cyan": (78, 227, 255),
    "blue": (74, 139, 255),
    "earth": (54, 155, 235),
    "land": (98, 206, 143),
    "primary": (255, 223, 111),
    "primary_hot": (255, 249, 214),
    "secondary": (255, 116, 77),
    "secondary_hot": (255, 184, 144),
    "gold": (255, 204, 86),
    "orange": (255, 144, 71),
    "red": (255, 84, 103),
    "green": (106, 239, 176),
    "violet": (184, 129, 255),
    "panel": (3, 9, 22),
}

SHOT_PLAN = [
    {"name": "double_sun_reveal", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "circumbinary", "start": 8.0 if not QUICK_MODE else 1.8, "end": 19.0 if not QUICK_MODE else 4.25},
    {"name": "two_suns_sky", "start": 19.0 if not QUICK_MODE else 4.25, "end": 30.0 if not QUICK_MODE else 6.7},
    {"name": "changing_light", "start": 30.0 if not QUICK_MODE else 6.7, "end": 41.0 if not QUICK_MODE else 9.15},
    {"name": "climate", "start": 41.0 if not QUICK_MODE else 9.15, "end": 51.0 if not QUICK_MODE else 11.35},
    {"name": "outro", "start": 51.0 if not QUICK_MODE else 11.35, "end": DURATION},
]

CAPTION_TEXTS = [
    "Two suns over Earth is not just science fiction. Astronomers have discovered real circumbinary planets — worlds that orbit around two stars instead of one.",
    "For this thought experiment, put a Sun-like star and a small red dwarf in a tight binary, then place Earth on a wide orbit around their shared center of mass. The stars orbit each other while Earth circles both.",
    "From the ground, the two suns would move relative to each other in the sky. Some days they would appear close together; at other times they would separate, producing double shadows and spectacular paired sunsets.",
    "Even with Earth's distance from the barycenter held near one astronomical unit, the bright star moves back and forth. In this illustrative setup the total incoming light varies by roughly fourteen percent from minimum to maximum.",
    "That would complicate the seasons and climate, but two stars do not automatically make a planet uninhabitable. Circumbinary habitable zones can exist, and climate depends on the stars, orbit, atmosphere, oceans, and axial tilt.",
    "So an Earth under two suns could be dynamically stable if it orbited far enough outside a close binary. The sky would be stranger, the year could change, and sunlight would pulse to the rhythm of two moving stars.",
]

CAPTIONS = [
    (
        shot["start"] + min(0.35, 0.07 * (shot["end"] - shot["start"])),
        shot["end"] - min(0.10, 0.035 * (shot["end"] - shot["start"])),
        text,
    )
    for shot, text in zip(SHOT_PLAN, CAPTION_TEXTS)
]


# =============================================================================
# Illustrative binary model
# =============================================================================

M1 = 1.00       # solar masses
M2 = 0.30       # solar masses
L1 = 1.00       # solar luminosities
L2 = 0.01       # solar luminosities
A_BINARY = 0.15 # AU, star-to-star separation
A_EARTH = 1.00  # AU around barycenter
DAYS_PER_YEAR = 365.256

TOTAL_MASS = M1 + M2
A1 = A_BINARY * M2 / TOTAL_MASS
A2 = A_BINARY * M1 / TOTAL_MASS
BINARY_PERIOD_DAYS = math.sqrt(A_BINARY**3 / TOTAL_MASS) * DAYS_PER_YEAR
EARTH_PERIOD_DAYS = math.sqrt(A_EARTH**3 / TOTAL_MASS) * DAYS_PER_YEAR


@dataclass
class TwoSunsSnapshot:
    generated_at_utc: str
    primary_mass_solar: float
    secondary_mass_solar: float
    primary_luminosity_solar: float
    secondary_luminosity_solar: float
    binary_separation_au: float
    primary_barycentric_radius_au: float
    secondary_barycentric_radius_au: float
    binary_period_days: float
    earth_barycentric_radius_au: float
    earth_period_days: float
    flux_min_earth_units: float
    flux_max_earth_units: float
    flux_peak_to_trough_pct: float
    scenario: str
    interpretation: str
    nasa_kepler16_url: str
    nasa_habitable_url: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(x: float) -> float:
    t = clamp(x)
    return t * t * (3.0 - 2.0 * t)


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def local_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - shot["start"]) / max(1e-6, shot["end"] - shot["start"]))


def caption_at(t: float) -> Optional[str]:
    """Return a short, readable page of the active caption.

    Long narration is split across its original time window so only about
    15-20 words are visible at once; the full text remains in the SRT file.
    """
    for start, end, text in CAPTIONS:
        if start <= t < end:
            words = str(text).split()
            if not words:
                return None
            target_words = 18
            page_count = max(1, (len(words) + target_words - 1) // target_words)
            base = len(words) // page_count
            extra = len(words) % page_count
            progress = (t - start) / max(end - start, 1e-9)
            page = min(page_count - 1, max(0, int(progress * page_count)))
            first = page * base + min(page, extra)
            count = base + (1 if page < extra else 0)
            return " ".join(words[first:first + count])
    return None


def binary_state(days: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Barycentric star positions in AU for a circular binary."""
    a = 2.0 * math.pi * days / BINARY_PERIOD_DAYS
    p1 = (A1 * math.cos(a), A1 * math.sin(a))
    p2 = (-A2 * math.cos(a), -A2 * math.sin(a))
    return p1, p2


def earth_state(days: float) -> Tuple[float, float]:
    a = 2.0 * math.pi * days / EARTH_PERIOD_DAYS
    return (A_EARTH * math.cos(a), A_EARTH * math.sin(a))


def total_flux(days: float) -> float:
    e = earth_state(days)
    p1, p2 = binary_state(days)
    d1 = math.hypot(e[0] - p1[0], e[1] - p1[1])
    d2 = math.hypot(e[0] - p2[0], e[1] - p2[1])
    return L1 / (d1 * d1) + L2 / (d2 * d2)


def sample_flux_extremes() -> Tuple[float, float]:
    vals = [total_flux(3.0 * EARTH_PERIOD_DAYS * i / 6000.0) for i in range(6001)]
    return min(vals), max(vals)


def build_snapshot() -> TwoSunsSnapshot:
    fmin, fmax = sample_flux_extremes()
    return TwoSunsSnapshot(
        generated_at_utc=iso_z(utc_now()),
        primary_mass_solar=M1,
        secondary_mass_solar=M2,
        primary_luminosity_solar=L1,
        secondary_luminosity_solar=L2,
        binary_separation_au=A_BINARY,
        primary_barycentric_radius_au=A1,
        secondary_barycentric_radius_au=A2,
        binary_period_days=BINARY_PERIOD_DAYS,
        earth_barycentric_radius_au=A_EARTH,
        earth_period_days=EARTH_PERIOD_DAYS,
        flux_min_earth_units=fmin,
        flux_max_earth_units=fmax,
        flux_peak_to_trough_pct=(fmax / fmin - 1.0) * 100.0,
        scenario="Illustrative circular circumbinary system: 1.0 Msun + 0.3 Msun stars separated by 0.15 AU; Earth at 1.0 AU from the barycenter.",
        interpretation="Educational two-body-plus-test-particle geometry. The sky views and climate graphics are schematic and are not a reconstruction of a specific observed exoplanet.",
        nasa_kepler16_url="https://science.nasa.gov/exoplanets/other-stars-other-worlds/kepler-16-b-almost-a-real-life-tatooine/",
        nasa_habitable_url="https://science.nasa.gov/universe/exoplanets/earth-sized-tatooine-planets-could-be-habitable/",
    )


def save_data(snapshot: TwoSunsSnapshot) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "illustrative_circumbinary_orbit.csv"
    json_path = DATA_ROOT / "two_suns_snapshot.json"
    rows: List[Dict[str, float]] = []
    for i in range(361):
        day = snapshot.earth_period_days * i / 360.0
        e = earth_state(day)
        s1, s2 = binary_state(day)
        d1 = math.hypot(e[0] - s1[0], e[1] - s1[1])
        d2 = math.hypot(e[0] - s2[0], e[1] - s2[1])
        rows.append({
            "sample": i,
            "day": day,
            "earth_x_au": e[0],
            "earth_y_au": e[1],
            "primary_x_au": s1[0],
            "primary_y_au": s1[1],
            "secondary_x_au": s2[0],
            "secondary_y_au": s2[1],
            "primary_distance_au": d1,
            "secondary_distance_au": d2,
            "relative_total_flux": L1 / d1**2 + L2 / d2**2,
        })
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"snapshot": asdict(snapshot), "samples": rows}, indent=2), encoding="utf-8")
    return csv_path, json_path


# =============================================================================
# Typography helpers
# =============================================================================

def get_font(size: int, bold: bool = False):
    px = max(7, int(size * SCALE))
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, px)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(image: Image.Image, value: str, xy: Tuple[int, int], size: int,
              fill=(255, 255, 255, 255), bold: bool = False, anchor: str = "la", stroke: int = 2):
    ImageDraw.Draw(image).text(
        xy, value, font=get_font(size, bold), fill=fill, anchor=anchor,
        stroke_width=max(1, int(stroke * SCALE)), stroke_fill=(0, 0, 0, 220),
    )


def draw_wrapped_text(image: Image.Image, value: str, xy: Tuple[int, int], max_width: int,
                      size: int, fill=(255, 255, 255, 245), bold: bool = False, spacing: int = 6):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold)
    words = value.split()
    lines: List[str] = []
    current = ""
    sw = max(1, int(2 * SCALE))
    for word in words:
        candidate = word if not current else current + " " + word
        box = draw.textbbox((0, 0), candidate, font=font, stroke_width=sw)
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
        draw.text((x, y), line, font=font, fill=fill, stroke_width=sw, stroke_fill=(0, 0, 0, 220))
        box = draw.textbbox((x, y), line, font=font, stroke_width=sw)
        y += (box[3] - box[1]) + int(spacing * SCALE)


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    h = ms // 3_600_000; ms %= 3_600_000
    m = ms // 60_000; ms %= 60_000
    s = ms // 1000; ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path) -> Path:
    lines: List[str] = []
    for i, (start, end, value) in enumerate(CAPTIONS, start=1):
        lines.extend([str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", value, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    rr = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * rr**1.8, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(W, H, float(CONFIG["vignette"]))


# =============================================================================
# Scene renderer
# =============================================================================

class TwoSunsScene:
    def __init__(self, snapshot: TwoSunsSnapshot):
        self.snapshot = snapshot
        rng = np.random.default_rng(16092026)
        self.stars = [
            (float(rng.uniform(0, W)), float(rng.uniform(0, H)), float(rng.uniform(.35, 2.0) * SCALE),
             int(rng.uniform(16, 95)), float(rng.uniform(0, 2 * math.pi)))
            for _ in range(170 if QUICK_MODE else 520)
        ]

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 178):
        layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle(box, radius=max(8, int(24 * SCALE)), fill=COLORS["panel"] + (alpha,),
                            outline=COLORS["cyan"] + (44,), width=max(1, int(2 * SCALE)))
        image.alpha_composite(layer)

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        arr[..., 0] = np.clip(2 + yy * 7, 0, 255)
        arr[..., 1] = np.clip(6 + yy * 16, 0, 255)
        arr[..., 2] = np.clip(18 + yy * 34, 0, 255)
        image = Image.fromarray(arr, "RGB").convert("RGBA")
        d = ImageDraw.Draw(image)
        for x, y, r, a0, phase in self.stars:
            a = int(a0 * (.70 + .30 * math.sin(t * .9 + phase)))
            d.ellipse((x-r, y-r, x+r, y+r), fill=COLORS["white"] + (max(3, a),))
        return image

    def draw_star(self, image: Image.Image, center: Tuple[int, int], radius: int,
                  base: Tuple[int, int, int], hot: Tuple[int, int, int], t: float, phase: float = 0.0):
        cx, cy = center
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for k, alpha in [(2.8, 10), (2.0, 22), (1.45, 42)]:
            rr = int(radius * k)
            gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=base + (alpha,))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4, int(22 * SCALE)))))
        d = ImageDraw.Draw(image)
        d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=base + (255,), outline=hot + (235,), width=max(1, int(3*SCALE)))
        for i in range(7):
            a = i * math.tau / 7 + t * .07 + phase
            rr = radius * (.46 + .12 * math.sin(t * .6 + i + phase))
            x = cx + math.cos(a) * rr; y = cy + math.sin(a) * rr
            q = max(2, int(radius * .09))
            d.ellipse((x-q, y-q, x+q, y+q), fill=hot + (38,))

    def draw_earth(self, image: Image.Image, center: Tuple[int, int], radius: int, t: float = 0.0):
        cx, cy = center
        d = ImageDraw.Draw(image)
        d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=COLORS["earth"] + (255,), outline=COLORS["cyan"] + (180,), width=max(1, int(2*SCALE)))
        lands = [
            [(cx-radius*.55,cy-radius*.20),(cx-radius*.15,cy-radius*.55),(cx+radius*.08,cy-radius*.30),(cx-radius*.05,cy+radius*.02),(cx-radius*.40,cy+radius*.12)],
            [(cx+radius*.10,cy-radius*.50),(cx+radius*.52,cy-radius*.18),(cx+radius*.42,cy+radius*.15),(cx+radius*.12,cy+radius*.45),(cx-radius*.02,cy+radius*.20)],
        ]
        for pts in lands:
            d.polygon(pts, fill=COLORS["land"] + (220,))
        d.arc((cx-radius*.8,cy-radius*.7,cx+radius*.9,cy+radius*.8), 190, 330, fill=COLORS["white"]+(85,), width=max(1,int(3*SCALE)))

    def draw_title(self, image: Image.Image, t: float):
        if t > (6.3 if not QUICK_MODE else 1.42):
            return
        a = int(245 * smoothstep(t / (.8 if not QUICK_MODE else .18)))
        draw_text(image, "WHAT IF EARTH ORBITED", (W//2, int(H*.064)), 35, COLORS["white"]+(a,), True, "ma", 2)
        draw_text(image, "TWO SUNS?", (W//2, int(H*.112)), 52, COLORS["cyan"]+(a,), True, "ma", 2)
        draw_text(image, "A CIRCUMBINARY EARTH", (W//2, int(H*.158)), 17, COLORS["muted"]+(min(a,225),), True, "ma", 1)

    def draw_source_hud(self, image: Image.Image):
        draw_text(image, "HYPOTHETICAL SYSTEM // REAL PHYSICS", (W-int(42*SCALE), int(54*SCALE)), 13,
                  COLORS["cyan"]+(215,), True, "ra", 1)
        draw_text(image, "NASA KEPLER // CIRCUMBINARY PLANETS", (W-int(42*SCALE), int(80*SCALE)), 10,
                  COLORS["muted"]+(190,), False, "ra", 1)

    def draw_caption(self, image: Image.Image, t: float):
        caption = caption_at(t)
        if not caption:
            return
        y0 = H - int(188*SCALE)
        layer = Image.new("RGBA", SIZE, (0,0,0,0)); d = ImageDraw.Draw(layer)
        d.rounded_rectangle((int(58*SCALE), y0, W-int(58*SCALE), y0+int(116*SCALE)), radius=max(8,int(20*SCALE)),
                            fill=(2,7,16,150), outline=COLORS["cyan"]+(48,), width=max(1,int(1*SCALE)))
        image.alpha_composite(layer)
        draw_wrapped_text(image, caption, (int(82*SCALE), y0+int(18*SCALE)), W-int(164*SCALE), 30,
                          COLORS["white"]+(240,), False, 4)

    def draw_double_sun_reveal(self, image: Image.Image, t: float, p: float):
        # horizon silhouette + two stars
        d = ImageDraw.Draw(image)
        horizon = int(H*.64)
        d.rectangle((0,horizon,W,H), fill=(8,19,27,255))
        for i in range(11):
            x = int(i * W / 10); hh = int((22 + 35 * (0.5+0.5*math.sin(i*1.7))) * SCALE)
            d.polygon([(x-int(55*SCALE),horizon),(x,horizon-hh),(x+int(65*SCALE),horizon)], fill=(10,29,34,255))
        sep = int(W * (.08 + .13 * smoothstep(p)))
        cy = int(H*.39)
        self.draw_star(image,(W//2-sep,cy),int(54*SCALE),COLORS["primary"],COLORS["primary_hot"],t)
        self.draw_star(image,(W//2+sep,cy+int(18*SCALE)),int(31*SCALE),COLORS["secondary"],COLORS["secondary_hot"],t,1.2)
        draw_text(image,"DOUBLE SUNSET",(W//2,int(H*.72)),26,COLORS["gold"]+(240,),True,"ma",2)
        draw_text(image,"not science fiction anymore",(W//2,int(H*.765)),15,COLORS["muted"]+(220,),False,"ma",1)
        self.panel(image,(int(W*.18),int(H*.80),int(W*.82),int(H*.87)),155)
        draw_text(image,"CIRCUMBINARY PLANETS ARE OBSERVED",(W//2,int(H*.835)),13,COLORS["cyan"]+(235,),True,"ma",1)

    def draw_circumbinary(self, image: Image.Image, t: float, p: float):
        draw_text(image,"ONE PLANET • TWO STARS • ONE BARYCENTER",(W//2,int(H*.14)),24,COLORS["white"]+(245,),True,"ma",2)
        cx=W//2; cy=int(H*.48); rE=int(W*.38)
        d=ImageDraw.Draw(image)
        d.ellipse((cx-rE,cy-rE,cx+rE,cy+rE),outline=COLORS["cyan"]+(80,),width=max(2,int(3*SCALE)))
        # star orbits
        rb1=int(35*SCALE); rb2=int(116*SCALE)
        d.ellipse((cx-rb1,cy-rb1,cx+rb1,cy+rb1),outline=COLORS["gold"]+(60,),width=max(1,int(2*SCALE)))
        d.ellipse((cx-rb2,cy-rb2,cx+rb2,cy+rb2),outline=COLORS["orange"]+(45,),width=max(1,int(2*SCALE)))
        a=t*2.3
        s1=(int(cx+rb1*math.cos(a)),int(cy+rb1*math.sin(a)))
        s2=(int(cx-rb2*math.cos(a)),int(cy-rb2*math.sin(a)))
        self.draw_star(image,s1,int(31*SCALE),COLORS["primary"],COLORS["primary_hot"],t)
        self.draw_star(image,s2,int(18*SCALE),COLORS["secondary"],COLORS["secondary_hot"],t,1.3)
        ea=-.7+p*math.tau*.55
        e=(int(cx+rE*math.cos(ea)),int(cy+rE*math.sin(ea)))
        self.draw_earth(image,e,int(24*SCALE),t)
        q=max(3,int(5*SCALE)); d.ellipse((cx-q,cy-q,cx+q,cy+q),fill=COLORS["violet"]+(245,))
        draw_text(image,"BARYCENTER",(cx,cy-int(22*SCALE)),11,COLORS["violet"]+(235,),True,"ma",1)
        self.panel(image,(int(W*.12),int(H*.72),int(W*.88),int(H*.83)),170)
        draw_text(image,f"STAR PAIR: {self.snapshot.binary_period_days:.1f}-DAY ORBIT",(W//2,int(H*.755)),15,COLORS["gold"]+(240,),True,"ma",1)
        draw_text(image,f"EARTH YEAR: {self.snapshot.earth_period_days:.1f} DAYS",(W//2,int(H*.795)),16,COLORS["cyan"]+(240,),True,"ma",1)

    def draw_two_suns_sky(self, image: Image.Image, t: float, p: float):
        d=ImageDraw.Draw(image)
        horizon=int(H*.62)
        # twilight gradient bands
        d.rectangle((0,int(H*.22),W,horizon),fill=(23,44,70,255))
        d.rectangle((0,int(H*.50),W,horizon),fill=(90,78,62,255))
        d.rectangle((0,horizon,W,H),fill=(8,18,23,255))
        phase=math.sin(p*math.pi)
        sep=int(W*(.04+.22*phase))
        y=int(H*(.39+.08*p))
        self.draw_star(image,(W//2-sep,y),int(48*SCALE),COLORS["primary"],COLORS["primary_hot"],t)
        self.draw_star(image,(W//2+sep,y+int(25*SCALE)),int(27*SCALE),COLORS["secondary"],COLORS["secondary_hot"],t,1.0)
        # object and two shadows
        x0=W//2; base=horizon-int(4*SCALE)
        d.rectangle((x0-int(14*SCALE),base-int(90*SCALE),x0+int(14*SCALE),base),fill=(28,32,37,255))
        d.ellipse((x0-int(29*SCALE),base-int(122*SCALE),x0+int(29*SCALE),base-int(68*SCALE)),fill=(36,41,46,255))
        shadow_len=int(150*SCALE)
        d.polygon([(x0-int(10*SCALE),base),(x0+int(10*SCALE),base),(x0+shadow_len,base+int(45*SCALE)),(x0+shadow_len-int(25*SCALE),base+int(54*SCALE))],fill=(0,0,0,105))
        d.polygon([(x0-int(8*SCALE),base),(x0+int(8*SCALE),base),(x0-shadow_len,base+int(38*SCALE)),(x0-shadow_len+int(28*SCALE),base+int(50*SCALE))],fill=(25,5,2,70))
        draw_text(image,"TWO MOVING SUNS",(W//2,int(H*.14)),30,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"changing separation • double shadows • paired sunsets",(W//2,int(H*.195)),13,COLORS["muted"]+(220,),False,"ma",1)
        draw_text(image,"SKY VIEW IS SCHEMATIC",(W//2,int(H*.75)),12,COLORS["cyan"]+(220,),True,"ma",1)

    def draw_changing_light(self, image: Image.Image, t: float, p: float):
        draw_text(image,"THE LIGHT WOULD PULSE",(W//2,int(H*.14)),30,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"the bright star shifts around the binary barycenter",(W//2,int(H*.195)),13,COLORS["muted"]+(220,),False,"ma",1)
        x0=int(W*.11); x1=int(W*.89); y0=int(H*.64); y1=int(H*.31)
        d=ImageDraw.Draw(image)
        d.line((x0,y0,x1,y0),fill=COLORS["white"]+(85,),width=max(2,int(3*SCALE)))
        d.line((x0,y0,x0,y1),fill=COLORS["white"]+(85,),width=max(2,int(3*SCALE)))
        pts=[]
        samples=160
        for i in range(samples):
            day=self.snapshot.earth_period_days*i/(samples-1)
            f=total_flux(day)
            xx=lerp(x0,x1,i/(samples-1))
            yy=lerp(y0,y1,clamp((f-.90)/.22))
            pts.append((xx,yy))
        reveal=max(2,int(samples*clamp(p)))
        d.line(pts[:reveal],fill=COLORS["cyan"]+(235,),width=max(2,int(5*SCALE)),joint="curve")
        for f,label,col in [(self.snapshot.flux_min_earth_units,"MIN",COLORS["blue"]),(1.0,"TODAY'S EARTH",COLORS["white"]),(self.snapshot.flux_max_earth_units,"MAX",COLORS["gold"])]:
            yy=lerp(y0,y1,clamp((f-.90)/.22))
            d.line((x0,yy,x1,yy),fill=col+(35,),width=max(1,int(2*SCALE)))
            draw_text(image,f"{label}  {f:.2f}×",(x1-int(4*SCALE),int(yy-int(9*SCALE))),10,col+(220,),True,"ra",1)
        self.panel(image,(int(W*.15),int(H*.71),int(W*.85),int(H*.82)),168)
        draw_text(image,f"≈ {self.snapshot.flux_peak_to_trough_pct:.1f}% PEAK-TO-TROUGH",(W//2,int(H*.75)),22,COLORS["gold"]+(245,),True,"ma",1)
        draw_text(image,"ILLUSTRATIVE SYSTEM — CLIMATE RESPONSE NOT MODELED",(W//2,int(H*.795)),10,COLORS["muted"]+(210,),True,"ma",1)

    def draw_climate(self, image: Image.Image, t: float, p: float):
        draw_text(image,"TWO SUNS ≠ AUTOMATIC DOOM",(W//2,int(H*.14)),29,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"circumbinary habitable zones can exist",(W//2,int(H*.195)),15,COLORS["green"]+(230,),True,"ma",1)
        # central Earth and orbiting light bands
        cx=W//2; cy=int(H*.46); r=int(105*SCALE)
        self.draw_earth(image,(cx,cy),r,t)
        d=ImageDraw.Draw(image)
        for rr,col,a in [(int(160*SCALE),COLORS["green"],70),(int(205*SCALE),COLORS["cyan"],45),(int(250*SCALE),COLORS["violet"],28)]:
            start=int(360*p)-55
            d.arc((cx-rr,cy-rr,cx+rr,cy+rr),start,start+125,fill=col+(a+80,),width=max(2,int(8*SCALE)))
        labels=[
            ("STAR MASSES + SEPARATION",COLORS["gold"]),("PLANET ORBIT",COLORS["cyan"]),("ATMOSPHERE + OCEANS",COLORS["green"]),("AXIAL TILT",COLORS["violet"])
        ]
        y=int(H*.67)
        for i,(lab,col) in enumerate(labels):
            draw_text(image,lab,(W//2,y+i*int(37*SCALE)),13,col+(235,),True,"ma",1)
        draw_text(image,"habitability depends on the whole system",(W//2,int(H*.84)),12,COLORS["muted"]+(215,),False,"ma",1)

    def draw_outro(self, image: Image.Image, t: float, p: float):
        cx=W//2; cy=int(H*.43); rE=int(W*.34)
        d=ImageDraw.Draw(image)
        d.ellipse((cx-rE,cy-rE,cx+rE,cy+rE),outline=COLORS["cyan"]+(65,),width=max(2,int(3*SCALE)))
        a=t*2.0
        self.draw_star(image,(int(cx+34*SCALE*math.cos(a)),int(cy+34*SCALE*math.sin(a))),int(33*SCALE),COLORS["primary"],COLORS["primary_hot"],t)
        self.draw_star(image,(int(cx-112*SCALE*math.cos(a)),int(cy-112*SCALE*math.sin(a))),int(19*SCALE),COLORS["secondary"],COLORS["secondary_hot"],t,1.2)
        ea=-1.0+p*.9
        self.draw_earth(image,(int(cx+rE*math.cos(ea)),int(cy+rE*math.sin(ea))),int(28*SCALE),t)
        draw_text(image,"A STABLE WORLD IS POSSIBLE",(W//2,int(H*.14)),30,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"if Earth orbits well outside the close binary",(W//2,int(H*.195)),14,COLORS["cyan"]+(225,),True,"ma",1)
        self.panel(image,(int(W*.13),int(H*.70),int(W*.87),int(H*.83)),175)
        draw_text(image,"2 SUNS",(int(W*.28),int(H*.755)),23,COLORS["gold"]+(245,),True,"ma",1)
        draw_text(image,"1 BARYCENTER",(W//2,int(H*.755)),18,COLORS["violet"]+(240,),True,"ma",1)
        draw_text(image,"1 EARTH",(int(W*.73),int(H*.755)),23,COLORS["cyan"]+(245,),True,"ma",1)
        draw_text(image,"A STRANGER SKY — NOT AN IMPOSSIBLE ONE",(W//2,int(H*.805)),11,COLORS["muted"]+(220,),True,"ma",1)

    def draw_scanlines(self, image: Image.Image, t: float):
        layer=Image.new("RGBA",SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        step=max(5,int(8*SCALE)); off=int((t*31)%step)
        for y in range(off,H,step):
            d.line((0,y,W,y),fill=(135,185,255,7),width=1)
        image.alpha_composite(layer)

    def render(self, t: float) -> Image.Image:
        image=self.background(t)
        shot=get_shot(t); p=local_progress(t,shot)
        name=shot["name"]
        if name=="double_sun_reveal": self.draw_double_sun_reveal(image,t,p)
        elif name=="circumbinary": self.draw_circumbinary(image,t,p)
        elif name=="two_suns_sky": self.draw_two_suns_sky(image,t,p)
        elif name=="changing_light": self.draw_changing_light(image,t,p)
        elif name=="climate": self.draw_climate(image,t,p)
        else: self.draw_outro(image,t,p)
        self.draw_title(image,t)
        self.draw_source_hud(image)
        self.draw_caption(image,t)
        self.draw_scanlines(image,t)
        arr=np.asarray(image.convert("RGB"),dtype=np.float32)
        arr*=VIGNETTE[...,None]
        arr=np.clip(arr,0,255).astype(np.uint8)
        graded=Image.fromarray(arr,"RGB")
        graded=ImageEnhance.Contrast(graded).enhance(float(CONFIG["contrast"]))
        graded=ImageEnhance.Color(graded).enhance(float(CONFIG["saturation"]))
        return graded


# =============================================================================
# Output
# =============================================================================

def render_preview_frames(scene: TwoSunsScene) -> List[Path]:
    paths=[]
    for i,shot in enumerate(SHOT_PLAN,1):
        t=shot["start"]+.52*(shot["end"]-shot["start"])
        image=scene.render(min(t,DURATION-.001))
        path=PREVIEW_ROOT/f"preview_{i:02d}_{shot['name']}.jpg"
        image.save(path,quality=92)
        paths.append(path)
    return paths


def make_contact_sheet(paths: Sequence[Path]) -> Path:
    images=[Image.open(p).convert("RGB") for p in paths]
    thumb_w=150 if QUICK_MODE else int(300*SCALE)
    thumb_h=267 if QUICK_MODE else int(533*SCALE)
    cols=3; rows=math.ceil(len(images)/cols); margin=max(12,int(18*SCALE))
    sheet=Image.new("RGB",(cols*thumb_w+(cols+1)*margin,rows*thumb_h+(rows+1)*margin),(5,8,16))
    for i,img in enumerate(images):
        thumb=img.copy(); thumb.thumbnail((thumb_w,thumb_h),Image.Resampling.LANCZOS)
        x=margin+(i%cols)*(thumb_w+margin); y=margin+(i//cols)*(thumb_h+margin)
        sheet.paste(thumb,(x,y))
    path=PREVIEW_ROOT/f"{CONFIG['basename']}_contact_sheet.jpg"
    sheet.save(path,quality=92)
    return path


def render_video(scene: TwoSunsScene) -> Path:
    path=OUTPUT_ROOT/f"{CONFIG['basename']}.mp4"
    total=max(1,int(round(DURATION*FPS)))
    writer=iio.get_writer(path,fps=FPS,codec="libx264",quality=7,pixelformat="yuv420p",ffmpeg_log_level="error",macro_block_size=2)
    try:
        for i in tqdm(range(total),desc="Rendering Two-Suns Earth Short"):
            writer.append_data(np.asarray(scene.render(i/FPS).convert("RGB")))
    finally:
        writer.close()
    return path


def copy_flat(video: Path, sheet: Path, srt: Path) -> Tuple[Path,Path,Path]:
    flat_video=Path(f"{CONFIG['basename']}_quick_preview.mp4") if QUICK_MODE else Path(f"{CONFIG['basename']}.mp4")
    flat_sheet=Path(f"{CONFIG['basename']}_contact_sheet.jpg")
    flat_srt=Path(f"{CONFIG['basename']}_subtitles.srt")
    shutil.copy2(video,flat_video); shutil.copy2(sheet,flat_sheet); shutil.copy2(srt,flat_srt)
    return flat_video,flat_sheet,flat_srt



# =============================================================================
# YouTube title / description sidecar
# =============================================================================

YOUTUBE_TITLE = 'What If Earth Orbited TWO Suns? 🌍☀️☀️ #rootjatin'
YOUTUBE_DESCRIPTION = 'Two suns are not purely science fiction: real circumbinary planets orbit pairs of stars. This video uses a hypothetical Earth-like world on a wide orbit around a close binary to visualize double suns, shifting shadows, paired sunsets, and changing illumination. The exact Earth-like system shown is illustrative, while the underlying circumbinary-orbit concept is real astronomy.'
YOUTUBE_HASHTAGS = '#rootjatin #TwoSuns #Circumbinary #Exoplanets #Earth #WhatIf #Astronomy #Space'

def write_youtube_metadata_txt() -> Path:
    cfg = globals().get("CONFIG") or globals().get("CFG") or {}
    basename = (
        cfg.get("basename")
        or cfg.get("output_basename")
        or cfg.get("file_stem")
        or Path(__file__).stem
    )
    root = globals().get("OUTPUT_ROOT") or globals().get("ROOT") or Path(".")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{basename}_title_description.txt"
    path.write_text(
        "TITLE\n" + YOUTUBE_TITLE +
        "\n\nDESCRIPTION\n" + YOUTUBE_DESCRIPTION +
        "\n\nHASHTAGS\n" + YOUTUBE_HASHTAGS + "\n",
        encoding="utf-8",
    )
    return path


