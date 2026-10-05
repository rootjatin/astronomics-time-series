from __future__ import annotations

"""
What Happens If Earth Stops Orbiting?
======================================

A cinematic vertical YouTube Short renderer for a controlled orbital-physics
thought experiment: Earth's heliocentric sideways orbital velocity is
instantaneously removed while the Sun and Earth otherwise remain unchanged.

Scientific framing
------------------
- "Stopping Earth's orbit" is not a natural event. Orbiting is continuous free
  fall with enough sideways velocity to keep missing the Sun.
- Earth moves around the Sun at about 29.78 km/s on average. This renderer uses
  a simplified circular 1-AU starting orbit and instantaneously removes that
  tangential velocity.
- With zero heliocentric tangential speed, Earth begins a radial fall toward the
  Sun under solar gravity. The initial solar acceleration at 1 AU is only about
  0.00593 m/s^2, so the fall starts slowly and then accelerates dramatically.
- In the idealized two-body model, starting from rest at 1 AU:
      distance of Venus's orbit (0.723 AU): ~41.2 days
      distance of Mercury's orbit (0.387 AU): ~57.0 days
      Sun photospheric radius (~695,700 km): ~64.6 days
  These are radial-distance milestones, not collisions with Venus or Mercury.
- The idealized radial speed is about 53 km/s at Mercury's orbital distance and
  about 616 km/s at the solar photosphere. The latter is a mathematical
  two-body value; the real Earth would undergo extreme heating and destructive
  interaction with the solar environment before any intact "impact."
- Solar irradiance rises approximately as 1/r^2 while solar luminosity is held
  fixed. At 0.723 AU it is ~1.9 times today's value; at 0.387 AU it is ~6.7
  times today's value. Exact climate and destruction timelines are not modeled.
- Earth's rotation is not stopped in this setup. The Moon and other planets are
  omitted from the dynamics so the video stays a clean Sun-Earth two-body
  illustration.

Primary references
------------------
NASA NSSDC — Earth Fact Sheet (mean orbital velocity, orbit parameters):
    https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
NASA Science — Earth facts:
    https://science.nasa.gov/earth/facts/
NASA Science — Basics of Space Flight / Solar System distances:
    https://science.nasa.gov/learn/basics-of-space-flight/chapter1-1/
NASA Science — Units of measure / astronomical unit:
    https://science.nasa.gov/learn/basics-of-space-flight/units/
NASA NSSDC — Sun Fact Sheet:
    https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html

Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    EARTH_STOP_ORBIT_QUICK=1 python what_happens_if_earth_stops_orbiting.py

Full 1080x1920 render
---------------------
    python what_happens_if_earth_stops_orbiting.py

4K vertical render
------------------
    EARTH_STOP_ORBIT_4K=1 python what_happens_if_earth_stops_orbiting.py
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

QUICK_MODE = os.environ.get("EARTH_STOP_ORBIT_QUICK", "0") == "1"
FOUR_K = os.environ.get("EARTH_STOP_ORBIT_4K", "0") == "1" and not QUICK_MODE

W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 6 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SIZE = (W, H)
SCALE = W / 1080.0

OUTPUT_ROOT = Path("what_happens_if_earth_stops_orbiting_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "WHAT HAPPENS IF EARTH STOPS ORBITING?",
    "subtitle": "Remove Earth's sideways speed — and gravity finally wins",
    "basename": "what_happens_if_earth_stops_orbiting",
    "contrast": 1.10,
    "saturation": 1.05,
    "vignette": 0.28,
}

COLORS = {
    "space": (2, 5, 15),
    "space2": (9, 17, 34),
    "white": (248, 251, 255),
    "muted": (166, 194, 216),
    "cyan": (78, 226, 255),
    "blue": (72, 142, 255),
    "earth": (52, 151, 231),
    "earth_land": (101, 208, 149),
    "sun": (255, 222, 102),
    "sun_hot": (255, 248, 205),
    "orange": (255, 145, 68),
    "gold": (255, 204, 82),
    "red": (255, 82, 99),
    "green": (108, 239, 176),
    "violet": (181, 126, 255),
    "panel": (3, 9, 21),
}

SHOT_PLAN = [
    {"name": "orbit_is_falling", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "velocity_removed", "start": 8.0 if not QUICK_MODE else 1.8, "end": 18.0 if not QUICK_MODE else 4.0},
    {"name": "fall_clock", "start": 18.0 if not QUICK_MODE else 4.0, "end": 31.0 if not QUICK_MODE else 6.9},
    {"name": "heating", "start": 31.0 if not QUICK_MODE else 6.9, "end": 41.0 if not QUICK_MODE else 9.1},
    {"name": "speed_up", "start": 41.0 if not QUICK_MODE else 9.1, "end": 51.0 if not QUICK_MODE else 11.3},
    {"name": "outro", "start": 51.0 if not QUICK_MODE else 11.3, "end": DURATION},
]

CAPTION_TEXTS = [
    "Earth does not stay up because gravity is weak. It is falling around the Sun at about thirty kilometers per second sideways — fast enough to keep missing it.",
    "Now erase that sideways orbital velocity instantly. Earth's rotation is left alone, but its heliocentric tangential speed becomes zero. Solar gravity points almost straight inward.",
    "The fall starts slowly, then accelerates. In a clean two-body model, Earth reaches the distance of Venus's orbit after about 41 days, Mercury's orbital distance after about 57 days, and the Sun's visible surface after about 64.6 days.",
    "Sunlight would intensify as the inverse square of distance. At Venus's orbital radius the flux is about 1.9 times today's value. At Mercury's radius it is about 6.7 times today's value. Exact climate collapse is not modeled here.",
    "Conservation of energy turns gravitational potential into speed. The idealized Earth is moving about 53 kilometers per second by Mercury's orbital distance, and the mathematical two-body speed near the solar surface is over 600 kilometers per second.",
    "So 'stopping Earth's orbit' does not make Earth hover. Orbit is sideways free fall. Remove roughly 29.8 kilometers per second of sideways motion, and Earth falls inward — reaching the Sun in about sixty-five days in this simplified model.",
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
# Physics model
# =============================================================================

AU_M = 149_597_870_700.0
MU_SUN = 1.32712440018e20
R_SUN_M = 695_700_000.0
EARTH_ORBIT_SPEED_KMS = 29.78
SECONDS_PER_DAY = 86_400.0


@dataclass
class StopOrbitSnapshot:
    generated_at_utc: str
    initial_distance_au: float
    initial_orbital_speed_kms: float
    removed_tangential_speed_kms: float
    initial_solar_acceleration_ms2: float
    venus_radius_au: float
    venus_radius_time_days: float
    venus_radius_speed_kms: float
    venus_radius_flux_multiple: float
    mercury_radius_au: float
    mercury_radius_time_days: float
    mercury_radius_speed_kms: float
    mercury_radius_flux_multiple: float
    sun_surface_radius_au: float
    sun_surface_time_days: float
    sun_surface_speed_kms_idealized: float
    scenario: str
    interpretation: str
    nasa_earth_url: str
    nasa_au_url: str
    nasa_sun_url: str


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


def fall_time_to_radius(r_m: float) -> float:
    """Seconds for radial free fall from rest at 1 AU to radius r around point-mass Sun."""
    r0 = AU_M
    x = clamp(r_m / r0, 0.0, 1.0)
    eta = math.acos(2.0 * x - 1.0)
    return math.sqrt(r0**3 / (8.0 * MU_SUN)) * (eta + math.sin(eta))


def fall_speed_at_radius(r_m: float) -> float:
    """m/s from energy conservation for radial fall from rest at 1 AU."""
    return math.sqrt(max(0.0, 2.0 * MU_SUN * (1.0 / r_m - 1.0 / AU_M)))


def radius_from_eta(eta: float) -> float:
    return 0.5 * AU_M * (1.0 + math.cos(eta))


def time_from_eta(eta: float) -> float:
    return math.sqrt(AU_M**3 / (8.0 * MU_SUN)) * (eta + math.sin(eta))


def eta_for_radius(r_m: float) -> float:
    return math.acos(2.0 * clamp(r_m / AU_M, 0.0, 1.0) - 1.0)


def build_snapshot() -> StopOrbitSnapshot:
    venus_au = 0.723
    mercury_au = 0.387
    venus_r = venus_au * AU_M
    mercury_r = mercury_au * AU_M
    sun_au = R_SUN_M / AU_M

    return StopOrbitSnapshot(
        generated_at_utc=iso_z(utc_now()),
        initial_distance_au=1.0,
        initial_orbital_speed_kms=EARTH_ORBIT_SPEED_KMS,
        removed_tangential_speed_kms=EARTH_ORBIT_SPEED_KMS,
        initial_solar_acceleration_ms2=MU_SUN / AU_M**2,
        venus_radius_au=venus_au,
        venus_radius_time_days=fall_time_to_radius(venus_r) / SECONDS_PER_DAY,
        venus_radius_speed_kms=fall_speed_at_radius(venus_r) / 1000.0,
        venus_radius_flux_multiple=1.0 / venus_au**2,
        mercury_radius_au=mercury_au,
        mercury_radius_time_days=fall_time_to_radius(mercury_r) / SECONDS_PER_DAY,
        mercury_radius_speed_kms=fall_speed_at_radius(mercury_r) / 1000.0,
        mercury_radius_flux_multiple=1.0 / mercury_au**2,
        sun_surface_radius_au=sun_au,
        sun_surface_time_days=fall_time_to_radius(R_SUN_M) / SECONDS_PER_DAY,
        sun_surface_speed_kms_idealized=fall_speed_at_radius(R_SUN_M) / 1000.0,
        scenario="Instantaneously remove Earth's heliocentric tangential velocity while preserving Earth's rotation; ignore Moon and other planets.",
        interpretation="Idealized radial two-body free fall from rest at 1 AU. Radial milestones are distances from the Sun, not encounters with Venus or Mercury.",
        nasa_earth_url="https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html",
        nasa_au_url="https://science.nasa.gov/learn/basics-of-space-flight/units/",
        nasa_sun_url="https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html",
    )


def save_data(snapshot: StopOrbitSnapshot) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "earth_radial_fall_from_1au.csv"
    json_path = DATA_ROOT / "earth_stop_orbit_snapshot.json"

    eta_end = eta_for_radius(R_SUN_M)
    rows: List[Dict[str, float]] = []
    for i in range(241):
        eta = eta_end * i / 240.0
        r_m = radius_from_eta(eta)
        t_s = time_from_eta(eta)
        speed = fall_speed_at_radius(r_m)
        r_au = r_m / AU_M
        rows.append({
            "sample": i,
            "time_days": t_s / SECONDS_PER_DAY,
            "distance_au": r_au,
            "distance_million_km": r_m / 1e9,
            "radial_speed_km_s": speed / 1000.0,
            "relative_solar_flux": 1.0 / (r_au * r_au),
        })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps({"snapshot": asdict(snapshot), "radial_fall_samples": rows}, indent=2),
        encoding="utf-8",
    )
    return csv_path, json_path


# =============================================================================
# Typography / rendering helpers
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


def draw_text(
    image: Image.Image,
    value: str,
    xy: Tuple[int, int],
    size: int,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    anchor: str = "la",
    stroke: int = 2,
):
    ImageDraw.Draw(image).text(
        xy,
        value,
        font=get_font(size, bold),
        fill=fill,
        anchor=anchor,
        stroke_width=max(1, int(stroke * SCALE)),
        stroke_fill=(0, 0, 0, 220),
    )


def draw_wrapped_text(
    image: Image.Image,
    value: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    spacing: int = 6,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold)
    words = value.split()
    lines: List[str] = []
    current = ""
    stroke_width = max(1, int(2 * SCALE))
    for word in words:
        candidate = word if not current else current + " " + word
        box = draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke_width)
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
            (x, y), line, font=font, fill=fill,
            stroke_width=stroke_width, stroke_fill=(0, 0, 0, 220),
        )
        box = draw.textbbox((x, y), line, font=font, stroke_width=stroke_width)
        y += (box[3] - box[1]) + int(spacing * SCALE)


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
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

class StopOrbitScene:
    def __init__(self, snapshot: StopOrbitSnapshot):
        self.snapshot = snapshot
        rng = np.random.default_rng(281104)
        self.stars = [
            (
                float(rng.uniform(0, W)), float(rng.uniform(0, H)),
                float(rng.uniform(0.35, 2.0) * SCALE),
                int(rng.uniform(18, 100)), float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(170 if QUICK_MODE else 520)
        ]

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 178):
        layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle(
            box,
            radius=max(8, int(24 * SCALE)),
            fill=COLORS["panel"] + (alpha,),
            outline=COLORS["cyan"] + (44,),
            width=max(1, int(2 * SCALE)),
        )
        image.alpha_composite(layer)

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        arr[..., 0] = np.clip(2 + yy * 7, 0, 255)
        arr[..., 1] = np.clip(6 + yy * 15, 0, 255)
        arr[..., 2] = np.clip(18 + yy * 30, 0, 255)
        image = Image.fromarray(arr, "RGB").convert("RGBA")
        d = ImageDraw.Draw(image)
        for x, y, r, a0, phase in self.stars:
            a = int(a0 * (0.70 + 0.30 * math.sin(t * 0.85 + phase)))
            d.ellipse((x-r, y-r, x+r, y+r), fill=COLORS["white"] + (max(3, a),))
        return image

    def draw_sun(self, image: Image.Image, center: Tuple[int, int], radius: int, t: float):
        cx, cy = center
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for k, alpha in [(2.6, 12), (1.85, 24), (1.35, 44)]:
            rr = int(radius * k)
            gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=COLORS["gold"] + (alpha,))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4, int(22*SCALE)))))
        d = ImageDraw.Draw(image)
        d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=COLORS["sun"] + (255,), outline=COLORS["sun_hot"] + (230,), width=max(1, int(3*SCALE)))
        for i in range(8):
            a = i * math.tau / 8 + t * 0.08
            rr = radius * (0.43 + 0.12 * math.sin(t * 0.6 + i))
            x = cx + math.cos(a) * rr
            y = cy + math.sin(a) * rr
            q = max(2, int(radius * 0.10))
            d.ellipse((x-q, y-q, x+q, y+q), fill=COLORS["orange"] + (45,))

    def draw_earth(self, image: Image.Image, center: Tuple[int, int], radius: int, hot: float = 0.0):
        cx, cy = center
        d = ImageDraw.Draw(image)
        base = tuple(int(lerp(COLORS["earth"][i], COLORS["orange"][i], clamp(hot))) for i in range(3))
        d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=base + (255,), outline=COLORS["cyan"] + (170,), width=max(1, int(2*SCALE)))
        d.arc((cx-radius*.75, cy-radius*.50, cx+radius*.40, cy+radius*.45), 205, 355, fill=COLORS["earth_land"]+(220,), width=max(2, int(radius*.24)))
        d.arc((cx-radius*.28, cy-radius*.85, cx+radius*.72, cy+radius*.35), 30, 150, fill=COLORS["earth_land"]+(180,), width=max(2, int(radius*.18)))
        if hot > 0.15:
            glow = Image.new("RGBA", SIZE, (0,0,0,0)); gd = ImageDraw.Draw(glow)
            rr = int(radius * (1.5 + 1.0*hot))
            gd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=COLORS["orange"]+(int(28+55*hot),))
            image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(3,int(12*SCALE)))))

    def draw_title(self, image: Image.Image, t: float):
        if t >= (5.8 if not QUICK_MODE else 1.30):
            return
        fade = smoothstep(t / (0.8 if not QUICK_MODE else 0.18))
        a = int(250 * fade)
        draw_text(image, "WHAT HAPPENS IF EARTH", (W//2, int(H*.060)), 35, COLORS["white"]+(a,), True, "ma", 2)
        draw_text(image, "STOPS ORBITING?", (W//2, int(H*.109)), 47, COLORS["cyan"]+(a,), True, "ma", 2)
        draw_text(image, CONFIG["subtitle"], (W//2, int(H*.157)), 15, COLORS["muted"]+(min(a,225),), True, "ma", 1)

    def draw_source_hud(self, image: Image.Image):
        draw_text(image, "TWO-BODY THOUGHT EXPERIMENT", (W-int(42*SCALE), int(65*SCALE)), 12, COLORS["cyan"]+(210,), True, "ra", 1)
        draw_text(image, "NASA ORBITAL PARAMETERS", (W-int(42*SCALE), int(92*SCALE)), 10, COLORS["muted"]+(190,), False, "ra", 1)

    def draw_caption(self, image: Image.Image, t: float):
        caption = caption_at(t)
        if not caption:
            return
        y0 = H - int(190*SCALE)
        layer = Image.new("RGBA", SIZE, (0,0,0,0)); d = ImageDraw.Draw(layer)
        d.rounded_rectangle(
            (int(58*SCALE), y0, W-int(58*SCALE), y0+int(116*SCALE)),
            radius=max(10,int(20*SCALE)), fill=(2,6,14,150), outline=COLORS["cyan"]+(48,), width=max(1,int(2*SCALE))
        )
        image.alpha_composite(layer)
        draw_wrapped_text(image, caption, (int(82*SCALE), y0+int(18*SCALE)), W-int(164*SCALE), 30, COLORS["white"]+(240,), False, 4)

    @staticmethod
    def arrow(d: ImageDraw.ImageDraw, p0: Tuple[float,float], p1: Tuple[float,float], fill, width: int):
        d.line((*p0,*p1), fill=fill, width=width)
        x0,y0=p0; x1,y1=p1
        ang=math.atan2(y1-y0,x1-x0)
        q=max(7,width*3.1)
        a1=ang+2.55; a2=ang-2.55
        d.polygon([(x1,y1),(x1+math.cos(a1)*q,y1+math.sin(a1)*q),(x1+math.cos(a2)*q,y1+math.sin(a2)*q)],fill=fill)

    def draw_orbit_is_falling(self, image: Image.Image, t: float, p: float):
        cx=W//2; cy=int(H*.44); orbit_r=int(270*SCALE)
        self.draw_sun(image,(cx,cy),int(56*SCALE),t)
        d=ImageDraw.Draw(image)
        d.ellipse((cx-orbit_r,cy-orbit_r, cx+orbit_r,cy+orbit_r), outline=COLORS["cyan"]+(70,), width=max(2,int(3*SCALE)))
        ang=-0.55 + p*1.10
        ex=cx+math.cos(ang)*orbit_r; ey=cy+math.sin(ang)*orbit_r
        self.draw_earth(image,(int(ex),int(ey)),int(29*SCALE))
        # tangential and gravity vectors
        tang=(-math.sin(ang), math.cos(ang))
        inward=(-math.cos(ang), -math.sin(ang))
        self.arrow(d,(ex,ey),(ex+tang[0]*120*SCALE,ey+tang[1]*120*SCALE),COLORS["cyan"]+(235,),max(2,int(5*SCALE)))
        self.arrow(d,(ex,ey),(ex+inward[0]*92*SCALE,ey+inward[1]*92*SCALE),COLORS["red"]+(220,),max(2,int(5*SCALE)))
        draw_text(image,"29.78 km/s SIDEWAYS",(int(ex+tang[0]*145*SCALE),int(ey+tang[1]*145*SCALE)),13,COLORS["cyan"]+(235,),True,"ma",1)
        draw_text(image,"GRAVITY",(int(ex+inward[0]*116*SCALE),int(ey+inward[1]*116*SCALE)),12,COLORS["red"]+(220,),True,"ma",1)
        draw_text(image,"ORBIT = FALLING SIDEWAYS",(W//2,int(H*.18)),31,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"Earth keeps missing the Sun",(W//2,int(H*.235)),16,COLORS["muted"]+(225,),False,"ma",1)

    def draw_velocity_removed(self, image: Image.Image, t: float, p: float):
        sx=int(W*.24); sy=int(H*.43); ex=int(W*.76); ey=sy
        self.draw_sun(image,(sx,sy),int(68*SCALE),t)
        self.draw_earth(image,(ex,ey),int(38*SCALE))
        d=ImageDraw.Draw(image)
        # old tangent vector fading out
        a=int(240*(1-smoothstep(p/.55)))
        if a>3:
            self.arrow(d,(ex,ey),(ex,ey-int(155*SCALE)),COLORS["cyan"]+(a,),max(2,int(6*SCALE)))
            draw_text(image,"29.78 km/s",(ex,ey-int(180*SCALE)),14,COLORS["cyan"]+(a,),True,"ma",1)
        # zero badge
        q=smoothstep((p-.20)/.45)
        if q>0:
            self.panel(image,(int(W*.56),int(H*.55),int(W*.90),int(H*.68)),155)
            draw_text(image,"SIDEWAYS SPEED",(int(W*.73),int(H*.585)),13,COLORS["muted"]+(int(230*q),),True,"ma",1)
            draw_text(image,"→ 0 km/s",(int(W*.73),int(H*.635)),25,COLORS["red"]+(int(245*q),),True,"ma",2)
        # inward acceleration
        self.arrow(d,(ex-int(40*SCALE),ey),(sx+int(82*SCALE),sy),COLORS["red"]+(230,),max(2,int(6*SCALE)))
        draw_text(image,"solar gravity ≈ 0.00593 m/s² at 1 AU",(W//2,int(H*.74)),15,COLORS["gold"]+(235,),True,"ma",1)
        draw_text(image,"REMOVE THE SIDEWAYS MOTION",(W//2,int(H*.17)),30,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"then there is nothing left to make Earth miss",(W//2,int(H*.22)),15,COLORS["muted"]+(220,),False,"ma",1)

    def draw_fall_clock(self, image: Image.Image, t: float, p: float):
        cx=int(W*.20); cy=int(H*.44); maxr=int(W*.68)
        self.draw_sun(image,(cx,cy),int(52*SCALE),t)
        d=ImageDraw.Draw(image)
        radii=[(1.0,"1.00 AU",COLORS["cyan"]),(self.snapshot.venus_radius_au,"0.723 AU",COLORS["gold"]),(self.snapshot.mercury_radius_au,"0.387 AU",COLORS["orange"])]
        for frac,label,col in radii:
            x=cx+maxr*frac
            d.line((x,int(H*.31),x,int(H*.62)),fill=col+(60,),width=max(1,int(2*SCALE)))
            draw_text(image,label,(int(x),int(H*.65)),11,col+(215,),True,"ma",1)
        # Earth position along a dramatized but monotonic radial track
        # use physical time fraction up to Sun-contact and invert approximately by eta interpolation
        target_days=p*self.snapshot.sun_surface_time_days
        target_s=target_days*SECONDS_PER_DAY
        lo,hi=0.0,eta_for_radius(R_SUN_M)
        for _ in range(32):
            mid=(lo+hi)/2
            if time_from_eta(mid)<target_s: lo=mid
            else: hi=mid
        eta=(lo+hi)/2
        r_au=radius_from_eta(eta)/AU_M
        earth_x=cx+maxr*r_au
        hot=clamp((0.65-r_au)/0.60)
        self.draw_earth(image,(int(earth_x),cy),int(27*SCALE),hot)
        d.line((cx+int(55*SCALE),cy,earth_x-int(30*SCALE),cy),fill=COLORS["red"]+(100,),width=max(2,int(4*SCALE)))
        draw_text(image,"THE FALL CLOCK",(W//2,int(H*.16)),32,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"radial distance from the Sun — not planet encounters",(W//2,int(H*.215)),13,COLORS["muted"]+(220,),False,"ma",1)
        self.panel(image,(int(W*.15),int(H*.70),int(W*.85),int(H*.83)),170)
        milestones=[
            ("VENUS DISTANCE",self.snapshot.venus_radius_time_days,COLORS["gold"]),
            ("MERCURY DISTANCE",self.snapshot.mercury_radius_time_days,COLORS["orange"]),
            ("SUN SURFACE",self.snapshot.sun_surface_time_days,COLORS["red"]),
        ]
        for i,(lab,days,col) in enumerate(milestones):
            x=int(W*(.27+.23*i))
            draw_text(image,lab,(x,int(H*.735)),10,col+(220,),True,"ma",1)
            draw_text(image,f"{days:.1f} DAYS",(x,int(H*.785)),18,COLORS["white"]+(240,),True,"ma",1)

    def draw_heating(self, image: Image.Image, t: float, p: float):
        d=ImageDraw.Draw(image)
        draw_text(image,"SUNLIGHT RISES AS 1 / r²",(W//2,int(H*.16)),31,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"same Sun luminosity — rapidly shrinking distance",(W//2,int(H*.215)),14,COLORS["muted"]+(220,),False,"ma",1)
        vals=[("EARTH",1.0,1.0,COLORS["cyan"]),("VENUS RADIUS",.723,self.snapshot.venus_radius_flux_multiple,COLORS["gold"]),("MERCURY RADIUS",.387,self.snapshot.mercury_radius_flux_multiple,COLORS["orange"])]
        y0=int(H*.34); gap=int(H*.14)
        for i,(lab,r,flux,col) in enumerate(vals):
            y=y0+i*gap
            draw_text(image,lab,(int(W*.12),y),14,col+(235,),True,"la",1)
            draw_text(image,f"{r:.3f} AU",(int(W*.38),y),13,COLORS["muted"]+(220,),False,"la",1)
            maxw=int(W*.34); bar=int(maxw*min(1.0,math.log10(flux+1)/math.log10(7.7)))
            d.rounded_rectangle((int(W*.55),y-int(11*SCALE),int(W*.55)+maxw,y+int(11*SCALE)),radius=max(3,int(8*SCALE)),fill=COLORS["panel"]+(210,))
            d.rounded_rectangle((int(W*.55),y-int(11*SCALE),int(W*.55)+bar,y+int(11*SCALE)),radius=max(3,int(8*SCALE)),fill=col+(220,))
            draw_text(image,f"{flux:.1f}×",(int(W*.91),y),17,col+(240,),True,"ra",1)
        self.panel(image,(int(W*.10),int(H*.72),int(W*.90),int(H*.81)),165)
        draw_text(image,"Exact climate / destruction timeline is not modeled",(W//2,int(H*.755)),14,COLORS["red"]+(230,),True,"ma",1)
        draw_text(image,"the approach becomes catastrophically hotter long before the photosphere",(W//2,int(H*.795)),11,COLORS["muted"]+(215,),False,"ma",1)

    def draw_speed_up(self, image: Image.Image, t: float, p: float):
        draw_text(image,"GRAVITY TURNS HEIGHT INTO SPEED",(W//2,int(H*.16)),29,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"idealized radial two-body energy conservation",(W//2,int(H*.215)),14,COLORS["muted"]+(220,),False,"ma",1)
        d=ImageDraw.Draw(image)
        x0=int(W*.13); x1=int(W*.87); y0=int(H*.66); y1=int(H*.31)
        d.line((x0,y0,x1,y0),fill=COLORS["white"]+(90,),width=max(2,int(3*SCALE)))
        d.line((x0,y0,x0,y1),fill=COLORS["white"]+(90,),width=max(2,int(3*SCALE)))
        points=[]
        for j in range(120):
            f=j/119
            r_au=lerp(1.0,self.snapshot.sun_surface_radius_au,f)
            v=fall_speed_at_radius(r_au*AU_M)/1000.0
            xx=lerp(x0,x1,f)
            yy=lerp(y0,y1,clamp(v/650.0))
            points.append((xx,yy))
        reveal=max(2,int(len(points)*clamp(p)))
        d.line(points[:reveal],fill=COLORS["cyan"]+(235,),width=max(2,int(5*SCALE)))
        # markers by radius fraction on horizontal axis
        for lab,r_au,v,col in [
            ("1 AU",1.0,0.0,COLORS["cyan"]),
            ("0.387 AU",self.snapshot.mercury_radius_au,self.snapshot.mercury_radius_speed_kms,COLORS["orange"]),
            ("SUN",self.snapshot.sun_surface_radius_au,self.snapshot.sun_surface_speed_kms_idealized,COLORS["red"]),
        ]:
            f=(1-r_au)/(1-self.snapshot.sun_surface_radius_au)
            x=lerp(x0,x1,f); y=lerp(y0,y1,clamp(v/650.0))
            q=max(3,int(6*SCALE)); d.ellipse((x-q,y-q,x+q,y+q),fill=col+(240,))
            draw_text(image,lab,(int(x),int(y-24*SCALE)),10,col+(230,),True,"ma",1)
        draw_text(image,"53 km/s at Mercury's orbital radius",(W//2,int(H*.72)),17,COLORS["orange"]+(240,),True,"ma",1)
        draw_text(image,"~616 km/s at solar surface — mathematical idealization",(W//2,int(H*.765)),15,COLORS["red"]+(235,),True,"ma",1)
        draw_text(image,"an intact Earth would not survive the approach",(W//2,int(H*.805)),11,COLORS["muted"]+(215,),False,"ma",1)

    def draw_outro(self, image: Image.Image, t: float, p: float):
        cx=int(W*.20); cy=int(H*.44); maxr=int(W*.67)
        self.draw_sun(image,(cx,cy),int(64*SCALE),t)
        d=ImageDraw.Draw(image)
        # ghost circular orbit
        d.ellipse((cx-maxr,cy-maxr,cx+maxr,cy+maxr),outline=COLORS["cyan"]+(30,),width=max(1,int(2*SCALE)))
        # radial plunge line and Earth approaching
        d.line((cx+int(70*SCALE),cy,cx+maxr,cy),fill=COLORS["red"]+(115,),width=max(2,int(5*SCALE)))
        r_au=lerp(1.0,.08,smoothstep(p))
        ex=cx+maxr*r_au
        self.draw_earth(image,(int(ex),cy),int(28*SCALE),clamp((.55-r_au)/.50))
        draw_text(image,"STOP ORBITING ≠ STOP MOVING",(W//2,int(H*.16)),30,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"remove the sideways velocity → radial free fall",(W//2,int(H*.215)),15,COLORS["cyan"]+(230,),True,"ma",1)
        self.panel(image,(int(W*.16),int(H*.66),int(W*.84),int(H*.80)),175)
        draw_text(image,"SIDEWAYS SPEED REMOVED",(W//2,int(H*.70)),12,COLORS["muted"]+(220,),True,"ma",1)
        draw_text(image,"≈ 29.8 km/s",(W//2,int(H*.745)),26,COLORS["cyan"]+(245,),True,"ma",1)
        draw_text(image,"SUN REACHED IN ≈ 64.6 DAYS",(W//2,int(H*.79)),17,COLORS["red"]+(240,),True,"ma",1)

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
        if name=="orbit_is_falling": self.draw_orbit_is_falling(image,t,p)
        elif name=="velocity_removed": self.draw_velocity_removed(image,t,p)
        elif name=="fall_clock": self.draw_fall_clock(image,t,p)
        elif name=="heating": self.draw_heating(image,t,p)
        elif name=="speed_up": self.draw_speed_up(image,t,p)
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

def render_preview_frames(scene: StopOrbitScene) -> List[Path]:
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


def render_video(scene: StopOrbitScene) -> Path:
    path=OUTPUT_ROOT/f"{CONFIG['basename']}.mp4"
    total=max(1,int(round(DURATION*FPS)))
    writer=iio.get_writer(path,fps=FPS,codec="libx264",quality=7,pixelformat="yuv420p",ffmpeg_log_level="error",macro_block_size=2)
    try:
        for i in tqdm(range(total),desc="Rendering Earth Stop-Orbit Short"):
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

YOUTUBE_TITLE = 'What Happens If Earth Suddenly Stops Orbiting? 🌍☀️ #rootjatin'
YOUTUBE_DESCRIPTION = "This is a hypothetical two-body thought experiment: Earth's sideways orbital velocity is instantly removed while solar gravity remains. Instead of hovering, Earth begins falling toward the Sun and accelerates as gravitational potential energy becomes speed. The timings, speeds, and sunlight changes shown come from a simplified orbital model; real climate destruction and a physically plausible mechanism for stopping Earth are not modeled."
YOUTUBE_HASHTAGS = '#rootjatin #Earth #Sun #Orbit #OrbitalMechanics #WhatIf #Space #Physics'

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


