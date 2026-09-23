from __future__ import annotations

"""
A Planet Orbiting Three Stars
=============================
output : https://www.youtube.com/shorts/cfBigudkhc4
A cinematic vertical YouTube Short renderer about a physically plausible
version of a planet orbiting three stars: a distant circUMTRIPLE planet around
an idealized hierarchical triple-star system.

Scientific framing
------------------
- Triple-star systems are real, and planets are known in triple-star systems.
  NASA's KOI-5Ab example orbits one star while two stellar companions remain
  gravitationally bound to the system.
- A compact hierarchical stellar triple can also be dynamically stable. NASA
  has noted that a sufficiently distant planet could orbit all three members
  of such a compact triple approximately as if their combined mass were one
  central object.
- This video does NOT claim a confirmed Earth-like planet is known to orbit all
  three stars in the exact configuration rendered here.
- The renderer uses a deliberately simplified, coplanar, circular toy system:
      Star A: 1.00 solar masses, 1.00 solar luminosity
      Star B: 0.70 solar masses, 0.20 solar luminosity
      Star C: 0.40 solar masses, 0.03 solar luminosity
      A-B inner separation:       0.08 AU
      (A+B)-C outer separation:   0.24 AU
      Planet barycentric radius:  1.20 AU
- In this toy model the A-B pair completes an orbit in about 6.34 days, Star C
  and the A-B barycenter orbit each other in about 29.63 days, and the planet
  completes a wide circumtriple orbit in about 331.3 days.
- Incoming light varies because all three stars move relative to the planet.
  The exact temperature or habitability response is NOT modeled.
- Stability in real hierarchical triples depends on masses, eccentricities,
  inclinations, separations, resonances, and formation history. The geometry
  here is educational rather than a formal N-body stability proof.

Primary references
------------------
NASA — KOI-5Ab, a planet in a triple-star system:
    https://www.nasa.gov/missions/kepler/planetary-sleuthing-finds-triple-star-world/
NASA — TESS compact stellar triplet; a distant planet could orbit all three:
    https://www.nasa.gov/universe/nasas-tess-spots-record-breaking-stellar-triplets/
NASA Science — Alpha Centauri is a nearby triple-star system:
    https://science.nasa.gov/missions/webb/nasas-webb-finds-new-evidence-for-planet-around-closest-solar-twin/


Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    THREE_STARS_SHORT_QUICK=1 python a_planet_orbiting_three_stars.py

Full 1080x1920 render
---------------------
    python a_planet_orbiting_three_stars.py

4K vertical render
------------------
    THREE_STARS_SHORT_4K=1 python a_planet_orbiting_three_stars.py
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

QUICK_MODE = os.environ.get("THREE_STARS_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("THREE_STARS_SHORT_4K", "0") == "1" and not QUICK_MODE

W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 6 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SIZE = (W, H)
SCALE = W / 1080.0

OUTPUT_ROOT = Path("a_planet_orbiting_three_stars_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "A PLANET ORBITING THREE STARS",
    "subtitle": "One world // three suns // one shared center of mass",
    "basename": "a_planet_orbiting_three_stars",
    "contrast": 1.11,
    "saturation": 1.08,
    "vignette": 0.28,
}

COLORS = {
    "space": (2, 5, 16),
    "space2": (7, 16, 36),
    "white": (248, 251, 255),
    "muted": (166, 193, 216),
    "cyan": (77, 229, 255),
    "blue": (73, 139, 255),
    "planet": (57, 156, 232),
    "land": (98, 209, 142),
    "star_a": (255, 225, 117),
    "star_a_hot": (255, 250, 218),
    "star_b": (255, 152, 89),
    "star_b_hot": (255, 208, 168),
    "star_c": (255, 93, 78),
    "star_c_hot": (255, 165, 139),
    "gold": (255, 204, 84),
    "orange": (255, 145, 71),
    "red": (255, 83, 103),
    "green": (106, 239, 176),
    "violet": (185, 129, 255),
    "panel": (3, 9, 22),
}

SHOT_PLAN = [
    {"name": "triple_reveal", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "hierarchy", "start": 8.0 if not QUICK_MODE else 1.8, "end": 19.0 if not QUICK_MODE else 4.25},
    {"name": "circumtriple", "start": 19.0 if not QUICK_MODE else 4.25, "end": 30.0 if not QUICK_MODE else 6.70},
    {"name": "sky", "start": 30.0 if not QUICK_MODE else 6.70, "end": 41.0 if not QUICK_MODE else 9.15},
    {"name": "changing_light", "start": 41.0 if not QUICK_MODE else 9.15, "end": 51.0 if not QUICK_MODE else 11.35},
    {"name": "outro", "start": 51.0 if not QUICK_MODE else 11.35, "end": DURATION},
]

CAPTION_TEXTS = [
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
# Illustrative hierarchical triple model
# =============================================================================

M_A = 1.00
M_B = 0.70
M_C = 0.40
L_A = 1.00
L_B = 0.20
L_C = 0.03
A_INNER = 0.08   # AU between A and B
A_OUTER = 0.24   # AU between AB barycenter and C
A_PLANET = 1.20  # AU from total-system barycenter
DAYS_PER_YEAR = 365.256

M_AB = M_A + M_B
M_TOTAL = M_AB + M_C

# Inner A-B barycentric radii
R_A_INNER = A_INNER * M_B / M_AB
R_B_INNER = A_INNER * M_A / M_AB

# Outer orbit radii around total barycenter
R_AB_OUTER = A_OUTER * M_C / M_TOTAL
R_C_OUTER = A_OUTER * M_AB / M_TOTAL

P_INNER_DAYS = math.sqrt(A_INNER**3 / M_AB) * DAYS_PER_YEAR
P_OUTER_DAYS = math.sqrt(A_OUTER**3 / M_TOTAL) * DAYS_PER_YEAR
P_PLANET_DAYS = math.sqrt(A_PLANET**3 / M_TOTAL) * DAYS_PER_YEAR


@dataclass
class ThreeStarsSnapshot:
    generated_at_utc: str
    star_a_mass_solar: float
    star_b_mass_solar: float
    star_c_mass_solar: float
    star_a_luminosity_solar: float
    star_b_luminosity_solar: float
    star_c_luminosity_solar: float
    inner_separation_au: float
    outer_separation_au: float
    planet_barycentric_radius_au: float
    inner_period_days: float
    outer_period_days: float
    planet_period_days: float
    flux_min_earth_units: float
    flux_max_earth_units: float
    flux_mean_earth_units: float
    flux_peak_to_trough_pct: float
    scenario: str
    interpretation: str
    nasa_triple_planet_url: str
    nasa_compact_triple_url: str


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


def triple_state(days: float) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Approximate coplanar circular hierarchical-triple positions in AU."""
    outer = 2.0 * math.pi * days / P_OUTER_DAYS
    inner = 2.0 * math.pi * days / P_INNER_DAYS + 0.55

    ab_x = R_AB_OUTER * math.cos(outer)
    ab_y = R_AB_OUTER * math.sin(outer)
    c = (-R_C_OUTER * math.cos(outer), -R_C_OUTER * math.sin(outer))

    a = (
        ab_x + R_A_INNER * math.cos(inner),
        ab_y + R_A_INNER * math.sin(inner),
    )
    b = (
        ab_x - R_B_INNER * math.cos(inner),
        ab_y - R_B_INNER * math.sin(inner),
    )
    return a, b, c


def planet_state(days: float) -> Tuple[float, float]:
    ang = 2.0 * math.pi * days / P_PLANET_DAYS
    return A_PLANET * math.cos(ang), A_PLANET * math.sin(ang)


def total_flux(days: float) -> float:
    p = planet_state(days)
    a, b, c = triple_state(days)
    flux = 0.0
    for lum, star in ((L_A, a), (L_B, b), (L_C, c)):
        d = math.hypot(p[0] - star[0], p[1] - star[1])
        flux += lum / max(1e-9, d * d)
    return flux


def sample_flux_stats() -> Tuple[float, float, float]:
    vals = [total_flux(5.0 * P_PLANET_DAYS * i / 10000.0) for i in range(10001)]
    return min(vals), max(vals), float(sum(vals) / len(vals))


def build_snapshot() -> ThreeStarsSnapshot:
    fmin, fmax, fmean = sample_flux_stats()
    return ThreeStarsSnapshot(
        generated_at_utc=iso_z(utc_now()),
        star_a_mass_solar=M_A,
        star_b_mass_solar=M_B,
        star_c_mass_solar=M_C,
        star_a_luminosity_solar=L_A,
        star_b_luminosity_solar=L_B,
        star_c_luminosity_solar=L_C,
        inner_separation_au=A_INNER,
        outer_separation_au=A_OUTER,
        planet_barycentric_radius_au=A_PLANET,
        inner_period_days=P_INNER_DAYS,
        outer_period_days=P_OUTER_DAYS,
        planet_period_days=P_PLANET_DAYS,
        flux_min_earth_units=fmin,
        flux_max_earth_units=fmax,
        flux_mean_earth_units=fmean,
        flux_peak_to_trough_pct=(fmax / fmin - 1.0) * 100.0,
        scenario=(
            "Illustrative coplanar hierarchical triple: 1.0 + 0.7 Msun inner binary "
            "at 0.08 AU; 0.4 Msun tertiary at 0.24 AU; test planet at 1.2 AU."
        ),
        interpretation=(
            "Educational Keplerian geometry, not a formal N-body stability proof or a "
            "reconstruction of a confirmed circumtriple exoplanet."
        ),
        nasa_triple_planet_url="https://www.nasa.gov/missions/kepler/planetary-sleuthing-finds-triple-star-world/",
        nasa_compact_triple_url="https://www.nasa.gov/universe/nasas-tess-spots-record-breaking-stellar-triplets/",
    )


def save_data(snapshot: ThreeStarsSnapshot) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "illustrative_circumtriple_orbit.csv"
    json_path = DATA_ROOT / "three_stars_snapshot.json"
    rows: List[Dict[str, float]] = []
    for i in range(361):
        day = snapshot.planet_period_days * i / 360.0
        p = planet_state(day)
        a, b, c = triple_state(day)
        distances = [math.hypot(p[0] - s[0], p[1] - s[1]) for s in (a, b, c)]
        rows.append({
            "sample": i,
            "day": day,
            "planet_x_au": p[0],
            "planet_y_au": p[1],
            "star_a_x_au": a[0],
            "star_a_y_au": a[1],
            "star_b_x_au": b[0],
            "star_b_y_au": b[1],
            "star_c_x_au": c[0],
            "star_c_y_au": c[1],
            "distance_a_au": distances[0],
            "distance_b_au": distances[1],
            "distance_c_au": distances[2],
            "relative_total_flux": total_flux(day),
        })
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"snapshot": asdict(snapshot), "samples": rows}, indent=2), encoding="utf-8")
    return csv_path, json_path


# =============================================================================
# Typography / output helpers
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
              fill=(255, 255, 255, 255), bold: bool = False,
              anchor: str = "la", stroke: int = 2):
    ImageDraw.Draw(image).text(
        xy, value, font=get_font(size, bold), fill=fill, anchor=anchor,
        stroke_width=max(1, int(stroke * SCALE)), stroke_fill=(0, 0, 0, 220),
    )


def draw_wrapped_text(image: Image.Image, value: str, xy: Tuple[int, int], max_width: int,
                      size: int, fill=(255, 255, 255, 245), bold: bool = False,
                      spacing: int = 6):
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

class ThreeStarsScene:
    def __init__(self, snapshot: ThreeStarsSnapshot):
        self.snapshot = snapshot
        rng = np.random.default_rng(21092026)
        self.stars = [
            (
                float(rng.uniform(0, W)), float(rng.uniform(0, H)),
                float(rng.uniform(.35, 2.0) * SCALE), int(rng.uniform(14, 92)),
                float(rng.uniform(0, math.tau)),
            )
            for _ in range(170 if QUICK_MODE else 520)
        ]

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 178):
        layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle(
            box, radius=max(8, int(24 * SCALE)), fill=COLORS["panel"] + (alpha,),
            outline=COLORS["cyan"] + (45,), width=max(1, int(2 * SCALE)),
        )
        image.alpha_composite(layer)

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        arr[..., 0] = np.clip(2 + yy * 8, 0, 255)
        arr[..., 1] = np.clip(6 + yy * 18, 0, 255)
        arr[..., 2] = np.clip(18 + yy * 35, 0, 255)
        image = Image.fromarray(arr, "RGB").convert("RGBA")
        d = ImageDraw.Draw(image)
        for x, y, r, a, phase in self.stars:
            tw = .66 + .34 * math.sin(t * .9 + phase)
            q = r * (.82 + .18 * math.sin(t * 1.3 + phase))
            d.ellipse((x-q, y-q, x+q, y+q), fill=COLORS["white"] + (int(a * tw),))
        return image

    def draw_star(self, image: Image.Image, center: Tuple[int, int], radius: int,
                  color: Tuple[int, int, int], hot: Tuple[int, int, int], t: float,
                  phase: float = 0.0):
        cx, cy = center
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        pulse = 1.0 + .025 * math.sin(t * 2.1 + phase)
        rr = max(2, int(radius * pulse))
        for mult, alpha in ((2.8, 18), (2.0, 28), (1.45, 48)):
            q = int(rr * mult)
            gd.ellipse((cx-q, cy-q, cx+q, cy+q), fill=color + (alpha,))
        glow = glow.filter(ImageFilter.GaussianBlur(max(3, int(15 * SCALE))))
        image.alpha_composite(glow)
        d = ImageDraw.Draw(image)
        d.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=hot + (255,), outline=color + (235,), width=max(1, int(2*SCALE)))
        d.ellipse((cx-int(rr*.63), cy-int(rr*.63), cx+int(rr*.63), cy+int(rr*.63)), fill=color + (92,))

    def draw_planet(self, image: Image.Image, center: Tuple[int, int], radius: int, t: float = 0.0):
        cx, cy = center
        d = ImageDraw.Draw(image)
        d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=COLORS["planet"]+(255,), outline=COLORS["cyan"]+(180,), width=max(1,int(2*SCALE)))
        d.ellipse((cx-int(radius*.55),cy-int(radius*.30),cx-int(radius*.05),cy+int(radius*.12)),fill=COLORS["land"]+(220,))
        d.ellipse((cx-int(radius*.04),cy+int(radius*.15),cx+int(radius*.50),cy+int(radius*.56)),fill=COLORS["land"]+(205,))
        shade=Image.new("RGBA",SIZE,(0,0,0,0)); sd=ImageDraw.Draw(shade)
        sd.pieslice((cx-radius,cy-radius,cx+radius,cy+radius),-90,90,fill=(0,0,10,105))
        image.alpha_composite(shade)

    def draw_title(self, image: Image.Image, t: float):
        if t >= (5.9 if not QUICK_MODE else 1.32):
            return
        a = int(245 * smoothstep(t / (.8 if not QUICK_MODE else .18)))
        draw_text(image,"A PLANET ORBITING",(W//2,int(H*.060)),35,COLORS["white"]+(a,),True,"ma",2)
        draw_text(image,"THREE STARS",(W//2,int(H*.111)),53,COLORS["cyan"]+(a,),True,"ma",2)
        draw_text(image,"one world • three suns • one barycenter",(W//2,int(H*.162)),15,COLORS["muted"]+(min(225,a),),False,"ma",1)

    def draw_source_hud(self, image: Image.Image):
        draw_text(image,"TRIPLE-STAR DYNAMICS // ILLUSTRATIVE",(W-int(40*SCALE),int(58*SCALE)),12,COLORS["cyan"]+(215,),True,"ra",1)
        draw_text(image,"NASA CONTEXT • KEPLERIAN TOY MODEL",(W-int(40*SCALE),int(84*SCALE)),10,COLORS["muted"]+(200,),False,"ra",1)

    def draw_caption(self, image: Image.Image, t: float):
        cap = caption_at(t)
        if not cap:
            return
        y0 = H - int(190*SCALE)
        self.panel(image,(int(58*SCALE),y0,W-int(58*SCALE),y0+int(116*SCALE)),150)
        draw_wrapped_text(image,cap,(int(82*SCALE),y0+int(18*SCALE)),W-int(164*SCALE),30,COLORS["white"]+(240,),False,4)

    def draw_triple_reveal(self, image: Image.Image, t: float, p: float):
        y=int(H*.43)
        sep=int(W*.18*(.45+.55*smoothstep(p)))
        self.draw_star(image,(W//2-sep,y),int(62*SCALE),COLORS["star_a"],COLORS["star_a_hot"],t)
        self.draw_star(image,(W//2+int(sep*.35),y-int(54*SCALE)),int(39*SCALE),COLORS["star_b"],COLORS["star_b_hot"],t,.7)
        self.draw_star(image,(W//2+sep,y+int(66*SCALE)),int(27*SCALE),COLORS["star_c"],COLORS["star_c_hot"],t,1.4)
        draw_text(image,"THREE SUNS. ONE SYSTEM.",(W//2,int(H*.68)),29,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"the stable version is hierarchical",(W//2,int(H*.73)),14,COLORS["violet"]+(230,),True,"ma",1)

    def draw_hierarchy(self, image: Image.Image, t: float, p: float):
        draw_text(image,"A HIERARCHICAL TRIPLE",(W//2,int(H*.14)),30,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"two stars tight inside • third star wider outside",(W//2,int(H*.193)),13,COLORS["muted"]+(220,),False,"ma",1)
        cx=W//2; cy=int(H*.48)
        d=ImageDraw.Draw(image)
        r_outer=int(215*SCALE)
        r_inner=int(83*SCALE)
        d.ellipse((cx-r_outer,cy-r_outer,cx+r_outer,cy+r_outer),outline=COLORS["violet"]+(95,),width=max(2,int(4*SCALE)))
        d.ellipse((cx-r_inner,cy-r_inner,cx+r_inner,cy+r_inner),outline=COLORS["cyan"]+(100,),width=max(2,int(4*SCALE)))
        day = p * 2.0 * P_OUTER_DAYS
        a,b,c = triple_state(day)
        scale = r_outer / (R_C_OUTER*1.18)
        for pos,r,col,hot,ph in [
            (a,int(32*SCALE),COLORS["star_a"],COLORS["star_a_hot"],0.0),
            (b,int(24*SCALE),COLORS["star_b"],COLORS["star_b_hot"],.7),
            (c,int(20*SCALE),COLORS["star_c"],COLORS["star_c_hot"],1.4),
        ]:
            self.draw_star(image,(int(cx+pos[0]*scale),int(cy+pos[1]*scale)),r,col,hot,t,ph)
        d.ellipse((cx-int(5*SCALE),cy-int(5*SCALE),cx+int(5*SCALE),cy+int(5*SCALE)),fill=COLORS["white"]+(245,))
        draw_text(image,"SYSTEM BARYCENTER",(cx,cy+int(34*SCALE)),11,COLORS["white"]+(220,),True,"ma",1)
        self.panel(image,(int(W*.13),int(H*.72),int(W*.87),int(H*.83)),168)
        draw_text(image,f"INNER PAIR  ≈ {self.snapshot.inner_period_days:.2f} DAYS",(W//2,int(H*.757)),15,COLORS["cyan"]+(240,),True,"ma",1)
        draw_text(image,f"THIRD STAR  ≈ {self.snapshot.outer_period_days:.2f} DAYS",(W//2,int(H*.802)),15,COLORS["violet"]+(240,),True,"ma",1)

    def draw_circumtriple(self, image: Image.Image, t: float, p: float):
        draw_text(image,"THE PLANET ORBITS ALL THREE",(W//2,int(H*.14)),29,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"wide outside • compact stars inside",(W//2,int(H*.195)),14,COLORS["cyan"]+(225,),True,"ma",1)
        cx=W//2; cy=int(H*.45)
        d=ImageDraw.Draw(image)
        r_planet=int(310*SCALE)
        r_outer=int(72*SCALE)
        d.ellipse((cx-r_planet,cy-r_planet,cx+r_planet,cy+r_planet),outline=COLORS["cyan"]+(80,),width=max(2,int(4*SCALE)))
        d.ellipse((cx-r_outer,cy-r_outer,cx+r_outer,cy+r_outer),outline=COLORS["violet"]+(65,),width=max(1,int(3*SCALE)))
        day=p*self.snapshot.planet_period_days*.74
        a,b,c=triple_state(day)
        sscale=r_outer/(R_C_OUTER*1.15)
        for pos,r,col,hot,ph in [
            (a,int(24*SCALE),COLORS["star_a"],COLORS["star_a_hot"],0),
            (b,int(18*SCALE),COLORS["star_b"],COLORS["star_b_hot"],.7),
            (c,int(14*SCALE),COLORS["star_c"],COLORS["star_c_hot"],1.4),
        ]:
            self.draw_star(image,(int(cx+pos[0]*sscale),int(cy+pos[1]*sscale)),r,col,hot,t,ph)
        ang=-1.25+2.15*p
        px=int(cx+r_planet*math.cos(ang)); py=int(cy+r_planet*math.sin(ang))
        self.draw_planet(image,(px,py),int(31*SCALE),t)
        self.panel(image,(int(W*.15),int(H*.71),int(W*.85),int(H*.82)),168)
        draw_text(image,"PLANET RADIUS",(int(W*.33),int(H*.752)),11,COLORS["muted"]+(220,),True,"ma",1)
        draw_text(image,f"{self.snapshot.planet_barycentric_radius_au:.2f} AU",(int(W*.33),int(H*.790)),22,COLORS["cyan"]+(245,),True,"ma",1)
        draw_text(image,"YEAR",(int(W*.67),int(H*.752)),11,COLORS["muted"]+(220,),True,"ma",1)
        draw_text(image,f"{self.snapshot.planet_period_days:.1f} DAYS",(int(W*.67),int(H*.790)),22,COLORS["gold"]+(245,),True,"ma",1)

    def draw_sky(self, image: Image.Image, t: float, p: float):
        d=ImageDraw.Draw(image)
        horizon=int(H*.67)
        # Atmosphere / horizon
        d.rectangle((0,int(H*.39),W,horizon),fill=(15,45,82,120))
        d.rectangle((0,horizon,W,H),fill=(18,25,31,255))
        # Suns rearrange over the shot
        x_a=int(W*(.23+.08*math.sin(p*math.pi)))
        x_b=int(W*(.50+.13*math.sin(p*math.pi*1.25+1.1)))
        x_c=int(W*(.76+.08*math.cos(p*math.pi*1.4)))
        y_a=int(H*(.39-.06*p)); y_b=int(H*(.44-.10*math.sin(p*math.pi))); y_c=int(H*(.49-.05*math.cos(p*math.pi)))
        self.draw_star(image,(x_a,y_a),int(47*SCALE),COLORS["star_a"],COLORS["star_a_hot"],t)
        self.draw_star(image,(x_b,y_b),int(30*SCALE),COLORS["star_b"],COLORS["star_b_hot"],t,.7)
        self.draw_star(image,(x_c,y_c),int(20*SCALE),COLORS["star_c"],COLORS["star_c_hot"],t,1.4)
        # Marker + three shadows
        ox=W//2; base=horizon-int(4*SCALE)
        d.rectangle((ox-int(8*SCALE),base-int(70*SCALE),ox+int(8*SCALE),base),fill=(42,45,49,255))
        d.ellipse((ox-int(24*SCALE),base-int(112*SCALE),ox+int(24*SCALE),base-int(66*SCALE)),fill=(48,52,57,255))
        for dx,dy,alpha,col in [(150,48,105,(0,0,0)),(-130,40,78,(20,4,2)),(88,58,60,(4,10,20))]:
            d.polygon([(ox-int(8*SCALE),base),(ox+int(8*SCALE),base),(ox+int(dx*SCALE),base+int(dy*SCALE)),(ox+int((dx-22 if dx>0 else dx+22)*SCALE),base+int((dy+12)*SCALE))],fill=col+(alpha,))
        draw_text(image,"THREE MOVING SUNS",(W//2,int(H*.14)),30,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"changing separation • multiple shadows • triple sunsets",(W//2,int(H*.195)),12,COLORS["muted"]+(220,),False,"ma",1)
        draw_text(image,"SKY VIEW IS SCHEMATIC",(W//2,int(H*.76)),11,COLORS["cyan"]+(220,),True,"ma",1)

    def draw_changing_light(self, image: Image.Image, t: float, p: float):
        draw_text(image,"THE LIGHT WOULD NEVER BE SIMPLE",(W//2,int(H*.14)),27,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"three moving sources change distance and alignment",(W//2,int(H*.195)),13,COLORS["muted"]+(220,),False,"ma",1)
        x0=int(W*.11); x1=int(W*.89); y0=int(H*.64); y1=int(H*.31)
        d=ImageDraw.Draw(image)
        d.line((x0,y0,x1,y0),fill=COLORS["white"]+(85,),width=max(2,int(3*SCALE)))
        d.line((x0,y0,x0,y1),fill=COLORS["white"]+(85,),width=max(2,int(3*SCALE)))
        pts=[]; samples=180
        lo=self.snapshot.flux_min_earth_units*.95; hi=self.snapshot.flux_max_earth_units*1.05
        for i in range(samples):
            day=self.snapshot.planet_period_days*2.2*i/(samples-1)
            f=total_flux(day)
            xx=lerp(x0,x1,i/(samples-1)); yy=lerp(y0,y1,clamp((f-lo)/(hi-lo)))
            pts.append((xx,yy))
        reveal=max(2,int(samples*clamp(p)))
        d.line(pts[:reveal],fill=COLORS["cyan"]+(235,),width=max(2,int(5*SCALE)),joint="curve")
        for f,label,col in [
            (self.snapshot.flux_min_earth_units,"MIN",COLORS["blue"]),
            (self.snapshot.flux_mean_earth_units,"MEAN",COLORS["white"]),
            (self.snapshot.flux_max_earth_units,"MAX",COLORS["gold"]),
        ]:
            yy=lerp(y0,y1,clamp((f-lo)/(hi-lo)))
            d.line((x0,yy,x1,yy),fill=col+(35,),width=max(1,int(2*SCALE)))
            draw_text(image,f"{label}  {f:.2f}×",(x1-int(4*SCALE),int(yy-int(9*SCALE))),10,col+(220,),True,"ra",1)
        self.panel(image,(int(W*.14),int(H*.71),int(W*.86),int(H*.83)),170)
        draw_text(image,f"≈ {self.snapshot.flux_peak_to_trough_pct:.1f}% LIGHT SWING",(W//2,int(H*.755)),21,COLORS["gold"]+(245,),True,"ma",1)
        draw_text(image,"TOY SYSTEM • CLIMATE RESPONSE NOT MODELED",(W//2,int(H*.805)),10,COLORS["muted"]+(210,),True,"ma",1)

    def draw_outro(self, image: Image.Image, t: float, p: float):
        draw_text(image,"HIERARCHY MAKES IT POSSIBLE",(W//2,int(H*.14)),28,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"compact stars inside • wide planet outside",(W//2,int(H*.193)),14,COLORS["cyan"]+(225,),True,"ma",1)
        cx=W//2; cy=int(H*.45); r_planet=int(305*SCALE); r_outer=int(74*SCALE)
        d=ImageDraw.Draw(image)
        d.ellipse((cx-r_planet,cy-r_planet,cx+r_planet,cy+r_planet),outline=COLORS["cyan"]+(70,),width=max(2,int(4*SCALE)))
        a,b,c=triple_state(t*8.0); sscale=r_outer/(R_C_OUTER*1.15)
        for pos,r,col,hot,ph in [
            (a,int(23*SCALE),COLORS["star_a"],COLORS["star_a_hot"],0),
            (b,int(17*SCALE),COLORS["star_b"],COLORS["star_b_hot"],.7),
            (c,int(14*SCALE),COLORS["star_c"],COLORS["star_c_hot"],1.4),
        ]:
            self.draw_star(image,(int(cx+pos[0]*sscale),int(cy+pos[1]*sscale)),r,col,hot,t,ph)
        ang=-1.0+p*.9
        self.draw_planet(image,(int(cx+r_planet*math.cos(ang)),int(cy+r_planet*math.sin(ang))),int(29*SCALE),t)
        self.panel(image,(int(W*.12),int(H*.70),int(W*.88),int(H*.84)),176)
        draw_text(image,"3 STARS",(int(W*.27),int(H*.755)),23,COLORS["gold"]+(245,),True,"ma",1)
        draw_text(image,"1 BARYCENTER",(W//2,int(H*.755)),17,COLORS["violet"]+(240,),True,"ma",1)
        draw_text(image,"1 PLANET",(int(W*.74),int(H*.755)),23,COLORS["cyan"]+(245,),True,"ma",1)
        draw_text(image,"REAL TRIPLE SYSTEMS EXIST • THIS ORBIT IS ILLUSTRATIVE",(W//2,int(H*.813)),10,COLORS["muted"]+(220,),True,"ma",1)

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
        if name=="triple_reveal": self.draw_triple_reveal(image,t,p)
        elif name=="hierarchy": self.draw_hierarchy(image,t,p)
        elif name=="circumtriple": self.draw_circumtriple(image,t,p)
        elif name=="sky": self.draw_sky(image,t,p)
        elif name=="changing_light": self.draw_changing_light(image,t,p)
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

def render_preview_frames(scene: ThreeStarsScene) -> List[Path]:
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


def render_video(scene: ThreeStarsScene) -> Path:
    path=OUTPUT_ROOT/f"{CONFIG['basename']}.mp4"
    total=max(1,int(round(DURATION*FPS)))
    writer=iio.get_writer(path,fps=FPS,codec="libx264",quality=7,pixelformat="yuv420p",ffmpeg_log_level="error",macro_block_size=2)
    try:
        for i in tqdm(range(total),desc="Rendering Three-Stars Planet Short"):
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

YOUTUBE_TITLE = 'What If a Planet Orbited THREE Stars? 🌍☀️☀️☀️ #rootjatin'
YOUTUBE_DESCRIPTION = 'What would it be like to live on a planet orbiting three stars? This video explores a hypothetical, scientifically inspired circumtriple system: two stars form a tight inner binary, a third star orbits farther out, and a planet travels on a much wider orbit around their shared center of mass. Triple-star systems are real, but the exact Earth-like planet shown here is an illustrative model rather than a confirmed exoplanet. The geometry is simplified for education and is not a formal N-body stability proof.'
YOUTUBE_HASHTAGS = '#rootjatin #Space #Astronomy #Exoplanets #TripleStarSystem #Science #Universe #WhatIf'

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



