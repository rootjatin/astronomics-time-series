from __future__ import annotations

"""
What Happens When Two Planets Share an Orbit?
==============================================

Output : https://youtube.com/shorts/iXDogbUp2OQ?feature=share

A cinematic vertical YouTube Short renderer about co-orbital dynamics.

The video compares three distinct meanings of "sharing an orbit":
1) Trojan-style 1:1 resonance, where two small planets remain separated by
   roughly 60 degrees around a much more massive star.
2) Horseshoe-style co-orbital motion, visualized in the rotating frame as one
   body slowly approaching, exchanging orbital energy, and retreating.
3) An unsafe close-start case, where two comparable planets are placed nearly
   on top of one another on the same orbital track and strong mutual gravity
   can drive close encounters.

Scientific interpretation
-------------------------
- Sharing a semimajor axis does NOT mean occupying the same place at the same time.
- L4/L5 geometry places co-orbiting bodies about 60 degrees ahead/behind a primary.
- Saturn's Janus and Epimetheus provide a real Solar System example of a 1:1
  co-orbital resonance: their nearby orbits swap inner/outer roles every few years.
- The horseshoe and unsafe-close scenes here are educational schematic animations,
  not a precision long-term N-body prediction for two Earth twins.
- Exact stability depends on masses, eccentricities, inclinations, phase, and
  perturbations from other bodies.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    COORBITAL_SHORT_QUICK=1 python what_happens_when_two_planets_share_an_orbit.py

Full 1080x1920 render
---------------------
    python what_happens_when_two_planets_share_an_orbit.py

4K vertical
-----------
    COORBITAL_SHORT_4K=1 python what_happens_when_two_planets_share_an_orbit.py
"""

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# =============================================================================
# Configuration
# =============================================================================

QUICK_MODE = os.environ.get("COORBITAL_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("COORBITAL_SHORT_4K", "0") == "1" and not QUICK_MODE

W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 6 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SIZE = (W, H)
SCALE = W / 1080.0

OUTPUT_ROOT = Path("what_happens_when_two_planets_share_an_orbit_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "WHAT HAPPENS WHEN TWO PLANETS SHARE AN ORBIT?",
    "subtitle": "1:1 resonance // Trojans // horseshoes // close encounters",
    "basename": "what_happens_when_two_planets_share_an_orbit",
    "contrast": 1.12,
    "saturation": 1.08,
    "vignette": 0.28,
}

COLORS = {
    "space": (2, 5, 16),
    "space2": (8, 18, 40),
    "white": (247, 251, 255),
    "muted": (164, 192, 216),
    "cyan": (75, 229, 255),
    "blue": (76, 139, 255),
    "gold": (255, 207, 90),
    "orange": (255, 142, 74),
    "red": (255, 82, 106),
    "green": (105, 241, 178),
    "violet": (190, 127, 255),
    "magenta": (241, 94, 193),
    "star": (255, 215, 93),
    "star_hot": (255, 249, 221),
    "planet_a": (84, 206, 255),
    "planet_b": (255, 119, 89),
    "panel": (3, 9, 24),
}

SHOT_PLAN = [
    {"name": "reveal", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "trojan", "start": 8.0 if not QUICK_MODE else 1.8, "end": 19.0 if not QUICK_MODE else 4.25},
    {"name": "horseshoe", "start": 19.0 if not QUICK_MODE else 4.25, "end": 31.0 if not QUICK_MODE else 6.95},
    {"name": "swap", "start": 31.0 if not QUICK_MODE else 6.95, "end": 41.5 if not QUICK_MODE else 9.3},
    {"name": "unsafe", "start": 41.5 if not QUICK_MODE else 9.3, "end": 51.5 if not QUICK_MODE else 11.55},
    {"name": "outro", "start": 51.5 if not QUICK_MODE else 11.55, "end": DURATION},
]

CAPTION_TEXTS = [
    "Two planets can share the same orbital period without sharing the same location. The trick is a one-to-one resonance: both worlds go around the star once in the same amount of time.",
    "One surprisingly stable arrangement is Trojan-style motion. If the planets are tiny compared with the star, one can remain about sixty degrees ahead of the other instead of catching it.",
    "Another possibility is a horseshoe orbit. In a frame rotating with one planet, the other appears to trace a giant horseshoe as their gravity prevents a direct pass.",
    "The mechanism is an exchange of orbital energy. The planet that is nudged inward moves faster; the one pushed outward moves slower. Their relative motion reverses before a collision.",
    "But co-orbital does not automatically mean safe. Put two massive planets too close together with the wrong phase, eccentricity, or inclination and close encounters can destroy the neat resonance.",
    "So two planets can share an orbit — but only when gravity organizes the traffic. Same year does not mean same place, and resonance can be the difference between a dance and a crash.",
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
# Science model
# =============================================================================

STAR_MASS_SOLAR = 1.0
PLANET_MASS_EARTH = 1.0
EARTH_MASS_SOLAR = 3.003e-6
A_AU = 1.0
YEAR_DAYS = 365.256
TROJAN_SEPARATION_DEG = 60.0
TROJAN_CHORD_AU = 2.0 * A_AU * math.sin(math.radians(TROJAN_SEPARATION_DEG / 2.0))
MUTUAL_HILL_AU = ((2.0 * EARTH_MASS_SOLAR) / (3.0 * STAR_MASS_SOLAR)) ** (1.0 / 3.0) * A_AU
TROJAN_DISTANCE_HILL = TROJAN_CHORD_AU / MUTUAL_HILL_AU

# NASA's Janus/Epimetheus overview values, used only as an observational analogy.
JANUS_EPIMETHEUS_ORBIT_KM = 151_000.0
JANUS_EPIMETHEUS_RADIAL_GAP_KM = 50.0
JANUS_EPIMETHEUS_CLOSEST_KM = 15_000.0
JANUS_EPIMETHEUS_SWAP_YEARS = 4.0


@dataclass
class CoorbitalSnapshot:
    generated_at_utc: str
    toy_star_mass_solar: float
    toy_planet_mass_earth_each: float
    toy_orbit_au: float
    toy_orbital_period_days: float
    trojan_separation_deg: float
    trojan_chord_distance_au: float
    mutual_hill_radius_au: float
    trojan_chord_in_mutual_hill_radii: float
    janus_epimetheus_orbit_km_approx: float
    janus_epimetheus_radial_gap_km_approx: float
    janus_epimetheus_closest_approach_km_approx: float
    janus_epimetheus_swap_interval_years_approx: float
    interpretation: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def snapshot() -> CoorbitalSnapshot:
    return CoorbitalSnapshot(
        generated_at_utc=iso_z(utc_now()),
        toy_star_mass_solar=STAR_MASS_SOLAR,
        toy_planet_mass_earth_each=PLANET_MASS_EARTH,
        toy_orbit_au=A_AU,
        toy_orbital_period_days=YEAR_DAYS,
        trojan_separation_deg=TROJAN_SEPARATION_DEG,
        trojan_chord_distance_au=TROJAN_CHORD_AU,
        mutual_hill_radius_au=MUTUAL_HILL_AU,
        trojan_chord_in_mutual_hill_radii=TROJAN_DISTANCE_HILL,
        janus_epimetheus_orbit_km_approx=JANUS_EPIMETHEUS_ORBIT_KM,
        janus_epimetheus_radial_gap_km_approx=JANUS_EPIMETHEUS_RADIAL_GAP_KM,
        janus_epimetheus_closest_approach_km_approx=JANUS_EPIMETHEUS_CLOSEST_KM,
        janus_epimetheus_swap_interval_years_approx=JANUS_EPIMETHEUS_SWAP_YEARS,
        interpretation=(
            "Trojan and horseshoe scenes are educational 1:1-resonance illustrations. "
            "Janus/Epimetheus numbers are a real-moon analogy; unsafe close-start behavior "
            "is schematic rather than a precision two-Earth N-body chronology."
        ),
    )


def scenario_table() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "scenario": "Trojan-style",
            "period_ratio": "1:1",
            "relative_geometry": "about 60 degrees apart",
            "visual_status": "idealized stable resonance",
            "notes": "two Earth-mass toy planets around a much more massive Sun-like star",
        },
        {
            "scenario": "Horseshoe-style",
            "period_ratio": "1:1",
            "relative_geometry": "large horseshoe in rotating frame",
            "visual_status": "schematic co-orbital resonance",
            "notes": "inspired by the Janus/Epimetheus exchange mechanism",
        },
        {
            "scenario": "Unsafe close start",
            "period_ratio": "near 1:1",
            "relative_geometry": "small angular separation",
            "visual_status": "illustrative close-encounter risk",
            "notes": "exact outcome depends on masses, phase, eccentricity, inclination, perturbations",
        },
    ])


# =============================================================================
# Helpers
# =============================================================================

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(x: float) -> float:
    t = clamp(x)
    return t * t * (3.0 - 2.0 * t)


def ease_in_out_sine(x: float) -> float:
    t = clamp(x)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


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


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=max(7, int(size * SCALE)))
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(image: Image.Image, value: str, xy: Tuple[int, int], size: int,
              fill=(255, 255, 255, 255), bold: bool = False, anchor: str = "la", stroke: int = 2):
    ImageDraw.Draw(image).text(
        xy, value, font=get_font(size, bold), fill=fill, anchor=anchor,
        stroke_width=max(1, int(stroke * SCALE)), stroke_fill=(0, 0, 0, 220)
    )


def draw_wrapped_text(image: Image.Image, value: str, xy: Tuple[int, int], max_width: int,
                      size: int, fill=(255, 255, 255, 245), bold: bool = False, spacing: int = 6):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold)
    words = value.split()
    lines: List[str] = []
    cur = ""
    for word in words:
        test = word if not cur else cur + " " + word
        box = draw.textbbox((0, 0), test, font=font, stroke_width=max(1, int(2 * SCALE)))
        if box[2] - box[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill,
                  stroke_width=max(1, int(2 * SCALE)), stroke_fill=(0, 0, 0, 220))
        box = draw.textbbox((x, y), line, font=font, stroke_width=max(1, int(2 * SCALE)))
        y += (box[3] - box[1]) + int(spacing * SCALE)


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000; ms %= 3_600_000
    m = ms // 60_000; ms %= 60_000
    s = ms // 1000; ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path) -> Path:
    lines: List[str] = []
    for i, (a, b, text) in enumerate(CAPTIONS, 1):
        lines.extend([str(i), f"{format_srt_time(a)} --> {format_srt_time(b)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    rr = np.sqrt(nx * nx + ny * ny)
    return np.clip(1 - strength * rr ** 1.8, 0, 1).astype(np.float32)


VIGNETTE = make_vignette(W, H, float(CONFIG["vignette"]))


# =============================================================================
# Scene renderer
# =============================================================================

class CoorbitalScene:
    def __init__(self, snap: CoorbitalSnapshot):
        self.snap = snap
        rng = np.random.default_rng(1147)
        self.bg_stars = [
            (float(rng.uniform(0, W)), float(rng.uniform(0, H)), float(rng.uniform(.4, 2.0) * SCALE),
             int(rng.uniform(12, 75)), float(rng.uniform(0, math.tau)))
            for _ in range(150 if QUICK_MODE else 430)
        ]
        self.base_bg = self._make_background()

    def _make_background(self) -> Image.Image:
        top = np.asarray(COLORS["space"], dtype=np.float32)
        bottom = np.asarray(COLORS["space2"], dtype=np.float32)
        f = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        rgb = (top[None, :] * (1 - f) + bottom[None, :] * f).astype(np.uint8)
        arr = np.empty((H, W, 4), dtype=np.uint8)
        arr[..., :3] = rgb[:, None, :]
        arr[..., 3] = 255
        return Image.fromarray(arr, "RGBA")

    def background(self, t: float) -> Image.Image:
        img = self.base_bg.copy()
        d = ImageDraw.Draw(img)
        for x, y, r, a, p in self.bg_stars:
            aa = int(a * (.78 + .22 * math.sin(t * .9 + p)))
            d.ellipse((x - r, y - r, x + r, y + r), fill=COLORS["white"] + (aa,))
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for cx, cy, col in [(W*.22, H*.26, COLORS["blue"]), (W*.78, H*.30, COLORS["violet"]), (W*.52, H*.72, COLORS["magenta"])]:
            rr = W * .27
            gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=col + (12,))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(18, int(55*SCALE)))))
        return img

    @staticmethod
    def panel(img: Image.Image, box: Tuple[int, int, int, int], alpha: int = 176):
        layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle(box, radius=max(8, int(24*SCALE)), fill=COLORS["panel"] + (alpha,),
                            outline=COLORS["cyan"] + (48,), width=max(1, int(2*SCALE)))
        img.alpha_composite(layer)

    def header(self, img: Image.Image, t: float):
        if t < (6.0 if not QUICK_MODE else 1.35):
            f = smoothstep(t / (.75 if not QUICK_MODE else .18))
            draw_text(img, "WHAT HAPPENS WHEN", (W//2, int(H*.050)), 31 if not QUICK_MODE else 15,
                      COLORS["white"] + (int(245*f),), True, "ma", 2)
            draw_text(img, "TWO PLANETS", (W//2, int(H*.094)), 52 if not QUICK_MODE else 26,
                      COLORS["cyan"] + (int(250*f),), True, "ma", 2)
            draw_text(img, "SHARE AN ORBIT?", (W//2, int(H*.142)), 39 if not QUICK_MODE else 19,
                      COLORS["gold"] + (int(245*f),), True, "ma", 2)
        else:
            labels = {
                "reveal": "SAME YEAR DOES NOT MEAN SAME PLACE",
                "trojan": "OPTION 1 // TROJAN 1:1 RESONANCE",
                "horseshoe": "OPTION 2 // HORSESHOE CO-ORBITAL",
                "swap": "GRAVITY CAN REVERSE THE CATCH-UP",
                "unsafe": "CO-ORBITAL DOES NOT AUTOMATICALLY MEAN SAFE",
                "outro": "RESONANCE ORGANIZES THE TRAFFIC",
            }
            draw_text(img, labels[get_shot(t)["name"]], (int(W*.055), int(H*.044)),
                      17 if not QUICK_MODE else 8, COLORS["muted"] + (230,), True, "la", 1)
        draw_text(img, "1:1 RESONANCE // EDUCATIONAL SCHEMATIC", (int(W*.945), int(H*.047)),
                  11 if not QUICK_MODE else 5, COLORS["cyan"] + (205,), True, "ra", 1)

    def caption(self, img: Image.Image, t: float):
        cap = caption_at(t)
        if not cap:
            return
        y0 = H - int(186*SCALE)
        self.panel(img, (int(58*SCALE), y0, W-int(58*SCALE), y0+int(116*SCALE)), 150)
        draw_wrapped_text(img, cap, (int(82*SCALE), y0+int(18*SCALE)), W-int(164*SCALE),
                          30, COLORS["white"] + (240,), False, 4)

    def draw_star(self, img: Image.Image, cx: float, cy: float, r: float = 25):
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for q, a in [(76, 19), (54, 34), (36, 70)]:
            rr = q*SCALE
            gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=COLORS["star"] + (a,))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4, int(12*SCALE)))))
        d = ImageDraw.Draw(img)
        rr = r*SCALE
        d.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=COLORS["star_hot"] + (255,))

    def draw_planet(self, img: Image.Image, x: float, y: float, color: Tuple[int,int,int], r: float = 12, label: Optional[str] = None):
        glow = Image.new("RGBA", SIZE, (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        rr = r*2.2*SCALE
        gd.ellipse((x-rr,y-rr,x+rr,y+rr), fill=color+(40,))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(2,int(5*SCALE)))))
        d = ImageDraw.Draw(img)
        rp = r*SCALE
        d.ellipse((x-rp,y-rp,x+rp,y+rp), fill=color+(255,), outline=COLORS["white"]+(170,), width=max(1,int(2*SCALE)))
        if label:
            draw_text(img, label, (int(x), int(y-rp-14*SCALE)), 13 if not QUICK_MODE else 6, color+(245,), True, "ma", 1)

    def orbit_geometry(self, yfrac: float = .46, radius_frac: float = .30):
        return W*.5, H*yfrac, min(W, H*.62)*radius_frac

    def draw_orbit(self, img: Image.Image, cx: float, cy: float, r: float, alpha: int = 80, width: int = 3):
        d = ImageDraw.Draw(img)
        d.ellipse((cx-r,cy-r,cx+r,cy+r), outline=COLORS["white"]+(alpha,), width=max(1,int(width*SCALE)))

    def draw_reveal(self, img: Image.Image, t: float, local: float):
        cx, cy, r = self.orbit_geometry(.47,.31)
        self.draw_orbit(img,cx,cy,r,90,3)
        self.draw_star(img,cx,cy,27)
        ang = -math.pi/2 + math.tau*(.12*local)
        sep = math.radians(28 + 32*smoothstep(local))
        p1=(cx+r*math.cos(ang), cy+r*math.sin(ang))
        p2=(cx+r*math.cos(ang+sep), cy+r*math.sin(ang+sep))
        self.draw_planet(img,*p1,COLORS["planet_a"],13,"A")
        self.draw_planet(img,*p2,COLORS["planet_b"],13,"B")
        self.panel(img,(int(W*.18),int(H*.69),int(W*.82),int(H*.79)),168)
        draw_text(img,"SAME ORBITAL PERIOD",(W//2,int(H*.718)),24 if not QUICK_MODE else 12,COLORS["cyan"]+(250,),True,"ma",1)
        draw_text(img,"≠ SAME POSITION",(W//2,int(H*.758)),20 if not QUICK_MODE else 10,COLORS["gold"]+(245,),True,"ma",1)

    def draw_trojan(self, img: Image.Image, t: float, local: float):
        cx,cy,r=self.orbit_geometry(.46,.30)
        self.draw_orbit(img,cx,cy,r,85,3); self.draw_star(img,cx,cy,25)
        base=-math.pi/2+math.tau*.17*local
        a1=base; a2=base+math.radians(60)
        p1=(cx+r*math.cos(a1),cy+r*math.sin(a1)); p2=(cx+r*math.cos(a2),cy+r*math.sin(a2))
        self.draw_planet(img,*p1,COLORS["planet_a"],13,"A")
        self.draw_planet(img,*p2,COLORS["planet_b"],13,"B")
        d=ImageDraw.Draw(img)
        # arc showing 60 degrees
        box=(cx-r*.58,cy-r*.58,cx+r*.58,cy+r*.58)
        start=math.degrees(a1); end=math.degrees(a2)
        d.arc(box,start,end,fill=COLORS["green"]+(235,),width=max(2,int(8*SCALE)))
        mid=(a1+a2)/2
        tx=cx+r*.67*math.cos(mid); ty=cy+r*.67*math.sin(mid)
        draw_text(img,"60°",(int(tx),int(ty)),25 if not QUICK_MODE else 12,COLORS["green"]+(250,),True,"ma",1)
        self.panel(img,(int(W*.15),int(H*.68),int(W*.85),int(H*.79)),168)
        draw_text(img,"TROJAN-STYLE 1:1 RESONANCE",(W//2,int(H*.713)),22 if not QUICK_MODE else 11,COLORS["green"]+(248,),True,"ma",1)
        draw_text(img,f"toy chord separation = {TROJAN_CHORD_AU:.2f} AU",(W//2,int(H*.754)),14 if not QUICK_MODE else 7,COLORS["white"]+(220,),True,"ma",1)

    def horseshoe_path(self, phase: float) -> Tuple[float,float]:
        # A schematic horseshoe in the frame rotating with Planet A.
        # The path avoids the planet near x>0 and reverses around the ends.
        u = math.tau*phase
        x = -0.08 + 0.78*math.cos(u)
        y = 0.50*math.sin(u)*(0.45 + 0.55*(1-math.cos(u))*.5)
        # carve a gap around +x (reference planet) by bending path vertically
        bump = math.exp(-((math.cos(u)-1.0)/0.22)**2)
        y += 0.20*math.sin(u*0.5)*bump
        return x,y

    def draw_horseshoe(self, img: Image.Image, t: float, local: float):
        cx=W*.5; cy=H*.47; rx=W*.34; ry=H*.20
        self.panel(img,(int(W*.09),int(H*.22),int(W*.91),int(H*.70)),150)
        d=ImageDraw.Draw(img)
        # reference star at center, planet A fixed at right in rotating frame
        self.draw_star(img,cx,cy,21)
        refx=cx+rx*.72; refy=cy
        self.draw_planet(img,refx,refy,COLORS["planet_a"],12,"A")
        pts=[]
        for q in np.linspace(0,1,260):
            x,y=self.horseshoe_path(float(q))
            pts.append((cx+x*rx,cy+y*ry))
        d.line(pts,fill=COLORS["violet"]+(190,),width=max(2,int(5*SCALE)),joint="curve")
        phase=(local*.86+.07)%1.0
        x,y=self.horseshoe_path(phase)
        bx=cx+x*rx; by=cy+y*ry
        self.draw_planet(img,bx,by,COLORS["planet_b"],12,"B")
        draw_text(img,"ROTATING FRAME",(W//2,int(H*.255)),16 if not QUICK_MODE else 8,COLORS["muted"]+(230,),True,"ma",1)
        draw_text(img,"HORSESHOE PATH",(W//2,int(H*.645)),24 if not QUICK_MODE else 12,COLORS["violet"]+(248,),True,"ma",1)
        draw_text(img,"the planets need not pass through each other",(W//2,int(H*.685)),13 if not QUICK_MODE else 6,COLORS["white"]+(210,),True,"ma",1)

    def draw_swap(self, img: Image.Image, t: float, local: float):
        cx,cy,r=self.orbit_geometry(.45,.30); d=ImageDraw.Draw(img)
        self.draw_star(img,cx,cy,24)
        # Inner/outer tracks and a catch-up / swap animation.
        r0=r*.93; r1=r*1.07
        d.ellipse((cx-r0,cy-r0,cx+r0,cy+r0),outline=COLORS["cyan"]+(55,),width=max(1,int(2*SCALE)))
        d.ellipse((cx-r1,cy-r1,cx+r1,cy+r1),outline=COLORS["orange"]+(55,),width=max(1,int(2*SCALE)))
        s=ease_in_out_sine(local)
        # before midpoint A is inner; after midpoint B is inner
        mix=smoothstep((local-.38)/.24)
        ra=lerp(r0,r1,mix); rb=lerp(r1,r0,mix)
        # B is initially ahead; A catches up then both separate after swap.
        sep_deg=lerp(36,8,smoothstep(local/.48)) if local<.5 else lerp(8,38,smoothstep((local-.5)/.5))
        base=-math.pi/2+math.tau*.28*s
        aa=base; ab=base+math.radians(sep_deg)
        pa=(cx+ra*math.cos(aa),cy+ra*math.sin(aa)); pb=(cx+rb*math.cos(ab),cy+rb*math.sin(ab))
        self.draw_planet(img,*pa,COLORS["planet_a"],12,"A")
        self.draw_planet(img,*pb,COLORS["planet_b"],12,"B")
        # arrows: inner = faster, outer = slower
        draw_text(img,"INNER → FASTER",(int(W*.18),int(H*.69)),17 if not QUICK_MODE else 8,COLORS["cyan"]+(242,),True,"la",1)
        draw_text(img,"OUTER → SLOWER",(int(W*.82),int(H*.69)),17 if not QUICK_MODE else 8,COLORS["orange"]+(242,),True,"ra",1)
        self.panel(img,(int(W*.13),int(H*.725),int(W*.87),int(H*.80)),166)
        draw_text(img,"GRAVITY EXCHANGES ORBITAL ENERGY",(W//2,int(H*.752)),20 if not QUICK_MODE else 10,COLORS["gold"]+(248,),True,"ma",1)
        draw_text(img,"Janus + Epimetheus do a real version of this",(W//2,int(H*.783)),12 if not QUICK_MODE else 6,COLORS["white"]+(210,),True,"ma",1)

    def draw_unsafe(self, img: Image.Image, t: float, local: float):
        cx,cy,r=self.orbit_geometry(.46,.30); self.draw_star(img,cx,cy,24); self.draw_orbit(img,cx,cy,r,70,2)
        d=ImageDraw.Draw(img)
        # start almost together; then kick trajectories apart to illustrate risk
        s=smoothstep(local)
        base=-math.pi/2+math.tau*.17*local
        sep=math.radians(lerp(9,3,smoothstep(local/.35)))
        kick=smoothstep((local-.35)/.65)
        ra=r*(1-.16*kick); rb=r*(1+.20*kick)
        aa=base-.18*kick; ab=base+sep+.28*kick
        pa=(cx+ra*math.cos(aa),cy+ra*math.sin(aa)); pb=(cx+rb*math.cos(ab),cy+rb*math.sin(ab))
        # risk zone
        midx=(pa[0]+pb[0])/2; midy=(pa[1]+pb[1])/2
        q=lerp(16,70,kick)*SCALE
        layer=Image.new("RGBA",SIZE,(0,0,0,0)); ld=ImageDraw.Draw(layer)
        ld.ellipse((midx-q,midy-q,midx+q,midy+q),fill=COLORS["red"]+(42,))
        img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(2,int(10*SCALE)))))
        self.draw_planet(img,*pa,COLORS["planet_a"],13,"A")
        self.draw_planet(img,*pb,COLORS["planet_b"],13,"B")
        if kick>.1:
            d.arc((cx-ra,cy-ra,cx+ra,cy+ra),210,332,fill=COLORS["cyan"]+(130,),width=max(2,int(4*SCALE)))
            d.arc((cx-rb,cy-rb,cx+rb,cy+rb),198,320,fill=COLORS["orange"]+(130,),width=max(2,int(4*SCALE)))
        self.panel(img,(int(W*.14),int(H*.69),int(W*.86),int(H*.80)),170)
        draw_text(img,"WRONG PHASE + STRONG MUTUAL GRAVITY",(W//2,int(H*.720)),19 if not QUICK_MODE else 9,COLORS["red"]+(250,),True,"ma",1)
        draw_text(img,"close encounters can break the resonance",(W//2,int(H*.759)),15 if not QUICK_MODE else 7,COLORS["white"]+(220,),True,"ma",1)
        draw_text(img,"ILLUSTRATIVE — NOT A PRECISE COLLISION TIMELINE",(W//2,int(H*.790)),10 if not QUICK_MODE else 5,COLORS["muted"]+(205,),True,"ma",1)

    def draw_outro(self, img: Image.Image, t: float, local: float):
        self.panel(img,(int(W*.09),int(H*.20),int(W*.91),int(H*.70)),174)
        rows=[
            ("TROJAN", "~60° APART", "CAN BE STABLE", COLORS["green"]),
            ("HORSESHOE", "1:1 RESONANCE", "CAN AVOID COLLISION", COLORS["violet"]),
            ("TOO CLOSE", "WRONG SETUP", "CAN DESTABILIZE", COLORS["red"]),
        ]
        y=int(H*.29)
        for i,(name,geom,status,col) in enumerate(rows):
            yy=y+i*int(118*SCALE)
            draw_text(img,name,(int(W*.16),yy),21 if not QUICK_MODE else 10,col+(248,),True,"la",1)
            draw_text(img,geom,(int(W*.84),yy),15 if not QUICK_MODE else 7,COLORS["white"]+(225,),True,"ra",1)
            draw_text(img,status,(int(W*.16),yy+int(39*SCALE)),14 if not QUICK_MODE else 7,COLORS["muted"]+(225,),False,"la",1)
            ImageDraw.Draw(img).line((int(W*.16),yy+int(72*SCALE),int(W*.84),yy+int(72*SCALE)),fill=COLORS["white"]+(25,),width=1)
        draw_text(img,"SAME YEAR ≠ SAME PLACE",(W//2,int(H*.724)),26 if not QUICK_MODE else 13,COLORS["gold"]+(250,),True,"ma",1)
        draw_text(img,"RESONANCE IS THE TRAFFIC CONTROL",(W//2,int(H*.765)),17 if not QUICK_MODE else 8,COLORS["cyan"]+(240,),True,"ma",1)

    def render(self, t: float) -> np.ndarray:
        img=self.background(t)
        shot=get_shot(t)
        local=clamp((t-shot["start"])/max(1e-9,shot["end"]-shot["start"]))
        self.header(img,t)
        if shot["name"]=="reveal": self.draw_reveal(img,t,local)
        elif shot["name"]=="trojan": self.draw_trojan(img,t,local)
        elif shot["name"]=="horseshoe": self.draw_horseshoe(img,t,local)
        elif shot["name"]=="swap": self.draw_swap(img,t,local)
        elif shot["name"]=="unsafe": self.draw_unsafe(img,t,local)
        else: self.draw_outro(img,t,local)
        self.caption(img,t)
        overlay=Image.new("RGBA",SIZE,(0,0,0,0)); od=ImageDraw.Draw(overlay); off=int((t*31)%9)
        for yy in range(off,H,9): od.line((0,yy,W,yy),fill=(120,205,240,7),width=1)
        img.alpha_composite(overlay)
        arr=np.asarray(img.convert("RGB")).astype(np.float32); arr*=VIGNETTE[...,None]; arr=np.clip(arr,0,255).astype(np.uint8)
        graded=Image.fromarray(arr); graded=ImageEnhance.Contrast(graded).enhance(float(CONFIG["contrast"])); graded=ImageEnhance.Color(graded).enhance(float(CONFIG["saturation"]))
        return np.asarray(graded)


# =============================================================================
# Outputs
# =============================================================================

def save_data(snap: CoorbitalSnapshot) -> Tuple[Path, Path]:
    csv_path=DATA_ROOT/"coorbital_scenarios.csv"
    json_path=DATA_ROOT/"coorbital_snapshot.json"
    scenario_table().to_csv(csv_path,index=False)
    json_path.write_text(json.dumps({
        "snapshot":asdict(snap),
        "source_notes":{
            "trojans":"L4/L5 objects lead or trail a primary by about 60 degrees in the standard restricted three-body picture.",
            "janus_epimetheus":"NASA reports a real 1:1 co-orbital resonance around Saturn; the two moons swap the inner/outer orbit every few years.",
            "visualization":"horseshoe and unsafe-close sequences are schematic educational graphics, not an ephemeris or long-term N-body forecast.",
        }
    },indent=2),encoding="utf-8")
    return csv_path,json_path


def save_preview_frames(scene: CoorbitalScene) -> List[Path]:
    paths=[]
    for i,shot in enumerate(SHOT_PLAN,1):
        t=(shot["start"]+shot["end"])*.5
        arr=scene.render(min(DURATION-1/FPS,t))
        p=PREVIEW_ROOT/f"preview_{i:02d}_{shot['name']}.jpg"
        Image.fromarray(arr).save(p,quality=92)
        paths.append(p)
    return paths


def make_contact_sheet(paths: Sequence[Path]) -> Path:
    tw=300 if not QUICK_MODE else 200; th=int(tw*16/9); thumbs=[]
    for p in paths:
        im=Image.open(p).convert("RGB"); im.thumbnail((tw,th),Image.Resampling.LANCZOS)
        c=Image.new("RGB",(tw,th),(5,8,18)); c.paste(im,((tw-im.width)//2,(th-im.height)//2)); thumbs.append(c)
    cols=3; rows=math.ceil(len(thumbs)/cols); sheet=Image.new("RGB",(tw*cols,th*rows),(3,6,15))
    for i,im in enumerate(thumbs): sheet.paste(im,((i%cols)*tw,(i//cols)*th))
    out=PREVIEW_ROOT/f"{CONFIG['basename']}_contact_sheet.jpg"; sheet.save(out,quality=92); return out


def render_video(scene: CoorbitalScene) -> Path:
    out=OUTPUT_ROOT/f"{CONFIG['basename']}{'_quick_preview' if QUICK_MODE else ''}.mp4"
    total=max(1,int(round(DURATION*FPS)))
    writer=iio.get_writer(out,fps=FPS,codec="libx264",quality=8,macro_block_size=None,
                          output_params=["-pix_fmt","yuv420p","-movflags","+faststart"])
    try:
        for frame in tqdm(range(total),desc="Rendering co-orbital short"):
            writer.append_data(scene.render(frame/FPS))
    finally:
        writer.close()
    return out


def write_readme(snap: CoorbitalSnapshot, outputs: Dict[str,str]) -> Path:
    path=OUTPUT_ROOT/"README.txt"
    lines=[
        CONFIG["title"],"="*len(CONFIG["title"]),"",
        "Educational co-orbital renderer: Trojan, horseshoe, and unsafe-close cases.","",
        f"Toy star mass: {snap.toy_star_mass_solar:.2f} solar masses",
        f"Toy planets: {snap.toy_planet_mass_earth_each:.1f} Earth mass each",
        f"Toy orbit: {snap.toy_orbit_au:.2f} AU / {snap.toy_orbital_period_days:.3f} days",
        f"Trojan separation: {snap.trojan_separation_deg:.1f} degrees",
        f"Trojan chord: {snap.trojan_chord_distance_au:.3f} AU",
        f"Mutual Hill radius: {snap.mutual_hill_radius_au:.5f} AU",
        f"Trojan chord / mutual Hill radius: {snap.trojan_chord_in_mutual_hill_radii:.1f}","",
        "Janus/Epimetheus analogy:",
        f"- Saturn distance: ~{snap.janus_epimetheus_orbit_km_approx:,.0f} km",
        f"- radial orbit difference: ~{snap.janus_epimetheus_radial_gap_km_approx:.0f} km",
        f"- closest approach: ~{snap.janus_epimetheus_closest_approach_km_approx:,.0f} km",
        f"- exchange cadence: ~{snap.janus_epimetheus_swap_interval_years_approx:.0f} Earth years","",
        "Important: horseshoe and unsafe-close scenes are schematic, not precision N-body forecasts.","",
        "Outputs:",
    ]
    lines.extend([f"- {k}: {v}" for k,v in outputs.items()])
    path.write_text("\n".join(lines),encoding="utf-8")
    return path



# =============================================================================
# YouTube title / description sidecar
# =============================================================================

YOUTUBE_TITLE = 'What Happens When Two Planets Share an Orbit? 🪐🪐 #rootjatin'
YOUTUBE_DESCRIPTION = "Two worlds can have the same orbital period without occupying the same place. This educational simulation explores co-orbital motion such as Trojan-style configurations and horseshoe behavior, where gravity and a 1:1 resonance can organize the planets' relative motion. The examples are illustrative; stability depends on mass, phase, eccentricity, inclination, and the wider system."
YOUTUBE_HASHTAGS = '#rootjatin #Planets #Orbit #Resonance #Astronomy #Space #OrbitalMechanics #Science'

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

def main():
    snap=snapshot()
    csv_path,json_path=save_data(snap)
    srt_path=write_srt(OUTPUT_ROOT/f"{CONFIG['basename']}_subtitles.srt")
    scene=CoorbitalScene(snap)
    previews=save_preview_frames(scene)
    contact=make_contact_sheet(previews)
    video=render_video(scene)
    outputs={
        "video":str(video),
        "subtitles":str(srt_path),
        "contact_sheet":str(contact),
        "scenario_csv":str(csv_path),
        "snapshot_json":str(json_path),
    }
    outputs["readme"]=str(write_readme(snap,outputs))
    print(json.dumps({"snapshot":asdict(snap),"outputs":outputs},indent=2))
    metadata_txt = write_youtube_metadata_txt()
    print("Title/description TXT:", metadata_txt.resolve())


if __name__=="__main__":
    main()
