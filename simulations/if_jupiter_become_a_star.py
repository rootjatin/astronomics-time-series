from __future__ import annotations

"""
What If Jupiter Became a Star?
==============================

A cinematic vertical YouTube Short renderer for a physics-first thought
experiment: what if Jupiter somehow gained enough mass to become a true star?

Scientific framing
------------------
- Jupiter cannot ignite sustained hydrogen fusion at its present mass.
- NASA commonly describes brown dwarfs as roughly 13-80 Jupiter masses and
  notes that an object above about 80 Jupiter masses can cross into the stellar
  regime. The exact hydrogen-burning boundary depends on composition.
- NASA also notes that Jupiter would have needed about 80 times its present mass
  to become a red-dwarf star rather than a planet.
- In this renderer, "Jupiter becomes a star" therefore means a hypothetical
  transformation to ~80 Jupiter masses (~0.076 solar masses), while its orbital
  distance initially remains near Jupiter's current ~5.2 AU. That is not a
  physically realistic way to create a star; it is a controlled thought experiment.
- A companion that massive would no longer be a small perturbing planet. The Sun
  and new low-mass star would orbit a barycenter far from the Sun's center, and
  the architecture of the present Solar System would be strongly altered.
- The animation does NOT claim that Earth would instantly be destroyed or that
  every orbit becomes unstable. Exact long-term outcomes depend on how the mass
  appeared, orbital eccentricities, inclinations, and N-body dynamics.
- A near-threshold red dwarf would be far fainter and cooler than the Sun. This
  renderer intentionally avoids assigning a precise luminosity or Earth-sky
  brightness because those depend on the final stellar mass, age, composition,
  and the varying Earth-Jupiter distance.

Primary sources
---------------
NASA — What Is Jupiter?:
    https://www.nasa.gov/solar-system/what-is-jupiter-grades-5-8/
NASA Science — Massive Beauty:
    https://science.nasa.gov/photojournal/massive-beauty/
NASA/JPL — What is a Brown Dwarf?:
    https://www.jpl.nasa.gov/images/pia23685-what-is-a-brown-dwarf/
NASA Science — Types of Stars:
    https://science.nasa.gov/universe/stars/types/
NASA NSSDC — Jupiter Fact Sheet:
    https://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html

Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    JUPITER_STAR_SHORT_QUICK=1 python what_if_jupiter_became_a_star.py

Full 1080x1920 render
---------------------
    python what_if_jupiter_became_a_star.py

4K vertical render
------------------
    JUPITER_STAR_SHORT_4K=1 python what_if_jupiter_became_a_star.py
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

QUICK_MODE = os.environ.get("JUPITER_STAR_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("JUPITER_STAR_SHORT_4K", "0") == "1" and not QUICK_MODE

W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 6 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SIZE = (W, H)
SCALE = W / 1080.0

OUTPUT_ROOT = Path("what_if_jupiter_became_a_star_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
for d in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT):
    d.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "WHAT IF JUPITER BECAME A STAR?",
    "subtitle": "A low-mass-star thought experiment at 5.2 AU",
    "basename": "what_if_jupiter_became_a_star",
    "contrast": 1.10,
    "saturation": 1.08,
    "vignette": 0.26,
}

COLORS = {
    "space": (2, 5, 15),
    "space2": (11, 10, 28),
    "white": (248, 251, 255),
    "muted": (166, 190, 213),
    "cyan": (78, 225, 255),
    "blue": (77, 139, 255),
    "gold": (255, 205, 86),
    "orange": (255, 139, 68),
    "red": (255, 83, 92),
    "deep_red": (126, 28, 38),
    "violet": (177, 122, 255),
    "green": (105, 239, 174),
    "jupiter_light": (225, 188, 137),
    "jupiter_dark": (132, 85, 57),
    "jupiter_cream": (241, 216, 176),
    "sun": (255, 229, 132),
    "earth": (57, 145, 222),
    "mars": (197, 89, 61),
    "panel": (3, 9, 21),
}

SHOT_PLAN = [
    {"name": "impossible_now", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "mass_ladder", "start": 8.0 if not QUICK_MODE else 1.8, "end": 18.5 if not QUICK_MODE else 4.1},
    {"name": "binary_gravity", "start": 18.5 if not QUICK_MODE else 4.1, "end": 30.0 if not QUICK_MODE else 6.6},
    {"name": "second_sun", "start": 30.0 if not QUICK_MODE else 6.6, "end": 40.5 if not QUICK_MODE else 8.95},
    {"name": "moons", "start": 40.5 if not QUICK_MODE else 8.95, "end": 51.0 if not QUICK_MODE else 11.3},
    {"name": "outro", "start": 51.0 if not QUICK_MODE else 11.3, "end": DURATION},
]

CAPTION_TEXTS = [
    "First, the catch: Jupiter cannot simply switch on like a light bulb. At its current mass it never reaches the core pressure needed for sustained hydrogen fusion.",
    "To cross into the stellar regime, Jupiter would need roughly eighty times its present mass. Below that, objects in the middle are generally called brown dwarfs — not true hydrogen-burning stars.",
    "At about eighty Jupiter masses, the new object would be roughly eight percent of the Sun's mass. At Jupiter's present distance, the Sun and this red dwarf would behave much more like a close stellar pair than a star with one giant planet.",
    "The new star would be cooler and far fainter than the Sun, but it would still produce its own light. From Earth, the exact brightness would depend on its final mass, age, and where Earth and Jupiter-star were in their orbits.",
    "Jupiter's moons would now be orbiting a low-mass star. Their exact futures would depend on how the transformation happened, but the entire Jovian system would stop being a simple planet-and-moons system.",
    "The biggest change is gravity. A stellar-mass Jupiter would strongly reshape the Solar System, especially the outer planets and small-body populations. This is a thought experiment — not an instant-destruction prediction.",
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
# Data model and helpers
# =============================================================================

@dataclass
class JupiterStarSnapshot:
    generated_at_utc: str
    jupiter_mass_kg: float
    jupiter_mass_solar: float
    jupiter_orbit_au: float
    assumed_stellar_threshold_jupiter_masses: float
    assumed_new_star_mass_solar: float
    approximate_sun_barycenter_radius_au: float
    brown_dwarf_range_jupiter_masses: str
    scenario: str
    interpretation: str
    nasa_jupiter_url: str
    nasa_brown_dwarf_url: str
    nasa_star_types_url: str
    nssdc_jupiter_url: str


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


def get_font(size: int, bold: bool = False):
    px = max(7, int(size * SCALE))
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(image: Image.Image, value: str, xy: Tuple[int, int], size: int,
              fill=(255, 255, 255, 255), bold: bool = False,
              anchor: str = "la", stroke: int = 2):
    ImageDraw.Draw(image).text(
        xy, value, font=get_font(size, bold), fill=fill, anchor=anchor,
        stroke_width=max(1, int(stroke * SCALE)), stroke_fill=(0, 0, 0, 220)
    )


def draw_wrapped_text(image: Image.Image, value: str, xy: Tuple[int, int],
                      max_width: int, size: int, fill=(255,255,255,245),
                      bold: bool = False, spacing: int = 6):
    d = ImageDraw.Draw(image)
    font = get_font(size, bold)
    words = value.split()
    lines: List[str] = []
    current = ""
    sw = max(1, int(2*SCALE))
    for word in words:
        test = word if not current else current + " " + word
        box = d.textbbox((0,0), test, font=font, stroke_width=sw)
        if box[2]-box[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        d.text((x,y), line, font=font, fill=fill, stroke_width=sw, stroke_fill=(0,0,0,220))
        box = d.textbbox((x,y), line, font=font, stroke_width=sw)
        y += box[3]-box[1] + int(spacing*SCALE)


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000; ms %= 3_600_000
    m = ms // 60_000; ms %= 60_000
    s = ms // 1000; ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path) -> Path:
    lines: List[str] = []
    for i, (a,b,text) in enumerate(CAPTIONS, 1):
        lines.extend([str(i), f"{format_srt_time(a)} --> {format_srt_time(b)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx-width/2)/(width/2)
    ny = (yy-height/2)/(height/2)
    rr = np.sqrt(nx*nx + ny*ny)
    return np.clip(1.0-strength*rr**1.8, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(W, H, float(CONFIG["vignette"]))


def build_snapshot() -> JupiterStarSnapshot:
    mjup_solar = 0.0009546
    threshold = 80.0
    new_mass_solar = threshold * mjup_solar
    orbit_au = 5.2
    bary = orbit_au * new_mass_solar / (1.0 + new_mass_solar)
    return JupiterStarSnapshot(
        generated_at_utc=iso_z(utc_now()),
        jupiter_mass_kg=1.89813e27,
        jupiter_mass_solar=mjup_solar,
        jupiter_orbit_au=orbit_au,
        assumed_stellar_threshold_jupiter_masses=threshold,
        assumed_new_star_mass_solar=new_mass_solar,
        approximate_sun_barycenter_radius_au=bary,
        brown_dwarf_range_jupiter_masses="~13 to 80 (general NASA framing)",
        scenario="Jupiter hypothetically gains enough mass to become a near-threshold red dwarf while beginning near its current 5.2 AU orbit.",
        interpretation="Illustrative thought experiment. It does not model a physically realistic mass-accretion event or solve the resulting N-body Solar System.",
        nasa_jupiter_url="https://www.nasa.gov/solar-system/what-is-jupiter-grades-5-8/",
        nasa_brown_dwarf_url="https://www.jpl.nasa.gov/images/pia23685-what-is-a-brown-dwarf/",
        nasa_star_types_url="https://science.nasa.gov/universe/stars/types/",
        nssdc_jupiter_url="https://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html",
    )


def save_data(snapshot: JupiterStarSnapshot) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "jupiter_to_star_mass_ladder.csv"
    rows = [
        ("Jupiter now", 1.0, "planet; no sustained hydrogen fusion"),
        ("brown-dwarf regime begins (generalized)", 13.0, "deuterium-burning boundary is model-dependent"),
        ("stellar threshold used here", 80.0, "near lower boundary for sustained hydrogen fusion"),
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "jupiter_masses", "note"])
        w.writerows(rows)
    json_path = DATA_ROOT / "jupiter_star_scenario.json"
    json_path.write_text(json.dumps(asdict(snapshot), indent=2), encoding="utf-8")
    return csv_path, json_path


# =============================================================================
# Renderer
# =============================================================================

@dataclass(frozen=True)
class StarPoint:
    x: float
    y: float
    r: float
    alpha: int
    phase: float


class JupiterStarScene:
    def __init__(self, snapshot: JupiterStarSnapshot):
        self.snapshot = snapshot
        rng = np.random.default_rng(20260921)
        self.stars = [
            StarPoint(float(rng.uniform(0,W)), float(rng.uniform(0,H)),
                      float(rng.uniform(.4,2.2)*SCALE), int(rng.uniform(35,180)),
                      float(rng.uniform(0,2*math.pi)))
            for _ in range(180 if QUICK_MODE else 520)
        ]

    def background(self, t: float) -> Image.Image:
        yy = np.linspace(0,1,H,dtype=np.float32)[:,None]
        top = np.array(COLORS["space"],dtype=np.float32)
        bot = np.array(COLORS["space2"],dtype=np.float32)
        rgb = top[None,:]*(1-yy) + bot[None,:]*yy
        arr = np.empty((H,W,4),dtype=np.uint8)
        arr[:,:,:3] = rgb[:,None,:].astype(np.uint8)
        arr[:,:,3] = 255
        image = Image.fromarray(arr,"RGBA")
        d = ImageDraw.Draw(image)
        for s in self.stars:
            a = int(s.alpha*(.72+.28*math.sin(t*.9+s.phase)))
            d.ellipse((s.x-s.r,s.y-s.r,s.x+s.r,s.y+s.r),fill=COLORS["white"]+(a,))
        return image

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int,int,int,int], alpha: int = 176):
        layer=Image.new("RGBA",SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        d.rounded_rectangle(box,radius=max(8,int(24*SCALE)),fill=COLORS["panel"]+(alpha,),outline=COLORS["cyan"]+(45,),width=max(1,int(2*SCALE)))
        image.alpha_composite(layer)

    def draw_source_hud(self, image: Image.Image):
        draw_text(image,"NASA / JPL // HYPOTHETICAL",(W-int(42*SCALE),int(64*SCALE)),14,COLORS["cyan"]+(225,),True,"ra",1)
        draw_text(image,"~80 JUPITER MASSES -> STELLAR REGIME",(W-int(42*SCALE),int(90*SCALE)),11,COLORS["muted"]+(205,),False,"ra",1)

    def draw_title(self, image: Image.Image, t: float):
        if t >= (5.8 if not QUICK_MODE else 1.30):
            return
        a=int(245*smoothstep(t/(.8 if not QUICK_MODE else .18)))
        draw_text(image,"WHAT IF JUPITER",(W//2,int(H*.062)),38,COLORS["white"]+(a,),True,"ma",2)
        draw_text(image,"BECAME A STAR?",(W//2,int(H*.113)),49,COLORS["orange"]+(a,),True,"ma",2)
        draw_text(image,"a physics-first thought experiment",(W//2,int(H*.159)),16,COLORS["cyan"]+(a,),True,"ma",1)

    def draw_caption(self, image: Image.Image, t: float):
        value=caption_at(t)
        if not value: return
        y0=H-int(188*SCALE)
        self.panel(image,(int(58*SCALE),y0,W-int(58*SCALE),y0+int(116*SCALE)),150)
        draw_wrapped_text(image,value,(int(82*SCALE),y0+int(18*SCALE)),W-int(164*SCALE),30, COLORS["white"]+(240,),False,4)

    def draw_jupiter(self, image: Image.Image, center: Tuple[int,int], r: int, t: float, alpha: int=255):
        cx,cy=center
        layer=Image.new("RGBA",SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        d.ellipse((cx-r,cy-r,cx+r,cy+r),fill=COLORS["jupiter_cream"]+(alpha,),outline=COLORS["white"]+(55,),width=max(1,int(2*SCALE)))
        # clipped-looking band approximations using arcs/rectangles masked by disc
        mask=Image.new("L",SIZE,0); md=ImageDraw.Draw(mask); md.ellipse((cx-r,cy-r,cx+r,cy+r),fill=255)
        bands=Image.new("RGBA",SIZE,(0,0,0,0)); bd=ImageDraw.Draw(bands)
        for i,(fy,hf,col) in enumerate([(-.58,.13,COLORS["jupiter_dark"]),(-.30,.10,COLORS["jupiter_light"]),(-.05,.15,COLORS["jupiter_dark"]),(.22,.11,COLORS["jupiter_light"]),(.48,.15,COLORS["jupiter_dark"])]):
            y=cy+fy*r + math.sin(t*.35+i)*2*SCALE
            hh=hf*r
            bd.rectangle((cx-r,y-hh,cx+r,y+hh),fill=col+(int(alpha*.75),))
        spotx=cx+int(r*.38); spoty=cy+int(r*.22)
        bd.ellipse((spotx-r*.16,spoty-r*.08,spotx+r*.16,spoty+r*.08),fill=COLORS["red"]+(int(alpha*.58),))
        bands.putalpha(Image.composite(bands.getchannel("A"),Image.new("L",SIZE,0),mask))
        layer.alpha_composite(bands)
        image.alpha_composite(layer)

    def draw_red_dwarf(self, image: Image.Image, center: Tuple[int,int], r: int, t: float, alpha: int=255):
        cx,cy=center
        glow=Image.new("RGBA",SIZE,(0,0,0,0)); gd=ImageDraw.Draw(glow)
        pulse=.94+.06*math.sin(t*1.6)
        for scale,a,col in [(2.3,18,COLORS["red"]),(1.75,32,COLORS["orange"]),(1.25,58,COLORS["orange"])]:
            rr=r*scale*pulse
            gd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=col+(int(a*alpha/255),))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4,int(24*SCALE)))))
        d=ImageDraw.Draw(image)
        d.ellipse((cx-r,cy-r,cx+r,cy+r),fill=(222,78,54,alpha),outline=COLORS["gold"]+(min(alpha,220),),width=max(1,int(3*SCALE)))
        for k in range(9):
            a=2*math.pi*k/9 + t*.05
            rr=r*(.25+.55*((k*37)%7)/7)
            q=max(1,int(r*(.025+.018*(k%3))))
            x=cx+math.cos(a)*rr; y=cy+math.sin(a)*rr
            d.ellipse((x-q,y-q,x+q,y+q),fill=(255,161,93,int(alpha*.45)))

    def draw_sun(self, image: Image.Image, center: Tuple[int,int], r: int, t: float):
        cx,cy=center
        glow=Image.new("RGBA",SIZE,(0,0,0,0)); gd=ImageDraw.Draw(glow)
        for scale,a in [(2.4,18),(1.7,34),(1.2,65)]:
            rr=r*scale
            gd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=COLORS["gold"]+(a,))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(3,int(19*SCALE)))))
        ImageDraw.Draw(image).ellipse((cx-r,cy-r,cx+r,cy+r),fill=COLORS["sun"]+(255,),outline=COLORS["white"]+(120,),width=max(1,int(2*SCALE)))

    def draw_impossible_now(self, image: Image.Image, t: float, p: float):
        cx=W//2; cy=int(H*.44); r=int(190*SCALE)
        self.draw_jupiter(image,(cx,cy),r,t)
        # failed ignition pulse in center
        glow=Image.new("RGBA",SIZE,(0,0,0,0)); gd=ImageDraw.Draw(glow)
        q=int(r*(.18+.08*math.sin(t*2.4)))
        gd.ellipse((cx-q,cy-q,cx+q,cy+q),fill=COLORS["orange"]+(55,))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(3,int(18*SCALE)))))
        draw_text(image,"JUPITER IS NOT A FAILED SUN",(W//2,int(H*.68)),28,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"it is ~80× too light for the stellar regime",(W//2,int(H*.725)),16,COLORS["orange"]+(240,),True,"ma",1)
        draw_text(image,"NO SUSTAINED HYDROGEN FUSION",(W//2,int(H*.77)),12,COLORS["cyan"]+(220,),True,"ma",1)

    def draw_mass_ladder(self, image: Image.Image, t: float, p: float):
        x0=int(W*.15); x1=int(W*.85); y=int(H*.48)
        d=ImageDraw.Draw(image)
        d.line((x0,y,x1,y),fill=COLORS["white"]+(150,),width=max(2,int(4*SCALE)))
        marks=[(1,"1×","JUPITER",COLORS["cyan"]),(13,"13×","BROWN DWARF\nREGIME",COLORS["violet"]),(80,"80×","STAR\nTHRESHOLD",COLORS["orange"])]
        logmin=0; logmax=math.log10(80)
        for mass,label,name,col in marks:
            frac=(math.log10(mass)-logmin)/(logmax-logmin)
            x=int(lerp(x0,x1,frac))
            d.line((x,y-int(18*SCALE),x,y+int(18*SCALE)),fill=col+(240,),width=max(2,int(4*SCALE)))
            d.ellipse((x-int(8*SCALE),y-int(8*SCALE),x+int(8*SCALE),y+int(8*SCALE)),fill=col+(245,))
            draw_text(image,label,(x,y-int(52*SCALE)),21,col+(245,),True,"ma",1)
            for j,line in enumerate(name.split("\n")):
                draw_text(image,line,(x,y+int((44+22*j)*SCALE)),13,col+(235,),True,"ma",1)
        reveal=clamp((p-.05)/.7)
        xx=int(lerp(x0,x1,reveal))
        d.line((x0,y-int(100*SCALE),xx,y-int(100*SCALE)),fill=COLORS["gold"]+(190,),width=max(2,int(5*SCALE)))
        draw_text(image,"MASS IS THE SWITCH",(W//2,int(H*.20)),31,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"planet -> brown dwarf -> hydrogen-burning star",(W//2,int(H*.25)),16,COLORS["muted"]+(225,),False,"ma",1)
        draw_text(image,"~80 Mⱼ ≈ 0.076 M☉ in this scenario",(W//2,int(H*.70)),20,COLORS["orange"]+(240,),True,"ma",1)
        draw_text(image,"threshold depends on composition",(W//2,int(H*.745)),12,COLORS["muted"]+(210,),False,"ma",1)

    def draw_binary_gravity(self, image: Image.Image, t: float, p: float):
        cx=W//2; cy=int(H*.46)
        scale=min(W*.40,H*.18)
        # conceptual orbit: barycenter at center; Sun closer, dwarf farther
        mass_ratio=self.snapshot.assumed_new_star_mass_solar
        sun_r=scale*mass_ratio/(1+mass_ratio)
        dwarf_r=scale/(1+mass_ratio)
        angle=t*.36
        sx=cx-math.cos(angle)*sun_r; sy=cy-math.sin(angle)*sun_r*.34
        dx=cx+math.cos(angle)*dwarf_r; dy=cy+math.sin(angle)*dwarf_r*.34
        d=ImageDraw.Draw(image)
        d.ellipse((cx-scale,cy-scale*.34,cx+scale,cy+scale*.34),outline=COLORS["white"]+(70,),width=max(1,int(2*SCALE)))
        # barycenter
        q=max(2,int(7*SCALE)); d.line((cx-q,cy,cx+q,cy),fill=COLORS["cyan"]+(240,),width=max(1,int(2*SCALE))); d.line((cx,cy-q,cx,cy+q),fill=COLORS["cyan"]+(240,),width=max(1,int(2*SCALE)))
        self.draw_sun(image,(int(sx),int(sy)),int(80*SCALE),t)
        self.draw_red_dwarf(image,(int(dx),int(dy)),int(53*SCALE),t)
        draw_text(image,"THE SUN WOULD WOBBLE A LOT",(W//2,int(H*.19)),30,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"~0.37 AU Sun-to-barycenter scale in this simplified setup",(W//2,int(H*.245)),14,COLORS["cyan"]+(230,),False,"ma",1)
        draw_text(image,"BARYCENTER",(cx,cy-int(28*SCALE)),11,COLORS["cyan"]+(230,),True,"ma",1)
        draw_text(image,"SUN",(int(sx),int(sy)+int(110*SCALE)),14,COLORS["gold"]+(235,),True,"ma",1)
        draw_text(image,"~0.08 M☉ RED DWARF",(int(dx),int(dy)+int(88*SCALE)),13,COLORS["orange"]+(235,),True,"ma",1)
        draw_text(image,"not a drop-in replacement for present-day Jupiter",(W//2,int(H*.72)),15,COLORS["red"]+(225,),True,"ma",1)

    def draw_second_sun(self, image: Image.Image, t: float, p: float):
        # Earth night horizon + bright red point
        d=ImageDraw.Draw(image)
        horizon=int(H*.69)
        d.rectangle((0,horizon,W,H),fill=(3,14,24,255))
        for i in range(14):
            x=int(W*i/13); h=int((18+45*((i*11)%9)/9)*SCALE)
            d.polygon([(x-int(45*SCALE),horizon),(x,horizon-h),(x+int(55*SCALE),horizon)],fill=(8,29,39,255))
        sx=int(W*.73); sy=int(H*.31); r=int(30*SCALE)
        self.draw_red_dwarf(image,(sx,sy),r,t)
        draw_text(image,"A SECOND SELF-LUMINOUS OBJECT",(W//2,int(H*.17)),29,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"cooler + much fainter than the Sun",(W//2,int(H*.225)),17,COLORS["orange"]+(240,),True,"ma",1)
        # angular separation / uncertainty cue
        d.arc((int(W*.16),int(H*.34),int(W*.84),int(H*.62)),205,333,fill=COLORS["cyan"]+(100,),width=max(1,int(3*SCALE)))
        draw_text(image,"exact apparent brightness depends on the star we create",(W//2,int(H*.76)),13,COLORS["muted"]+(215,),False,"ma",1)
        draw_text(image,"ILLUSTRATIVE SKY — NOT PHOTOMETRIC",(W//2,int(H*.80)),11,COLORS["red"]+(215,),True,"ma",1)

    def draw_moons(self, image: Image.Image, t: float, p: float):
        cx=W//2; cy=int(H*.44); r=int(92*SCALE)
        self.draw_red_dwarf(image,(cx,cy),r,t)
        d=ImageDraw.Draw(image)
        names=[("IO",.31,COLORS["gold"]),("EUROPA",.43,COLORS["white"]),("GANYMEDE",.56,COLORS["muted"]),("CALLISTO",.70,COLORS["violet"])]
        maxr=int(W*.40)
        for i,(name,frac,col) in enumerate(names):
            rr=maxr*frac
            d.ellipse((cx-rr,cy-rr*.28,cx+rr,cy+rr*.28),outline=col+(65,),width=max(1,int(2*SCALE)))
            a=t*(.55-.07*i)+i*1.45
            x=cx+math.cos(a)*rr; y=cy+math.sin(a)*rr*.28
            q=max(2,int((6+2*i)*SCALE))
            d.ellipse((x-q,y-q,x+q,y+q),fill=col+(240,))
            draw_text(image,name,(int(x),int(y)-int(18*SCALE)),10,col+(225,),True,"ma",1)
        draw_text(image,"JUPITER'S MOONS BECOME A STELLAR SYSTEM",(W//2,int(H*.19)),27,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"their exact futures depend on how the mass appeared",(W//2,int(H*.72)),14,COLORS["cyan"]+(230,),False,"ma",1)
        draw_text(image,"NO SIMPLE 'MOONS STAY THE SAME' ASSUMPTION",(W//2,int(H*.765)),11,COLORS["red"]+(215,),True,"ma",1)

    def draw_outro(self, image: Image.Image, t: float, p: float):
        # schematic inner system with Sun + red dwarf companion
        cx=W//2; cy=int(H*.43)
        self.draw_sun(image,(int(W*.35),cy),int(90*SCALE),t)
        self.draw_red_dwarf(image,(int(W*.72),cy+int(20*SCALE)),int(58*SCALE),t)
        d=ImageDraw.Draw(image)
        # Earth and outer orbit cues
        d.ellipse((int(W*.35)-int(145*SCALE),cy-int(48*SCALE),int(W*.35)+int(145*SCALE),cy+int(48*SCALE)),outline=COLORS["earth"]+(65,),width=max(1,int(2*SCALE)))
        ea=t*.9
        ex=int(W*.35)+math.cos(ea)*int(145*SCALE); ey=cy+math.sin(ea)*int(48*SCALE)
        q=max(2,int(7*SCALE)); d.ellipse((ex-q,ey-q,ex+q,ey+q),fill=COLORS["earth"]+(245,))
        a=int(245*smoothstep((p-.12)/.48))
        draw_text(image,"THE SOLAR SYSTEM BECOMES",(W//2,int(H*.67)),29,COLORS["white"]+(a,),True,"ma",2)
        draw_text(image,"A VERY DIFFERENT TWO-STAR DYNAMICAL PROBLEM",(W//2,int(H*.715)),22,COLORS["orange"]+(a,),True,"ma",2)
        draw_text(image,"strong perturbations — exact fate requires N-body modeling",(W//2,int(H*.77)),13,COLORS["cyan"]+(a,),False,"ma",1)

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
        if name=="impossible_now": self.draw_impossible_now(image,t,p)
        elif name=="mass_ladder": self.draw_mass_ladder(image,t,p)
        elif name=="binary_gravity": self.draw_binary_gravity(image,t,p)
        elif name=="second_sun": self.draw_second_sun(image,t,p)
        elif name=="moons": self.draw_moons(image,t,p)
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

def render_preview_frames(scene: JupiterStarScene) -> List[Path]:
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


def render_video(scene: JupiterStarScene) -> Path:
    path=OUTPUT_ROOT/f"{CONFIG['basename']}.mp4"
    total=max(1,int(round(DURATION*FPS)))
    writer=iio.get_writer(path,fps=FPS,codec="libx264",quality=7,pixelformat="yuv420p",ffmpeg_log_level="error",macro_block_size=2)
    try:
        for i in tqdm(range(total),desc="Rendering Jupiter-Star Short"):
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

YOUTUBE_TITLE = 'What If Jupiter Became a Star? 🪐⭐ #rootjatin'
YOUTUBE_DESCRIPTION = "This is a hypothetical thought experiment. Jupiter cannot simply ignite at its current mass; sustained hydrogen fusion requires an object far more massive, in the low-mass-star regime. The visualization explores how a stellar-mass 'Jupiter' would change the gravity, light, moons, and architecture of the Solar System. It is not an instant-transformation prediction or a physically possible event for present-day Jupiter."
YOUTUBE_HASHTAGS = '#rootjatin #Jupiter #Star #WhatIf #SolarSystem #Astronomy #Space #Science'

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
    snap=build_snapshot()
    csv_path,json_path=save_data(snap)
    srt_path=write_srt(OUTPUT_ROOT/f"{CONFIG['basename']}_subtitles.srt")
    scene=JupiterStarScene(snap)
    previews=render_preview_frames(scene)
    sheet=make_contact_sheet(previews)
    video=render_video(scene)
    flat_video,flat_sheet,flat_srt=copy_flat(video,sheet,srt_path)
    print("\nRender complete")
    print("="*72)
    print(CONFIG["title"])
    print(f"Mode: {'QUICK' if QUICK_MODE else ('4K' if FOUR_K else 'FULL HD')}")
    print(f"Frame: {W}x{H} @ {FPS} fps // {DURATION:.1f} s")
    print(f"Assumed star mass: {snap.assumed_stellar_threshold_jupiter_masses:.0f} Mj = {snap.assumed_new_star_mass_solar:.3f} Msun")
    print(f"Approx Sun-barycenter scale: {snap.approximate_sun_barycenter_radius_au:.3f} AU")
    for label,path in [("video",video),("subtitles",srt_path),("csv",csv_path),("json",json_path),("contact",sheet),("flat_video",flat_video),("flat_sheet",flat_sheet),("flat_srt",flat_srt)]:
        print(f"{label:12s} {path}")
    metadata_txt = write_youtube_metadata_txt()
    print("Title/description TXT:", metadata_txt.resolve())


if __name__ == "__main__":
    main()
