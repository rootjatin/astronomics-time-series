from __future__ import annotations

"""
I Put 1,000 Planets Around One Star
===================================

A cinematic vertical YouTube Short renderer about orbital crowding.

The visual setup places 1,000 equal Earth-mass planets on initially circular,
coplanar orbits from 0.10 to 10 AU around a one-solar-mass star. The animation
propagates those orbits with a STAR-ONLY Kepler model for visualization, while
separate mutual-Hill-radius diagnostics quantify how impossibly crowded the
massive system would be once planet-planet gravity is included.

Important interpretation
------------------------
- The 1,000 moving dots are NOT a full 1,000-body long-term N-body integration.
- Their displayed trajectories use independent circular Kepler orbits around a
  fixed 1-Msun star so the initial architecture can be rendered cheaply.
- Every planet is assigned one Earth mass for the crowding calculation.
- Neighbor spacing is measured in mutual Hill radii:
      R_H,m = ((m1+m2)/(3 Mstar))^(1/3) * (a1+a2)/2
      Delta = (a2-a1)/R_H,m
- For circular coplanar neighboring planets, the classic pairwise Hill criterion
  is Delta > 2*sqrt(3) ~= 3.46. Multi-planet systems that survive for very long
  times commonly require substantially wider spacing than that simple pairwise
  threshold.
- In this geometric 0.10-to-10-AU setup, all adjacent pairs are only about
  0.366 mutual Hill radii apart -- almost an order of magnitude inside the
  pairwise Hill-stable separation.
- Therefore the final "instability" graphics are diagnostic/illustrative, not a
  claimed exact chronology of collisions or ejections.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    THOUSAND_PLANETS_QUICK=1 python i_put_1000_planets_around_one_star.py

Full 1080x1920 render
---------------------
    python i_put_1000_planets_around_one_star.py

4K vertical
-----------
    THOUSAND_PLANETS_4K=1 python i_put_1000_planets_around_one_star.py
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

QUICK_MODE = os.environ.get("THOUSAND_PLANETS_QUICK", "0") == "1"
FOUR_K = os.environ.get("THOUSAND_PLANETS_4K", "0") == "1" and not QUICK_MODE

W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 6 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SIZE = (W, H)
SCALE = W / 1080.0

OUTPUT_ROOT = Path("i_put_1000_planets_around_one_star_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "I PUT 1,000 PLANETS AROUND ONE STAR",
    "subtitle": "one Sun // 1,000 Earth-mass worlds // orbital traffic jam",
    "basename": "i_put_1000_planets_around_one_star",
    "contrast": 1.12,
    "saturation": 1.08,
    "vignette": 0.28,
}

COLORS = {
    "space": (2, 4, 13),
    "space2": (8, 15, 36),
    "white": (247, 251, 255),
    "muted": (164, 191, 214),
    "cyan": (72, 229, 255),
    "blue": (80, 137, 255),
    "gold": (255, 207, 91),
    "orange": (255, 141, 72),
    "red": (255, 78, 103),
    "green": (103, 239, 174),
    "violet": (190, 126, 255),
    "magenta": (241, 91, 190),
    "star": (255, 220, 105),
    "star_hot": (255, 249, 220),
    "panel": (3, 8, 22),
}

SHOT_PLAN = [
    {"name": "reveal", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "count", "start": 8.0 if not QUICK_MODE else 1.8, "end": 18.0 if not QUICK_MODE else 4.0},
    {"name": "hill", "start": 18.0 if not QUICK_MODE else 4.0, "end": 29.0 if not QUICK_MODE else 6.45},
    {"name": "zoom", "start": 29.0 if not QUICK_MODE else 6.45, "end": 40.0 if not QUICK_MODE else 8.9},
    {"name": "instability", "start": 40.0 if not QUICK_MODE else 8.9, "end": 51.0 if not QUICK_MODE else 11.35},
    {"name": "outro", "start": 51.0 if not QUICK_MODE else 11.35, "end": DURATION},
]

CAPTION_TEXTS = [
    "I put one thousand Earth-mass planets around one Sun-like star, spreading them from one tenth of an astronomical unit all the way out to ten AU.",
    "The picture looks almost elegant when every world follows its own circular Kepler orbit. But that preview hides the hard part: planets do not ignore one another's gravity.",
    "For two neighboring Earth-mass planets, a classic Hill-stability threshold is about 3.46 mutual Hill radii. My adjacent planets are only about 0.37 mutual Hill radii apart.",
    "That means this system is not merely compact. Each orbit is buried deep inside the gravitational neighborhood of the next. Tiny perturbations would couple huge chains of nearby worlds.",
    "A true one-thousand-body integration would decide the exact sequence, but close encounters, collisions, mergers, and ejections would be the natural expectation for such extreme crowding.",
    "The strange lesson is that one thousand planets do fit on paper. The impossible part is making one thousand massive neighboring orbits stay dynamically independent for long.",
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
# Architecture and diagnostics
# =============================================================================

N_PLANETS = 1000
STAR_MASS_SOLAR = 1.0
EARTH_MASS_SOLAR = 3.003e-6
A_MIN_AU = 0.10
A_MAX_AU = 10.0
SEMIMAJOR_AXES = np.geomspace(A_MIN_AU, A_MAX_AU, N_PLANETS)
PHASES = np.mod(np.arange(N_PLANETS) * 2.399963229728653, math.tau)  # golden-angle phase spread
PERIOD_YEARS = np.sqrt(SEMIMAJOR_AXES**3 / STAR_MASS_SOLAR)
HILL_LIMIT = 2.0 * math.sqrt(3.0)

MEAN_A = 0.5 * (SEMIMAJOR_AXES[:-1] + SEMIMAJOR_AXES[1:])
MUTUAL_HILL = ((2.0 * EARTH_MASS_SOLAR) / (3.0 * STAR_MASS_SOLAR)) ** (1.0 / 3.0) * MEAN_A
DELTA_HILL = np.diff(SEMIMAJOR_AXES) / MUTUAL_HILL


def count_for_spacing(delta: float) -> int:
    k = ((2.0 * EARTH_MASS_SOLAR) / (3.0 * STAR_MASS_SOLAR)) ** (1.0 / 3.0)
    d = delta * k / 2.0
    ratio = (1.0 + d) / (1.0 - d)
    return int(math.floor(math.log(A_MAX_AU / A_MIN_AU) / math.log(ratio)) + 1)

PAIRWISE_HILL_CAPACITY = count_for_spacing(HILL_LIMIT)
TEN_HILL_CAPACITY = count_for_spacing(10.0)
TWELVE_HILL_CAPACITY = count_for_spacing(12.0)


@dataclass
class ThousandPlanetSnapshot:
    generated_at_utc: str
    star_mass_solar: float
    planet_count: int
    planet_mass_earth_each: float
    total_planet_mass_earth: float
    total_planet_mass_jupiter_approx: float
    inner_orbit_au: float
    outer_orbit_au: float
    inner_period_days: float
    outer_period_years: float
    mean_neighbor_spacing_mutual_hill: float
    pairwise_hill_limit: float
    mean_spacing_fraction_of_pairwise_limit: float
    estimated_count_over_same_range_at_pairwise_limit: int
    estimated_count_over_same_range_at_10_hill: int
    estimated_count_over_same_range_at_12_hill: int
    visual_dynamics: str
    interpretation: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def snapshot() -> ThousandPlanetSnapshot:
    return ThousandPlanetSnapshot(
        generated_at_utc=iso_z(utc_now()),
        star_mass_solar=STAR_MASS_SOLAR,
        planet_count=N_PLANETS,
        planet_mass_earth_each=1.0,
        total_planet_mass_earth=float(N_PLANETS),
        total_planet_mass_jupiter_approx=float(N_PLANETS / 317.8),
        inner_orbit_au=A_MIN_AU,
        outer_orbit_au=A_MAX_AU,
        inner_period_days=float(PERIOD_YEARS[0] * 365.25),
        outer_period_years=float(PERIOD_YEARS[-1]),
        mean_neighbor_spacing_mutual_hill=float(np.mean(DELTA_HILL)),
        pairwise_hill_limit=HILL_LIMIT,
        mean_spacing_fraction_of_pairwise_limit=float(np.mean(DELTA_HILL) / HILL_LIMIT),
        estimated_count_over_same_range_at_pairwise_limit=PAIRWISE_HILL_CAPACITY,
        estimated_count_over_same_range_at_10_hill=TEN_HILL_CAPACITY,
        estimated_count_over_same_range_at_12_hill=TWELVE_HILL_CAPACITY,
        visual_dynamics="independent circular Kepler orbits around fixed star; mutual planet gravity omitted from animation",
        interpretation="Hill-radius diagnostics quantify crowding; instability sequence is illustrative, not a 1000-body chronology.",
    )


def build_table() -> pd.DataFrame:
    df = pd.DataFrame({
        "planet_index": np.arange(1, N_PLANETS + 1),
        "semimajor_axis_au": SEMIMAJOR_AXES,
        "period_years": PERIOD_YEARS,
        "initial_phase_rad": PHASES,
        "planet_mass_earth": np.ones(N_PLANETS),
    })
    df["neighbor_delta_mutual_hill"] = np.nan
    df.loc[:N_PLANETS-2, "neighbor_delta_mutual_hill"] = DELTA_HILL
    return df


# =============================================================================
# Drawing helpers
# =============================================================================

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
              fill=(255,255,255,255), bold: bool=False, anchor: str="la", stroke: int=2):
    ImageDraw.Draw(image).text(
        xy, value, font=get_font(size, bold), fill=fill, anchor=anchor,
        stroke_width=max(1, int(stroke*SCALE)), stroke_fill=(0,0,0,220)
    )


def draw_wrapped_text(image: Image.Image, value: str, xy: Tuple[int,int], max_width: int,
                      size: int, fill=(255,255,255,245), bold: bool=False, spacing: int=6):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold)
    words = value.split()
    lines: List[str] = []
    cur = ""
    for word in words:
        test = word if not cur else cur + " " + word
        box = draw.textbbox((0,0), test, font=font, stroke_width=max(1,int(2*SCALE)))
        if box[2]-box[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    x,y = xy
    for line in lines:
        draw.text((x,y), line, font=font, fill=fill,
                  stroke_width=max(1,int(2*SCALE)), stroke_fill=(0,0,0,220))
        box = draw.textbbox((x,y), line, font=font, stroke_width=max(1,int(2*SCALE)))
        y += (box[3]-box[1]) + int(spacing*SCALE)


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds*1000))
    h = ms//3_600_000; ms%=3_600_000
    m = ms//60_000; ms%=60_000
    s = ms//1000; ms%=1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path) -> Path:
    lines: List[str] = []
    for i,(a,b,text) in enumerate(CAPTIONS,1):
        lines.extend([str(i), f"{format_srt_time(a)} --> {format_srt_time(b)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy,xx = np.mgrid[0:height,0:width]
    nx=(xx-width/2)/(width/2); ny=(yy-height/2)/(height/2)
    rr=np.sqrt(nx*nx+ny*ny)
    return np.clip(1-strength*rr**1.8,0,1).astype(np.float32)

VIGNETTE = make_vignette(W,H,float(CONFIG["vignette"]))


# =============================================================================
# Scene
# =============================================================================

class ThousandPlanetScene:
    def __init__(self, snap: ThousandPlanetSnapshot):
        self.snap = snap
        rng=np.random.default_rng(1000)
        self.bg_stars=[
            (float(rng.uniform(0,W)),float(rng.uniform(0,H)),float(rng.uniform(.4,2.0)*SCALE),int(rng.uniform(12,75)),float(rng.uniform(0,math.tau)))
            for _ in range(150 if QUICK_MODE else 430)
        ]
        self.base_bg=self._make_background()
        # color bands by log orbital radius
        q=(np.log10(SEMIMAJOR_AXES)-math.log10(A_MIN_AU))/(math.log10(A_MAX_AU)-math.log10(A_MIN_AU))
        c0=np.array(COLORS["cyan"],dtype=float); c1=np.array(COLORS["violet"],dtype=float); c2=np.array(COLORS["orange"],dtype=float)
        cols=[]
        for v in q:
            if v<.55:
                u=v/.55; c=c0*(1-u)+c1*u
            else:
                u=(v-.55)/.45; c=c1*(1-u)+c2*u
            cols.append(tuple(int(x) for x in c))
        self.planet_colors=cols

    def _make_background(self) -> Image.Image:
        top=np.asarray(COLORS["space"],dtype=np.float32)
        bottom=np.asarray(COLORS["space2"],dtype=np.float32)
        f=np.linspace(0,1,H,dtype=np.float32)[:,None]
        rgb=(top[None,:]*(1-f)+bottom[None,:]*f).astype(np.uint8)
        arr=np.empty((H,W,4),dtype=np.uint8); arr[...,:3]=rgb[:,None,:]; arr[...,3]=255
        return Image.fromarray(arr,"RGBA")

    def background(self,t: float) -> Image.Image:
        img=self.base_bg.copy(); d=ImageDraw.Draw(img)
        for x,y,r,a,p in self.bg_stars:
            aa=int(a*(.78+.22*math.sin(t*.9+p)))
            d.ellipse((x-r,y-r,x+r,y+r),fill=COLORS["white"]+(aa,))
        glow=Image.new("RGBA",SIZE,(0,0,0,0)); gd=ImageDraw.Draw(glow)
        for cx,cy,col in [(W*.24,H*.27,COLORS["blue"]),(W*.76,H*.28,COLORS["violet"]),(W*.52,H*.70,COLORS["red"])]:
            rr=W*.28; gd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=col+(13,))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(18,int(55*SCALE)))))
        return img

    @staticmethod
    def panel(img: Image.Image, box: Tuple[int,int,int,int], alpha: int=176):
        layer=Image.new("RGBA",SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        d.rounded_rectangle(box,radius=max(8,int(24*SCALE)),fill=COLORS["panel"]+(alpha,),outline=COLORS["cyan"]+(48,),width=max(1,int(2*SCALE)))
        img.alpha_composite(layer)

    def header(self,img: Image.Image,t: float):
        if t < (6.0 if not QUICK_MODE else 1.35):
            f=smoothstep(t/(.75 if not QUICK_MODE else .18))
            draw_text(img,"I PUT",(W//2,int(H*.053)),38 if not QUICK_MODE else 19,COLORS["white"]+(int(245*f),),True,"ma",2)
            draw_text(img,"1,000 PLANETS",(W//2,int(H*.098)),58 if not QUICK_MODE else 29,COLORS["cyan"]+(int(250*f),),True,"ma",2)
            draw_text(img,"AROUND ONE STAR",(W//2,int(H*.146)),35 if not QUICK_MODE else 17,COLORS["gold"]+(int(245*f),),True,"ma",2)
        else:
            labels={
                "reveal":"ONE STAR // ONE THOUSAND WORLDS",
                "count":"THE PREVIEW LOOKS ORDERLY",
                "hill":"THE HILL-RADIUS PROBLEM",
                "zoom":"EVERY ORBIT OVERLAPS A GRAVITATIONAL NEIGHBORHOOD",
                "instability":"REAL MUTUAL GRAVITY WOULD TAKE OVER",
                "outro":"FITS ON PAPER ≠ STAYS STABLE",
            }
            draw_text(img,labels[get_shot(t)["name"]],(int(W*.055),int(H*.044)),17 if not QUICK_MODE else 8,COLORS["muted"]+(230,),True,"la",1)
        draw_text(img,"1000 EARTH MASSES // STAR-ONLY ORBIT VISUALIZATION",(int(W*.945),int(H*.047)),11 if not QUICK_MODE else 5,COLORS["cyan"]+(205,),True,"ra",1)

    def caption(self,img: Image.Image,t: float):
        cap=caption_at(t)
        if not cap: return
        y0=H-int(186*SCALE)
        self.panel(img,(int(58*SCALE),y0,W-int(58*SCALE),y0+int(116*SCALE)),150)
        draw_wrapped_text(img,cap,(int(82*SCALE),y0+int(18*SCALE)),W-int(164*SCALE),30,COLORS["white"]+(240,),False,4)

    def positions(self, years: float) -> np.ndarray:
        ang=PHASES + math.tau*years/PERIOD_YEARS
        return np.column_stack((SEMIMAJOR_AXES*np.cos(ang),SEMIMAJOR_AXES*np.sin(ang)))

    def draw_star(self,img: Image.Image,cx: float,cy: float,r: float=12):
        glow=Image.new("RGBA",SIZE,(0,0,0,0)); gd=ImageDraw.Draw(glow)
        for q,a in [(42,26),(28,48),(18,86)]:
            rr=q*SCALE; gd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=COLORS["star"]+(a,))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4,int(10*SCALE)))))
        d=ImageDraw.Draw(img); rr=r*SCALE
        d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=COLORS["star_hot"]+(255,))

    def draw_disk_system(self,img: Image.Image,years: float, max_au: float=10.0, yfrac: float=.46, count: int=1000, highlight: bool=False):
        cx=W*.5; cy=H*yfrac; pix=min(W*.42,H*.27)/max_au
        d=ImageDraw.Draw(img)
        # sampled orbit rings only
        for au in [0.1,0.2,0.5,1,2,5,10]:
            if au>max_au: continue
            rr=au*pix
            d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=COLORS["white"]+(24,),width=max(1,int(SCALE)))
            if au in (1,5,10):
                draw_text(img,f"{au:g} AU",(int(cx+rr+5*SCALE),int(cy)),9 if not QUICK_MODE else 4,COLORS["muted"]+(110,),False,"lm",1)
        self.draw_star(img,cx,cy)
        pts=self.positions(years)
        indices=np.linspace(0,N_PLANETS-1,count).astype(int)
        for idx in indices:
            x,y=pts[idx]
            if math.hypot(x,y)>max_au*1.001: continue
            sx=cx+x*pix; sy=cy+y*pix
            rp=(2.0 if count>400 else 3.2)*SCALE
            if highlight and idx%41==0: rp*=1.7
            d.ellipse((sx-rp,sy-rp,sx+rp,sy+rp),fill=self.planet_colors[idx]+(220,))
        return cx,cy,pix

    def draw_reveal(self,img: Image.Image,t: float,local: float):
        count=max(12,int(lerp(12,1000,smoothstep(local))))
        self.draw_disk_system(img,years=.08*local,max_au=10,yfrac=.46,count=count)
        self.panel(img,(int(W*.16),int(H*.68),int(W*.84),int(H*.785)),168)
        draw_text(img,f"{count:,}",(W//2,int(H*.708)),42 if not QUICK_MODE else 21,COLORS["cyan"]+(250,),True,"ma",2)
        draw_text(img,"EARTH-MASS PLANETS",(W//2,int(H*.756)),18 if not QUICK_MODE else 9,COLORS["white"]+(230,),True,"ma",1)

    def draw_count(self,img: Image.Image,t: float,local: float):
        years=lerp(0,2.5,local)
        self.draw_disk_system(img,years,max_au=10,yfrac=.46,count=1000)
        self.panel(img,(int(W*.11),int(H*.685),int(W*.89),int(H*.79)),172)
        draw_text(img,"0.10 AU  →  10 AU",(W//2,int(H*.718)),24 if not QUICK_MODE else 12,COLORS["gold"]+(245,),True,"ma",1)
        draw_text(img,"STAR-ONLY KEPLER PREVIEW",(W//2,int(H*.758)),15 if not QUICK_MODE else 7,COLORS["muted"]+(225,),True,"ma",1)

    def draw_hill(self,img: Image.Image,t: float,local: float):
        x0=int(W*.12); x1=int(W*.88); y0=int(H*.28); y1=int(H*.61)
        self.panel(img,(x0-int(18*SCALE),y0-int(45*SCALE),x1+int(18*SCALE),y1+int(90*SCALE)),174)
        d=ImageDraw.Draw(img)
        # gauge from 0 to 4 Hill radii
        gy=int(H*.45)
        d.line((x0,gy,x1,gy),fill=COLORS["white"]+(85,),width=max(2,int(4*SCALE)))
        for val in [0,1,2,3,3.46,4]:
            x=lerp(x0,x1,val/4)
            d.line((x,gy-int(9*SCALE),x,gy+int(9*SCALE)),fill=COLORS["muted"]+(130,),width=max(1,int(2*SCALE)))
            draw_text(img,f"{val:g}",(int(x),gy+int(28*SCALE)),11 if not QUICK_MODE else 5,COLORS["muted"]+(210,),False,"ma",1)
        actual=float(np.mean(DELTA_HILL)); xa=lerp(x0,x1,actual/4); xh=lerp(x0,x1,HILL_LIMIT/4)
        d.line((xa,gy-int(48*SCALE),xa,gy+int(48*SCALE)),fill=COLORS["red"]+(245,),width=max(3,int(8*SCALE)))
        d.line((xh,gy-int(40*SCALE),xh,gy+int(40*SCALE)),fill=COLORS["green"]+(230,),width=max(2,int(5*SCALE)))
        draw_text(img,"MY NEIGHBORS",(int(xa),gy-int(63*SCALE)),15 if not QUICK_MODE else 7,COLORS["red"]+(245,),True,"ma",1)
        draw_text(img,"PAIRWISE HILL LIMIT",(int(xh),gy-int(62*SCALE)),14 if not QUICK_MODE else 7,COLORS["green"]+(235,),True,"ma",1)
        draw_text(img,f"Δ ≈ {actual:.3f} mutual Hill radii",(W//2,int(H*.68)),25 if not QUICK_MODE else 12,COLORS["red"]+(250,),True,"ma",1)
        draw_text(img,f"threshold ≈ {HILL_LIMIT:.2f}",(W//2,int(H*.724)),16 if not QUICK_MODE else 8,COLORS["white"]+(225,),True,"ma",1)

    def draw_zoom(self,img: Image.Image,t: float,local: float):
        # local slice around 1 AU; exaggerate radial spacing while showing Hill zones
        d=ImageDraw.Draw(img)
        self.panel(img,(int(W*.08),int(H*.24),int(W*.92),int(H*.69)),166)
        center_idx=np.searchsorted(SEMIMAJOR_AXES,1.0)
        ids=np.arange(center_idx-7,center_idx+8)
        yy0=int(H*.33); yy1=int(H*.59)
        xleft=int(W*.14); xright=int(W*.86)
        local_a=SEMIMAJOR_AXES[ids]
        amin,amax=float(local_a[0]),float(local_a[-1])
        for k,idx in enumerate(ids):
            x=lerp(xleft,xright,(float(SEMIMAJOR_AXES[idx])-amin)/(amax-amin))
            rh=((2*EARTH_MASS_SOLAR)/(3*STAR_MASS_SOLAR))**(1/3)*SEMIMAJOR_AXES[idx]
            # visually scale the Hill zone to show overlap relative to local spacing
            px_per_au=(xright-xleft)/(amax-amin)
            hr=rh*px_per_au
            d.rounded_rectangle((x-hr,yy0,x+hr,yy1),radius=int(8*SCALE),fill=COLORS["red"]+(20,),outline=COLORS["red"]+(65,),width=max(1,int(SCALE)))
            d.line((x,yy0,x,yy1),fill=self.planet_colors[idx]+(220,),width=max(2,int(3*SCALE)))
        draw_text(img,"15 NEIGHBORING ORBITS NEAR 1 AU",(W//2,int(H*.274)),18 if not QUICK_MODE else 9,COLORS["cyan"]+(240,),True,"ma",1)
        draw_text(img,"red zones = ~1 mutual Hill radius scale",(W//2,int(H*.635)),14 if not QUICK_MODE else 7,COLORS["muted"]+(220,),True,"ma",1)
        self.panel(img,(int(W*.13),int(H*.71),int(W*.87),int(H*.79)),166)
        draw_text(img,"ORBIT SPACING ≈ 0.37 Rₕ,m",(W//2,int(H*.742)),23 if not QUICK_MODE else 11,COLORS["red"]+(248,),True,"ma",1)

    def draw_instability(self,img: Image.Image,t: float,local: float):
        # illustrative scattering: smoothly distort a subset, explicitly labeled
        cx=W*.5; cy=H*.46; pix=min(W*.42,H*.27)/4.0
        d=ImageDraw.Draw(img)
        for au in [0.5,1,2,4]:
            rr=au*pix; d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=COLORS["white"]+(18,),width=1)
        self.draw_star(img,cx,cy)
        pts=self.positions(lerp(0,1.5,local))
        ids=np.linspace(250,750,260).astype(int)
        for j,idx in enumerate(ids):
            x,y=pts[idx]
            # purely illustrative perturbation envelope
            kick=smoothstep(local)*(.06+0.35*((j%17)/16))
            ang=math.atan2(y,x)+kick*math.sin(j*1.7)
            rad=math.hypot(x,y)*(1+kick*.55*math.sin(j*.81+1.3))
            if j%53==0:
                rad*=1+1.8*smoothstep((local-.45)/.55)
            sx=cx+rad*math.cos(ang)*pix; sy=cy+rad*math.sin(ang)*pix
            rp=2.6*SCALE
            d.ellipse((sx-rp,sy-rp,sx+rp,sy+rp),fill=self.planet_colors[idx]+(220,))
        draw_text(img,"ILLUSTRATIVE INSTABILITY — NOT AN N-BODY TIMELINE",(W//2,int(H*.685)),15 if not QUICK_MODE else 7,COLORS["orange"]+(245,),True,"ma",1)
        draw_text(img,"collisions • mergers • scattering • ejections",(W//2,int(H*.727)),19 if not QUICK_MODE else 9,COLORS["white"]+(230,),True,"ma",1)

    def draw_outro(self,img: Image.Image,t: float,local: float):
        # capacity comparison bars
        x0=int(W*.16); x1=int(W*.84); y0=int(H*.28); barh=int(46*SCALE)
        self.panel(img,(int(W*.10),int(H*.22),int(W*.90),int(H*.70)),174)
        rows=[
            ("MY SETUP",1000,COLORS["red"]),
            ("PAIRWISE HILL FLOOR",PAIRWISE_HILL_CAPACITY,COLORS["green"]),
            ("10 Rₕ SPACING",TEN_HILL_CAPACITY,COLORS["cyan"]),
            ("12 Rₕ SPACING",TWELVE_HILL_CAPACITY,COLORS["violet"]),
        ]
        for i,(label,val,col) in enumerate(rows):
            y=y0+i*int(82*SCALE)
            draw_text(img,label,(x0,y),14 if not QUICK_MODE else 7,COLORS["muted"]+(220,),True,"la",1)
            frac=math.log10(val+1)/math.log10(1001)
            bw=int((x1-x0)*frac*smoothstep(local+.15))
            d=ImageDraw.Draw(img); d.rounded_rectangle((x0,y+int(24*SCALE),x0+bw,y+int(24*SCALE)+barh),radius=int(10*SCALE),fill=col+(200,))
            draw_text(img,f"{val:,}",(x1,y+int(47*SCALE)),18 if not QUICK_MODE else 9,col+(245,),True,"ra",1)
        draw_text(img,"same radial range: 0.10 → 10 AU",(W//2,int(H*.665)),13 if not QUICK_MODE else 6,COLORS["white"]+(210,),True,"ma",1)
        draw_text(img,"CROWDING, NOT DRAWING SPACE, IS THE LIMIT",(W//2,int(H*.735)),19 if not QUICK_MODE else 9,COLORS["gold"]+(248,),True,"ma",1)

    def render(self,t: float) -> np.ndarray:
        img=self.background(t); shot=get_shot(t); local=clamp((t-shot["start"])/max(1e-9,shot["end"]-shot["start"]))
        self.header(img,t)
        if shot["name"]=="reveal": self.draw_reveal(img,t,local)
        elif shot["name"]=="count": self.draw_count(img,t,local)
        elif shot["name"]=="hill": self.draw_hill(img,t,local)
        elif shot["name"]=="zoom": self.draw_zoom(img,t,local)
        elif shot["name"]=="instability": self.draw_instability(img,t,local)
        else: self.draw_outro(img,t,local)
        self.caption(img,t)
        overlay=Image.new("RGBA",SIZE,(0,0,0,0)); od=ImageDraw.Draw(overlay); off=int((t*31)%9)
        for y in range(off,H,9): od.line((0,y,W,y),fill=(120,205,240,7),width=1)
        img.alpha_composite(overlay)
        arr=np.asarray(img.convert("RGB")).astype(np.float32); arr*=VIGNETTE[...,None]; arr=np.clip(arr,0,255).astype(np.uint8)
        graded=Image.fromarray(arr); graded=ImageEnhance.Contrast(graded).enhance(float(CONFIG["contrast"])); graded=ImageEnhance.Color(graded).enhance(float(CONFIG["saturation"]))
        return np.asarray(graded)


# =============================================================================
# Output
# =============================================================================

def save_data(snap: ThousandPlanetSnapshot) -> Tuple[Path,Path]:
    csv_path=DATA_ROOT/"thousand_planet_architecture.csv"
    json_path=DATA_ROOT/"thousand_planet_snapshot.json"
    build_table().to_csv(csv_path,index=False)
    json_path.write_text(json.dumps({"snapshot":asdict(snap),"sources_notes":{
        "hill_formula":"mutual Hill radius and pairwise 2sqrt(3) criterion",
        "visualization":"independent Kepler orbits around fixed star; no planet-planet gravity",
    }},indent=2),encoding="utf-8")
    return csv_path,json_path


def save_preview_frames(scene: ThousandPlanetScene) -> List[Path]:
    paths=[]
    for i,shot in enumerate(SHOT_PLAN,1):
        t=(shot["start"]+shot["end"])*.5
        arr=scene.render(min(DURATION-1/FPS,t))
        p=PREVIEW_ROOT/f"preview_{i:02d}_{shot['name']}.jpg"; Image.fromarray(arr).save(p,quality=92); paths.append(p)
    return paths


def make_contact_sheet(paths: Sequence[Path]) -> Path:
    tw=300 if not QUICK_MODE else 200; th=int(tw*16/9); thumbs=[]
    for p in paths:
        im=Image.open(p).convert("RGB"); im.thumbnail((tw,th),Image.Resampling.LANCZOS)
        c=Image.new("RGB",(tw,th),(5,8,18)); c.paste(im,((tw-im.width)//2,(th-im.height)//2)); thumbs.append(c)
    cols=3; rows=math.ceil(len(thumbs)/cols); sheet=Image.new("RGB",(tw*cols,th*rows),(3,6,15))
    for i,im in enumerate(thumbs): sheet.paste(im,((i%cols)*tw,(i//cols)*th))
    out=PREVIEW_ROOT/f"{CONFIG['basename']}_contact_sheet.jpg"; sheet.save(out,quality=92); return out


def render_video(scene: ThousandPlanetScene) -> Path:
    out=OUTPUT_ROOT/f"{CONFIG['basename']}{'_quick_preview' if QUICK_MODE else ''}.mp4"
    total=max(1,int(round(DURATION*FPS)))
    writer=iio.get_writer(out,fps=FPS,codec="libx264",quality=8,macro_block_size=None,output_params=["-pix_fmt","yuv420p","-movflags","+faststart"])
    try:
        for frame in tqdm(range(total),desc="Rendering 1,000-planet short"):
            writer.append_data(scene.render(frame/FPS))
    finally:
        writer.close()
    return out


def write_readme(snap: ThousandPlanetSnapshot, outputs: Dict[str,str]) -> Path:
    path=OUTPUT_ROOT/"README.txt"
    lines=[
        CONFIG["title"],"="*len(CONFIG["title"]),"",
        "This renderer shows 1,000 Earth-mass planets on independent Kepler orbits for visualization.",
        "It does NOT claim to be a full 1,000-body integration.","",
        f"Mean adjacent spacing: {snap.mean_neighbor_spacing_mutual_hill:.3f} mutual Hill radii",
        f"Classic circular coplanar pairwise Hill limit: {snap.pairwise_hill_limit:.3f}",
        f"Same-range count at that floor: about {snap.estimated_count_over_same_range_at_pairwise_limit}",
        f"Same-range count at 10 mutual Hill radii: about {snap.estimated_count_over_same_range_at_10_hill}",
        f"Same-range count at 12 mutual Hill radii: about {snap.estimated_count_over_same_range_at_12_hill}","",
        "Outputs:",
    ]
    lines.extend([f"- {k}: {v}" for k,v in outputs.items()])
    path.write_text("\n".join(lines),encoding="utf-8"); return path



# =============================================================================
# YouTube title / description sidecar
# =============================================================================

YOUTUBE_TITLE = 'I Put 1,000 Planets Around One Star 🪐 #rootjatin'
YOUTUBE_DESCRIPTION = 'This is a hypothetical orbital-crowding experiment: one thousand Earth-mass planets are packed around a Sun-like star from 0.1 to 10 AU. Circular Kepler orbits may look orderly at first, but the neighboring worlds are far inside ordinary Hill-stability spacing, so mutual gravity would drive strong interactions, close encounters, collisions, mergers, or ejections. The visualization is an educational stability experiment, not a real observed planetary system.'
YOUTUBE_HASHTAGS = '#rootjatin #Planets #OrbitalMechanics #Astronomy #Space #Simulation #Physics #Science'

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
    snap=snapshot(); csv_path,json_path=save_data(snap); srt_path=write_srt(OUTPUT_ROOT/f"{CONFIG['basename']}_subtitles.srt")
    scene=ThousandPlanetScene(snap); previews=save_preview_frames(scene); contact=make_contact_sheet(previews); video=render_video(scene)
    outputs={"video":str(video),"subtitles":str(srt_path),"contact_sheet":str(contact),"architecture_csv":str(csv_path),"snapshot_json":str(json_path)}
    outputs["readme"]=str(write_readme(snap,outputs))
    print(json.dumps({"snapshot":asdict(snap),"outputs":outputs},indent=2))
    metadata_txt = write_youtube_metadata_txt()
    print("Title/description TXT:", metadata_txt.resolve())


if __name__=="__main__":
    main()
