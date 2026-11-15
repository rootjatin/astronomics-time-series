
from __future__ import annotations

"""
WHICH IS THE COLDEST REGION OF THE SUN? — cinematic YouTube Shorts renderer
output : https://www.youtube.com/shorts/JaV3a4uQWKA

This short answers a surprisingly subtle question:

- The quiet solar atmosphere reaches a TEMPERATURE MINIMUM near the boundary between
  the photosphere and chromosphere, around 4,300 K in standard solar-atmosphere models.
- But the dark umbra of a SUNSPOT is a localized magnetic feature that can be cooler,
  around 3,700 K.

So:
    coldest normal atmospheric LAYER  -> temperature-minimum region (~4,300 K)
    cooler localized FEATURE          -> sunspot umbra (~3,700 K)

The renderer is an educational visualization, not a temperature measurement instrument.
It uses a procedural Sun and a simplified radial temperature profile based on standard
NASA solar-temperature reference values.

Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    SUN_COLD_QUICK=1 python which_is_coldest_region_of_the_sun.py

Outputs
-------
- final vertical MP4
- silent MP4 fallback
- ambient WAV
- SRT subtitles
- preview PNG frames
- CSV of reference temperatures
- JSON source / science notes

Reference context used by the script
------------------------------------
- Sun core: about 15 million °C / K-scale temperature
- Photosphere: about 5,500 °C, roughly 5,800 K
- Temperature minimum: about 4,300 K
- Sunspot dark center / umbra: about 3,700 K
- Corona: up to about 2 million °C

Primary reference pages:
- NASA Sun facts: https://science.nasa.gov/sun/facts/
- NASA Marshall Solar Physics sunspots:
  https://solarscience.msfc.nasa.gov/feature1.shtml
- NASA technical solar-atmosphere reference:
  https://ntrs.nasa.gov/
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
from typing import Any, Dict, List, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("SUN_COLD_QUICK", "0") == "1"

OUTPUT_ROOT = Path("which_is_coldest_region_of_the_sun_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12.0 if QUICK_MODE else 58.0,
    "title": "WHICH IS THE COLDEST REGION OF THE SUN?",
    "subtitle": "TEMPERATURE MINIMUM // SUNSPOTS // CORONA",
    "output_basename": "which_is_coldest_region_of_the_sun",
    "sample_rate": 22050 if QUICK_MODE else 44100,
    "grain_strength": 3.0,
    "contrast": 1.10,
    "saturation": 1.08,
    "vignette": 0.34,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0

COLORS = {
    "space0": (2, 2, 7),
    "space1": (10, 5, 13),
    "sun_dark": (96, 23, 8),
    "sun": (255, 133, 24),
    "sun_bright": (255, 226, 120),
    "hot": (255, 83, 31),
    "corona": (250, 240, 215),
    "cool": (66, 213, 255),
    "cool2": (85, 117, 255),
    "violet": (185, 92, 255),
    "white": (248, 250, 252),
    "muted": (170, 181, 191),
    "grid": (92, 75, 72),
    "dark": (8, 6, 10),
    "spot": (19, 14, 17),
    "penumbra": (75, 48, 39),
}

FULL_SHOT_PLAN = [
    {"name": "opening", "start": 0.0, "end": 6.0},
    {"name": "journey", "start": 6.0, "end": 20.0},
    {"name": "minimum", "start": 20.0, "end": 31.0},
    {"name": "sunspot", "start": 31.0, "end": 41.5},
    {"name": "why", "start": 41.5, "end": 50.5},
    {"name": "answer", "start": 50.5, "end": 56.0},
    {"name": "finale", "start": 56.0, "end": 58.0},
]

FULL_CAPTIONS = [
    (0.4, 5.7, "The coldest part of the Sun is not where you might expect."),
    (6.2, 19.7, "From the fifteen-million-degree core, temperature falls outward until the visible photosphere is only about fifty-eight hundred kelvin."),
    (20.2, 30.7, "Just above the photosphere, the quiet solar atmosphere reaches a temperature minimum near forty-three hundred kelvin."),
    (31.2, 41.2, "But a sunspot is different. Its dark umbra can be around thirty-seven hundred kelvin, cooler than the surrounding photosphere."),
    (41.8, 50.2, "Strong magnetic fields suppress convection, reducing the flow of heat into the sunspot."),
    (50.8, 55.7, "So the coldest normal layer is the temperature-minimum region. A sunspot can be an even cooler local feature."),
    (56.1, 57.8, "The Sun gets cooler outward — then suddenly heats to a million-degree corona."),
]

if QUICK_MODE:
    k = float(CONFIG["duration_s"]) / 58.0
    SHOT_PLAN = [{**s, "start": s["start"] * k, "end": s["end"] * k} for s in FULL_SHOT_PLAN]
    CAPTIONS = [(a * k, b * k, text) for a, b, text in FULL_CAPTIONS]
else:
    SHOT_PLAN = FULL_SHOT_PLAN
    CAPTIONS = FULL_CAPTIONS


# -----------------------------------------------------------------------------
# Science data
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SolarRegion:
    name: str
    altitude_km: float
    temperature_k: float
    kind: str
    note: str


REGIONS: List[SolarRegion] = [
    SolarRegion("CORE", -696_000, 15_000_000, "interior", "Energy-producing center"),
    SolarRegion("RADIATIVE ZONE", -500_000, 7_000_000, "interior", "Energy moves mainly by radiation"),
    SolarRegion("CONVECTION ZONE", -150_000, 2_000_000, "interior", "Hot plasma rises and falls"),
    SolarRegion("PHOTOSPHERE", 0, 5_800, "atmosphere", "Visible solar surface"),
    SolarRegion("TEMPERATURE MINIMUM", 500, 4_300, "atmosphere", "Coldest normal atmospheric layer"),
    SolarRegion("CHROMOSPHERE", 2_000, 10_000, "atmosphere", "Temperature starts climbing again"),
    SolarRegion("TRANSITION REGION", 5_000, 100_000, "atmosphere", "Very steep temperature rise"),
    SolarRegion("CORONA", 20_000, 1_500_000, "atmosphere", "Outer atmosphere"),
]

SUNSPOT_UMBRA_K = 3_700
PHOTOSPHERE_K = 5_800
TEMP_MIN_K = 4_300


def save_science_products() -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "solar_temperature_reference.csv"
    json_path = DATA_ROOT / "solar_temperature_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "altitude_km_relative_to_photosphere", "temperature_k", "kind", "note"])
        for r in REGIONS:
            writer.writerow([r.name, r.altitude_km, r.temperature_k, r.kind, r.note])
        writer.writerow(["SUNSPOT UMBRA", 0, SUNSPOT_UMBRA_K, "localized magnetic feature", "Cooler local feature within the photosphere"])

    summary = {
        "title": CONFIG["title"],
        "science_answer": {
            "coldest_quiet_atmospheric_layer": "temperature-minimum region",
            "temperature_minimum_k": TEMP_MIN_K,
            "localized_cool_feature": "sunspot umbra",
            "sunspot_umbra_k_about": SUNSPOT_UMBRA_K,
        },
        "visualization_note": "Temperatures and radial distances are schematic and not drawn to physical scale.",
        "source_urls": {
            "nasa_sun_facts": "https://science.nasa.gov/sun/facts/",
            "nasa_marshall_sunspots": "https://solarscience.msfc.nasa.gov/feature1.shtml",
            "nasa_technical_reports": "https://ntrs.nasa.gov/",
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def smoothstep(x: float) -> float:
    x = clamp(x)
    return x * x * (3 - 2 * x)


def smootherstep(x: float) -> float:
    x = clamp(x)
    return x * x * x * (x * (x * 6 - 15) + 10)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if float(shot["start"]) <= t < float(shot["end"]):
            return shot
    return SHOT_PLAN[-1]


def shot_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - float(shot["start"])) / max(float(shot["end"] - shot["start"]), 1e-9))


def get_font(size: int, bold: bool = False, condensed: bool = False):
    candidates: List[str] = []
    if condensed and bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
            "DejaVuSansCondensed-Bold.ttf",
        ]
    if condensed:
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
            return ImageFont.truetype(candidate, size=max(5, int(size)))
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
        font=get_font(size, bold, condensed),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(235, fill[3])),
    )


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for i, (a, b, text) in enumerate(captions, 1):
        lines.extend([str(i), f"{format_srt_time(a)} --> {format_srt_time(b)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius ** 1.8, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Renderer
# -----------------------------------------------------------------------------

class SunColdScene:
    def __init__(self):
        self.rng = np.random.default_rng(66091)
        self.stars = [
            (
                float(self.rng.uniform(0, OUT_W)),
                float(self.rng.uniform(0, OUT_H)),
                float(self.rng.uniform(0.3, 2.1 if not QUICK_MODE else 1.3)),
                float(self.rng.uniform(10, 75)),
            )
            for _ in range(180 if not QUICK_MODE else 65)
        ]
        self.granules = self._make_granules(360 if not QUICK_MODE else 120)
        self.dust = [
            (
                float(self.rng.uniform(0, OUT_W)),
                float(self.rng.uniform(0, OUT_H)),
                float(self.rng.uniform(0, math.tau)),
            )
            for _ in range(90 if not QUICK_MODE else 30)
        ]

    def _make_granules(self, count: int):
        vals = []
        rng = np.random.default_rng(118)
        for _ in range(count):
            theta = float(rng.uniform(0, math.tau))
            rr = math.sqrt(float(rng.uniform(0, 1)))
            vals.append({
                "theta": theta,
                "rr": rr,
                "size": float(rng.uniform(2.5, 10.5)),
                "phase": float(rng.uniform(0, math.tau)),
                "brightness": float(rng.uniform(0.45, 1.0)),
            })
        return vals

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
        top = np.array(COLORS["space1"], dtype=float)
        bottom = np.array(COLORS["space0"], dtype=float)
        for y in range(OUT_H):
            u = y / max(OUT_H - 1, 1)
            arr[y, :, :3] = (top * (1-u) + bottom * u).astype(np.uint8)
            arr[y, :, 3] = 255
        im = Image.fromarray(arr, "RGBA")
        dr = ImageDraw.Draw(im)
        for x, y, r, a in self.stars:
            pulse = 0.5 + 0.5 * math.sin(t*0.55 + x*0.011 + y*0.008) ** 2
            rr = r*SCALE
            dr.ellipse((x-rr, y-rr, x+rr, y+rr), fill=(234, 226, 220, int(a*pulse)))
        return im

    @property
    def sun_center(self):
        return OUT_W * 0.5, OUT_H * 0.48

    @property
    def sun_radius(self):
        return 305 * SCALE

    def draw_sun(self, image: Image.Image, t: float, spot_alpha: float = 0.0, zoom: float = 1.0):
        cx, cy = self.sun_center
        r = self.sun_radius * zoom

        glow = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        for mul, alpha in [(1.55, 7), (1.25, 15), (1.10, 32)]:
            rr = r * mul
            gd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=COLORS["sun"]+(alpha,))
        glow = glow.filter(ImageFilter.GaussianBlur(max(3,int(25*SCALE))))
        image.alpha_composite(glow)

        base = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        bd = ImageDraw.Draw(base)
        bd.ellipse((cx-r,cy-r,cx+r,cy+r),fill=COLORS["sun"]+(255,),outline=COLORS["sun_bright"]+(180,),width=max(1,int(3*SCALE)))

        for g in self.granules:
            theta = g["theta"] + 0.015*t
            rr = g["rr"] * r * 0.93
            x = cx + math.cos(theta)*rr
            y = cy + math.sin(theta)*rr
            if (x-cx)**2 + (y-cy)**2 > (r*0.95)**2:
                continue
            sz = g["size"]*SCALE*(0.8+0.2*math.sin(t*0.7+g["phase"]))
            b = g["brightness"]
            col = (
                int(245 + 10*b),
                int(115 + 75*b),
                int(18 + 36*b),
                int(65 + 65*b),
            )
            bd.ellipse((x-sz,y-sz*0.65,x+sz,y+sz*0.65),fill=col)

        # Limb darkening.
        shade = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        sd = ImageDraw.Draw(shade)
        for i in range(15):
            frac = (i+1)/15
            rr = r*(1-frac*0.055)
            alpha = int(4+frac*2)
            sd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=(50,7,3,alpha),width=max(1,int(5*SCALE)))
        base.alpha_composite(shade)

        if spot_alpha > 0:
            self.draw_sunspot(base, cx-r*0.20, cy+r*0.06, r*0.19, spot_alpha)

        image.alpha_composite(base)

    def draw_sunspot(self, image: Image.Image, cx: float, cy: float, r: float, alpha: float = 1.0):
        dr = ImageDraw.Draw(image)
        a = int(255*clamp(alpha))
        # Irregular penumbra rings.
        pts=[]
        for i in range(90):
            ang=i/90*math.tau
            noise=1.0+0.12*math.sin(ang*5+0.9)+0.07*math.sin(ang*11+1.7)
            pts.append((cx+math.cos(ang)*r*noise,cy+math.sin(ang)*r*0.75*noise))
        dr.polygon(pts,fill=COLORS["penumbra"]+(a,))
        pts2=[]
        for i in range(90):
            ang=i/90*math.tau
            noise=1.0+0.08*math.sin(ang*6+0.3)+0.05*math.sin(ang*13)
            pts2.append((cx+math.cos(ang)*r*0.53*noise,cy+math.sin(ang)*r*0.40*noise))
        dr.polygon(pts2,fill=COLORS["spot"]+(a,))
        for k in range(28):
            ang=(k/28)*math.tau + 0.13*math.sin(k)
            x0=cx+math.cos(ang)*r*0.52
            y0=cy+math.sin(ang)*r*0.40
            x1=cx+math.cos(ang)*r*0.95
            y1=cy+math.sin(ang)*r*0.69
            dr.line((x0,y0,x1,y1),fill=(195,91,49,int(a*0.38)),width=max(1,int(2*SCALE)))

    def draw_header(self, image: Image.Image, scene_name: str):
        draw_text(
            image, CONFIG["title"],
            (int(48*SCALE), int(64*SCALE)),
            int(31*SCALE),
            COLORS["white"]+(245,),
            True, True,
            stroke=max(1,int(2*SCALE)),
        )
        labels = {
            "opening": "HOTTEST → COLDEST → HOT AGAIN",
            "journey": "TEMPERATURE JOURNEY",
            "minimum": "THE TEMPERATURE MINIMUM",
            "sunspot": "BUT SUNSPOTS GO LOWER",
            "why": "MAGNETIC FIELDS BLOCK HEAT FLOW",
            "answer": "LAYER vs LOCAL FEATURE",
            "finale": "THE SUN'S TEMPERATURE PARADOX",
        }
        draw_text(
            image, labels.get(scene_name,""),
            (int(50*SCALE), int(116*SCALE)),
            int(14*SCALE),
            COLORS["cool"]+(220,),
            True, True,
            stroke=1,
        )

    def draw_opening(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smootherstep(shot_progress(t, shot))
        self.draw_sun(image,t,spot_alpha=0.0,zoom=0.88)
        cx,cy=self.sun_center
        draw_text(image,"15,000,000 K",(int(cx),int(cy-70*SCALE)),int(40*SCALE),COLORS["sun_bright"]+(245,),True,True,"ma")
        draw_text(image,"CORE",(int(cx),int(cy-20*SCALE)),int(18*SCALE),COLORS["white"]+(230,),True,True,"ma")
        temp=int(round(lerp(15_000_000,4_300,p)))
        label=f"{temp:,} K" if temp>100_000 else f"{temp:,.0f} K"
        draw_text(image,label,(int(cx),int(cy+385*SCALE)),int(42*SCALE),COLORS["cool"]+(245,),True,True,"ma")
        draw_text(image,"How low does the Sun go?",(int(cx),int(cy+440*SCALE)),int(18*SCALE),COLORS["muted"]+(220,),True,True,"ma")

    def draw_journey(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p=shot_progress(t,shot)
        self.draw_sun(image,t,zoom=0.70)
        cx,cy=self.sun_center

        # Temperature ladder on right.
        x=OUT_W*0.80
        y0=OUT_H*0.26
        y1=OUT_H*0.76
        dr=ImageDraw.Draw(image)
        dr.line((x,y0,x,y1),fill=COLORS["grid"]+(160,),width=max(2,int(4*SCALE)))
        stations=[
            ("CORE",15_000_000,COLORS["sun_bright"]),
            ("CONVECTION",2_000_000,COLORS["hot"]),
            ("PHOTOSPHERE",5_800,COLORS["sun"]),
            ("TEMP MIN",4_300,COLORS["cool"]),
            ("CORONA",1_500_000,COLORS["violet"]),
        ]
        ys=[y0,y0+(y1-y0)*0.25,y0+(y1-y0)*0.55,y0+(y1-y0)*0.68,y1]
        reached=int(p*(len(stations)-1)+0.999)
        for i,((name,temp,col),yy) in enumerate(zip(stations,ys)):
            active=i<=reached
            a=240 if active else 70
            dr.ellipse((x-7*SCALE,yy-7*SCALE,x+7*SCALE,yy+7*SCALE),fill=col+(a,))
            draw_text(image,name,(int(x-18*SCALE),int(yy)),int(13*SCALE),COLORS["white"]+(a,),True,True,"ra",1)
            txt=f"{temp/1_000_000:.1f}M K" if temp>=1_000_000 else f"{temp:,} K"
            draw_text(image,txt,(int(x+20*SCALE),int(yy)),int(12*SCALE),col+(a,),True,True,"la",1)

        # Moving radial scan line.
        ang=-math.pi/2+0.35
        rr=lerp(20*SCALE,self.sun_radius*0.78,p)
        sx=cx+math.cos(ang)*rr
        sy=cy+math.sin(ang)*rr
        dr.line((cx,cy,sx,sy),fill=COLORS["cool"]+(170,),width=max(1,int(3*SCALE)))
        dr.ellipse((sx-7*SCALE,sy-7*SCALE,sx+7*SCALE,sy+7*SCALE),fill=COLORS["cool"]+(245,))

    def draw_minimum(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p=smootherstep(shot_progress(t,shot))
        cx=OUT_W*0.5
        base_y=OUT_H*0.56
        dr=ImageDraw.Draw(image)

        # Atmospheric stack.
        bands=[
            ("PHOTOSPHERE",5_800,COLORS["sun"],0.16),
            ("TEMPERATURE MINIMUM",4_300,COLORS["cool2"],0.07),
            ("CHROMOSPHERE",10_000,COLORS["hot"],0.13),
            ("TRANSITION",100_000,COLORS["violet"],0.08),
            ("CORONA",1_500_000,COLORS["corona"],0.20),
        ]
        y=OUT_H*0.72
        for i,(name,temp,col,hfrac) in enumerate(bands):
            h=OUT_H*hfrac
            y0=y-h
            a=80 if i!=1 else int(120+100*p)
            dr.rectangle((58*SCALE,y0,OUT_W-58*SCALE,y),fill=col+(a,),outline=col+(130,),width=max(1,int(2*SCALE)))
            draw_text(image,name,(int(80*SCALE),int((y+y0)/2-8*SCALE)),int(14*SCALE),COLORS["white"]+(220,),True,True)
            draw_text(image,f"{temp:,} K",(int(OUT_W-80*SCALE),int((y+y0)/2+15*SCALE)),int(15*SCALE),col+(245,),True,True,"ra")
            y=y0

        draw_text(image,"COLDEST NORMAL LAYER",(int(cx),int(OUT_H*0.18)),int(30*SCALE),COLORS["white"]+(245,),True,True,"ma")
        draw_text(image,"≈ 4,300 K",(int(cx),int(OUT_H*0.24)),int(52*SCALE),COLORS["cool"]+(250,),True,True,"ma")
        draw_text(image,"a few hundred km above the visible surface",(int(cx),int(OUT_H*0.29)),int(15*SCALE),COLORS["muted"]+(215,),True,True,"ma",1)

    def draw_sunspot_scene(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p=smootherstep(shot_progress(t,shot))
        self.draw_sun(image,t,spot_alpha=p,zoom=0.90)
        cx,cy=self.sun_center
        sx=cx-self.sun_radius*0.90*0.20
        sy=cy+self.sun_radius*0.90*0.06
        dr=ImageDraw.Draw(image)
        dr.line((sx,sy,sx-190*SCALE,sy+220*SCALE),fill=COLORS["cool"]+(180,),width=max(1,int(2*SCALE)))
        draw_text(image,"SUNSPOT UMBRA",(int(sx-200*SCALE),int(sy+235*SCALE)),int(17*SCALE),COLORS["white"]+(235,),True,True,"ra",1)
        draw_text(image,"≈ 3,700 K",(int(sx-200*SCALE),int(sy+275*SCALE)),int(35*SCALE),COLORS["cool"]+(250,),True,True,"ra")
        draw_text(image,"cooler local feature",(int(sx-200*SCALE),int(sy+312*SCALE)),int(14*SCALE),COLORS["muted"]+(215,),True,True,"ra",1)

    def draw_why(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p=shot_progress(t,shot)
        cx,cy=OUT_W*0.5,OUT_H*0.48
        dr=ImageDraw.Draw(image)
        # Convection cells.
        for i in range(7):
            x=lerp(95*SCALE,OUT_W-95*SCALE,i/6)
            phase=p*math.tau*1.7+i*0.9
            y=cy+math.sin(phase)*95*SCALE
            dr.ellipse((x-28*SCALE,y-28*SCALE,x+28*SCALE,y+28*SCALE),fill=COLORS["hot"]+(90,),outline=COLORS["sun_bright"]+(100,),width=max(1,int(2*SCALE)))
        # Magnetic arch over center suppressing motion.
        for offset in (-70,-35,0,35,70):
            x0=cx+offset*SCALE
            box=(x0-95*SCALE,cy-230*SCALE,x0+95*SCALE,cy+40*SCALE)
            dr.arc(box,190,350,fill=COLORS["cool"]+(170,),width=max(2,int(4*SCALE)))
        dr.ellipse((cx-90*SCALE,cy-45*SCALE,cx+90*SCALE,cy+45*SCALE),fill=COLORS["spot"]+(230,),outline=COLORS["cool"]+(110,),width=max(2,int(3*SCALE)))
        draw_text(image,"STRONG MAGNETIC FIELD",(int(cx),int(OUT_H*0.69)),int(26*SCALE),COLORS["cool"]+(245,),True,True,"ma")
        draw_text(image,"suppresses convection",(int(cx),int(OUT_H*0.74)),int(18*SCALE),COLORS["white"]+(230,),True,True,"ma")
        draw_text(image,"less heat reaches the surface",(int(cx),int(OUT_H*0.79)),int(18*SCALE),COLORS["muted"]+(220,),False,True,"ma")

    def draw_answer(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p=smoothstep(shot_progress(t,shot))
        dr=ImageDraw.Draw(image)
        margin=60*SCALE
        gap=24*SCALE
        card_w=(OUT_W-2*margin-gap)/2
        y0=OUT_H*0.30
        y1=OUT_H*0.72

        cards=[
            (
                margin, "COLDEST\nNORMAL LAYER",
                "TEMPERATURE\nMINIMUM",
                "≈ 4,300 K",
                COLORS["cool2"],
            ),
            (
                margin+card_w+gap, "COOLER\nLOCAL FEATURE",
                "SUNSPOT\nUMBRA",
                "≈ 3,700 K",
                COLORS["cool"],
            ),
        ]
        for x0,title,name,temp,col in cards:
            x1=x0+card_w
            dr.rounded_rectangle((x0,y0,x1,y1),radius=26*SCALE,fill=COLORS["dark"]+(220,),outline=col+(190,),width=max(2,int(3*SCALE)))
            yy=y0+52*SCALE
            for line in title.split("\n"):
                draw_text(image,line,(int((x0+x1)/2),int(yy)),int(15*SCALE),COLORS["muted"]+(220,),True,True,"ma",1)
                yy+=27*SCALE
            yy=y0+175*SCALE
            for line in name.split("\n"):
                draw_text(image,line,(int((x0+x1)/2),int(yy)),int(25*SCALE),COLORS["white"]+(240,),True,True,"ma")
                yy+=37*SCALE
            draw_text(image,temp,(int((x0+x1)/2),int(y1-68*SCALE)),int(28*SCALE),col+(245,),True,True,"ma")
        draw_text(image,"Both are still hotter than molten rock.",(int(OUT_W/2),int(OUT_H*0.80)),int(17*SCALE),COLORS["white"]+(225,),True,True,"ma")

    def draw_finale(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        self.draw_sun(image,t,spot_alpha=1.0,zoom=0.58)
        cx=OUT_W/2
        draw_text(image,"THE SUN GETS COOLER…",(int(cx),int(OUT_H*0.29)),int(31*SCALE),COLORS["white"]+(245,),True,True,"ma")
        draw_text(image,"THEN HOTTER AGAIN",(int(cx),int(OUT_H*0.35)),int(34*SCALE),COLORS["violet"]+(245,),True,True,"ma")
        draw_text(image,"CORONA  ≳ 1,000,000 K",(int(cx),int(OUT_H*0.73)),int(18*SCALE),COLORS["corona"]+(230,),True,True,"ma",1)

    def draw_source(self, image: Image.Image):
        draw_text(
            image,
            "NASA SOLAR TEMPERATURE REFERENCES // SCHEMATIC VISUALIZATION",
            (int(48*SCALE), OUT_H-int(42*SCALE)),
            int(10*SCALE),
            COLORS["muted"]+(180,),
            True, True,
            stroke=1,
        )

    def draw_texture(self, image: Image.Image, t: float):
        ov=Image.new("RGBA",OUT_SIZE,(0,0,0,0))
        dr=ImageDraw.Draw(ov)
        for x,y,phase in self.dust:
            pulse=0.5+0.5*math.sin(t*0.8+phase)
            if pulse>0.70:
                xx=(x+t*3.5)%OUT_W
                yy=(y+math.sin(t*0.55+phase)*4*SCALE)%OUT_H
                dr.line((xx,yy,xx+8*SCALE,yy),fill=COLORS["sun_bright"]+(int(8*pulse),),width=1)
        offset=int((t*39)%9)
        for y in range(offset,OUT_H,9):
            dr.line((0,y,OUT_W,y),fill=(170,120,100,4),width=1)
        image.alpha_composite(ov)

    def render_frame(self, t: float) -> np.ndarray:
        shot=get_shot(t)
        name=str(shot["name"])
        image=self.background(t)

        if name=="opening":
            self.draw_opening(image,t,shot)
        elif name=="journey":
            self.draw_journey(image,t,shot)
        elif name=="minimum":
            self.draw_minimum(image,t,shot)
        elif name=="sunspot":
            self.draw_sunspot_scene(image,t,shot)
        elif name=="why":
            self.draw_why(image,t,shot)
        elif name=="answer":
            self.draw_answer(image,t,shot)
        else:
            self.draw_finale(image,t,shot)

        self.draw_header(image,name)
        self.draw_source(image)
        self.draw_texture(image,t)

        arr=np.asarray(image.convert("RGB"),dtype=np.float32)
        arr*=VIGNETTE[...,None]
        arr=np.clip(arr,0,255).astype(np.uint8)
        img=ImageEnhance.Contrast(Image.fromarray(arr)).enhance(float(CONFIG["contrast"]))
        img=ImageEnhance.Color(img).enhance(float(CONFIG["saturation"]))
        arr=np.asarray(img,dtype=np.int16)
        rng=np.random.default_rng(int(t*1000)+721)
        grain=rng.normal(0.0,float(CONFIG["grain_strength"]),arr.shape[:2])[:,:,None]
        return np.clip(arr+grain,0,255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Audio / video
# -----------------------------------------------------------------------------

def gaussian_envelope(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5*((times-center)/max(width,1e-6))**2)


def generate_soundtrack(path: Path) -> Path:
    sr=int(CONFIG["sample_rate"])
    dur=float(CONFIG["duration_s"])
    n=int(round(sr*dur))
    tt=np.arange(n,dtype=np.float64)/sr
    rng=np.random.default_rng(609)

    drone=0.09*np.sin(2*np.pi*41*tt)+0.045*np.sin(2*np.pi*61.5*tt+0.6)
    shimmer=0.025*np.sin(2*np.pi*(165+6*np.sin(2*np.pi*0.035*tt))*tt)

    noise=rng.normal(0,1,n)
    k=max(8,int(sr*0.018))
    kernel=np.ones(k)/k
    soft=np.convolve(noise,kernel,mode="same")*0.045

    impacts=np.zeros(n)
    for shot in SHOT_PLAN[1:]:
        c=float(shot["start"])+0.10
        impacts += 0.12*gaussian_envelope(tt,c,0.18)*np.sin(2*np.pi*78*tt)

    pulses=np.zeros(n)
    for c in np.arange(0.8,dur,2.6 if not QUICK_MODE else 0.8):
        pulses += 0.04*gaussian_envelope(tt,c,0.05)*np.sin(2*np.pi*420*tt)

    mix=drone+shimmer+soft+impacts+pulses
    fade=np.minimum(1.0,np.minimum(tt/0.7,(dur-tt)/0.8))
    mix*=np.clip(fade,0,1)
    mix=np.tanh(mix*1.45)
    pcm=np.clip(mix*32767*0.86,-32768,32767).astype(np.int16)

    with wave.open(str(path),"wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    ffmpeg=shutil.which("ffmpeg")
    if not ffmpeg and imageio_ffmpeg is not None:
        try:
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg=None
    if not ffmpeg:
        return False
    cmd=[
        ffmpeg,"-y","-i",str(video_path),"-i",str(audio_path),
        "-c:v","copy","-c:a","aac","-b:a","160k","-shortest",str(output_path),
    ]
    try:
        subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return output_path.exists() and output_path.stat().st_size>1000
    except Exception:
        return False


def render_video(scene: SunColdScene) -> Path:
    raw=OUTPUT_ROOT/(CONFIG["output_basename"]+"_silent.mp4")
    final=OUTPUT_ROOT/(CONFIG["output_basename"]+".mp4")
    audio=OUTPUT_ROOT/(CONFIG["output_basename"]+"_ambient.wav")
    fps=int(CONFIG["fps"])
    frames=int(round(float(CONFIG["duration_s"])*fps))
    times=np.arange(frames,dtype=float)/fps

    with iio.get_writer(
        raw,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times,desc="Rendering coldest-region-of-Sun short"):
            writer.append_data(scene.render_frame(float(t)))

    generate_soundtrack(audio)
    if mux_audio(raw,audio,final):
        return final
    shutil.copyfile(raw,final)
    return final


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


# =============================================================================
# YouTube title / description sidecar
# =============================================================================

YOUTUBE_TITLE = 'Which Is the Coldest Region of the Sun? ☀️❄️ #rootjatin'
YOUTUBE_DESCRIPTION = 'The Sun cools dramatically from its hot interior toward the visible surface, reaches a temperature minimum in the lower atmosphere, then heats again toward the corona. This video compares the quiet-Sun temperature-minimum region with cooler local features such as sunspot umbrae and explains how strong magnetic fields suppress convection in sunspots. It is an educational solar-physics visualization.'
YOUTUBE_HASHTAGS = '#rootjatin #Sun #SolarPhysics #Sunspots #Astronomy #Space #Science #Physics'

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
    csv_path,json_path=save_science_products()
    print("Title:",CONFIG["title"])
    print("Reference CSV:",csv_path.resolve())
    print("Summary:",json_path.resolve())
    print("Science answer: quiet-atmosphere temperature minimum ~4,300 K;")
    print("                localized sunspot umbra can be ~3,700 K.")

    scene=SunColdScene()

    preview_times=[
        1.3,
        min(10.0,float(CONFIG["duration_s"])*0.22),
        min(25.0,float(CONFIG["duration_s"])*0.45),
        min(35.0,float(CONFIG["duration_s"])*0.62),
        min(46.0,float(CONFIG["duration_s"])*0.80),
        float(CONFIG["duration_s"])-0.7,
    ]
    for pt in tqdm(preview_times,desc="Preview frames"):
        frame=scene.render_frame(float(pt))
        Image.fromarray(frame).save(PREVIEW_DIR/f"preview_{int(pt):02d}s.png")

    srt=write_srt(CAPTIONS,OUTPUT_ROOT/(CONFIG["output_basename"]+"_subtitles.srt"))
    final=render_video(scene)
    print("SRT:",srt.resolve())
    print("Final video:",final.resolve())
    print("Output directory:",OUTPUT_ROOT.resolve())
    metadata_txt = write_youtube_metadata_txt()
    print("Title/description TXT:", metadata_txt.resolve())


if __name__=="__main__":
    main()
