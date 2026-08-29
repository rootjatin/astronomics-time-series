from __future__ import annotations

"""
What If the Moon Had Oceans Like Earth? — cinematic YouTube Short renderer

Creates a vertical 1080x1920 science short imagining a Moon covered with large
liquid-water oceans. The scenario is intentionally hypothetical and the visuals
are cinematic/diagrammatic rather than a physical climate simulation.

Scientific framing used in the narration
-----------------------------------------
- The real Moon has essentially no substantial atmosphere, so stable Earth-like
  surface oceans require changing the premise: enough atmosphere and suitable
  temperatures must also be present.
- Lunar surface gravity is about one-sixth of Earth's, so an ocean world there
  would operate under very different gravity.
- Earth would be an important external tide-raising body for lunar oceans.
- A lunar solar day lasts about 29.5 Earth days, creating very long periods of
  daylight and darkness unless a thick atmosphere/ocean redistributed heat.
- With an atmosphere, evaporation, clouds, rain/snow and runoff could create a
  water cycle, but it would not simply copy Earth's climate.

Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    MOON_OCEAN_SHORT_QUICK=1 python what_if_moon_had_oceans_like_earth.py

Full render
-----------
    python what_if_moon_had_oceans_like_earth.py

4K vertical
-----------
    MOON_OCEAN_SHORT_4K=1 python what_if_moon_had_oceans_like_earth.py
"""

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("MOON_OCEAN_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("MOON_OCEAN_SHORT_4K", "0") == "1" and not QUICK_MODE

OUT_W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
OUT_H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 8 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SCALE = OUT_W / 1080.0
OUT_SIZE = (OUT_W, OUT_H)

OUTPUT_ROOT = Path("what_if_moon_had_oceans_like_earth_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "WHAT IF THE MOON HAD OCEANS LIKE EARTH?",
    "subtitle": "low gravity // Earth-raised tides // very long days",
    "output_basename": "what_if_moon_had_oceans_like_earth",
    "contrast": 1.09,
    "saturation": 1.10,
    "vignette": 0.26,
}

COLORS = {
    "space": (2, 5, 15),
    "space2": (7, 15, 34),
    "white": (246, 250, 255),
    "muted": (174, 193, 214),
    "cyan": (79, 225, 255),
    "blue": (48, 131, 238),
    "deep_blue": (8, 57, 126),
    "ocean": (16, 103, 184),
    "ocean_light": (67, 187, 240),
    "moon": (164, 167, 173),
    "moon_dark": (75, 79, 88),
    "highland": (190, 187, 174),
    "rock": (111, 105, 98),
    "gold": (255, 205, 91),
    "orange": (255, 143, 73),
    "red": (255, 85, 105),
    "violet": (176, 127, 255),
    "green": (103, 237, 170),
    "cloud": (235, 246, 255),
    "earth_ocean": (41, 115, 199),
    "earth_land": (86, 158, 104),
}

FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.4, 7.4, "Imagine looking up and seeing a blue Moon — not gray dust, but enormous oceans covering its lowlands."),
    (7.5, 17.2, "There is one catch: the real Moon has almost no atmosphere. Earth-like liquid oceans would need enough air and the right temperatures to stay stable."),
    (17.3, 27.2, "With only about one-sixth of Earth's surface gravity, this ocean world would feel very different. Water, waves and weather would all operate in weaker gravity."),
    (27.3, 38.0, "And Earth would loom in the lunar sky, raising tides in those oceans. Coastlines could repeatedly flood and retreat as the ocean responds to Earth's pull."),
    (38.1, 48.5, "A lunar day lasts about 29 and a half Earth days. That means roughly two weeks of sunlight followed by roughly two weeks of darkness at many locations."),
    (48.6, 57.4, "Give that Moon a thick enough atmosphere and a water cycle could emerge — evaporation, clouds, rain and rivers beneath a giant Earth in the sky."),
]

if QUICK_MODE:
    factor = DURATION / 58.0
    CAPTIONS = [(a * factor, b * factor, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "atmosphere", "start": 8.0 if not QUICK_MODE else 1.8, "end": 18.0 if not QUICK_MODE else 4.0},
    {"name": "gravity", "start": 18.0 if not QUICK_MODE else 4.0, "end": 28.0 if not QUICK_MODE else 6.25},
    {"name": "tides", "start": 28.0 if not QUICK_MODE else 6.25, "end": 39.0 if not QUICK_MODE else 8.7},
    {"name": "long_day", "start": 39.0 if not QUICK_MODE else 8.7, "end": 49.5 if not QUICK_MODE else 11.05},
    {"name": "water_cycle", "start": 49.5 if not QUICK_MODE else 11.05, "end": DURATION},
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(value: float) -> float:
    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
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
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(7, int(size * SCALE)))
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
    for word in words:
        candidate = word if not current else current + " " + word
        box = draw.textbbox((0, 0), candidate, font=font, stroke_width=max(1, int(2 * SCALE)))
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
            (x, y),
            line,
            font=font,
            fill=fill,
            stroke_width=max(1, int(2 * SCALE)),
            stroke_fill=(0, 0, 0, 220),
        )
        box = draw.textbbox((x, y), line, font=font, stroke_width=max(1, int(2 * SCALE)))
        y += (box[3] - box[1]) + int(spacing * SCALE)


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    rr = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * rr**1.8, 0.0, 1.0).astype(np.float32)


def apply_grade(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array)
    image = ImageEnhance.Contrast(image).enhance(float(CONFIG["contrast"]))
    image = ImageEnhance.Color(image).enhance(float(CONFIG["saturation"]))
    return np.asarray(image)


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for i, (start, end, value) in enumerate(captions, start=1):
        lines.extend([str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", value, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Renderer
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Cloud:
    u: float
    v: float
    size: float
    alpha: int
    speed: float
    phase: float


class MoonOceanScene:
    def __init__(self):
        rng = np.random.default_rng(20260818)
        self.stars = [
            (
                float(rng.uniform(0, OUT_W)),
                float(rng.uniform(0, OUT_H)),
                float(rng.uniform(0.35, 1.8) * SCALE),
                int(rng.uniform(30, 125)),
                float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(120 if QUICK_MODE else 340)
        ]
        self.clouds = [
            Cloud(
                float(rng.uniform(0, 1)),
                float(rng.uniform(-0.55, 0.55)),
                float(rng.uniform(22, 62) * SCALE),
                int(rng.uniform(28, 78)),
                float(rng.uniform(0.004, 0.015)),
                float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(24 if QUICK_MODE else 70)
        ]
        self.sparks = [
            (
                float(rng.uniform(-1, 1)),
                float(rng.uniform(-1, 1)),
                float(rng.uniform(0.2, 1.0)),
                float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(100)
        ]

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 175):
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.rounded_rectangle(
            box,
            radius=max(8, int(24 * SCALE)),
            fill=(3, 8, 20, alpha),
            outline=COLORS["cyan"] + (48,),
            width=max(1, int(2 * SCALE)),
        )
        image.alpha_composite(layer)

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        yy = np.linspace(0, 1, OUT_H)[:, None]
        arr[..., 0] = np.clip(2 + 8 * yy, 0, 255)
        arr[..., 1] = np.clip(5 + 14 * yy, 0, 255)
        arr[..., 2] = np.clip(15 + 28 * yy, 0, 255)
        image = Image.fromarray(arr, "RGB").convert("RGBA")
        draw = ImageDraw.Draw(image)
        for x, y, r, alpha, phase in self.stars:
            a = int(alpha * (0.72 + 0.28 * math.sin(t * 1.1 + phase)))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=COLORS["white"] + (a,))
        return image

    def draw_earth(self, image: Image.Image, t: float, x_frac: float = 0.78, y_frac: float = 0.22, scale: float = 1.0):
        cx = int(OUT_W * x_frac)
        cy = int(OUT_H * y_frac)
        r = int(82 * SCALE * scale)
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=COLORS["earth_ocean"] + (255,))
        continents = [
            [(-.58,-.15),(-.32,-.42),(-.08,-.34),(-.15,-.08),(-.40,.02)],
            [(.10,-.25),(.43,-.31),(.56,-.06),(.37,.12),(.14,.04)],
            [(-.08,.10),(.13,.12),(.25,.42),(.02,.55),(-.16,.30)],
        ]
        rot = 0.15 * math.sin(t * .08)
        for poly in continents:
            pts = []
            for px, py in poly:
                xx = px * math.cos(rot) - py * math.sin(rot)
                yy = px * math.sin(rot) + py * math.cos(rot)
                pts.append((cx + xx*r, cy + yy*r))
            draw.polygon(pts, fill=COLORS["earth_land"] + (235,))
        # cloud bands
        for k in range(3):
            yy = cy + int((-0.35 + 0.34*k) * r)
            draw.arc((cx-r*.86, yy-r*.20, cx+r*.86, yy+r*.20), 185, 355, fill=COLORS["white"]+(135,), width=max(1,int(5*SCALE)))
        # soft limb + glow
        draw.arc((cx-r, cy-r, cx+r, cy+r), 145, 330, fill=COLORS["white"]+(150,), width=max(1,int(3*SCALE)))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1, int(2*SCALE)))))
        image.alpha_composite(layer)
        glow = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse((cx-r-10*SCALE,cy-r-10*SCALE,cx+r+10*SCALE,cy+r+10*SCALE),outline=COLORS["cyan"]+(60,),width=max(2,int(9*SCALE)))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4,int(12*SCALE)))))

    def moon_geometry(self, center_y: float = .49, scale: float = 1.0) -> Tuple[int, int, int]:
        cx = OUT_W // 2
        cy = int(OUT_H * center_y)
        r = int(350 * SCALE * scale)
        return cx, cy, r

    def draw_ocean_moon(
        self,
        image: Image.Image,
        t: float,
        center_y: float = .49,
        scale: float = 1.0,
        ocean_alpha: int = 255,
        cloud_alpha: float = 0.0,
        atmosphere: float = 1.0,
        tide_strength: float = 0.0,
        terminator: Optional[float] = None,
    ):
        cx, cy, r = self.moon_geometry(center_y, scale)

        # atmosphere glow
        if atmosphere > 0:
            glow = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
            gd = ImageDraw.Draw(glow)
            for extra, alpha in [(34, int(20*atmosphere)), (18, int(48*atmosphere)), (8, int(82*atmosphere))]:
                rr = r + int(extra*SCALE)
                gd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=COLORS["cyan"]+(alpha,),width=max(2,int(8*SCALE)))
            image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(5,int(18*SCALE)))))

        layer = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        draw = ImageDraw.Draw(layer)
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=COLORS["ocean"] + (ocean_alpha,))

        # stylized highlands / continents — intentionally inspired by lunar highland shapes,
        # not a geographic reconstruction of a flooded Moon.
        highlands = [
            [(-.90,-.18),(-.72,-.52),(-.38,-.70),(-.10,-.62),(-.18,-.30),(-.45,-.16),(-.68,.03)],
            [(.08,-.72),(.40,-.60),(.72,-.31),(.79,-.02),(.54,.06),(.31,-.14),(.12,-.36)],
            [(-.57,.18),(-.28,.08),(-.02,.22),(-.07,.52),(-.28,.74),(-.55,.60),(-.72,.36)],
            [(.16,.16),(.43,.04),(.72,.20),(.64,.53),(.39,.70),(.12,.58),(.03,.36)],
        ]
        rot = 0.025 * math.sin(t * .18)
        for poly in highlands:
            pts=[]
            for px, py in poly:
                xx=px*math.cos(rot)-py*math.sin(rot)
                yy=px*math.sin(rot)+py*math.cos(rot)
                # mild roundness correction toward limb
                pts.append((cx+xx*r*.94,cy+yy*r*.94))
            draw.polygon(pts,fill=COLORS["highland"]+(238,))

        # smaller islands / crater rims
        for i, (px, py, w, phase) in enumerate(self.sparks[:30]):
            rr=(5+9*w)*SCALE
            x=cx+px*r*.77; y=cy+py*r*.77
            if (px*px+py*py) < .86:
                col=COLORS["rock"] if i%3 else COLORS["highland"]
                draw.ellipse((x-rr,y-rr*.6,x+rr,y+rr*.6),fill=col+(145,))

        # tidal bulge cue — exaggerated, purely diagrammatic
        if tide_strength > 0:
            tide = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
            td = ImageDraw.Draw(tide)
            bulge = int((12 + 26*tide_strength) * SCALE)
            td.arc((cx-r-bulge,cy-r*.75,cx+r+bulge,cy+r*.75),150,210,fill=COLORS["ocean_light"]+(185,),width=max(3,int(14*SCALE)))
            td.arc((cx-r-bulge,cy-r*.75,cx+r+bulge,cy+r*.75),-30,30,fill=COLORS["ocean_light"]+(185,),width=max(3,int(14*SCALE)))
            image.alpha_composite(tide.filter(ImageFilter.GaussianBlur(max(2,int(6*SCALE)))))
            image.alpha_composite(tide)

        # sunlit specular patch on ocean
        sheen = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        sd = ImageDraw.Draw(sheen)
        sx = cx-int(r*.34); sy=cy-int(r*.26)
        for k in range(5):
            rr=int((18+13*k)*SCALE)
            sd.ellipse((sx-rr*1.9,sy-rr*.55,sx+rr*1.9,sy+rr*.55),fill=COLORS["white"]+(max(0,45-7*k),))
        image.alpha_composite(sheen.filter(ImageFilter.GaussianBlur(max(5,int(14*SCALE)))))

        # terminator / long-night visualization
        if terminator is not None:
            shade = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
            sh = ImageDraw.Draw(shade)
            # terminator -1..1 moves from left to right; ellipse shadow gives spherical cue
            tx = cx + int(terminator * r * .72)
            sh.ellipse((tx-r*1.08,cy-r,tx+r*1.08,cy+r),fill=(1,4,14,165))
            image.alpha_composite(shade.filter(ImageFilter.GaussianBlur(max(2,int(8*SCALE)))))

        image.alpha_composite(layer)

        # rim
        rim = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        rd=ImageDraw.Draw(rim)
        rd.arc((cx-r,cy-r,cx+r,cy+r),155,340,fill=COLORS["cyan"]+(130,),width=max(2,int(5*SCALE)))
        image.alpha_composite(rim.filter(ImageFilter.GaussianBlur(max(2,int(7*SCALE)))))
        image.alpha_composite(rim)

        if cloud_alpha > 0:
            self.draw_clouds_on_moon(image,t,cx,cy,r,cloud_alpha)

    def draw_clouds_on_moon(self,image:Image.Image,t:float,cx:int,cy:int,r:int,alpha_scale:float):
        layer=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        for cloud in self.clouds:
            lon=((cloud.u+t*cloud.speed)%1.0)*2-1
            lat=cloud.v + .04*math.sin(t*.35+cloud.phase)
            if lon*lon + lat*lat > .88:
                continue
            # rough spherical compression near limb
            x=cx+lon*r*.87
            y=cy+lat*r*.78
            limb=max(.18,math.sqrt(max(0.0,1-lon*lon)))
            s=cloud.size*limb
            a=int(cloud.alpha*alpha_scale)
            for dx,dy,rr in [(-.45,.05,.40),(-.12,-.05,.54),(.25,0,.48),(.55,.08,.32)]:
                q=s*rr
                d.ellipse((x+dx*s-q,y+dy*s-q*.5,x+dx*s+q,y+dy*s+q*.5),fill=COLORS["cloud"]+(a,))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(2,int(5*SCALE)))))

    def draw_gravity_compare(self,image:Image.Image,t:float,local:float):
        # Earth / Moon gravity meter
        x0=int(OUT_W*.14); x1=int(OUT_W*.86); y=int(OUT_H*.69)
        d=ImageDraw.Draw(image)
        d.line((x0,y,x1,y),fill=COLORS["white"]+(95,),width=max(2,int(4*SCALE)))
        # Earth tick
        ex=int(lerp(x0,x1,1.0)); mx=int(lerp(x0,x1,1/6))
        d.line((mx,y-int(18*SCALE),mx,y+int(18*SCALE)),fill=COLORS["cyan"]+(230,),width=max(2,int(5*SCALE)))
        d.line((ex,y-int(18*SCALE),ex,y+int(18*SCALE)),fill=COLORS["gold"]+(230,),width=max(2,int(5*SCALE)))
        draw_text(image,"MOON ≈ 1/6 g",(mx,y-int(34*SCALE)),15 if not QUICK_MODE else 7,COLORS["cyan"]+(235,),True,"ma",1)
        draw_text(image,"EARTH = 1 g",(ex,y-int(34*SCALE)),15 if not QUICK_MODE else 7,COLORS["gold"]+(235,),True,"ma",1)

        # animated low-g splash arcs
        layer=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); ld=ImageDraw.Draw(layer)
        ox=int(OUT_W*.50); oy=int(OUT_H*.61)
        for i in range(16):
            p=(local*1.35+i/16)%1.0
            vx=(-1 if i%2 else 1)*(35+4*i)*SCALE
            vy=-(100+8*(i%5))*SCALE
            # stylized low-g ballistic arc, time normalized
            xx=ox+vx*p
            yy=oy+vy*p+72*SCALE*p*p
            rr=(2.5+2*(i%3))*SCALE
            ld.ellipse((xx-rr,yy-rr,xx+rr,yy+rr),fill=COLORS["ocean_light"]+(int(210*(1-p)),))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1,int(2*SCALE)))))

    def draw_tide_arrows(self,image:Image.Image,t:float,local:float):
        cx,cy,r=self.moon_geometry(.49,1.0)
        ex=int(OUT_W*.79); ey=int(OUT_H*.20)
        layer=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        # Earth-to-Moon pull line
        d.line((cx+int(r*.57),cy-int(r*.48),ex-int(45*SCALE),ey+int(45*SCALE)),fill=COLORS["violet"]+(120,),width=max(2,int(4*SCALE)))
        # tide arrows near lunar limbs
        for side in (-1,1):
            x=cx+side*int(r*.92); y=cy
            length=int((36+30*local)*SCALE)
            x2=x+side*length
            d.line((x,y,x2,y),fill=COLORS["cyan"]+(220,),width=max(2,int(5*SCALE)))
            tip=10*SCALE
            d.polygon([(x2,y),(x2-side*tip,y-tip*.65),(x2-side*tip,y+tip*.65)],fill=COLORS["cyan"]+(230,))
        image.alpha_composite(layer)

    def draw_day_clock(self,image:Image.Image,t:float,local:float):
        cx=int(OUT_W*.78); cy=int(OUT_H*.64); rr=int(93*SCALE)
        d=ImageDraw.Draw(image)
        d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=COLORS["white"]+(90,),width=max(2,int(4*SCALE)))
        d.arc((cx-rr,cy-rr,cx+rr,cy+rr),-90,90,fill=COLORS["gold"]+(230,),width=max(3,int(9*SCALE)))
        d.arc((cx-rr,cy-rr,cx+rr,cy+rr),90,270,fill=COLORS["blue"]+(210,),width=max(3,int(9*SCALE)))
        ang=math.radians(-90+360*local)
        ex=cx+math.cos(ang)*rr*.70; ey=cy+math.sin(ang)*rr*.70
        d.line((cx,cy,ex,ey),fill=COLORS["white"]+(235,),width=max(2,int(4*SCALE)))
        draw_text(image,"29.5 EARTH DAYS",(cx,cy+int(24*SCALE)),14 if not QUICK_MODE else 7,COLORS["white"]+(230,),True,"ma",1)

    def draw_water_cycle(self,image:Image.Image,t:float,local:float):
        cx,cy,r=self.moon_geometry(.49,1.0)
        layer=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        # evaporation arrows
        for i in range(5):
            x=cx+int((-0.45+i*.21)*r)
            y=cy+int(.35*r)
            wob=math.sin(t*1.2+i)*8*SCALE
            top=y-int((80+22*local)*SCALE)
            d.line((x,y,x+wob,top),fill=COLORS["cyan"]+(185,),width=max(1,int(4*SCALE)))
            d.polygon([(x+wob,top),(x+wob-7*SCALE,top+14*SCALE),(x+wob+7*SCALE,top+14*SCALE)],fill=COLORS["cyan"]+(195,))
        # rain
        for i in range(18):
            p=(t*.35+i/18)%1
            x=cx+int((.05+.47*((i%6)/5))*r)
            y=cy-int(.35*r)+p*int(.43*r)
            d.line((x,y,x-int(4*SCALE),y+int(18*SCALE)),fill=COLORS["ocean_light"]+(155,),width=max(1,int(2*SCALE)))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1,int(2*SCALE)))))
        image.alpha_composite(layer)

    def draw_title(self,image:Image.Image,t:float):
        if t >= (6.0 if not QUICK_MODE else 1.34):
            return
        fade=smoothstep(t/(.8 if not QUICK_MODE else .18))
        draw_text(image,"WHAT IF THE MOON",(OUT_W//2,int(OUT_H*.061)),37 if not QUICK_MODE else 18,COLORS["white"]+(int(245*fade),),True,"ma",2)
        draw_text(image,"HAD OCEANS",(OUT_W//2,int(OUT_H*.108)),54 if not QUICK_MODE else 27,COLORS["cyan"]+(int(250*fade),),True,"ma",2)
        draw_text(image,"LIKE EARTH?",(OUT_W//2,int(OUT_H*.157)),43 if not QUICK_MODE else 21,COLORS["gold"]+(int(245*fade),),True,"ma",2)

    def draw_caption(self,image:Image.Image,t:float):
        cap=caption_at(t)
        if not cap:
            return
        y0=OUT_H-(258 if not QUICK_MODE else 129)
        self.panel(image,(44 if not QUICK_MODE else 22,y0,OUT_W-(44 if not QUICK_MODE else 22),y0+(138 if not QUICK_MODE else 70)),180)
        draw_wrapped_text(image,cap,(68 if not QUICK_MODE else 34,y0+(27 if not QUICK_MODE else 14)),OUT_W-(136 if not QUICK_MODE else 68),27 if not QUICK_MODE else 13)

    def draw_corner_label(self,image:Image.Image,textv:str):
        draw_text(image,textv,(52 if not QUICK_MODE else 26,58 if not QUICK_MODE else 29),18 if not QUICK_MODE else 9,COLORS["muted"]+(205,),True,"la",1)

    def draw_source_hud(self,image:Image.Image):
        draw_text(image,"HYPOTHETICAL SCIENCE VISUALIZATION",(OUT_W-(48 if not QUICK_MODE else 24),72 if not QUICK_MODE else 36),14 if not QUICK_MODE else 7,COLORS["gold"]+(225,),True,"ra",1)
        draw_text(image,"NOT A CLIMATE OR OCEAN MODEL",(OUT_W-(48 if not QUICK_MODE else 24),100 if not QUICK_MODE else 50),13 if not QUICK_MODE else 6,COLORS["muted"]+(190,),False,"ra",1)

    def scene_intro(self,image:Image.Image,t:float):
        shot=SHOT_PLAN[0]; local=smoothstep((t-shot["start"])/max(shot["end"]-shot["start"],1e-9))
        self.draw_ocean_moon(image,t,.49,.96,cloud_alpha=.34+local*.26,atmosphere=.75)
        self.draw_earth(image,t,.78,.23,.82)
        self.panel(image,(int(OUT_W*.08),int(OUT_H*.69),int(OUT_W*.92),int(OUT_H*.83)),172)
        draw_text(image,"A BLUE MOON WITH REAL OCEANS",(OUT_W//2,int(OUT_H*.734)),25 if not QUICK_MODE else 12,COLORS["cyan"]+(248,),True,"ma",1)
        draw_text(image,"seas filling low basins • bright highlands becoming islands",(OUT_W//2,int(OUT_H*.783)),15 if not QUICK_MODE else 7,COLORS["white"]+(225,),False,"ma",1)
        draw_text(image,"but this changes more than just the color",(OUT_W//2,int(OUT_H*.816)),14 if not QUICK_MODE else 7,COLORS["gold"]+(220,),True,"ma",1)
        self.draw_corner_label(image,"1 // TURN THE MOON BLUE")

    def scene_atmosphere(self,image:Image.Image,t:float):
        shot=SHOT_PLAN[1]; local=smoothstep((t-shot["start"])/max(shot["end"]-shot["start"],1e-9))
        # transition from exposed/vacuum-like water to atmosphere-protected ocean
        self.draw_ocean_moon(image,t,.48,.96,cloud_alpha=.10+.55*local,atmosphere=.12+.88*local)
        cx,cy,r=self.moon_geometry(.48,.96)
        # atmosphere gauge
        d=ImageDraw.Draw(image)
        gx=int(OUT_W*.12); gy0=int(OUT_H*.26); gy1=int(OUT_H*.61)
        d.rounded_rectangle((gx-int(12*SCALE),gy0,gx+int(12*SCALE),gy1),radius=max(3,int(8*SCALE)),outline=COLORS["white"]+(100,),width=max(1,int(3*SCALE)))
        fill_y=int(lerp(gy1,gy0,local))
        d.rounded_rectangle((gx-int(8*SCALE),fill_y,gx+int(8*SCALE),gy1),radius=max(2,int(6*SCALE)),fill=COLORS["cyan"]+(220,))
        draw_text(image,"ATMOSPHERE",(gx+int(28*SCALE),int((gy0+gy1)/2)),15 if not QUICK_MODE else 7,COLORS["cyan"]+(225,),True,"lm",1)
        self.panel(image,(int(OUT_W*.08),int(OUT_H*.68),int(OUT_W*.92),int(OUT_H*.84)),184)
        draw_text(image,"LIQUID OCEANS NEED MORE THAN WATER",(OUT_W//2,int(OUT_H*.721)),23 if not QUICK_MODE else 11,COLORS["white"]+(245,),True,"ma",1)
        draw_text(image,"the real Moon has almost no atmosphere",(OUT_W//2,int(OUT_H*.769)),17 if not QUICK_MODE else 8,COLORS["red"]+(235,),True,"ma",1)
        draw_text(image,"our scenario also gives it enough air + suitable temperatures",(OUT_W//2,int(OUT_H*.812)),14 if not QUICK_MODE else 7,COLORS["cyan"]+(220,),False,"ma",1)
        self.draw_corner_label(image,"2 // FIRST: MAKE LIQUID WATER POSSIBLE")

    def scene_gravity(self,image:Image.Image,t:float):
        shot=SHOT_PLAN[2]; local=smoothstep((t-shot["start"])/max(shot["end"]-shot["start"],1e-9))
        self.draw_ocean_moon(image,t,.43,.84,cloud_alpha=.48,atmosphere=1.0)
        self.draw_gravity_compare(image,t,local)
        self.panel(image,(int(OUT_W*.08),int(OUT_H*.72),int(OUT_W*.92),int(OUT_H*.85)),178)
        draw_text(image,"THE MOON HAS ONLY ~1/6 EARTH'S SURFACE GRAVITY",(OUT_W//2,int(OUT_H*.758)),20 if not QUICK_MODE else 10,COLORS["gold"]+(245,),True,"ma",1)
        draw_text(image,"waves, spray and atmospheric motion would not behave exactly like Earth's",(OUT_W//2,int(OUT_H*.805)),14 if not QUICK_MODE else 7,COLORS["white"]+(220,),False,"ma",1)
        draw_text(image,"same water • very different world",(OUT_W//2,int(OUT_H*.836)),14 if not QUICK_MODE else 7,COLORS["cyan"]+(220,),True,"ma",1)
        self.draw_corner_label(image,"3 // OCEANS IN LOW GRAVITY")

    def scene_tides(self,image:Image.Image,t:float):
        shot=SHOT_PLAN[3]; local=smoothstep((t-shot["start"])/max(shot["end"]-shot["start"],1e-9))
        self.draw_ocean_moon(image,t,.49,.96,cloud_alpha=.52,atmosphere=1.0,tide_strength=.4+.6*local)
        self.draw_earth(image,t,.79,.20,1.08)
        self.draw_tide_arrows(image,t,local)
        self.panel(image,(int(OUT_W*.08),int(OUT_H*.68),int(OUT_W*.92),int(OUT_H*.84)),184)
        draw_text(image,"EARTH WOULD HELP RAISE LUNAR TIDES",(OUT_W//2,int(OUT_H*.723)),24 if not QUICK_MODE else 12,COLORS["violet"]+(245,),True,"ma",1)
        draw_text(image,"our planet is massive and close enough to strongly matter",(OUT_W//2,int(OUT_H*.772)),15 if not QUICK_MODE else 7,COLORS["white"]+(225,),False,"ma",1)
        draw_text(image,"tidal bulges shown here are exaggerated for clarity",(OUT_W//2,int(OUT_H*.812)),13 if not QUICK_MODE else 6,COLORS["muted"]+(215,),False,"ma",1)
        self.draw_corner_label(image,"4 // EARTH PULLS ON THE OCEAN")

    def scene_long_day(self,image:Image.Image,t:float):
        shot=SHOT_PLAN[4]; local=smoothstep((t-shot["start"])/max(shot["end"]-shot["start"],1e-9))
        terminator=-.7+1.4*local
        self.draw_ocean_moon(image,t,.45,.86,cloud_alpha=.40,atmosphere=1.0,terminator=terminator)
        self.draw_day_clock(image,t,local)
        # sun marker
        sx=int(OUT_W*.12); sy=int(OUT_H*.23); sr=int(42*SCALE)
        glow=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); gd=ImageDraw.Draw(glow)
        gd.ellipse((sx-sr,sy-sr,sx+sr,sy+sr),fill=COLORS["gold"]+(245,))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(5,int(18*SCALE)))))
        image.alpha_composite(glow)
        self.panel(image,(int(OUT_W*.08),int(OUT_H*.70),int(OUT_W*.92),int(OUT_H*.85)),186)
        draw_text(image,"ONE LUNAR SOLAR DAY ≈ 29.5 EARTH DAYS",(OUT_W//2,int(OUT_H*.744)),21 if not QUICK_MODE else 10,COLORS["gold"]+(245,),True,"ma",1)
        draw_text(image,"roughly two weeks of daylight, then roughly two weeks of darkness",(OUT_W//2,int(OUT_H*.791)),15 if not QUICK_MODE else 7,COLORS["white"]+(225,),False,"ma",1)
        draw_text(image,"oceans + atmosphere would have to redistribute a lot of heat",(OUT_W//2,int(OUT_H*.827)),14 if not QUICK_MODE else 7,COLORS["cyan"]+(220,),True,"ma",1)
        self.draw_corner_label(image,"5 // VERY LONG DAYS AND NIGHTS")

    def scene_water_cycle(self,image:Image.Image,t:float):
        shot=SHOT_PLAN[5]; local=smoothstep((t-shot["start"])/max(shot["end"]-shot["start"],1e-9))
        self.draw_ocean_moon(image,t,.48,.94,cloud_alpha=.75,atmosphere=1.0)
        self.draw_earth(image,t,.80,.19,.92)
        self.draw_water_cycle(image,t,local)
        self.panel(image,(int(OUT_W*.07),int(OUT_H*.64),int(OUT_W*.93),int(OUT_H*.85)),196)
        draw_text(image,"A LUNAR WATER CYCLE",(OUT_W//2,int(OUT_H*.688)),31 if not QUICK_MODE else 15,COLORS["cyan"]+(248,),True,"ma",1)
        draw_text(image,"EVAPORATION → CLOUDS → RAIN → RUNOFF",(OUT_W//2,int(OUT_H*.739)),18 if not QUICK_MODE else 9,COLORS["white"]+(240,),True,"ma",1)
        draw_text(image,"possible in the hypothetical atmosphere — but not an Earth copy",(OUT_W//2,int(OUT_H*.786)),14 if not QUICK_MODE else 7,COLORS["gold"]+(225,),False,"ma",1)
        draw_text(image,"a blue Moon would be an entirely different planet-like world",(OUT_W//2,int(OUT_H*.824)),15 if not QUICK_MODE else 7,COLORS["muted"]+(220,),False,"ma",1)
        self.draw_corner_label(image,"6 // BLUE MOON, DIFFERENT CLIMATE")

    def render_frame(self,t:float) -> np.ndarray:
        image=self.background(t)
        name=get_shot(t)["name"]
        if name=="intro":
            self.scene_intro(image,t)
        elif name=="atmosphere":
            self.scene_atmosphere(image,t)
        elif name=="gravity":
            self.scene_gravity(image,t)
        elif name=="tides":
            self.scene_tides(image,t)
        elif name=="long_day":
            self.scene_long_day(image,t)
        else:
            self.scene_water_cycle(image,t)

        self.draw_source_hud(image)
        self.draw_title(image,t)
        self.draw_caption(image,t)

        arr=np.asarray(image.convert("RGB"))
        arr=apply_grade(arr)
        arr=np.clip(arr.astype(np.float32)*VIGNETTE[...,None],0,255).astype(np.uint8)
        fade_in=smoothstep(t/(.85 if not QUICK_MODE else .20))
        fade_out=1.0-smoothstep((t-(DURATION-(1.10 if not QUICK_MODE else .25)))/(1.0 if not QUICK_MODE else .20))
        return np.clip(arr.astype(np.float32)*fade_in*fade_out,0,255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def save_summary() -> Path:
    summary = {
        "title": CONFIG["title"],
        "format": f"{OUT_W}x{OUT_H} vertical MP4",
        "fps": FPS,
        "duration_s": DURATION,
        "quick_mode": QUICK_MODE,
        "four_k": FOUR_K,
        "science_points": [
            "Earth-like surface oceans on the Moon require a hypothetical substantial atmosphere and suitable temperatures.",
            "Lunar surface gravity is about one-sixth of Earth's.",
            "Earth would be an important tide-raising body for hypothetical lunar oceans.",
            "A lunar solar day is about 29.5 Earth days, producing very long daylight/night cycles.",
            "With enough atmosphere, a water cycle involving evaporation, clouds and precipitation could be possible in principle.",
        ],
        "visual_warning": "The ocean coverage, coastlines, tidal bulges, clouds and climate behavior are cinematic and diagrammatic, not a physical simulation.",
    }
    path=OUTPUT_ROOT/"science_and_render_summary.json"
    path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return path


def render_video(scene:MoonOceanScene) -> Path:
    srt_path=OUTPUT_ROOT/f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS,srt_path)
    print("Subtitle sidecar:",srt_path.resolve())

    raw_video=OUTPUT_ROOT/f"{CONFIG['output_basename']}_raw.mp4"
    final_video=OUTPUT_ROOT/f"{CONFIG['output_basename']}_final.mp4"
    frame_count=int(round(DURATION*FPS))
    times=np.arange(frame_count)/FPS
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")

    with iio.get_writer(
        raw_video,
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times,desc="Rendering Moon-oceans short"):
            writer.append_data(scene.render_frame(float(t)))

    shutil.copyfile(raw_video,final_video)
    print("Final video:",final_video.resolve())
    return final_video

def main():
    print("Preparing Moon-oceans YouTube Short ...")
    print("Mode:","QUICK" if QUICK_MODE else ("4K" if FOUR_K else "FULL"))
    print("Canvas:",f"{OUT_W}x{OUT_H}")
    print("FPS:",FPS)
    print("Duration:",DURATION,"seconds")

    scene=MoonOceanScene()
    summary_path=save_summary()

    preview_times=[
        min(1.0,DURATION*.08),
        min(10.5,DURATION*.22),
        min(21.5,DURATION*.40),
        min(33.0,DURATION*.58),
        min(44.0,DURATION*.77),
        DURATION-(1.0 if not QUICK_MODE else .8),
    ]
    for preview_time in tqdm(preview_times,desc="Preview frames"):
        Image.fromarray(scene.render_frame(float(preview_time))).save(PREVIEW_DIR/f"preview_{int(preview_time*10):03d}.png")

    print("Summary:",summary_path.resolve())
    render_video(scene)
    print("Output directory:",OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-",path.name)

