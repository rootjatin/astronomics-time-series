from __future__ import annotations

"""
40 Years of Arctic Sea Ice — cinematic YouTube Short renderer

Creates a vertical 1080x1920 science short showing four decades of change in
Arctic sea ice, centered on the 1985 -> 2025 span. The polar maps are stylized,
not geospatial reconstructions; numerical callouts are based on NASA/NSIDC
satellite-record summaries cited in the source notes written beside the output.

Scientific framing used in the narration
-----------------------------------------
- Continuous satellite observations of Arctic sea ice extend back to 1979.
- NSIDC reports the long-term downward trend in annual minimum Arctic sea ice
  extent from 1979 through 2025 as about 12.1% per decade relative to the
  1981-2010 average.
- The satellite-era record minimum was 3.39 million km^2 on 17 September 2012.
- The 2025 minimum was 4.60 million km^2 on 10 September 2025, tied for the
  tenth-lowest minimum in the record at the time.
- The last 19 annual minimums, 2007-2025, were the 19 lowest minimum extents in
  the satellite record.
- The 2025 winter maximum was 14.33 million km^2, the lowest maximum in the
  47-year satellite record at the time.
- Sea ice varies strongly from year to year because winds, weather, ocean heat,
  and other conditions affect each melt season. The long-term trend is the key.

Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    ARCTIC_ICE_SHORT_QUICK=1 python 40_years_of_arctic_sea_ice.py

Full render
-----------
    python 40_years_of_arctic_sea_ice.py

4K vertical
-----------
    ARCTIC_ICE_SHORT_4K=1 python 40_years_of_arctic_sea_ice.py
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

QUICK_MODE = os.environ.get("ARCTIC_ICE_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("ARCTIC_ICE_SHORT_4K", "0") == "1" and not QUICK_MODE

OUT_W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
OUT_H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 8 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SCALE = OUT_W / 1080.0
OUT_SIZE = (OUT_W, OUT_H)

OUTPUT_ROOT = Path("40_years_of_arctic_sea_ice_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "40 YEARS OF ARCTIC SEA ICE",
    "subtitle": "1985 -> 2025 // satellites // summer minimum // long-term decline",
    "output_basename": "40_years_of_arctic_sea_ice",
    "contrast": 1.10,
    "saturation": 1.06,
    "vignette": 0.27,
}

COLORS = {
    "space": (2, 8, 18),
    "space2": (7, 18, 35),
    "ocean": (12, 63, 111),
    "ocean2": (22, 105, 161),
    "ice": (225, 244, 250),
    "ice_blue": (154, 222, 242),
    "ice_shadow": (84, 157, 188),
    "land": (93, 106, 94),
    "land2": (132, 124, 101),
    "white": (246, 250, 255),
    "muted": (171, 197, 214),
    "cyan": (76, 226, 255),
    "blue": (70, 137, 255),
    "gold": (255, 203, 91),
    "orange": (255, 137, 72),
    "red": (255, 80, 103),
    "violet": (177, 124, 255),
    "green": (105, 237, 170),
    "magenta": (238, 94, 194),
}

FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.4, 7.5, "Forty years apart, the Arctic at the end of summer can look like a different ocean. Satellite records show a much smaller ice cover today than in the 1980s."),
    (7.6, 17.2, "The change is not a smooth straight line. Weather moves the ice around every year — but the long-term direction is clear. Minimum extent has declined about 12 percent per decade since 1979."),
    (17.3, 27.3, "The most dramatic year was 2012. Arctic sea ice fell to a satellite-era record minimum of 3.39 million square kilometers."),
    (27.4, 38.1, "Then comes an important detail: a later year can have more ice than 2012 and still be part of the long-term decline. In 2025, the minimum was 4.60 million square kilometers."),
    (38.2, 49.0, "Sea ice also grows back every winter. But in 2025, even the winter maximum was the lowest in the 47-year satellite record — about 14.33 million square kilometers."),
    (49.1, 57.5, "Sea ice is more than a white cap. It changes how much sunlight the Arctic reflects and how heat and moisture move between the ocean and atmosphere. Four decades reveal a system being reshaped."),
]

if QUICK_MODE:
    factor = DURATION / 58.0
    CAPTIONS = [(a * factor, b * factor, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "forty_years", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "trend", "start": 8.0 if not QUICK_MODE else 1.8, "end": 18.0 if not QUICK_MODE else 4.0},
    {"name": "record_2012", "start": 18.0 if not QUICK_MODE else 4.0, "end": 28.0 if not QUICK_MODE else 6.25},
    {"name": "variability_2025", "start": 28.0 if not QUICK_MODE else 6.25, "end": 39.0 if not QUICK_MODE else 8.7},
    {"name": "seasonal_cycle", "start": 39.0 if not QUICK_MODE else 8.7, "end": 50.0 if not QUICK_MODE else 11.15},
    {"name": "why_it_matters", "start": 50.0 if not QUICK_MODE else 11.15, "end": DURATION},
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
class Spark:
    x: float
    y: float
    r: float
    a: int
    phase: float


class ArcticSeaIceScene:
    def __init__(self):
        rng = np.random.default_rng(20260818)
        self.sparks = [
            Spark(
                float(rng.uniform(0, OUT_W)),
                float(rng.uniform(0, OUT_H)),
                float(rng.uniform(0.4, 2.2) * SCALE),
                int(rng.uniform(10, 65)),
                float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(180 if QUICK_MODE else 520)
        ]
        self.ice_flecks = [
            (
                float(rng.uniform(-1, 1)),
                float(rng.uniform(-1, 1)),
                float(rng.uniform(2, 9) * SCALE),
                float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(70 if QUICK_MODE else 180)
        ]

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 176):
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle(
            box,
            radius=max(8, int(24 * SCALE)),
            fill=(2, 9, 19, alpha),
            outline=COLORS["cyan"] + (45,),
            width=max(1, int(2 * SCALE)),
        )
        image.alpha_composite(layer)

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        yy = np.linspace(0, 1, OUT_H)[:, None]
        arr[..., 0] = np.clip(3 + yy * 7, 0, 255)
        arr[..., 1] = np.clip(10 + yy * 21, 0, 255)
        arr[..., 2] = np.clip(24 + yy * 35, 0, 255)
        image = Image.fromarray(arr, "RGB").convert("RGBA")
        d = ImageDraw.Draw(image)
        for p in self.sparks:
            a = int(p.a * (0.72 + 0.28 * math.sin(t * 1.05 + p.phase)))
            d.ellipse((p.x-p.r, p.y-p.r, p.x+p.r, p.y+p.r), fill=COLORS["white"] + (a,))
        return image

    @staticmethod
    def polar_geometry(y_frac: float = .43, radius_scale: float = 1.0) -> Tuple[int, int, int]:
        return OUT_W // 2, int(OUT_H * y_frac), int(350 * SCALE * radius_scale)

    def ice_boundary(self, angle: float, t: float, scale: float, seed_phase: float = 0.0) -> float:
        # Stylized polar ice edge. It is intentionally not a real geospatial contour.
        wobble = (
            0.075 * math.sin(3 * angle + 0.18 * t + seed_phase)
            + 0.045 * math.sin(7 * angle - 0.10 * t + 1.4 + seed_phase)
            + 0.022 * math.sin(13 * angle + 0.25 * t + 0.2)
        )
        directional_loss = 0.055 * math.cos(angle - 0.6) - 0.035 * math.cos(2 * angle + 1.2)
        return clamp(scale + wobble + directional_loss, .18, .94)

    def draw_polar_map(
        self,
        image: Image.Image,
        t: float,
        ice_scale: float,
        y_frac: float = .43,
        radius_scale: float = 1.0,
        outline_scale: Optional[float] = None,
        alpha: int = 255,
    ):
        cx, cy, r = self.polar_geometry(y_frac, radius_scale)
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        # ocean disc + glow
        for extra, a in [(28, 20), (14, 38)]:
            rr = r + int(extra * SCALE)
            d.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=COLORS["cyan"] + (a,))
        d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=COLORS["ocean"] + (alpha,), outline=COLORS["white"] + (65,), width=max(1, int(2*SCALE)))

        # radial ocean texture
        for k in range(9):
            rr = int(r * (0.18 + k * 0.085))
            d.arc((cx-rr, cy-rr, cx+rr, cy+rr), 0, 360, fill=COLORS["ocean2"] + (24,), width=max(1, int(2*SCALE)))

        # stylized surrounding land masses as sectors/blobs
        lands = [
            (205, 278, .88, 1.07),  # North America / Greenland side
            (292, 350, .84, 1.08),  # Greenland / Atlantic
            (8, 75, .86, 1.10),     # Eurasia Atlantic sector
            (82, 158, .84, 1.10),   # Siberia
            (162, 196, .90, 1.06),  # Bering/Alaska
        ]
        for start, end, inner, outer in lands:
            pts = []
            for angd in np.linspace(start, end, 28):
                a = math.radians(float(angd))
                rr = r * outer * (1 + .025 * math.sin(a*5 + t*.04))
                pts.append((cx + math.cos(a)*rr, cy + math.sin(a)*rr))
            for angd in np.linspace(end, start, 28):
                a = math.radians(float(angd))
                rr = r * inner * (1 + .025 * math.cos(a*4 - t*.03))
                pts.append((cx + math.cos(a)*rr, cy + math.sin(a)*rr))
            d.polygon(pts, fill=COLORS["land"] + (235,))
            d.line(pts + [pts[0]], fill=COLORS["land2"] + (190,), width=max(1, int(3*SCALE)))

        # ice polygon
        ice_pts = []
        for ang in np.linspace(0, 2*math.pi, 220, endpoint=False):
            boundary = self.ice_boundary(float(ang), t, ice_scale)
            rr = r * boundary
            ice_pts.append((cx + math.cos(ang)*rr, cy + math.sin(ang)*rr))
        d.polygon(ice_pts, fill=COLORS["ice"] + (245,), outline=COLORS["ice_blue"] + (230,))

        # cracks / leads
        for i in range(18):
            a = i / 18 * 2 * math.pi + .2 * math.sin(t*.07+i)
            r0 = r * (.12 + .03*(i%3))
            r1 = r * (ice_scale * (.66 + .19*((i%5)/4)))
            x0 = cx + math.cos(a)*r0
            y0 = cy + math.sin(a)*r0
            x1 = cx + math.cos(a+.11*math.sin(i))*r1
            y1 = cy + math.sin(a+.11*math.sin(i))*r1
            d.line((x0,y0,x1,y1), fill=COLORS["ice_shadow"] + (70,), width=max(1,int(2*SCALE)))

        # floes near the edge
        for px, py, rr, phase in self.ice_flecks:
            rad = math.hypot(px, py)
            if rad < .58 or rad > .99:
                continue
            ang = math.atan2(py, px) + .04*math.sin(t*.18+phase)
            edge = self.ice_boundary(ang, t, ice_scale)
            if rad > edge + .03 and rad < edge + .17:
                x = cx + px*r*.88
                y = cy + py*r*.88
                q = rr * (.7 + .3*math.sin(t*.5+phase))
                d.ellipse((x-q,y-q*.55,x+q,y+q*.55), fill=COLORS["ice"] + (145,))

        if outline_scale is not None:
            outline = []
            for ang in np.linspace(0, 2*math.pi, 220, endpoint=False):
                rr = r * self.ice_boundary(float(ang), t, outline_scale, seed_phase=.6)
                outline.append((cx + math.cos(ang)*rr, cy + math.sin(ang)*rr))
            d.line(outline + [outline[0]], fill=COLORS["magenta"] + (220,), width=max(2, int(5*SCALE)))

        image.alpha_composite(layer)

        # subtle rim glow
        rim = Image.new("RGBA", OUT_SIZE, (0,0,0,0)); rd=ImageDraw.Draw(rim)
        rd.ellipse((cx-r,cy-r,cx+r,cy+r), outline=COLORS["cyan"]+(70,), width=max(2,int(5*SCALE)))
        image.alpha_composite(rim.filter(ImageFilter.GaussianBlur(max(2,int(7*SCALE)))))

    def draw_trend_chart(self, image: Image.Image, local: float):
        x0=int(OUT_W*.11); x1=int(OUT_W*.89); y0=int(OUT_H*.26); y1=int(OUT_H*.66)
        d=ImageDraw.Draw(image)
        d.line((x0,y1,x1,y1),fill=COLORS["white"]+(105,),width=max(2,int(3*SCALE)))
        d.line((x0,y0,x0,y1),fill=COLORS["white"]+(105,),width=max(2,int(3*SCALE)))

        for frac,label in [(0.0,"1985"),(.375,"2000"),(.675,"2012"),(1.0,"2025")]:
            x=int(lerp(x0,x1,frac))
            d.line((x,y1-int(7*SCALE),x,y1+int(7*SCALE)),fill=COLORS["muted"]+(140,),width=max(1,int(2*SCALE)))
            draw_text(image,label,(x,y1+int(25*SCALE)),13 if not QUICK_MODE else 6,COLORS["muted"]+(210,),False,"ma",1)

        # Schematic variability around the long-term trend. Not annual data.
        years=np.arange(1985,2026)
        values=[]
        for i,year in enumerate(years):
            trend=1.0 - .0102*(year-1985)
            noise=.055*math.sin(i*.78)+.030*math.sin(i*1.91+1.2)
            if year==2012:
                noise-=.115
            values.append(clamp(trend+noise,.43,1.03))
        reveal=max(2,int(local*(len(values)-1))+1)
        pts=[]
        for i,val in enumerate(values[:reveal]):
            x=lerp(x0,x1,i/(len(values)-1))
            y=lerp(y1,y0,clamp((val-.38)/.70))
            pts.append((x,y))
        if len(pts)>1:
            d.line(pts,fill=COLORS["cyan"]+(235,),width=max(2,int(6*SCALE)),joint="curve")
            x,y=pts[-1]
            d.ellipse((x-6*SCALE,y-6*SCALE,x+6*SCALE,y+6*SCALE),fill=COLORS["gold"]+(245,))

        # long-term direction arrow
        ax0=int(OUT_W*.19); ay0=int(OUT_H*.35); ax1=int(OUT_W*.80); ay1=int(OUT_H*.57)
        d.line((ax0,ay0,ax1,ay1),fill=COLORS["red"]+(120,),width=max(2,int(4*SCALE)))
        d.polygon([(ax1,ay1),(ax1-int(18*SCALE),ay1-int(18*SCALE)),(ax1-int(22*SCALE),ay1+int(9*SCALE))],fill=COLORS["red"]+(170,))
        draw_text(image,"SCHEMATIC YEAR-TO-YEAR VARIABILITY",(OUT_W//2,int(OUT_H*.205)),14 if not QUICK_MODE else 7,COLORS["muted"]+(215,),True,"ma",1)

    def draw_season_cycle(self, image: Image.Image, t: float, local: float):
        cx=int(OUT_W*.50); cy=int(OUT_H*.44); r=int(245*SCALE)
        d=ImageDraw.Draw(image)
        d.ellipse((cx-r,cy-r,cx+r,cy+r),outline=COLORS["white"]+(80,),width=max(2,int(4*SCALE)))
        # winter half and summer half
        d.arc((cx-r,cy-r,cx+r,cy+r),180,360,fill=COLORS["cyan"]+(225,),width=max(4,int(18*SCALE)))
        d.arc((cx-r,cy-r,cx+r,cy+r),0,180,fill=COLORS["gold"]+(220,),width=max(4,int(18*SCALE)))
        # moving hand through a year
        ang=math.radians(-90+360*local)
        x=cx+math.cos(ang)*r*.72; y=cy+math.sin(ang)*r*.72
        d.line((cx,cy,x,y),fill=COLORS["white"]+(235,),width=max(2,int(5*SCALE)))
        d.ellipse((cx-8*SCALE,cy-8*SCALE,cx+8*SCALE,cy+8*SCALE),fill=COLORS["white"]+(235,))
        draw_text(image,"WINTER GROWTH",(cx,cy-int(r*.53)),17 if not QUICK_MODE else 8,COLORS["cyan"]+(235,),True,"ma",1)
        draw_text(image,"SUMMER MELT",(cx,cy+int(r*.56)),17 if not QUICK_MODE else 8,COLORS["gold"]+(235,),True,"ma",1)

    def draw_albedo_demo(self, image: Image.Image, t: float, local: float):
        # Split scene: bright ice reflects, dark ocean absorbs.
        d=ImageDraw.Draw(image)
        y=int(OUT_H*.53)
        d.rounded_rectangle((int(OUT_W*.08),int(OUT_H*.29),int(OUT_W*.46),int(OUT_H*.67)),radius=int(26*SCALE),fill=COLORS["ice"]+(230,))
        d.rounded_rectangle((int(OUT_W*.54),int(OUT_H*.29),int(OUT_W*.92),int(OUT_H*.67)),radius=int(26*SCALE),fill=COLORS["ocean"]+(245,))
        # sun rays
        for i in range(6):
            x=int(lerp(OUT_W*.15,OUT_W*.85,i/5))
            d.line((x,int(OUT_H*.20),x-int(30*SCALE),int(OUT_H*.31)),fill=COLORS["gold"]+(185,),width=max(2,int(5*SCALE)))
        # reflected arrows on ice
        for i in range(3):
            x=int(OUT_W*(.16+.11*i))
            d.line((x,y,x-int(28*SCALE),int(OUT_H*.39)),fill=COLORS["cyan"]+(210,),width=max(2,int(5*SCALE)))
            d.polygon([(x-int(28*SCALE),int(OUT_H*.39)),(x-int(16*SCALE),int(OUT_H*.405)),(x-int(38*SCALE),int(OUT_H*.414))],fill=COLORS["cyan"]+(220,))
        # absorbed glow in ocean
        glow=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); gd=ImageDraw.Draw(glow)
        for rr,a in [(40,80),(75,45),(115,22)]:
            q=int(rr*SCALE*(.8+.2*local)); cx=int(OUT_W*.73); cy=int(OUT_H*.51)
            gd.ellipse((cx-q,cy-q,cx+q,cy+q),fill=COLORS["orange"]+(a,))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4,int(18*SCALE)))))
        draw_text(image,"BRIGHT ICE",(int(OUT_W*.27),int(OUT_H*.70)),17 if not QUICK_MODE else 8,COLORS["cyan"]+(240,),True,"ma",1)
        draw_text(image,"DARK OCEAN",(int(OUT_W*.73),int(OUT_H*.70)),17 if not QUICK_MODE else 8,COLORS["gold"]+(240,),True,"ma",1)

    def draw_title(self, image: Image.Image, t: float):
        if t >= (5.8 if not QUICK_MODE else 1.30):
            return
        fade=smoothstep(t/(.8 if not QUICK_MODE else .18))
        draw_text(image,"40 YEARS OF",(OUT_W//2,int(OUT_H*.061)),38 if not QUICK_MODE else 19,COLORS["white"]+(int(245*fade),),True,"ma",2)
        draw_text(image,"ARCTIC SEA ICE",(OUT_W//2,int(OUT_H*.111)),52 if not QUICK_MODE else 26,COLORS["cyan"]+(int(250*fade),),True,"ma",2)
        draw_text(image,"1985 -> 2025",(OUT_W//2,int(OUT_H*.157)),31 if not QUICK_MODE else 15,COLORS["gold"]+(int(245*fade),),True,"ma",2)

    def draw_caption(self, image: Image.Image, t: float):
        cap=caption_at(t)
        if not cap:
            return
        y0=OUT_H-(260 if not QUICK_MODE else 130)
        self.panel(image,(44 if not QUICK_MODE else 22,y0,OUT_W-(44 if not QUICK_MODE else 22),y0+(140 if not QUICK_MODE else 71)),182)
        draw_wrapped_text(image,cap,(68 if not QUICK_MODE else 34,y0+(27 if not QUICK_MODE else 14)),OUT_W-(136 if not QUICK_MODE else 68),27 if not QUICK_MODE else 13)

    def draw_corner_label(self, image: Image.Image, textv: str):
        draw_text(image,textv,(52 if not QUICK_MODE else 26,58 if not QUICK_MODE else 29),18 if not QUICK_MODE else 9,COLORS["muted"]+(205,),True,"la",1)

    def draw_source_hud(self, image: Image.Image):
        draw_text(image,"SATELLITE-ERA SCIENCE VISUALIZATION",(OUT_W-(48 if not QUICK_MODE else 24),72 if not QUICK_MODE else 36),14 if not QUICK_MODE else 7,COLORS["gold"]+(225,),True,"ra",1)
        draw_text(image,"POLAR MAPS ARE DIAGRAMMATIC • NSIDC/NASA NUMBERS",(OUT_W-(48 if not QUICK_MODE else 24),100 if not QUICK_MODE else 50),12 if not QUICK_MODE else 6,COLORS["muted"]+(190,),False,"ra",1)

    def scene_forty_years(self, image: Image.Image, t: float):
        s=SHOT_PLAN[0]; local=smoothstep((t-s["start"])/max(s["end"]-s["start"],1e-9))
        scale=lerp(.80,.59,local)
        self.draw_polar_map(image,t,scale,y_frac=.43,radius_scale=.94,outline_scale=.80)
        self.panel(image,(int(OUT_W*.08),int(OUT_H*.69),int(OUT_W*.92),int(OUT_H*.83)),174)
        year=int(round(lerp(1985,2025,local)))
        draw_text(image,f"{year}",(OUT_W//2,int(OUT_H*.733)),41 if not QUICK_MODE else 20,COLORS["gold"]+(248,),True,"ma",1)
        draw_text(image,"LATE-SUMMER ARCTIC SEA ICE",(OUT_W//2,int(OUT_H*.783)),18 if not QUICK_MODE else 9,COLORS["white"]+(230,),True,"ma",1)
        draw_text(image,"magenta outline = stylized 1985 reference edge",(OUT_W//2,int(OUT_H*.817)),13 if not QUICK_MODE else 6,COLORS["magenta"]+(220,),False,"ma",1)
        self.draw_corner_label(image,"1 // FOUR DECADES IN ONE VIEW")

    def scene_trend(self, image: Image.Image, t: float):
        s=SHOT_PLAN[1]; local=smoothstep((t-s["start"])/max(s["end"]-s["start"],1e-9))
        self.panel(image,(int(OUT_W*.07),int(OUT_H*.16),int(OUT_W*.93),int(OUT_H*.72)),160)
        self.draw_trend_chart(image,local)
        self.panel(image,(int(OUT_W*.11),int(OUT_H*.73),int(OUT_W*.89),int(OUT_H*.84)),184)
        draw_text(image,"LONG-TERM MINIMUM TREND",(OUT_W//2,int(OUT_H*.762)),18 if not QUICK_MODE else 9,COLORS["muted"]+(230,),True,"ma",1)
        draw_text(image,"−12.1% PER DECADE",(OUT_W//2,int(OUT_H*.808)),34 if not QUICK_MODE else 17,COLORS["red"]+(248,),True,"ma",1)
        self.draw_corner_label(image,"2 // THE TREND, NOT EVERY SINGLE YEAR")

    def scene_record_2012(self, image: Image.Image, t: float):
        s=SHOT_PLAN[2]; local=smoothstep((t-s["start"])/max(s["end"]-s["start"],1e-9))
        pulse=.5+.5*math.sin(t*3.0)
        self.draw_polar_map(image,t,.47,y_frac=.43,radius_scale=.94,outline_scale=.80)
        cx,cy,r=self.polar_geometry(.43,.94)
        layer=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        for k in range(4):
            rr=r*(.48+.04*k)+pulse*15*SCALE
            d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=COLORS["red"]+(max(20,150-30*k),),width=max(1,int((4-k*.5)*SCALE)))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1,int(3*SCALE)))))
        self.panel(image,(int(OUT_W*.12),int(OUT_H*.69),int(OUT_W*.88),int(OUT_H*.84)),190)
        draw_text(image,"2012 • SATELLITE-ERA RECORD LOW",(OUT_W//2,int(OUT_H*.724)),18 if not QUICK_MODE else 9,COLORS["muted"]+(230,),True,"ma",1)
        draw_text(image,"3.39 MILLION km²",(OUT_W//2,int(OUT_H*.775)),37 if not QUICK_MODE else 18,COLORS["red"]+(250,),True,"ma",1)
        draw_text(image,"annual minimum extent • 17 September 2012",(OUT_W//2,int(OUT_H*.817)),14 if not QUICK_MODE else 7,COLORS["gold"]+(225,),False,"ma",1)
        self.draw_corner_label(image,"3 // THE 2012 EXTREME")

    def scene_variability_2025(self, image: Image.Image, t: float):
        s=SHOT_PLAN[3]; local=smoothstep((t-s["start"])/max(s["end"]-s["start"],1e-9))
        self.draw_polar_map(image,t,.59,y_frac=.43,radius_scale=.94,outline_scale=.80)
        # comparison badge to 2012
        self.panel(image,(int(OUT_W*.09),int(OUT_H*.65),int(OUT_W*.91),int(OUT_H*.85)),190)
        draw_text(image,"2025 MINIMUM",(int(OUT_W*.30),int(OUT_H*.706)),18 if not QUICK_MODE else 9,COLORS["muted"]+(225,),True,"ma",1)
        draw_text(image,"4.60M km²",(int(OUT_W*.30),int(OUT_H*.760)),31 if not QUICK_MODE else 15,COLORS["cyan"]+(248,),True,"ma",1)
        draw_text(image,"2012 RECORD",(int(OUT_W*.70),int(OUT_H*.706)),18 if not QUICK_MODE else 9,COLORS["muted"]+(225,),True,"ma",1)
        draw_text(image,"3.39M km²",(int(OUT_W*.70),int(OUT_H*.760)),31 if not QUICK_MODE else 15,COLORS["red"]+(248,),True,"ma",1)
        d=ImageDraw.Draw(image)
        d.line((OUT_W//2,int(OUT_H*.69),OUT_W//2,int(OUT_H*.79)),fill=COLORS["white"]+(60,),width=max(1,int(2*SCALE)))
        draw_text(image,"2025 WAS NOT THE RECORD LOW — BUT STILL TIED FOR 10th LOWEST",(OUT_W//2,int(OUT_H*.818)),13 if not QUICK_MODE else 6,COLORS["gold"]+(225,),True,"ma",1)
        self.draw_corner_label(image,"4 // VARIABILITY DOES NOT ERASE THE TREND")

    def scene_seasonal_cycle(self, image: Image.Image, t: float):
        s=SHOT_PLAN[4]; local=smoothstep((t-s["start"])/max(s["end"]-s["start"],1e-9))
        self.draw_season_cycle(image,t,local)
        self.panel(image,(int(OUT_W*.08),int(OUT_H*.69),int(OUT_W*.92),int(OUT_H*.85)),188)
        draw_text(image,"2025 WINTER MAXIMUM",(OUT_W//2,int(OUT_H*.721)),18 if not QUICK_MODE else 9,COLORS["muted"]+(230,),True,"ma",1)
        draw_text(image,"14.33 MILLION km²",(OUT_W//2,int(OUT_H*.770)),34 if not QUICK_MODE else 17,COLORS["cyan"]+(248,),True,"ma",1)
        draw_text(image,"lowest annual maximum in the 47-year satellite record",(OUT_W//2,int(OUT_H*.817)),14 if not QUICK_MODE else 7,COLORS["gold"]+(225,),False,"ma",1)
        self.draw_corner_label(image,"5 // THE ICE GROWS BACK — BUT WINTER IS CHANGING TOO")

    def scene_why_it_matters(self, image: Image.Image, t: float):
        s=SHOT_PLAN[5]; local=smoothstep((t-s["start"])/max(s["end"]-s["start"],1e-9))
        self.draw_albedo_demo(image,t,local)
        self.panel(image,(int(OUT_W*.08),int(OUT_H*.74),int(OUT_W*.92),int(OUT_H*.87)),192)
        draw_text(image,"SEA ICE CHANGES THE ARCTIC ENERGY BALANCE",(OUT_W//2,int(OUT_H*.782)),21 if not QUICK_MODE else 10,COLORS["white"]+(245,),True,"ma",1)
        draw_text(image,"bright ice reflects more sunlight • dark ocean absorbs more",(OUT_W//2,int(OUT_H*.824)),15 if not QUICK_MODE else 7,COLORS["gold"]+(230,),False,"ma",1)
        draw_text(image,"and ice controls ocean-atmosphere heat + moisture exchange",(OUT_W//2,int(OUT_H*.855)),14 if not QUICK_MODE else 7,COLORS["cyan"]+(220,),False,"ma",1)
        self.draw_corner_label(image,"6 // WHY THE WHITE CAP MATTERS")

    def render_frame(self, t: float) -> np.ndarray:
        image=self.background(t)
        name=get_shot(t)["name"]
        if name=="forty_years": self.scene_forty_years(image,t)
        elif name=="trend": self.scene_trend(image,t)
        elif name=="record_2012": self.scene_record_2012(image,t)
        elif name=="variability_2025": self.scene_variability_2025(image,t)
        elif name=="seasonal_cycle": self.scene_seasonal_cycle(image,t)
        else: self.scene_why_it_matters(image,t)

        self.draw_source_hud(image)
        self.draw_title(image,t)
        self.draw_caption(image,t)

        arr=np.asarray(image.convert("RGB"))
        arr=apply_grade(arr)
        arr=np.clip(arr.astype(np.float32)*VIGNETTE[...,None],0,255).astype(np.uint8)
        fade_in=smoothstep(t/(.9 if not QUICK_MODE else .20))
        fade_out=1.0-smoothstep((t-(DURATION-(1.1 if not QUICK_MODE else .25)))/(1.0 if not QUICK_MODE else .20))
        return np.clip(arr.astype(np.float32)*fade_in*fade_out,0,255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def save_summary() -> Path:
    obj = {
        "title": CONFIG["title"],
        "format": f"{OUT_W}x{OUT_H} vertical MP4",
        "fps": FPS,
        "duration_s": DURATION,
        "quick_mode": QUICK_MODE,
        "four_k": FOUR_K,
        "visual_note": "Polar ice-edge maps and trend squiggle are cinematic/diagrammatic, not geospatial reconstructions or annual-data plots.",
        "facts": [
            "NSIDC: long-term annual minimum extent trend, 1979-2025: -12.1% per decade relative to 1981-2010 average.",
            "NSIDC: 2012 record minimum extent: 3.39 million km^2 on 17 September 2012.",
            "NSIDC: 2025 minimum extent: 4.60 million km^2 on 10 September 2025, tied for tenth lowest.",
            "NSIDC: the 19 annual minimums from 2007-2025 were the 19 lowest in the satellite record.",
            "NSIDC: 2025 annual maximum extent: 14.33 million km^2, lowest maximum in the 47-year satellite record.",
        ],
        "sources": [
            "https://nsidc.org/sea-ice-today/analyses/2025-arctic-sea-ice-minimum-squeezes-ten-lowest-minimums",
            "https://nsidc.org/sea-ice-today/analyses/arctic-sea-ice-sets-record-low-maximum-2025",
            "https://science.nasa.gov/earth/explore/earth-indicators/arctic-sea-ice-minimum-extent/",
            "https://nsidc.org/learn/parts-cryosphere/sea-ice/why-sea-ice-matters",
        ],
    }
    path=OUTPUT_ROOT/f"{CONFIG['output_basename']}_summary.json"
    path.write_text(json.dumps(obj,indent=2),encoding="utf-8")
    return path


def render_video(scene: ArcticSeaIceScene) -> Path:
    basename=CONFIG["output_basename"] + ("_quick" if QUICK_MODE else ("_4k" if FOUR_K else ""))
    output=OUTPUT_ROOT/f"{basename}.mp4"
    total_frames=max(1,int(round(DURATION*FPS)))

    writer=iio.get_writer(
        output,
        fps=FPS,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=None,
    )
    try:
        for frame_index in tqdm(range(total_frames),desc="Rendering Arctic sea ice short"):
            t=frame_index/FPS
            writer.append_data(scene.render_frame(t))
    finally:
        writer.close()
    return output

def render_previews(scene: ArcticSeaIceScene) -> List[Path]:
    times=[
        3.2 if not QUICK_MODE else .72,
        12.5 if not QUICK_MODE else 2.80,
        22.5 if not QUICK_MODE else 5.05,
        33.5 if not QUICK_MODE else 7.50,
        44.2 if not QUICK_MODE else 9.90,
        53.8 if not QUICK_MODE else 12.05,
    ]
    paths=[]
    for i,t in enumerate(times,1):
        path=PREVIEW_DIR/f"preview_{i:02d}_{t:.2f}s.png"
        Image.fromarray(scene.render_frame(t)).save(path)
        paths.append(path)
    return paths


def main():
    scene=ArcticSeaIceScene()
    preview_paths=render_previews(scene)
    srt_path=write_srt(CAPTIONS,OUTPUT_ROOT/f"{CONFIG['output_basename']}.srt")
    summary_path=save_summary()
    output_path=render_video(scene)

    print("\nRender complete")
    print(f"Video:   {output_path.resolve()}")
    print(f"SRT:     {srt_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print("Previews:")
    for path in preview_paths:
        print(f"  - {path.resolve()}")

    ffmpeg=shutil.which("ffmpeg")
    if ffmpeg:
        print(f"ffmpeg:  {ffmpeg}")


if __name__ == "__main__":
    main()

