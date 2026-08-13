from __future__ import annotations

"""
August 12, 2026 Total Solar Eclipse — Cinematic YouTube Shorts Renderer
=======================================================================

Creates a vertical 1080x1920 science Short explaining what happened during
Europe's August 12, 2026 total solar eclipse.

The renderer is intentionally split into two layers:


Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    ECLIPSE_SHORT_QUICK=1 python august_12_2026_total_solar_eclipse_cinematic_short.py

Full 1080x1920 render
---------------------
    python august_12_2026_total_solar_eclipse_cinematic_short.py

Outputs
-------
- MP4 video
- SRT subtitles / voiceover script
- PNG preview frames
- CSV of NASA central-line path points
- JSON fact sheet / source notes

Important safety note shown in the film:
Except during totality, direct solar viewing requires proper eclipse eye
protection. Cameras/binoculars/telescopes require a solar filter on the front.
"""

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# =============================================================================
# Configuration
# =============================================================================

QUICK_MODE = os.environ.get("ECLIPSE_SHORT_QUICK", "0") == "1"

OUTPUT_ROOT = Path("august_12_2026_eclipse_short_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
DATA_DIR = OUTPUT_ROOT / "data"
for directory in (OUTPUT_ROOT, PREVIEW_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0
DURATION = float(CONFIG["duration_s"])
FPS = int(CONFIG["fps"])

COLORS = {
    "space": (2, 5, 13),
    "white": (245, 249, 253),
    "muted": (155, 183, 205),
    "cyan": (99, 220, 255),
    "blue": (93, 151, 255),
    "violet": (183, 119, 255),
    "gold": (255, 197, 94),
    "orange": (255, 132, 62),
    "red": (255, 83, 95),
    "green": (105, 233, 169),
    "earth_ocean": (16, 61, 101),
    "earth_land": (76, 125, 91),
}


# =============================================================================
# NASA / GSFC eclipse data embedded for deterministic rendering
# =============================================================================

ECLIPSE_FACTS: Dict[str, Any] = {
    "date": "2026-08-12",
    "type": "Total Solar Eclipse",
    "greatest_eclipse_utc": "17:45:53.8",
    "greatest_lat_deg": 65.225,
    "greatest_lon_deg": -25.228333,
    "greatest_sun_altitude_deg": 25.8,
    "greatest_sun_azimuth_deg": 248.4,
    "greatest_path_width_km": 294.0,
    "greatest_central_duration_s": 138.2,
    "eclipse_magnitude_rounded": 1.039,
    "eclipse_magnitude_search_engine": 1.0386,
    "saros_series": 126,
    "prediction_note": "NASA GSFC path table: VSOP87/ELP2000-85, Delta T=71.4 s.",
}

# NASA central line. Columns: UTC, lat_deg, lon_deg, path_width_km, duration_s.
# Converted from degree+arcminute values in NASA's 120-second path table.
NASA_PATH: List[Tuple[str, float, float, float, float]] = [
    ("17:02", 82 + 16.5/60, 112 + 29.2/60, 273, 105.8),
    ("17:04", 85 + 17.7/60, 104 + 12.9/60, 274, 110.8),
    ("17:06", 87 + 16.7/60, 81 + 31.5/60, 274, 114.6),
    ("17:08", 87 + 49.4/60, 33 + 0.0/60, 275, 117.7),
    ("17:10", 86 + 50.1/60, -(1 + 38.3/60), 275, 120.4),
    ("17:12", 85 + 24.2/60, -(15 + 10.9/60), 275, 122.8),
    ("17:14", 83 + 55.9/60, -(21 + 11.2/60), 276, 124.9),
    ("17:16", 82 + 29.7/60, -(24 + 16.3/60), 276, 126.8),
    ("17:18", 81 + 6.6/60, -(25 + 59.5/60), 277, 128.5),
    ("17:20", 79 + 46.4/60, -(26 + 58.9/60), 278, 130.0),
    ("17:22", 78 + 29.0/60, -(27 + 32.4/60), 278, 131.4),
    ("17:24", 77 + 14.0/60, -(27 + 49.5/60), 279, 132.6),
    ("17:26", 76 + 1.1/60, -(27 + 55.7/60), 280, 133.7),
    ("17:28", 74 + 50.2/60, -(27 + 54.3/60), 281, 134.6),
    ("17:30", 73 + 41.0/60, -(27 + 47.3/60), 282, 135.4),
    ("17:32", 72 + 33.4/60, -(27 + 36.2/60), 283, 136.2),
    ("17:34", 71 + 27.0/60, -(27 + 21.7/60), 285, 136.8),
    ("17:36", 70 + 21.9/60, -(27 + 4.7/60), 286, 137.3),
    ("17:38", 69 + 17.9/60, -(26 + 45.6/60), 288, 137.7),
    ("17:40", 68 + 14.8/60, -(26 + 24.6/60), 289, 137.9),
    ("17:42", 67 + 12.6/60, -(26 + 1.9/60), 291, 138.1),
    ("17:44", 66 + 11.1/60, -(25 + 37.8/60), 292, 138.2),
    ("17:46", 65 + 10.3/60, -(25 + 12.3/60), 294, 138.2),
    ("17:48", 64 + 10.1/60, -(24 + 45.4/60), 296, 138.1),
    ("17:50", 63 + 10.3/60, -(24 + 17.2/60), 298, 137.9),
    ("17:52", 62 + 11.0/60, -(23 + 47.6/60), 300, 137.6),
    ("17:54", 61 + 12.0/60, -(23 + 16.6/60), 302, 137.1),
    ("17:56", 60 + 13.3/60, -(22 + 44.2/60), 304, 136.6),
    ("17:58", 59 + 14.7/60, -(22 + 10.2/60), 305, 136.0),
    ("18:00", 58 + 16.3/60, -(21 + 34.4/60), 307, 135.3),
    ("18:02", 57 + 17.8/60, -(20 + 56.8/60), 309, 134.5),
    ("18:04", 56 + 19.3/60, -(20 + 17.2/60), 311, 133.5),
    ("18:06", 55 + 20.6/60, -(19 + 35.3/60), 313, 132.5),
    ("18:08", 54 + 21.7/60, -(18 + 50.8/60), 315, 131.3),
    ("18:10", 53 + 22.3/60, -(18 + 3.4/60), 316, 130.0),
    ("18:12", 52 + 22.3/60, -(17 + 12.7/60), 318, 128.6),
    ("18:14", 51 + 21.6/60, -(16 + 18.2/60), 319, 127.0),
    ("18:16", 50 + 20.0/60, -(15 + 19.0/60), 319, 125.2),
    ("18:18", 49 + 17.1/60, -(14 + 14.3/60), 319, 123.3),
    ("18:20", 48 + 12.7/60, -(13 + 2.9/60), 319, 121.2),
    ("18:22", 47 + 6.1/60, -(11 + 42.9/60), 318, 118.8),
    ("18:24", 45 + 56.6/60, -(10 + 11.4/60), 315, 116.1),
    ("18:26", 44 + 42.8/60, -(8 + 23.9/60), 311, 113.0),
    ("18:28", 43 + 22.3/60, -(6 + 11.3/60), 304, 109.3),
    ("18:30", 41 + 49.0/60, -(3 + 11.1/60), 294, 104.6),
    ("18:32", 39 + 24.5/60, 2 + 57.0/60, 270, 95.8),
]

CITY_DATA = [
    # city, country, kind, partial begin, maximum/totality, partial end, coverage
    ("REYKJAVIK", "ICELAND", "TOTAL", "16:47", "17:48–17:49", "18:47", "TOTAL"),
    ("LEON", "SPAIN", "TOTAL", "19:32", "20:28–20:30", "21:22", "TOTAL"),
    ("ZARAGOZA", "SPAIN", "TOTAL", "19:34", "20:29–20:30", "21:07*", "TOTAL"),
    ("VALENCIA", "SPAIN", "TOTAL", "19:38", "20:32–20:33", "21:01*", "TOTAL"),
    ("MADRID", "SPAIN", "PARTIAL", "19:36", "20:32", "21:16*", "99%"),
    ("BARCELONA", "SPAIN", "PARTIAL", "19:35", "20:29", "20:54*", "99%"),
    ("DUBLIN", "IRELAND", "PARTIAL", "18:12", "19:10", "20:05", "94%"),
    ("PARIS", "FRANCE", "PARTIAL", "19:22", "20:17", "21:09", "92%"),
    ("LONDON", "U.K.", "PARTIAL", "18:17", "19:13", "20:06", "91%"),
    ("BERLIN", "GERMANY", "PARTIAL", "19:15", "20:08", "20:38*", "85%"),
]

# Geographical marker locations used only for approximate placement on our stylized globe.
CITY_COORDS = {
    "REYKJAVIK": (64.1466, -21.9426),
    "LEON": (42.5987, -5.5671),
    "ZARAGOZA": (41.6488, -0.8891),
    "VALENCIA": (39.4699, -0.3763),
    "MADRID": (40.4168, -3.7038),
    "BARCELONA": (41.3874, 2.1686),
    "DUBLIN": (53.3498, -6.2603),
    "PARIS": (48.8566, 2.3522),
    "LONDON": (51.5072, -0.1276),
    "BERLIN": (52.5200, 13.4050),
}

NASA_SOURCE_NOTES = [
    "Path / greatest eclipse: NASA GSFC eclipse path table by Fred Espenak.",
    "Selected city circumstances: NASA Science Aug. 12, 2026 eclipse page.",
    "Viewing safety: NASA Science eclipse eye-safety guidance.",
    "Earth geography in this renderer is deliberately simplified/stylized; eclipse path points are NASA-derived.",
]


# =============================================================================
# Story / narration
# =============================================================================

FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.4, 6.2, "On August 12, 2026, the Moon's shadow raced across the Arctic and North Atlantic — and parts of Europe experienced total darkness in daylight."),
    (6.3, 14.4, "A total solar eclipse happens when the Moon passes almost exactly between Earth and the Sun. The tiny inner shadow is the umbra: inside it, the Sun is completely covered."),
    (14.5, 24.2, "NASA's calculated center line began near far northern Siberia, curved past the pole, crossed Greenland and western Iceland, then swept across the North Atlantic into northern Spain."),
    (24.3, 33.0, "Near greatest eclipse at 17:45:54 UTC, the center of the shadow was near 65.2 degrees north, 25.2 degrees west. The totality path was about 294 kilometers wide and totality lasted about two minutes eighteen seconds."),
    (33.1, 42.6, "Europe was not equally dark. Reykjavik, Leon, Zaragoza and Valencia reached totality, while Madrid and Barcelona were extremely deep partial eclipses at about ninety-nine percent coverage."),
    (42.7, 49.6, "In Spain the eclipse arrived late in the evening, so the black disk of the Moon and the solar corona appeared low in the western sky, close to sunset."),
    (49.7, 55.5, "And one safety rule matters: except during the brief total phase, never look directly at the Sun without proper eclipse eye protection. Cameras and telescopes need front-mounted solar filters."),
]

SHOT_PLAN_FULL = [
    {"name": "hook", "start": 0.0, "end": 6.6},
    {"name": "alignment", "start": 6.6, "end": 14.8},
    {"name": "path", "start": 14.8, "end": 24.5},
    {"name": "greatest", "start": 24.5, "end": 33.3},
    {"name": "cities", "start": 33.3, "end": 42.9},
    {"name": "spain", "start": 42.9, "end": 49.9},
    {"name": "safety", "start": 49.9, "end": 56.0},
]

if QUICK_MODE:
    time_scale = DURATION / 56.0
    CAPTIONS = [(a*time_scale, b*time_scale, text) for a, b, text in FULL_CAPTIONS]
    SHOT_PLAN = [
        {"name": s["name"], "start": s["start"]*time_scale, "end": s["end"]*time_scale}
        for s in SHOT_PLAN_FULL
    ]
else:
    CAPTIONS = FULL_CAPTIONS
    SHOT_PLAN = SHOT_PLAN_FULL


# =============================================================================
# Utility helpers
# =============================================================================

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(value: float) -> float:
    x = clamp(value)
    return x*x*(3.0 - 2.0*x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b-a)*t


def ease_out_cubic(x: float) -> float:
    x = clamp(x)
    return 1.0 - (1.0-x)**3


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
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(7, int(size)))
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int = 28,
    fill=(255,255,255,255),
    bold: bool = False,
    stroke: int = 2,
    anchor: str = "la",
):
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold),
        fill=fill,
        anchor=anchor,
        stroke_width=max(0, int(stroke)),
        stroke_fill=(0, 0, 0, min(235, fill[3] if len(fill) > 3 else 235)),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int,int],
    max_width: int,
    size: int,
    fill=(255,255,255,245),
    bold: bool=False,
    line_spacing: int=6,
    anchor: str="la",
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        box = draw.textbbox((0,0), candidate, font=font, stroke_width=2)
        if box[2]-box[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    x, y = xy
    for line in lines:
        draw.text((x,y), line, font=font, fill=fill, anchor=anchor,
                  stroke_width=2, stroke_fill=(0,0,0,225))
        box = draw.textbbox((x,y), line, font=font, anchor=anchor, stroke_width=2)
        y += (box[3]-box[1]) + line_spacing


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds*1000.0))
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    ms %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float,float,str]], path: Path) -> Path:
    lines: List[str] = []
    for i, (start,end,text) in enumerate(captions, start=1):
        lines += [str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx-width/2.0)/(width/2.0)
    ny = (yy-height/2.0)/(height/2.0)
    r = np.sqrt(nx*nx + ny*ny)
    return np.clip(1.0-strength*r**1.8, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


def apply_grade(rgb: np.ndarray) -> np.ndarray:
    im = Image.fromarray(rgb)
    im = ImageEnhance.Contrast(im).enhance(float(CONFIG["contrast"]))
    im = ImageEnhance.Color(im).enhance(float(CONFIG["saturation"]))
    arr = np.asarray(im).astype(np.float32)
    arr *= VIGNETTE[:,:,None]
    return np.clip(arr,0,255).astype(np.uint8)


def soft_line(image: Image.Image, points: Sequence[Tuple[float,float]], fill, width=2.0, glow=6.0):
    if len(points) < 2:
        return
    gl = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
    gd = ImageDraw.Draw(gl)
    gd.line(points, fill=fill[:-1] + (max(6, int(fill[-1]*0.30)),), width=max(1,int(width*SCALE*3)))
    gl = gl.filter(ImageFilter.GaussianBlur(max(1, int(glow*SCALE))))
    image.alpha_composite(gl)
    ImageDraw.Draw(image).line(points, fill=fill, width=max(1,int(width*SCALE)))


def arrow(draw: ImageDraw.ImageDraw, start, end, fill, width: int):
    draw.line([start,end], fill=fill, width=width)
    dx, dy = end[0]-start[0], end[1]-start[1]
    ang = math.atan2(dy, dx)
    head = max(7*SCALE, width*2.6)
    for sign in (-1,1):
        a = ang + math.pi + sign*math.pi/6
        p = (end[0]+head*math.cos(a), end[1]+head*math.sin(a))
        draw.line([end,p], fill=fill, width=width)


def parse_hhmm(hhmm: str) -> float:
    h, m = hhmm.split(":")
    return int(h)*60.0 + int(m)


def interpolate_path(progress: float) -> Tuple[float,float,float,float,str]:
    """Smoothly interpolate NASA central-line table by progress 0..1."""
    p = clamp(progress)*(len(NASA_PATH)-1)
    i = min(len(NASA_PATH)-2, int(math.floor(p)))
    f = p-i
    a = NASA_PATH[i]
    b = NASA_PATH[i+1]
    lat = lerp(a[1], b[1], f)
    # avoid dateline issue (only first few points are +E then near pole); our globe centers Atlantic
    lon_a, lon_b = a[2], b[2]
    if abs(lon_b-lon_a) > 180:
        if lon_b < lon_a: lon_b += 360
        else: lon_a += 360
    lon = lerp(lon_a, lon_b, f)
    if lon > 180: lon -= 360
    width = lerp(a[3], b[3], f)
    dur = lerp(a[4], b[4], f)
    minute = lerp(parse_hhmm(a[0]), parse_hhmm(b[0]), f)
    hh = int(minute//60) % 24
    mm = int(round(minute%60))
    if mm == 60:
        hh = (hh+1)%24; mm=0
    return lat, lon, width, dur, f"{hh:02d}:{mm:02d} UTC"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(min(1.0, math.sqrt(a)))


def local_path_speed_km_s(index: int) -> float:
    i0 = max(0, index-1)
    i1 = min(len(NASA_PATH)-1, index+1)
    a, b = NASA_PATH[i0], NASA_PATH[i1]
    distance = haversine_km(a[1], a[2], b[1], b[2])
    dt = (i1-i0)*120.0
    return distance/max(dt, 1.0)


# =============================================================================
# Simplified geography
# =============================================================================

# These are deliberately coarse outlines for cinematic context, NOT GIS boundaries.
COAST_POLYGONS: Dict[str, List[Tuple[float,float]]] = {
    "GREENLAND": [
        (83,-35),(80,-18),(75,-18),(70,-22),(65,-39),(60,-44),(60,-50),
        (65,-54),(70,-56),(75,-60),(80,-55),(83,-35)
    ],
    "ICELAND": [(66.6,-24.8),(66.2,-14.5),(63.3,-13.5),(63.0,-22.5),(64.5,-24.8),(66.6,-24.8)],
    "UK_IRELAND": [(58.5,-6),(56,-3),(54,-4),(52,1),(50,-1),(51,-5),(54,-6),(58.5,-6)],
    "IBERIA": [(43.8,-9.3),(43.7,-1.5),(42.7,3.2),(40,1.2),(37,-0.7),(36,-6),(38,-9),(43.8,-9.3)],
    "FRANCE_BENELUX": [(51,2),(50,8),(47.5,7),(43,3),(43,-1.5),(46,-2),(49,-1),(51,2)],
    "SCANDINAVIA": [(71,25),(69,17),(64,12),(59,10),(55,13),(58,18),(63,24),(68,30),(71,25)],
    "EUROPE_EAST": [(55,13),(54,25),(50,31),(45,30),(42,20),(45,13),(50,8),(55,13)],
    "N_AFRICA": [(37,-9),(36,3),(37,10),(34,15),(30,10),(28,-5),(32,-12),(37,-9)],
    "E_CANADA": [(60,-75),(55,-58),(50,-55),(46,-60),(45,-67),(49,-72),(55,-80),(60,-75)],
}


def ortho_project(lat_deg: float, lon_deg: float, center_lat_deg: float, center_lon_deg: float,
                  cx: float, cy: float, radius: float) -> Optional[Tuple[float,float,float]]:
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    lat0 = math.radians(center_lat_deg); lon0 = math.radians(center_lon_deg)
    dl = lon-lon0
    cosc = math.sin(lat0)*math.sin(lat) + math.cos(lat0)*math.cos(lat)*math.cos(dl)
    if cosc < 0:
        return None
    x = radius*math.cos(lat)*math.sin(dl)
    y = -radius*(math.cos(lat0)*math.sin(lat)-math.sin(lat0)*math.cos(lat)*math.cos(dl))
    return cx+x, cy+y, cosc


def project_poly(poly, center_lat, center_lon, cx, cy, radius):
    pts = []
    for lat, lon in poly:
        p = ortho_project(lat,lon,center_lat,center_lon,cx,cy,radius)
        if p is not None:
            pts.append((p[0],p[1]))
    return pts


# =============================================================================
# Renderer
# =============================================================================

class EclipseScene:
    def __init__(self):
        rng = np.random.default_rng(20260812)
        self.stars = []
        for _ in range(int(CONFIG["background_stars"])):
            depth = float(rng.uniform(0.15,1.0))
            self.stars.append((
                float(rng.uniform(-0.05*OUT_W, 1.05*OUT_W)),
                float(rng.uniform(-0.04*OUT_H, 1.04*OUT_H)),
                float(rng.uniform(0.3,1.5)*max(SCALE,0.6)*(0.55+depth)),
                int(rng.uniform(30,155)*(0.55+0.45*depth)),
                float(rng.uniform(0,2*math.pi)), depth
            ))
        self.noise_rng = np.random.default_rng(88)

    # -------------------------------------------------------------------------
    # Global visual primitives
    # -------------------------------------------------------------------------

    def camera(self, t: float) -> Tuple[float,float,float]:
        return (
            10*SCALE*math.sin(t*0.17),
            15*SCALE*math.sin(t*0.11+1.1),
            1.0+0.018*math.sin(t*0.09),
        )

    def background(self, t: float) -> Image.Image:
        yy = np.linspace(0,1,OUT_H,dtype=np.float32)[:,None]
        top = np.array([2,5,13],dtype=np.float32)
        bottom = np.array([0,1,5],dtype=np.float32)
        rgb = top[None,None,:]*(1-yy[:,:,None]) + bottom[None,None,:]*yy[:,:,None]
        rgb = np.repeat(rgb,OUT_W,axis=1)
        image = Image.fromarray(np.clip(rgb,0,255).astype(np.uint8),"RGB").convert("RGBA")
        dx,dy,_ = self.camera(t)
        d = ImageDraw.Draw(image)
        for x,y,r,a,phase,depth in self.stars:
            sx=x+dx*depth; sy=y+dy*depth
            alpha=int(a*(0.82+0.18*math.sin(0.7*t+phase)))
            d.ellipse((sx-r,sy-r,sx+r,sy+r),fill=(226,235,245,alpha))

        haze=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); hd=ImageDraw.Draw(haze)
        hd.ellipse((-0.5*OUT_W,0.12*OUT_H,0.58*OUT_W,0.90*OUT_H),fill=(21,54,100,28))
        hd.ellipse((0.42*OUT_W,-0.20*OUT_H,1.35*OUT_W,0.58*OUT_H),fill=(76,46,101,18))
        haze=haze.filter(ImageFilter.GaussianBlur(max(34,int(150*SCALE))))
        image.alpha_composite(haze)
        return image

    def title(self, image: Image.Image, eyebrow: str, title: str, subtitle: str="", alpha: int=255):
        x=int(OUT_W*0.075)
        draw_text(image,eyebrow.upper(),(x,int(OUT_H*0.079)),size=max(8,int(12*SCALE)),
                  fill=(183,204,221,min(alpha,190)),bold=True,stroke=1,anchor="la")
        draw_text(image,title,(x,int(OUT_H*0.117)),size=max(17,int(40*SCALE)),
                  fill=(246,249,252,alpha),bold=True,stroke=1,anchor="la")
        if subtitle:
            draw_text(image,subtitle,(x,int(OUT_H*0.168)),size=max(9,int(15*SCALE)),
                      fill=(160,181,198,min(alpha,215)),bold=False,stroke=1,anchor="la")

    def label(self,image,text,xy,size=12,alpha=190,anchor="la",bold=True,fill=None):
        c = fill if fill is not None else (199,216,230,alpha)
        if len(c)==3: c=(*c,alpha)
        draw_text(image,text,xy,size=max(8,int(size*SCALE)),fill=c,bold=bold,stroke=1,anchor=anchor)

    def equation(self,image,text,xy,size=25,alpha=240,anchor="la"):
        draw_text(image,text,xy,size=max(10,int(size*SCALE)),fill=(225,235,244,alpha),bold=False,stroke=1,anchor=anchor)

    def draw_caption(self,image,t):
        text=caption_at(t)
        if not text: return
        y0=int(OUT_H*0.806)
        h=OUT_H-y0
        overlay=np.zeros((h,OUT_W,4),dtype=np.uint8)
        overlay[...,3]=np.linspace(0,215,h,dtype=np.uint8)[:,None]
        image.alpha_composite(Image.fromarray(overlay,"RGBA"),(0,y0))
        draw_wrapped_text(image,text,(int(OUT_W*0.075),int(OUT_H*0.858)),int(OUT_W*0.84),
                          max(10,int(17*SCALE)),fill=(232,238,244,240),line_spacing=max(2,int(6*SCALE)))

    def footer(self,image):
        y=int(OUT_H*0.972)
        self.label(image,"NASA/GSFC PATH DATA • CINEMATIC RECONSTRUCTION",(int(OUT_W*0.055),y),size=9,alpha=115,anchor="ls")
        self.label(image,"NOT TO VISUAL SCALE",(int(OUT_W*0.945),y),size=9,alpha=115,anchor="rs")

    def draw_sun(self,image,cx,cy,radius,t,corona=False):
        # Corona rays for totality scenes
        if corona:
            rays=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); rd=ImageDraw.Draw(rays)
            rng=np.random.default_rng(777)
            for i in range(110):
                ang=2*math.pi*i/110 + 0.012*math.sin(t*0.4+i)
                jitter=0.78+0.45*float(rng.random())
                r0=radius*1.02
                r1=radius*(2.2+2.3*jitter)
                a=int(16+30*jitter)
                p0=(cx+r0*math.cos(ang),cy+r0*math.sin(ang))
                p1=(cx+r1*math.cos(ang),cy+r1*math.sin(ang))
                rd.line([p0,p1],fill=(218,232,245,a),width=max(1,int((0.8+1.2*jitter)*SCALE)))
            rays=rays.filter(ImageFilter.GaussianBlur(max(2,int(7*SCALE))))
            image.alpha_composite(rays)

        bloom=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); bd=ImageDraw.Draw(bloom)
        for mult,a in [(5.0,7),(3.6,14),(2.5,30),(1.65,65),(1.17,105)]:
            rr=radius*mult
            bd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=(255,171,69,a))
        bloom=bloom.filter(ImageFilter.GaussianBlur(max(5,int(radius*0.75))))
        image.alpha_composite(bloom)
        rr=max(8,int(radius)); size=rr*2+4
        yy,xx=np.mgrid[-rr-2:rr+2,-rr-2:rr+2]
        rad=np.sqrt(xx*xx+yy*yy)/max(rr,1)
        mask=rad<=1
        shade=np.clip(1-rad,0,1)
        gran=0.94+0.05*np.sin(xx*0.19+yy*0.07+t*0.7)*np.sin(yy*0.13-t*0.31)
        arr=np.zeros((size,size,4),dtype=np.uint8)
        arr[...,0]=np.clip(239+16*shade,0,255)
        arr[...,1]=np.clip((126+106*shade)*gran,0,255)
        arr[...,2]=np.clip((35+58*shade)*gran,0,255)
        arr[...,3]=(mask*255).astype(np.uint8)
        image.alpha_composite(Image.fromarray(arr,"RGBA"),(int(cx-size/2),int(cy-size/2)))

    def draw_moon_disk(self,image,cx,cy,radius,t,lit=False):
        rr=max(8,int(radius)); size=rr*2+6
        yy,xx=np.mgrid[-rr-3:rr+3,-rr-3:rr+3]
        r2=(xx/max(rr,1))**2+(yy/max(rr,1))**2
        mask=r2<=1
        if lit:
            nz=np.sqrt(np.clip(1-r2,0,1))
            light=np.clip(-0.55*(xx/max(rr,1))-0.10*(yy/max(rr,1))+0.83*nz,0,1)
            base=28+150*light
            crater=8*np.sin(xx*0.17+yy*0.11)+6*np.cos(xx*0.07-yy*0.19)
            g=np.clip(base+crater,0,190)
        else:
            g=np.zeros_like(xx,dtype=float)+2
        arr=np.zeros((size,size,4),dtype=np.uint8)
        arr[...,0]=np.clip(g*0.96,0,255)
        arr[...,1]=np.clip(g*0.98,0,255)
        arr[...,2]=np.clip(g,0,255)
        arr[...,3]=(mask*255).astype(np.uint8)
        image.alpha_composite(Image.fromarray(arr,"RGBA"),(int(cx-size/2),int(cy-size/2)))

    def draw_totality(self,image,cx,cy,radius,t,diamond=0.0):
        # White-silver solar corona. This is a cinematic reconstruction, not a
        # measured coronal brightness map. The black disk radius is the Moon.
        corona=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); cd=ImageDraw.Draw(corona)
        rng=np.random.default_rng(1262026)
        for i in range(180):
            ang=2*math.pi*i/180.0
            structure=0.55+0.45*abs(math.cos(2.0*ang-0.35))
            jitter=0.82+0.36*float(rng.random())
            r0=radius*(1.005+0.02*float(rng.random()))
            r1=radius*(1.65+2.9*structure*jitter)
            alpha=int(15+36*structure*jitter)
            p0=(cx+r0*math.cos(ang),cy+r0*math.sin(ang))
            p1=(cx+r1*math.cos(ang),cy+r1*math.sin(ang))
            cd.line([p0,p1],fill=(222,235,248,alpha),width=max(1,int((0.7+0.9*structure)*SCALE)))
        # dense inner corona
        for mult,a,w in [(1.50,30,14),(1.30,46,11),(1.16,78,8),(1.07,125,4)]:
            rr=radius*mult
            cd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=(231,240,249,a),width=max(1,int(w*SCALE)))
        corona=corona.filter(ImageFilter.GaussianBlur(max(1,int(4.5*SCALE))))
        image.alpha_composite(corona)

        # crisp inner pearly rim
        rim=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); rd=ImageDraw.Draw(rim)
        rr=radius*1.014
        rd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=(244,248,252,210),width=max(1,int(2.2*SCALE)))
        rim=rim.filter(ImageFilter.GaussianBlur(max(0.6,1.2*SCALE)))
        image.alpha_composite(rim)

        self.draw_moon_disk(image,cx,cy,radius*0.985,t,lit=False)
        d=ImageDraw.Draw(image)
        d.ellipse((cx-radius*1.001,cy-radius*1.001,cx+radius*1.001,cy+radius*1.001),
                  outline=(255,102,86,72),width=max(1,int(1.0*SCALE)))
        if diamond>0:
            ang=-0.55
            px=cx+radius*math.cos(ang); py=cy+radius*math.sin(ang)
            gl=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); gd=ImageDraw.Draw(gl)
            rr=radius*0.18*diamond
            gd.ellipse((px-rr,py-rr,px+rr,py+rr),fill=(255,253,235,int(255*diamond)))
            gd.line((px-radius*1.05*diamond,py,px+radius*1.05*diamond,py),fill=(255,242,212,int(190*diamond)),width=max(1,int(2*SCALE)))
            gd.line((px,py-radius*0.55*diamond,px,py+radius*0.55*diamond),fill=(255,247,224,int(125*diamond)),width=max(1,int(1*SCALE)))
            gl=gl.filter(ImageFilter.GaussianBlur(max(1,int(4*SCALE))))
            image.alpha_composite(gl)

    def draw_earth_globe(self,image,cx,cy,radius,t,center_lat=55.0,center_lon=-22.0,with_path=False,path_progress=0.0,city_marks=False):
        # Atmospheric bloom
        atmos=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); ad=ImageDraw.Draw(atmos)
        for mult,a in [(1.08,18),(1.045,30),(1.015,58)]:
            rr=radius*mult
            ad.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=(93,181,245,a),width=max(1,int(3*SCALE)))
        atmos=atmos.filter(ImageFilter.GaussianBlur(max(2,int(9*SCALE))))
        image.alpha_composite(atmos)

        # Sphere with directional illumination
        rr=max(20,int(radius)); size=rr*2+8
        yy,xx=np.mgrid[-rr-4:rr+4,-rr-4:rr+4]
        nx=xx/max(rr,1); ny=yy/max(rr,1); q=nx*nx+ny*ny
        mask=q<=1
        nz=np.sqrt(np.clip(1-q,0,1))
        # Sunlit from upper left
        light=np.clip(-0.43*nx-0.18*ny+0.88*nz,0,1)
        limb=np.clip(nz,0,1)
        arr=np.zeros((size,size,4),dtype=np.uint8)
        arr[...,0]=np.clip(5+17*light,0,255)
        arr[...,1]=np.clip(20+58*light,0,255)
        arr[...,2]=np.clip(37+92*light,0,255)
        arr[...,3]=(mask*255).astype(np.uint8)
        sphere=Image.fromarray(arr,"RGBA")
        image.alpha_composite(sphere,(int(cx-size/2),int(cy-size/2)))

        # Coastlines & filled land polygons
        land_layer=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); ld=ImageDraw.Draw(land_layer)
        for name,poly in COAST_POLYGONS.items():
            pts=project_poly(poly,center_lat,center_lon,cx,cy,radius)
            if len(pts)>=3:
                ld.polygon(pts,fill=(62,99,73,175))
                ld.line(pts+[pts[0]],fill=(130,161,137,150),width=max(1,int(1.2*SCALE)))
        image.alpha_composite(land_layer)

        # Lat/lon graticule
        grid=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); gd=ImageDraw.Draw(grid)
        for lat in range(20,81,10):
            pts=[]
            for lon in np.linspace(-90,45,150):
                p=ortho_project(lat,float(lon),center_lat,center_lon,cx,cy,radius)
                if p: pts.append((p[0],p[1]))
            if len(pts)>1: gd.line(pts,fill=(159,190,207,32),width=max(1,int(SCALE)))
        for lon in range(-80,41,20):
            pts=[]
            for lat in np.linspace(20,85,130):
                p=ortho_project(float(lat),lon,center_lat,center_lon,cx,cy,radius)
                if p: pts.append((p[0],p[1]))
            if len(pts)>1: gd.line(pts,fill=(159,190,207,28),width=max(1,int(SCALE)))
        image.alpha_composite(grid)

        if with_path:
            projected=[]
            for _,lat,lon,_,_ in NASA_PATH:
                p=ortho_project(lat,lon,center_lat,center_lon,cx,cy,radius)
                if p: projected.append((p[0],p[1]))
            soft_line(image,projected,(255,94,106,185),width=3.0,glow=10.0)

            lat,lon,width,dur,clock=interpolate_path(path_progress)
            p=ortho_project(lat,lon,center_lat,center_lon,cx,cy,radius)
            if p:
                sx,sy,_=p
                # Visual shadow size is intentionally exaggerated for readability.
                shadow=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); sd=ImageDraw.Draw(shadow)
                sr=max(14*SCALE, radius*(width/40000.0)*16.0)
                for mult,a in [(3.2,18),(2.1,28),(1.5,50),(1.0,125)]:
                    r=sr*mult
                    sd.ellipse((sx-r,sy-r,sx+r,sy+r),fill=(0,0,4,a))
                shadow=shadow.filter(ImageFilter.GaussianBlur(max(2,int(8*SCALE))))
                image.alpha_composite(shadow)
                d=ImageDraw.Draw(image)
                d.ellipse((sx-4*SCALE,sy-4*SCALE,sx+4*SCALE,sy+4*SCALE),fill=(255,116,124,235))
                self.label(image,clock,(int(sx+14*SCALE),int(sy-12*SCALE)),size=10,alpha=190)

        if city_marks:
            d=ImageDraw.Draw(image)
            for city in ["REYKJAVIK","LEON","MADRID","BARCELONA","LONDON","PARIS"]:
                lat,lon=CITY_COORDS[city]
                p=ortho_project(lat,lon,center_lat,center_lon,cx,cy,radius)
                if p:
                    x,y,_=p
                    total=city in {"REYKJAVIK","LEON"}
                    col=(255,208,110,235) if total else (183,215,235,220)
                    d.ellipse((x-3*SCALE,y-3*SCALE,x+3*SCALE,y+3*SCALE),fill=col)
                    if city in {"REYKJAVIK","LEON","MADRID","LONDON"}:
                        self.label(image,city,(int(x+9*SCALE),int(y)),size=8,alpha=160,anchor="lm",fill=col)

    # -------------------------------------------------------------------------
    # Scenes
    # -------------------------------------------------------------------------

    def scene_hook(self,image,t,local):
        # Black Sun close-up: premium first-frame hook.
        p=smoothstep(local)
        cx=OUT_W*0.53; cy=OUT_H*0.43
        r=(170+28*p)*SCALE
        self.draw_totality(image,cx,cy,r,t,diamond=max(0.0,1.0-local*3.2))
        self.title(image,"12 AUGUST 2026","EUROPE'S SKY WENT DARK","A real total solar eclipse crossed the North Atlantic into Spain.")
        self.label(image,"TOTAL SOLAR ECLIPSE",(int(OUT_W*0.075),int(OUT_H*0.715)),size=13,alpha=180)
        draw_text(image,"2m 18s",(int(OUT_W*0.075),int(OUT_H*0.768)),size=max(20,int(48*SCALE)),
                  fill=(246,249,252,245),bold=True,stroke=1,anchor="ls")
        self.label(image,"MAX CENTRAL TOTALITY",(int(OUT_W*0.37),int(OUT_H*0.766)),size=10,alpha=150,anchor="ls")

    def scene_alignment(self,image,t,local):
        self.title(image,"THE GEOMETRY","SUN → MOON → EARTH","A near-perfect alignment turns a shadow into totality.")
        y=OUT_H*0.48
        sunx=OUT_W*0.15; moonx=OUT_W*0.50; earthx=OUT_W*0.84
        self.draw_sun(image,sunx,y,72*SCALE,t)
        self.draw_moon_disk(image,moonx,y,37*SCALE,t,lit=True)
        self.draw_earth_globe(image,earthx,y,90*SCALE,t,center_lat=28,center_lon=-15)

        # Penumbra and umbra cones
        cone=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); d=ImageDraw.Draw(cone)
        # outer partial shadow
        d.polygon([(moonx,y-45*SCALE),(earthx-70*SCALE,y-62*SCALE),(earthx-70*SCALE,y+62*SCALE),(moonx,y+45*SCALE)],
                  fill=(115,139,166,24))
        # inner total shadow
        d.polygon([(moonx,y-20*SCALE),(earthx-72*SCALE,y-11*SCALE),(earthx-72*SCALE,y+11*SCALE),(moonx,y+20*SCALE)],
                  fill=(0,0,2,145))
        cone=cone.filter(ImageFilter.GaussianBlur(max(1,int(2*SCALE))))
        image.alpha_composite(cone)
        self.label(image,"PENUMBRA = PARTIAL",(int(OUT_W*0.54),int(y-93*SCALE)),size=10,alpha=165)
        self.label(image,"UMBRA = TOTAL",(int(OUT_W*0.59),int(y+56*SCALE)),size=10,alpha=195,fill=(255,198,104))

        x=int(OUT_W*0.075)
        self.equation(image,"Moon angular diameter ≳ Sun angular diameter",(x,int(OUT_H*0.675)),size=18)
        self.label(image,"The objects are NOT drawn to scale here.",(x,int(OUT_H*0.720)),size=10,alpha=130,bold=False)

    def scene_path(self,image,t,local):
        self.title(image,"NASA CENTRAL-LINE DATA","THE SHADOW'S REAL PATH","Two-minute NASA/GSFC path points drive this animation.")
        cx=OUT_W*0.52; cy=OUT_H*0.52; r=360*SCALE
        # delayed sweep so viewer first reads the map
        p=smoothstep((local-0.08)/0.86)
        self.draw_earth_globe(image,cx,cy,r,t,center_lat=60,center_lon=-24,with_path=True,path_progress=p)
        self.label(image,"SIBERIA → ARCTIC → GREENLAND → ICELAND → ATLANTIC → SPAIN",(int(OUT_W*0.075),int(OUT_H*0.742)),size=10,alpha=175)
        lat,lon,width,dur,clock=interpolate_path(p)
        self.label(image,f"CENTER  {lat:05.1f}°N  {abs(lon):05.1f}°{'W' if lon<0 else 'E'}",(int(OUT_W*0.075),int(OUT_H*0.777)),size=10,alpha=150)

    def scene_greatest(self,image,t,local):
        self.title(image,"GREATEST ECLIPSE","17:45:54 UTC","The deepest geometry occurred between Greenland and Iceland.")
        # globe on right, data on left
        self.draw_earth_globe(image,OUT_W*0.68,OUT_H*0.50,250*SCALE,t,center_lat=66,center_lon=-25,with_path=True,path_progress=22/(len(NASA_PATH)-1))
        x=int(OUT_W*0.075)
        reveal=ease_out_cubic(local)
        self.label(image,"CENTER",(x,int(OUT_H*0.285)),size=10,alpha=int(145*reveal))
        draw_text(image,"65.2° N",(x,int(OUT_H*0.325)),size=max(18,int(31*SCALE)),fill=(240,245,250,int(250*reveal)),bold=True,stroke=1)
        draw_text(image,"25.2° W",(x,int(OUT_H*0.370)),size=max(18,int(31*SCALE)),fill=(240,245,250,int(250*reveal)),bold=True,stroke=1)
        self.label(image,"PATH WIDTH",(x,int(OUT_H*0.455)),size=10,alpha=int(145*reveal))
        draw_text(image,"294 km",(x,int(OUT_H*0.505)),size=max(20,int(39*SCALE)),fill=(255,204,112,int(250*reveal)),bold=True,stroke=1)
        self.label(image,"CENTRAL TOTALITY",(x,int(OUT_H*0.590)),size=10,alpha=int(145*reveal))
        draw_text(image,"2:18.2",(x,int(OUT_H*0.645)),size=max(22,int(45*SCALE)),fill=(245,249,252,int(250*reveal)),bold=True,stroke=1)
        self.label(image,"MAGNITUDE ≈ 1.039",(x,int(OUT_H*0.711)),size=11,alpha=int(175*reveal))

    def scene_cities(self,image,t,local):
        self.title(image,"EUROPE WASN'T EQUALLY DARK","TOTAL VS. DEEP PARTIAL","NASA city circumstances show the difference.")
        # mini globe top-right
        self.draw_earth_globe(image,OUT_W*0.76,OUT_H*0.37,175*SCALE,t,center_lat=52,center_lon=-8,with_path=True,path_progress=1.0,city_marks=True)

        # Table-like cinematic list, but sparse and readable on mobile
        rows=[
            ("REYKJAVIK","TOTAL","17:48–17:49"),
            ("LEON","TOTAL","20:28–20:30"),
            ("MADRID","99%","20:32 max"),
            ("BARCELONA","99%","20:29 max"),
            ("LONDON","91%","19:13 max"),
            ("PARIS","92%","20:17 max"),
        ]
        x0=int(OUT_W*0.075); y0=int(OUT_H*0.335)
        for i,(city,cover,tm) in enumerate(rows):
            y=y0+i*int(OUT_H*0.067)
            a=int(220*smoothstep(local*2.2-i*0.11))
            self.label(image,city,(x0,y),size=11,alpha=a)
            total=cover=="TOTAL"
            col=(255,203,106,a) if total else (194,219,236,a)
            draw_text(image,cover,(int(OUT_W*0.34),y),size=max(9,int(13*SCALE)),fill=col,bold=True,stroke=1,anchor="la")
            self.label(image,tm,(int(OUT_W*0.49),y),size=9,alpha=int(a*0.82),bold=False)
        self.label(image,"TIMES ARE LOCAL • % = MAXIMUM SOLAR DISK AREA COVERED",(x0,int(OUT_H*0.752)),size=8,alpha=110,bold=False)

    def scene_spain(self,image,t,local):
        self.title(image,"NORTHERN SPAIN","TOTALITY NEAR SUNSET","The eclipse reached Spain with the Sun already low in the west.")

        # Cinematic horizon gradient, intentionally not a local astronomical horizon solver.
        sky=np.zeros((int(OUT_H*0.60),OUT_W,4),dtype=np.uint8)
        h=sky.shape[0]
        for y in range(h):
            q=y/max(1,h-1)
            sky[y,:,0]=np.clip(16+90*q,0,255)
            sky[y,:,1]=np.clip(22+42*q,0,255)
            sky[y,:,2]=np.clip(48+15*q,0,255)
            sky[y,:,3]=220
        image.alpha_composite(Image.fromarray(sky,"RGBA"),(0,int(OUT_H*0.24)))
        horizon_y=OUT_H*0.69
        d=ImageDraw.Draw(image)
        # layered silhouettes
        mountains=[(0,horizon_y),(OUT_W*0.15,horizon_y-40*SCALE),(OUT_W*0.28,horizon_y-12*SCALE),(OUT_W*0.43,horizon_y-62*SCALE),(OUT_W*0.58,horizon_y-24*SCALE),(OUT_W*0.74,horizon_y-55*SCALE),(OUT_W,horizon_y-18*SCALE),(OUT_W,OUT_H),(0,OUT_H)]
        d.polygon(mountains,fill=(3,5,9,255))
        p=smoothstep(local)
        sx=lerp(OUT_W*0.65,OUT_W*0.55,p)
        sy=lerp(OUT_H*0.49,OUT_H*0.58,p)
        self.draw_totality(image,sx,sy,115*SCALE,t,diamond=max(0,1-local*2.3))
        self.label(image,"CINEMATIC HORIZON RECONSTRUCTION",(int(OUT_W*0.075),int(OUT_H*0.744)),size=8,alpha=105,bold=False)
        self.label(image,"LEON  totality ~20:28–20:30 local",(int(OUT_W*0.075),int(OUT_H*0.778)),size=10,alpha=175)

    def scene_safety(self,image,t,local):
        self.title(image,"ONE RULE TO REMEMBER","PROTECT YOUR EYES","Totality is the only brief exception.")
        cx=OUT_W*0.50; cy=OUT_H*0.43
        r=145*SCALE
        # partial phase graphic
        self.draw_sun(image,cx,cy,r,t)
        moon_offset=lerp(r*1.2,r*0.55,smoothstep(local))
        self.draw_moon_disk(image,cx-moon_offset,cy,r*1.01,t,lit=False)
        x=int(OUT_W*0.075)
        self.label(image,"PARTIAL PHASES",(x,int(OUT_H*0.665)),size=11,alpha=170)
        draw_text(image,"ECLIPSE GLASSES",(x,int(OUT_H*0.712)),size=max(18,int(31*SCALE)),fill=(255,203,108,245),bold=True,stroke=1)
        self.label(image,"CAMERA / BINOCULARS / TELESCOPE",(x,int(OUT_H*0.765)),size=9,alpha=140)
        draw_text(image,"FRONT SOLAR FILTER",(int(OUT_W*0.49),int(OUT_H*0.765)),size=max(9,int(14*SCALE)),fill=(205,226,240,220),bold=True,stroke=1,anchor="la")

    def render(self,t: float) -> np.ndarray:
        image=self.background(t)
        shot=get_shot(t)
        local=smoothstep((t-shot["start"])/max(shot["end"]-shot["start"],1e-9))
        name=shot["name"]
        if name=="hook": self.scene_hook(image,t,local)
        elif name=="alignment": self.scene_alignment(image,t,local)
        elif name=="path": self.scene_path(image,t,local)
        elif name=="greatest": self.scene_greatest(image,t,local)
        elif name=="cities": self.scene_cities(image,t,local)
        elif name=="spain": self.scene_spain(image,t,local)
        else: self.scene_safety(image,t,local)
        self.draw_caption(image,t)
        self.footer(image)
        return apply_grade(np.asarray(image.convert("RGB")))


# =============================================================================
# Output writers
# =============================================================================

def save_path_csv(path: Path):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["utc","central_lat_deg","central_lon_deg","path_width_km","central_duration_s","approx_ground_speed_km_s"])
        for i,row in enumerate(NASA_PATH):
            w.writerow([row[0],row[1],row[2],row[3],row[4],local_path_speed_km_s(i)])


def save_summary(path: Path):
    data={
        "title": CONFIG["title"],
        "event": ECLIPSE_FACTS,
        "city_data": [
            {"city":r[0],"country":r[1],"kind":r[2],"partial_begins_local":r[3],"maximum_or_totality_local":r[4],"partial_ends_local":r[5],"coverage":r[6]}
            for r in CITY_DATA
        ],
        "sources": [
            "NASA GSFC: https://eclipse.gsfc.nasa.gov/SEpath/SEpath2001/SE2026Aug12Tpath.html",
            "NASA GSFC: https://eclipse.gsfc.nasa.gov/SEsearch/SEdata.php?Ecl=20260812",
            "NASA Science: https://science.nasa.gov/eclipses/future-eclipses/total-solar-eclipse-on-august-12-2026/",
            "NASA Safety: https://science.nasa.gov/eclipses/safety/",
        ],
        "visualization_notes": NASA_SOURCE_NOTES,
        "narration": [text for _,_,text in FULL_CAPTIONS],
    }
    path.write_text(json.dumps(data,indent=2),encoding="utf-8")


def make_previews(scene: EclipseScene):
    times = [0.6,2.2,4.2,6.2,8.6,10.8] if QUICK_MODE else [2.5,10.0,19.5,28.0,37.5,46.2,52.5]
    for t in times:
        arr=scene.render(t)
        Image.fromarray(arr).save(PREVIEW_DIR/f"preview_{t:g}s.png")


def render_video(scene: EclipseScene, path: Path):
    frames=int(round(FPS*DURATION))
    writer=iio.get_writer(
        path,
        fps=FPS,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=[
            "-crf", "20" if QUICK_MODE else "17",
            "-preset", "medium" if QUICK_MODE else "slow",
            "-movflags", "+faststart",
        ],
    )
    try:
        for i in tqdm(range(frames),desc="Rendering Aug 12 2026 eclipse Short"):
            writer.append_data(scene.render(i/FPS))
    finally:
        writer.close()


def main():
    scene=EclipseScene()
    basename=str(CONFIG["output_basename"])
    mp4=OUTPUT_ROOT/f"{basename}_final.mp4"
    srt=OUTPUT_ROOT/f"{basename}.srt"
    csvp=DATA_DIR/"nasa_central_line_path.csv"
    jsonp=DATA_DIR/"eclipse_fact_sheet.json"

    write_srt(CAPTIONS,srt)
    save_path_csv(csvp)
    save_summary(jsonp)
    make_previews(scene)
    render_video(scene,mp4)

    print("\nRender complete")
    print(f"Video:     {mp4}")
    print(f"Subtitles: {srt}")
    print(f"Path CSV:  {csvp}")
    print(f"Fact JSON: {jsonp}")
    print("\nScience credit: Eclipse Predictions by Fred Espenak, NASA's GSFC.")

if __name__ == "__main__":
    main()
