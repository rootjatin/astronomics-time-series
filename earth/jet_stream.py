from __future__ import annotations

"""
The Jet Stream Is a River in the Sky — cinematic YouTube Short renderer

Creates a vertical 1080x1920 science short showing a stylized jet stream as
fast-moving ribbons of air near the top of the troposphere. The visuals are
cinematic and diagrammatic, not a numerical weather-model simulation.


Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    JET_STREAM_SHORT_QUICK=1 python the_jet_stream_is_a_river_in_the_sky.py

Full render
-----------
    python the_jet_stream_is_a_river_in_the_sky.py

4K vertical
-----------
    JET_STREAM_SHORT_4K=1 python the_jet_stream_is_a_river_in_the_sky.py
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

QUICK_MODE = os.environ.get("JET_STREAM_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("JET_STREAM_SHORT_4K", "0") == "1" and not QUICK_MODE

OUT_W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
OUT_H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 8 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SCALE = OUT_W / 1080.0
OUT_SIZE = (OUT_W, OUT_H)

OUTPUT_ROOT = Path("jet_stream_river_in_the_sky_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "THE JET STREAM IS A RIVER IN THE SKY",
    "subtitle": "fast air // temperature contrast // weather steering",
    "output_basename": "the_jet_stream_is_a_river_in_the_sky",
    "contrast": 1.09,
    "saturation": 1.08,
    "vignette": 0.24,
}

COLORS = {
    "space": (2, 8, 18),
    "upper_sky": (15, 54, 112),
    "sky": (34, 113, 193),
    "horizon": (113, 191, 235),
    "white": (246, 250, 255),
    "muted": (174, 207, 226),
    "cyan": (88, 224, 255),
    "blue": (72, 139, 255),
    "violet": (177, 126, 255),
    "gold": (255, 202, 91),
    "orange": (255, 141, 74),
    "red": (255, 91, 105),
    "green": (105, 235, 166),
    "cold": (101, 184, 255),
    "warm": (255, 139, 81),
    "land": (76, 139, 106),
    "ocean": (22, 88, 154),
    "night": (5, 18, 42),
}

FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.4, 7.2, "High above us, narrow bands of air race around the planet. These are jet streams."),
    (7.3, 17.0, "They usually form near the top of the troposphere, where strong temperature contrasts meet Earth's rotation."),
    (17.1, 27.4, "The wind can move far faster than the air near the ground, forming a long, fast ribbon that circles across continents and oceans."),
    (27.5, 38.4, "But the jet stream is not straight. It bends into giant waves, dipping south and arcing north as the atmosphere shifts."),
    (38.5, 49.0, "Those bends help steer weather systems and can pull cold air south or push warm air north."),
    (49.1, 57.5, "So the jet stream is not literally a river — but it behaves like a moving current in the atmosphere, constantly changing shape."),
]

if QUICK_MODE:
    factor = DURATION / 58.0
    CAPTIONS = [(a * factor, b * factor, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.8 if not QUICK_MODE else 1.75},
    {"name": "altitude", "start": 7.8 if not QUICK_MODE else 1.75, "end": 18.0 if not QUICK_MODE else 4.0},
    {"name": "speed", "start": 18.0 if not QUICK_MODE else 4.0, "end": 28.5 if not QUICK_MODE else 6.35},
    {"name": "waves", "start": 28.5 if not QUICK_MODE else 6.35, "end": 39.5 if not QUICK_MODE else 8.85},
    {"name": "weather", "start": 39.5 if not QUICK_MODE else 8.85, "end": 50.2 if not QUICK_MODE else 11.25},
    {"name": "finale", "start": 50.2 if not QUICK_MODE else 11.25, "end": DURATION},
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
    fnt = get_font(size, bold)
    words = value.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        box = draw.textbbox((0, 0), candidate, font=fnt, stroke_width=max(1, int(2 * SCALE)))
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
            font=fnt,
            fill=fill,
            stroke_width=max(1, int(2 * SCALE)),
            stroke_fill=(0, 0, 0, 220),
        )
        box = draw.textbbox((x, y), line, font=fnt, stroke_width=max(1, int(2 * SCALE)))
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
    x: float
    y: float
    size: float
    alpha: int
    speed: float
    phase: float


class JetStreamScene:
    def __init__(self):
        rng = np.random.default_rng(20260817)
        self.stars = [
            (float(rng.uniform(0, OUT_W)), float(rng.uniform(0, OUT_H * 0.34)), float(rng.uniform(0.4, 1.7) * SCALE), int(rng.uniform(20, 100)), float(rng.uniform(0, 2 * math.pi)))
            for _ in range(100 if QUICK_MODE else 260)
        ]
        self.clouds = [
            Cloud(
                x=float(rng.uniform(-0.1, 1.1)),
                y=float(rng.uniform(0.30, 0.78)),
                size=float(rng.uniform(35, 115) * SCALE),
                alpha=int(rng.uniform(35, 95)),
                speed=float(rng.uniform(0.002, 0.008)),
                phase=float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(30 if QUICK_MODE else 90)
        ]
        self.wind_particles = [
            {
                "u": float(rng.uniform(0, 1)),
                "offset": float(rng.normal(0, 0.42)),
                "size": float(rng.uniform(1.0, 3.8) * SCALE),
                "speed": float(rng.uniform(0.12, 0.28)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "alpha": int(rng.uniform(100, 235)),
            }
            for _ in range(85 if QUICK_MODE else 240)
        ]
        self.hud = [
            (float(rng.uniform(0, OUT_W)), float(rng.uniform(0, OUT_H)), float(rng.uniform(15, 110) * SCALE), int(rng.uniform(8, 34)), float(rng.uniform(0, 2 * math.pi)))
            for _ in range(35 if QUICK_MODE else 90)
        ]

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 168):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            box,
            radius=max(8, int(24 * SCALE)),
            fill=(2, 8, 20, alpha),
            outline=COLORS["cyan"] + (58,),
            width=max(1, int(2 * SCALE)),
        )
        image.alpha_composite(overlay)

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        yy = np.linspace(0, 1, OUT_H)[:, None]
        arr[..., 0] = np.clip(5 + 25 * yy, 0, 255)
        arr[..., 1] = np.clip(18 + 95 * yy, 0, 255)
        arr[..., 2] = np.clip(42 + 125 * yy, 0, 255)
        image = Image.fromarray(arr, "RGB").convert("RGBA")
        draw = ImageDraw.Draw(image)
        for x, y, r, a, ph in self.stars:
            alpha = int(a * (0.75 + 0.25 * math.sin(t * 1.3 + ph)))
            draw.ellipse((x-r, y-r, x+r, y+r), fill=COLORS["white"] + (alpha,))

        # Broad atmospheric glow near horizon.
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        horizon_y = int(OUT_H * 0.73)
        gd.rectangle((0, horizon_y - int(140*SCALE), OUT_W, horizon_y + int(120*SCALE)), fill=COLORS["horizon"] + (28,))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(8, int(45*SCALE)))))
        return image

    def draw_curved_earth(self, image: Image.Image, t: float, globe_center_y: float = 1.07):
        cx = OUT_W // 2
        cy = int(OUT_H * globe_center_y)
        r = int(OUT_W * 0.78)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=COLORS["ocean"] + (255,))

        # Stylized visible landmasses.
        continents = [
            [(0.20, -0.70), (0.40, -0.78), (0.56, -0.66), (0.49, -0.50), (0.28, -0.47), (0.14, -0.56)],
            [(-0.55, -0.64), (-0.33, -0.74), (-0.13, -0.61), (-0.18, -0.46), (-0.38, -0.44), (-0.58, -0.52)],
            [(-0.08, -0.43), (0.12, -0.49), (0.25, -0.34), (0.12, -0.23), (-0.10, -0.28)],
        ]
        rot = 0.04 * math.sin(t * 0.25)
        for poly in continents:
            pts = []
            for px, py in poly:
                x = px * math.cos(rot) - py * math.sin(rot)
                y = px * math.sin(rot) + py * math.cos(rot)
                pts.append((cx + x*r, cy + y*r))
            draw.polygon(pts, fill=COLORS["land"] + (235,))

        # Night shading on far side.
        shade = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        sd = ImageDraw.Draw(shade)
        sd.pieslice((cx-r, cy-r, cx+r, cy+r), start=240, end=75, fill=COLORS["night"] + (90,))
        shade = shade.filter(ImageFilter.GaussianBlur(max(2, int(6*SCALE))))
        overlay.alpha_composite(shade)
        image.alpha_composite(overlay)

        # Atmosphere rim.
        rim = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        rd = ImageDraw.Draw(rim)
        rd.arc((cx-r, cy-r, cx+r, cy+r), start=190, end=350, fill=COLORS["cyan"] + (160,), width=max(2, int(7*SCALE)))
        image.alpha_composite(rim.filter(ImageFilter.GaussianBlur(max(2, int(9*SCALE)))))
        image.alpha_composite(rim)

    def draw_clouds(self, image: Image.Image, t: float, alpha_scale: float = 1.0):
        overlay = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        for cloud in self.clouds:
            x = ((cloud.x + t * cloud.speed) % 1.2 - 0.1) * OUT_W
            y = cloud.y * OUT_H + math.sin(t*0.35 + cloud.phase) * 8*SCALE
            s = cloud.size
            a = int(cloud.alpha * alpha_scale)
            for dx, dy, rr in [(-0.55, 0.08, 0.42), (-0.18, -0.05, 0.55), (0.22, 0.00, 0.50), (0.55, 0.10, 0.36)]:
                r = s*rr
                draw.ellipse((x+dx*s-r, y+dy*s-r*0.55, x+dx*s+r, y+dy*s+r*0.55), fill=COLORS["white"] + (a,))
        overlay = overlay.filter(ImageFilter.GaussianBlur(max(2, int(7*SCALE))))
        image.alpha_composite(overlay)

    def jet_path(self, u: float, t: float, amplitude: float = 1.0, phase_shift: float = 0.0) -> Tuple[float, float]:
        x = lerp(-0.10*OUT_W, 1.10*OUT_W, u)
        base_y = OUT_H * 0.39
        wave1 = math.sin(u * 2.2 * math.pi + t * 0.35 + phase_shift)
        wave2 = 0.45 * math.sin(u * 5.0 * math.pi - t * 0.18 + phase_shift*0.5)
        y = base_y + amplitude * (wave1 + wave2) * 105 * SCALE
        return x, y

    def draw_jet_ribbon(self, image: Image.Image, t: float, amplitude: float = 1.0, alpha: int = 235, width_scale: float = 1.0, phase_shift: float = 0.0):
        pts = [self.jet_path(float(u), t, amplitude, phase_shift) for u in np.linspace(0, 1, 190)]
        overlay = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)

        # Wide soft glow.
        draw.line(pts, fill=COLORS["cyan"] + (70,), width=max(6, int(55*SCALE*width_scale)), joint="curve")
        glow = overlay.filter(ImageFilter.GaussianBlur(max(6, int(18*SCALE))))
        image.alpha_composite(glow)

        # Inner layered ribbon.
        layer = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        ld = ImageDraw.Draw(layer)
        ld.line(pts, fill=COLORS["blue"] + (int(alpha*0.45),), width=max(4, int(36*SCALE*width_scale)), joint="curve")
        ld.line(pts, fill=COLORS["cyan"] + (alpha,), width=max(3, int(15*SCALE*width_scale)), joint="curve")
        ld.line(pts, fill=COLORS["white"] + (int(alpha*0.72),), width=max(1, int(4*SCALE*width_scale)), joint="curve")
        image.alpha_composite(layer)

        # Moving air streaks on the ribbon.
        streaks = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        sd = ImageDraw.Draw(streaks)
        for item in self.wind_particles:
            u = (item["u"] + t*item["speed"]) % 1.0
            x, y = self.jet_path(u, t, amplitude, phase_shift)
            tangent_u = min(1.0, u + 0.008)
            x2, y2 = self.jet_path(tangent_u, t, amplitude, phase_shift)
            dx, dy = x2-x, y2-y
            norm = max(math.hypot(dx, dy), 1e-6)
            nx, ny = -dy/norm, dx/norm
            offset = item["offset"] * 30 * SCALE
            x += nx * offset
            y += ny * offset
            tail = (12 + 26 * (0.5 + 0.5*math.sin(item["phase"]))) * SCALE
            tx = x - dx/norm*tail
            ty = y - dy/norm*tail
            rr = item["size"]
            a = item["alpha"]
            sd.line((tx,ty,x,y), fill=COLORS["white"] + (a//2,), width=max(1,int(2*SCALE)))
            sd.ellipse((x-rr,y-rr,x+rr,y+rr), fill=COLORS["white"] + (a,))
        image.alpha_composite(streaks.filter(ImageFilter.GaussianBlur(max(1, int(2*SCALE)))))
        image.alpha_composite(streaks)

    def draw_temperature_bands(self, image: Image.Image, t: float, local: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        split_y = int(OUT_H*0.49 + math.sin(t*0.25)*18*SCALE)
        draw.rectangle((0, split_y, OUT_W, OUT_H), fill=COLORS["warm"] + (int(38*local),))
        draw.rectangle((0, 0, OUT_W, split_y), fill=COLORS["cold"] + (int(32*local),))
        image.alpha_composite(overlay)
        draw_text(image, "COLDER AIR", (int(OUT_W*0.14), int(OUT_H*0.24)), 17 if not QUICK_MODE else 8, COLORS["cold"]+(235,), True, "ma", 1)
        draw_text(image, "WARMER AIR", (int(OUT_W*0.78), int(OUT_H*0.61)), 17 if not QUICK_MODE else 8, COLORS["warm"]+(235,), True, "ma", 1)

    def draw_altitude_diagram(self, image: Image.Image, t: float, local: float):
        # Cross-section: Earth at bottom, atmosphere layers above.
        overlay = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        ground_y = int(OUT_H*0.69)
        draw.rectangle((0, ground_y, OUT_W, OUT_H), fill=(24, 71, 73, 255))
        draw.rectangle((0, int(OUT_H*0.50), OUT_W, ground_y), fill=COLORS["sky"]+(55,))
        draw.rectangle((0, int(OUT_H*0.30), OUT_W, int(OUT_H*0.50)), fill=COLORS["upper_sky"]+(65,))
        image.alpha_composite(overlay)

        # Altitude ruler.
        x = int(OUT_W*0.12)
        y0 = int(OUT_H*0.26)
        y1 = ground_y
        d = ImageDraw.Draw(image)
        d.line((x,y0,x,y1), fill=COLORS["white"]+(145,), width=max(2,int(3*SCALE)))
        for frac, label in [(0.0,"15 km"),(0.40,"10 km"),(0.75,"5 km"),(1.0,"0 km")]:
            y = int(lerp(y0,y1,frac))
            d.line((x-int(10*SCALE),y,x+int(10*SCALE),y), fill=COLORS["white"]+(160,), width=max(1,int(2*SCALE)))
            draw_text(image,label,(x-int(18*SCALE),y),14 if not QUICK_MODE else 7,COLORS["muted"]+(220,),False,"ra",1)

        # Jet layer ribbon.
        jet_y = int(OUT_H*0.37)
        glow = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        gd.rounded_rectangle((int(OUT_W*0.22), jet_y-int(28*SCALE), int(OUT_W*0.92), jet_y+int(28*SCALE)), radius=int(24*SCALE), fill=COLORS["cyan"]+(int(90*local),))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4,int(14*SCALE)))))
        d.rounded_rectangle((int(OUT_W*0.22), jet_y-int(10*SCALE), int(OUT_W*0.92), jet_y+int(10*SCALE)), radius=int(10*SCALE), fill=COLORS["cyan"]+(230,))
        for i in range(8):
            xx = int(lerp(OUT_W*0.28, OUT_W*0.88, ((i/8)+t*0.12)%1.0))
            d.polygon([(xx,jet_y),(xx-int(16*SCALE),jet_y-int(8*SCALE)),(xx-int(16*SCALE),jet_y+int(8*SCALE))], fill=COLORS["white"]+(220,))
        draw_text(image,"JET STREAM ZONE",(int(OUT_W*0.57),jet_y-int(38*SCALE)),18 if not QUICK_MODE else 9,COLORS["cyan"]+(245,),True,"ma",1)
        draw_text(image,"NEAR THE TOP OF THE TROPOSPHERE",(int(OUT_W*0.57),jet_y+int(42*SCALE)),14 if not QUICK_MODE else 7,COLORS["white"]+(220,),True,"ma",1)

    def draw_weather_systems(self, image: Image.Image, t: float, local: float):
        # Two stylized pressure systems below the jet.
        draw = ImageDraw.Draw(image)
        systems = [
            (int(OUT_W*0.28), int(OUT_H*0.61), "L", COLORS["cold"], t*0.33),
            (int(OUT_W*0.72), int(OUT_H*0.66), "H", COLORS["warm"], -t*0.25),
        ]
        for cx,cy,label,col,rot in systems:
            for ring in range(3):
                rr = int((55+ring*28)*SCALE)
                box = (cx-rr,cy-rr,cx+rr,cy+rr)
                start = (rot*60 + ring*30) % 360
                draw.arc(box, start=start, end=start+260, fill=col+(int(150-30*ring),), width=max(1,int((4-ring)*SCALE)))
            draw_text(image,label,(cx,cy),36 if not QUICK_MODE else 18,col+(245,),True,"mm",2)

        # Temperature arrows.
        arr = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        ad = ImageDraw.Draw(arr)
        cold_x = int(OUT_W*0.30)
        warm_x = int(OUT_W*0.68)
        arrow_alpha = int(210*local)
        ad.line((cold_x,int(OUT_H*0.34),cold_x,int(OUT_H*0.54)), fill=COLORS["cold"]+(arrow_alpha,), width=max(2,int(5*SCALE)))
        ad.polygon([(cold_x,int(OUT_H*0.56)),(cold_x-int(12*SCALE),int(OUT_H*0.53)),(cold_x+int(12*SCALE),int(OUT_H*0.53))], fill=COLORS["cold"]+(arrow_alpha,))
        ad.line((warm_x,int(OUT_H*0.72),warm_x,int(OUT_H*0.53)), fill=COLORS["warm"]+(arrow_alpha,), width=max(2,int(5*SCALE)))
        ad.polygon([(warm_x,int(OUT_H*0.50)),(warm_x-int(12*SCALE),int(OUT_H*0.53)),(warm_x+int(12*SCALE),int(OUT_H*0.53))], fill=COLORS["warm"]+(arrow_alpha,))
        image.alpha_composite(arr)

    def draw_title(self, image: Image.Image, t: float):
        intro_end = 5.6 if not QUICK_MODE else 1.2
        if t < intro_end:
            fade = smoothstep(t/(0.75 if not QUICK_MODE else 0.16))
            draw_text(image,"THE JET STREAM",(OUT_W//2,int(OUT_H*0.074)),38 if not QUICK_MODE else 19,COLORS["white"]+(int(245*fade),),True,"ma",2)
            draw_text(image,"IS A RIVER",(OUT_W//2,int(OUT_H*0.113)),48 if not QUICK_MODE else 24,COLORS["cyan"]+(int(250*fade),),True,"ma",2)
            draw_text(image,"IN THE SKY",(OUT_W//2,int(OUT_H*0.153)),38 if not QUICK_MODE else 19,COLORS["gold"]+(int(245*fade),),True,"ma",2)

    def draw_corner_label(self, image: Image.Image, label: str):
        draw_text(image,label,(54 if not QUICK_MODE else 27,58 if not QUICK_MODE else 29),18 if not QUICK_MODE else 9,COLORS["muted"]+(210,),True,"la",1)

    def draw_source_hud(self, image: Image.Image):
        draw_text(image,"CINEMATIC SCIENCE VISUALIZATION",(OUT_W-(48 if not QUICK_MODE else 24),72 if not QUICK_MODE else 36),15 if not QUICK_MODE else 7,COLORS["gold"]+(235,),True,"ra",1)
        draw_text(image,"AIRFLOW // DIAGRAMMATIC",(OUT_W-(48 if not QUICK_MODE else 24),101 if not QUICK_MODE else 51),14 if not QUICK_MODE else 7,COLORS["muted"]+(205,),False,"ra",1)
        draw_text(image,"JET POSITION CHANGES WITH WEATHER",(OUT_W-(48 if not QUICK_MODE else 24),128 if not QUICK_MODE else 64),14 if not QUICK_MODE else 7,COLORS["muted"]+(190,),False,"ra",1)

    def draw_caption(self, image: Image.Image, t: float):
        value = caption_at(t)
        if not value:
            return
        y0 = OUT_H - (250 if not QUICK_MODE else 126)
        overlay = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            (44 if not QUICK_MODE else 22, y0, OUT_W-(44 if not QUICK_MODE else 22), y0+(132 if not QUICK_MODE else 68)),
            radius=24 if not QUICK_MODE else 12,
            fill=(2,7,18,180),
            outline=COLORS["cyan"]+(60,),
            width=1,
        )
        image.alpha_composite(overlay)
        draw_wrapped_text(image,value,(68 if not QUICK_MODE else 34,y0+(28 if not QUICK_MODE else 14)),OUT_W-(136 if not QUICK_MODE else 68),28 if not QUICK_MODE else 14,COLORS["white"]+(245,))

    def draw_hud(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        for x,y,length,a,ph in self.hud:
            pulse = 0.5+0.5*math.sin(t*1.9+ph)
            if pulse < 0.74:
                continue
            yy = (y+t*7)%OUT_H
            draw.line((x,yy,x+length,yy), fill=COLORS["cyan"]+(int(a*pulse),), width=1)
        scan_y = int((t*140)%(OUT_H+180))-90
        draw.rectangle((0,scan_y,OUT_W,scan_y+(44 if not QUICK_MODE else 22)), fill=COLORS["cyan"]+(6,))
        image.alpha_composite(overlay)

    # scenes
    def scene_intro(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[0]
        local = smoothstep((t-shot["start"])/max(shot["end"]-shot["start"],1e-9))
        self.draw_curved_earth(image,t,1.04)
        self.draw_clouds(image,t,0.55)
        self.draw_jet_ribbon(image,t,amplitude=0.70+0.18*local,alpha=240,width_scale=1.18)
        self.panel(image,(int(OUT_W*0.09),int(OUT_H*0.67),int(OUT_W*0.91),int(OUT_H*0.81)),164)
        draw_text(image,"A FAST-MOVING CURRENT OF AIR CIRCLES THE PLANET",(OUT_W//2,int(OUT_H*0.719)),22 if not QUICK_MODE else 11,COLORS["white"]+(245,),True,"ma",1)
        draw_text(image,"high above the weather we feel at the ground",(OUT_W//2,int(OUT_H*0.765)),16 if not QUICK_MODE else 8,COLORS["cyan"]+(225,),False,"ma",1)
        self.draw_corner_label(image,"1 // THE RIVER ABOVE US")

    def scene_altitude(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[1]
        local = smoothstep((t-shot["start"])/max(shot["end"]-shot["start"],1e-9))
        self.draw_altitude_diagram(image,t,local)
        self.draw_clouds(image,t,0.28)
        self.draw_temperature_bands(image,t,0.70*local)
        self.panel(image,(int(OUT_W*0.09),int(OUT_H*0.70),int(OUT_W*0.91),int(OUT_H*0.83)),168)
        draw_text(image,"STRONG TEMPERATURE CONTRASTS HELP BUILD THE JET",(OUT_W//2,int(OUT_H*0.744)),20 if not QUICK_MODE else 10,COLORS["white"]+(245,),True,"ma",1)
        draw_text(image,"Earth's rotation helps turn that contrast into fast west-to-east flow",(OUT_W//2,int(OUT_H*0.788)),15 if not QUICK_MODE else 7,COLORS["gold"]+(225,),False,"ma",1)
        self.draw_corner_label(image,"2 // HIGH IN THE ATMOSPHERE")

    def scene_speed(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[2]
        local = smoothstep((t-shot["start"])/max(shot["end"]-shot["start"],1e-9))
        self.draw_curved_earth(image,t,1.08)
        self.draw_clouds(image,t,0.48)
        self.draw_jet_ribbon(image,t,amplitude=0.62,alpha=250,width_scale=1.05)

        # Speedometer-like arc.
        d = ImageDraw.Draw(image)
        cx,cy = int(OUT_W*0.79),int(OUT_H*0.61)
        rr = int(95*SCALE)
        d.arc((cx-rr,cy-rr,cx+rr,cy+rr),200,340,fill=COLORS["white"]+(90,),width=max(2,int(4*SCALE)))
        angle = math.radians(200+140*local)
        ex = cx+math.cos(angle)*rr*0.78
        ey = cy+math.sin(angle)*rr*0.78
        d.line((cx,cy,ex,ey),fill=COLORS["gold"]+(245,),width=max(2,int(5*SCALE)))
        draw_text(image,"VERY FAST",(cx,cy+int(35*SCALE)),16 if not QUICK_MODE else 8,COLORS["gold"]+(235,),True,"ma",1)

        self.panel(image,(int(OUT_W*0.08),int(OUT_H*0.70),int(OUT_W*0.92),int(OUT_H*0.83)),170)
        draw_text(image,"WINDS ALOFT CAN BE MUCH FASTER THAN WINDS BELOW",(OUT_W//2,int(OUT_H*0.744)),21 if not QUICK_MODE else 10,COLORS["white"]+(245,),True,"ma",1)
        draw_text(image,"a narrow ribbon of high-speed air stretches for thousands of kilometers",(OUT_W//2,int(OUT_H*0.788)),15 if not QUICK_MODE else 7,COLORS["cyan"]+(225,),False,"ma",1)
        self.draw_corner_label(image,"3 // THE FAST CURRENT")

    def scene_waves(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[3]
        local = smoothstep((t-shot["start"])/max(shot["end"]-shot["start"],1e-9))
        self.draw_curved_earth(image,t,1.08)
        self.draw_clouds(image,t,0.52)
        self.draw_jet_ribbon(image,t,amplitude=0.35+1.10*local,alpha=245,width_scale=1.02)
        # Highlight north/south excursions.
        draw_text(image,"NORTHWARD RIDGE",(int(OUT_W*0.72),int(OUT_H*0.24)),15 if not QUICK_MODE else 7,COLORS["warm"]+(225,),True,"ma",1)
        draw_text(image,"SOUTHWARD TROUGH",(int(OUT_W*0.28),int(OUT_H*0.53)),15 if not QUICK_MODE else 7,COLORS["cold"]+(225,),True,"ma",1)
        self.panel(image,(int(OUT_W*0.08),int(OUT_H*0.70),int(OUT_W*0.92),int(OUT_H*0.84)),172)
        draw_text(image,"THE JET STREAM MEANDERS LIKE A GIANT WAVE",(OUT_W//2,int(OUT_H*0.742)),22 if not QUICK_MODE else 11,COLORS["white"]+(245,),True,"ma",1)
        draw_text(image,"its path can bulge north, dip south, tighten, or stretch",(OUT_W//2,int(OUT_H*0.787)),16 if not QUICK_MODE else 8,COLORS["gold"]+(225,),False,"ma",1)
        draw_text(image,"and the whole pattern keeps moving",(OUT_W//2,int(OUT_H*0.820)),15 if not QUICK_MODE else 7,COLORS["muted"]+(215,),False,"ma",1)
        self.draw_corner_label(image,"4 // GIANT ATMOSPHERIC WAVES")

    def scene_weather(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[4]
        local = smoothstep((t-shot["start"])/max(shot["end"]-shot["start"],1e-9))
        self.draw_curved_earth(image,t,1.08)
        self.draw_temperature_bands(image,t,0.52)
        self.draw_clouds(image,t,0.60)
        self.draw_jet_ribbon(image,t,amplitude=1.12,alpha=245,width_scale=0.95)
        self.draw_weather_systems(image,t,local)
        self.panel(image,(int(OUT_W*0.07),int(OUT_H*0.70),int(OUT_W*0.93),int(OUT_H*0.85)),176)
        draw_text(image,"THE JET HELPS STEER WEATHER SYSTEMS",(OUT_W//2,int(OUT_H*0.741)),23 if not QUICK_MODE else 11,COLORS["white"]+(245,),True,"ma",1)
        draw_text(image,"its bends can guide storms and shift where warm and cold air travel",(OUT_W//2,int(OUT_H*0.787)),15 if not QUICK_MODE else 7,COLORS["cyan"]+(225,),False,"ma",1)
        draw_text(image,"the pattern changes from day to day",(OUT_W//2,int(OUT_H*0.823)),15 if not QUICK_MODE else 7,COLORS["gold"]+(220,),True,"ma",1)
        self.draw_corner_label(image,"5 // WEATHER STEERING")

    def scene_finale(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[5]
        local = smoothstep((t-shot["start"])/max(shot["end"]-shot["start"],1e-9))
        self.draw_curved_earth(image,t,1.06)
        self.draw_clouds(image,t,0.48)
        self.draw_jet_ribbon(image,t,amplitude=0.95,alpha=248,width_scale=1.12)
        # Secondary faint branch to imply multiple jets.
        self.draw_jet_ribbon(image,t,amplitude=0.55,alpha=100,width_scale=0.55,phase_shift=2.2)

        overlay = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        ImageDraw.Draw(overlay).rectangle((0,0,OUT_W,OUT_H),fill=(0,0,0,int(20+35*local)))
        image.alpha_composite(overlay)

        self.panel(image,(int(OUT_W*0.07),int(OUT_H*0.61),int(OUT_W*0.93),int(OUT_H*0.84)),190)
        draw_text(image,"THE JET STREAM",(OUT_W//2,int(OUT_H*0.659)),31 if not QUICK_MODE else 15,COLORS["cyan"]+(248,),True,"ma",1)
        draw_text(image,"A MOVING CURRENT OF FAST AIR",(OUT_W//2,int(OUT_H*0.708)),20 if not QUICK_MODE else 10,COLORS["white"]+(240,),True,"ma",1)
        draw_text(image,"high altitude • strong winds • giant waves",(OUT_W//2,int(OUT_H*0.758)),17 if not QUICK_MODE else 8,COLORS["gold"]+(228,),False,"ma",1)
        draw_text(image,"not one fixed river — a changing atmospheric pattern",(OUT_W//2,int(OUT_H*0.804)),15 if not QUICK_MODE else 7,COLORS["muted"]+(220,),False,"ma",1)
        self.draw_corner_label(image,"6 // A RIVER IN THE SKY")

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        name = shot["name"]
        image = self.background(t)

        if name == "intro":
            self.scene_intro(image,t)
        elif name == "altitude":
            self.scene_altitude(image,t)
        elif name == "speed":
            self.scene_speed(image,t)
        elif name == "waves":
            self.scene_waves(image,t)
        elif name == "weather":
            self.scene_weather(image,t)
        else:
            self.scene_finale(image,t)

        self.draw_source_hud(image)
        self.draw_title(image,t)
        self.draw_caption(image,t)
        self.draw_hud(image,t)

        array = np.asarray(image.convert("RGB"))
        array = apply_grade(array)
        array = np.clip(array.astype(np.float32)*VIGNETTE[...,None],0,255).astype(np.uint8)
        fade_in = smoothstep(t/(0.85 if not QUICK_MODE else 0.20))
        fade_out = 1.0-smoothstep((t-(DURATION-(1.10 if not QUICK_MODE else 0.25)))/(1.0 if not QUICK_MODE else 0.20))
        return np.clip(array.astype(np.float32)*fade_in*fade_out,0,255).astype(np.uint8)


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
            "Jet streams are narrow bands of strong winds high in the atmosphere.",
            "Strong horizontal temperature contrasts help support them.",
            "Earth's rotation shapes the large-scale flow.",
            "Jet streams commonly meander in large atmospheric waves.",
            "Their position helps steer weather systems and air masses.",
            "There is not one single permanent jet stream; multiple jets exist and vary over time.",
        ],
        "visual_warning": "The animation is cinematic and diagrammatic, not a numerical forecast or weather-model output.",
    }
    path = OUTPUT_ROOT/"science_and_render_summary.json"
    path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return path


def render_video(scene: JetStreamScene) -> Path:
    srt_path = OUTPUT_ROOT/f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS,srt_path)
    print("Subtitle sidecar:",srt_path.resolve())

    raw_video = OUTPUT_ROOT/f"{CONFIG['output_basename']}_raw.mp4"
    final_video = OUTPUT_ROOT/f"{CONFIG['output_basename']}_final.mp4"
    frame_count = int(round(DURATION*FPS))
    times = np.arange(frame_count)/FPS
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")

    with iio.get_writer(
        raw_video,
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times,desc="Rendering jet-stream short"):
            writer.append_data(scene.render_frame(float(t)))

    shutil.copyfile(raw_video,final_video)
    print("Final video:",final_video.resolve())
    return final_video


def main():
    print("Preparing jet-stream YouTube Short ...")
    print("Mode:","QUICK" if QUICK_MODE else ("4K" if FOUR_K else "FULL"))
    print("Canvas:",f"{OUT_W}x{OUT_H}")
    print("FPS:",FPS)
    print("Duration:",DURATION,"seconds")

    scene = JetStreamScene()
    summary_path = save_summary()

    preview_times = [
        min(1.0,DURATION*0.08),
        min(10.5,DURATION*0.22),
        min(21.5,DURATION*0.40),
        min(33.0,DURATION*0.58),
        min(44.0,DURATION*0.77),
        DURATION-(1.0 if not QUICK_MODE else 0.8),
    ]
    for preview_time in tqdm(preview_times,desc="Preview frames"):
        Image.fromarray(scene.render_frame(float(preview_time))).save(PREVIEW_DIR/f"preview_{int(preview_time*10):03d}.png")

    print("Summary:",summary_path.resolve())
    render_video(scene)
    print("Output directory:",OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-",path.name)


if __name__ == "__main__":
    main()
