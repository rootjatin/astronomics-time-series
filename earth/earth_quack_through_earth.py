from __future__ import annotations

"""
output : https://www.youtube.com/shorts/rFOogD8S_ys

Watch Earthquakes Travel Across the Planet — cinematic YouTube Short renderer
Creates a vertical 1080x1920 science short showing how seismic energy spreads
from an earthquake through and around Earth.

This is a diagrammatic educational animation, NOT a numerical seismology model
and NOT a live earthquake feed. It visualizes these scientifically important
ideas:

- Earthquakes begin when a fault ruptures and releases stored elastic energy.
- P waves are compressional body waves and travel fastest through Earth.
- S waves are shear body waves and are slower than P waves.
- S waves do not propagate through Earth's liquid outer core.
- Surface waves travel along Earth's exterior and often produce strong,
  long-duration shaking near the surface.
- Seismometers can distinguish arrivals because P, S, and surface waves reach
  stations at different times.

No external data or internet connection is required. All stars, particles,
wave paths, and terrain marks are deterministic so repeated renders match.

Recommended install
-------------------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview render
--------------------
    EARTHQUAKE_SHORT_QUICK=1 python watch_earthquakes_travel_across_the_planet.py

Full render
-----------
    python watch_earthquakes_travel_across_the_planet.py

Optional 4K vertical render
---------------------------
    EARTHQUAKE_SHORT_4K=1 python watch_earthquakes_travel_across_the_planet.py

Outputs
-------
- MP4 video
- SRT subtitles
- PNG preview frames
- JSON science/production summary
"""

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("EARTHQUAKE_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("EARTHQUAKE_SHORT_4K", "0") == "1" and not QUICK_MODE

OUTPUT_ROOT = Path("watch_earthquakes_travel_across_the_planet_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else (2160 if FOUR_K else 1080),
    "video_height": 960 if QUICK_MODE else (3840 if FOUR_K else 1920),
    "fps": 7 if QUICK_MODE else (30 if FOUR_K else 24),
    "duration_s": 12.0 if QUICK_MODE else 58.0,
    "output_basename": "watch_earthquakes_travel_across_the_planet",
    "title": "WATCH EARTHQUAKES TRAVEL ACROSS THE PLANET",
    "subtitle": "P waves // S waves // surface waves",
    "background_stars": 150 if QUICK_MODE else 420,
    "dust_particles": 60 if QUICK_MODE else 160,
    "contrast": 1.08,
    "saturation": 1.06,
    "vignette": 0.24,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0

COLORS = {
    "space": (2, 5, 15),
    "white": (244, 249, 255),
    "muted": (166, 192, 213),
    "cyan": (72, 226, 255),
    "blue": (73, 132, 255),
    "violet": (177, 111, 255),
    "gold": (255, 199, 92),
    "orange": (255, 128, 63),
    "red": (255, 76, 78),
    "green": (104, 242, 173),
    "earth_blue": (26, 102, 184),
    "ocean": (14, 62, 137),
    "land": (80, 165, 106),
    "crust": (151, 99, 63),
    "mantle": (168, 71, 44),
    "outer_core": (237, 136, 54),
    "inner_core": (255, 213, 118),
    "night": (4, 17, 40),
}

FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.4, 7.2, "An earthquake begins when a fault suddenly slips, releasing stored elastic energy into the planet."),
    (7.3, 17.0, "The fastest seismic signals are P waves. They squeeze and stretch material and can travel through both solid rock and liquid."),
    (17.1, 27.8, "S waves arrive later. Their side-to-side shear motion travels through solid rock, but it cannot pass through Earth's liquid outer core."),
    (27.9, 38.8, "That difference creates a global pattern of arrivals and shadow zones. Seismologists use those patterns to probe Earth's deep interior."),
    (38.9, 49.2, "Near the surface, seismic energy also spreads around the planet as surface waves, often producing strong, long-lasting shaking."),
    (49.3, 57.4, "A seismometer records the sequence: P first, then S, then the slower surface waves. One earthquake can make the whole planet ring."),
]

if QUICK_MODE:
    _caption_scale = float(CONFIG["duration_s"]) / 58.0
    CAPTIONS = [(a * _caption_scale, b * _caption_scale, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

if QUICK_MODE:
    SHOT_PLAN = [
        {"name": "rupture", "start": 0.0, "end": 1.55},
        {"name": "p_wave", "start": 1.55, "end": 3.55},
        {"name": "s_wave", "start": 3.55, "end": 5.80},
        {"name": "shadow", "start": 5.80, "end": 8.00},
        {"name": "surface", "start": 8.00, "end": 10.25},
        {"name": "seismogram", "start": 10.25, "end": 12.0},
    ]
else:
    SHOT_PLAN = [
        {"name": "rupture", "start": 0.0, "end": 7.7},
        {"name": "p_wave", "start": 7.7, "end": 18.0},
        {"name": "s_wave", "start": 18.0, "end": 28.9},
        {"name": "shadow", "start": 28.9, "end": 39.6},
        {"name": "surface", "start": 39.6, "end": 50.1},
        {"name": "seismogram", "start": 50.1, "end": 58.0},
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
    for start, end, value in CAPTIONS:
        if start <= t < end:
            return value
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
            return ImageFont.truetype(candidate, size=max(7, int(size * SCALE)))
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    value: str,
    xy: Tuple[int, int],
    size: int = 28,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    stroke: int = 2,
    anchor: str = "la",
):
    ImageDraw.Draw(image).text(
        xy,
        value,
        font=get_font(size, bold=bold),
        fill=fill,
        anchor=anchor,
        stroke_width=max(1, int(stroke * SCALE)),
        stroke_fill=(0, 0, 0, min(225, fill[3] if len(fill) > 3 else 225)),
    )


def draw_wrapped_text(
    image: Image.Image,
    value: str,
    box: Tuple[int, int, int, int],
    size: int = 28,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    spacing: int = 6,
    align: str = "center",
):
    x0, y0, x1, _ = box
    draw = ImageDraw.Draw(image)
    fnt = get_font(size, bold=bold)
    words = value.split()
    lines: List[str] = []
    current = ""
    max_width = x1 - x0

    for word in words:
        candidate = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), candidate, font=fnt, stroke_width=max(1, int(SCALE)))
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    y = y0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt, stroke_width=max(1, int(SCALE)))
        height = bbox[3] - bbox[1]
        x = (x0 + x1) // 2 if align == "center" else x0
        anchor = "ma" if align == "center" else "la"
        draw.text(
            (x, y), line, font=fnt, fill=fill, anchor=anchor,
            stroke_width=max(1, int(2 * SCALE)), stroke_fill=(0, 0, 0, 220),
        )
        y += height + int(spacing * SCALE)


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.8, 0.0, 1.0).astype(np.float32)


def apply_grade(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array)
    image = ImageEnhance.Contrast(image).enhance(float(CONFIG["contrast"]))
    image = ImageEnhance.Color(image).enhance(float(CONFIG["saturation"]))
    return np.asarray(image)


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000.0))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for index, (start, end, value) in enumerate(captions, start=1):
        lines.extend([str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", value, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class EarthquakeScene:
    def __init__(self):
        self.center = (int(OUT_W * 0.50), int(OUT_H * 0.405))
        self.radius = int(250 * SCALE)
        self.stars = self._make_stars(int(CONFIG["background_stars"]), 1906)
        self.dust = self._make_dust(int(CONFIG["dust_particles"]), 2011)
        self.epicenter_angle = math.radians(215)
        self.station_angles = [math.radians(v) for v in (255, 320, 35, 78, 122)]

    @staticmethod
    def _make_stars(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.35, 1.9) * SCALE),
                "a": float(rng.uniform(25, 125)),
                "phase": float(rng.uniform(0.0, 2 * math.pi)),
            }
            for _ in range(count)
        ]

    @staticmethod
    def _make_dust(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.7, 3.2) * SCALE),
                "phase": float(rng.uniform(0.0, 2 * math.pi)),
                "a": float(rng.uniform(15, 60)),
            }
            for _ in range(count)
        ]

    def panel(self, image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 172):
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.rounded_rectangle(
            box,
            radius=max(8, int(24 * SCALE)),
            fill=(2, 7, 18, alpha),
            outline=COLORS["cyan"] + (58,),
            width=max(1, int(2 * SCALE)),
        )
        image.alpha_composite(layer)

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, COLORS["space"] + (255,))
        draw = ImageDraw.Draw(image)
        for star in self.stars:
            alpha = int(star["a"] * (0.75 + 0.25 * math.sin(t * 1.25 + star["phase"])))
            r = star["r"]
            draw.ellipse(
                (star["x"] - r, star["y"] - r, star["x"] + r, star["y"] + r),
                fill=COLORS["white"] + (alpha,),
            )

        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        cx, cy = self.center
        for rr, alpha in [(470, 7), (360, 12), (300, 16)]:
            r = int(rr * SCALE)
            hd.ellipse((cx-r, cy-r, cx+r, cy+r), fill=COLORS["blue"] + (alpha,))
        haze = haze.filter(ImageFilter.GaussianBlur(max(10, int(55 * SCALE))))
        image.alpha_composite(haze)

        dust_layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        dd = ImageDraw.Draw(dust_layer)
        for item in self.dust:
            x = (item["x"] + t * 5.0 * SCALE) % OUT_W
            y = item["y"] + math.sin(t * 0.35 + item["phase"]) * 8 * SCALE
            r = item["r"]
            dd.ellipse((x-r, y-r, x+r, y+r), fill=COLORS["cyan"] + (int(item["a"]),))
        image.alpha_composite(dust_layer.filter(ImageFilter.GaussianBlur(max(1, int(2 * SCALE)))))
        return image

    def earth_surface_point(self, angle: float, radius_factor: float = 1.0) -> Tuple[float, float]:
        cx, cy = self.center
        r = self.radius * radius_factor
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    def draw_earth(self, image: Image.Image, t: float, cutaway: bool = False, alpha: int = 255):
        cx, cy = self.center
        r = self.radius

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for extra, a in [(46, 10), (28, 22), (13, 48)]:
            rr = r + int(extra * SCALE)
            gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=COLORS["cyan"] + (a,))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4, int(17 * SCALE)))))

        earth = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(earth)
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=COLORS["earth_blue"] + (alpha,))

        continents = [
            [(-0.64,-0.28),(-0.38,-0.52),(-0.08,-0.45),(0.02,-0.18),(-0.22,-0.03),(-0.48,-0.08)],
            [(0.08,-0.34),(0.39,-0.40),(0.62,-0.18),(0.50,0.10),(0.18,0.20),(-0.02,0.02)],
            [(-0.12,0.16),(0.12,0.12),(0.23,0.42),(0.02,0.70),(-0.22,0.49),(-0.29,0.28)],
        ]
        wobble = 0.045 * math.sin(t * 0.28)
        for poly in continents:
            pts = []
            for px, py in poly:
                x = px * math.cos(wobble) - py * math.sin(wobble)
                y = px * math.sin(wobble) + py * math.cos(wobble)
                pts.append((cx + x * r, cy + y * r))
            draw.polygon(pts, fill=COLORS["land"] + (min(alpha, 235),))

        shade = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        sd = ImageDraw.Draw(shade)
        sd.pieslice((cx-r, cy-r, cx+r, cy+r), 285, 105, fill=COLORS["night"] + (112,))
        earth.alpha_composite(shade.filter(ImageFilter.GaussianBlur(max(1, int(3 * SCALE)))))
        draw.arc((cx-r, cy-r, cx+r, cy+r), 105, 285, fill=COLORS["cyan"] + (150,), width=max(1, int(4 * SCALE)))
        image.alpha_composite(earth)

        if cutaway:
            self.draw_cutaway(image)

    def draw_cutaway(self, image: Image.Image):
        cx, cy = self.center
        r = self.radius
        layer = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        draw = ImageDraw.Draw(layer)

        # Right-side interior wedge. Relative radii are for visualization only.
        wedge_box = (cx-r, cy-r, cx+r, cy+r)
        draw.pieslice(wedge_box, start=-55, end=55, fill=COLORS["mantle"] + (235,))
        rr = int(r * 0.55)
        draw.pieslice((cx-rr,cy-rr,cx+rr,cy+rr), start=-55, end=55, fill=COLORS["outer_core"] + (245,))
        ir = int(r * 0.20)
        draw.pieslice((cx-ir,cy-ir,cx+ir,cy+ir), start=-55, end=55, fill=COLORS["inner_core"] + (250,))
        draw.line((cx,cy,cx+r*math.cos(math.radians(-55)),cy+r*math.sin(math.radians(-55))), fill=COLORS["white"]+(90,), width=max(1,int(2*SCALE)))
        draw.line((cx,cy,cx+r*math.cos(math.radians(55)),cy+r*math.sin(math.radians(55))), fill=COLORS["white"]+(90,), width=max(1,int(2*SCALE)))
        image.alpha_composite(layer)

        draw_text(image, "MANTLE", (int(cx+r*0.55), int(cy-r*0.40)), 14, COLORS["white"]+(215,), True, 1, "ma")
        draw_text(image, "LIQUID OUTER CORE", (int(cx+r*0.44), int(cy+2*SCALE)), 13, COLORS["gold"]+(240,), True, 1, "ma")
        draw_text(image, "INNER CORE", (int(cx+r*0.20), int(cy+r*0.19)), 11, COLORS["white"]+(210,), True, 1, "ma")

    def draw_epicenter(self, image: Image.Image, t: float, strength: float = 1.0):
        ex, ey = self.earth_surface_point(self.epicenter_angle, 0.98)
        layer = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        draw = ImageDraw.Draw(layer)
        pulse = 0.5 + 0.5 * math.sin(t * 8.0)
        for extra, a in [(24, 24), (14, 54), (5, 240)]:
            rr = (extra * (0.8 + 0.25*pulse)) * SCALE
            draw.ellipse((ex-rr,ey-rr,ex+rr,ey+rr), fill=COLORS["red"] + (int(a*strength),))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1,int(5*SCALE)))))
        image.alpha_composite(layer)
        draw_text(image, "EPICENTER", (int(ex-22*SCALE), int(ey+48*SCALE)), 14, COLORS["red"]+(240,), True, 1, "ra")

    def draw_fault_rupture(self, image: Image.Image, local: float, t: float):
        ex, ey = self.earth_surface_point(self.epicenter_angle, 0.93)
        draw = ImageDraw.Draw(image)
        length = 120 * SCALE * (0.35 + 0.65 * local)
        angle = self.epicenter_angle + math.pi/2
        dx, dy = math.cos(angle)*length/2, math.sin(angle)*length/2
        offset = 7 * SCALE * math.sin(t*10) * local
        draw.line((ex-dx-offset, ey-dy, ex+dx-offset, ey+dy), fill=COLORS["gold"]+(235,), width=max(2,int(6*SCALE)))
        draw.line((ex-dx+offset, ey-dy, ex+dx+offset, ey+dy), fill=COLORS["red"]+(220,), width=max(2,int(4*SCALE)))
        self.draw_epicenter(image,t,local)

    def draw_radial_body_waves(self, image: Image.Image, local: float, wave: str):
        cx, cy = self.center
        ex, ey = self.earth_surface_point(self.epicenter_angle, 0.92)
        layer = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        draw = ImageDraw.Draw(layer)

        if wave == "P":
            colour = COLORS["gold"]
            speed = 1.0
            max_radius = self.radius * 1.62
            line_width = max(1, int(5*SCALE))
        else:
            colour = COLORS["cyan"]
            speed = 0.78
            max_radius = self.radius * 1.48
            line_width = max(1, int(5*SCALE))

        phase = local * speed
        for k in range(5):
            p = phase - k*0.115
            if p <= 0:
                continue
            rr = p * max_radius
            alpha = int(215 * clamp(1.0 - p*0.45) * (1.0-k*0.10))
            draw.ellipse((ex-rr,ey-rr,ex+rr,ey+rr), outline=colour+(alpha,), width=line_width)

        # conceptual ray fan
        start_angle = self.epicenter_angle
        if wave == "P":
            ray_targets = np.linspace(-2.55, 0.55, 11)
        else:
            ray_targets = np.linspace(-2.35, 0.15, 9)
        for idx, target_angle in enumerate(ray_targets):
            end = self.earth_surface_point(float(target_angle), 0.98)
            control_pull = 0.30 if wave == "P" else 0.40
            mx = lerp(ex, cx, control_pull)
            my = lerp(ey, cy, control_pull)
            pts = []
            count = 70
            upto = max(2, int(count * clamp(local*1.12 - idx*0.018)))
            for j in range(upto):
                u = j/(count-1)
                v = 1-u
                x = v*v*ex + 2*v*u*mx + u*u*end[0]
                y = v*v*ey + 2*v*u*my + u*u*end[1]
                # S waves stop before traversing the liquid outer-core region.
                if wave == "S" and math.hypot(x-cx,y-cy) < self.radius*0.56:
                    break
                pts.append((x,y))
            if len(pts) >= 2:
                draw.line(pts, fill=colour+(115,), width=max(1,int(2*SCALE)))

        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1,int(3*SCALE)))))
        image.alpha_composite(layer)

    def draw_s_wave_core_block(self, image: Image.Image, t: float, local: float):
        cx,cy=self.center
        rr=int(self.radius*0.57)
        layer=Image.new("RGBA",OUT_SIZE,(0,0,0,0))
        draw=ImageDraw.Draw(layer)
        pulse=0.5+0.5*math.sin(t*4.0)
        draw.arc((cx-rr,cy-rr,cx+rr,cy+rr),-58,58,fill=COLORS["red"]+(int((110+80*pulse)*local),),width=max(2,int(7*SCALE)))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1,int(6*SCALE)))))
        image.alpha_composite(layer)
        draw_text(image,"S WAVES STOP IN LIQUID",(int(cx+self.radius*0.38),int(cy+self.radius*0.62)),14,COLORS["red"]+(235,),True,1,"ma")

    def draw_shadow_zones(self, image: Image.Image, local: float):
        cx,cy=self.center
        r=self.radius
        layer=Image.new("RGBA",OUT_SIZE,(0,0,0,0))
        draw=ImageDraw.Draw(layer)
        # Stylized angular wedges opposite the epicenter.
        alpha=int(72*local)
        draw.pieslice((cx-r,cy-r,cx+r,cy+r), start=-58, end=-18, fill=COLORS["violet"]+(alpha,))
        draw.pieslice((cx-r,cy-r,cx+r,cy+r), start=18, end=58, fill=COLORS["violet"]+(alpha,))
        draw.arc((cx-r,cy-r,cx+r,cy+r), -58,-18, fill=COLORS["violet"]+(int(190*local),), width=max(1,int(4*SCALE)))
        draw.arc((cx-r,cy-r,cx+r,cy+r), 18,58, fill=COLORS["violet"]+(int(190*local),), width=max(1,int(4*SCALE)))
        image.alpha_composite(layer)
        draw_text(image,"BODY-WAVE SHADOW REGIONS",(cx,int(cy+r+64*SCALE)),17,COLORS["violet"]+(235,),True,1,"ma")

    def draw_surface_waves(self, image: Image.Image, t: float, local: float):
        cx,cy=self.center
        r=self.radius*1.03
        layer=Image.new("RGBA",OUT_SIZE,(0,0,0,0))
        draw=ImageDraw.Draw(layer)
        start=self.epicenter_angle

        for direction,colour in [(-1,COLORS["orange"]),(1,COLORS["cyan"])]:
            travel=local*math.pi*2.15
            pts=[]
            count=150
            for i in range(count):
                u=i/(count-1)
                if u*math.pi*2.15 > travel:
                    break
                angle=start+direction*u*math.pi*2.15
                wobble=7*SCALE*math.sin(i*0.65-t*5.0)*(1-u*0.45)
                rr=r+wobble
                pts.append((cx+rr*math.cos(angle),cy+rr*math.sin(angle)))
            if len(pts)>2:
                draw.line(pts,fill=colour+(215,),width=max(2,int(6*SCALE)))
                ex,ey=pts[-1]
                dot=6*SCALE
                draw.ellipse((ex-dot,ey-dot,ex+dot,ey+dot),fill=COLORS["white"]+(240,))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1,int(4*SCALE)))))
        image.alpha_composite(layer)

    def draw_stations(self, image: Image.Image, active_fraction: float):
        draw=ImageDraw.Draw(image)
        for index,angle in enumerate(self.station_angles):
            x,y=self.earth_surface_point(angle,1.06)
            active=active_fraction >= (index+1)/len(self.station_angles)*0.82
            colour=COLORS["green"] if active else COLORS["muted"]
            rr=7*SCALE
            draw.ellipse((x-rr,y-rr,x+rr,y+rr),fill=colour+(240 if active else 130,))
            draw.line((x,y,x,y-17*SCALE),fill=colour+(220,),width=max(1,int(2*SCALE)))

    def draw_seismogram(self, image: Image.Image, local: float, t: float):
        x0,x1=int(OUT_W*.09),int(OUT_W*.91)
        y0,y1=int(OUT_H*.245),int(OUT_H*.655)
        self.panel(image,(x0,y0,x1,y1),188)
        draw=ImageDraw.Draw(image)
        mid=(y0+y1)//2
        draw.line((x0+34*SCALE,mid,x1-30*SCALE,mid),fill=COLORS["white"]+(45,),width=max(1,int(SCALE)))

        gx0=x0+42*SCALE; gx1=x1-34*SCALE
        progress=smoothstep(local)
        samples=720 if not QUICK_MODE else 280
        xs=np.linspace(gx0,gx1,samples)
        amp=np.zeros(samples,np.float64)
        p1=int(samples*.17); s1=int(samples*.42); surf=int(samples*.67)
        rng=np.random.default_rng(1889)
        amp+=rng.normal(0,.018,samples)
        for i in range(samples):
            if i>=p1:
                q=(i-p1)/samples
                amp[i]+=0.12*math.sin(i*.68)*math.exp(-q*3.0)
            if i>=s1:
                q=(i-s1)/samples
                amp[i]+=0.28*math.sin(i*.43+0.7)*math.exp(-q*2.5)
            if i>=surf:
                q=(i-surf)/samples
                amp[i]+=0.50*math.sin(i*.19+0.4)*math.exp(-q*2.0)
                amp[i]+=0.22*math.sin(i*.08)*math.exp(-q*1.4)
        upto=max(2,int(samples*progress))
        pts=[(float(xs[i]),float(mid-amp[i]*(y1-y0)*.55)) for i in range(upto)]
        draw.line(pts,fill=COLORS["white"]+(240,),width=max(1,int(3*SCALE)))

        for pos,label,colour in [(p1,"P",COLORS["gold"]),(s1,"S",COLORS["cyan"]),(surf,"SURFACE",COLORS["orange"])]:
            x=float(xs[pos])
            draw.line((x,y0+34*SCALE,x,y1-34*SCALE),fill=colour+(70,),width=max(1,int(2*SCALE)))
            draw_text(image,label,(int(x),int(y0+40*SCALE)),14,colour+(240,),True,1,"ma")

        cursor_x=lerp(gx0,gx1,progress)
        draw.line((cursor_x,y0+24*SCALE,cursor_x,y1-24*SCALE),fill=COLORS["green"]+(180,),width=max(1,int(2*SCALE)))
        draw_text(image,"SEISMOMETER RECORD",(OUT_W//2,int(y0+78*SCALE)),18,COLORS["muted"]+(220,),True,1,"ma")

    def draw_titles(self, image: Image.Image, t: float, shot_name: str):
        intro_end=5.3 if not QUICK_MODE else 1.0
        if t<intro_end:
            fade=smoothstep(t/(0.65 if not QUICK_MODE else 0.13))
            draw_text(image,"WATCH EARTHQUAKES",(OUT_W//2,int(OUT_H*.072)),36,COLORS["white"]+(int(245*fade),),True,2,"ma")
            draw_text(image,"TRAVEL ACROSS",(OUT_W//2,int(OUT_H*.108)),48,COLORS["cyan"]+(int(250*fade),),True,2,"ma")
            draw_text(image,"THE PLANET",(OUT_W//2,int(OUT_H*.150)),43,COLORS["gold"]+(int(245*fade),),True,2,"ma")

        labels={
            "rupture":"1 // THE RUPTURE",
            "p_wave":"2 // P WAVES — FASTEST",
            "s_wave":"3 // S WAVES — SOLIDS ONLY",
            "shadow":"4 // EARTH'S INTERIOR REVEALED",
            "surface":"5 // SURFACE WAVES",
            "seismogram":"6 // WHAT A STATION RECORDS",
        }
        if t>(5.0 if not QUICK_MODE else .95):
            draw_text(image,labels[shot_name],(52 if not QUICK_MODE else 26,58 if not QUICK_MODE else 29),18,COLORS["muted"]+(205,),True,1)

    def draw_caption(self, image: Image.Image, t: float):
        value=caption_at(t)
        if not value:
            return
        y0=int(OUT_H*.865)
        self.panel(image,(int(OUT_W*.055),y0,int(OUT_W*.945),int(OUT_H*.965)),158)
        draw_wrapped_text(image,value,(int(OUT_W*.10),int(OUT_H*.885),int(OUT_W*.90),int(OUT_H*.952)),15,COLORS["white"]+(245,),False,4,"center")

    def draw_hud(self, image: Image.Image, t: float):
        draw_text(image,"SCIENCE DIAGRAM // NOT TO SCALE",(OUT_W-int(44*SCALE),int(70*SCALE)),15,COLORS["gold"]+(230,),True,1,"ra")
        draw_text(image,"SEISMIC PATHS → CONCEPTUAL",(OUT_W-int(44*SCALE),int(99*SCALE)),14,COLORS["muted"]+(200,),False,1,"ra")
        draw=ImageDraw.Draw(image)
        offset=int((t*38)%8)
        for y in range(offset,OUT_H,12):
            draw.line((0,y,OUT_W,y),fill=(100,190,235,3),width=1)
        scan_y=int((t*170)%(OUT_H+240))-120
        draw.rectangle((0,scan_y,OUT_W,scan_y+44*SCALE),fill=(70,210,245,4))

    # ----- shot composition ---------------------------------------------------

    def scene_rupture(self, image: Image.Image, t: float, local: float):
        self.draw_earth(image,t,cutaway=False)
        self.draw_fault_rupture(image,local,t)
        self.panel(image,(int(OUT_W*.10),int(OUT_H*.695),int(OUT_W*.90),int(OUT_H*.815)),166)
        draw_text(image,"A FAULT SUDDENLY SLIPS",(OUT_W//2,int(OUT_H*.738)),23,COLORS["red"]+(245,),True,1,"ma")
        draw_text(image,"stored elastic energy launches seismic waves",(OUT_W//2,int(OUT_H*.780)),16,COLORS["white"]+(225,),False,1,"ma")

    def scene_p_wave(self, image: Image.Image, t: float, local: float):
        self.draw_earth(image,t,cutaway=True)
        self.draw_epicenter(image,t,1.0)
        self.draw_radial_body_waves(image,local,"P")
        self.panel(image,(int(OUT_W*.09),int(OUT_H*.695),int(OUT_W*.91),int(OUT_H*.825)),168)
        draw_text(image,"P WAVES RACE THROUGH THE PLANET",(OUT_W//2,int(OUT_H*.738)),22,COLORS["gold"]+(245,),True,1,"ma")
        draw_text(image,"compression waves • fastest arrivals • solids + liquids",(OUT_W//2,int(OUT_H*.780)),15,COLORS["white"]+(225,),False,1,"ma")

    def scene_s_wave(self, image: Image.Image, t: float, local: float):
        self.draw_earth(image,t,cutaway=True)
        self.draw_epicenter(image,t,1.0)
        self.draw_radial_body_waves(image,local,"S")
        self.draw_s_wave_core_block(image,t,local)
        self.panel(image,(int(OUT_W*.09),int(OUT_H*.695),int(OUT_W*.91),int(OUT_H*.825)),170)
        draw_text(image,"S WAVES CANNOT CROSS THE LIQUID OUTER CORE",(OUT_W//2,int(OUT_H*.736)),20,COLORS["cyan"]+(245,),True,1,"ma")
        draw_text(image,"shear waves • slower than P waves • travel through solids",(OUT_W//2,int(OUT_H*.780)),15,COLORS["white"]+(225,),False,1,"ma")

    def scene_shadow(self, image: Image.Image, t: float, local: float):
        self.draw_earth(image,t,cutaway=True)
        self.draw_epicenter(image,t,1.0)
        self.draw_radial_body_waves(image,min(1.0,local*1.4),"P")
        self.draw_radial_body_waves(image,min(1.0,local*1.2),"S")
        self.draw_shadow_zones(image,local)
        self.draw_stations(image,local)
        self.panel(image,(int(OUT_W*.09),int(OUT_H*.715),int(OUT_W*.91),int(OUT_H*.838)),174)
        draw_text(image,"WAVE ARRIVALS MAP EARTH'S HIDDEN INTERIOR",(OUT_W//2,int(OUT_H*.755)),20,COLORS["violet"]+(245,),True,1,"ma")
        draw_text(image,"different paths and missing arrivals reveal deep layers",(OUT_W//2,int(OUT_H*.797)),15,COLORS["white"]+(225,),False,1,"ma")

    def scene_surface(self, image: Image.Image, t: float, local: float):
        self.draw_earth(image,t,cutaway=False)
        self.draw_epicenter(image,t,1.0)
        self.draw_surface_waves(image,t,local)
        self.draw_stations(image,local)
        self.panel(image,(int(OUT_W*.09),int(OUT_H*.695),int(OUT_W*.91),int(OUT_H*.825)),170)
        draw_text(image,"SURFACE WAVES WRAP AROUND EARTH",(OUT_W//2,int(OUT_H*.738)),22,COLORS["orange"]+(245,),True,1,"ma")
        draw_text(image,"slower • concentrated near the surface • often strong shaking",(OUT_W//2,int(OUT_H*.780)),15,COLORS["white"]+(225,),False,1,"ma")

    def scene_seismogram(self, image: Image.Image, t: float, local: float):
        # faint globe behind the instrument trace
        self.draw_earth(image,t,cutaway=False,alpha=82)
        self.draw_seismogram(image,local,t)
        self.panel(image,(int(OUT_W*.08),int(OUT_H*.700),int(OUT_W*.92),int(OUT_H*.835)),186)
        draw_text(image,"P FIRST  →  S NEXT  →  SURFACE WAVES LAST",(OUT_W//2,int(OUT_H*.744)),20,COLORS["white"]+(248,),True,1,"ma")
        draw_text(image,"one earthquake can make the entire planet vibrate",(OUT_W//2,int(OUT_H*.786)),16,COLORS["gold"]+(235,),False,1,"ma")

    def render_frame(self, t: float) -> np.ndarray:
        shot=get_shot(t)
        local=smoothstep((t-shot["start"])/max(shot["end"]-shot["start"],1e-9))
        image=self.background(t)
        name=shot["name"]

        if name=="rupture":
            self.scene_rupture(image,t,local)
        elif name=="p_wave":
            self.scene_p_wave(image,t,local)
        elif name=="s_wave":
            self.scene_s_wave(image,t,local)
        elif name=="shadow":
            self.scene_shadow(image,t,local)
        elif name=="surface":
            self.scene_surface(image,t,local)
        else:
            self.scene_seismogram(image,t,local)

        self.draw_titles(image,t,name)
        self.draw_caption(image,t)
        self.draw_hud(image,t)

        array=np.asarray(image.convert("RGB"))
        array=apply_grade(array)
        array=np.clip(array.astype(np.float32)*VIGNETTE[...,None],0,255).astype(np.uint8)
        fade_in=smoothstep(t/0.85)
        fade_out=1.0-smoothstep((t-(float(CONFIG["duration_s"])-1.0))/.9)
        return np.clip(array.astype(np.float32)*fade_in*fade_out,0,255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def save_summary() -> Path:
    summary={
        "title": CONFIG["title"],
        "format": f"{OUT_W}x{OUT_H} vertical MP4",
        "fps": CONFIG["fps"],
        "duration_s": CONFIG["duration_s"],
        "quick_mode": QUICK_MODE,
        "four_k": FOUR_K,
        "science_points": [
            "Earthquakes release elastic energy when faults rupture.",
            "P waves are compressional body waves and usually arrive first.",
            "P waves can propagate through solids and liquids.",
            "S waves are shear body waves and cannot propagate through Earth's liquid outer core.",
            "Body-wave arrival patterns and shadow zones help constrain Earth's internal structure.",
            "Surface waves travel around Earth's exterior and can produce strong, long-duration shaking.",
            "Seismometers record different seismic phases at different arrival times.",
        ],
        "visual_warning": "All wave paths, speeds, layer sizes, station positions, amplitudes, and timescales are diagrammatic and not to scale.",
    }
    path=OUTPUT_ROOT/"science_and_render_summary.json"
    path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return path


def render_video(scene: EarthquakeScene) -> Path:
    srt_path=OUTPUT_ROOT/f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS,srt_path)
    print("Subtitle sidecar:",srt_path.resolve())

    raw_video=OUTPUT_ROOT/f"{CONFIG['output_basename']}_raw.mp4"
    final_video=OUTPUT_ROOT/f"{CONFIG['output_basename']}_final.mp4"
    frame_count=int(round(float(CONFIG["duration_s"])*int(CONFIG["fps"])))
    times=np.arange(frame_count)/int(CONFIG["fps"])
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")

    with iio.get_writer(
        raw_video,
        fps=int(CONFIG["fps"]),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times,desc="Rendering earthquake short"):
            writer.append_data(scene.render_frame(float(t)))

    shutil.copyfile(raw_video,final_video)
    try:
        raw_video.unlink()
    except OSError:
        pass
    print("Final video:",final_video.resolve())
    return final_video


def main():
    print("Preparing 'Watch Earthquakes Travel Across the Planet' YouTube Short ...")
    print("Mode:","QUICK" if QUICK_MODE else ("4K" if FOUR_K else "FULL"))
    print("Canvas:",f"{OUT_W}x{OUT_H}")
    print("FPS:",CONFIG["fps"])
    print("Duration:",CONFIG["duration_s"],"seconds")

    scene=EarthquakeScene()
    summary_path=save_summary()

    preview_times=[
        min(1.0,float(CONFIG["duration_s"])*.08),
        float(CONFIG["duration_s"])*.20,
        float(CONFIG["duration_s"])*.39,
        float(CONFIG["duration_s"])*.59,
        float(CONFIG["duration_s"])*.79,
        max(0.0,float(CONFIG["duration_s"])-1.0),
    ]
    for index,preview_time in enumerate(preview_times,1):
        Image.fromarray(scene.render_frame(float(preview_time))).save(
            PREVIEW_DIR/f"preview_{index:02d}_{preview_time:05.2f}s.png"
        )

    print("Summary:",summary_path.resolve())
    render_video(scene)
    print("Output directory:",OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-",path.name)


if __name__=="__main__":
    main()
