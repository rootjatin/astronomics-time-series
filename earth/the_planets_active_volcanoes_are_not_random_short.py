
from __future__ import annotations

"""
The Planet's Active Volcanoes Are Not Random — cinematic YouTube Short renderer

Creates a vertical 1080x1920 science Short explaining why Earth's volcanoes
cluster in recognizable global patterns rather than appearing randomly.



The plotted volcanoes are a curated, illustrative subset of well-known
historically or recently active volcanic systems. They are used to communicate
spatial pattern and are NOT a complete catalogue, a live eruption feed, or a
formal hazard product. Plate-boundary geometry is deliberately schematic.

No internet connection or external data is required. All decorative particles,
map jitter and animation timing are deterministic.

Recommended install
-------------------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview render
--------------------
    VOLCANO_SHORT_QUICK=1 python the_planets_active_volcanoes_are_not_random_short.py

Full render
-----------
    python the_planets_active_volcanoes_are_not_random_short.py

Outputs
-------
- MP4 video
- SRT subtitles
- PNG preview frames
- JSON production/science summary
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

QUICK_MODE = os.environ.get("VOLCANO_SHORT_QUICK", "0") == "1"

OUTPUT_ROOT = Path("the_planets_active_volcanoes_are_not_random_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "the_planets_active_volcanoes_are_not_random",
    "title": "THE PLANET'S ACTIVE VOLCANOES ARE NOT RANDOM",
    "subtitle": "plate boundaries // subduction // rifts // hotspots",
    "background_stars": 80 if QUICK_MODE else 190,
    "embers": 40 if QUICK_MODE else 120,
    "contrast": 1.08,
    "saturation": 1.10,
    "vignette": 0.23,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0

COLORS = {
    "space": (4, 7, 16),
    "white": (247, 250, 255),
    "muted": (158, 197, 214),
    "cyan": (74, 226, 255),
    "blue": (71, 126, 255),
    "violet": (181, 108, 255),
    "gold": (255, 194, 82),
    "orange": (255, 132, 55),
    "lava": (255, 76, 30),
    "red": (255, 71, 95),
    "green": (101, 230, 156),
    "ocean": (9, 34, 62),
    "ocean_light": (14, 54, 87),
    "land": (45, 70, 73),
    "land_light": (74, 105, 101),
    "crust": (69, 88, 92),
    "mantle": (109, 45, 33),
    "magma": (255, 111, 30),
}

FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.4, 7.3, "Plot active volcanoes on a world map and a pattern jumps out: they cluster in long arcs and belts instead of appearing randomly."),
    (7.4, 17.1, "Those belts closely follow tectonic plate boundaries. The most famous is the Pacific Ring of Fire, wrapping around the Pacific Ocean."),
    (17.2, 28.0, "At many convergent boundaries, one plate sinks beneath another. Water released from the descending slab helps the mantle above it melt, feeding volcanic arcs."),
    (28.1, 38.5, "Where plates pull apart, hot mantle rises and partially melts as pressure drops. That builds volcanic ridges and rift zones from Iceland to East Africa."),
    (38.6, 49.2, "Not every volcano sits on a plate edge. Hotspots can punch through a plate from below, leaving volcanic chains as the plate moves overhead."),
    (49.3, 57.5, "So the planet's active volcanoes are not random. Their geography is a visible fingerprint of moving plates, mantle flow, and Earth's internal heat."),
]

if QUICK_MODE:
    _time_scale = float(CONFIG["duration_s"]) / 58.0
    CAPTIONS = [(a * _time_scale, b * _time_scale, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "pattern", "start": 0.0, "end": 7.8 if not QUICK_MODE else 1.62},
    {"name": "boundaries", "start": 7.8 if not QUICK_MODE else 1.62, "end": 18.0 if not QUICK_MODE else 3.74},
    {"name": "subduction", "start": 18.0 if not QUICK_MODE else 3.74, "end": 29.0 if not QUICK_MODE else 6.02},
    {"name": "rift", "start": 29.0 if not QUICK_MODE else 6.02, "end": 39.4 if not QUICK_MODE else 8.18},
    {"name": "hotspot", "start": 39.4 if not QUICK_MODE else 8.18, "end": 50.4 if not QUICK_MODE else 10.45},
    {"name": "finale", "start": 50.4 if not QUICK_MODE else 10.45, "end": float(CONFIG["duration_s"])},
]


# -----------------------------------------------------------------------------
# Curated volcano markers
# -----------------------------------------------------------------------------

# name, latitude, longitude, tectonic setting
VOLCANOES: List[Tuple[str, float, float, str]] = [
    # Alaska / Aleutians / Cascades / Mexico / Central America
    ("Shishaldin", 54.76, -163.97, "subduction"),
    ("Pavlof", 55.42, -161.89, "subduction"),
    ("Augustine", 59.36, -153.43, "subduction"),
    ("Redoubt", 60.49, -152.74, "subduction"),
    ("Mount St. Helens", 46.20, -122.18, "subduction"),
    ("Mount Rainier", 46.85, -121.76, "subduction"),
    ("Mount Hood", 45.37, -121.70, "subduction"),
    ("Mount Shasta", 41.41, -122.19, "subduction"),
    ("Lassen Peak", 40.49, -121.51, "subduction"),
    ("Popocatepetl", 19.02, -98.62, "subduction"),
    ("Volcan de Colima", 19.51, -103.62, "subduction"),
    ("Fuego", 14.47, -90.88, "subduction"),
    ("Pacaya", 14.38, -90.60, "subduction"),
    ("Santa Maria", 14.76, -91.55, "subduction"),
    ("Masaya", 11.98, -86.16, "subduction"),
    ("Concepcion", 11.54, -85.62, "subduction"),
    # Andes
    ("Nevado del Ruiz", 4.89, -75.32, "subduction"),
    ("Galeras", 1.22, -77.37, "subduction"),
    ("Reventador", -0.08, -77.66, "subduction"),
    ("Cotopaxi", -0.68, -78.44, "subduction"),
    ("Sangay", -2.00, -78.34, "subduction"),
    ("Sabancaya", -15.78, -71.85, "subduction"),
    ("Ubinas", -16.36, -70.90, "subduction"),
    ("Lascar", -23.37, -67.73, "subduction"),
    ("Villarrica", -39.42, -71.93, "subduction"),
    ("Llaima", -38.69, -71.73, "subduction"),
    # Kamchatka / Kurils / Japan / Philippines
    ("Klyuchevskoy", 56.06, 160.64, "subduction"),
    ("Shiveluch", 56.65, 161.36, "subduction"),
    ("Bezymianny", 55.98, 160.59, "subduction"),
    ("Ebeko", 50.69, 156.01, "subduction"),
    ("Sakurajima", 31.59, 130.66, "subduction"),
    ("Aso", 32.88, 131.10, "subduction"),
    ("Mount Fuji", 35.36, 138.73, "subduction"),
    ("Mayon", 13.26, 123.69, "subduction"),
    ("Taal", 14.00, 120.99, "subduction"),
    ("Pinatubo", 15.14, 120.35, "subduction"),
    # Indonesia / PNG / NZ
    ("Merapi", -7.54, 110.45, "subduction"),
    ("Semeru", -8.11, 112.92, "subduction"),
    ("Anak Krakatau", -6.10, 105.42, "subduction"),
    ("Agung", -8.34, 115.51, "subduction"),
    ("Sinabung", 3.17, 98.39, "subduction"),
    ("Ibu", 1.49, 127.63, "subduction"),
    ("Dukono", 1.68, 127.88, "subduction"),
    ("Ulawun", -5.05, 151.33, "subduction"),
    ("Rabaul", -4.27, 152.20, "subduction"),
    ("Whakaari", -37.52, 177.18, "subduction"),
    ("Ruapehu", -39.28, 175.57, "subduction"),
    # Mediterranean
    ("Etna", 37.75, 14.99, "subduction"),
    ("Stromboli", 38.79, 15.21, "subduction"),
    ("Vesuvius", 40.82, 14.43, "subduction"),
    # Iceland / East African Rift / Red Sea-Afar
    ("Fagradalsfjall", 63.89, -22.27, "rift"),
    ("Hekla", 63.98, -19.70, "rift"),
    ("Katla", 63.63, -19.05, "rift"),
    ("Grimsvotn", 64.42, -17.33, "rift"),
    ("Erta Ale", 13.60, 40.67, "rift"),
    ("Ol Doinyo Lengai", -2.76, 35.91, "rift"),
    ("Nyiragongo", -1.52, 29.25, "rift"),
    ("Nyamuragira", -1.41, 29.20, "rift"),
    # Hotspots / intraplate
    ("Kilauea", 19.42, -155.29, "hotspot"),
    ("Mauna Loa", 19.48, -155.61, "hotspot"),
    ("Sierra Negra", -0.83, -91.17, "hotspot"),
    ("Piton de la Fournaise", -21.24, 55.71, "hotspot"),
    ("Cumbre Vieja", 28.57, -17.83, "hotspot"),
]


# -----------------------------------------------------------------------------
# General helpers
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
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(7, int(size)))
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int = 28,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    stroke: int = 2,
    anchor: str = "la",
):
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold=bold),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(225, fill[3] if len(fill) > 3 else 225)),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int = 28,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    line_spacing: int = 6,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        box = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)
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
        draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 220))
        box = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += (box[3] - box[1]) + line_spacing


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
    for index, (start, end, text) in enumerate(captions, start=1):
        lines.extend([str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def cubic_bezier(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    count: int = 90,
) -> List[Tuple[float, float]]:
    values = np.linspace(0.0, 1.0, count)
    out: List[Tuple[float, float]] = []
    for u in values:
        v = 1.0 - u
        out.append((
            v**3 * p0[0] + 3 * v * v * u * p1[0] + 3 * v * u * u * p2[0] + u**3 * p3[0],
            v**3 * p0[1] + 3 * v * v * u * p1[1] + 3 * v * u * u * p2[1] + u**3 * p3[1],
        ))
    return out


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class VolcanoPatternScene:
    def __init__(self):
        self.map_box = (
            int(OUT_W * 0.055),
            int(OUT_H * 0.205),
            int(OUT_W * 0.945),
            int(OUT_H * 0.645),
        )
        self.volcanoes = [
            {"name": name, "lat": lat, "lon": lon, "setting": setting}
            for name, lat, lon, setting in VOLCANOES
        ]
        self.stars = self._make_stars(int(CONFIG["background_stars"]), 1883)
        self.embers = self._make_embers(int(CONFIG["embers"]), 79)
        self.hud = self._make_hud(48 if not QUICK_MODE else 22, 1912)

    @staticmethod
    def _make_stars(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.3, 1.7) * SCALE),
                "a": float(rng.uniform(16, 82)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            }
            for _ in range(count)
        ]

    @staticmethod
    def _make_embers(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "phase": float(rng.uniform(0, 1)),
                "speed": float(rng.uniform(0.5, 1.3)),
                "size": float(rng.uniform(0.8, 2.8) * SCALE),
                "alpha": float(rng.uniform(30, 120)),
            }
            for _ in range(count)
        ]

    @staticmethod
    def _make_hud(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "length": float(rng.uniform(12, 90) * SCALE),
                "a": float(rng.uniform(8, 36)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            }
            for _ in range(count)
        ]

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, COLORS["space"] + (255,))
        draw = ImageDraw.Draw(image)
        for star in self.stars:
            alpha = int(star["a"] * (0.72 + 0.28 * math.sin(t * 1.1 + star["phase"])))
            r = star["r"]
            draw.ellipse((star["x"] - r, star["y"] - r, star["x"] + r, star["y"] + r), fill=COLORS["white"] + (alpha,))

        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for cx, cy, radius, color, alpha in [
            (OUT_W * 0.16, OUT_H * 0.33, 330 * SCALE, COLORS["lava"], 22),
            (OUT_W * 0.82, OUT_H * 0.42, 320 * SCALE, COLORS["violet"], 13),
            (OUT_W * 0.48, OUT_H * 0.84, 410 * SCALE, COLORS["orange"], 12),
        ]:
            hd.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color + (alpha,))
        haze = haze.filter(ImageFilter.GaussianBlur(max(15, int(58 * SCALE))))
        image.alpha_composite(haze)

        # Slowly rising embers keep the background alive.
        ed = ImageDraw.Draw(image)
        for ember in self.embers:
            y = OUT_H - ((t * 75 * SCALE * ember["speed"] + ember["phase"] * OUT_H) % (OUT_H * 1.15))
            x = ember["x"] + 18 * SCALE * math.sin(t * 0.8 + ember["phase"] * 9.0)
            r = ember["size"]
            ed.ellipse((x - r, y - r, x + r, y + r), fill=COLORS["orange"] + (int(ember["alpha"]),))
        return image

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 170):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            box,
            radius=max(8, int(24 * SCALE)),
            fill=(2, 7, 17, alpha),
            outline=COLORS["orange"] + (62,),
            width=max(1, int(2 * SCALE)),
        )
        image.alpha_composite(overlay)

    def project(self, lon: float, lat: float) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.map_box
        x = x0 + ((lon + 180.0) / 360.0) * (x1 - x0)
        y = y0 + ((90.0 - lat) / 180.0) * (y1 - y0)
        return x, y

    def project_poly(self, points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [self.project(lon, lat) for lon, lat in points]

    def draw_world_map(self, image: Image.Image, grid_alpha: int = 42, land_alpha: int = 235):
        x0, y0, x1, y1 = self.map_box
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            self.map_box,
            radius=max(10, int(22 * SCALE)),
            fill=COLORS["ocean"] + (232,),
            outline=COLORS["cyan"] + (54,),
            width=max(1, int(2 * SCALE)),
        )

        for lon in range(-150, 180, 30):
            xa, _ = self.project(lon, 0)
            draw.line((xa, y0, xa, y1), fill=COLORS["muted"] + (grid_alpha,), width=1)
        for lat in range(-60, 90, 30):
            _, ya = self.project(0, lat)
            draw.line((x0, ya, x1, ya), fill=COLORS["muted"] + (grid_alpha,), width=1)

        # Simplified continent polygons, deliberately stylized for readability.
        continents = [
            [(-168, 72), (-142, 58), (-128, 52), (-123, 40), (-112, 31), (-100, 23), (-82, 25), (-72, 44), (-52, 48), (-58, 64), (-100, 70)],
            [(-82, 13), (-70, 8), (-62, -8), (-52, -22), (-55, -39), (-68, -55), (-76, -39), (-80, -15)],
            [(-10, 36), (5, 44), (25, 46), (40, 52), (75, 58), (105, 54), (135, 48), (155, 58), (170, 45), (150, 30), (120, 23), (100, 8), (78, 9), (58, 20), (40, 28), (25, 32), (12, 35)],
            [(-17, 36), (5, 34), (20, 24), (33, 10), (39, -12), (30, -34), (14, -35), (1, -22), (-10, 0)],
            [(112, -11), (130, -12), (153, -26), (150, -40), (132, -43), (115, -31)],
            [(-51, 82), (-20, 79), (-21, 62), (-43, 60)],
            [(44, -13), (51, -16), (50, -26), (45, -25)],
        ]
        for poly in continents:
            pts = self.project_poly(poly)
            draw.polygon(pts, fill=COLORS["land"] + (land_alpha,), outline=COLORS["land_light"] + (min(255, land_alpha),))

        image.alpha_composite(overlay)

    def boundary_paths(self) -> Dict[str, List[List[Tuple[float, float]]]]:
        # Schematic lines: not intended as a formal plate-boundary dataset.
        return {
            "subduction": [
                [(-165, 57), (-150, 54), (-135, 48), (-124, 41), (-116, 30), (-106, 20), (-98, 15), (-90, 10), (-82, 3), (-78, -12), (-74, -28), (-72, -42), (-75, -55)],
                [(160, 58), (150, 50), (142, 42), (137, 35), (130, 28), (124, 20), (120, 12), (116, 3), (110, -7), (120, -18), (138, -29), (155, -38), (172, -44)],
                [(-8, 37), (4, 39), (16, 39), (28, 37)],
            ],
            "rift": [
                [(-25, 66), (-18, 58), (-22, 45), (-28, 28), (-32, 10), (-28, -12), (-22, -34), (-15, -54)],
                [(35, 30), (38, 18), (36, 6), (31, -4), (29, -15), (34, -27)],
            ],
        }

    def draw_boundaries(self, image: Image.Image, reveal: float = 1.0, glow: bool = True):
        reveal = clamp(reveal)
        glow_layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        line_layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow_layer)
        ld = ImageDraw.Draw(line_layer)

        colors = {"subduction": COLORS["red"], "rift": COLORS["cyan"]}
        for setting, paths in self.boundary_paths().items():
            for path in paths:
                pts = self.project_poly(path)
                count = max(2, int(round(len(pts) * reveal)))
                shown = pts[:count]
                color = colors[setting]
                if glow:
                    gd.line(shown, fill=color + (80,), width=max(4, int(11 * SCALE)), joint="curve")
                ld.line(shown, fill=color + (215,), width=max(1, int(3 * SCALE)), joint="curve")

        if glow:
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(max(2, int(8 * SCALE))))
            image.alpha_composite(glow_layer)
        image.alpha_composite(line_layer)

    def draw_volcanoes(self, image: Image.Image, reveal: float = 1.0, color_by_setting: bool = False, pulse: float = 0.0, alpha: int = 235):
        reveal = clamp(reveal)
        count = max(0, min(len(self.volcanoes), int(round(len(self.volcanoes) * reveal))))
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        dots = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        dd = ImageDraw.Draw(dots)
        setting_colors = {"subduction": COLORS["lava"], "rift": COLORS["cyan"], "hotspot": COLORS["gold"]}

        for i, volcano in enumerate(self.volcanoes[:count]):
            x, y = self.project(volcano["lon"], volcano["lat"])
            color = setting_colors[volcano["setting"]] if color_by_setting else COLORS["lava"]
            beat = 0.82 + 0.24 * math.sin(pulse * 3.0 + i * 0.71)
            r = (4.2 if not QUICK_MODE else 2.2) * SCALE * beat
            gr = r * 4.3
            gd.ellipse((x - gr, y - gr, x + gr, y + gr), fill=color + (58,))
            dd.ellipse((x - r, y - r, x + r, y + r), fill=color + (alpha,), outline=COLORS["white"] + (95,))

        glow = glow.filter(ImageFilter.GaussianBlur(max(2, int(7 * SCALE))))
        image.alpha_composite(glow)
        image.alpha_composite(dots)

    def draw_ring_of_fire(self, image: Image.Image, local: float):
        local = clamp(local)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        ring_paths = self.boundary_paths()["subduction"][:2]
        for path in ring_paths:
            pts = self.project_poly(path)
            width = max(3, int(7 * SCALE))
            draw.line(pts, fill=COLORS["gold"] + (int(80 + 120 * local),), width=width, joint="curve")
        overlay = overlay.filter(ImageFilter.GaussianBlur(max(1, int(5 * SCALE))))
        image.alpha_composite(overlay)

    def draw_pattern_scene(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[0]
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_world_map(image, grid_alpha=28)
        self.draw_volcanoes(image, reveal=min(1.0, local * 1.18), pulse=t)
        self.panel(image, (int(OUT_W * 0.08), int(OUT_H * 0.675), int(OUT_W * 0.92), int(OUT_H * 0.81)), alpha=174)
        draw_text(image, "THE DOTS DO NOT FILL THE MAP EVENLY", (OUT_W // 2, int(OUT_H * 0.718)), size=23 if not QUICK_MODE else 11, fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "they form arcs, chains and narrow belts", (OUT_W // 2, int(OUT_H * 0.765)), size=17 if not QUICK_MODE else 8, fill=COLORS["orange"] + (235,), anchor="ma", stroke=1)

    def draw_boundaries_scene(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "boundaries")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_world_map(image, grid_alpha=24)
        self.draw_boundaries(image, reveal=min(1.0, local * 1.25))
        self.draw_volcanoes(image, reveal=1.0, color_by_setting=True, pulse=t)
        self.draw_ring_of_fire(image, local)
        draw_text(image, "PACIFIC RING OF FIRE", (OUT_W // 2, int(OUT_H * 0.185)), size=22 if not QUICK_MODE else 11, fill=COLORS["gold"] + (240,), bold=True, anchor="ma", stroke=1)
        self.panel(image, (int(OUT_W * 0.08), int(OUT_H * 0.675), int(OUT_W * 0.92), int(OUT_H * 0.82)), alpha=174)
        draw_text(image, "VOLCANIC BELTS TRACK TECTONIC PLATE EDGES", (OUT_W // 2, int(OUT_H * 0.716)), size=22 if not QUICK_MODE else 11, fill=COLORS["cyan"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "red = convergent margins  •  cyan = rifts", (OUT_W // 2, int(OUT_H * 0.763)), size=16 if not QUICK_MODE else 8, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)
        draw_text(image, "schematic boundaries", (OUT_W // 2, int(OUT_H * 0.795)), size=13 if not QUICK_MODE else 6, fill=COLORS["muted"] + (185,), anchor="ma", stroke=1)

    def draw_subduction_scene(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "subduction")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        top = int(OUT_H * 0.21)
        bottom = int(OUT_H * 0.64)
        left = int(OUT_W * 0.07)
        right = int(OUT_W * 0.93)

        # Ocean / mantle background.
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((left, top, right, bottom), radius=max(10, int(24 * SCALE)), fill=COLORS["ocean"] + (235,), outline=COLORS["cyan"] + (45,), width=1)
        mantle_y = int(OUT_H * 0.47)
        draw.rectangle((left, mantle_y, right, bottom), fill=COLORS["mantle"] + (230,))

        # Continental overriding plate.
        continent = [(OUT_W * 0.57, OUT_H * 0.365), (OUT_W * 0.93, OUT_H * 0.365), (OUT_W * 0.93, OUT_H * 0.49), (OUT_W * 0.66, OUT_H * 0.49)]
        draw.polygon(continent, fill=COLORS["crust"] + (255,), outline=COLORS["land_light"] + (220,))
        draw.polygon([(OUT_W * 0.58, OUT_H * 0.365), (OUT_W * 0.68, OUT_H * 0.30), (OUT_W * 0.76, OUT_H * 0.365)], fill=COLORS["land_light"] + (255,))

        # Oceanic plate bends and descends.
        slab = cubic_bezier(
            (OUT_W * 0.08, OUT_H * 0.41),
            (OUT_W * 0.42, OUT_H * 0.41),
            (OUT_W * 0.58, OUT_H * 0.47),
            (OUT_W * 0.78, OUT_H * 0.63),
            110,
        )
        draw.line(slab, fill=COLORS["blue"] + (250,), width=max(8, int(24 * SCALE)))
        draw.line(slab, fill=COLORS["cyan"] + (160,), width=max(1, int(4 * SCALE)))

        # Motion arrows.
        arrow_a = (OUT_W * 0.20, OUT_H * 0.39)
        arrow_b = (OUT_W * 0.40, OUT_H * 0.39)
        draw.line((*arrow_a, *arrow_b), fill=COLORS["cyan"] + (230,), width=max(1, int(4 * SCALE)))
        draw.polygon([(arrow_b[0], arrow_b[1]), (arrow_b[0]-18*SCALE, arrow_b[1]-9*SCALE), (arrow_b[0]-18*SCALE, arrow_b[1]+9*SCALE)], fill=COLORS["cyan"] + (230,))

        # Fluids/volatiles rise from slab.
        fluid = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        fd = ImageDraw.Draw(fluid)
        for i in range(10):
            u = (i / 9.0 + t * 0.08) % 1.0
            x = lerp(OUT_W * 0.57, OUT_W * 0.67, u)
            y = lerp(OUT_H * 0.53, OUT_H * 0.43, u)
            r = (3.5 + 2.0 * math.sin(i + t)) * SCALE
            fd.ellipse((x-r, y-r, x+r, y+r), fill=COLORS["cyan"] + (150,))
        fluid = fluid.filter(ImageFilter.GaussianBlur(max(1, int(2*SCALE))))
        image.alpha_composite(fluid)

        # Melting zone and magma rise.
        magma = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        md = ImageDraw.Draw(magma)
        cx, cy = OUT_W * 0.67, OUT_H * 0.49
        radius = (55 + 22 * local) * SCALE
        md.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=COLORS["magma"] + (95,))
        md.line((cx, cy, OUT_W * 0.68, OUT_H * 0.34), fill=COLORS["magma"] + (235,), width=max(3, int(10 * SCALE)))
        magma = magma.filter(ImageFilter.GaussianBlur(max(2, int(8*SCALE))))
        image.alpha_composite(magma)

        draw_text(image, "OCEANIC PLATE", (int(OUT_W * 0.18), int(OUT_H * 0.345)), size=16 if not QUICK_MODE else 8, fill=COLORS["cyan"] + (230,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "DESCENDING SLAB", (int(OUT_W * 0.69), int(OUT_H * 0.605)), size=15 if not QUICK_MODE else 7, fill=COLORS["blue"] + (230,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "VOLCANIC ARC", (int(OUT_W * 0.68), int(OUT_H * 0.285)), size=15 if not QUICK_MODE else 7, fill=COLORS["lava"] + (235,), bold=True, anchor="ma", stroke=1)

        self.panel(image, (int(OUT_W * 0.08), int(OUT_H * 0.685), int(OUT_W * 0.92), int(OUT_H * 0.82)), alpha=182)
        draw_text(image, "SUBDUCTION BUILDS VOLCANIC ARCS", (OUT_W // 2, int(OUT_H * 0.728)), size=24 if not QUICK_MODE else 12, fill=COLORS["lava"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "a sinking plate helps trigger melting above the slab", (OUT_W // 2, int(OUT_H * 0.775)), size=16 if not QUICK_MODE else 8, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)

    def draw_rift_scene(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "rift")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        top = int(OUT_H * 0.22)
        bottom = int(OUT_H * 0.63)
        left = int(OUT_W * 0.07)
        right = int(OUT_W * 0.93)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((left, top, right, bottom), radius=max(10, int(24 * SCALE)), fill=(8, 21, 34, 240), outline=COLORS["cyan"] + (48,), width=1)
        mantle_y = int(OUT_H * 0.49)
        draw.rectangle((left, mantle_y, right, bottom), fill=COLORS["mantle"] + (240,))

        gap = lerp(26, 90, local) * SCALE
        cx = OUT_W * 0.50
        crust_y = OUT_H * 0.40
        thickness = 70 * SCALE
        left_plate = [(left, crust_y), (cx-gap, crust_y), (cx-gap*1.18, crust_y+thickness), (left, crust_y+thickness)]
        right_plate = [(cx+gap, crust_y), (right, crust_y), (right, crust_y+thickness), (cx+gap*1.18, crust_y+thickness)]
        draw.polygon(left_plate, fill=COLORS["crust"] + (255,), outline=COLORS["land_light"] + (210,))
        draw.polygon(right_plate, fill=COLORS["crust"] + (255,), outline=COLORS["land_light"] + (210,))

        # Upwelling mantle / magma.
        plume = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(plume)
        pd.polygon([(cx-90*SCALE, bottom), (cx+90*SCALE, bottom), (cx+gap*0.35, crust_y+28*SCALE), (cx-gap*0.35, crust_y+28*SCALE)], fill=COLORS["magma"] + (115,))
        pd.line((cx, bottom-20*SCALE, cx, crust_y-42*SCALE), fill=COLORS["magma"] + (235,), width=max(4, int(14*SCALE)))
        plume = plume.filter(ImageFilter.GaussianBlur(max(2, int(9*SCALE))))
        image.alpha_composite(plume)

        # Direction arrows.
        y = crust_y - 35 * SCALE
        for sign in (-1, 1):
            x0 = cx + sign * gap * 0.6
            x1 = cx + sign * 170 * SCALE
            draw.line((x0, y, x1, y), fill=COLORS["cyan"] + (235,), width=max(1, int(4*SCALE)))
            if sign < 0:
                draw.polygon([(x1, y), (x1+18*SCALE, y-9*SCALE), (x1+18*SCALE, y+9*SCALE)], fill=COLORS["cyan"] + (235,))
            else:
                draw.polygon([(x1, y), (x1-18*SCALE, y-9*SCALE), (x1-18*SCALE, y+9*SCALE)], fill=COLORS["cyan"] + (235,))

        # Small rift eruptions.
        for i in range(4):
            ex = cx + (i-1.5) * 16 * SCALE
            ey = crust_y - 8 * SCALE
            height = (26 + 12 * math.sin(t * 4 + i)) * SCALE * local
            draw.line((ex, ey, ex + 4*SCALE*math.sin(i), ey-height), fill=COLORS["lava"] + (220,), width=max(1, int(4*SCALE)))

        draw_text(image, "PLATES MOVE APART", (OUT_W // 2, int(OUT_H * 0.295)), size=19 if not QUICK_MODE else 9, fill=COLORS["cyan"] + (238,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "HOT MANTLE RISES", (OUT_W // 2, int(OUT_H * 0.585)), size=16 if not QUICK_MODE else 8, fill=COLORS["orange"] + (230,), bold=True, anchor="ma", stroke=1)

        self.panel(image, (int(OUT_W * 0.08), int(OUT_H * 0.685), int(OUT_W * 0.92), int(OUT_H * 0.82)), alpha=182)
        draw_text(image, "RIFTS MAKE VOLCANOES TOO", (OUT_W // 2, int(OUT_H * 0.728)), size=24 if not QUICK_MODE else 12, fill=COLORS["cyan"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "rising mantle partially melts as pressure drops", (OUT_W // 2, int(OUT_H * 0.775)), size=16 if not QUICK_MODE else 8, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)

    def draw_hotspot_scene(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "hotspot")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        top = int(OUT_H * 0.22)
        bottom = int(OUT_H * 0.63)
        left = int(OUT_W * 0.07)
        right = int(OUT_W * 0.93)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((left, top, right, bottom), radius=max(10, int(24 * SCALE)), fill=(8, 21, 34, 240), outline=COLORS["gold"] + (48,), width=1)
        mantle_y = int(OUT_H * 0.50)
        draw.rectangle((left, mantle_y, right, bottom), fill=COLORS["mantle"] + (238,))
        crust_y = OUT_H * 0.40
        draw.rectangle((left, crust_y, right, mantle_y), fill=COLORS["crust"] + (255,))

        # Mantle plume.
        plume = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(plume)
        cx = OUT_W * 0.55
        pd.polygon([(cx-70*SCALE, bottom), (cx+70*SCALE, bottom), (cx+22*SCALE, crust_y+38*SCALE), (cx-22*SCALE, crust_y+38*SCALE)], fill=COLORS["magma"] + (120,))
        pd.ellipse((cx-55*SCALE, crust_y+10*SCALE, cx+55*SCALE, crust_y+95*SCALE), fill=COLORS["magma"] + (145,))
        plume = plume.filter(ImageFilter.GaussianBlur(max(2, int(10*SCALE))))
        image.alpha_composite(plume)

        # Plate motion and island chain.
        motion = local * 105 * SCALE
        island_base = [(-220, 18), (-160, 31), (-100, 45), (-40, 64), (20, 96)]
        for i, (dx, size) in enumerate(island_base):
            x = cx + (dx - motion) * SCALE
            y = crust_y - 4*SCALE
            if left+25*SCALE < x < right-25*SCALE:
                s = size * SCALE
                color = COLORS["land_light"] if i < len(island_base)-1 else COLORS["lava"]
                draw.polygon([(x-s*0.38, y), (x, y-s*0.44), (x+s*0.38, y)], fill=color + (245,))
                if i == len(island_base)-1:
                    draw.line((x, y-s*0.43, x+5*SCALE*math.sin(t*3), y-s*0.72), fill=COLORS["lava"] + (230,), width=max(1, int(4*SCALE)))

        arrow_y = crust_y - 80*SCALE
        draw.line((OUT_W*0.33, arrow_y, OUT_W*0.72, arrow_y), fill=COLORS["cyan"] + (230,), width=max(1, int(4*SCALE)))
        draw.polygon([(OUT_W*0.33, arrow_y), (OUT_W*0.33+18*SCALE, arrow_y-9*SCALE), (OUT_W*0.33+18*SCALE, arrow_y+9*SCALE)], fill=COLORS["cyan"] + (230,))
        draw_text(image, "PLATE MOTION", (OUT_W // 2, int(arrow_y-24*SCALE)), size=15 if not QUICK_MODE else 7, fill=COLORS["cyan"] + (225,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "HOTSPOT", (int(cx), int(OUT_H * 0.595)), size=16 if not QUICK_MODE else 8, fill=COLORS["gold"] + (240,), bold=True, anchor="ma", stroke=1)

        self.panel(image, (int(OUT_W * 0.08), int(OUT_H * 0.685), int(OUT_W * 0.92), int(OUT_H * 0.82)), alpha=182)
        draw_text(image, "SOME VOLCANOES FORM INSIDE PLATES", (OUT_W // 2, int(OUT_H * 0.728)), size=23 if not QUICK_MODE else 11, fill=COLORS["gold"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "a moving plate can leave a chain above a hotspot", (OUT_W // 2, int(OUT_H * 0.775)), size=16 if not QUICK_MODE else 8, fill=COLORS["white"] + (225,), anchor="ma", stroke=1)

    def draw_finale_scene(self, image: Image.Image, t: float):
        shot = next(item for item in SHOT_PLAN if item["name"] == "finale")
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        self.draw_world_map(image, grid_alpha=20)
        self.draw_boundaries(image, reveal=1.0)
        self.draw_volcanoes(image, reveal=1.0, color_by_setting=True, pulse=t)
        self.draw_ring_of_fire(image, 0.8 + 0.2 * local)

        self.panel(image, (int(OUT_W * 0.065), int(OUT_H * 0.655), int(OUT_W * 0.935), int(OUT_H * 0.845)), alpha=192)
        draw_text(image, "THE PLANET'S ACTIVE VOLCANOES", (OUT_W // 2, int(OUT_H * 0.695)), size=28 if not QUICK_MODE else 14, fill=COLORS["white"] + (250,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "ARE NOT RANDOM", (OUT_W // 2, int(OUT_H * 0.742)), size=35 if not QUICK_MODE else 17, fill=COLORS["lava"] + (250,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "PLATE EDGES", (int(OUT_W * 0.26), int(OUT_H * 0.795)), size=17 if not QUICK_MODE else 8, fill=COLORS["cyan"] + (235,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "RIFTS", (int(OUT_W * 0.50), int(OUT_H * 0.795)), size=17 if not QUICK_MODE else 8, fill=COLORS["violet"] + (235,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "HOTSPOTS", (int(OUT_W * 0.74), int(OUT_H * 0.795)), size=17 if not QUICK_MODE else 8, fill=COLORS["gold"] + (235,), bold=True, anchor="ma", stroke=1)
        draw_text(image, "the map is a fingerprint of a moving planet", (OUT_W // 2, int(OUT_H * 0.827)), size=15 if not QUICK_MODE else 7, fill=COLORS["white"] + (215,), anchor="ma", stroke=1)

    def draw_titles(self, image: Image.Image, t: float, shot_name: str):
        intro_end = 5.4 if not QUICK_MODE else 1.08
        if t < intro_end:
            fade = smoothstep(t / (0.7 if not QUICK_MODE else 0.15))
            draw_text(image, "THE PLANET'S ACTIVE", (OUT_W // 2, int(OUT_H * 0.074)), size=34 if not QUICK_MODE else 17, fill=COLORS["white"] + (int(245 * fade),), bold=True, anchor="ma", stroke=2)
            draw_text(image, "VOLCANOES", (OUT_W // 2, int(OUT_H * 0.111)), size=53 if not QUICK_MODE else 26, fill=COLORS["lava"] + (int(250 * fade),), bold=True, anchor="ma", stroke=2)
            draw_text(image, "ARE NOT RANDOM", (OUT_W // 2, int(OUT_H * 0.154)), size=39 if not QUICK_MODE else 19, fill=COLORS["gold"] + (int(245 * fade),), bold=True, anchor="ma", stroke=2)

        labels = {
            "pattern": "1 // THE GLOBAL PATTERN",
            "boundaries": "2 // PLATE BOUNDARIES",
            "subduction": "3 // SUBDUCTION ZONES",
            "rift": "4 // RIFTS & RIDGES",
            "hotspot": "5 // HOTSPOTS",
            "finale": "6 // A MOVING PLANET",
        }
        if t > (5.0 if not QUICK_MODE else 1.0):
            draw_text(image, labels[shot_name], (54 if not QUICK_MODE else 27, 60 if not QUICK_MODE else 30), size=18 if not QUICK_MODE else 9, fill=COLORS["muted"] + (205,), bold=True, stroke=1)

    def draw_source_hud(self, image: Image.Image):
        draw_text(image, "SCHEMATIC SCIENCE VISUAL", (OUT_W - (46 if not QUICK_MODE else 23), 62 if not QUICK_MODE else 31), size=15 if not QUICK_MODE else 7, fill=COLORS["orange"] + (225,), bold=True, anchor="ra", stroke=1)
        draw_text(image, f"VOLCANO MARKERS // {len(self.volcanoes)}", (OUT_W - (46 if not QUICK_MODE else 23), 91 if not QUICK_MODE else 45), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (195,), anchor="ra", stroke=1)
        draw_text(image, "CURATED SUBSET // NOT LIVE", (OUT_W - (46 if not QUICK_MODE else 23), 117 if not QUICK_MODE else 58), size=13 if not QUICK_MODE else 6, fill=COLORS["muted"] + (175,), anchor="ra", stroke=1)

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        y0 = OUT_H - (250 if not QUICK_MODE else 126)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle(
            (44 if not QUICK_MODE else 22, y0, OUT_W - (44 if not QUICK_MODE else 22), y0 + (132 if not QUICK_MODE else 68)),
            radius=24 if not QUICK_MODE else 12,
            fill=(2, 6, 15, 180),
            outline=COLORS["orange"] + (64,),
            width=1,
        )
        image.alpha_composite(panel)
        draw_wrapped_text(
            image,
            text,
            (68 if not QUICK_MODE else 34, y0 + (26 if not QUICK_MODE else 13)),
            OUT_W - (136 if not QUICK_MODE else 68),
            size=20 if not QUICK_MODE else 10,
            fill=COLORS["white"] + (245,),
            line_spacing=5 if not QUICK_MODE else 2,
        )

    def draw_hud_noise(self, image: Image.Image, t: float):
        draw = ImageDraw.Draw(image)
        for item in self.hud:
            alpha = int(item["a"] * (0.6 + 0.4 * math.sin(t * 1.8 + item["phase"])))
            x, y = item["x"], item["y"]
            draw.line((x, y, x + item["length"], y), fill=COLORS["muted"] + (max(0, alpha),), width=1)

        # Upper-left / lower-right corner brackets.
        m = 28 if not QUICK_MODE else 14
        l = 28 if not QUICK_MODE else 14
        c = COLORS["orange"] + (90,)
        draw.line((m, m, m + l, m), fill=c, width=1)
        draw.line((m, m, m, m + l), fill=c, width=1)
        draw.line((OUT_W - m - l, OUT_H - m, OUT_W - m, OUT_H - m), fill=c, width=1)
        draw.line((OUT_W - m, OUT_H - m - l, OUT_W - m, OUT_H - m), fill=c, width=1)

    def render_frame(self, t: float) -> np.ndarray:
        image = self.background(t)
        shot = get_shot(t)
        name = shot["name"]

        if name == "pattern":
            self.draw_pattern_scene(image, t)
        elif name == "boundaries":
            self.draw_boundaries_scene(image, t)
        elif name == "subduction":
            self.draw_subduction_scene(image, t)
        elif name == "rift":
            self.draw_rift_scene(image, t)
        elif name == "hotspot":
            self.draw_hotspot_scene(image, t)
        else:
            self.draw_finale_scene(image, t)

        self.draw_titles(image, t, name)
        self.draw_source_hud(image)
        self.draw_caption(image, t)
        self.draw_hud_noise(image, t)

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        arr *= VIGNETTE[:, :, None]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return apply_grade(arr)


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def render_video(scene: VolcanoPatternScene) -> Path:
    fps = int(CONFIG["fps"])
    duration = float(CONFIG["duration_s"])
    frame_count = int(round(fps * duration))
    temp_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_silent.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"

    writer = iio.get_writer(
        temp_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_log_level="error",
    )
    try:
        for frame_index in tqdm(range(frame_count), desc="Rendering frames"):
            t = frame_index / fps
            writer.append_data(scene.render_frame(t))
    finally:
        writer.close()

    # No audio is generated, so simply copy the silent render to the final name.
    shutil.copyfile(temp_path, final_path)
    return final_path


def save_summary() -> Path:
    setting_counts: Dict[str, int] = {}
    for _, _, _, setting in VOLCANOES:
        setting_counts[setting] = setting_counts.get(setting, 0) + 1

    path = OUTPUT_ROOT / "volcano_short_summary.json"
    path.write_text(
        json.dumps(
            {
                "title": CONFIG["title"],
                "subtitle": CONFIG["subtitle"],
                "quick_mode": QUICK_MODE,
                "resolution": [OUT_W, OUT_H],
                "fps": CONFIG["fps"],
                "duration_s": CONFIG["duration_s"],
                "volcano_marker_count": len(VOLCANOES),
                "setting_counts": setting_counts,
                "science_scope": "curated illustrative subset of well-known historically or recently active volcanic systems",
                "not_a_live_feed": True,
                "plate_boundaries": "schematic, not a formal GIS tectonic-boundary dataset",
                "story": [
                    "global clustering",
                    "plate boundaries and Pacific Ring of Fire",
                    "subduction-zone volcanism",
                    "rift and divergent-boundary volcanism",
                    "intraplate hotspot volcanism",
                    "final synthesis",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path



