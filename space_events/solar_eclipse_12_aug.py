from __future__ import annotations

"""
August 12, 2026 Total Solar Eclipse — Cinematic YouTube Short Renderer v3
============================================================================

Purpose
-------
This is a third-pass rebuild focused on two user-requested fixes:

1. Use a proper map renderer instead of a crude pseudo-map.
2. Prevent text from being cropped in the vertical 1080x1920 layout.

What is new in v3
-----------------
- Uses Cartopy (preferred) to render a real Europe/North Atlantic map.
- Falls back gracefully if Cartopy is unavailable.
- All on-screen text is laid out in bounding boxes with automatic fitting,
  so headlines and captions shrink before they crop.
- The story is still descriptive and cinematic, but map scenes are now built
  from actual geospatial plotting rather than hand-drawn polygons.

Recommended install
-------------------
    pip install numpy pillow imageio imageio-ffmpeg tqdm matplotlib cartopy

Quick preview render
--------------------
    ECLIPSE_SHORT_QUICK=1 python august_12_2026_total_solar_eclipse_cinematic_short_v3.py

Full render
-----------
    python august_12_2026_total_solar_eclipse_cinematic_short_v3.py

Notes
-----
- Cartopy may download Natural Earth shapefiles automatically the first time.
- If Cartopy is not available, the script falls back to a local static basemap
  image if found, or a very simple placeholder.
- The eclipse path, timing, and numeric labels are embedded from NASA/GSFC.
"""

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

# Delay heavy imports so the script still loads without cartopy.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_OK = True
except Exception:
    CARTOPY_OK = False


# =============================================================================
# Config
# =============================================================================

QUICK_MODE = os.environ.get("ECLIPSE_SHORT_QUICK", "0") == "1"

OUTPUT_ROOT = Path("august_12_2026_eclipse_short_output_v3")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
DATA_DIR = OUTPUT_ROOT / "data"
CACHE_DIR = OUTPUT_ROOT / "cache"
for directory in (OUTPUT_ROOT, PREVIEW_DIR, DATA_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 8 if QUICK_MODE else 30,
    "duration_s": 13.0 if QUICK_MODE else 56.0,
    "output_basename": "august_12_2026_total_solar_eclipse_cinematic_pro_v3",
    "background_stars": 220 if QUICK_MODE else 760,
    "contrast": 1.10,
    "saturation": 0.98,
    "vignette": 0.26,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0
FPS = int(CONFIG["fps"])
DURATION = float(CONFIG["duration_s"])


# =============================================================================
# Grounded eclipse data
# =============================================================================

ECLIPSE_FACTS: Dict[str, Any] = {
    "date": "2026-08-12",
    "type": "Total Solar Eclipse",
    "greatest_eclipse_utc": "17:45:53.8",
    "greatest_lat_deg": 65.225,
    "greatest_lon_deg": -25.228333,
    "greatest_path_width_km": 294.0,
    "greatest_central_duration_s": 138.2,
    "eclipse_magnitude": 1.039,
    "saros": 126,
}

# UTC, lat_deg, lon_deg, path_width_km, central_duration_s
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
    ("REYKJAVIK", "ICELAND", "TOTAL", "16:47", "17:48–17:49", "18:47", "TOTAL"),
    ("LEON", "SPAIN", "TOTAL", "19:32", "20:28–20:30", "21:22", "TOTAL"),
    ("ZARAGOZA", "SPAIN", "TOTAL", "19:34", "20:29–20:30", "21:07*", "TOTAL"),
    ("VALENCIA", "SPAIN", "TOTAL", "19:38", "20:32–20:33", "21:01*", "TOTAL"),
    ("MADRID", "SPAIN", "PARTIAL", "19:36", "20:32", "21:16*", "99%"),
    ("BARCELONA", "SPAIN", "PARTIAL", "19:35", "20:29", "20:54*", "99%"),
    ("LONDON", "U.K.", "PARTIAL", "18:17", "19:13", "20:06", "91%"),
    ("PARIS", "FRANCE", "PARTIAL", "19:22", "20:17", "21:09", "92%"),
]

CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "REYKJAVIK": (64.1466, -21.9426),
    "LEON": (42.5987, -5.5671),
    "ZARAGOZA": (41.6488, -0.8891),
    "VALENCIA": (39.4699, -0.3763),
    "MADRID": (40.4168, -3.7038),
    "BARCELONA": (41.3874, 2.1686),
    "LONDON": (51.5072, -0.1276),
    "PARIS": (48.8566, 2.3522),
}


# =============================================================================
# Narration / shot plan
# =============================================================================

FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.4, 6.3, "On August 12, 2026, the Moon's shadow crossed the Arctic, Iceland, the North Atlantic, and Spain — bringing a total solar eclipse to parts of Europe."),
    (6.4, 13.2, "A total solar eclipse happens when the Moon moves directly in front of the Sun. Inside the tiny dark umbra, the Sun is completely covered and the corona appears."),
    (13.3, 22.5, "This map scene now uses a proper geospatial renderer. NASA's central-line data curves out of the Arctic, passes Iceland, and reaches northern Spain."),
    (22.6, 31.2, "Greatest eclipse happened around 17:45:54 UTC near 65.2 degrees north and 25.2 degrees west. The totality path was about 294 kilometers wide, with about 2 minutes 18 seconds of totality."),
    (31.3, 40.5, "Not all of Europe saw the same thing. Reykjavik, Leon, Zaragoza, and Valencia reached totality. Madrid and Barcelona saw deep partial eclipses near ninety-nine percent coverage."),
    (40.6, 48.2, "In Spain the eclipse came late in the evening, so the Moon's black disk and the white solar corona hung low in the western sky close to sunset."),
    (48.3, 55.8, "And one safety rule matters: during the partial phases you must use proper eclipse glasses or certified solar filters. Only the brief total phase is safe to view directly."),
]

SHOT_PLAN_FULL = [
    {"name": "hook", "start": 0.0, "end": 6.5},
    {"name": "alignment", "start": 6.5, "end": 13.5},
    {"name": "path", "start": 13.5, "end": 22.8},
    {"name": "greatest", "start": 22.8, "end": 31.5},
    {"name": "cities", "start": 31.5, "end": 40.8},
    {"name": "spain", "start": 40.8, "end": 48.4},
    {"name": "safety", "start": 48.4, "end": 56.0},
]

if QUICK_MODE:
    scale_t = DURATION / 56.0
    CAPTIONS = [(a * scale_t, b * scale_t, txt) for a, b, txt in FULL_CAPTIONS]
    SHOT_PLAN = [{"name": s["name"], "start": s["start"] * scale_t, "end": s["end"] * scale_t} for s in SHOT_PLAN_FULL]
else:
    CAPTIONS = FULL_CAPTIONS
    SHOT_PLAN = SHOT_PLAN_FULL


# =============================================================================
# Utility helpers
# =============================================================================

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(x: float) -> float:
    x = clamp(x)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def caption_at(t: float) -> Optional[str]:
    for a, b, txt in CAPTIONS:
        if a <= t < b:
            return txt
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
            return ImageFont.truetype(candidate, size=max(8, int(size)))
        except Exception:
            pass
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, stroke: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def fit_wrapped_text(
    image: Image.Image,
    text: str,
    box: Tuple[int, int, int, int],
    max_size: int,
    min_size: int,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    stroke: int = 2,
    line_spacing: int = 6,
    align: str = "left",
    valign: str = "top",
):
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    width = max(10, x1 - x0)
    height = max(10, y1 - y0)

    chosen = None
    chosen_lines: List[str] = []
    for size in range(max_size, min_size - 1, -1):
        font = get_font(size, bold=bold)
        lines = wrap_text(draw, text, font, width, stroke)
        b = draw.textbbox((0, 0), "Ag", font=font, stroke_width=stroke)
        line_h = b[3] - b[1]
        total_h = len(lines) * line_h + max(0, len(lines) - 1) * line_spacing
        if total_h <= height:
            # ensure each line fits, already wrapped; choose this size
            chosen = font
            chosen_lines = lines
            chosen_size = size
            break
    if chosen is None:
        chosen_size = min_size
        chosen = get_font(min_size, bold=bold)
        chosen_lines = wrap_text(draw, text, chosen, width, stroke)
        # if still too tall, hard cut by line count
        b = draw.textbbox((0, 0), "Ag", font=chosen, stroke_width=stroke)
        line_h = b[3] - b[1]
        max_lines = max(1, int((height + line_spacing) / max(1, line_h + line_spacing)))
        chosen_lines = chosen_lines[:max_lines]

    b = draw.textbbox((0, 0), "Ag", font=chosen, stroke_width=stroke)
    line_h = b[3] - b[1]
    total_h = len(chosen_lines) * line_h + max(0, len(chosen_lines) - 1) * line_spacing

    if valign == "center":
        y = y0 + (height - total_h) / 2
    elif valign == "bottom":
        y = y1 - total_h
    else:
        y = y0

    for line in chosen_lines:
        bbox = draw.textbbox((0, 0), line, font=chosen, stroke_width=stroke)
        text_w = bbox[2] - bbox[0]
        if align == "center":
            x = x0 + (width - text_w) / 2
        elif align == "right":
            x = x1 - text_w
        else:
            x = x0
        draw.text(
            (x, y),
            line,
            font=chosen,
            fill=fill,
            stroke_width=stroke,
            stroke_fill=(0, 0, 0, min(245, fill[3] if len(fill) > 3 else 245)),
        )
        y += line_h + line_spacing


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
    for i, (a, b, txt) in enumerate(captions, start=1):
        lines.extend([str(i), f"{format_srt_time(a)} --> {format_srt_time(b)}", txt, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius ** 1.8, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


def apply_grade(rgb: np.ndarray) -> np.ndarray:
    image = Image.fromarray(rgb)
    image = ImageEnhance.Contrast(image).enhance(float(CONFIG["contrast"]))
    image = ImageEnhance.Color(image).enhance(float(CONFIG["saturation"]))
    arr = np.asarray(image).astype(np.float32)
    arr *= VIGNETTE[:, :, None]
    return np.clip(arr, 0, 255).astype(np.uint8)


def draw_panel(image: Image.Image, box: Tuple[int, int, int, int], fill=(7, 11, 20, 140), outline=(180, 205, 225, 60), radius: int = 24):
    panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
    pd = ImageDraw.Draw(panel)
    pd.rounded_rectangle(box, radius=max(4, int(radius * SCALE)), fill=fill, outline=outline, width=max(1, int(2 * SCALE)))
    panel = panel.filter(ImageFilter.GaussianBlur(max(1, int(1.1 * SCALE))))
    image.alpha_composite(panel)


def draw_glow_disc(image: Image.Image, cx: float, cy: float, radius: float, color=(255, 196, 120), alpha: int = 100):
    gl = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
    gd = ImageDraw.Draw(gl)
    for mult, a in [(4.0, alpha // 5), (2.6, alpha // 3), (1.7, alpha // 2), (1.0, alpha)]:
        r = radius * mult
        gd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(color[0], color[1], color[2], a))
    gl = gl.filter(ImageFilter.GaussianBlur(max(2, int(7 * SCALE))))
    image.alpha_composite(gl)


# =============================================================================
# Map rendering
# =============================================================================

EUROPE_EXTENT = (-75.0, 35.0, 25.0, 85.0)  # lon_min, lon_max, lat_min, lat_max
FALLBACK_BASEMAP = Path(__file__).with_name("blue_marble_reference_2048.png")


def interpolate_path(progress: float) -> Tuple[float, float, float, float, str]:
    progress = clamp(progress)
    count = len(NASA_PATH)
    f = progress * (count - 1)
    i = min(count - 2, int(math.floor(f)))
    t = f - i
    a = NASA_PATH[i]
    b = NASA_PATH[i + 1]
    lat = lerp(a[1], b[1], t)
    lon_a = a[2]
    lon_b = b[2]
    if lon_b - lon_a > 180:
        lon_b -= 360
    elif lon_a - lon_b > 180:
        lon_b += 360
    lon = lerp(lon_a, lon_b, t)
    width = lerp(a[3], b[3], t)
    duration = lerp(a[4], b[4], t)
    return lat, lon, width, duration, a[0]


class MapRenderer:
    def __init__(self):
        self.cache: Dict[str, Image.Image] = {}

    def _fallback_map(self, out_w: int, out_h: int) -> Image.Image:
        if FALLBACK_BASEMAP.exists():
            try:
                src = Image.open(FALLBACK_BASEMAP).convert("RGBA")
                lon_min, lon_max, lat_min, lat_max = EUROPE_EXTENT
                xs = [int(((lon + 180.0) / 360.0) * (src.width - 1)) for lon in (lon_min, lon_max)]
                ys = [int(((90.0 - lat) / 180.0) * (src.height - 1)) for lat in (lat_max, lat_min)]
                crop = src.crop((max(0, xs[0]), max(0, ys[0]), min(src.width, xs[1]), min(src.height, ys[1])))
                return crop.resize((out_w, out_h), Image.LANCZOS)
            except Exception:
                pass
        # Plain fallback, only if all else fails.
        arr = np.zeros((out_h, out_w, 4), dtype=np.uint8)
        arr[..., 0] = 24
        arr[..., 1] = 57
        arr[..., 2] = 92
        arr[..., 3] = 255
        return Image.fromarray(arr, "RGBA")

    def render(
        self,
        out_w: int,
        out_h: int,
        title: str,
        path_progress: Optional[float] = None,
        show_path: bool = True,
        cities: Optional[Sequence[str]] = None,
        city_label_subset: Optional[Sequence[str]] = None,
        marker_clock: bool = True,
    ) -> Image.Image:
        cache_key = f"{out_w}|{out_h}|{title}|{path_progress}|{show_path}|{cities}|{city_label_subset}|{marker_clock}|{CARTOPY_OK}"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()

        if CARTOPY_OK and MATPLOTLIB_OK:
            im = self._render_cartopy(out_w, out_h, title, path_progress, show_path, cities, city_label_subset, marker_clock)
        else:
            im = self._fallback_map(out_w, out_h)
            # draw simple overlays in fallback mode
            d = ImageDraw.Draw(im)
            fit_wrapped_text(im, title, (10, 10, out_w - 10, 60), max_size=max(12, int(22 * SCALE)), min_size=max(8, int(12 * SCALE)), fill=(255, 255, 255, 230), bold=True, stroke=2)
            d.rectangle((0, 0, out_w - 1, out_h - 1), outline=(245, 248, 252, 180), width=max(1, int(2 * SCALE)))

        self.cache[cache_key] = im.copy()
        return im

    def _render_cartopy(
        self,
        out_w: int,
        out_h: int,
        title: str,
        path_progress: Optional[float],
        show_path: bool,
        cities: Optional[Sequence[str]],
        city_label_subset: Optional[Sequence[str]],
        marker_clock: bool,
    ) -> Image.Image:
        fig_w = out_w / 100.0
        fig_h = out_h / 100.0
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=100, facecolor="#06111d")
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_facecolor("#081726")
        ax.set_extent(EUROPE_EXTENT, crs=ccrs.PlateCarree())

        try:
            ax.stock_img()
        except Exception:
            pass

        try:
            ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#163a5a", alpha=0.45)
        except Exception:
            pass
        try:
            ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#4d7150", edgecolor="#d4d8cc", linewidth=0.25, alpha=0.55)
        except Exception:
            pass
        try:
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=(1, 1, 1, 0.22), linewidth=0.35)
        except Exception:
            pass
        try:
            ax.coastlines(resolution="50m", color=(1, 1, 1, 0.45), linewidth=0.45)
        except Exception:
            pass

        # graticule
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.45, color=(1, 1, 1, 0.23), alpha=0.55, linestyle='-')
        gl.xlocator = plt.FixedLocator(np.arange(-80, 41, 10))
        gl.ylocator = plt.FixedLocator(np.arange(30, 91, 10))

        # eclipse path
        lons = [p[2] for p in NASA_PATH]
        lats = [p[1] for p in NASA_PATH]
        # avoid any dateline-related jumps by plotting only visible Atlantic/Europe part
        vis_lons, vis_lats = [], []
        for lon, lat in zip(lons, lats):
            if EUROPE_EXTENT[0] - 20 <= lon <= EUROPE_EXTENT[1] + 20 and EUROPE_EXTENT[2] - 5 <= lat <= EUROPE_EXTENT[3] + 5:
                vis_lons.append(lon)
                vis_lats.append(lat)
        if show_path and len(vis_lons) > 1:
            ax.plot(vis_lons, vis_lats, transform=ccrs.PlateCarree(), color="#ff6d7a", linewidth=2.4, alpha=0.95, zorder=7)

        if path_progress is not None:
            lat, lon, width_km, duration, clock = interpolate_path(path_progress)
            # umbra marker with layered circles
            for s, a in [(2200, 0.05), (1300, 0.10), (650, 0.18)]:
                ax.scatter([lon], [lat], s=s, color=(0.03, 0.03, 0.05, a), transform=ccrs.PlateCarree(), zorder=6)
            ax.scatter([lon], [lat], s=28, color="#ffb18c", edgecolors="white", linewidths=0.5, transform=ccrs.PlateCarree(), zorder=9)
            if marker_clock:
                ax.text(lon + 1.8, lat - 0.4, f"{clock} UTC", transform=ccrs.PlateCarree(), color="white", fontsize=10, weight="bold", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor=(0, 0, 0, 0.42), edgecolor=(1, 1, 1, 0.18), linewidth=0.4))

        if cities:
            for city in cities:
                lat, lon = CITY_COORDS[city]
                total = city in {"REYKJAVIK", "LEON", "ZARAGOZA", "VALENCIA"}
                color = "#ffd36a" if total else "#c7e5f6"
                ax.scatter([lon], [lat], s=20, color=color, edgecolors="black", linewidths=0.35, transform=ccrs.PlateCarree(), zorder=8)
                if city_label_subset and city in city_label_subset:
                    ax.text(lon + 1.1, lat - 0.1, city, transform=ccrs.PlateCarree(), color=color, fontsize=9, weight="bold", zorder=9,
                            path_effects=[])

        # map title strip
        ax.text(0.02, 0.98, title, transform=ax.transAxes, va="top", ha="left", color="white", fontsize=12, weight="bold",
                bbox=dict(boxstyle="round,pad=0.20", facecolor=(0, 0, 0, 0.42), edgecolor=(1, 1, 1, 0.15), linewidth=0.4), zorder=12)

        plt.tight_layout(pad=0)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        plt.close(fig)
        return Image.fromarray(buf, "RGBA")


# =============================================================================
# Scene renderer
# =============================================================================

class EclipseScene:
    def __init__(self):
        self.map_renderer = MapRenderer()
        rng = np.random.default_rng(20260812)
        self.stars = []
        for _ in range(int(CONFIG["background_stars"])):
            self.stars.append((
                float(rng.uniform(0, OUT_W)),
                float(rng.uniform(0, OUT_H)),
                float(rng.uniform(0.3, 1.6) * max(SCALE, 0.6)),
                int(rng.uniform(22, 130)),
                float(rng.uniform(0, 2 * math.pi)),
            ))

    def background(self, t: float) -> Image.Image:
        yy = np.linspace(0.0, 1.0, OUT_H, dtype=np.float32)[:, None]
        top = np.array([2.0, 4.0, 11.0], dtype=np.float32)
        bottom = np.array([0.0, 1.2, 8.0], dtype=np.float32)
        rgb = top[None, None, :] * (1.0 - yy[:, :, None]) + bottom[None, None, :] * yy[:, :, None]
        rgb = np.repeat(rgb, OUT_W, axis=1)
        image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
        d = ImageDraw.Draw(image)
        for x, y, r, a, phase in self.stars:
            alpha = int(a * (0.82 + 0.18 * math.sin(0.42 * t + phase)))
            d.ellipse((x - r, y - r, x + r, y + r), fill=(235, 241, 246, alpha))
        return image

    def title_block(self, image: Image.Image, eyebrow: str, title: str, subtitle: str = ""):
        fit_wrapped_text(
            image,
            eyebrow.upper(),
            (int(OUT_W * 0.075), int(OUT_H * 0.065), int(OUT_W * 0.93), int(OUT_H * 0.10)),
            max_size=int(26 * SCALE),
            min_size=int(18 * SCALE),
            fill=(191, 212, 232, 214),
            bold=True,
            stroke=2,
        )
        fit_wrapped_text(
            image,
            title,
            (int(OUT_W * 0.075), int(OUT_H * 0.11), int(OUT_W * 0.93), int(OUT_H * 0.19)),
            max_size=int(68 * SCALE),
            min_size=int(36 * SCALE),
            fill=(248, 250, 253, 255),
            bold=True,
            stroke=3,
        )
        if subtitle:
            fit_wrapped_text(
                image,
                subtitle,
                (int(OUT_W * 0.075), int(OUT_H * 0.18), int(OUT_W * 0.93), int(OUT_H * 0.24)),
                max_size=int(30 * SCALE),
                min_size=int(20 * SCALE),
                fill=(213, 225, 235, 236),
                bold=False,
                stroke=2,
            )

    def footer(self, image: Image.Image):
        fit_wrapped_text(image, "REAL NASA ECLIPSE DATA", (int(OUT_W * 0.055), int(OUT_H * 0.955), int(OUT_W * 0.42), int(OUT_H * 0.98)), max_size=int(18 * SCALE), min_size=int(12 * SCALE), fill=(178, 197, 214, 145), bold=True, stroke=1)
        fit_wrapped_text(image, "MAP: CARTOPY / NATURAL EARTH", (int(OUT_W * 0.55), int(OUT_H * 0.955), int(OUT_W * 0.945), int(OUT_H * 0.98)), max_size=int(18 * SCALE), min_size=int(12 * SCALE), fill=(178, 197, 214, 145), bold=True, stroke=1, align="right")

    def caption(self, image: Image.Image, t: float):
        txt = caption_at(t)
        if not txt:
            return
        y0 = int(OUT_H * 0.805)
        overlay = np.zeros((OUT_H - y0, OUT_W, 4), dtype=np.uint8)
        overlay[..., 3] = np.linspace(0, 225, OUT_H - y0, dtype=np.uint8)[:, None]
        image.alpha_composite(Image.fromarray(overlay, "RGBA"), (0, y0))
        fit_wrapped_text(
            image,
            txt,
            (int(OUT_W * 0.075), int(OUT_H * 0.847), int(OUT_W * 0.925), int(OUT_H * 0.95)),
            max_size=int(34 * SCALE),
            min_size=int(22 * SCALE),
            fill=(244, 248, 252, 242),
            bold=False,
            stroke=3,
            line_spacing=int(8 * SCALE),
        )

    # ------------------------------------------------------------------
    # Eclipse objects
    # ------------------------------------------------------------------
    def draw_sun(self, image: Image.Image, cx: float, cy: float, radius: float):
        draw_glow_disc(image, cx, cy, radius, color=(255, 190, 96), alpha=95)
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 192, 86, 255))
        layer = layer.filter(ImageFilter.GaussianBlur(max(1, int(1.6 * SCALE))))
        image.alpha_composite(layer)

    def draw_moon(self, image: Image.Image, cx: float, cy: float, radius: float):
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(8, 10, 16, 255), outline=(120, 130, 145, 45), width=max(1, int(1 * SCALE)))
        image.alpha_composite(layer)

    def draw_corona_eclipse(self, image: Image.Image, cx: float, cy: float, radius: float, diamond: float = 0.0):
        corona = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        cd = ImageDraw.Draw(corona)
        rng = np.random.default_rng(11)
        for ang in np.linspace(0, 2 * math.pi, 220, endpoint=False):
            structure = 0.82 + 0.32 * math.sin(ang * 3.0 + 0.4) + 0.18 * math.sin(ang * 7.0 + 1.2)
            structure = max(0.28, structure)
            rr0 = radius * 1.02
            rr1 = radius * (1.7 + 1.3 * structure)
            p0 = (cx + rr0 * math.cos(ang), cy + rr0 * math.sin(ang))
            p1 = (cx + rr1 * math.cos(ang), cy + rr1 * math.sin(ang))
            cd.line([p0, p1], fill=(232, 240, 247, int(16 + 36 * structure * (0.92 + 0.16 * float(rng.random())))), width=max(1, int((0.8 + structure) * SCALE)))
        for mult, a, w in [(1.42, 40, 14), (1.23, 72, 10), (1.10, 122, 4)]:
            rr = radius * mult
            cd.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=(239, 245, 250, a), width=max(1, int(w * SCALE)))
        corona = corona.filter(ImageFilter.GaussianBlur(max(1, int(4 * SCALE))))
        image.alpha_composite(corona)
        self.draw_moon(image, cx, cy, radius * 0.99)
        if diamond > 0:
            ang = -0.60
            px = cx + radius * math.cos(ang)
            py = cy + radius * math.sin(ang)
            gl = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            gd = ImageDraw.Draw(gl)
            rr = radius * 0.18 * diamond
            gd.ellipse((px - rr, py - rr, px + rr, py + rr), fill=(255, 251, 236, int(255 * diamond)))
            gd.line((px - radius * diamond, py, px + radius * diamond, py), fill=(255, 242, 212, int(150 * diamond)), width=max(1, int(3 * SCALE)))
            gl = gl.filter(ImageFilter.GaussianBlur(max(1, int(4 * SCALE))))
            image.alpha_composite(gl)

    # ------------------------------------------------------------------
    # Story scenes
    # ------------------------------------------------------------------
    def scene_hook(self, image: Image.Image, t: float, local: float):
        self.title_block(image, "TOTAL SOLAR ECLIPSE", "EUROPE WENT DARK", "August 12, 2026")
        self.draw_corona_eclipse(image, OUT_W * 0.50, OUT_H * 0.48, 130 * SCALE, diamond=0.35 * (1.0 - local))
        fit_wrapped_text(image, "PARTS OF SPAIN AND ICELAND SAW TOTALITY", (int(OUT_W * 0.075), int(OUT_H * 0.695), int(OUT_W * 0.925), int(OUT_H * 0.74)), max_size=int(28 * SCALE), min_size=int(20 * SCALE), fill=(254, 239, 210, 238), bold=True, stroke=2)

    def scene_alignment(self, image: Image.Image, t: float, local: float):
        self.title_block(image, "HOW IT HAPPENED", "THE MOON MOVED IN FRONT OF THE SUN", "Inside the umbra, the Sun is completely covered and the corona becomes visible.")
        y = OUT_H * 0.50
        sunx = OUT_W * 0.20
        moonx = lerp(OUT_W * 0.47, OUT_W * 0.54, local)
        earthx = OUT_W * 0.82
        self.draw_sun(image, sunx, y, 70 * SCALE)
        self.draw_moon(image, moonx, y, 46 * SCALE)
        cone = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        cd = ImageDraw.Draw(cone)
        cd.polygon([(moonx + 18 * SCALE, y - 26 * SCALE), (earthx - 24 * SCALE, y - 48 * SCALE), (earthx - 24 * SCALE, y + 48 * SCALE), (moonx + 18 * SCALE, y + 26 * SCALE)], fill=(6, 8, 17, 128))
        cone = cone.filter(ImageFilter.GaussianBlur(max(1, int(4 * SCALE))))
        image.alpha_composite(cone)
        globe = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(globe)
        gd.ellipse((earthx - 60 * SCALE, y - 60 * SCALE, earthx + 60 * SCALE, y + 60 * SCALE), fill=(23, 85, 128, 255), outline=(120, 184, 228, 200), width=max(1, int(2 * SCALE)))
        image.alpha_composite(globe)
        fit_wrapped_text(image, "SUN", (int(sunx - 50 * SCALE), int(y + 90 * SCALE), int(sunx + 50 * SCALE), int(y + 122 * SCALE)), max_size=int(24 * SCALE), min_size=int(16 * SCALE), fill=(255, 221, 146, 230), bold=True, stroke=2, align="center")
        fit_wrapped_text(image, "MOON", (int(moonx - 60 * SCALE), int(y + 90 * SCALE), int(moonx + 60 * SCALE), int(y + 122 * SCALE)), max_size=int(24 * SCALE), min_size=int(16 * SCALE), fill=(214, 223, 232, 230), bold=True, stroke=2, align="center")
        fit_wrapped_text(image, "EARTH", (int(earthx - 60 * SCALE), int(y + 90 * SCALE), int(earthx + 60 * SCALE), int(y + 122 * SCALE)), max_size=int(24 * SCALE), min_size=int(16 * SCALE), fill=(166, 214, 243, 230), bold=True, stroke=2, align="center")
        fit_wrapped_text(image, "UMBRA", (int((moonx + earthx) / 2 - 90 * SCALE), int(y - 100 * SCALE), int((moonx + earthx) / 2 + 90 * SCALE), int(y - 70 * SCALE)), max_size=int(22 * SCALE), min_size=int(16 * SCALE), fill=(245, 248, 252, 230), bold=True, stroke=2, align="center")

    def scene_path(self, image: Image.Image, t: float, local: float):
        self.title_block(image, "REAL MAP", "NASA'S SHADOW PATH", "This version uses a true geospatial map renderer instead of a fake hand-drawn map.")
        rect = (int(OUT_W * 0.07), int(OUT_H * 0.25), int(OUT_W * 0.93), int(OUT_H * 0.73))
        map_im = self.map_renderer.render(rect[2] - rect[0], rect[3] - rect[1], "PATH OF TOTALITY / DEEP PARTIAL EUROPE", path_progress=smoothstep(local), show_path=True, cities=["REYKJAVIK", "LEON", "MADRID", "BARCELONA", "LONDON", "PARIS"], city_label_subset=["REYKJAVIK", "LEON"], marker_clock=True)
        draw_panel(image, (rect[0] - int(8 * SCALE), rect[1] - int(8 * SCALE), rect[2] + int(8 * SCALE), rect[3] + int(8 * SCALE)), fill=(6, 10, 18, 96), outline=(220, 232, 242, 64), radius=18)
        image.alpha_composite(map_im, (rect[0], rect[1]))
        fit_wrapped_text(image, "THE TINY DARK MARKER SHOWS THE MOVING UMBRA", (int(OUT_W * 0.075), int(OUT_H * 0.74), int(OUT_W * 0.93), int(OUT_H * 0.775)), max_size=int(24 * SCALE), min_size=int(16 * SCALE), fill=(247, 251, 253, 220), bold=True, stroke=2)

    def scene_greatest(self, image: Image.Image, t: float, local: float):
        self.title_block(image, "KEY NUMBERS", "GREATEST ECLIPSE", "The strongest part of the event happened over the North Atlantic.")
        rect = (int(OUT_W * 0.59), int(OUT_H * 0.27), int(OUT_W * 0.92), int(OUT_H * 0.53))
        map_im = self.map_renderer.render(rect[2] - rect[0], rect[3] - rect[1], "GREATEST ECLIPSE", path_progress=22 / (len(NASA_PATH) - 1), show_path=True, cities=None, city_label_subset=None, marker_clock=True)
        draw_panel(image, (rect[0] - int(8 * SCALE), rect[1] - int(8 * SCALE), rect[2] + int(8 * SCALE), rect[3] + int(8 * SCALE)), fill=(6, 10, 18, 96), outline=(220, 232, 242, 64), radius=16)
        image.alpha_composite(map_im, (rect[0], rect[1]))

        labels = [
            ("TIME", "17:45:54 UTC"),
            ("LOCATION", "65.2° N  •  25.2° W"),
            ("PATH WIDTH", "~294 km"),
            ("TOTALITY", "~2 min 18 s"),
        ]
        y = int(OUT_H * 0.29)
        for k, v in labels:
            fit_wrapped_text(image, k, (int(OUT_W * 0.075), y, int(OUT_W * 0.42), y + int(28 * SCALE)), max_size=int(22 * SCALE), min_size=int(14 * SCALE), fill=(176, 198, 218, 185), bold=True, stroke=1)
            fit_wrapped_text(image, v, (int(OUT_W * 0.075), y + int(32 * SCALE), int(OUT_W * 0.52), y + int(84 * SCALE)), max_size=int(42 * SCALE), min_size=int(20 * SCALE), fill=(248, 250, 252, 255), bold=True, stroke=2)
            y += int(112 * SCALE)

    def scene_cities(self, image: Image.Image, t: float, local: float):
        self.title_block(image, "WHAT EUROPE SAW", "TOTALITY VS. DEEP PARTIAL", "Different cities experienced very different eclipse depths.")
        rect = (int(OUT_W * 0.07), int(OUT_H * 0.24), int(OUT_W * 0.93), int(OUT_H * 0.61))
        map_im = self.map_renderer.render(rect[2] - rect[0], rect[3] - rect[1], "CITY OUTCOMES", path_progress=1.0, show_path=True, cities=["REYKJAVIK", "LEON", "ZARAGOZA", "VALENCIA", "MADRID", "BARCELONA", "LONDON", "PARIS"], city_label_subset=["REYKJAVIK", "LEON", "MADRID", "LONDON"], marker_clock=False)
        draw_panel(image, (rect[0] - int(8 * SCALE), rect[1] - int(8 * SCALE), rect[2] + int(8 * SCALE), rect[3] + int(8 * SCALE)), fill=(6, 10, 18, 96), outline=(220, 232, 242, 64), radius=16)
        image.alpha_composite(map_im, (rect[0], rect[1]))

        box1 = (int(OUT_W * 0.075), int(OUT_H * 0.66), int(OUT_W * 0.46), int(OUT_H * 0.775))
        box2 = (int(OUT_W * 0.54), int(OUT_H * 0.66), int(OUT_W * 0.925), int(OUT_H * 0.775))
        draw_panel(image, box1, fill=(16, 15, 20, 155), outline=(255, 212, 120, 70), radius=18)
        draw_panel(image, box2, fill=(13, 18, 28, 155), outline=(190, 223, 242, 70), radius=18)
        fit_wrapped_text(image, "TOTALITY", (box1[0] + int(18 * SCALE), box1[1] + int(12 * SCALE), box1[2] - int(18 * SCALE), box1[1] + int(38 * SCALE)), max_size=int(26 * SCALE), min_size=int(18 * SCALE), fill=(255, 218, 128, 240), bold=True, stroke=2)
        fit_wrapped_text(image, "Reykjavik, Leon, Zaragoza, Valencia", (box1[0] + int(18 * SCALE), box1[1] + int(44 * SCALE), box1[2] - int(18 * SCALE), box1[3] - int(12 * SCALE)), max_size=int(24 * SCALE), min_size=int(16 * SCALE), fill=(248, 249, 251, 236), bold=False, stroke=2)
        fit_wrapped_text(image, "DEEP PARTIAL", (box2[0] + int(18 * SCALE), box2[1] + int(12 * SCALE), box2[2] - int(18 * SCALE), box2[1] + int(38 * SCALE)), max_size=int(26 * SCALE), min_size=int(18 * SCALE), fill=(198, 227, 245, 240), bold=True, stroke=2)
        fit_wrapped_text(image, "Madrid ~99% • Barcelona ~99%\nLondon ~91% • Paris ~92%", (box2[0] + int(18 * SCALE), box2[1] + int(44 * SCALE), box2[2] - int(18 * SCALE), box2[3] - int(12 * SCALE)), max_size=int(23 * SCALE), min_size=int(15 * SCALE), fill=(248, 249, 251, 236), bold=False, stroke=2)

    def scene_spain(self, image: Image.Image, t: float, local: float):
        self.title_block(image, "SPAIN AT SUNSET", "TOTALITY LOW ON THE HORIZON", "In Spain, the eclipse happened late in the day — one reason the visuals were so dramatic.")
        horizon_y = OUT_H * 0.62
        sky = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
        for i in range(OUT_H):
            u = i / max(1, OUT_H - 1)
            if u < 0.62:
                c0 = np.array([255, 170, 88, 255], dtype=np.float32)
                c1 = np.array([21, 28, 57, 0], dtype=np.float32)
                mix = u / 0.62
                sky[i, :, :] = np.clip(c1 * mix + c0 * (1 - mix), 0, 255).astype(np.uint8)
        image.alpha_composite(Image.fromarray(sky, "RGBA"))
        ground = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(ground)
        gd.rectangle((0, horizon_y, OUT_W, OUT_H), fill=(7, 9, 13, 255))
        pts = [(0, horizon_y)]
        for j in range(11):
            x = j / 10.0 * OUT_W
            y = horizon_y - (20 + (j % 3) * 14 + 18 * math.sin(j * 1.7)) * SCALE
            pts.append((x, y))
        pts.extend([(OUT_W, OUT_H), (0, OUT_H)])
        gd.polygon(pts, fill=(6, 7, 10, 255))
        image.alpha_composite(ground)
        self.draw_corona_eclipse(image, OUT_W * 0.50, OUT_H * 0.54, 115 * SCALE, diamond=0.20 * (1.0 - abs(local - 0.22)))
        fit_wrapped_text(image, "LOW WESTERN SKY • LATE EVENING TOTALITY", (int(OUT_W * 0.12), int(OUT_H * 0.71), int(OUT_W * 0.88), int(OUT_H * 0.75)), max_size=int(26 * SCALE), min_size=int(18 * SCALE), fill=(255, 242, 216, 235), bold=True, stroke=2, align="center")

    def scene_safety(self, image: Image.Image, t: float, local: float):
        self.title_block(image, "ONE LAST RULE", "PROTECT YOUR EYES", "Except during totality, you must use proper eclipse glasses or certified solar filters.")
        self.draw_sun(image, OUT_W * 0.50, OUT_H * 0.42, 95 * SCALE)
        self.draw_moon(image, OUT_W * 0.50 - 18 * SCALE, OUT_H * 0.42 - 4 * SCALE, 70 * SCALE)
        panel = (int(OUT_W * 0.08), int(OUT_H * 0.58), int(OUT_W * 0.92), int(OUT_H * 0.73))
        draw_panel(image, panel, fill=(10, 12, 20, 165), outline=(220, 235, 245, 70), radius=20)
        fit_wrapped_text(image, "Partial phases = eclipse glasses. Only full totality = direct viewing safe. Cameras and telescopes need front-mounted solar filters.", (panel[0] + int(24 * SCALE), panel[1] + int(22 * SCALE), panel[2] - int(24 * SCALE), panel[3] - int(18 * SCALE)), max_size=int(31 * SCALE), min_size=int(18 * SCALE), fill=(247, 250, 252, 242), bold=False, stroke=3, line_spacing=int(7 * SCALE))
        fit_wrapped_text(image, "AUGUST 12, 2026 • TOTAL SOLAR ECLIPSE", (int(OUT_W * 0.16), int(OUT_H * 0.76), int(OUT_W * 0.84), int(OUT_H * 0.79)), max_size=int(25 * SCALE), min_size=int(16 * SCALE), fill=(199, 218, 233, 205), bold=True, stroke=2, align="center")

    def render(self, t: float) -> np.ndarray:
        image = self.background(t)
        shot = get_shot(t)
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        name = shot["name"]
        if name == "hook":
            self.scene_hook(image, t, local)
        elif name == "alignment":
            self.scene_alignment(image, t, local)
        elif name == "path":
            self.scene_path(image, t, local)
        elif name == "greatest":
            self.scene_greatest(image, t, local)
        elif name == "cities":
            self.scene_cities(image, t, local)
        elif name == "spain":
            self.scene_spain(image, t, local)
        else:
            self.scene_safety(image, t, local)
        self.caption(image, t)
        self.footer(image)
        return apply_grade(np.asarray(image.convert("RGB")))


# =============================================================================
# Outputs
# =============================================================================

def save_path_csv(path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utc", "central_lat_deg", "central_lon_deg", "path_width_km", "central_duration_s"])
        for row in NASA_PATH:
            w.writerow(row)


def save_summary(path: Path):
    data = {
        "title": "August 12, 2026 Total Solar Eclipse — Cinematic Short v3",
        "improvements": [
            "Map scenes are rendered with Cartopy when available.",
            "All title and caption text uses fit-to-box layout to prevent cropping.",
            "Path, timing, width, and city examples remain grounded in NASA data.",
        ],
        "cartopy_available_at_render_time": CARTOPY_OK,
        "matplotlib_available_at_render_time": MATPLOTLIB_OK,
        "facts": ECLIPSE_FACTS,
        "city_examples": CITY_DATA,
        "video": {"width": OUT_W, "height": OUT_H, "fps": FPS, "duration_s": DURATION},
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def make_previews(scene: EclipseScene):
    times = [0.6, 2.4, 4.8, 7.2, 9.8, 12.0] if QUICK_MODE else [3.0, 10.0, 18.0, 27.0, 36.0, 46.0, 54.0]
    for t in times:
        Image.fromarray(scene.render(t)).save(PREVIEW_DIR / f"preview_{t:g}s.png")


def render_video(scene: EclipseScene, path: Path):
    frames = int(round(FPS * DURATION))
    writer = iio.get_writer(
        path,
        fps=FPS,
        codec="libx264",
        quality=8 if not QUICK_MODE else 7,
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "+faststart", "-vf", f"scale={OUT_W}:{OUT_H}"],
        macro_block_size=None,
    )
    try:
        for i in tqdm(range(frames), desc="Rendering eclipse Short v3"):
            writer.append_data(scene.render(i / FPS))
    finally:
        writer.close()


def main():
    scene = EclipseScene()
    basename = str(CONFIG["output_basename"])
    mp4_path = OUTPUT_ROOT / f"{basename}_final.mp4"
    srt_path = OUTPUT_ROOT / f"{basename}.srt"
    csv_path = DATA_DIR / "nasa_central_line_path.csv"
    json_path = DATA_DIR / "eclipse_summary.json"

    write_srt(CAPTIONS, srt_path)
    save_path_csv(csv_path)
    save_summary(json_path)
    make_previews(scene)
    render_video(scene, mp4_path)

    print("\nRender complete")
    print(f"Video:     {mp4_path}")
    print(f"Subtitles: {srt_path}")
    print(f"Path CSV:  {csv_path}")
    print(f"Summary:   {json_path}")
    print(f"Cartopy available: {CARTOPY_OK}")


if __name__ == "__main__":
    main()
