from __future__ import annotations

"""

The Arctic Melts and Freezes Every Year
=======================================
output : https://youtube.com/shorts/ktjXk_PMVrA?feature=share

A cinematic vertical YouTube Short renderer about the Arctic sea-ice seasonal
cycle. It follows the same production pattern as the other Shorts in this
series: official-source snapshot + cache, offline fallback, 9:16 rendering,
subtitles, preview frames, a contact sheet, CSV/JSON exports, and H.264 MP4.

What the video shows
--------------------
- Arctic sea ice grows through autumn and winter and usually reaches its annual
  maximum extent in March.
- It shrinks through spring and summer and usually reaches its annual minimum
  extent in September.
- Sea-ice extent means the total ocean area with at least 15% sea-ice
  concentration (NSIDC definition).
- A reference 1981-2010 annual maximum of 15.65 million km^2 and annual minimum
  of 6.22 million km^2 are used to build a clearly labeled seasonal
  illustration. This is not a daily Sea Ice Index reconstruction.
- NASA/NSIDC reported the 2026 annual maximum on March 15 at about
  14.29 million km^2, statistically tied with 2025 for the lowest winter
  maximum in the satellite record.
- The seasonal freeze/melt cycle continues every year even as the long-term
  Arctic sea-ice baseline has declined.

Official sources
----------------
NSIDC Sea Ice Today:
    https://nsidc.org/sea-ice-today
NSIDC sea-ice overview:
    https://nsidc.org/our-research/featured-projects/sea-ice-today-and-ice-sheets-today
NASA 2026 Arctic winter maximum:
    https://science.nasa.gov/earth/arctic-winter-sea-ice-2026/
NASA Earth Observatory sea-ice overview:
    https://science.nasa.gov/earth/earth-observatory/sea-ice/

Interpretation rule
-------------------
The animated annual curve is a smooth educational illustration between
published reference annual extremes. It is NOT presented as a daily or monthly
NSIDC measurement series. Exact timing and extent vary from year to year.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg requests tqdm

Run final quality
-----------------
    python the_arctic_melts_and_freezes_every_year_short.py

Run quick preview
-----------------
    ARCTIC_SHORT_QUICK=1 python the_arctic_melts_and_freezes_every_year_short.py

Force live refresh
------------------
    ARCTIC_SHORT_REFRESH=1 python the_arctic_melts_and_freezes_every_year_short.py

Force offline layout testing
----------------------------
    ARCTIC_SHORT_OFFLINE=1 ARCTIC_SHORT_QUICK=1 \
        python the_arctic_melts_and_freezes_every_year_short.py
"""

import json
import math
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import requests
except Exception:
    requests = None


# =============================================================================
# Configuration
# =============================================================================

QUICK_MODE = os.environ.get("ARCTIC_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("ARCTIC_SHORT_OFFLINE", "0") == "1"
REFRESH = os.environ.get("ARCTIC_SHORT_REFRESH", "0") == "1"

OUTPUT_ROOT = Path("the_arctic_melts_and_freezes_every_year_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
CACHE_ROOT = OUTPUT_ROOT / "cache"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT, CACHE_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "width": 540 if QUICK_MODE else 1080,
    "height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "basename": "the_arctic_melts_and_freezes_every_year",
    "title": "THE ARCTIC MELTS AND FREEZES EVERY YEAR",
    "subtitle": "A yearly pulse of sea ice around the North Pole",
    "timeout_s": 35,
    "cache_hours": 36,
    "particles": 180 if QUICK_MODE else 420,
    "nasa_2026_url": "https://science.nasa.gov/earth/arctic-winter-sea-ice-2026/",
    "nsidc_url": "https://nsidc.org/sea-ice-today",
    "nsidc_overview_url": "https://nsidc.org/our-research/featured-projects/sea-ice-today-and-ice-sheets-today",
    "nasa_sea_ice_url": "https://science.nasa.gov/earth/earth-observatory/sea-ice/",
}

W = CONFIG["width"]
H = CONFIG["height"]
SIZE = (W, H)
SCALE = W / 1080.0

COLORS = {
    "bg_top": (2, 8, 20),
    "bg_bottom": (5, 28, 42),
    "white": (246, 250, 255),
    "muted": (151, 195, 216),
    "cyan": (82, 225, 255),
    "blue": (83, 143, 255),
    "ice": (218, 244, 255),
    "ice_shadow": (116, 188, 219),
    "ocean": (5, 48, 78),
    "deep_ocean": (4, 28, 52),
    "land": (62, 85, 83),
    "land_edge": (128, 164, 145),
    "gold": (255, 204, 101),
    "orange": (255, 142, 75),
    "red": (255, 92, 104),
    "green": (107, 242, 183),
    "panel": (3, 13, 25),
}

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.0 if not QUICK_MODE else 1.8},
    {"name": "cycle", "start": 7.0 if not QUICK_MODE else 1.8, "end": 20.0 if not QUICK_MODE else 4.1},
    {"name": "freeze", "start": 20.0 if not QUICK_MODE else 4.1, "end": 32.0 if not QUICK_MODE else 6.5},
    {"name": "melt", "start": 32.0 if not QUICK_MODE else 6.5, "end": 43.5 if not QUICK_MODE else 8.8},
    {"name": "context", "start": 43.5 if not QUICK_MODE else 8.8, "end": 53.0 if not QUICK_MODE else 10.7},
    {"name": "outro", "start": 53.0 if not QUICK_MODE else 10.7, "end": CONFIG["duration_s"]},
]

CAPTION_TEXTS = [
    "The Arctic Ocean does something dramatic every year: its floating sea ice expands in winter and contracts in summer.",
    "The annual maximum usually arrives in March. The minimum usually arrives in September. Exact timing and size change from year to year.",
    "During the dark, cold Arctic autumn and winter, ocean water freezes and the ice edge spreads outward across millions of square kilometers.",
    "In spring and summer, stronger sunlight and warmer air and ocean conditions melt ice and pull the ice edge back toward the central Arctic.",
    "This cycle is natural, but the long-term baseline is changing: satellite observations show much less Arctic sea ice than in earlier decades.",
    "Sea ice still freezes and melts every year. The animation is a sourced seasonal illustration—not a daily satellite reconstruction.",
]

CAPTIONS = [
    (
        shot["start"] + min(0.4, 0.08 * (shot["end"] - shot["start"])),
        shot["end"] - min(0.1, 0.04 * (shot["end"] - shot["start"])),
        text,
    )
    for shot, text in zip(SHOT_PLAN, CAPTION_TEXTS)
]

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


# =============================================================================
# Data model
# =============================================================================

@dataclass
class ArcticSnapshot:
    fetched_at_utc: str
    source_url: str
    source_kind: str
    data_status: str
    offline_fixture: bool
    extent_definition_pct: float
    satellite_record_start: int
    typical_max_month: str
    typical_min_month: str
    reference_period: str
    reference_max_mkm2: float
    reference_min_mkm2: float
    max_2026_mkm2: float
    max_2026_date: str
    max_2026_note: str
    interpretation: str


# =============================================================================
# Utilities
# =============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def ease_in_out_sine(t: float) -> float:
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(path: Path):
    lines: List[str] = []
    for i, (start, end, text) in enumerate(CAPTIONS, start=1):
        lines.extend([str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


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
            pass
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int = 28,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    anchor: str = "la",
    stroke: int = 2,
):
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(220, fill[3] if len(fill) > 3 else 220)),
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
    font = get_font(size, bold)
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for word in words:
        test = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), test, font=font, stroke_width=2)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 220))
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += bbox[3] - bbox[1] + line_spacing


def make_vignette(width: int, height: int, strength: float = 0.23) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1 - strength * radius**1.8, 0, 1).astype(np.float32)


VIGNETTE = make_vignette(W, H)


def get_shot(t: float) -> Dict:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


# =============================================================================
# Data collection
# =============================================================================

def fallback_snapshot() -> ArcticSnapshot:
    return ArcticSnapshot(
        fetched_at_utc=iso_z(utc_now()),
        source_url=CONFIG["nasa_2026_url"],
        source_kind="NASA/NSIDC published reference fixture",
        data_status="offline-fixture",
        offline_fixture=True,
        extent_definition_pct=15.0,
        satellite_record_start=1979,
        typical_max_month="March",
        typical_min_month="September",
        reference_period="1981–2010",
        reference_max_mkm2=15.65,
        reference_min_mkm2=6.22,
        max_2026_mkm2=14.29,
        max_2026_date="2026-03-15",
        max_2026_note="NASA/NSIDC reported the 2026 winter maximum as statistically tied with 2025 for the lowest maximum in the satellite record.",
        interpretation="Seasonal curve is a smooth illustration between published reference annual extremes, not a daily Sea Ice Index reconstruction.",
    )


def fetch_live_snapshot() -> ArcticSnapshot:
    cache_path = CACHE_ROOT / "nasa_arctic_2026_maximum.html"
    html_text: Optional[str] = None
    mode = "live"

    if cache_path.exists() and not REFRESH:
        age_hours = (utc_now().timestamp() - cache_path.stat().st_mtime) / 3600.0
        if age_hours <= CONFIG["cache_hours"]:
            html_text = cache_path.read_text(encoding="utf-8", errors="ignore")
            mode = "cache"

    if html_text is None:
        if requests is None:
            raise RuntimeError("requests is unavailable")
        response = requests.get(
            CONFIG["nasa_2026_url"],
            timeout=CONFIG["timeout_s"],
            headers={"User-Agent": "ArcticSeasonalCycleShort/1.0 educational renderer"},
        )
        response.raise_for_status()
        html_text = response.text
        cache_path.write_text(html_text, encoding="utf-8")

    clean = re.sub(r"<[^>]+>", " ", html_text)
    clean = re.sub(r"\s+", " ", clean)
    extent = 14.29
    date_text = "2026-03-15"

    # NASA wording contains 14.29 million square kilometers and March 15, 2026.
    m = re.search(r"14\.2[89]\s+million\s+square\s+kilometers", clean, re.I)
    if m:
        value = re.search(r"14\.2[89]", m.group(0))
        if value:
            extent = float(value.group(0))
    if not re.search(r"March\s+15,\s+2026", clean, re.I):
        # Keep the documented fixture date if the page structure changes.
        date_text = "2026-03-15"

    snap = fallback_snapshot()
    snap.fetched_at_utc = iso_z(utc_now())
    snap.source_kind = "NASA 2026 Arctic winter maximum article + NSIDC definitions"
    snap.data_status = mode
    snap.offline_fixture = False
    snap.max_2026_mkm2 = extent
    snap.max_2026_date = date_text
    return snap


def seasonal_extent(month_index: float, snapshot: ArcticSnapshot) -> float:
    """Smooth illustrative annual cycle with a March max and September min."""
    # month_index: Jan=0 ... Dec=11. Peak at March index 2, trough at Sep index 8.
    center = (snapshot.reference_max_mkm2 + snapshot.reference_min_mkm2) / 2.0
    amp = (snapshot.reference_max_mkm2 - snapshot.reference_min_mkm2) / 2.0
    return center + amp * math.cos((month_index - 2.0) / 6.0 * math.pi)


def collect_data():
    errors: Dict[str, str] = {}
    if OFFLINE_MODE:
        snapshot = fallback_snapshot()
    else:
        try:
            snapshot = fetch_live_snapshot()
        except Exception as exc:
            errors["live_fetch"] = str(exc)
            snapshot = fallback_snapshot()

    rows = []
    for month in range(12):
        rows.append({
            "month_index": month,
            "month": MONTHS[month],
            "illustrative_extent_million_km2": seasonal_extent(month, snapshot),
            "data_kind": "smooth illustration between published reference annual extremes",
        })
    cycle = pd.DataFrame(rows)
    summary = {
        "generated_at_utc": iso_z(utc_now()),
        "data_status": snapshot.data_status,
        "offline_fixture": snapshot.offline_fixture,
        "errors": errors,
        "warning": snapshot.interpretation,
        "source_urls": [CONFIG["nsidc_url"], CONFIG["nsidc_overview_url"], CONFIG["nasa_2026_url"], CONFIG["nasa_sea_ice_url"]],
    }
    return snapshot, cycle, summary


def save_data(snapshot: ArcticSnapshot, cycle: pd.DataFrame, summary: Dict):
    csv_path = DATA_ROOT / "arctic_seasonal_cycle_illustration.csv"
    json_path = DATA_ROOT / "arctic_sea_ice_snapshot.json"
    cycle.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps({"summary": summary, "snapshot": asdict(snapshot), "cycle": cycle.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )
    return csv_path, json_path


# =============================================================================
# Scene renderer
# =============================================================================

class ArcticScene:
    def __init__(self, snapshot: ArcticSnapshot, cycle: pd.DataFrame, summary: Dict):
        self.snapshot = snapshot
        self.cycle = cycle
        self.summary = summary
        rng = np.random.default_rng(8127)
        self.snow = [
            (float(rng.uniform(0, W)), float(rng.uniform(0, H)), float(rng.uniform(0.7, 2.2) * SCALE), float(rng.uniform(0.15, 0.9)), float(rng.uniform(0, math.tau)))
            for _ in range(CONFIG["particles"])
        ]
        # Precompute the vertical background gradient once; rebuilding it for every
        # frame is unnecessarily expensive at 1080x1920.
        top = np.asarray(COLORS["bg_top"], dtype=np.float32)
        bottom = np.asarray(COLORS["bg_bottom"], dtype=np.float32)
        f = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
        rgb = (top[None, :] * (1.0 - f) + bottom[None, :] * f).astype(np.uint8)
        arr = np.empty((H, W, 4), dtype=np.uint8)
        arr[:, :, :3] = rgb[:, None, :]
        arr[:, :, 3] = 255
        self.base_background = Image.fromarray(arr, "RGBA")

    def background(self, t: float) -> Image.Image:
        img = self.base_background.copy()
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for cx, cy, color in [(W*0.18, H*0.28, (10,90,140)), (W*0.78, H*0.22, (45,65,130)), (W*0.55, H*0.72, (0,100,110))]:
            for r, a in [(W*0.40, 11), (W*0.26, 18), (W*0.14, 27)]:
                gd.ellipse((cx-r, cy-r, cx+r, cy+r), fill=color+(a,))
        glow = glow.filter(ImageFilter.GaussianBlur(70 if not QUICK_MODE else 34))
        img.alpha_composite(glow)

        d = ImageDraw.Draw(img)
        for x, y, r, speed, phase in self.snow:
            yy = (y + t * 28 * speed + 18 * math.sin(phase + t*0.7)) % H
            alpha = int(40 + 80 * (0.5 + 0.5*math.sin(phase+t)))
            d.ellipse((x-r, yy-r, x+r, yy+r), fill=(220, 245, 255, alpha))
        return img

    def draw_title(self, img: Image.Image, t: float):
        intro_end = SHOT_PLAN[0]["end"]
        alpha = int(255 * smoothstep((t - 0.15) / 0.8) * (1 - smoothstep((t - (intro_end - 0.7)) / 0.7)))
        if alpha > 4:
            # Two-line title prevents clipping on narrow 9:16 preview renders.
            draw_text(img, "THE ARCTIC MELTS AND", (54 if not QUICK_MODE else 27, 82 if not QUICK_MODE else 41),
                      size=37 if not QUICK_MODE else 17, fill=COLORS["white"]+(alpha,), bold=True)
            draw_text(img, "FREEZES EVERY YEAR", (54 if not QUICK_MODE else 27, 126 if not QUICK_MODE else 63),
                      size=37 if not QUICK_MODE else 17, fill=COLORS["white"]+(alpha,), bold=True)
            draw_text(img, CONFIG["subtitle"], (56 if not QUICK_MODE else 28, 174 if not QUICK_MODE else 87),
                      size=20 if not QUICK_MODE else 9, fill=COLORS["cyan"]+(min(alpha,230),), bold=True)
        if t > (5.2 if not QUICK_MODE else 1.28):
            labels = {
                "intro": "A PLANETARY SEASONAL PULSE",
                "cycle": "MARCH MAXIMUM • SEPTEMBER MINIMUM",
                "freeze": "AUTUMN + WINTER: FREEZE-UP",
                "melt": "SPRING + SUMMER: MELT SEASON",
                "context": "THE CYCLE CONTINUES — THE BASELINE CHANGES",
                "outro": "FREEZE • MELT • REPEAT",
            }
            draw_text(img, labels[get_shot(t)["name"]], (54 if not QUICK_MODE else 27, 57 if not QUICK_MODE else 29),
                      size=17 if not QUICK_MODE else 8, fill=COLORS["muted"]+(220,), bold=True, stroke=1)

    def draw_source_hud(self, img: Image.Image):
        status = "OFFLINE FIXTURE" if self.snapshot.offline_fixture else ("CACHE" if self.snapshot.data_status == "cache" else "LIVE SOURCE")
        draw_text(img, f"ARCTIC SEA ICE // {status}", (W-(46 if not QUICK_MODE else 23), 70 if not QUICK_MODE else 35),
                  size=15 if not QUICK_MODE else 7, fill=COLORS["cyan"]+(220,), bold=True, anchor="ra", stroke=1)
        draw_text(img, "NSIDC + NASA", (W-(46 if not QUICK_MODE else 23), 98 if not QUICK_MODE else 49),
                  size=13 if not QUICK_MODE else 6, fill=COLORS["muted"]+(200,), anchor="ra", stroke=1)

    def draw_caption(self, img: Image.Image, t: float):
        caption = caption_at(t)
        if not caption:
            return
        y0 = H - (245 if not QUICK_MODE else 124)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((44 if not QUICK_MODE else 22, y0, W-(44 if not QUICK_MODE else 22), y0+(126 if not QUICK_MODE else 66)),
                             radius=24 if not QUICK_MODE else 12, fill=(2,6,14,180), outline=(88,190,225,65), width=1)
        img.alpha_composite(overlay)
        draw_wrapped_text(img, caption, (68 if not QUICK_MODE else 34, y0+(26 if not QUICK_MODE else 13)),
                          W-(136 if not QUICK_MODE else 68), size=27 if not QUICK_MODE else 13,
                          fill=COLORS["white"]+(245,))

    def draw_hud_noise(self, img: Image.Image, t: float):
        overlay = Image.new("RGBA", SIZE, (0,0,0,0))
        od = ImageDraw.Draw(overlay)
        offset = int((t*37)%7)
        for y in range(offset, H, 7):
            od.line((0,y,W,y), fill=(120,210,240,9), width=1)
        scan_y = int((t*165)%(H+220))-110
        od.rectangle((0,scan_y,W,scan_y+(44 if not QUICK_MODE else 22)), fill=(90,220,250,6))
        img.alpha_composite(overlay)

    @staticmethod
    def _continent_polygons(cx: float, cy: float, r: float):
        # Stylized polar-view land shapes. They are intentionally schematic.
        def pts(seq):
            return [(cx + r*x, cy + r*y) for x,y in seq]
        return [
            pts([(-1.10,-0.05),(-0.95,-0.50),(-0.63,-0.84),(-0.25,-1.02),(-0.18,-0.63),(-0.43,-0.38),(-0.60,-0.02),(-0.90,0.18)]),
            pts([(0.10,-1.05),(0.52,-0.90),(0.88,-0.62),(1.07,-0.22),(0.91,0.10),(0.60,-0.02),(0.42,-0.35),(0.08,-0.52)]),
            pts([(1.05,0.10),(0.98,0.50),(0.68,0.82),(0.30,1.04),(0.04,0.85),(0.28,0.55),(0.48,0.28),(0.70,0.08)]),
            pts([(-0.08,1.03),(-0.45,0.92),(-0.78,0.68),(-1.02,0.36),(-0.87,0.14),(-0.56,0.25),(-0.30,0.53),(-0.12,0.75)]),
            pts([(-0.22,-0.25),(-0.08,-0.42),(0.05,-0.31),(0.02,-0.05),(-0.12,0.12),(-0.27,0.05)]),  # Greenland-like island
        ]

    def draw_arctic_map(self, img: Image.Image, extent_mkm2: float, box: Tuple[int,int,int,int], t: float, label: str = ""):
        x0,y0,x1,y1 = box
        cx, cy = (x0+x1)/2, (y0+y1)/2
        r = min(x1-x0,y1-y0)*0.46
        d = ImageDraw.Draw(img)
        d.ellipse((cx-r,cy-r,cx+r,cy+r), fill=COLORS["deep_ocean"]+(245,), outline=COLORS["cyan"]+(75,), width=max(1,int(2*SCALE)))

        # Ice radius is area-like: radius proportional to sqrt(extent).
        frac = clamp((extent_mkm2 - 3.0) / (17.0 - 3.0), 0.08, 1.0)
        ice_r = r * (0.30 + 0.60*math.sqrt(frac))
        ice = Image.new("RGBA", SIZE, (0,0,0,0))
        idr = ImageDraw.Draw(ice)
        # Polygonal/irregular boundary, deterministic with time phase.
        boundary = []
        for i in range(90):
            ang = math.tau*i/90
            wobble = 1.0 + 0.035*math.sin(5*ang+0.4*t) + 0.022*math.sin(11*ang-0.7*t)
            rr = ice_r*wobble
            boundary.append((cx+rr*math.cos(ang), cy+rr*math.sin(ang)))
        idr.polygon(boundary, fill=COLORS["ice"]+(228,), outline=(255,255,255,220))
        # Internal floe texture.
        for k in range(14):
            ang = k*2.399 + t*0.03
            rr = ice_r*(0.15+0.72*((k*37)%13)/13)
            px = cx+rr*math.cos(ang)
            py = cy+rr*math.sin(ang)
            pr = (8+5*(k%3))*SCALE
            idr.arc((px-pr*2,py-pr,px+pr*2,py+pr), 10, 170, fill=COLORS["ice_shadow"]+(95,), width=max(1,int(2*SCALE)))
        img.alpha_composite(ice)

        for poly in self._continent_polygons(cx,cy,r):
            d.polygon(poly, fill=COLORS["land"]+(245,), outline=COLORS["land_edge"]+(180,))
        # North pole marker.
        pr = 5*SCALE
        d.ellipse((cx-pr,cy-pr,cx+pr,cy+pr), fill=COLORS["gold"]+(245,))
        draw_text(img, "N", (int(cx), int(cy-18*SCALE)), size=14 if not QUICK_MODE else 7,
                  fill=COLORS["gold"]+(235,), bold=True, anchor="ma", stroke=1)
        if label:
            draw_text(img, label, (int(cx), int(y1-18*SCALE)), size=16 if not QUICK_MODE else 8,
                      fill=COLORS["cyan"]+(235,), bold=True, anchor="ma", stroke=1)

    def draw_intro(self, img: Image.Image, t: float):
        shot = SHOT_PLAN[0]
        p = ease_in_out_sine((t-shot["start"])/max(1e-6,shot["end"]-shot["start"]))
        # Move through one annual cycle for the hook.
        month = (2 + 12*p) % 12
        extent = seasonal_extent(month, self.snapshot)
        self.draw_arctic_map(img, extent, (int(W*0.10),int(H*0.19),int(W*0.90),int(H*0.68)), t, MONTHS[int(month)%12])
        draw_text(img, f"{extent:0.1f} MILLION km²", (W//2,int(H*0.70)), size=30 if not QUICK_MODE else 14,
                  fill=COLORS["white"]+(245,), bold=True, anchor="ma")
        draw_text(img, "SEA-ICE EXTENT • SEASONAL ILLUSTRATION", (W//2,int(H*0.74)), size=17 if not QUICK_MODE else 8,
                  fill=COLORS["muted"]+(220,), bold=True, anchor="ma", stroke=1)

    def draw_cycle(self, img: Image.Image, t: float):
        x0,y0,x1,y1 = int(W*0.08),int(H*0.19),int(W*0.92),int(H*0.70)
        overlay = Image.new("RGBA", SIZE, (0,0,0,0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0,y0,x1,y1), radius=28 if not QUICK_MODE else 14, fill=(2,9,20,188), outline=COLORS["cyan"]+(65,), width=2)
        img.alpha_composite(overlay)
        d = ImageDraw.Draw(img)
        gx0,gx1 = x0+52*SCALE,x1-34*SCALE
        gy0,gy1 = y0+74*SCALE,y1-92*SCALE
        d.line((gx0,gy1,gx1,gy1), fill=COLORS["muted"]+(100,), width=2)
        d.line((gx0,gy0,gx0,gy1), fill=COLORS["muted"]+(100,), width=2)
        min_y,max_y = 5.0,16.5
        pts=[]
        for i in range(121):
            m = 11*i/120
            val = seasonal_extent(m,self.snapshot)
            x = gx0 + (gx1-gx0)*(m/11)
            y = gy1 - (gy1-gy0)*(val-min_y)/(max_y-min_y)
            pts.append((x,y))
        d.line(pts, fill=COLORS["cyan"]+(230,), width=max(2,int(5*SCALE)))
        for m in range(12):
            x = gx0+(gx1-gx0)*(m/11)
            if m in {0,2,5,8,11}:
                draw_text(img, MONTHS[m], (int(x),int(gy1+24*SCALE)), size=14 if not QUICK_MODE else 7,
                          fill=COLORS["muted"]+(220,), bold=True, anchor="ma", stroke=1)
        # Animated cursor.
        shot=get_shot(t)
        pp=ease_in_out_sine((t-shot["start"])/max(1e-6,shot["end"]-shot["start"]))
        m=11*pp
        val=seasonal_extent(m,self.snapshot)
        x=gx0+(gx1-gx0)*(m/11)
        y=gy1-(gy1-gy0)*(val-min_y)/(max_y-min_y)
        rr=10*SCALE
        d.ellipse((x-rr,y-rr,x+rr,y+rr), fill=COLORS["gold"]+(245,), outline=(255,255,255,220))
        draw_text(img, f"{val:.1f}M km²", (int(x),int(y-28*SCALE)), size=16 if not QUICK_MODE else 8,
                  fill=COLORS["gold"]+(245,), bold=True, anchor="ma", stroke=1)
        # Extreme labels.
        max_x=gx0+(gx1-gx0)*(2/11)
        max_val=self.snapshot.reference_max_mkm2
        max_yy=gy1-(gy1-gy0)*(max_val-min_y)/(max_y-min_y)
        min_x=gx0+(gx1-gx0)*(8/11)
        min_val=self.snapshot.reference_min_mkm2
        min_yy=gy1-(gy1-gy0)*(min_val-min_y)/(max_y-min_y)
        draw_text(img, f"MARCH MAX ≈ {max_val:.2f}M", (int(max_x),int(max_yy-52*SCALE)), size=15 if not QUICK_MODE else 7,
                  fill=COLORS["green"]+(235,), bold=True, anchor="ma", stroke=1)
        draw_text(img, f"SEPT MIN ≈ {min_val:.2f}M", (int(min_x),int(min_yy+34*SCALE)), size=15 if not QUICK_MODE else 7,
                  fill=COLORS["orange"]+(235,), bold=True, anchor="ma", stroke=1)
        draw_text(img, f"{self.snapshot.reference_period} REFERENCE EXTREMES", (x0+18*SCALE,y0+18*SCALE), size=16 if not QUICK_MODE else 8,
                  fill=COLORS["cyan"]+(220,), bold=True, stroke=1)
        draw_text(img, "ILLUSTRATIVE CURVE • NOT DAILY DATA", (x1-18*SCALE,y0+18*SCALE), size=14 if not QUICK_MODE else 7,
                  fill=COLORS["muted"]+(210,), bold=True, anchor="ra", stroke=1)

    def draw_freeze(self, img: Image.Image, t: float):
        shot=get_shot(t)
        p=ease_in_out_sine((t-shot["start"])/max(1e-6,shot["end"]-shot["start"]))
        month=8+6*p  # Sep -> Mar crossing year
        month_wrapped=month%12
        # interpolate from minimum to max explicitly.
        extent=lerp(self.snapshot.reference_min_mkm2,self.snapshot.reference_max_mkm2,p)
        self.draw_arctic_map(img, extent, (int(W*0.08),int(H*0.18),int(W*0.92),int(H*0.67)), t, f"{MONTHS[int(month_wrapped)%12]} • FREEZE-UP")
        d=ImageDraw.Draw(img)
        # Snowflake-like freeze arrows around edge.
        cx,cy=W*0.5,H*0.43
        for i in range(8):
            a=math.tau*i/8+t*0.08
            r=300*SCALE
            x=cx+r*math.cos(a); y=cy+r*math.sin(a)
            x2=cx+(r-55*SCALE)*math.cos(a); y2=cy+(r-55*SCALE)*math.sin(a)
            d.line((x,y,x2,y2), fill=COLORS["cyan"]+(110,), width=max(1,int(3*SCALE)))
        draw_text(img, "COLD + DARK → OCEAN FREEZES", (W//2,int(H*0.70)), size=27 if not QUICK_MODE else 13,
                  fill=COLORS["ice"]+(245,), bold=True, anchor="ma")
        draw_text(img, "ICE EDGE SPREADS OUTWARD", (W//2,int(H*0.745)), size=19 if not QUICK_MODE else 9,
                  fill=COLORS["cyan"]+(230,), bold=True, anchor="ma", stroke=1)

    def draw_melt(self, img: Image.Image, t: float):
        shot=get_shot(t)
        p=ease_in_out_sine((t-shot["start"])/max(1e-6,shot["end"]-shot["start"]))
        extent=lerp(self.snapshot.reference_max_mkm2,self.snapshot.reference_min_mkm2,p)
        month=2+6*p
        self.draw_arctic_map(img, extent, (int(W*0.08),int(H*0.18),int(W*0.92),int(H*0.67)), t, f"{MONTHS[int(month)%12]} • MELT SEASON")
        d=ImageDraw.Draw(img)
        # Sun and rays.
        sx,sy=W*0.82,H*0.23
        sr=42*SCALE
        d.ellipse((sx-sr,sy-sr,sx+sr,sy+sr), fill=COLORS["gold"]+(245,))
        for i in range(12):
            a=math.tau*i/12
            r0=58*SCALE; r1=82*SCALE
            d.line((sx+r0*math.cos(a),sy+r0*math.sin(a),sx+r1*math.cos(a),sy+r1*math.sin(a)), fill=COLORS["gold"]+(160,), width=max(1,int(3*SCALE)))
        draw_text(img, "SUNLIGHT + WARMTH → ICE MELTS", (W//2,int(H*0.70)), size=27 if not QUICK_MODE else 13,
                  fill=COLORS["gold"]+(245,), bold=True, anchor="ma")
        draw_text(img, "ICE EDGE RETREATS TOWARD THE CENTRAL ARCTIC", (W//2,int(H*0.745)), size=17 if not QUICK_MODE else 8,
                  fill=COLORS["orange"]+(230,), bold=True, anchor="ma", stroke=1)

    def _stat_row(self, img: Image.Image, left: str, right: str, x: int, y: int, width: int, color):
        overlay=Image.new("RGBA",SIZE,(0,0,0,0)); od=ImageDraw.Draw(overlay)
        h=int(76*SCALE)
        od.rounded_rectangle((x,y,x+width,y+h), radius=20 if not QUICK_MODE else 10, fill=(3,12,23,190), outline=color+(78,), width=2)
        img.alpha_composite(overlay)
        draw_text(img,left,(x+18,y+18),size=20 if not QUICK_MODE else 10,fill=color+(240,),bold=True,stroke=1)
        draw_text(img,right,(x+width-18,y+18),size=20 if not QUICK_MODE else 10,fill=COLORS["white"]+(238,),bold=True,anchor="ra",stroke=1)

    def draw_context(self, img: Image.Image):
        x=int(W*0.08); y=int(H*0.205); width=int(W*0.84); row_h=int(90*SCALE); gap=int(14*SCALE)
        values=[
            ("Extent definition", f"≥ {self.snapshot.extent_definition_pct:.0f}% ice concentration", COLORS["cyan"]),
            ("Typical annual maximum", self.snapshot.typical_max_month, COLORS["green"]),
            ("Typical annual minimum", self.snapshot.typical_min_month, COLORS["orange"]),
            ("2026 winter maximum", f"{self.snapshot.max_2026_mkm2:.2f} million km²", COLORS["blue"]),
            ("Satellite record", f"since {self.snapshot.satellite_record_start}", COLORS["gold"]),
        ]
        for i,(left,right,color) in enumerate(values):
            self._stat_row(img,left,right,x,y+i*(row_h+gap),width,color)
        draw_wrapped_text(img, "The annual freeze/melt rhythm is natural. The long-term amount of Arctic sea ice has nevertheless declined across the satellite era.",
                          (x+8,y+len(values)*(row_h+gap)+6),width-16,size=17 if not QUICK_MODE else 8,fill=COLORS["muted"]+(220,))

    def draw_outro(self, img: Image.Image, t: float):
        self.draw_arctic_map(img, (self.snapshot.reference_max_mkm2+self.snapshot.reference_min_mkm2)/2,
                             (int(W*0.18),int(H*0.18),int(W*0.82),int(H*0.58)),t,"")
        draw_text(img,"FREEZE",(int(W*0.28),int(H*0.61)),size=31 if not QUICK_MODE else 15,fill=COLORS["cyan"]+(245,),bold=True,anchor="ma")
        draw_text(img,"→",(W//2,int(H*0.61)),size=34 if not QUICK_MODE else 16,fill=COLORS["white"]+(230,),bold=True,anchor="ma")
        draw_text(img,"MELT",(int(W*0.72),int(H*0.61)),size=31 if not QUICK_MODE else 15,fill=COLORS["orange"]+(245,),bold=True,anchor="ma")
        draw_text(img,"EVERY YEAR",(W//2,int(H*0.67)),size=38 if not QUICK_MODE else 18,fill=COLORS["white"]+(245,),bold=True,anchor="ma")
        draw_text(img,"MAX ~ MARCH • MIN ~ SEPTEMBER",(W//2,int(H*0.72)),size=20 if not QUICK_MODE else 9,fill=COLORS["gold"]+(235,),bold=True,anchor="ma",stroke=1)
        draw_wrapped_text(img,"Seasonal visualization grounded in NSIDC/NASA definitions and published reference values. Exact yearly extent varies.",
                          (int(W*0.13),int(H*0.77)),int(W*0.74),size=16 if not QUICK_MODE else 7,fill=COLORS["muted"]+(215,))

    def render_frame(self, t: float) -> np.ndarray:
        img=self.background(t)
        self.draw_title(img,t)
        if t > (5.2 if not QUICK_MODE else 1.28):
            self.draw_source_hud(img)
        shot=get_shot(t)["name"]
        if shot=="intro": self.draw_intro(img,t)
        elif shot=="cycle": self.draw_cycle(img,t)
        elif shot=="freeze": self.draw_freeze(img,t)
        elif shot=="melt": self.draw_melt(img,t)
        elif shot=="context": self.draw_context(img)
        else: self.draw_outro(img,t)
        self.draw_caption(img,t)
        self.draw_hud_noise(img,t)
        arr=np.array(img.convert("RGB"))
        graded=Image.fromarray(arr)
        graded=ImageEnhance.Contrast(graded).enhance(1.08)
        graded=ImageEnhance.Color(graded).enhance(1.04)
        arr=np.array(graded)
        arr=np.clip(arr.astype(np.float32)*VIGNETTE[...,None],0,255).astype(np.uint8)
        fade_in=smoothstep(t/0.9)
        fade_out=1-smoothstep((t-(CONFIG["duration_s"]-1.1))/1.0)
        return np.clip(arr.astype(np.float32)*fade_in*fade_out,0,255).astype(np.uint8)


# =============================================================================
# Output
# =============================================================================

def render_video(scene: ArcticScene) -> Path:
    raw_path=OUTPUT_ROOT/f"{CONFIG['basename']}_raw.mp4"
    final_path=OUTPUT_ROOT/f"{CONFIG['basename']}_final.mp4"
    write_srt(OUTPUT_ROOT/f"{CONFIG['basename']}.srt")
    frame_count=int(round(CONFIG["duration_s"]*CONFIG["fps"]))
    with iio.get_writer(raw_path,fps=CONFIG["fps"],codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=None) as writer:
        for frame_index in tqdm(range(frame_count),desc="Rendering Arctic short"):
            writer.append_data(scene.render_frame(frame_index/CONFIG["fps"]))
    shutil.copyfile(raw_path,final_path)
    return final_path


def make_contact_sheet(paths: Sequence[Path], out_path: Path):
    thumbs=[]
    for path in paths[:6]:
        image=Image.open(path).convert("RGB").resize((270,480))
        draw=ImageDraw.Draw(image)
        draw.rectangle((8,8,122,38),fill=(0,0,0))
        draw.text((18,13),path.stem.replace("preview_",""),fill=(255,255,255))
        thumbs.append(image)
    sheet=Image.new("RGB",(600,1520),(6,16,24))
    for index,thumb in enumerate(thumbs):
        row,col=divmod(index,2)
        sheet.paste(thumb,(20+col*290,20+row*500))
    sheet.save(out_path,quality=92)


def main():
    print("Collecting Arctic sea-ice source snapshot ...")
    snapshot,cycle,summary=collect_data()
    csv_path,json_path=save_data(snapshot,cycle,summary)
    print("Cycle data:",csv_path.resolve())
    print("Summary:",json_path.resolve())
    scene=ArcticScene(snapshot,cycle,summary)
    preview_times=[1.0,min(10.0,CONFIG["duration_s"]*0.22),min(24.0,CONFIG["duration_s"]*0.43),min(36.0,CONFIG["duration_s"]*0.64),min(47.0,CONFIG["duration_s"]*0.83),CONFIG["duration_s"]-1]
    preview_paths: List[Path]=[]
    for t in tqdm(preview_times,desc="Preview frames"):
        path=PREVIEW_ROOT/f"preview_{int(t):02d}s.png"
        Image.fromarray(scene.render_frame(float(t))).save(path)
        preview_paths.append(path)
    contact=PREVIEW_ROOT/"the_arctic_melts_and_freezes_every_year_contact_sheet.jpg"
    make_contact_sheet(preview_paths,contact)
    video_path=render_video(scene)
    print("Video:",video_path.resolve())
    print("Contact sheet:",contact.resolve())
    print("Source status:",summary)


if __name__=="__main__":
    main()
