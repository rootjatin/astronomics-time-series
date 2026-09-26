from __future__ import annotations

"""
The Amazon River Breathes With the Seasons
==========================================

A cinematic vertical YouTube Short renderer about the Amazon flood pulse.
The production pattern intentionally mirrors the attached Arctic Shorts:
9:16 rendering, quick-preview mode, a multi-shot scientific story, animated
captions, preview frames, a contact sheet, JSON/CSV data exports, subtitles,
and an H.264 MP4.

What the video shows
--------------------
- The Amazon basin has a powerful annual flood pulse: rivers rise, spill into
  floodplains and flooded forests, then retreat during the lower-water season.
- Timing varies across this enormous basin. For the central Amazon / Manaus
  region, high water commonly occurs around June and low water around
  October-November; tributaries can peak in different months.
- A classic NASA Amazon-basin summary described main-channel depth increasing
  roughly 30-45 ft (9-14 m) during the high-water part of the cycle.
- The same NASA summary reported a large seasonal expansion of water-covered
  area, from about 110,000 km^2 in an average dry season to about 350,000 km^2
  in the wet season. Treat these as historical basin-scale reference estimates,
  not a live 2026 measurement.
- The seasonal curve in this animation is deliberately schematic. It is not a
  gauge record, forecast, or reconstruction of daily discharge.


Install
-------
    pip install numpy pillow imageio imageio-ffmpeg requests tqdm

Full render
-----------
    python the_amazon_river_breathes_with_the_seasons.py

Quick preview
-------------
    AMAZON_SHORT_QUICK=1 python the_amazon_river_breathes_with_the_seasons.py

Force offline/source-fixture mode
---------------------------------
    AMAZON_SHORT_OFFLINE=1 AMAZON_SHORT_QUICK=1 \\
        python the_amazon_river_breathes_with_the_seasons.py

Notes
-----
This script draws a stylized river/floodplain visualization. It does not claim
that the geometry is a real map of the Amazon basin.
"""

import csv
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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import requests
except Exception:
    requests = None


# =============================================================================
# Configuration
# =============================================================================

QUICK_MODE = os.environ.get("AMAZON_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("AMAZON_SHORT_OFFLINE", "0") == "1"
REFRESH = os.environ.get("AMAZON_SHORT_REFRESH", "0") == "1"

OUTPUT_ROOT = Path("the_amazon_river_breathes_with_the_seasons_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
CACHE_ROOT = OUTPUT_ROOT / "cache"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT, CACHE_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG = {

}

W = int(CONFIG["width"])
H = int(CONFIG["height"])
SIZE = (W, H)
SCALE = W / 1080.0

COLORS = {

}

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]



CAPTIONS = [
    (
        shot["start"] + min(0.35, 0.07 * (shot["end"] - shot["start"])),
        shot["end"] - min(0.10, 0.035 * (shot["end"] - shot["start"])),
        text,
    )
    for shot, text in zip(SHOT_PLAN, CAPTION_TEXTS)
]


# =============================================================================
# Data model and utilities
# =============================================================================

@dataclass
class AmazonSnapshot:
    fetched_at_utc: str
    source_url: str
    source_kind: str
    data_status: str
    offline_fixture: bool
    reference_kind: str
    central_high_month: str
    central_low_months: str
    dry_water_area_km2: int
    wet_water_area_km2: int
    channel_rise_m_min: float
    channel_rise_m_max: float
    timing_note: str
    interpretation: str


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


def get_shot(t: float) -> Dict:
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


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(path: Path) -> Path:
    lines: List[str] = []
    for i, (start, end, text) in enumerate(CAPTIONS, start=1):
        lines.extend([str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


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
            continue
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
    font = get_font(size, bold)
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)
        if bbox[2] - bbox[0] <= max_width:
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
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += bbox[3] - bbox[1] + line_spacing


def make_vignette(width: int, height: int, strength: float = 0.22) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1 - strength * radius**1.8, 0, 1).astype(np.float32)


VIGNETTE = make_vignette(W, H)


# =============================================================================
# Source snapshot and illustrative annual cycle
# =============================================================================


def fallback_snapshot() -> AmazonSnapshot:
    return AmazonSnapshot(
        fetched_at_utc=iso_z(utc_now()),
        source_url=CONFIG["nasa_escape_url"],
        source_kind="NASA Earth Observatory historical Amazon hydrology summary + SGB timing context",
        data_status="offline-fixture",
        offline_fixture=True,
        reference_kind="historical basin-scale reference estimates",
        central_high_month="June",
        central_low_months="October-November",
        dry_water_area_km2=110_000,
        wet_water_area_km2=350_000,
        channel_rise_m_min=9.0,
        channel_rise_m_max=14.0,
        timing_note="Timing differs across tributaries and from west to east; June / Oct-Nov is central-Amazon context, not a basin-wide rule.",
        interpretation="Animated annual curve is schematic and normalized; it is not a live gauge, discharge series, or daily reconstruction.",
    )


def fetch_live_snapshot() -> AmazonSnapshot:
    """Fetch a NASA reference page mainly to verify that the source remains reachable.

    Numerical values intentionally retain the documented historical reference fixture
    rather than pretending that a web page is a live basin measurement feed.
    """
    cache_path = CACHE_ROOT / "nasa_amazon_escape.html"
    html_text: Optional[str] = None
    mode = "live-reference"

    if cache_path.exists() and not REFRESH:
        age_hours = (utc_now().timestamp() - cache_path.stat().st_mtime) / 3600.0
        if age_hours <= CONFIG["cache_hours"]:
            html_text = cache_path.read_text(encoding="utf-8", errors="ignore")
            mode = "cache-reference"

    if html_text is None:
        if requests is None:
            raise RuntimeError("requests is unavailable")
        response = requests.get(
            CONFIG["nasa_escape_url"],
            timeout=CONFIG["timeout_s"],
            headers={"User-Agent": "AmazonFloodPulseShort/1.0 educational renderer"},
        )
        response.raise_for_status()
        html_text = response.text
        cache_path.write_text(html_text, encoding="utf-8")

    clean = re.sub(r"<[^>]+>", " ", html_text)
    clean = re.sub(r"\s+", " ", clean)
    snap = fallback_snapshot()
    snap.fetched_at_utc = iso_z(utc_now())
    snap.data_status = mode
    snap.offline_fixture = False

    # Light-touch validation only. Keep fixture values if wording changes.
    if re.search(r"110[, ]?000", clean, re.I) and re.search(r"350[, ]?000", clean, re.I):
        snap.source_kind = "NASA Earth Observatory page verified live; historical reference values retained"
    else:
        snap.source_kind = "NASA Earth Observatory page reachable; historical fixture retained after wording mismatch"
    return snap


def seasonal_level(month_index: float) -> float:
    """Normalized central-Amazon-style flood pulse: high ~June, low ~Nov.

    This is a smooth educational illustration, not observed monthly gauge data.
    """
    # Cosine peak at June (index 5), trough near November (index 10.5).
    return 0.5 + 0.5 * math.cos((month_index - 5.0) / 5.5 * math.pi)


def floodplain_area_km2(month_index: float, snapshot: AmazonSnapshot) -> float:
    level = clamp(seasonal_level(month_index))
    # Flooded area accelerates near higher stages; this is intentionally illustrative.
    shaped = level ** 1.28
    return lerp(snapshot.dry_water_area_km2, snapshot.wet_water_area_km2, shaped)


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
        rows.append(
            {
                "month_index": month,
                "month": MONTHS[month],
                "illustrative_normalized_level": seasonal_level(month),
                "illustrative_water_area_km2": floodplain_area_km2(month, snapshot),
                "data_kind": "smooth educational flood-pulse illustration",
            }
        )

    summary = {
        "generated_at_utc": iso_z(utc_now()),
        "data_status": snapshot.data_status,
        "offline_fixture": snapshot.offline_fixture,
        "errors": errors,
        "warning": snapshot.interpretation,
        "source_urls": [CONFIG["nasa_flow_url"], CONFIG["nasa_escape_url"], CONFIG["sgb_url"]],
    }
    return snapshot, rows, summary


def save_data(snapshot: AmazonSnapshot, rows: List[Dict], summary: Dict):
    csv_path = DATA_ROOT / "amazon_seasonal_flood_pulse_illustration.csv"
    json_path = DATA_ROOT / "amazon_flood_pulse_snapshot.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps({"summary": summary, "snapshot": asdict(snapshot), "cycle": rows}, indent=2),
        encoding="utf-8",
    )
    return csv_path, json_path


# =============================================================================
# Scene renderer
# =============================================================================

class AmazonScene:
    def __init__(self, snapshot: AmazonSnapshot, rows: List[Dict], summary: Dict):
        self.snapshot = snapshot
        self.rows = rows
        self.summary = summary
        rng = np.random.default_rng(20260921)
        self.motes = [
            (
                float(rng.uniform(0, W)),
                float(rng.uniform(0, H)),
                float(rng.uniform(0.7, 2.2) * SCALE),
                float(rng.uniform(0.15, 0.85)),
                float(rng.uniform(0, math.tau)),
            )
            for _ in range(CONFIG["particles"])
        ]
        self.tree_x = [float(x) for x in rng.uniform(W * 0.03, W * 0.97, 90 if QUICK_MODE else 170)]
        self.tree_phase = [float(x) for x in rng.uniform(0, math.tau, len(self.tree_x))]

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
        for cx, cy, color in [
            (W * 0.18, H * 0.24, (28, 120, 70)),
            (W * 0.80, H * 0.25, (22, 115, 122)),
            (W * 0.52, H * 0.70, (40, 104, 74)),
        ]:
            for r, a in [(W * 0.38, 13), (W * 0.24, 20), (W * 0.12, 30)]:
                gd.ellipse((cx-r, cy-r, cx+r, cy+r), fill=color + (a,))
        glow = glow.filter(ImageFilter.GaussianBlur(68 if not QUICK_MODE else 34))
        img.alpha_composite(glow)

        d = ImageDraw.Draw(img)
        for x, y, r, speed, phase in self.motes:
            xx = (x + 15 * math.sin(phase + t * 0.35)) % W
            yy = (y + t * 12 * speed) % H
            alpha = int(25 + 65 * (0.5 + 0.5 * math.sin(phase + t * 0.9)))
            d.ellipse((xx-r, yy-r, xx+r, yy+r), fill=(184, 242, 194, alpha))
        return img

    def draw_title(self, img: Image.Image, t: float):
        intro_end = SHOT_PLAN[0]["end"]
        alpha = int(255 * smoothstep((t - 0.12) / 0.8) * (1 - smoothstep((t - (intro_end - 0.8)) / 0.7)))
        if alpha > 3:
            x = 54 if not QUICK_MODE else 27
            draw_text(img, "THE AMAZON RIVER", (x, 80 if not QUICK_MODE else 40),
                      size=40 if not QUICK_MODE else 19, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(img, "BREATHES WITH THE SEASONS", (x, 130 if not QUICK_MODE else 65),
                      size=34 if not QUICK_MODE else 16, fill=COLORS["cyan"] + (alpha,), bold=True)
            draw_text(img, CONFIG["subtitle"], (x+2, 178 if not QUICK_MODE else 89),
                      size=19 if not QUICK_MODE else 9, fill=COLORS["muted"] + (min(alpha, 235),), bold=True)

        if t > (5.0 if not QUICK_MODE else 1.18):
            labels = {
                "intro": "A RIVER THAT EXPANDS AND CONTRACTS",
                "pulse": "THE ANNUAL FLOOD PULSE",
                "rise": "RAINS + TRIBUTARIES → RISING WATER",
                "high_water": "HIGH WATER: FOREST BECOMES FLOODPLAIN",
                "retreat": "THE WATERS RETURN TO THE CHANNELS",
                "outro": "RISE • FLOOD • RETREAT • REPEAT",
            }
            draw_text(img, labels[get_shot(t)["name"]], (54 if not QUICK_MODE else 27, 57 if not QUICK_MODE else 29),
                      size=17 if not QUICK_MODE else 8, fill=COLORS["muted"] + (225,), bold=True, stroke=1)

    def draw_source_hud(self, img: Image.Image):
        status = "REFERENCE FIXTURE" if self.snapshot.offline_fixture else self.snapshot.data_status.upper()
        draw_text(img, f"AMAZON FLOOD PULSE // {status}", (W-(46 if not QUICK_MODE else 23), 70 if not QUICK_MODE else 35),
                  size=14 if not QUICK_MODE else 7, fill=COLORS["cyan"] + (220,), bold=True, anchor="ra", stroke=1)
        draw_text(img, "NASA EARTH OBSERVATORY + SGB", (W-(46 if not QUICK_MODE else 23), 97 if not QUICK_MODE else 49),
                  size=12 if not QUICK_MODE else 6, fill=COLORS["muted"] + (205,), anchor="ra", stroke=1)

    def draw_caption(self, img: Image.Image, t: float):
        caption = caption_at(t)
        if not caption:
            return
        y0 = H - (190 if not QUICK_MODE else 95)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle(
            (58 if not QUICK_MODE else 29, y0, W-(58 if not QUICK_MODE else 29), y0+(116 if not QUICK_MODE else 58)),
            radius=20 if not QUICK_MODE else 10,
            fill=(2, 12, 15, 150),
            outline=COLORS["cyan"] + (48,),
            width=1,
        )
        img.alpha_composite(overlay)
        draw_wrapped_text(img, caption, (82 if not QUICK_MODE else 41, y0+(18 if not QUICK_MODE else 9)),
                          W-(164 if not QUICK_MODE else 82), size=30 if not QUICK_MODE else 15,
                          fill=COLORS["white"] + (240,))

    def draw_hud_noise(self, img: Image.Image, t: float):
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        offset = int((t * 31) % 8)
        for y in range(offset, H, 8):
            od.line((0, y, W, y), fill=(130, 238, 214, 7), width=1)
        scan_y = int((t * 140) % (H + 220)) - 110
        od.rectangle((0, scan_y, W, scan_y + (42 if not QUICK_MODE else 21)), fill=(90, 240, 210, 5))
        img.alpha_composite(overlay)

    @staticmethod
    def _river_center_y(x: float, t: float, ymid: float) -> float:
        return ymid + 0.065 * H * math.sin(x / W * math.tau * 1.35 + 0.3) + 0.022 * H * math.sin(x / W * math.tau * 3.1 - t * 0.025)

    def draw_floodplain(self, img: Image.Image, stage: float, t: float, box: Tuple[int, int, int, int], label: str = ""):
        """Stylized plan view of river + floodplain. stage=0 low, stage=1 high."""
        x0, y0, x1, y1 = box
        d = ImageDraw.Draw(img)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0, y0, x1, y1), radius=28 if not QUICK_MODE else 14,
                             fill=(3, 24, 22, 220), outline=COLORS["cyan"] + (58,), width=max(1, int(2*SCALE)))
        img.alpha_composite(overlay)

        # Forest floor / floodplain base.
        d = ImageDraw.Draw(img)
        inner = (x0 + int(16*SCALE), y0 + int(16*SCALE), x1 - int(16*SCALE), y1 - int(16*SCALE))
        d.rounded_rectangle(inner, radius=max(8, int(22*SCALE)), fill=COLORS["forest"] + (245,))

        # Subtle floodplain bands.
        for frac, alpha in [(0.90, 34), (0.72, 27), (0.55, 20)]:
            yy = lerp(inner[1], inner[3], frac)
            d.line((inner[0], yy, inner[2], yy), fill=COLORS["forest2"] + (alpha,), width=max(1, int(2*SCALE)))

        ymid = (y0 + y1) * 0.50
        xs = np.linspace(x0 + 30*SCALE, x1 - 30*SCALE, 180)
        center = [self._river_center_y(float(x), t, ymid) for x in xs]

        channel_half = lerp(18*SCALE, 32*SCALE, stage)
        flood_half = lerp(44*SCALE, 155*SCALE, stage ** 1.15)

        flood_top = []
        flood_bottom = []
        for i, (x, cy) in enumerate(zip(xs, center)):
            wobble = 1 + 0.09 * math.sin(i * 0.19 + t * 0.07) + 0.04 * math.sin(i * 0.53)
            width = flood_half * wobble
            flood_top.append((float(x), float(cy - width)))
            flood_bottom.append((float(x), float(cy + width)))
        flood_poly = flood_top + list(reversed(flood_bottom))
        d.polygon(flood_poly, fill=COLORS["flood"] + (105 if stage < 0.35 else 142,))

        # Oxbow/lake shapes swell with stage.
        for k in range(7):
            fx = lerp(x0 + 90*SCALE, x1 - 90*SCALE, (k + 0.35) / 7)
            cy = self._river_center_y(fx, t, ymid)
            side = -1 if k % 2 == 0 else 1
            fy = cy + side * lerp(65*SCALE, 130*SCALE, 0.35 + 0.08*(k%3))
            rr_x = lerp(20*SCALE, 54*SCALE, stage) * (0.8 + 0.2*((k%3)/2))
            rr_y = rr_x * 0.42
            d.ellipse((fx-rr_x, fy-rr_y, fx+rr_x, fy+rr_y), fill=COLORS["water"] + (110 + int(90*stage),))

        # Main channel.
        top = [(float(x), float(cy-channel_half*(1+0.10*math.sin(i*.23)))) for i, (x, cy) in enumerate(zip(xs, center))]
        bot = [(float(x), float(cy+channel_half*(1+0.08*math.cos(i*.27)))) for i, (x, cy) in enumerate(zip(xs, center))]
        d.polygon(top + list(reversed(bot)), fill=COLORS["deep_water"] + (255,))
        d.line(list(zip(xs, center)), fill=COLORS["water2"] + (130,), width=max(1, int(3*SCALE)))

        # Tributaries.
        for k in range(8):
            fx = lerp(x0 + 80*SCALE, x1 - 80*SCALE, (k + 0.5)/8)
            cy = self._river_center_y(fx, t, ymid)
            side = -1 if k % 2 == 0 else 1
            tx = fx + (-35 if k % 3 == 0 else 35) * SCALE
            ty = cy + side * lerp(105*SCALE, 205*SCALE, 0.35 + 0.07*(k%4))
            d.line((tx, ty, fx, cy), fill=COLORS["water"] + (180,), width=max(2, int(lerp(4, 8, stage)*SCALE)))

        # Trees; more are submerged at higher stage.
        tree_base_y0 = y0 + 45*SCALE
        tree_span = max(1.0, y1-y0-90*SCALE)
        for idx, x in enumerate(self.tree_x):
            if x < x0+25*SCALE or x > x1-25*SCALE:
                continue
            frac = (math.sin(idx*2.17) + 1) * 0.5
            y = tree_base_y0 + frac * tree_span
            cy = self._river_center_y(x, t, ymid)
            dist = abs(y-cy)
            # Keep a visual opening around the main river.
            if dist < channel_half*1.2:
                continue
            trunk_h = (12 + 8 * ((idx*7)%5)/4) * SCALE
            crown = (7 + 3 * ((idx*11)%4)) * SCALE
            water_here = dist < flood_half * (0.92 + 0.08*math.sin(idx))
            trunk_alpha = 120 if water_here else 190
            d.line((x, y, x, y-trunk_h), fill=COLORS["mud"] + (trunk_alpha,), width=max(1, int(2*SCALE)))
            d.ellipse((x-crown, y-trunk_h-crown*0.7, x+crown, y-trunk_h+crown*0.7),
                      fill=COLORS["canopy"] + (165 if water_here else 225,))

        if label:
            draw_text(img, label, (int((x0+x1)/2), int(y1-26*SCALE)), size=17 if not QUICK_MODE else 8,
                      fill=COLORS["cyan"] + (238,), bold=True, anchor="ma", stroke=1)

    def draw_intro(self, img: Image.Image, t: float):
        shot = SHOT_PLAN[0]
        p = ease_in_out_sine((t-shot["start"]) / max(1e-6, shot["end"]-shot["start"]))
        stage = 0.12 + 0.84 * (0.5 - 0.5 * math.cos(math.tau * p))
        self.draw_floodplain(img, stage, t, (int(W*.07), int(H*.20), int(W*.93), int(H*.67)), "FLOODPLAIN WIDTH CHANGES WITH WATER LEVEL")
        area = lerp(self.snapshot.dry_water_area_km2, self.snapshot.wet_water_area_km2, stage**1.25)
        draw_text(img, f"~{area/1000:.0f} THOUSAND km²", (W//2, int(H*.705)), size=31 if not QUICK_MODE else 15,
                  fill=COLORS["white"] + (245,), bold=True, anchor="ma")
        draw_text(img, "HISTORICAL WATER-COVERED AREA REFERENCE", (W//2, int(H*.744)), size=15 if not QUICK_MODE else 7,
                  fill=COLORS["muted"] + (220,), bold=True, anchor="ma", stroke=1)

    def draw_pulse_chart(self, img: Image.Image, t: float):
        x0, y0, x1, y1 = int(W*.08), int(H*.19), int(W*.92), int(H*.70)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0,y0,x1,y1), radius=28 if not QUICK_MODE else 14,
                             fill=(2,17,22,195), outline=COLORS["cyan"]+(62,), width=2)
        img.alpha_composite(overlay)
        d = ImageDraw.Draw(img)
        gx0, gx1 = x0+54*SCALE, x1-35*SCALE
        gy0, gy1 = y0+74*SCALE, y1-92*SCALE
        d.line((gx0,gy1,gx1,gy1), fill=COLORS["muted"]+(100,), width=2)
        d.line((gx0,gy0,gx0,gy1), fill=COLORS["muted"]+(100,), width=2)
        for q, label in [(0.15,"LOW"),(0.50,"MID"),(0.85,"HIGH")]:
            yy = gy1-(gy1-gy0)*q
            d.line((gx0,yy,gx1,yy), fill=COLORS["muted"]+(28,), width=1)
            draw_text(img, label, (int(gx0-10*SCALE),int(yy)), size=12 if not QUICK_MODE else 6,
                      fill=COLORS["muted"]+(185,), anchor="ra", stroke=1)

        pts=[]
        for i in range(181):
            m=11*i/180
            val=seasonal_level(m)
            x=gx0+(gx1-gx0)*(m/11)
            y=gy1-(gy1-gy0)*(0.08+0.84*val)
            pts.append((x,y))
        d.line(pts, fill=COLORS["cyan"]+(235,), width=max(2,int(5*SCALE)))
        for m in range(12):
            x=gx0+(gx1-gx0)*(m/11)
            if m in {0,2,5,8,10}:
                draw_text(img, MONTHS[m], (int(x),int(gy1+24*SCALE)), size=14 if not QUICK_MODE else 7,
                          fill=COLORS["muted"]+(220,), bold=True, anchor="ma", stroke=1)

        shot=get_shot(t)
        p=ease_in_out_sine((t-shot["start"])/max(1e-6,shot["end"]-shot["start"]))
        m=11*p
        val=seasonal_level(m)
        x=gx0+(gx1-gx0)*(m/11)
        y=gy1-(gy1-gy0)*(0.08+0.84*val)
        rr=10*SCALE
        d.ellipse((x-rr,y-rr,x+rr,y+rr), fill=COLORS["gold"]+(245,), outline=COLORS["white"]+(220,))
        area=floodplain_area_km2(m,self.snapshot)
        draw_text(img, f"~{area/1000:.0f}k km²", (int(x),int(y-28*SCALE)), size=15 if not QUICK_MODE else 7,
                  fill=COLORS["gold"]+(245,), bold=True, anchor="ma", stroke=1)

        draw_text(img, "SCHEMATIC CENTRAL-AMAZON SEASONAL PULSE", (x0+18*SCALE,y0+18*SCALE), size=15 if not QUICK_MODE else 7,
                  fill=COLORS["cyan"]+(225,), bold=True, stroke=1)
        draw_text(img, "TIMING VARIES ACROSS TRIBUTARIES", (x1-18*SCALE,y0+18*SCALE), size=13 if not QUICK_MODE else 6,
                  fill=COLORS["muted"]+(205,), bold=True, anchor="ra", stroke=1)
        draw_text(img, "HIGH WATER ~ JUN", (int(gx0+(gx1-gx0)*5/11),int(gy0+14*SCALE)), size=14 if not QUICK_MODE else 7,
                  fill=COLORS["green"]+(235,), bold=True, anchor="ma", stroke=1)
        draw_text(img, "LOW ~ OCT-NOV", (int(gx0+(gx1-gx0)*10/11),int(gy1-18*SCALE)), size=14 if not QUICK_MODE else 7,
                  fill=COLORS["orange"]+(235,), bold=True, anchor="ma", stroke=1)

    def draw_rise(self, img: Image.Image, t: float):
        shot=get_shot(t)
        p=ease_in_out_sine((t-shot["start"])/max(1e-6,shot["end"]-shot["start"]))
        stage=lerp(.18,.96,p)
        self.draw_floodplain(img, stage, t, (int(W*.07),int(H*.20),int(W*.93),int(H*.67)), "RISING WATER • CHANNELS CONNECT TO FLOODPLAINS")
        d=ImageDraw.Draw(img)
        # Rain streaks in upper part.
        for i in range(24 if QUICK_MODE else 54):
            x=(i*83*SCALE + (t*170)%W)%W
            y=H*.18 + ((i*117 + t*220)%(H*.32))
            d.line((x,y,x-7*SCALE,y+22*SCALE), fill=COLORS["cyan"]+(100,), width=max(1,int(2*SCALE)))
        # Inflow arrows.
        for side in (-1,1):
            x0=W*(.12 if side<0 else .88)
            y0=H*.50
            x1=W*.37 if side<0 else W*.63
            y1=H*.46
            d.line((x0,y0,x1,y1), fill=COLORS["gold"]+(180,), width=max(2,int(5*SCALE)))
            s=1 if side<0 else -1
            d.polygon([(x1,y1),(x1-18*SCALE*s,y1-10*SCALE),(x1-18*SCALE*s,y1+10*SCALE)], fill=COLORS["gold"]+(220,))
        draw_text(img, "RAIN ACROSS A CONTINENT-SCALE BASIN", (W//2,int(H*.705)), size=25 if not QUICK_MODE else 12,
                  fill=COLORS["white"]+(245,), bold=True, anchor="ma")
        draw_text(img, "PLUS WATER ARRIVING FROM DISTANT TRIBUTARIES", (W//2,int(H*.744)), size=17 if not QUICK_MODE else 8,
                  fill=COLORS["gold"]+(235,), bold=True, anchor="ma", stroke=1)

    def draw_high_water(self, img: Image.Image, t: float):
        shot=get_shot(t)
        p=ease_in_out_sine((t-shot["start"])/max(1e-6,shot["end"]-shot["start"]))
        stage=.92+.05*math.sin(p*math.pi)
        self.draw_floodplain(img, stage, t, (int(W*.07),int(H*.20),int(W*.93),int(H*.67)), "HIGH WATER • VÁRZEA / FLOODED FOREST")
        d=ImageDraw.Draw(img)
        # Fish-like glyphs moving through the floodplain.
        for k in range(9):
            x=W*(.16+.07*k)+25*SCALE*math.sin(t*.5+k)
            y=H*(.34+.025*(k%4))+18*SCALE*math.sin(t*.8+k*1.7)
            rr=8*SCALE
            d.ellipse((x-rr*1.5,y-rr*.7,x+rr*1.2,y+rr*.7), fill=COLORS["lime"]+(140,))
            d.polygon([(x+rr*1.2,y),(x+rr*2.1,y-rr*.8),(x+rr*2.1,y+rr*.8)], fill=COLORS["lime"]+(120,))
        draw_text(img, f"CHANNEL DEPTH CAN RISE ~{self.snapshot.channel_rise_m_min:.0f}-{self.snapshot.channel_rise_m_max:.0f} m", (W//2,int(H*.705)),
                  size=25 if not QUICK_MODE else 12, fill=COLORS["cyan"]+(245,), bold=True, anchor="ma")
        draw_text(img, "HISTORICAL NASA BASIN-SCALE SUMMARY", (W//2,int(H*.744)), size=15 if not QUICK_MODE else 7,
                  fill=COLORS["muted"]+(220,), bold=True, anchor="ma", stroke=1)

    def _stat_row(self, img: Image.Image, left: str, right: str, x: int, y: int, width: int, color):
        overlay=Image.new("RGBA",SIZE,(0,0,0,0))
        od=ImageDraw.Draw(overlay)
        h=int(76*SCALE)
        od.rounded_rectangle((x,y,x+width,y+h), radius=20 if not QUICK_MODE else 10,
                             fill=(3,18,22,194), outline=color+(72,), width=2)
        img.alpha_composite(overlay)
        draw_text(img,left,(x+18,y+18),size=18 if not QUICK_MODE else 9,fill=color+(240,),bold=True,stroke=1)
        draw_text(img,right,(x+width-18,y+18),size=18 if not QUICK_MODE else 9,fill=COLORS["white"]+(238,),bold=True,anchor="ra",stroke=1)

    def draw_retreat(self, img: Image.Image, t: float):
        shot=get_shot(t)
        p=ease_in_out_sine((t-shot["start"])/max(1e-6,shot["end"]-shot["start"]))
        stage=lerp(.96,.14,p)
        self.draw_floodplain(img, stage, t, (int(W*.07),int(H*.18),int(W*.93),int(H*.61)), "FALLING WATER • BEACHES AND BARS REAPPEAR")
        x=int(W*.09); y=int(H*.65); width=int(W*.82); row_h=int(68*SCALE); gap=int(9*SCALE)
        vals=[
            ("Dry-season water area", f"~{self.snapshot.dry_water_area_km2/1000:.0f}k km²", COLORS["orange"]),
            ("Wet-season water area", f"~{self.snapshot.wet_water_area_km2/1000:.0f}k km²", COLORS["cyan"]),
            ("Central high-water timing", f"~ {self.snapshot.central_high_month}", COLORS["green"]),
        ]
        for i,(left,right,color) in enumerate(vals):
            self._stat_row(img,left,right,x,y+i*(row_h+gap),width,color)

    def draw_outro(self, img: Image.Image, t: float):
        phase=((t-SHOT_PLAN[-1]["start"])/max(1e-6,SHOT_PLAN[-1]["end"]-SHOT_PLAN[-1]["start"]))
        stage=.5+.38*math.sin(phase*math.tau-math.pi/2)
        stage=clamp(stage,.12,.9)
        self.draw_floodplain(img,stage,t,(int(W*.13),int(H*.18),int(W*.87),int(H*.57)),"")
        draw_text(img,"RISE",(int(W*.19),int(H*.61)),size=28 if not QUICK_MODE else 14,fill=COLORS["cyan"]+(245,),bold=True,anchor="ma")
        draw_text(img,"→",(int(W*.34),int(H*.61)),size=30 if not QUICK_MODE else 15,fill=COLORS["white"]+(225,),bold=True,anchor="ma")
        draw_text(img,"FLOOD",(int(W*.50),int(H*.61)),size=28 if not QUICK_MODE else 14,fill=COLORS["green"]+(245,),bold=True,anchor="ma")
        draw_text(img,"→",(int(W*.66),int(H*.61)),size=30 if not QUICK_MODE else 15,fill=COLORS["white"]+(225,),bold=True,anchor="ma")
        draw_text(img,"RETREAT",(int(W*.82),int(H*.61)),size=28 if not QUICK_MODE else 14,fill=COLORS["orange"]+(245,),bold=True,anchor="ma")
        draw_text(img,"EVERY YEAR",(W//2,int(H*.675)),size=40 if not QUICK_MODE else 19,fill=COLORS["white"]+(245,),bold=True,anchor="ma")
        draw_text(img,"THE FLOOD PULSE CONNECTS RIVER + FOREST",(W//2,int(H*.72)),size=19 if not QUICK_MODE else 9,fill=COLORS["gold"]+(235,),bold=True,anchor="ma",stroke=1)
        draw_wrapped_text(img,"Stylized visualization grounded in NASA Earth Observatory references and central-Amazon seasonal timing. Exact stage, discharge, and timing vary by river and year.",
                          (int(W*.12),int(H*.765)),int(W*.76),size=15 if not QUICK_MODE else 7,fill=COLORS["muted"]+(215,))

    def render_frame(self, t: float) -> np.ndarray:
        img=self.background(t)
        self.draw_title(img,t)
        if t > (5.0 if not QUICK_MODE else 1.18):
            self.draw_source_hud(img)
        shot=get_shot(t)["name"]
        if shot=="intro":
            self.draw_intro(img,t)
        elif shot=="pulse":
            self.draw_pulse_chart(img,t)
        elif shot=="rise":
            self.draw_rise(img,t)
        elif shot=="high_water":
            self.draw_high_water(img,t)
        elif shot=="retreat":
            self.draw_retreat(img,t)
        else:
            self.draw_outro(img,t)
        self.draw_caption(img,t)
        self.draw_hud_noise(img,t)

        arr=np.array(img.convert("RGB"))
        graded=Image.fromarray(arr)
        graded=ImageEnhance.Contrast(graded).enhance(1.08)
        graded=ImageEnhance.Color(graded).enhance(1.06)
        arr=np.array(graded)
        arr=np.clip(arr.astype(np.float32)*VIGNETTE[...,None],0,255).astype(np.uint8)
        fade_in=smoothstep(t/0.9)
        fade_out=1-smoothstep((t-(CONFIG["duration_s"]-1.05))/0.95)
        return np.clip(arr.astype(np.float32)*fade_in*fade_out,0,255).astype(np.uint8)


# =============================================================================
# Output
# =============================================================================


def render_video(scene: AmazonScene) -> Path:
    raw_path=OUTPUT_ROOT/f"{CONFIG['basename']}_raw.mp4"
    final_path=OUTPUT_ROOT/f"{CONFIG['basename']}_final.mp4"
    write_srt(OUTPUT_ROOT/f"{CONFIG['basename']}_subtitles.srt")
    frame_count=int(round(CONFIG["duration_s"]*CONFIG["fps"]))
    with iio.get_writer(
        raw_path,
        fps=CONFIG["fps"],
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for frame_index in tqdm(range(frame_count),desc="Rendering Amazon short"):
            writer.append_data(scene.render_frame(frame_index/CONFIG["fps"]))
    shutil.copyfile(raw_path,final_path)
    return final_path


def make_contact_sheet(paths: Sequence[Path], out_path: Path):
    thumbs=[]
    for path in paths[:6]:
        image=Image.open(path).convert("RGB").resize((270,480))
        draw=ImageDraw.Draw(image)
        draw.rectangle((8,8,136,38),fill=(0,0,0))
        draw.text((18,13),path.stem.replace("preview_",""),fill=(255,255,255))
        thumbs.append(image)
    sheet=Image.new("RGB",(600,1520),(5,22,24))
    for index,thumb in enumerate(thumbs):
        row,col=divmod(index,2)
        sheet.paste(thumb,(20+col*290,20+row*500))
    sheet.save(out_path,quality=92)



# =============================================================================
# YouTube title / description sidecar
# =============================================================================

YOUTUBE_TITLE = 'The Amazon River Breathes With the Seasons 🌊🌳 #rootjatin'
YOUTUBE_DESCRIPTION = "Watch the Amazon's annual flood pulse expand rivers into floodplains and flooded forest, then retreat again. The timing shown is grounded in central-Amazon context near Manaus, where high water commonly arrives around June and seasonal lows often occur around October or November, while conditions vary across the basin. This is a sourced seasonal illustration, not a live gauge, exact basin map, or daily discharge record."
YOUTUBE_HASHTAGS = '#rootjatin #AmazonRiver #Amazon #EarthScience #Hydrology #Nature #Geography #Science'

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
