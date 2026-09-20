from __future__ import annotations

"""
Greenland Is Losing Ice
=======================

A cinematic vertical YouTube Short renderer about Greenland ice-sheet mass loss.
The structure intentionally mirrors a data-driven Shorts workflow: live-source
fetch with caching, an offline fallback, a 9:16 animated renderer, captions/SRT,
preview frames, a contact sheet, and MP4 export.

What the video shows
--------------------
- NASA's latest published Greenland mass-loss rate from the GRACE / GRACE-FO
  ice-sheet indicator page when it can be parsed successfully.
- A rate-based 2002 -> latest-year trend illustration.
- A stylized Greenland silhouette showing that losses are strongest around many
  lower-elevation coastal areas, especially West Greenland.
- Two major loss pathways: surface melt/runoff and glacier discharge/calving.
- Key facts: average mass-loss rate, sea-level contribution, observing missions,
  and the time span of the satellite record.

Important interpretation note
-----------------------------
The line chart in this video is a *rate-based educational illustration* made from
NASA's published average rate. It is NOT a reconstruction of the monthly GRACE
mass-anomaly time series. NASA Earthdata authentication is now required for the
underlying downloadable ice-sheet data, so this script deliberately avoids
pretending it has monthly values it did not retrieve.

Official sources
----------------
NASA Earth Indicator — Ice Sheets:
    https://science.nasa.gov/earth/explore/earth-indicators/ice-sheets/
NASA Scientific Visualization Studio — Greenland Ice Mass Loss 2002-2025:
    https://svs.gsfc.nasa.gov/31156/
NASA GRACE / GRACE-FO:
    https://grace.jpl.nasa.gov/

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg requests tqdm

Run final quality
-----------------
    python greenland_is_losing_ice_short.py

Run quick preview
-----------------
    GREENLAND_SHORT_QUICK=1 python greenland_is_losing_ice_short.py

Force live refresh
------------------
    GREENLAND_SHORT_REFRESH=1 python greenland_is_losing_ice_short.py

Force offline layout testing
----------------------------
    GREENLAND_SHORT_OFFLINE=1 GREENLAND_SHORT_QUICK=1 \
        python greenland_is_losing_ice_short.py
"""

import html
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

QUICK_MODE = os.environ.get("GREENLAND_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("GREENLAND_SHORT_OFFLINE", "0") == "1"
REFRESH = os.environ.get("GREENLAND_SHORT_REFRESH", "0") == "1"

OUTPUT_ROOT = Path("greenland_is_losing_ice_output")
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
    "basename": "greenland_is_losing_ice",
    "title": "GREENLAND IS LOSING ICE",
    "subtitle": "What NASA's GRACE satellites reveal since 2002",
    "timeout_s": 35,
    "particles": 240 if QUICK_MODE else 520,
    "cache_hours": 36,
    "indicator_url": "https://science.nasa.gov/earth/explore/earth-indicators/ice-sheets/",
    "svs_url": "https://svs.gsfc.nasa.gov/31156/",
    "grace_url": "https://grace.jpl.nasa.gov/",
}

W = CONFIG["width"]
H = CONFIG["height"]
SIZE = (W, H)
SCALE = W / 1080.0

COLORS = {
    "bg_top": (3, 11, 24),
    "bg_bottom": (3, 27, 39),
    "white": (246, 251, 255),
    "muted": (155, 199, 216),
    "ice": (213, 243, 255),
    "ice_shadow": (103, 183, 216),
    "cyan": (91, 225, 255),
    "blue": (89, 149, 255),
    "teal": (83, 245, 213),
    "gold": (255, 202, 100),
    "orange": (255, 145, 83),
    "red": (255, 91, 105),
    "ocean": (7, 48, 72),
    "panel": (3, 13, 25),
}

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.0 if not QUICK_MODE else 1.8},
    {"name": "trend", "start": 7.0 if not QUICK_MODE else 1.8, "end": 20.0 if not QUICK_MODE else 4.1},
    {"name": "coasts", "start": 20.0 if not QUICK_MODE else 4.1, "end": 32.5 if not QUICK_MODE else 6.6},
    {"name": "processes", "start": 32.5 if not QUICK_MODE else 6.6, "end": 43.5 if not QUICK_MODE else 8.8},
    {"name": "stats", "start": 43.5 if not QUICK_MODE else 8.8, "end": 53.0 if not QUICK_MODE else 10.7},
    {"name": "outro", "start": 53.0 if not QUICK_MODE else 10.7, "end": CONFIG["duration_s"]},
]

CAPTION_TEXTS = [
    "Greenland holds an enormous ice sheet, but satellites show it has been losing mass since 2002.",
    "NASA's latest GRACE and GRACE Follow-On summary puts the long-term loss at roughly a quarter-trillion metric tons of ice per year.",
    "The strongest losses are concentrated around many lower-elevation coastal areas, with especially large decreases along West Greenland.",
    "Ice leaves Greenland in two major ways: surface meltwater runs off, and fast-moving outlet glaciers discharge ice into the ocean.",
    "That lost land ice adds water to the ocean. The satellite record is built from GRACE and the GRACE Follow-On mission.",
    "This animation is data-grounded, but the trend line is an illustration of NASA's published average rate—not the monthly GRACE dataset.",
]

# Keep captions synchronized in both the 58-second final render and 12-second
# quick-preview render by deriving their timing directly from SHOT_PLAN.
CAPTIONS = [
    (shot["start"] + min(0.4, 0.08 * (shot["end"] - shot["start"])),
     shot["end"] - min(0.1, 0.04 * (shot["end"] - shot["start"])),
     text)
    for shot, text in zip(SHOT_PLAN, CAPTION_TEXTS)
]


# =============================================================================
# Data model
# =============================================================================

@dataclass
class GreenlandSnapshot:
    fetched_at_utc: str
    source_url: str
    source_kind: str
    data_status: str
    offline_fixture: bool
    start_year: int
    end_year: int
    loss_rate_gt_per_year: float
    sea_level_mm_per_year: float
    mission_primary: str
    mission_follow_on: str
    interpretation: str
    coastal_note: str


@dataclass
class TimelineEvent:
    year: int
    label: str
    note: str


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


def clip_text(text: str, n: int = 46) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= n else clean[: n - 1] + "…"


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


def make_vignette(width: int, height: int, strength: float = 0.25) -> np.ndarray:
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


def strip_html(raw: str) -> str:
    raw = re.sub(r"<script\b[^>]*>.*?</script>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<style\b[^>]*>.*?</style>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    return re.sub(r"\s+", " ", raw).strip()


# =============================================================================
# Data collection
# =============================================================================

def fallback_snapshot() -> GreenlandSnapshot:
    return GreenlandSnapshot(
        fetched_at_utc=iso_z(utc_now()),
        source_url=CONFIG["svs_url"],
        source_kind="NASA 2002-2025 published summary fixture",
        data_status="offline-fixture",
        offline_fixture=True,
        start_year=2002,
        end_year=2025,
        loss_rate_gt_per_year=264.0,
        sea_level_mm_per_year=0.8,
        mission_primary="GRACE (2002-2017)",
        mission_follow_on="GRACE-FO (2018-present)",
        interpretation="Rate-based trend illustration; not monthly GRACE mass-anomaly observations.",
        coastal_note="NASA visualizations show the largest mass decreases around lower-elevation coastal Greenland, especially the west coast.",
    )


def fetch_live_snapshot() -> GreenlandSnapshot:
    if requests is None:
        raise RuntimeError("requests is unavailable")

    cache_path = CACHE_ROOT / "nasa_ice_sheets.html"
    raw: Optional[str] = None
    mode = "live"

    if cache_path.exists() and not REFRESH:
        age_hours = (utc_now().timestamp() - cache_path.stat().st_mtime) / 3600.0
        if age_hours <= CONFIG["cache_hours"]:
            raw = cache_path.read_text(encoding="utf-8", errors="ignore")
            mode = "cache"

    if raw is None:
        response = requests.get(
            CONFIG["indicator_url"],
            timeout=CONFIG["timeout_s"],
            headers={"User-Agent": "GreenlandIceShort/1.0 educational renderer"},
        )
        response.raise_for_status()
        raw = response.text
        cache_path.write_text(raw, encoding="utf-8")

    text = strip_html(raw)

    # Best-case match for the detailed NASA summary paragraph.
    pattern = re.compile(
        r"between\s+2002\s+and\s+(20\d{2}).{0,900}?Greenland\s+shed\s+approximately\s+([0-9.]+)\s+gigatons"
        r".{0,500}?sea\s+level\s+to\s+rise\s+by.{0,100}?\(?([0-9.]+)\s+millimeters\)?\s+per\s+year",
        re.I,
    )
    match = pattern.search(text)

    if match:
        end_year = int(match.group(1))
        rate = float(match.group(2))
        sea_mm = float(match.group(3))
    else:
        # Secondary parser: find the dedicated Greenland rate and keep the
        # sea-level fallback if the page wording changed.
        rate_matches = [float(v) for v in re.findall(r"Greenland.{0,220}?(\d{3}(?:\.\d+)?)\s+(?:billion metric tons|gigatons)", text, flags=re.I)]
        rate_matches = [v for v in rate_matches if 200 <= v <= 350]
        if not rate_matches:
            raise RuntimeError("Could not parse a Greenland loss rate from NASA page")
        rate = rate_matches[-1]
        year_matches = [int(v) for v in re.findall(r"between\s+2002\s+and\s+(20\d{2})", text, flags=re.I)]
        end_year = max(year_matches) if year_matches else utc_now().year - 1
        sea_mm = 0.8

    return GreenlandSnapshot(
        fetched_at_utc=iso_z(utc_now()),
        source_url=CONFIG["indicator_url"],
        source_kind="NASA Earth Indicator — Ice Sheets",
        data_status=mode,
        offline_fixture=False,
        start_year=2002,
        end_year=end_year,
        loss_rate_gt_per_year=rate,
        sea_level_mm_per_year=sea_mm,
        mission_primary="GRACE (2002-2017)",
        mission_follow_on="GRACE-FO (2018-present)",
        interpretation="Rate-based trend illustration; not monthly GRACE mass-anomaly observations.",
        coastal_note="NASA visualizations show the largest mass decreases around lower-elevation coastal Greenland, especially the west coast.",
    )


def mission_timeline(snapshot: GreenlandSnapshot) -> List[TimelineEvent]:
    return [
        TimelineEvent(2002, "GRACE begins", "Twin satellites start tracking changes in Earth's gravity field."),
        TimelineEvent(2017, "GRACE ends", "The original GRACE mission concludes."),
        TimelineEvent(2018, "GRACE-FO begins", "Follow-On measurements resume the ice-mass record."),
        TimelineEvent(snapshot.end_year, f"Record through {snapshot.end_year}", "NASA's published indicator continues the long-term Greenland mass-loss record."),
    ]


def collect_data() -> Tuple[GreenlandSnapshot, pd.DataFrame, List[TimelineEvent], Dict]:
    errors: Dict[str, str] = {}
    if OFFLINE_MODE:
        snapshot = fallback_snapshot()
    else:
        try:
            snapshot = fetch_live_snapshot()
        except Exception as exc:
            errors["nasa_live_fetch"] = str(exc)
            snapshot = fallback_snapshot()

    years = np.arange(snapshot.start_year, snapshot.end_year + 1, dtype=int)
    # Educational trend from the published average rate. The negative sign means
    # mass is lower relative to the 2002 baseline.
    cumulative = -(years - snapshot.start_year) * snapshot.loss_rate_gt_per_year
    trend = pd.DataFrame({
        "year": years,
        "rate_based_mass_change_gt": cumulative,
        "note": "Illustrative linear trend from NASA published average rate; not monthly GRACE observations.",
    })

    events = mission_timeline(snapshot)
    summary = {
        "generated_at_utc": iso_z(utc_now()),
        "data_status": snapshot.data_status,
        "offline_fixture": snapshot.offline_fixture,
        "start_year": snapshot.start_year,
        "end_year": snapshot.end_year,
        "loss_rate_gt_per_year": snapshot.loss_rate_gt_per_year,
        "sea_level_mm_per_year": snapshot.sea_level_mm_per_year,
        "illustrated_mass_change_gt": float(cumulative[-1]) if len(cumulative) else 0.0,
        "errors": errors,
        "warning": snapshot.interpretation,
    }
    return snapshot, trend, events, summary


def save_data(snapshot: GreenlandSnapshot, trend: pd.DataFrame, events: Sequence[TimelineEvent], summary: Dict) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "greenland_rate_based_trend.csv"
    json_path = DATA_ROOT / "greenland_snapshot_summary.json"
    trend.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "snapshot": asdict(snapshot),
                "timeline": [asdict(e) for e in events],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path, json_path


# =============================================================================
# Scene renderer
# =============================================================================

class GreenlandScene:
    def __init__(self, snapshot: GreenlandSnapshot, trend: pd.DataFrame, events: Sequence[TimelineEvent], summary: Dict):
        self.snapshot = snapshot
        self.trend = trend.copy()
        self.events = list(events)
        self.summary = summary
        self.particles = self._make_particles(CONFIG["particles"], seed=8421)
        self.bg_base = self._build_gradient_background()
        self.chart_box = (int(W * 0.08), int(H * 0.23), int(W * 0.92), int(H * 0.66))
        self.greenland_poly = self._greenland_polygon()

    @staticmethod
    def _make_particles(n: int, seed: int):
        rng = np.random.default_rng(seed)
        return [
            (
                float(rng.uniform(0, W)),
                float(rng.uniform(0, H)),
                float(rng.uniform(0.5, 2.2) * SCALE),
                float(rng.uniform(7, 28) * SCALE),
                float(rng.uniform(0, math.tau)),
                int(rng.integers(25, 115)),
            )
            for _ in range(n)
        ]

    def _build_gradient_background(self) -> Image.Image:
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        top = np.array(COLORS["bg_top"], dtype=float)
        bottom = np.array(COLORS["bg_bottom"], dtype=float)
        for y in range(H):
            p = y / max(1, H - 1)
            rgb = top * (1 - p) + bottom * p
            arr[y, :, :3] = rgb.astype(np.uint8)
            arr[y, :, 3] = 255
        return Image.fromarray(arr, mode="RGBA")

    def _greenland_polygon(self) -> List[Tuple[int, int]]:
        # Deliberately stylized silhouette. This is not a geospatial boundary.
        pts = [
            (0.52, 0.05), (0.60, 0.10), (0.64, 0.16), (0.61, 0.22),
            (0.67, 0.28), (0.63, 0.36), (0.66, 0.44), (0.61, 0.52),
            (0.64, 0.61), (0.58, 0.70), (0.55, 0.80), (0.48, 0.93),
            (0.42, 0.87), (0.39, 0.76), (0.34, 0.67), (0.36, 0.58),
            (0.31, 0.49), (0.34, 0.40), (0.30, 0.31), (0.35, 0.23),
            (0.38, 0.14), (0.44, 0.09),
        ]
        x0, y0, x1, y1 = int(W * 0.16), int(H * 0.20), int(W * 0.84), int(H * 0.73)
        return [
            (int(x0 + px * (x1 - x0)), int(y0 + py * (y1 - y0)))
            for px, py in pts
        ]

    def background(self, t: float) -> Image.Image:
        img = self.bg_base.copy()
        d = ImageDraw.Draw(img)

        # Subtle latitude-like arcs.
        for frac in (0.28, 0.43, 0.58, 0.73):
            y = int(H * frac)
            d.arc((int(-W * 0.15), y - int(W * 0.15), int(W * 1.15), y + int(W * 0.15)), 188, 352,
                  fill=(80, 170, 205, 18), width=max(1, int(1 * SCALE)))

        # Drifting snow / ice particles.
        for x, y, r, speed, phase, alpha in self.particles:
            yy = (y + t * speed) % (H + 20) - 10
            xx = x + math.sin(t * 0.7 + phase) * 8 * SCALE
            rr = max(0.5, r)
            d.ellipse((xx - rr, yy - rr, xx + rr, yy + rr), fill=(220, 245, 255, alpha))
        return img

    def draw_title(self, img: Image.Image, t: float):
        intro_end = 6.4 if not QUICK_MODE else 1.55
        alpha = int(255 * smoothstep((t - 0.15) / 0.8) * (1 - smoothstep((t - intro_end) / 0.8)))
        if alpha > 4:
            draw_text(img, CONFIG["title"], (56 if not QUICK_MODE else 28, 88 if not QUICK_MODE else 44),
                      size=44 if not QUICK_MODE else 20, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(img, CONFIG["subtitle"], (58 if not QUICK_MODE else 30, 151 if not QUICK_MODE else 76),
                      size=22 if not QUICK_MODE else 10, fill=COLORS["cyan"] + (min(alpha, 235),), bold=True)

        shot_titles = {
            "intro": "A SATELLITE-WEIGHED ICE SHEET",
            "trend": "MASS LOSS SINCE 2002",
            "coasts": "WHERE LOSS IS STRONGEST",
            "processes": "HOW ICE LEAVES GREENLAND",
            "stats": "THE NUMBERS",
            "outro": "THE SIGNAL IS CLEAR",
        }
        if t > (5.0 if not QUICK_MODE else 1.25):
            draw_text(img, shot_titles[get_shot(t)["name"]], (56 if not QUICK_MODE else 28, 60 if not QUICK_MODE else 30),
                      size=19 if not QUICK_MODE else 9, fill=COLORS["muted"] + (215,), bold=True, stroke=1)

    def draw_source_hud(self, img: Image.Image):
        status = "OFFLINE FIXTURE" if self.snapshot.offline_fixture else ("CACHE" if self.snapshot.data_status == "cache" else "LIVE PAGE")
        draw_text(img, f"NASA ICE DATA // {status}", (W - (48 if not QUICK_MODE else 24), 72 if not QUICK_MODE else 36),
                  size=17 if not QUICK_MODE else 8, fill=COLORS["cyan"] + (220,), bold=True, anchor="ra", stroke=1)
        generated = self.summary["generated_at_utc"].replace("T", " ").replace("Z", " UTC")
        draw_text(img, generated, (W - (48 if not QUICK_MODE else 24), 102 if not QUICK_MODE else 51),
                  size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (190,), anchor="ra", stroke=1)

    def draw_caption(self, img: Image.Image, t: float):
        caption = caption_at(t)
        if not caption:
            return
        y0 = H - (244 if not QUICK_MODE else 124)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle(
            (44 if not QUICK_MODE else 22, y0, W - (44 if not QUICK_MODE else 22), y0 + (124 if not QUICK_MODE else 66)),
            radius=24 if not QUICK_MODE else 12,
            fill=(2, 8, 16, 184),
            outline=(90, 205, 235, 66),
            width=1,
        )
        img.alpha_composite(overlay)
        draw_wrapped_text(img, caption, (68 if not QUICK_MODE else 34, y0 + (28 if not QUICK_MODE else 14)),
                          W - (136 if not QUICK_MODE else 68), size=28 if not QUICK_MODE else 13,
                          fill=COLORS["white"] + (245,))

    def draw_hud_noise(self, img: Image.Image, t: float):
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        offset = int((t * 31) % 8)
        for y in range(offset, H, 8):
            od.line((0, y, W, y), fill=(110, 210, 235, 8), width=1)
        scan_y = int((t * 150) % (H + 180)) - 90
        od.rectangle((0, scan_y, W, scan_y + (42 if not QUICK_MODE else 21)), fill=(100, 220, 240, 6))
        img.alpha_composite(overlay)

    def _draw_greenland(self, img: Image.Image, scale: float = 1.0, shift_y: float = 0.0, glow: bool = True):
        pts = []
        cx = W * 0.5
        cy = H * 0.47 + shift_y
        base_cx = sum(p[0] for p in self.greenland_poly) / len(self.greenland_poly)
        base_cy = sum(p[1] for p in self.greenland_poly) / len(self.greenland_poly)
        for x, y in self.greenland_poly:
            pts.append((int(cx + (x - base_cx) * scale), int(cy + (y - base_cy) * scale)))

        if glow:
            layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
            ld = ImageDraw.Draw(layer)
            ld.polygon(pts, fill=COLORS["cyan"] + (45,))
            layer = layer.filter(ImageFilter.GaussianBlur(24 if not QUICK_MODE else 12))
            img.alpha_composite(layer)

        d = ImageDraw.Draw(img)
        d.polygon(pts, fill=COLORS["ice"] + (238,), outline=(255, 255, 255, 235))
        # Interior contour hints.
        for frac in (0.33, 0.48, 0.62):
            bbox = (
                int(W * (0.34 + frac * 0.03)), int(H * (0.28 + frac * 0.09) + shift_y),
                int(W * (0.66 - frac * 0.02)), int(H * (0.62 + frac * 0.05) + shift_y),
            )
            d.arc(bbox, 198, 342, fill=COLORS["ice_shadow"] + (80,), width=max(1, int(2 * SCALE)))
        return pts

    def draw_intro(self, img: Image.Image, t: float):
        shot = SHOT_PLAN[0]
        p = ease_in_out_sine((t - shot["start"]) / max(1e-6, shot["end"] - shot["start"]))
        pts = self._draw_greenland(img, scale=0.98 + 0.02 * math.sin(t * 1.4), shift_y=10 * SCALE)
        d = ImageDraw.Draw(img)

        # Stylized calving fragments drift away from the southeast edge.
        anchor = pts[10]
        for i in range(7):
            phase = (p * 1.5 + i * 0.16) % 1.0
            x = anchor[0] + phase * 175 * SCALE + i * 6 * SCALE
            y = anchor[1] + phase * 115 * SCALE + math.sin(i * 1.4) * 16 * SCALE
            r = (12 - i * 0.9) * SCALE
            d.polygon([(x-r, y-r*0.6), (x+r*0.8, y-r), (x+r, y+r*0.5), (x-r*0.6, y+r)],
                      fill=COLORS["ice"] + (int(230 * (1 - 0.55 * phase)),))

        # Big rate counter.
        rate = self.snapshot.loss_rate_gt_per_year
        shown = rate * smoothstep(p * 1.25)
        draw_text(img, f"≈ {shown:,.0f} Gt / YEAR", (W // 2, int(H * 0.73)),
                  size=39 if not QUICK_MODE else 18, fill=COLORS["orange"] + (245,), bold=True, anchor="ma")
        draw_text(img, f"NASA average • {self.snapshot.start_year}–{self.snapshot.end_year}", (W // 2, int(H * 0.77)),
                  size=20 if not QUICK_MODE else 9, fill=COLORS["muted"] + (230,), bold=True, anchor="ma", stroke=1)

    def _chart_xy(self, year: float, mass: float) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.chart_box
        xmin = self.snapshot.start_year
        xmax = self.snapshot.end_year
        y_top = 0.0
        y_bottom = min(-1000.0, float(self.trend["rate_based_mass_change_gt"].min()) * 1.08)
        x = x0 + (year - xmin) / max(1e-6, xmax - xmin) * (x1 - x0)
        y = y0 + (y_top - mass) / max(1e-6, y_top - y_bottom) * (y1 - y0)
        return x, y

    def draw_trend(self, img: Image.Image, t: float):
        x0, y0, x1, y1 = self.chart_box
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0, y0, x1, y1), radius=28 if not QUICK_MODE else 14,
                             fill=COLORS["panel"] + (188,), outline=(95, 198, 225, 80), width=2)
        img.alpha_composite(overlay)
        d = ImageDraw.Draw(img)

        # Axes and grid.
        y_bottom = float(self.trend["rate_based_mass_change_gt"].min()) * 1.08
        y_ticks = np.linspace(0, y_bottom, 5)
        for val in y_ticks:
            _, yy = self._chart_xy(self.snapshot.start_year, val)
            d.line((x0 + 54 * SCALE, yy, x1 - 20 * SCALE, yy), fill=(120, 190, 210, 35), width=1)
            draw_text(img, f"{int(round(val)):,}", (x0 + 44 * SCALE, int(yy)), size=14 if not QUICK_MODE else 7,
                      fill=COLORS["muted"] + (190,), anchor="ra", stroke=1)

        tick_years = sorted(set([self.snapshot.start_year, 2010, 2018, self.snapshot.end_year]))
        for yr in tick_years:
            if self.snapshot.start_year <= yr <= self.snapshot.end_year:
                xx, _ = self._chart_xy(yr, 0)
                d.line((xx, y0 + 22 * SCALE, xx, y1 - 18 * SCALE), fill=(120, 190, 210, 28), width=1)
                draw_text(img, str(yr), (int(xx), y1 + 20 * SCALE), size=15 if not QUICK_MODE else 7,
                          fill=COLORS["muted"] + (205,), anchor="ma", stroke=1)

        shot = get_shot(t)
        p = smoothstep((t - shot["start"]) / max(1e-6, shot["end"] - shot["start"]))
        reveal_year = lerp(self.snapshot.start_year, self.snapshot.end_year, p)
        visible = self.trend[self.trend["year"] <= reveal_year]
        points = [self._chart_xy(float(r.year), float(r.rate_based_mass_change_gt)) for r in visible.itertuples()]
        if len(points) >= 2:
            # Fill under the revealed line.
            poly = points + [(points[-1][0], y0 + 22 * SCALE), (points[0][0], y0 + 22 * SCALE)]
            fill = Image.new("RGBA", SIZE, (0, 0, 0, 0))
            fd = ImageDraw.Draw(fill)
            fd.polygon(poly, fill=COLORS["red"] + (26,))
            img.alpha_composite(fill)
            d.line(points, fill=COLORS["orange"] + (235,), width=max(2, int(5 * SCALE)))
            xx, yy = points[-1]
            rr = 10 * SCALE
            d.ellipse((xx-rr, yy-rr, xx+rr, yy+rr), fill=COLORS["red"] + (245,), outline=(255,255,255,230), width=1)

        # Mission gap marker (actual GRACE observations have a gap; the plotted line is illustrative).
        gap_x0, _ = self._chart_xy(2017.45, 0)
        gap_x1, _ = self._chart_xy(2018.45, 0)
        gap_layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gap_draw = ImageDraw.Draw(gap_layer)
        gap_draw.rectangle((gap_x0, y0 + 18 * SCALE, gap_x1, y1 - 18 * SCALE), fill=(255, 255, 255, 18))
        img.alpha_composite(gap_layer)
        draw_text(img, "MISSION GAP", (int((gap_x0 + gap_x1)/2), y0 + 32 * SCALE), size=13 if not QUICK_MODE else 6,
                  fill=COLORS["muted"] + (190,), bold=True, anchor="ma", stroke=1)

        draw_text(img, "GREENLAND MASS CHANGE (Gt)", (x0 + 20, y0 + 18), size=18 if not QUICK_MODE else 8,
                  fill=COLORS["cyan"] + (225,), bold=True, stroke=1)
        draw_text(img, f"AVERAGE RATE ≈ −{self.snapshot.loss_rate_gt_per_year:.0f} Gt / YEAR", (x1 - 20, y0 + 18),
                  size=17 if not QUICK_MODE else 8, fill=COLORS["orange"] + (230,), bold=True, anchor="ra", stroke=1)
        draw_wrapped_text(img, "Illustrative trend from NASA's published average rate — not the monthly GRACE data points.",
                          (x0 + 18, int(H * 0.70)), x1 - x0 - 36, size=18 if not QUICK_MODE else 8,
                          fill=COLORS["muted"] + (225,))

    def draw_coasts(self, img: Image.Image, t: float):
        pts = self._draw_greenland(img, scale=0.94, shift_y=5 * SCALE)
        d = ImageDraw.Draw(img)
        shot = get_shot(t)
        p = smoothstep((t - shot["start"]) / max(1e-6, shot["end"] - shot["start"]))

        # Stylized coastal hot spots: stronger along west/south perimeter.
        hot_indices = [14, 15, 16, 17, 18, 19, 10, 11, 12]
        for j, idx in enumerate(hot_indices):
            x, y = pts[idx % len(pts)]
            pulse = 0.65 + 0.35 * math.sin(t * 3.0 + j)
            r = (20 + 25 * p + 6 * pulse) * SCALE
            glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow)
            gd.ellipse((x-r, y-r, x+r, y+r), fill=COLORS["red"] + (65,))
            glow = glow.filter(ImageFilter.GaussianBlur(12 if not QUICK_MODE else 6))
            img.alpha_composite(glow)
            d.ellipse((x-r*0.34, y-r*0.34, x+r*0.34, y+r*0.34), fill=COLORS["orange"] + (200,))

        # Callout cards.
        card_y = int(H * 0.70)
        card_h = int(116 * SCALE)
        left = int(W * 0.08)
        width = int(W * 0.84)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((left, card_y, left+width, card_y+card_h), radius=22 if not QUICK_MODE else 11,
                             fill=(2, 9, 18, 200), outline=COLORS["orange"] + (80,), width=2)
        img.alpha_composite(overlay)
        draw_text(img, "COASTAL MARGINS", (left + 18, card_y + 18), size=20 if not QUICK_MODE else 9,
                  fill=COLORS["orange"] + (245,), bold=True, stroke=1)
        draw_wrapped_text(img, "NASA maps show the largest decreases around lower-elevation coastal areas — especially West Greenland.",
                          (left + 18, card_y + (50 if not QUICK_MODE else 25)), width - 36,
                          size=18 if not QUICK_MODE else 8, fill=COLORS["white"] + (235,))
        draw_text(img, "SCHEMATIC • NOT A GEOSPATIAL DATA MAP", (W // 2, int(H * 0.79)),
                  size=15 if not QUICK_MODE else 7, fill=COLORS["muted"] + (205,), bold=True, anchor="ma", stroke=1)

    def draw_processes(self, img: Image.Image, t: float):
        d = ImageDraw.Draw(img)
        x_margin = int(W * 0.07)
        gap = int(W * 0.04)
        card_w = int((W - 2*x_margin - gap) / 2)
        card_h = int(H * 0.48)
        y0 = int(H * 0.20)
        cards = [(x_margin, "SURFACE MELT", COLORS["gold"]), (x_margin + card_w + gap, "GLACIER DISCHARGE", COLORS["cyan"])]

        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for x, _, color in cards:
            od.rounded_rectangle((x, y0, x+card_w, y0+card_h), radius=28 if not QUICK_MODE else 14,
                                 fill=(3, 12, 23, 196), outline=color + (75,), width=2)
        img.alpha_composite(overlay)

        # Left card: sun -> meltwater runoff.
        lx = x_margin
        sun_x, sun_y = lx + card_w//2, y0 + int(112 * SCALE)
        sr = 38 * SCALE
        d.ellipse((sun_x-sr, sun_y-sr, sun_x+sr, sun_y+sr), fill=COLORS["gold"] + (240,))
        for a in np.linspace(0, math.tau, 10, endpoint=False):
            x1 = sun_x + math.cos(a) * 58 * SCALE
            y1 = sun_y + math.sin(a) * 58 * SCALE
            x2 = sun_x + math.cos(a) * 82 * SCALE
            y2 = sun_y + math.sin(a) * 82 * SCALE
            d.line((x1, y1, x2, y2), fill=COLORS["gold"] + (170,), width=max(1, int(3*SCALE)))
        ice_y = y0 + int(255 * SCALE)
        d.polygon([(lx+35*SCALE, ice_y), (lx+card_w-35*SCALE, ice_y), (lx+card_w-65*SCALE, ice_y+75*SCALE),
                   (lx+65*SCALE, ice_y+75*SCALE)], fill=COLORS["ice"] + (238,))
        phase = (t * 0.42) % 1.0
        for i in range(5):
            dx = lx + (75 + i*48) * SCALE
            dy = ice_y + 16*SCALE + ((phase + i*0.18) % 1.0) * 120*SCALE
            rr = 6*SCALE
            d.ellipse((dx-rr, dy-rr, dx+rr, dy+rr*1.3), fill=COLORS["cyan"] + (220,))

        # Right card: glacier front calving into ocean.
        rx = x_margin + card_w + gap
        ocean_y = y0 + int(335 * SCALE)
        d.rectangle((rx+20*SCALE, ocean_y, rx+card_w-20*SCALE, y0+card_h-28*SCALE), fill=COLORS["ocean"] + (240,))
        cliff_x = rx + int(card_w*0.46)
        d.polygon([(rx+28*SCALE, y0+160*SCALE), (cliff_x, y0+145*SCALE), (cliff_x, ocean_y), (rx+28*SCALE, ocean_y)],
                  fill=COLORS["ice"] + (240,))
        d.line((cliff_x, y0+155*SCALE, cliff_x, ocean_y), fill=(255,255,255,220), width=max(1,int(3*SCALE)))
        calve_p = (t * 0.38) % 1.0
        bx = cliff_x + (24 + 95*calve_p) * SCALE
        by = y0 + (205 + 105*calve_p) * SCALE
        br = 26 * SCALE
        d.polygon([(bx-br,by-br*0.5),(bx+br*0.5,by-br),(bx+br,by+br*0.3),(bx-br*0.4,by+br)],
                  fill=COLORS["ice"] + (235,))
        for i in range(4):
            yy = ocean_y + (20 + i*24)*SCALE
            d.arc((rx+35*SCALE, yy-10*SCALE, rx+card_w-30*SCALE, yy+10*SCALE), 180, 360,
                  fill=COLORS["cyan"] + (90,), width=max(1,int(2*SCALE)))

        for x, title, color in cards:
            draw_text(img, title, (x + card_w//2, y0 + 28*SCALE), size=21 if not QUICK_MODE else 10,
                      fill=color + (245,), bold=True, anchor="ma", stroke=1)

        draw_wrapped_text(img, "Meltwater runs off the surface.", (lx+22*SCALE, y0+card_h-88*SCALE), card_w-44*SCALE,
                          size=17 if not QUICK_MODE else 8, fill=COLORS["white"] + (230,))
        draw_wrapped_text(img, "Outlet glaciers move ice into the ocean, where it can calve as icebergs.",
                          (rx+22*SCALE, y0+card_h-105*SCALE), card_w-44*SCALE,
                          size=17 if not QUICK_MODE else 8, fill=COLORS["white"] + (230,))

    def _stat_row(self, img: Image.Image, left: str, right: str, x: int, y: int, width: int, color: Tuple[int,int,int]):
        overlay = Image.new("RGBA", SIZE, (0,0,0,0))
        od = ImageDraw.Draw(overlay)
        h = int(78*SCALE)
        od.rounded_rectangle((x,y,x+width,y+h), radius=20 if not QUICK_MODE else 10,
                             fill=(3,12,23,190), outline=color+(72,), width=2)
        img.alpha_composite(overlay)
        draw_text(img, left, (x+18,y+18), size=21 if not QUICK_MODE else 10, fill=color+(240,), bold=True, stroke=1)
        draw_text(img, right, (x+width-18,y+18), size=21 if not QUICK_MODE else 10, fill=COLORS["white"]+(238,), bold=True, anchor="ra", stroke=1)

    def draw_stats(self, img: Image.Image):
        x = int(W*0.08)
        y = int(H*0.21)
        width = int(W*0.84)
        row_h = int(92*SCALE)
        gap = int(15*SCALE)
        values = [
            ("Average mass loss", f"≈ {self.snapshot.loss_rate_gt_per_year:.0f} Gt / year", COLORS["orange"]),
            ("Sea-level contribution", f"≈ {self.snapshot.sea_level_mm_per_year:.1f} mm / year", COLORS["cyan"]),
            ("Satellite record", f"{self.snapshot.start_year} → {self.snapshot.end_year}", COLORS["teal"]),
            ("Original mission", "GRACE • 2002–2017", COLORS["gold"]),
            ("Follow-on mission", "GRACE-FO • since 2018", COLORS["blue"]),
        ]
        for i, (left,right,color) in enumerate(values):
            self._stat_row(img, left, right, x, y + i*(row_h+gap), width, color)

        draw_wrapped_text(img, "GRACE satellites infer mass change by measuring tiny changes in Earth's gravity field.",
                          (x+8, y + len(values)*(row_h+gap) + 8), width-16,
                          size=17 if not QUICK_MODE else 8, fill=COLORS["muted"]+(220,))

    def draw_outro(self, img: Image.Image):
        self._draw_greenland(img, scale=0.70, shift_y=-30*SCALE)
        draw_text(img, "GREENLAND IS LOSING ICE", (W//2, int(H*0.64)), size=38 if not QUICK_MODE else 17,
                  fill=COLORS["white"]+(245,), bold=True, anchor="ma")
        draw_text(img, f"≈ {self.snapshot.loss_rate_gt_per_year:.0f} Gt / YEAR", (W//2, int(H*0.69)),
                  size=34 if not QUICK_MODE else 16, fill=COLORS["orange"]+(245,), bold=True, anchor="ma")
        draw_text(img, f"NASA GRACE / GRACE-FO • {self.snapshot.start_year}–{self.snapshot.end_year}", (W//2, int(H*0.74)),
                  size=19 if not QUICK_MODE else 9, fill=COLORS["cyan"]+(230,), bold=True, anchor="ma", stroke=1)
        draw_wrapped_text(img, "Rate-based educational visualization. Source URLs and data snapshot are saved with the render.",
                          (int(W*0.13), int(H*0.78)), int(W*0.74), size=16 if not QUICK_MODE else 7,
                          fill=COLORS["muted"]+(215,))

    def render_frame(self, t: float) -> np.ndarray:
        img = self.background(t)
        self.draw_title(img, t)
        self.draw_source_hud(img)

        shot = get_shot(t)["name"]
        if shot == "intro":
            self.draw_intro(img, t)
        elif shot == "trend":
            self.draw_trend(img, t)
        elif shot == "coasts":
            self.draw_coasts(img, t)
        elif shot == "processes":
            self.draw_processes(img, t)
        elif shot == "stats":
            self.draw_stats(img)
        else:
            self.draw_outro(img)

        self.draw_caption(img, t)
        self.draw_hud_noise(img, t)

        arr = np.array(img.convert("RGB"))
        graded = Image.fromarray(arr)
        graded = ImageEnhance.Contrast(graded).enhance(1.08)
        graded = ImageEnhance.Color(graded).enhance(1.05)
        arr = np.array(graded)
        arr = np.clip(arr.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / 0.9)
        fade_out = 1 - smoothstep((t - (CONFIG["duration_s"] - 1.1)) / 1.0)
        return np.clip(arr.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)


# =============================================================================
# Output
# =============================================================================

def render_video(scene: GreenlandScene) -> Path:
    raw_path = OUTPUT_ROOT / f"{CONFIG['basename']}_raw.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['basename']}_final.mp4"
    write_srt(OUTPUT_ROOT / f"{CONFIG['basename']}.srt")
    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    with iio.get_writer(
        raw_path,
        fps=CONFIG["fps"],
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for frame_index in tqdm(range(frame_count), desc="Rendering Greenland short"):
            writer.append_data(scene.render_frame(frame_index / CONFIG["fps"]))
    shutil.copyfile(raw_path, final_path)
    return final_path


def make_contact_sheet(paths: Sequence[Path], out_path: Path):
    thumbs = []
    for path in paths[:6]:
        image = Image.open(path).convert("RGB").resize((270, 480))
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 120, 38), fill=(0, 0, 0))
        draw.text((18, 13), path.stem.replace("preview_", ""), fill=(255, 255, 255))
        thumbs.append(image)
    sheet = Image.new("RGB", (600, 1520), (6, 16, 24))
    for index, thumb in enumerate(thumbs):
        row, col = divmod(index, 2)
        sheet.paste(thumb, (20 + col * 290, 20 + row * 500))
    sheet.save(out_path, quality=92)


def main():
    print("Collecting Greenland ice data ...")
    snapshot, trend, events, summary = collect_data()
    csv_path, json_path = save_data(snapshot, trend, events, summary)
    print("Trend data:", csv_path.resolve())
    print("Summary:", json_path.resolve())

    scene = GreenlandScene(snapshot, trend, events, summary)
    preview_times = [
        1.0,
        min(10.0, CONFIG["duration_s"] * 0.22),
        min(23.0, CONFIG["duration_s"] * 0.42),
        min(36.0, CONFIG["duration_s"] * 0.64),
        min(47.0, CONFIG["duration_s"] * 0.83),
        CONFIG["duration_s"] - 1,
    ]
    preview_paths: List[Path] = []
    for t in tqdm(preview_times, desc="Preview frames"):
        path = PREVIEW_ROOT / f"preview_{int(t):02d}s.png"
        Image.fromarray(scene.render_frame(float(t))).save(path)
        preview_paths.append(path)

    make_contact_sheet(preview_paths, PREVIEW_ROOT / "greenland_is_losing_ice_contact_sheet.jpg")
    video_path = render_video(scene)
    print("Video:", video_path.resolve())
    print("Source status:", summary)


if __name__ == "__main__":
    main()
