from __future__ import annotations

"""
Every Solar Eclipse of the Next 100 Years — cinematic YouTube Short renderer

This script creates a vertical astronomy short using NASA eclipse-catalog data.
It is designed to fetch the official solar-eclipse century catalogs from NASA's
Eclipse Web Site / GSFC catalog pages and then render a 1080x1920 short that
shows every solar eclipse in the 100-year calendar window 2026-2125.


Scientific grounding from those NASA catalog pages:
- The 21st century (2001-2100) contains 224 solar eclipses.
- The 22nd century (2101-2200) contains 235 solar eclipses.
- Solar eclipses are cataloged by date/time of greatest eclipse, type, Saros,
  gamma, magnitude, latitude/longitude, and central-path data.
- NASA classifies solar eclipses as partial, annular, total, or hybrid.

Important note for offline environments:
If live NASA pages cannot be reached, the script falls back to a REAL but SMALL
fixture subset of catalog rows (spanning 2026-2035 and 2120-2125). That fallback
is only for local layout validation. For the full, correct 100-year video,
run this script in an internet-connected environment.

Recommended install:
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm requests beautifulsoup4 matplotlib

Quick preview render:
    SOLAR_ECLIPSES_SHORT_QUICK=1 python every_solar_eclipse_next_100_years_short.py
"""

import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


# %% Configuration

QUICK_MODE = os.environ.get("SOLAR_ECLIPSES_SHORT_QUICK", "0") == "1"
OUTPUT_ROOT = Path("solar_eclipses_next_100_years_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in [OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "every_solar_eclipse_next_100_years",
    "window_start": date(2026, 1, 1),
    "window_end": date(2125, 12, 31),
    "title_text": "EVERY SOLAR ECLIPSE OF THE NEXT 100 YEARS",
    "subtitle_text": "NASA eclipse catalogs // 2026-2125",
    "background_particle_count": 340,
    "hud_noise_count": 52,
    "contrast_boost": 1.08,
    "saturation_boost": 1.05,
    "vignette_strength": 0.24,
}

OUT_SIZE = (CONFIG["video_width"], CONFIG["video_height"])
TYPE_ORDER = ["Total", "Annular", "Hybrid", "Partial"]
TYPE_COLORS = {
    "Total": (90, 225, 255),
    "Annular": (255, 190, 95),
    "Hybrid": (255, 105, 190),
    "Partial": (170, 180, 195),
}
NASA_CATALOG_URLS = [
    "https://eclipse.gsfc.nasa.gov/SEcat5/SE2001-2100.html",
    "https://eclipse.gsfc.nasa.gov/SEcat5/SE2101-2200.html",
]

CAPTIONS = [
    (0.5, 6.8, "Global solar eclipses happen two to five times every year."),
    (6.9, 15.5, "This short maps every solar eclipse in the 100 calendar years from 2026 through 2125."),
    (15.6, 25.0, "Each dot is one eclipse from NASA's century catalogs—partial, annular, total, or hybrid."),
    (25.1, 35.0, "Most years have two eclipses. Some have three or four. Rare calendars can reach five."),
    (35.1, 46.0, "The pattern comes from new moons crossing one of the Moon's orbital nodes."),
    (46.1, 57.4, "So over the next century, the sky keeps repeating the same geometry in endlessly different places."),
]

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 8.0 if not QUICK_MODE else 2.0},
    {"name": "timeline", "start": 8.0 if not QUICK_MODE else 2.0, "end": 24.0 if not QUICK_MODE else 5.0},
    {"name": "counts", "start": 24.0 if not QUICK_MODE else 5.0, "end": 38.0 if not QUICK_MODE else 8.0},
    {"name": "cadence", "start": 38.0 if not QUICK_MODE else 8.0, "end": 50.0 if not QUICK_MODE else 10.0},
    {"name": "outro", "start": 50.0 if not QUICK_MODE else 10.0, "end": CONFIG["duration_s"]},
]


# %% Utilities

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))


def smoothstep(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def ease_in_out_sine(t):
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


def lerp(a, b, t):
    return a + (b - a) * t


def month_number(mon: str) -> int:
    return datetime.strptime(mon, "%b").month


def get_font(size: int, bold: bool = False):
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(image: Image.Image, text: str, xy: Tuple[int, int], size: int = 32,
              fill=(255, 255, 255, 255), bold: bool = False, stroke: int = 2,
              anchor: str = "la"):
    draw = ImageDraw.Draw(image)
    draw.text(
        xy,
        text,
        font=get_font(size, bold=bold),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(fill[3] if len(fill) > 3 else 255, 220)),
    )


def draw_wrapped_text(image: Image.Image, text: str, xy: Tuple[int, int], max_width: int,
                      size: int = 28, fill=(255, 255, 255, 245), bold: bool = False,
                      line_spacing: int = 6):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
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
        y += (bbox[3] - bbox[1]) + line_spacing


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cx = width / 2.0
    cy = height / 2.0
    nx = (xx - cx) / (width / 2.0)
    ny = (yy - cy) / (height / 2.0)
    radius = np.sqrt(nx**2 + ny**2)
    return np.clip(1.0 - strength * radius**1.8, 0.0, 1.0).astype(np.float32)


def apply_grade(arr: np.ndarray) -> np.ndarray:
    image = Image.fromarray(arr)
    image = ImageEnhance.Contrast(image).enhance(CONFIG["contrast_boost"])
    image = ImageEnhance.Color(image).enhance(CONFIG["saturation_boost"])
    return np.array(image)


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


def get_shot(t: float) -> Dict:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(captions, path: Path):
    lines = []
    for i, (start, end, text) in enumerate(captions, start=1):
        lines.append(str(i))
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def duration_to_seconds(value: Optional[str]) -> float:
    if not value or pd.isna(value):
        return 0.0
    match = re.match(r"(?:(\d+)m)?(\d+)s", str(value).strip())
    if not match:
        return 0.0
    mins = int(match.group(1) or 0)
    secs = int(match.group(2) or 0)
    return mins * 60 + secs


def classify_type(raw_type: str) -> str:
    rt = (raw_type or "").strip()
    if not rt:
        return "Unknown"
    initial = rt[0].upper()
    return {
        "P": "Partial",
        "A": "Annular",
        "T": "Total",
        "H": "Hybrid",
    }.get(initial, "Unknown")


VIGNETTE = make_vignette(OUT_SIZE[0], OUT_SIZE[1], CONFIG["vignette_strength"])


# %% NASA catalog parsing

FIXTURE_ROWS = [
    # 2026-2035
    "09565 2026 Feb 17 12:13:06 75 323 121 A -t -0.9743 0.9630 65S 87E 12 616 02m20s",
    "09566 2026 Aug 12 17:47:06 75 329 126 T -p 0.8977 1.0386 65N 25W 26 294 02m18s",
    "09567 2027 Feb 06 16:00:48 76 335 131 A -n -0.2952 0.9281 31S 48W 73 282 07m51s",
    "09568 2027 Aug 02 10:07:50 76 341 136 T nn 0.1421 1.0790 26N 33E 82 258 06m23s",
    "09569 2028 Jan 26 15:08:59 76 347 141 A p- 0.3901 0.9208 3N 52W 67 323 10m27s",
    "09570 2028 Jul 22 02:56:40 77 353 146 T p- -0.6056 1.0560 16S 127E 53 230 05m10s",
    "09571 2029 Jan 14 17:13:48 77 359 151 P t- 1.0553 0.8714 64N 114W 0",
    "09572 2029 Jun 12 04:06:13 77 364 118 P -t 1.2943 0.4576 67N 66W 0",
    "09573 2029 Jul 11 15:37:19 77 365 156 P t- -1.4191 0.2303 64S 86W 0",
    "09574 2029 Dec 05 15:03:58 77 370 123 P -t -1.0609 0.8911 68S 136E 0",
    "09575 2030 Jun 01 06:29:13 78 376 128 A -p 0.5626 0.9443 57N 80E 55 250 05m21s",
    "09576 2030 Nov 25 06:51:37 78 382 133 T -n -0.3867 1.0468 44S 71E 67 169 03m44s",
    "09577 2031 May 21 07:16:04 78 388 138 A nn -0.1970 0.9589 9N 72E 79 152 05m26s",
    "09578 2031 Nov 14 21:07:31 79 394 143 H n- 0.3078 1.0106 1S 138W 72 38 01m08s",
    "09579 2032 May 09 13:26:42 79 400 148 A t- -0.9375 0.9957 51S 7W 20 44 00m22s",
    "09580 2032 Nov 03 05:34:13 79 406 153 P t- 1.0643 0.8554 70N 133E 0",
    "09581 2033 Mar 30 18:02:36 80 411 120 T -t 0.9778 1.0462 71N 156W 11 781 02m37s",
    "09582 2033 Sep 23 13:54:31 80 417 125 P -t -1.1583 0.6890 72S 121W 0",
    "09583 2034 Mar 20 10:18:45 80 423 130 T -n 0.2894 1.0458 16N 22E 73 159 04m09s",
    "09584 2034 Sep 12 16:19:28 81 429 135 A -p -0.3936 0.9736 18S 73W 67 102 02m58s",
    "09585 2035 Mar 09 23:05:54 81 435 140 A n- -0.4368 0.9919 29S 155W 64 31 00m48s",
    "09586 2035 Sep 02 01:56:46 81 441 145 T p- 0.3727 1.0320 29N 158E 68 116 02m54s",
    # 2120-2125
    "09779 2120 Jul 25 14:40:02 252 1491 128 An -t 0.9948 0.9343 66N 90E 4 - 04m00s",
    "09780 2121 Jan 19 02:54:15 253 1497 133 T -n -0.4190 1.0371 44S 150E 65 137 02m52s",
    "09781 2121 Jul 14 16:42:39 255 1503 138 A nn 0.2125 0.9758 34N 64W 78 88 02m32s",
    "09782 2122 Jan 08 15:48:51 256 1509 143 A n- 0.2713 0.9907 7S 58W 74 34 01m02s",
    "09783 2122 Jul 04 01:25:31 257 1515 148 T p- -0.5649 1.0280 11S 155E 56 114 02m56s",
    "09784 2122 Dec 28 22:00:56 258 1521 153 A+ p- 1.0072 0.9450 65N 170W 0",
    "09785 2123 May 25 09:33:27 259 1526 120 P -t 1.2325 0.5729 68N 128W 0",
    "09786 2123 Jun 23 16:26:12 259 1527 158 P t- -1.2763 0.4882 66S 80W 0",
    "09787 2123 Nov 18 03:07:26 260 1532 125 P -t -1.3389 0.3848 69S 25W 0",
    "09788 2124 May 14 01:59:10 262 1538 130 T -p 0.5286 1.0464 50N 143E 58 182 03m34s",
    "09789 2124 Nov 06 06:36:34 263 1544 135 A -p -0.5921 0.9724 52S 67E 53 123 02m26s",
    "09790 2125 May 03 13:42:33 264 1550 140 A nn -0.2263 0.9915 3N 23W 77 31 00m59s",
    "09791 2125 Oct 26 17:30:49 265 1556 145 T n- 0.1461 1.0329 4S 84W 82 112 03m15s",
]

ROW_RE = re.compile(r"^\s*\d{5}\s+\d{4}\s+[A-Z][a-z]{2}\s+\d{2}\s+\d{2}:\d{2}:\d{2}\b")


def parse_catalog_row(row: str) -> Optional[Dict]:
    row = re.sub(r"\s+", " ", row.strip())
    if not row or not ROW_RE.match(row):
        return None
    tokens = row.split(" ")
    if len(tokens) < 15:
        return None
    catalog_number = int(tokens[0])
    year = int(tokens[1])
    month_abbr = tokens[2]
    day = int(tokens[3])
    greatest = tokens[4]
    delta_t = float(tokens[5])
    luna_num = int(tokens[6])
    saros_num = int(tokens[7])
    raw_type = tokens[8]
    qle = tokens[9]
    gamma = float(tokens[10])
    magnitude = float(tokens[11])
    lat = tokens[12]
    lon = tokens[13]
    sun_alt = float(tokens[14])
    path_width_km = None
    central_duration = None
    if len(tokens) >= 16:
        try:
            path_width_km = float(tokens[15])
        except Exception:
            path_width_km = None
    if len(tokens) >= 17:
        central_duration = tokens[16]

    d = date(year, month_number(month_abbr), day)
    dt = datetime.strptime(f"{year:04d}-{month_number(month_abbr):02d}-{day:02d} {greatest}", "%Y-%m-%d %H:%M:%S")
    return {
        "catalog_number": catalog_number,
        "date": d.isoformat(),
        "datetime_utc": dt.isoformat(),
        "year": year,
        "month": month_number(month_abbr),
        "day": day,
        "greatest_eclipse_utc": greatest,
        "delta_t_seconds": delta_t,
        "luna_number": luna_num,
        "saros_number": saros_num,
        "raw_type": raw_type,
        "type_group": classify_type(raw_type),
        "qle": qle,
        "gamma": gamma,
        "magnitude": magnitude,
        "latitude": lat,
        "longitude": lon,
        "sun_altitude_deg": sun_alt,
        "path_width_km": path_width_km,
        "central_duration": central_duration,
        "central_duration_seconds": duration_to_seconds(central_duration),
    }


def parse_catalog_html(html: str) -> List[Dict]:
    rows: List[Dict] = []
    texts = []
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        pre_blocks = soup.find_all("pre")
        if pre_blocks:
            texts.extend(block.get_text("\n") for block in pre_blocks)
        else:
            texts.append(soup.get_text("\n"))
    else:
        texts.append(html)

    for text in texts:
        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line.strip())
            if ROW_RE.match(line):
                item = parse_catalog_row(line)
                if item:
                    rows.append(item)
    return rows


def fetch_live_nasa_catalog() -> Tuple[pd.DataFrame, str]:
    if requests is None or BeautifulSoup is None:
        raise RuntimeError("requests and/or BeautifulSoup not available")
    all_rows: List[Dict] = []
    for url in NASA_CATALOG_URLS:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        all_rows.extend(parse_catalog_html(response.text))
    if not all_rows:
        raise RuntimeError("No rows parsed from live NASA catalog pages")
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["catalog_number"]).sort_values("datetime_utc").reset_index(drop=True)
    return df, "live_nasa_catalog"


def load_fallback_fixture() -> Tuple[pd.DataFrame, str]:
    rows = [parse_catalog_row(r) for r in FIXTURE_ROWS]
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows).sort_values("datetime_utc").reset_index(drop=True)
    return df, "offline_fixture_subset"


def load_eclipse_catalog() -> Tuple[pd.DataFrame, str, Optional[str]]:
    try:
        df, source = fetch_live_nasa_catalog()
        return df, source, None
    except Exception as exc:
        df, source = load_fallback_fixture()
        return df, source, str(exc)


def filter_window(df: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(CONFIG["window_start"])
    end = pd.Timestamp(CONFIG["window_end"])
    out = df.copy()
    out["date_ts"] = pd.to_datetime(out["date"])
    out = out[(out["date_ts"] >= start) & (out["date_ts"] <= end)].copy()
    out = out.sort_values("date_ts").reset_index(drop=True)
    return out


def create_summary(df: pd.DataFrame, source: str) -> Dict:
    if df.empty:
        raise RuntimeError("No eclipse rows available after filtering")
    counts_by_type = df["type_group"].value_counts().reindex(TYPE_ORDER, fill_value=0).to_dict()
    per_year = df.groupby("year").size().reindex(range(CONFIG["window_start"].year, CONFIG["window_end"].year + 1), fill_value=0)
    decade = (df["year"] // 10) * 10
    counts_by_decade = decade.value_counts().sort_index().to_dict()
    duration_df = df[df["central_duration_seconds"] > 0].copy()
    longest_event = None
    if not duration_df.empty:
        row = duration_df.sort_values("central_duration_seconds", ascending=False).iloc[0]
        longest_event = {
            "date": row["date"],
            "type_group": row["type_group"],
            "raw_type": row["raw_type"],
            "central_duration": row["central_duration"],
            "catalog_number": int(row["catalog_number"]),
        }
    summary = {
        "source": source,
        "window_start": CONFIG["window_start"].isoformat(),
        "window_end": CONFIG["window_end"].isoformat(),
        "num_eclipses": int(len(df)),
        "counts_by_type": {k: int(v) for k, v in counts_by_type.items()},
        "max_eclipses_in_one_year": int(per_year.max()),
        "min_eclipses_in_one_year": int(per_year.min()),
        "years_with_max_count": [int(y) for y, v in per_year.items() if int(v) == int(per_year.max())],
        "first_eclipse": df.iloc[0][["date", "type_group", "raw_type"]].to_dict(),
        "last_eclipse": df.iloc[-1][["date", "type_group", "raw_type"]].to_dict(),
        "longest_central_event": longest_event,
    }
    return summary


def save_data_products(df: pd.DataFrame, summary: Dict, source: str, error_note: Optional[str]):
    csv_path = DATA_ROOT / "solar_eclipses_2026_2125.csv"
    df.drop(columns=[c for c in ["date_ts"] if c in df.columns]).to_csv(csv_path, index=False)

    meta = {
        "summary": summary,
        "source": source,
        "error_note": error_note,
        "nasa_catalog_urls": NASA_CATALOG_URLS,
    }
    json_path = DATA_ROOT / "solar_eclipses_2026_2125_summary.json"
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Data written:", csv_path.resolve())
    print("Summary written:", json_path.resolve())
    return csv_path, json_path


def create_scientific_plots(df: pd.DataFrame, summary: Dict):
    per_year = df.groupby("year").size().reindex(range(CONFIG["window_start"].year, CONFIG["window_end"].year + 1), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(per_year.index, per_year.values)
    ax.set_title("Solar eclipses per year in the 2026-2125 window")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "eclipses_per_year.png", dpi=170)
    plt.close(fig)

    counts_by_type = pd.Series(summary["counts_by_type"]).reindex(TYPE_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(counts_by_type.index, counts_by_type.values)
    ax.set_title("Solar eclipse types in the 2026-2125 window")
    ax.set_xlabel("Type")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "eclipse_type_counts.png", dpi=170)
    plt.close(fig)


# %% Scene

class EclipseCenturyScene:
    def __init__(self, df: pd.DataFrame, summary: Dict, source: str):
        self.df = df.copy()
        self.summary = summary
        self.source = source
        self.start_ts = pd.Timestamp(CONFIG["window_start"])
        self.end_ts = pd.Timestamp(CONFIG["window_end"])
        self.total_days = max((self.end_ts - self.start_ts).days, 1)
        self.particles = self._make_particles(CONFIG["background_particle_count"], seed=13)
        self.hud_noise = self._make_hud_noise(CONFIG["hud_noise_count"], seed=99)
        self.timeline_points = self._prepare_timeline_points()
        self.per_year = self.df.groupby("year").size().reindex(range(CONFIG["window_start"].year, CONFIG["window_end"].year + 1), fill_value=0)
        self.type_counts = pd.Series(summary["counts_by_type"]).reindex(TYPE_ORDER).fillna(0)
        self.max_year_count = max(int(self.per_year.max()), 1)

    @staticmethod
    def _make_particles(count: int, seed: int):
        rng = np.random.default_rng(seed)
        items = []
        for _ in range(count):
            items.append({
                "x": float(rng.uniform(0, OUT_SIZE[0])),
                "y": float(rng.uniform(0, OUT_SIZE[1])),
                "r": float(rng.uniform(0.5, 2.2)),
                "a": int(rng.integers(20, 115)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "drift": float(rng.uniform(-10, 10)),
            })
        return items

    @staticmethod
    def _make_hud_noise(count: int, seed: int):
        rng = np.random.default_rng(seed)
        items = []
        for _ in range(count):
            items.append({
                "x": float(rng.uniform(0, OUT_SIZE[0])),
                "y": float(rng.uniform(0, OUT_SIZE[1])),
                "length": float(rng.uniform(8, 84)),
                "alpha": int(rng.integers(8, 48)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            })
        return items

    def _prepare_timeline_points(self):
        x_positions = {
            "Total": 0.18,
            "Annular": 0.41,
            "Hybrid": 0.64,
            "Partial": 0.87,
        }
        pts = []
        for _, row in self.df.iterrows():
            frac = (row["date_ts"] - self.start_ts).days / self.total_days
            pts.append({
                "x_frac": x_positions.get(row["type_group"], 0.87),
                "y_frac": float(frac),
                "type_group": row["type_group"],
                "date": row["date"],
                "raw_type": row["raw_type"],
                "duration": row.get("central_duration", None),
            })
        return pts

    def render_background(self, t: float) -> Image.Image:
        canvas = Image.new("RGBA", OUT_SIZE, (2, 7, 14, 255))
        draw = ImageDraw.Draw(canvas)

        for p in self.particles:
            x = (p["x"] + p["drift"] * 0.07 * t) % OUT_SIZE[0]
            y = (p["y"] + p["drift"] * 0.03 * t) % OUT_SIZE[1]
            twinkle = 0.7 + 0.3 * math.sin(1.7 * t + p["phase"])
            draw.ellipse((x-p["r"], y-p["r"], x+p["r"], y+p["r"]), fill=(215, 228, 255, int(p["a"] * twinkle)))

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        clouds = [
            (OUT_SIZE[0]*0.25, OUT_SIZE[1]*0.23, (45, 28, 120)),
            (OUT_SIZE[0]*0.72, OUT_SIZE[1]*0.39, (20, 80, 130)),
            (OUT_SIZE[0]*0.55, OUT_SIZE[1]*0.78, (11, 44, 95)),
        ]
        for cx, cy, color in clouds:
            for radius, alpha in [(480 * OUT_SIZE[0]/1080.0, 18), (320 * OUT_SIZE[0]/1080.0, 24), (220 * OUT_SIZE[0]/1080.0, 30)]:
                gd.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(color[0], color[1], color[2], alpha))
        glow = glow.filter(ImageFilter.GaussianBlur(65 if not QUICK_MODE else 32))
        canvas.alpha_composite(glow)
        return canvas

    def draw_sun_moon_hook(self, canvas: Image.Image, t: float):
        cx = OUT_SIZE[0] * 0.5
        cy = OUT_SIZE[1] * 0.34
        base_r = 135 * (OUT_SIZE[0] / 1080.0)
        moon_offset = lerp(240, -40, smoothstep((t - 0.8) / 4.0)) * (OUT_SIZE[0] / 1080.0)

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for scale, alpha in [(1.8, 18), (1.45, 38), (1.18, 78)]:
            r = base_r * scale
            gd.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(255, 180, 60, alpha))
        glow = glow.filter(ImageFilter.GaussianBlur(20 if not QUICK_MODE else 10))
        canvas.alpha_composite(glow)

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.ellipse((cx-base_r, cy-base_r, cx+base_r, cy+base_r), fill=(255, 194, 70, 255), outline=(255, 230, 160, 220), width=2)
        d.ellipse((cx+moon_offset-base_r*0.95, cy-base_r*0.95, cx+moon_offset+base_r*0.95, cy+base_r*0.95), fill=(6, 10, 16, 255))
        d.arc((cx-base_r*1.1, cy-base_r*1.1, cx+base_r*1.1, cy+base_r*1.1), start=220, end=325, fill=(95, 220, 245, 160), width=4)
        canvas.alpha_composite(layer)

        draw_text(canvas, "2 TO 5 GLOBAL SOLAR ECLIPSES PER YEAR", (int(cx), int(cy + base_r + 72 * (OUT_SIZE[0] / 1080.0))),
                  size=24 if not QUICK_MODE else 12, fill=(105, 232, 245, 230), bold=True, anchor="ma", stroke=1)

    def draw_timeline(self, canvas: Image.Image, t: float, large: bool = True):
        x_cols = {
            "Total": int(OUT_SIZE[0] * 0.18),
            "Annular": int(OUT_SIZE[0] * 0.41),
            "Hybrid": int(OUT_SIZE[0] * 0.64),
            "Partial": int(OUT_SIZE[0] * 0.87),
        }
        top = int(OUT_SIZE[1] * (0.24 if large else 0.30))
        bottom = int(OUT_SIZE[1] * (0.79 if large else 0.76))
        height = bottom - top
        reveal = smoothstep((t - (8.5 if not QUICK_MODE else 2.2)) / (10.0 if not QUICK_MODE else 2.2)) if large else 1.0
        draw = ImageDraw.Draw(canvas)

        for typ in TYPE_ORDER:
            x = x_cols[typ]
            draw.line((x, top, x, bottom), fill=TYPE_COLORS[typ] + (110,), width=2)
            draw_text(canvas, typ.upper(), (x, top - (34 if not QUICK_MODE else 18)), size=18 if not QUICK_MODE else 9,
                      fill=TYPE_COLORS[typ] + (220,), bold=True, anchor="ma", stroke=1)

        start_year = CONFIG["window_start"].year
        end_year = CONFIG["window_end"].year
        for year in range(start_year, end_year + 1, 10):
            frac = (year - start_year) / max(end_year - start_year, 1)
            y = top + int(frac * height)
            draw.line((int(OUT_SIZE[0] * 0.08), y, int(OUT_SIZE[0] * 0.92), y), fill=(90, 180, 205, 45), width=1)
            draw_text(canvas, str(year), (int(OUT_SIZE[0] * 0.04), y), size=16 if not QUICK_MODE else 8,
                      fill=(150, 205, 225, 200), anchor="lm", stroke=1)

        visible_points = int(len(self.timeline_points) * reveal)
        r = 8 if not QUICK_MODE else 4
        for idx, p in enumerate(self.timeline_points[:visible_points]):
            x = x_cols[p["type_group"]]
            y = top + int(p["y_frac"] * height)
            col = TYPE_COLORS[p["type_group"]]
            draw.ellipse((x-r, y-r, x+r, y+r), fill=col + (230,), outline=(255, 255, 255, 160))

        # moving scan cursor
        cursor_frac = (t / max(CONFIG["duration_s"], 1e-6))
        cursor_y = top + int(cursor_frac * height)
        draw.rectangle((int(OUT_SIZE[0] * 0.075), cursor_y-2, int(OUT_SIZE[0] * 0.925), cursor_y+2), fill=(255, 186, 90, 120))

    def draw_type_counts(self, canvas: Image.Image, t: float):
        x0 = int(OUT_SIZE[0] * 0.10)
        y0 = int(OUT_SIZE[1] * 0.32)
        max_width = int(OUT_SIZE[0] * 0.56)
        max_count = max(int(self.type_counts.max()), 1)
        step = 84 if not QUICK_MODE else 42
        draw_text(canvas, "TYPE BREAKDOWN", (x0, y0 - (58 if not QUICK_MODE else 28)), size=28 if not QUICK_MODE else 14,
                  fill=(105, 232, 245, 220), bold=True, stroke=1)
        draw = ImageDraw.Draw(canvas)
        reveal = smoothstep((t - (24.0 if not QUICK_MODE else 5.0)) / (5.0 if not QUICK_MODE else 1.0))
        for i, typ in enumerate(TYPE_ORDER):
            count = int(self.type_counts.get(typ, 0))
            bar_w = int(max_width * (count / max_count) * reveal)
            y = y0 + i * step
            color = TYPE_COLORS[typ]
            draw.rounded_rectangle((x0, y, x0 + max_width, y + (36 if not QUICK_MODE else 18)), radius=12 if not QUICK_MODE else 6,
                                   fill=(12, 24, 34, 165), outline=(80, 120, 145, 45))
            draw.rounded_rectangle((x0, y, x0 + max(bar_w, 1), y + (36 if not QUICK_MODE else 18)), radius=12 if not QUICK_MODE else 6,
                                   fill=color + (220,))
            draw_text(canvas, typ, (x0, y - (8 if not QUICK_MODE else 4)), size=22 if not QUICK_MODE else 11,
                      fill=(240, 245, 252, 240), bold=True, stroke=1)
            draw_text(canvas, str(count), (x0 + max_width + (22 if not QUICK_MODE else 10), y + (18 if not QUICK_MODE else 9)),
                      size=24 if not QUICK_MODE else 12, fill=color + (240,), bold=True, anchor="lm", stroke=1)

    def draw_yearly_cadence(self, canvas: Image.Image, t: float):
        x0 = int(OUT_SIZE[0] * 0.08)
        x1 = int(OUT_SIZE[0] * 0.94)
        y0 = int(OUT_SIZE[1] * 0.34)
        y1 = int(OUT_SIZE[1] * 0.78)
        draw = ImageDraw.Draw(canvas)
        draw_text(canvas, "ECLIPSE CADENCE BY YEAR", (x0, y0 - (62 if not QUICK_MODE else 30)), size=28 if not QUICK_MODE else 14,
                  fill=(255, 191, 95, 235), bold=True, stroke=1)
        years = list(self.per_year.index)
        n = len(years)
        bar_w = (x1 - x0) / max(n, 1)
        reveal = smoothstep((t - (38.0 if not QUICK_MODE else 8.0)) / (6.0 if not QUICK_MODE else 1.0))
        visible = max(1, int(n * reveal))
        for idx, year in enumerate(years[:visible]):
            count = int(self.per_year.loc[year])
            bh = (count / self.max_year_count) * (y1 - y0)
            left = x0 + idx * bar_w + 1
            right = left + max(bar_w - 2, 1)
            top = y1 - bh
            fill = (90, 225, 255, 210) if count >= 4 else (255, 188, 90, 190) if count == 3 else (155, 168, 185, 170)
            draw.rectangle((left, top, right, y1), fill=fill)
        for count in range(self.max_year_count + 1):
            yy = y1 - (count / max(self.max_year_count, 1)) * (y1 - y0)
            draw.line((x0, yy, x1, yy), fill=(100, 170, 200, 45), width=1)
            draw_text(canvas, str(count), (x0 - (18 if not QUICK_MODE else 10), int(yy)), size=14 if not QUICK_MODE else 7,
                      fill=(150, 205, 225, 180), anchor="rm", stroke=1)

        for year in range(2030, 2126, 10):
            idx = year - years[0]
            if 0 <= idx < n:
                xx = x0 + idx * bar_w
                draw_text(canvas, str(year), (int(xx), y1 + (22 if not QUICK_MODE else 10)), size=13 if not QUICK_MODE else 7,
                          fill=(160, 205, 225, 170), anchor="ma", stroke=1)

        draw_text(canvas, f"Most crowded year in window: {self.summary['max_eclipses_in_one_year']} eclipse(s)",
                  (x0, y1 + (60 if not QUICK_MODE else 28)), size=18 if not QUICK_MODE else 9,
                  fill=(240, 245, 252, 220), stroke=1)

    def draw_outro_cards(self, canvas: Image.Image):
        x0 = int(OUT_SIZE[0] * 0.08)
        y0 = int(OUT_SIZE[1] * 0.29)
        card_w = int(OUT_SIZE[0] * 0.84)
        card_h = 112 if not QUICK_MODE else 56
        gap = 28 if not QUICK_MODE else 14
        cards = []
        cards.append(("FIRST IN WINDOW", self.summary["first_eclipse"]))
        if self.summary.get("longest_central_event"):
            cards.append(("LONGEST CENTRAL EVENT", self.summary["longest_central_event"]))
        cards.append(("LAST IN WINDOW", self.summary["last_eclipse"]))
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for i, (title, data) in enumerate(cards):
            top = y0 + i * (card_h + gap)
            od.rounded_rectangle((x0, top, x0 + card_w, top + card_h), radius=22 if not QUICK_MODE else 11,
                                 fill=(2, 6, 16, 175), outline=(90, 200, 235, 80), width=1)
            draw_text(overlay, title, (x0 + 24, top + 22 if not QUICK_MODE else top + 10), size=22 if not QUICK_MODE else 11,
                      fill=(105, 230, 245, 230), bold=True, stroke=1)
            if title == "LONGEST CENTRAL EVENT":
                detail = f"{data['date']} // {data['type_group']} // {data['central_duration']}"
            else:
                detail = f"{data['date']} // {data['type_group']} // {data['raw_type']}"
            draw_wrapped_text(overlay, detail, (x0 + 24, top + 52 if not QUICK_MODE else top + 25), card_w - 48,
                              size=24 if not QUICK_MODE else 12, fill=(242, 246, 252, 235), bold=True)
        canvas.alpha_composite(overlay)

    def draw_source_banner(self, canvas: Image.Image):
        message = "SOURCE // NASA LIVE CATALOGS" if self.source == "live_nasa_catalog" else "PREVIEW SOURCE // OFFLINE FIXTURE SUBSET"
        color = (105, 232, 245, 235) if self.source == "live_nasa_catalog" else (255, 190, 95, 235)
        draw_text(canvas, message, (OUT_SIZE[0] - (48 if not QUICK_MODE else 24), 78 if not QUICK_MODE else 40),
                  size=18 if not QUICK_MODE else 9, fill=color, bold=True, stroke=1, anchor="ra")

    def draw_corner_stats(self, canvas: Image.Image):
        n = self.summary["num_eclipses"]
        draw_text(canvas, f"WINDOW // {CONFIG['window_start'].year}-{CONFIG['window_end'].year}",
                  (OUT_SIZE[0] - (48 if not QUICK_MODE else 24), 116 if not QUICK_MODE else 58),
                  size=17 if not QUICK_MODE else 8, fill=(160, 208, 228, 210), anchor="ra", stroke=1)
        draw_text(canvas, f"CATALOG ROWS // {n}", (OUT_SIZE[0] - (48 if not QUICK_MODE else 24), 146 if not QUICK_MODE else 73),
                  size=17 if not QUICK_MODE else 8, fill=(160, 208, 228, 200), anchor="ra", stroke=1)

    def draw_title_and_captions(self, canvas: Image.Image, t: float, shot_name: str):
        title_alpha = int(255 * smoothstep((t - 0.2) / 0.8) * (1.0 - smoothstep((t - (6.8 if not QUICK_MODE else 1.7)) / 0.7)))
        if title_alpha > 4:
            draw_text(canvas, CONFIG["title_text"], (56 if not QUICK_MODE else 28, 98 if not QUICK_MODE else 48), size=46 if not QUICK_MODE else 21,
                      fill=(245, 248, 252, title_alpha), bold=True)
            draw_text(canvas, CONFIG["subtitle_text"], (58 if not QUICK_MODE else 30, 160 if not QUICK_MODE else 79), size=24 if not QUICK_MODE else 11,
                      fill=(102, 232, 245, min(title_alpha, 225)), bold=True)

        shot_titles = {
            "intro": "SOLAR ECLIPSE GEOMETRY // GLOBAL",
            "timeline": "CENTURY TIMELINE // EVERY EVENT",
            "counts": "TYPE MIX // PARTIAL, ANNULAR, TOTAL, HYBRID",
            "cadence": "YEARLY CADENCE // HOW OFTEN THEY HAPPEN",
            "outro": "THE NEXT CENTURY OF ALIGNMENTS",
        }
        if t > (5.3 if not QUICK_MODE else 1.4):
            draw_text(canvas, shot_titles.get(shot_name, ""), (56 if not QUICK_MODE else 28, 64 if not QUICK_MODE else 31), size=20 if not QUICK_MODE else 10,
                      fill=(145, 210, 232, 205), bold=True, stroke=1)

        caption = caption_at(t)
        if caption:
            y0 = OUT_SIZE[1] - (244 if not QUICK_MODE else 124)
            panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            d = ImageDraw.Draw(panel)
            d.rounded_rectangle((44 if not QUICK_MODE else 22, y0, OUT_SIZE[0] - (44 if not QUICK_MODE else 22), y0 + (122 if not QUICK_MODE else 64)),
                                radius=24 if not QUICK_MODE else 12,
                                fill=(1, 5, 14, 170), outline=(62, 182, 220, 60), width=1)
            canvas.alpha_composite(panel)
            draw_wrapped_text(canvas, caption, (68 if not QUICK_MODE else 34, y0 + (27 if not QUICK_MODE else 13)),
                              max_width=OUT_SIZE[0] - (136 if not QUICK_MODE else 68), size=30 if not QUICK_MODE else 14,
                              fill=(245, 250, 255, 245))

    def draw_hud_noise(self, canvas: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for item in self.hud_noise:
            pulse = 0.5 + 0.5 * math.sin(1.9 * t + item["phase"])
            if pulse < 0.72:
                continue
            x = item["x"]
            y = (item["y"] + t * 10.0) % OUT_SIZE[1]
            draw.line((x, y, x + item["length"], y), fill=(90, 210, 240, int(item["alpha"] * pulse)), width=1)
        offset = int((t * 37) % 7)
        for y in range(offset, OUT_SIZE[1], 7):
            draw.line((0, y, OUT_SIZE[0], y), fill=(120, 205, 245, 12), width=1)
        scan_y = int((t * 156) % (OUT_SIZE[1] + 240)) - 120
        draw.rectangle((0, scan_y, OUT_SIZE[0], scan_y + (48 if not QUICK_MODE else 24)), fill=(80, 210, 240, 8))
        canvas.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        canvas = self.render_background(t)
        shot_name = shot["name"]

        if shot_name == "intro":
            self.draw_sun_moon_hook(canvas, t)
            self.draw_timeline(canvas, t, large=False)
        elif shot_name == "timeline":
            self.draw_timeline(canvas, t, large=True)
        elif shot_name == "counts":
            self.draw_timeline(canvas, t, large=False)
            self.draw_type_counts(canvas, t)
        elif shot_name == "cadence":
            self.draw_yearly_cadence(canvas, t)
        elif shot_name == "outro":
            self.draw_outro_cards(canvas)
            self.draw_timeline(canvas, t, large=False)

        self.draw_source_banner(canvas)
        self.draw_corner_stats(canvas)
        self.draw_title_and_captions(canvas, t, shot_name)
        self.draw_hud_noise(canvas, t)

        arr = np.array(canvas.convert("RGB"))
        arr = apply_grade(arr)
        arr = np.clip(arr.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / 0.9)
        fade_out = 1.0 - smoothstep((t - (CONFIG["duration_s"] - 1.1)) / 1.0)
        arr = np.clip(arr.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)
        return arr


# %% Render

def render_video(scene: EclipseCenturyScene):
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    print("Subtitle sidecar written:", srt_path.resolve())
    raw_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    final_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"

    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    times = np.arange(frame_count) / CONFIG["fps"]
    print(f"Rendering {frame_count:,} frames at {CONFIG['video_width']}x{CONFIG['video_height']} ...")
    with iio.get_writer(raw_video, fps=CONFIG["fps"], codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None) as writer:
        for t in tqdm(times, desc="Rendering eclipse century short"):
            writer.append_data(scene.render_frame(float(t)))
    shutil.copyfile(raw_video, final_video)
    print("Final video:", final_video.resolve())
    return final_video


# %% Main

def main():
    print("Loading solar eclipse catalog ...")
    df_all, source, error_note = load_eclipse_catalog()
    print("Source:", source)
    if error_note:
        print("Live fetch note:", error_note)

    df = filter_window(df_all)
    if df.empty:
        raise RuntimeError("No eclipses left after filtering to 2026-2125")

    summary = create_summary(df, source)
    save_data_products(df, summary, source, error_note)
    create_scientific_plots(df, summary)

    preview_times = [1.0, min(11.0, CONFIG["duration_s"] * 0.2), min(22.0, CONFIG["duration_s"] * 0.4), min(34.0, CONFIG["duration_s"] * 0.58), min(47.0, CONFIG["duration_s"] * 0.81), CONFIG["duration_s"] - 1.0]
    scene = EclipseCenturyScene(df, summary, source)
    for pt in tqdm(preview_times, desc="Preview frames"):
        frame = scene.render_frame(float(pt))
        Image.fromarray(frame).save(PREVIEW_DIR / f"preview_{int(pt):02d}s.png")
    print("Preview frames written:", PREVIEW_DIR.resolve())

    render_video(scene)

    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-", path.name)


if __name__ == "__main__":
    main()
