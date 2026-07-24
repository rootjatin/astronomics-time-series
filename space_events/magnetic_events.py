from __future__ import annotations

"""
The Sun's Magnetic Field Keeps Flipping — cinematic YouTube Short renderer

Creates a vertical 1080x1920 astronomy short explaining solar magnetic-pole
reversals. The preferred live data source is the Wilcox Solar Observatory (WSO)
polar-field record, which began in 1976 and measures the northern and southern
polar caps. When the live source is unavailable, a clearly labeled fallback model
based on published reversal epochs is used for local preview/testing only.

Science story:
- The Sun's global magnetic polarity reverses around solar maximum, roughly every
  11 years.
- Returning to the original magnetic orientation takes about 22 years: the Hale
  magnetic cycle.
- The two poles do not necessarily reverse at the same time.
- During reversal, polar fields weaken toward zero and can change sign more than
  once before a new polarity becomes established.
- "Flip" does not mean the physical rotation axis or geographic poles move.

Install:
    pip install numpy pandas matplotlib pillow imageio imageio-ffmpeg requests beautifulsoup4 tqdm

Quick render:
    SOLAR_FLIP_SHORT_QUICK=1 python the_suns_magnetic_field_keeps_flipping_short.py
"""

import hashlib
import io
import json
import math
import os
import re
import shutil
from datetime import datetime
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


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("SOLAR_FLIP_SHORT_QUICK", "0") == "1"
OUTPUT_ROOT = Path("solar_magnetic_flip_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in [OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "the_suns_magnetic_field_keeps_flipping",
    "wso_url": "https://wso.stanford.edu/Polar.html",
    "title": "THE SUN'S MAGNETIC FIELD KEEPS FLIPPING",
    "subtitle": "Polar-field reversals across the solar cycle",
    "background_stars": 300,
    "hud_noise": 48,
    "contrast": 1.09,
    "saturation": 1.07,
    "vignette": 0.25,
}

OUT_W = CONFIG["video_width"]
OUT_H = CONFIG["video_height"]
OUT_SIZE = (OUT_W, OUT_H)

FULL_CAPTIONS = [
    (0.5, 6.8, "About every eleven years, the Sun's global magnetic north and south poles reverse."),
    (6.9, 15.0, "Wilcox Solar Observatory measurements track the northern and southern polar fields separately."),
    (15.1, 24.0, "Near solar maximum, both fields weaken toward zero—and the two hemispheres may reverse at different times."),
    (24.1, 34.0, "Decaying magnetic regions are carried toward the poles, cancelling the old field and building the new one."),
    (34.1, 44.5, "One polarity reversal takes about eleven years, but returning to the original orientation takes roughly twenty-two."),
    (44.6, 57.2, "The Sun is not physically turning over. Its magnetic field is rebuilding itself on a stellar scale."),
]
if QUICK_MODE:
    scale = CONFIG["duration_s"] / 58.0
    CAPTIONS = [(a * scale, b * scale, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 8.0 if not QUICK_MODE else 2.0},
    {"name": "wso_data", "start": 8.0 if not QUICK_MODE else 2.0, "end": 24.0 if not QUICK_MODE else 5.0},
    {"name": "mechanism", "start": 24.0 if not QUICK_MODE else 5.0, "end": 38.0 if not QUICK_MODE else 8.0},
    {"name": "hale_cycle", "start": 38.0 if not QUICK_MODE else 8.0, "end": 49.5 if not QUICK_MODE else 10.0},
    {"name": "outro", "start": 49.5 if not QUICK_MODE else 10.0, "end": CONFIG["duration_s"]},
]

POLAR_COLORS = {
    "north": (95, 220, 255),
    "south": (255, 105, 145),
    "positive": (95, 220, 255),
    "negative": (255, 105, 145),
    "neutral": (230, 235, 240),
}

# Published approximate completed-reversal epochs for older cycles, plus recent
# Cycle 25 transition markers used only for the fallback model and callouts.
REVERSAL_MILESTONES = [
    {"cycle": 21, "north_year": 1981.0, "south_year": 1981.55, "note": "Cycle 21 reversal"},
    {"cycle": 22, "north_year": 1990.5, "south_year": 1991.2, "note": "Cycle 22 reversal"},
    {"cycle": 23, "north_year": 2000.6, "south_year": 2002.0, "note": "Cycle 23 reversal"},
    {"cycle": 24, "north_year": 2012.5, "south_year": 2013.55, "note": "Cycle 24 reversal"},
    {"cycle": 25, "north_year": 2023.4, "south_year": 2023.75, "note": "Cycle 25 first sign changes"},
]


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


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


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(img: Image.Image, text: str, xy: Tuple[int, int], size: int = 28,
              fill=(255, 255, 255, 255), bold: bool = False,
              stroke: int = 2, anchor: str = "la"):
    draw = ImageDraw.Draw(img)
    draw.text(
        xy,
        text,
        font=get_font(size, bold=bold),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(220, fill[3] if len(fill) > 3 else 220)),
    )


def draw_wrapped_text(img: Image.Image, text: str, xy: Tuple[int, int], max_width: int,
                      size: int = 28, fill=(255, 255, 255, 245), bold: bool = False,
                      line_spacing: int = 6):
    draw = ImageDraw.Draw(img)
    font = get_font(size, bold=bold)
    words = text.split()
    lines: List[str] = []
    cur = ""
    for word in words:
        candidate = word if not cur else f"{cur} {word}"
        bb = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)
        if bb[2] - bb[0] <= max_width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 220))
        bb = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += (bb[3] - bb[1]) + line_spacing


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    rr = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * rr**1.8, 0.0, 1.0).astype(np.float32)


def apply_grade(arr: np.ndarray) -> np.ndarray:
    img = Image.fromarray(arr)
    img = ImageEnhance.Contrast(img).enhance(CONFIG["contrast"])
    img = ImageEnhance.Color(img).enhance(CONFIG["saturation"])
    return np.array(img)


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(captions, path: Path):
    lines = []
    for i, (start, end, text) in enumerate(captions, start=1):
        lines.extend([str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


VIGNETTE = make_vignette(OUT_W, OUT_H, CONFIG["vignette"])


# -----------------------------------------------------------------------------
# WSO data loading and fallback model
# -----------------------------------------------------------------------------

def decimal_year_from_timestamp(ts: pd.Timestamp) -> float:
    year_start = pd.Timestamp(year=ts.year, month=1, day=1)
    next_start = pd.Timestamp(year=ts.year + 1, month=1, day=1)
    return ts.year + (ts - year_start).total_seconds() / (next_start - year_start).total_seconds()


def infer_polar_table_from_html(html: str) -> pd.DataFrame:
    """Try several parsers because the WSO page format is unusual and long."""
    candidates: List[pd.DataFrame] = []

    # 1) HTML tables, if present.
    try:
        for table in pd.read_html(io.StringIO(html)):
            if table.shape[1] >= 3:
                candidates.append(table)
    except Exception:
        pass

    # 2) Preformatted / body text rows. WSO pages have historically included
    # date strings resembling 1976:06:10_21h:07m:13s followed by polar values.
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text("\n")
    else:
        text = re.sub(r"<[^>]+>", "\n", html)

    rows = []
    date_re = re.compile(r"(19|20)\d{2}[:/-]\d{2}[:/-]\d{2}(?:[_ T]\d{2}h?:?\d{2}m?:?\d{2}s?)?")
    number_re = re.compile(r"[-+]?\d+(?:\.\d+)?")
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        m = date_re.search(line)
        if not m:
            continue
        nums = number_re.findall(line[m.end():])
        if len(nums) < 2:
            continue
        date_token = m.group(0)
        cleaned = date_token.replace("_", " ").replace("h", ":").replace("m", ":").replace("s", "")
        cleaned = cleaned.replace("/", "-").replace(":", "-", 2)
        try:
            ts = pd.to_datetime(cleaned, errors="raise")
        except Exception:
            try:
                y, mo, da = map(int, re.findall(r"\d+", date_token)[:3])
                ts = pd.Timestamp(year=y, month=mo, day=da)
            except Exception:
                continue
        vals = [float(x) for x in nums]
        rows.append({"date": ts, "north_raw": vals[0], "south_raw": vals[1]})

    if rows:
        candidates.append(pd.DataFrame(rows))

    def normalize_candidate(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        df = df.copy()
        # Flatten multi-index columns.
        df.columns = [" ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c) for c in df.columns]
        lower = {c: c.lower() for c in df.columns}
        date_col = next((c for c in df.columns if "date" in lower[c] or "time" in lower[c]), None)
        north_col = next((c for c in df.columns if "north" in lower[c] or re.search(r"\bn\b", lower[c])), None)
        south_col = next((c for c in df.columns if "south" in lower[c] or re.search(r"\bs\b", lower[c])), None)
        if date_col and north_col and south_col:
            out = pd.DataFrame({
                "date": pd.to_datetime(df[date_col], errors="coerce"),
                "north_raw": pd.to_numeric(df[north_col], errors="coerce"),
                "south_raw": pd.to_numeric(df[south_col], errors="coerce"),
            })
        elif {"date", "north_raw", "south_raw"}.issubset(df.columns):
            out = df[["date", "north_raw", "south_raw"]].copy()
        else:
            return None
        out = out.dropna().sort_values("date").drop_duplicates("date")
        if len(out) < 30:
            return None
        return out

    normalized = [x for x in (normalize_candidate(c) for c in candidates) if x is not None]
    if not normalized:
        raise RuntimeError("Could not identify a usable north/south polar-field table")
    return max(normalized, key=len).reset_index(drop=True)


def fetch_live_wso() -> Tuple[pd.DataFrame, str]:
    if requests is None:
        raise RuntimeError("requests package not available")
    response = requests.get(CONFIG["wso_url"], timeout=50)
    response.raise_for_status()
    df = infer_polar_table_from_html(response.text)

    # Normalize units to the plotting convention used by WSO pages: 0.01 gauss.
    # The page has historically published values in microtesla / 0.01 G-like
    # units. We preserve raw values and use normalized plotting units.
    df["decimal_year"] = df["date"].map(decimal_year_from_timestamp)
    df["north"] = df["north_raw"].astype(float)
    df["south"] = df["south_raw"].astype(float)
    df["average"] = (df["north"] - df["south"]) / 2.0
    df["source"] = "live_wso"
    return df, "live_wso"


def fallback_polar_model() -> Tuple[pd.DataFrame, str]:
    """Deterministic preview model anchored to published reversal epochs."""
    dates = pd.date_range("1976-06-01", "2026-06-01", freq="10D")
    years = np.array([decimal_year_from_timestamp(pd.Timestamp(d)) for d in dates])

    # Piecewise sign changes at approximate reversal epochs, with cycle-dependent
    # strengths and annual viewing modulation. This is not a substitute for WSO.
    north = np.zeros_like(years)
    south = np.zeros_like(years)
    north_flips = [1981.0, 1990.5, 2000.6, 2012.5, 2023.4]
    south_flips = [1981.55, 1991.2, 2002.0, 2013.55, 2023.75]

    def field_from_flips(x: np.ndarray, flips: List[float], initial_sign: float, phase: float) -> np.ndarray:
        sign = np.full_like(x, initial_sign, dtype=float)
        for flip in flips:
            sign *= np.where(x >= flip, -1.0, 1.0)
        # Polar field strong near minima and weak near reversals.
        distance = np.full_like(x, 20.0)
        for flip in flips:
            distance = np.minimum(distance, np.abs(x - flip))
        amplitude = 18.0 + 78.0 * np.tanh(distance / 2.3)
        annual = 9.0 * np.sin(2 * math.pi * x + phase)
        long_wave = 7.0 * np.sin(2 * math.pi * (x - 1976.0) / 10.8 + phase * 0.4)
        return sign * amplitude + annual + long_wave

    north = field_from_flips(years, north_flips, 1.0, 0.2)
    south = field_from_flips(years, south_flips, -1.0, 2.1)

    df = pd.DataFrame({
        "date": dates,
        "decimal_year": years,
        "north_raw": north,
        "south_raw": south,
        "north": north,
        "south": south,
        "average": (north - south) / 2.0,
        "source": "fallback_reversal_model",
    })
    return df, "fallback_reversal_model"


def load_polar_data() -> Tuple[pd.DataFrame, str, Optional[str]]:
    try:
        df, source = fetch_live_wso()
        return df, source, None
    except Exception as exc:
        df, source = fallback_polar_model()
        return df, source, str(exc)


def detect_zero_crossings(df: pd.DataFrame, column: str) -> List[float]:
    values = df[column].to_numpy(float)
    years = df["decimal_year"].to_numpy(float)
    out = []
    for i in range(1, len(values)):
        if not np.isfinite(values[i-1:i+1]).all():
            continue
        if values[i-1] == 0 or values[i] == 0 or np.sign(values[i-1]) != np.sign(values[i]):
            denom = abs(values[i-1]) + abs(values[i])
            frac = abs(values[i-1]) / denom if denom else 0.5
            out.append(float(lerp(years[i-1], years[i], frac)))
    # Keep crossings separated by at least 0.25 years to avoid counting every
    # wobble around zero as a distinct global-cycle reversal.
    filtered = []
    for x in out:
        if not filtered or x - filtered[-1] >= 0.25:
            filtered.append(x)
    return filtered


def summarize_data(df: pd.DataFrame, source: str) -> Dict:
    north_cross = detect_zero_crossings(df, "north")
    south_cross = detect_zero_crossings(df, "south")
    return {
        "source": source,
        "start_date": df.iloc[0]["date"].strftime("%Y-%m-%d"),
        "end_date": df.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "rows": int(len(df)),
        "north_zero_crossings": [round(x, 3) for x in north_cross],
        "south_zero_crossings": [round(x, 3) for x in south_cross],
        "latest_north": float(df.iloc[-1]["north"]),
        "latest_south": float(df.iloc[-1]["south"]),
        "latest_average": float(df.iloc[-1]["average"]),
        "milestones": REVERSAL_MILESTONES,
    }


def save_data_products(df: pd.DataFrame, summary: Dict, error_note: Optional[str]):
    csv_path = DATA_ROOT / "solar_polar_fields_1976_present.csv"
    df.to_csv(csv_path, index=False)
    json_path = DATA_ROOT / "solar_polar_fields_summary.json"
    json_path.write_text(json.dumps({
        "summary": summary,
        "error_note": error_note,
        "live_source_url": CONFIG["wso_url"],
        "fallback_warning": "Fallback model is only for preview/layout validation and is not observational data.",
    }, indent=2), encoding="utf-8")
    return csv_path, json_path


def create_scientific_plots(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(df["decimal_year"], df["north"], label="North polar field")
    ax.plot(df["decimal_year"], df["south"], label="South polar field")
    ax.axhline(0, linewidth=1)
    ax.set_title("Solar polar-field polarity through time")
    ax.set_xlabel("Year")
    ax.set_ylabel("WSO-like polar-field units")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "polar_field_time_series.png", dpi=170)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class MagneticFlipScene:
    def __init__(self, df: pd.DataFrame, summary: Dict, source: str):
        self.df = df.copy().reset_index(drop=True)
        self.summary = summary
        self.source = source
        self.stars = self._make_stars(CONFIG["background_stars"], 33)
        self.hud = self._make_hud(CONFIG["hud_noise"], 77)
        self.y_min = float(min(self.df["north"].min(), self.df["south"].min()))
        self.y_max = float(max(self.df["north"].max(), self.df["south"].max()))
        m = max(abs(self.y_min), abs(self.y_max), 1.0)
        self.y_min, self.y_max = -m, m

    @staticmethod
    def _make_stars(n: int, seed: int):
        rng = np.random.default_rng(seed)
        return [{
            "x": float(rng.uniform(0, OUT_W)), "y": float(rng.uniform(0, OUT_H)),
            "r": float(rng.uniform(0.5, 2.0)), "a": int(rng.integers(16, 95)),
            "phase": float(rng.uniform(0, 2 * math.pi)),
        } for _ in range(n)]

    @staticmethod
    def _make_hud(n: int, seed: int):
        rng = np.random.default_rng(seed)
        return [{
            "x": float(rng.uniform(0, OUT_W)), "y": float(rng.uniform(0, OUT_H)),
            "length": float(rng.uniform(12, 90)), "a": int(rng.integers(8, 42)),
            "phase": float(rng.uniform(0, 2 * math.pi)),
        } for _ in range(n)]

    def background(self, t: float) -> Image.Image:
        img = Image.new("RGBA", OUT_SIZE, (3, 7, 14, 255))
        d = ImageDraw.Draw(img)
        for s in self.stars:
            alpha = int(s["a"] * (0.72 + 0.28 * math.sin(t * 1.6 + s["phase"])))
            d.ellipse((s["x"]-s["r"], s["y"]-s["r"], s["x"]+s["r"], s["y"]+s["r"]), fill=(215, 230, 255, alpha))
        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for cx, cy, col in [
            (OUT_W*0.22, OUT_H*0.28, (80, 28, 95)),
            (OUT_W*0.75, OUT_H*0.36, (10, 78, 118)),
            (OUT_W*0.52, OUT_H*0.77, (65, 26, 35)),
        ]:
            for rr, aa in [(420*OUT_W/1080, 16), (280*OUT_W/1080, 25), (170*OUT_W/1080, 34)]:
                hd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=(col[0], col[1], col[2], aa))
        haze = haze.filter(ImageFilter.GaussianBlur(62 if not QUICK_MODE else 31))
        img.alpha_composite(haze)
        return img

    def draw_sun(self, img: Image.Image, center: Tuple[float, float], radius: float,
                 polarity_phase: float, activity: float, t: float):
        cx, cy = center
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for scale, aa in [(1.65, 18), (1.38, 38), (1.17, 75)]:
            rr = radius * scale
            gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=(255, 150, 45, aa))
        glow = glow.filter(ImageFilter.GaussianBlur(22 if not QUICK_MODE else 11))
        img.alpha_composite(glow)

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for i in range(22, 0, -1):
            frac = i / 22.0
            rr = radius * frac
            d.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=(255, int(132+78*frac), int(28+55*frac), 255))
        d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=(255, 235, 185, 155), width=2)

        # Active regions / sunspots peak near reversal.
        spot_count = int(2 + 12 * clamp(activity))
        seed = int((t * 10) + polarity_phase * 1000) + 12345
        rng = np.random.default_rng(seed)
        for _ in range(spot_count):
            ang = rng.uniform(0, 2*math.pi)
            rr = radius * math.sqrt(rng.uniform(0.05, 0.70))
            x = cx + math.cos(ang) * rr
            y = cy + math.sin(ang) * rr * 0.72
            sr = rng.uniform(radius*0.014, radius*0.035)
            d.ellipse((x-sr*2, y-sr*1.5, x+sr*2, y+sr*1.5), fill=(85, 45, 18, 120))
            d.ellipse((x-sr, y-sr, x+sr, y+sr), fill=(20, 16, 15, 235))

        img.alpha_composite(layer)
        self.draw_magnetic_field_lines(img, center, radius, polarity_phase, activity, t)

    def draw_magnetic_field_lines(self, img: Image.Image, center: Tuple[float, float], radius: float,
                                  polarity_phase: float, activity: float, t: float):
        cx, cy = center
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        # polarity_phase: -1 means north negative, +1 north positive, 0 confused.
        mix = clamp(abs(polarity_phase))
        north_col = POLAR_COLORS["positive"] if polarity_phase >= 0 else POLAR_COLORS["negative"]
        south_col = POLAR_COLORS["negative"] if polarity_phase >= 0 else POLAR_COLORS["positive"]

        n_lines = 9
        for i in range(n_lines):
            frac = (i + 1) / (n_lines + 1)
            spread = radius * (0.35 + 1.05 * frac)
            wobble = (1.0-mix) * radius * 0.34 * math.sin(t*1.8 + i)
            points = []
            for k in range(41):
                u = k / 40.0
                theta = math.pi * u
                x = cx + math.cos(theta) * spread + wobble * math.sin(theta*2 + i)
                y = cy - math.sin(theta) * radius * (1.3 + 0.7*frac)
                points.append((x, y))
            col = tuple(int(lerp(north_col[j], south_col[j], frac)) for j in range(3))
            d.line(points, fill=col + (int(70 + 70*mix),), width=2)

            points2 = [(x, 2*cy-y) for x, y in points]
            d.line(points2, fill=col + (int(65 + 70*mix),), width=2)

        # Polar caps
        cap_h = radius * 0.20
        d.ellipse((cx-radius*0.42, cy-radius*0.98, cx+radius*0.42, cy-radius*0.98+cap_h), fill=north_col + (90,))
        d.ellipse((cx-radius*0.42, cy+radius*0.98-cap_h, cx+radius*0.42, cy+radius*0.98), fill=south_col + (90,))
        img.alpha_composite(overlay)

    def draw_bar_magnet_labels(self, img: Image.Image, center: Tuple[float, float], radius: float, polarity: float):
        cx, cy = center
        north_symbol = "N+" if polarity >= 0 else "N−"
        south_symbol = "S−" if polarity >= 0 else "S+"
        draw_text(img, north_symbol, (int(cx), int(cy-radius-42*OUT_W/1080)), size=28 if not QUICK_MODE else 14,
                  fill=POLAR_COLORS["north"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(img, south_symbol, (int(cx), int(cy+radius+42*OUT_W/1080)), size=28 if not QUICK_MODE else 14,
                  fill=POLAR_COLORS["south"] + (240,), bold=True, anchor="ma", stroke=1)

    def field_row_at_fraction(self, frac: float) -> pd.Series:
        idx = int(round(clamp(frac) * (len(self.df)-1)))
        return self.df.iloc[idx]

    def draw_data_chart(self, img: Image.Image, t: float):
        x0 = int(OUT_W*0.07)
        x1 = int(OUT_W*0.93)
        y0 = int(OUT_H*0.25)
        y1 = int(OUT_H*0.76)
        w = x1-x0
        h = y1-y0
        overlay = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        d = ImageDraw.Draw(overlay)
        d.rounded_rectangle((x0, y0, x1, y1), radius=24 if not QUICK_MODE else 12,
                            fill=(3, 7, 15, 165), outline=(90, 175, 205, 60), width=1)
        # axes
        zero_y = y0 + int((self.y_max/(self.y_max-self.y_min))*h)
        d.line((x0+28, zero_y, x1-18, zero_y), fill=(230,235,240,90), width=1)
        for year in range(1980, 2030, 10):
            frac = (year - self.df["decimal_year"].iloc[0]) / (self.df["decimal_year"].iloc[-1] - self.df["decimal_year"].iloc[0])
            xx = x0+28 + frac*(w-46)
            d.line((xx, y0+20, xx, y1-24), fill=(100,170,195,35), width=1)
            draw_text(overlay, str(year), (int(xx), y1-14), size=14 if not QUICK_MODE else 7,
                      fill=(160,205,222,190), anchor="ma", stroke=1)

        reveal = smoothstep((t-(8.2 if not QUICK_MODE else 2.0))/(10.0 if not QUICK_MODE else 2.2))
        n = max(2, int(len(self.df)*reveal))
        sub = self.df.iloc[:n]
        xs = x0+28 + (sub["decimal_year"].to_numpy()-self.df["decimal_year"].iloc[0])/(self.df["decimal_year"].iloc[-1]-self.df["decimal_year"].iloc[0])*(w-46)
        def map_y(vals):
            return y0+20 + (self.y_max-vals)/(self.y_max-self.y_min)*(h-48)
        north_y = map_y(sub["north"].to_numpy(float))
        south_y = map_y(sub["south"].to_numpy(float))
        d.line(list(zip(xs, north_y)), fill=POLAR_COLORS["north"]+(230,), width=3 if not QUICK_MODE else 2)
        d.line(list(zip(xs, south_y)), fill=POLAR_COLORS["south"]+(230,), width=3 if not QUICK_MODE else 2)

        # Reversal markers from milestone list.
        for milestone in REVERSAL_MILESTONES:
            for key, col in [("north_year", POLAR_COLORS["north"]), ("south_year", POLAR_COLORS["south"])]:
                yr = milestone[key]
                frac = (yr-self.df["decimal_year"].iloc[0])/(self.df["decimal_year"].iloc[-1]-self.df["decimal_year"].iloc[0])
                if 0 <= frac <= 1:
                    xx = x0+28+frac*(w-46)
                    d.line((xx, zero_y-8, xx, zero_y+8), fill=col+(180,), width=2)
        img.alpha_composite(overlay)
        draw_text(img, "WSO POLAR-FIELD RECORD", (x0+22, y0+18), size=21 if not QUICK_MODE else 10,
                  fill=(110,230,245,235), bold=True, stroke=1)
        draw_text(img, "North", (x0+22, y0+54 if not QUICK_MODE else y0+27), size=17 if not QUICK_MODE else 8,
                  fill=POLAR_COLORS["north"]+(235,), bold=True, stroke=1)
        draw_text(img, "South", (x0+110 if not QUICK_MODE else x0+56, y0+54 if not QUICK_MODE else y0+27), size=17 if not QUICK_MODE else 8,
                  fill=POLAR_COLORS["south"]+(235,), bold=True, stroke=1)
        draw_text(img, "zero field / reversal zone", (x1-18, zero_y-10), size=15 if not QUICK_MODE else 7,
                  fill=(225,232,240,200), anchor="ra", stroke=1)

    def draw_mechanism(self, img: Image.Image, t: float):
        cx, cy = OUT_W*0.5, OUT_H*0.40
        radius = 190*OUT_W/1080
        phase = (t-(24 if not QUICK_MODE else 5))/(14 if not QUICK_MODE else 3)
        polarity = math.cos(math.pi*clamp(phase))
        activity = math.sin(math.pi*clamp(phase))
        self.draw_sun(img, (cx, cy), radius, polarity, activity, t)
        self.draw_bar_magnet_labels(img, (cx, cy), radius, polarity)

        overlay = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        d = ImageDraw.Draw(overlay)
        # Poleward flux arrows from sunspot belts.
        for side in [-1, 1]:
            for hemi in [-1, 1]:
                start_x = cx + side*radius*0.55
                start_y = cy + hemi*radius*0.32
                end_x = cx + side*radius*0.18
                end_y = cy + hemi*radius*0.92
                d.line((start_x, start_y, end_x, end_y), fill=(255,215,125,170), width=4)
                d.polygon([(end_x,end_y),(end_x-10,end_y-hemi*18),(end_x+10,end_y-hemi*18)], fill=(255,215,125,190))
        img.alpha_composite(overlay)
        draw_text(img, "DECAYING ACTIVE-REGION FLUX", (int(cx), int(cy+radius+92*OUT_W/1080)), size=22 if not QUICK_MODE else 11,
                  fill=(255,195,105,235), bold=True, anchor="ma", stroke=1)
        draw_text(img, "moves poleward → cancels old field → builds new field", (int(cx), int(cy+radius+130*OUT_W/1080)), size=18 if not QUICK_MODE else 9,
                  fill=(220,235,245,220), anchor="ma", stroke=1)

    def draw_hale_cycle(self, img: Image.Image, t: float):
        y = int(OUT_H*0.43)
        x0 = int(OUT_W*0.10)
        x1 = int(OUT_W*0.90)
        d = ImageDraw.Draw(img)
        d.line((x0,y,x1,y), fill=(125,205,230,150), width=3)
        positions = [x0, int(lerp(x0,x1,0.5)), x1]
        labels = ["START", "~11 YEARS", "~22 YEARS"]
        polarities = [1,-1,1]
        for i,(x,label,pol) in enumerate(zip(positions, labels, polarities)):
            d.ellipse((x-22*OUT_W/1080,y-22*OUT_W/1080,x+22*OUT_W/1080,y+22*OUT_W/1080), fill=(5,10,18,255), outline=(255,190,90,190), width=3)
            draw_text(img, label, (int(x), y+52 if not QUICK_MODE else y+26), size=20 if not QUICK_MODE else 10,
                      fill=(245,248,252,235), bold=True, anchor="ma", stroke=1)
            draw_text(img, "N+ / S−" if pol>0 else "N− / S+", (int(x), y-62 if not QUICK_MODE else y-31), size=22 if not QUICK_MODE else 11,
                      fill=(110,230,245,235) if pol>0 else (255,120,155,235), bold=True, anchor="ma", stroke=1)
        d.arc((positions[0], y-170*OUT_W/1080, positions[1], y+170*OUT_W/1080), 190, 350, fill=(255,190,90,150), width=3)
        d.arc((positions[1], y-170*OUT_W/1080, positions[2], y+170*OUT_W/1080), 190, 350, fill=(255,190,90,150), width=3)
        draw_text(img, "11-YEAR SUNSPOT CYCLE", (OUT_W//2, int(OUT_H*0.24)), size=30 if not QUICK_MODE else 14,
                  fill=(255,190,90,235), bold=True, anchor="ma", stroke=1)
        draw_text(img, "22-YEAR HALE MAGNETIC CYCLE", (OUT_W//2, int(OUT_H*0.66)), size=32 if not QUICK_MODE else 15,
                  fill=(110,230,245,240), bold=True, anchor="ma", stroke=1)
        draw_wrapped_text(img, "One flip reverses the poles. A second flip restores the original magnetic orientation.",
                          (int(OUT_W*0.12), int(OUT_H*0.71)), int(OUT_W*0.76), size=22 if not QUICK_MODE else 11,
                          fill=(238,244,250,230), bold=True)

    def draw_outro(self, img: Image.Image, t: float):
        cx, cy = OUT_W*0.5, OUT_H*0.34
        radius = 168*OUT_W/1080
        phase = math.sin(t*0.45)
        activity = 0.35+0.25*math.sin(t*0.9)
        self.draw_sun(img, (cx,cy), radius, phase, activity, t)
        self.draw_bar_magnet_labels(img, (cx,cy), radius, phase)
        x0 = int(OUT_W*0.10)
        y0 = int(OUT_H*0.61)
        w = int(OUT_W*0.80)
        overlay = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        d = ImageDraw.Draw(overlay)
        d.rounded_rectangle((x0,y0,x0+w,y0+(210 if not QUICK_MODE else 108)), radius=25 if not QUICK_MODE else 12,
                            fill=(3,7,15,175), outline=(90,190,220,70), width=1)
        img.alpha_composite(overlay)
        draw_text(img, "WHAT ACTUALLY FLIPS?", (x0+22,y0+(20 if not QUICK_MODE else 10)), size=24 if not QUICK_MODE else 12,
                  fill=(110,230,245,235), bold=True, stroke=1)
        lines = [
            "✓ global magnetic polarity",
            "✓ polar coronal-hole orientation",
            "✗ physical rotation axis",
            "✗ geographic poles",
        ]
        yy = y0+(66 if not QUICK_MODE else 32)
        for line in lines:
            col = (245,248,252,235) if "✓" in line else (255,175,105,230)
            draw_text(img, line, (x0+24,yy), size=22 if not QUICK_MODE else 11, fill=col, bold=True, stroke=1)
            yy += 34 if not QUICK_MODE else 17

    def draw_source_hud(self, img: Image.Image):
        source_label = "SOURCE // WSO LIVE" if self.source == "live_wso" else "PREVIEW SOURCE // REVERSAL MODEL"
        col = (110,230,245,230) if self.source == "live_wso" else (255,190,95,230)
        draw_text(img, source_label, (OUT_W-(48 if not QUICK_MODE else 24), 72 if not QUICK_MODE else 36), size=18 if not QUICK_MODE else 9,
                  fill=col, bold=True, anchor="ra", stroke=1)
        draw_text(img, f"RECORD // {self.summary['start_date']} to {self.summary['end_date']}",
                  (OUT_W-(48 if not QUICK_MODE else 24), 104 if not QUICK_MODE else 52), size=16 if not QUICK_MODE else 8,
                  fill=(165,205,222,205), anchor="ra", stroke=1)

    def draw_titles(self, img: Image.Image, t: float, shot_name: str):
        alpha = int(255*smoothstep((t-0.2)/0.8)*(1-smoothstep((t-(6.7 if not QUICK_MODE else 1.7))/0.7)))
        if alpha>4:
            draw_text(img, "THE SUN'S MAGNETIC FIELD", (56 if not QUICK_MODE else 28, 88 if not QUICK_MODE else 43), size=42 if not QUICK_MODE else 19,
                      fill=(245,248,252,alpha), bold=True)
            draw_text(img, "KEEPS FLIPPING", (56 if not QUICK_MODE else 28, 136 if not QUICK_MODE else 67), size=42 if not QUICK_MODE else 19,
                      fill=(245,248,252,alpha), bold=True)
            draw_text(img, CONFIG["subtitle"], (58 if not QUICK_MODE else 30, 188 if not QUICK_MODE else 94), size=22 if not QUICK_MODE else 10,
                      fill=(110,230,245,min(alpha,230)), bold=True)
        labels = {
            "intro":"GLOBAL POLARITY // NOT A PHYSICAL POLE SWAP",
            "wso_data":"POLAR FIELD DATA // NORTH AND SOUTH REVERSE SEPARATELY",
            "mechanism":"FLUX TRANSPORT // BUILDING THE NEXT POLARITY",
            "hale_cycle":"11 YEARS TO FLIP // 22 YEARS TO RETURN",
            "outro":"A STELLAR MAGNETIC RESET",
        }
        if t>(5.2 if not QUICK_MODE else 1.4):
            draw_text(img, labels[shot_name], (56 if not QUICK_MODE else 28, 62 if not QUICK_MODE else 31), size=19 if not QUICK_MODE else 9,
                      fill=(150,210,230,205), bold=True, stroke=1)

    def draw_caption(self, img: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        y0 = OUT_H-(244 if not QUICK_MODE else 124)
        panel = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        d = ImageDraw.Draw(panel)
        d.rounded_rectangle((44 if not QUICK_MODE else 22,y0,OUT_W-(44 if not QUICK_MODE else 22),y0+(124 if not QUICK_MODE else 66)),
                            radius=24 if not QUICK_MODE else 12, fill=(2,6,14,170), outline=(70,180,220,65), width=1)
        img.alpha_composite(panel)
        draw_wrapped_text(img,text,(68 if not QUICK_MODE else 34,y0+(28 if not QUICK_MODE else 14)),OUT_W-(136 if not QUICK_MODE else 68),
                          size=30 if not QUICK_MODE else 14,fill=(245,249,253,245))

    def draw_hud_noise(self, img: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        d = ImageDraw.Draw(overlay)
        for h in self.hud:
            pulse = 0.5+0.5*math.sin(t*1.9+h["phase"])
            if pulse<0.73:
                continue
            y=(h["y"]+t*9)%OUT_H
            d.line((h["x"],y,h["x"]+h["length"],y),fill=(90,210,240,int(h["a"]*pulse)),width=1)
        offset=int((t*39)%7)
        for y in range(offset,OUT_H,7):
            d.line((0,y,OUT_W,y),fill=(120,200,240,11),width=1)
        scan_y=int((t*164)%(OUT_H+220))-110
        d.rectangle((0,scan_y,OUT_W,scan_y+(48 if not QUICK_MODE else 24)),fill=(80,210,240,8))
        img.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        img = self.background(t)
        name = shot["name"]

        if name == "intro":
            frac = smoothstep(t/(7 if not QUICK_MODE else 1.8))
            polarity = math.cos(math.pi*frac)
            activity = math.sin(math.pi*frac)
            center=(OUT_W*0.5,OUT_H*0.40)
            radius=190*OUT_W/1080
            self.draw_sun(img,center,radius,polarity,activity,t)
            self.draw_bar_magnet_labels(img,center,radius,polarity)
            draw_text(img,"~11 YEARS",(OUT_W//2,int(OUT_H*0.67)),size=42 if not QUICK_MODE else 20,
                      fill=(255,190,90,240),bold=True,anchor="ma")
            draw_text(img,"between global polarity reversals",(OUT_W//2,int(OUT_H*0.72)),size=22 if not QUICK_MODE else 11,
                      fill=(235,242,248,225),anchor="ma",stroke=1)
        elif name == "wso_data":
            self.draw_data_chart(img,t)
        elif name == "mechanism":
            self.draw_mechanism(img,t)
        elif name == "hale_cycle":
            self.draw_hale_cycle(img,t)
        elif name == "outro":
            self.draw_outro(img,t)

        self.draw_source_hud(img)
        self.draw_titles(img,t,name)
        self.draw_caption(img,t)
        self.draw_hud_noise(img,t)

        arr=np.array(img.convert("RGB"))
        arr=apply_grade(arr)
        arr=np.clip(arr.astype(np.float32)*VIGNETTE[...,None],0,255).astype(np.uint8)
        fade_in=smoothstep(t/0.9)
        fade_out=1-smoothstep((t-(CONFIG["duration_s"]-1.1))/1.0)
        return np.clip(arr.astype(np.float32)*fade_in*fade_out,0,255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def render_video(scene: MagneticFlipScene):
    srt_path=OUTPUT_ROOT/f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS,srt_path)
    raw=OUTPUT_ROOT/f"{CONFIG['output_basename']}_raw.mp4"
    final=OUTPUT_ROOT/f"{CONFIG['output_basename']}_final.mp4"
    n=int(round(CONFIG["duration_s"]*CONFIG["fps"]))
    times=np.arange(n)/CONFIG["fps"]
    print(f"Rendering {n:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(raw,fps=CONFIG["fps"],codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=None) as writer:
        for t in tqdm(times,desc="Rendering magnetic-field short"):
            writer.append_data(scene.render_frame(float(t)))
    shutil.copyfile(raw,final)
    print("Final video:",final.resolve())
    return final


def main():
    print("Loading solar polar-field data ...")
    df,source,error_note=load_polar_data()
    summary=summarize_data(df,source)
    csv_path,json_path=save_data_products(df,summary,error_note)
    create_scientific_plots(df)
    print("Source:",source)
    if error_note:
        print("Live fetch note:",error_note)
    print("Data:",csv_path.resolve())
    print("Summary:",json_path.resolve())

    scene=MagneticFlipScene(df,summary,source)
    preview_times=[1.0,min(11.0,CONFIG["duration_s"]*.22),min(22.0,CONFIG["duration_s"]*.40),min(34.0,CONFIG["duration_s"]*.60),min(46.0,CONFIG["duration_s"]*.80),CONFIG["duration_s"]-1]
    for pt in tqdm(preview_times,desc="Preview frames"):
        Image.fromarray(scene.render_frame(float(pt))).save(PREVIEW_DIR/f"preview_{int(pt):02d}s.png")
    render_video(scene)
    print("Output directory:",OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
