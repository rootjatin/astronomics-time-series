
from __future__ import annotations

"""
A Year of Sunspots in 60 Seconds — cinematic vertical YouTube Short renderer.

What it does
------------
This script builds a 1080x1920 (or quick-preview) video that compresses one
full year of daily sunspot activity into a vertical astronomy short. It is
built around the official daily total sunspot number produced by WDC-SILSO.

Primary intended data source
----------------------------
WDC-SILSO / SIDC / Royal Observatory of Belgium:
    https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv
final result : https://youtube.com/shorts/LC2jbdP5Jbs

Dataset notes from the official SILSO documentation:
- Daily total sunspot number is available from 1818 to now.
- The daily sunspot number is defined as R = Ns + 10 * Ng, where Ns is the
  number of spots and Ng the number of groups across the whole solar disk.
- Missing values are marked with -1.
- The file uses semicolon delimiters.

If live download fails in the local environment, the script falls back to a
procedural demo year so that layout and rendering can still be tested. When
run in a normal internet-connected environment, it will fetch the official
SILSO data and use the latest 365 valid daily values by default.

Usage
-----
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm requests
    python a_year_of_sunspots_in_60_seconds_short.py

Quick preview
-------------
    SUNSPOTS_SHORT_QUICK=1 python a_year_of_sunspots_in_60_seconds_short.py
"""

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import requests
except Exception:
    requests = None

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("SUNSPOTS_SHORT_QUICK", "0") == "1"
OUTPUT_ROOT = Path("sunspots_year_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in [OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "a_year_of_sunspots_in_60_seconds",
    "sunspots_url": "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv",
    "days_in_window": 365,
    "title": "A YEAR OF SUNSPOTS IN 60 SECONDS",
    "subtitle": "WDC-SILSO daily sunspot number",
    "stars": 260,
    "hud_lines": 42,
    "contrast_boost": 1.08,
    "color_boost": 1.06,
    "vignette_strength": 0.25,
}

OUT_W = CONFIG["video_width"]
OUT_H = CONFIG["video_height"]
OUT_SIZE = (OUT_W, OUT_H)

CAPTIONS = [
    (0.5, 6.0, "This is one full year of daily sunspot activity, compressed into a single minute."),
    (6.1, 14.0, "The official daily sunspot number comes from WDC-SILSO at the Royal Observatory of Belgium."),
    (14.1, 22.0, "Each day on screen changes the visible spot pattern and updates the measured sunspot number."),
    (22.1, 31.0, "Sunspot number is computed as spots plus ten times the number of spot groups."),
    (31.1, 40.0, "As the Sun rotates and active regions grow or fade, the count rises and falls."),
    (40.1, 50.0, "The chart below traces that daily rhythm across the entire year."),
    (50.1, 57.2, "So in one minute, you can watch a year of solar activity pulse across the face of the Sun."),
]

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.0 if not QUICK_MODE else 2.0},
    {"name": "daily_sun", "start": 7.0 if not QUICK_MODE else 2.0, "end": 25.0 if not QUICK_MODE else 5.0},
    {"name": "stats", "start": 25.0 if not QUICK_MODE else 5.0, "end": 39.0 if not QUICK_MODE else 8.0},
    {"name": "timeline", "start": 39.0 if not QUICK_MODE else 8.0, "end": 51.0 if not QUICK_MODE else 10.0},
    {"name": "outro", "start": 51.0 if not QUICK_MODE else 10.0, "end": CONFIG["duration_s"]},
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(img: Image.Image, text: str, xy: Tuple[int, int], size: int = 28,
              fill=(255, 255, 255, 255), bold: bool = False,
              anchor: str = "la", stroke: int = 2):
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
    lines = []
    cur = ""
    for w in words:
        trial = w if not cur else cur + " " + w
        bbox = draw.textbbox((0, 0), trial, font=font, stroke_width=2)
        if bbox[2] - bbox[0] <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 220))
        bb = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += (bb[3] - bb[1]) + line_spacing


def date_to_str(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


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
    for idx, (start, end, text) in enumerate(captions, start=1):
        lines.append(str(idx))
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


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


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    rr = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * rr**1.8, 0.0, 1.0).astype(np.float32)


def apply_grade(arr: np.ndarray) -> np.ndarray:
    img = Image.fromarray(arr)
    img = ImageEnhance.Contrast(img).enhance(CONFIG["contrast_boost"])
    img = ImageEnhance.Color(img).enhance(CONFIG["color_boost"])
    return np.array(img)


VIGNETTE = make_vignette(OUT_W, OUT_H, CONFIG["vignette_strength"])

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_live_silso() -> Tuple[pd.DataFrame, str]:
    if requests is None:
        raise RuntimeError("requests not available")
    resp = requests.get(CONFIG["sunspots_url"], timeout=40)
    resp.raise_for_status()
    text = resp.text.strip().splitlines()
    rows = []
    for line in text:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 8:
            continue
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        dec_year = float(parts[3])
        ssn = float(parts[4])
        std = float(parts[5]) if parts[5] else np.nan
        n_obs = int(float(parts[6])) if parts[6] else 0
        definitive = int(parts[7]) if parts[7] != "" else 0
        rows.append({
            "date": pd.Timestamp(year=y, month=m, day=d),
            "decimal_year": dec_year,
            "sunspot_number": ssn,
            "std_dev": std,
            "observations": n_obs,
            "definitive": definitive,
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df = df[df["sunspot_number"] >= 0].copy()
    return df, "live_silso"


def load_fallback_demo() -> Tuple[pd.DataFrame, str]:
    # Deterministic demo curve so local preview can render without internet.
    # This is clearly marked as fallback in output metadata.
    end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(days=CONFIG["days_in_window"] - 1)
    dates = pd.date_range(start_date, end_date, freq="D")
    values = []
    for i, d in enumerate(dates):
        seasonal = 95 + 48 * math.sin(2 * math.pi * i / 118.0)
        short = 22 * math.sin(2 * math.pi * i / 27.0 + 0.8)
        burst = 18 * max(0.0, math.sin(2 * math.pi * i / 53.0 - 0.5))
        val = max(0.0, seasonal + short + burst)
        values.append(round(val, 1))
    df = pd.DataFrame({
        "date": dates,
        "decimal_year": [d.year + (d.dayofyear - 0.5) / (366 if d.is_leap_year else 365) for d in dates],
        "sunspot_number": values,
        "std_dev": np.nan,
        "observations": 0,
        "definitive": 0,
    })
    return df, "fallback_demo_curve"


def load_one_year_window() -> Tuple[pd.DataFrame, Dict, Optional[str]]:
    error_note = None
    try:
        all_df, source = load_live_silso()
    except Exception as exc:
        error_note = str(exc)
        all_df, source = load_fallback_demo()

    all_df = all_df.sort_values("date").reset_index(drop=True)
    valid_df = all_df[all_df["sunspot_number"] >= 0].copy()
    window_df = valid_df.tail(CONFIG["days_in_window"]).copy().reset_index(drop=True)
    window_df["day_index"] = np.arange(len(window_df))
    window_df["date_str"] = window_df["date"].dt.strftime("%Y-%m-%d")

    stats = {
        "source": source,
        "start_date": date_to_str(window_df.iloc[0]["date"]),
        "end_date": date_to_str(window_df.iloc[-1]["date"]),
        "num_days": int(len(window_df)),
        "mean_sunspot_number": round(float(window_df["sunspot_number"].mean()), 2),
        "median_sunspot_number": round(float(window_df["sunspot_number"].median()), 2),
        "max_day": {
            "date": date_to_str(window_df.iloc[window_df["sunspot_number"].idxmax()]["date"]),
            "sunspot_number": float(window_df["sunspot_number"].max()),
        },
        "min_day": {
            "date": date_to_str(window_df.iloc[window_df["sunspot_number"].idxmin()]["date"]),
            "sunspot_number": float(window_df["sunspot_number"].min()),
        },
        "spotless_days": int((window_df["sunspot_number"] == 0).sum()),
        "days_ge_100": int((window_df["sunspot_number"] >= 100).sum()),
        "days_ge_150": int((window_df["sunspot_number"] >= 150).sum()),
    }

    monthly = (
        window_df.assign(month_label=window_df["date"].dt.strftime("%Y-%m"))
        .groupby("month_label", as_index=False)["sunspot_number"]
        .mean()
    )
    stats["monthly_means"] = monthly.to_dict(orient="records")
    return window_df, stats, error_note


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------

class SunspotsYearScene:
    def __init__(self, df: pd.DataFrame, stats: Dict, source: str):
        self.df = df.copy()
        self.stats = stats
        self.source = source
        self.max_ssn = max(1.0, float(self.df["sunspot_number"].max()))
        self.monthly = (
            self.df.assign(month_label=self.df["date"].dt.strftime("%b"))
            .groupby("month_label", sort=False)["sunspot_number"]
            .mean()
            .reset_index()
        )
        self.stars = self._make_stars(CONFIG["stars"], seed=22)
        self.hud = self._make_hud(CONFIG["hud_lines"], seed=71)

    @staticmethod
    def _make_stars(n: int, seed: int):
        rng = np.random.default_rng(seed)
        items = []
        for _ in range(n):
            items.append({
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.5, 2.1)),
                "a": int(rng.integers(18, 100)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            })
        return items

    @staticmethod
    def _make_hud(n: int, seed: int):
        rng = np.random.default_rng(seed)
        items = []
        for _ in range(n):
            items.append({
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "length": float(rng.uniform(18, 92)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "alpha": int(rng.integers(8, 42)),
            })
        return items

    def background(self, t: float) -> Image.Image:
        img = Image.new("RGBA", OUT_SIZE, (6, 10, 18, 255))
        d = ImageDraw.Draw(img)
        for s in self.stars:
            alpha = int(s["a"] * (0.75 + 0.25 * math.sin(1.7 * t + s["phase"])))
            d.ellipse((s["x"]-s["r"], s["y"]-s["r"], s["x"]+s["r"], s["y"]+s["r"]), fill=(220, 230, 255, alpha))

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        clouds = [
            (OUT_W * 0.18, OUT_H * 0.24, (100, 58, 22)),
            (OUT_W * 0.74, OUT_H * 0.32, (140, 68, 16)),
            (OUT_W * 0.46, OUT_H * 0.78, (70, 42, 15)),
        ]
        for cx, cy, col in clouds:
            for rr, aa in [(420 * OUT_W/1080, 18), (290 * OUT_W/1080, 28), (180 * OUT_W/1080, 36)]:
                gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=(col[0], col[1], col[2], aa))
        glow = glow.filter(ImageFilter.GaussianBlur(60 if not QUICK_MODE else 30))
        img.alpha_composite(glow)
        return img

    def day_index_from_time(self, t: float) -> int:
        # Spread most of the year across the middle shots.
        if t < SHOT_PLAN[1]["start"]:
            frac = 0.0
        elif t < SHOT_PLAN[3]["end"]:
            span = SHOT_PLAN[3]["end"] - SHOT_PLAN[1]["start"]
            frac = clamp((t - SHOT_PLAN[1]["start"]) / max(span, 1e-6))
        else:
            frac = 1.0
        idx = int(round(frac * (len(self.df) - 1)))
        return max(0, min(len(self.df) - 1, idx))

    def current_row(self, t: float) -> pd.Series:
        return self.df.iloc[self.day_index_from_time(t)]

    def make_spot_layout(self, day_str: str, ssn: float):
        h = hashlib.sha256(day_str.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "big")
        rng = np.random.default_rng(seed)
        n_groups = int(np.clip(round(ssn / 24.0), 0, 12))
        clusters = []
        for _ in range(n_groups):
            angle = rng.uniform(0, 2 * math.pi)
            radius = math.sqrt(rng.uniform(0.02, 0.72))
            x = math.cos(angle) * radius
            y = math.sin(angle) * radius * 0.83
            size = rng.uniform(0.02, 0.06)
            n_spots = int(rng.integers(1, 5))
            clusters.append((x, y, size, n_spots, rng.integers(0, 1_000_000)))
        return clusters

    def draw_sun_disc(self, img: Image.Image, center: Tuple[float, float], radius: float, row: pd.Series):
        cx, cy = center
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for scale, aa in [(1.65, 20), (1.38, 40), (1.16, 80)]:
            r = radius * scale
            gd.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(255, 154, 40, aa))
        glow = glow.filter(ImageFilter.GaussianBlur(22 if not QUICK_MODE else 11))
        img.alpha_composite(glow)

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        # Radial color bands
        for i in range(24, 0, -1):
            frac = i / 24.0
            r = radius * frac
            col = (
                int(255),
                int(140 + 70 * frac),
                int(28 + 60 * frac),
                255,
            )
            d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=col)
        d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=(255, 250, 215, 150), width=max(1, int(radius * 0.01)))

        # Granulation-ish soft texture
        rng = np.random.default_rng(1234)
        for _ in range(120 if not QUICK_MODE else 60):
            ang = rng.uniform(0, 2 * math.pi)
            rr = radius * math.sqrt(rng.uniform(0, 0.95))
            x = cx + math.cos(ang) * rr
            y = cy + math.sin(ang) * rr
            r = rng.uniform(radius * 0.01, radius * 0.03)
            alpha = int(rng.integers(8, 18))
            d.ellipse((x-r, y-r, x+r, y+r), fill=(255, 230, 180, alpha))

        # Sunspots based on daily number
        for gx, gy, size, n_spots, seed in self.make_spot_layout(row["date_str"], float(row["sunspot_number"])):
            sx = cx + gx * radius * 0.82
            sy = cy + gy * radius * 0.82
            d.ellipse((sx-size*radius*2.8, sy-size*radius*2.8, sx+size*radius*2.8, sy+size*radius*2.8), fill=(80, 42, 16, 100))
            rng = np.random.default_rng(int(seed))
            for _ in range(n_spots):
                ox = rng.uniform(-1.0, 1.0) * size * radius * 1.4
                oy = rng.uniform(-1.0, 1.0) * size * radius * 1.0
                sr = rng.uniform(size * radius * 0.35, size * radius * 0.8)
                d.ellipse((sx+ox-sr*1.4, sy+oy-sr*1.4, sx+ox+sr*1.4, sy+oy+sr*1.4), fill=(96, 58, 22, 120))
                d.ellipse((sx+ox-sr, sy+oy-sr, sx+ox+sr, sy+oy+sr), fill=(18, 16, 17, 230))

        img.alpha_composite(layer)

    def draw_title_block(self, img: Image.Image, t: float):
        alpha = int(255 * smoothstep((t - 0.1) / 0.7) * (1.0 - smoothstep((t - (6.0 if not QUICK_MODE else 1.7)) / 0.7)))
        if alpha <= 5:
            return
        draw_text(img, CONFIG["title"], (56 if not QUICK_MODE else 28, 92 if not QUICK_MODE else 46), size=46 if not QUICK_MODE else 21,
                  fill=(245, 248, 252, alpha), bold=True)
        draw_text(img, CONFIG["subtitle"], (58 if not QUICK_MODE else 30, 156 if not QUICK_MODE else 78), size=24 if not QUICK_MODE else 11,
                  fill=(255, 185, 92, min(alpha, 230)), bold=True)

    def draw_header_hud(self, img: Image.Image, t: float, source_text: str):
        draw_text(img, source_text, (OUT_W - (48 if not QUICK_MODE else 24), 72 if not QUICK_MODE else 36), size=18 if not QUICK_MODE else 8,
                  fill=(110, 225, 245, 220), bold=True, anchor="ra", stroke=1)
        draw_text(img, f"WINDOW // {self.stats['start_date']} to {self.stats['end_date']}",
                  (OUT_W - (48 if not QUICK_MODE else 24), 104 if not QUICK_MODE else 52), size=16 if not QUICK_MODE else 8,
                  fill=(165, 205, 220, 210), anchor="ra", stroke=1)
        draw_text(img, f"DAYS // {self.stats['num_days']}", (OUT_W - (48 if not QUICK_MODE else 24), 132 if not QUICK_MODE else 66),
                  size=16 if not QUICK_MODE else 8, fill=(165, 205, 220, 200), anchor="ra", stroke=1)

    def draw_caption_panel(self, img: Image.Image, t: float):
        cap = caption_at(t)
        if not cap:
            return
        y0 = OUT_H - (244 if not QUICK_MODE else 124)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rounded_rectangle((44 if not QUICK_MODE else 22, y0, OUT_W - (44 if not QUICK_MODE else 22), y0 + (124 if not QUICK_MODE else 66)),
                            radius=24 if not QUICK_MODE else 12, fill=(4, 8, 15, 168), outline=(255, 176, 74, 60), width=1)
        img.alpha_composite(overlay)
        draw_wrapped_text(img, cap, (66 if not QUICK_MODE else 33, y0 + (28 if not QUICK_MODE else 14)),
                          max_width=OUT_W - (132 if not QUICK_MODE else 66), size=30 if not QUICK_MODE else 14,
                          fill=(245, 248, 252, 245))

    def draw_hud_scan(self, img: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for item in self.hud:
            pulse = 0.5 + 0.5 * math.sin(2.1 * t + item["phase"])
            if pulse < 0.72:
                continue
            x = item["x"]
            y = (item["y"] + 9.0 * t) % OUT_H
            d.line((x, y, x + item["length"], y), fill=(100, 210, 240, int(item["alpha"] * pulse)), width=1)
        offset = int((t * 41) % 7)
        for y in range(offset, OUT_H, 7):
            d.line((0, y, OUT_W, y), fill=(120, 190, 235, 10), width=1)
        scan_y = int((t * 170) % (OUT_H + 220)) - 110
        d.rectangle((0, scan_y, OUT_W, scan_y + (48 if not QUICK_MODE else 24)), fill=(90, 205, 235, 7))
        img.alpha_composite(overlay)

    def draw_day_card(self, img: Image.Image, row: pd.Series):
        x0 = 56 if not QUICK_MODE else 28
        y0 = int(OUT_H * 0.67)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(panel)
        w = int(OUT_W * 0.50)
        h = 154 if not QUICK_MODE else 78
        d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=24 if not QUICK_MODE else 12,
                            fill=(4, 8, 14, 170), outline=(255, 165, 66, 70), width=1)
        img.alpha_composite(panel)
        draw_text(img, "CURRENT DAY", (x0 + 18, y0 + (18 if not QUICK_MODE else 9)), size=20 if not QUICK_MODE else 10,
                  fill=(255, 185, 92, 235), bold=True, stroke=1)
        draw_text(img, row["date"].strftime("%b %d, %Y"), (x0 + 18, y0 + (54 if not QUICK_MODE else 27)), size=30 if not QUICK_MODE else 14,
                  fill=(245, 248, 252, 240), bold=True, stroke=1)
        draw_text(img, f"Sunspot Number // {row['sunspot_number']:.1f}", (x0 + 18, y0 + (96 if not QUICK_MODE else 48)), size=24 if not QUICK_MODE else 12,
                  fill=(110, 225, 245, 232), bold=True, stroke=1)
        draw_text(img, f"Index Day // {int(row['day_index']) + 1} / {len(self.df)}", (x0 + 18, y0 + (128 if not QUICK_MODE else 64)), size=18 if not QUICK_MODE else 9,
                  fill=(165, 205, 220, 220), stroke=1)

    def draw_line_chart(self, img: Image.Image, current_idx: int, top_left: Tuple[int, int], size: Tuple[int, int]):
        x0, y0 = top_left
        w, h = size
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=24 if not QUICK_MODE else 12,
                            fill=(4, 8, 14, 150), outline=(80, 140, 160, 60), width=1)
        # grid
        for frac in np.linspace(0, 1, 5):
            yy = y0 + int(h * frac)
            d.line((x0 + 20, yy, x0 + w - 20, yy), fill=(110, 170, 190, 35), width=1)
        vals = self.df["sunspot_number"].to_numpy()
        pts = []
        for i, val in enumerate(vals):
            xx = x0 + 20 + (i / max(len(vals) - 1, 1)) * (w - 40)
            yy = y0 + h - 20 - (val / self.max_ssn) * (h - 40)
            pts.append((xx, yy))
        if len(pts) > 1:
            d.line(pts, fill=(255, 182, 74, 220), width=3 if not QUICK_MODE else 2)
        for i in range(0, len(pts), max(1, len(pts) // 28)):
            xx, yy = pts[i]
            d.ellipse((xx-2, yy-2, xx+2, yy+2), fill=(255, 216, 160, 160))
        cx, cy = pts[current_idx]
        d.line((cx, y0 + 16, cx, y0 + h - 16), fill=(100, 225, 245, 120), width=2)
        d.ellipse((cx-7, cy-7, cx+7, cy+7), fill=(95, 225, 245, 255), outline=(255, 255, 255, 160))
        img.alpha_composite(overlay)
        draw_text(img, "DAILY SUNSPOT NUMBER", (x0 + 18, y0 + 18 if not QUICK_MODE else y0 + 9), size=18 if not QUICK_MODE else 9,
                  fill=(110, 225, 245, 230), bold=True, stroke=1)
        draw_text(img, f"0", (x0 + 10, y0 + h - 20), size=14 if not QUICK_MODE else 7, fill=(160, 205, 220, 190), anchor="rm", stroke=1)
        draw_text(img, f"{int(round(self.max_ssn))}", (x0 + 10, y0 + 20), size=14 if not QUICK_MODE else 7, fill=(160, 205, 220, 190), anchor="rm", stroke=1)

    def draw_stats_cards(self, img: Image.Image):
        card_w = int(OUT_W * 0.38)
        card_h = 108 if not QUICK_MODE else 56
        x_left = int(OUT_W * 0.08)
        x_right = int(OUT_W * 0.54)
        y_top = int(OUT_H * 0.29)
        y_gap = 28 if not QUICK_MODE else 14
        cards = [
            (x_left, y_top, "AVERAGE", f"{self.stats['mean_sunspot_number']:.1f}"),
            (x_right, y_top, "MEDIAN", f"{self.stats['median_sunspot_number']:.1f}"),
            (x_left, y_top + card_h + y_gap, "MAX DAY", f"{self.stats['max_day']['sunspot_number']:.1f}"),
            (x_right, y_top + card_h + y_gap, "SPOTLESS DAYS", f"{self.stats['spotless_days']}"),
            (x_left, y_top + 2*(card_h + y_gap), "DAYS ≥ 100", f"{self.stats['days_ge_100']}"),
            (x_right, y_top + 2*(card_h + y_gap), "DAYS ≥ 150", f"{self.stats['days_ge_150']}"),
        ]
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for x, y, title, value in cards:
            d.rounded_rectangle((x, y, x + card_w, y + card_h), radius=22 if not QUICK_MODE else 11,
                                fill=(4, 8, 14, 168), outline=(255, 176, 74, 68), width=1)
        img.alpha_composite(overlay)
        for x, y, title, value in cards:
            draw_text(img, title, (x + 18, y + (18 if not QUICK_MODE else 8)), size=20 if not QUICK_MODE else 9,
                      fill=(255, 185, 92, 235), bold=True, stroke=1)
            draw_text(img, value, (x + 18, y + (62 if not QUICK_MODE else 30)), size=34 if not QUICK_MODE else 16,
                      fill=(245, 248, 252, 240), bold=True, stroke=1)
        draw_text(img, f"Peak day // {self.stats['max_day']['date']}", (x_left, int(OUT_H * 0.77)), size=20 if not QUICK_MODE else 10,
                  fill=(110, 225, 245, 230), bold=True, stroke=1)
        draw_text(img, f"Quietest day // {self.stats['min_day']['date']} ({self.stats['min_day']['sunspot_number']:.1f})",
                  (x_left, int(OUT_H * 0.81)), size=20 if not QUICK_MODE else 10, fill=(160, 205, 220, 220), stroke=1)

    def draw_monthly_bars(self, img: Image.Image):
        x0 = int(OUT_W * 0.08)
        y0 = int(OUT_H * 0.31)
        w = int(OUT_W * 0.84)
        h = int(OUT_H * 0.44)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=24 if not QUICK_MODE else 12,
                            fill=(4, 8, 14, 160), outline=(90, 150, 170, 60), width=1)
        img.alpha_composite(overlay)
        draw_text(img, "MONTHLY MEAN SUNSPOT NUMBER", (x0 + 18, y0 + 18 if not QUICK_MODE else y0 + 9), size=20 if not QUICK_MODE else 10,
                  fill=(110, 225, 245, 230), bold=True, stroke=1)
        maxv = max(1.0, float(self.monthly["sunspot_number"].max()))
        n = len(self.monthly)
        bar_w = (w - 50) / max(n, 1)
        base_y = y0 + h - 38
        for i, row in self.monthly.reset_index(drop=True).iterrows():
            val = float(row["sunspot_number"])
            bh = (val / maxv) * (h - 86)
            left = x0 + 28 + i * bar_w + 2
            right = left + max(bar_w - 8, 2)
            top = base_y - bh
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((left, top, right, base_y), radius=8 if not QUICK_MODE else 4,
                                fill=(255, 176, 74, 210))
            draw_text(img, str(row["month_label"]), (int((left+right)/2), base_y + (18 if not QUICK_MODE else 10)), size=14 if not QUICK_MODE else 7,
                      fill=(170, 208, 222, 200), anchor="ma", stroke=1)
        draw_text(img, f"Source days above 100 // {self.stats['days_ge_100']}", (x0, base_y + (54 if not QUICK_MODE else 26)), size=18 if not QUICK_MODE else 9,
                  fill=(245, 248, 252, 225), stroke=1)

    def draw_outro_summary(self, img: Image.Image):
        x0 = int(OUT_W * 0.08)
        y0 = int(OUT_H * 0.28)
        w = int(OUT_W * 0.84)
        h = int(OUT_H * 0.44)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=26 if not QUICK_MODE else 13,
                            fill=(4, 8, 14, 172), outline=(255, 176, 74, 70), width=1)
        img.alpha_composite(overlay)
        draw_text(img, "YEAR SUMMARY", (x0 + 20, y0 + (20 if not QUICK_MODE else 10)), size=28 if not QUICK_MODE else 13,
                  fill=(255, 185, 92, 235), bold=True, stroke=1)
        items = [
            f"Window: {self.stats['start_date']} to {self.stats['end_date']}",
            f"Average daily sunspot number: {self.stats['mean_sunspot_number']:.1f}",
            f"Peak day: {self.stats['max_day']['date']} ({self.stats['max_day']['sunspot_number']:.1f})",
            f"Spotless days: {self.stats['spotless_days']}",
            f"Days at or above 150: {self.stats['days_ge_150']}",
        ]
        y = y0 + (72 if not QUICK_MODE else 36)
        for item in items:
            draw_wrapped_text(img, item, (x0 + 22, y), w - 44, size=24 if not QUICK_MODE else 12,
                              fill=(245, 248, 252, 238), bold=True)
            y += 56 if not QUICK_MODE else 28

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        img = self.background(t)
        row = self.current_row(t)
        source_label = "SOURCE // WDC-SILSO LIVE" if self.source == "live_silso" else "PREVIEW SOURCE // DEMO FALLBACK"
        self.draw_header_hud(img, t, source_label)
        self.draw_title_block(img, t)

        if shot["name"] == "intro":
            self.draw_sun_disc(img, (OUT_W * 0.5, OUT_H * 0.38), 190 * OUT_W / 1080, row)
            draw_text(img, "365 DAYS OF SOLAR ACTIVITY", (OUT_W // 2, int(OUT_H * 0.62)), size=24 if not QUICK_MODE else 12,
                      fill=(110, 225, 245, 230), bold=True, anchor="ma", stroke=1)
        elif shot["name"] == "daily_sun":
            self.draw_sun_disc(img, (OUT_W * 0.5, OUT_H * 0.31), 185 * OUT_W / 1080, row)
            self.draw_day_card(img, row)
            self.draw_line_chart(img, int(row["day_index"]), (int(OUT_W * 0.08), int(OUT_H * 0.79)), (int(OUT_W * 0.84), int(OUT_H * 0.12)))
        elif shot["name"] == "stats":
            self.draw_sun_disc(img, (OUT_W * 0.5, OUT_H * 0.17), 98 * OUT_W / 1080, row)
            self.draw_stats_cards(img)
        elif shot["name"] == "timeline":
            self.draw_sun_disc(img, (OUT_W * 0.82, OUT_H * 0.18), 86 * OUT_W / 1080, row)
            self.draw_monthly_bars(img)
            self.draw_line_chart(img, int(row["day_index"]), (int(OUT_W * 0.08), int(OUT_H * 0.79)), (int(OUT_W * 0.84), int(OUT_H * 0.11)))
        elif shot["name"] == "outro":
            self.draw_sun_disc(img, (OUT_W * 0.5, OUT_H * 0.18), 84 * OUT_W / 1080, row)
            self.draw_outro_summary(img)

        self.draw_caption_panel(img, t)
        self.draw_hud_scan(img, t)

        arr = np.array(img.convert("RGB"))
        arr = apply_grade(arr)
        arr = np.clip(arr.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / 0.9)
        fade_out = 1.0 - smoothstep((t - (CONFIG["duration_s"] - 1.1)) / 1.0)
        arr = np.clip(arr.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)
        return arr


# -----------------------------------------------------------------------------
# Render helpers
# -----------------------------------------------------------------------------

def save_data_outputs(df: pd.DataFrame, stats: Dict, error_note: Optional[str]):
    csv_path = DATA_ROOT / "one_year_daily_sunspots.csv"
    df_out = df.copy()
    df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
    df_out.to_csv(csv_path, index=False)
    meta_path = DATA_ROOT / "one_year_daily_sunspots_summary.json"
    meta = {
        "stats": stats,
        "error_note": error_note,
        "live_source_url": CONFIG["sunspots_url"],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path, meta_path


def render_video(scene: SunspotsYearScene):
    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    raw_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    print("Subtitle sidecar written:", srt_path.resolve())
    times = np.arange(frame_count) / CONFIG["fps"]
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(raw_path, fps=CONFIG["fps"], codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None) as writer:
        for t in tqdm(times, desc="Rendering sunspots short"):
            writer.append_data(scene.render_frame(float(t)))
    shutil.copyfile(raw_path, final_path)
    print("Final video:", final_path.resolve())
    return final_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Loading one year of daily sunspot data ...")
    df, stats, error_note = load_one_year_window()
    print("Source:", stats["source"])
    if error_note:
        print("Live fetch note:", error_note)

    csv_path, meta_path = save_data_outputs(df, stats, error_note)
    print("Data written:", csv_path.resolve())
    print("Summary written:", meta_path.resolve())

    scene = SunspotsYearScene(df, stats, stats["source"])

    preview_times = [1.0, min(10.0, CONFIG["duration_s"] * 0.25), min(21.0, CONFIG["duration_s"] * 0.42), min(34.0, CONFIG["duration_s"] * 0.58), min(47.0, CONFIG["duration_s"] * 0.80), CONFIG["duration_s"] - 1.0]
    for pt in tqdm(preview_times, desc="Preview frames"):
        frame = scene.render_frame(float(pt))
        Image.fromarray(frame).save(PREVIEW_DIR / f"preview_{int(pt):02d}s.png")
    print("Preview frames written:", PREVIEW_DIR.resolve())

    render_video(scene)
    print("Output directory:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
