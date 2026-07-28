from __future__ import annotations

"""
Watch a Solar Storm Hit Earth — cinematic vertical YouTube Short renderer.

Concept
-------
This script creates a 1080x1920 science short (or a smaller quick preview)
about a real geomagnetic storm hitting Earth. The default story is the May 2024
G5 geomagnetic storm (the "Gannon storm"), which NASA described as the most
intense geomagnetic storm to hit Earth in two decades.

Primary official grounding
--------------------------
NASA article (May 16, 2024):
- During May 7-11, 2024, multiple strong solar flares and at least seven CMEs
  stormed toward Earth.
- The CMEs reached Earth starting May 10.
- The resulting geomagnetic storm reached G5, the highest level on NOAA's
  geomagnetic storm scale, and the strongest in about two decades.
- From May 3-9, NASA's SDO observed 82 notable flares.

NOAA / SWPC grounding:
- G5 is the highest geomagnetic storm category and corresponds to Kp=9.
- G5 impacts can include widespread grid problems, satellite issues, degraded
  navigation, HF radio impacts, and aurora as low as Florida / southern Texas.
- Solar flares are electromagnetic outbursts whose effects on the sunlit side
  of Earth are effectively immediate because the radiation travels at the speed
  of light.

Live-data behavior
------------------
The script tries to fetch NASA DONKI event feeds (flares, CMEs, geomagnetic
storm entries) for the May 2024 storm. If that fails, it falls back to an
embedded event timeline and a proxy Kp series anchored to NASA/NOAA's published
storm timing and severity. The fallback preview is suitable for validating the
visual story, but it is not a substitute for the live event feeds.

Usage
-----
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm requests
    python watch_a_solar_storm_hit_earth_short.py

Quick preview
-------------
    SOLAR_STORM_SHORT_QUICK=1 python watch_a_solar_storm_hit_earth_short.py
"""

import json
import math
import os
import shutil
from datetime import datetime, timedelta, timezone
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

QUICK_MODE = os.environ.get("SOLAR_STORM_SHORT_QUICK", "0") == "1"
OUTPUT_ROOT = Path("solar_storm_hit_earth_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in [OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "watch_a_solar_storm_hit_earth",
    "title": "WATCH A SOLAR STORM HIT EARTH",
    "subtitle": "May 2024 G5 geomagnetic storm",
    "stars": 260,
    "hud_lines": 44,
    "contrast_boost": 1.08,
    "color_boost": 1.06,
    "vignette_strength": 0.25,
    "nasa_donki_base": "https://api.nasa.gov/DONKI",
    "donki_api_key": os.environ.get("NASA_API_KEY", "DEMO_KEY"),
}

OUT_W = CONFIG["video_width"]
OUT_H = CONFIG["video_height"]
OUT_SIZE = (OUT_W, OUT_H)

CAPTIONS = [
    (0.5, 7.0, "This is a real solar storm sequence: eruptions blast off from the Sun, race through space, and slam into Earth's magnetic shield."),
    (7.1, 15.0, "NASA says that from May 7 through May 11, 2024, multiple strong flares and at least seven coronal mass ejections stormed toward Earth."),
    (15.1, 24.0, "Solar flare radiation reaches Earth almost immediately, but the giant plasma clouds take much longer to arrive."),
    (24.1, 34.0, "When the CME waves hit, they compress the magnetosphere and drive a geomagnetic storm."),
    (34.1, 44.0, "This one reached G5 — the highest NOAA storm level — the strongest geomagnetic storm to reach Earth in about two decades."),
    (44.1, 52.0, "Auroras spread to unusually low latitudes while navigation, radio, satellites, and power systems faced elevated risk."),
    (52.1, 57.2, "So a solar storm doesn't mean fire hitting the ground — it means space weather shaking Earth's magnetic environment."),
]

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 8.0 if not QUICK_MODE else 2.0},
    {"name": "launch", "start": 8.0 if not QUICK_MODE else 2.0, "end": 18.0 if not QUICK_MODE else 4.0},
    {"name": "travel", "start": 18.0 if not QUICK_MODE else 4.0, "end": 30.0 if not QUICK_MODE else 6.0},
    {"name": "impact", "start": 30.0 if not QUICK_MODE else 6.0, "end": 42.0 if not QUICK_MODE else 8.0},
    {"name": "effects", "start": 42.0 if not QUICK_MODE else 8.0, "end": 51.0 if not QUICK_MODE else 10.0},
    {"name": "outro", "start": 51.0 if not QUICK_MODE else 10.0, "end": CONFIG["duration_s"]},
]


# -----------------------------------------------------------------------------
# Fallback data grounded on official NASA/NOAA facts
# -----------------------------------------------------------------------------

FALLBACK_TIMELINE = [
    {"time_utc": "2024-05-07T00:00:00Z", "phase": "eruption", "label": "Storm sequence begins", "details": "Strong flares erupt from active regions", "g_scale": 0, "kp_proxy": 2},
    {"time_utc": "2024-05-07T18:00:00Z", "phase": "cme", "label": "First CME wave launched", "details": "Earth-directed plasma cloud leaves the Sun", "g_scale": 0, "kp_proxy": 2},
    {"time_utc": "2024-05-08T18:00:00Z", "phase": "cme", "label": "More CME waves follow", "details": "Successive eruptions pile up behind the first", "g_scale": 0, "kp_proxy": 2},
    {"time_utc": "2024-05-09T18:00:00Z", "phase": "travel", "label": "Interplanetary transit", "details": "Multiple CME fronts race toward Earth", "g_scale": 0, "kp_proxy": 3},
    {"time_utc": "2024-05-10T12:00:00Z", "phase": "impact", "label": "First impact at Earth", "details": "The first CME wave reaches the magnetosphere", "g_scale": 2, "kp_proxy": 6},
    {"time_utc": "2024-05-10T21:00:00Z", "phase": "storm", "label": "Storm intensifies fast", "details": "Geomagnetic disturbance climbs into severe levels", "g_scale": 4, "kp_proxy": 8},
    {"time_utc": "2024-05-11T03:00:00Z", "phase": "peak", "label": "G5 peak", "details": "The geomagnetic storm reaches the highest NOAA level", "g_scale": 5, "kp_proxy": 9},
    {"time_utc": "2024-05-11T12:00:00Z", "phase": "aurora", "label": "Aurora spreads globally", "details": "Bright auroras become visible at unusually low latitudes", "g_scale": 4, "kp_proxy": 8},
    {"time_utc": "2024-05-12T12:00:00Z", "phase": "storm", "label": "Additional CME wave", "details": "Another arriving disturbance prolongs the storm", "g_scale": 4, "kp_proxy": 8},
    {"time_utc": "2024-05-13T12:00:00Z", "phase": "recovery", "label": "Recovery phase", "details": "Conditions gradually ease but remain elevated", "g_scale": 2, "kp_proxy": 6},
]

FALLBACK_KP_SERIES = [
    ("2024-05-09T00:00:00Z", 2),
    ("2024-05-09T06:00:00Z", 2),
    ("2024-05-09T12:00:00Z", 3),
    ("2024-05-09T18:00:00Z", 3),
    ("2024-05-10T00:00:00Z", 4),
    ("2024-05-10T06:00:00Z", 5),
    ("2024-05-10T12:00:00Z", 6),
    ("2024-05-10T18:00:00Z", 7),
    ("2024-05-11T00:00:00Z", 8),
    ("2024-05-11T06:00:00Z", 9),
    ("2024-05-11T12:00:00Z", 8),
    ("2024-05-11T18:00:00Z", 8),
    ("2024-05-12T00:00:00Z", 7),
    ("2024-05-12T06:00:00Z", 8),
    ("2024-05-12T12:00:00Z", 7),
    ("2024-05-12T18:00:00Z", 6),
    ("2024-05-13T00:00:00Z", 5),
    ("2024-05-13T06:00:00Z", 4),
    ("2024-05-13T12:00:00Z", 3),
    ("2024-05-13T18:00:00Z", 3),
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


def ease_in_out_sine(t: float) -> float:
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


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
    d = ImageDraw.Draw(img)
    d.text(
        xy,
        text,
        font=get_font(size, bold=bold),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(fill[3] if len(fill) > 3 else 255, 220)),
    )


def draw_wrapped_text(img: Image.Image, text: str, xy: Tuple[int, int], max_width: int,
                      size: int = 28, fill=(255, 255, 255, 245), bold: bool = False,
                      line_spacing: int = 6):
    d = ImageDraw.Draw(img)
    font = get_font(size, bold=bold)
    words = text.split()
    lines: List[str] = []
    current = ""
    for w in words:
        test = w if not current else current + " " + w
        bb = d.textbbox((0, 0), test, font=font, stroke_width=2)
        if bb[2] - bb[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        d.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 220))
        bb = d.textbbox((x, y), line, font=font, stroke_width=2)
        y += (bb[3] - bb[1]) + line_spacing


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
        lines.append(str(i))
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


def parse_iso(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True)


VIGNETTE = make_vignette(OUT_W, OUT_H, CONFIG["vignette_strength"])


# -----------------------------------------------------------------------------
# Data acquisition
# -----------------------------------------------------------------------------

def fetch_json(url: str):
    if requests is None:
        raise RuntimeError("requests not available")
    resp = requests.get(url, timeout=40)
    resp.raise_for_status()
    return resp.json()


def load_live_donki() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    base = CONFIG["nasa_donki_base"]
    key = CONFIG["donki_api_key"]
    flr_url = f"{base}/FLR?startDate=2024-05-07&endDate=2024-05-14&api_key={key}"
    cme_url = f"{base}/CMEAnalysis?startDate=2024-05-07&endDate=2024-05-14&mostAccurateOnly=true&speed=0&halfAngle=0&catalog=ALL&api_key={key}"
    gst_url = f"{base}/GST?startDate=2024-05-10&endDate=2024-05-13&api_key={key}"
    flares = fetch_json(flr_url)
    cmes = fetch_json(cme_url)
    gst = fetch_json(gst_url)

    # Flares
    flare_rows = []
    for item in flares:
        peak_time = item.get("peakTime") or item.get("beginTime")
        class_type = item.get("classType", "")
        if not peak_time:
            continue
        flare_rows.append({
            "time_utc": peak_time,
            "phase": "flare",
            "label": f"{class_type} flare",
            "details": f"Source region {item.get('activeRegionNum', 'unknown')}",
            "g_scale": 0,
            "kp_proxy": 0,
            "class_type": class_type,
        })

    # CMEs
    cme_rows = []
    for item in cmes:
        start_time = item.get("startTime")
        if not start_time:
            continue
        speed = item.get("speed")
        note = item.get("note") or "Coronal mass ejection launched"
        cme_rows.append({
            "time_utc": start_time,
            "phase": "cme",
            "label": "CME launched",
            "details": f"speed={speed} km/s" if speed is not None else note,
            "g_scale": 0,
            "kp_proxy": 0,
        })

    # GST Kp entries
    kp_rows = []
    gst_rows = []
    for item in gst:
        start_time = item.get("startTime")
        if start_time:
            gst_rows.append({
                "time_utc": start_time,
                "phase": "storm",
                "label": "Geomagnetic storm begins",
                "details": "DONKI GST event",
                "g_scale": 0,
                "kp_proxy": 0,
            })
        for kp in item.get("allKpIndex", []) or []:
            observed_time = kp.get("observedTime") or kp.get("kpTime") or kp.get("timeTag")
            value = kp.get("kpIndex")
            if observed_time is None or value is None:
                continue
            try:
                value = float(value)
            except Exception:
                continue
            kp_rows.append({"time_utc": observed_time, "kp": value})

    timeline = pd.DataFrame(flare_rows + cme_rows + gst_rows)
    if timeline.empty:
        raise RuntimeError("Live DONKI returned no parseable events")
    timeline["time_ts"] = pd.to_datetime(timeline["time_utc"], utc=True)
    timeline = timeline.sort_values("time_ts").reset_index(drop=True)

    if kp_rows:
        kp_df = pd.DataFrame(kp_rows)
        kp_df["time_ts"] = pd.to_datetime(kp_df["time_utc"], utc=True)
        kp_df = kp_df.sort_values("time_ts").drop_duplicates(subset=["time_ts"]).reset_index(drop=True)
    else:
        kp_df = pd.DataFrame(columns=["time_utc", "time_ts", "kp"])

    x_class = [r for r in flare_rows if str(r.get("class_type", "")).upper().startswith("X")]
    summary = {
        "source": "live_nasa_donki",
        "event_name": "May 2024 geomagnetic storm",
        "storm_window": "2024-05-07 to 2024-05-13",
        "flares_count": int(len(flare_rows)),
        "x_class_flare_count": int(len(x_class)),
        "cme_count": int(len(cme_rows)),
        "max_kp": float(kp_df["kp"].max()) if not kp_df.empty else None,
        "g_level_peak": 5 if not kp_df.empty and kp_df["kp"].max() >= 9 else None,
        "nasa_official_context": {
            "notable_flares_may3_9": 82,
            "at_least_cmes_may7_11": 7,
            "storm_reached_g5": True,
        },
    }
    return timeline, kp_df, summary


def load_fallback_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    timeline = pd.DataFrame(FALLBACK_TIMELINE)
    timeline["time_ts"] = pd.to_datetime(timeline["time_utc"], utc=True)
    kp_df = pd.DataFrame(FALLBACK_KP_SERIES, columns=["time_utc", "kp"])
    kp_df["time_ts"] = pd.to_datetime(kp_df["time_utc"], utc=True)
    summary = {
        "source": "fallback_official_milestones",
        "event_name": "May 2024 G5 geomagnetic storm",
        "storm_window": "2024-05-07 to 2024-05-13",
        "nasa_official_context": {
            "notable_flares_may3_9": 82,
            "at_least_cmes_may7_11": 7,
            "storm_reached_g5": True,
            "strongest_storm_in_two_decades": True,
        },
        "noaa_official_context": {
            "g5_is_highest_level": True,
            "g5_corresponds_to_kp9": True,
        },
        "peak_kp_proxy": int(kp_df["kp"].max()),
        "peak_g_scale": 5,
        "note": "Proxy Kp series anchored to official storm timing and severity; not a replacement for archived measured Kp.",
    }
    return timeline, kp_df, summary


def load_event_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Optional[str]]:
    error_note = None
    try:
        timeline, kp_df, summary = load_live_donki()
    except Exception as exc:
        error_note = str(exc)
        timeline, kp_df, summary = load_fallback_data()
    return timeline, kp_df, summary, error_note


def save_data_products(timeline: pd.DataFrame, kp_df: pd.DataFrame, summary: Dict, error_note: Optional[str]):
    timeline_csv = DATA_ROOT / "solar_storm_may_2024_timeline.csv"
    kp_csv = DATA_ROOT / "solar_storm_may_2024_kp_series.csv"
    summary_json = DATA_ROOT / "solar_storm_may_2024_summary.json"

    timeline_out = timeline.copy()
    if "time_ts" in timeline_out.columns:
        timeline_out = timeline_out.drop(columns=["time_ts"])
    timeline_out.to_csv(timeline_csv, index=False)

    kp_out = kp_df.copy()
    if "time_ts" in kp_out.columns:
        kp_out = kp_out.drop(columns=["time_ts"])
    kp_out.to_csv(kp_csv, index=False)

    meta = {
        "summary": summary,
        "error_note": error_note,
        "live_sources": {
            "nasa_donki_flr": "https://api.nasa.gov/DONKI/FLR",
            "nasa_donki_cme_analysis": "https://api.nasa.gov/DONKI/CMEAnalysis",
            "nasa_donki_gst": "https://api.nasa.gov/DONKI/GST",
        },
    }
    summary_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return timeline_csv, kp_csv, summary_json


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------

class SolarStormScene:
    def __init__(self, timeline: pd.DataFrame, kp_df: pd.DataFrame, summary: Dict, source: str):
        self.timeline = timeline.sort_values("time_ts").reset_index(drop=True)
        self.kp_df = kp_df.sort_values("time_ts").reset_index(drop=True)
        self.summary = summary
        self.source = source
        self.stars = self._make_stars(CONFIG["stars"], seed=15)
        self.hud = self._make_hud(CONFIG["hud_lines"], seed=54)
        self.start_ts = self.timeline["time_ts"].min()
        self.end_ts = self.timeline["time_ts"].max()
        self.total_seconds = max((self.end_ts - self.start_ts).total_seconds(), 1.0)

    @staticmethod
    def _make_stars(n: int, seed: int):
        rng = np.random.default_rng(seed)
        items = []
        for _ in range(n):
            items.append({
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.5, 2.2)),
                "a": int(rng.integers(18, 110)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            })
        return items

    @staticmethod
    def _make_hud(n: int, seed: int):
        rng = np.random.default_rng(seed)
        lines = []
        for _ in range(n):
            lines.append({
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "length": float(rng.uniform(20, 96)),
                "alpha": int(rng.integers(8, 44)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            })
        return lines

    def background(self, t: float) -> Image.Image:
        img = Image.new("RGBA", OUT_SIZE, (6, 10, 18, 255))
        d = ImageDraw.Draw(img)
        for s in self.stars:
            alpha = int(s["a"] * (0.72 + 0.28 * math.sin(1.7 * t + s["phase"])))
            d.ellipse((s["x"]-s["r"], s["y"]-s["r"], s["x"]+s["r"], s["y"]+s["r"]), fill=(220, 232, 255, alpha))
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        neb = [
            (OUT_W * 0.16, OUT_H * 0.25, (115, 52, 16)),
            (OUT_W * 0.76, OUT_H * 0.30, (25, 58, 110)),
            (OUT_W * 0.48, OUT_H * 0.78, (18, 36, 88)),
        ]
        for cx, cy, col in neb:
            for rr, aa in [(450 * OUT_W/1080, 16), (300 * OUT_W/1080, 26), (190 * OUT_W/1080, 34)]:
                gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=(col[0], col[1], col[2], aa))
        glow = glow.filter(ImageFilter.GaussianBlur(60 if not QUICK_MODE else 30))
        img.alpha_composite(glow)
        return img

    def event_fraction(self, t: float) -> float:
        # Map full video duration to whole event, with small padding at intro and outro.
        if t <= SHOT_PLAN[0]["end"]:
            frac = 0.0
        elif t >= SHOT_PLAN[-1]["start"]:
            frac = 1.0
        else:
            frac = clamp((t - SHOT_PLAN[1]["start"]) / max(SHOT_PLAN[-2]["end"] - SHOT_PLAN[1]["start"], 1e-6))
        return frac

    def current_time(self, t: float) -> pd.Timestamp:
        frac = self.event_fraction(t)
        seconds = frac * self.total_seconds
        return self.start_ts + pd.Timedelta(seconds=float(seconds))

    def current_kp(self, t: float) -> float:
        if self.kp_df.empty:
            return 0.0
        current = self.current_time(t)
        mask = self.kp_df["time_ts"] <= current
        if not mask.any():
            return float(self.kp_df.iloc[0]["kp"])
        return float(self.kp_df.loc[mask].iloc[-1]["kp"])

    def current_event_row(self, t: float) -> pd.Series:
        current = self.current_time(t)
        mask = self.timeline["time_ts"] <= current
        if not mask.any():
            return self.timeline.iloc[0]
        return self.timeline.loc[mask].iloc[-1]

    def draw_sun(self, img: Image.Image, center: Tuple[float, float], radius: float, intensity: float):
        cx, cy = center
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for scale, aa in [(1.9, 18), (1.5, 40), (1.25, 82)]:
            rr = radius * scale
            gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=(255, 138 + int(24*intensity), 28, aa))
        glow = glow.filter(ImageFilter.GaussianBlur(22 if not QUICK_MODE else 11))
        img.alpha_composite(glow)

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for i in range(28, 0, -1):
            frac = i / 28.0
            rr = radius * frac
            col = (255, int(125 + 90 * frac), int(20 + 65 * frac), 255)
            d.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=col)

        # active region and flare tongues
        flare_phase = intensity
        for k in range(3):
            ang = -0.45 + k * 0.35
            base_x = cx + math.cos(ang) * radius * 0.7
            base_y = cy + math.sin(ang) * radius * 0.45
            length = radius * (0.7 + 0.6 * flare_phase) * (0.7 + 0.15 * k)
            tip_x = base_x + length
            tip_y = base_y - radius * (0.15 + 0.12 * k)
            d.line((base_x, base_y, tip_x, tip_y), fill=(255, 230, 165, int(120 + 80 * flare_phase)), width=max(1, int(radius * 0.05)))
            d.ellipse((tip_x-radius*0.07, tip_y-radius*0.07, tip_x+radius*0.07, tip_y+radius*0.07), fill=(255, 245, 220, int(160 + 70 * flare_phase)))
        img.alpha_composite(layer)

    def draw_earth(self, img: Image.Image, center: Tuple[float, float], radius: float, kp: float):
        cx, cy = center
        compression = 1.0 - 0.08 * (kp / 9.0)
        nightside = 1.0 + 0.35 * (kp / 9.0)

        # magnetosphere field lines
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        field_color = (85, 220, 255, 155)
        for scale in [1.3, 1.65, 2.0, 2.45, 2.9]:
            rx_front = radius * scale * compression
            rx_back = radius * scale * nightside
            ry = radius * scale * 0.8
            # day-side arc
            d.arc((cx-rx_front, cy-ry, cx+rx_back, cy+ry), start=55, end=305, fill=field_color, width=2)
        # bow shock / impact front
        d.arc((cx-radius*3.2*compression, cy-radius*2.2, cx+radius*4.5*nightside, cy+radius*2.2), start=55, end=305,
              fill=(255, 176, 72, int(90 + 80 * (kp / 9.0))), width=2)
        img.alpha_composite(overlay)

        # aurora ribbons if stormy
        if kp >= 5:
            aur = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            ad = ImageDraw.Draw(aur)
            glow_alpha = int(65 + 120 * (kp / 9.0))
            ad.arc((cx-radius*1.06, cy-radius*1.06, cx+radius*1.06, cy+radius*1.06), start=205, end=335, fill=(70, 255, 130, glow_alpha), width=max(2, int(radius*0.13)))
            ad.arc((cx-radius*1.06, cy-radius*1.06, cx+radius*1.06, cy+radius*1.06), start=25, end=155, fill=(95, 220, 255, glow_alpha), width=max(2, int(radius*0.13)))
            aur = aur.filter(ImageFilter.GaussianBlur(8 if not QUICK_MODE else 4))
            img.alpha_composite(aur)

        # Earth disc
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for i in range(22, 0, -1):
            frac = i / 22.0
            rr = radius * frac
            col = (int(30 + 18 * frac), int(85 + 80 * frac), int(170 + 55 * frac), 255)
            d.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=col)
        # continents (stylized)
        cont = [
            [(cx-radius*0.32, cy-radius*0.10), (cx-radius*0.05, cy-radius*0.26), (cx+radius*0.10, cy-radius*0.06), (cx-radius*0.03, cy+radius*0.06)],
            [(cx+radius*0.18, cy-radius*0.02), (cx+radius*0.34, cy+radius*0.12), (cx+radius*0.25, cy+radius*0.28), (cx+radius*0.08, cy+radius*0.17)],
            [(cx-radius*0.15, cy+radius*0.18), (cx-radius*0.04, cy+radius*0.35), (cx-radius*0.20, cy+radius*0.40), (cx-radius*0.29, cy+radius*0.24)],
        ]
        for poly in cont:
            d.polygon(poly, fill=(88, 168, 86, 230))
        d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=(225, 250, 255, 155), width=max(1, int(radius * 0.03)))
        img.alpha_composite(layer)

    def draw_cme_packets(self, img: Image.Image, t: float, phase: str):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        sun_x, sun_y = OUT_W * 0.20, OUT_H * 0.33
        earth_x, earth_y = OUT_W * 0.78, OUT_H * 0.48
        stormy = self.current_kp(t) / 9.0
        shot = get_shot(t)["name"]
        if shot in {"intro"}:
            packet_count = 1
        elif shot in {"launch"}:
            packet_count = 3
        elif shot in {"travel"}:
            packet_count = 5
        else:
            packet_count = 6
        for i in range(packet_count):
            phase_shift = (i * 0.17) % 1.0
            if shot == "launch":
                prog = clamp((t - SHOT_PLAN[1]["start"]) / max(SHOT_PLAN[1]["end"] - SHOT_PLAN[1]["start"], 1e-6) - phase_shift)
            else:
                prog = clamp(self.event_fraction(t) * 1.1 - phase_shift)
            x = lerp(sun_x + OUT_W*0.08, earth_x - OUT_W*0.08, prog)
            y = lerp(sun_y + math.sin(i)*20, earth_y + math.cos(i*1.7)*35, prog)
            size = (24 + 10 * i + 28 * stormy) * (OUT_W / 1080)
            alpha = int(40 + 120 * clamp(1.2 - abs(prog - 0.8)))
            for scale, aa in [(1.8, alpha//4), (1.25, alpha//2), (1.0, alpha)]:
                r = size * scale
                d.ellipse((x-r, y-r*0.75, x+r, y+r*0.75), fill=(255, 145, 62, aa))
        overlay = overlay.filter(ImageFilter.GaussianBlur(12 if not QUICK_MODE else 6))
        img.alpha_composite(overlay)

    def draw_axes_title(self, img: Image.Image, t: float):
        alpha = int(255 * smoothstep((t - 0.1) / 0.7) * (1.0 - smoothstep((t - (6.4 if not QUICK_MODE else 1.6)) / 0.7)))
        if alpha > 5:
            draw_text(img, CONFIG["title"], (56 if not QUICK_MODE else 28, 92 if not QUICK_MODE else 46), size=44 if not QUICK_MODE else 20,
                      fill=(245, 248, 252, alpha), bold=True)
            draw_text(img, CONFIG["subtitle"], (58 if not QUICK_MODE else 30, 154 if not QUICK_MODE else 77), size=24 if not QUICK_MODE else 11,
                      fill=(255, 180, 88, min(alpha, 230)), bold=True)

        shot_title = {
            "intro": "ERUPTION OVERVIEW",
            "launch": "FLARES + CMES LEAVE THE SUN",
            "travel": "SOLAR PLASMA CROSSES INTERPLANETARY SPACE",
            "impact": "EARTH IMPACT + MAGNETOSPHERE COMPRESSION",
            "effects": "G5 STORM EFFECTS + AURORA",
            "outro": "WHY THIS COUNTS AS SPACE WEATHER",
        }.get(get_shot(t)["name"], "")
        if t > (5.5 if not QUICK_MODE else 1.4):
            draw_text(img, shot_title, (56 if not QUICK_MODE else 28, 62 if not QUICK_MODE else 31), size=20 if not QUICK_MODE else 9,
                      fill=(148, 210, 230, 210), bold=True, stroke=1)

    def draw_header_hud(self, img: Image.Image, t: float):
        source_text = "SOURCE // NASA DONKI LIVE" if self.source == "live_nasa_donki" else "PREVIEW SOURCE // OFFICIAL FALLBACK MODEL"
        draw_text(img, source_text, (OUT_W - (48 if not QUICK_MODE else 24), 72 if not QUICK_MODE else 36), size=18 if not QUICK_MODE else 8,
                  fill=(100, 228, 245, 220), bold=True, anchor="ra", stroke=1)
        draw_text(img, "EVENT // MAY 2024 G5 STORM", (OUT_W - (48 if not QUICK_MODE else 24), 103 if not QUICK_MODE else 51), size=16 if not QUICK_MODE else 8,
                  fill=(165, 205, 220, 210), anchor="ra", stroke=1)
        current = self.current_time(t).strftime("%Y-%m-%d %H:%M UTC")
        draw_text(img, current, (OUT_W - (48 if not QUICK_MODE else 24), 132 if not QUICK_MODE else 66), size=16 if not QUICK_MODE else 8,
                  fill=(165, 205, 220, 195), anchor="ra", stroke=1)

    def draw_caption_panel(self, img: Image.Image, t: float):
        cap = caption_at(t)
        if not cap:
            return
        y0 = OUT_H - (242 if not QUICK_MODE else 122)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rounded_rectangle((44 if not QUICK_MODE else 22, y0, OUT_W - (44 if not QUICK_MODE else 22), y0 + (122 if not QUICK_MODE else 66)),
                            radius=24 if not QUICK_MODE else 12,
                            fill=(4, 8, 14, 168), outline=(255, 176, 74, 60), width=1)
        img.alpha_composite(overlay)
        draw_wrapped_text(img, cap, (68 if not QUICK_MODE else 34, y0 + (28 if not QUICK_MODE else 14)),
                          max_width=OUT_W - (136 if not QUICK_MODE else 68), size=30 if not QUICK_MODE else 14,
                          fill=(245, 248, 252, 245))

    def draw_hud_noise(self, img: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for line in self.hud:
            pulse = 0.5 + 0.5 * math.sin(2.0 * t + line["phase"])
            if pulse < 0.72:
                continue
            x = line["x"]
            y = (line["y"] + 9.0 * t) % OUT_H
            d.line((x, y, x + line["length"], y), fill=(92, 210, 240, int(line["alpha"] * pulse)), width=1)
        offset = int((t * 37) % 7)
        for y in range(offset, OUT_H, 7):
            d.line((0, y, OUT_W, y), fill=(120, 195, 238, 10), width=1)
        scan_y = int((t * 165) % (OUT_H + 220)) - 110
        d.rectangle((0, scan_y, OUT_W, scan_y + (48 if not QUICK_MODE else 24)), fill=(92, 210, 240, 8))
        img.alpha_composite(overlay)

    def draw_timeline(self, img: Image.Image, current_ts: pd.Timestamp):
        x0 = int(OUT_W * 0.08)
        x1 = int(OUT_W * 0.92)
        y = int(OUT_H * 0.84)
        d = ImageDraw.Draw(img)
        d.line((x0, y, x1, y), fill=(90, 160, 180, 90), width=2)
        total = max((self.end_ts - self.start_ts).total_seconds(), 1.0)
        for _, row in self.timeline.iterrows():
            frac = (row["time_ts"] - self.start_ts).total_seconds() / total
            x = x0 + frac * (x1 - x0)
            passed = row["time_ts"] <= current_ts
            col = (255, 182, 74, 235) if passed else (110, 125, 145, 130)
            d.ellipse((x-6, y-6, x+6, y+6), fill=col)
        cur_frac = (current_ts - self.start_ts).total_seconds() / total
        cur_x = x0 + cur_frac * (x1 - x0)
        d.rectangle((cur_x-2, y-18, cur_x+2, y+18), fill=(100, 225, 245, 220))
        draw_text(img, self.start_ts.strftime("May 7"), (x0, y + (22 if not QUICK_MODE else 10)), size=16 if not QUICK_MODE else 8,
                  fill=(160, 205, 220, 180), anchor="ma", stroke=1)
        draw_text(img, self.end_ts.strftime("May 13"), (x1, y + (22 if not QUICK_MODE else 10)), size=16 if not QUICK_MODE else 8,
                  fill=(160, 205, 220, 180), anchor="ma", stroke=1)

    def draw_event_card(self, img: Image.Image, row: pd.Series, kp: float):
        x0 = 56 if not QUICK_MODE else 28
        y0 = int(OUT_H * 0.67)
        w = int(OUT_W * 0.54)
        h = 156 if not QUICK_MODE else 80
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=24 if not QUICK_MODE else 12,
                            fill=(4, 8, 14, 170), outline=(255, 176, 74, 70), width=1)
        img.alpha_composite(overlay)
        draw_text(img, row["label"].upper(), (x0 + 18, y0 + (18 if not QUICK_MODE else 9)), size=22 if not QUICK_MODE else 10,
                  fill=(255, 185, 92, 235), bold=True, stroke=1)
        draw_text(img, pd.Timestamp(row["time_ts"]).strftime("%b %d, %H:%M UTC"), (x0 + 18, y0 + (54 if not QUICK_MODE else 27)), size=28 if not QUICK_MODE else 13,
                  fill=(245, 248, 252, 240), bold=True, stroke=1)
        draw_wrapped_text(img, str(row["details"]), (x0 + 18, y0 + (88 if not QUICK_MODE else 45)), w - 36,
                          size=20 if not QUICK_MODE else 10, fill=(165, 210, 225, 228))
        draw_text(img, f"Kp // {kp:.0f}", (x0 + 18, y0 + (128 if not QUICK_MODE else 65)), size=20 if not QUICK_MODE else 10,
                  fill=(100, 228, 245, 225), bold=True, stroke=1)
        g_text = "G" + str(int(max(0, min(5, round(max(kp - 4, 0)))))) if kp >= 5 else "G0"
        draw_text(img, g_text, (x0 + w - (18 if not QUICK_MODE else 9), y0 + (128 if not QUICK_MODE else 65)), size=28 if not QUICK_MODE else 13,
                  fill=(255, 176, 74, 240), bold=True, anchor="ra", stroke=1)

    def draw_kp_chart(self, img: Image.Image, current_ts: pd.Timestamp, top_left: Tuple[int, int], size: Tuple[int, int]):
        x0, y0 = top_left
        w, h = size
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=24 if not QUICK_MODE else 12,
                            fill=(4, 8, 14, 155), outline=(82, 142, 160, 60), width=1)
        # grid
        for k in range(0, 10, 2):
            yy = y0 + h - 18 - (k / 9.0) * (h - 36)
            d.line((x0 + 18, yy, x0 + w - 18, yy), fill=(100, 160, 180, 35), width=1)
        # data line
        if not self.kp_df.empty:
            ts0 = self.kp_df["time_ts"].min()
            ts1 = self.kp_df["time_ts"].max()
            total = max((ts1 - ts0).total_seconds(), 1.0)
            pts = []
            for _, row in self.kp_df.iterrows():
                frac = (row["time_ts"] - ts0).total_seconds() / total
                xx = x0 + 18 + frac * (w - 36)
                yy = y0 + h - 18 - (float(row["kp"]) / 9.0) * (h - 36)
                pts.append((xx, yy))
            if len(pts) > 1:
                d.line(pts, fill=(255, 176, 74, 220), width=3 if not QUICK_MODE else 2)
            # current marker
            frac_c = (current_ts - ts0).total_seconds() / total
            frac_c = clamp(frac_c)
            xc = x0 + 18 + frac_c * (w - 36)
            # interpolation approx by latest <= current
            latest = self.kp_df[self.kp_df["time_ts"] <= current_ts]
            kp_now = float(latest.iloc[-1]["kp"]) if not latest.empty else float(self.kp_df.iloc[0]["kp"])
            yc = y0 + h - 18 - (kp_now / 9.0) * (h - 36)
            d.line((xc, y0 + 16, xc, y0 + h - 16), fill=(100, 225, 245, 120), width=2)
            d.ellipse((xc-6, yc-6, xc+6, yc+6), fill=(100, 225, 245, 255), outline=(255, 255, 255, 160))
        img.alpha_composite(overlay)
        draw_text(img, "Kp / GEOMAGNETIC DISTURBANCE", (x0 + 18, y0 + 18 if not QUICK_MODE else y0 + 9), size=18 if not QUICK_MODE else 9,
                  fill=(100, 228, 245, 228), bold=True, stroke=1)
        draw_text(img, "9", (x0 + 10, y0 + 18), size=14 if not QUICK_MODE else 7, fill=(160, 205, 220, 180), anchor="rm", stroke=1)
        draw_text(img, "0", (x0 + 10, y0 + h - 18), size=14 if not QUICK_MODE else 7, fill=(160, 205, 220, 180), anchor="rm", stroke=1)

    def draw_stats_cards(self, img: Image.Image):
        cards = [
            ("NOTABLE FLARES", str(self.summary.get("nasa_official_context", {}).get("notable_flares_may3_9", 82))),
            ("MINIMUM CMES", str(self.summary.get("nasa_official_context", {}).get("at_least_cmes_may7_11", 7)) + "+"),
            ("PEAK LEVEL", "G5"),
            ("PEAK Kp", str(self.summary.get("peak_kp_proxy", self.summary.get("max_kp", 9) or 9))),
        ]
        x0 = int(OUT_W * 0.08)
        y0 = int(OUT_H * 0.25)
        card_w = int(OUT_W * 0.38)
        card_h = 110 if not QUICK_MODE else 56
        gap_x = int(OUT_W * 0.08)
        gap_y = 28 if not QUICK_MODE else 14
        coords = [
            (x0, y0),
            (x0 + card_w + gap_x, y0),
            (x0, y0 + card_h + gap_y),
            (x0 + card_w + gap_x, y0 + card_h + gap_y),
        ]
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for (cx, cy), _ in zip(coords, cards):
            d.rounded_rectangle((cx, cy, cx + card_w, cy + card_h), radius=22 if not QUICK_MODE else 11,
                                fill=(4, 8, 14, 168), outline=(255, 176, 74, 70), width=1)
        img.alpha_composite(overlay)
        for (cx, cy), (title, value) in zip(coords, cards):
            draw_text(img, title, (cx + 18, cy + (18 if not QUICK_MODE else 8)), size=20 if not QUICK_MODE else 9,
                      fill=(255, 185, 92, 235), bold=True, stroke=1)
            draw_text(img, value, (cx + 18, cy + (64 if not QUICK_MODE else 31)), size=34 if not QUICK_MODE else 16,
                      fill=(245, 248, 252, 242), bold=True, stroke=1)
        draw_wrapped_text(img, "G5 is the highest NOAA geomagnetic storm category and corresponds to Kp = 9.",
                          (x0, int(OUT_H * 0.55)), int(OUT_W * 0.84), size=20 if not QUICK_MODE else 10,
                          fill=(165, 210, 225, 225))

    def draw_impacts_list(self, img: Image.Image):
        x0 = int(OUT_W * 0.08)
        y0 = int(OUT_H * 0.26)
        w = int(OUT_W * 0.84)
        h = int(OUT_H * 0.44)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=26 if not QUICK_MODE else 13,
                            fill=(4, 8, 14, 168), outline=(255, 176, 74, 70), width=1)
        img.alpha_composite(overlay)
        draw_text(img, "G5 EFFECTS TO WATCH", (x0 + 20, y0 + (20 if not QUICK_MODE else 10)), size=28 if not QUICK_MODE else 13,
                  fill=(255, 185, 92, 235), bold=True, stroke=1)
        bullets = [
            "Aurora visible far from the poles",
            "Navigation and GPS can degrade",
            "HF radio can be disrupted",
            "Satellites face charging and drag issues",
            "Power grids can experience serious stress",
        ]
        y = y0 + (70 if not QUICK_MODE else 35)
        for b in bullets:
            draw_wrapped_text(img, "• " + b, (x0 + 22, y), w - 44, size=24 if not QUICK_MODE else 12,
                              fill=(245, 248, 252, 236), bold=True)
            y += 54 if not QUICK_MODE else 28

    def draw_outro_summary(self, img: Image.Image):
        x0 = int(OUT_W * 0.08)
        y0 = int(OUT_H * 0.28)
        w = int(OUT_W * 0.84)
        h = int(OUT_H * 0.40)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=26 if not QUICK_MODE else 13,
                            fill=(4, 8, 14, 172), outline=(255, 176, 74, 70), width=1)
        img.alpha_composite(overlay)
        draw_text(img, "WHAT JUST HIT EARTH?", (x0 + 20, y0 + (20 if not QUICK_MODE else 10)), size=28 if not QUICK_MODE else 13,
                  fill=(255, 185, 92, 235), bold=True, stroke=1)
        items = [
            "1) Solar flares flash radiation across space at light speed.",
            "2) CMEs hurl magnetized plasma outward much more slowly.",
            "3) When the plasma reaches Earth, it shakes the magnetosphere.",
            "4) That disturbance is what we call a geomagnetic storm.",
        ]
        y = y0 + (70 if not QUICK_MODE else 36)
        for item in items:
            draw_wrapped_text(img, item, (x0 + 22, y), w - 44, size=22 if not QUICK_MODE else 11,
                              fill=(245, 248, 252, 236), bold=True)
            y += 56 if not QUICK_MODE else 29

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        current_ts = self.current_time(t)
        kp = self.current_kp(t)
        current_event = self.current_event_row(t)
        storm_intensity = kp / 9.0

        img = self.background(t)
        self.draw_axes_title(img, t)
        self.draw_header_hud(img, t)

        # core astronomical composition
        self.draw_sun(img, (OUT_W * 0.20, OUT_H * 0.33), 120 * OUT_W / 1080, clamp(0.55 + 0.45 * storm_intensity))
        self.draw_earth(img, (OUT_W * 0.78, OUT_H * 0.48), 82 * OUT_W / 1080, kp)
        self.draw_cme_packets(img, t, str(current_event.get("phase", "")))

        if shot["name"] in {"intro", "launch", "travel", "impact"}:
            self.draw_event_card(img, current_event, kp)
            self.draw_kp_chart(img, current_ts, (int(OUT_W * 0.08), int(OUT_H * 0.80)), (int(OUT_W * 0.84), int(OUT_H * 0.11)))
            self.draw_timeline(img, current_ts)
        elif shot["name"] == "effects":
            self.draw_impacts_list(img)
            self.draw_kp_chart(img, current_ts, (int(OUT_W * 0.08), int(OUT_H * 0.76)), (int(OUT_W * 0.84), int(OUT_H * 0.12)))
        elif shot["name"] == "outro":
            self.draw_outro_summary(img)
            self.draw_stats_cards(img)

        self.draw_caption_panel(img, t)
        self.draw_hud_noise(img, t)

        arr = np.array(img.convert("RGB"))
        arr = apply_grade(arr)
        arr = np.clip(arr.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / 0.9)
        fade_out = 1.0 - smoothstep((t - (CONFIG["duration_s"] - 1.1)) / 1.0)
        arr = np.clip(arr.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)
        return arr


# -----------------------------------------------------------------------------
# Render
# -----------------------------------------------------------------------------

def render_video(scene: SolarStormScene):
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    print("Subtitle sidecar written:", srt_path.resolve())
    raw_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    times = np.arange(frame_count) / CONFIG["fps"]
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(raw_path, fps=CONFIG["fps"], codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None) as writer:
        for t in tqdm(times, desc="Rendering solar storm short"):
            writer.append_data(scene.render_frame(float(t)))
    shutil.copyfile(raw_path, final_path)
    print("Final video:", final_path.resolve())
    return final_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Loading solar storm event data ...")
    timeline, kp_df, summary, error_note = load_event_data()
    print("Source:", summary["source"])
    if error_note:
        print("Live fetch note:", error_note)

    timeline_csv, kp_csv, summary_json = save_data_products(timeline, kp_df, summary, error_note)
    print("Timeline CSV:", timeline_csv.resolve())
    print("Kp series CSV:", kp_csv.resolve())
    print("Summary JSON:", summary_json.resolve())

    scene = SolarStormScene(timeline, kp_df, summary, summary["source"])
    preview_times = [1.0, min(10.0, CONFIG["duration_s"] * 0.25), min(20.0, CONFIG["duration_s"] * 0.40), min(31.0, CONFIG["duration_s"] * 0.55), min(44.0, CONFIG["duration_s"] * 0.76), CONFIG["duration_s"] - 1.0]
    for pt in tqdm(preview_times, desc="Preview frames"):
        frame = scene.render_frame(float(pt))
        Image.fromarray(frame).save(PREVIEW_DIR / f"preview_{int(pt):02d}s.png")
    print("Preview frames written:", PREVIEW_DIR.resolve())

    render_video(scene)
    print("Output directory:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
