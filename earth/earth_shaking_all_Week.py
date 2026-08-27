from __future__ import annotations

"""
The Earth Has Been Shaking All Week — cinematic YouTube Short renderer

Creates a vertical 1080x1920 science short from the USGS Earthquake Hazards
Program's rolling 7-day GeoJSON feed.

Live source
-----------
USGS all-earthquakes, past 7 days:
    https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson

Each dot is one earthquake event in the feed. Dot size is based on magnitude;
colour is based on depth. The map is a stylized equirectangular world view,
with schematic plate-boundary guides used only as explanatory context.

Important science notes
-----------------------
- "All week" means the rolling 7 days at render time, not a calendar week.
- Earthquake catalogues include many small events that people never feel.
- Magnitude measures earthquake size; shaking intensity varies by location.
- Event counts change as the USGS catalogue is updated/reviewed.
- Plate-boundary lines here are schematic, not a tectonic GIS dataset.
- This video is an educational visualization, not an earthquake forecast.

Offline behaviour
-----------------
If the USGS feed is unavailable, a deterministic synthetic fixture is used and
clearly labelled in the rendered HUD and summary metadata.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    EARTHQUAKE_WEEK_SHORT_QUICK=1 python the_earth_has_been_shaking_all_week_short.py

Force offline fixture
---------------------
    EARTHQUAKE_WEEK_SHORT_OFFLINE=1 python the_earth_has_been_shaking_all_week_short.py
"""

import json
import math
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("EARTHQUAKE_WEEK_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("EARTHQUAKE_WEEK_SHORT_OFFLINE", "0") == "1"
USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson"

OUTPUT_ROOT = Path("the_earth_has_been_shaking_all_week_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for d in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    d.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "the_earth_has_been_shaking_all_week",
    "title": "THE EARTH HAS BEEN SHAKING ALL WEEK",
    "subtitle": "USGS // rolling 7 days // real earthquake locations",
    "timeout_s": 30,
    "max_render_events": 1400 if QUICK_MODE else 5000,
    "background_stars": 120 if QUICK_MODE else 260,
    "contrast": 1.07,
    "saturation": 1.05,
    "vignette": 0.23,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)

COLORS = {
    "dark": (2, 7, 15),
    "ocean": (6, 19, 33),
    "cyan": (71, 225, 255),
    "ice": (168, 233, 255),
    "white": (246, 250, 255),
    "muted": (158, 199, 218),
    "green": (100, 236, 170),
    "gold": (255, 195, 87),
    "orange": (255, 137, 67),
    "rose": (255, 88, 129),
    "violet": (184, 112, 255),
    "red": (255, 74, 74),
}

FULL_CAPTIONS = [
    (0.4, 7.0, "The ground has been moving all week — not in one place, but across the entire planet."),
    (7.1, 16.8, "Every dot is an earthquake reported by the USGS during the rolling seven days before this render."),
    (16.9, 27.0, "Most are small. Magnitude measures the size of an earthquake, while how strongly people feel it depends on where they are."),
    (27.1, 38.0, "The pattern is not random. Many earthquakes trace plate boundaries, where huge pieces of Earth's crust interact."),
    (38.1, 48.8, "Depth matters too. Shallow earthquakes sit near the surface; some subduction-zone earthquakes happen hundreds of kilometres down."),
    (48.9, 57.5, "Earth is always releasing tectonic stress. This map is a seven-day snapshot — not a prediction of what happens next."),
]
if QUICK_MODE:
    scale = float(CONFIG["duration_s"]) / 58.0
    CAPTIONS = [(a * scale, b * scale, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.7 if not QUICK_MODE else 1.6},
    {"name": "week_map", "start": 7.7 if not QUICK_MODE else 1.6, "end": 18.3 if not QUICK_MODE else 3.8},
    {"name": "magnitude", "start": 18.3 if not QUICK_MODE else 3.8, "end": 28.9 if not QUICK_MODE else 6.0},
    {"name": "plates", "start": 28.9 if not QUICK_MODE else 6.0, "end": 40.0 if not QUICK_MODE else 8.3},
    {"name": "depth", "start": 40.0 if not QUICK_MODE else 8.3, "end": 50.4 if not QUICK_MODE else 10.4},
    {"name": "finale", "start": 50.4 if not QUICK_MODE else 10.4, "end": float(CONFIG["duration_s"])},
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def smoothstep(v: float) -> float:
    x = clamp(v)
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
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(image: Image.Image, text: str, xy: Tuple[int, int], size: int = 28,
              fill=(255, 255, 255, 255), bold: bool = False, anchor: str = "la", stroke: int = 2):
    ImageDraw.Draw(image).text(
        xy, text, font=get_font(size, bold), fill=fill, anchor=anchor,
        stroke_width=stroke, stroke_fill=(0, 0, 0, 220)
    )


def draw_wrapped(image: Image.Image, text: str, xy: Tuple[int, int], max_width: int,
                 size: int, fill=(255, 255, 255, 245), bold: bool = False, spacing: int = 6):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
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
        y += (box[3] - box[1]) + spacing


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for i, (a, b, text) in enumerate(captions, 1):
        lines += [str(i), f"{format_srt_time(a)} --> {format_srt_time(b)}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def utc_text(ms: float) -> str:
    try:
        return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).strftime("%d %b %Y %H:%M UTC").upper()
    except Exception:
        return "TIME UNKNOWN"


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    r = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * r**1.8, 0, 1).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def fetch_usgs() -> Tuple[pd.DataFrame, str, List[str]]:
    notes: List[str] = []
    req = urllib.request.Request(USGS_URL, headers={"User-Agent": "earthquake-week-short-renderer/1.0"})
    with urllib.request.urlopen(req, timeout=float(CONFIG["timeout_s"])) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows: List[Dict[str, Any]] = []
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or [None, None, None]
        if len(coords) < 3:
            continue
        rows.append({
            "event_id": feature.get("id", ""),
            "lon": coords[0], "lat": coords[1], "depth_km": coords[2],
            "mag": props.get("mag"), "place": props.get("place", ""),
            "time_ms": props.get("time"), "updated_ms": props.get("updated"),
            "type": props.get("type", "earthquake"), "status": props.get("status", ""),
            "felt": props.get("felt"), "sig": props.get("sig"), "alert": props.get("alert"),
            "url": props.get("url", ""),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("USGS feed returned no events")
    for c in ("lon", "lat", "depth_km", "mag", "time_ms", "updated_ms", "felt", "sig"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame = frame.dropna(subset=["lon", "lat", "time_ms"]).copy()
    frame = frame[frame["lat"].between(-90, 90) & frame["lon"].between(-180, 180)].copy()
    frame["mag"] = frame["mag"].fillna(0.0)
    frame["depth_km"] = frame["depth_km"].fillna(0.0).clip(lower=0.0)
    frame["place"] = frame["place"].fillna("Unknown location").astype(str)
    frame = frame.sort_values("time_ms").reset_index(drop=True)
    notes.append(f"USGS feed metadata title: {payload.get('metadata', {}).get('title', 'unknown')}")
    return frame, "usgs_all_week_geojson", notes


def fallback_fixture() -> Tuple[pd.DataFrame, str, List[str]]:
    rng = np.random.default_rng(20260812)
    now = datetime.now(timezone.utc)
    n = 900 if QUICK_MODE else 2600
    # Schematic clusters approximating common seismic belts for layout testing only.
    centers = np.array([
        [-150, 55], [-122, 40], [-105, 18], [-75, -20], [-72, -35],
        [140, 38], [145, 20], [125, 5], [110, -7], [165, -20],
        [175, -40], [30, 38], [45, 15], [70, 30], [95, 28],
        [-20, 64], [-30, -5], [35, -5]
    ], dtype=float)
    idx = rng.integers(0, len(centers), n)
    lon = centers[idx, 0] + rng.normal(0, 5.5, n)
    lat = centers[idx, 1] + rng.normal(0, 3.0, n)
    lon = ((lon + 180) % 360) - 180
    lat = np.clip(lat, -85, 85)
    mag = np.clip(rng.gamma(1.5, 0.8, n) + 0.4, 0.1, 7.1)
    depth = np.clip(rng.lognormal(3.1, 1.0, n), 1, 650)
    hours_ago = np.sort(rng.uniform(0, 168, n))[::-1]
    times = np.array([(now - timedelta(hours=float(h))).timestamp() * 1000 for h in hours_ago])
    frame = pd.DataFrame({
        "event_id": [f"fixture-{i:05d}" for i in range(n)],
        "lon": lon, "lat": lat, "depth_km": depth, "mag": mag,
        "place": "SYNTHETIC PREVIEW EVENT", "time_ms": times,
        "updated_ms": times, "type": "earthquake", "status": "fixture",
        "felt": np.nan, "sig": np.nan, "alert": "", "url": "",
    }).sort_values("time_ms").reset_index(drop=True)
    return frame, "offline_synthetic_fixture", ["Offline fixture: synthetic data, not observed earthquakes"]


def load_data() -> Tuple[pd.DataFrame, str, List[str]]:
    if OFFLINE_MODE:
        return fallback_fixture()
    try:
        return fetch_usgs()
    except Exception as exc:
        frame, source, notes = fallback_fixture()
        notes.append(f"USGS fallback reason: {exc}")
        return frame, source, notes


def prepare(frame: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = frame.copy()
    t0 = float(out["time_ms"].min())
    t1 = float(out["time_ms"].max())
    out["time_norm"] = (out["time_ms"] - t0) / max(t1 - t0, 1.0)
    out["mag_norm"] = np.clip((out["mag"] - 0.0) / 7.5, 0, 1)
    out["depth_norm"] = np.clip(out["depth_km"] / 700.0, 0, 1)
    mag4 = out[out["mag"] >= 4.0]
    largest = out.sort_values("mag", ascending=False).iloc[0]
    summary = {
        "source": source,
        "is_live": source == "usgs_all_week_geojson",
        "event_count": int(len(out)),
        "magnitude_4_plus": int(len(mag4)),
        "largest_mag": float(largest["mag"]),
        "largest_place": str(largest["place"]),
        "largest_time": utc_text(float(largest["time_ms"])),
        "max_depth_km": float(out["depth_km"].max()),
        "median_depth_km": float(out["depth_km"].median()),
        "start_time": utc_text(t0),
        "end_time": utc_text(t1),
        "feed_url": USGS_URL,
        "warning": "Rolling 7-day catalogue snapshot; not an earthquake forecast. Plate boundaries are schematic.",
    }
    return out, summary


def save_data(frame: pd.DataFrame, summary: Dict[str, Any], notes: List[str]):
    frame.to_csv(DATA_ROOT / "usgs_earthquakes_past_7_days.csv", index=False)
    (DATA_ROOT / "earthquake_week_summary.json").write_text(
        json.dumps({"summary": summary, "notes": notes}, indent=2), encoding="utf-8"
    )


# -----------------------------------------------------------------------------
# Stylized map and scene renderer
# -----------------------------------------------------------------------------

# Schematic tectonic boundary polylines: visual context only.
PLATE_LINES: List[List[Tuple[float, float]]] = [
    [(-160, 60), (-150, 50), (-140, 40), (-130, 30), (-122, 20), (-110, 10), (-100, 0), (-85, -15), (-75, -30), (-70, -45), (-75, -55)],
    [(145, 55), (140, 40), (145, 25), (135, 10), (125, 0), (115, -10), (130, -20), (150, -30), (165, -42), (178, -50)],
    [(-30, 65), (-28, 45), (-25, 25), (-22, 5), (-18, -15), (-15, -35), (-10, -55)],
    [(-10, 36), (5, 38), (20, 40), (35, 38), (45, 34), (60, 30), (75, 30), (90, 28), (105, 25)],
    [(30, 30), (36, 15), (40, 0), (35, -15), (30, -30)],
    [(95, 28), (100, 15), (105, 0), (110, -10), (120, -15)],
]


class EarthquakeWeekScene:
    def __init__(self, frame: pd.DataFrame, summary: Dict[str, Any]):
        if len(frame) > int(CONFIG["max_render_events"]):
            # Keep strongest events plus a uniform time sample.
            strongest = frame.nlargest(int(CONFIG["max_render_events"] * 0.35), "mag")
            rest = frame.drop(strongest.index)
            take = int(CONFIG["max_render_events"]) - len(strongest)
            if len(rest) > take:
                ids = np.linspace(0, len(rest) - 1, take).astype(int)
                rest = rest.iloc[ids]
            frame = pd.concat([strongest, rest]).drop_duplicates("event_id").sort_values("time_ms")
        self.frame = frame.reset_index(drop=True)
        self.summary = summary
        self.map_box = (int(OUT_W * 0.055), int(OUT_H * 0.245), int(OUT_W * 0.945), int(OUT_H * 0.665))
        self.xy = self.project(self.frame["lon"].to_numpy(float), self.frame["lat"].to_numpy(float))
        self.time = self.frame["time_norm"].to_numpy(float)
        self.mag = self.frame["mag"].to_numpy(float)
        self.depth = self.frame["depth_km"].to_numpy(float)
        self.order_time = np.argsort(self.time)
        self.order_mag = np.argsort(self.mag)
        self.stars = self.make_stars(int(CONFIG["background_stars"]))

    @staticmethod
    def make_stars(n: int):
        rng = np.random.default_rng(821)
        return [(float(rng.uniform(0, OUT_W)), float(rng.uniform(0, OUT_H)), float(rng.uniform(.4, 1.8)), int(rng.uniform(18, 90))) for _ in range(n)]

    def project(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.map_box
        x = x0 + (np.asarray(lon) + 180.0) / 360.0 * (x1 - x0)
        y = y1 - (np.asarray(lat) + 90.0) / 180.0 * (y1 - y0)
        return np.column_stack([x, y])

    def background(self, t: float) -> Image.Image:
        im = Image.new("RGBA", OUT_SIZE, COLORS["dark"] + (255,))
        d = ImageDraw.Draw(im)
        for x, y, r, a in self.stars:
            tw = .75 + .25 * math.sin(t * 1.2 + x * .01)
            d.ellipse((x-r, y-r, x+r, y+r), fill=COLORS["ice"] + (int(a*tw),))
        haze = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        hd = ImageDraw.Draw(haze)
        hd.ellipse((OUT_W*.05, OUT_H*.08, OUT_W*.95, OUT_H*.72), fill=(17, 70, 120, 24))
        haze = haze.filter(ImageFilter.GaussianBlur(70 if not QUICK_MODE else 35))
        im.alpha_composite(haze)
        return im

    def panel(self, im: Image.Image, box: Tuple[int,int,int,int], alpha: int = 170):
        ov = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        ImageDraw.Draw(ov).rounded_rectangle(box, radius=22 if not QUICK_MODE else 11,
            fill=(2, 8, 18, alpha), outline=COLORS["cyan"] + (55,), width=1)
        im.alpha_composite(ov)

    def draw_world(self, im: Image.Image, plates: bool = False, alpha: int = 85):
        x0,y0,x1,y1 = self.map_box
        d = ImageDraw.Draw(im)
        d.rounded_rectangle(self.map_box, radius=22 if not QUICK_MODE else 11,
                            fill=COLORS["ocean"] + (225,), outline=COLORS["cyan"] + (70,), width=2)
        for lon in range(-150, 180, 30):
            p = self.project(np.array([lon, lon]), np.array([-80,80]))
            d.line([tuple(p[0]), tuple(p[1])], fill=COLORS["muted"] + (30,), width=1)
        for lat in range(-60, 90, 30):
            p = self.project(np.array([-180,180]), np.array([lat,lat]))
            d.line([tuple(p[0]), tuple(p[1])], fill=COLORS["muted"] + (30,), width=1)
        # Minimal continent silhouettes, intentionally schematic.
        continents = [
            [(-165,70),(-140,55),(-125,50),(-110,30),(-90,20),(-80,8),(-95,5),(-110,20),(-130,25),(-150,45)],
            [(-82,12),(-70,5),(-60,-10),(-55,-30),(-65,-50),(-75,-35),(-78,-10)],
            [(-10,35),(10,55),(40,65),(70,55),(100,50),(125,35),(145,45),(160,55),(175,40),(150,10),(120,5),(95,20),(75,10),(55,25),(40,10),(25,35)],
            [(-20,35),(10,35),(35,20),(45,0),(35,-20),(20,-35),(5,-30),(-5,-5),(-15,15)],
            [(112,-10),(155,-12),(150,-40),(125,-45),(112,-25)],
            [(-55,82),(-25,80),(-20,65),(-45,58),(-60,68)],
        ]
        for poly in continents:
            arr = self.project(np.array([p[0] for p in poly]), np.array([p[1] for p in poly]))
            d.polygon([tuple(p) for p in arr], fill=(20,48,55,210), outline=(95,150,145,85))
        if plates:
            for line in PLATE_LINES:
                arr = self.project(np.array([p[0] for p in line]), np.array([p[1] for p in line]))
                d.line([tuple(p) for p in arr], fill=COLORS["gold"] + (alpha,), width=3 if not QUICK_MODE else 2)

    @staticmethod
    def depth_color(depth: float) -> Tuple[int,int,int]:
        t = clamp(depth / 600.0)
        if t < .5:
            u = t / .5
            return tuple(int(lerp(COLORS["cyan"][i], COLORS["gold"][i], u)) for i in range(3))
        u = (t-.5)/.5
        return tuple(int(lerp(COLORS["gold"][i], COLORS["rose"][i], u)) for i in range(3))

    def draw_events(self, im: Image.Image, indices: np.ndarray, reveal: float = 1.0,
                    mode: str = "depth", alpha: int = 220, pulse_t: float = 0.0):
        n = int(round(len(indices) * clamp(reveal)))
        if n <= 0:
            return
        selected = indices[:n]
        glow = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        ov = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        od = ImageDraw.Draw(ov)
        for j, idx in enumerate(selected):
            idx = int(idx)
            x,y = self.xy[idx]
            mag = max(float(self.mag[idx]), 0.0)
            depth = float(self.depth[idx])
            r = (1.5 + max(0.0, mag)**1.55 * .8) * OUT_W / 1080
            if mode == "magnitude":
                t = clamp(mag / 7.0)
                c = tuple(int(lerp(COLORS["ice"][k], COLORS["red"][k], t)) for k in range(3))
            else:
                c = self.depth_color(depth)
            if mag >= 5.0:
                pulse = 1.0 + .25 * math.sin(pulse_t*5 + j)
                rr = r * 2.8 * pulse
                gd.ellipse((x-rr,y-rr,x+rr,y+rr), fill=c + (55,))
            od.ellipse((x-r,y-r,x+r,y+r), fill=c + (alpha,))
        glow = glow.filter(ImageFilter.GaussianBlur(8 if not QUICK_MODE else 4))
        im.alpha_composite(glow)
        im.alpha_composite(ov)

    def source_hud(self, im: Image.Image):
        live = bool(self.summary["is_live"])
        c = COLORS["cyan"] if live else COLORS["gold"]
        label = "SOURCE // USGS PAST 7 DAYS" if live else "SOURCE // SYNTHETIC OFFLINE FIXTURE"
        draw_text(im, label, (OUT_W-34, 58 if not QUICK_MODE else 29), 15 if not QUICK_MODE else 7, c+(235,), True, "ra", 1)
        draw_text(im, f"EVENTS // {self.summary['event_count']:,}", (OUT_W-34, 88 if not QUICK_MODE else 44), 14 if not QUICK_MODE else 7, COLORS["muted"]+(210,), False, "ra", 1)

    def intro(self, im: Image.Image, t: float):
        shot = SHOT_PLAN[0]
        local = smoothstep((t-shot["start"])/(shot["end"]-shot["start"]))
        cx, cy = OUT_W*.5, OUT_H*.39
        d = ImageDraw.Draw(im)
        # Earth disc
        r = 190*OUT_W/1080
        d.ellipse((cx-r,cy-r,cx+r,cy+r), fill=(5,28,48,255), outline=COLORS["cyan"]+(100,), width=2)
        # Seismic rings
        for k in range(6):
            phase = (local*2.3 + k/6) % 1
            rr = r * (.25 + .9*phase)
            a = int(130*(1-phase))
            d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr), outline=COLORS["rose"]+(a,), width=max(1,int(3*OUT_W/1080)))
        draw_text(im, "THE PLANET NEVER REALLY STOPS MOVING", (OUT_W//2,int(OUT_H*.69)), 25 if not QUICK_MODE else 12, COLORS["white"]+(245,), True, "ma", 1)
        draw_text(im, "a rolling seven-day seismic snapshot", (OUT_W//2,int(OUT_H*.735)), 18 if not QUICK_MODE else 9, COLORS["cyan"]+(225,), False, "ma", 1)

    def week_map(self, im: Image.Image, t: float):
        shot = SHOT_PLAN[1]
        local = smoothstep((t-shot["start"])/(shot["end"]-shot["start"]))
        self.draw_world(im)
        self.draw_events(im, self.order_time, reveal=min(1,local*1.15), mode="depth", pulse_t=t)
        self.panel(im,(int(OUT_W*.08),int(OUT_H*.70),int(OUT_W*.92),int(OUT_H*.82)),155)
        draw_text(im,"SEVEN DAYS OF EARTHQUAKES",(OUT_W//2,int(OUT_H*.74)),24 if not QUICK_MODE else 12,COLORS["cyan"]+(245,),True,"ma",1)
        draw_text(im,f"{self.summary['event_count']:,} events in this catalogue snapshot",(OUT_W//2,int(OUT_H*.785)),17 if not QUICK_MODE else 8,COLORS["white"]+(225,),False,"ma",1)

    def magnitude_scene(self, im: Image.Image, t: float):
        shot = SHOT_PLAN[2]
        local = smoothstep((t-shot["start"])/(shot["end"]-shot["start"]))
        self.draw_world(im)
        self.draw_events(im,self.order_mag,reveal=1.0,mode="magnitude",alpha=205,pulse_t=t)
        largest = self.frame.sort_values("mag",ascending=False).iloc[0]
        self.panel(im,(int(OUT_W*.08),int(OUT_H*.695),int(OUT_W*.92),int(OUT_H*.84)),170)
        draw_text(im,"MAGNITUDE = EARTHQUAKE SIZE",(OUT_W//2,int(OUT_H*.735)),23 if not QUICK_MODE else 11,COLORS["rose"]+(245,),True,"ma",1)
        draw_text(im,f"largest in this snapshot // M {float(largest['mag']):.1f}",(OUT_W//2,int(OUT_H*.775)),19 if not QUICK_MODE else 9,COLORS["gold"]+(240,),True,"ma",1)
        place = str(largest["place"])[:68]
        draw_text(im,place,(OUT_W//2,int(OUT_H*.812)),14 if not QUICK_MODE else 7,COLORS["white"]+(210,),False,"ma",1)

    def plates_scene(self, im: Image.Image, t: float):
        shot = SHOT_PLAN[3]
        local = smoothstep((t-shot["start"])/(shot["end"]-shot["start"]))
        self.draw_world(im,plates=True,alpha=int(80+130*local))
        self.draw_events(im,self.order_time,reveal=1.0,mode="depth",alpha=190,pulse_t=t)
        self.panel(im,(int(OUT_W*.08),int(OUT_H*.70),int(OUT_W*.92),int(OUT_H*.83)),165)
        draw_text(im,"THE PATTERN FOLLOWS TECTONICS",(OUT_W//2,int(OUT_H*.742)),23 if not QUICK_MODE else 11,COLORS["gold"]+(245,),True,"ma",1)
        draw_text(im,"many earthquakes cluster along interacting plate boundaries",(OUT_W//2,int(OUT_H*.786)),16 if not QUICK_MODE else 8,COLORS["white"]+(220,),False,"ma",1)
        draw_text(im,"plate lines are schematic",(OUT_W//2,int(OUT_H*.815)),13 if not QUICK_MODE else 6,COLORS["muted"]+(190,),False,"ma",1)

    def depth_scene(self, im: Image.Image, t: float):
        shot = SHOT_PLAN[4]
        local = smoothstep((t-shot["start"])/(shot["end"]-shot["start"]))
        self.draw_world(im,plates=True,alpha=65)
        self.draw_events(im,self.order_time,reveal=1.0,mode="depth",alpha=215,pulse_t=t)
        self.panel(im,(int(OUT_W*.08),int(OUT_H*.685),int(OUT_W*.92),int(OUT_H*.845)),175)
        draw_text(im,"DEPTH REVEALS ANOTHER DIMENSION",(OUT_W//2,int(OUT_H*.725)),22 if not QUICK_MODE else 11,COLORS["cyan"]+(245,),True,"ma",1)
        labels=[("SHALLOW",COLORS["cyan"]),("INTERMEDIATE",COLORS["gold"]),("DEEP",COLORS["rose"])]
        for i,(name,c) in enumerate(labels):
            x=int(OUT_W*(.27+.23*i)); y=int(OUT_H*.775)
            ImageDraw.Draw(im).ellipse((x-8,y-8,x+8,y+8),fill=c+(240,))
            draw_text(im,name,(x,int(OUT_H*.806)),13 if not QUICK_MODE else 6,COLORS["white"]+(220,),True,"ma",1)

    def finale(self, im: Image.Image, t: float):
        shot = SHOT_PLAN[5]
        local = smoothstep((t-shot["start"])/(shot["end"]-shot["start"]))
        self.draw_world(im,plates=True,alpha=80)
        self.draw_events(im,self.order_time,reveal=1.0,mode="depth",alpha=215,pulse_t=t)
        self.panel(im,(int(OUT_W*.065),int(OUT_H*.625),int(OUT_W*.935),int(OUT_H*.84)),188)
        draw_text(im,"THE EARTH HAS BEEN SHAKING ALL WEEK",(OUT_W//2,int(OUT_H*.665)),27 if not QUICK_MODE else 13,COLORS["white"]+(250,),True,"ma",1)
        draw_text(im,f"{self.summary['event_count']:,} catalogued events // rolling 7 days",(OUT_W//2,int(OUT_H*.72)),18 if not QUICK_MODE else 9,COLORS["gold"]+(238,),True,"ma",1)
        draw_text(im,"a snapshot of tectonic stress being released",(OUT_W//2,int(OUT_H*.762)),17 if not QUICK_MODE else 8,COLORS["cyan"]+(225,),False,"ma",1)
        draw_text(im,"NOT AN EARTHQUAKE FORECAST",(OUT_W//2,int(OUT_H*.805)),14 if not QUICK_MODE else 7,COLORS["rose"]+(225,),True,"ma",1)

    def captions(self, im: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        box=(int(OUT_W*.065),int(OUT_H*.86),int(OUT_W*.935),int(OUT_H*.955))
        self.panel(im,box,178)
        draw_wrapped(im,text,(int(OUT_W*.095),int(OUT_H*.878)),int(OUT_W*.81),20 if not QUICK_MODE else 10,COLORS["white"]+(245,),False,5 if not QUICK_MODE else 2)

    def title_hud(self, im: Image.Image, shot_name: str):
        labels={"intro":"SEISMIC WEEK","week_map":"GLOBAL SNAPSHOT","magnitude":"MAGNITUDE","plates":"PLATE BOUNDARIES","depth":"DEPTH","finale":"SEVEN-DAY REVEAL"}
        draw_text(im,CONFIG["title"],(34,58 if not QUICK_MODE else 29),20 if not QUICK_MODE else 10,COLORS["white"]+(245,),True,"la",1)
        draw_text(im,labels[shot_name],(34,90 if not QUICK_MODE else 45),14 if not QUICK_MODE else 7,COLORS["rose"]+(230,),True,"la",1)

    def render(self, t: float) -> np.ndarray:
        im=self.background(t)
        shot=get_shot(t)
        name=shot["name"]
        if name=="intro": self.intro(im,t)
        elif name=="week_map": self.week_map(im,t)
        elif name=="magnitude": self.magnitude_scene(im,t)
        elif name=="plates": self.plates_scene(im,t)
        elif name=="depth": self.depth_scene(im,t)
        else: self.finale(im,t)
        self.title_hud(im,name)
        self.source_hud(im)
        self.captions(im,t)
        arr=np.asarray(im.convert("RGB"),dtype=np.float32)
        arr*=VIGNETTE[...,None]
        arr=np.clip(arr,0,255).astype(np.uint8)
        graded=Image.fromarray(arr)
        graded=ImageEnhance.Contrast(graded).enhance(float(CONFIG["contrast"]))
        graded=ImageEnhance.Color(graded).enhance(float(CONFIG["saturation"]))
        return np.asarray(graded)


# -----------------------------------------------------------------------------
# Render
# -----------------------------------------------------------------------------

def render_video(scene: EarthquakeWeekScene) -> Path:
    fps=int(CONFIG["fps"]); duration=float(CONFIG["duration_s"])
    total=int(round(fps*duration))
    out=OUTPUT_ROOT/(CONFIG["output_basename"]+"_final.mp4")
    writer=iio.get_writer(out,fps=fps,codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=1)
    try:
        for i in tqdm(range(total),desc="Rendering earthquake short"):
            writer.append_data(scene.render(i/fps))
    finally:
        writer.close()
    return out


def render_previews(scene: EarthquakeWeekScene):
    times=[1.0, 9.0, 22.0, 34.0, 44.5, 55.0] if not QUICK_MODE else [0.6,2.4,4.8,7.1,9.3,11.2]
    paths=[]
    for t in times:
        path=PREVIEW_DIR/f"preview_{t:04.1f}s.png"
        Image.fromarray(scene.render(t)).save(path)
        paths.append(path)
    return paths


def main():
    frame,source,notes=load_data()
    frame,summary=prepare(frame,source)
    save_data(frame,summary,notes)
    srt=write_srt(CAPTIONS,OUTPUT_ROOT/(CONFIG["output_basename"]+".srt"))
    scene=EarthquakeWeekScene(frame,summary)
    previews=render_previews(scene)
    video=render_video(scene)
    manifest={
        "video":str(video),"srt":str(srt),"previews":[str(p) for p in previews],
        "summary":summary,"notes":notes,"quick_mode":QUICK_MODE,
        "science_notes":[
            "The USGS feed is a rolling seven-day catalogue snapshot.",
            "Magnitude is earthquake size; shaking intensity varies by place.",
            "Plate-boundary guides are schematic and explanatory only.",
            "This visualization is not an earthquake forecast.",
        ]
    }
    (OUTPUT_ROOT/"render_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
