from __future__ import annotations

"""
Every Earthquake This Month — cinematic YouTube Short renderer

Creates a vertical 1080x1920 data-driven short that plots every earthquake in a
calendar month as luminous seismic pulses on a moving world map. The visual is
map-first and cinematic: no conventional chart panels, only chronological
accumulation, depth-coded light, a Pacific-margin sequence, selected-event
closeups, captions, film texture, and a generated ambient soundtrack.

Preferred live source
---------------------
USGS Earthquake Hazards Program / ANSS Comprehensive Earthquake Catalog
(ComCat), queried through the official FDSN Event Web Service. The script asks
for every event whose event type is "earthquake" between the first instant of
the target UTC month and the current instant (or the end of a completed month).
If the result would exceed the service's normal single-query limit, the month is
requested one UTC day at a time and deduplicated by event ID.

The default target is the current UTC month. To render a completed month:

    EARTHQUAKE_MONTH=2026-07 python every_earthquake_this_month_short.py



Offline behavior
----------------
If USGS or Natural Earth cannot be reached, deterministic procedural earthquake
fixtures and coarse built-in land polygons are used. The result remains useful
for timing and layout previews, but is prominently labeled as synthetic.

Recommended install
-------------------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm pyshp

Quick preview render
--------------------
    EARTHQUAKE_SHORT_QUICK=1 python every_earthquake_this_month_short.py

Force offline fixture mode
--------------------------
    EARTHQUAKE_SHORT_OFFLINE=1 python every_earthquake_this_month_short.py

Outputs
-------
- final vertical MP4 with generated ambient audio when ffmpeg is available
- silent MP4 fallback
- SRT subtitle sidecar
- preview PNG frames
- CSV export of all queried earthquakes
- JSON summary and source notes
- cached USGS GeoJSON and Natural Earth land geometry

Primary sources
---------------
- USGS FDSN Event Web Service: https://earthquake.usgs.gov/fdsnws/event/1/
- USGS GeoJSON summary format:
  https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php
- USGS ComCat documentation: https://earthquake.usgs.gov/data/comcat/
- USGS earthquake depth explanation:
  https://www.usgs.gov/programs/earthquake-hazards/determining-depth-earthquake
- Natural Earth land polygons: https://www.naturalearthdata.com/
"""

import calendar
import json
import math
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
import wave
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import shapefile  # pyshp
except Exception:
    shapefile = None

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None


# -----------------------------------------------------------------------------
# Configuration and month selection
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("EARTHQUAKE_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("EARTHQUAKE_SHORT_OFFLINE", "0") == "1"


def next_month_start(value: datetime) -> datetime:
    if value.month == 12:
        return datetime(value.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(value.year, value.month + 1, 1, tzinfo=timezone.utc)


def resolve_month_window() -> Tuple[datetime, datetime, str]:
    now = datetime.now(timezone.utc)
    override = os.environ.get("EARTHQUAKE_MONTH", "").strip()
    if override:
        try:
            selected = datetime.strptime(override, "%Y-%m").replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError("EARTHQUAKE_MONTH must use YYYY-MM, for example 2026-07") from exc
        start = datetime(selected.year, selected.month, 1, tzinfo=timezone.utc)
        next_start = next_month_start(start)
        end = min(now, next_start) if start.year == now.year and start.month == now.month else next_start
        if start > now:
            raise ValueError("EARTHQUAKE_MONTH cannot be in the future")
    else:
        start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        end = now
    label = f"{calendar.month_name[start.month].upper()} {start.year}"
    return start, end, label


MONTH_START, MONTH_END, MONTH_LABEL = resolve_month_window()
MONTH_KEY = MONTH_START.strftime("%Y_%m")
MONTH_IS_IN_PROGRESS = MONTH_END < next_month_start(MONTH_START)

OUTPUT_ROOT = Path(f"every_earthquake_{MONTH_KEY}_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
CACHE_ROOT = DATA_ROOT / "cache"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in [OUTPUT_ROOT, DATA_ROOT, CACHE_ROOT, PREVIEW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {

}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)

COLORS = {

}

FULL_SHOT_PLAN = [
    {"name": "awakening", "start": 0.0, "end": 8.0},
    {"name": "month_sweep", "start": 8.0, "end": 27.5},
    {"name": "pacific_margin", "start": 27.5, "end": 39.0},
    {"name": "depth", "start": 39.0, "end": 47.0},
    {"name": "largest", "start": 47.0, "end": 55.0},
    {"name": "all_events", "start": 55.0, "end": 58.0},
]

FULL_CAPTIONS = [
    (0.5, 7.6, "The ground is never completely still. Seismic networks locate earthquakes around the planet every day."),
    (8.1, 27.1, "Every pulse is one earthquake returned by the USGS catalog for this UTC month, appearing in chronological order."),
    (27.6, 38.6, "The brightest arcs gather around active plate margins—especially around the Pacific."),
    (39.1, 46.6, "Color marks depth: cyan is shallow, gold is intermediate, and magenta is hundreds of kilometers below the surface."),
    (47.1, 54.6, "The month’s largest events stand out, but most catalogued earthquakes are far smaller."),
    (55.1, 57.7, "One month of motion, recorded as thousands of points on a restless planet."),
]

if QUICK_MODE:
    _scale = float(CONFIG["duration_s"]) / 58.0
    SHOT_PLAN = [
        {"name": shot["name"], "start": shot["start"] * _scale, "end": shot["end"] * _scale}
        for shot in FULL_SHOT_PLAN
    ]
    CAPTIONS = [(a * _scale, b * _scale, text) for a, b, text in FULL_CAPTIONS]
else:
    SHOT_PLAN = FULL_SHOT_PLAN
    CAPTIONS = FULL_CAPTIONS

USGS_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_COUNT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/count"
NATURAL_EARTH_LAND_URL = "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip"


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Earthquake:
    event_id: str
    time_utc: datetime
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float
    place: str
    significance: int = 0
    felt: int = 0
    tsunami: int = 0
    status: str = ""
    url: str = ""

    @property
    def display_place(self) -> str:
        text = (self.place or "Location not specified").strip()
        return text if len(text) <= 58 else text[:55].rstrip() + "…"

    @property
    def magnitude_label(self) -> str:
        return "M ?" if not np.isfinite(self.magnitude) else f"M {self.magnitude:.1f}"

    @property
    def time_fraction(self) -> float:
        span = max((MONTH_END - MONTH_START).total_seconds(), 1.0)
        return clamp((self.time_utc - MONTH_START).total_seconds() / span)


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(value: float) -> float:
    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


def smootherstep(value: float) -> float:
    x = clamp(value)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def shot_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - float(shot["start"])) / max(float(shot["end"] - shot["start"]), 1e-9))


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


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


def get_font(size: int, bold: bool = False, condensed: bool = False):
    candidates: List[str] = []
    if condensed and bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
            "DejaVuSansCondensed-Bold.ttf",
        ])
    if condensed:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
            "DejaVuSansCondensed.ttf",
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ])
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int,
    fill: Tuple[int, int, int, int],
    bold: bool = False,
    condensed: bool = False,
    anchor: str = "la",
    stroke: int = 2,
):
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold=bold, condensed=condensed),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(230, fill[3])),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill: Tuple[int, int, int, int],
    line_spacing: int,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size)
    words = text.split()
    lines: List[str] = []
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
        draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 225))
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += bbox[3] - bbox[1] + line_spacing


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.78, 0.0, 1.0).astype(np.float32)


def apply_grade(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array)
    image = ImageEnhance.Contrast(image).enhance(float(CONFIG["contrast"]))
    image = ImageEnhance.Color(image).enhance(float(CONFIG["saturation"]))
    return np.asarray(image)


def request_bytes(url: str, timeout: int = 45) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; EveryEarthquakeShort/1.0; educational visualization)",
            "Accept": "application/geo+json,application/json,text/plain,application/zip,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def parse_utc_millis(value: Any) -> datetime:
    return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)


def alpha_composite_with_opacity(base: Image.Image, layer: Image.Image, opacity: float) -> None:
    opacity = clamp(opacity)
    if opacity <= 0:
        return
    if opacity < 0.999:
        layer = layer.copy()
        alpha = layer.getchannel("A").point(lambda value: int(value * opacity))
        layer.putalpha(alpha)
    base.alpha_composite(layer)


def zoom_and_shift(layer: Image.Image, zoom: float, dx: float, dy: float) -> Image.Image:
    zoom = max(1.0, float(zoom))
    new_size = (int(round(OUT_W * zoom)), int(round(OUT_H * zoom)))
    resized = layer.resize(new_size, Image.Resampling.BICUBIC)
    left = int((new_size[0] - OUT_W) / 2.0 - dx)
    top = int((new_size[1] - OUT_H) / 2.0 - dy)
    left = max(0, min(left, new_size[0] - OUT_W))
    top = max(0, min(top, new_size[1] - OUT_H))
    return resized.crop((left, top, left + OUT_W, top + OUT_H))


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# USGS ComCat loading
# -----------------------------------------------------------------------------

def build_query_url(base: str, start: datetime, end: datetime, include_format: bool = True) -> str:
    params: Dict[str, str] = {
        "starttime": iso_utc(start),
        "endtime": iso_utc(end),
        "eventtype": "earthquake",
    }
    if include_format:
        params.update({"format": "geojson", "orderby": "time-asc", "limit": "20000"})
    return base + "?" + urllib.parse.urlencode(params)


def parse_geojson(payload: Dict[str, Any]) -> List[Earthquake]:
    events: List[Earthquake] = []
    for feature in payload.get("features", []):
        try:
            event_id = str(feature.get("id") or "").strip()
            properties = feature.get("properties") or {}
            geometry = feature.get("geometry") or {}
            coordinates = geometry.get("coordinates") or []
            if len(coordinates) < 3 or not event_id:
                continue
            lon, lat, depth = map(float, coordinates[:3])
            time_utc = parse_utc_millis(properties.get("time"))
            if not (MONTH_START <= time_utc <= MONTH_END + timedelta(seconds=1)):
                continue
            magnitude_raw = properties.get("mag")
            magnitude = float(magnitude_raw) if magnitude_raw is not None else float("nan")
            events.append(
                Earthquake(
                    event_id=event_id,
                    time_utc=time_utc,
                    latitude=lat,
                    longitude=lon,
                    depth_km=max(float(depth), 0.0),
                    magnitude=magnitude,
                    place=str(properties.get("place") or "Location not specified"),
                    significance=int(properties.get("sig") or 0),
                    felt=int(properties.get("felt") or 0),
                    tsunami=int(properties.get("tsunami") or 0),
                    status=str(properties.get("status") or ""),
                    url=str(properties.get("url") or ""),
                )
            )
        except Exception:
            continue
    return events


def query_interval(start: datetime, end: datetime) -> Tuple[List[Earthquake], Dict[str, Any]]:
    url = build_query_url(USGS_QUERY_URL, start, end, include_format=True)
    payload = json.loads(request_bytes(url, timeout=75).decode("utf-8"))
    return parse_geojson(payload), payload.get("metadata") or {}


def query_count(start: datetime, end: datetime) -> int:
    url = build_query_url(USGS_COUNT_URL, start, end, include_format=False)
    return int(request_bytes(url, timeout=45).decode("utf-8").strip())


def fetch_month_from_usgs() -> Tuple[List[Earthquake], List[str], Dict[str, Any]]:
    notes: List[str] = []
    try:
        expected_count = query_count(MONTH_START, MONTH_END)
    except Exception as exc:
        expected_count = -1
        notes.append(f"USGS count request failed; attempting the event query directly: {exc}")
    metadata: Dict[str, Any] = {"expected_count": None if expected_count < 0 else expected_count}
    events: List[Earthquake] = []

    if expected_count < 0 or expected_count <= int(CONFIG["api_query_limit"]):
        events, query_metadata = query_interval(MONTH_START, MONTH_END)
        metadata.update(query_metadata)
        notes.append("Loaded the month in one USGS FDSN query")
    else:
        notes.append(
            f"USGS count was {expected_count:,}; queried one UTC day at a time to avoid the single-query limit"
        )
        cursor = MONTH_START
        while cursor < MONTH_END:
            interval_end = min(cursor + timedelta(days=1), MONTH_END)
            daily, _ = query_interval(cursor, interval_end)
            events.extend(daily)
            cursor = interval_end

    deduped = {event.event_id: event for event in events}
    result = sorted(deduped.values(), key=lambda event: event.time_utc)
    metadata["downloaded_unique_events"] = len(result)
    if expected_count >= 0 and expected_count != len(result):
        notes.append(
            f"USGS count endpoint returned {expected_count:,}; {len(result):,} unique events remained after parsing and deduplication"
        )
    return result, notes, metadata


def load_earthquakes() -> Tuple[List[Earthquake], str, List[str], Dict[str, Any], Optional[Path]]:
    notes: List[str] = []
    cache_path = CACHE_ROOT / f"usgs_earthquakes_{MONTH_KEY}.geojson"

    if OFFLINE_MODE:
        notes.append("Offline mode requested with EARTHQUAKE_SHORT_OFFLINE=1")
        return make_procedural_fixture(), "synthetic_procedural_fixture", notes, {}, None

    try:
        events, live_notes, metadata = fetch_month_from_usgs()
        if not events:
            raise RuntimeError("USGS returned no parseable earthquake events")
        notes.extend(live_notes)
        cache_payload = {
            "type": "FeatureCollection",
            "metadata": {
                **metadata,
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "month_start": MONTH_START.isoformat(),
                "month_end": MONTH_END.isoformat(),
            },
            "features": [
                {
                    "type": "Feature",
                    "id": event.event_id,
                    "geometry": {
                        "type": "Point",
                        "coordinates": [event.longitude, event.latitude, event.depth_km],
                    },
                    "properties": {
                        "time": int(event.time_utc.timestamp() * 1000),
                        "mag": None if not np.isfinite(event.magnitude) else event.magnitude,
                        "place": event.place,
                        "sig": event.significance,
                        "felt": event.felt,
                        "tsunami": event.tsunami,
                        "status": event.status,
                        "url": event.url,
                    },
                }
                for event in events
            ],
        }
        cache_path.write_text(json.dumps(cache_payload), encoding="utf-8")
        return events, "usgs_comcat_fdsn", notes, metadata, cache_path
    except Exception as exc:
        notes.append(f"Live USGS query failed: {exc}")

    if cache_path.exists() and cache_path.stat().st_size > 200:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            events = parse_geojson(payload)
            if events:
                notes.append("Loaded cached USGS GeoJSON after live query failure")
                return events, "usgs_comcat_cached", notes, payload.get("metadata") or {}, cache_path
        except Exception as exc:
            notes.append(f"Cached USGS GeoJSON could not be parsed: {exc}")

    notes.append("Using deterministic procedural earthquake fixture")
    return make_procedural_fixture(), "synthetic_procedural_fixture", notes, {}, None


# -----------------------------------------------------------------------------
# Deterministic offline fixture
# -----------------------------------------------------------------------------

def interpolate_path(points: Sequence[Tuple[float, float]], u: float) -> Tuple[float, float]:
    u = clamp(u)
    if len(points) == 1:
        return points[0]
    position = u * (len(points) - 1)
    index = min(int(position), len(points) - 2)
    frac = position - index
    lon = lerp(points[index][0], points[index + 1][0], frac)
    lat = lerp(points[index][1], points[index + 1][1], frac)
    return lon, lat


def make_procedural_fixture() -> List[Earthquake]:
    rng = np.random.default_rng(MONTH_START.year * 100 + MONTH_START.month)
    count = 2200 if QUICK_MODE else 7200
    belts: List[Tuple[float, Sequence[Tuple[float, float]], float]] = [
        (0.23, [(-75, -55), (-72, -25), (-78, 0), (-84, 18), (-103, 25), (-125, 42), (-150, 58), (178, 52)], 3.0),
        (0.22, [(178, 52), (155, 45), (142, 35), (132, 22), (125, 8), (120, -10), (132, -25), (165, -45)], 3.2),
        (0.18, [(165, -45), (178, -30), (-178, -20), (-172, -10), (-178, 5), (170, 18), (160, 28)], 2.8),
        (0.10, [(-45, -55), (-30, -25), (-20, 0), (-28, 25), (-35, 50), (-20, 65)], 2.4),
        (0.10, [(-10, 35), (20, 38), (42, 38), (65, 34), (88, 29), (100, 25)], 2.1),
        (0.09, [(20, 38), (30, 20), (36, 5), (32, -15), (28, -35)], 2.0),
        (0.08, [(-125, 32), (-117, 35), (-111, 42), (-105, 46)], 1.5),
    ]
    probabilities = np.array([item[0] for item in belts], dtype=float)
    probabilities /= probabilities.sum()
    span_seconds = max((MONTH_END - MONTH_START).total_seconds(), 1.0)
    events: List[Earthquake] = []

    for index in range(count):
        belt_index = int(rng.choice(len(belts), p=probabilities))
        _, path, scatter = belts[belt_index]
        u = float(rng.random())
        lon, lat = interpolate_path(path, u)
        lon += float(rng.normal(0.0, scatter))
        lat += float(rng.normal(0.0, scatter * 0.65))
        lon = ((lon + 180.0) % 360.0) - 180.0
        lat = float(np.clip(lat, -65.0, 80.0))

        # Most events are small; large events are intentionally rare.
        magnitude = float(np.clip(rng.exponential(0.72) + 0.1, -0.5, 6.4))
        if rng.random() < 0.012:
            magnitude = float(rng.uniform(5.3, 6.8))
        if rng.random() < 0.0015:
            magnitude = float(rng.uniform(6.8, 7.6))

        if belt_index in {1, 2} and rng.random() < 0.20:
            depth = float(rng.uniform(300, 680))
        elif rng.random() < 0.18:
            depth = float(rng.uniform(70, 300))
        else:
            depth = float(np.clip(rng.exponential(24.0), 1.0, 70.0))

        seconds = float(rng.uniform(0.0, span_seconds))
        time_utc = MONTH_START + timedelta(seconds=seconds)
        events.append(
            Earthquake(
                event_id=f"fixture{MONTH_KEY}{index:05d}",
                time_utc=time_utc,
                latitude=lat,
                longitude=lon,
                depth_km=depth,
                magnitude=magnitude,
                place="Procedural preview event",
                significance=int(max(0, (magnitude + 0.5) ** 3 * 7)),
            )
        )

    # Guaranteed dramatic events for montage timing and layout validation.
    fixtures = [
        (7.4, 44.0, 149.0, 35.0, "Kuril Islands region"),
        (7.1, -20.5, -176.2, 515.0, "Tonga region"),
        (6.8, -24.2, -70.1, 42.0, "Northern Chile"),
        (6.6, 38.4, 142.3, 54.0, "Off the east coast of Honshu, Japan"),
    ]
    for offset, (mag, lat, lon, depth, place) in enumerate(fixtures):
        events.append(
            Earthquake(
                event_id=f"fixture_featured_{offset}",
                time_utc=MONTH_START + timedelta(seconds=span_seconds * (0.18 + offset * 0.19)),
                latitude=lat,
                longitude=lon,
                depth_km=depth,
                magnitude=mag,
                place=place,
                significance=900 - offset * 70,
            )
        )
    return sorted(events, key=lambda event: event.time_utc)


# -----------------------------------------------------------------------------
# Natural Earth land geometry
# -----------------------------------------------------------------------------

BUILTIN_LAND_POLYGONS: List[List[Tuple[float, float]]] = [
    [(-168, 72), (-140, 70), (-124, 55), (-126, 42), (-115, 30), (-101, 20), (-83, 8), (-77, 18), (-82, 25), (-80, 32), (-66, 46), (-52, 56), (-75, 72), (-168, 72)],
    [(-82, 12), (-70, 12), (-53, 4), (-35, -8), (-42, -25), (-58, -55), (-72, -50), (-80, -20), (-82, 12)],
    [(-17, 37), (5, 36), (34, 31), (50, 12), (42, -12), (33, -35), (18, -35), (6, -12), (-15, 12), (-17, 37)],
    [(-10, 72), (40, 72), (78, 66), (110, 56), (145, 50), (180, 64), (180, 8), (140, 6), (122, 22), (105, 5), (78, 8), (52, 28), (28, 38), (10, 45), (-10, 58), (-10, 72)],
    [(112, -10), (154, -12), (154, -39), (138, -45), (116, -35), (112, -10)],
    [(-74, 59), (-44, 83), (-18, 72), (-35, 59), (-74, 59)],
    [(-180, -62), (-120, -70), (-60, -65), (0, -72), (60, -66), (120, -72), (180, -62), (180, -90), (-180, -90), (-180, -62)],
    [(44, -12), (51, -14), (49, -26), (44, -25), (44, -12)],
    [(166, -34), (178, -38), (174, -47), (166, -46), (166, -34)],
    [(95, 5), (141, 6), (151, -10), (130, -12), (108, -7), (95, 5)],
]


def load_land_polygons() -> Tuple[List[List[Tuple[float, float]]], str, List[str]]:
    notes: List[str] = []
    if OFFLINE_MODE or shapefile is None:
        if shapefile is None:
            notes.append("pyshp is unavailable; using coarse built-in land polygons")
        return BUILTIN_LAND_POLYGONS, "built_in_coarse_land", notes

    zip_path = CACHE_ROOT / "ne_110m_land.zip"
    extract_dir = CACHE_ROOT / "ne_110m_land"
    shp_path = extract_dir / "ne_110m_land.shp"
    try:
        if not shp_path.exists():
            if not zip_path.exists() or zip_path.stat().st_size < 10_000:
                zip_path.write_bytes(request_bytes(NATURAL_EARTH_LAND_URL, timeout=45))
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_dir)
        reader = shapefile.Reader(str(shp_path))
        polygons: List[List[Tuple[float, float]]] = []
        for shape in reader.shapes():
            points = shape.points
            parts = list(shape.parts) + [len(points)]
            for start, end in zip(parts[:-1], parts[1:]):
                polygon = [(float(lon), float(lat)) for lon, lat in points[start:end]]
                if len(polygon) >= 3:
                    polygons.append(polygon)
        notes.append("Loaded Natural Earth 110m land polygons")
        return polygons, "natural_earth_110m_land", notes
    except Exception as exc:
        notes.append(f"Natural Earth fallback: {exc}")
        return BUILTIN_LAND_POLYGONS, "built_in_coarse_land", notes


# -----------------------------------------------------------------------------
# Summaries and saved products
# -----------------------------------------------------------------------------

def finite_magnitude(event: Earthquake) -> float:
    return event.magnitude if np.isfinite(event.magnitude) else -9.0


def choose_featured_events(events: Sequence[Earthquake], maximum: int = 4) -> List[Earthquake]:
    ranked = sorted(events, key=lambda event: (finite_magnitude(event), event.significance), reverse=True)
    selected: List[Earthquake] = []
    for event in ranked:
        if not np.isfinite(event.magnitude):
            continue
        if any(
            abs(event.latitude - chosen.latitude) < 7.0
            and abs((((event.longitude - chosen.longitude) + 180.0) % 360.0) - 180.0) < 12.0
            for chosen in selected
        ):
            continue
        selected.append(event)
        if len(selected) >= maximum:
            break
    if len(selected) < maximum:
        for event in ranked:
            if event not in selected and np.isfinite(event.magnitude):
                selected.append(event)
            if len(selected) >= maximum:
                break
    return selected


def earthquake_summary(
    events: Sequence[Earthquake],
    source: str,
    land_source: str,
    query_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    magnitudes = np.array([event.magnitude for event in events if np.isfinite(event.magnitude)], dtype=float)
    depths = np.array([event.depth_km for event in events], dtype=float)
    featured = choose_featured_events(events)
    days = max((MONTH_END - MONTH_START).total_seconds() / 86400.0, 1e-9)
    return {
        "title": CONFIG["title"],
        "target_month": MONTH_START.strftime("%Y-%m"),
        "month_label": MONTH_LABEL,
        "month_start_utc": MONTH_START.isoformat(),
        "month_end_utc": MONTH_END.isoformat(),
        "month_in_progress": MONTH_IS_IN_PROGRESS,
        "data_source": source,
        "land_source": land_source,
        "earthquake_count": int(len(events)),
        "average_events_per_elapsed_day": float(len(events) / days),
        "events_with_reported_magnitude": int(len(magnitudes)),
        "minimum_reported_magnitude": None if not len(magnitudes) else float(np.min(magnitudes)),
        "median_reported_magnitude": None if not len(magnitudes) else float(np.median(magnitudes)),
        "maximum_reported_magnitude": None if not len(magnitudes) else float(np.max(magnitudes)),
        "shallow_0_70_km": int(np.sum(depths < 70.0)),
        "intermediate_70_300_km": int(np.sum((depths >= 70.0) & (depths < 300.0))),
        "deep_300_plus_km": int(np.sum(depths >= 300.0)),
        "deepest_depth_km": None if not len(depths) else float(np.max(depths)),
        "featured_events": [
            {
                "event_id": event.event_id,
                "time_utc": event.time_utc.isoformat(),
                "magnitude": event.magnitude,
                "depth_km": event.depth_km,
                "latitude": event.latitude,
                "longitude": event.longitude,
                "place": event.place,
                "url": event.url,
            }
            for event in featured
        ],
        "usgs_query_metadata": query_metadata,
        "important_caveat": (
            "Every means every event returned by the selected USGS ComCat query. Catalog completeness varies "
            "with region and network sensitivity, and event parameters can be revised."
        ),
    }


def save_data_products(
    events: Sequence[Earthquake],
    summary: Dict[str, Any],
    notes: Sequence[str],
) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / f"usgs_earthquakes_{MONTH_KEY}.csv"
    summary_path = DATA_ROOT / f"earthquake_summary_{MONTH_KEY}.json"
    pd.DataFrame(
        [
            {
                "event_id": event.event_id,
                "time_utc": event.time_utc.isoformat(),
                "latitude": event.latitude,
                "longitude": event.longitude,
                "depth_km": event.depth_km,
                "magnitude": None if not np.isfinite(event.magnitude) else event.magnitude,
                "place": event.place,
                "significance": event.significance,
                "felt_reports": event.felt,
                "tsunami_flag": event.tsunami,
                "status": event.status,
                "event_url": event.url,
            }
            for event in events
        ]
    ).to_csv(csv_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "notes": list(notes),
                "source_urls": {
                    "usgs_event_api": USGS_QUERY_URL,
                    "usgs_geojson_format": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php",
                    "usgs_comcat": "https://earthquake.usgs.gov/data/comcat/",
                    "usgs_depths": "https://www.usgs.gov/programs/earthquake-hazards/determining-depth-earthquake",
                    "natural_earth_land": NATURAL_EARTH_LAND_URL,
                },
                "fallback_warning": (
                    "synthetic_procedural_fixture is deterministic preview data, not observational data"
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return csv_path, summary_path


# -----------------------------------------------------------------------------
# Visual helpers
# -----------------------------------------------------------------------------

def depth_color(depth_km: float, alpha: int = 230) -> Tuple[int, int, int, int]:
    if depth_km >= 300.0:
        return COLORS["deep"] + (alpha,)
    if depth_km >= 70.0:
        return COLORS["intermediate"] + (alpha,)
    return COLORS["shallow"] + (alpha,)


def marker_radius(magnitude: float) -> float:
    scale = OUT_W / 1080.0
    if not np.isfinite(magnitude):
        return 1.2 * scale
    positive = max(magnitude, 0.0)
    return max(1.15 * scale, (1.25 + positive**1.48 * 1.18) * scale)


def unwrapped_relative_longitudes(polygon: Sequence[Tuple[float, float]], center_lon: float) -> List[float]:
    values: List[float] = []
    previous: Optional[float] = None
    for lon, _ in polygon:
        value = ((lon - center_lon + 180.0) % 360.0) - 180.0
        if previous is not None:
            while value - previous > 180.0:
                value -= 360.0
            while value - previous < -180.0:
                value += 360.0
        values.append(value)
        previous = value
    return values


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class EarthquakeScene:
    def __init__(
        self,
        events: List[Earthquake],
        land_polygons: List[List[Tuple[float, float]]],
        summary: Dict[str, Any],
    ):
        self.events = sorted(events, key=lambda event: event.time_utc)
        self.land_polygons = land_polygons
        self.summary = summary
        self.featured = choose_featured_events(self.events, maximum=4)
        self.deepest = max(self.events, key=lambda event: event.depth_km)
        self.particles = self._make_particles(int(CONFIG["background_particles"]), seed=44)
        self.dust = self._make_particles(int(CONFIG["dust_particles"]), seed=91)
        self.center_keys: Dict[str, float] = {
            "world": 0.0,
            "pacific": 160.0,
            "deep": self.deepest.longitude,
        }
        for index, event in enumerate(self.featured):
            self.center_keys[f"featured_{index}"] = event.longitude

        self.event_fractions = np.array([event.time_fraction for event in self.events], dtype=float)
        self.base_maps = {key: self._render_static_map(center) for key, center in self.center_keys.items()}
        self.all_layers = {
            key: self._make_event_layer(self.events, self.center_keys[key], alpha_scale=0.92)
            for key in ("world", "pacific", "deep")
        }
        self.timeline_layers = self._build_timeline_layers()

    @staticmethod
    def _make_particles(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.4, 1.8 if QUICK_MODE else 2.5)),
                "a": float(rng.uniform(12, 75)),
                "phase": float(rng.uniform(0, math.tau)),
                "speed": float(rng.uniform(2.0, 14.0)),
            }
            for _ in range(count)
        ]

    def project(self, lon: float, lat: float, center_lon: float = 0.0) -> Tuple[float, float]:
        relative = ((lon - center_lon + 180.0) % 360.0) - 180.0
        x0 = float(CONFIG["map_margin_x"])
        x1 = OUT_W - float(CONFIG["map_margin_x"])
        y0 = float(CONFIG["map_top"])
        y1 = float(CONFIG["map_bottom"])
        x = x0 + (relative + 180.0) / 360.0 * (x1 - x0)
        lat_clamped = float(np.clip(lat, CONFIG["map_lat_min"], CONFIG["map_lat_max"]))
        y = y0 + (float(CONFIG["map_lat_max"]) - lat_clamped) / (
            float(CONFIG["map_lat_max"]) - float(CONFIG["map_lat_min"])
        ) * (y1 - y0)
        return x, y

    def _render_static_map(self, center_lon: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, COLORS["ocean_bottom"] + (255,))
        array = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
        top = np.array(COLORS["ocean_top"], dtype=float)
        bottom = np.array(COLORS["ocean_bottom"], dtype=float)
        for y in range(OUT_H):
            u = y / max(OUT_H - 1, 1)
            rgb = (top * (1.0 - u) + bottom * u).astype(np.uint8)
            array[y, :, :3] = rgb
            array[y, :, 3] = 255
        image = Image.fromarray(array, "RGBA")
        draw = ImageDraw.Draw(image)
        x0 = int(CONFIG["map_margin_x"])
        x1 = OUT_W - x0
        y0 = int(CONFIG["map_top"])
        y1 = int(CONFIG["map_bottom"])

        # Latitude and longitude grid.
        for lat in range(-60, 76, 15):
            _, y = self.project(center_lon, float(lat), center_lon)
            draw.line((x0, y, x1, y), fill=COLORS["grid"] + (27,), width=1)
        for relative_lon in range(-150, 181, 30):
            lon = center_lon + relative_lon
            x, _ = self.project(lon, 0.0, center_lon)
            draw.line((x, y0, x, y1), fill=COLORS["grid"] + (23,), width=1)

        # Land polygons are unwrapped and drawn in three shifted copies so the
        # selected map seam does not create giant triangles.
        for polygon in self.land_polygons:
            relative_lons = unwrapped_relative_longitudes(polygon, center_lon)
            lats = [lat for _, lat in polygon]
            for shift in (-360.0, 0.0, 360.0):
                points: List[Tuple[float, float]] = []
                for relative, lat in zip(relative_lons, lats):
                    x = x0 + (relative + shift + 180.0) / 360.0 * (x1 - x0)
                    lat_clamped = float(np.clip(lat, CONFIG["map_lat_min"], CONFIG["map_lat_max"]))
                    y = y0 + (float(CONFIG["map_lat_max"]) - lat_clamped) / (
                        float(CONFIG["map_lat_max"]) - float(CONFIG["map_lat_min"])
                    ) * (y1 - y0)
                    points.append((x, y))
                if points and max(p[0] for p in points) >= x0 - 50 and min(p[0] for p in points) <= x1 + 50:
                    draw.polygon(points, fill=COLORS["land"] + (255,), outline=COLORS["land_edge"] + (118,))

        # Equator and frame.
        _, equator_y = self.project(center_lon, 0.0, center_lon)
        draw.line((x0, equator_y, x1, equator_y), fill=COLORS["grid"] + (55,), width=1)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=14 if QUICK_MODE else 28, outline=COLORS["grid"] + (70,), width=1)

        # Soft ocean atmosphere.
        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for cx, cy, color in [
            (OUT_W * 0.23, OUT_H * 0.38, COLORS["shallow"]),
            (OUT_W * 0.78, OUT_H * 0.52, COLORS["deep"]),
        ]:
            for radius, alpha in [
                (OUT_W * 0.40, 8),
                (OUT_W * 0.25, 12),
                (OUT_W * 0.13, 18),
            ]:
                hd.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color + (alpha,))
        haze = haze.filter(ImageFilter.GaussianBlur(55 if not QUICK_MODE else 28))
        image.alpha_composite(haze)
        return image

    def _draw_event_core(
        self,
        draw: ImageDraw.ImageDraw,
        event: Earthquake,
        center_lon: float,
        alpha_scale: float = 1.0,
        radius_scale: float = 1.0,
    ) -> Tuple[float, float, float]:
        x, y = self.project(event.longitude, event.latitude, center_lon)
        radius = marker_radius(event.magnitude) * radius_scale
        color = depth_color(event.depth_km, int(220 * clamp(alpha_scale)))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        return x, y, radius

    def _make_event_layer(
        self,
        events: Iterable[Earthquake],
        center_lon: float,
        alpha_scale: float = 1.0,
    ) -> Image.Image:
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        core = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        cd = ImageDraw.Draw(core)
        for event in events:
            x, y = self.project(event.longitude, event.latitude, center_lon)
            radius = marker_radius(event.magnitude)
            base_alpha = int(150 * clamp(alpha_scale))
            color = depth_color(event.depth_km, base_alpha)
            glow_radius = max(radius * 2.2, 2.0 * OUT_W / 1080.0)
            gd.ellipse(
                (x - glow_radius, y - glow_radius, x + glow_radius, y + glow_radius),
                fill=color,
            )
            core_color = depth_color(event.depth_km, int(220 * clamp(alpha_scale)))
            cd.ellipse((x - radius, y - radius, x + radius, y + radius), fill=core_color)
        glow = glow.filter(ImageFilter.GaussianBlur(5 if QUICK_MODE else 10))
        glow.alpha_composite(core)
        return glow

    def _build_timeline_layers(self) -> List[Image.Image]:
        buckets = int(CONFIG["timeline_buckets"])
        layers: List[Image.Image] = []
        for bucket in tqdm(range(1, buckets + 1), desc="Building timeline layers", leave=False):
            fraction = bucket / buckets
            index = int(np.searchsorted(self.event_fractions, fraction, side="right"))
            layers.append(self._make_event_layer(self.events[:index], 0.0, alpha_scale=0.90))
        return layers

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, COLORS["dark"] + (255,))
        draw = ImageDraw.Draw(image)
        for particle in self.particles:
            x = (particle["x"] + math.sin(t * 0.17 + particle["phase"]) * 10.0) % OUT_W
            y = (particle["y"] + t * particle["speed"] * 0.12) % OUT_H
            alpha = int(particle["a"] * (0.55 + 0.45 * math.sin(t * 0.8 + particle["phase"]) ** 2))
            radius = particle["r"]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(205, 231, 241, alpha))
        return image

    def compose_map(self, key: str, event_layer: Optional[Image.Image] = None) -> Image.Image:
        layer = self.base_maps[key].copy()
        if event_layer is not None:
            layer.alpha_composite(event_layer)
        return layer

    def draw_pulse(
        self,
        image: Image.Image,
        event: Earthquake,
        center_lon: float,
        age: float,
        strength: float = 1.0,
        label: bool = False,
    ):
        age = clamp(age)
        x, y = self.project(event.longitude, event.latitude, center_lon)
        base_radius = marker_radius(event.magnitude)
        color = depth_color(event.depth_km, 255)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        flash = (1.0 - age) ** 2 * strength
        glow_radius = base_radius * (2.0 + age * 5.0)
        draw.ellipse(
            (x - glow_radius, y - glow_radius, x + glow_radius, y + glow_radius),
            fill=color[:3] + (int(105 * flash),),
        )
        ring_radius = base_radius * (1.6 + age * 7.0)
        draw.ellipse(
            (x - ring_radius, y - ring_radius, x + ring_radius, y + ring_radius),
            outline=color[:3] + (int(225 * (1.0 - age) * strength),),
            width=max(1, int(2 * OUT_W / 1080.0)),
        )
        core_radius = base_radius * (1.0 + flash * 0.8)
        draw.ellipse(
            (x - core_radius, y - core_radius, x + core_radius, y + core_radius),
            fill=color[:3] + (int(245 * strength),),
        )
        overlay = overlay.filter(ImageFilter.GaussianBlur(1 if QUICK_MODE else 2))
        image.alpha_composite(overlay)

        if label:
            draw_text(
                image,
                event.magnitude_label,
                (int(x + base_radius + 8), int(y - base_radius - 4)),
                size=18 if not QUICK_MODE else 9,
                fill=COLORS["white"] + (245,),
                bold=True,
                stroke=1,
            )

    def draw_awaken(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        progress = shot_progress(t, shot)
        map_layer = self.compose_map("world")
        map_layer = zoom_and_shift(map_layer, 1.03 + 0.025 * progress, 0.0, -8.0 * OUT_W / 1080.0)
        alpha_composite_with_opacity(image, map_layer, smoothstep(progress * 1.8))

        sample_count = min(38 if not QUICK_MODE else 16, len(self.events))
        if sample_count:
            indices = np.linspace(0, len(self.events) - 1, sample_count).astype(int)
            for order, index in enumerate(indices):
                trigger = order / max(sample_count - 1, 1) * 0.72 + 0.10
                local_age = (progress - trigger) / 0.24
                if 0.0 <= local_age <= 1.0:
                    self.draw_pulse(image, self.events[index], 0.0, local_age, strength=0.9)
                elif local_age > 1.0:
                    overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
                    self._draw_event_core(ImageDraw.Draw(overlay), self.events[index], 0.0, alpha_scale=0.75)
                    image.alpha_composite(overlay)

        y = int(OUT_H * 0.71)
        points = []
        for index in range(180):
            u = index / 179.0
            amplitude = (10 if QUICK_MODE else 20) * (0.25 + 0.75 * smoothstep(progress))
            wave = math.sin(u * math.tau * 4.0 + t * 3.8) + 0.32 * math.sin(u * math.tau * 11.0 - t * 1.6)
            points.append((lerp(OUT_W * 0.10, OUT_W * 0.90, u), y + wave * amplitude))
        ImageDraw.Draw(image).line(points, fill=COLORS["shallow"] + (205,), width=2 if QUICK_MODE else 3)
        draw_text(
            image,
            "THE PLANET NEVER STOPS MOVING",
            (OUT_W // 2, int(OUT_H * 0.77)),
            size=16 if QUICK_MODE else 32,
            fill=COLORS["white"] + (235,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_month_sweep(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        progress = smootherstep(shot_progress(t, shot))
        bucket_count = len(self.timeline_layers)
        bucket = min(int(progress * bucket_count), bucket_count - 1)
        timeline = self.timeline_layers[bucket]
        map_layer = self.compose_map("world", timeline)
        camera_zoom = 1.025 + 0.035 * math.sin(progress * math.pi)
        map_layer = zoom_and_shift(map_layer, camera_zoom, math.sin(t * 0.20) * 7.0, -4.0)
        image.alpha_composite(map_layer)

        visible_count = int(np.searchsorted(self.event_fractions, progress, side="right"))
        current_time = MONTH_START + (MONTH_END - MONTH_START) * progress
        window = 0.018 if QUICK_MODE else 0.009
        lo = max(0, int(np.searchsorted(self.event_fractions, progress - window, side="left")))
        hi = min(len(self.events), int(np.searchsorted(self.event_fractions, progress + 0.003, side="right")))
        active = self.events[lo:hi]
        step = max(1, len(active) // (18 if QUICK_MODE else 45))
        for event in active[::step]:
            age = clamp((progress - event.time_fraction) / max(window, 1e-6))
            if progress >= event.time_fraction:
                self.draw_pulse(image, event, 0.0, age, strength=0.8)

        # Cinematic date and count HUD, not a chart.
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        left = 28 if QUICK_MODE else 56
        top = 650 if QUICK_MODE else 1290
        right = OUT_W - left
        bottom = top + (86 if QUICK_MODE else 172)
        pd.rounded_rectangle((left, top, right, bottom), radius=14 if QUICK_MODE else 28, fill=(2, 7, 14, 174), outline=COLORS["grid"] + (68,), width=1)
        image.alpha_composite(panel)
        draw_text(
            image,
            current_time.strftime("%d %B %Y").upper(),
            (left + (16 if QUICK_MODE else 30), top + (15 if QUICK_MODE else 28)),
            size=13 if QUICK_MODE else 26,
            fill=COLORS["muted"] + (230,),
            bold=True,
            condensed=True,
            stroke=1,
        )
        draw_text(
            image,
            f"{visible_count:,}",
            (right - (16 if QUICK_MODE else 30), top + (12 if QUICK_MODE else 22)),
            size=30 if QUICK_MODE else 60,
            fill=COLORS["white"] + (250,),
            bold=True,
            condensed=True,
            anchor="ra",
            stroke=1,
        )
        draw_text(
            image,
            "EARTHQUAKES SO FAR",
            (right - (16 if QUICK_MODE else 30), top + (57 if QUICK_MODE else 113)),
            size=10 if QUICK_MODE else 20,
            fill=COLORS["shallow"] + (230,),
            bold=True,
            condensed=True,
            anchor="ra",
            stroke=1,
        )

    def draw_pacific(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        progress = smoothstep(shot_progress(t, shot))
        map_layer = self.compose_map("pacific", self.all_layers["pacific"])
        map_layer = zoom_and_shift(
            map_layer,
            1.12 + 0.08 * math.sin(progress * math.pi),
            math.sin(progress * math.pi * 1.3) * 18.0 * OUT_W / 1080.0,
            -18.0 * OUT_W / 1080.0,
        )
        alpha_composite_with_opacity(image, map_layer, 0.58 + 0.42 * progress)

        # Trace an understated Pacific frame; it is a cinematic annotation, not
        # a plate-boundary data layer.
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        ring_points: List[Tuple[float, float]] = []
        path = [
            (160, 55), (145, 38), (128, 10), (150, -28), (178, -43),
            (-175, -22), (-165, 5), (-150, 50), (-125, 43), (-98, 20),
            (-78, -5), (-72, -38),
        ]
        for lon, lat in path:
            ring_points.append(self.project(lon, lat, 160.0))
        reveal = max(2, int(len(ring_points) * progress))
        if reveal > 1:
            draw.line(ring_points[:reveal], fill=COLORS["red"] + (105,), width=2 if QUICK_MODE else 4)
        overlay = overlay.filter(ImageFilter.GaussianBlur(1 if QUICK_MODE else 2))
        image.alpha_composite(overlay)

        draw_text(
            image,
            "PACIFIC MARGINS",
            (OUT_W // 2, int(OUT_H * 0.71)),
            size=23 if QUICK_MODE else 47,
            fill=COLORS["white"] + (245,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            "A DENSE ARC OF SEISMIC ACTIVITY",
            (OUT_W // 2, int(OUT_H * 0.755)),
            size=11 if QUICK_MODE else 22,
            fill=COLORS["red"] + (230,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_depth(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        progress = smoothstep(shot_progress(t, shot))
        map_layer = self.compose_map("deep", self.all_layers["deep"])
        x, y = self.project(self.deepest.longitude, self.deepest.latitude, self.deepest.longitude)
        zoom = 1.12 + 0.16 * progress
        dx = (OUT_W / 2.0 - x) * zoom * 0.30
        dy = (OUT_H * 0.46 - y) * zoom * 0.24
        map_layer = zoom_and_shift(map_layer, zoom, dx, dy)
        alpha_composite_with_opacity(image, map_layer, 0.82)

        # A stylized vertical cutaway replaces a conventional depth graph.
        cutaway = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(cutaway)
        shaft_x = int(OUT_W * 0.80)
        surface_y = int(OUT_H * 0.28)
        bottom_y = int(OUT_H * 0.72)
        draw.line((shaft_x, surface_y, shaft_x, bottom_y), fill=COLORS["white"] + (115,), width=2 if QUICK_MODE else 4)
        zones = [
            (0, 70, COLORS["shallow"], "0—70 KM  SHALLOW"),
            (70, 300, COLORS["intermediate"], "70—300 KM  INTERMEDIATE"),
            (300, 700, COLORS["deep"], "300—700 KM  DEEP"),
        ]
        for low, high, color, label in zones:
            y_start = lerp(surface_y, bottom_y, low / 700.0)
            y_end = lerp(surface_y, bottom_y, high / 700.0)
            draw.rectangle((shaft_x - (5 if QUICK_MODE else 9), y_start, shaft_x + (5 if QUICK_MODE else 9), y_end), fill=color + (130,))
            draw_text(
                cutaway,
                label,
                (shaft_x - (14 if QUICK_MODE else 25), int((y_start + y_end) / 2)),
                size=8 if QUICK_MODE else 16,
                fill=color + (235,),
                bold=True,
                condensed=True,
                anchor="rm",
                stroke=1,
            )
        deepest_y = lerp(surface_y, bottom_y, clamp(self.deepest.depth_km / 700.0))
        pulse = 0.5 + 0.5 * math.sin(t * 5.0)
        rr = (6 if QUICK_MODE else 12) * (1.0 + 0.3 * pulse)
        draw.ellipse((shaft_x - rr, deepest_y - rr, shaft_x + rr, deepest_y + rr), fill=depth_color(self.deepest.depth_km, 245))
        cutaway = cutaway.filter(ImageFilter.GaussianBlur(0.6 if QUICK_MODE else 1.2))
        image.alpha_composite(cutaway)

        draw_text(
            image,
            "DEPTH CHANGES THE COLOR",
            (OUT_W * 0.08, int(OUT_H * 0.72)),
            size=18 if QUICK_MODE else 36,
            fill=COLORS["white"] + (245,),
            bold=True,
            condensed=True,
            stroke=1,
        )
        draw_text(
            image,
            f"DEEPEST THIS MONTH // {self.deepest.depth_km:.0f} KM",
            (OUT_W * 0.08, int(OUT_H * 0.758)),
            size=10 if QUICK_MODE else 20,
            fill=COLORS["deep"] + (235,),
            bold=True,
            condensed=True,
            stroke=1,
        )

    def draw_largest(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        progress = shot_progress(t, shot)
        if not self.featured:
            self.draw_all_events(image, t, shot)
            return
        segment = progress * len(self.featured)
        index = min(int(segment), len(self.featured) - 1)
        local = segment - index
        event = self.featured[index]
        key = f"featured_{index}"
        center = self.center_keys[key]
        map_layer = self.compose_map(key)
        x, y = self.project(event.longitude, event.latitude, center)
        zoom = 1.50 + 0.28 * smoothstep(local)
        dx = (OUT_W / 2.0 - x) * zoom
        dy = (OUT_H * 0.43 - y) * zoom
        map_layer = zoom_and_shift(map_layer, zoom, dx, dy)
        image.alpha_composite(map_layer)

        # Nearby background events create regional context.
        nearby = [
            candidate
            for candidate in self.events
            if abs(candidate.latitude - event.latitude) < 22.0
            and abs((((candidate.longitude - event.longitude) + 180.0) % 360.0) - 180.0) < 34.0
        ]
        nearby_layer = self._make_event_layer(nearby, center, alpha_scale=0.55)
        nearby_layer = zoom_and_shift(nearby_layer, zoom, dx, dy)
        image.alpha_composite(nearby_layer)

        # Draw focal pulse after camera transform by placing it at the intended
        # screen center, making the shot feel like an impact rather than a pin.
        pulse_layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(pulse_layer)
        px, py = OUT_W / 2.0, OUT_H * 0.43
        color = depth_color(event.depth_km, 255)
        base = marker_radius(event.magnitude) * 2.0
        pulse_age = (local * 2.2) % 1.0
        for multiplier, alpha in [(2.0 + pulse_age * 5.0, 170), (1.2 + pulse_age * 3.0, 230)]:
            rr = base * multiplier
            draw.ellipse((px - rr, py - rr, px + rr, py + rr), outline=color[:3] + (int(alpha * (1.0 - pulse_age)),), width=2 if QUICK_MODE else 4)
        draw.ellipse((px - base, py - base, px + base, py + base), fill=color[:3] + (245,))
        pulse_layer = pulse_layer.filter(ImageFilter.GaussianBlur(1 if QUICK_MODE else 2))
        image.alpha_composite(pulse_layer)

        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        box = (26 if QUICK_MODE else 52, int(OUT_H * 0.63), OUT_W - (26 if QUICK_MODE else 52), int(OUT_H * 0.79))
        pd.rounded_rectangle(box, radius=15 if QUICK_MODE else 30, fill=(2, 7, 14, 188), outline=color[:3] + (90,), width=1)
        image.alpha_composite(panel)
        draw_text(
            image,
            event.magnitude_label,
            (box[0] + (18 if QUICK_MODE else 34), box[1] + (16 if QUICK_MODE else 28)),
            size=31 if QUICK_MODE else 64,
            fill=COLORS["white"] + (250,),
            bold=True,
            condensed=True,
            stroke=1,
        )
        draw_text(
            image,
            event.display_place.upper(),
            (box[0] + (18 if QUICK_MODE else 34), box[1] + (56 if QUICK_MODE else 112)),
            size=11 if QUICK_MODE else 22,
            fill=color[:3] + (240,),
            bold=True,
            condensed=True,
            stroke=1,
        )
        draw_text(
            image,
            f"{event.time_utc.strftime('%d %b %H:%M UTC').upper()}  //  DEPTH {event.depth_km:.0f} KM",
            (box[0] + (18 if QUICK_MODE else 34), box[1] + (78 if QUICK_MODE else 153)),
            size=8 if QUICK_MODE else 16,
            fill=COLORS["muted"] + (220,),
            bold=True,
            condensed=True,
            stroke=1,
        )

    def draw_all_events(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        progress = smoothstep(shot_progress(t, shot))
        map_layer = self.compose_map("world", self.all_layers["world"])
        map_layer = zoom_and_shift(map_layer, 1.08 - 0.04 * progress, 0.0, -8.0 * progress)
        alpha_composite_with_opacity(image, map_layer, 0.80 + 0.20 * progress)
        draw_text(
            image,
            f"{len(self.events):,}",
            (OUT_W // 2, int(OUT_H * 0.67)),
            size=70 if QUICK_MODE else 142,
            fill=COLORS["white"] + (255,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=2,
        )
        draw_text(
            image,
            "CATALOGUED EARTHQUAKES",
            (OUT_W // 2, int(OUT_H * 0.75)),
            size=17 if QUICK_MODE else 34,
            fill=COLORS["shallow"] + (240,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            MONTH_LABEL,
            (OUT_W // 2, int(OUT_H * 0.79)),
            size=11 if QUICK_MODE else 22,
            fill=COLORS["muted"] + (225,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_top_titles(self, image: Image.Image, t: float, shot_name: str):
        title_alpha = int(255 * smoothstep((t - 0.15) / 0.8) * (1.0 - smoothstep((t - (6.8 if not QUICK_MODE else 1.45)) / 0.7)))
        if title_alpha > 4:
            x = 28 if QUICK_MODE else 56
            draw_text(
                image,
                "EVERY EARTHQUAKE",
                (x, 38 if QUICK_MODE else 76),
                size=22 if QUICK_MODE else 46,
                fill=COLORS["white"] + (title_alpha,),
                bold=True,
                condensed=True,
                stroke=1,
            )
            draw_text(
                image,
                "THIS MONTH",
                (x, 65 if QUICK_MODE else 130),
                size=22 if QUICK_MODE else 46,
                fill=COLORS["white"] + (title_alpha,),
                bold=True,
                condensed=True,
                stroke=1,
            )
            draw_text(
                image,
                CONFIG["subtitle"],
                (x + 1, 92 if QUICK_MODE else 186),
                size=10 if QUICK_MODE else 20,
                fill=COLORS["shallow"] + (min(title_alpha, 235),),
                bold=True,
                condensed=True,
                stroke=1,
            )

        labels = {
            "awakening": "GLOBAL SEISMICITY // THE MONTH BEGINS",
            "month_sweep": "CHRONOLOGICAL SWEEP // EVERY CATALOGUED EVENT",
            "pacific_margin": "PACIFIC VIEW // CONCENTRATED PLATE-MARGIN ACTIVITY",
            "depth": "BELOW THE SURFACE // HYPOCENTRAL DEPTH",
            "largest": "THE MONTH'S LARGEST REPORTED MAGNITUDES",
            "all_events": "THE COMPLETE MONTHLY QUERY",
        }
        if t > (5.4 if not QUICK_MODE else 1.1):
            draw_text(
                image,
                labels[shot_name],
                (28 if QUICK_MODE else 56, 31 if QUICK_MODE else 62),
                size=9 if QUICK_MODE else 19,
                fill=COLORS["muted"] + (215,),
                bold=True,
                condensed=True,
                stroke=1,
            )

    def draw_source_hud(self, image: Image.Image):
        live = self.summary["data_source"] in {"usgs_comcat_fdsn", "usgs_comcat_cached"}
        source_text = "SOURCE // USGS COMCAT" if live else "PREVIEW SOURCE // SYNTHETIC FIXTURE"
        source_color = COLORS["shallow"] if live else COLORS["intermediate"]
        x = OUT_W - (26 if QUICK_MODE else 52)
        draw_text(
            image,
            source_text,
            (x, 37 if QUICK_MODE else 74),
            size=9 if QUICK_MODE else 18,
            fill=source_color + (235,),
            bold=True,
            condensed=True,
            anchor="ra",
            stroke=1,
        )
        draw_text(
            image,
            f"EVENTS // {len(self.events):,}",
            (x, 52 if QUICK_MODE else 104),
            size=8 if QUICK_MODE else 16,
            fill=COLORS["muted"] + (210,),
            bold=True,
            condensed=True,
            anchor="ra",
            stroke=1,
        )
        state = "MONTH TO DATE" if MONTH_IS_IN_PROGRESS else "COMPLETED MONTH"
        draw_text(
            image,
            state,
            (x, 66 if QUICK_MODE else 132),
            size=8 if QUICK_MODE else 16,
            fill=COLORS["muted"] + (195,),
            bold=True,
            condensed=True,
            anchor="ra",
            stroke=1,
        )

    def draw_depth_legend(self, image: Image.Image, shot_name: str):
        if shot_name in {"awakening", "largest"}:
            return
        x0 = 28 if QUICK_MODE else 56
        y = OUT_H - (176 if QUICK_MODE else 352)
        items = [
            (COLORS["shallow"], "SHALLOW"),
            (COLORS["intermediate"], "INTERMEDIATE"),
            (COLORS["deep"], "DEEP"),
        ]
        for index, (color, label) in enumerate(items):
            x = x0 + index * (100 if QUICK_MODE else 200)
            rr = 4 if QUICK_MODE else 8
            ImageDraw.Draw(image).ellipse((x - rr, y - rr, x + rr, y + rr), fill=color + (235,))
            draw_text(
                image,
                label,
                (x + (9 if QUICK_MODE else 17), y),
                size=8 if QUICK_MODE else 15,
                fill=COLORS["muted"] + (210,),
                bold=True,
                condensed=True,
                anchor="lm",
                stroke=1,
            )

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        y0 = OUT_H - (132 if QUICK_MODE else 264)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        margin = 22 if QUICK_MODE else 44
        draw.rounded_rectangle(
            (margin, y0, OUT_W - margin, y0 + (82 if QUICK_MODE else 164)),
            radius=14 if QUICK_MODE else 28,
            fill=(2, 6, 13, 180),
            outline=COLORS["grid"] + (62,),
            width=1,
        )
        image.alpha_composite(panel)
        draw_wrapped_text(
            image,
            text,
            (margin + (14 if QUICK_MODE else 28), y0 + (15 if QUICK_MODE else 29)),
            OUT_W - 2 * margin - (28 if QUICK_MODE else 56),
            size=13 if QUICK_MODE else 27,
            fill=COLORS["white"] + (246,),
            line_spacing=3 if QUICK_MODE else 6,
        )

    def draw_film_texture(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for particle in self.dust:
            pulse = 0.5 + 0.5 * math.sin(t * 1.4 + particle["phase"])
            if pulse < 0.58:
                continue
            x = (particle["x"] + t * particle["speed"] * 0.45) % OUT_W
            y = (particle["y"] + math.sin(t * 0.7 + particle["phase"]) * 5.0) % OUT_H
            length = (6 if QUICK_MODE else 12) + particle["r"] * 5
            draw.line((x, y, x + length, y), fill=COLORS["shallow"] + (int(18 * pulse),), width=1)
        offset = int((t * 41) % 7)
        for y in range(offset, OUT_H, 7):
            draw.line((0, y, OUT_W, y), fill=(125, 190, 210, 9), width=1)
        scan_y = int((t * 130) % (OUT_H + 180)) - 90
        draw.rectangle((0, scan_y, OUT_W, scan_y + (34 if QUICK_MODE else 68)), fill=(85, 220, 240, 6))
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        name = str(shot["name"])
        image = self.background(t)

        if name == "awakening":
            self.draw_awaken(image, t, shot)
        elif name == "month_sweep":
            self.draw_month_sweep(image, t, shot)
        elif name == "pacific_margin":
            self.draw_pacific(image, t, shot)
        elif name == "depth":
            self.draw_depth(image, t, shot)
        elif name == "largest":
            self.draw_largest(image, t, shot)
        else:
            self.draw_all_events(image, t, shot)

        self.draw_top_titles(image, t, name)
        self.draw_source_hud(image)
        self.draw_depth_legend(image, name)
        self.draw_caption(image, t)
        self.draw_film_texture(image, t)

        array = np.asarray(image.convert("RGB"))
        array = apply_grade(array)
        rng = np.random.default_rng(int(t * int(CONFIG["fps"])) + 613)
        grain = rng.normal(0.0, float(CONFIG["grain_strength"]), size=array.shape[:2]).astype(np.float32)
        array = np.clip(array.astype(np.float32) + grain[..., None], 0, 255)
        array *= VIGNETTE[..., None]
        fade_in = smoothstep(t / 0.85)
        fade_out = 1.0 - smoothstep((t - (float(CONFIG["duration_s"]) - 1.05)) / 0.95)
        return np.clip(array * fade_in * fade_out, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Soundtrack and rendering
# -----------------------------------------------------------------------------

def gaussian_envelope(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((times - center) / max(width, 1e-6)) ** 2)


def generate_ambient_soundtrack(path: Path, featured: Sequence[Earthquake]) -> Path:
    sample_rate = int(CONFIG["soundtrack_sample_rate"])
    duration = float(CONFIG["duration_s"])
    count = int(round(sample_rate * duration))
    times = np.arange(count, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(MONTH_START.year * 100 + MONTH_START.month + 700)

    audio = np.zeros(count, dtype=np.float64)
    audio += 0.10 * np.sin(math.tau * 31.0 * times + 0.45 * np.sin(math.tau * 0.07 * times))
    audio += 0.055 * np.sin(math.tau * 46.0 * times + 1.2)
    audio += 0.025 * np.sin(math.tau * 73.0 * times + 0.5 * np.sin(math.tau * 0.12 * times))

    control_count = max(8, int(duration * 5))
    controls = rng.normal(0.0, 1.0, control_count)
    slow_noise = np.interp(times, np.linspace(0.0, duration, control_count), controls)
    audio += 0.026 * slow_noise

    # Chronological micro-impacts across the month sweep.
    sweep_shot = next(shot for shot in SHOT_PLAN if shot["name"] == "month_sweep")
    for fraction in np.linspace(0.05, 0.95, 22 if not QUICK_MODE else 9):
        center = lerp(float(sweep_shot["start"]), float(sweep_shot["end"]), float(fraction))
        env = gaussian_envelope(times, center, 0.045 if QUICK_MODE else 0.075)
        frequency = float(rng.uniform(95.0, 220.0))
        audio += env * (0.018 * np.sin(math.tau * frequency * times))

    # Deep sub-bass impacts for featured events.
    largest_shot = next(shot for shot in SHOT_PLAN if shot["name"] == "largest")
    for index, event in enumerate(featured):
        center = lerp(
            float(largest_shot["start"]),
            float(largest_shot["end"]),
            (index + 0.18) / max(len(featured), 1),
        )
        local = np.maximum(times - center, 0.0)
        envelope = np.exp(-local * (4.5 if QUICK_MODE else 3.2)) * (times >= center)
        magnitude = max(event.magnitude, 0.0) if np.isfinite(event.magnitude) else 2.0
        frequency = 28.0 + (7.6 - min(magnitude, 7.6)) * 2.2
        audio += envelope * (0.12 + 0.012 * magnitude) * np.sin(math.tau * frequency * local)
        audio += envelope * 0.030 * rng.normal(0.0, 1.0, count)

    # Intro and final swells.
    intro_x = np.clip(times / max(1.8, duration * 0.09), 0.0, 1.0)
    outro_x = np.clip((times - (duration - 1.4)) / 1.2, 0.0, 1.0)
    intro = intro_x * intro_x * (3.0 - 2.0 * intro_x)
    outro = 1.0 - outro_x * outro_x * (3.0 - 2.0 * outro_x)
    audio *= intro * outro
    peak = max(float(np.max(np.abs(audio))), 1e-9)
    audio = np.clip(audio / peak * 0.88, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


def find_ffmpeg() -> Optional[str]:
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return shutil.which("ffmpeg")


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    command = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def render_video(scene: EarthquakeScene) -> Path:
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    raw_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_silent.mp4"
    final_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    audio_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_ambient.wav"
    frame_count = int(round(float(CONFIG["duration_s"]) * int(CONFIG["fps"])))
    times = np.arange(frame_count) / int(CONFIG["fps"])

    print("Subtitle sidecar:", srt_path.resolve())
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(
        raw_video,
        fps=int(CONFIG["fps"]),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times, desc="Rendering earthquake short"):
            writer.append_data(scene.render_frame(float(t)))

    generate_ambient_soundtrack(audio_path, scene.featured)
    if mux_audio(raw_video, audio_path, final_video):
        print("Final video with audio:", final_video.resolve())
        return final_video
    shutil.copyfile(raw_video, final_video)
    print("ffmpeg audio mux unavailable; copied silent video to:", final_video.resolve())
    return final_video

