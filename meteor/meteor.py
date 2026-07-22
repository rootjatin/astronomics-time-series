from __future__ import annotations

"""
One Night of Meteor Activity
============================

A cinematic vertical YouTube Short renderer built from real meteor trajectory
measurements published by the Global Meteor Network (GMN).

What the video shows
--------------------
- The latest complete UTC date available in the GMN public database, or a date
  chosen with METEOR_SHORT_DATE=YYYY-MM-DD.
- Instrumentally reconstructed meteor ground tracks.
- Detection times through the 24-hour UTC date.
- Geocentric radiants, velocities, begin/end heights, shower codes, brightness,
  duration, and number of observing stations when those values are published.


Important honesty notes
-----------------------
- This is not every meteor that entered Earth's atmosphere. It is the subset
  reconstructed by participating GMN cameras and accepted into the public
  trajectory database.
- A UTC calendar date combines local nighttime observations from different
  longitudes. The title "One Night" is cinematic shorthand for one global UTC
  observing date.
- Counts can change when GMN publishes or reprocesses trajectories.
- Brightness is normalized absolute meteor magnitude at 100 km where available.
- Ground tracks and radiant markers are observational reconstructions, not
  simulated meteor paths.

Offline behavior
----------------
If the live service is unavailable, the renderer uses one documented GMN API
example trajectory solely to keep the layout testable. The video clearly labels
this as an offline documentation sample and never presents it as a complete
night.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg requests tqdm

Run final quality
-----------------
    python one_night_of_meteor_activity_short.py

Run a quick preview
-------------------
    METEOR_SHORT_QUICK=1 python one_night_of_meteor_activity_short.py

Choose a UTC date
-----------------
    METEOR_SHORT_DATE=2026-07-12 python one_night_of_meteor_activity_short.py

Force offline layout testing
----------------------------
    METEOR_SHORT_OFFLINE=1 METEOR_SHORT_QUICK=1 \
        python one_night_of_meteor_activity_short.py
"""

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

QUICK_MODE = os.environ.get("METEOR_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("METEOR_SHORT_OFFLINE", "0") == "1"
REFRESH = os.environ.get("METEOR_SHORT_REFRESH", "0") == "1"
DATE_OVERRIDE = os.environ.get("METEOR_SHORT_DATE", "").strip()
MAX_PAGES = max(1, int(os.environ.get("METEOR_SHORT_MAX_PAGES", "20")))

OUTPUT_ROOT = Path("one_night_of_meteor_activity_output")
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
    "basename": "one_night_of_meteor_activity",
    "title": "ONE NIGHT OF METEOR ACTIVITY",
    "subtitle": "Real trajectories reconstructed by a global camera network",
    "timeout_s": 40,
    "stars": 720,
    "max_draw_events": 900 if not QUICK_MODE else 300,
    "api_base": "https://explore.globalmeteornetwork.org/gmn_rest_api",
    "summary_endpoint": "https://explore.globalmeteornetwork.org/gmn_rest_api/meteor_summary",
    "source_page": "https://globalmeteornetwork.org/data/",
    "offline_sample_date": "2018-12-25",
}

W = CONFIG["width"]
H = CONFIG["height"]
SIZE = (W, H)
SCALE = W / 1080.0

COLORS = {
    "cyan": (91, 229, 255),
    "blue": (84, 146, 255),
    "violet": (188, 101, 255),
    "orange": (255, 174, 76),
    "gold": (255, 221, 105),
    "green": (102, 255, 177),
    "white": (245, 249, 255),
    "muted": (155, 198, 220),
    "red": (255, 98, 98),
}

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.0 if not QUICK_MODE else 1.6},
    {"name": "timeline", "start": 7.0 if not QUICK_MODE else 1.6, "end": 18.0 if not QUICK_MODE else 3.8},
    {"name": "world", "start": 18.0 if not QUICK_MODE else 3.8, "end": 31.0 if not QUICK_MODE else 6.3},
    {"name": "radiants", "start": 31.0 if not QUICK_MODE else 6.3, "end": 41.5 if not QUICK_MODE else 8.5},
    {"name": "physics", "start": 41.5 if not QUICK_MODE else 8.5, "end": 50.5 if not QUICK_MODE else 10.4},
    {"name": "highlight", "start": 50.5 if not QUICK_MODE else 10.4, "end": 55.0 if not QUICK_MODE else 11.2},
    {"name": "outro", "start": 55.0 if not QUICK_MODE else 11.2, "end": CONFIG["duration_s"]},
]


# =============================================================================
# Data model
# =============================================================================

@dataclass
class MeteorEvent:
    trajectory_id: str
    beginning_utc_time: str
    iau_code: str
    iau_no: Optional[int]
    rageo_deg: float
    decgeo_deg: float
    vgeo_km_s: float
    vinit_km_s: float
    latbeg_n_deg: float
    lonbeg_e_deg: float
    htbeg_km: float
    latend_n_deg: float
    lonend_e_deg: float
    htend_km: float
    duration_sec: float
    peak_absmag: float
    peak_ht_km: float
    mass_kg: float
    num_stations: int
    participating_stations: str

    @property
    def dt(self) -> Optional[datetime]:
        return parse_datetime(self.beginning_utc_time)

    @property
    def shower_label(self) -> str:
        code = (self.iau_code or "").strip().upper()
        if not code or code in {"...", "-1", "NONE", "NULL", "SPO"}:
            return "SPORADIC"
        return code

    @property
    def has_ground_track(self) -> bool:
        values = [self.latbeg_n_deg, self.lonbeg_e_deg, self.latend_n_deg, self.lonend_e_deg]
        return all(np.isfinite(values))

    @property
    def has_radiant(self) -> bool:
        return np.isfinite(self.rageo_deg) and np.isfinite(self.decgeo_deg)


# =============================================================================
# General utilities
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


def get_shot(t: float) -> Dict:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def parse_datetime(value: object) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = pd.to_datetime(text, utc=True)
        return parsed.to_pydatetime()
    except Exception:
        return None


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def finite_values(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if np.isfinite(v)]


def json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def percentile(values: Iterable[float], q: float, default: float = np.nan) -> float:
    clean = finite_values(values)
    return float(np.percentile(clean, q)) if clean else float(default)


def get_font(size: int, bold: bool = False):
    size = max(7, int(size))
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


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    anchor: str = "la",
    stroke: int = 2,
):
    ImageDraw.Draw(image).text(
        xy,
        str(text),
        font=get_font(size, bold=bold),
        fill=fill,
        anchor=anchor,
        stroke_width=max(0, stroke),
        stroke_fill=(0, 0, 0, min(fill[3] if len(fill) > 3 else 255, 220)),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    line_spacing: int = 6,
    max_lines: Optional[int] = None,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
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
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;: ") + "…"
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 220))
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += bbox[3] - bbox[1] + line_spacing


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def make_vignette(width: int, height: int, strength: float = 0.25) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    rr = np.sqrt(nx * nx + ny * ny)
    return np.clip(1 - strength * rr**1.75, 0, 1).astype(np.float32)


VIGNETTE = make_vignette(W, H)


def velocity_color(speed: float, alpha: int = 255) -> Tuple[int, int, int, int]:
    if not np.isfinite(speed):
        return COLORS["muted"] + (alpha,)
    t = clamp((speed - 11.0) / 61.0)
    stops = [
        (0.00, (86, 190, 255)),
        (0.33, (103, 255, 180)),
        (0.66, (255, 211, 92)),
        (1.00, (255, 100, 100)),
    ]
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if a <= t <= b:
            f = (t - a) / max(b - a, 1e-6)
            rgb = tuple(int(lerp(ca[i], cb[i], f)) for i in range(3))
            return rgb + (alpha,)
    return stops[-1][1] + (alpha,)


def brightness_color(magnitude: float, alpha: int = 255) -> Tuple[int, int, int, int]:
    if not np.isfinite(magnitude):
        return COLORS["cyan"] + (alpha,)
    # More negative absolute magnitude means brighter.
    t = clamp((-magnitude - 1.0) / 10.0)
    rgb = tuple(int(lerp(COLORS["cyan"][i], COLORS["gold"][i], t)) for i in range(3))
    return rgb + (alpha,)


# =============================================================================
# Live GMN data retrieval
# =============================================================================

def cached_json_get(url: str, params: Dict[str, object], cache_name: str) -> Dict:
    cache_path = CACHE_ROOT / cache_name
    if cache_path.exists() and not REFRESH:
        age = utc_now().timestamp() - cache_path.stat().st_mtime
        if age < 6 * 3600:
            return json.loads(cache_path.read_text(encoding="utf-8"))
    if requests is None:
        raise RuntimeError("The requests package is unavailable")
    response = requests.get(
        url,
        params=params,
        timeout=CONFIG["timeout_s"],
        headers={"User-Agent": "OneNightMeteorActivityShort/1.0 educational renderer"},
    )
    response.raise_for_status()
    payload = response.json()
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def validate_date_text(text: str) -> str:
    parsed = datetime.strptime(text, "%Y-%m-%d").date()
    return parsed.isoformat()


def fetch_latest_complete_gmn_date() -> str:
    sql = (
        "SELECT MAX(date(beginning_utc_time)) AS latest_date "
        "FROM meteor WHERE date(beginning_utc_time) < date('now')"
    )
    payload = cached_json_get(
        CONFIG["api_base"],
        {"sql": sql, "data_shape": "objects", "data_format": "json"},
        "gmn_latest_complete_date.json",
    )
    rows = payload.get("rows") or []
    if not rows or not rows[0].get("latest_date"):
        raise RuntimeError("GMN API did not return a latest complete date")
    return validate_date_text(str(rows[0]["latest_date"]))


def fetch_gmn_page(date_text: str, page: int) -> Tuple[List[Dict], Dict]:
    where = f"date(beginning_utc_time)='{date_text}'"
    payload = cached_json_get(
        CONFIG["summary_endpoint"],
        {
            "where": where,
            "order_by": "beginning_utc_time ASC",
            "page": page,
            "data_shape": "objects",
            "data_format": "json",
        },
        f"gmn_{date_text}_page_{page:03d}.json",
    )
    return list(payload.get("rows") or []), payload


def parse_gmn_row(row: Dict) -> MeteorEvent:
    stations = row.get("participating_stations")
    if isinstance(stations, list):
        stations_text = ",".join(str(item) for item in stations)
    else:
        stations_text = str(stations or "")
    return MeteorEvent(
        trajectory_id=str(row.get("unique_trajectory_identifier") or "unknown"),
        beginning_utc_time=str(row.get("beginning_utc_time") or ""),
        iau_code=str(row.get("iau_code") or ""),
        iau_no=safe_int(row.get("iau_no"), -1),
        rageo_deg=safe_float(row.get("rageo_deg")),
        decgeo_deg=safe_float(row.get("decgeo_deg")),
        vgeo_km_s=safe_float(row.get("vgeo_km_s")),
        vinit_km_s=safe_float(row.get("vinit_km_s")),
        latbeg_n_deg=safe_float(row.get("latbeg_n_deg")),
        lonbeg_e_deg=safe_float(row.get("lonbeg_e_deg")),
        htbeg_km=safe_float(row.get("htbeg_km")),
        latend_n_deg=safe_float(row.get("latend_n_deg")),
        lonend_e_deg=safe_float(row.get("lonend_e_deg")),
        htend_km=safe_float(row.get("htend_km")),
        duration_sec=safe_float(row.get("duration_sec")),
        peak_absmag=safe_float(row.get("peak_absmag")),
        peak_ht_km=safe_float(row.get("peak_ht_km")),
        mass_kg=safe_float(row.get("mass_kg_tau_0_7")),
        num_stations=safe_int(row.get("num_stat"), 0),
        participating_stations=stations_text,
    )


def fetch_meteors_for_date(date_text: str) -> Tuple[List[MeteorEvent], Dict]:
    events: List[MeteorEvent] = []
    page_payloads = []
    for page in range(1, MAX_PAGES + 1):
        rows, payload = fetch_gmn_page(date_text, page)
        page_payloads.append({"page": page, "rows": len(rows), "truncated": payload.get("truncated")})
        if not rows:
            break
        events.extend(parse_gmn_row(row) for row in rows)
        if len(rows) < 1000:
            break
    if not events:
        raise RuntimeError(f"No GMN meteor trajectories returned for {date_text}")
    events.sort(key=lambda event: event.beginning_utc_time)
    return events, {"pages": page_payloads, "max_pages": MAX_PAGES}


def offline_documentation_sample() -> List[MeteorEvent]:
    # Values exposed in the public GMN REST API documentation example for
    # trajectory 20181225032412_2Sciw. Unknown fields remain NaN rather than
    # being fabricated.
    return [
        MeteorEvent(
            trajectory_id="20181225032412_2Sciw",
            beginning_utc_time="2018-12-25 03:24:12.742673",
            iau_code="",
            iau_no=None,
            rageo_deg=100.94855,
            decgeo_deg=23.51387,
            vgeo_km_s=np.nan,
            vinit_km_s=np.nan,
            latbeg_n_deg=np.nan,
            lonbeg_e_deg=np.nan,
            htbeg_km=np.nan,
            latend_n_deg=np.nan,
            lonend_e_deg=np.nan,
            htend_km=np.nan,
            duration_sec=np.nan,
            peak_absmag=np.nan,
            peak_ht_km=np.nan,
            mass_kg=np.nan,
            num_stations=2,
            participating_stations="US0002,US0008",
        )
    ]


def choose_draw_events(events: Sequence[MeteorEvent], maximum: int) -> List[MeteorEvent]:
    if len(events) <= maximum:
        return list(events)
    # Preserve extremes, then fill with deterministic evenly spaced samples.
    required: List[MeteorEvent] = []
    metrics = [
        (lambda e: e.vgeo_km_s, True),
        (lambda e: e.peak_absmag, False),
        (lambda e: e.duration_sec, True),
        (lambda e: e.num_stations, True),
    ]
    for getter, descending in metrics:
        valid = [e for e in events if np.isfinite(getter(e))]
        if valid:
            required.append(sorted(valid, key=getter, reverse=descending)[0])
    unique = {event.trajectory_id: event for event in required}
    remaining_slots = max(0, maximum - len(unique))
    if remaining_slots:
        indices = np.linspace(0, len(events) - 1, remaining_slots, dtype=int)
        for index in indices:
            unique[events[int(index)].trajectory_id] = events[int(index)]
            if len(unique) >= maximum:
                break
    return sorted(unique.values(), key=lambda event: event.beginning_utc_time)


def summarize_events(events: Sequence[MeteorEvent], date_text: str, source_mode: str, fetch_meta: Dict, errors: Dict) -> Dict:
    speeds = finite_values(event.vgeo_km_s for event in events)
    begin_heights = finite_values(event.htbeg_km for event in events)
    end_heights = finite_values(event.htend_km for event in events)
    durations = finite_values(event.duration_sec for event in events)
    magnitudes = finite_values(event.peak_absmag for event in events)
    station_counts = finite_values(event.num_stations for event in events)

    shower_counts: Dict[str, int] = {}
    for event in events:
        shower_counts[event.shower_label] = shower_counts.get(event.shower_label, 0) + 1
    top_showers = sorted(shower_counts.items(), key=lambda item: (-item[1], item[0]))[:8]

    fastest = max((event for event in events if np.isfinite(event.vgeo_km_s)), key=lambda e: e.vgeo_km_s, default=None)
    brightest = min((event for event in events if np.isfinite(event.peak_absmag)), key=lambda e: e.peak_absmag, default=None)
    longest = max((event for event in events if np.isfinite(event.duration_sec)), key=lambda e: e.duration_sec, default=None)

    return {
        "title": CONFIG["title"],
        "date_utc": date_text,
        "generated_at_utc": iso_z(utc_now()),
        "source_mode": source_mode,
        "count": len(events),
        "ground_track_count": sum(event.has_ground_track for event in events),
        "radiant_count": sum(event.has_radiant for event in events),
        "shower_counts": shower_counts,
        "top_showers": top_showers,
        "sporadic_count": shower_counts.get("SPORADIC", 0),
        "median_speed_km_s": float(np.median(speeds)) if speeds else np.nan,
        "p95_speed_km_s": percentile(speeds, 95),
        "median_begin_height_km": float(np.median(begin_heights)) if begin_heights else np.nan,
        "median_end_height_km": float(np.median(end_heights)) if end_heights else np.nan,
        "median_duration_s": float(np.median(durations)) if durations else np.nan,
        "brightest_magnitude": min(magnitudes) if magnitudes else np.nan,
        "median_station_count": float(np.median(station_counts)) if station_counts else np.nan,
        "fastest_id": fastest.trajectory_id if fastest else None,
        "fastest_speed_km_s": fastest.vgeo_km_s if fastest else np.nan,
        "brightest_id": brightest.trajectory_id if brightest else None,
        "brightest_time_utc": brightest.beginning_utc_time if brightest else None,
        "brightest_shower": brightest.shower_label if brightest else None,
        "brightest_peak_absmag": brightest.peak_absmag if brightest else np.nan,
        "brightest_speed_km_s": brightest.vgeo_km_s if brightest else np.nan,
        "brightest_begin_height_km": brightest.htbeg_km if brightest else np.nan,
        "brightest_end_height_km": brightest.htend_km if brightest else np.nan,
        "brightest_duration_s": brightest.duration_sec if brightest else np.nan,
        "brightest_num_stations": brightest.num_stations if brightest else 0,
        "longest_id": longest.trajectory_id if longest else None,
        "longest_duration_s": longest.duration_sec if longest else np.nan,
        "fetch_meta": fetch_meta,
        "errors": errors,
        "source_url": CONFIG["source_page"],
        "license": "GMN high-level data: CC BY 4.0",
        "warning": "Instrumentally reconstructed GMN trajectories; not every atmospheric meteor.",
    }


def collect_data() -> Tuple[List[MeteorEvent], Dict]:
    errors: Dict[str, str] = {}
    if OFFLINE_MODE:
        events = offline_documentation_sample()
        date_text = CONFIG["offline_sample_date"]
        summary = summarize_events(
            events,
            date_text,
            "offline documentation sample",
            {"pages": [], "max_pages": 0},
            {"offline_mode": "Live requests skipped by METEOR_SHORT_OFFLINE=1"},
        )
        return events, summary

    try:
        date_text = validate_date_text(DATE_OVERRIDE) if DATE_OVERRIDE else fetch_latest_complete_gmn_date()
        events, fetch_meta = fetch_meteors_for_date(date_text)
        summary = summarize_events(events, date_text, "live GMN REST API", fetch_meta, errors)
        return events, summary
    except Exception as exc:
        errors["live_fetch"] = str(exc)
        events = offline_documentation_sample()
        summary = summarize_events(
            events,
            CONFIG["offline_sample_date"],
            "offline documentation sample",
            {"pages": [], "max_pages": 0},
            errors,
        )
        return events, summary


def save_data(events: Sequence[MeteorEvent], summary: Dict) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "meteor_trajectories.csv"
    json_path = DATA_ROOT / "meteor_activity_summary.json"
    pd.DataFrame([asdict(event) for event in events]).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps({"summary": summary, "events": [asdict(event) for event in events]}, indent=2, allow_nan=True, default=json_default),
        encoding="utf-8",
    )
    return csv_path, json_path


# =============================================================================
# Captions
# =============================================================================

def compact_number(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} million"
    if value >= 1000:
        return f"{value / 1000:.1f} thousand"
    return str(value)


def make_captions(summary: Dict) -> List[Tuple[float, float, str]]:
    count_text = compact_number(int(summary["count"]))
    date_text = summary["date_utc"]
    median_speed = summary.get("median_speed_km_s", np.nan)
    speed_text = f"{median_speed:.1f} kilometers per second" if np.isfinite(median_speed) else "a range of measured speeds"
    top = summary.get("top_showers") or []
    top_text = top[0][0] if top else "sporadic meteors"
    offline = summary.get("source_mode", "").startswith("offline")
    first = (
        "This offline frame uses one documented API example and is only a layout test."
        if offline
        else f"On {date_text}, the Global Meteor Network published {count_text} reconstructed meteor trajectories."
    )
    texts = [
        first,
        "Each mark is placed at its recorded UTC time, revealing how the global observing night unfolded.",
        "Camera stations triangulate where a meteor began and ended high above Earth, producing a measured atmospheric path.",
        "Tracing the paths backward reveals sky radiants. Clusters can be associated with meteor showers; many remain sporadic.",
        f"For trajectories with published values, the median geocentric speed was {speed_text}. The leading activity label was {top_text}.",
        "The brightest reconstructed event is highlighted using its normalized absolute magnitude, speed, duration, and observing stations.",
        "This is real camera-network data, but it is not a count of every meteor entering Earth's atmosphere.",
    ]
    captions: List[Tuple[float, float, str]] = []
    for shot, text in zip(SHOT_PLAN, texts):
        start = shot["start"] + min(0.4, max(0.05, (shot["end"] - shot["start"]) * 0.08))
        end = shot["end"] - min(0.2, max(0.03, (shot["end"] - shot["start"]) * 0.04))
        captions.append((start, max(start + 0.05, end), text))
    return captions


def write_srt(path: Path, captions: Sequence[Tuple[float, float, str]]):
    lines: List[str] = []
    for i, (start, end, text) in enumerate(captions, start=1):
        lines.extend([str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Geometry and simple maps
# =============================================================================

# Deliberately low-detail continent outlines so the script remains standalone.
CONTINENTS = [
    [(-168, 72), (-140, 70), (-126, 55), (-124, 42), (-116, 31), (-101, 21), (-83, 24), (-80, 10), (-95, 15), (-110, 30), (-126, 48), (-155, 58)],
    [(-82, 12), (-70, 8), (-55, -5), (-48, -24), (-60, -52), (-73, -44), (-78, -15)],
    [(-18, 36), (10, 37), (32, 31), (50, 12), (42, -12), (31, -35), (12, -35), (-4, -10), (-16, 12)],
    [(-10, 36), (5, 55), (30, 70), (60, 72), (90, 62), (120, 52), (150, 55), (175, 66), (165, 44), (132, 34), (110, 20), (88, 8), (68, 24), (45, 40), (24, 38)],
    [(112, -10), (154, -12), (152, -40), (132, -44), (114, -29)],
    [(-52, 60), (-30, 72), (-18, 82), (-45, 84), (-65, 72)],
    [(-180, -72), (-120, -70), (-60, -73), (0, -70), (60, -74), (120, -71), (180, -72), (180, -88), (-180, -88)],
]


def map_xy(lon: float, lat: float, box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = box
    x = x0 + ((lon + 180.0) / 360.0) * (x1 - x0)
    y = y0 + ((90.0 - lat) / 180.0) * (y1 - y0)
    return x, y


def sky_xy(ra_deg: float, dec_deg: float, box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = box
    x = x0 + ((360.0 - ra_deg) % 360.0) / 360.0 * (x1 - x0)
    y = y0 + (90.0 - dec_deg) / 180.0 * (y1 - y0)
    return x, y


def interpolate_lon(lon0: float, lon1: float, t: float) -> float:
    delta = lon1 - lon0
    if delta > 180:
        delta -= 360
    elif delta < -180:
        delta += 360
    value = lon0 + delta * t
    while value > 180:
        value -= 360
    while value < -180:
        value += 360
    return value


# =============================================================================
# Scene renderer
# =============================================================================

class MeteorScene:
    def __init__(self, events: Sequence[MeteorEvent], summary: Dict):
        self.events = list(events)
        self.summary = summary
        self.draw_events = choose_draw_events(self.events, CONFIG["max_draw_events"])
        self.captions = make_captions(summary)
        self.stars = self._make_stars(CONFIG["stars"], 20260715)
        self.map_box = (int(W * 0.055), int(H * 0.205), int(W * 0.945), int(H * 0.665))
        self.sky_box = (int(W * 0.065), int(H * 0.215), int(W * 0.935), int(H * 0.67))
        self.timeline_bins = self._timeline_bins()
        self.shower_counts = summary.get("top_showers") or []
        self.brightest = self._brightest_event()

    @staticmethod
    def _make_stars(n: int, seed: int) -> List[Dict]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, W)),
                "y": float(rng.uniform(0, H)),
                "r": float(rng.uniform(0.35, 2.1) * SCALE),
                "a": int(rng.integers(28, 160)),
                "phase": float(rng.uniform(0, math.tau)),
            }
            for _ in range(n)
        ]

    def _timeline_bins(self) -> np.ndarray:
        bins = np.zeros(24, dtype=int)
        for event in self.events:
            dt = event.dt
            if dt:
                bins[dt.hour] += 1
        return bins

    def _brightest_event(self) -> MeteorEvent:
        valid = [event for event in self.events if np.isfinite(event.peak_absmag)]
        return min(valid, key=lambda event: event.peak_absmag) if valid else self.events[0]

    def caption_at(self, t: float) -> Optional[str]:
        for start, end, text in self.captions:
            if start <= t < end:
                return text
        return None

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", SIZE, (2, 5, 13, 255))
        nebula = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        nd = ImageDraw.Draw(nebula)
        clouds = [
            (W * 0.18, H * 0.24, W * 0.34, (20, 77, 135, 28)),
            (W * 0.76, H * 0.31, W * 0.38, (96, 34, 120, 23)),
            (W * 0.52, H * 0.78, W * 0.44, (15, 80, 90, 19)),
        ]
        for cx, cy, radius, fill in clouds:
            nd.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)
        nebula = nebula.filter(ImageFilter.GaussianBlur(max(20, int(75 * SCALE))))
        image.alpha_composite(nebula)
        d = ImageDraw.Draw(image)
        for star in self.stars:
            alpha = int(star["a"] * (0.72 + 0.28 * math.sin(1.45 * t + star["phase"])))
            r = star["r"]
            d.ellipse((star["x"] - r, star["y"] - r, star["x"] + r, star["y"] + r), fill=(220, 233, 255, alpha))
        return image

    def draw_header(self, image: Image.Image, t: float):
        shot = get_shot(t)["name"]
        shot_titles = {
            "intro": "A GLOBAL UTC OBSERVING DATE",
            "timeline": "WHEN THE CAMERAS SAW THEM",
            "world": "RECONSTRUCTED ATMOSPHERIC PATHS",
            "radiants": "WHERE THEY CAME FROM IN THE SKY",
            "physics": "SPEED • HEIGHT • SHOWER ACTIVITY",
            "highlight": "THE BRIGHTEST RECONSTRUCTED EVENT",
            "outro": "REAL OBSERVATIONS • INCOMPLETE SKY COVERAGE",
        }
        top = int(54 * SCALE)
        draw_text(image, shot_titles[shot], (int(54 * SCALE), top), int(20 * SCALE), COLORS["cyan"] + (225,), True, stroke=1)
        draw_text(
            image,
            f"UTC {self.summary['date_utc']}  //  {self.summary['count']:,} TRAJECTORIES",
            (W - int(54 * SCALE), top),
            int(17 * SCALE),
            COLORS["muted"] + (215,),
            True,
            anchor="ra",
            stroke=1,
        )
        if self.summary["source_mode"].startswith("offline"):
            draw_text(
                image,
                "OFFLINE DOCUMENTATION SAMPLE — NOT A COMPLETE NIGHT",
                (W // 2, int(96 * SCALE)),
                int(18 * SCALE),
                COLORS["orange"] + (245,),
                True,
                anchor="ma",
                stroke=1,
            )

    def draw_intro(self, image: Image.Image, t: float):
        alpha = int(255 * smoothstep((t - 0.15) / 0.8))
        draw_text(image, CONFIG["title"], (W // 2, int(H * 0.17)), int(50 * SCALE), COLORS["white"] + (alpha,), True, anchor="ma")
        draw_wrapped_text(
            image,
            CONFIG["subtitle"],
            (int(W * 0.13), int(H * 0.215)),
            int(W * 0.74),
            int(25 * SCALE),
            COLORS["cyan"] + (min(alpha, 235),),
            True,
            max_lines=2,
        )
        cx, cy = W * 0.5, H * 0.47
        earth_r = W * 0.23
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse((cx-earth_r*1.25, cy-earth_r*1.25, cx+earth_r*1.25, cy+earth_r*1.25), fill=(36, 134, 245, 36))
        glow = glow.filter(ImageFilter.GaussianBlur(int(30 * SCALE)))
        image.alpha_composite(glow)
        d = ImageDraw.Draw(image)
        d.ellipse((cx-earth_r, cy-earth_r, cx+earth_r, cy+earth_r), fill=(8, 39, 85, 255), outline=(120, 220, 255, 160), width=max(1, int(3*SCALE)))
        # Earth grid.
        for lat in [-60, -30, 0, 30, 60]:
            y = cy - earth_r * math.sin(math.radians(lat))
            width = earth_r * math.cos(math.radians(lat))
            d.ellipse((cx-width, y-earth_r*0.12, cx+width, y+earth_r*0.12), outline=(90, 195, 235, 34), width=1)
        for angle in [-60, -30, 0, 30, 60]:
            offset = earth_r * math.sin(math.radians(angle))
            d.ellipse((cx-earth_r*0.12+offset, cy-earth_r, cx+earth_r*0.12+offset, cy+earth_r), outline=(90, 195, 235, 34), width=1)
        # Animated meteor streaks around Earth.
        rng = np.random.default_rng(511)
        for i in range(42 if not QUICK_MODE else 18):
            phase = (t * (0.13 + (i % 5) * 0.015) + rng.uniform(0, 1)) % 1.0
            angle = rng.uniform(-math.pi, math.pi)
            radius = earth_r * rng.uniform(1.25, 1.72)
            x = cx + math.cos(angle) * radius
            y = cy + math.sin(angle) * radius * 0.82
            length = rng.uniform(24, 82) * SCALE
            direction = angle + math.pi + rng.uniform(-0.45, 0.45)
            ex = x + math.cos(direction) * length * phase
            ey = y + math.sin(direction) * length * phase
            color = velocity_color(rng.uniform(12, 70), 165)
            d.line((x, y, ex, ey), fill=color, width=max(1, int(rng.uniform(1, 4)*SCALE)))
        draw_text(
            image,
            f"{self.summary['count']:,} PUBLISHED TRAJECTORIES",
            (W // 2, int(H * 0.73)),
            int(27 * SCALE),
            COLORS["white"] + (240,),
            True,
            anchor="ma",
        )
        draw_text(
            image,
            self.summary["date_utc"],
            (W // 2, int(H * 0.775)),
            int(23 * SCALE),
            COLORS["cyan"] + (235,),
            True,
            anchor="ma",
        )

    def draw_timeline(self, image: Image.Image, t: float):
        x0, x1 = int(W * 0.085), int(W * 0.915)
        y0, y1 = int(H * 0.25), int(H * 0.68)
        panel = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rounded_rectangle((x0, y0, x1, y1), radius=int(25*SCALE), fill=(2, 8, 19, 190), outline=(95, 205, 235, 68), width=max(1, int(2*SCALE)))
        image.alpha_composite(panel)
        max_count = max(int(self.timeline_bins.max()), 1)
        d = ImageDraw.Draw(image)
        chart_left, chart_right = x0 + int(36*SCALE), x1 - int(30*SCALE)
        chart_top, chart_bottom = y0 + int(82*SCALE), y1 - int(72*SCALE)
        for hour in range(25):
            x = lerp(chart_left, chart_right, hour / 24)
            d.line((x, chart_top, x, chart_bottom), fill=(100, 180, 210, 26), width=1)
            if hour % 3 == 0 and hour < 24:
                draw_text(image, f"{hour:02d}", (int(x), chart_bottom + int(24*SCALE)), int(15*SCALE), COLORS["muted"] + (200,), anchor="ma", stroke=1)
        draw_text(image, "UTC HOUR", (chart_right, chart_bottom + int(55*SCALE)), int(14*SCALE), COLORS["muted"] + (190,), True, anchor="ra", stroke=1)
        reveal = smoothstep((t - SHOT_PLAN[1]["start"]) / max(SHOT_PLAN[1]["end"] - SHOT_PLAN[1]["start"], 1e-6))
        bars_to_draw = max(1, int(math.ceil(24 * reveal)))
        bar_w = (chart_right - chart_left) / 24 * 0.68
        for hour, count in enumerate(self.timeline_bins[:bars_to_draw]):
            x = lerp(chart_left, chart_right, (hour + 0.5) / 24)
            height = (chart_bottom - chart_top) * (count / max_count)
            color = velocity_color(12 + hour / 23 * 58, 220)
            d.rounded_rectangle((x-bar_w/2, chart_bottom-height, x+bar_w/2, chart_bottom), radius=max(1, int(5*SCALE)), fill=color)
            if count == max_count:
                draw_text(image, f"{count:,}", (int(x), int(chart_bottom-height-int(12*SCALE))), int(15*SCALE), COLORS["white"] + (230,), True, anchor="ms", stroke=1)
        draw_text(image, "DETECTIONS THROUGH ONE UTC DATE", (x0 + int(30*SCALE), y0 + int(28*SCALE)), int(23*SCALE), COLORS["cyan"] + (235,), True, stroke=1)
        peak_hour = int(np.argmax(self.timeline_bins))
        peak_count = int(self.timeline_bins[peak_hour])
        draw_text(image, f"BUSIEST UTC HOUR  {peak_hour:02d}:00–{(peak_hour+1)%24:02d}:00", (x0 + int(30*SCALE), y1 - int(38*SCALE)), int(18*SCALE), COLORS["white"] + (225,), True, stroke=1)
        draw_text(image, f"{peak_count:,} TRAJECTORIES", (x1 - int(30*SCALE), y1 - int(38*SCALE)), int(18*SCALE), COLORS["gold"] + (235,), True, anchor="ra", stroke=1)

    def draw_world_grid(self, image: Image.Image):
        x0, y0, x1, y1 = self.map_box
        panel = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rounded_rectangle((x0, y0, x1, y1), radius=int(25*SCALE), fill=(2, 8, 19, 198), outline=(95, 205, 235, 72), width=max(1, int(2*SCALE)))
        image.alpha_composite(panel)
        d = ImageDraw.Draw(image)
        for lon in range(-150, 181, 30):
            x, _ = map_xy(lon, 0, self.map_box)
            d.line((x, y0+int(12*SCALE), x, y1-int(12*SCALE)), fill=(90, 170, 200, 31), width=1)
        for lat in range(-60, 61, 30):
            _, y = map_xy(0, lat, self.map_box)
            d.line((x0+int(12*SCALE), y, x1-int(12*SCALE), y), fill=(90, 170, 200, 31), width=1)
        for polygon in CONTINENTS:
            points = [map_xy(lon, lat, self.map_box) for lon, lat in polygon]
            d.polygon(points, fill=(15, 48, 68, 190), outline=(79, 150, 172, 100))
        draw_text(image, "BEGIN → END POSITION AT OBSERVED HEIGHT", (x0 + int(20*SCALE), y0 + int(18*SCALE)), int(17*SCALE), COLORS["muted"] + (210,), True, stroke=1)

    def draw_world(self, image: Image.Image, t: float):
        self.draw_world_grid(image)
        x0, y0, x1, y1 = self.map_box
        d = ImageDraw.Draw(image)
        shot = SHOT_PLAN[2]
        reveal = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-6))
        visible = max(1, int(math.ceil(len(self.draw_events) * reveal)))
        tracks = [event for event in self.draw_events if event.has_ground_track][:visible]
        for idx, event in enumerate(tracks):
            color = velocity_color(event.vgeo_km_s, 112 if idx < len(tracks)-20 else 210)
            points = []
            for k in range(16):
                f = k / 15
                lon = interpolate_lon(event.lonbeg_e_deg, event.lonend_e_deg, f)
                lat = lerp(event.latbeg_n_deg, event.latend_n_deg, f)
                points.append(map_xy(lon, lat, self.map_box))
            # Dateline jumps are omitted rather than drawing a false line across the map.
            for a, b in zip(points[:-1], points[1:]):
                if abs(a[0] - b[0]) < (x1 - x0) * 0.35:
                    d.line((a[0], a[1], b[0], b[1]), fill=color, width=max(1, int(2.2*SCALE)))
            bx, by = points[0]
            ex, ey = points[-1]
            r = max(1.2, 2.6*SCALE)
            d.ellipse((bx-r, by-r, bx+r, by+r), fill=COLORS["cyan"] + (150,))
            d.ellipse((ex-r, ey-r, ex+r, ey+r), fill=COLORS["orange"] + (150,))
        draw_text(image, f"{self.summary['ground_track_count']:,} EVENTS WITH PUBLISHED GROUND COORDINATES", (x0 + int(18*SCALE), y1 + int(42*SCALE)), int(18*SCALE), COLORS["white"] + (230,), True, stroke=1)
        # Legend.
        legend_y = y1 + int(80*SCALE)
        draw_text(image, "11 km/s", (x0, legend_y), int(15*SCALE), COLORS["muted"] + (210,), stroke=1)
        gradient_x0 = x0 + int(95*SCALE)
        gradient_x1 = x1 - int(95*SCALE)
        for i in range(max(1, int(gradient_x1-gradient_x0))):
            speed = 11 + 61 * i / max(1, gradient_x1-gradient_x0-1)
            d.line((gradient_x0+i, legend_y, gradient_x0+i, legend_y+int(12*SCALE)), fill=velocity_color(speed, 225), width=1)
        draw_text(image, "72 km/s", (x1, legend_y), int(15*SCALE), COLORS["muted"] + (210,), anchor="ra", stroke=1)
        draw_text(image, "GEOCENTRIC SPEED", (W//2, legend_y + int(35*SCALE)), int(14*SCALE), COLORS["muted"] + (195,), True, anchor="ma", stroke=1)

    def draw_sky_grid(self, image: Image.Image):
        x0, y0, x1, y1 = self.sky_box
        panel = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rounded_rectangle((x0, y0, x1, y1), radius=int(25*SCALE), fill=(2, 8, 19, 196), outline=(95, 205, 235, 72), width=max(1, int(2*SCALE)))
        image.alpha_composite(panel)
        d = ImageDraw.Draw(image)
        for ra in range(0, 361, 30):
            x, _ = sky_xy(ra % 360, 0, self.sky_box)
            d.line((x, y0+int(12*SCALE), x, y1-int(12*SCALE)), fill=(90, 170, 200, 32), width=1)
        for dec in range(-60, 61, 30):
            _, y = sky_xy(0, dec, self.sky_box)
            d.line((x0+int(12*SCALE), y, x1-int(12*SCALE), y), fill=(90, 170, 200, 32), width=1)
        draw_text(image, "GEOCENTRIC RADIANTS • J2000 • RA RUNS RIGHT → LEFT", (x0 + int(18*SCALE), y0 + int(18*SCALE)), int(16*SCALE), COLORS["muted"] + (210,), True, stroke=1)

    def draw_radiants(self, image: Image.Image, t: float):
        self.draw_sky_grid(image)
        d = ImageDraw.Draw(image)
        shot = SHOT_PLAN[3]
        reveal = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-6))
        radiants = [event for event in self.draw_events if event.has_radiant]
        visible = max(1, int(math.ceil(len(radiants) * reveal)))
        for event in radiants[:visible]:
            x, y = sky_xy(event.rageo_deg, event.decgeo_deg, self.sky_box)
            radius = (1.8 + clamp((-event.peak_absmag if np.isfinite(event.peak_absmag) else 0) / 6.0) * 3.2) * SCALE
            color = velocity_color(event.vgeo_km_s, 110)
            d.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
        # Top shower summary cards.
        y = int(H * 0.715)
        card_w = int(W * 0.27)
        gap = int(W * 0.025)
        top3 = self.shower_counts[:3]
        for i in range(3):
            x = int(W * 0.06) + i * (card_w + gap)
            panel = Image.new("RGBA", SIZE, (0, 0, 0, 0))
            pd = ImageDraw.Draw(panel)
            pd.rounded_rectangle((x, y, x+card_w, y+int(H*0.105)), radius=int(16*SCALE), fill=(3, 9, 20, 188), outline=(100, 185, 215, 54), width=1)
            image.alpha_composite(panel)
            if i < len(top3):
                code, count = top3[i]
                draw_text(image, code, (x+int(16*SCALE), y+int(15*SCALE)), int(22*SCALE), COLORS["cyan"] + (235,), True, stroke=1)
                draw_text(image, f"{count:,}", (x+card_w-int(16*SCALE), y+int(15*SCALE)), int(23*SCALE), COLORS["gold"] + (240,), True, anchor="ra", stroke=1)
                draw_text(image, "TRAJECTORIES", (x+int(16*SCALE), y+int(52*SCALE)), int(13*SCALE), COLORS["muted"] + (205,), True, stroke=1)
        draw_text(image, f"{self.summary['radiant_count']:,} EVENTS WITH PUBLISHED RADIANTS", (W//2, int(H*0.855)), int(18*SCALE), COLORS["white"] + (230,), True, anchor="ma", stroke=1)

    def draw_metric_card(self, image: Image.Image, box: Tuple[int, int, int, int], title: str, value: str, subtitle: str, color: Tuple[int, int, int]):
        x0, y0, x1, y1 = box
        panel = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rounded_rectangle(box, radius=int(22*SCALE), fill=(3, 9, 20, 190), outline=color + (78,), width=max(1, int(2*SCALE)))
        image.alpha_composite(panel)
        draw_text(image, title, (x0+int(20*SCALE), y0+int(20*SCALE)), int(17*SCALE), color + (235,), True, stroke=1)
        draw_text(image, value, (x0+int(20*SCALE), y0+int(58*SCALE)), int(34*SCALE), COLORS["white"] + (245,), True, stroke=1)
        draw_wrapped_text(image, subtitle, (x0+int(20*SCALE), y0+int(110*SCALE)), x1-x0-int(40*SCALE), int(15*SCALE), COLORS["muted"] + (215,), max_lines=2)

    def draw_physics(self, image: Image.Image, t: float):
        margin = int(W * 0.07)
        gap = int(W * 0.035)
        card_w = int((W - 2*margin - gap) / 2)
        card_h = int(H * 0.205)
        y0 = int(H * 0.21)
        speed = self.summary.get("median_speed_km_s", np.nan)
        begin_h = self.summary.get("median_begin_height_km", np.nan)
        end_h = self.summary.get("median_end_height_km", np.nan)
        sporadic = int(self.summary.get("sporadic_count", 0))
        count = max(1, int(self.summary.get("count", 1)))
        cards = [
            ("MEDIAN SPEED", f"{speed:.1f} km/s" if np.isfinite(speed) else "NOT PUBLISHED", "Geocentric velocity for trajectories with a reported solution.", COLORS["red"]),
            ("MEDIAN HEIGHT", f"{begin_h:.0f} → {end_h:.0f} km" if np.isfinite(begin_h) and np.isfinite(end_h) else "NOT PUBLISHED", "Median observed beginning and ending altitude above WGS84.", COLORS["cyan"]),
            ("SPORADIC SHARE", f"{100*sporadic/count:.0f}%", f"{sporadic:,} trajectories were not assigned a named shower code.", COLORS["violet"]),
            ("CAMERA AGREEMENT", f"{self.summary.get('median_station_count', np.nan):.0f} stations" if np.isfinite(self.summary.get("median_station_count", np.nan)) else "NOT PUBLISHED", "Median number of GMN stations participating in each trajectory.", COLORS["green"]),
        ]
        for i, (title, value, subtitle, color) in enumerate(cards):
            col, row = i % 2, i // 2
            x = margin + col*(card_w+gap)
            y = y0 + row*(card_h+int(H*0.035))
            self.draw_metric_card(image, (x, y, x+card_w, y+card_h), title, value, subtitle, color)
        # Speed distribution strip.
        speeds = np.array(finite_values(event.vgeo_km_s for event in self.events), dtype=float)
        strip_y = int(H * 0.72)
        d = ImageDraw.Draw(image)
        draw_text(image, "VELOCITY DISTRIBUTION", (margin, strip_y-int(38*SCALE)), int(18*SCALE), COLORS["muted"] + (220,), True, stroke=1)
        strip_x0, strip_x1 = margin, W-margin
        d.rounded_rectangle((strip_x0, strip_y, strip_x1, strip_y+int(20*SCALE)), radius=int(8*SCALE), fill=(25, 46, 65, 220))
        if speeds.size:
            quantiles = np.percentile(speeds, [5, 25, 50, 75, 95])
            for qv, label in zip(quantiles, ["P05", "P25", "P50", "P75", "P95"]):
                x = lerp(strip_x0, strip_x1, clamp((qv-11)/61))
                d.line((x, strip_y-int(8*SCALE), x, strip_y+int(30*SCALE)), fill=velocity_color(qv, 245), width=max(1, int(3*SCALE)))
                draw_text(image, f"{label} {qv:.1f}", (int(x), strip_y+int(44*SCALE)), int(13*SCALE), COLORS["muted"] + (205,), anchor="ma", stroke=1)

    def draw_highlight(self, image: Image.Image, t: float):
        event = self.brightest
        x0, y0, x1, y1 = int(W*0.075), int(H*0.20), int(W*0.925), int(H*0.75)
        panel = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rounded_rectangle((x0, y0, x1, y1), radius=int(30*SCALE), fill=(3, 8, 18, 205), outline=COLORS["gold"] + (105,), width=max(1, int(2*SCALE)))
        image.alpha_composite(panel)
        mag = event.peak_absmag
        mag_text = f"{mag:.1f} mag" if np.isfinite(mag) else "Brightness unavailable"
        draw_text(image, "BRIGHTEST PUBLISHED TRAJECTORY", (x0+int(28*SCALE), y0+int(30*SCALE)), int(20*SCALE), COLORS["gold"] + (245,), True, stroke=1)
        draw_text(image, mag_text, (x0+int(28*SCALE), y0+int(82*SCALE)), int(50*SCALE), COLORS["white"] + (250,), True, stroke=1)
        dt = event.dt
        time_text = dt.strftime("%H:%M:%S UTC") if dt else "Time unavailable"
        draw_text(image, f"{time_text}  •  {event.shower_label}", (x0+int(28*SCALE), y0+int(155*SCALE)), int(22*SCALE), COLORS["cyan"] + (235,), True, stroke=1)
        # Stylized streak whose geometry is decorative, while text remains data-driven.
        d = ImageDraw.Draw(image)
        sx, sy = x0+int(85*SCALE), y0+int(260*SCALE)
        ex, ey = x1-int(85*SCALE), y0+int(385*SCALE)
        pulse = 0.78 + 0.22*math.sin(t*4.0)
        for width, alpha in [(18, 18), (10, 40), (4, 230)]:
            d.line((sx, sy, ex, ey), fill=COLORS["gold"] + (int(alpha*pulse),), width=max(1, int(width*SCALE)))
        r = int(12*SCALE)
        d.ellipse((ex-r, ey-r, ex+r, ey+r), fill=COLORS["white"] + (245,))
        metrics = [
            ("SPEED", f"{event.vgeo_km_s:.1f} km/s" if np.isfinite(event.vgeo_km_s) else "N/A"),
            ("HEIGHT", f"{event.htbeg_km:.0f} → {event.htend_km:.0f} km" if np.isfinite(event.htbeg_km) and np.isfinite(event.htend_km) else "N/A"),
            ("DURATION", f"{event.duration_sec:.2f} s" if np.isfinite(event.duration_sec) else "N/A"),
            ("STATIONS", str(event.num_stations) if event.num_stations else "N/A"),
        ]
        metric_y = y1-int(105*SCALE)
        for i, (label, value) in enumerate(metrics):
            x = lerp(x0+int(35*SCALE), x1-int(35*SCALE), i/3 if len(metrics)>1 else 0)
            anchor = "la" if i == 0 else "ra" if i == 3 else "ma"
            draw_text(image, label, (int(x), metric_y), int(14*SCALE), COLORS["muted"] + (205,), True, anchor=anchor, stroke=1)
            draw_text(image, value, (int(x), metric_y+int(30*SCALE)), int(19*SCALE), COLORS["white"] + (235,), True, anchor=anchor, stroke=1)
        draw_text(image, event.trajectory_id, (x0+int(28*SCALE), y1+int(40*SCALE)), int(14*SCALE), COLORS["muted"] + (190,), stroke=1)

    def draw_outro(self, image: Image.Image, t: float):
        self.draw_world_grid(image)
        d = ImageDraw.Draw(image)
        for event in [e for e in self.draw_events if e.has_ground_track][:300]:
            bx, by = map_xy(event.lonbeg_e_deg, event.latbeg_n_deg, self.map_box)
            ex, ey = map_xy(event.lonend_e_deg, event.latend_n_deg, self.map_box)
            if abs(bx-ex) < (self.map_box[2]-self.map_box[0])*0.35:
                d.line((bx, by, ex, ey), fill=velocity_color(event.vgeo_km_s, 66), width=max(1, int(1.5*SCALE)))
        draw_text(image, "OBSERVED — NOT ALL-SKY COMPLETE", (W//2, int(H*0.73)), int(28*SCALE), COLORS["white"] + (242,), True, anchor="ma")
        draw_wrapped_text(
            image,
            "GMN camera coverage, weather, daylight, geometry, sensitivity, and data-quality filters determine which meteors become published trajectories.",
            (int(W*0.12), int(H*0.775)),
            int(W*0.76),
            int(20*SCALE),
            COLORS["cyan"] + (225,),
            True,
            max_lines=3,
        )
        draw_text(image, "SOURCE: GLOBAL METEOR NETWORK • CC BY 4.0", (W//2, int(H*0.90)), int(16*SCALE), COLORS["muted"] + (210,), True, anchor="ma", stroke=1)

    def draw_caption(self, image: Image.Image, t: float):
        caption = self.caption_at(t)
        if not caption:
            return
        y0 = H - int(238*SCALE)
        x0 = int(44*SCALE)
        x1 = W - int(44*SCALE)
        panel = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rounded_rectangle((x0, y0, x1, y0+int(126*SCALE)), radius=int(23*SCALE), fill=(2, 6, 14, 180), outline=(80, 185, 220, 65), width=1)
        image.alpha_composite(panel)
        draw_wrapped_text(image, caption, (x0+int(25*SCALE), y0+int(26*SCALE)), x1-x0-int(50*SCALE), int(27*SCALE), COLORS["white"] + (245,), max_lines=3)

    def draw_scanlines(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        step = max(4, int(7*SCALE))
        for y in range(int((t*37)%step), H, step):
            od.line((0, y, W, y), fill=(110, 195, 230, 8), width=1)
        scan_y = int((t*170) % (H+180)) - 90
        od.rectangle((0, scan_y, W, scan_y+int(42*SCALE)), fill=(100, 210, 240, 7))
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        image = self.background(t)
        self.draw_header(image, t)
        shot = get_shot(t)["name"]
        if shot == "intro":
            self.draw_intro(image, t)
        elif shot == "timeline":
            self.draw_timeline(image, t)
        elif shot == "world":
            self.draw_world(image, t)
        elif shot == "radiants":
            self.draw_radiants(image, t)
        elif shot == "physics":
            self.draw_physics(image, t)
        elif shot == "highlight":
            self.draw_highlight(image, t)
        else:
            self.draw_outro(image, t)
        self.draw_caption(image, t)
        self.draw_scanlines(image, t)
        graded = ImageEnhance.Contrast(image.convert("RGB")).enhance(1.08)
        graded = ImageEnhance.Color(graded).enhance(1.06)
        arr = np.asarray(graded).astype(np.float32)
        arr *= VIGNETTE[..., None]
        fade_in = smoothstep(t / 0.8)
        fade_out = 1 - smoothstep((t - (CONFIG["duration_s"] - 0.9)) / 0.8)
        arr *= fade_in * fade_out
        return np.clip(arr, 0, 255).astype(np.uint8)


# =============================================================================
# Output helpers
# =============================================================================

def render_video(scene: MeteorScene) -> Path:
    raw_path = OUTPUT_ROOT / f"{CONFIG['basename']}_raw.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['basename']}_final.mp4"
    write_srt(OUTPUT_ROOT / f"{CONFIG['basename']}.srt", scene.captions)
    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    with iio.get_writer(
        raw_path,
        fps=CONFIG["fps"],
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for frame_index in tqdm(range(frame_count), desc="Rendering meteor activity short"):
            writer.append_data(scene.render_frame(frame_index / CONFIG["fps"]))
    shutil.copyfile(raw_path, final_path)
    return final_path


def make_contact_sheet(paths: Sequence[Path], output_path: Path):
    thumbs = []
    for path in paths[:6]:
        image = Image.open(path).convert("RGB").resize((270, 480))
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 118, 38), fill=(0, 0, 0))
        draw.text((16, 13), path.stem.replace("preview_", ""), fill=(255, 255, 255))
        thumbs.append(image)
    sheet = Image.new("RGB", (600, 1520), (8, 11, 18))
    for index, thumb in enumerate(thumbs):
        row, col = divmod(index, 2)
        sheet.paste(thumb, (20 + col*290, 20 + row*500))
    sheet.save(output_path, quality=92)


def main():
    print("Collecting GMN meteor trajectory data ...")
    events, summary = collect_data()
    csv_path, json_path = save_data(events, summary)
    print("Date:", summary["date_utc"])
    print("Source mode:", summary["source_mode"])
    print("Trajectories:", summary["count"])
    print("Data:", csv_path.resolve())
    print("Summary:", json_path.resolve())

    scene = MeteorScene(events, summary)
    preview_times = [
        1.0,
        min(10.0, CONFIG["duration_s"]*0.20),
        min(23.0, CONFIG["duration_s"]*0.40),
        min(35.0, CONFIG["duration_s"]*0.61),
        min(46.0, CONFIG["duration_s"]*0.80),
        CONFIG["duration_s"]-0.7,
    ]
    preview_paths: List[Path] = []
    for preview_time in tqdm(preview_times, desc="Preview frames"):
        path = PREVIEW_ROOT / f"preview_{int(preview_time):02d}s.png"
        Image.fromarray(scene.render_frame(float(preview_time))).save(path)
        preview_paths.append(path)
    make_contact_sheet(preview_paths, PREVIEW_ROOT / "one_night_of_meteor_activity_contact_sheet.jpg")
    video_path = render_video(scene)
    print("Video:", video_path.resolve())
    print("Source status:", json.dumps({k: summary[k] for k in ["source_mode", "date_utc", "count", "errors"]}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
