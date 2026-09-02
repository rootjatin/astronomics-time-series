from __future__ import annotations

"""
Every Tropical Cyclone on Earth This Year — cinematic YouTube Short renderer

Creates a vertical 1080x1920 data-driven short that introduces every tropical
cyclone represented in the selected calendar year one by one, while its track
appears on a global map. A running counter, basin label, storm name, peak wind,
and year-to-date cutoff turn the global archive into a fast cinematic roll call.



Default year
------------
The current UTC calendar year is used by default. For a completed year:

    CYCLONE_YEAR=2025 python every_tropical_cyclone_on_earth_this_year.py

For an in-progress year, the animation ends at the current UTC instant.

Scientific framing
------------------
- One on-screen entry corresponds to one unique IBTrACS storm identifier (SID)
  with at least two track points inside the selected calendar-year window.
- The plotted line is the portion of that storm's IBTrACS track inside the year.
- Recent-year records can be provisional and may be revised after operational
  analysis; rerunning the script refreshes the roll call.
- Wind values prefer WMO wind when available, then USA and other agency fields.
  Agencies use different wind-averaging periods, so cross-basin comparisons are
  approximate and the color bands are visualization thresholds in knots.
- A tropical cyclone can contain subtropical or post-tropical track points in
  the archive. The storm count is therefore a count of unique IBTrACS cyclone
  records entering the selected year, not a count of individual advisories.
- For the current year, "every" means every storm present in the latest loaded
  IBTrACS update through the UTC cutoff shown in the video.

Offline behavior
----------------
If IBTrACS or Natural Earth cannot be reached, deterministic synthetic tracks
and coarse built-in land polygons are used. Synthetic mode is prominently
labeled and is intended only for timing/layout previews.

Recommended install
-------------------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm pyshp

Quick preview render
--------------------
    CYCLONE_SHORT_QUICK=1 python every_tropical_cyclone_on_earth_this_year.py

Force offline fixture mode
--------------------------
    CYCLONE_SHORT_OFFLINE=1 python every_tropical_cyclone_on_earth_this_year.py

Outputs
-------
- final vertical MP4 with generated ambient audio when ffmpeg is available
- silent MP4 fallback
- SRT subtitle sidecar
- preview PNG frames
- CSV export of selected-year track points
- JSON summary and source notes
- cached IBTrACS CSV and Natural Earth land geometry

Primary sources
---------------
- NOAA/NCEI IBTrACS: https://www.ncei.noaa.gov/products/international-best-track-archive
- IBTrACS v04r01 CSV directory:
  https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/
- Natural Earth land polygons: https://www.naturalearthdata.com/
"""

import calendar
import io
import json
import math
import os
import shutil
import subprocess
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
# Configuration and year selection
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("CYCLONE_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("CYCLONE_SHORT_OFFLINE", "0") == "1"


def resolve_year_window() -> Tuple[datetime, datetime, int, bool]:
    now = datetime.now(timezone.utc)
    override = os.environ.get("CYCLONE_YEAR", "").strip()
    if override:
        try:
            year = int(override)
        except ValueError as exc:
            raise ValueError("CYCLONE_YEAR must be a four-digit year, for example 2025") from exc
        if year < 1840 or year > now.year:
            raise ValueError(f"CYCLONE_YEAR must be between 1840 and {now.year}")
    else:
        year = now.year
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    next_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    end = min(now, next_year) if year == now.year else next_year
    return start, end, year, end < next_year


YEAR_START, YEAR_END, TARGET_YEAR, YEAR_IN_PROGRESS = resolve_year_window()
YEAR_LABEL = str(TARGET_YEAR)
YEAR_KEY = str(TARGET_YEAR)

OUTPUT_ROOT = Path(f"every_tropical_cyclone_on_earth_this_year_{YEAR_KEY}_output")
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
SCALE = OUT_W / 1080.0

COLORS = {

}

FULL_SHOT_PLAN = [

]

FULL_CAPTIONS = [

]

if QUICK_MODE:
    scale_time = float(CONFIG["duration_s"]) / 58.0
    SHOT_PLAN = [
        {"name": shot["name"], "start": shot["start"] * scale_time, "end": shot["end"] * scale_time}
        for shot in FULL_SHOT_PLAN
    ]
    CAPTIONS = [(a * scale_time, b * scale_time, text) for a, b, text in FULL_CAPTIONS]
else:
    SHOT_PLAN = FULL_SHOT_PLAN
    CAPTIONS = FULL_CAPTIONS

IBTRACS_URL = (

)
NATURAL_EARTH_LAND_URL = "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip"

BASIN_NAMES = {

}
BASIN_ORDER = ["NA", "EP", "WP", "NI", "SI", "SP", "SA"]


# -----------------------------------------------------------------------------
# Data model and helpers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackPoint:
    time_utc: datetime
    latitude: float
    longitude: float
    wind_kt: float
    pressure_mb: float

    @property
    def time_fraction(self) -> float:
        span = max((YEAR_END - YEAR_START).total_seconds(), 1.0)
        return clamp((self.time_utc - YEAR_START).total_seconds() / span)


@dataclass
class Storm:
    sid: str
    name: str
    basin: str
    points: List[TrackPoint]

    @property
    def start(self) -> datetime:
        return self.points[0].time_utc

    @property
    def end(self) -> datetime:
        return self.points[-1].time_utc

    @property
    def start_fraction(self) -> float:
        return self.points[0].time_fraction

    @property
    def max_wind_kt(self) -> float:
        vals = [p.wind_kt for p in self.points if np.isfinite(p.wind_kt)]
        return max(vals) if vals else float("nan")

    @property
    def min_pressure_mb(self) -> float:
        vals = [p.pressure_mb for p in self.points if np.isfinite(p.pressure_mb) and p.pressure_mb > 0]
        return min(vals) if vals else float("nan")

    @property
    def peak_point(self) -> TrackPoint:
        valid = [p for p in self.points if np.isfinite(p.wind_kt)]
        return max(valid, key=lambda p: p.wind_kt) if valid else self.points[len(self.points) // 2]

    @property
    def display_name(self) -> str:
        text = (self.name or "UNNAMED").strip().upper()
        if not text or text in {"NOT_NAMED", "UNNAMED", "NAN"}:
            return f"STORM {self.sid[-4:]}"
        return text[:24]


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
    for index, (start, end, text) in enumerate(captions, 1):
        lines.extend([str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def get_font(size: int, bold: bool = False, condensed: bool = False):
    candidates: List[str] = []
    if condensed and bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
            "DejaVuSansCondensed-Bold.ttf",
        ]
    if condensed:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
            "DejaVuSansCondensed.ttf",
        ]
    candidates += [
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
        stroke_fill=(0, 0, 0, min(235, fill[3])),
    )


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.75, 0.0, 1.0).astype(np.float32)


def request_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; EveryCycloneShort/1.0; educational visualization)",
            "Accept": "text/csv,application/zip,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def choose_numeric(row: pd.Series, candidates: Sequence[str]) -> float:
    for column in candidates:
        if column not in row.index:
            continue
        try:
            value = float(row[column])
            if np.isfinite(value) and value > 0:
                return value
        except Exception:
            continue
    return float("nan")


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# IBTrACS loading
# -----------------------------------------------------------------------------

WIND_COLUMNS = [
    "WMO_WIND",
    "USA_WIND",
    "TOKYO_WIND",
    "CMA_WIND",
    "HKO_WIND",
    "NEWDELHI_WIND",
    "REUNION_WIND",
    "BOM_WIND",
    "NADI_WIND",
    "WELLINGTON_WIND",
]
PRESSURE_COLUMNS = [
    "WMO_PRES",
    "USA_PRES",
    "TOKYO_PRES",
    "CMA_PRES",
    "HKO_PRES",
    "NEWDELHI_PRES",
    "REUNION_PRES",
    "BOM_PRES",
    "NADI_PRES",
    "WELLINGTON_PRES",
]


def parse_ibtracs_csv(path: Path) -> List[Storm]:
    df = pd.read_csv(path, skiprows=[1], low_memory=False)
    if "ISO_TIME" not in df.columns or "SID" not in df.columns:
        raise RuntimeError("IBTrACS CSV is missing expected ISO_TIME/SID columns")
    df["_time"] = pd.to_datetime(df["ISO_TIME"], errors="coerce", utc=True)
    df = df[(df["_time"] >= YEAR_START) & (df["_time"] < YEAR_END)].copy()
    if df.empty:
        return []

    storms: List[Storm] = []
    for sid, group in df.groupby("SID", sort=False):
        points: List[TrackPoint] = []
        group = group.sort_values("_time")
        names = [str(v).strip() for v in group.get("NAME", pd.Series(dtype=str)).tolist() if str(v).strip()]
        name = next((n for n in names if n.upper() not in {"NAN", "NOT_NAMED", "UNNAMED"}), names[0] if names else "UNNAMED")
        basin_values = [str(v).strip().upper() for v in group.get("BASIN", pd.Series(dtype=str)).tolist() if str(v).strip()]
        basin = max(set(basin_values), key=basin_values.count) if basin_values else "XX"

        for _, row in group.iterrows():
            try:
                lat = float(row["LAT"])
                lon = float(row["LON"])
                if not (np.isfinite(lat) and np.isfinite(lon)):
                    continue
                when = row["_time"].to_pydatetime()
                wind = choose_numeric(row, WIND_COLUMNS)
                pressure = choose_numeric(row, PRESSURE_COLUMNS)
                points.append(TrackPoint(when, lat, lon, wind, pressure))
            except Exception:
                continue
        if len(points) >= 2:
            # Remove duplicate timestamps that can appear in provisional data.
            dedup: Dict[datetime, TrackPoint] = {p.time_utc: p for p in points}
            points = sorted(dedup.values(), key=lambda p: p.time_utc)
            if len(points) >= 2:
                storms.append(Storm(str(sid), name, basin, points))
    storms.sort(key=lambda s: s.start)
    return storms


def load_storms() -> Tuple[List[Storm], str, List[str], Optional[Path]]:
    notes: List[str] = []
    cache_path = CACHE_ROOT / "ibtracs.last3years.list.v04r01.csv"

    if OFFLINE_MODE:
        notes.append("Offline mode requested with CYCLONE_SHORT_OFFLINE=1")
        return make_synthetic_storms(), "synthetic_procedural_fixture", notes, None

    try:
        payload = request_bytes(IBTRACS_URL, timeout=90)
        if len(payload) < 50_000:
            raise RuntimeError("IBTrACS download was unexpectedly small")
        cache_path.write_bytes(payload)
        storms = parse_ibtracs_csv(cache_path)
        if not storms:
            raise RuntimeError(f"No IBTrACS storms found for {TARGET_YEAR}")
        notes.append("Downloaded NOAA/NCEI IBTrACS last-3-years CSV")
        return storms, "noaa_ncei_ibtracs_v04r01", notes, cache_path
    except Exception as exc:
        notes.append(f"Live IBTrACS download failed: {exc}")

    if cache_path.exists() and cache_path.stat().st_size > 50_000:
        try:
            storms = parse_ibtracs_csv(cache_path)
            if storms:
                notes.append("Loaded cached IBTrACS CSV after live-download failure")
                return storms, "noaa_ncei_ibtracs_v04r01_cached", notes, cache_path
        except Exception as exc:
            notes.append(f"Cached IBTrACS parse failed: {exc}")

    notes.append("Using deterministic synthetic storm-track fixture")
    return make_synthetic_storms(), "synthetic_procedural_fixture", notes, None


# -----------------------------------------------------------------------------
# Deterministic synthetic preview tracks
# -----------------------------------------------------------------------------

BASIN_TEMPLATES: Dict[str, List[Tuple[float, float]]] = {
    "NA": [(-25, 11), (-38, 14), (-52, 18), (-64, 24), (-72, 31), (-68, 40), (-52, 48)],
    "EP": [(-98, 11), (-108, 14), (-120, 17), (-132, 20), (-143, 24)],
    "WP": [(132, 9), (139, 13), (145, 18), (142, 24), (134, 31), (126, 38)],
    "NI": [(88, 8), (86, 12), (84, 17), (82, 21)],
    "SI": [(78, -11), (70, -15), (62, -20), (55, -26), (49, -31)],
    "SP": [(170, -12), (176, -16), (-177, -20), (-170, -25), (-164, -31)],
    "SA": [(-31, -18), (-35, -24), (-39, -30)],
}


def interp_template(points: Sequence[Tuple[float, float]], u: float) -> Tuple[float, float]:
    u = clamp(u)
    pos = u * (len(points) - 1)
    idx = min(int(pos), len(points) - 2)
    f = pos - idx
    lon0 = float(points[idx][0])
    lon1 = float(points[idx + 1][0])
    delta = lon1 - lon0
    if delta > 180.0:
        delta -= 360.0
    elif delta < -180.0:
        delta += 360.0
    lon = lon0 + delta * f
    lon = ((lon + 180.0) % 360.0) - 180.0
    lat = lerp(points[idx][1], points[idx + 1][1], f)
    return lon, lat


def make_synthetic_storms() -> List[Storm]:
    rng = np.random.default_rng(TARGET_YEAR + 917)
    basin_counts = {"NA": 15, "EP": 14, "WP": 23, "NI": 7, "SI": 17, "SP": 11, "SA": 2}
    if QUICK_MODE:
        basin_counts = {key: max(2, int(value * 0.58)) for key, value in basin_counts.items()}
    span = max((YEAR_END - YEAR_START).total_seconds(), 86400.0)
    storms: List[Storm] = []
    serial = 0

    for basin, count in basin_counts.items():
        template = BASIN_TEMPLATES[basin]
        for _ in range(count):
            serial += 1
            if basin in {"SI", "SP"}:
                month_bias = rng.choice([0.05, 0.12, 0.20, 0.85, 0.92])
            elif basin == "NA":
                month_bias = rng.uniform(0.42, 0.83)
            elif basin == "EP":
                month_bias = rng.uniform(0.35, 0.78)
            elif basin == "WP":
                month_bias = rng.uniform(0.25, 0.88)
            elif basin == "NI":
                month_bias = rng.choice([rng.uniform(0.28, 0.46), rng.uniform(0.72, 0.88)])
            else:
                month_bias = rng.uniform(0.05, 0.95)
            start_seconds = clamp(float(month_bias), 0.0, 0.96) * span
            duration_days = float(rng.uniform(3.5, 12.0))
            point_count = int(rng.integers(12, 32))
            lateral = float(rng.normal(0.0, 4.0))
            meridional = float(rng.normal(0.0, 2.2))
            peak = float(np.clip(rng.gamma(2.0, 25.0) + 28.0, 30.0, 165.0))
            if rng.random() < 0.10:
                peak = float(rng.uniform(120, 175))
            points: List[TrackPoint] = []

            for j in range(point_count):
                u = j / max(point_count - 1, 1)
                lon, lat = interp_template(template, u)
                lon += lateral + float(rng.normal(0.0, 0.8)) + math.sin(u * math.pi) * float(rng.normal(0, 1.8))
                lat += meridional + float(rng.normal(0.0, 0.45))
                lon = ((lon + 180.0) % 360.0) - 180.0
                when = YEAR_START + timedelta(seconds=start_seconds + duration_days * 86400.0 * u)
                if when >= YEAR_END:
                    break
                envelope = math.sin(math.pi * clamp(u)) ** 1.45
                wind = max(20.0, 22.0 + (peak - 22.0) * envelope + float(rng.normal(0, 3.0)))
                pressure = 1010.0 - max(0.0, wind - 25.0) * 0.72 + float(rng.normal(0.0, 3.0))
                points.append(TrackPoint(when, lat, lon, wind, pressure))
            if len(points) >= 2:
                storms.append(Storm(f"SYN{TARGET_YEAR}{serial:03d}", f"PREVIEW {serial:02d}", basin, points))

    storms.sort(key=lambda s: s.start)
    return storms


# -----------------------------------------------------------------------------
# Land geometry
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
            notes.append("pyshp unavailable; using coarse built-in land polygons")
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
            pts = shape.points
            parts = list(shape.parts) + [len(pts)]
            for start, end in zip(parts[:-1], parts[1:]):
                poly = [(float(lon), float(lat)) for lon, lat in pts[start:end]]
                if len(poly) >= 3:
                    polygons.append(poly)
        notes.append("Loaded Natural Earth 110m land polygons")
        return polygons, "natural_earth_110m_land", notes
    except Exception as exc:
        notes.append(f"Natural Earth fallback: {exc}")
        return BUILTIN_LAND_POLYGONS, "built_in_coarse_land", notes


# -----------------------------------------------------------------------------
# Summaries and saved products
# -----------------------------------------------------------------------------

def basin_counts(storms: Sequence[Storm]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for storm in storms:
        result[storm.basin] = result.get(storm.basin, 0) + 1
    return result


def choose_strongest(storms: Sequence[Storm], maximum: int = 4) -> List[Storm]:
    ranked = sorted(
        storms,
        key=lambda s: s.max_wind_kt if np.isfinite(s.max_wind_kt) else -1.0,
        reverse=True,
    )
    return ranked[:maximum]


def storm_summary(storms: Sequence[Storm], source: str, land_source: str) -> Dict[str, Any]:
    strongest = choose_strongest(storms, 5)
    wind_values = [s.max_wind_kt for s in storms if np.isfinite(s.max_wind_kt)]
    return {
        "title": CONFIG["title"],
        "target_year": TARGET_YEAR,
        "year_start_utc": YEAR_START.isoformat(),
        "year_end_utc": YEAR_END.isoformat(),
        "year_in_progress": YEAR_IN_PROGRESS,
        "data_source": source,
        "land_source": land_source,
        "storm_count": len(storms),
        "track_point_count": sum(len(s.points) for s in storms),
        "basin_counts": basin_counts(storms),
        "maximum_selected_wind_kt": max(wind_values) if wind_values else None,
        "strongest_storms": [
            {
                "sid": s.sid,
                "name": s.display_name,
                "basin": s.basin,
                "max_wind_kt": None if not np.isfinite(s.max_wind_kt) else s.max_wind_kt,
                "min_pressure_mb": None if not np.isfinite(s.min_pressure_mb) else s.min_pressure_mb,
                "peak_time_utc": s.peak_point.time_utc.isoformat(),
                "peak_latitude": s.peak_point.latitude,
                "peak_longitude": s.peak_point.longitude,
            }
            for s in strongest
        ],
        "important_caveat": (
            "IBTrACS combines best-track information from multiple agencies. Wind averaging periods and operational "
            "practices differ among agencies, and recent provisional tracks may later be revised."
        ),
    }


def save_data_products(storms: Sequence[Storm], summary: Dict[str, Any], notes: Sequence[str]) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / f"storm_track_points_{TARGET_YEAR}.csv"
    summary_path = DATA_ROOT / f"storm_track_summary_{TARGET_YEAR}.json"
    rows: List[Dict[str, Any]] = []
    for storm in storms:
        for p in storm.points:
            rows.append(
                {
                    "sid": storm.sid,
                    "name": storm.display_name,
                    "basin": storm.basin,
                    "time_utc": p.time_utc.isoformat(),
                    "latitude": p.latitude,
                    "longitude": p.longitude,
                    "wind_kt": None if not np.isfinite(p.wind_kt) else p.wind_kt,
                    "pressure_mb": None if not np.isfinite(p.pressure_mb) else p.pressure_mb,
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "notes": list(notes),
                "source_urls": {
                    "ibtracs": "https://www.ncei.noaa.gov/products/international-best-track-archive",
                    "ibtracs_csv": IBTRACS_URL,
                    "natural_earth_land": NATURAL_EARTH_LAND_URL,
                },
                "fallback_warning": "synthetic_procedural_fixture is preview data, not observational data",
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

def intensity_color(wind_kt: float, alpha: int = 235) -> Tuple[int, int, int, int]:
    if not np.isfinite(wind_kt) or wind_kt < 34:
        return COLORS["weak"] + (alpha,)
    if wind_kt < 64:
        return COLORS["ts"] + (alpha,)
    if wind_kt < 83:
        return COLORS["hurr"] + (alpha,)
    if wind_kt < 96:
        return COLORS["strong"] + (alpha,)
    if wind_kt < 113:
        return COLORS["major"] + (alpha,)
    return COLORS["extreme"] + (alpha,)


def intensity_label(wind_kt: float) -> str:
    if not np.isfinite(wind_kt):
        return "WIND N/A"
    if wind_kt < 34:
        return "BELOW TROPICAL-STORM FORCE"
    if wind_kt < 64:
        return "TROPICAL-STORM FORCE"
    if wind_kt < 96:
        return "HURRICANE / TYPHOON FORCE"
    if wind_kt < 113:
        return "MAJOR INTENSITY BAND"
    return "EXTREME INTENSITY BAND"


def unwrapped_relative_longitudes(polygon: Sequence[Tuple[float, float]], center_lon: float = 0.0) -> List[float]:
    values: List[float] = []
    previous: Optional[float] = None
    for lon, _ in polygon:
        value = ((lon - center_lon + 180.0) % 360.0) - 180.0
        if previous is not None:
            while value - previous > 180:
                value -= 360
            while value - previous < -180:
                value += 360
        values.append(value)
        previous = value
    return values


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class StormTrackScene:
    def __init__(self, storms: List[Storm], land_polygons: List[List[Tuple[float, float]]], summary: Dict[str, Any]):
        self.storms = sorted(storms, key=lambda s: s.start)
        self.land_polygons = land_polygons
        self.summary = summary
        self.strongest = choose_strongest(self.storms, 4)
        self.particles = self._make_particles(int(CONFIG["background_particles"]), 71)
        self.dust = self._make_particles(int(CONFIG["dust_particles"]), 119)
        self.base_map = self._render_static_map()
        self.all_tracks = self._render_tracks(self.storms, 1.0, alpha_scale=0.88)
        self.timeline_layers = self._build_timeline_layers()
        self.basin_layers = {
            basin: self._render_tracks([s for s in self.storms if s.basin == basin], 1.0, alpha_scale=0.96)
            for basin in BASIN_ORDER
            if any(s.basin == basin for s in self.storms)
        }
        self.start_fractions = np.array([s.start_fraction for s in self.storms], dtype=float)

    @staticmethod
    def _make_particles(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.4, 1.8 if QUICK_MODE else 2.3)),
                "a": float(rng.uniform(10, 60)),
                "phase": float(rng.uniform(0, math.tau)),
                "speed": float(rng.uniform(1.0, 9.0)),
            }
            for _ in range(count)
        ]

    def project(self, lon: float, lat: float) -> Tuple[float, float]:
        x0 = float(CONFIG["map_margin_x"])
        x1 = OUT_W - x0
        y0 = float(CONFIG["map_top"])
        y1 = float(CONFIG["map_bottom"])
        x = x0 + (((lon + 180.0) % 360.0) / 360.0) * (x1 - x0)
        lat_clip = float(np.clip(lat, -68.0, 82.0))
        y = y0 + (82.0 - lat_clip) / 150.0 * (y1 - y0)
        return x, y

    def _render_static_map(self) -> Image.Image:
        arr = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
        top = np.array(COLORS["ocean_top"], dtype=float)
        bottom = np.array(COLORS["ocean_bottom"], dtype=float)
        for y in range(OUT_H):
            u = y / max(OUT_H - 1, 1)
            rgb = (top * (1.0 - u) + bottom * u).astype(np.uint8)
            arr[y, :, :3] = rgb
            arr[y, :, 3] = 255
        image = Image.fromarray(arr, "RGBA")
        draw = ImageDraw.Draw(image)
        x0, x1 = int(CONFIG["map_margin_x"]), OUT_W - int(CONFIG["map_margin_x"])
        y0, y1 = int(CONFIG["map_top"]), int(CONFIG["map_bottom"])

        for lat in range(-60, 76, 15):
            _, y = self.project(0, lat)
            draw.line((x0, y, x1, y), fill=COLORS["grid"] + (24,), width=1)
        for lon in range(-150, 181, 30):
            x, _ = self.project(lon, 0)
            draw.line((x, y0, x, y1), fill=COLORS["grid"] + (22,), width=1)

        for polygon in self.land_polygons:
            rel = unwrapped_relative_longitudes(polygon)
            lats = [lat for _, lat in polygon]
            for shift in (-360.0, 0.0, 360.0):
                pts: List[Tuple[float, float]] = []
                for relative, lat in zip(rel, lats):
                    lon = relative + shift
                    x = x0 + (lon + 180.0) / 360.0 * (x1 - x0)
                    lat_clip = float(np.clip(lat, -68, 82))
                    y = y0 + (82 - lat_clip) / 150.0 * (y1 - y0)
                    pts.append((x, y))
                if pts and max(p[0] for p in pts) >= x0 - 50 and min(p[0] for p in pts) <= x1 + 50:
                    draw.polygon(pts, fill=COLORS["land"] + (255,), outline=COLORS["land_edge"] + (102,))

        draw.rounded_rectangle((x0, y0, x1, y1), radius=max(10, int(26 * SCALE)), outline=COLORS["grid"] + (66,), width=1)
        _, eq_y = self.project(0, 0)
        draw.line((x0, eq_y, x1, eq_y), fill=COLORS["grid"] + (43,), width=1)
        return image

    def _segment_points(self, p1: TrackPoint, p2: TrackPoint) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if abs(p2.longitude - p1.longitude) > 180.0:
            return None
        return self.project(p1.longitude, p1.latitude), self.project(p2.longitude, p2.latitude)

    def _render_tracks(self, storms: Sequence[Storm], fraction: float, alpha_scale: float = 1.0) -> Image.Image:
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        core = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        cd = ImageDraw.Draw(core)
        core_w = max(1, int(round(3.2 * SCALE)))
        glow_w = max(core_w + 1, int(round(8.0 * SCALE)))

        for storm in storms:
            visible = [p for p in storm.points if p.time_fraction <= fraction + 1e-9]
            if len(visible) < 2:
                continue
            for p1, p2 in zip(visible[:-1], visible[1:]):
                seg = self._segment_points(p1, p2)
                if seg is None:
                    continue
                color = intensity_color(p2.wind_kt, int(225 * clamp(alpha_scale)))
                gd.line((*seg[0], *seg[1]), fill=color[:3] + (int(80 * alpha_scale),), width=glow_w)
                cd.line((*seg[0], *seg[1]), fill=color, width=core_w)

        glow = glow.filter(ImageFilter.GaussianBlur(max(2, int(5 * SCALE))))
        glow.alpha_composite(core)
        return glow

    def _build_timeline_layers(self) -> List[Image.Image]:
        layers: List[Image.Image] = []
        buckets = int(CONFIG["timeline_buckets"])
        for bucket in tqdm(range(1, buckets + 1), desc="Building storm-track timeline", leave=False):
            fraction = bucket / buckets
            layers.append(self._render_tracks(self.storms, fraction, alpha_scale=0.9))
        return layers

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, COLORS["dark"] + (255,))
        draw = ImageDraw.Draw(image)
        for p in self.particles:
            x = (p["x"] + math.sin(t * 0.11 + p["phase"]) * 8.0) % OUT_W
            y = (p["y"] + t * p["speed"] * 0.09) % OUT_H
            alpha = int(p["a"] * (0.55 + 0.45 * math.sin(t * 0.65 + p["phase"]) ** 2))
            r = p["r"]
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(190, 222, 235, alpha))
        return image

    def compose_map(self, track_layer: Optional[Image.Image] = None, opacity: float = 1.0) -> Image.Image:
        layer = self.base_map.copy()
        if track_layer is not None:
            layer.alpha_composite(track_layer)
        if opacity < 0.999:
            alpha = layer.getchannel("A").point(lambda v: int(v * opacity))
            layer.putalpha(alpha)
        return layer

    def draw_opening(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        map_layer = self.compose_map(None, opacity=0.30 + 0.62 * p)
        image.alpha_composite(map_layer)

        # Radar-like scan reveals the empty globe before the roll call begins.
        scan = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        sd = ImageDraw.Draw(scan)
        x = lerp(CONFIG["map_margin_x"], OUT_W - CONFIG["map_margin_x"], p)
        sd.rectangle((x - 2, CONFIG["map_top"], x + 2, CONFIG["map_bottom"]), fill=COLORS["ts"] + (135,))
        scan = scan.filter(ImageFilter.GaussianBlur(max(1, int(9 * SCALE))))
        image.alpha_composite(scan)

        draw_text(image, "EVERY", (OUT_W // 2, int(OUT_H * 0.68)), 45 if QUICK_MODE else 90,
                  COLORS["muted"] + (245,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, "TROPICAL CYCLONE", (OUT_W // 2, int(OUT_H * 0.76)), 39 if QUICK_MODE else 78,
                  COLORS["white"] + (255,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, f"ON EARTH // {TARGET_YEAR}", (OUT_W // 2, int(OUT_H * 0.84)), 19 if QUICK_MODE else 38,
                  COLORS["white"] + (245,), bold=True, condensed=True, anchor="ma", stroke=1)
        cutoff = YEAR_END.strftime("THROUGH %d %B %Y UTC").upper() if YEAR_IN_PROGRESS else "COMPLETE CALENDAR YEAR"
        draw_text(image, cutoff, (OUT_W // 2, int(OUT_H * 0.90)), 10 if QUICK_MODE else 20,
                  COLORS["muted"] + (220,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_storm_rollcall(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        if not self.storms:
            image.alpha_composite(self.compose_map(None))
            return
        p = clamp(shot_progress(t, shot))
        total = len(self.storms)
        slot = p * total
        idx = min(int(slot), total - 1)
        local = slot - idx
        storm = self.storms[idx]

        # Keep earlier-year tracks faintly accumulated using the time-bucket cache.
        base_fraction = max(0.0, storm.start_fraction - 0.002)
        bucket = min(int(base_fraction * len(self.timeline_layers)), len(self.timeline_layers) - 1)
        base = self.compose_map(self.timeline_layers[bucket], opacity=0.72)
        image.alpha_composite(base)

        # Draw this storm progressively so every unique SID receives a distinct beat.
        if len(storm.points) >= 2:
            frac = lerp(storm.start_fraction, max(pt.time_fraction for pt in storm.points), smoothstep(local))
            highlight = self._render_tracks([storm], frac, alpha_scale=1.0)
            image.alpha_composite(highlight)
            visible = [pt for pt in storm.points if pt.time_fraction <= frac + 1e-9]
            cursor = visible[-1] if visible else storm.points[0]
            x, y = self.project(cursor.longitude, cursor.latitude)
            pulse = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            pd = ImageDraw.Draw(pulse)
            r = (6 + 5 * math.sin(local * math.pi) ** 2) * SCALE
            color = intensity_color(cursor.wind_kt, 255)
            pd.ellipse((x-r, y-r, x+r, y+r), fill=color)
            pd.ellipse((x-r*2.4, y-r*2.4, x+r*2.4, y+r*2.4), outline=color[:3] + (150,), width=max(1, int(2*SCALE)))
            pulse = pulse.filter(ImageFilter.GaussianBlur(max(1, int(1.4*SCALE))))
            image.alpha_composite(pulse)

        # Cinematic lower-third card.
        left = int(48 * SCALE)
        top = int(1460 * SCALE)
        card = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        cd = ImageDraw.Draw(card)
        cd.rounded_rectangle((left, top, OUT_W-left, top+int(300*SCALE)), radius=max(12, int(28*SCALE)),
                             fill=(2, 7, 14, 205), outline=COLORS["grid"] + (78,), width=1)
        image.alpha_composite(card)

        color = intensity_color(storm.max_wind_kt, 255)
        draw_text(image, f"CYCLONE {idx+1:02d} / {total:02d}", (left+int(28*SCALE), top+int(34*SCALE)),
                  15 if QUICK_MODE else 30, color[:3] + (245,), bold=True, condensed=True, stroke=1)
        draw_text(image, storm.display_name, (left+int(28*SCALE), top+int(92*SCALE)),
                  29 if QUICK_MODE else 58, COLORS["white"] + (255,), bold=True, condensed=True, stroke=2)
        basin = BASIN_NAMES.get(storm.basin, storm.basin)
        wind = "WIND N/A" if not np.isfinite(storm.max_wind_kt) else f"PEAK {storm.max_wind_kt:.0f} KT  //  {storm.max_wind_kt*1.15078:.0f} MPH"
        draw_text(image, basin, (left+int(28*SCALE), top+int(172*SCALE)), 11 if QUICK_MODE else 22,
                  COLORS["muted"] + (235,), bold=True, condensed=True, stroke=1)
        draw_text(image, wind, (left+int(28*SCALE), top+int(215*SCALE)), 11 if QUICK_MODE else 22,
                  color[:3] + (245,), bold=True, condensed=True, stroke=1)
        dates = f"{storm.start.strftime('%d %b').upper()} — {storm.end.strftime('%d %b %Y').upper()}"
        draw_text(image, dates, (left+int(28*SCALE), top+int(258*SCALE)), 9 if QUICK_MODE else 18,
                  COLORS["muted"] + (215,), bold=True, condensed=True, stroke=1)

    def draw_basin_totals(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = shot_progress(t, shot)
        available = [b for b in BASIN_ORDER if b in self.basin_layers]
        if not available:
            image.alpha_composite(self.compose_map(self.all_tracks))
            return
        idx = min(int(p * len(available)), len(available) - 1)
        basin = available[idx]
        base = self.compose_map(self.all_tracks, opacity=0.30)
        image.alpha_composite(base)
        layer = self.basin_layers[basin].copy()
        image.alpha_composite(layer)
        count = sum(1 for s in self.storms if s.basin == basin)
        draw_text(image, BASIN_NAMES.get(basin, basin), (OUT_W//2, int(OUT_H*0.72)),
                  27 if QUICK_MODE else 54, COLORS["white"] + (255,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, f"{count} CYCLONE{'S' if count != 1 else ''}", (OUT_W//2, int(OUT_H*0.80)),
                  23 if QUICK_MODE else 46, COLORS["ts"] + (245,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, f"OF {len(self.storms)} GLOBAL RECORDS", (OUT_W//2, int(OUT_H*0.86)),
                  11 if QUICK_MODE else 22, COLORS["muted"] + (230,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_strongest(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = shot_progress(t, shot)
        if not self.strongest:
            image.alpha_composite(self.compose_map(self.all_tracks))
            return
        idx = min(int(p * len(self.strongest)), len(self.strongest) - 1)
        storm = self.strongest[idx]
        image.alpha_composite(self.compose_map(self.all_tracks, opacity=0.32))
        image.alpha_composite(self._render_tracks([storm], 1.0, alpha_scale=1.0))
        peak = storm.peak_point
        x, y = self.project(peak.longitude, peak.latitude)
        pulse = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        pd = ImageDraw.Draw(pulse)
        rr = (16 + 9 * math.sin(t*5.5)**2) * SCALE
        color = intensity_color(peak.wind_kt, 255)
        pd.ellipse((x-rr, y-rr, x+rr, y+rr), outline=color, width=max(1, int(3*SCALE)))
        image.alpha_composite(pulse.filter(ImageFilter.GaussianBlur(max(1, int(1.4*SCALE)))))
        draw_text(image, "AMONG THE STRONGEST", (int(58*SCALE), int(1455*SCALE)), 13 if QUICK_MODE else 26,
                  COLORS["muted"] + (235,), bold=True, condensed=True, stroke=1)
        draw_text(image, storm.display_name, (int(58*SCALE), int(1510*SCALE)), 34 if QUICK_MODE else 68,
                  COLORS["white"] + (255,), bold=True, condensed=True, stroke=2)
        wind = "WIND N/A" if not np.isfinite(storm.max_wind_kt) else f"{storm.max_wind_kt:.0f} KT  //  {storm.max_wind_kt*1.15078:.0f} MPH"
        draw_text(image, wind, (int(58*SCALE), int(1600*SCALE)), 18 if QUICK_MODE else 36,
                  color[:3] + (250,), bold=True, condensed=True, stroke=1)
        draw_text(image, BASIN_NAMES.get(storm.basin, storm.basin), (int(58*SCALE), int(1665*SCALE)),
                  11 if QUICK_MODE else 22, COLORS["muted"] + (225,), bold=True, condensed=True, stroke=1)

    def draw_all_storms(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        image.alpha_composite(self.compose_map(self.all_tracks))
        panel = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        pd = ImageDraw.Draw(panel)
        top = int(1465*SCALE)
        pd.rounded_rectangle((int(45*SCALE), top, OUT_W-int(45*SCALE), top+int(285*SCALE)),
                             radius=max(12,int(28*SCALE)), fill=(2,7,14,int(175+25*p)), outline=COLORS["grid"]+(72,), width=1)
        image.alpha_composite(panel)
        draw_text(image, f"{len(self.storms):,}", (OUT_W//2, top+int(92*SCALE)), 78 if QUICK_MODE else 156,
                  COLORS["white"]+(255,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, "TROPICAL CYCLONES", (OUT_W//2, top+int(180*SCALE)), 22 if QUICK_MODE else 44,
                  COLORS["white"]+(250,), bold=True, condensed=True, anchor="ma", stroke=1)
        cutoff = f"RECORDED IN IBTRACS THROUGH {YEAR_END.strftime('%d %b %Y').upper()}" if YEAR_IN_PROGRESS else f"IN {TARGET_YEAR}"
        draw_text(image, cutoff, (OUT_W//2, top+int(232*SCALE)), 10 if QUICK_MODE else 20,
                  COLORS["muted"]+(225,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_finale(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        image.alpha_composite(self.compose_map(self.all_tracks))
        image.alpha_composite(Image.new("RGBA", OUT_SIZE, (0,0,0,int(58*p))))
        draw_text(image, "EVERY CYCLONE", (OUT_W//2, int(OUT_H*0.72)), 34 if QUICK_MODE else 68,
                  COLORS["white"]+(255,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, "ONE EARTH", (OUT_W//2, int(OUT_H*0.79)), 28 if QUICK_MODE else 56,
                  COLORS["white"]+(255,), bold=True, condensed=True, anchor="ma", stroke=2)
        draw_text(image, f"{TARGET_YEAR} // {'YEAR TO DATE' if YEAR_IN_PROGRESS else 'COMPLETE YEAR'}", (OUT_W//2, int(OUT_H*0.86)),
                  14 if QUICK_MODE else 28, COLORS["ts"]+(245,), bold=True, condensed=True, anchor="ma", stroke=1)
        draw_text(image, "LATEST LOADED IBTRACS UPDATE", (OUT_W//2, int(OUT_H*0.91)), 9 if QUICK_MODE else 18,
                  COLORS["muted"]+(220,), bold=True, condensed=True, anchor="ma", stroke=1)

    def draw_header(self, image: Image.Image, name: str):
        if name == "opening":
            return
        draw_text(image, "EVERY TROPICAL CYCLONE", (int(48*SCALE), int(72*SCALE)), 20 if not QUICK_MODE else 10,
                  COLORS["white"]+(245,), bold=True, condensed=True, stroke=1)
        draw_text(image, "ON EARTH THIS YEAR", (int(48*SCALE), int(116*SCALE)), 36 if not QUICK_MODE else 18,
                  COLORS["white"]+(245,), bold=True, condensed=True, stroke=2)
        draw_text(image, CONFIG["subtitle"], (int(50*SCALE), int(170*SCALE)), 14 if not QUICK_MODE else 7,
                  COLORS["muted"]+(220,), bold=True, condensed=True, stroke=1)

    def draw_source_hud(self, image: Image.Image):
        source_text = "NOAA / NCEI IBTRACS V4R01"
        if str(self.summary.get("data_source", "")).startswith("synthetic"):
            source_text = "SYNTHETIC PREVIEW // NOT OBSERVATIONAL DATA"
        draw_text(image, source_text, (int(48 * SCALE), OUT_H - int(46 * SCALE)), 11 if not QUICK_MODE else 6, COLORS["muted"] + (185,), bold=True, condensed=True, stroke=1)

    def draw_legend(self, image: Image.Image, name: str):
        if name not in {"storm_rollcall", "all_storms", "finale"}:
            return
        items = [("<34 KT", 25), ("34+", 45), ("64+", 70), ("96+", 100), ("113+", 125)]
        x = int(55*SCALE)
        y = int(1395*SCALE)
        for label, wind in items:
            color = intensity_color(float(wind), 235)
            r = max(2, int(5*SCALE))
            ImageDraw.Draw(image).ellipse((x-r, y-r, x+r, y+r), fill=color)
            draw_text(image, label, (x+int(15*SCALE), y), 10 if not QUICK_MODE else 5,
                      COLORS["muted"]+(215,), bold=True, condensed=True, anchor="lm", stroke=1)
            x += int(190*SCALE)

    def draw_film_texture(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for p in self.dust:
            pulse = 0.5 + 0.5 * math.sin(t * 1.1 + p["phase"])
            if pulse < 0.68:
                continue
            x = (p["x"] + t * p["speed"] * 0.35) % OUT_W
            y = (p["y"] + math.sin(t * 0.5 + p["phase"]) * 4.0) % OUT_H
            length = (5 if QUICK_MODE else 10) + p["r"] * 4
            draw.line((x, y, x + length, y), fill=COLORS["ts"] + (int(12 * pulse),), width=1)
        offset = int((t * 43) % 8)
        for y in range(offset, OUT_H, 8):
            draw.line((0, y, OUT_W, y), fill=(120, 170, 190, 6), width=1)
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        name = str(shot["name"])
        image = self.background(t)

        if name == "opening":
            self.draw_opening(image, t, shot)
        elif name == "storm_rollcall":
            self.draw_storm_rollcall(image, t, shot)
        elif name == "basin_totals":
            self.draw_basin_totals(image, t, shot)
        elif name == "strongest":
            self.draw_strongest(image, t, shot)
        elif name == "all_storms":
            self.draw_all_storms(image, t, shot)
        else:
            self.draw_finale(image, t, shot)

        self.draw_header(image, name)
        self.draw_legend(image, name)
        self.draw_source_hud(image)
        self.draw_film_texture(image, t)

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        arr *= VIGNETTE[..., None]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img = ImageEnhance.Contrast(img).enhance(float(CONFIG["contrast"]))
        img = ImageEnhance.Color(img).enhance(float(CONFIG["saturation"]))
        arr = np.asarray(img, dtype=np.int16)
        rng = np.random.default_rng(int(t * 1000) + 331)
        grain = rng.normal(0.0, float(CONFIG["grain_strength"]), arr.shape[:2])[:, :, None]
        arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
        return arr


# -----------------------------------------------------------------------------
# Soundtrack and video rendering
# -----------------------------------------------------------------------------

def gaussian_envelope(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((times - center) / max(width, 1e-6)) ** 2)


def generate_ambient_soundtrack(path: Path) -> Path:
    sample_rate = int(CONFIG["soundtrack_sample_rate"])
    duration = float(CONFIG["duration_s"])
    count = int(round(sample_rate * duration))
    times = np.arange(count, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(TARGET_YEAR + 808)

    audio = np.zeros(count, dtype=np.float64)
    audio += 0.085 * np.sin(math.tau * 34.0 * times + 0.35 * np.sin(math.tau * 0.06 * times))
    audio += 0.045 * np.sin(math.tau * 52.0 * times + 1.4)
    audio += 0.020 * np.sin(math.tau * 91.0 * times + 0.5 * np.sin(math.tau * 0.11 * times))

    controls = rng.normal(0.0, 1.0, max(8, int(duration * 5)))
    slow_noise = np.interp(times, np.linspace(0.0, duration, len(controls)), controls)
    audio += 0.022 * slow_noise

    # Chronological pings through the sweep.
    sweep = next(s for s in SHOT_PLAN if s["name"] == "storm_rollcall")
    for fraction in np.linspace(0.04, 0.96, 20 if not QUICK_MODE else 8):
        center = lerp(float(sweep["start"]), float(sweep["end"]), float(fraction))
        env = gaussian_envelope(times, center, 0.055 if not QUICK_MODE else 0.08)
        audio += env * (0.035 * np.sin(math.tau * (280 + 140 * fraction) * times))

    # Scene-impact swells.
    for scene_name, strength in [("basin_totals", 0.08), ("strongest", 0.12), ("all_storms", 0.11), ("finale", 0.15)]:
        shot = next(s for s in SHOT_PLAN if s["name"] == scene_name)
        center = float(shot["start"]) + 0.25 * (float(shot["end"]) - float(shot["start"]))
        env = gaussian_envelope(times, center, 0.75 if not QUICK_MODE else 0.24)
        audio += env * strength * np.sin(math.tau * 44.0 * times)

    intro_x = np.clip(times / max(1.6, duration * 0.08), 0.0, 1.0)
    outro_x = np.clip((times - (duration - 1.2)) / 1.0, 0.0, 1.0)
    intro = intro_x * intro_x * (3.0 - 2.0 * intro_x)
    outro = 1.0 - outro_x * outro_x * (3.0 - 2.0 * outro_x)
    audio *= intro * outro
    peak = max(float(np.max(np.abs(audio))), 1e-9)
    audio = np.clip(audio / peak * 0.88, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

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


def render_video(scene: StormTrackScene) -> Path:
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
        for t in tqdm(times, desc="Rendering every-cyclone short"):
            writer.append_data(scene.render_frame(float(t)))

    generate_ambient_soundtrack(audio_path)
    if mux_audio(raw_video, audio_path, final_video):
        print("Final video with audio:", final_video.resolve())
        return final_video
    shutil.copyfile(raw_video, final_video)
    print("ffmpeg audio mux unavailable; copied silent video to:", final_video.resolve())
    return final_video



