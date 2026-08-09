from __future__ import annotations

"""
175 Years of Atlantic Hurricanes — cinematic YouTube Short renderer

Creates a vertical 1080x1920 short built around a living Atlantic map rather
than conventional charts. Hurricane tracks accumulate from 1851 through 2025,
with atmospheric motion, glowing intensity trails, era transitions, a
storm-by-storm cinematic montage, captions, preview frames, and a generated
ambient soundtrack.

The title is intentionally locked to the 175 inclusive Atlantic hurricane
seasons from 1851 through 2025.

Preferred live source
---------------------
NOAA National Hurricane Center Atlantic HURDAT2 best-track data. The script
first discovers the current 1851-2025 text file from the NHC data page, then
falls back to the official filename published in February 2026.

Scientific framing
------------------
- HURDAT2 supplies six-hourly best-track positions and intensities.
- Only storms that reached hurricane strength (>= 64 kt) are included in the
  main visual accumulation.
- Track segments retain the full lifecycle of those storms, while color and
  width respond to maximum sustained wind.
- The early record is less complete because storms could be missed before
  aircraft reconnaissance and continuous satellite monitoring.
- The animation therefore tells a history-of-observation story as well as a
  hurricane story; it is not a direct visualization of a climate trend.

Offline behavior
----------------
If NOAA or Natural Earth cannot be reached, deterministic procedural fixture
tracks and coarse built-in land polygons are used. The result remains useful
for timing and layout previews, but is clearly labeled as synthetic.

Recommended install
-------------------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm pyshp

Quick preview render
--------------------
    ATLANTIC_HURRICANES_SHORT_QUICK=1 python 175_years_atlantic_hurricanes_short.py

Force offline fixture mode
--------------------------
    ATLANTIC_HURRICANES_SHORT_OFFLINE=1 python 175_years_atlantic_hurricanes_short.py


"""

import json
import math
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
import wave
import zipfile
from dataclasses import dataclass
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
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("ATLANTIC_HURRICANES_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("ATLANTIC_HURRICANES_SHORT_OFFLINE", "0") == "1"

OUTPUT_ROOT = Path("atlantic_hurricanes_175_years_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
CACHE_ROOT = DATA_ROOT / "cache"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in [OUTPUT_ROOT, DATA_ROOT, CACHE_ROOT, PREVIEW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
START_YEAR = int(CONFIG["start_year"])
END_YEAR = int(CONFIG["end_year"])

COLORS = {
    "ocean_top": (2, 10, 22),
    "ocean_bottom": (1, 4, 12),
    "land": (15, 30, 39),
    "land_edge": (80, 127, 135),
    "grid": (84, 145, 159),
    "cyan": (83, 231, 255),
    "ice": (171, 236, 255),
    "gold": (255, 197, 91),
    "orange": (255, 132, 64),
    "red": (255, 73, 75),
    "magenta": (255, 75, 173),
    "white": (245, 249, 252),
    "muted": (158, 196, 207),
    "dark": (2, 7, 14),
}

# Full-duration shot plan. Quick mode scales every timestamp automatically.
FULL_SHOT_PLAN = [
    {"name": "awakening", "start": 0.0, "end": 8.2},
    {"name": "early_record", "start": 8.2, "end": 19.2},
    {"name": "century_unfolds", "start": 19.2, "end": 33.7},
    {"name": "satellite_age", "start": 33.7, "end": 45.8},
    {"name": "storm_montage", "start": 45.8, "end": 53.8},
    {"name": "all_tracks", "start": 53.8, "end": 58.0},
]

FULL_CAPTIONS = [
    (0.5, 7.7, "From sail-era logs to satellites, the Atlantic hurricane record spans 175 seasons."),
    (8.3, 18.8, "The archive begins in 1851. Early storms were found by ships, coastlines, and survival—not by a global observing system."),
    (19.3, 33.2, "Each luminous path is a storm that reached hurricane strength. Its track is reconstructed from six-hourly best-track positions."),
    (33.8, 45.3, "Aircraft reconnaissance and satellite coverage changed what could be detected, especially far from land."),
    (45.9, 53.4, "Some hurricanes crossed the entire ocean. Others intensified close to the coast. Every path tells a different story."),
    (53.9, 57.5, "175 years. Hundreds of hurricanes. One restless Atlantic."),
]

if QUICK_MODE:
    SCALE = float(CONFIG["duration_s"]) / 58.0
    SHOT_PLAN = [
        {"name": shot["name"], "start": shot["start"] * SCALE, "end": shot["end"] * SCALE}
        for shot in FULL_SHOT_PLAN
    ]
    CAPTIONS = [(a * SCALE, b * SCALE, text) for a, b, text in FULL_CAPTIONS]
else:
    SHOT_PLAN = FULL_SHOT_PLAN
    CAPTIONS = FULL_CAPTIONS

ERAS: List[Tuple[int, int, str]] = [
    (1851, 1900, "SAIL + COASTAL OBSERVATIONS"),
    (1901, 1943, "RADIO + EXPANDING SHIP REPORTS"),
    (1944, 1965, "AIRCRAFT RECONNAISSANCE ERA"),
    (1966, 1989, "EARLY SATELLITE ERA"),
    (1990, 2004, "MODERN GLOBAL OBSERVATION"),
    (2005, 2014, "HIGH-RESOLUTION DIGITAL ERA"),
    (2015, 2025, "TODAY'S BEST-TRACK ARCHIVE"),
]

# Official NHC location published for the dataset included through 2025.
KNOWN_HURDAT_URL = (
    "https://www.nhc.noaa.gov/data/hurdat/"
    "hurdat2-1851-2025-02272026.txt"
)
NHC_DATA_PAGE = "https://www.nhc.noaa.gov/data/"
NATURAL_EARTH_LAND_URL = (
    "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip"
)


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass
class TrackPoint:
    date: str
    time: str
    status: str
    lat: float
    lon: float
    wind_kt: float
    pressure_mb: float
    record_id: str = ""


@dataclass
class Storm:
    storm_id: str
    name: str
    year: int
    points: List[TrackPoint]

    @property
    def max_wind_kt(self) -> float:
        values = [point.wind_kt for point in self.points if np.isfinite(point.wind_kt)]
        return max(values) if values else 0.0

    @property
    def min_pressure_mb(self) -> float:
        values = [point.pressure_mb for point in self.points if point.pressure_mb > 0]
        return min(values) if values else float("nan")

    @property
    def reached_hurricane_strength(self) -> bool:
        return self.max_wind_kt >= 64.0 or any(point.status == "HU" for point in self.points)

    @property
    def reached_major_strength(self) -> bool:
        return self.max_wind_kt >= 96.0

    @property
    def display_name(self) -> str:
        cleaned = self.name.strip().upper()
        return "UNNAMED" if cleaned in {"", "UNNAMED"} else cleaned


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
    draw = ImageDraw.Draw(image)
    draw.text(
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
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
            stroke_width=2,
            stroke_fill=(0, 0, 0, 225),
        )
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += bbox[3] - bbox[1] + line_spacing


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.75, 0.0, 1.0).astype(np.float32)


def apply_grade(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array)
    image = ImageEnhance.Contrast(image).enhance(float(CONFIG["contrast"]))
    image = ImageEnhance.Color(image).enhance(float(CONFIG["saturation"]))
    return np.asarray(image)


def request_bytes(url: str, timeout: int = 35) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; AtlanticHurricanesShort/1.0; educational visualization)",
            "Accept": "text/html,text/plain,application/zip,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# NOAA HURDAT2 loading
# -----------------------------------------------------------------------------

def discover_hurdat_url() -> str:
    """Find the current official 1851-2025 Atlantic HURDAT2 text file."""
    try:
        html = request_bytes(NHC_DATA_PAGE).decode("utf-8", errors="ignore")
        pattern = r'href=["\']([^"\']*hurdat2-1851-2025-[^"\']+\.txt)["\']'
        matches = re.findall(pattern, html, flags=re.IGNORECASE)
        if matches:
            return urllib.parse.urljoin(NHC_DATA_PAGE, matches[-1])
    except Exception:
        pass
    return KNOWN_HURDAT_URL


def parse_coordinate(value: str) -> float:
    value = value.strip().upper()
    if not value:
        return float("nan")
    hemisphere = value[-1]
    number = float(value[:-1])
    if hemisphere in {"S", "W"}:
        number *= -1.0
    return number


def parse_hurdat2(text: str) -> List[Storm]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    storms: List[Storm] = []
    index = 0
    header_pattern = re.compile(r"^(AL\d{6})\s*,\s*([^,]+)\s*,\s*(\d+)\s*$")

    while index < len(lines):
        header = header_pattern.match(lines[index])
        if not header:
            index += 1
            continue
        storm_id, name, count_text = header.groups()
        count = int(count_text)
        year = int(storm_id[-4:])
        points: List[TrackPoint] = []
        for row in lines[index + 1:index + 1 + count]:
            fields = [field.strip() for field in row.split(",")]
            if len(fields) < 8:
                continue
            try:
                date, time_text, record_id, status = fields[:4]
                lat = parse_coordinate(fields[4])
                lon = parse_coordinate(fields[5])
                wind = float(fields[6]) if fields[6] else float("nan")
                pressure = float(fields[7]) if fields[7] else float("nan")
                points.append(
                    TrackPoint(
                        date=date,
                        time=time_text,
                        status=status.upper(),
                        lat=lat,
                        lon=lon,
                        wind_kt=wind,
                        pressure_mb=pressure,
                        record_id=record_id,
                    )
                )
            except Exception:
                continue
        if START_YEAR <= year <= END_YEAR and len(points) >= 2:
            storms.append(Storm(storm_id=storm_id, name=name.strip(), year=year, points=points))
        index += count + 1
    return storms


def load_hurdat2() -> Tuple[List[Storm], str, List[str], Optional[Path]]:
    notes: List[str] = []
    cache_path = CACHE_ROOT / "hurdat2_atlantic_1851_2025.txt"

    if OFFLINE_MODE:
        notes.append("Offline mode requested with ATLANTIC_HURRICANES_SHORT_OFFLINE=1")
        return make_procedural_fixture(), "synthetic_procedural_fixture", notes, None

    if cache_path.exists() and cache_path.stat().st_size > 100_000:
        try:
            text = cache_path.read_text(encoding="utf-8", errors="ignore")
            storms = parse_hurdat2(text)
            if len(storms) > 300:
                notes.append("Loaded cached NOAA HURDAT2 file")
                return storms, "noaa_nhc_hurdat2", notes, cache_path
        except Exception as exc:
            notes.append(f"Cached HURDAT2 could not be parsed: {exc}")

    try:
        url = discover_hurdat_url()
        payload = request_bytes(url, timeout=60)
        text = payload.decode("utf-8", errors="ignore")
        storms = parse_hurdat2(text)
        if len(storms) < 300:
            raise RuntimeError(f"Only {len(storms)} storms parsed; expected a full Atlantic archive")
        cache_path.write_bytes(payload)
        notes.append(f"Downloaded official NOAA HURDAT2 file: {url}")
        return storms, "noaa_nhc_hurdat2", notes, cache_path
    except Exception as exc:
        notes.append(f"NOAA HURDAT2 fallback: {exc}")
        return make_procedural_fixture(), "synthetic_procedural_fixture", notes, None


# -----------------------------------------------------------------------------
# Deterministic offline fixture
# -----------------------------------------------------------------------------

def bezier_point(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:
    u = 1.0 - t
    x = u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]
    return x, y


def make_procedural_fixture() -> List[Storm]:
    rng = np.random.default_rng(1751851)
    storms: List[Storm] = []
    serial = 1
    for year in range(START_YEAR, END_YEAR + 1):
        observation_factor = 0.55 if year < 1944 else (0.82 if year < 1966 else 1.0)
        count = int(np.clip(rng.poisson(3.1 * observation_factor) + 1, 1, 8))
        for local_index in range(count):
            start = (float(rng.uniform(-54, -18)), float(rng.uniform(8, 20)))
            recurves = rng.random() < 0.66
            if recurves:
                end = (float(rng.uniform(-65, -20)), float(rng.uniform(35, 55)))
                c1 = (float(rng.uniform(-73, -42)), float(rng.uniform(12, 25)))
                c2 = (float(rng.uniform(-82, -38)), float(rng.uniform(25, 46)))
            else:
                end = (float(rng.uniform(-98, -70)), float(rng.uniform(18, 34)))
                c1 = (float(rng.uniform(-68, -45)), float(rng.uniform(12, 24)))
                c2 = (float(rng.uniform(-92, -70)), float(rng.uniform(16, 34)))
            point_count = int(rng.integers(12, 30))
            peak = float(np.clip(rng.normal(92, 27), 64, 165))
            points: List[TrackPoint] = []
            for point_index in range(point_count):
                u = point_index / max(point_count - 1, 1)
                lon, lat = bezier_point(start, c1, c2, end, u)
                lat += float(rng.normal(0, 0.22))
                lon += float(rng.normal(0, 0.28))
                intensity_shape = math.sin(math.pi * u) ** 1.1
                wind = 30.0 + (peak - 30.0) * intensity_shape + float(rng.normal(0, 3.0))
                status = "HU" if wind >= 64 else ("TS" if wind >= 34 else "TD")
                points.append(
                    TrackPoint(
                        date=f"{year}0801",
                        time=f"{(point_index * 6) % 24:02d}00",
                        status=status,
                        lat=lat,
                        lon=lon,
                        wind_kt=wind,
                        pressure_mb=1010.0 - max(wind - 25.0, 0.0) * 0.72,
                    )
                )
            name = f"FIXTURE {serial:03d}"
            storms.append(Storm(storm_id=f"AL{local_index + 1:02d}{year}", name=name, year=year, points=points))
            serial += 1
    return storms


# -----------------------------------------------------------------------------
# Natural Earth land geometry
# -----------------------------------------------------------------------------

BUILTIN_LAND_POLYGONS: List[List[Tuple[float, float]]] = [
    # North America + Central America, deliberately coarse fallback.
    [
        (-105, 58), (-80, 58), (-62, 52), (-54, 47), (-61, 42), (-69, 43),
        (-74, 40), (-81, 31), (-80, 26), (-82, 24), (-88, 30), (-97, 26),
        (-105, 28), (-105, 58),
    ],
    [
        (-98, 26), (-90, 21), (-87, 16), (-83, 10), (-78, 8), (-77, 18),
        (-81, 25), (-88, 30), (-98, 26),
    ],
    # Northern South America.
    [
        (-82, 12), (-70, 13), (-60, 10), (-51, 5), (-48, 0), (-82, 0), (-82, 12),
    ],
    # Greenland.
    [
        (-73, 58), (-66, 66), (-48, 75), (-20, 74), (-16, 61), (-44, 58), (-73, 58),
    ],
    # Europe + northwest Africa.
    [
        (-11, 58), (10, 58), (10, 0), (-17, 0), (-18, 17), (-10, 30), (-9, 37),
        (-6, 43), (-11, 51), (-11, 58),
    ],
    # Cuba and Hispaniola as simplified islands.
    [(-85, 23), (-74, 20), (-75, 19), (-84, 21), (-85, 23)],
    [(-74, 20), (-68, 19), (-69, 18), (-73, 18), (-74, 20)],
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
# Summaries and saved data products
# -----------------------------------------------------------------------------

def storm_summary(storms: List[Storm], source: str, land_source: str) -> Dict[str, Any]:
    hurricanes = [storm for storm in storms if storm.reached_hurricane_strength]
    major = [storm for storm in hurricanes if storm.reached_major_strength]
    strongest = max(hurricanes, key=lambda storm: storm.max_wind_kt) if hurricanes else None
    annual = pd.DataFrame({"year": range(START_YEAR, END_YEAR + 1)})
    counts = pd.Series([storm.year for storm in hurricanes]).value_counts().to_dict()
    annual["hurricanes"] = annual["year"].map(counts).fillna(0).astype(int)
    return {
        "title_years_inclusive": END_YEAR - START_YEAR + 1,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "data_source": source,
        "land_source": land_source,
        "all_tropical_cyclones": int(len(storms)),
        "hurricanes": int(len(hurricanes)),
        "major_hurricanes": int(len(major)),
        "track_points_for_hurricanes": int(sum(len(storm.points) for storm in hurricanes)),
        "strongest_storm_in_archive": None if strongest is None else {
            "id": strongest.storm_id,
            "name": strongest.display_name,
            "year": strongest.year,
            "max_wind_kt": strongest.max_wind_kt,
            "min_pressure_mb": strongest.min_pressure_mb,
        },
        "annual_hurricane_counts": annual.to_dict(orient="records"),
        "important_caveat": (
            "Early Atlantic records are incomplete; changing observation systems affect detection. "
            "This visualization should not be read as a simple climate trend chart."
        ),
    }


def save_data_products(
    storms: List[Storm],
    summary: Dict[str, Any],
    notes: List[str],
) -> Tuple[Path, Path]:
    summary_path = DATA_ROOT / "atlantic_hurricane_summary_1851_2025.json"
    tracks_path = DATA_ROOT / "atlantic_hurricane_tracks_1851_2025.csv"

    rows: List[Dict[str, Any]] = []
    for storm in storms:
        if not storm.reached_hurricane_strength:
            continue
        for sequence, point in enumerate(storm.points):
            rows.append({
                "storm_id": storm.storm_id,
                "name": storm.display_name,
                "year": storm.year,
                "sequence": sequence,
                "date": point.date,
                "time": point.time,
                "status": point.status,
                "latitude": point.lat,
                "longitude": point.lon,
                "wind_kt": point.wind_kt,
                "pressure_mb": point.pressure_mb,
                "record_id": point.record_id,
                "storm_max_wind_kt": storm.max_wind_kt,
            })
    pd.DataFrame(rows).to_csv(tracks_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "notes": notes,
                "source_urls": {
                    "nhc_data_archive": NHC_DATA_PAGE,
                    "known_hurdat2_file": KNOWN_HURDAT_URL,
                    "historical_hurricane_tracks": "https://coast.noaa.gov/hurricanes/",
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
    return tracks_path, summary_path


# -----------------------------------------------------------------------------
# Visual mapping helpers
# -----------------------------------------------------------------------------

def intensity_color(wind_kt: float, alpha: int = 220) -> Tuple[int, int, int, int]:
    if wind_kt >= 137:
        return COLORS["magenta"] + (alpha,)
    if wind_kt >= 113:
        return COLORS["red"] + (alpha,)
    if wind_kt >= 96:
        return COLORS["orange"] + (alpha,)
    if wind_kt >= 83:
        return COLORS["gold"] + (alpha,)
    return COLORS["cyan"] + (alpha,)


def intensity_width(wind_kt: float) -> int:
    base = 1 if QUICK_MODE else 2
    if wind_kt >= 137:
        return base + (2 if QUICK_MODE else 4)
    if wind_kt >= 113:
        return base + (1 if QUICK_MODE else 3)
    if wind_kt >= 96:
        return base + (1 if QUICK_MODE else 2)
    if wind_kt >= 83:
        return base + 1
    return base


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
    zoom = max(1.0, zoom)
    new_size = (int(round(OUT_W * zoom)), int(round(OUT_H * zoom)))
    resized = layer.resize(new_size, Image.Resampling.BICUBIC)
    left = int((new_size[0] - OUT_W) / 2.0 - dx)
    top = int((new_size[1] - OUT_H) / 2.0 - dy)
    left = max(0, min(left, new_size[0] - OUT_W))
    top = max(0, min(top, new_size[1] - OUT_H))
    return resized.crop((left, top, left + OUT_W, top + OUT_H))


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class AtlanticHurricaneScene:
    def __init__(
        self,
        storms: List[Storm],
        source: str,
        land_polygons: List[List[Tuple[float, float]]],
        summary: Dict[str, Any],
    ):
        self.storms = storms
        self.hurricanes = sorted(
            [storm for storm in storms if storm.reached_hurricane_strength],
            key=lambda storm: (storm.year, storm.storm_id),
        )
        self.source = source
        self.land_polygons = land_polygons
        self.summary = summary
        self.map_box = (
            int(CONFIG["map_margin_x"]),
            int(CONFIG["map_top"]),
            OUT_W - int(CONFIG["map_margin_x"]),
            int(CONFIG["map_bottom"]),
        )
        self.background_particles = self._make_particles(int(CONFIG["background_particles"]), seed=1851)
        self.rain_particles = self._make_rain(int(CONFIG["rain_particles"]), seed=2025)
        self.map_base = self._render_static_map()
        self.projected_tracks = self._project_tracks()
        self.era_layers = self._build_era_layers()
        self.full_track_layer = self._make_track_layer(self.hurricanes, alpha_scale=0.72)
        self.montage_storms = self._select_montage_storms()
        self.year_counts = self._year_counts()
        # Small LRU cache: several video frames usually share the same displayed
        # year, so the completed historical accumulation should not be redrawn
        # for every frame. Keeping only a few full-resolution RGBA layers avoids
        # excessive memory use during 1080x1920 rendering.
        self.year_layer_cache: Dict[int, Tuple[Image.Image, str]] = {}
        self.year_layer_cache_order: List[int] = []
        self.year_layer_cache_limit = 6 if QUICK_MODE else 4

    def _make_particles(self, count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.5, 2.4 if not QUICK_MODE else 1.6)),
                "a": float(rng.uniform(8, 48)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "speed": float(rng.uniform(1.0, 7.0)),
            }
            for _ in range(count)
        ]

    def _make_rain(self, count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(-OUT_W * 0.2, OUT_W * 1.2)),
                "y": float(rng.uniform(-OUT_H, OUT_H)),
                "length": float(rng.uniform(18, 70) * OUT_W / 1080.0),
                "speed": float(rng.uniform(210, 620) * OUT_H / 1920.0),
                "alpha": float(rng.uniform(12, 48)),
            }
            for _ in range(count)
        ]

    def project(self, lon: float, lat: float) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.map_box
        lon_min = float(CONFIG["map_lon_min"])
        lon_max = float(CONFIG["map_lon_max"])
        lat_min = float(CONFIG["map_lat_min"])
        lat_max = float(CONFIG["map_lat_max"])
        x = x0 + (lon - lon_min) / (lon_max - lon_min) * (x1 - x0)
        y = y1 - (lat - lat_min) / (lat_max - lat_min) * (y1 - y0)
        return x, y

    def _polygon_intersects_view(self, polygon: List[Tuple[float, float]]) -> bool:
        if not polygon:
            return False
        lons = [point[0] for point in polygon]
        lats = [point[1] for point in polygon]
        return not (
            max(lons) < float(CONFIG["map_lon_min"])
            or min(lons) > float(CONFIG["map_lon_max"])
            or max(lats) < float(CONFIG["map_lat_min"])
            or min(lats) > float(CONFIG["map_lat_max"])
        )

    def _render_static_map(self) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = self.map_box

        # Faint ocean panel and latitude/longitude grid.
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=22 if not QUICK_MODE else 11,
            fill=(2, 13, 27, 184),
            outline=(96, 177, 186, 58),
            width=1,
        )
        for lon in range(-100, 11, 10):
            gx, _ = self.project(float(lon), 0.0)
            draw.line((gx, y0, gx, y1), fill=COLORS["grid"] + (24,), width=1)
        for lat in range(10, 60, 10):
            _, gy = self.project(-105.0, float(lat))
            draw.line((x0, gy, x1, gy), fill=COLORS["grid"] + (24,), width=1)

        land_fill = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        land_draw = ImageDraw.Draw(land_fill)
        for polygon in self.land_polygons:
            if not self._polygon_intersects_view(polygon):
                continue
            projected = [self.project(lon, lat) for lon, lat in polygon]
            if len(projected) >= 3:
                land_draw.polygon(projected, fill=COLORS["land"] + (245,))
                land_draw.line(projected + [projected[0]], fill=COLORS["land_edge"] + (125,), width=1)
        image.alpha_composite(land_fill)

        # Basin label integrated into the map instead of an axis-heavy chart.
        draw_text(
            image,
            "NORTH ATLANTIC BASIN",
            (x0 + (16 if QUICK_MODE else 30), y1 - (22 if QUICK_MODE else 42)),
            size=10 if QUICK_MODE else 19,
            fill=COLORS["muted"] + (145,),
            bold=True,
            condensed=True,
            stroke=1,
        )
        return image

    def _project_tracks(self) -> Dict[str, List[Tuple[float, float, float]]]:
        projected: Dict[str, List[Tuple[float, float, float]]] = {}
        for storm in self.hurricanes:
            points: List[Tuple[float, float, float]] = []
            for point in storm.points:
                if not np.isfinite(point.lon) or not np.isfinite(point.lat):
                    continue
                x, y = self.project(point.lon, point.lat)
                points.append((x, y, float(point.wind_kt)))
            projected[storm.storm_id] = points
        return projected

    def _draw_storm_track(
        self,
        core: Image.Image,
        glow: Image.Image,
        storm: Storm,
        reveal: float = 1.0,
        alpha_scale: float = 1.0,
        head: bool = False,
    ) -> Optional[Tuple[float, float, float]]:
        points = self.projected_tracks.get(storm.storm_id, [])
        if len(points) < 2:
            return None
        visible_count = max(2, int(math.ceil(len(points) * clamp(reveal))))
        points = points[:visible_count]
        core_draw = ImageDraw.Draw(core)
        glow_draw = ImageDraw.Draw(glow)
        alpha_scale = clamp(alpha_scale, 0.0, 1.5)

        for p0, p1 in zip(points[:-1], points[1:]):
            x0, y0, wind0 = p0
            x1, y1, wind1 = p1
            wind = max(wind0, wind1)
            color = intensity_color(wind, alpha=int(220 * min(alpha_scale, 1.0)))
            width = intensity_width(wind)
            glow_draw.line((x0, y0, x1, y1), fill=color[:3] + (int(105 * alpha_scale),), width=width * 4)
            core_draw.line((x0, y0, x1, y1), fill=color, width=width)

        if head:
            hx, hy, hwind = points[-1]
            radius = (4 if QUICK_MODE else 8) + intensity_width(hwind)
            glow_draw.ellipse((hx - radius * 2, hy - radius * 2, hx + radius * 2, hy + radius * 2), fill=intensity_color(hwind, int(100 * alpha_scale)))
            core_draw.ellipse((hx - radius, hy - radius, hx + radius, hy + radius), fill=COLORS["white"] + (245,), outline=intensity_color(hwind, 255), width=2)
            return hx, hy, hwind
        return points[-1]

    def _make_track_layer(self, storms: Iterable[Storm], alpha_scale: float = 1.0) -> Image.Image:
        core = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        for storm in storms:
            self._draw_storm_track(core, glow, storm, reveal=1.0, alpha_scale=alpha_scale, head=False)
        glow = glow.filter(ImageFilter.GaussianBlur(6 if not QUICK_MODE else 3))
        glow.alpha_composite(core)
        return glow

    def _build_era_layers(self) -> List[Tuple[Tuple[int, int, str], Image.Image, List[Storm]]]:
        layers: List[Tuple[Tuple[int, int, str], Image.Image, List[Storm]]] = []
        for era in ERAS:
            start, end, _ = era
            era_storms = [storm for storm in self.hurricanes if start <= storm.year <= end]
            layers.append((era, self._make_track_layer(era_storms, alpha_scale=0.78), era_storms))
        return layers

    def _select_montage_storms(self) -> List[Storm]:
        windows = [
            (1851, 1939),
            (1940, 1969),
            (1970, 1989),
            (1990, 2004),
            (2005, 2014),
            (2015, 2025),
        ]
        selected: List[Storm] = []
        for start, end in windows:
            candidates = [
                storm for storm in self.hurricanes
                if start <= storm.year <= end and storm.display_name != "UNNAMED"
            ]
            if not candidates:
                candidates = [storm for storm in self.hurricanes if start <= storm.year <= end]
            if candidates:
                selected.append(max(candidates, key=lambda storm: (storm.max_wind_kt, len(storm.points))))
        return selected

    def _year_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {year: 0 for year in range(START_YEAR, END_YEAR + 1)}
        for storm in self.hurricanes:
            counts[storm.year] = counts.get(storm.year, 0) + 1
        return counts

    def background(self, t: float) -> Image.Image:
        # Vertical ocean gradient.
        yy = np.linspace(0.0, 1.0, OUT_H, dtype=np.float32)[:, None, None]
        top = np.array(COLORS["ocean_top"], dtype=np.float32)[None, None, :]
        bottom = np.array(COLORS["ocean_bottom"], dtype=np.float32)[None, None, :]
        base = top * (1.0 - yy) + bottom * yy
        base = np.repeat(base, OUT_W, axis=1)
        image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")

        # Slowly moving atmospheric haze.
        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        haze_draw = ImageDraw.Draw(haze)
        cloud_specs = [
            (0.18, 0.28, 0.40, (20, 89, 107), 25),
            (0.78, 0.38, 0.46, (33, 59, 108), 20),
            (0.54, 0.75, 0.42, (13, 79, 99), 18),
        ]
        for index, (fx, fy, fr, color, alpha) in enumerate(cloud_specs):
            cx = OUT_W * fx + math.sin(t * 0.10 + index) * OUT_W * 0.035
            cy = OUT_H * fy + math.cos(t * 0.08 + index) * OUT_H * 0.025
            radius = OUT_W * fr
            haze_draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color + (alpha,))
        haze = haze.filter(ImageFilter.GaussianBlur(58 if not QUICK_MODE else 29))
        image.alpha_composite(haze)

        # Floating salt/spray particles.
        particle_layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(particle_layer)
        for particle in self.background_particles:
            x = (particle["x"] + t * particle["speed"]) % OUT_W
            y = particle["y"] + math.sin(t * 0.7 + particle["phase"]) * 8.0
            alpha = int(particle["a"] * (0.65 + 0.35 * math.sin(t * 1.4 + particle["phase"])))
            r = particle["r"]
            pd.ellipse((x - r, y - r, x + r, y + r), fill=COLORS["ice"] + (max(alpha, 0),))
        image.alpha_composite(particle_layer)
        return image

    def draw_rain(self, image: Image.Image, t: float, intensity: float):
        intensity = clamp(intensity)
        if intensity <= 0.02:
            return
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for drop in self.rain_particles:
            y = (drop["y"] + t * drop["speed"]) % (OUT_H + 200) - 100
            x = drop["x"] - y * 0.12
            length = drop["length"]
            draw.line(
                (x, y, x - length * 0.32, y + length),
                fill=COLORS["ice"] + (int(drop["alpha"] * intensity),),
                width=1,
            )
        image.alpha_composite(layer)

    def draw_hurricane_symbol(
        self,
        image: Image.Image,
        center: Tuple[float, float],
        radius: float,
        rotation: float,
        alpha: int = 230,
    ):
        cx, cy = center
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for ring_scale, ring_alpha in [(1.25, 26), (0.95, 42), (0.68, 65)]:
            rr = radius * ring_scale
            gd.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=COLORS["cyan"] + (ring_alpha,))
        glow = glow.filter(ImageFilter.GaussianBlur(18 if not QUICK_MODE else 9))
        image.alpha_composite(glow)

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for arm in range(5):
            points: List[Tuple[float, float]] = []
            for step in range(85):
                u = step / 84.0
                theta = rotation + arm * 2.0 * math.pi / 5.0 + u * math.pi * 1.75
                rr = radius * (0.10 + 0.88 * u)
                squash = 0.74 + 0.18 * math.sin(u * math.pi)
                points.append((cx + rr * math.cos(theta), cy + rr * math.sin(theta) * squash))
            draw.line(points, fill=COLORS["ice"] + (alpha,), width=5 if not QUICK_MODE else 2)
        eye = radius * 0.09
        draw.ellipse((cx - eye, cy - eye, cx + eye, cy + eye), fill=COLORS["dark"] + (255,), outline=COLORS["white"] + (210,), width=2)
        image.alpha_composite(layer)

    def draw_map_camera(self, image: Image.Image, map_layer: Image.Image, t: float, shot_name: str):
        zoom_targets = {
            "awakening": 1.07,
            "early_record": 1.035,
            "century_unfolds": 1.02,
            "satellite_age": 1.025,
            "storm_montage": 1.065,
            "all_tracks": 1.0,
        }
        zoom = zoom_targets.get(shot_name, 1.02) + 0.006 * math.sin(t * 0.22)
        dx = math.sin(t * 0.16) * OUT_W * 0.012
        dy = math.cos(t * 0.13) * OUT_H * 0.006
        transformed = zoom_and_shift(map_layer, zoom=zoom, dx=dx, dy=dy)
        image.alpha_composite(transformed)

    def compose_map_with_tracks(self, track_layer: Optional[Image.Image] = None) -> Image.Image:
        layer = self.map_base.copy()
        if track_layer is not None:
            layer.alpha_composite(track_layer)
        return layer

    def draw_awaken(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = shot_progress(t, shot)
        map_layer = self.map_base.copy()
        map_opacity = smoothstep((p - 0.28) / 0.45)
        if map_opacity > 0:
            alpha_composite_with_opacity(map_layer, self.full_track_layer, 0.06 * map_opacity)
        self.draw_map_camera(image, map_layer, t, "awakening")

        center = (OUT_W * 0.50, OUT_H * 0.43)
        radius = (152 if QUICK_MODE else 305) * (0.82 + 0.12 * math.sin(t * 0.55))
        self.draw_hurricane_symbol(image, center, radius, rotation=-t * 0.95, alpha=int(220 * (1.0 - smoothstep((p - 0.72) / 0.24))))

        count = int(round(lerp(1, END_YEAR - START_YEAR + 1, smootherstep(p))))
        draw_text(
            image,
            f"{count}",
            (OUT_W // 2, int(OUT_H * 0.655)),
            size=76 if QUICK_MODE else 158,
            fill=COLORS["white"] + (245,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=3 if not QUICK_MODE else 2,
        )
        draw_text(
            image,
            "SEASONS OF HURRICANE HISTORY",
            (OUT_W // 2, int(OUT_H * 0.735)),
            size=14 if QUICK_MODE else 28,
            fill=COLORS["gold"] + (235,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def layer_through_year(self, year: int, active_reveal: float = 1.0) -> Tuple[Image.Image, str]:
        # Cache the accumulation through the end of the preceding year. The
        # currently active year's storms remain dynamic because their tracks
        # are progressively revealed within the year.
        cached = self.year_layer_cache.get(year)
        if cached is not None:
            base_layer, era_label = cached
            if year in self.year_layer_cache_order:
                self.year_layer_cache_order.remove(year)
            self.year_layer_cache_order.append(year)
            layer = base_layer.copy()
        else:
            layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            era_label = ""
            for era, era_layer, era_storms in self.era_layers:
                start, end, label = era
                if year > end:
                    layer.alpha_composite(era_layer)
                    continue
                if start <= year <= end:
                    era_label = label
                    completed = [storm for storm in era_storms if storm.year < year]
                    if completed:
                        layer.alpha_composite(self._make_track_layer(completed, alpha_scale=0.78))
                    break
            self.year_layer_cache[year] = (layer.copy(), era_label)
            self.year_layer_cache_order.append(year)
            while len(self.year_layer_cache_order) > self.year_layer_cache_limit:
                oldest = self.year_layer_cache_order.pop(0)
                self.year_layer_cache.pop(oldest, None)

        current = [storm for storm in self.hurricanes if storm.year == year]
        core = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        for storm in current:
            self._draw_storm_track(core, glow, storm, reveal=active_reveal, alpha_scale=1.0, head=True)
        glow = glow.filter(ImageFilter.GaussianBlur(8 if not QUICK_MODE else 4))
        glow.alpha_composite(core)
        layer.alpha_composite(glow)
        return layer, era_label

    def draw_year_sweep(
        self,
        image: Image.Image,
        t: float,
        shot: Dict[str, Any],
        year_start: int,
        year_end: int,
        shot_name: str,
    ):
        p = smootherstep(shot_progress(t, shot))
        year_float = lerp(float(year_start), float(year_end) + 0.999, p)
        year = int(min(math.floor(year_float), year_end))
        within_year = year_float - math.floor(year_float)
        tracks, era_label = self.layer_through_year(year, active_reveal=within_year)
        map_layer = self.compose_map_with_tracks(tracks)
        self.draw_map_camera(image, map_layer, t, shot_name)

        # Cinematic year marker, not a graph.
        draw_text(
            image,
            str(year),
            (OUT_W // 2, int(OUT_H * 0.755)),
            size=70 if QUICK_MODE else 144,
            fill=COLORS["white"] + (248,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=3 if not QUICK_MODE else 2,
        )
        draw_text(
            image,
            era_label,
            (OUT_W // 2, int(OUT_H * 0.815)),
            size=11 if QUICK_MODE else 22,
            fill=COLORS["gold"] + (235,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )
        annual_count = self.year_counts.get(year, 0)
        draw_text(
            image,
            f"{annual_count} HURRICANE{'S' if annual_count != 1 else ''} IN THE ARCHIVE",
            (OUT_W // 2, int(OUT_H * 0.845)),
            size=10 if QUICK_MODE else 19,
            fill=COLORS["muted"] + (205,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_storm_montage(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = shot_progress(t, shot)
        count = max(len(self.montage_storms), 1)
        slot_float = p * count
        slot = min(int(slot_float), count - 1)
        local = slot_float - slot
        storm = self.montage_storms[slot]

        # Dim accumulation in the background, then isolate one storm.
        map_layer = self.map_base.copy()
        alpha_composite_with_opacity(map_layer, self.full_track_layer, 0.17)
        core = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        head = self._draw_storm_track(
            core,
            glow,
            storm,
            reveal=smootherstep(min(local / 0.74, 1.0)),
            alpha_scale=1.25,
            head=True,
        )
        glow = glow.filter(ImageFilter.GaussianBlur(12 if not QUICK_MODE else 6))
        glow.alpha_composite(core)
        map_layer.alpha_composite(glow)
        self.draw_map_camera(image, map_layer, t, "storm_montage")

        fade = smoothstep(local / 0.16) * (1.0 - smoothstep((local - 0.82) / 0.16))
        label_alpha = int(245 * fade)
        draw_text(
            image,
            storm.display_name,
            (OUT_W // 2, int(OUT_H * 0.735)),
            size=49 if QUICK_MODE else 98,
            fill=COLORS["white"] + (label_alpha,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=3 if not QUICK_MODE else 2,
        )
        draw_text(
            image,
            f"{storm.year}  //  PEAK {storm.max_wind_kt:.0f} KT",
            (OUT_W // 2, int(OUT_H * 0.795)),
            size=13 if QUICK_MODE else 27,
            fill=intensity_color(storm.max_wind_kt, label_alpha),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )
        if head is not None:
            hx, hy, _ = head
            self.draw_hurricane_symbol(
                image,
                (hx, hy),
                18 if QUICK_MODE else 36,
                rotation=-t * 1.8,
                alpha=int(170 * fade),
            )

    def draw_all_tracks(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = shot_progress(t, shot)
        map_layer = self.map_base.copy()
        alpha_composite_with_opacity(map_layer, self.full_track_layer, smoothstep(p / 0.48))
        self.draw_map_camera(image, map_layer, t, "all_tracks")

        # Radial pulse draws the eye into the basin without adding a chart.
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cx, cy = OUT_W * 0.49, OUT_H * 0.47
        for ring in range(3):
            phase = (p * 2.2 + ring / 3.0) % 1.0
            radius = lerp(50 if QUICK_MODE else 100, 240 if QUICK_MODE else 480, phase)
            alpha = int(65 * (1.0 - phase))
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=COLORS["cyan"] + (alpha,), width=2)
        image.alpha_composite(overlay)

        draw_text(
            image,
            "EVERY LINE IS A HURRICANE",
            (OUT_W // 2, int(OUT_H * 0.775)),
            size=24 if QUICK_MODE else 50,
            fill=COLORS["white"] + (245,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=2,
        )
        draw_text(
            image,
            "1851—2025",
            (OUT_W // 2, int(OUT_H * 0.825)),
            size=18 if QUICK_MODE else 36,
            fill=COLORS["gold"] + (240,),
            bold=True,
            condensed=True,
            anchor="ma",
            stroke=1,
        )

    def draw_top_titles(self, image: Image.Image, t: float, shot_name: str):
        intro_fade = smoothstep((t - (0.22 if not QUICK_MODE else 0.05)) / (0.85 if not QUICK_MODE else 0.18))
        intro_out = 1.0 - smoothstep((t - (6.3 if not QUICK_MODE else 1.30)) / (1.15 if not QUICK_MODE else 0.24))
        alpha = int(255 * intro_fade * intro_out)
        if alpha > 3:
            draw_text(
                image,
                "175 YEARS OF",
                (OUT_W // 2, 70 if QUICK_MODE else 138),
                size=29 if QUICK_MODE else 58,
                fill=COLORS["white"] + (alpha,),
                bold=True,
                condensed=True,
                anchor="ma",
                stroke=2,
            )
            draw_text(
                image,
                "ATLANTIC HURRICANES",
                (OUT_W // 2, 105 if QUICK_MODE else 207),
                size=29 if QUICK_MODE else 58,
                fill=COLORS["white"] + (alpha,),
                bold=True,
                condensed=True,
                anchor="ma",
                stroke=2,
            )
            draw_text(
                image,
                CONFIG["subtitle"],
                (OUT_W // 2, 141 if QUICK_MODE else 279),
                size=11 if QUICK_MODE else 22,
                fill=COLORS["cyan"] + (min(alpha, 235),),
                bold=True,
                condensed=True,
                anchor="ma",
                stroke=1,
            )

        labels = {
            "awakening": "THE ATLANTIC REMEMBERS",
            "early_record": "THE EARLY RECORD // 1851—1900",
            "century_unfolds": "A CENTURY OF TRACKS // 1901—1965",
            "satellite_age": "THE SATELLITE AGE // 1966—2025",
            "storm_montage": "SIX ERAS // SIX EXTREME TRACKS",
            "all_tracks": "THE COMPLETE 175-SEASON VIEW",
        }
        if t > (6.0 if not QUICK_MODE else 1.22):
            draw_text(
                image,
                labels[shot_name],
                (OUT_W // 2, 52 if QUICK_MODE else 102),
                size=11 if QUICK_MODE else 22,
                fill=COLORS["muted"] + (215,),
                bold=True,
                condensed=True,
                anchor="ma",
                stroke=1,
            )

    def draw_source_hud(self, image: Image.Image):
        live = self.source == "noaa_nhc_hurdat2"
        label = "SOURCE // NOAA NHC HURDAT2" if live else "PREVIEW // SYNTHETIC TRACK FIXTURE"
        color = COLORS["cyan"] if live else COLORS["gold"]
        draw_text(
            image,
            label,
            (OUT_W - (22 if QUICK_MODE else 44), OUT_H - (28 if QUICK_MODE else 56)),
            size=8 if QUICK_MODE else 16,
            fill=color + (220,),
            bold=True,
            condensed=True,
            anchor="ra",
            stroke=1,
        )

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        y0 = OUT_H - (145 if QUICK_MODE else 286)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle(
            (22 if QUICK_MODE else 44, y0, OUT_W - (22 if QUICK_MODE else 44), y0 + (76 if QUICK_MODE else 150)),
            radius=14 if QUICK_MODE else 28,
            fill=(1, 6, 14, 183),
            outline=(89, 190, 204, 65),
            width=1,
        )
        image.alpha_composite(panel)
        draw_wrapped_text(
            image,
            text,
            (34 if QUICK_MODE else 68, y0 + (15 if QUICK_MODE else 30)),
            OUT_W - (68 if QUICK_MODE else 136),
            size=14 if QUICK_MODE else 29,
            fill=COLORS["white"] + (245,),
            line_spacing=4 if QUICK_MODE else 8,
        )

    def draw_film_texture(self, image: Image.Image, t: float, rain_intensity: float):
        self.draw_rain(image, t, rain_intensity)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Fine horizontal scan texture.
        offset = int((t * 47.0) % 9)
        for y in range(offset, OUT_H, 9):
            draw.line((0, y, OUT_W, y), fill=(160, 220, 230, 7), width=1)

        # Occasional lightning-like flash, deterministic and brief.
        flash = 0.0
        for strike_time in ([18.4, 33.7, 45.8, 53.8] if not QUICK_MODE else [3.8, 7.0, 9.5, 11.1]):
            flash = max(flash, math.exp(-((t - strike_time) / (0.075 if not QUICK_MODE else 0.04)) ** 2))
        if flash > 0.01:
            draw.rectangle((0, 0, OUT_W, OUT_H), fill=COLORS["ice"] + (int(65 * flash),))
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        shot_name = str(shot["name"])
        image = self.background(t)

        if shot_name == "awakening":
            self.draw_awaken(image, t, shot)
        elif shot_name == "early_record":
            self.draw_year_sweep(image, t, shot, 1851, 1900, shot_name)
        elif shot_name == "century_unfolds":
            self.draw_year_sweep(image, t, shot, 1901, 1965, shot_name)
        elif shot_name == "satellite_age":
            self.draw_year_sweep(image, t, shot, 1966, 2025, shot_name)
        elif shot_name == "storm_montage":
            self.draw_storm_montage(image, t, shot)
        elif shot_name == "all_tracks":
            self.draw_all_tracks(image, t, shot)

        self.draw_top_titles(image, t, shot_name)
        self.draw_caption(image, t)
        self.draw_source_hud(image)

        rain_by_shot = {
            "awakening": 0.30,
            "early_record": 0.15,
            "century_unfolds": 0.22,
            "satellite_age": 0.38,
            "storm_montage": 0.72,
            "all_tracks": 0.22,
        }
        self.draw_film_texture(image, t, rain_by_shot.get(shot_name, 0.2))

        array = np.asarray(image.convert("RGB"))
        array = apply_grade(array)

        # Deterministic film grain.
        frame_index = int(round(t * int(CONFIG["fps"])))
        rng = np.random.default_rng(9000 + frame_index)
        grain = rng.normal(0.0, float(CONFIG["grain_strength"]), size=array.shape[:2] + (1,))
        array = np.clip(array.astype(np.float32) + grain, 0, 255)
        array *= VIGNETTE[..., None]

        fade_in = smoothstep(t / (0.85 if not QUICK_MODE else 0.18))
        fade_out = 1.0 - smoothstep((t - (float(CONFIG["duration_s"]) - (1.15 if not QUICK_MODE else 0.24))) / (1.0 if not QUICK_MODE else 0.20))
        return np.clip(array * fade_in * fade_out, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Audio generation and video rendering
# -----------------------------------------------------------------------------

def gaussian_envelope(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((times - center) / max(width, 1e-6)) ** 2)


def generate_ambient_soundtrack(path: Path) -> Path:
    sample_rate = int(CONFIG["soundtrack_sample_rate"])
    duration = float(CONFIG["duration_s"])
    sample_count = int(round(sample_rate * duration))
    times = np.arange(sample_count, dtype=np.float32) / sample_rate
    rng = np.random.default_rng(1752025)

    # Slow sub-bass and distant harmonic drone.
    drone = 0.12 * np.sin(2 * math.pi * 42.0 * times)
    drone += 0.055 * np.sin(2 * math.pi * 63.0 * times + 0.7)
    drone *= 0.72 + 0.28 * np.sin(2 * math.pi * 0.055 * times + 1.1)

    # Wind built from low-rate noise interpolation, avoiding heavy filtering.
    control_rate = 55.0
    control_times = np.arange(int(duration * control_rate) + 3, dtype=np.float32) / control_rate
    control_noise = rng.normal(0.0, 1.0, size=len(control_times)).astype(np.float32)
    wind = np.interp(times, control_times, control_noise).astype(np.float32)
    wind *= 0.045 + 0.018 * np.sin(2 * math.pi * 0.083 * times)
    hiss = rng.normal(0.0, 0.010, size=sample_count).astype(np.float32)

    audio = drone + wind + hiss

    transitions = [shot["start"] for shot in SHOT_PLAN[1:]]
    for transition in transitions:
        dt = times - float(transition)
        env = gaussian_envelope(times, float(transition), 0.34 if not QUICK_MODE else 0.08)
        chirp = np.sin(2 * math.pi * (95.0 * dt + 22.0 * dt * dt))
        audio += 0.085 * env * chirp

    # Low thunder impacts near major visual transitions.
    strike_times = [0.8, 19.2, 33.7, 45.8, 53.8]
    if QUICK_MODE:
        strike_times = [value * float(CONFIG["duration_s"]) / 58.0 for value in strike_times]
    for strike in strike_times:
        dt = np.maximum(times - strike, 0.0)
        env = np.exp(-dt / (0.72 if not QUICK_MODE else 0.17)) * (times >= strike)
        boom = np.sin(2 * math.pi * (31.0 * dt + 6.0 * dt * dt))
        audio += 0.12 * env * boom

    # Clean fade and conservative normalization.
    fade_duration = 0.9 if not QUICK_MODE else 0.18
    fade_samples = max(1, int(sample_rate * fade_duration))
    fade = np.ones(sample_count, dtype=np.float32)
    fade[:fade_samples] = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade[-fade_samples:] = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    audio *= fade
    peak = max(float(np.max(np.abs(audio))), 1e-6)
    audio = np.clip(audio / peak * 0.78, -1.0, 1.0)

    # Slight stereo motion from a delayed right channel.
    delay = max(1, int(sample_rate * 0.012))
    right = np.roll(audio, delay)
    right[:delay] = 0.0
    stereo = np.stack([audio, right], axis=1)
    pcm = (stereo * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
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
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("Audio mux failed:", result.stderr[-1200:])
        return False
    return True


def render_video(scene: AtlanticHurricaneScene) -> Path:
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    print("Subtitle sidecar:", srt_path.resolve())

    silent_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_silent.mp4"
    final_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    soundtrack = OUTPUT_ROOT / f"{CONFIG['output_basename']}_ambient.wav"
    frame_count = int(round(float(CONFIG["duration_s"]) * int(CONFIG["fps"])))
    times = np.arange(frame_count, dtype=float) / int(CONFIG["fps"])

    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(
        silent_video,
        fps=int(CONFIG["fps"]),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times, desc="Rendering hurricane short"):
            writer.append_data(scene.render_frame(float(t)))

    print("Generating atmospheric soundtrack ...")
    generate_ambient_soundtrack(soundtrack)
    if mux_audio(silent_video, soundtrack, final_video):
        print("Final video with audio:", final_video.resolve())
        return final_video

    shutil.copyfile(silent_video, final_video)
    print("ffmpeg unavailable; copied silent video to:", final_video.resolve())
    return final_video


