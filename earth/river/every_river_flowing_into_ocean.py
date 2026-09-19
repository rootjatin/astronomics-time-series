from __future__ import annotations

"""
Every River Flowing Into the Ocean
==================================

A cinematic vertical YouTube Short renderer that lights up a global river map
and explains how streamflow carries water from land back toward the ocean.

The structure matches the data-driven Shorts workflow used in the companion
scripts: live-source download with caching, an offline fallback, 9:16 animated
scenes, captions/SRT, preview frames, a contact sheet, CSV/JSON data output,
and H.264 MP4 export.

What the video shows
--------------------
- A global river-network visualization from Natural Earth's mapped single-line
  drainages (10m in final quality; 110m in quick-preview mode for speed).
- A progressive "light-up" of mapped river lines across the continents.
- Major river-mouth examples around the world.
- A simple watershed / gravity diagram showing runoff joining streams and rivers.
- A crucial caveat: not every real-world river reaches the ocean; some drain to
  closed basins, disappear into the ground, evaporate, or are heavily diverted.

Important interpretation note
-----------------------------
The title is a visual hook. The map is NOT literally every stream or channel on
Earth. It renders every river feature available in the selected Natural Earth
river layer after excluding lake-centerline records. Natural Earth is a global
cartographic dataset, not a complete hydrologic census.

Sources
-------
Natural Earth — Rivers + lake centerlines:
    https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-rivers-lake-centerlines/
USGS — Streamflow and the Water Cycle:
    https://www.usgs.gov/water-science-school/science/streamflow-and-water-cycle
NASA — The Water Cycle:
    https://science.nasa.gov/earth/earth-observatory/the-water-cycle/

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg requests tqdm pyshp

Run final quality
-----------------
    python every_river_flowing_into_the_ocean_short.py

Run quick preview
-----------------
    RIVER_SHORT_QUICK=1 python every_river_flowing_into_the_ocean_short.py

Force live refresh
------------------
    RIVER_SHORT_REFRESH=1 python every_river_flowing_into_the_ocean_short.py

Force offline layout testing
----------------------------
    RIVER_SHORT_OFFLINE=1 RIVER_SHORT_QUICK=1 \
        python every_river_flowing_into_the_ocean_short.py
"""

import json
import math
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import requests
except Exception:
    requests = None

try:
    import shapefile  # pyshp
except Exception:
    shapefile = None


# =============================================================================
# Configuration
# =============================================================================

QUICK_MODE = os.environ.get("RIVER_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("RIVER_SHORT_OFFLINE", "0") == "1"
REFRESH = os.environ.get("RIVER_SHORT_REFRESH", "0") == "1"

OUTPUT_ROOT = Path("every_river_flowing_into_the_ocean_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
CACHE_ROOT = OUTPUT_ROOT / "cache"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT, CACHE_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

RIVER_SCALE = "110m" if QUICK_MODE else "10m"
LAND_SCALE = "110m" if QUICK_MODE else "50m"

CONFIG = {
    "width": 540 if QUICK_MODE else 1080,
    "height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "basename": "every_river_flowing_into_the_ocean",
    "title": "EVERY RIVER FLOWING INTO THE OCEAN",
    "subtitle": "A global river-network visualization",
    "timeout_s": 45,
    "cache_hours": 24 * 14,
    "river_scale": RIVER_SCALE,
    "land_scale": LAND_SCALE,
    "river_urls": [
        f"https://naciscdn.org/naturalearth/{RIVER_SCALE}/physical/ne_{RIVER_SCALE}_rivers_lake_centerlines.zip",
        f"https://naturalearth.s3.amazonaws.com/{RIVER_SCALE}_physical/ne_{RIVER_SCALE}_rivers_lake_centerlines.zip",
    ],
    "land_urls": [
        f"https://naciscdn.org/naturalearth/{LAND_SCALE}/physical/ne_{LAND_SCALE}_land.zip",
        f"https://naturalearth.s3.amazonaws.com/{LAND_SCALE}_physical/ne_{LAND_SCALE}_land.zip",
    ],
    "natural_earth_page": "https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-rivers-lake-centerlines/",
    "usgs_url": "https://www.usgs.gov/water-science-school/science/streamflow-and-water-cycle",
    "nasa_url": "https://science.nasa.gov/earth/earth-observatory/the-water-cycle/",
}

W = CONFIG["width"]
H = CONFIG["height"]
SIZE = (W, H)
SCALE = W / 1080.0

COLORS = {
    "bg": (2, 10, 20),
    "ocean": (4, 30, 54),
    "ocean2": (5, 44, 69),
    "land": (16, 37, 45),
    "land_edge": (42, 80, 88),
    "river": (70, 173, 255),
    "river_dim": (48, 111, 160),
    "river_hot": (94, 232, 255),
    "white": (246, 251, 255),
    "muted": (151, 194, 211),
    "cyan": (91, 227, 255),
    "teal": (89, 239, 199),
    "gold": (255, 203, 103),
    "orange": (255, 151, 88),
    "red": (255, 103, 112),
    "panel": (3, 14, 25),
}

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.0 if not QUICK_MODE else 1.8},
    {"name": "network", "start": 7.0 if not QUICK_MODE else 1.8, "end": 21.0 if not QUICK_MODE else 4.4},
    {"name": "outlets", "start": 21.0 if not QUICK_MODE else 4.4, "end": 33.0 if not QUICK_MODE else 6.8},
    {"name": "watershed", "start": 33.0 if not QUICK_MODE else 6.8, "end": 43.5 if not QUICK_MODE else 8.8},
    {"name": "exceptions", "start": 43.5 if not QUICK_MODE else 8.8, "end": 53.0 if not QUICK_MODE else 10.7},
    {"name": "outro", "start": 53.0 if not QUICK_MODE else 10.7, "end": CONFIG["duration_s"]},
]

CAPTION_TEXTS = [
    "From rain and snow on land, gravity gathers water into streams and rivers that form branching networks across the continents.",
    "This map lights up every river feature in the selected Natural Earth drainage layer. It is a cartographic dataset, not every tiny stream on Earth.",
    "Huge watersheds funnel water toward river mouths around the world—from the Amazon and Congo to the Mississippi, Ganges-Brahmaputra, Yangtze, and Lena.",
    "A watershed is land that drains toward the same outlet. Runoff moves downhill, joins tributaries, and much of it eventually returns to the ocean.",
    "But not every river reaches the sea. Some end in closed basins, seep underground, evaporate, or are heavily diverted before reaching their historic mouths.",
    "Rivers are one of the great return paths in Earth's water cycle: land to channel, channel to coast, coast back to ocean.",
]

CAPTIONS = [
    (
        shot["start"] + min(0.35, 0.08 * (shot["end"] - shot["start"])),
        shot["end"] - min(0.10, 0.04 * (shot["end"] - shot["start"])),
        text,
    )
    for shot, text in zip(SHOT_PLAN, CAPTION_TEXTS)
]

# Examples used only as labeled river-mouth markers. Coordinates are approximate
# mouth / delta positions in longitude, latitude.
MAJOR_OUTLETS = [
    ("AMAZON", -50.0, -1.2, "ATLANTIC"),
    ("CONGO", 12.3, -6.0, "ATLANTIC"),
    ("MISSISSIPPI", -89.2, 29.0, "GULF OF MEXICO"),
    ("GANGES–BRAHMAPUTRA", 90.2, 22.0, "BAY OF BENGAL"),
    ("YANGTZE", 121.9, 31.2, "EAST CHINA SEA"),
    ("MEKONG", 106.7, 9.6, "SOUTH CHINA SEA"),
    ("NILE", 31.2, 31.4, "MEDITERRANEAN"),
    ("DANUBE", 29.6, 45.1, "BLACK SEA"),
    ("LENA", 126.5, 72.0, "ARCTIC OCEAN"),
    ("OB", 72.8, 66.5, "ARCTIC OCEAN"),
    ("PARANÁ", -57.8, -35.0, "RÍO DE LA PLATA"),
]


# =============================================================================
# Data model
# =============================================================================

@dataclass
class RiverSnapshot:
    fetched_at_utc: str
    source_url: str
    source_kind: str
    data_status: str
    offline_fixture: bool
    river_scale: str
    land_scale: str
    total_line_features: int
    river_features: int
    named_river_features: int
    line_parts: int
    version_label: str
    scope_note: str


@dataclass
class RiverFeature:
    name: str
    feature_class: str
    scalerank: float
    parts: List[List[Tuple[float, float]]]


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
    max_lines: Optional[int] = None,
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
    if max_lines is not None and len(lines) > max_lines:
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


def write_srt(path: Path):
    lines = []
    for i, (start, end, text) in enumerate(CAPTIONS, start=1):
        lines.extend([str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


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


def project_lonlat(lon: float, lat: float, box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = box
    # Equirectangular with small polar crop for readability.
    lat = max(-82.0, min(82.0, float(lat)))
    x = x0 + (float(lon) + 180.0) / 360.0 * (x1 - x0)
    y = y0 + (82.0 - lat) / 164.0 * (y1 - y0)
    return x, y


def split_shape_parts(points: Sequence[Tuple[float, float]], indices: Sequence[int]) -> List[List[Tuple[float, float]]]:
    starts = list(indices) + [len(points)]
    out: List[List[Tuple[float, float]]] = []
    for i in range(len(starts) - 1):
        part = [(float(x), float(y)) for x, y in points[starts[i]: starts[i + 1]]]
        if len(part) >= 2:
            out.append(part)
    return out


def polyline_point(points: Sequence[Tuple[float, float]], frac: float) -> Tuple[float, float]:
    if not points:
        return 0.0, 0.0
    if len(points) == 1:
        return points[0]
    lengths = []
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        lengths.append(d)
        total += d
    if total <= 1e-9:
        return points[0]
    target = clamp(frac) * total
    acc = 0.0
    for i, d in enumerate(lengths):
        if acc + d >= target:
            f = (target - acc) / max(d, 1e-9)
            return lerp(points[i][0], points[i + 1][0], f), lerp(points[i][1], points[i + 1][1], f)
        acc += d
    return points[-1]


# =============================================================================
# Data collection
# =============================================================================

def _download_zip(urls: Sequence[str], tag: str) -> Path:
    if requests is None:
        raise RuntimeError("requests is unavailable")
    zip_path = CACHE_ROOT / f"{tag}.zip"
    fresh = False
    if zip_path.exists() and not REFRESH:
        age_hours = (utc_now().timestamp() - zip_path.stat().st_mtime) / 3600.0
        fresh = age_hours <= CONFIG["cache_hours"]
    if not fresh:
        errors = []
        for url in urls:
            try:
                response = requests.get(url, timeout=CONFIG["timeout_s"], headers={"User-Agent": "RiverOceanShort/1.0 educational renderer"})
                response.raise_for_status()
                zip_path.write_bytes(response.content)
                break
            except Exception as exc:
                errors.append(f"{url}: {exc}")
        else:
            raise RuntimeError("Natural Earth download failed: " + " | ".join(errors))
    extract_dir = CACHE_ROOT / tag
    if REFRESH and extract_dir.exists():
        shutil.rmtree(extract_dir)
    if not extract_dir.exists() or not any(extract_dir.glob("*.shp")):
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    return extract_dir


def _find_shp(directory: Path) -> Path:
    files = sorted(directory.glob("*.shp"))
    if not files:
        raise FileNotFoundError(f"No .shp file in {directory}")
    return files[0]


def load_live_natural_earth() -> Tuple[List[RiverFeature], List[List[Tuple[float, float]]], RiverSnapshot]:
    if shapefile is None:
        raise RuntimeError("pyshp is unavailable; install with: pip install pyshp")

    river_dir = _download_zip(CONFIG["river_urls"], f"natural_earth_rivers_{RIVER_SCALE}")
    land_dir = _download_zip(CONFIG["land_urls"], f"natural_earth_land_{LAND_SCALE}")

    river_reader = shapefile.Reader(str(_find_shp(river_dir)))
    land_reader = shapefile.Reader(str(_find_shp(land_dir)))

    river_features: List[RiverFeature] = []
    total_features = 0
    named = 0
    line_parts = 0

    for sr in river_reader.iterShapeRecords():
        total_features += 1
        rec = sr.record.as_dict() if hasattr(sr.record, "as_dict") else {}
        feature_class = str(rec.get("featurecla") or rec.get("FEATURECLA") or "River")
        # Exclude lake centerlines from the river count / render.
        if "lake" in feature_class.lower():
            continue
        name = str(rec.get("name") or rec.get("NAME") or "").strip()
        try:
            scalerank = float(rec.get("scalerank") if rec.get("scalerank") is not None else rec.get("SCALERANK", 8))
        except Exception:
            scalerank = 8.0
        parts = split_shape_parts(sr.shape.points, sr.shape.parts)
        if not parts:
            continue
        if name:
            named += 1
        line_parts += len(parts)
        river_features.append(RiverFeature(name, feature_class, scalerank, parts))

    land_parts: List[List[Tuple[float, float]]] = []
    for shape in land_reader.iterShapes():
        starts = list(shape.parts) + [len(shape.points)]
        for i in range(len(starts) - 1):
            part = [(float(x), float(y)) for x, y in shape.points[starts[i]: starts[i + 1]]]
            if len(part) >= 3:
                land_parts.append(part)

    snapshot = RiverSnapshot(
        fetched_at_utc=iso_z(utc_now()),
        source_url=CONFIG["natural_earth_page"],
        source_kind="Natural Earth global river cartography",
        data_status="cache-or-live",
        offline_fixture=False,
        river_scale=RIVER_SCALE,
        land_scale=LAND_SCALE,
        total_line_features=total_features,
        river_features=len(river_features),
        named_river_features=named,
        line_parts=line_parts,
        version_label="Natural Earth river theme v5.0.0",
        scope_note="All rendered lines are river features in the selected Natural Earth layer; lake centerlines are excluded.",
    )
    return river_features, land_parts, snapshot


def fallback_data() -> Tuple[List[RiverFeature], List[List[Tuple[float, float]]], RiverSnapshot]:
    # Stylized low-detail fallback used only when the live/cartographic files are
    # unavailable. It is intentionally labeled OFFLINE FIXTURE in the video.
    rivers_raw = [
        ("Amazon", [(-75, -4), (-68, -4), (-60, -3), (-54, -2), (-50, -1)]),
        ("Congo", [(25, -4), (21, -4), (17, -5), (12, -6)]),
        ("Mississippi", [(-95, 47), (-93, 40), (-91, 34), (-89, 29)]),
        ("Nile", [(31, 3), (31, 13), (30, 24), (31, 31)]),
        ("Danube", [(9, 48), (18, 47), (24, 46), (29, 45)]),
        ("Ganges-Brahmaputra", [(78, 30), (84, 27), (90, 22)]),
        ("Yangtze", [(92, 34), (104, 31), (114, 30), (122, 31)]),
        ("Mekong", [(101, 31), (103, 22), (105, 15), (107, 10)]),
        ("Lena", [(108, 54), (116, 61), (123, 68), (126, 72)]),
        ("Parana", [(-55, -20), (-58, -27), (-58, -35)]),
        ("Murray", [(146, -36), (141, -35), (138, -35)]),
        ("Niger", [(11, 12), (5, 12), (3, 6), (5, 4)]),
    ]
    rivers = [RiverFeature(name, "River", 1.0, [pts]) for name, pts in rivers_raw]

    # Very rough continent silhouettes for layout testing only.
    land_parts = [
        [(-168, 15), (-140, 70), (-80, 72), (-52, 48), (-82, 8), (-110, 5), (-168, 15)],
        [(-82, 12), (-35, 10), (-52, -55), (-75, -50), (-82, 12)],
        [(-18, 36), (5, 72), (45, 70), (90, 76), (160, 60), (150, 8), (105, -10), (45, -35), (10, -35), (-18, 36)],
        [(110, -10), (155, -10), (154, -44), (115, -39), (110, -10)],
        [(-52, 60), (-20, 82), (-45, 83), (-63, 72), (-52, 60)],
    ]
    snapshot = RiverSnapshot(
        fetched_at_utc=iso_z(utc_now()),
        source_url=CONFIG["natural_earth_page"],
        source_kind="offline river layout fixture",
        data_status="offline-fixture",
        offline_fixture=True,
        river_scale="fixture",
        land_scale="fixture",
        total_line_features=len(rivers),
        river_features=len(rivers),
        named_river_features=len(rivers),
        line_parts=len(rivers),
        version_label="offline fixture",
        scope_note="Approximate major-river paths for layout testing only; not a complete map.",
    )
    return rivers, land_parts, snapshot


def collect_data() -> Tuple[List[RiverFeature], List[List[Tuple[float, float]]], RiverSnapshot, Dict]:
    errors: Dict[str, str] = {}
    if OFFLINE_MODE:
        rivers, land, snapshot = fallback_data()
    else:
        try:
            rivers, land, snapshot = load_live_natural_earth()
        except Exception as exc:
            errors["natural_earth_fetch"] = str(exc)
            rivers, land, snapshot = fallback_data()

    summary = {
        "generated_at_utc": iso_z(utc_now()),
        "data_status": snapshot.data_status,
        "offline_fixture": snapshot.offline_fixture,
        "river_scale": snapshot.river_scale,
        "land_scale": snapshot.land_scale,
        "river_features": snapshot.river_features,
        "named_river_features": snapshot.named_river_features,
        "line_parts": snapshot.line_parts,
        "major_outlet_examples": len(MAJOR_OUTLETS),
        "errors": errors,
        "interpretation": "The title is a hook; the rendered network is the selected Natural Earth river layer, not every stream on Earth.",
    }
    return rivers, land, snapshot, summary


def save_data(rivers: Sequence[RiverFeature], snapshot: RiverSnapshot, summary: Dict) -> Tuple[Path, Path]:
    rows = []
    for feat in rivers:
        rows.append({
            "name": feat.name,
            "feature_class": feat.feature_class,
            "scalerank": feat.scalerank,
            "parts": len(feat.parts),
            "points": sum(len(p) for p in feat.parts),
        })
    csv_path = DATA_ROOT / "natural_earth_river_feature_summary.csv"
    json_path = DATA_ROOT / "river_short_snapshot.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"summary": summary, "snapshot": asdict(snapshot)}, indent=2), encoding="utf-8")
    return csv_path, json_path


# =============================================================================
# Scene renderer
# =============================================================================

class RiverScene:
    def __init__(self, rivers: Sequence[RiverFeature], land_parts: Sequence[Sequence[Tuple[float, float]]], snapshot: RiverSnapshot, summary: Dict):
        self.rivers = list(rivers)
        self.land_parts = [list(p) for p in land_parts]
        self.snapshot = snapshot
        self.summary = summary
        self.map_box = (int(W * 0.035), int(H * 0.19), int(W * 0.965), int(H * 0.71))
        self.small_map_box = (int(W * 0.08), int(H * 0.27), int(W * 0.92), int(H * 0.65))
        self._rng = np.random.default_rng(72831)
        self.stars = [(float(self._rng.uniform(0, W)), float(self._rng.uniform(0, H)), float(self._rng.uniform(.3, 1.5) * SCALE), int(self._rng.integers(15, 85))) for _ in range(180 if QUICK_MODE else 360)]
        self.base_map = self._build_base_map()
        self.river_layer = self._build_river_layer(alpha=190)
        self.river_dim_layer = self._build_river_layer(alpha=72, dim=True)
        self.projected_river_parts = self._project_river_parts()
        # For animation efficiency, choose a stable set of long-ish parts.
        ranked = sorted(self.projected_river_parts, key=lambda p: -sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a,b in zip(p[:-1], p[1:])))
        self.pulse_parts = ranked[: (34 if QUICK_MODE else 90)]

    def _project_river_parts(self) -> List[List[Tuple[float, float]]]:
        parts: List[List[Tuple[float, float]]] = []
        for feat in self.rivers:
            for part in feat.parts:
                mapped = [project_lonlat(lon, lat, self.map_box) for lon, lat in part]
                if len(mapped) >= 2:
                    parts.append(mapped)
        return parts

    def _build_base_map(self) -> Image.Image:
        layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        x0, y0, x1, y1 = self.map_box
        d.rounded_rectangle((x0, y0, x1, y1), radius=int(28*SCALE), fill=COLORS["ocean"] + (238,), outline=COLORS["cyan"] + (45,), width=max(1, int(2*SCALE)))
        # subtle latitude bands
        for lat in (-60, -30, 0, 30, 60):
            xa, yy = project_lonlat(-180, lat, self.map_box)
            xb, _ = project_lonlat(180, lat, self.map_box)
            d.line((xa, yy, xb, yy), fill=(82, 139, 158, 18), width=1)
        for lon in (-120, -60, 0, 60, 120):
            xx, ya = project_lonlat(lon, 82, self.map_box)
            _, yb = project_lonlat(lon, -82, self.map_box)
            d.line((xx, ya, xx, yb), fill=(82, 139, 158, 14), width=1)
        # land
        for part in self.land_parts:
            mapped = [project_lonlat(lon, lat, self.map_box) for lon, lat in part]
            if len(mapped) >= 3:
                try:
                    d.polygon(mapped, fill=COLORS["land"] + (255,), outline=COLORS["land_edge"] + (150,))
                except Exception:
                    pass
        return layer

    def _build_river_layer(self, alpha: int, dim: bool = False) -> Image.Image:
        layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for feat in self.rivers:
            # Natural Earth scalerank is relative importance. Lower rank = larger.
            rank = feat.scalerank if np.isfinite(feat.scalerank) else 8.0
            width = max(1, int((3.8 - min(3.0, rank * 0.28)) * SCALE))
            if QUICK_MODE:
                width = max(1, int(width * 0.8))
            color = COLORS["river_dim"] if dim else COLORS["river"]
            local_alpha = int(alpha * (1.0 if rank <= 3 else 0.72 if rank <= 6 else 0.48))
            for part in feat.parts:
                mapped = [project_lonlat(lon, lat, self.map_box) for lon, lat in part]
                if len(mapped) >= 2:
                    d.line(mapped, fill=color + (local_alpha,), width=width, joint="curve")
        return layer

    def background(self, t: float) -> Image.Image:
        img = Image.new("RGBA", SIZE, COLORS["bg"] + (255,))
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for cx, cy, col in [(W*.18,H*.23,(14,81,110)), (W*.77,H*.30,(8,66,104)), (W*.52,H*.78,(9,55,86))]:
            for rr, aa in [(W*.42, 16), (W*.26, 26), (W*.14, 34)]:
                gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=col + (aa,))
        glow = glow.filter(ImageFilter.GaussianBlur(60 if not QUICK_MODE else 28))
        img.alpha_composite(glow)
        d = ImageDraw.Draw(img)
        for x, y, r, a in self.stars:
            twinkle = 0.55 + 0.45 * math.sin(t * 1.6 + x * .01 + y * .007)
            d.ellipse((x-r, y-r, x+r, y+r), fill=(183, 226, 242, int(a*twinkle)))
        return img

    def draw_header(self, img: Image.Image, t: float):
        shot = get_shot(t)["name"]
        shot_titles = {
            "intro": "LAND → RIVER → OCEAN",
            "network": "GLOBAL RIVER NETWORK",
            "outlets": "WHERE RIVERS MEET THE SEA",
            "watershed": "HOW A WATERSHED WORKS",
            "exceptions": "THE IMPORTANT EXCEPTIONS",
            "outro": "THE WATER CYCLE CLOSES THE LOOP",
        }
        if t < (6.2 if not QUICK_MODE else 1.55):
            alpha = int(255 * smoothstep((t - .12) / .7))
            draw_wrapped_text(img, CONFIG["title"], (int(W*.055), int(H*.048)), int(W*.88), size=43 if not QUICK_MODE else 19, fill=COLORS["white"] + (alpha,), bold=True, line_spacing=2, max_lines=2)
            draw_text(img, CONFIG["subtitle"], (int(W*.058), int(H*.125)), size=21 if not QUICK_MODE else 9, fill=COLORS["cyan"] + (min(alpha,235),), bold=True, stroke=1)
        else:
            draw_text(img, shot_titles[shot], (int(W*.055), int(H*.052)), size=20 if not QUICK_MODE else 9, fill=COLORS["cyan"] + (225,), bold=True, stroke=1)

    def draw_source_hud(self, img: Image.Image, t: float):
        if t < (5.7 if not QUICK_MODE else 1.45):
            return
        status = "OFFLINE FIXTURE" if self.snapshot.offline_fixture else self.snapshot.river_scale.upper()
        draw_text(img, f"RIVER MAP // {status}", (int(W*.955), int(H*.052)), size=15 if not QUICK_MODE else 7, fill=COLORS["muted"] + (210,), bold=True, anchor="ra", stroke=1)
        if not self.snapshot.offline_fixture:
            draw_text(img, f"{self.snapshot.river_features:,} RIVER FEATURES", (int(W*.955), int(H*.074)), size=14 if not QUICK_MODE else 6, fill=COLORS["muted"] + (180,), anchor="ra", stroke=1)

    def draw_caption(self, img: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        y0 = H - (246 if not QUICK_MODE else 124)
        overlay = Image.new("RGBA", SIZE, (0,0,0,0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((int(W*.04), y0, int(W*.96), y0 + int(126*SCALE)), radius=int(24*SCALE), fill=(2,8,16,184), outline=COLORS["cyan"] + (54,), width=1)
        img.alpha_composite(overlay)
        draw_wrapped_text(img, text, (int(W*.063), y0 + int(25*SCALE)), int(W*.874), size=27 if not QUICK_MODE else 13, fill=COLORS["white"] + (244,), line_spacing=4, max_lines=3)

    def draw_intro(self, img: Image.Image, t: float):
        img.alpha_composite(self.base_map)
        d = ImageDraw.Draw(img)
        p = ease_in_out_sine((t - SHOT_PLAN[0]["start"]) / max(1e-6, SHOT_PLAN[0]["end"] - SHOT_PLAN[0]["start"]))
        # reveal dim network then brighten
        alpha = int(70 + 150 * p)
        rivers = self.river_layer.copy()
        rivers.putalpha(rivers.getchannel("A").point(lambda a: int(a * alpha / 220)))
        img.alpha_composite(rivers)
        # moving coast-to-ocean rings at selected mouths
        for idx, (_, lon, lat, _) in enumerate(MAJOR_OUTLETS[:6]):
            x,y = project_lonlat(lon, lat, self.map_box)
            phase = (p*1.4 + idx*.13) % 1.0
            r = (5 + 25*phase) * SCALE
            a = int(180*(1-phase))
            d.ellipse((x-r,y-r,x+r,y+r), outline=COLORS["cyan"] + (a,), width=max(1,int(2*SCALE)))
        draw_text(img, "RAIN + SNOW", (int(W*.18), int(H*.76)), size=19 if not QUICK_MODE else 9, fill=COLORS["muted"] + (220,), bold=True, anchor="ma", stroke=1)
        draw_text(img, "→", (int(W*.36), int(H*.76)), size=28 if not QUICK_MODE else 13, fill=COLORS["cyan"] + (230,), bold=True, anchor="ma", stroke=1)
        draw_text(img, "RIVERS", (int(W*.52), int(H*.76)), size=21 if not QUICK_MODE else 10, fill=COLORS["cyan"] + (235,), bold=True, anchor="ma", stroke=1)
        draw_text(img, "→", (int(W*.68), int(H*.76)), size=28 if not QUICK_MODE else 13, fill=COLORS["cyan"] + (230,), bold=True, anchor="ma", stroke=1)
        draw_text(img, "OCEAN", (int(W*.84), int(H*.76)), size=21 if not QUICK_MODE else 10, fill=COLORS["teal"] + (235,), bold=True, anchor="ma", stroke=1)

    def draw_network(self, img: Image.Image, t: float):
        img.alpha_composite(self.base_map)
        shot = get_shot(t)
        p = smoothstep((t - shot["start"]) / max(1e-6, shot["end"] - shot["start"]))
        # reveal network by vertical wipe with a glow edge
        mask = Image.new("L", SIZE, 0)
        md = ImageDraw.Draw(mask)
        x0,y0,x1,y1 = self.map_box
        reveal_x = int(lerp(x0, x1, p))
        md.rectangle((x0,y0,reveal_x,y1), fill=255)
        river = self.river_layer.copy()
        river.putalpha(Image.composite(river.getchannel("A"), Image.new("L", SIZE, 0), mask))
        img.alpha_composite(river)
        d = ImageDraw.Draw(img)
        d.line((reveal_x,y0,reveal_x,y1), fill=COLORS["cyan"] + (130,), width=max(1,int(2*SCALE)))
        # pulses travel along selected lines. Direction is deliberately not stated.
        for i, part in enumerate(self.pulse_parts):
            phase = (t*.22 + i*.071) % 1.0
            x,y = polyline_point(part, phase)
            rr = max(1.5, 3.7*SCALE)
            d.ellipse((x-rr,y-rr,x+rr,y+rr), fill=COLORS["river_hot"] + (170,))
        draw_text(img, "EVERY RIVER FEATURE IN THIS MAP LAYER", (W//2, int(H*.755)), size=22 if not QUICK_MODE else 10, fill=COLORS["white"] + (235,), bold=True, anchor="ma", stroke=1)
        scope = "Natural Earth 10m" if not self.snapshot.offline_fixture else "Offline layout fixture"
        draw_text(img, scope, (W//2, int(H*.786)), size=16 if not QUICK_MODE else 7, fill=COLORS["muted"] + (215,), anchor="ma", stroke=1)

    def draw_outlets(self, img: Image.Image, t: float):
        img.alpha_composite(self.base_map)
        img.alpha_composite(self.river_dim_layer)
        d = ImageDraw.Draw(img)
        shot = get_shot(t)
        local = (t - shot["start"]) / max(1e-6, shot["end"] - shot["start"])
        count = max(1, int(math.ceil(clamp(local) * len(MAJOR_OUTLETS))))
        for i, (name, lon, lat, ocean) in enumerate(MAJOR_OUTLETS[:count]):
            x,y = project_lonlat(lon, lat, self.map_box)
            pulse = .5 + .5*math.sin(t*4.2+i)
            r = (5 + 4*pulse)*SCALE
            d.ellipse((x-r,y-r,x+r,y+r), fill=COLORS["gold"] + (245,), outline=COLORS["white"] + (220,), width=1)
            if i < 6 or (not QUICK_MODE and i in {6,8,10}):
                anchor = "la" if x < W*.72 else "ra"
                tx = int(x + 10*SCALE) if anchor == "la" else int(x - 10*SCALE)
                draw_text(img, name, (tx, int(y - 4*SCALE)), size=13 if not QUICK_MODE else 6, fill=COLORS["white"] + (235,), bold=True, anchor=anchor, stroke=1)
        # lower info card
        y = int(H*.745)
        overlay = Image.new("RGBA", SIZE, (0,0,0,0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((int(W*.08), y, int(W*.92), y+int(92*SCALE)), radius=int(20*SCALE), fill=COLORS["panel"] + (190,), outline=COLORS["gold"] + (60,), width=1)
        img.alpha_composite(overlay)
        draw_text(img, "A RIVER MOUTH IS A WATERSHED'S EXIT", (W//2, y+int(22*SCALE)), size=20 if not QUICK_MODE else 9, fill=COLORS["gold"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(img, "Tributaries merge → main stem → delta / estuary / coast", (W//2, y+int(56*SCALE)), size=16 if not QUICK_MODE else 7, fill=COLORS["muted"] + (225,), anchor="ma", stroke=1)

    def draw_watershed(self, img: Image.Image, t: float):
        # stylized cross-section: mountains -> tributaries -> river -> ocean
        d = ImageDraw.Draw(img)
        panel = (int(W*.07), int(H*.20), int(W*.93), int(H*.72))
        x0,y0,x1,y1 = panel
        d.rounded_rectangle(panel, radius=int(28*SCALE), fill=COLORS["panel"] + (230,), outline=COLORS["cyan"] + (60,), width=max(1,int(2*SCALE)))
        ocean_y = int(y1 - 72*SCALE)
        # mountains
        mountain = [(x0+20*SCALE,ocean_y),(x0+130*SCALE,y0+90*SCALE),(x0+230*SCALE,ocean_y),(x0+300*SCALE,y0+130*SCALE),(x0+390*SCALE,ocean_y)]
        d.polygon(mountain, fill=(31,58,58,255), outline=(65,101,95,220))
        # ocean
        d.rectangle((x0,ocean_y,x1,y1), fill=COLORS["ocean2"] + (255,))
        for yy in range(ocean_y+8, y1, max(8,int(18*SCALE))):
            d.line((x0,yy,x1,yy), fill=COLORS["cyan"] + (18,), width=1)
        # main river path
        river = [(x0+135*SCALE,y0+118*SCALE),(x0+185*SCALE,y0+205*SCALE),(x0+315*SCALE,y0+280*SCALE),(x0+500*SCALE,y0+325*SCALE),(x1-40*SCALE,ocean_y+2*SCALE)]
        river = [(float(a),float(b)) for a,b in river]
        d.line(river, fill=COLORS["river"] + (230,), width=max(2,int(8*SCALE)), joint="curve")
        tributaries = [
            [(x0+260*SCALE,y0+145*SCALE),(x0+280*SCALE,y0+210*SCALE),(x0+315*SCALE,y0+280*SCALE)],
            [(x0+390*SCALE,y0+160*SCALE),(x0+410*SCALE,y0+255*SCALE),(x0+500*SCALE,y0+325*SCALE)],
            [(x0+520*SCALE,y0+170*SCALE),(x0+540*SCALE,y0+265*SCALE),(x0+610*SCALE,y0+350*SCALE)],
        ]
        for tr in tributaries:
            d.line(tr, fill=COLORS["river"] + (190,), width=max(1,int(4*SCALE)), joint="curve")
        # rain animation
        shot = get_shot(t)
        p = clamp((t-shot["start"]) / max(1e-6, shot["end"]-shot["start"]))
        for i in range(26 if not QUICK_MODE else 12):
            rx = x0 + (32 + (i*67)%520)*SCALE
            base = y0 + (25 + (i*31)%110)*SCALE
            ry = base + ((t*90*SCALE + i*18*SCALE) % (165*SCALE))
            d.line((rx,ry,rx-5*SCALE,ry+12*SCALE), fill=COLORS["cyan"] + (120,), width=max(1,int(2*SCALE)))
        # moving water dots on main stem
        for k in range(7):
            frac = (p*1.7 + k/7) % 1.0
            x,y = polyline_point(river, frac)
            rr = 5*SCALE
            d.ellipse((x-rr,y-rr,x+rr,y+rr), fill=COLORS["river_hot"] + (245,))
        draw_text(img, "WATERSHED", (x0+int(22*SCALE), y0+int(22*SCALE)), size=19 if not QUICK_MODE else 9, fill=COLORS["cyan"] + (235,), bold=True, stroke=1)
        draw_text(img, "gravity", (x0+int(150*SCALE), y0+int(170*SCALE)), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (210,), anchor="ma", stroke=1)
        draw_text(img, "tributaries", (x0+int(420*SCALE), y0+int(230*SCALE)), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (210,), anchor="ma", stroke=1)
        draw_text(img, "river mouth", (x1-int(75*SCALE), ocean_y-int(20*SCALE)), size=14 if not QUICK_MODE else 7, fill=COLORS["gold"] + (225,), anchor="ma", stroke=1)
        draw_text(img, "OCEAN", (x1-int(85*SCALE), ocean_y+int(42*SCALE)), size=17 if not QUICK_MODE else 8, fill=COLORS["cyan"] + (230,), bold=True, anchor="ma", stroke=1)
        draw_text(img, "LAND THAT DRAINS TO THE SAME OUTLET", (W//2, int(H*.765)), size=21 if not QUICK_MODE else 10, fill=COLORS["white"] + (235,), bold=True, anchor="ma", stroke=1)

    def draw_exceptions(self, img: Image.Image, t: float):
        cards = [
            ("CLOSED BASINS", "Some rivers terminate inland in lakes, wetlands, salt flats, or depressions.", COLORS["gold"]),
            ("INFILTRATION + EVAPORATION", "Water can seep into the ground or return to the atmosphere before reaching a coast.", COLORS["cyan"]),
            ("HUMAN DIVERSION", "Dams and withdrawals can greatly reduce or even interrupt flow to a historic river mouth.", COLORS["orange"]),
        ]
        y0 = int(H*.21)
        card_h = int(H*.16)
        gap = int(H*.035)
        for i,(title,body,col) in enumerate(cards):
            y = y0 + i*(card_h+gap)
            overlay = Image.new("RGBA", SIZE, (0,0,0,0))
            od = ImageDraw.Draw(overlay)
            od.rounded_rectangle((int(W*.075),y,int(W*.925),y+card_h), radius=int(24*SCALE), fill=COLORS["panel"] + (220,), outline=col + (75,), width=max(1,int(2*SCALE)))
            img.alpha_composite(overlay)
            draw_text(img, title, (int(W*.105), y+int(23*SCALE)), size=22 if not QUICK_MODE else 10, fill=col + (240,), bold=True, stroke=1)
            draw_wrapped_text(img, body, (int(W*.105), y+int(62*SCALE)), int(W*.76), size=18 if not QUICK_MODE else 8, fill=COLORS["white"] + (232,), line_spacing=3, max_lines=2)
        draw_text(img, "SO THE TITLE IS A VISUAL HOOK — NOT A LITERAL CENSUS", (W//2, int(H*.78)), size=18 if not QUICK_MODE else 8, fill=COLORS["muted"] + (225,), bold=True, anchor="ma", stroke=1)

    def draw_outro(self, img: Image.Image, t: float):
        img.alpha_composite(self.base_map)
        img.alpha_composite(self.river_layer)
        d = ImageDraw.Draw(img)
        for i,(_,lon,lat,_) in enumerate(MAJOR_OUTLETS):
            x,y = project_lonlat(lon,lat,self.map_box)
            phase = (t*.5+i*.08)%1
            rr = (4+18*phase)*SCALE
            d.ellipse((x-rr,y-rr,x+rr,y+rr), outline=COLORS["teal"] + (int(170*(1-phase)),), width=max(1,int(2*SCALE)))
        draw_text(img, "LAND → RIVERS → OCEAN", (W//2, int(H*.755)), size=28 if not QUICK_MODE else 13, fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(img, "ONE LOOP IN EARTH'S WATER CYCLE", (W//2, int(H*.797)), size=18 if not QUICK_MODE else 8, fill=COLORS["cyan"] + (230,), bold=True, anchor="ma", stroke=1)
        draw_text(img, "Source map: Natural Earth • Water-cycle context: USGS / NASA", (W//2, int(H*.84)), size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (210,), anchor="ma", stroke=1)

    def draw_scanlines(self, img: Image.Image, t: float):
        overlay = Image.new("RGBA", SIZE, (0,0,0,0))
        od = ImageDraw.Draw(overlay)
        offset = int((t*31)%8)
        for y in range(offset,H,8):
            od.line((0,y,W,y), fill=(95,190,220,8), width=1)
        scan_y = int((t*140)%(H+180))-90
        od.rectangle((0,scan_y,W,scan_y+int(42*SCALE)), fill=(76,195,230,7))
        img.alpha_composite(overlay)

    def render(self, t: float) -> Image.Image:
        img = self.background(t)
        shot = get_shot(t)["name"]
        if shot == "intro":
            self.draw_intro(img,t)
        elif shot == "network":
            self.draw_network(img,t)
        elif shot == "outlets":
            self.draw_outlets(img,t)
        elif shot == "watershed":
            self.draw_watershed(img,t)
        elif shot == "exceptions":
            self.draw_exceptions(img,t)
        else:
            self.draw_outro(img,t)
        self.draw_header(img,t)
        self.draw_source_hud(img,t)
        self.draw_caption(img,t)
        self.draw_scanlines(img,t)
        return img.convert("RGB")


# =============================================================================
# Export
# =============================================================================

def make_contact_sheet(paths: Sequence[Path], out_path: Path):
    imgs = [Image.open(p).convert("RGB") for p in paths if p.exists()]
    if not imgs:
        return
    thumb_w = 270 if not QUICK_MODE else 180
    thumbs = []
    for img in imgs:
        ratio = thumb_w / img.width
        thumbs.append(img.resize((thumb_w, int(img.height*ratio)), Image.Resampling.LANCZOS))
    pad = 12
    cols = 3
    rows = int(math.ceil(len(thumbs)/cols))
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols*thumb_w+(cols+1)*pad, rows*cell_h+(rows+1)*pad), (5,12,19))
    for i,thumb in enumerate(thumbs):
        x = pad + (i%cols)*(thumb_w+pad)
        y = pad + (i//cols)*(cell_h+pad)
        sheet.paste(thumb,(x,y))
    sheet.save(out_path, quality=92)


def render_preview_frames(scene: RiverScene) -> Tuple[List[Path], Path]:
    times = []
    for shot in SHOT_PLAN:
        times.append((shot["start"] + shot["end"])/2)
    paths: List[Path] = []
    for i,t in enumerate(times, start=1):
        p = PREVIEW_ROOT / f"scene_{i:02d}_{get_shot(t)['name']}.jpg"
        scene.render(t).save(p, quality=92)
        paths.append(p)
    contact = PREVIEW_ROOT / "every_river_flowing_into_the_ocean_contact_sheet.jpg"
    make_contact_sheet(paths, contact)
    return paths, contact


def render_video(scene: RiverScene, out_path: Path):
    fps = CONFIG["fps"]
    frames = int(round(CONFIG["duration_s"]*fps))
    writer_kwargs = {
        "fps": fps,
        "codec": "libx264",
        "quality": 7 if QUICK_MODE else 8,
        "pixelformat": "yuv420p",
        "macro_block_size": 1,
    }
    with iio.get_writer(out_path, format="FFMPEG", **writer_kwargs) as writer:
        for idx in tqdm(range(frames), desc="Rendering river Short"):
            t = idx / fps
            writer.append_data(np.asarray(scene.render(t)))


def main():
    rivers, land, snapshot, summary = collect_data()
    csv_path, json_path = save_data(rivers, snapshot, summary)
    srt_path = OUTPUT_ROOT / f"{CONFIG['basename']}.srt"
    write_srt(srt_path)

    scene = RiverScene(rivers, land, snapshot, summary)
    preview_paths, contact_path = render_preview_frames(scene)
    video_path = OUTPUT_ROOT / f"{CONFIG['basename']}.mp4"
    render_video(scene, video_path)

    print("\nDONE")
    print(f"Data status : {snapshot.data_status}")
    print(f"River scale : {snapshot.river_scale}")
    print(f"River feats : {snapshot.river_features:,}")
    print(f"CSV         : {csv_path}")
    print(f"JSON        : {json_path}")
    print(f"SRT         : {srt_path}")
    print(f"Contact     : {contact_path}")
    print(f"Video       : {video_path}")


if __name__ == "__main__":
    main()
