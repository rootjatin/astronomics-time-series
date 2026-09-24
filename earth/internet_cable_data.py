from __future__ import annotations

"""
Earth's Submarine Cables in 3D
==============================
output : https://www.youtube.com/shorts/XrT_I2ESlxk
A cinematic vertical YouTube Short renderer that wraps submarine cable routes
around a rotating 3D-style Earth.

Data strategy
-------------
- When online, the script downloads and caches the public TeleGeography-derived
  cable GeoJSON and landing-point GeoJSON snapshot maintained at:
  https://github.com/JesseCallahanBryant/undersea-cables
- That public dataset is published under CC BY-NC-SA 3.0 and was refreshed in
  March 2026 in the repository metadata.
- When offline, the script falls back to a SMALL, representative route skeleton
  assembled from approximate landing-city coordinates. The fallback is for layout
  testing only and is explicitly labeled in the render.

Scientific / infrastructure framing
-----------------------------------
- ITU reports that submarine cables carry more than 99% of international data
  traffic.
- TeleGeography's 2026 printed cable map depicts 694 cable systems and 1,893
  landing stations, including in-service and planned systems.
- Cable routes shown from the public GeoJSON are geographic route geometries;
  the rendered ocean depth, glow, packet motion, and cable thickness are visual
  devices, not bathymetric or physical-scale reconstructions.
- Submarine cables are fiber-optic systems. In deep water the cable itself is
  typically only about garden-hose width; the glowing lines in this animation are
  intentionally enlarged so they remain visible on a phone screen.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg requests tqdm

Quick preview (offline-safe)
----------------------------
    SUBMARINE_CABLES_QUICK=1 SUBMARINE_CABLES_OFFLINE=1 \
        python earths_submarine_cables_in_3d.py

Full 1080x1920 render
---------------------
    python earths_submarine_cables_in_3d.py

4K vertical
-----------
    SUBMARINE_CABLES_4K=1 python earths_submarine_cables_in_3d.py

Force a fresh public-data download
----------------------------------
    SUBMARINE_CABLES_REFRESH=1 python earths_submarine_cables_in_3d.py
"""

import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

QUICK_MODE = os.environ.get("SUBMARINE_CABLES_QUICK", "0") == "1"
FOUR_K = os.environ.get("SUBMARINE_CABLES_4K", "0") == "1" and not QUICK_MODE
OFFLINE_MODE = os.environ.get("SUBMARINE_CABLES_OFFLINE", "0") == "1"
REFRESH = os.environ.get("SUBMARINE_CABLES_REFRESH", "0") == "1"

W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 6 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SIZE = (W, H)
SCALE = W / 1080.0

OUTPUT_ROOT = Path("earths_submarine_cables_in_3d_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
CACHE_ROOT = OUTPUT_ROOT / "cache"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT, CACHE_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "EARTH'S SUBMARINE CABLES IN 3D",
    "subtitle": "the hidden fiber-optic network beneath the oceans",
    "basename": "earths_submarine_cables_in_3d",
    "contrast": 1.12,
    "saturation": 1.09,
    "vignette": 0.26,
    "timeout_s": 40,
    "cache_hours": 72,
    "cable_geojson_url": "https://raw.githubusercontent.com/JesseCallahanBryant/undersea-cables/main/data/cable-geo.json",
    "landing_geojson_url": "https://raw.githubusercontent.com/JesseCallahanBryant/undersea-cables/main/data/landing-point-geo.json",
    "dataset_repo": "https://github.com/JesseCallahanBryant/undersea-cables",
    "telegeography_2026": "https://resources.telegeography.com/2026-submarine-cable-map",
    "itu_report": "https://www.itu.int/itu-d/reports/statistics/global-connectivity-report-2025/",
    "itu_resilience": "https://www.itu.int/digital-resilience/submarine-cables/",
}

COLORS = {
    "space": (1, 5, 14),
    "space2": (5, 17, 34),
    "white": (245, 250, 255),
    "muted": (155, 190, 211),
    "cyan": (67, 233, 255),
    "blue": (58, 141, 255),
    "green": (91, 238, 176),
    "gold": (255, 205, 88),
    "orange": (255, 137, 72),
    "red": (255, 80, 103),
    "violet": (185, 118, 255),
    "magenta": (244, 89, 194),
    "ocean": (6, 39, 75),
    "ocean_lit": (14, 91, 133),
    "land": (50, 91, 82),
    "land_edge": (110, 174, 143),
    "panel": (2, 10, 23),
}

SHOT_PLAN = [
    {"name": "reveal", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "backbone", "start": 8.0 if not QUICK_MODE else 1.8, "end": 19.0 if not QUICK_MODE else 4.25},
    {"name": "atlantic", "start": 19.0 if not QUICK_MODE else 4.25, "end": 30.0 if not QUICK_MODE else 6.7},
    {"name": "landings", "start": 30.0 if not QUICK_MODE else 6.7, "end": 40.5 if not QUICK_MODE else 9.05},
    {"name": "fragility", "start": 40.5 if not QUICK_MODE else 9.05, "end": 51.0 if not QUICK_MODE else 11.4},
    {"name": "outro", "start": 51.0 if not QUICK_MODE else 11.4, "end": DURATION},
]

CAPTION_TEXTS = [
    "Under the oceans is a physical network of glass fiber linking continents. From space it would look like a glowing web wrapped around Earth.",
    "More than ninety-nine percent of international data traffic travels through submarine cables. Satellites are visible; most of the global internet backbone is not.",
    "Some of the densest corridors cross the Atlantic, Pacific, Mediterranean, Red Sea and Indian Ocean. Light pulses can cross an ocean inside fibers only millimeters across.",
    "Every route eventually reaches land. Landing stations connect the wet plant offshore to terrestrial fiber, data centers and national networks.",
    "The network is enormous, but individual cables are vulnerable. Anchors, fishing, earthquakes and other events can break them, so redundancy and repair ships matter.",
    "TeleGeography's 2026 map depicts hundreds of active and planned systems and nearly two thousand landing stations. The internet has a geography — and most of it runs under water.",
]

CAPTIONS = [
    (
        shot["start"] + min(0.35, 0.07 * (shot["end"] - shot["start"])),
        shot["end"] - min(0.10, 0.035 * (shot["end"] - shot["start"])),
        text,
    )
    for shot, text in zip(SHOT_PLAN, CAPTION_TEXTS)
]

TELEGEOGRAPHY_2026_SYSTEMS = 694
TELEGEOGRAPHY_2026_LANDINGS = 1893
ITU_INTL_DATA_SHARE = 99.0
ITU_REPAIRS_2023 = 200


# =============================================================================
# Data model
# =============================================================================

@dataclass
class CableSnapshot:
    generated_at_utc: str
    data_status: str
    offline_fallback: bool
    route_feature_count: int
    landing_point_count_loaded: int
    public_dataset_refresh_note: str
    telegeography_2026_systems_active_and_planned: int
    telegeography_2026_landing_stations: int
    international_data_share_percent_gt: float
    reported_repairs_2023_gt: int
    interpretation: str


# =============================================================================
# Helpers
# =============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path) -> Path:
    lines: List[str] = []
    for idx, (start, end, text) in enumerate(CAPTIONS, 1):
        lines += [str(idx), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, max(7, int(size * SCALE)))
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[float, float],
    size: int,
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
        stroke_width=max(1, int(stroke * SCALE)),
        stroke_fill=(0, 0, 0, 220),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    spacing: int = 6,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold)
    words = str(text).split()
    lines: List[str] = []
    cur = ""
    for word in words:
        candidate = word if not cur else cur + " " + word
        box = draw.textbbox((0, 0), candidate, font=font, stroke_width=max(1, int(2 * SCALE)))
        if box[2] - box[0] <= max_width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    x, y = xy
    for line in lines:
        draw.text(
            (x, y), line, font=font, fill=fill,
            stroke_width=max(1, int(2 * SCALE)), stroke_fill=(0, 0, 0, 220)
        )
        box = draw.textbbox((x, y), line, font=font, stroke_width=max(1, int(2 * SCALE)))
        y += (box[3] - box[1]) + int(spacing * SCALE)


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    rr = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * rr**1.8, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(W, H, float(CONFIG["vignette"]))


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def caption_at(t: float) -> Optional[str]:
    """Return a short, readable page of the active caption.

    Long narration is split across its original time window so only about
    15-20 words are visible at once; the full text remains in the SRT file.
    """
    for start, end, text in CAPTIONS:
        if start <= t < end:
            words = str(text).split()
            if not words:
                return None
            target_words = 18
            page_count = max(1, (len(words) + target_words - 1) // target_words)
            base = len(words) // page_count
            extra = len(words) % page_count
            progress = (t - start) / max(end - start, 1e-9)
            page = min(page_count - 1, max(0, int(progress * page_count)))
            first = page * base + min(page, extra)
            count = base + (1 if page < extra else 0)
            return " ".join(words[first:first + count])
    return None


def local_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - shot["start"]) / max(1e-9, shot["end"] - shot["start"]))


# =============================================================================
# Cable data
# =============================================================================

# Offline fallback: approximate landing-city coordinates and representative systems.
# These are NOT full surveyed cable geometries; they are great-circle skeletons for
# layout/testing when the public GeoJSON cannot be downloaded.
FALLBACK_LINKS: List[Tuple[str, Tuple[float, float], Tuple[float, float]]] = [
    ("MAREA-like North Atlantic", (-75.98, 36.85), (-3.00, 43.26)),
    ("Dunant-like Atlantic", (-75.98, 36.85), (-1.94, 46.17)),
    ("Grace Hopper west leg", (-74.00, 40.71), (-4.54, 50.83)),
    ("Grace Hopper east leg", (-74.00, 40.71), (-3.00, 43.26)),
    ("EllaLink-like", (-8.87, 37.96), (-38.54, -3.73)),
    ("Monet-like", (-80.08, 26.37), (-38.54, -3.73)),
    ("South Atlantic", (-38.54, -3.73), (13.23, -8.84)),
    ("Brazil-South Africa", (-46.33, -23.96), (18.42, -33.93)),
    ("Curie-like", (-118.24, 34.05), (-71.61, -33.05)),
    ("Pacific US-Japan", (-124.16, 44.64), (140.11, 35.61)),
    ("Pacific US-Japan 2", (-122.42, 37.77), (139.69, 35.69)),
    ("Japan-Philippines", (139.69, 35.69), (120.98, 14.60)),
    ("Japan-Singapore", (139.69, 35.69), (103.82, 1.35)),
    ("Southern Cross", (-118.24, 34.05), (151.21, -33.87)),
    ("Hawaiki-like", (-124.16, 44.64), (174.76, -36.85)),
    ("Australia-New Zealand", (151.21, -33.87), (174.76, -36.85)),
    ("Australia-Singapore", (115.86, -31.95), (103.82, 1.35)),
    ("Perth-Jakarta", (115.86, -31.95), (106.85, -6.21)),
    ("SEA-ME-WE west", (5.37, 43.30), (31.23, 30.04)),
    ("SEA-ME-WE Red Sea", (31.23, 30.04), (43.15, 11.59)),
    ("SEA-ME-WE Arabian", (43.15, 11.59), (72.88, 19.08)),
    ("SEA-ME-WE east", (72.88, 19.08), (103.82, 1.35)),
    ("Europe-Africa west", (-9.14, 38.72), (-17.45, 14.69)),
    ("West Africa trunk", (-17.45, 14.69), (-0.19, 5.56)),
    ("West Africa south", (-0.19, 5.56), (18.42, -33.93)),
    ("East Africa north", (39.28, -6.82), (45.32, 2.05)),
    ("East Africa-Arabia", (45.32, 2.05), (55.27, 25.20)),
    ("India-Singapore", (72.88, 19.08), (80.27, 13.08)),
    ("Chennai-Singapore", (80.27, 13.08), (103.82, 1.35)),
    ("Singapore-Hong Kong", (103.82, 1.35), (114.17, 22.32)),
    ("Hong Kong-Tokyo", (114.17, 22.32), (139.69, 35.69)),
    ("Mediterranean west", (-5.35, 36.14), (14.27, 40.85)),
    ("Mediterranean east", (14.27, 40.85), (29.92, 31.20)),
    ("UK-Iceland", (-4.54, 50.83), (-21.94, 64.15)),
    ("Iceland-Canada", (-21.94, 64.15), (-52.71, 47.56)),
    ("Caribbean trunk", (-80.19, 25.76), (-66.11, 18.47)),
    ("Caribbean-South America", (-66.11, 18.47), (-66.90, 10.48)),
]


def lonlat_to_vec(lon: float, lat: float) -> np.ndarray:
    lo = math.radians(lon)
    la = math.radians(lat)
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)], dtype=float)


def vec_to_lonlat(v: np.ndarray) -> Tuple[float, float]:
    v = v / np.linalg.norm(v)
    lon = math.degrees(math.atan2(v[1], v[0]))
    lat = math.degrees(math.asin(clamp(v[2], -1.0, 1.0)))
    return lon, lat


def great_circle(a: Tuple[float, float], b: Tuple[float, float], n: int = 42) -> List[Tuple[float, float]]:
    va = lonlat_to_vec(*a)
    vb = lonlat_to_vec(*b)
    dot = float(np.clip(np.dot(va, vb), -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-8:
        return [a, b]
    so = math.sin(omega)
    out: List[Tuple[float, float]] = []
    for f in np.linspace(0.0, 1.0, n):
        v = math.sin((1-f)*omega)/so*va + math.sin(f*omega)/so*vb
        out.append(vec_to_lonlat(v))
    return out


def fallback_routes() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    routes: List[Dict[str, Any]] = []
    landing_map: Dict[Tuple[float, float], Dict[str, Any]] = {}
    for idx, (name, a, b) in enumerate(FALLBACK_LINKS):
        routes.append({"name": name, "coords": great_circle(a, b, 42), "feature_index": idx})
        for lon, lat in (a, b):
            landing_map[(lon, lat)] = {"name": "offline representative landing", "lon": lon, "lat": lat}
    return routes, list(landing_map.values())


def _cache_fresh(path: Path) -> bool:
    if not path.exists() or REFRESH:
        return False
    age_hours = (utc_now().timestamp() - path.stat().st_mtime) / 3600.0
    return age_hours <= float(CONFIG["cache_hours"])


def _fetch_json(url: str, cache_path: Path) -> Tuple[Dict[str, Any], str]:
    if _cache_fresh(cache_path):
        return json.loads(cache_path.read_text(encoding="utf-8")), "cache"
    if OFFLINE_MODE:
        raise RuntimeError("offline mode enabled")
    if requests is None:
        raise RuntimeError("requests is unavailable")
    response = requests.get(url, timeout=float(CONFIG["timeout_s"]), headers={"User-Agent": "SubmarineCables3DShort/1.0 educational renderer"})
    response.raise_for_status()
    cache_path.write_text(response.text, encoding="utf-8")
    return response.json(), "live"


def _extract_routes(geo: Dict[str, Any]) -> List[Dict[str, Any]]:
    routes: List[Dict[str, Any]] = []
    for idx, feature in enumerate(geo.get("features", [])):
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        name = str(props.get("name") or props.get("cable_name") or props.get("id") or f"cable-{idx}")
        segments: Iterable = []
        if gtype == "LineString":
            segments = [coords]
        elif gtype == "MultiLineString":
            segments = coords
        else:
            continue
        for seg_i, segment in enumerate(segments):
            if len(segment) < 2:
                continue
            # Downsample extremely dense surveyed polylines for animation speed.
            stride = max(1, int(math.ceil(len(segment) / (60 if QUICK_MODE else 100))))
            sampled = segment[::stride]
            if sampled[-1] != segment[-1]:
                sampled = list(sampled) + [segment[-1]]
            ll = [(float(p[0]), float(p[1])) for p in sampled if isinstance(p, (list, tuple)) and len(p) >= 2]
            if len(ll) >= 2:
                routes.append({"name": name, "coords": ll, "feature_index": idx, "segment_index": seg_i})
    return routes


def _extract_landings(geo: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for feature in geo.get("features", []):
        geom = feature.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        props = feature.get("properties") or {}
        out.append({
            "name": str(props.get("name") or props.get("location") or "landing point"),
            "lon": float(coords[0]),
            "lat": float(coords[1]),
        })
    return out


def collect_data() -> Tuple[CableSnapshot, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    errors: Dict[str, str] = {}
    mode = "offline-fallback"
    try:
        cable_geo, mode1 = _fetch_json(CONFIG["cable_geojson_url"], CACHE_ROOT / "cable-geo.json")
        landing_geo, mode2 = _fetch_json(CONFIG["landing_geojson_url"], CACHE_ROOT / "landing-point-geo.json")
        routes = _extract_routes(cable_geo)
        landings = _extract_landings(landing_geo)
        if not routes:
            raise RuntimeError("downloaded GeoJSON contained no route geometries")
        mode = "live" if "live" in (mode1, mode2) else "cache"
        offline = False
    except Exception as exc:
        errors["public_data"] = str(exc)
        routes, landings = fallback_routes()
        offline = True

    # Quick mode caps route count only for frame speed; full render uses all loaded routes.
    if QUICK_MODE and len(routes) > 180:
        step = max(1, len(routes) // 180)
        routes = routes[::step][:180]
    if QUICK_MODE and len(landings) > 500:
        step = max(1, len(landings) // 500)
        landings = landings[::step][:500]

    snapshot = CableSnapshot(
        generated_at_utc=iso_z(utc_now()),
        data_status=mode,
        offline_fallback=offline,
        route_feature_count=len(routes),
        landing_point_count_loaded=len(landings),
        public_dataset_refresh_note="repository metadata says last refreshed 2026-03-12",
        telegeography_2026_systems_active_and_planned=TELEGEOGRAPHY_2026_SYSTEMS,
        telegeography_2026_landing_stations=TELEGEOGRAPHY_2026_LANDINGS,
        international_data_share_percent_gt=ITU_INTL_DATA_SHARE,
        reported_repairs_2023_gt=ITU_REPAIRS_2023,
        interpretation="Cable paths come from public GeoJSON when available. Offline fallback paths are representative great-circle skeletons. Globe depth, glow, packets and line thickness are illustrative.",
    )
    summary = {
        "generated_at_utc": snapshot.generated_at_utc,
        "errors": errors,
        "source_urls": [CONFIG["dataset_repo"], CONFIG["telegeography_2026"], CONFIG["itu_report"], CONFIG["itu_resilience"]],
        "license_note": "TeleGeography-derived public route data: CC BY-NC-SA 3.0 per repository README.",
    }
    return snapshot, routes, landings, summary


def save_data(snapshot: CableSnapshot, routes: List[Dict[str, Any]], landings: List[Dict[str, Any]], summary: Dict[str, Any]):
    route_rows = []
    for i, route in enumerate(routes):
        coords = route["coords"]
        route_rows.append({
            "route_index": i,
            "name": route.get("name", f"route-{i}"),
            "vertex_count_rendered": len(coords),
            "start_lon": coords[0][0],
            "start_lat": coords[0][1],
            "end_lon": coords[-1][0],
            "end_lat": coords[-1][1],
            "data_status": snapshot.data_status,
        })
    df = pd.DataFrame(route_rows)
    csv_path = DATA_ROOT / "submarine_cable_route_summary.csv"
    df.to_csv(csv_path, index=False)
    json_path = DATA_ROOT / "submarine_cable_snapshot.json"
    json_path.write_text(json.dumps({"snapshot": asdict(snapshot), "summary": summary}, indent=2), encoding="utf-8")
    note_path = DATA_ROOT / "source_notes.txt"
    note_path.write_text(
        "Earth's Submarine Cables in 3D — source notes\n\n"
        "Public route dataset: " + CONFIG["dataset_repo"] + "\n"
        "TeleGeography 2026 map context: " + CONFIG["telegeography_2026"] + "\n"
        "ITU Global Connectivity Report 2025: " + CONFIG["itu_report"] + "\n"
        "ITU submarine cable resilience: " + CONFIG["itu_resilience"] + "\n\n"
        + snapshot.interpretation + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path, note_path


# =============================================================================
# Coarse coastline outlines for context only
# =============================================================================

BUILTIN_CONTINENTS: Dict[str, List[Tuple[float, float]]] = {
    "north_america": [(-168,72),(-140,69),(-125,55),(-124,42),(-117,32),(-103,22),(-97,18),(-83,10),(-77,8),(-80,25),(-69,45),(-60,52),(-58,65),(-85,75),(-120,76),(-168,72)],
    "south_america": [(-81,12),(-70,8),(-51,5),(-35,-7),(-40,-23),(-54,-38),(-69,-55),(-75,-42),(-80,-20),(-81,12)],
    "greenland": [(-73,60),(-47,60),(-20,72),(-28,82),(-55,83),(-73,60)],
    "europe_asia": [(-11,36),(-10,58),(5,70),(30,71),(60,75),(100,72),(140,58),(170,55),(179,48),(150,35),(123,23),(105,7),(79,8),(60,24),(35,32),(20,34),(-11,36)],
    "africa": [(-17,36),(10,37),(33,31),(44,12),(51,-10),(39,-35),(18,-35),(5,-30),(-10,-5),(-17,15),(-17,36)],
    "australia": [(112,-11),(130,-10),(153,-28),(145,-43),(116,-35),(112,-11)],
    "antarctica": [(-180,-70),(-120,-73),(-60,-72),(0,-76),(60,-71),(120,-74),(180,-70)],
}

# Bundled low-resolution GSHHS coastline geometry for accurate world outlines.
ACCURATE_LAND_POLYGONS = [[[-180.0, 68.9938], [-179.9984, 68.9929], [-175.2567, 67.6588], [-173.945, 66.1446], [-174.6259, 67.0517], [-173.6301, 67.1171],
  [-171.6667, 66.9446], [-169.674, 66.1365], [-170.5651, 65.6129], [-171.4435, 65.8321], [-171.05, 65.4671], [-173.0092, 65.7017],
  [-172.0859, 65.4867], [-172.1193, 65.0491], [-173.2109, 64.7575], [-172.2358, 64.4025], [-173.15, 64.2396], [-173.325, 64.6113],
  [-173.6534, 64.3212], [-175.4351, 64.7637], [-176.4951, 65.5371], [-178.4452, 65.4671], [-178.451, 66.3617], [-179.3817, 66.3213],
  [-179.2759, 65.5266], [-180.0, 65.0845], [-180.0, 68.9938]],
 [[180.0, 65.0845], [179.525, 64.7946], [175.1426, 64.7224], [176.1667, 64.6956], [176.0716, 64.4471], [177.5083, 64.7413],
  [178.2484, 64.1671], [178.234, 64.4333], [179.6226, 62.7274], [179.1483, 62.2854], [177.115, 62.5462], [173.5317, 61.7504],
  [170.5491, 60.4267], [170.31, 59.9271], [169.2533, 60.6129], [167.0117, 60.4129], [166.2784, 59.8112], [166.275, 60.4721],
  [164.8366, 59.7829], [164.4866, 60.1012], [163.6373, 59.9986], [161.8907, 57.9774], [163.3192, 57.7275], [162.6759, 57.3366],
  [163.3608, 56.1875], [162.0875, 56.0942], [162.1441, 54.7483], [159.9841, 54.1541], [160.0491, 53.0899], [158.4173, 52.9986],
  [158.5424, 52.2892], [156.6734, 50.8621], [155.5625, 55.1758], [155.9891, 56.6783], [156.8508, 57.7975], [158.3199, 57.9771],
  [161.9416, 60.4179], [163.7891, 60.8225], [164.1491, 62.2708], [165.3086, 62.469], [164.6099, 62.6813], [163.2616, 62.5179],
  [163.3291, 61.6608], [162.3883, 61.662], [160.2134, 60.5788], [160.4408, 61.0333], [159.7917, 60.9271], [159.7725, 61.2383],
  [160.3966, 61.9554], [156.6816, 61.5221], [154.2308, 59.8733], [154.1291, 59.4524], [154.9799, 59.4929], [155.1884, 59.1712],
  [151.3416, 58.8346], [151.1034, 59.1062], [152.3208, 59.2249], [149.4334, 59.7588], [148.9317, 59.2304], [146.505, 59.4629],
  [146.0467, 59.1387], [145.6916, 59.4263], [142.4617, 59.2021], [138.2275, 56.4208], [135.2274, 54.91], [135.7433, 54.5654],
  [136.8325, 54.6467], [136.7833, 53.7645], [137.7358, 54.3208], [137.3133, 53.5346], [138.58, 53.9954], [138.6525, 54.29],
  [139.7416, 54.3004], [141.4241, 53.295], [141.2033, 52.9979], [140.6308, 53.1458], [141.5458, 52.1508], [140.4374, 50.7175],
  [140.3913, 48.9733], [138.1058, 46.2362], [135.1358, 43.5021], [133.0433, 42.6729], [131.76, 43.3378], [131.2209, 42.5529],
  [130.6734, 42.6754], [130.6996, 42.2974], [129.7596, 41.765], [129.7158, 40.8337], [127.5071, 39.7367], [127.3788, 39.2133],
  [128.3609, 38.6879], [129.4312, 37.0599], [129.5838, 36.0158], [129.0908, 35.0504], [128.5892, 35.2112], [128.4241, 34.7612],
  [127.7208, 34.9987], [127.8033, 34.5879], [127.5158, 34.8887], [127.34, 34.4446], [127.2425, 34.7771], [126.525, 34.2929],
  [126.2796, 34.5958], [126.6221, 34.6266], [126.2396, 35.1117], [126.8938, 36.1416], [126.4933, 36.1337], [126.5112, 36.6742],
  [126.4191, 36.3996], [126.1095, 36.7733], [126.4966, 37.0603], [126.8579, 36.7849], [126.5203, 37.7658], [125.6275, 38.0412],
  [125.5151, 37.8954], [124.9762, 37.9308], [125.26, 38.0854], [124.652, 38.1283], [124.9608, 38.5929], [125.6037, 38.6491],
  [125.1262, 38.8767], [125.5592, 39.6654], [123.6258, 39.9354], [121.1392, 38.7245], [121.9429, 39.4033], [121.2487, 39.385],
  [122.2779, 40.465], [121.8841, 41.0487], [120.9733, 40.8304], [119.005, 39.1971], [117.7179, 39.1058], [117.6912, 38.3891],
  [118.8399, 38.2137], [119.185, 37.1421], [120.7417, 37.8329], [122.6996, 37.4075], [122.5167, 36.897], [121.9692, 37.0246],
  [120.7775, 36.6279], [120.7142, 36.1338], [120.1267, 36.2038], [120.2979, 35.97], [119.2012, 35.005], [119.1841, 34.6904],
  [120.2846, 34.3033], [120.8504, 32.6166], [121.9104, 31.7366], [120.0559, 31.9671], [120.788, 31.8527], [121.9121, 30.9383],
  [120.1379, 30.1958], [121.2641, 30.3279], [122.1379, 29.8925], [121.4271, 29.475], [121.9979, 29.5908], [121.9742, 29.2229],
  [121.4187, 29.1633], [121.7321, 28.9417], [121.346, 28.7056], [121.6254, 28.2658], [120.5829, 28.1033], [120.8587, 27.885],
  [120.1208, 26.6362], [119.5429, 26.8133], [119.8313, 26.45], [119.58, 26.4237], [119.9587, 26.3758], [119.5121, 26.075],
  [119.6546, 25.3558], [119.3292, 25.6146], [119.2816, 25.1712], [118.8696, 25.2475], [118.9859, 24.8812], [118.5629, 24.9083],
  [118.6509, 24.5554], [117.7799, 24.5062], [118.1346, 24.265], [117.585, 23.7154], [117.3525, 23.9487], [117.2441, 23.6046],
  [116.4574, 23.4487], [116.7404, 23.25], [116.4958, 22.9396], [115.5725, 22.6588], [115.3141, 22.9029], [114.8817, 22.5446],
  [114.78, 22.8371], [114.2087, 22.5375], [114.2975, 22.2604], [113.3683, 23.1113], [113.5474, 22.1886], [113.2833, 22.3686],
  [113.1141, 22.072], [112.9217, 22.5146], [112.9125, 21.8521], [112.365, 21.9962], [112.3434, 21.7079], [111.675, 21.7962],
  [111.6383, 21.5137], [110.4467, 21.1879], [110.3817, 21.4079], [110.285, 20.2412], [109.9262, 20.2292], [109.6654, 20.9359],
  [109.9654, 21.4833], [109.6091, 21.7587], [109.1608, 21.4054], [108.5541, 21.9203], [108.4774, 21.5629], [107.7809, 21.5238],
  [107.3491, 20.9996], [106.6053, 21.0366], [106.8129, 20.6733], [106.5525, 20.7596], [106.5729, 20.2833], [105.927, 19.8842],
  [105.6412, 18.9007], [107.1319, 16.8541], [108.3267, 16.1429], [108.9421, 15.245], [109.4671, 12.6475], [109.1962, 12.63],
  [109.0191, 11.3554], [107.2592, 10.3754], [106.7457, 10.5728], [106.7791, 10.277], [106.4353, 10.3108], [106.7783, 10.0812],
  [106.3562, 10.1925], [106.6166, 9.8129], [106.1791, 10.1754], [106.5025, 9.5537], [105.8367, 9.9937], [106.1988, 9.375],
  [105.2808, 8.7529], [104.7937, 8.8008], [105.0878, 10.0049], [103.8642, 10.7079], [103.6142, 10.5004], [103.5583, 11.1654],
  [103.1275, 10.8671], [102.6442, 12.1779], [101.7892, 12.7087], [100.8558, 12.6462], [100.9883, 13.5004], [99.9511, 13.3092],
  [100.0246, 12.1875], [99.1505, 10.3441], [99.2246, 9.255], [99.8462, 9.2908], [100.4258, 7.1604], [101.5684, 6.8386], [103.1513, 5.34],
  [103.4779, 4.5333], [103.4579, 2.825], [104.2766, 1.3679], [103.9408, 1.6629], [103.5117, 1.2695], [101.2913, 2.8467], [100.5579, 4.315],
  [100.1153, 6.5541], [98.6708, 8.3904], [98.2854, 8.2033], [98.7627, 10.3258], [98.5526, 9.9737], [98.4556, 10.6867], [98.8854, 11.7],
  [98.542, 11.8824], [98.7204, 12.81], [98.2241, 13.9729], [98.1262, 13.5525], [97.7879, 14.8708], [97.7421, 16.5625], [97.3538, 16.5049],
  [96.8739, 17.5094], [96.8929, 16.8192], [96.4508, 16.4896], [96.1175, 16.8004], [96.3237, 16.4208], [95.7175, 16.2471],
  [95.4558, 15.7346], [95.1277, 16.1201], [94.9558, 15.7412], [94.6367, 16.0941], [94.3942, 15.8296], [94.6227, 16.2563],
  [94.2425, 15.9529], [94.5979, 17.5533], [94.0862, 19.3642], [93.9367, 18.8437], [93.4696, 19.3667], [94.0229, 19.3716], [93.7554, 19.915],
  [92.9958, 20.1379], [93.08, 20.5854], [92.7612, 20.1983], [91.4474, 22.7636], [90.6866, 22.7629], [90.8313, 22.1541], [90.6237, 22.0],
  [90.5333, 22.6054], [90.1942, 21.8029], [89.9024, 22.3905], [89.7783, 21.8252], [89.5224, 22.0854], [89.2191, 21.7478],
  [89.0424, 22.0754], [89.0266, 21.5986], [88.6391, 22.0762], [88.2466, 21.5612], [88.0196, 22.2229], [87.8042, 21.6962],
  [86.9121, 21.3383], [87.0696, 20.7208], [86.3716, 19.9521], [85.0383, 19.3921], [84.1292, 18.3096], [82.3062, 17.0383],
  [82.3033, 16.5587], [81.2675, 16.2929], [80.9408, 15.7112], [80.2637, 15.6717], [80.0479, 15.0741], [80.3462, 13.2833],
  [79.7588, 11.6716], [79.8804, 10.3083], [79.2941, 10.2604], [78.9004, 9.4867], [79.1896, 9.2808], [78.2658, 9.0179], [78.0704, 8.3741],
  [77.5375, 8.0737], [76.5471, 8.8983], [75.8696, 11.1241], [73.4571, 16.0533], [72.6554, 19.8333], [72.9304, 20.76], [72.5979, 21.2983],
  [72.9279, 21.6758], [72.54, 21.6604], [72.5079, 21.9758], [72.9108, 22.2662], [72.3267, 22.3087], [72.1113, 21.1991], [70.8225, 20.6904],
  [68.9354, 22.3067], [70.1742, 22.542], [70.4479, 22.97], [69.1975, 22.8354], [68.4254, 23.5092], [68.8016, 23.8846], [68.2658, 23.5563],
  [68.3466, 23.9662], [67.4808, 23.9129], [67.2946, 24.7749], [66.6574, 24.8321], [66.4566, 25.6179], [61.7566, 25.007], [57.315, 25.7829],
  [56.68, 27.1996], [54.7958, 26.4954], [53.7291, 26.7004], [52.4358, 27.6446], [51.4017, 27.9221], [50.0954, 30.1783], [49.5542, 30.0004],
  [48.9254, 30.3849], [48.8833, 30.0095], [47.9565, 30.0711], [48.1879, 29.5466], [47.7029, 29.3741], [48.1004, 29.3525],
  [48.8471, 27.6016], [50.1646, 26.6366], [50.003, 25.9874], [50.7708, 24.7137], [50.9914, 25.9832], [51.5778, 25.9141], [51.2984, 24.2937],
  [52.1108, 23.9066], [52.5941, 24.1954], [54.0934, 24.1204], [56.5058, 26.3596], [56.2663, 25.6308], [56.6236, 24.4816],
  [57.1816, 23.9321], [58.7533, 23.522], [59.8079, 22.22], [58.5196, 20.41], [58.195, 20.6096], [57.8453, 20.2383], [57.8412, 18.9992],
  [56.7983, 18.7462], [56.3517, 17.9187], [55.4283, 17.8179], [55.02, 17.0012], [52.4833, 16.4396], [52.1958, 15.6079], [49.3424, 14.6379],
  [48.6983, 14.0421], [45.6583, 13.347], [45.0399, 12.7512], [43.4633, 12.6745], [42.5987, 15.2308], [42.7038, 16.7392], [40.7954, 19.7291],
  [39.1979, 21.085], [39.1421, 22.4049], [38.4554, 23.7817], [37.4279, 24.3708], [37.2396, 25.1883], [35.2212, 28.0467], [34.5754, 28.0883],
  [35.002, 29.5267], [34.2426, 27.7246], [32.5669, 29.9272], [32.3052, 31.2029], [34.237, 31.3383], [35.9888, 34.5373], [35.7136, 35.5867],
  [36.1537, 36.8373], [35.355, 36.5329], [34.64, 36.7971], [33.9616, 36.2204], [32.7925, 36.0262], [31.1708, 36.8504], [30.6367, 36.8562],
  [30.405, 36.1996], [29.6849, 36.1312], [28.4617, 36.8796], [27.4049, 36.6629], [28.3354, 37.0542], [27.2641, 36.9637], [27.6471, 37.2441],
  [27.0046, 37.6575], [27.2562, 37.9758], [26.2316, 38.2654], [26.4183, 38.6795], [26.6766, 38.3112], [27.1737, 38.4458], [26.7221, 38.65],
  [26.9479, 39.565], [26.0725, 39.4704], [26.1487, 39.915], [26.7474, 40.4046], [28.985, 40.3562], [28.975, 40.6404], [29.9358, 40.7179],
  [29.2525, 40.8013], [29.1616, 41.2212], [31.2108, 41.0996], [33.3416, 42.0262], [35.0133, 42.0987], [36.4091, 41.2445],
  [38.3774, 40.9088], [40.1708, 40.9212], [41.4049, 41.377], [41.7795, 41.8425], [41.4358, 42.7371], [36.6004, 45.1933], [37.5325, 45.3796],
  [38.2749, 46.2588], [38.59, 46.0371], [37.6962, 46.7025], [38.5933, 46.6621], [39.2625, 47.2621], [35.9208, 46.6554], [34.9908, 46.0771],
  [35.1858, 46.5279], [34.4654, 45.915], [34.9733, 45.3612], [35.4637, 45.2825], [34.815, 46.1462], [35.465, 45.3012], [36.6463, 45.3741],
  [33.9766, 44.3879], [33.3821, 44.5792], [33.2033, 45.4421], [32.4804, 45.3867], [33.6866, 45.8329], [33.5974, 46.1537],
  [31.7875, 46.2671], [32.0279, 46.4383], [31.5091, 46.5796], [32.4462, 46.4883], [31.9891, 46.8896], [30.7592, 46.5496], [30.48, 46.0779],
  [30.1408, 46.3821], [30.4771, 46.0733], [29.7208, 45.5829], [30.1312, 45.9025], [29.7424, 45.602], [29.6312, 45.8241], [29.6087, 44.8508],
  [28.995, 44.6829], [28.8633, 44.9538], [28.5836, 43.4841], [27.3062, 42.3833], [27.7217, 42.4088], [28.9816, 40.9987], [27.5158, 40.9762],
  [26.1824, 40.0395], [26.8082, 40.6454], [25.1366, 41.012], [23.7333, 40.7537], [24.4029, 40.1558], [23.735, 40.3554], [23.9491, 39.9388],
  [22.9283, 40.6371], [22.5854, 40.4667], [23.3504, 39.1858], [22.9358, 39.3587], [23.0571, 39.0158], [22.527, 38.8575], [24.0813, 38.1625],
  [24.0308, 37.6471], [23.52, 38.0437], [22.9937, 37.8775], [23.5158, 37.4304], [22.7179, 37.5608], [23.1983, 36.4346], [22.635, 36.8054],
  [22.4825, 36.3879], [22.1258, 37.0246], [21.7037, 36.8141], [21.1045, 37.8508], [21.3687, 38.2175], [22.8958, 37.9337], [23.2254, 38.155],
  [21.0991, 38.3345], [20.7287, 38.815], [21.1178, 39.039], [20.7258, 38.9504], [19.2887, 40.42], [19.595, 41.8121], [18.7625, 42.4904],
  [17.0083, 43.0054], [17.7421, 42.8342], [16.8867, 43.3996], [15.9517, 43.5046], [14.5591, 45.2962], [13.905, 44.7671], [13.544, 45.7946],
  [12.1429, 45.3866], [12.3958, 44.2054], [13.6267, 43.5521], [14.7883, 42.0454], [16.1533, 41.9096], [15.9616, 41.4604],
  [18.5162, 40.1358], [18.3475, 39.7904], [17.8583, 40.2837], [16.8949, 40.4304], [16.4887, 39.7592], [17.2087, 39.0275], [16.605, 38.8129],
  [16.065, 37.9238], [15.6345, 38.0175], [16.2212, 38.8533], [15.6767, 40.0304], [11.0933, 42.3946], [10.18, 43.9612], [8.7475, 44.4271],
  [6.1583, 43.0296], [4.085, 43.5612], [3.3341, 43.2871], [2.9512, 42.8333], [3.2033, 41.8912], [0.9799, 41.0346], [0.0, 39.9413],
  [-0.1804, 39.74], [-0.2421, 39.1308], [0.0, 38.9318], [0.2354, 38.7383], [0.0, 38.6097], [-0.5046, 38.3341], [-0.7175, 37.6071],
  [-1.4517, 37.4921], [-2.1258, 36.7304], [-4.415, 36.7187], [-5.6084, 36.0004], [-6.8667, 37.2929], [-8.9954, 37.0225], [-8.8567, 38.385],
  [-8.6304, 38.4083], [-9.2254, 38.4117], [-8.9458, 38.9971], [-9.5004, 38.7833], [-8.7929, 40.1183], [-8.5997, 42.345], [-9.3021, 43.0542],
  [-7.6884, 43.7921], [-7.0492, 43.4812], [-1.6051, 43.4304], [-1.0959, 45.5596], [-1.238, 45.6925], [-1.1013, 46.3167], [-1.8125, 46.4929],
  [-2.39, 47.5062], [-4.3767, 47.7962], [-4.6334, 48.2834], [-4.1576, 48.2501], [-4.7801, 48.5068], [-3.0767, 48.8838], [-2.6867, 48.4962],
  [-1.3575, 48.6338], [-1.94, 49.7279], [0.0, 49.3279], [0.0, 49.3279], [1.6658, 50.1833], [1.5833, 50.8729], [4.2503, 51.3521],
  [3.4306, 51.5333], [4.6458, 51.7133], [4.0075, 51.9633], [4.7333, 52.9671], [5.2533, 52.3103], [5.8441, 52.5841], [5.3341, 53.0758],
  [5.8783, 53.3963], [7.4284, 53.2225], [7.093, 53.5857], [7.9567, 53.7183], [8.5057, 53.3691], [8.6683, 53.8946], [9.8258, 53.5316],
  [8.5741, 54.3016], [9.014, 54.4866], [8.6108, 54.8817], [8.2757, 54.7525], [8.5966, 57.1212], [10.6575, 57.7375], [10.1241, 56.7116],
  [10.9608, 56.4417], [9.5591, 55.7075], [9.4208, 54.8282], [10.1433, 54.3204], [11.1324, 54.3908], [10.8882, 53.9595], [13.0216, 54.4396],
  [14.5912, 53.6017], [14.7316, 54.0262], [17.9483, 54.8321], [18.7966, 54.3437], [19.8783, 54.6404], [19.3733, 54.2204],
  [20.4125, 54.6775], [19.8899, 54.6378], [19.9357, 54.9166], [20.5676, 55.0818], [21.0834, 55.7271], [20.5143, 54.955], [21.2475, 54.9507],
  [20.9807, 56.5225], [21.6975, 57.5666], [22.6099, 57.7604], [23.6366, 56.9619], [24.3939, 57.2401], [24.5858, 58.3224], [23.7358, 58.35],
  [23.5033, 59.2304], [25.6966, 59.6762], [27.9149, 59.4029], [28.0982, 59.7904], [30.3466, 59.9504], [28.6233, 60.3528],
  [28.6582, 60.7428], [22.8934, 59.8087], [23.0924, 60.37], [22.52, 60.2129], [21.4258, 60.6034], [21.7975, 61.4916], [21.1241, 62.8],
  [25.5092, 65.0216], [24.5917, 65.8436], [22.6399, 65.9071], [22.3966, 65.5346], [21.7458, 65.7517], [21.7466, 65.1913],
  [21.2325, 65.3334], [21.6191, 64.4433], [20.6741, 63.7958], [18.1216, 62.7688], [17.6842, 63.005], [17.1757, 60.6824], [17.99, 60.6046],
  [19.0908, 59.7483], [18.0841, 59.4233], [18.7124, 59.2783], [17.9232, 58.8038], [17.6533, 59.1845], [17.5067, 58.7821],
  [16.1874, 58.6417], [16.9491, 58.4775], [16.4158, 58.4775], [16.7291, 57.4441], [15.8483, 56.0737], [14.6883, 56.1671],
  [14.1966, 55.3821], [12.8125, 55.3783], [12.4458, 56.3041], [12.9408, 56.5833], [11.7025, 57.7049], [11.9107, 58.3417],
  [11.5683, 57.9279], [11.7057, 58.4367], [11.2107, 58.3508], [10.7533, 59.9087], [10.4224, 59.0634], [9.5183, 59.1346], [8.2167, 58.1087],
  [7.0483, 57.9804], [5.6625, 58.5508], [5.5742, 59.0325], [6.2858, 58.8392], [6.6474, 59.0525], [5.8891, 59.0841], [6.5974, 59.5592],
  [5.6483, 59.2588], [5.2242, 59.5258], [7.1408, 60.4975], [5.5466, 59.8863], [5.7658, 60.3908], [5.4416, 60.1321], [5.1692, 60.3742],
  [5.7991, 60.7341], [5.2766, 60.5404], [4.9475, 60.8083], [5.5892, 60.8717], [5.0225, 61.0342], [6.6466, 61.1754], [7.1234, 60.8621],
  [7.7125, 61.2342], [7.5867, 61.4937], [7.3283, 61.1571], [6.7533, 61.4187], [6.5178, 61.1974], [5.2317, 61.1046], [5.6741, 61.3683],
  [4.9791, 61.4216], [5.8559, 61.4558], [4.9974, 61.58], [5.1917, 61.8821], [6.8458, 61.8708], [5.1784, 61.8954], [5.0991, 62.1933],
  [6.3133, 62.0596], [5.9308, 62.2225], [6.5516, 62.1045], [6.684, 62.34], [7.1983, 62.0988], [6.259, 62.585], [8.1475, 62.695],
  [6.9833, 62.9562], [8.5316, 62.6687], [7.8825, 62.99], [8.6991, 62.8167], [8.5149, 63.3821], [9.6752, 63.6013], [9.8266, 63.3129],
  [10.9609, 63.4475], [11.385, 64.1129], [10.4216, 63.5695], [9.5024, 63.6458], [10.9549, 64.6079], [11.2183, 64.3121], [11.8603, 64.4484],
  [11.3859, 64.6875], [12.17, 64.9596], [11.2883, 64.8787], [12.9608, 65.3141], [12.0441, 65.2142], [12.6692, 65.9233], [13.2008, 65.8283],
  [12.6766, 66.0713], [14.1708, 66.3232], [13.0175, 66.1783], [13.9908, 66.7975], [13.4891, 66.9517], [15.4725, 67.1], [14.3108, 67.2641],
  [15.8874, 67.5791], [14.8499, 67.8779], [15.865, 67.9079], [15.2775, 68.0625], [16.0032, 68.2671], [16.5166, 67.817], [16.2849, 68.3763],
  [17.9049, 68.4145], [16.4558, 68.5075], [17.6906, 68.6708], [18.1482, 69.4654], [19.5498, 69.2146], [18.9392, 69.6242], [19.72, 69.8129],
  [19.6466, 69.4146], [20.2882, 69.9788], [19.9099, 69.2712], [21.235, 70.0121], [22.1183, 69.7363], [21.5625, 70.3229], [23.4682, 69.9812],
  [24.6399, 70.9921], [25.9137, 70.8741], [25.1525, 70.0654], [26.6599, 70.9679], [26.4861, 70.3583], [27.091, 70.4632], [27.6574, 71.1312],
  [28.5538, 70.97], [27.8548, 70.427], [28.9724, 70.8846], [31.0312, 70.3975], [30.1824, 70.0696], [28.5561, 70.1725], [29.4898, 69.6562],
  [30.2982, 69.8804], [31.2617, 69.557], [32.0917, 69.9412], [33.1308, 69.7167], [32.1016, 69.7421], [32.155, 69.4121], [33.4991, 69.4017],
  [33.0316, 68.8829], [33.5682, 69.2996], [35.3066, 69.2513], [39.7767, 68.1496], [41.3907, 67.1151], [39.1117, 66.1009],
  [31.8695, 67.1391], [33.7131, 66.4126], [32.9768, 66.2475], [34.8597, 65.8648], [34.2446, 65.3752], [34.7835, 64.5121],
  [37.4175, 63.7814], [38.042, 64.2903], [36.5337, 64.7148], [36.8491, 65.1526], [38.3666, 64.8438], [38.0036, 64.5806], [38.4325, 64.8076],
  [40.4636, 64.5122], [39.7007, 65.4656], [40.74, 65.9847], [42.211, 66.5203], [44.1062, 65.9619], [44.2208, 68.2692], [43.2816, 68.6721],
  [46.5459, 68.125], [46.6934, 67.8067], [44.9245, 67.4244], [46.019, 66.8165], [47.56, 66.8926], [47.8263, 67.5773], [49.0718, 67.6003],
  [48.5928, 67.9132], [52.2166, 68.5738], [52.3266, 68.3012], [53.8734, 68.9596], [54.5658, 68.9783], [53.6092, 68.8003],
  [53.2183, 68.1862], [54.85, 68.1396], [55.3616, 68.5404], [57.3033, 68.5321], [59.0033, 68.9937], [59.7649, 68.3379], [59.9449, 68.7162],
  [61.189, 68.8083], [60.1842, 69.565], [60.8449, 69.8646], [66.1849, 69.0779], [65.5291, 69.1433], [68.2583, 68.1271], [69.2641, 68.9375],
  [68.1016, 69.5296], [66.8074, 69.5517], [66.7288, 71.1067], [68.3388, 71.5867], [69.3962, 72.9475], [71.6324, 72.9004],
  [72.8786, 72.6933], [71.8112, 71.4642], [72.8111, 70.87], [72.5391, 68.9533], [73.6258, 68.4233], [71.7499, 66.9046], [69.1274, 66.6125],
  [72.1332, 66.1988], [74.7725, 67.6999], [74.6666, 68.7662], [76.5848, 68.9662], [77.0941, 67.7617], [78.7842, 67.4683],
  [77.4858, 67.7441], [78.1841, 68.245], [77.6516, 68.892], [75.9417, 69.2546], [73.7642, 69.1566], [74.3261, 70.5908], [73.0087, 71.3967],
  [75.0188, 72.19], [74.8288, 72.8308], [76.1325, 71.1612], [78.5087, 70.9041], [75.9662, 71.8858], [77.9849, 71.8262], [77.4637, 72.2075],
  [78.4475, 72.3862], [81.4524, 71.7196], [83.2637, 71.7217], [82.0137, 70.5558], [82.8474, 70.9587], [82.4236, 70.1708],
  [83.6216, 69.7179], [83.6688, 71.6408], [80.8512, 72.4283], [80.5062, 73.5575], [87.2366, 73.8349], [85.9174, 74.8521],
  [87.0075, 74.6013], [86.9833, 75.1346], [94.1151, 75.9167], [92.8364, 75.8787], [93.2734, 76.1046], [96.1266, 76.1171],
  [95.7032, 75.8262], [98.8267, 76.2596], [99.8451, 76.0425], [98.8782, 76.485], [101.1283, 76.4641], [101.3549, 77.0892],
  [104.1467, 77.7162], [106.345, 77.3658], [104.1665, 77.0829], [107.5448, 76.914], [106.5332, 76.4996], [111.0934, 76.7654],
  [113.9349, 75.835], [113.5833, 75.5312], [112.3483, 75.84], [113.5783, 75.2125], [107.0174, 73.6321], [105.4998, 72.7446],
  [106.3399, 73.1896], [111.0349, 73.6879], [109.69, 73.6737], [110.0223, 74.0062], [111.5725, 74.0521], [112.3075, 73.6896],
  [112.845, 73.9871], [113.5147, 73.2027], [115.6873, 73.7054], [118.8799, 73.5304], [118.4061, 73.225], [119.8477, 72.9575],
  [122.8425, 72.8525], [124.4374, 73.8012], [126.2813, 73.2592], [128.8162, 73.2], [129.4512, 72.6517], [128.4428, 72.518],
  [129.5561, 72.2125], [128.3061, 72.0808], [131.0449, 70.7029], [132.7799, 71.9512], [133.8176, 71.4146], [136.0224, 71.627],
  [137.8626, 71.1004], [138.115, 71.5704], [140.1173, 71.4321], [139.3061, 71.9425], [140.2011, 72.2783], [139.1237, 72.2525],
  [141.3013, 72.5817], [140.8901, 72.8721], [146.8625, 72.3629], [144.2448, 72.2487], [146.9438, 72.3175], [145.9824, 71.8629],
  [145.4986, 72.2483], [145.3475, 71.5546], [147.115, 72.3154], [148.3674, 72.3313], [150.0637, 71.9291], [148.8287, 71.6891],
  [152.5312, 70.815], [155.9224, 71.0887], [158.9249, 70.8904], [161.1133, 69.1604], [162.3467, 69.6738], [166.8734, 69.4621],
  [167.8084, 69.7645], [168.3216, 69.2221], [170.4816, 68.7637], [171.1792, 69.03], [170.1624, 69.5975], [170.5499, 70.1196],
  [173.2232, 69.7654], [176.0816, 69.8837], [180.0, 68.9938], [180.0, 65.0845]],
 [[31.9192, 31.5312], [32.2845, 31.2328], [32.5587, 29.9351], [32.4784, 29.9404], [32.6254, 28.9766], [33.5862, 27.8475],
  [35.1387, 24.5133], [35.7937, 23.9092], [35.4762, 23.9342], [35.6679, 22.9466], [36.8954, 22.0633], [37.3187, 21.065], [37.1112, 21.2142],
  [37.4121, 18.875], [38.5571, 18.0601], [39.712, 15.095], [39.875, 15.5079], [40.21, 14.9487], [41.19, 14.6237], [43.327, 12.4758],
  [43.3762, 11.9884], [42.5275, 11.5229], [43.1566, 11.6129], [44.5758, 10.3754], [45.8058, 10.8679], [46.4408, 10.6837],
  [47.4125, 11.1796], [49.4267, 11.3379], [50.7976, 11.9861], [51.2912, 11.8333], [51.0221, 10.4208], [51.4154, 10.4433],
  [50.9021, 10.3149], [50.8304, 9.4192], [47.9654, 4.4775], [46.0117, 2.4295], [43.5592, 0.7146], [41.315, -1.9613], [40.7746, -1.9342],
  [40.9879, -2.2642], [40.1863, -2.7358], [39.2019, -4.6809], [38.7729, -6.0525], [39.5513, -7.0175], [39.2171, -7.8508],
  [39.4471, -7.8134], [39.2554, -8.2925], [39.7044, -10.0318], [40.6421, -10.6843], [40.3321, -11.3133], [40.6421, -12.7467],
  [40.3946, -12.9284], [40.8421, -14.8325], [39.8566, -16.4388], [36.855, -17.8788], [36.3125, -18.8771], [34.885, -19.8529],
  [34.6312, -19.6508], [34.6612, -20.5551], [35.5462, -22.1717], [35.5021, -24.1075], [32.4987, -25.9601], [32.8341, -26.2929],
  [32.9644, -26.0841], [32.3854, -28.5492], [31.2746, -29.4542], [30.0092, -31.2971], [27.0116, -33.5654], [24.84, -34.2063],
  [22.5558, -33.9787], [20.02, -34.8304], [18.7908, -34.0804], [18.3999, -34.3021], [17.8479, -32.82], [18.3271, -32.53],
  [18.2004, -31.6825], [14.9463, -26.3175], [14.502, -22.5426], [11.7995, -17.9792], [11.7446, -15.8358], [12.5329, -13.4184],
  [13.6571, -12.235], [13.8628, -11.0084], [12.9946, -9.0875], [13.3921, -8.3883], [12.2729, -6.14], [12.6729, -6.0109], [12.1679, -5.7276],
  [11.7279, -4.4509], [9.6196, -2.4084], [8.7004, -0.6684], [9.1304, -0.7142], [9.3516, 0.3579], [9.5017, 0.0504], [9.9871, 0.1725],
  [9.3096, 0.5125], [9.6637, 0.4525], [9.696, 1.0788], [9.3437, 1.1758], [9.9754, 3.0791], [9.5429, 3.8108], [9.8337, 3.8092],
  [9.7279, 4.0975], [9.2142, 3.9412], [8.8304, 4.7467], [8.555, 4.4821], [8.2358, 4.9287], [8.2783, 4.5362], [7.34, 4.4437],
  [7.0758, 4.7379], [7.0058, 4.3729], [6.7791, 4.6679], [6.8675, 4.3596], [6.7233, 4.6129], [6.705, 4.3387], [5.9833, 4.3104],
  [5.3621, 5.1633], [5.6804, 5.5408], [5.3617, 5.3904], [5.1762, 5.5683], [5.5429, 5.6192], [5.0804, 5.7183], [5.3671, 5.97],
  [5.0642, 5.7645], [4.4308, 6.3504], [3.4013, 6.3958], [3.8216, 6.6212], [1.3434, 6.1612], [0.0, 5.5929], [-1.9817, 4.7546],
  [-4.017, 5.2584], [-6.1242, 4.997], [-7.4808, 4.3604], [-8.7175, 4.8071], [-11.2308, 6.7895], [-12.5213, 7.3841], [-12.4563, 7.7666],
  [-13.2938, 8.4216], [-12.8371, 8.5608], [-13.1746, 8.5317], [-13.3039, 9.3349], [-13.7237, 9.5066], [-13.5371, 9.7966],
  [-14.4558, 10.2087], [-14.6971, 11.0766], [-15.0784, 10.9187], [-15.0046, 11.1833], [-15.235, 10.9979], [-15.5046, 11.3283],
  [-15.0338, 11.6433], [-15.5096, 11.7892], [-14.9392, 11.9887], [-15.9558, 11.7346], [-15.8538, 11.9925], [-16.3521, 12.0967],
  [-16.0142, 12.3454], [-16.7762, 12.4066], [-16.2205, 12.61], [-16.7638, 12.5716], [-16.7179, 13.46], [-16.355, 13.2496],
  [-15.5746, 13.5158], [-16.5067, 13.3613], [-16.5138, 13.8792], [-17.5338, 14.7417], [-16.5771, 15.6975], [-16.0437, 17.7475],
  [-16.5479, 19.38], [-16.1979, 20.225], [-16.9125, 21.1612], [-17.1072, 20.8375], [-16.983, 21.7981], [-15.7046, 23.9725],
  [-15.9154, 23.8033], [-14.9038, 24.6859], [-14.4654, 26.1967], [-13.6513, 26.6592], [-12.9159, 27.9563], [-11.525, 28.2988],
  [-10.2179, 29.3091], [-9.6629, 30.0716], [-9.8462, 31.3975], [-9.2871, 32.5458], [-6.8125, 34.0254], [-5.9346, 35.7908],
  [-5.4709, 35.9171], [-4.6884, 35.2063], [-2.9791, 35.442], [-2.8301, 35.1087], [-1.9575, 35.0754], [-0.9375, 35.7279], [0.0, 35.8429],
  [1.0841, 36.4938], [3.8959, 36.9254], [5.3441, 36.6412], [6.4158, 37.0863], [7.9, 36.8429], [9.745, 37.3496], [10.4316, 36.7129],
  [11.0391, 37.0896], [10.4678, 36.1358], [11.1629, 35.2158], [10.0179, 34.1875], [10.2141, 33.7971], [12.2991, 32.8379],
  [13.3883, 32.8979], [15.205, 32.3821], [15.7417, 31.3979], [17.8533, 30.9254], [18.99, 30.2746], [20.0571, 30.8509], [20.1, 32.1821],
  [21.7083, 32.9412], [23.1075, 32.6354], [23.3199, 32.1529], [24.9798, 31.9711], [25.2008, 31.5236], [26.9825, 31.4421],
  [29.0649, 30.8221], [31.0283, 31.6004], [31.9192, 31.5312]],
 [[-81.0826, 8.8084], [-79.9095, 9.3594], [-79.8751, 9.3203], [-80.0862, 9.0189], [-79.7862, 9.001], [-79.6407, 9.0505], [-79.5459, 8.9436],
  [-80.4734, 8.2308], [-79.9925, 7.5165], [-80.4396, 7.2398], [-80.89, 7.2056], [-81.0562, 7.9149], [-81.5156, 7.7091], [-81.7607, 8.2135],
  [-82.7181, 8.3267], [-82.8927, 8.0332], [-83.3804, 8.7448], [-83.2773, 8.3864], [-83.5692, 8.4406], [-83.6224, 9.0412],
  [-84.7414, 9.9689], [-85.2826, 10.2677], [-84.8624, 9.8298], [-85.1101, 9.5549], [-85.6726, 9.9049], [-85.6611, 10.776],
  [-85.9526, 10.8926], [-85.6707, 11.0603], [-87.6924, 12.9095], [-87.3035, 12.925], [-87.4041, 13.4177], [-88.7175, 13.2642],
  [-88.4679, 13.1574], [-91.3053, 13.9488], [-94.3191, 16.14], [-96.5634, 15.645], [-101.0717, 17.2665], [-101.9501, 17.9609],
  [-103.5099, 18.3127], [-105.5193, 20.0325], [-105.6953, 20.4099], [-105.2468, 20.5824], [-105.5403, 20.7701], [-105.1903, 21.4609],
  [-105.9734, 22.8799], [-107.9956, 24.6467], [-108.0442, 25.0687], [-109.3996, 25.6816], [-109.0934, 26.2864], [-110.5239, 27.2914],
  [-110.5015, 27.8692], [-111.1064, 27.9361], [-112.1749, 28.9678], [-113.1072, 31.2051], [-114.9548, 31.881], [-114.5657, 30.0113],
  [-112.8457, 28.4412], [-112.7039, 27.7535], [-111.4419, 26.5194], [-110.6817, 24.3599], [-109.8268, 24.0612], [-109.4591, 23.1983],
  [-110.0088, 22.8924], [-110.3284, 23.5603], [-112.1761, 24.8141], [-112.2389, 26.084], [-113.1671, 26.9928], [-113.6265, 26.7192],
  [-115.0268, 27.7404], [-113.9578, 27.6561], [-114.2776, 27.9032], [-114.0475, 28.4651], [-115.7, 29.7548], [-117.435, 33.2542],
  [-118.5346, 34.0338], [-120.6387, 34.5594], [-120.6414, 35.1351], [-122.4071, 37.1966], [-122.4784, 37.8096], [-122.0089, 37.4642],
  [-122.257, 38.0608], [-123.023, 37.9944], [-123.7317, 38.9216], [-124.4125, 40.4417], [-124.064, 41.4447], [-124.5684, 42.8392],
  [-124.0234, 46.2299], [-123.4313, 46.2442], [-124.0883, 46.2706], [-123.8063, 46.9717], [-124.1739, 46.9273], [-124.7313, 48.3817],
  [-122.7568, 48.1453], [-123.1586, 47.3535], [-122.6059, 47.942], [-122.4946, 47.5117], [-122.9078, 47.0465], [-122.201, 47.5067],
  [-122.4914, 48.7524], [-123.1613, 49.0175], [-123.1708, 49.6929], [-123.9696, 49.5125], [-123.5263, 49.7042], [-123.8518, 50.1627],
  [-124.2658, 49.7421], [-124.8083, 50.0107], [-124.3398, 50.5068], [-125.0809, 50.3242], [-124.8215, 50.9292], [-125.1176, 50.4283],
  [-125.5959, 50.4492], [-125.4717, 50.7213], [-125.7067, 50.4212], [-126.2992, 50.6317], [-125.5367, 50.9962], [-126.2227, 50.6959],
  [-126.5859, 50.8492], [-126.1934, 50.9479], [-127.5309, 51.0142], [-126.6566, 51.1959], [-127.7859, 51.1625], [-127.0609, 51.3517],
  [-127.7825, 51.3217], [-127.2458, 51.6816], [-127.6784, 51.4621], [-127.8991, 51.8192], [-127.0041, 52.6662], [-127.8764, 52.1963],
  [-128.0191, 52.5248], [-128.4109, 52.2908], [-127.8318, 52.7362], [-128.9886, 53.545], [-128.1213, 53.4914], [-128.7699, 53.5474],
  [-128.686, 53.9976], [-129.3152, 53.3712], [-130.1109, 53.95], [-129.5575, 54.2208], [-130.4892, 54.3641], [-130.4485, 54.6487],
  [-129.9875, 54.3041], [-130.4426, 54.6541], [-129.6608, 54.9783], [-130.1718, 55.0737], [-129.9331, 55.9488], [-130.138, 55.3289],
  [-130.8667, 54.7704], [-130.7051, 55.7558], [-131.0309, 56.102], [-132.1859, 55.5892], [-131.4911, 56.2366], [-133.5725, 57.1833],
  [-133.0575, 57.3576], [-133.6625, 57.7142], [-132.8937, 57.4959], [-133.5892, 57.7701], [-133.1358, 57.8642], [-134.056, 58.0624],
  [-133.7133, 58.5382], [-134.1469, 58.202], [-134.7827, 58.3933], [-135.3496, 59.4808], [-135.0909, 58.2408], [-135.9209, 58.3808],
  [-136.1857, 59.0681], [-136.2368, 58.7512], [-137.0481, 59.0675], [-136.0392, 58.385], [-136.6867, 58.2104], [-139.8627, 59.5358],
  [-139.5052, 59.9912], [-139.3002, 59.5612], [-138.8858, 59.8066], [-139.5149, 60.0566], [-140.315, 59.692], [-141.4113, 60.1398],
  [-144.2518, 60.0246], [-144.8783, 60.4573], [-145.9484, 60.4537], [-146.0451, 60.7958], [-146.6566, 60.6849], [-146.053, 60.8248],
  [-146.7385, 60.9082], [-146.3156, 61.1339], [-147.385, 60.8712], [-147.52, 61.1546], [-147.8767, 60.8254], [-147.7211, 61.2766],
  [-148.7142, 60.7867], [-147.9359, 60.4383], [-148.4263, 59.9489], [-149.4267, 60.1229], [-149.5259, 59.705], [-149.7233, 59.9542],
  [-149.7406, 59.6366], [-151.7426, 59.1566], [-150.93, 59.7915], [-151.8667, 59.7607], [-151.4142, 60.72], [-150.3904, 61.04],
  [-148.9692, 60.8075], [-150.079, 61.1573], [-149.1534, 61.503], [-150.5813, 61.3599], [-154.1413, 59.3735], [-154.1655, 59.0103],
  [-153.25, 58.8525], [-154.2125, 58.1344], [-155.6149, 57.7924], [-159.6, 55.5632], [-159.8169, 55.8559], [-161.2367, 55.3549],
  [-161.5796, 55.6208], [-161.9621, 55.1062], [-162.6452, 55.3034], [-162.5526, 54.9575], [-162.9739, 55.0288], [-163.3567, 54.8112],
  [-161.8039, 55.8965], [-160.2567, 55.7679], [-160.3349, 56.2943], [-157.4009, 57.4867], [-157.5355, 58.3991], [-156.8417, 59.0357],
  [-158.1888, 58.6117], [-158.5394, 59.1416], [-158.8851, 58.3917], [-160.3392, 59.0773], [-162.1362, 58.6339], [-161.521, 59.1046],
  [-162.1954, 60.1576], [-164.128, 59.8366], [-164.4923, 60.5544], [-163.4182, 60.7265], [-165.1289, 60.9166], [-164.7834, 61.1393],
  [-165.5621, 61.0876], [-166.1251, 61.4935], [-164.4267, 63.2116], [-163.3083, 63.0038], [-162.3134, 63.5437], [-161.143, 63.5035],
  [-160.7287, 63.8708], [-161.5361, 64.4174], [-160.7863, 64.7179], [-161.7473, 64.8454], [-162.7938, 64.3234], [-163.1344, 64.6529],
  [-163.1677, 64.3959], [-166.2124, 64.5787], [-166.9602, 65.1638], [-166.0535, 65.2517], [-168.1053, 65.6851], [-164.4041, 66.5824],
  [-163.761, 66.0613], [-161.0918, 66.1124], [-162.4715, 66.9524], [-161.5835, 66.4417], [-160.2117, 66.458], [-161.4884, 66.5258],
  [-161.6535, 67.0223], [-163.6979, 67.1066], [-164.1693, 67.6263], [-166.848, 68.3378], [-166.2304, 68.8771], [-163.9518, 68.9895],
  [-161.9252, 70.2946], [-160.1277, 70.6071], [-159.8437, 70.3108], [-159.6575, 70.7962], [-157.8168, 70.8668], [-156.4851, 71.3887],
  [-155.5689, 71.1616], [-155.9975, 70.7498], [-155.0751, 71.1304], [-154.1876, 70.7688], [-152.2626, 70.8404], [-152.6039, 70.5458],
  [-151.2, 70.3521], [-143.2551, 70.1179], [-135.295, 68.6399], [-135.9994, 69.2063], [-134.4984, 69.7088], [-134.2059, 69.2479],
  [-133.0211, 69.3597], [-129.668, 70.2556], [-130.881, 69.3059], [-127.4491, 70.1409], [-128.0401, 70.5773], [-125.4645, 69.3114],
  [-124.5257, 70.2005], [-124.3743, 69.3359], [-121.8802, 69.8173], [-117.0697, 68.8786], [-114.9778, 68.871], [-113.8887, 68.398],
  [-115.2486, 68.1889], [-115.0977, 67.8026], [-111.8661, 67.6682], [-110.036, 68.0109], [-107.9643, 67.2833], [-108.536, 67.0623],
  [-107.2067, 66.3473], [-107.7627, 66.9616], [-107.0665, 66.8186], [-107.0439, 67.1379], [-108.0288, 67.7753], [-105.7409, 68.5951],
  [-107.5599, 68.1666], [-108.8113, 68.2658], [-106.1889, 68.9467], [-104.9966, 68.1548], [-103.3841, 68.1586], [-102.2168, 67.6594],
  [-98.8863, 67.6882], [-98.5639, 68.1056], [-97.3113, 67.52], [-97.2065, 67.9502], [-98.0673, 67.9169], [-98.681, 68.3988],
  [-97.3037, 68.5071], [-96.3679, 68.3205], [-96.6677, 68.0063], [-95.8823, 68.301], [-96.4461, 67.4659], [-95.304, 67.1665],
  [-95.4684, 68.0628], [-93.4556, 68.5795], [-93.8222, 69.0142], [-94.6159, 68.7517], [-94.2609, 69.3218], [-93.3418, 69.3775],
  [-96.1859, 69.8623], [-95.7541, 70.7169], [-96.5999, 70.8238], [-94.4836, 71.9942], [-91.505, 70.1839], [-92.3578, 70.2445],
  [-92.8762, 69.6909], [-90.3491, 69.4593], [-91.4274, 69.3583], [-90.2484, 68.1867], [-89.3309, 69.2446], [-88.2259, 68.9189],
  [-87.4997, 67.1071], [-86.5123, 67.3341], [-85.6209, 68.7371], [-84.7113, 68.7282], [-85.5423, 69.8619], [-82.5479, 69.7011],
  [-83.2553, 69.5374], [-81.3217, 69.1991], [-82.0415, 68.8814], [-81.238, 68.6401], [-82.6437, 68.4385], [-81.1884, 67.4544],
  [-83.3498, 66.3452], [-83.9101, 66.9167], [-85.1773, 66.9102], [-83.807, 66.1436], [-86.7694, 66.5399], [-85.8578, 66.1593],
  [-87.3402, 65.3203], [-89.8218, 65.9924], [-90.9235, 65.9259], [-86.9105, 65.1408], [-88.1132, 64.1454], [-89.9803, 64.1622],
  [-90.6503, 63.4454], [-93.8077, 64.2144], [-93.9488, 63.9177], [-92.2274, 63.7458], [-92.4629, 63.5172], [-91.7175, 63.6849],
  [-90.6292, 63.1582], [-91.3864, 62.7806], [-92.4, 62.8396], [-91.8506, 62.5721], [-93.6412, 61.9531], [-93.2614, 61.7349],
  [-94.7572, 60.5156], [-94.9689, 59.0594], [-94.2568, 58.406], [-94.2016, 58.789], [-93.1768, 58.737], [-92.4282, 57.3509],
  [-92.8119, 56.9139], [-90.8032, 57.3232], [-85.2086, 55.2355], [-82.3265, 55.1514], [-82.3308, 52.9438], [-80.5686, 51.7009],
  [-80.6288, 51.2705], [-79.7578, 51.1326], [-79.3113, 51.6639], [-78.8595, 51.1726], [-79.0457, 51.763], [-78.406, 52.233],
  [-79.032, 54.1698], [-79.7614, 54.6343], [-77.6316, 55.2782], [-76.5254, 56.3757], [-77.002, 58.0211], [-78.6006, 58.6775],
  [-77.1956, 60.0437], [-77.5094, 60.8404], [-78.2265, 60.7838], [-77.4835, 61.5332], [-78.1946, 62.2547], [-77.4279, 62.5819],
  [-74.5322, 62.1069], [-73.6947, 62.4822], [-72.2707, 61.5742], [-71.5546, 61.6172], [-71.6197, 61.1536], [-69.8974, 60.7993],
  [-69.5041, 61.0676], [-69.6173, 60.0714], [-70.8747, 60.0459], [-69.7175, 59.9666], [-69.7612, 59.3216], [-69.238, 59.3229],
  [-69.5461, 58.8091], [-69.8464, 59.0599], [-70.2556, 58.7772], [-68.3485, 58.7816], [-68.6848, 58.0026], [-68.0076, 58.5849],
  [-67.6747, 58.0151], [-66.3769, 58.8548], [-65.9709, 58.2727], [-66.0864, 58.8184], [-65.3155, 59.0457], [-65.7477, 59.2705],
  [-65.0139, 59.3835], [-65.5367, 59.7457], [-64.8484, 60.3729], [-64.3675, 60.2467], [-64.6992, 60.025], [-63.3576, 59.1975],
  [-64.0509, 59.0192], [-63.1383, 59.0546], [-62.8525, 58.7008], [-63.6076, 58.2917], [-62.5576, 58.48], [-63.3559, 57.9691],
  [-62.45, 58.1729], [-62.6792, 57.9291], [-61.8925, 57.6325], [-62.5326, 57.4891], [-61.3625, 57.1], [-61.6717, 56.6187],
  [-62.5392, 56.7558], [-61.6508, 56.5392], [-62.2243, 56.4716], [-61.5808, 56.2825], [-62.0267, 56.2263], [-61.3209, 56.2241],
  [-61.7533, 55.9646], [-60.595, 55.8121], [-60.67, 55.5488], [-60.3226, 55.7775], [-60.6958, 54.9867], [-59.7783, 55.3354],
  [-59.9875, 55.1075], [-59.4159, 55.1583], [-59.935, 54.7446], [-59.1966, 55.2395], [-59.405, 54.9687], [-59.0284, 55.1546],
  [-58.9734, 54.8271], [-57.9434, 54.9304], [-58.1892, 54.7483], [-57.3342, 54.5808], [-59.5934, 54.0371], [-58.3809, 54.2275],
  [-60.7725, 53.2458], [-58.1751, 54.2379], [-57.1275, 53.94], [-57.3717, 53.4254], [-56.4834, 53.7887], [-55.8009, 53.335],
  [-56.1942, 52.8316], [-55.7659, 52.5933], [-56.5408, 52.6033], [-55.6275, 52.4449], [-56.2508, 52.45], [-55.6925, 52.0883],
  [-56.9584, 51.4238], [-58.69, 51.2612], [-59.905, 50.2629], [-66.4567, 50.2679], [-67.3767, 49.3179], [-69.9892, 48.2574],
  [-69.9062, 47.769], [-71.1934, 46.8131], [-68.1841, 48.6354], [-66.1717, 49.2071], [-64.6033, 49.1113], [-64.3167, 48.4179],
  [-65.2434, 48.0112], [-66.7955, 47.9871], [-65.8309, 47.9101], [-65.6696, 47.6012], [-64.8057, 47.8051], [-65.4189, 47.0703],
  [-64.7969, 47.0746], [-64.7093, 46.3146], [-63.2814, 45.7079], [-61.9092, 45.8829], [-60.9604, 45.3225], [-61.3112, 45.1908],
  [-63.6625, 44.7263], [-63.6484, 44.4313], [-64.3057, 44.5662], [-65.4693, 43.4504], [-66.1705, 43.8002], [-66.1117, 44.4987],
  [-64.4956, 45.3367], [-64.1034, 45.0025], [-63.3773, 45.3609], [-64.9437, 45.3302], [-64.338, 45.8807], [-66.4583, 45.0587],
  [-67.1872, 45.2265], [-66.9488, 44.8166], [-68.0589, 44.3316], [-68.2444, 44.5864], [-68.8249, 44.31], [-68.8257, 44.6664],
  [-69.2109, 43.9326], [-69.7902, 44.0899], [-69.7803, 43.7426], [-70.2598, 43.7165], [-71.0592, 42.3708], [-70.3343, 41.7087],
  [-70.0566, 42.0416], [-69.9546, 41.6466], [-71.1946, 41.4528], [-71.4051, 41.8194], [-71.4823, 41.359], [-73.5892, 41.0429],
  [-74.2909, 40.509], [-74.0971, 39.7666], [-74.8625, 38.9421], [-75.5371, 39.7317], [-75.0529, 38.41], [-75.9483, 37.1195],
  [-75.6413, 37.9758], [-76.3792, 38.8524], [-75.8762, 39.5395], [-76.6311, 39.2607], [-76.3238, 38.0374], [-77.2361, 38.6624],
  [-77.3378, 38.345], [-76.2376, 37.8908], [-76.3631, 37.6082], [-76.9296, 37.9839], [-76.2986, 37.5611], [-76.5038, 37.2455],
  [-76.8092, 37.532], [-76.4107, 37.0908], [-76.9861, 37.2423], [-75.9966, 36.9232], [-75.5384, 35.7767], [-75.9559, 36.7207],
  [-75.7964, 36.0715], [-76.21, 36.3004], [-76.7371, 35.9333], [-76.0583, 35.9929], [-76.0342, 35.6488], [-75.8494, 35.9749],
  [-75.7278, 35.6255], [-76.1541, 35.3262], [-76.9955, 35.4836], [-76.4662, 35.2625], [-76.9759, 35.0004], [-76.3309, 34.8854],
  [-78.7774, 33.7666], [-80.6176, 32.2557], [-80.8268, 32.4973], [-80.6693, 32.2157], [-81.6021, 31.21], [-80.0317, 26.8059],
  [-80.5667, 24.9546], [-80.5529, 25.1689], [-81.09, 25.1171], [-81.0263, 25.5641], [-81.7321, 25.9108], [-82.0005, 26.9767],
  [-82.3588, 26.915], [-82.6938, 27.4742], [-82.4046, 27.9458], [-82.8538, 27.87], [-82.6263, 28.8716], [-83.6822, 29.9249],
  [-84.2083, 30.1396], [-85.3566, 29.6633], [-85.5825, 30.3112], [-86.5159, 30.39], [-86.1994, 30.4938], [-87.1864, 30.3424],
  [-87.0277, 30.6066], [-88.0302, 30.2217], [-87.7764, 30.3749], [-88.0012, 30.7751], [-88.1367, 30.3107], [-90.4076, 30.2124],
  [-89.2379, 29.8875], [-89.7079, 29.5625], [-88.9986, 29.1826], [-89.4142, 28.9196], [-89.7667, 29.4796], [-90.1961, 29.5691],
  [-90.1149, 29.1407], [-90.445, 29.3499], [-90.8583, 29.0883], [-91.8288, 29.8327], [-92.3236, 29.5323], [-93.8354, 29.6931],
  [-93.8629, 29.9905], [-93.8422, 29.6789], [-94.7715, 29.3608], [-94.4686, 29.5574], [-94.9915, 29.7142], [-95.1226, 29.0684],
  [-96.2175, 28.4904], [-95.9735, 28.6581], [-96.6634, 28.722], [-96.4013, 28.4433], [-97.1928, 28.1702], [-97.4046, 27.3308],
  [-97.7821, 27.2816], [-97.4221, 27.2633], [-97.1445, 25.9601], [-97.5275, 25.0154], [-97.41, 25.4246], [-97.8096, 25.2583],
  [-97.8663, 24.525], [-97.8911, 22.5975], [-97.7732, 22.073], [-95.8574, 18.7166], [-95.1858, 18.7191], [-94.4168, 18.0608],
  [-92.3451, 18.6792], [-91.5563, 18.4397], [-90.7192, 19.3631], [-90.2822, 21.0532], [-88.1368, 21.63], [-86.7899, 21.3426],
  [-87.7436, 19.6719], [-87.4152, 19.5923], [-87.8392, 18.1882], [-88.052, 18.8568], [-88.3982, 18.3783], [-88.0931, 18.3552],
  [-88.2174, 16.9574], [-88.9365, 15.8906], [-88.1466, 15.6817], [-85.0218, 15.9873], [-83.3852, 15.2509], [-83.1974, 14.3281],
  [-83.88, 11.2926], [-82.2449, 8.9983], [-81.7892, 8.9433], [-81.8948, 9.1882], [-81.0826, 8.8084]],
 [[-73.3617, -53.0004], [-73.3626, -53.0], [-72.8909, -52.5017], [-73.7034, -52.7963], [-73.7392, -52.0434], [-72.9983, -52.0838],
  [-72.9884, -52.3638], [-72.7708, -51.9692], [-72.9184, -52.4763], [-72.4792, -51.7925], [-73.1442, -51.5217], [-72.9335, -51.1569],
  [-73.4259, -51.465], [-72.6031, -51.7722], [-73.3626, -51.555], [-73.171, -52.0364], [-73.4026, -51.6342], [-73.3898, -52.0427],
  [-73.9309, -51.6283], [-73.6609, -51.1117], [-74.2375, -50.9233], [-73.3392, -50.6942], [-73.57, -50.3871], [-74.0367, -50.8513],
  [-73.9624, -50.375], [-74.7009, -50.21], [-73.8616, -50.3013], [-74.3192, -49.64], [-73.6493, -49.7242], [-74.0383, -49.1929],
  [-74.4125, -49.3958], [-74.053, -48.745], [-74.3954, -48.6017], [-73.8414, -48.4076], [-74.6521, -48.01], [-73.3868, -48.2888],
  [-73.6788, -47.9217], [-73.2272, -48.0192], [-73.7309, -47.6255], [-74.7296, -47.7142], [-73.9154, -47.5092], [-74.5137, -47.4542],
  [-73.9238, -47.2117], [-74.2117, -46.7547], [-74.85, -46.8087], [-74.8692, -46.4371], [-75.5221, -46.7167], [-75.2984, -46.9471],
  [-75.6221, -46.7292], [-74.8546, -46.3725], [-74.5638, -46.0476], [-75.0912, -45.885], [-74.7275, -45.8138], [-74.4309, -46.0479],
  [-74.0843, -45.8146], [-74.3337, -46.1325], [-73.9946, -46.1525], [-74.4596, -46.1992], [-73.8692, -46.1446], [-73.6375, -46.5396],
  [-73.1413, -45.6534], [-73.5725, -45.7846], [-73.4929, -45.4584], [-72.7721, -45.4075], [-73.5163, -45.2], [-72.5496, -44.5275],
  [-73.2871, -44.165], [-72.7404, -43.845], [-73.1029, -43.4592], [-72.8254, -42.5159], [-72.4963, -42.6192], [-72.8446, -42.3025],
  [-72.4892, -42.5129], [-72.3529, -42.1758], [-72.8846, -41.9141], [-72.2829, -41.3933], [-72.5785, -41.5875], [-72.9509, -41.4746],
  [-73.765, -41.7487], [-73.4863, -41.5692], [-73.9596, -41.0067], [-73.1937, -39.4425], [-73.6854, -37.3325], [-73.1554, -37.115],
  [-71.6188, -33.6258], [-71.7196, -30.5992], [-71.2771, -29.91], [-71.5204, -28.9133], [-70.4496, -25.3675], [-70.6288, -23.505],
  [-70.0571, -21.44], [-70.1229, -20.0092], [-70.3072, -18.4392], [-71.4767, -17.2912], [-75.1433, -15.4063], [-75.9163, -14.6542],
  [-76.3988, -13.9075], [-76.2021, -13.3958], [-77.6579, -11.29], [-78.9846, -8.2125], [-79.9813, -6.7509], [-81.1521, -5.9792],
  [-80.8521, -5.6567], [-81.2554, -4.2725], [-80.0047, -3.375], [-79.7146, -2.5967], [-79.8775, -2.2171], [-80.2475, -2.7388],
  [-80.8988, -2.3308], [-80.9112, -1.0542], [-80.2754, -0.67], [-80.5004, -0.3925], [-79.9484, 0.1703], [-80.0908, 0.7755],
  [-78.9285, 1.0638], [-79.0334, 1.6357], [-78.5304, 1.7665], [-78.5625, 2.4415], [-77.7677, 2.6634], [-77.1319, 3.6372],
  [-77.0235, 3.9195], [-77.5308, 4.2057], [-77.2941, 6.5582], [-78.4016, 7.9238], [-78.0589, 8.4242], [-78.421, 8.3474], [-79.0878, 9.0408],
  [-79.5389, 8.9506], [-79.9051, 9.3663], [-79.6243, 9.6165], [-78.971, 9.5647], [-77.8729, 9.1166], [-76.771, 7.914], [-76.9371, 8.5575],
  [-75.6196, 9.4516], [-75.7025, 10.1477], [-75.2588, 10.8076], [-74.8433, 11.1137], [-74.2767, 11.0021], [-74.1425, 11.3412],
  [-73.2717, 11.2921], [-71.6659, 12.4645], [-71.1096, 12.0515], [-71.9647, 11.5574], [-71.5788, 10.705], [-72.1312, 9.8134],
  [-71.7094, 9.0521], [-71.0723, 9.3115], [-71.025, 9.7166], [-71.5904, 10.7967], [-70.1375, 11.5712], [-69.7883, 11.4313],
  [-69.8034, 11.6971], [-70.2354, 11.6358], [-70.0199, 12.2024], [-69.6233, 11.4745], [-68.3959, 11.1871], [-68.1292, 10.4888],
  [-66.2342, 10.6537], [-65.0551, 10.0637], [-63.6429, 10.4925], [-64.2609, 10.667], [-61.8666, 10.7462], [-62.8887, 10.375],
  [-62.3389, 9.8182], [-61.6367, 9.9029], [-60.855, 9.4379], [-60.8759, 8.5904], [-60.2267, 8.6428], [-59.045, 7.9587], [-58.4799, 7.3347],
  [-58.69, 6.3821], [-58.3417, 6.8921], [-57.1436, 5.8366], [-54.1693, 5.8837], [-52.2909, 4.9479], [-51.6574, 4.0595], [-51.5488, 4.4272],
  [-51.3013, 4.2505], [-50.7667, 2.1021], [-49.9327, 1.7117], [-50.1384, 1.2098], [-49.8985, 1.1981], [-51.3113, -0.0792],
  [-51.7009, -0.9924], [-51.994, -1.3949], [-52.712, -1.6035], [-52.201, -1.6903], [-50.858, -0.9132], [-50.523, -1.9273],
  [-49.276, -1.7334], [-49.689, -2.6763], [-49.4859, -2.3829], [-48.6525, -1.3946], [-48.4466, -1.7071], [-48.0671, -1.55],
  [-48.4946, -1.4626], [-48.0534, -0.6596], [-47.3671, -0.8292], [-47.2909, -0.5963], [-46.3475, -1.1138], [-46.1967, -0.8888],
  [-46.1475, -1.3046], [-45.9875, -1.0454], [-45.695, -1.4313], [-45.4154, -1.2917], [-45.5217, -1.5121], [-45.3267, -1.3179],
  [-45.2392, -1.8513], [-44.8329, -1.4142], [-44.4829, -2.0509], [-44.9879, -2.4484], [-44.3605, -2.3367], [-44.7496, -2.6233],
  [-44.8183, -3.3579], [-43.4217, -2.3321], [-42.2609, -2.8596], [-40.0225, -2.8371], [-38.475, -3.7029], [-37.1503, -4.9742],
  [-35.6167, -5.1121], [-35.2604, -5.4817], [-34.7946, -7.1542], [-34.9396, -8.3558], [-38.0687, -12.6667], [-38.4717, -13.0146],
  [-38.6909, -12.5846], [-38.8457, -12.8348], [-39.1371, -17.6908], [-39.6987, -18.3733], [-39.8087, -19.605], [-41.0713, -21.5017],
  [-40.9846, -22.0025], [-41.9617, -22.5312], [-42.0142, -22.9971], [-43.0509, -22.9821], [-43.0833, -22.6788], [-43.5575, -23.0779],
  [-44.4125, -22.9438], [-45.4134, -23.8246], [-46.3802, -23.8987], [-47.5725, -24.6788], [-48.2125, -25.4688], [-48.7409, -25.3729],
  [-48.3487, -25.5717], [-48.8121, -26.3092], [-48.4654, -27.1459], [-48.7621, -28.5384], [-49.6996, -29.3017], [-50.7946, -31.1425],
  [-52.0859, -32.1646], [-50.5688, -30.4642], [-51.3037, -30.005], [-51.0954, -30.3717], [-52.2742, -31.7779], [-52.5954, -33.0592],
  [-54.1509, -34.6705], [-56.1617, -34.9354], [-57.8558, -34.4771], [-58.4296, -33.9193], [-58.4425, -33.0704], [-58.5696, -34.4175],
  [-57.1979, -35.3033], [-57.3796, -35.97], [-56.7304, -36.3342], [-56.6688, -36.8875], [-57.5475, -38.1037], [-58.5067, -38.543],
  [-61.1108, -39.0029], [-62.3409, -38.7704], [-62.0062, -39.3825], [-62.3137, -39.2984], [-62.3825, -40.9113], [-63.7901, -41.1671],
  [-65.0079, -40.7184], [-65.0471, -42.0558], [-64.4709, -42.4413], [-63.7651, -42.0754], [-63.5846, -42.6167], [-64.0925, -42.8779],
  [-64.395, -42.5188], [-64.9792, -42.6529], [-64.3063, -42.9776], [-65.3321, -43.6634], [-65.5212, -44.9317], [-66.9492, -45.2671],
  [-67.6271, -46.0525], [-66.7942, -46.9987], [-65.7396, -47.2025], [-65.7655, -47.9175], [-67.5725, -49.0433], [-67.9176, -50.0092],
  [-68.3434, -50.1263], [-68.7275, -49.7567], [-68.3608, -50.1667], [-69.4293, -51.0917], [-68.9542, -51.57], [-69.5859, -51.6075],
  [-68.9675, -51.6259], [-68.3525, -52.3406], [-69.2167, -52.2028], [-70.8829, -52.739], [-71.2834, -53.9004], [-72.4842, -53.395],
  [-71.8551, -53.2313], [-71.9667, -53.5788], [-71.3493, -53.1275], [-71.5451, -52.5689], [-72.9268, -52.5563], [-72.9776, -53.0824],
  [-73.3617, -53.0004]],
 [[146.23, -38.6971], [146.2762, -39.0], [145.3716, -38.5396], [145.4883, -38.2354], [144.6584, -38.3113], [145.1163, -38.1483],
  [144.9283, -37.8429], [143.5566, -38.8587], [140.5817, -38.0329], [139.7404, -37.1833], [139.6121, -36.1567], [139.7225, -36.2888],
  [139.5176, -35.9613], [138.8908, -35.5338], [138.0979, -35.6267], [138.5679, -34.8267], [138.0942, -34.1354], [137.7626, -35.1179],
  [136.8517, -35.2854], [137.0158, -34.8954], [137.4529, -34.9083], [137.4504, -34.14], [137.9779, -33.5534], [137.7541, -32.4588],
  [137.781, -33.0], [135.9337, -34.5342], [135.9575, -35.0079], [135.1129, -34.59], [135.5188, -34.6142], [134.7079, -33.1817],
  [134.0596, -32.9141], [134.1817, -32.4854], [131.1525, -31.4646], [125.9575, -32.2888], [124.2341, -33.0196], [123.5267, -33.938],
  [119.9034, -33.9346], [117.9533, -35.1288], [116.6241, -35.0579], [115.1283, -34.3729], [115.0004, -33.5284], [115.6979, -33.3009],
  [115.8829, -31.9617], [115.0521, -30.5075], [114.8671, -29.1142], [113.1529, -26.1492], [113.8379, -26.5917], [113.5091, -25.5054],
  [113.7221, -26.2], [113.8708, -25.9421], [114.2312, -26.3159], [113.3962, -24.4067], [113.9904, -21.8725], [114.33, -22.5221],
  [114.6458, -21.8379], [116.8042, -20.5238], [117.3775, -20.7771], [119.0925, -19.9579], [121.1133, -19.5396], [122.3696, -18.1175],
  [122.1729, -17.2633], [122.9225, -16.3879], [123.5667, -17.6271], [123.5695, -17.0376], [123.9308, -17.2679], [123.9713, -16.8258],
  [123.5046, -16.6625], [123.5575, -16.1738], [124.4, -16.5654], [124.9762, -16.3817], [124.3913, -16.3392], [124.7504, -15.8109],
  [124.495, -16.003], [124.3604, -15.67], [124.675, -15.2546], [125.2496, -15.5817], [124.8296, -15.1575], [125.5067, -15.1721],
  [125.1379, -14.7458], [125.6142, -14.2305], [125.9184, -14.6771], [126.0092, -13.9204], [126.23, -14.2504], [126.5358, -13.9321],
  [126.6125, -14.2504], [126.875, -13.7471], [127.4283, -13.9429], [128.2271, -14.7142], [128.01, -15.5137], [128.1283, -15.1862],
  [128.3721, -15.4967], [128.1996, -15.0692], [128.5591, -14.7663], [129.0925, -14.9046], [129.1259, -15.2813], [129.29, -14.8621],
  [129.6058, -15.2063], [129.9988, -14.7059], [129.3654, -14.34], [129.8166, -13.4996], [130.318, -13.3733], [130.3512, -12.67],
  [130.7321, -12.7283], [130.5758, -12.4071], [130.965, -12.6713], [130.8121, -12.4083], [131.2875, -12.0454], [131.4625, -12.2854],
  [132.7629, -12.1546], [132.723, -11.6233], [131.7679, -11.3258], [131.9733, -11.1271], [133.5083, -11.8796], [133.9158, -11.7421],
  [134.1509, -12.1746], [134.7484, -11.9512], [135.2154, -12.3009], [135.9125, -11.9529], [135.6529, -12.2067], [136.045, -12.0646],
  [136.0167, -12.4988], [136.5641, -11.8796], [136.978, -12.3483], [136.4621, -12.7775], [136.4617, -13.2538], [135.9096, -13.285],
  [136.0721, -13.6675], [135.4112, -14.935], [136.6767, -15.9263], [139.0366, -16.9129], [139.5984, -17.5388], [140.6092, -17.6138],
  [141.667, -15.0409], [141.4612, -13.8525], [141.9768, -12.5948], [141.5862, -12.5584], [142.0796, -11.9859], [142.1304, -10.9617],
  [142.5342, -10.6887], [143.5454, -12.8442], [143.7729, -14.3992], [144.5151, -14.1663], [145.3479, -14.9492], [145.4021, -16.435],
  [145.9579, -16.8975], [146.2604, -18.8634], [148.7729, -20.2367], [149.6305, -22.585], [149.8125, -22.3821], [150.0604, -22.6584],
  [150.0417, -22.1254], [150.6359, -22.6604], [150.6683, -22.3471], [150.7917, -23.5096], [151.7696, -24.0209], [153.1929, -25.9317],
  [153.0246, -27.2942], [153.6396, -28.6384], [153.0654, -31.0584], [152.3596, -32.1867], [152.5429, -32.4442], [151.2012, -33.5108],
  [150.603, -34.8658], [150.8429, -35.0667], [150.153, -35.701], [149.9796, -37.5051], [147.6575, -37.8504], [147.4541, -38.0796],
  [147.9721, -37.8934], [146.23, -38.6971]],
 [[-44.6993, 59.995], [-44.4683, 60.5387], [-45.1759, 60.1292], [-44.6284, 60.7104], [-45.2068, 60.3937], [-45.2151, 60.7637],
  [-45.9891, 60.565], [-45.3968, 61.0004], [-46.2359, 60.7375], [-45.2117, 61.1954], [-47.6584, 60.8029], [-47.9201, 61.3179],
  [-49.0892, 61.385], [-48.2659, 61.5275], [-49.2726, 61.5325], [-48.6075, 61.6316], [-49.1909, 61.6967], [-48.8384, 61.9837],
  [-49.4975, 61.8041], [-48.8376, 62.0708], [-49.7149, 61.962], [-49.2792, 62.265], [-50.3176, 62.4866], [-49.6959, 63.05],
  [-50.3633, 62.7721], [-50.0542, 63.2167], [-51.1093, 63.3317], [-50.1641, 63.3775], [-51.2426, 63.4308], [-50.5059, 63.6533],
  [-51.5526, 63.6683], [-50.9076, 63.9217], [-51.6051, 64.0321], [-50.0959, 64.1267], [-51.7575, 64.1691], [-50.2058, 64.4358],
  [-50.8899, 64.6137], [-49.6476, 64.3383], [-50.9967, 65.2179], [-50.6525, 64.7349], [-52.0017, 64.1879], [-52.1675, 64.6833],
  [-51.2658, 64.9933], [-52.21, 64.7971], [-51.9708, 65.3124], [-52.5759, 65.3092], [-50.4643, 65.6725], [-51.235, 65.8146],
  [-52.49, 65.3779], [-53.2042, 65.7525], [-51.7576, 66.0733], [-53.4809, 66.0291], [-50.3475, 66.84], [-50.9617, 66.9688],
  [-53.6826, 66.1083], [-53.0778, 66.3331], [-53.5951, 66.5196], [-52.4175, 66.5233], [-53.4934, 66.6454], [-52.2325, 66.8408],
  [-53.8826, 67.1558], [-51.4526, 67.3384], [-53.8076, 67.4075], [-50.2559, 67.8508], [-52.268, 67.7979], [-51.6101, 67.9771],
  [-53.6884, 67.4796], [-53.2067, 68.0321], [-52.0676, 67.9542], [-53.3334, 68.1887], [-50.3309, 67.9208], [-51.4409, 68.1866],
  [-50.8017, 68.5038], [-53.4725, 68.31], [-52.25, 68.6479], [-50.6601, 68.5062], [-51.1042, 69.1358], [-50.3717, 68.8954],
  [-49.9725, 69.18], [-51.1542, 69.2058], [-50.245, 70.0521], [-52.3551, 70.0496], [-54.6164, 70.6525], [-52.9276, 70.7688],
  [-50.5613, 70.3308], [-51.9688, 71.0259], [-50.9363, 71.0192], [-52.2263, 71.1133], [-51.5163, 71.3042], [-52.5387, 71.1666],
  [-51.3687, 71.4842], [-53.0063, 71.4225], [-51.7037, 71.7425], [-53.2776, 71.7146], [-53.5724, 72.3596], [-53.9075, 71.4446],
  [-55.8688, 71.6867], [-54.3938, 72.2425], [-55.25, 71.9271], [-54.6937, 72.3691], [-55.6262, 72.4508], [-54.2713, 72.4883],
  [-55.6963, 73.0583], [-56.0763, 74.28], [-57.3038, 74.1024], [-56.1512, 74.5566], [-60.9102, 76.1637], [-63.3966, 76.3804],
  [-65.4201, 76.0196], [-67.11, 76.2504], [-66.4733, 75.9104], [-68.5267, 76.0904], [-69.6118, 76.4183], [-67.9184, 76.685],
  [-71.3617, 77.0642], [-66.3567, 77.1104], [-69.115, 77.2758], [-66.1601, 77.2529], [-66.1751, 77.6041], [-69.32, 77.4704],
  [-73.0616, 78.1767], [-69.0435, 79.0396], [-66.0086, 79.1175], [-63.8476, 80.1475], [-67.01, 80.0537], [-67.4678, 80.3483],
  [-64.0803, 81.1171], [-63.1077, 80.7392], [-63.4702, 81.2038], [-61.7875, 80.9966], [-61.8326, 81.7858], [-59.7552, 81.8562],
  [-57.2175, 81.3341], [-60.1203, 82.0237], [-55.1945, 82.3304], [-54.8053, 81.6046], [-53.7455, 81.9832], [-50.3499, 81.6035],
  [-51.7317, 82.5147], [-45.3079, 81.756], [-45.6189, 82.259], [-43.2675, 82.2333], [-46.5453, 82.7789], [-40.8796, 82.3633],
  [-41.1769, 82.7222], [-45.9078, 82.9285], [-44.2704, 83.2751], [-39.736, 82.742], [-38.8607, 83.5304], [-26.7998, 83.3422],
  [-34.8388, 83.1657], [-36.8808, 82.7118], [-26.2497, 83.1685], [-26.9403, 82.7732], [-21.9247, 82.6292], [-25.7737, 82.1681],
  [-32.8009, 82.1862], [-34.2726, 81.8111], [-26.1567, 81.9861], [-27.8114, 81.3994], [-22.9361, 82.0709], [-24.7808, 80.5333],
  [-21.0716, 81.5901], [-18.3983, 81.4346], [-16.7746, 81.9295], [-12.8728, 81.7452], [-14.7851, 80.8946], [-21.2776, 80.57],
  [-16.1075, 80.5174], [-20.6277, 80.1125], [-20.4169, 79.762], [-17.6316, 79.9975], [-21.8417, 78.1408], [-21.9868, 77.6796],
  [-20.8868, 78.0187], [-19.1718, 77.735], [-21.1019, 77.5408], [-18.405, 77.3342], [-18.2818, 76.82], [-22.7351, 76.7041],
  [-21.8366, 76.2138], [-19.785, 76.1633], [-21.8717, 75.9675], [-19.3517, 75.7217], [-19.6034, 75.1288], [-22.5017, 75.5475],
  [-20.5652, 75.1892], [-22.4549, 75.1742], [-20.6317, 75.0525], [-21.0602, 74.6496], [-18.9688, 74.4842], [-19.6451, 74.2362],
  [-22.1089, 74.6042], [-22.4864, 74.0558], [-21.8276, 73.6413], [-21.7252, 74.0629], [-20.2926, 73.8587], [-20.4113, 73.4775],
  [-22.4026, 73.2479], [-24.0939, 73.6759], [-22.2276, 73.6129], [-24.1426, 73.8079], [-24.4776, 73.5362], [-25.7626, 73.9488],
  [-24.6913, 73.5042], [-26.0, 73.2362], [-27.3276, 73.4962], [-26.3539, 73.235], [-27.7163, 73.1242], [-25.0964, 73.0658],
  [-27.4662, 72.8375], [-26.6425, 72.5238], [-25.265, 72.8012], [-24.5889, 72.4842], [-26.3338, 72.3883], [-25.2563, 72.3958],
  [-25.5126, 72.1154], [-24.62, 72.4246], [-22.5387, 71.9242], [-23.155, 71.6288], [-21.9213, 71.7383], [-22.5787, 71.465],
  [-21.7588, 71.4933], [-22.3363, 71.0525], [-21.4987, 70.52], [-22.4239, 70.465], [-22.4901, 70.8671], [-23.3776, 70.4571],
  [-24.6439, 71.3475], [-28.3701, 71.9987], [-27.3464, 71.7059], [-28.46, 71.5454], [-25.4337, 71.275], [-26.8026, 70.9379],
  [-27.9375, 71.1554], [-29.2513, 70.4591], [-26.5226, 70.4713], [-28.5964, 70.1058], [-27.4017, 69.9512], [-25.2802, 70.4196],
  [-22.1038, 70.0992], [-26.3867, 68.6679], [-29.4218, 68.2171], [-29.9918, 68.4346], [-30.0751, 68.1371], [-31.6267, 68.0804],
  [-32.5917, 68.6446], [-32.0508, 67.9358], [-33.2343, 67.6825], [-34.7901, 66.3196], [-35.8866, 66.4429], [-35.5859, 66.1167],
  [-36.9966, 65.8546], [-37.2551, 66.0987], [-37.2001, 65.7429], [-37.8059, 65.8658], [-37.1558, 66.3083], [-37.8717, 66.4179],
  [-38.1934, 65.6146], [-39.9301, 65.5646], [-40.2184, 64.9837], [-41.1334, 65.1354], [-40.3458, 64.3433], [-41.5693, 64.2508],
  [-40.5858, 64.1066], [-40.5092, 63.685], [-41.5117, 63.8571], [-40.7458, 63.5783], [-41.6009, 63.4841], [-41.1401, 63.2937],
  [-41.9292, 63.4642], [-41.3108, 63.0541], [-42.1134, 63.2479], [-41.7817, 62.8196], [-43.1558, 62.7458], [-42.1475, 62.3783],
  [-42.9976, 62.5133], [-42.1108, 61.9992], [-42.8741, 61.7608], [-42.2326, 61.6975], [-43.0809, 61.5899], [-42.4042, 61.4883],
  [-43.2509, 61.3267], [-42.6384, 61.0854], [-43.6292, 61.1241], [-42.7042, 61.0558], [-43.5292, 60.8117], [-42.8716, 60.5462],
  [-44.2193, 60.5525], [-43.1391, 60.0717], [-44.1134, 60.3529], [-44.6993, 59.995]],
 [[150.0, -10.0879], [150.005, -10.0879], [150.8771, -10.23], [150.2017, -10.7046], [149.7433, -10.3487], [147.725, -10.1071],
  [146.09, -8.0954], [144.8616, -7.7813], [144.7525, -7.4071], [144.5366, -7.6954], [144.3729, -7.4366], [144.5058, -7.8187],
  [143.6375, -7.4313], [143.9288, -7.9934], [143.3913, -7.9292], [143.6942, -8.2329], [142.2108, -8.1771], [143.4029, -8.7583],
  [142.6392, -9.3354], [142.2191, -9.0779], [141.1258, -9.234], [140.0367, -8.0071], [138.9551, -8.2596], [139.1396, -7.5809],
  [138.6712, -7.2142], [139.2438, -7.1409], [138.5437, -6.9542], [138.9213, -6.8309], [138.3954, -6.3358], [138.3671, -5.6692],
  [138.0538, -5.7334], [138.1166, -5.3771], [135.1983, -4.4588], [134.618, -4.1158], [134.9325, -3.9188], [133.9691, -3.8671],
  [133.6179, -3.475], [133.8075, -2.9104], [133.4479, -3.8667], [132.8958, -4.0871], [132.837, -3.3025], [131.9471, -2.7742],
  [132.7241, -2.8063], [133.2192, -2.4071], [133.6667, -2.7304], [133.9396, -2.0909], [132.3083, -2.2854], [131.9634, -1.4779],
  [130.9305, -1.435], [131.2221, -0.8284], [132.4156, -0.3386], [133.9833, -0.7229], [134.1746, -2.3533], [134.4583, -2.8655],
  [134.6304, -2.48], [135.1075, -3.3746], [135.7658, -3.1205], [136.3913, -2.2179], [137.1708, -2.1138], [137.1133, -1.7955],
  [137.8909, -1.4646], [144.5233, -3.8112], [145.8054, -4.8509], [145.7592, -5.4796], [147.6254, -6.1117], [147.8567, -6.6521],
  [146.9504, -6.7342], [147.1763, -7.4625], [148.1304, -8.0642], [148.6033, -9.0846], [149.2509, -8.9988], [149.2375, -9.5021],
  [150.0126, -9.6313], [149.7104, -9.8201], [150.0, -10.0879]],
 [[115.0, -4.0146], [114.9984, -4.0163], [114.5946, -4.1675], [114.5308, -3.3505], [113.635, -3.4646], [113.6142, -3.1496],
  [113.0222, -2.9382], [111.9025, -3.5696], [111.7016, -2.8029], [110.2592, -2.9662], [109.9554, -1.1084], [109.4308, -1.2871],
  [109.6404, -0.9867], [109.2446, -0.6608], [109.5912, -0.74], [109.1104, -0.5151], [109.2821, 0.0083], [108.9104, 0.3216],
  [108.9071, 1.1608], [109.6417, 2.0839], [109.8923, 1.6979], [111.1708, 1.3646], [111.4333, 2.7071], [113.035, 3.1679], [113.9717, 4.6012],
  [115.3759, 4.9061], [115.6071, 5.23], [115.3704, 5.4108], [115.8637, 5.5808], [116.7459, 7.0345], [116.8133, 6.5496], [117.1507, 7.0079],
  [117.7379, 6.4241], [117.5581, 5.9022], [118.0075, 6.0646], [117.9408, 5.6696], [118.35, 5.8296], [119.2621, 5.36], [118.1237, 4.8833],
  [118.5508, 4.3521], [117.3596, 4.1624], [117.7683, 3.6479], [117.0137, 3.5967], [118.0946, 2.3133], [117.7346, 2.1892],
  [117.8437, 1.8625], [118.9996, 0.9842], [118.3641, 0.8004], [117.8887, 1.12], [118.0362, 0.7867], [117.4621, 0.0908], [117.6146, -0.7808],
  [116.8975, -1.2713], [116.7021, -1.085], [116.1654, -1.8034], [116.6012, -2.1958], [116.3129, -2.9608], [116.0996, -2.8625],
  [116.2779, -3.13], [115.0, -4.0146]],
 [[44.9991, -25.4871], [44.9983, -25.4871], [44.0271, -25.0], [43.2304, -22.325], [43.4979, -21.3125], [44.4838, -19.9875],
  [43.9296, -17.4967], [44.4375, -16.1904], [46.175, -15.7038], [46.5025, -15.9979], [46.4725, -15.5104], [46.9583, -15.2013],
  [46.8912, -15.6084], [47.2096, -15.4717], [47.0471, -15.1784], [47.4633, -14.6713], [47.4867, -15.0888], [47.7621, -14.2492],
  [48.042, -14.265], [47.8929, -13.5942], [48.2641, -13.8171], [48.8121, -13.3792], [48.7204, -12.4425], [49.2725, -11.9546],
  [49.1913, -12.305], [49.9304, -13.0442], [50.4654, -15.4475], [50.1616, -15.9963], [49.8975, -15.4354], [49.6212, -15.5392],
  [49.8371, -16.8392], [49.4437, -17.2117], [49.4304, -18.1642], [47.1012, -24.9933], [44.9991, -25.4871]],
 [[-65.9991, 61.9549], [-65.9984, 61.9537], [-71.6411, 63.1333], [-72.1209, 63.4421], [-71.2, 63.5942], [-72.6395, 63.8451],
  [-73.2836, 64.6604], [-73.5769, 64.3138], [-74.6069, 64.9024], [-74.4606, 64.3838], [-78.0301, 64.4274], [-77.3764, 65.4715],
  [-75.803, 65.2339], [-73.4716, 65.4477], [-74.4578, 66.1652], [-73.0716, 66.7207], [-72.0517, 66.6702], [-73.0047, 66.7787],
  [-72.1527, 67.2704], [-73.6607, 68.6648], [-74.2308, 68.5193], [-75.4725, 69.027], [-76.6466, 68.6949], [-75.5882, 69.2357],
  [-76.1957, 69.6725], [-77.1888, 69.642], [-76.9933, 69.9887], [-77.6272, 69.7486], [-78.9861, 70.708], [-79.6008, 70.3744],
  [-78.7946, 69.888], [-81.7284, 70.1458], [-80.9329, 69.7212], [-83.0294, 70.3146], [-81.7612, 69.8743], [-85.8193, 70.0093],
  [-86.3818, 70.5282], [-87.8581, 70.24], [-88.7926, 70.4981], [-89.5308, 71.091], [-85.0, 71.205], [-84.7724, 70.933], [-84.4808, 71.6374],
  [-86.0387, 72.0211], [-85.2258, 72.2666], [-84.1489, 72.0066], [-85.7016, 72.9047], [-83.9516, 72.7471], [-85.543, 73.0331],
  [-83.6984, 73.0072], [-85.0, 73.3481], [-81.5481, 73.724], [-80.2257, 72.7362], [-80.9623, 71.8839], [-79.7904, 72.5047],
  [-77.7306, 71.7464], [-78.8812, 72.2341], [-77.0038, 72.1361], [-78.5502, 72.4384], [-77.5751, 72.7619], [-75.1502, 72.4907],
  [-76.4011, 71.8592], [-74.2087, 72.0611], [-75.3659, 71.6844], [-74.6165, 71.6598], [-75.0476, 71.1788], [-73.5992, 71.7792],
  [-73.8554, 71.0494], [-72.5499, 71.6607], [-71.1548, 71.2731], [-72.5658, 70.6156], [-70.6043, 71.059], [-71.9725, 70.42],
  [-71.1655, 70.5464], [-71.543, 70.0265], [-69.8842, 70.8841], [-70.6153, 70.4575], [-69.4476, 70.7949], [-68.2856, 70.5255],
  [-70.178, 70.0323], [-68.6764, 70.2073], [-70.0232, 69.6212], [-68.1605, 70.3184], [-67.259, 69.9724], [-67.2056, 69.7179],
  [-70.0622, 69.5424], [-67.2309, 69.4675], [-66.7011, 69.1707], [-69.0401, 69.358], [-68.0838, 69.223], [-69.0121, 68.9774],
  [-68.2, 69.1517], [-67.7757, 68.7852], [-69.4081, 68.8151], [-66.6739, 68.4469], [-67.8828, 68.2692], [-67.019, 68.331],
  [-66.7479, 67.9324], [-66.3145, 68.1246], [-66.3742, 67.7681], [-65.9404, 68.0327], [-66.0329, 67.5966], [-65.8035, 67.9731],
  [-65.3428, 67.5888], [-64.7217, 67.9779], [-65.2302, 67.6421], [-64.53, 67.8104], [-63.9509, 67.3425], [-64.795, 67.3687],
  [-63.9575, 67.2666], [-64.7201, 67.0079], [-63.5001, 67.2354], [-63.8133, 66.9004], [-63.1183, 67.3321], [-63.81, 66.8046],
  [-62.8418, 66.9679], [-62.7335, 66.6563], [-62.0267, 67.0529], [-61.2925, 66.6674], [-62.1975, 66.615], [-61.5434, 66.3187],
  [-62.8959, 66.3341], [-61.9492, 66.0], [-62.9492, 66.1467], [-62.6767, 65.5746], [-63.5292, 65.93], [-63.5368, 64.8762],
  [-65.5015, 65.7407], [-64.3584, 66.3554], [-65.8746, 65.9441], [-65.481, 66.3857], [-65.9853, 66.1072], [-66.7375, 66.5974],
  [-67.9398, 66.6336], [-67.1327, 66.0182], [-68.3675, 66.0727], [-68.236, 65.4276], [-67.3161, 65.6653], [-66.7307, 64.7207],
  [-66.6767, 65.0418], [-66.3144, 64.6101], [-65.6868, 64.8484], [-65.0402, 64.4307], [-65.6714, 64.3031], [-64.6126, 63.9767],
  [-64.5167, 63.2421], [-65.3159, 63.8052], [-64.8975, 62.6317], [-67.9152, 63.7695], [-67.6885, 63.3689], [-68.9834, 63.7559],
  [-65.9991, 61.9549]],
 [[103.9991, -5.3254], [103.9984, -5.3246], [101.5813, -3.2042], [100.3246, -0.8542], [99.1362, 0.2533], [98.7971, 1.7308],
  [97.7509, 2.2762], [96.8671, 3.695], [95.5371, 4.6617], [95.2275, 5.5829], [97.5033, 5.2554], [98.2887, 4.4308], [98.1558, 4.0846],
  [99.7533, 3.1795], [100.9333, 1.8096], [101.0475, 2.3029], [102.4133, 0.8054], [102.91, 0.7171], [103.1062, 0.4559], [102.6825, 0.2254],
  [103.3275, 0.5496], [103.7288, 0.2808], [103.2762, -0.7059], [104.3675, -1.0179], [104.4796, -1.99], [104.8788, -2.0842],
  [104.7129, -2.6], [104.8475, -2.2846], [105.6191, -2.3921], [106.0871, -3.2258], [105.7233, -5.9046], [105.2842, -5.4463],
  [105.1608, -5.8096], [104.5299, -5.5221], [104.5817, -5.9454], [103.9991, -5.3254]],
 [[136.0, 33.7104], [135.9979, 33.7083], [135.7533, 33.4362], [135.0579, 33.8817], [135.3367, 34.7254], [134.1858, 34.7412],
  [132.5567, 34.0696], [132.4008, 34.3763], [132.1391, 33.8313], [130.905, 33.9129], [130.8663, 34.2917], [133.0925, 35.6071],
  [136.0708, 35.6621], [136.7567, 37.3638], [137.3454, 37.5158], [136.8604, 37.0817], [137.3317, 36.7629], [139.425, 38.1487],
  [140.1037, 39.6792], [140.07, 39.9988], [139.7037, 39.9283], [139.8604, 40.615], [140.3391, 41.2604], [141.1308, 40.8729],
  [140.8345, 41.4191], [141.4633, 41.4312], [142.0704, 39.5475], [141.5262, 38.2749], [140.9187, 38.2], [140.9821, 36.995],
  [140.5595, 36.2791], [140.8646, 35.6924], [139.89, 34.8988], [140.1271, 35.5708], [139.7816, 35.6729], [139.6805, 35.1408],
  [139.1708, 35.2546], [138.8449, 34.6004], [138.7441, 35.1329], [138.2292, 34.5954], [137.0158, 34.5787], [137.3746, 34.78],
  [136.9824, 34.9629], [136.8899, 34.7196], [136.7125, 35.0546], [136.5029, 34.6525], [136.8996, 34.2767], [136.0, 33.7104]],
 [[-113.5544, 68.9991], [-113.5555, 69.0], [-116.5006, 69.4075], [-117.3647, 70.0393], [-111.4435, 70.3424], [-113.821, 70.7158],
  [-117.684, 70.6316], [-118.3298, 71.0442], [-115.0375, 71.5341], [-118.234, 71.3945], [-117.7437, 71.6695], [-119.1224, 71.7722],
  [-118.4977, 72.5042], [-114.4801, 73.3765], [-114.5248, 72.5761], [-112.7882, 72.9944], [-111.1861, 72.7267], [-112.0403, 72.2758],
  [-111.0087, 72.2816], [-110.6835, 72.5749], [-109.7455, 72.4286], [-110.7272, 72.9622], [-109.9009, 72.9753], [-107.7554, 71.6032],
  [-108.2461, 73.1728], [-106.7621, 73.2995], [-105.2812, 72.8263], [-104.4865, 71.0294], [-100.9816, 70.1882], [-100.9529, 69.6672],
  [-101.4347, 69.9353], [-102.5758, 69.542], [-103.5219, 69.7047], [-103.1761, 69.1038], [-102.2912, 69.5161], [-101.7725, 69.0078],
  [-105.1246, 68.896], [-106.5944, 69.4954], [-109.0114, 68.7288], [-113.188, 68.4611], [-113.5544, 68.9991]],
 [[0.0, 50.7879], [-0.005, 50.7879], [-3.4767, 50.6871], [-3.6434, 50.2187], [-5.7159, 50.0617], [-4.2309, 51.1866], [-3.0567, 51.1771],
  [-2.3476, 51.7975], [-3.4034, 51.3804], [-5.3208, 51.8608], [-3.9375, 52.5533], [-4.0584, 52.9229], [-4.7692, 52.795], [-4.1992, 53.2125],
  [-2.6809, 53.355], [-3.1076, 53.5459], [-2.7993, 54.2399], [-3.1517, 54.0612], [-3.6425, 54.5083], [-3.0526, 54.9825], [-4.8567, 54.8688],
  [-4.8567, 54.6312], [-5.1458, 54.8549], [-4.8842, 55.9425], [-4.4275, 55.9033], [-4.7517, 56.2071], [-5.3159, 55.8516],
  [-5.8059, 55.3025], [-5.0042, 56.7125], [-6.2292, 56.7258], [-5.3875, 57.1083], [-5.8142, 57.855], [-5.0725, 57.8191], [-5.0034, 58.6279],
  [-3.0225, 58.6433], [-4.4384, 57.4862], [-1.8192, 57.6108], [-2.5393, 56.5666], [-3.3775, 56.3808], [-2.5959, 56.2675],
  [-3.8491, 56.1175], [-2.1376, 55.9158], [-1.2126, 54.5808], [-0.0775, 54.1199], [0.0, 53.7639], [0.0, 53.7637], [0.0, 53.7637],
  [-0.7808, 53.6958], [-0.2242, 53.6508], [0.0, 53.5437], [-0.0075, 52.885], [0.0, 52.8875], [0.0283, 52.897], [1.3016, 52.9337],
  [1.7658, 52.4783], [0.9325, 51.5908], [0.2542, 51.4659], [1.4266, 51.3929], [1.3942, 51.1533], [0.0, 50.7879]],
 [[-77.0881, 83.1256], [-77.0, 83.1295], [-66.3563, 82.9352], [-68.5894, 82.6228], [-63.5656, 82.8329], [-61.1103, 82.3593],
  [-62.3704, 81.9979], [-65.8151, 81.6212], [-69.2931, 81.7085], [-66.6168, 81.5151], [-70.0159, 81.082], [-64.5926, 81.4042],
  [-69.4923, 80.3702], [-70.7962, 80.558], [-70.1094, 80.1879], [-72.3636, 80.2176], [-70.4557, 80.0906], [-71.4636, 79.7227],
  [-74.2466, 79.8858], [-73.2014, 79.5166], [-78.0457, 79.3521], [-74.438, 79.0306], [-78.8632, 79.0649], [-74.5763, 78.5872],
  [-76.6304, 78.5243], [-75.0048, 78.3263], [-76.8826, 78.2128], [-75.865, 77.9577], [-78.1819, 77.9737], [-78.7277, 77.314],
  [-81.9607, 77.6837], [-81.7994, 77.1748], [-78.0051, 76.9917], [-78.3899, 76.4574], [-81.0656, 76.1315], [-82.7405, 76.8247],
  [-82.2669, 76.3939], [-83.3968, 76.7619], [-83.1991, 76.4138], [-84.3187, 76.6563], [-85.247, 76.279], [-86.61, 76.6415],
  [-86.6861, 76.3426], [-87.5277, 76.6281], [-87.6121, 76.336], [-88.4206, 76.7705], [-88.5949, 76.3995], [-89.6643, 76.5718],
  [-86.6164, 77.1811], [-88.2286, 77.8522], [-84.4679, 77.2953], [-82.2983, 78.0773], [-84.7894, 77.5206], [-84.5965, 78.5954],
  [-85.4515, 78.1018], [-85.8205, 78.3866], [-87.4828, 78.123], [-86.8529, 78.7388], [-85.0781, 78.9223], [-82.3457, 78.5676],
  [-83.256, 78.8444], [-81.4822, 79.0574], [-84.7051, 79.0196], [-83.3539, 79.0536], [-86.4967, 80.2981], [-83.7828, 80.2471],
  [-80.5979, 79.5598], [-83.1943, 80.3266], [-76.4525, 80.8747], [-78.9678, 80.8789], [-76.8417, 81.4528], [-80.9324, 80.6538],
  [-83.1152, 80.54], [-81.7337, 80.8154], [-83.2137, 80.8391], [-86.0618, 80.5293], [-85.5098, 80.8018], [-82.3061, 81.1776],
  [-87.4949, 80.623], [-89.4173, 80.912], [-84.6428, 81.2882], [-89.6989, 81.0058], [-87.2107, 81.4906], [-90.4047, 81.3625],
  [-89.5042, 81.6268], [-91.8911, 81.6581], [-87.9663, 82.1242], [-84.5032, 81.8811], [-86.8601, 82.1978], [-85.187, 82.4805],
  [-80.42, 82.0407], [-82.3566, 82.675], [-77.0881, 83.1256]],
 [[119.4783, -5.0004], [119.4779, -5.0], [119.5096, -3.5559], [118.9312, -3.57], [118.7529, -2.7767], [119.3588, -1.8984],
  [119.3346, -1.17], [119.7317, -0.6429], [119.8762, -0.86], [119.6087, -0.0075], [119.8412, -0.0991], [120.0437, 0.7216],
  [120.2825, 0.9862], [120.5983, 0.762], [120.9083, 1.3604], [123.9517, 0.8446], [125.1733, 1.6879], [124.3133, 0.4013], [120.3096, 0.4191],
  [120.0604, -0.6309], [120.6758, -1.4112], [121.0933, -1.4296], [121.6275, -0.8012], [123.4596, -0.7692], [121.6675, -1.9479],
  [121.2896, -1.8417], [122.4838, -3.1742], [122.1988, -3.6], [122.885, -4.413], [122.1109, -4.4929], [122.0067, -4.8988],
  [121.5434, -4.7696], [121.6087, -4.0617], [120.9029, -3.5675], [121.0929, -2.7158], [120.7542, -2.6229], [120.1854, -2.9717],
  [120.4704, -5.6251], [119.44, -5.6013], [119.4783, -5.0004]],
 [[170.0, -46.2554], [169.9979, -46.2575], [169.01, -46.6788], [167.6508, -46.1679], [166.6758, -46.2104], [166.8083, -45.3137],
  [167.1676, -45.4663], [166.9738, -45.1426], [168.37, -44.0112], [169.2216, -43.9729], [170.7491, -43.1013], [172.1338, -40.8525],
  [172.9784, -40.5288], [172.6546, -40.66], [173.195, -41.3329], [174.315, -40.9963], [173.9071, -41.2809], [174.3188, -41.215],
  [174.2788, -41.7417], [172.7788, -43.1283], [172.6521, -43.64], [173.0808, -43.8562], [172.2175, -43.9013], [171.6687, -43.5242],
  [172.2129, -43.9025], [171.3221, -44.005], [170.7521, -45.8708], [170.0, -46.2554]],
 [[114.0, -8.5904], [113.9984, -8.5921], [110.7075, -8.2021], [108.8625, -7.6096], [107.8433, -7.7396], [106.4017, -7.3846],
  [106.505, -6.9655], [105.2071, -6.7517], [105.7979, -6.4892], [106.0383, -5.8746], [108.3025, -6.2404], [108.9325, -6.8413],
  [110.4067, -6.9521], [111.0292, -6.4163], [112.5475, -6.8429], [113.1558, -7.7462], [114.4383, -7.7887], [114.5929, -8.7525],
  [114.0, -8.5904]],
 [[174.9375, -41.0029], [174.9406, -41.0], [175.0579, -39.9392], [173.7529, -39.2767], [174.5938, -38.8192], [174.9746, -37.7976],
  [174.5388, -37.0559], [174.9321, -37.0558], [174.4888, -37.0458], [174.1546, -36.4501], [174.4483, -36.6471], [174.5096, -36.2551],
  [173.9125, -36.0063], [174.0488, -36.3984], [172.6775, -34.4254], [173.0475, -34.4138], [173.2366, -35.0154], [173.3991, -34.783],
  [174.1049, -35.3479], [174.3317, -35.1763], [174.3196, -35.8358], [174.8687, -36.3667], [174.6579, -36.8808], [175.5759, -37.2438],
  [175.495, -36.5096], [176.0399, -37.6813], [177.1516, -38.0454], [177.9834, -37.5396], [178.5529, -37.69], [178.0038, -39.1159],
  [177.0437, -39.2017], [175.965, -41.2471], [175.2383, -41.6096], [174.6154, -41.2942], [174.9375, -41.0029]],
 [[-55.2016, 46.9989], [-55.2017, 46.9983], [-55.9911, 46.9534], [-54.7038, 47.6716], [-56.1312, 47.4643], [-55.7903, 47.9645],
  [-56.8565, 47.5273], [-58.3173, 47.7891], [-59.1676, 47.5648], [-59.4122, 47.8963], [-58.2746, 48.5133], [-59.2704, 48.4666],
  [-58.4076, 49.1304], [-57.8929, 48.9558], [-58.2371, 49.3933], [-57.7055, 49.4558], [-57.4109, 50.7025], [-55.9017, 51.6312],
  [-55.4942, 51.3725], [-56.0843, 51.3633], [-55.7308, 51.0933], [-56.8713, 49.5425], [-56.1601, 50.1562], [-56.2092, 49.9254],
  [-55.4747, 49.9675], [-56.1409, 49.4313], [-55.1596, 49.5449], [-55.3959, 49.0421], [-54.4792, 49.5704], [-54.4984, 49.2562],
  [-53.5041, 49.2837], [-54.1987, 48.385], [-53.0029, 48.5467], [-53.9634, 48.2329], [-53.6121, 48.0542], [-53.9427, 47.8559],
  [-53.5599, 47.5191], [-52.9209, 48.1729], [-53.2826, 47.5472], [-52.7937, 47.81], [-52.6217, 47.5244], [-53.1615, 46.6257],
  [-53.6251, 46.6413], [-53.5532, 47.2185], [-54.2004, 46.8211], [-53.7939, 47.4391], [-54.0854, 47.8791], [-55.2016, 46.9989]],
 [[-75.0, 19.9155], [-75.0001, 19.9155], [-77.7375, 19.842], [-77.1155, 20.3567], [-77.2409, 20.6654], [-78.1433, 20.7587],
  [-78.7509, 21.6371], [-81.8059, 22.1738], [-82.1596, 22.395], [-81.6479, 22.4899], [-81.8634, 22.6771], [-82.7575, 22.7029],
  [-84.0359, 21.9063], [-84.9521, 21.8708], [-84.2792, 22.0037], [-84.025, 22.7171], [-81.1417, 23.2062], [-79.7917, 22.9054],
  [-76.8925, 21.2988], [-75.7, 21.1188], [-75.7183, 20.6812], [-74.1429, 20.2458], [-75.0, 19.9155]],
 [[123.9992, 12.9679], [123.9987, 12.9675], [123.3183, 13.0079], [122.5333, 13.9646], [122.5958, 13.1646], [121.75, 13.9687],
  [121.2858, 13.5979], [120.6196, 13.8125], [120.9579, 14.6391], [120.5533, 14.8162], [120.475, 14.4104], [120.0804, 14.7892],
  [119.7479, 15.9649], [119.8799, 16.3962], [120.4221, 16.1667], [120.6217, 18.5479], [122.3196, 18.38], [122.1371, 17.7824],
  [122.4621, 16.8875], [121.3721, 15.3383], [121.6571, 14.4016], [122.2291, 13.8979], [122.2491, 14.2454], [122.7733, 14.3238],
  [123.1133, 13.6837], [123.2542, 14.0779], [123.715, 13.9421], [123.9746, 13.7199], [123.5304, 13.5683], [123.76, 13.0612],
  [124.1896, 13.065], [124.0791, 12.5379], [123.9992, 12.9679]],
 [[-17.0, 63.7937], [-16.9993, 63.7933], [-18.7334, 63.3904], [-21.0533, 63.9412], [-22.7043, 63.8008], [-21.3559, 64.3875],
  [-24.0459, 64.8883], [-21.8018, 65.0262], [-22.6041, 65.1866], [-21.6809, 65.4559], [-24.5276, 65.5025], [-23.7725, 65.5342],
  [-24.0966, 65.8071], [-23.2576, 65.6784], [-23.4701, 66.1987], [-22.4325, 65.8333], [-23.1367, 66.4321], [-21.3225, 66.0083],
  [-21.0867, 65.1579], [-20.4211, 66.0888], [-19.4051, 65.7171], [-18.7867, 66.1954], [-18.0518, 65.6479], [-18.2701, 66.1771],
  [-16.5751, 66.0837], [-16.1933, 66.5404], [-15.3983, 66.1596], [-14.5267, 66.3804], [-15.1842, 66.1067], [-13.4909, 65.0775],
  [-14.2325, 65.035], [-13.6808, 64.9142], [-14.9584, 64.2404], [-17.0, 63.7937]],
 [[125.0, 5.8587], [124.9991, 5.8596], [124.0579, 6.3808], [124.2613, 7.3741], [123.6875, 7.8146], [123.4083, 7.3571], [122.6258, 7.7771],
  [122.1517, 6.9096], [121.8962, 7.1116], [122.2221, 7.9617], [123.3784, 8.7296], [123.8371, 8.4299], [123.6659, 7.9537],
  [124.3833, 8.5963], [124.7416, 8.4987], [124.7966, 8.9971], [125.54, 8.9646], [125.4392, 9.8279], [126.2071, 9.3083], [126.0804, 8.6108],
  [126.3929, 8.4991], [126.6046, 7.2866], [126.1554, 6.9092], [126.1916, 6.2737], [125.6958, 7.2979], [125.3737, 6.7275],
  [125.6854, 5.9708], [125.385, 5.5596], [125.2651, 6.0937], [125.0, 5.8587]],
 [[-8.0, 51.8587], [-8.0017, 51.8579], [-9.775, 51.4446], [-9.4584, 51.7296], [-10.1642, 51.6116], [-9.5559, 51.8825], [-10.4042, 51.8433],
  [-9.7592, 52.1566], [-10.3634, 52.2354], [-8.6976, 52.6533], [-9.9409, 52.5575], [-8.8792, 53.2083], [-10.1841, 53.4083],
  [-9.5509, 53.8008], [-10.1142, 54.2408], [-8.5184, 54.2112], [-8.1159, 54.6491], [-8.8109, 54.6991], [-7.9851, 55.2262],
  [-7.6301, 54.9621], [-7.3884, 55.3812], [-6.9225, 55.2366], [-7.315, 55.0062], [-6.1417, 55.2279], [-5.4309, 54.4841], [-6.3992, 54.0133],
  [-6.3142, 52.2408], [-8.0, 51.8587]],
 [[143.2587, 41.9991], [143.2587, 41.9983], [141.6699, 42.6495], [141.0033, 42.3012], [140.4767, 42.5887], [140.2913, 42.2492],
  [141.1871, 41.8008], [139.9837, 41.5558], [139.8237, 42.6142], [140.5229, 42.9909], [140.3434, 43.3354], [141.417, 43.3225],
  [141.9133, 45.5187], [143.7349, 44.1062], [144.7758, 43.9271], [145.3433, 44.3421], [145.2295, 43.3367], [145.8179, 43.3866],
  [143.9941, 42.9271], [143.2587, 41.9991]],
 [[142.0, 45.9688], [141.9987, 45.97], [142.2541, 51.1234], [141.6258, 52.3275], [141.7676, 53.3758], [142.7858, 53.7058],
  [142.2509, 54.3034], [142.7066, 54.4287], [143.2058, 51.5175], [144.7462, 48.6441], [144.0442, 49.2554], [142.9901, 49.1354],
  [142.5295, 47.7809], [143.6163, 46.3717], [143.4149, 46.0221], [143.3767, 46.547], [142.5116, 46.7221], [142.0, 45.9688]],
 [[-71.1504, 17.9983], [-71.1513, 17.9975], [-71.4258, 17.6012], [-72.0535, 18.2336], [-73.8833, 18.0204], [-74.4829, 18.425],
  [-74.2159, 18.6737], [-72.3543, 18.5319], [-72.822, 19.0372], [-72.7006, 19.4557], [-73.458, 19.6883], [-72.8234, 19.9525],
  [-69.9625, 19.6812], [-69.7442, 19.2838], [-69.1529, 19.3033], [-69.6283, 19.0821], [-68.3254, 18.6066], [-68.645, 18.2079],
  [-70.6717, 18.4287], [-71.1504, 17.9983]],
 [[-120.3342, 71.9981], [-120.3456, 71.9959], [-120.6319, 71.4921], [-123.1011, 71.079], [-124.893, 71.9575], [-125.9907, 71.966],
  [-123.7517, 73.7526], [-124.7341, 74.3505], [-121.4512, 74.5566], [-119.1263, 73.9958], [-117.6254, 74.251], [-115.3224, 73.5408],
  [-120.3342, 71.9981]],
 [[80.2575, 5.9971], [80.2546, 6.0], [79.6954, 8.1975], [79.8248, 7.9704], [80.0537, 9.5983], [80.3988, 9.4866], [79.9062, 9.7559],
  [80.2091, 9.8321], [81.3546, 8.4908], [81.8804, 7.0233], [81.325, 6.1937], [80.2575, 5.9971]],
 [[145.9991, -43.3304], [146.0, -43.3304], [145.2554, -42.6241], [145.1683, -42.1963], [145.5554, -42.3576], [144.7437, -41.4217],
  [144.6837, -40.6684], [146.6067, -41.2604], [148.0776, -40.7704], [148.3612, -42.1975], [148.2066, -41.9504], [148.0063, -43.2309],
  [147.1967, -42.7279], [146.87, -43.6404], [145.9991, -43.3304]],
 [[-92.1665, 74.9923], [-92.1698, 75.0], [-93.0909, 76.3622], [-95.4029, 76.231], [-96.8012, 76.9836], [-93.6905, 76.9171],
  [-93.5531, 76.3871], [-90.9708, 76.6519], [-91.472, 76.4528], [-89.1956, 76.2358], [-91.6188, 76.2645], [-90.1941, 76.062],
  [-91.2005, 75.8172], [-89.9322, 76.0069], [-88.9315, 75.4288], [-88.7648, 75.6847], [-86.5536, 75.3608], [-81.5955, 75.8082],
  [-79.5716, 75.4574], [-80.3685, 75.0164], [-79.3486, 74.8886], [-81.7649, 74.4688], [-83.5125, 74.9031], [-83.5908, 74.5419],
  [-87.7156, 74.4551], [-88.5509, 74.9074], [-89.9811, 74.5295], [-92.1665, 74.9923]],
 [[-66.7242, -55.0025], [-66.7242, -55.0], [-71.9592, -54.6242], [-70.8167, -54.3238], [-70.7517, -54.6263], [-70.1192, -54.5675],
  [-70.9284, -54.1179], [-69.7668, -54.5838], [-69.455, -54.3413], [-69.3484, -54.7138], [-68.9636, -54.4759], [-70.0726, -54.065],
  [-69.3275, -53.4459], [-70.4342, -53.3809], [-70.4208, -52.7768], [-68.7584, -52.5553], [-67.7675, -53.8317], [-65.1142, -54.6417],
  [-66.7242, -55.0025]],
 [[58.2574, 73.9996], [58.2549, 73.9996], [57.4975, 74.1929], [57.875, 73.7654], [56.5937, 73.8808], [57.5988, 73.6316], [56.3763, 73.7392],
  [56.6949, 73.2454], [54.0737, 73.3466], [55.8463, 73.6467], [53.6925, 73.7921], [55.0349, 74.157], [56.3913, 74.0083], [55.1261, 74.2267],
  [56.9636, 74.67], [56.0084, 75.1833], [61.1264, 76.2879], [64.7799, 76.3296], [68.6067, 76.9554], [68.405, 76.2125], [61.3166, 75.3071],
  [59.1549, 74.4187], [58.2187, 74.5433], [58.2574, 73.9996]],
 [[-81.0, 63.4375], [-81.0081, 63.4356], [-82.4941, 63.6572], [-83.0042, 64.1813], [-85.2111, 63.1068], [-85.6798, 63.7133],
  [-87.1651, 63.5755], [-86.1788, 64.0888], [-85.5949, 65.9134], [-84.9102, 65.2145], [-84.5089, 65.4791], [-81.8252, 64.5485],
  [-81.9878, 63.9788], [-80.9211, 64.1224], [-80.1354, 63.7752], [-81.0, 63.4375]],
 [[-115.5598, 74.9946], [-115.5683, 74.9934], [-117.6638, 75.2521], [-115.0075, 75.6992], [-117.2174, 75.5815], [-114.7823, 75.8975],
  [-116.7309, 75.9051], [-114.6275, 76.1701], [-115.8991, 76.2904], [-114.8958, 76.5215], [-111.2242, 75.5167], [-108.8854, 75.4804],
  [-110.4153, 76.3551], [-109.1076, 76.8261], [-108.0, 75.7875], [-106.8408, 75.6504], [-106.5696, 76.0696], [-105.4249, 75.8539],
  [-106.0, 75.0526], [-108.7985, 75.074], [-112.3577, 74.4243], [-114.4266, 74.6958], [-111.0417, 75.278], [-113.9173, 75.0524],
  [-114.0301, 75.4748], [-115.5598, 74.9946]],
 [[-94.197, 78.9981], [-94.1939, 79.0], [-90.3333, 79.2438], [-95.1457, 79.2738], [-94.3316, 79.7694], [-96.6341, 79.8798],
  [-96.6783, 80.1463], [-94.3897, 79.9793], [-96.6196, 80.3509], [-93.7666, 80.5202], [-95.502, 80.8132], [-93.4926, 81.3797],
  [-88.3412, 80.0777], [-87.6149, 80.4064], [-87.4353, 79.5323], [-85.7039, 79.6196], [-84.8796, 79.2688], [-87.6132, 78.6421],
  [-88.1598, 78.9951], [-88.8626, 78.1466], [-89.9622, 78.6126], [-89.4763, 78.149], [-92.0551, 78.2034], [-92.9875, 78.4754],
  [-91.6602, 78.568], [-94.197, 78.9981]],
 [[15.4382, 77.0], [13.9149, 77.5242], [16.0999, 77.4596], [14.7384, 77.6533], [16.9916, 77.9292], [13.9498, 77.7162], [13.5984, 78.0567],
  [17.3984, 78.4294], [16.57, 78.7162], [15.4233, 78.4579], [15.3465, 78.8504], [13.0, 78.2046], [10.6647, 79.5466], [13.8065, 79.877],
  [12.3481, 79.58], [14.0432, 79.2637], [14.7265, 79.7888], [16.5266, 78.9054], [16.2698, 80.0587], [17.6717, 79.3608], [18.283, 79.6212],
  [21.5314, 78.7474], [19.0282, 78.4225], [16.9899, 76.5962], [15.4382, 77.0]],
 [[-50.7764, -0.783], [-50.3392, -0.0963], [-48.3896, -0.2825], [-48.8362, -1.4459], [-49.8065, -1.8046], [-50.565, -1.7623],
  [-50.7862, -1.099], [-50.4798, -1.0408], [-50.7764, -0.783]],
 [[130.6689, 30.9991], [130.7983, 31.6996], [130.5875, 31.1554], [130.1062, 31.4117], [130.6621, 32.6333], [130.4579, 33.2983],
  [130.118, 33.1291], [130.3505, 32.6733], [129.7383, 32.5679], [129.6758, 33.0971], [130.0096, 32.8466], [129.5604, 33.3316],
  [130.9667, 33.8461], [132.0862, 32.9308], [131.3496, 31.3649], [130.6689, 30.9991]],
 [[120.8641, 21.9996], [120.8633, 21.9996], [120.0296, 23.0758], [120.1288, 23.6325], [121.5725, 25.3004], [121.9996, 25.0074],
  [120.8641, 21.9996]],
 [[151.0529, -6.0017], [151.0529, -6.005], [149.6067, -6.2929], [148.3171, -5.5675], [149.8709, -5.5213], [150.0917, -4.9979],
  [150.0934, -5.5188], [150.8967, -5.4888], [151.6862, -4.8683], [151.4896, -4.2125], [152.1683, -4.1346], [152.4046, -4.3383],
  [151.9487, -5.0051], [152.1196, -5.3984], [151.0529, -6.0017]],
 [[-85.0, 71.2809], [-85.0033, 71.2809], [-86.9838, 71.2153], [-89.9999, 71.4543], [-89.1492, 73.183], [-86.5578, 73.8572],
  [-84.8316, 73.7437], [-86.2838, 72.7028], [-86.0963, 71.9778], [-85.0, 71.2809]],
 [[108.9992, 18.3604], [108.9966, 18.3604], [108.6312, 19.2683], [109.26, 19.9029], [110.9321, 20.0183], [111.042, 19.64],
  [110.4117, 18.6612], [109.5725, 18.1604], [108.9992, 18.3604]],
 [[-97.0, 71.7499], [-97.0038, 71.7464], [-98.1824, 71.6456], [-98.2199, 71.8954], [-98.1784, 71.4241], [-99.2139, 71.3519],
  [-102.6862, 72.693], [-102.2284, 73.0928], [-100.1657, 72.7873], [-99.775, 73.2109], [-101.6224, 73.4957], [-100.4251, 73.4074],
  [-100.8589, 73.8404], [-96.9579, 73.7485], [-98.4889, 73.0091], [-96.5221, 72.756], [-97.0, 71.7499]],
 [[53.9975, 70.7371], [53.9861, 70.7409], [53.5013, 71.5434], [51.91, 71.4554], [51.4062, 71.8325], [53.6587, 72.6417], [52.3636, 72.715],
  [53.6112, 72.8892], [53.1962, 73.15], [54.8099, 73.2565], [56.5188, 73.1509], [55.0811, 72.5775], [55.1862, 71.8992], [57.6111, 70.7125],
  [53.9975, 70.7371]],
 [[-125.6726, 48.9996], [-125.6722, 49.0], [-127.095, 50.1463], [-127.9175, 50.1142], [-127.4193, 50.6058], [-128.0483, 50.4423],
  [-128.4323, 50.7855], [-125.4451, 50.3354], [-123.2656, 48.4524], [-125.1017, 48.7204], [-124.8267, 49.2554], [-125.6726, 48.9996]],
 [[124.0, -10.2779], [123.9967, -10.2779], [123.4546, -10.36], [123.6663, -9.6292], [125.1234, -8.6479], [127.3012, -8.4009],
  [124.0, -10.2779]],
 [[14.9992, 36.7012], [14.9975, 36.7012], [12.6583, 37.5654], [12.4888, 38.0183], [15.6508, 38.2713], [14.9992, 36.7012]],
 [[-95.0, 71.982], [-95.0169, 71.9804], [-95.2909, 73.9931], [-90.1869, 73.9152], [-92.0995, 72.7407], [-94.2929, 72.7741],
  [-93.4519, 72.4411], [-95.0, 71.982]],
 [[8.4512, 38.9974], [8.4512, 38.995], [8.1762, 40.9383], [9.2308, 41.2596], [9.832, 40.5283], [9.6345, 39.2974], [8.4512, 38.9974]],
 [[137.9938, 74.845], [137.9936, 74.8466], [136.9883, 75.585], [138.7832, 76.2021], [140.57, 75.5954], [141.38, 76.1762],
  [145.4084, 75.4916], [143.4032, 75.0437], [142.4033, 75.7337], [143.5874, 74.8938], [137.9938, 74.845]],
 [[132.9987, 32.9833], [132.9987, 32.979], [132.6308, 32.7612], [132.4229, 33.4633], [132.0179, 33.3467], [132.94, 34.1438],
  [133.1317, 33.9104], [133.895, 34.3838], [134.5891, 34.2404], [134.7504, 33.835], [134.1775, 33.2446], [133.5558, 33.5546],
  [132.9987, 32.9833]],
 [[128.0, -0.303], [128.0004, -0.305], [128.4433, -0.9137], [127.6571, -0.2225], [127.3954, 1.0449], [127.95, 2.2271], [128.0162, 1.3108],
  [127.6296, 0.9174], [128.72, 1.5788], [128.7004, 1.0667], [128.1887, 0.7875], [128.9187, 0.2016], [127.9241, 0.4613], [128.0, -0.303]],
 [[127.9991, -3.0713], [128.0, -3.0713], [129.0409, -2.7879], [130.3358, -2.9688], [130.8504, -3.8634], [129.9067, -3.3296],
  [128.4492, -3.4521], [128.1775, -3.0563], [127.91, -3.5662], [127.9991, -3.0713]],
 [[167.0, -22.3432], [166.9967, -22.3438], [164.9442, -21.3613], [163.9978, -20.0799], [167.0, -22.3432]],
 [[-100.0166, 74.9874], [-100.0563, 74.9864], [-100.7793, 75.3551], [-99.6299, 75.6913], [-102.8603, 75.6289], [-100.9174, 75.8127],
  [-101.8967, 76.4477], [-99.9138, 75.8734], [-100.9851, 76.5032], [-99.0563, 76.392], [-98.4929, 76.6847], [-97.2633, 75.3981],
  [-97.9731, 75.0238], [-100.0166, 74.9874]],
 [[-117.0, 77.4387], [-117.0, 77.4398], [-115.351, 77.3142], [-117.0684, 76.3063], [-117.8385, 76.8293], [-119.7582, 75.8764],
  [-120.8541, 76.2133], [-122.5814, 75.9386], [-119.2133, 77.3113], [-117.0, 77.4387]],
 [[116.9996, -9.0951], [116.9996, -9.0942], [116.7479, -8.6575], [117.1167, -8.3663], [118.2746, -8.6559], [117.6896, -8.2367],
  [117.9258, -8.0813], [119.197, -8.6109], [116.9996, -9.0951]],
 [[95.9898, 78.9771], [95.9833, 78.9771], [92.8615, 79.5575], [97.6799, 80.1663], [97.6633, 79.8454], [98.7751, 80.072],
  [100.1116, 79.8032], [100.0048, 78.9092], [95.9898, 78.9771]],
 [[27.0, 79.9446], [27.0049, 79.9433], [23.6566, 79.1813], [20.0999, 79.4529], [21.7117, 79.8025], [18.7466, 79.7112], [17.7376, 80.1258],
  [19.5699, 80.1362], [19.6249, 80.5054], [22.26, 79.9712], [22.7948, 80.5046], [23.0747, 80.1037], [24.8302, 80.3463], [27.0, 79.9446]],
 [[123.0, -8.3379], [122.9992, -8.3379], [121.6508, -8.9138], [119.8003, -8.7384], [120.4233, -8.2288], [121.5058, -8.6238],
  [122.4317, -8.5491], [122.8658, -8.0688], [123.0, -8.3379]],
 [[-99.0059, 68.9172], [-99.0103, 68.9168], [-97.9594, 69.9041], [-95.1554, 68.8706], [-96.5419, 68.4517], [-99.0059, 68.9172]],
 [[123.2159, 9.9996], [123.2149, 9.9996], [123.3112, 9.3025], [123.0125, 9.0404], [122.3754, 9.8392], [122.8521, 10.075],
  [123.1942, 11.0045], [123.5654, 10.7958], [123.2159, 9.9996]],
 [[125.0084, 11.3045], [125.0429, 11.7516], [124.2791, 12.5846], [125.145, 12.5854], [125.5129, 12.1958], [125.7588, 11.0191],
  [125.0084, 11.3045]],
 [[138.0, -8.3938], [137.9992, -8.3929], [137.6154, -8.3884], [138.1674, -7.5338], [139.1054, -7.575], [138.9558, -8.0513],
  [138.0, -8.3938]],
 [[106.6917, -3.0014], [106.6916, -3.0022], [105.9467, -2.8096], [105.7583, -2.1396], [105.1291, -2.0871], [105.3416, -1.6471],
  [106.0437, -1.6], [106.2912, -2.415], [106.8446, -2.5725], [106.6917, -3.0014]],
 [[118.0, 8.8837], [117.9992, 8.8846], [117.1738, 8.3425], [119.2571, 10.4883], [119.2191, 10.9554], [119.4721, 10.7267],
  [119.5008, 11.4188], [119.7171, 10.5108], [118.7508, 9.9287], [118.0, 8.8837]],
 [[121.9967, 10.4404], [121.9958, 10.4395], [121.8833, 11.9021], [122.8675, 11.4212], [123.1521, 11.6], [123.1271, 11.1716],
  [121.9967, 10.4404]],
 [[-99.0, 77.9047], [-98.9929, 77.9033], [-102.7744, 78.3965], [-104.4361, 78.2664], [-103.299, 78.735], [-104.9863, 78.7968],
  [-105.4517, 79.3367], [-100.3566, 78.8343], [-99.0, 77.9047]],
 [[100.3249, 77.995], [100.3165, 77.9929], [102.4762, 79.4145], [102.4678, 78.8124], [103.8866, 79.1704], [105.4713, 78.4924],
  [100.3249, 77.995]],
 [[-80.1328, 72.9989], [-80.1312, 73.0], [-80.8121, 73.7543], [-77.4098, 73.5536], [-76.0544, 72.8611], [-79.3356, 72.7386],
  [-80.1328, 72.9989]],
 [[-77.0, 17.8438], [-77.0004, 17.8441], [-77.74, 17.8537], [-78.3455, 18.3575], [-76.8984, 18.4129], [-76.1846, 17.9167],
  [-77.0, 17.8438]],
 [[120.0, -10.0496], [119.9987, -10.0484], [118.9446, -9.515], [119.9416, -9.2746], [120.8262, -10.0292], [120.4525, -10.3179],
  [120.0, -10.0496]],
 [[-59.8463, 45.9983], [-59.8463, 45.9941], [-61.2925, 45.5479], [-61.5545, 46.0401], [-60.4133, 47.0393], [-60.4149, 46.2933],
  [-61.1637, 45.7108], [-60.7893, 45.7263], [-60.2914, 46.3423], [-59.8463, 45.9983]],
 [[178.647, -18.0009], [178.647, -18.0051], [177.8983, -18.2771], [177.2462, -17.9584], [178.1833, -17.3104], [178.647, -18.0009]],
 [[-155.7869, 18.9982], [-155.788, 19.0], [-155.8534, 20.2728], [-154.8058, 19.5191], [-155.7869, 18.9982]],
 [[49.3517, 39.3214], [49.4881, 40.1506], [50.3589, 40.3703], [49.5281, 40.6628], [47.7383, 42.6339], [47.6983, 43.8689],
  [47.4072, 43.5011], [46.7125, 44.6478], [47.3803, 45.7447], [48.7875, 45.8347], [49.3019, 46.5839], [51.19, 47.1147], [52.9056, 46.9619],
  [53.2072, 46.6467], [52.7331, 45.5494], [53.2567, 45.3303], [51.4147, 45.3844], [50.9531, 44.8619], [51.5703, 44.5139],
  [50.3156, 44.6586], [50.2306, 44.3744], [50.84, 44.1931], [51.2656, 43.1531], [52.7383, 42.7103], [52.4053, 42.0883], [52.8578, 41.0625],
  [52.9467, 41.9731], [53.6594, 42.1433], [54.7661, 41.0386], [53.7464, 40.6153], [52.9186, 41.0817], [52.6903, 40.2719],
  [53.0389, 39.7397], [52.9056, 40.0122], [53.5758, 39.9661], [53.7339, 39.5122], [53.2642, 39.6553], [53.16, 39.18], [53.5639, 39.3575],
  [53.9839, 38.9158], [54.0286, 36.8247], [51.8739, 36.5842], [49.2133, 37.5742], [48.83, 38.8594], [49.3403, 39.3281],
  [49.3517, 39.3214]],
 [[-86.2292, 42.9997], [-86.6203, 41.9122], [-87.3978, 41.6372], [-87.9097, 43.2339], [-87.3269, 44.7819], [-87.9894, 44.7169],
  [-86.9789, 45.92], [-86.6236, 45.605], [-85.5289, 46.1014], [-83.9036, 45.9697], [-84.2965, 46.4238], [-85.0244, 46.4917],
  [-84.9564, 46.7853], [-87.3497, 46.5117], [-87.8167, 46.9042], [-88.4836, 46.765], [-88.6489, 47.2344], [-90.4478, 46.5683],
  [-90.9503, 46.6019], [-90.8619, 46.9667], [-92.11, 46.7686], [-89.1386, 48.4844], [-88.7464, 48.3661], [-88.4803, 48.8544],
  [-88.5597, 48.4303], [-88.2044, 48.6067], [-88.2569, 48.9975], [-86.4347, 48.7772], [-85.8603, 47.9853], [-84.8486, 47.9528],
  [-84.5172, 46.4822], [-80.8069, 45.9508], [-79.6831, 44.8792], [-80.0897, 44.4681], [-80.9342, 44.5933], [-81.6953, 45.2647],
  [-81.2658, 44.6233], [-81.7456, 43.3458], [-82.4153, 43.0153], [-82.7928, 44.0206], [-83.9464, 43.73], [-83.3333, 44.3356],
  [-83.3889, 45.2811], [-84.9903, 45.7733], [-85.5283, 44.7717], [-85.5561, 45.2175], [-86.265, 44.7006], [-86.2342, 43.005],
  [-86.2292, 42.9997]],
 [[31.6133, -2.0456], [32.0181, -0.05], [32.6508, 0.2911], [32.9789, 0.0889], [33.3644, 0.4919], [33.4881, 0.1828], [33.9875, 0.2706],
  [34.2778, -0.3925], [34.7372, -0.0897], [34.8394, -0.3097], [34.06, -0.5644], [33.9569, -1.5239], [33.2161, -2.0425], [33.8353, -2.2258],
  [33.4261, -2.5572], [32.9158, -2.4189], [33.0261, -2.8806], [32.7569, -3.0139], [32.8347, -2.5142], [32.2183, -2.2517],
  [31.7947, -2.8344], [31.6139, -2.0442], [31.6133, -2.0456]],
 [[61.4519, 44.7356], [60.7269, 44.0189], [61.0161, 43.6294], [59.7575, 43.3906], [59.4958, 43.7964], [58.3311, 43.7522],
  [58.1792, 44.8525], [58.6717, 45.8342], [59.4914, 45.8142], [59.4264, 46.3108], [59.8886, 46.0967], [60.235, 46.7164], [61.0853, 46.4839],
  [61.7114, 46.7458], [60.8964, 46.1058], [60.9328, 45.6406], [61.965, 45.0436], [61.4525, 44.7356], [61.4519, 44.7356]],
 [[29.0456, -4.3092], [29.3108, -3.3458], [29.9489, -5.8206], [29.7142, -6.2589], [30.5689, -6.9678], [31.1786, -8.7517],
  [30.4733, -8.5136], [30.1478, -7.2872], [29.1903, -6.0814], [29.2378, -4.0531], [29.0456, -4.3092]],
 [[109.4856, 55.7244], [109.965, 55.6719], [109.5597, 54.1431], [108.5275, 53.5094], [108.9658, 53.3547], [106.3333, 52.3389],
  [105.9194, 51.7281], [104.7592, 51.4647], [103.74, 51.6622], [105.2247, 51.9156], [106.5753, 52.7228], [108.2078, 53.9225],
  [109.4856, 55.7253], [109.4856, 55.7244]],
 [[-123.4919, 65.1633], [-122.1397, 65.7839], [-122.7856, 65.9664], [-121.2292, 66.0792], [-123.0086, 66.255], [-124.79, 65.9089],
  [-124.9756, 66.3067], [-119.8064, 67.0469], [-118.9528, 66.8747], [-120.4517, 66.3783], [-118.5578, 66.3311], [-117.6036, 66.6747],
  [-117.8069, 65.7242], [-119.6914, 65.7939], [-119.4747, 65.3217], [-121.1772, 64.8103], [-120.4078, 65.5858], [-122.0394, 64.9389],
  [-123.4914, 65.1603], [-123.4919, 65.1633]],
 [[-115.0219, 60.8878], [-117.6839, 61.3025], [-115.9611, 61.1725], [-115.5456, 61.7492], [-114.6214, 61.8556], [-115.9925, 62.7792],
  [-113.3186, 61.9897], [-110.7158, 62.9206], [-108.9008, 62.7997], [-111.3911, 62.6319], [-111.6708, 62.3592], [-110.4147, 62.6801],
  [-109.6831, 62.6656], [-110.7714, 62.5392], [-109.8503, 62.5553], [-111.0625, 62.3806], [-113.7419, 61.0086], [-115.0272, 60.8819],
  [-115.0219, 60.8878]],
 [[33.9283, -9.6997], [34.0436, -9.4847], [34.5047, -9.975], [34.9597, -11.4867], [34.6903, -12.4272], [35.2406, -14.4053],
  [34.5644, -14.1106], [33.9961, -12.2525], [34.3272, -11.6478], [33.9222, -9.7031], [33.9283, -9.6997]],
 [[-81.6547, 42.4781], [-80.1006, 42.5431], [-80.4494, 42.6233], [-78.8881, 43.0494], [-79.1447, 42.5703], [-81.7244, 41.4969],
  [-83.4617, 41.7219], [-82.7186, 42.6928], [-82.4019, 42.3875], [-82.9945, 42.2701], [-82.9175, 41.9853], [-81.6558, 42.4783],
  [-81.6547, 42.4781]],
 [[-98.4353, 53.0467], [-99.2158, 53.33], [-98.9083, 53.8789], [-97.8842, 53.7031], [-98.0111, 54.4144], [-97.7317, 54.0864],
  [-96.2581, 51.2128], [-96.6619, 50.3975], [-96.7228, 51.6064], [-97.2592, 51.4311], [-97.4869, 52.1311], [-98.0256, 51.9436],
  [-98.8983, 52.8506], [-98.5247, 52.9975], [-98.4353, 53.0467]],
 [[-75.8122, 44.5025], [-76.3636, 44.1214], [-76.1911, 43.5781], [-76.9444, 43.2542], [-79.7689, 43.2892], [-79.0925, 43.8264],
  [-77.1344, 43.8556], [-76.8467, 44.1183], [-77.5672, 44.1136], [-75.8136, 44.5019], [-75.8122, 44.5025]],
 [[30.9419, 61.7292], [32.9556, 60.6508], [31.4106, 59.8992], [29.7961, 61.2161], [30.9419, 61.7292]],
 [[74.1897, 46.4289], [75.075, 46.8439], [78.3142, 46.4556], [79.2542, 46.6625], [78.425, 46.2797], [75.4772, 46.6725], [74.2319, 45.9775],
  [74.1347, 44.9592], [73.4175, 45.8], [74.1903, 46.4297], [74.1897, 46.4289]],
 [[15.1728, 13.0411], [14.0031, 12.4883], [13.225, 13.8636], [13.6328, 13.9917], [13.8592, 13.3328], [14.2178, 13.4978], [14.2353, 13.1628],
  [15.1714, 13.0439], [15.1728, 13.0411]],
 [[27.8325, 61.1467], [27.2714, 61.6675], [28.0733, 61.4797], [28.8728, 61.8208], [27.2589, 62.4606], [27.7036, 62.8975],
  [26.8964, 63.5922], [28.9689, 62.3853], [27.95, 62.7647], [27.9322, 62.2589], [29.3186, 62.1522], [29.2822, 62.6078], [29.9497, 62.4422],
  [29.7308, 61.825], [28.2967, 61.3564], [28.8681, 61.2222], [27.8547, 61.1439], [27.8325, 61.1467]],
 [[34.6411, 62.8606], [35.8358, 62.4403], [36.4583, 61.18], [35.8692, 60.8533], [34.8328, 61.0131], [35.69, 61.1372], [34.3178, 61.8214],
  [33.9656, 62.5411], [34.7297, 62.0214], [34.4633, 62.5931], [35.0578, 62.0442], [35.5892, 62.2406], [34.6411, 62.8606]],
 [[-53.741, -2.1558], [-54.0958, -2.3844], [-54.9778, -2.4769], [-55.3956, -3.6214], [-55.2128, -2.3661], [-54.7239, -2.3942],
  [-55.6433, -2.2414], [-55.0928, -2.2197], [-55.6097, -1.9746], [-56.6728, -2.6039], [-57.5292, -2.4147], [-58.6303, -3.3747],
  [-59.7144, -3.1837], [-61.8136, -3.9803], [-62.3861, -3.6942], [-63.5247, -4.4797], [-63.6139, -4.0903], [-63.3256, -4.0084],
  [-64.685, -3.3594], [-64.7861, -3.5761], [-65.2719, -2.7619], [-65.9906, -2.4892], [-67.3653, -2.5494], [-68.1461, -3.4886],
  [-69.25, -3.4467], [-69.5883, -3.9339], [-69.3644, -4.0767], [-69.9083, -4.3861], [-70.2867, -3.8711], [-71.455, -3.7447],
  [-71.8092, -3.1917], [-73.0094, -3.46], [-72.6647, -3.29], [-72.8922, -3.1892], [-72.4279, -3.2988], [-71.8203, -3.1592],
  [-71.45, -3.7214], [-70.2753, -3.7964], [-69.8194, -4.3311], [-69.3983, -4.0806], [-69.6361, -3.9486], [-69.2753, -3.4189],
  [-68.0855, -3.3607], [-67.3633, -2.5186], [-65.3956, -2.4128], [-64.3318, -3.5021], [-63.0694, -4.0583], [-62.3869, -3.6736],
  [-61.9144, -3.9261], [-60.9631, -3.5964], [-60.8319, -3.16], [-59.9153, -3.1931], [-60.5806, -3.0317], [-61.6211, -1.4606],
  [-63.6275, -0.4108], [-66.3461, -0.4022], [-65.8506, -0.2406], [-63.6903, -0.2936], [-61.5942, -1.4267], [-60.0033, -3.1558],
  [-58.4269, -3.1587], [-57.5747, -2.4031], [-56.6725, -2.5675], [-56.0414, -1.9414], [-54.8792, -1.9814], [-54.5317, -2.1153],
  [-54.5722, -2.4121], [-54.2531, -2.3811], [-52.7061, -1.5614], [-53.741, -2.1558]],
 [[120.2328, 60.3325], [118.2197, 59.6667], [116.8014, 60.4042], [114.9675, 60.6861], [112.3533, 59.2306], [114.9608, 60.6986],
  [116.8575, 60.4172], [118.327, 59.7334], [121.2267, 60.6672], [124.1078, 60.5925], [129.2381, 61.8075], [129.5256, 63.2326],
  [126.4511, 64.3719], [125.2831, 63.8578], [123.6764, 64.0922], [118.9958, 63.4742], [123.67, 64.1033], [125.2506, 63.9013],
  [126.1694, 64.2951], [124.3803, 65.1397], [122.9633, 67.54], [125.75, 70.4614], [127.5483, 70.8042], [127.2961, 71.4014],
  [127.615, 70.805], [125.8422, 70.4539], [123.2698, 67.7618], [124.4228, 65.3147], [128.3381, 63.7701], [133.2172, 63.3956],
  [133.2839, 63.0747], [135.5503, 62.6683], [135.1197, 60.3719], [134.5472, 60.4456], [135.0252, 60.4402], [135.1808, 61.2228],
  [135.5117, 62.6547], [133.2578, 63.0517], [133.195, 63.3672], [129.452, 63.4742], [130.0711, 62.2956], [128.7275, 61.2678],
  [124.1097, 60.5714], [121.2775, 60.6458], [120.6817, 60.4289], [121.1347, 60.0714], [120.2383, 60.3353], [120.2328, 60.3325]],
 [[24.5906, 0.6819], [22.3856, 2.1508], [19.31, 1.4904], [18.3986, 0.8564], [18.4469, -0.0144], [16.2397, -2.1358], [16.1889, -3.0625],
  [16.1197, -2.2072], [17.7019, -0.4461], [18.2214, 2.43], [17.7892, -0.4242], [18.4553, 1.0492], [20.2086, 2.0022], [22.4761, 2.1747],
  [24.5867, 0.6875], [24.5906, 0.6819]],
 [[65.8931, 63.7978], [65.0917, 62.985], [68.613, 61.3342], [74.4639, 61.2308], [78.1208, 60.572], [79.3422, 59.5247], [81.2419, 59.1242],
  [83.8739, 57.7392], [84.4864, 57.1231], [84.0122, 56.2822], [84.4483, 57.1231], [82.9809, 58.1157], [79.2961, 59.5147],
  [78.2458, 60.4095], [75.2083, 61.047], [69.2369, 61.074], [69.8906, 60.6581], [69.9442, 59.8244], [68.9069, 59.4403], [68.2231, 58.1614],
  [68.8736, 59.4519], [69.9169, 59.8344], [69.8375, 60.6492], [65.0189, 62.9986], [65.7505, 63.9467], [65.2083, 65.0753],
  [65.6564, 66.1339], [66.5172, 66.6517], [68.9697, 66.8069], [68.075, 66.4892], [66.5278, 66.6136], [65.7781, 66.1475], [65.2606, 65.085],
  [65.8894, 63.8017], [65.8931, 63.7978]],
 [[-55.7853, -27.4333], [-58.8381, -27.4722], [-59.5614, -29.455], [-59.5672, -30.6431], [-60.6853, -31.9578], [-60.4567, -33.1892],
  [-60.755, -31.88], [-59.6864, -30.7719], [-59.6094, -29.2647], [-59.42, -29.0108], [-58.895, -27.4692], [-58.3539, -27.2297],
  [-55.7872, -27.4306], [-55.7853, -27.4333]],
 [[105.8194, 10.0017], [105.1278, 10.7031], [104.5478, 12.5433], [104.9383, 11.5725], [105.5789, 12.3014], [105.9826, 12.3584],
  [105.7951, 14.8959], [105.2547, 15.2397], [105.5326, 15.6709], [104.7439, 16.5225], [104.7828, 17.4125], [103.9717, 18.3253],
  [103.0867, 18.1381], [103.3008, 18.4278], [103.9822, 18.3331], [104.7869, 17.4244], [104.7572, 16.5247], [105.4283, 16.0131],
  [105.9022, 14.9858], [105.8826, 14.1459], [106.2342, 13.575], [105.9951, 13.4584], [106.0675, 12.3467], [105.0326, 11.5959],
  [105.4483, 10.5611], [106.4394, 10.3153], [105.3417, 10.5525], [105.8325, 10.0039], [105.8194, 10.0017]],
 [[130.3278, 46.8292], [132.4936, 47.6942], [131.2164, 47.7269], [132.547, 47.7242], [135.9724, 49.0418], [136.4474, 49.6293],
  [136.1792, 49.8403], [136.7099, 49.9418], [136.9289, 50.5158], [140.1736, 51.9152], [140.4117, 52.3942], [139.6875, 52.9797],
  [140.6622, 53.105], [139.7642, 52.9464], [140.4919, 52.3494], [140.0367, 51.6656], [140.3617, 52.0258], [140.2269, 51.6853],
  [140.7645, 51.6217], [139.0349, 51.3168], [137.3099, 50.6168], [137.4392, 50.2006], [137.1119, 50.3064], [135.7997, 48.7547],
  [133.2599, 47.9043], [131.9819, 47.24], [130.3364, 46.8269], [130.3278, 46.8292]],
 [[87.6247, 64.0078], [87.2664, 63.3681], [89.0144, 62.3625], [87.215, 63.3844], [87.9632, 65.7105], [87.4794, 66.3636], [86.2089, 66.5911],
  [86.0481, 69.4442], [83.9961, 69.7544], [85.4283, 69.8119], [86.1922, 69.4508], [86.3286, 66.5789], [87.5344, 66.3783],
  [87.9007, 65.8262], [89.0108, 65.9681], [89.99, 65.6606], [90.4444, 64.7978], [92.0778, 64.3781], [90.395, 64.7692], [89.9756, 65.6239],
  [88.807, 65.9293], [88.0228, 65.7769], [87.62, 64.0128], [87.6247, 64.0078]],
 [[119.6147, 32.1839], [118.7575, 32.1283], [118.3114, 31.2714], [116.1911, 29.7219], [116.1256, 29.2436], [116.7161, 29.1303],
  [116.5225, 28.7736], [115.9161, 29.2056], [116.1833, 29.7472], [114.2933, 30.5844], [114.8736, 30.6161], [115.5163, 30.069],
  [116.2542, 29.7947], [118.3169, 31.3153], [118.7394, 32.215], [119.6058, 32.1997], [119.6147, 32.1839]],
 [[-67.4597, 6.1981], [-67.1139, 7.1347], [-66.3406, 7.6675], [-65.5522, 7.9314], [-64.82, 7.6806], [-62.2564, 8.5953], [-62.2039, 8.8893],
  [-62.3719, 9.7458], [-62.1094, 8.7122], [-61.6486, 8.6247], [-64.0801, 7.914], [-64.8453, 7.6408], [-65.5844, 7.8831], [-66.3194, 7.64],
  [-67.4544, 6.1931], [-67.4597, 6.1981]],
 [[-71.2992, 46.7422], [-75.7792, 44.4878], [-74.0156, 45.4444], [-76.36, 45.4561], [-77.6897, 46.1861], [-76.3531, 45.4711],
  [-74.0417, 45.4775], [-71.3028, 46.7561], [-71.2992, 46.7422]],
 [[-124.7889, 64.4247], [-125.28, 64.8631], [-128.7197, 65.6319], [-128.6852, 66.2673], [-130.8114, 67.3094], [-132.8192, 67.1928],
  [-134.2772, 67.6561], [-134.2331, 68.6933], [-133.715, 67.5145], [-130.3542, 67.3119], [-128.6469, 66.3619], [-128.7008, 65.6923],
  [-125.1317, 64.8722], [-124.7856, 64.4264], [-124.7889, 64.4247]],
 [[89.6697, 24.0044], [89.7058, 25.7161], [90.5967, 26.2333], [89.7556, 25.6906], [89.6881, 24.0042], [89.6697, 24.0044]],
 [[-157.9117, 64.8547], [-159.9796, 62.8858], [-160.0703, 61.9467], [-161.9681, 61.5586], [-162.53, 61.9791], [-163.8919, 62.0964],
  [-164.4803, 62.7608], [-163.9236, 62.1594], [-162.1742, 61.9694], [-161.9353, 61.5878], [-160.089, 62.2358], [-160.2189, 62.6833],
  [-157.9197, 64.8542], [-157.9117, 64.8547]],
 [[57.0225, 65.3872], [56.1761, 66.3192], [54.2783, 65.4517], [52.1053, 65.4344], [52.0942, 67.5819], [54.091, 68.1547], [52.1319, 67.5697],
  [52.1832, 65.46], [54.2922, 65.4767], [56.2106, 66.3406], [57.0242, 65.3931], [57.0225, 65.3872]],
 [[-51.3767, -2.3353], [-51.6089, -1.7978], [-50.6789, -1.8114], [-50.9894, -2.2215], [-50.8367, -2.5058], [-51.2706, -1.8184],
  [-51.3786, -2.3272], [-51.3767, -2.3353]],
 [[69.6167, 28.2667], [68.2992, 27.5686], [67.9172, 26.9542], [68.4411, 25.6503], [67.7816, 24.1156], [68.0279, 23.949], [67.6591, 23.9278],
  [68.4239, 25.6536], [67.9047, 26.965], [68.2961, 27.5747], [69.5872, 28.265], [69.6167, 28.2667]],
 [[6.3778, 8.6592], [6.7668, 7.8085], [9.0322, 7.8703], [6.7814, 7.7939], [6.4781, 5.4294], [6.74, 7.8403], [6.3778, 8.6592]],
 [[43.4131, 56.6589], [44.6763, 56.0602], [46.5714, 56.3719], [47.9261, 56.1569], [49.0792, 55.4842], [47.8536, 56.1189],
  [46.5669, 56.3564], [44.6275, 56.0503], [43.4189, 56.6553], [43.4131, 56.6589]],
 [[-12.7769, 15.1475], [-14.3492, 16.6336], [-16.1289, 16.5478], [-16.4141, 16.2849], [-16.1497, 16.56], [-14.3411, 16.6433],
  [-13.4847, 16.1528], [-12.7769, 15.1475]],
 [[45.1286, 62.0014], [42.0367, 63.0936], [41.3875, 63.7572], [41.8278, 64.2639], [40.5389, 64.5628], [41.8497, 64.2814],
  [41.4342, 63.7533], [42.0956, 63.1033], [45.12, 62.0075], [45.1286, 62.0014]],
 [[87.8475, 25.0608], [89.2131, 23.9167], [88.0797, 24.5456], [87.8556, 25.0453], [87.8475, 25.0608]],
 [[160.7667, 68.5356], [158.5306, 68.7396], [156.2853, 67.6419], [153.8919, 67.4933], [156.1681, 67.6771], [156.0939, 67.9656],
  [157.5306, 68.2896], [158.4772, 68.8097], [160.74, 68.5628], [160.7667, 68.5356]],
 [[111.3836, 23.4711], [110.0864, 23.4003], [111.3856, 23.4819], [111.6528, 23.1592], [112.7931, 23.1647], [113.1739, 22.5667],
  [112.7792, 23.15], [111.6467, 23.1394], [111.3836, 23.4711]],
 [[44.0331, 34.0744], [44.4567, 33.9369], [44.5764, 33.0678], [45.5858, 32.6068], [46.6633, 32.4917], [46.8758, 32.0031],
  [46.6525, 32.4808], [45.8239, 32.4361], [44.5697, 33.0575], [44.4392, 33.9411], [44.0422, 34.0706], [44.0331, 34.0744]],
 [[34.1444, -16.6522], [34.8567, -16.9564], [35.54, -17.9886], [34.8592, -16.9933], [34.1461, -16.6586], [34.1444, -16.6522]],
 [[105.0686, 72.7644], [102.1292, 71.9458], [102.9847, 71.5289], [102.0289, 71.8986], [99.5558, 71.4458], [105.0444, 72.7836],
  [105.0686, 72.7644]],
 [[123.7475, 72.0764], [122.4803, 71.0353], [123.6598, 72.0309], [120.5878, 72.8708], [123.7475, 72.0764]],
 [[150.0236, 71.2075], [147.4692, 70.2536], [147.435, 69.0181], [147.4114, 70.3411], [150.0, 71.2206], [150.0236, 71.2075]],
 [[-121.1892, 61.8642], [-123.4156, 62.2853], [-123.2278, 62.9531], [-123.3268, 62.2944], [-121.1892, 61.8642]],
 [[-0.0578, 16.2639], [-0.0514, 16.2631], [0.0, 16.2006], [0.0028, 16.1972], [0.3022, 15.8083], [0.0, 16.1908], [-0.0578, 16.2639]],
 [[-180.0, -77.8188], [-179.06, -77.8652], [-176.48, -78.298], [-163.261, -78.5976], [-158.636, -77.8722], [-158.212, -77.0705],
  [-151.852, -77.3906], [-148.387, -75.7466], [-145.548, -75.3684], [-140.851, -75.528], [-135.42, -74.5642], [-128.19, -74.3146],
  [-127.129, -73.2788], [-124.029, -73.8553], [-122.371, -73.6679], [-115.244, -74.1034], [-114.23, -73.8631], [-110.55, -74.2421],
  [-108.406, -75.2553], [-106.554, -74.73], [-101.788, -75.0881], [-101.188, -74.71], [-102.46, -74.5134], [-101.357, -74.1916],
  [-102.957, -73.5947], [-101.342, -73.6374], [-103.308, -73.1476], [-103.413, -72.2518], [-98.73, -71.7199], [-96.238, -71.8461],
  [-94.183, -72.5242], [-90.707, -72.7522], [-89.856, -72.4497], [-82.204, -73.8405], [-80.536, -72.9539], [-78.723, -73.4008],
  [-77.382, -73.2906], [-77.369, -72.9872], [-79.317, -72.9521], [-78.946, -72.2948], [-72.21, -73.1368], [-72.645, -72.2698],
  [-75.442, -71.8465], [-73.317, -71.295], [-73.302, -70.8396], [-76.49, -71.0673], [-73.922, -70.522], [-75.809, -69.9246],
  [-74.825, -69.7226], [-73.489, -70.2977], [-72.862, -69.431], [-71.728, -69.6428], [-72.014, -68.9545], [-70.46, -68.7805],
  [-68.436, -70.0847], [-68.843, -69.4028], [-67.121, -69.2565], [-66.616, -68.2405], [-67.494, -67.0417], [-66.46, -67.3369],
  [-66.544, -66.6429], [-65.675, -66.6863], [-65.723, -66.1448], [-63.789, -65.6034], [-63.833, -65.0303], [-63.087, -65.168],
  [-60.99, -64.0403], [-57.298, -63.2129], [-56.77, -63.606], [-58.274, -63.7372], [-58.809, -64.5513], [-59.474, -64.3094],
  [-60.742, -64.746], [-59.453, -65.2472], [-61.22, -65.0044], [-62.203, -65.3242], [-60.534, -65.9593], [-60.045, -66.8174],
  [-61.508, -69.4018], [-59.659, -72.4781], [-61.656, -74.8418], [-52.742, -76.8435], [-50.378, -76.9917], [-48.713, -77.7724],
  [-36.036, -78.2405], [-33.714, -77.3058], [-26.646, -76.0967], [-27.661, -75.609], [-25.244, -75.1448], [-25.666, -74.0909],
  [-23.88, -73.8521], [-22.082, -74.2691], [-19.152, -72.6692], [-12.92, -72.0136], [-12.252, -71.3085], [-8.749, -70.5283],
  [-3.431, -70.4489], [-1.7155, -70.0162], [-0.8577, -69.8012], [0.0, -69.5863], [0.2772, -70.1231], [0.8713, -69.8942], [4.054, -70.2806],
  [5.7143, -69.9956], [7.4076, -70.2082], [8.4571, -69.8673], [12.7879, -70.0732], [14.819, -69.4624], [15.573, -69.841],
  [17.7339, -69.652], [19.0475, -69.9966], [19.9498, -69.7207], [24.6274, -70.3472], [33.6037, -68.6541], [37.0573, -69.6832],
  [38.0549, -69.282], [37.2629, -69.7146], [38.8313, -70.0565], [39.8517, -68.828], [46.2586, -67.6539], [46.4339, -67.2816],
  [48.4537, -67.5817], [49.2567, -67.3608], [48.3398, -67.0258], [49.1966, -66.8217], [50.8895, -67.2159], [50.4179, -66.3355],
  [53.7781, -65.841], [55.6169, -66.0227], [57.2427, -66.5434], [56.9287, -67.0441], [59.0992, -67.4397], [69.5817, -67.7413],
  [70.3624, -68.7043], [72.7323, -68.6391], [74.2621, -69.3384], [73.6879, -69.7509], [74.5305, -69.7573], [77.8902, -69.0609],
  [78.7831, -68.2443], [81.2581, -67.8289], [81.7131, -67.2966], [84.5107, -66.9641], [84.044, -66.6701], [86.0424, -66.3003],
  [89.5628, -66.8496], [91.9799, -66.4995], [94.3357, -66.646], [95.5721, -66.1718], [95.5313, -65.2593], [96.3815, -64.9744],
  [97.9674, -65.7338], [100.233, -65.7373], [100.926, -65.3847], [102.534, -65.8857], [102.909, -65.1271], [103.987, -65.5442],
  [103.886, -65.9747], [108.775, -66.9462], [110.734, -66.4787], [110.931, -66.0481], [113.225, -65.7641], [114.467, -66.4593],
  [117.61, -67.0054], [120.682, -66.9169], [122.047, -66.5307], [122.585, -66.8142], [124.328, -66.5126], [124.877, -66.7163],
  [126.274, -66.2398], [128.783, -67.0357], [130.375, -66.1164], [134.656, -65.9895], [142.486, -67.0281], [143.53, -66.8449],
  [144.715, -67.2303], [146.218, -66.6727], [145.266, -67.5143], [147.724, -68.338], [150.926, -68.3417], [151.444, -68.6394],
  [153.876, -68.2765], [155.423, -68.9501], [159.779, -69.5046], [161.078, -70.3058], [166.719, -70.6123], [170.186, -71.6758],
  [170.275, -71.2936], [170.973, -71.8363], [169.76, -72.2228], [169.18, -73.4518], [166.599, -73.5911], [166.273, -74.0835],
  [164.822, -74.0847], [165.432, -74.6416], [164.059, -74.6268], [162.58, -75.2285], [165.832, -75.4695], [162.515, -75.4963],
  [162.285, -76.956], [164.425, -78.0811], [166.85, -77.8896], [166.69, -77.1617], [180.0, -77.8188], [180.0, -89.9999], [-180.0, -89.9999],
  [-180.0, -77.8188]]]

CONTINENTS: Dict[str, List[Tuple[float, float]]] = {
    f"coast_{i:03d}": [(float(lon), float(lat)) for lon, lat in poly]
    for i, poly in enumerate(ACCURATE_LAND_POLYGONS)
}




# =============================================================================
# 3D projection
# =============================================================================

def project_lonlat(lon: float, lat: float, center_lon: float, center_lat: float) -> Tuple[float, float, float]:
    lam = math.radians(lon - center_lon)
    phi = math.radians(lat)
    phi0 = math.radians(center_lat)
    cp = math.cos(phi)
    x = cp * math.sin(lam)
    y = math.cos(phi0) * math.sin(phi) - math.sin(phi0) * cp * math.cos(lam)
    z = math.sin(phi0) * math.sin(phi) + math.cos(phi0) * cp * math.cos(lam)
    return x, y, z


def sphere_screen(x: float, y: float, cx: float, cy: float, radius: float) -> Tuple[int, int]:
    return int(cx + x * radius), int(cy - y * radius)


# =============================================================================
# Renderer
# =============================================================================

class CableEarthScene:
    def __init__(self, snapshot: CableSnapshot, routes: List[Dict[str, Any]], landings: List[Dict[str, Any]], summary: Dict[str, Any]):
        self.snapshot = snapshot
        self.routes = routes
        self.landings = landings
        self.summary = summary
        rng = np.random.default_rng(20260921)
        self.stars = [
            (float(rng.uniform(0, W)), float(rng.uniform(0, H)), float(rng.uniform(.4, 2.1) * SCALE), int(rng.uniform(12, 76)), float(rng.uniform(0, math.tau)))
            for _ in range(170 if QUICK_MODE else 430)
        ]
        self.palette = [COLORS["cyan"], COLORS["blue"], COLORS["green"], COLORS["violet"], COLORS["magenta"], COLORS["gold"]]
        self.base_bg = self._build_background()

    def _build_background(self) -> Image.Image:
        top = np.array(COLORS["space"], dtype=np.float32)
        bottom = np.array(COLORS["space2"], dtype=np.float32)
        f = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        rgb = (top[None, :] * (1-f) + bottom[None, :] * f).astype(np.uint8)
        arr = np.empty((H, W, 4), dtype=np.uint8)
        arr[..., :3] = rgb[:, None, :]
        arr[..., 3] = 255
        return Image.fromarray(arr, "RGBA")

    def background(self, t: float) -> Image.Image:
        img = self.base_bg.copy()
        d = ImageDraw.Draw(img)
        for x, y, r, a, phase in self.stars:
            aa = int(a * (0.75 + 0.25 * math.sin(t*0.9 + phase)))
            d.ellipse((x-r, y-r, x+r, y+r), fill=COLORS["white"] + (aa,))
        return img

    def camera(self, t: float, shot: Dict[str, Any], local: float) -> Tuple[float, float, float, float, float]:
        # returns center_lon, center_lat, cx, cy, radius
        name = shot["name"]
        if name == "atlantic":
            lon = lerp(-52, -18, smoothstep(local))
            lat = 18 + 7 * math.sin(local * math.pi)
            r = lerp(360, 405, smoothstep(local)) * SCALE
        elif name == "landings":
            lon = lerp(15, 105, smoothstep(local))
            lat = 12
            r = 378 * SCALE
        elif name == "fragility":
            lon = lerp(112, 145, local)
            lat = -3
            r = 330 * SCALE
        elif name == "outro":
            lon = lerp(145, 225, local)
            lat = lerp(-2, 18, local)
            r = lerp(330, 300, smoothstep(local)) * SCALE
        else:
            lon = -35 + 75 * (t / max(DURATION, 1)) + 17 * math.sin(t * .09)
            lat = 12 + 7 * math.sin(t * .13)
            r = (325 + 36 * smoothstep(local)) * SCALE
        return lon, lat, W * 0.5, H * 0.435, r

    def draw_globe_base(self, image: Image.Image, center_lon: float, center_lat: float, cx: float, cy: float, r: float, t: float):
        layer = Image.new("RGBA", SIZE, (0,0,0,0))
        d = ImageDraw.Draw(layer)
        # Atmosphere glow
        for extra, alpha in [(34, 14), (20, 24), (10, 44)]:
            rr = r + extra * SCALE
            d.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=COLORS["cyan"] + (alpha,))
        # Ocean sphere
        d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=COLORS["ocean"] + (255,), outline=COLORS["cyan"] + (80,), width=max(1,int(2*SCALE)))
        # Directional shading bands clipped approximately by nested circles.
        for k in range(12):
            frac = k / 11
            rr = r * (1 - frac * .82)
            xoff = -r * .18 + frac * r * .42
            alpha = int(8 + 13 * frac)
            d.ellipse((cx-rr+xoff, cy-rr, cx+rr+xoff, cy+rr), fill=COLORS["ocean_lit"] + (alpha,))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1,int(2*SCALE)))))

        # Graticule
        gd = ImageDraw.Draw(image)
        for lat in range(-60, 61, 30):
            pts = []
            for lon in np.linspace(-180, 180, 145):
                x,y,z = project_lonlat(float(lon), float(lat), center_lon, center_lat)
                if z >= 0:
                    pts.append(sphere_screen(x,y,cx,cy,r))
                elif len(pts) > 1:
                    gd.line(pts, fill=COLORS["cyan"]+(22,), width=max(1,int(SCALE)))
                    pts=[]
            if len(pts)>1:
                gd.line(pts, fill=COLORS["cyan"]+(22,), width=max(1,int(SCALE)))
        for lon in range(-180, 180, 30):
            pts=[]
            for lat in np.linspace(-88,88,120):
                x,y,z=project_lonlat(float(lon),float(lat),center_lon,center_lat)
                if z>=0:
                    pts.append(sphere_screen(x,y,cx,cy,r))
                elif len(pts)>1:
                    gd.line(pts, fill=COLORS["cyan"]+(18,), width=max(1,int(SCALE)))
                    pts=[]
            if len(pts)>1:
                gd.line(pts, fill=COLORS["cyan"]+(18,), width=max(1,int(SCALE)))

    def draw_continents(self, image: Image.Image, center_lon: float, center_lat: float, cx: float, cy: float, r: float):
        # Accurate bundled coastline geometry; coarse built-ins remain an offline fallback.
        d = ImageDraw.Draw(image)
        for poly in CONTINENTS.values():
            segment=[]
            previous_lon=None
            for lon,lat in poly:
                if previous_lon is not None and abs(lon-previous_lon) > 180.0:
                    if len(segment)>1:
                        d.line(segment, fill=COLORS["land_edge"]+(135,), width=max(1,int(3*SCALE)))
                    segment=[]
                previous_lon=lon
                x,y,z=project_lonlat(lon,lat,center_lon,center_lat)
                if z>=0.01:
                    segment.append(sphere_screen(x,y,cx,cy,r))
                elif len(segment)>1:
                    d.line(segment, fill=COLORS["land_edge"]+(135,), width=max(1,int(3*SCALE)))
                    segment=[]
            if len(segment)>1:
                d.line(segment, fill=COLORS["land_edge"]+(135,), width=max(1,int(3*SCALE)))

    def draw_cables(self, image: Image.Image, center_lon: float, center_lat: float, cx: float, cy: float, r: float, t: float, reveal: float = 1.0, packets: bool = False):
        glow = Image.new("RGBA", SIZE, (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        sharp = Image.new("RGBA", SIZE, (0,0,0,0))
        sd = ImageDraw.Draw(sharp)
        count = max(1, int(len(self.routes) * clamp(reveal)))
        for i, route in enumerate(self.routes[:count]):
            color = self.palette[i % len(self.palette)]
            coords = route["coords"]
            visible_pts: List[Tuple[int,int,float]] = []
            for lon,lat in coords:
                x,y,z = project_lonlat(lon,lat,center_lon,center_lat)
                if z >= 0.0:
                    px,py=sphere_screen(x,y,cx,cy,r*1.006)
                    visible_pts.append((px,py,z))
                else:
                    if len(visible_pts)>1:
                        self._draw_cable_segment(gd, sd, visible_pts, color)
                    visible_pts=[]
            if len(visible_pts)>1:
                self._draw_cable_segment(gd, sd, visible_pts, color)

            if packets and len(coords) >= 2 and (i % (4 if QUICK_MODE else 7) == 0):
                phase=(t*.22 + i*.137)%1.0
                k=min(len(coords)-1,int(phase*(len(coords)-1)))
                lon,lat=coords[k]
                x,y,z=project_lonlat(lon,lat,center_lon,center_lat)
                if z>0.02:
                    px,py=sphere_screen(x,y,cx,cy,r*1.012)
                    rr=max(2,int((3.0+2.0*z)*SCALE))
                    sd.ellipse((px-rr,py-rr,px+rr,py+rr),fill=COLORS["white"]+(235,))
                    gd.ellipse((px-rr*3,py-rr*3,px+rr*3,py+rr*3),fill=color+(80,))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(2,int(8*SCALE)))))
        image.alpha_composite(sharp)

    @staticmethod
    def _draw_cable_segment(gd: ImageDraw.ImageDraw, sd: ImageDraw.ImageDraw, pts: List[Tuple[int,int,float]], color: Tuple[int,int,int]):
        xy=[(p[0],p[1]) for p in pts]
        mean_z=float(np.mean([p[2] for p in pts]))
        alpha=int(95+145*clamp(mean_z))
        gd.line(xy,fill=color+(90,),width=max(2,int(7*SCALE)),joint="curve")
        sd.line(xy,fill=color+(alpha,),width=max(1,int(2.2*SCALE)),joint="curve")

    def draw_landings(self, image: Image.Image, center_lon: float, center_lat: float, cx: float, cy: float, r: float, t: float, density: float = 1.0):
        layer=Image.new("RGBA",SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        n=max(1,int(len(self.landings)*clamp(density)))
        for i,p in enumerate(self.landings[:n]):
            x,y,z=project_lonlat(float(p["lon"]),float(p["lat"]),center_lon,center_lat)
            if z<=0.02: continue
            px,py=sphere_screen(x,y,cx,cy,r*1.012)
            pulse=.5+.5*math.sin(t*3.2+i*.71)
            rr=(2.2+2.8*pulse)*SCALE
            d.ellipse((px-rr,py-rr,px+rr,py+rr),fill=COLORS["gold"]+(int(110+120*z),))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1,int(2*SCALE)))))

    def draw_title(self, image: Image.Image, t: float):
        intro_end=SHOT_PLAN[0]["end"]
        if t < intro_end:
            fade=smoothstep((t-.1)/(.9 if not QUICK_MODE else .20))
            draw_text(image,"EARTH'S SUBMARINE",(W//2,int(H*.065)),37 if not QUICK_MODE else 18,COLORS["white"]+(int(245*fade),),True,"ma",2)
            draw_text(image,"CABLES IN 3D",(W//2,int(H*.111)),54 if not QUICK_MODE else 27,COLORS["cyan"]+(int(250*fade),),True,"ma",2)
            draw_text(image,CONFIG["subtitle"],(W//2,int(H*.151)),17 if not QUICK_MODE else 8,COLORS["muted"]+(int(230*fade),),False,"ma",1)
        else:
            labels={
                "backbone":"THE INTERNET'S PHYSICAL BACKBONE",
                "atlantic":"OCEAN-SPANNING FIBER ROUTES",
                "landings":"WHERE THE OCEAN MEETS THE NETWORK",
                "fragility":"GLOBAL — BUT NOT INVULNERABLE",
                "outro":"THE INTERNET HAS A GEOGRAPHY",
                "reveal":"EARTH'S SUBMARINE CABLES",
            }
            draw_text(image,labels[get_shot(t)["name"]],(int(W*.055),int(H*.050)),16 if not QUICK_MODE else 8,COLORS["muted"]+(225,),True,"la",1)

    def draw_source_hud(self, image: Image.Image):
        status="OFFLINE REPRESENTATIVE ROUTES" if self.snapshot.offline_fallback else ("CACHED PUBLIC GEOJSON" if self.snapshot.data_status=="cache" else "PUBLIC GEOJSON")
        draw_text(image,status,(int(W*.95),int(H*.052)),12 if not QUICK_MODE else 6,COLORS["cyan"]+(210,),True,"ra",1)
        draw_text(image,f"ROUTES LOADED: {self.snapshot.route_feature_count:,}",(int(W*.95),int(H*.075)),11 if not QUICK_MODE else 5,COLORS["muted"]+(205,),False,"ra",1)

    def draw_caption(self, image: Image.Image, t: float):
        caption=caption_at(t)
        if not caption: return
        y0=H-int(190*SCALE)
        layer=Image.new("RGBA",SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        d.rounded_rectangle((int(58*SCALE),y0,W-int(58*SCALE),y0+int(116*SCALE)),radius=max(8,int(20*SCALE)),fill=(2,7,17,150),outline=COLORS["cyan"]+(48,),width=max(1,int(SCALE)))
        image.alpha_composite(layer)
        draw_wrapped_text(image,caption,(int(82*SCALE),y0+int(18*SCALE)),W-int(164*SCALE),30,COLORS["white"]+(240,))

    def draw_backbone_stats(self, image: Image.Image, local: float):
        a=smoothstep(local/.35)
        x=int(W*.07); y=int(H*.19)
        draw_text(image,">99%",(x,y),78 if not QUICK_MODE else 39,COLORS["cyan"]+(int(250*a),),True,"la",2)
        draw_text(image,"OF INTERNATIONAL DATA TRAFFIC",(x,y+int(72*SCALE)),18 if not QUICK_MODE else 9,COLORS["white"]+(int(235*a),),True,"la",1)
        draw_text(image,"travels through submarine cables",(x,y+int(102*SCALE)),17 if not QUICK_MODE else 8,COLORS["muted"]+(int(220*a),),False,"la",1)

    def draw_atlantic_callout(self, image: Image.Image, local: float):
        draw_text(image,"TRANSOCEANIC CORRIDORS",(W//2,int(H*.185)),19 if not QUICK_MODE else 9,COLORS["white"]+(235,),True,"ma",1)
        draw_text(image,"route geometry enlarged for visibility",(W//2,int(H*.215)),13 if not QUICK_MODE else 6,COLORS["muted"]+(205,),False,"ma",1)
        x0=int(W*.10); y=int(H*.71); x1=int(W*.90)
        d=ImageDraw.Draw(image)
        d.line((x0,y,x1,y),fill=COLORS["cyan"]+(90,),width=max(1,int(2*SCALE)))
        p=lerp(x0,x1,(local*1.8)%1.0)
        rr=max(3,int(6*SCALE))
        d.ellipse((p-rr,y-rr,p+rr,y+rr),fill=COLORS["white"]+(245,))
        draw_text(image,"LIGHT IN GLASS",(W//2,y+int(36*SCALE)),15 if not QUICK_MODE else 7,COLORS["cyan"]+(225,),True,"ma",1)

    def draw_landing_callout(self, image: Image.Image, local: float):
        draw_text(image,"LANDING STATIONS",(int(W*.07),int(H*.185)),30 if not QUICK_MODE else 15,COLORS["gold"]+(245,),True,"la",2)
        draw_text(image,"ocean cable → terrestrial network",(int(W*.07),int(H*.225)),16 if not QUICK_MODE else 8,COLORS["white"]+(230,),False,"la",1)
        draw_text(image,f"2026 MAP CONTEXT: {TELEGEOGRAPHY_2026_LANDINGS:,} LANDINGS",(int(W*.07),int(H*.255)),13 if not QUICK_MODE else 6,COLORS["muted"]+(215,),True,"la",1)

    def draw_cable_cross_section(self, image: Image.Image, local: float):
        # Stylized cable cross-section; sizes are not to scale.
        cx=int(W*.50); cy=int(H*.425); base=int(125*SCALE)
        d=ImageDraw.Draw(image)
        for frac,color in [(1.0,(40,52,61)),(.82,(164,171,176)),(.64,(233,205,118)),(.47,(93,110,120)),(.30,(235,244,248))]:
            rr=base*frac
            d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=color+(245,),outline=COLORS["white"]+(40,),width=max(1,int(2*SCALE)))
        for ang in np.linspace(0,math.tau,8,endpoint=False):
            rr=base*.16
            fx=cx+math.cos(ang)*base*.16; fy=cy+math.sin(ang)*base*.16
            q=base*.045
            d.ellipse((fx-q,fy-q,fx+q,fy+q),fill=COLORS["cyan"]+(245,))
        # damage flash
        if local>.53:
            flash=smoothstep((local-.53)/.12)*(1-smoothstep((local-.78)/.18))
            for k in range(5):
                ang=-.9+k*.36
                x0=cx+math.cos(ang)*base*.95; y0=cy+math.sin(ang)*base*.95
                x1=cx+math.cos(ang)*base*1.45; y1=cy+math.sin(ang)*base*1.45
                d.line((x0,y0,x1,y1),fill=COLORS["red"]+(int(220*flash),),width=max(2,int(5*SCALE)))
        draw_text(image,"FIBER-OPTIC CABLE",(cx,int(H*.235)),23 if not QUICK_MODE else 11,COLORS["white"]+(240,),True,"ma",1)
        draw_text(image,"deep-ocean cable: roughly garden-hose width",(cx,int(H*.255)),13 if not QUICK_MODE else 6,COLORS["muted"]+(220,),False,"ma",1)
        draw_text(image,">200 REPAIRS REPORTED WORLDWIDE IN 2023",(cx,int(H*.625)),16 if not QUICK_MODE else 8,COLORS["red"]+(235,),True,"ma",1)
        draw_text(image,"ITU / ICPC context",(cx,int(H*.652)),12 if not QUICK_MODE else 6,COLORS["muted"]+(210,),False,"ma",1)

    def draw_outro_stats(self, image: Image.Image, local: float):
        alpha=int(245*smoothstep(local/.30))
        y=int(H*.17)
        draw_text(image,"TELEGEOGRAPHY 2026 MAP",(W//2,y),18 if not QUICK_MODE else 9,COLORS["muted"]+(alpha,),True,"ma",1)
        draw_text(image,f"{TELEGEOGRAPHY_2026_SYSTEMS}",(int(W*.32),y+int(70*SCALE)),58 if not QUICK_MODE else 29,COLORS["cyan"]+(alpha,),True,"ma",2)
        draw_text(image,"CABLE SYSTEMS",(int(W*.32),y+int(116*SCALE)),13 if not QUICK_MODE else 6,COLORS["white"]+(alpha,),True,"ma",1)
        draw_text(image,f"{TELEGEOGRAPHY_2026_LANDINGS:,}",(int(W*.70),y+int(70*SCALE)),50 if not QUICK_MODE else 25,COLORS["gold"]+(alpha,),True,"ma",2)
        draw_text(image,"LANDING STATIONS",(int(W*.70),y+int(116*SCALE)),13 if not QUICK_MODE else 6,COLORS["white"]+(alpha,),True,"ma",1)
        draw_text(image,"ACTIVE + PLANNED SYSTEMS",(W//2,int(H*.315)),14 if not QUICK_MODE else 7,COLORS["muted"]+(alpha,),True,"ma",1)

    def render(self, t: float) -> np.ndarray:
        shot=get_shot(t); local=local_progress(t,shot)
        image=self.background(t)
        lon,lat,cx,cy,r=self.camera(t,shot,local)

        if shot["name"] != "fragility":
            self.draw_globe_base(image,lon,lat,cx,cy,r,t)
            self.draw_continents(image,lon,lat,cx,cy,r)
            if shot["name"]=="reveal":
                reveal=smoothstep(local)
                self.draw_cables(image,lon,lat,cx,cy,r,t,reveal=reveal,packets=local>.45)
                self.draw_landings(image,lon,lat,cx,cy,r,t,density=smoothstep((local-.35)/.45))
            elif shot["name"]=="backbone":
                self.draw_cables(image,lon,lat,cx,cy,r,t,1.0,True)
                self.draw_landings(image,lon,lat,cx,cy,r,t,.55)
                self.draw_backbone_stats(image,local)
            elif shot["name"]=="atlantic":
                self.draw_cables(image,lon,lat,cx,cy,r,t,1.0,True)
                self.draw_landings(image,lon,lat,cx,cy,r,t,.75)
                self.draw_atlantic_callout(image,local)
            elif shot["name"]=="landings":
                self.draw_cables(image,lon,lat,cx,cy,r,t,1.0,True)
                self.draw_landings(image,lon,lat,cx,cy,r,t,1.0)
                self.draw_landing_callout(image,local)
            elif shot["name"]=="outro":
                self.draw_cables(image,lon,lat,cx,cy,r,t,1.0,True)
                self.draw_landings(image,lon,lat,cx,cy,r,t,1.0)
                self.draw_outro_stats(image,local)
        else:
            # Keep a dim globe behind the cross-section.
            dim=image.copy()
            self.draw_globe_base(dim,lon,lat,cx,cy,r,t)
            self.draw_continents(dim,lon,lat,cx,cy,r)
            self.draw_cables(dim,lon,lat,cx,cy,r,t,.65,False)
            dim=ImageEnhance.Brightness(dim).enhance(.47)
            image=dim
            self.draw_cable_cross_section(image,local)

        self.draw_title(image,t)
        self.draw_source_hud(image)
        self.draw_caption(image,t)

        # Scanline / HUD texture
        overlay=Image.new("RGBA",SIZE,(0,0,0,0)); od=ImageDraw.Draw(overlay)
        off=int((t*31)%9)
        for yy in range(off,H,9):
            od.line((0,yy,W,yy),fill=(100,215,245,7),width=1)
        image.alpha_composite(overlay)

        arr=np.asarray(image.convert("RGB")).astype(np.float32)
        arr*=VIGNETTE[...,None]
        arr=np.clip(arr,0,255).astype(np.uint8)
        graded=Image.fromarray(arr)
        graded=ImageEnhance.Contrast(graded).enhance(float(CONFIG["contrast"]))
        graded=ImageEnhance.Color(graded).enhance(float(CONFIG["saturation"]))
        return np.asarray(graded)


# =============================================================================
# Output helpers
# =============================================================================

def save_previews(scene: CableEarthScene) -> List[Path]:
    times=[]
    for shot in SHOT_PLAN:
        times.append(shot["start"] + .52*(shot["end"]-shot["start"]))
    paths=[]
    for idx,t in enumerate(times):
        frame=scene.render(t)
        path=PREVIEW_ROOT/f"preview_{idx+1:02d}_{get_shot(t)['name']}.jpg"
        Image.fromarray(frame).save(path,quality=91)
        paths.append(path)
    return paths


def make_contact_sheet(paths: Sequence[Path]) -> Path:
    thumbs=[]
    tw=270 if not QUICK_MODE else 216
    th=int(tw*16/9)
    for p in paths:
        im=Image.open(p).convert("RGB")
        im.thumbnail((tw,th),Image.Resampling.LANCZOS)
        canvas=Image.new("RGB",(tw,th),(3,8,18))
        canvas.paste(im,((tw-im.width)//2,(th-im.height)//2))
        thumbs.append(canvas)
    cols=3; rows=math.ceil(len(thumbs)/cols)
    sheet=Image.new("RGB",(cols*tw,rows*th),(3,8,18))
    for i,im in enumerate(thumbs):
        sheet.paste(im,((i%cols)*tw,(i//cols)*th))
    path=PREVIEW_ROOT/f"{CONFIG['basename']}_contact_sheet.jpg"
    sheet.save(path,quality=91)
    return path


def render_video(scene: CableEarthScene) -> Path:
    out=OUTPUT_ROOT/(CONFIG["basename"] + ("_quick_preview.mp4" if QUICK_MODE else "_4k.mp4" if FOUR_K else ".mp4"))
    total=int(round(DURATION*FPS))
    writer=iio.get_writer(
        out,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=2,
        ffmpeg_params=["-pix_fmt","yuv420p","-movflags","+faststart"],
    )
    try:
        for i in tqdm(range(total),desc="Rendering submarine cable short"):
            writer.append_data(scene.render(i/FPS))
    finally:
        writer.close()
    return out


def export_top_level(paths: Sequence[Path]):
    # Convenience copies beside the script when running in a sandbox/notebook.
    for path in paths:
        if not path.exists():
            continue
        target=Path.cwd()/path.name
        try:
            if path.resolve()!=target.resolve():
                shutil.copy2(path,target)
        except Exception:
            pass



# =============================================================================
# YouTube title / description sidecar
# =============================================================================

YOUTUBE_TITLE = "Earth's Submarine Internet Cables in 3D 🌍🌐 #rootjatin"
YOUTUBE_DESCRIPTION = 'A 3D look at the submarine fiber-optic cables that connect continents and carry most international data traffic. The renderer uses public cable-route geometry when available and shows landing points, major ocean corridors, redundancy, and the physical geography behind the global internet. Cable glow, depth, packet motion, and line thickness are illustrative; the route geometry is data-grounded when the public dataset is available.'
YOUTUBE_HASHTAGS = '#rootjatin #SubmarineCables #Internet #Technology #Data #Earth #Geography #Science'

def write_youtube_metadata_txt() -> Path:
    cfg = globals().get("CONFIG") or globals().get("CFG") or {}
    basename = (
        cfg.get("basename")
        or cfg.get("output_basename")
        or cfg.get("file_stem")
        or Path(__file__).stem
    )
    root = globals().get("OUTPUT_ROOT") or globals().get("ROOT") or Path(".")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{basename}_title_description.txt"
    path.write_text(
        "TITLE\n" + YOUTUBE_TITLE +
        "\n\nDESCRIPTION\n" + YOUTUBE_DESCRIPTION +
        "\n\nHASHTAGS\n" + YOUTUBE_HASHTAGS + "\n",
        encoding="utf-8",
    )
    return path

def main():
    snapshot,routes,landings,summary=collect_data()
    csv_path,json_path,note_path=save_data(snapshot,routes,landings,summary)
    srt_path=write_srt(OUTPUT_ROOT/f"{CONFIG['basename']}_subtitles.srt")
    scene=CableEarthScene(snapshot,routes,landings,summary)
    previews=save_previews(scene)
    contact=make_contact_sheet(previews)
    video=render_video(scene)

    export_top_level([csv_path,json_path,note_path,srt_path,contact,video])
    print("\nRender complete")
    print("data status:",snapshot.data_status)
    print("routes loaded:",snapshot.route_feature_count)
    print("landings loaded:",snapshot.landing_point_count_loaded)
    print("video:",video)
    print("contact sheet:",contact)
    print("subtitles:",srt_path)
    print("route summary:",csv_path)
    metadata_txt = write_youtube_metadata_txt()
    print("Title/description TXT:", metadata_txt.resolve())

if __name__ == "__main__":
    main()
