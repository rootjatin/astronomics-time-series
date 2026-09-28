from __future__ import annotations

"""
Watch Halley's Comet Travel Through Time
=======================================

A cinematic vertical YouTube Short renderer about 1P/Halley, combining a
current orbit snapshot from NASA/JPL's Small-Body Database with a historical
apparition timeline.

What the video shows
--------------------
- Halley's current published osculating orbit from JPL SBDB.
- A historical timeline of well-known recorded Halley apparitions from
  240 BCE through the predicted 2061 return.
- A top-down solar-system view of Halley's highly elongated retrograde orbit.
- Key facts such as eccentricity, perihelion distance, inclination, period,
  and next perihelion.
- Famous moments including 1066, Halley's 1705 prediction, the 1759 return,
  the bright 1910 apparition, the 1986 spacecraft encounter, and the next
  predicted return in 2061.

Official live source
--------------------
NASA/JPL Small-Body Database API:
    https://ssd-api.jpl.nasa.gov/sbdb.api
    https://ssd-api.jpl.nasa.gov/doc/sbdb.html

Honesty / interpretation rules
------------------------------
- The orbital curve is reconstructed from JPL's published osculating elements.
  It is not a full n-body ephemeris or a spacecraft-tracking product.
- The historical apparition list is a compiled educational timeline of Halley's
  major recorded returns. It is included explicitly in this script so the video
  can render consistently even when a live history feed is unavailable.
- The animated comet position is a stylized two-body propagation anchored to
  the current JPL orbital solution and the published perihelion time.
- BCE labels are shown in the familiar historical style, while the internal
  timeline uses a simple numeric axis for animation.

Offline fallback
----------------
If JPL cannot be reached, the script uses a clearly labeled fixture with
approximate modern Halley orbit values and the same historical timeline.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg requests tqdm

Run final quality
-----------------
    python watch_halleys_comet_travel_through_time_short.py

Run quick preview
-----------------
    HALLEY_SHORT_QUICK=1 python watch_halleys_comet_travel_through_time_short.py

Force live refresh
------------------
    HALLEY_SHORT_REFRESH=1 python watch_halleys_comet_travel_through_time_short.py

Force offline layout testing
----------------------------
    HALLEY_SHORT_OFFLINE=1 HALLEY_SHORT_QUICK=1 \
        python watch_halleys_comet_travel_through_time_short.py
"""

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
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

QUICK_MODE = os.environ.get("HALLEY_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("HALLEY_SHORT_OFFLINE", "0") == "1"
REFRESH = os.environ.get("HALLEY_SHORT_REFRESH", "0") == "1"

OUTPUT_ROOT = Path("watch_halleys_comet_travel_through_time_output")
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
    "basename": "watch_halleys_comet_travel_through_time",
    "title": "WATCH HALLEY'S COMET TRAVEL THROUGH TIME",
    "subtitle": "A 240 BCE → 2061 journey using NASA/JPL orbit data",
    "timeout_s": 35,
    "stars": 580 if QUICK_MODE else 1100,
    "cache_hours": 36,
    "api_url": "https://ssd-api.jpl.nasa.gov/sbdb.api",
    "api_docs": "https://ssd-api.jpl.nasa.gov/doc/sbdb.html",
    "source_url": "https://science.nasa.gov/solar-system/comets/1p-halley/",
}

W = CONFIG["width"]
H = CONFIG["height"]
SIZE = (W, H)
SCALE = W / 1080.0

COLORS = {
    "bg": (4, 8, 16),
    "white": (246, 249, 255),
    "muted": (150, 198, 222),
    "cyan": (92, 223, 255),
    "gold": (255, 197, 92),
    "violet": (201, 116, 255),
    "green": (104, 255, 181),
    "red": (255, 115, 125),
    "sun": (255, 205, 79),
    "earth": (102, 174, 255),
    "orbit": (130, 200, 235),
    "halley": (255, 240, 215),
    "trail": (141, 230, 255),
}

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.0 if not QUICK_MODE else 1.8},
    {"name": "timeline", "start": 7.0 if not QUICK_MODE else 1.8, "end": 18.0 if not QUICK_MODE else 4.0},
    {"name": "orbit", "start": 18.0 if not QUICK_MODE else 4.0, "end": 31.0 if not QUICK_MODE else 6.6},
    {"name": "highlights", "start": 31.0 if not QUICK_MODE else 6.6, "end": 43.0 if not QUICK_MODE else 8.8},
    {"name": "stats", "start": 43.0 if not QUICK_MODE else 8.8, "end": 53.0 if not QUICK_MODE else 10.7},
    {"name": "outro", "start": 53.0 if not QUICK_MODE else 10.7, "end": CONFIG["duration_s"]},
]

CAPTIONS = [
    (0.4, 6.9, "Halley's Comet is one of the most famous objects in the sky, returning roughly every 75 years."),
    (7.0, 18.0, "This timeline follows Halley from the first confirmed historical return in 240 BCE to its predicted 2061 comeback."),
    (18.1, 31.0, "Its orbit is huge, steeply tilted, and retrograde—meaning it moves around the Sun opposite the planets."),
    (31.1, 43.0, "Some returns became legends: 1066, Halley's prediction in 1705, the successful 1759 return, the bright 1910 apparition, and the 1986 spacecraft encounter."),
    (43.1, 53.0, "The numbers here come from JPL's published orbital solution: eccentricity, perihelion, semimajor axis, inclination, and next perihelion."),
    (53.1, 57.7, "This is a data-grounded educational reconstruction of Halley's path through time—not a live telescope feed or a full dynamical simulation."),
]

PLANETS = [
    ("MERCURY", 0.387, (180, 188, 198)),
    ("VENUS", 0.723, (232, 197, 130)),
    ("EARTH", 1.000, COLORS["earth"]),
    ("MARS", 1.524, (231, 126, 87)),
    ("JUPITER", 5.203, (232, 185, 124)),
    ("SATURN", 9.537, (225, 204, 142)),
    ("URANUS", 19.191, (116, 218, 230)),
    ("NEPTUNE", 30.069, (93, 123, 255)),
]


# =============================================================================
# Data model
# =============================================================================

@dataclass
class HalleySnapshot:
    object_name: str
    fetched_at_utc: str
    source_url: str
    source_kind: str
    offline_fixture: bool
    data_status: str
    epoch_jd: float
    epoch_cd: str
    eccentricity: float
    perihelion_au: float
    semimajor_axis_au: float
    inclination_deg: float
    node_deg: float
    arg_peri_deg: float
    mean_anomaly_deg: float
    period_days: float
    perihelion_time_jd: float
    perihelion_time_cd: str
    diameter_km: float
    absolute_magnitude: float
    note: str = ""


@dataclass
class Apparition:
    numeric_year: int
    label: str
    year_text: str
    note: str
    priority: int = 0


# =============================================================================
# Utilities
# =============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def safe_int(value, default=0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


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


def julian_date(dt: datetime) -> float:
    return dt.astimezone(timezone.utc).timestamp() / 86400.0 + 2440587.5


def jd_to_datetime(jd: float) -> datetime:
    ts = (float(jd) - 2440587.5) * 86400.0
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def year_from_jd(jd: float) -> float:
    dt = jd_to_datetime(jd)
    year_start = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
    next_year = datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc)
    span = (next_year - year_start).total_seconds()
    frac = (dt - year_start).total_seconds() / span if span else 0.0
    return dt.year + frac


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
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 220))
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += bbox[3] - bbox[1] + line_spacing


def clip_text(text: str, n: int = 34) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


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


def make_vignette(width: int, height: int, strength: float = 0.25) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1 - strength * radius**1.8, 0, 1).astype(np.float32)


VIGNETTE = make_vignette(W, H)


def get_shot(t: float) -> Dict:
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


# =============================================================================
# Historical timeline
# =============================================================================


def halley_apparitions() -> List[Apparition]:
    raw = [
        (-240, "240 BCE", "240 BCE", "Earliest confirmed recorded return", 0),
        (-164, "164 BCE", "164 BCE", "Recorded historical appearance", 0),
        (-87, "87 BCE", "87 BCE", "Recorded historical appearance", 0),
        (-12, "12 BCE", "12 BCE", "Recorded historical appearance", 0),
        (66, "66", "66", "Recorded historical appearance", 0),
        (141, "141", "141", "Recorded historical appearance", 0),
        (218, "218", "218", "Recorded historical appearance", 0),
        (295, "295", "295", "Recorded historical appearance", 0),
        (374, "374", "374", "Recorded historical appearance", 0),
        (451, "451", "451", "Recorded historical appearance", 0),
        (530, "530", "530", "Recorded historical appearance", 0),
        (607, "607", "607", "Recorded historical appearance", 0),
        (684, "684", "684", "Recorded historical appearance", 0),
        (760, "760", "760", "Recorded historical appearance", 0),
        (837, "837", "837", "Very close and dramatic return", 1),
        (912, "912", "912", "Recorded historical appearance", 0),
        (989, "989", "989", "Recorded historical appearance", 0),
        (1066, "1066", "1066", "Seen during the Norman Conquest era; famously shown on the Bayeux Tapestry", 2),
        (1145, "1145", "1145", "Recorded historical appearance", 0),
        (1222, "1222", "1222", "Recorded historical appearance", 0),
        (1301, "1301", "1301", "Appearance associated with medieval skywatching traditions", 1),
        (1378, "1378", "1378", "Recorded historical appearance", 0),
        (1456, "1456", "1456", "A prominent pre-telescopic return", 1),
        (1531, "1531", "1531", "Observed by Petrus Apianus", 1),
        (1607, "1607", "1607", "Observed by Johannes Kepler's generation", 1),
        (1682, "1682", "1682", "Observed by Edmond Halley before his prediction", 2),
        (1705, "1705", "1705", "Edmond Halley publishes the prediction that the comet will return", 3),
        (1759, "1759", "1759", "Predicted return confirmed after Halley's death", 4),
        (1835, "1835", "1835", "Nineteenth-century return", 1),
        (1910, "1910", "1910", "A bright and famous modern apparition", 4),
        (1986, "1986", "1986", "Visited by multiple spacecraft including Giotto and Vega", 5),
        (2061, "2061", "2061", "Next predicted return", 5),
    ]
    return [Apparition(*row) for row in raw]


# =============================================================================
# Data collection
# =============================================================================


def find_named_value(items: Sequence[Dict], name: str) -> Tuple[float, str]:
    for item in items:
        if str(item.get("name") or "").lower() == name.lower():
            return safe_float(item.get("value")), str(item.get("value") or "")
    return np.nan, ""


def parse_sbdb_snapshot(payload: Dict) -> HalleySnapshot:
    obj = payload.get("object") or {}
    orbit = payload.get("orbit") or {}
    elements = orbit.get("elements") or []
    phys = payload.get("phys_par") or []

    def elem(name: str) -> float:
        value, _ = find_named_value(elements, name)
        return value

    def elem_text(name: str) -> str:
        for item in elements:
            if str(item.get("name") or "").lower() == name.lower():
                return str(item.get("value") or "")
        return ""

    diameter = np.nan
    absolute_h = np.nan
    if isinstance(phys, list):
        for item in phys:
            key = str(item.get("name") or "").lower()
            if key == "diameter":
                diameter = safe_float(item.get("value"))
            if key == "h":
                absolute_h = safe_float(item.get("value"))

    return HalleySnapshot(
        object_name=str(obj.get("fullname") or obj.get("des") or obj.get("name") or "1P/Halley"),
        fetched_at_utc=iso_z(utc_now()),
        source_url=CONFIG["api_url"],
        source_kind="official JPL SBDB orbit solution",
        offline_fixture=False,
        data_status="live",
        epoch_jd=safe_float(orbit.get("epoch")),
        epoch_cd=str(orbit.get("epoch_cd") or orbit.get("epoch-cal") or ""),
        eccentricity=elem("e"),
        perihelion_au=elem("q"),
        semimajor_axis_au=elem("a"),
        inclination_deg=elem("i"),
        node_deg=elem("om"),
        arg_peri_deg=elem("w"),
        mean_anomaly_deg=elem("ma"),
        period_days=elem("per"),
        perihelion_time_jd=elem("tp"),
        perihelion_time_cd=str(orbit.get("cd_tp") or elem_text("cd_tp") or ""),
        diameter_km=diameter,
        absolute_magnitude=absolute_h,
        note="Orbit values are read from the current JPL SBDB API response.",
    )



def fallback_snapshot() -> HalleySnapshot:
    return HalleySnapshot(
        object_name="1P/Halley",
        fetched_at_utc=iso_z(utc_now()),
        source_url=CONFIG["api_url"],
        source_kind="offline Halley orbit fixture",
        offline_fixture=True,
        data_status="offline-fixture",
        epoch_jd=2460800.5,
        epoch_cd="2025-Jul-01 00:00",
        eccentricity=0.96714,
        perihelion_au=0.5860,
        semimajor_axis_au=17.834,
        inclination_deg=162.26,
        node_deg=58.42,
        arg_peri_deg=111.33,
        mean_anomaly_deg=38.0,
        period_days=75.32 * 365.25,
        perihelion_time_jd=julian_date(datetime(2061, 7, 28, tzinfo=timezone.utc)),
        perihelion_time_cd="2061-Jul-28 00:00",
        diameter_km=11.0,
        absolute_magnitude=5.5,
        note="Approximate modern Halley values for offline layout testing only.",
    )



def fetch_live_snapshot() -> HalleySnapshot:
    cache_path = CACHE_ROOT / "halley_sbdb.json"
    payload: Optional[Dict] = None
    mode = "live"

    if cache_path.exists() and not REFRESH:
        age_hours = (utc_now().timestamp() - cache_path.stat().st_mtime) / 3600.0
        if age_hours <= CONFIG["cache_hours"]:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            mode = "cache"

    if payload is None:
        if requests is None:
            raise RuntimeError("requests is unavailable")
        response = requests.get(
            CONFIG["api_url"],
            params={
                "sstr": "1P/Halley",
                "cd-epoch": "1",
                "cd-tp": "1",
                "phys-par": "1",
                "full-prec": "1",
            },
            timeout=CONFIG["timeout_s"],
            headers={"User-Agent": "HalleyTravelThroughTimeShort/1.0 educational renderer"},
        )
        response.raise_for_status()
        payload = response.json()
        cache_path.write_text(json.dumps(payload), encoding="utf-8")

    snapshot = parse_sbdb_snapshot(payload)
    snapshot.data_status = mode
    snapshot.fetched_at_utc = iso_z(utc_now())
    return snapshot



def collect_data() -> Tuple[HalleySnapshot, List[Apparition], Dict]:
    apparitions = halley_apparitions()
    errors = {}
    if OFFLINE_MODE:
        snapshot = fallback_snapshot()
    else:
        try:
            snapshot = fetch_live_snapshot()
        except Exception as exc:
            errors["halley_live_fetch"] = str(exc)
            snapshot = fallback_snapshot()

    period_years = snapshot.period_days / 365.25 if np.isfinite(snapshot.period_days) else np.nan
    next_perihelion_year = year_from_jd(snapshot.perihelion_time_jd) if np.isfinite(snapshot.perihelion_time_jd) else 2061.0
    summary = {
        "generated_at_utc": iso_z(utc_now()),
        "object_name": snapshot.object_name,
        "data_status": snapshot.data_status,
        "offline_fixture": snapshot.offline_fixture,
        "apparition_count": len(apparitions),
        "first_label": apparitions[0].label,
        "last_label": apparitions[-1].label,
        "period_years": period_years,
        "next_perihelion_year": next_perihelion_year,
        "retrograde": bool(np.isfinite(snapshot.inclination_deg) and snapshot.inclination_deg > 90.0),
        "errors": errors,
        "warning": "Historical apparition years are a compiled educational timeline; live orbit values come from JPL SBDB when available.",
    }
    return snapshot, apparitions, summary



def save_data(snapshot: HalleySnapshot, apparitions: Sequence[Apparition], summary: Dict) -> Tuple[Path, Path]:
    app_df = pd.DataFrame([asdict(a) for a in apparitions])
    csv_path = DATA_ROOT / "halley_apparition_timeline.csv"
    json_path = DATA_ROOT / "halley_snapshot_summary.json"
    app_df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps({
            "summary": summary,
            "snapshot": asdict(snapshot),
            "apparitions": [asdict(a) for a in apparitions],
        }, indent=2),
        encoding="utf-8",
    )
    return csv_path, json_path


# =============================================================================
# Orbit math
# =============================================================================


def rotation_matrix(node_deg: float, inc_deg: float, arg_deg: float) -> np.ndarray:
    om = math.radians(node_deg)
    inc = math.radians(inc_deg)
    arg = math.radians(arg_deg)

    cos_om, sin_om = math.cos(om), math.sin(om)
    cos_i, sin_i = math.cos(inc), math.sin(inc)
    cos_w, sin_w = math.cos(arg), math.sin(arg)

    return np.array([
        [cos_om * cos_w - sin_om * sin_w * cos_i, -cos_om * sin_w - sin_om * cos_w * cos_i, sin_om * sin_i],
        [sin_om * cos_w + cos_om * sin_w * cos_i, -sin_om * sin_w + cos_om * cos_w * cos_i, -cos_om * sin_i],
        [sin_w * sin_i, cos_w * sin_i, cos_i],
    ], dtype=float)



def orbit_xyz(snapshot: HalleySnapshot, samples: int = 240, radius_limit: float = 40.0) -> np.ndarray:
    a = snapshot.semimajor_axis_au
    e = snapshot.eccentricity
    if not (np.isfinite(a) and a > 0 and np.isfinite(e)):
        return np.empty((0, 3))

    E = np.linspace(0, 2 * math.pi, samples)
    xp = a * (np.cos(E) - e)
    yp = a * np.sqrt(max(0.0, 1 - e * e)) * np.sin(E)

    rot = rotation_matrix(snapshot.node_deg, snapshot.inclination_deg, snapshot.arg_peri_deg)
    pts = np.vstack([xp, yp, np.zeros_like(xp)]).T @ rot.T
    radii = np.linalg.norm(pts, axis=1)
    pts[(~np.isfinite(radii)) | (radii > radius_limit * 1.06)] = np.nan
    return pts



def solve_kepler_elliptic(mean_anomaly_rad: float, e: float) -> float:
    M = (mean_anomaly_rad + math.pi) % (2 * math.pi) - math.pi
    E = M if e < 0.8 else math.pi
    for _ in range(15):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        if abs(fp) < 1e-12:
            break
        step = f / fp
        E -= step
        if abs(step) < 1e-12:
            break
    return E



def approximate_position_xyz(snapshot: HalleySnapshot, target_jd: float) -> Optional[np.ndarray]:
    a = snapshot.semimajor_axis_au
    e = snapshot.eccentricity
    if not (np.isfinite(a) and a > 0 and np.isfinite(e) and e < 1.0):
        return None

    if np.isfinite(snapshot.mean_anomaly_deg) and np.isfinite(snapshot.epoch_jd):
        n_deg_day = 0.9856076686 / (a ** 1.5)
        mean_anomaly_deg = snapshot.mean_anomaly_deg + n_deg_day * (target_jd - snapshot.epoch_jd)
    elif np.isfinite(snapshot.period_days) and snapshot.period_days > 0 and np.isfinite(snapshot.perihelion_time_jd):
        mean_anomaly_deg = 360.0 * (target_jd - snapshot.perihelion_time_jd) / snapshot.period_days
    else:
        return None

    E = solve_kepler_elliptic(math.radians(mean_anomaly_deg), e)
    xp = a * (math.cos(E) - e)
    yp = a * math.sqrt(max(0.0, 1.0 - e * e)) * math.sin(E)
    rot = rotation_matrix(snapshot.node_deg, snapshot.inclination_deg, snapshot.arg_peri_deg)
    return rot @ np.array([xp, yp, 0.0], dtype=float)



def compressed_radius(radius_au: float, max_au: float) -> float:
    radius_au = max(0.0, float(radius_au))
    return math.asinh(radius_au / 1.8) / max(1e-9, math.asinh(max_au / 1.8))



def map_top_down(point: Sequence[float], box: Tuple[int, int, int, int], max_au: float, log_scale: bool) -> Tuple[float, float]:
    x, y = float(point[0]), float(point[1])
    radius = math.hypot(x, y)
    if radius == 0:
        ux, uy = 0.0, 0.0
    else:
        radial = compressed_radius(radius, max_au) if log_scale else radius / max_au
        ux, uy = x / radius * radial, y / radius * radial
    x0, y0, x1, y1 = box
    scale = min(x1 - x0, y1 - y0) * 0.47
    return (x0 + x1) / 2 + ux * scale, (y0 + y1) / 2 - uy * scale



def split_valid_polyline(points: np.ndarray) -> List[List[Tuple[float, float]]]:
    chunks: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []
    for x, y in points:
        if np.isfinite(x) and np.isfinite(y):
            current.append((float(x), float(y)))
        else:
            if len(current) >= 2:
                chunks.append(current)
            current = []
    if len(current) >= 2:
        chunks.append(current)
    return chunks


# =============================================================================
# Scene renderer
# =============================================================================

class HalleyScene:
    def __init__(self, snapshot: HalleySnapshot, apparitions: Sequence[Apparition], summary: Dict):
        self.snapshot = snapshot
        self.apparitions = list(apparitions)
        self.summary = summary
        self.stars = self._make_stars(CONFIG["stars"], seed=7194)
        self.timeline_box = (int(W * 0.08), int(H * 0.29), int(W * 0.92), int(H * 0.55))
        self.orbit_box = (int(W * 0.04), int(H * 0.17), int(W * 0.96), int(H * 0.74))
        self.inner_box = (int(W * 0.58), int(H * 0.45), int(W * 0.92), int(H * 0.76))
        self.deep_radius = 40.0
        self.orbit_layer = self._build_orbit_layer(self.orbit_box, self.deep_radius, log_scale=True)
        self.inner_orbit_layer = self._build_orbit_layer(self.inner_box, 6.0, log_scale=False)
        self.timeline_start = self.apparitions[0].numeric_year
        self.timeline_end = self.apparitions[-1].numeric_year
        self.highlight_events = [a for a in self.apparitions if a.priority >= 2]

    @staticmethod
    def _make_stars(n: int, seed: int):
        rng = np.random.default_rng(seed)
        return [
            (float(rng.uniform(0, W)), float(rng.uniform(0, H)), float(rng.uniform(.4, 2.0) * SCALE),
             int(rng.integers(25, 145)), float(rng.uniform(0, math.tau)))
            for _ in range(n)
        ]

    def background(self, t: float) -> Image.Image:
        img = Image.new("RGBA", SIZE, COLORS["bg"] + (255,))
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        clouds = [
            (W * 0.18, H * 0.28, (16, 70, 125)),
            (W * 0.74, H * 0.22, (90, 35, 110)),
            (W * 0.52, H * 0.78, (14, 58, 96)),
        ]
        for cx, cy, color in clouds:
            for radius, alpha in [(W * 0.46, 13), (W * 0.30, 23), (W * 0.18, 32)]:
                gd.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=color + (alpha,))
        glow = glow.filter(ImageFilter.GaussianBlur(65 if not QUICK_MODE else 32))
        img.alpha_composite(glow)

        d = ImageDraw.Draw(img)
        for x, y, r, a, phase in self.stars:
            alpha = int(a * (0.72 + 0.28 * math.sin(1.5 * t + phase)))
            d.ellipse((x-r, y-r, x+r, y+r), fill=(214, 228, 255, alpha))
        return img

    def draw_title(self, img: Image.Image, t: float):
        alpha = int(255 * smoothstep((t - 0.15) / 0.8) * (1 - smoothstep((t - (6.4 if not QUICK_MODE else 1.55)) / 0.8)))
        if alpha > 4:
            draw_text(img, CONFIG["title"], (56 if not QUICK_MODE else 28, 90 if not QUICK_MODE else 45),
                      size=42 if not QUICK_MODE else 19, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(img, CONFIG["subtitle"], (58 if not QUICK_MODE else 30, 151 if not QUICK_MODE else 76),
                      size=22 if not QUICK_MODE else 10, fill=COLORS["cyan"] + (min(alpha, 230),), bold=True)
        shot_titles = {
            "intro": "ONE COMET • MANY CENTURIES",
            "timeline": "RECORDED RETURNS ACROSS HISTORY",
            "orbit": "HALLEY'S RETROGRADE ORBIT",
            "highlights": "FAMOUS APPARITIONS",
            "stats": "JPL ORBIT NUMBERS",
            "outro": "NEXT STOP: 2061",
        }
        if t > (5.0 if not QUICK_MODE else 1.25):
            draw_text(img, shot_titles[get_shot(t)["name"]], (56 if not QUICK_MODE else 28, 61 if not QUICK_MODE else 30),
                      size=19 if not QUICK_MODE else 9, fill=COLORS["muted"] + (210,), bold=True, stroke=1)

    def draw_source_hud(self, img: Image.Image):
        status = "OFFLINE FIXTURE" if self.snapshot.offline_fixture else ("CACHE" if self.snapshot.data_status == "cache" else "LIVE")
        label = f"HALLEY DATA // {status}"
        draw_text(img, label, (W - (48 if not QUICK_MODE else 24), 72 if not QUICK_MODE else 36),
                  size=17 if not QUICK_MODE else 8, fill=COLORS["cyan"] + (220,), bold=True, anchor="ra", stroke=1)
        generated = self.summary["generated_at_utc"].replace("T", " ").replace("Z", " UTC")
        draw_text(img, generated, (W - (48 if not QUICK_MODE else 24), 102 if not QUICK_MODE else 51),
                  size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (195,), anchor="ra", stroke=1)

    def draw_caption(self, img: Image.Image, t: float):
        caption = caption_at(t)
        if not caption:
            return
        y0 = H - (190 if not QUICK_MODE else 95)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((58 if not QUICK_MODE else 29, y0, W-(58 if not QUICK_MODE else 29), y0+(116 if not QUICK_MODE else 58)),
                             radius=20 if not QUICK_MODE else 10, fill=(2, 6, 14, 148), outline=(80, 185, 220, 48), width=1)
        img.alpha_composite(overlay)
        draw_wrapped_text(img, caption, (82 if not QUICK_MODE else 41, y0+(18 if not QUICK_MODE else 9)),
                          W-(164 if not QUICK_MODE else 82), size=30 if not QUICK_MODE else 15,
                          fill=COLORS["white"] + (240,))

    def draw_hud_noise(self, img: Image.Image, t: float):
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        offset = int((t * 39) % 7)
        for y in range(offset, H, 7):
            od.line((0, y, W, y), fill=(120, 200, 240, 10), width=1)
        scan_y = int((t * 165) % (H + 220)) - 110
        od.rectangle((0, scan_y, W, scan_y + (48 if not QUICK_MODE else 24)), fill=(90, 210, 240, 7))
        img.alpha_composite(overlay)

    def apparition_x(self, numeric_year: float, x0: int, x1: int) -> float:
        return x0 + (numeric_year - self.timeline_start) / (self.timeline_end - self.timeline_start) * (x1 - x0)

    def timeline_year(self, t: float) -> float:
        shot = get_shot(t)
        if shot["name"] == "timeline":
            p = ease_in_out_sine((t - shot["start"]) / max(1e-6, shot["end"] - shot["start"]))
        elif shot["name"] in {"orbit", "highlights", "stats", "outro"}:
            p = 1.0
        else:
            p = 0.18
        return lerp(self.timeline_start, self.timeline_end, p)

    def nearest_apparition(self, numeric_year: float) -> Apparition:
        return min(self.apparitions, key=lambda a: abs(a.numeric_year - numeric_year))

    def _build_orbit_layer(self, box, max_au: float, log_scale: bool) -> Image.Image:
        layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        pts = orbit_xyz(self.snapshot, samples=280 if not QUICK_MODE else 160, radius_limit=max_au)
        mapped = []
        for point in pts:
            if not np.isfinite(point).all():
                mapped.append((np.nan, np.nan))
            else:
                mapped.append(map_top_down(point, box, max_au, log_scale))
        for chunk in split_valid_polyline(np.asarray(mapped, dtype=float)):
            draw.line(chunk, fill=COLORS["trail"] + (125,), width=max(2, int(2.2 * SCALE)))
        return layer

    def draw_intro(self, img: Image.Image, t: float):
        cx, cy = W * 0.5, H * 0.43
        d = ImageDraw.Draw(img)
        sun_r = 56 * SCALE
        d.ellipse((cx-sun_r, cy-sun_r, cx+sun_r, cy+sun_r), fill=COLORS["sun"] + (255,), outline=(255, 245, 210, 220), width=2)
        orbit_r_x = 330 * SCALE
        orbit_r_y = 140 * SCALE
        d.ellipse((cx-orbit_r_x, cy-orbit_r_y, cx+orbit_r_x, cy+orbit_r_y), outline=COLORS["trail"] + (65,), width=3)
        phase = -1.5 + 3.2 * ease_in_out_sine((t - SHOT_PLAN[0]["start"]) / max(1e-6, SHOT_PLAN[0]["end"] - SHOT_PLAN[0]["start"]))
        comet_x = cx + orbit_r_x * math.cos(phase)
        comet_y = cy + orbit_r_y * math.sin(phase)
        tail = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        td = ImageDraw.Draw(tail)
        for i in range(12):
            frac = i / 11
            tx = lerp(comet_x, comet_x + 110 * SCALE, frac)
            ty = lerp(comet_y, comet_y - 18 * SCALE, frac)
            r = (14 - 10 * frac) * SCALE
            td.ellipse((tx-r, ty-r, tx+r, ty+r), fill=(160, 235, 255, int(70 * (1 - frac))))
        tail = tail.filter(ImageFilter.GaussianBlur(10 if not QUICK_MODE else 5))
        img.alpha_composite(tail)
        r = 16 * SCALE
        d.ellipse((comet_x-r, comet_y-r, comet_x+r, comet_y+r), fill=COLORS["halley"] + (250,), outline=(255, 255, 255, 220), width=1)
        draw_text(img, "1P/HALLEY", (int(comet_x), int(comet_y + 34 * SCALE)), size=20 if not QUICK_MODE else 9,
                  fill=COLORS["cyan"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(img, "~75-YEAR PERIOD • RETROGRADE • HIGHLY ELONGATED", (W // 2, int(H * 0.67)),
                  size=24 if not QUICK_MODE else 11, fill=COLORS["white"] + (235,), bold=True, anchor="ma", stroke=1)

    def draw_timeline(self, img: Image.Image, t: float):
        x0, y0, x1, y1 = self.timeline_box
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0, y0, x1, y1), radius=28 if not QUICK_MODE else 14,
                             fill=(2, 6, 14, 178), outline=(88, 185, 220, 78), width=2)
        img.alpha_composite(overlay)
        d = ImageDraw.Draw(img)
        line_y = (y0 + y1) // 2
        d.line((x0 + 28, line_y, x1 - 28, line_y), fill=COLORS["muted"] + (120,), width=3)

        for app in self.apparitions:
            x = self.apparition_x(app.numeric_year, x0 + 28, x1 - 28)
            tick_h = 24 if app.priority >= 2 else 16
            d.line((x, line_y - tick_h, x, line_y + tick_h), fill=COLORS["muted"] + (175,), width=2)

        major_labels = [self.apparitions[0], self.nearest_apparition(1066), self.nearest_apparition(1759), self.nearest_apparition(1986), self.apparitions[-1]]
        used = set()
        for app in major_labels:
            if app.numeric_year in used:
                continue
            used.add(app.numeric_year)
            x = self.apparition_x(app.numeric_year, x0 + 28, x1 - 28)
            draw_text(img, app.label, (int(x), line_y + 42 * SCALE), size=18 if not QUICK_MODE else 8,
                      fill=COLORS["white"] + (230,), bold=True, anchor="ma", stroke=1)

        current_year = self.timeline_year(t)
        current_x = self.apparition_x(current_year, x0 + 28, x1 - 28)
        pulse = 0.68 + 0.32 * math.sin(t * 4.0)
        d.line((current_x, y0 + 20, current_x, y1 - 20), fill=COLORS["gold"] + (120,), width=3)
        r = (14 if not QUICK_MODE else 7) * (1 + 0.14 * pulse)
        d.ellipse((current_x-r, line_y-r, current_x+r, line_y+r), fill=COLORS["gold"] + (245,), outline=(255, 255, 255, 220), width=1)

        nearest = self.nearest_apparition(current_year)
        year_label = nearest.label if abs(nearest.numeric_year - current_year) < 14 else (f"{int(round(current_year))} BCE" if current_year < 0 else f"{int(round(current_year))}")
        draw_text(img, year_label, (int(current_x), y0 + 28), size=25 if not QUICK_MODE else 11,
                  fill=COLORS["gold"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_wrapped_text(img, clip_text(nearest.note, 64), (x0 + 24, y1 - int(72 * SCALE)), x1 - x0 - 48,
                          size=18 if not QUICK_MODE else 8, fill=COLORS["muted"] + (225,), bold=False)

        draw_text(img, "RECORDED HALLEY APPARITIONS", (x0 + 18, y0 + 18), size=18 if not QUICK_MODE else 8,
                  fill=COLORS["cyan"] + (210,), bold=True, stroke=1)
        draw_text(img, f"COUNT // {len(self.apparitions)}", (x1 - 18, y0 + 18), size=16 if not QUICK_MODE else 7,
                  fill=COLORS["muted"] + (210,), bold=True, anchor="ra", stroke=1)

    def animated_year_for_orbit(self, t: float) -> float:
        if get_shot(t)["name"] == "orbit":
            p = ease_in_out_sine((t - SHOT_PLAN[2]["start"]) / max(1e-6, SHOT_PLAN[2]["end"] - SHOT_PLAN[2]["start"]))
        elif get_shot(t)["name"] in {"highlights", "stats", "outro"}:
            p = 1.0
        else:
            p = 0.86
        return lerp(self.timeline_start, self.timeline_end, p)

    def approx_jd_from_year(self, year_value: float) -> float:
        target_year = float(year_value)
        peri_year = year_from_jd(self.snapshot.perihelion_time_jd) if np.isfinite(self.snapshot.perihelion_time_jd) else 2061.57
        delta_years = target_year - peri_year
        return self.snapshot.perihelion_time_jd + delta_years * 365.2425

    def draw_planet_orbits(self, img: Image.Image, box, max_au: float, log_scale: bool):
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        draw_text(img, "TOP-DOWN HELIOCENTRIC VIEW", (x0 + 16, y0 + 16), size=18 if not QUICK_MODE else 8,
                  fill=COLORS["cyan"] + (220,), bold=True, stroke=1)
        for name, radius, color in PLANETS:
            if radius > max_au:
                continue
            if log_scale:
                rr = compressed_radius(radius, max_au) * min(x1 - x0, y1 - y0) * 0.47
            else:
                rr = radius / max_au * min(x1 - x0, y1 - y0) * 0.47
            d.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), outline=(110, 150, 175, 36), width=1)
            px = cx + rr
            py = cy
            pr = 5 * SCALE
            d.ellipse((px-pr, py-pr, px+pr, py+pr), fill=color + (220,))
            if name in {"EARTH", "JUPITER", "NEPTUNE"}:
                draw_text(img, name, (int(px + 10 * SCALE), int(py)), size=14 if not QUICK_MODE else 7,
                          fill=color + (220,), anchor="la", stroke=1)
        sun_r = 10 * SCALE if box != self.inner_box else 12 * SCALE
        d.ellipse((cx-sun_r, cy-sun_r, cx+sun_r, cy+sun_r), fill=COLORS["sun"] + (255,), outline=(255, 245, 210, 220), width=1)
        draw_text(img, "SUN", (int(cx), int(cy - 20 * SCALE)), size=14 if not QUICK_MODE else 7,
                  fill=COLORS["sun"] + (230,), bold=True, anchor="ma", stroke=1)

    def draw_orbit(self, img: Image.Image, t: float):
        x0, y0, x1, y1 = self.orbit_box
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0, y0, x1, y1), radius=28 if not QUICK_MODE else 14,
                             fill=(2, 6, 14, 164), outline=(88, 185, 220, 78), width=2)
        img.alpha_composite(overlay)
        self.draw_planet_orbits(img, self.orbit_box, self.deep_radius, log_scale=True)
        img.alpha_composite(self.orbit_layer)

        current_year = self.animated_year_for_orbit(t)
        jd = self.approx_jd_from_year(current_year)
        pos = approximate_position_xyz(self.snapshot, jd)
        if pos is not None:
            x, y = map_top_down(pos, self.orbit_box, self.deep_radius, True)
            tail = Image.new("RGBA", SIZE, (0, 0, 0, 0))
            td = ImageDraw.Draw(tail)
            for i in range(10):
                frac = i / 9
                tx = lerp(x, x + 52 * SCALE, frac)
                ty = lerp(y, y - 11 * SCALE, frac)
                r = (12 - 9 * frac) * SCALE
                td.ellipse((tx-r, ty-r, tx+r, ty+r), fill=(150, 232, 255, int(65 * (1 - frac))))
            tail = tail.filter(ImageFilter.GaussianBlur(8 if not QUICK_MODE else 4))
            img.alpha_composite(tail)
            r = 10 * SCALE
            d = ImageDraw.Draw(img)
            d.ellipse((x-r, y-r, x+r, y+r), fill=COLORS["halley"] + (250,), outline=(255, 255, 255, 220), width=1)
            draw_text(img, "HALLEY", (int(x), int(y + 24 * SCALE)), size=16 if not QUICK_MODE else 8,
                      fill=COLORS["white"] + (235,), bold=True, anchor="ma", stroke=1)

        draw_text(img, f"ANIMATED YEAR // {int(round(current_year)) if current_year >= 1 else str(abs(int(round(current_year)))) + ' BCE'}",
                  (x1 - 16, y0 + 16), size=17 if not QUICK_MODE else 8,
                  fill=COLORS["muted"] + (220,), bold=True, anchor="ra", stroke=1)

        # inner-system inset
        ix0, iy0, ix1, iy1 = self.inner_box
        inset = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        idr = ImageDraw.Draw(inset)
        idr.rounded_rectangle((ix0, iy0, ix1, iy1), radius=22 if not QUICK_MODE else 11,
                              fill=(3, 8, 17, 200), outline=(90, 180, 210, 70), width=1)
        img.alpha_composite(inset)
        self.draw_planet_orbits(img, self.inner_box, 6.0, log_scale=False)
        img.alpha_composite(self.inner_orbit_layer)
        if pos is not None:
            xi, yi = map_top_down(pos, self.inner_box, 6.0, False)
            if ix0 <= xi <= ix1 and iy0 <= yi <= iy1:
                d = ImageDraw.Draw(img)
                rr = 7 * SCALE
                d.ellipse((xi-rr, yi-rr, xi+rr, yi+rr), fill=COLORS["halley"] + (250,), outline=(255, 255, 255, 220), width=1)
        draw_text(img, "INNER SOLAR SYSTEM", (ix0 + 12, iy0 + 12), size=14 if not QUICK_MODE else 7,
                  fill=COLORS["cyan"] + (210,), bold=True, stroke=1)
        draw_wrapped_text(img, "Halley dives inside Venus' orbit at perihelion, then travels outward beyond Neptune.",
                          (x0 + 16, y1 - int(84 * SCALE)), x1 - x0 - 32, size=18 if not QUICK_MODE else 8,
                          fill=COLORS["muted"] + (225,))

    def draw_highlights(self, img: Image.Image, t: float):
        x0 = int(W * 0.07)
        y0 = int(H * 0.20)
        card_w = int(W * 0.40)
        card_h = int(H * 0.16)
        xs = [x0, int(W * 0.53)]
        ys = [y0, int(H * 0.39), int(H * 0.58)]
        cards = [
            self.nearest_apparition(1066),
            self.nearest_apparition(1705),
            self.nearest_apparition(1759),
            self.nearest_apparition(1910),
            self.nearest_apparition(1986),
            self.nearest_apparition(2061),
        ]
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for idx, app in enumerate(cards):
            x = xs[idx % 2]
            y = ys[idx // 2]
            od.rounded_rectangle((x, y, x + card_w, y + card_h), radius=24 if not QUICK_MODE else 12,
                                 fill=(3, 8, 17, 186), outline=COLORS["cyan"] + (80,), width=2)
        img.alpha_composite(overlay)

        for idx, app in enumerate(cards):
            x = xs[idx % 2]
            y = ys[idx // 2]
            draw_text(img, app.label, (x + 18, y + 18), size=25 if not QUICK_MODE else 11,
                      fill=COLORS["gold"] + (245,), bold=True, stroke=1)
            draw_wrapped_text(img, app.note, (x + 18, y + 58 if not QUICK_MODE else y + 28), card_w - 36,
                              size=17 if not QUICK_MODE else 8, fill=COLORS["white"] + (236,))

    def stat_line(self, img: Image.Image, left: str, right: str, x: int, y: int, width: int, color: Tuple[int, int, int]):
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x, y, x + width, y + int(74 * SCALE)), radius=22 if not QUICK_MODE else 10,
                             fill=(3, 8, 17, 185), outline=color + (80,), width=2)
        img.alpha_composite(overlay)
        draw_text(img, left, (x + 18, y + 18), size=22 if not QUICK_MODE else 10,
                  fill=color + (240,), bold=True, stroke=1)
        draw_text(img, right, (x + width - 18, y + 18), size=22 if not QUICK_MODE else 10,
                  fill=COLORS["white"] + (235,), bold=True, anchor="ra", stroke=1)

    def draw_stats(self, img: Image.Image):
        x = int(W * 0.08)
        y = int(H * 0.22)
        width = int(W * 0.84)
        gap = int(18 * SCALE)
        period_years = self.snapshot.period_days / 365.25 if np.isfinite(self.snapshot.period_days) else np.nan
        values = [
            ("Orbital period", f"{period_years:.2f} years" if np.isfinite(period_years) else "n/a", COLORS["cyan"]),
            ("Eccentricity", f"{self.snapshot.eccentricity:.5f}" if np.isfinite(self.snapshot.eccentricity) else "n/a", COLORS["green"]),
            ("Perihelion", f"{self.snapshot.perihelion_au:.3f} AU" if np.isfinite(self.snapshot.perihelion_au) else "n/a", COLORS["gold"]),
            ("Semimajor axis", f"{self.snapshot.semimajor_axis_au:.3f} AU" if np.isfinite(self.snapshot.semimajor_axis_au) else "n/a", COLORS["violet"]),
            ("Inclination", f"{self.snapshot.inclination_deg:.2f}° retrograde" if np.isfinite(self.snapshot.inclination_deg) else "n/a", COLORS["red"]),
            ("Next perihelion", clip_text(self.snapshot.perihelion_time_cd or "2061", 24), COLORS["muted"]),
        ]
        row_h = int(86 * SCALE)
        for idx, (left, right, color) in enumerate(values):
            self.stat_line(img, left, right, x, y + idx * (row_h + gap), width, color)
        draw_wrapped_text(img, clip_text(self.snapshot.note, 130), (x + 8, y + len(values) * (row_h + gap) + 8),
                          width - 16, size=17 if not QUICK_MODE else 8, fill=COLORS["muted"] + (220,))

    def draw_outro(self, img: Image.Image):
        self.draw_timeline(img, SHOT_PLAN[-1]["end"])
        draw_text(img, "NEXT PREDICTED RETURN", (W // 2, int(H * 0.64)), size=28 if not QUICK_MODE else 13,
                  fill=COLORS["white"] + (240,), bold=True, anchor="ma", stroke=1)
        draw_text(img, clip_text(self.snapshot.perihelion_time_cd or "2061", 24), (W // 2, int(H * 0.69)),
                  size=36 if not QUICK_MODE else 16, fill=COLORS["gold"] + (245,), bold=True, anchor="ma", stroke=1)
        draw_text(img, "Run the script again later for a fresh JPL orbit snapshot", (W // 2, int(H * 0.75)),
                  size=20 if not QUICK_MODE else 9, fill=COLORS["cyan"] + (230,), bold=True, anchor="ma", stroke=1)

    def render_frame(self, t: float) -> np.ndarray:
        img = self.background(t)
        self.draw_title(img, t)
        self.draw_source_hud(img)

        shot = get_shot(t)["name"]
        if shot == "intro":
            self.draw_intro(img, t)
        elif shot == "timeline":
            self.draw_timeline(img, t)
        elif shot == "orbit":
            self.draw_orbit(img, t)
        elif shot == "highlights":
            self.draw_highlights(img, t)
        elif shot == "stats":
            self.draw_stats(img)
        else:
            self.draw_outro(img)

        self.draw_caption(img, t)
        self.draw_hud_noise(img, t)

        arr = np.array(img.convert("RGB"))
        graded = Image.fromarray(arr)
        graded = ImageEnhance.Contrast(graded).enhance(1.08)
        graded = ImageEnhance.Color(graded).enhance(1.06)
        arr = np.array(graded)
        arr = np.clip(arr.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / 0.9)
        fade_out = 1 - smoothstep((t - (CONFIG["duration_s"] - 1.1)) / 1.0)
        return np.clip(arr.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)


# =============================================================================
# Output
# =============================================================================


def render_video(scene: HalleyScene) -> Path:
    raw_path = OUTPUT_ROOT / f"{CONFIG['basename']}_raw.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['basename']}_final.mp4"
    write_srt(OUTPUT_ROOT / f"{CONFIG['basename']}_subtitles.srt")
    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    with iio.get_writer(raw_path, fps=CONFIG["fps"], codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None) as writer:
        for frame_index in tqdm(range(frame_count), desc="Rendering Halley short"):
            writer.append_data(scene.render_frame(frame_index / CONFIG["fps"]))
    shutil.copyfile(raw_path, final_path)
    return final_path



def make_contact_sheet(paths: Sequence[Path], out_path: Path):
    thumbs = []
    for path in paths[:6]:
        image = Image.open(path).convert("RGB").resize((270, 480))
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 120, 38), fill=(0, 0, 0))
        draw.text((18, 13), path.stem.replace("preview_", ""), fill=(255, 255, 255))
        thumbs.append(image)
    sheet = Image.new("RGB", (600, 1520), (8, 11, 18))
    for index, thumb in enumerate(thumbs):
        row, col = divmod(index, 2)
        sheet.paste(thumb, (20 + col * 290, 20 + row * 500))
    sheet.save(out_path, quality=92)




# =============================================================================
# YouTube title / description sidecar
# =============================================================================

YOUTUBE_TITLE = "Watch Halley's Comet Travel Through Time ☄️ #rootjatin"
YOUTUBE_DESCRIPTION = "Follow Halley's Comet through its famous historical returns and toward its predicted 2061 comeback. The animation uses published orbital parameters to illustrate Halley's large, inclined, retrograde orbit and highlights milestones such as 1066, the prediction of its return, 1910, and the 1986 spacecraft encounter. This is a data-grounded educational reconstruction, not a live telescope feed or a full long-term dynamical integration."
YOUTUBE_HASHTAGS = '#rootjatin #HalleysComet #Comet #Astronomy #Space #SolarSystem #History #Science'

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


