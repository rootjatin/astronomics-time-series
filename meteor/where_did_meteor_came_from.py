from __future__ import annotations

"""
Where Did This Meteor Come From?
================================

A cinematic vertical YouTube Short renderer that selects one real meteor
trajectory from the Global Meteor Network (GMN) and traces it backward from the
atmosphere to its apparent radiant and pre-atmospheric heliocentric orbit.


Selection controls
------------------
Choose a UTC date:
    METEOR_ORIGIN_DATE=2025-12-14

Choose an exact GMN trajectory identifier:
    METEOR_ORIGIN_ID=YYYYMMDDhhmmss_hash

Schema and licensing:
    https://gmn-python-api.readthedocs.io/en/latest/data_schemas.html
    https://globalmeteornetwork.org/data/

interpretation rules
------------------------------
- The ground track, radiant, velocity, and orbital elements are reconstructed
  from multi-station camera measurements; they are not direct samples of the
  meteoroid before it reached Earth.
- The heliocentric curve is an osculating two-body conic drawn from GMN's
  published elements. It is not a long-term n-body integration.
- A radiant is an apparent incoming direction, not a physical birthplace.
- Shower membership links a meteor to a stream. A named parent body is shown
  only for a small set of widely established shower-parent associations.
- Tisserand-parameter labels are broad dynamical hints, not proof of physical
  composition or a unique source object.
- If no parent is securely mapped, the video says so rather than inventing one.


Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg requests tqdm

Run final quality
-----------------
    python where_did_this_meteor_come_from_short.py

Run quick preview
-----------------
    METEOR_ORIGIN_SHORT_QUICK=1 python where_did_this_meteor_come_from_short.py

Force offline layout testing
----------------------------
    METEOR_ORIGIN_SHORT_OFFLINE=1 METEOR_ORIGIN_SHORT_QUICK=1 \
        python where_did_this_meteor_come_from_short.py
"""

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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

QUICK_MODE = os.environ.get("METEOR_ORIGIN_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("METEOR_ORIGIN_SHORT_OFFLINE", "0") == "1"
REFRESH = os.environ.get("METEOR_ORIGIN_SHORT_REFRESH", "0") == "1"
DATE_OVERRIDE = os.environ.get("METEOR_ORIGIN_DATE", "").strip()
ID_OVERRIDE = os.environ.get("METEOR_ORIGIN_ID", "").strip()
MAX_PAGES = max(1, int(os.environ.get("METEOR_ORIGIN_MAX_PAGES", "12")))

OUTPUT_ROOT = Path("where_did_this_meteor_come_from_output")
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
    "basename": "where_did_this_meteor_come_from",
    "title": "WHERE DID THIS METEOR COME FROM?",
    "subtitle": "Tracing one measured meteor backward through space",
    "timeout_s": 40,
    "stars": 640 if QUICK_MODE else 1100,
    "api_base": "https://explore.globalmeteornetwork.org/gmn_rest_api",
    "summary_endpoint": "https://explore.globalmeteornetwork.org/gmn_rest_api/meteor_summary",
    "source_page": "https://globalmeteornetwork.org/data/",
    "cache_hours": 6,
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
    "blue": (85, 145, 255),
    "gold": (255, 205, 92),
    "violet": (201, 116, 255),
    "green": (104, 255, 181),
    "red": (255, 115, 125),
    "orange": (255, 160, 84),
    "sun": (255, 205, 79),
    "earth": (93, 174, 255),
}

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.0 if not QUICK_MODE else 1.7},
    {"name": "atmosphere", "start": 7.0 if not QUICK_MODE else 1.7, "end": 18.0 if not QUICK_MODE else 3.9},
    {"name": "radiant", "start": 18.0 if not QUICK_MODE else 3.9, "end": 29.0 if not QUICK_MODE else 6.1},
    {"name": "orbit", "start": 29.0 if not QUICK_MODE else 6.1, "end": 43.0 if not QUICK_MODE else 8.8},
    {"name": "origin", "start": 43.0 if not QUICK_MODE else 8.8, "end": 53.0 if not QUICK_MODE else 10.8},
    {"name": "outro", "start": 53.0 if not QUICK_MODE else 10.8, "end": CONFIG["duration_s"]},
]

CAPTION_TEXTS = [
    "This meteor was reconstructed from observations by multiple cameras in the Global Meteor Network.",
    "The measured beginning and ending points reveal its path through the upper atmosphere, including its speed and luminous heights.",
    "Extending that path backward gives the geocentric radiant—the direction in the sky from which the meteor appeared to arrive.",
    "GMN also publishes a heliocentric orbit. This curve traces the meteoroid's path around the Sun before Earth intercepted it.",
    "A shower code can connect the meteor to a wider stream, and sometimes to a well-established parent comet or asteroid.",
    "The reconstruction narrows down its origin, but the radiant is not a birthplace and the orbit is not a full long-term simulation.",
]
CAPTIONS = [
    (SHOT_PLAN[i]["start"] + (0.2 if i == 0 else 0.05), SHOT_PLAN[i]["end"] - 0.05, CAPTION_TEXTS[i])
    for i in range(len(SHOT_PLAN))
]

# Conservative, widely used shower-parent associations. The renderer only uses
# this map when the selected meteor's GMN IAU shower code matches exactly.
SHOWER_PARENTS = {
    "GEM": ("Geminids", "3200 Phaethon", "asteroid-like active body"),
    "PER": ("Perseids", "109P/Swift–Tuttle", "comet"),
    "LEO": ("Leonids", "55P/Tempel–Tuttle", "comet"),
    "LYR": ("Lyrids", "C/1861 G1 Thatcher", "long-period comet"),
    "QUA": ("Quadrantids", "2003 EH1", "asteroid-like body"),
    "ORI": ("Orionids", "1P/Halley", "comet"),
    "ETA": ("eta Aquariids", "1P/Halley", "comet"),
    "DRA": ("Draconids", "21P/Giacobini–Zinner", "comet"),
    "URS": ("Ursids", "8P/Tuttle", "comet"),
    "STA": ("Southern Taurids", "2P/Encke / Taurid complex", "comet-stream complex"),
    "NTA": ("Northern Taurids", "2P/Encke / Taurid complex", "comet-stream complex"),
}

PLANETS = [
    ("MERCURY", 0.387, (180, 188, 198)),
    ("VENUS", 0.723, (232, 197, 130)),
    ("EARTH", 1.000, COLORS["earth"]),
    ("MARS", 1.524, (231, 126, 87)),
    ("JUPITER", 5.203, (232, 185, 124)),
]


# =============================================================================
# Data model
# =============================================================================

@dataclass
class MeteorOriginEvent:
    trajectory_id: str
    beginning_utc_time: str
    iau_code: str
    iau_no: int
    sol_lon_deg: float
    rageo_deg: float
    decgeo_deg: float
    lamgeo_deg: float
    betgeo_deg: float
    vgeo_km_s: float
    lamhel_deg: float
    bethel_deg: float
    vhel_km_s: float
    a_au: float
    eccentricity: float
    inclination_deg: float
    arg_peri_deg: float
    node_deg: float
    perihelion_au: float
    true_anomaly_deg: float
    mean_anomaly_deg: float
    aphelion_au: float
    period_years: float
    tisserand_jupiter: float
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
        try:
            return pd.to_datetime(self.beginning_utc_time, utc=True).to_pydatetime()
        except Exception:
            return None

    @property
    def shower_code(self) -> str:
        code = str(self.iau_code or "").strip().upper()
        if not code or code in {"...", "-1", "NONE", "NULL", "SPO"}:
            return "SPORADIC"
        return code

    @property
    def shower_parent(self) -> Optional[Tuple[str, str, str]]:
        return SHOWER_PARENTS.get(self.shower_code)

    @property
    def has_ground_track(self) -> bool:
        values = [self.latbeg_n_deg, self.lonbeg_e_deg, self.latend_n_deg, self.lonend_e_deg]
        return all(np.isfinite(values))

    @property
    def has_orbit(self) -> bool:
        values = [self.a_au, self.eccentricity, self.inclination_deg, self.arg_peri_deg, self.node_deg]
        return all(np.isfinite(values)) and self.a_au > 0 and 0 <= self.eccentricity < 1


# =============================================================================
# General utilities
# =============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def safe_int(value, default=0) -> int:
    try:
        if value is None or value == "":
            return int(default)
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


def clip_text(text: str, n: int = 34) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


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
        str(text),
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


# =============================================================================
# Live GMN data retrieval
# =============================================================================


def cached_json_get(url: str, params: Dict[str, object], cache_name: str) -> Dict:
    cache_path = CACHE_ROOT / cache_name
    if cache_path.exists() and not REFRESH:
        age_hours = (utc_now().timestamp() - cache_path.stat().st_mtime) / 3600.0
        if age_hours <= CONFIG["cache_hours"]:
            return json.loads(cache_path.read_text(encoding="utf-8"))
    if requests is None:
        raise RuntimeError("requests is unavailable")
    response = requests.get(
        url,
        params=params,
        timeout=CONFIG["timeout_s"],
        headers={"User-Agent": "MeteorOriginShort/1.0 educational renderer"},
    )
    response.raise_for_status()
    payload = response.json()
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def validate_date_text(text: str) -> str:
    return datetime.strptime(text, "%Y-%m-%d").date().isoformat()


def fetch_latest_complete_date() -> str:
    sql = "SELECT MAX(date(beginning_utc_time)) AS latest_date FROM meteor WHERE date(beginning_utc_time) < date('now')"
    payload = cached_json_get(
        CONFIG["api_base"],
        {"sql": sql, "data_shape": "objects", "data_format": "json"},
        "latest_complete_date.json",
    )
    rows = payload.get("rows") or []
    if not rows or not rows[0].get("latest_date"):
        raise RuntimeError("GMN API did not return a latest complete date")
    return validate_date_text(str(rows[0]["latest_date"]))


def parse_gmn_row(row: Dict) -> MeteorOriginEvent:
    stations = row.get("participating_stations")
    if isinstance(stations, list):
        stations_text = ",".join(str(item) for item in stations)
    else:
        stations_text = str(stations or "")
    return MeteorOriginEvent(
        trajectory_id=str(row.get("unique_trajectory_identifier") or "unknown"),
        beginning_utc_time=str(row.get("beginning_utc_time") or ""),
        iau_code=str(row.get("iau_code") or ""),
        iau_no=safe_int(row.get("iau_no"), -1),
        sol_lon_deg=safe_float(row.get("sol_lon_deg")),
        rageo_deg=safe_float(row.get("rageo_deg")),
        decgeo_deg=safe_float(row.get("decgeo_deg")),
        lamgeo_deg=safe_float(row.get("lamgeo_deg")),
        betgeo_deg=safe_float(row.get("betgeo_deg")),
        vgeo_km_s=safe_float(row.get("vgeo_km_s")),
        lamhel_deg=safe_float(row.get("lamhel_deg")),
        bethel_deg=safe_float(row.get("bethel_deg")),
        vhel_km_s=safe_float(row.get("vhel_km_s")),
        a_au=safe_float(row.get("a_au")),
        eccentricity=safe_float(row.get("e")),
        inclination_deg=safe_float(row.get("i_deg")),
        arg_peri_deg=safe_float(row.get("peri_deg")),
        node_deg=safe_float(row.get("node_deg")),
        perihelion_au=safe_float(row.get("q_au")),
        true_anomaly_deg=safe_float(row.get("f_deg")),
        mean_anomaly_deg=safe_float(row.get("m_deg")),
        aphelion_au=safe_float(row.get("q_au_")),
        period_years=safe_float(row.get("t_years")),
        tisserand_jupiter=safe_float(row.get("tisserandj")),
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


def fetch_exact_trajectory(trajectory_id: str) -> MeteorOriginEvent:
    payload = cached_json_get(
        CONFIG["summary_endpoint"],
        {
            "where": f"meteor.unique_trajectory_identifier='{trajectory_id.replace(chr(39), chr(39)+chr(39))}'",
            "data_shape": "objects",
            "data_format": "json",
        },
        f"trajectory_{trajectory_id}.json",
    )
    rows = payload.get("rows") or []
    if not rows:
        raise RuntimeError(f"No GMN trajectory found for {trajectory_id}")
    return parse_gmn_row(rows[0])


def fetch_date_page(date_text: str, page: int) -> List[MeteorOriginEvent]:
    payload = cached_json_get(
        CONFIG["summary_endpoint"],
        {
            "where": f"date(beginning_utc_time)='{date_text}'",
            "order_by": "peak_absmag ASC, num_stat DESC",
            "page": page,
            "data_shape": "objects",
            "data_format": "json",
        },
        f"date_{date_text}_page_{page:03d}.json",
    )
    return [parse_gmn_row(row) for row in (payload.get("rows") or [])]


def event_selection_score(event: MeteorOriginEvent) -> float:
    if not event.has_orbit:
        return -1e9
    score = 0.0
    if event.shower_parent:
        score += 75.0
    elif event.shower_code != "SPORADIC":
        score += 25.0
    if event.has_ground_track:
        score += 15.0
    score += min(max(event.num_stations, 0), 8) * 3.0
    if np.isfinite(event.peak_absmag):
        score += clamp((-event.peak_absmag + 1.0) / 8.0) * 30.0
    if np.isfinite(event.duration_sec):
        score += clamp(event.duration_sec / 3.0) * 6.0
    if np.isfinite(event.vgeo_km_s):
        score += clamp((event.vgeo_km_s - 11.0) / 60.0) * 5.0
    return score


def fetch_best_event_for_date(date_text: str) -> Tuple[MeteorOriginEvent, int, Dict]:
    events: List[MeteorOriginEvent] = []
    page_meta = []
    for page in range(1, MAX_PAGES + 1):
        rows = fetch_date_page(date_text, page)
        page_meta.append({"page": page, "rows": len(rows)})
        if not rows:
            break
        events.extend(rows)
        if len(rows) < 1000:
            break
    if not events:
        raise RuntimeError(f"No GMN events returned for {date_text}")
    candidates = [event for event in events if event.has_orbit]
    if not candidates:
        raise RuntimeError(f"No event with a drawable heliocentric orbit was returned for {date_text}")
    chosen = max(candidates, key=event_selection_score)
    return chosen, len(events), {"pages": page_meta, "max_pages": MAX_PAGES}


def offline_fixture() -> MeteorOriginEvent:
    # Approximate Geminid-style values for layout testing only. The output is
    # prominently labeled OFFLINE FIXTURE and must not be treated as a real
    # trajectory record.
    return MeteorOriginEvent(
        trajectory_id="OFFLINE_GEMINID_LAYOUT_FIXTURE",
        beginning_utc_time="2025-12-14 02:17:42.000000",
        iau_code="GEM",
        iau_no=4,
        sol_lon_deg=262.1,
        rageo_deg=112.4,
        decgeo_deg=32.8,
        lamgeo_deg=108.2,
        betgeo_deg=10.5,
        vgeo_km_s=34.7,
        lamhel_deg=231.2,
        bethel_deg=-2.0,
        vhel_km_s=35.1,
        a_au=1.27,
        eccentricity=0.89,
        inclination_deg=22.1,
        arg_peri_deg=324.4,
        node_deg=262.1,
        perihelion_au=0.14,
        true_anomaly_deg=174.2,
        mean_anomaly_deg=135.0,
        aphelion_au=2.40,
        period_years=1.43,
        tisserand_jupiter=4.5,
        vinit_km_s=35.2,
        latbeg_n_deg=44.71,
        lonbeg_e_deg=15.92,
        htbeg_km=96.2,
        latend_n_deg=44.49,
        lonend_e_deg=16.28,
        htend_km=78.4,
        duration_sec=1.25,
        peak_absmag=-3.8,
        peak_ht_km=84.1,
        mass_kg=0.012,
        num_stations=4,
        participating_stations="FIX01,FIX02,FIX03,FIX04",
    )


def orbital_hint(event: MeteorOriginEvent) -> Tuple[str, str]:
    tj = event.tisserand_jupiter
    if not np.isfinite(tj):
        return "UNCLASSIFIED ORBIT", "Tisserand parameter unavailable"
    if tj < 2.0:
        return "LONG-PERIOD / HALLEY-TYPE HINT", "Very comet-like dynamical regime"
    if tj < 3.0:
        return "JUPITER-FAMILY COMET-LIKE HINT", "Comet-like Tisserand regime"
    return "ASTEROID-LIKE DYNAMICAL HINT", "Tisserand parameter above 3"


def collect_data() -> Tuple[MeteorOriginEvent, Dict]:
    errors = {}
    fetch_meta = {}
    source_mode = "live"
    selected_date = None
    date_event_count = 1

    if OFFLINE_MODE:
        event = offline_fixture()
        source_mode = "offline-fixture"
        selected_date = "2025-12-14"
    else:
        try:
            if ID_OVERRIDE:
                event = fetch_exact_trajectory(ID_OVERRIDE)
                selected_date = event.beginning_utc_time[:10]
                fetch_meta = {"selection": "exact trajectory identifier"}
            else:
                selected_date = validate_date_text(DATE_OVERRIDE) if DATE_OVERRIDE else fetch_latest_complete_date()
                event, date_event_count, fetch_meta = fetch_best_event_for_date(selected_date)
        except Exception as exc:
            errors["live_fetch"] = str(exc)
            event = offline_fixture()
            source_mode = "offline-fixture"
            selected_date = "2025-12-14"

    parent = event.shower_parent
    hint_title, hint_note = orbital_hint(event)
    summary = {
        "title": CONFIG["title"],
        "generated_at_utc": iso_z(utc_now()),
        "source_mode": source_mode,
        "selected_date_utc": selected_date,
        "date_event_count": date_event_count,
        "trajectory_id": event.trajectory_id,
        "selection_score": event_selection_score(event),
        "shower_code": event.shower_code,
        "shower_name": parent[0] if parent else ("Sporadic" if event.shower_code == "SPORADIC" else event.shower_code),
        "mapped_parent_body": parent[1] if parent else None,
        "mapped_parent_type": parent[2] if parent else None,
        "orbital_hint": hint_title,
        "orbital_hint_note": hint_note,
        "fetch_meta": fetch_meta,
        "errors": errors,
        "source_url": CONFIG["source_page"],
        "license": "GMN high-level trajectory data: CC BY 4.0",
        "warning": "Radiant and osculating orbit constrain the incoming stream; they do not identify a unique physical birthplace unless an established shower-parent association exists.",
    }
    return event, summary


def save_data(event: MeteorOriginEvent, summary: Dict) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "selected_meteor_origin.csv"
    json_path = DATA_ROOT / "selected_meteor_origin_summary.json"
    pd.DataFrame([asdict(event)]).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"summary": summary, "event": asdict(event)}, indent=2), encoding="utf-8")
    return csv_path, json_path


# =============================================================================
# Orbit and map geometry
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


def orbit_xyz(event: MeteorOriginEvent, samples: int = 320) -> np.ndarray:
    a = event.a_au
    e = event.eccentricity
    if not event.has_orbit:
        return np.empty((0, 3))
    E = np.linspace(0.0, 2 * math.pi, samples, endpoint=False)
    xp = a * (np.cos(E) - e)
    yp = a * np.sqrt(max(0.0, 1.0 - e * e)) * np.sin(E)
    rot = rotation_matrix(event.node_deg, event.inclination_deg, event.arg_peri_deg)
    return np.vstack([xp, yp, np.zeros_like(xp)]).T @ rot.T


def true_anomaly_to_eccentric(nu_deg: float, e: float) -> float:
    nu = math.radians(nu_deg)
    return 2.0 * math.atan2(math.sqrt(max(0.0, 1-e)) * math.sin(nu/2), math.sqrt(1+e) * math.cos(nu/2))


def position_from_eccentric(event: MeteorOriginEvent, E: float) -> np.ndarray:
    a, e = event.a_au, event.eccentricity
    xp = a * (math.cos(E) - e)
    yp = a * math.sqrt(max(0.0, 1.0-e*e)) * math.sin(E)
    return rotation_matrix(event.node_deg, event.inclination_deg, event.arg_peri_deg) @ np.array([xp, yp, 0.0])


def map_top_down(point: Sequence[float], box: Tuple[int, int, int, int], max_au: float) -> Tuple[float, float]:
    x, y = float(point[0]), float(point[1])
    x0, y0, x1, y1 = box
    scale = min(x1-x0, y1-y0) * 0.46 / max(max_au, 1e-6)
    return (x0+x1)/2 + x*scale, (y0+y1)/2 - y*scale


def lonlat_xy(lon_deg: float, lat_deg: float, box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = box
    x = x0 + ((lon_deg + 180.0) % 360.0) / 360.0 * (x1-x0)
    y = y0 + (90.0-lat_deg)/180.0 * (y1-y0)
    return x, y


def sky_xy(ra_deg: float, dec_deg: float, box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = box
    x = x0 + ((360.0-ra_deg) % 360.0)/360.0*(x1-x0)
    y = y0 + (90.0-dec_deg)/180.0*(y1-y0)
    return x, y


# =============================================================================
# Scene renderer
# =============================================================================

class MeteorOriginScene:
    def __init__(self, event: MeteorOriginEvent, summary: Dict):
        self.event = event
        self.summary = summary
        self.stars = self._make_stars(CONFIG["stars"], seed=5903)
        self.world_box = (int(W*.07), int(H*.21), int(W*.93), int(H*.68))
        self.sky_box = (int(W*.07), int(H*.19), int(W*.93), int(H*.70))
        self.orbit_box = (int(W*.045), int(H*.17), int(W*.955), int(H*.72))
        self.orbit_pts = orbit_xyz(event, 280 if QUICK_MODE else 420)
        self.orbit_max = max(5.5, float(np.nanmax(np.linalg.norm(self.orbit_pts, axis=1))) * 1.08) if len(self.orbit_pts) else 5.5
        self.impact_E = true_anomaly_to_eccentric(event.true_anomaly_deg if np.isfinite(event.true_anomaly_deg) else 180.0, event.eccentricity)

    @staticmethod
    def _make_stars(n: int, seed: int):
        rng = np.random.default_rng(seed)
        return [
            (float(rng.uniform(0,W)), float(rng.uniform(0,H)), float(rng.uniform(.4,2.0)*SCALE), int(rng.integers(25,145)), float(rng.uniform(0,math.tau)))
            for _ in range(n)
        ]

    def background(self, t: float) -> Image.Image:
        img = Image.new("RGBA", SIZE, COLORS["bg"]+(255,))
        glow = Image.new("RGBA", SIZE, (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        for cx,cy,color in [(W*.18,H*.29,(16,70,125)),(W*.74,H*.22,(90,35,110)),(W*.52,H*.78,(14,58,96))]:
            for radius,alpha in [(W*.46,13),(W*.30,23),(W*.18,32)]:
                gd.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),fill=color+(alpha,))
        glow=glow.filter(ImageFilter.GaussianBlur(65 if not QUICK_MODE else 32))
        img.alpha_composite(glow)
        d=ImageDraw.Draw(img)
        for x,y,r,a,phase in self.stars:
            alpha=int(a*(.72+.28*math.sin(1.5*t+phase)))
            d.ellipse((x-r,y-r,x+r,y+r),fill=(214,228,255,alpha))
        return img

    def draw_title(self,img:Image.Image,t:float):
        alpha=int(255*smoothstep((t-.15)/.8)*(1-smoothstep((t-(6.4 if not QUICK_MODE else 1.5))/.8)))
        if alpha>4:
            draw_text(img,CONFIG["title"],(56 if not QUICK_MODE else 28,90 if not QUICK_MODE else 45),size=42 if not QUICK_MODE else 19,fill=COLORS["white"]+(alpha,),bold=True)
            draw_text(img,CONFIG["subtitle"],(58 if not QUICK_MODE else 30,151 if not QUICK_MODE else 76),size=22 if not QUICK_MODE else 10,fill=COLORS["cyan"]+(min(alpha,230),),bold=True)
        titles={"intro":"ONE METEOR • THREE CLUES","atmosphere":"CLUE 1 // THE ATMOSPHERIC PATH","radiant":"CLUE 2 // THE INCOMING RADIANT","orbit":"CLUE 3 // THE HELIOCENTRIC ORBIT","origin":"THE BEST-SUPPORTED ORIGIN","outro":"WHAT THE DATA CAN—AND CANNOT—SAY"}
        if t>(5.0 if not QUICK_MODE else 1.22):
            draw_text(img,titles[get_shot(t)["name"]],(56 if not QUICK_MODE else 28,61 if not QUICK_MODE else 30),size=19 if not QUICK_MODE else 9,fill=COLORS["muted"]+(210,),bold=True,stroke=1)

    def draw_source_hud(self,img:Image.Image):
        status="OFFLINE FIXTURE" if self.summary["source_mode"]=="offline-fixture" else "LIVE"
        draw_text(img,f"GMN DATA // {status}",(W-(48 if not QUICK_MODE else 24),72 if not QUICK_MODE else 36),size=17 if not QUICK_MODE else 8,fill=COLORS["cyan"]+(220,),bold=True,anchor="ra",stroke=1)
        draw_text(img,clip_text(self.event.trajectory_id,28),(W-(48 if not QUICK_MODE else 24),102 if not QUICK_MODE else 51),size=14 if not QUICK_MODE else 7,fill=COLORS["muted"]+(195,),anchor="ra",stroke=1)

    def draw_caption(self,img:Image.Image,t:float):
        caption=caption_at(t)
        if not caption:return
        y0=H-(244 if not QUICK_MODE else 124)
        overlay=Image.new("RGBA",SIZE,(0,0,0,0));od=ImageDraw.Draw(overlay)
        od.rounded_rectangle((44 if not QUICK_MODE else 22,y0,W-(44 if not QUICK_MODE else 22),y0+(124 if not QUICK_MODE else 66)),radius=24 if not QUICK_MODE else 12,fill=(2,6,14,172),outline=(80,185,220,65),width=1)
        img.alpha_composite(overlay)
        draw_wrapped_text(img,caption,(68 if not QUICK_MODE else 34,y0+(28 if not QUICK_MODE else 14)),W-(136 if not QUICK_MODE else 68),size=29 if not QUICK_MODE else 14,fill=COLORS["white"]+(245,))

    def draw_hud_noise(self,img:Image.Image,t:float):
        overlay=Image.new("RGBA",SIZE,(0,0,0,0));od=ImageDraw.Draw(overlay)
        offset=int((t*39)%7)
        for y in range(offset,H,7):od.line((0,y,W,y),fill=(120,200,240,10),width=1)
        scan_y=int((t*165)%(H+220))-110;od.rectangle((0,scan_y,W,scan_y+(48 if not QUICK_MODE else 24)),fill=(90,210,240,7));img.alpha_composite(overlay)

    def draw_intro(self,img:Image.Image,t:float):
        cx,cy=W*.5,H*.42
        earth=Image.new("RGBA",SIZE,(0,0,0,0));ed=ImageDraw.Draw(earth)
        r=150*SCALE
        ed.ellipse((cx-r,cy-r,cx+r,cy+r),fill=(20,70,145,255),outline=(150,230,255,180),width=2)
        for lat in [-60,-30,0,30,60]:
            rr=r*math.cos(math.radians(lat));yy=cy-r*math.sin(math.radians(lat));ed.ellipse((cx-rr,yy-rr*.15,cx+rr,yy+rr*.15),outline=(100,190,230,35),width=1)
        img.alpha_composite(earth)
        p=ease_in_out_sine((t-SHOT_PLAN[0]["start"])/max(1e-6,SHOT_PLAN[0]["end"]-SHOT_PLAN[0]["start"]))
        x=lerp(W*.18,W*.62,p);y=lerp(H*.20,H*.44,p)
        trail=Image.new("RGBA",SIZE,(0,0,0,0));td=ImageDraw.Draw(trail)
        for i in range(15):
            f=i/14;tx=lerp(x,x-170*SCALE,f);ty=lerp(y,y-95*SCALE,f);rr=(18-14*f)*SCALE
            td.ellipse((tx-rr,ty-rr,tx+rr,ty+rr),fill=(170,235,255,int(95*(1-f))))
        trail=trail.filter(ImageFilter.GaussianBlur(10 if not QUICK_MODE else 5));img.alpha_composite(trail)
        d=ImageDraw.Draw(img);rr=13*SCALE;d.ellipse((x-rr,y-rr,x+rr,y+rr),fill=COLORS["white"]+(250,),outline=COLORS["gold"]+(240,),width=2)
        draw_text(img,f"{self.event.vgeo_km_s:.1f} km/s" if np.isfinite(self.event.vgeo_km_s) else "SPEED n/a",(int(W*.5),int(H*.66)),size=42 if not QUICK_MODE else 19,fill=COLORS["gold"]+(245,),bold=True,anchor="ma",stroke=1)
        draw_text(img,clip_text(self.event.beginning_utc_time,28),(int(W*.5),int(H*.71)),size=20 if not QUICK_MODE else 9,fill=COLORS["cyan"]+(230,),bold=True,anchor="ma",stroke=1)

    def draw_world_grid(self,img:Image.Image):
        x0,y0,x1,y1=self.world_box;d=ImageDraw.Draw(img)
        d.rounded_rectangle((x0,y0,x1,y1),radius=28 if not QUICK_MODE else 14,fill=(2,6,14,178),outline=(88,185,220,78),width=2)
        for lon in range(-150,181,30):
            x,_=lonlat_xy(lon,0,self.world_box);d.line((x,y0+14,x,y1-14),fill=(90,170,200,38),width=1)
        for lat in range(-60,61,30):
            _,y=lonlat_xy(0,lat,self.world_box);d.line((x0+14,y,x1-14,y),fill=(90,170,200,38),width=1)
        draw_text(img,"MEASURED ATMOSPHERIC TRAJECTORY",(x0+18,y0+18),size=18 if not QUICK_MODE else 8,fill=COLORS["cyan"]+(215,),bold=True,stroke=1)

    def draw_atmosphere(self,img:Image.Image,t:float):
        self.draw_world_grid(img);d=ImageDraw.Draw(img)
        if self.event.has_ground_track:
            xb,yb=lonlat_xy(self.event.lonbeg_e_deg,self.event.latbeg_n_deg,self.world_box);xe,ye=lonlat_xy(self.event.lonend_e_deg,self.event.latend_n_deg,self.world_box)
            p=smoothstep((t-SHOT_PLAN[1]["start"])/max(1e-6,SHOT_PLAN[1]["end"]-SHOT_PLAN[1]["start"]))
            xc,yc=lerp(xb,xe,p),lerp(yb,ye,p)
            d.line((xb,yb,xc,yc),fill=COLORS["gold"]+(245,),width=max(4,int(5*SCALE)))
            for x,y,label,color in [(xb,yb,"BEGIN",COLORS["cyan"]),(xe,ye,"END",COLORS["red"])]:
                rr=10*SCALE;d.ellipse((x-rr,y-rr,x+rr,y+rr),fill=color+(245,),outline=(255,255,255,180),width=1);draw_text(img,label,(int(x),int(y+24*SCALE)),size=15 if not QUICK_MODE else 7,fill=color+(230,),bold=True,anchor="ma",stroke=1)
        cards=[("BEGIN HEIGHT",f"{self.event.htbeg_km:.1f} km" if np.isfinite(self.event.htbeg_km) else "n/a",COLORS["cyan"]),("END HEIGHT",f"{self.event.htend_km:.1f} km" if np.isfinite(self.event.htend_km) else "n/a",COLORS["red"]),("DURATION",f"{self.event.duration_sec:.2f} s" if np.isfinite(self.event.duration_sec) else "n/a",COLORS["violet"]),("STATIONS",str(self.event.num_stations),COLORS["green"])]
        cw=int(W*.19);cy=int(H*.72)
        for i,(label,val,color) in enumerate(cards):
            x=int(W*.07)+i*int(W*.22);ov=Image.new("RGBA",SIZE,(0,0,0,0));od=ImageDraw.Draw(ov);od.rounded_rectangle((x,cy,x+cw,cy+int(H*.095)),radius=20 if not QUICK_MODE else 10,fill=(3,8,17,185),outline=color+(75,),width=2);img.alpha_composite(ov)
            draw_text(img,label,(x+12,cy+13),size=14 if not QUICK_MODE else 7,fill=color+(230,),bold=True,stroke=1);draw_text(img,val,(x+12,cy+44 if not QUICK_MODE else cy+22),size=20 if not QUICK_MODE else 9,fill=COLORS["white"]+(235,),bold=True,stroke=1)

    def draw_radiant(self,img:Image.Image,t:float):
        x0,y0,x1,y1=self.sky_box;d=ImageDraw.Draw(img)
        d.rounded_rectangle((x0,y0,x1,y1),radius=28 if not QUICK_MODE else 14,fill=(2,6,14,178),outline=(88,185,220,78),width=2)
        for ra in range(0,361,30):
            x,_=sky_xy(ra%360,0,self.sky_box);d.line((x,y0+14,x,y1-14),fill=(90,170,200,35),width=1)
        for dec in range(-60,61,30):
            _,y=sky_xy(0,dec,self.sky_box);d.line((x0+14,y,x1-14,y),fill=(90,170,200,35),width=1)
        rng=np.random.default_rng(772)
        for _ in range(180 if QUICK_MODE else 360):
            ra=float(rng.uniform(0,360));dec=float(np.degrees(np.arcsin(rng.uniform(-1,1))));x,y=sky_xy(ra,dec,self.sky_box);rr=float(rng.uniform(.4,1.6)*SCALE);d.ellipse((x-rr,y-rr,x+rr,y+rr),fill=(220,232,255,int(rng.integers(30,125))))
        rx,ry=sky_xy(self.event.rageo_deg,self.event.decgeo_deg,self.sky_box);p=smoothstep((t-SHOT_PLAN[2]["start"])/max(1e-6,SHOT_PLAN[2]["end"]-SHOT_PLAN[2]["start"]))
        for k in range(3):
            rr=(24+18*k)*SCALE*p;d.ellipse((rx-rr,ry-rr,rx+rr,ry+rr),outline=COLORS["gold"]+(170-k*35,),width=2)
        d.line((rx-50*SCALE,ry,rx+50*SCALE,ry),fill=COLORS["gold"]+(220,),width=2);d.line((rx,ry-50*SCALE,rx,ry+50*SCALE),fill=COLORS["gold"]+(220,),width=2)
        draw_text(img,"GEOCENTRIC RADIANT",(x0+18,y0+18),size=18 if not QUICK_MODE else 8,fill=COLORS["cyan"]+(215,),bold=True,stroke=1)
        draw_text(img,f"RA {self.event.rageo_deg:.2f}°",(x0+18,y1-64 if not QUICK_MODE else y1-32),size=20 if not QUICK_MODE else 9,fill=COLORS["white"]+(235,),bold=True,stroke=1)
        draw_text(img,f"DEC {self.event.decgeo_deg:+.2f}°",(x1-18,y1-64 if not QUICK_MODE else y1-32),size=20 if not QUICK_MODE else 9,fill=COLORS["white"]+(235,),bold=True,anchor="ra",stroke=1)
        draw_text(img,"This is the incoming direction—not a physical birthplace.",(W//2,int(H*.76)),size=19 if not QUICK_MODE else 9,fill=COLORS["muted"]+(225,),bold=True,anchor="ma",stroke=1)

    def draw_orbit(self,img:Image.Image,t:float):
        x0,y0,x1,y1=self.orbit_box;d=ImageDraw.Draw(img)
        d.rounded_rectangle((x0,y0,x1,y1),radius=28 if not QUICK_MODE else 14,fill=(2,6,14,170),outline=(88,185,220,78),width=2)
        cx,cy=(x0+x1)/2,(y0+y1)/2
        for name,radius,color in PLANETS:
            if radius>self.orbit_max:continue
            rr=radius/self.orbit_max*min(x1-x0,y1-y0)*.46;d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=(110,150,175,42),width=1)
            if name in {"EARTH","JUPITER"}:draw_text(img,name,(int(cx+rr+8*SCALE),int(cy)),size=14 if not QUICK_MODE else 7,fill=color+(220,),stroke=1)
        sr=11*SCALE;d.ellipse((cx-sr,cy-sr,cx+sr,cy+sr),fill=COLORS["sun"]+(255,),outline=(255,245,210,220),width=1)
        if len(self.orbit_pts):
            pts=[map_top_down(p,self.orbit_box,self.orbit_max) for p in self.orbit_pts]
            d.line(pts+[pts[0]],fill=COLORS["cyan"]+(140,),width=max(2,int(2.3*SCALE)))
            progress=ease_in_out_sine((t-SHOT_PLAN[3]["start"])/max(1e-6,SHOT_PLAN[3]["end"]-SHOT_PLAN[3]["start"]))
            # Travel backward from the encounter anomaly by almost one full orbit.
            E=self.impact_E-progress*math.tau*.88;pos=position_from_eccentric(self.event,E);mx,my=map_top_down(pos,self.orbit_box,self.orbit_max)
            trail=Image.new("RGBA",SIZE,(0,0,0,0));td=ImageDraw.Draw(trail)
            for i in range(10):
                f=i/9;tx=lerp(mx,mx+45*SCALE,f);ty=lerp(my,my-10*SCALE,f);rr=(10-7*f)*SCALE;td.ellipse((tx-rr,ty-rr,tx+rr,ty+rr),fill=(150,232,255,int(60*(1-f))))
            trail=trail.filter(ImageFilter.GaussianBlur(7 if not QUICK_MODE else 4));img.alpha_composite(trail);rr=8*SCALE;d.ellipse((mx-rr,my-rr,mx+rr,my+rr),fill=COLORS["white"]+(250,),outline=COLORS["gold"]+(240,),width=1)
            impact=position_from_eccentric(self.event,self.impact_E);ix,iy=map_top_down(impact,self.orbit_box,self.orbit_max);d.ellipse((ix-10*SCALE,iy-10*SCALE,ix+10*SCALE,iy+10*SCALE),outline=COLORS["red"]+(230,),width=3);draw_text(img,"EARTH INTERSECTION",(int(ix),int(iy+24*SCALE)),size=14 if not QUICK_MODE else 7,fill=COLORS["red"]+(230,),bold=True,anchor="ma",stroke=1)
        draw_text(img,"PUBLISHED PRE-ATMOSPHERIC ORBIT",(x0+18,y0+18),size=18 if not QUICK_MODE else 8,fill=COLORS["cyan"]+(215,),bold=True,stroke=1)
        stats=[("a",f"{self.event.a_au:.2f} AU"),("e",f"{self.event.eccentricity:.3f}"),("i",f"{self.event.inclination_deg:.1f}°"),("q",f"{self.event.perihelion_au:.2f} AU")]
        for idx,(label,val) in enumerate(stats):
            xx=x0+18+idx*int((x1-x0-36)/4);draw_text(img,label.upper(),(xx,y1-66 if not QUICK_MODE else y1-33),size=14 if not QUICK_MODE else 7,fill=COLORS["muted"]+(220,),bold=True,stroke=1);draw_text(img,val,(xx,y1-38 if not QUICK_MODE else y1-19),size=18 if not QUICK_MODE else 8,fill=COLORS["white"]+(235,),bold=True,stroke=1)

    def draw_origin(self,img:Image.Image):
        x0,y0,w=int(W*.08),int(H*.21),int(W*.84);parent=self.event.shower_parent
        ov=Image.new("RGBA",SIZE,(0,0,0,0));od=ImageDraw.Draw(ov);od.rounded_rectangle((x0,y0,x0+w,y0+int(H*.49)),radius=28 if not QUICK_MODE else 14,fill=(2,6,14,180),outline=(88,185,220,78),width=2);img.alpha_composite(ov)
        shower_name=self.summary["shower_name"]
        draw_text(img,"SHOWER ASSOCIATION",(x0+24,y0+26),size=18 if not QUICK_MODE else 8,fill=COLORS["cyan"]+(220,),bold=True,stroke=1)
        draw_text(img,f"{shower_name} [{self.event.shower_code}]",(x0+24,y0+70),size=34 if not QUICK_MODE else 15,fill=COLORS["gold"]+(245,),bold=True,stroke=1)
        if parent:
            draw_text(img,"ESTABLISHED PARENT CONNECTION",(x0+24,y0+132 if not QUICK_MODE else y0+66),size=17 if not QUICK_MODE else 8,fill=COLORS["green"]+(225,),bold=True,stroke=1)
            draw_text(img,parent[1],(x0+24,y0+170 if not QUICK_MODE else y0+85),size=29 if not QUICK_MODE else 13,fill=COLORS["white"]+(240,),bold=True,stroke=1)
            draw_text(img,parent[2],(x0+24,y0+211 if not QUICK_MODE else y0+105),size=18 if not QUICK_MODE else 8,fill=COLORS["muted"]+(225,),stroke=1)
        else:
            message="No conservative parent-body mapping is built into this script for this shower." if self.event.shower_code!="SPORADIC" else "This meteor was classified as sporadic, so no shower parent is claimed."
            draw_wrapped_text(img,message,(x0+24,y0+142 if not QUICK_MODE else y0+71),w-48,size=20 if not QUICK_MODE else 9,fill=COLORS["white"]+(235,))
        hint_title,hint_note=orbital_hint(self.event)
        yy=y0+int(H*.30);d=ImageDraw.Draw(img);d.line((x0+24,yy,x0+w-24,yy),fill=(90,170,200,60),width=1)
        draw_text(img,"DYNAMICAL HINT",(x0+24,yy+22),size=17 if not QUICK_MODE else 8,fill=COLORS["violet"]+(225,),bold=True,stroke=1)
        draw_text(img,hint_title,(x0+24,yy+61 if not QUICK_MODE else yy+30),size=24 if not QUICK_MODE else 11,fill=COLORS["white"]+(240,),bold=True,stroke=1)
        draw_text(img,f"Tisserand-Jupiter = {self.event.tisserand_jupiter:.2f}" if np.isfinite(self.event.tisserand_jupiter) else "Tisserand-Jupiter unavailable",(x0+24,yy+101 if not QUICK_MODE else yy+50),size=18 if not QUICK_MODE else 8,fill=COLORS["muted"]+(225,),stroke=1)
        draw_wrapped_text(img,hint_note+". This is not proof of composition or a unique source body.",(x0+24,yy+137 if not QUICK_MODE else yy+68),w-48,size=17 if not QUICK_MODE else 8,fill=COLORS["muted"]+(220,))

    def draw_outro(self,img:Image.Image):
        x0,y0,w=int(W*.08),int(H*.23),int(W*.84);ov=Image.new("RGBA",SIZE,(0,0,0,0));od=ImageDraw.Draw(ov);od.rounded_rectangle((x0,y0,x0+w,y0+int(H*.42)),radius=28 if not QUICK_MODE else 14,fill=(2,6,14,180),outline=(88,185,220,78),width=2);img.alpha_composite(ov)
        parent=self.event.shower_parent
        answer=(f"MOST LIKELY STREAM // {parent[0]}\nPARENT CONNECTION // {parent[1]}" if parent else (f"STREAM // {self.summary['shower_name']}\nPARENT BODY // NOT SECURELY IDENTIFIED"))
        draw_text(img,"BEST DATA-SUPPORTED ANSWER",(W//2,y0+44),size=24 if not QUICK_MODE else 11,fill=COLORS["cyan"]+(230,),bold=True,anchor="ma",stroke=1)
        draw_wrapped_text(img,answer,(x0+34,y0+100),w-68,size=31 if not QUICK_MODE else 14,fill=COLORS["gold"]+(245,),bold=True,line_spacing=12)
        draw_wrapped_text(img,"The atmospheric path, radiant, and osculating orbit constrain the incoming stream. They do not identify the exact rock's birthplace with absolute certainty.",(x0+34,y0+230),w-68,size=19 if not QUICK_MODE else 9,fill=COLORS["white"]+(235,))

    def render_frame(self,t:float)->np.ndarray:
        img=self.background(t);self.draw_title(img,t);self.draw_source_hud(img);shot=get_shot(t)["name"]
        if shot=="intro":self.draw_intro(img,t)
        elif shot=="atmosphere":self.draw_atmosphere(img,t)
        elif shot=="radiant":self.draw_radiant(img,t)
        elif shot=="orbit":self.draw_orbit(img,t)
        elif shot=="origin":self.draw_origin(img)
        else:self.draw_outro(img)
        self.draw_caption(img,t);self.draw_hud_noise(img,t)
        arr=np.array(img.convert("RGB"));graded=Image.fromarray(arr);graded=ImageEnhance.Contrast(graded).enhance(1.08);graded=ImageEnhance.Color(graded).enhance(1.06);arr=np.array(graded);arr=np.clip(arr.astype(np.float32)*VIGNETTE[...,None],0,255).astype(np.uint8)
        fade_in=smoothstep(t/.9);fade_out=1-smoothstep((t-(CONFIG["duration_s"]-1.1))/1.0)
        return np.clip(arr.astype(np.float32)*fade_in*fade_out,0,255).astype(np.uint8)


# =============================================================================
# Output
# =============================================================================


def render_video(scene:MeteorOriginScene)->Path:
    raw_path=OUTPUT_ROOT/f"{CONFIG['basename']}_raw.mp4";final_path=OUTPUT_ROOT/f"{CONFIG['basename']}_final.mp4";write_srt(OUTPUT_ROOT/f"{CONFIG['basename']}.srt")
    frame_count=int(round(CONFIG["duration_s"]*CONFIG["fps"]))
    with iio.get_writer(raw_path,fps=CONFIG["fps"],codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=None) as writer:
        for frame_index in tqdm(range(frame_count),desc="Rendering meteor origin short"):writer.append_data(scene.render_frame(frame_index/CONFIG["fps"]))
    shutil.copyfile(raw_path,final_path);return final_path


def make_contact_sheet(paths:Sequence[Path],out_path:Path):
    thumbs=[]
    for path in paths[:6]:
        image=Image.open(path).convert("RGB").resize((270,480));draw=ImageDraw.Draw(image);draw.rectangle((8,8,120,38),fill=(0,0,0));draw.text((18,13),path.stem.replace("preview_",""),fill=(255,255,255));thumbs.append(image)
    sheet=Image.new("RGB",(600,1520),(8,11,18))
    for index,thumb in enumerate(thumbs):row,col=divmod(index,2);sheet.paste(thumb,(20+col*290,20+row*500))
    sheet.save(out_path,quality=92)


