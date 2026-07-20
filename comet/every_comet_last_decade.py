from __future__ import annotations

"""
Every Known Comet Orbiting the Sun
==================================

A cinematic vertical YouTube Short renderer built from orbital solutions in
NASA/JPL's Small-Body Database (SBDB).

What the video shows
--------------------
- Every comet record returned by the JPL SBDB Query API when the script runs.
- Bound elliptical comet orbits, comet fragments, and the catalog's parabolic
  and hyperbolic/open solutions.
- A logarithmically compressed deep-solar-system view, an inner-system view,
  orbital-family statistics, inclination structure, perihelion distances, and
  selected famous comets found in the live catalog.



Honesty / interpretation rules
------------------------------
- "Every known comet" means every comet record returned by JPL SBDB at runtime.
  The catalog changes as discoveries and orbit solutions are updated.
- Fragments are included by default because they are separate SBDB comet
  records. Set COMET_SHORT_EXCLUDE_FRAGMENTS=1 to omit them.
- The curves are osculating two-body conics reconstructed from the published
  elements. They are not full n-body integrations and are not live ephemerides.
- Bound-comet position dots are approximate two-body propagation to the render
  timestamp. Open/parabolic trajectories are drawn as paths without a claimed
  current position.
- Deep-view radial distance is logarithmically compressed. Very large and open
  trajectories are clipped at the labeled plotting radius.
- Some SBDB records can lack enough published elements to draw a curve; the
  video reports catalog count and drawable count separately.

Offline fallback
----------------
If JPL cannot be reached, a small, clearly labeled layout fixture of well-known
comets is used. It is not presented as the complete current catalog.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg requests tqdm

Run final quality
-----------------
    python every_known_comet_orbiting_the_sun_short.py

Run quick preview
-----------------
    COMET_SHORT_QUICK=1 python every_known_comet_orbiting_the_sun_short.py

Force live refresh
------------------
    COMET_SHORT_REFRESH=1 python every_known_comet_orbiting_the_sun_short.py

Force offline layout testing
----------------------------
    COMET_SHORT_OFFLINE=1 COMET_SHORT_QUICK=1 \
        python every_known_comet_orbiting_the_sun_short.py
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

QUICK_MODE = os.environ.get("COMET_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("COMET_SHORT_OFFLINE", "0") == "1"
REFRESH = os.environ.get("COMET_SHORT_REFRESH", "0") == "1"
EXCLUDE_FRAGMENTS = os.environ.get("COMET_SHORT_EXCLUDE_FRAGMENTS", "0") == "1"

OUTPUT_ROOT = Path("every_known_comet_orbiting_the_sun_output")
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
    "basename": "every_known_comet_orbiting_the_sun",
    "title": "EVERY KNOWN COMET ORBITING THE SUN",
    "subtitle": "Orbital solutions from NASA/JPL's Small-Body Database",
    "timeout_s": 60,
    "api_url": "https://ssd-api.jpl.nasa.gov/sbdb_query.api",
    "source_url": "https://ssd.jpl.nasa.gov/sb/orbits.html",
    "api_docs": "https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html",
    "stars": 760,
    "orbit_samples": 80 if QUICK_MODE else 128,
    "max_position_dots": 1500 if QUICK_MODE else 5000,
    "cache_hours": 18,
}

FIELDS = [
    "spkid", "full_name", "kind", "prefix", "class", "epoch", "e", "a", "q",
    "i", "om", "w", "ma", "tp", "per", "ad", "t_jup", "moid",
    "condition_code", "data_arc", "n_obs_used", "soln_date", "source",
    "M1", "M2", "diameter",
]

W = CONFIG["width"]
H = CONFIG["height"]
SIZE = (W, H)
SCALE = W / 1080.0

COLORS = {
    "ETc": (93, 226, 255),
    "JFc": (86, 166, 255),
    "JFC": (86, 166, 255),
    "CTc": (103, 255, 181),
    "HTC": (194, 111, 255),
    "PAR": (255, 218, 105),
    "HYP": (255, 154, 72),
    "COM": (212, 226, 244),
    "OTHER": (165, 192, 218),
    "sun": (255, 205, 79),
    "earth": (92, 176, 255),
    "jupiter": (232, 185, 124),
    "neptune": (93, 123, 255),
    "white": (246, 249, 255),
    "muted": (152, 199, 222),
    "red": (255, 102, 112),
}

CLASS_LABELS = {
    "ETc": "ENCKE-TYPE",
    "JFc": "JUPITER-FAMILY",
    "JFC": "JUPITER-FAMILY*",
    "CTc": "CHIRON-TYPE",
    "HTC": "HALLEY-TYPE",
    "PAR": "PARABOLIC",
    "HYP": "HYPERBOLIC",
    "COM": "OTHER COMET",
    "OTHER": "OTHER / UNCLASSIFIED",
}

PLANETS = [
    ("MERCURY", 0.387, (180, 188, 198)),
    ("VENUS", 0.723, (232, 197, 130)),
    ("EARTH", 1.000, COLORS["earth"]),
    ("MARS", 1.524, (231, 126, 87)),
    ("JUPITER", 5.203, COLORS["jupiter"]),
    ("SATURN", 9.537, (225, 204, 142)),
    ("URANUS", 19.191, (116, 218, 230)),
    ("NEPTUNE", 30.069, COLORS["neptune"]),
]

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.0 if not QUICK_MODE else 1.5},
    {"name": "deep", "start": 7.0 if not QUICK_MODE else 1.5, "end": 18.0 if not QUICK_MODE else 3.7},
    {"name": "inner", "start": 18.0 if not QUICK_MODE else 3.7, "end": 30.0 if not QUICK_MODE else 6.1},
    {"name": "families", "start": 30.0 if not QUICK_MODE else 6.1, "end": 40.0 if not QUICK_MODE else 8.2},
    {"name": "inclination", "start": 40.0 if not QUICK_MODE else 8.2, "end": 48.0 if not QUICK_MODE else 9.8},
    {"name": "highlights", "start": 48.0 if not QUICK_MODE else 9.8, "end": 55.0 if not QUICK_MODE else 11.2},
    {"name": "outro", "start": 55.0 if not QUICK_MODE else 11.2, "end": CONFIG["duration_s"]},
]


# =============================================================================
# Data model
# =============================================================================

@dataclass
class CometOrbit:
    spkid: str
    full_name: str
    kind: str
    prefix: str
    orbit_class: str
    epoch_jd: float
    eccentricity: float
    semimajor_axis_au: float
    perihelion_au: float
    inclination_deg: float
    node_deg: float
    arg_peri_deg: float
    mean_anomaly_deg: float
    perihelion_time_jd: float
    period_days: float
    aphelion_au: float
    tisserand_jupiter: float
    earth_moid_au: float
    condition_code: str
    data_arc_days: float
    observations_used: int
    solution_date: str
    orbit_source: str
    total_mag_parameter: float
    total_mag_slope: float
    diameter_km: float

    @property
    def family(self) -> str:
        return self.orbit_class if self.orbit_class in CLASS_LABELS else "OTHER"

    @property
    def is_bound(self) -> bool:
        return (
            np.isfinite(self.eccentricity)
            and self.eccentricity < 1.0
            and np.isfinite(self.semimajor_axis_au)
            and self.semimajor_axis_au > 0.0
        )

    @property
    def is_open(self) -> bool:
        return np.isfinite(self.eccentricity) and self.eccentricity >= 1.0

    @property
    def is_fragment(self) -> bool:
        text = self.full_name.upper()
        return "FRAGMENT" in text or bool(__import__("re").search(r"(?:^|[ /-])FRAG(?:MENT)?(?:$|[ /-])", text))

    @property
    def drawable(self) -> bool:
        needed = [self.eccentricity, self.perihelion_au, self.inclination_deg, self.node_deg, self.arg_peri_deg]
        return all(np.isfinite(needed)) and self.perihelion_au > 0

    @property
    def period_years(self) -> float:
        return self.period_days / 365.25 if np.isfinite(self.period_days) else np.nan


# =============================================================================
# Generic helpers
# =============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def julian_date(dt: datetime) -> float:
    return dt.timestamp() / 86400.0 + 2440587.5


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def clip_text(text: str, length: int = 35) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= length else text[: length - 1] + "…"


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


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def make_vignette(width: int, height: int, strength: float = 0.26) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1 - strength * radius**1.8, 0, 1).astype(np.float32)


VIGNETTE = make_vignette(W, H)


# =============================================================================
# Data collection
# =============================================================================

def row_to_comet(row: Dict[str, object]) -> CometOrbit:
    return CometOrbit(
        spkid=str(row.get("spkid") or ""),
        full_name=str(row.get("full_name") or "Unnamed comet").strip(),
        kind=str(row.get("kind") or ""),
        prefix=str(row.get("prefix") or ""),
        orbit_class=str(row.get("class") or "OTHER"),
        epoch_jd=safe_float(row.get("epoch")),
        eccentricity=safe_float(row.get("e")),
        semimajor_axis_au=safe_float(row.get("a")),
        perihelion_au=safe_float(row.get("q")),
        inclination_deg=safe_float(row.get("i")),
        node_deg=safe_float(row.get("om")),
        arg_peri_deg=safe_float(row.get("w")),
        mean_anomaly_deg=safe_float(row.get("ma")),
        perihelion_time_jd=safe_float(row.get("tp")),
        period_days=safe_float(row.get("per")),
        aphelion_au=safe_float(row.get("ad")),
        tisserand_jupiter=safe_float(row.get("t_jup")),
        earth_moid_au=safe_float(row.get("moid")),
        condition_code=str(row.get("condition_code") or ""),
        data_arc_days=safe_float(row.get("data_arc")),
        observations_used=safe_int(row.get("n_obs_used")),
        solution_date=str(row.get("soln_date") or ""),
        orbit_source=str(row.get("source") or ""),
        total_mag_parameter=safe_float(row.get("M1")),
        total_mag_slope=safe_float(row.get("M2")),
        diameter_km=safe_float(row.get("diameter")),
    )


def parse_api_payload(payload: Dict) -> Tuple[List[CometOrbit], int]:
    fields = payload.get("fields") or []
    data = payload.get("data") or []
    if not fields or not isinstance(data, list):
        raise RuntimeError("JPL response did not contain fields/data arrays")
    records = []
    for values in data:
        row = dict(zip(fields, values))
        records.append(row_to_comet(row))
    count = safe_int(payload.get("count"), len(records))
    return records, count


def fetch_jpl_comets() -> Tuple[List[CometOrbit], Dict]:
    cache_path = CACHE_ROOT / ("jpl_comets_no_fragments.json" if EXCLUDE_FRAGMENTS else "jpl_comets_all.json")
    payload: Optional[Dict] = None
    source_mode = "live"

    if cache_path.exists() and not REFRESH:
        age_hours = (utc_now().timestamp() - cache_path.stat().st_mtime) / 3600.0
        if age_hours <= CONFIG["cache_hours"]:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            source_mode = "cache"

    if payload is None:
        if requests is None:
            raise RuntimeError("requests is unavailable")
        params = {
            "fields": ",".join(FIELDS),
            "sb-kind": "c",
            "full-prec": "1",
            "sort": "full_name",
        }
        if EXCLUDE_FRAGMENTS:
            params["sb-xfrag"] = "1"
        response = requests.get(
            CONFIG["api_url"],
            params=params,
            timeout=CONFIG["timeout_s"],
            headers={"User-Agent": "EveryKnownCometShort/1.0 educational renderer"},
        )
        response.raise_for_status()
        payload = response.json()
        cache_path.write_text(json.dumps(payload), encoding="utf-8")

    comets, catalog_count = parse_api_payload(payload)
    summary = summarize_catalog(comets, catalog_count)
    summary.update({
        "generated_at_utc": iso_z(utc_now()),
        "data_status": source_mode,
        "source_url": CONFIG["api_url"],
        "fragments_excluded": EXCLUDE_FRAGMENTS,
        "offline_fixture": False,
    })
    return comets, summary


def fixture_comets() -> List[CometOrbit]:
    # Approximate elements used only for offline layout testing. This fixture is
    # deliberately small and never described as the current complete catalog.
    raw = [
        ("1000036", "1P/Halley", "cn", "P", "HTC", 17.834, .96714, .58598, 162.26, 58.42, 111.33, 30.0),
        ("1000025", "2P/Encke", "cn", "P", "ETc", 2.216, .8483, .336, 11.78, 334.57, 186.54, 90.0),
        ("1000012", "12P/Pons-Brooks", "cn", "P", "HTC", 17.14, .9546, .780, 74.19, 255.86, 199.03, 150.0),
        ("1000067", "67P/Churyumov-Gerasimenko", "cn", "P", "JFC", 3.464, .641, 1.243, 7.04, 50.15, 12.78, 210.0),
        ("1000096", "96P/Machholz 1", "cn", "P", "ETc", 3.03, .959, .124, 58.31, 94.32, 14.76, 250.0),
        ("1000109", "109P/Swift-Tuttle", "cn", "P", "HTC", 26.1, .963, .959, 113.45, 139.38, 152.98, 310.0),
        ("1000198", "C/1995 O1 (Hale-Bopp)", "cu", "C", "COM", 186.0, .9951, .914, 89.43, 282.47, 130.59, 20.0),
        ("1000200", "C/2020 F3 (NEOWISE)", "cu", "C", "COM", 358.0, .9992, .295, 128.94, 61.01, 37.28, 60.0),
        ("1000201", "C/2017 K2 (PANSTARRS)", "cu", "C", "COM", 1600.0, .9997, 1.80, 87.56, 88.18, 236.2, 110.0),
        ("1000202", "C/2014 UN271 (Bernardinelli-Bernstein)", "cu", "C", "COM", 10000.0, .9995, 10.95, 95.47, 190.0, 326.0, 180.0),
        ("1000203", "C/1980 E1 (Bowell)", "cu", "C", "HYP", -2500.0, 1.057, 3.36, 1.66, 114.6, 135.1, np.nan),
        ("1000204", "C/2023 A3 (Tsuchinshan-ATLAS)", "cu", "C", "COM", 500.0, .998, .391, 139.1, 21.6, 308.5, 275.0),
        ("1000205", "3D/Biela", "cn", "D", "JFC", 3.53, .751, .879, 13.2, 250.7, 221.7, 15.0),
        ("1000206", "73P/Schwassmann-Wachmann 3 fragment B", "cn", "P", "JFC", 3.09, .694, .939, 11.4, 69.9, 198.8, 340.0),
        ("1000207", "C/2025 X1 fixture parabolic", "cu", "C", "PAR", np.nan, 1.0, .72, 52.0, 140.0, 70.0, np.nan),
    ]
    now_jd = julian_date(utc_now())
    comets = []
    for spkid, name, kind, prefix, cls, a, e, q, inc, node, arg, ma in raw:
        if EXCLUDE_FRAGMENTS and "fragment" in name.lower():
            continue
        period = 365.25 * (a ** 1.5) if a > 0 and e < 1 else np.nan
        aphelion = a * (1 + e) if a > 0 and e < 1 else np.nan
        comets.append(CometOrbit(
            spkid=spkid,
            full_name=name,
            kind=kind,
            prefix=prefix,
            orbit_class=cls,
            epoch_jd=now_jd,
            eccentricity=e,
            semimajor_axis_au=a,
            perihelion_au=q,
            inclination_deg=inc,
            node_deg=node,
            arg_peri_deg=arg,
            mean_anomaly_deg=ma,
            perihelion_time_jd=np.nan,
            period_days=period,
            aphelion_au=aphelion,
            tisserand_jupiter=np.nan,
            earth_moid_au=np.nan,
            condition_code="fixture",
            data_arc_days=np.nan,
            observations_used=0,
            solution_date="offline layout fixture",
            orbit_source="offline fixture",
            total_mag_parameter=np.nan,
            total_mag_slope=np.nan,
            diameter_km=np.nan,
        ))
    return comets


def summarize_catalog(comets: Sequence[CometOrbit], catalog_count: int) -> Dict:
    drawable = [c for c in comets if c.drawable]
    bound = [c for c in drawable if c.is_bound]
    open_orbits = [c for c in drawable if c.is_open]
    fragments = [c for c in comets if c.is_fragment]
    class_counts: Dict[str, int] = {}
    for comet in comets:
        class_counts[comet.family] = class_counts.get(comet.family, 0) + 1

    perihelia = finite(c.perihelion_au for c in drawable)
    inclinations = finite(c.inclination_deg for c in drawable)
    periods = finite(c.period_years for c in bound)
    aphelia = finite(c.aphelion_au for c in bound)

    nearest = min(drawable, key=lambda c: c.perihelion_au) if drawable else None
    longest = max(bound, key=lambda c: c.period_years if np.isfinite(c.period_years) else -1) if bound else None
    highest_i = max(drawable, key=lambda c: c.inclination_deg) if drawable else None

    return {
        "catalog_count": int(catalog_count),
        "records_received": len(comets),
        "drawable_orbits": len(drawable),
        "bound_elliptical": len(bound),
        "open_or_parabolic": len(open_orbits),
        "fragment_records": len(fragments),
        "class_counts": class_counts,
        "median_perihelion_au": float(np.median(perihelia)) if len(perihelia) else None,
        "median_inclination_deg": float(np.median(inclinations)) if len(inclinations) else None,
        "median_bound_period_years": float(np.median(periods)) if len(periods) else None,
        "deep_plot_radius_au": choose_deep_radius(aphelia),
        "nearest_perihelion": asdict(nearest) if nearest else None,
        "longest_bound_period": asdict(longest) if longest else None,
        "highest_inclination": asdict(highest_i) if highest_i else None,
    }


def choose_deep_radius(aphelia: np.ndarray) -> float:
    if len(aphelia) == 0:
        return 1000.0
    positive = aphelia[aphelia > 0]
    if len(positive) == 0:
        return 1000.0
    radius = float(np.percentile(positive, 99.2))
    choices = [100, 300, 1000, 3000, 10000, 30000, 100000]
    for choice in choices:
        if radius <= choice:
            return float(choice)
    return 100000.0


def collect_catalog() -> Tuple[List[CometOrbit], Dict]:
    if not OFFLINE_MODE:
        try:
            return fetch_jpl_comets()
        except Exception as exc:
            error = str(exc)
    else:
        error = "Live request skipped by COMET_SHORT_OFFLINE=1"

    comets = fixture_comets()
    summary = summarize_catalog(comets, len(comets))
    summary.update({
        "generated_at_utc": iso_z(utc_now()),
        "data_status": "offline fixture",
        "source_url": CONFIG["api_url"],
        "fragments_excluded": EXCLUDE_FRAGMENTS,
        "offline_fixture": True,
        "live_error": error,
    })
    return comets, summary


def save_data(comets: Sequence[CometOrbit], summary: Dict) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "jpl_comet_orbits_snapshot.csv"
    json_path = DATA_ROOT / "jpl_comet_orbits_summary.json"
    pd.DataFrame([asdict(c) for c in comets]).to_csv(csv_path, index=False)
    payload = {"summary": summary, "comets": [asdict(c) for c in comets]}
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    return csv_path, json_path


# =============================================================================
# Orbital geometry
# =============================================================================

def rotation_matrix(node_deg: float, inc_deg: float, arg_deg: float) -> np.ndarray:
    node, inc, arg = np.radians([node_deg, inc_deg, arg_deg])
    co, so = math.cos(node), math.sin(node)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(arg), math.sin(arg)
    return np.array([
        [co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si],
        [so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si],
        [sw * si, cw * si, ci],
    ], dtype=float)


def orbit_xyz(comet: CometOrbit, samples: int, radius_limit: float) -> np.ndarray:
    if not comet.drawable:
        return np.empty((0, 3), dtype=float)
    e = comet.eccentricity
    q = comet.perihelion_au
    rot = rotation_matrix(comet.node_deg, comet.inclination_deg, comet.arg_peri_deg)

    if e < 1.0:
        a = comet.semimajor_axis_au
        if not np.isfinite(a) or a <= 0:
            a = q / max(1e-8, 1 - e)
        E = np.linspace(0, math.tau, samples, endpoint=True)
        xp = a * (np.cos(E) - e)
        yp = a * math.sqrt(max(0.0, 1 - e * e)) * np.sin(E)
    else:
        if e > 1.0:
            limit = math.acos(clamp(-1.0 / e, -1.0, 1.0)) - 0.008
        else:
            limit = math.radians(168.0)
        nu = np.linspace(-limit, limit, samples)
        p = q * (1.0 + e)
        denom = 1.0 + e * np.cos(nu)
        denom = np.where(np.abs(denom) < 1e-8, np.nan, denom)
        r = p / denom
        xp = r * np.cos(nu)
        yp = r * np.sin(nu)

    pts = np.vstack([xp, yp, np.zeros_like(xp)]).T @ rot.T
    radii = np.linalg.norm(pts, axis=1)
    pts[(~np.isfinite(radii)) | (radii > radius_limit * 1.05)] = np.nan
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
        if abs(step) < 1e-11:
            break
    return E


def approximate_position_xyz(comet: CometOrbit, target_jd: float) -> Optional[np.ndarray]:
    if not comet.is_bound or not comet.drawable:
        return None
    a = comet.semimajor_axis_au
    e = comet.eccentricity
    if not np.isfinite(a) or a <= 0:
        return None

    n_deg_day = 0.9856076686 / (a ** 1.5)
    if np.isfinite(comet.mean_anomaly_deg) and np.isfinite(comet.epoch_jd):
        M_deg = comet.mean_anomaly_deg + n_deg_day * (target_jd - comet.epoch_jd)
    elif np.isfinite(comet.perihelion_time_jd):
        M_deg = n_deg_day * (target_jd - comet.perihelion_time_jd)
    else:
        return None

    E = solve_kepler_elliptic(math.radians(M_deg), e)
    xp = a * (math.cos(E) - e)
    yp = a * math.sqrt(max(0.0, 1 - e * e)) * math.sin(E)
    rot = rotation_matrix(comet.node_deg, comet.inclination_deg, comet.arg_peri_deg)
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


def map_side(point: Sequence[float], box: Tuple[int, int, int, int], max_au: float) -> Tuple[float, float]:
    x, z = float(point[0]), float(point[2])
    radius = math.hypot(x, z)
    if radius == 0:
        ux, uz = 0.0, 0.0
    else:
        radial = compressed_radius(radius, max_au)
        ux, uz = x / radius * radial, z / radius * radial
    x0, y0, x1, y1 = box
    sx = (x1 - x0) * 0.47
    sy = (y1 - y0) * 0.42
    return (x0 + x1) / 2 + ux * sx, (y0 + y1) / 2 - uz * sy


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

class CometScene:
    def __init__(self, comets: Sequence[CometOrbit], summary: Dict):
        self.comets = list(comets)
        self.summary = summary
        self.drawable = [c for c in self.comets if c.drawable]
        self.bound = [c for c in self.drawable if c.is_bound]
        self.render_jd = julian_date(utc_now())
        self.deep_radius = float(summary.get("deep_plot_radius_au") or 1000.0)
        self.deep_box = (int(W * .045), int(H * .17), int(W * .955), int(H * .73))
        self.inner_box = self.deep_box
        self.side_box = (int(W * .055), int(H * .22), int(W * .945), int(H * .69))
        self.stars = self._make_stars(CONFIG["stars"], seed=9351)
        self.deep_layer = self._build_orbit_layer(self.deep_box, self.deep_radius, log_scale=True, side=False)
        self.inner_layer = self._build_orbit_layer(self.inner_box, 35.0, log_scale=False, side=False)
        self.side_layer = self._build_orbit_layer(self.side_box, self.deep_radius, log_scale=True, side=True)
        self.position_dots_deep = self._build_position_layer(self.deep_box, self.deep_radius, True)
        self.position_dots_inner = self._build_position_layer(self.inner_box, 35.0, False)
        self.highlights = self._find_highlights()
        self.captions = self._make_captions()

    @staticmethod
    def _make_stars(n: int, seed: int):
        rng = np.random.default_rng(seed)
        return [
            (float(rng.uniform(0, W)), float(rng.uniform(0, H)), float(rng.uniform(.4, 2.0) * SCALE),
             int(rng.integers(25, 145)), float(rng.uniform(0, math.tau)))
            for _ in range(n)
        ]

    def _make_captions(self) -> List[Tuple[float, float, str]]:
        count = int(self.summary["catalog_count"])
        drawable = int(self.summary["drawable_orbits"])
        bound = int(self.summary["bound_elliptical"])
        open_count = int(self.summary["open_or_parabolic"])
        if self.summary.get("offline_fixture"):
            first = "Offline layout fixture: this sample is not the full current comet catalog."
        else:
            first = f"JPL returned {count:,} known comet records when this visualization was generated."
        return [
            (.4, 7.0, first),
            (7.1, 18.0, f"{drawable:,} records contain enough orbital elements to draw an osculating conic."),
            (18.1, 30.0, "Inside Neptune's orbit, thousands of paths overlap—many are strongly shaped by Jupiter."),
            (30.1, 40.0, "Jupiter-family, Halley-type, Encke-type, Chiron-type, parabolic, and hyperbolic solutions occupy different regimes."),
            (40.1, 48.0, "Comet orbits are not confined to the planetary plane; some travel backwards on steep retrograde paths."),
            (48.1, 55.0, f"The catalog includes {bound:,} bound ellipses and {open_count:,} parabolic or hyperbolic solutions in this snapshot."),
            (55.1, 57.7, "These are orbital solutions, not glowing tails—and the catalog changes whenever discoveries and observations update the orbits."),
        ]

    def _build_orbit_layer(self, box, max_au: float, log_scale: bool, side: bool) -> Image.Image:
        layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        samples = CONFIG["orbit_samples"]
        for comet in self.drawable:
            xyz = orbit_xyz(comet, samples=samples, radius_limit=max_au)
            if len(xyz) < 2:
                continue
            mapped = []
            for point in xyz:
                if not np.isfinite(point).all():
                    mapped.append((np.nan, np.nan))
                else:
                    mapped.append(map_side(point, box, max_au) if side else map_top_down(point, box, max_au, log_scale))
            alpha = 30 if not QUICK_MODE else 42
            if comet.family in {"PAR", "HYP"}:
                alpha = 52 if not QUICK_MODE else 62
            color = COLORS.get(comet.family, COLORS["OTHER"])
            for chunk in split_valid_polyline(np.asarray(mapped, dtype=float)):
                draw.line(chunk, fill=color + (alpha,), width=max(1, int(SCALE)))
        return layer

    def _build_position_layer(self, box, max_au: float, log_scale: bool) -> Image.Image:
        layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        candidates = []
        for comet in self.bound:
            pos = approximate_position_xyz(comet, self.render_jd)
            if pos is None:
                continue
            radius = float(np.linalg.norm(pos))
            if radius <= max_au:
                candidates.append((comet, pos))
        if len(candidates) > CONFIG["max_position_dots"]:
            rng = np.random.default_rng(441)
            chosen = rng.choice(len(candidates), CONFIG["max_position_dots"], replace=False)
            candidates = [candidates[int(i)] for i in chosen]
        for comet, pos in candidates:
            x, y = map_top_down(pos, box, max_au, log_scale)
            r = max(1.0, 1.5 * SCALE)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=COLORS.get(comet.family, COLORS["OTHER"]) + (150,))
        return layer

    def _find_highlights(self) -> List[CometOrbit]:
        terms = ["HALLEY", "ENCKE", "CHURYUMOV", "HALE-BOPP", "NEOWISE", "PONS-BROOKS", "SWIFT-TUTTLE"]
        found = []
        used = set()
        for term in terms:
            match = next((c for c in self.drawable if term in c.full_name.upper()), None)
            if match and match.spkid not in used:
                found.append(match)
                used.add(match.spkid)
        if len(found) < 5:
            extras = sorted(self.bound, key=lambda c: c.perihelion_au)[: 5 - len(found)]
            for comet in extras:
                if comet.spkid not in used:
                    found.append(comet)
                    used.add(comet.spkid)
        return found[:7]

    def caption_at(self, t: float) -> Optional[str]:
        for start, end, text in self.captions:
            if start <= t < end:
                return text
        return None

    def background(self, t: float) -> Image.Image:
        img = Image.new("RGBA", SIZE, (2, 5, 12, 255))
        nebula = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        nd = ImageDraw.Draw(nebula)
        clouds = [
            (W * .22, H * .30, (28, 88, 125)),
            (W * .76, H * .38, (88, 36, 120)),
            (W * .45, H * .77, (20, 76, 92)),
        ]
        for cx, cy, color in clouds:
            for radius, alpha in [(W * .5, 10), (W * .30, 20), (W * .15, 29)]:
                nd.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=color + (alpha,))
        img.alpha_composite(nebula.filter(ImageFilter.GaussianBlur(70 if not QUICK_MODE else 35)))
        d = ImageDraw.Draw(img)
        for x, y, radius, alpha, phase in self.stars:
            a = int(alpha * (.72 + .28 * math.sin(t * 1.7 + phase)))
            d.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(218, 232, 255, a))
        return img

    def draw_header(self, img: Image.Image, t: float, section: str):
        intro_alpha = int(255 * smoothstep((t - .1) / .8) * (1 - smoothstep((t - (6.2 if not QUICK_MODE else 1.25)) / .7)))
        if intro_alpha > 3:
            draw_text(img, "EVERY KNOWN COMET", (54 if not QUICK_MODE else 27, 82 if not QUICK_MODE else 41),
                      size=38 if not QUICK_MODE else 18, fill=(246, 249, 255, intro_alpha), bold=True)
            draw_text(img, "ORBITING THE SUN", (54 if not QUICK_MODE else 27, 130 if not QUICK_MODE else 65),
                      size=38 if not QUICK_MODE else 18, fill=(246, 249, 255, intro_alpha), bold=True)
            draw_text(img, CONFIG["subtitle"], (56 if not QUICK_MODE else 28, 180 if not QUICK_MODE else 90),
                      size=20 if not QUICK_MODE else 9, fill=(104, 226, 245, min(230, intro_alpha)), bold=True)
        if t > (5.0 if not QUICK_MODE else 1.15):
            draw_text(img, section, (54 if not QUICK_MODE else 27, 60 if not QUICK_MODE else 30),
                      size=19 if not QUICK_MODE else 9, fill=(145, 211, 234, 220), bold=True, stroke=1)

        label = "OFFLINE FIXTURE" if self.summary.get("offline_fixture") else "JPL SBDB LIVE/CACHED SNAPSHOT"
        color = COLORS["red"] if self.summary.get("offline_fixture") else COLORS["ETc"]
        draw_text(img, label, (W - int(45*SCALE), int(66*SCALE)), size=int(16*SCALE),
                  fill=color + (225,), bold=True, anchor="ra", stroke=1)
        draw_text(img, self.summary["generated_at_utc"].replace("T", " ").replace("Z", " UTC"),
                  (W-int(45*SCALE), int(96*SCALE)), size=int(13*SCALE),
                  fill=(156, 199, 220, 195), anchor="ra", stroke=1)

    def draw_sun(self, img: Image.Image, center: Tuple[float, float], radius: float):
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        cx, cy = center
        for mult, alpha in [(5.0, 12), (3.2, 20), (2.0, 42)]:
            r = radius * mult
            gd.ellipse((cx-r, cy-r, cx+r, cy+r), fill=COLORS["sun"] + (alpha,))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4, int(radius * 1.4)))))
        d = ImageDraw.Draw(img)
        d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=COLORS["sun"] + (255,), outline=(255, 244, 187, 235), width=max(1, int(2*SCALE)))

    def draw_planet_rings(self, img: Image.Image, box, max_au: float, log_scale: bool, label_planets: bool = True):
        d = ImageDraw.Draw(img)
        cx = (box[0]+box[2])/2
        cy = (box[1]+box[3])/2
        base = min(box[2]-box[0], box[3]-box[1]) * .47
        for name, radius_au, color in PLANETS:
            if radius_au > max_au:
                continue
            frac = compressed_radius(radius_au, max_au) if log_scale else radius_au/max_au
            radius = base * frac
            d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=color + (58,), width=max(1, int(SCALE)))
            if label_planets and name in {"EARTH", "JUPITER", "NEPTUNE"}:
                draw_text(img, name, (int(cx+radius+6*SCALE), int(cy)), size=int(12*SCALE), fill=color + (190,), stroke=1)

    def draw_intro(self, img: Image.Image, t: float):
        box = (int(W*.08), int(H*.20), int(W*.92), int(H*.70))
        self.draw_planet_rings(img, box, 35.0, False, False)
        img.alpha_composite(self.inner_layer)
        img.alpha_composite(self.position_dots_inner)
        self.draw_sun(img, ((box[0]+box[2])/2, (box[1]+box[3])/2), 12*SCALE)
        count = int(self.summary["catalog_count"])
        draw_text(img, f"{count:,}", (W//2, int(H*.73)), size=int(68*SCALE), fill=COLORS["white"]+(250,), bold=True, anchor="ma")
        draw_text(img, "COMET RECORDS IN THIS SNAPSHOT", (W//2, int(H*.785)), size=int(22*SCALE), fill=COLORS["ETc"]+(230,), bold=True, anchor="ma")
        if self.summary.get("offline_fixture"):
            draw_text(img, "LAYOUT SAMPLE — NOT THE COMPLETE CATALOG", (W//2, int(H*.825)), size=int(17*SCALE), fill=COLORS["red"]+(235,), bold=True, anchor="ma")

    def draw_deep(self, img: Image.Image):
        box = self.deep_box
        self.draw_planet_rings(img, box, self.deep_radius, True, True)
        img.alpha_composite(self.deep_layer)
        img.alpha_composite(self.position_dots_deep)
        self.draw_sun(img, ((box[0]+box[2])/2, (box[1]+box[3])/2), 10*SCALE)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle(box, radius=int(24*SCALE), outline=(84, 178, 213, 70), width=max(1, int(2*SCALE)))
        draw_text(img, f"LOG-COMPRESSED RADIUS • CLIPPED AT {self.deep_radius:,.0f} AU", (box[0]+int(18*SCALE), box[1]+int(18*SCALE)),
                  size=int(16*SCALE), fill=(151, 208, 231, 220), bold=True, stroke=1)
        self.draw_stat_strip(img, [
            ("CATALOG", f"{self.summary['catalog_count']:,}"),
            ("DRAWABLE", f"{self.summary['drawable_orbits']:,}"),
            ("FRAGMENTS", f"{self.summary['fragment_records']:,}"),
        ])

    def draw_inner(self, img: Image.Image):
        box = self.inner_box
        self.draw_planet_rings(img, box, 35.0, False, True)
        img.alpha_composite(self.inner_layer)
        img.alpha_composite(self.position_dots_inner)
        self.draw_sun(img, ((box[0]+box[2])/2, (box[1]+box[3])/2), 13*SCALE)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle(box, radius=int(24*SCALE), outline=(84, 178, 213, 70), width=max(1, int(2*SCALE)))
        draw_text(img, "INNER SOLAR SYSTEM • LINEAR SCALE TO 35 AU", (box[0]+int(18*SCALE), box[1]+int(18*SCALE)),
                  size=int(16*SCALE), fill=(151, 208, 231, 220), bold=True, stroke=1)
        nearest = self.summary.get("nearest_perihelion") or {}
        draw_text(img, "CLOSEST PUBLISHED PERIHELION", (int(W*.08), int(H*.765)), size=int(16*SCALE), fill=COLORS["muted"]+(220,), bold=True)
        draw_text(img, clip_text(nearest.get("full_name", "Not available"), 42), (int(W*.08), int(H*.80)), size=int(25*SCALE), fill=COLORS["white"]+(245,), bold=True)
        q = nearest.get("perihelion_au")
        q_text = f"q = {q:.4f} AU" if isinstance(q, (int, float)) and np.isfinite(q) else "q unavailable"
        draw_text(img, q_text, (int(W*.08), int(H*.84)), size=int(20*SCALE), fill=COLORS["PAR"]+(235,), bold=True)

    def draw_stat_strip(self, img: Image.Image, stats: Sequence[Tuple[str, str]]):
        y = int(H*.755)
        x0, x1 = int(W*.06), int(W*.94)
        overlay = Image.new("RGBA", SIZE, (0,0,0,0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0, y, x1, y+int(118*SCALE)), radius=int(22*SCALE), fill=(3,8,18,185), outline=(80,175,210,65), width=1)
        img.alpha_composite(overlay)
        width = (x1-x0)/len(stats)
        for idx, (label, value) in enumerate(stats):
            cx = x0 + width*(idx+.5)
            draw_text(img, value, (int(cx), y+int(29*SCALE)), size=int(29*SCALE), fill=COLORS["white"]+(245,), bold=True, anchor="ma")
            draw_text(img, label, (int(cx), y+int(72*SCALE)), size=int(14*SCALE), fill=COLORS["muted"]+(220,), bold=True, anchor="ma")

    def draw_families(self, img: Image.Image):
        counts = self.summary.get("class_counts", {})
        ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:7]
        total = max(1, sum(counts.values()))
        x0, y0 = int(W*.08), int(H*.19)
        bar_w = int(W*.82)
        row_h = int(H*.075)
        overlay = Image.new("RGBA", SIZE, (0,0,0,0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0-int(18*SCALE), y0-int(25*SCALE), x0+bar_w+int(18*SCALE), y0+row_h*len(ordered)+int(28*SCALE)),
                             radius=int(24*SCALE), fill=(3,8,18,185), outline=(80,175,210,65), width=1)
        img.alpha_composite(overlay)
        d = ImageDraw.Draw(img)
        for idx, (family, count) in enumerate(ordered):
            y = y0 + idx*row_h
            color = COLORS.get(family, COLORS["OTHER"])
            frac = count/total
            d.rounded_rectangle((x0, y+int(28*SCALE), x0+bar_w, y+int(47*SCALE)), radius=int(8*SCALE), fill=(70,92,112,55))
            d.rounded_rectangle((x0, y+int(28*SCALE), x0+max(5, int(bar_w*frac)), y+int(47*SCALE)), radius=int(8*SCALE), fill=color+(220,))
            draw_text(img, CLASS_LABELS.get(family, family), (x0, y), size=int(18*SCALE), fill=color+(240,), bold=True, stroke=1)
            draw_text(img, f"{count:,}", (x0+bar_w, y), size=int(18*SCALE), fill=COLORS["white"]+(235,), bold=True, anchor="ra", stroke=1)
        draw_text(img, "JPL ORBIT-CLASS COUNTS", (x0, int(H*.75)), size=int(20*SCALE), fill=COLORS["ETc"]+(235,), bold=True)
        draw_wrapped_text(img, "Class codes describe orbital regimes, not how bright or active a comet is today.",
                          (x0, int(H*.79)), bar_w, size=int(18*SCALE), fill=COLORS["muted"]+(225,))

    def draw_inclination(self, img: Image.Image):
        box = self.side_box
        img.alpha_composite(self.side_layer)
        d = ImageDraw.Draw(img)
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        d.line((box[0], cy, box[2], cy), fill=(112, 187, 218, 90), width=max(1, int(2*SCALE)))
        self.draw_sun(img, (cx, cy), 11*SCALE)
        d.rounded_rectangle(box, radius=int(24*SCALE), outline=(84,178,213,70), width=max(1,int(2*SCALE)))
        draw_text(img, "SIDE VIEW • ECLIPTIC PLANE", (box[0]+int(18*SCALE), box[1]+int(18*SCALE)), size=int(16*SCALE), fill=(151,208,231,220), bold=True, stroke=1)
        inclinations = finite(c.inclination_deg for c in self.drawable)
        retro = sum(1 for c in self.drawable if np.isfinite(c.inclination_deg) and c.inclination_deg > 90)
        median = float(np.median(inclinations)) if len(inclinations) else 0
        self.draw_stat_strip(img, [
            ("MEDIAN TILT", f"{median:.1f}°"),
            ("RETROGRADE", f"{retro:,}"),
            ("MAXIMUM", f"{max(inclinations) if len(inclinations) else 0:.1f}°"),
        ])

    def draw_highlights(self, img: Image.Image, t: float):
        box = (int(W*.05), int(H*.16), int(W*.95), int(H*.69))
        self.draw_planet_rings(img, box, 35.0, False, False)
        self.draw_sun(img, ((box[0]+box[2])/2, (box[1]+box[3])/2), 13*SCALE)
        d = ImageDraw.Draw(img)
        for idx, comet in enumerate(self.highlights):
            xyz = orbit_xyz(comet, samples=180 if not QUICK_MODE else 90, radius_limit=35.0)
            mapped = []
            for p in xyz:
                mapped.append((np.nan,np.nan) if not np.isfinite(p).all() else map_top_down(p, box, 35.0, False))
            color = COLORS.get(comet.family, COLORS["OTHER"])
            for chunk in split_valid_polyline(np.asarray(mapped)):
                d.line(chunk, fill=color+(205,), width=max(2, int(3*SCALE)))
            pos = approximate_position_xyz(comet, self.render_jd)
            if pos is not None and np.linalg.norm(pos) <= 35:
                x, y = map_top_down(pos, box, 35.0, False)
                r = 5*SCALE
                d.ellipse((x-r,y-r,x+r,y+r), fill=color+(255,), outline=(255,255,255,220), width=1)
        d.rounded_rectangle(box, radius=int(24*SCALE), outline=(84,178,213,70), width=max(1,int(2*SCALE)))
        y0 = int(H*.72)
        for idx, comet in enumerate(self.highlights[:4]):
            x = int(W*(.08 + (idx%2)*.46))
            y = y0 + (idx//2)*int(66*SCALE)
            color = COLORS.get(comet.family, COLORS["OTHER"])
            draw_text(img, "●", (x,y), size=int(18*SCALE), fill=color+(245,), stroke=1)
            draw_text(img, clip_text(comet.full_name, 24), (x+int(26*SCALE),y), size=int(16*SCALE), fill=COLORS["white"]+(235,), bold=True, stroke=1)

    def draw_outro(self, img: Image.Image):
        box = self.deep_box
        img.alpha_composite(self.deep_layer)
        self.draw_planet_rings(img, box, self.deep_radius, True, False)
        self.draw_sun(img, ((box[0]+box[2])/2, (box[1]+box[3])/2), 10*SCALE)
        draw_text(img, "THE CATALOG IS NEVER FINISHED", (W//2, int(H*.755)), size=int(28*SCALE), fill=COLORS["white"]+(245,), bold=True, anchor="ma")
        draw_text(img, "New discoveries and observations reshape this map", (W//2, int(H*.80)), size=int(19*SCALE), fill=COLORS["ETc"]+(235,), bold=True, anchor="ma")
        draw_text(img, "OSCULATING ORBITS • NOT LIVE TAILS OR FULL N-BODY TRAJECTORIES", (W//2, int(H*.85)), size=int(14*SCALE), fill=COLORS["muted"]+(215,), bold=True, anchor="ma")

    def draw_caption(self, img: Image.Image, t: float):
        caption = self.caption_at(t)
        if not caption:
            return
        y0 = H - int(242*SCALE)
        overlay = Image.new("RGBA", SIZE, (0,0,0,0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((int(43*SCALE), y0, W-int(43*SCALE), y0+int(125*SCALE)), radius=int(23*SCALE),
                             fill=(2,6,14,175), outline=(80,185,220,65), width=1)
        img.alpha_composite(overlay)
        draw_wrapped_text(img, caption, (int(67*SCALE), y0+int(27*SCALE)), W-int(134*SCALE),
                          size=int(28*SCALE), fill=COLORS["white"]+(245,))

    def draw_scanlines(self, img: Image.Image, t: float):
        overlay = Image.new("RGBA", SIZE, (0,0,0,0))
        od = ImageDraw.Draw(overlay)
        offset = int((t*35) % max(4,int(7*SCALE)))
        step = max(4,int(7*SCALE))
        for y in range(offset,H,step):
            od.line((0,y,W,y), fill=(100,205,235,9), width=1)
        scan_y = int((t*170)%(H+180))-90
        od.rectangle((0,scan_y,W,scan_y+int(46*SCALE)), fill=(90,210,240,6))
        img.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        img = self.background(t)
        shot = next((s for s in SHOT_PLAN if s["start"] <= t < s["end"]), SHOT_PLAN[-1])["name"]
        titles = {
            "intro": "ONE DATABASE • THOUSANDS OF COMET SOLUTIONS",
            "deep": "THE DEEP COMET CLOUD • LOGARITHMIC SCALE",
            "inner": "COMET ORBITS THROUGH THE PLANETARY SYSTEM",
            "families": "ORBITAL FAMILIES",
            "inclination": "THE SOLAR SYSTEM IS NOT FLAT FOR COMETS",
            "highlights": "FAMILIAR NAMES INSIDE THE SWARM",
            "outro": "A LIVING CATALOG OF ORBITS",
        }
        self.draw_header(img,t,titles[shot])
        if shot == "intro":
            self.draw_intro(img,t)
        elif shot == "deep":
            self.draw_deep(img)
        elif shot == "inner":
            self.draw_inner(img)
        elif shot == "families":
            self.draw_families(img)
        elif shot == "inclination":
            self.draw_inclination(img)
        elif shot == "highlights":
            self.draw_highlights(img,t)
        else:
            self.draw_outro(img)
        self.draw_caption(img,t)
        self.draw_scanlines(img,t)

        graded = ImageEnhance.Contrast(img.convert("RGB")).enhance(1.08)
        graded = ImageEnhance.Color(graded).enhance(1.05)
        arr = np.asarray(graded).astype(np.float32)
        arr *= VIGNETTE[...,None]
        fade_in = smoothstep(t/.8)
        fade_out = 1-smoothstep((t-(CONFIG["duration_s"]-1.0))/.9)
        return np.clip(arr*fade_in*fade_out,0,255).astype(np.uint8)


# =============================================================================
# Output helpers
# =============================================================================

def write_srt(path: Path, captions: Sequence[Tuple[float,float,str]]):
    lines=[]
    for idx,(start,end,text) in enumerate(captions,start=1):
        lines.extend([str(idx),f"{format_srt_time(start)} --> {format_srt_time(end)}",text,""])
    path.write_text("\n".join(lines),encoding="utf-8")


def render_video(scene: CometScene) -> Path:
    raw_path = OUTPUT_ROOT / f"{CONFIG['basename']}_raw.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['basename']}_final.mp4"
    write_srt(OUTPUT_ROOT / f"{CONFIG['basename']}.srt", scene.captions)
    frame_count = int(round(CONFIG["duration_s"]*CONFIG["fps"]))
    with iio.get_writer(raw_path,fps=CONFIG["fps"],codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=None) as writer:
        for frame_index in tqdm(range(frame_count),desc="Rendering comet-orbit short"):
            writer.append_data(scene.render_frame(frame_index/CONFIG["fps"]))
    shutil.copyfile(raw_path,final_path)
    return final_path


def make_contact_sheet(paths: Sequence[Path], out_path: Path):
    thumbs=[]
    for path in paths[:6]:
        image=Image.open(path).convert("RGB").resize((270,480))
        draw=ImageDraw.Draw(image)
        draw.rectangle((8,8,122,38),fill=(0,0,0))
        draw.text((17,13),path.stem.replace("preview_",""),fill=(255,255,255))
        thumbs.append(image)
    sheet=Image.new("RGB",(600,1520),(8,11,18))
    for index,thumb in enumerate(thumbs):
        row,col=divmod(index,2)
        sheet.paste(thumb,(20+col*290,20+row*500))
    sheet.save(out_path,quality=92)


def main():
    print("Collecting JPL comet orbit catalog ...")
    comets, summary = collect_catalog()
    csv_path,json_path=save_data(comets,summary)
    print("Data:",csv_path.resolve())
    print("Summary:",json_path.resolve())
    print("Catalog status:",summary["data_status"])
    print("Catalog records:",summary["catalog_count"],"Drawable:",summary["drawable_orbits"])

    scene=CometScene(comets,summary)
    preview_times=[1.0,min(10.0,CONFIG["duration_s"]*.20),min(22.0,CONFIG["duration_s"]*.40),min(35.0,CONFIG["duration_s"]*.62),min(45.0,CONFIG["duration_s"]*.80),CONFIG["duration_s"]-1]
    preview_paths=[]
    for t in tqdm(preview_times,desc="Preview frames"):
        path=PREVIEW_ROOT/f"preview_{int(t):02d}s.png"
        Image.fromarray(scene.render_frame(float(t))).save(path)
        preview_paths.append(path)
    make_contact_sheet(preview_paths,PREVIEW_ROOT/"every_known_comet_orbiting_the_sun_contact_sheet.jpg")
    video_path=render_video(scene)
    print("Video:",video_path.resolve())


if __name__ == "__main__":
    main()
