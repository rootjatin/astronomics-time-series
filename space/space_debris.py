from __future__ import annotations

"""
output : https://youtube.com/shorts/ZnfC6CTckAk?feature=share
SPACE DEBRIS — cinematic YouTube Shorts renderer

Creates a vertical 1080x1920 data-driven short about catalogued orbital debris.
The visual design deliberately follows the same overall production pattern as the
provided tropical-cyclone renderer: live data first, cached data second, a clear
offline fixture third, plus previews, captions, CSV/JSON exports and an MP4.


Reference context shown in the video comes from ESA Space Debris Environment
Statistics, last updated 31 July 2026:
- ~46,590 regularly tracked space objects
- 54,000 modeled objects >10 cm
- 1.2 million debris objects 1–10 cm
- 140 million debris objects 1 mm–1 cm
Those modeled totals include objects too small to appear in SATCAT.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    SPACE_DEBRIS_QUICK=1 python space_debris_youtube_short.py

Force offline fixture
---------------------
    SPACE_DEBRIS_OFFLINE=1 python space_debris_youtube_short.py

REAL DATA
---------
Primary live source: CelesTrak SATCAT CSV.
The script filters current on-orbit records whose OBJECT_TYPE is DEBRIS, then uses
real catalog metadata such as NORAD catalog number, launch date, inclination,
perigee and apogee. The particles shown are a cinematic geocentric projection of
those real orbital parameters; they are NOT precise propagated sky positions.

Reference context shown in the video comes from ESA Space Debris Environment
Statistics, last updated 31 July 2026:
- ~46,590 regularly tracked space objects
- 54,000 modeled objects >10 cm
- 1.2 million debris objects 1–10 cm
- 140 million debris objects 1 mm–1 cm
Those modeled totals include objects too small to appear in SATCAT.

Outputs
-------
- final vertical MP4 (with generated ambient audio when ffmpeg is available)
- silent MP4 fallback
- SRT subtitle sidecar
- preview PNG frames
- CSV of the debris records used
- JSON summary/source notes
- cached CelesTrak SATCAT CSV

Sources
-------
- CelesTrak SATCAT: https://celestrak.org/satcat/
- Raw SATCAT CSV: https://celestrak.org/pub/satcat.csv
- SATCAT format: https://celestrak.org/satcat/satcat-format.php
- ESA statistics: https://sdup.esoc.esa.int/discosweb/statistics/
"""

import io
import json
import math
import os
import re
import shutil
import subprocess
import urllib.request
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("SPACE_DEBRIS_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("SPACE_DEBRIS_OFFLINE", "0") == "1"
NOW_UTC = datetime.now(timezone.utc)
CUTOFF_YEAR = int(os.environ.get("SPACE_DEBRIS_CUTOFF_YEAR", str(NOW_UTC.year)))

OUTPUT_ROOT = Path("space_debris_youtube_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
CACHE_ROOT = DATA_ROOT / "cache"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for d in (OUTPUT_ROOT, DATA_ROOT, CACHE_ROOT, PREVIEW_DIR):
    d.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12.0 if QUICK_MODE else 58.0,
    "title": "SPACE DEBRIS",
    "subtitle": "CATALOGUED OBJECTS // EARTH ORBIT",
    "output_basename": "space_debris",
    "max_particles": 2600 if QUICK_MODE else 10500,
    "star_count": 130 if QUICK_MODE else 420,
    "dust_count": 45 if QUICK_MODE else 140,
    "grain_strength": 3.2,
    "contrast": 1.10,
    "saturation": 1.05,
    "vignette": 0.35,
    "sample_rate": 22050 if QUICK_MODE else 44100,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0

COLORS = {
    "space": (1, 4, 12),
    "space2": (3, 10, 22),
    "earth_dark": (5, 22, 37),
    "earth": (21, 82, 112),
    "earth_glow": (64, 192, 238),
    "white": (246, 250, 252),
    "muted": (151, 185, 199),
    "grid": (67, 111, 129),
    "leo": (58, 224, 255),
    "meo": (255, 209, 83),
    "geo": (255, 114, 79),
    "high": (227, 92, 255),
    "danger": (255, 71, 91),
}

FULL_SHOT_PLAN = [
    {"name": "opening", "start": 0.0, "end": 6.0},
    {"name": "catalog_cloud", "start": 6.0, "end": 24.0},
    {"name": "altitude_bands", "start": 24.0, "end": 34.0},
    {"name": "families", "start": 34.0, "end": 43.0},
    {"name": "scale_gap", "start": 43.0, "end": 51.0},
    {"name": "all_debris", "start": 51.0, "end": 56.0},
    {"name": "finale", "start": 56.0, "end": 58.0},
]

FULL_CAPTIONS = [
    (0.4, 5.7, "Space around Earth is filled with human-made debris."),
    (6.2, 23.7, "These points come from catalogued debris records, using real orbital altitude and inclination data."),
    (24.2, 33.7, "Most tracked debris is concentrated in low Earth orbit, but fragments also occupy higher orbital bands."),
    (34.2, 42.7, "Some breakup and collision events created large families of fragments that remain in orbit for years."),
    (43.2, 50.7, "The tracked catalogue is only the visible part of the problem. Smaller debris is far more numerous."),
    (51.2, 55.7, "Every dot here represents a catalogued debris object loaded by this render."),
    (56.1, 57.8, "Space debris. Thousands tracked. Millions more estimated."),
]

if QUICK_MODE:
    k = float(CONFIG["duration_s"]) / 58.0
    SHOT_PLAN = [{**s, "start": s["start"] * k, "end": s["end"] * k} for s in FULL_SHOT_PLAN]
    CAPTIONS = [(a * k, b * k, text) for a, b, text in FULL_CAPTIONS]
else:
    SHOT_PLAN = FULL_SHOT_PLAN
    CAPTIONS = FULL_CAPTIONS

SATCAT_URL = "https://celestrak.org/pub/satcat.csv"
SATCAT_QUERY_FALLBACKS = [
    "https://celestrak.org/satcat/records.php?NAME=DEB&FORMAT=CSV&ONORBIT=1",
    "https://www.celestrak.org/pub/satcat.csv",
]
ESA_STATS = {
    "as_of": "2026-07-31",
    "tracked_space_objects_about": 46590,
    "modeled_gt_10cm": 54000,
    "modeled_1cm_to_10cm": 1_200_000,
    "modeled_1mm_to_1cm": 140_000_000,
    "fragmentation_events_more_than": 660,
    "mass_tonnes_more_than": 17000,
}


# -----------------------------------------------------------------------------
# Models and helpers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Debris:
    norad_id: int
    name: str
    object_id: str
    owner: str
    launch_date: Optional[datetime]
    period_min: float
    inclination_deg: float
    apogee_km: float
    perigee_km: float
    rcs_m2: float

    @property
    def mean_altitude_km(self) -> float:
        vals = [v for v in (self.apogee_km, self.perigee_km) if np.isfinite(v) and v >= 0]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def family(self) -> str:
        return debris_family(self.name)

    @property
    def regime(self) -> str:
        p = self.perigee_km
        a = self.apogee_km
        m = self.mean_altitude_km
        if np.isfinite(a) and a < 2000:
            return "LEO"
        if np.isfinite(p) and np.isfinite(a) and p > 31500 and a < 40500:
            return "GEO"
        if np.isfinite(m) and m < 31570:
            return "MEO"
        return "HIGH / CROSSING"


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(x: float) -> float:
    x = clamp(x)
    return x * x * (3.0 - 2.0 * x)


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def shot_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))


def parse_date(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text or text.upper() in {"NAN", "NONE"}:
        return None
    ts = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def num(row: pd.Series, key: str) -> float:
    try:
        value = float(row.get(key, np.nan))
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def debris_family(name: str) -> str:
    text = re.sub(r"\s+", " ", (name or "UNKNOWN").upper()).strip()
    text = re.sub(r"\s+DEB.*$", "", text)
    text = re.sub(r"\s+R/B.*$", "", text)
    text = re.sub(r"\s+FRAGMENT.*$", "", text)
    # Preserve famous debris-family names while avoiding thousands of one-offs.
    for known in [
        "FENGYUN 1C", "COSMOS 2251", "IRIDIUM 33", "COSMOS 1408",
        "SL-16", "SL-14", "SL-8", "DELTA 1", "BREEZE-M",
    ]:
        if text.startswith(known):
            return known
    # Drop trailing object suffixes/numbers that are often fragment identifiers.
    text = re.sub(r"\s+[A-Z]{1,3}\d{0,4}$", "", text)
    text = re.sub(r"\s+\d{2,6}$", "", text)
    return text[:28] or "UNKNOWN"


def regime_color(regime: str, alpha: int = 240) -> Tuple[int, int, int, int]:
    if regime == "LEO":
        return COLORS["leo"] + (alpha,)
    if regime == "MEO":
        return COLORS["meo"] + (alpha,)
    if regime == "GEO":
        return COLORS["geo"] + (alpha,)
    return COLORS["high"] + (alpha,)


def get_font(size: int, bold: bool = False, condensed: bool = False):
    candidates: List[str] = []
    if condensed and bold:
        candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf", "DejaVuSansCondensed-Bold.ttf"]
    if condensed:
        candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf", "DejaVuSansCondensed.ttf"]
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(image: Image.Image, text: str, xy: Tuple[int, int], size: int,
              fill: Tuple[int, int, int, int], bold: bool = False,
              condensed: bool = False, anchor: str = "la", stroke: int = 2):
    ImageDraw.Draw(image).text(
        xy, text, font=get_font(size, bold, condensed), fill=fill, anchor=anchor,
        stroke_width=stroke, stroke_fill=(0, 0, 0, min(235, fill[3]))
    )


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for i, (a, b, text) in enumerate(captions, 1):
        lines.extend([str(i), f"{format_srt_time(a)} --> {format_srt_time(b)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def request_bytes(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; SpaceDebrisShort/1.0; educational visualization)",
            "Accept": "text/csv,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    r = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * r**1.75, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def parse_satcat_csv(payload: bytes) -> List[Debris]:
    df = pd.read_csv(io.BytesIO(payload), low_memory=False)
    required = {"NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_TYPE"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"SATCAT CSV missing expected columns: {sorted(required - set(df.columns))}")

    obj_type = df["OBJECT_TYPE"].astype(str).str.upper().str.strip()
    debris = df[obj_type.eq("DEBRIS")].copy()
    if "DECAY_DATE" in debris.columns:
        decay = debris["DECAY_DATE"].astype(str).str.strip()
        debris = debris[decay.isin(["", "nan", "NaN", "NAN", "None"]) | decay.str.startswith("0000")]

    records: List[Debris] = []
    for _, row in debris.iterrows():
        try:
            cat = int(float(row["NORAD_CAT_ID"]))
        except Exception:
            continue
        launch = parse_date(row.get("LAUNCH_DATE"))
        if launch and launch.year > CUTOFF_YEAR:
            continue
        records.append(
            Debris(
                norad_id=cat,
                name=str(row.get("OBJECT_NAME", "UNKNOWN")).strip() or "UNKNOWN",
                object_id=str(row.get("OBJECT_ID", "")).strip(),
                owner=str(row.get("OWNER", "")).strip(),
                launch_date=launch,
                period_min=num(row, "PERIOD"),
                inclination_deg=num(row, "INCLINATION"),
                apogee_km=num(row, "APOGEE"),
                perigee_km=num(row, "PERIGEE"),
                rcs_m2=num(row, "RCS"),
            )
        )
    if not records:
        raise RuntimeError("No current debris records were parsed from SATCAT")
    return records


def make_synthetic_debris(count: int = 3200) -> List[Debris]:
    rng = np.random.default_rng(20260902)
    families = [
        ("FENGYUN 1C DEB", 0.23, 850, 98),
        ("COSMOS 2251 DEB", 0.13, 780, 74),
        ("IRIDIUM 33 DEB", 0.08, 770, 86),
        ("COSMOS 1408 DEB", 0.10, 490, 83),
        ("SL-16 DEB", 0.12, 900, 71),
        ("MISC DEBRIS", 0.34, 1100, 58),
    ]
    probs = np.array([f[1] for f in families], dtype=float)
    probs /= probs.sum()
    records: List[Debris] = []
    for i in range(count):
        idx = int(rng.choice(len(families), p=probs))
        base, _, alt0, inc0 = families[idx]
        if rng.random() < 0.86:
            mean_alt = float(np.clip(rng.normal(alt0, 260), 180, 1900))
            ecc = abs(float(rng.normal(0.0, 0.12)))
        else:
            mean_alt = float(rng.choice([rng.uniform(3000, 18000), rng.normal(35786, 900)]))
            ecc = abs(float(rng.normal(0.12, 0.16)))
        spread = mean_alt * min(ecc, 0.7)
        peri = max(120.0, mean_alt - spread)
        apo = mean_alt + spread
        launch_year = int(rng.integers(1961, min(CUTOFF_YEAR, NOW_UTC.year) + 1))
        records.append(Debris(
            norad_id=800000 + i,
            name=f"{base} {i:04d}",
            object_id=f"{launch_year}-XXX{i%999:03d}",
            owner="SYN",
            launch_date=datetime(launch_year, 1, 1, tzinfo=timezone.utc),
            period_min=float(90 + mean_alt / 85),
            inclination_deg=float(np.clip(rng.normal(inc0, 18), 0, 179)),
            apogee_km=float(apo), perigee_km=float(peri), rcs_m2=float(10 ** rng.uniform(-3.5, 0.8))
        ))
    return records


def load_debris() -> Tuple[List[Debris], str, List[str], Optional[Path]]:
    notes: List[str] = []
    cache = CACHE_ROOT / "satcat.csv"
    if OFFLINE_MODE:
        notes.append("Offline mode requested; deterministic synthetic fixture used")
        return make_synthetic_debris(1600 if QUICK_MODE else 4200), "synthetic_procedural_fixture", notes, None

    urls = [SATCAT_URL] + SATCAT_QUERY_FALLBACKS
    for url in urls:
        try:
            payload = request_bytes(url)
            if len(payload) < 20_000:
                raise RuntimeError("download was unexpectedly small")
            records = parse_satcat_csv(payload)
            cache.write_bytes(payload)
            notes.append(f"Downloaded current CelesTrak SATCAT CSV from {url}")
            notes.append("Particle positions are stylized from real altitude/inclination; they are not propagated current coordinates")
            return records, "celestrak_satcat_live", notes, cache
        except Exception as exc:
            notes.append(f"Live SATCAT attempt failed ({url}): {exc}")

    if cache.exists() and cache.stat().st_size > 20_000:
        try:
            records = parse_satcat_csv(cache.read_bytes())
            notes.append("Loaded cached CelesTrak SATCAT CSV after live-download failure")
            return records, "celestrak_satcat_cached", notes, cache
        except Exception as exc:
            notes.append(f"Cached SATCAT parse failed: {exc}")

    notes.append("Using deterministic synthetic debris fixture for preview/timing only")
    return make_synthetic_debris(1600 if QUICK_MODE else 4200), "synthetic_procedural_fixture", notes, None


# -----------------------------------------------------------------------------
# Summaries / exports
# -----------------------------------------------------------------------------

def summarize(records: Sequence[Debris], source: str) -> Dict[str, Any]:
    regime_counts: Dict[str, int] = {}
    family_counts: Dict[str, int] = {}
    launch_years: List[int] = []
    alts: List[float] = []
    for d in records:
        regime_counts[d.regime] = regime_counts.get(d.regime, 0) + 1
        family_counts[d.family] = family_counts.get(d.family, 0) + 1
        if d.launch_date:
            launch_years.append(d.launch_date.year)
        if np.isfinite(d.mean_altitude_km):
            alts.append(d.mean_altitude_km)
    top_families = sorted(family_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "title": CONFIG["title"],
        "generated_utc": NOW_UTC.isoformat(),
        "cutoff_year": CUTOFF_YEAR,
        "data_source": source,
        "catalogued_debris_count_loaded": len(records),
        "regime_counts": regime_counts,
        "top_families": top_families,
        "median_mean_altitude_km": float(np.median(alts)) if alts else None,
        "earliest_launch_year": min(launch_years) if launch_years else None,
        "esa_reference_statistics": ESA_STATS,
        "visualization_caveat": (
            "The catalog records are real when CelesTrak loads successfully. The screen positions are a deterministic "
            "cinematic projection derived from catalog altitude and inclination, not SGP4-propagated positions."
        ),
    }


def save_data(records: Sequence[Debris], summary: Dict[str, Any], notes: Sequence[str]) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "space_debris_catalog_used.csv"
    json_path = DATA_ROOT / "space_debris_summary.json"
    rows = []
    for d in records:
        rows.append({
            "norad_id": d.norad_id,
            "name": d.name,
            "object_id": d.object_id,
            "owner": d.owner,
            "launch_date": d.launch_date.isoformat() if d.launch_date else None,
            "period_min": d.period_min,
            "inclination_deg": d.inclination_deg,
            "apogee_km": d.apogee_km,
            "perigee_km": d.perigee_km,
            "mean_altitude_km": d.mean_altitude_km,
            "rcs_m2": d.rcs_m2,
            "regime": d.regime,
            "family": d.family,
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "summary": summary,
        "notes": list(notes),
        "source_urls": {
            "celestrak_satcat": "https://celestrak.org/satcat/",
            "celestrak_satcat_csv": SATCAT_URL,
            "esa_statistics": "https://sdup.esoc.esa.int/discosweb/statistics/",
        },
        "fallback_warning": "Synthetic fixture is not observational data."
    }, indent=2, default=str), encoding="utf-8")
    return csv_path, json_path


# -----------------------------------------------------------------------------
# Visual scene
# -----------------------------------------------------------------------------

class DebrisScene:
    EARTH_RADIUS_PX = 226

    def __init__(self, records: List[Debris], summary: Dict[str, Any]):
        self.records = records
        self.summary = summary
        self.rng = np.random.default_rng(4127)
        self.stars = self._make_stars(int(CONFIG["star_count"]))
        self.dust = self._make_stars(int(CONFIG["dust_count"]))
        self.particles = self._build_particle_table(records)
        self.family_order = [x[0] for x in summary.get("top_families", []) if x[1] >= 2][:6]
        self.regime_order = [r for r in ["LEO", "MEO", "GEO", "HIGH / CROSSING"] if summary.get("regime_counts", {}).get(r, 0)]

    def _make_stars(self, count: int) -> List[Tuple[float, float, float, float]]:
        return [(float(self.rng.uniform(0, OUT_W)), float(self.rng.uniform(0, OUT_H)),
                 float(self.rng.uniform(0.4, 2.0 if QUICK_MODE else 2.8)), float(self.rng.uniform(12, 70)))
                for _ in range(count)]

    def _build_particle_table(self, records: Sequence[Debris]) -> List[Dict[str, Any]]:
        if not records:
            return []
        max_particles = int(CONFIG["max_particles"])
        # Stable stratified sample if the live catalogue is larger than render budget.
        ordered = sorted(records, key=lambda d: d.norad_id)
        if len(ordered) > max_particles:
            indices = np.linspace(0, len(ordered) - 1, max_particles).astype(int)
            selected = [ordered[i] for i in indices]
        else:
            selected = ordered
        particles = []
        for d in selected:
            seed = (d.norad_id * 2654435761) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            alt = d.mean_altitude_km
            if not np.isfinite(alt):
                alt = float(rng.uniform(350, 1600))
            inc = d.inclination_deg if np.isfinite(d.inclination_deg) else float(rng.uniform(0, 110))
            # Compress altitude logarithmically so LEO remains visually separable while GEO still fits.
            radius = self.EARTH_RADIUS_PX * SCALE + (72 + 162 * math.log10(1 + max(0.0, alt) / 300.0)) * SCALE
            radius = float(np.clip(radius, 285 * SCALE, 474 * SCALE))
            particles.append({
                "d": d,
                "phase": float(rng.uniform(0, math.tau)),
                "node": float(rng.uniform(0, math.tau)),
                "radius": radius,
                "inc": math.radians(float(np.clip(inc, 0, 180))),
                "speed": float(rng.uniform(0.45, 1.35)),
                "size": float(rng.uniform(0.75, 1.9 if not QUICK_MODE else 1.35)),
                "alpha": int(rng.integers(95, 240)),
            })
        return particles

    @property
    def earth_center(self) -> Tuple[float, float]:
        return OUT_W / 2.0, OUT_H * 0.47

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
        top = np.array(COLORS["space2"], dtype=float)
        bottom = np.array(COLORS["space"], dtype=float)
        for y in range(OUT_H):
            u = y / max(OUT_H - 1, 1)
            rgb = (top * (1-u) + bottom * u).astype(np.uint8)
            arr[y, :, :3] = rgb
            arr[y, :, 3] = 255
        im = Image.fromarray(arr, "RGBA")
        dr = ImageDraw.Draw(im)
        for x, y, r, a in self.stars:
            pulse = 0.55 + 0.45 * math.sin(t * 0.55 + x * 0.011 + y * 0.007) ** 2
            rr = r * SCALE
            dr.ellipse((x-rr, y-rr, x+rr, y+rr), fill=(204, 231, 244, int(a * pulse)))
        return im

    def draw_earth(self, image: Image.Image, alpha: int = 255):
        cx, cy = self.earth_center
        r = self.EARTH_RADIUS_PX * SCALE
        glow = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        for k in range(6, 0, -1):
            rr = r + k * 14 * SCALE
            gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), outline=COLORS["earth_glow"] + (max(5, int(alpha * 0.035 * k)),), width=max(1, int(9*SCALE)))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(2, int(16*SCALE)))))
        dr = ImageDraw.Draw(image)
        dr.ellipse((cx-r, cy-r, cx+r, cy+r), fill=COLORS["earth_dark"] + (alpha,), outline=COLORS["earth_glow"] + (int(alpha*0.7),), width=max(1,int(3*SCALE)))
        # Abstract night-side continents, intentionally not cartographic.
        land = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        ld = ImageDraw.Draw(land)
        for pts in [
            [(-0.62,-0.45),(-0.28,-0.64),(0.03,-0.44),(-0.08,-0.12),(-0.34,0.02),(-0.56,-0.16)],
            [(0.10,-0.55),(0.47,-0.49),(0.66,-0.18),(0.41,0.01),(0.13,-0.12)],
            [(-0.25,0.12),(-0.02,0.04),(0.17,0.34),(0.02,0.70),(-0.22,0.56),(-0.40,0.28)],
        ]:
            poly = [(cx + px*r, cy + py*r) for px,py in pts]
            ld.polygon(poly, fill=COLORS["earth"] + (int(alpha*0.75),))
        land = land.filter(ImageFilter.GaussianBlur(max(1,int(1.4*SCALE))))
        image.alpha_composite(land)
        # Horizon highlight.
        arc = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        ad = ImageDraw.Draw(arc)
        ad.arc((cx-r,cy-r,cx+r,cy+r), 195, 342, fill=COLORS["earth_glow"]+(int(alpha*0.9),), width=max(1,int(5*SCALE)))
        image.alpha_composite(arc.filter(ImageFilter.GaussianBlur(max(1,int(2*SCALE)))))

    def particle_xy(self, p: Dict[str, Any], t: float) -> Tuple[float, float, float]:
        cx, cy = self.earth_center
        theta = p["phase"] + t * 0.11 * p["speed"]
        # Tilted projected orbit. This uses real inclination but a deterministic phase/node.
        x0 = math.cos(theta) * p["radius"]
        y0 = math.sin(theta) * p["radius"] * (0.28 + 0.64 * abs(math.cos(p["inc"])))
        ca, sa = math.cos(p["node"]), math.sin(p["node"])
        x = x0 * ca - y0 * sa
        y = x0 * sa + y0 * ca
        depth = math.sin(theta)
        return cx + x, cy + y, depth

    def draw_particles(self, image: Image.Image, t: float, fraction: float = 1.0,
                       regime: Optional[str] = None, family: Optional[str] = None,
                       dim_others: bool = False):
        core = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        glow = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        cd, gd = ImageDraw.Draw(core), ImageDraw.Draw(glow)
        limit = int(len(self.particles) * clamp(fraction))
        for i, p in enumerate(self.particles):
            if i >= limit:
                break
            d: Debris = p["d"]
            match = (regime is None or d.regime == regime) and (family is None or d.family == family)
            if not match and not dim_others:
                continue
            x, y, depth = self.particle_xy(p, t)
            if x < -10 or x > OUT_W+10 or y < -10 or y > OUT_H+10:
                continue
            alpha = int(p["alpha"] * (1.0 if match else 0.13))
            color = regime_color(d.regime, alpha)
            r = max(0.8, p["size"] * SCALE * (0.85 + 0.2 * depth))
            if match:
                gd.ellipse((x-r*2.5,y-r*2.5,x+r*2.5,y+r*2.5), fill=color[:3]+(min(90,alpha//2),))
            cd.ellipse((x-r,y-r,x+r,y+r), fill=color)
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(1,int(3*SCALE)))))
        image.alpha_composite(core)

    def draw_opening(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        self.draw_earth(image, int(120 + 135*p))
        self.draw_particles(image, t, fraction=0.03 + 0.18*p)
        draw_text(image, "SPACE", (OUT_W//2, int(OUT_H*0.72)), 50 if QUICK_MODE else 100, COLORS["muted"]+(245,), True, True, "ma")
        draw_text(image, "DEBRIS", (OUT_W//2, int(OUT_H*0.81)), 68 if QUICK_MODE else 136, COLORS["white"]+(255,), True, True, "ma")
        draw_text(image, "THE HUMAN-MADE CLOUD AROUND EARTH", (OUT_W//2, int(OUT_H*0.89)), 11 if QUICK_MODE else 22, COLORS["earth_glow"]+(240,), True, True, "ma", 1)

    def draw_catalog_cloud(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = shot_progress(t, shot)
        self.draw_earth(image)
        self.draw_particles(image, t, fraction=smoothstep(p))
        count = int(round(len(self.records) * smoothstep(p)))
        top = int(1450*SCALE)
        draw_text(image, f"{count:,}", (OUT_W//2, top), 69 if QUICK_MODE else 138, COLORS["white"]+(255,), True, True, "ma")
        draw_text(image, "CATALOGUED DEBRIS OBJECTS LOADED", (OUT_W//2, top+int(104*SCALE)), 13 if QUICK_MODE else 26, COLORS["muted"]+(235,), True, True, "ma", 1)
        draw_text(image, "REAL SATCAT ALTITUDE + INCLINATION", (OUT_W//2, top+int(154*SCALE)), 9 if QUICK_MODE else 18, COLORS["earth_glow"]+(225,), True, True, "ma", 1)

    def draw_altitude_bands(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = shot_progress(t, shot)
        self.draw_earth(image)
        if not self.regime_order:
            self.draw_particles(image,t)
            return
        idx = min(int(p * len(self.regime_order)), len(self.regime_order)-1)
        regime = self.regime_order[idx]
        self.draw_particles(image, t, regime=regime, dim_others=True)
        count = self.summary.get("regime_counts", {}).get(regime, 0)
        color = regime_color(regime,255)
        draw_text(image, regime, (OUT_W//2, int(OUT_H*0.76)), 42 if QUICK_MODE else 84, COLORS["white"]+(255,), True, True, "ma")
        draw_text(image, f"{count:,} LOADED OBJECTS", (OUT_W//2, int(OUT_H*0.83)), 22 if QUICK_MODE else 44, color, True, True, "ma")
        hint = {"LEO":"BELOW 2,000 KM", "MEO":"MEDIUM-EARTH ALTITUDES", "GEO":"~35,786 KM BAND", "HIGH / CROSSING":"HIGH OR CROSSING ORBITS"}.get(regime,"")
        draw_text(image, hint, (OUT_W//2, int(OUT_H*0.88)), 10 if QUICK_MODE else 20, COLORS["muted"]+(225,), True, True, "ma", 1)

    def draw_families(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = shot_progress(t, shot)
        self.draw_earth(image)
        if not self.family_order:
            self.draw_particles(image,t)
            return
        idx = min(int(p * len(self.family_order)), len(self.family_order)-1)
        family = self.family_order[idx]
        self.draw_particles(image, t, family=family, dim_others=True)
        count = dict(self.summary.get("top_families", [])).get(family, 0)
        draw_text(image, "DEBRIS FAMILY", (int(54*SCALE), int(1450*SCALE)), 13 if QUICK_MODE else 26, COLORS["muted"]+(230,), True, True, stroke=1)
        draw_text(image, family, (int(54*SCALE), int(1510*SCALE)), 34 if QUICK_MODE else 68, COLORS["white"]+(255,), True, True)
        draw_text(image, f"{count:,} CATALOG RECORDS IN THIS LOAD", (int(54*SCALE), int(1600*SCALE)), 13 if QUICK_MODE else 26, COLORS["danger"]+(245,), True, True, stroke=1)
        draw_text(image, "BREAKUPS + COLLISIONS CAN CREATE LONG-LIVED FRAGMENT CLOUDS", (int(54*SCALE), int(1660*SCALE)), 9 if QUICK_MODE else 18, COLORS["muted"]+(215,), True, True, stroke=1)

    def draw_scale_gap(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        self.draw_earth(image, 200)
        self.draw_particles(image,t, fraction=1.0)
        shade = Image.new("RGBA", OUT_SIZE, (0,0,0,int(95+45*p)))
        image.alpha_composite(shade)
        rows = [
            ("TRACKED OBJECTS", f"~{ESA_STATS['tracked_space_objects_about']:,}", "NETWORK CATALOGUES"),
            ("> 10 CM", f"~{ESA_STATS['modeled_gt_10cm']:,}", "MODELED POPULATION"),
            ("1–10 CM", f"~{ESA_STATS['modeled_1cm_to_10cm']/1_000_000:.1f} MILLION", "MODELED DEBRIS"),
            ("1 MM–1 CM", f"~{ESA_STATS['modeled_1mm_to_1cm']/1_000_000:.0f} MILLION", "MODELED DEBRIS"),
        ]
        y0 = int(460*SCALE)
        for i,(label,value,note) in enumerate(rows):
            y = y0 + int(i*245*SCALE)
            draw_text(image,label,(int(72*SCALE),y),13 if QUICK_MODE else 26,COLORS["muted"]+(235,),True,True,stroke=1)
            draw_text(image,value,(int(72*SCALE),y+int(70*SCALE)),32 if QUICK_MODE else 64,COLORS["white"]+(255,),True,True)
            draw_text(image,note,(int(72*SCALE),y+int(135*SCALE)),9 if QUICK_MODE else 18,COLORS["earth_glow"]+(225,),True,True,stroke=1)
        draw_text(image,"ESA MODEL / STATISTICS — 31 JUL 2026",(OUT_W//2,int(OUT_H*0.91)),9 if QUICK_MODE else 18,COLORS["muted"]+(200,),True,True,"ma",1)

    def draw_all_debris(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        self.draw_earth(image)
        self.draw_particles(image,t)
        panel = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        pd = ImageDraw.Draw(panel)
        top = int(1450*SCALE)
        pd.rounded_rectangle((int(45*SCALE),top,OUT_W-int(45*SCALE),top+int(295*SCALE)),radius=max(12,int(28*SCALE)),fill=(2,7,15,int(185+25*p)),outline=COLORS["grid"]+(78,),width=1)
        image.alpha_composite(panel)
        draw_text(image,f"{len(self.records):,}",(OUT_W//2,top+int(100*SCALE)),78 if QUICK_MODE else 156,COLORS["white"]+(255,),True,True,"ma")
        draw_text(image,"CATALOGUED DEBRIS OBJECTS",(OUT_W//2,top+int(190*SCALE)),20 if QUICK_MODE else 40,COLORS["white"]+(245,),True,True,"ma",1)
        draw_text(image,"THIS RENDER // CATALOGUE ≠ COMPLETE DEBRIS POPULATION",(OUT_W//2,top+int(245*SCALE)),9 if QUICK_MODE else 18,COLORS["muted"]+(220,),True,True,"ma",1)

    def draw_finale(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        self.draw_earth(image)
        self.draw_particles(image,t)
        image.alpha_composite(Image.new("RGBA", OUT_SIZE, (0,0,0,int(50+70*p))))
        draw_text(image,"SPACE DEBRIS",(OUT_W//2,int(OUT_H*0.74)),43 if QUICK_MODE else 86,COLORS["white"]+(255,),True,True,"ma")
        draw_text(image,"THOUSANDS TRACKED",(OUT_W//2,int(OUT_H*0.82)),24 if QUICK_MODE else 48,COLORS["earth_glow"]+(245,),True,True,"ma")
        draw_text(image,"MILLIONS MORE ESTIMATED",(OUT_W//2,int(OUT_H*0.87)),19 if QUICK_MODE else 38,COLORS["danger"]+(245,),True,True,"ma")

    def draw_header(self, image: Image.Image, name: str):
        if name == "opening":
            return
        draw_text(image,"SPACE DEBRIS",(int(48*SCALE),int(72*SCALE)),37 if not QUICK_MODE else 19,COLORS["white"]+(250,),True,True)
        draw_text(image,CONFIG["subtitle"],(int(50*SCALE),int(126*SCALE)),13 if not QUICK_MODE else 7,COLORS["muted"]+(220,),True,True,stroke=1)

    def draw_source(self, image: Image.Image):
        source = str(self.summary.get("data_source", ""))
        if source.startswith("synthetic"):
            text = "SYNTHETIC PREVIEW // NOT OBSERVATIONAL DATA"
        else:
            text = "CELESTRAK SATCAT // ORBIT PARAMETERS, NOT LIVE SKY POSITIONS"
        draw_text(image,text,(int(48*SCALE),OUT_H-int(44*SCALE)),10 if not QUICK_MODE else 5,COLORS["muted"]+(180,),True,True,stroke=1)

    def draw_film_texture(self, image: Image.Image, t: float):
        ov = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        dr = ImageDraw.Draw(ov)
        for x,y,r,a in self.dust:
            xx = (x + t * (4+r*2)) % OUT_W
            yy = (y + math.sin(t*0.8+x)*4) % OUT_H
            dr.line((xx,yy,xx+6*SCALE+r*3*SCALE,yy),fill=COLORS["earth_glow"]+(int(a*0.18),),width=1)
        offset = int((t*41)%9)
        for y in range(offset,OUT_H,9):
            dr.line((0,y,OUT_W,y),fill=(130,175,190,5),width=1)
        image.alpha_composite(ov)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        name = str(shot["name"])
        image = self.background(t)
        if name == "opening":
            self.draw_opening(image,t,shot)
        elif name == "catalog_cloud":
            self.draw_catalog_cloud(image,t,shot)
        elif name == "altitude_bands":
            self.draw_altitude_bands(image,t,shot)
        elif name == "families":
            self.draw_families(image,t,shot)
        elif name == "scale_gap":
            self.draw_scale_gap(image,t,shot)
        elif name == "all_debris":
            self.draw_all_debris(image,t,shot)
        else:
            self.draw_finale(image,t,shot)
        self.draw_header(image,name)
        self.draw_source(image)
        self.draw_film_texture(image,t)

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        arr *= VIGNETTE[...,None]
        arr = np.clip(arr,0,255).astype(np.uint8)
        img = ImageEnhance.Contrast(Image.fromarray(arr)).enhance(float(CONFIG["contrast"]))
        img = ImageEnhance.Color(img).enhance(float(CONFIG["saturation"]))
        arr = np.asarray(img,dtype=np.int16)
        rng = np.random.default_rng(int(t*1000)+971)
        grain = rng.normal(0.0,float(CONFIG["grain_strength"]),arr.shape[:2])[:,:,None]
        return np.clip(arr+grain,0,255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Audio / video
# -----------------------------------------------------------------------------

def gaussian_envelope(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((times-center)/max(width,1e-6))**2)


def generate_soundtrack(path: Path) -> Path:
    sr = int(CONFIG["sample_rate"])
    dur = float(CONFIG["duration_s"])
    n = int(round(sr*dur))
    times = np.arange(n,dtype=np.float64)/sr
    rng = np.random.default_rng(9191)
    audio = 0.075*np.sin(math.tau*31.0*times + 0.3*np.sin(math.tau*0.045*times))
    audio += 0.042*np.sin(math.tau*47.5*times + 1.7)
    audio += 0.018*np.sin(math.tau*95.0*times)
    controls = rng.normal(0,1,max(8,int(dur*4)))
    audio += 0.018*np.interp(times,np.linspace(0,dur,len(controls)),controls)
    sweep = next(s for s in SHOT_PLAN if s["name"]=="catalog_cloud")
    for frac in np.linspace(0.08,0.95,18 if not QUICK_MODE else 6):
        center = lerp(sweep["start"],sweep["end"],float(frac))
        env = gaussian_envelope(times,center,0.055 if not QUICK_MODE else 0.09)
        audio += env * 0.032*np.sin(math.tau*(260+180*frac)*times)
    for scene,strength in [("altitude_bands",0.08),("families",0.10),("scale_gap",0.14),("all_debris",0.12),("finale",0.16)]:
        s = next(x for x in SHOT_PLAN if x["name"]==scene)
        center = s["start"]+0.24*(s["end"]-s["start"])
        env = gaussian_envelope(times,center,0.7 if not QUICK_MODE else 0.22)
        audio += env*strength*np.sin(math.tau*42.0*times)
    intro = smooth_array(np.clip(times/max(1.5,dur*0.08),0,1))
    outro_x = np.clip((times-(dur-1.2))/1.0,0,1)
    outro = 1.0-smooth_array(outro_x)
    audio *= intro*outro
    peak=max(float(np.max(np.abs(audio))),1e-9)
    pcm=(np.clip(audio/peak*0.88,-1,1)*32767).astype(np.int16)
    with wave.open(str(path),"wb") as h:
        h.setnchannels(1); h.setsampwidth(2); h.setframerate(sr); h.writeframes(pcm.tobytes())
    return path


def smooth_array(x: np.ndarray) -> np.ndarray:
    return x*x*(3.0-2.0*x)


def find_ffmpeg() -> Optional[str]:
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return shutil.which("ffmpeg")


def mux_audio(video: Path, audio: Path, output: Path) -> bool:
    ffmpeg=find_ffmpeg()
    if not ffmpeg:
        return False
    cmd=[ffmpeg,"-y","-i",str(video),"-i",str(audio),"-c:v","copy","-c:a","aac","-b:a","192k","-shortest",str(output)]
    try:
        subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return output.exists() and output.stat().st_size>0
    except Exception:
        return False


def render_video(scene: DebrisScene) -> Path:
    base=str(CONFIG["output_basename"])
    srt=write_srt(CAPTIONS,OUTPUT_ROOT/f"{base}.srt")
    raw=OUTPUT_ROOT/f"{base}_silent.mp4"
    final=OUTPUT_ROOT/f"{base}_final.mp4"
    wav=OUTPUT_ROOT/f"{base}_ambient.wav"
    frame_count=int(round(float(CONFIG["duration_s"])*int(CONFIG["fps"])))
    times=np.arange(frame_count)/int(CONFIG["fps"])
    print("Subtitle sidecar:",srt.resolve())
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(raw,fps=int(CONFIG["fps"]),codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=None) as writer:
        for t in tqdm(times,desc="Rendering space-debris short"):
            writer.append_data(scene.render_frame(float(t)))
    generate_soundtrack(wav)
    if mux_audio(raw,wav,final):
        print("Final video with audio:",final.resolve())
        return final
    shutil.copyfile(raw,final)
    print("Audio mux unavailable; copied silent video to:",final.resolve())
    return final

def main():
    print("SPACE DEBRIS YouTube Short")
    print("Loading CelesTrak SATCAT debris records ...")
    records,source,notes,cache=load_debris()
    summary=summarize(records,source)
    csv_path,json_path=save_data(records,summary,notes)
    print("Data source:",source)
    print("Catalogued debris loaded:",f"{len(records):,}")
    print("Regimes:",summary["regime_counts"])
    print("CSV:",csv_path.resolve())
    print("Summary:",json_path.resolve())
    if cache:
        print("Cache:",cache.resolve())
    for n in notes:
        print("Data note:",n)

    scene=DebrisScene(records,summary)
    preview_times=[1.2,min(10.0,float(CONFIG["duration_s"])*0.23),min(28.0,float(CONFIG["duration_s"])*0.49),min(38.0,float(CONFIG["duration_s"])*0.67),min(47.0,float(CONFIG["duration_s"])*0.82),float(CONFIG["duration_s"])-0.6]
    for t in tqdm(preview_times,desc="Preview frames"):
        Image.fromarray(scene.render_frame(float(t))).save(PREVIEW_DIR/f"preview_{int(t):02d}s.png")
    render_video(scene)
    print("Output directory:",OUTPUT_ROOT.resolve())



if __name__ == "__main__":
    main()

