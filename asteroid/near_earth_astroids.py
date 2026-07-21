
from __future__ import annotations

"""
EVERY KNOWN NEAR-EARTH ASTEROID — vertical cinematic data short
================================================================

This script downloads the current NASA/JPL Small-Body Database (SBDB) catalog
of known near-Earth asteroids and renders a vertical 1080x1920 YouTube Short.

Every catalog point comes from the JPL SBDB Query API:
    https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html

#### renderer method : youtube shorts 
The renderer uses:
- JPL osculating orbital elements and epochs,
- vectorized two-body propagation for cinematic motion,
- Atira / Aten / Apollo / Amor orbit classes,
- JPL's PHA flag and Earth MOID,
- measured diameter when available,
- absolute magnitude H when diameter is unavailable,
- observation-arc and orbit-condition metadata.

Scientific fidelity notes
-------------------------
1. This is a catalog visualization, not an impact prediction.
2. "Potentially hazardous asteroid" is a screening category, not a statement
   that an impact is expected.
3. The moving cloud is propagated with a two-body Kepler model from each
   object's SBDB osculating elements. High-precision trajectories require JPL
   Horizons / numerical n-body integrations.
4. Asteroid marks are enormously enlarged for visibility.
5. The overview uses a declared asinh radial compression so that highly
   eccentric NEOs can remain visible in one vertical frame.
6. "Every known" means every asteroid returned by the live JPL NEO catalog at
   download time. The known catalog changes as discoveries and orbit solutions
   are added or revised.

Install:
    pip install numpy pandas pillow imageio imageio-ffmpeg requests tqdm

Render:
    python every_known_near_earth_asteroid_short.py

Fast test:
    NEA_SHORT_QUICK=1 python every_known_near_earth_asteroid_short.py

Force a fresh catalog download:
    NEA_SHORT_REFRESH=1 python every_known_near_earth_asteroid_short.py
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.getenv("NEA_SHORT_QUICK", "0") == "1"
FORCE_REFRESH = os.getenv("NEA_SHORT_REFRESH", "0") == "1"

OUTPUT_ROOT = Path("every_known_near_earth_asteroid_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"

for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, object] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 8 if QUICK_MODE else 24,
    "duration_s": 18.0 if QUICK_MODE else 58.0,
    "output_basename": "every_known_near_earth_asteroid",

    "api_url": "https://ssd-api.jpl.nasa.gov/sbdb_query.api",
    "api_expected_version": "1.0",
    "page_size": 10000,
    "request_timeout": (20, 180),
    "request_retries": 5,
    "retry_backoff_s": 1.5,

    # The complete current catalog is rendered. No object-count sampling.
    "fields": [
        "spkid", "pdes", "name", "full_name", "class", "pha", "neo",
        "epoch", "e", "a", "q", "ad", "i", "om", "w", "ma", "n",
        "per_y", "moid", "H", "diameter", "diameter_sigma", "albedo",
        "condition_code", "data_arc", "first_obs", "last_obs",
        "n_obs_used", "source", "soln_date",
    ],

    # Playback covers one synthetic calendar year, driven by the real elements.
    "playback_days": 365.25,

    # The display is radially compressed and labelled as such.
    "radial_softening_au": 0.55,
    "radial_percentile": 99.7,
    "radial_min_cap_au": 5.0,
    "radial_max_cap_au": 80.0,

    "background_star_count": 520 if QUICK_MODE else 1100,
    "hud_noise_count": 55 if QUICK_MODE else 100,
    "density_render_scale": 0.58 if QUICK_MODE else 0.48,

    "title_text": "EVERY KNOWN NEAR-EARTH ASTEROID",
    "subtitle_text": "A live NASA/JPL SBDB catalog snapshot",
    "credit_text": "Data: NASA/JPL Small-Body Database Query API",
    "scientific_note": (
        "Two-body playback from JPL osculating elements; radial distance is "
        "asinh-compressed and object marks are enlarged. Not an impact forecast."
    ),

    "audio_path": None,
    "burn_subtitles": False,
    "write_subtitle_sidecar": True,
}

WIDTH = int(CONFIG["video_width"])
HEIGHT = int(CONFIG["video_height"])
OUT_SIZE = (WIDTH, HEIGHT)
SCALE = WIDTH / 1080.0

CLASS_LABELS = {
    "IEO": "ATIRA",
    "ATE": "ATEN",
    "APO": "APOLLO",
    "AMO": "AMOR",
}

CLASS_COLORS = {
    "IEO": (145, 110, 255),
    "ATE": (70, 225, 255),
    "APO": (255, 105, 82),
    "AMO": (255, 205, 85),
    "OTHER": (190, 205, 225),
}

CAPTIONS_FULL = [
    (0.5, 6.5, "Every point is a known near-Earth asteroid in JPL's current catalog."),
    (7.0, 18.0, "Their orbits approach within 1.3 astronomical units of the Sun."),
    (18.5, 30.5, "Four main orbit families thread around Earth's path."),
    (31.0, 42.5, "Potentially hazardous is a screening category — not a predicted impact."),
    (43.0, 51.0, "Many asteroid sizes are estimated from brightness because no diameter is measured."),
    (51.5, 57.5, "Known does not mean complete. The catalog keeps growing."),
]

SHOT_PLAN_FULL = [
    {"name": "reveal", "start": 0.0, "end": 7.0, "zoom": (0.78, 0.98), "tilt": (0.48, 0.58)},
    {"name": "cloud", "start": 7.0, "end": 19.0, "zoom": (0.98, 1.08), "tilt": (0.58, 0.66)},
    {"name": "families", "start": 19.0, "end": 31.0, "zoom": (1.08, 1.02), "tilt": (0.66, 0.58)},
    {"name": "pha", "start": 31.0, "end": 43.0, "zoom": (1.02, 1.18), "tilt": (0.58, 0.70)},
    {"name": "sizes", "start": 43.0, "end": 51.5, "zoom": (1.18, 0.92), "tilt": (0.70, 0.52)},
    {"name": "outro", "start": 51.5, "end": 58.0, "zoom": (0.92, 0.72), "tilt": (0.52, 0.44)},
]


def remap_timeline(items: Sequence, source_duration: float, target_duration: float):
    factor = target_duration / source_duration
    output = []
    for item in items:
        if isinstance(item, tuple):
            output.append((item[0] * factor, item[1] * factor, item[2]))
        else:
            copied = dict(item)
            copied["start"] *= factor
            copied["end"] *= factor
            output.append(copied)
    return output


DURATION = float(CONFIG["duration_s"])
CAPTIONS = remap_timeline(CAPTIONS_FULL, 58.0, DURATION)
SHOT_PLAN = remap_timeline(SHOT_PLAN_FULL, 58.0, DURATION)


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


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


def deterministic_unit(text: str, salt: str = "") -> float:
    digest = hashlib.sha256(f"{text}|{salt}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64 - 1)


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def find_ffmpeg() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def get_font(size: int, bold: bool = False):
    size = max(8, int(size))
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
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
    xy: Tuple[float, float],
    size: int,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    anchor: str = "la",
    stroke: int = 2,
):
    draw = ImageDraw.Draw(image)
    draw.text(
        xy,
        text,
        font=get_font(int(size * SCALE), bold=bold),
        fill=fill,
        anchor=anchor,
        stroke_width=max(1, int(stroke * SCALE)),
        stroke_fill=(0, 0, 0, min(225, fill[3] if len(fill) > 3 else 225)),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    line_spacing: int = 8,
):
    draw = ImageDraw.Draw(image)
    font = get_font(int(size * SCALE), bold=bold)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        box = draw.textbbox((0, 0), candidate, font=font, stroke_width=max(1, int(2 * SCALE)))
        if box[2] - box[0] <= max_width:
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
            (x, y), line, font=font, fill=fill,
            stroke_width=max(1, int(2 * SCALE)), stroke_fill=(0, 0, 0, 220),
        )
        box = draw.textbbox((x, y), line, font=font, stroke_width=max(1, int(2 * SCALE)))
        y += box[3] - box[1] + int(line_spacing * SCALE)


def make_vignette(width: int, height: int, strength: float = 0.33) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.8, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(WIDTH, HEIGHT)


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for index, (start, end, text) in enumerate(captions, 1):
        lines += [str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# JPL catalog retrieval
# -----------------------------------------------------------------------------


def build_retry_session() -> requests.Session:
    retries = Retry(
        total=int(CONFIG["request_retries"]),
        connect=int(CONFIG["request_retries"]),
        read=int(CONFIG["request_retries"]),
        status=int(CONFIG["request_retries"]),
        backoff_factor=float(CONFIG["retry_backoff_s"]),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retries, pool_connections=2, pool_maxsize=2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "every-known-near-earth-asteroid-short/1.0 educational-visualization"
    })
    return session


def parse_api_payload(response: requests.Response) -> dict:
    if response.status_code >= 400:
        raise RuntimeError(
            f"JPL SBDB Query API HTTP {response.status_code}: {response.reason}\n"
            f"Response preview: {response.text[:1200]}"
        )
    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(
            "JPL SBDB Query API did not return valid JSON.\n"
            f"Response preview: {response.text[:1200]}"
        ) from exc


def query_current_count(session: requests.Session) -> int:
    response = session.get(
        str(CONFIG["api_url"]),
        params={"sb-kind": "a", "sb-group": "neo"},
        timeout=CONFIG["request_timeout"],
    )
    payload = parse_api_payload(response)
    return int(payload.get("count", 0))


def fetch_live_catalog(force_refresh: bool = False) -> pd.DataFrame:
    csv_path = DATA_ROOT / "jpl_all_known_near_earth_asteroids.csv"
    metadata_path = DATA_ROOT / "jpl_all_known_near_earth_asteroids_metadata.json"

    if csv_path.exists() and metadata_path.exists() and not force_refresh:
        print("Using cached JPL near-Earth asteroid catalog:", csv_path)
        return clean_catalog(pd.read_csv(csv_path), save_path=None)

    session = build_retry_session()
    count = query_current_count(session)
    if count <= 0:
        raise RuntimeError("JPL returned zero known near-Earth asteroids.")

    print(f"JPL reports {count:,} known near-Earth asteroids for this query.")
    fields = list(CONFIG["fields"])
    page_size = int(CONFIG["page_size"])
    pages: List[pd.DataFrame] = []
    signatures: List[dict] = []

    for offset in tqdm(range(0, count, page_size), desc="Downloading JPL SBDB catalog"):
        params = {
            "fields": ",".join(fields),
            "sb-kind": "a",
            "sb-group": "neo",
            "sort": "spkid",
            "limit": page_size,
            "limit-from": offset,
            "full-prec": "true",
        }
        response = session.get(
            str(CONFIG["api_url"]), params=params, timeout=CONFIG["request_timeout"]
        )
        payload = parse_api_payload(response)
        returned_fields = payload.get("fields") or fields
        data = payload.get("data") or []
        if not data:
            break
        pages.append(pd.DataFrame(data, columns=returned_fields))
        signatures.append(payload.get("signature") or {})

    if not pages:
        raise RuntimeError("No rows were returned by the JPL SBDB Query API.")

    raw = pd.concat(pages, ignore_index=True)
    catalog = clean_catalog(raw, save_path=csv_path)

    metadata = {
        "acquired_utc": datetime.now(timezone.utc).isoformat(),
        "query": {"sb-kind": "a", "sb-group": "neo"},
        "requested_count": count,
        "rows_downloaded": int(len(raw)),
        "rows_retained": int(len(catalog)),
        "fields": fields,
        "api_url": CONFIG["api_url"],
        "signatures": signatures[:3],
        "scientific_note": CONFIG["scientific_note"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return catalog


def clean_catalog(df: pd.DataFrame, save_path: Optional[Path]) -> pd.DataFrame:
    df = df.copy()
    required = ["spkid", "class", "epoch", "e", "a", "i", "om", "w", "ma"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"JPL catalog is missing required fields: {missing}")

    numeric_columns = [
        "epoch", "e", "a", "q", "ad", "i", "om", "w", "ma", "n", "per_y",
        "moid", "H", "diameter", "diameter_sigma", "albedo", "condition_code",
        "data_arc", "n_obs_used",
    ]
    for column in numeric_columns:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    string_columns = [
        "spkid", "pdes", "name", "full_name", "class", "pha", "neo", "first_obs",
        "last_obs", "source", "soln_date",
    ]
    for column in string_columns:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str).str.strip()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["epoch", "e", "a", "i", "om", "w", "ma"]).copy()
    df = df[(df["a"] > 0) & (df["e"] >= 0) & (df["e"] < 1)].copy()
    df = df.drop_duplicates("spkid").reset_index(drop=True)

    # Derive mean motion if it is absent. Gaussian two-body approximation:
    # Earth is approximately 0.9856076686 degrees/day at a=1 au.
    missing_n = ~np.isfinite(df["n"]) | (df["n"] <= 0)
    df.loc[missing_n, "n"] = 0.9856076686 / np.power(df.loc[missing_n, "a"], 1.5)

    df["class_group"] = df["class"].where(df["class"].isin(CLASS_LABELS), "OTHER")
    df["is_pha"] = df["pha"].str.upper().eq("Y")
    df["has_measured_diameter"] = np.isfinite(df["diameter"]) & (df["diameter"] > 0)

    # Illustrative diameter estimate from H at assumed geometric albedo 0.14.
    # D(km) = 1329/sqrt(p) * 10^(-H/5)
    df["diameter_estimate_km_p014"] = (
        1329.0 / math.sqrt(0.14) * np.power(10.0, -df["H"] / 5.0)
    )
    df.loc[~np.isfinite(df["H"]), "diameter_estimate_km_p014"] = np.nan

    df["display_name"] = df["full_name"]
    blank = df["display_name"].eq("")
    df.loc[blank, "display_name"] = df.loc[blank, "pdes"]
    df.loc[df["display_name"].eq(""), "display_name"] = df.loc[df["display_name"].eq(""), "spkid"]

    if save_path is not None:
        df.to_csv(save_path, index=False)
        print("Saved cleaned JPL catalog:", save_path.resolve())

    return df


# -----------------------------------------------------------------------------
# Orbital propagation and projection
# -----------------------------------------------------------------------------


def timestamp_to_jd(timestamp: pd.Timestamp) -> float:
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return float(timestamp.to_julian_date())


def solve_kepler_vectorized(mean_anomaly: np.ndarray, eccentricity: np.ndarray) -> np.ndarray:
    M = np.mod(mean_anomaly, 2.0 * np.pi)
    E = np.where(eccentricity < 0.8, M, np.pi)
    for _ in range(8):
        f = E - eccentricity * np.sin(E) - M
        fp = 1.0 - eccentricity * np.cos(E)
        E -= f / np.maximum(fp, 1e-12)
    return E


@dataclass
class OrbitArrays:
    epoch: np.ndarray
    e: np.ndarray
    a: np.ndarray
    inc: np.ndarray
    node: np.ndarray
    argp: np.ndarray
    ma: np.ndarray
    n: np.ndarray


def build_orbit_arrays(df: pd.DataFrame) -> OrbitArrays:
    return OrbitArrays(
        epoch=df["epoch"].to_numpy(float),
        e=df["e"].to_numpy(float),
        a=df["a"].to_numpy(float),
        inc=np.radians(df["i"].to_numpy(float)),
        node=np.radians(df["om"].to_numpy(float)),
        argp=np.radians(df["w"].to_numpy(float)),
        ma=np.radians(df["ma"].to_numpy(float)),
        n=np.radians(df["n"].to_numpy(float)),
    )


def propagate_xyz(orbits: OrbitArrays, jd: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = orbits.ma + orbits.n * (jd - orbits.epoch)
    E = solve_kepler_vectorized(M, orbits.e)

    x_orbit = orbits.a * (np.cos(E) - orbits.e)
    y_orbit = orbits.a * np.sqrt(np.maximum(1.0 - orbits.e**2, 0.0)) * np.sin(E)

    cos_O = np.cos(orbits.node)
    sin_O = np.sin(orbits.node)
    cos_i = np.cos(orbits.inc)
    sin_i = np.sin(orbits.inc)
    cos_w = np.cos(orbits.argp)
    sin_w = np.sin(orbits.argp)

    # Perifocal to heliocentric J2000-ecliptic rotation.
    p11 = cos_O * cos_w - sin_O * sin_w * cos_i
    p12 = -cos_O * sin_w - sin_O * cos_w * cos_i
    p21 = sin_O * cos_w + cos_O * sin_w * cos_i
    p22 = -sin_O * sin_w + cos_O * cos_w * cos_i
    p31 = sin_w * sin_i
    p32 = cos_w * sin_i

    x = p11 * x_orbit + p12 * y_orbit
    y = p21 * x_orbit + p22 * y_orbit
    z = p31 * x_orbit + p32 * y_orbit
    return x, y, z


def earth_xyz(jd: float, reference_jd: float) -> Tuple[float, float, float]:
    angle = 2.0 * math.pi * ((jd - reference_jd) / 365.256363004)
    return math.cos(angle), math.sin(angle), 0.0


# -----------------------------------------------------------------------------
# Cinematic scene
# -----------------------------------------------------------------------------


class NearEarthAsteroidScene:
    def __init__(self, catalog: pd.DataFrame):
        self.catalog = catalog.reset_index(drop=True).copy()
        self.orbits = build_orbit_arrays(self.catalog)
        self.count = len(self.catalog)
        self.reference_timestamp = pd.Timestamp.now(tz="UTC").floor("D")
        self.reference_jd = timestamp_to_jd(self.reference_timestamp)

        self.classes = self.catalog["class_group"].to_numpy(object)
        self.is_pha = self.catalog["is_pha"].to_numpy(bool)
        self.h = self.catalog["H"].to_numpy(float)
        self.moid = self.catalog["moid"].to_numpy(float)
        self.diameter = self.catalog["diameter"].to_numpy(float)
        self.has_diameter = self.catalog["has_measured_diameter"].to_numpy(bool)
        self.names = self.catalog["display_name"].to_numpy(object)
        self.spkids = self.catalog["spkid"].to_numpy(object)

        # Determine a stable radial cap from a reference propagation.
        x, y, z = propagate_xyz(self.orbits, self.reference_jd)
        r = np.sqrt(x*x + y*y + z*z)
        finite_r = r[np.isfinite(r)]
        percentile = np.percentile(finite_r, float(CONFIG["radial_percentile"]))
        self.radial_cap = float(np.clip(
            percentile,
            float(CONFIG["radial_min_cap_au"]),
            float(CONFIG["radial_max_cap_au"]),
        ))

        self.class_masks = {
            code: self.classes == code for code in ["IEO", "ATE", "APO", "AMO", "OTHER"]
        }
        self.class_counts = {code: int(mask.sum()) for code, mask in self.class_masks.items()}
        self.pha_count = int(self.is_pha.sum())
        self.measured_diameter_count = int(self.has_diameter.sum())

        finite_h = self.h[np.isfinite(self.h)]
        self.h_median = float(np.median(finite_h)) if len(finite_h) else float("nan")

        self.stars = self._make_stars(int(CONFIG["background_star_count"]))
        self.noise = self._make_noise(int(CONFIG["hud_noise_count"]))
        self.highlight_indices = self._find_highlights()

    def _make_stars(self, count: int):
        rng = np.random.default_rng(2019)
        return [
            (
                float(rng.uniform(0, WIDTH)),
                float(rng.uniform(0, HEIGHT)),
                float(rng.uniform(0.35, 1.8) * SCALE),
                int(rng.integers(20, 120)),
                float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(count)
        ]

    def _make_noise(self, count: int):
        rng = np.random.default_rng(808)
        return [
            (
                float(rng.uniform(0, WIDTH)),
                float(rng.uniform(0, HEIGHT)),
                float(rng.uniform(8, 90) * SCALE),
                int(rng.integers(10, 55)),
                float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(count)
        ]

    def _find_highlights(self) -> np.ndarray:
        desired = ["Apophis", "Bennu", "Eros", "Didymos", "Ryugu", "Toutatis"]
        indices: List[int] = []
        names_lower = pd.Series(self.names).astype(str).str.lower()
        for name in desired:
            matches = np.where(names_lower.str.contains(name.lower(), regex=False).to_numpy())[0]
            if len(matches):
                indices.append(int(matches[0]))
        if len(indices) < 6:
            candidates = np.argsort(np.nan_to_num(self.moid, nan=999.0))
            for idx in candidates:
                if int(idx) not in indices:
                    indices.append(int(idx))
                if len(indices) >= 6:
                    break
        return np.array(indices[:6], dtype=int)

    def get_shot(self, t: float) -> dict:
        for shot in SHOT_PLAN:
            if shot["start"] <= t < shot["end"]:
                return shot
        return SHOT_PLAN[-1]

    def shot_state(self, t: float):
        shot = self.get_shot(t)
        u = clamp((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-6))
        e = ease_in_out_sine(u)
        zoom = lerp(shot["zoom"][0], shot["zoom"][1], e)
        tilt = lerp(shot["tilt"][0], shot["tilt"][1], e)
        return shot, zoom, tilt, u

    def playback_fraction(self, t: float) -> float:
        return smoothstep((t - 3.0 * DURATION / 58.0) / (DURATION - 6.0 * DURATION / 58.0))

    def playback_jd(self, t: float) -> float:
        return self.reference_jd + self.playback_fraction(t) * float(CONFIG["playback_days"])

    def compressed_xyz(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        r = np.sqrt(x*x + y*y + z*z)
        soft = float(CONFIG["radial_softening_au"])
        denom = math.asinh(self.radial_cap / soft)
        mapped_r = np.arcsinh(np.minimum(r, self.radial_cap) / soft) / denom
        inv_r = np.divide(1.0, r, out=np.zeros_like(r), where=r > 1e-12)
        return x * inv_r * mapped_r, y * inv_r * mapped_r, z * inv_r * mapped_r, r

    def project(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
        t: float, zoom: float, tilt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        angle = 0.26 * t + 0.18 * math.sin(t * 0.13)
        ca, sa = math.cos(angle), math.sin(angle)
        xr = ca * x - sa * y
        yr = sa * x + ca * y

        # A cinematic oblique projection. The underlying orbital positions remain 3D.
        yp = yr * tilt - z * (0.75 + 0.18 * math.sin(t * 0.17))
        depth = yr * (1.0 - tilt) + z * 0.35

        center_x = WIDTH * (0.5 + 0.015 * math.sin(t * 0.11))
        center_y = HEIGHT * (0.505 + 0.018 * math.cos(t * 0.09))
        radius_px = min(WIDTH * 0.46, HEIGHT * 0.305) * zoom
        sx = center_x + xr * radius_px
        sy = center_y + yp * radius_px
        return sx, sy, depth

    def render_background(self, t: float) -> Image.Image:
        canvas = Image.new("RGBA", OUT_SIZE, (1, 3, 10, 255))
        draw = ImageDraw.Draw(canvas)
        for x, y, radius, alpha, phase in self.stars:
            twinkle = 0.65 + 0.35 * math.sin(t * 0.9 + phase)
            draw.ellipse(
                (x-radius, y-radius, x+radius, y+radius),
                fill=(205, 225, 255, int(alpha * twinkle)),
            )

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for cx, cy, radius, color, alpha in [
            (WIDTH * 0.50, HEIGHT * 0.50, WIDTH * 0.62, (10, 75, 130), 32),
            (WIDTH * 0.18, HEIGHT * 0.23, WIDTH * 0.36, (70, 25, 110), 24),
            (WIDTH * 0.84, HEIGHT * 0.76, WIDTH * 0.42, (0, 95, 100), 22),
        ]:
            gd.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(*color, alpha))
        glow = glow.filter(ImageFilter.GaussianBlur(max(10, int(85 * SCALE))))
        canvas.alpha_composite(glow)
        return canvas

    def draw_reference_orbits(self, canvas: Image.Image, t: float, zoom: float, tilt: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)

        # Circular reference tracks for Mercury, Venus, Earth and Mars.
        for radius_au, label, color in [
            (0.387, "MERCURY", (120, 150, 175, 45)),
            (0.723, "VENUS", (205, 175, 105, 50)),
            (1.000, "EARTH", (75, 180, 255, 125)),
            (1.524, "MARS", (220, 105, 75, 55)),
        ]:
            angles = np.linspace(0, 2*np.pi, 190)
            x = radius_au * np.cos(angles)
            y = radius_au * np.sin(angles)
            z = np.zeros_like(x)
            cx, cy, cz, _ = self.compressed_xyz(x, y, z)
            sx, sy, _ = self.project(cx, cy, cz, t, zoom, tilt)
            points = [(float(px), float(py)) for px, py in zip(sx, sy)]
            d.line(points, fill=color, width=max(1, int((2 if radius_au == 1 else 1) * SCALE)))

        canvas.alpha_composite(overlay)

    def rasterize_density(
        self,
        sx: np.ndarray,
        sy: np.ndarray,
        masks_and_colors: Sequence[Tuple[np.ndarray, Tuple[int, int, int]]],
        alpha_scale: float,
        blur_px: float = 1.6,
    ) -> Image.Image:
        scale = float(CONFIG["density_render_scale"])
        rw = max(180, int(WIDTH * scale))
        rh = max(320, int(HEIGHT * scale))
        result = np.zeros((rh, rw, 4), dtype=np.float32)

        xi_all = np.floor(sx * scale).astype(np.int32)
        yi_all = np.floor(sy * scale).astype(np.int32)
        valid_base = (xi_all >= 0) & (xi_all < rw) & (yi_all >= 0) & (yi_all < rh)

        for mask, color in masks_and_colors:
            valid = valid_base & mask
            if not np.any(valid):
                continue
            xi = xi_all[valid]
            yi = yi_all[valid]
            counts = np.zeros((rh, rw), dtype=np.float32)
            np.add.at(counts, (yi, xi), 1.0)
            intensity = 1.0 - np.exp(-counts * 0.95)
            result[..., 0] += intensity * color[0]
            result[..., 1] += intensity * color[1]
            result[..., 2] += intensity * color[2]
            result[..., 3] += intensity * 255.0 * alpha_scale

        result[..., :3] = np.clip(result[..., :3], 0, 255)
        result[..., 3] = np.clip(result[..., 3], 0, 255)
        image = Image.fromarray(result.astype(np.uint8), mode="RGBA")
        if blur_px > 0:
            image = image.filter(ImageFilter.GaussianBlur(max(0.4, blur_px * scale)))
        image = image.resize(OUT_SIZE, Image.Resampling.BILINEAR)
        return image

    def draw_asteroids(self, canvas: Image.Image, t: float, zoom: float, tilt: float, shot_name: str):
        jd = self.playback_jd(t)
        x, y, z = propagate_xyz(self.orbits, jd)
        cx, cy, cz, actual_r = self.compressed_xyz(x, y, z)
        sx, sy, depth = self.project(cx, cy, cz, t, zoom, tilt)

        reveal_alpha = smoothstep(t / max(2.8 * DURATION / 58.0, 1e-6))
        if shot_name == "families":
            masks_colors = [
                (self.class_masks[code], CLASS_COLORS[code])
                for code in ["IEO", "ATE", "APO", "AMO", "OTHER"]
            ]
            alpha = 0.82 * reveal_alpha
        elif shot_name == "pha":
            masks_colors = [
                (~self.is_pha, (55, 80, 105)),
                (self.is_pha, (255, 108, 66)),
            ]
            alpha = 0.92
        elif shot_name == "sizes":
            masks_colors = [
                (~self.has_diameter, (65, 125, 160)),
                (self.has_diameter, (255, 210, 92)),
            ]
            alpha = 0.86
        else:
            masks_colors = [
                (np.ones(self.count, dtype=bool), (85, 220, 245)),
            ]
            alpha = 0.78 * reveal_alpha

        glow = self.rasterize_density(sx, sy, masks_colors, alpha_scale=alpha * 0.42, blur_px=4.2)
        sharp = self.rasterize_density(sx, sy, masks_colors, alpha_scale=alpha, blur_px=0.45)
        canvas.alpha_composite(glow)
        canvas.alpha_composite(sharp)

        # Earth marker propagated on a circular reference orbit.
        ex, ey, ez = earth_xyz(jd, self.reference_jd)
        ecx, ecy, ecz, _ = self.compressed_xyz(
            np.array([ex]), np.array([ey]), np.array([ez])
        )
        esx, esy, _ = self.project(ecx, ecy, ecz, t, zoom, tilt)
        self.draw_earth_marker(canvas, float(esx[0]), float(esy[0]), t)

        # A few named examples are called out only after the catalog is visible.
        if t > 9.0 * DURATION / 58.0 and shot_name not in {"sizes", "outro"}:
            self.draw_highlight_labels(canvas, sx, sy, t, shot_name)

        return jd, actual_r

    def draw_earth_marker(self, canvas: Image.Image, x: float, y: float, t: float):
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        pulse = 10 * SCALE + 3 * SCALE * math.sin(t * 4.2)
        d.ellipse((x-pulse, y-pulse, x+pulse, y+pulse), outline=(80, 205, 255, 220), width=max(1, int(2*SCALE)))
        d.ellipse((x-4*SCALE, y-4*SCALE, x+4*SCALE, y+4*SCALE), fill=(190, 240, 255, 245))
        draw_text(layer, "EARTH", (x+15*SCALE, y-2*SCALE), 18, (170, 230, 255, 230), True, "lm", 1)
        canvas.alpha_composite(layer)

    def draw_highlight_labels(self, canvas: Image.Image, sx: np.ndarray, sy: np.ndarray, t: float, shot_name: str):
        d = ImageDraw.Draw(canvas)
        for rank, idx in enumerate(self.highlight_indices):
            x, y = float(sx[idx]), float(sy[idx])
            if not (30*SCALE < x < WIDTH-30*SCALE and 180*SCALE < y < HEIGHT-330*SCALE):
                continue
            if shot_name == "pha" and not self.is_pha[idx]:
                continue
            phase = deterministic_unit(str(self.spkids[idx]), "label") * 2 * math.pi
            radius = (13 + 3 * math.sin(t * 3.3 + phase)) * SCALE
            d.arc((x-radius, y-radius, x+radius, y+radius), 5, 130, fill=(255, 210, 110, 190), width=max(1, int(2*SCALE)))
            d.arc((x-radius, y-radius, x+radius, y+radius), 185, 310, fill=(255, 210, 110, 190), width=max(1, int(2*SCALE)))
            label = str(self.names[idx]).strip()[:22]
            draw_text(canvas, label, (x+18*SCALE, y-12*SCALE), 15, (245, 245, 250, 220), True, "la", 1)

    def draw_sun(self, canvas: Image.Image, t: float):
        cx = WIDTH * 0.5
        cy = HEIGHT * 0.505
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for radius, alpha in [(44, 20), (28, 42), (16, 100)]:
            r = radius * SCALE
            d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(255, 175, 55, alpha))
        layer = layer.filter(ImageFilter.GaussianBlur(max(2, int(8*SCALE))))
        canvas.alpha_composite(layer)
        d = ImageDraw.Draw(canvas)
        r = 6.5 * SCALE
        d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(255, 244, 195, 255))
        draw_text(canvas, "SUN", (cx, cy+20*SCALE), 16, (255, 215, 135, 230), True, "ma", 1)

    def draw_class_panel(self, canvas: Image.Image, t: float, shot_name: str):
        if shot_name != "families":
            return
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(panel)
        x0, y0 = int(50*SCALE), int(220*SCALE)
        w, h = int(410*SCALE), int(300*SCALE)
        d.rounded_rectangle((x0, y0, x0+w, y0+h), radius=int(22*SCALE), fill=(2, 7, 16, 180), outline=(80, 190, 225, 85), width=max(1, int(2*SCALE)))
        canvas.alpha_composite(panel)
        draw_text(canvas, "ORBIT FAMILIES", (x0+20*SCALE, y0+22*SCALE), 22, (225, 242, 250, 245), True, "la", 1)
        total_main = max(sum(self.class_counts[c] for c in ["IEO", "ATE", "APO", "AMO"]), 1)
        for row, code in enumerate(["IEO", "ATE", "APO", "AMO"]):
            y = y0 + (72 + row*52)*SCALE
            color = (*CLASS_COLORS[code], 245)
            d = ImageDraw.Draw(canvas)
            d.ellipse((x0+20*SCALE, y-6*SCALE, x0+32*SCALE, y+6*SCALE), fill=color)
            draw_text(canvas, CLASS_LABELS[code], (x0+46*SCALE, y), 19, color, True, "lm", 1)
            draw_text(canvas, f"{self.class_counts[code]:,}", (x0+w-20*SCALE, y), 19, (245, 248, 252, 240), True, "rm", 1)

    def draw_pha_panel(self, canvas: Image.Image, shot_name: str):
        if shot_name != "pha":
            return
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(panel)
        x0, y0 = int(55*SCALE), int(230*SCALE)
        w, h = int(500*SCALE), int(245*SCALE)
        d.rounded_rectangle((x0, y0, x0+w, y0+h), radius=int(24*SCALE), fill=(5, 5, 12, 188), outline=(255, 110, 72, 110), width=max(1, int(2*SCALE)))
        canvas.alpha_composite(panel)
        draw_text(canvas, "POTENTIALLY HAZARDOUS", (x0+22*SCALE, y0+24*SCALE), 22, (255, 145, 98, 250), True, "la", 1)
        draw_text(canvas, f"{self.pha_count:,} catalog objects", (x0+22*SCALE, y0+70*SCALE), 31, (250, 250, 252, 250), True, "la", 2)
        draw_text(canvas, "JPL flag based on orbit proximity and brightness", (x0+22*SCALE, y0+120*SCALE), 17, (205, 220, 232, 230), False, "la", 1)
        draw_wrapped_text(
            canvas,
            "It does not mean an impact is predicted.",
            (int(x0+22*SCALE), int(y0+154*SCALE)),
            max_width=int((w-44*SCALE)), size=21,
            fill=(255, 210, 175, 245), bold=True,
        )

    def draw_size_panel(self, canvas: Image.Image, shot_name: str):
        if shot_name != "sizes":
            return
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(panel)
        x0, y0 = int(50*SCALE), int(210*SCALE)
        w, h = int(540*SCALE), int(280*SCALE)
        d.rounded_rectangle((x0, y0, x0+w, y0+h), radius=int(24*SCALE), fill=(3, 7, 15, 188), outline=(255, 210, 92, 95), width=max(1, int(2*SCALE)))
        canvas.alpha_composite(panel)
        draw_text(canvas, "WHAT SIZE ARE THEY?", (x0+22*SCALE, y0+24*SCALE), 23, (255, 220, 115, 250), True, "la", 1)
        draw_text(canvas, f"Measured diameter: {self.measured_diameter_count:,}", (x0+22*SCALE, y0+78*SCALE), 24, (250, 250, 252, 250), True, "la", 2)
        unknown = self.count - self.measured_diameter_count
        draw_text(canvas, f"No measured diameter: {unknown:,}", (x0+22*SCALE, y0+123*SCALE), 22, (155, 215, 240, 245), True, "la", 1)
        draw_wrapped_text(
            canvas,
            "For many objects, H brightness plus an assumed reflectivity gives only an estimated size.",
            (int(x0+22*SCALE), int(y0+166*SCALE)),
            max_width=int(w-44*SCALE), size=18,
            fill=(205, 220, 235, 240),
        )

    def draw_top_hud(self, canvas: Image.Image, t: float, shot_name: str, jd: float):
        if t < 6.5 * DURATION / 58.0:
            alpha = int(255 * smoothstep(t / max(1.1 * DURATION / 58.0, 1e-6)))
            draw_text(canvas, CONFIG["title_text"], (52*SCALE, 96*SCALE), 46, (246, 249, 252, alpha), True, "la", 3)
            draw_text(canvas, CONFIG["subtitle_text"], (55*SCALE, 158*SCALE), 23, (105, 225, 245, min(alpha, 230)), True, "la", 1)
        else:
            headings = {
                "cloud": "THE ORBITAL CLOUD",
                "families": "FOUR MAIN NEA FAMILIES",
                "pha": "THE HAZARD-SCREENING SUBSET",
                "sizes": "BRIGHTNESS IS NOT A RULER",
                "outro": "THE CATALOG IS STILL GROWING",
                "reveal": "JPL SBDB CATALOG",
            }
            draw_text(canvas, headings.get(shot_name, "NEAR-EARTH ASTEROIDS"), (52*SCALE, 70*SCALE), 23, (160, 225, 242, 225), True, "la", 1)

        date = pd.to_datetime(jd, unit="D", origin="julian", utc=True)
        draw_text(canvas, f"CATALOG OBJECTS // {self.count:,}", (WIDTH-52*SCALE, 88*SCALE), 21, (115, 230, 245, 235), True, "ra", 1)
        draw_text(canvas, f"PLAYBACK // {date.strftime('%Y-%m-%d')}", (WIDTH-52*SCALE, 122*SCALE), 17, (175, 205, 225, 215), False, "ra", 1)
        draw_text(canvas, "RADIAL DISPLAY // ASINH COMPRESSED", (WIDTH-52*SCALE, 151*SCALE), 15, (160, 190, 210, 190), False, "ra", 1)

    def draw_counter_reveal(self, canvas: Image.Image, t: float):
        end = 6.0 * DURATION / 58.0
        if t > end:
            return
        u = smoothstep(t / max(end, 1e-6))
        shown = int(round(self.count * u))
        draw_text(canvas, f"{shown:,}", (WIDTH*0.5, HEIGHT*0.775), 72, (245, 248, 252, 255), True, "ma", 3)
        draw_text(canvas, "KNOWN NEAR-EARTH ASTEROIDS", (WIDTH*0.5, HEIGHT*0.82), 22, (100, 225, 245, 235), True, "ma", 1)

    def draw_timeline(self, canvas: Image.Image, t: float):
        x0, x1 = 68*SCALE, WIDTH-68*SCALE
        y = HEIGHT-310*SCALE
        d = ImageDraw.Draw(canvas)
        d.line((x0, y, x1, y), fill=(75, 180, 220, 190), width=max(1, int(2*SCALE)))
        f = self.playback_fraction(t)
        cx = lerp(x0, x1, f)
        d.line((cx, y-27*SCALE, cx, y+27*SCALE), fill=(255, 185, 82, 245), width=max(2, int(3*SCALE)))
        draw_text(canvas, "ONE-YEAR TWO-BODY PLAYBACK", (x0, y-48*SCALE), 17, (160, 210, 230, 220), True, "la", 1)
        draw_text(canvas, "DAY 0", (x0, y+26*SCALE), 15, (155, 190, 210, 200), False, "la", 1)
        draw_text(canvas, "DAY 365", (x1, y+26*SCALE), 15, (155, 190, 210, 200), False, "ra", 1)

    def draw_caption(self, canvas: Image.Image, t: float):
        text = None
        for start, end, caption in CAPTIONS:
            if start <= t < end:
                text = caption
                break
        if not text:
            return
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(panel)
        y0 = int(HEIGHT-242*SCALE)
        d.rounded_rectangle((42*SCALE, y0, WIDTH-42*SCALE, y0+126*SCALE), radius=int(24*SCALE), fill=(1, 4, 12, 178), outline=(55, 170, 215, 72), width=max(1, int(SCALE)))
        canvas.alpha_composite(panel)
        draw_wrapped_text(canvas, text, (int(68*SCALE), int(y0+25*SCALE)), int(WIDTH-136*SCALE), 29, (245, 249, 252, 248))

    def draw_outro(self, canvas: Image.Image, t: float, shot_name: str):
        if shot_name != "outro":
            return
        alpha = int(245 * smoothstep((t - 52.0*DURATION/58.0) / max(2.3*DURATION/58.0, 1e-6)))
        draw_text(canvas, "KNOWN ≠ COMPLETE", (WIDTH*0.5, HEIGHT*0.28), 48, (250, 250, 252, alpha), True, "ma", 3)
        draw_wrapped_text(
            canvas,
            "New discoveries and improved orbit solutions keep changing this picture.",
            (int(110*SCALE), int(HEIGHT*0.33)), int(WIDTH-220*SCALE), 27,
            (180, 225, 240, alpha), True,
        )
        draw_text(canvas, CONFIG["credit_text"], (55*SCALE, HEIGHT-104*SCALE), 17, (215, 228, 240, alpha), False, "la", 1)
        draw_wrapped_text(canvas, CONFIG["scientific_note"], (int(55*SCALE), int(HEIGHT-78*SCALE)), int(WIDTH-110*SCALE), 14, (175, 200, 220, alpha))

    def draw_noise(self, canvas: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for x, y, length, alpha, phase in self.noise:
            pulse = 0.5 + 0.5*math.sin(t*1.8 + phase)
            if pulse > 0.75:
                yy = (y + t*9*SCALE) % HEIGHT
                d.line((x, yy, x+length, yy), fill=(90, 205, 235, int(alpha*pulse)), width=1)
        offset = int((t*37) % max(3, int(7*SCALE)))
        step = max(3, int(7*SCALE))
        for y in range(offset, HEIGHT, step):
            d.line((0, y, WIDTH, y), fill=(120, 200, 235, 11), width=1)
        scan_y = int((t*148*SCALE) % (HEIGHT+180*SCALE)) - int(90*SCALE)
        d.rectangle((0, scan_y, WIDTH, scan_y+42*SCALE), fill=(80, 210, 240, 8))
        canvas.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot, zoom, tilt, _ = self.shot_state(t)
        canvas = self.render_background(t)
        self.draw_reference_orbits(canvas, t, zoom, tilt)
        jd, actual_r = self.draw_asteroids(canvas, t, zoom, tilt, shot["name"])
        self.draw_sun(canvas, t)
        self.draw_class_panel(canvas, t, shot["name"])
        self.draw_pha_panel(canvas, shot["name"])
        self.draw_size_panel(canvas, shot["name"])
        self.draw_top_hud(canvas, t, shot["name"], jd)
        self.draw_counter_reveal(canvas, t)
        self.draw_timeline(canvas, t)
        self.draw_caption(canvas, t)
        self.draw_outro(canvas, t, shot["name"])
        self.draw_noise(canvas, t)

        image = canvas.convert("RGB")
        image = ImageEnhance.Contrast(image).enhance(1.13)
        image = ImageEnhance.Color(image).enhance(1.08)
        arr = np.asarray(image).astype(np.float32)
        arr *= VIGNETTE[..., None]

        fade_in = smoothstep(t / max(0.85*DURATION/58.0, 1e-6))
        fade_out = 1.0 - smoothstep((t - (DURATION-1.0*DURATION/58.0)) / max(0.9*DURATION/58.0, 1e-6))
        arr *= fade_in * fade_out
        return np.clip(arr, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Summary files and rendering
# -----------------------------------------------------------------------------


def create_catalog_summary(catalog: pd.DataFrame) -> dict:
    class_counts = catalog["class_group"].value_counts(dropna=False).to_dict()
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "known_near_earth_asteroids": int(len(catalog)),
        "orbit_class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "potentially_hazardous_count": int(catalog["is_pha"].sum()),
        "measured_diameter_count": int(catalog["has_measured_diameter"].sum()),
        "without_measured_diameter_count": int((~catalog["has_measured_diameter"]).sum()),
        "median_absolute_magnitude_H": float(catalog["H"].median(skipna=True)),
        "median_earth_moid_au": float(catalog["moid"].median(skipna=True)),
        "source": "NASA/JPL SBDB Query API",
        "query": "sb-kind=a, sb-group=neo",
        "notes": [
            "PHA is a screening category, not an impact prediction.",
            "Estimated diameter uses H and assumed albedo 0.14 only where shown.",
            "Video motion uses two-body propagation from osculating elements.",
        ],
    }
    path = DATA_ROOT / "catalog_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def create_contact_sheet(preview_paths: Sequence[Path]) -> Optional[Path]:
    if not preview_paths:
        return None
    images = [Image.open(path).convert("RGB") for path in preview_paths]
    thumb_w = 270 if not QUICK_MODE else 180
    thumb_h = int(thumb_w * HEIGHT / WIDTH)
    margin = 18
    sheet = Image.new("RGB", (thumb_w*3 + margin*4, thumb_h*2 + margin*3), (5, 7, 13))
    for index, image in enumerate(images[:6]):
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = margin + (index % 3) * (thumb_w + margin)
        y = margin + (index // 3) * (thumb_h + margin)
        sheet.paste(thumb, (x, y))
    path = PREVIEW_DIR / "every_known_near_earth_asteroid_contact_sheet.jpg"
    sheet.save(path, quality=92)
    return path


def run_ffmpeg(command: List[str]):
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def render_video(scene: NearEarthAsteroidScene) -> Path:
    basename = str(CONFIG["output_basename"])
    raw_path = OUTPUT_ROOT / f"{basename}_raw.mp4"
    final_path = OUTPUT_ROOT / f"{basename}_final.mp4"
    subbed_path = OUTPUT_ROOT / f"{basename}_subbed.mp4"
    audio_path_out = OUTPUT_ROOT / f"{basename}_with_audio.mp4"
    srt_path = OUTPUT_ROOT / f"{basename}.srt"

    if CONFIG.get("write_subtitle_sidecar", True):
        write_srt(CAPTIONS, srt_path)
        print("Subtitle sidecar:", srt_path.resolve())

    frame_count = int(round(DURATION * int(CONFIG["fps"])))
    times = np.arange(frame_count, dtype=float) / int(CONFIG["fps"])
    print(f"Rendering {frame_count:,} frames at {WIDTH}x{HEIGHT}...")

    with iio.get_writer(
        raw_path,
        fps=int(CONFIG["fps"]),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times, desc="Rendering every known NEA"):
            writer.append_data(scene.render_frame(float(t)))

    candidate = raw_path
    ffmpeg = find_ffmpeg()

    if CONFIG.get("burn_subtitles", False) and ffmpeg and srt_path.exists():
        run_ffmpeg([
            ffmpeg, "-y", "-i", str(candidate),
            "-vf", f"subtitles={srt_path}:force_style=Fontname=DejaVu Sans,Fontsize=22,Outline=1.2,BorderStyle=3,MarginV=90",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", str(subbed_path),
        ])
        candidate = subbed_path

    audio_path = CONFIG.get("audio_path")
    if audio_path and Path(str(audio_path)).exists() and ffmpeg:
        run_ffmpeg([
            ffmpeg, "-y", "-i", str(candidate), "-i", str(audio_path),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(audio_path_out),
        ])
        candidate = audio_path_out

    shutil.copyfile(candidate, final_path)
    print("Final video:", final_path.resolve())
    return final_path


def main():
    print("Starting Every Known Near-Earth Asteroid pipeline...")
    print("Quick mode:", QUICK_MODE)
    catalog = fetch_live_catalog(force_refresh=FORCE_REFRESH)
    summary = create_catalog_summary(catalog)

    print(f"Catalog rows retained: {len(catalog):,}")
    print(f"PHA flag count: {summary['potentially_hazardous_count']:,}")
    print(f"Measured diameters: {summary['measured_diameter_count']:,}")

    scene = NearEarthAsteroidScene(catalog)
    preview_times_full = [1.3, 10.0, 24.0, 36.0, 47.0, 56.0]
    preview_times = [value * DURATION / 58.0 for value in preview_times_full]
    preview_paths: List[Path] = []

    for index, t in enumerate(tqdm(preview_times, desc="Preview frames"), 1):
        frame = scene.render_frame(float(t))
        path = PREVIEW_DIR / f"preview_{index:02d}_{t:05.2f}s.png"
        Image.fromarray(frame).save(path)
        preview_paths.append(path)

    contact_sheet = create_contact_sheet(preview_paths)
    if contact_sheet:
        print("Contact sheet:", contact_sheet.resolve())

    render_video(scene)
    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if path.is_file():
            print("-", path.relative_to(OUTPUT_ROOT))


if __name__ == "__main__":
    main()
