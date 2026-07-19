from __future__ import annotations

"""
NASA TRACKED THIS ASTEROID FOR DECADES — Apophis cinematic data short
====================================================================

A vertical 1080x1920 YouTube Short about asteroid 99942 Apophis.

The story uses official NASA/JPL facts and a JPL Horizons Earth-centered
trajectory around the April 13, 2029 close approach. On an internet-connected
run, the script requests a fresh Horizons vector table. A small official
Horizons snapshot is embedded as a reproducible fallback.



Scientific fidelity notes
-------------------------
1. The 2029 Earth-centered path is interpolated from JPL Horizons state vectors
   when online. The embedded fallback contains official Horizons samples near
   closest approach.
2. The 2004 risk sequence visualizes an uncertainty corridor. A 2.7% impact
   probability meant Earth overlapped a large early orbit-uncertainty region;
   it was not a deterministic prediction of impact.
3. The procedural asteroid shape, telescope, radar dish, dust and spacecraft
   are explanatory artwork. Apophis has no high-resolution surface imagery.
4. Earth, geosynchronous altitude and the Horizons flyby scale are linearly
   related in the close-approach chapter, but the asteroid marker is enlarged.
5. NASA states Apophis will safely pass about 32,000 km above Earth's surface
   on April 13, 2029, and poses no impact risk for at least 100 years.

Install:
    pip install numpy pandas pillow imageio imageio-ffmpeg requests tqdm

Render:
    python nasa_tracked_this_asteroid_for_decades_short.py

Fast test:
    APOPHIS_SHORT_QUICK=1 python nasa_tracked_this_asteroid_for_decades_short.py

Force a fresh Horizons request:
    APOPHIS_SHORT_REFRESH=1 python nasa_tracked_this_asteroid_for_decades_short.py
"""

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

QUICK_MODE = os.getenv("APOPHIS_SHORT_QUICK", "0") == "1"
FORCE_REFRESH = os.getenv("APOPHIS_SHORT_REFRESH", "0") == "1"

OUTPUT_ROOT = Path("nasa_tracked_apophis_for_decades_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, object] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 16.0 if QUICK_MODE else 58.0,
    "output_basename": "nasa_tracked_this_asteroid_for_decades",

    "horizons_api_url": "https://ssd.jpl.nasa.gov/api/horizons.api",
    "horizons_start": "2029-04-13 20:30",
    "horizons_stop": "2029-04-13 23:00",
    "horizons_step": "1 m",
    "request_timeout": (20, 180),
    "request_retries": 4,
    "retry_backoff_s": 1.5,

    "earth_radius_km": 6378.137,
    "geo_altitude_km": 35786.0,
    "moon_distance_km": 384400.0,
    "apophis_mean_diameter_m": 340.0,
    "apophis_long_axis_m": 450.0,
    "apophis_rotation_h": 31.0,
    "apophis_rocking_h": 264.0,

    "background_star_count": 390 if QUICK_MODE else 850,
    "dust_particle_count": 110 if QUICK_MODE else 260,
    "hud_noise_count": 35 if QUICK_MODE else 80,

    "title_text": "NASA TRACKED THIS ASTEROID FOR DECADES",
    "subtitle_text": "99942 APOPHIS // 2004 → 2029",
    "credit_text": "Data: NASA/JPL CNEOS, Goldstone radar and JPL Horizons",
    "scientific_note": (
        "The asteroid marker and explanatory hardware are enlarged. The 2029 "
        "Earth-centered trajectory is driven by JPL Horizons vectors."
    ),

    "audio_path": None,
    "burn_subtitles": False,
    "write_subtitle_sidecar": True,
}

WIDTH = int(CONFIG["video_width"])
HEIGHT = int(CONFIG["video_height"])
OUT_SIZE = (WIDTH, HEIGHT)
SCALE = WIDTH / 1080.0
DURATION = float(CONFIG["duration_s"])


CAPTIONS_FULL = [
    (0.5, 6.5, "In 2004, astronomers found a 340-metre asteroid crossing Earth's orbit."),
    (7.0, 17.5, "With only a short observation arc, Earth overlapped the asteroid's uncertainty corridor."),
    (18.0, 29.5, "Optical images and radar echoes kept shrinking the range of possible paths."),
    (30.0, 43.0, "JPL Horizons places the 2029 pass about 32,000 kilometres above Earth's surface."),
    (43.5, 51.0, "Earth's gravity will bend the orbit and may change how the asteroid tumbles."),
    (51.5, 57.5, "The result of decades of tracking: Apophis is not an impact threat for at least 100 years."),
]

SHOT_PLAN_FULL = [
    {"name": "discovery", "start": 0.0, "end": 7.0},
    {"name": "uncertainty", "start": 7.0, "end": 18.0},
    {"name": "radar", "start": 18.0, "end": 30.0},
    {"name": "flyby", "start": 30.0, "end": 43.5},
    {"name": "gravity", "start": 43.5, "end": 51.5},
    {"name": "safe", "start": 51.5, "end": 58.0},
]

MILESTONES = [
    {
        "date": "2004-06-19",
        "year_label": "2004",
        "title": "DISCOVERY",
        "detail": "Kitt Peak // 2004 MN4",
        "status": "two-night discovery arc",
    },
    {
        "date": "2004-12-20",
        "year_label": "2004",
        "title": "RECOVERED",
        "detail": "Siding Spring observations",
        "status": "orbit linked again",
    },
    {
        "date": "2004-12-27",
        "year_label": "2004",
        "title": "EARLY RISK PEAK",
        "detail": "briefly 2.7% for 2029",
        "status": "large uncertainty corridor",
    },
    {
        "date": "2005-01-27",
        "year_label": "2005",
        "title": "RADAR",
        "detail": "Arecibo ranging campaign",
        "status": "distance and velocity refined",
    },
    {
        "date": "2013-01-09",
        "year_label": "2013",
        "title": "CLOSE OBSERVING PASS",
        "detail": "Goldstone + Arecibo radar",
        "status": "2036 impact ruled out",
    },
    {
        "date": "2021-03-08",
        "year_label": "2021",
        "title": "PRECISION RADAR",
        "detail": "Goldstone + Green Bank",
        "status": "range accuracy about 150 m",
    },
    {
        "date": "2021-03-25",
        "year_label": "2021",
        "title": "RISK REMOVED",
        "detail": "2068 possibility eliminated",
        "status": "safe for 100+ years",
    },
    {
        "date": "2029-04-13",
        "year_label": "2029",
        "title": "HISTORIC FLYBY",
        "detail": "about 32,000 km altitude",
        "status": "safe close approach",
    },
    {
        "date": "2029-06-01",
        "year_label": "2029",
        "title": "OSIRIS-APEX",
        "detail": "spacecraft rendezvous",
        "status": "post-flyby science",
    },
]

# Official JPL Horizons samples near closest approach. The columns are
# Earth-centered J2000 ecliptic vectors in km and km/s. The closest sample is
# about 38,011 km from Earth's center, or about 31,633 km above the reference
# equatorial radius used here.
FALLBACK_HORIZONS_ROWS = [
    ("2029-Apr-13 21:40:07.2000", -20977.35372471809, 31283.63152449978, 5483.140961522571, 6.290913661197688, 3.470508261865354, 1.856282069257982, 38062.84021794153, -0.3472773742028405),
    ("2029-Apr-13 21:41:01.2000", -20637.42440357804, 31470.70845249874, 5583.322011166836, 6.299040959239424, 3.458251934784281, 1.854120188158626, 38045.79163581636, -0.2841363707724236),
    ("2029-Apr-13 21:41:55.2000", -20297.05952170491, 31657.12115066571, 5683.385188972457, 6.307045073735152, 3.445907753586182, 1.851916532155664, 38032.15497420294, -0.2209129368034556),
    ("2029-Apr-13 21:42:49.2000", -19956.26580226421, 31842.86496751693, 5783.328253780341, 6.314923407199982, 3.433479126557995, 1.849671653665465, 38021.93419331941, -0.1576253189518573),
    ("2029-Apr-13 21:43:43.2000", -19615.05010601878, 32027.93543765031, 5883.148995215557, 6.322673461351584, 3.420969538980661, 1.847386140449769, 38015.13226545226, -0.09429185406414986),
    ("2029-Apr-13 21:44:37.2000", -19273.41942387819, 32212.32828308473, 5982.845235802728, 6.330292840922189, 3.408382548389778, 1.845060614876165, 38011.75116752073, -0.03093094346897193),
    ("2029-Apr-13 21:44:58.8000", -19136.65254601836, 32285.89476616475, 6022.688390473735, 6.333303487532032, 3.403326889155427, 1.844119347053573, 38011.35681566291, -0.005583024308693571),
    ("2029-Apr-13 21:45:20.4000", -18999.82086855173, 32359.35192066401, 6062.511146216211, 6.336292717190213, 3.398259659582398, 1.843171825351337, 38011.50998264119, 0.01976515483933656),
    ("2029-Apr-13 21:46:14.4000", -18657.46123403311, 32542.51488972356, 6161.977908803639, 6.343671188669049, 3.385542520517435, 1.840775953215431, 38014.28819267713, 0.08312898935365387),
    ("2029-Apr-13 21:47:08.4000", -18314.70684599876, 32724.98930373012, 6261.314262583186, 6.350912856142251, 3.372758281883587, 1.838341988239132, 38020.48747818670, 0.1464686229401884),
    ("2029-Apr-13 21:48:02.4000", -17971.56514620781, 32906.77164224705, 6360.518170442634, 6.358015746127775, 3.359910729987276, 1.835870676206884, 38030.10603608564, 0.2097656455362943),
    ("2029-Apr-13 21:49:07.2000", -17559.29448539019, 33123.99225581904, 6479.385223960620, 6.366353423483067, 3.344415495159553, 1.832856897991648, 38046.15773857715, 0.2856399949253567),
    ("2029-Apr-13 21:50:01.2000", -17215.32739308476, 33304.24065058347, 6578.290785377163, 6.373144640348730, 3.331442207575828, 1.830306193667832, 38063.28745944115, 0.3487788034572551),
    ("2029-Apr-13 21:51:06.0000", -16802.08862620592, 33519.61191518996, 6696.794236740898, 6.381103506799462, 3.315807414225584, 1.827199534435538, 38088.33966185805, 0.4244104760461518),
    ("2029-Apr-13 21:52:00.0000", -16457.33360758424, 33698.31256755872, 6795.392266436920, 6.387575364244952, 3.302727347220431, 1.824573489360027, 38112.95652959710, 0.4873033747411517),
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
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def find_ffmpeg() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return found
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
    size: int = 42,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    stroke: int = 2,
    anchor: str = "la",
):
    draw = ImageDraw.Draw(image)
    font = get_font(int(size * SCALE), bold=bold)
    scaled_xy = (int(xy[0] * SCALE), int(xy[1] * SCALE))
    draw.text(
        scaled_xy,
        text,
        font=font,
        fill=fill,
        anchor=anchor,
        stroke_width=max(0, int(stroke * SCALE)),
        stroke_fill=(0, 0, 0, min(230, fill[3] if len(fill) > 3 else 230)),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[float, float],
    max_width: float,
    size: int = 31,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    line_spacing: int = 8,
):
    draw = ImageDraw.Draw(image)
    font = get_font(int(size * SCALE), bold=bold)
    max_width_px = int(max_width * SCALE)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=max(1, int(2 * SCALE)))
        if bbox[2] - bbox[0] <= max_width_px:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x = int(xy[0] * SCALE)
    y = int(xy[1] * SCALE)
    for line in lines:
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
            stroke_width=max(1, int(2 * SCALE)),
            stroke_fill=(0, 0, 0, 220),
        )
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=max(1, int(2 * SCALE)))
        y += bbox[3] - bbox[1] + int(line_spacing * SCALE)


def draw_round_panel(
    image: Image.Image,
    box: Tuple[float, float, float, float],
    fill=(2, 7, 18, 180),
    outline=(65, 190, 225, 90),
    radius: float = 22,
    width: float = 2,
):
    d = ImageDraw.Draw(image)
    scaled = tuple(int(v * SCALE) for v in box)
    d.rounded_rectangle(
        scaled,
        radius=max(2, int(radius * SCALE)),
        fill=fill,
        outline=outline,
        width=max(1, int(width * SCALE)),
    )


def make_vignette(width: int, height: int, strength: float = 0.30) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.8, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(WIDTH, HEIGHT)


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


def get_shot(t: float) -> Dict:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def local_shot_fraction(t: float, shot: Dict) -> float:
    return clamp((t - float(shot["start"])) / max(float(shot["end"]) - float(shot["start"]), 1e-6))


# -----------------------------------------------------------------------------
# Real data acquisition and reproducible fallbacks
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
    session.headers.update({"User-Agent": "apophis-decades-short/1.0 educational visualization"})
    return session


def fallback_horizons_dataframe() -> pd.DataFrame:
    rows = []
    for date_text, x, y, z, vx, vy, vz, rg, rr in FALLBACK_HORIZONS_ROWS:
        rows.append({
            "time_utc": pd.to_datetime(date_text, format="%Y-%b-%d %H:%M:%S.%f", utc=True),
            "x_km": x,
            "y_km": y,
            "z_km": z,
            "vx_km_s": vx,
            "vy_km_s": vy,
            "vz_km_s": vz,
            "range_km": rg,
            "range_rate_km_s": rr,
            "source": "embedded official JPL Horizons snapshot",
        })
    return pd.DataFrame(rows)


def parse_horizons_result(result_text: str) -> pd.DataFrame:
    if "$$SOE" not in result_text or "$$EOE" not in result_text:
        raise ValueError("Horizons result is missing ephemeris delimiters")
    body = result_text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    parsed_rows = []
    date_pattern = re.compile(r"(?:A\.D\.\s+)?\d{4}-[A-Za-z]{3}-\d{2}\s+\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?")

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tokens = [token.strip() for token in next(csv.reader([line]))]
        date_index = None
        date_text = None
        for index, token in enumerate(tokens):
            match = date_pattern.search(token)
            if match:
                date_index = index
                date_text = match.group(0).replace("A.D. ", "")
                break
        if date_index is None or date_text is None:
            continue

        values: List[float] = []
        for token in tokens[date_index + 1:]:
            try:
                values.append(float(token))
            except Exception:
                continue
        if len(values) < 9:
            continue

        time_value = pd.to_datetime(date_text, utc=True, errors="coerce")
        if pd.isna(time_value):
            continue
        x, y, z, vx, vy, vz, _light_time, rg, rr = values[:9]
        parsed_rows.append({
            "time_utc": time_value,
            "x_km": x,
            "y_km": y,
            "z_km": z,
            "vx_km_s": vx,
            "vy_km_s": vy,
            "vz_km_s": vz,
            "range_km": rg,
            "range_rate_km_s": rr,
            "source": "live JPL Horizons API",
        })

    frame = pd.DataFrame(parsed_rows)
    if len(frame) < 3:
        raise ValueError("Could not parse enough Horizons vector rows")
    return frame.sort_values("time_utc").drop_duplicates("time_utc").reset_index(drop=True)


def fetch_horizons_vectors(force_refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_ROOT / "apophis_2029_jpl_horizons_vectors.csv"
    metadata_path = DATA_ROOT / "apophis_2029_jpl_horizons_metadata.json"

    if cache_path.exists() and not force_refresh:
        frame = pd.read_csv(cache_path)
        frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["time_utc", "x_km", "y_km", "z_km"])
        if len(frame) >= 3:
            return frame.reset_index(drop=True)

    params = {
        "format": "json",
        "COMMAND": "'99942'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@399'",
        "START_TIME": f"'{CONFIG['horizons_start']}'",
        "STOP_TIME": f"'{CONFIG['horizons_stop']}'",
        "STEP_SIZE": f"'{CONFIG['horizons_step']}'",
        "OUT_UNITS": "'KM-S'",
        "REF_PLANE": "'ECLIPTIC'",
        "REF_SYSTEM": "'J2000'",
        "VEC_TABLE": "'3'",
        "VEC_LABELS": "'YES'",
        "CSV_FORMAT": "'YES'",
        "TIME_TYPE": "'UTC'",
        "TIME_DIGITS": "'SECONDS'",
        "VECT_CORR": "'NONE'",
        "OBJ_DATA": "'YES'",
    }

    source_mode = "fallback"
    error_text = None
    try:
        session = build_retry_session()
        response = session.get(
            str(CONFIG["horizons_api_url"]),
            params=params,
            timeout=CONFIG["request_timeout"],
        )
        response.raise_for_status()
        payload = response.json()
        frame = parse_horizons_result(str(payload.get("result", "")))
        source_mode = "live JPL Horizons API"
    except Exception as exc:
        error_text = repr(exc)
        print("Horizons request unavailable; using embedded official snapshot.")
        print("Reason:", error_text)
        frame = fallback_horizons_dataframe()

    save_frame = frame.copy()
    save_frame["time_utc"] = save_frame["time_utc"].astype(str)
    save_frame.to_csv(cache_path, index=False)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_mode": source_mode,
        "request_url": CONFIG["horizons_api_url"],
        "request_parameters": params,
        "fallback_error": error_text,
        "reference_frame": "Earth-centered J2000 ecliptic geometric vectors",
        "units": "km and km/s",
        "scientific_note": "Use JPL Horizons directly for precision mission or hazard analysis.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame.reset_index(drop=True)


def write_story_data(ephemeris: pd.DataFrame):
    milestone_path = DATA_ROOT / "apophis_tracking_milestones_2004_2029.csv"
    pd.DataFrame(MILESTONES).to_csv(milestone_path, index=False)

    closest_index = int(ephemeris["range_km"].astype(float).idxmin())
    closest = ephemeris.loc[closest_index]
    speed = math.sqrt(
        float(closest["vx_km_s"]) ** 2
        + float(closest["vy_km_s"]) ** 2
        + float(closest["vz_km_s"]) ** 2
    )
    summary = {
        "object": "99942 Apophis (2004 MN4)",
        "discovery_date": "2004-06-19",
        "mean_diameter_m": CONFIG["apophis_mean_diameter_m"],
        "long_axis_at_least_m": CONFIG["apophis_long_axis_m"],
        "rotation_period_h_approx": CONFIG["apophis_rotation_h"],
        "rocking_period_h_approx": CONFIG["apophis_rocking_h"],
        "early_2029_impact_probability_peak": 0.027,
        "current_impact_risk_statement": "No impact risk for at least 100 years (NASA/JPL)",
        "closest_sample_time_utc": str(closest["time_utc"]),
        "closest_sample_geocentric_distance_km": float(closest["range_km"]),
        "closest_sample_altitude_above_reference_earth_radius_km": float(closest["range_km"]) - float(CONFIG["earth_radius_km"]),
        "closest_sample_geocentric_speed_km_s": speed,
        "official_public_rounding": "about 32,000 km above Earth's surface on 2029-04-13",
        "osiris_apex_rendezvous": "June 2029",
    }
    (DATA_ROOT / "apophis_story_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------


class ApophisScene:
    def __init__(self, ephemeris: pd.DataFrame, summary: Dict):
        self.ephemeris = ephemeris.copy().sort_values("time_utc").reset_index(drop=True)
        self.summary = summary
        self.times = self.ephemeris["time_utc"].astype("int64").to_numpy(dtype=np.float64) / 1e9
        self.positions = self.ephemeris[["x_km", "y_km", "z_km"]].to_numpy(float)
        self.velocities = self.ephemeris[["vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)
        self.ranges = self.ephemeris["range_km"].to_numpy(float)
        self.closest_index = int(np.argmin(self.ranges))
        self.closest_position = self.positions[self.closest_index]
        self.closest_velocity = self.velocities[self.closest_index]
        self.closest_range = float(self.ranges[self.closest_index])
        self.closest_altitude = self.closest_range - float(CONFIG["earth_radius_km"])
        self.closest_speed = float(np.linalg.norm(self.closest_velocity))

        forward = self.closest_velocity / max(np.linalg.norm(self.closest_velocity), 1e-9)
        side_raw = self.closest_position - np.dot(self.closest_position, forward) * forward
        side = side_raw / max(np.linalg.norm(side_raw), 1e-9)
        depth = np.cross(forward, side)
        depth = depth / max(np.linalg.norm(depth), 1e-9)
        self.flyby_forward = forward
        self.flyby_side = side
        self.flyby_depth = depth

        self.stars = self._make_stars(int(CONFIG["background_star_count"]), 173)
        self.dust = self._make_dust(int(CONFIG["dust_particle_count"]), 271)
        self.hud_noise = self._make_hud_noise(int(CONFIG["hud_noise_count"]), 337)
        self.asteroid_points = self._make_asteroid_polygon(64)
        self.craters = self._make_craters(18)

    @staticmethod
    def _make_stars(count: int, seed: int):
        rng = np.random.default_rng(seed)
        return [
            (
                float(rng.uniform(0, WIDTH)),
                float(rng.uniform(0, HEIGHT)),
                float(rng.uniform(0.35, 1.65) * SCALE),
                int(rng.integers(35, 155)),
                float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(count)
        ]

    @staticmethod
    def _make_dust(count: int, seed: int):
        rng = np.random.default_rng(seed)
        return [
            (
                float(rng.uniform(0, WIDTH)),
                float(rng.uniform(0, HEIGHT)),
                float(rng.uniform(0.5, 2.8) * SCALE),
                int(rng.integers(10, 75)),
                float(rng.uniform(-20, 20) * SCALE),
                float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(count)
        ]

    @staticmethod
    def _make_hud_noise(count: int, seed: int):
        rng = np.random.default_rng(seed)
        return [
            (
                float(rng.uniform(0, WIDTH)),
                float(rng.uniform(0, HEIGHT)),
                float(rng.uniform(12, 100) * SCALE),
                int(rng.integers(8, 45)),
                float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(count)
        ]

    @staticmethod
    def _make_asteroid_polygon(count: int):
        points = []
        for index in range(count):
            angle = 2 * math.pi * index / count
            lobe = 1.0 + 0.18 * math.cos(2 * angle)
            noise = 0.87 + 0.23 * deterministic_unit(str(index), "apophis-edge")
            radius = lobe * noise
            points.append((math.cos(angle) * radius, math.sin(angle) * radius))
        return points

    @staticmethod
    def _make_craters(count: int):
        craters = []
        for index in range(count):
            angle = deterministic_unit(str(index), "crater-angle") * 2 * math.pi
            radial = math.sqrt(deterministic_unit(str(index), "crater-r")) * 0.72
            radius = 0.035 + 0.09 * deterministic_unit(str(index), "crater-size")
            craters.append((math.cos(angle) * radial, math.sin(angle) * radial, radius))
        return craters

    def render_background(self, t: float) -> Image.Image:
        canvas = Image.new("RGBA", OUT_SIZE, (1, 3, 10, 255))
        draw = ImageDraw.Draw(canvas)
        for x0, y0, radius, alpha, phase in self.stars:
            x = (x0 + 1.5 * SCALE * t) % WIDTH
            y = (y0 + 0.35 * SCALE * t) % HEIGHT
            twinkle = 0.68 + 0.32 * math.sin(t * 1.15 + phase)
            a = int(alpha * twinkle)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(205, 225, 255, a))

        nebula = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        nd = ImageDraw.Draw(nebula)
        for cx, cy, rx, ry, color in [
            (WIDTH * 0.20, HEIGHT * 0.24, WIDTH * 0.44, HEIGHT * 0.25, (18, 52, 118, 34)),
            (WIDTH * 0.78, HEIGHT * 0.62, WIDTH * 0.40, HEIGHT * 0.24, (100, 24, 90, 25)),
            (WIDTH * 0.50, HEIGHT * 0.48, WIDTH * 0.55, HEIGHT * 0.36, (0, 108, 138, 18)),
        ]:
            nd.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color)
        nebula = nebula.filter(ImageFilter.GaussianBlur(max(10, int(80 * SCALE))))
        canvas.alpha_composite(nebula)

        d = ImageDraw.Draw(canvas)
        for x0, y0, radius, alpha, drift, phase in self.dust:
            x = (x0 + drift * t * 0.05) % WIDTH
            y = (y0 + drift * t * 0.015) % HEIGHT
            pulse = 0.5 + 0.5 * math.sin(t * 0.7 + phase)
            a = int(alpha * (0.45 + 0.55 * pulse))
            d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(160, 190, 215, a))
        return canvas

    def draw_earth(self, canvas: Image.Image, center: Tuple[float, float], radius: float, t: float, show_geo: bool = False):
        cx, cy = center
        r = radius
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for factor, alpha in [(1.55, 22), (1.30, 42), (1.12, 85)]:
            rr = r * factor
            gd.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(30, 125, 255, alpha))
        glow = glow.filter(ImageFilter.GaussianBlur(max(4, int(24 * SCALE))))
        canvas.alpha_composite(glow)

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(8, 42, 98, 255), outline=(110, 220, 255, 245), width=max(1, int(3 * SCALE)))
        shadow_offset = r * 0.42
        d.ellipse((cx - r + shadow_offset, cy - r * 1.03, cx + r * 1.22 + shadow_offset, cy + r * 1.03), fill=(1, 5, 17, 214))
        for lat in (-55, -25, 0, 25, 55):
            yy = cy + math.sin(math.radians(lat)) * r * 0.79
            half = math.cos(math.radians(lat)) * r * 0.82
            d.arc((cx - half, yy - r * 0.10, cx + half, yy + r * 0.10), 0, 180, fill=(72, 170, 225, 70), width=max(1, int(SCALE)))
        rotation = (t * 11) % 180
        for offset in (-65, -32, 0, 32, 65):
            xx = cx + math.sin(math.radians(rotation + offset)) * r * 0.54
            squash = 0.22 + 0.65 * abs(math.cos(math.radians(rotation + offset)))
            d.ellipse((xx - r * 0.15 * squash, cy - r * 0.83, xx + r * 0.15 * squash, cy + r * 0.83), outline=(60, 165, 225, 45), width=1)
        canvas.alpha_composite(layer)

        if show_geo:
            earth_radius_km = float(CONFIG["earth_radius_km"])
            geo_center_radius = earth_radius_km + float(CONFIG["geo_altitude_km"])
            geo_r = r * geo_center_radius / earth_radius_km
            dd = ImageDraw.Draw(canvas)
            dd.ellipse((cx - geo_r, cy - geo_r * 0.26, cx + geo_r, cy + geo_r * 0.26), outline=(255, 195, 85, 120), width=max(1, int(2 * SCALE)))
            draw_text(canvas, "GEO ALTITUDE", ((cx + geo_r * 0.72) / SCALE, (cy - geo_r * 0.18) / SCALE), size=17, fill=(255, 205, 105, 210), bold=True, stroke=1)

    def draw_asteroid(self, canvas: Image.Image, center: Tuple[float, float], radius: float, t: float, alpha: int = 255, label: bool = False):
        cx, cy = center
        angle = t * 1.4
        ca, sa = math.cos(angle), math.sin(angle)
        points = []
        for px, py in self.asteroid_points:
            x = px * 1.12
            y = py * 0.78
            rx = x * ca - y * sa
            ry = x * sa + y * ca
            points.append((cx + rx * radius, cy + ry * radius))

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.polygon(points, fill=(255, 145, 70, int(alpha * 0.30)))
        glow = glow.filter(ImageFilter.GaussianBlur(max(3, int(radius * 0.16))))
        canvas.alpha_composite(glow)

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.polygon(points, fill=(111, 91, 79, alpha), outline=(255, 190, 125, alpha), width=max(1, int(2 * SCALE)))
        for crater_x, crater_y, crater_r in self.craters:
            x = crater_x * 1.05
            y = crater_y * 0.75
            rx = x * ca - y * sa
            ry = x * sa + y * ca
            rr = crater_r * radius
            ccx = cx + rx * radius
            ccy = cy + ry * radius
            d.ellipse((ccx - rr, ccy - rr * 0.75, ccx + rr, ccy + rr * 0.75), fill=(58, 48, 45, int(alpha * 0.55)), outline=(160, 130, 108, int(alpha * 0.35)), width=1)
        # Lit limb.
        d.arc((cx - radius * 1.12, cy - radius * 0.82, cx + radius * 1.12, cy + radius * 0.82), 205, 35, fill=(255, 218, 165, alpha), width=max(1, int(3 * SCALE)))
        canvas.alpha_composite(layer)
        if label:
            draw_text(canvas, "99942 APOPHIS", (cx / SCALE, (cy + radius + 26 * SCALE) / SCALE), size=20, fill=(255, 215, 155, alpha), bold=True, anchor="ma", stroke=1)

    def draw_title_hud(self, canvas: Image.Image, t: float, shot: Dict):
        if t < SHOT_PLAN[0]["end"]:
            alpha = int(255 * smoothstep((t - 0.2) / max(0.7 * DURATION / 58.0, 0.1)) * (1 - smoothstep((t - SHOT_PLAN[0]["end"] + 1.0 * DURATION / 58.0) / max(0.8 * DURATION / 58.0, 0.1))))
            if alpha > 3:
                draw_text(canvas, "NASA TRACKED THIS", (54, 88), size=48, fill=(245, 250, 255, alpha), bold=True)
                draw_text(canvas, "ASTEROID FOR DECADES", (54, 148), size=48, fill=(245, 250, 255, alpha), bold=True)
                draw_text(canvas, CONFIG["subtitle_text"], (58, 218), size=24, fill=(105, 225, 245, min(alpha, 230)), bold=True)
        else:
            labels = {
                "uncertainty": "ORBIT UNCERTAINTY // 2004",
                "radar": "RANGE + DOPPLER // 2005–2021",
                "flyby": "JPL HORIZONS // 2029-04-13",
                "gravity": "EARTH GRAVITY ASSIST // SAFE PASS",
                "safe": "PLANETARY DEFENSE // TRACK, REFINE, VERIFY",
            }
            draw_text(canvas, labels.get(str(shot["name"]), "APOPHIS TRACKING"), (52, 70), size=21, fill=(130, 220, 240, 215), bold=True, stroke=1)

    def draw_year_timeline(self, canvas: Image.Image, progress: float, active_index: int):
        x = 80 * SCALE
        y0 = 330 * SCALE
        y1 = 1450 * SCALE
        d = ImageDraw.Draw(canvas)
        d.line((x, y0, x, y1), fill=(85, 190, 225, 120), width=max(1, int(2 * SCALE)))
        count = len(MILESTONES)
        for index, milestone in enumerate(MILESTONES):
            yy = y0 + (y1 - y0) * index / max(count - 1, 1)
            passed = index <= active_index
            color = (255, 180, 80, 245) if index == active_index else ((95, 225, 245, 210) if passed else (105, 135, 160, 90))
            rr = (8 if index == active_index else 4) * SCALE
            d.ellipse((x - rr, yy - rr, x + rr, yy + rr), fill=color)
            if index in {0, 2, 4, 5, 7, 8}:
                draw_text(canvas, milestone["year_label"], ((x + 22 * SCALE) / SCALE, (yy - 10 * SCALE) / SCALE), size=17, fill=color, bold=True, stroke=1)
        cursor_y = y0 + (y1 - y0) * clamp(progress)
        d.line((x - 18 * SCALE, cursor_y, x + 18 * SCALE, cursor_y), fill=(255, 205, 95, 230), width=max(1, int(2 * SCALE)))

    def draw_discovery(self, canvas: Image.Image, t: float, u: float):
        # Telescope field and faint moving detections.
        cx, cy = 570 * SCALE, 780 * SCALE
        radius = 330 * SCALE
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(2, 8, 20, 185), outline=(65, 195, 225, 145), width=max(1, int(3 * SCALE)))
        d.line((cx - radius, cy, cx + radius, cy), fill=(70, 160, 195, 45), width=1)
        d.line((cx, cy - radius, cx, cy + radius), fill=(70, 160, 195, 45), width=1)
        for i in range(70):
            ang = deterministic_unit(str(i), "scope-a") * 2 * math.pi
            rad = math.sqrt(deterministic_unit(str(i), "scope-r")) * radius * 0.92
            sx = cx + math.cos(ang) * rad
            sy = cy + math.sin(ang) * rad
            rr = (0.7 + 1.8 * deterministic_unit(str(i), "scope-s")) * SCALE
            d.ellipse((sx - rr, sy - rr, sx + rr, sy + rr), fill=(225, 235, 255, int(60 + 145 * deterministic_unit(str(i), "scope-alpha"))))
        canvas.alpha_composite(layer)

        trail_u = smoothstep((u - 0.18) / 0.58)
        start = np.array([385, 925], float) * SCALE
        end = np.array([735, 650], float) * SCALE
        current = start + (end - start) * trail_u
        d = ImageDraw.Draw(canvas)
        for offset in (0.0, -0.11, -0.22):
            p = start + (end - start) * clamp(trail_u + offset)
            rr = 5 * SCALE
            d.ellipse((p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr), fill=(255, 185, 90, int(245 * (1 + offset * 2))))
        d.line((start[0], start[1], current[0], current[1]), fill=(255, 165, 75, 110), width=max(1, int(2 * SCALE)))
        draw_text(canvas, "2004 MN4", (620, 500), size=23, fill=(255, 195, 100, 245), bold=True)
        draw_text(canvas, "TWO NIGHTS OF POSITIONS", (620, 540), size=18, fill=(170, 215, 235, 210), bold=True)

        draw_round_panel(canvas, (165, 1230, 1015, 1538), fill=(2, 7, 18, 175), outline=(80, 195, 225, 85))
        draw_text(canvas, "DISCOVERED // 19 JUNE 2004", (200, 1270), size=24, fill=(105, 230, 245, 235), bold=True)
        draw_text(canvas, "MEAN DIAMETER", (200, 1330), size=18, fill=(170, 205, 225, 210), bold=True)
        draw_text(canvas, "≈ 340 m", (200, 1368), size=34, fill=(245, 250, 255, 245), bold=True)
        draw_text(canvas, "EARLY ORBIT", (565, 1330), size=18, fill=(170, 205, 225, 210), bold=True)
        draw_text(canvas, "POORLY CONSTRAINED", (565, 1372), size=26, fill=(255, 190, 95, 245), bold=True)
        draw_wrapped_text(canvas, "A short observation arc can fit many possible future paths.", (200, 1430), max_width=760, size=19, fill=(205, 222, 238, 220))

    def draw_uncertainty(self, canvas: Image.Image, t: float, u: float):
        earth_center = (650 * SCALE, 850 * SCALE)
        self.draw_earth(canvas, earth_center, 115 * SCALE, t)
        d = ImageDraw.Draw(canvas)
        shrink = 1.0 - 0.86 * smoothstep(u)
        corridor_half = lerp(330, 48, smoothstep(u)) * SCALE
        start = np.array([100, 1430], float) * SCALE
        end = np.array([1020, 380], float) * SCALE
        direction = end - start
        normal = np.array([-direction[1], direction[0]])
        normal /= max(np.linalg.norm(normal), 1e-9)
        poly = [tuple(start + normal * corridor_half), tuple(end + normal * corridor_half), tuple(end - normal * corridor_half), tuple(start - normal * corridor_half)]
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.polygon(poly, fill=(255, 115, 65, int(52 + 50 * shrink)), outline=(255, 160, 80, 100))
        for i in range(17):
            offset = lerp(-corridor_half, corridor_half, i / 16)
            p0 = start + normal * offset
            p1 = end + normal * offset
            ld.line((p0[0], p0[1], p1[0], p1[1]), fill=(255, 160, 85, int(16 + 42 * shrink)), width=max(1, int(SCALE)))
        layer = layer.filter(ImageFilter.GaussianBlur(max(1, int(2 * SCALE))))
        canvas.alpha_composite(layer)
        d.line((start[0], start[1], end[0], end[1]), fill=(255, 215, 120, 220), width=max(1, int(3 * SCALE)))

        # Observation marks accumulate and squeeze the corridor.
        for i in range(8):
            reveal = smoothstep((u - i * 0.09) / 0.12)
            if reveal <= 0:
                continue
            q = 0.12 + 0.10 * i
            p = start + direction * q
            rr = 8 * SCALE
            d.ellipse((p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr), fill=(95, 235, 255, int(235 * reveal)))

        draw_round_panel(canvas, (145, 250, 545, 520), fill=(2, 7, 18, 185), outline=(255, 145, 75, 100))
        draw_text(canvas, "LATE 2004", (178, 282), size=19, fill=(255, 185, 90, 235), bold=True)
        draw_text(canvas, "2.7%", (178, 330), size=58, fill=(255, 218, 165, 250), bold=True)
        draw_text(canvas, "BRIEF CALCULATED", (178, 405), size=18, fill=(210, 222, 236, 215), bold=True)
        draw_text(canvas, "2029 IMPACT PROBABILITY", (178, 438), size=18, fill=(210, 222, 236, 215), bold=True)
        draw_wrapped_text(canvas, "Earth overlapped the early uncertainty region. More measurements tested which paths were actually possible.", (178, 478), max_width=335, size=16, fill=(185, 207, 226, 210))

        draw_round_panel(canvas, (560, 1190, 1010, 1485), fill=(2, 7, 18, 180), outline=(75, 205, 230, 90))
        draw_text(canvas, "UNCERTAINTY WIDTH", (590, 1225), size=18, fill=(160, 215, 235, 220), bold=True)
        draw_text(canvas, f"{lerp(100, 8, smoothstep(u)):.0f}%", (590, 1270), size=46, fill=(105, 235, 250, 245), bold=True)
        bar_x0, bar_y, bar_w = 590 * SCALE, 1365 * SCALE, 360 * SCALE
        d.rectangle((bar_x0, bar_y, bar_x0 + bar_w, bar_y + 20 * SCALE), fill=(30, 65, 85, 180))
        d.rectangle((bar_x0, bar_y, bar_x0 + bar_w * shrink, bar_y + 20 * SCALE), fill=(75, 225, 245, 225))
        draw_text(canvas, "MORE DATA → SMALLER REGION", (590, 1415), size=17, fill=(190, 215, 232, 215), bold=True)

    def draw_radar_dish(self, canvas: Image.Image, center: Tuple[float, float], size: float, t: float):
        cx, cy = center
        d = ImageDraw.Draw(canvas)
        d.arc((cx - size, cy - size * 0.6, cx + size, cy + size * 0.6), 195, 345, fill=(175, 215, 235, 235), width=max(1, int(5 * SCALE)))
        d.line((cx, cy + size * 0.18, cx, cy + size * 0.72), fill=(150, 195, 220, 220), width=max(1, int(4 * SCALE)))
        d.line((cx - size * 0.42, cy + size * 0.72, cx + size * 0.42, cy + size * 0.72), fill=(150, 195, 220, 220), width=max(1, int(5 * SCALE)))
        d.line((cx, cy + size * 0.03, cx + size * 0.17, cy - size * 0.25), fill=(255, 200, 100, 235), width=max(1, int(3 * SCALE)))
        rr = 7 * SCALE
        d.ellipse((cx + size * 0.17 - rr, cy - size * 0.25 - rr, cx + size * 0.17 + rr, cy - size * 0.25 + rr), fill=(255, 220, 135, 245))

    def draw_radar(self, canvas: Image.Image, t: float, u: float):
        earth_center = (480 * SCALE, 1120 * SCALE)
        earth_r = 160 * SCALE
        self.draw_earth(canvas, earth_center, earth_r, t)
        dish_center = (510 * SCALE, 1320 * SCALE)
        self.draw_radar_dish(canvas, dish_center, 120 * SCALE, t)
        asteroid_center = (820 * SCALE, 520 * SCALE)
        self.draw_asteroid(canvas, asteroid_center, 90 * SCALE, t, label=True)

        d = ImageDraw.Draw(canvas)
        start = np.array([dish_center[0] + 18 * SCALE, dish_center[1] - 55 * SCALE])
        end = np.array(asteroid_center)
        pulse_phase = (u * 5.0) % 1.0
        d.line((start[0], start[1], end[0], end[1]), fill=(70, 225, 250, 90), width=max(1, int(2 * SCALE)))
        for i in range(5):
            q = (pulse_phase - i * 0.13) % 1.0
            p = start + (end - start) * q
            rr = (5 + 10 * (1 - q)) * SCALE
            d.ellipse((p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr), outline=(100, 245, 255, int(230 * (1 - q * 0.55))), width=max(1, int(2 * SCALE)))

        active = min(len(MILESTONES) - 1, 3 + int(u * 4))
        self.draw_year_timeline(canvas, 0.28 + 0.52 * u, active)
        milestone = MILESTONES[active]
        draw_round_panel(canvas, (175, 255, 650, 495), fill=(2, 7, 18, 185), outline=(75, 205, 230, 95))
        draw_text(canvas, milestone["year_label"], (205, 282), size=21, fill=(105, 230, 245, 235), bold=True)
        draw_text(canvas, milestone["title"], (205, 324), size=27, fill=(245, 250, 255, 245), bold=True)
        draw_text(canvas, milestone["detail"], (205, 374), size=19, fill=(210, 225, 238, 220), bold=True)
        draw_wrapped_text(canvas, milestone["status"], (205, 414), max_width=390, size=18, fill=(255, 195, 100, 225), bold=True)

        draw_round_panel(canvas, (590, 1070, 1015, 1440), fill=(2, 7, 18, 180), outline=(255, 175, 85, 90))
        draw_text(canvas, "2021 GOLDSTONE", (620, 1105), size=20, fill=(255, 190, 95, 235), bold=True)
        draw_text(canvas, "RANGE ACCURACY", (620, 1150), size=18, fill=(180, 210, 228, 215), bold=True)
        draw_text(canvas, "≈ 150 m", (620, 1195), size=42, fill=(245, 250, 255, 245), bold=True)
        draw_text(canvas, "ASTEROID DISTANCE", (620, 1270), size=18, fill=(180, 210, 228, 215), bold=True)
        draw_text(canvas, "17 million km", (620, 1312), size=29, fill=(105, 230, 245, 240), bold=True)
        draw_wrapped_text(canvas, "Radar measures echo delay and Doppler shift — powerful constraints on an orbit.", (620, 1360), max_width=350, size=16, fill=(195, 215, 232, 210))

    def interpolate_vector(self, fraction: float) -> Tuple[np.ndarray, np.ndarray, float, pd.Timestamp]:
        fraction = clamp(fraction)
        target = self.times[0] + (self.times[-1] - self.times[0]) * fraction
        x = np.array([np.interp(target, self.times, self.positions[:, i]) for i in range(3)])
        v = np.array([np.interp(target, self.times, self.velocities[:, i]) for i in range(3)])
        rg = float(np.interp(target, self.times, self.ranges))
        return x, v, rg, pd.to_datetime(target, unit="s", utc=True)

    def project_flyby(self, position: np.ndarray, earth_center: Tuple[float, float], earth_radius_px: float) -> Tuple[float, float, float]:
        earth_radius_km = float(CONFIG["earth_radius_km"])
        px_per_km = earth_radius_px / earth_radius_km
        along = float(np.dot(position, self.flyby_forward))
        side = float(np.dot(position, self.flyby_side))
        depth = float(np.dot(position, self.flyby_depth))
        x = earth_center[0] + along * px_per_km
        y = earth_center[1] - side * px_per_km
        return x, y, depth

    def draw_flyby(self, canvas: Image.Image, t: float, u: float):
        earth_center = (540 * SCALE, 1110 * SCALE)
        earth_r = 72 * SCALE
        self.draw_earth(canvas, earth_center, earth_r, t, show_geo=True)
        d = ImageDraw.Draw(canvas)

        # Project the real Horizons path.
        path_points = []
        for position in self.positions:
            x, y, _ = self.project_flyby(position, earth_center, earth_r)
            path_points.append((x, y))
        if len(path_points) >= 2:
            d.line(path_points, fill=(255, 175, 75, 160), width=max(1, int(3 * SCALE)))

        position, velocity, rg, timestamp = self.interpolate_vector(u)
        ax, ay, depth = self.project_flyby(position, earth_center, earth_r)
        self.draw_asteroid(canvas, (ax, ay), 32 * SCALE, t, label=False)
        reticle = 55 * SCALE
        d.arc((ax - reticle, ay - reticle, ax + reticle, ay + reticle), 10, 120, fill=(255, 205, 95, 235), width=max(1, int(2 * SCALE)))
        d.arc((ax - reticle, ay - reticle, ax + reticle, ay + reticle), 190, 300, fill=(255, 205, 95, 235), width=max(1, int(2 * SCALE)))

        # Closest-approach line and altitude bracket.
        closest = self.positions[self.closest_index]
        cx, cy, _ = self.project_flyby(closest, earth_center, earth_r)
        d.line((earth_center[0], earth_center[1], cx, cy), fill=(105, 225, 245, 105), width=max(1, int(2 * SCALE)))
        draw_text(canvas, "CLOSEST SAMPLE", (cx / SCALE, (cy - 72 * SCALE) / SCALE), size=16, fill=(105, 230, 245, 220), bold=True, anchor="ma", stroke=1)

        # Moon scale arrow (offscreen distance declared).
        d.line((165 * SCALE, 1460 * SCALE, 915 * SCALE, 1460 * SCALE), fill=(110, 185, 215, 130), width=max(1, int(2 * SCALE)))
        d.line((165 * SCALE, 1449 * SCALE, 165 * SCALE, 1471 * SCALE), fill=(110, 185, 215, 170), width=max(1, int(2 * SCALE)))
        d.line((915 * SCALE, 1449 * SCALE, 915 * SCALE, 1471 * SCALE), fill=(110, 185, 215, 170), width=max(1, int(2 * SCALE)))
        draw_text(canvas, "EARTH–MOON DISTANCE = 384,400 km", (540, 1490), size=17, fill=(165, 205, 225, 205), bold=True, anchor="ma", stroke=1)
        draw_text(canvas, "APOPHIS PASS ≈ 0.10 LUNAR DISTANCE", (540, 1523), size=18, fill=(255, 195, 100, 230), bold=True, anchor="ma", stroke=1)

        draw_round_panel(canvas, (45, 245, 515, 585), fill=(2, 7, 18, 188), outline=(75, 205, 230, 95))
        draw_text(canvas, "JPL HORIZONS VECTOR", (78, 278), size=19, fill=(105, 230, 245, 235), bold=True)
        draw_text(canvas, timestamp.strftime("%Y-%m-%d"), (78, 325), size=28, fill=(245, 250, 255, 245), bold=True)
        draw_text(canvas, timestamp.strftime("%H:%M:%S UTC"), (78, 367), size=22, fill=(205, 225, 238, 225), bold=True)
        draw_text(canvas, "GEOCENTRIC RANGE", (78, 425), size=17, fill=(170, 205, 225, 210), bold=True)
        draw_text(canvas, f"{rg:,.0f} km", (78, 462), size=34, fill=(255, 205, 110, 245), bold=True)
        speed = float(np.linalg.norm(velocity))
        draw_text(canvas, f"{speed:.2f} km/s", (78, 520), size=24, fill=(105, 230, 245, 235), bold=True)

        draw_round_panel(canvas, (615, 245, 1030, 585), fill=(2, 7, 18, 188), outline=(255, 175, 85, 95))
        draw_text(canvas, "CLOSEST APPROACH", (648, 278), size=19, fill=(255, 190, 95, 235), bold=True)
        draw_text(canvas, "≈ 32,000 km", (648, 330), size=34, fill=(245, 250, 255, 245), bold=True)
        draw_text(canvas, "ABOVE EARTH'S SURFACE", (648, 378), size=17, fill=(205, 220, 235, 220), bold=True)
        draw_text(canvas, "CLOSER THAN", (648, 436), size=17, fill=(170, 205, 225, 210), bold=True)
        draw_text(canvas, "GEO ALTITUDE", (648, 474), size=27, fill=(255, 205, 105, 240), bold=True)
        draw_wrapped_text(canvas, "Safe trajectory. It does not cross the populated equatorial geosynchronous ring.", (648, 522), max_width=330, size=15, fill=(190, 215, 232, 210))

    def draw_gravity(self, canvas: Image.Image, t: float, u: float):
        earth_center = (540 * SCALE, 870 * SCALE)
        self.draw_earth(canvas, earth_center, 155 * SCALE, t)
        d = ImageDraw.Draw(canvas)

        # Stylized pre/post paths based on NASA's stated orbital-period change.
        x0 = 70 * SCALE
        x1 = 1010 * SCALE
        points_before = []
        points_after = []
        for i in range(120):
            q = i / 119
            x = lerp(x0, x1, q)
            y_base = 1180 * SCALE - 610 * SCALE * q
            bend = 120 * SCALE * math.exp(-((q - 0.52) / 0.17) ** 2)
            points_before.append((x, y_base))
            points_after.append((x, y_base - bend * (0.25 + 0.75 * smoothstep(u))))
        d.line(points_before, fill=(100, 175, 205, 85), width=max(1, int(2 * SCALE)))
        d.line(points_after, fill=(255, 175, 75, 225), width=max(1, int(4 * SCALE)))

        q = smoothstep(u)
        point_index = min(len(points_after) - 1, int(q * (len(points_after) - 1)))
        self.draw_asteroid(canvas, points_after[point_index], 62 * SCALE, t * 1.8, label=True)

        # Gravity vectors near Earth.
        for angle_deg in range(0, 360, 30):
            angle = math.radians(angle_deg)
            start_r = 300 * SCALE
            end_r = 190 * SCALE
            sx = earth_center[0] + math.cos(angle) * start_r
            sy = earth_center[1] + math.sin(angle) * start_r
            ex = earth_center[0] + math.cos(angle) * end_r
            ey = earth_center[1] + math.sin(angle) * end_r
            d.line((sx, sy, ex, ey), fill=(85, 220, 245, 55), width=max(1, int(SCALE)))

        draw_round_panel(canvas, (65, 245, 500, 565), fill=(2, 7, 18, 185), outline=(75, 205, 230, 95))
        draw_text(canvas, "BEFORE 2029", (98, 280), size=19, fill=(105, 230, 245, 230), bold=True)
        draw_text(canvas, "ATEN ORBIT", (98, 330), size=28, fill=(245, 250, 255, 245), bold=True)
        draw_text(canvas, "≈ 0.9 YEAR PERIOD", (98, 380), size=20, fill=(200, 220, 235, 220), bold=True)
        draw_text(canvas, "AFTER 2029", (98, 445), size=19, fill=(255, 190, 95, 230), bold=True)
        draw_text(canvas, "APOLLO ORBIT", (98, 485), size=28, fill=(255, 218, 165, 245), bold=True)
        draw_text(canvas, "≈ 1.2 YEAR PERIOD", (98, 530), size=20, fill=(200, 220, 235, 220), bold=True)

        draw_round_panel(canvas, (610, 1180, 1020, 1500), fill=(2, 7, 18, 180), outline=(255, 175, 85, 90))
        draw_text(canvas, "EARTH TIDES MAY", (642, 1215), size=19, fill=(255, 190, 95, 235), bold=True)
        draw_text(canvas, "CHANGE ITS SPIN", (642, 1260), size=31, fill=(245, 250, 255, 245), bold=True)
        draw_text(canvas, "TUMBLE ≈ 31 h", (642, 1322), size=21, fill=(105, 230, 245, 235), bold=True)
        draw_text(canvas, "ROCKING ≈ 264 h", (642, 1362), size=21, fill=(105, 230, 245, 235), bold=True)
        draw_wrapped_text(canvas, "Scientists will look for spin changes, quakes or small landslides after the encounter.", (642, 1410), max_width=330, size=15, fill=(195, 215, 232, 210))

    def draw_spacecraft(self, canvas: Image.Image, center: Tuple[float, float], size: float, t: float):
        cx, cy = center
        d = ImageDraw.Draw(canvas)
        body_w = size * 0.62
        body_h = size * 0.40
        d.rounded_rectangle((cx - body_w / 2, cy - body_h / 2, cx + body_w / 2, cy + body_h / 2), radius=max(2, int(8 * SCALE)), fill=(155, 165, 178, 245), outline=(230, 240, 250, 240), width=max(1, int(2 * SCALE)))
        panel_w = size * 0.78
        panel_h = size * 0.24
        d.rectangle((cx - body_w / 2 - panel_w, cy - panel_h / 2, cx - body_w / 2, cy + panel_h / 2), fill=(25, 75, 145, 240), outline=(110, 190, 245, 225), width=max(1, int(2 * SCALE)))
        d.rectangle((cx + body_w / 2, cy - panel_h / 2, cx + body_w / 2 + panel_w, cy + panel_h / 2), fill=(25, 75, 145, 240), outline=(110, 190, 245, 225), width=max(1, int(2 * SCALE)))
        dish_r = size * 0.23
        d.arc((cx - dish_r, cy - body_h * 0.9 - dish_r * 2, cx + dish_r, cy - body_h * 0.9), 180, 360, fill=(245, 245, 235, 235), width=max(1, int(4 * SCALE)))
        d.line((cx, cy - body_h / 2, cx, cy - body_h * 0.9), fill=(220, 230, 238, 230), width=max(1, int(3 * SCALE)))
        flame = 12 * SCALE * (0.5 + 0.5 * math.sin(t * 8))
        d.polygon([(cx - body_w * 0.18, cy + body_h / 2), (cx, cy + body_h / 2 + flame), (cx + body_w * 0.18, cy + body_h / 2)], fill=(255, 175, 75, 170))

    def draw_safe(self, canvas: Image.Image, t: float, u: float):
        earth_center = (330 * SCALE, 1090 * SCALE)
        self.draw_earth(canvas, earth_center, 170 * SCALE, t)
        asteroid_center = (790 * SCALE, 830 * SCALE)
        self.draw_asteroid(canvas, asteroid_center, 115 * SCALE, t, label=True)
        craft_center = (790 * SCALE + math.sin(t * 0.8) * 140 * SCALE, 830 * SCALE - 205 * SCALE + math.cos(t * 0.8) * 55 * SCALE)
        self.draw_spacecraft(canvas, craft_center, 105 * SCALE, t)

        alpha = int(255 * smoothstep(u / 0.25))
        draw_round_panel(canvas, (80, 245, 1000, 600), fill=(2, 8, 19, 194), outline=(70, 230, 205, 125))
        draw_text(canvas, "NO IMPACT RISK", (540, 290), size=50, fill=(160, 255, 225, alpha), bold=True, anchor="ma")
        draw_text(canvas, "FOR AT LEAST 100 YEARS", (540, 365), size=34, fill=(245, 250, 255, alpha), bold=True, anchor="ma")
        draw_text(canvas, "DECADES OF OPTICAL + RADAR TRACKING", (540, 430), size=20, fill=(105, 230, 245, min(alpha, 235)), bold=True, anchor="ma", stroke=1)
        draw_wrapped_text(canvas, "The scary number disappeared because the orbit became better known — not because the asteroid was deflected.", (155, 482), max_width=770, size=20, fill=(205, 225, 238, min(alpha, 225)), bold=False)

        draw_round_panel(canvas, (535, 1205, 1015, 1515), fill=(2, 7, 18, 180), outline=(255, 175, 85, 90))
        draw_text(canvas, "OSIRIS-APEX", (568, 1240), size=27, fill=(255, 195, 100, 240), bold=True)
        draw_text(canvas, "RENDEZVOUS // JUNE 2029", (568, 1292), size=18, fill=(210, 225, 238, 220), bold=True)
        draw_wrapped_text(canvas, "NASA's spacecraft will study how the close encounter changed Apophis — including its orbit, spin and surface.", (568, 1340), max_width=390, size=17, fill=(195, 215, 232, 215))

        self.draw_year_timeline(canvas, 1.0, len(MILESTONES) - 1)

    def draw_caption_and_credits(self, canvas: Image.Image, t: float):
        caption = caption_at(t)
        if caption:
            y0 = 1645
            draw_round_panel(canvas, (45, y0, 1035, y0 + 145), fill=(1, 4, 12, 176), outline=(60, 180, 220, 70), radius=24, width=1)
            draw_wrapped_text(canvas, caption, (72, y0 + 28), max_width=930, size=28, fill=(245, 250, 255, 245))

        credit_alpha = int(225 * smoothstep((t - DURATION * 0.84) / max(DURATION * 0.09, 0.1)))
        if credit_alpha > 4:
            draw_text(canvas, str(CONFIG["credit_text"]), (55, 1840), size=17, fill=(215, 230, 242, credit_alpha), bold=True, stroke=1)
            draw_wrapped_text(canvas, str(CONFIG["scientific_note"]), (55, 1872), max_width=970, size=14, fill=(180, 205, 225, credit_alpha))

    def draw_hud_noise(self, canvas: Image.Image, t: float):
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for x, y, length, alpha, phase in self.hud_noise:
            pulse = 0.5 + 0.5 * math.sin(t * 1.8 + phase)
            if pulse < 0.75:
                continue
            yy = (y + t * 9 * SCALE) % HEIGHT
            d.line((x, yy, x + length, yy), fill=(90, 210, 240, int(alpha * pulse)), width=1)
        offset = int((t * 37) % max(2, int(7 * SCALE)))
        step = max(3, int(7 * SCALE))
        for yy in range(offset, HEIGHT, step):
            d.line((0, yy, WIDTH, yy), fill=(110, 195, 235, 10), width=1)
        scan_y = int((t * 145 * SCALE) % (HEIGHT + 160 * SCALE)) - int(80 * SCALE)
        d.rectangle((0, scan_y, WIDTH, scan_y + 38 * SCALE), fill=(80, 210, 240, 8))
        canvas.alpha_composite(layer)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        u = local_shot_fraction(t, shot)
        canvas = self.render_background(t)
        name = str(shot["name"])
        if name == "discovery":
            self.draw_discovery(canvas, t, u)
        elif name == "uncertainty":
            self.draw_uncertainty(canvas, t, u)
        elif name == "radar":
            self.draw_radar(canvas, t, u)
        elif name == "flyby":
            self.draw_flyby(canvas, t, u)
        elif name == "gravity":
            self.draw_gravity(canvas, t, u)
        else:
            self.draw_safe(canvas, t, u)

        self.draw_title_hud(canvas, t, shot)
        self.draw_caption_and_credits(canvas, t)
        self.draw_hud_noise(canvas, t)

        rgb = np.array(canvas.convert("RGB"))
        image = Image.fromarray(rgb)
        image = ImageEnhance.Contrast(image).enhance(1.13)
        image = ImageEnhance.Color(image).enhance(1.08)
        arr = np.array(image).astype(np.float32)
        arr *= VIGNETTE[..., None]

        fade_in = smoothstep(t / max(0.75 * DURATION / 58.0, 0.1))
        fade_out = 1.0 - smoothstep((t - (DURATION - max(0.95 * DURATION / 58.0, 0.15))) / max(0.85 * DURATION / 58.0, 0.1))
        arr *= fade_in * fade_out
        return np.clip(arr, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Subtitle, preview and render pipeline
# -----------------------------------------------------------------------------


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
    for index, (start, end, text) in enumerate(CAPTIONS, start=1):
        lines.extend([str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def create_contact_sheet(preview_paths: List[Path], output_path: Path):
    images = [Image.open(path).convert("RGB") for path in preview_paths]
    thumb_w = 360 if not QUICK_MODE else 270
    thumb_h = int(thumb_w * HEIGHT / WIDTH)
    margin = 22
    cols = 3 if len(images) >= 3 else len(images)
    rows = int(math.ceil(len(images) / cols))
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * margin, rows * thumb_h + (rows + 1) * margin), (4, 7, 14))
    for index, image in enumerate(images):
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        col = index % cols
        row = index // cols
        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + margin)
        sheet.paste(thumb, (x, y))
    sheet.save(output_path, quality=92)
    for image in images:
        image.close()


def run_ffmpeg(command: List[str]):
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def render_video(scene: ApophisScene) -> Path:
    raw_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    subbed_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_subbed.mp4"
    audio_path_out = OUTPUT_ROOT / f"{CONFIG['output_basename']}_with_audio.mp4"
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"

    if bool(CONFIG.get("write_subtitle_sidecar", True)):
        write_srt(srt_path)

    frame_count = int(round(DURATION * int(CONFIG["fps"])))
    times = np.arange(frame_count, dtype=float) / int(CONFIG["fps"])
    print(f"Rendering {frame_count:,} frames at {WIDTH}x{HEIGHT}...")
    with iio.get_writer(raw_path, fps=int(CONFIG["fps"]), codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None) as writer:
        for t in tqdm(times, desc="Rendering Apophis decades short"):
            writer.append_data(scene.render_frame(float(t)))

    final_candidate = raw_path
    ffmpeg = find_ffmpeg()
    if bool(CONFIG.get("burn_subtitles")) and ffmpeg and srt_path.exists():
        run_ffmpeg([
            ffmpeg, "-y", "-i", str(final_candidate),
            "-vf", f"subtitles={srt_path}:force_style=Fontname=DejaVu Sans,Fontsize=22,Outline=1.2,BorderStyle=3,MarginV=90",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", str(subbed_path),
        ])
        final_candidate = subbed_path

    audio_path = CONFIG.get("audio_path")
    if audio_path and Path(str(audio_path)).exists() and ffmpeg:
        run_ffmpeg([
            ffmpeg, "-y", "-i", str(final_candidate), "-i", str(audio_path),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(audio_path_out),
        ])
        final_candidate = audio_path_out

    shutil.copyfile(final_candidate, final_path)
    return final_path


def main():
    print("Starting Apophis decades data-short pipeline...")
    ephemeris = fetch_horizons_vectors(force_refresh=FORCE_REFRESH)
    summary = write_story_data(ephemeris)

    print("Ephemeris rows:", len(ephemeris))
    print("Source:", ephemeris.iloc[0].get("source", "unknown"))
    print("Closest sampled geocentric distance:", f"{summary['closest_sample_geocentric_distance_km']:,.1f} km")
    print("Closest sampled altitude:", f"{summary['closest_sample_altitude_above_reference_earth_radius_km']:,.1f} km")
    print("Closest sampled geocentric speed:", f"{summary['closest_sample_geocentric_speed_km_s']:.3f} km/s")

    scene = ApophisScene(ephemeris, summary)
    preview_times_full = [1.2, 10.8, 23.0, 36.0, 47.0, 56.5]
    preview_times = [value * DURATION / 58.0 for value in preview_times_full]
    preview_paths = []
    for index, preview_time in enumerate(tqdm(preview_times, desc="Preview frames"), start=1):
        path = PREVIEW_DIR / f"preview_{index:02d}_{preview_time:05.2f}s.png"
        Image.fromarray(scene.render_frame(preview_time)).save(path)
        preview_paths.append(path)

    contact_sheet = PREVIEW_DIR / "nasa_tracked_apophis_for_decades_contact_sheet.jpg"
    create_contact_sheet(preview_paths, contact_sheet)
    print("Contact sheet:", contact_sheet.resolve())

    final_video = render_video(scene)
    print("Final video:", final_video.resolve())
    print("Output directory:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
