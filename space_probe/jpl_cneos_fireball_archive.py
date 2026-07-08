# %% [markdown]
# # Earth Fireball Archive — YouTube Short from Real NASA/JPL CNEOS Data
#
# This script creates a vertical 1080×1920 cinematic astronomy short from the
# NASA/JPL CNEOS Fireball Data API.
#
# Theme: EARTH FIREBALL ARCHIVE — bright atmospheric fireball / bolide events
#
# Real catalog values used:
# - peak-brightness date and time,
# - geographic latitude / longitude when reported,
# - peak-brightness altitude when reported,
# - approximate total radiated energy,
# - calculated total impact energy,
# - Earth-fixed entry-velocity components when reported.
#
# Visual language:
# - rotating Earth with atmospheric halo and instrument grid,
# - cataloged event locations projected onto the globe,
# - chronological replay in peak-brightness time order,
# - impact energy controls flare radius and bloom,
# - altitude controls atmospheric height offset when available,
# - entry-velocity components control streak direction when available,
# - deterministic editorial streak direction only where velocity is missing,
# - energy leaderboard, altitude strip, event timeline, scanlines and HUD noise.
#
# Scientific fidelity note:
# The data layer is the public CNEOS fireball dataset derived from U.S.
# Government sensor detections. CNEOS notes that the events are not a complete
# or real-time record and reported parameters may be revised. The globe location,
# time, altitude, energy and available velocity components are data-driven.
# Decorative Earth texture, camera motion and velocity-missing streak directions
# are editorial.
#
# Recommended install:
#
#     pip install numpy pandas matplotlib pillow imageio imageio-ffmpeg requests tqdm
#
# Quick test render:
#
#     CONFIG["fps"] = 12
#     CONFIG["duration_s"] = 12
#     CONFIG["video_width"] = 540
#     CONFIG["video_height"] = 960

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry


# %% [markdown]
# ## Configuration

OUTPUT_ROOT = Path("earth_fireball_archive_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"

for directory in [OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CONFIG = {
    # Final delivery
    "video_width": 1080,
    "video_height": 1920,
    "fps": 24,
    "duration_s": 58,
    "output_basename": "jpl_cneos_earth_fireball_archive_short",

    # Reproducible catalog window
    "catalog_start_date": "1988-01-01",
    "catalog_end_date": "2025-12-31",
    "require_location": True,

    # NASA/JPL Fireball Data API
    "fireball_api_url": "https://ssd-api.jpl.nasa.gov/fireball.api",
    "expected_api_version": "1.2",
    "request_timeout": (20, 180),
    "request_retries": 5,
    "retry_backoff_s": 1.5,

    # Visual limits
    "max_events_for_video": 1200,
    "max_simultaneous_events": 28,
    "highlight_count": 16,
    "leaderboard_count": 6,

    # Rendering
    "earth_radius_px": 360,
    "atmosphere_height_scale_px": 0.92,
    "contrast_boost": 1.14,
    "saturation_boost": 1.10,
    "vignette_strength": 0.28,
    "background_particle_count": 520,
    "hud_noise_count": 75,

    # Text
    "title_text": "EARTH FIREBALL ARCHIVE",
    "subtitle_text": "Bright atmospheric events // NASA/JPL CNEOS data",
    "credit_text": "Data: NASA/JPL CNEOS Fireball Data API",
    "scientific_note": (
        "CNEOS says this sensor-derived public record is not complete or real-time "
        "and reported parameters may be revised. Missing-velocity streak direction "
        "is editorial; reported locations, times, energies and altitudes are catalog values."
    ),

    # Optional finishing
    "audio_path": None,
    "burn_subtitles": False,
    "write_subtitle_sidecar": True,
}

OUT_SIZE = (CONFIG["video_width"], CONFIG["video_height"])


# %% [markdown]
# ## Network and data helpers


def build_retry_session(config: Dict) -> requests.Session:
    retries = Retry(
        total=int(config.get("request_retries", 5)),
        connect=int(config.get("request_retries", 5)),
        read=int(config.get("request_retries", 5)),
        status=int(config.get("request_retries", 5)),
        backoff_factor=float(config.get("retry_backoff_s", 1.5)),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retries, pool_connections=2, pool_maxsize=2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "earth-fireball-archive-short/1.0 (educational visualization)"
    })
    return session


def deterministic_unit(text: str, salt: str = "") -> float:
    payload = f"{text}|{salt}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64 - 1)


def fetch_fireballs(config: Dict, force_refresh: bool = False) -> pd.DataFrame:
    raw_path = DATA_ROOT / "jpl_cneos_fireball_raw.json"
    clean_path = DATA_ROOT / "jpl_cneos_fireball_clean.csv"

    if clean_path.exists() and raw_path.exists() and not force_refresh:
        print("Using cached CNEOS fireball data.")
        df = pd.read_csv(clean_path)
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
        return clean_fireballs(df, save_path=None)

    params = {
        "date-min": config["catalog_start_date"],
        "date-max": config["catalog_end_date"],
        "sort": "date",
        "vel-comp": "true",
    }
    if config.get("require_location", True):
        params["req-loc"] = "true"

    session = build_retry_session(config)
    print("Requesting NASA/JPL CNEOS Fireball Data API ...")
    response = session.get(
        config["fireball_api_url"],
        params=params,
        timeout=config["request_timeout"],
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"JPL Fireball API HTTP {response.status_code}: {response.reason}\n"
            f"Response preview:\n{response.text[:1600]}"
        )

    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(
            "JPL Fireball API did not return valid JSON.\n"
            f"Response preview:\n{response.text[:1600]}"
        ) from exc

    signature = payload.get("signature") or {}
    version = str(signature.get("version") or "")

    if version != str(config["expected_api_version"]):
        raise RuntimeError(
            "JPL Fireball API version changed.\n"
            f"Expected: {config['expected_api_version']}\n"
            f"Received: {version!r}\n"
            "Review the official API documentation before rendering so field parsing "
            "does not silently use an incompatible schema."
        )

    fields = payload.get("fields")
    data = payload.get("data")

    if not fields or data is None:
        if int(payload.get("count", 0) or 0) == 0:
            raise RuntimeError("The JPL Fireball API returned zero events for this query.")
        raise RuntimeError(f"Unexpected Fireball API payload:\n{json.dumps(payload)[:1600]}")

    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    frame = pd.DataFrame(data, columns=fields)
    return clean_fireballs(frame, save_path=clean_path)


def clean_fireballs(df: pd.DataFrame, save_path: Optional[Path]) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "date": "event_time_text",
        "lat": "latitude_deg",
        "lon": "longitude_deg",
        "lat-dir": "latitude_dir",
        "lon-dir": "longitude_dir",
        "alt": "altitude_km",
        "energy": "radiated_energy_1e10_j",
        "impact-e": "impact_energy_kt",
        "vx": "velocity_x_km_s",
        "vy": "velocity_y_km_s",
        "vz": "velocity_z_km_s",
    }
    for source, target in rename_map.items():
        if source in df.columns and target not in df.columns:
            df = df.rename(columns={source: target})

    if "event_time" not in df.columns:
        if "event_time_text" not in df.columns:
            raise RuntimeError("Fireball data is missing the peak-brightness date/time field.")
        df["event_time"] = pd.to_datetime(
            df["event_time_text"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
            utc=True,
        )
    else:
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")

    numeric_columns = [
        "latitude_deg",
        "longitude_deg",
        "altitude_km",
        "radiated_energy_1e10_j",
        "impact_energy_kt",
        "velocity_x_km_s",
        "velocity_y_km_s",
        "velocity_z_km_s",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan

    if "latitude_dir" not in df.columns:
        df["latitude_dir"] = ""
    if "longitude_dir" not in df.columns:
        df["longitude_dir"] = ""

    lat_dir = df["latitude_dir"].fillna("").astype(str).str.upper().str.strip()
    lon_dir = df["longitude_dir"].fillna("").astype(str).str.upper().str.strip()

    df["latitude_signed_deg"] = np.where(
        lat_dir.eq("S"), -df["latitude_deg"].abs(), df["latitude_deg"].abs()
    )
    df["longitude_signed_deg"] = np.where(
        lon_dir.eq("W"), -df["longitude_deg"].abs(), df["longitude_deg"].abs()
    )

    velocity_components = df[[
        "velocity_x_km_s",
        "velocity_y_km_s",
        "velocity_z_km_s",
    ]]
    df["velocity_reported"] = velocity_components.notna().all(axis=1)
    df["entry_velocity_km_s"] = np.sqrt(
        velocity_components.fillna(0.0).pow(2).sum(axis=1)
    )
    df.loc[~df["velocity_reported"], "entry_velocity_km_s"] = np.nan

    df["radiated_energy_j"] = df["radiated_energy_1e10_j"] * 1.0e10

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["event_time", "impact_energy_kt", "radiated_energy_1e10_j"])
    df = df[(df["impact_energy_kt"] > 0) & (df["radiated_energy_1e10_j"] > 0)].copy()

    if CONFIG.get("require_location", True):
        df = df.dropna(subset=["latitude_signed_deg", "longitude_signed_deg"]).copy()

    df = df[
        df["latitude_signed_deg"].between(-90.0, 90.0, inclusive="both")
        & df["longitude_signed_deg"].between(-180.0, 180.0, inclusive="both")
    ].copy()

    # Stable event ID and editorial phases.
    df["event_id"] = df.apply(
        lambda row: (
            f"{row['event_time'].isoformat()}|"
            f"{row['latitude_signed_deg']:.4f}|"
            f"{row['longitude_signed_deg']:.4f}"
        ),
        axis=1,
    )
    df["visual_phase"] = df["event_id"].apply(
        lambda text: deterministic_unit(text, "phase") * 2.0 * math.pi
    )
    df["fallback_angle"] = df["event_id"].apply(
        lambda text: deterministic_unit(text, "entry-angle") * 2.0 * math.pi
    )

    altitude_bonus = np.where(
        df["altitude_km"].notna(),
        np.clip((70.0 - df["altitude_km"].fillna(70.0)) / 70.0, 0.0, 1.0),
        0.0,
    )
    velocity_bonus = np.log1p(df["entry_velocity_km_s"].fillna(0.0))
    df["highlight_score"] = (
        7.0 * np.log1p(df["impact_energy_kt"] * 25.0)
        + 0.8 * velocity_bonus
        + 1.2 * altitude_bonus
    )

    df = (
        df.sort_values(["event_time", "impact_energy_kt"], ascending=[True, False])
        .drop_duplicates(["event_id"])
        .reset_index(drop=True)
    )

    if save_path is not None:
        save_df = df.copy()
        save_df["event_time"] = save_df["event_time"].astype(str)
        save_df.to_csv(save_path, index=False)
        print("Saved cleaned CNEOS fireball CSV:", save_path)

    print(f"Fireball events retained: {len(df):,}")
    print(
        "Impact-energy range:",
        f"{df['impact_energy_kt'].min():.4f}–{df['impact_energy_kt'].max():.1f} kt",
    )
    if df["altitude_km"].notna().any():
        print(
            "Peak-brightness altitude range:",
            f"{df['altitude_km'].min():.1f}–{df['altitude_km'].max():.1f} km",
        )
    print(f"Events with reported velocity components: {int(df['velocity_reported'].sum()):,}")
    return df


def select_video_events(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    max_events = int(config["max_events_for_video"])
    if len(df) <= max_events:
        return df.copy().reset_index(drop=True)

    highlight_n = min(260, max_events // 3)
    highlights = df.nlargest(highlight_n, "highlight_score")

    remaining_count = max_events - len(highlights)
    timeline_indices = np.linspace(0, len(df) - 1, max(remaining_count, 1), dtype=int)
    timeline_sample = df.iloc[timeline_indices]

    selected = pd.concat([highlights, timeline_sample], ignore_index=True)
    selected = (
        selected.drop_duplicates(["event_id"])
        .sort_values("event_time")
        .head(max_events)
        .reset_index(drop=True)
    )
    return selected


# %% [markdown]
# ## General visual helpers


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(t):
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def ease_in_out_sine(t):
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


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
            continue
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int = 42,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    stroke: int = 2,
    anchor: str = "la",
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
    draw.text(
        xy,
        text,
        font=font,
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(fill[3] if len(fill) > 3 else 255, 220)),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int = 31,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    line_spacing: int = 8,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
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
            stroke_fill=(0, 0, 0, 220),
        )
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += (bbox[3] - bbox[1]) + line_spacing


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cx = width / 2.0
    cy = height / 2.0
    nx = (xx - cx) / (width / 2.0)
    ny = (yy - cy) / (height / 2.0)
    radius = np.sqrt(nx**2 + ny**2)
    return np.clip(1.0 - strength * radius**1.85, 0.0, 1.0).astype(np.float32)


def apply_grade(arr: np.ndarray) -> np.ndarray:
    image = Image.fromarray(arr)
    image = ImageEnhance.Contrast(image).enhance(CONFIG["contrast_boost"])
    image = ImageEnhance.Color(image).enhance(CONFIG["saturation_boost"])
    return np.array(image)


def format_energy_kt(value: float) -> str:
    if value >= 100:
        return f"{value:.0f} kt"
    if value >= 10:
        return f"{value:.1f} kt"
    if value >= 1:
        return f"{value:.2f} kt"
    return f"{value:.3f} kt"


def format_velocity(value: float) -> str:
    return "not reported" if not np.isfinite(value) else f"{value:.1f} km/s"


def format_altitude(value: float) -> str:
    return "altitude not reported" if not np.isfinite(value) else f"{value:.1f} km altitude"


VIGNETTE = make_vignette(OUT_SIZE[0], OUT_SIZE[1], CONFIG["vignette_strength"])


# %% [markdown]
# ## Cinematic timeline

CAPTIONS = [
    (0.4, 5.7, "These are cataloged bright atmospheric events detected by U.S. Government sensors."),
    (6.2, 16.8, "The archive replays events in peak-brightness time order across the rotating Earth."),
    (17.2, 28.5, "Each flare uses the reported latitude and longitude of peak brightness."),
    (29.0, 40.5, "Larger flashes represent greater calculated total impact energy."),
    (41.0, 49.5, "When velocity components are reported, the entry streak follows their projected direction."),
    (50.0, 57.2, "This public archive is not complete or real-time, and reported values may be revised."),
]

SHOT_PLAN = [
    {
        "name": "boot",
        "start": 0.0,
        "end": 6.0,
        "earth_x_start": 555,
        "earth_x_end": 540,
        "earth_y_start": 1035,
        "earth_y_end": 1000,
        "radius_start": 290,
        "radius_end": 340,
        "lon_start": -20,
        "lon_end": 10,
        "caption": "CNEOS SENSOR ARCHIVE // ATMOSPHERIC EVENTS",
    },
    {
        "name": "replay",
        "start": 6.0,
        "end": 18.0,
        "earth_x_start": 540,
        "earth_x_end": 505,
        "earth_y_start": 1000,
        "earth_y_end": 955,
        "radius_start": 340,
        "radius_end": 380,
        "lon_start": 10,
        "lon_end": 115,
        "caption": "CHRONOLOGICAL REPLAY // PEAK BRIGHTNESS UTC",
    },
    {
        "name": "location",
        "start": 18.0,
        "end": 30.0,
        "earth_x_start": 505,
        "earth_x_end": 560,
        "earth_y_start": 955,
        "earth_y_end": 935,
        "radius_start": 380,
        "radius_end": 400,
        "lon_start": 115,
        "lon_end": 225,
        "caption": "GLOBE PROJECTION // REPORTED LATITUDE + LONGITUDE",
    },
    {
        "name": "energy",
        "start": 30.0,
        "end": 41.0,
        "earth_x_start": 560,
        "earth_x_end": 520,
        "earth_y_start": 935,
        "earth_y_end": 1010,
        "radius_start": 400,
        "radius_end": 370,
        "lon_start": 225,
        "lon_end": 315,
        "caption": "IMPACT ENERGY // FLARE SCALE",
    },
    {
        "name": "velocity",
        "start": 41.0,
        "end": 50.0,
        "earth_x_start": 520,
        "earth_x_end": 600,
        "earth_y_start": 1010,
        "earth_y_end": 960,
        "radius_start": 370,
        "radius_end": 410,
        "lon_start": 315,
        "lon_end": 395,
        "caption": "ENTRY VECTOR // REPORTED COMPONENTS WHEN AVAILABLE",
    },
    {
        "name": "outro",
        "start": 50.0,
        "end": CONFIG["duration_s"],
        "earth_x_start": 600,
        "earth_x_end": 540,
        "earth_y_start": 960,
        "earth_y_end": 1020,
        "radius_start": 410,
        "radius_end": 300,
        "lon_start": 395,
        "lon_end": 455,
        "caption": "PUBLIC FIREBALL RECORD // EVENTS MAY BE REVISED",
    },
]


def get_shot(t: float) -> Dict:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def shot_state(t: float):
    shot = get_shot(t)
    duration = max(shot["end"] - shot["start"], 1e-6)
    u = (t - shot["start"]) / duration
    e = ease_in_out_sine(u)
    earth_x = lerp(shot["earth_x_start"], shot["earth_x_end"], e)
    earth_y = lerp(shot["earth_y_start"], shot["earth_y_end"], e)
    radius = lerp(shot["radius_start"], shot["radius_end"], e)
    central_lon_deg = lerp(shot["lon_start"], shot["lon_end"], e)
    return shot, earth_x, earth_y, radius, central_lon_deg


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


# %% [markdown]
# ## Earth fireball renderer

class EarthFireballScene:
    def __init__(self, event_df: pd.DataFrame):
        self.events = event_df.reset_index(drop=True).copy()

        start = self.events["event_time"].min()
        end = self.events["event_time"].max()
        if pd.isna(start) or pd.isna(end) or start >= end:
            raise RuntimeError("Fireball catalog timeline is invalid.")

        self.start_time = start
        self.end_time = end
        self.timeline_seconds = (end - start).total_seconds()
        self.event_times = (
            (self.events["event_time"] - start).dt.total_seconds().to_numpy(float)
            / self.timeline_seconds
        )

        self.lat = np.deg2rad(self.events["latitude_signed_deg"].to_numpy(float))
        self.lon = np.deg2rad(self.events["longitude_signed_deg"].to_numpy(float))
        self.altitude = self.events["altitude_km"].to_numpy(float)
        self.energy = self.events["impact_energy_kt"].to_numpy(float)
        self.radiated = self.events["radiated_energy_1e10_j"].to_numpy(float)
        self.velocity = self.events["entry_velocity_km_s"].to_numpy(float)
        self.vx = self.events["velocity_x_km_s"].to_numpy(float)
        self.vy = self.events["velocity_y_km_s"].to_numpy(float)
        self.vz = self.events["velocity_z_km_s"].to_numpy(float)
        self.velocity_reported = self.events["velocity_reported"].to_numpy(bool)
        self.phases = self.events["visual_phase"].to_numpy(float)
        self.fallback_angles = self.events["fallback_angle"].to_numpy(float)

        self.max_log_energy = max(float(np.log1p(np.nanmax(self.energy))), 1e-6)
        finite_velocity = self.velocity[np.isfinite(self.velocity)]
        self.max_velocity = max(float(np.nanmax(finite_velocity)) if len(finite_velocity) else 1.0, 1.0)

        self.highlight_indices = (
            self.events["highlight_score"]
            .sort_values(ascending=False)
            .head(CONFIG["highlight_count"])
            .index
            .to_numpy(int)
        )
        self.energy_rank_indices = np.argsort(-self.energy)

        self.particles = self._make_particles(CONFIG["background_particle_count"], seed=41)
        self.hud_noise = self._make_hud_noise(CONFIG["hud_noise_count"], seed=103)
        self.land_blobs = self._make_land_blobs(seed=72)

    @staticmethod
    def _make_particles(count: int, seed: int):
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_SIZE[0])),
                "y": float(rng.uniform(0, OUT_SIZE[1])),
                "r": float(rng.uniform(0.4, 1.9)),
                "a": int(rng.integers(18, 112)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "drift": float(rng.uniform(-18, 18)),
            }
            for _ in range(count)
        ]

    @staticmethod
    def _make_hud_noise(count: int, seed: int):
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_SIZE[0])),
                "y": float(rng.uniform(0, OUT_SIZE[1])),
                "length": float(rng.uniform(8, 95)),
                "alpha": int(rng.integers(12, 58)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            }
            for _ in range(count)
        ]

    @staticmethod
    def _make_land_blobs(seed: int):
        # Decorative texture only; intentionally not a geographic coastline map.
        rng = np.random.default_rng(seed)
        blobs = []
        for _ in range(48):
            blobs.append({
                "lat": float(rng.uniform(-68, 75)),
                "lon": float(rng.uniform(-180, 180)),
                "rx": float(rng.uniform(0.035, 0.13)),
                "ry": float(rng.uniform(0.025, 0.09)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            })
        return blobs

    def catalog_fraction(self, t: float) -> float:
        start_t = 4.8
        end_t = CONFIG["duration_s"] - 4.0
        return smoothstep((t - start_t) / max(end_t - start_t, 1e-6))

    def catalog_time(self, t: float) -> pd.Timestamp:
        fraction = self.catalog_fraction(t)
        return self.start_time + pd.to_timedelta(fraction * self.timeline_seconds, unit="s")

    def active_event_indices(self, t: float) -> np.ndarray:
        fraction = self.catalog_fraction(t)
        screen_window = 0.035
        delta = fraction - self.event_times
        active = np.where((delta >= -0.008) & (delta <= screen_window))[0]

        if len(active) > CONFIG["max_simultaneous_events"]:
            order = np.argsort(-self.energy[active])
            active = active[order[: CONFIG["max_simultaneous_events"]]]
        return active

    def project_points(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        earth_x: float,
        earth_y: float,
        radius: float,
        central_lon_deg: float,
        radial_scale: Optional[np.ndarray] = None,
    ):
        central_lon = math.radians(central_lon_deg)
        rel = lon - central_lon
        cos_lat = np.cos(lat)
        sx = cos_lat * np.sin(rel)
        sy = np.sin(lat)
        depth = cos_lat * np.cos(rel)

        if radial_scale is None:
            radial_scale = np.ones_like(sx)

        px = earth_x + radius * radial_scale * sx
        py = earth_y - radius * radial_scale * sy
        return px, py, depth

    def render_background(self, t: float) -> Image.Image:
        canvas = Image.new("RGBA", OUT_SIZE, (1, 3, 10, 255))
        draw = ImageDraw.Draw(canvas)

        for particle in self.particles:
            x = (particle["x"] + particle["drift"] * 0.075 * t) % OUT_SIZE[0]
            y = (particle["y"] + particle["drift"] * 0.022 * t) % OUT_SIZE[1]
            twinkle = 0.68 + 0.32 * math.sin(t * 1.05 + particle["phase"])
            alpha = int(particle["a"] * twinkle)
            r = particle["r"]
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(205, 228, 255, alpha))

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        centers = [
            (OUT_SIZE[0] * 0.50, OUT_SIZE[1] * 0.50, (0, 95, 160)),
            (OUT_SIZE[0] * 0.13, OUT_SIZE[1] * 0.28, (25, 40, 125)),
            (OUT_SIZE[0] * 0.86, OUT_SIZE[1] * 0.70, (120, 20, 80)),
        ]
        for cx, cy, color in centers:
            for r, alpha in [(650, 17), (430, 24), (220, 32)]:
                gd.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(*color, alpha))
        glow = glow.filter(ImageFilter.GaussianBlur(90))
        canvas.alpha_composite(glow)
        return canvas

    def draw_earth(
        self,
        canvas: Image.Image,
        earth_x: float,
        earth_y: float,
        radius: float,
        central_lon_deg: float,
        t: float,
    ):
        atmosphere = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        ad = ImageDraw.Draw(atmosphere)
        for scale, alpha in [(1.15, 16), (1.09, 28), (1.045, 60)]:
            r = radius * scale
            ad.ellipse((earth_x-r, earth_y-r, earth_x+r, earth_y+r), fill=(40, 150, 255, alpha))
        atmosphere = atmosphere.filter(ImageFilter.GaussianBlur(28))
        canvas.alpha_composite(atmosphere)

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        draw.ellipse(
            (earth_x-radius, earth_y-radius, earth_x+radius, earth_y+radius),
            fill=(5, 31, 72, 255),
            outline=(95, 220, 255, 230),
            width=3,
        )

        # Soft day-side glow and night-side shadow.
        light = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        ld = ImageDraw.Draw(light)
        ld.ellipse(
            (earth_x-radius*1.12, earth_y-radius*1.05, earth_x+radius*0.35, earth_y+radius*1.05),
            fill=(15, 115, 175, 90),
        )
        light = light.filter(ImageFilter.GaussianBlur(max(8, int(radius * 0.08))))
        layer.alpha_composite(light)

        draw.ellipse(
            (earth_x-radius*0.05, earth_y-radius*1.03, earth_x+radius*1.35, earth_y+radius*1.03),
            fill=(0, 3, 13, 150),
        )

        # Decorative pseudo-land texture projected like real spherical points.
        land_lat = np.deg2rad(np.array([b["lat"] for b in self.land_blobs], dtype=float))
        land_lon = np.deg2rad(np.array([b["lon"] for b in self.land_blobs], dtype=float))
        px, py, depth = self.project_points(
            land_lat, land_lon, earth_x, earth_y, radius, central_lon_deg
        )
        for i, blob in enumerate(self.land_blobs):
            if depth[i] <= 0.04:
                continue
            pulse = 0.88 + 0.12 * math.sin(t * 0.28 + blob["phase"])
            rx = radius * blob["rx"] * (0.45 + 0.55 * depth[i])
            ry = radius * blob["ry"] * pulse
            alpha = int(30 + 65 * depth[i])
            draw.ellipse(
                (px[i]-rx, py[i]-ry, px[i]+rx, py[i]+ry),
                fill=(35, 118, 120, alpha),
            )

        # Latitude lines.
        for latitude in [-60, -30, 0, 30, 60]:
            lat = math.radians(latitude)
            y = earth_y - radius * math.sin(lat)
            half_width = radius * math.cos(lat)
            height = max(8.0, radius * 0.10 * math.cos(lat))
            draw.ellipse(
                (earth_x-half_width, y-height, earth_x+half_width, y+height),
                outline=(75, 175, 220, 65),
                width=1,
            )

        # Longitude curves, shifted with camera longitude.
        rotation = math.radians(central_lon_deg)
        for lon_offset_deg in [-120, -60, 0, 60, 120, 180]:
            phase = math.radians(lon_offset_deg) - rotation
            x = earth_x + radius * math.sin(phase) * 0.88
            width = radius * (0.05 + 0.16 * abs(math.cos(phase)))
            draw.ellipse(
                (x-width, earth_y-radius*0.94, x+width, earth_y+radius*0.94),
                outline=(60, 170, 220, 47),
                width=1,
            )

        # Planet scan arc.
        sweep = (t * 0.55) % (2 * math.pi)
        for offset, alpha in [(0.0, 120), (-0.055, 50), (-0.11, 22)]:
            angle = sweep + offset
            x2 = earth_x + math.cos(angle) * radius
            y2 = earth_y + math.sin(angle) * radius
            draw.line((earth_x, earth_y, x2, y2), fill=(80, 245, 220, alpha), width=2)

        canvas.alpha_composite(layer)

    def velocity_screen_direction(self, idx: int, central_lon_deg: float) -> Tuple[float, float]:
        if self.velocity_reported[idx]:
            camera_lon = math.radians(central_lon_deg)
            # Earth-fixed world axes: screen horizontal is camera-tangent in XY;
            # screen vertical is opposite Earth-fixed Z.
            dx = -math.sin(camera_lon) * self.vx[idx] + math.cos(camera_lon) * self.vy[idx]
            dy = -self.vz[idx]
            norm = math.hypot(dx, dy)
            if norm > 1e-8:
                return dx / norm, dy / norm

        angle = self.fallback_angles[idx]
        return math.cos(angle), math.sin(angle)

    def event_geometry(
        self,
        idx: int,
        t: float,
        earth_x: float,
        earth_y: float,
        radius: float,
        central_lon_deg: float,
    ):
        fraction = self.catalog_fraction(t)
        event_fraction = self.event_times[idx]
        screen_window = 0.035
        local = (fraction - event_fraction + 0.008) / (screen_window + 0.008)
        local = clamp(local)

        altitude = self.altitude[idx]
        alt_offset = 0.0
        if np.isfinite(altitude):
            alt_offset = np.clip(altitude, 0.0, 120.0) * CONFIG["atmosphere_height_scale_px"]

        radial_scale = np.array([1.0 + alt_offset / max(radius, 1.0)])
        px, py, depth = self.project_points(
            np.array([self.lat[idx]]),
            np.array([self.lon[idx]]),
            earth_x,
            earth_y,
            radius,
            central_lon_deg,
            radial_scale=radial_scale,
        )
        x = float(px[0])
        y = float(py[0])
        z = float(depth[0])

        dx, dy = self.velocity_screen_direction(idx, central_lon_deg)
        velocity_norm = 0.35 + 0.65 * (
            min(self.velocity[idx], self.max_velocity) / self.max_velocity
            if np.isfinite(self.velocity[idx]) else 0.45
        )
        trail_length = radius * (0.16 + 0.24 * velocity_norm)

        # Before the flare peaks, the streak travels toward the event location.
        approach = 1.0 - smoothstep(local / 0.42)
        head_x = x + dx * trail_length * approach
        head_y = y + dy * trail_length * approach

        return {
            "local": local,
            "x": x,
            "y": y,
            "depth": z,
            "dx": dx,
            "dy": dy,
            "trail_length": trail_length,
            "head_x": head_x,
            "head_y": head_y,
        }

    def draw_events(
        self,
        canvas: Image.Image,
        earth_x: float,
        earth_y: float,
        radius: float,
        central_lon_deg: float,
        t: float,
    ):
        active = self.active_event_indices(t)
        if not len(active):
            return

        bloom = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        bd = ImageDraw.Draw(bloom)
        sharp = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sharp)

        highlight_set = set(self.highlight_indices.tolist())

        for idx in active:
            geom = self.event_geometry(idx, t, earth_x, earth_y, radius, central_lon_deg)
            if geom["depth"] <= -0.04:
                continue

            local = geom["local"]
            visible = clamp((geom["depth"] + 0.04) / 0.25)
            if visible <= 0:
                continue

            energy_norm = math.log1p(self.energy[idx]) / self.max_log_energy
            energy_norm = clamp(energy_norm)
            track_alpha = int(190 * visible * (1.0 - smoothstep((local - 0.60) / 0.35)))

            dx = geom["dx"]
            dy = geom["dy"]
            x = geom["x"]
            y = geom["y"]
            hx = geom["head_x"]
            hy = geom["head_y"]
            tail_length = geom["trail_length"] * (0.42 + 0.50 * energy_norm)
            tx = hx + dx * tail_length
            ty = hy + dy * tail_length

            if track_alpha > 3:
                sd.line((tx, ty, hx, hy), fill=(255, 185, 95, track_alpha), width=max(2, int(3 + 5 * energy_norm)))
                bd.line((tx, ty, hx, hy), fill=(255, 90, 25, int(track_alpha * 0.55)), width=max(7, int(12 + 14 * energy_norm)))

            # Brightness pulse around local ~0.42.
            pulse = math.exp(-((local - 0.43) / 0.115) ** 2)
            afterglow = 0.45 * math.exp(-max(local - 0.43, 0.0) * 4.0)
            flare = clamp((pulse + afterglow) * visible)
            if flare <= 0.005:
                continue

            flare_r = 5.0 + 30.0 * energy_norm + 38.0 * energy_norm * pulse
            bd.ellipse(
                (x-flare_r*2.2, y-flare_r*2.2, x+flare_r*2.2, y+flare_r*2.2),
                fill=(255, 90, 25, int(75 * flare)),
            )
            bd.ellipse(
                (x-flare_r*1.2, y-flare_r*1.2, x+flare_r*1.2, y+flare_r*1.2),
                fill=(255, 195, 65, int(130 * flare)),
            )
            sd.ellipse(
                (x-flare_r*0.24, y-flare_r*0.24, x+flare_r*0.24, y+flare_r*0.24),
                fill=(255, 250, 220, int(250 * flare)),
            )

            # Target lock for editorially selected high-energy events.
            if idx in highlight_set and 0.32 <= local <= 0.82:
                lock_alpha = int(220 * visible * (1.0 - abs(local - 0.56) / 0.30))
                lock_alpha = max(0, lock_alpha)
                ring = 28 + 28 * energy_norm + 8 * math.sin(t * 7.0 + self.phases[idx])
                sd.ellipse((x-ring, y-ring, x+ring, y+ring), outline=(110, 245, 225, lock_alpha), width=2)
                sd.line((x-ring-14, y, x-ring+2, y), fill=(110, 245, 225, lock_alpha), width=2)
                sd.line((x+ring-2, y, x+ring+14, y), fill=(110, 245, 225, lock_alpha), width=2)
                sd.line((x, y-ring-14, x, y-ring+2), fill=(110, 245, 225, lock_alpha), width=2)
                sd.line((x, y+ring-2, x, y+ring+14), fill=(110, 245, 225, lock_alpha), width=2)

                event_time = self.events.iloc[idx]["event_time"].strftime("%Y-%m-%d")
                label = f"{event_time} // {format_energy_kt(self.energy[idx])}"
                draw_text(
                    sharp,
                    label,
                    (int(x + ring + 18), int(y - ring - 5)),
                    size=19,
                    fill=(235, 250, 255, min(lock_alpha, 230)),
                    bold=True,
                    stroke=1,
                )

        bloom = bloom.filter(ImageFilter.GaussianBlur(16))
        canvas.alpha_composite(bloom)
        canvas.alpha_composite(sharp)

    def draw_energy_spectrum(self, canvas: Image.Image, t: float):
        if t < 27.5:
            return
        alpha = int(205 * smoothstep((t - 27.5) / 2.2))
        x0, y0 = 64, 1345
        w, h = 390, 205

        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((x0, y0, x0+w, y0+h), radius=20, fill=(2, 9, 19, int(alpha*0.52)), outline=(65, 190, 220, int(alpha*0.45)), width=1)
        draw_text(panel, "IMPACT ENERGY SPECTRUM", (x0+20, y0+19), size=20, fill=(170, 235, 245, alpha), bold=True, stroke=1)

        loge = np.log10(np.maximum(self.energy, 1e-5))
        low, high = float(np.nanmin(loge)), float(np.nanmax(loge))
        bins = np.linspace(low, high, 18)
        counts, _ = np.histogram(loge, bins=bins)
        max_count = max(int(counts.max()), 1)
        base_y = y0 + h - 37
        bar_w = (w - 42) / len(counts)
        for i, count in enumerate(counts):
            bh = 92 * count / max_count
            bx = x0 + 21 + i * bar_w
            draw.rectangle((bx, base_y-bh, bx+max(2, bar_w-3), base_y), fill=(255, 135, 65, int(alpha*0.75)))

        draw_text(panel, "LOW", (x0+20, y0+h-20), size=16, fill=(180, 205, 220, alpha), stroke=1)
        draw_text(panel, "HIGH", (x0+w-22, y0+h-20), size=16, fill=(255, 205, 155, alpha), stroke=1, anchor="ra")
        canvas.alpha_composite(panel)

    def draw_altitude_strip(self, canvas: Image.Image, t: float):
        if t < 16.0:
            return
        alpha = int(200 * smoothstep((t - 16.0) / 2.2))
        finite = self.altitude[np.isfinite(self.altitude)]
        if not len(finite):
            return

        x0, y0 = 700, 480
        w, h = 315, 390
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((x0, y0, x0+w, y0+h), radius=20, fill=(2, 9, 19, int(alpha*0.50)), outline=(65, 190, 220, int(alpha*0.42)), width=1)
        draw_text(panel, "PEAK-BRIGHTNESS ALTITUDE", (x0+18, y0+18), size=18, fill=(165, 232, 245, alpha), bold=True, stroke=1)

        scale_top = y0 + 65
        scale_bottom = y0 + h - 34
        for altitude in [0, 20, 40, 60, 80, 100]:
            yy = scale_bottom - (altitude / 100.0) * (scale_bottom - scale_top)
            draw.line((x0+62, yy, x0+w-22, yy), fill=(95, 170, 205, int(alpha*0.25)), width=1)
            draw_text(panel, f"{altitude:>3} km", (x0+18, int(yy)), size=15, fill=(175, 205, 220, int(alpha*0.75)), stroke=1, anchor="lm")

        active = self.active_event_indices(t)
        for idx in active:
            if not np.isfinite(self.altitude[idx]):
                continue
            alt = clamp(self.altitude[idx] / 100.0)
            yy = scale_bottom - alt * (scale_bottom - scale_top)
            energy_norm = clamp(math.log1p(self.energy[idx]) / self.max_log_energy)
            rr = 3 + 7 * energy_norm
            draw.ellipse((x0+w-55-rr, yy-rr, x0+w-55+rr, yy+rr), fill=(255, 180, 75, int(alpha*0.90)))

        median_alt = float(np.nanmedian(finite))
        draw_text(panel, f"catalog median {median_alt:.1f} km", (x0+18, y0+h-16), size=16, fill=(210, 225, 235, alpha), stroke=1, anchor="ls")
        canvas.alpha_composite(panel)

    def draw_energy_leaderboard(self, canvas: Image.Image, t: float):
        if t < 39.5:
            return
        alpha = int(225 * smoothstep((t - 39.5) / 2.0))
        x0, y0 = 55, 450
        w, h = 560, 335
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((x0, y0, x0+w, y0+h), radius=22, fill=(1, 8, 18, int(alpha*0.62)), outline=(80, 210, 225, int(alpha*0.50)), width=1)
        draw_text(panel, "STRONGEST EVENTS // CALCULATED IMPACT ENERGY", (x0+20, y0+18), size=18, fill=(165, 238, 245, alpha), bold=True, stroke=1)

        count = int(CONFIG["leaderboard_count"])
        for rank, idx in enumerate(self.energy_rank_indices[:count], start=1):
            yy = y0 + 66 + (rank-1) * 42
            date = self.events.iloc[idx]["event_time"].strftime("%Y-%m-%d")
            lat = self.events.iloc[idx]["latitude_signed_deg"]
            lon = self.events.iloc[idx]["longitude_signed_deg"]
            hemi_lat = "N" if lat >= 0 else "S"
            hemi_lon = "E" if lon >= 0 else "W"
            location = f"{abs(lat):.1f}{hemi_lat} {abs(lon):.1f}{hemi_lon}"
            draw_text(panel, f"{rank:02d}", (x0+20, yy), size=18, fill=(100, 230, 220, alpha), bold=True, stroke=1, anchor="lm")
            draw_text(panel, date, (x0+68, yy), size=18, fill=(235, 244, 250, alpha), bold=True, stroke=1, anchor="lm")
            draw_text(panel, location, (x0+245, yy), size=16, fill=(185, 210, 225, alpha), stroke=1, anchor="lm")
            draw_text(panel, format_energy_kt(self.energy[idx]), (x0+w-20, yy), size=18, fill=(255, 190, 105, alpha), bold=True, stroke=1, anchor="rm")

        canvas.alpha_composite(panel)

    def draw_timeline(self, canvas: Image.Image, t: float):
        alpha = int(205 * smoothstep((t - 4.3) / 1.8))
        if alpha <= 4:
            return

        x0, x1 = 70, OUT_SIZE[0] - 70
        y = 1645
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.line((x0, y, x1, y), fill=(85, 190, 220, int(alpha*0.60)), width=2)

        for fraction in np.linspace(0, 1, 9):
            xx = lerp(x0, x1, fraction)
            draw.line((xx, y-9, xx, y+9), fill=(105, 205, 225, int(alpha*0.55)), width=1)

        fraction = self.catalog_fraction(t)
        xx = lerp(x0, x1, fraction)
        draw.ellipse((xx-8, y-8, xx+8, y+8), fill=(255, 200, 90, alpha))

        current = self.catalog_time(t)
        draw_text(panel, self.start_time.strftime("%Y"), (x0, y+22), size=18, fill=(175, 205, 220, alpha), stroke=1)
        draw_text(panel, self.end_time.strftime("%Y"), (x1, y+22), size=18, fill=(175, 205, 220, alpha), stroke=1, anchor="ra")
        draw_text(panel, current.strftime("%Y-%m-%d // %H:%M UTC"), ((x0+x1)//2, y-28), size=22, fill=(240, 247, 250, alpha), bold=True, stroke=1, anchor="ma")
        canvas.alpha_composite(panel)

    def draw_corner_hud(self, canvas: Image.Image, t: float, active_count: int):
        alpha = int(200 * smoothstep((t - 2.0) / 1.8))
        if alpha <= 4:
            return

        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        x0, y0 = 63, 270
        w, h = 395, 140
        draw.rounded_rectangle((x0, y0, x0+w, y0+h), radius=18, fill=(1, 7, 16, int(alpha*0.44)), outline=(65, 185, 215, int(alpha*0.42)), width=1)
        current_fraction = self.catalog_fraction(t)
        passed = int(np.searchsorted(self.event_times, current_fraction, side="right"))

        draw_text(panel, "ARCHIVE LINK", (x0+18, y0+17), size=18, fill=(105, 235, 220, alpha), bold=True, stroke=1)
        draw_text(panel, f"events loaded  {len(self.events):,}", (x0+18, y0+52), size=19, fill=(225, 238, 245, alpha), stroke=1)
        draw_text(panel, f"events replayed {passed:,}", (x0+18, y0+82), size=19, fill=(225, 238, 245, alpha), stroke=1)
        draw_text(panel, f"active flares    {active_count:02d}", (x0+18, y0+112), size=19, fill=(255, 190, 105, alpha), stroke=1)
        canvas.alpha_composite(panel)

    def add_text_layers(self, canvas: Image.Image, t: float, shot: Dict):
        title_alpha = int(255 * smoothstep((t - 0.25) / 1.0) * (1.0 - smoothstep((t - 5.4) / 1.2)))
        if title_alpha > 5:
            draw_text(canvas, CONFIG["title_text"], (62, 118), size=56, fill=(255, 255, 255, title_alpha), bold=True)
            draw_text(canvas, CONFIG["subtitle_text"], (64, 202), size=25, fill=(215, 232, 255, min(225, title_alpha)))

        if t > 6.0:
            draw_text(canvas, shot["caption"], (62, 78), size=23, fill=(195, 222, 245, 190), stroke=1)

        caption = caption_at(t)
        if caption:
            panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            draw = ImageDraw.Draw(panel)
            y0 = OUT_SIZE[1] - 252
            draw.rounded_rectangle((46, y0, OUT_SIZE[0]-46, y0+126), radius=26, fill=(0, 0, 0, 132), outline=(55, 180, 220, 70), width=1)
            canvas.alpha_composite(panel)
            draw_wrapped_text(canvas, caption, (70, y0+27), max_width=OUT_SIZE[0]-140, size=30, fill=(245, 250, 255, 245))

        note_alpha = int(220 * smoothstep((t - 50.5) / 3.0))
        if note_alpha > 4:
            draw_wrapped_text(canvas, CONFIG["credit_text"], (65, OUT_SIZE[1]-111), max_width=940, size=19, fill=(220, 232, 245, note_alpha))
            draw_wrapped_text(canvas, CONFIG["scientific_note"], (65, OUT_SIZE[1]-79), max_width=940, size=17, fill=(190, 210, 232, note_alpha))

    def draw_hud_noise(self, canvas: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for item in self.hud_noise:
            pulse = 0.5 + 0.5 * math.sin(t * 1.85 + item["phase"])
            if pulse < 0.72:
                continue
            x = item["x"]
            y = (item["y"] + t * 11.0) % OUT_SIZE[1]
            draw.line((x, y, x+item["length"], y), fill=(90, 210, 240, int(item["alpha"]*pulse)), width=1)

        offset = int((t * 39) % 7)
        for y in range(offset, OUT_SIZE[1], 7):
            draw.line((0, y, OUT_SIZE[0], y), fill=(120, 205, 245, 12), width=1)

        scan_y = int((t * 152) % (OUT_SIZE[1] + 260)) - 130
        draw.rectangle((0, scan_y, OUT_SIZE[0], scan_y+54), fill=(80, 210, 240, 9))
        canvas.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot, earth_x, earth_y, radius, central_lon_deg = shot_state(t)
        active = self.active_event_indices(t)

        canvas = self.render_background(t)
        self.draw_earth(canvas, earth_x, earth_y, radius, central_lon_deg, t)
        self.draw_events(canvas, earth_x, earth_y, radius, central_lon_deg, t)
        self.draw_altitude_strip(canvas, t)
        self.draw_energy_spectrum(canvas, t)
        self.draw_energy_leaderboard(canvas, t)
        self.draw_timeline(canvas, t)
        self.draw_corner_hud(canvas, t, len(active))
        self.add_text_layers(canvas, t, shot)
        self.draw_hud_noise(canvas, t)

        arr = np.array(canvas.convert("RGB"))
        arr = apply_grade(arr)
        arr = np.clip(arr.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)

        fade_in = smoothstep(t / 0.9)
        fade_out = 1.0 - smoothstep((t - (CONFIG["duration_s"] - 1.1)) / 1.0)
        arr = np.clip(arr.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)
        return arr


# %% [markdown]
# ## Scientific preview plots


def create_scientific_previews(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df["longitude_signed_deg"],
        df["latitude_signed_deg"],
        s=8 + 20 * np.clip(np.log1p(df["impact_energy_kt"]), 0, 5),
        alpha=0.45,
    )
    ax.set_title("CNEOS fireball event locations in selected catalog window")
    ax.set_xlabel("Longitude [degrees]")
    ax.set_ylabel("Latitude [degrees]")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "fireball_event_locations.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    finite_alt = df.dropna(subset=["altitude_km"])
    ax.scatter(
        finite_alt["altitude_km"],
        finite_alt["impact_energy_kt"],
        s=14,
        alpha=0.5,
    )
    ax.set_yscale("log")
    ax.set_title("Peak-brightness altitude vs calculated impact energy")
    ax.set_xlabel("Altitude above reference geoid [km]")
    ax.set_ylabel("Calculated impact energy [kt]")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "altitude_vs_impact_energy.png", dpi=170)
    plt.close(fig)

    dates = df["event_time"].dt.tz_convert(None)
    yearly_counts = pd.Series(np.ones(len(df), dtype=int), index=dates).resample("YS").sum()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(yearly_counts.index.year.astype(str), yearly_counts.to_numpy())
    tick_step = max(1, len(yearly_counts) // 12)
    ticks = np.arange(len(yearly_counts))[::tick_step]
    ax.set_xticks(ticks)
    ax.set_xticklabels(yearly_counts.index.year.astype(str)[::tick_step], rotation=45, ha="right")
    ax.set_title("Reported CNEOS fireball events by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Reported event count")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "reported_events_by_year.png", dpi=170)
    plt.close(fig)

    strongest = df.nlargest(15, "impact_energy_kt").copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    y_positions = np.arange(len(strongest))
    ax.barh(y_positions, strongest["impact_energy_kt"].to_numpy())
    ax.set_yticks(y_positions)
    ax.set_yticklabels(strongest["event_time"].dt.strftime("%Y-%m-%d"))
    ax.set_xscale("log")
    ax.set_title("Strongest events in selected public catalog window")
    ax.set_xlabel("Calculated total impact energy [kt, log scale]")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "strongest_fireball_events.png", dpi=170)
    plt.close(fig)

    print("Scientific preview plots written to:", PREVIEW_DIR.resolve())


# %% [markdown]
# ## Subtitle sidecar


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(captions, path: Path):
    lines = []
    for index, (start, end, text) in enumerate(captions, start=1):
        lines.append(str(index))
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# %% [markdown]
# ## Video render and optional FFmpeg finishing


def run_ffmpeg(command: List[str]):
    print("Running:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def render_video(scene: EarthFireballScene):
    raw_video_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    subbed_video_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_subbed.mp4"
    audio_video_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_with_audio.mp4"
    final_video_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"

    if CONFIG.get("write_subtitle_sidecar", True):
        write_srt(CAPTIONS, srt_path)
        print("Subtitle sidecar written:", srt_path.resolve())

    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    times = np.arange(frame_count) / CONFIG["fps"]

    print(
        f"Rendering {frame_count:,} frames at "
        f"{CONFIG['video_width']}×{CONFIG['video_height']} ..."
    )

    with iio.get_writer(
        raw_video_path,
        fps=CONFIG["fps"],
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times, desc="Rendering Earth fireball archive"):
            writer.append_data(scene.render_frame(float(t)))

    print("Raw video written:", raw_video_path.resolve())

    ffmpeg = find_ffmpeg()
    print("FFmpeg detected:", ffmpeg)
    final_candidate = raw_video_path

    if CONFIG.get("burn_subtitles", False) and ffmpeg and srt_path.exists():
        command = [
            ffmpeg,
            "-y",
            "-i", str(final_candidate),
            "-vf",
            (
                f"subtitles={srt_path}:"
                "force_style=Fontname=DejaVu Sans,"
                "Fontsize=22,Outline=1.2,BorderStyle=3,MarginV=90"
            ),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(subbed_video_path),
        ]
        run_ffmpeg(command)
        final_candidate = subbed_video_path
        print("Subtitled video written:", subbed_video_path.resolve())

    audio_path = CONFIG.get("audio_path")
    if audio_path and Path(audio_path).exists() and ffmpeg:
        command = [
            ffmpeg,
            "-y",
            "-i", str(final_candidate),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(audio_video_path),
        ]
        run_ffmpeg(command)
        final_candidate = audio_video_path
        print("Audio-muxed video written:", audio_video_path.resolve())
    elif audio_path:
        print("audio_path was set, but the file was not found or FFmpeg was unavailable. Skipping audio.")

    if final_candidate.exists():
        shutil.copyfile(final_candidate, final_video_path)
        print("Final video:", final_video_path.resolve())

    return final_video_path


# %% [markdown]
# ## Preview frames


def render_preview_frames(scene: EarthFireballScene):
    preview_times = [1.0, 10.0, 24.0, 42.0, CONFIG["duration_s"] - 1.0]
    for t in tqdm(preview_times, desc="Preview frames"):
        arr = scene.render_frame(float(t))
        Image.fromarray(arr).save(PREVIEW_DIR / f"preview_{int(t):02d}s.png")
    print("Preview images written to:", PREVIEW_DIR.resolve())


# %% [markdown]
# ## Main pipeline


def main():
    print("Starting Earth Fireball Archive pipeline ...")
    df = fetch_fireballs(CONFIG, force_refresh=False)
    create_scientific_previews(df)

    video_df = select_video_events(df, CONFIG)
    print(f"Events selected for video: {len(video_df):,}")

    scene = EarthFireballScene(video_df)
    render_preview_frames(scene)
    final_path = render_video(scene)

    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-", path.name)
    print("Final deliverable:", final_path.resolve())


if __name__ == "__main__":
    main()


# %% [markdown]
# # Suggested voiceover and YouTube Shorts caption
#
# ## Suggested voiceover
#
# > Earth is hit by tiny pieces of space rock all the time, but a few atmospheric
# > entries become exceptionally bright fireballs. This public NASA/JPL CNEOS
# > archive records events detected by U.S. Government sensors. Each point appears
# > at its reported peak-brightness location, and larger flashes represent greater
# > calculated impact energy. Some events also include altitude and entry-velocity
# > components. The archive is not a complete or real-time list, and values may be
# > revised. But together, these detections reveal a planet continually meeting
# > material from space — usually high in the atmosphere.
#
# ## Suggested YouTube Shorts caption
#
# A chronological replay of bright atmospheric fireball events from the public
# NASA/JPL CNEOS archive. Flare locations, times, calculated impact energies and
# reported altitudes come from the catalog; velocity-driven streak direction is
# used where entry-velocity components are available.
#
# #Astronomy #NASA #Fireball #Meteor #Space #PlanetaryDefense #ScienceShorts
