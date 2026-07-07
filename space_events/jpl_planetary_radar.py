# %% [markdown]
# # Planetary Defense Radar — YouTube Short from Real NASA/JPL NEO Flyby Data
#
# This script creates a vertical 1080×1920 cinematic astronomy short from the
# NASA/JPL SBDB Close-Approach Data API.
#
# Theme: PLANETARY DEFENSE RADAR — near-Earth object flybys
#
# Real catalog values used:
# - object designation / full name,
# - close-approach calendar time,
# - nominal, minimum, and maximum approach distance,
# - Earth-relative velocity at close approach,
# - absolute magnitude H,
# - known diameter when available.
#
# Visual language:
# - luminous Earth limb and atmospheric halo,
# - concentric lunar-distance gates,
# - radar sweep and targeting reticles,
# - asteroid flyby streaks triggered in catalog-time order,
# - approach distance controls closest visual pass,
# - relative velocity controls streak length and motion speed,
# - uncertainty ribbons use catalog min/max distance where available,
# - closest-approach leaderboard,
# - velocity spectrum and distance timeline,
# - CRT scanlines, HUD noise, bloom, and camera drift.
#
# Scientific fidelity note:
# Event time, nominal/min/max distance, relative velocity, H, and known diameter
# come from the JPL close-approach catalog.
#
# The API does not provide a sky-plane direction or a literal rendered trajectory
# for each event in this table. Therefore, the flyby angle is deterministically
# assigned from the object designation for cinematic layout. The pass distance is
# data-driven, but the screen-space direction is editorial.
#
# Recommended install:
#
#     pip install numpy pandas matplotlib pillow imageio imageio-ffmpeg requests tqdm
#
# For a quick test render:
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

OUTPUT_ROOT = Path("planetary_defense_radar_short_output")
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
    "output_basename": "jpl_planetary_defense_radar_short",

    # Reproducible close-approach window
    "catalog_start_date": "2020-01-01",
    "catalog_end_date": "2025-01-01",
    "dist_max_lunar": 15.0,

    # NASA/JPL SBDB Close Approach Data API
    "cad_api_url": "https://ssd-api.jpl.nasa.gov/cad.api",
    "expected_api_version": "1.5",
    "request_timeout": (20, 180),
    "request_retries": 5,
    "retry_backoff_s": 1.5,

    # Physical conversion constants used only for displayed units
    "au_km": 149_597_870.7,
    "lunar_distance_km": 384_400.0,

    # Visual limits
    "max_events_for_video": 900,
    "max_simultaneous_flybys": 24,
    "highlight_count": 12,
    "leaderboard_count": 6,

    # Rendering
    "earth_radius_px": 224,
    "outer_gate_radius_px": 468,
    "contrast_boost": 1.15,
    "saturation_boost": 1.10,
    "vignette_strength": 0.27,
    "background_particle_count": 420,
    "hud_noise_count": 60,

    # Text
    "title_text": "PLANETARY DEFENSE RADAR",
    "subtitle_text": "Near-Earth flybys // NASA/JPL close-approach data",
    "credit_text": "Data: NASA/JPL SBDB Close-Approach Data API",
    "scientific_note": (
        "Times, distances and relative velocities are catalog values. "
        "Screen-space flyby direction is editorial because the CAD table does not "
        "supply a rendered approach direction."
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
        "User-Agent": "planetary-defense-radar-short/1.0 (educational visualization)"
    })
    return session


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def deterministic_unit(designation: str, salt: str = "") -> float:
    text = f"{designation}|{salt}".encode("utf-8")
    digest = hashlib.sha256(text).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64 - 1)


def fetch_close_approaches(
    config: Dict,
    force_refresh: bool = False,
) -> pd.DataFrame:
    raw_path = DATA_ROOT / "jpl_cad_raw.json"
    clean_path = DATA_ROOT / "jpl_close_approaches_clean.csv"

    if clean_path.exists() and raw_path.exists() and not force_refresh:
        print("Using cached JPL close-approach data.")
        df = pd.read_csv(clean_path)
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
        return clean_close_approaches(df, save_path=None)

    session = build_retry_session(config)

    params = {
        "date-min": config["catalog_start_date"],
        "date-max": config["catalog_end_date"],
        "dist-max": f"{config['dist_max_lunar']}LD",
        "body": "Earth",
        "neo": "true",
        "sort": "date",
        "diameter": "true",
        "fullname": "true",
    }

    print("Requesting NASA/JPL SBDB Close-Approach Data API ...")
    response = session.get(
        config["cad_api_url"],
        params=params,
        timeout=config["request_timeout"],
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"JPL CAD API HTTP {response.status_code}: {response.reason}\n"
            f"Response preview:\n{response.text[:1600]}"
        )

    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(
            "JPL CAD API did not return valid JSON.\n"
            f"Response preview:\n{response.text[:1600]}"
        ) from exc

    signature = payload.get("signature") or {}
    version = str(signature.get("version") or "")

    if version != str(config["expected_api_version"]):
        raise RuntimeError(
            "JPL CAD API version changed.\n"
            f"Expected: {config['expected_api_version']}\n"
            f"Received: {version!r}\n"
            "Review the official API documentation before rendering so field parsing "
            "does not silently use an incompatible schema."
        )

    fields = payload.get("fields")
    data = payload.get("data")

    if not fields or data is None:
        count = payload.get("count", 0)
        if int(count or 0) == 0:
            raise RuntimeError(
                "The JPL CAD API returned zero close approaches for this query."
            )
        raise RuntimeError(
            f"Unexpected JPL CAD API payload:\n{json.dumps(payload)[:1600]}"
        )

    raw_path.write_text(json.dumps(payload), encoding="utf-8")

    frame = pd.DataFrame(data, columns=fields)
    frame = clean_close_approaches(frame, save_path=clean_path)

    return frame


def clean_close_approaches(
    df: pd.DataFrame,
    save_path: Optional[Path],
) -> pd.DataFrame:
    df = df.copy()

    # Normalize alternate/cached names.
    rename_map = {
        "cd": "close_time_text",
        "dist": "distance_au",
        "dist_min": "distance_min_au",
        "dist_max": "distance_max_au",
        "v_rel": "velocity_km_s",
        "h": "absolute_magnitude_h",
        "diameter": "diameter_km",
        "diameter_sigma": "diameter_sigma_km",
        "fullname": "full_name",
    }

    for source, target in rename_map.items():
        if source in df.columns and target not in df.columns:
            df = df.rename(columns={source: target})

    if "close_time" not in df.columns:
        if "close_time_text" not in df.columns:
            raise RuntimeError("Close-approach data is missing calendar close time.")
        # JPL CAD calendar output is TDB text. For cinematic ordering and labels,
        # parse it as a timezone-naive timestamp and then mark UTC for pandas
        # convenience. The displayed source label continues to say catalog time.
        df["close_time"] = pd.to_datetime(
            df["close_time_text"],
            format="%Y-%b-%d %H:%M",
            errors="coerce",
            utc=True,
        )
    else:
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")

    numeric_columns = [
        "distance_au",
        "distance_min_au",
        "distance_max_au",
        "velocity_km_s",
        "v_inf",
        "absolute_magnitude_h",
        "diameter_km",
        "diameter_sigma_km",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan

    if "des" not in df.columns:
        raise RuntimeError("Close-approach data is missing object designation 'des'.")

    df["des"] = df["des"].fillna("").astype(str).str.strip()

    if "full_name" not in df.columns:
        df["full_name"] = df["des"]
    else:
        df["full_name"] = (
            df["full_name"]
            .fillna(df["des"])
            .astype(str)
            .str.strip()
        )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=[
            "close_time",
            "distance_au",
            "velocity_km_s",
        ]
    ).copy()

    df = df[
        (df["distance_au"] > 0)
        & (df["velocity_km_s"] > 0)
    ].copy()

    au_km = float(CONFIG["au_km"])
    lunar_km = float(CONFIG["lunar_distance_km"])

    df["distance_km"] = df["distance_au"] * au_km
    df["distance_ld"] = df["distance_km"] / lunar_km
    df["distance_min_ld"] = df["distance_min_au"] * au_km / lunar_km
    df["distance_max_ld"] = df["distance_max_au"] * au_km / lunar_km

    # Deterministic editorial angle and lane direction.
    df["visual_angle"] = df["des"].apply(
        lambda name: deterministic_unit(name, "angle") * 2.0 * math.pi
    )
    df["visual_lane"] = df["des"].apply(
        lambda name: deterministic_unit(name, "lane") * 2.0 - 1.0
    )
    df["visual_phase"] = df["des"].apply(
        lambda name: deterministic_unit(name, "phase") * 2.0 * math.pi
    )

    # A ranking score for callouts. This is explicitly editorial:
    # closer + faster + known large diameter gets more attention.
    known_diameter = df["diameter_km"].fillna(0.0)
    df["highlight_score"] = (
        16.0 / np.maximum(df["distance_ld"], 0.08)
        + 0.09 * df["velocity_km_s"]
        + 1.8 * np.log1p(known_diameter * 20.0)
    )

    df = (
        df.sort_values(["close_time", "distance_ld", "des"])
        .drop_duplicates(["des", "close_time"])
        .reset_index(drop=True)
    )

    if save_path is not None:
        save_df = df.copy()
        save_df["close_time"] = save_df["close_time"].astype(str)
        save_df.to_csv(save_path, index=False)
        print("Saved cleaned JPL close-approach CSV:", save_path)

    print(f"Close approaches retained: {len(df):,}")
    print(
        "Distance range:",
        f"{df['distance_ld'].min():.3f}–{df['distance_ld'].max():.3f} LD",
    )
    print(
        "Relative velocity range:",
        f"{df['velocity_km_s'].min():.2f}–{df['velocity_km_s'].max():.2f} km/s",
    )

    return df


def select_video_events(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    if len(df) <= int(config["max_events_for_video"]):
        return df.copy().reset_index(drop=True)

    # Keep globally interesting events, while also preserving broad catalog-time coverage.
    highlight_n = min(220, int(config["max_events_for_video"]) // 3)

    highlights = (
        df.sort_values("highlight_score", ascending=False)
        .head(highlight_n)
    )

    remaining_count = int(config["max_events_for_video"]) - len(highlights)
    timeline_indices = np.linspace(
        0,
        len(df) - 1,
        max(remaining_count, 1),
        dtype=int,
    )
    timeline_sample = df.iloc[timeline_indices]

    selected = pd.concat([highlights, timeline_sample], ignore_index=True)
    selected = (
        selected.drop_duplicates(["des", "close_time"])
        .sort_values("close_time")
        .head(int(config["max_events_for_video"]))
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
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
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
        bbox = draw.textbbox(
            (0, 0),
            candidate,
            font=font,
            stroke_width=2,
        )

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

    return np.clip(
        1.0 - strength * radius**1.85,
        0.0,
        1.0,
    ).astype(np.float32)


def apply_grade(arr: np.ndarray) -> np.ndarray:
    image = Image.fromarray(arr)
    image = ImageEnhance.Contrast(image).enhance(CONFIG["contrast_boost"])
    image = ImageEnhance.Color(image).enhance(CONFIG["saturation_boost"])
    return np.array(image)


def format_distance_ld(value: float) -> str:
    if value < 1:
        return f"{value:.3f} LD"
    return f"{value:.2f} LD"


def format_velocity(value: float) -> str:
    return f"{value:.1f} km/s"


def format_diameter(value: float) -> str:
    if not np.isfinite(value):
        return "diameter unknown"
    if value >= 1.0:
        return f"{value:.2f} km diameter"
    return f"{value * 1000.0:.0f} m diameter"


VIGNETTE = make_vignette(
    OUT_SIZE[0],
    OUT_SIZE[1],
    CONFIG["vignette_strength"],
)


# %% [markdown]
# ## Cinematic timeline

CAPTIONS = [
    (0.4, 5.7, "Every streak is a cataloged near-Earth close approach."),
    (6.2, 16.8, "The clock replays five years of flybys in close-approach time order."),
    (17.2, 28.5, "The distance gates are measured in lunar distances from Earth."),
    (29.0, 40.5, "Faster objects draw longer, sharper tracks through the defense display."),
    (41.0, 49.5, "Closest passes are locked, labeled, and ranked by catalog distance."),
    (50.0, 57.2, "Close in astronomy can still mean hundreds of thousands of kilometers."),
]

SHOT_PLAN = [
    {
        "name": "boot",
        "start": 0.0,
        "end": 6.0,
        "earth_x_start": 575,
        "earth_x_end": 540,
        "earth_y_start": 1005,
        "earth_y_end": 980,
        "zoom_start": 0.82,
        "zoom_end": 0.95,
        "caption": "CNEOS DATA LINK // PLANETARY DEFENSE DISPLAY",
    },
    {
        "name": "replay",
        "start": 6.0,
        "end": 18.0,
        "earth_x_start": 540,
        "earth_x_end": 505,
        "earth_y_start": 980,
        "earth_y_end": 930,
        "zoom_start": 0.95,
        "zoom_end": 1.04,
        "caption": "CATALOG REPLAY // CLOSE-APPROACH TIME",
    },
    {
        "name": "distance",
        "start": 18.0,
        "end": 30.0,
        "earth_x_start": 505,
        "earth_x_end": 555,
        "earth_y_start": 930,
        "earth_y_end": 945,
        "zoom_start": 1.04,
        "zoom_end": 1.09,
        "caption": "DISTANCE GATES // LUNAR DISTANCE SCALE",
    },
    {
        "name": "velocity",
        "start": 30.0,
        "end": 41.0,
        "earth_x_start": 555,
        "earth_x_end": 525,
        "earth_y_start": 945,
        "earth_y_end": 1010,
        "zoom_start": 1.09,
        "zoom_end": 1.01,
        "caption": "RELATIVE VELOCITY // TRACK INTENSITY",
    },
    {
        "name": "lock",
        "start": 41.0,
        "end": 50.0,
        "earth_x_start": 525,
        "earth_x_end": 590,
        "earth_y_start": 1010,
        "earth_y_end": 965,
        "zoom_start": 1.01,
        "zoom_end": 1.12,
        "caption": "CLOSEST PASSES // TARGET LOCK",
    },
    {
        "name": "outro",
        "start": 50.0,
        "end": CONFIG["duration_s"],
        "earth_x_start": 590,
        "earth_x_end": 540,
        "earth_y_start": 965,
        "earth_y_end": 1015,
        "zoom_start": 1.12,
        "zoom_end": 0.80,
        "caption": "EARTH ORBITAL NEIGHBORHOOD // STILL MOVING",
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
    zoom = lerp(shot["zoom_start"], shot["zoom_end"], e)

    return shot, earth_x, earth_y, zoom


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


# %% [markdown]
# ## Planetary defense radar renderer

class PlanetaryDefenseScene:
    def __init__(self, event_df: pd.DataFrame):
        self.events = event_df.reset_index(drop=True).copy()

        start = self.events["close_time"].min()
        end = self.events["close_time"].max()

        if pd.isna(start) or pd.isna(end) or start >= end:
            raise RuntimeError("Close-approach timeline is invalid.")

        self.start_time = start
        self.end_time = end
        self.timeline_seconds = (end - start).total_seconds()

        self.event_times = (
            (self.events["close_time"] - start)
            .dt.total_seconds()
            .to_numpy(float)
            / self.timeline_seconds
        )

        self.designations = self.events["des"].to_numpy(object)
        self.full_names = self.events["full_name"].to_numpy(object)
        self.distance_ld = self.events["distance_ld"].to_numpy(float)
        self.distance_min_ld = self.events["distance_min_ld"].to_numpy(float)
        self.distance_max_ld = self.events["distance_max_ld"].to_numpy(float)
        self.velocity = self.events["velocity_km_s"].to_numpy(float)
        self.h = self.events["absolute_magnitude_h"].to_numpy(float)
        self.diameter = self.events["diameter_km"].to_numpy(float)
        self.angles = self.events["visual_angle"].to_numpy(float)
        self.lanes = self.events["visual_lane"].to_numpy(float)
        self.phases = self.events["visual_phase"].to_numpy(float)

        self.max_velocity = max(float(np.nanmax(self.velocity)), 1.0)
        self.highlight_indices = (
            self.events["highlight_score"]
            .sort_values(ascending=False)
            .head(CONFIG["highlight_count"])
            .index
            .to_numpy(int)
        )

        self.closest_indices = np.argsort(self.distance_ld)
        self.particles = self._make_particles(
            CONFIG["background_particle_count"],
            seed=31,
        )
        self.hud_noise = self._make_hud_noise(
            CONFIG["hud_noise_count"],
            seed=92,
        )

    @staticmethod
    def _make_particles(count: int, seed: int):
        rng = np.random.default_rng(seed)
        particles = []

        for _ in range(count):
            particles.append({
                "x": float(rng.uniform(0, OUT_SIZE[0])),
                "y": float(rng.uniform(0, OUT_SIZE[1])),
                "r": float(rng.uniform(0.4, 1.9)),
                "a": int(rng.integers(20, 115)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "drift": float(rng.uniform(-16, 16)),
            })

        return particles

    @staticmethod
    def _make_hud_noise(count: int, seed: int):
        rng = np.random.default_rng(seed)
        noise = []

        for _ in range(count):
            noise.append({
                "x": float(rng.uniform(0, OUT_SIZE[0])),
                "y": float(rng.uniform(0, OUT_SIZE[1])),
                "length": float(rng.uniform(8, 85)),
                "alpha": int(rng.integers(12, 55)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            })

        return noise

    def catalog_fraction(self, t: float) -> float:
        start_t = 4.8
        end_t = CONFIG["duration_s"] - 4.0
        return smoothstep((t - start_t) / max(end_t - start_t, 1e-6))

    def catalog_time(self, t: float) -> pd.Timestamp:
        fraction = self.catalog_fraction(t)
        return self.start_time + pd.to_timedelta(
            fraction * self.timeline_seconds,
            unit="s",
        )

    def active_event_indices(self, t: float) -> np.ndarray:
        fraction = self.catalog_fraction(t)

        # A fixed screen-time activation window makes nearby events overlap
        # cinematically even when catalog dates are far apart.
        screen_window = 0.032
        delta = fraction - self.event_times

        active = np.where(
            (delta >= -0.010)
            & (delta <= screen_window)
        )[0]

        if len(active) > CONFIG["max_simultaneous_flybys"]:
            order = np.argsort(self.distance_ld[active])
            active = active[order[: CONFIG["max_simultaneous_flybys"]]]

        return active

    def render_background(self, t: float) -> Image.Image:
        canvas = Image.new("RGBA", OUT_SIZE, (1, 4, 10, 255))
        draw = ImageDraw.Draw(canvas)

        for particle in self.particles:
            x = (
                particle["x"]
                + particle["drift"] * 0.08 * t
            ) % OUT_SIZE[0]
            y = (
                particle["y"]
                + particle["drift"] * 0.025 * t
            ) % OUT_SIZE[1]
            twinkle = 0.68 + 0.32 * math.sin(
                t * 1.1 + particle["phase"]
            )
            alpha = int(particle["a"] * twinkle)
            radius = particle["r"]

            draw.ellipse(
                (
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                ),
                fill=(200, 225, 255, alpha),
            )

        # Deep cyan instrument glow.
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)

        centers = [
            (OUT_SIZE[0] * 0.50, OUT_SIZE[1] * 0.52, (0, 105, 150)),
            (OUT_SIZE[0] * 0.17, OUT_SIZE[1] * 0.28, (10, 50, 120)),
            (OUT_SIZE[0] * 0.84, OUT_SIZE[1] * 0.73, (90, 15, 105)),
        ]

        for cx, cy, color in centers:
            for radius, alpha in [(620, 18), (420, 25), (220, 31)]:
                gd.ellipse(
                    (
                        cx - radius,
                        cy - radius,
                        cx + radius,
                        cy + radius,
                    ),
                    fill=(color[0], color[1], color[2], alpha),
                )

        glow = glow.filter(ImageFilter.GaussianBlur(80))
        canvas.alpha_composite(glow)

        return canvas

    def draw_earth(
        self,
        canvas: Image.Image,
        earth_x: float,
        earth_y: float,
        zoom: float,
        t: float,
    ):
        radius = CONFIG["earth_radius_px"] * zoom

        bloom = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        bd = ImageDraw.Draw(bloom)

        for scale, alpha in [(1.35, 18), (1.20, 28), (1.09, 52)]:
            r = radius * scale
            bd.ellipse(
                (
                    earth_x - r,
                    earth_y - r,
                    earth_x + r,
                    earth_y + r,
                ),
                fill=(20, 145, 255, alpha),
            )

        bloom = bloom.filter(ImageFilter.GaussianBlur(30))
        canvas.alpha_composite(bloom)

        earth_layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(earth_layer)

        # Base ocean sphere.
        draw.ellipse(
            (
                earth_x - radius,
                earth_y - radius,
                earth_x + radius,
                earth_y + radius,
            ),
            fill=(8, 38, 88, 255),
            outline=(85, 220, 255, 240),
            width=max(2, int(3 * zoom)),
        )

        # Night-side shadow.
        shadow_offset = radius * 0.42
        draw.ellipse(
            (
                earth_x - radius + shadow_offset,
                earth_y - radius * 1.02,
                earth_x + radius * 1.18 + shadow_offset,
                earth_y + radius * 1.02,
            ),
            fill=(1, 5, 16, 212),
        )

        # Stylized latitude bands.
        for latitude in [-60, -35, 0, 35, 60]:
            y = earth_y + math.sin(math.radians(latitude)) * radius * 0.82
            band_width = math.cos(math.radians(latitude)) * radius * 1.75

            draw.arc(
                (
                    earth_x - band_width / 2,
                    y - radius * 0.16,
                    earth_x + band_width / 2,
                    y + radius * 0.16,
                ),
                0,
                180,
                fill=(55, 155, 220, 82),
                width=max(1, int(2 * zoom)),
            )

        # Rotating pseudo-longitude arcs. These are decorative Earth grid lines.
        rotation = (t * 8.0) % 180.0

        for offset in [-70, -35, 0, 35, 70]:
            x = earth_x + math.sin(math.radians(rotation + offset)) * radius * 0.57
            squash = 0.30 + 0.65 * abs(
                math.cos(math.radians(rotation + offset))
            )

            draw.ellipse(
                (
                    x - radius * 0.15 * squash,
                    earth_y - radius * 0.86,
                    x + radius * 0.15 * squash,
                    earth_y + radius * 0.86,
                ),
                outline=(50, 155, 220, 52),
                width=1,
            )

        # Artificial city glints for cinematic texture, not geospatial data.
        for index in range(34):
            angle = deterministic_unit(str(index), "earth-dot") * 2 * math.pi
            radial = math.sqrt(
                deterministic_unit(str(index), "earth-radius")
            ) * radius * 0.76

            x = earth_x + math.cos(angle) * radial
            y = earth_y + math.sin(angle) * radial
            phase = deterministic_unit(str(index), "earth-phase") * 2 * math.pi
            alpha = int(55 + 95 * (0.5 + 0.5 * math.sin(t * 1.4 + phase)))

            draw.ellipse(
                (x - 1.5, y - 1.5, x + 1.5, y + 1.5),
                fill=(110, 220, 255, alpha),
            )

        canvas.alpha_composite(earth_layer)

    def distance_radius(
        self,
        distance_ld: float,
        zoom: float,
    ) -> float:
        max_ld = max(float(CONFIG["dist_max_lunar"]), 1.0)
        # sqrt compression makes close passes more readable.
        normalized = math.sqrt(
            np.clip(distance_ld / max_ld, 0.0, 1.0)
        )
        return (
            CONFIG["earth_radius_px"] * zoom * 1.12
            + normalized
            * (
                CONFIG["outer_gate_radius_px"] * zoom
                - CONFIG["earth_radius_px"] * zoom * 1.12
            )
        )

    def draw_distance_gates(
        self,
        canvas: Image.Image,
        earth_x: float,
        earth_y: float,
        zoom: float,
        t: float,
    ):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        gate_values = [1, 3, 5, 10, 15]

        for gate in gate_values:
            if gate > CONFIG["dist_max_lunar"]:
                continue

            radius = self.distance_radius(gate, zoom)
            alpha = 100 if gate in [1, 5, 15] else 58

            draw.ellipse(
                (
                    earth_x - radius,
                    earth_y - radius,
                    earth_x + radius,
                    earth_y + radius,
                ),
                outline=(55, 205, 235, alpha),
                width=2 if gate in [1, 5, 15] else 1,
            )

            label_x = int(earth_x + radius * 0.72)
            label_y = int(earth_y - radius * 0.69)

            draw_text(
                overlay,
                f"{gate} LD",
                (label_x, label_y),
                size=19,
                fill=(105, 225, 245, min(215, alpha + 80)),
                bold=True,
                stroke=1,
            )

        # Radar sweep.
        outer = self.distance_radius(CONFIG["dist_max_lunar"], zoom)
        sweep_angle = t * 0.82

        sweep_x = earth_x + math.cos(sweep_angle) * outer
        sweep_y = earth_y + math.sin(sweep_angle) * outer

        draw.line(
            (earth_x, earth_y, sweep_x, sweep_y),
            fill=(90, 255, 220, 120),
            width=3,
        )

        for offset, alpha in [(0.055, 45), (0.11, 25), (0.17, 12)]:
            angle = sweep_angle - offset
            x = earth_x + math.cos(angle) * outer
            y = earth_y + math.sin(angle) * outer

            draw.line(
                (earth_x, earth_y, x, y),
                fill=(80, 235, 210, alpha),
                width=6,
            )

        canvas.alpha_composite(overlay)

    def flyby_geometry(
        self,
        event_index: int,
        t: float,
        earth_x: float,
        earth_y: float,
        zoom: float,
    ):
        fraction = self.catalog_fraction(t)
        event_fraction = self.event_times[event_index]
        screen_window = 0.032

        local = (fraction - event_fraction + 0.010) / (screen_window + 0.010)
        local = np.clip(local, 0.0, 1.0)

        angle = self.angles[event_index]
        direction = np.array(
            [math.cos(angle), math.sin(angle)],
            dtype=float,
        )
        normal = np.array(
            [-direction[1], direction[0]],
            dtype=float,
        )

        closest_radius = self.distance_radius(
            self.distance_ld[event_index],
            zoom,
        )

        lane = self.lanes[event_index] * 0.22
        closest_center = (
            np.array([earth_x, earth_y], dtype=float)
            + normal * closest_radius
            + direction * lane * closest_radius
        )

        speed_norm = np.clip(
            self.velocity[event_index] / self.max_velocity,
            0.0,
            1.0,
        )

        travel_length = (
            620.0
            + 450.0 * speed_norm
        ) * zoom

        along = (local - 0.5) * 2.0 * travel_length
        position = closest_center + direction * along

        streak_length = (
            70.0
            + 260.0 * speed_norm
        ) * zoom

        return (
            position,
            direction,
            normal,
            float(local),
            float(streak_length),
            float(closest_radius),
        )

    def draw_uncertainty_arc(
        self,
        draw: ImageDraw.ImageDraw,
        event_index: int,
        earth_x: float,
        earth_y: float,
        zoom: float,
    ):
        min_ld = self.distance_min_ld[event_index]
        max_ld = self.distance_max_ld[event_index]

        if not np.isfinite(min_ld) or not np.isfinite(max_ld):
            return

        if max_ld <= min_ld:
            return

        inner = self.distance_radius(min_ld, zoom)
        outer = self.distance_radius(max_ld, zoom)

        # Only draw readable uncertainty bands.
        if outer - inner < 1.2:
            return

        angle_deg = math.degrees(self.angles[event_index]) + 90.0

        for radius in [inner, outer]:
            draw.arc(
                (
                    earth_x - radius,
                    earth_y - radius,
                    earth_x + radius,
                    earth_y + radius,
                ),
                start=angle_deg - 13,
                end=angle_deg + 13,
                fill=(255, 185, 70, 85),
                width=2,
            )

    def draw_flybys(
        self,
        canvas: Image.Image,
        earth_x: float,
        earth_y: float,
        zoom: float,
        t: float,
    ):
        active = self.active_event_indices(t)

        glow_layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer)

        sharp_layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        sharp_draw = ImageDraw.Draw(sharp_layer)

        for event_index in active:
            (
                position,
                direction,
                normal,
                local,
                streak_length,
                closest_radius,
            ) = self.flyby_geometry(
                int(event_index),
                t,
                earth_x,
                earth_y,
                zoom,
            )

            x, y = position

            if (
                x < -350
                or x > OUT_SIZE[0] + 350
                or y < -350
                or y > OUT_SIZE[1] + 350
            ):
                continue

            velocity_norm = np.clip(
                self.velocity[event_index] / self.max_velocity,
                0.0,
                1.0,
            )

            distance_norm = np.clip(
                self.distance_ld[event_index] / CONFIG["dist_max_lunar"],
                0.0,
                1.0,
            )

            # Close passes trend warmer; fast tracks trend brighter.
            color = (
                int(lerp(255, 80, distance_norm)),
                int(lerp(95, 225, distance_norm)),
                int(lerp(75, 255, distance_norm)),
                245,
            )

            tail = position - direction * streak_length
            core_width = int(np.clip(2 + 4 * velocity_norm, 2, 7))
            glow_width = core_width * 5

            alpha_envelope = smoothstep(local / 0.15) * (
                1.0 - smoothstep((local - 0.82) / 0.18)
            )
            alpha = int(245 * alpha_envelope)

            if alpha <= 3:
                continue

            glow_draw.line(
                (float(tail[0]), float(tail[1]), float(x), float(y)),
                fill=(color[0], color[1], color[2], int(alpha * 0.28)),
                width=glow_width,
            )

            sharp_draw.line(
                (float(tail[0]), float(tail[1]), float(x), float(y)),
                fill=(color[0], color[1], color[2], alpha),
                width=core_width,
            )

            # Bright object head.
            radius = (
                2.0
                + 3.0 * velocity_norm
                + (
                    3.0 * np.clip(self.diameter[event_index], 0.0, 1.5)
                    if np.isfinite(self.diameter[event_index])
                    else 0.0
                )
            ) * zoom

            sharp_draw.ellipse(
                (
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                ),
                fill=(255, 250, 235, alpha),
            )

            self.draw_uncertainty_arc(
                sharp_draw,
                int(event_index),
                earth_x,
                earth_y,
                zoom,
            )

            # Close/highlighted events get target reticles near closest approach.
            is_highlight = event_index in self.highlight_indices
            near_closest = abs(local - 0.5) < 0.12

            if is_highlight and near_closest:
                reticle = 24 + 8 * math.sin(t * 5.0 + self.phases[event_index])

                sharp_draw.arc(
                    (
                        x - reticle,
                        y - reticle,
                        x + reticle,
                        y + reticle,
                    ),
                    10,
                    120,
                    fill=(255, 210, 90, alpha),
                    width=2,
                )
                sharp_draw.arc(
                    (
                        x - reticle,
                        y - reticle,
                        x + reticle,
                        y + reticle,
                    ),
                    190,
                    300,
                    fill=(255, 210, 90, alpha),
                    width=2,
                )

                sharp_draw.line(
                    (x - reticle - 10, y, x - reticle + 4, y),
                    fill=(255, 210, 90, alpha),
                    width=2,
                )
                sharp_draw.line(
                    (x + reticle - 4, y, x + reticle + 10, y),
                    fill=(255, 210, 90, alpha),
                    width=2,
                )

        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(12))
        canvas.alpha_composite(glow_layer)
        canvas.alpha_composite(sharp_layer)

        self.draw_active_callout(
            canvas,
            active,
            earth_x,
            earth_y,
            zoom,
            t,
        )

    def draw_active_callout(
        self,
        canvas: Image.Image,
        active: np.ndarray,
        earth_x: float,
        earth_y: float,
        zoom: float,
        t: float,
    ):
        if len(active) == 0:
            return

        # Prefer the currently active event with the smallest catalog distance.
        event_index = int(active[np.argmin(self.distance_ld[active])])

        (
            position,
            direction,
            normal,
            local,
            streak_length,
            closest_radius,
        ) = self.flyby_geometry(
            event_index,
            t,
            earth_x,
            earth_y,
            zoom,
        )

        if not (0.28 <= local <= 0.72):
            return

        x, y = position
        panel_x = 55 if x > OUT_SIZE[0] / 2 else OUT_SIZE[0] - 500
        panel_y = 245

        alpha = int(
            240
            * smoothstep((local - 0.28) / 0.10)
            * (1.0 - smoothstep((local - 0.62) / 0.10))
        )

        if alpha <= 4:
            return

        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)

        draw.rounded_rectangle(
            (
                panel_x,
                panel_y,
                panel_x + 445,
                panel_y + 205,
            ),
            radius=20,
            fill=(2, 7, 16, 176),
            outline=(255, 170, 70, 115),
            width=2,
        )

        canvas.alpha_composite(panel)

        name = str(self.full_names[event_index]).strip() or str(
            self.designations[event_index]
        )

        draw_text(
            canvas,
            "CLOSE APPROACH LOCK",
            (panel_x + 22, panel_y + 18),
            size=20,
            fill=(255, 190, 90, alpha),
            bold=True,
            stroke=1,
        )

        draw_text(
            canvas,
            name[:31],
            (panel_x + 22, panel_y + 52),
            size=27,
            fill=(245, 250, 255, alpha),
            bold=True,
            stroke=2,
        )

        draw_text(
            canvas,
            format_distance_ld(self.distance_ld[event_index]),
            (panel_x + 22, panel_y + 100),
            size=26,
            fill=(110, 235, 255, alpha),
            bold=True,
            stroke=1,
        )

        draw_text(
            canvas,
            format_velocity(self.velocity[event_index]),
            (panel_x + 225, panel_y + 100),
            size=26,
            fill=(110, 235, 255, alpha),
            bold=True,
            stroke=1,
        )

        diameter_text = format_diameter(self.diameter[event_index])
        date_text = self.events.iloc[event_index]["close_time"].strftime(
            "%Y-%m-%d"
        )

        draw_text(
            canvas,
            diameter_text,
            (panel_x + 22, panel_y + 145),
            size=20,
            fill=(215, 228, 242, alpha),
            stroke=1,
        )

        draw_text(
            canvas,
            f"catalog date // {date_text}",
            (panel_x + 22, panel_y + 174),
            size=18,
            fill=(175, 205, 230, alpha),
            stroke=1,
        )

    def draw_velocity_spectrum(
        self,
        canvas: Image.Image,
        t: float,
    ):
        alpha = int(
            225
            * smoothstep((t - 27.0) / 5.0)
            * (1.0 - smoothstep((t - 44.0) / 5.0))
        )

        if alpha <= 4:
            return

        x0 = 63
        y0 = OUT_SIZE[1] - 520
        width = OUT_SIZE[0] - 126
        height = 128

        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)

        draw.rounded_rectangle(
            (
                x0 - 20,
                y0 - 58,
                x0 + width + 20,
                y0 + height + 45,
            ),
            radius=24,
            fill=(1, 5, 14, 158),
            outline=(65, 180, 220, 80),
            width=1,
        )

        canvas.alpha_composite(panel)

        hist, edges = np.histogram(
            self.velocity,
            bins=44,
            range=(0, max(self.max_velocity, 1.0)),
        )

        hist = hist.astype(float)
        hist = hist / max(hist.max(), 1.0)

        d = ImageDraw.Draw(canvas)
        bar_width = width / len(hist)

        for index, value in enumerate(hist):
            x = x0 + index * bar_width
            bar_height = value * height

            d.rectangle(
                (
                    x,
                    y0 + height - bar_height,
                    x + max(1, bar_width - 1),
                    y0 + height,
                ),
                fill=(75, 220, 255, int(alpha * 0.72)),
            )

        draw_text(
            canvas,
            "EARTH-RELATIVE VELOCITY SPECTRUM",
            (x0, y0 - 39),
            size=21,
            fill=(180, 230, 250, alpha),
            bold=True,
            stroke=1,
        )

        draw_text(
            canvas,
            "0 km/s",
            (x0, y0 + height + 15),
            size=17,
            fill=(170, 205, 225, alpha),
            stroke=1,
        )

        draw_text(
            canvas,
            f"{self.max_velocity:.0f} km/s",
            (x0 + width, y0 + height + 15),
            size=17,
            fill=(170, 205, 225, alpha),
            stroke=1,
            anchor="ra",
        )

    def draw_closest_board(
        self,
        canvas: Image.Image,
        t: float,
    ):
        alpha = int(235 * smoothstep((t - 39.0) / 4.0))

        if alpha <= 4:
            return

        count = int(CONFIG["leaderboard_count"])
        indices = self.closest_indices[:count]

        x0 = 55
        y0 = 255
        width = 495
        row_height = 52

        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)

        draw.rounded_rectangle(
            (
                x0 - 20,
                y0 - 54,
                x0 + width,
                y0 + row_height * count + 30,
            ),
            radius=22,
            fill=(2, 6, 15, 172),
            outline=(255, 155, 70, 95),
            width=1,
        )

        canvas.alpha_composite(panel)

        draw_text(
            canvas,
            "CLOSEST CATALOG PASSES",
            (x0, y0 - 38),
            size=22,
            fill=(255, 195, 100, alpha),
            bold=True,
            stroke=1,
        )

        for rank, event_index in enumerate(indices, start=1):
            y = y0 + (rank - 1) * row_height
            name = (
                str(self.full_names[event_index]).strip()
                or str(self.designations[event_index])
            )

            draw_text(
                canvas,
                f"{rank:02d}",
                (x0, y + 8),
                size=20,
                fill=(255, 165, 80, alpha),
                bold=True,
                stroke=1,
            )

            draw_text(
                canvas,
                name[:23],
                (x0 + 48, y + 8),
                size=20,
                fill=(235, 245, 255, alpha),
                bold=True,
                stroke=1,
            )

            draw_text(
                canvas,
                format_distance_ld(self.distance_ld[event_index]),
                (x0 + 470, y + 8),
                size=19,
                fill=(110, 230, 255, alpha),
                bold=True,
                stroke=1,
                anchor="ra",
            )

    def draw_timeline(
        self,
        canvas: Image.Image,
        t: float,
    ):
        alpha = 220
        x0 = 75
        x1 = OUT_SIZE[0] - 75
        y = OUT_SIZE[1] - 320
        width = x1 - x0

        d = ImageDraw.Draw(canvas)

        d.line(
            (x0, y, x1, y),
            fill=(90, 195, 225, alpha),
            width=2,
        )

        # Event barcode.
        if len(self.event_times) > 0:
            sample_count = min(300, len(self.event_times))
            sample_indices = np.linspace(
                0,
                len(self.event_times) - 1,
                sample_count,
                dtype=int,
            )

            for event_index in sample_indices:
                x = x0 + self.event_times[event_index] * width
                height = 8 + 16 * np.clip(
                    self.velocity[event_index] / self.max_velocity,
                    0.0,
                    1.0,
                )

                d.line(
                    (x, y - height, x, y + height),
                    fill=(75, 210, 240, 75),
                    width=1,
                )

        fraction = self.catalog_fraction(t)
        cursor_x = x0 + fraction * width

        d.line(
            (cursor_x, y - 35, cursor_x, y + 35),
            fill=(255, 180, 85, 245),
            width=3,
        )

        catalog_time = self.catalog_time(t)

        draw_text(
            canvas,
            "CLOSE-APPROACH CATALOG TIME",
            (x0, y - 66),
            size=20,
            fill=(165, 220, 240, 220),
            bold=True,
            stroke=1,
        )

        draw_text(
            canvas,
            catalog_time.strftime("%Y-%m-%d"),
            (x1, y - 66),
            size=30,
            fill=(245, 250, 255, 245),
            bold=True,
            stroke=2,
            anchor="ra",
        )

        draw_text(
            canvas,
            CONFIG["catalog_start_date"][:4],
            (x0, y + 28),
            size=17,
            fill=(165, 200, 220, 210),
            stroke=1,
        )

        draw_text(
            canvas,
            CONFIG["catalog_end_date"][:4],
            (x1, y + 28),
            size=17,
            fill=(165, 200, 220, 210),
            stroke=1,
            anchor="ra",
        )

    def draw_corner_hud(
        self,
        canvas: Image.Image,
        t: float,
        active_count: int,
    ):
        draw_text(
            canvas,
            f"TRACKS // {active_count:02d}",
            (OUT_SIZE[0] - 55, 83),
            size=22,
            fill=(105, 230, 245, 220),
            bold=True,
            stroke=1,
            anchor="ra",
        )

        draw_text(
            canvas,
            f"GATE // {CONFIG['dist_max_lunar']:.0f} LD",
            (OUT_SIZE[0] - 55, 118),
            size=20,
            fill=(150, 205, 225, 205),
            bold=True,
            stroke=1,
            anchor="ra",
        )

        sweep = int((t * 17.0) % 100)

        draw_text(
            canvas,
            f"SCAN // {sweep:02d}",
            (OUT_SIZE[0] - 55, 151),
            size=18,
            fill=(150, 205, 225, 190),
            stroke=1,
            anchor="ra",
        )

    def add_text_layers(
        self,
        canvas: Image.Image,
        t: float,
        shot: Dict,
    ):
        title_alpha = int(
            255
            * smoothstep((t - 0.25) / 1.0)
            * (1.0 - smoothstep((t - 5.3) / 0.9))
        )

        if title_alpha > 4:
            draw_text(
                canvas,
                CONFIG["title_text"],
                (54, 102),
                size=50,
                fill=(245, 250, 255, title_alpha),
                bold=True,
            )

            draw_text(
                canvas,
                CONFIG["subtitle_text"],
                (58, 170),
                size=25,
                fill=(100, 230, 245, min(title_alpha, 225)),
                bold=True,
            )

        if t > 5.4:
            draw_text(
                canvas,
                shot["caption"],
                (54, 65),
                size=21,
                fill=(140, 215, 235, 205),
                bold=True,
                stroke=1,
            )

        caption = caption_at(t)

        if caption:
            panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            draw = ImageDraw.Draw(panel)
            y0 = OUT_SIZE[1] - 246

            draw.rounded_rectangle(
                (
                    45,
                    y0,
                    OUT_SIZE[0] - 45,
                    y0 + 126,
                ),
                radius=24,
                fill=(1, 4, 12, 170),
                outline=(55, 180, 220, 70),
                width=1,
            )

            canvas.alpha_composite(panel)

            draw_wrapped_text(
                canvas,
                caption,
                (70, y0 + 27),
                max_width=OUT_SIZE[0] - 140,
                size=30,
                fill=(245, 250, 255, 245),
            )

        note_alpha = int(220 * smoothstep((t - 50.5) / 3.0))

        if note_alpha > 4:
            draw_wrapped_text(
                canvas,
                CONFIG["credit_text"],
                (65, OUT_SIZE[1] - 111),
                max_width=940,
                size=19,
                fill=(220, 232, 245, note_alpha),
            )

            draw_wrapped_text(
                canvas,
                CONFIG["scientific_note"],
                (65, OUT_SIZE[1] - 79),
                max_width=940,
                size=17,
                fill=(190, 210, 232, note_alpha),
            )

    def draw_hud_noise(self, canvas: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for item in self.hud_noise:
            pulse = 0.5 + 0.5 * math.sin(
                t * 1.9 + item["phase"]
            )

            if pulse < 0.72:
                continue

            x = item["x"]
            y = (item["y"] + t * 12.0) % OUT_SIZE[1]

            draw.line(
                (x, y, x + item["length"], y),
                fill=(90, 210, 240, int(item["alpha"] * pulse)),
                width=1,
            )

        offset = int((t * 41) % 7)

        for y in range(offset, OUT_SIZE[1], 7):
            draw.line(
                (0, y, OUT_SIZE[0], y),
                fill=(120, 205, 245, 13),
                width=1,
            )

        scan_y = int((t * 158) % (OUT_SIZE[1] + 260)) - 130

        draw.rectangle(
            (0, scan_y, OUT_SIZE[0], scan_y + 54),
            fill=(80, 210, 240, 9),
        )

        canvas.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot, earth_x, earth_y, zoom = shot_state(t)
        active = self.active_event_indices(t)

        canvas = self.render_background(t)
        self.draw_distance_gates(
            canvas,
            earth_x,
            earth_y,
            zoom,
            t,
        )
        self.draw_earth(
            canvas,
            earth_x,
            earth_y,
            zoom,
            t,
        )
        self.draw_flybys(
            canvas,
            earth_x,
            earth_y,
            zoom,
            t,
        )
        self.draw_velocity_spectrum(canvas, t)
        self.draw_closest_board(canvas, t)
        self.draw_timeline(canvas, t)
        self.draw_corner_hud(canvas, t, len(active))
        self.add_text_layers(canvas, t, shot)
        self.draw_hud_noise(canvas, t)

        arr = np.array(canvas.convert("RGB"))
        arr = apply_grade(arr)

        arr = np.clip(
            arr.astype(np.float32) * VIGNETTE[..., None],
            0,
            255,
        ).astype(np.uint8)

        fade_in = smoothstep(t / 0.9)
        fade_out = 1.0 - smoothstep(
            (t - (CONFIG["duration_s"] - 1.1)) / 1.0
        )

        arr = np.clip(
            arr.astype(np.float32) * fade_in * fade_out,
            0,
            255,
        ).astype(np.uint8)

        return arr


# %% [markdown]
# ## Scientific preview plots

def create_scientific_previews(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        df["distance_ld"],
        df["velocity_km_s"],
        s=12,
        alpha=0.45,
    )
    ax.set_title("JPL close approaches: distance vs relative velocity")
    ax.set_xlabel("Nominal approach distance [lunar distances]")
    ax.set_ylabel("Earth-relative velocity [km/s]")
    plt.tight_layout()
    plt.savefig(
        PREVIEW_DIR / "distance_vs_velocity.png",
        dpi=170,
    )
    plt.close(fig)

    dates = df["close_time"].dt.tz_convert(None)
    monthly_counts = (
        pd.Series(np.ones(len(df), dtype=int), index=dates)
        .resample("MS")
        .sum()
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    if len(monthly_counts):
        labels = monthly_counts.index.strftime("%Y-%m")
        x_positions = np.arange(len(monthly_counts))

        ax.bar(
            x_positions,
            monthly_counts.to_numpy(),
        )

        # Keep the preview readable even for many months.
        tick_step = max(1, len(monthly_counts) // 12)
        tick_positions = x_positions[::tick_step]
        tick_labels = labels[::tick_step]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            tick_labels,
            rotation=45,
            ha="right",
        )

    ax.set_title("Cataloged close approaches by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Close-approach count")
    plt.tight_layout()
    plt.savefig(
        PREVIEW_DIR / "close_approaches_by_month.png",
        dpi=170,
    )
    plt.close(fig)

    closest = df.nsmallest(15, "distance_ld").copy()
    closest = closest.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    y_positions = np.arange(len(closest))
    ax.barh(
        y_positions,
        closest["distance_ld"].to_numpy(),
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        closest["full_name"].astype(str).str.strip().str.slice(0, 28)
    )
    ax.set_title("Closest nominal approaches in selected catalog window")
    ax.set_xlabel("Nominal approach distance [lunar distances]")
    plt.tight_layout()
    plt.savefig(
        PREVIEW_DIR / "closest_approaches.png",
        dpi=170,
    )
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
        lines.append(
            f"{format_srt_time(start)} --> {format_srt_time(end)}"
        )
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


def render_video(scene: PlanetaryDefenseScene):
    raw_video_path = (
        OUTPUT_ROOT
        / f"{CONFIG['output_basename']}_raw.mp4"
    )
    subbed_video_path = (
        OUTPUT_ROOT
        / f"{CONFIG['output_basename']}_subbed.mp4"
    )
    audio_video_path = (
        OUTPUT_ROOT
        / f"{CONFIG['output_basename']}_with_audio.mp4"
    )
    final_video_path = (
        OUTPUT_ROOT
        / f"{CONFIG['output_basename']}_final.mp4"
    )
    srt_path = (
        OUTPUT_ROOT
        / f"{CONFIG['output_basename']}.srt"
    )

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
        for t in tqdm(times, desc="Rendering planetary defense radar"):
            writer.append_data(scene.render_frame(float(t)))

    print("Raw video written:", raw_video_path.resolve())

    ffmpeg = find_ffmpeg()
    print("FFmpeg detected:", ffmpeg)

    final_candidate = raw_video_path

    if (
        CONFIG.get("burn_subtitles", False)
        and ffmpeg
        and srt_path.exists()
    ):
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(final_candidate),
            "-vf",
            (
                f"subtitles={srt_path}:"
                "force_style=Fontname=DejaVu Sans,"
                "Fontsize=22,Outline=1.2,BorderStyle=3,MarginV=90"
            ),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
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
            "-i",
            str(final_candidate),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(audio_video_path),
        ]

        run_ffmpeg(command)
        final_candidate = audio_video_path
        print("Audio-muxed video written:", audio_video_path.resolve())
    elif audio_path:
        print(
            "audio_path was set, but the file was not found "
            "or FFmpeg was unavailable. Skipping audio."
        )

    if final_candidate.exists():
        shutil.copyfile(final_candidate, final_video_path)
        print("Final video:", final_video_path.resolve())

    return final_video_path


# %% [markdown]
# ## Main pipeline

def main():
    print("Starting Planetary Defense Radar pipeline ...")

    event_df = fetch_close_approaches(
        CONFIG,
        force_refresh=False,
    )

    video_events = select_video_events(
        event_df,
        CONFIG,
    )

    print(f"Video events selected: {len(video_events):,}")
    print(
        "Catalog playback:",
        video_events["close_time"].min(),
        "to",
        video_events["close_time"].max(),
    )

    create_scientific_previews(event_df)

    scene = PlanetaryDefenseScene(video_events)

    preview_times = [
        1.0,
        10.0,
        23.0,
        36.0,
        46.0,
        CONFIG["duration_s"] - 1.0,
    ]

    for preview_time in tqdm(preview_times, desc="Preview frames"):
        frame = scene.render_frame(float(preview_time))
        Image.fromarray(frame).save(
            PREVIEW_DIR
            / f"preview_{int(preview_time):02d}s.png"
        )

    print("Preview frames written to:", PREVIEW_DIR.resolve())

    render_video(scene)

    print("Output directory:", OUTPUT_ROOT.resolve())

    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-", path.name)


if __name__ == "__main__":
    main()


# %% [markdown]
# ## Suggested narration
#
# Every streak is a near-Earth object recorded in JPL's close-approach catalog.
# The display replays five years of encounters in catalog-time order.
# Each ring is a lunar-distance gate around Earth.
# A closer pass cuts deeper into the radar field.
# Faster objects leave longer tracks.
# The direction on screen is cinematic, but the distance, date, and velocity are real.
# In astronomy, "close" can still mean hundreds of thousands of kilometers.
# Our orbital neighborhood is not empty.
# It is continuously measured.
#
# Suggested YouTube Shorts caption:
#
# Five years of near-Earth object flybys turned into a planetary-defense radar.
# Close-approach time, nominal distance, uncertainty bounds, and Earth-relative
# velocity come from the NASA/JPL SBDB Close-Approach Data API.
#
# #Astronomy #Asteroids #PlanetaryDefense #NASA #JPL #SpaceData #Python
