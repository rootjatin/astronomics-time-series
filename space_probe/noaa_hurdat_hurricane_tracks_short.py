# %% [markdown]
# # Cinematic YouTube Short from Real NOAA/NHC Hurricane Data
#
# Theme: 175 YEARS OF ATLANTIC HURRICANES
#
# Data source:
# NOAA / National Hurricane Center Atlantic HURDAT2 best-track database.
#
# The script:
# - discovers the newest Atlantic HURDAT2 text file from the official NHC archive;
# - downloads and caches the catalog;
# - parses real six-hourly storm positions, status, maximum sustained wind,
#   minimum central pressure and track metadata;
# - creates scientific preview plots;
# - renders a 1080x1920 vertical Short;
# - replays storm tracks chronologically from 1851 through the newest catalog year;
# - uses real wind speed to drive line width / glow;
# - uses Saffir-Simpson wind thresholds only as a display classification;
# - labels selected high-intensity storms from the actual catalog;
# - writes an SRT sidecar;
# - exports raw and final MP4 files.
#
# Scientific fidelity:
# Track positions, dates, storm names, status, maximum sustained winds and
# minimum central pressures are HURDAT2 values.
#
# Camera movement, glow, fading trails, color mapping and text are cinematic
# encodings for science communication. The animation is not a forecast.

from __future__ import annotations

import math
import re
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

plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["axes.grid"] = True

print("Imports loaded.")

# %% [markdown]
# ## Configuration

OUTPUT_ROOT = Path("atlantic_hurricane_history_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"

for p in [OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR]:
    p.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "video_width": 1080,
    "video_height": 1920,
    "fps": 24,
    "duration_s": 58,
    "output_basename": "noaa_atlantic_hurricane_history_short",

    "nhc_hurdat_index": "https://www.nhc.noaa.gov/data/hurdat/",
    "request_timeout": (20, 180),
    "request_retries": 6,
    "retry_backoff_s": 1.8,

    # Visual selection.
    "min_track_wind_kt": 34,
    "max_storms_for_video": 1700,
    "highlight_storm_count": 16,
    "trail_lifetime_s": 5.5,

    # Rendering.
    "vignette_strength": 0.30,
    "contrast_boost": 1.12,
    "saturation_boost": 1.08,

    "title_text": "175 YEARS OF HURRICANES",
    "subtitle_text": "Atlantic storm tracks replayed from NOAA best-track data",
    "credit_text": "Best-track data: NOAA / National Hurricane Center HURDAT2",
    "scientific_note": (
        "Positions and maximum winds are catalog values. "
        "Glow, trail fading and camera motion are cinematic encodings."
    ),

    "audio_path": None,
    "burn_subtitles": False,
    "write_subtitle_sidecar": True,
}

print("Configuration ready.")

# %% [markdown]
# ## Download the newest official Atlantic HURDAT2 file

def build_retry_session(config: Dict) -> requests.Session:
    retries = Retry(
        total=int(config["request_retries"]),
        connect=int(config["request_retries"]),
        read=int(config["request_retries"]),
        status=int(config["request_retries"]),
        backoff_factor=float(config["retry_backoff_s"]),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retries, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "atlantic-hurricane-history-short/1.0 educational visualization"
    })
    return session


def discover_latest_atlantic_hurdat2(config: Dict) -> str:
    session = build_retry_session(config)
    response = session.get(
        config["nhc_hurdat_index"],
        timeout=config["request_timeout"],
    )
    response.raise_for_status()

    names = re.findall(
        r'href="(hurdat2-1851-[^"]+?\.txt)"',
        response.text,
        flags=re.IGNORECASE,
    )

    names = sorted(set(names))
    if not names:
        raise RuntimeError(
            "Could not discover an Atlantic HURDAT2 text file from the NHC index."
        )

    # NHC filenames include the ending catalog year and release date.
    def sort_key(name: str):
        nums = [int(v) for v in re.findall(r"\d+", name)]
        return tuple(nums)

    latest_name = max(names, key=sort_key)
    return config["nhc_hurdat_index"] + latest_name


def download_hurdat2(config: Dict, force_refresh: bool = False) -> Path:
    url = discover_latest_atlantic_hurdat2(config)
    filename = url.rsplit("/", 1)[-1]
    local_path = DATA_ROOT / filename

    if local_path.exists() and local_path.stat().st_size > 1_000_000 and not force_refresh:
        print("Using cached HURDAT2:", local_path)
        return local_path

    print("Official HURDAT2 URL:")
    print(url)

    response = build_retry_session(config).get(
        url,
        timeout=config["request_timeout"],
    )
    response.raise_for_status()
    local_path.write_bytes(response.content)

    if local_path.stat().st_size < 1_000_000:
        raise RuntimeError("Downloaded HURDAT2 file is unexpectedly small.")

    print("Saved:", local_path)
    return local_path


HURDAT_PATH = download_hurdat2(CONFIG, force_refresh=False)

# %% [markdown]
# ## Parse HURDAT2

def parse_lat_lon(token: str) -> float:
    token = token.strip().upper()
    if not token:
        return np.nan
    value = float(token[:-1])
    hemisphere = token[-1]
    if hemisphere in ("S", "W"):
        value *= -1
    return value


def parse_int(token: str) -> float:
    token = token.strip()
    if not token:
        return np.nan
    value = int(token)
    return np.nan if value == -999 else float(value)


def parse_hurdat2(path: Path) -> pd.DataFrame:
    rows: List[Dict] = []

    current_id = ""
    current_name = ""
    current_count = 0
    current_index = 0

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    for line in lines:
        parts = [p.strip() for p in line.split(",")]

        # Header:
        # AL011851, UNNAMED, 14,
        if (
            len(parts) >= 3
            and re.fullmatch(r"[A-Z]{2}\d{6}", parts[0] or "")
        ):
            current_id = parts[0]
            current_name = parts[1] or "UNNAMED"
            current_count = int(parts[2])
            current_index = 0
            continue

        if not current_id or len(parts) < 8:
            continue

        date_text = parts[0]
        time_text = parts[1].zfill(4)
        record_id = parts[2]
        status = parts[3]

        timestamp = pd.to_datetime(
            date_text + time_text,
            format="%Y%m%d%H%M",
            utc=True,
            errors="coerce",
        )

        rows.append({
            "storm_id": current_id,
            "storm_name": current_name,
            "catalog_point_index": current_index,
            "catalog_point_count": current_count,
            "timestamp": timestamp,
            "record_id": record_id,
            "status": status,
            "latitude": parse_lat_lon(parts[4]),
            "longitude": parse_lat_lon(parts[5]),
            "max_wind_kt": parse_int(parts[6]),
            "min_pressure_mb": parse_int(parts[7]),
        })

        current_index += 1

    df = pd.DataFrame(rows)

    numeric_cols = [
        "catalog_point_index",
        "catalog_point_count",
        "latitude",
        "longitude",
        "max_wind_kt",
        "min_pressure_mb",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["timestamp", "latitude", "longitude", "max_wind_kt"]
    ).copy()

    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ].copy()

    df["year"] = df["timestamp"].dt.year
    df["date_label"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df = df.sort_values(
        ["timestamp", "storm_id", "catalog_point_index"]
    ).reset_index(drop=True)

    return df


track_df = parse_hurdat2(HURDAT_PATH)

print("Track points:", f"{len(track_df):,}")
print("Storms:", f"{track_df['storm_id'].nunique():,}")
print("Years:", int(track_df["year"].min()), "to", int(track_df["year"].max()))
track_df.head()

# %% [markdown]
# ## Storm-level statistics

def wind_category(wind_kt: float) -> str:
    wind_kt = float(wind_kt)

    if wind_kt < 34:
        return "DEPRESSION"
    if wind_kt < 64:
        return "TROPICAL STORM"
    if wind_kt < 83:
        return "CATEGORY 1"
    if wind_kt < 96:
        return "CATEGORY 2"
    if wind_kt < 113:
        return "CATEGORY 3"
    if wind_kt < 137:
        return "CATEGORY 4"
    return "CATEGORY 5"


storm_df = (
    track_df.groupby(["storm_id", "storm_name"], as_index=False)
    .agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        start_year=("year", "min"),
        max_wind_kt=("max_wind_kt", "max"),
        min_pressure_mb=("min_pressure_mb", "min"),
        point_count=("timestamp", "size"),
    )
)

storm_df["max_category"] = storm_df["max_wind_kt"].apply(wind_category)
storm_df["duration_days"] = (
    storm_df["end_time"] - storm_df["start_time"]
).dt.total_seconds() / 86400.0

print(storm_df["max_category"].value_counts())
print()
print("Strongest storms in catalog:")
print(
    storm_df.nlargest(20, "max_wind_kt")[
        [
            "storm_id",
            "storm_name",
            "start_year",
            "max_wind_kt",
            "min_pressure_mb",
        ]
    ].to_string(index=False)
)

# %% [markdown]
# ## Scientific previews

fig, ax = plt.subplots(figsize=(11, 5))
annual_counts = storm_df.groupby("start_year").size()
ax.plot(annual_counts.index, annual_counts.values, linewidth=1)
ax.set_title("Atlantic named and unnamed tropical cyclones in HURDAT2")
ax.set_xlabel("Year")
ax.set_ylabel("Storm count")
plt.tight_layout()
plt.savefig(PREVIEW_DIR / "storms_per_year.png", dpi=180)
plt.show()

fig, ax = plt.subplots(figsize=(11, 6))
preview_points = track_df[track_df["max_wind_kt"] >= 64]
ax.scatter(
    preview_points["longitude"],
    preview_points["latitude"],
    s=1.3,
    alpha=0.18,
)
ax.set_xlim(-105, 20)
ax.set_ylim(0, 65)
ax.set_title("Atlantic hurricane best-track positions")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig(PREVIEW_DIR / "hurricane_positions.png", dpi=180)
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(storm_df["max_wind_kt"], bins=np.arange(0, 190, 5))
ax.set_title("Maximum sustained wind by Atlantic storm")
ax.set_xlabel("Maximum sustained wind [kt]")
ax.set_ylabel("Storm count")
plt.tight_layout()
plt.savefig(PREVIEW_DIR / "maximum_wind_distribution.png", dpi=180)
plt.show()

# %% [markdown]
# ## Choose storms for the cinematic render

eligible_storms = storm_df[
    storm_df["max_wind_kt"] >= float(CONFIG["min_track_wind_kt"])
].copy()

if len(eligible_storms) > int(CONFIG["max_storms_for_video"]):
    # Keep strongest storms plus a time-stratified selection.
    strongest_n = min(350, int(CONFIG["max_storms_for_video"]) // 3)
    strongest = eligible_storms.nlargest(strongest_n, "max_wind_kt")

    remaining_n = int(CONFIG["max_storms_for_video"]) - len(strongest)
    pool = eligible_storms.drop(index=strongest.index).copy()
    pool["year_bin"] = (pool["start_year"] // 5) * 5

    per_bin = max(
        1,
        math.ceil(
            remaining_n / max(pool["year_bin"].nunique(), 1)
        ),
    )

    sampled = (
        pool.sort_values(
            ["year_bin", "max_wind_kt"],
            ascending=[True, False],
        )
        .groupby("year_bin", group_keys=False)
        .head(per_bin)
        .nlargest(remaining_n, "max_wind_kt")
    )

    selected_storms = pd.concat(
        [strongest, sampled],
        ignore_index=True,
    )
else:
    selected_storms = eligible_storms.copy()

selected_ids = set(selected_storms["storm_id"])

video_tracks = track_df[
    track_df["storm_id"].isin(selected_ids)
    & (track_df["max_wind_kt"] >= float(CONFIG["min_track_wind_kt"]))
].copy()

highlight_storms = (
    storm_df[
        storm_df["max_wind_kt"] >= 100
    ]
    .sort_values(
        ["max_wind_kt", "min_pressure_mb"],
        ascending=[False, True],
    )
    .drop_duplicates(subset=["storm_name", "start_year"])
    .head(int(CONFIG["highlight_storm_count"]))
    .sort_values("start_time")
    .reset_index(drop=True)
)

print("Video storms:", selected_storms["storm_id"].nunique())
print("Video track points:", len(video_tracks))
print("Highlight storms:")
print(
    highlight_storms[
        ["storm_name", "start_year", "max_wind_kt"]
    ].to_string(index=False)
)

# %% [markdown]
# ## Map catalog time into video time

DATA_T0 = track_df["timestamp"].min()
DATA_T1 = track_df["timestamp"].max()
DATA_SPAN_S = max((DATA_T1 - DATA_T0).total_seconds(), 1.0)

VIDEO_DATA_START = 5.5
VIDEO_DATA_END = 51.5
VIDEO_DATA_SPAN = VIDEO_DATA_END - VIDEO_DATA_START


def catalog_to_video_time(series) -> np.ndarray:
    times = pd.to_datetime(pd.Series(series), utc=True)
    seconds = (times - DATA_T0).dt.total_seconds().to_numpy(float)
    normalized = np.clip(seconds / DATA_SPAN_S, 0, 1)
    return VIDEO_DATA_START + normalized * VIDEO_DATA_SPAN


video_tracks["video_t"] = catalog_to_video_time(video_tracks["timestamp"])
highlight_storms["video_t"] = catalog_to_video_time(highlight_storms["start_time"])

# Group tracks once for renderer.
TRACKS = []

for storm_id, group in tqdm(
    video_tracks.groupby("storm_id"),
    desc="Preparing hurricane tracks",
):
    group = group.sort_values("timestamp")

    TRACKS.append({
        "storm_id": storm_id,
        "name": str(group["storm_name"].iloc[0]),
        "year": int(group["year"].iloc[0]),
        "lon": group["longitude"].to_numpy(float),
        "lat": group["latitude"].to_numpy(float),
        "wind": group["max_wind_kt"].to_numpy(float),
        "pressure": group["min_pressure_mb"].to_numpy(float),
        "video_t": group["video_t"].to_numpy(float),
    })

HIGHLIGHTS = highlight_storms.to_dict(orient="records")

# %% [markdown]
# ## Rendering helpers

OUT_SIZE = (
    int(CONFIG["video_width"]),
    int(CONFIG["video_height"]),
)

CATEGORY_COLORS = {
    "TROPICAL STORM": (80, 205, 255, 220),
    "CATEGORY 1": (95, 255, 205, 235),
    "CATEGORY 2": (255, 235, 110, 240),
    "CATEGORY 3": (255, 165, 70, 245),
    "CATEGORY 4": (255, 85, 80, 250),
    "CATEGORY 5": (255, 100, 235, 255),
}


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def ease_in_out_sine(t):
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1) / 2


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
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue

    return ImageFont.load_default()


def draw_text(
    img: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int = 36,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    stroke: int = 2,
    anchor: str = "la",
):
    draw = ImageDraw.Draw(img)
    font = get_font(size, bold=bold)
    draw.text(
        xy,
        text,
        font=font,
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, 220),
    )


def draw_wrapped_text(
    img: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int = 30,
    fill=(255, 255, 255, 245),
    bold: bool = False,
):
    draw = ImageDraw.Draw(img)
    font = get_font(size, bold=bold)

    words = str(text).split()
    lines: List[str] = []
    current = ""

    for word in words:
        candidate = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), candidate, font=font)
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
        bbox = draw.textbbox((x, y), line, font=font)
        y += bbox[3] - bbox[1] + 8


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    rr = np.sqrt(nx**2 + ny**2)

    return np.clip(
        1 - strength * rr**1.65,
        0,
        1,
    ).astype(np.float32)


def apply_grade(arr: np.ndarray) -> np.ndarray:
    img = Image.fromarray(arr)
    img = ImageEnhance.Contrast(img).enhance(
        float(CONFIG["contrast_boost"])
    )
    img = ImageEnhance.Color(img).enhance(
        float(CONFIG["saturation_boost"])
    )
    return np.array(img)


VIGNETTE = make_vignette(
    OUT_SIZE[0],
    OUT_SIZE[1],
    float(CONFIG["vignette_strength"]),
)

# %% [markdown]
# ## Atlantic map projection
#
# Simple equirectangular science-communication map.

LON_MIN = -105.0
LON_MAX = 15.0
LAT_MIN = 0.0
LAT_MAX = 65.0


def map_rect(canvas: Image.Image) -> Tuple[float, float, float, float]:
    return (
        60.0,
        270.0,
        canvas.size[0] - 60.0,
        1420.0,
    )


def project_lon_lat(
    lon: np.ndarray,
    lat: np.ndarray,
    canvas: Image.Image,
    camera_lon_shift: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = map_rect(canvas)

    shifted_lon = np.asarray(lon, dtype=float) + camera_lon_shift
    lat = np.asarray(lat, dtype=float)

    px = x0 + (
        (shifted_lon - LON_MIN)
        / (LON_MAX - LON_MIN)
    ) * (x1 - x0)

    py = y1 - (
        (lat - LAT_MIN)
        / (LAT_MAX - LAT_MIN)
    ) * (y1 - y0)

    return px, py


def render_background(width: int, height: int, t: float) -> Image.Image:
    img = Image.new(
        "RGBA",
        (width, height),
        (2, 8, 18, 255),
    )

    glow = Image.new(
        "RGBA",
        (width, height),
        (0, 0, 0, 0),
    )
    gd = ImageDraw.Draw(glow)

    cx = width * (
        0.58 + 0.03 * math.sin(t * 0.08)
    )
    cy = height * 0.46

    for radius, alpha in [
        (700, 12),
        (500, 20),
        (300, 25),
    ]:
        gd.ellipse(
            (
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
            ),
            fill=(20, 105, 170, alpha),
        )

    glow = glow.filter(
        ImageFilter.GaussianBlur(80)
    )
    img.alpha_composite(glow)

    return img


def draw_map_grid(canvas: Image.Image, t: float):
    overlay = Image.new(
        "RGBA",
        canvas.size,
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(overlay)

    x0, y0, x1, y1 = map_rect(canvas)

    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=28,
        fill=(1, 14, 30, 165),
        outline=(100, 190, 255, 100),
        width=2,
    )

    for lon in range(-100, 11, 10):
        px, _ = project_lon_lat(
            np.array([lon]),
            np.array([0]),
            canvas,
        )
        x = float(px[0])
        draw.line(
            (x, y0, x, y1),
            fill=(100, 190, 255, 30),
            width=1,
        )

    for lat in range(10, 61, 10):
        _, py = project_lon_lat(
            np.array([LON_MIN]),
            np.array([lat]),
            canvas,
        )
        y = float(py[0])
        draw.line(
            (x0, y, x1, y),
            fill=(100, 190, 255, 30),
            width=1,
        )

    # Moving scan line.
    scan_x = x0 + (
        (t * 0.105) % 1.0
    ) * (x1 - x0)

    draw.line(
        (scan_x, y0, scan_x, y1),
        fill=(150, 225, 255, 60),
        width=2,
    )

    canvas.alpha_composite(overlay)

# %% [markdown]
# ## Draw real hurricane tracks

def active_track_segment(track: Dict, t: float):
    times = track["video_t"]

    reached = np.where(times <= t)[0]
    if len(reached) == 0:
        return None

    latest = int(reached[-1])

    age = t - times[: latest + 1]
    keep = age <= float(CONFIG["trail_lifetime_s"])

    indices = np.where(keep)[0]
    if len(indices) == 0:
        return None

    return indices


def draw_hurricane_tracks(canvas: Image.Image, t: float):
    glow_layer = Image.new(
        "RGBA",
        canvas.size,
        (0, 0, 0, 0),
    )
    line_layer = Image.new(
        "RGBA",
        canvas.size,
        (0, 0, 0, 0),
    )

    gd = ImageDraw.Draw(glow_layer)
    ld = ImageDraw.Draw(line_layer)

    for track in TRACKS:
        indices = active_track_segment(track, t)
        if indices is None or len(indices) < 2:
            continue

        lon = track["lon"][indices]
        lat = track["lat"][indices]
        wind = track["wind"][indices]
        times = track["video_t"][indices]

        px, py = project_lon_lat(
            lon,
            lat,
            canvas,
        )

        for j in range(1, len(indices)):
            age = t - times[j]
            fade = (
                1.0
                - clamp(
                    age / float(CONFIG["trail_lifetime_s"])
                )
            ) ** 1.25

            category = wind_category(wind[j])
            color = CATEGORY_COLORS.get(
                category,
                CATEGORY_COLORS["TROPICAL STORM"],
            )

            wind_gain = np.clip(
                (wind[j] - 34.0) / 125.0,
                0.0,
                1.0,
            )

            alpha = int(
                color[3] * fade
            )
            width = max(
                1,
                int(2 + 5 * wind_gain),
            )
            glow_width = width + 12

            p0 = (
                float(px[j - 1]),
                float(py[j - 1]),
            )
            p1 = (
                float(px[j]),
                float(py[j]),
            )

            gd.line(
                (p0, p1),
                fill=(
                    color[0],
                    color[1],
                    color[2],
                    max(1, alpha // 4),
                ),
                width=glow_width,
            )

            ld.line(
                (p0, p1),
                fill=(
                    color[0],
                    color[1],
                    color[2],
                    alpha,
                ),
                width=width,
            )

        # Current observed track point.
        j = len(indices) - 1
        x = float(px[j])
        y = float(py[j])
        current_wind = float(wind[j])

        radius = 3.0 + np.clip(
            (current_wind - 34.0) / 125.0,
            0,
            1,
        ) * 9.0

        category = wind_category(current_wind)
        color = CATEGORY_COLORS.get(
            category,
            CATEGORY_COLORS["TROPICAL STORM"],
        )

        ld.ellipse(
            (
                x - radius,
                y - radius,
                x + radius,
                y + radius,
            ),
            fill=color,
        )

    glow_layer = glow_layer.filter(
        ImageFilter.GaussianBlur(7)
    )

    canvas.alpha_composite(glow_layer)
    canvas.alpha_composite(line_layer)

# %% [markdown]
# ## Category legend and historical timeline

CAPTIONS = [
    (
        0.5,
        5.0,
        "This is the Atlantic hurricane record, compressed into less than a minute.",
    ),
    (
        5.8,
        16.0,
        "Every moving trail follows real best-track positions from the hurricane archive.",
    ),
    (
        16.3,
        27.0,
        "Color changes as maximum sustained wind crosses hurricane intensity thresholds.",
    ),
    (
        27.3,
        39.0,
        "The archive reaches back to 1851, long before weather satellites.",
    ),
    (
        39.3,
        50.5,
        "The strongest storms become landmarks in more than a century of Atlantic history.",
    ),
    (
        51.0,
        57.3,
        "A hurricane season disappears. The historical track remains in the data.",
    ),
]


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


def current_catalog_date(t: float) -> pd.Timestamp:
    progress = np.clip(
        (t - VIDEO_DATA_START)
        / VIDEO_DATA_SPAN,
        0,
        1,
    )

    return DATA_T0 + pd.to_timedelta(
        progress * DATA_SPAN_S,
        unit="s",
    )


def draw_timeline(canvas: Image.Image, t: float):
    x0 = 70
    x1 = canvas.size[0] - 70
    y = 1545

    overlay = Image.new(
        "RGBA",
        canvas.size,
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(overlay)

    draw.line(
        (x0, y, x1, y),
        fill=(140, 215, 255, 100),
        width=2,
    )

    for year in range(
        int(DATA_T0.year // 25 * 25 + 25),
        int(DATA_T1.year) + 1,
        25,
    ):
        frac = (
            pd.Timestamp(
                f"{year}-01-01",
                tz="UTC",
            )
            - DATA_T0
        ).total_seconds() / DATA_SPAN_S

        x = x0 + frac * (x1 - x0)

        draw.line(
            (x, y - 10, x, y + 10),
            fill=(140, 215, 255, 100),
            width=1,
        )

        draw_text(
            overlay,
            str(year),
            (int(x), y + 22),
            size=16,
            fill=(180, 225, 250, 190),
            stroke=1,
            anchor="ma",
        )

    progress = np.clip(
        (t - VIDEO_DATA_START)
        / VIDEO_DATA_SPAN,
        0,
        1,
    )

    cursor_x = x0 + progress * (x1 - x0)

    draw.line(
        (cursor_x, y - 28, cursor_x, y + 20),
        fill=(255, 255, 255, 240),
        width=3,
    )

    catalog_date = current_catalog_date(t)

    draw_text(
        overlay,
        str(catalog_date.year),
        (int(cursor_x), y - 48),
        size=30,
        fill=(255, 255, 255, 245),
        bold=True,
        anchor="ma",
    )

    canvas.alpha_composite(overlay)


def draw_category_legend(canvas: Image.Image, t: float):
    alpha = int(
        220
        * smoothstep((t - 12.0) / 3.0)
        * (
            1
            - smoothstep((t - 43.0) / 5.0)
        )
    )

    if alpha <= 5:
        return

    overlay = Image.new(
        "RGBA",
        canvas.size,
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(overlay)

    x = 70
    y = 380

    draw_text(
        overlay,
        "MAXIMUM WIND",
        (x, y),
        size=21,
        fill=(180, 230, 255, alpha),
        bold=True,
        stroke=1,
    )

    items = [
        ("TS", "TROPICAL STORM"),
        ("C1", "CATEGORY 1"),
        ("C2", "CATEGORY 2"),
        ("C3", "CATEGORY 3"),
        ("C4", "CATEGORY 4"),
        ("C5", "CATEGORY 5"),
    ]

    y += 42

    for short, category in items:
        color = CATEGORY_COLORS[category]

        draw.line(
            (x, y + 9, x + 35, y + 9),
            fill=(
                color[0],
                color[1],
                color[2],
                alpha,
            ),
            width=5,
        )

        draw_text(
            overlay,
            short,
            (x + 48, y - 3),
            size=18,
            fill=(220, 240, 255, alpha),
            bold=True,
            stroke=1,
        )

        y += 33

    canvas.alpha_composite(overlay)

# %% [markdown]
# ## Highlight strong historical storms

def highlight_at(t: float) -> Optional[Dict]:
    candidates = []

    for item in HIGHLIGHTS:
        age = t - float(item["video_t"])

        if -0.15 <= age <= 2.8:
            candidates.append(
                (abs(age - 0.6), item)
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda pair: pair[0]
    )

    return candidates[0][1]


def draw_highlight(canvas: Image.Image, t: float):
    item = highlight_at(t)

    if item is None:
        return

    age = t - float(item["video_t"])

    fade = (
        smoothstep((age + 0.15) / 0.45)
        * (
            1
            - smoothstep((age - 1.9) / 0.75)
        )
    )

    alpha = int(245 * fade)

    if alpha <= 5:
        return

    overlay = Image.new(
        "RGBA",
        canvas.size,
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(overlay)

    x0, y0 = 70, 190
    x1, y1 = 815, 350

    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=26,
        fill=(0, 7, 18, int(190 * fade)),
        outline=(160, 220, 255, int(90 * fade)),
        width=1,
    )

    draw_text(
        overlay,
        "HISTORICAL STORM",
        (x0 + 24, y0 + 20),
        size=20,
        fill=(170, 225, 255, alpha),
        bold=True,
        stroke=1,
    )

    draw_text(
        overlay,
        str(item["storm_name"]),
        (x0 + 24, y0 + 54),
        size=40,
        fill=(255, 255, 255, alpha),
        bold=True,
    )

    draw_text(
        overlay,
        str(int(item["start_year"])),
        (x1 - 24, y0 + 30),
        size=28,
        fill=(220, 240, 255, alpha),
        bold=True,
        anchor="ra",
    )

    category = wind_category(
        float(item["max_wind_kt"])
    )

    draw_text(
        overlay,
        f"{category}  //  {float(item['max_wind_kt']):.0f} KT",
        (x0 + 26, y0 + 112),
        size=21,
        fill=(255, 220, 190, alpha),
        bold=True,
        stroke=1,
    )

    canvas.alpha_composite(overlay)

# %% [markdown]
# ## Text layers and frame render

def add_text_layers(canvas: Image.Image, t: float):
    title_alpha = int(
        255
        * smoothstep((t - 0.25) / 1.0)
        * (
            1
            - smoothstep((t - 5.0) / 0.9)
        )
    )

    if title_alpha > 5:
        draw_text(
            canvas,
            CONFIG["title_text"],
            (54, 105),
            size=52,
            fill=(255, 255, 255, title_alpha),
            bold=True,
        )

        draw_wrapped_text(
            canvas,
            CONFIG["subtitle_text"],
            (58, 175),
            max_width=900,
            size=25,
            fill=(170, 225, 255, title_alpha),
        )

    cap = caption_at(t)

    if cap:
        overlay = Image.new(
            "RGBA",
            canvas.size,
            (0, 0, 0, 0),
        )

        draw = ImageDraw.Draw(overlay)

        y0 = canvas.size[1] - 210

        draw.rounded_rectangle(
            (
                42,
                y0,
                canvas.size[0] - 42,
                y0 + 130,
            ),
            radius=25,
            fill=(0, 5, 14, 175),
        )

        canvas.alpha_composite(overlay)

        draw_wrapped_text(
            canvas,
            cap,
            (66, y0 + 26),
            max_width=canvas.size[0] - 132,
            size=29,
            fill=(255, 255, 255, 245),
            bold=True,
        )

    note_alpha = int(
        220
        * smoothstep((t - 51.0) / 2.5)
    )

    if note_alpha > 5:
        draw_text(
            canvas,
            CONFIG["credit_text"],
            (58, canvas.size[1] - 66),
            size=17,
            fill=(205, 230, 245, note_alpha),
            stroke=1,
        )

        draw_wrapped_text(
            canvas,
            CONFIG["scientific_note"],
            (58, canvas.size[1] - 40),
            max_width=950,
            size=14,
            fill=(165, 200, 220, note_alpha),
        )


def render_frame(t: float) -> np.ndarray:
    canvas = render_background(
        OUT_SIZE[0],
        OUT_SIZE[1],
        t,
    )

    draw_map_grid(canvas, t)
    draw_hurricane_tracks(canvas, t)
    draw_category_legend(canvas, t)
    draw_timeline(canvas, t)
    draw_highlight(canvas, t)
    add_text_layers(canvas, t)

    arr = np.array(
        canvas.convert("RGB")
    )

    arr = apply_grade(arr)

    arr = np.clip(
        arr.astype(np.float32)
        * VIGNETTE[..., None],
        0,
        255,
    ).astype(np.uint8)

    fade_in = smoothstep(t / 0.9)
    fade_out = (
        1.0
        - smoothstep(
            (
                t
                - (
                    CONFIG["duration_s"]
                    - 1.15
                )
            )
            / 1.0
        )
    )

    arr = np.clip(
        arr.astype(np.float32)
        * fade_in
        * fade_out,
        0,
        255,
    ).astype(np.uint8)

    return arr


print("Hurricane renderer ready.")

# %% [markdown]
# ## Preview frames

preview_times = [
    1.0,
    10.0,
    22.0,
    34.0,
    46.0,
    CONFIG["duration_s"] - 1.0,
]

preview_arrays = []

for t in tqdm(
    preview_times,
    desc="Preview frames",
):
    arr = render_frame(float(t))
    preview_arrays.append(arr)

    Image.fromarray(arr).save(
        PREVIEW_DIR / f"preview_{int(t):02d}s.png"
    )

fig, axes = plt.subplots(
    1,
    len(preview_arrays),
    figsize=(19, 10),
)

for ax, image, t in zip(
    axes,
    preview_arrays,
    preview_times,
):
    ax.imshow(image)
    ax.set_title(f"{t:.0f}s")
    ax.set_axis_off()

plt.tight_layout()
plt.savefig(
    PREVIEW_DIR / "preview_grid.png",
    dpi=170,
)
plt.show()

print(
    "Preview images written to:",
    PREVIEW_DIR.resolve(),
)

# %% [markdown]
# ## Subtitle sidecar

def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))

    hours = ms // 3_600_000
    ms %= 3_600_000

    minutes = ms // 60_000
    ms %= 60_000

    secs = ms // 1000
    ms %= 1000

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{ms:03d}"
    )


def write_srt(captions, path: Path):
    lines = []

    for index, (
        start,
        end,
        text,
    ) in enumerate(captions, start=1):
        lines.append(str(index))
        lines.append(
            f"{format_srt_time(start)} --> "
            f"{format_srt_time(end)}"
        )
        lines.append(text)
        lines.append("")

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path


SRT_PATH = OUTPUT_ROOT / (
    CONFIG["output_basename"] + ".srt"
)

if CONFIG.get(
    "write_subtitle_sidecar",
    True,
):
    write_srt(
        CAPTIONS,
        SRT_PATH,
    )

    print(
        "Subtitle sidecar:",
        SRT_PATH.resolve(),
    )

# %% [markdown]
# ## Render the full vertical MP4

RAW_VIDEO_PATH = OUTPUT_ROOT / (
    CONFIG["output_basename"] + "_raw.mp4"
)

SUBBED_VIDEO_PATH = OUTPUT_ROOT / (
    CONFIG["output_basename"] + "_subbed.mp4"
)

AUDIO_VIDEO_PATH = OUTPUT_ROOT / (
    CONFIG["output_basename"] + "_with_audio.mp4"
)

FINAL_VIDEO_PATH = OUTPUT_ROOT / (
    CONFIG["output_basename"] + "_final.mp4"
)

nframes = int(
    round(
        CONFIG["duration_s"]
        * CONFIG["fps"]
    )
)

times = (
    np.arange(nframes)
    / CONFIG["fps"]
)

print(
    f"Rendering {nframes:,} frames at "
    f"{CONFIG['video_width']}x"
    f"{CONFIG['video_height']} ..."
)

with iio.get_writer(
    RAW_VIDEO_PATH,
    fps=CONFIG["fps"],
    codec="libx264",
    quality=8,
    pixelformat="yuv420p",
    macro_block_size=None,
) as writer:
    for t in tqdm(
        times,
        desc="Rendering video",
    ):
        writer.append_data(
            render_frame(float(t))
        )

print(
    "Raw video:",
    RAW_VIDEO_PATH.resolve(),
)

# %% [markdown]
# ## Optional subtitles and audio

def run_ffmpeg(cmd: List[str]):
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(
        cmd,
        check=True,
    )


ffmpeg = find_ffmpeg()

print(
    "FFmpeg detected:",
    ffmpeg,
)

final_candidate = RAW_VIDEO_PATH

if (
    CONFIG.get("burn_subtitles", False)
    and ffmpeg
    and SRT_PATH.exists()
):
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(final_candidate),
        "-vf",
        (
            f"subtitles={SRT_PATH}:"
            "force_style="
            "Fontname=DejaVu Sans,"
            "Fontsize=22,"
            "Outline=1.2,"
            "BorderStyle=3,"
            "MarginV=90"
        ),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(SUBBED_VIDEO_PATH),
    ]

    run_ffmpeg(command)

    final_candidate = SUBBED_VIDEO_PATH

audio_path = CONFIG.get("audio_path")

if (
    audio_path
    and Path(audio_path).exists()
    and ffmpeg
):
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
        str(AUDIO_VIDEO_PATH),
    ]

    run_ffmpeg(command)

    final_candidate = AUDIO_VIDEO_PATH

elif audio_path:
    print(
        "audio_path was set, but audio "
        "or FFmpeg was unavailable."
    )

if final_candidate.exists():
    shutil.copyfile(
        final_candidate,
        FINAL_VIDEO_PATH,
    )

    print(
        "Final video:",
        FINAL_VIDEO_PATH.resolve(),
    )

print(
    "Output directory:",
    OUTPUT_ROOT.resolve(),
)

for path in sorted(
    OUTPUT_ROOT.glob("*")
):
    print("-", path.name)

# %% [markdown]
# # Suggested narration / Shorts copy
#
# Voiceover:
#
# > This is the Atlantic hurricane record, compressed into less than a minute.
# > Every trail follows real positions preserved in NOAA's best-track archive.
# > Blue tracks are tropical storms.
# > Green and yellow storms cross the first hurricane categories.
# > Orange, red and magenta mark the strongest winds.
# > The record begins in 1851, decades before satellites watched the Atlantic.
# > Famous storms appear for a moment, then disappear into history.
# > The storm is gone.
# > Its track remains in the data.
#
# Suggested caption:
#
# More than a century of Atlantic tropical cyclone tracks replayed from NOAA/NHC
# HURDAT2 best-track data. Position, date and maximum sustained wind come from
# the historical archive; fading trails and glow are cinematic data encodings.
#
# #Hurricane #NOAA #NHC #HURDAT2 #Weather #ClimateData #DataVisualization
# #ScienceShorts #Python #AtlanticHurricane #Meteorology #StormTracks
