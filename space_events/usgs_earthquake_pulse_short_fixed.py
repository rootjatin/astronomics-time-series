# %% [markdown]
# # Cinematic YouTube Short from Real USGS Earthquake Data
#
# This script creates a vertical 1080x1920 science-visualization Short from
# historical earthquake events returned by the USGS FDSN Event Web Service.
#
# Theme: EARTHQUAKE PULSE — a year of a moving planet
#
# What it produces:
# - Downloads real earthquake origin time, epicenter, depth, magnitude, place,
#   significance, felt reports, and tsunami flags from the USGS catalog.
# - Caches monthly GeoJSON responses so the data layer is reproducible and robust.
# - Builds scientific preview plots.
# - Renders a vertical MP4 with an entirely different visual language from a
#   space fly-through:
#     * rotating orthographic "seismic radar" globe,
#     * expanding epicenter shock rings,
#     * a moving radar sweep,
#     * a depth-core panel driven by catalog depth,
#     * a magnitude/timing-driven synthetic monitor trace,
#     * event callouts and an animated catalog timeline,
#     * optional SRT subtitles and optional audio muxing.
#
# Scientific fidelity note:
# Epicenters, origin times, depths, magnitudes, places, and catalog metadata are
# USGS values. Shock rings, radar sweeps, glow, camera motion, and the monitor
# trace are cinematic encodings for science communication. The monitor trace is
# NOT recorded seismometer ground-motion data.

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams["axes.grid"] = True

print("Imports loaded.")

# %% [markdown]
# ## Configuration
#
# Recommended install:
#
# ```bash
# pip install numpy pandas matplotlib pillow imageio imageio-ffmpeg requests tqdm
# ```
#
# Fast test settings:
#
# ```python
# CONFIG["video_width"] = 540
# CONFIG["video_height"] = 960
# CONFIG["fps"] = 12
# CONFIG["duration_s"] = 12
# CONFIG["max_events_for_video"] = 900
# ```

OUTPUT_ROOT = Path("earthquake_pulse_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
RAW_GEOJSON_DIR = DATA_ROOT / "monthly_geojson"
PREVIEW_DIR = OUTPUT_ROOT / "previews"

for path in [OUTPUT_ROOT, DATA_ROOT, RAW_GEOJSON_DIR, PREVIEW_DIR]:
    path.mkdir(parents=True, exist_ok=True)

CONFIG = {
    # Final delivery
    "video_width": 1080,
    "video_height": 1920,
    "fps": 24,
    "duration_s": 58,
    "output_basename": "usgs_earthquake_pulse_short",

    # Reproducible catalog window. Change these dates to make another edition.
    "catalog_start_date": "2024-01-01",
    "catalog_end_date": "2025-01-01",
    "minimum_magnitude": 4.5,

    # Official USGS FDSN Event query endpoint
    "usgs_query_url": "https://earthquake.usgs.gov/fdsnws/event/1/query",
    "request_timeout": (20, 180),
    "request_retries": 5,
    "retry_backoff_s": 1.8,

    # Visual selection. The full downloaded catalog is retained for statistics.
    "max_events_for_video": 3600,
    "highlight_event_count": 14,
    "active_pulse_lifetime_s": 4.2,

    # Rendering style
    "globe_radius_px": 425,
    "globe_center_y_fraction": 0.47,
    "background_grain_count": 620,
    "scanline_spacing_px": 8,
    "vignette_strength": 0.34,
    "contrast_boost": 1.14,
    "saturation_boost": 1.08,

    # Text
    "title_text": "EARTHQUAKE PULSE",
    "subtitle_text": "One year of a moving planet",
    "credit_text": "Catalog data: U.S. Geological Survey Earthquake Hazards Program",
    "scientific_note": (
        "Epicenters, times, depths and magnitudes are catalog values. "
        "Rings and monitor trace are cinematic data-driven effects, not recorded waveforms."
    ),

    # Optional audio / subtitles
    "audio_path": None,       # Example: "audio/low_rumble_ambient.mp3"
    "burn_subtitles": False,  # requires ffmpeg subtitle support
    "write_subtitle_sidecar": True,
}

print("Configuration ready.")

# %% [markdown]
# ## Download the USGS earthquake catalog
#
# The request is split into calendar-month chunks. This keeps responses compact,
# makes retries/cache reuse practical, and avoids relying on one giant request.


def build_retry_session(config: Dict) -> requests.Session:
    retries = Retry(
        total=int(config.get("request_retries", 5)),
        connect=int(config.get("request_retries", 5)),
        read=int(config.get("request_retries", 5)),
        status=int(config.get("request_retries", 5)),
        backoff_factor=float(config.get("retry_backoff_s", 1.8)),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retries, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "earthquake-pulse-short/1.0 (educational data visualization)"
    })
    return session


def month_windows(start_date: str, end_date: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")
    if end <= start:
        raise ValueError("catalog_end_date must be later than catalog_start_date")

    boundaries = list(pd.date_range(start=start, end=end, freq="MS"))
    if not boundaries or boundaries[0] != start:
        boundaries.insert(0, start)
    if boundaries[-1] != end:
        boundaries.append(end)

    windows = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b > a:
            windows.append((a, b))
    return windows


def timestamp_for_query(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def parse_usgs_geojson(payload: Dict) -> pd.DataFrame:
    rows: List[Dict] = []

    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or [np.nan, np.nan, np.nan]
        coords = list(coords) + [np.nan, np.nan, np.nan]

        rows.append({
            "event_id": feature.get("id", ""),
            "longitude": coords[0],
            "latitude": coords[1],
            "depth_km": coords[2],
            "magnitude": props.get("mag"),
            "place": props.get("place", ""),
            "time_ms": props.get("time"),
            "updated_ms": props.get("updated"),
            "felt": props.get("felt"),
            "cdi": props.get("cdi"),
            "mmi": props.get("mmi"),
            "alert": props.get("alert"),
            "status": props.get("status"),
            "tsunami": props.get("tsunami"),
            "significance": props.get("sig"),
            "network": props.get("net"),
            "code": props.get("code"),
            "nst": props.get("nst"),
            "dmin": props.get("dmin"),
            "rms": props.get("rms"),
            "gap": props.get("gap"),
            "magnitude_type": props.get("magType"),
            "event_type": props.get("type"),
            "title": props.get("title", ""),
            "url": props.get("url", ""),
            "detail_url": props.get("detail", ""),
        })

    return pd.DataFrame(rows)


def clean_earthquake_dataframe(df: pd.DataFrame, save_path: Optional[Path] = None) -> pd.DataFrame:
    numeric_cols = [
        "longitude", "latitude", "depth_km", "magnitude", "time_ms", "updated_ms",
        "felt", "cdi", "mmi", "tsunami", "significance", "nst", "dmin", "rms", "gap",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = [
        "event_id", "place", "alert", "status", "network", "code",
        "magnitude_type", "event_type", "title", "url", "detail_url",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    required = ["event_id", "longitude", "latitude", "depth_km", "magnitude", "time_ms"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required USGS columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["longitude", "latitude", "depth_km", "magnitude", "time_ms"]).copy()
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))].copy()
    df = df[df["depth_km"] >= 0].copy()
    df = df[df["magnitude"] >= float(CONFIG["minimum_magnitude"])].copy()

    if "event_type" in df.columns:
        df = df[df["event_type"].str.lower().eq("earthquake")].copy()

    df["event_time"] = pd.to_datetime(df["time_ms"], unit="ms", utc=True)
    df["date_label"] = df["event_time"].dt.strftime("%Y-%m-%d")
    df["month_label"] = df["event_time"].dt.strftime("%Y-%m")
    df["tsunami"] = df["tsunami"].fillna(0).astype(int)
    df["felt"] = df["felt"].fillna(0)
    df["significance"] = df["significance"].fillna(0)
    df["depth_group"] = pd.cut(
        df["depth_km"],
        bins=[-0.001, 70, 300, np.inf],
        labels=["shallow", "intermediate", "deep"],
    ).astype(str)

    # De-duplicate overlap at month boundaries or catalog revisions.
    df = df.drop_duplicates(subset=["event_id"], keep="last")
    df = df.sort_values(["event_time", "event_id"]).reset_index(drop=True)

    if save_path is not None:
        df.to_csv(save_path, index=False)
        print("Saved cleaned catalog:", save_path)

    print(f"Earthquake events retained: {len(df):,}")
    if len(df):
        print("Catalog window:", df["event_time"].min(), "to", df["event_time"].max())
        print("Maximum magnitude:", float(df["magnitude"].max()))
    return df


def fetch_usgs_catalog(config: Dict, force_refresh: bool = False) -> pd.DataFrame:
    clean_path = DATA_ROOT / "usgs_earthquakes_clean.csv"
    if clean_path.exists() and not force_refresh:
        print("Using cached cleaned USGS catalog:", clean_path)
        return clean_earthquake_dataframe(pd.read_csv(clean_path), save_path=None)

    session = build_retry_session(config)
    frames: List[pd.DataFrame] = []

    for start, end in month_windows(config["catalog_start_date"], config["catalog_end_date"]):
        cache_name = f"usgs_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_m{float(config['minimum_magnitude']):.1f}.geojson"
        cache_path = RAW_GEOJSON_DIR / cache_name

        payload: Optional[Dict] = None
        if cache_path.exists() and not force_refresh:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            print("Using cached month:", cache_path.name)
        else:
            params = {
                "format": "geojson",
                "starttime": timestamp_for_query(start),
                "endtime": timestamp_for_query(end),
                "minmagnitude": float(config["minimum_magnitude"]),
                "eventtype": "earthquake",
                "orderby": "time-asc",
            }
            print(f"Requesting USGS events {start.date()} to {end.date()} ...")
            response = session.get(
                config["usgs_query_url"],
                params=params,
                timeout=config["request_timeout"],
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"USGS HTTP {response.status_code}: {response.reason}\n"
                    f"Response preview:\n{response.text[:1500]}"
                )
            payload = response.json()
            cache_path.write_text(json.dumps(payload), encoding="utf-8")

        frames.append(parse_usgs_geojson(payload or {}))

    if not frames:
        raise RuntimeError("No USGS catalog chunks were returned.")

    combined = pd.concat(frames, ignore_index=True)
    return clean_earthquake_dataframe(combined, save_path=clean_path)


event_df = fetch_usgs_catalog(CONFIG, force_refresh=False)
event_df.head()

# %% [markdown]
# ## Choose a readable event subset for the video
#
# The full catalog remains available for statistics and preview charts. The
# renderer prioritizes higher-magnitude, high-significance, tsunami-flagged, and
# widely felt events, while retaining a time-stratified sample so the year does
# not visually collapse into only a handful of extreme events.


def select_video_events(df: pd.DataFrame, max_events: int) -> pd.DataFrame:
    work = df.copy()
    if len(work) <= max_events:
        return work.sort_values("event_time").reset_index(drop=True)

    work["visual_score"] = (
        (work["magnitude"] - float(CONFIG["minimum_magnitude"])).clip(lower=0) * 160
        + np.log1p(work["significance"].clip(lower=0)) * 28
        + np.log1p(work["felt"].clip(lower=0)) * 18
        + work["tsunami"].clip(0, 1) * 260
    )

    # Keep globally strongest/most significant events.
    reserve_count = min(max(120, max_events // 5), max_events)
    reserve = work.nlargest(reserve_count, "visual_score")

    remaining_n = max_events - len(reserve)
    pool = work.drop(index=reserve.index)
    if remaining_n > 0 and len(pool):
        # Time-stratified selection: top-scoring events from each slice of the year.
        bin_count = min(120, max(24, remaining_n // 20))
        pool = pool.copy()
        normalized_rank = pool["event_time"].rank(method="first", pct=True)
        pool["time_bin"] = np.minimum((normalized_rank * bin_count).astype(int), bin_count - 1)
        per_bin = max(1, math.ceil(remaining_n / bin_count))
        sampled = (
            pool.sort_values(["time_bin", "visual_score"], ascending=[True, False])
            .groupby("time_bin", group_keys=False)
            .head(per_bin)
            .nlargest(remaining_n, "visual_score")
        )
        chosen = pd.concat([reserve, sampled], ignore_index=True)
    else:
        chosen = reserve

    return (
        chosen.drop_duplicates(subset=["event_id"])
        .sort_values("event_time")
        .head(max_events)
        .reset_index(drop=True)
    )


video_events = select_video_events(event_df, int(CONFIG["max_events_for_video"]))

highlight_events = (
    event_df.assign(
        highlight_score=(
            event_df["magnitude"] * 1000
            + np.log1p(event_df["significance"].clip(lower=0)) * 90
            + event_df["tsunami"].clip(0, 1) * 250
            + np.log1p(event_df["felt"].clip(lower=0)) * 20
        )
    )
    .nlargest(int(CONFIG["highlight_event_count"]), "highlight_score")
    .sort_values("event_time")
    .reset_index(drop=True)
)

print(f"Video events: {len(video_events):,}")
print("Depth groups:")
print(video_events["depth_group"].value_counts())
print("Highlighted events:")
print(highlight_events[["date_label", "magnitude", "depth_km", "place"]].to_string(index=False))

# %% [markdown]
# ## Scientific preview plots

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(event_df["magnitude"], bins=np.arange(CONFIG["minimum_magnitude"], event_df["magnitude"].max() + 0.25, 0.2))
ax.set_title("USGS catalog magnitude distribution")
ax.set_xlabel("Magnitude")
ax.set_ylabel("Event count")
plt.tight_layout()
plt.savefig(PREVIEW_DIR / "magnitude_distribution.png", dpi=170)
plt.show()

fig, ax = plt.subplots(figsize=(9, 6))
plot_df = event_df.sample(min(len(event_df), 7000), random_state=11) if len(event_df) else event_df
if len(plot_df):
    ax.scatter(plot_df["magnitude"], plot_df["depth_km"], s=8, alpha=0.45)
ax.invert_yaxis()
ax.set_title("Earthquake magnitude vs catalog depth")
ax.set_xlabel("Magnitude")
ax.set_ylabel("Depth [km]")
plt.tight_layout()
plt.savefig(PREVIEW_DIR / "magnitude_vs_depth.png", dpi=170)
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
monthly_counts = event_df.set_index("event_time").resample("MS").size()

# Draw the bars explicitly instead of using Series.plot(kind="bar").
# Some pandas/Matplotlib combinations try to route a DatetimeIndex with
# freq=<MonthBegin> through PeriodConverter, which can raise:
# ValueError: <MonthBegin> is not supported as period frequency
if len(monthly_counts):
    month_labels = monthly_counts.index.strftime("%Y-%m")
    x_positions = np.arange(len(monthly_counts))
    ax.bar(x_positions, monthly_counts.to_numpy())
    ax.set_xticks(x_positions)
    ax.set_xticklabels(month_labels, rotation=45, ha="right")

ax.set_title("Catalog events by month")
ax.set_xlabel("Month")
ax.set_ylabel("Event count")
plt.tight_layout()
plt.savefig(PREVIEW_DIR / "events_by_month.png", dpi=170)
plt.show()

# %% [markdown]
# ## Cinematic rendering helpers

OUT_SIZE = (int(CONFIG["video_width"]), int(CONFIG["video_height"]))

DEPTH_COLORS = {
    "shallow": (255, 92, 80, 245),
    "intermediate": (255, 190, 72, 235),
    "deep": (84, 220, 255, 235),
}


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3 - 2 * t)


def ease_in_out_sine(t: float) -> float:
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1) / 2


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
        "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
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
    size: int = 38,
    fill=(235, 255, 245, 255),
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
        stroke_fill=(0, 0, 0, min(220, fill[3] if len(fill) > 3 else 220)),
    )


def draw_wrapped_text(
    img: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int = 32,
    fill=(245, 255, 250, 245),
    bold: bool = False,
    line_spacing: int = 8,
):
    draw = ImageDraw.Draw(img)
    font = get_font(size, bold=bold)
    words = str(text).split()
    lines: List[str] = []
    current = ""

    for word in words:
        candidate = word if not current else current + " " + word
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
        draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 220))
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += (bbox[3] - bbox[1]) + line_spacing


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    radius = np.sqrt(nx**2 + ny**2)
    return np.clip(1 - strength * radius**1.7, 0, 1).astype(np.float32)


def apply_grade(arr: np.ndarray) -> np.ndarray:
    img = Image.fromarray(arr)
    img = ImageEnhance.Contrast(img).enhance(float(CONFIG["contrast_boost"]))
    img = ImageEnhance.Color(img).enhance(float(CONFIG["saturation_boost"]))
    return np.array(img)


VIGNETTE = make_vignette(OUT_SIZE[0], OUT_SIZE[1], float(CONFIG["vignette_strength"]))

# %% [markdown]
# ## Map catalog time into video time

DATA_T0 = event_df["event_time"].min()
DATA_T1 = event_df["event_time"].max()
DATA_SPAN_S = max((DATA_T1 - DATA_T0).total_seconds(), 1.0)

VIDEO_DATA_START_S = 5.8
VIDEO_DATA_END_S = 50.8
VIDEO_DATA_SPAN_S = VIDEO_DATA_END_S - VIDEO_DATA_START_S


def catalog_time_to_video_seconds(times: Sequence[pd.Timestamp]) -> np.ndarray:
    series = pd.to_datetime(pd.Series(times), utc=True)
    seconds = (series - DATA_T0).dt.total_seconds().to_numpy(float)
    normalized = np.clip(seconds / DATA_SPAN_S, 0, 1)
    return VIDEO_DATA_START_S + normalized * VIDEO_DATA_SPAN_S


video_events = video_events.copy()
video_events["video_t"] = catalog_time_to_video_seconds(video_events["event_time"])

highlight_events = highlight_events.copy()
highlight_events["video_t"] = catalog_time_to_video_seconds(highlight_events["event_time"])

EV = {
    "lon": video_events["longitude"].to_numpy(float),
    "lat": video_events["latitude"].to_numpy(float),
    "depth": video_events["depth_km"].to_numpy(float),
    "mag": video_events["magnitude"].to_numpy(float),
    "sig": video_events["significance"].to_numpy(float),
    "felt": video_events["felt"].to_numpy(float),
    "tsunami": video_events["tsunami"].to_numpy(int),
    "place": video_events["place"].to_numpy(object),
    "date": video_events["date_label"].to_numpy(object),
    "group": video_events["depth_group"].to_numpy(object),
    "video_t": video_events["video_t"].to_numpy(float),
}

HI = {
    "lon": highlight_events["longitude"].to_numpy(float),
    "lat": highlight_events["latitude"].to_numpy(float),
    "depth": highlight_events["depth_km"].to_numpy(float),
    "mag": highlight_events["magnitude"].to_numpy(float),
    "place": highlight_events["place"].to_numpy(object),
    "date": highlight_events["date_label"].to_numpy(object),
    "video_t": highlight_events["video_t"].to_numpy(float),
}

# %% [markdown]
# ## Shot plan and captions

CAPTIONS = [
    (0.5, 5.2, "This is one year of earthquakes, compressed into less than a minute."),
    (6.0, 15.8, "Every pulse starts at a catalog epicenter at its real position on Earth."),
    (16.2, 27.2, "Shallow earthquakes flash red. Intermediate events burn amber. Deep events glow cyan."),
    (27.6, 39.0, "Magnitude changes the size and energy of each cinematic pulse."),
    (39.4, 49.8, "The strongest events become landmarks in a planet-wide seismic timeline."),
    (50.2, 57.2, "Earth is never perfectly still. The catalog turns that motion into data."),
]

SHOT_PLAN = [
    {
        "name": "boot",
        "start": 0.0,
        "end": 6.0,
        "lon_start": -165,
        "lon_end": -135,
        "lat_start": 10,
        "lat_end": 18,
        "caption": "SEISMIC CATALOG // SYSTEM ONLINE",
    },
    {
        "name": "global_pulse",
        "start": 6.0,
        "end": 18.0,
        "lon_start": -135,
        "lon_end": -20,
        "lat_start": 18,
        "lat_end": 5,
        "caption": "EPICENTERS REPLAYED IN CATALOG TIME",
    },
    {
        "name": "depth_scan",
        "start": 18.0,
        "end": 31.0,
        "lon_start": -20,
        "lon_end": 105,
        "lat_start": 5,
        "lat_end": -10,
        "caption": "DEPTH CORE // 0 TO 700+ KM",
    },
    {
        "name": "energy",
        "start": 31.0,
        "end": 44.0,
        "lon_start": 105,
        "lon_end": 225,
        "lat_start": -10,
        "lat_end": 15,
        "caption": "MAGNITUDE DRIVES PULSE SCALE",
    },
    {
        "name": "landmarks",
        "start": 44.0,
        "end": 51.5,
        "lon_start": 225,
        "lon_end": 300,
        "lat_start": 15,
        "lat_end": 8,
        "caption": "HIGH-SIGNIFICANCE EVENTS MARKED",
    },
    {
        "name": "outro",
        "start": 51.5,
        "end": CONFIG["duration_s"],
        "lon_start": 300,
        "lon_end": 345,
        "lat_start": 8,
        "lat_end": 12,
        "caption": "A MOVING PLANET // A GROWING CATALOG",
    },
]


def get_shot(t: float) -> Dict:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def shot_state(t: float) -> Tuple[Dict, float, float]:
    shot = get_shot(t)
    u = (t - shot["start"]) / max(shot["end"] - shot["start"], 1e-6)
    e = ease_in_out_sine(u)
    center_lon = lerp(shot["lon_start"], shot["lon_end"], e)
    center_lat = lerp(shot["lat_start"], shot["lat_end"], e)
    return shot, center_lon, center_lat


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None

# %% [markdown]
# ## Orthographic globe projection


def orthographic_project(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    center_lat_deg: float,
    center_lon_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    center_lat = math.radians(center_lat_deg)
    center_lon = math.radians(center_lon_deg)
    dlon = lon - center_lon

    x = np.cos(lat) * np.sin(dlon)
    y = (
        math.cos(center_lat) * np.sin(lat)
        - math.sin(center_lat) * np.cos(lat) * np.cos(dlon)
    )
    z = (
        math.sin(center_lat) * np.sin(lat)
        + math.cos(center_lat) * np.cos(lat) * np.cos(dlon)
    )
    return x, y, z


def globe_geometry(canvas: Image.Image) -> Tuple[float, float, float]:
    radius = float(CONFIG["globe_radius_px"]) * canvas.size[0] / 1080.0
    cx = canvas.size[0] / 2
    cy = canvas.size[1] * float(CONFIG["globe_center_y_fraction"])
    return cx, cy, radius


def project_to_canvas(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    center_lat_deg: float,
    center_lon_deg: float,
    canvas: Image.Image,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = orthographic_project(lat_deg, lon_deg, center_lat_deg, center_lon_deg)
    cx, cy, radius = globe_geometry(canvas)
    px = cx + x * radius
    py = cy - y * radius
    return px, py, z

# %% [markdown]
# ## Data-driven monitor signal
#
# This is intentionally synthetic. It uses catalog timing, magnitude, and depth to
# generate a damped monitor trace for visual storytelling. It is not waveform data.


def build_cinematic_monitor_signal(events: pd.DataFrame, duration_s: float, samples: int = 9000):
    grid_t = np.linspace(0, duration_s, samples)
    signal = np.zeros(samples, dtype=np.float64)

    # Strongest events provide the visible monitor kicks.
    drivers = events.nlargest(min(520, len(events)), "magnitude").copy()
    for _, row in drivers.iterrows():
        event_t = float(row["video_t"])
        mag = float(row["magnitude"])
        depth = float(row["depth_km"])
        dt = grid_t - event_t
        mask = (dt >= 0) & (dt <= 3.6)
        if not np.any(mask):
            continue

        tau = dt[mask]
        amp = 0.18 * (10 ** (0.25 * (mag - float(CONFIG["minimum_magnitude"]))))
        frequency = np.clip(8.0 - depth / 160.0, 2.4, 8.0)
        decay = np.exp(-tau * (1.45 + depth / 800.0))
        signal[mask] += amp * decay * (
            np.sin(2 * np.pi * frequency * tau)
            + 0.32 * np.sin(2 * np.pi * frequency * 2.15 * tau + 0.7)
        )

    # Low-level deterministic electronic drift.
    signal += 0.028 * np.sin(2 * np.pi * 1.1 * grid_t)
    signal += 0.018 * np.sin(2 * np.pi * 3.7 * grid_t + 1.4)

    scale = np.percentile(np.abs(signal), 99.3) if np.any(signal) else 1.0
    signal = np.clip(signal / max(scale, 1e-6), -1.0, 1.0)
    return grid_t, signal


MONITOR_T, MONITOR_SIGNAL = build_cinematic_monitor_signal(video_events, CONFIG["duration_s"])

# %% [markdown]
# ## Background, globe grid, and radar sweep


def generate_grain(count: int, seed: int = 54):
    rng = np.random.default_rng(seed)
    return [
        (
            float(rng.uniform(0, OUT_SIZE[0])),
            float(rng.uniform(0, OUT_SIZE[1])),
            float(rng.uniform(0.3, 1.6)),
            int(rng.integers(10, 55)),
            float(rng.uniform(0, 2 * np.pi)),
        )
        for _ in range(count)
    ]


GRAIN = generate_grain(int(CONFIG["background_grain_count"]))


def render_background(width: int, height: int, t: float) -> Image.Image:
    img = Image.new("RGBA", (width, height), (2, 8, 10, 255))

    # Large diffused radar/CRT glow.
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx = width * (0.5 + 0.015 * math.sin(t * 0.11))
    cy = height * 0.45
    for radius, alpha in [(760, 14), (560, 18), (370, 24), (220, 22)]:
        gd.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(10, 120, 98, alpha))
    glow = glow.filter(ImageFilter.GaussianBlur(70))
    img.alpha_composite(glow)

    draw = ImageDraw.Draw(img)
    for x, y, radius, alpha, phase in GRAIN:
        px = (x + 5 * math.sin(t * 0.7 + phase)) % width
        py = (y + 2 * math.cos(t * 0.9 + phase)) % height
        a = int(alpha * (0.65 + 0.35 * math.sin(t * 1.7 + phase)))
        draw.ellipse((px-radius, py-radius, px+radius, py+radius), fill=(110, 220, 190, max(1, a)))

    # Horizontal scanlines.
    spacing = max(4, int(CONFIG["scanline_spacing_px"] * height / 1920))
    for y in range(0, height, spacing):
        draw.line((0, y, width, y), fill=(70, 160, 140, 12), width=1)

    # Moving bright scanner line.
    scanner_y = int((t * 96) % (height + 160)) - 80
    draw.rectangle((0, scanner_y, width, scanner_y + 2), fill=(120, 255, 220, 28))
    return img


def draw_polyline_visible(
    draw: ImageDraw.ImageDraw,
    px: np.ndarray,
    py: np.ndarray,
    visible: np.ndarray,
    fill: Tuple[int, int, int, int],
    width: int = 1,
):
    segment: List[Tuple[float, float]] = []
    for x, y, is_visible in zip(px, py, visible):
        if is_visible:
            segment.append((float(x), float(y)))
        else:
            if len(segment) >= 2:
                draw.line(segment, fill=fill, width=width)
            segment = []
    if len(segment) >= 2:
        draw.line(segment, fill=fill, width=width)


def draw_globe_grid(canvas: Image.Image, center_lon: float, center_lat: float, t: float):
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy, radius = globe_geometry(canvas)

    # Globe disc and layered rim.
    draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(2, 25, 25, 118))
    draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=(80, 245, 205, 125), width=2)
    draw.ellipse((cx-radius+8, cy-radius+8, cx+radius-8, cy+radius-8), outline=(80, 210, 180, 42), width=1)

    sample_lon = np.linspace(-180, 180, 361)
    for lat in range(-60, 61, 30):
        lat_values = np.full_like(sample_lon, lat, dtype=float)
        px, py, z = project_to_canvas(lat_values, sample_lon, center_lat, center_lon, canvas)
        draw_polyline_visible(draw, px, py, z >= 0, (80, 220, 190, 50), width=1)

    sample_lat = np.linspace(-90, 90, 241)
    for lon in range(-180, 180, 30):
        lon_values = np.full_like(sample_lat, lon, dtype=float)
        px, py, z = project_to_canvas(sample_lat, lon_values, center_lat, center_lon, canvas)
        draw_polyline_visible(draw, px, py, z >= 0, (80, 220, 190, 44), width=1)

    # Range circles make the object feel more like a tracking instrument than a map.
    for frac in [0.25, 0.5, 0.75]:
        rr = radius * frac
        draw.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), outline=(80, 220, 190, 28), width=1)

    # Radar sweep line and wedge.
    angle = t * 0.72 - math.pi / 2
    sweep = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(sweep)
    points = [(cx, cy)]
    for delta in np.linspace(-0.23, 0.0, 16):
        a = angle + delta
        points.append((cx + math.cos(a) * radius, cy + math.sin(a) * radius))
    sd.polygon(points, fill=(80, 255, 205, 16))
    sd.line((cx, cy, cx + math.cos(angle) * radius, cy + math.sin(angle) * radius), fill=(125, 255, 220, 110), width=2)
    sweep = sweep.filter(ImageFilter.GaussianBlur(2))

    canvas.alpha_composite(overlay)
    canvas.alpha_composite(sweep)

# %% [markdown]
# ## Earthquake pulses


def active_event_indices(t: float) -> np.ndarray:
    age = t - EV["video_t"]
    return np.where((age >= 0) & (age <= float(CONFIG["active_pulse_lifetime_s"])))[0]


def draw_earthquake_pulses(canvas: Image.Image, center_lon: float, center_lat: float, t: float):
    indices = active_event_indices(t)
    if len(indices) == 0:
        return

    px, py, z = project_to_canvas(EV["lat"][indices], EV["lon"][indices], center_lat, center_lon, canvas)
    visible_mask = z >= 0

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    life = float(CONFIG["active_pulse_lifetime_s"])
    for local_idx, event_idx in enumerate(indices):
        if not visible_mask[local_idx]:
            continue

        age = t - EV["video_t"][event_idx]
        u = clamp(age / life)
        x = float(px[local_idx])
        y = float(py[local_idx])
        mag = float(EV["mag"][event_idx])
        group = str(EV["group"][event_idx])
        base = DEPTH_COLORS.get(group, (220, 255, 240, 220))

        # Magnitude changes pulse scale. This is a cinematic encoding, not rupture radius.
        mag_gain = np.clip((mag - float(CONFIG["minimum_magnitude"])) / 3.8, 0, 1.25)
        max_radius = 28 + 95 * mag_gain
        radius = 6 + max_radius * smoothstep(u)
        alpha = int(210 * (1 - u) ** 1.35)

        # Diffused impact glow.
        glow_radius = 7 + 19 * mag_gain
        draw.ellipse(
            (x-glow_radius, y-glow_radius, x+glow_radius, y+glow_radius),
            fill=(base[0], base[1], base[2], int(55 * (1-u))),
        )

        # Two staggered shock rings.
        draw.ellipse(
            (x-radius, y-radius, x+radius, y+radius),
            outline=(base[0], base[1], base[2], alpha),
            width=max(1, int(2 + 2 * mag_gain)),
        )
        inner_u = clamp((u - 0.17) / 0.83)
        inner_radius = 5 + max_radius * 0.68 * smoothstep(inner_u)
        inner_alpha = int(130 * (1 - inner_u) ** 1.5)
        draw.ellipse(
            (x-inner_radius, y-inner_radius, x+inner_radius, y+inner_radius),
            outline=(base[0], base[1], base[2], inner_alpha),
            width=1,
        )

        # Epicenter core and magnitude tick.
        core = 2.0 + 3.8 * mag_gain
        draw.ellipse((x-core, y-core, x+core, y+core), fill=(245, 255, 245, min(255, alpha + 35)))
        if mag >= 6.5:
            tick = 12 + mag_gain * 12
            draw.line((x, y-tick, x, y+tick), fill=(base[0], base[1], base[2], alpha), width=1)
            draw.line((x-tick, y, x+tick, y), fill=(base[0], base[1], base[2], alpha), width=1)

    glow = overlay.filter(ImageFilter.GaussianBlur(7))
    canvas.alpha_composite(glow)
    canvas.alpha_composite(overlay)

# %% [markdown]
# ## Depth core panel


def draw_depth_core(canvas: Image.Image, t: float):
    alpha = int(225 * smoothstep((t - 15.5) / 4.0) * (1 - 0.75 * smoothstep((t - 50.0) / 4.0)))
    if alpha <= 5:
        return

    x0 = canvas.size[0] - 190
    y0 = 360
    y1 = 1280
    shaft_x = x0 + 76

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((x0-28, y0-72, canvas.size[0]-36, y1+80), radius=26, fill=(0, 8, 10, 128))
    draw_text(overlay, "DEPTH", (x0, y0-48), size=22, fill=(170, 255, 225, alpha), bold=True, stroke=1)
    draw.line((shaft_x, y0, shaft_x, y1), fill=(130, 240, 210, int(alpha * 0.65)), width=2)

    depth_ticks = [0, 70, 300, 700]
    for depth in depth_ticks:
        frac = np.clip(depth / 700.0, 0, 1)
        y = y0 + frac * (y1 - y0)
        draw.line((shaft_x-18, y, shaft_x+18, y), fill=(150, 235, 210, int(alpha * 0.55)), width=1)
        draw_text(overlay, f"{depth}", (x0, int(y)-10), size=16, fill=(190, 235, 220, alpha), stroke=1)

    indices = active_event_indices(t)
    for event_idx in indices:
        age = t - EV["video_t"][event_idx]
        u = clamp(age / float(CONFIG["active_pulse_lifetime_s"]))
        depth = float(EV["depth"][event_idx])
        frac = np.clip(depth / 700.0, 0, 1)
        y = y0 + frac * (y1 - y0)
        group = str(EV["group"][event_idx])
        color = DEPTH_COLORS.get(group, (220, 255, 240, 220))
        mag_gain = np.clip((EV["mag"][event_idx] - CONFIG["minimum_magnitude"]) / 3.8, 0, 1.2)
        radius = 2.0 + 4.5 * mag_gain
        event_alpha = int(alpha * (1-u))
        jitter = 20 * math.sin(event_idx * 2.4 + t * 3.1)
        draw.ellipse(
            (shaft_x+jitter-radius, y-radius, shaft_x+jitter+radius, y+radius),
            fill=(color[0], color[1], color[2], event_alpha),
        )

    # Traveling depth scan marker.
    scan_u = (t * 0.23) % 1.0
    scan_y = y0 + scan_u * (y1 - y0)
    draw.line((shaft_x-30, scan_y, shaft_x+30, scan_y), fill=(220, 255, 240, int(alpha * 0.8)), width=2)

    canvas.alpha_composite(overlay)

# %% [markdown]
# ## Monitor trace and catalog timeline


def draw_monitor_trace(canvas: Image.Image, t: float):
    x0, x1 = 62, canvas.size[0] - 62
    y0 = canvas.size[1] - 510
    height = 150
    width = x1 - x0

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((x0-20, y0-58, x1+20, y0+height+46), radius=24, fill=(0, 7, 9, 155))
    draw_text(overlay, "CATALOG-DRIVEN MONITOR // NOT RECORDED WAVEFORM", (x0, y0-40), size=19, fill=(150, 245, 215, 205), bold=True, stroke=1)

    window_s = 7.5
    sample_t = np.linspace(t-window_s, t, 620)
    sample_y = np.interp(sample_t, MONITOR_T, MONITOR_SIGNAL, left=0, right=0)
    px = np.linspace(x0, x1, len(sample_t))
    py = y0 + height/2 - sample_y * height * 0.42

    draw.line((x0, y0+height/2, x1, y0+height/2), fill=(80, 185, 160, 42), width=1)
    points = list(zip(px.tolist(), py.tolist()))
    if len(points) >= 2:
        draw.line(points, fill=(115, 255, 210, 230), width=2)

    canvas.alpha_composite(overlay)


def draw_catalog_timeline(canvas: Image.Image, t: float):
    x0, x1 = 72, canvas.size[0] - 72
    y0 = canvas.size[1] - 300
    width = x1 - x0
    alpha = 220

    counts, _ = np.histogram(EV["video_t"], bins=110, range=(VIDEO_DATA_START_S, VIDEO_DATA_END_S))
    counts = counts.astype(float)
    counts /= max(counts.max(), 1)

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw_text(overlay, "CATALOG TIME", (x0, y0-38), size=20, fill=(160, 245, 215, alpha), bold=True, stroke=1)
    draw.line((x0, y0, x1, y0), fill=(120, 230, 200, 100), width=1)

    bar_w = width / len(counts)
    for i, value in enumerate(counts):
        x = x0 + i * bar_w
        bar_h = 12 + value * 65
        draw.rectangle((x, y0-bar_h, x+max(1, bar_w-1), y0), fill=(80, 220, 185, int(45 + 115 * value)))

    progress = np.clip((t - VIDEO_DATA_START_S) / VIDEO_DATA_SPAN_S, 0, 1)
    cursor_x = x0 + progress * width
    draw.line((cursor_x, y0-92, cursor_x, y0+20), fill=(245, 255, 240, 225), width=2)

    # Translate video time back to catalog date for the readout.
    data_seconds = np.clip(progress, 0, 1) * DATA_SPAN_S
    readout_time = DATA_T0 + pd.to_timedelta(data_seconds, unit="s")
    draw_text(overlay, readout_time.strftime("%Y-%m-%d"), (int(cursor_x), y0+28), size=18, fill=(230, 255, 245, 235), bold=True, stroke=1, anchor="ma")

    canvas.alpha_composite(overlay)

# %% [markdown]
# ## Highlight callouts


def shorten_place(place: str, max_chars: int = 42) -> str:
    place = str(place).strip() or "Unnamed catalog location"
    return place if len(place) <= max_chars else place[:max_chars-1].rstrip() + "…"


def current_highlight_index(t: float) -> Optional[int]:
    if len(HI["video_t"]) == 0:
        return None
    dt = t - HI["video_t"]
    candidates = np.where((dt >= -0.15) & (dt <= 3.0))[0]
    if len(candidates) == 0:
        return None
    return int(candidates[np.argmin(np.abs(dt[candidates] - 0.55))])


def draw_highlight_callout(canvas: Image.Image, center_lon: float, center_lat: float, t: float):
    idx = current_highlight_index(t)
    if idx is None:
        return

    px, py, z = project_to_canvas(
        np.array([HI["lat"][idx]]),
        np.array([HI["lon"][idx]]),
        center_lat,
        center_lon,
        canvas,
    )

    age = t - float(HI["video_t"][idx])
    fade = smoothstep((age + 0.15) / 0.45) * (1 - smoothstep((age - 2.15) / 0.75))
    alpha = int(245 * fade)
    if alpha <= 5:
        return

    panel = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    x0, y0 = 54, 280
    x1, y1 = 700, 482
    draw.rounded_rectangle((x0, y0, x1, y1), radius=26, fill=(0, 8, 10, int(175 * fade)), outline=(135, 255, 220, int(85 * fade)), width=1)
    draw_text(panel, "SEISMIC LANDMARK", (x0+26, y0+24), size=21, fill=(135, 255, 220, alpha), bold=True, stroke=1)
    draw_text(panel, f"M {HI['mag'][idx]:.1f}", (x0+26, y0+62), size=48, fill=(255, 245, 220, alpha), bold=True)
    draw_text(panel, f"DEPTH {HI['depth'][idx]:.0f} KM", (x0+250, y0+76), size=23, fill=(220, 250, 235, alpha), bold=True, stroke=1)
    draw_wrapped_text(panel, shorten_place(HI["place"][idx]), (x0+28, y0+126), max_width=590, size=24, fill=(240, 255, 248, alpha), bold=True)
    draw_text(panel, str(HI["date"][idx]), (x1-28, y1-30), size=19, fill=(180, 235, 215, alpha), bold=False, stroke=1, anchor="ra")

    if z[0] >= 0:
        x, y = float(px[0]), float(py[0])
        anchor_x = x1
        anchor_y = y0 + 130
        draw.line((anchor_x, anchor_y, x, y), fill=(150, 255, 220, int(120 * fade)), width=1)
        draw.ellipse((x-8, y-8, x+8, y+8), outline=(245, 255, 240, alpha), width=2)

    canvas.alpha_composite(panel)

# %% [markdown]
# ## Legend and text layers


def draw_depth_legend(canvas: Image.Image, t: float):
    alpha = int(220 * smoothstep((t - 13.5) / 3.5) * (1 - smoothstep((t - 44.5) / 5.0)))
    if alpha <= 5:
        return

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x0, y0 = 56, 530
    draw.rounded_rectangle((x0-14, y0-18, x0+420, y0+150), radius=22, fill=(0, 8, 10, 105))
    draw_text(overlay, "DEPTH COLOR", (x0, y0), size=21, fill=(160, 250, 220, alpha), bold=True, stroke=1)

    items = [
        ("SHALLOW  <70 KM", DEPTH_COLORS["shallow"]),
        ("MID      70–300 KM", DEPTH_COLORS["intermediate"]),
        ("DEEP     >300 KM", DEPTH_COLORS["deep"]),
    ]
    y = y0 + 42
    for label, color in items:
        draw.ellipse((x0, y, x0+16, y+16), fill=(color[0], color[1], color[2], alpha))
        draw_text(overlay, label, (x0+30, y-3), size=19, fill=(215, 245, 232, alpha), stroke=1)
        y += 34

    canvas.alpha_composite(overlay)


def add_text_layers(canvas: Image.Image, t: float, shot: Dict):
    title_alpha = int(255 * smoothstep((t - 0.25) / 1.0) * (1 - smoothstep((t - 5.0) / 0.9)))
    if title_alpha > 5:
        draw_text(canvas, CONFIG["title_text"], (54, 112), size=58, fill=(220, 255, 240, title_alpha), bold=True)
        draw_text(canvas, CONFIG["subtitle_text"], (58, 194), size=27, fill=(145, 245, 210, min(230, title_alpha)), bold=False, stroke=1)

    if t > 5.2:
        draw_text(canvas, shot["caption"], (54, 76), size=21, fill=(135, 245, 210, 185), bold=True, stroke=1)

    cap = caption_at(t)
    if cap:
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        y0 = canvas.size[1] - 188
        draw.rounded_rectangle((42, y0, canvas.size[0]-42, y0+122), radius=25, fill=(0, 5, 7, 165))
        canvas.alpha_composite(overlay)
        draw_wrapped_text(canvas, cap, (66, y0+24), max_width=canvas.size[0]-132, size=29, fill=(245, 255, 250, 245), bold=True)

    note_alpha = int(220 * smoothstep((t - 50.8) / 2.4))
    if note_alpha > 5:
        draw_wrapped_text(canvas, CONFIG["credit_text"], (56, canvas.size[1]-64), max_width=960, size=18, fill=(210, 240, 228, note_alpha))
        draw_wrapped_text(canvas, CONFIG["scientific_note"], (56, canvas.size[1]-38), max_width=960, size=15, fill=(165, 215, 198, note_alpha))

# %% [markdown]
# ## Render one frame


def render_frame(t: float) -> np.ndarray:
    shot, center_lon, center_lat = shot_state(t)

    canvas = render_background(OUT_SIZE[0], OUT_SIZE[1], t)
    draw_globe_grid(canvas, center_lon, center_lat, t)
    draw_earthquake_pulses(canvas, center_lon, center_lat, t)
    draw_depth_core(canvas, t)
    draw_depth_legend(canvas, t)
    draw_monitor_trace(canvas, t)
    draw_catalog_timeline(canvas, t)
    draw_highlight_callout(canvas, center_lon, center_lat, t)
    add_text_layers(canvas, t, shot)

    arr = np.array(canvas.convert("RGB"))
    arr = apply_grade(arr)
    arr = np.clip(arr.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)

    fade_in = smoothstep(t / 0.9)
    fade_out = 1.0 - smoothstep((t - (CONFIG["duration_s"] - 1.15)) / 1.0)
    arr = np.clip(arr.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)
    return arr


print("Cinematic earthquake renderer ready.")

# %% [markdown]
# ## Preview frames

preview_times = [1.2, 10.0, 22.0, 35.0, 46.0, CONFIG["duration_s"] - 1.0]
preview_arrays = []
for t in tqdm(preview_times, desc="Preview frames"):
    arr = render_frame(float(t))
    preview_arrays.append(arr)
    Image.fromarray(arr).save(PREVIEW_DIR / f"preview_{int(t):02d}s.png")

fig, axes = plt.subplots(1, len(preview_arrays), figsize=(19, 10))
for ax, img, t in zip(axes, preview_arrays, preview_times):
    ax.imshow(img)
    ax.set_title(f"{t:.0f}s")
    ax.set_axis_off()
plt.tight_layout()
plt.savefig(PREVIEW_DIR / "preview_grid.png", dpi=170)
plt.show()

print("Preview images written to:", PREVIEW_DIR.resolve())

# %% [markdown]
# ## Subtitle sidecar


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(captions, path: Path):
    lines = []
    for idx, (start, end, text) in enumerate(captions, start=1):
        lines.append(str(idx))
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


SRT_PATH = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
if CONFIG.get("write_subtitle_sidecar", True):
    write_srt(CAPTIONS, SRT_PATH)
    print("Subtitle sidecar written:", SRT_PATH.resolve())

# %% [markdown]
# ## Render the full 9:16 MP4
#
# This is the expensive section. Use the fast test settings near the top before
# committing to 1080x1920 at 24 fps.

RAW_VIDEO_PATH = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
SUBBED_VIDEO_PATH = OUTPUT_ROOT / f"{CONFIG['output_basename']}_subbed.mp4"
AUDIO_VIDEO_PATH = OUTPUT_ROOT / f"{CONFIG['output_basename']}_with_audio.mp4"
FINAL_VIDEO_PATH = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"

nframes = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
times = np.arange(nframes) / CONFIG["fps"]

print(f"Rendering {nframes:,} frames at {CONFIG['video_width']}x{CONFIG['video_height']} ...")
with iio.get_writer(
    RAW_VIDEO_PATH,
    fps=CONFIG["fps"],
    codec="libx264",
    quality=8,
    pixelformat="yuv420p",
    macro_block_size=None,
) as writer:
    for t in tqdm(times, desc="Rendering video"):
        writer.append_data(render_frame(float(t)))

print("Raw video written:", RAW_VIDEO_PATH.resolve())

# %% [markdown]
# ## Optional: burn subtitles or add audio
#
# Use only audio you have permission to use.


def run_ffmpeg(cmd: List[str]):
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


ffmpeg = find_ffmpeg()
print("FFmpeg detected:", ffmpeg)

final_candidate = RAW_VIDEO_PATH

if CONFIG.get("burn_subtitles", False) and ffmpeg and SRT_PATH.exists():
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(final_candidate),
        "-vf", f"subtitles={SRT_PATH}:force_style=Fontname=DejaVu Sans Mono,Fontsize=21,Outline=1.1,BorderStyle=3,MarginV=88",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(SUBBED_VIDEO_PATH),
    ]
    run_ffmpeg(cmd)
    final_candidate = SUBBED_VIDEO_PATH
    print("Subtitled video written:", SUBBED_VIDEO_PATH.resolve())

audio_path = CONFIG.get("audio_path")
if audio_path and Path(audio_path).exists() and ffmpeg:
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(final_candidate),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(AUDIO_VIDEO_PATH),
    ]
    run_ffmpeg(cmd)
    final_candidate = AUDIO_VIDEO_PATH
    print("Audio-muxed video written:", AUDIO_VIDEO_PATH.resolve())
elif audio_path:
    print("audio_path was set, but the file was not found or ffmpeg was unavailable. Skipping audio.")

if final_candidate.exists():
    shutil.copyfile(final_candidate, FINAL_VIDEO_PATH)
    print("Final video:", FINAL_VIDEO_PATH.resolve())

print("Output directory:", OUTPUT_ROOT.resolve())
for path in sorted(OUTPUT_ROOT.glob("*")):
    print("-", path.name)

# %% [markdown]
# # Suggested narration / YouTube Shorts description
#
# Suggested voiceover:
#
# > This is one year of earthquakes, compressed into less than a minute.
# > Every pulse begins at a real catalog epicenter.
# > Red events are shallow. Amber events are deeper. Cyan events descend hundreds of kilometers into Earth.
# > Magnitude changes the scale of the pulse, while the timeline replays the catalog in order.
# > The strongest earthquakes become landmarks, but thousands of smaller events keep the planet in motion.
# > Earth is never perfectly still.
# > The catalog turns that motion into data.
#
# Suggested Shorts caption:
#
# A cinematic replay of real USGS earthquake catalog data. Epicenter position,
# origin time, depth, and magnitude come from the earthquake catalog; the radar,
# shock rings, and monitor trace are data-driven visual effects for storytelling.
#
# #Earthquake #USGS #DataVisualization #ScienceShorts #Geology #Seismology #Python
