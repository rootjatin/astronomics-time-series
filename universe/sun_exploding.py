# %% [markdown]
# # The Sun Has Been "Exploding" All Month — Data-Driven YouTube Short
#
# Creates a vertical 1080x1920 cinematic solar-activity Short from real event data.
#
# Reproducible fallback snapshot:
# - Major events verified from NOAA/SWPC news posts for 2026-06-11 to 2026-07-11.
#
# Scientific language:
# - "Exploding" is a headline metaphor. The measured events are solar flares.
# - A flare is electromagnetic radiation; a CME is expelled plasma and magnetic field.
# - Not every flare launches a CME, and not every CME is directed toward Earth.
# - Solar surface, loops, plasma jets, camera motion, stars and Earth are cinematic.
# - Event dates, classes and listed active-region numbers come from the data records.
# - Screen-space event locations are editorial when a source location is unavailable.
#
# Install:
#     pip install numpy pandas pillow imageio imageio-ffmpeg requests tqdm
#
# Fast test:
#     CONFIG["video_width"] = 540
#     CONFIG["video_height"] = 960
#     CONFIG["fps"] = 12
#     CONFIG["duration_s"] = 18

from __future__ import annotations

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


# %% [markdown]
# ## Configuration

OUTPUT_ROOT = Path("sun_exploding_all_month_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict = {
    "video_width": 1080,
    "video_height": 1920,
    "fps": 24,
    "duration_s": 58,
    "output_basename": "the_sun_has_been_exploding_all_month",

    # Fixed 30-day story window for reproducibility.
    "window_start": "2026-06-11",
    "window_end": "2026-07-11",

    # Live sources. NASA_API_KEY can replace DEMO_KEY through the environment.
    "donki_flr_url": "https://api.nasa.gov/DONKI/FLR",
    "noaa_goes_flares_url": (
        "https://services.swpc.noaa.gov/json/goes/primary/"
        "xray-flares-7-day.json"
    ),
    "request_timeout": (15, 90),
    "request_retries": 4,
    "retry_backoff_s": 1.2,

    # Render.
    "sun_radius_px": 355,
    "sun_center_y": 875,
    "surface_texture_size": 640,
    "corona_ray_count": 150,
    "star_count": 330,
    "surface_cell_count": 125,
    "plasma_particle_count": 360,
    "contrast_boost": 1.12,
    "saturation_boost": 1.13,
    "vignette_strength": 0.26,

    # Story.
    "title_text": "THE SUN HAS BEEN\n\"EXPLODING\" ALL MONTH",
    "subtitle_text": "30 days of recorded solar flare activity",
    "credit_text": "Event data: NASA DONKI + NOAA/SWPC GOES and event reports",
    "scientific_note": (
        "Headline uses 'exploding' as a metaphor. Flares are radiation bursts; "
        "CMEs are plasma and magnetic-field expulsions. Event locations are "
        "editorial when the source record has no disk coordinate."
    ),

    # Optional finishing.
    "audio_path": None,
    "burn_subtitles": False,
    "write_subtitle_sidecar": True,
}

OUT_SIZE = (int(CONFIG["video_width"]), int(CONFIG["video_height"]))


# %% [markdown]
# ## Reproducible NOAA-verified fallback snapshot

# These are not intended to replace a complete GOES/DONKI flare catalog. They
# keep the renderer useful offline and anchor the major events in this story.
FALLBACK_EVENTS = [
    {
        "event_id": "NOAA-20260621-M6.8",
        "begin_time": "2026-06-21T20:55:00Z",
        "peak_time": "2026-06-21T21:12:00Z",
        "end_time": "2026-06-21T21:35:00Z",
        "class_type": "M6.8",
        "active_region": 4473,
        "source_location": "",
        "cme_status": "possible; Earth-directed component unfavorable",
        "source": "NOAA/SWPC event report",
        "source_note": "R2 event from AR 4473; possible CME radio signature.",
    },
    {
        "event_id": "NOAA-20260630-M5.8",
        "begin_time": "2026-06-30T12:44:00Z",
        "peak_time": "2026-06-30T12:57:00Z",
        "end_time": "2026-06-30T13:20:00Z",
        "class_type": "M5.8",
        "active_region": np.nan,
        "source_location": "west limb",
        "cme_status": "no CME indication in initial NOAA assessment",
        "source": "NOAA/SWPC event report",
        "source_note": "M5.8 flare at 12:57 UTC.",
    },
    {
        "event_id": "NOAA-20260630-X1.1",
        "begin_time": "2026-06-30T20:35:00Z",
        "peak_time": "2026-06-30T20:50:00Z",
        "end_time": "2026-06-30T21:10:00Z",
        "class_type": "X1.1",
        "active_region": 4479,
        "source_location": "north center disk",
        "cme_status": "under assessment in initial NOAA report",
        "source": "NOAA/SWPC event report",
        "source_note": "R3-Strong radio blackout flare.",
    },
    {
        "event_id": "NOAA-20260704-X1.3",
        "begin_time": "2026-07-04T20:25:00Z",
        "peak_time": "2026-07-04T20:41:00Z",
        "end_time": "2026-07-04T21:05:00Z",
        "class_type": "X1.3",
        "active_region": 4482,
        "source_location": "",
        "cme_status": "not specified in cited NOAA flare notice",
        "source": "NOAA/SWPC event report",
        "source_note": "R3-Strong radio blackout flare.",
    },
]


# %% [markdown]
# ## Data helpers

def build_retry_session(config: Dict) -> requests.Session:
    retries = Retry(
        total=int(config.get("request_retries", 4)),
        connect=int(config.get("request_retries", 4)),
        read=int(config.get("request_retries", 4)),
        status=int(config.get("request_retries", 4)),
        backoff_factor=float(config.get("retry_backoff_s", 1.2)),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retries, pool_connections=2, pool_maxsize=2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "solar-flare-month-short/1.0 educational-visualization"
    })
    return session


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def deterministic_unit(text: str, salt: str = "") -> float:
    digest = hashlib.sha256(f"{text}|{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def parse_utc(value) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def flare_class_to_flux(class_type: str) -> float:
    """Convert GOES class label to nominal W/m^2 peak flux."""
    if not class_type:
        return np.nan
    text = str(class_type).strip().upper()
    bases = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
    letter = text[:1]
    if letter not in bases:
        return np.nan
    value = safe_float(text[1:], 1.0)
    return bases[letter] * value


def flux_to_flare_class(flux: float) -> str:
    if not np.isfinite(flux) or flux <= 0:
        return "?"
    bands = [(1e-4, "X"), (1e-5, "M"), (1e-6, "C"), (1e-7, "B"), (1e-8, "A")]
    for base, letter in bands:
        if flux >= base:
            return f"{letter}{flux / base:.1f}"
    return f"A{flux / 1e-8:.1f}"


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_id", "begin_time", "peak_time", "end_time", "class_type",
        "active_region", "source_location", "cme_status", "source", "source_note",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan if col == "active_region" else ""

    out = df[columns].copy()
    for col in ("begin_time", "peak_time", "end_time"):
        out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    out["peak_time"] = out["peak_time"].fillna(out["begin_time"])
    out["begin_time"] = out["begin_time"].fillna(out["peak_time"])
    out["end_time"] = out["end_time"].fillna(out["peak_time"] + pd.Timedelta(minutes=25))
    out["class_type"] = out["class_type"].fillna("").astype(str).str.upper().str.strip()
    out["peak_flux_w_m2"] = out["class_type"].map(flare_class_to_flux)
    out["active_region"] = pd.to_numeric(out["active_region"], errors="coerce")
    for col in ("event_id", "source_location", "cme_status", "source", "source_note"):
        out[col] = out[col].fillna("").astype(str).str.strip()

    out = out.dropna(subset=["peak_time", "peak_flux_w_m2"])
    out = out[out["peak_flux_w_m2"] >= 1e-6].copy()  # C-class and above.
    out["event_key"] = (
        out["peak_time"].dt.floor("min").astype(str)
        + "|" + out["class_type"]
    )
    out = (
        out.sort_values(["peak_time", "peak_flux_w_m2"], ascending=[True, False])
        .drop_duplicates("event_key")
        .reset_index(drop=True)
    )
    return out


def fetch_donki_flares(config: Dict, session: requests.Session) -> pd.DataFrame:
    api_key = os.getenv("NASA_API_KEY", "DEMO_KEY")
    params = {
        "startDate": config["window_start"],
        "endDate": config["window_end"],
        "api_key": api_key,
    }
    response = session.get(
        config["donki_flr_url"], params=params, timeout=config["request_timeout"]
    )
    if response.status_code >= 400:
        raise RuntimeError(f"NASA DONKI returned HTTP {response.status_code}")
    payload = response.json()
    (DATA_ROOT / "nasa_donki_flr_raw.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    rows = []
    for item in payload:
        linked = item.get("linkedEvents") or []
        linked_types = sorted({str(v.get("activityID", ""))[:3] for v in linked})
        has_cme = any(str(v.get("activityID", "")).startswith("CME") for v in linked)
        rows.append({
            "event_id": item.get("flrID") or "",
            "begin_time": item.get("beginTime"),
            "peak_time": item.get("peakTime"),
            "end_time": item.get("endTime"),
            "class_type": item.get("classType") or "",
            "active_region": item.get("activeRegionNum"),
            "source_location": item.get("sourceLocation") or "",
            "cme_status": "linked CME in DONKI" if has_cme else "no linked CME in DONKI record",
            "source": "NASA DONKI FLR API",
            "source_note": ", ".join(linked_types),
        })
    return normalize_events(pd.DataFrame(rows))


def fetch_noaa_goes_flares(config: Dict, session: requests.Session) -> pd.DataFrame:
    response = session.get(
        config["noaa_goes_flares_url"], timeout=config["request_timeout"]
    )
    if response.status_code >= 400:
        raise RuntimeError(f"NOAA GOES JSON returned HTTP {response.status_code}")
    payload = response.json()
    (DATA_ROOT / "noaa_goes_flares_7day_raw.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    rows = []
    for item in payload:
        class_type = (
            item.get("max_class") or item.get("class_type") or
            flux_to_flare_class(safe_float(item.get("max_xrlong")))
        )
        rows.append({
            "event_id": item.get("event_id") or item.get("id") or "",
            "begin_time": item.get("begin_time") or item.get("beginTime"),
            "peak_time": item.get("max_time") or item.get("peak_time") or item.get("peakTime"),
            "end_time": item.get("end_time") or item.get("endTime"),
            "class_type": class_type,
            "active_region": item.get("region") or item.get("active_region"),
            "source_location": item.get("location") or "",
            "cme_status": "not supplied by GOES flare table",
            "source": "NOAA GOES primary X-ray flare JSON",
            "source_note": f"satellite={item.get('satellite', '')}",
        })
    return normalize_events(pd.DataFrame(rows))


def fallback_event_frame() -> pd.DataFrame:
    return normalize_events(pd.DataFrame(FALLBACK_EVENTS))


def gather_event_data(config: Dict, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_ROOT / "solar_flare_events_clean.csv"
    if cache_path.exists() and not force_refresh:
        cached = pd.read_csv(cache_path)
        return normalize_events(cached)

    frames: List[pd.DataFrame] = []
    session = build_retry_session(config)

    for label, loader in (
        ("NASA DONKI", fetch_donki_flares),
        ("NOAA GOES", fetch_noaa_goes_flares),
    ):
        try:
            frame = loader(config, session)
            if len(frame):
                frames.append(frame)
                print(f"{label}: {len(frame):,} flare records")
        except Exception as exc:
            print(f"{label} unavailable: {exc}")

    # Always merge the verified major-event snapshot; duplicates are removed.
    frames.append(fallback_event_frame())
    events = normalize_events(pd.concat(frames, ignore_index=True))

    start = pd.Timestamp(config["window_start"], tz="UTC")
    end = pd.Timestamp(config["window_end"], tz="UTC") + pd.Timedelta(days=1)
    events = events[(events["peak_time"] >= start) & (events["peak_time"] < end)].copy()
    events = events.sort_values("peak_time").reset_index(drop=True)

    save = events.copy()
    for col in ("begin_time", "peak_time", "end_time"):
        save[col] = save[col].astype(str)
    save.to_csv(cache_path, index=False)
    print("Clean event table:", cache_path.resolve())
    print("Events retained:", len(events))
    if len(events):
        print("Strongest:", events.loc[events["peak_flux_w_m2"].idxmax(), "class_type"])
    return events


# %% [markdown]
# ## Visual utilities

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


def get_font(size: int, bold: bool = False):
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
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
    image: Image.Image,
    text: str,
    xy: Tuple[float, float],
    size: int = 40,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    stroke: int = 2,
    anchor: str = "la",
):
    ImageDraw.Draw(image).text(
        xy, text, font=get_font(size, bold), fill=fill, anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(230, fill[3] if len(fill) > 3 else 230)),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int = 30,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    line_spacing: int = 7,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold)
    words = text.split()
    lines, current = [], ""
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
            (x, y), line, font=font, fill=fill, stroke_width=2,
            stroke_fill=(0, 0, 0, 220),
        )
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += bbox[3] - bbox[1] + line_spacing


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.85, 0.0, 1.0).astype(np.float32)


def find_ffmpeg() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


VIGNETTE = make_vignette(OUT_SIZE[0], OUT_SIZE[1], CONFIG["vignette_strength"])


# %% [markdown]
# ## Timeline and narration

CAPTIONS = [
    (0.4, 6.0, "The Sun has been 'exploding' all month — but that headline needs translation."),
    (6.3, 17.8, "These markers replay recorded solar flares across a 30-day window."),
    (18.1, 29.5, "M-class flares are strong. X-class flares are the most intense GOES category."),
    (29.8, 39.8, "June 30 produced an M5.8 — then an X1.1 only hours later."),
    (40.1, 48.8, "A flare is radiation. A CME is a cloud of plasma and magnetic field."),
    (49.0, 57.3, "Not every flare launches a CME. Not every CME is aimed at Earth."),
]

SHOT_PLAN = [
    {"name": "headline", "start": 0.0, "end": 6.2, "zoom0": 0.84, "zoom1": 1.03, "x0": 560, "x1": 520, "y0": 930, "y1": 885},
    {"name": "month", "start": 6.2, "end": 18.0, "zoom0": 1.03, "zoom1": 0.95, "x0": 520, "x1": 555, "y0": 885, "y1": 855},
    {"name": "classes", "start": 18.0, "end": 30.0, "zoom0": 0.95, "zoom1": 1.14, "x0": 555, "x1": 505, "y0": 855, "y1": 910},
    {"name": "double", "start": 30.0, "end": 40.0, "zoom0": 1.14, "zoom1": 1.06, "x0": 505, "x1": 580, "y0": 910, "y1": 865},
    {"name": "difference", "start": 40.0, "end": 49.0, "zoom0": 1.06, "zoom1": 0.78, "x0": 580, "x1": 390, "y0": 865, "y1": 870},
    {"name": "earth", "start": 49.0, "end": 58.0, "zoom0": 0.78, "zoom1": 0.66, "x0": 390, "x1": 330, "y0": 870, "y1": 890},
]


def get_shot(t: float) -> Dict:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def shot_state(t: float):
    shot = get_shot(t)
    u = (t - shot["start"]) / max(shot["end"] - shot["start"], 1e-6)
    e = ease_in_out_sine(u)
    return (
        shot,
        lerp(shot["x0"], shot["x1"], e),
        lerp(shot["y0"], shot["y1"], e),
        lerp(shot["zoom0"], shot["zoom1"], e),
    )


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


# %% [markdown]
# ## Cinematic scene

@dataclass
class ActiveRegionVisual:
    event_index: int
    seed_x: float
    seed_y: float
    phase: float


class SolarMonthScene:
    def __init__(self, events: pd.DataFrame):
        self.events = events.reset_index(drop=True).copy()
        self.window_start = pd.Timestamp(CONFIG["window_start"], tz="UTC")
        self.window_end = pd.Timestamp(CONFIG["window_end"], tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        self.window_seconds = (self.window_end - self.window_start).total_seconds()

        if len(self.events) == 0:
            raise RuntimeError("No flare events are available for rendering.")

        self.event_fraction = (
            (self.events["peak_time"] - self.window_start).dt.total_seconds()
            / self.window_seconds
        ).to_numpy(float)
        self.event_strength = np.log10(self.events["peak_flux_w_m2"].to_numpy(float))
        self.event_strength_norm = np.clip((self.event_strength + 6.0) / 2.2, 0.0, 1.0)

        self.stars = self._make_stars(CONFIG["star_count"], 918)
        self.corona = self._make_corona(CONFIG["corona_ray_count"], 442)
        self.surface_cells = self._make_surface_cells(CONFIG["surface_cell_count"], 774)
        self.particles = self._make_particles(CONFIG["plasma_particle_count"], 208)
        self.region_visuals = self._make_region_visuals()
        self.base_surface = self._make_base_surface_texture(CONFIG["surface_texture_size"], 167)

        self.strongest_index = int(self.events["peak_flux_w_m2"].idxmax())
        self.strongest_class = str(self.events.iloc[self.strongest_index]["class_type"])
        self.class_counts = {
            letter: int(self.events["class_type"].str.startswith(letter).sum())
            for letter in ("C", "M", "X")
        }

    @staticmethod
    def _make_stars(count: int, seed: int):
        rng = np.random.default_rng(seed)
        return [
            (float(rng.uniform(0, OUT_SIZE[0])), float(rng.uniform(0, OUT_SIZE[1])),
             float(rng.uniform(0.35, 1.8)), int(rng.integers(25, 120)),
             float(rng.uniform(0, 2 * math.pi)))
            for _ in range(count)
        ]

    @staticmethod
    def _make_corona(count: int, seed: int):
        rng = np.random.default_rng(seed)
        return [
            (float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(0.7, 1.55)),
             float(rng.uniform(0.25, 1.0)), float(rng.uniform(0, 2 * math.pi)))
            for _ in range(count)
        ]

    @staticmethod
    def _make_surface_cells(count: int, seed: int):
        rng = np.random.default_rng(seed)
        cells = []
        for _ in range(count):
            r = math.sqrt(float(rng.uniform(0, 1))) * 0.93
            a = float(rng.uniform(0, 2 * math.pi))
            cells.append((r * math.cos(a), r * math.sin(a), float(rng.uniform(0.02, 0.095)), float(rng.uniform(0, 2 * math.pi))))
        return cells

    @staticmethod
    def _make_particles(count: int, seed: int):
        rng = np.random.default_rng(seed)
        return [
            (float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(0.1, 1.0)),
             float(rng.uniform(0.4, 1.5)), float(rng.uniform(0, 2 * math.pi)))
            for _ in range(count)
        ]

    def _make_region_visuals(self) -> List[ActiveRegionVisual]:
        visuals = []
        for i, row in self.events.iterrows():
            key = str(row.get("active_region", "")) + str(row["event_id"])
            loc = str(row.get("source_location", "")).lower()
            if "west limb" in loc:
                seed_x = 0.84
            elif "center" in loc:
                seed_x = 0.05
            elif "east" in loc:
                seed_x = -0.65
            else:
                seed_x = deterministic_unit(key, "x") * 1.45 - 0.72
            if "north" in loc:
                seed_y = -0.30
            elif "south" in loc:
                seed_y = 0.32
            else:
                seed_y = deterministic_unit(key, "y") * 0.82 - 0.41
            visuals.append(ActiveRegionVisual(i, seed_x, seed_y, deterministic_unit(key, "phase") * 2 * math.pi))
        return visuals

    @staticmethod
    def _make_base_surface_texture(size: int, seed: int) -> Image.Image:
        rng = np.random.default_rng(seed)
        small = rng.normal(0.5, 0.20, (80, 80)).clip(0, 1)
        small_img = Image.fromarray(np.uint8(small * 255), "L")
        noise = small_img.resize((size, size), Image.Resampling.BICUBIC)
        noise = noise.filter(ImageFilter.GaussianBlur(3.0))
        return noise

    def playback_fraction(self, t: float) -> float:
        return smoothstep((t - 4.4) / (CONFIG["duration_s"] - 8.0))

    def playback_time(self, t: float) -> pd.Timestamp:
        return self.window_start + pd.to_timedelta(
            self.playback_fraction(t) * self.window_seconds, unit="s"
        )

    def active_events(self, t: float) -> List[int]:
        f = self.playback_fraction(t)
        delta = f - self.event_fraction
        active = np.where((delta >= -0.022) & (delta <= 0.050))[0]
        return [int(i) for i in active]

    def current_event(self, t: float) -> Optional[int]:
        active = self.active_events(t)
        if not active:
            return None
        return max(active, key=lambda i: self.events.iloc[i]["peak_flux_w_m2"])

    def render_background(self, t: float) -> Image.Image:
        canvas = Image.new("RGBA", OUT_SIZE, (1, 2, 7, 255))
        d = ImageDraw.Draw(canvas)
        for x, y, r, alpha, phase in self.stars:
            twinkle = 0.65 + 0.35 * math.sin(t * 0.8 + phase)
            drift_x = (x + 0.55 * t * math.sin(phase)) % OUT_SIZE[0]
            d.ellipse((drift_x-r, y-r, drift_x+r, y+r), fill=(215, 225, 255, int(alpha * twinkle)))

        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for cx, cy, radius, color in [
            (850, 390, 610, (95, 12, 20)),
            (170, 1420, 520, (40, 10, 55)),
            (620, 930, 750, (100, 28, 5)),
        ]:
            hd.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(*color, 25))
        canvas.alpha_composite(haze.filter(ImageFilter.GaussianBlur(105)))
        return canvas

    def draw_corona(self, canvas: Image.Image, cx: float, cy: float, radius: float, t: float):
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for scale, alpha in ((1.62, 18), (1.35, 30), (1.18, 55), (1.06, 88)):
            rr = radius * scale
            gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=(255, 80, 12, alpha))
        for angle, length, width, phase in self.corona:
            pulse = 0.72 + 0.28 * math.sin(t * 0.45 + phase)
            inner = radius * 0.92
            outer = radius * (1.03 + 0.40 * length * pulse)
            bend = 0.06 * math.sin(t * 0.3 + phase)
            p0 = (cx + math.cos(angle) * inner, cy + math.sin(angle) * inner)
            p1 = (cx + math.cos(angle + bend) * outer, cy + math.sin(angle + bend) * outer)
            gd.line((*p0, *p1), fill=(255, 100, 25, int(16 + 30 * width)), width=max(1, int(1 + width * 2)))
        canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(26)))

    def make_solar_disk(self, radius: int, t: float) -> Image.Image:
        diameter = max(64, int(radius * 2))
        texture = self.base_surface.resize((diameter, diameter), Image.Resampling.BICUBIC)
        arr = np.asarray(texture, dtype=np.float32) / 255.0
        yy, xx = np.mgrid[0:diameter, 0:diameter]
        nx = (xx - diameter / 2) / max(radius, 1)
        ny = (yy - diameter / 2) / max(radius, 1)
        rr = np.sqrt(nx * nx + ny * ny)
        mask = rr <= 1.0
        limb = np.clip(np.sqrt(np.maximum(0.0, 1.0 - rr * rr)), 0, 1)
        bands = 0.09 * np.sin(nx * 22 + t * 0.42) + 0.055 * np.sin(ny * 39 - t * 0.25)
        granular = np.clip(0.72 + 0.34 * arr + bands, 0, 1.35)
        brightness = granular * (0.48 + 0.70 * limb)
        r = np.clip(255 * brightness * 1.28, 0, 255)
        g = np.clip(255 * brightness * 0.42, 0, 220)
        b = np.clip(255 * brightness * 0.075, 0, 90)
        a = np.where(mask, 255, 0)
        rgba = np.dstack([r, g, b, a]).astype(np.uint8)
        disk = Image.fromarray(rgba, "RGBA")

        cell_layer = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
        cd = ImageDraw.Draw(cell_layer)
        for sx, sy, sr, phase in self.surface_cells:
            x = diameter / 2 + sx * radius
            y = diameter / 2 + sy * radius
            local_r = sr * radius * (0.78 + 0.22 * math.sin(t * 0.5 + phase))
            alpha = int(18 + 25 * (0.5 + 0.5 * math.sin(t * 0.7 + phase)))
            cd.ellipse((x-local_r, y-local_r, x+local_r, y+local_r), outline=(255, 214, 80, alpha), width=max(1, int(radius / 220)))
        disk.alpha_composite(cell_layer.filter(ImageFilter.GaussianBlur(max(0.5, radius / 300))))
        return disk

    def event_disk_position(self, event_index: int, playback_date: pd.Timestamp, radius: float) -> Tuple[float, float]:
        visual = self.region_visuals[event_index]
        event_date = self.events.iloc[event_index]["peak_time"]
        days = (playback_date - event_date).total_seconds() / 86400.0
        # Approximate synodic rotation for cinematic disk tracking. It affects only
        # screen position; it is not presented as a measured heliographic solution.
        x0 = visual.seed_x
        angle0 = math.asin(np.clip(x0, -0.95, 0.95))
        angle = angle0 + math.radians(13.2 * days)
        x = math.sin(angle)
        y = visual.seed_y * (0.96 - 0.12 * math.cos(angle))
        return x * radius * 0.86, y * radius * 0.83

    def draw_sunspots(self, canvas: Image.Image, cx: float, cy: float, radius: float, t: float):
        date = self.playback_time(t)
        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for visual in self.region_visuals:
            event_date = self.events.iloc[visual.event_index]["peak_time"]
            days = abs((date - event_date).total_seconds() / 86400.0)
            if days > 13.8:
                continue
            dx, dy = self.event_disk_position(visual.event_index, date, radius)
            if abs(dx) > radius * 0.87:
                continue
            strength = self.event_strength_norm[visual.event_index]
            spot_r = radius * (0.012 + 0.018 * strength)
            x, y = cx + dx, cy + dy
            for ox, oy, scale in ((0, 0, 1), (spot_r*1.5, spot_r*0.25, 0.55), (-spot_r*1.1, -spot_r*0.4, 0.42)):
                rr = spot_r * scale
                d.ellipse((x+ox-rr*1.7, y+oy-rr, x+ox+rr*1.7, y+oy+rr), fill=(35, 7, 2, 185), outline=(255, 120, 20, 80), width=1)
        canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(1.0)))

    def draw_flare(self, canvas: Image.Image, cx: float, cy: float, radius: float, event_index: int, t: float):
        f = self.playback_fraction(t)
        ef = self.event_fraction[event_index]
        local = (f - ef + 0.022) / 0.072
        if local < 0 or local > 1:
            return
        strength = float(self.event_strength_norm[event_index])
        date = self.playback_time(t)
        dx, dy = self.event_disk_position(event_index, date, radius)
        x, y = cx + dx, cy + dy
        radial = np.array([dx, dy], dtype=float)
        norm = float(np.linalg.norm(radial))
        if norm < 1:
            radial = np.array([0.2, -1.0])
            norm = float(np.linalg.norm(radial))
        radial /= norm
        tangent = np.array([-radial[1], radial[0]])
        envelope = smoothstep(local / 0.20) * (1.0 - smoothstep((local - 0.72) / 0.28))
        alpha = int(255 * envelope)
        if alpha < 3:
            return

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        sharp = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sharp)

        burst = radius * (0.08 + 0.16 * strength) * (0.65 + 0.8 * math.sin(math.pi * local))
        gd.ellipse((x-burst, y-burst, x+burst, y+burst), fill=(255, 245, 170, int(alpha * 0.70)))

        # Magnetic-loop motif.
        loop_count = 3 + int(4 * strength)
        for k in range(loop_count):
            phase = self.region_visuals[event_index].phase + k * 0.77
            side = (k - (loop_count - 1) / 2) * radius * 0.025
            base = np.array([x, y]) + tangent * side
            height = radius * (0.18 + 0.32 * strength) * (0.75 + 0.25 * math.sin(phase))
            width = radius * (0.08 + 0.10 * strength)
            pts = []
            for j in range(31):
                u = j / 30
                p = base + tangent * ((u - 0.5) * 2 * width) - radial * (math.sin(math.pi * u) * height)
                pts.append((float(p[0]), float(p[1])))
            gd.line(pts, fill=(255, 70, 20, int(alpha * 0.44)), width=max(4, int(7 * strength + 3)))
            sd.line(pts, fill=(255, 230, 135, int(alpha * 0.88)), width=max(1, int(2 + 2 * strength)))

        # Radial jet and particles.
        jet_len = radius * (0.26 + 0.65 * strength) * smoothstep(local / 0.45)
        tip = np.array([x, y]) + radial * jet_len
        gd.line((x, y, float(tip[0]), float(tip[1])), fill=(255, 80, 20, int(alpha * 0.72)), width=max(8, int(18 * strength)))
        sd.line((x, y, float(tip[0]), float(tip[1])), fill=(255, 248, 205, alpha), width=max(2, int(4 + 4 * strength)))

        for idx, (a, speed, scale, phase) in enumerate(self.particles):
            if idx % max(1, 7 - int(4 * strength)) != 0:
                continue
            fan = (a % 1.0 - 0.5) * 0.65
            direction = radial * math.cos(fan) + tangent * math.sin(fan)
            travel = jet_len * speed * local * (0.8 + 0.2 * math.sin(phase))
            p = np.array([x, y]) + direction * travel
            rr = 1.0 + scale * (1.2 + strength * 2.5)
            sd.ellipse((p[0]-rr, p[1]-rr, p[0]+rr, p[1]+rr), fill=(255, 230, 150, int(alpha * 0.72)))

        canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(18)))
        canvas.alpha_composite(sharp)

    def draw_sun(self, canvas: Image.Image, cx: float, cy: float, zoom: float, t: float):
        radius = float(CONFIG["sun_radius_px"]) * zoom
        self.draw_corona(canvas, cx, cy, radius, t)
        disk = self.make_solar_disk(int(radius), t)
        canvas.alpha_composite(disk, (int(cx-radius), int(cy-radius)))
        self.draw_sunspots(canvas, cx, cy, radius, t)
        for i in self.active_events(t):
            self.draw_flare(canvas, cx, cy, radius, i, t)
        return radius

    def event_curve(self, samples: int = 360) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0, 1, samples)
        # Event-marker reconstruction, not continuous GOES flux.
        y = np.full(samples, 1.8e-7, dtype=float)
        for ef, flux in zip(self.event_fraction, self.events["peak_flux_w_m2"]):
            width = 0.009 + 0.006 * np.clip(np.log10(flux) + 6, 0, 2)
            y = np.maximum(y, 1.8e-7 + flux * np.exp(-0.5 * ((x - ef) / width) ** 2))
        return x, y

    def draw_xray_timeline(self, canvas: Image.Image, t: float):
        x0, x1 = 65, OUT_SIZE[0] - 65
        y0, y1 = OUT_SIZE[1] - 470, OUT_SIZE[1] - 285
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rounded_rectangle((40, y0-75, OUT_SIZE[0]-40, y1+70), radius=25, fill=(2, 4, 10, 174), outline=(255, 105, 35, 75), width=1)
        canvas.alpha_composite(panel)
        d = ImageDraw.Draw(canvas)

        for flux, label in ((1e-6, "C"), (1e-5, "M"), (1e-4, "X")):
            yy = y1 - (math.log10(flux) + 7.0) / 3.2 * (y1-y0)
            d.line((x0, yy, x1, yy), fill=(255, 145, 70, 55), width=1)
            draw_text(canvas, label, (x0-15, yy), size=18, fill=(255, 175, 95, 210), bold=True, stroke=1, anchor="ra")

        xs, ys = self.event_curve()
        pts = []
        for xf, flux in zip(xs, ys):
            px = x0 + xf * (x1-x0)
            py = y1 - np.clip((math.log10(max(flux, 1e-8)) + 7.0) / 3.2, 0, 1) * (y1-y0)
            pts.append((float(px), float(py)))
        d.line(pts, fill=(255, 115, 35, 230), width=3)

        for i, row in self.events.iterrows():
            px = x0 + self.event_fraction[i] * (x1-x0)
            flux = row["peak_flux_w_m2"]
            py = y1 - np.clip((math.log10(max(flux, 1e-8)) + 7.0) / 3.2, 0, 1) * (y1-y0)
            rr = 3 + 4 * self.event_strength_norm[i]
            d.ellipse((px-rr, py-rr, px+rr, py+rr), fill=(255, 238, 180, 245), outline=(255, 85, 25, 245), width=2)

        cursor_x = x0 + self.playback_fraction(t) * (x1-x0)
        d.line((cursor_x, y0-12, cursor_x, y1+16), fill=(255, 240, 190, 245), width=3)
        draw_text(canvas, "FLARE EVENT INTENSITY // LOG GOES CLASS SCALE", (x0, y0-55), size=20, fill=(255, 196, 130, 220), bold=True, stroke=1)
        draw_text(canvas, "Event-peak reconstruction; not a continuous irradiance trace", (x0, y1+28), size=16, fill=(185, 190, 205, 205), stroke=1)
        draw_text(canvas, CONFIG["window_start"], (x0, y1+52), size=17, fill=(190, 205, 220, 210), stroke=1)
        draw_text(canvas, CONFIG["window_end"], (x1, y1+52), size=17, fill=(190, 205, 220, 210), stroke=1, anchor="ra")

    def draw_event_callout(self, canvas: Image.Image, t: float):
        idx = self.current_event(t)
        if idx is None:
            return
        row = self.events.iloc[idx]
        f = self.playback_fraction(t)
        local = (f - self.event_fraction[idx] + 0.018) / 0.060
        alpha = int(245 * smoothstep(local / 0.20) * (1 - smoothstep((local - 0.68) / 0.28)))
        if alpha < 4:
            return
        x0, y0 = 55, 245
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(panel)
        d.rounded_rectangle((x0, y0, x0+490, y0+242), radius=22, fill=(3, 4, 9, 190), outline=(255, 118, 40, 125), width=2)
        canvas.alpha_composite(panel)
        draw_text(canvas, "GOES FLARE EVENT", (x0+22, y0+18), size=20, fill=(255, 160, 75, alpha), bold=True, stroke=1)
        draw_text(canvas, str(row["class_type"]), (x0+22, y0+55), size=58, fill=(255, 246, 205, alpha), bold=True, stroke=2)
        ar = "AR unknown" if not np.isfinite(row["active_region"]) else f"AR {int(row['active_region'])}"
        draw_text(canvas, ar, (x0+190, y0+76), size=27, fill=(255, 190, 105, alpha), bold=True, stroke=1)
        draw_text(canvas, row["peak_time"].strftime("%Y-%m-%d  %H:%M UTC"), (x0+22, y0+133), size=22, fill=(235, 240, 248, alpha), bold=True, stroke=1)
        note = str(row["cme_status"] or "CME relationship not supplied")
        draw_wrapped_text(canvas, f"CME note: {note}", (x0+22, y0+174), 440, size=18, fill=(200, 215, 230, alpha))

    def draw_class_panel(self, canvas: Image.Image, t: float):
        if not (17.4 <= t <= 40.6):
            return
        alpha = int(225 * smoothstep((t-17.4)/2.0) * (1-smoothstep((t-38.2)/2.4)))
        x0, y0 = OUT_SIZE[0]-365, 250
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(panel)
        d.rounded_rectangle((x0, y0, OUT_SIZE[0]-45, y0+264), radius=22, fill=(2, 4, 10, 172), outline=(255, 115, 40, 85), width=1)
        canvas.alpha_composite(panel)
        draw_text(canvas, "EVENTS IN DATA TABLE", (x0+20, y0+20), size=19, fill=(255, 180, 105, alpha), bold=True, stroke=1)
        y = y0+65
        rows = [("C", "COMMON"), ("M", "STRONG"), ("X", "MOST INTENSE")]
        rows = [(letter, label) for letter, label in rows if self.class_counts[letter] > 0]
        for letter, label in rows:
            draw_text(canvas, letter, (x0+20, y), size=37, fill=(255, 235, 190, alpha), bold=True, stroke=1)
            draw_text(canvas, str(self.class_counts[letter]), (x0+92, y+3), size=30, fill=(255, 135, 55, alpha), bold=True, stroke=1)
            draw_text(canvas, label, (x0+145, y+8), size=17, fill=(190, 210, 225, alpha), bold=True, stroke=1)
            y += 59
        draw_text(canvas, f"STRONGEST // {self.strongest_class}", (x0+20, y0+224), size=18, fill=(255, 190, 110, alpha), bold=True, stroke=1)

    def draw_flare_cme_explainer(self, canvas: Image.Image, t: float):
        alpha = int(240 * smoothstep((t-39.5)/2.2) * (1-smoothstep((t-49.5)/2.0)))
        if alpha < 4:
            return
        x0, y0 = 530, 300
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(panel)
        d.rounded_rectangle((x0, y0, OUT_SIZE[0]-45, y0+560), radius=25, fill=(2, 4, 10, 187), outline=(255, 130, 45, 95), width=1)
        d.line((x0+245, y0+80, x0+245, y0+510), fill=(255, 135, 55, 70), width=2)
        canvas.alpha_composite(panel)
        draw_text(canvas, "FLARE", (x0+38, y0+28), size=34, fill=(255, 240, 200, alpha), bold=True)
        draw_text(canvas, "CME", (x0+290, y0+28), size=34, fill=(255, 240, 200, alpha), bold=True)
        draw_wrapped_text(canvas, "Electromagnetic radiation burst", (x0+28, y0+102), 200, size=22, fill=(255, 182, 105, alpha), bold=True)
        draw_wrapped_text(canvas, "Plasma plus embedded magnetic field", (x0+275, y0+102), 215, size=22, fill=(255, 182, 105, alpha), bold=True)
        draw_text(canvas, "~8 MIN", (x0+36, y0+245), size=39, fill=(255, 242, 205, alpha), bold=True)
        draw_wrapped_text(canvas, "Light-speed effects reach Earth's sunlit atmosphere", (x0+28, y0+300), 205, size=19, fill=(205, 220, 235, alpha))
        draw_text(canvas, "15–96 H", (x0+284, y0+245), size=38, fill=(255, 242, 205, alpha), bold=True)
        draw_wrapped_text(canvas, "Possible travel time for an Earth-directed CME", (x0+275, y0+300), 215, size=19, fill=(205, 220, 235, alpha))
        draw_wrapped_text(canvas, "Association is common — not automatic.", (x0+28, y0+455), 465, size=22, fill=(255, 175, 95, alpha), bold=True)

    def draw_earth_scene(self, canvas: Image.Image, t: float, sun_cx: float, sun_cy: float, sun_radius: float):
        if t < 47.5:
            return
        alpha = int(245 * smoothstep((t-47.5)/2.5))
        earth_x, earth_y, earth_r = 860, 905, 78
        glow = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        for scale, a in ((1.55, 20), (1.28, 40), (1.10, 80)):
            rr = earth_r*scale
            gd.ellipse((earth_x-rr, earth_y-rr, earth_x+rr, earth_y+rr), fill=(40, 135, 255, int(a*alpha/255)))
        canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(18)))
        d = ImageDraw.Draw(canvas)
        d.ellipse((earth_x-earth_r, earth_y-earth_r, earth_x+earth_r, earth_y+earth_r), fill=(16, 68, 132, alpha), outline=(105, 225, 255, alpha), width=3)
        d.arc((earth_x-earth_r*0.9, earth_y-earth_r*0.35, earth_x+earth_r*0.9, earth_y+earth_r*0.35), 0, 180, fill=(80, 190, 230, int(alpha*0.6)), width=2)
        d.ellipse((earth_x+8, earth_y-earth_r, earth_x+earth_r*1.45, earth_y+earth_r), fill=(1, 4, 14, int(alpha*0.82)))

        # Radiation wave reaches Earth quickly.
        u = smoothstep((t-49.0)/3.8)
        wave_x = lerp(sun_cx + sun_radius*0.85, earth_x-earth_r, u)
        d.line((sun_cx+sun_radius*0.85, sun_cy, wave_x, earth_y), fill=(255, 245, 185, int(alpha*0.85)), width=3)
        for k in range(3):
            rr = 14 + 16*k + 8*math.sin(t*4+k)
            d.arc((wave_x-rr, earth_y-rr, wave_x+rr, earth_y+rr), -70, 70, fill=(255, 190, 85, int(alpha*(0.75-k*0.17))), width=2)

        # Slower, broader CME shell as a conceptual comparison only.
        cme_u = smoothstep((t-51.0)/7.0)
        shell_x = lerp(sun_cx+sun_radius*0.95, earth_x-earth_r-25, cme_u)
        shell_h = 145 + 60*cme_u
        d.arc((shell_x-45, earth_y-shell_h, shell_x+45, earth_y+shell_h), 90, 270, fill=(255, 85, 35, int(alpha*0.65)), width=7)
        draw_text(canvas, "RADIATION // ~8 MIN", (560, 690), size=20, fill=(255, 235, 185, alpha), bold=True, stroke=1)
        draw_text(canvas, "CME // HOURS TO DAYS", (560, 1160), size=20, fill=(255, 145, 80, alpha), bold=True, stroke=1)
        draw_wrapped_text(canvas, "Concept diagram — not a trajectory for a specific event", (565, 1200), 420, size=17, fill=(185, 200, 220, alpha))

    def draw_header_and_caption(self, canvas: Image.Image, t: float, shot: Dict):
        title_alpha = int(255 * smoothstep((t-0.25)/1.1) * (1-smoothstep((t-5.6)/0.7)))
        if title_alpha > 4:
            lines = CONFIG["title_text"].split("\n")
            draw_text(canvas, lines[0], (55, 95), size=47, fill=(255, 248, 220, title_alpha), bold=True)
            draw_text(canvas, lines[1], (55, 157), size=55, fill=(255, 110, 40, title_alpha), bold=True)
            draw_text(canvas, CONFIG["subtitle_text"], (58, 230), size=24, fill=(255, 195, 125, min(title_alpha, 225)), bold=True)
        else:
            labels = {
                "month": "30-DAY SOLAR ACTIVITY REPLAY",
                "classes": "GOES X-RAY FLARE CLASSES",
                "double": "JUNE 30 // TWO MAJOR FLARES",
                "difference": "FLARE IS NOT THE SAME AS CME",
                "earth": "WHAT CAN REACH EARTH?",
                "headline": "SOLAR ACTIVITY DATA REPLAY",
            }
            draw_text(canvas, labels.get(shot["name"], "SOLAR ACTIVITY"), (55, 68), size=21, fill=(255, 185, 105, 215), bold=True, stroke=1)

        date = self.playback_time(t)
        draw_text(canvas, date.strftime("%Y-%m-%d"), (OUT_SIZE[0]-55, 75), size=30, fill=(245, 248, 255, 240), bold=True, anchor="ra")
        draw_text(canvas, "UTC PLAYBACK DATE", (OUT_SIZE[0]-55, 116), size=17, fill=(175, 195, 215, 205), bold=True, stroke=1, anchor="ra")

        caption = caption_at(t)
        if caption:
            y0 = OUT_SIZE[1]-248
            panel = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
            d = ImageDraw.Draw(panel)
            d.rounded_rectangle((45,y0,OUT_SIZE[0]-45,y0+130), radius=24, fill=(2,4,10,178), outline=(255,110,40,72), width=1)
            canvas.alpha_composite(panel)
            draw_wrapped_text(canvas, caption, (72,y0+28), OUT_SIZE[0]-145, size=30, fill=(248,250,255,245))

        note_alpha = int(215 * smoothstep((t-51.5)/3.0))
        if note_alpha > 4:
            draw_text(canvas, CONFIG["credit_text"], (60, OUT_SIZE[1]-104), size=18, fill=(220,228,240,note_alpha), stroke=1)
            draw_wrapped_text(canvas, CONFIG["scientific_note"], (60, OUT_SIZE[1]-76), 950, size=15, fill=(180,198,218,note_alpha))

    def draw_scanlines(self, canvas: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        d = ImageDraw.Draw(overlay)
        offset = int(t*37)%7
        for y in range(offset, OUT_SIZE[1], 7):
            d.line((0,y,OUT_SIZE[0],y), fill=(255,160,90,10), width=1)
        scan_y = int((t*145)%(OUT_SIZE[1]+220))-110
        d.rectangle((0,scan_y,OUT_SIZE[0],scan_y+44), fill=(255,115,55,7))
        canvas.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot, cx, cy, zoom = shot_state(t)
        canvas = self.render_background(t)
        sun_radius = self.draw_sun(canvas, cx, cy, zoom, t)
        if shot["name"] not in ("difference", "earth"):
            self.draw_xray_timeline(canvas, t)
        self.draw_event_callout(canvas, t)
        self.draw_class_panel(canvas, t)
        self.draw_flare_cme_explainer(canvas, t)
        self.draw_earth_scene(canvas, t, cx, cy, sun_radius)
        self.draw_header_and_caption(canvas, t, shot)
        self.draw_scanlines(canvas, t)

        image = ImageEnhance.Contrast(canvas.convert("RGB")).enhance(CONFIG["contrast_boost"])
        image = ImageEnhance.Color(image).enhance(CONFIG["saturation_boost"])
        arr = np.asarray(image, dtype=np.float32)
        arr *= VIGNETTE[..., None]
        fade = smoothstep(t/0.85) * (1-smoothstep((t-(CONFIG["duration_s"]-1.1))/1.0))
        return np.clip(arr*fade, 0, 255).astype(np.uint8)


# %% [markdown]
# ## Subtitles, previews and render

def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds*1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float,float,str]], path: Path):
    lines = []
    for i,(start,end,text) in enumerate(captions,1):
        lines.extend([str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def create_data_summary(events: pd.DataFrame):
    summary = {
        "window_start": CONFIG["window_start"],
        "window_end": CONFIG["window_end"],
        "event_count": int(len(events)),
        "class_counts": {letter: int(events["class_type"].str.startswith(letter).sum()) for letter in ("C","M","X")},
        "strongest_class": str(events.loc[events["peak_flux_w_m2"].idxmax(), "class_type"]),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "The plotted curve is reconstructed from event peaks, not continuous GOES irradiance.",
            "The renderer tries NASA DONKI and NOAA GOES live data, then merges verified fallback events.",
        ],
    }
    (DATA_ROOT / "render_data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def create_previews(scene: SolarMonthScene):
    times = [1.2, 10.0, 22.0, 34.0, 44.0, 55.0]
    images = []
    for t in tqdm(times, desc="Preview frames"):
        arr = scene.render_frame(t)
        path = PREVIEW_DIR / f"preview_{int(t):02d}s.png"
        Image.fromarray(arr).save(path)
        images.append(Image.fromarray(arr).resize((270,480), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (810, 960), (6,7,11))
    for i,img in enumerate(images):
        sheet.paste(img, ((i%3)*270, (i//3)*480))
    sheet_path = OUTPUT_ROOT / "sun_exploding_all_month_preview_contact_sheet.jpg"
    sheet.save(sheet_path, quality=92)
    print("Contact sheet:", sheet_path.resolve())
    return sheet_path


def run_ffmpeg(command: List[str]):
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def render_video(scene: SolarMonthScene):
    raw_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    if CONFIG.get("write_subtitle_sidecar", True):
        write_srt(CAPTIONS, srt_path)

    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    times = np.arange(frame_count) / CONFIG["fps"]
    with iio.get_writer(raw_path, fps=CONFIG["fps"], codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None) as writer:
        for t in tqdm(times, desc="Rendering solar month short"):
            writer.append_data(scene.render_frame(float(t)))

    candidate = raw_path
    ffmpeg = find_ffmpeg()
    if CONFIG.get("burn_subtitles", False) and ffmpeg and srt_path.exists():
        subbed = OUTPUT_ROOT / f"{CONFIG['output_basename']}_subbed.mp4"
        run_ffmpeg([ffmpeg,"-y","-i",str(candidate),"-vf",f"subtitles={srt_path}:force_style=Fontname=DejaVu Sans,Fontsize=22,Outline=1.2,BorderStyle=3,MarginV=90","-c:v","libx264","-pix_fmt","yuv420p",str(subbed)])
        candidate = subbed

    audio_path = CONFIG.get("audio_path")
    if audio_path and Path(audio_path).exists() and ffmpeg:
        with_audio = OUTPUT_ROOT / f"{CONFIG['output_basename']}_with_audio.mp4"
        run_ffmpeg([ffmpeg,"-y","-i",str(candidate),"-i",str(audio_path),"-c:v","copy","-c:a","aac","-b:a","192k","-shortest",str(with_audio)])
        candidate = with_audio

    shutil.copyfile(candidate, final_path)
    print("Final video:", final_path.resolve())
    return final_path


def render_quick_preview(scene: SolarMonthScene, path: Path, width=360, height=640, fps=6, duration=8.0):
    # Samples the complete narrative in a short low-resolution motion check.
    frames = int(round(fps*duration))
    with iio.get_writer(path, fps=fps, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=None) as writer:
        for i in tqdm(range(frames), desc="Quick preview"):
            source_t = (i/max(frames-1,1))*(CONFIG["duration_s"]-0.8)+0.4
            frame = Image.fromarray(scene.render_frame(float(source_t))).resize((width,height), Image.Resampling.LANCZOS)
            writer.append_data(np.asarray(frame))
    return path


# %% [markdown]
# ## Main

def main():
    print("Starting data-driven solar flare Short pipeline ...")
    events = gather_event_data(CONFIG, force_refresh=False)
    create_data_summary(events)
    scene = SolarMonthScene(events)
    create_previews(scene)
    render_video(scene)
    print("Output directory:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()


# %% [markdown]
# ## Suggested narration
#
# The Sun has been "exploding" all month — but that headline needs translation.
# These are solar flares: sudden releases of electromagnetic radiation.
# This timeline replays the recorded flare events across thirty days.
# M-class means strong. X-class is the most intense GOES category.
# On June thirtieth, an M5.8 flare was followed only hours later by an X1.1.
# Then, on July fourth, Region 4482 produced an X1.3 flare.
# But a flare is not the same thing as a coronal mass ejection.
# Flare radiation reaches Earth in about eight minutes.
# An Earth-directed CME can take from hours to several days.
# Not every flare launches a CME — and not every CME is aimed at Earth.
# The Sun is active. The important part is measuring exactly what it released.
#
# Suggested caption:
# Thirty days of solar flare records turned into a cinematic space-weather replay.
# Event dates and GOES classes are data-driven; surface motion and eruption geometry
# are cinematic. "Exploding" is a headline metaphor, not a literal scientific term.
#
# #Sun #SolarFlare #SpaceWeather #NASA #NOAA #Astronomy #Python
