# %% [markdown]
# # SPACETIME COLLISIONS — YouTube Short from Real GWOSC Gravitational-Wave Events
#
# This script creates a vertical 1080×1920 cinematic astronomy short from
# public gravitational-wave event catalogs served by the Gravitational Wave
# Open Science Center (GWOSC).
#
# Theme: SPACETIME COLLISIONS — compact-object mergers detected by LIGO/Virgo/KAGRA
#
# Real catalog values used when available:
# - event name and GPS event time,
# - detector network,
# - component source-frame masses,
# - chirp mass,
# - final source-frame mass,
# - luminosity distance,
# - redshift,
# - effective inspiral spin chi_eff,
# - astrophysical probability p_astro,
# - false-alarm rate,
# - network matched-filter SNR.
#
# Visual language:
# - warped spacetime grid,
# - chronological event replay,
# - expanding gravitational-wave ripple fronts,
# - binary inspiral / merger glyphs,
# - source mass controls merger scale,
# - network SNR controls signal intensity,
# - luminosity distance controls visual depth compression,
# - chirp mass controls the editorial waveform acceleration,
# - detector-network HUD,
# - loudest-event locks and mass leaderboard,
# - CRT scanlines, bloom, vignette, dust and camera drift.
#
# Scientific fidelity note:
# Event names, GPS times, detector lists and preferred/default catalog parameters
# come from GWOSC event-catalog API responses.
#
# Screen position, grid deformation, ripple geometry and the rendered waveform
# are editorial science-communication graphics. The bottom "chirp" is NOT detector
# strain and is NOT a parameter-estimation waveform. It is a deterministic visual
# mapping driven by catalog chirp mass and SNR.
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
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from tqdm.auto import tqdm
from urllib3.util.retry import Retry


# %% [markdown]
# ## Configuration

OUTPUT_ROOT = Path("spacetime_collisions_short_output")
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
    "output_basename": "gwosc_spacetime_collisions_short",

    # GWOSC event API
    "gwosc_catalog_api": "https://gwosc.org/api/v2/catalogs/{catalog}/events",
    "catalogs": [
        "GWTC-1-confident",
        "GWTC-2.1-confident",
        "GWTC-3-confident",
        "GWTC-4.1",
    ],
    "catalog_priority": {
        "GWTC-1-confident": 1,
        "GWTC-2.1-confident": 2,
        "GWTC-3-confident": 3,
        "GWTC-4.1": 4,
    },
    "request_timeout": (20, 180),
    "request_retries": 5,
    "retry_backoff_s": 1.8,
    "api_page_size": 200,

    # Visual limits
    "max_events_for_video": 260,
    "max_simultaneous_events": 12,
    "highlight_count": 14,
    "leaderboard_count": 6,

    # Renderer
    "background_star_count": 700,
    "dust_particle_count": 150,
    "hud_noise_count": 70,
    "grid_vertical_lines": 11,
    "grid_horizontal_lines": 18,
    "grid_samples_per_line": 42,
    "vignette_strength": 0.29,
    "contrast_boost": 1.15,
    "saturation_boost": 1.08,

    # Text
    "title_text": "SPACETIME COLLISIONS",
    "subtitle_text": "Gravitational-wave events // GWOSC catalog replay",
    "credit_text": "Data: Gravitational Wave Open Science Center (GWOSC)",
    "scientific_note": (
        "Catalog event parameters drive mass, distance and signal-intensity mappings. "
        "Grid warps, screen positions and the rendered chirp graphic are editorial."
    ),

    # Optional finishing
    "audio_path": None,
    "burn_subtitles": False,
    "write_subtitle_sidecar": True,
}

OUT_SIZE = (CONFIG["video_width"], CONFIG["video_height"])


# %% [markdown]
# ## Network and catalog helpers

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
        "User-Agent": "spacetime-collisions-short/1.1 (educational visualization)",
        # GWOSC uses content negotiation for its browsable API.  Ask for the
        # machine-readable renderer explicitly so a successful HTTP 200 cannot
        # silently become the HTML documentation page.
        "Accept": "application/json",
    })
    return session


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def deterministic_unit(text: str, salt: str = "") -> float:
    digest = hashlib.sha256(f"{text}|{salt}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64 - 1)


def parse_event_calendar_time(name: str, gps: float) -> pd.Timestamp:
    """
    Parse UTC-like date/time from the public GW event name for display/order labels.

    Modern names commonly use GWYYMMDD_HHMMSS.
    Early named events such as GW150914 encode only the UTC date.
    GPS remains the canonical numeric event-time field retained in the dataframe.
    """
    name = str(name).strip()

    full_match = re.fullmatch(r"GW(\d{6})_(\d{6})", name)
    if full_match:
        date_text, time_text = full_match.groups()
        parsed = pd.to_datetime(
            f"20{date_text} {time_text}",
            format="%Y%m%d %H%M%S",
            utc=True,
            errors="coerce",
        )
        if pd.notna(parsed):
            return parsed

    date_match = re.fullmatch(r"GW(\d{6})", name)
    if date_match:
        date_text = date_match.group(1)
        parsed = pd.to_datetime(
            f"20{date_text}",
            format="%Y%m%d",
            utc=True,
            errors="coerce",
        )
        if pd.notna(parsed):
            return parsed

    # GPS is retained separately and still sorts the events correctly.
    # This fallback is intentionally approximate for calendar display only.
    return pd.Timestamp("2000-01-01", tz="UTC") + pd.to_timedelta(
        max(safe_float(gps, 0.0), 0.0),
        unit="s",
    )


def parameter_dict(event: Dict) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for item in event.get("default_parameters") or []:
        name = str(item.get("name") or "").strip()
        if name:
            result[name] = safe_float(item.get("best"))
    return result


def request_catalog_pages(
    session: requests.Session,
    catalog: str,
    config: Dict,
) -> List[Dict]:
    url = config["gwosc_catalog_api"].format(catalog=catalog)
    params = {
        # IMPORTANT: format=api is GWOSC's human-facing browsable API renderer
        # and may return text/html.  format=json is the raw JSON view.
        "format": "json",
        "include-default-parameters": "true",
        "pagesize": str(int(config.get("api_page_size", 200))),
    }

    events: List[Dict] = []
    next_url: Optional[str] = url
    next_params: Optional[Dict] = params

    while next_url:
        try:
            response = session.get(
                next_url,
                params=next_params,
                timeout=config["request_timeout"],
            )
        except RequestException as exc:
            raise RuntimeError(
                f"GWOSC request failed for {catalog}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise RuntimeError(
                f"GWOSC HTTP {response.status_code} for {catalog}: {response.reason}\n"
                f"Response preview:\n{response.text[:1600]}"
            )

        content_type = str(response.headers.get("Content-Type") or "").lower()
        body_prefix = response.content.lstrip()[:32].lower()

        if "application/json" not in content_type or body_prefix.startswith(
            (b"<!doctype html", b"<html")
        ):
            raise RuntimeError(
                "GWOSC returned a non-JSON response for "
                f"{catalog}.\n"
                f"Requested URL: {response.url}\n"
                f"Content-Type: {content_type or '<missing>'}\n"
                "The downloader requires GWOSC's raw JSON renderer "
                "(format=json), not the browsable HTML API page.\n"
                f"Response preview:\n{response.text[:1600]}"
            )

        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"GWOSC returned malformed JSON for {catalog}.\n"
                f"Requested URL: {response.url}\n"
                f"Content-Type: {content_type or '<missing>'}\n"
                f"Response preview:\n{response.text[:1600]}"
            ) from exc

        rows = payload.get("results")
        if rows is None:
            raise RuntimeError(
                f"Unexpected GWOSC catalog payload for {catalog}:\n"
                f"{json.dumps(payload)[:1600]}"
            )

        events.extend(rows)
        print(f"{catalog}: downloaded {len(events):,} event rows")

        next_url = payload.get("next")
        next_params = None
        if next_url:
            # Older/alternate renderer links can carry format=api.  Keep every
            # pagination hop on the raw JSON renderer.
            next_url = str(next_url).replace("format=api", "format=json")
            time.sleep(0.20)

    return events


def flatten_event(event: Dict, catalog: str, config: Dict) -> Dict:
    params = parameter_dict(event)

    name = str(event.get("name") or "").strip()
    gps = safe_float(event.get("gps"))
    detectors = event.get("detectors") or []

    row = {
        "name": name,
        "short_name": str(event.get("shortName") or "").strip(),
        "gps": gps,
        "version": safe_float(event.get("version")),
        "catalog": catalog,
        "catalog_priority": int(config["catalog_priority"].get(catalog, 0)),
        "detectors": ",".join(str(d) for d in detectors),
        "detector_count": len(detectors),
        "detail_url": str(event.get("detail_url") or ""),
    }

    parameter_names = [
        "mass_1_source",
        "mass_2_source",
        "chirp_mass_source",
        "final_mass_source",
        "luminosity_distance",
        "redshift",
        "chi_eff",
        "far",
        "p_astro",
        "network_matched_filter_snr",
    ]

    for parameter_name in parameter_names:
        row[parameter_name] = params.get(parameter_name, np.nan)

    return row


def fetch_gwosc_events(
    config: Dict,
    force_refresh: bool = False,
) -> pd.DataFrame:
    raw_path = DATA_ROOT / "gwosc_catalog_events_raw.json"
    clean_path = DATA_ROOT / "gwosc_catalog_events_clean.csv"

    if clean_path.exists() and raw_path.exists() and not force_refresh:
        print("Using cached GWOSC event catalog data.")
        cached = pd.read_csv(clean_path)
        return clean_event_dataframe(cached, save_path=None)

    session = build_retry_session(config)
    raw_by_catalog: Dict[str, List[Dict]] = {}
    flat_rows: List[Dict] = []

    try:
        for catalog in config["catalogs"]:
            print(f"Requesting GWOSC catalog: {catalog}")
            rows = request_catalog_pages(session, catalog, config)
            raw_by_catalog[catalog] = rows
            flat_rows.extend(
                flatten_event(event, catalog, config)
                for event in rows
            )
    except Exception as exc:
        if raw_path.exists() and not force_refresh:
            print("Live GWOSC request failed; using cached raw JSON.")
            print("Reason:", exc)
            raw_by_catalog = json.loads(raw_path.read_text(encoding="utf-8"))
            flat_rows = []
            for catalog, rows in raw_by_catalog.items():
                flat_rows.extend(
                    flatten_event(event, catalog, config)
                    for event in rows
                )
        else:
            raise

    if not flat_rows:
        raise RuntimeError("No gravitational-wave event rows were returned by GWOSC.")

    raw_path.write_text(
        json.dumps(raw_by_catalog, indent=2),
        encoding="utf-8",
    )

    frame = pd.DataFrame(flat_rows)
    return clean_event_dataframe(frame, save_path=clean_path)


def classify_compact_binary(row: pd.Series) -> str:
    """
    Simplified visualization bucket from source-frame component masses.

    This is an editorial rule for the short, not an official GWOSC source label:
    - both component best values <= 3 M_sun -> BNS-like
    - one <= 3 and one > 3 -> NSBH-like
    - both > 3 -> BBH-like
    - missing mass values -> no-PE / unclassified
    """
    m1 = safe_float(row.get("mass_1_source"))
    m2 = safe_float(row.get("mass_2_source"))

    if not np.isfinite(m1) or not np.isfinite(m2):
        return "no PE / unclassified"

    lower = min(m1, m2)
    upper = max(m1, m2)

    if upper <= 3.0:
        return "BNS-like"
    if lower <= 3.0 < upper:
        return "NSBH-like"
    return "BBH-like"


HIGHLIGHT_EVENTS = {
    "GW150914": "FIRST DIRECT DETECTION",
    "GW170817": "BINARY NEUTRON STAR",
    "GW190425": "MASSIVE BNS-LIKE EVENT",
    "GW190521": "HIGH-MASS MERGER",
    "GW190814": "ASYMMETRIC MASS SYSTEM",
    "GW200105_162426": "NSBH-LIKE EVENT",
    "GW200115_042309": "NSBH-LIKE EVENT",
    "GW230529_181500": "O4 EVENT",
    "GW231123_135430": "VERY HIGH-MASS O4 EVENT",
}


def clean_event_dataframe(
    df: pd.DataFrame,
    save_path: Optional[Path],
) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = [
        "gps",
        "version",
        "catalog_priority",
        "detector_count",
        "mass_1_source",
        "mass_2_source",
        "chirp_mass_source",
        "final_mass_source",
        "luminosity_distance",
        "redshift",
        "chi_eff",
        "far",
        "p_astro",
        "network_matched_filter_snr",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan

    required = ["name", "gps", "catalog"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"GWOSC dataframe is missing required fields: {missing}")

    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["catalog"] = df["catalog"].fillna("").astype(str).str.strip()
    df["detectors"] = df.get("detectors", "").fillna("").astype(str)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["gps"]).copy()
    df = df[(df["gps"] > 0) & (df["name"].str.len() > 0)].copy()

    # The same event can appear in multiple historical releases.
    # Keep the newest configured catalog priority, then the highest event version.
    df = (
        df.sort_values(
            ["name", "catalog_priority", "version"],
            ascending=[True, False, False],
        )
        .drop_duplicates("name", keep="first")
        .reset_index(drop=True)
    )

    df["event_time"] = [
        parse_event_calendar_time(name, gps)
        for name, gps in zip(df["name"], df["gps"])
    ]

    df["source_group"] = df.apply(classify_compact_binary, axis=1)
    df["highlight_label"] = df["name"].map(HIGHLIGHT_EVENTS)

    # Useful derived values.
    df["total_mass_source"] = (
        df["mass_1_source"] + df["mass_2_source"]
    )

    # Catalog-driven visual attention score.
    snr = df["network_matched_filter_snr"].fillna(7.0).clip(lower=0)
    total_mass = df["total_mass_source"].fillna(0).clip(lower=0)
    p_astro = df["p_astro"].fillna(0.5).clip(0, 1)
    detector_count = df["detector_count"].fillna(1).clip(lower=1)

    df["highlight_score"] = (
        0.85 * snr
        + 0.055 * total_mass
        + 4.0 * p_astro
        + 0.8 * detector_count
        + 25.0 * df["highlight_label"].notna().astype(float)
    )

    # Deterministic editorial layout.
    df["visual_angle"] = df["name"].apply(
        lambda name: deterministic_unit(name, "angle") * 2.0 * math.pi
    )
    df["visual_radius"] = df["name"].apply(
        lambda name: 0.20 + 0.78 * math.sqrt(deterministic_unit(name, "radius"))
    )
    df["visual_phase"] = df["name"].apply(
        lambda name: deterministic_unit(name, "phase") * 2.0 * math.pi
    )
    df["visual_lane"] = df["name"].apply(
        lambda name: deterministic_unit(name, "lane") * 2.0 - 1.0
    )

    df = df.sort_values(["gps", "name"]).reset_index(drop=True)

    if save_path is not None:
        save_df = df.copy()
        save_df["event_time"] = save_df["event_time"].astype(str)
        save_df.to_csv(save_path, index=False)
        print("Saved cleaned GWOSC CSV:", save_path)

    print(f"Unique gravitational-wave events retained: {len(df):,}")
    print("Source visualization groups:")
    print(df["source_group"].value_counts(dropna=False))

    mass_rows = df["total_mass_source"].dropna()
    if len(mass_rows):
        print(
            "Source total-mass best-value range:",
            f"{mass_rows.min():.2f}–{mass_rows.max():.2f} M_sun",
        )

    snr_rows = df["network_matched_filter_snr"].dropna()
    if len(snr_rows):
        print(
            "Network SNR best-value range:",
            f"{snr_rows.min():.1f}–{snr_rows.max():.1f}",
        )

    return df


def select_video_events(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    max_events = int(config["max_events_for_video"])
    if len(df) <= max_events:
        return df.copy().reset_index(drop=True)

    highlight_n = min(max_events // 3, 90)
    highlights = (
        df.sort_values("highlight_score", ascending=False)
        .head(highlight_n)
    )

    remaining = max_events - len(highlights)
    timeline_indices = np.linspace(
        0,
        len(df) - 1,
        max(remaining, 1),
        dtype=int,
    )
    timeline_sample = df.iloc[timeline_indices]

    selected = pd.concat([highlights, timeline_sample], ignore_index=True)
    selected = (
        selected.drop_duplicates("name")
        .sort_values(["gps", "name"])
        .head(max_events)
        .reset_index(drop=True)
    )
    return selected


# %% [markdown]
# ## Scientific preview plots

def create_scientific_previews(df: pd.DataFrame):
    years = df["event_time"].dt.year

    fig, ax = plt.subplots(figsize=(10, 5))
    counts = years.value_counts().sort_index()
    if len(counts):
        ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("GWOSC events retained by event year")
    ax.set_xlabel("Event year")
    ax.set_ylabel("Event count")
    plt.tight_layout()
    path = PREVIEW_DIR / "events_by_year.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)

    mass_df = df.dropna(subset=["mass_1_source", "mass_2_source"]).copy()
    fig, ax = plt.subplots(figsize=(8, 7))
    if len(mass_df):
        for group, sdf in mass_df.groupby("source_group"):
            ax.scatter(
                sdf["mass_1_source"],
                sdf["mass_2_source"],
                s=24,
                alpha=0.65,
                label=group,
            )
    ax.set_title("Source-frame component mass best values")
    ax.set_xlabel("Mass 1 [solar masses]")
    ax.set_ylabel("Mass 2 [solar masses]")
    if len(mass_df):
        ax.legend(fontsize=8)
    plt.tight_layout()
    path = PREVIEW_DIR / "component_mass_map.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)

    snr_df = df.dropna(
        subset=["luminosity_distance", "network_matched_filter_snr"]
    ).copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    if len(snr_df):
        ax.scatter(
            snr_df["luminosity_distance"],
            snr_df["network_matched_filter_snr"],
            s=25,
            alpha=0.60,
        )
        ax.set_xscale("log")
    ax.set_title("Network SNR vs luminosity distance")
    ax.set_xlabel("Luminosity distance [Mpc], log scale")
    ax.set_ylabel("Network matched-filter SNR")
    plt.tight_layout()
    path = PREVIEW_DIR / "snr_vs_distance.png"
    plt.savefig(path, dpi=170)
    plt.close(fig)

    print("Scientific previews written to:", PREVIEW_DIR.resolve())


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


def format_mass(value: float) -> str:
    if not np.isfinite(value):
        return "-- M_sun"
    if value >= 100:
        return f"{value:.0f} M_sun"
    if value >= 10:
        return f"{value:.1f} M_sun"
    return f"{value:.2f} M_sun"


def format_distance(value: float) -> str:
    if not np.isfinite(value):
        return "distance unavailable"
    if value >= 1000:
        return f"{value / 1000.0:.2f} Gpc"
    return f"{value:.0f} Mpc"


def format_snr(value: float) -> str:
    if not np.isfinite(value):
        return "SNR --"
    return f"SNR {value:.1f}"


VIGNETTE = make_vignette(
    OUT_SIZE[0],
    OUT_SIZE[1],
    CONFIG["vignette_strength"],
)


# %% [markdown]
# ## Cinematic timeline

CAPTIONS = [
    (0.5, 5.8, "These are not pictures of black holes. They are events measured through gravitational waves."),
    (6.4, 17.5, "Each pulse replays a cataloged event in chronological order."),
    (18.0, 29.5, "Component masses control the scale of the merging pair in this display."),
    (30.0, 41.5, "Higher network SNR burns brighter. Chirp mass drives the speed of the visual signal sweep."),
    (42.0, 50.5, "Some collisions leave an enormous final compact object after the merger."),
    (51.0, 57.3, "The universe was always ringing. Our detectors learned how to listen."),
]

SHOT_PLAN = [
    {
        "name": "boot",
        "start": 0.0,
        "end": 6.0,
        "center_x_start": 560,
        "center_x_end": 540,
        "center_y_start": 1010,
        "center_y_end": 970,
        "zoom_start": 0.88,
        "zoom_end": 1.00,
        "caption": "GWOSC LINK // SPACETIME EVENT DISPLAY",
    },
    {
        "name": "replay",
        "start": 6.0,
        "end": 19.0,
        "center_x_start": 540,
        "center_x_end": 500,
        "center_y_start": 970,
        "center_y_end": 930,
        "zoom_start": 1.00,
        "zoom_end": 1.06,
        "caption": "CHRONOLOGICAL REPLAY // DETECTION EVENTS",
    },
    {
        "name": "mass",
        "start": 19.0,
        "end": 31.0,
        "center_x_start": 500,
        "center_x_end": 565,
        "center_y_start": 930,
        "center_y_end": 960,
        "zoom_start": 1.06,
        "zoom_end": 1.12,
        "caption": "SOURCE-FRAME MASS // MERGER SCALE",
    },
    {
        "name": "chirp",
        "start": 31.0,
        "end": 42.0,
        "center_x_start": 565,
        "center_x_end": 520,
        "center_y_start": 960,
        "center_y_end": 1010,
        "zoom_start": 1.12,
        "zoom_end": 1.02,
        "caption": "SIGNAL HUD // SNR + CHIRP-MASS MAPPING",
    },
    {
        "name": "lock",
        "start": 42.0,
        "end": 51.0,
        "center_x_start": 520,
        "center_x_end": 585,
        "center_y_start": 1010,
        "center_y_end": 955,
        "zoom_start": 1.02,
        "zoom_end": 1.15,
        "caption": "HIGH-ENERGY EVENTS // TARGET LOCK",
    },
    {
        "name": "outro",
        "start": 51.0,
        "end": CONFIG["duration_s"],
        "center_x_start": 585,
        "center_x_end": 540,
        "center_y_start": 955,
        "center_y_end": 1020,
        "zoom_start": 1.15,
        "zoom_end": 0.82,
        "caption": "THE UNIVERSE // STILL RINGING",
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

    center_x = lerp(shot["center_x_start"], shot["center_x_end"], e)
    center_y = lerp(shot["center_y_start"], shot["center_y_end"], e)
    zoom = lerp(shot["zoom_start"], shot["zoom_end"], e)

    return shot, center_x, center_y, zoom


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


# %% [markdown]
# ## Spacetime collision renderer

GROUP_COLORS = {
    "BBH-like": (120, 205, 255, 235),
    "BNS-like": (255, 225, 145, 240),
    "NSBH-like": (235, 155, 255, 238),
    "no PE / unclassified": (180, 195, 225, 175),
}


class SpacetimeCollisionScene:
    def __init__(self, event_df: pd.DataFrame):
        self.events = event_df.reset_index(drop=True).copy()

        if len(self.events) < 2:
            raise RuntimeError("At least two GW events are required for the replay.")

        self.gps = self.events["gps"].to_numpy(float)
        self.gps_start = float(np.nanmin(self.gps))
        self.gps_end = float(np.nanmax(self.gps))
        self.gps_span = max(self.gps_end - self.gps_start, 1.0)

        # Blend calendar spacing with rank spacing.
        # Order remains chronological, while the rank term prevents years of
        # observational downtime from leaving the short visually empty.
        date_fraction = (self.gps - self.gps_start) / self.gps_span
        rank_fraction = np.linspace(0.0, 1.0, len(self.events))
        self.event_fractions = 0.22 * date_fraction + 0.78 * rank_fraction

        self.names = self.events["name"].to_numpy(object)
        self.catalogs = self.events["catalog"].to_numpy(object)
        self.detectors = self.events["detectors"].to_numpy(object)
        self.detector_count = self.events["detector_count"].fillna(0).to_numpy(float)
        self.source_groups = self.events["source_group"].to_numpy(object)
        self.highlight_labels = self.events["highlight_label"].to_numpy(object)

        self.mass_1 = self.events["mass_1_source"].to_numpy(float)
        self.mass_2 = self.events["mass_2_source"].to_numpy(float)
        self.total_mass = self.events["total_mass_source"].to_numpy(float)
        self.chirp_mass = self.events["chirp_mass_source"].to_numpy(float)
        self.final_mass = self.events["final_mass_source"].to_numpy(float)
        self.distance = self.events["luminosity_distance"].to_numpy(float)
        self.redshift = self.events["redshift"].to_numpy(float)
        self.chi_eff = self.events["chi_eff"].to_numpy(float)
        self.p_astro = self.events["p_astro"].to_numpy(float)
        self.far = self.events["far"].to_numpy(float)
        self.snr = self.events["network_matched_filter_snr"].to_numpy(float)

        self.visual_angle = self.events["visual_angle"].to_numpy(float)
        self.visual_radius = self.events["visual_radius"].to_numpy(float)
        self.visual_phase = self.events["visual_phase"].to_numpy(float)
        self.visual_lane = self.events["visual_lane"].to_numpy(float)

        finite_snr = self.snr[np.isfinite(self.snr)]
        self.snr_min = float(np.min(finite_snr)) if len(finite_snr) else 6.0
        self.snr_max = float(np.max(finite_snr)) if len(finite_snr) else 25.0

        finite_mass = self.total_mass[np.isfinite(self.total_mass)]
        self.mass_max = float(np.max(finite_mass)) if len(finite_mass) else 120.0

        finite_distance = self.distance[np.isfinite(self.distance) & (self.distance > 0)]
        self.distance_max = float(np.max(finite_distance)) if len(finite_distance) else 5000.0

        self.highlight_indices = (
            self.events["highlight_score"]
            .sort_values(ascending=False)
            .head(CONFIG["highlight_count"])
            .index
            .to_numpy(int)
        )

        leaderboard = self.events.dropna(subset=["final_mass_source"]).copy()
        self.leaderboard_indices = (
            leaderboard.sort_values("final_mass_source", ascending=False)
            .head(CONFIG["leaderboard_count"])
            .index
            .to_numpy(int)
        )

        self.stars = self._make_stars(CONFIG["background_star_count"], seed=11)
        self.dust = self._make_dust(CONFIG["dust_particle_count"], seed=33)
        self.hud_noise = self._make_hud_noise(CONFIG["hud_noise_count"], seed=77)

    @staticmethod
    def _make_stars(count: int, seed: int):
        rng = np.random.default_rng(seed)
        stars = []
        for _ in range(count):
            stars.append({
                "x": float(rng.uniform(0, OUT_SIZE[0])),
                "y": float(rng.uniform(0, OUT_SIZE[1])),
                "r": float(rng.uniform(0.4, 1.8)),
                "a": int(rng.integers(18, 105)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "drift": float(rng.uniform(-13, 13)),
            })
        return stars

    @staticmethod
    def _make_dust(count: int, seed: int):
        rng = np.random.default_rng(seed)
        dust = []
        for _ in range(count):
            dust.append({
                "x": float(rng.uniform(0, OUT_SIZE[0])),
                "y": float(rng.uniform(0, OUT_SIZE[1])),
                "r": float(rng.uniform(0.7, 3.0)),
                "a": int(rng.integers(10, 42)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "speed": float(rng.uniform(7, 28)),
            })
        return dust

    @staticmethod
    def _make_hud_noise(count: int, seed: int):
        rng = np.random.default_rng(seed)
        noise = []
        for _ in range(count):
            noise.append({
                "x": float(rng.uniform(0, OUT_SIZE[0])),
                "y": float(rng.uniform(0, OUT_SIZE[1])),
                "length": float(rng.uniform(8, 90)),
                "alpha": int(rng.integers(8, 42)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            })
        return noise

    def replay_fraction(self, t: float) -> float:
        start_t = 4.8
        end_t = CONFIG["duration_s"] - 4.2
        return smoothstep((t - start_t) / max(end_t - start_t, 1e-6))

    def replay_event_index(self, t: float) -> int:
        fraction = self.replay_fraction(t)
        return int(
            np.clip(
                np.searchsorted(self.event_fractions, fraction, side="right") - 1,
                0,
                len(self.events) - 1,
            )
        )

    def active_event_indices(self, t: float) -> np.ndarray:
        fraction = self.replay_fraction(t)
        screen_window = 0.052
        delta = fraction - self.event_fractions

        active = np.where(
            (delta >= -0.014)
            & (delta <= screen_window)
        )[0]

        if len(active) > int(CONFIG["max_simultaneous_events"]):
            score = self.events.iloc[active]["highlight_score"].to_numpy(float)
            order = np.argsort(score)[::-1]
            active = active[order[: int(CONFIG["max_simultaneous_events"])]]

        return active

    def local_phase(self, event_index: int, t: float) -> float:
        fraction = self.replay_fraction(t)
        start_pad = 0.014
        screen_window = 0.052
        local = (
            fraction - self.event_fractions[event_index] + start_pad
        ) / (screen_window + start_pad)
        return clamp(local)

    def event_screen_position(
        self,
        event_index: int,
        center_x: float,
        center_y: float,
        zoom: float,
        t: float,
    ) -> Tuple[float, float, float]:
        angle = self.visual_angle[event_index] + 0.045 * math.sin(
            t * 0.17 + self.visual_phase[event_index]
        )

        distance = self.distance[event_index]
        if np.isfinite(distance) and distance > 0:
            depth = math.log1p(distance) / math.log1p(max(self.distance_max, 1.0))
        else:
            depth = 0.68

        radius = (
            135.0
            + self.visual_radius[event_index] * 560.0
            + 70.0 * depth
        ) * zoom

        x = center_x + math.cos(angle) * radius
        y = center_y + math.sin(angle) * radius * 1.22
        scale = 1.18 - 0.50 * depth

        return x, y, scale

    def snr_strength(self, event_index: int) -> float:
        value = self.snr[event_index]
        if not np.isfinite(value):
            return 0.42
        return clamp(
            (math.sqrt(max(value, 0.0)) - math.sqrt(max(self.snr_min, 0.0)))
            / max(
                math.sqrt(max(self.snr_max, 0.0))
                - math.sqrt(max(self.snr_min, 0.0)),
                1e-6,
            ),
            0.08,
            1.0,
        )

    def mass_strength(self, event_index: int) -> float:
        value = self.total_mass[event_index]
        if not np.isfinite(value):
            return 0.30
        return clamp(
            math.log1p(max(value, 0.0)) / math.log1p(max(self.mass_max, 1.0)),
            0.08,
            1.0,
        )

    def render_background(self, t: float) -> Image.Image:
        canvas = Image.new("RGBA", OUT_SIZE, (2, 2, 9, 255))
        draw = ImageDraw.Draw(canvas)

        for star in self.stars:
            x = (star["x"] + star["drift"] * 0.055 * t) % OUT_SIZE[0]
            y = (star["y"] + star["drift"] * 0.018 * t) % OUT_SIZE[1]
            twinkle = 0.70 + 0.30 * math.sin(
                t * 1.05 + star["phase"]
            )
            alpha = int(star["a"] * twinkle)
            r = star["r"]
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=(205, 220, 255, alpha),
            )

        wash = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        wd = ImageDraw.Draw(wash)

        centers = [
            (OUT_SIZE[0] * 0.48, OUT_SIZE[1] * 0.50, (18, 60, 120)),
            (OUT_SIZE[0] * 0.15, OUT_SIZE[1] * 0.28, (70, 20, 100)),
            (OUT_SIZE[0] * 0.88, OUT_SIZE[1] * 0.70, (0, 95, 105)),
        ]

        for cx, cy, color in centers:
            for radius, alpha in [(700, 14), (470, 20), (250, 28)]:
                wd.ellipse(
                    (
                        cx - radius,
                        cy - radius,
                        cx + radius,
                        cy + radius,
                    ),
                    fill=(color[0], color[1], color[2], alpha),
                )

        wash = wash.filter(ImageFilter.GaussianBlur(90))
        canvas.alpha_composite(wash)

        for particle in self.dust:
            y = (particle["y"] + particle["speed"] * 0.17 * t) % OUT_SIZE[1]
            x = (
                particle["x"]
                + 17.0 * math.sin(t * 0.11 + particle["phase"])
            ) % OUT_SIZE[0]
            r = particle["r"] * (0.82 + 0.18 * math.sin(t + particle["phase"]))
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=(130, 160, 220, particle["a"]),
            )

        return canvas

    def active_warp_sources(
        self,
        active: Iterable[int],
        center_x: float,
        center_y: float,
        zoom: float,
        t: float,
    ):
        sources = []
        for event_index in active:
            x, y, scale = self.event_screen_position(
                int(event_index),
                center_x,
                center_y,
                zoom,
                t,
            )
            local = self.local_phase(int(event_index), t)
            pulse = math.sin(math.pi * local)
            strength = (
                16.0
                + 35.0 * self.mass_strength(int(event_index))
                + 18.0 * self.snr_strength(int(event_index))
            ) * pulse * scale
            sources.append((x, y, strength))
        return sources

    @staticmethod
    def warp_point(
        x: float,
        y: float,
        sources: List[Tuple[float, float, float]],
        t: float,
    ) -> Tuple[float, float]:
        warped_x = x
        warped_y = y

        for sx, sy, strength in sources:
            dx = x - sx
            dy = y - sy
            distance = math.hypot(dx, dy) + 1e-6
            ring = math.sin(distance * 0.028 - t * 5.2)
            envelope = math.exp(-distance / 420.0)
            offset = strength * ring * envelope

            warped_x += (dx / distance) * offset
            warped_y += (dy / distance) * offset

        return warped_x, warped_y

    def draw_spacetime_grid(
        self,
        canvas: Image.Image,
        active: Iterable[int],
        center_x: float,
        center_y: float,
        zoom: float,
        t: float,
    ):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        sources = self.active_warp_sources(
            active,
            center_x,
            center_y,
            zoom,
            t,
        )

        vertical_count = int(CONFIG["grid_vertical_lines"])
        horizontal_count = int(CONFIG["grid_horizontal_lines"])
        samples = int(CONFIG["grid_samples_per_line"])

        top = 220
        bottom = OUT_SIZE[1] - 250
        left = -80
        right = OUT_SIZE[0] + 80

        perspective_center = center_x + 28.0 * math.sin(t * 0.11)

        for line_index in range(vertical_count):
            u = line_index / max(vertical_count - 1, 1)
            base_x = lerp(left, right, u)
            points = []

            for sample_index in range(samples):
                v = sample_index / max(samples - 1, 1)
                y = lerp(top, bottom, v)
                convergence = (v - 0.5) * 0.12
                x = base_x + (perspective_center - base_x) * convergence
                x, y = self.warp_point(x, y, sources, t)
                points.append((x, y))

            draw.line(
                points,
                fill=(55, 170, 210, 46),
                width=1,
            )

        for line_index in range(horizontal_count):
            v = line_index / max(horizontal_count - 1, 1)
            base_y = lerp(top, bottom, v)
            points = []

            for sample_index in range(samples):
                u = sample_index / max(samples - 1, 1)
                x = lerp(left, right, u)
                y = base_y + 16.0 * math.sin(
                    u * math.pi * 2.0 + t * 0.10 + v * 2.0
                )
                x, y = self.warp_point(x, y, sources, t)
                points.append((x, y))

            alpha = 52 if line_index % 3 == 0 else 34
            draw.line(
                points,
                fill=(60, 155, 205, alpha),
                width=1,
            )

        canvas.alpha_composite(overlay)

    def draw_event_ripples(
        self,
        canvas: Image.Image,
        active: Iterable[int],
        center_x: float,
        center_y: float,
        zoom: float,
        t: float,
    ):
        bloom = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        bd = ImageDraw.Draw(bloom)

        sharp = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sharp)

        for event_index in active:
            event_index = int(event_index)
            x, y, depth_scale = self.event_screen_position(
                event_index,
                center_x,
                center_y,
                zoom,
                t,
            )
            local = self.local_phase(event_index, t)
            snr_strength = self.snr_strength(event_index)
            mass_strength = self.mass_strength(event_index)

            group = str(self.source_groups[event_index])
            color = GROUP_COLORS.get(
                group,
                GROUP_COLORS["no PE / unclassified"],
            )

            # Merger flash around local ~= 0.38.
            merger_flash = math.exp(-((local - 0.38) / 0.105) ** 2)
            core_radius = (
                3.0
                + 7.0 * mass_strength
                + 7.0 * merger_flash
            ) * depth_scale

            glow_radius = core_radius * (
                4.5 + 3.5 * snr_strength
            )

            bd.ellipse(
                (
                    x - glow_radius,
                    y - glow_radius,
                    x + glow_radius,
                    y + glow_radius,
                ),
                fill=(
                    color[0],
                    color[1],
                    color[2],
                    int(35 + 85 * merger_flash * snr_strength),
                ),
            )

            sd.ellipse(
                (
                    x - core_radius,
                    y - core_radius,
                    x + core_radius,
                    y + core_radius,
                ),
                fill=(
                    min(255, color[0] + 35),
                    min(255, color[1] + 35),
                    min(255, color[2] + 35),
                    int(120 + 130 * snr_strength),
                ),
            )

            # Expanding gravitational-wave fronts.
            for ring_index in range(4):
                ring_phase = local - ring_index * 0.105
                if ring_phase <= 0:
                    continue

                radius = (
                    24.0
                    + ring_phase * (215.0 + 165.0 * mass_strength)
                ) * depth_scale

                alpha = int(
                    190
                    * (1.0 - clamp(ring_phase))
                    * (0.35 + 0.65 * snr_strength)
                )

                if alpha <= 3:
                    continue

                squash = 0.78 + 0.08 * self.visual_lane[event_index]
                sd.ellipse(
                    (
                        x - radius,
                        y - radius * squash,
                        x + radius,
                        y + radius * squash,
                    ),
                    outline=(color[0], color[1], color[2], alpha),
                    width=max(1, int(1 + 2 * snr_strength)),
                )

            # Small crosshair for high-interest events.
            if (
                event_index in self.highlight_indices
                and local >= 0.25
            ):
                reticle = 18 + 14 * math.sin(local * math.pi)
                sd.line(
                    (x - reticle, y, x - 6, y),
                    fill=(255, 235, 170, 180),
                    width=2,
                )
                sd.line(
                    (x + 6, y, x + reticle, y),
                    fill=(255, 235, 170, 180),
                    width=2,
                )
                sd.line(
                    (x, y - reticle, x, y - 6),
                    fill=(255, 235, 170, 180),
                    width=2,
                )
                sd.line(
                    (x, y + 6, x, y + reticle),
                    fill=(255, 235, 170, 180),
                    width=2,
                )

        bloom = bloom.filter(ImageFilter.GaussianBlur(18))
        canvas.alpha_composite(bloom)
        canvas.alpha_composite(sharp)

    def strongest_active_event(
        self,
        active: Iterable[int],
    ) -> Optional[int]:
        active = list(int(i) for i in active)
        if not active:
            return None

        scores = self.events.iloc[active]["highlight_score"].to_numpy(float)
        return active[int(np.nanargmax(scores))]

    def draw_binary_merger(
        self,
        canvas: Image.Image,
        event_index: Optional[int],
        center_x: float,
        center_y: float,
        zoom: float,
        t: float,
    ):
        if event_index is None:
            return

        local = self.local_phase(event_index, t)
        visible = smoothstep((local - 0.02) / 0.18) * (
            1.0 - smoothstep((local - 0.84) / 0.16)
        )

        if visible <= 0.01:
            return

        m1 = self.mass_1[event_index]
        m2 = self.mass_2[event_index]
        chirp = self.chirp_mass[event_index]
        snr_strength = self.snr_strength(event_index)
        mass_strength = self.mass_strength(event_index)

        if not np.isfinite(m1):
            m1 = 20.0
        if not np.isfinite(m2):
            m2 = 14.0

        total = max(m1 + m2, 0.1)
        mass_ratio_1 = m1 / total
        mass_ratio_2 = m2 / total

        display_x = center_x
        display_y = center_y + 18.0

        # Inspiral tightens until merger.
        merger_u = smoothstep(local / 0.52)
        separation = lerp(
            170.0 * zoom,
            12.0 * zoom,
            merger_u,
        )

        chirp_norm = (
            math.log1p(max(chirp, 0.0)) / math.log1p(80.0)
            if np.isfinite(chirp)
            else 0.45
        )
        orbital_phase = (
            t * (2.2 + 2.8 * chirp_norm)
            + 16.0 * merger_u**2
            + self.visual_phase[event_index]
        )

        dx = math.cos(orbital_phase) * separation
        dy = math.sin(orbital_phase) * separation * 0.48

        p1 = (
            display_x + dx * mass_ratio_2,
            display_y + dy * mass_ratio_2,
        )
        p2 = (
            display_x - dx * mass_ratio_1,
            display_y - dy * mass_ratio_1,
        )

        layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        # Orbit / inspiral guide.
        orbit_w = separation * 1.30
        orbit_h = separation * 0.64
        orbit_alpha = int(95 * visible * (1.0 - merger_u))
        if orbit_alpha > 3:
            draw.ellipse(
                (
                    display_x - orbit_w,
                    display_y - orbit_h,
                    display_x + orbit_w,
                    display_y + orbit_h,
                ),
                outline=(120, 205, 255, orbit_alpha),
                width=2,
            )

        radius_1 = (
            12.0 + 25.0 * math.sqrt(max(m1, 0.0) / max(total, 1e-6))
        ) * zoom
        radius_2 = (
            12.0 + 25.0 * math.sqrt(max(m2, 0.0) / max(total, 1e-6))
        ) * zoom

        premerge = 1.0 - smoothstep((local - 0.43) / 0.12)

        for (px, py), radius, color in [
            (p1, radius_1, (115, 210, 255)),
            (p2, radius_2, (235, 170, 255)),
        ]:
            alpha = int(245 * visible * premerge)
            glow_r = radius * (3.0 + snr_strength)
            draw.ellipse(
                (
                    px - glow_r,
                    py - glow_r,
                    px + glow_r,
                    py + glow_r,
                ),
                fill=(color[0], color[1], color[2], int(36 * visible)),
            )
            draw.ellipse(
                (
                    px - radius,
                    py - radius,
                    px + radius,
                    py + radius,
                ),
                fill=(color[0], color[1], color[2], alpha),
                outline=(255, 255, 255, min(230, alpha)),
                width=2,
            )
            inner = radius * 0.56
            draw.ellipse(
                (
                    px - inner,
                    py - inner,
                    px + inner,
                    py + inner,
                ),
                fill=(1, 2, 7, alpha),
            )

        # Merger remnant.
        postmerge = smoothstep((local - 0.42) / 0.13)
        if postmerge > 0:
            remnant_radius = (
                18.0 + 35.0 * mass_strength
            ) * zoom * (0.82 + 0.18 * postmerge)
            glow_radius = remnant_radius * (
                4.5 + 3.5 * snr_strength
            )

            draw.ellipse(
                (
                    display_x - glow_radius,
                    display_y - glow_radius,
                    display_x + glow_radius,
                    display_y + glow_radius,
                ),
                fill=(
                    120,
                    220,
                    255,
                    int(52 * postmerge * visible),
                ),
            )
            draw.ellipse(
                (
                    display_x - remnant_radius,
                    display_y - remnant_radius * 0.76,
                    display_x + remnant_radius,
                    display_y + remnant_radius * 0.76,
                ),
                fill=(1, 2, 6, int(255 * postmerge * visible)),
                outline=(
                    190,
                    235,
                    255,
                    int(240 * postmerge * visible),
                ),
                width=3,
            )

            # Accretion-like editorial ring.
            ring_w = remnant_radius * 2.2
            ring_h = remnant_radius * 0.42
            draw.ellipse(
                (
                    display_x - ring_w,
                    display_y - ring_h,
                    display_x + ring_w,
                    display_y + ring_h,
                ),
                outline=(
                    255,
                    210,
                    145,
                    int(150 * postmerge * visible),
                ),
                width=3,
            )

        layer = layer.filter(ImageFilter.GaussianBlur(0.6))
        canvas.alpha_composite(layer)

        # Data label next to the central glyph.
        label_alpha = int(235 * visible)
        draw_text(
            canvas,
            str(self.names[event_index]),
            (72, 300),
            size=36,
            fill=(255, 245, 210, label_alpha),
            bold=True,
        )

        line_1 = (
            f"M1 {format_mass(self.mass_1[event_index])}  //  "
            f"M2 {format_mass(self.mass_2[event_index])}"
        )
        draw_text(
            canvas,
            line_1,
            (74, 348),
            size=22,
            fill=(215, 230, 250, label_alpha),
            stroke=1,
        )

        line_2 = (
            f"{format_snr(self.snr[event_index])}  //  "
            f"{format_distance(self.distance[event_index])}"
        )
        draw_text(
            canvas,
            line_2,
            (74, 382),
            size=22,
            fill=(190, 220, 245, label_alpha),
            stroke=1,
        )

        highlight = self.highlight_labels[event_index]
        if isinstance(highlight, str) and highlight:
            draw_text(
                canvas,
                highlight,
                (74, 420),
                size=21,
                fill=(255, 205, 130, label_alpha),
                bold=True,
                stroke=1,
            )

    def draw_waveform_strip(
        self,
        canvas: Image.Image,
        event_index: Optional[int],
        t: float,
    ):
        alpha = int(
            225
            * smoothstep((t - 28.0) / 4.0)
            * (1.0 - smoothstep((t - 51.5) / 4.0))
        )
        if alpha <= 5 or event_index is None:
            return

        x0 = 60
        y0 = OUT_SIZE[1] - 505
        width = OUT_SIZE[0] - 120
        height = 165

        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rounded_rectangle(
            (x0 - 18, y0 - 52, x0 + width + 18, y0 + height + 44),
            radius=24,
            fill=(0, 0, 0, 112),
            outline=(80, 170, 220, 70),
            width=1,
        )
        canvas.alpha_composite(panel)

        draw_text(
            canvas,
            "EDITORIAL CHIRP DISPLAY",
            (x0, y0 - 34),
            size=21,
            fill=(205, 225, 248, alpha),
            bold=True,
            stroke=1,
        )

        local = self.local_phase(event_index, t)
        chirp = self.chirp_mass[event_index]
        snr_strength = self.snr_strength(event_index)

        chirp_norm = (
            math.log1p(max(chirp, 0.0)) / math.log1p(80.0)
            if np.isfinite(chirp)
            else 0.45
        )

        sample_count = 260
        u_values = np.linspace(0.0, 1.0, sample_count)
        points = []

        visible_u = clamp(local * 1.35)
        amplitude = 16.0 + 48.0 * snr_strength

        for u in u_values:
            if u > visible_u:
                break

            envelope = 0.18 + 0.82 * smoothstep(u)
            cycles = 2.0 + (8.0 + 10.0 * chirp_norm) * (u**2.25)
            phase = 2.0 * math.pi * cycles
            value = math.sin(phase) * amplitude * envelope

            x = x0 + u * width
            y = y0 + height * 0.5 - value
            points.append((x, y))

        draw = ImageDraw.Draw(canvas)
        draw.line(
            (x0, y0 + height * 0.5, x0 + width, y0 + height * 0.5),
            fill=(100, 150, 190, int(alpha * 0.35)),
            width=1,
        )

        if len(points) >= 2:
            draw.line(
                points,
                fill=(
                    120,
                    225,
                    255,
                    int(alpha * (0.55 + 0.45 * snr_strength)),
                ),
                width=3,
            )

        chirp_text = (
            f"chirp mass {format_mass(chirp)}  //  "
            f"{format_snr(self.snr[event_index])}"
        )
        draw_text(
            canvas,
            chirp_text,
            (x0, y0 + height + 12),
            size=19,
            fill=(205, 220, 245, alpha),
            stroke=1,
        )

    def draw_mass_leaderboard(self, canvas: Image.Image, t: float):
        alpha = int(225 * smoothstep((t - 41.0) / 3.4))
        if alpha <= 5 or len(self.leaderboard_indices) == 0:
            return

        x0 = 62
        y0 = 545
        width = 440
        row_h = 52

        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)

        panel_h = 88 + row_h * len(self.leaderboard_indices)
        draw.rounded_rectangle(
            (x0 - 18, y0 - 22, x0 + width, y0 + panel_h),
            radius=24,
            fill=(0, 0, 0, 118),
            outline=(100, 175, 220, 70),
            width=1,
        )
        canvas.alpha_composite(panel)

        draw_text(
            canvas,
            "LARGEST FINAL-MASS BEST VALUES",
            (x0, y0),
            size=22,
            fill=(230, 238, 255, alpha),
            bold=True,
            stroke=1,
        )

        y = y0 + 50
        for rank, event_index in enumerate(self.leaderboard_indices, start=1):
            event_index = int(event_index)
            name = str(self.names[event_index])
            value = self.final_mass[event_index]

            draw_text(
                canvas,
                f"{rank:02d}",
                (x0, y),
                size=20,
                fill=(130, 205, 245, alpha),
                bold=True,
                stroke=1,
            )
            draw_text(
                canvas,
                name,
                (x0 + 48, y),
                size=20,
                fill=(225, 232, 248, alpha),
                stroke=1,
            )
            draw_text(
                canvas,
                format_mass(value),
                (x0 + 288, y),
                size=20,
                fill=(255, 215, 145, alpha),
                bold=True,
                stroke=1,
            )
            y += row_h

    def draw_detector_hud(
        self,
        canvas: Image.Image,
        event_index: Optional[int],
        t: float,
    ):
        if t < 5.0:
            return

        alpha = 205
        x0 = OUT_SIZE[0] - 330
        y0 = 92

        draw_text(
            canvas,
            "DETECTOR NETWORK",
            (x0, y0),
            size=19,
            fill=(195, 218, 245, alpha),
            bold=True,
            stroke=1,
        )

        active_detectors: List[str] = []
        if event_index is not None:
            active_detectors = [
                item.strip()
                for item in str(self.detectors[event_index]).split(",")
                if item.strip()
            ]

        detector_labels = [
            ("H1", "LIGO HANFORD"),
            ("L1", "LIGO LIVINGSTON"),
            ("V1", "VIRGO"),
            ("K1", "KAGRA"),
            ("G1", "GEO600"),
        ]

        draw = ImageDraw.Draw(canvas)
        y = y0 + 38

        for code, label in detector_labels:
            active = code in active_detectors
            pulse = 0.72 + 0.28 * math.sin(t * 3.2 + len(code))
            radius = 7 if active else 5
            color = (
                (120, 255, 220, int(235 * pulse))
                if active
                else (100, 120, 155, 100)
            )

            draw.ellipse(
                (
                    x0,
                    y + 5 - radius,
                    x0 + 2 * radius,
                    y + 5 + radius,
                ),
                fill=color,
            )
            draw_text(
                canvas,
                f"{code}  {label}",
                (x0 + 25, y),
                size=17,
                fill=(
                    220,
                    235,
                    250,
                    alpha if active else 105,
                ),
                bold=active,
                stroke=1,
            )
            y += 31

    def draw_timeline(
        self,
        canvas: Image.Image,
        event_index: int,
        t: float,
    ):
        x0 = 70
        x1 = OUT_SIZE[0] - 70
        y = OUT_SIZE[1] - 310

        fraction = self.replay_fraction(t)
        draw = ImageDraw.Draw(canvas)

        draw.line(
            (x0, y, x1, y),
            fill=(110, 180, 220, 105),
            width=2,
        )
        draw.line(
            (x0, y, lerp(x0, x1, fraction), y),
            fill=(120, 235, 255, 215),
            width=4,
        )

        knob_x = lerp(x0, x1, fraction)
        draw.ellipse(
            (
                knob_x - 7,
                y - 7,
                knob_x + 7,
                y + 7,
            ),
            fill=(235, 250, 255, 245),
        )

        current_time = self.events.iloc[event_index]["event_time"]
        year_text = (
            str(current_time.year)
            if pd.notna(current_time)
            else str(self.names[event_index])
        )

        draw_text(
            canvas,
            year_text,
            (x0, y - 40),
            size=24,
            fill=(220, 235, 250, 225),
            bold=True,
            stroke=1,
        )

        draw_text(
            canvas,
            f"{event_index + 1:,} / {len(self.events):,} EVENTS",
            (x1, y - 40),
            size=19,
            fill=(190, 215, 240, 200),
            bold=True,
            stroke=1,
            anchor="ra",
        )

    def draw_scanlines_and_noise(self, canvas: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for y in range(0, OUT_SIZE[1], 7):
            draw.line(
                (0, y, OUT_SIZE[0], y),
                fill=(0, 0, 0, 14),
                width=1,
            )

        for item in self.hud_noise:
            flicker = 0.5 + 0.5 * math.sin(t * 5.0 + item["phase"])
            alpha = int(item["alpha"] * flicker)
            x = item["x"]
            y = (item["y"] + t * 3.0) % OUT_SIZE[1]
            draw.line(
                (x, y, x + item["length"], y),
                fill=(100, 220, 230, alpha),
                width=1,
            )

        canvas.alpha_composite(overlay)

    def add_text_layers(
        self,
        canvas: Image.Image,
        t: float,
        shot: Dict,
    ):
        title_alpha = int(
            255
            * smoothstep((t - 0.25) / 1.1)
            * (1.0 - smoothstep((t - 5.6) / 1.1))
        )

        if title_alpha > 5:
            draw_text(
                canvas,
                CONFIG["title_text"],
                (62, 120),
                size=58,
                fill=(255, 255, 255, title_alpha),
                bold=True,
            )
            draw_text(
                canvas,
                CONFIG["subtitle_text"],
                (66, 204),
                size=27,
                fill=(205, 228, 255, min(230, title_alpha)),
            )

        if t > 5.5:
            draw_text(
                canvas,
                shot["caption"],
                (62, 78),
                size=22,
                fill=(185, 215, 245, 190),
                stroke=1,
            )

        cap = caption_at(t)
        if cap:
            overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            y0 = OUT_SIZE[1] - 250
            draw.rounded_rectangle(
                (46, y0, OUT_SIZE[0] - 46, y0 + 126),
                radius=26,
                fill=(0, 0, 0, 142),
            )
            canvas.alpha_composite(overlay)

            draw_wrapped_text(
                canvas,
                cap,
                (70, y0 + 28),
                max_width=OUT_SIZE[0] - 140,
                size=31,
                fill=(255, 255, 255, 245),
            )

        note_alpha = int(220 * smoothstep((t - 50.5) / 3.2))
        if note_alpha > 5:
            draw_wrapped_text(
                canvas,
                CONFIG["credit_text"],
                (68, OUT_SIZE[1] - 118),
                max_width=940,
                size=20,
                fill=(225, 230, 245, note_alpha),
            )
            draw_wrapped_text(
                canvas,
                CONFIG["scientific_note"],
                (68, OUT_SIZE[1] - 84),
                max_width=940,
                size=18,
                fill=(200, 215, 238, note_alpha),
            )

    def render_frame(self, t: float) -> np.ndarray:
        shot, center_x, center_y, zoom = shot_state(t)
        active = self.active_event_indices(t)
        current_index = self.replay_event_index(t)
        strongest = self.strongest_active_event(active)

        if strongest is None:
            strongest = current_index

        canvas = self.render_background(t)
        self.draw_spacetime_grid(
            canvas,
            active,
            center_x,
            center_y,
            zoom,
            t,
        )
        self.draw_event_ripples(
            canvas,
            active,
            center_x,
            center_y,
            zoom,
            t,
        )
        self.draw_binary_merger(
            canvas,
            strongest,
            center_x,
            center_y,
            zoom,
            t,
        )
        self.draw_detector_hud(canvas, strongest, t)
        self.draw_waveform_strip(canvas, strongest, t)
        self.draw_mass_leaderboard(canvas, t)
        self.draw_timeline(canvas, current_index, t)
        self.add_text_layers(canvas, t, shot)
        self.draw_scanlines_and_noise(canvas, t)

        arr = np.array(canvas.convert("RGB"))
        arr = apply_grade(arr)
        arr = np.clip(
            arr.astype(np.float32) * VIGNETTE[..., None],
            0,
            255,
        ).astype(np.uint8)

        fade_in = smoothstep(t / 1.0)
        fade_out = 1.0 - smoothstep(
            (t - (CONFIG["duration_s"] - 1.2)) / 1.0
        )
        arr = np.clip(
            arr.astype(np.float32) * fade_in * fade_out,
            0,
            255,
        ).astype(np.uint8)

        return arr


# %% [markdown]
# ## Preview frame export

def render_preview_frames(scene: SpacetimeCollisionScene):
    preview_times = [
        1.2,
        10.0,
        24.0,
        36.0,
        46.0,
        CONFIG["duration_s"] - 1.0,
    ]

    preview_arrays = []

    for t in tqdm(preview_times, desc="Preview frames"):
        arr = scene.render_frame(float(t))
        preview_arrays.append(arr)
        Image.fromarray(arr).save(
            PREVIEW_DIR / f"preview_{int(t):02d}s.png"
        )

    fig, axes = plt.subplots(
        1,
        len(preview_arrays),
        figsize=(18, 9),
    )

    for ax, image, t in zip(axes, preview_arrays, preview_times):
        ax.imshow(image)
        ax.set_title(f"{t:.0f}s")
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(
        PREVIEW_DIR / "preview_grid.png",
        dpi=170,
    )
    plt.close(fig)

    print("Preview frames written to:", PREVIEW_DIR.resolve())


# %% [markdown]
# ## Subtitle sidecar

def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    seconds_int = milliseconds // 1000
    milliseconds %= 1000

    return (
        f"{hours:02d}:{minutes:02d}:{seconds_int:02d},"
        f"{milliseconds:03d}"
    )


def write_srt(
    captions: Iterable[Tuple[float, float, str]],
    path: Path,
):
    lines: List[str] = []

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
# ## MP4 rendering and finishing

def render_video(
    scene: SpacetimeCollisionScene,
    raw_video_path: Path,
):
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
        for t in tqdm(times, desc="Rendering video"):
            writer.append_data(scene.render_frame(float(t)))

    print("Raw video written:", raw_video_path.resolve())


def run_ffmpeg(command: List[str]):
    print("Running:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def finish_video(
    raw_video_path: Path,
    srt_path: Path,
    final_video_path: Path,
):
    ffmpeg = find_ffmpeg()
    print("FFmpeg detected:", ffmpeg)

    subbed_path = (
        OUTPUT_ROOT
        / f"{CONFIG['output_basename']}_subbed.mp4"
    )
    audio_path_output = (
        OUTPUT_ROOT
        / f"{CONFIG['output_basename']}_with_audio.mp4"
    )

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
            str(subbed_path),
        ]
        run_ffmpeg(command)
        final_candidate = subbed_path

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
            str(audio_path_output),
        ]
        run_ffmpeg(command)
        final_candidate = audio_path_output
    elif audio_path:
        print(
            "audio_path was set, but the file was not found or "
            "ffmpeg was unavailable. Skipping audio."
        )

    if final_candidate.exists():
        shutil.copyfile(final_candidate, final_video_path)
        print("Final video:", final_video_path.resolve())


# %% [markdown]
# ## Main pipeline

def main(force_refresh: bool = False):
    print("=" * 72)
    print("SPACETIME COLLISIONS — GWOSC GRAVITATIONAL-WAVE SHORT")
    print("=" * 72)

    event_df = fetch_gwosc_events(
        CONFIG,
        force_refresh=force_refresh,
    )

    video_events = select_video_events(event_df, CONFIG)

    print(f"Events selected for video: {len(video_events):,}")
    print("Catalog coverage:")
    print(video_events["catalog"].value_counts())
    print("Detector-network strings:")
    print(video_events["detectors"].value_counts().head(10))

    create_scientific_previews(event_df)

    scene = SpacetimeCollisionScene(video_events)
    render_preview_frames(scene)

    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"

    if CONFIG.get("write_subtitle_sidecar", True):
        write_srt(CAPTIONS, srt_path)
        print("Subtitle sidecar written:", srt_path.resolve())

    raw_video_path = (
        OUTPUT_ROOT
        / f"{CONFIG['output_basename']}_raw.mp4"
    )
    final_video_path = (
        OUTPUT_ROOT
        / f"{CONFIG['output_basename']}_final.mp4"
    )

    render_video(scene, raw_video_path)

    finish_video(
        raw_video_path,
        srt_path,
        final_video_path,
    )

    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-", path.name)


if __name__ == "__main__":
    main(force_refresh=False)


# %% [markdown]
# # Suggested narration / description
#
# Suggested voiceover:
#
# > These are not photographs of black holes.
# > They are collisions measured through gravitational waves.
# > Each pulse is a cataloged event, replayed in chronological order.
# > The masses of the two compact objects control the scale of the merging pair.
# > Stronger network signals burn brighter.
# > Then the inspiral accelerates, the system merges, and spacetime carries the signal away.
# > Some events leave behind an enormous compact remnant.
# > The universe was always ringing.
# > Our detectors learned how to listen.
#
# Suggested YouTube Shorts caption:
#
# A cinematic replay of public gravitational-wave events using real GWOSC catalog
# parameters. Source-frame masses, luminosity distance and network SNR drive the
# visual mappings; the spacetime grid and chirp graphic are editorial science
# communication.
#
# #Space #GravitationalWaves #LIGO #Virgo #BlackHoles #Astronomy #ScienceShorts
