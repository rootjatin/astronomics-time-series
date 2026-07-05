# %% [markdown]
# # Cinematic YouTube Short from Real Exoplanet Archive Data
#
# This script creates a vertical 1080×1920 ambience-style space film from
# confirmed exoplanet system data in the NASA Exoplanet Archive.
#
# Theme: **Exoplanet Atlas — worlds around other stars**
#
# What it produces:
# - Downloads real confirmed exoplanet / host-star data from the NASA Exoplanet Archive TAP service.
# - Cleans coordinates, distances, planet radius, mass, stellar temperature, and discovery year.
# - Builds scientific preview plots.
# - Renders a cinematic vertical MP4 with:
#   - drifting starfield,
#   - 3D fly-through of known exoplanet host systems,
#   - highlighted famous worlds such as Proxima Cen b, TRAPPIST-1 planets, 51 Peg b, K2-18 b, and TOI-700 d when present,
#   - soft ambient glow, parallax, labels, captions, and credits,
#   - optional subtitle sidecar and optional audio muxing.
#
# Scientific fidelity note:
# The data layer uses catalog values from the NASA Exoplanet Archive.
# The camera motion, glow, particles, color categories, and planet-bead animations
# are cinematic effects for science communication, not a literal spaceflight simulation.

from __future__ import annotations

import io
import math
import os
import shutil
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

plt.rcParams["figure.figsize"] = (9, 9)
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
# For testing, reduce `duration_s` to 8–12 and `fps` to 12.

OUTPUT_ROOT = Path("exoplanet_ambience_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"

for p in [OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR]:
    p.mkdir(parents=True, exist_ok=True)

CONFIG = {
    # Final delivery
    "video_width": 1080,
    "video_height": 1920,
    "fps": 24,
    "duration_s": 58,
    "output_basename": "exoplanet_archive_ambience_short",

    # NASA Exoplanet Archive TAP service
    "tap_url": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
    "request_timeout": (20, 180),
    "request_retries": 5,
    "retry_backoff_s": 2.0,

    # Data limits / visual limits
    "max_distance_pc_for_visual": 900.0,
    "max_systems_for_video": 4200,
    "label_distance_pc_limit": 150.0,

    # Rendering style
    "background_star_count": 1150,
    "dust_particle_count": 180,
    "vignette_strength": 0.28,
    "contrast_boost": 1.12,
    "saturation_boost": 1.08,
    "camera_focal_length": 980.0,

    # Text
    "title_text": "Exoplanet Atlas",
    "subtitle_text": "Known worlds around other stars",
    "credit_text": "Data source: NASA Exoplanet Archive / IPAC / Caltech",
    "scientific_note": "System positions use catalog RA, Dec, and distance; motion and glow are cinematic.",

    # Optional audio / subtitles
    "audio_path": None,       # Example: "audio/deep_space_ambient.mp3"
    "burn_subtitles": False,  # requires ffmpeg with subtitle support
    "write_subtitle_sidecar": True,
}

print("Configuration ready.")

# %% [markdown]
# ## Download confirmed exoplanet data
#
# The script uses the NASA Exoplanet Archive TAP service and the `pscomppars`
# table. `PSCompPars` has one row per planet, which is useful for a statistical
# view of the confirmed exoplanet population.

FIELDS = [
    "pl_name",
    "hostname",
    "discoverymethod",
    "disc_year",
    "pl_orbper",
    "pl_orbsmax",
    "pl_orbeccen",
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "st_teff",
    "st_rad",
    "st_mass",
    "sy_dist",
    "sy_snum",
    "sy_pnum",
    "ra",
    "dec",
    "glon",
    "glat",
    "sy_vmag",
    "sy_gaiamag",
    "pl_controv_flag",
]

HIGHLIGHT_PLANETS = [
    "Proxima Cen b",
    "TRAPPIST-1 b",
    "TRAPPIST-1 c",
    "TRAPPIST-1 d",
    "TRAPPIST-1 e",
    "TRAPPIST-1 f",
    "TRAPPIST-1 g",
    "51 Peg b",
    "HD 209458 b",
    "Kepler-186 f",
    "K2-18 b",
    "TOI-700 d",
    "LHS 1140 b",
    "WASP-12 b",
]

HIGHLIGHT_HOSTS = [
    "Proxima Cen",
    "TRAPPIST-1",
    "51 Peg",
    "HD 209458",
    "Kepler-186",
    "K2-18",
    "TOI-700",
    "LHS 1140",
    "WASP-12",
]


def build_retry_session(config: Dict) -> requests.Session:
    """Create a requests session that survives temporary network/API slowdowns."""
    retries = Retry(
        total=int(config.get("request_retries", 5)),
        connect=int(config.get("request_retries", 5)),
        read=int(config.get("request_retries", 5)),
        status=int(config.get("request_retries", 5)),
        backoff_factor=float(config.get("retry_backoff_s", 2.0)),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retries, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "exoplanet-archive-ambience-short/1.1 (educational visualization)"
    })
    return session


def build_exoplanet_query() -> str:
    """ADQL query for confirmed exoplanet composite parameters.

    Notes:
    - Keep this field list restricted to columns documented for PSCompPars.
    - Use unquoted dec. NASA/IPAC's Oracle-backed TAP service stores column
      identifiers in uppercase internally; quoted "dec" becomes a lowercase,
      case-sensitive identifier and can fail with ORA-00904 invalid identifier.
    - Do not request rowupdate here; it is not needed for the video.
    """
    query_fields = [
        "pl_name",
        "hostname",
        "discoverymethod",
        "disc_year",
        "pl_orbper",
        "pl_orbsmax",
        "pl_orbeccen",
        "pl_rade",
        "pl_bmasse",
        "pl_eqt",
        "st_teff",
        "st_rad",
        "st_mass",
        "sy_dist",
        "sy_snum",
        "sy_pnum",
        "ra",
        "dec",
        "glon",
        "glat",
        "sy_vmag",
        "sy_gaiamag",
        "pl_controv_flag",
    ]
    field_text = ", ".join(query_fields)
    return f"""
    SELECT {field_text}
    FROM pscomppars
    WHERE sy_dist IS NOT NULL
      AND ra IS NOT NULL
      AND dec IS NOT NULL
      AND pl_controv_flag = 0
    ORDER BY sy_dist ASC
    """.strip()


def fetch_exoplanet_archive(config: Dict, force_refresh: bool = False) -> pd.DataFrame:
    """Download exoplanet rows from NASA Exoplanet Archive, cache as CSV, and clean."""
    csv_path = DATA_ROOT / "nasa_exoplanet_archive_pscomppars_clean.csv"
    raw_path = DATA_ROOT / "nasa_exoplanet_archive_pscomppars_raw.csv"

    if csv_path.exists() and not force_refresh:
        print(f"Using cached NASA Exoplanet Archive CSV: {csv_path}")
        return clean_planet_dataframe(pd.read_csv(csv_path), save_path=None)

    query = build_exoplanet_query()
    params = {"query": query, "format": "csv"}
    session = build_retry_session(config)

    try:
        print("Requesting NASA Exoplanet Archive TAP data ...")

        # POST is more robust for TAP/ADQL because it avoids URL length and
        # escaping edge cases. If a server/proxy rejects POST, fall back to GET.
        response = session.post(config["tap_url"], data=params, timeout=config["request_timeout"])
        if response.status_code >= 400:
            print("POST query failed; retrying with GET ...")
            response = session.get(config["tap_url"], params=params, timeout=config["request_timeout"])

        if response.status_code >= 400:
            raise RuntimeError(
                f"TAP HTTP {response.status_code}: {response.reason}\n"
                f"Response body preview:\n{response.text[:1500]}"
            )

        text = response.text
        if not text.strip() or "ERROR" in text[:500].upper():
            raise RuntimeError(f"Unexpected TAP response:\n{text[:1500]}")
        raw_path.write_text(text, encoding="utf-8")
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        if raw_path.exists() and not force_refresh:
            print("Live request failed, so using cached raw CSV instead.")
            print(f"Reason: {exc}")
            df = pd.read_csv(raw_path)
        else:
            encoded = urllib.parse.urlencode(params)
            raise RuntimeError(
                "NASA Exoplanet Archive request failed and no cache exists. "
                "Check internet access, VPN/firewall settings, or the TAP endpoint.\n"
                f"URL: {config['tap_url']}?{encoded}\nOriginal error: {exc}"
            ) from exc

    return clean_planet_dataframe(df, save_path=csv_path)


def clean_planet_dataframe(df: pd.DataFrame, save_path: Optional[Path] = None) -> pd.DataFrame:
    """Normalize numeric columns and keep rows with useful sky positions."""
    numeric_cols = [
        "disc_year", "pl_orbper", "pl_orbsmax", "pl_orbeccen", "pl_rade", "pl_bmasse",
        "pl_eqt", "st_teff", "st_rad", "st_mass", "sy_dist", "sy_snum", "sy_pnum",
        "ra", "dec", "glon", "glat", "sy_vmag", "sy_gaiamag", "pl_controv_flag",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["pl_name", "hostname", "discoverymethod"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Some TAP services preserve quoted identifiers in the output header.
    # Normalize that back to the column name used by the renderer.
    for possible_dec_name in ['"dec"', 'DEC']:
        if "dec" not in df.columns and possible_dec_name in df.columns:
            df = df.rename(columns={possible_dec_name: "dec"})

    required = ["pl_name", "hostname", "sy_dist", "ra", "dec"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns from Exoplanet Archive response: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["sy_dist", "ra", "dec"])
    df = df[(df["sy_dist"] > 0) & (df["sy_dist"] < 20000)].copy()
    if "pl_controv_flag" in df.columns:
        df = df[(df["pl_controv_flag"].fillna(0) == 0)].copy()

    df["highlight"] = df["pl_name"].apply(highlight_name)
    df["planet_group"] = df.apply(classify_planet, axis=1)
    df = df.sort_values(["sy_dist", "hostname", "pl_name"]).reset_index(drop=True)

    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"Saved cleaned data: {save_path}")
    print(f"Planet rows retained: {len(df):,}")
    print(f"Host systems retained: {df['hostname'].nunique():,}")
    return df


def highlight_name(planet_name: str) -> Optional[str]:
    """Return a display label when a planet is in the highlight list."""
    p = str(planet_name).strip()
    for target in HIGHLIGHT_PLANETS:
        if p.lower() == target.lower():
            return target
    # Looser matching for archive naming variants.
    for target in HIGHLIGHT_PLANETS:
        if target.lower() in p.lower():
            return target
    return None


def classify_planet(row: pd.Series) -> str:
    """Simple visual buckets for cinematic color coding."""
    if highlight_name(row.get("pl_name", "")):
        return "highlight world"

    radius = row.get("pl_rade", np.nan)
    period = row.get("pl_orbper", np.nan)
    eqt = row.get("pl_eqt", np.nan)

    if np.isfinite(radius) and np.isfinite(eqt) and 0.7 <= radius <= 1.8 and 180 <= eqt <= 330:
        return "warm rocky-size"
    if np.isfinite(radius) and radius <= 1.8:
        return "rocky-size"
    if np.isfinite(radius) and 1.8 < radius <= 4.2:
        return "sub-Neptune"
    if np.isfinite(radius) and radius >= 8.0 and np.isfinite(period) and period <= 10:
        return "hot giant"
    if np.isfinite(radius) and radius >= 8.0:
        return "gas giant"
    return "other confirmed planet"


planet_df = fetch_exoplanet_archive(CONFIG, force_refresh=False)
planet_df.head()

# %% [markdown]
# ## Create a host-system table
#
# Multiple planets can share the same host star, so the video aggregates planet rows
# into host systems for the 3D map. Famous highlighted planets are still labelled.


def make_system_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for host, sdf in df.groupby("hostname", sort=False):
        sdf = sdf.sort_values(["highlight", "pl_name"], na_position="last")
        highlight_values = [h for h in sdf["highlight"].dropna().astype(str).tolist() if h]
        highlight = highlight_values[0] if highlight_values else None

        # Use the first catalog position/distance for the shared host.
        first = sdf.iloc[0]
        # Dominant visual group: prioritize highlighted and interesting compact worlds.
        if highlight:
            group = "highlight system"
        elif (sdf["planet_group"] == "warm rocky-size").any():
            group = "warm rocky-size system"
        elif (sdf["planet_group"] == "rocky-size").any():
            group = "rocky-size system"
        elif (sdf["planet_group"] == "sub-Neptune").any():
            group = "sub-Neptune system"
        elif (sdf["planet_group"] == "hot giant").any():
            group = "hot giant system"
        elif (sdf["planet_group"] == "gas giant").any():
            group = "gas giant system"
        else:
            group = "other exoplanet system"

        rows.append({
            "hostname": host,
            "planet_count": int(sdf["pl_name"].nunique()),
            "planet_names": "; ".join(sdf["pl_name"].head(9).tolist()),
            "highlight": highlight,
            "visual_group": group,
            "sy_dist": first.get("sy_dist", np.nan),
            "ra": first.get("ra", np.nan),
            "dec": first.get("dec", np.nan),
            "glon": first.get("glon", np.nan),
            "glat": first.get("glat", np.nan),
            "st_teff": np.nanmedian(sdf["st_teff"].to_numpy(float)) if "st_teff" in sdf else np.nan,
            "sy_vmag": np.nanmedian(sdf["sy_vmag"].to_numpy(float)) if "sy_vmag" in sdf else np.nan,
            "sy_gaiamag": np.nanmedian(sdf["sy_gaiamag"].to_numpy(float)) if "sy_gaiamag" in sdf else np.nan,
            "disc_year_min": np.nanmin(sdf["disc_year"].to_numpy(float)) if "disc_year" in sdf else np.nan,
            "disc_year_max": np.nanmax(sdf["disc_year"].to_numpy(float)) if "disc_year" in sdf else np.nan,
        })
    out = pd.DataFrame(rows)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["sy_dist", "ra", "dec"])
    out = out.sort_values(["sy_dist", "hostname"]).reset_index(drop=True)
    return out


system_df = make_system_table(planet_df)

# Keep a readable subset for video while prioritizing highlights, nearby systems, and multi-planet systems.
video_systems = system_df.copy()
video_systems = video_systems[video_systems["sy_dist"] <= CONFIG["max_distance_pc_for_visual"]].copy()
video_systems["priority"] = (
    video_systems["highlight"].notna().astype(int) * 10000
    + (CONFIG["max_distance_pc_for_visual"] - video_systems["sy_dist"].clip(0, CONFIG["max_distance_pc_for_visual"]))
    + video_systems["planet_count"].clip(1, 10) * 80
)
video_systems = video_systems.sort_values("priority", ascending=False).head(CONFIG["max_systems_for_video"])
video_systems = video_systems.sort_values("sy_dist").reset_index(drop=True)

print(f"Video host systems: {len(video_systems):,}")
print(video_systems["visual_group"].value_counts())
video_systems.head()

# %% [markdown]
# ## Scientific preview plots

fig, ax = plt.subplots(figsize=(10, 5))
years = planet_df["disc_year"].dropna().astype(int)
if len(years):
    years.hist(bins=range(int(years.min()), int(years.max()) + 2), ax=ax)
ax.set_title("Confirmed exoplanet discoveries by year")
ax.set_xlabel("Discovery year")
ax.set_ylabel("Planet count")
plt.tight_layout()
plt.savefig(PREVIEW_DIR / "discoveries_by_year.png", dpi=170)
plt.show()

fig, ax = plt.subplots(figsize=(9, 6))
plot_df = planet_df.dropna(subset=["sy_dist", "pl_rade"]).copy()
plot_df = plot_df[plot_df["sy_dist"] <= CONFIG["max_distance_pc_for_visual"]]
if len(plot_df):
    ax.scatter(plot_df["sy_dist"], plot_df["pl_rade"].clip(upper=22), s=8, alpha=0.45)
ax.set_title("Planet radius vs host-system distance")
ax.set_xlabel("Distance [parsec]")
ax.set_ylabel("Planet radius [Earth radii], clipped at 22")
plt.tight_layout()
plt.savefig(PREVIEW_DIR / "radius_vs_distance.png", dpi=170)
plt.show()

# %% [markdown]
# ## Cinematic rendering helpers

OUT_SIZE = (CONFIG["video_width"], CONFIG["video_height"])

GROUP_COLORS = {
    "highlight system": (255, 230, 150, 255),
    "warm rocky-size system": (120, 230, 210, 230),
    "rocky-size system": (140, 205, 255, 220),
    "sub-Neptune system": (170, 155, 255, 215),
    "hot giant system": (255, 120, 150, 225),
    "gas giant system": (255, 175, 90, 220),
    "other exoplanet system": (200, 220, 255, 175),
}


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(t):
    t = clamp(float(t), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ease_in_out_sine(t):
    t = clamp(float(t), 0.0, 1.0)
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
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(
    img: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int = 42,
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
        stroke_fill=(0, 0, 0, min(fill[3] if len(fill) > 3 else 255, 220)),
    )


def draw_wrapped_text(
    img: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int = 34,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    line_spacing: int = 9,
):
    draw = ImageDraw.Draw(img)
    font = get_font(size, bold=bold)
    words = text.split()
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


def make_vignette(width: int, height: int, strength: float = 0.28) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width / 2, height / 2
    nx = (xx - cx) / (width / 2)
    ny = (yy - cy) / (height / 2)
    rr = np.sqrt(nx**2 + ny**2)
    return np.clip(1 - strength * rr**1.85, 0, 1).astype(np.float32)


def apply_grade(arr: np.ndarray) -> np.ndarray:
    img = Image.fromarray(arr)
    img = ImageEnhance.Contrast(img).enhance(CONFIG["contrast_boost"])
    img = ImageEnhance.Color(img).enhance(CONFIG["saturation_boost"])
    return np.array(img)


VIGNETTE = make_vignette(OUT_SIZE[0], OUT_SIZE[1], CONFIG["vignette_strength"])

# %% [markdown]
# ## Build 3D system coordinates
#
# Right ascension and declination place systems on the sky. Distance is compressed
# logarithmically so nearby and faraway systems can appear in one cinematic frame.


def compute_visual_xyz(sdf: pd.DataFrame, max_distance_pc: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra = np.deg2rad(sdf["ra"].to_numpy(float))
    dec = np.deg2rad(sdf["dec"].to_numpy(float))
    d_pc = np.clip(sdf["sy_dist"].to_numpy(float), 0.001, max_distance_pc)

    # Log compression keeps the nearby Solar neighborhood visible without throwing away distant systems.
    r = np.log1p(d_pc) / np.log1p(max_distance_pc) * 155.0

    x = r * np.cos(dec) * np.cos(ra)
    y = r * np.cos(dec) * np.sin(ra)
    z = r * np.sin(dec)
    return x, y, z


X0, Y0, Z0 = compute_visual_xyz(video_systems, CONFIG["max_distance_pc_for_visual"])

SYS = {
    "x": X0,
    "y": Y0,
    "z": Z0,
    "hostname": video_systems["hostname"].to_numpy(object),
    "planet_count": video_systems["planet_count"].to_numpy(int),
    "highlight": video_systems["highlight"].to_numpy(object),
    "group": video_systems["visual_group"].to_numpy(object),
    "distance": video_systems["sy_dist"].to_numpy(float),
    "teff": video_systems["st_teff"].to_numpy(float),
    "vmag": video_systems["sy_vmag"].to_numpy(float),
    "disc_year_min": video_systems["disc_year_min"].to_numpy(float),
}

# %% [markdown]
# ## Frame design

CAPTIONS = [
    (0.5, 5.6, "Every point is a star system where planets have been confirmed."),
    (7.0, 15.0, "The nearest systems sit only a few parsecs from the Sun."),
    (15.4, 25.5, "Some worlds are rocky-size. Others are mini-Neptunes or giant planets."),
    (26.0, 37.5, "A few names changed astronomy: 51 Peg b, TRAPPIST-1, Proxima Centauri b."),
    (38.0, 48.8, "This map is not empty space. It is a growing atlas of other solar systems."),
    (49.2, 57.2, "Each discovery is a coordinate in the search for worlds like our own."),
]

SHOT_PLAN = [
    {
        "name": "title",
        "start": 0.0,
        "end": 7.0,
        "yaw_start": -28,
        "yaw_end": -12,
        "pitch_start": -8,
        "pitch_end": -4,
        "camera_z_start": 305,
        "camera_z_end": 265,
        "caption": "Confirmed exoplanet systems from real catalog data",
    },
    {
        "name": "nearby_flythrough",
        "start": 7.0,
        "end": 21.0,
        "yaw_start": -12,
        "yaw_end": 12,
        "pitch_start": -4,
        "pitch_end": 8,
        "camera_z_start": 265,
        "camera_z_end": 215,
        "caption": "The Solar neighborhood begins to fill with known worlds",
    },
    {
        "name": "planet_types",
        "start": 21.0,
        "end": 35.0,
        "yaw_start": 12,
        "yaw_end": 42,
        "pitch_start": 8,
        "pitch_end": -7,
        "camera_z_start": 215,
        "camera_z_end": 190,
        "caption": "Color groups separate rough planet-system types",
    },
    {
        "name": "famous_systems",
        "start": 35.0,
        "end": 49.0,
        "yaw_start": 42,
        "yaw_end": 65,
        "pitch_start": -7,
        "pitch_end": 5,
        "camera_z_start": 190,
        "camera_z_end": 170,
        "caption": "Famous systems glow brighter for orientation",
    },
    {
        "name": "outro",
        "start": 49.0,
        "end": CONFIG["duration_s"],
        "yaw_start": 65,
        "yaw_end": 85,
        "pitch_start": 5,
        "pitch_end": 0,
        "camera_z_start": 170,
        "camera_z_end": 250,
        "caption": "The archive keeps growing as new worlds are confirmed",
    },
]


def get_shot(t: float) -> Dict:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def shot_state(t: float) -> Tuple[Dict, float, float, float]:
    shot = get_shot(t)
    u = (t - shot["start"]) / max(shot["end"] - shot["start"], 1e-6)
    e = ease_in_out_sine(u)
    yaw = lerp(shot["yaw_start"], shot["yaw_end"], e)
    pitch = lerp(shot["pitch_start"], shot["pitch_end"], e)
    camera_z = lerp(shot["camera_z_start"], shot["camera_z_end"], e)
    return shot, yaw, pitch, camera_z


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


def rotate_xyz(x, y, z, yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    # Yaw around Z.
    x1 = x * np.cos(yaw) - y * np.sin(yaw)
    y1 = x * np.sin(yaw) + y * np.cos(yaw)
    z1 = z

    # Pitch around X.
    x2 = x1
    y2 = y1 * np.cos(pitch) - z1 * np.sin(pitch)
    z2 = y1 * np.sin(pitch) + z1 * np.cos(pitch)

    # Roll around Z.
    x3 = x2 * np.cos(roll) - y2 * np.sin(roll)
    y3 = x2 * np.sin(roll) + y2 * np.cos(roll)
    z3 = z2
    return x3, y3, z3


def project_points(x, y, z, yaw: float, pitch: float, camera_z: float, width: int, height: int, t: float):
    # Slow forward drift creates parallax without changing the catalog coordinates.
    drift = 14.0 * smoothstep(t / CONFIG["duration_s"])
    xr, yr, zr = rotate_xyz(x, y, z - drift, yaw, pitch, roll_deg=0.9 * math.sin(t * 0.17))
    zcam = zr + camera_z
    zcam = np.maximum(zcam, 15.0)
    scale = CONFIG["camera_focal_length"] / zcam
    px = width / 2 + xr * scale
    py = height * 0.50 + yr * scale
    return px, py, zcam, scale


def generate_starfield(width: int, height: int, count: int, seed: int = 22):
    rng = np.random.default_rng(seed)
    stars = []
    for _ in range(count):
        stars.append({
            "x": float(rng.uniform(0, width)),
            "y": float(rng.uniform(0, height)),
            "r": float(rng.uniform(0.45, 2.2)),
            "a": int(rng.integers(28, 165)),
            "phase": float(rng.uniform(0, 2*np.pi)),
            "drift": float(rng.uniform(-16, 16)),
        })
    return stars


def generate_dust(count: int, seed: int = 88):
    rng = np.random.default_rng(seed)
    dust = []
    for _ in range(count):
        dust.append({
            "x": float(rng.uniform(-180, 180)),
            "y": float(rng.uniform(-220, 220)),
            "z": float(rng.uniform(-130, 130)),
            "r": float(rng.uniform(0.9, 3.4)),
            "phase": float(rng.uniform(0, 2*np.pi)),
        })
    return dust


STARS = generate_starfield(OUT_SIZE[0], OUT_SIZE[1], CONFIG["background_star_count"], seed=18)
DUST = generate_dust(CONFIG["dust_particle_count"], seed=47)


def render_starfield(width: int, height: int, stars, t: float) -> Image.Image:
    img = Image.new("RGBA", (width, height), (1, 3, 14, 255))
    draw = ImageDraw.Draw(img)

    # Faint blue-black nebula wash.
    nebula = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    nd = ImageDraw.Draw(nebula)
    cx = width * (0.34 + 0.04 * math.sin(t * 0.05))
    cy = height * (0.38 + 0.03 * math.cos(t * 0.06))
    for r, a in [(700, 20), (520, 28), (350, 34), (220, 30)]:
        nd.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(30, 55, 115, a))
    nebula = nebula.filter(ImageFilter.GaussianBlur(60))
    img.alpha_composite(nebula)

    for s in stars:
        x = (s["x"] + 0.36 * s["drift"] * t) % width
        y = (s["y"] + 0.09 * s["drift"] * t) % height
        twinkle = 0.74 + 0.26 * math.sin(1.15*t + s["phase"])
        a = int(s["a"] * twinkle)
        r = s["r"]
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(225, 235, 255, a))
    return img


def draw_dust(canvas: Image.Image, yaw: float, pitch: float, camera_z: float, t: float):
    if not DUST:
        return
    x = np.array([d["x"] for d in DUST], dtype=float)
    y = np.array([d["y"] for d in DUST], dtype=float)
    z = np.array([d["z"] for d in DUST], dtype=float)
    phase = np.array([d["phase"] for d in DUST], dtype=float)
    radii = np.array([d["r"] for d in DUST], dtype=float)

    z = z + 12 * np.sin(t * 0.12 + phase)
    px, py, zcam, scale = project_points(x, y, z, yaw, pitch, camera_z, canvas.size[0], canvas.size[1], t)
    order = np.argsort(zcam)[::-1]
    draw = ImageDraw.Draw(canvas)
    for idx in order:
        if px[idx] < -20 or px[idx] > canvas.size[0] + 20 or py[idx] < -20 or py[idx] > canvas.size[1] + 20:
            continue
        r = float(np.clip(radii[idx] * scale[idx] * 2.2, 0.6, 4.8))
        a = int(np.clip(34 * (1.7 - scale[idx]), 10, 42))
        draw.ellipse((px[idx]-r, py[idx]-r, px[idx]+r, py[idx]+r), fill=(160, 180, 255, a))


def star_temperature_tint(teff: float, base_color: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Small visual tint by stellar temperature; still editorial, not calibrated color."""
    if not np.isfinite(teff):
        return base_color
    r, g, b, a = base_color
    if teff >= 6500:
        return (min(255, int(r * 0.88)), min(255, int(g * 0.96)), min(255, int(b * 1.15)), a)
    if teff <= 3800:
        return (min(255, int(r * 1.16)), min(255, int(g * 0.92)), min(255, int(b * 0.78)), a)
    return base_color


def draw_system_points(canvas: Image.Image, yaw: float, pitch: float, camera_z: float, t: float):
    px, py, zcam, scale = project_points(SYS["x"], SYS["y"], SYS["z"], yaw, pitch, camera_z, canvas.size[0], canvas.size[1], t)
    order = np.argsort(zcam)[::-1]  # far to near

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for idx in order:
        if px[idx] < -35 or px[idx] > canvas.size[0] + 35 or py[idx] < -35 or py[idx] > canvas.size[1] + 35:
            continue
        group = SYS["group"][idx]
        color = star_temperature_tint(SYS["teff"][idx], GROUP_COLORS.get(group, (210, 220, 255, 170)))
        planet_count = int(SYS["planet_count"][idx])
        highlight = isinstance(SYS["highlight"][idx], str) and len(SYS["highlight"][idx]) > 0

        # Apparent size is mostly editorial: nearby, multi-planet, and highlighted systems read better.
        dist_bonus = np.clip(1.25 - SYS["distance"][idx] / CONFIG["max_distance_pc_for_visual"], 0.15, 1.25)
        size = (1.5 + 0.36 * min(planet_count, 8) + 1.3 * dist_bonus) * np.clip(scale[idx], 1.2, 4.2)
        size = float(np.clip(size, 1.2, 8.5))

        if highlight:
            glow = size * (4.2 + 0.4 * math.sin(t * 1.2))
            draw.ellipse((px[idx]-glow, py[idx]-glow, px[idx]+glow, py[idx]+glow), fill=(255, 220, 130, 38))
            draw.ellipse((px[idx]-size*1.35, py[idx]-size*1.35, px[idx]+size*1.35, py[idx]+size*1.35), fill=(255, 245, 190, 245))
        else:
            draw.ellipse((px[idx]-size, py[idx]-size, px[idx]+size, py[idx]+size), fill=color)

        # Multi-planet systems get a subtle ring.
        if planet_count >= 3 or highlight:
            ring = size * (2.0 + 0.08 * math.sin(t * 0.7 + idx))
            draw.ellipse((px[idx]-ring, py[idx]-ring, px[idx]+ring, py[idx]+ring), outline=(color[0], color[1], color[2], 70), width=1)

        # Decorative planet beads near highlighted systems.
        if highlight:
            beads = min(planet_count, 7)
            for j in range(beads):
                ang = t * (0.45 + 0.05*j) + j * 2*np.pi / max(beads, 1)
                rr = size * (2.8 + 0.22*j)
                bx = px[idx] + math.cos(ang) * rr
                by = py[idx] + math.sin(ang) * rr * 0.55
                br = max(1.1, size * 0.25)
                draw.ellipse((bx-br, by-br, bx+br, by+br), fill=(200, 225, 255, 180))

    canvas.alpha_composite(overlay)
    draw_system_labels(canvas, px, py, zcam, t)


def draw_system_labels(canvas: Image.Image, px, py, zcam, t: float):
    label_overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    label_count = 0
    label_alpha_global = smoothstep((t - 27.0) / 8.0)

    # First label famous systems, then a few very nearby systems.
    candidates = []
    for idx in range(len(SYS["hostname"])):
        highlight = isinstance(SYS["highlight"][idx], str) and len(SYS["highlight"][idx]) > 0
        nearby = SYS["distance"][idx] <= CONFIG["label_distance_pc_limit"]
        if highlight or (nearby and SYS["planet_count"][idx] >= 2):
            candidates.append((0 if highlight else 1, SYS["distance"][idx], idx))
    candidates.sort()

    used_boxes: List[Tuple[int, int, int, int]] = []
    for _, _, idx in candidates:
        if label_count >= 16:
            break
        x = int(px[idx])
        y = int(py[idx])
        if x < 35 or x > canvas.size[0] - 210 or y < 250 or y > canvas.size[1] - 320:
            continue
        text = str(SYS["highlight"][idx] or SYS["hostname"][idx])
        if not text:
            continue
        # Avoid label overlap crudely.
        box = (x + 14, y - 34, x + 260, y + 10)
        if any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in used_boxes):
            continue
        used_boxes.append(box)
        alpha = int(230 * label_alpha_global)
        if alpha <= 4:
            continue
        draw_text(label_overlay, text, (x + 14, y - 12), size=24, fill=(255, 242, 205, alpha), bold=True, stroke=2)
        label_count += 1

    canvas.alpha_composite(label_overlay)


def draw_legend(canvas: Image.Image, t: float):
    alpha = int(220 * smoothstep((t - 18) / 5.0) * (1 - smoothstep((t - 46) / 6.0)))
    if alpha <= 5:
        return
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x0, y0 = 62, 272
    draw.rounded_rectangle((x0 - 18, y0 - 22, x0 + 420, y0 + 250), radius=24, fill=(0, 0, 0, 95))
    canvas.alpha_composite(overlay)

    items = [
        ("warm rocky-size", GROUP_COLORS["warm rocky-size system"]),
        ("rocky-size", GROUP_COLORS["rocky-size system"]),
        ("sub-Neptune", GROUP_COLORS["sub-Neptune system"]),
        ("hot / gas giant", GROUP_COLORS["hot giant system"]),
        ("highlighted world", GROUP_COLORS["highlight system"]),
    ]
    draw_text(canvas, "Visual groups", (x0, y0), size=25, fill=(230, 238, 255, alpha), bold=True, stroke=1)
    y = y0 + 42
    d = ImageDraw.Draw(canvas)
    for label, color in items:
        d.ellipse((x0, y, x0 + 18, y + 18), fill=(color[0], color[1], color[2], min(alpha, color[3])))
        draw_text(canvas, label, (x0 + 34, y - 2), size=22, fill=(220, 228, 245, alpha), bold=False, stroke=1)
        y += 38


def draw_discovery_timeline(canvas: Image.Image, t: float):
    alpha = int(215 * smoothstep((t - 43.0) / 5.0))
    if alpha <= 5:
        return
    years = planet_df["disc_year"].dropna().astype(int)
    if years.empty:
        return
    counts = years.value_counts().sort_index()
    min_y, max_y = int(counts.index.min()), int(counts.index.max())
    full_years = np.arange(min_y, max_y + 1)
    vals = np.array([counts.get(y, 0) for y in full_years], dtype=float)
    vals = vals / max(vals.max(), 1)

    x0, y0 = 72, canvas.size[1] - 450
    w, h = canvas.size[0] - 144, 128
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((x0 - 22, y0 - 54, x0 + w + 22, y0 + h + 54), radius=25, fill=(0, 0, 0, 116))
    canvas.alpha_composite(overlay)
    draw_text(canvas, "Discovery timeline", (x0, y0 - 36), size=24, fill=(230, 238, 255, alpha), bold=True, stroke=1)

    d = ImageDraw.Draw(canvas)
    n = len(vals)
    if n > 0:
        bar_w = max(1, w / n)
        for k, val in enumerate(vals):
            x = x0 + k * bar_w
            bh = val * h
            d.rectangle((x, y0 + h - bh, x + max(1, bar_w - 0.5), y0 + h), fill=(150, 190, 255, int(alpha * 0.75)))
    d.line((x0, y0 + h, x0 + w, y0 + h), fill=(220, 230, 255, int(alpha * 0.55)), width=1)
    draw_text(canvas, str(min_y), (x0, y0 + h + 18), size=18, fill=(215, 224, 245, alpha), stroke=1)
    draw_text(canvas, str(max_y), (x0 + w - 62, y0 + h + 18), size=18, fill=(215, 224, 245, alpha), stroke=1)


def add_text_layers(canvas: Image.Image, t: float, shot: Dict):
    # Title block.
    title_alpha = int(255 * smoothstep((t - 0.3) / 1.2) * (1 - smoothstep((t - 5.8) / 1.1)))
    if title_alpha > 5:
        draw_text(canvas, CONFIG["title_text"], (62, 120), size=60, fill=(255, 255, 255, title_alpha), bold=True)
        draw_text(canvas, CONFIG["subtitle_text"], (66, 204), size=28, fill=(218, 232, 255, min(225, title_alpha)), bold=False)

    # Corner label.
    if t > 6:
        draw_text(canvas, shot["caption"], (62, 78), size=24, fill=(200, 220, 255, 190), bold=False, stroke=1)

    # Main caption box.
    cap = caption_at(t)
    if cap:
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        y0 = canvas.size[1] - 250
        draw.rounded_rectangle((46, y0, canvas.size[0] - 46, y0 + 126), radius=26, fill=(0, 0, 0, 138))
        canvas.alpha_composite(overlay)
        draw_wrapped_text(canvas, cap, (70, y0 + 28), max_width=canvas.size[0] - 140, size=32, fill=(255, 255, 255, 245))

    # Credits / scientific note in the ending.
    note_alpha = int(220 * smoothstep((t - 49) / 3.0))
    if note_alpha > 5:
        draw_wrapped_text(canvas, CONFIG["credit_text"], (68, canvas.size[1] - 118), max_width=940, size=21, fill=(225, 230, 245, note_alpha))
        draw_wrapped_text(canvas, CONFIG["scientific_note"], (68, canvas.size[1] - 84), max_width=940, size=19, fill=(205, 215, 238, note_alpha))


def render_frame(t: float) -> np.ndarray:
    shot, yaw, pitch, camera_z = shot_state(t)

    canvas = render_starfield(OUT_SIZE[0], OUT_SIZE[1], STARS, t)
    draw_dust(canvas, yaw, pitch, camera_z, t)
    draw_system_points(canvas, yaw, pitch, camera_z, t)
    draw_legend(canvas, t)
    draw_discovery_timeline(canvas, t)
    add_text_layers(canvas, t, shot)

    arr_img = np.array(canvas.convert("RGB"))
    arr_img = apply_grade(arr_img)
    arr_img = np.clip(arr_img.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)

    fade_in = smoothstep(t / 1.1)
    fade_out = 1.0 - smoothstep((t - (CONFIG["duration_s"] - 1.25)) / 1.1)
    arr_img = np.clip(arr_img.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)
    return arr_img

print("Cinematic renderer ready.")

# %% [markdown]
# ## Preview frames

preview_times = [1.0, 11.0, 24.0, 40.0, CONFIG["duration_s"] - 1.0]
preview_arrays = []
for t in tqdm(preview_times, desc="Preview frames"):
    arr_img = render_frame(float(t))
    preview_arrays.append(arr_img)
    Image.fromarray(arr_img).save(PREVIEW_DIR / f"preview_{int(t):02d}s.png")

fig, axes = plt.subplots(1, len(preview_arrays), figsize=(18, 9))
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
# ## Render the full vertical MP4
#
# This is the longest section. During testing, set:
#
# ```python
# CONFIG["fps"] = 12
# CONFIG["duration_s"] = 12
# ```

RAW_VIDEO_PATH = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
SUBBED_VIDEO_PATH = OUTPUT_ROOT / f"{CONFIG['output_basename']}_subbed.mp4"
AUDIO_VIDEO_PATH = OUTPUT_ROOT / f"{CONFIG['output_basename']}_with_audio.mp4"
FINAL_VIDEO_PATH = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"

nframes = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
times = np.arange(nframes) / CONFIG["fps"]

print(f"Rendering {nframes:,} frames at {CONFIG['video_width']}×{CONFIG['video_height']} ...")
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
# Set `CONFIG["audio_path"]` to a local `.mp3` or `.wav` file before running this section.
# Use music or ambience only if you have permission to use it.


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
        "-vf", f"subtitles={SRT_PATH}:force_style=Fontname=DejaVu Sans,Fontsize=22,Outline=1.2,BorderStyle=3,MarginV=90",
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
# # Suggested narration / description
#
# Suggested voiceover:
#
# > Every point in this map is a star where planets have been confirmed.  
# > Some are nearby, only a few parsecs away.  
# > Some host compact systems of rocky-size worlds.  
# > Others hold hot giants, mini-Neptunes, and planets unlike anything in our Solar System.  
# > Names like 51 Pegasi b, Proxima Centauri b, TRAPPIST-1, and K2-18 b mark turning points in the search for other worlds.  
# > This is not empty space.  
# > It is an atlas that keeps growing with every discovery.
#
# Suggested YouTube Shorts caption:
#
# A cinematic map of confirmed exoplanet systems using real NASA Exoplanet Archive data. Every glowing point represents a star with known worlds — from nearby systems like Proxima Centauri to famous discoveries like TRAPPIST-1 and 51 Pegasi b.
#
# #Space #Exoplanets #NASA #Astronomy #ScienceShorts #DeepSpace #Universe
