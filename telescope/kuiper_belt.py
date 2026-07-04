# %% [markdown]
# # Cinematic YouTube Shorts from Real Kuiper Belt Data
# 
# This notebook creates a **vertical 1080×1920 cinematic YouTube Short** from **real Trans-Neptunian Object / Kuiper Belt orbital data**.
# 
# Instead of using a telescope image as the base layer, this notebook uses real orbital elements from NASA/JPL's **Small-Body Database Query API**. That is the better scientific representation for the Kuiper Belt because most Kuiper Belt objects are too small and faint to show as detailed public images, but their orbital architecture is measurable and data-rich.
# 
# ## What this notebook produces
# 
# - Downloads real TNO / Kuiper Belt object orbital data from JPL SBDB.
# - Builds a cleaned dataframe of real objects with:
#   - semimajor axis `a`,
#   - eccentricity `e`,
#   - inclination `i`,
#   - perihelion `q`,
#   - aphelion `ad`,
#   - mean anomaly `ma`,
#   - orbital period `per_y`,
#   - optional physical fields such as `H`, `diameter`, and `albedo`.
# - Creates scientific preview plots:
#   - semimajor-axis distribution,
#   - eccentricity vs semimajor axis,
#   - top-down orbit map.
# - Renders a cinematic vertical video with:
#   - animated orbit trails,
#   - real object positions estimated from orbital elements,
#   - Neptune and Kuiper Belt guide rings,
#   - Pluto / Arrokoth / dwarf-planet labels when present,
#   - parallax starfield overlay,
#   - subtitles and credits.
# - Exports:
#   - raw vertical MP4,
#   - optional subtitle sidecar `.srt`,
#   - optional audio-muxed MP4,
#   - final MP4.
# 
# ## Scientific fidelity note
# 
# The **data layer** comes from real JPL SBDB orbital elements.  
# The **cinematic layer** contains editorial effects: starfield, glow, zoom, camera rotation, text, vignette, and particles.
# 
# The animation uses a simplified two-body Keplerian visualisation. It is excellent for storytelling and orbit-structure intuition, but it is **not** a replacement for high-precision ephemerides such as JPL Horizons.

# %%
# Recommended installation:
#   pip install numpy pandas matplotlib pillow imageio imageio-ffmpeg requests tqdm
#
# Optional for notebook execution from command line:
#   pip install jupyter nbconvert

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import urllib.parse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

plt.rcParams["figure.figsize"] = (9, 9)
plt.rcParams["axes.grid"] = True

print("Imports loaded.")

# %% [markdown]
# ## Data source
# 
# This notebook uses the **JPL Small-Body Database Query API**:
# 
# ```text
# https://ssd-api.jpl.nasa.gov/sbdb_query.api
# ```
# 
# The query asks for objects with JPL orbit class code:
# 
# ```text
# sb-class=TNO
# ```
# 
# In the JPL filter documentation, `TNO` means **Trans-Neptunian Object**, defined as objects with orbits outside Neptune, using `a > 30.1 au`.
# 
# Important interpretation:
# 
# - This is real catalog data.
# - The retrieved objects are not all guaranteed to be visually located in the classical 30–50 AU Kuiper Belt at the rendered moment.
# - The video uses the TNO catalog to show the larger trans-Neptunian population, with visual guide bands for the Kuiper Belt region.

# %%
OUTPUT_ROOT = Path("kuiper_belt_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
FRAME_DIR = OUTPUT_ROOT / "frames"
PREVIEW_DIR = OUTPUT_ROOT / "previews"

for p in [OUTPUT_ROOT, DATA_ROOT, FRAME_DIR, PREVIEW_DIR]:
    p.mkdir(parents=True, exist_ok=True)

CONFIG = {
    # Final delivery
    "video_width": 1080,
    "video_height": 1920,
    "fps": 24,
    "duration_s": 58,
    "output_basename": "kuiper_belt_real_data_short",

    # JPL SBDB query
    "jpl_api_url": "https://ssd-api.jpl.nasa.gov/sbdb_query.api",
    "sb_class": "TNO",
    "sb_kind": "a",
    "fields": [
        "spkid", "full_name", "pdes", "name", "class",
        "epoch", "e", "a", "q", "i", "om", "w", "ma", "per_y", "ad",
        "H", "diameter", "albedo",
        "condition_code", "soln_date", "n_obs_used", "data_arc"
    ],
    "full_precision": True,
    "page_limit": 5000,

    # Visual selection limits
    "max_objects_for_points": 2500,
    "max_orbits_drawn": 380,
    "orbit_sample_points": 240,

    # Science / visual guide rings in AU
    "neptune_a_au": 30.07,
    "kuiper_inner_au": 30.0,
    "kuiper_outer_au": 50.0,
    "pluto_resonance_3_2_au": 39.4,
    "resonance_2_1_au": 47.8,

    # Rendering style
    "background_star_count": 900,
    "decorative_particle_count": 150,
    "vignette_strength": 0.24,
    "contrast_boost": 1.10,
    "saturation_boost": 1.06,

    # Text
    "title_text": "The Kuiper Belt, Built from Real Orbits",
    "subtitle_text": "JPL Small-Body Database • TNO orbital elements • cinematic map",
    "credit_text": "Data: NASA/JPL Small-Body Database Query API",
    "scientific_note": "Positions are Keplerian visual estimates from catalog orbital elements, not precision ephemerides.",

    # Optional audio / subtitles
    "audio_path": None,       # Example: "audio/space_ambient.mp3"
    "burn_subtitles": False,  # requires ffmpeg with subtitle support
    "write_subtitle_sidecar": True,
}

print("Configuration ready.")

# %% [markdown]
# ## Download the real JPL dataset
# 
# This cell downloads catalog data directly from JPL.
# 
# The notebook caches the JSON and CSV locally so you can rerun video rendering without repeatedly calling the API. Delete the files in `kuiper_belt_short_output/data/` if you want a fresh query.

# %%
def request_json(url: str, params: Dict, timeout: int = 60) -> Dict:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if "code" in payload and str(payload.get("code")) != "200":
        raise RuntimeError(f"JPL API returned code={payload.get('code')}: {payload}")
    return payload

def fetch_jpl_sbdb_tnos(config: Dict, force_refresh: bool = False) -> pd.DataFrame:
    json_path = DATA_ROOT / "jpl_sbdb_tno_raw.json"
    csv_path = DATA_ROOT / "jpl_sbdb_tno_clean.csv"

    if csv_path.exists() and not force_refresh:
        print(f"Using cached CSV: {csv_path}")
        return pd.read_csv(csv_path)

    fields = ",".join(config["fields"])
    all_rows = []
    field_names = None
    limit = int(config["page_limit"])
    offset = 0

    while True:
        params = {
            "fields": fields,
            "sb-class": config["sb_class"],
            "sb-kind": config["sb_kind"],
            "full-prec": "true" if config.get("full_precision", True) else "false",
            "limit": str(limit),
            "limit-from": str(offset),
            "sort": "a",
        }
        print(f"Requesting JPL SBDB TNO rows {offset} to {offset + limit - 1} ...")
        payload = request_json(config["jpl_api_url"], params=params)

        if field_names is None:
            field_names = payload.get("fields", config["fields"])

        rows = payload.get("data", [])
        if not rows:
            break

        all_rows.extend(rows)

        # If fewer than limit rows came back, pagination is finished.
        if len(rows) < limit:
            break

        offset += limit

        # Safety stop for notebook use. Increase if needed.
        if offset >= 30000:
            print("Stopping at 30,000 rows for safety.")
            break

    if not all_rows:
        raise RuntimeError("No rows returned from JPL SBDB. Check internet access or API parameters.")

    raw_payload = {"fields": field_names, "data": all_rows}
    json_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")

    df = pd.DataFrame(all_rows, columns=field_names)

    numeric_cols = [
        "epoch", "e", "a", "q", "i", "om", "w", "ma", "per_y", "ad",
        "H", "diameter", "albedo", "n_obs_used", "data_arc"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Basic physically useful filters for an orbit-map video.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["a", "e", "i", "om", "w", "ma", "per_y"])
    df = df[(df["a"] > 0) & (df["e"] >= 0) & (df["e"] < 0.99) & (df["per_y"] > 0)].copy()
    df = df.sort_values(["a", "H"], na_position="last").reset_index(drop=True)

    df.to_csv(csv_path, index=False)
    print(f"Saved cleaned data: {csv_path}")
    print(f"Objects retained for visualisation: {len(df):,}")

    return df

df = fetch_jpl_sbdb_tnos(CONFIG, force_refresh=False)
df.head()

# %% [markdown]
# ## Data cleaning and object groups
# 
# The colour categories below are **visual labels**, not visible-light colours.
# 
# They help the viewer read the structure of the data:
# 
# - **Named highlight**: Pluto, Arrokoth, Eris, Makemake, Haumea, Quaoar, Sedna, Orcus.
# - **Cold classical-ish**: low eccentricity, low inclination, semimajor axis near the main Kuiper Belt.
# - **High-inclination**: dynamically tilted orbits.
# - **Distant / detached-ish**: high semimajor axis or perihelion far from Neptune.
# - **Other TNOs**: objects that do not match those simplified visual buckets.
# 
# This is intentionally simplified for a 60-second science-communication video.

# %%
HIGHLIGHT_NAMES = [
    "Pluto", "Arrokoth", "Eris", "Makemake", "Haumea", "Quaoar", "Sedna", "Orcus", "Gonggong"
]

def object_label(row: pd.Series) -> str:
    for col in ["name", "full_name", "pdes"]:
        value = row.get(col, "")
        if pd.isna(value):
            continue
        value = str(value)
        if value.strip():
            return value.strip()
    return f"SPK {row.get('spkid', '')}".strip()

def contains_highlight(row: pd.Series) -> Optional[str]:
    combined = " ".join(str(row.get(c, "")) for c in ["full_name", "name", "pdes"])
    for target in HIGHLIGHT_NAMES:
        if target.lower() in combined.lower():
            return target
    # Arrokoth often appears as 486958 or 2014 MU69
    if "486958" in combined or "2014 MU69" in combined:
        return "Arrokoth"
    if "134340" in combined:
        return "Pluto"
    return None

def classify_tno(row: pd.Series) -> str:
    h = contains_highlight(row)
    if h:
        return "named highlight"
    a = float(row["a"])
    e = float(row["e"])
    inc = float(row["i"])
    q = float(row.get("q", np.nan))
    if 42.0 <= a <= 47.5 and e < 0.10 and inc < 5:
        return "cold classical-ish"
    if inc >= 20:
        return "high inclination"
    if a >= 80 or (np.isfinite(q) and q >= 40 and a >= 50):
        return "distant / detached-ish"
    if 30 <= a <= 55:
        return "main belt region"
    return "other TNO"

df["display_label"] = df.apply(object_label, axis=1)
df["highlight"] = df.apply(contains_highlight, axis=1)
df["visual_group"] = df.apply(classify_tno, axis=1)

group_counts = df["visual_group"].value_counts()
print(group_counts)

df[df["highlight"].notna()][["display_label", "highlight", "a", "e", "i", "q", "ad", "H", "diameter"]].head(20)

# %% [markdown]
# ## Scientific preview plots
# 
# These preview plots are not the final video. They help you understand what the video is about before rendering:
# 
# 1. How far TNOs orbit from the Sun.
# 2. How eccentric the orbits are.
# 3. Where the major guide rings sit relative to the object distribution.

# %%
fig, ax = plt.subplots(figsize=(10, 5))
df["a"].clip(upper=200).hist(bins=80, ax=ax)
ax.axvspan(CONFIG["kuiper_inner_au"], CONFIG["kuiper_outer_au"], alpha=0.18, label="30–50 AU guide band")
ax.axvline(CONFIG["neptune_a_au"], linestyle="--", label="Neptune ~30 AU")
ax.set_title("Semimajor axis distribution of JPL SBDB TNOs")
ax.set_xlabel("Semimajor axis a [AU], clipped at 200 AU")
ax.set_ylabel("Object count")
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
sample = df[df["a"] <= 180].copy()
for group, sdf in sample.groupby("visual_group"):
    ax.scatter(sdf["a"], sdf["e"], s=9, alpha=0.6, label=group)
ax.axvspan(CONFIG["kuiper_inner_au"], CONFIG["kuiper_outer_au"], alpha=0.12)
ax.set_title("TNO eccentricity vs semimajor axis")
ax.set_xlabel("Semimajor axis a [AU]")
ax.set_ylabel("Eccentricity e")
ax.set_ylim(0, min(1, max(0.8, sample["e"].quantile(0.995))))
ax.legend(markerscale=2, fontsize=8)
plt.show()

# %% [markdown]
# ## Orbit mathematics used for the visualisation
# 
# For each object, the notebook uses the catalog orbital elements:
# 
# - `a`: semimajor axis,
# - `e`: eccentricity,
# - `i`: inclination,
# - `om`: longitude of ascending node,
# - `w`: argument of perihelion,
# - `ma`: mean anomaly,
# - `per_y`: orbital period in Julian years.
# 
# The code solves Kepler's equation to estimate a point along each orbit for the animation. The orbit trails are drawn from the same elements.
# 
# This is a cinematic two-body visual map, not an N-body precision ephemeris.

# %%
def deg2rad(x):
    return np.deg2rad(x.astype(float) if hasattr(x, "astype") else x)

def solve_kepler_newton(M: np.ndarray, e: np.ndarray, iterations: int = 8) -> np.ndarray:
    """Solve M = E - e sin(E) for elliptic orbits."""
    M = np.asarray(M, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    E = M.copy()
    # Better initial guess for high-e objects
    E = np.where(e > 0.75, np.pi * np.sign(np.sin(M)), E)
    E = np.where(E == 0, M, E)

    for _ in range(iterations):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        E = E - f / np.where(np.abs(fp) < 1e-10, 1e-10, fp)
    return E

def elements_to_xyz(
    a: np.ndarray,
    e: np.ndarray,
    inc_deg: np.ndarray,
    om_deg: np.ndarray,
    w_deg: np.ndarray,
    ma_deg: np.ndarray,
    per_y: np.ndarray,
    dt_years: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized orbital-element to heliocentric ecliptic XYZ conversion."""
    a = np.asarray(a, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    per_y = np.asarray(per_y, dtype=np.float64)

    mean_motion_deg_per_year = 360.0 / per_y
    M = np.deg2rad((np.asarray(ma_deg, dtype=np.float64) + mean_motion_deg_per_year * dt_years) % 360.0)

    E = solve_kepler_newton(M, e)
    cosE = np.cos(E)
    sinE = np.sin(E)

    r = a * (1 - e * cosE)
    nu = np.arctan2(np.sqrt(1 - e * e) * sinE, cosE - e)

    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    inc = np.deg2rad(np.asarray(inc_deg, dtype=np.float64))
    om = np.deg2rad(np.asarray(om_deg, dtype=np.float64))
    w = np.deg2rad(np.asarray(w_deg, dtype=np.float64))

    cosO, sinO = np.cos(om), np.sin(om)
    cosi, sini = np.cos(inc), np.sin(inc)
    cosw, sinw = np.cos(w), np.sin(w)

    # Rotation Rz(om) * Rx(i) * Rz(w)
    x = (cosO * cosw - sinO * sinw * cosi) * x_orb + (-cosO * sinw - sinO * cosw * cosi) * y_orb
    y = (sinO * cosw + cosO * sinw * cosi) * x_orb + (-sinO * sinw + cosO * cosw * cosi) * y_orb
    z = (sinw * sini) * x_orb + (cosw * sini) * y_orb

    return x, y, z

def orbit_trace(row: pd.Series, n: int = 240) -> np.ndarray:
    """Return Nx3 array for one full orbit trace."""
    a = float(row["a"])
    e = float(row["e"])
    inc = float(row["i"])
    om = float(row["om"])
    w = float(row["w"])

    nu = np.linspace(0, 2 * np.pi, n)
    r = a * (1 - e * e) / np.maximum(1 + e * np.cos(nu), 1e-6)
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    inc_r, om_r, w_r = np.deg2rad([inc, om, w])
    cosO, sinO = np.cos(om_r), np.sin(om_r)
    cosi, sini = np.cos(inc_r), np.sin(inc_r)
    cosw, sinw = np.cos(w_r), np.sin(w_r)

    x = (cosO * cosw - sinO * sinw * cosi) * x_orb + (-cosO * sinw - sinO * cosw * cosi) * y_orb
    y = (sinO * cosw + cosO * sinw * cosi) * x_orb + (-sinO * sinw + cosO * cosw * cosi) * y_orb
    z = (sinw * sini) * x_orb + (cosw * sini) * y_orb

    return np.vstack([x, y, z]).T

# Quick sanity check using a few rows.
x, y, z = elements_to_xyz(
    df["a"].head(5), df["e"].head(5), df["i"].head(5),
    df["om"].head(5), df["w"].head(5), df["ma"].head(5),
    df["per_y"].head(5), dt_years=0
)
print("Sample positions [AU]:")
print(np.vstack([x, y, z]).T)

# %% [markdown]
# ## Preview orbit map
# 
# The final video uses a fast PIL renderer, but this Matplotlib preview is useful for checking that the dataset loaded correctly.

# %%
fig, ax = plt.subplots(figsize=(9, 9))

preview = df[df["a"] <= 90].sample(min(500, len(df[df["a"] <= 90])), random_state=7)
for _, row in preview.iterrows():
    tr = orbit_trace(row, n=160)
    ax.plot(tr[:, 0], tr[:, 1], linewidth=0.35, alpha=0.18)

x, y, z = elements_to_xyz(
    preview["a"], preview["e"], preview["i"],
    preview["om"], preview["w"], preview["ma"], preview["per_y"],
    dt_years=0
)
ax.scatter(x, y, s=5, alpha=0.8)

# Guide rings
theta = np.linspace(0, 2*np.pi, 500)
for radius, label in [
    (CONFIG["neptune_a_au"], "Neptune"),
    (CONFIG["kuiper_inner_au"], "30 AU"),
    (CONFIG["kuiper_outer_au"], "50 AU"),
]:
    ax.plot(radius*np.cos(theta), radius*np.sin(theta), linestyle="--", linewidth=1, label=label)

ax.scatter([0], [0], s=80, marker="*", label="Sun")
ax.set_aspect("equal", "box")
ax.set_xlim(-95, 95)
ax.set_ylim(-95, 95)
ax.set_title("Preview: top-down TNO orbit map from JPL SBDB elements")
ax.set_xlabel("AU")
ax.set_ylabel("AU")
ax.legend()
plt.show()

# %% [markdown]
# ## Cinematic rendering helpers
# 
# The final video is built with a custom renderer so it can animate thousands of points efficiently.
# 
# Visual conventions:
# 
# - **Sun**: center glow.
# - **Neptune ring**: guide circle near 30 AU.
# - **Kuiper Belt band**: approximate 30–50 AU guide zone.
# - **Orbit lines**: real catalog orbital elements, simplified into two-body trails.
# - **Object dots**: estimated positions from the catalog elements at the animated time.
# - **Starfield and particles**: decorative overlays only.

# %%
OUT_SIZE = (CONFIG["video_width"], CONFIG["video_height"])

GROUP_COLORS = {
    "named highlight": (255, 224, 140, 255),
    "cold classical-ish": (120, 210, 255, 220),
    "main belt region": (170, 160, 255, 210),
    "high inclination": (255, 120, 170, 220),
    "distant / detached-ish": (255, 170, 90, 220),
    "other TNO": (190, 210, 255, 170),
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
    stroke=2,
    anchor="la",
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

def draw_multiline(
    img: Image.Image,
    lines: List[str],
    xy: Tuple[int, int],
    size: int = 42,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    line_spacing: int = 8,
):
    x, y = xy
    font = get_font(size, bold=bold)
    draw = ImageDraw.Draw(img)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 220))
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += (bbox[3] - bbox[1]) + line_spacing

def make_vignette(width: int, height: int, strength: float = 0.24) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width / 2, height / 2
    nx = (xx - cx) / (width / 2)
    ny = (yy - cy) / (height / 2)
    rr = np.sqrt(nx**2 + ny**2)
    return np.clip(1 - strength * rr**1.7, 0, 1).astype(np.float32)

def apply_grade(arr: np.ndarray) -> np.ndarray:
    img = Image.fromarray(arr)
    img = ImageEnhance.Contrast(img).enhance(CONFIG["contrast_boost"])
    img = ImageEnhance.Color(img).enhance(CONFIG["saturation_boost"])
    return np.array(img)

VIGNETTE = make_vignette(OUT_SIZE[0], OUT_SIZE[1], CONFIG["vignette_strength"])
print("Renderer helpers ready.")

# %%
# Select objects for video.
# Points: many objects, but clipped to a useful distance so the map remains readable.
video_df = df[df["a"] <= 220].copy()
if len(video_df) > CONFIG["max_objects_for_points"]:
    # Prefer brighter / named / closer objects for readability.
    video_df["_priority"] = (
        video_df["highlight"].notna().astype(int) * 1000
        + (100 - video_df["a"].clip(0, 100))
        + (30 - video_df["H"].fillna(30)).clip(-30, 30)
    )
    video_df = video_df.sort_values("_priority", ascending=False).head(CONFIG["max_objects_for_points"]).drop(columns=["_priority"])
video_df = video_df.reset_index(drop=True)

orbit_df = video_df.copy()
orbit_df["_orbit_priority"] = (
    orbit_df["highlight"].notna().astype(int) * 1000
    + (100 - orbit_df["a"].clip(0, 100))
    + (30 - orbit_df["H"].fillna(30)).clip(-30, 30)
)
orbit_df = orbit_df.sort_values("_orbit_priority", ascending=False).head(CONFIG["max_orbits_drawn"]).drop(columns=["_orbit_priority"])

# Precompute orbit traces.
orbit_traces = []
for _, row in tqdm(orbit_df.iterrows(), total=len(orbit_df), desc="Precomputing orbit traces"):
    tr = orbit_trace(row, n=CONFIG["orbit_sample_points"])
    orbit_traces.append({
        "trace": tr,
        "group": row["visual_group"],
        "label": row["display_label"],
        "highlight": row["highlight"],
        "a": row["a"],
    })

# Precompute arrays for current positions.
arr = {
    "a": video_df["a"].to_numpy(float),
    "e": video_df["e"].to_numpy(float),
    "i": video_df["i"].to_numpy(float),
    "om": video_df["om"].to_numpy(float),
    "w": video_df["w"].to_numpy(float),
    "ma": video_df["ma"].to_numpy(float),
    "per_y": video_df["per_y"].to_numpy(float),
    "H": video_df["H"].to_numpy(float) if "H" in video_df else np.full(len(video_df), np.nan),
    "group": video_df["visual_group"].to_numpy(object),
    "highlight": video_df["highlight"].to_numpy(object),
    "label": video_df["display_label"].to_numpy(object),
}

print(f"Video points: {len(video_df):,}")
print(f"Orbit trails: {len(orbit_df):,}")

# %%
def generate_starfield(width: int, height: int, count: int, seed: int = 44):
    rng = np.random.default_rng(seed)
    stars = []
    for _ in range(count):
        stars.append({
            "x": float(rng.uniform(0, width)),
            "y": float(rng.uniform(0, height)),
            "r": float(rng.uniform(0.5, 2.2)),
            "a": int(rng.integers(35, 170)),
            "phase": float(rng.uniform(0, 2*np.pi)),
            "drift": float(rng.uniform(-10, 10)),
        })
    return stars

def render_starfield(width: int, height: int, stars, t: float) -> Image.Image:
    img = Image.new("RGBA", (width, height), (2, 3, 12, 255))
    draw = ImageDraw.Draw(img)
    for s in stars:
        x = (s["x"] + 0.35 * s["drift"] * t) % width
        y = (s["y"] + 0.08 * s["drift"] * t) % height
        twinkle = 0.75 + 0.25 * math.sin(1.4*t + s["phase"])
        a = int(s["a"] * twinkle)
        r = s["r"]
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(230, 238, 255, a))
    return img

STARS = generate_starfield(OUT_SIZE[0], OUT_SIZE[1], CONFIG["background_star_count"], seed=18)

SHOT_PLAN = [
    {
        "name": "title_outer_system",
        "start": 0.0,
        "end": 7.0,
        "view_start": 125,
        "view_end": 95,
        "rotation_start": -18,
        "rotation_end": -7,
        "caption": "Beyond Neptune: a belt of icy worlds",
    },
    {
        "name": "real_catalog_reveal",
        "start": 7.0,
        "end": 20.0,
        "view_start": 95,
        "view_end": 72,
        "rotation_start": -7,
        "rotation_end": 8,
        "caption": "Each dot comes from real JPL orbital data",
    },
    {
        "name": "belt_structure",
        "start": 20.0,
        "end": 35.0,
        "view_start": 72,
        "view_end": 58,
        "rotation_start": 8,
        "rotation_end": 25,
        "caption": "The guide band marks the 30–50 AU Kuiper Belt region",
    },
    {
        "name": "new_horizons_targets",
        "start": 35.0,
        "end": 49.0,
        "view_start": 58,
        "view_end": 48,
        "rotation_start": 25,
        "rotation_end": 38,
        "caption": "Pluto and Arrokoth were visited by New Horizons",
    },
    {
        "name": "outro_explain",
        "start": 49.0,
        "end": CONFIG["duration_s"],
        "view_start": 48,
        "view_end": 82,
        "rotation_start": 38,
        "rotation_end": 48,
        "caption": "The glow is cinematic. The orbit data are real.",
    },
]

CAPTIONS = [
    (0.5, 5.6, "The Kuiper Belt is not one object — it is a population."),
    (7.2, 15.8, "This video maps real Trans-Neptunian Object orbital elements."),
    (16.0, 25.5, "Neptune marks the inner edge; the 30–50 AU band frames the classical belt."),
    (25.7, 36.8, "Every orbit trail is drawn from semimajor axis, eccentricity, inclination, and angle elements."),
    (37.0, 48.5, "Pluto and Arrokoth anchor the story of New Horizons exploration."),
    (49.0, 57.4, "Colours are categories, not real surface colours. Motion is cinematic, data is real."),
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
    view_au = lerp(shot["view_start"], shot["view_end"], e)
    rot_deg = lerp(shot["rotation_start"], shot["rotation_end"], e)
    # dt_years exaggerates orbital motion so viewers can feel scale.
    dt_years = 0.0 + 115.0 * smoothstep(t / CONFIG["duration_s"])
    return shot, view_au, rot_deg, dt_years

def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None

def project_xy(x, y, z, view_au: float, rot_deg: float, width: int, height: int):
    rot = np.deg2rad(rot_deg)
    xr = x * np.cos(rot) - y * np.sin(rot)
    yr = x * np.sin(rot) + y * np.cos(rot)

    # Tiny faux perspective using z; keeps inclination visible but readable.
    yr = yr + 0.18 * z

    scale = min(width, height) * 0.46 / view_au
    px = width / 2 + xr * scale
    py = height * 0.52 + yr * scale
    return px, py

def draw_ring(draw: ImageDraw.ImageDraw, radius_au: float, view_au: float, rot_deg: float, width: int, height: int, fill, line_width=2, steps=360):
    th = np.linspace(0, 2*np.pi, steps)
    x = radius_au * np.cos(th)
    y = radius_au * np.sin(th)
    z = np.zeros_like(x)
    px, py = project_xy(x, y, z, view_au, rot_deg, width, height)
    pts = list(zip(px, py))
    draw.line(pts + [pts[0]], fill=fill, width=line_width)

def draw_kuiper_band(canvas: Image.Image, view_au: float, rot_deg: float):
    # Draw filled annulus by drawing many transparent rings.
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for r in np.linspace(CONFIG["kuiper_inner_au"], CONFIG["kuiper_outer_au"], 18):
        alpha = int(8 + 12 * (1 - abs(r - 40) / 20))
        draw_ring(draw, r, view_au, rot_deg, canvas.size[0], canvas.size[1], fill=(90, 120, 255, alpha), line_width=3)
    canvas.alpha_composite(overlay)

def draw_orbit_trails(canvas: Image.Image, view_au: float, rot_deg: float, t: float):
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for item in orbit_traces:
        tr = item["trace"]
        # Skip huge traces outside view for readability.
        if np.nanmin(np.hypot(tr[:,0], tr[:,1])) > view_au * 1.4:
            continue
        group = item["group"]
        base = GROUP_COLORS.get(group, (190, 210, 255, 130))
        alpha = 70 if item["highlight"] else 36
        width = 2 if item["highlight"] else 1
        px, py = project_xy(tr[:,0], tr[:,1], tr[:,2], view_au, rot_deg, canvas.size[0], canvas.size[1])
        pts = [(float(x), float(y)) for x, y in zip(px, py)]
        if len(pts) > 2:
            draw.line(pts, fill=(base[0], base[1], base[2], alpha), width=width)
    canvas.alpha_composite(overlay)

def draw_objects(canvas: Image.Image, view_au: float, rot_deg: float, dt_years: float, t: float):
    x, y, z = elements_to_xyz(
        arr["a"], arr["e"], arr["i"], arr["om"], arr["w"], arr["ma"], arr["per_y"],
        dt_years=dt_years
    )
    px, py = project_xy(x, y, z, view_au, rot_deg, canvas.size[0], canvas.size[1])

    order = np.argsort(z)  # back to front
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for idx in order:
        if px[idx] < -30 or px[idx] > canvas.size[0] + 30 or py[idx] < -30 or py[idx] > canvas.size[1] + 30:
            continue
        group = arr["group"][idx]
        color = GROUP_COLORS.get(group, (190, 210, 255, 160))
        H = arr["H"][idx]
        if np.isfinite(H):
            size = np.clip(5.6 - 0.20 * H, 1.4, 7.5)
        else:
            size = 2.2
        if isinstance(arr["highlight"][idx], str) and arr["highlight"][idx]:
            size = max(size, 7.0)
            glow = size * 3
            draw.ellipse((px[idx]-glow, py[idx]-glow, px[idx]+glow, py[idx]+glow), fill=(color[0], color[1], color[2], 32))
            draw.ellipse((px[idx]-size, py[idx]-size, px[idx]+size, py[idx]+size), fill=(255, 245, 180, 245))
        else:
            draw.ellipse((px[idx]-size, py[idx]-size, px[idx]+size, py[idx]+size), fill=color)

    canvas.alpha_composite(overlay)

    # Labels for highlights, drawn after points.
    label_overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    for idx in order:
        h = arr["highlight"][idx]
        if not (isinstance(h, str) and h):
            continue
        if px[idx] < 30 or px[idx] > canvas.size[0] - 30 or py[idx] < 260 or py[idx] > canvas.size[1] - 260:
            continue
        alpha = int(230 * smoothstep((t - 31.0) / 7.0))
        if alpha <= 5:
            continue
        draw_text(label_overlay, h, (int(px[idx] + 12), int(py[idx] - 8)), size=24, fill=(255, 245, 210, alpha), bold=True, stroke=2)
    canvas.alpha_composite(label_overlay)

def draw_sun_and_guides(canvas: Image.Image, view_au: float, rot_deg: float):
    draw_kuiper_band(canvas, view_au, rot_deg)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Guide rings
    draw_ring(draw, CONFIG["neptune_a_au"], view_au, rot_deg, *canvas.size, fill=(80, 190, 255, 110), line_width=3)
    draw_ring(draw, CONFIG["kuiper_inner_au"], view_au, rot_deg, *canvas.size, fill=(130, 160, 255, 80), line_width=2)
    draw_ring(draw, CONFIG["kuiper_outer_au"], view_au, rot_deg, *canvas.size, fill=(130, 160, 255, 80), line_width=2)
    draw_ring(draw, CONFIG["pluto_resonance_3_2_au"], view_au, rot_deg, *canvas.size, fill=(255, 210, 120, 65), line_width=1)
    draw_ring(draw, CONFIG["resonance_2_1_au"], view_au, rot_deg, *canvas.size, fill=(255, 210, 120, 50), line_width=1)

    sx, sy = project_xy(np.array([0.0]), np.array([0.0]), np.array([0.0]), view_au, rot_deg, canvas.size[0], canvas.size[1])
    sx, sy = float(sx[0]), float(sy[0])
    for r, a in [(42, 28), (26, 50), (13, 140)]:
        draw.ellipse((sx-r, sy-r, sx+r, sy+r), fill=(255, 210, 95, a))
    draw.ellipse((sx-5, sy-5, sx+5, sy+5), fill=(255, 245, 180, 245))

    canvas.alpha_composite(overlay)

def add_text_layers(canvas: Image.Image, t: float, shot: Dict):
    # Title
    title_alpha = int(255 * smoothstep((t - 0.3) / 1.2) * (1 - smoothstep((t - 5.5) / 1.2)))
    if title_alpha > 5:
        draw_multiline(
            canvas,
            [CONFIG["title_text"]],
            (62, 118),
            size=58,
            fill=(255, 255, 255, title_alpha),
            bold=True,
        )
        draw_multiline(
            canvas,
            [CONFIG["subtitle_text"]],
            (64, 204),
            size=26,
            fill=(218, 232, 255, min(225, title_alpha)),
        )

    # Live caption box
    cap = caption_at(t)
    if cap:
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        y0 = canvas.size[1] - 252
        draw.rounded_rectangle((46, y0, canvas.size[0] - 46, y0 + 125), radius=26, fill=(0, 0, 0, 132))
        canvas.alpha_composite(overlay)
        draw_multiline(canvas, [cap], (70, y0 + 28), size=32, fill=(255, 255, 255, 245), bold=False)

    # Small scientific note during the ending.
    note_alpha = int(220 * smoothstep((t - 49) / 3.0))
    if note_alpha > 5:
        draw_multiline(
            canvas,
            [CONFIG["credit_text"], CONFIG["scientific_note"]],
            (68, canvas.size[1] - 118),
            size=22,
            fill=(225, 230, 245, note_alpha),
        )

    # Corner shot label
    if t > 6:
        draw_text(canvas, shot["caption"], (62, 78), size=24, fill=(200, 220, 255, 190), bold=False, stroke=1)

def render_frame(t: float) -> np.ndarray:
    shot, view_au, rot_deg, dt_years = shot_state(t)

    canvas = render_starfield(OUT_SIZE[0], OUT_SIZE[1], STARS, t)

    draw_sun_and_guides(canvas, view_au, rot_deg)
    draw_orbit_trails(canvas, view_au, rot_deg, t)
    draw_objects(canvas, view_au, rot_deg, dt_years, t)
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
# 
# Run this before full rendering. It gives you five representative frames from the short.

# %%
preview_times = [1.0, 10.0, 24.0, 42.0, CONFIG["duration_s"] - 1.0]
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
plt.show()

print("Preview images written to:", PREVIEW_DIR.resolve())

# %% [markdown]
# ## Subtitle sidecar
# 
# The notebook writes a simple `.srt` file so the video can be uploaded with captions. You can also burn subtitles into the video later if you set `CONFIG["burn_subtitles"] = True`.

# %%
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
# This is the longest cell. On a normal laptop, reduce `fps` or `duration_s` during testing, then return to 24 fps and ~58 seconds for final export.
# 
# Recommended testing settings:
# 
# ```python
# CONFIG["fps"] = 12
# CONFIG["duration_s"] = 12
# ```

# %%
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
# Set `CONFIG["audio_path"]` to a local `.mp3` or `.wav` file before running this cell.
# 
# Examples:
# 
# ```python
# CONFIG["audio_path"] = "audio/space_ambient.mp3"
# CONFIG["burn_subtitles"] = True
# ```
# 
# Use music only if you have permission to use it.

# %%
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
# # Final explanation: what the video image represents
# 
# Use this section in your YouTube description, portfolio write-up, or voiceover.
# 
# ## What is real in the video?
# 
# The real data are the **orbital elements of Trans-Neptunian Objects** downloaded from NASA/JPL's Small-Body Database Query API.
# 
# Each dot represents one catalog object. Each orbit trail is computed from real elements such as semimajor axis, eccentricity, inclination, longitude of ascending node, argument of perihelion, mean anomaly, and orbital period.
# 
# ## What is cinematic?
# 
# The starfield, glow, particles, camera zoom, rotation, fade, text overlays, and vignette are editorial effects. They are added to make the short understandable and visually engaging. They do not represent new measured astronomical structure.
# 
# ## Why use orbits instead of telescope images?
# 
# For galaxies or nebulae, a telescope image is a natural visual base. For the Kuiper Belt, the scientifically meaningful story is the **population structure**: thousands of icy bodies orbiting beyond Neptune. Most of them are not resolved as detailed public images, so orbital data is a stronger real-data foundation.
# 
# ## What each visual element means
# 
# - **Central glow**: the Sun, not to scale.
# - **Blue ring near 30 AU**: Neptune's approximate orbital distance.
# - **Soft 30–50 AU guide band**: the approximate Kuiper Belt region used for visual context.
# - **Thin orbit trails**: simplified two-body orbit paths from JPL catalog elements.
# - **Moving dots**: estimated object positions from orbital elements.
# - **Larger glowing labels**: named objects such as Pluto, Arrokoth, Eris, Makemake, Haumea, Quaoar, Sedna, Orcus, or Gonggong when they are present in the downloaded result.
# - **Colours**: visual categories for communication, not true surface colours.
# - **Motion**: accelerated cinematic time. It helps viewers feel the orbital structure but should not be interpreted as exact real-time motion.
# 
# ## Suggested voiceover
# 
# > Beyond Neptune, the Solar System becomes a frozen archive.  
# > This is the Kuiper Belt — not drawn from imagination, but built from real orbital data.  
# > Every dot is a known Trans-Neptunian Object.  
# > Every curve is an orbit shaped by distance, eccentricity, and inclination.  
# > Pluto is here. Arrokoth is here — the far world visited by New Horizons.  
# > The glow is cinematic. The structure is data.  
# > This is the outer Solar System, mapped from real measurements.
# 
# ## Suggested YouTube Shorts caption
# 
# Real Kuiper Belt data turned into a cinematic orbit map.  
# Data source: NASA/JPL Small-Body Database Query API.  
# Visualisation: Python, orbital elements, cinematic motion design.  
# Note: dots and orbits are data-driven; colours and glow are presentation layers.
# 
# #Astronomy #Python #DataVisualization #NASA #JPL #KuiperBelt #YouTubeShorts

