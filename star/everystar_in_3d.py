from __future__ import annotations

"""
The Closest Stars to Earth in 3D — cinematic YouTube Short renderer

Creates a vertical 1080x1920 astronomy short that turns the nearby-star sky into
an actual three-dimensional solar-neighbourhood map. The highlighted systems use
a compact curated reference table of well-known neighbours. When online, the
script also queries the official ESA Gaia DR3 archive for a quality-filtered
sample inside roughly 12 parsecs and uses those real astrometric sources as the
background point cloud.

The geometry is scientific rather than decorative:
- distance is derived from parallax for live Gaia sources;
- right ascension, declination and distance are converted to Cartesian XYZ;
- the camera rotates around a Sun-centred coordinate system;
- proper-motion arrows show that the neighbourhood is changing with time.

The built-in named-star table is always preserved because a few extremely bright,
close or multiple systems can be incomplete or awkward in a single Gaia source
query. If Gaia is unavailable, a deterministic background fixture is used and is
clearly labelled on-screen and in the metadata.

Official/reference sources:
- ESA Gaia Archive: https://gea.esac.esa.int/archive/
- ESA Gaia programmatic access:
  https://www.cosmos.esa.int/web/gaia-users/archive/programmatic-access
- RECONS 100 nearest systems: https://www.recons.org/TOP100.posted.htm

Install:
    pip install numpy pandas matplotlib pillow imageio imageio-ffmpeg tqdm \
        astropy astroquery

Full render:
    python the_closest_stars_to_earth_in_3d_short.py

Quick preview:
    NEARBY_STARS_SHORT_QUICK=1 python the_closest_stars_to_earth_in_3d_short.py

Forced offline preview:
    NEARBY_STARS_SHORT_QUICK=1 NEARBY_STARS_SHORT_OFFLINE=1 \
        python the_closest_stars_to_earth_in_3d_short.py
"""

import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    from astroquery.gaia import Gaia
except Exception:
    Gaia = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("NEARBY_STARS_SHORT_QUICK", "0") == "1"
FORCE_OFFLINE = os.environ.get("NEARBY_STARS_SHORT_OFFLINE", "0") == "1"

OUTPUT_ROOT = Path("closest_stars_to_earth_3d_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "the_closest_stars_to_earth_in_3d",
    "title": "THE CLOSEST STARS TO EARTH IN 3D",
    "subtitle": "A Sun-centred map of the stellar neighbourhood",
    "gaia_table": "gaiadr3.gaia_source",
    "gaia_radius_pc": 12.0,
    "gaia_max_rows": 900 if QUICK_MODE else 2600,
    "render_background_rows": 340 if QUICK_MODE else 1100,
    "background_stars": 220 if QUICK_MODE else 430,
    "contrast": 1.09,
    "saturation": 1.06,
    "vignette": 0.25,
}

OUT_W = CONFIG["video_width"]
OUT_H = CONFIG["video_height"]
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0

FULL_CAPTIONS = [
    (0.5, 7.2, "The night sky looks flat, but every nearby star sits at a different distance and direction."),
    (7.3, 17.8, "Proxima Centauri is the nearest known star beyond the Sun—about 4.25 light-years away."),
    (17.9, 29.0, "Right ascension and declination give direction. Parallax gives distance. Together they become XYZ coordinates."),
    (29.1, 40.0, "Most of the Sun's closest neighbours are faint red dwarfs, not the bright stars that dominate our constellations."),
    (40.1, 50.0, "Proper motion means this map is not frozen. Nearby stars sweep across space on their own Galactic orbits."),
    (50.1, 57.3, "Zoom out only twelve light-years, and the Sun becomes one point inside a sparse, moving three-dimensional neighbourhood."),
]
if QUICK_MODE:
    factor = CONFIG["duration_s"] / 58.0
    CAPTIONS = [(a * factor, b * factor, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 8.0 if not QUICK_MODE else 2.0},
    {"name": "nearest", "start": 8.0 if not QUICK_MODE else 2.0, "end": 22.0 if not QUICK_MODE else 4.7},
    {"name": "map3d", "start": 22.0 if not QUICK_MODE else 4.7, "end": 38.0 if not QUICK_MODE else 7.9},
    {"name": "motion", "start": 38.0 if not QUICK_MODE else 7.9, "end": 49.5 if not QUICK_MODE else 10.2},
    {"name": "outro", "start": 49.5 if not QUICK_MODE else 10.2, "end": CONFIG["duration_s"]},
]

# Curated named systems. Distances are representative modern values rounded for
# visual storytelling. RA/Dec are J2000-style degrees and determine the actual
# 3D direction used by the renderer.
NAMED_SYSTEMS = [
    {"name": "Sun", "distance_ly": 0.0, "ra": 0.0, "dec": 0.0, "spectral": "G2V", "kind": "Sun", "pmra": 0.0, "pmdec": 0.0},
    {"name": "Proxima Centauri", "distance_ly": 4.2465, "ra": 217.4292, "dec": -62.6795, "spectral": "M5.5V", "kind": "Red dwarf", "pmra": -3775.6, "pmdec": 765.5},
    {"name": "Alpha Centauri A/B", "distance_ly": 4.36, "ra": 219.9021, "dec": -60.8340, "spectral": "G2V + K1V", "kind": "Binary", "pmra": -3679.3, "pmdec": 473.7},
    {"name": "Barnard's Star", "distance_ly": 5.963, "ra": 269.4521, "dec": 4.6934, "spectral": "M4V", "kind": "Red dwarf", "pmra": -802.3, "pmdec": 10362.5},
    {"name": "Luhman 16 A/B", "distance_ly": 6.503, "ra": 162.3125, "dec": -53.3183, "spectral": "L7.5 + T0.5", "kind": "Brown-dwarf binary", "pmra": -2762.0, "pmdec": 354.0},
    {"name": "WISE 0855−0714", "distance_ly": 7.43, "ra": 133.7948, "dec": -7.2441, "spectral": "Y4", "kind": "Sub-brown dwarf", "pmra": -8123.0, "pmdec": 673.0},
    {"name": "Wolf 359", "distance_ly": 7.856, "ra": 164.1208, "dec": 7.0147, "spectral": "M6V", "kind": "Red dwarf", "pmra": -3842.0, "pmdec": -2725.0},
    {"name": "Lalande 21185", "distance_ly": 8.304, "ra": 165.8342, "dec": 35.9699, "spectral": "M2V", "kind": "Red dwarf", "pmra": -580.3, "pmdec": -4765.9},
    {"name": "Sirius A/B", "distance_ly": 8.60, "ra": 101.2872, "dec": -16.7161, "spectral": "A1V + DA2", "kind": "Star + white dwarf", "pmra": -546.0, "pmdec": -1223.1},
    {"name": "Luyten 726-8 A/B", "distance_ly": 8.73, "ra": 24.7550, "dec": -17.9500, "spectral": "M5.5V + M6V", "kind": "Flare-star binary", "pmra": 3296.0, "pmdec": 563.0},
    {"name": "Ross 154", "distance_ly": 9.69, "ra": 270.1613, "dec": -23.8996, "spectral": "M3.5V", "kind": "Red dwarf", "pmra": 639.0, "pmdec": -1915.0},
    {"name": "Ross 248", "distance_ly": 10.31, "ra": 355.4800, "dec": 44.1700, "spectral": "M6V", "kind": "Red dwarf", "pmra": 111.0, "pmdec": -1591.0},
    {"name": "Epsilon Eridani", "distance_ly": 10.48, "ra": 53.2327, "dec": -9.4583, "spectral": "K2V", "kind": "Orange dwarf", "pmra": -975.2, "pmdec": 19.5},
    {"name": "Lacaille 9352", "distance_ly": 10.72, "ra": 346.4668, "dec": -35.8531, "spectral": "M0.5V", "kind": "Red dwarf", "pmra": 6768.2, "pmdec": 1327.5},
    {"name": "Ross 128", "distance_ly": 11.01, "ra": 176.9375, "dec": 0.7991, "spectral": "M4V", "kind": "Red dwarf", "pmra": 605.6, "pmdec": -1219.3},
    {"name": "EZ Aquarii A/B/C", "distance_ly": 11.11, "ra": 339.7208, "dec": -15.3003, "spectral": "M5V system", "kind": "Triple red dwarf", "pmra": 2314.0, "pmdec": 2295.0},
    {"name": "61 Cygni A/B", "distance_ly": 11.40, "ra": 316.7275, "dec": 38.7494, "spectral": "K5V + K7V", "kind": "Binary", "pmra": 4164.0, "pmdec": 3260.0},
    {"name": "Procyon A/B", "distance_ly": 11.46, "ra": 114.8255, "dec": 5.2250, "spectral": "F5IV + DQZ", "kind": "Star + white dwarf", "pmra": -716.6, "pmdec": -1034.6},
    {"name": "Groombridge 34 A/B", "distance_ly": 11.62, "ra": 5.8722, "dec": 44.0230, "spectral": "M1.5V + M3.5V", "kind": "Red-dwarf binary", "pmra": 2890.0, "pmdec": 411.0},
    {"name": "Epsilon Indi", "distance_ly": 11.87, "ra": 330.8402, "dec": -56.7860, "spectral": "K5V", "kind": "Orange dwarf system", "pmra": 3968.0, "pmdec": -2536.0},
    {"name": "Tau Ceti", "distance_ly": 11.91, "ra": 26.0170, "dec": -15.9375, "spectral": "G8V", "kind": "Sun-like dwarf", "pmra": -1721.0, "pmdec": 855.5},
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(value: float) -> float:
    value = clamp(value)
    return value * value * (3.0 - 2.0 * value)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_shot(t: float) -> Dict:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(6, int(size)))
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(image: Image.Image, text: str, xy: Tuple[int, int], size: int = 28,
              fill=(255, 255, 255, 255), bold: bool = False, stroke: int = 2,
              anchor: str = "la"):
    draw = ImageDraw.Draw(image)
    draw.text(
        xy,
        text,
        font=get_font(size, bold),
        fill=fill,
        anchor=anchor,
        stroke_width=max(0, int(stroke)),
        stroke_fill=(0, 0, 0, min(225, fill[3] if len(fill) > 3 else 225)),
    )


def draw_wrapped_text(image: Image.Image, text: str, xy: Tuple[int, int], max_width: int,
                      size: int = 28, fill=(255, 255, 255, 245), bold: bool = False,
                      line_spacing: int = 6):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bb = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)
        if bb[2] - bb[0] <= max_width:
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
        bb = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += bb[3] - bb[1] + line_spacing


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius ** 1.8, 0.0, 1.0).astype(np.float32)


def apply_grade(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array)
    image = ImageEnhance.Contrast(image).enhance(CONFIG["contrast"])
    image = ImageEnhance.Color(image).enhance(CONFIG["saturation"])
    return np.array(image)


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(captions: Iterable[Tuple[float, float, str]], path: Path):
    lines: List[str] = []
    for index, (start, end, text) in enumerate(captions, 1):
        lines.extend([str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


VIGNETTE = make_vignette(OUT_W, OUT_H, CONFIG["vignette"])


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def equatorial_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray, distance_ly: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    distance = np.asarray(distance_ly, dtype=float)
    x = distance * np.cos(dec) * np.cos(ra)
    y = distance * np.cos(dec) * np.sin(ra)
    z = distance * np.sin(dec)
    return x, y, z


def make_named_catalog() -> pd.DataFrame:
    df = pd.DataFrame(NAMED_SYSTEMS)
    x, y, z = equatorial_xyz(df["ra"], df["dec"], df["distance_ly"])
    df["x_ly"], df["y_ly"], df["z_ly"] = x, y, z
    df["proper_motion_total"] = np.sqrt(df["pmra"] ** 2 + df["pmdec"] ** 2)
    df["source"] = "curated_named_reference"
    return df


def fetch_gaia_nearby() -> Tuple[pd.DataFrame, str]:
    if FORCE_OFFLINE:
        raise RuntimeError("NEARBY_STARS_SHORT_OFFLINE=1")
    if Gaia is None:
        raise RuntimeError("astroquery.gaia is not installed")

    min_parallax = 1000.0 / float(CONFIG["gaia_radius_pc"])
    query = f"""
        SELECT TOP {int(CONFIG['gaia_max_rows'])}
            source_id, ra, dec, parallax, parallax_error, pmra, pmdec,
            phot_g_mean_mag, bp_rp, ruwe
        FROM {CONFIG['gaia_table']}
        WHERE parallax >= {min_parallax:.5f}
          AND parallax_over_error >= 10
          AND ruwe < 1.4
          AND phot_g_mean_mag IS NOT NULL
        ORDER BY parallax DESC
    """
    job = Gaia.launch_job_async(query, dump_to_file=False, verbose=False)
    table = job.get_results()
    if len(table) < 30:
        raise RuntimeError(f"Gaia query returned only {len(table)} usable rows")
    df = table.to_pandas()
    numeric = ["ra", "dec", "parallax", "parallax_error", "pmra", "pmdec", "phot_g_mean_mag", "bp_rp", "ruwe"]
    for column in numeric:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["ra", "dec", "parallax", "phot_g_mean_mag"]).copy()
    df = df[df["parallax"] > 0].reset_index(drop=True)
    df["distance_pc"] = 1000.0 / df["parallax"]
    df["distance_ly"] = df["distance_pc"] * 3.261563777
    x, y, z = equatorial_xyz(df["ra"], df["dec"], df["distance_ly"])
    df["x_ly"], df["y_ly"], df["z_ly"] = x, y, z
    df["proper_motion_total"] = np.sqrt(df["pmra"].fillna(0) ** 2 + df["pmdec"].fillna(0) ** 2)
    df["source"] = "live_gaia_dr3"
    return df, "live_gaia_dr3"


def fallback_background() -> Tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(120326)
    n = int(CONFIG["gaia_max_rows"])
    # Volume-uniform points inside 12 pc / 39 ly, then retain the inner visual
    # neighbourhood. These are not observational data.
    radius_ly = 12.0 * 3.261563777
    r = radius_ly * np.cbrt(rng.random(n))
    cos_theta = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2 * math.pi, n)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta
    distance_ly = np.sqrt(x * x + y * y + z * z)
    # Nearby populations are dominated numerically by cool dwarfs.
    bp_rp = np.clip(rng.normal(1.75, 0.78, n), -0.25, 4.2)
    apparent_g = np.clip(4.5 + 1.4 * distance_ly + 2.2 * bp_rp + rng.normal(0, 1.2, n), 4, 21)
    pm_angle = rng.uniform(0, 2 * math.pi, n)
    pm_total = np.clip(rng.lognormal(np.log(350), 0.8, n) * (8.0 / np.maximum(distance_ly, 2.0)), 5, 8000)
    df = pd.DataFrame({
        "source_id": np.arange(92_000_000_000_000_000, 92_000_000_000_000_000 + n, dtype=np.int64),
        "ra": np.rad2deg(np.arctan2(y, x)) % 360.0,
        "dec": np.rad2deg(np.arcsin(np.clip(z / np.maximum(distance_ly, 1e-9), -1, 1))),
        "parallax": 1000.0 / np.maximum(distance_ly / 3.261563777, 1e-6),
        "pmra": pm_total * np.cos(pm_angle),
        "pmdec": pm_total * np.sin(pm_angle),
        "phot_g_mean_mag": apparent_g,
        "bp_rp": bp_rp,
        "distance_pc": distance_ly / 3.261563777,
        "distance_ly": distance_ly,
        "x_ly": x,
        "y_ly": y,
        "z_ly": z,
        "proper_motion_total": pm_total,
        "source": "offline_spatial_fixture",
    })
    return df, "offline_spatial_fixture"


def load_catalogs() -> Tuple[pd.DataFrame, pd.DataFrame, str, Optional[str]]:
    named = make_named_catalog()
    try:
        background, source = fetch_gaia_nearby()
        return named, background, source, None
    except Exception as exc:
        background, source = fallback_background()
        return named, background, source, str(exc)


def summarize_catalog(named: pd.DataFrame, background: pd.DataFrame, source: str) -> Dict:
    systems = named[named["name"] != "Sun"].sort_values("distance_ly")
    red_like = systems[systems["kind"].str.contains("Red dwarf|red dwarf|dwarf binary|Flare", regex=True)]
    return {
        "source": source,
        "named_systems": int(len(systems)),
        "background_rows": int(len(background)),
        "nearest_named_star": systems.iloc[0][["name", "distance_ly", "spectral"]].to_dict(),
        "farthest_named_system": systems.iloc[-1][["name", "distance_ly", "spectral"]].to_dict(),
        "red_dwarf_like_named_count": int(len(red_like)),
        "map_radius_light_years": 12.0,
        "gaia_query_radius_parsecs": float(CONFIG["gaia_radius_pc"]),
    }


def save_data_products(named: pd.DataFrame, background: pd.DataFrame, summary: Dict, error_note: Optional[str]):
    named_path = DATA_ROOT / "named_nearby_star_systems.csv"
    background_path = DATA_ROOT / "nearby_gaia_or_fixture_sample.csv"
    summary_path = DATA_ROOT / "closest_stars_3d_summary.json"
    named.to_csv(named_path, index=False)
    background.to_csv(background_path, index=False)
    summary_path.write_text(json.dumps({
        "summary": summary,
        "live_query_error": error_note,
        "offline_warning": "The offline background is a deterministic spatial fixture, not observational data.",
        "named_table_note": "Named positions/distances are a compact rounded reference table for visual storytelling.",
        "official_sources": [
            "https://gea.esac.esa.int/archive/",
            "https://www.cosmos.esa.int/web/gaia-users/archive/programmatic-access",
            "https://www.recons.org/TOP100.posted.htm",
        ],
    }, indent=2), encoding="utf-8")
    return named_path, background_path, summary_path


def create_scientific_plots(named: pd.DataFrame, background: pd.DataFrame):
    sample = background.sample(min(len(background), 1200), random_state=7) if len(background) > 1200 else background
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sample["x_ly"], sample["y_ly"], sample["z_ly"], s=4, alpha=0.22)
    focus = named[named["name"] != "Sun"]
    ax.scatter(focus["x_ly"], focus["y_ly"], focus["z_ly"], s=28)
    ax.scatter([0], [0], [0], s=70, marker="*")
    ax.set_title("Nearby-star render sample in Sun-centred XYZ coordinates")
    ax.set_xlabel("X (light-years)")
    ax.set_ylabel("Y (light-years)")
    ax.set_zlabel("Z (light-years)")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "nearby_stars_3d_scientific.png", dpi=170)
    plt.close(fig)

    ranks = focus.sort_values("distance_ly").head(12)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(ranks["name"][::-1], ranks["distance_ly"][::-1])
    ax.set_xlabel("Distance (light-years)")
    ax.set_title("Closest highlighted stellar systems")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "nearest_system_distances.png", dpi=170)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Renderer
# -----------------------------------------------------------------------------

class NearbyStarsScene:
    def __init__(self, named: pd.DataFrame, background: pd.DataFrame, summary: Dict, source: str):
        self.named = named.copy().reset_index(drop=True)
        self.background_df = background.copy().reset_index(drop=True)
        self.summary = summary
        self.source = source
        if len(self.background_df) > CONFIG["render_background_rows"]:
            self.render_background_df = self.background_df.sample(CONFIG["render_background_rows"], random_state=44).reset_index(drop=True)
        else:
            self.render_background_df = self.background_df.copy()
        self.named_sorted = self.named[self.named["name"] != "Sun"].sort_values("distance_ly").reset_index(drop=True)
        self.space_dust = self._make_space_dust(CONFIG["background_stars"], 22)

    @staticmethod
    def _make_space_dust(n: int, seed: int):
        rng = np.random.default_rng(seed)
        return [{
            "x": float(rng.uniform(0, OUT_W)),
            "y": float(rng.uniform(0, OUT_H)),
            "r": float(rng.uniform(0.4, 1.9) * max(SCALE, 0.55)),
            "alpha": int(rng.integers(18, 90)),
            "phase": float(rng.uniform(0, 2 * math.pi)),
        } for _ in range(n)]

    @staticmethod
    def star_colour(bp_rp: float) -> Tuple[int, int, int]:
        if not np.isfinite(bp_rp):
            return (205, 225, 245)
        t = clamp((float(bp_rp) + 0.35) / 4.1)
        if t < 0.35:
            q = t / 0.35
            return (int(lerp(125, 245, q)), int(lerp(185, 245, q)), 255)
        q = (t - 0.35) / 0.65
        return (255, int(lerp(245, 115, q)), int(lerp(225, 95, q)))

    @staticmethod
    def spectral_colour(spectral: str) -> Tuple[int, int, int]:
        s = spectral.upper()
        if "Y" in s or "T" in s or "L" in s:
            return (185, 120, 255)
        if "M" in s:
            return (255, 112, 88)
        if "K" in s:
            return (255, 177, 90)
        if "G" in s:
            return (255, 230, 145)
        if "F" in s or "A" in s:
            return (185, 220, 255)
        if "D" in s:
            return (175, 235, 255)
        return (220, 230, 245)

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, (2, 6, 14, 255))
        draw = ImageDraw.Draw(image)
        for item in self.space_dust:
            pulse = 0.72 + 0.28 * math.sin(1.7 * t + item["phase"])
            r = item["r"]
            draw.ellipse((item["x"] - r, item["y"] - r, item["x"] + r, item["y"] + r), fill=(215, 230, 255, int(item["alpha"] * pulse)))
        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for cx, cy, col in [
            (OUT_W * 0.22, OUT_H * 0.27, (43, 31, 120)),
            (OUT_W * 0.76, OUT_H * 0.40, (16, 88, 124)),
            (OUT_W * 0.50, OUT_H * 0.78, (81, 33, 67)),
        ]:
            for radius, alpha in [(420 * SCALE, 15), (280 * SCALE, 23), (175 * SCALE, 32)]:
                hd.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=col + (alpha,))
        haze = haze.filter(ImageFilter.GaussianBlur(max(14, int(62 * SCALE))))
        image.alpha_composite(haze)
        return image

    @staticmethod
    def camera_project(points: np.ndarray, yaw: float, pitch: float, zoom: float,
                       center: Tuple[float, float], perspective: float = 32.0):
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        rot_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
        rot_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=float)
        p = points @ rot_yaw.T @ rot_pitch.T
        depth = p[:, 1]
        denom = np.maximum(perspective + depth, perspective * 0.28)
        factor = perspective / denom
        sx = center[0] + p[:, 0] * zoom * factor
        sy_screen = center[1] - p[:, 2] * zoom * factor
        return sx, sy_screen, depth, factor

    def draw_grid(self, overlay: Image.Image, yaw: float, pitch: float, zoom: float,
                  center: Tuple[float, float], radius_ly: float = 12.0):
        draw = ImageDraw.Draw(overlay)
        for ring in (4, 8, 12):
            theta = np.linspace(0, 2 * math.pi, 100)
            pts = np.column_stack([ring * np.cos(theta), ring * np.sin(theta), np.zeros_like(theta)])
            sx, sy, depth, _ = self.camera_project(pts, yaw, pitch, zoom, center)
            order = np.argsort(depth)
            coords = [(float(sx[i]), float(sy[i])) for i in order]
            draw.line(coords, fill=(100, 195, 225, 36 if ring != 12 else 68), width=max(1, int(SCALE)))
        axes = [
            (np.array([[-radius_ly, 0, 0], [radius_ly, 0, 0]]), (255, 120, 110, 90), "X"),
            (np.array([[0, -radius_ly, 0], [0, radius_ly, 0]]), (110, 230, 160, 90), "Y"),
            (np.array([[0, 0, -radius_ly], [0, 0, radius_ly]]), (110, 190, 255, 90), "Z"),
        ]
        for pts, col, label in axes:
            sx, sy, _, _ = self.camera_project(pts, yaw, pitch, zoom, center)
            draw.line((sx[0], sy[0], sx[1], sy[1]), fill=col, width=max(1, int(2 * SCALE)))
            draw_text(overlay, label, (int(sx[1]), int(sy[1])), size=int(15 * SCALE), fill=col[:3] + (180,), bold=True, stroke=1, anchor="mm")

    def draw_star_map(self, image: Image.Image, t: float, yaw: float, pitch: float, zoom: float,
                      center: Tuple[float, float], reveal: float = 1.0,
                      label_count: int = 8, radius_limit: float = 12.2,
                      show_background: bool = True, show_motion: bool = False):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        self.draw_grid(overlay, yaw, pitch, zoom, center, radius_limit)
        draw = ImageDraw.Draw(overlay)

        if show_background:
            bg = self.render_background_df
            pts = bg[["x_ly", "y_ly", "z_ly"]].to_numpy(float)
            mask = np.linalg.norm(pts, axis=1) <= radius_limit * 3.261563777 if radius_limit <= 4 else np.linalg.norm(pts, axis=1) <= max(radius_limit, 12.2)
            pts = pts[mask]
            bg2 = bg.loc[mask].reset_index(drop=True)
            visible_n = int(len(pts) * clamp(reveal))
            pts = pts[:visible_n]
            bg2 = bg2.iloc[:visible_n]
            if len(pts):
                sx, sy, depth, factor = self.camera_project(pts, yaw, pitch, zoom, center)
                order = np.argsort(depth)[::-1]
                for idx in order:
                    if not (-40 <= sx[idx] <= OUT_W + 40 and -40 <= sy[idx] <= OUT_H + 40):
                        continue
                    mag = float(bg2.iloc[idx].get("phot_g_mean_mag", 15.0))
                    radius = clamp((20.5 - mag) / 5.8, 0.45, 2.4) * max(SCALE, 0.58) * clamp(factor[idx], 0.65, 1.7)
                    colour = self.star_colour(float(bg2.iloc[idx].get("bp_rp", 1.4)))
                    draw.ellipse((sx[idx] - radius, sy[idx] - radius, sx[idx] + radius, sy[idx] + radius), fill=colour + (95,))

        named = self.named[np.linalg.norm(self.named[["x_ly", "y_ly", "z_ly"]].to_numpy(float), axis=1) <= radius_limit + 0.2].copy()
        pts = named[["x_ly", "y_ly", "z_ly"]].to_numpy(float)
        sx, sy, depth, factor = self.camera_project(pts, yaw, pitch, zoom, center)
        order = np.argsort(depth)[::-1]
        for idx in order:
            row = named.iloc[idx]
            colour = self.spectral_colour(str(row["spectral"]))
            base = 9.5 if row["name"] == "Sun" else 5.7
            radius = base * SCALE * clamp(factor[idx], 0.72, 1.55)
            for mult, alpha in ((2.8, 18), (1.8, 36)):
                rr = radius * mult
                draw.ellipse((sx[idx] - rr, sy[idx] - rr, sx[idx] + rr, sy[idx] + rr), fill=colour + (alpha,))
            draw.ellipse((sx[idx] - radius, sy[idx] - radius, sx[idx] + radius, sy[idx] + radius), fill=colour + (245,), outline=(255, 255, 255, 185))

            if show_motion and row["name"] != "Sun":
                pm = float(row["proper_motion_total"])
                angle = math.atan2(float(row["pmdec"]), float(row["pmra"])) + yaw * 0.4
                length = clamp(20 + 10 * math.log10(max(pm, 10)), 30, 86) * SCALE
                ex = sx[idx] + math.cos(angle) * length
                ey = sy[idx] - math.sin(angle) * length
                draw.line((sx[idx], sy[idx], ex, ey), fill=colour + (170,), width=max(1, int(3 * SCALE)))
                ah = 8 * SCALE
                draw.polygon([(ex, ey), (ex - ah * math.cos(angle - 0.5), ey + ah * math.sin(angle - 0.5)), (ex - ah * math.cos(angle + 0.5), ey + ah * math.sin(angle + 0.5))], fill=colour + (190,))

        labels = named[named["name"] != "Sun"].sort_values("distance_ly").head(label_count)
        label_indices = list(labels.index)
        for data_index in label_indices:
            local_idx = list(named.index).index(data_index)
            row = named.loc[data_index]
            x, y = sx[local_idx], sy[local_idx]
            if not (20 < x < OUT_W - 20 and 80 < y < OUT_H - 130):
                continue
            colour = self.spectral_colour(str(row["spectral"]))
            offset_x = 13 * SCALE if x < center[0] else -13 * SCALE
            anchor = "lm" if x < center[0] else "rm"
            draw_text(overlay, str(row["name"]), (int(x + offset_x), int(y - 3 * SCALE)), size=int(16 * SCALE), fill=colour + (235,), bold=True, stroke=1, anchor=anchor)
            draw_text(overlay, f"{row['distance_ly']:.2f} ly", (int(x + offset_x), int(y + 16 * SCALE)), size=int(13 * SCALE), fill=(218, 230, 242, 210), stroke=1, anchor=anchor)

        image.alpha_composite(overlay)

    def draw_intro(self, image: Image.Image, t: float):
        local_end = 8.0 if not QUICK_MODE else 2.0
        progress = smoothstep(t / max(local_end * 0.9, 0.01))
        yaw = -0.8 + t * (0.18 if not QUICK_MODE else 0.75)
        pitch = -0.32
        zoom = lerp(26 * SCALE, 43 * SCALE, progress)
        self.draw_star_map(image, t, yaw, pitch, zoom, (OUT_W * 0.5, OUT_H * 0.43), reveal=progress, label_count=2, radius_limit=6.2, show_background=False)
        draw_text(image, "4.25 LIGHT-YEARS", (OUT_W // 2, int(OUT_H * 0.67)), size=int(54 * SCALE), fill=(245, 249, 255, 245), bold=True, anchor="ma", stroke=max(1, int(3 * SCALE)))
        draw_text(image, "to Proxima Centauri", (OUT_W // 2, int(OUT_H * 0.725)), size=int(24 * SCALE), fill=(110, 230, 248, 235), bold=True, anchor="ma", stroke=1)

    def draw_nearest_cards(self, image: Image.Image, t: float):
        shot_start = 8.0 if not QUICK_MODE else 2.0
        shot_duration = 14.0 if not QUICK_MODE else 2.7
        frac = clamp((t - shot_start) / shot_duration)
        count = min(7, len(self.named_sorted))
        active = min(count - 1, int(frac * count))
        row = self.named_sorted.iloc[active]
        yaw = -0.25 + 0.55 * frac
        pitch = -0.22 + 0.10 * math.sin(frac * math.pi)
        zoom = 31 * SCALE
        self.draw_star_map(image, t, yaw, pitch, zoom, (OUT_W * 0.5, OUT_H * 0.39), reveal=1.0, label_count=7, radius_limit=8.2, show_background=False)

        x0, y0 = int(OUT_W * 0.08), int(OUT_H * 0.62)
        card_w, card_h = int(OUT_W * 0.84), int(OUT_H * 0.15)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        pdw = ImageDraw.Draw(panel)
        pdw.rounded_rectangle((x0, y0, x0 + card_w, y0 + card_h), radius=max(12, int(24 * SCALE)), fill=(2, 7, 16, 185), outline=(95, 200, 230, 80), width=1)
        image.alpha_composite(panel)
        colour = self.spectral_colour(str(row["spectral"]))
        draw_text(image, f"#{active + 1}  {row['name']}", (x0 + int(24 * SCALE), y0 + int(24 * SCALE)), size=int(27 * SCALE), fill=colour + (245,), bold=True, stroke=1)
        draw_text(image, f"{row['distance_ly']:.3f} light-years  //  {row['spectral']}", (x0 + int(24 * SCALE), y0 + int(62 * SCALE)), size=int(20 * SCALE), fill=(242, 247, 252, 235), bold=True, stroke=1)
        draw_text(image, str(row["kind"]), (x0 + int(24 * SCALE), y0 + int(96 * SCALE)), size=int(18 * SCALE), fill=(165, 210, 230, 215), stroke=1)

    def draw_full_map(self, image: Image.Image, t: float):
        shot_start = 22.0 if not QUICK_MODE else 4.7
        duration = 16.0 if not QUICK_MODE else 3.2
        frac = clamp((t - shot_start) / duration)
        yaw = -1.15 + frac * 2.15
        pitch = -0.46 + 0.22 * math.sin(frac * math.pi)
        zoom = lerp(29 * SCALE, 23 * SCALE, frac)
        self.draw_star_map(image, t, yaw, pitch, zoom, (OUT_W * 0.5, OUT_H * 0.46), reveal=smoothstep(frac * 1.7), label_count=10, radius_limit=12.2, show_background=True)
        draw_text(image, "SUN-CENTRED XYZ MAP", (OUT_W // 2, int(OUT_H * 0.73)), size=int(30 * SCALE), fill=(110, 232, 248, 238), bold=True, anchor="ma", stroke=1)
        draw_text(image, "distance is encoded as physical depth", (OUT_W // 2, int(OUT_H * 0.77)), size=int(19 * SCALE), fill=(235, 243, 250, 220), anchor="ma", stroke=1)

    def draw_motion(self, image: Image.Image, t: float):
        shot_start = 38.0 if not QUICK_MODE else 7.9
        duration = 11.5 if not QUICK_MODE else 2.3
        frac = clamp((t - shot_start) / duration)
        yaw = 0.65 + frac * 0.75
        pitch = -0.30
        self.draw_star_map(image, t, yaw, pitch, 26 * SCALE, (OUT_W * 0.5, OUT_H * 0.45), reveal=1.0, label_count=7, radius_limit=12.2, show_background=False, show_motion=True)
        draw_text(image, "PROPER MOTION", (OUT_W // 2, int(OUT_H * 0.71)), size=int(34 * SCALE), fill=(255, 190, 100, 240), bold=True, anchor="ma", stroke=1)
        draw_text(image, "arrows show angular motion on the sky", (OUT_W // 2, int(OUT_H * 0.755)), size=int(19 * SCALE), fill=(237, 244, 250, 220), anchor="ma", stroke=1)

    def draw_outro(self, image: Image.Image, t: float):
        shot_start = 49.5 if not QUICK_MODE else 10.2
        frac = clamp((t - shot_start) / max(CONFIG["duration_s"] - shot_start, 0.1))
        yaw = 1.45 + frac * 0.7
        pitch = -0.38 + 0.12 * math.sin(frac * math.pi)
        zoom = lerp(25 * SCALE, 18 * SCALE, smoothstep(frac))
        self.draw_star_map(image, t, yaw, pitch, zoom, (OUT_W * 0.5, OUT_H * 0.39), reveal=1.0, label_count=5, radius_limit=12.2, show_background=True)

        x0, y0 = int(OUT_W * 0.08), int(OUT_H * 0.64)
        w, h = int(OUT_W * 0.84), int(OUT_H * 0.14)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=max(12, int(24 * SCALE)), fill=(2, 7, 16, 182), outline=(90, 195, 225, 75), width=1)
        image.alpha_composite(panel)
        draw_text(image, "12 LIGHT-YEAR NEIGHBOURHOOD", (OUT_W // 2, y0 + int(30 * SCALE)), size=int(26 * SCALE), fill=(110, 232, 248, 238), bold=True, anchor="ma", stroke=1)
        draw_text(image, f"{self.summary['named_systems']} highlighted systems  //  {self.summary['background_rows']:,} map sources", (OUT_W // 2, y0 + int(70 * SCALE)), size=int(17 * SCALE), fill=(240, 246, 252, 225), bold=True, anchor="ma", stroke=1)
        draw_text(image, "Most nearby stars are faint cool dwarfs", (OUT_W // 2, y0 + int(104 * SCALE)), size=int(18 * SCALE), fill=(255, 183, 105, 228), anchor="ma", stroke=1)

    def draw_source_hud(self, image: Image.Image):
        if self.source == "live_gaia_dr3":
            label = "MAP SOURCE // GAIA DR3 + CURATED LABELS"
            colour = (110, 232, 248, 230)
        else:
            label = "PREVIEW SOURCE // SPATIAL FIXTURE + CURATED LABELS"
            colour = (255, 190, 95, 230)
        draw_text(image, label, (OUT_W - int(46 * SCALE), int(72 * SCALE)), size=int(17 * SCALE), fill=colour, bold=True, anchor="ra", stroke=1)
        draw_text(image, "ORIGIN // THE SUN", (OUT_W - int(46 * SCALE), int(104 * SCALE)), size=int(15 * SCALE), fill=(160, 205, 225, 200), anchor="ra", stroke=1)

    def draw_titles(self, image: Image.Image, t: float, shot_name: str):
        alpha = int(255 * smoothstep((t - 0.2) / 0.8) * (1.0 - smoothstep((t - (6.8 if not QUICK_MODE else 1.7)) / 0.7)))
        if alpha > 4:
            draw_text(image, "THE CLOSEST STARS", (int(56 * SCALE), int(88 * SCALE)), size=int(44 * SCALE), fill=(245, 249, 253, alpha), bold=True)
            draw_text(image, "TO EARTH IN 3D", (int(56 * SCALE), int(138 * SCALE)), size=int(44 * SCALE), fill=(245, 249, 253, alpha), bold=True)
            draw_text(image, CONFIG["subtitle"], (int(58 * SCALE), int(194 * SCALE)), size=int(21 * SCALE), fill=(110, 232, 248, min(alpha, 230)), bold=True)
        labels = {
            "intro": "THE SOLAR NEIGHBOURHOOD // DEPTH REVEALED",
            "nearest": "NEAREST SYSTEMS // DISTANCE RANKING",
            "map3d": "XYZ SPACE // NOT A FLAT STAR CHART",
            "motion": "THE MAP MOVES // STELLAR VELOCITY",
            "outro": "TWELVE LIGHT-YEARS FROM HOME",
        }
        if t > (5.2 if not QUICK_MODE else 1.4):
            draw_text(image, labels[shot_name], (int(56 * SCALE), int(62 * SCALE)), size=int(18 * SCALE), fill=(150, 210, 230, 205), bold=True, stroke=1)

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        y0 = OUT_H - int(244 * SCALE)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((int(44 * SCALE), y0, OUT_W - int(44 * SCALE), y0 + int(124 * SCALE)), radius=max(10, int(24 * SCALE)), fill=(2, 6, 14, 172), outline=(70, 180, 220, 65), width=1)
        image.alpha_composite(panel)
        draw_wrapped_text(image, text, (int(68 * SCALE), y0 + int(28 * SCALE)), OUT_W - int(136 * SCALE), size=int(29 * SCALE), fill=(245, 249, 253, 245))

    def draw_hud_noise(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        offset = int((t * 39) % 7)
        for y in range(offset, OUT_H, 7):
            draw.line((0, y, OUT_W, y), fill=(120, 200, 240, 10), width=1)
        scan_y = int((t * 160) % (OUT_H + 220)) - 110
        draw.rectangle((0, scan_y, OUT_W, scan_y + int(46 * SCALE)), fill=(80, 210, 240, 8))
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        image = self.background(t)
        name = shot["name"]
        if name == "intro":
            self.draw_intro(image, t)
        elif name == "nearest":
            self.draw_nearest_cards(image, t)
        elif name == "map3d":
            self.draw_full_map(image, t)
        elif name == "motion":
            self.draw_motion(image, t)
        else:
            self.draw_outro(image, t)

        self.draw_source_hud(image)
        self.draw_titles(image, t, name)
        self.draw_caption(image, t)
        self.draw_hud_noise(image, t)

        array = np.array(image.convert("RGB"))
        array = apply_grade(array)
        array = np.clip(array.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / 0.9)
        fade_out = 1.0 - smoothstep((t - (CONFIG["duration_s"] - 1.1)) / 1.0)
        return np.clip(array.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Render
# -----------------------------------------------------------------------------

def render_video(scene: NearbyStarsScene):
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    raw_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    times = np.arange(frame_count) / CONFIG["fps"]
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(raw_path, fps=CONFIG["fps"], codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None) as writer:
        for t in tqdm(times, desc="Rendering nearby-stars 3D short"):
            writer.append_data(scene.render_frame(float(t)))
    shutil.copyfile(raw_path, final_path)
    print("Final video:", final_path.resolve())
    return final_path, srt_path


def main():
    print("Loading nearby-star data ...")
    named, background, source, error_note = load_catalogs()
    summary = summarize_catalog(named, background, source)
    paths = save_data_products(named, background, summary, error_note)
    create_scientific_plots(named, background)
    print("Source:", source)
    if error_note:
        print("Live query note:", error_note)
    for path in paths:
        print("Data product:", path.resolve())

    scene = NearbyStarsScene(named, background, summary, source)
    preview_times = [
        1.0,
        min(11.0, CONFIG["duration_s"] * 0.22),
        min(23.0, CONFIG["duration_s"] * 0.42),
        min(34.0, CONFIG["duration_s"] * 0.60),
        min(45.0, CONFIG["duration_s"] * 0.80),
        CONFIG["duration_s"] - 1.0,
    ]
    for pt in tqdm(preview_times, desc="Preview frames"):
        Image.fromarray(scene.render_frame(float(pt))).save(PREVIEW_DIR / f"preview_{int(pt):02d}s.png")
    render_video(scene)
    print("Output directory:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
