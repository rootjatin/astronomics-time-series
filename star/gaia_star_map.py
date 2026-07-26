from __future__ import annotations

"""
Result : https://youtube.com/shorts/OhXaEZEM30E
Gaia Mapped Billions of Stars — cinematic YouTube Short renderer

Creates a vertical 1080x1920 astronomy short about ESA's Gaia mission and the
three measurements that turn a flat sky into a moving 3D Milky Way:
position, parallax, and proper motion. The renderer tries to download a small,
random Gaia DR3 sample from the official ESA Gaia Archive. If the archive or
optional astronomy packages are unavailable, it uses a deterministic synthetic
Milky-Way fixture that is clearly labelled on-screen and in the metadata.

Current mission context used by the story:
- Gaia scanned the sky from 27 July 2014 through 15 January 2025.
- It accumulated more than three trillion observations of about two billion
  stars and other objects.
- Gaia DR3 contains astrometry and photometry for about 1.8 billion sources.
- Gaia measures tiny annual parallax shifts and proper motions, while colour and
  brightness reveal stellar populations in a colour-magnitude diagram.

Official sources:
- https://www.esa.int/Science_Exploration/Space_Science/Gaia
- https://www.cosmos.esa.int/web/gaia/dr3
- https://gea.esac.esa.int/archive/

Recommended install:
    pip install numpy pandas matplotlib pillow imageio imageio-ffmpeg tqdm \
        astropy astroquery


"""

import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
except Exception:
    u = None
    SkyCoord = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("GAIA_SHORT_QUICK", "0") == "1"
FORCE_OFFLINE = os.environ.get("GAIA_SHORT_OFFLINE", "0") == "1"

OUTPUT_ROOT = Path("gaia_mapped_billions_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "gaia_mapped_billions_of_stars",
    "title": "GAIA MAPPED BILLIONS OF STARS",
    "subtitle": "Turning a flat sky into a moving 3D Milky Way",
    "archive_table": "gaiadr3.gaia_source",
    "live_sample_rows": 6500 if QUICK_MODE else 24000,
    "render_sample_rows": 3500 if QUICK_MODE else 9000,
    "background_stars": 260 if QUICK_MODE else 520,
    "contrast": 1.08,
    "saturation": 1.06,
    "vignette": 0.24,
}

OUT_W = CONFIG["video_width"]
OUT_H = CONFIG["video_height"]
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0

MISSION_FACTS = {
    "science_operations_start": "2014-07-27",
    "sky_scanning_end": "2025-01-15",
    "mission_observations": 3_000_000_000_000,
    "mission_objects": 2_000_000_000,
    "dr3_sources": 1_800_000_000,
    "milky_way_fraction_percent": 1.0,
}

FULL_CAPTIONS = [
    (0.5, 7.5, "Gaia repeatedly scanned the whole sky, measuring about two billion stars and other objects."),
    (7.6, 18.0, "Gaia Data Release 3 contains roughly 1.8 billion sources—but this animation needs only a small random sample."),
    (18.1, 29.5, "A star's annual parallax shift reveals distance: nearby stars appear to move more against the distant background."),
    (29.6, 40.5, "Proper motion tracks each star drifting across the sky, turning a catalogue of positions into a movie of the Galaxy."),
    (40.6, 50.5, "Colour plus intrinsic brightness builds a stellar family portrait: the main sequence, red giants, and white dwarfs."),
    (50.6, 57.3, "Gaia did not photograph every Milky Way star. It measured enough of them to reconstruct our Galaxy in motion."),
]
if QUICK_MODE:
    _factor = CONFIG["duration_s"] / 58.0
    CAPTIONS = [(a * _factor, b * _factor, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 8.0 if not QUICK_MODE else 2.0},
    {"name": "all_sky", "start": 8.0 if not QUICK_MODE else 2.0, "end": 21.0 if not QUICK_MODE else 4.6},
    {"name": "parallax", "start": 21.0 if not QUICK_MODE else 4.6, "end": 33.0 if not QUICK_MODE else 7.0},
    {"name": "motion_hr", "start": 33.0 if not QUICK_MODE else 7.0, "end": 49.5 if not QUICK_MODE else 10.1},
    {"name": "outro", "start": 49.5 if not QUICK_MODE else 10.1, "end": CONFIG["duration_s"]},
]


# -----------------------------------------------------------------------------
# General helpers
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


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int = 28,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    stroke: int = 2,
    anchor: str = "la",
):
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


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int = 28,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    line_spacing: int = 6,
):
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
    rr = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * rr**1.75, 0.0, 1.0).astype(np.float32)


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


def write_srt(captions, path: Path):
    lines: List[str] = []
    for index, (start, end, text) in enumerate(captions, 1):
        lines.extend([str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def format_compact_number(value: float) -> str:
    value = float(value)
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.1f}T"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value))


VIGNETTE = make_vignette(OUT_W, OUT_H, CONFIG["vignette"])


# -----------------------------------------------------------------------------
# Gaia data loading
# -----------------------------------------------------------------------------

def fetch_gaia_dr3_sample() -> Tuple[pd.DataFrame, str]:
    if FORCE_OFFLINE:
        raise RuntimeError("GAIA_SHORT_OFFLINE=1")
    if Gaia is None:
        raise RuntimeError("astroquery.gaia is not installed")

    n = int(CONFIG["live_sample_rows"])
    query = f"""
        SELECT TOP {n}
            source_id, ra, dec, parallax, pmra, pmdec,
            phot_g_mean_mag, bp_rp, random_index
        FROM {CONFIG['archive_table']}
        WHERE random_index IS NOT NULL
          AND random_index < {max(n * 3, n)}
          AND phot_g_mean_mag IS NOT NULL
          AND bp_rp IS NOT NULL
          AND parallax IS NOT NULL
          AND pmra IS NOT NULL
          AND pmdec IS NOT NULL
        ORDER BY random_index
    """
    job = Gaia.launch_job_async(query, dump_to_file=False, verbose=False)
    table = job.get_results()
    if len(table) < 500:
        raise RuntimeError(f"Gaia query returned only {len(table)} rows")
    df = table.to_pandas()
    for column in ["ra", "dec", "parallax", "pmra", "pmdec", "phot_g_mean_mag", "bp_rp"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["ra", "dec", "phot_g_mean_mag", "bp_rp"]).reset_index(drop=True)
    if len(df) < 500:
        raise RuntimeError("Too few usable rows after cleaning Gaia sample")
    return enrich_catalog(df), "live_gaia_dr3"


def equatorial_to_galactic_approx(ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised IAU J2000 equatorial-to-galactic rotation without Astropy."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    rotation = np.array([
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [0.4941094279, -0.4448296300, 0.7469822445],
        [-0.8676661490, -0.1980763734, 0.4559837762],
    ])
    xyz = rotation @ np.vstack([x, y, z])
    l = np.rad2deg(np.arctan2(xyz[1], xyz[0])) % 360.0
    b = np.rad2deg(np.arcsin(np.clip(xyz[2], -1.0, 1.0)))
    return l, b


def enrich_catalog(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if SkyCoord is not None and u is not None:
        try:
            coords = SkyCoord(ra=out["ra"].to_numpy() * u.deg, dec=out["dec"].to_numpy() * u.deg, frame="icrs")
            out["gal_l"] = coords.galactic.l.deg
            out["gal_b"] = coords.galactic.b.deg
        except Exception:
            out["gal_l"], out["gal_b"] = equatorial_to_galactic_approx(out["ra"].to_numpy(), out["dec"].to_numpy())
    else:
        out["gal_l"], out["gal_b"] = equatorial_to_galactic_approx(out["ra"].to_numpy(), out["dec"].to_numpy())

    p = pd.to_numeric(out["parallax"], errors="coerce")
    positive = p > 0
    out["distance_pc"] = np.where(positive, 1000.0 / p, np.nan)
    out["abs_g_mag"] = np.where(
        positive,
        out["phot_g_mean_mag"] + 5.0 * np.log10(p.clip(lower=1e-4)) - 10.0,
        np.nan,
    )
    out["proper_motion_total"] = np.sqrt(out["pmra"] ** 2 + out["pmdec"] ** 2)
    return out.replace([np.inf, -np.inf], np.nan)


def fallback_catalog() -> Tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(42025)
    n = int(CONFIG["live_sample_rows"])

    disk = rng.random(n) < 0.84
    gal_l = rng.uniform(0.0, 360.0, n)
    gal_b = np.empty(n)
    gal_b[disk] = np.clip(rng.normal(0.0, 10.5, disk.sum()), -75, 75)
    sin_b = rng.uniform(-1.0, 1.0, (~disk).sum())
    gal_b[~disk] = np.rad2deg(np.arcsin(sin_b))

    population = rng.choice(["main", "giant", "white_dwarf"], n, p=[0.82, 0.13, 0.05])
    bp_rp = np.empty(n)
    abs_g = np.empty(n)

    main = population == "main"
    bp_rp[main] = np.clip(rng.normal(1.15, 0.72, main.sum()), -0.35, 3.6)
    abs_g[main] = 1.6 + 3.8 * bp_rp[main] + 0.75 * bp_rp[main] ** 2 + rng.normal(0, 0.65, main.sum())

    giant = population == "giant"
    bp_rp[giant] = np.clip(rng.normal(1.45, 0.42, giant.sum()), 0.45, 3.2)
    abs_g[giant] = rng.normal(0.35, 1.05, giant.sum()) + 0.45 * (bp_rp[giant] - 1.4)

    wd = population == "white_dwarf"
    bp_rp[wd] = np.clip(rng.normal(0.15, 0.38, wd.sum()), -0.55, 1.3)
    abs_g[wd] = rng.normal(12.1, 1.25, wd.sum()) + 2.2 * bp_rp[wd]

    log_distance = rng.uniform(np.log10(35), np.log10(6500), n)
    distance_pc = 10 ** log_distance
    parallax = 1000.0 / distance_pc + rng.normal(0, 0.03, n)
    extinction = np.clip(0.00018 * distance_pc * np.exp(-np.abs(gal_b) / 12.0), 0, 2.2)
    apparent_g = abs_g + 5.0 * np.log10(distance_pc / 10.0) + extinction

    keep = apparent_g < 20.6
    gal_l = gal_l[keep]
    gal_b = gal_b[keep]
    bp_rp = bp_rp[keep]
    abs_g = abs_g[keep]
    distance_pc = distance_pc[keep]
    parallax = parallax[keep]
    apparent_g = apparent_g[keep]
    population = population[keep]
    n2 = len(gal_l)

    velocity_scale = rng.lognormal(mean=np.log(26.0), sigma=0.55, size=n2)
    angle = rng.uniform(0, 2 * math.pi, n2)
    proper_motion = 1000.0 * velocity_scale / (4.74047 * distance_pc)
    pmra = proper_motion * np.cos(angle) + rng.normal(0, 0.15, n2)
    pmdec = proper_motion * np.sin(angle) + rng.normal(0, 0.15, n2)

    # The renderer primarily uses galactic coordinates. Equatorial placeholders
    # are kept for schema compatibility with a live catalogue.
    ra = (gal_l + 266.4) % 360.0
    dec = np.clip(gal_b - 28.9 * np.cos(np.deg2rad(gal_l)), -90, 90)
    df = pd.DataFrame({
        "source_id": np.arange(10_000_000_000_000_000, 10_000_000_000_000_000 + n2, dtype=np.int64),
        "ra": ra,
        "dec": dec,
        "parallax": parallax,
        "pmra": pmra,
        "pmdec": pmdec,
        "phot_g_mean_mag": apparent_g,
        "bp_rp": bp_rp,
        "random_index": np.arange(n2),
        "gal_l": gal_l,
        "gal_b": gal_b,
        "distance_pc": distance_pc,
        "abs_g_mag": abs_g,
        "proper_motion_total": np.sqrt(pmra * pmra + pmdec * pmdec),
        "fixture_population": population,
    })
    return df.reset_index(drop=True), "offline_milky_way_fixture"


def load_catalog() -> Tuple[pd.DataFrame, str, Optional[str]]:
    try:
        df, source = fetch_gaia_dr3_sample()
        return df, source, None
    except Exception as exc:
        df, source = fallback_catalog()
        return df, source, str(exc)


def summarize_catalog(df: pd.DataFrame, source: str) -> Dict:
    positive = df[df["parallax"] > 0].copy()
    hr = positive.dropna(subset=["bp_rp", "abs_g_mag"])
    pm = df["proper_motion_total"].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "source": source,
        "sample_rows": int(len(df)),
        "positive_parallax_rows": int(len(positive)),
        "hr_diagram_rows": int(len(hr)),
        "median_g_mag": float(df["phot_g_mean_mag"].median()),
        "median_bp_rp": float(df["bp_rp"].median()),
        "median_parallax_mas": float(positive["parallax"].median()) if len(positive) else None,
        "median_proper_motion_mas_per_year": float(pm.median()) if len(pm) else None,
        "mission_facts": MISSION_FACTS,
        "archive_table": CONFIG["archive_table"],
    }


def save_data_products(df: pd.DataFrame, summary: Dict, error_note: Optional[str]):
    csv_path = DATA_ROOT / "gaia_dr3_render_sample.csv"
    json_path = DATA_ROOT / "gaia_dr3_render_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "summary": summary,
        "live_query_error": error_note,
        "offline_warning": "The deterministic fixture is for layout/testing only and is not Gaia observational data.",
        "official_sources": [
            "https://www.esa.int/Science_Exploration/Space_Science/Gaia",
            "https://www.cosmos.esa.int/web/gaia/dr3",
            "https://gea.esac.esa.int/archive/",
        ],
    }, indent=2), encoding="utf-8")
    return csv_path, json_path


def create_scientific_plots(df: pd.DataFrame):
    sample = df.sample(min(len(df), 14000), random_state=2) if len(df) > 14000 else df

    fig, ax = plt.subplots(figsize=(10, 4.8))
    l_wrap = ((sample["gal_l"].to_numpy() + 180.0) % 360.0) - 180.0
    ax.scatter(l_wrap, sample["gal_b"], s=1, alpha=0.25)
    ax.set_title("Gaia render sample in Galactic coordinates")
    ax.set_xlabel("Galactic longitude (deg)")
    ax.set_ylabel("Galactic latitude (deg)")
    ax.set_xlim(180, -180)
    ax.set_ylim(-90, 90)
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "gaia_sample_all_sky.png", dpi=170)
    plt.close(fig)

    hr = sample[(sample["parallax"] > 0) & sample["abs_g_mag"].notna()].copy()
    if len(hr):
        fig, ax = plt.subplots(figsize=(6.8, 6.8))
        ax.scatter(hr["bp_rp"], hr["abs_g_mag"], s=2, alpha=0.28)
        ax.set_title("Gaia sample colour–magnitude diagram")
        ax.set_xlabel("BP − RP colour")
        ax.set_ylabel("Absolute G magnitude")
        ax.invert_yaxis()
        ax.set_xlim(-0.7, 4.1)
        ax.set_ylim(16, -6)
        plt.tight_layout()
        plt.savefig(PREVIEW_DIR / "gaia_sample_hr_diagram.png", dpi=170)
        plt.close(fig)

    pm = sample["proper_motion_total"].dropna().clip(upper=300)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(pm, bins=60)
    ax.set_title("Proper-motion distribution in render sample")
    ax.set_xlabel("Total proper motion (mas/year, clipped)")
    ax.set_ylabel("Sample count")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "gaia_sample_proper_motion.png", dpi=170)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class GaiaMapScene:
    def __init__(self, df: pd.DataFrame, summary: Dict, source: str):
        self.df = df.copy().reset_index(drop=True)
        self.summary = summary
        self.source = source
        self.rng = np.random.default_rng(8821)
        self.background_points = self._make_background_points(CONFIG["background_stars"])
        self.render_df = self._select_render_rows()
        self.map_points = self._prepare_map_points()
        self.hr_points = self._prepare_hr_points()
        self.motion_points = self._prepare_motion_points()

    def _make_background_points(self, count: int):
        rng = np.random.default_rng(22)
        return [{
            "x": float(rng.uniform(0, OUT_W)),
            "y": float(rng.uniform(0, OUT_H)),
            "r": float(rng.uniform(0.35 * SCALE, 2.0 * SCALE)),
            "alpha": int(rng.integers(18, 110)),
            "phase": float(rng.uniform(0, 2 * math.pi)),
        } for _ in range(count)]

    def _select_render_rows(self) -> pd.DataFrame:
        n = min(int(CONFIG["render_sample_rows"]), len(self.df))
        if len(self.df) <= n:
            return self.df.copy()
        weights = np.clip(21.0 - self.df["phot_g_mean_mag"].to_numpy(float), 0.25, 12.0)
        weights = weights / weights.sum()
        indices = self.rng.choice(len(self.df), size=n, replace=False, p=weights)
        return self.df.iloc[indices].reset_index(drop=True)

    @staticmethod
    def star_rgb(bp_rp: float, brightness: float = 1.0) -> Tuple[int, int, int]:
        c = clamp((bp_rp + 0.45) / 4.0)
        # Blue-white to amber-red approximation for storytelling, not calibrated colour.
        if c < 0.45:
            q = c / 0.45
            rgb = (lerp(115, 242, q), lerp(175, 245, q), 255)
        else:
            q = (c - 0.45) / 0.55
            rgb = (255, lerp(238, 118, q), lerp(230, 75, q))
        return tuple(int(clamp(v * brightness / 255.0) * 255) for v in rgb)

    def _prepare_map_points(self):
        points = []
        for _, row in self.render_df.iterrows():
            lon = ((float(row["gal_l"]) + 180.0) % 360.0) - 180.0
            lat = float(row["gal_b"])
            x = lon / 180.0
            y = -lat / 90.0
            mag = float(row["phot_g_mean_mag"])
            radius = np.clip((20.5 - mag) * 0.34 * SCALE, 0.45 * SCALE, 2.7 * SCALE)
            alpha = int(np.clip(65 + (20.5 - mag) * 17, 55, 225))
            points.append((x, y, radius, alpha, float(row["bp_rp"])))
        points.sort(key=lambda item: item[2])
        return points

    def _prepare_hr_points(self):
        hr = self.render_df[(self.render_df["parallax"] > 0) & self.render_df["abs_g_mag"].notna()].copy()
        hr = hr[(hr["bp_rp"] > -0.8) & (hr["bp_rp"] < 4.3) & (hr["abs_g_mag"] > -7) & (hr["abs_g_mag"] < 17)]
        if len(hr) > 5000:
            hr = hr.sample(5000, random_state=44)
        points = []
        for _, row in hr.iterrows():
            x = (float(row["bp_rp"]) + 0.7) / 4.8
            y = (float(row["abs_g_mag"]) + 6.0) / 22.0
            points.append((x, y, float(row["bp_rp"]), float(row["phot_g_mean_mag"])))
        return points

    def _prepare_motion_points(self):
        pm = self.render_df.dropna(subset=["pmra", "pmdec", "proper_motion_total"]).copy()
        pm = pm.sort_values("proper_motion_total", ascending=False).head(42 if not QUICK_MODE else 24)
        rng = np.random.default_rng(91)
        points = []
        for _, row in pm.iterrows():
            x = float(rng.uniform(0.12, 0.88))
            y = float(rng.uniform(0.27, 0.70))
            scale = 0.0018 / max(float(row["proper_motion_total"]), 1e-6)
            # Normalize directions, then give high-PM stars somewhat longer arrows.
            dx = float(row["pmra"]) * scale
            dy = -float(row["pmdec"]) * scale
            norm = math.hypot(dx, dy) or 1.0
            length = np.clip(0.038 + 0.035 * math.log10(1 + float(row["proper_motion_total"])), 0.045, 0.12)
            dx = dx / norm * length
            dy = dy / norm * length
            points.append((x, y, dx, dy, float(row["bp_rp"]), float(row["proper_motion_total"])))
        return points

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, (2, 6, 14, 255))
        draw = ImageDraw.Draw(image)
        for star in self.background_points:
            pulse = 0.72 + 0.28 * math.sin(1.5 * t + star["phase"])
            alpha = int(star["alpha"] * pulse)
            x, y, r = star["x"], star["y"], star["r"]
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(215, 230, 255, alpha))

        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for cx, cy, colour in [
            (OUT_W * 0.22, OUT_H * 0.28, (38, 34, 125)),
            (OUT_W * 0.74, OUT_H * 0.42, (15, 86, 125)),
            (OUT_W * 0.52, OUT_H * 0.78, (84, 32, 72)),
        ]:
            for radius, alpha in [(430 * SCALE, 15), (280 * SCALE, 22), (175 * SCALE, 32)]:
                hd.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=colour + (alpha,))
        haze = haze.filter(ImageFilter.GaussianBlur(max(15, int(62 * SCALE))))
        image.alpha_composite(haze)
        return image

    def draw_scanning_satellite(self, image: Image.Image, t: float):
        cx, cy = OUT_W * 0.5, OUT_H * 0.36
        orbit_r = 180 * SCALE
        scan_phase = t * 0.9
        satellite_x = cx + math.cos(scan_phase) * orbit_r
        satellite_y = cy + math.sin(scan_phase) * orbit_r * 0.34

        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.ellipse((cx - orbit_r, cy - orbit_r * 0.34, cx + orbit_r, cy + orbit_r * 0.34), outline=(100, 210, 245, 70), width=max(1, int(2 * SCALE)))

        # Stylised Gaia sunshield and optical bench.
        shield_r = 35 * SCALE
        draw.polygon([
            (satellite_x, satellite_y - shield_r),
            (satellite_x + shield_r * 0.95, satellite_y),
            (satellite_x, satellite_y + shield_r),
            (satellite_x - shield_r * 0.95, satellite_y),
        ], fill=(230, 210, 145, 230), outline=(255, 245, 200, 230))
        draw.rectangle((satellite_x - 11 * SCALE, satellite_y - 17 * SCALE, satellite_x + 11 * SCALE, satellite_y + 17 * SCALE), fill=(70, 105, 145, 245))
        beam_angle = scan_phase * 2.1
        beam_len = 240 * SCALE
        for offset in (-0.10, 0.10):
            ang = beam_angle + offset
            ex = satellite_x + math.cos(ang) * beam_len
            ey = satellite_y + math.sin(ang) * beam_len
            draw.line((satellite_x, satellite_y, ex, ey), fill=(90, 225, 255, 60), width=max(1, int(3 * SCALE)))
        image.alpha_composite(overlay)

    def draw_intro(self, image: Image.Image, t: float):
        self.draw_scanning_satellite(image, t)
        local_end = 8.0 if not QUICK_MODE else 2.0
        progress = smoothstep(t / max(local_end * 0.82, 0.01))
        objects = int(MISSION_FACTS["mission_objects"] * progress)
        observations = int(MISSION_FACTS["mission_observations"] * progress)
        draw_text(image, format_compact_number(objects), (OUT_W // 2, int(OUT_H * 0.61)), size=int(84 * SCALE), fill=(245, 249, 255, 245), bold=True, anchor="ma", stroke=max(1, int(3 * SCALE)))
        draw_text(image, "STARS + OTHER OBJECTS", (OUT_W // 2, int(OUT_H * 0.68)), size=int(25 * SCALE), fill=(105, 232, 250, 235), bold=True, anchor="ma", stroke=1)
        draw_text(image, f"{format_compact_number(observations)} OBSERVATIONS", (OUT_W // 2, int(OUT_H * 0.735)), size=int(23 * SCALE), fill=(255, 194, 100, 235), bold=True, anchor="ma", stroke=1)

    def draw_all_sky_map(self, image: Image.Image, t: float):
        left, top = int(OUT_W * 0.06), int(OUT_H * 0.23)
        right, bottom = int(OUT_W * 0.94), int(OUT_H * 0.72)
        cx, cy = (left + right) / 2, (top + bottom) / 2
        rx, ry = (right - left) / 2, (bottom - top) / 2
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.ellipse((left, top, right, bottom), fill=(3, 9, 21, 185), outline=(95, 210, 240, 105), width=max(1, int(2 * SCALE)))

        for lon in (-120, -60, 0, 60, 120):
            x = cx + (lon / 180.0) * rx
            height = ry * math.sqrt(max(0.0, 1.0 - ((x - cx) / rx) ** 2))
            draw.line((x, cy - height, x, cy + height), fill=(100, 175, 210, 28), width=1)
        for lat in (-60, -30, 0, 30, 60):
            y = cy - (lat / 90.0) * ry
            width = rx * math.sqrt(max(0.0, 1.0 - ((y - cy) / ry) ** 2))
            draw.line((cx - width, y, cx + width, y), fill=(100, 175, 210, 32 if lat else 52), width=1)

        shot_start = 8.0 if not QUICK_MODE else 2.0
        reveal_duration = 8.5 if not QUICK_MODE else 1.7
        reveal = smoothstep((t - shot_start) / reveal_duration)
        visible = int(len(self.map_points) * reveal)
        for x_norm, y_norm, radius, alpha, colour_index in self.map_points[:visible]:
            x = cx + x_norm * rx
            y = cy + y_norm * ry
            ellipse_term = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
            if ellipse_term > 1.0:
                continue
            colour = self.star_rgb(colour_index)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour + (alpha,))

        scan_x = left + (right - left) * ((t * 0.22) % 1.0)
        scan_h = ry * math.sqrt(max(0.0, 1.0 - ((scan_x - cx) / rx) ** 2))
        draw.rectangle((scan_x - 2 * SCALE, cy - scan_h, scan_x + 2 * SCALE, cy + scan_h), fill=(100, 230, 255, 60))
        image.alpha_composite(overlay)

        draw_text(image, "THE MILKY WAY // ALL-SKY SAMPLE", (left + int(18 * SCALE), top - int(50 * SCALE)), size=int(25 * SCALE), fill=(110, 232, 248, 235), bold=True, stroke=1)
        draw_text(image, f"RENDERING {len(self.render_df):,} OF ~1.8 BILLION DR3 SOURCES", (OUT_W // 2, bottom + int(38 * SCALE)), size=int(18 * SCALE), fill=(235, 242, 250, 220), bold=True, anchor="ma", stroke=1)

    def draw_parallax(self, image: Image.Image, t: float):
        cx, cy = OUT_W * 0.50, OUT_H * 0.48
        orbit_rx, orbit_ry = 205 * SCALE, 74 * SCALE
        shot_start = 21.0 if not QUICK_MODE else 4.6
        phase = 2 * math.pi * clamp((t - shot_start) / (12.0 if not QUICK_MODE else 2.4))
        earth_x = cx + math.cos(phase) * orbit_rx
        earth_y = cy + math.sin(phase) * orbit_ry

        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.ellipse((cx - orbit_rx, cy - orbit_ry, cx + orbit_rx, cy + orbit_ry), outline=(90, 195, 230, 90), width=max(1, int(2 * SCALE)))

        sun_r = 35 * SCALE
        for scale_factor, alpha in [(2.1, 20), (1.55, 45), (1.15, 100)]:
            r = sun_r * scale_factor
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 172, 60, alpha))
        draw.ellipse((cx - sun_r, cy - sun_r, cx + sun_r, cy + sun_r), fill=(255, 205, 92, 255))

        earth_r = 14 * SCALE
        draw.ellipse((earth_x - earth_r, earth_y - earth_r, earth_x + earth_r, earth_y + earth_r), fill=(70, 155, 235, 255), outline=(175, 225, 255, 230))

        star_x = OUT_W * 0.50 + math.cos(phase) * 38 * SCALE
        star_y = OUT_H * 0.26
        star_r = 12 * SCALE
        draw.line((earth_x, earth_y, star_x, star_y), fill=(110, 230, 250, 90), width=max(1, int(2 * SCALE)))
        draw.ellipse((star_x - star_r * 2.5, star_y - star_r * 2.5, star_x + star_r * 2.5, star_y + star_r * 2.5), fill=(105, 220, 255, 30))
        draw.ellipse((star_x - star_r, star_y - star_r, star_x + star_r, star_y + star_r), fill=(210, 242, 255, 255))

        # Distant reference stars remain fixed while the nearby star appears to shift.
        rng = np.random.default_rng(8)
        for _ in range(34 if not QUICK_MODE else 20):
            x = rng.uniform(OUT_W * 0.12, OUT_W * 0.88)
            y = rng.uniform(OUT_H * 0.18, OUT_H * 0.38)
            if abs(x - star_x) < 40 * SCALE:
                continue
            r = rng.uniform(1.0, 2.4) * SCALE
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(205, 220, 240, int(rng.integers(70, 155))))

        baseline_y = cy + orbit_ry + 43 * SCALE
        draw.line((cx - orbit_rx, baseline_y, cx + orbit_rx, baseline_y), fill=(255, 194, 100, 170), width=max(1, int(3 * SCALE)))
        draw.polygon([(cx - orbit_rx, baseline_y), (cx - orbit_rx + 14 * SCALE, baseline_y - 7 * SCALE), (cx - orbit_rx + 14 * SCALE, baseline_y + 7 * SCALE)], fill=(255, 194, 100, 190))
        draw.polygon([(cx + orbit_rx, baseline_y), (cx + orbit_rx - 14 * SCALE, baseline_y - 7 * SCALE), (cx + orbit_rx - 14 * SCALE, baseline_y + 7 * SCALE)], fill=(255, 194, 100, 190))
        image.alpha_composite(overlay)

        draw_text(image, "PARALLAX = DEPTH", (OUT_W // 2, int(OUT_H * 0.67)), size=int(32 * SCALE), fill=(110, 232, 248, 240), bold=True, anchor="ma", stroke=1)
        draw_text(image, "A larger apparent shift means a nearer star", (OUT_W // 2, int(OUT_H * 0.72)), size=int(20 * SCALE), fill=(238, 244, 250, 225), anchor="ma", stroke=1)
        draw_text(image, "EARTH'S ORBIT PROVIDES THE BASELINE", (OUT_W // 2, int(baseline_y + 33 * SCALE)), size=int(17 * SCALE), fill=(255, 195, 105, 225), bold=True, anchor="ma", stroke=1)

    def draw_motion_panel(self, image: Image.Image, t: float):
        left, top = int(OUT_W * 0.06), int(OUT_H * 0.22)
        right, bottom = int(OUT_W * 0.94), int(OUT_H * 0.48)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle((left, top, right, bottom), radius=int(24 * SCALE), fill=(3, 8, 18, 175), outline=(90, 195, 225, 65), width=1)
        shot_start = 33.0 if not QUICK_MODE else 7.0
        reveal = smoothstep((t - shot_start) / (5.5 if not QUICK_MODE else 1.0))
        count = int(len(self.motion_points) * reveal)
        for x_norm, y_norm, dx_norm, dy_norm, colour_index, pm in self.motion_points[:count]:
            x = left + x_norm * (right - left)
            y = y_norm * OUT_H
            dx = dx_norm * OUT_W
            dy = dy_norm * OUT_H
            colour = self.star_rgb(colour_index)
            r = np.clip(1.3 + math.log10(1 + pm) * 0.6, 1.4, 4.0) * SCALE
            draw.ellipse((x - r, y - r, x + r, y + r), fill=colour + (240,))
            ex, ey = x + dx * reveal, y + dy * reveal
            draw.line((x, y, ex, ey), fill=colour + (155,), width=max(1, int(2 * SCALE)))
            norm = math.hypot(dx, dy) or 1.0
            ux, uy = dx / norm, dy / norm
            px, py = -uy, ux
            arrow = 7 * SCALE
            draw.polygon([
                (ex, ey),
                (ex - ux * arrow + px * arrow * 0.45, ey - uy * arrow + py * arrow * 0.45),
                (ex - ux * arrow - px * arrow * 0.45, ey - uy * arrow - py * arrow * 0.45),
            ], fill=colour + (175,))
        image.alpha_composite(overlay)
        draw_text(image, "PROPER MOTION // STARS DRIFT", (left + int(18 * SCALE), top + int(16 * SCALE)), size=int(22 * SCALE), fill=(110, 232, 248, 235), bold=True, stroke=1)
        draw_text(image, "arrows exaggerated for visibility", (right - int(18 * SCALE), top + int(18 * SCALE)), size=int(14 * SCALE), fill=(170, 205, 222, 195), anchor="ra", stroke=1)

    def draw_hr_diagram(self, image: Image.Image, t: float):
        left, top = int(OUT_W * 0.10), int(OUT_H * 0.54)
        right, bottom = int(OUT_W * 0.90), int(OUT_H * 0.81)
        width, height = right - left, bottom - top
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle((left, top, right, bottom), radius=int(24 * SCALE), fill=(3, 8, 18, 180), outline=(90, 195, 225, 65), width=1)

        # axes and population guides
        x_axis = left + int(42 * SCALE)
        y_axis = bottom - int(28 * SCALE)
        draw.line((x_axis, top + int(24 * SCALE), x_axis, y_axis), fill=(195, 220, 235, 85), width=1)
        draw.line((x_axis, y_axis, right - int(20 * SCALE), y_axis), fill=(195, 220, 235, 85), width=1)

        shot_start = 38.0 if not QUICK_MODE else 8.0
        reveal = smoothstep((t - shot_start) / (7.0 if not QUICK_MODE else 1.35))
        visible = int(len(self.hr_points) * reveal)
        for x_norm, y_norm, colour_index, apparent_mag in self.hr_points[:visible]:
            x = x_axis + clamp(x_norm) * (right - x_axis - int(25 * SCALE))
            y = top + int(26 * SCALE) + clamp(y_norm) * (y_axis - top - int(32 * SCALE))
            colour = self.star_rgb(colour_index)
            radius = np.clip((20.5 - apparent_mag) * 0.16 + 0.55, 0.45, 1.7) * SCALE
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour + (105,))

        image.alpha_composite(overlay)
        draw_text(image, "STELLAR FAMILY PORTRAIT", (left + int(18 * SCALE), top + int(12 * SCALE)), size=int(21 * SCALE), fill=(255, 195, 105, 235), bold=True, stroke=1)
        draw_text(image, "BLUE  ←  BP−RP COLOUR  →  RED", (OUT_W // 2, bottom + int(25 * SCALE)), size=int(15 * SCALE), fill=(210, 225, 238, 205), anchor="ma", stroke=1)
        draw_text(image, "BRIGHT", (left - int(8 * SCALE), top + int(30 * SCALE)), size=int(13 * SCALE), fill=(210, 225, 238, 195), anchor="ra", stroke=1)
        draw_text(image, "FAINT", (left - int(8 * SCALE), bottom - int(28 * SCALE)), size=int(13 * SCALE), fill=(210, 225, 238, 195), anchor="ra", stroke=1)
        draw_text(image, "MAIN SEQUENCE", (int(OUT_W * 0.58), int(OUT_H * 0.67)), size=int(15 * SCALE), fill=(230, 238, 245, 190), bold=True, stroke=1)
        draw_text(image, "GIANTS", (int(OUT_W * 0.69), int(OUT_H * 0.58)), size=int(14 * SCALE), fill=(255, 175, 105, 200), bold=True, stroke=1)
        draw_text(image, "WHITE DWARFS", (int(OUT_W * 0.24), int(OUT_H * 0.77)), size=int(14 * SCALE), fill=(145, 215, 255, 205), bold=True, stroke=1)

    def draw_outro(self, image: Image.Image, t: float):
        self.draw_all_sky_map(image, 21.0 if not QUICK_MODE else 4.6)
        panel_top = int(OUT_H * 0.61)
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle((int(OUT_W * 0.08), panel_top, int(OUT_W * 0.92), int(OUT_H * 0.79)), radius=int(28 * SCALE), fill=(2, 7, 17, 205), outline=(100, 210, 240, 80), width=1)
        image.alpha_composite(overlay)
        facts = [
            ("POSITION", "where each source is"),
            ("PARALLAX", "how far away it is"),
            ("PROPER MOTION", "how it moves across the sky"),
        ]
        y = panel_top + int(25 * SCALE)
        for title, detail in facts:
            draw_text(image, title, (int(OUT_W * 0.12), y), size=int(21 * SCALE), fill=(110, 232, 248, 235), bold=True, stroke=1)
            draw_text(image, detail, (int(OUT_W * 0.88), y), size=int(18 * SCALE), fill=(238, 244, 250, 220), anchor="ra", stroke=1)
            y += int(52 * SCALE)
        draw_text(image, "A 3D MAP THAT ALSO MOVES", (OUT_W // 2, int(OUT_H * 0.845)), size=int(30 * SCALE), fill=(255, 195, 105, 240), bold=True, anchor="ma", stroke=1)

    def draw_source_hud(self, image: Image.Image):
        live = self.source == "live_gaia_dr3"
        label = "SOURCE // ESA GAIA DR3 LIVE SAMPLE" if live else "PREVIEW SOURCE // SYNTHETIC MILKY-WAY FIXTURE"
        colour = (110, 232, 248, 235) if live else (255, 193, 100, 235)
        draw_text(image, label, (OUT_W - int(44 * SCALE), int(68 * SCALE)), size=int(17 * SCALE), fill=colour, bold=True, anchor="ra", stroke=1)
        draw_text(image, f"SAMPLE ROWS // {len(self.df):,}", (OUT_W - int(44 * SCALE), int(98 * SCALE)), size=int(15 * SCALE), fill=(165, 205, 224, 205), anchor="ra", stroke=1)

    def draw_titles(self, image: Image.Image, t: float, shot_name: str):
        alpha = int(255 * smoothstep((t - 0.2) / 0.8) * (1.0 - smoothstep((t - (6.7 if not QUICK_MODE else 1.65)) / 0.7)))
        if alpha > 4:
            draw_text(image, "GAIA MAPPED", (int(54 * SCALE), int(86 * SCALE)), size=int(47 * SCALE), fill=(245, 249, 255, alpha), bold=True, stroke=max(1, int(2 * SCALE)))
            draw_text(image, "BILLIONS OF STARS", (int(54 * SCALE), int(142 * SCALE)), size=int(47 * SCALE), fill=(245, 249, 255, alpha), bold=True, stroke=max(1, int(2 * SCALE)))
            draw_text(image, CONFIG["subtitle"], (int(57 * SCALE), int(203 * SCALE)), size=int(21 * SCALE), fill=(110, 232, 248, min(alpha, 230)), bold=True, stroke=1)

        labels = {
            "intro": "THE GALAXY-SCALE SURVEY",
            "all_sky": "SKY POSITION // THE FIRST LAYER",
            "parallax": "PARALLAX // TURNING ANGLES INTO DISTANCE",
            "motion_hr": "MOTION + COLOUR // THE GALAXY COMES ALIVE",
            "outro": "A MOVING 3D MAP OF THE MILKY WAY",
        }
        if t > (5.1 if not QUICK_MODE else 1.35):
            draw_text(image, labels[shot_name], (int(54 * SCALE), int(58 * SCALE)), size=int(18 * SCALE), fill=(150, 210, 230, 210), bold=True, stroke=1)

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        y0 = OUT_H - int(244 * SCALE)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((int(42 * SCALE), y0, OUT_W - int(42 * SCALE), y0 + int(126 * SCALE)), radius=int(24 * SCALE), fill=(2, 6, 14, 178), outline=(75, 185, 220, 65), width=1)
        image.alpha_composite(panel)
        draw_wrapped_text(image, text, (int(67 * SCALE), y0 + int(28 * SCALE)), OUT_W - int(134 * SCALE), size=int(29 * SCALE), fill=(245, 249, 253, 245), line_spacing=int(6 * SCALE))

    def draw_hud_noise(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        offset = int((t * 37) % max(3, int(7 * SCALE)))
        step = max(3, int(7 * SCALE))
        for y in range(offset, OUT_H, step):
            draw.line((0, y, OUT_W, y), fill=(120, 205, 245, 10), width=1)
        scan_y = int((t * 158 * SCALE) % (OUT_H + int(220 * SCALE))) - int(110 * SCALE)
        draw.rectangle((0, scan_y, OUT_W, scan_y + int(48 * SCALE)), fill=(80, 210, 240, 8))
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        image = self.background(t)
        name = shot["name"]

        if name == "intro":
            self.draw_intro(image, t)
        elif name == "all_sky":
            self.draw_all_sky_map(image, t)
        elif name == "parallax":
            self.draw_parallax(image, t)
        elif name == "motion_hr":
            self.draw_motion_panel(image, t)
            self.draw_hr_diagram(image, t)
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
# Rendering
# -----------------------------------------------------------------------------

def render_video(scene: GaiaMapScene):
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    raw_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"

    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    times = np.arange(frame_count) / CONFIG["fps"]
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(raw_path, fps=CONFIG["fps"], codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None) as writer:
        for t in tqdm(times, desc="Rendering Gaia short"):
            writer.append_data(scene.render_frame(float(t)))
    shutil.copyfile(raw_path, final_path)
    print("Final video:", final_path.resolve())
    return final_path, srt_path


def main():
    print("Loading Gaia catalogue sample ...")
    df, source, error_note = load_catalog()
    summary = summarize_catalog(df, source)
    csv_path, json_path = save_data_products(df, summary, error_note)
    create_scientific_plots(df)

    print("Source:", source)
    if error_note:
        print("Live Gaia query note:", error_note)
    print("Data:", csv_path.resolve())
    print("Summary:", json_path.resolve())

    scene = GaiaMapScene(df, summary, source)
    preview_times = [
        1.0,
        min(11.0, CONFIG["duration_s"] * 0.22),
        min(25.0, CONFIG["duration_s"] * 0.43),
        min(37.0, CONFIG["duration_s"] * 0.64),
        min(46.0, CONFIG["duration_s"] * 0.80),
        CONFIG["duration_s"] - 1.0,
    ]
    for pt in tqdm(preview_times, desc="Preview frames"):
        Image.fromarray(scene.render_frame(float(pt))).save(PREVIEW_DIR / f"preview_{int(pt):02d}s.png")

    render_video(scene)
    print("Output directory:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
