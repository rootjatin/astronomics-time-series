from __future__ import annotations

"""
THE OCEAN IS STORING THE HEAT — cinematic YouTube Shorts renderer

Creates a vertical 1080x1920 science Short showing how Earth's ocean stores most
of the excess heat accumulating in the climate system.

Story beats
-----------
1. Roughly 90% of the excess heat from planetary warming is absorbed by the ocean.
2. NOAA's global ocean-heat-content record rises strongly over the modern era.
3. The warming is not only at the surface: heat is stored through the upper 2000 m.
4. Warmer seawater expands, contributing to global sea-level rise.
5. Extra stored heat raises the background for marine heatwaves and ecosystem stress.
6. The atmosphere is where we notice warming quickly; the ocean is the main heat reservoir.

Production pattern intentionally matches the companion Shorts renderers:
- NOAA NCEI live download first
- cached text files second
- deterministic synthetic fixture third
- quick-preview mode
- preview PNGs
- CSV / JSON exports
- SRT captions
- generated ambient soundtrack
- final MP4 when ffmpeg is available


Scientific context:
- NASA: about 90% of excess heat from planetary warming has been absorbed by the ocean.
- NOAA: ocean heat content is a key measure of stored heat; modern 0-2000 m observing
  is strongly supported by the Argo float network.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    OCEAN_HEAT_QUICK=1 python the_ocean_is_storing_the_heat_short.py

Force offline fixture
---------------------
    OCEAN_HEAT_OFFLINE=1 python the_ocean_is_storing_the_heat_short.py

Full render
-----------
    python the_ocean_is_storing_the_heat_short.py

Outputs
-------
- final vertical MP4 with generated ambient audio when ffmpeg is available
- silent MP4 fallback
- SRT subtitle sidecar
- preview PNG frames
- CSV with yearly global ocean heat-content changes
- JSON summary and source notes
- cached NOAA text files


"""

import io
import json
import math
import os
import shutil
import subprocess
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("OCEAN_HEAT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("OCEAN_HEAT_OFFLINE", "0") == "1"

OUTPUT_ROOT = Path("the_ocean_is_storing_the_heat_output")
DATA_ROOT = OUTPUT_ROOT / "data"
CACHE_ROOT = DATA_ROOT / "cache"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, CACHE_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {

}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0

COLORS = {

}

FULL_SHOT_PLAN = [

]

FULL_CAPTIONS: List[Tuple[float, float, str]] = [
]

if QUICK_MODE:
    _k = float(CONFIG["duration_s"]) / 58.0
    SHOT_PLAN = [{**s, "start": s["start"] * _k, "end": s["end"] * _k} for s in FULL_SHOT_PLAN]
    CAPTIONS = [(a * _k, b * _k, text) for a, b, text in FULL_CAPTIONS]
else:
    SHOT_PLAN = FULL_SHOT_PLAN
    CAPTIONS = FULL_CAPTIONS

NOAA_URL_700 = (
    "https://www.ncei.noaa.gov/data/oceans/woa/DATA_ANALYSIS/"
    "3M_HEAT_CONTENT/DATA/basin/yearly/h22-w0-700m.dat"
)
NOAA_URL_2000 = (
    "https://www.ncei.noaa.gov/data/oceans/woa/DATA_ANALYSIS/"
    "3M_HEAT_CONTENT/DATA/basin/yearly/h22-w0-2000m.dat"
)


# -----------------------------------------------------------------------------
# Models / helpers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class OceanHeatDataset:
    years: np.ndarray
    total_2000_zj: np.ndarray
    total_2000_se_zj: np.ndarray
    upper_700_zj: np.ndarray
    upper_700_se_zj: np.ndarray
    deep_700_2000_zj: np.ndarray
    source: str
    notes: List[str]


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(value: float) -> float:
    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if float(shot["start"]) <= t < float(shot["end"]):
            return shot
    return SHOT_PLAN[-1]


def shot_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - float(shot["start"])) / max(float(shot["end"] - shot["start"]), 1e-9))


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for index, (start, end, text) in enumerate(captions, 1):
        lines.extend([str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def get_font(size: int, bold: bool = False, condensed: bool = False):
    candidates: List[str] = []
    if condensed and bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
            "DejaVuSansCondensed-Bold.ttf",
        ]
    if condensed:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
            "DejaVuSansCondensed.ttf",
        ]
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=max(7, int(size)))
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int,
    fill: Tuple[int, int, int, int],
    bold: bool = False,
    condensed: bool = False,
    anchor: str = "la",
    stroke: int = 2,
):
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold=bold, condensed=condensed),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(235, fill[3])),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill: Tuple[int, int, int, int],
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
        box = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)
        if box[2] - box[0] <= max_width:
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
        box = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += (box[3] - box[1]) + line_spacing


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.75, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# NOAA ocean-heat data
# -----------------------------------------------------------------------------

def request_text(url: str, timeout: int = 90) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; OceanHeatShort/1.0; educational visualization)",
            "Accept": "text/plain,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return payload.decode("utf-8", errors="replace")


def parse_noaa_basin_series(text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse NOAA seven-column basin heat time series.

    Expected columns:
      year+0.5, ocean basin, basin SE, north, north SE, south, south SE
    Heat units are 10^22 joules. Returns raw NOAA units.
    """
    rows: List[List[float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "!", ";")):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 7:
            continue
        try:
            values = [float(x) for x in parts[:7]]
        except ValueError:
            continue
        if 1900 <= values[0] <= 2200:
            rows.append(values)
    if len(rows) < 10:
        raise RuntimeError(f"NOAA basin series parse produced only {len(rows)} numeric rows")
    arr = np.asarray(rows, dtype=float)
    years = np.floor(arr[:, 0]).astype(np.int32)
    return years, arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6]


def _series_to_change_zj(values_22j: np.ndarray) -> np.ndarray:
    values = np.asarray(values_22j, dtype=float)
    # 1 x 10^22 J = 10 zettajoules. Rebase to the first year for an intuitive
    # "change in stored heat" storyline independent of NOAA's anomaly baseline.
    return (values - values[0]) * 10.0


def build_dataset_from_text(text_700: str, text_2000: str, source: str, notes: List[str]) -> OceanHeatDataset:
    y7, w7, se7, *_ = parse_noaa_basin_series(text_700)
    y2, w2, se2, *_ = parse_noaa_basin_series(text_2000)

    common = sorted(set(int(y) for y in y7).intersection(int(y) for y in y2))
    if len(common) < 10:
        raise RuntimeError("Too few common years between NOAA 0-700 m and 0-2000 m series")

    index7 = {int(y): i for i, y in enumerate(y7)}
    index2 = {int(y): i for i, y in enumerate(y2)}
    years = np.asarray(common, dtype=np.int32)
    raw700 = np.asarray([w7[index7[int(y)]] for y in years], dtype=float)
    raw2000 = np.asarray([w2[index2[int(y)]] for y in years], dtype=float)
    se700 = np.asarray([se7[index7[int(y)]] for y in years], dtype=float) * 10.0
    se2000 = np.asarray([se2[index2[int(y)]] for y in years], dtype=float) * 10.0

    upper = _series_to_change_zj(raw700)
    total = _series_to_change_zj(raw2000)
    deep = total - upper

    return OceanHeatDataset(
        years=years,
        total_2000_zj=total.astype(np.float32),
        total_2000_se_zj=se2000.astype(np.float32),
        upper_700_zj=upper.astype(np.float32),
        upper_700_se_zj=se700.astype(np.float32),
        deep_700_2000_zj=deep.astype(np.float32),
        source=source,
        notes=notes,
    )


def make_synthetic_dataset() -> OceanHeatDataset:
    years = np.arange(1955, 2026, dtype=np.int32)
    x = years - 1955
    rng = np.random.default_rng(19552025)

    # Deterministic shape calibrated only for visual timing / offline preview.
    total = 0.050 * x**2 + 1.8 * x + 8.0 * np.sin(x * 0.22)
    total -= total[0]
    upper_share = 0.79 - 0.0018 * x
    upper = total * upper_share + 3.0 * np.sin(x * 0.31)
    upper -= upper[0]
    deep = total - upper
    noise = rng.normal(0.0, 1.6, size=len(years))
    total = np.maximum.accumulate(total + noise * 0.35)
    upper = np.minimum(total, np.maximum.accumulate(upper + noise * 0.22))
    deep = total - upper
    se2 = np.linspace(16.0, 2.5, len(years))
    se7 = np.linspace(12.0, 2.0, len(years))

    return OceanHeatDataset(
        years=years,
        total_2000_zj=total.astype(np.float32),
        total_2000_se_zj=se2.astype(np.float32),
        upper_700_zj=upper.astype(np.float32),
        upper_700_se_zj=se7.astype(np.float32),
        deep_700_2000_zj=deep.astype(np.float32),
        source="synthetic_procedural_fixture",
        notes=[
            "Deterministic synthetic ocean-heat fixture for preview/timing only",
            "Not observational data; final values are intentionally illustrative",
        ],
    )


def load_ocean_heat_data() -> OceanHeatDataset:
    cache700 = CACHE_ROOT / "h22-w0-700m.dat"
    cache2000 = CACHE_ROOT / "h22-w0-2000m.dat"
    notes: List[str] = []

    if OFFLINE_MODE:
        return make_synthetic_dataset()

    try:
        text700 = request_text(NOAA_URL_700)
        text2000 = request_text(NOAA_URL_2000)
        if len(text700) < 500 or len(text2000) < 500:
            raise RuntimeError("NOAA response was unexpectedly small")
        cache700.write_text(text700, encoding="utf-8")
        cache2000.write_text(text2000, encoding="utf-8")
        return build_dataset_from_text(
            text700,
            text2000,
            "noaa_ncei_global_ocean_heat_live",
            [
                "Downloaded NOAA/NCEI World Ocean yearly heat-content series",
                "Heat values converted from 10^22 joules to zettajoules and rebased to first common year",
            ],
        )
    except Exception as exc:
        notes.append(f"Live NOAA download failed: {exc}")

    if cache700.exists() and cache2000.exists():
        try:
            return build_dataset_from_text(
                cache700.read_text(encoding="utf-8", errors="replace"),
                cache2000.read_text(encoding="utf-8", errors="replace"),
                "noaa_ncei_global_ocean_heat_cached",
                notes
                + [
                    "Loaded cached NOAA/NCEI World Ocean heat-content series",
                    "Heat values converted to zettajoules and rebased to first common year",
                ],
            )
        except Exception as exc:
            notes.append(f"Cached NOAA parse failed: {exc}")

    fixture = make_synthetic_dataset()
    return OceanHeatDataset(
        fixture.years,
        fixture.total_2000_zj,
        fixture.total_2000_se_zj,
        fixture.upper_700_zj,
        fixture.upper_700_se_zj,
        fixture.deep_700_2000_zj,
        fixture.source,
        notes + fixture.notes,
    )


def save_data_products(data: OceanHeatDataset) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "global_ocean_heat_content_change.csv"
    json_path = DATA_ROOT / "ocean_heat_summary.json"

    pd.DataFrame(
        {
            "year": data.years,
            "change_0_2000m_zettajoules": data.total_2000_zj,
            "standard_error_0_2000m_zettajoules": data.total_2000_se_zj,
            "change_0_700m_zettajoules": data.upper_700_zj,
            "standard_error_0_700m_zettajoules": data.upper_700_se_zj,
            "derived_change_700_2000m_zettajoules": data.deep_700_2000_zj,
        }
    ).to_csv(csv_path, index=False)

    latest = len(data.years) - 1
    summary = {
        "title": CONFIG["title"],
        "data_source": data.source,
        "start_year": int(data.years[0]),
        "end_year": int(data.years[-1]),
        "latest_change_0_2000m_zettajoules": float(data.total_2000_zj[latest]),
        "latest_change_0_700m_zettajoules": float(data.upper_700_zj[latest]),
        "latest_derived_change_700_2000m_zettajoules": float(data.deep_700_2000_zj[latest]),
        "unit_note": "1 zettajoule = 10^21 joules; NOAA basin heat series are published in 10^22 joules",
        "rebasing_note": "Renderer subtracts the first common year from each series to show change in stored heat",
        "science_context": [
            "About 90% of excess heat from planetary warming is absorbed by the ocean (NASA)",
            "Ocean heat content integrates stored heat through depth and is a key climate indicator (NOAA)",
            "Thermal expansion of warming seawater contributes to global sea-level rise",
        ],
        "source_notes": data.notes,
        "source_urls": {
            "noaa_global_ocean_heat_content": "https://www.ncei.noaa.gov/access/global-ocean-heat-content/",
            "noaa_0_700m": NOAA_URL_700,
            "noaa_0_2000m": NOAA_URL_2000,
            "nasa_ocean_warming": "https://science.nasa.gov/earth/explore/earth-indicators/ocean-warming/",
        },
        "fallback_warning": "synthetic_procedural_fixture is preview data, not observational data",
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class OceanHeatScene:
    def __init__(self, data: OceanHeatDataset):
        self.data = data
        self.rng = np.random.default_rng(9001)
        self.stars = [
            (
                float(self.rng.uniform(0, OUT_W)),
                float(self.rng.uniform(0, OUT_H)),
                float(self.rng.uniform(0.4, 1.8) * SCALE),
                float(self.rng.uniform(8, 44)),
                float(self.rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(75 if QUICK_MODE else 180)
        ]
        self.bubbles = [
            (
                float(self.rng.uniform(0.05, 0.95)),
                float(self.rng.uniform(0.0, 1.0)),
                float(self.rng.uniform(0.5, 1.6)),
                float(self.rng.uniform(1.0, 4.5) * SCALE),
                float(self.rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(45 if QUICK_MODE else 120)
        ]

        self.chart_box = (
            int(88 * SCALE),
            int(510 * SCALE),
            OUT_W - int(70 * SCALE),
            int(1325 * SCALE),
        )

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
        top = np.array(COLORS["space"], dtype=float)
        bottom = np.array(COLORS["space2"], dtype=float)
        for y in range(OUT_H):
            u = y / max(OUT_H - 1, 1)
            arr[y, :, :3] = (top * (1 - u) + bottom * u).astype(np.uint8)
            arr[y, :, 3] = 255
        image = Image.fromarray(arr, "RGBA")
        draw = ImageDraw.Draw(image)
        for x, y, r, a, phase in self.stars:
            alpha = int(a * (0.65 + 0.35 * math.sin(t * 0.75 + phase)))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=COLORS["white"] + (max(0, alpha),))

        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for cx, cy, rad, color, alpha in [
            (OUT_W * 0.18, OUT_H * 0.32, 330 * SCALE, COLORS["blue"], 26),
            (OUT_W * 0.82, OUT_H * 0.58, 360 * SCALE, COLORS["cyan"], 15),
            (OUT_W * 0.50, OUT_H * 0.88, 420 * SCALE, COLORS["orange"], 10),
        ]:
            hd.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=color + (alpha,))
        haze = haze.filter(ImageFilter.GaussianBlur(max(12, int(70 * SCALE))))
        image.alpha_composite(haze)
        return image

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 165, accent: Optional[Tuple[int, int, int]] = None):
        accent = accent or COLORS["cyan"]
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            box,
            radius=max(8, int(24 * SCALE)),
            fill=(2, 11, 24, alpha),
            outline=accent + (68,),
            width=max(1, int(2 * SCALE)),
        )
        image.alpha_composite(overlay)

    def draw_header(self, image: Image.Image, shot_name: str):
        draw_text(
            image,
            CONFIG["title"],
            (int(54 * SCALE), int(78 * SCALE)),
            31 if not QUICK_MODE else 16,
            COLORS["white"] + (248,),
            True,
            True,
            "la",
            2,
        )
        labels = {
            "opening": "EARTH'S HEAT RESERVOIR",
            "timeline": "GLOBAL OCEAN HEAT CONTENT",
            "depth": "WHERE THE HEAT GOES",
            "expansion": "WARM WATER EXPANDS",
            "marine": "A HOTTER BACKGROUND STATE",
            "finale": "THE CLIMATE HEAT BATTERY",
        }
        draw_text(
            image,
            labels.get(shot_name, "OCEAN HEAT"),
            (int(56 * SCALE), int(132 * SCALE)),
            12 if not QUICK_MODE else 6,
            COLORS["cyan"] + (220,),
            True,
            True,
            "la",
            1,
        )
        draw = ImageDraw.Draw(image)
        draw.line(
            (int(54 * SCALE), int(165 * SCALE), OUT_W - int(54 * SCALE), int(165 * SCALE)),
            fill=COLORS["cyan"] + (46,),
            width=max(1, int(2 * SCALE)),
        )

    def draw_source_hud(self, image: Image.Image):
        if self.data.source.startswith("synthetic"):
            text = "SYNTHETIC PREVIEW // NOT OBSERVATIONAL DATA"
        else:
            text = "NOAA/NCEI GLOBAL OCEAN HEAT // WORLD OCEAN // 0–2000 m"
        draw_text(
            image,
            text,
            (int(48 * SCALE), OUT_H - int(42 * SCALE)),
            10 if not QUICK_MODE else 5,
            COLORS["muted"] + (185,),
            True,
            True,
            "la",
            1,
        )

    def draw_caption(self, image: Image.Image, t: float):
        caption = None
        for start, end, text in CAPTIONS:
            if start <= t < end:
                caption = text
                break
        if not caption:
            return
        box = (int(56 * SCALE), int(1515 * SCALE), OUT_W - int(56 * SCALE), int(1790 * SCALE))
        self.panel(image, box, alpha=138, accent=COLORS["blue"])
        draw_wrapped_text(
            image,
            caption,
            (int(86 * SCALE), int(1560 * SCALE)),
            max_width=OUT_W - int(172 * SCALE),
            size=26 if not QUICK_MODE else 13,
            fill=COLORS["white"] + (244,),
            bold=False,
            line_spacing=max(4, int(8 * SCALE)),
        )

    def draw_opening(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        cx = OUT_W * 0.50
        cy = OUT_H * 0.47
        radius = 320 * SCALE

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse((cx - radius * 1.08, cy - radius * 1.08, cx + radius * 1.08, cy + radius * 1.08), fill=COLORS["cyan"] + (40,))
        glow = glow.filter(ImageFilter.GaussianBlur(max(8, int(42 * SCALE))))
        image.alpha_composite(glow)

        sphere = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sphere)
        sd.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=COLORS["ocean"] + (255,), outline=COLORS["cyan"] + (150,), width=max(1, int(4 * SCALE)))

        # Stylized continents.
        land = COLORS["land2"] + (225,)
        sd.polygon([
            (cx - 185 * SCALE, cy - 155 * SCALE),
            (cx - 95 * SCALE, cy - 220 * SCALE),
            (cx - 20 * SCALE, cy - 150 * SCALE),
            (cx - 55 * SCALE, cy - 55 * SCALE),
            (cx - 125 * SCALE, cy - 20 * SCALE),
            (cx - 180 * SCALE, cy - 80 * SCALE),
        ], fill=land)
        sd.polygon([
            (cx + 5 * SCALE, cy - 160 * SCALE),
            (cx + 125 * SCALE, cy - 185 * SCALE),
            (cx + 205 * SCALE, cy - 95 * SCALE),
            (cx + 125 * SCALE, cy - 25 * SCALE),
            (cx + 70 * SCALE, cy + 35 * SCALE),
            (cx + 10 * SCALE, cy - 20 * SCALE),
        ], fill=land)
        sd.polygon([
            (cx - 35 * SCALE, cy + 15 * SCALE),
            (cx + 35 * SCALE, cy + 20 * SCALE),
            (cx + 65 * SCALE, cy + 170 * SCALE),
            (cx - 5 * SCALE, cy + 235 * SCALE),
            (cx - 70 * SCALE, cy + 125 * SCALE),
        ], fill=land)
        image.alpha_composite(sphere)

        # 90% ring.
        ring_box = (cx - radius * 1.18, cy - radius * 1.18, cx + radius * 1.18, cy + radius * 1.18)
        rd = ImageDraw.Draw(image)
        rd.arc(ring_box, start=-90, end=-90 + 360 * 0.90 * p, fill=COLORS["gold"] + (245,), width=max(3, int(18 * SCALE)))
        rd.arc(ring_box, start=-90 + 360 * 0.90, end=270, fill=COLORS["muted"] + (65,), width=max(3, int(18 * SCALE)))

        draw_text(image, "~90%", (int(cx), int(cy - 25 * SCALE)), 112 if not QUICK_MODE else 56, COLORS["white"] + (252,), True, True, "mm", 3)
        draw_text(image, "OF EXCESS HEAT", (int(cx), int(cy + 72 * SCALE)), 26 if not QUICK_MODE else 13, COLORS["gold"] + (250,), True, True, "ma", 2)
        draw_text(image, "GOES INTO THE OCEAN", (int(cx), int(cy + 112 * SCALE)), 24 if not QUICK_MODE else 12, COLORS["white"] + (238,), True, True, "ma", 2)

    def chart_xy(self, year: float, value: float, ymin: float, ymax: float) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.chart_box
        xmin = float(self.data.years[0])
        xmax = float(self.data.years[-1])
        x = x0 + (year - xmin) / max(xmax - xmin, 1e-9) * (x1 - x0)
        y = y1 - (value - ymin) / max(ymax - ymin, 1e-9) * (y1 - y0)
        return x, y

    def draw_timeline(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        x0, y0, x1, y1 = self.chart_box
        self.panel(image, (x0 - int(30 * SCALE), y0 - int(90 * SCALE), x1 + int(20 * SCALE), y1 + int(75 * SCALE)), alpha=162)

        vals = self.data.total_2000_zj.astype(float)
        ymin = min(-10.0, float(np.nanmin(vals)) - 10.0)
        ymax = max(100.0, float(np.nanmax(vals)) * 1.10)
        draw = ImageDraw.Draw(image)

        # Grid / labels.
        for frac in np.linspace(0, 1, 5):
            y = y1 - frac * (y1 - y0)
            value = ymin + frac * (ymax - ymin)
            draw.line((x0, y, x1, y), fill=COLORS["muted"] + (30,), width=1)
            draw_text(image, f"{value:,.0f}", (x0 - int(16 * SCALE), int(y)), 12 if not QUICK_MODE else 6, COLORS["muted"] + (190,), False, True, "rm", 1)
        for year in [int(self.data.years[0]), 1975, 1995, 2015, int(self.data.years[-1])]:
            if year < int(self.data.years[0]) or year > int(self.data.years[-1]):
                continue
            x, _ = self.chart_xy(year, ymin, ymin, ymax)
            draw.line((x, y0, x, y1), fill=COLORS["muted"] + (22,), width=1)
            draw_text(image, str(year), (int(x), y1 + int(28 * SCALE)), 12 if not QUICK_MODE else 6, COLORS["muted"] + (205,), False, True, "ma", 1)

        draw_text(image, "CHANGE IN STORED HEAT // ZETTAJOULES", (x0, y0 - int(52 * SCALE)), 16 if not QUICK_MODE else 8, COLORS["cyan"] + (225,), True, True, "la", 1)

        count = max(2, int(round(2 + p * (len(self.data.years) - 2))))
        years = self.data.years[:count]
        shown = vals[:count]
        pts = [self.chart_xy(float(y), float(v), ymin, ymax) for y, v in zip(years, shown)]

        # Fill area under line.
        fill_layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        fd = ImageDraw.Draw(fill_layer)
        poly = [(pts[0][0], y1)] + pts + [(pts[-1][0], y1)]
        fd.polygon(poly, fill=COLORS["blue"] + (38,))
        image.alpha_composite(fill_layer)

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.line(pts, fill=COLORS["cyan"] + (110,), width=max(4, int(12 * SCALE)), joint="curve")
        glow = glow.filter(ImageFilter.GaussianBlur(max(2, int(8 * SCALE))))
        image.alpha_composite(glow)
        draw = ImageDraw.Draw(image)
        draw.line(pts, fill=COLORS["cyan"] + (245,), width=max(2, int(4 * SCALE)), joint="curve")

        last_year = int(years[-1])
        last_value = float(shown[-1])
        lx, ly = pts[-1]
        rr = max(3, int(9 * SCALE))
        draw.ellipse((lx - rr, ly - rr, lx + rr, ly + rr), fill=COLORS["gold"] + (255,))
        draw_text(image, str(last_year), (int(lx), int(ly - 34 * SCALE)), 16 if not QUICK_MODE else 8, COLORS["white"] + (245,), True, True, "ma", 1)
        draw_text(image, f"{last_value:+.0f} ZJ", (int(lx), int(ly + 32 * SCALE)), 16 if not QUICK_MODE else 8, COLORS["gold"] + (245,), True, True, "ma", 1)

    def draw_depth(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        box = (int(80 * SCALE), int(390 * SCALE), OUT_W - int(80 * SCALE), int(1390 * SCALE))
        self.panel(image, box, alpha=158)
        x0, y0, x1, y1 = box
        draw = ImageDraw.Draw(image)

        # Ocean column.
        water_box = (x0 + int(110 * SCALE), y0 + int(90 * SCALE), x1 - int(80 * SCALE), y1 - int(95 * SCALE))
        wx0, wy0, wx1, wy1 = water_box
        gradient = np.zeros((max(1, wy1 - wy0), max(1, wx1 - wx0), 4), dtype=np.uint8)
        top = np.array(COLORS["ocean_light"])
        bottom = np.array((4, 19, 48))
        for yy in range(gradient.shape[0]):
            u = yy / max(gradient.shape[0] - 1, 1)
            gradient[yy, :, :3] = (top * (1 - u) + bottom * u).astype(np.uint8)
            gradient[yy, :, 3] = 248
        image.alpha_composite(Image.fromarray(gradient, "RGBA"), (wx0, wy0))
        draw = ImageDraw.Draw(image)
        draw.rectangle(water_box, outline=COLORS["cyan"] + (85,), width=max(1, int(2 * SCALE)))

        # Depth labels.
        y700 = wy0 + (700 / 2000.0) * (wy1 - wy0)
        draw.line((wx0, y700, wx1, y700), fill=COLORS["white"] + (80,), width=max(1, int(2 * SCALE)))
        for depth, yy in [(0, wy0), (700, y700), (2000, wy1)]:
            draw_text(image, f"{depth:,} m", (wx0 - int(24 * SCALE), int(yy)), 14 if not QUICK_MODE else 7, COLORS["muted"] + (230,), True, True, "rm", 1)

        latest_total = float(self.data.total_2000_zj[-1])
        latest_upper = float(self.data.upper_700_zj[-1])
        latest_deep = float(self.data.deep_700_2000_zj[-1])
        denom = max(abs(latest_total), 1.0)
        upper_frac = clamp(latest_upper / denom, 0.0, 1.0)
        deep_frac = clamp(latest_deep / denom, 0.0, 1.0)

        # Heat clouds, increasing as shot progresses.
        heat = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(heat)
        rng = np.random.default_rng(4421)
        n = 28 if QUICK_MODE else 75
        for i in range(n):
            phase = i / max(n - 1, 1)
            if phase > p:
                continue
            layer_upper = rng.random() < max(0.35, upper_frac)
            yy = rng.uniform(wy0 + 10 * SCALE, y700 - 10 * SCALE) if layer_upper else rng.uniform(y700 + 10 * SCALE, wy1 - 10 * SCALE)
            xx = rng.uniform(wx0 + 15 * SCALE, wx1 - 15 * SCALE)
            rad = rng.uniform(8, 30) * SCALE
            color = COLORS["orange"] if layer_upper else COLORS["magenta"]
            alpha = int(rng.uniform(24, 76))
            hd.ellipse((xx - rad, yy - rad, xx + rad, yy + rad), fill=color + (alpha,))
        heat = heat.filter(ImageFilter.GaussianBlur(max(5, int(20 * SCALE))))
        image.alpha_composite(heat)

        draw_text(image, "0–700 m", (wx0 + int(28 * SCALE), int(wy0 + 48 * SCALE)), 22 if not QUICK_MODE else 11, COLORS["gold"] + (245,), True, True, "la", 2)
        draw_text(image, f"{latest_upper:+.0f} ZJ change", (wx0 + int(28 * SCALE), int(wy0 + 88 * SCALE)), 16 if not QUICK_MODE else 8, COLORS["white"] + (225,), False, True, "la", 1)
        draw_text(image, "700–2000 m", (wx0 + int(28 * SCALE), int(y700 + 50 * SCALE)), 22 if not QUICK_MODE else 11, COLORS["magenta"] + (245,), True, True, "la", 2)
        draw_text(image, f"{latest_deep:+.0f} ZJ derived change", (wx0 + int(28 * SCALE), int(y700 + 90 * SCALE)), 16 if not QUICK_MODE else 8, COLORS["white"] + (225,), False, True, "la", 1)

        # Animated Argo-like float silhouette.
        fx = wx1 - int(120 * SCALE)
        fy = int(lerp(wy0 + 80 * SCALE, wy1 - 120 * SCALE, 0.25 + 0.55 * p))
        draw.line((fx, fy - 42 * SCALE, fx, fy + 42 * SCALE), fill=COLORS["white"] + (210,), width=max(1, int(5 * SCALE)))
        draw.ellipse((fx - 18 * SCALE, fy - 18 * SCALE, fx + 18 * SCALE, fy + 18 * SCALE), fill=COLORS["cyan"] + (230,))
        draw_text(image, "ARGO", (int(fx), int(fy + 65 * SCALE)), 12 if not QUICK_MODE else 6, COLORS["cyan"] + (220,), True, True, "ma", 1)

    def draw_expansion(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        box = (int(90 * SCALE), int(410 * SCALE), OUT_W - int(90 * SCALE), int(1380 * SCALE))
        self.panel(image, box, alpha=160, accent=COLORS["gold"])
        x0, y0, x1, y1 = box
        draw = ImageDraw.Draw(image)

        # Two transparent cylinders: cool vs warm.
        centers = [x0 + (x1 - x0) * 0.30, x0 + (x1 - x0) * 0.70]
        base = y1 - int(160 * SCALE)
        h = int(620 * SCALE)
        widths = int(210 * SCALE)
        cool_top = base - h
        warm_extra = int(90 * SCALE * p)
        warm_top = base - h - warm_extra

        for idx, (cx, top, color, label) in enumerate([
            (centers[0], cool_top, COLORS["blue"], "COOLER"),
            (centers[1], warm_top, COLORS["orange"], "WARMER"),
        ]):
            left = cx - widths / 2
            right = cx + widths / 2
            draw.rounded_rectangle((left, top, right, base), radius=max(8, int(26 * SCALE)), fill=color + (75,), outline=color + (210,), width=max(1, int(3 * SCALE)))
            # water surface line
            draw.line((left + 8 * SCALE, top + 14 * SCALE, right - 8 * SCALE, top + 14 * SCALE), fill=COLORS["white"] + (180,), width=max(1, int(2 * SCALE)))
            draw_text(image, label, (int(cx), base + int(52 * SCALE)), 18 if not QUICK_MODE else 9, color + (245,), True, True, "ma", 1)

        # Expansion arrow.
        ax = centers[1] + widths * 0.72
        draw.line((ax, cool_top, ax, warm_top + 8 * SCALE), fill=COLORS["gold"] + (235,), width=max(2, int(5 * SCALE)))
        draw.polygon([
            (ax, warm_top),
            (ax - 14 * SCALE, warm_top + 24 * SCALE),
            (ax + 14 * SCALE, warm_top + 24 * SCALE),
        ], fill=COLORS["gold"] + (245,))
        draw_text(image, "THERMAL EXPANSION", (int(ax), int((cool_top + warm_top) / 2)), 14 if not QUICK_MODE else 7, COLORS["gold"] + (235,), True, True, "mm", 1)

        draw_text(image, "SAME WATER MASS", (OUT_W // 2, int(y0 + 70 * SCALE)), 18 if not QUICK_MODE else 9, COLORS["muted"] + (220,), True, True, "ma", 1)
        draw_text(image, "MORE HEAT  →  MORE VOLUME", (OUT_W // 2, int(y0 + 118 * SCALE)), 28 if not QUICK_MODE else 14, COLORS["white"] + (245,), True, True, "ma", 2)

    def draw_marine(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        box = (int(72 * SCALE), int(390 * SCALE), OUT_W - int(72 * SCALE), int(1400 * SCALE))
        self.panel(image, box, alpha=155, accent=COLORS["hot"])
        x0, y0, x1, y1 = box
        draw = ImageDraw.Draw(image)

        # Stylized ocean map / thermal field.
        map_box = (x0 + int(45 * SCALE), y0 + int(95 * SCALE), x1 - int(45 * SCALE), y0 + int(610 * SCALE))
        mx0, my0, mx1, my1 = map_box
        draw.rounded_rectangle(map_box, radius=max(8, int(20 * SCALE)), fill=COLORS["ocean"] + (245,), outline=COLORS["cyan"] + (60,), width=max(1, int(2 * SCALE)))

        # Simple land silhouettes.
        def mp(lon: float, lat: float) -> Tuple[float, float]:
            return mx0 + (lon + 180) / 360 * (mx1 - mx0), my0 + (90 - lat) / 180 * (my1 - my0)

        continents = [
            [(-168, 72), (-140, 60), (-122, 40), (-100, 23), (-80, 25), (-65, 46), (-50, 60), (-95, 70)],
            [(-82, 12), (-66, 5), (-52, -20), (-60, -50), (-74, -35)],
            [(-10, 70), (45, 65), (100, 55), (145, 50), (160, 25), (120, 12), (78, 10), (45, 30), (15, 45)],
            [(-15, 36), (10, 34), (35, 12), (30, -30), (10, -35), (-5, -5)],
            [(112, -10), (150, -20), (145, -42), (118, -35)],
        ]
        for poly in continents:
            draw.polygon([mp(lon, lat) for lon, lat in poly], fill=COLORS["land"] + (230,), outline=COLORS["land2"] + (210,))

        # Moving heatwave blobs.
        heat = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(heat)
        blobs = [
            (0.18, 0.42, 0.15, COLORS["orange"]),
            (0.47, 0.34, 0.13, COLORS["hot"]),
            (0.72, 0.48, 0.17, COLORS["magenta"]),
            (0.61, 0.68, 0.11, COLORS["orange"]),
        ]
        for i, (ux, uy, ur, color) in enumerate(blobs):
            pulse = 0.78 + 0.22 * math.sin(t * 2.0 + i * 1.7)
            rad = ur * (mx1 - mx0) * (0.5 + 0.5 * p) * pulse
            cx = mx0 + ux * (mx1 - mx0)
            cy = my0 + uy * (my1 - my0)
            hd.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=color + (int(70 * p),))
        heat = heat.filter(ImageFilter.GaussianBlur(max(6, int(30 * SCALE))))
        image.alpha_composite(heat)

        draw_text(image, "MARINE HEATWAVES", (OUT_W // 2, int(y0 + 675 * SCALE)), 30 if not QUICK_MODE else 15, COLORS["hot"] + (245,), True, True, "ma", 2)
        draw_text(image, "CORAL STRESS  •  ECOSYSTEM SHIFTS  •  STRONGER OCEAN HEAT", (OUT_W // 2, int(y0 + 730 * SCALE)), 13 if not QUICK_MODE else 7, COLORS["white"] + (225,), True, True, "ma", 1)

        # Coral silhouette.
        coral_y = y1 - int(110 * SCALE)
        for base_x in np.linspace(x0 + 90 * SCALE, x1 - 90 * SCALE, 7):
            sway = 10 * SCALE * math.sin(t * 1.5 + base_x * 0.01)
            draw.line((base_x, coral_y, base_x + sway, coral_y - 105 * SCALE), fill=COLORS["hot"] + (160,), width=max(2, int(7 * SCALE)))
            draw.line((base_x + sway, coral_y - 70 * SCALE, base_x + sway - 34 * SCALE, coral_y - 108 * SCALE), fill=COLORS["hot"] + (145,), width=max(2, int(5 * SCALE)))
            draw.line((base_x + sway, coral_y - 54 * SCALE, base_x + sway + 32 * SCALE, coral_y - 88 * SCALE), fill=COLORS["orange"] + (145,), width=max(2, int(5 * SCALE)))

    def draw_finale(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smoothstep(shot_progress(t, shot))
        latest = float(self.data.total_2000_zj[-1])
        year = int(self.data.years[-1])

        # Ocean disk.
        cx = OUT_W // 2
        cy = int(780 * SCALE)
        radius = 300 * SCALE
        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse((cx - radius * 1.18, cy - radius * 1.18, cx + radius * 1.18, cy + radius * 1.18), fill=COLORS["cyan"] + (38,))
        glow = glow.filter(ImageFilter.GaussianBlur(max(10, int(55 * SCALE))))
        image.alpha_composite(glow)
        draw = ImageDraw.Draw(image)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=COLORS["ocean_mid"] + (230,), outline=COLORS["cyan"] + (170,), width=max(2, int(5 * SCALE)))

        # Heat bands.
        for i in range(7):
            rr = radius * (0.30 + i * 0.09)
            alpha = int((35 + i * 8) * p)
            draw.arc((cx - rr, cy - rr, cx + rr, cy + rr), start=200, end=340, fill=COLORS["orange"] + (alpha,), width=max(2, int(9 * SCALE)))

        draw_text(image, str(year), (cx, int(cy - 70 * SCALE)), 62 if not QUICK_MODE else 31, COLORS["white"] + (250,), True, True, "ma", 2)
        draw_text(image, f"{latest:+.0f} ZJ", (cx, int(cy + 15 * SCALE)), 94 if not QUICK_MODE else 47, COLORS["gold"] + (250,), True, True, "mm", 3)
        draw_text(image, "CHANGE IN 0–2000 m OCEAN HEAT", (cx, int(cy + 92 * SCALE)), 18 if not QUICK_MODE else 9, COLORS["white"] + (230,), True, True, "ma", 1)
        draw_text(image, "SINCE THE START OF THIS RECORD", (cx, int(cy + 126 * SCALE)), 14 if not QUICK_MODE else 7, COLORS["muted"] + (220,), True, True, "ma", 1)

        draw_text(image, "THE OCEAN IS", (cx, int(1260 * SCALE)), 34 if not QUICK_MODE else 17, COLORS["white"] + (245,), True, True, "ma", 2)
        draw_text(image, "STORING THE HEAT", (cx, int(1325 * SCALE)), 56 if not QUICK_MODE else 28, COLORS["cyan"] + (250,), True, True, "ma", 3)

    def draw_film_texture(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        offset = int((t * 47) % 10)
        for y in range(offset, OUT_H, 10):
            draw.line((0, y, OUT_W, y), fill=(145, 170, 185, 5), width=1)
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        name = str(shot["name"])
        image = self.background(t)
        if name == "opening":
            self.draw_opening(image, t, shot)
        elif name == "timeline":
            self.draw_timeline(image, t, shot)
        elif name == "depth":
            self.draw_depth(image, t, shot)
        elif name == "expansion":
            self.draw_expansion(image, t, shot)
        elif name == "marine":
            self.draw_marine(image, t, shot)
        else:
            self.draw_finale(image, t, shot)

        self.draw_header(image, name)
        self.draw_caption(image, t)
        self.draw_source_hud(image)
        self.draw_film_texture(image, t)

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        arr *= VIGNETTE[..., None]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        graded = ImageEnhance.Contrast(Image.fromarray(arr)).enhance(float(CONFIG["contrast"]))
        graded = ImageEnhance.Color(graded).enhance(float(CONFIG["saturation"]))
        arr = np.asarray(graded, dtype=np.int16)
        rng = np.random.default_rng(int(t * 1000) + 8128)
        grain = rng.normal(0.0, float(CONFIG["grain_strength"]), arr.shape[:2])[:, :, None]
        arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
        return arr


# -----------------------------------------------------------------------------
# Audio / video
# -----------------------------------------------------------------------------

def generate_ambient_soundtrack(path: Path) -> Path:
    sr = int(CONFIG["sample_rate"])
    duration = float(CONFIG["duration_s"])
    n = int(round(duration * sr))
    times = np.arange(n, dtype=float) / sr
    rng = np.random.default_rng(2229)

    # Low oceanic drone + slow pulses + filtered-noise approximation.
    drone = (
        0.30 * np.sin(2 * np.pi * 42.0 * times)
        + 0.18 * np.sin(2 * np.pi * 63.0 * times + 0.4)
        + 0.10 * np.sin(2 * np.pi * 84.0 * times + 1.2)
    )
    pulse = (0.5 + 0.5 * np.sin(2 * np.pi * 0.09 * times - 0.5)) ** 2
    drone *= 0.52 + 0.48 * pulse

    noise = rng.normal(0.0, 1.0, n)
    kernel = np.ones(max(8, int(sr * 0.025)), dtype=float)
    kernel /= np.sum(kernel)
    wash = np.convolve(noise, kernel, mode="same")
    wash /= max(float(np.max(np.abs(wash))), 1e-9)

    # Shot transitions get subtle tonal swells.
    swells = np.zeros_like(times)
    for center in [7.2, 23.8, 36.2, 46.2, 54.1]:
        if QUICK_MODE:
            center *= float(CONFIG["duration_s"]) / 58.0
        swells += np.exp(-0.5 * ((times - center) / (0.55 if not QUICK_MODE else 0.18)) ** 2)

    audio = 0.56 * drone + 0.16 * wash + 0.08 * swells * np.sin(2 * np.pi * 126.0 * times)
    intro = np.clip(times / 1.2, 0, 1)
    intro = intro * intro * (3 - 2 * intro)
    outro_x = np.clip((times - (duration - 1.5)) / 1.2, 0, 1)
    outro = 1 - outro_x * outro_x * (3 - 2 * outro_x)
    audio *= intro * outro

    peak = max(float(np.max(np.abs(audio))), 1e-9)
    pcm = (np.clip(audio / peak * 0.86, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
    return path


def find_ffmpeg() -> Optional[str]:
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return shutil.which("ffmpeg")


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def render_video(scene: OceanHeatScene) -> Path:
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)

    raw_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_silent.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    audio_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_ambient.wav"

    frame_count = int(round(float(CONFIG["duration_s"]) * int(CONFIG["fps"])))
    times = np.arange(frame_count) / int(CONFIG["fps"])
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")

    with iio.get_writer(
        raw_path,
        fps=int(CONFIG["fps"]),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times, desc="Rendering ocean-heat short"):
            writer.append_data(scene.render_frame(float(t)))

    generate_ambient_soundtrack(audio_path)
    if mux_audio(raw_path, audio_path, final_path):
        print("Final video with audio:", final_path.resolve())
        return final_path

    shutil.copyfile(raw_path, final_path)
    print("ffmpeg audio mux unavailable; copied silent video to:", final_path.resolve())
    return final_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Title:", CONFIG["title"])
    print(f"Mode: {'QUICK' if QUICK_MODE else 'FULL'} // {OUT_W}x{OUT_H} @ {CONFIG['fps']} fps")
    print("Loading NOAA global ocean heat-content series ...")

    data = load_ocean_heat_data()
    csv_path, json_path = save_data_products(data)

    print("Data source:", data.source)
    print("Years:", int(data.years[0]), "to", int(data.years[-1]))
    print("Latest 0-2000 m change:", f"{float(data.total_2000_zj[-1]):+.1f} ZJ")
    print("Latest 0-700 m change:", f"{float(data.upper_700_zj[-1]):+.1f} ZJ")
    print("Latest derived 700-2000 m change:", f"{float(data.deep_700_2000_zj[-1]):+.1f} ZJ")
    print("CSV:", csv_path.resolve())
    print("Summary:", json_path.resolve())
    for note in data.notes:
        print("Data note:", note)

    scene = OceanHeatScene(data)
    if QUICK_MODE:
        preview_times = [0.8, 2.4, 4.8, 7.1, 9.5, 11.2]
    else:
        preview_times = [3.0, 13.0, 29.0, 41.0, 50.0, 56.0]

    for preview_time in tqdm(preview_times, desc="Preview frames"):
        Image.fromarray(scene.render_frame(float(preview_time))).save(
            PREVIEW_DIR / f"preview_{int(preview_time):02d}s.png"
        )

    render_video(scene)
    print("Output directory:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
