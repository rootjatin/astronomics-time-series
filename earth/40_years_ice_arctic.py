from __future__ import annotations

"""
40 Years of Arctic Sea Ice — cinematic YouTube Short renderer

Creates a vertical 1080x1920 science short showing four decades of change in
Arctic sea ice, centered on the 1985 -> 2025 span. The polar maps are stylized,
not geospatial reconstructions; numerical callouts are based on NASA/NSIDC
satellite-record summaries cited in the source notes written beside the output.

Scientific framing used in the narration
-----------------------------------------
- Continuous satellite observations of Arctic sea ice extend back to 1979.
- NSIDC reports the long-term downward trend in annual minimum Arctic sea ice
  extent from 1979 through 2025 as about 12.1% per decade relative to the
  1981-2010 average.
- The satellite-era record minimum was 3.39 million km^2 on 17 September 2012.
- The 2025 minimum was 4.60 million km^2 on 10 September 2025, tied for the
  tenth-lowest minimum in the record at the time.
- The last 19 annual minimums, 2007-2025, were the 19 lowest minimum extents in
  the satellite record.
- The 2025 winter maximum was 14.33 million km^2, the lowest maximum in the
  47-year satellite record at the time.
- Sea ice varies strongly from year to year because winds, weather, ocean heat,
  and other conditions affect each melt season. The long-term trend is the key.

Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    ARCTIC_ICE_SHORT_QUICK=1 python 40_years_of_arctic_sea_ice.py

Full render
-----------
    python 40_years_of_arctic_sea_ice.py

4K vertical
-----------
    ARCTIC_ICE_SHORT_4K=1 python 40_years_of_arctic_sea_ice.py
"""

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("ARCTIC_ICE_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("ARCTIC_ICE_SHORT_4K", "0") == "1" and not QUICK_MODE

OUT_W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
OUT_H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 8 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SCALE = OUT_W / 1080.0
OUT_SIZE = (OUT_W, OUT_H)

OUTPUT_ROOT = Path("40_years_of_arctic_sea_ice_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "40 YEARS OF ARCTIC SEA ICE",
    "subtitle": "1985 -> 2025 // satellites // summer minimum // long-term decline",
    "output_basename": "40_years_of_arctic_sea_ice",
    "contrast": 1.10,
    "saturation": 1.06,
    "vignette": 0.27,
}

COLORS = {
    "space": (2, 8, 18),
    "space2": (7, 18, 35),
    "ocean": (12, 63, 111),
    "ocean2": (22, 105, 161),
    "ice": (225, 244, 250),
    "ice_blue": (154, 222, 242),
    "ice_shadow": (84, 157, 188),
    "land": (93, 106, 94),
    "land2": (132, 124, 101),
    "white": (246, 250, 255),
    "muted": (171, 197, 214),
    "cyan": (76, 226, 255),
    "blue": (70, 137, 255),
    "gold": (255, 203, 91),
    "orange": (255, 137, 72),
    "red": (255, 80, 103),
    "violet": (177, 124, 255),
    "green": (105, 237, 170),
    "magenta": (238, 94, 194),
}

FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.4, 7.5, "Forty years apart, the Arctic at the end of summer can look like a different ocean. Satellite records show a much smaller ice cover today than in the 1980s."),
    (7.6, 17.2, "The change is not a smooth straight line. Weather moves the ice around every year — but the long-term direction is clear. Minimum extent has declined about 12 percent per decade since 1979."),
    (17.3, 27.3, "The most dramatic year was 2012. Arctic sea ice fell to a satellite-era record minimum of 3.39 million square kilometers."),
    (27.4, 38.1, "Then comes an important detail: a later year can have more ice than 2012 and still be part of the long-term decline. In 2025, the minimum was 4.60 million square kilometers."),
    (38.2, 49.0, "Sea ice also grows back every winter. But in 2025, even the winter maximum was the lowest in the 47-year satellite record — about 14.33 million square kilometers."),
    (49.1, 57.5, "Sea ice is more than a white cap. It changes how much sunlight the Arctic reflects and how heat and moisture move between the ocean and atmosphere. Four decades reveal a system being reshaped."),
]

if QUICK_MODE:
    factor = DURATION / 58.0
    CAPTIONS = [(a * factor, b * factor, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "forty_years", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "trend", "start": 8.0 if not QUICK_MODE else 1.8, "end": 18.0 if not QUICK_MODE else 4.0},
    {"name": "record_2012", "start": 18.0 if not QUICK_MODE else 4.0, "end": 28.0 if not QUICK_MODE else 6.25},
    {"name": "variability_2025", "start": 28.0 if not QUICK_MODE else 6.25, "end": 39.0 if not QUICK_MODE else 8.7},
    {"name": "seasonal_cycle", "start": 39.0 if not QUICK_MODE else 8.7, "end": 50.0 if not QUICK_MODE else 11.15},
    {"name": "why_it_matters", "start": 50.0 if not QUICK_MODE else 11.15, "end": DURATION},
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(value: float) -> float:
    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]
