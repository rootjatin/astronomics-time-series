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
