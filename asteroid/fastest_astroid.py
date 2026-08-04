from __future__ import annotations

"""
The Fastest Asteroids Ever Discovered
====================================

A cinematic vertical YouTube Short renderer that ranks asteroids by their
Earth-relative close-approach speed using NASA/JPL's close-approach database.

What the video shows
--------------------
- A live leaderboard of the fastest asteroid close-approach records returned by
  the NASA/JPL SBDB Close-Approach Data API.
- One fastest encounter per asteroid (deduplicated by designation), ranked by
  maximum published Earth-relative close-approach speed `v_rel` in km/s.
- A bar-chart leaderboard, detailed cards, a timeline of encounter dates, and
  speed comparisons against familiar benchmarks.

Official live source
--------------------
NASA/JPL SBDB Close-Approach Data API:
    https://ssd-api.jpl.nasa.gov/cad.api
    https://ssd-api.jpl.nasa.gov/doc/cad.html

 interpretation rules
------------------------------
- "Fastest" in this video means the highest published Earth-relative
  close-approach speed (`v_rel`) among asteroid close-approach records returned
  by the JPL CAD API for the configured date span.
- This is not the same as the asteroid's maximum heliocentric speed around the
  Sun, and it is not a measure of impact energy.
- The same asteroid may appear many times in the close-approach database. The
  script keeps only the single fastest encounter for each asteroid.
- Rankings may change if JPL's database, orbit solutions, or future date span
  changes.


Offline fallback
----------------
If the JPL API cannot be reached, the script uses a clearly labeled layout
fixture with approximate example values so the video can still be rendered.

Install
-------
    pip install numpy pandas pillow imageio imageio-ffmpeg requests tqdm

Run final quality
-----------------
    python the_fastest_asteroids_ever_discovered_short.py

Run quick preview
-----------------
    ASTEROID_SPEED_SHORT_QUICK=1 python the_fastest_asteroids_ever_discovered_short.py

Force live refresh
------------------
    ASTEROID_SPEED_SHORT_REFRESH=1 python the_fastest_asteroids_ever_discovered_short.py

Force offline layout testing
----------------------------
    ASTEROID_SPEED_SHORT_OFFLINE=1 ASTEROID_SPEED_SHORT_QUICK=1 \
        python the_fastest_asteroids_ever_discovered_short.py
"""

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import requests
except Exception:
    requests = None


# =============================================================================
# Configuration
# =============================================================================

QUICK_MODE = os.environ.get("ASTEROID_SPEED_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("ASTEROID_SPEED_SHORT_OFFLINE", "0") == "1"
REFRESH = os.environ.get("ASTEROID_SPEED_SHORT_REFRESH", "0") == "1"

OUTPUT_ROOT = Path("the_fastest_asteroids_ever_discovered_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
CACHE_ROOT = OUTPUT_ROOT / "cache"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT, CACHE_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "width": 540 if QUICK_MODE else 1080,
    "height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "basename": "the_fastest_asteroids_ever_discovered",
    "title": "THE FASTEST ASTEROIDS EVER DISCOVERED",
    "subtitle": "Ranked by Earth-close-approach speed from NASA/JPL",
    "timeout_s": 45,
    "cache_hours": 18,
    "api_url": "https://ssd-api.jpl.nasa.gov/cad.api",
    "api_docs": "https://ssd-api.jpl.nasa.gov/doc/cad.html",
    "source_url": "https://ssd.jpl.nasa.gov/tools/cad.html",
    "date_min": "1900-01-01",
    "date_max": "2200-01-01",
    "fetch_limit": 2500,
    "leaderboard_count": 10,
    "detail_count": 6,
    "timeline_count": 20,
    "stars": 620 if QUICK_MODE else 1100,
}

W = CONFIG["width"]
H = CONFIG["height"]
SIZE = (W, H)
SCALE = W / 1080.0
LD_PER_AU = 149597870.7 / 384400.0

COLORS = {
    "bg": (4, 8, 16),
    "white": (246, 249, 255),
    "muted": (150, 198, 222),
    "cyan": (92, 223, 255),
    "gold": (255, 197, 92),
    "violet": (201, 116, 255),
    "green": (104, 255, 181),
    "red": (255, 115, 125),
    "orange": (255, 160, 84),
}

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.0 if not QUICK_MODE else 1.8},
    {"name": "leaderboard", "start": 7.0 if not QUICK_MODE else 1.8, "end": 20.0 if not QUICK_MODE else 4.3},
    {"name": "cards", "start": 20.0 if not QUICK_MODE else 4.3, "end": 33.0 if not QUICK_MODE else 6.9},
    {"name": "timeline", "start": 33.0 if not QUICK_MODE else 6.9, "end": 43.0 if not QUICK_MODE else 8.9},
    {"name": "compare", "start": 43.0 if not QUICK_MODE else 8.9, "end": 52.0 if not QUICK_MODE else 10.8},
    {"name": "outro", "start": 52.0 if not QUICK_MODE else 10.8, "end": CONFIG["duration_s"]},
]

CAPTIONS = [
    (0.4, 6.8, "These are the fastest asteroids in this snapshot of NASA/JPL's close-approach database."),
    (6.9, 20.0, "Here, fastest means the highest Earth-relative close-approach speed, not the top speed around the Sun."),
    (20.1, 33.0, "Each asteroid may have many recorded encounters, so this ranking keeps only the single fastest one for each object."),
    (33.1, 43.0, "The encounter dates stretch across decades, and the leaderboard can change as orbit solutions and future close approaches are updated."),
    (43.1, 52.0, "For scale, even the fastest entries move several times faster than Earth escape speed and far faster than spacecraft in low Earth orbit."),
    (52.1, 57.7, "This video is a data-grounded ranking from JPL CAD, not a full hazard ranking or an impact forecast."),
]


# =============================================================================
# Data model
# =============================================================================

@dataclass
class AsteroidEncounter:
    designation: str
    fullname: str
    close_approach_cd: str
    jd: float
    dist_au: float
    dist_ld: float
    v_rel_km_s: float
    v_inf_km_s: float
    h_mag: float
    diameter_km: float
    orbit_id: str


# =============================================================================
# Utilities
# =============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def ease_in_out_sine(t: float) -> float:
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


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


def clip_text(text: str, n: int = 34) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(7, int(size)))
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int = 28,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    anchor: str = "la",
    stroke: int = 2,
):
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(220, fill[3] if len(fill) > 3 else 220)),
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
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for word in words:
        test = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), test, font=font, stroke_width=2)
        if bbox[2] - bbox[0] <= max_width:
            current = test
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
        y += bbox[3] - bbox[1] + line_spacing


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(path: Path):
    lines = []
    for i, (start, end, text) in enumerate(CAPTIONS, start=1):
        lines.extend([str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_vignette(width: int, height: int, strength: float = 0.25) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1 - strength * radius**1.8, 0, 1).astype(np.float32)


VIGNETTE = make_vignette(W, H)


# =============================================================================
# Data collection
# =============================================================================


def row_to_encounter(row: Dict[str, object]) -> AsteroidEncounter:
    dist_au = safe_float(row.get("dist"))
    return AsteroidEncounter(
        designation=str(row.get("des") or ""),
        fullname=str(row.get("fullname") or row.get("des") or "Unknown asteroid"),
        close_approach_cd=str(row.get("cd") or ""),
        jd=safe_float(row.get("jd")),
        dist_au=dist_au,
        dist_ld=dist_au * LD_PER_AU if np.isfinite(dist_au) else np.nan,
        v_rel_km_s=safe_float(row.get("v_rel")),
        v_inf_km_s=safe_float(row.get("v_inf")),
        h_mag=safe_float(row.get("h")),
        diameter_km=safe_float(row.get("diameter")),
        orbit_id=str(row.get("orbit_id") or ""),
    )



def parse_cad_payload(payload: Dict) -> List[AsteroidEncounter]:
    fields = payload.get("fields") or []
    rows = payload.get("data") or []
    if not fields or not isinstance(rows, list):
        raise RuntimeError("JPL CAD response did not contain fields/data arrays")
    records = []
    for values in rows:
        row = dict(zip(fields, values))
        records.append(row_to_encounter(row))
    return records



def fetch_live_encounters() -> Tuple[List[AsteroidEncounter], Dict]:
    cache_path = CACHE_ROOT / "jpl_fast_asteroids_cad.json"
    payload: Optional[Dict] = None
    mode = "live"

    if cache_path.exists() and not REFRESH:
        age_hours = (utc_now().timestamp() - cache_path.stat().st_mtime) / 3600.0
        if age_hours <= CONFIG["cache_hours"]:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            mode = "cache"

    if payload is None:
        if requests is None:
            raise RuntimeError("requests is unavailable")
        response = requests.get(
            CONFIG["api_url"],
            params={
                "body": "Earth",
                "kind": "a",
                "date-min": CONFIG["date_min"],
                "date-max": CONFIG["date_max"],
                "sort": "-v-rel",
                "limit": str(CONFIG["fetch_limit"]),
                "fullname": "1",
                "diameter": "1",
            },
            timeout=CONFIG["timeout_s"],
            headers={"User-Agent": "FastestAsteroidsShort/1.0 educational renderer"},
        )
        response.raise_for_status()
        payload = response.json()
        cache_path.write_text(json.dumps(payload), encoding="utf-8")

    records = parse_cad_payload(payload)
    summary = {
        "generated_at_utc": iso_z(utc_now()),
        "data_status": mode,
        "source_url": CONFIG["api_url"],
        "query_date_min": CONFIG["date_min"],
        "query_date_max": CONFIG["date_max"],
        "fetched_close_approach_records": len(records),
        "offline_fixture": False,
    }
    return records, summary



def fixture_encounters() -> Tuple[List[AsteroidEncounter], Dict]:
    # Approximate example values used only for offline layout testing.
    raw = [
        ("2021 PH27", "(2021 PH27)", "2021-Aug-14 00:00", 2459440.5, 0.098, 52.6, 48.2, 17.0, np.nan),
        ("2020 AV2", "(2020 AV2)", "2020-Jan-04 00:00", 2458852.5, 0.121, 46.8, 19.0, 16.4, np.nan),
        ("2005 HC4", "(2005 HC4)", "2005-Apr-15 00:00", 2453470.5, 0.143, 45.7, 17.1, 18.0, np.nan),
        ("2019 LF6", "(2019 LF6)", "2019-Jun-10 00:00", 2458644.5, 0.154, 44.9, 18.2, 16.9, np.nan),
        ("2015 XX169", "(2015 XX169)", "2015-Dec-11 00:00", 2457368.5, 0.049, 43.8, 25.1, 24.0, np.nan),
        ("99942 Apophis", "99942 Apophis", "2029-Apr-13 21:46", 2462245.4, 0.00025, 37.4, 19.7, 19.7, 0.37),
        ("3200 Phaethon", "3200 Phaethon", "2017-Dec-16 11:00", 2458103.9, 0.069, 35.8, 14.0, 14.3, 5.1),
        ("2004 FU162", "(2004 FU162)", "2004-Mar-31 03:00", 2453095.6, 0.00009, 29.5, 28.0, 28.4, np.nan),
        ("2011 CQ1", "(2011 CQ1)", "2011-Feb-04 19:39", 2455597.3, 0.00004, 31.2, 28.4, 30.5, np.nan),
        ("2012 DA14", "(2012 DA14)", "2013-Feb-15 19:25", 2456339.3, 0.00023, 28.1, 24.0, 24.4, 0.045),
        ("2001 FO32", "(231937) 2001 FO32", "2021-Mar-21 16:03", 2459295.2, 0.0135, 34.4, 17.0, 17.7, 0.77),
        ("2019 OK", "(2019 OK)", "2019-Jul-25 01:22", 2458689.6, 0.00048, 24.5, 24.2, 25.6, 0.13),
    ]
    rows = [
        AsteroidEncounter(
            designation=des,
            fullname=fullname,
            close_approach_cd=cd,
            jd=jd,
            dist_au=dist_au,
            dist_ld=dist_au * LD_PER_AU,
            v_rel_km_s=v_rel,
            v_inf_km_s=v_inf,
            h_mag=hmag,
            diameter_km=diam,
            orbit_id="fixture",
        )
        for des, fullname, cd, jd, dist_au, v_rel, v_inf, hmag, diam in raw
    ]
    summary = {
        "generated_at_utc": iso_z(utc_now()),
        "data_status": "offline-fixture",
        "source_url": CONFIG["api_url"],
        "query_date_min": CONFIG["date_min"],
        "query_date_max": CONFIG["date_max"],
        "fetched_close_approach_records": len(rows),
        "offline_fixture": True,
    }
    return rows, summary



def rank_unique_fastest(records: Sequence[AsteroidEncounter]) -> List[AsteroidEncounter]:
    best: Dict[str, AsteroidEncounter] = {}
    for rec in records:
        key = rec.designation.strip() or rec.fullname.strip()
        if not key:
            continue
        if not np.isfinite(rec.v_rel_km_s):
            continue
        if key not in best or rec.v_rel_km_s > best[key].v_rel_km_s:
            best[key] = rec
    ranked = sorted(best.values(), key=lambda r: (-r.v_rel_km_s, r.jd))
    return ranked



def collect_data() -> Tuple[List[AsteroidEncounter], Dict]:
    errors = {}
    if OFFLINE_MODE:
        records, summary = fixture_encounters()
    else:
        try:
            records, summary = fetch_live_encounters()
        except Exception as exc:
            errors["cad_fetch"] = str(exc)
            records, summary = fixture_encounters()

    ranked = rank_unique_fastest(records)
    top = ranked[0] if ranked else None
    summary.update({
        "unique_asteroids_ranked": len(ranked),
        "leaderboard_count": min(CONFIG["leaderboard_count"], len(ranked)),
        "top_speed_km_s": top.v_rel_km_s if top else np.nan,
        "top_object": top.fullname if top else "n/a",
        "top_date": top.close_approach_cd if top else "n/a",
        "errors": errors,
        "warning": "Fastest means highest JPL CAD Earth-relative close-approach speed (v_rel) in the queried date span, one fastest encounter per asteroid.",
    })
    return ranked, summary



def save_data(ranked: Sequence[AsteroidEncounter], summary: Dict) -> Tuple[Path, Path]:
    df = pd.DataFrame([asdict(row) for row in ranked])
    csv_path = DATA_ROOT / "fastest_asteroids_ranked.csv"
    json_path = DATA_ROOT / "fastest_asteroids_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"summary": summary, "ranked": [asdict(row) for row in ranked]}, indent=2), encoding="utf-8")
    return csv_path, json_path


# =============================================================================
# Scene renderer
# =============================================================================

class FastAsteroidsScene:
    def __init__(self, ranked: Sequence[AsteroidEncounter], summary: Dict):
        self.ranked = list(ranked)
        self.summary = summary
        self.top_n = self.ranked[: CONFIG["leaderboard_count"]]
        self.detail = self.ranked[: CONFIG["detail_count"]]
        self.timeline = self.ranked[: CONFIG["timeline_count"]]
        self.top_speed = max((r.v_rel_km_s for r in self.ranked if np.isfinite(r.v_rel_km_s)), default=1.0)
        self.timeline_min = min((r.jd for r in self.timeline if np.isfinite(r.jd)), default=2450000.0)
        self.timeline_max = max((r.jd for r in self.timeline if np.isfinite(r.jd)), default=2465000.0)
        self.stars = self._make_stars(CONFIG["stars"], seed=3482)

    @staticmethod
    def _make_stars(n: int, seed: int):
        rng = np.random.default_rng(seed)
        return [
            (float(rng.uniform(0, W)), float(rng.uniform(0, H)), float(rng.uniform(.4, 2.0) * SCALE),
             int(rng.integers(25, 145)), float(rng.uniform(0, math.tau)))
            for _ in range(n)
        ]

    def background(self, t: float) -> Image.Image:
        img = Image.new("RGBA", SIZE, COLORS["bg"] + (255,))
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        clouds = [
            (W * 0.18, H * 0.31, (16, 70, 125)),
            (W * 0.72, H * 0.25, (90, 35, 110)),
            (W * 0.48, H * 0.76, (15, 65, 92)),
        ]
        for cx, cy, color in clouds:
            for radius, alpha in [(W * 0.46, 13), (W * 0.30, 23), (W * 0.18, 32)]:
                gd.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=color + (alpha,))
        glow = glow.filter(ImageFilter.GaussianBlur(65 if not QUICK_MODE else 32))
        img.alpha_composite(glow)

        d = ImageDraw.Draw(img)
        for x, y, r, a, phase in self.stars:
            alpha = int(a * (0.72 + 0.28 * math.sin(1.6 * t + phase)))
            d.ellipse((x-r, y-r, x+r, y+r), fill=(214, 228, 255, alpha))
        return img

    def draw_title(self, img: Image.Image, t: float):
        alpha = int(255 * smoothstep((t - 0.15) / 0.8) * (1 - smoothstep((t - (6.4 if not QUICK_MODE else 1.55)) / 0.8)))
        if alpha > 4:
            draw_text(img, CONFIG["title"], (56 if not QUICK_MODE else 28, 90 if not QUICK_MODE else 45),
                      size=41 if not QUICK_MODE else 19, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(img, CONFIG["subtitle"], (58 if not QUICK_MODE else 30, 151 if not QUICK_MODE else 76),
                      size=22 if not QUICK_MODE else 10, fill=COLORS["cyan"] + (min(alpha, 230),), bold=True)
        shot_titles = {
            "intro": "ONE METRIC • MANY FAST FLYBYS",
            "leaderboard": "TOP SPEEDS IN THE JPL DATABASE SNAPSHOT",
            "cards": "FASTEST ENCOUNTERS • ONE PER ASTEROID",
            "timeline": "WHEN THE FASTEST FLYBYS OCCUR",
            "compare": "HOW FAST IS FAST?",
            "outro": "RANKINGS CAN CHANGE",
        }
        if t > (5.0 if not QUICK_MODE else 1.25):
            draw_text(img, shot_titles[get_shot(t)["name"]], (56 if not QUICK_MODE else 28, 61 if not QUICK_MODE else 30),
                      size=19 if not QUICK_MODE else 9, fill=COLORS["muted"] + (210,), bold=True, stroke=1)

    def draw_source_hud(self, img: Image.Image):
        status = "OFFLINE FIXTURE" if self.summary.get("offline_fixture") else ("CACHE" if self.summary.get("data_status") == "cache" else "LIVE")
        label = f"CAD DATA // {status}"
        draw_text(img, label, (W - (48 if not QUICK_MODE else 24), 72 if not QUICK_MODE else 36),
                  size=17 if not QUICK_MODE else 8, fill=COLORS["cyan"] + (220,), bold=True, anchor="ra", stroke=1)
        generated = self.summary["generated_at_utc"].replace("T", " ").replace("Z", " UTC")
        draw_text(img, generated, (W - (48 if not QUICK_MODE else 24), 102 if not QUICK_MODE else 51),
                  size=14 if not QUICK_MODE else 7, fill=COLORS["muted"] + (195,), anchor="ra", stroke=1)

    def draw_caption(self, img: Image.Image, t: float):
        caption = caption_at(t)
        if not caption:
            return
        y0 = H - (244 if not QUICK_MODE else 124)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((44 if not QUICK_MODE else 22, y0, W-(44 if not QUICK_MODE else 22), y0+(124 if not QUICK_MODE else 66)),
                             radius=24 if not QUICK_MODE else 12, fill=(2, 6, 14, 172), outline=(80, 185, 220, 65), width=1)
        img.alpha_composite(overlay)
        draw_wrapped_text(img, caption, (68 if not QUICK_MODE else 34, y0+(28 if not QUICK_MODE else 14)),
                          W-(136 if not QUICK_MODE else 68), size=29 if not QUICK_MODE else 14,
                          fill=COLORS["white"] + (245,))

    def draw_hud_noise(self, img: Image.Image, t: float):
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        offset = int((t * 39) % 7)
        for y in range(offset, H, 7):
            od.line((0, y, W, y), fill=(120, 200, 240, 10), width=1)
        scan_y = int((t * 165) % (H + 220)) - 110
        od.rectangle((0, scan_y, W, scan_y + (48 if not QUICK_MODE else 24)), fill=(90, 210, 240, 7))
        img.alpha_composite(overlay)

    def draw_intro(self, img: Image.Image, t: float):
        d = ImageDraw.Draw(img)
        cx, cy = W * 0.5, H * 0.40
        radius = 210 * SCALE
        start_ang = math.pi * 0.80
        end_ang = math.pi * 0.20
        bbox = (cx-radius, cy-radius, cx+radius, cy+radius)
        d.arc(bbox, start=math.degrees(start_ang), end=math.degrees(2 * math.pi + end_ang), fill=COLORS["muted"] + (90,), width=max(6, int(7*SCALE)))
        if self.top_n:
            frac = clamp((0.35 + 0.65 * ease_in_out_sine((t - SHOT_PLAN[0]["start"]) / max(1e-6, SHOT_PLAN[0]["end"] - SHOT_PLAN[0]["start"]))))
            needle_speed = self.top_n[0].v_rel_km_s * frac
        else:
            needle_speed = 0.0
        needle_frac = clamp(needle_speed / max(self.top_speed, 1e-6))
        ang = lerp(start_ang, end_ang, needle_frac)
        nx = cx + math.cos(ang) * radius * 0.82
        ny = cy - math.sin(ang) * radius * 0.82
        d.line((cx, cy, nx, ny), fill=COLORS["gold"] + (235,), width=max(4, int(5*SCALE)))
        d.ellipse((cx-14*SCALE, cy-14*SCALE, cx+14*SCALE, cy+14*SCALE), fill=COLORS["gold"] + (255,))
        for i, label in enumerate(np.linspace(0, self.top_speed, 5)):
            frac = i / 4
            a = lerp(start_ang, end_ang, frac)
            tx = cx + math.cos(a) * radius * 0.96
            ty = cy - math.sin(a) * radius * 0.96
            draw_text(img, f"{label:.0f}", (int(tx), int(ty)), size=16 if not QUICK_MODE else 7,
                      fill=COLORS["muted"] + (220,), anchor="ma", stroke=1)
        draw_text(img, f"{needle_speed:.1f} km/s", (int(cx), int(cy + 42 * SCALE)), size=54 if not QUICK_MODE else 24,
                  fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=1)
        top_name = clip_text(self.top_n[0].fullname if self.top_n else "n/a", 30)
        draw_text(img, top_name, (int(cx), int(cy + 94 * SCALE)), size=24 if not QUICK_MODE else 11,
                  fill=COLORS["cyan"] + (230,), bold=True, anchor="ma", stroke=1)
        draw_text(img, "FASTEST EARTH-RELATIVE APPROACH IN THIS SNAPSHOT", (W // 2, int(H * 0.64)),
                  size=20 if not QUICK_MODE else 9, fill=COLORS["muted"] + (220,), bold=True, anchor="ma", stroke=1)

    def draw_leaderboard(self, img: Image.Image, t: float):
        x0, y0, x1, y1 = int(W * 0.07), int(H * 0.18), int(W * 0.93), int(H * 0.72)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0, y0, x1, y1), radius=28 if not QUICK_MODE else 14,
                             fill=(2, 6, 14, 176), outline=(88, 185, 220, 78), width=2)
        img.alpha_composite(overlay)
        draw_text(img, "TOP UNIQUE ASTEROIDS BY MAX V_REL", (x0 + 18, y0 + 18), size=18 if not QUICK_MODE else 8,
                  fill=COLORS["cyan"] + (220,), bold=True, stroke=1)

        row_h = (y1 - y0 - 56) / max(len(self.top_n), 1)
        reveal = smoothstep((t - SHOT_PLAN[1]["start"]) / max(1e-6, SHOT_PLAN[1]["end"] - SHOT_PLAN[1]["start"]))
        count = max(1, int(math.ceil(len(self.top_n) * reveal))) if self.top_n else 0
        for idx, rec in enumerate(self.top_n[:count]):
            yy = y0 + 46 + idx * row_h
            bar_x = x0 + 60
            bar_w = (x1 - x0 - 280) * clamp(rec.v_rel_km_s / max(self.top_speed, 1e-6))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((bar_x, yy + 8, bar_x + (x1 - x0 - 280), yy + row_h - 10), radius=12 if not QUICK_MODE else 6,
                                fill=(20, 35, 48, 90))
            color = COLORS["gold"] if idx == 0 else COLORS["cyan"]
            d.rounded_rectangle((bar_x, yy + 8, bar_x + bar_w, yy + row_h - 10), radius=12 if not QUICK_MODE else 6,
                                fill=color + (215,))
            draw_text(img, f"#{idx+1}", (x0 + 18, int(yy + row_h * 0.42)), size=20 if not QUICK_MODE else 9,
                      fill=COLORS["white"] + (235,), bold=True, stroke=1)
            draw_text(img, clip_text(rec.fullname, 28), (bar_x + 14, int(yy + row_h * 0.42)), size=18 if not QUICK_MODE else 8,
                      fill=(10, 18, 30, 240) if idx == 0 else (8, 18, 30, 230), bold=True, stroke=1)
            draw_text(img, f"{rec.v_rel_km_s:.1f} km/s", (x1 - 18, int(yy + row_h * 0.42)), size=18 if not QUICK_MODE else 8,
                      fill=COLORS["white"] + (235,), bold=True, anchor="ra", stroke=1)

    def draw_cards(self, img: Image.Image):
        x_positions = [int(W * 0.07), int(W * 0.53)]
        y_positions = [int(H * 0.20), int(H * 0.39), int(H * 0.58)]
        card_w = int(W * 0.40)
        card_h = int(H * 0.15)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for idx, rec in enumerate(self.detail):
            x = x_positions[idx % 2]
            y = y_positions[idx // 2]
            color = COLORS["gold"] if idx == 0 else COLORS["cyan"]
            od.rounded_rectangle((x, y, x + card_w, y + card_h), radius=24 if not QUICK_MODE else 12,
                                 fill=(3, 8, 17, 186), outline=color + (80,), width=2)
        img.alpha_composite(overlay)
        for idx, rec in enumerate(self.detail):
            x = x_positions[idx % 2]
            y = y_positions[idx // 2]
            color = COLORS["gold"] if idx == 0 else COLORS["cyan"]
            draw_text(img, f"#{idx+1} {clip_text(rec.fullname, 24)}", (x + 16, y + 16), size=21 if not QUICK_MODE else 10,
                      fill=color + (240,), bold=True, stroke=1)
            draw_text(img, f"{rec.v_rel_km_s:.1f} km/s", (x + 16, y + 48 if not QUICK_MODE else y + 24), size=20 if not QUICK_MODE else 9,
                      fill=COLORS["white"] + (235,), bold=True, stroke=1)
            draw_text(img, clip_text(rec.close_approach_cd, 22), (x + card_w - 16, y + 48 if not QUICK_MODE else y + 24), size=15 if not QUICK_MODE else 7,
                      fill=COLORS["muted"] + (220,), anchor="ra", stroke=1)
            diam = f"{rec.diameter_km:.2f} km" if np.isfinite(rec.diameter_km) else "n/a"
            draw_text(img, f"Dist {rec.dist_ld:.2f} LD", (x + 16, y + card_h - 44 if not QUICK_MODE else y + card_h - 22), size=15 if not QUICK_MODE else 7,
                      fill=COLORS["muted"] + (220,), stroke=1)
            draw_text(img, f"Diameter {diam}", (x + card_w - 16, y + card_h - 44 if not QUICK_MODE else y + card_h - 22), size=15 if not QUICK_MODE else 7,
                      fill=COLORS["muted"] + (220,), anchor="ra", stroke=1)

    def time_x(self, jd: float, x0: int, x1: int) -> float:
        if self.timeline_max <= self.timeline_min:
            return x0
        return x0 + (jd - self.timeline_min) / (self.timeline_max - self.timeline_min) * (x1 - x0)

    def draw_timeline(self, img: Image.Image):
        x0, y0, x1, y1 = int(W * 0.08), int(H * 0.22), int(W * 0.92), int(H * 0.70)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0, y0, x1, y1), radius=28 if not QUICK_MODE else 14,
                             fill=(2, 6, 14, 176), outline=(88, 185, 220, 78), width=2)
        img.alpha_composite(overlay)
        draw_text(img, "TOP ENCOUNTER DATES VS SPEED", (x0 + 18, y0 + 18), size=18 if not QUICK_MODE else 8,
                  fill=COLORS["cyan"] + (220,), bold=True, stroke=1)
        d = ImageDraw.Draw(img)
        plot_x0, plot_y0, plot_x1, plot_y1 = x0 + 40, y0 + 70, x1 - 28, y1 - 48
        d.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=COLORS["muted"] + (160,), width=2)
        d.line((plot_x0, plot_y0, plot_x0, plot_y1), fill=COLORS["muted"] + (160,), width=2)

        for frac in [0.25, 0.5, 0.75, 1.0]:
            y = plot_y1 - frac * (plot_y1 - plot_y0)
            d.line((plot_x0, y, plot_x1, y), fill=(80, 120, 145, 50), width=1)
            draw_text(img, f"{self.top_speed * frac:.0f}", (plot_x0 - 10, int(y)), size=14 if not QUICK_MODE else 7,
                      fill=COLORS["muted"] + (210,), anchor="ra", stroke=1)

        if self.timeline:
            years = []
            for rec in self.timeline:
                dt = pd.to_datetime(rec.close_approach_cd, utc=True, errors="coerce")
                years.append(int(dt.year) if not pd.isna(dt) else None)
            min_year = min(y for y in years if y is not None)
            max_year = max(y for y in years if y is not None)
            for year in np.linspace(min_year, max_year, 5):
                year = int(round(year))
                dt = datetime(year, 1, 1, tzinfo=timezone.utc)
                jd = dt.timestamp() / 86400.0 + 2440587.5
                x = self.time_x(jd, plot_x0, plot_x1)
                d.line((x, plot_y0, x, plot_y1), fill=(80, 120, 145, 50), width=1)
                draw_text(img, str(year), (int(x), plot_y1 + 14), size=14 if not QUICK_MODE else 7,
                          fill=COLORS["muted"] + (210,), anchor="ma", stroke=1)

        for idx, rec in enumerate(self.timeline):
            x = self.time_x(rec.jd, plot_x0, plot_x1)
            y = plot_y1 - clamp(rec.v_rel_km_s / max(self.top_speed, 1e-6)) * (plot_y1 - plot_y0)
            color = COLORS["gold"] if idx == 0 else COLORS["cyan"]
            r = 7 * SCALE if idx < 3 else 5 * SCALE
            d.ellipse((x-r, y-r, x+r, y+r), fill=color + (240,), outline=(255, 255, 255, 180), width=1)
            if idx < 5:
                draw_text(img, clip_text(rec.designation, 14), (int(x), int(y - 18 * SCALE)), size=14 if not QUICK_MODE else 7,
                          fill=COLORS["white"] + (225,), anchor="ma", stroke=1)

    def draw_compare(self, img: Image.Image):
        x0 = int(W * 0.08)
        y0 = int(H * 0.23)
        w = int(W * 0.84)
        labels = [
            ("Rifle bullet", 1.0, COLORS["muted"]),
            ("ISS in low orbit", 7.66, COLORS["green"]),
            ("Earth escape speed", 11.2, COLORS["violet"]),
            ("Earth around Sun", 29.78, COLORS["orange"]),
            (clip_text(self.top_n[0].designation if self.top_n else "Fastest asteroid", 18), self.top_n[0].v_rel_km_s if self.top_n else 0.0, COLORS["gold"]),
        ]
        maxv = max(v for _, v, _ in labels) if labels else 1.0
        row_h = int(H * 0.10)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0, y0 - 24, x0 + w, y0 + row_h * len(labels) + 20), radius=26 if not QUICK_MODE else 13,
                             fill=(3, 8, 17, 180), outline=(90, 180, 210, 70), width=1)
        img.alpha_composite(overlay)
        draw_text(img, "SPEED COMPARISON", (x0 + 20, y0 - 10), size=22 if not QUICK_MODE else 10,
                  fill=COLORS["cyan"] + (230,), bold=True, stroke=1)
        for idx, (label, value, color) in enumerate(labels):
            y = y0 + idx * row_h
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((x0 + 20, y + 24, x0 + w - 20, y + 56), radius=12 if not QUICK_MODE else 6,
                                fill=(20, 35, 48, 90))
            bw = (w - 40) * clamp(value / maxv)
            d.rounded_rectangle((x0 + 20, y + 24, x0 + 20 + bw, y + 56), radius=12 if not QUICK_MODE else 6,
                                fill=color + (220,))
            draw_text(img, label, (x0 + 22, y + 2), size=19 if not QUICK_MODE else 8,
                      fill=COLORS["white"] + (235,), bold=True, stroke=1)
            draw_text(img, f"{value:.2f} km/s", (x0 + w - 22, y + 2), size=19 if not QUICK_MODE else 8,
                      fill=COLORS["muted"] + (225,), bold=True, anchor="ra", stroke=1)

    def draw_outro(self, img: Image.Image):
        x0 = int(W * 0.08)
        y0 = int(H * 0.25)
        w = int(W * 0.84)
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((x0, y0, x0 + w, y0 + int(H * 0.38)), radius=28 if not QUICK_MODE else 14,
                             fill=(2, 6, 14, 176), outline=(88, 185, 220, 78), width=2)
        img.alpha_composite(overlay)
        top = self.top_n[0] if self.top_n else None
        draw_text(img, "FASTEST IN THIS SNAPSHOT", (W // 2, y0 + 42), size=26 if not QUICK_MODE else 12,
                  fill=COLORS["white"] + (240,), bold=True, anchor="ma", stroke=1)
        if top:
            draw_text(img, clip_text(top.fullname, 32), (W // 2, y0 + 95), size=36 if not QUICK_MODE else 16,
                      fill=COLORS["gold"] + (245,), bold=True, anchor="ma", stroke=1)
            draw_text(img, f"{top.v_rel_km_s:.1f} km/s • {clip_text(top.close_approach_cd, 24)}", (W // 2, y0 + 145), size=22 if not QUICK_MODE else 10,
                      fill=COLORS["cyan"] + (230,), bold=True, anchor="ma", stroke=1)
        draw_wrapped_text(img, "Refresh the JPL CAD data later and the rankings may change as new discoveries, future flybys, and updated orbit solutions enter the database.",
                          (x0 + 24, y0 + 210), w - 48, size=20 if not QUICK_MODE else 9, fill=COLORS["muted"] + (225,))

    def render_frame(self, t: float) -> np.ndarray:
        img = self.background(t)
        self.draw_title(img, t)
        self.draw_source_hud(img)
        shot = get_shot(t)["name"]

        if shot == "intro":
            self.draw_intro(img, t)
        elif shot == "leaderboard":
            self.draw_leaderboard(img, t)
        elif shot == "cards":
            self.draw_cards(img)
        elif shot == "timeline":
            self.draw_timeline(img)
        elif shot == "compare":
            self.draw_compare(img)
        else:
            self.draw_outro(img)

        self.draw_caption(img, t)
        self.draw_hud_noise(img, t)

        arr = np.array(img.convert("RGB"))
        graded = Image.fromarray(arr)
        graded = ImageEnhance.Contrast(graded).enhance(1.08)
        graded = ImageEnhance.Color(graded).enhance(1.06)
        arr = np.array(graded)
        arr = np.clip(arr.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / 0.9)
        fade_out = 1 - smoothstep((t - (CONFIG["duration_s"] - 1.1)) / 1.0)
        return np.clip(arr.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)


# =============================================================================
# Output
# =============================================================================


def render_video(scene: FastAsteroidsScene) -> Path:
    raw_path = OUTPUT_ROOT / f"{CONFIG['basename']}_raw.mp4"
    final_path = OUTPUT_ROOT / f"{CONFIG['basename']}_final.mp4"
    write_srt(OUTPUT_ROOT / f"{CONFIG['basename']}.srt")
    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    with iio.get_writer(raw_path, fps=CONFIG["fps"], codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None) as writer:
        for frame_index in tqdm(range(frame_count), desc="Rendering fast asteroids short"):
            writer.append_data(scene.render_frame(frame_index / CONFIG["fps"]))
    shutil.copyfile(raw_path, final_path)
    return final_path



def make_contact_sheet(paths: Sequence[Path], out_path: Path):
    """todo : """ 
    print("Source status:", summary)


if __name__ == "__main__":
    main()
