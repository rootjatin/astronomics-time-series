
from __future__ import annotations

"""
What If the Sun Suddenly Lost 10% of Its Mass?
===============================================
output : https://www.youtube.com/shorts/Vu7qTwLr4bE
A cinematic vertical YouTube Short renderer for a controlled orbital-physics
thought experiment: the Sun instantaneously and isotropically loses 10% of its
mass, without the removed mass/energy striking the planets.

Scientific framing
------------------
- This cannot happen to the real Sun through ordinary stellar evolution. The
  instantaneous mass loss is a deliberately artificial setup.
- Changes in the gravitational field propagate at the speed of light. At Earth,
  the gravitational change would therefore arrive after about the same ~8.3
  minutes as a change in sunlight from the Sun.
- At that moment Earth keeps essentially the same instantaneous position and
  tangential velocity it had before the loss, but the Sun's gravitational
  parameter GM has fallen to 90% of its previous value.
- For an initially circular 1 AU orbit and an instantaneous isotropic 10% mass
  loss, the post-loss two-body orbit is still bound. In the idealized model:
      perihelion = 1.000 AU
      semimajor axis = 1.125 AU
      eccentricity = 1/9 ≈ 0.1111
      aphelion = 1.250 AU
      orbital period ≈ 1.258 old years ≈ 459.4 days
- The animation treats Earth as initially circular to make the derivation clear.
  Real planetary orbits are already eccentric and mutually interacting.
- The gravitational result does NOT specify what the Sun's luminosity would do.
  If luminosity were somehow unchanged, irradiance would follow the inverse-
  square law and fall to 64% of today's 1-AU value at 1.25 AU. A real mechanism
  capable of removing 10% of the Sun instantly would not preserve normal solar
  structure, so climate claims are intentionally avoided.
- The same fractional mass loss would alter every planetary orbit. Exact Solar
  System outcomes require N-body dynamics and depend on the timing and geometry
  of the impossible mass-loss event.

Primary references
------------------
NASA Science — Sun facts:
    https://science.nasa.gov/sun/facts/
NASA NSSDC — Sun fact sheet:
    https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
NASA Cosmicopia — gravity changes propagate at light speed:
    https://cosmicopia.gsfc.nasa.gov/qa_sp_gr.html
NASA/JPL — Gravity overview:
    https://gracefo.jpl.nasa.gov/news/5/what-is-gravity/
NASA Imagine the Universe — orbital gravity relation:
    https://imagine.gsfc.nasa.gov/features/yba/CygX1_mass/gravity/sun_mass.html

Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    SUN_MASS_SHORT_QUICK=1 python what_if_the_sun_suddenly_lost_10_percent_of_its_mass.py

Full 1080x1920 render
---------------------
    python what_if_the_sun_suddenly_lost_10_percent_of_its_mass.py

4K vertical render
------------------
    SUN_MASS_SHORT_4K=1 python what_if_the_sun_suddenly_lost_10_percent_of_its_mass.py
"""

import csv
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# =============================================================================
# Configuration
# =============================================================================

QUICK_MODE = os.environ.get("SUN_MASS_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("SUN_MASS_SHORT_4K", "0") == "1" and not QUICK_MODE

W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 6 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SIZE = (W, H)
SCALE = W / 1080.0

OUTPUT_ROOT = Path("what_if_the_sun_suddenly_lost_10_percent_of_its_mass_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "WHAT IF THE SUN SUDDENLY LOST 10% OF ITS MASS?",
    "subtitle": "An instantaneous mass-loss orbital thought experiment",
    "basename": "what_if_the_sun_suddenly_lost_10_percent_of_its_mass",
    "contrast": 1.10,
    "saturation": 1.06,
    "vignette": 0.27,
}

COLORS = {
    "space": (2, 5, 15),
    "space2": (9, 16, 34),
    "white": (248, 251, 255),
    "muted": (164, 192, 215),
    "cyan": (76, 226, 255),
    "blue": (72, 141, 255),
    "earth": (52, 149, 230),
    "earth_land": (102, 208, 150),
    "sun": (255, 224, 112),
    "sun_hot": (255, 250, 211),
    "orange": (255, 145, 72),
    "gold": (255, 205, 83),
    "red": (255, 83, 98),
    "green": (108, 239, 176),
    "violet": (181, 126, 255),
    "panel": (3, 9, 21),
}

SHOT_PLAN = [
    {"name": "delay", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "gravity_drop", "start": 8.0 if not QUICK_MODE else 1.8, "end": 18.0 if not QUICK_MODE else 4.0},
    {"name": "new_orbit", "start": 18.0 if not QUICK_MODE else 4.0, "end": 31.0 if not QUICK_MODE else 6.9},
    {"name": "longer_year", "start": 31.0 if not QUICK_MODE else 6.9, "end": 41.0 if not QUICK_MODE else 9.1},
    {"name": "whole_system", "start": 41.0 if not QUICK_MODE else 9.1, "end": 51.0 if not QUICK_MODE else 11.3},
    {"name": "caveat", "start": 51.0 if not QUICK_MODE else 11.3, "end": DURATION},
]

CAPTION_TEXTS = [
    "For about eight minutes, Earth would notice nothing. In general relativity, changes in gravity propagate at the speed of light — just like a change in sunlight from the Sun.",
    "Then the Sun's gravitational pull at Earth's distance would drop by ten percent. Earth would not suddenly slow down: it would keep the sideways orbital speed it already had.",
    "That speed is now too fast for a circular orbit around a 10-percent-lighter Sun. In a simple two-body model, Earth's present distance becomes perihelion, while the new orbit stretches out to about 1.25 astronomical units.",
    "The new ellipse has a semimajor axis of 1.125 AU and an eccentricity of about 0.111. Kepler's law then gives a year of about 1.258 old years — roughly 459 days.",
    "Every planet would have its orbit altered by the weaker central gravity. In this clean circular-orbit illustration, each orbit expands into a similar ellipse. The real Solar System would require a full N-body calculation.",
    "One important caveat: losing mass does not tell us what the Sun's luminosity would become. If its light output somehow stayed unchanged, sunlight at 1.25 AU would be only 64 percent of today's 1-AU level. The real stellar response is undefined in this impossible scenario.",
]

CAPTIONS = [
    (
        shot["start"] + min(0.35, 0.07 * (shot["end"] - shot["start"])),
        shot["end"] - min(0.10, 0.035 * (shot["end"] - shot["start"])),
        text,
    )
    for shot, text in zip(SHOT_PLAN, CAPTION_TEXTS)
]


# =============================================================================
# Physics model
# =============================================================================

@dataclass
class SunMassLossSnapshot:
    generated_at_utc: str
    mass_fraction_remaining: float
    initial_radius_au: float
    initial_orbit_assumption: str
    new_perihelion_au: float
    new_semimajor_axis_au: float
    new_eccentricity: float
    new_aphelion_au: float
    new_period_old_years: float
    new_period_days: float
    sunlight_at_aphelion_if_luminosity_constant: float
    earth_light_time_minutes: float
    scenario: str
    interpretation: str
    nasa_sun_url: str
    nasa_gravity_url: str
    nasa_orbit_url: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(x: float) -> float:
    t = clamp(x)
    return t * t * (3.0 - 2.0 * t)


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def local_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - shot["start"]) / max(1e-6, shot["end"] - shot["start"]))


def caption_at(t: float) -> Optional[str]:
    """Return a short, readable page of the active caption.

    Long narration is split across its original time window so only about
    15-20 words are visible at once; the full text remains in the SRT file.
    """
    for start, end, text in CAPTIONS:
        if start <= t < end:
            words = str(text).split()
            if not words:
                return None
            target_words = 18
            page_count = max(1, (len(words) + target_words - 1) // target_words)
            base = len(words) // page_count
            extra = len(words) % page_count
            progress = (t - start) / max(end - start, 1e-9)
            page = min(page_count - 1, max(0, int(progress * page_count)))
            first = page * base + min(page, extra)
            count = base + (1 if page < extra else 0)
            return " ".join(words[first:first + count])
    return None


def build_snapshot() -> SunMassLossSnapshot:
    f = 0.90
    r0 = 1.0

    # Initial circular orbit: v^2 = mu0/r0.
    # After mass loss, epsilon = v^2/2 - f*mu0/r0
    # = (1/2 - f) mu0/r0. Then a = -mu'/(2 epsilon).
    a = f * r0 / (2.0 * f - 1.0)
    rp = r0
    e = 1.0 - rp / a
    ra = a * (1.0 + e)
    period_old_years = math.sqrt(a**3 / f)
    period_days = 365.256 * period_old_years
    flux_ra = 1.0 / (ra * ra)

    return SunMassLossSnapshot(
        generated_at_utc=iso_z(utc_now()),
        mass_fraction_remaining=f,
        initial_radius_au=r0,
        initial_orbit_assumption="Earth begins on an idealized circular 1 AU orbit; mass loss is instantaneous, isotropic, and gives Earth no impulse.",
        new_perihelion_au=rp,
        new_semimajor_axis_au=a,
        new_eccentricity=e,
        new_aphelion_au=ra,
        new_period_old_years=period_old_years,
        new_period_days=period_days,
        sunlight_at_aphelion_if_luminosity_constant=flux_ra,
        earth_light_time_minutes=8.32,
        scenario="Controlled thought experiment; no claim that the real Sun can lose 10% of its mass instantaneously.",
        interpretation="Two-body calculation for orbital consequences only. Real planets are eccentric and mutually perturbing; stellar luminosity response is not specified.",
        nasa_sun_url="https://science.nasa.gov/sun/facts/",
        nasa_gravity_url="https://gracefo.jpl.nasa.gov/news/5/what-is-gravity/",
        nasa_orbit_url="https://imagine.gsfc.nasa.gov/features/yba/CygX1_mass/gravity/sun_mass.html",
    )


def orbit_xy(a: float, e: float, E: float) -> Tuple[float, float]:
    """Ellipse with the Sun at one focus; E is eccentric anomaly."""
    x = a * (math.cos(E) - e)
    y = a * math.sqrt(max(0.0, 1.0 - e * e)) * math.sin(E)
    return x, y


def save_data(snapshot: SunMassLossSnapshot) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "earth_post_mass_loss_orbit.csv"
    json_path = DATA_ROOT / "sun_mass_loss_snapshot.json"

    rows: List[Dict[str, float]] = []
    for i in range(181):
        E = 2.0 * math.pi * i / 180.0
        x, y = orbit_xy(snapshot.new_semimajor_axis_au, snapshot.new_eccentricity, E)
        r = math.hypot(x, y)
        rows.append({
            "eccentric_anomaly_deg": math.degrees(E),
            "x_au": x,
            "y_au": y,
            "distance_au": r,
            "relative_irradiance_if_luminosity_constant": 1.0 / (r * r),
        })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps({"snapshot": asdict(snapshot), "orbit_samples": rows}, indent=2),
        encoding="utf-8",
    )
    return csv_path, json_path


# =============================================================================
# Typography / utility rendering
# =============================================================================

def get_font(size: int, bold: bool = False):
    px = max(7, int(size * SCALE))
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, px)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    value: str,
    xy: Tuple[int, int],
    size: int,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    anchor: str = "la",
    stroke: int = 2,
):
    ImageDraw.Draw(image).text(
        xy,
        value,
        font=get_font(size, bold),
        fill=fill,
        anchor=anchor,
        stroke_width=max(1, int(stroke * SCALE)),
        stroke_fill=(0, 0, 0, 220),
    )


def draw_wrapped_text(
    image: Image.Image,
    value: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    spacing: int = 6,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold)
    words = value.split()
    lines: List[str] = []
    current = ""
    stroke_width = max(1, int(2 * SCALE))
    for word in words:
        candidate = word if not current else current + " " + word
        box = draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke_width)
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
        draw.text(
            (x, y), line, font=font, fill=fill,
            stroke_width=stroke_width, stroke_fill=(0, 0, 0, 220),
        )
        box = draw.textbbox((x, y), line, font=font, stroke_width=stroke_width)
        y += (box[3] - box[1]) + int(spacing * SCALE)


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path) -> Path:
    lines: List[str] = []
    for i, (start, end, value) in enumerate(CAPTIONS, start=1):
        lines.extend([str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", value, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    rr = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * rr**1.8, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(W, H, float(CONFIG["vignette"]))


# =============================================================================
# Scene renderer
# =============================================================================

class SunMassLossScene:
    def __init__(self, snapshot: SunMassLossSnapshot):
        self.snapshot = snapshot
        rng = np.random.default_rng(100917)
        self.stars = [
            (
                float(rng.uniform(0, W)), float(rng.uniform(0, H)),
                float(rng.uniform(0.35, 2.0) * SCALE),
                int(rng.uniform(18, 100)), float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(170 if QUICK_MODE else 520)
        ]

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 178):
        layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle(
            box,
            radius=max(8, int(24 * SCALE)),
            fill=COLORS["panel"] + (alpha,),
            outline=COLORS["cyan"] + (44,),
            width=max(1, int(2 * SCALE)),
        )
        image.alpha_composite(layer)

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        arr[..., 0] = np.clip(2 + yy * 7, 0, 255)
        arr[..., 1] = np.clip(6 + yy * 15, 0, 255)
        arr[..., 2] = np.clip(18 + yy * 30, 0, 255)
        image = Image.fromarray(arr, "RGB").convert("RGBA")
        d = ImageDraw.Draw(image)
        for x, y, r, a0, phase in self.stars:
            a = int(a0 * (0.70 + 0.30 * math.sin(t * 0.85 + phase)))
            d.ellipse((x-r, y-r, x+r, y+r), fill=COLORS["white"] + (max(3, a),))
        return image

    def draw_sun(self, image: Image.Image, center: Tuple[int, int], radius: int, t: float, mass_frac: float = 1.0):
        cx, cy = center
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for k, alpha in [(2.2, 16), (1.65, 28), (1.25, 48)]:
            rr = int(radius * k)
            gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=COLORS["gold"] + (alpha,))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4, int(20*SCALE)))))
        d = ImageDraw.Draw(image)
        d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=COLORS["sun"] + (255,), outline=COLORS["sun_hot"] + (220,), width=max(1, int(3*SCALE)))
        for i in range(8):
            a = i * math.tau / 8 + t * 0.08
            rr = radius * (0.45 + 0.12 * math.sin(t * 0.6 + i))
            x = cx + math.cos(a) * rr
            y = cy + math.sin(a) * rr
            q = max(2, int(radius * 0.10))
            d.ellipse((x-q, y-q, x+q, y+q), fill=COLORS["orange"] + (45,))
        if mass_frac < 0.999:
            draw_text(image, f"{mass_frac*100:.0f}% MASS", (cx, cy+radius+int(38*SCALE)), 13, COLORS["orange"]+(235,), True, "ma", 1)

    def draw_earth(self, image: Image.Image, center: Tuple[int, int], radius: int):
        cx, cy = center
        d = ImageDraw.Draw(image)
        d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=COLORS["earth"] + (255,), outline=COLORS["cyan"] + (170,), width=max(1, int(2*SCALE)))
        d.arc((cx-radius*.75, cy-radius*.50, cx+radius*.40, cy+radius*.45), 205, 355, fill=COLORS["earth_land"]+(220,), width=max(2, int(radius*.24)))
        d.arc((cx-radius*.28, cy-radius*.85, cx+radius*.72, cy+radius*.35), 30, 150, fill=COLORS["earth_land"]+(180,), width=max(2, int(radius*.18)))

    def draw_title(self, image: Image.Image, t: float):
        if t >= (5.8 if not QUICK_MODE else 1.30):
            return
        fade = smoothstep(t / (0.8 if not QUICK_MODE else 0.18))
        a = int(250 * fade)
        draw_text(image, "WHAT IF THE SUN SUDDENLY", (W//2, int(H*.060)), 34, COLORS["white"]+(a,), True, "ma", 2)
        draw_text(image, "LOST 10% OF ITS MASS?", (W//2, int(H*.108)), 43, COLORS["gold"]+(a,), True, "ma", 2)
        draw_text(image, CONFIG["subtitle"], (W//2, int(H*.154)), 16, COLORS["cyan"]+(min(a,230),), True, "ma", 1)

    def draw_source_hud(self, image: Image.Image):
        draw_text(image, "ORBIT MODEL // NASA CONSTANTS + NEWTON/KEPLER", (W-int(34*SCALE), int(42*SCALE)), 11, COLORS["cyan"]+(205,), True, "ra", 1)
        draw_text(image, "CONTROLLED HYPOTHETICAL", (W-int(34*SCALE), int(67*SCALE)), 10, COLORS["muted"]+(190,), False, "ra", 1)

    def draw_caption(self, image: Image.Image, t: float):
        value = caption_at(t)
        if not value:
            return
        y0 = H - int(188*SCALE)
        layer = Image.new("RGBA", SIZE, (0,0,0,0))
        d = ImageDraw.Draw(layer)
        margin = int(58*SCALE)
        box_h = int(116*SCALE)
        d.rounded_rectangle(
            (margin, y0, W-margin, y0+box_h), radius=max(8,int(20*SCALE)),
            fill=(2,7,17,150), outline=COLORS["cyan"]+(48,), width=max(1,int(2*SCALE)),
        )
        image.alpha_composite(layer)
        draw_wrapped_text(
            image, value, (margin+int(24*SCALE), y0+int(18*SCALE)),
            W-2*margin-int(48*SCALE), 30,
            COLORS["white"]+(240,), False, 4,
        )

    def draw_delay(self, image: Image.Image, t: float, p: float):
        cx = int(W*.50); cy = int(H*.43)
        self.draw_sun(image, (cx, cy), int(116*SCALE), t)
        d = ImageDraw.Draw(image)
        er = int(19*SCALE)
        earth_x = int(W*.84); earth_y = cy
        self.draw_earth(image, (earth_x, earth_y), er)
        # expanding information wave toward Earth
        max_r = earth_x - cx
        wave_r = max_r * smoothstep(p)
        d.ellipse((cx-wave_r, cy-wave_r, cx+wave_r, cy+wave_r), outline=COLORS["cyan"]+(150,), width=max(1,int(4*SCALE)))
        d.line((cx+int(120*SCALE),cy,earth_x-int(24*SCALE),earth_y),fill=COLORS["white"]+(50,),width=max(1,int(2*SCALE)))
        draw_text(image,"FOR ~8 MINUTES… NOTHING CHANGES HERE",(W//2,int(H*.205)),28,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"gravity changes do not propagate instantly",(W//2,int(H*.255)),15,COLORS["cyan"]+(230,),False,"ma",1)
        sec = min(8.32, 8.32*p)
        draw_text(image,f"SIGNAL TRAVEL: {sec:04.2f} min",(W//2,int(H*.66)),19,COLORS["gold"]+(240,),True,"ma",1)

    def draw_gravity_drop(self, image: Image.Image, t: float, p: float):
        self.draw_sun(image,(int(W*.28),int(H*.43)),int(92*SCALE),t,0.90)
        self.draw_earth(image,(int(W*.78),int(H*.43)),int(25*SCALE))
        d=ImageDraw.Draw(image)
        x0=int(W*.72); y=int(H*.43)
        old_len=int(W*.32); new_len=int(old_len*.90)
        # arrows point toward Sun
        d.line((x0,y-int(42*SCALE),x0-old_len,y-int(42*SCALE)),fill=COLORS["muted"]+(110,),width=max(2,int(5*SCALE)))
        d.polygon([(x0-old_len,y-int(42*SCALE)),(x0-old_len+int(20*SCALE),y-int(54*SCALE)),(x0-old_len+int(20*SCALE),y-int(30*SCALE))],fill=COLORS["muted"]+(130,))
        d.line((x0,y+int(42*SCALE),x0-new_len,y+int(42*SCALE)),fill=COLORS["red"]+(230,),width=max(2,int(7*SCALE)))
        d.polygon([(x0-new_len,y+int(42*SCALE)),(x0-new_len+int(22*SCALE),y+int(28*SCALE)),(x0-new_len+int(22*SCALE),y+int(56*SCALE))],fill=COLORS["red"]+(235,))
        draw_text(image,"GRAVITY AT 1 AU DROPS 10%",(W//2,int(H*.19)),31,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"OLD PULL 100%",(W//2,int(H*.54)),14,COLORS["muted"]+(210,),True,"ma",1)
        draw_text(image,"NEW PULL 90%",(W//2,int(H*.60)),20,COLORS["red"]+(240,),True,"ma",1)
        draw_text(image,"Earth keeps its instantaneous sideways speed",(W//2,int(H*.69)),16,COLORS["cyan"]+(230,),False,"ma",1)

    def draw_new_orbit(self, image: Image.Image, t: float, p: float):
        cx=int(W*.46); cy=int(H*.45)
        scale=int(W*.285)
        d=ImageDraw.Draw(image)
        # old circular orbit, radius=1 AU
        d.ellipse((cx-scale,cy-scale*.36,cx+scale,cy+scale*.36),outline=COLORS["muted"]+(75,),width=max(1,int(3*SCALE)))
        # new ellipse with Sun at focus. Map physical x,y to screen.
        a=self.snapshot.new_semimajor_axis_au; e=self.snapshot.new_eccentricity
        pts=[]
        for k in range(241):
            E=2*math.pi*k/240
            x,y=orbit_xy(a,e,E)
            pts.append((cx + x*scale, cy + y*scale*.36))
        reveal=max(2,int(len(pts)*smoothstep(p)))
        d.line(pts[:reveal],fill=COLORS["cyan"]+(235,),width=max(2,int(5*SCALE)),joint="curve")
        sun_x=cx; sun_y=cy
        self.draw_sun(image,(sun_x,sun_y),int(47*SCALE),t,0.90)
        # start is perihelion at x=+1 AU in this parameterization (E=0 => a(1-e)=1)
        ex=cx+scale; ey=cy
        self.draw_earth(image,(int(ex),int(ey)),int(15*SCALE))
        d.line((cx,cy-int(15*SCALE),cx+scale,cy-int(15*SCALE)),fill=COLORS["gold"]+(100,),width=max(1,int(2*SCALE)))
        draw_text(image,"1.00 AU",(cx+scale*.50,cy-int(28*SCALE)),11,COLORS["gold"]+(225,),True,"ma",1)
        draw_text(image,"EARTH MOVES ONTO A LARGER ELLIPSE",(W//2,int(H*.17)),28,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"perihelion 1.00 AU  •  aphelion 1.25 AU",(W//2,int(H*.225)),16,COLORS["cyan"]+(235,),True,"ma",1)
        self.panel(image,(int(W*.12),int(H*.65),int(W*.88),int(H*.78)),165)
        draw_text(image,"a = 1.125 AU",(int(W*.20),int(H*.69)),18,COLORS["white"]+(235,),True,"la",1)
        draw_text(image,"e = 0.111",(int(W*.20),int(H*.735)),18,COLORS["white"]+(235,),True,"la",1)
        draw_text(image,"still bound",(int(W*.80),int(H*.714)),17,COLORS["green"]+(235,),True,"ra",1)

    def draw_longer_year(self, image: Image.Image, t: float, p: float):
        cx=W//2; cy=int(H*.43); r=int(205*SCALE)
        d=ImageDraw.Draw(image)
        d.ellipse((cx-r,cy-r,cx+r,cy+r),outline=COLORS["white"]+(70,),width=max(2,int(4*SCALE)))
        d.arc((cx-r,cy-r,cx+r,cy+r),-90,-90+360*clamp(p),fill=COLORS["cyan"]+(235,),width=max(4,int(15*SCALE)))
        # hand
        ang=math.radians(-90+360*p)
        x=cx+math.cos(ang)*r*.72; y=cy+math.sin(ang)*r*.72
        d.line((cx,cy,x,y),fill=COLORS["gold"]+(230,),width=max(2,int(5*SCALE)))
        d.ellipse((cx-int(7*SCALE),cy-int(7*SCALE),cx+int(7*SCALE),cy+int(7*SCALE)),fill=COLORS["white"]+(240,))
        draw_text(image,"THE YEAR GETS LONGER",(W//2,int(H*.17)),31,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"T ≈ 1.258 old years",(W//2,int(H*.69)),24,COLORS["cyan"]+(240,),True,"ma",1)
        draw_text(image,"≈ 459.4 days",(W//2,int(H*.74)),31,COLORS["gold"]+(245,),True,"ma",2)
        draw_text(image,"Kepler: T² ∝ a³ / M",(W//2,int(H*.79)),13,COLORS["muted"]+(210,),False,"ma",1)

    def draw_whole_system(self, image: Image.Image, t: float, p: float):
        cx=W//2; cy=int(H*.44)
        self.draw_sun(image,(cx,cy),int(42*SCALE),t,0.90)
        d=ImageDraw.Draw(image)
        # Use compressed visual radii, not physical scale.
        planets=[
            ("MERCURY",.18,COLORS["muted"]),("VENUS",.27,COLORS["gold"]),("EARTH",.37,COLORS["earth"]),
            ("MARS",.47,COLORS["orange"]),("JUPITER",.62,COLORS["gold"]),("SATURN",.76,COLORS["violet"]),
        ]
        maxr=min(W*.43,H*.27)
        for i,(name,frac,col) in enumerate(planets):
            old=maxr*frac
            # new ellipse: perihelion old radius, aphelion 1.25 old radius; center shifted outward from focus
            a_screen=old*1.125
            b_screen=a_screen*math.sqrt(1-(1/9)**2)*.34
            shift=old*.125
            box=(cx-a_screen-shift,cy-b_screen,cx+a_screen-shift,cy+b_screen)
            d.ellipse(box,outline=col+(45,),width=max(1,int(2*SCALE)))
            ang=t*(.44-.045*i)+i*.8
            # visual only, based on ellipse centered at cx-shift
            px=(cx-shift)+math.cos(ang)*a_screen
            py=cy+math.sin(ang)*b_screen
            q=max(2,int((4+0.7*i)*SCALE))
            d.ellipse((px-q,py-q,px+q,py+q),fill=col+(230,))
        draw_text(image,"THE WHOLE SOLAR SYSTEM RE-ADJUSTS",(W//2,int(H*.16)),29,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"weaker central gravity changes every orbit",(W//2,int(H*.215)),16,COLORS["cyan"]+(230,),False,"ma",1)
        draw_text(image,"SCHEMATIC — ORBITS NOT TO SCALE",(W//2,int(H*.74)),12,COLORS["red"]+(215,),True,"ma",1)
        draw_text(image,"real outcome → N-body dynamics",(W//2,int(H*.785)),15,COLORS["muted"]+(220,),False,"ma",1)

    def draw_caveat(self, image: Image.Image, t: float, p: float):
        # Split: gravity is defined by the setup, luminosity is not.
        self.panel(image,(int(W*.08),int(H*.23),int(W*.47),int(H*.65)),180)
        self.panel(image,(int(W*.53),int(H*.23),int(W*.92),int(H*.65)),180)
        draw_text(image,"GRAVITY",(int(W*.275),int(H*.29)),22,COLORS["cyan"]+(240,),True,"ma",1)
        draw_text(image,"DEFINED",(int(W*.275),int(H*.35)),28,COLORS["green"]+(245,),True,"ma",2)
        draw_text(image,"M → 0.90 M☉",(int(W*.275),int(H*.44)),18,COLORS["white"]+(235,),True,"ma",1)
        draw_text(image,"orbit can be",(int(W*.275),int(H*.52)),13,COLORS["muted"]+(215,),False,"ma",1)
        draw_text(image,"calculated",(int(W*.275),int(H*.56)),18,COLORS["white"]+(235,),True,"ma",1)

        draw_text(image,"LUMINOSITY",(int(W*.725),int(H*.29)),22,COLORS["gold"]+(240,),True,"ma",1)
        draw_text(image,"NOT DEFINED",(int(W*.725),int(H*.35)),26,COLORS["red"]+(245,),True,"ma",2)
        draw_text(image,"depends on impossible",(int(W*.725),int(H*.47)),12,COLORS["muted"]+(215,),False,"ma",1)
        draw_text(image,"mass-loss mechanism",(int(W*.725),int(H*.51)),15,COLORS["white"]+(235,),True,"ma",1)
        draw_text(image,"NO SIMPLE CLIMATE CLAIM",(int(W*.725),int(H*.59)),11,COLORS["red"]+(220,),True,"ma",1)

        a=int(245*smoothstep((p-.10)/.55))
        draw_text(image,"IF luminosity somehow stayed constant:",(W//2,int(H*.72)),15,COLORS["white"]+(a,),True,"ma",1)
        draw_text(image,"sunlight at 1.25 AU = 64% of today's 1-AU value",(W//2,int(H*.765)),18,COLORS["gold"]+(a,),True,"ma",1)
        draw_text(image,"inverse-square law only — not a climate forecast",(W//2,int(H*.805)),11,COLORS["muted"]+(a,),False,"ma",1)

    def draw_scanlines(self, image: Image.Image, t: float):
        layer=Image.new("RGBA",SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        step=max(5,int(8*SCALE)); off=int((t*31)%step)
        for y in range(off,H,step):
            d.line((0,y,W,y),fill=(135,185,255,7),width=1)
        image.alpha_composite(layer)

    def render(self, t: float) -> Image.Image:
        image=self.background(t)
        shot=get_shot(t); p=local_progress(t,shot)
        name=shot["name"]
        if name=="delay": self.draw_delay(image,t,p)
        elif name=="gravity_drop": self.draw_gravity_drop(image,t,p)
        elif name=="new_orbit": self.draw_new_orbit(image,t,p)
        elif name=="longer_year": self.draw_longer_year(image,t,p)
        elif name=="whole_system": self.draw_whole_system(image,t,p)
        else: self.draw_caveat(image,t,p)
        self.draw_title(image,t)
        self.draw_source_hud(image)
        self.draw_caption(image,t)
        self.draw_scanlines(image,t)
        arr=np.asarray(image.convert("RGB"),dtype=np.float32)
        arr*=VIGNETTE[...,None]
        arr=np.clip(arr,0,255).astype(np.uint8)
        graded=Image.fromarray(arr,"RGB")
        graded=ImageEnhance.Contrast(graded).enhance(float(CONFIG["contrast"]))
        graded=ImageEnhance.Color(graded).enhance(float(CONFIG["saturation"]))
        return graded


# =============================================================================
# Output
# =============================================================================

def render_preview_frames(scene: SunMassLossScene) -> List[Path]:
    paths=[]
    for i,shot in enumerate(SHOT_PLAN,1):
        t=shot["start"]+.52*(shot["end"]-shot["start"])
        image=scene.render(min(t,DURATION-.001))
        path=PREVIEW_ROOT/f"preview_{i:02d}_{shot['name']}.jpg"
        image.save(path,quality=92)
        paths.append(path)
    return paths


def make_contact_sheet(paths: Sequence[Path]) -> Path:
    images=[Image.open(p).convert("RGB") for p in paths]
    thumb_w=150 if QUICK_MODE else int(300*SCALE)
    thumb_h=267 if QUICK_MODE else int(533*SCALE)
    cols=3; rows=math.ceil(len(images)/cols); margin=max(12,int(18*SCALE))
    sheet=Image.new("RGB",(cols*thumb_w+(cols+1)*margin,rows*thumb_h+(rows+1)*margin),(5,8,16))
    for i,img in enumerate(images):
        thumb=img.copy(); thumb.thumbnail((thumb_w,thumb_h),Image.Resampling.LANCZOS)
        x=margin+(i%cols)*(thumb_w+margin); y=margin+(i//cols)*(thumb_h+margin)
        sheet.paste(thumb,(x,y))
    path=PREVIEW_ROOT/f"{CONFIG['basename']}_contact_sheet.jpg"
    sheet.save(path,quality=92)
    return path


def render_video(scene: SunMassLossScene) -> Path:
    path=OUTPUT_ROOT/f"{CONFIG['basename']}.mp4"
    total=max(1,int(round(DURATION*FPS)))
    writer=iio.get_writer(path,fps=FPS,codec="libx264",quality=7,pixelformat="yuv420p",ffmpeg_log_level="error",macro_block_size=2)
    try:
        for i in tqdm(range(total),desc="Rendering Sun Mass-Loss Short"):
            writer.append_data(np.asarray(scene.render(i/FPS).convert("RGB")))
    finally:
        writer.close()
    return path


def copy_flat(video: Path, sheet: Path, srt: Path) -> Tuple[Path,Path,Path]:
    flat_video=Path(f"{CONFIG['basename']}_quick_preview.mp4") if QUICK_MODE else Path(f"{CONFIG['basename']}.mp4")
    flat_sheet=Path(f"{CONFIG['basename']}_contact_sheet.jpg")
    flat_srt=Path(f"{CONFIG['basename']}_subtitles.srt")
    shutil.copy2(video,flat_video); shutil.copy2(sheet,flat_sheet); shutil.copy2(srt,flat_srt)
    return flat_video,flat_sheet,flat_srt



# =============================================================================
# YouTube title / description sidecar
# =============================================================================

YOUTUBE_TITLE = 'What If the Sun Suddenly Lost 10% of Its Mass? ☀️ #rootjatin'
YOUTUBE_DESCRIPTION = "A hypothetical orbital-mechanics experiment asks what happens if the Sun instantly loses ten percent of its mass while Earth keeps its existing orbital velocity. In the simplified two-body model, Earth's current circular orbit becomes a larger ellipse and the year changes. The mass-loss event itself is not a realistic solar process, and the Sun's luminosity response and the full Solar System require more detailed physics than this illustration models."
YOUTUBE_HASHTAGS = '#rootjatin #Sun #Earth #WhatIf #Orbit #Physics #Astronomy #Space'

def write_youtube_metadata_txt() -> Path:
    cfg = globals().get("CONFIG") or globals().get("CFG") or {}
    basename = (
        cfg.get("basename")
        or cfg.get("output_basename")
        or cfg.get("file_stem")
        or Path(__file__).stem
    )
    root = globals().get("OUTPUT_ROOT") or globals().get("ROOT") or Path(".")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{basename}_title_description.txt"
    path.write_text(
        "TITLE\n" + YOUTUBE_TITLE +
        "\n\nDESCRIPTION\n" + YOUTUBE_DESCRIPTION +
        "\n\nHASHTAGS\n" + YOUTUBE_HASHTAGS + "\n",
        encoding="utf-8",
    )
    return path

def main():
    snapshot=build_snapshot()
    csv_path,json_path=save_data(snapshot)
    srt_path=write_srt(OUTPUT_ROOT/f"{CONFIG['basename']}_subtitles.srt")
    scene=SunMassLossScene(snapshot)
    previews=render_preview_frames(scene)
    sheet=make_contact_sheet(previews)
    video=render_video(scene)
    flat_video,flat_sheet,flat_srt=copy_flat(video,sheet,srt_path)
    print("\nRender complete")
    print("="*76)
    print(CONFIG["title"])
    print(f"Mode: {'QUICK' if QUICK_MODE else ('4K' if FOUR_K else 'FULL HD')}")
    print(f"Frame: {W}x{H} @ {FPS} fps // {DURATION:.1f} s")
    print(f"Post-loss Earth orbit: rp={snapshot.new_perihelion_au:.3f} AU, a={snapshot.new_semimajor_axis_au:.3f} AU, e={snapshot.new_eccentricity:.4f}, ra={snapshot.new_aphelion_au:.3f} AU")
    print(f"New orbital period: {snapshot.new_period_old_years:.4f} old years = {snapshot.new_period_days:.1f} days")
    print(f"Irradiance at 1.25 AU if luminosity unchanged: {snapshot.sunlight_at_aphelion_if_luminosity_constant*100:.1f}%")
    for label,path in [("video",video),("subtitles",srt_path),("csv",csv_path),("json",json_path),("contact",sheet),("flat_video",flat_video),("flat_sheet",flat_sheet),("flat_srt",flat_srt)]:
        print(f"{label:12s} {path}")
    metadata_txt = write_youtube_metadata_txt()
    print("Title/description TXT:", metadata_txt.resolve())


if __name__ == "__main__":
    main()
