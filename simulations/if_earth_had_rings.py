from __future__ import annotations

"""
What If Earth Had Rings Like Saturn?
====================================

output:https://www.youtube.com/shorts/rRIyUfL8yag

A cinematic vertical YouTube Short renderer about a hypothetical Earth with a
bright planetary ring system.

The production pattern mirrors the companion science Shorts in this series:
9:16 rendering, quick-preview mode, a multi-shot story, animated captions,
preview frames, a contact sheet, CSV/JSON exports, subtitles, and H.264 MP4.

Scientific framing
------------------
- Earth does not currently have a planetary ring system.
- Planetary rings are not solid disks. They are huge populations of particles
  independently orbiting a planet.
- NASA describes Saturn's main rings as mostly water-ice particles, with the
  main ring system enormously wide but very thin compared with its diameter.
  The hypothetical Earth rings here are inspired by that geometry, but their
  composition, brightness, width, and optical depth are deliberately fictional.
- Inner planetary rings and nearby moons generally orbit close to a planet's
  equatorial plane. This renderer therefore assumes a long-lived equatorial
  ring system around Earth.
- Because the ring plane is fixed to Earth's equator, its apparent shape in the
  sky would depend strongly on observer latitude: nearly edge-on at the equator,
  a broad arch at mid-latitudes, and lower toward the horizon at high latitudes.
- Dense rings can cast shadows on their planet. A sufficiently broad/opaque
  terrestrial ring system could modify incoming sunlight regionally and
  seasonally, but the climate effect depends on ring properties and requires a
  real radiative-climate model. This video does not claim a numerical climate
  forecast.
- Ring brightness at night would depend on particle reflectivity, optical depth,
  solar illumination, season, and viewing geometry. The night-sky scene is an
  illustrative visualization, not a photometric prediction.
- Roche-limit graphics are conceptual. NASA notes that for a planet and orbiting
  body of equal density, the classical Roche-limit scale is about 2.5 planetary
  radii; the exact value changes with density and material behavior.

Primary sources for the framing
-------------------------------
NASA Science — Facts About Earth:
    https://science.nasa.gov/earth/facts/
NASA Science — Saturn's Rings:
    https://science.nasa.gov/resource/saturns-rings-2/
NASA Science — Cassini FAQ (rings, equatorial plane, Roche limit):
    https://science.nasa.gov/mission/cassini/faq/
NASA Science — Epimetheus Above the Rings:
    https://science.nasa.gov/resource/epimetheus-above-the-rings/
NASA NSSDC — Earth Fact Sheet:
    https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html

Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Full render
-----------
    python what_if_earth_had_rings_like_saturn.py

Quick preview
-------------
    EARTH_RINGS_SHORT_QUICK=1 python what_if_earth_had_rings_like_saturn.py

4K vertical render
------------------
    EARTH_RINGS_SHORT_4K=1 python what_if_earth_had_rings_like_saturn.py
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

QUICK_MODE = os.environ.get("EARTH_RINGS_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("EARTH_RINGS_SHORT_4K", "0") == "1" and not QUICK_MODE

W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 6 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SIZE = (W, H)
SCALE = W / 1080.0

OUTPUT_ROOT = Path("what_if_earth_had_rings_like_saturn_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "WHAT IF EARTH HAD RINGS LIKE SATURN?",
    "subtitle": "A geometry-first thought experiment about a ringed Earth",
    "basename": "what_if_earth_had_rings_like_saturn",
    "width": W,
    "height": H,
    "fps": FPS,
    "duration_s": DURATION,
    "contrast": 1.09,
    "saturation": 1.07,
    "vignette": 0.25,
}

COLORS = {
    "space": (2, 5, 15),
    "space2": (8, 17, 36),
    "white": (248, 251, 255),
    "muted": (165, 190, 214),
    "cyan": (82, 229, 255),
    "blue": (70, 143, 255),
    "earth": (24, 94, 166),
    "earth2": (41, 151, 205),
    "ocean": (11, 70, 132),
    "land": (63, 144, 93),
    "land2": (112, 179, 106),
    "cloud": (236, 249, 255),
    "ring_ice": (224, 234, 244),
    "ring_warm": (223, 201, 169),
    "ring_dark": (94, 105, 123),
    "gold": (255, 207, 93),
    "orange": (255, 145, 75),
    "red": (255, 87, 104),
    "violet": (179, 126, 255),
    "green": (105, 239, 176),
    "panel": (3, 9, 21),
}

SHOT_PLAN = [
    {"name": "reveal", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "latitudes", "start": 8.0 if not QUICK_MODE else 1.8, "end": 19.0 if not QUICK_MODE else 4.2},
    {"name": "shadows", "start": 19.0 if not QUICK_MODE else 4.2, "end": 30.0 if not QUICK_MODE else 6.6},
    {"name": "night", "start": 30.0 if not QUICK_MODE else 6.6, "end": 40.5 if not QUICK_MODE else 8.95},
    {"name": "particles", "start": 40.5 if not QUICK_MODE else 8.95, "end": 51.5 if not QUICK_MODE else 11.35},
    {"name": "outro", "start": 51.5 if not QUICK_MODE else 11.35, "end": DURATION},
]

CAPTION_TEXTS = [
    "Imagine Earth wrapped in a vast equatorial ring system. It would not be a solid halo — it would be countless particles, each independently orbiting our planet.",
    "The view would depend on where you lived. Near the equator the rings would look almost edge-on overhead. At mid-latitudes they would sweep across the sky as a huge arch, while high latitudes would see them lower toward the horizon.",
    "The rings could also cast shadows onto Earth. How strong any cooling or seasonal climate effect became would depend on the rings' width, density, reflectivity, and exact geometry — so this scene is illustrative, not a climate forecast.",
    "Night would look different too. Sunlit ring particles could create an enormous luminous structure across the sky, but its actual brightness would vary with season, ring material, optical depth, and viewing angle.",
    "And the rings would not behave like one object. Each grain, rock, or icy fragment would orbit on its own. Close to a planet, tidal gravity can keep debris from assembling into a large moon — the basic idea behind the Roche-limit zone.",
    "A ringed Earth could be visually spectacular, but every frame here is a controlled thought experiment. Earth has no rings today, and the climate and brightness consequences would require a much more detailed physical model.",
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
# Data model and helpers
# =============================================================================

@dataclass
class RingedEarthSnapshot:
    generated_at_utc: str
    scenario: str
    earth_obliquity_deg: float
    earth_rotation_hours: float
    earth_has_rings_now: bool
    assumed_ring_plane: str
    roche_limit_scale_equal_density_planet_radii: float
    ring_material_reference: str
    interpretation: str
    nasa_earth_url: str
    nasa_saturn_rings_url: str
    nasa_cassini_faq_url: str
    nasa_equatorial_plane_url: str
    nssdc_earth_url: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    x = clamp(t)
    return x * x * (3.0 - 2.0 * x)


def ease_in_out_sine(t: float) -> float:
    x = clamp(t)
    return -(math.cos(math.pi * x) - 1.0) / 2.0


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


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


def local_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - shot["start"]) / max(1e-6, shot["end"] - shot["start"]))


def get_font(size: int, bold: bool = False):
    px = max(7, int(size * SCALE))
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=px)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[float, float],
    size: int,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    anchor: str = "la",
    stroke: int = 2,
):
    ImageDraw.Draw(image).text(
        (int(xy[0]), int(xy[1])),
        text,
        font=get_font(size, bold),
        fill=fill,
        anchor=anchor,
        stroke_width=max(1, int(stroke * SCALE)),
        stroke_fill=(0, 0, 0, 220),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    spacing: int = 6,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold)
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        box = draw.textbbox((0, 0), candidate, font=font, stroke_width=max(1, int(2 * SCALE)))
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
            (x, y),
            line,
            font=font,
            fill=fill,
            stroke_width=max(1, int(2 * SCALE)),
            stroke_fill=(0, 0, 0, 220),
        )
        box = draw.textbbox((x, y), line, font=font, stroke_width=max(1, int(2 * SCALE)))
        y += (box[3] - box[1]) + int(spacing * SCALE)


def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 178):
    layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(
        box,
        radius=max(8, int(24 * SCALE)),
        fill=COLORS["panel"] + (alpha,),
        outline=COLORS["cyan"] + (45,),
        width=max(1, int(2 * SCALE)),
    )
    image.alpha_composite(layer)


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
    for i, (start, end, text) in enumerate(CAPTIONS, start=1):
        lines.extend([str(i), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    rr = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * rr**1.8, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(W, H, float(CONFIG["vignette"]))


def build_snapshot() -> RingedEarthSnapshot:
    return RingedEarthSnapshot(
        generated_at_utc=iso_z(utc_now()),
        scenario="Hypothetical long-lived equatorial planetary ring system around modern Earth",
        earth_obliquity_deg=23.44,
        earth_rotation_hours=23.9345,
        earth_has_rings_now=False,
        assumed_ring_plane="equatorial",
        roche_limit_scale_equal_density_planet_radii=2.5,
        ring_material_reference="Saturn comparison: mostly water-ice particles; hypothetical Earth composition unspecified",
        interpretation="All ring dimensions, opacity, brightness, climate shading, and surface-sky views are schematic educational illustrations, not a numerical orbital, radiative, or climate simulation.",
        nasa_earth_url="https://science.nasa.gov/earth/facts/",
        nasa_saturn_rings_url="https://science.nasa.gov/resource/saturns-rings-2/",
        nasa_cassini_faq_url="https://science.nasa.gov/mission/cassini/faq/",
        nasa_equatorial_plane_url="https://science.nasa.gov/resource/epimetheus-above-the-rings/",
        nssdc_earth_url="https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html",
    )


def save_data(snapshot: RingedEarthSnapshot) -> Tuple[Path, Path]:
    json_path = DATA_ROOT / "ringed_earth_snapshot.json"
    csv_path = DATA_ROOT / "ringed_earth_story_beats.csv"
    json_path.write_text(json.dumps(asdict(snapshot), indent=2), encoding="utf-8")

    rows = [
        ("ring plane", "equatorial", "Long-lived inner planetary rings are expected close to the equatorial plane."),
        ("surface view", "latitude-dependent", "Edge-on near equator; increasingly open apparent arch away from the equator in this schematic geometry."),
        ("shadow", "possible", "Opaque ring sectors can block some sunlight; actual radiative effect needs a detailed model."),
        ("night sky", "potentially bright", "Brightness depends on material reflectivity, optical depth, season, and solar/viewing geometry."),
        ("particle behavior", "independent orbits", "A ring is many orbiting particles, not a rigid disk."),
        ("Roche zone", "conceptual ~2.5 R_E for equal densities", "Exact Roche limit depends on density and material response."),
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "illustrated_result", "interpretation"])
        writer.writerows(rows)
    return csv_path, json_path


# =============================================================================
# Scene renderer
# =============================================================================

@dataclass(frozen=True)
class Star:
    x: float
    y: float
    r: float
    alpha: int
    phase: float


@dataclass(frozen=True)
class RingParticle:
    radius_frac: float
    angle: float
    size: float
    alpha: int
    band: int


class RingedEarthScene:
    def __init__(self, snapshot: RingedEarthSnapshot):
        self.snapshot = snapshot
        rng = np.random.default_rng(20260921)
        star_count = 170 if QUICK_MODE else (800 if FOUR_K else 460)
        self.stars = [
            Star(
                float(rng.uniform(0, W)),
                float(rng.uniform(0, H)),
                float(rng.uniform(0.35, 1.8) * SCALE),
                int(rng.uniform(45, 190)),
                float(rng.uniform(0.0, math.tau)),
            )
            for _ in range(star_count)
        ]
        particle_count = 160 if QUICK_MODE else (850 if FOUR_K else 430)
        self.ring_particles = [
            RingParticle(
                float(rng.uniform(1.22, 2.18)),
                float(rng.uniform(0, math.tau)),
                float(rng.uniform(0.7, 2.6) * SCALE),
                int(rng.uniform(45, 180)),
                int(rng.integers(0, 4)),
            )
            for _ in range(particle_count)
        ]

        yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
        top = np.array(COLORS["space"], dtype=np.float32)
        bottom = np.array(COLORS["space2"], dtype=np.float32)
        rgb = top[None, :] * (1.0 - yy) + bottom[None, :] * yy
        arr = np.empty((H, W, 4), dtype=np.uint8)
        arr[:, :, :3] = np.clip(rgb[:, None, :], 0, 255).astype(np.uint8)
        arr[:, :, 3] = 255
        self.base = Image.fromarray(arr, "RGBA")

    def background(self, t: float) -> Image.Image:
        image = self.base.copy()
        d = ImageDraw.Draw(image)
        for star in self.stars:
            a = int(star.alpha * (0.72 + 0.28 * math.sin(t * 0.9 + star.phase)))
            d.ellipse(
                (star.x - star.r, star.y - star.r, star.x + star.r, star.y + star.r),
                fill=COLORS["white"] + (max(10, a),),
            )
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for cx, cy, color in [
            (W * 0.18, H * 0.28, COLORS["blue"]),
            (W * 0.82, H * 0.23, COLORS["violet"]),
            (W * 0.55, H * 0.71, COLORS["cyan"]),
        ]:
            for radius, alpha in [(W * 0.34, 8), (W * 0.20, 13), (W * 0.10, 20)]:
                gd.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=color + (alpha,))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(18, int(70 * SCALE)))))
        return image

    def draw_earth(self, image: Image.Image, center: Tuple[int, int], radius: int, t: float, alpha: int = 255):
        cx, cy = center
        layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        for extra, a in [(28, 22), (14, 42)]:
            rr = radius + int(extra * SCALE)
            d.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=COLORS["cyan"] + (a,))
        d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=COLORS["earth"] + (alpha,))

        # Stylized continents rotate slowly. They are deliberately not geographic maps.
        rot = 0.10 * t
        blobs = [
            (-0.28, -0.14, 0.34, 0.26, 0.2),
            (0.30, -0.24, 0.26, 0.20, -0.4),
            (0.21, 0.24, 0.30, 0.31, 0.7),
            (-0.36, 0.31, 0.18, 0.24, -0.7),
        ]
        for bx, by, bw, bh, phase in blobs:
            x = cx + radius * (bx + 0.10 * math.sin(rot + phase))
            y = cy + radius * by
            d.ellipse(
                (x-radius*bw, y-radius*bh, x+radius*bw, y+radius*bh),
                fill=COLORS["land"] + (min(alpha, 235),),
            )
        # cloud bands
        for k in range(5):
            yy = cy + radius * (-0.55 + k * 0.27)
            xoff = radius * 0.15 * math.sin(t * 0.18 + k)
            d.arc(
                (cx-radius*.78+xoff, yy-radius*.09, cx+radius*.78+xoff, yy+radius*.09),
                5, 175,
                fill=COLORS["cloud"] + (70,),
                width=max(1, int(3*SCALE)),
            )

        # day-night terminator approximation
        shade = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shade)
        sd.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(0,0,0,0))
        sd.ellipse((cx-radius*.24, cy-radius, cx+radius*1.08, cy+radius), fill=(0,0,12,80))
        layer.alpha_composite(shade)
        image.alpha_composite(layer)

    def ring_ellipse(self, image: Image.Image, center: Tuple[int,int], planet_r: int, tilt: float, t: float,
                     alpha: int = 255, front_only: bool = False, shadow: bool = False):
        cx, cy = center
        layer = Image.new("RGBA", SIZE, (0,0,0,0))
        d = ImageDraw.Draw(layer)
        tilt_factor = max(0.055, abs(math.sin(math.radians(tilt))))
        bands = [
            (1.27, 1.46, COLORS["ring_dark"], 95),
            (1.49, 1.69, COLORS["ring_warm"], 185),
            (1.72, 1.95, COLORS["ring_ice"], 210),
            (1.99, 2.18, COLORS["ring_warm"], 130),
        ]
        for inner, outer, color, base_a in bands:
            for frac in np.linspace(inner, outer, 8):
                rx = planet_r * frac
                ry = rx * tilt_factor
                a = int(base_a * alpha / 255)
                box=(cx-rx, cy-ry, cx+rx, cy+ry)
                if front_only:
                    d.arc(box, 0, 180, fill=color+(a,), width=max(1,int(3*SCALE)))
                else:
                    d.ellipse(box, outline=color+(a,), width=max(1,int(3*SCALE)))

        # gaps
        for frac in (1.47, 1.70, 1.97):
            rx=planet_r*frac; ry=rx*tilt_factor
            d.ellipse((cx-rx,cy-ry,cx+rx,cy+ry),outline=COLORS["space"]+(200,),width=max(1,int(6*SCALE)))

        if shadow:
            # broad shadow band on the globe, intentionally schematic
            sy=cy+int(planet_r*.12*math.sin(t*.25))
            sh=Image.new("RGBA",SIZE,(0,0,0,0)); sd=ImageDraw.Draw(sh)
            sd.ellipse((cx-planet_r*.90,sy-planet_r*.12,cx+planet_r*.90,sy+planet_r*.12),fill=(2,4,9,100))
            image.alpha_composite(sh.filter(ImageFilter.GaussianBlur(max(2,int(5*SCALE)))))

        image.alpha_composite(layer)

    def draw_ringed_earth(self, image: Image.Image, center: Tuple[int,int], radius: int, t: float,
                          ring_tilt: float = 24.0, show_shadow: bool = False):
        # back half of rings
        self.ring_ellipse(image, center, radius, ring_tilt, t, alpha=235, front_only=False)
        self.draw_earth(image, center, radius, t)
        if show_shadow:
            self.ring_ellipse(image, center, radius, ring_tilt, t, alpha=0, shadow=True)
        # bright front rim to improve depth
        self.ring_ellipse(image, center, radius, ring_tilt, t, alpha=255, front_only=True)

    def draw_title(self, image: Image.Image, t: float):
        intro_end = SHOT_PLAN[0]["end"]
        a = int(255 * smoothstep(t / (0.75 if not QUICK_MODE else .15)) * (1-smoothstep((t-(intro_end-.8))/.7)))
        if a > 4:
            draw_text(image,"WHAT IF EARTH HAD",(W//2,int(H*.055)),37 if not QUICK_MODE else 18,COLORS["white"]+(a,),True,"ma",2)
            draw_text(image,"RINGS LIKE SATURN?",(W//2,int(H*.105)),46 if not QUICK_MODE else 23,COLORS["cyan"]+(a,),True,"ma",2)
            draw_text(image,CONFIG["subtitle"],(W//2,int(H*.153)),15 if not QUICK_MODE else 7,COLORS["muted"]+(min(a,230),),True,"ma",1)
        elif t > intro_end-.4:
            labels={
                "reveal":"A RINGED EARTH",
                "latitudes":"THE VIEW CHANGES WITH LATITUDE",
                "shadows":"RINGS CAN CAST SHADOWS",
                "night":"A NEW NIGHT SKY",
                "particles":"NOT A SOLID DISK",
                "outro":"BEAUTIFUL — BUT HYPOTHETICAL",
            }
            draw_text(image,labels[get_shot(t)["name"]],(W//2,int(H*.052)),17 if not QUICK_MODE else 8,COLORS["muted"]+(225,),True,"ma",1)

    def draw_source_hud(self, image: Image.Image):
        draw_text(image,"RINGED EARTH // THOUGHT EXPERIMENT",(W-int(35*SCALE),int(55*SCALE)),13 if not QUICK_MODE else 6,COLORS["cyan"]+(215,),True,"ra",1)
        draw_text(image,"NASA RING PHYSICS + EARTH GEOMETRY",(W-int(35*SCALE),int(80*SCALE)),11 if not QUICK_MODE else 5,COLORS["muted"]+(190,),False,"ra",1)

    def draw_caption(self, image: Image.Image, t: float):
        text=caption_at(t)
        if not text:
            return
        y0=H-int(190*SCALE)
        panel(image,(int(58*SCALE),y0,W-int(58*SCALE),y0+int(116*SCALE)),150)
        draw_wrapped_text(image,text,(int(82*SCALE),y0+int(18*SCALE)),W-int(164*SCALE),30,COLORS["white"]+(240,))

    def draw_reveal(self, image: Image.Image, t: float, p: float):
        cx=W//2; cy=int(H*.46); r=int(210*SCALE)
        scale=.72+.28*ease_in_out_sine(p)
        self.draw_ringed_earth(image,(cx,cy),int(r*scale),t,ring_tilt=26)
        a=int(240*smoothstep((p-.35)/.35))
        draw_text(image,"ONE PLANET • BILLIONS OF ORBITING PARTICLES",(W//2,int(H*.735)),18 if not QUICK_MODE else 9,COLORS["gold"]+(a,),True,"ma",1)
        draw_text(image,"not a solid halo",(W//2,int(H*.775)),15 if not QUICK_MODE else 7,COLORS["white"]+(a,),False,"ma",1)

    def _mini_ground_view(self, image: Image.Image, box: Tuple[int,int,int,int], latitude: int, t: float):
        x0,y0,x1,y1=box
        layer=Image.new("RGBA",SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        d.rounded_rectangle(box,radius=max(6,int(18*SCALE)),fill=(3,8,20,205),outline=COLORS["cyan"]+(45,),width=max(1,int(2*SCALE)))
        # horizon
        horizon=int(y1-(y1-y0)*.24)
        d.rectangle((x0,horizon,x1,y1),fill=(12,43,54,235))
        d.line((x0,horizon,x1,horizon),fill=COLORS["green"]+(160,),width=max(1,int(2*SCALE)))
        cx=(x0+x1)//2
        width=(x1-x0)*.83
        # Equator: almost edge-on overhead. More latitude -> more open arch and lower apex.
        open_frac=max(.05,math.sin(math.radians(abs(latitude))))
        arch_h=(y1-y0)*(.05+.34*open_frac)
        apex=y0+(y1-y0)*(.30+.20*open_frac)
        for k, frac in enumerate((.0,.08,.15)):
            color=[COLORS["ring_ice"],COLORS["ring_warm"],COLORS["ring_dark"]][k]
            bbox=(cx-width/2,apex-arch_h*(1+frac),cx+width/2,apex+arch_h*(1+frac))
            d.arc(bbox,195,345,fill=color+(210-35*k,),width=max(1,int((5-1*k)*SCALE)))
        draw_text(layer,f"{latitude}° LAT",(cx,y0+int((y1-y0)*.08)),13 if not QUICK_MODE else 6,COLORS["white"]+(235,),True,"ma",1)
        image.alpha_composite(layer)

    def draw_latitudes(self, image: Image.Image, t: float, p: float):
        draw_text(image,"SAME RINGS — DIFFERENT SKY",(W//2,int(H*.15)),27 if not QUICK_MODE else 13,COLORS["white"]+(245,),True,"ma",2)
        margin=int(55*SCALE); gap=int(18*SCALE)
        total_w=W-2*margin
        panel_w=int((total_w-2*gap)/3)
        y0=int(H*.27); y1=int(H*.63)
        for i,lat in enumerate((0,35,70)):
            x0=margin+i*(panel_w+gap); x1=x0+panel_w
            self._mini_ground_view(image,(x0,y0,x1,y1),lat,t)
        draw_text(image,"schematic apparent geometry",(W//2,int(H*.68)),14 if not QUICK_MODE else 7,COLORS["muted"]+(210,),False,"ma",1)
        # little equatorial reference globe
        cx=W//2; cy=int(H*.77); r=int(75*SCALE)
        self.draw_earth(image,(cx,cy),r,t)
        d=ImageDraw.Draw(image)
        d.line((cx-int(r*1.45),cy,cx+int(r*1.45),cy),fill=COLORS["gold"]+(220,),width=max(1,int(4*SCALE)))
        draw_text(image,"EQUATORIAL RING PLANE",(cx,cy+int(110*SCALE)),14 if not QUICK_MODE else 7,COLORS["gold"]+(235,),True,"ma",1)

    def draw_shadows(self, image: Image.Image, t: float, p: float):
        cx=W//2; cy=int(H*.45); r=int(215*SCALE)
        # sun
        sx=int(W*.13); sy=int(H*.27); sr=int(45*SCALE)
        glow=Image.new("RGBA",SIZE,(0,0,0,0)); gd=ImageDraw.Draw(glow)
        for rr,a in [(sr*2.5,16),(sr*1.7,30),(sr,255)]:
            gd.ellipse((sx-rr,sy-rr,sx+rr,sy+rr),fill=COLORS["gold"]+(a,))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(2,int(8*SCALE)))))
        self.draw_ringed_earth(image,(cx,cy),r,t,ring_tilt=22,show_shadow=True)
        d=ImageDraw.Draw(image)
        # rays and shadow cue
        for off in (-45,0,45):
            d.line((sx+sr,sy+off*SCALE,cx-r*.8,cy+off*.25*SCALE),fill=COLORS["gold"]+(90,),width=max(1,int(2*SCALE)))
        draw_text(image,"RING SHADOW",(W//2,int(H*.705)),26 if not QUICK_MODE else 13,COLORS["orange"]+(245,),True,"ma",2)
        draw_text(image,"possible sunlight reduction — magnitude depends on ring properties",(W//2,int(H*.75)),14 if not QUICK_MODE else 7,COLORS["white"]+(225,),False,"ma",1)
        draw_text(image,"NOT A CLIMATE FORECAST",(W//2,int(H*.79)),12 if not QUICK_MODE else 6,COLORS["red"]+(220,),True,"ma",1)

    def draw_night(self, image: Image.Image, t: float, p: float):
        # ground silhouette
        d=ImageDraw.Draw(image)
        horizon=int(H*.70)
        d.rectangle((0,horizon,W,H),fill=(4,18,25,255))
        for i in range(18):
            x=int(W*i/17); y=horizon-int((18+28*((i*13)%7)/7)*SCALE)
            d.polygon([(x-int(30*SCALE),horizon),(x,y),(x+int(36*SCALE),horizon)],fill=(8,31,37,255))
        # huge luminous ring arch
        cx=W//2; cy=int(H*.56); rx=int(W*.66); ry=int(H*.29)
        for k,(scale,color,a) in enumerate([(1.0,COLORS["ring_ice"],225),(.90,COLORS["ring_warm"],190),(.78,COLORS["ring_dark"],100)]):
            box=(cx-rx*scale,cy-ry*scale,cx+rx*scale,cy+ry*scale)
            d.arc(box,195,345,fill=color+(a,),width=max(2,int((11-2*k)*SCALE)))
        glow=Image.new("RGBA",SIZE,(0,0,0,0)); gd=ImageDraw.Draw(glow)
        gd.arc((cx-rx,cy-ry,cx+rx,cy+ry),195,345,fill=COLORS["ring_ice"]+(95,),width=max(4,int(20*SCALE)))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(5,int(16*SCALE)))))
        draw_text(image,"THE NIGHT SKY CHANGES",(W//2,int(H*.18)),31 if not QUICK_MODE else 15,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"sunlit particles could form a vast luminous arc",(W//2,int(H*.235)),16 if not QUICK_MODE else 8,COLORS["cyan"]+(235,),True,"ma",1)
        draw_text(image,"brightness here is illustrative — not photometrically modeled",(W//2,int(H*.76)),13 if not QUICK_MODE else 6,COLORS["muted"]+(210,),False,"ma",1)

    def draw_particles(self, image: Image.Image, t: float, p: float):
        cx=W//2; cy=int(H*.43); pr=int(125*SCALE)
        self.draw_earth(image,(cx,cy),pr,t)
        d=ImageDraw.Draw(image)
        # conceptual Roche zone marker
        roche_r=pr*2.5
        d.ellipse((cx-roche_r,cy-roche_r,cx+roche_r,cy+roche_r),outline=COLORS["red"]+(95,),width=max(1,int(3*SCALE)))
        draw_text(image,"~2.5 Rₑ CONCEPT MARKER",(cx,cy-int(roche_r)-int(18*SCALE)),12 if not QUICK_MODE else 6,COLORS["red"]+(220,),True,"ma",1)

        # many independently orbiting particles in a flattened projected plane
        for rp in self.ring_particles:
            ang=rp.angle+t*(.17+.03*rp.band)
            rr=pr*rp.radius_frac
            x=cx+math.cos(ang)*rr
            y=cy+math.sin(ang)*rr*.24
            size=rp.size*(1.0+.18*math.sin(t*.7+rp.angle))
            color=[COLORS["ring_ice"],COLORS["ring_warm"],COLORS["white"],COLORS["ring_dark"]][rp.band]
            d.ellipse((x-size,y-size*.65,x+size,y+size*.65),fill=color+(rp.alpha,))

        # orbital arrows
        for scale in (1.45,1.85):
            rx=pr*scale; ry=rx*.24
            d.arc((cx-rx,cy-ry,cx+rx,cy+ry),200,510,fill=COLORS["cyan"]+(100,),width=max(1,int(2*SCALE)))
        draw_text(image,"A RING IS A SWARM OF ORBITS",(W//2,int(H*.69)),27 if not QUICK_MODE else 13,COLORS["white"]+(245,),True,"ma",2)
        draw_text(image,"each particle follows its own path around Earth",(W//2,int(H*.735)),15 if not QUICK_MODE else 7,COLORS["cyan"]+(235,),False,"ma",1)
        draw_text(image,"Roche limit depends on density + material strength",(W//2,int(H*.775)),12 if not QUICK_MODE else 6,COLORS["muted"]+(205,),False,"ma",1)

    def draw_outro(self, image: Image.Image, t: float, p: float):
        cx=W//2; cy=int(H*.43); r=int(225*SCALE)
        self.draw_ringed_earth(image,(cx,cy),r,t,ring_tilt=27,show_shadow=True)
        a=int(245*smoothstep((p-.18)/.48))
        draw_text(image,"EARTH WOULD LOOK ALIEN",(W//2,int(H*.70)),31 if not QUICK_MODE else 15,COLORS["white"]+(a,),True,"ma",2)
        draw_text(image,"but the real consequences depend on the rings we imagine",(W//2,int(H*.75)),16 if not QUICK_MODE else 8,COLORS["gold"]+(a,),True,"ma",1)
        draw_text(image,"EARTH TODAY: NO PLANETARY RING SYSTEM",(W//2,int(H*.80)),12 if not QUICK_MODE else 6,COLORS["green"]+(a,),True,"ma",1)

    def draw_scanlines(self, image: Image.Image, t: float):
        layer=Image.new("RGBA",SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        spacing=max(5,int(8*SCALE)); off=int((t*32)%spacing)
        for y in range(off,H,spacing):
            d.line((0,y,W,y),fill=(130,190,255,7),width=1)
        image.alpha_composite(layer)

    def render(self, t: float) -> Image.Image:
        image=self.background(t)
        shot=get_shot(t); p=local_progress(t,shot)

        if shot["name"]=="reveal":
            self.draw_reveal(image,t,p)
        elif shot["name"]=="latitudes":
            self.draw_latitudes(image,t,p)
        elif shot["name"]=="shadows":
            self.draw_shadows(image,t,p)
        elif shot["name"]=="night":
            self.draw_night(image,t,p)
        elif shot["name"]=="particles":
            self.draw_particles(image,t,p)
        else:
            self.draw_outro(image,t,p)

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
# Output helpers
# =============================================================================

def render_preview_frames(scene: RingedEarthScene) -> List[Path]:
    times=[]
    for shot in SHOT_PLAN:
        times.append(shot["start"] + .52*(shot["end"]-shot["start"]))
    paths=[]
    for i,t in enumerate(times,1):
        image=scene.render(min(t,DURATION-.001))
        path=PREVIEW_ROOT/f"preview_{i:02d}_{get_shot(t)['name']}.jpg"
        image.save(path,quality=92)
        paths.append(path)
    return paths


def make_contact_sheet(paths: Sequence[Path]) -> Path:
    images=[Image.open(p).convert("RGB") for p in paths]
    thumb_w=int(300*SCALE if not QUICK_MODE else 150)
    thumb_h=int(533*SCALE if not QUICK_MODE else 267)
    cols=3
    rows=math.ceil(len(images)/cols)
    margin=max(12,int(18*SCALE))
    sheet=Image.new("RGB",(cols*thumb_w+(cols+1)*margin,rows*thumb_h+(rows+1)*margin),(5,8,16))
    for i,img in enumerate(images):
        thumb=img.copy(); thumb.thumbnail((thumb_w,thumb_h),Image.Resampling.LANCZOS)
        x=margin+(i%cols)*(thumb_w+margin); y=margin+(i//cols)*(thumb_h+margin)
        sheet.paste(thumb,(x,y))
    path=PREVIEW_ROOT/f"{CONFIG['basename']}_contact_sheet.jpg"
    sheet.save(path,quality=92)
    return path


def render_video(scene: RingedEarthScene) -> Path:
    path=OUTPUT_ROOT/f"{CONFIG['basename']}.mp4"
    total_frames=max(1,int(round(DURATION*FPS)))
    writer=iio.get_writer(
        path,
        fps=FPS,
        codec="libx264",
        quality=7,
        pixelformat="yuv420p",
        ffmpeg_log_level="error",
        macro_block_size=2,
    )
    try:
        for i in tqdm(range(total_frames),desc="Rendering Ringed Earth Short"):
            t=i/FPS
            frame=np.asarray(scene.render(t).convert("RGB"))
            writer.append_data(frame)
    finally:
        writer.close()
    return path


def copy_flat_artifacts(video_path: Path, contact_sheet: Path, srt_path: Path) -> Tuple[Path, Path, Path]:
    flat_video=Path(f"{CONFIG['basename']}_quick_preview.mp4") if QUICK_MODE else Path(f"{CONFIG['basename']}.mp4")
    flat_sheet=Path(f"{CONFIG['basename']}_contact_sheet.jpg")
    flat_srt=Path(f"{CONFIG['basename']}_subtitles.srt")
    shutil.copy2(video_path,flat_video)
    shutil.copy2(contact_sheet,flat_sheet)
    shutil.copy2(srt_path,flat_srt)
    return flat_video,flat_sheet,flat_srt


def print_summary(paths: Dict[str, Path], snapshot: RingedEarthSnapshot):
    print("\nRender complete")
    print("="*72)
    print(f"Title:       {CONFIG['title']}")
    print(f"Mode:        {'QUICK' if QUICK_MODE else ('4K' if FOUR_K else 'FULL HD')}")
    print(f"Frame size:  {W}x{H} @ {FPS} fps")
    print(f"Duration:    {DURATION:.1f} s")
    print(f"Ring plane:  {snapshot.assumed_ring_plane}")
    print(f"Earth rings: {snapshot.earth_has_rings_now}")
    for key,path in paths.items():
        print(f"{key:12s} {path}")



# =============================================================================
# YouTube title / description sidecar
# =============================================================================

YOUTUBE_TITLE = 'What If Earth Had Rings Like Saturn? 🌍💍 #rootjatin'
YOUTUBE_DESCRIPTION = 'A hypothetical look at Earth surrounded by a broad equatorial ring system made of countless orbiting particles. The video explores how the rings could appear from different latitudes, how ring shadows might cross Earth, and why the brightness and climate effects would depend on ring width, density, material, and geometry. Earth has no such rings today, so the scenes are a controlled thought experiment rather than a forecast.'
YOUTUBE_HASHTAGS = '#rootjatin #Earth #Saturn #PlanetaryRings #WhatIf #Astronomy #Space #Science'

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

    scene=RingedEarthScene(snapshot)
    preview_paths=render_preview_frames(scene)
    contact_sheet=make_contact_sheet(preview_paths)
    video_path=render_video(scene)
    flat_video,flat_sheet,flat_srt=copy_flat_artifacts(video_path,contact_sheet,srt_path)

    print_summary({
        "video":video_path,
        "subtitles":srt_path,
        "csv":csv_path,
        "json":json_path,
        "contact":contact_sheet,
        "flat_video":flat_video,
        "flat_sheet":flat_sheet,
        "flat_srt":flat_srt,
    },snapshot)
    metadata_txt = write_youtube_metadata_txt()
    print("Title/description TXT:", metadata_txt.resolve())


if __name__ == "__main__":
    main()

