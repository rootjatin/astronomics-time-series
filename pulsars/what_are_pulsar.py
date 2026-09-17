from __future__ import annotations

"""
Understand What Pulsars Are — cinematic YouTube Short renderer

A vertical 1080x1920 science explainer that visualizes why pulsars appear to
"blink": a compact neutron star rotates, its magnetic axis is tilted relative
to its spin axis, and narrow radiation beams sweep through space like a
lighthouse. A pulse is detected only when one of those beams crosses our line
of sight.

SCIENTIFIC FRAMING
------------------
- Pulsars are rapidly rotating, highly magnetized neutron stars.
- Neutron stars are compact remnants of massive stars after supernovae.
- The apparent pulse is a viewing-geometry effect: the beam sweeps past Earth.
- Pulsars can emit across radio, X-ray, and gamma-ray wavelengths.
- NASA describes neutron stars as city-sized objects containing more mass than
  the Sun in a sphere less than about 17 miles (27 km) wide.
- NASA lists PSR J1748-2446ad as the fastest known pulsar at 716 rotations/s.

The neutron-star surface, magnetic field lines, beams, pulse train, supernova,
and camera motion in this video are SCHEMATIC VISUALIZATIONS, not telescope
imagery. The numerical facts called out on-screen are sourced in the metadata.

Quick preview
-------------
    PULSAR_SHORT_QUICK=1 python understand_what_pulsars_are.py

Skip audio
----------
    PULSAR_SHORT_NO_AUDIO=1 python understand_what_pulsars_are.py

Optional measured pulse profile
-------------------------------
Provide a CSV with columns phase,intensity (phase usually 0..1):

    PULSAR_PROFILE_CSV=/path/to/profile.csv python understand_what_pulsars_are.py

Recommended install
-------------------
    pip install numpy pandas pillow imageio imageio-ffmpeg tqdm

Outputs
-------
- final vertical MP4 with generated cinematic soundtrack
- silent MP4 fallback
- SRT subtitle sidecar
- preview PNG frames
- CSV of science facts used in the story
- JSON summary and source notes

Primary references
------------------
- NASA Science — Pulsars:
  https://science.nasa.gov/mission/hubble/science/science-behind-the-discoveries/hubble-pulsars/
- NASA Science — Fermi mission nets 300 gamma-ray pulsars:
  https://science.nasa.gov/universe/stars/neutron-stars/pulsars/nasas-fermi-mission-nets-300-gamma-ray-pulsars-and-counting/
- NASA Imagine the Universe — Neutron Stars / Pulsars:
  https://imagine.gsfc.nasa.gov/science/objects/neutron_stars1.html
"""

import csv
import json
import math
import os
import shutil
import subprocess
import wave
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

QUICK_MODE = os.environ.get("PULSAR_SHORT_QUICK", "0") == "1"
NO_AUDIO = os.environ.get("PULSAR_SHORT_NO_AUDIO", "0") == "1"
PROFILE_CSV = os.environ.get("PULSAR_PROFILE_CSV", "").strip()

OUTPUT_ROOT = Path("understand_what_pulsars_are_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12.0 if QUICK_MODE else 58.0,
    "output_basename": "understand_what_pulsars_are",
    "title_1": "UNDERSTAND",
    "title_2": "WHAT PULSARS ARE",
    "subtitle": "NEUTRON STAR // MAGNETIC BEAM // COSMIC LIGHTHOUSE",
    "soundtrack_sample_rate": 22050 if QUICK_MODE else 44100,
    "grain_strength": 4.2,
    "vignette": 0.48,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
S = OUT_W / 1080.0

COLORS = {
    "black": (1, 2, 7),
    "space": (3, 5, 16),
    "deep_blue": (7, 15, 37),
    "violet": (128, 91, 255),
    "blue": (67, 170, 255),
    "cyan": (94, 241, 255),
    "beam": (145, 239, 255),
    "magenta": (244, 93, 255),
    "amber": (255, 193, 84),
    "orange": (255, 126, 65),
    "red": (255, 72, 92),
    "white": (248, 250, 255),
    "muted": (161, 184, 211),
    "grid": (83, 113, 157),
}

FACTS = [
    {
        "fact": "definition",
        "value": "rapidly rotating, highly magnetized neutron star observed as pulses",
        "source": "NASA Science — Pulsars",
    },
    {
        "fact": "pulse_geometry",
        "value": "radiation beam sweeps past Earth like a lighthouse",
        "source": "NASA Science — Pulsars",
    },
    {
        "fact": "compactness",
        "value": "more mass than the Sun in a sphere less than about 17 miles / 27 km wide",
        "source": "NASA Science — Fermi pulsars",
    },
    {
        "fact": "fastest_known",
        "value": "PSR J1748-2446ad — 716 rotations per second",
        "source": "NASA Science — Fermi pulsars",
    },
    {
        "fact": "discovery",
        "value": "pulsars were discovered in 1967 by Jocelyn Bell Burnell",
        "source": "NASA Science — Pulsars",
    },
]

SOURCE_URLS = {
    "nasa_pulsars": "https://science.nasa.gov/mission/hubble/science/science-behind-the-discoveries/hubble-pulsars/",
    "nasa_fermi_pulsars": "https://science.nasa.gov/universe/stars/neutron-stars/pulsars/nasas-fermi-mission-nets-300-gamma-ray-pulsars-and-counting/",
    "nasa_neutron_stars": "https://imagine.gsfc.nasa.gov/science/objects/neutron_stars1.html",
}

FULL_SHOT_PLAN = [
    {"name": "cold_open", "start": 0.0, "end": 6.5},
    {"name": "collapse", "start": 6.5, "end": 17.0},
    {"name": "lighthouse", "start": 17.0, "end": 31.0},
    {"name": "pulse_train", "start": 31.0, "end": 42.5},
    {"name": "fastest", "start": 42.5, "end": 52.5},
    {"name": "definition", "start": 52.5, "end": 58.0},
]

FULL_CAPTIONS = [
    (0.4, 6.2, "A pulsar looks like a star that switches on and off with impossible precision. But the star itself is not blinking."),
    (6.8, 16.6, "A massive star can explode and leave behind a neutron star: a city-sized object with more mass than the Sun packed into an extraordinarily small volume."),
    (17.3, 30.6, "If the neutron star spins rapidly and its magnetic axis is tilted, beams of radiation sweep through space like a lighthouse."),
    (31.3, 42.1, "Each time a beam crosses our line of sight, a telescope records a pulse. The pulse rate tells us how fast the neutron star is rotating."),
    (42.8, 52.0, "Some pulsars spin hundreds of times every second. The fastest known, PSR J1748-2446ad, rotates 716 times per second."),
    (52.8, 57.8, "So a pulsar is not a flashing star. It is a rotating neutron star whose beam repeatedly sweeps across Earth."),
]

if QUICK_MODE:
    scale = float(CONFIG["duration_s"]) / 58.0
    SHOT_PLAN = [
        {"name": shot["name"], "start": shot["start"] * scale, "end": shot["end"] * scale}
        for shot in FULL_SHOT_PLAN
    ]
    CAPTIONS = [(a * scale, b * scale, text) for a, b, text in FULL_CAPTIONS]
else:
    SHOT_PLAN = FULL_SHOT_PLAN
    CAPTIONS = FULL_CAPTIONS


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(value: float) -> float:
    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


def smootherstep(value: float) -> float:
    x = clamp(value)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def mix_color(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = clamp(t)
    return tuple(int(round(lerp(x, y, t))) for x, y in zip(a, b))


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def shot_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - float(shot["start"])) / max(float(shot["end"] - shot["start"]), 1e-9))


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    hh = ms // 3_600_000
    ms %= 3_600_000
    mm = ms // 60_000
    ms %= 60_000
    ss = ms // 1000
    ms %= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for idx, (start, end, text) in enumerate(captions, 1):
        lines += [str(idx), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def get_font(size: int, bold: bool = False, condensed: bool = False):
    candidates: List[str] = []
    if condensed:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
            "DejaVuSansCondensed-Bold.ttf" if bold else "DejaVuSansCondensed.ttf",
        ]
    candidates += [
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
    xy: Tuple[float, float],
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
        stroke_width=max(0, stroke),
        stroke_fill=(0, 0, 0, min(235, fill[3])),
    )


def rounded_panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 165):
    layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle(
        box,
        radius=max(10, int(24 * S)),
        fill=(3, 8, 19, alpha),
        outline=COLORS["grid"] + (80,),
        width=max(1, int(2 * S)),
    )
    image.alpha_composite(layer)


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int,
    fill: Tuple[int, int, int, int],
    line_gap: int,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=1)
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
        draw.text(
            (x, y), line, font=font, fill=fill,
            stroke_width=1, stroke_fill=(0, 0, 0, 220)
        )
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=1)
        y += bbox[3] - bbox[1] + line_gap


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.65, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


def apply_finish(image: Image.Image, t: float) -> np.ndarray:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    arr *= VIGNETTE[..., None]
    rng = np.random.default_rng(int(t * 1000) + 733)
    grain = rng.normal(0.0, float(CONFIG["grain_strength"]), arr.shape[:2])[..., None]
    arr += grain
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)
    out = ImageEnhance.Contrast(out).enhance(1.08)
    out = ImageEnhance.Color(out).enhance(1.05)
    return np.asarray(out)


def rotate_point(x: float, y: float, cx: float, cy: float, angle: float) -> Tuple[float, float]:
    ca, sa = math.cos(angle), math.sin(angle)
    dx, dy = x - cx, y - cy
    return cx + dx * ca - dy * sa, cy + dx * sa + dy * ca


def cubic_bezier(p0, p1, p2, p3, samples: int = 50):
    pts = []
    for i in range(samples):
        u = i / max(samples - 1, 1)
        v = 1.0 - u
        x = v**3 * p0[0] + 3 * v * v * u * p1[0] + 3 * v * u * u * p2[0] + u**3 * p3[0]
        y = v**3 * p0[1] + 3 * v * v * u * p1[1] + 3 * v * u * u * p2[1] + u**3 * p3[1]
        pts.append((x, y))
    return pts


# -----------------------------------------------------------------------------
# Optional pulse profile
# -----------------------------------------------------------------------------

def load_profile() -> Tuple[np.ndarray, np.ndarray, str]:
    if PROFILE_CSV:
        try:
            frame = pd.read_csv(PROFILE_CSV)
            lower = {str(c).strip().lower(): c for c in frame.columns}
            if "phase" in lower and "intensity" in lower:
                phase = pd.to_numeric(frame[lower["phase"]], errors="coerce").to_numpy(dtype=float)
                intensity = pd.to_numeric(frame[lower["intensity"]], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(phase) & np.isfinite(intensity)
                phase, intensity = phase[mask], intensity[mask]
                if len(phase) >= 8:
                    order = np.argsort(phase)
                    phase, intensity = phase[order], intensity[order]
                    intensity = intensity - np.min(intensity)
                    peak = max(float(np.max(intensity)), 1e-9)
                    return phase, intensity / peak, "user_csv"
        except Exception:
            pass

    phase = np.linspace(0.0, 1.0, 420)
    # A deliberately schematic double-peaked profile, not assigned to a real pulsar.
    g1 = np.exp(-0.5 * ((phase - 0.22) / 0.035) ** 2)
    g2 = 0.72 * np.exp(-0.5 * ((phase - 0.63) / 0.055) ** 2)
    shoulder = 0.16 * np.exp(-0.5 * ((phase - 0.47) / 0.12) ** 2)
    intensity = (g1 + g2 + shoulder)
    intensity /= max(float(np.max(intensity)), 1e-9)
    return phase, intensity, "schematic"


PROFILE_PHASE, PROFILE_INTENSITY, PROFILE_SOURCE = load_profile()


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class PulsarScene:
    def __init__(self):
        self.rng = np.random.default_rng(1967)
        count = 250 if QUICK_MODE else 620
        self.stars = [
            (
                float(self.rng.uniform(0, OUT_W)),
                float(self.rng.uniform(0, OUT_H)),
                float(self.rng.uniform(0.35, 1.9 if QUICK_MODE else 2.7)),
                int(self.rng.integers(45, 210)),
                float(self.rng.uniform(0, math.tau)),
            )
            for _ in range(count)
        ]
        self.dust = [
            (
                float(self.rng.uniform(0, OUT_W)),
                float(self.rng.uniform(0, OUT_H)),
                float(self.rng.uniform(0.1, 1.0)),
                float(self.rng.uniform(0, math.tau)),
            )
            for _ in range(80 if QUICK_MODE else 180)
        ]

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
        top = np.array(COLORS["deep_blue"], dtype=float)
        bottom = np.array(COLORS["black"], dtype=float)
        for y in range(OUT_H):
            u = y / max(OUT_H - 1, 1)
            rgb = (top * (1.0 - u) + bottom * u).astype(np.uint8)
            arr[y, :, :3] = rgb
            arr[y, :, 3] = 255
        image = Image.fromarray(arr, "RGBA")

        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for cx, cy, rgb in [
            (OUT_W * 0.30, OUT_H * 0.30, COLORS["violet"]),
            (OUT_W * 0.76, OUT_H * 0.55, COLORS["blue"]),
        ]:
            for radius, alpha in [(OUT_W * 0.33, 9), (OUT_W * 0.20, 14), (OUT_W * 0.09, 20)]:
                hd.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=rgb + (alpha,))
        image.alpha_composite(haze.filter(ImageFilter.GaussianBlur(max(20, int(55*S)))))

        draw = ImageDraw.Draw(image)
        for x, y, r, alpha, phase in self.stars:
            twinkle = 0.55 + 0.45 * math.sin(t * 0.9 + phase) ** 2
            drift_x = math.sin(t * 0.04 + phase) * 2.0 * S
            drift_y = math.cos(t * 0.035 + phase) * 1.3 * S
            rr = max(0.45, r * S)
            a = int(alpha * twinkle)
            draw.ellipse((x+drift_x-rr, y+drift_y-rr, x+drift_x+rr, y+drift_y+rr), fill=(220, 235, 255, a))
        return image

    def draw_header(self, image: Image.Image, t: float, scene_name: str):
        reveal = smoothstep(t / (2.0 if not QUICK_MODE else 0.5))
        if scene_name == "cold_open":
            alpha = int(245 * reveal)
            draw_text(image, CONFIG["title_1"], (OUT_W/2, 92*S), int(35*S), COLORS["muted"]+(alpha,), True, True, "ma", 1)
            draw_text(image, CONFIG["title_2"], (OUT_W/2, 135*S), int(60*S), COLORS["white"]+(alpha,), True, True, "ma", 2)
        else:
            draw_text(image, "PULSARS // COSMIC LIGHTHOUSES", (54*S, 62*S), int(21*S), COLORS["muted"]+(225,), True, True, "la", 1)
            ImageDraw.Draw(image).line((54*S, 91*S, OUT_W-54*S, 91*S), fill=COLORS["grid"]+(70,), width=max(1,int(2*S)))

    def draw_footer(self, image: Image.Image, text: str = "SCHEMATIC VISUALIZATION // NASA-SOURCED FACTS"):
        draw_text(image, text, (OUT_W/2, OUT_H-34*S), int(14*S), COLORS["muted"]+(175,), True, True, "ms", 1)

    def draw_neutron_star(
        self,
        image: Image.Image,
        center: Tuple[float, float],
        radius: float,
        t: float,
        spin_hz_visual: float = 0.35,
        show_field: bool = True,
        show_beams: bool = True,
        beam_alpha: float = 1.0,
        magnetic_tilt: float = math.radians(33),
    ):
        cx, cy = center
        angle = t * math.tau * spin_hz_visual

        if show_beams:
            beam_layer = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
            bd = ImageDraw.Draw(beam_layer)
            axis_angle = angle + magnetic_tilt
            length = max(OUT_W, OUT_H) * 0.85
            half_width = radius * 0.25
            for sign in (-1, 1):
                a = axis_angle + (0 if sign > 0 else math.pi)
                ux, uy = math.cos(a), math.sin(a)
                px, py = -uy, ux
                start = radius * 0.55
                p1 = (cx + ux*start + px*half_width, cy + uy*start + py*half_width)
                p2 = (cx + ux*length + px*half_width*5.0, cy + uy*length + py*half_width*5.0)
                p3 = (cx + ux*length - px*half_width*5.0, cy + uy*length - py*half_width*5.0)
                p4 = (cx + ux*start - px*half_width, cy + uy*start - py*half_width)
                alpha = int(88 * clamp(beam_alpha))
                bd.polygon([p1,p2,p3,p4], fill=COLORS["beam"]+(alpha,))
                bd.line((cx+ux*start, cy+uy*start, cx+ux*length, cy+uy*length), fill=COLORS["white"]+(int(110*beam_alpha),), width=max(1,int(2*S)))
            beam_layer = beam_layer.filter(ImageFilter.GaussianBlur(max(3,int(12*S))))
            image.alpha_composite(beam_layer)

        if show_field:
            field = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
            fd = ImageDraw.Draw(field)
            for side in (-1, 1):
                for scale in (1.6, 2.15, 2.85):
                    # Build a curved dipole-like line, then rotate it with the magnetic axis.
                    p0 = (cx, cy - radius*0.62*side)
                    p1 = (cx + radius*scale, cy - radius*scale*0.85*side)
                    p2 = (cx + radius*scale, cy + radius*scale*0.85*side)
                    p3 = (cx, cy + radius*0.62*side)
                    pts = cubic_bezier(p0,p1,p2,p3,45)
                    pts = [rotate_point(x,y,cx,cy,angle+magnetic_tilt-math.pi/2) for x,y in pts]
                    fd.line(pts, fill=COLORS["cyan"]+(40,), width=max(1,int(2*S)))
                    mirrored = [(2*cx-x,y) for x,y in pts]
                    fd.line(mirrored, fill=COLORS["violet"]+(32,), width=max(1,int(2*S)))
            image.alpha_composite(field)

        # Soft glow.
        glow = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        for mul, alpha in [(2.2,22),(1.6,32),(1.2,48)]:
            rr = radius*mul
            gd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr), fill=COLORS["cyan"]+(alpha,))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(8,int(26*S)))))

        # Radial-ish sphere shading with layered circles.
        sphere = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        sd = ImageDraw.Draw(sphere)
        steps = 26 if QUICK_MODE else 42
        for i in range(steps, 0, -1):
            u = i / steps
            rr = radius*u
            col = mix_color((30,38,76), (191,236,255), 1.0-u)
            shade = mix_color(col, COLORS["violet"], 0.16*(1-u))
            sd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr), fill=shade+(255,))
        # Hot caps rotate around the star.
        for offset, col in [(0, COLORS["white"]), (math.pi, COLORS["magenta"])]:
            a = angle + magnetic_tilt + offset
            hx = cx + math.cos(a)*radius*0.48
            hy = cy + math.sin(a)*radius*0.28
            hr = radius*0.16
            sd.ellipse((hx-hr,hy-hr,hx+hr,hy+hr), fill=col+(220,))
        # Equatorial streaks imply rotation.
        for k in range(7):
            yy = cy + (k-3)*radius*0.12
            span = radius*math.sqrt(max(0.0,1.0-((yy-cy)/radius)**2))
            phase = (t*spin_hz_visual*3.0 + k*0.17) % 1.0
            x0 = cx-span + phase*span*0.8
            sd.arc((cx-radius, yy-radius*0.14, cx+radius, yy+radius*0.14), 195, 345, fill=COLORS["white"]+(28,), width=max(1,int(2*S)))
        image.alpha_composite(sphere)

        # Spin-axis line.
        d = radius*2.2
        axis = ImageDraw.Draw(image)
        axis.line((cx,cy-d,cx,cy+d), fill=COLORS["muted"]+(60,), width=max(1,int(2*S)))
        axis.ellipse((cx-radius*0.07, cy-radius*1.47, cx+radius*0.07, cy-radius*1.33), fill=COLORS["white"]+(150,))

    def draw_collapse(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smootherstep(shot_progress(t,shot))
        cx, cy = OUT_W*0.50, OUT_H*0.48
        start_r = OUT_W*0.24
        end_r = OUT_W*0.075
        # Supernova shell expands while core contracts.
        shell_r = lerp(OUT_W*0.16, OUT_W*0.72, smoothstep(p))
        shell = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        sd = ImageDraw.Draw(shell)
        for mul, alpha, col in [
            (1.0, int(120*(1-p)), COLORS["orange"]),
            (0.88, int(75*(1-p)), COLORS["amber"]),
            (0.72, int(42*(1-p)), COLORS["magenta"]),
        ]:
            rr=shell_r*mul
            sd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr), outline=col+(alpha,), width=max(1,int(5*S)))
        image.alpha_composite(shell.filter(ImageFilter.GaussianBlur(max(4,int(10*S)))))

        star_r = lerp(start_r,end_r,p)
        core = Image.new("RGBA",OUT_SIZE,(0,0,0,0))
        cd=ImageDraw.Draw(core)
        for i in range(20,0,-1):
            u=i/20
            rr=star_r*u
            col=mix_color(COLORS["orange"],COLORS["white"],1-u)
            cd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=col+(255,))
        image.alpha_composite(core)

        draw_text(image,"A MASSIVE STAR DIES",(OUT_W/2,OUT_H*0.73),int(28*S),COLORS["muted"]+(230,),True,True,"ma",1)
        if p>0.42:
            a=int(255*smoothstep((p-0.42)/0.25))
            draw_text(image,"THE CORE COLLAPSES",(OUT_W/2,OUT_H*0.775),int(49*S),COLORS["white"]+(a,),True,True,"ma",2)
        if p>0.70:
            a=int(240*smoothstep((p-0.70)/0.20))
            rounded_panel(image,(int(120*S),int(1480*S),int(960*S),int(1690*S)),170)
            draw_text(image,"NEUTRON STAR",(OUT_W/2,1518*S),int(37*S),COLORS["cyan"]+(a,),True,True,"ma",1)
            draw_text(image,"CITY-SIZED  //  MORE MASS THAN THE SUN",(OUT_W/2,1588*S),int(21*S),COLORS["white"]+(a,),True,True,"ma",1)
            draw_text(image,"< 27 KM WIDE (NASA DESCRIPTION)",(OUT_W/2,1642*S),int(18*S),COLORS["muted"]+(a,),True,True,"ma",1)

    def draw_cold_open(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p=shot_progress(t,shot)
        cx,cy=OUT_W*0.5,OUT_H*0.49
        # A distant object flashes on a steady cadence.
        period = 0.92 if not QUICK_MODE else 0.46
        phase=(t%period)/period
        pulse=math.exp(-0.5*((phase-0.10)/0.045)**2)
        glow=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); gd=ImageDraw.Draw(glow)
        rr=lerp(7*S,24*S,pulse)
        gd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=COLORS["cyan"]+(int(90+150*pulse),))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(6,int(22*S)))))
        ImageDraw.Draw(image).ellipse((cx-5*S,cy-5*S,cx+5*S,cy+5*S),fill=COLORS["white"]+(255,))
        if p>0.18:
            a=int(255*smoothstep((p-0.18)/0.22))
            draw_text(image,"WHY DOES A DEAD STAR",(OUT_W/2,OUT_H*0.66),int(34*S),COLORS["muted"]+(a,),True,True,"ma",1)
            draw_text(image,"BLINK?",(OUT_W/2,OUT_H*0.72),int(72*S),COLORS["white"]+(a,),True,True,"ma",2)
        if p>0.70:
            a=int(235*smoothstep((p-0.70)/0.18))
            draw_text(image,"IT DOESN'T.",(OUT_W/2,OUT_H*0.82),int(34*S),COLORS["cyan"]+(a,),True,True,"ma",1)

    def beam_alignment(self, t: float, spin_visual: float = 0.24) -> float:
        # A 2-D stand-in for how directly the sweeping beam points toward Earth.
        angle = (t * math.tau * spin_visual + math.radians(33)) % math.tau
        # Earth is to the right, so strongest when beam angle approaches 0 or pi.
        return max(0.0, math.cos(angle)) ** 18

    def draw_lighthouse(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p=shot_progress(t,shot)
        center=(OUT_W*0.42,OUT_H*0.49)
        radius=OUT_W*(0.11+0.018*math.sin(p*math.pi))
        spin_visual=0.18+0.18*smoothstep(p)
        self.draw_neutron_star(image,center,radius,t,spin_hz_visual=spin_visual,show_field=True,show_beams=True,beam_alpha=0.9)

        # Earth / observer line of sight.
        earth=(OUT_W*0.86,OUT_H*0.49)
        d=ImageDraw.Draw(image)
        d.line((center[0]+radius*1.2,center[1],earth[0]-20*S,earth[1]),fill=COLORS["muted"]+(72,),width=max(1,int(2*S)))
        er=15*S
        d.ellipse((earth[0]-er,earth[1]-er,earth[0]+er,earth[1]+er),fill=(55,109,172,255),outline=COLORS["cyan"]+(180,),width=max(1,int(2*S)))
        draw_text(image,"EARTH",(earth[0],earth[1]+34*S),int(16*S),COLORS["muted"]+(220,),True,True,"ma",1)

        alignment=self.beam_alignment(t,spin_visual)
        if alignment>0.12:
            flash=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); fd=ImageDraw.Draw(flash)
            rr=er*(1.4+alignment*3.0)
            fd.ellipse((earth[0]-rr,earth[1]-rr,earth[0]+rr,earth[1]+rr),fill=COLORS["cyan"]+(int(120*alignment),))
            image.alpha_composite(flash.filter(ImageFilter.GaussianBlur(max(4,int(12*S)))))

        rounded_panel(image,(int(96*S),int(1390*S),int(984*S),int(1680*S)),155)
        draw_text(image,"SPIN AXIS",(142*S,1445*S),int(18*S),COLORS["muted"]+(220,),True,True,"la",1)
        draw_text(image,"MAGNETIC AXIS",(142*S,1505*S),int(18*S),COLORS["cyan"]+(230,),True,True,"la",1)
        draw_text(image,"THE AXES ARE TILTED",(OUT_W/2,1584*S),int(34*S),COLORS["white"]+(245,),True,True,"ma",2)
        draw_text(image,"THE BEAM SWEEPS ACROSS SPACE",(OUT_W/2,1634*S),int(19*S),COLORS["muted"]+(220,),True,True,"ma",1)

    def draw_profile_graph(self,image:Image.Image,box:Tuple[int,int,int,int],phase_offset:float=0.0,active_phase:Optional[float]=None):
        x0,y0,x1,y1=box
        draw=ImageDraw.Draw(image)
        draw.rounded_rectangle(box,radius=max(10,int(18*S)),fill=(2,7,19,175),outline=COLORS["grid"]+(80,),width=max(1,int(2*S)))
        baseline=y1-42*S
        draw.line((x0+26*S,baseline,x1-24*S,baseline),fill=COLORS["grid"]+(100,),width=max(1,int(2*S)))
        draw.line((x0+26*S,y0+24*S,x0+26*S,baseline),fill=COLORS["grid"]+(70,),width=max(1,int(2*S)))
        pts=[]
        for ph,intensity in zip(PROFILE_PHASE,PROFILE_INTENSITY):
            ph2=(float(ph)+phase_offset)%1.0
            xx=lerp(x0+30*S,x1-28*S,ph2)
            yy=baseline-float(intensity)*(baseline-(y0+34*S))
            pts.append((xx,yy))
        pts.sort(key=lambda p:p[0])
        draw.line(pts,fill=COLORS["cyan"]+(235,),width=max(1,int(4*S)))
        if active_phase is not None:
            xx=lerp(x0+30*S,x1-28*S,active_phase%1.0)
            draw.line((xx,y0+20*S,xx,baseline),fill=COLORS["white"]+(120,),width=max(1,int(2*S)))
        draw_text(image,"PULSE PROFILE",(x0+32*S,y0+18*S),int(17*S),COLORS["muted"]+(215,),True,True,"la",1)

    def draw_pulse_train(self,image:Image.Image,t:float,shot:Dict[str,Any]):
        p=shot_progress(t,shot)
        center=(OUT_W*0.5,OUT_H*0.35)
        spin_visual=0.56 if not QUICK_MODE else 0.8
        self.draw_neutron_star(image,center,OUT_W*0.085,t,spin_hz_visual=spin_visual,show_field=False,show_beams=True,beam_alpha=0.78)

        graph_box=(int(80*S),int(1040*S),int(1000*S),int(1460*S))
        active=(t*spin_visual)%1.0
        self.draw_profile_graph(image,graph_box,0.0,active)
        draw_text(image,"ONE SWEEP PAST EARTH = ONE PULSE",(OUT_W/2,1510*S),int(29*S),COLORS["white"]+(245,),True,True,"ma",2)
        draw_text(image,"PULSE PERIOD = ROTATION PERIOD",(OUT_W/2,1562*S),int(19*S),COLORS["cyan"]+(230,),True,True,"ma",1)
        label="USER PULSE PROFILE" if PROFILE_SOURCE=="user_csv" else "SCHEMATIC PROFILE"
        draw_text(image,label,(OUT_W/2,1612*S),int(15*S),COLORS["muted"]+(185,),True,True,"ma",1)

    def draw_fastest(self,image:Image.Image,t:float,shot:Dict[str,Any]):
        p=smootherstep(shot_progress(t,shot))
        # Increase visual spin, but never attempt to render 716 physical rotations/s.
        visual_hz=lerp(0.7,3.2,p)
        center=(OUT_W*0.5,OUT_H*0.44)
        radius=OUT_W*0.095
        self.draw_neutron_star(image,center,radius,t,spin_hz_visual=visual_hz,show_field=True,show_beams=True,beam_alpha=0.78)
        # Circular motion streaks.
        streak=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); sd=ImageDraw.Draw(streak)
        for rmul,a in [(1.4,55),(1.75,37),(2.15,22)]:
            rr=radius*rmul
            start=(t*220*visual_hz)%360
            sd.arc((center[0]-rr,center[1]-rr,center[0]+rr,center[1]+rr),start,start+250,fill=COLORS["cyan"]+(a,),width=max(1,int(3*S)))
        image.alpha_composite(streak.filter(ImageFilter.GaussianBlur(max(1,int(3*S)))))

        rounded_panel(image,(int(86*S),int(1290*S),int(994*S),int(1698*S)),180)
        draw_text(image,"FASTEST KNOWN PULSAR",(OUT_W/2,1340*S),int(23*S),COLORS["muted"]+(235,),True,True,"ma",1)
        draw_text(image,"PSR J1748−2446ad",(OUT_W/2,1402*S),int(36*S),COLORS["white"]+(255,),True,True,"ma",2)
        count=int(round(lerp(40,716,p)))
        draw_text(image,f"{count}",(OUT_W/2,1520*S),int(94*S),COLORS["cyan"]+(255,),True,True,"ma",2)
        draw_text(image,"ROTATIONS PER SECOND",(OUT_W/2,1600*S),int(25*S),COLORS["white"]+(245,),True,True,"ma",1)
        draw_text(image,"VISUAL SPIN SLOWED DRAMATICALLY",(OUT_W/2,1660*S),int(14*S),COLORS["muted"]+(185,),True,True,"ma",1)

    def draw_definition(self,image:Image.Image,t:float,shot:Dict[str,Any]):
        p=shot_progress(t,shot)
        center=(OUT_W*0.5,OUT_H*0.39)
        self.draw_neutron_star(image,center,OUT_W*0.085,t,spin_hz_visual=0.48,show_field=True,show_beams=True,beam_alpha=0.72)
        a=int(255*smoothstep(p*1.7))
        draw_text(image,"PULSAR =",(OUT_W/2,1185*S),int(30*S),COLORS["muted"]+(a,),True,True,"ma",1)
        draw_text(image,"ROTATING NEUTRON STAR",(OUT_W/2,1260*S),int(44*S),COLORS["white"]+(a,),True,True,"ma",2)
        draw_text(image,"+",(OUT_W/2,1333*S),int(40*S),COLORS["cyan"]+(a,),True,True,"ma",1)
        draw_text(image,"BEAM CROSSES EARTH",(OUT_W/2,1402*S),int(44*S),COLORS["white"]+(a,),True,True,"ma",2)
        draw_text(image,"THAT SWEEP IS THE 'PULSE'",(OUT_W/2,1486*S),int(25*S),COLORS["cyan"]+(a,),True,True,"ma",1)
        if p>0.62:
            aa=int(245*smoothstep((p-0.62)/0.24))
            draw_text(image,"A COSMIC LIGHTHOUSE",(OUT_W/2,1602*S),int(40*S),COLORS["amber"]+(aa,),True,True,"ma",2)

    def render_frame(self,t:float)->np.ndarray:
        shot=get_shot(t)
        name=str(shot["name"])
        image=self.background(t)
        if name=="cold_open":
            self.draw_cold_open(image,t,shot)
        elif name=="collapse":
            self.draw_collapse(image,t,shot)
        elif name=="lighthouse":
            self.draw_lighthouse(image,t,shot)
        elif name=="pulse_train":
            self.draw_pulse_train(image,t,shot)
        elif name=="fastest":
            self.draw_fastest(image,t,shot)
        else:
            self.draw_definition(image,t,shot)

        self.draw_header(image,t,name)
        self.draw_footer(image)
        return apply_finish(image,t)


# -----------------------------------------------------------------------------
# Data products
# -----------------------------------------------------------------------------

def save_data_products() -> Tuple[Path,Path]:
    csv_path=DATA_ROOT/"pulsar_science_facts.csv"
    json_path=DATA_ROOT/"pulsar_short_summary.json"
    pd.DataFrame(FACTS).to_csv(csv_path,index=False)
    payload={
        "title": CONFIG["title_1"]+" "+CONFIG["title_2"],
        "visualization_type": "schematic educational animation",
        "pulse_profile_source": PROFILE_SOURCE,
        "fastest_known_pulsar": {"name":"PSR J1748-2446ad","rotations_per_second":716},
        "important_note": "The star, magnetic field, beam geometry, supernova, and pulse train are illustrative; they are not raw telescope imagery.",
        "sources": SOURCE_URLS,
        "facts": FACTS,
    }
    json_path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return csv_path,json_path


# -----------------------------------------------------------------------------
# Audio
# -----------------------------------------------------------------------------

def gaussian(times:np.ndarray,center:float,width:float)->np.ndarray:
    return np.exp(-0.5*((times-center)/max(width,1e-6))**2)


def generate_soundtrack(path:Path)->Path:
    sr=int(CONFIG["soundtrack_sample_rate"])
    duration=float(CONFIG["duration_s"])
    n=int(round(sr*duration))
    tt=np.arange(n,dtype=np.float64)/sr
    rng=np.random.default_rng(716)
    audio=np.zeros(n,dtype=np.float64)

    # Deep cinematic bed.
    audio += 0.055*np.sin(math.tau*31.0*tt + 0.35*np.sin(math.tau*0.06*tt))
    audio += 0.035*np.sin(math.tau*47.0*tt + 0.8)
    audio += 0.018*np.sin(math.tau*79.0*tt + 1.4)
    controls=rng.normal(0,1,max(8,int(duration*4)))
    noise=np.interp(tt,np.linspace(0,duration,len(controls)),controls)
    audio += 0.018*noise

    # Cold-open pulses.
    cold=next(s for s in SHOT_PLAN if s["name"]=="cold_open")
    pulse_period=(0.92 if not QUICK_MODE else 0.46)
    x=float(cold["start"])+0.5*pulse_period
    while x<float(cold["end"]):
        env=gaussian(tt,x,0.028 if not QUICK_MODE else 0.018)
        audio += env*(0.16*np.sin(math.tau*820*tt)+0.10*np.sin(math.tau*1240*tt))
        x += pulse_period

    # Collapse impact.
    collapse=next(s for s in SHOT_PLAN if s["name"]=="collapse")
    c=lerp(float(collapse["start"]),float(collapse["end"]),0.63)
    env=gaussian(tt,c,0.55 if not QUICK_MODE else 0.18)
    audio += env*(0.10*np.sin(math.tau*48*tt)+0.08*np.sin(math.tau*73*tt))

    # Lighthouse clicks.
    light=next(s for s in SHOT_PLAN if s["name"]=="lighthouse")
    for frac in np.linspace(0.08,0.92,10 if not QUICK_MODE else 5):
        c=lerp(float(light["start"]),float(light["end"]),float(frac))
        env=gaussian(tt,c,0.018 if not QUICK_MODE else 0.014)
        audio += env*(0.11*np.sin(math.tau*1080*tt)+0.05*np.sin(math.tau*1620*tt))

    # Faster pulse train.
    train=next(s for s in SHOT_PLAN if s["name"]=="pulse_train")
    count=24 if not QUICK_MODE else 8
    for c in np.linspace(float(train["start"])+0.18,float(train["end"])-0.15,count):
        env=gaussian(tt,float(c),0.012 if not QUICK_MODE else 0.009)
        audio += env*0.095*np.sin(math.tau*1450*tt)

    # Fastest reveal swell.
    fast=next(s for s in SHOT_PLAN if s["name"]=="fastest")
    c=lerp(float(fast["start"]),float(fast["end"]),0.72)
    env=gaussian(tt,c,0.8 if not QUICK_MODE else 0.20)
    audio += env*(0.13*np.sin(math.tau*62*tt)+0.08*np.sin(math.tau*124*tt))

    intro=np.clip(tt/max(1.2,duration*0.05),0,1)
    intro=intro*intro*(3-2*intro)
    outro_start=max(0,duration-1.4)
    outro=np.clip((duration-tt)/max(duration-outro_start,1e-6),0,1)
    outro=outro*outro*(3-2*outro)
    audio*=intro*outro
    peak=max(float(np.max(np.abs(audio))),1e-9)
    audio=np.clip(audio/peak*0.88,-1,1)
    pcm=(audio*32767).astype(np.int16)
    with wave.open(str(path),"wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(sr); handle.writeframes(pcm.tobytes())
    return path


def find_ffmpeg()->Optional[str]:
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return shutil.which("ffmpeg")


def mux_audio(video_path:Path,audio_path:Path,output_path:Path)->bool:
    ffmpeg=find_ffmpeg()
    if not ffmpeg:
        return False
    command=[ffmpeg,"-y","-i",str(video_path),"-i",str(audio_path),"-c:v","copy","-c:a","aac","-b:a","192k","-shortest",str(output_path)]
    try:
        subprocess.run(command,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return output_path.exists() and output_path.stat().st_size>0
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def render_video(scene:PulsarScene)->Path:
    srt_path=OUTPUT_ROOT/f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS,srt_path)
    raw_video=OUTPUT_ROOT/f"{CONFIG['output_basename']}_silent.mp4"
    final_video=OUTPUT_ROOT/f"{CONFIG['output_basename']}_final.mp4"
    audio_path=OUTPUT_ROOT/f"{CONFIG['output_basename']}_cinematic.wav"
    frame_count=int(round(float(CONFIG["duration_s"])*int(CONFIG["fps"])))
    times=np.arange(frame_count)/int(CONFIG["fps"])

    print("Subtitle sidecar:",srt_path.resolve())
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(raw_video,fps=int(CONFIG["fps"]),codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=None) as writer:
        for t in tqdm(times,desc="Rendering pulsar short"):
            writer.append_data(scene.render_frame(float(t)))

    if NO_AUDIO:
        shutil.copyfile(raw_video,final_video)
        return final_video
    generate_soundtrack(audio_path)
    if mux_audio(raw_video,audio_path,final_video):
        print("Final video with audio:",final_video.resolve())
        return final_video
    shutil.copyfile(raw_video,final_video)
    print("ffmpeg audio mux unavailable; copied silent video to:",final_video.resolve())
    return final_video


def main():
    print("Title:",CONFIG["title_1"],CONFIG["title_2"])
    print("Pulse profile:",PROFILE_SOURCE)
    csv_path,json_path=save_data_products()
    print("Facts CSV:",csv_path.resolve())
    print("Summary JSON:",json_path.resolve())

    scene=PulsarScene()
    preview_times=[
        1.2,
        min(10.0,float(CONFIG["duration_s"])*0.22),
        min(22.0,float(CONFIG["duration_s"])*0.42),
        min(35.0,float(CONFIG["duration_s"])*0.64),
        min(47.0,float(CONFIG["duration_s"])*0.82),
        max(0.2,float(CONFIG["duration_s"])-0.7),
    ]
    for preview_time in tqdm(preview_times,desc="Preview frames"):
        frame=scene.render_frame(float(preview_time))
        Image.fromarray(frame).save(PREVIEW_DIR/f"preview_{int(preview_time):02d}s.png")
    render_video(scene)
    print("Output directory:",OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-",path.name)


if __name__=="__main__":
    main()
