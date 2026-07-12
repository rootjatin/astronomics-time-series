# %% [markdown]
# # TON 618 Gravitational Lensing — Cinematic YouTube Short
#
# This script creates a vertical 1080×1920 astronomy short about TON 618,
# an ultramassive quasar black-hole system often discussed among the most
# massive known black holes.
#
# Visual language:
# - deep-space star field and volumetric nebula haze,
# - camera drift and slow push-ins,
# - a dark black-hole shadow surrounded by an emissive photon-ring motif,
# - a tilted, rotating accretion disk with a lensed far-side arc,
# - point-lens remapping of background stars into paired images / tangential arcs,
# - cinematic light-ray tracks bending around the central mass,
# - mass and Schwarzschild-scale HUD callouts,
# - bloom, chromatic glow, scan noise, vignette, and subtitles.
#
# Scientific fidelity note:
# The physical scale readouts are computed from a configurable representative
# black-hole mass using the Schwarzschild relation r_s = 2GM/c^2.
#
# The star remapping uses the standard point-mass thin-lens image equation as a
# readable visual explanation of gravitational lensing. The near-shadow photon
# ring and accretion-disk appearance are stylized cinematic motifs, NOT a full
# general-relativistic radiative-transfer or Kerr ray-tracing calculation.
#
# Recommended install:
#
#     pip install numpy pillow imageio imageio-ffmpeg tqdm
#
# Quick test render:
#
#     CONFIG["fps"] = 10
#     CONFIG["duration_s"] = 10
#     CONFIG["video_width"] = 540
#     CONFIG["video_height"] = 960
#     CONFIG["render_scale"] = 1.0
#
# References informing the wording / visual note:
# - Shemmer et al. 2004, ApJ 614, 547 (H-beta quasar spectroscopy / BH mass proxy)
# - Ge et al. 2019, AJ 157, 148 (C IV mass-proxy bias and blueshift)
# - Gralla, Holz & Wald 2019, Phys. Rev. D 100, 024018 (shadows/rings/lensing)
# - Event Horizon Telescope Collaboration 2019, ApJL 875, L6 (lensed emission near shadow)

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# %% [markdown]
# ## Configuration

OUTPUT_ROOT = Path("ton618_gravitational_lensing_short_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"

for directory in [OUTPUT_ROOT, PREVIEW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CONFIG = {
    # Final delivery
    "video_width": 1080,
    "video_height": 1920,
    "render_scale": 0.50,  # render internally smaller, then upscale for speed
    "fps": 24,
    "duration_s": 58,
    "output_basename": "ton618_gravitational_lensing_short",

    # Representative source parameters
    # Keep the copy cautious: virial black-hole mass estimates are model-dependent.
    "object_name": "TON 618",
    "object_label": "ULTRAMASSIVE QUASAR BLACK HOLE",
    "mass_solar": 40.7e9,
    "mass_display": "≈ 40 BILLION SOLAR MASSES",
    "mass_qualifier": "representative published single-epoch mass scale",

    # Physical constants
    "G": 6.67430e-11,
    "c": 299_792_458.0,
    "solar_mass_kg": 1.98847e30,
    "au_m": 149_597_870_700.0,

    # Scene composition in full-resolution coordinates; helpers scale internally.
    "shadow_radius_px": 174,
    "photon_ring_radius_px": 218,
    "disk_outer_radius_px": 470,
    "einstein_radius_px": 250,
    "star_count": 760,
    "dust_count": 330,
    "hud_noise_count": 70,

    # Grade / finish
    "contrast_boost": 1.15,
    "saturation_boost": 1.14,
    "vignette_strength": 0.31,

    # Text
    "title_text": "THE LIGHT-BENDING MONSTER",
    "subtitle_text": "TON 618 // one of the most massive known black-hole systems",
    "credit_text": "Cinematic visualization // physical scale from rₛ = 2GM/c²",
    "scientific_note": (
        "Background-star lensing uses a point-mass thin-lens model. "
        "Near-shadow rings and disk warping are stylized, not full GR ray tracing."
    ),

    # Optional finishing
    "audio_path": None,
    "burn_subtitles": False,
    "write_subtitle_sidecar": True,
}

OUT_SIZE = (CONFIG["video_width"], CONFIG["video_height"])
INTERNAL_SIZE = (
    max(2, int(round(CONFIG["video_width"] * CONFIG["render_scale"]))),
    max(2, int(round(CONFIG["video_height"] * CONFIG["render_scale"]))),
)
SCALE = INTERNAL_SIZE[0] / OUT_SIZE[0]


# %% [markdown]
# ## Physical scale helpers


def physical_scales(config: Dict) -> Dict[str, float]:
    mass_kg = float(config["mass_solar"]) * float(config["solar_mass_kg"])
    gravitational_radius_m = config["G"] * mass_kg / (config["c"] ** 2)
    schwarzschild_radius_m = 2.0 * gravitational_radius_m
    critical_impact_parameter_m = 3.0 * math.sqrt(3.0) * gravitational_radius_m

    return {
        "mass_kg": mass_kg,
        "gravitational_radius_m": gravitational_radius_m,
        "schwarzschild_radius_m": schwarzschild_radius_m,
        "schwarzschild_radius_au": schwarzschild_radius_m / config["au_m"],
        "horizon_diameter_au": 2.0 * schwarzschild_radius_m / config["au_m"],
        "critical_impact_parameter_au": critical_impact_parameter_m / config["au_m"],
        "shadow_diameter_scale_au": 2.0 * critical_impact_parameter_m / config["au_m"],
    }


SCALES = physical_scales(CONFIG)


# %% [markdown]
# ## General helpers


def sx(value: float) -> int:
    return int(round(value * SCALE))


def sxf(value: float) -> float:
    return float(value * SCALE)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t: float) -> float:
    t = clamp(t)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def ease_in_out_sine(t: float) -> float:
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


def rgba(color: Sequence[int], alpha: int) -> Tuple[int, int, int, int]:
    return int(color[0]), int(color[1]), int(color[2]), int(alpha)


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
    size = max(8, sx(size))
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue

    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[float, float],
    size: int = 42,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    stroke: int = 2,
    anchor: str = "la",
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)

    draw.text(
        (sxf(xy[0]), sxf(xy[1])),
        text,
        font=font,
        fill=fill,
        anchor=anchor,
        stroke_width=max(0, sx(stroke)),
        stroke_fill=(0, 0, 0, min(fill[3] if len(fill) > 3 else 255, 225)),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[float, float],
    max_width: int,
    size: int = 31,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    line_spacing: int = 8,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
    words = text.split()
    max_width_px = sx(max_width)

    lines: List[str] = []
    current = ""

    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=sx(2))

        if bbox[2] - bbox[0] <= max_width_px:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    x, y = sxf(xy[0]), sxf(xy[1])

    for line in lines:
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
            stroke_width=sx(2),
            stroke_fill=(0, 0, 0, 220),
        )
        bbox = draw.textbbox((x, y), line, font=font, stroke_width=sx(2))
        y += (bbox[3] - bbox[1]) + sx(line_spacing)


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cx = width / 2.0
    cy = height / 2.0
    nx = (xx - cx) / (width / 2.0)
    ny = (yy - cy) / (height / 2.0)
    radius = np.sqrt(nx**2 + ny**2)

    return np.clip(1.0 - strength * radius**1.8, 0.0, 1.0).astype(np.float32)


def apply_grade(arr: np.ndarray) -> np.ndarray:
    image = Image.fromarray(arr)
    image = ImageEnhance.Contrast(image).enhance(CONFIG["contrast_boost"])
    image = ImageEnhance.Color(image).enhance(CONFIG["saturation_boost"])
    return np.array(image)


VIGNETTE = make_vignette(
    INTERNAL_SIZE[0],
    INTERNAL_SIZE[1],
    CONFIG["vignette_strength"],
)


# %% [markdown]
# ## Cinematic timeline

CAPTIONS = [
    (0.5, 6.0, "This is TON 618 — a quasar powered by an ultramassive black hole."),
    (6.4, 15.2, "At roughly forty billion solar masses, even its horizon scale is enormous."),
    (15.7, 27.8, "Gravity curves spacetime. Background starlight reaches us along bent paths."),
    (28.3, 39.7, "A single source can appear twice — then stretch into arcs near alignment."),
    (40.2, 49.7, "Near the shadow, light can orbit, lens and pile into razor-thin ring structure."),
    (50.0, 57.2, "The exact close-up is stylized. The scale of the gravity is not."),
]

SHOT_PLAN = [
    {
        "name": "reveal",
        "start": 0.0,
        "end": 7.0,
        "cx0": 560,
        "cx1": 535,
        "cy0": 1030,
        "cy1": 980,
        "zoom0": 0.64,
        "zoom1": 0.82,
        "caption": "TON 618 // ULTRAMASSIVE GRAVITY",
    },
    {
        "name": "mass",
        "start": 7.0,
        "end": 16.0,
        "cx0": 535,
        "cx1": 520,
        "cy0": 980,
        "cy1": 930,
        "zoom0": 0.82,
        "zoom1": 1.00,
        "caption": "MASS SCALE // BILLIONS OF SUNS",
    },
    {
        "name": "bend",
        "start": 16.0,
        "end": 29.0,
        "cx0": 520,
        "cx1": 585,
        "cy0": 930,
        "cy1": 965,
        "zoom0": 1.00,
        "zoom1": 1.08,
        "caption": "SPACETIME CURVATURE // LIGHT DEFLECTED",
    },
    {
        "name": "lens",
        "start": 29.0,
        "end": 41.0,
        "cx0": 585,
        "cx1": 500,
        "cy0": 965,
        "cy1": 1015,
        "zoom0": 1.08,
        "zoom1": 1.18,
        "caption": "GRAVITATIONAL LENS // PAIRED IMAGES",
    },
    {
        "name": "ring",
        "start": 41.0,
        "end": 50.0,
        "cx0": 500,
        "cx1": 545,
        "cy0": 1015,
        "cy1": 955,
        "zoom0": 1.18,
        "zoom1": 1.38,
        "caption": "SHADOW EDGE // RING STRUCTURE",
    },
    {
        "name": "outro",
        "start": 50.0,
        "end": CONFIG["duration_s"],
        "cx0": 545,
        "cx1": 540,
        "cy0": 955,
        "cy1": 1025,
        "zoom0": 1.38,
        "zoom1": 0.72,
        "caption": "TON 618 // THE UNIVERSE BENDS AROUND MASS",
    },
]

CHAPTERS = [
    ("MASS", 0.0, 16.0),
    ("BENDING", 16.0, 29.0),
    ("LENS", 29.0, 41.0),
    ("RINGS", 41.0, 50.0),
    ("SCALE", 50.0, CONFIG["duration_s"]),
]


def get_shot(t: float) -> Dict:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def shot_state(t: float):
    shot = get_shot(t)
    duration = max(shot["end"] - shot["start"], 1e-6)
    u = clamp((t - shot["start"]) / duration)
    e = ease_in_out_sine(u)

    cx = lerp(shot["cx0"], shot["cx1"], e)
    cy = lerp(shot["cy0"], shot["cy1"], e)
    zoom = lerp(shot["zoom0"], shot["zoom1"], e)

    # Tiny handheld / telescope drift; deterministic and sub-pixel at full output.
    cx += 4.0 * math.sin(t * 0.31) + 1.7 * math.sin(t * 0.79)
    cy += 5.0 * math.cos(t * 0.27) + 1.5 * math.sin(t * 0.63)

    return shot, cx, cy, zoom


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


# %% [markdown]
# ## Scene renderer

class BlackHoleLensingScene:
    def __init__(self):
        self.stars = self._make_stars(CONFIG["star_count"], seed=618)
        self.dust = self._make_dust(CONFIG["dust_count"], seed=1407)
        self.hud_noise = self._make_hud_noise(CONFIG["hud_noise_count"], seed=922)
        self.ray_offsets = np.array([-390, -275, -180, -105, -48, 48, 105, 180, 275, 390], dtype=float)

    @staticmethod
    def _make_stars(count: int, seed: int):
        rng = np.random.default_rng(seed)
        stars = []

        palette = [
            (195, 220, 255),
            (225, 235, 255),
            (255, 245, 220),
            (255, 218, 180),
            (165, 205, 255),
        ]

        for _ in range(count):
            stars.append({
                "x": float(rng.uniform(-80, OUT_SIZE[0] + 80)),
                "y": float(rng.uniform(-80, OUT_SIZE[1] + 80)),
                "radius": float(rng.uniform(0.55, 2.05)),
                "alpha": int(rng.integers(42, 210)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "drift_x": float(rng.uniform(-5.5, 5.5)),
                "drift_y": float(rng.uniform(-2.2, 2.2)),
                "color": palette[int(rng.integers(0, len(palette)))],
                "arc_bias": float(rng.uniform(0.65, 1.35)),
            })

        return stars

    @staticmethod
    def _make_dust(count: int, seed: int):
        rng = np.random.default_rng(seed)
        dust = []

        for _ in range(count):
            dust.append({
                "x": float(rng.uniform(0, OUT_SIZE[0])),
                "y": float(rng.uniform(0, OUT_SIZE[1])),
                "r": float(rng.uniform(0.7, 3.2)),
                "alpha": int(rng.integers(10, 52)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "speed": float(rng.uniform(1.0, 7.0)),
            })

        return dust

    @staticmethod
    def _make_hud_noise(count: int, seed: int):
        rng = np.random.default_rng(seed)
        items = []

        for _ in range(count):
            items.append({
                "x": float(rng.uniform(0, OUT_SIZE[0])),
                "y": float(rng.uniform(0, OUT_SIZE[1])),
                "length": float(rng.uniform(8, 100)),
                "alpha": int(rng.integers(10, 48)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            })

        return items

    def render_background(self, t: float) -> Image.Image:
        canvas = Image.new("RGBA", INTERNAL_SIZE, (1, 2, 9, 255))

        haze = Image.new("RGBA", INTERNAL_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(haze)

        nebulae = [
            (180, 400, 590, (22, 45, 128), 24),
            (870, 760, 620, (75, 20, 100), 23),
            (430, 1370, 720, (8, 80, 105), 20),
            (830, 1660, 460, (70, 28, 58), 15),
        ]

        for cx, cy, radius, color, alpha in nebulae:
            wobble_x = 32 * math.sin(t * 0.035 + cx * 0.01)
            wobble_y = 24 * math.cos(t * 0.041 + cy * 0.01)
            r = sxf(radius)
            x = sxf(cx + wobble_x)
            y = sxf(cy + wobble_y)
            draw.ellipse(
                (x - r, y - r * 0.62, x + r, y + r * 0.62),
                fill=(color[0], color[1], color[2], alpha),
            )

        haze = haze.filter(ImageFilter.GaussianBlur(sx(105)))
        canvas.alpha_composite(haze)

        # Sparse drifting dust before the lensed star layer.
        dust_layer = Image.new("RGBA", INTERNAL_SIZE, (0, 0, 0, 0))
        dd = ImageDraw.Draw(dust_layer)

        for item in self.dust:
            x = (item["x"] + t * item["speed"] * 0.45) % OUT_SIZE[0]
            y = (item["y"] + t * item["speed"] * 0.12) % OUT_SIZE[1]
            pulse = 0.55 + 0.45 * math.sin(t * 0.45 + item["phase"])
            radius = sxf(item["r"])
            dd.ellipse(
                (
                    sxf(x) - radius,
                    sxf(y) - radius,
                    sxf(x) + radius,
                    sxf(y) + radius,
                ),
                fill=(150, 185, 225, int(item["alpha"] * pulse)),
            )

        dust_layer = dust_layer.filter(ImageFilter.GaussianBlur(sx(1.3)))
        canvas.alpha_composite(dust_layer)
        return canvas

    @staticmethod
    def point_lens_images(beta: float, theta_e: float):
        """Return absolute image radii and signed magnifications for a point lens."""
        beta = max(float(beta), 1e-5)
        theta_e = max(float(theta_e), 1e-5)
        root = math.sqrt(beta * beta + 4.0 * theta_e * theta_e)
        theta_plus = 0.5 * (beta + root)
        theta_minus = 0.5 * (root - beta)

        u = beta / theta_e
        mu_total = (u * u + 2.0) / max(u * math.sqrt(u * u + 4.0), 1e-6)
        mu_plus = 0.5 * (1.0 + mu_total)
        mu_minus = 0.5 * max(mu_total - 1.0, 0.0)
        return theta_plus, theta_minus, mu_plus, mu_minus

    def draw_star_image(
        self,
        sharp: ImageDraw.ImageDraw,
        glow: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        radius_from_lens: float,
        angle: float,
        magnification: float,
        star: Dict,
        alpha: int,
        opposite: bool = False,
    ):
        if opposite:
            angle += math.pi

        cx, cy = center
        x = cx + math.cos(angle) * radius_from_lens
        y = cy + math.sin(angle) * radius_from_lens

        color = star["color"]
        base_radius = max(0.55, star["radius"] * SCALE)
        brightness = min(3.2, 0.75 + math.sqrt(max(magnification, 0.0)))
        final_alpha = int(np.clip(alpha * min(2.0, brightness), 0, 255))

        stretch = np.clip(math.sqrt(max(magnification, 1.0)) * star["arc_bias"], 1.0, 8.0)

        if stretch >= 2.0 and radius_from_lens > sxf(8):
            # Tangential arc around the lens center.
            span = float(np.clip(0.9 + stretch * 1.9, 1.0, 16.0))
            deg = math.degrees(angle)
            bbox = (
                cx - radius_from_lens,
                cy - radius_from_lens,
                cx + radius_from_lens,
                cy + radius_from_lens,
            )
            glow.arc(
                bbox,
                start=deg - span,
                end=deg + span,
                fill=(color[0], color[1], color[2], int(final_alpha * 0.34)),
                width=max(2, sx(7)),
            )
            sharp.arc(
                bbox,
                start=deg - span,
                end=deg + span,
                fill=(color[0], color[1], color[2], final_alpha),
                width=max(1, sx(1.6 + star["radius"])),
            )
        else:
            r = base_radius * np.clip(brightness, 0.8, 2.5)
            glow.ellipse(
                (x - r * 3.6, y - r * 3.6, x + r * 3.6, y + r * 3.6),
                fill=(color[0], color[1], color[2], int(final_alpha * 0.25)),
            )
            sharp.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=(color[0], color[1], color[2], final_alpha),
            )

    def draw_lensed_stars(
        self,
        canvas: Image.Image,
        t: float,
        center: Tuple[float, float],
        zoom: float,
    ):
        sharp_layer = Image.new("RGBA", INTERNAL_SIZE, (0, 0, 0, 0))
        glow_layer = Image.new("RGBA", INTERNAL_SIZE, (0, 0, 0, 0))
        sharp = ImageDraw.Draw(sharp_layer)
        glow = ImageDraw.Draw(glow_layer)

        cx_full = center[0] / SCALE
        cy_full = center[1] / SCALE
        theta_e_full = CONFIG["einstein_radius_px"] * zoom
        theta_e = sxf(theta_e_full)

        lens_mix = 0.30 + 0.70 * smoothstep((t - 13.0) / 8.0)
        lens_mix *= 1.0 - 0.20 * smoothstep((t - 51.0) / 5.0)

        for star in self.stars:
            source_x = star["x"] + star["drift_x"] * t
            source_y = star["y"] + star["drift_y"] * t

            dx_full = source_x - cx_full
            dy_full = source_y - cy_full
            beta_full = max(math.hypot(dx_full, dy_full), 1e-5)
            angle = math.atan2(dy_full, dx_full)
            beta = sxf(beta_full)

            twinkle = 0.68 + 0.32 * math.sin(t * 1.15 + star["phase"])
            alpha = int(star["alpha"] * twinkle)

            theta_plus, theta_minus, mu_plus, mu_minus = self.point_lens_images(beta, theta_e)

            # Interpolate from unlensed source radius to lensed image radius.
            plus_radius = lerp(beta, theta_plus, lens_mix)
            self.draw_star_image(
                sharp,
                glow,
                center,
                plus_radius,
                angle,
                lerp(1.0, mu_plus, lens_mix),
                star,
                alpha,
                opposite=False,
            )

            # Secondary image becomes visible only once lensing is explained.
            secondary_alpha = int(alpha * lens_mix * np.clip(mu_minus * 2.2, 0.0, 0.92))
            if secondary_alpha > 4 and beta_full < theta_e_full * 5.0:
                self.draw_star_image(
                    sharp,
                    glow,
                    center,
                    theta_minus,
                    angle,
                    max(mu_minus, 0.02),
                    star,
                    secondary_alpha,
                    opposite=True,
                )

        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(sx(4.5)))
        canvas.alpha_composite(glow_layer)
        canvas.alpha_composite(sharp_layer)

    def draw_polar_haze(
        self,
        canvas: Image.Image,
        center: Tuple[float, float],
        zoom: float,
        t: float,
    ):
        layer = Image.new("RGBA", INTERNAL_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx, cy = center

        strength = 0.55 + 0.45 * smoothstep((t - 5.0) / 8.0)
        outer = sxf(650 * zoom)
        half = sxf(90 * zoom)

        for direction in [-1, 1]:
            end_y = cy + direction * outer
            polygon = [
                (cx - half * 0.32, cy),
                (cx - half, end_y),
                (cx + half, end_y),
                (cx + half * 0.32, cy),
            ]
            draw.polygon(polygon, fill=(35, 105, 165, int(12 * strength)))

        layer = layer.filter(ImageFilter.GaussianBlur(sx(55)))
        canvas.alpha_composite(layer)

    def draw_bending_rays(
        self,
        canvas: Image.Image,
        t: float,
        center: Tuple[float, float],
        zoom: float,
    ):
        reveal = smoothstep((t - 15.0) / 4.0) * (1.0 - smoothstep((t - 43.0) / 5.0))
        if reveal <= 0.01:
            return

        layer = Image.new("RGBA", INTERNAL_SIZE, (0, 0, 0, 0))
        glow_layer = Image.new("RGBA", INTERNAL_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        glow = ImageDraw.Draw(glow_layer)

        cx, cy = center
        ring = sxf(CONFIG["photon_ring_radius_px"] * zoom)
        travel = sxf(720)

        for index, offset_full in enumerate(self.ray_offsets):
            offset = sxf(offset_full * zoom * 0.76)
            sign = 1 if offset >= 0 else -1
            closest = max(abs(offset), ring * 1.10)
            bend_strength = ring * ring / max(closest, 1.0)
            phase = (t * 0.19 + index * 0.13) % 1.0

            points = []
            samples = 84
            for j in range(samples):
                u = j / (samples - 1)
                x = cx - travel + 2.0 * travel * u
                dx = x - cx
                attract = math.exp(-((dx / (sxf(260) * max(zoom, 0.6))) ** 2))
                y = cy + offset - sign * bend_strength * 0.65 * attract
                # Small wave travels along the beam to make it feel alive.
                y += sxf(3.5) * math.sin(u * 15.0 - t * 3.0 + index) * reveal
                points.append((x, y))

            alpha = int((32 + 38 * (0.5 + 0.5 * math.sin(index * 1.7))) * reveal)
            glow.line(points, fill=(70, 195, 255, int(alpha * 0.42)), width=max(2, sx(8)))
            draw.line(points, fill=(125, 225, 255, alpha), width=max(1, sx(1.4)))

            # A bright photon packet moves along a subset of the path.
            if index % 2 == 0:
                packet_index = int(phase * (samples - 1))
                px, py = points[packet_index]
                rr = sxf(3.2)
                draw.ellipse((px - rr, py - rr, px + rr, py + rr), fill=(245, 252, 255, int(210 * reveal)))

        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(sx(8)))
        canvas.alpha_composite(glow_layer)
        canvas.alpha_composite(layer)

    def draw_accretion_disk(
        self,
        canvas: Image.Image,
        t: float,
        center: Tuple[float, float],
        zoom: float,
    ):
        cx, cy = center
        shadow_r = sxf(CONFIG["shadow_radius_px"] * zoom)
        ring_r = sxf(CONFIG["photon_ring_radius_px"] * zoom)
        outer_r = sxf(CONFIG["disk_outer_radius_px"] * zoom)
        tilt = 0.24
        rotation = t * 0.52

        glow_layer = Image.new("RGBA", INTERNAL_SIZE, (0, 0, 0, 0))
        sharp_layer = Image.new("RGBA", INTERNAL_SIZE, (0, 0, 0, 0))
        glow = ImageDraw.Draw(glow_layer)
        sharp = ImageDraw.Draw(sharp_layer)

        ring_count = 24
        segments = 96

        for ring_index in range(ring_count):
            f = ring_index / max(ring_count - 1, 1)
            radius = lerp(ring_r * 1.08, outer_r, f ** 0.82)
            line_width = max(1, sx(1.1 + 2.2 * (1.0 - f)))

            hot = 1.0 - f
            base_color = (
                int(lerp(255, 205, f)),
                int(lerp(252, 72, f)),
                int(lerp(232, 28, f)),
            )

            points = []
            brightness = []
            for j in range(segments + 1):
                phi = 2.0 * math.pi * j / segments + rotation * (1.6 - 0.65 * f)
                x = cx + radius * math.cos(phi)
                y = cy + radius * tilt * math.sin(phi)
                points.append((x, y))

                # Editorial Doppler-like asymmetry: one side brighter.
                approach = 0.5 + 0.5 * math.cos(phi - 0.12)
                pulse = 0.86 + 0.14 * math.sin(t * 2.3 + ring_index * 0.61 + phi * 2.0)
                brightness.append((0.43 + 0.57 * approach) * pulse)

            for j in range(segments):
                b = brightness[j]
                alpha = int(np.clip((60 + 150 * hot) * b, 12, 235))
                p0 = points[j]
                p1 = points[j + 1]
                glow.line((p0, p1), fill=rgba(base_color, int(alpha * 0.40)), width=line_width * 5)
                sharp.line((p0, p1), fill=rgba(base_color, alpha), width=line_width)

        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(sx(13)))
        canvas.alpha_composite(glow_layer)
        canvas.alpha_composite(sharp_layer)

        # Opaque black shadow occludes the direct disk image.
        shadow_layer = Image.new("RGBA", INTERNAL_SIZE, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow_layer)

        # Very deep central depression with a subtle blue-black edge.
        sd.ellipse(
            (cx - shadow_r, cy - shadow_r, cx + shadow_r, cy + shadow_r),
            fill=(0, 0, 1, 255),
            outline=(9, 19, 34, 235),
            width=max(1, sx(3)),
        )
        canvas.alpha_composite(shadow_layer)

        # Stylized lensed image of the far side of the disk above and below the shadow.
        lens_layer = Image.new("RGBA", INTERNAL_SIZE, (0, 0, 0, 0))
        ld = ImageDraw.Draw(lens_layer)
        for scale, alpha, width in [
            (1.00, 220, 4.0),
            (1.045, 110, 6.0),
            (1.09, 48, 9.0),
        ]:
            rr = ring_r * scale
            bbox = (cx - rr, cy - rr, cx + rr, cy + rr)
            ld.arc(bbox, 188, 352, fill=(255, 224, 145, alpha), width=max(1, sx(width)))
            ld.arc(bbox, 8, 172, fill=(255, 120, 55, int(alpha * 0.66)), width=max(1, sx(width * 0.75)))

        # Thin photon-ring motif surrounding the shadow.
        pulse = 0.82 + 0.18 * math.sin(t * 2.0)
        photon_r = ring_r * 0.995
        ld.ellipse(
            (cx - photon_r, cy - photon_r, cx + photon_r, cy + photon_r),
            outline=(255, 244, 214, int(240 * pulse)),
            width=max(1, sx(2.0)),
        )

        lens_glow = lens_layer.filter(ImageFilter.GaussianBlur(sx(11)))
        canvas.alpha_composite(lens_glow)
        canvas.alpha_composite(lens_layer)

    def draw_mass_panel(self, canvas: Image.Image, t: float):
        alpha = int(235 * smoothstep((t - 5.8) / 2.5) * (1.0 - smoothstep((t - 18.5) / 3.0)))
        if alpha <= 4:
            return

        x0, y0, x1, y1 = 55, 260, 595, 500
        panel = Image.new("RGBA", INTERNAL_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle(
            (sx(x0), sx(y0), sx(x1), sx(y1)),
            radius=sx(24),
            fill=(3, 5, 14, int(alpha * 0.76)),
            outline=(110, 205, 255, int(alpha * 0.38)),
            width=max(1, sx(2)),
        )
        canvas.alpha_composite(panel)

        draw_text(canvas, "MASS SCALE", (x0 + 24, y0 + 22), size=20, fill=(100, 220, 255, alpha), bold=True, stroke=1)
        draw_text(canvas, CONFIG["mass_display"], (x0 + 24, y0 + 63), size=31, fill=(250, 252, 255, alpha), bold=True)
        draw_text(
            canvas,
            f"Schwarzschild radius ≈ {SCALES['schwarzschild_radius_au']:,.0f} AU",
            (x0 + 24, y0 + 122),
            size=24,
            fill=(255, 205, 120, alpha),
            bold=True,
            stroke=1,
        )
        draw_text(
            canvas,
            f"Horizon diameter ≈ {SCALES['horizon_diameter_au']:,.0f} AU",
            (x0 + 24, y0 + 164),
            size=22,
            fill=(205, 225, 245, alpha),
            stroke=1,
        )
        draw_text(
            canvas,
            CONFIG["mass_qualifier"],
            (x0 + 24, y0 + 203),
            size=16,
            fill=(160, 190, 215, alpha),
            stroke=1,
        )

    def draw_lens_equation_panel(self, canvas: Image.Image, t: float):
        alpha = int(225 * smoothstep((t - 20.0) / 3.5) * (1.0 - smoothstep((t - 40.5) / 4.0)))
        if alpha <= 4:
            return

        x0, y0, x1, y1 = 590, 250, 1025, 468
        panel = Image.new("RGBA", INTERNAL_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle(
            (sx(x0), sx(y0), sx(x1), sx(y1)),
            radius=sx(22),
            fill=(2, 5, 14, int(alpha * 0.72)),
            outline=(86, 193, 245, int(alpha * 0.35)),
            width=max(1, sx(1)),
        )
        canvas.alpha_composite(panel)

        draw_text(canvas, "POINT-LENS VISUAL MODEL", (x0 + 22, y0 + 20), size=19, fill=(115, 225, 255, alpha), bold=True, stroke=1)
        draw_text(canvas, "β = θ − θᴇ² / θ", (x0 + 22, y0 + 62), size=30, fill=(245, 250, 255, alpha), bold=True)
        draw_text(canvas, "one source → two image solutions", (x0 + 22, y0 + 118), size=21, fill=(255, 203, 125, alpha), bold=True, stroke=1)
        draw_wrapped_text(
            canvas,
            "Near alignment, magnification rises and point images stretch tangentially into arcs.",
            (x0 + 22, y0 + 155),
            max_width=390,
            size=17,
            fill=(190, 216, 238, alpha),
        )

    def draw_ring_callout(
        self,
        canvas: Image.Image,
        t: float,
        center: Tuple[float, float],
        zoom: float,
    ):
        alpha = int(235 * smoothstep((t - 39.0) / 4.0))
        if alpha <= 4:
            return

        cx, cy = center
        ring_r = sxf(CONFIG["photon_ring_radius_px"] * zoom)
        line_end = (sxf(910), sxf(585))
        anchor_angle = -0.68
        anchor = (
            cx + math.cos(anchor_angle) * ring_r,
            cy + math.sin(anchor_angle) * ring_r,
        )

        draw = ImageDraw.Draw(canvas)
        draw.line((anchor, line_end), fill=(255, 205, 115, alpha), width=max(1, sx(2)))
        rr = sxf(8)
        draw.ellipse((anchor[0] - rr, anchor[1] - rr, anchor[0] + rr, anchor[1] + rr), outline=(255, 230, 170, alpha), width=max(1, sx(2)))

        draw_text(canvas, "LENSED RING STRUCTURE", (1005, 520), size=22, fill=(255, 218, 140, alpha), bold=True, stroke=1, anchor="ra")
        draw_wrapped_text(
            canvas,
            "Strong light bending produces nested image structure around the shadow.",
            (615, 560),
            max_width=390,
            size=18,
            fill=(215, 230, 245, alpha),
        )

    def draw_scale_ruler(self, canvas: Image.Image, t: float):
        alpha = int(230 * smoothstep((t - 47.5) / 4.0))
        if alpha <= 4:
            return

        x0, x1 = 105, 975
        y = 1450
        draw = ImageDraw.Draw(canvas)
        draw.line((sx(x0), sx(y), sx(x1), sx(y)), fill=(100, 210, 245, alpha), width=max(1, sx(2)))

        divisions = 8
        horizon_diameter = SCALES["horizon_diameter_au"]
        for i in range(divisions + 1):
            x = lerp(x0, x1, i / divisions)
            tick = 18 if i in [0, divisions // 2, divisions] else 10
            draw.line((sx(x), sx(y - tick), sx(x), sx(y + tick)), fill=(100, 210, 245, alpha), width=max(1, sx(2)))

        draw_text(canvas, "EVENT-HORIZON DIAMETER SCALE", (x0, y - 68), size=21, fill=(150, 225, 245, alpha), bold=True, stroke=1)
        draw_text(canvas, "0 AU", (x0, y + 34), size=17, fill=(180, 205, 225, alpha), stroke=1)
        draw_text(canvas, f"{horizon_diameter / 2:,.0f} AU", ((x0 + x1) / 2, y + 34), size=17, fill=(255, 205, 120, alpha), bold=True, stroke=1, anchor="ma")
        draw_text(canvas, f"{horizon_diameter:,.0f} AU", (x1, y + 34), size=17, fill=(180, 205, 225, alpha), stroke=1, anchor="ra")

    def draw_progress_hud(self, canvas: Image.Image, t: float):
        x0 = 70
        x1 = OUT_SIZE[0] - 70
        y = OUT_SIZE[1] - 330
        width = x1 - x0
        draw = ImageDraw.Draw(canvas)

        draw.line((sx(x0), sx(y), sx(x1), sx(y)), fill=(75, 160, 205, 185), width=max(1, sx(2)))

        for label, start, end in CHAPTERS:
            chapter_x = x0 + width * start / CONFIG["duration_s"]
            draw.line((sx(chapter_x), sx(y - 11), sx(chapter_x), sx(y + 11)), fill=(90, 190, 230, 155), width=max(1, sx(1)))
            draw_text(canvas, label, (chapter_x, y + 29), size=14, fill=(135, 180, 205, 185), stroke=1)

        fraction = clamp(t / CONFIG["duration_s"])
        cursor_x = x0 + width * fraction
        draw.line((sx(cursor_x), sx(y - 27), sx(cursor_x), sx(y + 27)), fill=(255, 185, 90, 245), width=max(1, sx(3)))

        draw_text(canvas, "GRAVITY VISUALIZATION // SEQUENCE", (x0, y - 60), size=18, fill=(145, 210, 235, 205), bold=True, stroke=1)
        draw_text(canvas, f"{t:04.1f}s", (x1, y - 60), size=20, fill=(235, 245, 255, 225), bold=True, stroke=1, anchor="ra")

    def draw_corner_hud(self, canvas: Image.Image, t: float, shot: Dict):
        draw_text(canvas, CONFIG["object_name"], (OUT_SIZE[0] - 55, 85), size=25, fill=(125, 230, 255, 225), bold=True, stroke=1, anchor="ra")
        draw_text(canvas, shot["name"].upper(), (OUT_SIZE[0] - 55, 124), size=18, fill=(175, 205, 225, 205), bold=True, stroke=1, anchor="ra")
        draw_text(canvas, f"CURVATURE // {int(55 + 44 * math.sin(t * 0.17) ** 2):02d}", (OUT_SIZE[0] - 55, 158), size=17, fill=(165, 198, 220, 190), stroke=1, anchor="ra")

    def add_text_layers(self, canvas: Image.Image, t: float, shot: Dict):
        title_alpha = int(255 * smoothstep((t - 0.25) / 1.0) * (1.0 - smoothstep((t - 6.0) / 0.9)))

        if title_alpha > 4:
            draw_text(canvas, CONFIG["title_text"], (54, 102), size=48, fill=(245, 250, 255, title_alpha), bold=True)
            draw_text(canvas, CONFIG["subtitle_text"], (58, 170), size=22, fill=(110, 225, 250, min(title_alpha, 225)), bold=True)

        if t > 5.5:
            draw_text(canvas, shot["caption"], (54, 65), size=20, fill=(145, 215, 238, 205), bold=True, stroke=1)

        caption = caption_at(t)
        if caption:
            panel = Image.new("RGBA", INTERNAL_SIZE, (0, 0, 0, 0))
            draw = ImageDraw.Draw(panel)
            y0 = OUT_SIZE[1] - 245
            draw.rounded_rectangle(
                (sx(45), sx(y0), sx(OUT_SIZE[0] - 45), sx(y0 + 126)),
                radius=sx(24),
                fill=(1, 4, 12, 176),
                outline=(55, 180, 220, 72),
                width=max(1, sx(1)),
            )
            canvas.alpha_composite(panel)
            draw_wrapped_text(canvas, caption, (70, y0 + 26), max_width=OUT_SIZE[0] - 140, size=29, fill=(245, 250, 255, 245))

        note_alpha = int(220 * smoothstep((t - 50.0) / 3.0))
        if note_alpha > 4:
            draw_wrapped_text(canvas, CONFIG["credit_text"], (65, OUT_SIZE[1] - 112), max_width=940, size=18, fill=(220, 232, 245, note_alpha))
            draw_wrapped_text(canvas, CONFIG["scientific_note"], (65, OUT_SIZE[1] - 80), max_width=940, size=16, fill=(188, 210, 230, note_alpha))

    def draw_hud_noise(self, canvas: Image.Image, t: float):
        overlay = Image.new("RGBA", INTERNAL_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for item in self.hud_noise:
            pulse = 0.5 + 0.5 * math.sin(t * 1.8 + item["phase"])
            if pulse < 0.75:
                continue

            x = item["x"]
            y = (item["y"] + t * 10.0) % OUT_SIZE[1]
            draw.line(
                (sx(x), sx(y), sx(x + item["length"]), sx(y)),
                fill=(90, 210, 240, int(item["alpha"] * pulse)),
                width=max(1, sx(1)),
            )

        offset = int((t * 37) % 8)
        for y in range(offset, OUT_SIZE[1], 8):
            draw.line((0, sx(y), INTERNAL_SIZE[0], sx(y)), fill=(120, 205, 245, 11), width=max(1, sx(1)))

        scan_y = int((t * 148) % (OUT_SIZE[1] + 260)) - 130
        draw.rectangle((0, sx(scan_y), INTERNAL_SIZE[0], sx(scan_y + 50)), fill=(80, 210, 240, 8))
        canvas.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot, cx_full, cy_full, zoom = shot_state(t)
        center = (sxf(cx_full), sxf(cy_full))

        canvas = self.render_background(t)
        self.draw_lensed_stars(canvas, t, center, zoom)
        self.draw_polar_haze(canvas, center, zoom, t)
        self.draw_bending_rays(canvas, t, center, zoom)
        self.draw_accretion_disk(canvas, t, center, zoom)

        self.draw_mass_panel(canvas, t)
        self.draw_lens_equation_panel(canvas, t)
        self.draw_ring_callout(canvas, t, center, zoom)
        self.draw_scale_ruler(canvas, t)
        self.draw_progress_hud(canvas, t)
        self.draw_corner_hud(canvas, t, shot)
        self.add_text_layers(canvas, t, shot)
        self.draw_hud_noise(canvas, t)

        arr = np.array(canvas.convert("RGB"))
        arr = apply_grade(arr)
        arr = np.clip(arr.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)

        fade_in = smoothstep(t / 0.9)
        fade_out = 1.0 - smoothstep((t - (CONFIG["duration_s"] - 1.15)) / 1.0)
        arr = np.clip(arr.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)

        if INTERNAL_SIZE != OUT_SIZE:
            arr = np.array(
                Image.fromarray(arr).resize(OUT_SIZE, resample=Image.Resampling.LANCZOS)
            )

        return arr


# %% [markdown]
# ## Subtitle sidecar


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
    lines = []
    for index, (start, end, text) in enumerate(captions, start=1):
        lines.append(str(index))
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# %% [markdown]
# ## Video render and optional FFmpeg finishing


def run_ffmpeg(command: List[str]):
    print("Running:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def render_video(scene: BlackHoleLensingScene):
    raw_video_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    subbed_video_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_subbed.mp4"
    audio_video_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_with_audio.mp4"
    final_video_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"

    if CONFIG.get("write_subtitle_sidecar", True):
        write_srt(CAPTIONS, srt_path)
        print("Subtitle sidecar written:", srt_path.resolve())

    frame_count = int(round(CONFIG["duration_s"] * CONFIG["fps"]))
    times = np.arange(frame_count) / CONFIG["fps"]

    print(
        f"Rendering {frame_count:,} frames at "
        f"{CONFIG['video_width']}×{CONFIG['video_height']} "
        f"(internal {INTERNAL_SIZE[0]}×{INTERNAL_SIZE[1]}) ..."
    )

    with iio.get_writer(
        raw_video_path,
        fps=CONFIG["fps"],
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times, desc="Rendering TON 618 lensing short"):
            writer.append_data(scene.render_frame(float(t)))

    print("Raw video written:", raw_video_path.resolve())

    ffmpeg = find_ffmpeg()
    print("FFmpeg detected:", ffmpeg)
    final_candidate = raw_video_path

    if CONFIG.get("burn_subtitles", False) and ffmpeg and srt_path.exists():
        subtitle_filter = (
            f"subtitles={srt_path}:"
            "force_style=Fontname=DejaVu Sans,"
            "Fontsize=22,Outline=1.2,BorderStyle=3,MarginV=90"
        )
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(final_candidate),
            "-vf",
            subtitle_filter,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(subbed_video_path),
        ]
        run_ffmpeg(command)
        final_candidate = subbed_video_path
        print("Subtitled video written:", subbed_video_path.resolve())

    audio_path = CONFIG.get("audio_path")
    if audio_path and Path(audio_path).exists() and ffmpeg:
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(final_candidate),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(audio_video_path),
        ]
        run_ffmpeg(command)
        final_candidate = audio_video_path
        print("Audio-muxed video written:", audio_video_path.resolve())
    elif audio_path:
        print("audio_path was set, but the file was not found or FFmpeg was unavailable. Skipping audio.")

    if final_candidate.exists():
        shutil.copyfile(final_candidate, final_video_path)
        print("Final video:", final_video_path.resolve())

    return final_video_path


# %% [markdown]
# ## Main pipeline


def main():
    print("Starting TON 618 gravitational-lensing short pipeline ...")
    print(f"Representative mass scale: {CONFIG['mass_solar'] / 1e9:.1f} billion solar masses")
    print(f"Schwarzschild radius: {SCALES['schwarzschild_radius_au']:,.1f} AU")
    print(f"Horizon diameter: {SCALES['horizon_diameter_au']:,.1f} AU")
    print(f"Schwarzschild critical impact parameter: {SCALES['critical_impact_parameter_au']:,.1f} AU")

    scene = BlackHoleLensingScene()

    preview_times = [
        1.0,
        10.0,
        22.0,
        34.0,
        45.0,
        CONFIG["duration_s"] - 1.0,
    ]

    for preview_time in tqdm(preview_times, desc="Preview frames"):
        frame = scene.render_frame(float(preview_time))
        Image.fromarray(frame).save(PREVIEW_DIR / f"preview_{int(preview_time):02d}s.png")

    print("Preview frames written to:", PREVIEW_DIR.resolve())
    render_video(scene)

    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-", path.name)


if __name__ == "__main__":
    main()


# %% [markdown]
# ## Suggested narration
#
# This is TON 618 — a quasar powered by one of the most massive known black-hole systems.
# At a representative mass scale of about forty billion Suns, even the horizon spans an
# almost incomprehensible distance.
#
# But mass does something stranger than pull.
# It curves spacetime.
#
# Light from a star behind the black hole can reach us along more than one path.
# One source becomes two images.
# Near alignment, those images stretch into arcs.
#
# Closer to the shadow, light can loop around the black hole and build narrow ring structure.
# The exact close-up here is cinematic — not a full relativistic ray trace.
# But the central idea is real:
# around enough mass, even light changes direction.
#
# Suggested YouTube Shorts caption:
#
# TON 618 visualized as a cinematic gravitational lens. The horizon-scale readout is
# computed from r_s = 2GM/c² using a representative ~40.7-billion-solar-mass scale.
# Background stars use a point-mass lens equation; near-shadow ring and accretion-disk
# visuals are stylized rather than full GR radiative transfer.
#
# #BlackHole #TON618 #GravitationalLensing #Space #Astronomy #Python #ScienceVisualization
