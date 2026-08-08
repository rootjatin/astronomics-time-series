
from __future__ import annotations

"""

"""

import csv
import math
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("MARS_PRESSURE_SHORT_QUICK", "0") == "1"
FOUR_K_MODE = os.environ.get("MARS_PRESSURE_SHORT_4K", "0") == "1" and not QUICK_MODE
DATA_PATH = os.environ.get("MARS_PRESSURE_DATA_PATH", "").strip()

OUTPUT_ROOT = Path("the_real_pressure_changes_on_mars_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
DATA_DIR = OUTPUT_ROOT / "data"
for p in (OUTPUT_ROOT, PREVIEW_DIR, DATA_DIR):
    p.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
FPS = int(CONFIG["fps"])
DURATION = float(CONFIG["duration_s"])
FRAME_COUNT = int(round(FPS * DURATION))

COLORS = {
    "white": (244, 245, 240),
    "muted": (190, 196, 200),
    "gold": (245, 188, 80),
    "orange": (255, 129, 64),
    "blue": (104, 178, 255),
    "red": (255, 99, 70),
    "dark": (6, 7, 14),
    "mars1": (61, 20, 14),
    "mars2": (120, 43, 24),
    "mars3": (193, 88, 40),
    "mars4": (235, 146, 72),
}

FULL_CAPTIONS = [
    (0.0, 3.6, "Mars has weather — and even its air pressure rises and falls through the year."),
    (4.0, 7.6, "This is not one global number. It is Curiosity's pressure record at Gale Crater."),
    (14.0, 18.0, "At Gale, pressure can range from about 700 to 970 pascals across a Martian year."),
    (20.0, 24.0, "The poles drive the cycle: carbon dioxide freezes out in winter and pressure drops."),
    (27.0, 31.0, "When sunlight returns, that CO₂ sublimates back into the air — and pressure rises."),
    (33.0, 37.0, "Mars also has daily atmospheric tides: small pressure pulses every sol."),
    (39.0, 43.0, "Dust, temperature, and crater topography all nudge the pattern at Curiosity's site."),
    (45.0, 49.0, "So Mars' atmosphere does something beautiful: it breathes in a repeating seasonal rhythm."),
]

SHOT_PLAN_FULL = [
    ("intro", 0.0, 4.0),
    ("scope", 4.0, 8.0),
    ("dive", 8.0, 14.0),
    ("annual_curve", 14.0, 20.0),
    ("polar_drop", 20.0, 27.0),
    ("polar_rise", 27.0, 33.0),
    ("daily_tides", 33.0, 39.0),
    ("dust_local", 39.0, 45.0),
    ("finale", 45.0, 52.0),
]

if QUICK_MODE:
    scale = DURATION / 52.0
    CAPTIONS = [(a * scale, b * scale, text) for a, b, text in FULL_CAPTIONS]
    SHOT_PLAN = [(name, a * scale, b * scale) for name, a, b in SHOT_PLAN_FULL]
else:
    CAPTIONS = FULL_CAPTIONS
    SHOT_PLAN = SHOT_PLAN_FULL

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def smoothstep(x: float) -> float:
    x = clamp(x)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_shot(t: float) -> Tuple[str, float, float]:
    for shot in SHOT_PLAN:
        if shot[1] <= t < shot[2]:
            return shot
    return SHOT_PLAN[-1]


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    hh = ms // 3_600_000
    ms %= 3_600_000
    mm = ms // 60_000
    ms %= 60_000
    ss = ms // 1000
    ms %= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> None:
    lines: List[str] = []
    for i, (a, b, text) in enumerate(captions, start=1):
        lines.extend([str(i), f"{format_srt_time(a)} --> {format_srt_time(b)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(image: Image.Image, text: str, xy: Tuple[int, int], size: int, fill, bold=False, anchor="la", stroke=2):
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold=bold),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(fill[3], 220) if len(fill) > 3 else 220),
    )


def draw_wrapped_text(image: Image.Image, text: str, box: Tuple[int, int, int, int], size: int, fill, bold=False):
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
    words = text.split()
    lines: List[str] = []
    current = ""
    max_width = x1 - x0
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
    y = y0
    for line in lines:
        draw.text((x0, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 200))
        bb = draw.textbbox((x0, y), line, font=font, stroke_width=2)
        y += (bb[3] - bb[1]) + 8
        if y > y1:
            break


def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 140):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(box, radius=max(8, int((box[3] - box[1]) * 0.18)), fill=(6, 8, 18, alpha), outline=(240, 176, 72, 50), width=1)
    image.alpha_composite(overlay)


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    r = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * r ** 1.8, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


@dataclass
class Star:
    x: float
    y: float
    radius: float
    alpha: float
    phase: float


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def fallback_pressure_profile() -> Tuple[np.ndarray, np.ndarray, str]:
    """Broad seasonal Gale profile approximating published REMS annual range."""
    sol_frac = np.array([0.00, 0.08, 0.16, 0.25, 0.33, 0.41, 0.50, 0.58, 0.66, 0.75, 0.83, 0.91, 1.00], dtype=np.float32)
    pressure_pa = np.array([735, 710, 720, 760, 815, 880, 940, 970, 945, 900, 845, 785, 735], dtype=np.float32)
    return sol_frac, pressure_pa, "fallback_seasonal_profile_approximation"


def load_pressure_profile() -> Tuple[np.ndarray, np.ndarray, str, Path]:
    if DATA_PATH:
        path = Path(DATA_PATH).expanduser()
        xs: List[float] = []
        ys: List[float] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("CSV has no header")
            fieldnames = {name.lower().strip(): name for name in reader.fieldnames}
            x_key = fieldnames.get("sol_frac") or fieldnames.get("fraction") or fieldnames.get("x")
            y_key = fieldnames.get("pressure_pa") or fieldnames.get("pressure") or fieldnames.get("pa")
            if not x_key or not y_key:
                raise ValueError("CSV must include sol_frac and pressure_pa columns")
            for row in reader:
                try:
                    x = float(row[x_key])
                    y = float(row[y_key])
                except Exception:
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    xs.append(x)
                    ys.append(y)
            if len(xs) < 4:
                raise ValueError("CSV needs at least 4 valid rows")
        order = np.argsort(xs)
        x = np.asarray(xs, dtype=np.float32)[order]
        y = np.asarray(ys, dtype=np.float32)[order]
        x = np.clip(x, 0.0, 1.0)
        out_csv = DATA_DIR / "input_pressure_profile_copy.csv"
        shutil.copy2(path, out_csv)
        return x, y, f"csv:{path}", out_csv

    x, y, source = fallback_pressure_profile()
    out_csv = DATA_DIR / "fallback_pressure_profile.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sol_frac", "pressure_pa"])
        for a, b in zip(x, y):
            writer.writerow([float(a), float(b)])
    return x, y, source, out_csv


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------


class MarsPressureScene:
    def __init__(self, pressure_x: np.ndarray, pressure_y: np.ndarray, source_label: str):
        self.pressure_x = pressure_x
        self.pressure_y = pressure_y
        self.source_label = source_label
        self.stars = self._make_stars(int(CONFIG["background_stars"]), seed=77)
        self.terrain = self._make_terrain(seed=1307)
        self.pressure_min = float(np.min(pressure_y))
        self.pressure_max = float(np.max(pressure_y))
        self.curve_dense_x = np.linspace(0.0, 1.0, 512)
        self.curve_dense_y = np.interp(self.curve_dense_x, pressure_x, pressure_y)

    @staticmethod
    def _make_stars(count: int, seed: int) -> List[Star]:
        rng = np.random.default_rng(seed)
        return [
            Star(
                x=float(rng.uniform(0, OUT_W)),
                y=float(rng.uniform(0, OUT_H * 0.75)),
                radius=float(rng.uniform(0.3, 2.0) * OUT_W / 1080),
                alpha=float(rng.uniform(15, 120)),
                phase=float(rng.uniform(0, 2 * math.pi)),
            )
            for _ in range(count)
        ]

    @staticmethod
    def _make_terrain(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        width, depth = 220, 170
        base = rng.normal(0.0, 1.0, (depth, width)).astype(np.float32)
        img = Image.fromarray(np.uint8(np.clip((base - base.min()) / max(float(np.ptp(base)), 1e-6) * 255, 0, 255)))
        for radius in (16, 9, 5, 2):
            img = img.filter(ImageFilter.GaussianBlur(radius))
            base += (np.asarray(img, dtype=np.float32) / 255.0) * (radius / 6.0)
        yy, xx = np.mgrid[0:depth, 0:width]
        ridges = np.sin(xx / 17.0 + yy / 31.0) + 0.8 * np.cos(xx / 27.0)
        base += ridges.astype(np.float32)
        base -= base.min()
        base /= max(float(base.max()), 1e-6)
        return base

    def pressure_value(self, sol_frac: float) -> float:
        return float(np.interp(sol_frac % 1.0, self.pressure_x, self.pressure_y))

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", (OUT_W, OUT_H), (4, 6, 14, 255))
        draw = ImageDraw.Draw(image)

        for y in range(OUT_H):
            p = y / max(OUT_H - 1, 1)
            col = (
                int(lerp(4, 18, p)),
                int(lerp(6, 14, p)),
                int(lerp(14, 28, p)),
                255,
            )
            draw.line((0, y, OUT_W, y), fill=col)

        for star in self.stars:
            alpha = int(star.alpha * (0.74 + 0.26 * math.sin(star.phase + t * 1.2)))
            r = star.radius
            draw.ellipse((star.x - r, star.y - r, star.x + r, star.y + r), fill=COLORS["white"] + (alpha,))

        haze = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for cx, cy, color in [
            (OUT_W * 0.18, OUT_H * 0.30, (24, 20, 60)),
            (OUT_W * 0.80, OUT_H * 0.42, (60, 18, 28)),
            (OUT_W * 0.55, OUT_H * 0.78, (70, 25, 18)),
        ]:
            radius = 320 * OUT_W / 1080
            hd.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color + (18,))
        image.alpha_composite(haze.filter(ImageFilter.GaussianBlur(max(28, int(60 * OUT_W / 1080)))))
        return image

    def draw_mars(self, image: Image.Image, center: Tuple[int, int], radius: int, t: float, polar_phase: float = 0.0, dust_strength: float = 0.0):
        size = radius * 2
        yy, xx = np.mgrid[0:size, 0:size]
        nx = (xx - radius + 0.5) / radius
        ny = (yy - radius + 0.5) / radius
        r2 = nx * nx + ny * ny
        mask = r2 <= 1.0
        z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

        long_wave = np.sin((nx * 3.2 + 0.4 * np.cos(ny * 4.7)) * math.pi)
        fine_wave = np.sin((nx * 12.0 + ny * 9.0 + t * 0.2)) * 0.18 + np.cos((nx * 8.0 - ny * 10.0)) * 0.11
        texture = np.clip(0.42 + 0.26 * long_wave + fine_wave, 0.0, 1.0)

        light_dir = np.array([0.65, -0.35, 0.68], dtype=np.float32)
        shade = np.clip(nx * light_dir[0] + ny * light_dir[1] + z * light_dir[2], 0.0, 1.0)
        rim = np.clip((1.0 - z) ** 1.8, 0.0, 1.0)

        # Polar cap breathing.
        north_cap = clamp(0.06 + 0.06 * (0.5 + 0.5 * math.cos(2 * math.pi * polar_phase)))
        south_cap = clamp(0.05 + 0.08 * (0.5 + 0.5 * math.cos(2 * math.pi * (polar_phase + 0.5))))
        cap_mask = ((ny < -0.72 + north_cap) | (ny > 0.72 - south_cap)) & mask

        rgb = np.zeros((size, size, 4), dtype=np.uint8)
        base_r = np.clip(55 + 110 * texture + 125 * shade, 0, 255)
        base_g = np.clip(18 + 58 * texture + 45 * shade, 0, 255)
        base_b = np.clip(12 + 32 * texture + 18 * shade, 0, 255)

        dust = dust_strength * (0.55 + 0.45 * np.maximum(0.0, np.sin((nx * 5 + ny * 4 + t * 0.7) * math.pi)))
        base_r = np.clip(base_r + 34 * dust, 0, 255)
        base_g = np.clip(base_g + 18 * dust, 0, 255)
        base_b = np.clip(base_b + 6 * dust, 0, 255)

        rgb[..., 0] = base_r.astype(np.uint8)
        rgb[..., 1] = base_g.astype(np.uint8)
        rgb[..., 2] = base_b.astype(np.uint8)
        rgb[..., 3] = np.where(mask, 255, 0).astype(np.uint8)

        # Rim atmosphere
        rgb[..., 0] = np.clip(rgb[..., 0] + 40 * rim, 0, 255).astype(np.uint8)
        rgb[..., 1] = np.clip(rgb[..., 1] + 18 * rim, 0, 255).astype(np.uint8)

        # Polar caps
        rgb[cap_mask, 0] = 236
        rgb[cap_mask, 1] = 238
        rgb[cap_mask, 2] = 242

        mars = Image.fromarray(rgb, mode="RGBA")

        glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for shrink, alpha in [(0.0, 18), (0.08, 14), (0.16, 8)]:
            d = int(radius * shrink)
            gd.ellipse((d, d, size - d, size - d), outline=COLORS["orange"] + (alpha,), width=max(1, int(radius * 0.02)))
        glow = glow.filter(ImageFilter.GaussianBlur(max(3, int(radius * 0.05))))
        mars.alpha_composite(glow)

        image.alpha_composite(mars, (center[0] - radius, center[1] - radius))

        orbit = Image.new("RGBA", image.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(orbit)
        ry = int(radius * 0.42)
        od.ellipse((center[0] - int(radius * 1.18), center[1] - ry, center[0] + int(radius * 1.18), center[1] + ry), outline=COLORS["gold"] + (130,), width=max(1, int(OUT_W / 800)))
        dot_angle = t * 0.38
        dx = center[0] + int(math.cos(dot_angle) * radius * 1.18)
        dy = center[1] + int(math.sin(dot_angle) * ry)
        d = int(8 * OUT_W / 1080)
        od.ellipse((dx - d, dy - d, dx + d, dy + d), fill=COLORS["gold"] + (220,), outline=COLORS["white"] + (180,), width=1)
        image.alpha_composite(orbit)

    def draw_pressure_ribbon(self, image: Image.Image, box: Tuple[int, int, int, int], reveal: float, show_labels: bool = True, pulse_sol: Optional[float] = None):
        x0, y0, x1, y1 = box
        draw = ImageDraw.Draw(image)
        width = x1 - x0
        height = y1 - y0
        mid_y = int(lerp(y0 + height * 0.68, y0 + height * 0.34, 0.0))

        # Baseline and ticks.
        draw.line((x0, y1 - int(height * 0.18), x1, y1 - int(height * 0.18)), fill=(255, 220, 160, 58), width=1)
        if show_labels:
            draw_text(image, "Ls 0°", (x0, y0 - int(height * 0.10)), size=max(10, int(18 * OUT_W / 1080)), fill=COLORS["white"] + (220,), anchor="la", stroke=1)
            draw_text(image, "Ls 180°", ((x0 + x1) // 2, y0 - int(height * 0.10)), size=max(10, int(18 * OUT_W / 1080)), fill=COLORS["white"] + (220,), anchor="ma", stroke=1)
            draw_text(image, "Ls 360°", (x1, y0 - int(height * 0.10)), size=max(10, int(18 * OUT_W / 1080)), fill=COLORS["white"] + (220,), anchor="ra", stroke=1)

        # Curve
        points: List[Tuple[int, int]] = []
        max_n = max(2, int(reveal * len(self.curve_dense_x)))
        for x, pressure in zip(self.curve_dense_x[:max_n], self.curve_dense_y[:max_n]):
            px = int(lerp(x0, x1, float(x)))
            norm = (pressure - self.pressure_min) / max(self.pressure_max - self.pressure_min, 1e-6)
            py = int(lerp(y1 - height * 0.18, y0 + height * 0.12, norm))
            points.append((px, py))
        if len(points) >= 2:
            for i in range(len(points) - 1):
                a = i / max(1, len(points) - 1)
                color = tuple(int(lerp(COLORS["blue"][c], COLORS["orange"][c], a)) for c in range(3)) + (240,)
                draw.line((points[i], points[i + 1]), fill=color, width=max(2, int(4 * OUT_W / 1080)))
            # glow pass
            glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow)
            gd.line(points, fill=COLORS["gold"] + (90,), width=max(5, int(12 * OUT_W / 1080)))
            image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4, int(10 * OUT_W / 1080)))))

        # Seasonal dividers and labels
        if show_labels:
            seasons = [(0.00, "N WINTER", COLORS["blue"]), (0.25, "SPRING", COLORS["white"]), (0.50, "N SUMMER", COLORS["gold"]), (0.75, "AUTUMN", COLORS["orange"]), (1.0, "N WINTER", COLORS["blue"])]
            for frac, label, color in seasons:
                px = int(lerp(x0, x1, frac))
                draw.line((px, y1 - int(height * 0.12), px, y1 + int(height * 0.08)), fill=(255, 210, 160, 80), width=1)
                anchor = "la" if frac == 0 else ("ra" if frac == 1.0 else "ma")
                draw_text(image, label, (px, y1 + int(height * 0.14)), size=max(10, int(16 * OUT_W / 1080)), fill=color + (220,), anchor=anchor, stroke=1)

        if pulse_sol is not None:
            marker_x = int(lerp(x0, x1, pulse_sol))
            marker_p = self.pressure_value(pulse_sol)
            norm = (marker_p - self.pressure_min) / max(self.pressure_max - self.pressure_min, 1e-6)
            marker_y = int(lerp(y1 - height * 0.18, y0 + height * 0.12, norm))
            d = int(7 * OUT_W / 1080)
            draw.ellipse((marker_x - d, marker_y - d, marker_x + d, marker_y + d), fill=COLORS["white"] + (240,), outline=COLORS["gold"] + (235,), width=2)

    def draw_daily_tide(self, image: Image.Image, box: Tuple[int, int, int, int], phase: float):
        x0, y0, x1, y1 = box
        draw = ImageDraw.Draw(image)
        panel(image, box, alpha=120)
        draw_text(image, "ONE SOL", ((x0 + x1) // 2, y0 + int((y1 - y0) * 0.12)), size=max(12, int(20 * OUT_W / 1080)), fill=COLORS["white"] + (230,), bold=True, anchor="ma", stroke=1)

        pts: List[Tuple[int, int]] = []
        for i in range(160):
            frac = i / 159.0
            x = int(lerp(x0 + 18, x1 - 18, frac))
            yv = math.sin(2 * math.pi * frac + 2 * math.pi * phase) + 0.30 * math.sin(4 * math.pi * frac + 2 * math.pi * phase * 0.7)
            y = int(lerp(y0 + (y1 - y0) * 0.78, y0 + (y1 - y0) * 0.32, (yv + 1.4) / 2.8))
            pts.append((x, y))
        glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.line(pts, fill=COLORS["gold"] + (110,), width=max(4, int(10 * OUT_W / 1080)))
        image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4, int(8 * OUT_W / 1080)))))
        draw.line(pts, fill=COLORS["white"] + (225,), width=max(2, int(4 * OUT_W / 1080)))
        draw_text(image, "daily atmospheric tide", ((x0 + x1) // 2, y1 - int((y1 - y0) * 0.10)), size=max(10, int(16 * OUT_W / 1080)), fill=COLORS["muted"] + (210,), anchor="ma", stroke=1)

    def draw_terrain_and_rover(self, image: Image.Image, horizon_y: int, t: float, dust: float = 0.0):
        terrain_rgba = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(terrain_rgba)

        # Sky glow near horizon.
        glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        sun_x = int(OUT_W * 0.83)
        sun_y = horizon_y + int(OUT_H * 0.04)
        for r, alpha in [(int(OUT_W * 0.24), 14), (int(OUT_W * 0.16), 26), (int(OUT_W * 0.08), 45)]:
            gd.ellipse((sun_x - r, sun_y - r, sun_x + r, sun_y + r), fill=COLORS["orange"] + (alpha,))
        terrain_rgba.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(6, int(22 * OUT_W / 1080)))))
        gd = ImageDraw.Draw(terrain_rgba)
        gd.ellipse((sun_x - int(6 * OUT_W / 1080), sun_y - int(6 * OUT_W / 1080), sun_x + int(6 * OUT_W / 1080), sun_y + int(6 * OUT_W / 1080)), fill=COLORS["white"] + (245,))

        # Distant canyon edges
        left_points = [(0, horizon_y + 35)]
        for i in range(8):
            frac = i / 7
            x = int(frac * OUT_W * 0.42)
            y = horizon_y + int(40 + 30 * math.sin(i * 0.7) + 16 * math.cos(i * 1.2))
            left_points.append((x, y))
        left_points += [(int(OUT_W * 0.42), OUT_H), (0, OUT_H)]
        gd.polygon(left_points, fill=(83, 34, 20, 255))

        right_points = [(OUT_W, horizon_y + 45)]
        for i in range(8):
            frac = i / 7
            x = int(OUT_W - frac * OUT_W * 0.28)
            y = horizon_y + int(52 + 26 * math.sin(i * 0.75))
            right_points.append((x, y))
        right_points += [(int(OUT_W * 0.72), OUT_H), (OUT_W, OUT_H)]
        gd.polygon(right_points, fill=(74, 28, 18, 255))

        # Foreground terrain slices from procedural height map.
        hmap = self.terrain
        depth, width = hmap.shape
        for y in range(horizon_y, OUT_H):
            frac_y = (y - horizon_y) / max(OUT_H - horizon_y, 1)
            sample_row = min(depth - 1, int(frac_y ** 1.25 * (depth - 1)))
            strip = hmap[sample_row]
            for x in range(0, OUT_W, 8):
                idx = min(width - 1, int(x / OUT_W * (width - 1)))
                val = strip[idx]
                shade = 0.55 + 0.45 * frac_y
                r = int(lerp(44, 130, val * shade))
                g = int(lerp(20, 70, val * shade * 0.8))
                b = int(lerp(10, 34, val * shade * 0.5))
                gd.rectangle((x, y, min(OUT_W, x + 8), min(OUT_H, y + 4)), fill=(r, g, b, 255))

        # Rover silhouette.
        scale = OUT_W / 1080
        rx = int(OUT_W * 0.44)
        ry = int(OUT_H * 0.84)
        # wheels
        wheel_r = int(24 * scale)
        wheel_centers = [(rx - 90 * scale, ry + 10 * scale), (rx - 28 * scale, ry + 12 * scale), (rx + 35 * scale, ry + 12 * scale), (rx + 96 * scale, ry + 8 * scale)]
        for wx, wy in wheel_centers:
            gd.ellipse((wx - wheel_r, wy - wheel_r, wx + wheel_r, wy + wheel_r), fill=(18, 12, 10, 255), outline=(80, 45, 30, 255), width=2)
        # body and mast
        body = [(rx - 72 * scale, ry - 8 * scale), (rx + 28 * scale, ry - 18 * scale), (rx + 68 * scale, ry + 0 * scale), (rx + 32 * scale, ry + 34 * scale), (rx - 50 * scale, ry + 30 * scale)]
        gd.polygon(body, fill=(24, 17, 12, 255), outline=(92, 52, 32, 255))
        gd.rectangle((rx - 6 * scale, ry - 110 * scale, rx + 6 * scale, ry - 18 * scale), fill=(28, 18, 14, 255))
        gd.rectangle((rx - 22 * scale, ry - 138 * scale, rx + 24 * scale, ry - 110 * scale), fill=(26, 16, 14, 255), outline=(96, 58, 36, 255))
        # suspension lines
        for wx, wy in wheel_centers:
            gd.line((rx - 8 * scale, ry + 12 * scale, wx, wy - wheel_r // 2), fill=(80, 45, 30, 255), width=max(1, int(3 * scale)))

        if dust > 0.0:
            dust_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            dd = ImageDraw.Draw(dust_layer)
            rng = np.random.default_rng(2025)
            count = int(70 + 120 * dust)
            for _ in range(count):
                px = float(rng.uniform(0, OUT_W))
                py = float(rng.uniform(horizon_y - 20, OUT_H - 40))
                rad_x = float(rng.uniform(12, 48) * scale)
                rad_y = rad_x * rng.uniform(0.30, 0.75)
                alpha = int(rng.uniform(6, 26) * (0.4 + dust))
                dd.ellipse((px - rad_x, py - rad_y, px + rad_x, py + rad_y), fill=COLORS["orange"] + (alpha,))
            dust_layer = dust_layer.filter(ImageFilter.GaussianBlur(max(4, int(16 * scale))))
            terrain_rgba.alpha_composite(dust_layer)

        image.alpha_composite(terrain_rgba)

    def draw_title(self, image: Image.Image, title: str, subtitle: Optional[str] = None):
        draw_text(image, title, (OUT_W // 2, int(OUT_H * 0.10)), size=max(18, int(70 * OUT_W / 1080)), fill=COLORS["white"] + (245,), bold=True, anchor="ma", stroke=2)
        if subtitle:
            draw_text(image, subtitle, (OUT_W // 2, int(OUT_H * 0.145)), size=max(10, int(28 * OUT_W / 1080)), fill=COLORS["gold"] + (240,), anchor="ma", stroke=1)

    def draw_bottom_label(self, image: Image.Image, text: str):
        box = (int(OUT_W * 0.35), int(OUT_H * 0.93), int(OUT_W * 0.65), int(OUT_H * 0.965))
        panel(image, box, alpha=155)
        draw_text(image, text, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), size=max(11, int(24 * OUT_W / 1080)), fill=COLORS["gold"] + (245,), anchor="mm", stroke=1)

    def draw_caption(self, image: Image.Image, caption: str, t: float):
        if not caption:
            return
        # Find caption window for fade in/out
        start, end = 0.0, 0.0
        for a, b, text in CAPTIONS:
            if text == caption and a <= t < b:
                start, end = a, b
                break
        fade_in = clamp((t - start) / 0.35)
        fade_out = clamp((end - t) / 0.45)
        alpha = int(220 * min(fade_in, fade_out, 1.0))
        if alpha <= 0:
            return
        box = (int(OUT_W * 0.08), int(OUT_H * 0.74), int(OUT_W * 0.92), int(OUT_H * 0.84))
        panel(image, box, alpha=min(100, alpha // 2))
        draw_wrapped_text(image, caption, (box[0] + 24, box[1] + 18, box[2] - 24, box[3] - 18), size=max(12, int(30 * OUT_W / 1080)), fill=COLORS["white"] + (alpha,), bold=False)

    def frame(self, t: float) -> np.ndarray:
        shot, t0, t1 = get_shot(t)
        local = smoothstep((t - t0) / max(t1 - t0, 1e-9))
        image = self.background(t)

        if shot == "intro":
            self.draw_mars(image, (OUT_W // 2, int(OUT_H * 0.38)), int(OUT_W * 0.28), t, polar_phase=0.15, dust_strength=0.05)
            self.draw_title(image, "THE REAL PRESSURE", "CHANGES ON MARS")

        elif shot == "scope":
            self.draw_mars(image, (OUT_W // 2, int(OUT_H * 0.38)), int(OUT_W * 0.29), t, polar_phase=0.20, dust_strength=0.05)
            box = (int(OUT_W * 0.18), int(OUT_H * 0.16), int(OUT_W * 0.82), int(OUT_H * 0.215))
            panel(image, box, alpha=125)
            draw_text(image, "Curiosity • Gale Crater", ((box[0] + box[2]) // 2, int(OUT_H * 0.187)), size=max(12, int(32 * OUT_W / 1080)), fill=COLORS["gold"] + (240,), anchor="mm", stroke=1)

        elif shot == "dive":
            zoom = lerp(0.26, 0.38, local)
            self.draw_mars(image, (OUT_W // 2, int(OUT_H * 0.28)), int(OUT_W * zoom), t, polar_phase=0.22, dust_strength=0.06)
            self.draw_terrain_and_rover(image, int(OUT_H * 0.66), t, dust=0.08)
            self.draw_bottom_label(image, "GALE CRATER")

        elif shot == "annual_curve":
            self.draw_mars(image, (OUT_W // 2, int(OUT_H * 0.30)), int(OUT_W * 0.29), t, polar_phase=0.28, dust_strength=0.04)
            panel(image, (int(OUT_W * 0.07), int(OUT_H * 0.52), int(OUT_W * 0.93), int(OUT_H * 0.76)), 118)
            draw_text(image, "ONE MARTIAN YEAR", (OUT_W // 2, int(OUT_H * 0.56)), size=max(14, int(34 * OUT_W / 1080)), fill=COLORS["gold"] + (240,), bold=True, anchor="ma", stroke=1)
            self.draw_pressure_ribbon(image, (int(OUT_W * 0.10), int(OUT_H * 0.60), int(OUT_W * 0.90), int(OUT_H * 0.70)), reveal=local)
            draw_text(image, "~700 to ~970 Pa", (OUT_W // 2, int(OUT_H * 0.73)), size=max(12, int(28 * OUT_W / 1080)), fill=COLORS["white"] + (235,), anchor="ma", stroke=1)

        elif shot == "polar_drop":
            polar_phase = lerp(0.12, 0.02, local)
            self.draw_mars(image, (OUT_W // 2, int(OUT_H * 0.32)), int(OUT_W * 0.31), t, polar_phase=polar_phase, dust_strength=0.04)
            self.draw_pressure_ribbon(image, (int(OUT_W * 0.10), int(OUT_H * 0.62), int(OUT_W * 0.90), int(OUT_H * 0.72)), reveal=1.0, pulse_sol=lerp(0.05, 0.23, local))
            draw_text(image, "CO₂ freezes onto a pole", (OUT_W // 2, int(OUT_H * 0.56)), size=max(13, int(30 * OUT_W / 1080)), fill=COLORS["gold"] + (235,), bold=True, anchor="ma", stroke=1)

        elif shot == "polar_rise":
            polar_phase = lerp(0.03, 0.45, local)
            self.draw_mars(image, (OUT_W // 2, int(OUT_H * 0.32)), int(OUT_W * 0.31), t, polar_phase=polar_phase, dust_strength=0.04)
            self.draw_pressure_ribbon(image, (int(OUT_W * 0.10), int(OUT_H * 0.62), int(OUT_W * 0.90), int(OUT_H * 0.72)), reveal=1.0, pulse_sol=lerp(0.30, 0.58, local))
            draw_text(image, "Sunlight returns • pressure rises", (OUT_W // 2, int(OUT_H * 0.56)), size=max(13, int(30 * OUT_W / 1080)), fill=COLORS["gold"] + (235,), bold=True, anchor="ma", stroke=1)

        elif shot == "daily_tides":
            self.draw_mars(image, (OUT_W // 2, int(OUT_H * 0.26)), int(OUT_W * 0.23), t, polar_phase=0.38, dust_strength=0.02)
            self.draw_daily_tide(image, (int(OUT_W * 0.12), int(OUT_H * 0.48), int(OUT_W * 0.88), int(OUT_H * 0.70)), phase=local)
            self.draw_terrain_and_rover(image, int(OUT_H * 0.73), t, dust=0.04)

        elif shot == "dust_local":
            dust = lerp(0.10, 0.55, local)
            self.draw_mars(image, (OUT_W // 2, int(OUT_H * 0.25)), int(OUT_W * 0.22), t, polar_phase=0.36, dust_strength=dust)
            self.draw_pressure_ribbon(image, (int(OUT_W * 0.10), int(OUT_H * 0.46), int(OUT_W * 0.90), int(OUT_H * 0.56)), reveal=1.0, pulse_sol=0.77)
            self.draw_terrain_and_rover(image, int(OUT_H * 0.64), t, dust=dust)
            draw_text(image, "dust • temperature • local topography", (OUT_W // 2, int(OUT_H * 0.60)), size=max(11, int(24 * OUT_W / 1080)), fill=COLORS["muted"] + (220,), anchor="ma", stroke=1)
            self.draw_bottom_label(image, "GALE CRATER")

        elif shot == "finale":
            self.draw_mars(image, (OUT_W // 2, int(OUT_H * 0.22)), int(OUT_W * 0.19), t, polar_phase=0.30, dust_strength=0.06)
            self.draw_pressure_ribbon(image, (int(OUT_W * 0.08), int(OUT_H * 0.36), int(OUT_W * 0.92), int(OUT_H * 0.46)), reveal=1.0, pulse_sol=0.54)
            self.draw_terrain_and_rover(image, int(OUT_H * 0.57), t, dust=0.12)
            self.draw_bottom_label(image, "MARS BREATHES")
            draw_text(image, "Curiosity • Gale Crater • One Martian Year", (OUT_W // 2, int(OUT_H * 0.51)), size=max(12, int(26 * OUT_W / 1080)), fill=COLORS["gold"] + (235,), anchor="ma", stroke=1)

        caption = caption_at(t)
        if caption:
            self.draw_caption(image, caption, t)

        # tiny unobtrusive source label at bottom left
        draw_text(image, self.source_label.replace("csv:", "data: "), (int(OUT_W * 0.03), int(OUT_H * 0.985)), size=max(8, int(13 * OUT_W / 1080)), fill=(210, 210, 215, 120), anchor="ls", stroke=1)

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        arr = np.clip(arr * VIGNETTE[..., None], 0, 255)
        arr = np.asarray(ImageEnhance.Contrast(Image.fromarray(arr.astype(np.uint8))).enhance(float(CONFIG["contrast"])))
        arr = np.asarray(ImageEnhance.Color(Image.fromarray(arr)).enhance(float(CONFIG["saturation"])))
        return arr.astype(np.uint8)


# -----------------------------------------------------------------------------
# Audio
# -----------------------------------------------------------------------------


def envelope(n: int, attack: float, release: float) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    a = int(n * attack)
    r = int(n * release)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
    if r > 0:
        env[-r:] = np.minimum(env[-r:], np.linspace(1.0, 0.0, r, endpoint=True))
    return env


def make_audio(path: Path, duration_s: float):
    sr = int(CONFIG["audio_rate"])
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float32) / sr

    drone = 0.18 * np.sin(2 * math.pi * 55.0 * t) + 0.09 * np.sin(2 * math.pi * 82.5 * t + 0.3)
    shimmer = 0.04 * np.sin(2 * math.pi * 440.0 * t) * (0.5 + 0.5 * np.sin(2 * math.pi * 0.1 * t))
    air = 0.018 * np.random.default_rng(11).normal(0, 1, n).astype(np.float32)

    bed = drone + shimmer + air

    # Transition pulses aligned to shot changes.
    for _, start, _ in SHOT_PLAN[1:]:
        start_idx = int(start * sr)
        dur = int(0.55 * sr)
        end_idx = min(n, start_idx + dur)
        tt = np.arange(end_idx - start_idx, dtype=np.float32) / sr
        hit = 0.16 * np.sin(2 * math.pi * 110.0 * tt) * np.exp(-tt * 5.0)
        whoosh = 0.025 * np.random.default_rng(int(start * 1000) + 17).normal(0, 1, end_idx - start_idx).astype(np.float32)
        whoosh *= np.exp(-tt * 3.0)
        bed[start_idx:end_idx] += hit + whoosh

    # Small pulsing motif during daily tides segment.
    for shot_name, a, b in SHOT_PLAN:
        if shot_name == "daily_tides":
            times = np.arange(a, b, 0.78)
            for pulse_t in times:
                i0 = int(pulse_t * sr)
                length = int(0.19 * sr)
                i1 = min(n, i0 + length)
                tt = np.arange(i1 - i0, dtype=np.float32) / sr
                pulse = 0.08 * np.sin(2 * math.pi * 260.0 * tt) * envelope(len(tt), 0.05, 0.75)
                bed[i0:i1] += pulse
            break

    # Finale shimmer.
    i0 = int(max(0, duration_s - 5.0) * sr)
    tt = np.arange(n - i0, dtype=np.float32) / sr
    bed[i0:] += 0.05 * np.sin(2 * math.pi * 660.0 * tt) * np.exp(-tt * 0.6)

    bed = bed / max(float(np.max(np.abs(bed))), 1e-6) * 0.72
    # Very slight stereo spread.
    left = np.clip(bed * (0.96 + 0.04 * np.sin(2 * math.pi * 0.03 * t)), -1.0, 1.0)
    right = np.clip(bed * (0.96 + 0.04 * np.cos(2 * math.pi * 0.03 * t + 0.9)), -1.0, 1.0)
    stereo = np.stack([left, right], axis=1)

    pcm = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm.tobytes())


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def render_video(scene: MarsPressureScene, mp4_silent_path: Path):
    with iio.get_writer(str(mp4_silent_path), fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p") as writer:
        for i in range(FRAME_COUNT):
            t = i / FPS
            writer.append_data(scene.frame(t))


def mux_audio(video_path: Path, audio_path: Path, final_path: Path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        shutil.copy2(video_path, final_path)
        return False
    cmd = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(final_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        shutil.copy2(video_path, final_path)
        return False


def save_summary(path: Path, source_label: str, pressure_x: np.ndarray, pressure_y: np.ndarray, used_audio: bool):
    summary = {
        "title": CONFIG["title"],
        "subtitle": CONFIG["subtitle"],
        "data_source": source_label,
        "pressure_min_pa": float(np.min(pressure_y)),
        "pressure_max_pa": float(np.max(pressure_y)),
        "pressure_min_mbar": float(np.min(pressure_y) / 100.0),
        "pressure_max_mbar": float(np.max(pressure_y) / 100.0),
        "notes": [
            "Location-specific story: Gale Crater / Curiosity REMS.",
            "If using the fallback profile, values represent a hand-entered annual shape built around the well-known REMS seasonal range rather than a raw archived series.",
            "Text overlays intentionally clear after a few seconds to keep the visuals readable.",
            f"AAC audio muxed via ffmpeg: {used_audio}",
        ],
    }
    import json
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


