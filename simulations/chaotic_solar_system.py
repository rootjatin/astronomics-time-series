from __future__ import annotations

"""
The Most Chaotic Solar System I Could Simulate
==============================================

A cinematic vertical YouTube Short renderer built around an intentionally
unstable Newtonian N-body toy system: one Sun-like star plus six giant planets
packed far too closely together.

Scientific framing
------------------
- This is NOT a model of the real Solar System and not a claim about the
  "most chaotic" physically possible planetary system. The title is a cinematic
  framing for an intentionally unstable simulation.
- Gravity is Newtonian and every body pulls on every other body.
- The system uses one 1-solar-mass star and six equal 0.0015-solar-mass giant
  planets (about 1.57 Jupiter masses each) between 0.55 and 1.39 AU.
- Planetary orbits begin nearly circular but are much too tightly packed for
  long-term stability at these masses.
- A second copy of the system is started with one planet shifted outward by
  only 1e-8 AU (~1.5 km). The two simulations soon diverge dramatically.
- The integrator is velocity-Verlet / leapfrog-like with a small Plummer-style
  softening length (0.005 AU) so close encounters remain visually stable.
  Because of that softening, close-encounter details are educational rather
  than precision celestial-mechanics predictions.
- In the default deterministic run, the twin trajectories separate by more
  than 0.01 AU after roughly seven simulated years, and one planet eventually
  travels beyond several AU after repeated gravitational encounters. Exact
  encounter order is highly sensitive to numerical and initial-condition
  details -- which is itself part of the chaos lesson.

Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    CHAOTIC_SYSTEM_QUICK=1 python the_most_chaotic_solar_system_i_could_simulate.py

Full 1080x1920 render
---------------------
    python the_most_chaotic_solar_system_i_could_simulate.py

4K vertical render
------------------
    CHAOTIC_SYSTEM_4K=1 python the_most_chaotic_solar_system_i_could_simulate.py
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

QUICK_MODE = os.environ.get("CHAOTIC_SYSTEM_QUICK", "0") == "1"
FOUR_K = os.environ.get("CHAOTIC_SYSTEM_4K", "0") == "1" and not QUICK_MODE

W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 6 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SIZE = (W, H)
SCALE = W / 1080.0

OUTPUT_ROOT = Path("the_most_chaotic_solar_system_i_could_simulate_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "THE MOST CHAOTIC SOLAR SYSTEM I COULD SIMULATE",
    "subtitle": "six giant planets // packed too close // tiny changes explode",
    "basename": "the_most_chaotic_solar_system_i_could_simulate",
    "contrast": 1.12,
    "saturation": 1.10,
    "vignette": 0.29,
}

COLORS = {
    "space": (2, 4, 14),
    "space2": (8, 13, 35),
    "white": (248, 251, 255),
    "muted": (165, 190, 214),
    "cyan": (72, 230, 255),
    "blue": (75, 132, 255),
    "gold": (255, 204, 84),
    "orange": (255, 139, 70),
    "red": (255, 76, 102),
    "green": (103, 241, 174),
    "violet": (188, 124, 255),
    "magenta": (244, 88, 190),
    "star": (255, 224, 115),
    "star_hot": (255, 250, 220),
    "panel": (3, 8, 22),
}

PLANET_COLORS = [
    (98, 219, 255),
    (112, 147, 255),
    (129, 239, 179),
    (255, 210, 94),
    (255, 139, 80),
    (223, 112, 255),
]

SHOT_PLAN = [
    {"name": "packed", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "many_body", "start": 8.0 if not QUICK_MODE else 1.8, "end": 18.0 if not QUICK_MODE else 4.0},
    {"name": "butterfly", "start": 18.0 if not QUICK_MODE else 4.0, "end": 29.0 if not QUICK_MODE else 6.45},
    {"name": "encounters", "start": 29.0 if not QUICK_MODE else 6.45, "end": 40.0 if not QUICK_MODE else 8.9},
    {"name": "ejection", "start": 40.0 if not QUICK_MODE else 8.9, "end": 51.0 if not QUICK_MODE else 11.35},
    {"name": "outro", "start": 51.0 if not QUICK_MODE else 11.35, "end": DURATION},
]

CAPTION_TEXTS = [
    "I packed six super-Jupiter planets between 0.55 and 1.39 astronomical units. They start almost circular, but at these masses the spacing is a gravitational traffic jam.",
    "This is a true many-body problem: every planet pulls on the star, the star pulls on every planet, and every planet tugs on every other planet at the same time.",
    "Then I cloned the system and moved just one planet by about one and a half kilometers. For a while the two copies look identical. Then their trajectories separate explosively.",
    "Close encounters trade orbital energy and angular momentum. A planet can be kicked inward, thrown outward, or reshuffle the paths of several worlds in a single encounter.",
    "In this deliberately unstable run, one planet eventually wanders beyond several astronomical units. The exact escape sequence is sensitive to tiny numerical and initial differences.",
    "That is orbital chaos: deterministic rules, but rapidly vanishing predictability. Tiny changes in the starting state can grow into completely different planetary histories.",
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
# N-body toy model
# =============================================================================

G = 4.0 * math.pi**2  # AU^3 / (Msun * yr^2)
STAR_MASS = 1.0
PLANET_MASS = 0.0015
N_PLANETS = 6
MASSES = np.asarray([STAR_MASS] + [PLANET_MASS] * N_PLANETS, dtype=np.float64)
INITIAL_RADII_AU = np.asarray([0.55, 0.68, 0.82, 0.98, 1.17, 1.39], dtype=np.float64)
INITIAL_PHASES = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
SOFTENING_AU = 0.005
PERTURBATION_AU = 1e-8  # about 1.5 km
SIM_YEARS = 60.0
SIM_DT_YEARS = 0.0005
SIM_SAMPLE_YEARS = 0.01


@dataclass
class ChaosSnapshot:
    generated_at_utc: str
    star_mass_solar: float
    planet_mass_solar_each: float
    planet_mass_jupiter_each_approx: float
    initial_radii_au: List[float]
    softening_au: float
    perturbation_au: float
    perturbation_km_approx: float
    simulation_years: float
    timestep_days: float
    divergence_0p01_au_year: Optional[float]
    divergence_0p1_au_year: Optional[float]
    first_radius_gt_3au_year: Optional[float]
    first_radius_gt_5au_year: Optional[float]
    minimum_planet_pair_separation_au: float
    maximum_planet_radius_au: float
    scenario: str
    interpretation: str


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


def ease_in_out_sine(x: float) -> float:
    t = clamp(x)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


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


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(7, int(size * SCALE)))
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


def initial_state(perturbation_au: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    n = 1 + N_PLANETS
    pos = np.zeros((n, 2), dtype=np.float64)
    vel = np.zeros((n, 2), dtype=np.float64)
    for i, (a0, phase) in enumerate(zip(INITIAL_RADII_AU, INITIAL_PHASES), start=1):
        a = float(a0 + (perturbation_au if i == 3 else 0.0))
        c, s = math.cos(float(phase)), math.sin(float(phase))
        pos[i] = (a * c, a * s)
        speed = math.sqrt(G * (STAR_MASS + MASSES[i]) / a)
        vel[i] = speed * np.asarray((-s, c))
    # Put the whole system approximately in the barycentric frame.
    vel[0] = -np.sum(MASSES[1:, None] * vel[1:], axis=0) / STAR_MASS
    pos[0] = -np.sum(MASSES[1:, None] * pos[1:], axis=0) / STAR_MASS
    return pos, vel


def accelerations(pos: np.ndarray) -> np.ndarray:
    # delta[i, j] = r_j - r_i
    delta = pos[None, :, :] - pos[:, None, :]
    r2 = np.sum(delta * delta, axis=2) + SOFTENING_AU**2
    inv_r3 = r2 ** -1.5
    np.fill_diagonal(inv_r3, 0.0)
    return G * np.sum(delta * (MASSES[None, :, None] * inv_r3[:, :, None]), axis=1)


def integrate_system(perturbation_au: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos, vel = initial_state(perturbation_au)
    acc = accelerations(pos)
    sample_every = max(1, int(round(SIM_SAMPLE_YEARS / SIM_DT_YEARS)))
    steps = int(round(SIM_YEARS / SIM_DT_YEARS))
    times: List[float] = []
    positions: List[np.ndarray] = []
    velocities: List[np.ndarray] = []

    for step in range(steps + 1):
        if step % sample_every == 0:
            times.append(step * SIM_DT_YEARS)
            positions.append(pos.copy())
            velocities.append(vel.copy())
        if step == steps:
            break
        pos = pos + vel * SIM_DT_YEARS + 0.5 * acc * SIM_DT_YEARS**2
        new_acc = accelerations(pos)
        vel = vel + 0.5 * (acc + new_acc) * SIM_DT_YEARS
        acc = new_acc

    return np.asarray(times), np.asarray(positions), np.asarray(velocities)


def first_time(times: np.ndarray, mask: np.ndarray) -> Optional[float]:
    idx = np.where(mask)[0]
    return None if len(idx) == 0 else float(times[int(idx[0])])


def analyze_simulation(times: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[ChaosSnapshot, Dict[str, np.ndarray]]:
    rel = a[:, 1:, :] - a[:, 0:1, :]
    radii = np.linalg.norm(rel, axis=2)
    pair = rel[:, :, None, :] - rel[:, None, :, :]
    pair_d = np.linalg.norm(pair, axis=3)
    pair_d[:, np.arange(N_PLANETS), np.arange(N_PLANETS)] = 1e9
    min_pair = pair_d.min(axis=(1, 2))

    divergence = np.sqrt(np.mean(np.sum((a[:, 1:, :] - b[:, 1:, :]) ** 2, axis=2), axis=1))
    max_radius = radii.max(axis=1)

    snapshot = ChaosSnapshot(
        generated_at_utc=iso_z(utc_now()),
        star_mass_solar=STAR_MASS,
        planet_mass_solar_each=PLANET_MASS,
        planet_mass_jupiter_each_approx=PLANET_MASS / 0.0009543,
        initial_radii_au=[float(x) for x in INITIAL_RADII_AU],
        softening_au=SOFTENING_AU,
        perturbation_au=PERTURBATION_AU,
        perturbation_km_approx=PERTURBATION_AU * 149_597_870.7,
        simulation_years=SIM_YEARS,
        timestep_days=SIM_DT_YEARS * 365.256,
        divergence_0p01_au_year=first_time(times, divergence > 0.01),
        divergence_0p1_au_year=first_time(times, divergence > 0.1),
        first_radius_gt_3au_year=first_time(times, max_radius > 3.0),
        first_radius_gt_5au_year=first_time(times, max_radius > 5.0),
        minimum_planet_pair_separation_au=float(min_pair.min()),
        maximum_planet_radius_au=float(max_radius.max()),
        scenario="Intentionally unstable six-giant-planet Newtonian toy system with two nearly identical initial conditions.",
        interpretation="The exact close-encounter sequence is not a precision prediction. The point is sensitive dependence: tiny initial differences become macroscopic orbital differences.",
    )
    metrics = {
        "divergence_au": divergence,
        "max_radius_au": max_radius,
        "min_pair_separation_au": min_pair,
        "planet_radii_au": radii,
    }
    return snapshot, metrics


def collect_data():
    times, positions_a, velocities_a = integrate_system(0.0)
    times_b, positions_b, velocities_b = integrate_system(PERTURBATION_AU)
    if not np.allclose(times, times_b):
        raise RuntimeError("Twin integrations produced mismatched sample times")
    snapshot, metrics = analyze_simulation(times, positions_a, positions_b)
    return snapshot, times, positions_a, velocities_a, positions_b, velocities_b, metrics


def save_data(
    snapshot: ChaosSnapshot,
    times: np.ndarray,
    positions_a: np.ndarray,
    metrics: Dict[str, np.ndarray],
) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "chaos_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_years", "twin_divergence_au", "minimum_planet_pair_separation_au", "maximum_planet_radius_au"])
        stride = max(1, len(times) // 1200)
        for i in range(0, len(times), stride):
            writer.writerow([
                f"{times[i]:.6f}",
                f"{metrics['divergence_au'][i]:.9f}",
                f"{metrics['min_pair_separation_au'][i]:.9f}",
                f"{metrics['max_radius_au'][i]:.9f}",
            ])

    json_path = DATA_ROOT / "chaotic_system_snapshot.json"
    json_path.write_text(
        json.dumps(
            {
                "snapshot": asdict(snapshot),
                "notes": {
                    "units": "AU, solar masses, Julian years",
                    "integrator": "velocity-Verlet / leapfrog-like",
                    "gravity": "Newtonian inverse-square with Plummer-style softening",
                    "planet_count": N_PLANETS,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path, json_path


# =============================================================================
# Scene renderer
# =============================================================================

class ChaosScene:
    def __init__(
        self,
        snapshot: ChaosSnapshot,
        times: np.ndarray,
        positions_a: np.ndarray,
        positions_b: np.ndarray,
        metrics: Dict[str, np.ndarray],
    ):
        self.snapshot = snapshot
        self.times = times
        self.a = positions_a
        self.b = positions_b
        self.metrics = metrics
        rng = np.random.default_rng(20260921)
        self.stars = [
            (
                float(rng.uniform(0, W)),
                float(rng.uniform(0, H)),
                float(rng.uniform(0.4, 2.0) * SCALE),
                int(rng.uniform(15, 80)),
                float(rng.uniform(0, math.tau)),
            )
            for _ in range(160 if QUICK_MODE else 460)
        ]
        self.base_bg = self._make_background()

    def _make_background(self) -> Image.Image:
        top = np.asarray(COLORS["space"], dtype=np.float32)
        bottom = np.asarray(COLORS["space2"], dtype=np.float32)
        f = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        rgb = (top[None, :] * (1 - f) + bottom[None, :] * f).astype(np.uint8)
        arr = np.empty((H, W, 4), dtype=np.uint8)
        arr[..., :3] = rgb[:, None, :]
        arr[..., 3] = 255
        return Image.fromarray(arr, "RGBA")

    def background(self, t: float) -> Image.Image:
        img = self.base_bg.copy()
        d = ImageDraw.Draw(img)
        for x, y, r, a, phase in self.stars:
            alpha = int(a * (0.78 + 0.22 * math.sin(t * 0.85 + phase)))
            d.ellipse((x-r, y-r, x+r, y+r), fill=COLORS["white"] + (alpha,))
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for cx, cy, col in [
            (W * 0.24, H * 0.30, COLORS["blue"]),
            (W * 0.76, H * 0.25, COLORS["violet"]),
            (W * 0.52, H * 0.72, COLORS["red"]),
        ]:
            rr = int(W * 0.28)
            gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=col + (15,))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(18, int(55*SCALE)))))
        return img

    @staticmethod
    def panel(img: Image.Image, box: Tuple[int, int, int, int], alpha: int = 176):
        layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle(
            box,
            radius=max(8, int(24 * SCALE)),
            fill=COLORS["panel"] + (alpha,),
            outline=COLORS["cyan"] + (48,),
            width=max(1, int(2 * SCALE)),
        )
        img.alpha_composite(layer)

    def draw_header(self, img: Image.Image, t: float):
        if t < (6.0 if not QUICK_MODE else 1.35):
            fade = smoothstep(t / (0.75 if not QUICK_MODE else 0.18))
            draw_text(img, "THE MOST CHAOTIC", (W//2, int(H*0.055)), 40 if not QUICK_MODE else 20,
                      COLORS["white"] + (int(245*fade),), True, "ma", 2)
            draw_text(img, "SOLAR SYSTEM", (W//2, int(H*0.100)), 54 if not QUICK_MODE else 27,
                      COLORS["red"] + (int(250*fade),), True, "ma", 2)
            draw_text(img, "I COULD SIMULATE", (W//2, int(H*0.145)), 34 if not QUICK_MODE else 17,
                      COLORS["cyan"] + (int(240*fade),), True, "ma", 2)
        else:
            labels = {
                "packed": "SIX GIANTS // TOO CLOSE",
                "many_body": "EVERY BODY PULLS ON EVERY BODY",
                "butterfly": "A 1.5 km CHANGE",
                "encounters": "CLOSE ENCOUNTERS TRADE ENERGY",
                "ejection": "ONE KICK CAN REWRITE THE SYSTEM",
                "outro": "DETERMINISTIC ≠ PREDICTABLE",
            }
            draw_text(img, labels[get_shot(t)["name"]], (int(W*.055), int(H*.045)), 17 if not QUICK_MODE else 8,
                      COLORS["muted"] + (230,), True, "la", 1)
        draw_text(img, "NEWTONIAN N-BODY // INTENTIONALLY UNSTABLE TOY MODEL",
                  (int(W*.945), int(H*.047)), 12 if not QUICK_MODE else 6,
                  COLORS["cyan"] + (205,), True, "ra", 1)

    def draw_caption(self, img: Image.Image, t: float):
        cap = caption_at(t)
        if not cap:
            return
        y0 = H - int(170 * SCALE)
        self.panel(img, (int(58*SCALE), y0, W-int(58*SCALE), y0+int(116*SCALE)), 150)
        draw_wrapped_text(img, cap, (int(82*SCALE), y0+int(18*SCALE)), W-int(164*SCALE),
                          30, COLORS["white"]+(240,), False, 4)

    def sim_index(self, sim_year: float) -> int:
        return int(np.clip(np.searchsorted(self.times, sim_year), 0, len(self.times)-1))

    def frame_to_sim_year(self, t: float) -> float:
        shot = get_shot(t)
        local = clamp((t - shot["start"]) / max(1e-9, shot["end"] - shot["start"]))
        if shot["name"] == "packed":
            return lerp(0.0, 1.0, local)
        if shot["name"] == "many_body":
            return lerp(0.0, 3.0, local)
        if shot["name"] == "butterfly":
            return lerp(0.0, 10.0, local)
        if shot["name"] == "encounters":
            return lerp(0.0, 18.0, local)
        if shot["name"] == "ejection":
            return lerp(15.0, 55.0, local)
        return lerp(0.0, 60.0, local)

    def system_transform(self, positions: np.ndarray, scale_au: float = 1.7, y_frac: float = .47):
        cx = W * 0.5
        cy = H * y_frac
        radius = min(W * .41, H * .27)
        return cx, cy, radius / scale_au

    def draw_orbit_grid(self, img: Image.Image, scale_au: float, y_frac: float = .47):
        cx, cy, px_per_au = self.system_transform(np.empty((0,2)), scale_au, y_frac)
        d = ImageDraw.Draw(img)
        for au in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            if au > scale_au:
                continue
            rr = au * px_per_au
            d.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), outline=COLORS["white"]+(24,), width=max(1,int(1*SCALE)))
            if au in (1.0, 3.0, 5.0):
                draw_text(img, f"{au:g} AU", (int(cx+rr+6*SCALE), int(cy)), 10 if not QUICK_MODE else 5,
                          COLORS["muted"]+(120,), False, "lm", 1)

    def draw_system(self, img: Image.Image, sim_year: float, trails: bool = True, twin: bool = False,
                    scale_au: float = 1.7, y_frac: float = .47, show_forces: bool = False):
        idx = self.sim_index(sim_year)
        arr = self.a if not twin else self.b
        p = arr[idx]
        star = p[0]
        rel = p - star
        cx, cy, px_per_au = self.system_transform(rel, scale_au, y_frac)
        d = ImageDraw.Draw(img)

        if trails:
            start = max(0, idx - int(1.2 / SIM_SAMPLE_YEARS))
            stride = max(1, (idx-start)//90)
            for j in range(1, 1+N_PLANETS):
                pts = []
                for k in range(start, idx+1, stride):
                    rr = arr[k, j] - arr[k, 0]
                    pts.append((cx + rr[0]*px_per_au, cy + rr[1]*px_per_au))
                if len(pts) > 1:
                    d.line(pts, fill=PLANET_COLORS[j-1]+(95,), width=max(1,int(2*SCALE)))

        # star glow
        glow = Image.new("RGBA", SIZE, (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        for rad, alpha in [(42,28),(27,50),(17,90)]:
            rr = rad*SCALE
            gd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr), fill=COLORS["star"]+(alpha,))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4,int(10*SCALE)))))
        d = ImageDraw.Draw(img)
        rs = 11*SCALE
        d.ellipse((cx-rs,cy-rs,cx+rs,cy+rs), fill=COLORS["star_hot"]+(255,))

        if show_forces:
            for j in range(1, 1+N_PLANETS):
                q = rel[j]
                x = cx + q[0]*px_per_au
                y = cy + q[1]*px_per_au
                vx = cx - x; vy = cy - y
                norm = max(1e-6, math.hypot(vx,vy))
                ux,uy = vx/norm,vy/norm
                L = 36*SCALE
                d.line((x,y,x+ux*L,y+uy*L),fill=COLORS["red"]+(170,),width=max(1,int(3*SCALE)))
            # interplanetary tug lines for nearest pairs
            planet_rel = rel[1:]
            dist = np.linalg.norm(planet_rel[:,None]-planet_rel[None,:],axis=2)+np.eye(N_PLANETS)*999
            pairs = np.dstack(np.unravel_index(np.argsort(dist.ravel())[:8], dist.shape))[0]
            seen=set()
            for a,b in pairs:
                if a==b or (int(b),int(a)) in seen: continue
                seen.add((int(a),int(b)))
                qa,qb=planet_rel[a],planet_rel[b]
                xa,ya=cx+qa[0]*px_per_au,cy+qa[1]*px_per_au
                xb,yb=cx+qb[0]*px_per_au,cy+qb[1]*px_per_au
                d.line((xa,ya,xb,yb),fill=COLORS["violet"]+(65,),width=max(1,int(1*SCALE)))

        for j in range(1, 1+N_PLANETS):
            q = rel[j]
            x = cx + q[0]*px_per_au
            y = cy + q[1]*px_per_au
            rp = (7 + 1.5*(j%2))*SCALE
            col = PLANET_COLORS[j-1]
            d.ellipse((x-rp,y-rp,x+rp,y+rp),fill=col+(245,),outline=COLORS["white"]+(130,),width=max(1,int(1*SCALE)))

    def draw_packed(self, img: Image.Image, t: float, local: float):
        self.draw_orbit_grid(img, 1.7)
        self.draw_system(img, self.frame_to_sim_year(t), True, False, 1.7)
        self.panel(img, (int(W*.10), int(H*.69), int(W*.90), int(H*.78)), 164)
        draw_text(img, "6 GIANT PLANETS", (W//2, int(H*.715)), 23 if not QUICK_MODE else 11,
                  COLORS["red"]+(245,), True, "ma", 1)
        draw_text(img, "0.55 AU  →  1.39 AU", (W//2, int(H*.753)), 17 if not QUICK_MODE else 8,
                  COLORS["white"]+(225,), True, "ma", 1)

    def draw_many_body(self, img: Image.Image, t: float, local: float):
        self.draw_orbit_grid(img, 1.8)
        self.draw_system(img, self.frame_to_sim_year(t), True, False, 1.8, show_forces=True)
        draw_text(img, "NOT SIX SEPARATE TWO-BODY ORBITS", (W//2, int(H*.715)), 19 if not QUICK_MODE else 9,
                  COLORS["orange"]+(245,), True, "ma", 1)
        draw_text(img, "ALL 7 BODIES SHARE ONE GRAVITATIONAL PROBLEM", (W//2, int(H*.750)), 15 if not QUICK_MODE else 7,
                  COLORS["muted"]+(225,), True, "ma", 1)

    def draw_butterfly(self, img: Image.Image, t: float, local: float):
        sim_year = self.frame_to_sim_year(t)
        idx = self.sim_index(sim_year)
        # split into twin panels
        xmid = W//2
        d = ImageDraw.Draw(img)
        d.line((xmid, int(H*.20), xmid, int(H*.70)), fill=COLORS["white"]+(35,), width=max(1,int(2*SCALE)))
        # mini systems manually projected
        def mini(arr, xcenter, col_outline):
            p=arr[idx]; rel=p-p[0]
            scale=125*SCALE
            for au in [0.5,1.0,1.5]:
                rr=au*scale
                d.ellipse((xcenter-rr,int(H*.44)-rr,xcenter+rr,int(H*.44)+rr),outline=COLORS["white"]+(18,),width=1)
            rs=8*SCALE
            d.ellipse((xcenter-rs,int(H*.44)-rs,xcenter+rs,int(H*.44)+rs),fill=COLORS["star_hot"]+(250,))
            for j in range(1,7):
                x=xcenter+rel[j,0]*scale; y=int(H*.44)+rel[j,1]*scale; rp=5.5*SCALE
                d.ellipse((x-rp,y-rp,x+rp,y+rp),fill=PLANET_COLORS[j-1]+(235,),outline=col_outline+(150,),width=max(1,int(SCALE)))
        mini(self.a, int(W*.26), COLORS["cyan"])
        mini(self.b, int(W*.74), COLORS["magenta"])
        draw_text(img, "SYSTEM A", (int(W*.26), int(H*.235)), 16 if not QUICK_MODE else 8, COLORS["cyan"]+(240,), True, "ma", 1)
        draw_text(img, "+1.5 km CLONE", (int(W*.74), int(H*.235)), 16 if not QUICK_MODE else 8, COLORS["magenta"]+(240,), True, "ma", 1)
        div=float(self.metrics["divergence_au"][idx])
        self.panel(img,(int(W*.13),int(H*.69),int(W*.87),int(H*.78)),165)
        draw_text(img,f"SIM YEAR  {sim_year:05.2f}",(int(W*.18),int(H*.716)),15 if not QUICK_MODE else 7,COLORS["muted"]+(220,),True,"la",1)
        draw_text(img,f"TWIN SEPARATION  {div:.4f} AU",(int(W*.82),int(H*.750)),18 if not QUICK_MODE else 9,
                  COLORS["red"]+(245,) if div>.01 else COLORS["white"]+(235,),True,"ra",1)

    def draw_encounters(self, img: Image.Image, t: float, local: float):
        sy=self.frame_to_sim_year(t); idx=self.sim_index(sy)
        scale=max(1.8, min(3.0, float(self.metrics["max_radius_au"][idx])*1.15))
        self.draw_orbit_grid(img, scale)
        self.draw_system(img, sy, True, False, scale)
        minsep=float(self.metrics["min_pair_separation_au"][idx])
        self.panel(img,(int(W*.12),int(H*.69),int(W*.88),int(H*.78)),170)
        draw_text(img,f"SIM YEAR  {sy:05.2f}",(int(W*.17),int(H*.715)),15 if not QUICK_MODE else 7,COLORS["muted"]+(225,),True,"la",1)
        draw_text(img,f"CLOSEST PAIR  {minsep:.3f} AU",(int(W*.83),int(H*.750)),18 if not QUICK_MODE else 9,
                  COLORS["orange"]+(245,),True,"ra",1)

    def draw_ejection(self, img: Image.Image, t: float, local: float):
        sy=self.frame_to_sim_year(t); idx=self.sim_index(sy)
        maxr=float(self.metrics["max_radius_au"][idx])
        scale=max(2.2, min(11.0, maxr*1.12))
        self.draw_orbit_grid(img, scale)
        self.draw_system(img, sy, True, False, scale)
        self.panel(img,(int(W*.11),int(H*.69),int(W*.89),int(H*.785)),176)
        draw_text(img,f"SIM YEAR  {sy:05.1f}",(int(W*.17),int(H*.715)),16 if not QUICK_MODE else 8,COLORS["muted"]+(230,),True,"la",1)
        draw_text(img,f"FARTHEST PLANET  {maxr:05.2f} AU",(int(W*.83),int(H*.750)),20 if not QUICK_MODE else 10,
                  COLORS["red"]+(248,),True,"ra",1)

    def draw_outro(self, img: Image.Image, t: float, local: float):
        # divergence graph + ghost trajectories
        x0=int(W*.12); x1=int(W*.88); y0=int(H*.25); y1=int(H*.60)
        d=ImageDraw.Draw(img)
        self.panel(img,(x0-int(18*SCALE),y0-int(32*SCALE),x1+int(18*SCALE),y1+int(80*SCALE)),170)
        d.line((x0,y1,x1,y1),fill=COLORS["white"]+(75,),width=max(1,int(2*SCALE)))
        d.line((x0,y0,x0,y1),fill=COLORS["white"]+(75,),width=max(1,int(2*SCALE)))
        # log divergence chart from 1e-8 to few AU
        vals=np.maximum(self.metrics["divergence_au"],1e-9)
        reveal=max(3,int(lerp(3,len(vals),local)))
        pts=[]
        loglo,loghi=-8.5,0.6
        for i in np.linspace(0,reveal-1,min(reveal,420)).astype(int):
            x=lerp(x0,x1,self.times[i]/SIM_YEARS)
            lv=math.log10(float(vals[i]))
            y=lerp(y1,y0,clamp((lv-loglo)/(loghi-loglo)))
            pts.append((x,y))
        if len(pts)>1:
            d.line(pts,fill=COLORS["magenta"]+(240,),width=max(2,int(5*SCALE)))
        for year,label in [(0,"0"),(10,"10y"),(30,"30y"),(60,"60y")]:
            x=lerp(x0,x1,year/SIM_YEARS)
            draw_text(img,label,(int(x),y1+int(22*SCALE)),11 if not QUICK_MODE else 5,COLORS["muted"]+(190,),False,"ma",1)
        draw_text(img,"TWIN-SYSTEM DIVERGENCE",(W//2,y0-int(12*SCALE)),18 if not QUICK_MODE else 9,COLORS["cyan"]+(235,),True,"ma",1)
        draw_text(img,"tiny initial difference → macroscopic orbital difference",(W//2,y1+int(55*SCALE)),15 if not QUICK_MODE else 7,COLORS["white"]+(225,),True,"ma",1)

    def render(self, t: float) -> np.ndarray:
        img=self.background(t)
        shot=get_shot(t)
        local=clamp((t-shot["start"])/max(1e-9,shot["end"]-shot["start"]))
        self.draw_header(img,t)
        if shot["name"]=="packed": self.draw_packed(img,t,local)
        elif shot["name"]=="many_body": self.draw_many_body(img,t,local)
        elif shot["name"]=="butterfly": self.draw_butterfly(img,t,local)
        elif shot["name"]=="encounters": self.draw_encounters(img,t,local)
        elif shot["name"]=="ejection": self.draw_ejection(img,t,local)
        else: self.draw_outro(img,t,local)
        self.draw_caption(img,t)

        # Scanline / HUD texture
        overlay=Image.new("RGBA",SIZE,(0,0,0,0)); od=ImageDraw.Draw(overlay)
        offset=int((t*31)%9)
        for y in range(offset,H,9): od.line((0,y,W,y),fill=(120,205,240,7),width=1)
        img.alpha_composite(overlay)

        arr=np.asarray(img.convert("RGB")).astype(np.float32)
        arr*=VIGNETTE[...,None]
        arr=np.clip(arr,0,255).astype(np.uint8)
        graded=Image.fromarray(arr)
        graded=ImageEnhance.Contrast(graded).enhance(float(CONFIG["contrast"]))
        graded=ImageEnhance.Color(graded).enhance(float(CONFIG["saturation"]))
        return np.asarray(graded)


# =============================================================================
# Output pipeline
# =============================================================================

def save_preview_frames(scene: ChaosScene) -> List[Path]:
    preview_times=[]
    for shot in SHOT_PLAN:
        preview_times.append((shot["start"]+shot["end"])*0.5)
    paths=[]
    for i,t in enumerate(preview_times,1):
        arr=scene.render(min(DURATION-1/FPS,t))
        path=PREVIEW_ROOT/f"preview_{i:02d}_{get_shot(t)['name']}.jpg"
        Image.fromarray(arr).save(path,quality=92)
        paths.append(path)
    return paths


def make_contact_sheet(paths: Sequence[Path]) -> Path:
    thumbs=[]
    tw=300 if not QUICK_MODE else 200
    th=int(tw*16/9)
    for p in paths:
        im=Image.open(p).convert("RGB")
        im.thumbnail((tw,th),Image.Resampling.LANCZOS)
        canvas=Image.new("RGB",(tw,th),(5,8,18))
        canvas.paste(im,((tw-im.width)//2,(th-im.height)//2))
        thumbs.append(canvas)
    cols=3; rows=math.ceil(len(thumbs)/cols)
    sheet=Image.new("RGB",(tw*cols,th*rows),(3,6,15))
    for i,im in enumerate(thumbs): sheet.paste(im,((i%cols)*tw,(i//cols)*th))
    out=PREVIEW_ROOT/f"{CONFIG['basename']}_contact_sheet.jpg"
    sheet.save(out,quality=92)
    return out


def render_video(scene: ChaosScene) -> Path:
    out=OUTPUT_ROOT/f"{CONFIG['basename']}{'_quick_preview' if QUICK_MODE else ''}.mp4"
    total=max(1,int(round(DURATION*FPS)))
    writer=iio.get_writer(out,fps=FPS,codec="libx264",quality=8,macro_block_size=None,
                          output_params=["-pix_fmt","yuv420p","-movflags","+faststart"])
    try:
        for frame in tqdm(range(total),desc="Rendering chaotic system"):
            writer.append_data(scene.render(frame/FPS))
    finally:
        writer.close()
    return out


def write_readme(snapshot: ChaosSnapshot, outputs: Dict[str,str]) -> Path:
    path=OUTPUT_ROOT/"README.txt"
    lines=[
        CONFIG["title"],
        "="*len(CONFIG["title"]),
        "",
        "This is an intentionally unstable educational N-body toy system, not the real Solar System.",
        f"Each giant planet mass: {snapshot.planet_mass_jupiter_each_approx:.2f} Jupiter masses",
        f"Twin perturbation: {snapshot.perturbation_km_approx:.2f} km",
        f"Twin divergence >0.01 AU: {snapshot.divergence_0p01_au_year:.2f} simulated years" if snapshot.divergence_0p01_au_year is not None else "Twin divergence >0.01 AU: not reached",
        f"First planet beyond 3 AU: {snapshot.first_radius_gt_3au_year:.2f} simulated years" if snapshot.first_radius_gt_3au_year is not None else "First planet beyond 3 AU: not reached",
        "",
        "Outputs:",
    ]
    lines.extend([f"- {k}: {v}" for k,v in outputs.items()])
    path.write_text("\n".join(lines),encoding="utf-8")
    return path



# =============================================================================
# YouTube title / description sidecar
# =============================================================================

YOUTUBE_TITLE = 'The Most Chaotic Solar System I Could Simulate 🪐💥 #rootjatin'
YOUTUBE_DESCRIPTION = 'A deliberately unstable, hypothetical many-body system packs six super-Jupiter planets into a tight region around one star. A cloned run begins with one planet shifted by only a tiny distance, then the two trajectories diverge as close encounters exchange orbital energy and angular momentum. The setup is designed to demonstrate deterministic orbital chaos and sensitivity to initial conditions; it is not a real observed solar system.'
YOUTUBE_HASHTAGS = '#rootjatin #ChaosTheory #SolarSystem #NBody #Astronomy #Space #Simulation #Physics'

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

from __future__ import annotations

"""
The Most Chaotic Solar System I Could Simulate
==============================================

A cinematic vertical YouTube Short renderer built around an intentionally
unstable Newtonian N-body toy system: one Sun-like star plus six giant planets
packed far too closely together.

Scientific framing
------------------
- This is NOT a model of the real Solar System and not a claim about the
  "most chaotic" physically possible planetary system. The title is a cinematic
  framing for an intentionally unstable simulation.
- Gravity is Newtonian and every body pulls on every other body.
- The system uses one 1-solar-mass star and six equal 0.0015-solar-mass giant
  planets (about 1.57 Jupiter masses each) between 0.55 and 1.39 AU.
- Planetary orbits begin nearly circular but are much too tightly packed for
  long-term stability at these masses.
- A second copy of the system is started with one planet shifted outward by
  only 1e-8 AU (~1.5 km). The two simulations soon diverge dramatically.
- The integrator is velocity-Verlet / leapfrog-like with a small Plummer-style
  softening length (0.005 AU) so close encounters remain visually stable.
  Because of that softening, close-encounter details are educational rather
  than precision celestial-mechanics predictions.
- In the default deterministic run, the twin trajectories separate by more
  than 0.01 AU after roughly seven simulated years, and one planet eventually
  travels beyond several AU after repeated gravitational encounters. Exact
  encounter order is highly sensitive to numerical and initial-condition
  details -- which is itself part of the chaos lesson.

Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview
-------------
    CHAOTIC_SYSTEM_QUICK=1 python the_most_chaotic_solar_system_i_could_simulate.py

Full 1080x1920 render
---------------------
    python the_most_chaotic_solar_system_i_could_simulate.py

4K vertical render
------------------
    CHAOTIC_SYSTEM_4K=1 python the_most_chaotic_solar_system_i_could_simulate.py
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

QUICK_MODE = os.environ.get("CHAOTIC_SYSTEM_QUICK", "0") == "1"
FOUR_K = os.environ.get("CHAOTIC_SYSTEM_4K", "0") == "1" and not QUICK_MODE

W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 6 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SIZE = (W, H)
SCALE = W / 1080.0

OUTPUT_ROOT = Path("the_most_chaotic_solar_system_i_could_simulate_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "THE MOST CHAOTIC SOLAR SYSTEM I COULD SIMULATE",
    "subtitle": "six giant planets // packed too close // tiny changes explode",
    "basename": "the_most_chaotic_solar_system_i_could_simulate",
    "contrast": 1.12,
    "saturation": 1.10,
    "vignette": 0.29,
}

COLORS = {
    "space": (2, 4, 14),
    "space2": (8, 13, 35),
    "white": (248, 251, 255),
    "muted": (165, 190, 214),
    "cyan": (72, 230, 255),
    "blue": (75, 132, 255),
    "gold": (255, 204, 84),
    "orange": (255, 139, 70),
    "red": (255, 76, 102),
    "green": (103, 241, 174),
    "violet": (188, 124, 255),
    "magenta": (244, 88, 190),
    "star": (255, 224, 115),
    "star_hot": (255, 250, 220),
    "panel": (3, 8, 22),
}

PLANET_COLORS = [
    (98, 219, 255),
    (112, 147, 255),
    (129, 239, 179),
    (255, 210, 94),
    (255, 139, 80),
    (223, 112, 255),
]

SHOT_PLAN = [
    {"name": "packed", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "many_body", "start": 8.0 if not QUICK_MODE else 1.8, "end": 18.0 if not QUICK_MODE else 4.0},
    {"name": "butterfly", "start": 18.0 if not QUICK_MODE else 4.0, "end": 29.0 if not QUICK_MODE else 6.45},
    {"name": "encounters", "start": 29.0 if not QUICK_MODE else 6.45, "end": 40.0 if not QUICK_MODE else 8.9},
    {"name": "ejection", "start": 40.0 if not QUICK_MODE else 8.9, "end": 51.0 if not QUICK_MODE else 11.35},
    {"name": "outro", "start": 51.0 if not QUICK_MODE else 11.35, "end": DURATION},
]

CAPTION_TEXTS = [
    "I packed six super-Jupiter planets between 0.55 and 1.39 astronomical units. They start almost circular, but at these masses the spacing is a gravitational traffic jam.",
    "This is a true many-body problem: every planet pulls on the star, the star pulls on every planet, and every planet tugs on every other planet at the same time.",
    "Then I cloned the system and moved just one planet by about one and a half kilometers. For a while the two copies look identical. Then their trajectories separate explosively.",
    "Close encounters trade orbital energy and angular momentum. A planet can be kicked inward, thrown outward, or reshuffle the paths of several worlds in a single encounter.",
    "In this deliberately unstable run, one planet eventually wanders beyond several astronomical units. The exact escape sequence is sensitive to tiny numerical and initial differences.",
    "That is orbital chaos: deterministic rules, but rapidly vanishing predictability. Tiny changes in the starting state can grow into completely different planetary histories.",
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
# N-body toy model
# =============================================================================

G = 4.0 * math.pi**2  # AU^3 / (Msun * yr^2)
STAR_MASS = 1.0
PLANET_MASS = 0.0015
N_PLANETS = 6
MASSES = np.asarray([STAR_MASS] + [PLANET_MASS] * N_PLANETS, dtype=np.float64)
INITIAL_RADII_AU = np.asarray([0.55, 0.68, 0.82, 0.98, 1.17, 1.39], dtype=np.float64)
INITIAL_PHASES = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
SOFTENING_AU = 0.005
PERTURBATION_AU = 1e-8  # about 1.5 km
SIM_YEARS = 60.0
SIM_DT_YEARS = 0.0005
SIM_SAMPLE_YEARS = 0.01


@dataclass
class ChaosSnapshot:
    generated_at_utc: str
    star_mass_solar: float
    planet_mass_solar_each: float
    planet_mass_jupiter_each_approx: float
    initial_radii_au: List[float]
    softening_au: float
    perturbation_au: float
    perturbation_km_approx: float
    simulation_years: float
    timestep_days: float
    divergence_0p01_au_year: Optional[float]
    divergence_0p1_au_year: Optional[float]
    first_radius_gt_3au_year: Optional[float]
    first_radius_gt_5au_year: Optional[float]
    minimum_planet_pair_separation_au: float
    maximum_planet_radius_au: float
    scenario: str
    interpretation: str


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


def ease_in_out_sine(x: float) -> float:
    t = clamp(x)
    return -(math.cos(math.pi * t) - 1.0) / 2.0


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


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(7, int(size * SCALE)))
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


def initial_state(perturbation_au: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    n = 1 + N_PLANETS
    pos = np.zeros((n, 2), dtype=np.float64)
    vel = np.zeros((n, 2), dtype=np.float64)
    for i, (a0, phase) in enumerate(zip(INITIAL_RADII_AU, INITIAL_PHASES), start=1):
        a = float(a0 + (perturbation_au if i == 3 else 0.0))
        c, s = math.cos(float(phase)), math.sin(float(phase))
        pos[i] = (a * c, a * s)
        speed = math.sqrt(G * (STAR_MASS + MASSES[i]) / a)
        vel[i] = speed * np.asarray((-s, c))
    # Put the whole system approximately in the barycentric frame.
    vel[0] = -np.sum(MASSES[1:, None] * vel[1:], axis=0) / STAR_MASS
    pos[0] = -np.sum(MASSES[1:, None] * pos[1:], axis=0) / STAR_MASS
    return pos, vel


def accelerations(pos: np.ndarray) -> np.ndarray:
    # delta[i, j] = r_j - r_i
    delta = pos[None, :, :] - pos[:, None, :]
    r2 = np.sum(delta * delta, axis=2) + SOFTENING_AU**2
    inv_r3 = r2 ** -1.5
    np.fill_diagonal(inv_r3, 0.0)
    return G * np.sum(delta * (MASSES[None, :, None] * inv_r3[:, :, None]), axis=1)


def integrate_system(perturbation_au: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos, vel = initial_state(perturbation_au)
    acc = accelerations(pos)
    sample_every = max(1, int(round(SIM_SAMPLE_YEARS / SIM_DT_YEARS)))
    steps = int(round(SIM_YEARS / SIM_DT_YEARS))
    times: List[float] = []
    positions: List[np.ndarray] = []
    velocities: List[np.ndarray] = []

    for step in range(steps + 1):
        if step % sample_every == 0:
            times.append(step * SIM_DT_YEARS)
            positions.append(pos.copy())
            velocities.append(vel.copy())
        if step == steps:
            break
        pos = pos + vel * SIM_DT_YEARS + 0.5 * acc * SIM_DT_YEARS**2
        new_acc = accelerations(pos)
        vel = vel + 0.5 * (acc + new_acc) * SIM_DT_YEARS
        acc = new_acc

    return np.asarray(times), np.asarray(positions), np.asarray(velocities)


def first_time(times: np.ndarray, mask: np.ndarray) -> Optional[float]:
    idx = np.where(mask)[0]
    return None if len(idx) == 0 else float(times[int(idx[0])])


def analyze_simulation(times: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[ChaosSnapshot, Dict[str, np.ndarray]]:
    rel = a[:, 1:, :] - a[:, 0:1, :]
    radii = np.linalg.norm(rel, axis=2)
    pair = rel[:, :, None, :] - rel[:, None, :, :]
    pair_d = np.linalg.norm(pair, axis=3)
    pair_d[:, np.arange(N_PLANETS), np.arange(N_PLANETS)] = 1e9
    min_pair = pair_d.min(axis=(1, 2))

    divergence = np.sqrt(np.mean(np.sum((a[:, 1:, :] - b[:, 1:, :]) ** 2, axis=2), axis=1))
    max_radius = radii.max(axis=1)

    snapshot = ChaosSnapshot(
        generated_at_utc=iso_z(utc_now()),
        star_mass_solar=STAR_MASS,
        planet_mass_solar_each=PLANET_MASS,
        planet_mass_jupiter_each_approx=PLANET_MASS / 0.0009543,
        initial_radii_au=[float(x) for x in INITIAL_RADII_AU],
        softening_au=SOFTENING_AU,
        perturbation_au=PERTURBATION_AU,
        perturbation_km_approx=PERTURBATION_AU * 149_597_870.7,
        simulation_years=SIM_YEARS,
        timestep_days=SIM_DT_YEARS * 365.256,
        divergence_0p01_au_year=first_time(times, divergence > 0.01),
        divergence_0p1_au_year=first_time(times, divergence > 0.1),
        first_radius_gt_3au_year=first_time(times, max_radius > 3.0),
        first_radius_gt_5au_year=first_time(times, max_radius > 5.0),
        minimum_planet_pair_separation_au=float(min_pair.min()),
        maximum_planet_radius_au=float(max_radius.max()),
        scenario="Intentionally unstable six-giant-planet Newtonian toy system with two nearly identical initial conditions.",
        interpretation="The exact close-encounter sequence is not a precision prediction. The point is sensitive dependence: tiny initial differences become macroscopic orbital differences.",
    )
    metrics = {
        "divergence_au": divergence,
        "max_radius_au": max_radius,
        "min_pair_separation_au": min_pair,
        "planet_radii_au": radii,
    }
    return snapshot, metrics


def collect_data():
    times, positions_a, velocities_a = integrate_system(0.0)
    times_b, positions_b, velocities_b = integrate_system(PERTURBATION_AU)
    if not np.allclose(times, times_b):
        raise RuntimeError("Twin integrations produced mismatched sample times")
    snapshot, metrics = analyze_simulation(times, positions_a, positions_b)
    return snapshot, times, positions_a, velocities_a, positions_b, velocities_b, metrics


def save_data(
    snapshot: ChaosSnapshot,
    times: np.ndarray,
    positions_a: np.ndarray,
    metrics: Dict[str, np.ndarray],
) -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "chaos_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_years", "twin_divergence_au", "minimum_planet_pair_separation_au", "maximum_planet_radius_au"])
        stride = max(1, len(times) // 1200)
        for i in range(0, len(times), stride):
            writer.writerow([
                f"{times[i]:.6f}",
                f"{metrics['divergence_au'][i]:.9f}",
                f"{metrics['min_pair_separation_au'][i]:.9f}",
                f"{metrics['max_radius_au'][i]:.9f}",
            ])

    json_path = DATA_ROOT / "chaotic_system_snapshot.json"
    json_path.write_text(
        json.dumps(
            {
                "snapshot": asdict(snapshot),
                "notes": {
                    "units": "AU, solar masses, Julian years",
                    "integrator": "velocity-Verlet / leapfrog-like",
                    "gravity": "Newtonian inverse-square with Plummer-style softening",
                    "planet_count": N_PLANETS,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path, json_path


# =============================================================================
# Scene renderer
# =============================================================================

class ChaosScene:
    def __init__(
        self,
        snapshot: ChaosSnapshot,
        times: np.ndarray,
        positions_a: np.ndarray,
        positions_b: np.ndarray,
        metrics: Dict[str, np.ndarray],
    ):
        self.snapshot = snapshot
        self.times = times
        self.a = positions_a
        self.b = positions_b
        self.metrics = metrics
        rng = np.random.default_rng(20260921)
        self.stars = [
            (
                float(rng.uniform(0, W)),
                float(rng.uniform(0, H)),
                float(rng.uniform(0.4, 2.0) * SCALE),
                int(rng.uniform(15, 80)),
                float(rng.uniform(0, math.tau)),
            )
            for _ in range(160 if QUICK_MODE else 460)
        ]
        self.base_bg = self._make_background()

    def _make_background(self) -> Image.Image:
        top = np.asarray(COLORS["space"], dtype=np.float32)
        bottom = np.asarray(COLORS["space2"], dtype=np.float32)
        f = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        rgb = (top[None, :] * (1 - f) + bottom[None, :] * f).astype(np.uint8)
        arr = np.empty((H, W, 4), dtype=np.uint8)
        arr[..., :3] = rgb[:, None, :]
        arr[..., 3] = 255
        return Image.fromarray(arr, "RGBA")

    def background(self, t: float) -> Image.Image:
        img = self.base_bg.copy()
        d = ImageDraw.Draw(img)
        for x, y, r, a, phase in self.stars:
            alpha = int(a * (0.78 + 0.22 * math.sin(t * 0.85 + phase)))
            d.ellipse((x-r, y-r, x+r, y+r), fill=COLORS["white"] + (alpha,))
        glow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for cx, cy, col in [
            (W * 0.24, H * 0.30, COLORS["blue"]),
            (W * 0.76, H * 0.25, COLORS["violet"]),
            (W * 0.52, H * 0.72, COLORS["red"]),
        ]:
            rr = int(W * 0.28)
            gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=col + (15,))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(18, int(55*SCALE)))))
        return img

    @staticmethod
    def panel(img: Image.Image, box: Tuple[int, int, int, int], alpha: int = 176):
        layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle(
            box,
            radius=max(8, int(24 * SCALE)),
            fill=COLORS["panel"] + (alpha,),
            outline=COLORS["cyan"] + (48,),
            width=max(1, int(2 * SCALE)),
        )
        img.alpha_composite(layer)

    def draw_header(self, img: Image.Image, t: float):
        if t < (6.0 if not QUICK_MODE else 1.35):
            fade = smoothstep(t / (0.75 if not QUICK_MODE else 0.18))
            draw_text(img, "THE MOST CHAOTIC", (W//2, int(H*0.055)), 40 if not QUICK_MODE else 20,
                      COLORS["white"] + (int(245*fade),), True, "ma", 2)
            draw_text(img, "SOLAR SYSTEM", (W//2, int(H*0.100)), 54 if not QUICK_MODE else 27,
                      COLORS["red"] + (int(250*fade),), True, "ma", 2)
            draw_text(img, "I COULD SIMULATE", (W//2, int(H*0.145)), 34 if not QUICK_MODE else 17,
                      COLORS["cyan"] + (int(240*fade),), True, "ma", 2)
        else:
            labels = {
                "packed": "SIX GIANTS // TOO CLOSE",
                "many_body": "EVERY BODY PULLS ON EVERY BODY",
                "butterfly": "A 1.5 km CHANGE",
                "encounters": "CLOSE ENCOUNTERS TRADE ENERGY",
                "ejection": "ONE KICK CAN REWRITE THE SYSTEM",
                "outro": "DETERMINISTIC ≠ PREDICTABLE",
            }
            draw_text(img, labels[get_shot(t)["name"]], (int(W*.055), int(H*.045)), 17 if not QUICK_MODE else 8,
                      COLORS["muted"] + (230,), True, "la", 1)
        draw_text(img, "NEWTONIAN N-BODY // INTENTIONALLY UNSTABLE TOY MODEL",
                  (int(W*.945), int(H*.047)), 12 if not QUICK_MODE else 6,
                  COLORS["cyan"] + (205,), True, "ra", 1)

    def draw_caption(self, img: Image.Image, t: float):
        cap = caption_at(t)
        if not cap:
            return
        y0 = H - int(170 * SCALE)
        self.panel(img, (int(58*SCALE), y0, W-int(58*SCALE), y0+int(116*SCALE)), 150)
        draw_wrapped_text(img, cap, (int(82*SCALE), y0+int(18*SCALE)), W-int(164*SCALE),
                          30, COLORS["white"]+(240,), False, 4)

    def sim_index(self, sim_year: float) -> int:
        return int(np.clip(np.searchsorted(self.times, sim_year), 0, len(self.times)-1))

    def frame_to_sim_year(self, t: float) -> float:
        shot = get_shot(t)
        local = clamp((t - shot["start"]) / max(1e-9, shot["end"] - shot["start"]))
        if shot["name"] == "packed":
            return lerp(0.0, 1.0, local)
        if shot["name"] == "many_body":
            return lerp(0.0, 3.0, local)
        if shot["name"] == "butterfly":
            return lerp(0.0, 10.0, local)
        if shot["name"] == "encounters":
            return lerp(0.0, 18.0, local)
        if shot["name"] == "ejection":
            return lerp(15.0, 55.0, local)
        return lerp(0.0, 60.0, local)

    def system_transform(self, positions: np.ndarray, scale_au: float = 1.7, y_frac: float = .47):
        cx = W * 0.5
        cy = H * y_frac
        radius = min(W * .41, H * .27)
        return cx, cy, radius / scale_au

    def draw_orbit_grid(self, img: Image.Image, scale_au: float, y_frac: float = .47):
        cx, cy, px_per_au = self.system_transform(np.empty((0,2)), scale_au, y_frac)
        d = ImageDraw.Draw(img)
        for au in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            if au > scale_au:
                continue
            rr = au * px_per_au
            d.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), outline=COLORS["white"]+(24,), width=max(1,int(1*SCALE)))
            if au in (1.0, 3.0, 5.0):
                draw_text(img, f"{au:g} AU", (int(cx+rr+6*SCALE), int(cy)), 10 if not QUICK_MODE else 5,
                          COLORS["muted"]+(120,), False, "lm", 1)

    def draw_system(self, img: Image.Image, sim_year: float, trails: bool = True, twin: bool = False,
                    scale_au: float = 1.7, y_frac: float = .47, show_forces: bool = False):
        idx = self.sim_index(sim_year)
        arr = self.a if not twin else self.b
        p = arr[idx]
        star = p[0]
        rel = p - star
        cx, cy, px_per_au = self.system_transform(rel, scale_au, y_frac)
        d = ImageDraw.Draw(img)

        if trails:
            start = max(0, idx - int(1.2 / SIM_SAMPLE_YEARS))
            stride = max(1, (idx-start)//90)
            for j in range(1, 1+N_PLANETS):
                pts = []
                for k in range(start, idx+1, stride):
                    rr = arr[k, j] - arr[k, 0]
                    pts.append((cx + rr[0]*px_per_au, cy + rr[1]*px_per_au))
                if len(pts) > 1:
                    d.line(pts, fill=PLANET_COLORS[j-1]+(95,), width=max(1,int(2*SCALE)))

        # star glow
        glow = Image.new("RGBA", SIZE, (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        for rad, alpha in [(42,28),(27,50),(17,90)]:
            rr = rad*SCALE
            gd.ellipse((cx-rr,cy-rr,cx+rr,cy+rr), fill=COLORS["star"]+(alpha,))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4,int(10*SCALE)))))
        d = ImageDraw.Draw(img)
        rs = 11*SCALE
        d.ellipse((cx-rs,cy-rs,cx+rs,cy+rs), fill=COLORS["star_hot"]+(255,))

        if show_forces:
            for j in range(1, 1+N_PLANETS):
                q = rel[j]
                x = cx + q[0]*px_per_au
                y = cy + q[1]*px_per_au
                vx = cx - x; vy = cy - y
                norm = max(1e-6, math.hypot(vx,vy))
                ux,uy = vx/norm,vy/norm
                L = 36*SCALE
                d.line((x,y,x+ux*L,y+uy*L),fill=COLORS["red"]+(170,),width=max(1,int(3*SCALE)))
            # interplanetary tug lines for nearest pairs
            planet_rel = rel[1:]
            dist = np.linalg.norm(planet_rel[:,None]-planet_rel[None,:],axis=2)+np.eye(N_PLANETS)*999
            pairs = np.dstack(np.unravel_index(np.argsort(dist.ravel())[:8], dist.shape))[0]
            seen=set()
            for a,b in pairs:
                if a==b or (int(b),int(a)) in seen: continue
                seen.add((int(a),int(b)))
                qa,qb=planet_rel[a],planet_rel[b]
                xa,ya=cx+qa[0]*px_per_au,cy+qa[1]*px_per_au
                xb,yb=cx+qb[0]*px_per_au,cy+qb[1]*px_per_au
                d.line((xa,ya,xb,yb),fill=COLORS["violet"]+(65,),width=max(1,int(1*SCALE)))

        for j in range(1, 1+N_PLANETS):
            q = rel[j]
            x = cx + q[0]*px_per_au
            y = cy + q[1]*px_per_au
            rp = (7 + 1.5*(j%2))*SCALE
            col = PLANET_COLORS[j-1]
            d.ellipse((x-rp,y-rp,x+rp,y+rp),fill=col+(245,),outline=COLORS["white"]+(130,),width=max(1,int(1*SCALE)))

    def draw_packed(self, img: Image.Image, t: float, local: float):
        self.draw_orbit_grid(img, 1.7)
        self.draw_system(img, self.frame_to_sim_year(t), True, False, 1.7)
        self.panel(img, (int(W*.10), int(H*.69), int(W*.90), int(H*.78)), 164)
        draw_text(img, "6 GIANT PLANETS", (W//2, int(H*.715)), 23 if not QUICK_MODE else 11,
                  COLORS["red"]+(245,), True, "ma", 1)
        draw_text(img, "0.55 AU  →  1.39 AU", (W//2, int(H*.753)), 17 if not QUICK_MODE else 8,
                  COLORS["white"]+(225,), True, "ma", 1)

    def draw_many_body(self, img: Image.Image, t: float, local: float):
        self.draw_orbit_grid(img, 1.8)
        self.draw_system(img, self.frame_to_sim_year(t), True, False, 1.8, show_forces=True)
        draw_text(img, "NOT SIX SEPARATE TWO-BODY ORBITS", (W//2, int(H*.715)), 19 if not QUICK_MODE else 9,
                  COLORS["orange"]+(245,), True, "ma", 1)
        draw_text(img, "ALL 7 BODIES SHARE ONE GRAVITATIONAL PROBLEM", (W//2, int(H*.750)), 15 if not QUICK_MODE else 7,
                  COLORS["muted"]+(225,), True, "ma", 1)

    def draw_butterfly(self, img: Image.Image, t: float, local: float):
        sim_year = self.frame_to_sim_year(t)
        idx = self.sim_index(sim_year)
        # split into twin panels
        xmid = W//2
        d = ImageDraw.Draw(img)
        d.line((xmid, int(H*.20), xmid, int(H*.70)), fill=COLORS["white"]+(35,), width=max(1,int(2*SCALE)))
        # mini systems manually projected
        def mini(arr, xcenter, col_outline):
            p=arr[idx]; rel=p-p[0]
            scale=125*SCALE
            for au in [0.5,1.0,1.5]:
                rr=au*scale
                d.ellipse((xcenter-rr,int(H*.44)-rr,xcenter+rr,int(H*.44)+rr),outline=COLORS["white"]+(18,),width=1)
            rs=8*SCALE
            d.ellipse((xcenter-rs,int(H*.44)-rs,xcenter+rs,int(H*.44)+rs),fill=COLORS["star_hot"]+(250,))
            for j in range(1,7):
                x=xcenter+rel[j,0]*scale; y=int(H*.44)+rel[j,1]*scale; rp=5.5*SCALE
                d.ellipse((x-rp,y-rp,x+rp,y+rp),fill=PLANET_COLORS[j-1]+(235,),outline=col_outline+(150,),width=max(1,int(SCALE)))
        mini(self.a, int(W*.26), COLORS["cyan"])
        mini(self.b, int(W*.74), COLORS["magenta"])
        draw_text(img, "SYSTEM A", (int(W*.26), int(H*.235)), 16 if not QUICK_MODE else 8, COLORS["cyan"]+(240,), True, "ma", 1)
        draw_text(img, "+1.5 km CLONE", (int(W*.74), int(H*.235)), 16 if not QUICK_MODE else 8, COLORS["magenta"]+(240,), True, "ma", 1)
        div=float(self.metrics["divergence_au"][idx])
        self.panel(img,(int(W*.13),int(H*.69),int(W*.87),int(H*.78)),165)
        draw_text(img,f"SIM YEAR  {sim_year:05.2f}",(int(W*.18),int(H*.716)),15 if not QUICK_MODE else 7,COLORS["muted"]+(220,),True,"la",1)
        draw_text(img,f"TWIN SEPARATION  {div:.4f} AU",(int(W*.82),int(H*.750)),18 if not QUICK_MODE else 9,
                  COLORS["red"]+(245,) if div>.01 else COLORS["white"]+(235,),True,"ra",1)

    def draw_encounters(self, img: Image.Image, t: float, local: float):
        sy=self.frame_to_sim_year(t); idx=self.sim_index(sy)
        scale=max(1.8, min(3.0, float(self.metrics["max_radius_au"][idx])*1.15))
        self.draw_orbit_grid(img, scale)
        self.draw_system(img, sy, True, False, scale)
        minsep=float(self.metrics["min_pair_separation_au"][idx])
        self.panel(img,(int(W*.12),int(H*.69),int(W*.88),int(H*.78)),170)
        draw_text(img,f"SIM YEAR  {sy:05.2f}",(int(W*.17),int(H*.715)),15 if not QUICK_MODE else 7,COLORS["muted"]+(225,),True,"la",1)
        draw_text(img,f"CLOSEST PAIR  {minsep:.3f} AU",(int(W*.83),int(H*.750)),18 if not QUICK_MODE else 9,
                  COLORS["orange"]+(245,),True,"ra",1)

    def draw_ejection(self, img: Image.Image, t: float, local: float):
        sy=self.frame_to_sim_year(t); idx=self.sim_index(sy)
        maxr=float(self.metrics["max_radius_au"][idx])
        scale=max(2.2, min(11.0, maxr*1.12))
        self.draw_orbit_grid(img, scale)
        self.draw_system(img, sy, True, False, scale)
        self.panel(img,(int(W*.11),int(H*.69),int(W*.89),int(H*.785)),176)
        draw_text(img,f"SIM YEAR  {sy:05.1f}",(int(W*.17),int(H*.715)),16 if not QUICK_MODE else 8,COLORS["muted"]+(230,),True,"la",1)
        draw_text(img,f"FARTHEST PLANET  {maxr:05.2f} AU",(int(W*.83),int(H*.750)),20 if not QUICK_MODE else 10,
                  COLORS["red"]+(248,),True,"ra",1)

    def draw_outro(self, img: Image.Image, t: float, local: float):
        # divergence graph + ghost trajectories
        x0=int(W*.12); x1=int(W*.88); y0=int(H*.25); y1=int(H*.60)
        d=ImageDraw.Draw(img)
        self.panel(img,(x0-int(18*SCALE),y0-int(32*SCALE),x1+int(18*SCALE),y1+int(80*SCALE)),170)
        d.line((x0,y1,x1,y1),fill=COLORS["white"]+(75,),width=max(1,int(2*SCALE)))
        d.line((x0,y0,x0,y1),fill=COLORS["white"]+(75,),width=max(1,int(2*SCALE)))
        # log divergence chart from 1e-8 to few AU
        vals=np.maximum(self.metrics["divergence_au"],1e-9)
        reveal=max(3,int(lerp(3,len(vals),local)))
        pts=[]
        loglo,loghi=-8.5,0.6
        for i in np.linspace(0,reveal-1,min(reveal,420)).astype(int):
            x=lerp(x0,x1,self.times[i]/SIM_YEARS)
            lv=math.log10(float(vals[i]))
            y=lerp(y1,y0,clamp((lv-loglo)/(loghi-loglo)))
            pts.append((x,y))
        if len(pts)>1:
            d.line(pts,fill=COLORS["magenta"]+(240,),width=max(2,int(5*SCALE)))
        for year,label in [(0,"0"),(10,"10y"),(30,"30y"),(60,"60y")]:
            x=lerp(x0,x1,year/SIM_YEARS)
            draw_text(img,label,(int(x),y1+int(22*SCALE)),11 if not QUICK_MODE else 5,COLORS["muted"]+(190,),False,"ma",1)
        draw_text(img,"TWIN-SYSTEM DIVERGENCE",(W//2,y0-int(12*SCALE)),18 if not QUICK_MODE else 9,COLORS["cyan"]+(235,),True,"ma",1)
        draw_text(img,"tiny initial difference → macroscopic orbital difference",(W//2,y1+int(55*SCALE)),15 if not QUICK_MODE else 7,COLORS["white"]+(225,),True,"ma",1)

    def render(self, t: float) -> np.ndarray:
        img=self.background(t)
        shot=get_shot(t)
        local=clamp((t-shot["start"])/max(1e-9,shot["end"]-shot["start"]))
        self.draw_header(img,t)
        if shot["name"]=="packed": self.draw_packed(img,t,local)
        elif shot["name"]=="many_body": self.draw_many_body(img,t,local)
        elif shot["name"]=="butterfly": self.draw_butterfly(img,t,local)
        elif shot["name"]=="encounters": self.draw_encounters(img,t,local)
        elif shot["name"]=="ejection": self.draw_ejection(img,t,local)
        else: self.draw_outro(img,t,local)
        self.draw_caption(img,t)

        # Scanline / HUD texture
        overlay=Image.new("RGBA",SIZE,(0,0,0,0)); od=ImageDraw.Draw(overlay)
        offset=int((t*31)%9)
        for y in range(offset,H,9): od.line((0,y,W,y),fill=(120,205,240,7),width=1)
        img.alpha_composite(overlay)

        arr=np.asarray(img.convert("RGB")).astype(np.float32)
        arr*=VIGNETTE[...,None]
        arr=np.clip(arr,0,255).astype(np.uint8)
        graded=Image.fromarray(arr)
        graded=ImageEnhance.Contrast(graded).enhance(float(CONFIG["contrast"]))
        graded=ImageEnhance.Color(graded).enhance(float(CONFIG["saturation"]))
        return np.asarray(graded)


# =============================================================================
# Output pipeline
# =============================================================================

def save_preview_frames(scene: ChaosScene) -> List[Path]:
    preview_times=[]
    for shot in SHOT_PLAN:
        preview_times.append((shot["start"]+shot["end"])*0.5)
    paths=[]
    for i,t in enumerate(preview_times,1):
        arr=scene.render(min(DURATION-1/FPS,t))
        path=PREVIEW_ROOT/f"preview_{i:02d}_{get_shot(t)['name']}.jpg"
        Image.fromarray(arr).save(path,quality=92)
        paths.append(path)
    return paths


def make_contact_sheet(paths: Sequence[Path]) -> Path:
    thumbs=[]
    tw=300 if not QUICK_MODE else 200
    th=int(tw*16/9)
    for p in paths:
        im=Image.open(p).convert("RGB")
        im.thumbnail((tw,th),Image.Resampling.LANCZOS)
        canvas=Image.new("RGB",(tw,th),(5,8,18))
        canvas.paste(im,((tw-im.width)//2,(th-im.height)//2))
        thumbs.append(canvas)
    cols=3; rows=math.ceil(len(thumbs)/cols)
    sheet=Image.new("RGB",(tw*cols,th*rows),(3,6,15))
    for i,im in enumerate(thumbs): sheet.paste(im,((i%cols)*tw,(i//cols)*th))
    out=PREVIEW_ROOT/f"{CONFIG['basename']}_contact_sheet.jpg"
    sheet.save(out,quality=92)
    return out


def render_video(scene: ChaosScene) -> Path:
    out=OUTPUT_ROOT/f"{CONFIG['basename']}{'_quick_preview' if QUICK_MODE else ''}.mp4"
    total=max(1,int(round(DURATION*FPS)))
    writer=iio.get_writer(out,fps=FPS,codec="libx264",quality=8,macro_block_size=None,
                          output_params=["-pix_fmt","yuv420p","-movflags","+faststart"])
    try:
        for frame in tqdm(range(total),desc="Rendering chaotic system"):
            writer.append_data(scene.render(frame/FPS))
    finally:
        writer.close()
    return out


def write_readme(snapshot: ChaosSnapshot, outputs: Dict[str,str]) -> Path:
    path=OUTPUT_ROOT/"README.txt"
    lines=[
        CONFIG["title"],
        "="*len(CONFIG["title"]),
        "",
        "This is an intentionally unstable educational N-body toy system, not the real Solar System.",
        f"Each giant planet mass: {snapshot.planet_mass_jupiter_each_approx:.2f} Jupiter masses",
        f"Twin perturbation: {snapshot.perturbation_km_approx:.2f} km",
        f"Twin divergence >0.01 AU: {snapshot.divergence_0p01_au_year:.2f} simulated years" if snapshot.divergence_0p01_au_year is not None else "Twin divergence >0.01 AU: not reached",
        f"First planet beyond 3 AU: {snapshot.first_radius_gt_3au_year:.2f} simulated years" if snapshot.first_radius_gt_3au_year is not None else "First planet beyond 3 AU: not reached",
        "",
        "Outputs:",
    ]
    lines.extend([f"- {k}: {v}" for k,v in outputs.items()])
    path.write_text("\n".join(lines),encoding="utf-8")
    return path



# =============================================================================
# YouTube title / description sidecar
# =============================================================================

YOUTUBE_TITLE = 'The Most Chaotic Solar System I Could Simulate 🪐💥 #rootjatin'
YOUTUBE_DESCRIPTION = 'A deliberately unstable, hypothetical many-body system packs six super-Jupiter planets into a tight region around one star. A cloned run begins with one planet shifted by only a tiny distance, then the two trajectories diverge as close encounters exchange orbital energy and angular momentum. The setup is designed to demonstrate deterministic orbital chaos and sensitivity to initial conditions; it is not a real observed solar system.'
YOUTUBE_HASHTAGS = '#rootjatin #ChaosTheory #SolarSystem #NBody #Astronomy #Space #Simulation #Physics'

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
    snapshot,times,a,va,b,vb,metrics=collect_data()
    csv_path,json_path=save_data(snapshot,times,a,metrics)
    srt_path=write_srt(OUTPUT_ROOT/f"{CONFIG['basename']}_subtitles.srt")
    scene=ChaosScene(snapshot,times,a,b,metrics)
    previews=save_preview_frames(scene)
    contact=make_contact_sheet(previews)
    video=render_video(scene)
    outputs={
        "video": str(video),
        "subtitles": str(srt_path),
        "contact_sheet": str(contact),
        "metrics_csv": str(csv_path),
        "snapshot_json": str(json_path),
    }
    readme=write_readme(snapshot,outputs)
    outputs["readme"]=str(readme)
    print(json.dumps({"snapshot":asdict(snapshot),"outputs":outputs},indent=2))
    metadata_txt = write_youtube_metadata_txt()
    print("Title/description TXT:", metadata_txt.resolve())


if __name__ == "__main__":
    main()


