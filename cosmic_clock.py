from __future__ import annotations

"""
HOW DOES A COSMIC CLOCK WORK? — cinematic YouTube Shorts renderer + HackRF One pulsar demo

Concept
-------
This short explains why pulsars are often called "cosmic clocks". A rotating neutron star
sweeps a narrow beam past Earth. Each detected pulse becomes a timestamp. Repeated pulse
arrival times can be compared with a timing model, and tiny early/late residuals can reveal
motion, clock error, propagation effects, or gravitational phenomena.

The video is a SIMULATION / explainer. It does not claim that a HackRF One can receive
astronomical pulsars with a normal antenna. Real pulsar radio astronomy generally requires
large collecting area, low-noise front ends, filtering, long integrations, and specialized
signal processing.

HackRF mode
-----------
The script can optionally generate a synthetic 8-bit signed complex-IQ pulse train suitable
for hackrf_transfer. This is intended for a CLOSED BENCH TEST ONLY:

    HackRF TX -> 40 to 60 dB RF attenuation -> SDR/HackRF RX

Do not connect a transmit antenna. Do not radiate the test signal. Choose RF settings that
comply with your local regulations and lab setup.

Install
-------
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick cinematic preview
-----------------------
    COSMIC_CLOCK_QUICK=1 python how_cosmic_clock_works_hackrf_one.py

Generate HackRF IQ only
-----------------------
    COSMIC_CLOCK_IQ_ONLY=1 python how_cosmic_clock_works_hackrf_one.py

Optional HackRF settings
------------------------
    COSMIC_CLOCK_SAMPLE_RATE=2000000
    COSMIC_CLOCK_RF_HZ=915000000
    COSMIC_CLOCK_PULSE_PERIOD=0.125
    COSMIC_CLOCK_IQ_DURATION=10
    COSMIC_CLOCK_DRIFT_PPM=0
    COSMIC_CLOCK_NOISE=0.035

The generated IQ file contains interleaved signed int8 I,Q samples (.cs8).
Transmit it only through a cabled/attenuated bench path, for example:

    hackrf_transfer -t cosmic_clock_hackrf.cs8 -f 915000000 -s 2000000 -x 0

Use the lowest practical TX gain and enough external attenuation. Verify levels before
connecting a second receiver.

Outputs
-------
- final vertical MP4
- silent MP4 fallback
- ambient WAV
- SRT subtitle sidecar
- preview PNG frames
- timing CSV
- JSON summary/source notes
- optional HackRF .cs8 IQ waveform
"""

import csv
import json
import math
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("COSMIC_CLOCK_QUICK", "0") == "1"
IQ_ONLY = os.environ.get("COSMIC_CLOCK_IQ_ONLY", "0") == "1"

OUTPUT_ROOT = Path("how_cosmic_clock_works_hackrf_one_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for d in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    d.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12.0 if QUICK_MODE else 58.0,
    "title": "HOW DOES A COSMIC CLOCK WORK?",
    "subtitle": "PULSAR TIMING // HACKRF ONE LAB SIMULATION",
    "output_basename": "how_cosmic_clock_works",
    "sample_rate_audio": 22050 if QUICK_MODE else 44100,
    "grain_strength": 3.0,
    "contrast": 1.10,
    "saturation": 1.06,
    "vignette": 0.34,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
SCALE = OUT_W / 1080.0

COLORS = {

}

FULL_SHOT_PLAN = [

]

FULL_CAPTIONS = [

]

if QUICK_MODE:
    k = float(CONFIG["duration_s"]) / 58.0
    SHOT_PLAN = [{**s, "start": s["start"] * k, "end": s["end"] * k} for s in FULL_SHOT_PLAN]
    CAPTIONS = [(a * k, b * k, text) for a, b, text in FULL_CAPTIONS]
else:
    SHOT_PLAN = FULL_SHOT_PLAN
    CAPTIONS = FULL_CAPTIONS


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def smoothstep(x: float) -> float:
    x = clamp(x)
    return x * x * (3.0 - 2.0 * x)


def smootherstep(x: float) -> float:
    x = clamp(x)
    return x * x * x * (x * (x * 6 - 15) + 10)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if float(shot["start"]) <= t < float(shot["end"]):
            return shot
    return SHOT_PLAN[-1]


def shot_progress(t: float, shot: Dict[str, Any]) -> float:
    return clamp((t - float(shot["start"])) / max(float(shot["end"] - shot["start"]), 1e-9))


def get_font(size: int, bold: bool = False, condensed: bool = False):
    paths: List[str] = []
    if condensed and bold:
        paths += ["/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf", "DejaVuSansCondensed-Bold.ttf"]
    if condensed:
        paths += ["/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf", "DejaVuSansCondensed.ttf"]
    paths += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size=max(5, int(size)))
        except Exception:
            pass
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
        xy, text,
        font=get_font(size, bold, condensed),
        fill=fill, anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(235, fill[3])),
    )


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for i, (a, b, text) in enumerate(captions, 1):
        lines.extend([str(i), f"{format_srt_time(a)} --> {format_srt_time(b)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    r = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * r ** 1.8, 0.0, 1.0).astype(np.float32)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Synthetic cosmic-clock timing model
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Tick:
    index: int
    expected_s: float
    observed_s: float
    residual_us: float
    amplitude: float


def generate_ticks(count: int = 150, period_s: float = 0.125) -> List[Tick]:
    rng = np.random.default_rng(1974)
    ticks: List[Tick] = []
    for i in range(count):
        expected = i * period_s
        slow_orbit = 24.0 * math.sin(i * 0.083)       # microseconds
        clock_wander = 6.0 * math.sin(i * 0.017 + 1.1)
        measurement = float(rng.normal(0.0, 3.2))
        residual_us = slow_orbit + clock_wander + measurement
        observed = expected + residual_us * 1e-6
        amplitude = float(np.clip(rng.normal(1.0, 0.16), 0.45, 1.45))
        ticks.append(Tick(i, expected, observed, residual_us, amplitude))
    return ticks


TICKS = generate_ticks()


def save_timing_products() -> Tuple[Path, Path]:
    csv_path = DATA_ROOT / "simulated_pulsar_timing.csv"
    json_path = DATA_ROOT / "cosmic_clock_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pulse_index", "expected_s", "observed_s", "residual_us", "amplitude"])
        for tick in TICKS:
            w.writerow([tick.index, f"{tick.expected_s:.9f}", f"{tick.observed_s:.9f}", f"{tick.residual_us:.4f}", f"{tick.amplitude:.4f}"])
    summary = {
        "title": CONFIG["title"],
        "simulation": True,
        "pulse_period_s": 0.125,
        "pulse_count": len(TICKS),
        "residual_rms_us": float(np.sqrt(np.mean([t.residual_us ** 2 for t in TICKS]))),
        "scientific_note": "Pulsar timing compares measured pulse times-of-arrival against a timing model. This renderer uses synthetic pulses and residuals for explanation.",
        "hackrf_note": "HackRF output is a laboratory pulse-train simulation, not an astronomical pulsar observation.",
        "source_urls": {
            "pulsar_timing_context": "https://www.jb.man.ac.uk/doublepulsar/papers/physworld.pdf",
            "hackrf_one": "https://greatscottgadgets.com/hackrf/one/",
            "hackrf_clock": "https://hackrf.readthedocs.io/en/latest/external_clock_interface.html",
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path


# -----------------------------------------------------------------------------
# HackRF IQ generator
# -----------------------------------------------------------------------------

def generate_hackrf_iq() -> Path:
    sample_rate = int(os.environ.get("COSMIC_CLOCK_SAMPLE_RATE", "2000000"))
    duration_s = float(os.environ.get("COSMIC_CLOCK_IQ_DURATION", "10" if not QUICK_MODE else "3"))
    period_s = float(os.environ.get("COSMIC_CLOCK_PULSE_PERIOD", "0.125"))
    drift_ppm = float(os.environ.get("COSMIC_CLOCK_DRIFT_PPM", "0"))
    noise_level = float(os.environ.get("COSMIC_CLOCK_NOISE", "0.035"))

    if sample_rate < 2_000_000:
        raise ValueError("HackRF One normally uses sample rates of at least 2 Msps; set COSMIC_CLOCK_SAMPLE_RATE >= 2000000.")
    if duration_s <= 0 or period_s <= 0:
        raise ValueError("IQ duration and pulse period must be positive.")

    n = int(round(sample_rate * duration_s))
    path = DATA_ROOT / "cosmic_clock_hackrf.cs8"

    # Stream in chunks to avoid large memory usage in full mode.
    chunk = min(sample_rate // 2, 500_000)
    rng = np.random.default_rng(424242)
    pulse_sigma_s = min(0.006, period_s * 0.07)
    carrier_offset_hz = min(110_000.0, sample_rate * 0.08)

    with path.open("wb") as f:
        start = 0
        while start < n:
            length = min(chunk, n - start)
            idx = np.arange(start, start + length, dtype=np.float64)
            time = idx / sample_rate

            # Tiny programmable period drift for timing demonstrations.
            effective_period = period_s * (1.0 + drift_ppm * 1e-6)
            phase_in_period = np.mod(time + effective_period / 2, effective_period) - effective_period / 2
            envelope = np.exp(-0.5 * (phase_in_period / pulse_sigma_s) ** 2)

            # Add a weak two-component profile, similar to a stylized pulsar pulse shape.
            secondary_phase = np.mod(time - 0.018 + effective_period / 2, effective_period) - effective_period / 2
            envelope += 0.34 * np.exp(-0.5 * (secondary_phase / (pulse_sigma_s * 0.62)) ** 2)

            # Complex low-IF tone under the amplitude envelope.
            phi = 2.0 * np.pi * carrier_offset_hz * time
            sig = 0.72 * envelope * np.exp(1j * phi)

            # Low-level receiver-like complex noise.
            noise = noise_level * (rng.normal(size=length) + 1j * rng.normal(size=length))
            sig = np.clip(sig.real + noise.real, -0.95, 0.95) + 1j * np.clip(sig.imag + noise.imag, -0.95, 0.95)

            iq = np.empty(length * 2, dtype=np.int8)
            iq[0::2] = np.round(sig.real * 120).astype(np.int8)
            iq[1::2] = np.round(sig.imag * 120).astype(np.int8)
            f.write(iq.tobytes())
            start += length

    return path


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class CosmicClockScene:
    def __init__(self):
        self.rng = np.random.default_rng(8841)
        self.stars = [
            (float(self.rng.uniform(0, OUT_W)), float(self.rng.uniform(0, OUT_H)),
             float(self.rng.uniform(0.4, 2.2 if not QUICK_MODE else 1.4)),
             float(self.rng.uniform(18, 90)))
            for _ in range(240 if not QUICK_MODE else 80)
        ]
        self.dust = [
            (float(self.rng.uniform(0, OUT_W)), float(self.rng.uniform(0, OUT_H)), float(self.rng.uniform(0, math.tau)))
            for _ in range(120 if not QUICK_MODE else 35)
        ]

    def background(self, t: float) -> Image.Image:
        arr = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
        top = np.array(COLORS["space1"], dtype=float)
        bottom = np.array(COLORS["space0"], dtype=float)
        for y in range(OUT_H):
            u = y / max(OUT_H - 1, 1)
            arr[y, :, :3] = (top * (1-u) + bottom * u).astype(np.uint8)
            arr[y, :, 3] = 255
        im = Image.fromarray(arr, "RGBA")
        dr = ImageDraw.Draw(im)
        for x, y, r, a in self.stars:
            pulse = 0.55 + 0.45 * math.sin(t * 0.55 + x * 0.013 + y * 0.009) ** 2
            rr = r * SCALE
            dr.ellipse((x-rr, y-rr, x+rr, y+rr), fill=(211, 231, 247, int(a * pulse)))
        return im

    def draw_header(self, image: Image.Image, scene_name: str):
        draw_text(image, CONFIG["title"], (int(48*SCALE), int(64*SCALE)), int(34*SCALE), COLORS["white"]+(245,), True, True, stroke=max(1, int(2*SCALE)))
        label = {
            "opening": "A STAR AS A CLOCK",
            "lighthouse": "ROTATION → BEAM → PULSE",
            "pulse_train": "EACH PULSE IS A TICK",
            "arrival_times": "PREDICT → MEASURE → COMPARE",
            "hackrf": "HACKRF ONE // LAB SIMULATION",
            "residuals": "TIMING RESIDUALS",
            "finale": "COSMIC CLOCK",
        }.get(scene_name, "")
        draw_text(image, label, (int(50*SCALE), int(116*SCALE)), int(15*SCALE), COLORS["cyan"]+(220,), True, True, stroke=1)

    def draw_opening(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = smootherstep(shot_progress(t, shot))
        cx, cy = OUT_W * 0.50, OUT_H * 0.48
        glow = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        for r, a in [(180, 9), (120, 18), (70, 40)]:
            rr = r * SCALE * (0.7 + 0.3*p)
            gd.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=COLORS["violet"]+(a,))
        glow = glow.filter(ImageFilter.GaussianBlur(max(2, int(20*SCALE))))
        image.alpha_composite(glow)

        dr = ImageDraw.Draw(image)
        radius = 50 * SCALE
        dr.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(236,242,255,255), outline=COLORS["cyan"]+(255,), width=max(2,int(4*SCALE)))

        ang = -math.pi/2 + t * 3.8
        beam_len = 360 * SCALE
        spread = 0.14
        pts = [(cx,cy)]
        for a in (ang-spread, ang+spread):
            pts.append((cx+math.cos(a)*beam_len, cy+math.sin(a)*beam_len))
        beam = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
        bd = ImageDraw.Draw(beam)
        bd.polygon(pts, fill=COLORS["cyan"]+(45,))
        beam = beam.filter(ImageFilter.GaussianBlur(max(1,int(9*SCALE))))
        image.alpha_composite(beam)

        draw_text(image, "A PULSAR", (int(cx), int(cy+115*SCALE)), int(30*SCALE), COLORS["white"]+(245,), True, True, "ma")
        draw_text(image, "can tick across the galaxy", (int(cx), int(cy+160*SCALE)), int(17*SCALE), COLORS["muted"]+(220,), False, True, "ma")

    def draw_lighthouse(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = shot_progress(t, shot)
        dr = ImageDraw.Draw(image)
        cx, cy = OUT_W * 0.50, OUT_H * 0.45
        r = 76 * SCALE
        dr.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(20,31,57,255), outline=COLORS["violet"]+(230,), width=max(2,int(4*SCALE)))
        dr.arc((cx-r*1.5,cy-r*0.55,cx+r*1.5,cy+r*0.55),0,360,fill=COLORS["grid"]+(130,),width=max(1,int(2*SCALE)))
        theta = p * math.tau * 4.5
        for offset in (0, math.pi):
            a = theta + offset
            ex, ey = cx+math.cos(a)*470*SCALE, cy+math.sin(a)*470*SCALE
            beam = Image.new("RGBA", OUT_SIZE, (0,0,0,0))
            bd = ImageDraw.Draw(beam)
            bd.line((cx,cy,ex,ey), fill=COLORS["cyan"]+(80,), width=max(3,int(12*SCALE)))
            beam = beam.filter(ImageFilter.GaussianBlur(max(2,int(15*SCALE))))
            image.alpha_composite(beam)
        earth_x, earth_y = OUT_W*0.80, OUT_H*0.70
        er = 22*SCALE
        dr.ellipse((earth_x-er,earth_y-er,earth_x+er,earth_y+er),fill=(39,105,159,255),outline=COLORS["cyan"]+(200,),width=max(1,int(2*SCALE)))
        draw_text(image, "EARTH", (int(earth_x), int(earth_y+46*SCALE)), int(13*SCALE), COLORS["muted"]+(210,), True, True, "ma", 1)
        draw_text(image, "When the beam crosses us…", (int(54*SCALE), int(OUT_H*0.76)), int(25*SCALE), COLORS["white"]+(245,), True, True)
        draw_text(image, "we record a pulse.", (int(54*SCALE), int(OUT_H*0.81)), int(25*SCALE), COLORS["cyan"]+(245,), True, True)

    def draw_pulse_train(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = shot_progress(t, shot)
        x0, x1 = 55*SCALE, OUT_W-45*SCALE
        y0 = OUT_H*0.52
        dr = ImageDraw.Draw(image)
        dr.line((x0,y0,x1,y0),fill=COLORS["grid"]+(170,),width=max(1,int(2*SCALE)))
        visible = int(10 + 22*p)
        for i in range(visible):
            x = x0 + (i/max(visible-1,1))*(x1-x0)
            amp = 55 + 55*(0.5+0.5*math.sin(i*1.73))
            width = 6*SCALE
            points=[]
            for k in range(-12,13):
                xx=x+k*width/4
                yy=y0-amp*SCALE*math.exp(-0.5*(k/3.2)**2)
                points.append((xx,yy))
            dr.line(points,fill=COLORS["cyan"]+(230,),width=max(1,int(3*SCALE)))
        tick = int(p*1000)
        draw_text(image, f"TICK {tick:04d}", (int(54*SCALE), int(OUT_H*0.67)), int(38*SCALE), COLORS["white"]+(245,), True, True)
        draw_text(image, "period ≈ 125 ms  //  simulated", (int(56*SCALE), int(OUT_H*0.73)), int(16*SCALE), COLORS["muted"]+(220,), True, True)

    def draw_arrival_times(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = shot_progress(t, shot)
        dr = ImageDraw.Draw(image)
        x0, x1 = 80*SCALE, OUT_W-60*SCALE
        y0 = OUT_H*0.46
        span = x1-x0
        n = 16
        for i in range(n):
            x = x0 + i/(n-1)*span
            dr.line((x,y0-70*SCALE,x,y0+85*SCALE),fill=COLORS["grid"]+(80,),width=1)
            expected = y0
            resid = TICKS[i+35].residual_us
            observed = y0 + resid*1.6*SCALE
            dr.ellipse((x-4*SCALE,expected-4*SCALE,x+4*SCALE,expected+4*SCALE),fill=COLORS["muted"]+(180,))
            if i <= int(p*(n-1)):
                dr.line((x,expected,x,observed),fill=COLORS["amber"]+(180,),width=max(1,int(2*SCALE)))
                dr.ellipse((x-5*SCALE,observed-5*SCALE,x+5*SCALE,observed+5*SCALE),fill=COLORS["cyan"]+(245,))
        draw_text(image, "EXPECTED", (int(82*SCALE), int(y0-120*SCALE)), int(14*SCALE), COLORS["muted"]+(210,), True, True)
        draw_text(image, "OBSERVED", (int(82*SCALE), int(y0+140*SCALE)), int(14*SCALE), COLORS["cyan"]+(230,), True, True)
        resid = TICKS[min(len(TICKS)-1, 35+int(p*15))].residual_us
        draw_text(image, f"Δt  {resid:+.1f} μs", (int(54*SCALE), int(OUT_H*0.72)), int(38*SCALE), COLORS["amber"]+(245,), True, True)
        draw_text(image, "That difference is the clue.", (int(56*SCALE), int(OUT_H*0.78)), int(19*SCALE), COLORS["white"]+(230,), True, True)

    def draw_hackrf(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = shot_progress(t, shot)
        dr = ImageDraw.Draw(image)
        # Simple block-diagram instead of a literal branded hardware drawing.
        boxes = [
            ("SYNTHETIC\nPULSAR", OUT_W*0.16, COLORS["violet"]),
            ("HACKRF\nTX", OUT_W*0.39, COLORS["cyan"]),
            ("50 dB\nATTEN.", OUT_W*0.62, COLORS["amber"]),
            ("SDR\nRX", OUT_W*0.84, COLORS["blue"]),
        ]
        cy = OUT_H*0.47
        for idx,(label,cx,color) in enumerate(boxes):
            w,h=150*SCALE,120*SCALE
            dr.rounded_rectangle((cx-w/2,cy-h/2,cx+w/2,cy+h/2),radius=14*SCALE,fill=COLORS["dark"]+(230,),outline=color+(220,),width=max(1,int(3*SCALE)))
            for j,line in enumerate(label.split("\n")):
                draw_text(image,line,(int(cx),int(cy+(j-0.5)*28*SCALE)),int(15*SCALE),color+(240,),True,True,"mm",1)
            if idx < len(boxes)-1:
                nx=boxes[idx+1][1]
                dr.line((cx+w/2,cy,nx-w/2,cy),fill=COLORS["white"]+(120,),width=max(1,int(2*SCALE)))
                ax=lerp(cx+w/2,nx-w/2,0.74)
                dr.polygon([(ax,cy),(ax-10*SCALE,cy-7*SCALE),(ax-10*SCALE,cy+7*SCALE)],fill=COLORS["white"]+(150,))
        draw_text(image, "CLOSED BENCH TEST", (int(OUT_W/2), int(OUT_H*0.64)), int(30*SCALE), COLORS["white"]+(245,), True, True, "ma")
        draw_text(image, "COAX + ATTENUATION // NO TX ANTENNA", (int(OUT_W/2), int(OUT_H*0.69)), int(14*SCALE), COLORS["red"]+(240,), True, True, "ma", 1)

        # Waterfall-like pulses below.
        x0,x1=70*SCALE,OUT_W-70*SCALE
        y0,y1=OUT_H*0.75,OUT_H*0.89
        for row in range(28 if not QUICK_MODE else 12):
            yy=lerp(y0,y1,row/max(1,(27 if not QUICK_MODE else 11)))
            phase=(row*0.13+p*3.0)%1.0
            xx=lerp(x0,x1,phase)
            dr.line((x0,yy,x1,yy),fill=COLORS["grid"]+(32,),width=1)
            dr.line((xx-20*SCALE,yy,xx+20*SCALE,yy),fill=COLORS["cyan"]+(150,),width=max(1,int(3*SCALE)))

    def draw_residuals(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p = shot_progress(t, shot)
        dr = ImageDraw.Draw(image)
        x0,x1=70*SCALE,OUT_W-55*SCALE
        y0,y1=OUT_H*0.34,OUT_H*0.70
        dr.rectangle((x0,y0,x1,y1),outline=COLORS["grid"]+(140,),width=max(1,int(2*SCALE)))
        dr.line((x0,(y0+y1)/2,x1,(y0+y1)/2),fill=COLORS["grid"]+(120,),width=1)
        count=max(3,int(95*p))
        vals=TICKS[:count]
        pts=[]
        for i,tick in enumerate(vals):
            x=lerp(x0,x1,i/max(count-1,1))
            y=lerp((y0+y1)/2,y0,tick.residual_us/45.0) if tick.residual_us>=0 else lerp((y0+y1)/2,y1,-tick.residual_us/45.0)
            pts.append((x,y))
        if len(pts)>=2:
            dr.line(pts,fill=COLORS["amber"]+(235,),width=max(2,int(4*SCALE)))
        for x,y in pts[::max(1,len(pts)//18)]:
            dr.ellipse((x-3*SCALE,y-3*SCALE,x+3*SCALE,y+3*SCALE),fill=COLORS["cyan"]+(245,))
        draw_text(image, "+45 μs", (int(x0-10*SCALE),int(y0)), int(12*SCALE), COLORS["muted"]+(190,), True, True, "ra", 1)
        draw_text(image, "−45 μs", (int(x0-10*SCALE),int(y1)), int(12*SCALE), COLORS["muted"]+(190,), True, True, "ra", 1)
        draw_text(image, "TIMING RESIDUAL", (int(70*SCALE),int(OUT_H*0.76)), int(24*SCALE), COLORS["white"]+(240,), True, True)
        draw_text(image, "small offsets can encode real physics", (int(72*SCALE),int(OUT_H*0.81)), int(17*SCALE), COLORS["muted"]+(220,), False, True)

    def draw_finale(self, image: Image.Image, t: float, shot: Dict[str, Any]):
        p=smoothstep(shot_progress(t,shot))
        draw_text(image,"A STAR",(int(OUT_W/2),int(OUT_H*0.39)),int(55*SCALE),COLORS["white"]+(245,),True,True,"ma")
        draw_text(image,"BECOMES A CLOCK",(int(OUT_W/2),int(OUT_H*0.47)),int(43*SCALE),COLORS["cyan"]+(245,),True,True,"ma")
        draw_text(image,"ACROSS THE GALAXY",(int(OUT_W/2),int(OUT_H*0.54)),int(28*SCALE),COLORS["violet"]+(235,),True,True,"ma")
        rr=(44+14*math.sin(t*8)**2)*SCALE
        cx,cy=OUT_W/2,OUT_H*0.68
        dr=ImageDraw.Draw(image)
        dr.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=COLORS["amber"]+(int(220*p),),width=max(2,int(5*SCALE)))

    def draw_source_hud(self, image: Image.Image):
        text="PULSAR TIMING EXPLAINER // SYNTHETIC SIGNAL // HACKRF LAB DEMO"
        draw_text(image,text,(int(48*SCALE),OUT_H-int(42*SCALE)),int(10*SCALE),COLORS["muted"]+(180,),True,True,stroke=1)

    def draw_texture(self,image:Image.Image,t:float):
        ov=Image.new("RGBA",OUT_SIZE,(0,0,0,0))
        dr=ImageDraw.Draw(ov)
        for x,y,phase in self.dust:
            pulse=0.5+0.5*math.sin(t*0.9+phase)
            if pulse>0.72:
                xx=(x+t*3.0)%OUT_W
                yy=(y+math.sin(t*0.6+phase)*4*SCALE)%OUT_H
                dr.line((xx,yy,xx+9*SCALE,yy),fill=COLORS["cyan"]+(int(11*pulse),),width=1)
        offset=int((t*41)%9)
        for y in range(offset,OUT_H,9):
            dr.line((0,y,OUT_W,y),fill=(120,165,185,5),width=1)
        image.alpha_composite(ov)

    def render_frame(self,t:float)->np.ndarray:
        shot=get_shot(t)
        name=str(shot["name"])
        image=self.background(t)
        if name=="opening":
            self.draw_opening(image,t,shot)
        elif name=="lighthouse":
            self.draw_lighthouse(image,t,shot)
        elif name=="pulse_train":
            self.draw_pulse_train(image,t,shot)
        elif name=="arrival_times":
            self.draw_arrival_times(image,t,shot)
        elif name=="hackrf":
            self.draw_hackrf(image,t,shot)
        elif name=="residuals":
            self.draw_residuals(image,t,shot)
        else:
            self.draw_finale(image,t,shot)

        self.draw_header(image,name)
        self.draw_source_hud(image)
        self.draw_texture(image,t)

        arr=np.asarray(image.convert("RGB"),dtype=np.float32)
        arr*=VIGNETTE[...,None]
        arr=np.clip(arr,0,255).astype(np.uint8)
        img=ImageEnhance.Contrast(Image.fromarray(arr)).enhance(float(CONFIG["contrast"]))
        img=ImageEnhance.Color(img).enhance(float(CONFIG["saturation"]))
        arr=np.asarray(img,dtype=np.int16)
        rng=np.random.default_rng(int(t*1000)+884)
        grain=rng.normal(0.0,float(CONFIG["grain_strength"]),arr.shape[:2])[:,:,None]
        return np.clip(arr+grain,0,255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Audio / video
# -----------------------------------------------------------------------------

def gaussian_envelope(times: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5*((times-center)/max(width,1e-6))**2)


def generate_soundtrack(path: Path) -> Path:
    sr=int(CONFIG["sample_rate_audio"])
    dur=float(CONFIG["duration_s"])
    n=int(round(sr*dur))
    tt=np.arange(n,dtype=np.float64)/sr
    rng=np.random.default_rng(4104)

    drone=0.10*np.sin(2*np.pi*43*tt)+0.055*np.sin(2*np.pi*64.5*tt+0.7)
    shimmer=0.025*np.sin(2*np.pi*(180+8*np.sin(2*np.pi*0.04*tt))*tt)
    noise=rng.normal(0,1,n)
    kernel=np.ones(max(4,int(sr*0.015)))/max(4,int(sr*0.015))
    soft=np.convolve(noise,kernel,mode="same")*0.06

    ticks=np.zeros(n,dtype=np.float64)
    period=0.125 if not QUICK_MODE else 0.18
    for c in np.arange(0.45,dur,period):
        env=gaussian_envelope(tt,c,0.008)
        ticks += 0.12*env*np.sin(2*np.pi*820*tt)

    impacts=np.zeros(n,dtype=np.float64)
    for center in [s["start"] for s in SHOT_PLAN[1:]]:
        env=gaussian_envelope(tt,float(center)+0.12,0.16)
        impacts += 0.13*env*np.sin(2*np.pi*82*tt)

    mix=drone+shimmer+soft+ticks+impacts
    fade=np.minimum(1.0,np.minimum(tt/0.7,(dur-tt)/0.8))
    mix*=np.clip(fade,0,1)
    mix=np.tanh(mix*1.4)
    pcm=np.clip(mix*32767*0.86,-32768,32767).astype(np.int16)

    with wave.open(str(path),"wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    ffmpeg=shutil.which("ffmpeg")
    if not ffmpeg and imageio_ffmpeg is not None:
        try:
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg=None
    if not ffmpeg:
        return False
    cmd=[
        ffmpeg,"-y","-i",str(video_path),"-i",str(audio_path),
        "-c:v","copy","-c:a","aac","-b:a","160k","-shortest",str(output_path)
    ]
    try:
        subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return output_path.exists() and output_path.stat().st_size>1000
    except Exception:
        return False


def render_video(scene: CosmicClockScene) -> Path:
    raw=OUTPUT_ROOT/(CONFIG["output_basename"]+"_silent.mp4")
    final=OUTPUT_ROOT/(CONFIG["output_basename"]+".mp4")
    audio=OUTPUT_ROOT/(CONFIG["output_basename"]+"_ambient.wav")
    fps=int(CONFIG["fps"])
    frames=int(round(float(CONFIG["duration_s"])*fps))
    times=np.arange(frames,dtype=float)/fps

    with iio.get_writer(
        raw, fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=None
    ) as writer:
        for t in tqdm(times,desc="Rendering cosmic-clock short"):
            writer.append_data(scene.render_frame(float(t)))

    generate_soundtrack(audio)
    if mux_audio(raw,audio,final):
        return final
    shutil.copyfile(raw,final)
    return final


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    csv_path,json_path=save_timing_products()
    iq_path=generate_hackrf_iq()

    print("Title:",CONFIG["title"])
    print("Timing CSV:",csv_path.resolve())
    print("Summary:",json_path.resolve())
    print("HackRF IQ:",iq_path.resolve())
    print("IQ format: interleaved signed int8 I,Q (.cs8)")
    print("LAB ONLY: use coax + external attenuation; do not attach a TX antenna.")

    if IQ_ONLY:
        print("IQ-only mode requested; skipping video render.")
        return

    scene=CosmicClockScene()
    preview_times=[
        1.4,
        min(9.0,float(CONFIG["duration_s"])*0.20),
        min(20.0,float(CONFIG["duration_s"])*0.38),
        min(32.0,float(CONFIG["duration_s"])*0.57),
        min(43.0,float(CONFIG["duration_s"])*0.76),
        float(CONFIG["duration_s"])-0.7,
    ]
    for pt in tqdm(preview_times,desc="Preview frames"):
        Image.fromarray(scene.render_frame(float(pt))).save(PREVIEW_DIR/f"preview_{int(pt):02d}s.png")

    srt=write_srt(CAPTIONS,OUTPUT_ROOT/(CONFIG["output_basename"]+".srt"))
    final=render_video(scene)
    print("SRT:",srt.resolve())
    print("Final video:",final.resolve())
    print("Output directory:",OUTPUT_ROOT.resolve())


if __name__=="__main__":
    main()
