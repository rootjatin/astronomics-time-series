#!/usr/bin/env python3
"""Render a vertical science Short offline using its supplied narration mix.Output : https://youtube.com/shorts/_plKLruKCtc?feature=share
Sources
MIT — Joukowski airfoil and circulation — https://web.mit.edu/16.unified/www/SPRING/fluids/Lecture_Notes/F5LectureNotes.pdf
NASA Glenn — Angle of attack and lift — https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/incline.html
"""
import argparse
from functools import lru_cache
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 540, 960
TAU = 2 * math.pi
DURATION = 36
BEAT = 6
GRID_X, GRID_Y = np.meshgrid(np.linspace(0, 1, W, dtype=np.float32),
                            np.linspace(0, 1, H, dtype=np.float32))
VIGNETTE = np.clip(1 - .46 * ((GRID_X - .5)**2 + (GRID_Y - .45)**2), 0, 1)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def blend(a, b, u):
    return tuple(round(x + (y - x) * u) for x, y in zip(a, b))


def rgb(hexcode):
    return tuple(bytes.fromhex(hexcode.lstrip("#")))


def bgr(color):
    return color[::-1]


def point(xy):
    return tuple(int(round(float(value))) for value in xy)


def line(im, a, b, color, width=1):
    cv2.line(im, point(a), point(b), bgr(color), width, cv2.LINE_AA)


def path(im, points, color, width=1, close=False):
    cv2.polylines(im, [np.round(points).astype(np.int32)], close, bgr(color), width, cv2.LINE_AA)


def fill(im, points, color):
    cv2.fillPoly(im, [np.round(points).astype(np.int32)], bgr(color), cv2.LINE_AA)


def dot(im, xy, radius, color, width=-1):
    cv2.circle(im, point(xy), max(1, round(radius)), bgr(color), width, cv2.LINE_AA)


def arrow(im, a, b, color, width=2, head=.20):
    cv2.arrowedLine(im, point(a), point(b), bgr(color), width, cv2.LINE_AA, tipLength=head)


def ellipse(im, xy, radii, color, width=1, angle=0):
    cv2.ellipse(im, point(xy), point(radii), angle, 0, 360, bgr(color), width, cv2.LINE_AA)


def noise(seed, scales=(10, 28, 80), size=(W, H)):
    rng = np.random.default_rng(seed)
    result = np.zeros((size[1], size[0]), np.float32)
    for i, n in enumerate(scales):
        lattice = rng.random((max(3, round(n * size[1] / size[0])), n), dtype=np.float32)
        result += cv2.resize(lattice - .5, size, interpolation=cv2.INTER_CUBIC) * (.56**i)
    return result


def gradient(stops, seed=1, texture=3, spot=None):
    base = np.stack([np.interp(GRID_Y[:, 0], np.linspace(0, 1, len(stops)), c)
                     for c in np.asarray(stops).T[::-1]], axis=-1).astype(np.float32)
    picture = base[:, None, :] + texture * noise(seed)[..., None]
    if spot:
        x, y, rx, ry, color, strength = spot
        field = strength * np.exp(-((GRID_X - x) / rx)**2 - ((GRID_Y - y) / ry)**2)
        picture += field[..., None] * np.asarray(bgr(color))
    return np.uint8(np.clip(picture, 0, 255))


def grade(im, bloom=.50):
    small = cv2.resize(im, (135, 240), interpolation=cv2.INTER_AREA).astype(np.float32)
    lum = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    highlights = small * (np.clip((lum - 119) / 126, 0, 1)**2)[..., None]
    near = cv2.GaussianBlur(highlights, (0, 0), 2.8)
    far = cv2.GaussianBlur(highlights, (0, 0), 10.2)
    light = cv2.resize(.73*near+.27*far, (W, H), interpolation=cv2.INTER_LINEAR)
    linear = im.astype(np.float32) * VIGNETTE[...,None]
    shoulder = 1-smooth((linear-160)/115)*.48
    return np.uint8(np.clip(linear+bloom*light*shoulder,0,255))


def sprites(im, x, y, strength, color, blur=1.0, fine=.55):
    """Render subpixel particle density and a soft optical halo."""
    x = np.asarray(x)
    y = np.asarray(y)
    ix = np.rint(x / 2).astype(int)
    iy = np.rint(y / 2).astype(int)
    visible = (ix >= 0) & (ix < W // 2) & (iy >= 0) & (iy < H // 2)
    indices = iy[visible] * (W // 2) + ix[visible]
    weights = np.broadcast_to(strength, x.shape)[visible]
    hist = np.bincount(indices, weights=weights, minlength=W * H // 4).reshape(H // 2, W // 2)
    hist = np.minimum(hist, 3).astype(np.float32)
    soft = cv2.GaussianBlur(hist, (0, 0), blur) if blur else hist
    field = cv2.resize((1 - fine) * soft + fine * hist, (W, H))
    layer = np.uint8(np.clip(field[..., None] * np.asarray(bgr(color)), 0, 255))
    cv2.add(im, layer, dst=im)


def grid(im, spacing, color, top=235, bottom=760):
    for x in range(0, W, spacing):
        line(im, (x, top), (x, bottom), color)
    for y in range(top, bottom + 1, spacing):
        line(im, (0, y), (W, y), color)


@lru_cache(maxsize=160)
def font(size, family="sans", bold=False):
    names = {"sans": "DejaVuSans", "serif": "DejaVuSerif", "mono": "DejaVuSansMono",
             "display":"DejaVuSerifCondensed", "italic":"DejaVuSerifCondensed"}
    suffix = "-Italic.ttf" if family == "italic" else "-Bold.ttf" if bold else ".ttf"
    name = names[family]+suffix
    candidates = [name, str(Path("/usr/share/fonts/truetype/dejavu")/name)]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def wrap(draw, value, face, width):
    rows = [""]
    for word in value.split():
        candidate = " ".join(filter(None, (rows[-1], word)))
        if rows[-1] and draw.textlength(candidate, font=face) > width:
            rows.append(word)
        else:
            rows[-1] = candidate
    return rows


@lru_cache(maxsize=480)
def label_bitmap(value, size, color, family, bold):
    face = font(size, family, bold)
    left, top, right, bottom = face.getbbox(value)
    canvas = Image.new("RGBA", (max(1, right - left + 2), max(1, bottom - top + 2)))
    ImageDraw.Draw(canvas).text((1-left, 1-top), value, font=face, fill=color)
    return np.asarray(canvas)


def tag(im, xy, value, color, size=15, family="mono", align="left", bold=False):
    bitmap = label_bitmap(value, size, color, family, bold)
    x, y = point(xy)
    bw = bitmap.shape[1]
    if align == "right":
        x -= bw
    elif align == "center":
        x -= bw // 2
    composite(im, bitmap, x, y)


def composite(im, bitmap, x=0, y=0, opacity=1):
    l, r = max(0, x), min(W, x + bitmap.shape[1])
    t, b = max(0, y), min(H, y + bitmap.shape[0])
    if l >= r or t >= b:
        return
    sample = bitmap[t-y:b-y, l-x:r-x]
    alpha = sample[..., 3:4].astype(np.float32) * (opacity / 255)
    region = im[t:b, l:r]
    region[:] = np.uint8(region * (1 - alpha) + sample[..., 2::-1] * alpha + .5)


# Finishing operations work in the scene's 540 × 960 coordinate system.
XX = GRID_X * W
YY = GRID_Y * H


def lit(im, field, color):
    """Screen-composite a scalar light field in place."""
    field = np.clip(field, 0, 1).astype(np.float32)
    light = np.asarray(bgr(color), np.float32)
    im[:] = np.uint8(im + (255.0-im) * field[...,None] * (light/255) + .5)


def material(field, stops, colors):
    return np.uint8(np.stack([np.interp(field, stops, c) for c in np.asarray(colors).T[::-1]], axis=-1))


def camera(im,t):
    # Each pair of six-second beats has a dedicated framing. Fast eased moves
    # at the editorial joins leave the diagrams stable while they are read.
    index = min(2,int(t//12))
    local = (t-index*12)/12
    a, b = CAMERA[index], CAMERA[min(2,index+1)]
    change = float(smooth((local-.89)/.11))
    z, x, y, drift = (v+(w-v)*change for v,w in zip(a,b))
    z += drift*math.sin(math.pi*local)**2
    m=np.float32([[z,0,(1-z)*x], [0,z,(1-z)*y]])
    return cv2.warpAffine(im,m,(W,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT_101)


def editorial(im,t,title,captions,scale):
    """Render typography at the delivery size over the graded scene."""
    width, height = im.shape[1], im.shape[0]
    accent = rgb(VIDEO["accent"])
    ink = rgb(VIDEO["ink"])
    panel = rgb(VIDEO["panel"])
    # The preview carries the same caption layout as the final export.
    vf = t / BEAT
    cue = min(5,int(vf))
    appearance = min(1., .84+.16*float(smooth((vf%1)*9)))
    rgba_stamp(im,captions[cue],0,height-captions[cue].shape[0],appearance)
    if t < 5.8:
        opacity = (.91+.09*float(smooth(t/.35))) * (1-float(smooth((t-4.95)/.85)))
        rgba_stamp(im,title,0,0,opacity)
    else:
        rgba_stamp(im,section_bitmap(cue,width),0,0,1)
    bar_y=round(920*scale)
    start,end=round(35*scale),round(502*scale)
    line(im,(start,bar_y),(end,bar_y),blend(panel,ink,.25),max(1,round(scale)))
    line(im,(start,bar_y),(start+(end-start)*t/DURATION,bar_y),accent,max(1,round(scale*1.5)))


def rgba_stamp(im, bitmap, x, y, opacity):
    h,w=bitmap.shape[:2]
    sample=im[y:y+h,x:x+w]
    alpha=bitmap[...,3:4].astype(np.float32)*(opacity/255)
    sample[:] = np.uint8(sample*(1-alpha)+bitmap[...,2::-1]*alpha+.5)


@lru_cache(maxsize=24)
def section_bitmap(index,width):
    s=width/W
    dark,accent,ink=map(rgb,(VIDEO["panel"],VIDEO["accent"],VIDEO["ink"]))
    box=Image.new("RGBA",(width,round(116*s)))
    d=ImageDraw.Draw(box)
    for y in range(box.height):
        d.line((0,y,width,y),fill=dark+(round(224*(1-y/box.height)**1.3),))
    d.line((35*s,49*s,63*s,49*s),fill=accent,width=max(1,round(2*s)))
    d.text((74*s,44*s),VIDEO["eyebrow"],font=font(round(12*s),"mono"),fill=accent)
    d.text((34*s,70*s),CUES[index],font=font(round(22*s),"sans",True),fill=ink)
    return np.asarray(box)


def editorial_cards(width):
    s=width/W
    accent,paper,dark=map(rgb,(VIDEO["accent"],VIDEO["ink"],VIDEO["panel"]))
    family,align=VIDEO["type"]
    if family=="sans":
        family="display"
    head=Image.new("RGBA",(width,round(255*s)))
    d=ImageDraw.Draw(head)
    for yy in range(head.height):
        d.line((0,yy,width,yy),fill=dark+(round(230*(1-yy/head.height)**.93),))
    d.line((35*s,55*s,69*s,55*s),fill=accent,width=max(1,round(2*s)))
    d.text((81*s,48*s),VIDEO["eyebrow"],font=font(round(13*s),"mono"),fill=accent)
    left,right=34,502
    a={"left":"lt","center":"mt","right":"rt"}[align]
    x={"left":left,"center":(left+right)/2,"right":right}[align]
    words=VIDEO["headline"].split("\n")
    for logical_size in range(54,29,-1):
        regular=font(round(logical_size*s),family,family=="mono")
        accented=font(round(logical_size*s),"italic") if family!="mono" else regular
        faces=[accented if i==len(words)-1 else regular for i in range(len(words))]
        if all(d.textlength(word,font=face)<(right-left)*s for word,face in zip(words,faces)):
            break
    leading=logical_size*1.035
    for i,(word,face) in enumerate(zip(words,faces)):
        d.text((x*s,(88+i*leading)*s),word,anchor=a,font=face,
               fill=accent if i==len(words)-1 else paper)
    cards=[]
    for caption in VIDEO["beats"]:
        box=Image.new("RGBA",(width,round(187*s)))
        d=ImageDraw.Draw(box)
        for yy in range(box.height):
            a=yy/box.height
            d.line((0,yy,width,yy),fill=dark+(round(244*smooth(a/.72)),))
        d.line((35*s,43*s,35*s,114*s),fill=accent+(220,),width=max(1,round(2*s)))
        for size in (24,23,22,21):
            face=font(round(size*s),"sans",True)
            rows=wrap(d,caption,face,431*s)
            if len(rows)<=3:
                break
        for j,row in enumerate(rows):
            d.text((50*s,(79+(j-(len(rows)-1)/2)*28)*s),row,anchor="lm",font=face,fill=paper,
                   stroke_width=max(1,round(s*.48)),stroke_fill=dark)
        model_face=font(round(11*s),"mono")
        d.text((35*s,126*s),VIDEO["model"],font=model_face,fill=accent)
        cards.append(np.asarray(box))
    return np.asarray(head),cards


def finish_frame(state,t,width,height,title,captions,grain):
    canvas=camera(scene(state,t),t)
    im=grade(canvas,VIDEO.get("bloom",.50))
    if (width,height)!=(W,H):
        im=cv2.resize(im,(width,height),interpolation=cv2.INTER_AREA if width<W else cv2.INTER_CUBIC)
    # Low-amplitude luma dither counters visible gradients in darker footage.
    soft_noise=np.roll(grain,int(t)%9,axis=1)
    im=cv2.add(im,soft_noise,dtype=cv2.CV_8U)
    editorial(im,t,title,captions,width/W)
    return im


def timecode(seconds):
    ms = round(seconds * 1000)
    hour, ms = divmod(ms, 3600000)
    minute, ms = divmod(ms, 60000)
    second, ms = divmod(ms, 1000)
    return f"{hour:02}:{minute:02}:{second:02},{ms:03}"


VOICE_CREDIT = 'Narration generated with Sulabh Katiyar’s Indic Tiny TTS (Indian English), trained on the LIMMITS 2024 corpus (SYSPIN project, SPIRE Lab, Indian Institute of Science, Bangalore).\nModel: https://huggingface.co/sulabhkatiyar/indic-tiny-tts-english\nCorpus terms: https://ee.iisc.ac.in/limmitsdataset/SYSPIN-Dataset-Terms.pdf\nCC BY 4.0: https://creativecommons.org/licenses/by/4.0/\n'


def write_sidecars(output):
    slug = VIDEO["slug"]
    srt = "".join(f"{i+1}\n{timecode(i*BEAT)} --> {timecode((i+1)*BEAT)}\n{caption}\n\n"
                  for i, caption in enumerate(VIDEO["beats"]))
    sources = "\n".join(f"{label} — {url}" for label, url in VIDEO["sources"])
    chapters = "\n".join(f"{i*BEAT//60:02}:{i*BEAT%60:02} {caption}"
                         for i, caption in enumerate(VIDEO["beats"]))
    metadata = (f"TITLE\n{VIDEO['title']}\n\nDESCRIPTION\n{VIDEO['description']}\n\n"
                f"{VIDEO['method']}\n\n{VOICE_CREDIT}\nSources\n{sources}\n\n"
                f"{VIDEO['hashtags']}\n\n"
                f"NARRATION AND TIMING\n{chapters}\n\n"
                f"TAGS\n{', '.join(VIDEO['tags'])}\n")
    (output / f"{slug}.srt").write_text(srt, encoding="utf-8")
    (output / f"{slug}_youtube.txt").write_text(metadata, encoding="utf-8")


def score_only(duration, output):
    rate = 24000
    t = np.arange(round(duration * rate), dtype=np.float64) / rate
    seed = VIDEO["seed"]
    base, tempo, texture, cue = VIDEO["sound"]
    rng = np.random.default_rng(seed + 299)
    phase = TAU * tempo * t / 60
    audio = np.zeros_like(t)
    for i, ratio in enumerate((1, 1.5, 2, 2.997)):
        audio += (.070 / (1+i)) * np.sin(TAU * base * ratio * t + .65*np.sin(.43*t+i))
    pulse = np.exp(-5 * (phase % TAU) / TAU) * (.5 + .5 * np.sin(.23*t)**2)
    audio += .045 * pulse * np.sin(TAU * (base / 2) * t)
    hiss = np.convolve(rng.normal(size=t.size), np.ones(13) / 13, "same")
    audio += texture * hiss * (.42 + .58 * np.sin(.27*t + 1)**2)
    for onset in range(0, math.ceil(duration), BEAT):
        delta = np.maximum(0, t - onset)
        audio += .06 * np.exp(-3.2*delta) * np.sin(TAU * cue * np.log1p(4*delta) / 4) * (t >= onset)
    for onset in VIDEO.get("effects", []):
        delta = np.maximum(0, t - onset)
        audio += .26 * np.exp(-14*delta) * np.sin(TAU*66*delta) * (t >= onset)
    audio *= .56 * smooth(t / .14) * smooth((duration - t) / .35)
    data = (np.clip(audio,-1,1)*32767).astype("<i2")
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())


def narration_path():
    script_dir=Path(__file__).resolve().parent
    filename=VIDEO["slug"]+".m4a"
    for folder in (script_dir.parent/"audio", script_dir/"audio", script_dir):
        path=folder/filename
        if path.is_file():
            return path
    raise SystemExit(
        f"Missing {filename}. Keep the ZIP's audio folder next to scripts, "
        "or run with --no-voice for music only."
    )


def main():
    parser = argparse.ArgumentParser(description=VIDEO["title"])
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--width", type=int, default=1080, help="Vertical 9:16 video width. Default 1080.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preview-seconds", type=float, help="Render only the opening. Full sidecars are written.")
    parser.add_argument("--stills", action="store_true", help="Export three production stills alongside the video.")
    parser.add_argument("--no-voice", action="store_true", help="Use a synthetic score without narration.")
    parser.add_argument("--metadata-only", action="store_true", help="Write the full YouTube TXT and SRT without rendering.")
    args = parser.parse_args()
    if args.width < 180 or args.width % 2 or not 1 <= args.fps <= 120:
        parser.error("width must be even and at least 180, and fps must be 1–120")
    if args.preview_seconds is not None and (not math.isfinite(args.preview_seconds) or args.preview_seconds <= 0):
        parser.error("preview-seconds must be finite and greater than zero")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_sidecars(output)
    if args.metadata_only:
        print(f"Wrote SRT and YouTube TXT to {output}")
        return

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg and not args.no_voice:
        parser.error("FFmpeg is required for narrated output. Install it or use --no-voice.")
    baked_audio = None if args.no_voice else narration_path()
    cv2.setNumThreads(1)
    duration = min(args.preview_seconds or DURATION, DURATION)
    count = math.ceil(duration * args.fps)
    width, height = args.width, 2 * round(args.width * 8 / 9)
    slug = VIDEO["slug"] + ("_preview" if duration < DURATION else "")
    final = output / f"{slug}.mp4"
    state = setup()
    title, captions = editorial_cards(width)
    rng=np.random.default_rng(VIDEO["seed"]+208)
    luma_grain=np.int16(np.rint(rng.normal(0,.65,(height,width))))
    grain=np.repeat(luma_grain[...,None],3,axis=2)
    if args.stills:
        folder=output / (VIDEO["slug"]+"_stills")
        folder.mkdir(exist_ok=True)
        for stamp in (1,15,28):
            frame=finish_frame(state,stamp,width,height,title,captions,grain)
            Image.fromarray(frame[...,::-1]).save(folder / f"{stamp:02d}s.png")
    with tempfile.TemporaryDirectory(dir=output, prefix=".render-") as tmp:
        tmp = Path(tmp)
        silent = tmp / "video.mp4"
        soundtrack = baked_audio or tmp / "score.wav"
        rendered = tmp / "finished.mp4"
        if ffmpeg:
            if args.no_voice:
                score_only(count/args.fps, soundtrack)
            audio_encoding = ["-c:a", "aac", "-b:a", "128k"] if args.no_voice else ["-c:a", "copy"]
            command = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "bgr24",
                       "-video_size", f"{width}x{height}", "-framerate", str(args.fps), "-i", "-",
                       "-i", str(soundtrack), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
                       "-preset", "fast", "-threads", "2", "-crf", "16", "-pix_fmt", "yuv420p",
                       "-vf","scale=out_color_matrix=bt709","-colorspace","bt709",
                       "-color_trc","bt709","-color_primaries","bt709",
                       *audio_encoding, "-t", f"{count/args.fps:.9f}",
                       "-movflags", "+faststart", str(rendered)]
            writer = subprocess.Popen(command, stdin=subprocess.PIPE)
        else:
            writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("OpenCV could not open the MP4 writer. Install FFmpeg for H.264/AAC.")
        try:
            for index in range(count):
                t = index / args.fps
                im=finish_frame(state,t,width,height,title,captions,grain)
                if ffmpeg:
                    writer.stdin.write(im.tobytes())
                else:
                    writer.write(im)
                if index % (args.fps * BEAT) == 0:
                    print(f"{VIDEO['title']} | {index+1}/{count} frames", flush=True)
        finally:
            if ffmpeg:
                try:
                    writer.stdin.close()
                finally:
                    status = writer.wait()
            else:
                writer.release()
        if ffmpeg and status:
            raise RuntimeError(f"FFmpeg exited with status {status}")
        (rendered if ffmpeg else silent).replace(final)
    print(f"Created {final}")
    if ffmpeg:
        print("Audio: Indian English male narration and stereo mix" if baked_audio else "Audio: score only")
    else:
        print("Audio: silent MP4. Install FFmpeg for soundtrack and narration.")


VIDEO = {'slug': '34_fluid_flow_around_a_wing',
 'title': 'Fluid Flow Around a Wing',
 'headline': 'Fluid Flow\nAround a Wing',
 'eyebrow': 'AIRFOIL FLOW',
 'look': 'Glass wind tunnel with curved tracer streaks, a brushed-metal foil and local callouts',
 'type': ('sans', 'right'),
 'accent': '97dfc9',
 'ink': 'e1fbec',
 'panel': '071f28',
 'model': 'JOUKOWSKI AIRFOIL / POTENTIAL FLOW',
 'seed': 3403,
 'sound': (65.4, 82, 0.014, 356),
 'description': 'Follow tracer particles past the nose, along the curved surfaces and out at the '
                'trailing edge of a wing section.',
 'method': 'This two-dimensional, incompressible inviscid model maps flow around a circle to a Joukowski '
           'airfoil. Circulation satisfies the Kutta condition. Particle timing represents relative '
           'speed; the model omits viscosity, separation and three-dimensional wing-tip flow.',
 'tags': ['shorts',
          'physics',
          'python',
          'fluid dynamics',
          'airfoil',
          'streamlines',
          'potential flow',
          'wind tunnel'],
 'hashtags': '#Shorts #Physics #Airfoil',
 'sources': [('MIT — Joukowski airfoil and circulation',
              'https://web.mit.edu/16.unified/www/SPRING/fluids/Lecture_Notes/F5LectureNotes.pdf'),
             ('NASA Glenn — Angle of attack and lift',
              'https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/incline.html')],
 'beats': ['These moving dots trace the airflow around a slice through a wing.',
           'Near the nose, incoming flow divides and travels along both surfaces.',
           'Follow the dots above the wing. Their speed changes along the curved surface.',
           'The lines show the path each tracer follows in this steady flow.',
           'At the pointed trailing edge, the modeled flow leaves smoothly.',
           'Right next to a real wing, viscosity also creates a thin boundary layer.']}

CAMERA = ((1.025, 265, 476, 0.1), (1.04, 184, 477, 0.04), (1.035, 385, 488, 0.055))
CUES = ('Track the airflow', 'Flow splits at the nose', 'Speed over the wing', 'Read the streamlines', 'Trailing-edge departure', 'The boundary layer')


def setup():
    bg = gradient([rgb("061b2d"),rgb("123745"),rgb("06242e")], 43, 10,
                  (.52,.48,.45,.25,rgb("1b7c78"),.48))
    center = -.09 + .075j
    radius = abs(1-center)
    alpha = .13
    gamma = -2 * np.imag((1-center) * np.exp(-1j * alpha))
    # The complex potential is expressed in a circle-centered frame.
    # The chosen circulation cancels velocity at the mapped trailing-edge cusp.
    def mapping(q):
        z = q + center
        return (z + 1/z) * np.exp(-1j*alpha)

    def velocity(q):
        return np.conj(np.exp(-1j*alpha) - radius**2*np.exp(1j*alpha)/q**2 + 1j*gamma/q)

    angle = np.linspace(0, TAU, 305)
    foil = mapping(radius*np.exp(1j*angle))
    # March actual streamlines in the circle plane with normalized RK2 steps.
    traces = []
    for y in np.r_[np.linspace(-2.6,-.22,16), np.linspace(.06,2.6,19)]:
        q = complex(-4, y)
        trail = []
        for _ in range(540):
            trail.append(q)
            v = velocity(q)
            mid = q + .012 * v/max(abs(v), 1e-8)
            v = velocity(mid)
            q += .024*v/max(abs(v), 1e-8)
            if abs(q) < radius:
                q *= (radius+.0005)/abs(q)
            if q.real > 4.2:
                break
        q = np.asarray(trail)
        z = mapping(q)
        speed = np.abs(np.exp(-1j*alpha) - radius**2*np.exp(1j*alpha)/q**2 + 1j*gamma/q)
        speed /= np.maximum(np.abs(1-1/(q+center)**2),.06)
        xy = np.c_[270+81*z.real, 480-81*z.imag]
        length = np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1) / np.clip(speed[1:],.22,2.2))]
        traces.append((xy,length))
    foil_xy = np.c_[270+81*foil.real, 480-81*foil.imag]
    # Tunnel lighting follows the glass walls and the test-section center.
    lit(bg,.30*np.exp(-((YY-(333+.17*(XX-270)))/87)**2),rgb("2c8e92"))
    for j in range(5):
        y0=302+j*91
        line(bg,(0,y0+29),(W,y0-27),rgb("28515c"),1)
    return bg, foil_xy, traces

def scene(state,t):
    bg,foil,traces=state
    im=bg.copy()
    for j,(xy,travel) in enumerate(traces):
        path(im,xy,rgb("3a786f") if j%4 else rgb("609d89"))
        u=(np.arange(0,travel[-1]+59,59)+t*58+13*j)%travel[-1]
        head=np.c_[np.interp(u,travel,xy[:,0]),np.interp(u,travel,xy[:,1])]
        start=np.maximum(0,u-16)
        tail=np.c_[np.interp(start,travel,xy[:,0]),np.interp(start,travel,xy[:,1])]
        for tip,trail in zip(head,tail):
            line(im,trail,tip,rgb("92c1a9"))
            dot(im,tip,1.3,rgb("d7f9d8"))
    mask=np.zeros((H,W),np.uint8)
    cv2.fillPoly(mask,[np.round(foil).astype(np.int32)],255)
    m=mask.astype(np.float32)/255
    band=.5+.5*np.sin((YY-XX*.10)*.31)
    glint=np.exp(-((YY-470+XX*.085)/21)**2)
    sheen=material(np.clip(.23+.47*glint+.16*band,0,1),(0,.42,1),
                   (rgb("182b38"),rgb("385968"),rgb("d1efe8")))
    im[:]=np.uint8(im*(1-m[...,None])+sheen*m[...,None])
    path(im,foil,rgb("b8e2d7"),2,True)
    if 6<t<13:
        x,y=foil[np.argmin(foil[:,0])]
        line(im,(x,y),(x-27,y-54),rgb("d1eee8"))
        line(im,(x-27,y-54),(58,y-54),rgb("d1eee8"))
        dot(im,(x,y),3,rgb("d8f2e8"))
        tag(im,(51,y-79),"LEADING EDGE",rgb("e0f7e6"),14)
    elif 19<t:
        x,y=foil[np.argmax(foil[:,0])]
        line(im,(x,y),(x-37,y-70),rgb("c9f5e4"))
        line(im,(x-37,y-70),(x-132,y-70),rgb("c9f5e4"))
        tag(im,(x-143,y-94),"TRAILING EDGE",rgb("c9f5e4"),14)
    arrow(im,(43,263),(119,263),rgb("8ccbc2"))
    tag(im,(43,237),"INCOMING FLOW",rgb("a4dec9"),14)
    return im



if __name__ == "__main__":
    main()

