
#!/usr/bin/env python3
"""Render a vertical science Short offline using its supplied narration mix."""

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


VIDEO = {'slug': '38_water_flowing_around_an_obstacle',
 'title': 'Water Flowing Around an Obstacle',
 'headline': 'Water Flowing\nAround an\nObstacle',
 'eyebrow': 'RIVER FLOW',
 'look': 'Riverbed caustics, drifting foam tracers and an irregular shaded stone',
 'type': ('serif', 'left'),
 'accent': 'e3dda4',
 'ink': 'f3f6d9',
 'panel': '193a3b',
 'model': 'CYLINDER FLOW / INVISCID MODEL',
 'seed': 3807,
 'sound': (61.7, 68, 0.049, 284),
 'description': 'Follow bright tracer particles as a current divides around a rock. Find the stagnation '
                'point and see the modeled flow speed up around the sides.',
 'method': 'The paths are integrated from steady, inviscid potential flow around an impermeable circular '
           'cylinder. Water shading and the stone are artistic. A real submerged obstacle can create a '
           'separated wake and a moving free surface.',
 'tags': ['shorts',
          'physics',
          'python',
          'fluid dynamics',
          'potential flow',
          'water flow',
          'streamlines',
          'stagnation point'],
 'hashtags': '#Shorts #Physics #PotentialFlow',
 'sources': [('Journal of Engineering Mathematics — Potential flow around a circular cylinder, section 2',
              'https://link.springer.com/article/10.1007/s10665-024-10386-8'),
             ('NASA Earth Observatory — Two Views of Von Kármán Vortices',
              'https://science.nasa.gov/earth/earth-observatory/two-views-of-von-karman-vortices-90734/')],
 'beats': ['Drop a rock into a current and the approaching water has to turn.',
           'The ideal flow slows at the point directly upstream of the rock.',
           "That spot is the stagnation point. The model's velocity goes to zero there.",
           'Water farther to either side curves around the rock and speeds up.',
           'In this ideal flow, the paths straighten again downstream.',
           'Real water has viscosity, so a rock can also leave an unsteady wake.']}

CAMERA = ((1.02, 262, 477, 0.07), (1.1, 185, 478, 0.025), (1.06, 286, 486, 0.028))
CUES = ('A rock interrupts the current', 'The flow slows upstream', 'The stagnation point', 'Water speeds up at the sides', 'Straight paths downstream', 'What viscosity changes')


def setup():
    bg=gradient([rgb("193f46"),rgb("447d75"),rgb("18484b")],505,30)
    x,y=np.meshgrid(np.arange(W),np.arange(H))
    radius = 74
    theta=np.arctan2(y-481,x-261)
    r=np.hypot(y-481,x-261)
    stone_height=radius*(1+.066*np.sin(3*theta)+.055*np.sin(5*theta+.4))
    s = noise(534,(9,23,89),size=(W,H))
    rgba = np.zeros((H,W,4),np.uint8)
    seams=np.sin(x*.066+y*.084+6*s)
    grains=noise(377,(94,173,283))
    values=68*s+23*(1-r/radius)+20*(261-x)/radius+13*(481-y)/radius
    values+=np.tanh(grains*9)*13-20*np.exp(-(seams/.16)**2)
    for k, base in enumerate((158,153,112)):
        rgba[...,k]=np.uint8(np.clip(base+values,0,255))
    rgba[...,3]=np.uint8(np.clip((stone_height-r)*135,0,255))
    # Uniform stream plus a doublet gives an impermeable circular boundary.
    paths=[]
    for y0 in np.r_[np.linspace(-205,-6.5,22),np.linspace(6.5,205,22)]:
        z = -300 + 1j*y0
        trace=[]
        elapsed=[0.]
        for step in range(510):
            trace.append((z.real+261,z.imag+481))
            v = np.conj(1-radius**2/z**2)
            dt=1.7/max(.28,abs(v))
            mid=z+.5*dt*v
            z+=dt*np.conj(1-radius**2/mid**2)
            if z.real > 306:
                break
            elapsed.append(elapsed[-1]+dt)
        paths.append((np.array(trace),np.array(elapsed[:len(trace)])))
    return bg,rgba,paths,noise(499,(8,35,95), (270,480))


def scene(state,t):
    bg,stone,tracks,tex=state
    im=bg.copy()
    water=np.roll(tex,round(t*13),axis=1)
    caustic=np.maximum(0,np.sin(17*water+5*np.roll(water,round(t*3),axis=0))- .64)
    field=cv2.resize(caustic,(W,H))
    light=np.exp(-((XX-198)/218)**2-((YY-389)/240)**2)
    lit(im,.72*field*(.58+.42*light),rgb("b7eff0"))
    wet=np.exp(-((np.hypot(XX-261,YY-481)-77)/13)**2)
    im[:]=np.uint8(im*(1-.30*wet[...,None]))
    lit(im,.36*wet*smooth((261-XX)/61),rgb("e8f5d0"))
    for j,(xy,elapsed) in enumerate(tracks):
        path(im,xy,rgb("78ad9b") if j%4 else rgb("a8c5ac"))
        u=(np.arange(0,elapsed[-1],63)+t*51+17*j)%elapsed[-1]
        x=np.interp(u,elapsed,xy[:,0])
        y=np.interp(u,elapsed,xy[:,1])
        back=np.maximum(0,u-10)
        trail=np.c_[np.interp(back,elapsed,xy[:,0]),np.interp(back,elapsed,xy[:,1])]
        for (px,py),from_xy in zip(zip(x,y),trail):
            line(im,from_xy,(px,py),rgb("9fcbae"))
            dot(im,(px,py),1.45,rgb("e5f1d0"))

    composite(im,stone)
    if t>5.4:
        arrow(im,(54,254),(176,254),rgb("e7f4d6"),2,.13)
        tag(im,(55,272),"UPSTREAM",rgb("e7f4d6"),16)
        dot(im,(187,481),5,rgb("fff1bb"))
        line(im,(187,481),(123,580),rgb("e8e6bf"),1)
        tag(im,(32,589),"STAGNATION POINT",rgb("f3efc9"),15)
        if 16<t<28:
            arrow(im,(251,375),(332,388),rgb("f3dfaa"),2)
            tag(im,(186,324),"FASTER AROUND THE SIDE",rgb("f4e6bd"),13)

