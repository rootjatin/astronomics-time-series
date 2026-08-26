from __future__ import annotations

"""
Result : https://youtube.com/shorts/BiiL20rgyM0

The Deepest Earthquakes Ever Recorded — cinematic YouTube Short renderer

Creates a vertical 1080x1920 science short about deep-focus earthquakes inside
subducting slabs. The visualizations are diagrammatic and not to scale.

Scientific framing used in the narration:
- USGS classifies shallow earthquakes as 0–70 km, intermediate as 70–300 km,
  and deep earthquakes as 300–700 km.
- The 24 May 2013 Sea of Okhotsk Mw 8.3 earthquake occurred at 609 km depth.
- The 30 May 2015 Bonin / Ogasawara deep earthquake was about 680 km deep.
- A 2021 study reported aftershocks near ~750 km, but a 2025 reanalysis
  challenged the deepest-event interpretation. The short labels this as debated.

Install:
    pip install numpy pillow imageio imageio-ffmpeg tqdm

Quick preview:
    DEEP_QUAKE_SHORT_QUICK=1 python the_deepest_earthquakes_ever_recorded.py

Full render:
    python the_deepest_earthquakes_ever_recorded.py

4K vertical:
    DEEP_QUAKE_SHORT_4K=1 python the_deepest_earthquakes_ever_recorded.py
"""

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm

QUICK_MODE = os.environ.get("DEEP_QUAKE_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("DEEP_QUAKE_SHORT_4K", "0") == "1" and not QUICK_MODE

OUT_W = 540 if QUICK_MODE else (2160 if FOUR_K else 1080)
OUT_H = 960 if QUICK_MODE else (3840 if FOUR_K else 1920)
FPS = 8 if QUICK_MODE else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK_MODE else 58.0
SCALE = OUT_W / 1080.0
OUT_SIZE = (OUT_W, OUT_H)

OUTPUT_ROOT = Path("deepest_earthquakes_ever_recorded_output")
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for d in (OUTPUT_ROOT, PREVIEW_DIR):
    d.mkdir(parents=True, exist_ok=True)

CONFIG: Dict[str, Any] = {
    "title": "THE DEEPEST EARTHQUAKES EVER RECORDED",
    "subtitle": "subducting slabs // mantle transition zone // deep focus",
    "output_basename": "the_deepest_earthquakes_ever_recorded",
    "contrast": 1.08,
    "saturation": 1.08,
    "vignette": 0.28,
}

COLORS = {
    "space": (2, 5, 12),
    "ocean": (18, 82, 145),
    "crust": (111, 85, 66),
    "mantle": (98, 42, 38),
    "mantle2": (55, 25, 35),
    "slab": (52, 92, 126),
    "slab_edge": (113, 205, 241),
    "white": (245, 249, 255),
    "muted": (170, 193, 211),
    "cyan": (84, 224, 255),
    "blue": (73, 132, 255),
    "gold": (255, 202, 98),
    "orange": (255, 139, 72),
    "red": (255, 83, 101),
    "violet": (177, 122, 255),
    "green": (107, 245, 177),
    "deep": (6, 11, 22),
}

FULL_CAPTIONS: List[Tuple[float, float, str]] = [
    (0.4, 7.5, "Most earthquakes happen close to the surface. But some begin hundreds of kilometers deep inside sinking tectonic plates."),
    (7.6, 17.0, "USGS classifies deep earthquakes as events roughly 300 to 700 kilometers below the surface."),
    (17.1, 27.2, "In 2013, a magnitude 8.3 earthquake beneath the Sea of Okhotsk ruptured about 609 kilometers deep."),
    (27.3, 38.0, "In 2015, a huge Bonin Islands earthquake struck near 680 kilometers depth — close to the base of the mantle transition zone."),
    (38.1, 48.0, "At these pressures and temperatures, ordinary brittle cracking should be difficult. Deep earthquakes may involve mineral transformations and unstable deformation inside cold slabs."),
    (48.1, 57.2, "Some researchers reported aftershocks near 750 kilometers after the 2015 event. A later reanalysis challenged that deepest-earthquake claim — so the record remains scientifically debated."),
]

if QUICK_MODE:
    f = DURATION / 58.0
    CAPTIONS = [(a*f, b*f, text) for a,b,text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 8.0 if not QUICK_MODE else 1.8},
    {"name": "depth_scale", "start": 8.0 if not QUICK_MODE else 1.8, "end": 18.0 if not QUICK_MODE else 4.0},
    {"name": "okhotsk", "start": 18.0 if not QUICK_MODE else 4.0, "end": 28.0 if not QUICK_MODE else 6.2},
    {"name": "bonin", "start": 28.0 if not QUICK_MODE else 6.2, "end": 39.0 if not QUICK_MODE else 8.5},
    {"name": "mechanism", "start": 39.0 if not QUICK_MODE else 8.5, "end": 49.0 if not QUICK_MODE else 10.7},
    {"name": "debate", "start": 49.0 if not QUICK_MODE else 10.7, "end": DURATION},
]


def clamp(x: float, lo: float=0.0, hi: float=1.0) -> float:
    return max(lo, min(hi, float(x)))


def smoothstep(x: float) -> float:
    x = clamp(x)
    return x*x*(3.0-2.0*x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b-a)*t


def get_shot(t: float) -> Dict[str, Any]:
    for s in SHOT_PLAN:
        if s["start"] <= t < s["end"]:
            return s
    return SHOT_PLAN[-1]


def caption_at(t: float) -> Optional[str]:
    for a,b,text in CAPTIONS:
        if a <= t < b:
            return text
    return None


def get_font(size: int, bold: bool=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, max(7, int(size*SCALE)))
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(image: Image.Image, value: str, xy: Tuple[int,int], size: int=28,
              fill=(255,255,255,255), bold: bool=False, anchor: str="la", stroke: int=2):
    ImageDraw.Draw(image).text(
        xy, value, font=get_font(size,bold), fill=fill, anchor=anchor,
        stroke_width=max(1,int(stroke*SCALE)), stroke_fill=(0,0,0,220)
    )


def draw_wrapped(image: Image.Image, value: str, x: int, y: int, max_width: int,
                 size: int=26, fill=(255,255,255,245), bold: bool=False):
    draw = ImageDraw.Draw(image)
    fnt = get_font(size,bold)
    words = value.split()
    lines: List[str] = []
    cur = ""
    for word in words:
        test = word if not cur else cur + " " + word
        box = draw.textbbox((0,0),test,font=fnt,stroke_width=max(1,int(2*SCALE)))
        if box[2]-box[0] <= max_width:
            cur = test
        else:
            if cur: lines.append(cur)
            cur = word
    if cur: lines.append(cur)
    for line in lines:
        draw.text((x,y),line,font=fnt,fill=fill,stroke_width=max(1,int(2*SCALE)),stroke_fill=(0,0,0,220))
        box = draw.textbbox((x,y),line,font=fnt,stroke_width=max(1,int(2*SCALE)))
        y += (box[3]-box[1]) + int(6*SCALE)


def make_vignette() -> np.ndarray:
    yy,xx = np.mgrid[0:OUT_H,0:OUT_W]
    nx=(xx-OUT_W/2)/(OUT_W/2); ny=(yy-OUT_H/2)/(OUT_H/2)
    r=np.sqrt(nx*nx+ny*ny)
    return np.clip(1.0-CONFIG["vignette"]*r**1.8,0.0,1.0).astype(np.float32)


VIGNETTE = make_vignette()


def srt_time(seconds: float) -> str:
    ms=int(round(seconds*1000)); h,ms=divmod(ms,3600000); m,ms=divmod(ms,60000); s,ms=divmod(ms,1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path) -> Path:
    lines=[]
    for i,(a,b,text) in enumerate(CAPTIONS,1):
        lines += [str(i),f"{srt_time(a)} --> {srt_time(b)}",text,""]
    path.write_text("\n".join(lines),encoding="utf-8")
    return path


@dataclass(frozen=True)
class Dust:
    x: float; y: float; r: float; a: int; phase: float


class DeepQuakeScene:
    def __init__(self):
        rng=np.random.default_rng(20260817)
        self.dust=[Dust(float(rng.uniform(0,OUT_W)),float(rng.uniform(0,OUT_H)),float(rng.uniform(.4,2.3)*SCALE),int(rng.uniform(10,55)),float(rng.uniform(0,2*math.pi))) for _ in range(180 if QUICK_MODE else 520)]
        self.points=[(float(rng.uniform(-1,1)),float(rng.uniform(-1,1)),float(rng.uniform(.2,1))) for _ in range(80)]

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int,int,int,int], alpha: int=172):
        layer=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        d.rounded_rectangle(box,radius=max(8,int(24*SCALE)),fill=(3,7,15,alpha),outline=COLORS["cyan"]+(48,),width=max(1,int(2*SCALE)))
        image.alpha_composite(layer)

    def background(self,t:float) -> Image.Image:
        arr=np.zeros((OUT_H,OUT_W,3),np.uint8)
        yy=np.linspace(0,1,OUT_H)[:,None]
        arr[...,0]=np.clip(4+yy*10,0,255)
        arr[...,1]=np.clip(7+yy*5,0,255)
        arr[...,2]=np.clip(18+yy*12,0,255)
        image=Image.fromarray(arr,"RGB").convert("RGBA")
        d=ImageDraw.Draw(image)
        for p in self.dust:
            a=int(p.a*(.68+.32*math.sin(t*1.3+p.phase)))
            d.ellipse((p.x-p.r,p.y-p.r,p.x+p.r,p.y+p.r),fill=COLORS["white"]+(a,))
        return image

    def earth_cutaway(self,image:Image.Image,t:float,zoom:float=1.0,quake_depth:Optional[float]=None,
                      pulse:float=0.0,label:Optional[str]=None, debated:bool=False):
        cx=int(OUT_W*.53); cy=int(OUT_H*.44); R=int(325*SCALE*zoom)
        layer=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        # atmosphere + globe body
        for extra,a in [(25,14),(13,32)]:
            rr=R+int(extra*SCALE); d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=COLORS["cyan"]+(a,))
        d.ellipse((cx-R,cy-R,cx+R,cy+R),fill=COLORS["mantle2"]+(255,),outline=COLORS["white"]+(70,),width=max(1,int(2*SCALE)))
        # mantle rings, exaggerated for clarity
        for frac,col in [(0.88,COLORS["mantle"]),(0.62,(82,34,42)),(0.36,(126,74,48)),(0.18,(235,165,82))]:
            rr=int(R*frac); d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=col+(255,))
        # upper ocean/crust cap
        d.pieslice((cx-R,cy-R,cx+R,cy+R),188,352,fill=COLORS["ocean"]+(255,))
        # continental arc
        pts=[]
        for i in range(40):
            u=i/39; ang=math.radians(195+150*u)
            rr=R*(.96+.025*math.sin(u*10+t*.1))
            pts.append((cx+rr*math.cos(ang),cy+rr*math.sin(ang)))
        d.line(pts,fill=COLORS["crust"]+(255,),width=max(3,int(14*SCALE)))
        # subducting slab — cold plate diving through mantle
        slab=[]
        for i in range(90):
            u=i/89
            x=cx-R*.75+u*R*1.05
            y=cy-R*.55+u*R*1.38 + math.sin(u*math.pi)*R*.06
            slab.append((x,y))
        d.line(slab,fill=COLORS["slab"]+(235,),width=max(8,int(30*SCALE)))
        d.line(slab,fill=COLORS["slab_edge"]+(180,),width=max(2,int(5*SCALE)))
        # transition-zone reference rings (410, 660 km as conceptual markers)
        for frac,textv in [(0.73,"410 km"),(0.48,"660 km")]:
            rr=int(R*frac)
            d.arc((cx-rr,cy-rr,cx+rr,cy+rr),195,345,fill=COLORS["gold"]+(75,),width=max(1,int(2*SCALE)))
            tx=int(cx+rr*.82); ty=int(cy+rr*.44)
            draw_text(layer,textv,(tx,ty),12 if not QUICK_MODE else 6,COLORS["gold"]+(175,),False,"la",1)
        image.alpha_composite(layer)

        if quake_depth is not None:
            # map 0..800 km along slab path for cinematic illustration
            u=clamp(quake_depth/800.0)
            idx=min(len(slab)-1,max(0,int(u*(len(slab)-1))))
            qx,qy=slab[idx]
            qlayer=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); qd=ImageDraw.Draw(qlayer)
            base=10*SCALE
            for k in range(5):
                rr=base + pulse*(18+28*k)*SCALE
                a=max(0,int(190-32*k-100*pulse))
                qd.ellipse((qx-rr,qy-rr,qx+rr,qy+rr),outline=(COLORS["violet"] if debated else COLORS["red"])+(a,),width=max(1,int((4-k*.45)*SCALE)))
            qd.ellipse((qx-7*SCALE,qy-7*SCALE,qx+7*SCALE,qy+7*SCALE),fill=(COLORS["violet"] if debated else COLORS["gold"])+(250,))
            if label:
                draw_text(qlayer,label,(int(qx+18*SCALE),int(qy-14*SCALE)),15 if not QUICK_MODE else 7,(COLORS["violet"] if debated else COLORS["white"])+(235,),True,"la",1)
            image.alpha_composite(qlayer.filter(ImageFilter.GaussianBlur(max(1,int(2*SCALE)))))
            image.alpha_composite(qlayer)

    def depth_ladder(self,image:Image.Image,t:float,highlight:float=680.0):
        x=int(OUT_W*.16); y0=int(OUT_H*.20); y1=int(OUT_H*.78); d=ImageDraw.Draw(image)
        d.line((x,y0,x,y1),fill=COLORS["white"]+(80,),width=max(2,int(3*SCALE)))
        ticks=[(0,"SURFACE"),(70,"70 km"),(300,"300 km"),(609,"609 km"),(680,"680 km"),(750,"~750?"),(800,"800 km")]
        for dep,label in ticks:
            yy=int(lerp(y0,y1,dep/800))
            col=COLORS["gold"] if abs(dep-highlight)<3 else (COLORS["violet"] if dep==750 else COLORS["muted"])
            d.line((x-int(11*SCALE),yy,x+int(11*SCALE),yy),fill=col+(190,),width=max(1,int(2*SCALE)))
            draw_text(image,label,(x+int(22*SCALE),yy),14 if not QUICK_MODE else 7,col+(220,),dep in (609,680,750),"lm",1)
        marker_y=int(lerp(y0,y1,clamp(highlight/800)))
        d.ellipse((x-int(10*SCALE),marker_y-int(10*SCALE),x+int(10*SCALE),marker_y+int(10*SCALE)),fill=COLORS["red"]+(245,))

    def seismic_rays(self,image:Image.Image,t:float,strength:float=1.0):
        layer=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        cx=int(OUT_W*.53); cy=int(OUT_H*.44)
        for k in range(8):
            phase=(t*.55+k/8)%1.0
            r=phase*OUT_W*.55
            a=int((1-phase)*95*strength)
            d.arc((cx-r,cy-r,cx+r,cy+r),200,338,fill=COLORS["cyan"]+(a,),width=max(1,int(3*SCALE)))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1,int(4*SCALE)))))
        image.alpha_composite(layer)

    def title(self,image:Image.Image,t:float):
        if t < (5.6 if not QUICK_MODE else 1.25):
            fade=smoothstep(t/(.8 if not QUICK_MODE else .18))
            draw_text(image,"THE DEEPEST",(OUT_W//2,int(OUT_H*.067)),35 if not QUICK_MODE else 17,COLORS["white"]+(int(245*fade),),True,"ma",2)
            draw_text(image,"EARTHQUAKES",(OUT_W//2,int(OUT_H*.107)),51 if not QUICK_MODE else 25,COLORS["red"]+(int(250*fade),),True,"ma",2)
            draw_text(image,"EVER RECORDED",(OUT_W//2,int(OUT_H*.151)),38 if not QUICK_MODE else 19,COLORS["gold"]+(int(245*fade),),True,"ma",2)

    def caption(self,image:Image.Image,t:float):
        cap=caption_at(t)
        if not cap: return
        y0=OUT_H-(252 if not QUICK_MODE else 126)
        self.panel(image,(44 if not QUICK_MODE else 22,y0,OUT_W-(44 if not QUICK_MODE else 22),y0+(132 if not QUICK_MODE else 68)),176)
        draw_wrapped(image,cap,68 if not QUICK_MODE else 34,y0+(28 if not QUICK_MODE else 14),OUT_W-(136 if not QUICK_MODE else 68),28 if not QUICK_MODE else 14)

    def label(self,image:Image.Image,textv:str):
        draw_text(image,textv,(52 if not QUICK_MODE else 26,58 if not QUICK_MODE else 29),18 if not QUICK_MODE else 9,COLORS["muted"]+(205,),True,"la",1)

    def source_hud(self,image:Image.Image):
        draw_text(image,"SCIENCE VISUALIZATION // NOT TO SCALE",(OUT_W-(48 if not QUICK_MODE else 24),72 if not QUICK_MODE else 36),14 if not QUICK_MODE else 7,COLORS["gold"]+(225,),True,"ra",1)
        draw_text(image,"DEPTHS SHOWN AS REFERENCE MARKERS",(OUT_W-(48 if not QUICK_MODE else 24),100 if not QUICK_MODE else 50),13 if not QUICK_MODE else 6,COLORS["muted"]+(190,),False,"ra",1)

    def intro(self,image:Image.Image,t:float):
        s=SHOT_PLAN[0]; local=smoothstep((t-s["start"])/max(s["end"]-s["start"],1e-9))
        self.earth_cutaway(image,t,.92,quake_depth=120+460*local,pulse=(t*.9)%1,label="DEEP FOCUS")
        self.seismic_rays(image,t,.7)
        self.panel(image,(int(OUT_W*.08),int(OUT_H*.69),int(OUT_W*.92),int(OUT_H*.82)),168)
        draw_text(image,"SOME EARTHQUAKES BEGIN HUNDREDS OF KILOMETERS DOWN",(OUT_W//2,int(OUT_H*.733)),21 if not QUICK_MODE else 10,COLORS["white"]+(245,),True,"ma",1)
        draw_text(image,"inside cold tectonic slabs sinking into the mantle",(OUT_W//2,int(OUT_H*.777)),16 if not QUICK_MODE else 8,COLORS["cyan"]+(225,),False,"ma",1)
        self.label(image,"1 // EARTHQUAKES BELOW THE CRUST")

    def depth_scale_scene(self,image:Image.Image,t:float):
        s=SHOT_PLAN[1]; local=smoothstep((t-s["start"])/max(s["end"]-s["start"],1e-9))
        self.earth_cutaway(image,t,.88,quake_depth=300+380*local,pulse=(t*.8)%1)
        self.depth_ladder(image,t,highlight=300+380*local)
        self.panel(image,(int(OUT_W*.38),int(OUT_H*.69),int(OUT_W*.93),int(OUT_H*.82)),170)
        draw_text(image,"DEEP: 300–700 km",(int(OUT_W*.655),int(OUT_H*.733)),25 if not QUICK_MODE else 12,COLORS["gold"]+(245,),True,"ma",1)
        draw_text(image,"USGS depth classification",(int(OUT_W*.655),int(OUT_H*.777)),15 if not QUICK_MODE else 7,COLORS["white"]+(220,),False,"ma",1)
        self.label(image,"2 // HOW DEEP IS 'DEEP'?")

    def okhotsk(self,image:Image.Image,t:float):
        self.earth_cutaway(image,t,.93,quake_depth=609,pulse=(t*.9)%1,label="609 km")
        self.depth_ladder(image,t,609)
        self.panel(image,(int(OUT_W*.37),int(OUT_H*.67),int(OUT_W*.93),int(OUT_H*.84)),182)
        draw_text(image,"SEA OF OKHOTSK • 2013",(int(OUT_W*.65),int(OUT_H*.714)),20 if not QUICK_MODE else 10,COLORS["muted"]+(230,),True,"ma",1)
        draw_text(image,"Mw 8.3",(int(OUT_W*.65),int(OUT_H*.760)),38 if not QUICK_MODE else 19,COLORS["red"]+(248,),True,"ma",1)
        draw_text(image,"DEPTH 609 km",(int(OUT_W*.65),int(OUT_H*.808)),20 if not QUICK_MODE else 10,COLORS["gold"]+(238,),True,"ma",1)
        self.label(image,"3 // A GIANT DEEP-FOCUS EARTHQUAKE")

    def bonin(self,image:Image.Image,t:float):
        self.earth_cutaway(image,t,.93,quake_depth=680,pulse=(t*.86)%1,label="~680 km")
        self.depth_ladder(image,t,680)
        self.panel(image,(int(OUT_W*.37),int(OUT_H*.67),int(OUT_W*.93),int(OUT_H*.84)),184)
        draw_text(image,"BONIN / OGASAWARA • 2015",(int(OUT_W*.65),int(OUT_H*.714)),19 if not QUICK_MODE else 9,COLORS["muted"]+(230,),True,"ma",1)
        draw_text(image,"NEAR 680 km",(int(OUT_W*.65),int(OUT_H*.762)),34 if not QUICK_MODE else 17,COLORS["red"]+(248,),True,"ma",1)
        draw_text(image,"near the base of the mantle transition zone",(int(OUT_W*.65),int(OUT_H*.809)),14 if not QUICK_MODE else 7,COLORS["white"]+(220,),False,"ma",1)
        self.label(image,"4 // NEAR THE 660-km BOUNDARY")

    def mechanism(self,image:Image.Image,t:float):
        s=SHOT_PLAN[4]; local=smoothstep((t-s["start"])/max(s["end"]-s["start"],1e-9))
        self.earth_cutaway(image,t,.93,quake_depth=620,pulse=(t*.95)%1)
        # add transformation spark field
        layer=Image.new("RGBA",OUT_SIZE,(0,0,0,0)); d=ImageDraw.Draw(layer)
        cx=int(OUT_W*.58); cy=int(OUT_H*.58)
        for i,(px,py,w) in enumerate(self.points[:45]):
            ang=t*.7+i*.5; x=cx+px*110*SCALE; y=cy+py*95*SCALE
            rr=(2+3*w)*SCALE*(.7+.3*math.sin(ang))
            col=COLORS["violet"] if i%2 else COLORS["cyan"]
            d.ellipse((x-rr,y-rr,x+rr,y+rr),fill=col+(int(70+120*local),))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1,int(3*SCALE)))))
        self.panel(image,(int(OUT_W*.08),int(OUT_H*.68),int(OUT_W*.92),int(OUT_H*.84)),188)
        draw_text(image,"WHY CAN ROCK 'BREAK' THIS DEEP?",(OUT_W//2,int(OUT_H*.716)),23 if not QUICK_MODE else 11,COLORS["white"]+(245,),True,"ma",1)
        draw_text(image,"mineral transformations + unstable deformation",(OUT_W//2,int(OUT_H*.764)),18 if not QUICK_MODE else 9,COLORS["violet"]+(235,),True,"ma",1)
        draw_text(image,"deep-earthquake physics is still an active research problem",(OUT_W//2,int(OUT_H*.809)),14 if not QUICK_MODE else 7,COLORS["muted"]+(220,),False,"ma",1)
        self.label(image,"5 // THE DEEP-EARTHQUAKE PUZZLE")

    def debate(self,image:Image.Image,t:float):
        s=SHOT_PLAN[5]; local=smoothstep((t-s["start"])/max(s["end"]-s["start"],1e-9))
        self.earth_cutaway(image,t,.93,quake_depth=750,pulse=(t*.8)%1,label="~750 km?",debated=True)
        self.depth_ladder(image,t,750)
        self.panel(image,(int(OUT_W*.35),int(OUT_H*.64),int(OUT_W*.94),int(OUT_H*.85)),196)
        draw_text(image,"THE ~750 km CLAIM",(int(OUT_W*.65),int(OUT_H*.689)),24 if not QUICK_MODE else 12,COLORS["violet"]+(245,),True,"ma",1)
        draw_text(image,"REPORTED IN 2021",(int(OUT_W*.65),int(OUT_H*.735)),17 if not QUICK_MODE else 8,COLORS["white"]+(225,),True,"ma",1)
        draw_text(image,"CHALLENGED BY A 2025 REANALYSIS",(int(OUT_W*.65),int(OUT_H*.776)),17 if not QUICK_MODE else 8,COLORS["gold"]+(235,),True,"ma",1)
        draw_text(image,"deepest-event record: scientifically debated",(int(OUT_W*.65),int(OUT_H*.818)),14 if not QUICK_MODE else 7,COLORS["muted"]+(220,),False,"ma",1)
        self.label(image,"6 // HOW DEEP IS THE RECORD?")

    def render_frame(self,t:float) -> np.ndarray:
        image=self.background(t); name=get_shot(t)["name"]
        if name=="intro": self.intro(image,t)
        elif name=="depth_scale": self.depth_scale_scene(image,t)
        elif name=="okhotsk": self.okhotsk(image,t)
        elif name=="bonin": self.bonin(image,t)
        elif name=="mechanism": self.mechanism(image,t)
        else: self.debate(image,t)
        self.source_hud(image); self.title(image,t); self.caption(image,t)
        arr=np.asarray(image.convert("RGB"))
        arr=np.asarray(ImageEnhance.Contrast(Image.fromarray(arr)).enhance(CONFIG["contrast"]))
        arr=np.asarray(ImageEnhance.Color(Image.fromarray(arr)).enhance(CONFIG["saturation"]))
        arr=np.clip(arr.astype(np.float32)*VIGNETTE[...,None],0,255).astype(np.uint8)
        fade_in=smoothstep(t/(.9 if not QUICK_MODE else .2))
        fade_out=1.0-smoothstep((t-(DURATION-(1.1 if not QUICK_MODE else .25)))/(1.0 if not QUICK_MODE else .2))
        return np.clip(arr.astype(np.float32)*fade_in*fade_out,0,255).astype(np.uint8)


def save_summary() -> Path:
    obj={
        "title":CONFIG["title"],"format":f"{OUT_W}x{OUT_H} vertical MP4","fps":FPS,
        "duration_s":DURATION,"quick_mode":QUICK_MODE,"four_k":FOUR_K,
        "facts":[
            "USGS: shallow 0–70 km, intermediate 70–300 km, deep 300–700 km.",
            "2013 Sea of Okhotsk Mw 8.3: 609 km depth (USGS).",
            "2015 Bonin/Ogasawara event: approximately 680 km deep.",
            "A 2021 study reported aftershocks near 750 km; a 2025 reanalysis challenged the deepest-event claim.",
        ],
        "visual_warning":"Earth cutaway, slab geometry, wave rings and relative layer sizes are diagrammatic and not to scale."
    }
    p=OUTPUT_ROOT/"science_and_render_summary.json"; p.write_text(json.dumps(obj,indent=2),encoding="utf-8"); return p


def render_video(scene:DeepQuakeScene) -> Path:
    write_srt(OUTPUT_ROOT/f"{CONFIG['output_basename']}.srt")
    raw=OUTPUT_ROOT/f"{CONFIG['output_basename']}_raw.mp4"; final=OUTPUT_ROOT/f"{CONFIG['output_basename']}_final.mp4"
    count=int(round(DURATION*FPS)); times=np.arange(count)/FPS
    with iio.get_writer(raw,fps=FPS,codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=None) as writer:
        for t in tqdm(times,desc="Rendering deepest-earthquakes short"):
            writer.append_data(scene.render_frame(float(t)))
    shutil.copyfile(raw,final)
    return final


def main():
    print("Preparing deepest-earthquakes YouTube Short ...")
    print("Mode:","QUICK" if QUICK_MODE else ("4K" if FOUR_K else "FULL"))
    print("Canvas:",f"{OUT_W}x{OUT_H}","FPS:",FPS,"Duration:",DURATION)
    scene=DeepQuakeScene(); summary=save_summary(); print("Summary:",summary.resolve())
    preview_times=[min(1.1,DURATION*.08),min(10,DURATION*.22),min(21,DURATION*.40),min(33,DURATION*.58),min(44,DURATION*.77),DURATION-(1 if not QUICK_MODE else .8)]
    for pt in tqdm(preview_times,desc="Preview frames"):
        Image.fromarray(scene.render_frame(float(pt))).save(PREVIEW_DIR/f"preview_{int(pt*10):03d}.png")
    final=render_video(scene); print("Final video:",final.resolve())


if __name__=="__main__":
    main()
