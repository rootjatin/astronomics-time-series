#!/usr/bin/env python3
Output : https://www.youtube.com/shorts/_lNKvic5s_A
# This script is standalone. It generates its own imagery, audio, captions and metadata.
# No images, fonts, soundtracks, ephemerides, network access, or helper .py files required.
# Install: python -m pip install numpy opencv-python pillow
# System recommendation: ffmpeg; espeak for optional offline voice-over.
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Physical parameters have sources in VIDEO['sources']; visual exaggerations are labeled.
G_AU = 4.0 * math.pi * math.pi                  # AU^3/(solar mass*year^2)
SUN_MASS_KG = 1.98847e30
JUPITER_MASS_KG = 1.89813e27
JUPITER_MASS_SOLAR = JUPITER_MASS_KG/SUN_MASS_KG
AU_KM = 149597870.7
C_KMS = 299792.458

VIDEO = {'slug': '18_time_slows_down_near_this_object', 'title': 'Time Slows Down Near This Object', 'eyebrow': 'SCHWARZSCHILD • TIME DILATION', 'hud_label': 'dTAU / dT = sqrt(1 - Rs / r)', 'method': 'Time-dilation example for stationary (hovering) clocks in Schwarzschild geometry, dτ/dt=sqrt(1-Rs/r). Speeds are relative to the Schwarzschild coordinate time of a distant observer; hovering arbitrarily close to the event horizon requires unbounded acceleration.', 'tags': ['space', 'black hole', 'relativity', 'science simulation', 'shorts'], 'color': [127, 65, 161], 'accent': [255, 177, 107], 'description': 'Cinematic, independently runnable Python visualization of Time Slows Down Near This Object. Scientific caveat: Time-dilation example for stationary (hovering) clocks in Schwarzschild geometry, dτ/dt=sqrt(1-Rs/r). Speeds are relative to the Schwarzschild coordinate time of a distant observer; hovering arbitrarily close to the event horizon requires unbounded acceleration.\n\nScientific sources (not image imports):\\n• NASA: time dilation near a black hole: https://science.nasa.gov/universe/what-happens-when-something-gets-too-close-to-a-black-hole/\n• NASA: anatomy of black holes: https://science.nasa.gov/universe/black-holes/anatomy/\\n\\nProgrammatically generated artwork and soundtrack. #Shorts #BlackHole #Relativity #Python', 'beats': [[0, 6, 'Gravity can change how fast two clocks tick relative to each other.'], [6, 12, 'Far away from a black hole, a clock ticks almost normally.'], [12, 18, 'Closer in, a stationary clock advances more slowly.'], [18, 24, 'At two Schwarzschild radii, its rate is about seventy-one percent.'], [24, 30, 'At eight Schwarzschild radii, it is about ninety-four percent.'], [30, 36, 'No observer feels their own local seconds stretch.'], [36, 42, 'The difference appears when the clocks are compared.']], 'sources': [('NASA: time dilation near a black hole', 'https://science.nasa.gov/universe/what-happens-when-something-gets-too-close-to-a-black-hole/'), ('NASA: anatomy of black holes', 'https://science.nasa.gov/universe/black-holes/anatomy/')], 'duration': 42.0, 'watermark': 'SCIENCE-BASED VISUAL  /  SEE SOURCES'}

SEED = 5689


def ease(x):
    v = np.clip(x, 0.0, 1.0)
    return v*v*(3 - 2*v)


def rgba(color):
    return tuple(int(x) for x in color)


def colmix(a, b, f):
    return tuple(int(a[i]*(1-f) + b[i]*f) for i in range(3))


def fmt_srt(sec):
    ms = max(0, int(round(sec*1000)))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02}:{m:02}:{s:02},{ms:03}'


def font_path(bold=False):
    paths = (['/usr/share/fonts/truetype/lato/Lato-Heavy.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'] if bold else
             ['/usr/share/fonts/truetype/lato/Lato-Regular.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'])
    for p in paths:
        if os.path.exists(p): return p
    return None


def get_font(size, bold=False):
    p = font_path(bold)
    return ImageFont.truetype(p, size) if p else ImageFont.load_default()


def text_wrap(draw, phrase, font, max_width):
    words, result, line = phrase.split(), [], ''
    for word in words:
        candidate = (line + ' ' + word).strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width or not line:
            line = candidate
        else:
            result.append(line)
            line = word
    if line: result.append(line)
    return result


def title_wrap(title, max_width, size):
    dummy = Image.new('RGB', (20,20))
    dr = ImageDraw.Draw(dummy)
    f = get_font(size, True)
    lines = text_wrap(dr, title.upper(), f, max_width)
    if len(lines) > 4:
        return title_wrap(title, max_width, int(size*.88))
    return lines, f


def synth_audio(path, duration, beats, voice=True, rate=22050):
    """Offline synthetic soundtrack + optional espeak narration; uses only temporary WAVs."""
    n = int(duration*rate)
    t = np.arange(n, dtype=np.float64)/rate
    rng = np.random.default_rng(SEED+5)
    # Slowly modulated cinematic pad, sub-bass heartbeat, rising high shimmers.
    pad = (0.23*np.sin(2*np.pi*55*t + .9*np.sin(2*np.pi*.09*t)) +
           0.14*np.sin(2*np.pi*82.41*t + .3*np.sin(2*np.pi*.07*t)) +
           0.09*np.sin(2*np.pi*110*t))
    lfo = .55 + .25*np.cos(2*np.pi*.13*t)
    audio = .29*pad*lfo
    for j, (start, stop, _) in enumerate(beats):
        impact = np.maximum(0,t-start)
        env = np.exp(-impact*5.0)*(t >= start)
        audio += .065*env*np.sin(2*np.pi*(48+14*impact)*impact)
    noise = rng.normal(0, 1, n)
    noise = np.convolve(noise, np.ones(21)/21, mode='same')
    audio += (.008 + .016*(t/duration))*noise
    narration_state = 'Synthetic cinematic score (no narration requested or espeak unavailable)'
    if voice and shutil.which('espeak'):
        narration_state = 'Offline espeak voice-over and synthetic score'
        with tempfile.TemporaryDirectory() as td:
            for j, (start, stop, sentence) in enumerate(beats):
                wav_file = os.path.join(td, f'v{j}.wav')
                command = ['espeak','-v','en-us','-s','205','-p','37','-a','145','-w',wav_file,sentence]
                subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                if not os.path.isfile(wav_file): continue
                try:
                    with wave.open(wav_file, 'rb') as f:
                        raw = f.readframes(f.getnframes())
                        voice_rate = f.getframerate()
                        channels = f.getnchannels()
                        samples = np.frombuffer(raw,dtype='<i2').astype(np.float64)/32768
                        if channels>1: samples=samples.reshape(-1,channels).mean(axis=1)
                except Exception: continue
                ix=np.arange(int(len(samples)*rate/voice_rate),dtype=float)*voice_rate/rate
                samples = np.interp(ix,np.arange(len(samples)),samples)
                samples = samples[:max(1,int((stop-start-.12)*rate))]
                st=int((start+.14)*rate)
                en=min(n,st+len(samples))
                if en>st: audio[st:en] += .43*samples[:en-st]
    # Taper edges; audio gain normalized to prevent clipping.
    fadein=np.minimum(1,t/.55)
    fadeout=np.minimum(1,(duration-t)/.85)
    audio *= np.maximum(0,fadein*fadeout)
    peak=max(np.max(np.abs(audio)), .5)
    data=(np.clip(audio/peak*.86,-1,1)*32767).astype('<i2')
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate); f.writeframes(data.tobytes())
    return narration_state


class Renderer:
    def __init__(self, w, h):
        self.w, self.h=w,h
        self.rng=np.random.default_rng(SEED)
        self.color=tuple(VIDEO['color']) # BGR
        self.accent=tuple(VIDEO['accent']) # BGR
        self.font_small=get_font(max(15,int(w*.026)),True)
        self.font_caption=get_font(max(25,int(w*.047)),True)
        self.font_label=get_font(max(18,int(w*.032)),True)
        self.stars=self.rng.random((1250,3))
        self.disks={}
        self.belt_a=self.rng.uniform(2.1,3.4,3600)
        self.belt_phi=self.rng.uniform(0,2*np.pi,3600)
        self.belt_e=self.rng.uniform(0,.11,3600)
        self.belt_brightness=self.rng.uniform(.25,1,3600)
        self.cloud_phi=self.rng.uniform(0,2*np.pi,11500)
        self.cloud_r=self.rng.power(.8,11500)
        self._background=self.make_background()
        self._jupiter_track=None
        self._jupiter_years=None
        cinematic_setup(self)

    def make_background(self):
        h,w=self.h,self.w
        rng=np.random.default_rng(SEED+99)
        yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
        x=xx/w; y=yy/h
        n=(rng.random((max(9,h//28),max(9,w//28)))*255).astype(np.uint8)
        neb=cv2.resize(n,(w,h),interpolation=cv2.INTER_CUBIC)
        neb=cv2.GaussianBlur(neb,(0,0),w*.14).astype(np.float32)/255
        mist=np.exp(-((x-.42)**2/.17+(y-.50)**2/.11))
        haze=np.exp(-((x-.8)**2/.08+(y-.25)**2/.12))
        canvas=np.empty((h,w,3),dtype=np.uint8)
        for c in range(3):
            base=4.0 + 2.2*y + 12*neb + (20*(self.color[c]/255))*mist + (10*(self.accent[c]/255))*haze
            canvas[:,:,c]=np.clip(base,0,255)
        for sx,sy,st in self.stars:
            px=int(sx*w); py=int(sy*h)
            if st>.966:
                cv2.circle(canvas,(px,py),1,(175,199,225),-1,cv2.LINE_AA)
            elif st>.76:
                canvas[py,px]=(38,48,66)
        return canvas

    def start(self, t):
        out=self._background.copy()
        # a faint drifting grid adds cinematic perspective and scale
        glow=int(9+6*math.sin(t*.45))
        for j in range(8):
            y=int(self.h*(.18+j*.105)+5*math.sin(t*.2+j))
            cv2.line(out,(0,y),(self.w,y),(glow//4,glow//3,glow),1,cv2.LINE_AA)
        return out

    def pointfield(self, img, x, y, color=None, intensity=1., blur=2.0):
        """Binned luminous point sprites. ~100,000 points still render rapidly."""
        color=np.asarray(color if color is not None else self.color,float)
        ratio=3 if max(self.w,self.h)>800 else 2
        lw=max(2,self.w//ratio); lh=max(2,self.h//ratio)
        xs=(np.asarray(x)*lw/self.w).astype(np.int32)
        ys=(np.asarray(y)*lh/self.h).astype(np.int32)
        sel=(xs>=0)&(xs<lw)&(ys>=0)&(ys<lh)
        if not np.any(sel):return
        ids=ys[sel]*lw+xs[sel]
        weight=np.minimum(3.,np.asarray(intensity) if np.ndim(intensity) else float(intensity))
        if np.ndim(weight):
            v=weight[sel]
        else: v=np.full(ids.shape,weight)
        hist=np.bincount(ids,weights=v,minlength=lw*lh).reshape(lh,lw).astype(np.float32)
        hist=np.minimum(255,hist*100)
        if blur>0: hist=cv2.GaussianBlur(hist,(0,0),blur)
        layer=cv2.resize(hist,(self.w,self.h),interpolation=cv2.INTER_LINEAR)
        layer=np.clip(layer,0,210).astype(np.uint8)
        for k in range(3):
            img[:,:,k]=cv2.add(img[:,:,k],np.uint8(layer*(float(color[k])/255)))
        # Pinpoint stars stay sharp above the soft nebular particle glow.
        ix=np.flatnonzero(sel)
        if len(ix)>0:
            skip=max(1,len(ix)//650)
            picks=ix[::skip]
            px=np.asarray(x)[picks].astype(np.int32)
            py=np.asarray(y)[picks].astype(np.int32)
            good=(px>=0)&(px<self.w)&(py>=0)&(py<self.h)
            if np.any(good):
                px=px[good];py=py[good]
                white=np.uint8(np.clip(color*.58+np.array([108,108,108]),0,255))
                img[py,px]=np.maximum(img[py,px],white)

    def glow(self,img,x,y,r,color=None,power=1.0):
        x,y,r=int(x),int(y),max(1,int(r)); color=rgba(color if color is not None else self.accent)
        extent=max(14,int(r*4.2))
        x1=max(0,x-extent);y1=max(0,y-extent)
        x2=min(self.w,x+extent+1);y2=min(self.h,y+extent+1)
        if x1>=x2 or y1>=y2:return
        patch=img[y1:y2,x1:x2]
        layer=np.zeros_like(patch)
        cv2.circle(layer,(x-x1,y-y1),max(r*3,6),color,-1,cv2.LINE_AA)
        blurred=cv2.GaussianBlur(layer,(0,0),max(r*1.4,4))
        cv2.addWeighted(blurred,min(.8,power*.5),patch,1.0,0,patch)
        cv2.circle(img,(x,y),r,colmix(color,(255,255,255),.68),-1,cv2.LINE_AA)

    def orbit(self,img,cx,cy,r,flatten=.7,color=(45,52,75),width=1):
        cv2.ellipse(img,(int(cx),int(cy)),(max(1,int(r)),max(1,int(r*flatten))),
                    0,0,360,color,width,cv2.LINE_AA)

    def disk(self,name,n=15000,arms=3):
        if name not in self.disks:
            rng=np.random.default_rng(SEED+sum(ord(c) for c in name))
            r=np.sqrt(rng.random(n))*.96 + .025
            arm=rng.integers(0,arms,n)
            base_phi=(arm*2*np.pi/arms + 4.9*np.log(r+.19) + rng.normal(0,.31,n))
            # continuous disk field beneath arm contrast
            mask=rng.random(n)<.26
            base_phi[mask]=rng.uniform(0,2*np.pi,mask.sum())
            self.disks[name]=(r,base_phi,rng.uniform(.25,1,n))
        return self.disks[name]

    def draw_disk(self,img,name,t,cx,cy,radius=255,tilt=.62,n=18000,
                  color=None,arms=3,spin=1.,twist=0.,tidal=0.,opacity=1.):
        r,ph,lum=self.disk(name,n=n,arms=arms)
        angle=ph + (t*.22*spin)/(r+.18) + twist*r
        wave=tidal*np.sin(angle*2.0+2.4*t)*(r**2)
        rr=r+wave
        x=cx+radius*rr*np.cos(angle)
        y=cy+radius*tilt*rr*np.sin(angle)
        self.pointfield(img,x,y,color=color,intensity=opacity*lum,blur=1.2)
        return x,y

    def draw_halo(self,img,cx,cy,r,color=None,alpha=.2):
        c=np.array(color or self.color,float)
        h,w=self.h,self.w
        rr=int(max(2,r)); x=int(cx); y=int(cy)
        x1=max(0,x-rr);x2=min(w,x+rr+1);y1=max(0,y-rr);y2=min(h,y+rr+1)
        if x2<=x1 or y2<=y1:return
        xx=(np.arange(x1,x2,dtype=np.float32)-x)/rr
        yy=(np.arange(y1,y2,dtype=np.float32)-y)/rr
        field=np.exp(-((xx[None,:])**2+(yy[:,None])**2)*3.2)*alpha
        for k in range(3):
            patch=img[y1:y2,x1:x2,k]
            patch[:]=np.clip(patch.astype(np.float32)+field*c[k],0,255).astype(np.uint8)

    def circle_text(self,img,txt,x,y,color=None,scale=1):
        cv2.putText(img,txt,(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,
                    max(.35,self.w/720*.52*scale),color or (190,204,228),1,cv2.LINE_AA)

    def simulate_jupiter(self):
        """Planar Sun-Jupiter restricted three-body leapfrog, 600 asteroid tracers."""
        if self._jupiter_track is not None:return
        rng=np.random.default_rng(SEED+12)
        n=600
        a=rng.uniform(4.45,5.83,n)
        phase=rng.uniform(-.72,.72,n)
        x=a*np.cos(phase); y=a*np.sin(phase)
        vx=-2*np.pi*np.sin(phase)/np.sqrt(a)
        vy=2*np.pi*np.cos(phase)/np.sqrt(a)
        dt=.012
        steps=int(23/dt)
        store=[]
        def acc(qx,qy,jx,jy):
            r2=qx*qx+qy*qy+.0005
            d2=(qx-jx)**2+(qy-jy)**2+.0025
            return (-G_AU*qx/r2**1.5-G_AU*JUPITER_MASS_SOLAR*(qx-jx)/d2**1.5,
                    -G_AU*qy/r2**1.5-G_AU*JUPITER_MASS_SOLAR*(qy-jy)/d2**1.5)
        ax,ay=acc(x,y,5.204,0)
        for k in range(steps+1):
            if k%3==0:store.append(np.stack([x,y],axis=1).copy())
            vx += .5*dt*ax; vy += .5*dt*ay
            x += dt*vx; y += dt*vy
            period=(k+1)*dt/11.862
            jx=5.204*np.cos(2*np.pi*period); jy=5.204*np.sin(2*np.pi*period)
            ax,ay=acc(x,y,jx,jy)
            vx += .5*dt*ax; vy += .5*dt*ay
        self._jupiter_track=np.stack(store)
        self._jupiter_years=np.arange(len(store))*3*dt

    def label(self,frame,t,duration):
        h,w=self.h,self.w
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        d=ImageDraw.Draw(im,'RGBA')
        mag=w/720
        ink=(238,242,255,255)
        accent=(self.accent[2],self.accent[1],self.accent[0],255)
        d.rectangle((0,0,w,int(h*.115)), fill=(4,5,18,165))
        eyebrow=VIDEO['eyebrow'].upper()
        font_top=self.font_small; fs=max(10,int(w*.026))
        while d.textbbox((0,0),eyebrow,font=font_top)[2]>w-70*mag and fs>10:
            fs-=1; font_top=get_font(fs,True)
        while len(eyebrow)>3 and d.textbbox((0,0),eyebrow,font=font_top)[2]>w-70*mag:
            eyebrow=eyebrow[:-3]+'…'
        d.text((int(34*mag),int(h*.044)),eyebrow,font=font_top,
               fill=(164,185,211,250))
        d.line((int(33*mag),int(h*.092),int(33*mag+w*.75*(t/duration)),int(h*.092)),
               fill=accent,width=max(3,int(3*mag)))
        if t<4.1:
            fade=max(.0,1-ease((t-2.4)/1.7))
            d.rounded_rectangle((int(24*mag),int(h*.152),int(w-24*mag),int(h*.33)),
                                radius=int(16*mag),fill=(3,8,22,int(198*fade)))
            lines,f=title_wrap(VIDEO['title'],w-90*mag,int(w*.055))
            start_y=int(h*.18)
            for i,line in enumerate(lines):
                d.text((int(43*mag),start_y+int(i*w*.066)),line,font=f,
                       fill=(255,255,255,int(255*fade)),stroke_width=max(0,int(1*mag)),
                       stroke_fill=(6,9,22,int(255*fade)))
        # Header key fact, helps distinguish modeled from measured numbers.
        main_label=VIDEO['hud_label']
        d.rounded_rectangle((int(22*mag),int(h*.71),int(w-22*mag),int(h*.764)),
                            radius=int(11*mag),fill=(2,8,22,180),outline=(90,118,150,100),width=1)
        hfont=self.font_label
        hsize=max(12,int(w*.032))
        while d.textbbox((0,0),main_label,font=hfont)[2] > w-75*mag and hsize>11:
            hsize-=1
            hfont=get_font(hsize,True)
        d.text((int(38*mag),int(h*.724)),main_label,font=hfont,fill=accent)
        speech=''
        for st,en,line in VIDEO['beats']:
            if st<=t<en:
                speech=line; break
        if speech:
            ff=self.font_caption
            caption_lines=text_wrap(d,speech,ff,w-112*mag)
            fs=max(19,int(w*.047))
            while len(caption_lines)>3 and fs>max(16,int(w*.031)):
                fs=max(16,fs-2)
                ff=get_font(fs,True)
                caption_lines=text_wrap(d,speech,ff,w-112*mag)
            delta=max(int(fs*1.32),int(w*.044))
            cy=int(h*(.78 if w<500 else .81))
            boxh=(len(caption_lines)+1)*delta
            d.rounded_rectangle((int(24*mag),cy-int(16*mag),int(w-24*mag),cy+boxh),
                                radius=int(13*mag),fill=(2,5,16,215))
            for j,line in enumerate(caption_lines):
                bbox=d.textbbox((0,0),line,font=ff)
                xx=(w-(bbox[2]-bbox[0]))/2
                d.text((int(xx),cy+j*delta),line,font=ff,fill=ink,
                       stroke_width=max(1,int(1.2*mag)),stroke_fill=(5,8,17,255))
        footer=VIDEO['watermark']
        counter=f'{int(t):02d}s / {int(duration):02d}s'
        footerfont=self.font_small; fs=max(10,int(w*.026))
        maxfoot=w-240*mag
        while d.textbbox((0,0),footer,font=footerfont)[2]>maxfoot and fs>10:
            fs-=1;footerfont=get_font(fs,True)
        while len(footer)>3 and d.textbbox((0,0),footer,font=footerfont)[2]>maxfoot:
            footer=footer[:-3]+'…'
        d.text((int(34*mag),int(h*.949)),footer,font=footerfont,
               fill=(140,162,188,230))
        d.text((int(w-145*mag),int(h*.949)),counter,
               font=footerfont,fill=(140,162,188,230))
        return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)




# CINEMATIC RENDERING v2 -- all processing remains within this standalone file.
def cinematic_grade(R, frame, t, duration):
    """Scene-linear-inspired highlight bloom, soft flares, ACES-like contrast,
    subtle camera drift and film grain. Applied before typography."""
    w, h = R.w, R.h
    # Lens breathing and a nearly imperceptible camera track, preserving screen edges.
    s = 1.021 + .006*math.sin(.19*t)
    tx = 2.8*math.sin(.13*t + .3)
    ty = 4.0*math.cos(.11*t)
    M=np.array([[s,0,(1-s)*w*.5+tx], [0,s,(1-s)*h*.455+ty]],np.float32)
    frame[:] = cv2.warpAffine(frame,M,(w,h),flags=cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REFLECT_101)
    # Three complementary bloom scales, performed at quarter resolution.
    tiny=cv2.resize(frame,(max(2,w//4),max(2,h//4)),interpolation=cv2.INTER_AREA)
    luminance=(tiny[:,:,0].astype(np.float32)*.114 +
               tiny[:,:,1].astype(np.float32)*.587 +
               tiny[:,:,2].astype(np.float32)*.299)
    bright=np.clip((luminance-76)/105.,0.,1.)**1.5
    hot=(tiny.astype(np.float32)*bright[:,:,None])
    blur1=cv2.GaussianBlur(hot,(0,0),2.2)
    blur2=cv2.GaussianBlur(hot,(0,0),10.)
    bloom=cv2.resize(.28*blur1 + .15*blur2,(w,h),interpolation=cv2.INTER_LINEAR)
    # Narrow horizontal streaks from brightest areas; restrained.
    streak=cv2.GaussianBlur(hot,(0,0),sigmaX=max(4,w/36),sigmaY=.35)
    streak=cv2.resize(streak,(w,h),interpolation=cv2.INTER_LINEAR)
    arr=frame.astype(np.float32)
    arr += bloom + .14*streak
    # Gentle photographic tone curve: darker shadow floor, controlled highlights.
    arr=np.maximum(0,arr-2.2)
    x=arr*(1.15/255.0)
    arr=255.*np.clip((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0,1)
    # Lens falloff, stronger in the corners but subtle around central subject.
    arr *= R._cinema_vignette[:,:,None]
    rng=np.random.default_rng(SEED+int(t*97))
    # Tiny sensor dither (prevents gradient banding in video compression).
    grain=rng.normal(0,1.25,(max(1,h//2),max(1,w//2))).astype(np.float32)
    grain=cv2.resize(grain,(w,h),interpolation=cv2.INTER_LINEAR)
    arr += grain[:,:,None]
    frame[:]=np.uint8(np.clip(arr,0,255))
    return frame


def cinematic_setup(R):
    y,x=np.ogrid[0:R.h,0:R.w]
    rr=((x-R.w*.50)/(R.w*.78))**2+((y-R.h*.46)/(R.h*.75))**2
    R._cinema_vignette=np.clip(1.-.23*rr, .67,1.).astype(np.float32)


def cinematic_atmosphere(R,frame,t):
    """Depth-separated luminous mote layers and wispy traveling light shafts."""
    w,h=R.w,R.h
    if not hasattr(R,'_motefield'):
        rng=np.random.default_rng(SEED+888)
        R._motefield=(rng.uniform(0,w,90),rng.uniform(.12*h,.79*h,90),
                      rng.uniform(.17,1,90),rng.uniform(-1,1,90))
    x,y,dep,spd=R._motefield
    xx=(x + (9+25*dep)*np.sin(t*.1+spd*5))%w
    yy=(y + (1+6*dep)*np.sin(t*.12+spd*11))
    for j in range(len(x)):
        if not (.13*h<yy[j]<.77*h): continue
        rad=1 if dep[j]<.75 else 2
        alpha=.10+.26*dep[j]
        color=tuple(int(v*alpha) for v in ((130,155,190) if j%4 else (220,174,109)))
        cv2.circle(frame,(int(xx[j]),int(yy[j])),rad,color,-1,cv2.LINE_AA)
    return frame


# Shared visual primitives; bundled here so every script is fully self-contained.
def bh_core(R, img, x, y, rs, draw_ring=True, halo=True):
    """Volumetric-looking photon/shadow illustration; Schwarzschild b_crit ≈ 2.598 Rs.
    Circular ring on image plane is illustrative, not GR ray tracing."""
    shadow=2.598076211*rs
    if halo: R.draw_halo(img,x,y,shadow*2.45,(78,113,217),.20)
    cx=int(x);cy=int(y)
    if draw_ring:
        ext=int(max(10,shadow*2.5))
        x0=max(0,cx-ext);x1=min(R.w,cx+ext+1)
        y0=max(0,cy-ext);y1=min(R.h,cy+ext+1)
        if x1>x0 and y1>y0:
            gy,gx=np.ogrid[y0:y1,x0:x1]
            dx=(gx-x).astype(np.float32);dy=(gy-y).astype(np.float32)
            radius=np.sqrt(dx*dx+dy*dy)/(shadow+1e-6)
            phi=np.arctan2(dy,dx)
            # Opposite sides have unequal brightness (inspired by Doppler beaming).
            asym=np.clip(1.05+.68*np.cos(phi+.24),.30,1.85)
            thin=np.exp(-.5*((radius-1.052)/.026)**2)
            fringe=np.exp(-.5*((radius-1.115)/.055)**2)
            halo_ring=np.exp(-.5*((radius-1.15)/.19)**2)
            intensity=(172*thin+45*fringe+11*halo_ring)*asym
            layer=np.empty((y1-y0,x1-x0,3),np.float32)
            # BGR: cyan-white hot region; cooler burnt-amber on far side.
            layer[:,:,0]=intensity*(.60+.22*np.cos(phi))
            layer[:,:,1]=intensity*(.78+.14*np.cos(phi))
            layer[:,:,2]=intensity*(1.08-.10*np.cos(phi))
            dest=img[y0:y1,x0:x1,:]
            dest[:]=np.uint8(np.clip(dest.astype(np.float32)+layer,0,255))
    cv2.circle(img,(cx,cy),max(2,int(shadow*.973)),(0,1,4),-1,cv2.LINE_AA)
    # Keep a subtly illuminated silhouette edge to give depth.
    if draw_ring:
        cv2.circle(img,(cx,cy),max(2,int(shadow*.990)),(10,16,27),1,cv2.LINE_AA)
    return shadow

def disk_cloud(R,img,cx,cy,r0,t,n=6000,tilt=.24,brighten=True):
    # Kepler-like differential angular velocity; NOT GR ray tracing or MHD.
    if 'disk_cloud_v2' not in R.disks:
        g=np.random.default_rng(SEED+712)
        r=np.sqrt(g.random(n)*.88+.12)
        a=g.uniform(0,2*np.pi,n)
        intensity=g.uniform(.25,1.0,n)
        R.disks['disk_cloud_v2']=(r,a,intensity)
    r,a,lum=R.disks['disk_cloud_v2']
    th=a + .46*t/(.2+r)**1.5
    rr=r0*(.7+1.5*r)
    x=cx+rr*np.cos(th)
    y=cy+rr*tilt*np.sin(th)
    approaching=np.clip(.55+.45*np.cos(th),.1,1.)
    lum=lum*approaching if brighten else lum
    R.pointfield(img,x,y,(73,169,255),lum*.95,1.3)
    # Behind-the-hole far-side glow is lifted into a stylized lensed arch.
    theta=np.linspace(.08,np.pi-.08,180)
    for scale,col in ((1.,(75,159,240)),(1.24,(49,108,174))):
        ax=cx+scale*r0*1.58*np.cos(theta)
        ay=cy-r0*(.78+.19*scale)*np.sin(theta)
        pts=np.stack([ax,ay],axis=-1).astype(np.int32)
        cv2.polylines(img,[pts],False,col,max(1,int(R.w/380)),cv2.LINE_AA)


def clock_face(R,img,x,y,r,phase,color=(215,215,255),label=None):
    x=int(x);y=int(y);r=max(8,int(r))
    cv2.circle(img,(x,y),r,(42,73,118),max(1,int(R.w/450)),cv2.LINE_AA)
    cv2.circle(img,(x,y),int(r*.92),color,1,cv2.LINE_AA)
    for j in range(12):
        a=2*np.pi*j/12
        p=(x+int(.82*r*np.sin(a)),y-int(.82*r*np.cos(a)))
        q=(x+int(.91*r*np.sin(a)),y-int(.91*r*np.cos(a)))
        cv2.line(img,p,q,color,2 if j%3==0 else 1,cv2.LINE_AA)
    hand=(x+int(.66*r*np.sin(phase)),y-int(.66*r*np.cos(phase)))
    cv2.line(img,(x,y),hand,color,max(1,int(R.w/330)),cv2.LINE_AA)
    cv2.circle(img,(x,y),max(2,int(r*.055)),(235,244,255),-1,cv2.LINE_AA)
    if label:R.circle_text(img,label,x-r,y+r+24*R.w/720,(178,207,241),.75)


def wave_signal(t,tc=1.0):
    # Shape mimics a compact-binary chirp but is not a detector reconstruction.
    s=np.clip(np.asarray(t,dtype=float),0.0,tc)
    u=np.maximum(.04,1.0-s/tc)
    phase=46*(1-u**.625)
    amp=.12/np.sqrt(np.sqrt(u))
    taper=np.where(s<.84*tc,1.,np.exp(-(s-.84*tc)*21))
    return amp*taper*np.sin(phase)


def elastic_grid(R,img,cx,cy,depth=1.0,skew=0.0):
    # An embedding diagram for curved SPACE in one slice; not spacetime itself.
    for which in range(2):
        for idx in range(-8,9):
            points=[]
            for v in np.linspace(-1,1,100):
                u=(idx/9.0) if which==0 else v
                z=v if which==0 else (idx/9.0)
                d=np.sqrt(u*u+z*z+.035)
                bend=-depth/(1+6*d*d)
                xx=cx+R.w*.38*(u+.11*z)+skew*R.w*.01*z
                yy=cy+R.h*.18*z+R.h*.10*bend
                points.append((int(xx),int(yy)))
            c=(30,61,102) if idx%2 else (47,102,156)
            cv2.polylines(img,[np.asarray(points,np.int32)],False,c,1,cv2.LINE_AA)

def scene(R,img,t,dur):
    w,h=R.w,R.h;q=t/dur;cx=w*.50;cy=h*.43;rs=w*.045
    bh_core(R,img,cx,cy,rs)
    a=np.sqrt(1-1/2);b=np.sqrt(1-1/8)
    clock_face(R,img,w*.22,h*.635,w*.103,2*np.pi*(t*a/5.1),label='2 Rs / NEAR')
    clock_face(R,img,w*.77,h*.635,w*.103,2*np.pi*(t*b/5.1),label='8 Rs / FAR')
    cv2.line(img,(int(w*.23),int(h*.53)),(int(cx-w*.13),int(cy)),(54,119,184),1,cv2.LINE_AA)
    cv2.line(img,(int(w*.77),int(h*.53)),(int(cx+w*.13),int(cy)),(54,119,184),1,cv2.LINE_AA)
    R.circle_text(img,'STATIC CLOCKS / SCHWARZSCHILD',w*.12,h*.28,(199,225,250),.79)
    R.circle_text(img,'0.707x',w*.13,h*.735,(187,221,255),.91)
    R.circle_text(img,'0.935x',w*.67,h*.735,(187,221,255),.91)
    R.circle_text(img,'LOCAL CLOCK RATE / DISTANT TIME',w*.10,h*.34,(180,199,232),.73)
 

def generate(args):
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    name=VIDEO['slug']
    meta_path=output/(name+'_youtube.txt')
    srt_path=output/(name+'.srt')
    video_path=output/(name+'.mp4')
    duration=float(VIDEO['duration'])
    beats=[tuple(b) for b in VIDEO['beats']]
    with open(srt_path,'w',encoding='utf8') as f:
        for j,(st,en,line) in enumerate(beats,1):
            f.write(f'{j}\n{fmt_srt(st)} --> {fmt_srt(en)}\n{line}\n\n')
    with open(meta_path,'w',encoding='utf8') as f:
        f.write('TITLE\n'+VIDEO['title']+'\n\nDESCRIPTION\n'+VIDEO['description']+'\n\n')
        f.write('METHOD & SCIENTIFIC ACCURACY\n'+VIDEO['method']+'\n\n')
        f.write('SOURCES\n')
        for a,b in VIDEO['sources']: f.write(a+'\n'+b+'\n')
        f.write('\nNARRATION SCRIPT\n'+ '\n'.join(row[2] for row in beats)+'\n')
        f.write('\nTAGS\n'+', '.join(VIDEO['tags'])+'\n')
    width=args.width; height=int(round(width*16/9))
    if width%2:width+=1
    if height%2:height+=1
    fps=args.fps
    cut=min(float(args.preview_seconds),duration) if args.preview_seconds else duration
    renderer=Renderer(width,height)
    frame_count=int(math.ceil(cut*fps))
    with tempfile.TemporaryDirectory() as td:
        raw=str(Path(td)/'silent.mp4')
        wav=str(Path(td)/'soundtrack.wav')
        if not shutil.which('ffmpeg'):
            # OpenCV output works without ffmpeg executable; audio is unavailable.
            print('WARNING: ffmpeg executable unavailable; using silent video.')
        codec=cv2.VideoWriter_fourcc(*'mp4v')
        vid=cv2.VideoWriter(raw,codec,fps,(width,height))
        if not vid.isOpened(): raise RuntimeError('Could not open MP4 writer (OpenCV mp4v).')
        for i in range(frame_count):
            t=i/fps
            frame=renderer.start(t)
            cinematic_atmosphere(renderer,frame,t)
            scene(renderer,frame,t,duration)
            frame=cinematic_grade(renderer,frame,t,duration)
            frame=renderer.label(frame,t,duration)
            vid.write(frame)
            if i%max(1,round(fps*5))==0:
                print(f'  {i+1}/{frame_count} frames',flush=True)
        vid.release()
        if shutil.which('ffmpeg'):
            narration=synth_audio(wav,cut,beats,voice=not args.no_voice)
            call=['ffmpeg','-y','-loglevel','error','-i',raw,'-i',wav,
                  '-map','0:v:0','-map','1:a:0','-c:v','libx264',
                  '-preset','medium','-crf','17','-pix_fmt','yuv420p',
                  '-c:a','aac','-b:a','128k','-movflags','+faststart',
                  '-shortest',str(video_path)]
            r=subprocess.run(call,capture_output=True,text=True)
            if r.returncode:
                print('ffmpeg H264/AAC failed; keeping OpenCV MP4:',r.stderr[-500:])
                shutil.copyfile(raw,video_path)
        else:
            narration='No audio: ffmpeg executable not found'
            shutil.copyfile(raw,video_path)
    print('\nCreated:')
    for p in (video_path,srt_path,meta_path):print(' -',p)
    print('Audio:',narration)
    print('Scientific mode:',VIDEO['method'])


def cli():
    ap=argparse.ArgumentParser(description='Standalone cinematic space Shorts renderer')
    ap.add_argument('--output',default='.',help='Folder for generated MP4, SRT and YouTube TXT')
    ap.add_argument('--width',type=int,default=720,help='9:16 video width; 1080 for Full HD (default 720)')
    ap.add_argument('--fps',type=int,default=30)
    ap.add_argument('--preview-seconds',type=float,default=None,help='Fast rendering test; leaves full SRT & metadata')
    ap.add_argument('--no-voice',action='store_true',help='Generate score only, no optional espeak voice-over')
    args=ap.parse_args()
    if args.width<180 or args.fps<1:ap.error('width must be >= 180 and fps >= 1')

#!/usr/bin/env python3
# This script is standalone. It generates its own imagery, audio, captions and metadata.
# No images, fonts, soundtracks, ephemerides, network access, or helper .py files required.
# Install: python -m pip install numpy opencv-python pillow
# System recommendation: ffmpeg; espeak for optional offline voice-over.
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Physical parameters have sources in VIDEO['sources']; visual exaggerations are labeled.
G_AU = 4.0 * math.pi * math.pi                  # AU^3/(solar mass*year^2)
SUN_MASS_KG = 1.98847e30
JUPITER_MASS_KG = 1.89813e27
JUPITER_MASS_SOLAR = JUPITER_MASS_KG/SUN_MASS_KG
AU_KM = 149597870.7
C_KMS = 299792.458

VIDEO = {'slug': '18_time_slows_down_near_this_object', 'title': 'Time Slows Down Near This Object', 'eyebrow': 'SCHWARZSCHILD • TIME DILATION', 'hud_label': 'dTAU / dT = sqrt(1 - Rs / r)', 'method': 'Time-dilation example for stationary (hovering) clocks in Schwarzschild geometry, dτ/dt=sqrt(1-Rs/r). Speeds are relative to the Schwarzschild coordinate time of a distant observer; hovering arbitrarily close to the event horizon requires unbounded acceleration.', 'tags': ['space', 'black hole', 'relativity', 'science simulation', 'shorts'], 'color': [127, 65, 161], 'accent': [255, 177, 107], 'description': 'Cinematic, independently runnable Python visualization of Time Slows Down Near This Object. Scientific caveat: Time-dilation example for stationary (hovering) clocks in Schwarzschild geometry, dτ/dt=sqrt(1-Rs/r). Speeds are relative to the Schwarzschild coordinate time of a distant observer; hovering arbitrarily close to the event horizon requires unbounded acceleration.\n\nScientific sources (not image imports):\\n• NASA: time dilation near a black hole: https://science.nasa.gov/universe/what-happens-when-something-gets-too-close-to-a-black-hole/\n• NASA: anatomy of black holes: https://science.nasa.gov/universe/black-holes/anatomy/\\n\\nProgrammatically generated artwork and soundtrack. #Shorts #BlackHole #Relativity #Python', 'beats': [[0, 6, 'Gravity can change how fast two clocks tick relative to each other.'], [6, 12, 'Far away from a black hole, a clock ticks almost normally.'], [12, 18, 'Closer in, a stationary clock advances more slowly.'], [18, 24, 'At two Schwarzschild radii, its rate is about seventy-one percent.'], [24, 30, 'At eight Schwarzschild radii, it is about ninety-four percent.'], [30, 36, 'No observer feels their own local seconds stretch.'], [36, 42, 'The difference appears when the clocks are compared.']], 'sources': [('NASA: time dilation near a black hole', 'https://science.nasa.gov/universe/what-happens-when-something-gets-too-close-to-a-black-hole/'), ('NASA: anatomy of black holes', 'https://science.nasa.gov/universe/black-holes/anatomy/')], 'duration': 42.0, 'watermark': 'SCIENCE-BASED VISUAL  /  SEE SOURCES'}

SEED = 5689


def ease(x):
    v = np.clip(x, 0.0, 1.0)
    return v*v*(3 - 2*v)


def rgba(color):
    return tuple(int(x) for x in color)


def colmix(a, b, f):
    return tuple(int(a[i]*(1-f) + b[i]*f) for i in range(3))


def fmt_srt(sec):
    ms = max(0, int(round(sec*1000)))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02}:{m:02}:{s:02},{ms:03}'


def font_path(bold=False):
    paths = (['/usr/share/fonts/truetype/lato/Lato-Heavy.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'] if bold else
             ['/usr/share/fonts/truetype/lato/Lato-Regular.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'])
    for p in paths:
        if os.path.exists(p): return p
    return None


def get_font(size, bold=False):
    p = font_path(bold)
    return ImageFont.truetype(p, size) if p else ImageFont.load_default()


def text_wrap(draw, phrase, font, max_width):
    words, result, line = phrase.split(), [], ''
    for word in words:
        candidate = (line + ' ' + word).strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width or not line:
            line = candidate
        else:
            result.append(line)
            line = word
    if line: result.append(line)
    return result


def title_wrap(title, max_width, size):
    dummy = Image.new('RGB', (20,20))
    dr = ImageDraw.Draw(dummy)
    f = get_font(size, True)
    lines = text_wrap(dr, title.upper(), f, max_width)
    if len(lines) > 4:
        return title_wrap(title, max_width, int(size*.88))
    return lines, f


def synth_audio(path, duration, beats, voice=True, rate=22050):
    """Offline synthetic soundtrack + optional espeak narration; uses only temporary WAVs."""
    n = int(duration*rate)
    t = np.arange(n, dtype=np.float64)/rate
    rng = np.random.default_rng(SEED+5)
    # Slowly modulated cinematic pad, sub-bass heartbeat, rising high shimmers.
    pad = (0.23*np.sin(2*np.pi*55*t + .9*np.sin(2*np.pi*.09*t)) +
           0.14*np.sin(2*np.pi*82.41*t + .3*np.sin(2*np.pi*.07*t)) +
           0.09*np.sin(2*np.pi*110*t))
    lfo = .55 + .25*np.cos(2*np.pi*.13*t)
    audio = .29*pad*lfo
    for j, (start, stop, _) in enumerate(beats):
        impact = np.maximum(0,t-start)
        env = np.exp(-impact*5.0)*(t >= start)
        audio += .065*env*np.sin(2*np.pi*(48+14*impact)*impact)
    noise = rng.normal(0, 1, n)
    noise = np.convolve(noise, np.ones(21)/21, mode='same')
    audio += (.008 + .016*(t/duration))*noise
    narration_state = 'Synthetic cinematic score (no narration requested or espeak unavailable)'
    if voice and shutil.which('espeak'):
        narration_state = 'Offline espeak voice-over and synthetic score'
        with tempfile.TemporaryDirectory() as td:
            for j, (start, stop, sentence) in enumerate(beats):
                wav_file = os.path.join(td, f'v{j}.wav')
                command = ['espeak','-v','en-us','-s','205','-p','37','-a','145','-w',wav_file,sentence]
                subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                if not os.path.isfile(wav_file): continue
                try:
                    with wave.open(wav_file, 'rb') as f:
                        raw = f.readframes(f.getnframes())
                        voice_rate = f.getframerate()
                        channels = f.getnchannels()
                        samples = np.frombuffer(raw,dtype='<i2').astype(np.float64)/32768
                        if channels>1: samples=samples.reshape(-1,channels).mean(axis=1)
                except Exception: continue
                ix=np.arange(int(len(samples)*rate/voice_rate),dtype=float)*voice_rate/rate
                samples = np.interp(ix,np.arange(len(samples)),samples)
                samples = samples[:max(1,int((stop-start-.12)*rate))]
                st=int((start+.14)*rate)
                en=min(n,st+len(samples))
                if en>st: audio[st:en] += .43*samples[:en-st]
    # Taper edges; audio gain normalized to prevent clipping.
    fadein=np.minimum(1,t/.55)
    fadeout=np.minimum(1,(duration-t)/.85)
    audio *= np.maximum(0,fadein*fadeout)
    peak=max(np.max(np.abs(audio)), .5)
    data=(np.clip(audio/peak*.86,-1,1)*32767).astype('<i2')
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate); f.writeframes(data.tobytes())
    return narration_state


class Renderer:
    def __init__(self, w, h):
        self.w, self.h=w,h
        self.rng=np.random.default_rng(SEED)
        self.color=tuple(VIDEO['color']) # BGR
        self.accent=tuple(VIDEO['accent']) # BGR
        self.font_small=get_font(max(15,int(w*.026)),True)
        self.font_caption=get_font(max(25,int(w*.047)),True)
        self.font_label=get_font(max(18,int(w*.032)),True)
        self.stars=self.rng.random((1250,3))
        self.disks={}
        self.belt_a=self.rng.uniform(2.1,3.4,3600)
        self.belt_phi=self.rng.uniform(0,2*np.pi,3600)
        self.belt_e=self.rng.uniform(0,.11,3600)
        self.belt_brightness=self.rng.uniform(.25,1,3600)
        self.cloud_phi=self.rng.uniform(0,2*np.pi,11500)
        self.cloud_r=self.rng.power(.8,11500)
        self._background=self.make_background()
        self._jupiter_track=None
        self._jupiter_years=None
        cinematic_setup(self)

    def make_background(self):
        h,w=self.h,self.w
        rng=np.random.default_rng(SEED+99)
        yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
        x=xx/w; y=yy/h
        n=(rng.random((max(9,h//28),max(9,w//28)))*255).astype(np.uint8)
        neb=cv2.resize(n,(w,h),interpolation=cv2.INTER_CUBIC)
        neb=cv2.GaussianBlur(neb,(0,0),w*.14).astype(np.float32)/255
        mist=np.exp(-((x-.42)**2/.17+(y-.50)**2/.11))
        haze=np.exp(-((x-.8)**2/.08+(y-.25)**2/.12))
        canvas=np.empty((h,w,3),dtype=np.uint8)
        for c in range(3):
            base=4.0 + 2.2*y + 12*neb + (20*(self.color[c]/255))*mist + (10*(self.accent[c]/255))*haze
            canvas[:,:,c]=np.clip(base,0,255)
        for sx,sy,st in self.stars:
            px=int(sx*w); py=int(sy*h)
            if st>.966:
                cv2.circle(canvas,(px,py),1,(175,199,225),-1,cv2.LINE_AA)
            elif st>.76:
                canvas[py,px]=(38,48,66)
        return canvas

    def start(self, t):
        out=self._background.copy()
        # a faint drifting grid adds cinematic perspective and scale
        glow=int(9+6*math.sin(t*.45))
        for j in range(8):
            y=int(self.h*(.18+j*.105)+5*math.sin(t*.2+j))
            cv2.line(out,(0,y),(self.w,y),(glow//4,glow//3,glow),1,cv2.LINE_AA)
        return out

    def pointfield(self, img, x, y, color=None, intensity=1., blur=2.0):
        """Binned luminous point sprites. ~100,000 points still render rapidly."""
        color=np.asarray(color if color is not None else self.color,float)
        ratio=3 if max(self.w,self.h)>800 else 2
        lw=max(2,self.w//ratio); lh=max(2,self.h//ratio)
        xs=(np.asarray(x)*lw/self.w).astype(np.int32)
        ys=(np.asarray(y)*lh/self.h).astype(np.int32)
        sel=(xs>=0)&(xs<lw)&(ys>=0)&(ys<lh)
        if not np.any(sel):return
        ids=ys[sel]*lw+xs[sel]
        weight=np.minimum(3.,np.asarray(intensity) if np.ndim(intensity) else float(intensity))
        if np.ndim(weight):
            v=weight[sel]
        else: v=np.full(ids.shape,weight)
        hist=np.bincount(ids,weights=v,minlength=lw*lh).reshape(lh,lw).astype(np.float32)
        hist=np.minimum(255,hist*100)
        if blur>0: hist=cv2.GaussianBlur(hist,(0,0),blur)
        layer=cv2.resize(hist,(self.w,self.h),interpolation=cv2.INTER_LINEAR)
        layer=np.clip(layer,0,210).astype(np.uint8)
        for k in range(3):
            img[:,:,k]=cv2.add(img[:,:,k],np.uint8(layer*(float(color[k])/255)))
        # Pinpoint stars stay sharp above the soft nebular particle glow.
        ix=np.flatnonzero(sel)
        if len(ix)>0:
            skip=max(1,len(ix)//650)
            picks=ix[::skip]
            px=np.asarray(x)[picks].astype(np.int32)
            py=np.asarray(y)[picks].astype(np.int32)
            good=(px>=0)&(px<self.w)&(py>=0)&(py<self.h)
            if np.any(good):
                px=px[good];py=py[good]
                white=np.uint8(np.clip(color*.58+np.array([108,108,108]),0,255))
                img[py,px]=np.maximum(img[py,px],white)

    def glow(self,img,x,y,r,color=None,power=1.0):
        x,y,r=int(x),int(y),max(1,int(r)); color=rgba(color if color is not None else self.accent)
        extent=max(14,int(r*4.2))
        x1=max(0,x-extent);y1=max(0,y-extent)
        x2=min(self.w,x+extent+1);y2=min(self.h,y+extent+1)
        if x1>=x2 or y1>=y2:return
        patch=img[y1:y2,x1:x2]
        layer=np.zeros_like(patch)
        cv2.circle(layer,(x-x1,y-y1),max(r*3,6),color,-1,cv2.LINE_AA)
        blurred=cv2.GaussianBlur(layer,(0,0),max(r*1.4,4))
        cv2.addWeighted(blurred,min(.8,power*.5),patch,1.0,0,patch)
        cv2.circle(img,(x,y),r,colmix(color,(255,255,255),.68),-1,cv2.LINE_AA)

    def orbit(self,img,cx,cy,r,flatten=.7,color=(45,52,75),width=1):
        cv2.ellipse(img,(int(cx),int(cy)),(max(1,int(r)),max(1,int(r*flatten))),
                    0,0,360,color,width,cv2.LINE_AA)

    def disk(self,name,n=15000,arms=3):
        if name not in self.disks:
            rng=np.random.default_rng(SEED+sum(ord(c) for c in name))
            r=np.sqrt(rng.random(n))*.96 + .025
            arm=rng.integers(0,arms,n)
            base_phi=(arm*2*np.pi/arms + 4.9*np.log(r+.19) + rng.normal(0,.31,n))
            # continuous disk field beneath arm contrast
            mask=rng.random(n)<.26
            base_phi[mask]=rng.uniform(0,2*np.pi,mask.sum())
            self.disks[name]=(r,base_phi,rng.uniform(.25,1,n))
        return self.disks[name]

    def draw_disk(self,img,name,t,cx,cy,radius=255,tilt=.62,n=18000,
                  color=None,arms=3,spin=1.,twist=0.,tidal=0.,opacity=1.):
        r,ph,lum=self.disk(name,n=n,arms=arms)
        angle=ph + (t*.22*spin)/(r+.18) + twist*r
        wave=tidal*np.sin(angle*2.0+2.4*t)*(r**2)
        rr=r+wave
        x=cx+radius*rr*np.cos(angle)
        y=cy+radius*tilt*rr*np.sin(angle)
        self.pointfield(img,x,y,color=color,intensity=opacity*lum,blur=1.2)
        return x,y

    def draw_halo(self,img,cx,cy,r,color=None,alpha=.2):
        c=np.array(color or self.color,float)
        h,w=self.h,self.w
        rr=int(max(2,r)); x=int(cx); y=int(cy)
        x1=max(0,x-rr);x2=min(w,x+rr+1);y1=max(0,y-rr);y2=min(h,y+rr+1)
        if x2<=x1 or y2<=y1:return
        xx=(np.arange(x1,x2,dtype=np.float32)-x)/rr
        yy=(np.arange(y1,y2,dtype=np.float32)-y)/rr
        field=np.exp(-((xx[None,:])**2+(yy[:,None])**2)*3.2)*alpha
        for k in range(3):
            patch=img[y1:y2,x1:x2,k]
            patch[:]=np.clip(patch.astype(np.float32)+field*c[k],0,255).astype(np.uint8)

    def circle_text(self,img,txt,x,y,color=None,scale=1):
        cv2.putText(img,txt,(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,
                    max(.35,self.w/720*.52*scale),color or (190,204,228),1,cv2.LINE_AA)

    def simulate_jupiter(self):
        """Planar Sun-Jupiter restricted three-body leapfrog, 600 asteroid tracers."""
        if self._jupiter_track is not None:return
        rng=np.random.default_rng(SEED+12)
        n=600
        a=rng.uniform(4.45,5.83,n)
        phase=rng.uniform(-.72,.72,n)
        x=a*np.cos(phase); y=a*np.sin(phase)
        vx=-2*np.pi*np.sin(phase)/np.sqrt(a)
        vy=2*np.pi*np.cos(phase)/np.sqrt(a)
        dt=.012
        steps=int(23/dt)
        store=[]
        def acc(qx,qy,jx,jy):
            r2=qx*qx+qy*qy+.0005
            d2=(qx-jx)**2+(qy-jy)**2+.0025
            return (-G_AU*qx/r2**1.5-G_AU*JUPITER_MASS_SOLAR*(qx-jx)/d2**1.5,
                    -G_AU*qy/r2**1.5-G_AU*JUPITER_MASS_SOLAR*(qy-jy)/d2**1.5)
        ax,ay=acc(x,y,5.204,0)
        for k in range(steps+1):
            if k%3==0:store.append(np.stack([x,y],axis=1).copy())
            vx += .5*dt*ax; vy += .5*dt*ay
            x += dt*vx; y += dt*vy
            period=(k+1)*dt/11.862
            jx=5.204*np.cos(2*np.pi*period); jy=5.204*np.sin(2*np.pi*period)
            ax,ay=acc(x,y,jx,jy)
            vx += .5*dt*ax; vy += .5*dt*ay
        self._jupiter_track=np.stack(store)
        self._jupiter_years=np.arange(len(store))*3*dt

    def label(self,frame,t,duration):
        h,w=self.h,self.w
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        d=ImageDraw.Draw(im,'RGBA')
        mag=w/720
        ink=(238,242,255,255)
        accent=(self.accent[2],self.accent[1],self.accent[0],255)
        d.rectangle((0,0,w,int(h*.115)), fill=(4,5,18,165))
        eyebrow=VIDEO['eyebrow'].upper()
        font_top=self.font_small; fs=max(10,int(w*.026))
        while d.textbbox((0,0),eyebrow,font=font_top)[2]>w-70*mag and fs>10:
            fs-=1; font_top=get_font(fs,True)
        while len(eyebrow)>3 and d.textbbox((0,0),eyebrow,font=font_top)[2]>w-70*mag:
            eyebrow=eyebrow[:-3]+'…'
        d.text((int(34*mag),int(h*.044)),eyebrow,font=font_top,
               fill=(164,185,211,250))
        d.line((int(33*mag),int(h*.092),int(33*mag+w*.75*(t/duration)),int(h*.092)),
               fill=accent,width=max(3,int(3*mag)))
        if t<4.1:
            fade=max(.0,1-ease((t-2.4)/1.7))
            d.rounded_rectangle((int(24*mag),int(h*.152),int(w-24*mag),int(h*.33)),
                                radius=int(16*mag),fill=(3,8,22,int(198*fade)))
            lines,f=title_wrap(VIDEO['title'],w-90*mag,int(w*.055))
            start_y=int(h*.18)
            for i,line in enumerate(lines):
                d.text((int(43*mag),start_y+int(i*w*.066)),line,font=f,
                       fill=(255,255,255,int(255*fade)),stroke_width=max(0,int(1*mag)),
                       stroke_fill=(6,9,22,int(255*fade)))
        # Header key fact, helps distinguish modeled from measured numbers.
        main_label=VIDEO['hud_label']
        d.rounded_rectangle((int(22*mag),int(h*.71),int(w-22*mag),int(h*.764)),
                            radius=int(11*mag),fill=(2,8,22,180),outline=(90,118,150,100),width=1)
        hfont=self.font_label
        hsize=max(12,int(w*.032))
        while d.textbbox((0,0),main_label,font=hfont)[2] > w-75*mag and hsize>11:
            hsize-=1
            hfont=get_font(hsize,True)
        d.text((int(38*mag),int(h*.724)),main_label,font=hfont,fill=accent)
        speech=''
        for st,en,line in VIDEO['beats']:
            if st<=t<en:
                speech=line; break
        if speech:
            ff=self.font_caption
            caption_lines=text_wrap(d,speech,ff,w-112*mag)
            fs=max(19,int(w*.047))
            while len(caption_lines)>3 and fs>max(16,int(w*.031)):
                fs=max(16,fs-2)
                ff=get_font(fs,True)
                caption_lines=text_wrap(d,speech,ff,w-112*mag)
            delta=max(int(fs*1.32),int(w*.044))
            cy=int(h*(.78 if w<500 else .81))
            boxh=(len(caption_lines)+1)*delta
            d.rounded_rectangle((int(24*mag),cy-int(16*mag),int(w-24*mag),cy+boxh),
                                radius=int(13*mag),fill=(2,5,16,215))
            for j,line in enumerate(caption_lines):
                bbox=d.textbbox((0,0),line,font=ff)
                xx=(w-(bbox[2]-bbox[0]))/2
                d.text((int(xx),cy+j*delta),line,font=ff,fill=ink,
                       stroke_width=max(1,int(1.2*mag)),stroke_fill=(5,8,17,255))
        footer=VIDEO['watermark']
        counter=f'{int(t):02d}s / {int(duration):02d}s'
        footerfont=self.font_small; fs=max(10,int(w*.026))
        maxfoot=w-240*mag
        while d.textbbox((0,0),footer,font=footerfont)[2]>maxfoot and fs>10:
            fs-=1;footerfont=get_font(fs,True)
        while len(footer)>3 and d.textbbox((0,0),footer,font=footerfont)[2]>maxfoot:
            footer=footer[:-3]+'…'
        d.text((int(34*mag),int(h*.949)),footer,font=footerfont,
               fill=(140,162,188,230))
        d.text((int(w-145*mag),int(h*.949)),counter,
               font=footerfont,fill=(140,162,188,230))
        return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)




# CINEMATIC RENDERING v2 -- all processing remains within this standalone file.
def cinematic_grade(R, frame, t, duration):
    """Scene-linear-inspired highlight bloom, soft flares, ACES-like contrast,
    subtle camera drift and film grain. Applied before typography."""
    w, h = R.w, R.h
    # Lens breathing and a nearly imperceptible camera track, preserving screen edges.
    s = 1.021 + .006*math.sin(.19*t)
    tx = 2.8*math.sin(.13*t + .3)
    ty = 4.0*math.cos(.11*t)
    M=np.array([[s,0,(1-s)*w*.5+tx], [0,s,(1-s)*h*.455+ty]],np.float32)
    frame[:] = cv2.warpAffine(frame,M,(w,h),flags=cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REFLECT_101)
    # Three complementary bloom scales, performed at quarter resolution.
    tiny=cv2.resize(frame,(max(2,w//4),max(2,h//4)),interpolation=cv2.INTER_AREA)
    luminance=(tiny[:,:,0].astype(np.float32)*.114 +
               tiny[:,:,1].astype(np.float32)*.587 +
               tiny[:,:,2].astype(np.float32)*.299)
    bright=np.clip((luminance-76)/105.,0.,1.)**1.5
    hot=(tiny.astype(np.float32)*bright[:,:,None])
    blur1=cv2.GaussianBlur(hot,(0,0),2.2)
    blur2=cv2.GaussianBlur(hot,(0,0),10.)
    bloom=cv2.resize(.28*blur1 + .15*blur2,(w,h),interpolation=cv2.INTER_LINEAR)
    # Narrow horizontal streaks from brightest areas; restrained.
    streak=cv2.GaussianBlur(hot,(0,0),sigmaX=max(4,w/36),sigmaY=.35)
    streak=cv2.resize(streak,(w,h),interpolation=cv2.INTER_LINEAR)
    arr=frame.astype(np.float32)
    arr += bloom + .14*streak
    # Gentle photographic tone curve: darker shadow floor, controlled highlights.
    arr=np.maximum(0,arr-2.2)
    x=arr*(1.15/255.0)
    arr=255.*np.clip((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0,1)
    # Lens falloff, stronger in the corners but subtle around central subject.
    arr *= R._cinema_vignette[:,:,None]
    rng=np.random.default_rng(SEED+int(t*97))
    # Tiny sensor dither (prevents gradient banding in video compression).
    grain=rng.normal(0,1.25,(max(1,h//2),max(1,w//2))).astype(np.float32)
    grain=cv2.resize(grain,(w,h),interpolation=cv2.INTER_LINEAR)
    arr += grain[:,:,None]
    frame[:]=np.uint8(np.clip(arr,0,255))
    return frame


def cinematic_setup(R):
    y,x=np.ogrid[0:R.h,0:R.w]
    rr=((x-R.w*.50)/(R.w*.78))**2+((y-R.h*.46)/(R.h*.75))**2
    R._cinema_vignette=np.clip(1.-.23*rr, .67,1.).astype(np.float32)


def cinematic_atmosphere(R,frame,t):
    """Depth-separated luminous mote layers and wispy traveling light shafts."""
    w,h=R.w,R.h
    if not hasattr(R,'_motefield'):
        rng=np.random.default_rng(SEED+888)
        R._motefield=(rng.uniform(0,w,90),rng.uniform(.12*h,.79*h,90),
                      rng.uniform(.17,1,90),rng.uniform(-1,1,90))
    x,y,dep,spd=R._motefield
    xx=(x + (9+25*dep)*np.sin(t*.1+spd*5))%w
    yy=(y + (1+6*dep)*np.sin(t*.12+spd*11))
    for j in range(len(x)):
        if not (.13*h<yy[j]<.77*h): continue
        rad=1 if dep[j]<.75 else 2
        alpha=.10+.26*dep[j]
        color=tuple(int(v*alpha) for v in ((130,155,190) if j%4 else (220,174,109)))
        cv2.circle(frame,(int(xx[j]),int(yy[j])),rad,color,-1,cv2.LINE_AA)
    return frame


# Shared visual primitives; bundled here so every script is fully self-contained.
def bh_core(R, img, x, y, rs, draw_ring=True, halo=True):
    """Volumetric-looking photon/shadow illustration; Schwarzschild b_crit ≈ 2.598 Rs.
    Circular ring on image plane is illustrative, not GR ray tracing."""
    shadow=2.598076211*rs
    if halo: R.draw_halo(img,x,y,shadow*2.45,(78,113,217),.20)
    cx=int(x);cy=int(y)
    if draw_ring:
        ext=int(max(10,shadow*2.5))
        x0=max(0,cx-ext);x1=min(R.w,cx+ext+1)
        y0=max(0,cy-ext);y1=min(R.h,cy+ext+1)
        if x1>x0 and y1>y0:
            gy,gx=np.ogrid[y0:y1,x0:x1]
            dx=(gx-x).astype(np.float32);dy=(gy-y).astype(np.float32)
            radius=np.sqrt(dx*dx+dy*dy)/(shadow+1e-6)
            phi=np.arctan2(dy,dx)
            # Opposite sides have unequal brightness (inspired by Doppler beaming).
            asym=np.clip(1.05+.68*np.cos(phi+.24),.30,1.85)
            thin=np.exp(-.5*((radius-1.052)/.026)**2)
            fringe=np.exp(-.5*((radius-1.115)/.055)**2)
            halo_ring=np.exp(-.5*((radius-1.15)/.19)**2)
            intensity=(172*thin+45*fringe+11*halo_ring)*asym
            layer=np.empty((y1-y0,x1-x0,3),np.float32)
            # BGR: cyan-white hot region; cooler burnt-amber on far side.
            layer[:,:,0]=intensity*(.60+.22*np.cos(phi))
            layer[:,:,1]=intensity*(.78+.14*np.cos(phi))
            layer[:,:,2]=intensity*(1.08-.10*np.cos(phi))
            dest=img[y0:y1,x0:x1,:]
            dest[:]=np.uint8(np.clip(dest.astype(np.float32)+layer,0,255))
    cv2.circle(img,(cx,cy),max(2,int(shadow*.973)),(0,1,4),-1,cv2.LINE_AA)
    # Keep a subtly illuminated silhouette edge to give depth.
    if draw_ring:
        cv2.circle(img,(cx,cy),max(2,int(shadow*.990)),(10,16,27),1,cv2.LINE_AA)
    return shadow

def disk_cloud(R,img,cx,cy,r0,t,n=6000,tilt=.24,brighten=True):
    # Kepler-like differential angular velocity; NOT GR ray tracing or MHD.
    if 'disk_cloud_v2' not in R.disks:
        g=np.random.default_rng(SEED+712)
        r=np.sqrt(g.random(n)*.88+.12)
        a=g.uniform(0,2*np.pi,n)
        intensity=g.uniform(.25,1.0,n)
        R.disks['disk_cloud_v2']=(r,a,intensity)
    r,a,lum=R.disks['disk_cloud_v2']
    th=a + .46*t/(.2+r)**1.5
    rr=r0*(.7+1.5*r)
    x=cx+rr*np.cos(th)
    y=cy+rr*tilt*np.sin(th)
    approaching=np.clip(.55+.45*np.cos(th),.1,1.)
    lum=lum*approaching if brighten else lum
    R.pointfield(img,x,y,(73,169,255),lum*.95,1.3)
    # Behind-the-hole far-side glow is lifted into a stylized lensed arch.
    theta=np.linspace(.08,np.pi-.08,180)
    for scale,col in ((1.,(75,159,240)),(1.24,(49,108,174))):
        ax=cx+scale*r0*1.58*np.cos(theta)
        ay=cy-r0*(.78+.19*scale)*np.sin(theta)
        pts=np.stack([ax,ay],axis=-1).astype(np.int32)
        cv2.polylines(img,[pts],False,col,max(1,int(R.w/380)),cv2.LINE_AA)


def clock_face(R,img,x,y,r,phase,color=(215,215,255),label=None):
    x=int(x);y=int(y);r=max(8,int(r))
    cv2.circle(img,(x,y),r,(42,73,118),max(1,int(R.w/450)),cv2.LINE_AA)
    cv2.circle(img,(x,y),int(r*.92),color,1,cv2.LINE_AA)
    for j in range(12):
        a=2*np.pi*j/12
        p=(x+int(.82*r*np.sin(a)),y-int(.82*r*np.cos(a)))
        q=(x+int(.91*r*np.sin(a)),y-int(.91*r*np.cos(a)))
        cv2.line(img,p,q,color,2 if j%3==0 else 1,cv2.LINE_AA)
    hand=(x+int(.66*r*np.sin(phase)),y-int(.66*r*np.cos(phase)))
    cv2.line(img,(x,y),hand,color,max(1,int(R.w/330)),cv2.LINE_AA)
    cv2.circle(img,(x,y),max(2,int(r*.055)),(235,244,255),-1,cv2.LINE_AA)
    if label:R.circle_text(img,label,x-r,y+r+24*R.w/720,(178,207,241),.75)


def wave_signal(t,tc=1.0):
    # Shape mimics a compact-binary chirp but is not a detector reconstruction.
    s=np.clip(np.asarray(t,dtype=float),0.0,tc)
    u=np.maximum(.04,1.0-s/tc)
    phase=46*(1-u**.625)
    amp=.12/np.sqrt(np.sqrt(u))
    taper=np.where(s<.84*tc,1.,np.exp(-(s-.84*tc)*21))
    return amp*taper*np.sin(phase)


def elastic_grid(R,img,cx,cy,depth=1.0,skew=0.0):
    # An embedding diagram for curved SPACE in one slice; not spacetime itself.
    for which in range(2):
        for idx in range(-8,9):
            points=[]
            for v in np.linspace(-1,1,100):
                u=(idx/9.0) if which==0 else v
                z=v if which==0 else (idx/9.0)
                d=np.sqrt(u*u+z*z+.035)
                bend=-depth/(1+6*d*d)
                xx=cx+R.w*.38*(u+.11*z)+skew*R.w*.01*z
                yy=cy+R.h*.18*z+R.h*.10*bend
                points.append((int(xx),int(yy)))
            c=(30,61,102) if idx%2 else (47,102,156)
            cv2.polylines(img,[np.asarray(points,np.int32)],False,c,1,cv2.LINE_AA)

def scene(R,img,t,dur):
    w,h=R.w,R.h;q=t/dur;cx=w*.50;cy=h*.43;rs=w*.045
    bh_core(R,img,cx,cy,rs)
    a=np.sqrt(1-1/2);b=np.sqrt(1-1/8)
    clock_face(R,img,w*.22,h*.635,w*.103,2*np.pi*(t*a/5.1),label='2 Rs / NEAR')
    clock_face(R,img,w*.77,h*.635,w*.103,2*np.pi*(t*b/5.1),label='8 Rs / FAR')
    cv2.line(img,(int(w*.23),int(h*.53)),(int(cx-w*.13),int(cy)),(54,119,184),1,cv2.LINE_AA)
    cv2.line(img,(int(w*.77),int(h*.53)),(int(cx+w*.13),int(cy)),(54,119,184),1,cv2.LINE_AA)
    R.circle_text(img,'STATIC CLOCKS / SCHWARZSCHILD',w*.12,h*.28,(199,225,250),.79)
    R.circle_text(img,'0.707x',w*.13,h*.735,(187,221,255),.91)
    R.circle_text(img,'0.935x',w*.67,h*.735,(187,221,255),.91)
    R.circle_text(img,'LOCAL CLOCK RATE / DISTANT TIME',w*.10,h*.34,(180,199,232),.73)
 

def generate(args):
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    name=VIDEO['slug']
    meta_path=output/(name+'_youtube.txt')
    srt_path=output/(name+'.srt')
    video_path=output/(name+'.mp4')
    duration=float(VIDEO['duration'])
    beats=[tuple(b) for b in VIDEO['beats']]
    with open(srt_path,'w',encoding='utf8') as f:
        for j,(st,en,line) in enumerate(beats,1):
            f.write(f'{j}\n{fmt_srt(st)} --> {fmt_srt(en)}\n{line}\n\n')
    with open(meta_path,'w',encoding='utf8') as f:
        f.write('TITLE\n'+VIDEO['title']+'\n\nDESCRIPTION\n'+VIDEO['description']+'\n\n')
        f.write('METHOD & SCIENTIFIC ACCURACY\n'+VIDEO['method']+'\n\n')
        f.write('SOURCES\n')
        for a,b in VIDEO['sources']: f.write(a+'\n'+b+'\n')
        f.write('\nNARRATION SCRIPT\n'+ '\n'.join(row[2] for row in beats)+'\n')
        f.write('\nTAGS\n'+', '.join(VIDEO['tags'])+'\n')
    width=args.width; height=int(round(width*16/9))
    if width%2:width+=1
    if height%2:height+=1
    fps=args.fps
    cut=min(float(args.preview_seconds),duration) if args.preview_seconds else duration
    renderer=Renderer(width,height)
    frame_count=int(math.ceil(cut*fps))
    with tempfile.TemporaryDirectory() as td:
        raw=str(Path(td)/'silent.mp4')
        wav=str(Path(td)/'soundtrack.wav')
        if not shutil.which('ffmpeg'):
            # OpenCV output works without ffmpeg executable; audio is unavailable.
            print('WARNING: ffmpeg executable unavailable; using silent video.')
        codec=cv2.VideoWriter_fourcc(*'mp4v')
        vid=cv2.VideoWriter(raw,codec,fps,(width,height))
        if not vid.isOpened(): raise RuntimeError('Could not open MP4 writer (OpenCV mp4v).')
        for i in range(frame_count):
            t=i/fps
            frame=renderer.start(t)
            cinematic_atmosphere(renderer,frame,t)
            scene(renderer,frame,t,duration)
            frame=cinematic_grade(renderer,frame,t,duration)
            frame=renderer.label(frame,t,duration)
            vid.write(frame)
            if i%max(1,round(fps*5))==0:
                print(f'  {i+1}/{frame_count} frames',flush=True)
        vid.release()
        if shutil.which('ffmpeg'):
            narration=synth_audio(wav,cut,beats,voice=not args.no_voice)
            call=['ffmpeg','-y','-loglevel','error','-i',raw,'-i',wav,
                  '-map','0:v:0','-map','1:a:0','-c:v','libx264',
                  '-preset','medium','-crf','17','-pix_fmt','yuv420p',
                  '-c:a','aac','-b:a','128k','-movflags','+faststart',
                  '-shortest',str(video_path)]
            r=subprocess.run(call,capture_output=True,text=True)
            if r.returncode:
                print('ffmpeg H264/AAC failed; keeping OpenCV MP4:',r.stderr[-500:])
                shutil.copyfile(raw,video_path)
        else:
            narration='No audio: ffmpeg executable not found'
            shutil.copyfile(raw,video_path)
    print('\nCreated:')
    for p in (video_path,srt_path,meta_path):print(' -',p)
    print('Audio:',narration)
    print('Scientific mode:',VIDEO['method'])


def cli():
    ap=argparse.ArgumentParser(description='Standalone cinematic space Shorts renderer')
    ap.add_argument('--output',default='.',help='Folder for generated MP4, SRT and YouTube TXT')
    ap.add_argument('--width',type=int,default=720,help='9:16 video width; 1080 for Full HD (default 720)')
    ap.add_argument('--fps',type=int,default=30)
    ap.add_argument('--preview-seconds',type=float,default=None,help='Fast rendering test; leaves full SRT & metadata')
    ap.add_argument('--no-voice',action='store_true',help='Generate score only, no optional espeak voice-over')
    args=ap.parse_args()
    if args.width<180 or args.fps<1:ap.error('width must be >= 180 and fps >= 1')
    generate(args)

if __name__=='__main__':
    cli()

