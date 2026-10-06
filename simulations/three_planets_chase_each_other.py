from cinematic_space_engine_v2 import *


def draw(r, img, t, sh, p):
    c = (r.W*0.5, r.H*0.42)
    R = r.W*0.31
    cols = [(70,220,255), (255,110,145), (186,130,255)]
    if sh.name == 'cold_open':
        r.glow_circle(img, c, r.W*0.044, (255,188,80), core=(255,247,220,255), blur=34)
        r.orbit(img, c, R, R*0.34, alpha=110, width=3)
        for j, col in enumerate(cols):
            a = t*0.92 + j*2.094
            pts = [(c[0]+math.cos(a-k*0.06)*R, c[1]+math.sin(a-k*0.06)*R*0.34) for k in range(28, -1, -1)]
            r.trail(img, pts, col, 5)
            pt = pts[-1]
            r.glow_circle(img, pt, r.W*0.018, col, core=(*col,255), blur=12)
        r.text_center(img, 'THREE WORLDS. ONE TRACK.', int(r.H*0.18), size=55)
    elif sh.name == 'triangle_lock':
        r.glow_circle(img, c, r.W*0.043, (255,190,84), core=(255,248,224,255), blur=34)
        r.orbit(img, c, R, R*0.35, alpha=100, width=3)
        pts=[]
        for j, col in enumerate(cols):
            a = -0.6 + j*2.094
            pt = (c[0]+math.cos(a)*R, c[1]+math.sin(a)*R*0.35)
            pts.append(pt)
            r.glow_circle(img, pt, r.W*0.02, col, core=(*col,255), blur=10)
        d=ImageDraw.Draw(img,'RGBA')
        for a, b in zip(pts, pts[1:]+pts[:1]):
            d.line([a,b], fill=(205,230,255,120), width=max(1,int(3*r.S)))
        r.text_center(img, 'KEEP THEM 120° APART', int(r.H*0.20), size=60, fill=(115,230,255,255))
    elif sh.name == 'rotating_frame':
        r.glow_circle(img, c, r.W*0.04, (255,190,80), core=(255,248,220,255), blur=30)
        r.orbit(img, c, R, R*0.36, alpha=90, width=2, dash=True)
        for j, col in enumerate(cols):
            a = t*0.25 + j*2.094
            pt = (c[0]+math.cos(a)*R, c[1]+math.sin(a)*R*0.36)
            r.glow_circle(img, pt, r.W*0.018, col, core=(*col,255), blur=10)
            r.arrow(img, pt, (pt[0]-math.sin(a)*54*r.S, pt[1]+math.cos(a)*22*r.S), color=col, width=4)
        r.hud(img, 'ROTATING FRAME', 'co-orbital triangle', int(r.H*0.16))
        r.hud(img, 'SPACING', 'stable only if phase holds', int(r.H*0.23), color=(255,190,110))
    elif sh.name == 'perturbation':
        r.glow_circle(img, c, r.W*0.04, (255,190,80), core=(255,248,220,255), blur=30)
        r.orbit(img, c, R, R*0.36, alpha=80, width=2)
        for j, col in enumerate(cols):
            jitter = 0.05*math.sin(t*7+j*2)*p
            a = t*0.7 + j*2.094 + jitter
            rr = R*(1+0.09*math.sin(t*2.5+j)*p)
            pt = (c[0]+math.cos(a)*rr, c[1]+math.sin(a)*R*0.36*(1+0.06*math.cos(t*4+j)*p))
            r.glow_circle(img, pt, r.W*0.018, col, core=(*col,255), blur=9)
        r.meter(img, int(r.W*0.13), int(r.H*0.22), int(r.W*0.74), 0.34+0.46*p, 'CHAOS RISK', color=(255,130,110))
        r.text_center(img, 'TINY NUDGES CAN RUIN IT', int(r.H*0.18), size=48)
    elif sh.name == 'phase_space':
        ox, oy = r.W*0.12, r.H*0.67
        ww, hh = r.W*0.76, r.H*0.28
        r.draw_grid(img, ox, oy-hh, ww, hh, rows=4, cols=5)
        pts=[]
        for i in range(220):
            q=i/219
            x=ox+ww*q
            y=oy-hh*0.52-math.sin(q*math.tau*2.2+t*0.45)*hh*0.22*(1-q*0.25)
            pts.append((x,y))
        r.trail(img, pts, (85,228,255), 4)
        r.text_center(img, 'A DANCE, NOT A GUARANTEE', int(r.H*0.19), size=46)
        r.small_label(img, 'energy exchange', int(r.W*0.14), int(r.H*0.44))
    elif sh.name == 'close_encounter':
        r.glow_circle(img, c, r.W*0.04, (255,190,80), core=(255,248,220,255), blur=30)
        for j, col in enumerate(cols):
            a = 1.1 + j*0.45 + (-1 if j==2 else 1)*p*0.35
            pt = (c[0]+math.cos(a)*R*(0.9+0.1*j*0.1), c[1]+math.sin(a)*R*0.36)
            pts = [(pt[0]-k*6*r.S*(j-1), pt[1]+math.sin(k*0.22+t+j)*3*r.S) for k in range(10)]
            r.trail(img, pts, col, 5)
            r.glow_circle(img, pt, r.W*0.022 if j==1 else r.W*0.018, col, core=(*col,255), blur=11)
        r.text_center(img, 'GET TOO CLOSE — AND IT BREAKS', int(r.H*0.19), size=44, fill=(255,180,110,255))
    else:
        zoom = lerp(1.0, 0.18, ease(p))
        rr = R * zoom
        r.glow_circle(img, c, r.W*0.038*max(0.55,zoom), (255,190,80), core=(255,248,220,255), blur=28)
        for j, col in enumerate(cols):
            a = t*0.68 + j*2.094
            pt = (c[0]+math.cos(a)*rr, c[1]+math.sin(a)*rr*0.36)
            r.glow_circle(img, pt, max(3, r.W*0.018*zoom), col, core=(*col,255), blur=10)
        r.text_center(img, 'ONLY UNDER VERY SPECIAL CONDITIONS', int(r.H*0.19), size=42)

shots = [
    Shot('cold_open', 0, 7, 'Could three planets share one orbit and chase one another forever? In theory, yes — but the geometry has to be almost perfect.'),
    Shot('triangle_lock', 7, 14, 'The cleanest setup places the three planets roughly 120 degrees apart, forming an orbiting triangle.'),
    Shot('rotating_frame', 14, 22, 'In the rotating frame, each world nearly holds position relative to the other two.'),
    Shot('perturbation', 22, 31, 'But even tiny perturbations can grow. Stability is the whole problem.'),
    Shot('phase_space', 31, 40, 'Real long-term survival depends on mass ratio, resonance locking, and whether energy gets traded gently or violently.'),
    Shot('close_encounter', 40, 50, 'If the spacing collapses, the planets can scatter one another out of the elegant pattern.'),
    Shot('outro', 50, 58, 'So three planets can chase each other for a very long time — but only as a finely tuned gravitational choreography.')
]

spec = Spec(
    title='CAN THREE PLANETS CHASE EACH OTHER FOREVER?',
    subtitle='co-orbital stability // resonance geometry // a delicate three-body dance',
    basename='can_three_planets_chase_each_other_forever',
    shots=shots,
    draw=draw,
    notes=[
        'This is a cinematic educational rendering of a co-orbital three-planet concept, not a precision N-body proof.',
        'The 120-degree arrangement is inspired by stable Lagrange-style configurations.',
        'Long-term survival would depend on masses, eccentricities, inclinations, migration history, and external perturbations.'
    ]
)

