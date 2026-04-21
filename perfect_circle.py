"""
Perfect Circle — Python Version
================================
A gesture-controlled precision challenge using:
  - MediaPipe Hands  : hand/finger tracking via webcam
  - OpenCV           : webcam capture
  - Pygame           : rendering & UI

Install dependencies:
  pip uninstall opencv-python -y
  pip install mediapipe opencv-python-headless pygame

Run:
  python perfect_circle.py
"""

import pygame
import cv2
import math
import json

# MediaPipe version-safe import
try:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    import mediapipe as mp
    # Test if old API works
    _ = mp.solutions.hands
    USE_NEW_MP = False
except AttributeError:
    USE_NEW_MP = True
except Exception:
    import mediapipe as mp
    USE_NEW_MP = False
import os
import time
import sys
import random

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
LB_FILE          = "perfect_circle_scores.json"
GESTURE_CONFIRM  = 10
MIN_POINTS       = 60
CLOSURE_PX       = 40
MIN_RADIUS       = 75
HAND_LOST_MS     = 1500

TIERS = [
    {"min": 90, "label": "Perfect Circle",   "color": (255, 215,   0)},
    {"min": 78, "label": "Legendary Circle", "color": (  0, 229, 255)},
    {"min": 65, "label": "Master Circle",    "color": (  0, 255, 136)},
    {"min": 50, "label": "Great Circle",     "color": (  0, 255, 136)},
    {"min": 30, "label": "Decent Attempt",   "color": (245, 255,   0)},
    {"min":  0, "label": "Keep Practicing",  "color": (255,  61,  90)},
]

BG_COLOR   = ( 5,  10,  14)
GREEN      = ( 0, 255, 136)
YELLOW     = (245, 255,   0)
CYAN       = (  0, 229, 255)
RED        = (255,  61,  90)
GOLD       = (255, 215,   0)
DIM_GREEN  = (  0, 102,  55)
WHITE      = (255, 255, 255)

STATE_NAME    = 0
STATE_IDLE    = 1
STATE_DRAWING = 2
STATE_SCORING = 3

# ─── INIT ─────────────────────────────────────────────────────────────────────
pygame.init()
INFO   = pygame.display.Info()
W, H   = INFO.current_w, INFO.current_h
screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
pygame.display.set_caption("Perfect Circle")
clock  = pygame.time.Clock()
CX, CY = W // 2, H // 2

def get_font(size, bold=False):
    for name in ["Courier New", "Consolas", "DejaVu Sans Mono", "monospace"]:
        try:
            f = pygame.font.SysFont(name, size, bold=bold)
            if f: return f
        except: pass
    return pygame.font.Font(None, size)

font_sm    = get_font(18)
font_md    = get_font(24)
font_lg    = get_font(38, bold=True)
font_xl    = get_font(96, bold=True)
font_title = get_font(68, bold=True)

# ─── LEADERBOARD ──────────────────────────────────────────────────────────────
def load_board():
    if not os.path.exists(LB_FILE): return []
    try:
        with open(LB_FILE) as f: return json.load(f)
    except: return []

def save_score(name, score):
    board = load_board()
    name  = name.strip().upper()
    found = next((e for e in board if e["name"] == name), None)
    if found:
        if score > found["score"]: found["score"] = score
    else:
        board.append({"name": name, "score": score})
    board.sort(key=lambda x: -x["score"])
    board = board[:5]
    with open(LB_FILE, "w") as f: json.dump(board, f)
    return board

# ─── SCORING ──────────────────────────────────────────────────────────────────
def taubin_fit(pts):
    """Taubin algebraic circle fit — finds the mathematically best-fit circle
    through a set of points. Returns (cx, cy, radius).
    More accurate than centroid-based fitting, especially for partial arcs."""
    n = len(pts)
    if n < 3:
        cx = sum(p[0] for p in pts)/n
        cy = sum(p[1] for p in pts)/n
        r  = sum(math.hypot(p[0]-cx,p[1]-cy) for p in pts)/n
        return cx, cy, r

    mx = sum(p[0] for p in pts)/n
    my = sum(p[1] for p in pts)/n

    # Shift to centroid for numerical stability
    xs = [p[0]-mx for p in pts]
    ys = [p[1]-my for p in pts]

    Suu = sum(x*x for x in xs)/n
    Svv = sum(y*y for y in ys)/n
    Suv = sum(xs[i]*ys[i] for i in range(n))/n
    Suuu= sum(xs[i]**3 for i in range(n))/n
    Svvv= sum(ys[i]**3 for i in range(n))/n
    Suvv= sum(xs[i]*ys[i]**2 for i in range(n))/n
    Svuu= sum(ys[i]*xs[i]**2 for i in range(n))/n

    # Solve 2x2 linear system
    A = [[2*Suu, 2*Suv],
         [2*Suv, 2*Svv]]
    b = [Suuu + Suvv,
         Svvv + Svuu]
    det = A[0][0]*A[1][1] - A[0][1]*A[1][0]
    if abs(det) < 1e-10:
        # Degenerate — fall back to centroid
        r = sum(math.hypot(xs[i],ys[i]) for i in range(n))/n
        return mx, my, r

    uc = (b[0]*A[1][1] - b[1]*A[0][1]) / det
    vc = (A[0][0]*b[1] - A[1][0]*b[0]) / det

    cx = uc + mx
    cy = vc + my
    r  = math.sqrt(uc**2 + vc**2 + Suu + Svv)
    return cx, cy, r


def compute_score(pts):
    """Option C scoring: circularity × centration.

    Part A — CIRCULARITY (how close to a perfect circle is the shape?)
      Uses Taubin best-fit circle. Measures how much each point deviates
      from the fitted radius, plus smoothness and closure.

    Part B — CENTRATION (is the circle centered on the green dot?)
      Measures the distance between the fitted circle center and (CX,CY),
      expressed as a fraction of the fitted radius.

    Final = circularity_score × centration_multiplier
    So a perfect oval centered on the dot still loses points for shape,
    and a perfect circle drawn off-center still loses points for placement.
    """
    if len(pts) < 10: return 0, 0, 0

    # ── Part A: Circularity ──────────────────────────────────────────────────

    # Taubin fit: best-fit circle through all drawn points
    fit_cx, fit_cy, fit_r = taubin_fit(pts)
    if fit_r < 1: return 0, 0, 0

    # 1. Radius consistency — how far each point is from the fitted radius
    radii = [math.hypot(p[0]-fit_cx, p[1]-fit_cy) for p in pts]
    mean  = sum(radii)/len(radii)
    var   = sum((r-fit_r)**2 for r in radii)/len(radii)
    sigma = math.sqrt(var)
    cv    = sigma / fit_r
    radius_score = max(0.0, 100.0 * math.exp(-5.0 * cv))

    # 2. Smoothness — angular speed should be constant around the circle
    angles = [math.atan2(p[1]-fit_cy, p[0]-fit_cx) for p in pts]
    for i in range(1, len(angles)):
        d = angles[i]-angles[i-1]
        if d >  math.pi: angles[i] -= 2*math.pi
        if d < -math.pi: angles[i] += 2*math.pi
    steps = [angles[i]-angles[i-1] for i in range(1,len(angles))]
    if steps:
        ms   = sum(steps)/len(steps)
        sv   = sum((s-ms)**2 for s in steps)/len(steps)
        scv  = math.sqrt(sv)/abs(ms) if ms else 1.0
        smooth_score = max(0.0, 100.0 * math.exp(-3.0 * scv))
    else:
        smooth_score = 0.0

    # 3. Closure — did the path actually close?
    gap = math.hypot(pts[-1][0]-pts[0][0], pts[-1][1]-pts[0][1])
    closure_score = max(0.0, 100.0 * math.exp(-4.0 * (gap/fit_r)))

    # 4. Fullness — full 360° drawn?
    total_angle = abs(angles[-1]-angles[0]) if len(angles)>1 else 0
    fullness_score = min(total_angle/(2*math.pi), 1.0) * 100.0

    # Weighted circularity
    circularity = (
        0.60 * radius_score  +
        0.20 * smooth_score  +
        0.12 * closure_score +
        0.08 * fullness_score
    )

    # ── Part B: Centration ───────────────────────────────────────────────────
    # Distance from fitted circle center to the green anchor dot (CX, CY)
    center_dist = math.hypot(fit_cx - CX, fit_cy - CY)
    # Express as fraction of the fitted radius
    # 0 = perfectly centered, 1 = center is one full radius away (very off)
    center_ratio = center_dist / fit_r
    # Multiplier: 1.0 at perfect center, drops smoothly
    # At 10% off-center → ~0.90×, 25% → ~0.70×, 50% → ~0.40×, 100% → ~0.08×
    centration_mult = math.exp(-1.5 * center_ratio)

    # ── Final score ──────────────────────────────────────────────────────────
    score = circularity * centration_mult
    score = max(0.0, min(100.0, score))
    return score, fit_r, sigma, circularity, centration_mult * 100.0

def point_in_poly(px_, py_, poly):
    inside, j = False, len(poly)-1
    for i in range(len(poly)):
        xi,yi = poly[i]; xj,yj = poly[j]
        if ((yi>py_)!=(yj>py_)) and (px_ < (xj-xi)*(py_-yi)/(yj-yi)+xi):
            inside = not inside
        j = i
    return inside

def validate_attempt(pts):
    if len(pts) < MIN_POINTS: return False, "DRAW A COMPLETE CIRCLE"
    _, mean, _, _, _ = compute_score(pts)
    if mean < MIN_RADIUS: return False, "CIRCLE TOO SMALL"
    if not point_in_poly(CX, CY, pts): return False, "CENTER DOT MUST BE INSIDE"
    return True, ""

def get_tier(score):
    for t in TIERS:
        if score >= t["min"]: return t
    return TIERS[-1]

# ─── GESTURE DETECTION ────────────────────────────────────────────────────────
def is_index_only(lm, h, w):
    return (lm[8].y < lm[6].y and lm[6].y < lm[5].y and
            lm[12].y > lm[10].y and lm[16].y > lm[14].y and lm[20].y > lm[18].y)

def is_peace_sign(lm, h, w):
    return (lm[8].y < lm[6].y and lm[12].y < lm[10].y and
            lm[16].y > lm[14].y and lm[20].y > lm[18].y)

def is_thumb_down(lm, h, w):
    return (lm[4].y > lm[3].y and lm[3].y > lm[2].y and
            lm[8].y > lm[6].y and lm[12].y > lm[10].y and
            lm[16].y > lm[14].y and lm[20].y > lm[18].y)

def lm_to_screen(lm, idx):
    # Frame already flipped by cv2.flip, so use x directly
    return int(lm[idx].x*W), int(lm[idx].y*H)

# ─── POINT SMOOTHING ──────────────────────────────────────────────────────────
# Raw MediaPipe landmarks are jittery — apply exponential smoothing to the
# finger tip position, and Chaikin curve subdivision before scoring/display.

_smooth_tip = None   # exponentially smoothed tip position

def smooth_tip_position(raw_tip, alpha=0.35):
    """Exponential moving average. Lower alpha = smoother but more lag."""
    global _smooth_tip
    if _smooth_tip is None or raw_tip is None:
        _smooth_tip = raw_tip
        return raw_tip
    sx = _smooth_tip[0] + alpha * (raw_tip[0] - _smooth_tip[0])
    sy = _smooth_tip[1] + alpha * (raw_tip[1] - _smooth_tip[1])
    _smooth_tip = (sx, sy)
    return (int(sx), int(sy))

def reset_smooth_tip():
    global _smooth_tip
    _smooth_tip = None

def chaikin_smooth(pts, iterations=2):
    """Chaikin corner-cutting: repeatedly replace each segment with two points
    at 1/4 and 3/4 along it. Produces a smooth curve through the data."""
    for _ in range(iterations):
        if len(pts) < 3:
            break
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i+1]
            new_pts.append((int(0.75*x0 + 0.25*x1), int(0.75*y0 + 0.25*y1)))
            new_pts.append((int(0.25*x0 + 0.75*x1), int(0.25*y0 + 0.75*y1)))
        new_pts.append(pts[-1])
        pts = new_pts
    return pts

MIN_DIST_PX = 4   # only record a new point if finger moved at least this far

# ─── PARTICLES ────────────────────────────────────────────────────────────────
class Particle:
    def __init__(self, cx, cy):
        angle = random.uniform(0, math.pi*2)
        speed = random.uniform(2, 11)
        self.x, self.y = float(cx), float(cy)
        self.vx = math.cos(angle)*speed
        self.vy = math.sin(angle)*speed - random.uniform(0,3)
        self.life  = 1.0
        self.decay = 0.012 + random.uniform(0, 0.008)
        self.size  = random.uniform(1.5, 5.5)
        h = random.randint(35, 65)
        self.color = (255, int(255*(h/60)), 0)
    def update(self):
        self.x+=self.vx; self.y+=self.vy
        self.vy+=0.18; self.vx*=0.98; self.life-=self.decay
    def draw(self, surf):
        if self.life <= 0: return
        a = int(self.life**2 * 255)
        r = max(1, int(self.size*(0.4+self.life*0.6)))
        tmp = pygame.Surface((r*2+2,r*2+2), pygame.SRCALPHA)
        pygame.draw.circle(tmp, (*self.color, a), (r+1,r+1), r)
        surf.blit(tmp, (int(self.x)-r-1, int(self.y)-r-1))

# ─── PULSE RINGS ──────────────────────────────────────────────────────────────
class PulseRing:
    def __init__(self, cx, cy, delay):
        self.cx=cx; self.cy=cy; self.r=0; self.max_r=280; self.delay=delay; self.life=1.0
    def update(self, dt):
        if self.delay>0: self.delay-=dt; return
        self.r = min(self.r+6, self.max_r)
        self.life = 1.0 - self.r/self.max_r
    def draw(self, surf):
        if self.delay>0 or self.r==0 or self.life<=0: return
        a = int(self.life*0.8*255)
        w = max(1, int(2.5*(1-self.r/self.max_r)+0.5))
        tmp = pygame.Surface((W,H), pygame.SRCALPHA)
        pygame.draw.circle(tmp, (0,229,255,a), (self.cx,self.cy), int(self.r), w)
        surf.blit(tmp, (0,0))

# ─── DRAW HELPERS ─────────────────────────────────────────────────────────────
def draw_anchor_dot(surf, t):
    pulse  = 1 + math.sin(t*2)*0.15
    radius = int(6*pulse)
    glow   = pygame.Surface((radius*6,radius*6), pygame.SRCALPHA)
    for i in range(5,0,-1):
        pygame.draw.circle(glow,(0,255,136,int(40*(i/5))),(radius*3,radius*3),radius*i)
    surf.blit(glow,(CX-radius*3,CY-radius*3))
    pygame.draw.circle(surf,GREEN,(CX,CY),radius)
    pygame.draw.circle(surf,WHITE,(CX,CY),3)

def draw_path(surf, pts, color, width=3):
    if len(pts)>=2: pygame.draw.lines(surf,color,False,pts,width)

def draw_cursor(surf, tip, is_drawing):
    if not tip: return
    col = YELLOW if is_drawing else CYAN
    pygame.draw.circle(surf,col,tip,12,2)
    pygame.draw.circle(surf,col,tip,3)

def draw_corners(surf):
    s,t,pad=28,2,16
    for x,y,sx,sy in [(pad,pad,1,1),(W-pad,pad,-1,1),(pad,H-pad,1,-1),(W-pad,H-pad,-1,-1)]:
        pygame.draw.lines(surf,DIM_GREEN,False,[(x+s*sx,y),(x,y),(x,y+s*sy)],t)

def draw_hud(surf, state, hand_det, player_name, pts_count, fps):
    names  = {STATE_IDLE:"IDLE",STATE_DRAWING:"DRAWING",STATE_SCORING:"SCORING"}
    colors = {STATE_IDLE:GREEN,STATE_DRAWING:YELLOW,STATE_SCORING:CYAN}
    bar = pygame.Surface((W,42),pygame.SRCALPHA); bar.fill((5,10,14,180))
    surf.blit(bar,(0,H-42))
    pygame.draw.line(surf,DIM_GREEN,(0,H-42),(W,H-42),1)
    items=[
        ("FPS",str(fps),GREEN if fps>=28 else (YELLOW if fps>=20 else RED)),
        ("HAND","1/1" if hand_det else "0/1",GREEN if hand_det else RED),
        ("STATE",names.get(state,"—"),colors.get(state,GREEN)),
        ("PLAYER",player_name or "—",CYAN),
        ("PTS",str(pts_count) if state==STATE_DRAWING else "—",YELLOW),
    ]
    x=20
    for lbl,val,col in items:
        surf.blit(font_sm.render(lbl,True,DIM_GREEN),(x,H-36))
        surf.blit(font_sm.render(val,True,col),(x,H-18)); x+=140

def draw_leaderboard(surf, highlight_name):
    board = load_board()
    pw,ph = 240, 44+max(len(board),1)*28
    panel = pygame.Surface((pw,ph),pygame.SRCALPHA)
    panel.fill((5,10,14,200))
    pygame.draw.rect(panel,DIM_GREEN,panel.get_rect(),1)
    panel.blit(font_sm.render("◈ LEADERBOARD  TOP 5",True,GREEN),(10,8))
    rank_cols=[GOLD,(192,192,192),(205,127,50),DIM_GREEN,DIM_GREEN]
    if not board:
        panel.blit(font_sm.render("NO SCORES YET",True,DIM_GREEN),(10,36))
    for i,e in enumerate(board):
        y=34+i*28
        rc = rank_cols[i] if i<len(rank_cols) else DIM_GREEN
        nc = CYAN if e["name"]==highlight_name.upper() else GREEN
        panel.blit(font_sm.render(f"#{i+1}",True,rc),(8,y))
        panel.blit(font_sm.render(e["name"][:12],True,nc),(40,y))
        ss=font_sm.render(f"{e['score']:.1f}%",True,YELLOW)
        panel.blit(ss,(pw-ss.get_width()-8,y))
    surf.blit(panel,(20,20))

def draw_idle_prompt(surf, t):
    a=int((0.55+math.sin(t*1.5)*0.25)*255)
    for j,(fnt,txt,col) in enumerate([
        (font_md,"INDEX FINGER UP TO BEGIN",GREEN),
        (font_sm,"DRAW A CIRCLE AROUND THE DOT",DIM_GREEN),
        (font_sm,"THUMB DOWN — CHANGE PLAYER",RED),
    ]):
        s=fnt.render(txt,True,col); s.set_alpha(a)
        surf.blit(s,s.get_rect(center=(CX,CY+70+j*28)))

def draw_closure_guide(surf, pts, tip, has_moved_away):
    if len(pts)<20 or not has_moved_away or not tip: return
    sx,sy=pts[0]
    dist=math.hypot(tip[0]-sx,tip[1]-sy)
    closeness=max(0,1-dist/(CLOSURE_PX*3))
    if closeness<0.1: return
    tmp=pygame.Surface((W,H),pygame.SRCALPHA)
    pygame.draw.circle(tmp,(*YELLOW,int(closeness*0.7*255)),(int(sx),int(sy)),CLOSURE_PX,2)
    surf.blit(tmp,(0,0))

def draw_score_overlay(surf, score, tier, show_tier, t, circularity=None, centration=None):
    # ── Dark frosted panel behind everything so text is always readable ──
    panel_w, panel_h = 460, 260
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((5, 10, 14, 210))
    pygame.draw.rect(panel, tier["color"], panel.get_rect(), 2)
    # Pin panel to bottom-center so it never overlaps the drawn circle
    px = CX - panel_w // 2
    py = H - panel_h - 55   # just above the HUD bar
    surf.blit(panel, (px, py))

    # ── Label ──
    ls = font_sm.render("CIRCULARITY SCORE", True, DIM_GREEN)
    surf.blit(ls, ls.get_rect(center=(CX, py + 22)))

    # ── Big score number ──
    ss = font_xl.render(f"{score:.1f}%", True, tier["color"])
    surf.blit(ss, ss.get_rect(center=(CX, py + 95)))

    # ── Sub-scores and tier (shown after animation) ──
    if show_tier and circularity is not None and centration is not None:
        # SHAPE | CENTERING pills
        for i, (lbl, val, col) in enumerate([
            ("SHAPE",     f"{circularity:.0f}%", GREEN),
            ("CENTERING", f"{centration:.0f}%",  CYAN),
        ]):
            bx = CX - 110 + i * 220
            by = py + 155
            pill = pygame.Surface((160, 44), pygame.SRCALPHA)
            pill.fill((0, 0, 0, 160))
            pygame.draw.rect(pill, col, pill.get_rect(), 1)
            surf.blit(pill, (bx - 80, by - 22))
            surf.blit(font_sm.render(lbl, True, DIM_GREEN),
                      font_sm.render(lbl, True, DIM_GREEN).get_rect(center=(bx, by - 7)))
            surf.blit(font_md.render(val, True, col),
                      font_md.render(val, True, col).get_rect(center=(bx, by + 12)))

        # Tier label
        ts = font_lg.render(tier["label"].upper(), True, tier["color"])
        surf.blit(ts, ts.get_rect(center=(CX, py + 218)))

    # ── Blinking hint ──
    if int(t * 0.5) % 2 == 0:
        hs = font_sm.render("PEACE SIGN → PLAY AGAIN   |   THUMB DOWN → CHANGE PLAYER", True, DIM_GREEN)
        surf.blit(hs, hs.get_rect(center=(CX, H - 52)))

def draw_invalid_msg(surf, msg, alpha):
    if not msg or alpha<=0: return
    bw=font_sm.size(msg)[0]+40
    bg=pygame.Surface((bw,36),pygame.SRCALPHA)
    bg.fill((255,61,90,int(0.15*255)))
    pygame.draw.rect(bg,(*RED,int(0.4*255)),bg.get_rect(),1)
    surf.blit(bg,bg.get_rect(center=(CX,H-100)))
    ms=font_sm.render(msg,True,RED); ms.set_alpha(alpha)
    surf.blit(ms,ms.get_rect(center=(CX,H-100)))

# ─── NAME SCREEN ──────────────────────────────────────────────────────────────
def run_name_screen(cam_surf):
    name_text=""; error_flash=0
    while True:
        dt=clock.tick(60)/1000.0
        for event in pygame.event.get():
            if event.type==pygame.QUIT: pygame.quit(); sys.exit()
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE: pygame.quit(); sys.exit()
                elif event.key==pygame.K_RETURN:
                    if name_text.strip(): return name_text.strip().upper()
                    else: error_flash=0.8
                elif event.key==pygame.K_BACKSPACE: name_text=name_text[:-1]
                elif len(name_text)<12 and event.unicode.isprintable():
                    name_text+=event.unicode.upper()
        screen.fill(BG_COLOR)
        if cam_surf:
            bg=pygame.transform.scale(cam_surf,(W,H)); bg.set_alpha(180); screen.blit(bg,(0,0))
        ov=pygame.Surface((W,H),pygame.SRCALPHA); ov.fill((5,10,14,230)); screen.blit(ov,(0,0))
        ts=font_title.render("PERFECT CIRCLE",True,GREEN)
        screen.blit(ts,ts.get_rect(center=(CX,CY-120)))
        ss=font_sm.render("A GESTURE-CONTROLLED PRECISION CHALLENGE",True,DIM_GREEN)
        screen.blit(ss,ss.get_rect(center=(CX,CY-65)))
        bw,bh=380,52
        box=pygame.Rect(CX-bw//2,CY-bh//2,bw,bh)
        bc=RED if error_flash>0 else GREEN
        pygame.draw.rect(screen,(0,30,15),box)
        pygame.draw.rect(screen,bc,box,1)
        screen.blit(font_sm.render("ENTER YOUR CALLSIGN",True,DIM_GREEN),(box.x,box.y-24))
        disp=name_text+("_" if int(time.time()*2)%2==0 else " ")
        inp=font_md.render(disp,True,GREEN if name_text else DIM_GREEN)
        screen.blit(inp,(box.x+14,box.y+14))
        btn=pygame.Rect(CX-bw//2,CY+40,bw,44)
        bcol=YELLOW if btn.collidepoint(pygame.mouse.get_pos()) else GREEN
        pygame.draw.rect(screen,bcol,btn)
        bl=font_sm.render("BEGIN  →",True,BG_COLOR)
        screen.blit(bl,bl.get_rect(center=btn.center))
        if pygame.mouse.get_pressed()[0] and btn.collidepoint(pygame.mouse.get_pos()):
            if name_text.strip(): return name_text.strip().upper()
            else: error_flash=0.8
        hl=font_sm.render("OR PRESS ENTER",True,DIM_GREEN)
        screen.blit(hl,hl.get_rect(center=(CX,CY+100)))
        if error_flash>0: error_flash-=dt
        pygame.display.flip()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # Version-safe MediaPipe Hands setup
    if USE_NEW_MP:
        # mediapipe >= 0.10 new Tasks API
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python.core.base_options import BaseOptions
        import urllib.request, tempfile, os as _os
        model_path = _os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
        if not _os.path.exists(model_path):
            print("Downloading hand landmarker model (~8MB)...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                model_path)
        _options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO)  # VIDEO mode: uses temporal smoothing
        hands_det = HandLandmarker.create_from_options(_options)
        _frame_ts = 0  # timestamp in ms for VIDEO mode
        def process_frame(fr):
            import mediapipe as mp2
            nonlocal _frame_ts
            _frame_ts += 33  # ~30fps cadence
            mp_image = mp2.Image(image_format=mp2.ImageFormat.SRGB, data=fr)
            res = hands_det.detect_for_video(mp_image, _frame_ts)
            if res.hand_landmarks:
                # Wrap in an object that mimics old API landmark structure
                class _LM:
                    def __init__(self, x, y, z): self.x=x; self.y=y; self.z=z
                class _Result:
                    def __init__(self, lms): self.multi_hand_landmarks=[type('H', (), {'landmark': lms})()]
                lms = [_LM(lm.x, lm.y, lm.z) for lm in res.hand_landmarks[0]]
                return _Result(lms)
            return type('R', (), {'multi_hand_landmarks': None})()
    else:
        mp_hands  = mp.solutions.hands
        _detector = mp_hands.Hands(
            static_image_mode=False,      # video mode: reuses tracking between frames
            max_num_hands=1,
            model_complexity=1,           # 0=lite/fast, 1=full accuracy — keep at 1
            min_detection_confidence=0.6, # lower = detects hand sooner
            min_tracking_confidence=0.5,  # lower = holds tracking through fast motion
        )
        def process_frame(fr):
            return _detector.process(fr)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    state=STATE_NAME; player_name=""
    hand_lm=None; last_hand_time=time.time()*1000
    index_count=peace_count=thumb_dn_count=0
    points=[]; has_moved_away=False
    final_score=0.0; final_tier=None
    score_anim=0.0; score_anim_start=0; show_tier=False; score_saved=False
    score_circ=0.0; score_cent=0.0
    particles_list=[]; pulse_rings=[]; glow_active=False; red_blink=False; effect_circle=None
    invalid_msg_text=""; invalid_msg_alpha=0; invalid_msg_timer=0.0
    cam_surf=None; fps=60

    ret,frame=cap.read()
    if ret:
        fr=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        fr=cv2.flip(fr,1); fr=cv2.resize(fr,(W,H))
        cam_surf=pygame.surfarray.make_surface(fr.transpose(1,0,2))

    player_name=run_name_screen(cam_surf)
    state=STATE_IDLE

    def show_invalid(msg):
        nonlocal invalid_msg_text,invalid_msg_alpha,invalid_msg_timer
        invalid_msg_text=msg; invalid_msg_alpha=255; invalid_msg_timer=2.2

    def go_idle():
        nonlocal state,points,has_moved_away,particles_list,pulse_rings
        nonlocal glow_active,red_blink,effect_circle,index_count,peace_count
        nonlocal score_anim,show_tier,score_saved,score_circ,score_cent
        state=STATE_IDLE; points=[]; has_moved_away=False
        particles_list=[]; pulse_rings=[]; glow_active=False; red_blink=False
        effect_circle=None; index_count=0; peace_count=0
        score_anim=0.0; show_tier=False; score_saved=False
        score_circ=0.0; score_cent=0.0

    def go_drawing():
        nonlocal state,points,has_moved_away
        state=STATE_DRAWING; points=[]; has_moved_away=False
        reset_smooth_tip()

    while True:
        dt=clock.tick(60)/1000.0; fps=int(clock.get_fps())
        now_ms=time.time()*1000; t=time.time()
        for event in pygame.event.get():
            if event.type==pygame.QUIT: cap.release(); pygame.quit(); sys.exit()
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE: cap.release(); pygame.quit(); sys.exit()

        ret,frame=cap.read()
        if ret:
            fr=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            fr=cv2.flip(fr,1); fr=cv2.resize(fr,(W,H))
            cam_surf=pygame.surfarray.make_surface(fr.transpose(1,0,2))
            results=process_frame(fr)
            hand_lm=None
            if results.multi_hand_landmarks:
                hand_lm=results.multi_hand_landmarks[0].landmark
                last_hand_time=now_ms

        if hand_lm is None:
            index_count=peace_count=thumb_dn_count=0
        else:
            index_count    = min(index_count+1,   GESTURE_CONFIRM+5) if is_index_only(hand_lm,H,W)  else 0
            peace_count    = min(peace_count+1,   GESTURE_CONFIRM+5) if is_peace_sign(hand_lm,H,W)  else 0
            thumb_dn_count = min(thumb_dn_count+1,GESTURE_CONFIRM+5) if is_thumb_down(hand_lm,H,W)  else 0

        tip = lm_to_screen(hand_lm,8) if hand_lm else None

        if state==STATE_DRAWING:
            if hand_lm: last_hand_time=now_ms
            elif now_ms-last_hand_time>HAND_LOST_MS:
                show_invalid("HAND LOST — TRY AGAIN"); go_idle()

        if state==STATE_IDLE:
            if index_count>=GESTURE_CONFIRM: go_drawing()
            if thumb_dn_count>=GESTURE_CONFIRM:
                cam_surf2=cam_surf
                player_name=run_name_screen(cam_surf2); state=STATE_IDLE

        elif state==STATE_DRAWING:
            if tip:
                stip = smooth_tip_position(tip)  # smoothed tip for recording
                # Only add point if finger moved far enough (reduces jitter noise)
                if not points or math.hypot(stip[0]-points[-1][0], stip[1]-points[-1][1]) >= MIN_DIST_PX:
                    points.append(stip)
                if not has_moved_away and len(points)>1:
                    if math.hypot(stip[0]-points[0][0],stip[1]-points[0][1])>CLOSURE_PX*2:
                        has_moved_away=True
                if has_moved_away and len(points)>=MIN_POINTS:
                    if math.hypot(stip[0]-points[0][0],stip[1]-points[0][1])<=CLOSURE_PX:
                        points.append(points[0])
                        # Smooth the full path with Chaikin before scoring/display
                        smooth_pts = chaikin_smooth(points, iterations=3)
                        valid,reason=validate_attempt(smooth_pts)
                        if not valid:
                            show_invalid(reason); go_idle()
                        else:
                            final_score,_,_,_circ,_cent=compute_score(smooth_pts)
                            final_tier=get_tier(final_score)
                            score_circ=_circ; score_cent=_cent
                            effect_circle=smooth_pts; state=STATE_SCORING
                            score_anim=0.0; score_anim_start=t; show_tier=False; score_saved=False
                            particles_list=[]; pulse_rings=[]; glow_active=False; red_blink=False
                            if   final_tier["label"]=="Perfect Circle":   particles_list=[Particle(CX,CY) for _ in range(160)]
                            elif final_tier["label"]=="Legendary Circle": pulse_rings=[PulseRing(CX,CY,d) for d in (0,.25,.5)]
                            elif final_tier["label"] in ("Master Circle","Great Circle"): glow_active=True
                            elif final_tier["label"]=="Keep Practicing":  red_blink=True
            if peace_count>=GESTURE_CONFIRM:
                show_invalid("CANCELLED"); go_idle()

        elif state==STATE_SCORING:
            prog=min((t-score_anim_start)/1.4,1.0)
            score_anim=final_score*(1-(1-prog)**3)
            if prog>=1.0 and not score_saved:
                show_tier=True; score_saved=True; save_score(player_name,final_score)
            particles_list=[p for p in particles_list if p.life>0]
            for p in particles_list: p.update()
            for r in pulse_rings: r.update(dt)
            pulse_rings=[r for r in pulse_rings if r.life>0 or r.delay>0]
            if peace_count>=GESTURE_CONFIRM:    go_idle()
            if thumb_dn_count>=GESTURE_CONFIRM:
                player_name=run_name_screen(cam_surf); state=STATE_IDLE

        # ── Render ──
        screen.fill(BG_COLOR)
        if cam_surf:
            bg=pygame.transform.scale(cam_surf,(W,H)); bg.set_alpha(200); screen.blit(bg,(0,0))
        ov=pygame.Surface((W,H),pygame.SRCALPHA); ov.fill((5,10,14,115)); screen.blit(ov,(0,0))

        draw_corners(screen)
        draw_anchor_dot(screen,t)

        if state==STATE_IDLE:
            draw_idle_prompt(screen,t)
        elif state==STATE_DRAWING:
            if points:
                display_pts = chaikin_smooth(points, iterations=2)
                draw_path(screen, display_pts, GREEN)
            draw_closure_guide(screen,points,tip,has_moved_away)
        elif state==STATE_SCORING:
            if effect_circle and len(effect_circle)>=2:
                col=final_tier["color"] if final_tier else GREEN
                draw_path(screen,effect_circle,col,3)
            for p in particles_list: p.draw(screen)
            for r in pulse_rings: r.draw(screen)
            if red_blink and int(t*10)%2==0:
                rb=pygame.Surface((W,H),pygame.SRCALPHA); rb.fill((255,61,90,30)); screen.blit(rb,(0,0))
            draw_score_overlay(screen,score_anim,final_tier or TIERS[-1],show_tier,t,
                               score_circ if show_tier else None,
                               score_cent if show_tier else None)

        draw_cursor(screen,tip,state==STATE_DRAWING)
        draw_leaderboard(screen,player_name)
        draw_hud(screen,state,hand_lm is not None,player_name,len(points),fps)

        if invalid_msg_timer>0:
            invalid_msg_timer-=dt
            invalid_msg_alpha=int(min(255,invalid_msg_timer/2.2*255))
            draw_invalid_msg(screen,invalid_msg_text,invalid_msg_alpha)

        pygame.display.flip()

    cap.release(); pygame.quit()

if __name__=="__main__":
    main()