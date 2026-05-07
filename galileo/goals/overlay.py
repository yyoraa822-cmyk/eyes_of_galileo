"""Visual residual overlays for GCD.

Three overlay primitives, all PIL-only (post-render), no DSL coupling:

  - overlay_position(img, target_xy, actual_xy, passed, ruler=True)
  - overlay_scalar  (img, target_v, actual_v, vmin, vmax, passed)
  - overlay_phase   (img, target_deg, actual_deg, passed, center=None, r=None)

Design rules (matches GCD spec):
  * PASS / FAIL badge (top-right) is the ONLY textual signal.
  * NO numeric residual is rendered into the image.
  * Residual magnitude is only conveyed visually — arrow length, bar gap,
    angular wedge — paired with on-image tick marks so the agent can
    "measure" by counting ticks.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

BADGE_PASS_FILL = (0, 160, 80)
BADGE_FAIL_FILL = (200, 50, 50)
BADGE_TEXT      = (255, 255, 255)

TARGET_COLOR = (60, 220, 90)
ACTUAL_COLOR = (240, 70, 70)
ARROW_COLOR  = (255, 255, 255)
TICK_COLOR   = (140, 140, 160)

PAD = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_font(size: int = 14):
    for p in (
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_pass_badge(img: Image.Image, passed: bool) -> Image.Image:
    """Top-right PASS/FAIL badge. Mutates `img` (also returns for chaining)."""
    d = ImageDraw.Draw(img)
    font = _get_font(16)
    text = "PASS" if passed else "FAIL"
    fill = BADGE_PASS_FILL if passed else BADGE_FAIL_FILL
    w, h = img.size
    bw, bh = 70, 26
    x0 = w - bw - PAD
    y0 = PAD
    d.rectangle([(x0, y0), (x0 + bw, y0 + bh)], fill=fill)
    tx = x0 + bw // 2
    ty = y0 + bh // 2
    try:
        d.text((tx, ty), text, fill=BADGE_TEXT, font=font, anchor="mm")
    except TypeError:
        # Older PIL: no anchor support
        d.text((tx - 18, ty - 8), text, fill=BADGE_TEXT, font=font)
    return img


def _draw_arrow(d: ImageDraw.ImageDraw, p0: Tuple[float, float],
                p1: Tuple[float, float], color=ARROW_COLOR, width: int = 2,
                head: int = 6) -> None:
    d.line([p0, p1], fill=color, width=width)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    tip = p1
    base_x = p1[0] - ux * head
    base_y = p1[1] - uy * head
    left = (base_x + px * head * 0.55, base_y + py * head * 0.55)
    right = (base_x - px * head * 0.55, base_y - py * head * 0.55)
    d.polygon([tip, left, right], fill=color)


# ---------------------------------------------------------------------------
# 1. POSITION overlay
# ---------------------------------------------------------------------------

def overlay_position(img: Image.Image,
                     target_xy: Tuple[float, float],
                     actual_xy: Tuple[float, float],
                     passed: bool,
                     ruler: bool = True,
                     ruler_y: Optional[float] = None) -> Image.Image:
    """Overlay green target ring + red X + arrow between them.

    target_xy/actual_xy: pixel coordinates in the input image.
    ruler: if True, draws a horizontal tick row at `ruler_y` (default: image
      height - 14) to give the agent a built-in length scale.
    """
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    if ruler:
        ry = ruler_y if ruler_y is not None else h - 14
        d.line([(PAD, ry), (w - PAD, ry)], fill=TICK_COLOR, width=1)
        for x in range(PAD, w - PAD + 1, 24):
            d.line([(x, ry - 3), (x, ry + 3)], fill=TICK_COLOR, width=1)

    tx, ty = target_xy
    ax, ay = actual_xy
    r = 8
    d.ellipse([(tx - r, ty - r), (tx + r, ty + r)],
              outline=TARGET_COLOR, width=2)
    d.line([(ax - r, ay - r), (ax + r, ay + r)], fill=ACTUAL_COLOR, width=2)
    d.line([(ax + r, ay - r), (ax - r, ay + r)], fill=ACTUAL_COLOR, width=2)

    if not passed:
        # Bidirectional arrow between actual and target along the line.
        dx, dy = tx - ax, ty - ay
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        a_start = (ax + ux * (r + 2), ay + uy * (r + 2))
        a_end = (tx - ux * (r + 2), ty - uy * (r + 2))
        _draw_arrow(d, a_start, a_end)
        _draw_arrow(d, a_end, a_start)

    draw_pass_badge(img, passed)
    return img


# ---------------------------------------------------------------------------
# 2. SCALAR overlay
# ---------------------------------------------------------------------------

def overlay_scalar(img: Image.Image,
                   target_v: float, actual_v: float,
                   vmin: float, vmax: float,
                   passed: bool,
                   label_target: str = "target",
                   label_actual: str = "actual") -> Image.Image:
    """Append two side-by-side bars (actual red / target green) and a delta arrow.

    The bars are drawn into a margin on the right side of the image.
    No numeric value is shown; only the bar heights + tick marks.
    """
    img = img.convert("RGB").copy()
    w, h = img.size
    panel_w = 130
    out = Image.new("RGB", (w + panel_w, h), color=(20, 20, 24))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    font = _get_font(11)

    bar_w = 34
    bar_top = 30
    bar_bot = h - 30
    bar_h = bar_bot - bar_top
    x_actual = w + 18
    x_target = w + 18 + bar_w + 22

    def _val_to_y(v: float) -> int:
        v = max(vmin, min(vmax, v))
        f = (v - vmin) / max(vmax - vmin, 1e-9)
        return int(bar_bot - f * bar_h)

    for i in range(0, 11):
        ty = bar_top + int(i / 10 * bar_h)
        d.line([(x_actual - 4, ty), (x_actual, ty)], fill=TICK_COLOR, width=1)
        d.line([(x_target + bar_w, ty), (x_target + bar_w + 4, ty)],
               fill=TICK_COLOR, width=1)

    d.rectangle([(x_actual, bar_top), (x_actual + bar_w, bar_bot)],
                outline=(150, 150, 160), width=1)
    d.rectangle([(x_target, bar_top), (x_target + bar_w, bar_bot)],
                outline=(150, 150, 160), width=1)

    a_y = _val_to_y(actual_v)
    t_y = _val_to_y(target_v)
    d.rectangle([(x_actual, a_y), (x_actual + bar_w, bar_bot)],
                fill=ACTUAL_COLOR)
    d.rectangle([(x_target, t_y), (x_target + bar_w, bar_bot)],
                fill=TARGET_COLOR)

    d.text((x_actual + bar_w / 2 - 16, bar_bot + 4),
           label_actual, fill=ACTUAL_COLOR, font=font)
    d.text((x_target + bar_w / 2 - 16, bar_bot + 4),
           label_target, fill=TARGET_COLOR, font=font)

    if not passed:
        p0 = (x_actual + bar_w, a_y)
        p1 = (x_target, t_y)
        _draw_arrow(d, p0, p1)

    draw_pass_badge(out, passed)
    return out


# ---------------------------------------------------------------------------
# 3. PHASE overlay
# ---------------------------------------------------------------------------

def overlay_phase(img: Image.Image,
                  target_deg: float, actual_deg: float,
                  passed: bool,
                  center: Optional[Tuple[int, int]] = None,
                  radius: Optional[int] = None) -> Image.Image:
    """Polar dial with a green target marker and a red actual marker.

    Angles are in degrees, 0 = +x axis, CCW positive. If center/radius are
    None, dial is drawn in top-right corner.
    """
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    if center is None:
        center = (w - 70, 70)
    if radius is None:
        radius = 42
    cx, cy = center

    d.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)],
              outline=TICK_COLOR, width=1)
    for a_deg in range(0, 360, 30):
        a = math.radians(a_deg)
        x1 = cx + (radius - 3) * math.cos(a)
        y1 = cy - (radius - 3) * math.sin(a)
        x2 = cx + (radius + 3) * math.cos(a)
        y2 = cy - (radius + 3) * math.sin(a)
        d.line([(x1, y1), (x2, y2)], fill=TICK_COLOR, width=1)
    d.ellipse([(cx - 3, cy - 3), (cx + 3, cy + 3)], fill=(140, 140, 140))

    def _pt(angle_deg: float) -> Tuple[float, float]:
        a = math.radians(angle_deg)
        return (cx + radius * math.cos(a), cy - radius * math.sin(a))

    tx, ty = _pt(target_deg)
    ax, ay = _pt(actual_deg)
    r = 6
    d.ellipse([(tx - r, ty - r), (tx + r, ty + r)], fill=TARGET_COLOR)
    d.ellipse([(ax - r, ay - r), (ax + r, ay + r)], fill=ACTUAL_COLOR)

    if not passed:
        # Draw arc from target to actual along the shorter direction.
        diff = (actual_deg - target_deg + 540) % 360 - 180
        steps = max(8, int(abs(diff)))
        prev = (tx, ty)
        for i in range(1, steps + 1):
            a = target_deg + diff * i / steps
            cur = _pt(a)
            d.line([prev, cur], fill=ARROW_COLOR, width=2)
            prev = cur

    draw_pass_badge(img, passed)
    return img
