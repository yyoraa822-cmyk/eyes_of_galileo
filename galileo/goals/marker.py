"""Per-scene goal-target markers — pure PIL post-render primitives.

These are explicitly NOT DSL entities. They sit at the same abstraction
level as `goals/overlay.py`: take an already-rendered PIL image plus a
target pixel coordinate, draw the green goal-object on top, return the
mutated image. Composing a full GCD render is:

    img = compile_and_render(yaml, ...)            # DSL pipeline
    img = draw_goal_marker(img, kind, xy_px)       # green target
    img = overlay_position(img, target_xy=xy_px,
                           actual_xy=...,
                           passed=...)             # red X + arrow + badge

This keeps the DSL layer ignorant of GCD; goals/ owns every pixel that
encodes a goal.
"""
from __future__ import annotations

from typing import Literal, Tuple

from PIL import Image, ImageDraw

GoalKind = Literal["cart", "bell", "hoop", "underwater_target",
                   "piston_line", "mark"]

GOAL_COLOR = (60, 220, 90)        # bright green
GOAL_OUTLINE = (10, 80, 30)
GOAL_FILL_LIGHT = (140, 240, 170)


def _draw_cart(d: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
    """Small green cart sitting on the ground; cx/cy is the cart's
    landing-target reference point (centre of the basket opening)."""
    body_w, body_h = 38, 18
    wheel_r = 5
    bx0 = cx - body_w // 2
    by0 = cy - body_h
    d.rectangle([(bx0, by0), (bx0 + body_w, by0 + body_h)],
                fill=GOAL_COLOR, outline=GOAL_OUTLINE, width=2)
    # basket opening rim
    d.line([(bx0 + 4, by0), (bx0 + body_w - 4, by0)],
           fill=GOAL_OUTLINE, width=2)
    # wheels
    for wx in (bx0 + 8, bx0 + body_w - 8):
        d.ellipse([(wx - wheel_r, by0 + body_h - wheel_r),
                   (wx + wheel_r, by0 + body_h + wheel_r)],
                  fill=(40, 40, 40), outline=GOAL_OUTLINE, width=1)


def _draw_bell(d: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
    """Small green bell whose striker mouth is at (cx, cy)."""
    h = 22
    w = 18
    body = [
        (cx - w // 2, cy - h),
        (cx + w // 2, cy - h),
        (cx + w // 2 + 3, cy),
        (cx - w // 2 - 3, cy),
    ]
    d.polygon(body, fill=GOAL_COLOR, outline=GOAL_OUTLINE)
    d.ellipse([(cx - 2, cy - 3), (cx + 2, cy + 1)],
              fill=GOAL_OUTLINE)
    # hanger
    d.line([(cx, cy - h), (cx, cy - h - 6)], fill=GOAL_OUTLINE, width=2)


def _draw_hoop(d: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
    """A vertical hoop ring; cx/cy is the centre of the hoop opening."""
    r = 14
    d.ellipse([(cx - r, cy - r // 2), (cx + r, cy + r // 2)],
              outline=GOAL_COLOR, width=3)
    d.ellipse([(cx - r + 2, cy - r // 2 + 2),
               (cx + r - 2, cy + r // 2 - 2)],
              outline=GOAL_OUTLINE, width=1)


def _draw_underwater_target(d: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
    """Concentric green rings — submerged target."""
    for i, rr in enumerate((14, 9, 4)):
        col = GOAL_COLOR if i % 2 == 0 else GOAL_FILL_LIGHT
        d.ellipse([(cx - rr, cy - rr), (cx + rr, cy + rr)],
                  outline=col, width=2)
    d.ellipse([(cx - 2, cy - 2), (cx + 2, cy + 2)], fill=GOAL_OUTLINE)


def _draw_piston_line(d: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
    """A horizontal green tick spanning the gas column at the goal mark."""
    half = 24
    d.line([(cx - half, cy), (cx + half, cy)],
           fill=GOAL_COLOR, width=4)
    # inward triangles to make it unambiguous
    d.polygon([(cx - half, cy - 5), (cx - half + 6, cy),
               (cx - half, cy + 5)], fill=GOAL_COLOR)
    d.polygon([(cx + half, cy - 5), (cx + half - 6, cy),
               (cx + half, cy + 5)], fill=GOAL_COLOR)


def _draw_mark(d: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
    """Generic green crosshair-on-a-disk mark."""
    r = 10
    d.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
              fill=GOAL_FILL_LIGHT, outline=GOAL_COLOR, width=2)
    d.line([(cx - r, cy), (cx + r, cy)], fill=GOAL_OUTLINE, width=1)
    d.line([(cx, cy - r), (cx, cy + r)], fill=GOAL_OUTLINE, width=1)


_RENDERERS = {
    "cart": _draw_cart,
    "bell": _draw_bell,
    "hoop": _draw_hoop,
    "underwater_target": _draw_underwater_target,
    "piston_line": _draw_piston_line,
    "mark": _draw_mark,
}


def draw_goal_marker(img: Image.Image, kind: GoalKind,
                     target_xy_px: Tuple[int, int]) -> Image.Image:
    """Draw the green goal-object onto `img` at pixel (x, y).

    `kind` selects the visual style. `target_xy_px` is the marker's
    semantic anchor (cart-mouth centre / bell mouth / hoop centre /
    target ring centre / piston-line midpoint). Returns a new image;
    the input is not mutated.
    """
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    fn = _RENDERERS.get(kind)
    if fn is None:
        # unknown kind — degrade to generic mark instead of failing.
        fn = _draw_mark
    cx, cy = int(target_xy_px[0]), int(target_xy_px[1])
    w, h = out.size
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))
    fn(d, cx, cy)
    return out
