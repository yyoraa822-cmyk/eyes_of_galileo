"""Physics-bound anchor finder for GCD goal markers.

Replaces the heuristic ``projector._MARKER_PLACERS`` for strobe-style
mechanical scenes (s1-s4): instead of guessing a fractional pixel
location, we render the scene twice (at ``u_star`` and ``u_actual``)
and locate the FINAL strobe centroid in each image. Those pixel
coordinates are physics-bound — the goal marker sits exactly where
the ball would have been, and the red X sits exactly where it
actually was.

For matplotlib scenario_backend scenes the per-slug marker shapes are
already drawn into the canvas by the scenario itself; for those we
fall back to ``project_goal()``'s heuristic.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image


# Strobe yellow used by compiler.py: (255, 235, 60). We accept a
# generous Manhattan-distance tolerance so anti-aliased pixels and
# slight gamma differences still match.
_STROBE_RGB = np.array([255, 235, 60], dtype=np.int16)


def _yellow_mask(arr: np.ndarray, tol: int = 50) -> np.ndarray:
    diff = np.abs(arr.astype(np.int16) - _STROBE_RGB)
    return (diff.sum(axis=-1) < tol)


def find_final_strobe(
    img: Image.Image,
    *,
    direction: str = "bottom",
) -> Optional[tuple[int, int]]:
    """Locate the final-strobe centroid in a rendered scene image.

    `direction` controls which extreme strobe is "final":
      - "bottom": lowest-y yellow cluster (freefall, spring)
      - "right":  largest-x yellow cluster (projectile, ramp)
      - "apex":   farthest-from-vertical-axis (pendulum)

    Returns (x, y) in image pixels, or None if no strobe pixels
    were detected (pixel-search failed)."""
    arr = np.asarray(img.convert("RGB"))
    mask = _yellow_mask(arr)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    if direction == "bottom":
        # Pick the cluster with the largest y (lowest on screen). Use
        # the cluster's centroid, not a single pixel, to handle the
        # finite-radius dot.
        max_y = int(ys.max())
        sel = (ys >= max_y - 6)
    elif direction == "right":
        max_x = int(xs.max())
        sel = (xs >= max_x - 6)
    elif direction == "apex":
        # Furthest from image-x-centre; ties broken by lowest y.
        cx = arr.shape[1] / 2
        dist = np.abs(xs - cx)
        max_d = int(dist.max())
        sel = (dist >= max_d - 6)
    else:
        raise ValueError(f"unknown direction: {direction}")

    return (int(xs[sel].mean()), int(ys[sel].mean()))


# Per-view direction for finding the final strobe.
_DIRECTION: dict[str, str] = {
    "s1_freefall": "bottom",
    "s2_mass":     "bottom",
    "s3_pendulum": "apex",
    "s4_ramp":     "bottom",
    "s6_launch":   "right",
    "s9_spring":   "bottom",
    "s20_hooke":   "bottom",
}


def physics_anchor_for(view_id: str, img: Image.Image
                       ) -> Optional[tuple[int, int]]:
    direction = _DIRECTION.get(view_id)
    if direction is None:
        return None
    return find_final_strobe(img, direction=direction)
