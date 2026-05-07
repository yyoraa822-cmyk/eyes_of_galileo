"""World -> pixel projection for goal markers.

This is the missing glue between `materialize_goal_pool()` (which
generates a `target_y` in physical units) and `draw_goal_marker()`
(which expects a pixel coordinate).

Design contract
---------------
- The projector is **heuristic, not physically exact**. Per-scene
  projection-from-physics-to-matplotlib-axes-to-pixels is a rabbit hole
  we deliberately skip; instead we map `target_y` -> a fraction
  `f in [0, 1]` over the training pool's observable range, and place the
  green marker on a per-slug "intuitive" image path (bottom row for
  cart-style scenes, right edge for hoop, vertical track for piston).
- The agent gets quantitative feedback ONLY from the post-attempt
  `overlay_position` (red X + arrow + ruler ticks) -- the green marker
  just communicates "where the goal lives" approximately.
- `project_goal()` returns a structured payload regardless of overlay
  type so the GCD runner has one uniform interface for every scene.

Scalar scenes (heat / decay / blackbody / cooling) get pixel coords
ONLY for compositional consistency (so `draw_goal_marker` always has
something to render); the truth-bearing visualisation for those is
`overlay_scalar`'s side bars, fed from the returned bounds.

Phase scenes (orbital) return `deg` for `overlay_phase` and a default
pixel anchor for any optional in-scene marker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import VIEW_GOAL_META, Goal


@dataclass
class GoalProjection:
    """Payload returned by project_goal()."""
    overlay_type: str              # "position" | "scalar" | "phase"
    target_xy_px: tuple[int, int]  # always populated; for scalar/phase it
                                   # is just the marker draw point
    actual_xy_px: Optional[tuple[int, int]] = None
    # for scalar overlays: bounds for bar visualisation
    scalar_bounds: Optional[tuple[float, float]] = None
    # for phase overlays: target/actual angles (deg)
    target_deg: Optional[float] = None
    actual_deg: Optional[float] = None


# ---------------------------------------------------------------------------
# Per-slug marker placement (W, H -> tuple[int,int])
# ---------------------------------------------------------------------------
# Each callable receives `f` (the goal's position in [0, 1] over the
# observable range) and the image (W, H), and returns the pixel anchor
# for the green marker. The mapping is empirically chosen to look
# correct on each scenario's matplotlib canvas; tweak per-slug as needed.

def _cart_on_bottom(f: float, W: int, H: int) -> tuple[int, int]:
    """Cart marker: lands somewhere along the bottom row."""
    return (int(W * (0.15 + 0.7 * f)), int(H * 0.88))


def _vertical_track(f: float, W: int, H: int) -> tuple[int, int]:
    """Falling-object marker on a vertical track (mid-x)."""
    return (int(W * 0.50), int(H * (0.18 + 0.70 * f)))


def _bell_at_swing_apex(f: float, W: int, H: int) -> tuple[int, int]:
    """Pendulum bell: arc from middle to right as f grows."""
    return (int(W * (0.40 + 0.40 * f)), int(H * 0.55))


def _hoop_landing(f: float, W: int, H: int) -> tuple[int, int]:
    """Projectile hoop: distance left-to-right at low altitude."""
    return (int(W * (0.10 + 0.85 * f)), int(H * 0.78))


def _piston_line(f: float, W: int, H: int) -> tuple[int, int]:
    """Horizontal piston-line at the goal column length (vertical move)."""
    return (int(W * 0.55), int(H * (0.85 - 0.65 * f)))


def _underwater_target(f: float, W: int, H: int) -> tuple[int, int]:
    """Submerged ring at the refracted-ray endpoint."""
    return (int(W * (0.30 + 0.55 * f)), int(H * 0.78))


def _generic_mark(f: float, W: int, H: int) -> tuple[int, int]:
    """Right-side scalar/phase marker — overlay_scalar/phase do the work."""
    return (int(W * 0.85), int(H * (0.20 + 0.55 * f)))


_MARKER_PLACERS: dict[str, callable] = {
    "s1_freefall":    _vertical_track,
    "s2_mass":        _vertical_track,
    "s3_pendulum":    _bell_at_swing_apex,
    "s4_ramp":        _cart_on_bottom,
    "s6_launch":      _hoop_landing,
    "s9_spring":      _piston_line,
    "s17_refraction": _underwater_target,
    "s18_boyle":      _piston_line,
    "s19_coulomb":    _cart_on_bottom,
    "s20_hooke":      _piston_line,
    # scalar / phase scenes — marker is incidental
    "s8_heat":        _generic_mark,
    "s16_decay":      _generic_mark,
    "s21_weber":      _generic_mark,
    "s22_cooling":    _generic_mark,
    "s10_circular":   _generic_mark,
}


# ---------------------------------------------------------------------------
# Observable-range probe
# ---------------------------------------------------------------------------

_SLUG_MAP = {
    "s1_freefall":    "freefall",
    "s2_mass":        "freefall",
    "s3_pendulum":    "pendulum",
    "s4_ramp":        "freefall",
    "s6_launch":      "projectile",
    "s8_heat":        "heat",
    "s9_spring":      "spring",
    "s10_circular":   "orbital",
    "s16_decay":      "decay",
    "s17_refraction": "refraction",
    "s18_boyle":      "boyle",
    "s19_coulomb":    "coulomb",
    "s20_hooke":      "spring",
    "s21_weber":      "blackbody",
    "s22_cooling":    "cooling",
}

# Cache observable-range probes; scenarios are deterministic so this
# is safe per-process.
_RANGE_CACHE: dict[str, tuple[float, float]] = {}


def _observable_range(view_id: str) -> Optional[tuple[float, float]]:
    """Return (y_min, y_max) over the *extended* control range covering
    both the training pool and the GCD goal pool.

    We sample observables on a grid over `[0.5*u_min, 2.5*u_max]` (same
    bounds the solver uses for hold-out goals). Without this widening
    the projector saturates `f=1.0` for nearly every hold-out goal --
    target and actual land at the same pixel and the agent loses its
    visual residual.
    """
    if view_id in _RANGE_CACHE:
        return _RANGE_CACHE[view_id]
    from galileo.scenarios import get_scenario

    slug = _SLUG_MAP.get(view_id)
    if slug is None:
        return None
    try:
        sc = get_scenario(slug)
    except Exception:
        return None
    pool = list(sc.default_controls)
    if not pool:
        return None
    u_min = max(0.5 * float(min(pool)), 1e-6)
    u_max = 2.5 * float(max(pool))
    ys: list[float] = []
    n_grid = 12
    for i in range(n_grid + 1):
        u = u_min + (u_max - u_min) * i / n_grid
        try:
            ys.append(float(sc.get_observable(u)))
        except Exception:
            continue
    if not ys:
        return None
    y_lo, y_hi = float(min(ys)), float(max(ys))
    if y_lo == y_hi:
        return (y_lo - 1.0, y_hi + 1.0)
    _RANGE_CACHE[view_id] = (y_lo, y_hi)
    return (y_lo, y_hi)


def _normalised_f(view_id: str, target_y: float) -> float:
    rng = _observable_range(view_id)
    if rng is None:
        return 0.5
    y_lo, y_hi = rng
    f = (float(target_y) - y_lo) / max(y_hi - y_lo, 1e-9)
    return max(0.0, min(1.0, f))


# ---------------------------------------------------------------------------
# Phase-angle conversion (orbital scene)
# ---------------------------------------------------------------------------

def _phase_target_deg(view_id: str, target_y: float) -> float:
    """Convert observable to degrees on the dial.

    For `s10_circular` (orbital), `target_y` is a period in years. Map
    it to degrees by `(period_y / max_pool_period_y) * 360`. Clamped
    into [0, 359.99] so the dial wraps cleanly.
    """
    rng = _observable_range(view_id)
    if rng is None:
        return 0.0
    _, y_hi = rng
    deg = (float(target_y) / max(y_hi, 1e-9)) * 360.0
    return float(deg % 360.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def project_goal(view_id: str, goal: Goal,
                 image_size: tuple[int, int],
                 actual_y: Optional[float] = None) -> GoalProjection:
    """Compute the pixel/scalar/phase visualisation payload for `goal`.

    Parameters
    ----------
    view_id : id used in SCENE_VIEWS / VIEW_GOAL_META.
    goal : Goal whose target_y we are visualising.
    image_size : (W, H) of the rendered DSL frame.
    actual_y : if provided, also computes the agent's observed pixel /
        scalar / phase representation so the GCD runner can hand
        target+actual to overlay_position / overlay_scalar / overlay_phase
        in a single shot.
    """
    W, H = int(image_size[0]), int(image_size[1])
    placer = _MARKER_PLACERS.get(view_id, _generic_mark)
    f_target = _normalised_f(view_id, goal.target_y)
    target_xy = placer(f_target, W, H)

    actual_xy = None
    if actual_y is not None:
        f_actual = _normalised_f(view_id, float(actual_y))
        actual_xy = placer(f_actual, W, H)

    if goal.overlay_type == "scalar":
        rng = _observable_range(view_id)
        bounds = rng if rng is not None else (0.0, max(1.0, goal.target_y * 2.0))
        return GoalProjection(
            overlay_type="scalar",
            target_xy_px=target_xy,
            actual_xy_px=actual_xy,
            scalar_bounds=bounds,
        )
    if goal.overlay_type == "phase":
        target_deg = _phase_target_deg(view_id, goal.target_y)
        actual_deg = (_phase_target_deg(view_id, float(actual_y))
                      if actual_y is not None else None)
        return GoalProjection(
            overlay_type="phase",
            target_xy_px=target_xy,
            actual_xy_px=actual_xy,
            target_deg=target_deg,
            actual_deg=actual_deg,
        )
    # default: position
    return GoalProjection(
        overlay_type="position",
        target_xy_px=target_xy,
        actual_xy_px=actual_xy,
    )


def goal_kind_for(view_id: str) -> str:
    """Return the GoalKind to pass to draw_goal_marker for `view_id`.

    Falls back to 'mark' if not declared in VIEW_GOAL_META.goal_kind."""
    meta = VIEW_GOAL_META.get(view_id, {})
    kind = meta.get("goal_kind")
    return kind if kind is not None else "mark"
