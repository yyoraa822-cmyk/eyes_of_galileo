"""Oracle inverse-solver: given target_y, return the control u* that hits it.

This is used ONLY for:
  - generating the hold-out goal pool (knowing y* given u*)
  - evaluator-side ground-truth (judging whether agent's choice passed)

Agents NEVER call this. It uses the true CF physics directly.

Implementation strategy (kept minimal): for each scenario slug we either
  (a) close-form invert when possible (e.g. freefall, projectile), or
  (b) bracketed bisection on `scenario.get_observable(u)` otherwise.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional


def _bisect(fn, target: float, lo: float, hi: float,
            tol: float = 1e-4, max_iter: int = 80) -> Optional[float]:
    f_lo = fn(lo) - target
    f_hi = fn(hi) - target
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = fn(mid) - target
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def solve_for_control(scenario, target_y: float,
                      u_lo: float, u_hi: float,
                      monotone: bool = True) -> Optional[float]:
    """Return u in [u_lo, u_hi] such that scenario.get_observable(u) ≈ target_y.

    `monotone=True` enables the bisection fast-path; if the underlying
    observable is non-monotone in u, set False and we will do a coarse grid
    search before bisecting around the best bracket.
    """
    fn = scenario.get_observable
    if monotone:
        return _bisect(fn, target_y, u_lo, u_hi)

    grid_n = 32
    best_lo, best_hi = None, None
    prev_u = u_lo
    prev_v = fn(u_lo) - target_y
    for i in range(1, grid_n + 1):
        u = u_lo + (u_hi - u_lo) * i / grid_n
        v = fn(u) - target_y
        if prev_v == 0:
            return prev_u
        if v == 0:
            return u
        if prev_v * v < 0:
            best_lo, best_hi = prev_u, u
            break
        prev_u, prev_v = u, v
    if best_lo is None:
        return None
    return _bisect(fn, target_y, best_lo, best_hi)


# ---------------------------------------------------------------------------
# Goal-pool materialisation
# ---------------------------------------------------------------------------

# Per-view goal materialisation policy.
#
# The training pool (scenario.default_controls) defines a control range
# [u_min, u_max]. Hold-out goals must lie OUTSIDE that range so the agent
# can't memorise them. We pick 5 goal control values stratified as:
#
#   2 easy   — just past the boundary (~1.10× / 0.90× of the boundary)
#   2 medium — well past the boundary (~1.50× / 0.66×)
#   1 hard   — far past the boundary (~2.00× / 0.50×)
#
# Each goal's `target_y` is then the oracle observable at that u*. We
# also widen the agent-visible `control_range` to [0.5*u_min, 2.5*u_max]
# so the agent has search room around the held-out values.

_DIFFICULTY_FACTORS: list[tuple[str, float]] = [
    ("easy",   1.10),   # just above max
    ("easy",   0.90),   # just below min
    ("medium", 1.50),
    ("medium", 0.66),
    ("hard",   2.00),
]


def _solve_one(scenario, u_target: float, u_lo: float, u_hi: float
               ) -> tuple[Optional[float], float]:
    """Compute observable at u_target and return (u_target_clamped, y).

    We do NOT need to invert — the goal is "y at this u_target". The
    inverter exists for the agent-side judge, which compares y_hit
    (oracle's observable at the agent's chosen u) against y_target.
    """
    u_clamped = max(u_lo, min(u_hi, u_target))
    try:
        y = float(scenario.get_observable(u_clamped))
    except Exception:
        return None, 0.0
    return u_clamped, y


def materialize_goal_pool(view_ids: Optional[list[str]] = None,
                          cache_path: Optional[Path] = None,
                          force: bool = False) -> dict[str, list[dict]]:
    """Populate the global GOAL_POOL with 5 hold-out goals per view.

    Returns the materialised dict (also writes to `_pool_cache.json` so
    successive runs are deterministic). If `cache_path` already exists
    and `force=False`, returns the cached payload without recomputation.
    """
    from galileo.scenarios import (
        DEFAULT_COUNTERFACTUALS,
        DEFAULT_SCENARIO_KWARGS,
        get_scenario,
    )
    from .config import GOAL_POOL, VIEW_GOAL_META, Goal

    if cache_path is None:
        cache_path = Path(__file__).resolve().parent / "_pool_cache.json"

    if cache_path.exists() and not force:
        try:
            data = json.loads(cache_path.read_text())
            for view_id, goal_dicts in data.items():
                GOAL_POOL[view_id] = [Goal(**g) for g in goal_dicts]
            return data
        except Exception:
            # corrupt cache — recompute
            pass

    # Map each view_id back to its underlying scenario slug. Sourced
    # from run_dsl_unified.SCENE_VIEWS but kept inline here so goals/
    # stays decoupled from the runner. Keep this in sync with
    # SCENE_VIEWS' (id, slug) pairs.
    VIEW_TO_SLUG: dict[str, str] = {
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

    out: dict[str, list[dict]] = {}
    selected = view_ids or list(VIEW_TO_SLUG.keys())

    for vid in selected:
        slug = VIEW_TO_SLUG.get(vid)
        if slug is None:
            continue
        try:
            sc = get_scenario(slug)
        except Exception:
            continue
        meta = sc.meta
        pool = list(sc.default_controls)
        if not pool:
            continue
        u_min, u_max = float(min(pool)), float(max(pool))
        u_search_lo = max(0.5 * u_min, 1e-6)
        u_search_hi = 2.5 * u_max

        view_meta = VIEW_GOAL_META.get(vid, {})
        overlay_kind = view_meta.get("overlay", "scalar")
        units = view_meta.get("units", "")
        name = view_meta.get("name", vid)

        goals_for_view: list[dict] = []
        for diff, factor in _DIFFICULTY_FACTORS:
            # Above-max goals use the upper-end factor; below-min goals
            # use the lower-end factor (1/factor).
            if factor >= 1.0:
                u_goal = u_max * factor
            else:
                u_goal = u_min * factor
            u_clamped, y = _solve_one(sc, u_goal, u_search_lo, u_search_hi)
            if u_clamped is None:
                continue
            tol = max(abs(y) * 0.05, 1e-3)
            g = Goal(
                view_id=vid,
                slug=slug,
                name=name,
                overlay_type=overlay_kind,  # type: ignore[arg-type]
                target_y=round(y, 6),
                tolerance=round(tol, 6),
                control_range=(round(u_search_lo, 6),
                               round(u_search_hi, 6)),
                difficulty=diff,  # type: ignore[arg-type]
                units=units,
                description=(f"{name}: drive {meta.observable_label} to "
                             f"the value rendered as the green target."),
            )
            goals_for_view.append(g.__dict__)

        out[vid] = goals_for_view
        GOAL_POOL[vid] = [Goal(**gd) for gd in goals_for_view]

    cache_path.write_text(json.dumps(out, indent=2))
    return out
