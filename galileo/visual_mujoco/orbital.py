"""MuJoCo visualisation for s10_circular (Kepler's third law).

Tycho Brahe + Kepler (1609): a planet orbits a central star at radius
a; the agent picks a (semi-major axis) and reads the period T,
discovering T² ∝ a^alpha (textbook alpha = 3).

Apparatus mapping: top-down "armillary" view of a star at the origin
with a planet on a circular orbit. The orbit radius corresponds to
the control value. As the animation plays, the planet sweeps along
the orbit; faster (T smaller) → it covers more arc per frame, so the
period becomes visible from how much arc gets traced in a fixed
window. A faint dashed circle shows the full orbit; a glowing trail
marks the swept arc; the live planet sits at the current angular
position.
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from PIL import Image

from ._base import (
    make_scene_mjcf,
    render_frames_with_state,
    set_mocap_pos,
)

N_RING_DOTS = 64        # dashed full-orbit ring
N_TRAIL = 24            # bright swept arc

PLANET_R = 0.14
STAR_R = 0.30


def _build_mjcf(a: float) -> str:
    extra_assets = """
  <asset>
    <material name="space" rgba="0.02 0.02 0.05 1"/>
    <material name="star" rgba="1.00 0.85 0.30 1" emission="0.9"/>
    <material name="planet" rgba="0.30 0.55 0.95 1" specular="0.40"
              shininess="0.5" emission="0.10"/>
    <material name="ring_dot" rgba="0.30 0.45 0.65 0.55"/>
    <material name="trail" rgba="0.55 0.85 1.00 0.9" emission="0.55"/>
    <material name="start" rgba="0.30 0.95 0.40 1" emission="0.6"/>
  </asset>
"""
    parts: list[str] = []

    # Star at origin
    parts.append(
        f'    <body name="star" pos="0 0 0.30">\n'
        f'      <geom type="sphere" size="{STAR_R:.3f}" material="star"/>\n'
        f'    </body>'
    )

    # Faint dashed orbit ring (64 small dots arranged in a circle).
    for i in range(N_RING_DOTS):
        theta = 2 * np.pi * i / N_RING_DOTS
        x = a * np.cos(theta)
        y = a * np.sin(theta)
        parts.append(
            f'    <body name="ring_{i}" pos="{x:.3f} {y:.3f} 0.32">\n'
            f'      <geom type="sphere" size="0.025" material="ring_dot"/>\n'
            f'    </body>'
        )

    # Trail dots (mocap-controlled), revealed as the orbit progresses.
    for k in range(N_TRAIL):
        parts.append(
            f'    <body name="trail_{k}" mocap="true" pos="0 0 -10">\n'
            f'      <geom name="trail_geom_{k}" type="sphere" '
            f'size="0.05" material="trail"/>\n'
            f'    </body>'
        )

    # Start marker (green) and live planet (mocap).
    parts.append(
        f'    <body name="start_mark" pos="{a:.3f} 0 0.32">\n'
        f'      <geom type="box" size="0.06 0.06 0.06" material="start"/>\n'
        f'    </body>'
    )
    parts.append(
        f'    <body name="planet" mocap="true" pos="{a:.3f} 0 0.32">\n'
        f'      <geom type="sphere" size="{PLANET_R:.3f}" material="planet"/>\n'
        f'    </body>'
    )

    extra = "\n".join(parts)

    # Top-down camera. Place camera above origin looking straight down,
    # framing the orbit with a small margin.
    cam_h = max(6.0, a * 2.4 + 1.5)
    return make_scene_mjcf(
        cam_pos=(0.0, 0.0, cam_h),
        cam_xyaxes=(1, 0, 0, 0, 1, 0),
        floor_size=(max(a + 1.0, 4.0), max(a + 1.0, 4.0), 0.02),
        light_pos=(0.0, 0.0, cam_h + 1.0),
        extra_assets=extra_assets,
        extra_worldbody=extra,
    )


def render_animation(
    scenario: Any, control_value: float, *,
    n_frames: int = 32, width: int = 720, height: int = 720,
) -> tuple[list[Image.Image], int]:
    """Animate the planet sweeping along its orbit. Trail dots
    accumulate behind the planet so the swept arc is visible."""
    sim = scenario.simulate(float(control_value))
    a = float(sim["a"])
    T = float(sim["period"])
    fps = 8
    SETTLE_TAU = 1.0  # use the whole window to show the orbit progressing

    mjcf = _build_mjcf(a)

    # Total angular sweep over the animation. We pick obs_time so that
    # the textbook (alpha=3) case at a=1 sweeps ~270 deg, and faster
    # alphas correspondingly sweep more. We reuse the scenario's
    # _FIXED_OBS_TIME if available; otherwise fall back to T*0.75.
    obs_time = getattr(scenario, "_FIXED_OBS_TIME", T * 0.75)
    omega = 2 * np.pi / max(T, 1e-6)
    total_sweep = omega * obs_time

    def state_fn(tau: float, model: mujoco.MjModel,
                 data: mujoco.MjData) -> None:
        phase = tau if SETTLE_TAU >= 1.0 else min(1.0, tau / SETTLE_TAU)
        theta = phase * total_sweep
        x = a * np.cos(theta)
        y = a * np.sin(theta)
        set_mocap_pos(model, data, "planet", (x, y, 0.32))

        # Trail: place k-th trail dot at angle (k / N_TRAIL) * theta;
        # if it's beyond the live planet's angle, hide it at z=-10.
        for k in range(N_TRAIL):
            t_frac = (k + 1) / N_TRAIL
            t_theta = phase * total_sweep * t_frac
            if t_frac <= phase + 1e-3:
                tx = a * np.cos(t_theta)
                ty = a * np.sin(t_theta)
                set_mocap_pos(model, data, f"trail_{k}", (tx, ty, 0.32))
            else:
                set_mocap_pos(model, data, f"trail_{k}", (0.0, 0.0, -10.0))

    frames = render_frames_with_state(
        mjcf, state_fn, n_frames=n_frames, width=width, height=height,
    )
    return frames, fps
