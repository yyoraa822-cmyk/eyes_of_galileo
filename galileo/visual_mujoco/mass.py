"""MuJoCo visualisation for s2_mass.

Galileo's "all bodies fall the same regardless of mass" demonstration.
The DSL exposes the number of weights N as a knob (via the existing
`MassStack` entity, or via the scenario's `_mass` field), and the VLM
agent picks N. The visualisation shows a stack of N standard physics
weights ("砝码") falling from a brass release bar — exactly the kind
of variable-mass apparatus a student would assemble:

      ╔══════════════ release bar ═══════════╗
      │                                       │
      │             ▣  ← knob                  │
      │           ▆▆▆▆▆ ← weight N             │
      │           ▆▆▆▆▆ ← weight N-1           │
      │             ⋮                         │
      │           ▆▆▆▆▆ ← weight 1             │
      │                                       │
      └────────── floor pad ──────────────────┘

All N weights move together as a single rigid stack; their joint fall
trajectory follows scenario.simulate (s = 0.5*g*t^alpha) regardless
of N — that's the lesson. Strobe afterimages capture the same
rigid-body trajectory at evenly spaced times.
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

H_TOP = 5.0                # release height (top weight's centre at frame 0)
N_STROBES = 10             # afterimages
WEIGHT_DISC_R = 0.30       # radius of each cylindrical weight
WEIGHT_DISC_H = 0.20       # half-height of each weight (so full height 0.40)
KNOB_R = 0.07
KNOB_H = 0.07              # half-height
DROP_X = 0.0               # all weights drop along this x-line
PAD_Z = 0.06               # top of floor pad


def _scenario_mass(scenario: Any) -> int:
    """Resolve the mass count exposed by the DSL/agent. Defaults to 1."""
    n = getattr(scenario, "_mass", None)
    try:
        return max(1, int(n))
    except Exception:
        return 1


def _build_mjcf(n_weights: int) -> str:
    extra_assets = """
  <asset>
    <material name="weight_dark" rgba="0.30 0.30 0.34 1" specular="0.55"
              shininess="0.6" reflectance="0.15"/>
    <material name="weight_band" rgba="0.78 0.60 0.28 1" specular="0.40"
              shininess="0.5"/>
    <material name="ghost_dark" rgba="0.30 0.30 0.34 0.40"/>
    <material name="ghost_band" rgba="0.78 0.60 0.28 0.40"/>
  </asset>
"""
    parts: list[str] = []

    # Live weight stack: a single mocap body holding N stacked disks
    # + a top knob, so the whole stack moves rigidly. Origin is at the
    # bottom of the bottom-most disk.
    parts.append(
        f'    <body name="stack" mocap="true" pos="0 0 {H_TOP:.3f}">\n'
    )
    for i in range(n_weights):
        cz = WEIGHT_DISC_H * (1 + 2 * i)
        parts.append(
            f'      <geom type="cylinder" size="{WEIGHT_DISC_R:.3f} '
            f'{WEIGHT_DISC_H:.3f}" pos="0 0 {cz:.3f}"\n'
            f'            material="weight_dark"/>\n'
        )
        parts.append(
            f'      <geom type="cylinder" size="{WEIGHT_DISC_R + 0.02:.3f} '
            f'0.025" pos="0 0 {cz:.3f}" material="weight_band"/>\n'
        )
    knob_cz = WEIGHT_DISC_H * (2 * n_weights) + KNOB_H
    parts.append(
        f'      <geom type="cylinder" size="{KNOB_R:.3f} {KNOB_H:.3f}"\n'
        f'            pos="0 0 {knob_cz:.3f}" material="weight_band"/>\n'
    )
    parts.append('    </body>')

    # No strobe afterimages baked in — a `StrobeTrail` instrument can
    # be added separately via the DSL when the agent wants to capture
    # the trajectory.

    extra = "\n".join(parts)

    # Camera and floor framing match s1_freefall: distant view of the
    # grey grid floor with the apparatus filling the upper portion.
    return make_scene_mjcf(
        cam_pos=(0.0, -8.0, 2.0),
        cam_xyaxes=(1, 0, 0, 0, -0.20, 0.98),
        floor_size=(6.0, 6.0, 0.05),
        light_pos=(1.5, -3.5, 5.0),
        extra_assets=extra_assets,
        extra_worldbody=extra,
    )


def render_animation(
    scenario: Any, control_value: float, *,
    n_frames: int = 32, width: int = 480, height: int = 540,
) -> tuple[list[Image.Image], int]:
    """Drop a stack of N standard weights from H_TOP. N is read from
    the scenario's `_mass` field (set via DSL or by the agent's
    apparatus config). The trajectory follows the scenario's hidden
    law and is mass-independent — the visual N just makes the agent's
    choice of mass tangible."""
    n_weights = _scenario_mass(scenario)
    # control_value is observation time; clamp to a reasonable max so
    # the animation doesn't fly off-screen for huge max_t values.
    obs_t = max(0.05, float(control_value))

    def fall_distance(t: float) -> float:
        return float(scenario.get_observable(max(0.0, t)))

    s_max = fall_distance(obs_t)
    z_floor_top = 0.05
    avail = H_TOP - z_floor_top - 0.05
    scale = avail / max(s_max, 1e-6) if s_max > 0 else 1.0

    mjcf = _build_mjcf(n_weights)
    fps = 8

    def stack_z(t: float) -> float:
        return max(z_floor_top, H_TOP - scale * fall_distance(t))

    def state_fn(tau: float, model: mujoco.MjModel,
                 data: mujoco.MjData) -> None:
        t = tau * obs_t
        z = stack_z(t)
        set_mocap_pos(model, data, "stack", (DROP_X, 0.0, z))

    frames = render_frames_with_state(
        mjcf, state_fn, n_frames=n_frames, width=width, height=height,
    )
    return frames, fps
