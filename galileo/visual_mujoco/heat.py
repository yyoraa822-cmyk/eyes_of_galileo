"""MuJoCo visualisation for s8_heat.

Ingenhousz's wax-coated metal rod (1789): a metal rod is heated at one
end; the rate at which the heat front advances reveals the diffusion
exponent x_front ∝ t^alpha (textbook alpha = 0.5).

Apparatus mapping:

   ┌──┐
   │🔥│  flame at left end
   └──┘
     ║════════════════════════║
                rod (10 m visualised in apparatus units)
     ▼  ▼  ▼  ▼  ▼  ▽  ▽  ▽  ▽
     wax pellets — orange = melted (behind the front), grey = solid

We slice the rod into ~32 segments and color them along a thermal
gradient. The heat front position (`scenario.get_observable(t)`) is
marked by a bright green ring on the rod. A row of small wax pellets
beneath the rod turns from grey to orange as the front passes them.
The animation interpolates the front from x=0 to x_front_target so
the agent can watch one experiment evolve.
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from PIL import Image

from ._base import (
    make_scene_mjcf,
    render_frames_with_state,
    set_geom_rgba,
    set_mocap_pos,
    temperature_to_rgb,
)

ROD_LENGTH = 10.0
ROD_RADIUS = 0.18
N_SEGMENTS = 32
WAX_SPACING = 0.5  # one pellet every 0.5 m along rod (matches scenarios/heat.py)


def _segment_x(i: int) -> float:
    """Centre x of rod segment i in [0, ROD_LENGTH]."""
    seg_len = ROD_LENGTH / N_SEGMENTS
    return seg_len * (i + 0.5)


def _build_mjcf() -> str:
    extra_assets = """
  <asset>
    <material name="brass" rgba="0.78 0.60 0.28 1" specular="0.40"
              shininess="0.5"/>
    <material name="flame" rgba="1.00 0.55 0.20 1" emission="0.8"/>
    <material name="wax_solid" rgba="0.85 0.85 0.85 1" specular="0.10"
              shininess="0.2"/>
    <material name="front" rgba="0.30 0.95 0.40 1" emission="0.6"/>
    <material name="stand" rgba="0.42 0.34 0.27 1" specular="0.10"
              shininess="0.2"/>
  </asset>
"""
    seg_len = ROD_LENGTH / N_SEGMENTS
    seg_half = seg_len / 2.0

    parts: list[str] = []

    # Rod centred along x in [0, ROD_LENGTH], at y=0, z=0.6.
    # Each segment is a brass capsule whose colour we update each frame.
    for i in range(N_SEGMENTS):
        cx = _segment_x(i)
        parts.append(
            f'    <body name="seg_{i}" pos="{cx:.3f} 0 0.60">\n'
            f'      <geom name="seg_geom_{i}" type="capsule"\n'
            f'            size="{ROD_RADIUS:.3f} {seg_half:.3f}"\n'
            f'            quat="0.7071 0 0.7071 0" material="brass"/>\n'
            f'    </body>'
        )

    # Wax pellets every 0.5 m along the rod — historical Ingenhousz
    # apparatus. Dynamic: each pellet turns from grey (solid) to
    # amber (melted) once the heat front passes its position. The
    # agent can disable them by attaching a different DSL instrument
    # set; we render them by default so the scene matches the
    # historical 1789 setup out of the box.
    n_wax = int(ROD_LENGTH / WAX_SPACING)
    for k in range(n_wax):
        wx = (k + 1) * WAX_SPACING
        parts.append(
            f'    <body name="wax_{k}" pos="{wx:.3f} 0 0.30">\n'
            f'      <geom name="wax_geom_{k}" type="sphere" size="0.10"\n'
            f'            material="wax_solid"/>\n'
            f'    </body>'
        )

    # Stand: two short brass posts holding the rod. Posts are placed
    # at 1/4 and 3/4 along the rod so they don't overlap the heated
    # end or the cool tip.
    parts.append(
        f'    <body name="stand_left" pos="{ROD_LENGTH * 0.25:.3f} 0 0.30">\n'
        f'      <geom type="cylinder" size="0.05 0.30" material="stand"/>\n'
        f'    </body>'
    )
    parts.append(
        f'    <body name="stand_right" pos="{ROD_LENGTH * 0.75:.3f} 0 0.30">\n'
        f'      <geom type="cylinder" size="0.05 0.30" material="stand"/>\n'
        f'    </body>'
    )

    # Flame at the heated (x=0) end: a glowing capsule + a small
    # cylinder base hinting at a Bunsen burner.
    parts.append(
        f'    <body name="burner" pos="-0.25 0 0.20">\n'
        f'      <geom type="cylinder" size="0.08 0.20" material="brass"/>\n'
        f'    </body>'
    )
    parts.append(
        f'    <body name="flame" pos="-0.25 0 0.55">\n'
        f'      <geom type="capsule" size="0.08 0.12" material="flame"/>\n'
        f'    </body>'
    )

    # Front marker: a glowing green ring around the rod, animated
    # along x by mocap.
    parts.append(
        f'    <body name="front_marker" mocap="true" pos="0 0 0.60">\n'
        f'      <geom name="front_marker_geom" type="cylinder"\n'
        f'            size="{ROD_RADIUS + 0.04:.3f} 0.025"\n'
        f'            quat="0.7071 0 0.7071 0" material="front"/>\n'
        f'    </body>'
    )

    extra = "\n".join(parts)

    return make_scene_mjcf(
        cam_pos=(ROD_LENGTH / 2, -7.0, 0.5),
        cam_xyaxes=(1, 0, 0, 0, -0.18, 0.98),
        floor_size=(ROD_LENGTH * 1.2, 6.0, 0.05),
        light_pos=(ROD_LENGTH / 2, -3.5, 4.5),
        extra_assets=extra_assets,
        extra_worldbody=extra,
    )


def render_animation(
    scenario: Any, control_value: float, *,
    n_frames: int = 32, width: int = 1200, height: int = 480,
) -> tuple[list[Image.Image], int]:
    """Animate heat front advancing along the rod from t=0 to
    t=control_value.

    For each animation frame we compute t=phase*control_value and
    update each rod segment's colour by a thermal gradient (cool
    blue at the cold end, hot orange/yellow near the front).
    """
    sim = scenario.simulate(float(control_value))
    x_front_target = float(sim["front_position"])
    fps = 8
    SETTLE_TAU = 0.65

    mjcf = _build_mjcf()

    def state_fn(tau: float, model: mujoco.MjModel,
                 data: mujoco.MjData) -> None:
        phase = min(1.0, tau / SETTLE_TAU)
        # erfc-style smoothness: front advances faster early, slower late
        s = phase ** (1.0 / 1.5)
        x_front = s * x_front_target

        # Color each rod segment by distance to the heated end.
        # Behind the front: hot (red/orange); ahead: cool blue/grey.
        for i in range(N_SEGMENTS):
            cx = _segment_x(i)
            if x_front <= 1e-6:
                t_norm = 0.0
            else:
                # Smooth profile: 1 at x=0, decays to 0 at x=x_front,
                # essentially 0 beyond.
                u = cx / x_front
                t_norm = float(np.clip(np.exp(-u * 1.6) - 0.05, 0.0, 1.0))
            r, g, b = temperature_to_rgb(t_norm)
            set_geom_rgba(model, f"seg_geom_{i}", (r, g, b, 1.0))

        # Wax pellets: amber when front has passed them, grey otherwise.
        n_wax = int(ROD_LENGTH / WAX_SPACING)
        for k in range(n_wax):
            wx = (k + 1) * WAX_SPACING
            if wx <= x_front:
                set_geom_rgba(model, f"wax_geom_{k}",
                              (0.95, 0.55, 0.18, 1.0))
            else:
                set_geom_rgba(model, f"wax_geom_{k}",
                              (0.85, 0.85, 0.85, 1.0))

        # Move the green front marker.
        set_mocap_pos(model, data, "front_marker",
                      (x_front, 0.0, 0.60))

    frames = render_frames_with_state(
        mjcf, state_fn, n_frames=n_frames, width=width, height=height,
    )
    return frames, fps
