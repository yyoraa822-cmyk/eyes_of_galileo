"""MuJoCo visualisation for s16_decay.

Becquerel-style phosphorescent vial (1896): a glowing substance whose
brightness fades as B(t) = B0 * exp(-rate * t). The agent picks
observation times and reads off how much glow remains, discovering
the exponential law.

Apparatus mapping: a tall rectangular bar of phosphorescent material
mounted on a brass stand. The bar's height encodes B(t) / B0; its
colour brightens (more saturated red/orange) when the substance is
fresh and dims (deep red) as it decays. A faint dashed silhouette
shows the original B0 length for visual reference.
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
)

BAR_MAX_LEN = 2.40
BAR_RADIUS = 0.18
BAR_BASE_Z = 0.20


def _set_capsule_size_z(model: mujoco.MjModel, geom_name: str,
                        half_height: float) -> None:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        return
    sz = model.geom_size[gid].copy()
    sz[1] = max(0.005, half_height)
    model.geom_size[gid] = sz


def _build_mjcf() -> str:
    extra_assets = """
  <asset>
    <material name="brass" rgba="0.78 0.60 0.28 1" specular="0.40"
              shininess="0.5"/>
    <material name="phos" rgba="1.00 0.55 0.20 1" emission="0.80"/>
    <material name="ghost" rgba="0.30 0.30 0.30 0.30"/>
    <material name="bench" rgba="0.42 0.34 0.27 1" specular="0.10"
              shininess="0.2"/>
    <material name="mark" rgba="0.30 0.95 0.40 1" emission="0.5"/>
  </asset>
"""
    extra = f"""
    <body name="bench" pos="0 0 {BAR_BASE_Z - 0.05:.3f}">
      <geom type="box" size="0.80 0.40 0.05" material="bench"/>
    </body>
    <body name="ghost_outline" pos="0 0 {BAR_BASE_Z + BAR_MAX_LEN / 2:.3f}">
      <geom type="cylinder" size="{BAR_RADIUS + 0.02:.3f} {BAR_MAX_LEN / 2:.3f}"
            material="ghost"/>
    </body>
    <body name="phos_bar" mocap="true"
          pos="0 0 {BAR_BASE_Z + BAR_MAX_LEN / 2:.3f}">
      <geom name="phos_geom" type="cylinder"
            size="{BAR_RADIUS:.3f} {BAR_MAX_LEN / 2:.3f}"
            material="phos"/>
    </body>
    <body name="cap" pos="0 0 {BAR_BASE_Z + BAR_MAX_LEN + 0.02:.3f}">
      <geom type="cylinder" size="{BAR_RADIUS + 0.04:.3f} 0.025"
            material="brass"/>
    </body>
    <body name="top_marker" mocap="true"
          pos="0 0 {BAR_BASE_Z + BAR_MAX_LEN:.3f}">
      <geom name="top_marker_geom" type="cylinder"
            size="{BAR_RADIUS + 0.05:.3f} 0.018" material="mark"/>
    </body>
    """
    # Tick marks every 10% of the maximum height — small brass capsules
    # on the right side, suggesting a ruler.
    ticks = []
    for i in range(11):
        z = BAR_BASE_Z + (i / 10.0) * BAR_MAX_LEN
        x_in = BAR_RADIUS + 0.08
        x_out = BAR_RADIUS + 0.20
        ticks.append(
            f'    <body name="dtick_{i}" pos="0 0 0">\n'
            f'      <geom type="capsule" size="0.012" material="brass"\n'
            f'            fromto="{x_in:.3f} 0 {z:.3f} '
            f'{x_out:.3f} 0 {z:.3f}"/>\n'
            f'    </body>'
        )
    extra += "\n" + "\n".join(ticks)

    return make_scene_mjcf(
        cam_pos=(0.0, -4.5, 1.30),
        cam_xyaxes=(1, 0, 0, 0, -0.20, 0.98),
        floor_size=(5.0, 5.0, 0.05),
        light_pos=(1.0, -2.5, 4.0),
        extra_assets=extra_assets,
        extra_worldbody=extra,
    )


def render_animation(
    scenario: Any, control_value: float, *,
    n_frames: int = 32, width: int = 720, height: int = 720,
) -> tuple[list[Image.Image], int]:
    """Animate the vial decaying from B0 (full bar) at internal time 0
    toward B(control_value) by the end of the swing phase, then hold.
    """
    sim = scenario.simulate(float(control_value))
    t_target = float(sim["time"])
    B0 = float(sim["B0"])
    fps = 8
    SETTLE_TAU = 0.65

    mjcf = _build_mjcf()

    def state_fn(tau: float, model: mujoco.MjModel,
                 data: mujoco.MjData) -> None:
        phase = min(1.0, tau / SETTLE_TAU)
        t_cur = phase * t_target
        b = float(scenario.get_observable(t_cur))
        frac = max(0.0, min(1.0, b / max(B0, 1e-6)))

        # Resize bar height + reposition centre so it grows from the base.
        bar_half = max(0.005, frac * BAR_MAX_LEN / 2.0)
        bar_mid = BAR_BASE_Z + bar_half
        _set_capsule_size_z(model, "phos_geom", bar_half)
        set_mocap_pos(model, data, "phos_bar", (0.0, 0.0, bar_mid))

        # Brightness/colour: fresh = orange-yellow + bright; faded = dark red.
        r = 0.95 - 0.30 * (1.0 - frac)
        g = 0.55 * frac + 0.10 * (1.0 - frac)
        bcol = 0.20 * frac + 0.05 * (1.0 - frac)
        set_geom_rgba(model, "phos_geom", (r, g, bcol, 1.0))

        # Top marker: green ring at the current top of the bar.
        set_mocap_pos(model, data, "top_marker",
                      (0.0, 0.0, BAR_BASE_Z + frac * BAR_MAX_LEN))

    frames = render_frames_with_state(
        mjcf, state_fn, n_frames=n_frames, width=width, height=height,
    )
    return frames, fps
