"""MuJoCo visualisation for s17_refraction.

Snell's semicircular protractor (Snellius, ~1621): a flat horizontal
interface separates two media; a collimated ray enters from the air
side at angle theta1 from the normal and refracts inside the glass at
angle theta2. The agent measures (theta1, theta2) pairs to discover
n1 sin(theta1) = n2 sin^alpha(theta2).

Visualisation choice: we render the protractor as a brass disk lying
in the camera plane (xz, y=0), with two emissive rays drawn directly
on top so neither ray gets occluded by the medium volumes. Air is
hinted by a faint blue tint on the upper half-disk; glass by a
warmer tint on the lower half-disk. The interface is a brass bar.
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

RAY_LEN = 1.55
PROT_R = 1.75
INTERFACE_Z = 0.0


def _set_mocap_quat_y(model: mujoco.MjModel, data: mujoco.MjData,
                      body_name: str, theta_y: float) -> None:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return
    mocap_id = model.body_mocapid[bid]
    if mocap_id < 0:
        return
    half = 0.5 * theta_y
    data.mocap_quat[mocap_id] = np.array(
        [np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float64)


def _build_mjcf() -> str:
    extra_assets = """
  <asset>
    <material name="air_disk" rgba="0.42 0.62 0.85 0.55" specular="0.10"
              shininess="0.2"/>
    <material name="glass_disk" rgba="0.80 0.62 0.42 0.55" specular="0.30"
              shininess="0.5"/>
    <material name="brass" rgba="0.78 0.60 0.28 1" specular="0.40"
              shininess="0.5"/>
    <material name="incident" rgba="1.00 0.95 0.30 1" emission="0.85"/>
    <material name="refracted" rgba="0.35 0.85 0.95 1" emission="0.85"/>
    <material name="normal_line" rgba="0.92 0.92 0.92 1" emission="0.20"/>
    <material name="entry" rgba="0.95 0.20 0.20 1" emission="0.5"/>
    <material name="tick" rgba="0.95 0.85 0.55 1" emission="0.30"/>
  </asset>
"""
    # Air half-disk (top) and glass half-disk (bottom). Each is
    # implemented as a thin cylinder rotated to lie in the y=0 plane,
    # then half-hidden by a thin box at z<0 / z>0 respectively. To
    # keep the MJCF simple we instead use two short box panels.
    extra = f"""
    <body name="air_panel" pos="0 0.05 {PROT_R / 2:.3f}"
          quat="0.7071 0 0 0.7071">
      <geom type="cylinder" size="{PROT_R:.3f} 0.012" material="air_disk"/>
    </body>
    <body name="glass_panel" pos="0 0.05 {-PROT_R / 2:.3f}"
          quat="0.7071 0 0 0.7071">
      <geom type="cylinder" size="{PROT_R:.3f} 0.012" material="glass_disk"/>
    </body>
    <body name="interface_bar" pos="0 0 {INTERFACE_Z:.3f}">
      <geom type="box" size="{PROT_R + 0.25:.3f} 0.06 0.020"
            material="brass"/>
    </body>
    <body name="normal_axis" pos="0 0 {INTERFACE_Z:.3f}">
      <geom type="capsule" size="0.012"
            fromto="0 0 -{PROT_R - 0.05:.3f}  0 0 {PROT_R - 0.05:.3f}"
            material="normal_line"/>
    </body>
    <body name="entry_ball" pos="0 0 {INTERFACE_Z:.3f}">
      <geom type="sphere" size="0.07" material="entry"/>
    </body>
    <body name="incident_ray" mocap="true" pos="0 0 {INTERFACE_Z:.3f}">
      <geom name="incident_ray_geom" type="capsule" size="0.030"
            fromto="0 0 0  0 0 {RAY_LEN:.3f}" material="incident"/>
    </body>
    <body name="refracted_ray" mocap="true" pos="0 0 {INTERFACE_Z:.3f}">
      <geom name="refracted_ray_geom" type="capsule" size="0.030"
            fromto="0 0 0  0 0 -{RAY_LEN:.3f}" material="refracted"/>
    </body>
    """
    # Protractor ticks intentionally omitted: ticks belong to a
    # separate `Protractor` instrument that the agent adds via DSL,
    # not to the static apparatus.

    return make_scene_mjcf(
        cam_pos=(0.0, -5.5, 0.0),
        cam_xyaxes=(1, 0, 0, 0, 0, 1),
        floor_size=(6.0, 6.0, 0.05),
        light_pos=(1.5, -3.0, 4.0),
        extra_assets=extra_assets,
        extra_worldbody=extra,
    )


def render_animation(
    scenario: Any, control_value: float, *,
    n_frames: int = 32, width: int = 960, height: int = 600,
) -> tuple[list[Image.Image], int]:
    """Render Snell's apparatus with the incident ray sweeping from the
    normal (theta=0) up to the requested theta1. Refracted ray
    follows by Snell's law (already encoded by `scenario.simulate`)."""
    sim = scenario.simulate(float(control_value))
    theta1_target = float(np.deg2rad(sim["theta1_deg"]))
    theta2_target = float(np.deg2rad(sim["theta2_deg"]))
    fps = 8
    SETTLE_TAU = 0.65  # sweep for first 65% of timeline, then hold

    mjcf = _build_mjcf()

    def state_fn(tau: float, model: mujoco.MjModel,
                 data: mujoco.MjData) -> None:
        phase = min(1.0, tau / SETTLE_TAU)
        s = phase * phase * (3.0 - 2.0 * phase)  # smoothstep
        theta1 = s * theta1_target
        theta2 = s * theta2_target
        set_mocap_pos(model, data, "incident_ray", (0.0, 0.0, INTERFACE_Z))
        _set_mocap_quat_y(model, data, "incident_ray", -theta1)
        set_mocap_pos(model, data, "refracted_ray", (0.0, 0.0, INTERFACE_Z))
        _set_mocap_quat_y(model, data, "refracted_ray", -theta2)

    frames = render_frames_with_state(
        mjcf, state_fn, n_frames=n_frames, width=width, height=height,
    )
    return frames, fps
