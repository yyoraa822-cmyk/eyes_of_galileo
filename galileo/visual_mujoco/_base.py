"""Shared MuJoCo-rendering helpers for visual_mujoco/*.

A single `render_animation` entry pattern: scene authors define
(1) an MJCF string and (2) a state-update callback that maps a
normalised time `tau in [0, 1]` to either body positions, geom
colours, or both. This module owns the boilerplate (model build,
renderer setup, frame loop, PIL conversion).
"""
from __future__ import annotations

from typing import Callable, Optional

import mujoco
import numpy as np
from PIL import Image


# Palette + lighting preamble matched against demo.html DEFAULT cells:
# near-black background, soft ambient + 1 directional light, dark grey
# checkered floor for depth cues. All scenes import this preamble so
# the gallery feels uniform.

_VISUAL_DEFAULTS = """\
  <visual>
    <global offwidth="1920" offheight="1280" fovy="35"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.85 0.85 0.85"
               specular="0.20 0.20 0.20"/>
    <rgba haze="0.0 0.0 0.0 1.0"/>
  </visual>
"""

_ASSETS_DEFAULT = """\
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.50 0.50 0.50"
             rgb2="0.40 0.40 0.40" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.05"/>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.05 0.05 0.07" rgb2="0.02 0.02 0.04"
             width="64" height="64"/>
  </asset>
"""


def make_scene_mjcf(
    *,
    cam_pos: tuple[float, float, float] = (0.0, -7.5, 0.4),
    cam_xyaxes: tuple[float, float, float, float, float, float] = (
        1, 0, 0, 0, -0.15, 0.99
    ),
    floor_size: tuple[float, float, float] = (12.0, 12.0, 0.05),
    light_pos: tuple[float, float, float] = (1.5, -3.5, 4.5),
    extra_assets: str = "",
    extra_worldbody: str = "",
    extra_actuators: str = "",
) -> str:
    # Default cam_pos / cam_xyaxes are tuned to put the floor's horizon
    # line near y/h ~= 0.65 (matches demo.html DEFAULT cells which had
    # apparatus filling the upper 65% with floor in bottom 35%):
    #   - camera height z=0.4 (slightly below apparatus mid-height),
    #   - up vector (0, -0.15, 0.99) tilts the camera up by ~9 degrees,
    #     giving roughly the cinematography of demo.html screenshots.
    """Assemble a stock dark-themed MJCF with one camera + floor + light,
    plus whatever `extra_worldbody` the scene needs. Geom colours and
    body positions can be tweaked at runtime via mjData; this template
    only owns the static look-and-feel."""
    cx, cy, cz = cam_pos
    a, b, c, d, e, f = cam_xyaxes
    fsx, fsy, fsz = floor_size
    lx, ly, lz = light_pos
    return f"""<mujoco model="visual">
  <compiler angle="degree" coordinate="local"/>
  <option gravity="0 0 -9.81" timestep="0.001"/>
{_VISUAL_DEFAULTS}{_ASSETS_DEFAULT}{extra_assets}
  <worldbody>
    <camera name="scene_cam" pos="{cx:.3f} {cy:.3f} {cz:.3f}"
            xyaxes="{a} {b} {c}  {d} {e} {f}"/>
    <light name="key" pos="{lx:.2f} {ly:.2f} {lz:.2f}"
           dir="-0.2 0.7 -1" diffuse="0.85 0.85 0.85"
           specular="0.2 0.2 0.2" ambient="0.05 0.05 0.05"/>
    <geom name="floor" type="plane" size="{fsx} {fsy} {fsz}"
          material="grid" pos="0 0 0"/>
{extra_worldbody}
  </worldbody>
{extra_actuators}
</mujoco>
"""


StateFn = Callable[
    [float, "mujoco.MjModel", "mujoco.MjData"], None
]


def render_frames_with_state(
    mjcf: str,
    state_fn: StateFn,
    *,
    n_frames: int = 32,
    width: int = 960,
    height: int = 600,
    cam_name: str = "scene_cam",
    bg_rgb: tuple[float, float, float] = (0.02, 0.02, 0.04),
) -> list[Image.Image]:
    """Build the model from `mjcf`, then for tau in [0..n_frames-1]
    normalised to [0,1] call `state_fn(tau, model, data)`, run
    `mj_forward`, render via the offscreen renderer, and return the
    list of PIL images."""
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    rndr = mujoco.Renderer(model, height=height, width=width)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

    out: list[Image.Image] = []
    try:
        for k in range(max(1, n_frames)):
            tau = 0.0 if n_frames <= 1 else k / float(n_frames - 1)
            state_fn(tau, model, data)
            mujoco.mj_forward(model, data)
            rndr.update_scene(data, camera=cam_id)
            arr = rndr.render().copy()
            out.append(Image.fromarray(arr).convert("RGB"))
    finally:
        rndr.close()
    return out


def set_geom_pos(model: mujoco.MjModel, data: mujoco.MjData,
                 name: str, xyz: tuple[float, float, float]) -> None:
    """Convenience: move a worldbody-level geom by name. We use mocap
    bodies in MJCFs that need motion; for pure geoms attached to
    worldbody (no joints) we modify model.geom_pos directly so the
    next mj_forward picks it up. For animation accuracy callers should
    normally use mocap bodies (see set_mocap_pos)."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        return
    model.geom_pos[gid] = np.asarray(xyz, dtype=np.float64)


def set_geom_rgba(model: mujoco.MjModel, name: str,
                  rgba: tuple[float, float, float, float]) -> None:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        return
    model.geom_rgba[gid] = np.asarray(rgba, dtype=np.float32)


def set_mocap_pos(model: mujoco.MjModel, data: mujoco.MjData,
                  body_name: str,
                  xyz: tuple[float, float, float]) -> None:
    """Move a mocap body by name. Mocap bodies are the recommended way
    to animate kinematic objects in MuJoCo without writing actuators."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return
    mocap_id = model.body_mocapid[bid]
    if mocap_id >= 0:
        data.mocap_pos[mocap_id] = np.asarray(xyz, dtype=np.float64)


def temperature_to_rgb(t_norm: float) -> tuple[float, float, float]:
    """Map t_norm in [0,1] (0=cold, 1=hot) to a perceptual blue->red
    colour for thermal scenes. Used by heat / cooling / blackbody."""
    t = float(np.clip(t_norm, 0.0, 1.0))
    if t < 0.5:
        s = t / 0.5
        return (s * 0.6, 0.3 + s * 0.4, 1.0 - s * 0.5)
    s = (t - 0.5) / 0.5
    return (0.6 + s * 0.4, 0.7 - s * 0.5, 0.5 - s * 0.5)


def blackbody_to_rgb(temp_K: float) -> tuple[float, float, float]:
    """Approximate Tanner Helland blackbody colour for `temp_K` Kelvin,
    clipped to a perceptually pleasant [0.05, 1.0] range. Used by
    Weber/blackbody scene."""
    t = float(temp_K) / 100.0
    if t <= 66.0:
        r = 1.0
        g = max(0.0, min(1.0, (99.4708025861 * np.log(max(t, 1e-3))
                               - 161.1195681661) / 255.0))
        if t <= 19:
            b = 0.0
        else:
            b = max(0.0, min(1.0, (138.5177312231 * np.log(max(t - 10, 1e-3))
                                   - 305.0447927307) / 255.0))
    else:
        r = max(0.0, min(1.0, (329.698727446 * (t - 60) ** -0.1332047592)
                          / 255.0))
        g = max(0.0, min(1.0, (288.1221695283 * (t - 60) ** -0.0755148492)
                          / 255.0))
        b = 1.0
    # gamma-ish lift so dim values still register against dark bg
    return (max(0.05, r), max(0.05, g), max(0.05, b))
