"""MuJoCo visualisation for s22_cooling.

Newton's cooling experiment (1701): a hot vessel cools toward an
ambient temperature, with the excess temperature decaying as
ΔT(t) = ΔT0 * exp(rate * t). The agent picks observation times and
reads off the remaining excess temperature.

Apparatus mapping:

   ┌─┐                              ╱╲
   │ │  thermometer with mercury    ╱──╲ ambient marker
   │█│  level (height = T - T_env)  ╲──╱ (light blue)
   │█│                              ╲╱
   │█│                               │
   └─┘──────── bench ────────────────┘
        │
        │  cooling cup (brass beaker, hot)
        │
   ╔═══╗
   ║▓▓▓║  cup colored by current temperature
   ║▓▓▓║
   ╚═══╝
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

THERMO_X = +1.20
CUP_X = -1.10
THERMO_BASE_Z = 0.30
THERMO_LEN = 1.80      # bulb to top of column
THERMO_R = 0.07
CUP_R = 0.45
CUP_H = 0.70
CUP_BASE_Z = 0.10


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
    <material name="bench" rgba="0.42 0.34 0.27 1" specular="0.10"
              shininess="0.2"/>
    <material name="glass" rgba="0.82 0.92 0.96 0.20" specular="0.40"
              shininess="0.6"/>
    <material name="mercury" rgba="0.85 0.20 0.20 1" emission="0.30"/>
    <material name="ambient" rgba="0.50 0.78 0.95 1" emission="0.30"/>
    <material name="cup" rgba="0.78 0.45 0.25 1" specular="0.40"
              shininess="0.4"/>
    <material name="liquid" rgba="0.90 0.55 0.25 1" emission="0.45"/>
    <material name="steam" rgba="0.85 0.85 0.92 0.35"/>
  </asset>
"""
    parts: list[str] = []

    # Bench
    parts.append(
        f'    <body name="bench" pos="0 0 0.05">\n'
        f'      <geom type="box" size="2.4 0.55 0.05" material="bench"/>\n'
        f'    </body>'
    )
    # Cup (brass beaker on the left)
    parts.append(
        f'    <body name="cup" pos="{CUP_X:.3f} 0 {CUP_BASE_Z + CUP_H / 2:.3f}">\n'
        f'      <geom type="cylinder" size="{CUP_R:.3f} {CUP_H / 2:.3f}"\n'
        f'            material="cup"/>\n'
        f'    </body>'
    )
    # Liquid inside the cup, mocap-coloured by current temperature.
    parts.append(
        f'    <body name="liquid" pos="{CUP_X:.3f} 0 {CUP_BASE_Z + CUP_H - 0.05:.3f}">\n'
        f'      <geom name="liquid_geom" type="cylinder"\n'
        f'            size="{CUP_R - 0.04:.3f} 0.04" material="liquid"/>\n'
        f'    </body>'
    )
    # Steam puffs (small white spheres above the cup) — fade with cooling.
    for k, dx in enumerate((-0.20, +0.05, +0.25)):
        parts.append(
            f'    <body name="steam_{k}" pos="{CUP_X + dx:.3f} 0 '
            f'{CUP_BASE_Z + CUP_H + 0.18 + 0.05 * k:.3f}">\n'
            f'      <geom name="steam_geom_{k}" type="sphere" size="0.08"\n'
            f'            material="steam"/>\n'
            f'    </body>'
        )

    # Thermometer body: tall transparent glass capsule.
    therm_top_z = THERMO_BASE_Z + THERMO_LEN + 0.08
    therm_mid_z = THERMO_BASE_Z + THERMO_LEN / 2
    parts.append(
        f'    <body name="therm_glass" pos="{THERMO_X:.3f} 0 {therm_mid_z:.3f}">\n'
        f'      <geom type="cylinder" size="{THERMO_R + 0.012:.3f} '
        f'{THERMO_LEN / 2:.3f}" material="glass"/>\n'
        f'    </body>'
    )
    parts.append(
        f'    <body name="therm_bulb" pos="{THERMO_X:.3f} 0 {THERMO_BASE_Z - 0.06:.3f}">\n'
        f'      <geom type="sphere" size="{THERMO_R + 0.04:.3f}" '
        f'material="mercury"/>\n'
        f'    </body>'
    )
    parts.append(
        f'    <body name="therm_cap" pos="{THERMO_X:.3f} 0 {therm_top_z + 0.02:.3f}">\n'
        f'      <geom type="cylinder" size="{THERMO_R + 0.04:.3f} 0.02" '
        f'material="brass"/>\n'
        f'    </body>'
    )
    # Mercury column inside thermometer (mocap-resized).
    parts.append(
        f'    <body name="therm_mercury" mocap="true"\n'
        f'          pos="{THERMO_X:.3f} 0 {THERMO_BASE_Z + 0.5:.3f}">\n'
        f'      <geom name="therm_mercury_geom" type="cylinder"\n'
        f'            size="{THERMO_R:.3f} 0.5" material="mercury"/>\n'
        f'    </body>'
    )
    # Ambient marker: a small blue arrow next to the thermometer at
    # the height corresponding to T_env (frac=0).
    parts.append(
        f'    <body name="ambient_marker" pos="{THERMO_X + 0.25:.3f} 0 '
        f'{THERMO_BASE_Z:.3f}">\n'
        f'      <geom type="capsule" size="0.025"\n'
        f'            fromto="0 0 0  -0.20 0 0" material="ambient"/>\n'
        f'    </body>'
    )
    # Tick marks every 10% of the column.
    for i in range(11):
        z = THERMO_BASE_Z + (i / 10.0) * THERMO_LEN
        x_in = THERMO_X + THERMO_R + 0.08
        x_out = THERMO_X + THERMO_R + 0.20
        parts.append(
            f'    <body name="ctick_{i}" pos="0 0 0">\n'
            f'      <geom type="capsule" size="0.012" material="brass"\n'
            f'            fromto="{x_in:.3f} 0 {z:.3f} '
            f'{x_out:.3f} 0 {z:.3f}"/>\n'
            f'    </body>'
        )

    extra = "\n".join(parts)

    return make_scene_mjcf(
        cam_pos=(0.2, -4.5, 1.10),
        cam_xyaxes=(1, 0, 0, 0, -0.20, 0.98),
        floor_size=(6.0, 5.0, 0.05),
        light_pos=(1.0, -2.5, 4.5),
        extra_assets=extra_assets,
        extra_worldbody=extra,
    )


def render_animation(
    scenario: Any, control_value: float, *,
    n_frames: int = 32, width: int = 960, height: int = 600,
) -> tuple[list[Image.Image], int]:
    """Animate the cup cooling from t=0 (full hot) to t=control_value.

    The mercury column tracks frac = (T - T_env) / (T0 - T_env);
    cup colour, liquid colour and steam visibility all anneal to
    ambient as t -> control_value.
    """
    sim = scenario.simulate(float(control_value))
    t_target = float(sim["time"])
    T0 = float(sim["T0"])
    T_env = float(sim["T_env"])
    fps = 8
    SETTLE_TAU = 0.65

    mjcf = _build_mjcf()

    def state_fn(tau: float, model: mujoco.MjModel,
                 data: mujoco.MjData) -> None:
        phase = min(1.0, tau / SETTLE_TAU)
        t_cur = phase * t_target
        excess = float(scenario.get_observable(t_cur))
        T_cur = T_env + excess
        frac = max(0.0, min(1.0, excess / max(T0 - T_env, 1e-6)))

        # Mercury column height proportional to excess temperature.
        col_half = max(0.005, frac * THERMO_LEN / 2.0)
        col_mid = THERMO_BASE_Z + col_half
        _set_capsule_size_z(model, "therm_mercury_geom", col_half)
        set_mocap_pos(model, data, "therm_mercury",
                      (THERMO_X, 0.0, col_mid))

        # Cup + liquid colours follow temperature_to_rgb (blue->red).
        r, g, b = temperature_to_rgb(frac)
        set_geom_rgba(model, "liquid_geom", (r, g, b, 1.0))

        # Steam fades with cooling (alpha drops as frac drops).
        steam_alpha = 0.10 + 0.50 * frac
        for k in range(3):
            set_geom_rgba(model, f"steam_geom_{k}",
                          (0.85, 0.85, 0.92, steam_alpha))

    frames = render_frames_with_state(
        mjcf, state_fn, n_frames=n_frames, width=width, height=height,
    )
    return frames, fps
