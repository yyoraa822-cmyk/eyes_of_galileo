"""MuJoCo visualisation for s18_boyle.

Boyle's law `P · V^α = const`. Modern lab demonstration: a sealed
syringe is held vertically with the plunger pressed down by a
weight (or by hand). The trapped gas column shrinks as the applied
pressure grows. The agent reads the gas-column length as the
observable for each pressure setting.

Apparatus mapping:

       ┌──┐    plunger handle
       │  │
       │  │
    ──╤══╤──   plunger disk (pressed down)
       ║▒▒║    trapped gas (cyan)         <- shrinks with P
       ║▒▒║
       ║▒▒║
       ╚══╝    sealed bottom + ring
       █  █    base (brass) on the table

The plunger and gas column are mocap-driven so the apparatus snaps to
the correct equilibrium for the requested pressure. A red ring marks
the gas-meniscus / plunger-bottom contact line so vision agents can
track the gas length the same way they track the matplotlib version.
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


# Geometry
SYR_R = 0.18                 # inner radius of the syringe barrel
WALL = 0.012                 # glass wall thickness
BARREL_LEN = 2.20            # full barrel length (gas + sliding plunger range)
BASE_Z = 0.20                # bottom of barrel sits at this z
TOP_Z = BASE_Z + BARREL_LEN
PLUNGER_DISC_H = 0.05        # half-height of the plunger disc
PLUNGER_HANDLE_LEN = 0.50    # half-length of the handle rod above the disc
GAS_MAX_LEN = 1.80           # gas column length at P = P0 (1 atm)


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
    <material name="glass" rgba="0.65 0.82 0.88 0.10" specular="0.30"
              shininess="0.5" reflectance="0.05"/>
    <material name="gas" rgba="0.55 0.85 1.00 0.65" specular="0.10"
              shininess="0.2"/>
    <material name="brass" rgba="0.78 0.60 0.28 1" specular="0.40"
              shininess="0.5"/>
    <material name="plunger" rgba="0.42 0.42 0.48 1" specular="0.55"
              shininess="0.6"/>
    <material name="handle" rgba="0.85 0.85 0.90 1" specular="0.50"
              shininess="0.6"/>
    <material name="redmark" rgba="0.95 0.20 0.20 1"/>
    <material name="bench" rgba="0.42 0.34 0.27 1" specular="0.10"
              shininess="0.2"/>
    <material name="weight" rgba="0.30 0.30 0.34 1" specular="0.55"
              shininess="0.6"/>
  </asset>
"""
    parts: list[str] = []

    # Bench under the syringe.
    parts.append(
        f'    <body name="bench" pos="0 0 {BASE_Z - 0.04:.3f}">\n'
        f'      <geom type="box" size="0.55 0.55 0.04" material="bench"/>\n'
        f'    </body>'
    )
    # Brass sealed bottom of the syringe.
    parts.append(
        f'    <body name="syr_bottom" pos="0 0 {BASE_Z + 0.018:.3f}">\n'
        f'      <geom type="cylinder" size="{SYR_R + WALL + 0.012:.3f} 0.018"\n'
        f'            material="brass"/>\n'
        f'    </body>'
    )
    # Glass barrel (faint, almost invisible) and brass top/bottom rings.
    arm_mid_z = BASE_Z + BARREL_LEN / 2
    parts.append(
        f'    <body name="syr_glass" pos="0 0 {arm_mid_z:.3f}">\n'
        f'      <geom type="cylinder" size="{SYR_R + WALL:.3f} '
        f'{BARREL_LEN / 2:.3f}" material="glass"/>\n'
        f'    </body>'
    )
    parts.append(
        f'    <body name="syr_botring" pos="0 0 {BASE_Z + 0.005:.3f}">\n'
        f'      <geom type="cylinder" size="{SYR_R + WALL + 0.008:.3f} 0.012"\n'
        f'            material="brass"/>\n'
        f'    </body>'
    )
    parts.append(
        f'    <body name="syr_topring" pos="0 0 {TOP_Z - 0.005:.3f}">\n'
        f'      <geom type="cylinder" size="{SYR_R + WALL + 0.008:.3f} 0.012"\n'
        f'            material="brass"/>\n'
        f'    </body>'
    )

    # Trapped gas column (mocap-resized).
    parts.append(
        f'    <body name="gas_col" mocap="true" pos="0 0 {BASE_Z + 0.5:.3f}">\n'
        f'      <geom name="gas_col_geom" type="cylinder"\n'
        f'            size="{SYR_R - 0.005:.3f} 0.5" material="gas"/>\n'
        f'    </body>'
    )

    # Red marker ring at the plunger bottom (gas / plunger interface).
    parts.append(
        f'    <body name="meniscus_ring" mocap="true"\n'
        f'          pos="0 0 {BASE_Z + 0.5:.3f}">\n'
        f'      <geom name="meniscus_ring_geom" type="cylinder"\n'
        f'            size="{SYR_R + WALL + 0.004:.3f} 0.012"\n'
        f'            material="redmark"/>\n'
        f'    </body>'
    )

    # Plunger disc + handle rod, both translated together as a single
    # mocap body. Origin sits at the plunger-bottom face.
    parts.append(
        f'    <body name="plunger" mocap="true" pos="0 0 {BASE_Z + 1.0:.3f}">\n'
        f'      <geom type="cylinder" size="{SYR_R - 0.002:.3f} '
        f'{PLUNGER_DISC_H:.3f}" pos="0 0 {PLUNGER_DISC_H:.3f}"\n'
        f'            material="plunger"/>\n'
        f'      <geom type="capsule" size="0.04"\n'
        f'            fromto="0 0 {2 * PLUNGER_DISC_H:.3f}  '
        f'0 0 {2 * PLUNGER_DISC_H + PLUNGER_HANDLE_LEN * 2:.3f}"\n'
        f'            material="handle"/>\n'
        f'      <geom type="box" size="0.18 0.04 0.04"\n'
        f'            pos="0 0 {2 * PLUNGER_DISC_H + PLUNGER_HANDLE_LEN * 2:.3f}"\n'
        f'            material="handle"/>\n'
        f'    </body>'
    )

    # Optional weight icon on top of the plunger (mocap-shown when
    # the agent applies extra pressure). For now we always show it as
    # a visual hint that the plunger is being pressed.
    parts.append(
        f'    <body name="press_weight" mocap="true" pos="0 0 -10">\n'
        f'      <geom type="cylinder" size="0.20 0.10" material="weight"/>\n'
        f'    </body>'
    )

    extra = "\n".join(parts)

    return make_scene_mjcf(
        cam_pos=(0.0, -3.6, 1.40),
        cam_xyaxes=(1, 0, 0, 0, -0.18, 0.98),
        floor_size=(5.0, 5.0, 0.05),
        light_pos=(1.0, -2.5, 4.0),
        extra_assets=extra_assets,
        extra_worldbody=extra,
    )


def render_animation(
    scenario: Any, control_value: float, *,
    n_frames: int = 32, width: int = 480, height: int = 600,
) -> tuple[list[Image.Image], int]:
    """Render the syringe with the plunger held at the equilibrium
    position for the requested pressure. The visual gas column
    matches V/V0; the plunger sits on top of the gas; the weight
    icon hovers above the handle when P > 1 atm (visual cue that
    pressure is being applied)."""
    sim = scenario.simulate(float(control_value))
    P_target = float(sim["pressure"])
    V_target = float(sim["volume"])
    V_at_P0 = float(scenario.get_observable(1.0))
    fps = 8
    SETTLE_TAU = 0.65

    # Animation: the plunger pushes down from the uncompressed (P=1,
    # V=V_at_P0) state to the equilibrium (P_target, V_target) over
    # the swing window, then holds for the rest of the timeline. The
    # weight icon snaps in once we're past P=1 atm.
    mjcf = _build_mjcf()

    def state_fn(tau: float, model: mujoco.MjModel,
                 data: mujoco.MjData) -> None:
        phase = min(1.0, tau / SETTLE_TAU)
        s = phase * phase * (3.0 - 2.0 * phase)
        P = 1.0 + s * (P_target - 1.0)
        V = V_at_P0 + s * (V_target - V_at_P0)

        gas_len = max(0.05, GAS_MAX_LEN * (V / max(V_at_P0, 1e-6)))
        plunger_bottom_z = BASE_Z + gas_len
        weight_z = (plunger_bottom_z + 2 * PLUNGER_DISC_H
                    + 2 * PLUNGER_HANDLE_LEN + 0.10)

        gas_half = max(0.005, gas_len / 2.0)
        gas_mid = BASE_Z + gas_half
        _set_capsule_size_z(model, "gas_col_geom", gas_half)
        set_mocap_pos(model, data, "gas_col", (0.0, 0.0, gas_mid))
        set_mocap_pos(model, data, "meniscus_ring",
                      (0.0, 0.0, plunger_bottom_z))
        set_mocap_pos(model, data, "plunger",
                      (0.0, 0.0, plunger_bottom_z))
        if P > 1.0001:
            set_mocap_pos(model, data, "press_weight",
                          (0.0, 0.0, weight_z))
        else:
            set_mocap_pos(model, data, "press_weight",
                          (0.0, 0.0, -10.0))

    frames = render_frames_with_state(
        mjcf, state_fn, n_frames=n_frames, width=width, height=height,
    )
    return frames, fps
