"""MuJoCo visualisation for s20_hooke.

Same physics as s9_spring (x = (mg/k)^alpha) but framed as the Hooke
"F = k x" force-extension law on a horizontal spring + table top —
visually distinct from the vertical hanging-mass apparatus of s9.

Apparatus mapping:

   ┌─wall─┐
   │ ████ │~~~~~~~~~~~~~~~~~~~~~~ ▣  →   force arrow
   │ ████ │     coil spring        block (pulled rightward)
   │ ████ │
   └──────┘═════════ table ═══════════

We drive the apparatus by the same control_value (mass) and call
scenario.get_observable to obtain the equilibrium extension; the
extension is then drawn horizontally on the table. A bright orange
arrow indicates the applied force F = k * x to underline the
Hooke-law framing.
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

WALL_X = -1.30
TABLE_Z = 0.40
NATURAL_LEN = 0.50
N_COILS = 12
COIL_RADIUS = 0.10
WIRE_RADIUS = 0.018
BLOCK_HALF = 0.16


def _build_mjcf(max_extension: float) -> str:
    extra_assets = """
  <asset>
    <material name="brass" rgba="0.78 0.60 0.28 1" specular="0.40"
              shininess="0.5"/>
    <material name="wall" rgba="0.45 0.40 0.35 1" specular="0.10"
              shininess="0.2"/>
    <material name="table" rgba="0.62 0.45 0.28 1" specular="0.20"
              shininess="0.3"/>
    <material name="coil" rgba="0.70 0.70 0.74 1" specular="0.55"
              shininess="0.6"/>
    <material name="block" rgba="0.85 0.30 0.30 1" specular="0.30"
              shininess="0.4"/>
    <material name="force" rgba="1.00 0.55 0.20 1" emission="0.55"/>
    <material name="rule_body" rgba="0.85 0.78 0.55 1"/>
  </asset>
"""
    parts: list[str] = []

    # Table top
    parts.append(
        f'    <body name="table" pos="0.5 0 {TABLE_Z - 0.04:.3f}">\n'
        f'      <geom type="box" size="2.4 0.55 0.04" material="table"/>\n'
        f'    </body>'
    )
    # Wall on the left
    parts.append(
        f'    <body name="wall" pos="{WALL_X - 0.10:.3f} 0 {TABLE_Z + 0.45:.3f}">\n'
        f'      <geom type="box" size="0.10 0.55 0.45" material="wall"/>\n'
        f'    </body>'
    )
    # Anchor cap on the wall
    parts.append(
        f'    <body name="anchor" pos="{WALL_X - 0.02:.3f} 0 '
        f'{TABLE_Z + 0.18:.3f}">\n'
        f'      <geom type="cylinder" size="0.07 0.04"\n'
        f'            quat="0.7071 0 0.7071 0" material="brass"/>\n'
        f'    </body>'
    )

    # Spring coils: stack of vertical-axis cylinders translated along x.
    for i in range(N_COILS):
        parts.append(
            f'    <body name="coil_{i}" mocap="true"\n'
            f'          pos="0 0 {TABLE_Z + 0.18:.3f}">\n'
            f'      <geom type="cylinder" size="{COIL_RADIUS:.3f} '
            f'{WIRE_RADIUS:.3f}"\n'
            f'            quat="0.7071 0 0.7071 0" material="coil"/>\n'
            f'    </body>'
        )

    # Hanging block (the pulled mass)
    parts.append(
        f'    <body name="block" mocap="true" pos="0 0 {TABLE_Z + 0.18:.3f}">\n'
        f'      <geom type="box"\n'
        f'            size="{BLOCK_HALF:.3f} {BLOCK_HALF:.3f} {BLOCK_HALF:.3f}"\n'
        f'            material="block"/>\n'
        f'    </body>'
    )

    # Force arrow tail (extends to the right from the block).
    parts.append(
        f'    <body name="force_shaft" mocap="true" pos="0 0 '
        f'{TABLE_Z + 0.18:.3f}">\n'
        f'      <geom type="capsule" size="0.025"\n'
        f'            fromto="0 0 0  0.50 0 0" material="force"/>\n'
        f'    </body>'
    )
    parts.append(
        f'    <body name="force_head" mocap="true" pos="0 0 '
        f'{TABLE_Z + 0.18:.3f}">\n'
        f'      <geom type="cylinder" size="0.07 0.06"\n'
        f'            quat="0.7071 0 0.7071 0" material="force"/>\n'
        f'    </body>'
    )

    # Ruler etched into the table — small brass tick marks.
    rul_x_min = WALL_X + 0.05
    rul_x_max = WALL_X + 0.05 + NATURAL_LEN + max_extension + 0.50
    n_t = 13
    for i in range(n_t):
        x = rul_x_min + (rul_x_max - rul_x_min) * i / (n_t - 1)
        long_tick = (i % 3 == 0)
        parts.append(
            f'    <body name="hrtick_{i}" pos="0 0 0">\n'
            f'      <geom type="capsule" size="0.012" material="brass"\n'
            f'            fromto="{x:.3f} 0 {TABLE_Z + 0.005:.3f}  '
            f'{x:.3f} 0 {TABLE_Z + (0.10 if long_tick else 0.06):.3f}"/>\n'
            f'    </body>'
        )

    extra = "\n".join(parts)

    return make_scene_mjcf(
        cam_pos=(0.4, -3.4, 0.85),
        cam_xyaxes=(1, 0, 0, 0, -0.10, 0.99),
        floor_size=(5.0, 5.0, 0.05),
        light_pos=(1.0, -2.5, 4.0),
        extra_assets=extra_assets,
        extra_worldbody=extra,
    )


def render_animation(
    scenario: Any, control_value: float, *,
    n_frames: int = 32, width: int = 720, height: int = 480,
) -> tuple[list[Image.Image], int]:
    """Animate the block being pulled out from rest to the equilibrium
    extension under force F = k*x. Force arrow grows in proportion."""
    sim = scenario.simulate(float(control_value))
    ext = float(sim["extension"])
    fps = 8
    SETTLE_TAU = 0.65

    # Cap visual extension so the arrow + block stay in frame for any
    # mass the agent picks; the data-side observable still reflects
    # the true extension via scenario.get_observable.
    visual_ext = min(ext, 1.50)

    mjcf = _build_mjcf(visual_ext)
    OMEGA = 2.0 * np.pi * 1.4
    ZETA = 0.30

    def coil_centre_x(coil_i: int, current_ext: float) -> float:
        total_len = NATURAL_LEN + current_ext
        frac = (coil_i + 0.5) / N_COILS
        return WALL_X + 0.06 + frac * total_len

    def state_fn(tau: float, model: mujoco.MjModel,
                 data: mujoco.MjData) -> None:
        if tau < SETTLE_TAU:
            t = (tau / SETTLE_TAU) * 1.6
            env = np.exp(-ZETA * OMEGA * t)
            omega_d = OMEGA * np.sqrt(max(1e-6, 1.0 - ZETA * ZETA))
            current_ext = visual_ext * (1.0 - env * np.cos(omega_d * t))
        else:
            current_ext = visual_ext

        coil_z = TABLE_Z + 0.18
        # Coils slide along x and zigzag a tiny bit in z so they read
        # as a coil instead of a row of identical cylinders.
        for i in range(N_COILS):
            cx = coil_centre_x(i, current_ext)
            cz = coil_z + COIL_RADIUS * 0.18 * (1 if i % 2 == 0 else -1)
            set_mocap_pos(model, data, f"coil_{i}", (cx, 0.0, cz))

        # Block sits at the right end of the coil, on the table.
        block_x = WALL_X + 0.06 + NATURAL_LEN + current_ext + BLOCK_HALF
        block_z = TABLE_Z + BLOCK_HALF
        set_mocap_pos(model, data, "block", (block_x, 0.0, block_z))

        # Force arrow grows with the extension. Force shaft extends
        # from the block's right face out by F_LEN; arrow head sits
        # at the tip.
        F_frac = current_ext / max(visual_ext, 1e-6)
        F_LEN = 0.20 + 0.50 * F_frac
        shaft_x = block_x + BLOCK_HALF + F_LEN / 2
        head_x = block_x + BLOCK_HALF + F_LEN + 0.07
        # Resize the shaft capsule by translating its endpoints (we
        # rebuild the capsule via mocap pos; geom_size cylinder length
        # is fixed in the MJCF, so we instead place the body so its
        # capsule's effective range covers (block_x+BLOCK_HALF) -> tip).
        # The capsule was authored as fromto="0 0 0  0.50 0 0" centred
        # at the body origin, so translating the body by `shaft_x -
        # 0.25` would shift it; but we'd lose the length scaling. To
        # keep the implementation simple we just resize the cylinder
        # half-length via geom_size at runtime.
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                "force_shaft")
        # `force_shaft` is the body name — to get the geom we need to
        # look it up by geom name; in this MJCF the geom is unnamed, so
        # iterate through geoms attached to that body.
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                "force_shaft")
        if bid >= 0:
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == bid:
                    sz = model.geom_size[g].copy()
                    sz[1] = max(0.005, F_LEN / 2)
                    model.geom_size[g] = sz
        set_mocap_pos(model, data, "force_shaft",
                      (shaft_x - F_LEN / 2 + F_LEN / 2, 0.0,
                       TABLE_Z + 0.18))
        set_mocap_pos(model, data, "force_shaft",
                      (block_x + BLOCK_HALF + F_LEN / 2, 0.0,
                       TABLE_Z + 0.18))
        set_mocap_pos(model, data, "force_head",
                      (head_x, 0.0, TABLE_Z + 0.18))

    frames = render_frames_with_state(
        mjcf, state_fn, n_frames=n_frames, width=width, height=height,
    )
    return frames, fps
