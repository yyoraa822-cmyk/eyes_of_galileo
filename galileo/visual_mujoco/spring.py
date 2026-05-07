"""MuJoCo visualisation for s9_spring / s20_hooke (Hooke's Law).

Hooke's spring with weights (1676): a vertical spring is hung from a
fixed ceiling; a known mass is attached at the bottom; the
equilibrium extension is read off a ruler. The agent varies mass and
discovers x ∝ m^alpha (textbook alpha = 1).

Apparatus mapping: a brass ceiling block with a hanging coil spring
(rendered as a stack of small toroidal disks for a coil look), a
red mass block attached at the bottom, and a tall brass ruler on
the right with brass tick marks every 0.05 m. The block's z is
animated from the unloaded position down to the equilibrium
extension during the swing phase, with mild overshoot (zeta=0.3)
to look like an actual settling experiment.
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

CEILING_Z = 2.40
NATURAL_LEN = 0.40         # spring rest length (m, scaled visually)
N_COILS = 14
COIL_RADIUS = 0.12
WIRE_RADIUS = 0.018
BLOCK_HALF = 0.14
RULER_X = 0.95


def _build_mjcf(max_extension: float) -> str:
    extra_assets = """
  <asset>
    <material name="brass" rgba="0.78 0.60 0.28 1" specular="0.40"
              shininess="0.5"/>
    <material name="coil" rgba="0.65 0.65 0.70 1" specular="0.50"
              shininess="0.5"/>
    <material name="block" rgba="0.85 0.30 0.30 1" specular="0.30"
              shininess="0.4"/>
    <material name="ceiling" rgba="0.45 0.40 0.35 1" specular="0.10"
              shininess="0.2"/>
    <material name="rule_body" rgba="0.85 0.78 0.55 1"/>
  </asset>
"""
    parts: list[str] = []

    # Ceiling slab
    parts.append(
        f'    <body name="ceiling" pos="0 0 {CEILING_Z + 0.04:.3f}">\n'
        f'      <geom type="box" size="0.85 0.30 0.04" material="ceiling"/>\n'
        f'    </body>'
    )
    # Anchor cap under the ceiling.
    parts.append(
        f'    <body name="anchor" pos="0 0 {CEILING_Z - 0.02:.3f}">\n'
        f'      <geom type="cylinder" size="0.10 0.04" material="brass"/>\n'
        f'    </body>'
    )

    # Spring coils: N_COILS small torus-like rings stacked from
    # CEILING_Z downward over NATURAL_LEN initially. We move them
    # each frame so the spring stretches under load. Each coil is a
    # mocap body holding a thin cylinder, then the (x, z) of each
    # coil is set by `state_fn`.
    for i in range(N_COILS):
        parts.append(
            f'    <body name="coil_{i}" mocap="true"\n'
            f'          pos="0 0 {CEILING_Z - 0.05 - i * 0.05:.3f}">\n'
            f'      <geom name="coil_geom_{i}" type="cylinder"\n'
            f'            size="{COIL_RADIUS:.3f} {WIRE_RADIUS:.3f}"\n'
            f'            material="coil"/>\n'
            f'    </body>'
        )

    # Hanging block (mass)
    parts.append(
        f'    <body name="block" mocap="true" pos="0 0 0.60">\n'
        f'      <geom type="box"\n'
        f'            size="{BLOCK_HALF:.3f} {BLOCK_HALF:.3f} {BLOCK_HALF:.3f}"\n'
        f'            material="block"/>\n'
        f'    </body>'
    )

    # Ruler on the right
    rul_h = CEILING_Z + 0.02
    parts.append(
        f'    <body name="ruler" pos="{RULER_X:.3f} 0 {rul_h / 2:.3f}">\n'
        f'      <geom type="box" size="0.04 0.04 {rul_h / 2:.3f}"\n'
        f'            material="rule_body"/>\n'
        f'    </body>'
    )
    n_ticks = 21
    for i in range(n_ticks):
        z = (i / (n_ticks - 1)) * rul_h
        x_in = RULER_X
        x_out = RULER_X + (0.18 if i % 5 == 0 else 0.10)
        parts.append(
            f'    <body name="rtick_{i}" pos="0 0 0">\n'
            f'      <geom type="capsule" size="0.012" material="brass"\n'
            f'            fromto="{x_in:.3f} 0 {z:.3f} '
            f'{x_out:.3f} 0 {z:.3f}"/>\n'
            f'    </body>'
        )

    extra = "\n".join(parts)

    return make_scene_mjcf(
        cam_pos=(0.4, -3.6, 1.20),
        cam_xyaxes=(1, 0, 0, 0, -0.05, 1.0),
        floor_size=(4.0, 4.0, 0.05),
        light_pos=(1.0, -2.5, 4.5),
        extra_assets=extra_assets,
        extra_worldbody=extra,
    )


def render_animation(
    scenario: Any, control_value: float, *,
    n_frames: int = 32, width: int = 600, height: int = 720,
) -> tuple[list[Image.Image], int]:
    """Animate spring stretching: block descends from rest length to the
    target extension with mild damped oscillation, then holds."""
    sim = scenario.simulate(float(control_value))
    ext = float(sim["extension"])
    fps = 8
    SETTLE_TAU = 0.65

    # Cap visual extension so the apparatus stays in frame for any
    # control value the agent picks (the underlying physics value is
    # still encoded by the equilibrium block position relative to the
    # ruler — we only clamp the visualisation, not the data).
    visual_ext = min(ext, 1.40)

    mjcf = _build_mjcf(visual_ext)
    OMEGA = 2.0 * np.pi * 1.4
    ZETA = 0.30

    def coil_centre_z(coil_i: int, current_ext: float) -> float:
        # Coil i occupies a fraction (i + 0.5) / N_COILS of the spring's
        # total length; total length = NATURAL_LEN + current_ext.
        total_len = NATURAL_LEN + current_ext
        frac = (coil_i + 0.5) / N_COILS
        return CEILING_Z - 0.06 - frac * total_len

    def state_fn(tau: float, model: mujoco.MjModel,
                 data: mujoco.MjData) -> None:
        if tau < SETTLE_TAU:
            t = (tau / SETTLE_TAU) * 1.6
            env = np.exp(-ZETA * OMEGA * t)
            omega_d = OMEGA * np.sqrt(max(1e-6, 1.0 - ZETA * ZETA))
            current_ext = visual_ext * (1.0 - env * np.cos(omega_d * t))
        else:
            current_ext = visual_ext

        for i in range(N_COILS):
            cz = coil_centre_z(i, current_ext)
            # Add a tiny x-zigzag so the coils read as a coil, not just
            # a stack of disks. Alternates left/right by ±0.02.
            cx = COIL_RADIUS * 0.18 * (1 if i % 2 == 0 else -1)
            set_mocap_pos(model, data, f"coil_{i}", (cx, 0.0, cz))

        # Block sits just below the bottom coil.
        block_z = CEILING_Z - 0.06 - (NATURAL_LEN + current_ext) - BLOCK_HALF
        set_mocap_pos(model, data, "block", (0.0, 0.0, block_z))

    frames = render_frames_with_state(
        mjcf, state_fn, n_frames=n_frames, width=width, height=height,
    )
    return frames, fps
