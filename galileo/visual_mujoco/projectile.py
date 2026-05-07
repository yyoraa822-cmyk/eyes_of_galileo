"""MuJoCo visualisation for s6_launch (projectile motion).

Galileo's table-edge experiment (1638): a ball is launched at a fixed
angle with varying initial speed; the agent observes the horizontal
range to discover R ∝ v^alpha (textbook alpha = 2).

Apparatus mapping: a brass-mounted spring-loaded launcher fires a
ball at 45 degrees. The ball traces a parabolic path through the
air; we drop a strobe of small after-image spheres along the
trajectory so the trajectory is readable in a single static frame.
A landing marker (red diamond) sits at the predicted range.
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

LAUNCHER_X = -1.5
N_STROBES = 12   # number of after-image dots along the path
BALL_R = 0.10


def _build_mjcf(R: float, max_y: float) -> str:
    extra_assets = """
  <asset>
    <material name="brass" rgba="0.78 0.60 0.28 1" specular="0.40"
              shininess="0.5"/>
    <material name="ball" rgba="0.30 0.55 0.95 1" specular="0.40"
              shininess="0.5"/>
    <material name="ball_ghost" rgba="0.30 0.55 0.95 0.45"/>
    <material name="bench" rgba="0.42 0.34 0.27 1" specular="0.10"
              shininess="0.2"/>
    <material name="land" rgba="0.95 0.20 0.20 1" emission="0.5"/>
    <material name="launcher" rgba="0.55 0.55 0.62 1" specular="0.35"
              shininess="0.5"/>
  </asset>
"""
    parts: list[str] = []

    # Long ground track from launcher to past the landing point.
    parts.append(
        f'    <body name="track" pos="{(LAUNCHER_X + R) / 2:.3f} 0 0.03">\n'
        f'      <geom type="box" size="{(R + 1.5) / 2:.3f} 0.40 0.03"\n'
        f'            material="bench"/>\n'
        f'    </body>'
    )
    # Launcher: brass block + tilted barrel pointing up-right at 45 deg.
    parts.append(
        f'    <body name="launcher_base" pos="{LAUNCHER_X:.3f} 0 0.18">\n'
        f'      <geom type="box" size="0.15 0.18 0.10" material="launcher"/>\n'
        f'    </body>'
    )
    # Barrel: a capsule rotated 45 deg around y, pointing up-right.
    # quat for 45 deg rot around y: (cos22.5, 0, sin22.5, 0).
    half_a = np.deg2rad(22.5)
    qw = np.cos(half_a)
    qy = np.sin(half_a)
    parts.append(
        f'    <body name="barrel" pos="{LAUNCHER_X + 0.12:.3f} 0 0.36"\n'
        f'          quat="{qw:.4f} 0 {qy:.4f} 0">\n'
        f'      <geom type="capsule" size="0.045"\n'
        f'            fromto="0 0 -0.20  0 0 0.30" material="brass"/>\n'
        f'    </body>'
    )

    # Strobe ghost balls placed each frame via mocap.
    for i in range(N_STROBES):
        parts.append(
            f'    <body name="ghost_{i}" mocap="true" pos="0 0 -10">\n'
            f'      <geom name="ghost_geom_{i}" type="sphere" '
            f'size="{BALL_R * 0.85:.3f}" material="ball_ghost"/>\n'
            f'    </body>'
        )

    # Live ball that "lives" at the current animated position.
    parts.append(
        f'    <body name="ball_now" mocap="true" pos="{LAUNCHER_X:.3f} 0 0.36">\n'
        f'      <geom type="sphere" size="{BALL_R:.3f}" material="ball"/>\n'
        f'    </body>'
    )

    # Landing marker (diamond shape via small box rotated 45 deg).
    parts.append(
        f'    <body name="landing" pos="{LAUNCHER_X + R:.3f} 0 0.10"\n'
        f'          quat="0.9239 0 0.3827 0">\n'
        f'      <geom type="box" size="0.10 0.10 0.04" material="land"/>\n'
        f'    </body>'
    )

    extra = "\n".join(parts)

    apex = max_y / 2 + 0.4
    return make_scene_mjcf(
        cam_pos=(LAUNCHER_X + R / 2, -max(5.0, R + 2.0), apex),
        cam_xyaxes=(1, 0, 0, 0, -0.10, 0.99),
        floor_size=(max(R, 6.0) + 4.0, 6.0, 0.05),
        light_pos=(LAUNCHER_X + R / 2, -3.0, 4.5),
        extra_assets=extra_assets,
        extra_worldbody=extra,
    )


def render_animation(
    scenario: Any, control_value: float, *,
    n_frames: int = 32, width: int = 1200, height: int = 600,
) -> tuple[list[Image.Image], int]:
    """Animate the ball flying from launcher to landing.

    During the swing phase (tau in [0, 0.65)) the ball travels along
    its parabolic arc; during the hold phase the ball sits at the
    landing point and the strobe afterimages stay in place.
    """
    sim = scenario.simulate(float(control_value))
    times = np.asarray(sim["times"])
    xs = np.asarray(sim["x"])
    ys = np.asarray(sim["y"])
    R = float(sim["range"])
    max_y = float(np.max(ys)) if len(ys) else 1.0

    fps = 8
    SETTLE_TAU = 0.65
    mjcf = _build_mjcf(R, max_y)

    # Pre-compute strobe positions evenly spaced along the trajectory.
    strobe_xy: list[tuple[float, float]] = []
    if len(xs) > 1:
        idx = np.linspace(0, len(xs) - 1, N_STROBES).astype(int)
        for k in range(N_STROBES):
            strobe_xy.append((float(xs[idx[k]]), float(ys[idx[k]])))
    else:
        for _ in range(N_STROBES):
            strobe_xy.append((0.0, 0.0))

    def state_fn(tau: float, model: mujoco.MjModel,
                 data: mujoco.MjData) -> None:
        if tau < SETTLE_TAU:
            phase = tau / SETTLE_TAU
            # Live ball moves continuously along the arc; ghost balls
            # appear progressively as the live ball passes them.
            i = int(phase * (len(xs) - 1)) if len(xs) > 1 else 0
            x_now = LAUNCHER_X + 0.12 + float(xs[i])
            y_now = float(ys[i]) + 0.36
            set_mocap_pos(model, data, "ball_now", (x_now, 0.0, y_now))

            # Reveal ghost balls up to the strobe index that the live
            # ball has reached. Hidden ghosts stay at z=-10.
            reached = int(phase * N_STROBES)
            for k in range(N_STROBES):
                if k <= reached:
                    sx, sy = strobe_xy[k]
                    set_mocap_pos(model, data, f"ghost_{k}",
                                  (LAUNCHER_X + 0.12 + sx, 0.0, sy + 0.36))
                else:
                    set_mocap_pos(model, data, f"ghost_{k}", (0.0, 0.0, -10.0))
        else:
            # Hold: full strobe trail visible, ball at landing.
            for k in range(N_STROBES):
                sx, sy = strobe_xy[k]
                set_mocap_pos(model, data, f"ghost_{k}",
                              (LAUNCHER_X + 0.12 + sx, 0.0, sy + 0.36))
            set_mocap_pos(model, data, "ball_now",
                          (LAUNCHER_X + 0.12 + R, 0.0, 0.36))

    frames = render_frames_with_state(
        mjcf, state_fn, n_frames=n_frames, width=width, height=height,
    )
    return frames, fps
