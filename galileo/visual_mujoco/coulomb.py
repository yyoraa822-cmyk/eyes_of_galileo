"""MuJoCo visualisation for s19_coulomb.

Physical setup (per demo.html intent + user spec):

       __________________________________________
      ┌──────────────────────────────────────────┐    <- horizontal beam
      │ │                                      │ │
      │ │ post                              post │
      │ │                                      │ │
      │ │     ┌── string (mocap, rotates)      │ │
      │ │     │                                │ │
      │ │     │           ┌── rigid brass rod   │ │
      │ │     │           │                    │ │
      │ │   (◯)~~~ d ~~~(●)                    │ │
      │ │   left          right                  │ │
      │ │   moves         fixed                  │ │
      └─┴─[base]─────────────────────────[base]──┘    <- floor

Constraints honored:
  - Both balls sit at the SAME z (z = rest_z = 1.0). Right ball is
    hung on a rigid vertical brass rod from the beam.
  - Posts placed far enough left/right that the swinging left ball
    NEVER crosses past the left post (theta_eq capped at 45°, plus
    overshoot stays within a configurable margin).
  - Animation: damped swing from vertical (theta=0) to theta_eq,
    then a clear hold-at-rest segment.
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


def _set_mocap_quat_y(model: mujoco.MjModel, data: mujoco.MjData,
                      body_name: str, theta_y: float) -> None:
    """Rotate a mocap body by `theta_y` radians around world y-axis."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return
    mocap_id = model.body_mocapid[bid]
    if mocap_id < 0:
        return
    half = 0.5 * theta_y
    data.mocap_quat[mocap_id] = np.array(
        [np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float64)


# Equilibrium deflection angle from real Coulomb-pendulum physics:
# the suspended ball settles where F_coulomb (horizontal) is balanced
# by gravity*tan(θ). So θ = atan(F(d) / mg). F is read from the
# scenario, so any alpha (Newton or non-Newton) propagates naturally:
# alpha=-2 reproduces the textbook deflection curve, alpha=-1 gives
# a slower fall-off, etc.
#
# `mg_eff` is calibrated so the median control in the scenario's
# `default_controls` pool gives a visually pleasing ~25 degree
# deflection. This makes apparatus look reasonable across alphas
# without forcing the agent to see a single fixed angle.

# Visual cap chosen so MAX swing position (theta_eq * (1 + overshoot))
# stays inside the safety margin between left ball anchor and left post.
# With overshoot ≈ 30%, capping theta_eq at 45° keeps peak ≤ ~58°,
# i.e. horizontal deflection ≤ L*sin(58°) ≈ 1.19 m. Post sits at 1.5 m
# from anchor, so a 0.3 m clearance remains.
THETA_EQ_CAP = np.deg2rad(45.0)
POST_OFFSET = 1.5
STRING_LEN = 1.4
BEAM_Z = 2.4
REST_Z = BEAM_Z - STRING_LEN  # 1.0


def _calibrate_mg(scenario: Any) -> float:
    pool = scenario.default_controls
    d_anchor = float(pool[len(pool) // 2])
    f_anchor = float(scenario.get_observable(d_anchor))
    # Want θ_anchor ≈ 25° at d=anchor, so mg = F / tan(25°)
    return max(1e-3, f_anchor / np.tan(np.deg2rad(25.0)))


def _equilibrium_deflection(scenario: Any, d: float, mg_eff: float
                            ) -> float:
    f = float(scenario.get_observable(d))
    theta = float(np.arctan(f / mg_eff))
    return min(theta, THETA_EQ_CAP)


def _build_mjcf(d: float) -> str:
    # Right ball is FIXED, mounted on a rigid brass rod hanging from
    # the beam down to z = REST_Z so it sits at the SAME horizontal
    # line as the left ball at rest.
    right_x = d / 2.0
    left_anchor_x = -d / 2.0
    rod_top_z = BEAM_Z - 0.05      # rod top sits just under beam
    rod_bot_z = REST_Z             # rod bottom = ball center

    # Posts placed POST_OFFSET m outboard of each ball anchor so the
    # swinging left ball never crosses past the left post.
    post_left_x = left_anchor_x - POST_OFFSET
    post_right_x = right_x + POST_OFFSET
    beam_half = (post_right_x - post_left_x) / 2.0 + 0.05
    beam_center_x = (post_left_x + post_right_x) / 2.0

    # Camera framing scales with d so apparatus stays well-composed.
    cam_dist = max(6.5, (post_right_x - post_left_x) * 0.95 + 3.5)
    extra_assets = """
  <asset>
    <texture name="brass" type="2d" builtin="flat"
             rgb1="0.55 0.42 0.22" width="64" height="64"/>
    <material name="brass" texture="brass" specular="0.30"
              shininess="0.4" reflectance="0.05"/>
    <material name="post" rgba="0.42 0.34 0.27 1" specular="0.10"
              shininess="0.1"/>
    <material name="string" rgba="0.85 0.85 0.85 1"/>
  </asset>
"""
    extra = f"""
    <body name="post_left" pos="{post_left_x:.3f} 0 {BEAM_Z/2:.3f}">
      <geom type="cylinder" size="0.06 {BEAM_Z/2:.3f}" material="post"/>
    </body>
    <body name="post_right" pos="{post_right_x:.3f} 0 {BEAM_Z/2:.3f}">
      <geom type="cylinder" size="0.06 {BEAM_Z/2:.3f}" material="post"/>
    </body>
    <body name="beam" pos="{beam_center_x:.3f} 0 {BEAM_Z + 0.05:.3f}">
      <geom type="box" size="{beam_half:.3f} 0.06 0.05" material="brass"/>
    </body>
    <body name="base_left" pos="{post_left_x:.3f} 0 0.05">
      <geom type="box" size="0.20 0.20 0.05" material="brass"/>
    </body>
    <body name="base_right" pos="{post_right_x:.3f} 0 0.05">
      <geom type="box" size="0.20 0.20 0.05" material="brass"/>
    </body>
    <body name="charge_right" pos="{right_x:.3f} 0 {rod_bot_z:.3f}">
      <geom type="capsule" size="0.04"
            fromto="0 0 {(rod_top_z - rod_bot_z):.3f}  0 0 0.18"
            material="brass"/>
      <geom name="ball_right" type="sphere" size="0.18"
            rgba="0.95 0.30 0.30 1"/>
    </body>
    <body name="suspension" pos="{left_anchor_x:.3f} 0 {BEAM_Z - 0.05:.3f}">
      <geom name="anchor_block" type="box" size="0.05 0.05 0.04"
            material="brass"/>
    </body>
    <body name="string_body" mocap="true"
          pos="{left_anchor_x:.3f} 0 {BEAM_Z - 0.05:.3f}">
      <geom name="string_geom" type="capsule" size="0.012"
            fromto="0 0 0  0 0 -{STRING_LEN:.3f}"
            material="string"/>
    </body>
    <body name="charge_left" mocap="true"
          pos="{left_anchor_x:.3f} 0 {REST_Z:.3f}">
      <geom name="ball_left" type="sphere" size="0.18"
            rgba="0.30 0.45 0.95 1"/>
    </body>
    """
    return make_scene_mjcf(
        cam_pos=(0.0, -cam_dist, 0.55),
        cam_xyaxes=(1, 0, 0, 0, -0.13, 0.99),
        floor_size=(max(8.0, post_right_x - post_left_x + 3.0), 6.0, 0.05),
        light_pos=(2.0, -3.0, 4.5),
        extra_assets=extra_assets,
        extra_worldbody=extra,
    )


def render_animation(
    scenario: Any, control_value: float, *,
    n_frames: int = 32, width: int = 960, height: int = 600,
) -> tuple[list[Image.Image], int]:
    """Render coulomb-pendulum animation.

    Default n_frames=32 @ 8 fps = 4 sec total:
        tau ∈ [0, 0.65)  -> swinging from vertical to theta_eq (damped)
        tau ∈ [0.65, 1]  -> hold at theta_eq (settled)
    """
    sim = scenario.simulate(float(control_value))
    d = float(sim["distance"])
    left_anchor_x = -d / 2.0
    anchor_z = BEAM_Z - 0.05

    mg_eff = _calibrate_mg(scenario)
    theta_eq = _equilibrium_deflection(scenario, d, mg_eff)
    mjcf = _build_mjcf(d)

    fps = 8
    total_t = n_frames / float(fps)
    SETTLE_TAU = 0.65          # fraction of timeline spent swinging
    OMEGA = 2.0 * np.pi * 1.1  # ≈1.1 Hz visible swing
    ZETA = 0.45                # damping ratio (overshoot ~ 20%)

    def state_fn(tau: float, model: mujoco.MjModel,
                 data: mujoco.MjData) -> None:
        if tau < SETTLE_TAU:
            t = tau * total_t
            # Damped approach from theta=0 to theta_eq:
            #   theta(t) = theta_eq * [1 - exp(-zeta*omega*t) * cos(omega_d*t)]
            env = np.exp(-ZETA * OMEGA * t)
            omega_d = OMEGA * np.sqrt(max(1e-6, 1.0 - ZETA * ZETA))
            theta = theta_eq * (1.0 - env * np.cos(omega_d * t))
        else:
            theta = theta_eq

        # Pendulum kinematics around the beam anchor; deflection is
        # negative-x (ball pushed away from right charge):
        #   bx = anchor_x - L sin θ
        #   bz = anchor_z - L cos θ
        bx = left_anchor_x - STRING_LEN * np.sin(theta)
        bz = anchor_z - STRING_LEN * np.cos(theta)
        set_mocap_pos(model, data, "charge_left", (bx, 0.0, bz))
        set_mocap_pos(model, data, "string_body",
                      (left_anchor_x, 0.0, anchor_z))
        # Rotate string capsule around y-axis by +theta so it tracks
        # the deflected ball: (0,0,-1) -> (-sin θ, 0, -cos θ).
        _set_mocap_quat_y(model, data, "string_body", theta)

    frames = render_frames_with_state(
        mjcf, state_fn, n_frames=n_frames, width=width, height=height,
    )
    return frames, fps
