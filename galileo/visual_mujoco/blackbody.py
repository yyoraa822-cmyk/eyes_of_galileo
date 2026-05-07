"""MuJoCo visualisation for s21_weber (Wien's displacement law).

Langley's bolometer (1878) spectroscopy: a heated metal sphere or
filament radiates with a peak wavelength that shifts with temperature.
The agent dials in T and observes the dominant emission colour to
discover lambda_peak ∝ T^alpha (textbook alpha = -1).

Apparatus mapping: a rounded brass-mounted cube glows a colour
matching its blackbody temperature. A thin spectrum strip below the
sample shows visible-light bins (380→750 nm) with a green pointer
hovering at the simulated peak wavelength. A brass ring at the back
of the apparatus represents the bolometer's thermopile collector.
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from PIL import Image

from ._base import (
    blackbody_to_rgb,
    make_scene_mjcf,
    render_frames_with_state,
    set_geom_rgba,
    set_mocap_pos,
)

SAMPLE_X = -1.7
SAMPLE_Z = 0.85
SAMPLE_R = 0.32
SPECTRUM_Z = 1.55      # high so it's right in frame next to the sample
SPECTRUM_LEN = 2.4     # x-extent of spectrum strip
SPECTRUM_X0 = -SPECTRUM_LEN / 2.0
SPECTRUM_Y = 0.0       # same depth as the sample


def _wavelength_to_rgb(lam_nm: float) -> tuple[float, float, float]:
    """Match the matplotlib rendering's visible-light palette (cf.
    galileo/scenarios/blackbody.py:_wavelength_to_rgb)."""
    if lam_nm < 380:
        return (0.40, 0.00, 0.60)
    if lam_nm < 440:
        t = (lam_nm - 380) / 60
        return (0.40 * (1 - t), 0.00, 0.60 + 0.40 * t)
    if lam_nm < 490:
        t = (lam_nm - 440) / 50
        return (0.00, t, 1.00)
    if lam_nm < 510:
        t = (lam_nm - 490) / 20
        return (0.00, 1.00, 1.00 - t)
    if lam_nm < 580:
        t = (lam_nm - 510) / 70
        return (t, 1.00, 0.00)
    if lam_nm < 645:
        t = (lam_nm - 580) / 65
        return (1.00, 1.00 - t, 0.00)
    if lam_nm < 780:
        return (1.00, 0.00, 0.00)
    return (0.50, 0.00, 0.00)


def _build_mjcf() -> str:
    extra_assets = """
  <asset>
    <material name="brass" rgba="0.78 0.60 0.28 1" specular="0.40"
              shininess="0.5"/>
    <material name="bench" rgba="0.42 0.34 0.27 1" specular="0.10"
              shininess="0.2"/>
    <material name="sample" rgba="0.85 0.45 0.20 1" emission="0.7"/>
    <material name="strip_frame" rgba="0.22 0.22 0.25 1"/>
    <material name="pointer" rgba="0.30 0.95 0.40 1" emission="0.6"/>
  </asset>
"""
    parts: list[str] = []

    # Bench / mounting block.
    parts.append(
        f'    <body name="bench" pos="0 0 0.05">\n'
        f'      <geom type="box" size="2.0 0.55 0.05" material="bench"/>\n'
        f'    </body>'
    )
    # Bolometer thermopile collector — large brass ring behind the sample.
    parts.append(
        f'    <body name="collector" pos="0 0.55 {SAMPLE_Z:.3f}"\n'
        f'          quat="0.7071 0.7071 0 0">\n'
        f'      <geom type="cylinder" size="0.50 0.04" material="brass"/>\n'
        f'    </body>'
    )
    # Stand pillar for the heated sample.
    parts.append(
        f'    <body name="stand" pos="{SAMPLE_X:.3f} 0 {SAMPLE_Z / 2:.3f}">\n'
        f'      <geom type="cylinder" size="0.06 {SAMPLE_Z / 2:.3f}"\n'
        f'            material="brass"/>\n'
        f'    </body>'
    )
    # The heated sample: a sphere whose colour is updated each frame.
    parts.append(
        f'    <body name="sample" pos="{SAMPLE_X:.3f} 0 {SAMPLE_Z:.3f}">\n'
        f'      <geom name="sample_geom" type="sphere" size="{SAMPLE_R:.3f}"\n'
        f'            material="sample"/>\n'
        f'    </body>'
    )

    # Spectrum strip placed in front of the bench so it's not occluded.
    n_bins = 50
    for i in range(n_bins):
        lam = 380 + (750 - 380) * (i + 0.5) / n_bins
        r, g, bcol = _wavelength_to_rgb(lam)
        x = SPECTRUM_X0 + SPECTRUM_LEN * (i + 0.5) / n_bins
        bin_w = SPECTRUM_LEN / n_bins / 2.0
        parts.append(
            f'    <body name="bin_{i}" pos="{x:.3f} {SPECTRUM_Y:.3f} {SPECTRUM_Z:.3f}">\n'
            f'      <geom type="box" size="{bin_w:.3f} 0.06 0.10"\n'
            f'            rgba="{r:.3f} {g:.3f} {bcol:.3f} 1"/>\n'
            f'    </body>'
        )
    parts.append(
        f'    <body name="strip_frame" pos="0 {SPECTRUM_Y + 0.07:.3f} {SPECTRUM_Z:.3f}">\n'
        f'      <geom type="box" size="{SPECTRUM_LEN / 2 + 0.06:.3f} 0.04 0.12"\n'
        f'            material="strip_frame"/>\n'
        f'    </body>'
    )
    parts.append(
        f'    <body name="peak_pointer" mocap="true"\n'
        f'          pos="0 {SPECTRUM_Y - 0.10:.3f} {SPECTRUM_Z + 0.22:.3f}">\n'
        f'      <geom name="peak_pointer_geom" type="capsule" size="0.030"\n'
        f'            fromto="0 0 0  0 0 -0.18" material="pointer"/>\n'
        f'    </body>'
    )
    extra = "\n".join(parts)

    return make_scene_mjcf(
        cam_pos=(0.5, -5.5, 0.9),
        cam_xyaxes=(1, 0, 0, 0, -0.10, 0.99),
        floor_size=(6.0, 5.0, 0.05),
        light_pos=(1.0, -2.5, 4.5),
        extra_assets=extra_assets,
        extra_worldbody=extra,
    )


def render_animation(
    scenario: Any, control_value: float, *,
    n_frames: int = 32, width: int = 960, height: int = 600,
) -> tuple[list[Image.Image], int]:
    """Animate the heated sample warming up from a baseline temperature
    toward `control_value` (Kelvin), then hold."""
    sim = scenario.simulate(float(control_value))
    T_target = float(sim["temperature"])
    lam_target = float(sim["peak_wavelength"])
    fps = 8
    SETTLE_TAU = 0.65
    T_BASELINE = 1000.0  # start dim red

    mjcf = _build_mjcf()

    def state_fn(tau: float, model: mujoco.MjModel,
                 data: mujoco.MjData) -> None:
        phase = min(1.0, tau / SETTLE_TAU)
        T_cur = T_BASELINE + phase * (T_target - T_BASELINE)
        lam_cur = float(scenario.get_observable(T_cur))

        r, g, b = blackbody_to_rgb(T_cur)
        set_geom_rgba(model, "sample_geom", (r, g, b, 1.0))

        # Pointer along x-axis of spectrum strip. Clamp lambda into the
        # visible band [380, 750] for placement (peaks outside still
        # appear at one of the strip's ends).
        lam_clip = float(np.clip(lam_cur, 380.0, 750.0))
        frac = (lam_clip - 380.0) / (750.0 - 380.0)
        x_ptr = SPECTRUM_X0 + frac * SPECTRUM_LEN
        set_mocap_pos(model, data, "peak_pointer",
                      (x_ptr, SPECTRUM_Y - 0.10, SPECTRUM_Z + 0.22))

    frames = render_frames_with_state(
        mjcf, state_fn, n_frames=n_frames, width=width, height=height,
    )
    return frames, fps
