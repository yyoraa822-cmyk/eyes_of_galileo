"""Stokes' drag / terminal velocity scenario.

Hidden law:  v_terminal ∝ r^α   (textbook α = 2)

The agent sees spheres of different radii falling through a viscous
fluid, each at their terminal velocity. The control variable is the
sphere radius; the observable is the terminal velocity.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class Viscosity(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 C: float = 1.0):
        self._C = C
        super().__init__(alpha=alpha, seed=seed)

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Stokes Terminal Velocity",
            slug="viscosity",
            description=(
                "Spheres of different sizes fall through a viscous fluid. "
                "Each reaches a terminal (constant) velocity that depends "
                "on its radius. Discover how terminal velocity scales with radius."
            ),
            control_var="radius",
            control_label="Sphere radius (mm)",
            observable_label="Terminal velocity (mm/s)",
            true_exponent=2.0,
            law_template="v_term ∝ r^α",
            historical_instrument="Stokes' viscometry experiments (1851)",
        )

    @property
    def default_controls(self) -> list[float]:
        return [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    def _terminal_velocity(self, r: float) -> float:
        return self._C * r ** self.alpha

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float = 1.0) -> dict[str, Any]:
        r = max(0.1, control_value)
        v = self._terminal_velocity(r)
        return {
            "times": np.array([0.0]),
            "radius": r,
            "velocity": v,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (5, 8),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        radius = sim_data["radius"]
        velocity = sim_data["velocity"]
        v_max = self._terminal_velocity(5.0)

        fig, ax = make_clean_fig(figsize=figsize)

        tube_w = 6.0
        tube_h = 12.0
        ax.set_xlim(-1, tube_w + 1)
        ax.set_ylim(-1, tube_h + 1)
        ax.set_aspect("equal")

        ax.add_patch(plt.Rectangle(
            (0, 0), tube_w, tube_h,
            facecolor="#d4e6f1", edgecolor="#555555", linewidth=2))

        for yy in np.arange(0, tube_h, 0.4):
            ax.plot([0, tube_w], [yy, yy], "-", color="#c0d8ea",
                    linewidth=0.3, alpha=0.5)

        sphere_x = tube_w / 2
        sphere_y = tube_h * 0.6
        sphere_screen_r = 0.2 + radius * 0.15

        sphere = plt.Circle((sphere_x, sphere_y), sphere_screen_r,
                            facecolor="#cc5533", edgecolor="#882211",
                            linewidth=1.5, zorder=5)
        ax.add_patch(sphere)

        arrow_len = max(0.3, 3.0 * velocity / v_max)
        ax.annotate("", xy=(sphere_x, sphere_y - sphere_screen_r - arrow_len),
                    xytext=(sphere_x, sphere_y - sphere_screen_r),
                    arrowprops=dict(arrowstyle="-|>", color="#00aa44",
                                   linewidth=2.5, mutation_scale=15),
                    zorder=6)

        scale_x = tube_w + 0.3
        for frac in np.linspace(0, tube_h, 13):
            tick_len = 0.2 if int(frac) % 2 == 0 else 0.1
            ax.plot([scale_x, scale_x + tick_len], [frac, frac],
                    color="#888888", linewidth=0.5)

        if clean:
            ax.set_title("Experiment")
        else:
            ax.set_title(f"Terminal velocity  r={radius:.1f} mm")
        ax.set_xticks([])
        ax.set_yticks([])

        if not clean:
            ax.text(sphere_x, sphere_y - sphere_screen_r - arrow_len - 0.5,
                    f"v={velocity:.2f}", ha="center", fontsize=9, color="#00aa44")

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._terminal_velocity(control_value)
