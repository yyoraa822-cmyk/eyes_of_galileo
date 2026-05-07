"""Snell's refraction scenario.

Hidden law:  n₁ sin(θ₁) = n₂ sin^α(θ₂)   (true α = 1)
Equivalently:  sin(θ₂) = (n₁/n₂)^(1/α) · sin^(1/α)(θ₁)

Historical instrument: Snellius (~1621) used a semicircular glass block
with a protractor to measure angles of incidence and refraction.

The agent varies the angle of incidence and observes how the refracted
ray bends, then discovers the exponent in Snell's law.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class Refraction(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 n1: float = 1.0, n2: float = 1.5):
        super().__init__(alpha=alpha, seed=seed)
        self._n1 = n1
        self._n2 = n2

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Snell's Refraction",
            slug="refraction",
            description=(
                "A light ray enters a glass block at a variable angle. "
                "You can change the angle of incidence and observe the "
                "angle of refraction. Discover the law relating them."
            ),
            control_var="angle_incidence_deg",
            control_label="Angle of incidence (degrees)",
            observable_label="Angle of refraction (degrees)",
            true_exponent=1.0,
            law_template="n₁ sin(θ₁) = n₂ sin^α(θ₂)",
            historical_instrument=(
                "Semicircular glass block with protractor (Snellius, ~1621)"
            ),
        )

    @property
    def default_controls(self) -> list[float]:
        return [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    def _refracted_angle(self, theta1_deg: float) -> float:
        """Compute θ₂ in degrees from θ₁ using the counterfactual law."""
        theta1 = np.radians(theta1_deg)
        sin_theta1 = np.sin(theta1)
        ratio = self._n1 / self._n2
        # n₁ sin(θ₁) = n₂ sin^α(θ₂)
        # sin^α(θ₂) = (n₁/n₂) sin(θ₁)
        # sin(θ₂) = [(n₁/n₂) sin(θ₁)]^(1/α)
        val = ratio * sin_theta1
        val = np.clip(val, 0, 1)
        if self.alpha == 0:
            return 0.0
        sin_theta2 = np.clip(val ** (1.0 / self.alpha), 0, 1)
        return np.degrees(np.arcsin(sin_theta2))

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float = 1.0) -> dict[str, Any]:
        theta1 = control_value
        theta2 = self._refracted_angle(theta1)
        return {
            "times": np.array([0.0]),
            "theta1_deg": theta1,
            "theta2_deg": theta2,
            "n1": self._n1,
            "n2": self._n2,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (6, 7),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        theta1 = sim_data["theta1_deg"]
        theta2 = sim_data["theta2_deg"]
        t1_rad = np.radians(theta1)
        t2_rad = np.radians(theta2)

        fig, ax = make_clean_fig(figsize=figsize)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")

        if clean:
            ax.set_title("Experiment")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.set_title(f"Refraction   θ₁ = {theta1:.1f}°")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        # interface (horizontal line at y=0)
        ax.axhline(0, color="#444444", linewidth=2)

        # media labels
        if not clean:
            ax.text(-2.5, 2.5, f"n₁ = {self._n1:.1f}  (air)",
                    fontsize=10, color="#4488cc")
            ax.text(-2.5, -2.5, f"n₂ = {self._n2:.1f}  (glass)",
                    fontsize=10, color="#cc8844")

        # upper medium background
        ax.fill_between([-3, 3], 0, 3, color="#e8f0ff", alpha=0.4)
        # lower medium background
        ax.fill_between([-3, 3], -3, 0, color="#fff0e0", alpha=0.4)

        # normal (dashed vertical)
        ax.plot([0, 0], [-2.8, 2.8], "--", color="#aaaaaa", linewidth=1)
        if not clean:
            ax.text(0.1, 2.6, "normal", fontsize=7, color="#aaaaaa")

        # incident ray (from upper-left to origin)
        ray_len = 2.5
        ix = -ray_len * np.sin(t1_rad)
        iy = ray_len * np.cos(t1_rad)
        ax.annotate("", xy=(0, 0), xytext=(ix, iy),
                     arrowprops=dict(arrowstyle="->,head_width=0.15",
                                     color="#2266cc", linewidth=2))

        # refracted ray (from origin to lower-right)
        rx = ray_len * np.sin(t2_rad)
        ry = -ray_len * np.cos(t2_rad)
        ax.annotate("", xy=(rx, ry), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->,head_width=0.15",
                                     color="#cc6622", linewidth=2))

        # angle arcs
        if theta1 > 1:
            arc1 = patches.Arc((0, 0), 1.2, 1.2, angle=0,
                               theta1=90 - theta1, theta2=90,
                               color="#2266cc", linewidth=1.5)
            ax.add_patch(arc1)
            if not clean:
                ax.text(-0.3, 0.8, f"θ₁={theta1:.1f}°",
                        fontsize=8, color="#2266cc")

        if theta2 > 1:
            arc2 = patches.Arc((0, 0), 1.0, 1.0, angle=0,
                               theta1=270, theta2=270 + theta2,
                               color="#cc6622", linewidth=1.5)
            ax.add_patch(arc2)
            if not clean:
                ax.text(0.3, -0.8, f"θ₂={theta2:.1f}°",
                        fontsize=8, color="#cc6622")

        # protractor markings on the semicircle
        for deg in range(0, 91, 10):
            rad = np.radians(deg)
            r_inner, r_outer = 2.2, 2.4
            # upper half
            ax.plot([r_inner * np.sin(rad), r_outer * np.sin(rad)],
                    [r_inner * np.cos(rad), r_outer * np.cos(rad)],
                    color="#cccccc", linewidth=0.5)
            ax.plot([-r_inner * np.sin(rad), -r_outer * np.sin(rad)],
                    [r_inner * np.cos(rad), r_outer * np.cos(rad)],
                    color="#cccccc", linewidth=0.5)

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._refracted_angle(control_value)
