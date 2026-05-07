"""Coulomb's law scenario.

Hidden law:  F ∝ d^α   (textbook α = −2, inverse-square law)

The agent sees two charged spheres separated by varying distances,
with force arrows indicating the electrostatic interaction strength.
The control variable is the separation distance; the observable is
the force magnitude.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class Coulomb(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 k_eff: float = 10.0):
        self._k = k_eff
        super().__init__(alpha=alpha, seed=seed)

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Coulomb's Law",
            slug="coulomb",
            description=(
                "Two charged objects are placed at different distances. "
                "Force arrows indicate the interaction strength. "
                "Discover how the force depends on separation distance."
            ),
            control_var="distance",
            control_label="Separation distance (m)",
            observable_label="Force magnitude (N)",
            true_exponent=-2.0,
            law_template="F ∝ d^α",
            historical_instrument="Coulomb's torsion balance (1785)",
        )

    @property
    def default_controls(self) -> list[float]:
        return [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

    def _force(self, d: float) -> float:
        d = max(d, 0.1)
        return self._k * d ** self.alpha

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float = 1.0) -> dict[str, Any]:
        d = max(0.1, control_value)
        f = self._force(d)
        return {
            "times": np.array([0.0]),
            "distance": d,
            "force": f,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (8, 4),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        d = sim_data["distance"]
        f = sim_data["force"]
        f_max = self._force(0.5)

        fig, ax = make_clean_fig(figsize=figsize)
        ax.set_xlim(-2, 14)
        ax.set_ylim(-2, 4)
        ax.set_aspect("equal")

        if clean:
            ax.set_title("Experiment")
        else:
            ax.set_title(f"Coulomb interaction  d={d:.1f}")
        ax.set_xticks([])
        ax.set_yticks([])

        q1_x, q2_x = 2.0, 2.0 + d * 1.2
        q_y = 1.0
        r_q = 0.5

        c1 = plt.Circle((q1_x, q_y), r_q, facecolor="#cc4444",
                        edgecolor="#881111", linewidth=2)
        c2 = plt.Circle((q2_x, q_y), r_q, facecolor="#4444cc",
                        edgecolor="#111188", linewidth=2)
        ax.add_patch(c1)
        ax.add_patch(c2)
        ax.text(q1_x, q_y, "+", ha="center", va="center",
                fontsize=16, fontweight="bold", color="white")
        ax.text(q2_x, q_y, "−", ha="center", va="center",
                fontsize=16, fontweight="bold", color="white")

        arrow_len = max(0.3, 3.0 * (f / f_max))
        ax.annotate("", xy=(q1_x + r_q + arrow_len, q_y + 1.2),
                    xytext=(q1_x + r_q, q_y + 1.2),
                    arrowprops=dict(arrowstyle="->", color="#cc4444",
                                   linewidth=2.5))
        ax.annotate("", xy=(q2_x - r_q - arrow_len, q_y + 1.2),
                    xytext=(q2_x - r_q, q_y + 1.2),
                    arrowprops=dict(arrowstyle="->", color="#4444cc",
                                   linewidth=2.5))

        mid_x = (q1_x + q2_x) / 2
        ax.annotate("", xy=(q2_x - r_q - 0.1, q_y - 1.0),
                    xytext=(q1_x + r_q + 0.1, q_y - 1.0),
                    arrowprops=dict(arrowstyle="<->", color="#888888",
                                   linewidth=1.2))

        if not clean:
            ax.text(mid_x, q_y - 1.5, f"d = {d:.1f}",
                    ha="center", fontsize=9, color="#666666")

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._force(control_value)
