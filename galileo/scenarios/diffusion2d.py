"""2D diffusion (random walk) scenario.

Hidden law:  r_spread ∝ t^α   (textbook α = 0.5, anomalous diffusion)

The agent sees particles spreading outward from a central point.
The control variable is time; the observable is the RMS spread radius.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class Diffusion2D(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 n_particles: int = 200, D: float = 1.0):
        self._n_particles = n_particles
        self._D = D
        super().__init__(alpha=alpha, seed=seed)

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="2D Diffusion",
            slug="diffusion2d",
            description=(
                "Particles released from a central point spread outward. "
                "Observe the spread pattern at different times to discover "
                "how the spread radius depends on time."
            ),
            control_var="time",
            control_label="Observation time (s)",
            observable_label="RMS spread radius",
            true_exponent=0.5,
            law_template="r_spread ∝ t^α",
            historical_instrument="Brown's pollen observations (1827)",
        )

    @property
    def default_controls(self) -> list[float]:
        return [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    def _spread_radius(self, t: float) -> float:
        return np.sqrt(self._D) * t ** self.alpha

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float = 1.0) -> dict[str, Any]:
        t = max(0.01, control_value)
        r_spread = self._spread_radius(t)

        rng = np.random.default_rng(self.rng.integers(0, 2**31) + int(t * 1000))
        angles = rng.uniform(0, 2 * np.pi, self._n_particles)
        radii = rng.exponential(r_spread, self._n_particles)

        x = radii * np.cos(angles)
        y = radii * np.sin(angles)

        return {
            "times": np.array([t]),
            "time": t,
            "x": x,
            "y": y,
            "r_spread": r_spread,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (6, 6),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        t = sim_data["time"]
        x = sim_data["x"]
        y = sim_data["y"]
        r_spread = sim_data["r_spread"]

        fig, ax = make_clean_fig(figsize=figsize)

        lim = max(r_spread * 3, 2.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")

        if clean:
            ax.set_title("Experiment")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_title(f"2D Diffusion  t={t:.1f}s")
            ax.set_xticks([])
            ax.set_yticks([])

        ax.scatter(x, y, s=8, c="#3366cc", alpha=0.5, edgecolors="none")

        ax.plot(0, 0, "+", color="#cc3333", markersize=15,
                markeredgewidth=2, zorder=5)

        circle = plt.Circle((0, 0), r_spread, fill=False,
                            edgecolor="#00aa44", linewidth=2,
                            linestyle="--", zorder=4)
        ax.add_patch(circle)

        ax.plot([0, r_spread], [0, 0], "-", color="#00aa44",
                linewidth=1.5, zorder=4)
        ax.plot(r_spread, 0, "o", color="#00aa44", markersize=8, zorder=5)

        if not clean:
            ax.text(r_spread / 2, -0.3 * lim / 5,
                    f"r={r_spread:.2f}", fontsize=9,
                    ha="center", color="#00aa44")

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._spread_radius(control_value)
