"""Stefan-Boltzmann law scenario.

Hidden law:  P ∝ T^α   (textbook α = 4)

The agent sees a glowing body whose brightness changes with temperature.
The control variable is temperature; the observable is the total
radiated power (shown as brightness of the object).
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class Stefan(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 sigma_eff: float = 1.0):
        self._sigma = sigma_eff
        super().__init__(alpha=alpha, seed=seed)

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Stefan-Boltzmann Radiation",
            slug="stefan",
            description=(
                "A body radiates energy. Its brightness (total power) "
                "changes dramatically with temperature. "
                "Discover how the radiated power depends on temperature."
            ),
            control_var="temperature",
            control_label="Temperature (arb. units)",
            observable_label="Radiated power (arb. units)",
            true_exponent=4.0,
            law_template="P ∝ T^α",
            historical_instrument="Stefan's thermopile measurements (1879)",
        )

    @property
    def default_controls(self) -> list[float]:
        return [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    def _power(self, T: float) -> float:
        return self._sigma * T ** self.alpha

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float = 1.0) -> dict[str, Any]:
        T = max(0.1, control_value)
        P = self._power(T)
        return {
            "times": np.array([0.0]),
            "temperature": T,
            "power": P,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (6, 6),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        T = sim_data["temperature"]
        P = sim_data["power"]
        P_ref = self._power(5.0)
        brightness = np.clip(P / P_ref, 0, 1)

        fig, ax = make_clean_fig(figsize=figsize)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_facecolor("#111122")
        fig.set_facecolor("#111122")

        if clean:
            ax.set_title("Experiment", color="white")
        else:
            ax.set_title(f"Radiation  T={T:.1f}", color="white")
        ax.set_xticks([])
        ax.set_yticks([])

        n_rings = 5
        for i in range(n_rings, 0, -1):
            r = 0.6 + i * 0.3
            alpha_ring = brightness * (0.6 / i)
            glow = plt.Circle((0, 0), r, facecolor="#ffaa44",
                             edgecolor="none", alpha=alpha_ring)
            ax.add_patch(glow)

        r_val = min(0.5 + 0.3 * brightness, 1.0)
        g_val = min(0.3 + 0.5 * brightness, 1.0)
        b_val = min(0.1 + 0.2 * brightness, 0.5)
        core_color = (r_val, g_val, b_val)

        core = plt.Circle((0, 0), 0.6, facecolor=core_color,
                          edgecolor="#ffcc66", linewidth=2)
        ax.add_patch(core)

        bar_x = -2.5
        bar_h = 4.0 * brightness
        ax.add_patch(plt.Rectangle(
            (bar_x, -2), 0.5, 4.0,
            facecolor="none", edgecolor="#555555", linewidth=1))
        ax.add_patch(plt.Rectangle(
            (bar_x, -2), 0.5, bar_h,
            facecolor="#ffaa44", edgecolor="none", alpha=0.8))
        ax.plot(bar_x + 0.25, -2 + bar_h, "_", color="#00ff44",
                markersize=12, markeredgewidth=2, zorder=5)

        for tick_f in np.linspace(0, 4.0, 9):
            ax.plot([bar_x - 0.1, bar_x], [-2 + tick_f] * 2,
                    color="#555555", linewidth=0.5)

        if not clean:
            ax.text(2.2, -2.5, f"P={P:.2f}", fontsize=10, color="#ffaa44")

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._power(control_value)
