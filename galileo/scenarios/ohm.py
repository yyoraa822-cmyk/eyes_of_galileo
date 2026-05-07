"""Ohm's law scenario (linear response).

Hidden law:  I = slope · V   (linear, textbook slope = 1.0 for unit resistance)

The agent sees a circuit with an ammeter. As voltage increases, current
increases linearly. The control variable is voltage; the observable is
the measured current.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class OhmLaw(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 slope: float | None = None, offset: float = 0.0):
        self._slope = slope if slope is not None else 1.8
        self._offset = offset
        super().__init__(alpha=alpha, seed=seed)

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Ohm's Law",
            slug="ohm",
            description=(
                "A simple circuit where voltage drives current through a resistor. "
                "Vary the applied voltage and observe the resulting current. "
                "Discover the relationship between voltage and current."
            ),
            control_var="voltage",
            control_label="Applied voltage (V)",
            observable_label="Current (A)",
            true_exponent=0.0,
            law_template="I = slope · V + offset",
            historical_instrument="Ohm's galvanometer experiments (1827)",
            formula_type="linear",
            true_params={"slope": self._slope, "offset": self._offset},
        )

    @property
    def default_controls(self) -> list[float]:
        return [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]

    def _current(self, V: float) -> float:
        return self._slope * V + self._offset

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float = 1.0) -> dict[str, Any]:
        V = max(0, control_value)
        I = self._current(V)
        return {
            "times": np.array([0.0]),
            "voltage": V,
            "current": I,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (7, 5),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        V = sim_data["voltage"]
        I = sim_data["current"]
        I_max = self._current(10.0)

        fig, ax = make_clean_fig(figsize=figsize)
        ax.set_xlim(-1, 12)
        ax.set_ylim(-1, 8)
        ax.set_aspect("equal")

        if clean:
            ax.set_title("Experiment")
        else:
            ax.set_title(f"Ohm's Law  V={V:.1f} V")
        ax.set_xticks([])
        ax.set_yticks([])

        v_bar_h = 6.0 * (V / 10.0)
        ax.add_patch(plt.Rectangle(
            (1, 0.5), 1.5, 6.0,
            facecolor="none", edgecolor="#aaaaaa", linewidth=1, linestyle="--"))
        ax.add_patch(plt.Rectangle(
            (1, 0.5), 1.5, v_bar_h,
            facecolor="#cc6644", edgecolor="#884422", linewidth=1.5, alpha=0.8))

        i_bar_h = 6.0 * (I / I_max) if I_max > 0 else 0
        ax.add_patch(plt.Rectangle(
            (5, 0.5), 1.5, 6.0,
            facecolor="none", edgecolor="#aaaaaa", linewidth=1, linestyle="--"))
        ax.add_patch(plt.Rectangle(
            (5, 0.5), 1.5, i_bar_h,
            facecolor="#4488cc", edgecolor="#224466", linewidth=1.5, alpha=0.8))

        ax.plot(5.75, 0.5 + i_bar_h, "v", color="#00cc44",
                markersize=10, zorder=5)

        for bar_x in [1, 5]:
            for frac in np.linspace(0, 6.0, 13):
                tick_len = 0.15
                ax.plot([bar_x - tick_len, bar_x], [0.5 + frac, 0.5 + frac],
                        color="#888888", linewidth=0.5)

        ax.annotate("", xy=(4.0, 3.5), xytext=(3.2, 3.5),
                    arrowprops=dict(arrowstyle="->", color="#666666",
                                   linewidth=1.5))

        resistor_x, resistor_y = 8.5, 3.0
        ax.add_patch(plt.Rectangle(
            (resistor_x - 0.8, resistor_y - 0.4), 1.6, 0.8,
            facecolor="#eecc88", edgecolor="#886633", linewidth=1.5))
        zz_x = np.linspace(resistor_x - 0.6, resistor_x + 0.6, 20)
        zz_y = resistor_y + 0.15 * np.sin(np.linspace(0, 4 * np.pi, 20))
        ax.plot(zz_x, zz_y, "-", color="#886633", linewidth=1.5)

        if clean:
            ax.text(1.75, -0.3, "VOLTAGE", ha="center", fontsize=8, color="#cc6644")
            ax.text(5.75, -0.3, "CURRENT", ha="center", fontsize=8, color="#4488cc")
        else:
            ax.text(1.75, -0.3, f"V={V:.1f}", ha="center", fontsize=9)
            ax.text(5.75, -0.3, f"I={I:.2f}", ha="center", fontsize=9)

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._current(control_value)
