"""RC capacitor charging scenario.

Hidden law:  V_deficit(t) = V0 * exp(rate * t)

The agent sees a capacitor charging — the voltage deficit (gap from
full charge) decays exponentially. The control variable is time;
the observable is the remaining voltage deficit.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class Capacitor(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 V0: float = 10.0, rate: float | None = None):
        self._V0 = V0
        self._rate = rate if rate is not None else -1.3
        super().__init__(alpha=alpha, seed=seed)

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="RC Capacitor Charging",
            slug="capacitor",
            description=(
                "A capacitor charges through a resistor. "
                "The voltage gap (how far from fully charged) decreases over time. "
                "Discover how the remaining voltage deficit depends on time."
            ),
            control_var="time",
            control_label="Charging time (s)",
            observable_label="Voltage deficit (V)",
            true_exponent=0.0,
            law_template="V_deficit = V₀ · exp(rate · t)",
            historical_instrument="Leyden jar experiments (1746)",
            formula_type="exponential",
            true_params={"rate": self._rate},
        )

    @property
    def default_controls(self) -> list[float]:
        return [0.1, 0.2, 0.5, 1.0, 2.0, 3.0]

    def _deficit(self, t: float) -> float:
        return self._V0 * np.exp(self._rate * t)

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float = 1.0) -> dict[str, Any]:
        t = max(0, control_value)
        deficit = self._deficit(t)
        voltage = self._V0 - deficit
        return {
            "times": np.array([0.0]),
            "time": t,
            "deficit": deficit,
            "voltage": voltage,
            "V0": self._V0,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (7, 5),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        t = sim_data["time"]
        voltage = sim_data["voltage"]
        deficit = sim_data["deficit"]
        V0 = sim_data["V0"]
        frac = voltage / V0

        fig, ax = make_clean_fig(figsize=figsize)
        ax.set_xlim(-1, 12)
        ax.set_ylim(-1, 8)
        ax.set_aspect("equal")

        if clean:
            ax.set_title("Experiment")
        else:
            ax.set_title(f"RC Charging  t={t:.2f}s")
        ax.set_xticks([])
        ax.set_yticks([])

        bar_max = 6.0
        charged_h = bar_max * frac
        deficit_h = bar_max - charged_h

        ax.add_patch(plt.Rectangle(
            (2, 0.5), 3, bar_max,
            facecolor="none", edgecolor="#888888", linewidth=1.5))

        ax.add_patch(plt.Rectangle(
            (2, 0.5), 3, charged_h,
            facecolor="#44aa66", edgecolor="none", alpha=0.8))

        ax.add_patch(plt.Rectangle(
            (2, 0.5 + charged_h), 3, deficit_h,
            facecolor="#ffcccc", edgecolor="none", alpha=0.5))

        ax.plot(3.5, 0.5 + charged_h, "_", color="#cc3333",
                markersize=20, markeredgewidth=3, zorder=5)

        for frac_tick in np.linspace(0, bar_max, 11):
            ax.plot([1.7, 2.0], [0.5 + frac_tick] * 2,
                    color="#888888", linewidth=0.5)

        ax.text(3.5, 7.2, "FULL", ha="center", fontsize=8, color="#888888")
        ax.plot([2, 5], [0.5 + bar_max] * 2, "--", color="#44aa66",
                linewidth=1, alpha=0.6)

        if not clean:
            ax.text(8, 4, f"V = {voltage:.2f} V", fontsize=10, color="#44aa66")
            ax.text(8, 3, f"Deficit = {deficit:.2f} V", fontsize=10, color="#cc3333")

        ax.add_patch(plt.Rectangle((7, 1), 0.3, 2,
                                   facecolor="#888888", edgecolor="#333"))
        ax.add_patch(plt.Rectangle((8.2, 1), 0.3, 2,
                                   facecolor="#888888", edgecolor="#333"))

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._deficit(control_value)
