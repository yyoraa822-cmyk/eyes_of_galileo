"""Newton's cooling law scenario.

Hidden law:  T(t) = T_env + dT0 * exp(rate * t)

The agent sees a thermometer/heat bar whose temperature drops over time.
The control variable is the observation time; the observable is the
remaining temperature above ambient.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class Cooling(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 T0: float = 90.0, T_env: float = 20.0,
                 rate: float | None = None):
        self._T0 = T0
        self._T_env = T_env
        self._rate = rate if rate is not None else -0.18
        super().__init__(alpha=alpha, seed=seed)

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Newton's Cooling",
            slug="cooling",
            description=(
                "A hot object cools down in a room-temperature environment. "
                "You can observe it at different times. "
                "Discover how the excess temperature depends on time."
            ),
            control_var="time",
            control_label="Observation time (s)",
            observable_label="Temperature above ambient (°C)",
            true_exponent=0.0,
            law_template="ΔT = ΔT₀ · exp(rate · t)",
            historical_instrument="Newton's cooling experiments (1701)",
            formula_type="exponential",
            true_params={"rate": self._rate},
        )

    @property
    def default_controls(self) -> list[float]:
        return [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]

    def _temperature(self, t: float) -> float:
        dT0 = self._T0 - self._T_env
        return self._T_env + dT0 * np.exp(self._rate * t)

    def _excess(self, t: float) -> float:
        return self._temperature(t) - self._T_env

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float = 1.0) -> dict[str, Any]:
        t = max(0, control_value)
        temp = self._temperature(t)
        excess = self._excess(t)
        return {
            "times": np.array([0.0]),
            "time": t,
            "temperature": temp,
            "excess": excess,
            "T0": self._T0,
            "T_env": self._T_env,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (6, 6),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        t = sim_data["time"]
        temp = sim_data["temperature"]
        T0 = sim_data["T0"]
        T_env = sim_data["T_env"]
        frac = (temp - T_env) / (T0 - T_env)
        frac = np.clip(frac, 0, 1)

        fig, ax = make_clean_fig(figsize=figsize)
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 11)
        ax.set_aspect("equal")

        if clean:
            ax.set_title("Experiment")
        else:
            ax.set_title(f"Newton's Cooling  t={t:.1f}s")
        ax.set_xticks([])
        ax.set_yticks([])

        bar_max = 9.0
        bar_h = bar_max * frac

        r = int(200 * frac + 50 * (1 - frac))
        g = int(50 * frac + 50 * (1 - frac))
        b_c = int(50 * frac + 200 * (1 - frac))
        color = f"#{r:02x}{g:02x}{b_c:02x}"

        ax.add_patch(plt.Rectangle(
            (1.5, 0.5), 2.0, bar_max,
            facecolor="none", edgecolor="#aaaaaa", linewidth=1, linestyle="--"))
        ax.add_patch(plt.Rectangle(
            (1.5, 0.5), 2.0, bar_h,
            facecolor=color, edgecolor="#333333", linewidth=1.5))

        env_y = bar_max * 0.0
        ax.axhline(0.5, color="#66aaff", linewidth=1, linestyle=":", xmin=0.15, xmax=0.85)

        ax.plot(2.5, 0.5 + bar_h, "v", color="#00cc44", markersize=12, zorder=5)

        for frac_tick in np.linspace(0, bar_max, 10):
            tick_len = 0.15
            ax.plot([1.3, 1.5], [0.5 + frac_tick] * 2,
                    color="#888888", linewidth=0.5)

        if not clean:
            ax.text(2.5, 0.5 + bar_h + 0.5, f"T={temp:.1f}°C",
                    ha="center", fontsize=9, color="#333333")
            ax.text(2.5, -0.3, f"Ambient={T_env:.0f}°C",
                    ha="center", fontsize=8, color="#6688aa")

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._excess(control_value)
