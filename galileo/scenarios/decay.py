"""Exponential decay scenario.

Hidden law:  brightness = B0 * exp(-rate * t)

The agent sees a glowing bar whose brightness decays over time.
The control variable is the observation time; the observable is
the remaining brightness (encoded as the length/intensity of a
colored bar in the image).
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class ExponentialDecay(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 B0: float = 10.0, rate: float | None = None):
        self._B0 = B0
        self._rate = rate if rate is not None else -0.35
        super().__init__(alpha=alpha, seed=seed)

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Exponential Decay",
            slug="decay",
            description=(
                "A glowing substance that fades over time. "
                "You can choose when to observe it (the time parameter). "
                "Discover how the remaining brightness depends on time."
            ),
            control_var="time",
            control_label="Observation time (s)",
            observable_label="Remaining brightness",
            true_exponent=0.0,
            law_template="B = B0 · exp(rate · t)",
            historical_instrument="Phosphorescent materials, Becquerel (1896)",
            formula_type="exponential",
            true_params={"rate": self._rate},
        )

    @property
    def default_controls(self) -> list[float]:
        return [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]

    def _brightness(self, t: float) -> float:
        return self._B0 * np.exp(self._rate * t)

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float = 1.0) -> dict[str, Any]:
        t = max(0, control_value)
        b = self._brightness(t)
        return {
            "times": np.array([0.0]),
            "time": t,
            "brightness": b,
            "B0": self._B0,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (6, 5),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        t = sim_data["time"]
        b = sim_data["brightness"]
        b_frac = b / self._B0

        fig, ax = make_clean_fig(figsize=figsize)
        ax.set_xlim(-0.5, 11)
        ax.set_ylim(-0.5, 2.5)
        ax.set_aspect("equal")

        if clean:
            ax.set_title("Experiment")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_title(f"Exponential Decay  t={t:.1f}s")
            ax.set_xticks([])
            ax.set_yticks([])

        bar_max_width = 10.0
        bar_width = bar_max_width * b_frac
        bar_height = 1.0

        r = int(50 + 205 * b_frac)
        g = int(200 * b_frac)
        b_color = int(50 * b_frac)
        color = f"#{r:02x}{g:02x}{b_color:02x}"

        ax.add_patch(plt.Rectangle(
            (0, 0.5), bar_width, bar_height,
            facecolor=color, edgecolor="#333333", linewidth=1.5))

        ax.add_patch(plt.Rectangle(
            (0, 0.5), bar_max_width, bar_height,
            facecolor="none", edgecolor="#aaaaaa", linewidth=1, linestyle="--"))

        scale_ticks = np.linspace(0, bar_max_width, 11)
        for sx in scale_ticks:
            tick_h = 0.15 if int(sx) % 2 == 0 else 0.08
            ax.plot([sx, sx], [0.3, 0.3 + tick_h], color="#888888", linewidth=0.8)

        ax.plot(bar_width, 1.0, "v", color="#00cc44", markersize=12, zorder=5)

        if not clean:
            ax.text(bar_width, 2.0, f"B={b:.2f}", ha="center", fontsize=9,
                    color="#333333")

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._brightness(control_value)
