"""Logarithmic response scenario (Weber-Fechner law analogy).

Hidden law:  response = coeff * log(stimulus) + offset

The agent sees a stimulus bar (input) and a response bar (output).
The control variable is the stimulus intensity; the observable is
the response magnitude, which grows logarithmically.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class LogarithmicResponse(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 coeff: float | None = None, offset: float = 1.0):
        self._coeff = coeff if coeff is not None else 2.5
        self._offset = offset
        super().__init__(alpha=alpha, seed=seed)

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Logarithmic Response",
            slug="logarithmic",
            description=(
                "A system where increasing the input stimulus produces "
                "a response that grows slower and slower. "
                "Discover how the response depends on stimulus intensity."
            ),
            control_var="stimulus",
            control_label="Stimulus intensity",
            observable_label="Response magnitude",
            true_exponent=0.0,
            law_template="R = a · log(S) + b",
            historical_instrument="Weber-Fechner psychophysics (1860)",
            formula_type="logarithmic",
            true_params={"coefficient": self._coeff, "offset": self._offset},
        )

    @property
    def default_controls(self) -> list[float]:
        return [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    def _response(self, stimulus: float) -> float:
        return self._coeff * np.log(max(stimulus, 0.01)) + self._offset

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float = 1.0) -> dict[str, Any]:
        s = max(0.01, control_value)
        r = self._response(s)
        return {
            "times": np.array([0.0]),
            "stimulus": s,
            "response": r,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (7, 5),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        stimulus = sim_data["stimulus"]
        response = sim_data["response"]

        fig, ax = make_clean_fig(figsize=figsize)

        stim_max = 50.0
        resp_max = self._coeff * np.log(stim_max) + self._offset + 1.0

        ax.set_xlim(-1, 12)
        ax.set_ylim(-1, 8)
        ax.set_aspect("equal")

        if clean:
            ax.set_title("Experiment")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_title(f"Logarithmic Response  S={stimulus:.1f}")
            ax.set_xticks([])
            ax.set_yticks([])

        stim_bar_height = 6.0 * (stimulus / stim_max)
        ax.add_patch(plt.Rectangle(
            (1, 0), 1.5, stim_bar_height,
            facecolor="#4488cc", edgecolor="#224466", linewidth=1.5, alpha=0.8))
        ax.add_patch(plt.Rectangle(
            (1, 0), 1.5, 6.0,
            facecolor="none", edgecolor="#aaaaaa", linewidth=1, linestyle="--"))

        resp_bar_height = max(0, 6.0 * (response / resp_max))
        resp_color = "#cc6644"
        ax.add_patch(plt.Rectangle(
            (5, 0), 1.5, resp_bar_height,
            facecolor=resp_color, edgecolor="#663322", linewidth=1.5, alpha=0.8))
        ax.add_patch(plt.Rectangle(
            (5, 0), 1.5, 6.0,
            facecolor="none", edgecolor="#aaaaaa", linewidth=1, linestyle="--"))

        ax.plot([5, 6.5], [resp_bar_height, resp_bar_height],
                "-", color="#00cc44", linewidth=2.5, zorder=5)
        ax.plot(5.75, resp_bar_height, "v", color="#00cc44", markersize=10, zorder=5)

        for bar_x in [1, 5]:
            for frac in np.linspace(0, 6.0, 13):
                tick_len = 0.2 if int(frac * 2) % 2 == 0 else 0.1
                ax.plot([bar_x - tick_len, bar_x], [frac, frac],
                        color="#888888", linewidth=0.5)

        if clean:
            ax.text(1.75, -0.6, "INPUT", ha="center", fontsize=9, color="#4488cc")
            ax.text(5.75, -0.6, "OUTPUT", ha="center", fontsize=9, color="#cc6644")
        else:
            ax.text(1.75, -0.6, f"Stimulus={stimulus:.1f}", ha="center", fontsize=8)
            ax.text(5.75, -0.6, f"Response={response:.2f}", ha="center", fontsize=8)

        ax.annotate("", xy=(3.5, 3.5), xytext=(2.8, 3.5),
                    arrowprops=dict(arrowstyle="->", color="#666666", linewidth=1.5))

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._response(control_value)
