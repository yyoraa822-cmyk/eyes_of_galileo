"""Wave on a string scenario.

Hidden law:  v ∝ tension^α   (textbook α = 0.5)

The agent sees a vibrating string at fixed frequency. As tension changes,
the wavelength (and thus wave speed) changes. The control variable is
string tension; the observable is the wave speed.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class WaveOnString(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 freq: float = 2.0, C: float = 1.0):
        self._freq = freq
        self._C = C
        super().__init__(alpha=alpha, seed=seed)

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Wave on a String",
            slug="wave",
            description=(
                "A string vibrates at a fixed frequency. Increasing the "
                "tension changes the wavelength and wave speed. "
                "Discover how wave speed depends on tension."
            ),
            control_var="tension",
            control_label="String tension (N)",
            observable_label="Wave speed (m/s)",
            true_exponent=0.5,
            law_template="v ∝ T^α",
            historical_instrument="Mersenne's string experiments (1637)",
        )

    @property
    def default_controls(self) -> list[float]:
        return [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    def _wave_speed(self, tension: float) -> float:
        return self._C * tension ** self.alpha

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float = 1.0) -> dict[str, Any]:
        tension = max(0.1, control_value)
        speed = self._wave_speed(tension)
        wavelength = speed / self._freq
        return {
            "times": np.array([0.0]),
            "tension": tension,
            "speed": speed,
            "wavelength": wavelength,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (10, 4),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        tension = sim_data["tension"]
        speed = sim_data["speed"]
        wavelength = sim_data["wavelength"]

        fig, ax = make_clean_fig(figsize=figsize)

        L = 10.0
        x = np.linspace(0, L, 500)
        k = 2 * np.pi / wavelength
        amplitude = 0.8
        y = amplitude * np.sin(k * x)

        ax.plot(x, y, "-", color="#2266cc", linewidth=2.5)

        ax.plot([0, 0], [-1.2, 1.2], "-", color="#333333", linewidth=4)
        ax.plot([L, L], [-1.2, 1.2], "-", color="#333333", linewidth=4)

        first_full = wavelength
        if first_full < L:
            ax.annotate("", xy=(first_full, -1.3), xytext=(0, -1.3),
                        arrowprops=dict(arrowstyle="<->", color="#00aa44",
                                       linewidth=1.5))
            if not clean:
                ax.text(first_full / 2, -1.6, f"λ={wavelength:.2f}",
                        ha="center", fontsize=9, color="#00aa44")

        ax.set_xlim(-0.5, L + 0.5)
        ax.set_ylim(-2.0, 2.0)
        ax.set_aspect("auto")

        if clean:
            ax.set_title("Experiment")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_title(f"Wave on string  tension={tension:.1f} N")
            ax.set_xticks([])
            ax.set_yticks([])

        ax.axhline(0, color="#cccccc", linewidth=0.5, linestyle=":")

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._wave_speed(control_value)
