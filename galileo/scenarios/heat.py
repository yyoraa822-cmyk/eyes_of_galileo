"""Fourier heat conduction scenario.

Hidden law:  x_front ∝ t^α   (true α = 0.5)

Historical instrument: Jan Ingenhousz (1789) coated metal rods with wax
and heated one end; the distance at which the wax melted over time
revealed the rate of heat conduction.

The agent sees a metal rod with one heated end. The heat front (color
change boundary) advances along the rod over time. The agent must
discover how the front position depends on time.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class HeatConduction(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 diffusivity: float = 1.0, rod_length: float = 10.0):
        super().__init__(alpha=alpha, seed=seed)
        self._kappa = diffusivity
        self._L = rod_length

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Fourier Heat Conduction",
            slug="heat",
            description=(
                "A metal rod is heated at one end. The heat front "
                "(visible as a color change) advances along the rod. "
                "Observe the rod at different times to discover how "
                "the front position depends on time."
            ),
            control_var="time_s",
            control_label="Observation time (s)",
            observable_label="Heat front position (m)",
            true_exponent=0.5,
            law_template="x_front ∝ t^α",
            historical_instrument=(
                "Wax-coated metal rod, heated at one end (Ingenhousz, 1789)"
            ),
        )

    @property
    def default_controls(self) -> list[float]:
        return [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    def _front_position(self, t: float) -> float:
        """Position of the heat front at time t."""
        return np.sqrt(self._kappa) * t ** self.alpha

    def _temperature_profile(self, x: np.ndarray, t: float) -> np.ndarray:
        """Temperature along the rod at time t.
        Uses erfc for true diffusion, but scales the argument by
        the counterfactual front position.
        """
        from scipy.special import erfc
        x_front = self._front_position(t)
        if x_front < 1e-8:
            return np.zeros_like(x)
        T_boundary = 100.0
        return T_boundary * erfc(x / (2 * x_front))

    def simulate(self, control_value: float, dt: float = 0.1,
                 max_t: float | None = None) -> dict[str, Any]:
        t = control_value
        x = np.linspace(0, self._L, 200)
        T_profile = self._temperature_profile(x, t)
        x_front = self._front_position(t)

        return {
            "times": np.array([t]),
            "x": x,
            "temperature": T_profile,
            "front_position": x_front,
            "observation_time": t,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (10, 4),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        x = sim_data["x"]
        T = sim_data["temperature"]
        t_obs = sim_data["observation_time"]
        x_front = sim_data["front_position"]

        if clean:
            fig = plt.figure(figsize=(10, 3), facecolor="white")
            ax_rod = fig.add_axes([0.05, 0.15, 0.85, 0.55])
        else:
            fig = plt.figure(figsize=figsize, facecolor="white")
            ax_rod = fig.add_axes([0.1, 0.55, 0.75, 0.25])

        T_2d = T.reshape(1, -1)
        cmap = plt.cm.hot
        ax_rod.imshow(T_2d, aspect="auto", cmap=cmap,
                      extent=[x[0], x[-1], -0.5, 0.5],
                      vmin=0, vmax=100)
        ax_rod.set_yticks([])

        if clean:
            ax_rod.set_title("Experiment")
            ax_rod.set_xticks([])
            ax_rod.set_xlabel("")
        else:
            ax_rod.set_xlabel("")
            ax_rod.set_title(f"Heat conduction   t = {t_obs:.2f} s")

        # front marker — bold and obvious
        if 0 < x_front < self._L:
            ax_rod.axvline(x_front, color="#00ff00", linewidth=3,
                           linestyle="-", zorder=5)
            ax_rod.plot(x_front, 0, "v", color="#00ff00", markersize=14,
                        zorder=6)
            if not clean:
                ax_rod.text(x_front, 0.7, f"front",
                            fontsize=7, color="#00aa00", ha="center")

        # wax droplet marks along the rod — larger, more distinct
        wax_positions = np.arange(0.5, self._L, 0.5)
        for wp in wax_positions:
            if wp < x_front:
                marker_color = "#ffaa00"
            else:
                marker_color = "#dddddd"
            ax_rod.plot(wp, -0.35, "v", color=marker_color, markersize=7)

        # position ticks along rod bottom (unlabeled in clean)
        for pos in range(int(self._L) + 1):
            ax_rod.plot([pos, pos], [-0.5, -0.45], color="#888888",
                        linewidth=1)
            if not clean:
                ax_rod.text(pos, -0.6, str(pos), fontsize=6,
                            ha="center", va="top", color="#888888")

        if not clean:
            # temperature profile plot (only in data mode)
            ax_plot = fig.add_axes([0.1, 0.12, 0.75, 0.35])
            ax_plot.plot(x, T, "-", color="#cc3333", linewidth=2)
            ax_plot.axhline(50, color="#aaaaaa", linewidth=0.5, linestyle=":")
            ax_plot.set_xlim(x[0], x[-1])
            ax_plot.set_ylim(-5, 105)
            ax_plot.set_xlabel("Position along rod (m)")
            ax_plot.set_ylabel("Temperature (°C)")
            if 0 < x_front < self._L:
                ax_plot.axvline(x_front, color="#00aa00", linewidth=1.5,
                                linestyle="--")

            ax_cb = fig.add_axes([0.88, 0.55, 0.02, 0.25])
            norm = mcolors.Normalize(vmin=0, vmax=100)
            cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                              cax=ax_cb)
            cb.set_label("T (°C)", fontsize=8)

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._front_position(control_value)
