"""Orbital motion / Kepler's third law scenario.

Hidden law:  T² ∝ a^α   (true α = 3)
Equivalently:  T ∝ a^(α/2)

Historical instrument: Kepler (1609-1619) used Tycho Brahe's naked-eye
observations of Mars spanning decades, combined with sextants, quadrants,
and geometric methods to derive the orbital parameters.

The agent sees planets orbiting a star at different semi-major axes
and must discover how the orbital period depends on the orbit size.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class Orbital(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 GM: float = 40.0):
        super().__init__(alpha=alpha, seed=seed)
        self._GM = GM

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Orbital Motion (Kepler)",
            slug="orbital",
            description=(
                "A planet orbits a central star. "
                "You can change the orbital radius and observe the motion. "
                "Discover how the orbital period depends on the orbit size."
            ),
            control_var="semi_major_axis",
            control_label="Semi-major axis (AU)",
            observable_label="Orbital period (years)",
            true_exponent=3.0,
            law_template="T² ∝ a^α",
            historical_instrument=(
                "Tycho Brahe's sextant + geometric analysis (Kepler, 1609)"
            ),
        )

    @property
    def default_controls(self) -> list[float]:
        return [1.0, 1.5, 2.0, 3.0, 4.0]

    def _period(self, a: float) -> float:
        """T² = (4π²/GM) a^α  →  T = 2π/√(GM) · a^(α/2)"""
        return 2 * np.pi / np.sqrt(self._GM) * a ** (self.alpha / 2)

    # Fixed observation window so arc fraction encodes the period.
    # Chosen so a=1 completes ~75% of orbit, a=4 completes ~10%.
    _FIXED_OBS_TIME = 0.75

    def simulate(self, control_value: float, dt: float = 0.02,
                 max_t: float | None = None) -> dict[str, Any]:
        a = control_value
        T = self._period(a)
        if max_t is None:
            max_t = self._FIXED_OBS_TIME
        omega = 2 * np.pi / T if T > 0 else 1.0

        times = np.arange(0, max_t + dt / 2, dt)
        x = a * np.cos(omega * times)
        y = a * np.sin(omega * times)

        return {
            "times": times,
            "x": x,
            "y": y,
            "a": a,
            "period": T,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (6, 6),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        x = sim_data["x"]
        y = sim_data["y"]
        a = sim_data["a"]
        T = sim_data["period"]
        times = sim_data["times"]

        # Fixed viewport so orbit radii are directly comparable across experiments
        max_a = max(self.default_controls)
        lim = max(max_a * 1.3, a * 1.3)

        # Use the LAST frame: planet position after a fixed observation time.
        # The fraction of orbit completed encodes the period.
        idx = len(times) - 1

        fig, ax = make_clean_fig(figsize=figsize, bg="#0a0a2a")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_color("#333366")

        if clean:
            ax.set_title("Experiment", color="white")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_title(f"Orbit  a={a:.1f} AU   t={times[idx]:.2f}",
                         color="white")
            ax.tick_params(colors="white")

        # full orbit path (faint dashed)
        theta_full = np.linspace(0, 2 * np.pi, 200)
        ax.plot(a * np.cos(theta_full), a * np.sin(theta_full),
                "--", color="#223355", linewidth=1, alpha=0.5)

        # traversed arc (bright thick) — the key visual signal
        ax.plot(x[:idx + 1], y[:idx + 1],
                "-", color="#44aaff", linewidth=3, alpha=0.9)

        # start position marker (green)
        ax.plot(x[0], y[0], "s", color="#44ff44", markersize=10, zorder=5)

        # star
        ax.plot(0, 0, "*", color="#ffdd44", markersize=22, zorder=5)

        # planet at current position (bright blue)
        ax.plot(x[idx], y[idx], "o", color="#4488ff", markersize=12, zorder=5)

        # reference circles at integer AU (faint grid)
        for r_ref in range(1, int(lim) + 1):
            circle = plt.Circle((0, 0), r_ref, fill=False,
                                color="#222244", linewidth=0.5,
                                linestyle=":")
            ax.add_patch(circle)

        if not clean:
            ax.text(0.02, 0.02, f"T = {T:.2f}",
                    transform=ax.transAxes, fontsize=9, color="#aaaacc")

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._period(control_value)
