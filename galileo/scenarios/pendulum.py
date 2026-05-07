"""Pendulum scenario — Galileo's cathedral chandelier.

Hidden law:  T = 2π (L / g)^α   (true α = 0.5)

Historical instrument: Galileo timed the swing of a cathedral chandelier
using his own pulse, later using string pendulums of different lengths
and counting oscillation cycles against a reference.

The agent observes pendulums of different lengths swinging and must
discover how the period depends on length.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class Pendulum(Scenario):

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Pendulum",
            slug="pendulum",
            description=(
                "A simple pendulum swings back and forth. "
                "You can change the string length and observe the motion. "
                "Discover how the period depends on string length."
            ),
            control_var="length_m",
            control_label="String length (m)",
            observable_label="Period (s)",
            true_exponent=0.5,
            law_template="T ∝ L^α",
            historical_instrument="String pendulum timed by pulse (Galileo, ~1602)",
        )

    @property
    def default_controls(self) -> list[float]:
        return [0.25, 0.5, 1.0, 2.0, 4.0]

    def _period(self, length: float) -> float:
        return 2 * np.pi * (length / self._g) ** self.alpha

    def simulate(self, control_value: float, dt: float = 0.02,
                 max_t: float = 10.0) -> dict[str, Any]:
        L = control_value
        T = self._period(L)
        omega = 2 * np.pi / T if T > 0 else 1.0
        theta_max = 0.3  # ~17 degrees, small angle

        times = np.arange(0, max_t + dt / 2, dt)
        theta = theta_max * np.cos(omega * times)

        bob_x = L * np.sin(theta)
        bob_y = -L * np.cos(theta)

        return {
            "times": times,
            "theta": theta,
            "bob_x": bob_x,
            "bob_y": bob_y,
            "length": L,
            "period": T,
            "dt": dt,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (6, 6),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        times = sim_data["times"]
        bob_x = sim_data["bob_x"]
        bob_y = sim_data["bob_y"]
        L = sim_data["length"]

        margin = L * 0.3
        xlim = (-L - margin, L + margin)
        ylim = (-L - margin, margin)

        # Single final frame showing the pendulum at its current position
        idx = len(times) - 1

        fig, ax = make_clean_fig(figsize=figsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")

        if clean:
            ax.set_title("Experiment")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.set_title(f"Pendulum  L={L:.2f}m   t={times[idx]:.2f}s")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

        # sweep arc showing full swing range (faint trail)
        theta_max = np.max(np.abs(np.arctan2(bob_x, -bob_y)))
        arc_theta = np.linspace(-theta_max, theta_max, 100)
        arc_x = L * np.sin(arc_theta)
        arc_y = -L * np.cos(arc_theta)
        ax.plot(arc_x, arc_y, "-", color="#ddcccc", linewidth=1.5, alpha=0.5)

        # pivot
        ax.plot(0, 0, "ks", markersize=10)

        # vertical reference line
        ax.plot([0, 0], [0, -L], ":", color="#aaaaaa", linewidth=1)

        # string
        ax.plot([0, bob_x[idx]], [0, bob_y[idx]], "-",
                color="#555555", linewidth=2)

        # bob (large, distinct)
        ax.plot(bob_x[idx], bob_y[idx], "o",
                color="#cc3333", markersize=18, zorder=5)

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._period(control_value)
