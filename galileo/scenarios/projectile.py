"""Projectile motion scenario — Galileo's table-edge experiment.

Hidden law:  R = (v² sin2θ / g)^(α/2)   simplified as R ∝ v^α  (true α = 2)
at fixed launch angle.

Historical instrument: Galileo rolled balls off table edges with ink on them,
measuring where they landed on the floor using paper and a ruler.

The agent sees a ball launched at a fixed angle with varying initial speed
and must discover how the range depends on speed.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class Projectile(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 launch_angle_deg: float = 45.0):
        super().__init__(alpha=alpha, seed=seed)
        self._theta = np.radians(launch_angle_deg)

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Projectile Motion",
            slug="projectile",
            description=(
                "A ball is launched at a fixed angle with varying speed. "
                "You can change the launch speed and observe the trajectory. "
                "Discover how the horizontal range depends on launch speed."
            ),
            control_var="speed_mps",
            control_label="Launch speed (m/s)",
            observable_label="Horizontal range (m)",
            true_exponent=2.0,
            law_template="R ∝ v^α",
            historical_instrument="Ink-marked ball on paper (Galileo, 1638)",
        )

    @property
    def default_controls(self) -> list[float]:
        return [2.0, 4.0, 6.0, 8.0, 10.0]

    def _range(self, v: float) -> float:
        """Counterfactual range: R = C * v^α where C absorbs sin2θ/g."""
        C = np.sin(2 * self._theta) / self._g
        return C * (v ** self.alpha)

    def simulate(self, control_value: float, dt: float = 0.02,
                 max_t: float | None = None) -> dict[str, Any]:
        v = control_value
        vx = v * np.cos(self._theta)
        vy = v * np.sin(self._theta)

        R = self._range(v)
        if vx > 0:
            t_land = R / vx
        else:
            t_land = 1.0
        if max_t is None:
            max_t = t_land * 1.1

        times = np.arange(0, max_t + dt / 2, dt)
        x = vx * times
        y = vy * times - 0.5 * self._g * times ** 2
        y = np.maximum(y, 0.0)

        return {
            "times": times,
            "x": x,
            "y": y,
            "speed": v,
            "range": R,
            "t_land": t_land,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (8, 5),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        times = sim_data["times"]
        x = sim_data["x"]
        y = sim_data["y"]
        v = sim_data["speed"]
        R = sim_data["range"]

        x_max = max(R * 1.15, 1.0)
        y_max = max(np.max(y) * 1.3, 1.0)

        # single final frame showing the complete trajectory
        fig, ax = make_clean_fig(figsize=figsize)
        ax.set_xlim(-0.5, x_max)
        ax.set_ylim(-y_max * 0.08, y_max)

        if clean:
            ax.set_title("Experiment")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.set_xlabel("Horizontal distance (m)")
            ax.set_ylabel("Height (m)")
            ax.set_title(f"Projectile  v={v:.1f} m/s")

        # ground
        ax.axhline(0, color="#886644", linewidth=2)
        ax.fill_between([-0.5, x_max], -y_max * 0.08, 0,
                        color="#eeddcc", alpha=0.5)

        # grid lines (unlabeled in clean)
        grid_step_x = max(1.0, round(x_max / 10))
        grid_step_y = max(0.5, round(y_max / 8, 1))
        for gx in np.arange(0, x_max, grid_step_x):
            ax.axvline(gx, color="#dddddd", linewidth=0.5)
        for gy in np.arange(0, y_max, grid_step_y):
            ax.axhline(gy, color="#dddddd", linewidth=0.5)

        # full trajectory (thick colored arc)
        ax.plot(x, y, "-", color="#4488cc", linewidth=2.5)

        # strobe dots along trajectory
        n_dots = min(12, len(times))
        dot_idx = np.linspace(0, len(times) - 1, n_dots, dtype=int)
        ax.plot(x[dot_idx], y[dot_idx], "o", color="#4488cc",
                markersize=6, alpha=0.5)

        # launch point (green)
        ax.plot(0, 0, "^", color="#44aa44", markersize=12, zorder=5)

        # landing point (red diamond on ground)
        ax.plot(R, 0, "D", color="#cc3333", markersize=10, zorder=5,
                markeredgecolor="#881111", markeredgewidth=1.5)

        if not clean:
            ax.annotate("", xy=(R, -y_max * 0.04), xytext=(0, -y_max * 0.04),
                        arrowprops=dict(arrowstyle="<->", color="#cc3333",
                                        linewidth=1.5))
            ax.text(R / 2, -y_max * 0.06, f"R={R:.2f}m", fontsize=8,
                    ha="center", color="#cc3333")

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._range(control_value)
