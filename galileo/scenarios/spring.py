"""Spring / Hooke's Law scenario.

Hidden law:  x = (m g / k)^α   (true α = 1)

Historical instrument: Robert Hooke (1676) hung known weights on springs
and measured the extension with a ruler.

The agent sees a vertical spring with different masses hung from it and
must discover how extension depends on applied mass.
"""

from __future__ import annotations

from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class Spring(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 k: float = 20.0):
        super().__init__(alpha=alpha, seed=seed)
        self._k = k  # spring constant N/m

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Spring (Hooke's Law)",
            slug="spring",
            description=(
                "A vertical spring with a mass hung from it. "
                "You can change the mass and observe the equilibrium extension. "
                "Discover how the extension depends on mass."
            ),
            control_var="mass_kg",
            control_label="Hanging mass (kg)",
            observable_label="Spring extension (m)",
            true_exponent=1.0,
            law_template="x ∝ m^α",
            historical_instrument="Spring with ruler and known weights (Hooke, 1676)",
        )

    @property
    def default_controls(self) -> list[float]:
        return [0.1, 0.2, 0.5, 1.0, 2.0]

    def _extension(self, mass: float) -> float:
        return (mass * self._g / self._k) ** self.alpha

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float = 2.0) -> dict[str, Any]:
        mass = control_value
        x_eq = self._extension(mass)
        return {
            "times": np.array([0.0]),
            "mass": mass,
            "extension": x_eq,
            "natural_length": 0.3,
        }

    def _draw_spring_zigzag(self, ax: plt.Axes, y_top: float, y_bot: float,
                             x_center: float = 0.0, n_coils: int = 12,
                             width: float = 0.08):
        """Draw a zigzag spring between y_top and y_bot."""
        ys = np.linspace(y_top, y_bot, 2 * n_coils + 1)
        xs = [x_center]
        for i in range(1, len(ys) - 1):
            xs.append(x_center + width * (1 if i % 2 == 0 else -1))
        xs.append(x_center)
        ax.plot(xs, ys, "-", color="#666666", linewidth=1.5)

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (4, 8),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        mass = sim_data["mass"]
        ext = sim_data["extension"]
        nat = sim_data["natural_length"]

        spring_top = 0.0
        spring_bot = -(nat + ext)
        total_height = abs(spring_bot) + 0.3

        fig, ax = make_clean_fig(figsize=figsize)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-total_height - 0.2, 0.2)
        ax.set_aspect("equal")

        if clean:
            ax.set_title("Experiment")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel("")
        else:
            ax.set_title(f"Spring  m={mass:.2f} kg")
            ax.set_ylabel("Position (m)")
            ax.set_xticks([])

        # ceiling
        ax.plot([-0.4, 0.4], [0, 0], "k-", linewidth=3)
        ax.fill_between([-0.4, 0.4], 0, 0.1, color="#cccccc",
                        hatch="///")

        # spring
        self._draw_spring_zigzag(ax, spring_top, spring_bot)

        # mass block
        block_h = 0.1
        rect = patches.FancyBboxPatch(
            (-0.1, spring_bot - block_h), 0.2, block_h,
            boxstyle="round,pad=0.01", facecolor="#cc5555",
            edgecolor="#333333", linewidth=1.5,
        )
        ax.add_patch(rect)
        if not clean:
            ax.text(0, spring_bot - block_h / 2, f"{mass:.1f}kg",
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold")

        # ruler on the right (tick marks only in clean mode, no numbers)
        ruler_x = 0.3
        ruler_ticks = np.arange(0, total_height + 0.05, 0.05)
        for rt in ruler_ticks:
            tick_len = 0.06 if (rt * 100) % 10 < 1 else 0.03
            ax.plot([ruler_x, ruler_x + tick_len], [-rt, -rt],
                    color="#888888", linewidth=0.5)
            if not clean and (rt * 100) % 10 < 1:
                ax.text(ruler_x + 0.08, -rt, f"{rt:.1f}",
                        fontsize=5, va="center", color="#888888")

        # extension arrow + label (skip entirely in clean mode)
        if not clean and ext > 0.02:
            ax.annotate("", xy=(0.2, spring_bot), xytext=(0.2, -nat),
                        arrowprops=dict(arrowstyle="<->", color="#2266cc",
                                        linewidth=1.5))
            ax.text(0.25, -(nat + ext / 2),
                    f"x={ext:.3f}m", fontsize=7, color="#2266cc",
                    va="center")

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._extension(control_value)
