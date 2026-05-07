"""Boyle's law scenario.

Hidden law:  P · V^α = const   (true α = 1)
Equivalently:  V ∝ P^(-1/α)

Historical instrument: Robert Boyle (1662) trapped air in a sealed
J-shaped glass tube, poured mercury down the open end, and measured
how the gas column length changed with pressure.

The agent sees a virtual J-tube: mercury is added to the open side,
compressing the trapped gas. The agent varies the mercury height
(pressure) and observes the gas column length (proportional to volume).
"""

from __future__ import annotations

from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class Boyle(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 P0: float = 1.0, V0: float = 10.0):
        """P0 in atm, V0 in arbitrary units (gas column length cm)."""
        super().__init__(alpha=alpha, seed=seed)
        self._P0 = P0
        self._V0 = V0
        self._const = P0 * V0 ** self.alpha

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Boyle's Law",
            slug="boyle",
            description=(
                "A J-shaped glass tube with trapped air on one side. "
                "You add mercury to the open side to increase pressure "
                "and observe how the gas column length changes. "
                "Discover the exponent α in the gas law: P · V^α = const. "
                "For example, α=1 means PV=const (standard Boyle's law). "
                "α could be any positive number."
            ),
            control_var="pressure_atm",
            control_label="Applied pressure (atm)",
            observable_label="Gas column length (cm)",
            true_exponent=1.0,
            law_template="P · V^α = const  (find α; hint: V ∝ P^(-1/α))",
            historical_instrument="J-tube with mercury column (Boyle, 1662)",
        )

    @property
    def default_controls(self) -> list[float]:
        return [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    def _volume(self, P: float) -> float:
        """V from PV^α = const  →  V = (const/P)^(1/α)"""
        if self.alpha == 0:
            return self._V0
        return (self._const / P) ** (1.0 / self.alpha)

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float = 1.0) -> dict[str, Any]:
        P = control_value
        V = self._volume(P)
        return {
            "times": np.array([0.0]),
            "pressure": P,
            "volume": V,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (5, 9),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        P = sim_data["pressure"]
        V = sim_data["volume"]

        tube_width = 1.0
        tube_height = 20.0
        gas_height = V
        mercury_open = max(0, (P - self._P0) * 5)

        fig, ax = make_clean_fig(figsize=figsize)
        ax.set_xlim(-2, 5)
        ax.set_ylim(-1, tube_height + 1)
        ax.set_aspect("equal")
        ax.set_xticks([])

        if clean:
            ax.set_title("Experiment")
            ax.set_yticks([])
            ax.set_ylabel("")
        else:
            ax.set_title(f"Boyle's J-tube   P = {P:.2f} atm")
            ax.set_ylabel("Height (cm)")

        # sealed side (left tube)
        sealed_x = 0
        sealed_bottom = 2.0
        ax.add_patch(patches.Rectangle(
            (sealed_x, sealed_bottom), tube_width, tube_height - sealed_bottom,
            fill=False, edgecolor="#444444", linewidth=2))

        # gas in sealed side (top)
        gas_bottom = tube_height - gas_height
        ax.add_patch(patches.Rectangle(
            (sealed_x + 0.05, gas_bottom), tube_width - 0.1, gas_height,
            facecolor="#3399ff", edgecolor="none", alpha=0.85))
        if not clean:
            ax.text(sealed_x + tube_width / 2, gas_bottom + gas_height / 2,
                    "gas", ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold")

        # red marker at gas-mercury boundary for pixel detection
        ax.plot(sealed_x + tube_width / 2, gas_bottom, "o",
                color="#ff2222", markersize=8, zorder=5)
        ax.plot([sealed_x + 0.05, sealed_x + tube_width - 0.05],
                [gas_bottom, gas_bottom], "-", color="#ff2222",
                linewidth=2, zorder=5)

        # mercury in sealed side (below gas)
        mercury_sealed_height = max(0, tube_height - gas_height - sealed_bottom)
        if mercury_sealed_height > 0:
            ax.add_patch(patches.Rectangle(
                (sealed_x + 0.05, sealed_bottom),
                tube_width - 0.1, mercury_sealed_height,
                facecolor="#666666", edgecolor="none", alpha=0.9))

        # open side (right tube)
        open_x = 2.5
        ax.add_patch(patches.Rectangle(
            (open_x, 0), tube_width, tube_height,
            fill=False, edgecolor="#444444", linewidth=2))

        # mercury in open side
        if mercury_open > 0:
            ax.add_patch(patches.Rectangle(
                (open_x + 0.05, 0), tube_width - 0.1, min(mercury_open, tube_height),
                facecolor="#888888", edgecolor="none", alpha=0.8))

        # U-bend connecting tubes
        bend_y = sealed_bottom
        ax.plot([sealed_x + tube_width, open_x],
                [bend_y, 0], "-", color="#444444", linewidth=2)

        # ruler on the far right (tick marks only in clean mode)
        ruler_x = 4.0
        for cm in range(int(tube_height) + 1):
            tick_len = 0.3 if cm % 5 == 0 else 0.15
            ax.plot([ruler_x, ruler_x + tick_len], [cm, cm],
                    color="#888888", linewidth=0.5)
            if not clean and cm % 5 == 0:
                ax.text(ruler_x + 0.4, cm, str(cm),
                        fontsize=6, va="center", color="#888888")

        # gas column length annotation (skip entirely in clean mode)
        if not clean:
            ax.annotate("", xy=(-0.5, gas_bottom),
                        xytext=(-0.5, tube_height),
                        arrowprops=dict(arrowstyle="<->", color="#2266cc",
                                        linewidth=1.5))
            ax.text(-1.5, gas_bottom + gas_height / 2,
                    f"V={V:.1f}cm", fontsize=9, color="#2266cc",
                    ha="center", va="center", rotation=90)

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._volume(control_value)
