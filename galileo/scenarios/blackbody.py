"""Wien's displacement law scenario (blackbody radiation).

Hidden law:  λ_peak ∝ T^α   (textbook α = −1)

The agent sees a simulated emission spectrum that shifts with temperature.
The control variable is temperature; the observable is the peak wavelength.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class Blackbody(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 b_eff: float = 3000.0):
        self._b = b_eff
        super().__init__(alpha=alpha, seed=seed)

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Wien's Displacement Law",
            slug="blackbody",
            description=(
                "A heated object emits light whose color (peak wavelength) "
                "shifts with temperature. Observe the spectrum at different "
                "temperatures to discover how the peak wavelength changes."
            ),
            control_var="temperature",
            control_label="Temperature (K)",
            observable_label="Peak wavelength (nm)",
            true_exponent=-1.0,
            law_template="λ_peak ∝ T^α",
            historical_instrument="Langley's bolometer (1878)",
        )

    @property
    def default_controls(self) -> list[float]:
        return [1000, 2000, 3000, 4000, 6000, 8000, 10000]

    def _peak_wavelength(self, T: float) -> float:
        T = max(T, 100)
        return self._b * T ** self.alpha

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float = 1.0) -> dict[str, Any]:
        T = max(100, control_value)
        lam_peak = self._peak_wavelength(T)
        return {
            "times": np.array([0.0]),
            "temperature": T,
            "peak_wavelength": lam_peak,
        }

    def _wavelength_to_rgb(self, lam_nm: float):
        """Approximate visible wavelength to RGB."""
        if lam_nm < 380:
            return (0.4, 0.0, 0.6)
        elif lam_nm < 440:
            t = (lam_nm - 380) / 60
            return (0.4 * (1 - t), 0.0, 0.6 + 0.4 * t)
        elif lam_nm < 490:
            t = (lam_nm - 440) / 50
            return (0.0, t, 1.0)
        elif lam_nm < 510:
            t = (lam_nm - 490) / 20
            return (0.0, 1.0, 1.0 - t)
        elif lam_nm < 580:
            t = (lam_nm - 510) / 70
            return (t, 1.0, 0.0)
        elif lam_nm < 645:
            t = (lam_nm - 580) / 65
            return (1.0, 1.0 - t, 0.0)
        elif lam_nm < 780:
            return (1.0, 0.0, 0.0)
        else:
            return (0.5, 0.0, 0.0)

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (8, 5),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        T = sim_data["temperature"]
        lam_peak = sim_data["peak_wavelength"]

        fig, ax = make_clean_fig(figsize=figsize)

        wavelengths = np.linspace(100, 2000, 500)
        intensities = []
        for lam in wavelengths:
            x = lam / max(lam_peak, 1)
            intensity = (x ** (-5)) * np.exp(-1.0 / max(x, 0.01))
            intensities.append(intensity)
        intensities = np.array(intensities)
        intensities = intensities / max(intensities.max(), 1e-10)

        ax.fill_between(wavelengths, intensities, alpha=0.3, color="#ff6644")
        ax.plot(wavelengths, intensities, "-", color="#cc3333", linewidth=2)

        ax.axvline(lam_peak, color="#00aa44", linewidth=2, linestyle="--", zorder=5)
        ax.plot(lam_peak, 1.0, "v", color="#00aa44", markersize=12, zorder=6)

        vis_min, vis_max = 380, 750
        for wl in np.linspace(vis_min, vis_max, 50):
            rgb = self._wavelength_to_rgb(wl)
            ax.axvspan(wl - 4, wl + 4, ymin=0, ymax=0.06,
                      color=rgb, alpha=0.8)

        if clean:
            ax.set_title("Experiment")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_title(f"Blackbody Spectrum  T={T:.0f} K")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Relative intensity")

        ax.set_xlim(0, 2000)
        ax.set_ylim(-0.05, 1.15)

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        return self._peak_wavelength(control_value)
