"""Damped oscillation scenario.

Hidden law:  A(t) = A₀ · exp(-γ · t^α)   (true α = 1)

The agent observes the exponent in the damping envelope. If α=1
the decay is the familiar exponential; counterfactual α≠1 gives
stretched/compressed exponential decay.

Historical instrument: Galileo and later physicists observed pendulums
losing amplitude over time. The agent must track how the peak amplitude
decreases across successive swings — requiring a "peak detector" or
"envelope tracer" instrument.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class DampedOscillation(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 gamma: float = 0.15, omega: float = 4.0,
                 A0: float = 1.0):
        super().__init__(alpha=alpha, seed=seed)
        self._gamma = gamma
        self._omega = omega
        self._A0 = A0

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Damped Oscillation",
            slug="damped",
            description=(
                "A pendulum swings with gradually decreasing amplitude. "
                "Observe how the peak amplitude decays over time. "
                "Discover the law governing the amplitude envelope."
            ),
            control_var="max_t",
            control_label="Observation duration (s)",
            observable_label="Amplitude envelope",
            true_exponent=1.0,
            law_template="A(t) = A₀ · exp(-γ · t^α)",
            historical_instrument=(
                "Pendulum with amplitude markings "
                "(Galileo & 18th-century physicists)"
            ),
        )

    @property
    def default_controls(self) -> list[float]:
        return [5.0, 10.0, 15.0, 20.0, 30.0]

    def _envelope(self, t: float | np.ndarray) -> float | np.ndarray:
        return self._A0 * np.exp(-self._gamma * np.power(t, self.alpha))

    def simulate(self, control_value: float, dt: float = 0.02,
                 max_t: float | None = None) -> dict[str, Any]:
        if max_t is None:
            max_t = control_value
        times = np.arange(0, max_t + dt / 2, dt)

        envelope = self._envelope(times)
        displacement = envelope * np.cos(self._omega * times)

        # find peaks (local maxima of |displacement|)
        peak_indices = []
        for i in range(1, len(displacement) - 1):
            if (displacement[i] > displacement[i - 1] and
                    displacement[i] > displacement[i + 1]):
                peak_indices.append(i)
        peak_times = times[peak_indices] if peak_indices else np.array([])
        peak_amps = envelope[peak_indices] if peak_indices else np.array([])

        return {
            "times": times,
            "displacement": displacement,
            "envelope_upper": envelope,
            "envelope_lower": -envelope,
            "peak_times": peak_times,
            "peak_amps": peak_amps,
            "max_t": max_t,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (10, 5),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        times = sim_data["times"]
        disp = sim_data["displacement"]
        env_upper = sim_data["envelope_upper"]
        env_lower = sim_data["envelope_lower"]
        peak_t = sim_data["peak_times"]
        peak_a = sim_data["peak_amps"]

        # Single final frame showing full oscillation
        fig, ax = make_clean_fig(figsize=figsize)
        ax.set_xlim(0, times[-1])
        ax.set_ylim(-self._A0 * 1.2, self._A0 * 1.2)

        if clean:
            ax.set_title("Experiment")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Displacement")
            ax.set_title(f"Damped oscillation   max_t = {times[-1]:.1f} s")

        # horizontal reference grid at amplitude fractions
        for frac in [0.25, 0.5, 0.75, 1.0]:
            ax.axhline(self._A0 * frac, color="#eeeeee", linewidth=0.5)
            ax.axhline(-self._A0 * frac, color="#eeeeee", linewidth=0.5)
        ax.axhline(0, color="#cccccc", linewidth=0.8)

        # full displacement waveform
        ax.plot(times, disp, "-", color="#2266cc", linewidth=1.5)

        # envelope curves
        ax.plot(times, env_upper, "--", color="#cc3333", linewidth=1.5,
                alpha=0.8)
        ax.plot(times, env_lower, "--", color="#cc3333", linewidth=1.5,
                alpha=0.8)

        # peak markers (large, clear)
        if len(peak_t) > 0:
            ax.plot(peak_t, peak_a, "o", color="#cc3333", markersize=8,
                    zorder=5)

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        """Envelope amplitude at time = control_value."""
        return float(self._envelope(control_value))
