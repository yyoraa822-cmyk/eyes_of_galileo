"""Freefall scenario — Galileo's inclined plane.

Hidden law:  s = ½ g t^α   (true α = 2)

Historical instrument: Galileo used an inclined plane with bells/marks at
equal distance intervals to slow the motion, effectively creating a
"strobe" record of position vs time.

The agent sees a ball falling and must figure out the time-distance
relationship.  It controls nothing directly — it observes the same ball
at different time snapshots across multiple experiments at different
observation durations.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import Scenario, ScenarioMeta, fig_to_pil, make_clean_fig


class Freefall(Scenario):

    def __init__(self, alpha: float | None = None, seed: int = 42):
        super().__init__(alpha=alpha, seed=seed)
        self._ramp_angle: float = 90.0
        self._obs_window: float = 3.0
        self._n_strobes: int = 12
        self._show_ruler: bool = False
        self._show_timer: bool = False
        # Visual-only mass count for MassStack-style apparatus. Newtonian
        # freefall is mass-independent; this knob doesn't enter simulate().
        self._mass: int = 1

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="Freefall",
            slug="freefall",
            description=(
                "A ball is released from rest and falls under gravity. "
                "You can observe the ball's position at successive time steps. "
                "Discover the relationship between distance fallen and time."
            ),
            control_var="max_t",
            control_label="Observation duration (s)",
            observable_label="Distance fallen (m)",
            true_exponent=2.0,
            law_template="s ∝ t^α",
            historical_instrument="Inclined plane with position marks (Galileo, 1604)",
        )

    @property
    def g_eff(self) -> float:
        """Effective gravitational acceleration along the ramp."""
        return self._g * np.sin(np.radians(self._ramp_angle))

    def modify_apparatus(self, **kwargs) -> dict[str, str]:
        """Modify the experimental apparatus. Returns a description of changes."""
        changes = []
        if "ramp_angle" in kwargs:
            angle = float(kwargs["ramp_angle"])
            angle = max(1.0, min(90.0, angle))
            self._ramp_angle = angle
            changes.append(f"ramp_angle set to {angle:.0f}°")
        if "observation_window" in kwargs:
            win = float(kwargs["observation_window"])
            win = max(0.5, min(20.0, win))
            self._obs_window = win
            changes.append(f"observation_window set to {win:.1f}s")
        if "num_strobes" in kwargs:
            n = int(kwargs["num_strobes"])
            n = max(4, min(30, n))
            self._n_strobes = n
            changes.append(f"num_strobes set to {n}")
        if "add_ruler" in kwargs:
            self._show_ruler = bool(kwargs["add_ruler"])
            changes.append(f"ruler {'enabled' if self._show_ruler else 'disabled'}")
        if "add_timer" in kwargs:
            self._show_timer = bool(kwargs["add_timer"])
            changes.append(f"timer marks {'enabled' if self._show_timer else 'disabled'}")
        if "mass" in kwargs:
            m = max(1, int(round(float(kwargs["mass"]))))
            self._mass = m
            changes.append(f"mass set to {m}")
        return {
            "status": "apparatus modified" if changes else "no changes",
            "changes": "; ".join(changes) if changes else "none",
            "current_setup": (
                f"ramp_angle={self._ramp_angle:.0f}°, "
                f"observation_window={self._obs_window:.1f}s, "
                f"strobes={self._n_strobes}, "
                f"ruler={'on' if self._show_ruler else 'off'}, "
                f"timer={'on' if self._show_timer else 'off'}"
            ),
        }

    @property
    def default_controls(self) -> list[float]:
        return [1.0, 1.5, 2.0, 2.5, 3.0]

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float | None = None) -> dict[str, Any]:
        if max_t is None:
            max_t = control_value
        times = np.arange(0, max_t + dt / 2, dt)
        distances = 0.5 * self.g_eff * np.power(times, self.alpha)
        return {
            "times": times,
            "distances": distances,
            "max_t": max_t,
            "dt": dt,
            "ramp_angle": self._ramp_angle,
        }

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (4, 8),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        times = sim_data["times"]
        distances = sim_data["distances"]
        ramp_angle = sim_data.get("ramp_angle", 90.0)
        max_dist = distances[-1] * 1.1 if distances[-1] > 0 else 10.0

        is_ramp = ramp_angle < 85.0

        if is_ramp:
            fig, ax = make_clean_fig(figsize=(8, 6))
            theta = np.radians(ramp_angle)
            ramp_len = max_dist / np.sin(theta) if np.sin(theta) > 0.01 else max_dist
            margin = ramp_len * 0.15
            ax.set_xlim(-margin, ramp_len * np.cos(theta) + margin)
            ax.set_ylim(-ramp_len * np.sin(theta) - margin, margin)
            ax.set_aspect("equal")

            rx = np.array([0, ramp_len * np.cos(theta)])
            ry = np.array([0, -ramp_len * np.sin(theta)])
            ax.plot(rx, ry, "-", color="#888888", linewidth=3, zorder=1)

            ax.plot([0, ramp_len * np.cos(theta)],
                    [-ramp_len * np.sin(theta), -ramp_len * np.sin(theta)],
                    ":", color="#cccccc", linewidth=1)
            ax.plot([0, 0], [0, -ramp_len * np.sin(theta)],
                    ":", color="#cccccc", linewidth=1)

            if self._show_ruler:
                for d_mark in np.arange(0, max_dist + 0.5, max(0.5, max_dist / 10)):
                    mx = (d_mark / max_dist) * ramp_len * np.cos(theta) if max_dist > 0 else 0
                    my = -(d_mark / max_dist) * ramp_len * np.sin(theta) if max_dist > 0 else 0
                    perp_x = -np.sin(theta) * 0.15
                    perp_y = -np.cos(theta) * 0.15
                    ax.plot([mx + perp_x, mx - perp_x],
                            [my + perp_y, my - perp_y],
                            "-", color="#aaaaaa", linewidth=0.5)

            n_strobe = min(self._n_strobes, len(times) - 1)
            strobe_idx = np.linspace(1, len(times) - 1, n_strobe, dtype=int)

            for i, si in enumerate(strobe_idx[:-1]):
                frac = distances[si] / max_dist if max_dist > 0 else 0
                sx = frac * ramp_len * np.cos(theta)
                sy = -frac * ramp_len * np.sin(theta)
                color_r = int(180 + 75 * (i / max(n_strobe - 1, 1)))
                color_g = int(180 - 80 * (i / max(n_strobe - 1, 1)))
                color_b = int(180 - 80 * (i / max(n_strobe - 1, 1)))
                c = f"#{color_r:02x}{color_g:02x}{color_b:02x}"
                ax.plot(sx, sy, "o", color=c, markersize=9, alpha=0.7, zorder=3)
                if self._show_timer:
                    ax.text(sx + 0.2, sy + 0.15, f"t={times[si]:.2f}",
                            fontsize=5, color="#666666")

            frac_end = 1.0
            ex = frac_end * ramp_len * np.cos(theta)
            ey = -frac_end * ramp_len * np.sin(theta)
            ax.plot(ex, ey, "o", color="#2266cc", markersize=13, zorder=5)
            ax.plot(0, 0, "s", color="#44aa44", markersize=10, zorder=5)

            if clean:
                ax.set_title("Experiment")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_title(f"Inclined plane  θ={ramp_angle:.0f}°  "
                             f"t_max={times[-1]:.2f}s")

        else:
            fig, ax = make_clean_fig(figsize=figsize)
            ax.set_xlim(-1, 1)
            ax.set_ylim(max_dist, 0)
            ax.set_xticks([])

            if clean:
                ax.set_title("Experiment")
                ax.set_yticks([])
                ax.set_ylabel("")
            else:
                ax.set_ylabel("Distance fallen (m)")
                ax.set_title(f"Freefall  t_max = {times[-1]:.2f} s")

            if self._show_ruler:
                n_ruler = 20
                ruler_step = max(0.5, round(max_dist / n_ruler, 1))
                for rt in np.arange(0, max_dist + ruler_step, ruler_step):
                    ax.axhline(rt, color="#dddddd", linewidth=0.5)
                    if not clean and rt % (ruler_step * 5) < 0.01:
                        ax.text(-0.9, rt, f"{rt:.1f}", fontsize=6,
                                va="center", color="#888888")

            n_strobe = min(self._n_strobes, len(times) - 1)
            strobe_idx = np.linspace(1, len(times) - 1, n_strobe, dtype=int)

            for i, si in enumerate(strobe_idx[:-1]):
                frac = i / max(len(strobe_idx) - 1, 1)
                r = int(180 + 75 * frac)
                g = int(180 - 80 * frac)
                b = int(180 - 80 * frac)
                color = f"#{r:02x}{g:02x}{b:02x}"
                ax.plot(0, distances[si], "o", color=color, markersize=10,
                        alpha=0.7, zorder=3)
                if self._show_timer:
                    ax.text(0.15, distances[si], f"t={times[si]:.2f}",
                            fontsize=5, color="#666666")

            ax.plot(0, distances[-1], "o", color="#2266cc", markersize=14,
                    zorder=5)
            ax.plot(0, 0, "s", color="#44aa44", markersize=10, zorder=5)

            # MassStack: N unit cubes stacked above the release point.
            # Each cube is the SAME visual size regardless of mass count,
            # so a VLM counts cubes rather than reading volume.
            if self._mass > 1:
                import matplotlib.patches as patches
                cube_w = 0.18
                cube_h = max_dist * 0.025
                for i in range(self._mass):
                    cy = -cube_h * (i + 1) - 0.02
                    rect = patches.Rectangle(
                        (-cube_w / 2, cy), cube_w, cube_h,
                        facecolor="#cc8833", edgecolor="#553311",
                        linewidth=1.0, zorder=6,
                    )
                    ax.add_patch(rect)
                # extend axis upward so the stack is visible
                cur_top = -cube_h * self._mass - 0.04
                ax.set_ylim(max_dist, cur_top - 0.05)

        return [fig_to_pil(fig, dpi=dpi)]

    def get_observable(self, control_value: float) -> float:
        """Distance fallen after `control_value` seconds."""
        return 0.5 * self.g_eff * (control_value ** self.alpha)


class FreefallMass(Scenario):
    """Counterfactual freefall where mass DOES affect distance fallen.

    Hidden law:  d = 0.5 * g * T_fixed^α_time * m^α_mass
    where T_fixed is a fixed observation time and m is the control variable.

    In this alien world, heavier objects fall farther in a given time
    (unlike Newtonian physics where mass is irrelevant).
    """

    def __init__(self, alpha: float | None = None, seed: int = 42,
                 fixed_time: float = 2.0):
        super().__init__(alpha=alpha, seed=seed)
        self._fixed_time = fixed_time

    @property
    def meta(self) -> ScenarioMeta:
        return ScenarioMeta(
            name="FreefallMass",
            slug="freefall_mass",
            description=(
                "A ball is released from rest and falls for a fixed duration. "
                "The ball's mass varies between experiments. "
                "Discover how the distance fallen depends on mass."
            ),
            control_var="mass_kg",
            control_label="Ball mass (kg)",
            observable_label="Distance fallen (m)",
            true_exponent=1.0,
            law_template="d ∝ m^α",
            historical_instrument="Balance + drop tower (thought experiment)",
        )

    @property
    def default_controls(self) -> list[float]:
        return [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]

    def simulate(self, control_value: float, dt: float = 0.05,
                 max_t: float | None = None) -> dict[str, Any]:
        mass = max(control_value, 0.1)
        t = self._fixed_time
        times = np.arange(0, t + dt / 2, dt)
        distances = 0.5 * self._g * np.power(times, 2.0) * (mass ** self.alpha)
        return {
            "times": times,
            "distances": distances,
            "max_t": t,
            "dt": dt,
            "mass": mass,
        }

    def get_observable(self, control_value: float) -> float:
        """Distance fallen in fixed time as function of mass."""
        mass = max(control_value, 0.1)
        base_dist = 0.5 * self._g * (self._fixed_time ** 2.0)
        return base_dist * (mass ** self.alpha)

    def render_frames(self, sim_data: dict[str, Any],
                      figsize: tuple[float, float] = (4, 8),
                      dpi: int = 100,
                      clean: bool = False) -> list[Image.Image]:
        fig, ax = make_clean_fig(figsize, dpi, clean)
        mass = sim_data.get("mass", 1.0)
        distances = sim_data["distances"]
        max_dist = distances[-1] if len(distances) > 0 else 1.0
        ax.set_xlim(-1, 1)
        ax.set_ylim(max_dist * 1.1, -0.5)
        ax.add_patch(plt.Circle((0, -distances[-1]), 0.18, color="red"))
        ax.set_title(f"m={mass:.1f} kg", fontsize=10)
        return [fig_to_pil(fig, dpi=dpi)]
