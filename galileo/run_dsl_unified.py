"""Unified DSL runner — every scene is expressed as a YAML-style DSL
source string and rendered through `compile_and_render`. The DSL goes
through the full pipeline (parse, Gate 1/2/3 validation, render
dispatch) for ALL 15 scenes:

  - Mechanical scenes (`freefall`, `pendulum`, ramp, ...) use DSL
    motion-source entities (`FreefallBall`, `Pendulum`,
    `InclinedRamp`, ...) and render via the MuJoCo strobe pipeline.
  - Non-mechanical scenes (heat, decay, refraction, ...) use the
    `ScenarioBackend` entity, which still parses + validates as a
    real DSL scene, but `compile_and_render` dispatches it to a
    registered `Scenario.*` class for matplotlib rendering.

In both cases the agent receives the same prompt + 3 tools
(`run_experiment` / `request_more_scenes` / `submit_law`), and
scoring uses `scenario.get_observable` evaluated against the agent's
submitted closed-form expression.

Per-scene `summary.json` tags `render_path = "mujoco_strobe" |
"scenario_backend"` so the report can break down accuracy by which
DSL render path produced the image.

Usage:
    python -m galileo.run_dsl_unified --scene all --model gpt-5.5-medium
    python -m galileo.run_dsl_unified --scene s1_freefall --model claude-opus-4-6 --shots 1
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sympy as sp
from PIL import Image

from galileo.providers import VLMProvider, Message, image_to_data_url
from galileo.scenarios import (
    DEFAULT_COUNTERFACTUALS,
    DEFAULT_SCENARIO_KWARGS,
    Scenario,
    get_scenario,
)
from galileo.dsl.api import compile_and_render, validate as dsl_validate
from galileo.dsl.parser import DSLError, parse as dsl_parse


# ---------------------------------------------------------------------------
# Per-scene DSL YAML builders. Every scene yields a YAML source string
# that compile_and_render parses + validates + renders. Mechanical
# scenes use real DSL motion sources (FreefallBall, Pendulum, ...) and
# render via the MuJoCo strobe path. Non-mechanical scenes use a
# `ScenarioBackend` entity which dispatches to scenario.render_frames
# but still passes through Gate 1/2/3.
# ---------------------------------------------------------------------------


def _yaml_freefall_mass_dsl(view: dict[str, Any]) -> Callable[[float], str]:
    """DSL strobe scene for freefall where mass is the control variable.
    The number of MassStack cubes varies with cv (mass_kg)."""
    n_strobes = int(view.get("num_strobes", 10))

    def build(cv: float) -> str:
        mass = max(1, int(round(cv)))
        ys = (
            "scene:\n"
            "  name: freefall_mass_dsl\n"
            "  entities:\n"
            "    - name: ff1\n"
            "      type: FreefallBall\n"
            "      params:\n"
            "        ball: {name: bball, radius: 0.18, color: red}\n"
            "        release_height: 50.0\n"
            "    - name: trail1\n"
            "      type: StrobeTrail\n"
            f"      params: {{target_body: bball, n_samples: {n_strobes}}}\n"
            "    - name: mstack\n"
            "      type: MassStack\n"
            f"      params: {{target_body: bball, mass: {mass}, "
            "unit_size: 0.18, position: above}\n"
        )
        return ys

    return build


def _yaml_freefall_dsl(view: dict[str, Any]) -> Callable[[float], str]:
    """DSL strobe scene for vertical freefall. Optional MassStack
    visualises integer mass without affecting Newtonian physics."""
    mass = int(view.get("mass", 1))
    n_strobes = int(view.get("num_strobes", 12))

    def build(cv: float) -> str:
        ys = (
            "scene:\n"
            "  name: freefall_dsl\n"
            "  entities:\n"
            "    - name: ff1\n"
            "      type: FreefallBall\n"
            "      params:\n"
            "        ball: {name: bball, radius: 0.18, color: red}\n"
            "        release_height: 50.0\n"
            "    - name: trail1\n"
            "      type: StrobeTrail\n"
            f"      params: {{target_body: bball, n_samples: {n_strobes}}}\n"
        )
        if mass > 1:
            ys += (
                "    - name: mstack\n"
                "      type: MassStack\n"
                f"      params: {{target_body: bball, mass: {mass}, "
                "unit_size: 0.18, position: above}\n"
            )
        return ys

    return build


def _yaml_pendulum_dsl(view: dict[str, Any]) -> Callable[[float], str]:
    """DSL strobe scene for pendulum; string length tracks the agent's cv."""

    def build(cv: float) -> str:
        L = max(float(cv), 0.05)
        return (
            "scene:\n"
            "  name: pendulum_dsl\n"
            "  entities:\n"
            "    - name: p1\n"
            "      type: Pendulum\n"
            "      params:\n"
            f"        string: {{name: pstring, length: {L:g}}}\n"
            "        ball: {name: pbob, radius: 0.16, color: red}\n"
            "        pivot_height: 4.5\n"
            "        theta_max_deg: 17\n"
            "    - name: trail1\n"
            "      type: StrobeTrail\n"
            "      params: {target_body: pbob, n_samples: 12}\n"
        )

    return build


def _yaml_ramp_dsl(view: dict[str, Any]) -> Callable[[float], str]:
    """DSL strobe scene for an inclined-ramp roll; ball auto-injected."""
    angle = float(view.get("ramp_angle", 30.0))

    def build(cv: float) -> str:
        return (
            "scene:\n"
            "  name: ramp_dsl\n"
            "  entities:\n"
            "    - name: r1\n"
            "      type: InclinedRamp\n"
            "      params:\n"
            f"        ramp: {{name: rramp, angle_deg: {angle:g}, length: 20.0}}\n"
            "    - name: trail1\n"
            "      type: StrobeTrail\n"
            "      params: {target_body: r1, n_samples: 12}\n"
        )

    return build


def _yaml_scenario_backend(view: dict[str, Any]) -> Callable[[float], str]:
    """ScenarioBackend YAML: delegates to the named Scenario.* class.
    DSL still parses + validates the scene end-to-end."""
    slug = view["slug"]
    name = view["id"]

    def build(cv: float) -> str:
        return (
            "scene:\n"
            f"  name: {name}_sb\n"
            "  entities:\n"
            f"    - name: {slug}_backend\n"
            "      type: ScenarioBackend\n"
            "      params:\n"
            f"        slug: {slug}\n"
            "        clean: false\n"
        )

    return build


SCENE_VIEWS: list[dict[str, Any]] = [
    {
        "id": "s1_freefall", "slug": "freefall",
        "render_path": "mujoco_strobe",
        "yaml_builder": _yaml_freefall_dsl({"num_strobes": 12}),
        "target_body": "bball",
        # Mechanical scenes: NO motion-source lock. The hidden alpha is
        # shared across all DSL motion sources (FreefallBall, Pendulum,
        # InclinedRamp, HorizontalLaunch, SpringBlock, CircularMotion),
        # so the model is free to swap apparatus and read alpha through
        # whichever geometry it prefers. Gate 3 still requires AT LEAST
        # ONE motion source to be present.
        "physics_core": {},
    },
    {
        "id": "s2_mass", "slug": "freefall_mass",
        "render_path": "mujoco_strobe",
        "yaml_builder": _yaml_freefall_mass_dsl({}),
        "target_body": "bball",
        "note": "Mass affects freefall in this world (d ∝ m^α).",
        "physics_core": {},
    },
    {
        "id": "s3_pendulum", "slug": "pendulum",
        "render_path": "mujoco_strobe",
        "yaml_builder": _yaml_pendulum_dsl({}),
        "target_body": "pbob",
        "physics_core": {},
    },
    {
        "id": "s4_ramp", "slug": "freefall",
        "render_path": "mujoco_strobe",
        "yaml_builder": _yaml_ramp_dsl({"ramp_angle": 30.0}),
        "target_body": None,
        "scenario_kwargs": {"_ramp_angle": 30.0},
        "note": "Same scenario as s1 but on an inclined ramp.",
        "physics_core": {},
    },
    {
        "id": "s6_launch", "slug": "projectile",
        "render_path": "scenario_backend",
        "physics_core": {"locked_scenario_slug": "projectile"},
    },
    {
        "id": "s8_heat", "slug": "heat",
        "render_path": "scenario_backend",
        "physics_core": {"locked_scenario_slug": "heat"},
    },
    {
        "id": "s9_spring", "slug": "spring",
        "render_path": "scenario_backend",
        "physics_core": {"locked_scenario_slug": "spring"},
    },
    {
        "id": "s10_circular", "slug": "orbital",
        "render_path": "scenario_backend",
        "physics_core": {"locked_scenario_slug": "orbital"},
    },
    {
        "id": "s16_decay", "slug": "decay",
        "render_path": "scenario_backend",
        "physics_core": {"locked_scenario_slug": "decay"},
    },
    {
        "id": "s17_refraction", "slug": "refraction",
        "render_path": "scenario_backend",
        "physics_core": {"locked_scenario_slug": "refraction"},
    },
    {
        "id": "s18_boyle", "slug": "boyle",
        "render_path": "scenario_backend",
        "physics_core": {"locked_scenario_slug": "boyle"},
    },
    {
        "id": "s19_coulomb", "slug": "coulomb",
        "render_path": "scenario_backend",
        "physics_core": {"locked_scenario_slug": "coulomb"},
    },
    {
        "id": "s20_hooke", "slug": "spring",
        "render_path": "scenario_backend",
        "note": "Hooke F=kx variant; reuses spring scenario.",
        "physics_core": {"locked_scenario_slug": "spring"},
    },
    {
        "id": "s21_weber", "slug": "blackbody",
        "render_path": "scenario_backend",
        "physics_core": {"locked_scenario_slug": "blackbody"},
    },
    {
        "id": "s22_cooling", "slug": "cooling",
        "render_path": "scenario_backend",
        "physics_core": {"locked_scenario_slug": "cooling"},
    },
]
for _v in SCENE_VIEWS:
    if "yaml_builder" not in _v:
        _v["yaml_builder"] = _yaml_scenario_backend(_v)


# ---------------------------------------------------------------------------
# Candidate quantities per scene — includes the real observable + distractors.
# The model must identify which quantity actually responds to the control.
# ---------------------------------------------------------------------------

CANDIDATE_QUANTITIES: dict[str, list[dict[str, str]]] = {
    "freefall": [
        {"symbol": "d", "label": "vertical displacement of the ball (m)"},
        {"symbol": "v_f", "label": "final velocity at observation cutoff (m/s)"},
        {"symbol": "m", "label": "ball mass (kg)", "note": "visible as stacked cubes"},
        {"symbol": "r", "label": "ball radius (cm)"},
        {"symbol": "g_local", "label": "local gravitational field strength (m/s²)"},
        {"symbol": "E_k", "label": "kinetic energy at cutoff (J)"},
        {"symbol": "n_bounces", "label": "number of bounces before rest"},
    ],
    "freefall_mass": [
        {"symbol": "d", "label": "vertical displacement of the ball (m)"},
        {"symbol": "T_fall", "label": "total fall duration (s)", "note": "fixed across experiments"},
        {"symbol": "v_f", "label": "final velocity at ground (m/s)"},
        {"symbol": "r", "label": "ball radius (cm)"},
        {"symbol": "g_local", "label": "local gravitational field strength (m/s²)"},
        {"symbol": "E_k", "label": "kinetic energy at ground (J)"},
        {"symbol": "n_cubes", "label": "number of stacked cubes visible"},
    ],
    "pendulum": [
        {"symbol": "T", "label": "oscillation period (s)"},
        {"symbol": "theta_max", "label": "maximum angular displacement (rad)"},
        {"symbol": "m_bob", "label": "bob mass (kg)"},
        {"symbol": "v_max", "label": "maximum speed at lowest point (m/s)"},
        {"symbol": "tension_max", "label": "peak string tension (N)"},
        {"symbol": "E_total", "label": "total mechanical energy (J)"},
        {"symbol": "n_oscillations", "label": "number of full oscillations in window"},
    ],
    "spring": [
        {"symbol": "T", "label": "oscillation period (s)"},
        {"symbol": "x_max", "label": "maximum displacement from equilibrium (m)"},
        {"symbol": "k_eff", "label": "effective spring constant (N/m)"},
        {"symbol": "v_max", "label": "maximum speed (m/s)"},
        {"symbol": "E_elastic", "label": "maximum elastic potential energy (J)"},
        {"symbol": "f", "label": "oscillation frequency (Hz)"},
        {"symbol": "damping_ratio", "label": "amplitude decay factor per cycle"},
    ],
    "projectile": [
        {"symbol": "R", "label": "horizontal range (m)"},
        {"symbol": "H", "label": "maximum height (m)"},
        {"symbol": "T_flight", "label": "total flight time (s)"},
        {"symbol": "theta_opt", "label": "optimal launch angle for max range (deg)"},
        {"symbol": "v_impact", "label": "impact speed (m/s)"},
        {"symbol": "m", "label": "projectile mass (kg)"},
        {"symbol": "drag_coeff", "label": "aerodynamic drag coefficient"},
    ],
    "heat": [
        {"symbol": "x_front", "label": "heat front position along rod (m)"},
        {"symbol": "T_max", "label": "maximum temperature at source end (°C)"},
        {"symbol": "dT_dx", "label": "temperature gradient at midpoint (°C/m)"},
        {"symbol": "kappa", "label": "thermal diffusivity (m²/s)"},
        {"symbol": "rod_length", "label": "total rod length (m)"},
        {"symbol": "Q_total", "label": "total heat transferred (J)"},
        {"symbol": "T_ambient", "label": "ambient temperature (°C)"},
    ],
    "orbital": [
        {"symbol": "T_orbit", "label": "orbital period (time units)"},
        {"symbol": "v_orbital", "label": "orbital velocity (distance/time)"},
        {"symbol": "m_satellite", "label": "satellite mass (kg)"},
        {"symbol": "e", "label": "orbital eccentricity"},
        {"symbol": "E_total", "label": "total orbital energy"},
        {"symbol": "L", "label": "angular momentum"},
        {"symbol": "r_peri", "label": "periapsis distance"},
    ],
    "decay": [
        {"symbol": "N", "label": "remaining quantity (counts or concentration)"},
        {"symbol": "dN_dt", "label": "instantaneous decay rate"},
        {"symbol": "N_0", "label": "initial quantity at t=0"},
        {"symbol": "t_half", "label": "half-life (s)"},
        {"symbol": "T_ambient", "label": "ambient temperature (°C)"},
        {"symbol": "m_total", "label": "total sample mass (g)"},
        {"symbol": "activity", "label": "disintegrations per second (Bq)"},
    ],
    "refraction": [
        {"symbol": "theta_r", "label": "refraction angle (deg)"},
        {"symbol": "n_medium", "label": "refractive index of medium"},
        {"symbol": "lambda", "label": "wavelength of light (nm)"},
        {"symbol": "v_phase", "label": "phase velocity in medium (m/s)"},
        {"symbol": "I_transmitted", "label": "transmitted intensity (%)"},
        {"symbol": "delta", "label": "lateral displacement of ray (mm)"},
        {"symbol": "d_slab", "label": "slab thickness (cm)"},
    ],
    "boyle": [
        {"symbol": "V", "label": "gas volume (L)"},
        {"symbol": "T_gas", "label": "gas temperature (K)"},
        {"symbol": "n_mol", "label": "amount of substance (mol)"},
        {"symbol": "rho", "label": "gas density (kg/m³)"},
        {"symbol": "v_rms", "label": "root-mean-square molecular speed (m/s)"},
        {"symbol": "U_internal", "label": "internal energy (J)"},
        {"symbol": "compressibility", "label": "isothermal compressibility (1/atm)"},
    ],
    "coulomb": [
        {"symbol": "F", "label": "electrostatic force magnitude (N)"},
        {"symbol": "E_field", "label": "electric field strength (N/C)"},
        {"symbol": "V_pot", "label": "electric potential (V)"},
        {"symbol": "q1", "label": "charge of particle 1 (C)"},
        {"symbol": "q2", "label": "charge of particle 2 (C)"},
        {"symbol": "U_elec", "label": "electrostatic potential energy (J)"},
        {"symbol": "sigma", "label": "surface charge density (C/m²)"},
    ],
    "blackbody": [
        {"symbol": "lambda_peak", "label": "peak emission wavelength (nm)"},
        {"symbol": "P_total", "label": "total radiated power (W/m²)"},
        {"symbol": "T_surface", "label": "surface temperature (K)"},
        {"symbol": "emissivity", "label": "surface emissivity"},
        {"symbol": "spectral_width", "label": "FWHM of emission spectrum (nm)"},
        {"symbol": "photon_flux", "label": "photon emission rate (photons/s/m²)"},
        {"symbol": "color_index", "label": "B-V color index"},
    ],
    "cooling": [
        {"symbol": "T", "label": "object temperature (°C)"},
        {"symbol": "dT_dt", "label": "cooling rate (°C/s)"},
        {"symbol": "T_env", "label": "environment temperature (°C)"},
        {"symbol": "m", "label": "object mass (kg)"},
        {"symbol": "c_p", "label": "specific heat capacity (J/kg·K)"},
        {"symbol": "h_conv", "label": "convection coefficient (W/m²·K)"},
        {"symbol": "A_surface", "label": "surface area (m²)"},
    ],
}
DSL_INSTRUMENT_CHEATSHEET = """\
DSL surface available to you:

  Physical motion source (REQUIRED, do NOT remove or replace):
    - {motion_source}

  This scene's render_path = `{render_path}`. APPLICABILITY OF EACH
  INSTRUMENT depends on this:

  Universal (works on every scene, mechanical or scenario_backend):
    - BackgroundGrid     params: cell_px (int >=16), axis_labels (bool)
                         — faint 2D pixel-space grid overlay.

  Mechanical-only (visible only on render_path="mujoco_strobe";
  silently no-op on render_path="scenario_backend"):
    - StrobeTrail        params: target_body (str), n_samples (int 5-20)
    - FadingTrail        params: target_body (str), n_samples (int 20-80)
    - MassStack          params: target_body (str), mass (int >=1),
                                 unit_size (float), position (above|right)
    - ReferenceScale     bodies: reference_cube{{name, edge_length}}
                         params: position [x,y,z]
    - GridFloor          params: cell_size (float), extent (float)
    - AngleProtractor    params: origin [x,y,z], radius (float),
                                 max_angle_deg (float),
                                 tick_step_deg (float)
    - LightStrip         params: target_body (str), n_samples (int),
                                 channel ("luminance")
    - SpectrumBand       params: target_body (str), palette ("thermal"),
                                 min/max (float), anchor ("right")

YAML format (the parser accepts this exactly):
  scene:
    name: <any>
    entities:
      - name: <unique>
        type: <EntityKind>
        params:
          <key>: <value>

When you call `propose_apparatus(dsl_yaml=...)` the runner runs three
gates in sequence; if any fails you'll see the gate number and an
actionable reason. Re-call propose_apparatus with a corrected YAML.
"""


PHYSICS_PROFILE_ALPHA: dict[str, dict[str, float]] = {
    # Current benchmark defaults (counterfactual / non-Newtonian).
    "non_newton": {},
    # Textbook-ish exponents for easier sanity runs.
    "textbook": {
        "freefall": 2.0,
        # For the custom freefall-mass task, use linear mass scaling so the
        # relation remains non-degenerate and easier to fit.
        "freefall_mass": 1.0,
        "pendulum": 0.5,
        "spring": 1.0,
        "projectile": 2.0,
        "refraction": 1.0,
        "orbital": 3.0,
        "boyle": 1.0,
        "heat": 0.5,
        "damped": 1.0,
        "diffusion2d": 0.5,
        "coulomb": -2.0,
        "blackbody": -1.0,
        "wave": 0.5,
        "stefan": 4.0,
        "viscosity": 2.0,
    },
}

PHYSICS_PROFILE_KWARGS: dict[str, dict[str, dict[str, float]]] = {
    "non_newton": {},
    "textbook": {
        "decay": {"rate": -0.5},
        "logarithmic": {"coeff": 1.0},
        "cooling": {"rate": -0.30},
        "capacitor": {"rate": -2.0},
        "ohm": {"slope": 1.0},
    },
}


def _build_scenario(view: dict[str, Any], seed: int,
                    physics_profile: str = "non_newton") -> Scenario:
    """Build a Scenario instance only for ground-truth SCORING. Rendering
    of every scene goes through compile_and_render -- this object is
    never used by the renderer, only by `score_law`."""
    slug = view["slug"]
    alpha_override = PHYSICS_PROFILE_ALPHA.get(physics_profile, {}).get(slug)
    kw_override = dict(PHYSICS_PROFILE_KWARGS.get(physics_profile, {}).get(slug, {}))
    sc = get_scenario(slug, alpha=alpha_override, **kw_override)
    sc.rng = np.random.default_rng(seed)
    for k, v in view.get("scenario_kwargs", {}).items():
        setattr(sc, k, v)
    return sc


def check_physics_core(view: dict[str, Any], yaml_src: str
                       ) -> tuple[bool, str]:
    """Reject proposed YAML that drops or replaces the locked motion
    source / scenario slug. Runs AFTER Gate 1 parse succeeds."""
    try:
        scene = dsl_parse(yaml_src)
    except DSLError as e:
        return False, f"physics_core: cannot parse for core check: {e}"
    core = view.get("physics_core", {})
    required = core.get("required_entity_types", [])
    types_present = {e.kind for e in scene.entities.values()}
    for t in required:
        if t not in types_present:
            return False, (
                f"physics_core: scene must contain at least one entity of "
                f"type '{t}' (this is the locked motion source). Add it back."
            )
    locked = core.get("locked_scenario_slug")
    if locked is not None:
        sb = next((e for e in scene.entities.values()
                   if e.kind == "ScenarioBackend"), None)
        if sb is None:
            return False, (
                f"physics_core: scene must contain a ScenarioBackend entity "
                f"with slug='{locked}'."
            )
        actual = str(sb.params.get("slug", "")).strip()
        if actual != locked:
            return False, (
                f"physics_core: ScenarioBackend.slug must be '{locked}', "
                f"got '{actual}'. The scenario physics is fixed; you may "
                f"only adjust visualisation around it."
            )
    else:
        # Mechanical scene: forbid downgrading to ScenarioBackend, even
        # though Gate 3 nominally accepts it as a motion source. The
        # render_path for these scenes is "mujoco_strobe"; routing
        # through scenario.render_frames silently changes the visual
        # contract and bypasses every DSL instrument. This guard plugs
        # Bug #3 (mechanical -> ScenarioBackend downgrade loophole).
        sb_present = any(e.kind == "ScenarioBackend"
                         for e in scene.entities.values())
        if sb_present:
            return False, (
                "physics_core: this is a mechanical scene "
                "(render_path='mujoco_strobe'). Replacing the motion "
                "source with `ScenarioBackend` would silently route "
                "rendering through scenario.render_frames and skip "
                "every DSL instrument. Use one of FreefallBall, "
                "HorizontalLaunch, Pendulum, InclinedRamp, "
                "SpringBlock, or CircularMotion instead."
            )
    return True, ""


# ---------------------------------------------------------------------------
# Prompt assembly  (mirrors run_unified.py)
# ---------------------------------------------------------------------------

GIF_ONLY_PROMPT_TEMPLATE = """\
=== SETTING ===

WARNING: This is NOT the real world. You are in a SIMULATED universe where
the fundamental laws of physics are DIFFERENT from Earth's. Newtonian
mechanics, Kepler's laws, Fourier's law, Snell's law, Hooke's law, and
ALL other textbook formulas DO NOT necessarily hold here. The exponents,
coefficients, and functional forms may be completely different from what
you learned in school. You MUST discover the actual law purely from
observation — treating this as an alien world with unknown physics.

Your only channel is short animation clips (keyframes) returned by the
experiment tools. You have NO measurement instruments — no grid, no
rulers, no strobe freeze-frames. You can only watch the animation and
estimate by eye.

Scenario: {description}

=== VARIABLE IDENTIFICATION ===

The experimental parameter you can vary is denoted `{control_var}`.
The following quantities are accessible or visible in this experiment:

{candidate_quantities_block}

NOT all quantities listed above are relevant. Some may be constants,
some may be visual distractors, and some may be redundant. Your first
task is to identify — from the animations alone — WHICH quantity is the
meaningful OBSERVABLE that responds to changes in `{control_var}`, and
HOW it depends on it. {extra_note}

=== GOAL ===

Discover the law `observable = f({control_var})` and submit it as a
closed-form expression via `submit_law`. The oracle evaluates your
expression at multiple hidden reference values of `{control_var}`. You
PASS if the joint log-residual is at or below tolerance E <= {tol}.
A single expression must fit ALL reference values at once.

You have {n_shots} submission(s). Pool of allowed values for
`{control_var}`: {pool}.

=== METHODOLOGY (required) ===

You MUST follow this protocol — do NOT skip directly to submission:

  Step 1 — EXPLORE: request animations at >=3 well-spaced values of
           `{control_var}` (use `request_more_scenes`). Watch the
           animations and identify what VISUALLY CHANGES across them.

  Step 2 — IDENTIFY: state which observable quantity is changing and
           what its approximate values are at each control setting. Use
           visual estimates from the animation frames — relative speeds,
           arc sizes, positions — NOT prior physics knowledge.

  Step 3 — ESTIMATE: without measurement instruments, estimate the
           observable at each control value as best you can from the
           animation. Record (control_value -> estimated_observable)
           data pairs.

  Step 4 — FIT: You must find the mathematical relationship between the
           control variable and the observable FROM YOUR DATA. The
           relationship could be ANY of the following forms — do NOT
           assume one before checking:

           • Power law:       y = A * x^p       (check: log-log plot linear?)
           • Linear:          y = A * x + B     (check: direct proportionality?)
           • Exponential:     y = A * exp(k*x)  (check: semi-log plot linear?)
           • Logarithmic:     y = A * ln(x) + B (check: grows slower than linear?)
           • Square root:     y = A * sqrt(x)   (special case of power law, p=0.5)
           • Inverse:         y = A / x^p       (check: product x^p * y ≈ const?)
           • Trigonometric:   y = A * sin(x)    (check: periodic/bounded?)
           • Rational:        y = A*x / (B+x)   (check: saturating behavior?)

           METHOD: compute ratios y₂/y₁ for known x₂/x₁ ratios.
           If x doubles and y doubles → linear (p=1).
           If x doubles and y quadruples → quadratic (p=2).
           If x doubles and y increases by ~41% → sqrt (p=0.5).
           If x doubles and y increases by ~19% → fourth root (p=0.25).
           Use at least 3 data points to verify consistency.

           REMINDER: In this world, freefall might be linear in time,
           pendulum period might go as L^0.25, orbits might scale as r^1.
           Do NOT default to textbook exponents.

  Step 5 — SUBMIT: only after Step 4 gives a consistent fit, call
           `submit_law` with the derived expression.

CRITICAL: Do NOT assume textbook exponents. The physics here is ALIEN.
DERIVE EVERYTHING FROM YOUR DATA. If your data says the exponent is 1,
submit exponent 1 — even if "every physics textbook says 2."

=== TOOLS ===

  - run_experiment(config)
      Show the phenomenon as an animation (keyframes) at the given
      numeric value of `{control_var}`.

  - request_more_scenes(values, reason)
      Show animations for multiple control values in one call.

  - submit_law(law_expr, rationale)
      Commit a closed-form expression in the single variable
      `{control_var}`. Allowed: numeric literals, the symbol
      `{control_var}`, operators + - * / **, and standard math functions
      (sqrt, exp, log, sin, cos, pi). Implicit free constants like
      `C * x**p` are NOT allowed (every coefficient and exponent must be
      an explicit numeric literal).
      IMPORTANT: submit ONLY the RHS expression (e.g. `4.9*max_t`), not
      `d = ...` / `y = ...` / any variable assignment.
      Examples: `2*pi*sqrt(L/9.81)`, `0.5*9.81*t**2`, `3.0*exp(-0.5*t)`.

=== RATIONALE REQUIREMENT ===

Your `submit_law.rationale` MUST list, in order:
  (1) VARIABLE IDENTIFICATION — which observable you identified and why
      you ruled out the others.
  (2) VISUAL OBSERVATIONS — for each animation, what you saw (speeds,
      positions, relative changes).
  (3) DATA TABLE — explicit (control_value, estimated_observable) pairs.
  (4) FIT METHOD — log-log slope, ratio test, or regression you used.
  (5) DERIVATION — algebra from data to the submitted expression.

=== TURN BUDGET ===

You have {max_turns} total assistant turns. Every tool result contains
turns_remaining. You MUST submit at least one `submit_law` before turns
run out. When turns_remaining <= 2, your next call MUST be `submit_law`.
"""


SYSTEM_PROMPT_TEMPLATE = """\
=== SETTING ===

WARNING: This is NOT the real world. You are in a SIMULATED universe where
the fundamental laws of physics are DIFFERENT from Earth's. Newtonian
mechanics, Kepler's laws, Fourier's law, Snell's law, Hooke's law, and
ALL other textbook formulas DO NOT necessarily hold here. The exponents,
coefficients, and functional forms may be completely different from what
you learned in school. You MUST discover the actual law purely from
observation — treating this as an alien world with unknown physics.

Your only channel is the rendered images returned by the experiment tool.

Scenario: {description}

=== VARIABLE IDENTIFICATION ===

The experimental parameter you can vary is denoted `{control_var}`.
The following quantities are accessible or visible in this experiment:

{candidate_quantities_block}

NOT all quantities listed above are relevant. Some may be constants,
some may be visual distractors, and some may be redundant. Your first
task is to identify — from the images alone — WHICH quantity is the
meaningful OBSERVABLE that responds to changes in `{control_var}`, and
HOW it depends on it. {extra_note}

=== GOAL ===

Discover the law `observable = f({control_var})` and submit it as a
closed-form expression via `submit_law`. The oracle evaluates your
expression at multiple hidden reference values of `{control_var}`. You
PASS if the joint log-residual is at or below tolerance E <= {tol}.
A single expression must fit ALL reference values at once.

You have {n_shots} submission(s). Pool of allowed values for
`{control_var}`: {pool}.

=== METHODOLOGY (required) ===

You MUST follow this protocol — do NOT skip directly to submission:

  Step 1 — EXPLORE: request images at ≥3 well-spaced values of
           `{control_var}` (use `request_more_scenes`). Look at the
           images and identify what VISUALLY CHANGES across them.

  Step 2 — IDENTIFY: state which observable quantity is changing and
           what its approximate values are at each control setting. Use
           pixel positions, grid cells, geometric ratios, or other
           spatial cues from the images — NOT prior physics knowledge.

  Step 3 — QUANTIFY: attach instruments (BackgroundGrid, ReferenceScale,
           etc.) if helpful, then re-render to extract numeric readouts.
           Record (control_value → measured_observable) data pairs.

  Step 4 — FIT: You must find the mathematical relationship between the
           control variable and the observable FROM YOUR DATA. The
           relationship could be ANY of the following forms — do NOT
           assume one before checking:

           • Power law:       y = A * x^p       (check: log-log plot linear?)
           • Linear:          y = A * x + B     (check: direct proportionality?)
           • Exponential:     y = A * exp(k*x)  (check: semi-log plot linear?)
           • Logarithmic:     y = A * ln(x) + B (check: grows slower than linear?)
           • Square root:     y = A * sqrt(x)   (special case of power law, p=0.5)
           • Inverse:         y = A / x^p       (check: product x^p * y ≈ const?)
           • Trigonometric:   y = A * sin(x)    (check: periodic/bounded?)
           • Rational:        y = A*x / (B+x)   (check: saturating behavior?)

           METHOD: compute ratios y₂/y₁ for known x₂/x₁ ratios.
           If x doubles and y doubles → linear (p=1).
           If x doubles and y quadruples → quadratic (p=2).
           If x doubles and y increases by ~41% → sqrt (p=0.5).
           If x doubles and y increases by ~19% → fourth root (p=0.25).
           Use at least 3 data points to verify consistency.

           REMINDER: In this world, freefall might be linear in time,
           pendulum period might go as L^0.25, orbits might scale as r^1.
           Do NOT default to textbook exponents.

  Step 5 — SUBMIT: only after Step 4 gives a consistent fit, call
           `submit_law` with the derived expression.

CRITICAL: Do NOT assume textbook exponents. The physics here is ALIEN.
DERIVE EVERYTHING FROM YOUR DATA. If your data says the exponent is 1,
submit exponent 1 — even if "every physics textbook says 2."

=== APPARATUS / DSL ===

Every render image is produced from a YAML scene description (DSL).
The DEFAULT apparatus for this scene is:

```yaml
{starter_yaml}```

You may revise the apparatus at any time by calling
`propose_apparatus(dsl_yaml)`. The runner then runs three gates:

  Gate 1 (syntactic): YAML parse + entity / param schema check.
  Gate 2 (physical):  references resolve, MuJoCo MJCF compiles.
  Gate 3 (degeneracy): scene is observable (motion source present, ranges
                       sane).

If any gate fails, you'll get the gate number and a precise reason --
revise the YAML and call `propose_apparatus` again. Once a YAML is
accepted it becomes the CURRENT apparatus and every subsequent
`run_experiment` / `request_more_scenes` call will render through it.

You may attach / detach instrument entities to gain more readouts. The
locked physics core for THIS scene must NOT be removed or replaced.

IMPORTANT: even if you swap to a different motion source (e.g. observe
the same scenario through an InclinedRamp instead of a FreefallBall),
your `submit_law` expression must still predict the SAME observable
defined for this scene -- the geometric prefactor of your apparatus may
differ from the prefactor in the scoring formula, so translate any
readings back to the canonical observable before submitting.

{dsl_cheatsheet}

=== TOOLS ===

  - propose_apparatus(dsl_yaml, reason)
      Submit a new YAML apparatus. On success the runner stores it and
      replies with a preview image rendered at the midpoint of the
      control pool. On failure you'll see the gate error verbatim. You
      have a budget of {max_proposals} apparatus proposals.

  - run_experiment(config)
      Render the CURRENT apparatus at the given numeric value of
      `{control_var}` and return an image.

  - request_more_scenes(values, reason)
      Batch-render multiple control values through the current apparatus.

  - submit_law(law_expr, rationale)
      Commit a closed-form expression in the single variable
      `{control_var}`. Allowed: numeric literals, the symbol
      `{control_var}`, operators + - * / **, and standard math functions
      (sqrt, exp, log, sin, cos, pi). Implicit free constants like
      `C * x**p` are NOT allowed (every coefficient and exponent must be
      an explicit numeric literal).
      IMPORTANT: submit ONLY the RHS expression (e.g. `4.9*max_t`), not
      `d = ...` / `y = ...` / any variable assignment.
      Examples: `2*pi*sqrt(L/9.81)`, `0.5*9.81*t**2`, `3.0*exp(-0.5*t)`.

=== RATIONALE REQUIREMENT ===

Your `submit_law.rationale` MUST list, in order:
  (1) VARIABLE IDENTIFICATION — which observable you identified and why
      you ruled out the others.
  (2) RAW READOUT — for each image you saw, the quantitative marks you
      extracted (positions, lengths, counts, pixel ratios).
  (3) DATA TABLE — explicit (control_value, measured_observable) pairs.
  (4) FIT METHOD — log-log slope, ratio test, or regression you used.
  (5) DERIVATION — algebra from data to the submitted expression.

=== TURN BUDGET ===

You have {max_turns} total assistant turns. Every tool result contains
turns_remaining. You MUST submit at least one `submit_law` before turns
run out. When turns_remaining <= 2, your next call MUST be `submit_law`.
"""


def _format_motion_source_doc(view: dict[str, Any]) -> str:
    core = view.get("physics_core", {})
    if "locked_scenario_slug" in core:
        slug = core["locked_scenario_slug"]
        return (f"ScenarioBackend with slug='{slug}' (LOCKED; you may add "
                f"overlay instruments around it but cannot change the slug)")
    types = core.get("required_entity_types", [])
    if types:
        return ", ".join(types)
    return ("ANY ONE of {FreefallBall, HorizontalLaunch, Pendulum, "
            "InclinedRamp, SpringBlock, CircularMotion} -- the hidden "
            "alpha is the same across all of them, so swap whichever "
            "apparatus is most informative; you can also stack multiple "
            "in a single scene.")


def build_system_prompt(view: dict[str, Any], scenario: Scenario,
                        n_shots: int, max_turns: int, tol: float,
                        max_proposals: int, *,
                        gif_only: bool = False) -> str:
    meta = scenario.meta
    pool = scenario.default_controls
    cv0 = float(pool[len(pool) // 2])
    starter = view["yaml_builder"](cv0)

    slug = view["slug"]
    candidates = CANDIDATE_QUANTITIES.get(slug, [])
    if candidates:
        lines = []
        for cq in candidates:
            note = f"  [{cq['note']}]" if "note" in cq else ""
            lines.append(f"  • `{cq['symbol']}` — {cq['label']}{note}")
        cq_block = "\n".join(lines)
    else:
        cq_block = f"  • (observable: {meta.observable_label})"

    if gif_only:
        return GIF_ONLY_PROMPT_TEMPLATE.format(
            description=meta.description.strip(),
            control_var=meta.control_var,
            candidate_quantities_block=cq_block,
            extra_note=view.get("note", ""),
            tol=tol,
            n_shots=n_shots,
            pool=", ".join(f"{p:g}" for p in pool),
            max_turns=max_turns,
        )

    cheat = DSL_INSTRUMENT_CHEATSHEET.format(
        motion_source=_format_motion_source_doc(view),
        render_path=view.get("render_path", "scenario_backend"),
    )
    return SYSTEM_PROMPT_TEMPLATE.format(
        description=meta.description.strip(),
        control_var=meta.control_var,
        candidate_quantities_block=cq_block,
        extra_note=view.get("note", ""),
        tol=tol,
        n_shots=n_shots,
        pool=", ".join(f"{p:g}" for p in pool),
        max_turns=max_turns,
        max_proposals=max_proposals,
        starter_yaml=starter,
        dsl_cheatsheet=cheat,
    )


def build_tools(control_var: str, observable_label: str,
                pool: list[float], *, gif_only: bool = False) -> list[dict]:
    pool_str = ", ".join(f"{p:g}" for p in pool)
    tools = []
    if not gif_only:
        tools.append({
            "type": "function",
            "function": {
                "name": "propose_apparatus",
                "description": (
                    "Submit a new DSL YAML apparatus. The runner runs "
                    "Gate 1/2/3 + the physics-core lock. If any check "
                    "fails the gate index and reason are returned -- "
                    "revise the YAML and call again. On success the "
                    "YAML becomes the CURRENT apparatus and all later "
                    "run_experiment / request_more_scenes calls render "
                    "through it. A preview image is returned."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dsl_yaml": {
                            "type": "string",
                            "description": (
                                "Full DSL YAML source string. Must "
                                "preserve the locked physics core for "
                                "this scene."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "One-sentence reason for the change."
                            ),
                        },
                    },
                    "required": ["dsl_yaml"],
                },
            },
        })
    tools.extend([{
            "type": "function",
            "function": {
                "name": "run_experiment",
                "description": (
                    f"Render the scenario at one numeric value of "
                    f"`{control_var}` and return an image of the result."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "config": {
                            "type": "number",
                            "description": (
                                f"Numeric value of `{control_var}` "
                                f"to render. Must lie within the "
                                f"advertised pool: [{pool_str}]."
                            ),
                        },
                    },
                    "required": ["config"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "request_more_scenes",
                "description": (
                    "Render multiple values of the control variable "
                    "in one call. Useful for log-log slope estimation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "values": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": (
                                f"List of `{control_var}` values to "
                                f"render. Must be a subset of the "
                                f"advertised pool: [{pool_str}]."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": "One-sentence reason.",
                        },
                    },
                    "required": ["values"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_law",
                "description": (
                    f"COMMIT. Submit a closed-form expression for the "
                    f"observable you identified, as a function of "
                    f"`{control_var}`. Submission is irreversible."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "law_expr": {
                            "type": "string",
                            "description": (
                                f"A closed-form expression in the "
                                f"single variable `{control_var}`. "
                                f"All coefficients and exponents must "
                                f"be explicit numbers. Submit ONLY the "
                                f"RHS expression (e.g. `4.9*{control_var}`), "
                                f"not `y=...` or `d=...`."
                            ),
                        },
                        "rationale": {
                            "type": "string",
                            "description": (
                                "Must include VARIABLE IDENTIFICATION, "
                                "DATA TABLE, FIT METHOD, DERIVATION."
                            ),
                        },
                    },
                    "required": ["law_expr", "rationale"],
                },
            },
        },
    ])
    return tools


# ---------------------------------------------------------------------------
# Render dispatch — one path for everything: DSL parse + validate +
# compile_and_render. compile_and_render itself routes ScenarioBackend
# scenes to the matplotlib path; mechanical scenes go through MuJoCo.
# ---------------------------------------------------------------------------


def _bake_cv_into_yaml(yaml_src: str, control_value: float) -> str:
    """Bind `control_value` into geometric params for motion sources whose
    DSL params are themselves the runtime control variable.

    Without this, e.g. a model-proposed `Pendulum.string.length=1.0` would
    clash with `compile_and_render(control_value=2.5)`: the rendered
    string is 1m long but the motion-loop and time-axis logic uses L=2.5.
    Same problem with `SpringBlock.mass` and `CircularMotion.radius`.

    We rewrite the parsed YAML in-place, save it back, and return the new
    source string. Other motion sources (FreefallBall, HorizontalLaunch,
    InclinedRamp) drive cv through the time axis only -- no geometry
    rebinding required.
    """
    import yaml as _yaml
    try:
        raw = _yaml.safe_load(yaml_src)
    except _yaml.YAMLError:
        return yaml_src
    if not isinstance(raw, dict) or "scene" not in raw:
        return yaml_src
    scene = raw["scene"]
    ents = scene.get("entities") or []
    cv = float(control_value)
    changed = False
    for ent in ents:
        if not isinstance(ent, dict):
            continue
        kind = ent.get("type")
        params = ent.get("params") or {}
        if kind == "Pendulum":
            string = params.get("string") or {}
            if isinstance(string, dict) and "length" in string:
                if float(string.get("length", 0)) != cv:
                    string["length"] = cv
                    params["string"] = string
                    ent["params"] = params
                    changed = True
        elif kind == "SpringBlock":
            if float(params.get("mass", 0)) != cv:
                params["mass"] = cv
                ent["params"] = params
                changed = True
        elif kind == "CircularMotion":
            if float(params.get("radius", 0)) != cv:
                params["radius"] = cv
                ent["params"] = params
                changed = True
    if not changed:
        return yaml_src
    return _yaml.safe_dump(raw, sort_keys=False, default_flow_style=False)


def frames_to_gif_bytes(frames: "list[Image.Image]", fps: int = 8) -> bytes:
    """Encode a frame list as an animated GIF. Returns raw bytes (so
    the caller can either save to disk or base64-embed in HTML).
    Single-frame inputs degenerate to a static GIF.
    """
    if not frames:
        return b""
    buf = io.BytesIO()
    duration_ms = max(1, int(1000 / max(fps, 1)))
    if len(frames) == 1:
        frames[0].save(buf, format="GIF", optimize=True)
    else:
        frames[0].save(
            buf, format="GIF", save_all=True,
            append_images=frames[1:], duration=duration_ms,
            loop=0, optimize=True, disposal=2,
        )
    return buf.getvalue()


def render_animation_unified(
    view: dict[str, Any], scenario: Scenario, control_value: float,
    *, n_frames: int = 24, width: int = 960, height: int = 600,
    seed: int = 0,
) -> tuple["list[Image.Image]", int]:
    """Unified GIF-friendly entry for demo builders. Returns
    `(frames, fps)`.

    Routing:
      * mechanical scenes -> compile_and_render_animation (DSL/MuJoCo)
      * SB scenes that visual_mujoco supports -> visual_mujoco.render_animation
      * fallback (shouldn't happen) -> single-frame from compile_and_render

    Mechanical scenes already render dark/3D via DSL. The 7 SB scenes
    use the parallel visual_mujoco/* package to bypass matplotlib and
    hit the same dark-themed MuJoCo aesthetic as `draft/demo.html`.
    """
    from galileo.dsl.api import compile_and_render_animation
    from galileo.visual_mujoco import render_animation as _vm_render
    from galileo.visual_mujoco import supports as _vm_supports

    if _vm_supports(view["id"]):
        return _vm_render(view["id"], scenario, control_value,
                          n_frames=n_frames, width=width, height=height)

    yaml_src = view["yaml_builder"](float(control_value))
    yaml_src = _bake_cv_into_yaml(yaml_src, control_value)
    target = view.get("target_body")
    alpha = float(getattr(scenario, "alpha", 1.0))
    frames, fps = compile_and_render_animation(
        yaml_src,
        control_value=float(control_value),
        alpha=alpha, width=width, height=height,
        target=target,
    )
    return frames, fps


def render_image(view: dict[str, Any], scenario: Scenario,
                 control_value: float, *, yaml_override: str | None = None,
                 seed: int = 0, clean: bool = False) -> Image.Image:
    """Render the scene at `control_value`.

    Routing:
      * scenario_backend scenes with visual_mujoco support → dark 3D
        MuJoCo render (last frame of animation at the target state).
      * mechanical scenes → DSL compile_and_render pipeline (MuJoCo
        strobe).

    `clean` forwards to compile_and_render to suppress on-canvas text.
    """
    from galileo.visual_mujoco import supports as _vm_supports
    from galileo.visual_mujoco import render_animation as _vm_render

    scene_id = view.get("id", "")
    if (view.get("render_path") == "scenario_backend"
            and _vm_supports(scene_id)):
        frames, _fps = _vm_render(
            scene_id, scenario, float(control_value),
            n_frames=2, width=1280, height=960,
        )
        return frames[-1].convert("RGB")

    if yaml_override is not None:
        yaml_src = _bake_cv_into_yaml(yaml_override, control_value)
        target = None
    else:
        yaml_src = view["yaml_builder"](float(control_value))
        target = view.get("target_body")
    alpha = float(getattr(scenario, "alpha", 1.0))
    return compile_and_render(
        yaml_src,
        control_value=float(control_value),
        alpha=alpha,
        width=1280, height=960,
        target=target,
        seed=seed,
        clean=clean,
    )


def render_gif_keyframes(
    view: dict[str, Any], scenario: Scenario, control_value: float,
    *, n_keyframes: int = 6, seed: int = 0,
) -> list[Image.Image]:
    """Render evenly spaced animation keyframes from the scene's default
    builder. Returns `n_keyframes` RGB PIL images.

    Used by --gif-only baseline and by the instrument mode's initial
    observation (before any propose_apparatus call).
    """
    frames, _fps = render_animation_unified(
        view, scenario, float(control_value),
        n_frames=max(n_keyframes * 4, 24),
        width=1280, height=960, seed=seed,
    )
    if len(frames) <= n_keyframes:
        return [f.convert("RGB") for f in frames]
    step = max(1, len(frames) // n_keyframes)
    selected = [frames[i * step].convert("RGB")
                for i in range(n_keyframes)
                if i * step < len(frames)]
    if len(selected) < n_keyframes and frames:
        selected.append(frames[-1].convert("RGB"))
    return selected


# ---------------------------------------------------------------------------
# Scoring (same log-residual judge as run_unified.py)
# ---------------------------------------------------------------------------

def parse_law(expr_str: str, control_var: str) -> sp.Expr:
    x = sp.symbols(control_var, real=True, positive=True)
    local = {control_var: x, "pi": sp.pi, "e": sp.E,
             "sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log,
             "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
             "ln": sp.log}
    return sp.sympify(expr_str, locals=local)


def normalize_law_expr(expr_str: str) -> str:
    """Normalize model-submitted law expressions.

    Common model mistakes include returning `y = ...` or wrapping in
    backticks/fences. We keep only the RHS expression and map `^` -> `**`.
    """
    s = (expr_str or "").strip()
    if not s:
        return s
    s = s.strip("`")
    if "\n" in s:
        s = " ".join(line.strip() for line in s.splitlines() if line.strip())
    # Accept assignments like `d = 4.9*t**2` by keeping RHS.
    if "=" in s:
        parts = s.split("=")
        if len(parts) >= 2:
            s = parts[-1].strip()
    # Remove leading "law_expr:" if model adds labels.
    s = re.sub(r"^(law_expr|expression|equation)\s*:\s*", "", s, flags=re.I)
    # Sympy expects **, while many models output caret.
    s = s.replace("^", "**")
    return s.strip()


NEWTON_EXPONENTS: dict[str, float] = {
    "freefall": 2.0,
    "freefall_mass": 0.0,   # Newtonian freefall is mass-independent.
    "pendulum": 0.5,
    "spring": 1.0,
    "projectile": 2.0,
    "refraction": 1.0,
    "orbital": 3.0,
    "boyle": 1.0,
    "heat": 0.5,
    "damped": 1.0,
    "diffusion2d": 0.5,
    "coulomb": -2.0,
    "blackbody": -1.0,
    "wave": 0.5,
    "stefan": 4.0,
    "viscosity": 2.0,
}


def estimate_power_exponent(rows: list[dict[str, Any]]) -> float | None:
    """Estimate effective power exponent p from y_hat ~ x^p in log-space."""
    xs: list[float] = []
    ys: list[float] = []
    for r in rows:
        if "y_hat" not in r:
            continue
        x = float(r["cv"])
        y = float(r["y_hat"])
        if (not np.isfinite(x) or not np.isfinite(y)
                or x <= 0.0 or y <= 0.0):
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < 2:
        return None
    lx = np.log(np.asarray(xs))
    ly = np.log(np.asarray(ys))
    try:
        p, _b = np.polyfit(lx, ly, 1)
        if np.isfinite(p):
            return float(p)
    except Exception:
        return None
    return None


def score_law(
    scenario: Scenario,
    expr_str: str,
    tol: float,
    pass_mode: str = "strict",
    exponent_tol: float = 0.30,
) -> dict[str, Any]:
    var = scenario.meta.control_var
    refs = scenario.default_controls
    try:
        expr = parse_law(expr_str, var)
    except Exception as e:
        return {"ok": False, "reason": f"parse_error: {e}",
                "tol": tol, "joint_log_residual": None}
    x = sp.symbols(var, real=True, positive=True)
    f = sp.lambdify(x, expr, modules=["numpy"])
    rows: list[dict[str, float]] = []
    for cv in refs:
        try:
            y_hat = float(f(cv))
        except Exception as e:
            rows.append({"cv": cv, "error": f"eval_error: {e}"})
            continue
        y_true = float(scenario.get_observable(cv))
        rows.append({
            "cv": cv, "y_true": y_true, "y_hat": y_hat,
            "abs_err": abs(y_hat - y_true),
            "log_err": abs(math.log(abs(y_hat) + 1e-9)
                           - math.log(abs(y_true) + 1e-9)),
        })
    log_errs = [r["log_err"] for r in rows if "log_err" in r]
    if not log_errs:
        return {"ok": False, "reason": "all evaluations failed",
                "rows": rows, "tol": tol, "joint_log_residual": None}
    joint = float(np.sqrt(np.mean(np.square(log_errs))))
    strict_ok = joint <= tol

    slug = getattr(getattr(scenario, "meta", None), "slug", None)
    inferred_p = estimate_power_exponent(rows)
    gt_p = (float(getattr(scenario, "alpha", np.nan))
            if slug in NEWTON_EXPONENTS else None)
    newton_p = NEWTON_EXPONENTS.get(slug) if slug else None

    relaxed_ok = False
    dist_gt = None
    dist_newton = None
    if (inferred_p is not None and gt_p is not None
            and newton_p is not None and np.isfinite(gt_p)):
        dist_gt = abs(inferred_p - gt_p)
        dist_newton = abs(inferred_p - newton_p)
        close_enough = dist_gt <= float(exponent_tol)
        closer_than_newton = dist_gt < dist_newton
        relaxed_ok = bool(close_enough or closer_than_newton)

    ok = strict_ok
    if pass_mode == "relaxed_exponent":
        ok = bool(strict_ok or relaxed_ok)

    return {
        "ok": ok,
        "strict_ok": bool(strict_ok),
        "relaxed_ok": bool(relaxed_ok),
        "pass_mode": pass_mode,
        "joint_log_residual": round(joint, 4),
        "tol": tol,
        "rows": rows,
        "inferred_exponent": inferred_p,
        "groundtruth_exponent": gt_p,
        "newton_exponent": newton_p,
        "exp_dist_to_groundtruth": dist_gt,
        "exp_dist_to_newton": dist_newton,
        "exponent_tol": float(exponent_tol),
    }


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    scene_id: str
    slug: str
    render_path: str
    model: str
    seed: int
    alpha: float | None
    kwargs: dict[str, Any]
    dsl_yaml: str = ""
    submitted: dict[str, Any] | None = None
    score: dict[str, Any] | None = None
    turns_used: int = 0
    elapsed_s: float = 0.0
    error: str | None = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)


def run_one_scene(view: dict[str, Any], provider: VLMProvider,
                  *, seed: int, max_turns: int, n_shots: int, tol: float,
                  max_proposals: int, gif_only: bool = False,
                  physics_profile: str = "non_newton",
                  pass_mode: str = "strict",
                  exponent_tol: float = 0.30,
                  out_dir: Path) -> RunRecord:
    slug = view["slug"]
    scenario = _build_scenario(view, seed=seed, physics_profile=physics_profile)
    pool = scenario.default_controls
    init_cv_preview = float(pool[len(pool) // 2])

    starter_yaml = view["yaml_builder"](init_cv_preview)
    current_apparatus_yaml = starter_yaml
    proposals_left = int(max_proposals)
    apparatus_history: list[dict[str, Any]] = []

    record = RunRecord(
        scene_id=view["id"], slug=slug,
        render_path=view.get("render_path", "scenario_backend"),
        model=provider.model, seed=seed,
        alpha=getattr(scenario, "alpha", None),
        kwargs={k: v for k, v in DEFAULT_SCENARIO_KWARGS.get(slug, {}).items()},
        dsl_yaml=starter_yaml,
    )

    t0 = time.time()
    sys_prompt = build_system_prompt(view, scenario, n_shots, max_turns, tol,
                                     max_proposals,
                                     gif_only=gif_only)
    tools = build_tools(scenario.meta.control_var,
                        scenario.meta.observable_label, pool,
                        gif_only=gif_only)

    messages: list[Message] = [Message(role="system", content=sys_prompt)]

    init_cv = init_cv_preview
    images_dir = out_dir / f"{view['id']}_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if gif_only:
        try:
            keyframes = render_gif_keyframes(
                view, scenario, init_cv, n_keyframes=6, seed=seed)
        except Exception as e:
            record.error = f"render_error: {e}"
            record.elapsed_s = round(time.time() - t0, 2)
            out_json = out_dir / f"{view['id']}.json"
            out_json.write_text(json.dumps(asdict(record), indent=2))
            return record
        init_content: list[dict] = [
            {"type": "text",
             "text": (f"Animation keyframes at "
                      f"`{scenario.meta.control_var}`={init_cv:g}. "
                      f"You are seeing {len(keyframes)} frames of the "
                      f"phenomenon in motion (earliest → latest). "
                      f"Pool of allowed control values: "
                      f"{[round(p, 4) for p in pool]}. "
                      f"turns_remaining={max_turns}.")},
        ]
        for fi, kf in enumerate(keyframes):
            init_content.append({"type": "text",
                                 "text": f"--- frame {fi+1}/{len(keyframes)} ---"})
            init_content.append({"type": "image_url",
                                 "image_url": {"url": image_to_data_url(kf),
                                               "detail": "high"}})
            kf.save(images_dir / f"init_cv{init_cv:g}_f{fi:02d}.png")
        messages.append(Message(role="user", content=init_content))
        record.tool_trace.append({"step": 0, "type": "initial_gif",
                                  "control_value": init_cv,
                                  "n_keyframes": len(keyframes)})
    else:
        try:
            keyframes = render_gif_keyframes(
                view, scenario, init_cv, n_keyframes=6, seed=seed)
        except Exception as e:
            record.error = f"render_error: {e}"
            record.elapsed_s = round(time.time() - t0, 2)
            out_json = out_dir / f"{view['id']}.json"
            out_json.write_text(json.dumps(asdict(record), indent=2))
            return record
        init_content = [
            {"type": "text",
             "text": (f"Initial animation keyframes through the DEFAULT "
                      f"apparatus at "
                      f"`{scenario.meta.control_var}`={init_cv:g}. Pool of "
                      f"allowed control values: "
                      f"{[round(p, 4) for p in pool]}. apparatus_proposals_"
                      f"remaining={proposals_left}, "
                      f"turns_remaining={max_turns}.")},
        ]
        for fi, kf in enumerate(keyframes):
            init_content.append({"type": "text",
                                 "text": f"--- frame {fi+1}/{len(keyframes)} ---"})
            init_content.append({"type": "image_url",
                                 "image_url": {"url": image_to_data_url(kf),
                                               "detail": "high"}})
            kf.save(images_dir / f"init_cv{init_cv:g}_f{fi:02d}.png")
        messages.append(Message(role="user", content=init_content))
        record.tool_trace.append({"step": 0, "type": "initial_gif",
                                  "control_value": init_cv,
                                  "apparatus_yaml": current_apparatus_yaml,
                                  "n_keyframes": len(keyframes)})

    submitted: dict[str, Any] | None = None
    turns_used = 0
    for turn in range(max_turns):
        turns_used = turn + 1
        try:
            assistant = provider.chat(messages, tools=tools)
        except Exception as e:
            record.error = f"provider_error: {e}"
            break
        messages.append(assistant)

        if not assistant.tool_calls:
            if turn >= max_turns - 1:
                record.error = "no_submission_in_budget"
                break
            avail = ("run_experiment, request_more_scenes, or submit_law"
                     if gif_only else
                     "propose_apparatus, run_experiment, "
                     "request_more_scenes, or submit_law")
            messages.append(Message(role="user", content=(
                f"You produced no tool call. You must use one of "
                f"{avail}. turns_remaining={max_turns - turns_used}.")))
            continue

        for tc in assistant.tool_calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except Exception:
                args = {}
            tool_id = tc.get("id", "tc")
            if fn == "propose_apparatus":
                proposed_yaml = str(args.get("dsl_yaml", "")).strip()
                reason = str(args.get("reason", ""))
                if proposals_left <= 0:
                    messages.append(Message(
                        role="tool", tool_call_id=tool_id, name=fn,
                        content="REJECTED: no apparatus proposals left."))
                    record.tool_trace.append({
                        "step": turn, "type": fn, "ok": False,
                        "reason": "budget exhausted",
                    })
                    continue
                proposals_left -= 1
                gate_res = dsl_validate(proposed_yaml)
                if not gate_res.ok:
                    msg = (f"REJECTED [gate {gate_res.gate}]: "
                           f"{gate_res.reason}\n"
                           f"apparatus_proposals_remaining={proposals_left}.")
                    messages.append(Message(
                        role="tool", tool_call_id=tool_id, name=fn,
                        content=msg))
                    apparatus_history.append({
                        "step": turn, "ok": False, "gate": gate_res.gate,
                        "reason": gate_res.reason, "yaml": proposed_yaml,
                    })
                    record.tool_trace.append({
                        "step": turn, "type": fn, "ok": False,
                        "gate": gate_res.gate, "reason": gate_res.reason,
                    })
                    continue
                core_ok, core_reason = check_physics_core(view, proposed_yaml)
                if not core_ok:
                    messages.append(Message(
                        role="tool", tool_call_id=tool_id, name=fn,
                        content=(f"REJECTED: {core_reason}\n"
                                 f"apparatus_proposals_remaining="
                                 f"{proposals_left}.")))
                    apparatus_history.append({
                        "step": turn, "ok": False,
                        "reason": core_reason, "yaml": proposed_yaml,
                    })
                    record.tool_trace.append({
                        "step": turn, "type": fn, "ok": False,
                        "reason": core_reason,
                    })
                    continue
                try:
                    preview = render_image(view, scenario, init_cv,
                                           yaml_override=proposed_yaml,
                                           seed=seed, clean=True)
                except Exception as e:
                    messages.append(Message(
                        role="tool", tool_call_id=tool_id, name=fn,
                        content=(f"REJECTED [render]: {e}\n"
                                 f"apparatus_proposals_remaining="
                                 f"{proposals_left}.")))
                    apparatus_history.append({
                        "step": turn, "ok": False,
                        "reason": f"render: {e}", "yaml": proposed_yaml,
                    })
                    record.tool_trace.append({
                        "step": turn, "type": fn, "ok": False,
                        "reason": f"render: {e}",
                    })
                    continue
                current_apparatus_yaml = proposed_yaml
                images_dir.mkdir(parents=True, exist_ok=True)
                preview.save(images_dir / f"t{turn:02d}_apparatus_preview.png")
                apparatus_history.append({
                    "step": turn, "ok": True, "reason": reason,
                    "yaml": proposed_yaml,
                })
                record.tool_trace.append({
                    "step": turn, "type": fn, "ok": True,
                    "reason": reason,
                })
                messages.append(Message(
                    role="tool", tool_call_id=tool_id, name=fn,
                    content=[
                        {"type": "text",
                         "text": (f"OK: apparatus accepted. Preview at "
                                  f"`{scenario.meta.control_var}`="
                                  f"{init_cv:g}. apparatus_proposals_"
                                  f"remaining={proposals_left}, "
                                  f"turns_remaining="
                                  f"{max_turns - turns_used}.")},
                        {"type": "image_url",
                         "image_url": {"url": image_to_data_url(preview),
                                       "detail": "high"}},
                    ]))
            elif fn == "run_experiment":
                cv = float(args.get("config", init_cv))
                cv = min(max(cv, min(pool)), max(pool))
                if gif_only:
                    try:
                        kfs = render_gif_keyframes(
                            view, scenario, cv, n_keyframes=6, seed=seed)
                    except Exception as e:
                        messages.append(Message(
                            role="tool", tool_call_id=tool_id, name=fn,
                            content=f"render_error: {e}"))
                        record.tool_trace.append({"step": turn, "type": fn,
                                                  "control_value": cv,
                                                  "error": str(e)})
                        continue
                    images_dir.mkdir(parents=True, exist_ok=True)
                    gif_content: list[dict] = [
                        {"type": "text",
                         "text": (f"Animation at cv={cv:g} "
                                  f"({len(kfs)} frames). "
                                  f"turns_remaining="
                                  f"{max_turns - turns_used}.")},
                    ]
                    for fi, kf in enumerate(kfs):
                        kf.save(images_dir /
                                f"t{turn:02d}_cv{cv:g}_f{fi:02d}.png")
                        gif_content.append(
                            {"type": "text",
                             "text": f"--- frame {fi+1}/{len(kfs)} ---"})
                        gif_content.append(
                            {"type": "image_url",
                             "image_url": {"url": image_to_data_url(kf),
                                           "detail": "high"}})
                    record.tool_trace.append({"step": turn, "type": fn,
                                              "control_value": cv})
                    messages.append(Message(
                        role="tool", tool_call_id=tool_id, name=fn,
                        content=gif_content))
                else:
                    try:
                        img = render_image(view, scenario, cv,
                                           yaml_override=current_apparatus_yaml,
                                           seed=seed, clean=True)
                    except Exception as e:
                        messages.append(Message(
                            role="tool", tool_call_id=tool_id, name=fn,
                            content=f"render_error: {e}"))
                        record.tool_trace.append({"step": turn, "type": fn,
                                                  "control_value": cv,
                                                  "error": str(e)})
                        continue
                    images_dir.mkdir(parents=True, exist_ok=True)
                    img.save(images_dir / f"t{turn:02d}_cv{cv:g}.png")
                    record.tool_trace.append({"step": turn, "type": fn,
                                              "control_value": cv})
                    messages.append(Message(
                        role="tool", tool_call_id=tool_id, name=fn,
                        content=[
                            {"type": "text",
                             "text": f"rendered cv={cv:g}. "
                                     f"turns_remaining="
                                     f"{max_turns - turns_used}."},
                            {"type": "image_url",
                             "image_url": {"url": image_to_data_url(img),
                                           "detail": "high"}},
                        ]))
            elif fn == "request_more_scenes":
                vals = [float(v) for v in args.get("values", [])][:6]
                imgs_content: list[dict] = [
                    {"type": "text",
                     "text": (f"rendered {len(vals)} scenes. "
                              f"turns_remaining={max_turns - turns_used}.")}
                ]
                rendered_cvs = []
                for v in vals:
                    v = min(max(v, min(pool)), max(pool))
                    if gif_only:
                        try:
                            kfs = render_gif_keyframes(
                                view, scenario, v,
                                n_keyframes=6, seed=seed)
                        except Exception as e:
                            imgs_content.append(
                                {"type": "text",
                                 "text": f"cv={v:g} ERROR: {e}"})
                            continue
                        images_dir.mkdir(parents=True, exist_ok=True)
                        rendered_cvs.append(v)
                        imgs_content.append(
                            {"type": "text",
                             "text": f"--- cv={v:g} "
                                     f"({len(kfs)} frames) ---"})
                        for fi, kf in enumerate(kfs):
                            kf.save(images_dir /
                                    f"t{turn:02d}_cv{v:g}_f{fi:02d}.png")
                            imgs_content.append(
                                {"type": "image_url",
                                 "image_url": {
                                     "url": image_to_data_url(kf),
                                     "detail": "high"}})
                    else:
                        try:
                            img = render_image(view, scenario, v,
                                               yaml_override=
                                               current_apparatus_yaml,
                                               seed=seed, clean=True)
                        except Exception as e:
                            imgs_content.append(
                                {"type": "text",
                                 "text": f"cv={v:g} ERROR: {e}"})
                            continue
                        images_dir.mkdir(parents=True, exist_ok=True)
                        img.save(images_dir / f"t{turn:02d}_cv{v:g}.png")
                        rendered_cvs.append(v)
                        imgs_content.append(
                            {"type": "text", "text": f"--- cv={v:g} ---"})
                        imgs_content.append(
                            {"type": "image_url",
                             "image_url": {"url": image_to_data_url(img),
                                           "detail": "high"}})
                record.tool_trace.append({"step": turn, "type": fn,
                                          "control_values": rendered_cvs})
                messages.append(Message(
                    role="tool", tool_call_id=tool_id, name=fn,
                    content=imgs_content))
            elif fn == "submit_law":
                raw_expr = str(args.get("law_expr", ""))
                expr_str = normalize_law_expr(raw_expr)
                rationale = str(args.get("rationale", ""))
                score = score_law(
                    scenario, expr_str, tol,
                    pass_mode=pass_mode, exponent_tol=exponent_tol,
                )
                # Do not terminate the episode on pure formatting/parse errors:
                # give the model a chance to re-submit a valid expression.
                if str(score.get("reason", "")).startswith("parse_error"):
                    messages.append(Message(
                        role="tool", tool_call_id=tool_id, name=fn,
                        content=("REJECTED: parse_error. Submit ONLY a valid "
                                 "RHS expression (no `y=`/`d=` prefix), e.g. "
                                 "`4.9*max_t`.")))
                    record.tool_trace.append({
                        "step": turn, "type": fn,
                        "law_expr_raw": raw_expr,
                        "law_expr_norm": expr_str,
                        "ok": False,
                        "reason": score.get("reason"),
                    })
                    continue

                submitted = {"law_expr": expr_str, "rationale": rationale}
                record.tool_trace.append({"step": turn, "type": fn,
                                          "law_expr_raw": raw_expr,
                                          "law_expr": expr_str,
                                          "ok": bool(score.get("ok"))})
                messages.append(Message(
                    role="tool", tool_call_id=tool_id, name=fn,
                    content=("PASS" if score.get("ok") else "FAIL")
                            + f" joint_log_residual="
                              f"{score.get('joint_log_residual')}"))
                record.submitted = submitted
                record.score = score
                break
            else:
                messages.append(Message(
                    role="tool", tool_call_id=tool_id, name=fn,
                    content=f"unknown tool: {fn}"))
        if submitted is not None:
            break

    record.turns_used = turns_used
    record.elapsed_s = round(time.time() - t0, 2)
    record.dsl_yaml = current_apparatus_yaml
    if record.submitted is None and record.error is None:
        record.error = "exited_without_submission"

    out_json = out_dir / f"{view['id']}.json"
    serialised = asdict(record)
    serialised["apparatus_history"] = apparatus_history
    out_json.write_text(json.dumps(serialised, indent=2))
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="all",
                    help="scene id (e.g. s1_freefall) or 'all'")
    ap.add_argument("--model", default="gpt-5.5-medium")
    ap.add_argument("--api-base", default=os.environ.get(
        "GALILEO_API_BASE", "https://api.qingyuntop.top/v1"))
    ap.add_argument("--api-key", default=os.environ.get(
        "GALILEO_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    ap.add_argument("--max-turns", type=int, default=14)
    ap.add_argument("--max-proposals", type=int, default=5,
                    help="apparatus proposal budget per scene")
    ap.add_argument("--gif-only", action="store_true",
                    help="No-instrument baseline: show animation frames "
                         "only (no StrobeTrail, no propose_apparatus)")
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--tol", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="galileo/outputs/unified_dsl")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--physics-profile", default="non_newton",
                    choices=["non_newton", "textbook"],
                    help="Physics parameter profile used for scoring/render "
                         "ground truth.")
    ap.add_argument("--pass-mode", default="strict",
                    choices=["strict", "relaxed_exponent"],
                    help="Pass criterion: strict residual, or exponent-based "
                         "relaxed mode.")
    ap.add_argument("--exponent-tol", type=float, default=0.30,
                    help="Exponent-distance threshold for relaxed mode.")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for v in SCENE_VIEWS:
            print(f"{v['id']:16s}  slug={v['slug']:12s}  "
                  f"render_path={v['render_path']:18s}  "
                  f"alpha={DEFAULT_COUNTERFACTUALS.get(v['slug'])}")
        return 0

    if not args.api_key:
        print("ERROR: --api-key or env GALILEO_API_KEY is required.",
              file=sys.stderr)
        return 2

    out_dir = Path(args.out).resolve() / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scene == "all":
        views = SCENE_VIEWS
    else:
        views = [v for v in SCENE_VIEWS if v["id"] == args.scene]
        if not views:
            print(f"unknown scene: {args.scene}", file=sys.stderr)
            print(f"available: {[v['id'] for v in SCENE_VIEWS]}",
                  file=sys.stderr)
            return 2

    provider = VLMProvider(
        api_base=args.api_base, api_key=args.api_key,
        model=args.model, seed=args.seed,
        temperature=args.temperature, max_tokens=4096,
    )

    summary: list[dict[str, Any]] = []
    for v in views:
        print(f"\n=== {v['id']} ({v['slug']}, "
              f"render_path={v['render_path']}) ===", flush=True)
        try:
            rec = run_one_scene(
                v, provider, seed=args.seed, max_turns=args.max_turns,
                n_shots=args.shots, tol=args.tol,
                max_proposals=args.max_proposals,
                gif_only=args.gif_only,
                physics_profile=args.physics_profile,
                pass_mode=args.pass_mode,
                exponent_tol=args.exponent_tol,
                out_dir=out_dir,
            )
        except Exception as e:
            traceback.print_exc()
            rec = RunRecord(
                scene_id=v["id"], slug=v["slug"],
                render_path=v.get("render_path", "scenario_backend"),
                model=args.model, seed=args.seed, alpha=None, kwargs={},
                error=str(e),
            )
        summary.append({
            "id": rec.scene_id, "slug": rec.slug,
            "render_path": rec.render_path,
            "submitted": (rec.submitted or {}).get("law_expr"),
            "score": (rec.score or {}).get("joint_log_residual"),
            "pass": bool((rec.score or {}).get("ok")),
            "turns": rec.turns_used, "elapsed_s": rec.elapsed_s,
            "error": rec.error,
        })
        print(f"  -> submitted={summary[-1]['submitted']!r}  "
              f"score={summary[-1]['score']}  "
              f"pass={summary[-1]['pass']}  "
              f"turns={summary[-1]['turns']}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    n_pass = sum(1 for s in summary if s["pass"])
    n_mj = sum(1 for s in summary if s["render_path"] == "mujoco_strobe")
    n_mj_pass = sum(1 for s in summary
                    if s["render_path"] == "mujoco_strobe" and s["pass"])
    n_sb = sum(1 for s in summary if s["render_path"] == "scenario_backend")
    n_sb_pass = sum(1 for s in summary
                    if s["render_path"] == "scenario_backend" and s["pass"])
    print(f"\nWrote {out_dir}/summary.json "
          f"({n_pass}/{len(summary)} pass; "
          f"mujoco_strobe={n_mj_pass}/{n_mj}, "
          f"scenario_backend={n_sb_pass}/{n_sb})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
