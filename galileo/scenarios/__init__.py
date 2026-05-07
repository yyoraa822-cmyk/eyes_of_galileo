"""Scenario registry — 20 counterfactual physics scenarios."""
from __future__ import annotations

from .base import Scenario, ScenarioMeta
from .freefall import Freefall, FreefallMass
from .pendulum import Pendulum
from .spring import Spring
from .projectile import Projectile
from .refraction import Refraction
from .orbital import Orbital
from .boyle import Boyle
from .heat import HeatConduction
from .damped import DampedOscillation
from .decay import ExponentialDecay
from .logarithmic import LogarithmicResponse
from .cooling import Cooling
from .capacitor import Capacitor
from .diffusion2d import Diffusion2D
from .coulomb import Coulomb
from .blackbody import Blackbody
from .wave import WaveOnString
from .stefan import Stefan
from .viscosity import Viscosity
from .ohm import OhmLaw

SCENARIO_REGISTRY: dict[str, type[Scenario]] = {
    # ── Original 11 ──
    "freefall": Freefall,
    "freefall_mass": FreefallMass,
    "pendulum": Pendulum,
    "spring": Spring,
    "projectile": Projectile,
    "refraction": Refraction,
    "orbital": Orbital,
    "boyle": Boyle,
    "heat": HeatConduction,
    "damped": DampedOscillation,
    "decay": ExponentialDecay,
    "logarithmic": LogarithmicResponse,
    # ── New 9 ──
    "cooling": Cooling,
    "capacitor": Capacitor,
    "diffusion2d": Diffusion2D,
    "coulomb": Coulomb,
    "blackbody": Blackbody,
    "wave": WaveOnString,
    "stefan": Stefan,
    "viscosity": Viscosity,
    "ohm": OhmLaw,
}

# counterfactual exponents / parameters (different from textbook values)
DEFAULT_COUNTERFACTUALS: dict[str, float] = {
    # Original 11 — gaps widened to ≥40% for visual distinguishability
    "freefall": 1.0,       # textbook = 2.0   (50% gap — linear vs quadratic!)
    "freefall_mass": 0.5,  # textbook = 0 (mass irrelevant in Newton); here d ∝ m^0.5
    "pendulum": 0.25,      # textbook = 0.5   (50% gap)
    "spring": 0.5,         # textbook = 1.0   (50% gap)
    "projectile": 1.0,     # textbook = 2.0   (50% gap)
    "refraction": 0.5,     # textbook = 1.0   (50% gap)
    "orbital": 2.0,        # textbook = 3.0   (33% gap)
    "boyle": 0.5,          # textbook = 1.0   (50% gap)
    "heat": 0.25,          # textbook = 0.5   (50% gap)
    "damped": 0.5,         # textbook = 1.0   (50% gap)
    "decay": 0.0,          # not used (rate param instead)
    "logarithmic": 0.0,    # not used (coefficient param instead)
    # New power-law scenarios
    "diffusion2d": 0.25,   # textbook = 0.5   (50% gap)
    "coulomb": -1.0,       # textbook = -2.0  (50% gap)
    "blackbody": -0.5,     # textbook = -1.0  (50% gap)
    "wave": 0.25,          # textbook = 0.5   (50% gap)
    "stefan": 2.5,         # textbook = 4.0   (37.5% gap)
    "viscosity": 1.0,      # textbook = 2.0   (50% gap)
    # Non-power-law (use kwargs)
    "cooling": 0.0,
    "capacitor": 0.0,
    "ohm": 0.0,
}

# non-power-law scenarios use kwargs instead of alpha
DEFAULT_SCENARIO_KWARGS: dict[str, dict] = {
    "decay": {"rate": -0.20},         # textbook ~-0.5  (60% gap)
    "logarithmic": {"coeff": 3.0},    # textbook ~1.0   (200% gap)
    "cooling": {"rate": -0.10},       # textbook ~-0.30 (67% gap)
    "capacitor": {"rate": -0.8},      # textbook ~-2.0  (60% gap)
    "ohm": {"slope": 2.5},            # textbook ~1.0   (150% gap)
}


def get_scenario(name: str, alpha: float | None = None, **kw) -> Scenario:
    """Instantiate a scenario by slug name.

    If alpha is None, uses the default counterfactual exponent.
    For non-power-law scenarios, also injects default kwargs (rate, coeff, etc.).
    """
    cls = SCENARIO_REGISTRY[name]
    if alpha is None:
        alpha = DEFAULT_COUNTERFACTUALS.get(name)
    extra = dict(DEFAULT_SCENARIO_KWARGS.get(name, {}))
    extra.update(kw)
    return cls(alpha=alpha, **extra)


def list_scenarios() -> list[str]:
    return list(SCENARIO_REGISTRY.keys())
