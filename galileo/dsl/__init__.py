"""Galileo-DSL: a domain-specific language for VLM apparatus invention.

Three-level abstraction:
    body    — primitive objects (ball, ramp, wall, ruler, marker, reference_cube)
    entity  — physically-meaningful compositions (InclinedRamp, FreefallBall,
              Pendulum, StrobeTrail, ReferenceScale)
    scene   — a set of entities + connections + camera

Three-gate validator:
    Gate 1 (syntactic):  schema + reference resolution
    Gate 2 (physical):   MuJoCo compile + placement sanity
    Gate 3 (degeneracy): camera visibility + observability

Rendering reuses the counterfactual-override trick from mujoco_freefall.py:
the ball's position over time is NOT from MuJoCo integration, but computed
from a hidden law s(t) = C * t^alpha. The DSL-described apparatus is rendered
faithfully; only the dynamics of the tracked body are overridden.
"""
from .api import (
    compile_and_render, compile_and_render_animation,
    scene_has_animation, list_targets, validate,
    DSLError, ValidationResult,
)

__all__ = [
    "compile_and_render", "compile_and_render_animation",
    "scene_has_animation", "list_targets", "validate",
    "DSLError", "ValidationResult",
]
