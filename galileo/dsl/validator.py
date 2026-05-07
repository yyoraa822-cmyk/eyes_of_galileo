"""Three-gate validator for Galileo-DSL.

Gate 1 (syntactic):  handled by parser.py.
Gate 2 (physical):   all referenced bodies/entities exist, connections are
                     compatible, MuJoCo compiles the resulting MJCF.
Gate 3 (degeneracy): the apparatus is observable — camera can see the key
                     bodies, the tracked target is defined, the ramp is
                     non-trivial, etc.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco

from .schema import ENTITY_SCHEMA, Scene


@dataclass
class ValidationResult:
    ok: bool
    gate: int  # 1, 2, or 3; 0 = all passed
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


def _gate2_references(scene: Scene) -> ValidationResult:
    """All connection endpoints and cross-entity references must resolve."""
    defined_bodies: set[str] = set()
    defined_connection_points: set[str] = set()
    for ent in scene.entities.values():
        for body in ent.bodies.values():
            defined_bodies.add(body.name)
        for cp in ENTITY_SCHEMA[ent.kind].get("connection_points", []):
            defined_connection_points.add(f"{ent.name}.{cp}")
        defined_connection_points.add(ent.name)

    for ent in scene.entities.values():
        if ent.kind == "ScenarioBackend":
            slug = str(ent.params.get("slug", "")).strip()
            if not slug:
                return ValidationResult(
                    ok=False, gate=2,
                    reason=(f"ScenarioBackend '{ent.name}' missing required "
                            "param 'slug'."),
                )
            if slug not in KNOWN_SCENARIO_SLUGS:
                return ValidationResult(
                    ok=False, gate=2,
                    reason=(
                        f"ScenarioBackend '{ent.name}' slug='{slug}' is not "
                        f"a known scenario. Known: "
                        f"{sorted(KNOWN_SCENARIO_SLUGS)}."
                    ),
                )

    for ent in scene.entities.values():
        if ent.kind in ("StrobeTrail", "ChimeTrack",
                        "FadingTrail", "Filmstrip", "Animation",
                        "MassStack", "LightStrip", "SpectrumBand"):
            tgt = ent.params.get("target_body")
            if tgt not in defined_bodies:
                return ValidationResult(
                    ok=False, gate=2,
                    reason=(
                        f"{ent.kind}.target_body references '{tgt}', which "
                        f"is not a defined body. Defined bodies: "
                        f"{sorted(defined_bodies)}."
                    ),
                )
        if ent.kind == "StrobeTrail":
            n = ent.params.get("n_samples", 10)
            try:
                n_int = int(n)
            except Exception:
                return ValidationResult(
                    ok=False, gate=2,
                    reason=(
                        f"StrobeTrail '{ent.name}'.n_samples must be an "
                        f"integer, got {n!r}."
                    ),
                )
            if not (5 <= n_int <= 20):
                return ValidationResult(
                    ok=False, gate=2,
                    reason=(
                        f"StrobeTrail '{ent.name}'.n_samples={n_int} is "
                        f"out of range; must be an integer in [5, 20]."
                    ),
                )

    for i, conn in enumerate(scene.connections):
        for side, endpoint in (("from", conn.src), ("to", conn.dst)):
            base = endpoint.split(".")[0]
            if base not in scene.entities and base not in defined_bodies:
                return ValidationResult(
                    ok=False, gate=2,
                    reason=(
                        f"connections[{i}].{side}='{endpoint}' refers to "
                        f"unknown entity/body. Known entities: "
                        f"{sorted(scene.entities)}."
                    ),
                )
            if "." in endpoint:
                if endpoint not in defined_connection_points:
                    ent_name = endpoint.split(".")[0]
                    if ent_name in scene.entities:
                        valid_cps = ENTITY_SCHEMA[
                            scene.entities[ent_name].kind
                        ].get("connection_points", [])
                        return ValidationResult(
                            ok=False, gate=2,
                            reason=(
                                f"connections[{i}].{side}='{endpoint}': "
                                f"entity '{ent_name}' has no connection "
                                f"point '{endpoint.split('.', 1)[1]}'. "
                                f"Valid: {valid_cps}."
                            ),
                        )
    return ValidationResult(ok=True, gate=0)


def _gate2_mjcf_compiles(mjcf: str) -> ValidationResult:
    try:
        mujoco.MjModel.from_xml_string(mjcf)
    except Exception as e:
        return ValidationResult(
            ok=False, gate=2,
            reason=f"MuJoCo failed to compile the scene: {e}",
        )
    return ValidationResult(ok=True, gate=0)


MOTION_SOURCES = {
    "FreefallBall", "HorizontalLaunch", "Pendulum", "InclinedRamp",
    "SpringBlock", "CircularMotion", "ScenarioBackend",
}


KNOWN_SCENARIO_SLUGS = {
    "freefall", "pendulum", "projectile", "spring", "orbital",
    "heat", "decay", "refraction", "boyle", "coulomb", "blackbody",
    "cooling", "buoyancy", "drag", "ohm", "diffusion",
}


def _gate3_observability(scene: Scene) -> ValidationResult:
    has_tracked = any(e.kind in MOTION_SOURCES for e in scene.entities.values())
    if not has_tracked:
        return ValidationResult(
            ok=False, gate=3,
            reason=(
                "No motion source in the scene. Add at least one of "
                f"{sorted(MOTION_SOURCES)} so the experiment has something "
                "to observe."
            ),
        )

    for ent in scene.entities.values():
        if ent.kind == "InclinedRamp":
            ramp = ent.bodies.get("ramp")
            if ramp is not None:
                angle = float(ramp.params.get("angle_deg", 0))
                length = float(ramp.params.get("length", 0))
                if not (0.5 <= angle <= 89.5):
                    return ValidationResult(
                        ok=False, gate=3,
                        reason=(
                            f"InclinedRamp '{ent.name}' angle_deg={angle} "
                            f"is out of range [0.5, 89.5]."
                        ),
                    )
                if not (0.2 <= length <= 20.0):
                    return ValidationResult(
                        ok=False, gate=3,
                        reason=(
                            f"InclinedRamp '{ent.name}' length={length} is "
                            f"out of range [0.2, 20]."
                        ),
                    )
        if ent.kind == "HorizontalLaunch":
            v0 = float(ent.params.get("v0", 1.0))
            if not (0.05 <= v0 <= 20.0):
                return ValidationResult(
                    ok=False, gate=3,
                    reason=(
                        f"HorizontalLaunch '{ent.name}' v0={v0} out of "
                        f"range [0.05, 20]."
                    ),
                )
            h = float(ent.params.get("launch_height", 2.0))
            if not (0.2 <= h <= 15.0):
                return ValidationResult(
                    ok=False, gate=3,
                    reason=(
                        f"HorizontalLaunch '{ent.name}' launch_height={h} "
                        f"out of range [0.2, 15]."
                    ),
                )
        if ent.kind == "ChimeTrack":
            pos = ent.params.get("positions", [])
            if not isinstance(pos, list) or len(pos) < 2:
                return ValidationResult(
                    ok=False, gate=3,
                    reason=(
                        f"ChimeTrack '{ent.name}' needs >=2 positions, "
                        f"got {pos!r}."
                    ),
                )
            try:
                fs = [float(x) for x in pos]
            except Exception:
                return ValidationResult(
                    ok=False, gate=3,
                    reason=(
                        f"ChimeTrack '{ent.name}' positions must be floats."
                    ),
                )
            if any(f <= 0.0 for f in fs):
                return ValidationResult(
                    ok=False, gate=3,
                    reason=(
                        f"ChimeTrack '{ent.name}' positions must be > 0 "
                        f"(distance from origin along motion axis)."
                    ),
                )
            if sorted(fs) != fs:
                return ValidationResult(
                    ok=False, gate=3,
                    reason=(
                        f"ChimeTrack '{ent.name}' positions must be strictly "
                        f"increasing."
                    ),
                )
        if ent.kind == "TimingGate":
            axis = str(ent.params.get("axis", "x")).lower()
            if axis not in {"x", "y"}:
                return ValidationResult(
                    ok=False, gate=3,
                    reason=(
                        f"TimingGate '{ent.name}' axis must be 'x' or 'y', "
                        f"got {axis!r}."
                    ),
                )
    return ValidationResult(ok=True, gate=0)


def validate_pre_compile(scene: Scene) -> ValidationResult:
    """Gate 2 (reference resolution) + Gate 3 (observability)."""
    r = _gate2_references(scene)
    if not r.ok:
        return r
    return _gate3_observability(scene)


def validate_mjcf(mjcf: str) -> ValidationResult:
    """Gate 2 (MuJoCo compilation)."""
    return _gate2_mjcf_compiles(mjcf)
