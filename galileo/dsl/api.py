"""Public API for Galileo-DSL.

Thin wrapper that chains: parse -> validate (Gate 1 handled in parser,
Gate 2/3 here) -> compile -> render. Any stage can raise DSLError or
return a failing ValidationResult, which callers (including the VLM
agent loop) should surface back as an actionable error message.

Scenes whose sole motion source is `ScenarioBackend` skip MuJoCo
rendering and dispatch to a registered Scenario.* class — the parse
+ validate gates still run end-to-end, so all 15 benchmark scenes go
through the same DSL pipeline regardless of which renderer produces
the final image.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .compiler import (compile_scene, normalize_scene, render_scene,
                       render_scene_animation)
from .parser import DSLError, parse
from .validator import ValidationResult, validate_mjcf, validate_pre_compile


def validate(dsl_source: str) -> ValidationResult:
    """Run all three gates on a DSL source string without rendering."""
    try:
        scene = parse(dsl_source)
    except DSLError as e:
        return ValidationResult(ok=False, gate=1, reason=str(e))

    normalize_scene(scene)
    pre = validate_pre_compile(scene)
    if not pre.ok:
        return pre

    try:
        compiled = compile_scene(scene)
    except Exception as e:
        return ValidationResult(ok=False, gate=2, reason=f"compile error: {e}")

    return validate_mjcf(compiled.mjcf)


def _scenario_backend_setup(
    sb: dict[str, Any], control_value: float, alpha: float | None,
    seed: int, clean: bool,
):
    """Common setup for scenario-backend rendering: returns
    (sc, sim, eff_clean) so that callers can pull either the first frame
    (PNG) or the full animation (GIF) without duplicating boilerplate."""
    from galileo.scenarios import get_scenario

    slug = sb["slug"]
    sc = get_scenario(slug)
    sc.rng = np.random.default_rng(seed)
    eff_alpha = sb.get("alpha")
    if eff_alpha is None:
        eff_alpha = alpha
    if eff_alpha is not None:
        sc.alpha = float(eff_alpha)
    for k, v in (sb.get("kwargs") or {}).items():
        if hasattr(sc, k):
            setattr(sc, k, v)
        elif hasattr(sc, "modify_apparatus"):
            try:
                sc.modify_apparatus(**{k: v})
            except Exception:
                pass
    sim = sc.simulate(float(control_value))
    eff_clean = bool(sb.get("clean", False)) or bool(clean)
    return sc, sim, eff_clean


def _apply_sb_overlays(img: Image.Image, sb: dict[str, Any]) -> Image.Image:
    """Apply pixel-space overlays declared alongside the ScenarioBackend
    entity (currently only BackgroundGrid)."""
    from .compiler import draw_background_grid_pil
    out = img
    for ent in sb.get("overlay_entities", []):
        if ent.kind == "BackgroundGrid":
            out = draw_background_grid_pil(
                out,
                cell_px=int(ent.params.get("cell_px", 64)),
                axis_labels=bool(ent.params.get("axis_labels", False)),
            )
    return out


def _render_via_scenario_backend(
    sb: dict[str, Any], control_value: float, alpha: float | None,
    width: int, height: int, seed: int = 0, clean: bool = False,
) -> Image.Image:
    """Build a Scenario instance from a `ScenarioBackend` metadata dict
    and render its first frame at `control_value`.

    `clean` (caller override) forces matplotlib axis/label suppression
    even if the YAML's `clean` param is False. Used by demo builders
    and the GCD runner to honour "no on-canvas text" design rules.
    """
    sc, sim, eff_clean = _scenario_backend_setup(
        sb, control_value, alpha, seed, clean)
    frames = sc.render_frames(sim, clean=eff_clean)
    if not frames:
        return Image.new("RGB", (width, height), (60, 60, 60))
    img = frames[0].convert("RGB").copy()
    return _apply_sb_overlays(img, sb)


def _render_via_scenario_backend_animation(
    sb: dict[str, Any], control_value: float, alpha: float | None,
    width: int, height: int, seed: int = 0, clean: bool = False,
) -> tuple[list[Image.Image], int]:
    """Multi-frame variant: returns (frames, fps). Pulls every frame
    `sc.render_frames` produces (matplotlib animation), applies the
    pixel-space overlays per frame, and pairs with a default fps so
    the caller can assemble a GIF directly."""
    sc, sim, eff_clean = _scenario_backend_setup(
        sb, control_value, alpha, seed, clean)
    frames = sc.render_frames(sim, clean=eff_clean)
    if not frames:
        return [Image.new("RGB", (width, height), (60, 60, 60))], 8
    out: list[Image.Image] = []
    for fr in frames:
        img = fr.convert("RGB").copy()
        out.append(_apply_sb_overlays(img, sb))
    fps = int(sb.get("fps", 8))
    return out, max(1, fps)


def compile_and_render(
    dsl_source: str,
    control_value: float,
    alpha: float = 2.5,
    width: int = 640,
    height: int = 480,
    target: str | None = None,
    seed: int = 0,
    clean: bool = False,
) -> Image.Image:
    """Full pipeline: parse -> validate -> render. Renders via MuJoCo
    strobe by default; delegates to a registered Scenario.* class when
    the scene contains a `ScenarioBackend` entity.

    `clean=True` strips ALL on-canvas text:
      - mujoco_strobe path: suppresses 1..N numeric strobe labels and
        the bottom filmstrip;
      - scenario_backend path: forces matplotlib axes/ticks/labels off.
    Use clean=True for GCD images and visualisation demos; leave the
    default False for the discovery runner where the agent actively
    needs numeric measurement aids.
    """
    scene = parse(dsl_source)
    normalize_scene(scene)
    pre = validate_pre_compile(scene)
    if not pre.ok:
        raise ValueError(f"[gate {pre.gate}] {pre.reason}")

    compiled = compile_scene(scene, width=width, height=height)
    sb = compiled.metadata.get("scenario_backend")
    if sb:
        mjcf_check = validate_mjcf(compiled.mjcf)
        if not mjcf_check.ok:
            raise ValueError(f"[gate {mjcf_check.gate}] {mjcf_check.reason}")
        return _render_via_scenario_backend(
            sb, control_value=control_value, alpha=alpha,
            width=width, height=height, seed=seed, clean=clean,
        )

    return render_scene(
        scene,
        control_value=control_value,
        alpha=alpha,
        width=width,
        height=height,
        target=target,
        clean=clean,
    )


def compile_and_render_animation(
    dsl_source: str,
    control_value: float,
    alpha: float = 2.5,
    width: int = 640,
    height: int = 480,
    target: str | None = None,
) -> tuple[list[Image.Image], int]:
    """Full pipeline that returns animation frames + fps for scenes
    containing an `Animation` entity. Returns (frames, fps); the caller
    is responsible for serialising as a GIF (or any other format).

    For scenes with a `ScenarioBackend` motion source this still works
    but only delegates to the scenario class's matplotlib animation;
    callers wanting a unified entry point should use
    `compile_and_render_frames` instead.
    """
    scene = parse(dsl_source)
    normalize_scene(scene)
    pre = validate_pre_compile(scene)
    if not pre.ok:
        raise ValueError(f"[gate {pre.gate}] {pre.reason}")
    return render_scene_animation(
        scene,
        control_value=control_value,
        alpha=alpha,
        width=width,
        height=height,
        target=target,
    )


def compile_and_render_frames(
    dsl_source: str,
    control_value: float,
    alpha: float = 2.5,
    width: int = 640,
    height: int = 480,
    target: str | None = None,
    seed: int = 0,
    clean: bool = False,
) -> tuple[list[Image.Image], int]:
    """Unified GIF-friendly entry. Returns `(frames, fps)` for ANY
    of the 15 benchmark scenes, regardless of motion source:

      * mechanical (MuJoCo) scenes -> render_scene_animation
      * scenario_backend scenes    -> matplotlib animation
        from `Scenario.render_frames`

    `clean=True` strips on-canvas text in both paths (numeric strobe
    labels suppressed in MuJoCo path; matplotlib axes/ticks/labels
    forced off in SB path). Use this for the demo builders so each
    cell looks like the original demo.html / instruments.html GIFs.
    """
    scene = parse(dsl_source)
    normalize_scene(scene)
    pre = validate_pre_compile(scene)
    if not pre.ok:
        raise ValueError(f"[gate {pre.gate}] {pre.reason}")

    compiled = compile_scene(scene, width=width, height=height)
    sb = compiled.metadata.get("scenario_backend")
    if sb:
        mjcf_check = validate_mjcf(compiled.mjcf)
        if not mjcf_check.ok:
            raise ValueError(f"[gate {mjcf_check.gate}] {mjcf_check.reason}")
        return _render_via_scenario_backend_animation(
            sb, control_value=control_value, alpha=alpha,
            width=width, height=height, seed=seed, clean=clean,
        )

    return render_scene_animation(
        scene,
        control_value=control_value,
        alpha=alpha,
        width=width,
        height=height,
        target=target,
    )


def scene_has_animation(dsl_source: str) -> bool:
    """True iff the parsed scene contains at least one Animation entity.
    Cheap pre-check that lets a caller decide between PNG and GIF."""
    try:
        scene = parse(dsl_source)
    except DSLError:
        return False
    return any(e.kind == "Animation" for e in scene.entities.values())


def list_targets(dsl_source: str) -> list[str]:
    """Return the names of tracked bodies available in the scene,
    in the order the compiler registered them. Useful for the runner
    when exposing `target` to the VLM."""
    from .compiler import compile_scene
    scene = parse(dsl_source)
    normalize_scene(scene)
    pre = validate_pre_compile(scene)
    if not pre.ok:
        raise ValueError(f"[gate {pre.gate}] {pre.reason}")
    compiled = compile_scene(scene)
    return [tb.name for tb in compiled.tracked_bodies]
