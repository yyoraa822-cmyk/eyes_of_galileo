"""Goal-Conditioned Discovery (GCD) runner.

Phase 2 of the Galileo benchmark: instead of asking the agent to
discover the law, we hand it a *goal* — a held-out target observable
value `y*` rendered as a green marker on top of the scenario — and
ask it to drive the scenario to hit `y*` within tolerance, by
choosing one or more values of the scenario's control variable.

The runner reuses every piece of `run_dsl_unified`'s infrastructure
(YAML builders, scenario factory, DSL parse/render pipeline, physics
core lock) but replaces:

  - the system prompt (no "discover law"; instead "hit the green target")
  - the toolset (`attempt_goal(u)` instead of `submit_law(expr)`)
  - the success metric (`|y_hit - y*| <= tol` instead of joint
    log-residual against the closed-form law)

Every per-attempt feedback image goes through the full primitive
chain:

    img = compile_and_render(yaml, cv=u_attempt, ...)        # DSL
    img = draw_goal_marker(img, kind, target_xy_px)          # green
    img = overlay_position(img, target_xy, actual_xy, passed=ok)

so the agent sees *visually* how far it missed by, with NO numeric
residual rendered into the image (PASS/FAIL is the only text).

CLI parity with run_dsl_unified.py:
    python -m galileo.run_gcd --scene s1_freefall --model gpt-5.5-medium
    python -m galileo.run_gcd --scene all       --model gpt-5.5-medium

If `--api-key` is not provided / not set, the runner uses a built-in
oracle agent (the solver) for dry-run verification of the full image
pipeline — useful for validating goal/marker/overlay composition
without hitting an external API.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from galileo.providers import VLMProvider, Message, image_to_data_url, gif_to_data_url
from galileo.scenarios import Scenario
from galileo.goals import (
    GOAL_POOL,
    Goal,
    VIEW_GOAL_META,
    draw_goal_marker,
    goal_kind_for,
    materialize_goal_pool,
    overlay_phase,
    overlay_position,
    overlay_scalar,
    project_goal,
    solve_for_control,
)
from galileo.run_dsl_unified import (
    SCENE_VIEWS,
    _build_scenario,
    render_image as _render_image_dsl,
    render_animation_unified,
    DSL_INSTRUMENT_CHEATSHEET,
    check_physics_core,
    _format_motion_source_doc,
)
from galileo.visual_mujoco import render_animation as _vm_render, supports as _vm_supports
from galileo.dsl.api import compile_and_render, validate as dsl_validate
from galileo.dsl.parser import DSLError


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are an experimentalist playing Goal-Conditioned Discovery.

A scenario "{description}" is rendered through a DSL apparatus. Somewhere
in the rendered image is a GREEN goal marker -- this marks the
observable value `y*` (in `{observable_label}`) you must achieve. You
do NOT see `y*` as a number; you see only its visual placement.

You drive the scenario by setting one number: the control variable
`{control_var}`. The advertised pool of allowed values is:

    {control_var} in [{ctrl_lo:g}, {ctrl_hi:g}]

Tools

  attempt_goal(control_value, rationale)
      Render the scenario at this `{control_var}`. Returns a
      composite image that shows
        * the GREEN goal marker (target),
        * a RED X at the actual outcome,
        * a connecting arrow if you missed,
        * a top-right PASS/FAIL badge.
      No numeric residual is rendered. attempts_remaining decreases by 1.

  propose_apparatus(dsl_yaml, reason)
      Submit a revised DSL YAML apparatus. This lets you attach or
      detach instrument entities (e.g. BackgroundGrid, ReferenceScale,
      AngleProtractor) to gain better visual readouts, without changing
      the physical motion source. On success, the new apparatus is used
      for all subsequent attempt_goal renders. On failure you'll see
      the gate error — revise and try again.
      Proposals do NOT consume attempts_remaining.

  give_up(reason)
      End the episode immediately, recorded as FAIL. Use only if
      attempts_remaining == 0 or you are confident no further attempt
      will help.

  submit_physical_law(law_expression, rationale, confidence)
      Record your current best physical law hypothesis linking
      `{control_var}` to `{observable_label}`. Does NOT consume attempts.

Strategy tips

  * Use the visual residual (arrow length, bar gap, angular wedge) to
    BRACKET and BISECT. If attempt A overshot and attempt B undershot,
    your next guess should be between A and B.
  * Count ticks / grid cells to estimate the gap quantitatively.
  * If you find the scene hard to read, call propose_apparatus to add
    instruments (e.g. BackgroundGrid for a spatial ruler,
    AngleProtractor for an angle reference).

Hard rules

  * The ONLY way to PASS is to call `attempt_goal` with a value that
    drives the observable within tolerance of `y*`.
  * Stay inside the advertised pool; out-of-range values are clamped.
  * One attempt per turn. Use the visual residual (arrow length,
    bar gap, dial wedge) on each FAIL feedback image to refine your
    next guess.
  * The first PASS ends the episode. There is no "double-check" and
    no retraction.

{dsl_cheatsheet}

Budget: attempts_remaining={{max_turns}}, tolerance is intentionally
strict; rely on the visual residual rather than guessing.
"""

SYSTEM_PROMPT_TEMPLATE_NO_INST = """You are an experimentalist playing Goal-Conditioned Discovery.

A scenario "{description}" is simulated. Somewhere in the scene is an
observable value `y*` (in `{observable_label}`) that you must achieve.
You do NOT see `y*` as a number.

You drive the scenario by setting one number: the control variable
`{control_var}`. The advertised pool of allowed values is:

    {control_var} in [{ctrl_lo:g}, {ctrl_hi:g}]

What you observe:
  * For each attempt you will receive an ANIMATED GIF showing the raw
    physics motion at your chosen control value, followed by a static
    comparison image showing the GREEN goal marker (target) and a RED X
    at the actual outcome with a PASS/FAIL badge.
  * No measurement instruments, grids, or scales are provided. You must
    judge distances and directions purely from the raw animation.

Tools

  attempt_goal(control_value, rationale)
      Render the scenario at this `{control_var}`. Returns an animated
      GIF of the motion plus a static comparison image with PASS/FAIL
      badge. attempts_remaining decreases by 1.

  give_up(reason)
      End the episode immediately, recorded as FAIL. Use only if
      attempts_remaining == 0 or you are confident no further attempt
      will help.

  submit_physical_law(law_expression, rationale, confidence)
      Record your current best physical law hypothesis linking
      `{control_var}` to `{observable_label}`. Does NOT consume attempts.

Strategy tips

  * Use the visual residual (arrow length, position gap) to BRACKET
    and BISECT. If attempt A overshot and attempt B undershot, your
    next guess should be between A and B.
  * Watch the animation carefully to judge motion magnitude.

Hard rules

  * The ONLY way to PASS is to call `attempt_goal` with a value that
    drives the observable within tolerance of `y*`.
  * Stay inside the advertised pool; out-of-range values are clamped.
  * One attempt per turn. Use the visual feedback on each FAIL to
    refine your next guess.
  * The first PASS ends the episode. There is no "double-check" and
    no retraction.

Budget: attempts_remaining={{max_turns}}, tolerance is intentionally
strict; rely on the visual feedback rather than guessing.
"""


def build_gcd_tools(control_var: str, *,
                    include_apparatus: bool = True) -> list[dict]:
    tools: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "attempt_goal",
                "description": (
                    f"Render the scenario at one numeric value of "
                    f"`{control_var}` and return the composite "
                    f"goal+residual image with PASS/FAIL badge."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "control_value": {
                            "type": "number",
                            "description": (
                                f"Numeric value of `{control_var}`. "
                                f"Will be clamped to the advertised "
                                f"pool if out of range."
                            ),
                        },
                        "rationale": {
                            "type": "string",
                            "description": (
                                "One short sentence justifying the choice "
                                "(e.g. \"arrow on previous fail pointed "
                                "down by ~30%, halving cv\")."
                            ),
                        },
                    },
                    "required": ["control_value"],
                },
            },
        },
    ]
    if include_apparatus:
        tools.append({
            "type": "function",
            "function": {
                "name": "propose_apparatus",
                "description": (
                    "Submit a revised DSL YAML apparatus. Attach or "
                    "detach instrument entities (BackgroundGrid, "
                    "ReferenceScale, AngleProtractor, etc.) to gain "
                    "better visual readouts. Does NOT consume "
                    "attempts_remaining. On success returns a preview "
                    "image; on failure returns the gate error."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dsl_yaml": {
                            "type": "string",
                            "description": (
                                "Full DSL YAML source string. Must "
                                "preserve the locked physics core."
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
    tools.append({
        "type": "function",
        "function": {
            "name": "submit_physical_law",
            "description": (
                "Record your best physical law hypothesis for this scene. "
                "Does not consume attempts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "law_expression": {
                        "type": "string",
                        "description": (
                            f"Symbolic law linking `{control_var}` to the "
                            "observable."
                        ),
                    },
                    "rationale": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["law_expression"],
            },
        },
    })
    tools.append({
        "type": "function",
        "function": {
            "name": "give_up",
            "description": (
                "Abort the episode. Recorded as FAIL. "
                "Use sparingly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                },
                "required": ["reason"],
            },
        },
    })
    return tools


# ---------------------------------------------------------------------------
# Render dispatch: prefer visual_mujoco for all supported scenes
# ---------------------------------------------------------------------------


def render_image(view: dict[str, Any], scenario: Scenario,
                 control_value: float, *, seed: int = 0,
                 clean: bool = False,
                 yaml_override: Optional[str] = None) -> Image.Image:
    """Render via MuJoCo (visual_mujoco) when available, else fall back
    to the DSL pipeline. If yaml_override is provided (agent proposed a
    custom apparatus), always render through compile_and_render."""
    if yaml_override is not None:
        alpha = float(getattr(scenario, "alpha", 1.0))
        return compile_and_render(
            yaml_override, control_value=float(control_value),
            alpha=alpha, width=1280, height=960,
            seed=seed, clean=clean,
        )
    scene_id = view["id"]
    if _vm_supports(scene_id):
        frames, _fps = _vm_render(
            scene_id, scenario, control_value,
            n_frames=2, width=1280, height=960,
        )
        return frames[-1].convert("RGB")
    return _render_image_dsl(view, scenario, control_value,
                             seed=seed, clean=clean)


def render_gif_bytes(view: dict[str, Any], scenario: Scenario,
                     control_value: float, *, seed: int = 0,
                     n_frames: int = 24) -> bytes:
    """Render an animated GIF of the scene at `control_value`.
    Used in no-instrument mode so the agent sees raw motion."""
    import io as _io
    frames, fps = render_animation_unified(
        view, scenario, control_value,
        n_frames=n_frames, width=960, height=600, seed=seed,
    )
    buf = _io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True,
                   append_images=frames[1:],
                   duration=int(1000 / max(fps, 1)),
                   loop=0, optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Composite renderer: DSL -> goal marker -> residual overlay
# ---------------------------------------------------------------------------


def render_with_marker(view: dict[str, Any], scenario: Scenario,
                       goal: Goal, control_value: float,
                       *, seed: int = 0,
                       yaml_override: Optional[str] = None
                       ) -> tuple[Image.Image, tuple[int, int]]:
    """Render the scene at `control_value` and paint the green goal
    marker on top. Returns (image, target_xy_px)."""
    img = render_image(view, scenario, control_value, seed=seed,
                       clean=True, yaml_override=yaml_override)
    proj = project_goal(view["id"], goal, img.size)
    img = draw_goal_marker(img, goal_kind_for(view["id"]),
                           proj.target_xy_px)
    return img, proj.target_xy_px


def render_attempt_feedback(view: dict[str, Any], scenario: Scenario,
                            goal: Goal, attempt_u: float,
                            *, seed: int = 0,
                            yaml_override: Optional[str] = None
                            ) -> tuple[Image.Image, dict[str, Any]]:
    """Run one agent attempt and produce the feedback composite.

    Returns (image, judge) where `judge` is:
        {
          "control_value": float,
          "actual_y": float,
          "target_y": float,
          "tolerance": float,
          "abs_error": float,
          "passed": bool,
        }
    """
    actual_y = float(scenario.get_observable(float(attempt_u)))
    abs_err = abs(actual_y - goal.target_y)
    passed = abs_err <= goal.tolerance

    img = render_image(view, scenario, attempt_u, seed=seed, clean=True,
                       yaml_override=yaml_override)
    proj = project_goal(view["id"], goal, img.size, actual_y=actual_y)
    img = draw_goal_marker(img, goal_kind_for(view["id"]),
                           proj.target_xy_px)
    if proj.overlay_type == "position":
        img = overlay_position(img,
                               target_xy=proj.target_xy_px,
                               actual_xy=proj.actual_xy_px or proj.target_xy_px,
                               passed=passed)
    elif proj.overlay_type == "scalar":
        bounds = proj.scalar_bounds or (
            min(goal.target_y, actual_y) - 1.0,
            max(goal.target_y, actual_y) + 1.0,
        )
        img = overlay_scalar(img,
                             target_v=goal.target_y,
                             actual_v=actual_y,
                             vmin=bounds[0], vmax=bounds[1],
                             passed=passed)
    else:
        img = overlay_phase(img,
                            target_deg=proj.target_deg or 0.0,
                            actual_deg=proj.actual_deg or 0.0,
                            passed=passed)

    judge = {
        "control_value": float(attempt_u),
        "actual_y": actual_y,
        "target_y": float(goal.target_y),
        "tolerance": float(goal.tolerance),
        "abs_error": abs_err,
        "passed": bool(passed),
    }
    return img, judge


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class GoalAttempt:
    turn: int
    control_value: float
    actual_y: float
    abs_error: float
    passed: bool
    rationale: str = ""


@dataclass
class GoalRecord:
    scene_id: str
    slug: str
    goal_idx: int
    goal: dict[str, Any]
    model: str
    seed: int
    attempts: list[GoalAttempt] = field(default_factory=list)
    passed: bool = False
    pass_turn: Optional[int] = None
    elapsed_s: float = 0.0
    error: Optional[str] = None
    gave_up: bool = False
    give_up_reason: Optional[str] = None
    submitted_law: Optional[str] = None
    submitted_law_rationale: Optional[str] = None
    submitted_law_confidence: Optional[float] = None
    submitted_law_turn: Optional[int] = None
    inferred_law: Optional[str] = None
    inferred_law_rationale: Optional[str] = None
    inferred_law_confidence: Optional[float] = None
    ground_truth_law: Optional[str] = None


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    txt = (text or "").strip()
    if not txt:
        return None
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _ground_truth_law_for(scenario: Scenario) -> str:
    slug = scenario.meta.slug
    alpha = float(getattr(scenario, "alpha", 0.0))
    if slug == "freefall":
        g_eff = float(getattr(scenario, "g_eff", 9.81))
        return f"y = 0.5*{g_eff:.6g}*u^{alpha:.6g}"
    if slug == "pendulum":
        g = float(getattr(scenario, "_g", 9.81))
        return f"T = 2*pi*(u/{g:.6g})^{alpha:.6g}"
    if slug == "spring":
        g = float(getattr(scenario, "_g", 9.81))
        k = float(getattr(scenario, "_k", 20.0))
        return f"x = ((u*{g:.6g})/{k:.6g})^{alpha:.6g}"
    if slug == "refraction":
        n1 = float(getattr(scenario, "_n1", 1.0))
        n2 = float(getattr(scenario, "_n2", 1.5))
        return (f"n1*sin(theta1)=n2*sin(theta2)^{alpha:.6g}; "
                f"n1={n1:.6g}, n2={n2:.6g}")
    return f"{scenario.meta.law_template}; alpha={alpha:.6g}"


def _posthoc_infer_law(provider: Optional[VLMProvider], *,
                       messages: list[Message],
                       record: GoalRecord) -> None:
    if provider is None:
        return
    prompt = (
        "Summarize your best physical law hypothesis for this episode. "
        "Return JSON only with keys: law_expression, rationale, confidence."
    )
    try:
        ans = provider.chat(messages + [Message(role="user", content=prompt)])
    except Exception:
        return
    obj = _extract_json_object(str(ans.content))
    if not obj:
        return
    law = str(obj.get("law_expression", "")).strip()
    if not law:
        return
    record.inferred_law = law
    rat = str(obj.get("rationale", "")).strip()
    record.inferred_law_rationale = rat or None
    try:
        conf = obj.get("confidence", None)
        record.inferred_law_confidence = (
            float(conf) if conf is not None else None
        )
    except Exception:
        record.inferred_law_confidence = None


# ---------------------------------------------------------------------------
# Per-goal episode loop
# ---------------------------------------------------------------------------


def run_one_goal(view: dict[str, Any], goal: Goal, goal_idx: int,
                 provider: Optional[VLMProvider], *,
                 seed: int, max_turns: int, out_dir: Path,
                 oracle_dry_run: bool = False,
                 include_apparatus: bool = True) -> GoalRecord:
    scenario = _build_scenario(view, seed=seed)
    pool_lo, pool_hi = goal.control_range

    record = GoalRecord(
        scene_id=view["id"], slug=view["slug"],
        goal_idx=goal_idx,
        goal=asdict(goal) if hasattr(goal, "__dataclass_fields__")
                          else goal.__dict__,
        model=(provider.model if provider is not None else "oracle"),
        seed=seed,
    )
    record.ground_truth_law = _ground_truth_law_for(scenario)

    images_dir = out_dir / f"{view['id']}_goal{goal_idx}_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    init_cv = 0.5 * (pool_lo + pool_hi)
    try:
        init_img, target_xy = render_with_marker(view, scenario, goal,
                                                 init_cv, seed=seed)
    except Exception as e:
        record.error = f"render_error: {e}"
        record.elapsed_s = 0.0
        out_json = out_dir / f"{view['id']}_goal{goal_idx}.json"
        out_json.write_text(json.dumps(asdict(record), indent=2))
        return record
    init_img.save(images_dir / "00_initial.png")

    # ---- Oracle dry-run path: bypass external API ----------------------
    if oracle_dry_run or provider is None:
        u_star = solve_for_control(scenario, goal.target_y,
                                   pool_lo, pool_hi)
        if u_star is None:
            # try non-monotone fallback
            u_star = solve_for_control(scenario, goal.target_y,
                                       pool_lo, pool_hi, monotone=False)
        if u_star is None:
            record.error = "oracle_unsolvable"
        else:
            t0 = time.time()
            img, judge = render_attempt_feedback(
                view, scenario, goal, u_star, seed=seed)
            img.save(images_dir / "01_oracle_attempt.png")
            record.attempts.append(GoalAttempt(
                turn=1, control_value=u_star,
                actual_y=judge["actual_y"],
                abs_error=judge["abs_error"],
                passed=judge["passed"],
                rationale="oracle bisection",
            ))
            record.passed = judge["passed"]
            record.pass_turn = 1 if judge["passed"] else None
            record.elapsed_s = round(time.time() - t0, 3)

        out_json = out_dir / f"{view['id']}_goal{goal_idx}.json"
        out_json.write_text(json.dumps(asdict(record), indent=2))
        return record

    # ---- Real agent path ----------------------------------------------
    dsl_cheatsheet = DSL_INSTRUMENT_CHEATSHEET.format(
        motion_source=_format_motion_source_doc(view),
        render_path=view.get("render_path", "scenario_backend"),
    ) if include_apparatus else ""
    if include_apparatus:
        sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            description=scenario.meta.description.strip(),
            observable_label=scenario.meta.observable_label,
            control_var=scenario.meta.control_var,
            ctrl_lo=pool_lo, ctrl_hi=pool_hi,
            max_turns=max_turns,
            dsl_cheatsheet=dsl_cheatsheet,
        )
    else:
        sys_prompt = SYSTEM_PROMPT_TEMPLATE_NO_INST.format(
            description=scenario.meta.description.strip(),
            observable_label=scenario.meta.observable_label,
            control_var=scenario.meta.control_var,
            ctrl_lo=pool_lo, ctrl_hi=pool_hi,
            max_turns=max_turns,
        )
    tools = build_gcd_tools(scenario.meta.control_var,
                            include_apparatus=include_apparatus)
    messages: list[Message] = [Message(role="system", content=sys_prompt)]

    if include_apparatus:
        messages.append(Message(role="user", content=[
            {"type": "text",
             "text": (f"Goal {goal_idx + 1} for {view['id']} "
                      f"({goal.name}). Difficulty: {goal.difficulty}. "
                      f"You see the GREEN target; drive "
                      f"`{scenario.meta.control_var}` to make the actual "
                      f"outcome match it. attempts_remaining={max_turns}.")},
            {"type": "image_url",
             "image_url": {"url": image_to_data_url(init_img),
                           "detail": "high"}},
        ]))
    else:
        init_gif = render_gif_bytes(view, scenario, init_cv, seed=seed)
        (images_dir / "00_initial.gif").write_bytes(init_gif)
        messages.append(Message(role="user", content=[
            {"type": "text",
             "text": (f"Goal {goal_idx + 1} for {view['id']} "
                      f"({goal.name}). Difficulty: {goal.difficulty}. "
                      f"Below: (1) animated GIF of the scene at the "
                      f"midpoint control value, (2) a static image "
                      f"showing the GREEN goal target you must match. "
                      f"Drive `{scenario.meta.control_var}` to make the "
                      f"actual outcome match the green target. "
                      f"attempts_remaining={max_turns}.")},
            {"type": "image_url",
             "image_url": {"url": gif_to_data_url(init_gif),
                           "detail": "high"}},
            {"type": "image_url",
             "image_url": {"url": image_to_data_url(init_img),
                           "detail": "high"}},
        ]))

    current_yaml: Optional[str] = None
    proposals_left = 3

    t0 = time.time()
    for turn in range(max_turns):
        try:
            assistant = provider.chat(messages, tools=tools)
        except Exception as e:
            record.error = f"provider_error: {e}"
            break
        messages.append(assistant)

        if not assistant.tool_calls:
            if turn >= max_turns - 1:
                record.error = "no_tool_call_in_budget"
                break
            available = ("attempt_goal, submit_physical_law, "
                         "propose_apparatus, or give_up"
                         if include_apparatus
                         else "attempt_goal, submit_physical_law, or give_up")
            messages.append(Message(role="user", content=(
                f"You must call {available}. "
                f"attempts_remaining={max_turns - turn - 1}.")))
            continue

        for tc in assistant.tool_calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except Exception:
                args = {}
            tool_id = tc.get("id", "tc")

            if fn == "give_up":
                reason = str(args.get("reason", ""))
                record.gave_up = True
                record.give_up_reason = reason
                messages.append(Message(
                    role="tool", tool_call_id=tool_id, name=fn,
                    content="acknowledged; episode FAIL."))
                break

            if fn == "propose_apparatus":
                proposed_yaml = str(args.get("dsl_yaml", "")).strip()
                reason = str(args.get("reason", ""))
                if proposals_left <= 0:
                    messages.append(Message(
                        role="tool", tool_call_id=tool_id, name=fn,
                        content="REJECTED: no apparatus proposals left."))
                    continue
                proposals_left -= 1
                # Validate through Gate 1/2/3 + physics core
                ok, err_msg = check_physics_core(view, proposed_yaml)
                if not ok:
                    messages.append(Message(
                        role="tool", tool_call_id=tool_id, name=fn,
                        content=f"REJECTED: {err_msg}"))
                    continue
                vr = dsl_validate(proposed_yaml)
                if not vr.ok:
                    messages.append(Message(
                        role="tool", tool_call_id=tool_id, name=fn,
                        content=(f"REJECTED [gate {vr.gate}]: "
                                 f"{vr.reason}")))
                    continue
                # Accepted — render a preview
                current_yaml = proposed_yaml
                try:
                    preview = render_with_marker(
                        view, scenario, goal, init_cv, seed=seed,
                        yaml_override=current_yaml)[0]
                    preview.save(images_dir /
                                 f"apparatus_t{turn:02d}.png")
                    messages.append(Message(
                        role="tool", tool_call_id=tool_id, name=fn,
                        content=[
                            {"type": "text",
                             "text": (f"ACCEPTED. Apparatus updated "
                                      f"({reason}). Preview below. "
                                      f"proposals_remaining="
                                      f"{proposals_left}.")},
                            {"type": "image_url",
                             "image_url": {
                                 "url": image_to_data_url(preview),
                                 "detail": "high"}},
                        ]))
                except Exception as e:
                    current_yaml = None
                    messages.append(Message(
                        role="tool", tool_call_id=tool_id, name=fn,
                        content=f"REJECTED: render_error: {e}"))
                continue

            if fn == "submit_physical_law":
                law = str(args.get("law_expression", "")).strip()
                rationale = str(args.get("rationale", "")).strip()
                conf = args.get("confidence", None)
                if not law:
                    messages.append(Message(
                        role="tool", tool_call_id=tool_id, name=fn,
                        content="ERROR: law_expression is required."))
                    continue
                record.submitted_law = law
                record.submitted_law_rationale = rationale or None
                try:
                    record.submitted_law_confidence = (
                        float(conf) if conf is not None else None
                    )
                except Exception:
                    record.submitted_law_confidence = None
                record.submitted_law_turn = turn + 1
                messages.append(Message(
                    role="tool", tool_call_id=tool_id, name=fn,
                    content="acknowledged: law hypothesis recorded."))
                continue

            if fn != "attempt_goal":
                messages.append(Message(
                    role="tool", tool_call_id=tool_id, name=fn,
                    content=f"unknown tool: {fn}"))
                continue

            try:
                u = float(args.get("control_value"))
            except Exception:
                messages.append(Message(
                    role="tool", tool_call_id=tool_id, name=fn,
                    content="ERROR: control_value not a number."))
                continue
            rationale = str(args.get("rationale", ""))
            u_clamped = float(min(max(u, pool_lo), pool_hi))

            try:
                img, judge = render_attempt_feedback(
                    view, scenario, goal, u_clamped, seed=seed,
                    yaml_override=current_yaml)
            except Exception as e:
                messages.append(Message(
                    role="tool", tool_call_id=tool_id, name=fn,
                    content=f"render_error: {e}"))
                continue

            img.save(images_dir / f"t{turn:02d}_u{u_clamped:g}.png")
            record.attempts.append(GoalAttempt(
                turn=turn + 1, control_value=u_clamped,
                actual_y=judge["actual_y"],
                abs_error=judge["abs_error"],
                passed=judge["passed"],
                rationale=rationale,
            ))
            attempts_remaining = max_turns - (turn + 1)

            if include_apparatus:
                messages.append(Message(
                    role="tool", tool_call_id=tool_id, name=fn,
                    content=[
                        {"type": "text",
                         "text": (
                            f"{'PASS' if judge['passed'] else 'FAIL'}. "
                            f"attempts_remaining={attempts_remaining}.")},
                        {"type": "image_url",
                         "image_url": {"url": image_to_data_url(img),
                                       "detail": "high"}},
                    ]))
            else:
                try:
                    attempt_gif = render_gif_bytes(
                        view, scenario, u_clamped, seed=seed)
                    (images_dir /
                     f"t{turn:02d}_u{u_clamped:g}.gif").write_bytes(
                        attempt_gif)
                except Exception:
                    attempt_gif = None
                content_parts: list[dict] = [
                    {"type": "text",
                     "text": (
                        f"{'PASS' if judge['passed'] else 'FAIL'}. "
                        f"attempts_remaining={attempts_remaining}.")},
                ]
                if attempt_gif:
                    content_parts.append(
                        {"type": "image_url",
                         "image_url": {"url": gif_to_data_url(attempt_gif),
                                       "detail": "high"}})
                content_parts.append(
                    {"type": "image_url",
                     "image_url": {"url": image_to_data_url(img),
                                   "detail": "high"}})
                messages.append(Message(
                    role="tool", tool_call_id=tool_id, name=fn,
                    content=content_parts))
            if judge["passed"]:
                record.passed = True
                record.pass_turn = turn + 1
                break

        if record.passed or record.gave_up:
            break

    record.elapsed_s = round(time.time() - t0, 2)
    if not record.submitted_law:
        _posthoc_infer_law(provider, messages=messages, record=record)

    out_json = out_dir / f"{view['id']}_goal{goal_idx}.json"
    out_json.write_text(json.dumps(asdict(record), indent=2))
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="all",
                    help="scene id or 'all'")
    ap.add_argument("--goal-idx", type=int, default=None,
                    help="single goal idx within the pool; if omitted, "
                         "iterates through all 5 goals per scene")
    ap.add_argument("--difficulty", default=None,
                    help="filter goals by difficulty (easy|medium|hard)")
    ap.add_argument("--model", default="gpt-5.5-medium")
    ap.add_argument("--api-base", default=os.environ.get(
        "GALILEO_API_BASE", "https://api.example.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get(
        "GALILEO_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--out", default="galileo/outputs/gcd")
    ap.add_argument("--oracle-dry-run", action="store_true",
                    help="bypass the API and have the oracle solver "
                         "play one perfect attempt per goal. Useful "
                         "for verifying the full image pipeline "
                         "(DSL -> marker -> overlay) end-to-end.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-instruments", action="store_true",
                    help="disable propose_apparatus tool (baseline mode)")
    args = ap.parse_args()

    materialize_goal_pool(force=False)

    if args.list:
        for v in SCENE_VIEWS:
            goals = GOAL_POOL.get(v["id"], [])
            print(f"{v['id']:16s} {v['render_path']:18s} "
                  f"{len(goals)} goals  "
                  f"kind={goal_kind_for(v['id'])}")
        return 0

    use_oracle = args.oracle_dry_run or not args.api_key
    if use_oracle:
        provider = None
        print("[INFO] running in oracle dry-run mode "
              "(no external API calls).", file=sys.stderr)
    else:
        provider = VLMProvider(
            api_base=args.api_base, api_key=args.api_key,
            model=args.model, seed=args.seed,
            temperature=args.temperature, max_tokens=2048,
        )

    out_dir = Path(args.out) / (args.model.replace("/", "_")
                                if not use_oracle else "oracle")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scene == "all":
        views = SCENE_VIEWS
    else:
        views = [v for v in SCENE_VIEWS if v["id"] == args.scene]
        if not views:
            print(f"unknown scene: {args.scene}", file=sys.stderr)
            return 2

    summary: list[dict[str, Any]] = []
    for v in views:
        goals = GOAL_POOL.get(v["id"], [])
        if not goals:
            print(f"{v['id']}: no goals materialised, skipping",
                  file=sys.stderr)
            continue
        targets: list[tuple[int, Goal]] = []
        if args.goal_idx is not None:
            if 0 <= args.goal_idx < len(goals):
                targets = [(args.goal_idx, goals[args.goal_idx])]
        else:
            for i, g in enumerate(goals):
                if args.difficulty and g.difficulty != args.difficulty:
                    continue
                targets.append((i, g))
        if not targets:
            continue

        for gi, goal in targets:
            print(f"\n=== {v['id']} goal {gi} "
                  f"({goal.difficulty}, target_y={goal.target_y:g} "
                  f"{goal.units}) ===", flush=True)
            try:
                rec = run_one_goal(
                    v, goal, gi, provider,
                    seed=args.seed, max_turns=args.max_turns,
                    out_dir=out_dir, oracle_dry_run=use_oracle,
                    include_apparatus=not args.no_instruments,
                )
            except Exception as e:
                traceback.print_exc()
                rec = GoalRecord(
                    scene_id=v["id"], slug=v["slug"],
                    goal_idx=gi, goal=goal.__dict__,
                    model=(provider.model if provider else "oracle"),
                    seed=args.seed, error=str(e),
                )
            summary.append({
                "id": rec.scene_id, "slug": rec.slug,
                "goal_idx": rec.goal_idx,
                "difficulty": goal.difficulty,
                "target_y": goal.target_y,
                "passed": rec.passed,
                "pass_turn": rec.pass_turn,
                "attempts": len(rec.attempts),
                "gave_up": rec.gave_up,
                "elapsed_s": rec.elapsed_s,
                "error": rec.error,
            })
            print(f"  -> passed={rec.passed} "
                  f"attempts={len(rec.attempts)} "
                  f"pass_turn={rec.pass_turn} "
                  f"err={rec.error}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    n_pass = sum(1 for s in summary if s["passed"])
    print(f"\nWrote {out_dir}/summary.json "
          f"({n_pass}/{len(summary)} pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
