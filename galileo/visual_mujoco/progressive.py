"""Progressive-build animations for visual_mujoco scenes.

Wraps the existing per-scene `render_animation` so that an animation
GIF starts from an empty floor and adds the apparatus pieces one at a
time before running the physics. We work entirely from the MJCF
string produced by each scene's `_build_mjcf` (no per-scene refactor):

  1. Parse the worldbody and identify the top-level bodies (skipping
     floor / camera / light, which always stay).
  2. For build stage k in [0, N], rebuild the MJCF with only the
     first k bodies and render one frame.
  3. After all bodies are in place, run the scene's normal physics
     animation as the "tail" of the GIF.

The label "DSL-add" comes from the project's own DSL surface that
constructs apparatuses by attaching one Instrument at a time — this
helper visualises the same mental model on the visual_mujoco side.
"""
from __future__ import annotations

import importlib
import io
import xml.etree.ElementTree as ET
from typing import Any, Optional

import mujoco
import numpy as np
from PIL import Image

from ._base import render_frames_with_state


# Bodies whose name matches one of these prefixes are treated as part
# of the static stage (always present in every frame), not as a tool
# being added. Most scenes don't have any.
_BASE_NAME_PREFIXES = ("__base__",)


def _group_key(body_name: str) -> str:
    """Bodies that share a structured prefix (`tick_0`, `tick_1`, ...)
    are visually one tool — batch them into a single stage so the
    build animation doesn't drag for 18 frames adding tick marks."""
    import re
    m = re.match(r"^([a-zA-Z][a-zA-Z_]*?)_-?\d+$", body_name)
    if m:
        return m.group(1)
    return body_name


def _group_bodies(bodies: list[tuple[str, str]]
                  ) -> list[list[tuple[str, str]]]:
    """Pack consecutive bodies that share a `_group_key` into one stage."""
    out: list[list[tuple[str, str]]] = []
    last_key: str | None = None
    for n, x in bodies:
        k = _group_key(n)
        if last_key is not None and k == last_key and out:
            out[-1].append((n, x))
        else:
            out.append([(n, x)])
            last_key = k
    return out


def _split_bodies(mjcf: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (mjcf_without_extra_bodies, [(body_name, body_xml) ...])
    for every top-level body inside <worldbody>. Camera / light /
    floor stay in the prefix MJCF because they're not <body> elements
    (camera and light are siblings of <body>, floor is a <geom>)."""
    # ElementTree mangles formatting but we only need the structure.
    root = ET.fromstring(mjcf)
    wb = root.find("worldbody")
    if wb is None:
        return mjcf, []
    extracted: list[tuple[str, str]] = []
    for body in list(wb.findall("body")):
        name = body.get("name") or ""
        xml = ET.tostring(body, encoding="unicode")
        wb.remove(body)
        extracted.append((name, xml))
    base_mjcf = ET.tostring(root, encoding="unicode")
    return base_mjcf, extracted


def _stitch(base_mjcf: str, body_xmls: list[str]) -> str:
    root = ET.fromstring(base_mjcf)
    wb = root.find("worldbody")
    for x in body_xmls:
        wb.append(ET.fromstring(x))
    return ET.tostring(root, encoding="unicode")


def render_progressive(
    scene_id: str,
    scenario: Any,
    control_value: float,
    *,
    build_per_stage_frames: int = 2,
    physics_frames: int = 24,
    width: int = 480,
    height: int = 320,
) -> tuple[list[Image.Image], int]:
    """Render an animation that builds up the apparatus body-by-body
    and then runs the physics. Returns (frames, fps)."""
    mod_name = {
        "s19_coulomb":    "coulomb",
        "s18_boyle":      "boyle",
        "s17_refraction": "refraction",
        "s8_heat":        "heat",
        "s16_decay":      "decay",
        "s21_weber":      "blackbody",
        "s22_cooling":    "cooling",
        "s6_launch":      "projectile",
        "s9_spring":      "spring",
        "s10_circular":   "orbital",
        "s20_hooke":      "hooke",
        "s2_mass":        "mass",
    }[scene_id]
    mod = importlib.import_module(f"galileo.visual_mujoco.{mod_name}")

    # Scenes that take build args expose `_build_mjcf(...)` with a
    # scene-specific signature. We call them through a helper:
    if mod_name == "projectile":
        sim = scenario.simulate(float(control_value))
        full_mjcf = mod._build_mjcf(float(sim["range"]),
                                    float(np.max(sim["y"])))
    elif mod_name == "orbital":
        sim = scenario.simulate(float(control_value))
        full_mjcf = mod._build_mjcf(float(sim["a"]))
    elif mod_name == "spring":
        sim = scenario.simulate(float(control_value))
        ext = float(sim["extension"])
        full_mjcf = mod._build_mjcf(min(ext, 1.40))
    elif mod_name == "hooke":
        sim = scenario.simulate(float(control_value))
        ext = float(sim["extension"])
        full_mjcf = mod._build_mjcf(min(ext, 1.50))
    elif mod_name == "coulomb":
        sim = scenario.simulate(float(control_value))
        full_mjcf = mod._build_mjcf(float(sim["distance"]))
    else:
        full_mjcf = mod._build_mjcf()

    base_mjcf, bodies = _split_bodies(full_mjcf)

    # Filter out "base" bodies (always present in every frame).
    fixed_bodies = [
        (n, x) for (n, x) in bodies
        if any(n.startswith(p) for p in _BASE_NAME_PREFIXES)
    ]
    tool_bodies = [
        (n, x) for (n, x) in bodies
        if not any(n.startswith(p) for p in _BASE_NAME_PREFIXES)
    ]

    fps = 8
    out: list[Image.Image] = []

    # Group bodies by name-prefix so e.g. 18 protractor ticks become
    # one stage instead of 18.
    grouped = _group_bodies(tool_bodies)

    # Build phase: 1 stage per group, repeated for k frames per stage
    # so the viewer can see what was added. Stage 0 = empty floor.
    cumulative_xmls: list[str] = [x for _, x in fixed_bodies]
    for stage in range(0, len(grouped) + 1):
        if stage > 0:
            for _, x in grouped[stage - 1]:
                cumulative_xmls.append(x)
        sub = list(cumulative_xmls)
        mjcf = _stitch(base_mjcf, sub)
        # Render `build_per_stage_frames` identical frames at this stage.
        try:
            frames = render_frames_with_state(
                mjcf,
                lambda tau, m, d: None,
                n_frames=build_per_stage_frames,
                width=width, height=height,
            )
            out.extend(frames)
        except (ValueError, RuntimeError) as e:
            # Some intermediate subsets may be invalid (e.g. a mocap
            # body referencing a missing parent); we just skip them.
            pass

    # Physics phase: full apparatus running its normal animation.
    physics_frames_imgs, _ = mod.render_animation(
        scenario, control_value,
        n_frames=physics_frames, width=width, height=height,
    )
    out.extend(physics_frames_imgs)

    return out, fps


def save_gif(frames: list[Image.Image], path: str, fps: int = 8) -> None:
    duration_ms = int(1000 / max(fps, 1))
    frames[0].save(path, format="GIF", save_all=True,
                   append_images=frames[1:], duration=duration_ms,
                   loop=0, optimize=True)
