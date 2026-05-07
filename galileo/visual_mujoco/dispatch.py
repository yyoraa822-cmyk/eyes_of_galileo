"""Public dispatch: scene_id -> render_animation.

Lazy-imports per-scene module so missing implementations only fail
when actually requested. Demo builders call `render_animation` once
per scene; mechanical scenes still go through DSL/MuJoCo at the
caller's discretion (`supports` returns False for them).
"""
from __future__ import annotations

import importlib
from typing import Any

from PIL import Image

# Map of scene_id -> module relative to galileo.visual_mujoco.
# Mechanical scenes are intentionally absent here; demo builders
# fall back to the DSL pipeline for those (compile_and_render).
_SCENE_MODULES: dict[str, str] = {
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
}


def supports(scene_id: str) -> bool:
    return scene_id in _SCENE_MODULES


def render_animation(
    scene_id: str, scenario: Any, control_value: float, *,
    n_frames: int = 24, width: int = 960, height: int = 600,
) -> tuple[list[Image.Image], int]:
    """Return (frames, fps) for a SB scene's MuJoCo visualisation.
    Raises KeyError for unsupported scenes (use `supports` first)."""
    mod_name = _SCENE_MODULES[scene_id]
    mod = importlib.import_module(f"galileo.visual_mujoco.{mod_name}")
    return mod.render_animation(
        scenario, control_value,
        n_frames=n_frames, width=width, height=height,
    )
