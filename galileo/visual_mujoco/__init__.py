"""MuJoCo-based visualisation for ALL 15 benchmark scenes.

This package mirrors the `galileo/scenarios/` matplotlib renderers but
returns a list of PIL Images produced by MuJoCo's offscreen renderer.
The visual aesthetic is intentionally aligned with the original
`draft/demo.html` reference (dark grey floor, ambient lighting,
3D primitives), since that page predates this repo's matplotlib
fallbacks but only its output GIFs survived.

Per-scene modules expose:

    render_animation(scenario, control_value, *,
                     n_frames=32, width=960, height=600
                     ) -> tuple[list[PIL.Image.Image], int]

returning (frames, fps). fps defaults to 8 to match demo.html's
GIF cadence.

Mechanical scenes (s1/s3/s9/...) keep going through the existing DSL
pipeline; this module only re-implements the 7 ScenarioBackend scenes
plus a uniform animation entry point. The dispatch helper below
provides a single call site for demo builders.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .dispatch import render_animation, supports

if TYPE_CHECKING:
    from PIL import Image  # noqa: F401

__all__ = ["render_animation", "supports"]
