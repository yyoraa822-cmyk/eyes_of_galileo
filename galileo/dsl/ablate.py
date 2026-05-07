"""Shortcut ablations for Galileo-DSL scenes.

Given a DSL YAML source, produce an *ablated* variant that removes a
chosen set of affordances (e.g. ReferenceScale, ruler_along) and render
both side-by-side for visual sanity checking or for downstream
experiments that ask "did this affordance actually help the VLM?".
"""
from __future__ import annotations

from typing import Iterable

import yaml
from PIL import Image

from .api import compile_and_render


def strip_entities(dsl_source: str, kinds: Iterable[str]) -> str:
    """Return a YAML string with all entities whose `type` is in `kinds` removed.

    Also drops `ruler_along` optional bodies from surviving entities when
    `"ruler_along"` is in `kinds` (it is a body, not an entity, but the
    caller thinks of rulers as an affordance).
    """
    kinds = set(kinds)
    body_kinds_to_strip = {"ruler_along"} & kinds
    entity_kinds_to_strip = kinds - body_kinds_to_strip

    raw = yaml.safe_load(dsl_source)
    if not isinstance(raw, dict) or "scene" not in raw:
        return dsl_source
    scene = raw["scene"]

    ents = scene.get("entities", []) or []
    kept = []
    for e in ents:
        if not isinstance(e, dict):
            kept.append(e)
            continue
        if e.get("type") in entity_kinds_to_strip:
            continue
        params = e.get("params") or {}
        for bk in list(body_kinds_to_strip):
            if bk in params:
                del params[bk]
        kept.append(e)
    scene["entities"] = kept
    return yaml.safe_dump(raw, sort_keys=False)


def ablate_and_render_pair(
    dsl_source: str,
    control_value: float,
    alpha: float = 2.5,
    ablate_kinds: Iterable[str] = ("ReferenceScale", "ruler_along"),
    width: int = 640,
    height: int = 480,
) -> tuple[Image.Image, Image.Image, str]:
    """Render the full apparatus and an ablated variant at the same cv/alpha.

    Returns (full_img, ablated_img, ablated_yaml).
    """
    full = compile_and_render(
        dsl_source, control_value=control_value, alpha=alpha,
        width=width, height=height,
    )
    ablated_src = strip_entities(dsl_source, ablate_kinds)
    ablated = compile_and_render(
        ablated_src, control_value=control_value, alpha=alpha,
        width=width, height=height,
    )
    return full, ablated, ablated_src


def side_by_side(left: Image.Image, right: Image.Image,
                 pad: int = 10, bg=(0, 0, 0)) -> Image.Image:
    """Stack two same-height images horizontally with a thin separator."""
    h = max(left.height, right.height)
    w = left.width + right.width + pad
    out = Image.new("RGB", (w, h), bg)
    out.paste(left, (0, 0))
    out.paste(right, (left.width + pad, 0))
    return out
