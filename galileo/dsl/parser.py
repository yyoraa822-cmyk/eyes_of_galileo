"""YAML-based parser for Galileo-DSL.

The DSL is a restricted YAML dialect. We parse raw YAML into our Scene/Entity
dataclasses and raise descriptive errors that can be surfaced back to the VLM.
"""
from __future__ import annotations

from typing import Any

import yaml

from .schema import (
    BODY_SCHEMA,
    CONNECTION_TYPES,
    ENTITY_SCHEMA,
    FORBIDDEN_KEYS,
    Body,
    Camera,
    Connection,
    Entity,
    Scene,
)


class DSLError(Exception):
    """Raised when the DSL source is syntactically invalid.

    The message is designed to be returned to the VLM, so it should be
    actionable (point to the offending key / suggest a fix).
    """


def _check_forbidden(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            if k in FORBIDDEN_KEYS:
                raise DSLError(
                    f"{path}.{k}: forbidden key '{k}'. Text/numeric labels "
                    f"are not allowed; apparatus must be observable only "
                    f"through physical geometry."
                )
            _check_forbidden(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _check_forbidden(item, f"{path}[{i}]")


def _parse_body(kind: str, spec: dict[str, Any], path: str) -> Body:
    if kind not in BODY_SCHEMA:
        raise DSLError(
            f"{path}: unknown body type '{kind}'. "
            f"Allowed: {sorted(BODY_SCHEMA)}."
        )
    schema = BODY_SCHEMA[kind]
    if not isinstance(spec, dict):
        raise DSLError(f"{path}: body spec must be a mapping, got {type(spec).__name__}.")
    for req in schema["required"]:
        if req not in spec:
            raise DSLError(
                f"{path}: body of type '{kind}' requires '{req}'. "
                f"Got keys: {sorted(spec)}."
            )
    allowed = set(schema["required"]) | set(schema["optional"])
    for key in spec:
        if key not in allowed:
            raise DSLError(
                f"{path}.{key}: unknown parameter for body '{kind}'. "
                f"Allowed: {sorted(allowed)}."
            )
    params = dict(schema["optional"])
    params.update({k: v for k, v in spec.items() if k != "name"})
    return Body(kind=kind, name=spec["name"], params=params)


def _parse_entity(raw: dict[str, Any], path: str) -> Entity:
    if not isinstance(raw, dict):
        raise DSLError(f"{path}: entity must be a mapping.")
    for req in ("name", "type", "params"):
        if req not in raw:
            raise DSLError(f"{path}: entity missing '{req}'.")
    kind = raw["type"]
    if kind not in ENTITY_SCHEMA:
        raise DSLError(
            f"{path}: unknown entity type '{kind}'. "
            f"Allowed: {sorted(ENTITY_SCHEMA)}."
        )
    schema = ENTITY_SCHEMA[kind]
    ent_params = raw["params"] or {}
    if not isinstance(ent_params, dict):
        raise DSLError(f"{path}.params: must be a mapping.")

    bodies: dict[str, Body] = {}
    for body_slot in schema.get("required_bodies", []):
        if body_slot not in ent_params:
            body_schema = BODY_SCHEMA.get(body_slot, {})
            req = body_schema.get("required", [])
            opt = list(body_schema.get("optional", {}).keys())
            hint_parts = []
            if req:
                hint_parts.append(
                    f"required fields: {req}"
                )
            if opt:
                hint_parts.append(
                    f"optional fields: {opt}"
                )
            all_slots = (schema.get("required_bodies", []) +
                         schema.get("optional_bodies", []))
            raise DSLError(
                f"{path}: entity '{kind}' requires body slot "
                f"'{body_slot}' inside params. Entity body slots: "
                f"{all_slots}. For this slot use "
                f"{body_slot}: {{{', '.join(req)}}} "
                + ("(" + "; ".join(hint_parts) + ")" if hint_parts else "")
            )
        bodies[body_slot] = _parse_body(
            _guess_body_kind(body_slot, ent_params[body_slot]),
            ent_params[body_slot],
            f"{path}.params.{body_slot}",
        )
    for body_slot in schema.get("optional_bodies", []):
        if body_slot in ent_params and ent_params[body_slot] is not None:
            bodies[body_slot] = _parse_body(
                _guess_body_kind(body_slot, ent_params[body_slot]),
                ent_params[body_slot],
                f"{path}.params.{body_slot}",
            )

    pure_params = dict(schema.get("optional_params", {}))
    for req in schema.get("required_params", []):
        if req not in ent_params:
            raise DSLError(f"{path}: entity '{kind}' requires param '{req}'.")
        pure_params[req] = ent_params[req]
    for k, v in ent_params.items():
        if k in schema.get("required_bodies", []):
            continue
        if k in schema.get("optional_bodies", []):
            continue
        if k in schema.get("required_params", []):
            continue
        if k in schema.get("optional_params", {}):
            pure_params[k] = v
            continue
        raise DSLError(
            f"{path}.params.{k}: unknown param for entity '{kind}'."
        )

    return Entity(name=raw["name"], kind=kind, bodies=bodies, params=pure_params)


def _guess_body_kind(slot: str, spec: dict[str, Any]) -> str:
    """Infer a body's type from its slot name (ruler_along -> ruler, etc.).

    This lets the YAML be concise: the user doesn't have to write
    `type: ruler` when the slot name unambiguously implies it.
    """
    aliases = {
        "ruler_along": "ruler",
    }
    if slot in aliases:
        return aliases[slot]
    if slot in BODY_SCHEMA:
        return slot
    if isinstance(spec, dict) and "type" in spec:
        return spec["type"]
    raise DSLError(
        f"Cannot infer body type for slot '{slot}'. "
        f"Rename the slot to one of {sorted(BODY_SCHEMA)} or add `type:`."
    )


def _parse_connection(raw: dict[str, Any], path: str) -> Connection:
    if not isinstance(raw, dict):
        raise DSLError(f"{path}: connection must be a mapping.")
    for req in ("type", "from", "to"):
        if req not in raw:
            raise DSLError(f"{path}: connection missing '{req}'.")
    if raw["type"] not in CONNECTION_TYPES:
        raise DSLError(
            f"{path}.type: unknown connection '{raw['type']}'. "
            f"Allowed: {sorted(CONNECTION_TYPES)}."
        )
    return Connection(kind=raw["type"], src=str(raw["from"]), dst=str(raw["to"]))


def _parse_camera(raw: dict[str, Any] | None) -> Camera:
    if raw is None:
        return Camera()
    c = Camera()
    if "position" in raw:
        c.position = tuple(float(x) for x in raw["position"])
    if "target" in raw:
        c.target = tuple(float(x) for x in raw["target"])
    if "fov" in raw:
        c.fov = float(raw["fov"])
    return c


def parse(dsl_source: str) -> Scene:
    """Parse a YAML DSL source string into a Scene.

    Performs only **Gate 1** (syntactic) checks: schema + forbidden keys.
    Gate 2 and Gate 3 run later in validator.py / compiler.py.
    """
    try:
        raw = yaml.safe_load(dsl_source)
    except yaml.YAMLError as e:
        raise DSLError(f"YAML parse error: {e}") from e
    if not isinstance(raw, dict) or "scene" not in raw:
        raise DSLError("Top-level must be a mapping with a 'scene:' key.")
    scene_raw = raw["scene"]
    if not isinstance(scene_raw, dict):
        raise DSLError("'scene' must be a mapping.")

    _check_forbidden(scene_raw, "scene")

    name = scene_raw.get("name", "unnamed")
    entities_raw = scene_raw.get("entities", []) or []
    connections_raw = scene_raw.get("connections", []) or []
    camera_raw = scene_raw.get("camera")

    entities: dict[str, Entity] = {}
    for i, raw_ent in enumerate(entities_raw):
        ent = _parse_entity(raw_ent, f"scene.entities[{i}]")
        if ent.name in entities:
            raise DSLError(f"Duplicate entity name: {ent.name!r}.")
        entities[ent.name] = ent

    connections = [
        _parse_connection(c, f"scene.connections[{i}]")
        for i, c in enumerate(connections_raw)
    ]

    return Scene(
        name=str(name),
        entities=entities,
        connections=connections,
        camera=_parse_camera(camera_raw),
    )
