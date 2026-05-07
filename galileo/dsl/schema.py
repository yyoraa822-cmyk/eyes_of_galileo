"""Schema definitions for Galileo-DSL bodies, entities, and scenes.

Allowed bodies and entities are explicitly enumerated here — the validator
uses this as the single source of truth. Adding a new primitive requires
editing only this file plus the corresponding MJCF template in compiler.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BODY_SCHEMA: dict[str, dict[str, Any]] = {
    "ball": {
        "required": ["name"],
        "optional": {"radius": 0.12, "color": "red"},
    },
    "ramp": {
        "required": ["name", "angle_deg", "length"],
        "optional": {"friction": 0.0},
    },
    "wall": {
        "required": ["name", "length"],
        "optional": {"normal": [1, 0, 0]},
    },
    "rod": {
        "required": ["name", "length"],
        "optional": {},
    },
    "string": {
        "required": ["name", "length"],
        "optional": {},
    },
    "ruler": {
        "required": ["name", "length", "tick_spacing"],
        "optional": {},
    },
    "marker": {
        "required": ["name"],
        "optional": {"shape": "cross", "color": "white", "size": 0.05},
    },
    "reference_cube": {
        "required": ["name", "edge_length"],
        "optional": {},
    },
}


ENTITY_SCHEMA: dict[str, dict[str, Any]] = {
    "InclinedRamp": {
        "required_bodies": ["ramp"],
        "optional_bodies": ["ruler_along"],
        "optional_params": {"origin": [0.0, 0.0, 0.0]},
        "connection_points": ["top", "bottom", "surface"],
    },
    "FreefallBall": {
        "required_bodies": ["ball"],
        "optional_params": {
            "release_height": 0.0,
            "origin": [0.0, 0.0, 0.0],
            "on_ramp": "",
        },
        "connection_points": ["release_point"],
    },
    "HorizontalLaunch": {
        "required_bodies": ["ball"],
        "optional_params": {
            "v0": 1.0,
            "launch_height": 2.0,
            "origin": [0.0, 0.0, 0.0],
        },
        "connection_points": ["release_point"],
    },
    "CircularMotion": {
        "required_bodies": ["ball"],
        "optional_params": {
            "radius": 1.0,
            "v": 2.0,
            "center_height": 2.0,
            "origin": [0.0, 0.0, 0.0],
        },
        "connection_points": ["center"],
    },
    "SpringBlock": {
        "required_bodies": ["ball"],
        "optional_params": {
            "k": 10.0,
            "mass": 1.0,
            "amplitude": 0.8,
            "surface_height": 0.0,
            "origin": [0.0, 0.0, 0.0],
        },
        "connection_points": ["anchor"],
    },
    "Pendulum": {
        "required_bodies": ["string", "ball"],
        "optional_params": {
            "pivot_height": 2.5,
            "theta_max_deg": 17.0,
            "n_samples": 10,
            "origin": [0.0, 0.0, 0.0],
        },
        "connection_points": ["pivot"],
    },
    "SecondBall": {
        "required_bodies": ["ball"],
        "optional_params": {"position": [0.0, 0.0, 0.5]},
        "connection_points": ["anchor"],
    },
    "StrobeTrail": {
        "required_params": ["target_body"],
        "optional_params": {"n_samples": 10, "sampling": "equal_time"},
        "connection_points": [],
    },
    "ReferenceScale": {
        "required_bodies": ["reference_cube"],
        "optional_params": {"position": [0.0, 0.0, 0.0]},
        "connection_points": ["anchor"],
    },
    "TimingGate": {
        "optional_params": {
            "axis": "x",
            "position": 1.0,
            "height": 1.2,
            "color": "yellow",
        },
        "connection_points": [],
    },
    "ChimeTrack": {
        "required_params": ["target_body"],
        "optional_params": {
            "positions": [1.0, 2.0, 3.0, 4.0, 5.0],
            "metronome_ticks": 40,
        },
        "connection_points": [],
    },
    "GridFloor": {
        "optional_params": {
            "cell_size": 0.5,
            "extent": 6.0,
        },
        "connection_points": [],
    },
    "FadingTrail": {
        "required_params": ["target_body"],
        "optional_params": {
            "trail_width": 4,
            "n_samples": 60,
        },
        "connection_points": [],
    },
    "AngleProtractor": {
        "optional_params": {
            "origin": [0.0, 0.0, 0.0],
            "radius": 1.5,
            "max_angle_deg": 90.0,
            "tick_step_deg": 10.0,
        },
        "connection_points": [],
    },
    "Filmstrip": {
        "required_params": ["target_body"],
        "optional_params": {
            "columns": 5,
            "frame_size": 200,
        },
        "connection_points": [],
    },
    "Animation": {
        "required_params": ["target_body"],
        "optional_params": {
            "fps": 8,
            "n_frames": 32,
        },
        "connection_points": [],
    },
    "BackgroundGrid": {
        "optional_params": {
            "cell_px": 64,
            "axis_labels": False,
        },
        "connection_points": [],
    },
    "MassStack": {
        "required_params": ["target_body"],
        "optional_params": {
            "mass": 1.0,
            "unit_size": 0.1,
            "position": "above",
        },
        "connection_points": [],
    },
    "LightStrip": {
        "required_params": ["target_body"],
        "optional_params": {
            "n_samples": 12,
            "channel": "luminance",
        },
        "connection_points": [],
    },
    "SpectrumBand": {
        "required_params": ["target_body"],
        "optional_params": {
            "palette": "thermal",
            "min": 0.0,
            "max": 1.0,
            "anchor": "right",
        },
        "connection_points": [],
    },
    "ScenarioBackend": {
        # Delegate rendering to a registered Scenario.* class (matplotlib
        # apparatus). The DSL still parses + validates the scene; render
        # dispatch in api.compile_and_render routes to scenario.render_frames.
        "required_params": ["slug"],
        "optional_params": {
            "alpha": None,
            "kwargs": {},
            "clean": False,
        },
        "connection_points": [],
    },
    # ── visual_mujoco-side instruments (registered here so the agent
    # can write them in YAML; rendering happens inside the per-scene
    # visual_mujoco module that recognises them) ────────────────────
    "WaxPellets": {
        # Ingenhousz's wax-coated rod (1789): a row of small wax beads
        # along the heated rod. Each turns from grey (solid) to amber
        # (melted) once the heat front passes its position. Dynamic.
        "optional_params": {
            "spacing": 0.5,
            "radius": 0.10,
            "axis": "x",
        },
        "connection_points": [],
    },
    "PlainRuler": {
        # A featureless rod with no tick marks — a pure visual primitive
        # the agent can place to compare left/right or judge orientation
        # without introducing numeric scale information.
        "required_params": ["length"],
        "optional_params": {
            "origin": [0.0, 0.0, 0.0],
            "axis": "x",
            "color": "white",
        },
        "connection_points": [],
    },
    "SideBySide": {
        # Render multiple control_value runs of the same scene tiled
        # next to each other so the agent sees several experiments at
        # once. The list of `controls` and per-cell `width` tile the
        # output horizontally.
        "required_params": ["controls"],
        "optional_params": {
            "rows": 1,
            "cell_width": 320,
            "cell_height": 240,
            "gap_px": 6,
        },
        "connection_points": [],
    },
}


CONNECTION_TYPES = {"place_on", "attach", "pivot_at"}


FORBIDDEN_KEYS = {
    "text", "label", "annotation", "show_numbers", "display_value",
    "print_position", "readout",
}


@dataclass
class Body:
    kind: str
    name: str
    params: dict[str, Any]


@dataclass
class Entity:
    name: str
    kind: str
    bodies: dict[str, Body]
    params: dict[str, Any]


@dataclass
class Connection:
    kind: str
    src: str
    dst: str


@dataclass
class Camera:
    position: tuple[float, float, float] = (0.0, -8.0, 2.0)
    target: tuple[float, float, float] = (2.5, 0.0, 1.0)
    fov: float = 45.0


@dataclass
class Scene:
    name: str
    entities: dict[str, Entity] = field(default_factory=dict)
    connections: list[Connection] = field(default_factory=list)
    camera: Camera = field(default_factory=Camera)
