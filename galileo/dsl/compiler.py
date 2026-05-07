"""DSL -> MJCF compiler + strobe rendering with counterfactual physics.

A scene may contain multiple independent motion sources (ramp + ball,
free-falling ball, pendulum, horizontal launch). Each motion source
registers a `TrackedBody` with its own hidden-law recipe; at render
time we step through strobe samples and drive each tracked body's
qpos according to its recipe. MuJoCo renders the apparatus faithfully
(no real dynamics run).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .schema import Body, Entity, Scene


G = 9.81


COLOR_MAP = {
    "red":    "0.85 0.20 0.20 1",
    "blue":   "0.20 0.45 0.90 1",
    "green":  "0.25 0.70 0.25 1",
    "yellow": "1.00 0.95 0.20 1",
    "white":  "0.95 0.95 0.95 1",
    "black":  "0.10 0.10 0.10 1",
    "gray":   "0.55 0.55 0.55 1",
    "orange": "0.95 0.55 0.15 1",
}


def _rgba(color: str) -> str:
    return COLOR_MAP.get(str(color).lower(), COLOR_MAP["red"])


@dataclass
class TrackedBody:
    """Declares how one body moves under the hidden counterfactual law.

    kind is one of: "ramp", "freefall_vertical", "pendulum",
    "horizontal_launch". Each kind knows how to compute a qpos tuple
    given (t, cv, alpha, params).
    """
    name: str
    kind: str
    joints: list[str]             # joint names whose qpos we will set
    params: dict                  # per-kind parameters (length, theta, v0, ...)


@dataclass
class CompiledScene:
    mjcf: str
    tracked_bodies: list[TrackedBody]
    default_target: str           # which tracked body gets strobed by default
    metadata: dict = field(default_factory=dict)


# ---------- geometry helpers ----------

def _inclined_ramp_xml(ent: Entity) -> tuple[str, dict]:
    ramp: Body = ent.bodies["ramp"]
    angle = float(ramp.params["angle_deg"])
    length = float(ramp.params["length"])
    origin = ent.params.get("origin", [0.0, 0.0, 0.0])
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    theta = np.radians(angle)
    sin_t, cos_t = np.sin(theta), np.cos(theta)

    thickness = max(0.08, length * 0.012)
    width = max(0.8, length * 0.08)

    cx = ox + length * cos_t * 0.5
    cz = oz - length * sin_t * 0.5
    cy = oy

    xml = (
        f'    <geom name="{ramp.name}" type="box" '
        f'size="{length*0.5:.4f} {width*0.5:.4f} {thickness*0.5:.4f}" '
        f'pos="{cx:.4f} {cy:.4f} {cz:.4f}" '
        f'euler="0 {angle:.2f} 0" material="ramp_mat"/>\n'
    )
    if "ruler_along" in ent.bodies:
        ruler = ent.bodies["ruler_along"]
        tick_spacing = float(ruler.params["tick_spacing"])
        ruler_len = float(ruler.params["length"])
        n_ticks = max(2, int(ruler_len / tick_spacing))
        post_h = max(0.25, length * 0.07)
        post_hw = max(0.025, length * 0.006)
        y_off = width * 0.5 + 0.08
        for k in range(1, n_ticks + 1):
            d = k * tick_spacing
            if d > length:
                break
            tx = ox + d * cos_t + np.sin(theta) * post_h * 0.5
            tz = oz - d * sin_t + np.cos(theta) * post_h * 0.5
            for y_sign in (+1, -1):
                xml += (
                    f'    <geom name="{ruler.name}_{k}_{y_sign}" type="box" '
                    f'size="{post_hw:.4f} {post_hw:.4f} {post_h*0.5:.4f}" '
                    f'pos="{tx:.4f} {oy + y_sign*y_off:.4f} {tz:.4f}" '
                    f'euler="0 {angle:.2f} 0" rgba="1 0.95 0.2 1"/>\n'
                )

    return xml, {
        "name": ent.name,
        "angle_deg": angle,
        "length": length,
        "thickness": thickness,
        "width": width,
        "cos_t": cos_t,
        "sin_t": sin_t,
        "ox": ox, "oy": oy, "oz": oz,
    }


def _reference_scale_xml(ent: Entity) -> str:
    cube: Body = ent.bodies["reference_cube"]
    edge = float(cube.params["edge_length"])
    pos = ent.params.get("position", [0.0, 0.0, 0.0])
    half = edge * 0.5
    return (
        f'    <geom name="{cube.name}" type="box" '
        f'size="{half:.4f} {half:.4f} {half:.4f}" '
        f'pos="{float(pos[0]):.4f} {float(pos[1]):.4f} '
        f'{float(pos[2]):.4f}" rgba="0.30 0.30 0.80 1"/>\n'
    )


def _second_ball_xml(ent: Entity) -> str:
    ball: Body = ent.bodies["ball"]
    radius = float(ball.params.get("radius", 0.12))
    color = ball.params.get("color", "blue")
    pos = ent.params.get("position", [0.0, 0.0, 0.5])
    return (
        f'    <geom name="{ball.name}" type="sphere" '
        f'size="{radius:.4f}" '
        f'pos="{float(pos[0]):.4f} {float(pos[1]):.4f} {float(pos[2]):.4f}" '
        f'rgba="{_rgba(color)}"/>\n'
    )


def _chime_bells_xml(ent: Entity, tb: "TrackedBody",
                     positions: list[float]) -> tuple[str, list[dict]]:
    """Place one small static 'bell' geom at each requested distance.

    Returns (mjcf_fragment, list_of_bell_dicts) where each dict has
    keys {"name": str, "index": int, "x": float, "y": float, "z": float,
    "distance": float}. `distance` is the input position (arbitrary
    distance unit along the motion axis of `tb`).
    """
    p = tb.params
    out_xml = ""
    bells: list[dict] = []
    base = ent.name
    for i, d in enumerate(positions):
        if tb.kind == "freefall_vertical":
            ox = float(p.get("ox", 0.0))
            oy = float(p.get("oy", 0.0))
            start_z = float(p.get("start_z", 2.0))
            x, y, z = ox + 0.35, oy, start_z - d
            if z < 0.02:
                continue
            shape = "sphere"
            size = "0.08"
        elif tb.kind == "ramp":
            ox = float(p.get("ox", 0.0))
            oy = float(p.get("oy", 0.0))
            oz = float(p.get("oz", 0.0))
            angle = np.radians(float(p.get("angle_deg", 20.0)))
            length = float(p.get("length", 4.0))
            thickness = float(p.get("thickness", 0.08))
            width = float(p.get("width", 0.8))
            if d > length:
                continue
            cos_t, sin_t = float(np.cos(angle)), float(np.sin(angle))
            # surface-normal lift + lateral offset to the +Y side of
            # the ramp so the bell doesn't occlude the ball
            lift = thickness * 0.5 + 0.04
            x = ox + d * cos_t + sin_t * lift
            z = oz - d * sin_t + cos_t * lift
            y = oy + width * 0.5 + 0.06
            shape = "sphere"
            size = "0.07"
        elif tb.kind == "horizontal_launch":
            ox = float(p.get("ox", 0.0))
            oy = float(p.get("oy", 0.0))
            x, y, z = ox + d, oy, 0.02
            shape = "cylinder"
            size = "0.08 0.01"
        else:
            continue
        name = f"{base}_bell{i+1}"
        if shape == "sphere":
            out_xml += (
                f'    <geom name="{name}" type="sphere" '
                f'size="{size}" '
                f'pos="{x:.4f} {y:.4f} {z:.4f}" '
                f'rgba="1.0 0.85 0.1 1"/>\n'
            )
        else:
            out_xml += (
                f'    <geom name="{name}" type="{shape}" '
                f'size="{size}" '
                f'pos="{x:.4f} {y:.4f} {z:.4f}" '
                f'euler="0 0 0" '
                f'rgba="1.0 0.85 0.1 1"/>\n'
            )
        bells.append({
            "name": name, "index": i + 1,
            "x": float(x), "y": float(y), "z": float(z),
            "distance": float(d),
            "target_kind": tb.kind,
            "target_name": tb.name,
        })
    return out_xml, bells


def _timing_gate_xml(ent: Entity) -> str:
    axis = str(ent.params.get("axis", "x")).lower()
    pos = float(ent.params.get("position", 1.0))
    height = float(ent.params.get("height", 1.2))
    color = ent.params.get("color", "yellow")
    half_h = height * 0.5
    # a thin vertical "laser" at the given x (or y) position
    if axis == "x":
        return (
            f'    <geom name="{ent.name}_gate" type="box" '
            f'size="0.010 0.010 {half_h:.4f}" '
            f'pos="{pos:.4f} 0 {half_h:.4f}" '
            f'rgba="{_rgba(color)}"/>\n'
        )
    else:
        return (
            f'    <geom name="{ent.name}_gate" type="box" '
            f'size="0.010 0.010 {half_h:.4f}" '
            f'pos="0 {pos:.4f} {half_h:.4f}" '
            f'rgba="{_rgba(color)}"/>\n'
        )


def _mass_stack_xml(ent: Entity, body_pos: tuple[float, float, float],
                    body_radius: float) -> str:
    """Render N stacked unit-cube weights to denote mass on a target body.

    Each unit cube is fixed-size (unit_size on each side). To represent
    mass = m we stack round(m) cubes adjacent to the body, so a VLM
    counts cubes rather than estimating volume from a single deformed
    sphere.
    """
    mass = float(ent.params.get("mass", 1.0))
    unit = float(ent.params.get("unit_size", 0.1))
    position = str(ent.params.get("position", "above")).lower()
    n = max(1, int(round(mass)))
    half = unit * 0.5
    bx, by, bz = body_pos
    xml = ""
    if position == "above":
        base_z = bz + body_radius + half
        for i in range(n):
            cz = base_z + i * unit
            xml += (
                f'    <geom name="{ent.name}_unit{i}" type="box" '
                f'size="{half:.4f} {half:.4f} {half:.4f}" '
                f'pos="{bx:.4f} {by:.4f} {cz:.4f}" '
                f'rgba="0.85 0.55 0.20 1"/>\n'
            )
    else:  # stack horizontally on the +x side of the body
        base_x = bx + body_radius + half
        for i in range(n):
            cx = base_x + i * unit
            xml += (
                f'    <geom name="{ent.name}_unit{i}" type="box" '
                f'size="{half:.4f} {half:.4f} {half:.4f}" '
                f'pos="{cx:.4f} {by:.4f} {bz:.4f}" '
                f'rgba="0.85 0.55 0.20 1"/>\n'
            )
    return xml


def _grid_floor_xml(ent: Entity, floor_z: float) -> str:
    """Render a grid as a series of thin boxes on the floor (cell_size m)."""
    cell = float(ent.params.get("cell_size", 0.5))
    extent = float(ent.params.get("extent", 6.0))
    n = max(2, int(extent / cell))
    line_hw = 0.006
    xml = ""
    for i in range(-n, n + 1):
        x = i * cell
        xml += (
            f'    <geom name="{ent.name}_vx_{i+n}" type="box" '
            f'size="{line_hw:.4f} {extent:.4f} 0.001" '
            f'pos="{x:.4f} 0 {floor_z+0.002:.4f}" rgba="0.35 0.35 0.35 1"/>\n'
        )
        y = i * cell
        xml += (
            f'    <geom name="{ent.name}_vy_{i+n}" type="box" '
            f'size="{extent:.4f} {line_hw:.4f} 0.001" '
            f'pos="0 {y:.4f} {floor_z+0.002:.4f}" rgba="0.35 0.35 0.35 1"/>\n'
        )
    return xml


def _pendulum_xml(ent: Entity) -> tuple[str, str, dict]:
    string: Body = ent.bodies["string"]
    ball: Body = ent.bodies["ball"]
    length = float(string.params["length"])
    bob_radius = float(ball.params.get("radius", 0.08))
    bob_color = ball.params.get("color", "red")
    pivot_h = float(ent.params.get("pivot_height", max(length + 0.4, 2.5)))
    origin = ent.params.get("origin", [0.0, 0.0, 0.0])
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])

    pivot_hw = 0.08
    support_hw = 0.04
    support_h = pivot_h

    world_geom = (
        f'    <geom name="{ent.name}_support" type="box" '
        f'size="{support_hw:.4f} {support_hw:.4f} {support_h*0.5:.4f}" '
        f'pos="{ox - length*0.6:.4f} {oy:.4f} {oz + support_h*0.5:.4f}" '
        f'rgba="0.4 0.3 0.2 1"/>\n'
        f'    <geom name="{ent.name}_beam" type="box" '
        f'size="{length*0.7:.4f} {pivot_hw:.4f} {pivot_hw:.4f}" '
        f'pos="{ox - length*0.6 + length*0.5:.4f} {oy:.4f} '
        f'{oz + support_h+pivot_hw:.4f}" rgba="0.4 0.3 0.2 1"/>\n'
        f'    <geom name="{ent.name}_pivot" type="sphere" size="0.05" '
        f'pos="{ox:.4f} {oy:.4f} {oz+pivot_h:.4f}" rgba="0.15 0.15 0.15 1"/>\n'
    )

    string_radius = max(0.012, length * 0.008)
    string_name = f"{ent.name}_string_geom"
    body_xml = f"""    <body name="{ent.name}_arm" pos="{ox:.4f} {oy:.4f} {oz+pivot_h:.4f}">
      <joint name="{ball.name}_hinge" type="hinge" axis="0 1 0" damping="0"/>
      <geom name="{string_name}" type="capsule"
            fromto="0 0 0 0 0 {-length:.4f}"
            size="{string_radius:.4f}" rgba="0.1 0.1 0.1 1"/>
      <body name="{ball.name}_body" pos="0 0 {-length:.4f}">
        <geom name="{ball.name}" type="sphere" size="{bob_radius:.4f}" rgba="{_rgba(bob_color)}"/>
      </body>
    </body>
"""
    info = {
        "name": ent.name,
        "length": length,
        "pivot_height": pivot_h,
        "theta_max_deg": float(ent.params.get("theta_max_deg", 17.0)),
        "n_samples": int(ent.params.get("n_samples", 10)),
        "bob_name": ball.name,
        "bob_radius": bob_radius,
        "hinge_name": f"{ball.name}_hinge",
        "ox": ox, "oy": oy, "oz": oz,
    }
    return world_geom, body_xml, info


def _freefall_ball_body_xml(
    ent: Entity, ramp_info: Optional[dict], start_z_default: float = 2.0,
) -> tuple[str, str, dict]:
    ball: Body = ent.bodies["ball"]
    radius = float(ball.params.get("radius", 0.12))
    color = ball.params.get("color", "red")
    origin = ent.params.get("origin", [0.0, 0.0, 0.0])
    ox, oy = float(origin[0]), float(origin[1])

    if ramp_info is not None:
        thickness = ramp_info["thickness"]
        start_z = ramp_info["oz"] + thickness * 0.5 + radius
        ox = ox or ramp_info["ox"]
        oy = oy or ramp_info["oy"]
    else:
        release_h = float(ent.params.get("release_height", 0.0))
        start_z = max(start_z_default, release_h)

    body_xml = f"""    <body name="{ball.name}_body" pos="{ox:.4f} {oy:.4f} {start_z:.4f}">
      <joint name="{ball.name}_slide_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="{ball.name}_slide_z" type="slide" axis="0 0 1" damping="0"/>
      <geom name="{ball.name}" type="sphere" size="{radius:.4f}" rgba="{_rgba(color)}"/>
    </body>
"""
    info = {
        "ball_name": ball.name,
        "ox": ox, "oy": oy, "start_z": start_z, "radius": radius,
    }
    return body_xml, ball.name, info


def _circular_motion_xml(ent: Entity) -> tuple[str, str, dict]:
    """CircularMotion: a ball moving on a vertical circle of radius r,
    visualized as a thin hoop guide. Motion is prescribed:
    x(t) = r*cos(omega*t), z(t) = cz + r*sin(omega*t),
    with omega = (v/r)**alpha so T(r, v) = 2*pi*(r/v)**alpha
    (Newton: alpha=1 recovers T = 2*pi*r/v)."""
    ball: Body = ent.bodies["ball"]
    radius = float(ball.params.get("radius", 0.12))
    color = ball.params.get("color", "red")
    r_circle = float(ent.params.get("radius", 1.0))
    v = float(ent.params.get("v", 2.0))
    cz = float(ent.params.get("center_height", max(r_circle + 0.4, 2.0)))
    origin = ent.params.get("origin", [0.0, 0.0, 0.0])
    ox, oy, _ = float(origin[0]), float(origin[1]), float(origin[2])

    # draw a thin hoop as a ring of small box geoms (approximation)
    n_seg = 48
    hoop_xml = ""
    for i in range(n_seg):
        ang = 2 * np.pi * i / n_seg
        gx_i = ox + r_circle * np.cos(ang)
        gz_i = cz + r_circle * np.sin(ang)
        hoop_xml += (
            f'    <geom name="{ent.name}_hoop_{i}" type="sphere" '
            f'size="0.025" '
            f'pos="{gx_i:.4f} {oy:.4f} {gz_i:.4f}" '
            f'rgba="0.4 0.4 0.4 1"/>\n'
        )
    # center marker
    hoop_xml += (
        f'    <geom name="{ent.name}_center" type="sphere" size="0.04" '
        f'pos="{ox:.4f} {oy:.4f} {cz:.4f}" rgba="0.15 0.15 0.15 1"/>\n'
    )

    start_x = ox + r_circle  # angle=0 start
    start_z = cz
    body_xml = f"""    <body name="{ball.name}_body" pos="{start_x:.4f} {oy:.4f} {start_z:.4f}">
      <joint name="{ball.name}_slide_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="{ball.name}_slide_z" type="slide" axis="0 0 1" damping="0"/>
      <geom name="{ball.name}" type="sphere" size="{radius:.4f}" rgba="{_rgba(color)}"/>
    </body>
"""
    info = {
        "ent_name": ent.name, "ball_name": ball.name,
        "r_circle": r_circle, "v": v, "cz": cz,
        "ox": ox, "oy": oy, "radius": radius,
    }
    return hoop_xml, body_xml, info


def _spring_block_xml(ent: Entity) -> tuple[str, str, dict]:
    """SpringBlock: a mass sliding on a flat surface, connected to a
    wall by a visual spring. Motion is x(t) = A*cos(omega*t) with
    omega = (k/m)**alpha * sqrt unit — i.e. Newtonian period
    T_newton = 2*pi*sqrt(m/k) is only the alpha=0.5 case.
    """
    ball: Body = ent.bodies["ball"]
    radius = float(ball.params.get("radius", 0.12))
    color = ball.params.get("color", "blue")
    k = float(ent.params.get("k", 10.0))
    m = float(ent.params.get("mass", 1.0))
    A = float(ent.params.get("amplitude", 0.8))
    surface_h = float(ent.params.get("surface_height", 0.0))
    origin = ent.params.get("origin", [0.0, 0.0, 0.0])
    ox, oy, _ = float(origin[0]), float(origin[1]), float(origin[2])

    wall_h = max(0.6, A * 1.2)
    wall_hw = 0.06
    wall_x = ox - A * 1.4
    center_z = surface_h + radius + 0.02

    world_geom = (
        f'    <geom name="{ent.name}_wall" type="box" '
        f'size="{wall_hw:.4f} 0.2 {wall_h*0.5:.4f}" '
        f'pos="{wall_x:.4f} {oy:.4f} {surface_h + wall_h*0.5:.4f}" '
        f'rgba="0.35 0.25 0.2 1"/>\n'
        f'    <geom name="{ent.name}_rail" type="box" '
        f'size="{A*1.6:.4f} 0.05 0.02" '
        f'pos="{ox - 0.15:.4f} {oy:.4f} {surface_h - 0.02:.4f}" '
        f'rgba="0.55 0.55 0.55 1"/>\n'
    )
    body_xml = f"""    <body name="{ball.name}_body" pos="{ox:.4f} {oy:.4f} {center_z:.4f}">
      <joint name="{ball.name}_slide_x" type="slide" axis="1 0 0" damping="0"/>
      <geom name="{ball.name}" type="sphere" size="{radius:.4f}" rgba="{_rgba(color)}"/>
    </body>
"""
    info = {
        "ent_name": ent.name, "ball_name": ball.name,
        "k": k, "mass": m, "amplitude": A,
        "ox": ox, "oy": oy, "center_z": center_z,
        "wall_x": wall_x, "surface_h": surface_h,
        "radius": radius,
    }
    return world_geom, body_xml, info


def _horizontal_launch_xml(ent: Entity) -> tuple[str, str, dict]:
    ball: Body = ent.bodies["ball"]
    radius = float(ball.params.get("radius", 0.12))
    color = ball.params.get("color", "red")
    v0 = float(ent.params.get("v0", 1.0))
    h = float(ent.params.get("launch_height", 2.0))
    origin = ent.params.get("origin", [0.0, 0.0, 0.0])
    ox, oy = float(origin[0]), float(origin[1])

    body_xml = f"""    <body name="{ball.name}_body" pos="{ox:.4f} {oy:.4f} {h:.4f}">
      <joint name="{ball.name}_slide_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="{ball.name}_slide_z" type="slide" axis="0 0 1" damping="0"/>
      <geom name="{ball.name}" type="sphere" size="{radius:.4f}" rgba="{_rgba(color)}"/>
    </body>
"""
    info = {
        "ball_name": ball.name,
        "v0": v0, "launch_height": h,
        "ox": ox, "oy": oy, "radius": radius,
    }
    return body_xml, ball.name, info


# ---------- normalization ----------

def normalize_scene(scene: Scene) -> Scene:
    """Rewrite a scene in-place to make `InclinedRamp` a standalone
    motion source.

    Semantics: writing `- type: InclinedRamp` alone means "a ball
    rolls down this ramp from rest", matching Galileo's own setup
    and VLM expectations. Concretely this step:

      1. For each InclinedRamp that no explicit FreefallBall refers
         to via `on_ramp: <ramp>`, synthesize a FreefallBall entity
         with a new `ball` body placed on that ramp.
      2. Rewrite any StrobeTrail/ChimeTrack whose `target_body`
         equals a ramp's ENTITY name (not a body name) to point at
         the ball body that rolls on that ramp (the auto-injected
         one, or the explicit one if it exists).

    Idempotent: running this on an already-normalized scene is a
    no-op.
    """
    ramp_entity_names = [
        e.name for e in scene.entities.values()
        if e.kind == "InclinedRamp"
    ]
    if not ramp_entity_names:
        return scene

    # ramp_name -> ball body name that rides it (explicit first)
    ramp_to_ball: dict[str, str] = {}
    for e in scene.entities.values():
        if e.kind != "FreefallBall":
            continue
        on_ramp = str(e.params.get("on_ramp", "")) or None
        if on_ramp and on_ramp in ramp_entity_names:
            ball = e.bodies.get("ball")
            if ball is not None:
                ramp_to_ball[on_ramp] = ball.name

    # auto-inject for ramps with no explicit ball
    for ramp_name in ramp_entity_names:
        if ramp_name in ramp_to_ball:
            continue
        existing_ent = set(scene.entities.keys())
        existing_body = {
            b.name for e in scene.entities.values()
            for b in e.bodies.values()
        }
        auto_ent = f"_auto_ffb_{ramp_name}"
        auto_ball = f"_auto_ball_{ramp_name}"
        suf = 0
        while auto_ent in existing_ent or auto_ball in existing_body:
            suf += 1
            auto_ent = f"_auto_ffb_{ramp_name}_{suf}"
            auto_ball = f"_auto_ball_{ramp_name}_{suf}"
        scene.entities[auto_ent] = Entity(
            name=auto_ent, kind="FreefallBall",
            bodies={"ball": Body(
                kind="ball", name=auto_ball,
                params={"radius": 0.22, "color": "red"},
            )},
            params={"on_ramp": ramp_name,
                    "origin": [0.0, 0.0, 0.0],
                    "release_height": 0.0},
        )
        ramp_to_ball[ramp_name] = auto_ball

    # rewrite StrobeTrail/ChimeTrack target_body when it names a ramp
    for e in scene.entities.values():
        if e.kind not in ("StrobeTrail", "ChimeTrack"):
            continue
        tgt = str(e.params.get("target_body", ""))
        if tgt in ramp_to_ball:
            e.params["target_body"] = ramp_to_ball[tgt]

    return scene


# ---------- compile ----------

def compile_scene(scene: Scene, width: int = 640, height: int = 480) -> CompiledScene:
    scene = normalize_scene(scene)

    # Short-circuit: a scene whose only motion source is ScenarioBackend
    # delegates rendering to a Scenario.* class. We still emit a minimal
    # valid MJCF so Gate 2's MuJoCo compile check passes.
    sb_ents = [e for e in scene.entities.values() if e.kind == "ScenarioBackend"]
    has_dsl_motion = any(
        e.kind in {"FreefallBall", "HorizontalLaunch", "Pendulum",
                   "InclinedRamp", "SpringBlock", "CircularMotion"}
        for e in scene.entities.values()
    )
    if sb_ents and not has_dsl_motion:
        sb = sb_ents[0]
        stub_mjcf = (
            "<mujoco model=\"scenario_backend\">\n"
            "  <worldbody>\n"
            "    <camera name=\"scene_cam\" pos=\"0 -8 2\" "
            "xyaxes=\"1 0 0  0 0 1\"/>\n"
            "    <geom name=\"_sb_floor\" type=\"plane\" "
            "size=\"5 5 0.05\" rgba=\"0.6 0.6 0.6 1\"/>\n"
            "  </worldbody>\n"
            "</mujoco>\n"
        )
        return CompiledScene(
            mjcf=stub_mjcf,
            tracked_bodies=[],
            default_target="",
            metadata={
                "scenario_backend": {
                    "slug": str(sb.params.get("slug", "")).strip(),
                    "alpha": sb.params.get("alpha"),
                    "kwargs": dict(sb.params.get("kwargs", {})),
                    "clean": bool(sb.params.get("clean", False)),
                    # Pass entity objects (not just kinds) so the SB renderer
                    # can read params like cell_px on BackgroundGrid. Only
                    # 2D pixel-space overlays apply on SB scenes; mechanical
                    # 3D overlays (ReferenceScale, GridFloor) are no-ops here.
                    "overlay_entities": [
                        e for e in scene.entities.values()
                        if e.kind in {"BackgroundGrid"}
                    ],
                },
                "strobe_entities": [],
                "fading_trail_entities": [],
            },
        )

    geom_xml = ""
    body_xml = ""
    ramp_infos: dict[str, dict] = {}
    pendulum_infos: list[dict] = []
    spring_infos: list[dict] = []
    circle_infos: list[dict] = []
    tracked: list[TrackedBody] = []
    has_grid_floor = False
    grid_floor_ent: Optional[Entity] = None
    strobe_ents: list[Entity] = []
    # body_name -> {"x", "y", "z", "radius"} for MassStack target lookup
    body_anchors: dict[str, dict] = {}

    # pass 1: scan ramps first so FreefallBall can reference them
    for ent in scene.entities.values():
        if ent.kind == "InclinedRamp":
            gx, info = _inclined_ramp_xml(ent)
            geom_xml += gx
            ramp_infos[ent.name] = info

    for ent in scene.entities.values():
        if ent.kind == "InclinedRamp":
            continue
        if ent.kind == "ReferenceScale":
            geom_xml += _reference_scale_xml(ent)
        elif ent.kind == "SecondBall":
            geom_xml += _second_ball_xml(ent)
        elif ent.kind == "TimingGate":
            geom_xml += _timing_gate_xml(ent)
        elif ent.kind == "GridFloor":
            has_grid_floor = True
            grid_floor_ent = ent
        elif ent.kind == "FreefallBall":
            on_ramp = str(ent.params.get("on_ramp", "")) or None
            ramp_info = ramp_infos.get(on_ramp) if on_ramp else (
                next(iter(ramp_infos.values())) if ramp_infos else None
            )
            bx, ball_name, info = _freefall_ball_body_xml(ent, ramp_info)
            body_xml += bx
            body_anchors[ball_name] = {
                "x": info["ox"], "y": info["oy"],
                "z": info["start_z"], "radius": info["radius"],
            }
            if ramp_info is not None:
                tracked.append(TrackedBody(
                    name=ball_name, kind="ramp",
                    joints=[f"{ball_name}_slide_x", f"{ball_name}_slide_z"],
                    params={
                        "angle_deg": ramp_info["angle_deg"],
                        "length": ramp_info["length"],
                        "ox": ramp_info["ox"],
                        "oy": ramp_info["oy"],
                        "oz": ramp_info["oz"],
                        "width": ramp_info["width"],
                        "thickness": ramp_info["thickness"],
                        "start_z": info["start_z"],
                    },
                ))
            else:
                tracked.append(TrackedBody(
                    name=ball_name, kind="freefall_vertical",
                    joints=[f"{ball_name}_slide_x", f"{ball_name}_slide_z"],
                    params={
                        "start_z": info["start_z"],
                        "ox": info.get("ox", 0.0),
                        "oy": info.get("oy", 0.0),
                    },
                ))
        elif ent.kind == "HorizontalLaunch":
            bx, ball_name, info = _horizontal_launch_xml(ent)
            body_xml += bx
            body_anchors[ball_name] = {
                "x": info["ox"], "y": info["oy"],
                "z": info["launch_height"], "radius": info["radius"],
            }
            tracked.append(TrackedBody(
                name=ball_name, kind="horizontal_launch",
                joints=[f"{ball_name}_slide_x", f"{ball_name}_slide_z"],
                params={
                    "v0": info["v0"], "launch_height": info["launch_height"],
                    "ox": info["ox"],
                },
            ))
        elif ent.kind == "Pendulum":
            gx, bx, pinfo = _pendulum_xml(ent)
            geom_xml += gx
            body_xml += bx
            pendulum_infos.append(pinfo)
            body_anchors[pinfo["bob_name"]] = {
                "x": pinfo["ox"],
                "y": pinfo["oy"],
                "z": pinfo["oz"] + pinfo["pivot_height"] - pinfo["length"],
                "radius": pinfo["bob_radius"],
            }
            tracked.append(TrackedBody(
                name=pinfo["bob_name"], kind="pendulum",
                joints=[pinfo["hinge_name"]],
                params={
                    "length": pinfo["length"],
                    "theta_max_deg": pinfo["theta_max_deg"],
                    "n_samples": pinfo["n_samples"],
                },
            ))
        elif ent.kind == "CircularMotion":
            gx, bx, cinfo = _circular_motion_xml(ent)
            geom_xml += gx
            body_xml += bx
            circle_infos.append(cinfo)
            body_anchors[cinfo["ball_name"]] = {
                "x": cinfo["ox"] + cinfo["r_circle"],
                "y": cinfo["oy"],
                "z": cinfo["cz"],
                "radius": cinfo["radius"],
            }
            tracked.append(TrackedBody(
                name=cinfo["ball_name"], kind="circular",
                joints=[f"{cinfo['ball_name']}_slide_x",
                        f"{cinfo['ball_name']}_slide_z"],
                params={
                    "r_circle": cinfo["r_circle"], "v": cinfo["v"],
                    "cz": cinfo["cz"], "ox": cinfo["ox"],
                },
            ))
        elif ent.kind == "SpringBlock":
            gx, bx, sinfo = _spring_block_xml(ent)
            geom_xml += gx
            body_xml += bx
            spring_infos.append(sinfo)
            body_anchors[sinfo["ball_name"]] = {
                "x": sinfo["ox"], "y": sinfo["oy"],
                "z": sinfo["center_z"], "radius": sinfo["radius"],
            }
            tracked.append(TrackedBody(
                name=sinfo["ball_name"], kind="spring",
                joints=[f"{sinfo['ball_name']}_slide_x"],
                params={
                    "k": sinfo["k"], "mass": sinfo["mass"],
                    "amplitude": sinfo["amplitude"],
                    "ox": sinfo["ox"], "center_z": sinfo["center_z"],
                },
            ))
        elif ent.kind == "StrobeTrail":
            strobe_ents.append(ent)

    if not tracked:
        raise ValueError(
            "compile_scene requires at least one motion source "
            "(FreefallBall, HorizontalLaunch, Pendulum, SpringBlock, "
            "or CircularMotion)."
        )

    # default strobe target: first explicit StrobeTrail's target_body, else first tracked
    default_target = tracked[0].name
    tracked_names = {t.name for t in tracked}
    for s in strobe_ents:
        tgt = s.params.get("target_body")
        if tgt in tracked_names:
            default_target = tgt
            break

    # ---- append bell geoms for every ChimeTrack ----
    chime_bell_info: list[dict] = []  # per-bell: {ent, i, x, y, z, size}
    for ent in scene.entities.values():
        if ent.kind != "ChimeTrack":
            continue
        tgt_name = ent.params.get("target_body")
        tb = next((t for t in tracked if t.name == tgt_name), None)
        if tb is None:
            continue
        positions = [float(p) for p in ent.params.get("positions", [])]
        bell_xml, bells = _chime_bells_xml(ent, tb, positions)
        geom_xml += bell_xml
        chime_bell_info.extend(bells)

    # ---- append unit-cube stacks for every MassStack ----
    for ent in scene.entities.values():
        if ent.kind != "MassStack":
            continue
        tgt_name = ent.params.get("target_body")
        anc = body_anchors.get(tgt_name)
        if anc is None:
            continue
        geom_xml += _mass_stack_xml(
            ent, (anc["x"], anc["y"], anc["z"]), float(anc["radius"]),
        )

    # camera & floor framing: bbox over ramps + pendulums + tracked starts
    xs, zs = [], []
    for ri in ramp_infos.values():
        xs.extend([ri["ox"], ri["ox"] + ri["length"] * ri["cos_t"]])
        zs.extend([ri["oz"], ri["oz"] - ri["length"] * ri["sin_t"]])
    for pi in pendulum_infos:
        xs.extend([pi["ox"] - pi["length"], pi["ox"] + pi["length"]])
        zs.extend([pi["oz"], pi["oz"] + pi["pivot_height"]])
    for ci in circle_infos:
        r = ci["r_circle"]
        xs.extend([ci["ox"] - r, ci["ox"] + r])
        zs.extend([ci["cz"] - r, ci["cz"] + r])
    for si in spring_infos:
        A = si["amplitude"]
        xs.extend([si["wall_x"] - 0.3, si["ox"] + A * 1.6])
        zs.extend([si["surface_h"], si["center_z"] + 0.5])
    for tb in tracked:
        if tb.kind == "horizontal_launch":
            xs.extend([tb.params["ox"], tb.params["ox"] + 4.0])
            zs.extend([0.0, tb.params["launch_height"]])
        elif tb.kind == "freefall_vertical":
            zs.extend([0.0, tb.params["start_z"]])
    if not xs:
        xs = [0.0, 1.0]
    if not zs:
        zs = [0.0, 1.0]
    x_min, x_max = min(xs) - 0.5, max(xs) + 0.5
    z_min, z_max = min(zs) - 0.3, max(zs) + 0.3
    span = max(x_max - x_min, z_max - z_min, 2.0)
    cam_x = 0.5 * (x_min + x_max)
    cam_z = 0.5 * (z_min + z_max)
    cam_dist = span * 1.5 + 1.5
    cam_y = -cam_dist

    # user override
    default_cam = scene.camera.position == (0.0, -8.0, 2.0)
    if not default_cam:
        cam_x, cam_y, cam_z = (float(v) for v in scene.camera.position)
        cam_dist = max(abs(cam_y), 1.0) + 1.0

    floor_z = z_min - 0.1
    floor_size = max(4.0, span * 1.2)

    if has_grid_floor:
        grid_xml = _grid_floor_xml(grid_floor_ent, floor_z)
        geom_xml = grid_xml + geom_xml
        floor_material = "floor_plain_mat"
    else:
        floor_material = "floor_mat"

    mjcf = f"""<mujoco model="{scene.name}">
  <option gravity="0 0 -9.81" timestep="0.001"/>
  <visual>
    <global offwidth="{width}" offheight="{height}"/>
    <map znear="0.01" zfar="{cam_dist * 4:.0f}"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.85 0.85 0.85" rgb2="0.75 0.75 0.75"
             width="256" height="256"/>
    <material name="floor_mat" texture="grid" texrepeat="6 6" reflectance="0.0"/>
    <material name="floor_plain_mat" rgba="0.90 0.90 0.88 1" reflectance="0.0"/>
    <material name="ramp_mat" rgba="0.62 0.45 0.30 1" specular="0.25" shininess="0.3"/>
  </asset>
  <worldbody>
    <light pos="2 {-cam_dist*0.3:.2f} {cam_dist*0.8+2:.2f}"
           dir="-0.2 0.5 -0.8" diffuse="0.95 0.95 0.95" castshadow="false"/>
    <light pos="-2 {-cam_dist*0.2:.2f} {cam_dist*0.5+2:.2f}"
           diffuse="0.45 0.48 0.55" castshadow="false"/>
    <geom type="plane" size="{floor_size:.2f} {floor_size:.2f} 0.05"
          pos="{cam_x:.3f} 0 {floor_z:.4f}" material="{floor_material}"/>
{geom_xml}{body_xml}
    <camera name="scene_cam" pos="{cam_x:.3f} {cam_y:.3f} {cam_z:.3f}"
            xyaxes="1 0 0 0 0 1" mode="fixed"/>
  </worldbody>
</mujoco>
"""

    return CompiledScene(
        mjcf=mjcf,
        tracked_bodies=tracked,
        default_target=default_target,
        metadata={
            "ramp_infos": ramp_infos,
            "pendulum_infos": pendulum_infos,
            "spring_infos": spring_infos,
            "circle_infos": circle_infos,
            "strobe_entities": strobe_ents,
            "timing_gates": [
                e for e in scene.entities.values() if e.kind == "TimingGate"
            ],
            "chime_bells": chime_bell_info,
            "chime_entities": [
                e for e in scene.entities.values() if e.kind == "ChimeTrack"
            ],
            "fading_trail_entities": [
                e for e in scene.entities.values() if e.kind == "FadingTrail"
            ],
            "protractor_entities": [
                e for e in scene.entities.values() if e.kind == "AngleProtractor"
            ],
            "filmstrip_entities": [
                e for e in scene.entities.values() if e.kind == "Filmstrip"
            ],
            "animation_entities": [
                e for e in scene.entities.values() if e.kind == "Animation"
            ],
            "background_grid_entities": [
                e for e in scene.entities.values() if e.kind == "BackgroundGrid"
            ],
            "light_strip_entities": [
                e for e in scene.entities.values() if e.kind == "LightStrip"
            ],
            "spectrum_band_entities": [
                e for e in scene.entities.values() if e.kind == "SpectrumBand"
            ],
            "body_anchors": body_anchors,
            "camera": {"x": cam_x, "y": cam_y, "z": cam_z},
        },
    )


# ---------- motion recipes ----------

def _trajectory_for(tb: TrackedBody, times: np.ndarray, cv: float,
                    alpha: float) -> list[tuple[str, float]]:
    """Return list of (joint_name, qpos) updates per time sample.

    Each kind's hidden law:
      ramp:               s = 0.5 * g * sin(theta) * t^alpha, capped by length
      freefall_vertical:  z_drop = 0.5 * g * t^alpha, capped by start_z
      horizontal_launch:  dx = v0 * t, dz_drop = 0.5 * g * t^alpha
      pendulum:           angle = theta_max * cos(2*pi/T * t),
                          T = 2*pi*(L/g)^alpha, control=L
    Returns a LIST of (joint, qpos) tuples, one per joint per sample,
    but this function returns only ONE sample's worth (the list is over
    joints, not over time). Caller iterates over time.
    """
    raise NotImplementedError


def _apply_tracked_motion(
    model: mujoco.MjModel, data: mujoco.MjData,
    tb: TrackedBody, t: float, cv: float, alpha: float,
) -> tuple[float, float, float]:
    """Drive one tracked body's qpos at time t; return its (x, y, z)
    world position after mj_forward (used by timing-gate detection)."""
    p = tb.params
    if tb.kind == "ramp":
        angle = np.radians(p["angle_deg"])
        cap = p["length"] * 0.995
        s = min(0.5 * G * _ramp_sin_coef(angle) * (t ** alpha), cap)
        dx = s * np.cos(angle)
        dz = -s * np.sin(angle)
        _set_joint(model, data, f"{tb.name}_slide_x", dx)
        _set_joint(model, data, f"{tb.name}_slide_z", dz)
        return (p["ox"] + dx, 0.0, p["start_z"] + dz)
    if tb.kind == "freefall_vertical":
        drop = 0.5 * _g_scale() * G * (t ** alpha)
        drop = min(drop, p["start_z"] - 0.05)
        _set_joint(model, data, f"{tb.name}_slide_x", 0.0)
        _set_joint(model, data, f"{tb.name}_slide_z", -drop)
        return (0.0, 0.0, p["start_z"] - drop)
    if tb.kind == "horizontal_launch":
        dx = p["v0"] * t
        drop = 0.5 * _g_scale() * G * (t ** alpha)
        drop = min(drop, p["launch_height"] - 0.05)
        _set_joint(model, data, f"{tb.name}_slide_x", dx)
        _set_joint(model, data, f"{tb.name}_slide_z", -drop)
        return (p["ox"] + dx, 0.0, p["launch_height"] - drop)
    if tb.kind == "pendulum":
        L = max(float(cv), 0.05)  # control is string length
        period = 2 * np.pi * (L / G) ** alpha
        omega = 2 * np.pi / period if period > 1e-6 else 0.0
        theta_max = np.radians(p["theta_max_deg"])
        th = theta_max * np.cos(omega * t)
        _set_joint(model, data, tb.joints[0], float(th))
        return (0.0, 0.0, 0.0)
    if tb.kind == "circular":
        r = max(float(p["r_circle"]), 1e-6)
        v = max(float(p["v"]), 1e-6)
        cz = float(p["cz"])
        ox = float(p["ox"])
        omega = (v / r) ** float(alpha)
        dx = r * np.cos(omega * t) - r
        dz = r * np.sin(omega * t)
        _set_joint(model, data, f"{tb.name}_slide_x", float(dx))
        _set_joint(model, data, f"{tb.name}_slide_z", float(dz))
        return (ox + r + dx, 0.0, cz + dz)
    if tb.kind == "spring":
        k = max(float(p["k"]), 1e-6)
        m = max(float(p["mass"]), 1e-6)
        A = float(p["amplitude"])
        period = 2 * np.pi * (m / k) ** alpha
        omega = 2 * np.pi / period if period > 1e-6 else 0.0
        dx = A * np.cos(omega * t)
        _set_joint(model, data, tb.joints[0], float(dx))
        return (p["ox"] + dx, 0.0, p["center_z"])
    raise ValueError(f"Unknown tracked kind {tb.kind!r}")


def _set_joint(model: mujoco.MjModel, data: mujoco.MjData,
               joint_name: str, value: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    adr = model.jnt_qposadr[jid]
    data.qpos[adr] = value


# Global strobe-rate mode. When a runner sets this to a positive
# float (e.g. 4.0 => 4 strobes per second of simulated motion), the
# StrobeTrail's n_samples is OVERRIDDEN: instead, the number of
# strobes becomes clamp(round(rate * t_end), 3, 20). This models a
# real flash-photography setup: the strobe frequency is global and
# fixed, so slower motions (ramps, pendulums) accumulate more
# exposures than fast falls. Leave as None to keep the legacy
# behaviour (n_samples directly taken from the DSL).
STROBE_RATE: Optional[float] = None

# Override the effective gravitational scale applied to freefall
# and horizontal-launch motion. When set, y_drop uses
# 0.5 * G_SCALE * G * t^alpha, allowing per-scene mass scaling
# such as G_SCALE = 1 / mass ** beta in S22-style F=m*a tests.
# Leave as None for the legacy behaviour.
G_SCALE_OVERRIDE: Optional[float] = None


def _g_scale() -> float:
    return 1.0 if G_SCALE_OVERRIDE is None else float(G_SCALE_OVERRIDE)


# Override gamma for ramp motion. When set, ramp acceleration
# becomes 0.5 * G * sin(angle)^gamma * t^alpha (instead of the
# legacy 0.5 * G * sin(angle) * t^alpha). Used by scenarios that
# treat sin(angle) dependence as a second counterfactual exponent.
# Leave None for the legacy behaviour.
GAMMA_OVERRIDE: Optional[float] = None


def _ramp_sin_coef(angle_rad: float) -> float:
    """Return sin(angle)^gamma for ramp motion. Gamma=1 unless
    GAMMA_OVERRIDE is set."""
    g = 1.0 if GAMMA_OVERRIDE is None else float(GAMMA_OVERRIDE)
    return float(np.sin(angle_rad)) ** g


def _choose_times(compiled: CompiledScene, target: str, cv: float,
                  n_strobes: int, alpha: float = 2.0,
                  ) -> tuple[np.ndarray, int]:
    """Pick the time axis for strobe sampling.

    We cap the time axis at min(cv, t_land) where t_land is the time
    when the tracked body first hits its physical limit (floor for
    freefall/launch, end of ramp for ramp). This avoids piling many
    strobes onto a single clamped-at-floor position.

    Returns (times, n_strobes_effective). When STROBE_RATE is set,
    n_strobes_effective is derived from the global rate and the
    actual duration; otherwise it equals the input n_strobes.
    """
    target_tb = next(t for t in compiled.tracked_bodies if t.name == target)
    p = target_tb.params
    a = max(float(alpha), 1e-3)

    if target_tb.kind == "pendulum":
        L = max(float(cv), 0.05)
        # alpha-independent time window; VLM sees how many swings
        # happen within this window and must infer the true period
        # (which depends on alpha via T = 2*pi*(L/G)**alpha).
        period_newton = 2 * np.pi * (L / G) ** 0.5
        t_end = float(3.0 * period_newton)
    elif target_tb.kind == "circular":
        r = max(float(p["r_circle"]), 1e-6)
        v = max(float(p["v"]), 1e-6)
        period_newton = 2 * np.pi * r / v
        t_end = float(3.0 * period_newton)
    elif target_tb.kind == "spring":
        k = max(float(p["k"]), 1e-6)
        m = max(float(p["mass"]), 1e-6)
        period_newton = 2 * np.pi * (m / k) ** 0.5
        t_end = float(3.0 * period_newton)
    else:
        if target_tb.kind == "freefall_vertical":
            h = max(p["start_z"] - 0.05, 0.05)
            t_land = (2 * h / (_g_scale() * G)) ** (1.0 / a)
        elif target_tb.kind == "horizontal_launch":
            h = max(p["launch_height"] - 0.05, 0.05)
            t_land = (2 * h / (_g_scale() * G)) ** (1.0 / a)
        elif target_tb.kind == "ramp":
            angle = np.radians(p["angle_deg"])
            cap = p["length"] * 0.995
            denom = 0.5 * G * _ramp_sin_coef(angle)
            t_land = (cap / denom) ** (1.0 / a) if denom > 1e-6 else float("inf")
        else:
            t_land = float("inf")
        t_end = min(max(float(cv), 0.05), float(t_land))

    if STROBE_RATE is not None and STROBE_RATE > 0:
        n_eff = int(round(STROBE_RATE * t_end))
        n_eff = int(np.clip(n_eff, 3, 20))
    else:
        n_eff = n_strobes
    return np.linspace(0.0, t_end, n_eff), n_eff


MUJOCO_DEFAULT_FOVY_DEG = 45.0


def _project_point(
    cam_x: float, cam_y: float, cam_z: float,
    width: int, height: int,
    X: float, Y: float, Z: float,
    fovy_deg: float = MUJOCO_DEFAULT_FOVY_DEG,
) -> Optional[tuple[float, float, float]]:
    """Project a world point to pixel coords for our fixed camera.

    Our camera has pos=(cam_x, cam_y, cam_z), xyaxes="1 0 0  0 0 1",
    so it looks along +Y with +X to the right and +Z up. Returns
    (u, v, depth) in pixel space, or None if the point is behind the
    camera. The image origin is top-left.
    """
    depth = Y - cam_y
    if depth <= 1e-4:
        return None
    f = height / (2.0 * np.tan(np.radians(fovy_deg) * 0.5))
    u = width * 0.5 + f * (X - cam_x) / depth
    v = height * 0.5 - f * (Z - cam_z) / depth
    return float(u), float(v), float(depth)


def _text_with_outline(
    draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
    font, fill=(255, 255, 255), outline=(0, 0, 0),
) -> None:
    x, y = xy
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            if ox == 0 and oy == 0:
                continue
            draw.text((x + ox, y + oy), text, fill=outline, font=font)
    draw.text((x, y), text, fill=fill, font=font)


def _nice_tick(extent: float) -> float:
    """Pick a tick spacing so we get ~5-8 labelled major ticks across."""
    target = extent / 6.0
    for step in (0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0):
        if step >= target:
            return step
    return 50.0


def _strobe_distances_for(tb: "TrackedBody", times: np.ndarray,
                          cv: float, alpha: float) -> np.ndarray:
    """For each strobe time, compute distance-along-motion-axis (meters).

    - freefall_vertical: distance fallen from release point.
    - ramp: distance along ramp from top.
    - horizontal_launch: returns 2D (dx, dz_drop) per sample.
    - pendulum: returns angle in degrees.
    """
    a = max(float(alpha), 1e-3)
    p = tb.params
    if tb.kind == "freefall_vertical":
        start_z = float(p.get("start_z", 2.0))
        ds = 0.5 * _g_scale() * G * np.power(np.maximum(times, 0.0), a)
        ds = np.minimum(ds, start_z - 0.05)
        return ds  # distance fallen
    if tb.kind == "ramp":
        length = float(p.get("length", 4.0))
        angle_deg = float(p.get("angle_deg", 20.0))
        theta = np.radians(angle_deg)
        s = 0.5 * G * _ramp_sin_coef(theta) * np.power(np.maximum(times, 0.0), a)
        return np.minimum(s, length * 0.995)
    if tb.kind == "horizontal_launch":
        v0 = float(p.get("v0", 1.0))
        launch_h = float(p.get("launch_height", 2.0))
        dx = v0 * times
        dz = 0.5 * _g_scale() * G * np.power(np.maximum(times, 0.0), a)
        dz = np.minimum(dz, launch_h - 0.05)
        return np.stack([dx, dz], axis=1)  # (N, 2)
    if tb.kind == "pendulum":
        length = float(p.get("length", 1.0))
        theta_max = float(p.get("theta_max_deg", 17.0))
        T = 2 * np.pi * np.power(length / G, a)
        omega = 2 * np.pi / max(T, 1e-3)
        return theta_max * np.cos(omega * times)
    if tb.kind == "circular":
        r = max(float(p.get("r_circle", 1.0)), 1e-6)
        v = max(float(p.get("v", 1.0)), 1e-6)
        omega = (v / r) ** a
        return np.degrees(omega * times)
    if tb.kind == "spring":
        k = max(float(p.get("k", 10.0)), 1e-6)
        m = max(float(p.get("mass", 1.0)), 1e-6)
        A = float(p.get("amplitude", 0.8))
        T = 2 * np.pi * np.power(m / k, a)
        omega = 2 * np.pi / max(T, 1e-3)
        return A * np.cos(omega * times)
    return np.zeros_like(times)


def _arrival_time(tb: "TrackedBody", distance: float, alpha: float) -> Optional[float]:
    """Solve for the time at which the ball reaches `distance` along
    its motion axis. Returns None if the ball never reaches it.

    - freefall_vertical: distance = 0.5 * G * t^alpha  (distance fallen)
    - ramp:              distance = 0.5 * G * sin(theta) * t^alpha
    - horizontal_launch: distance = v0 * t (horizontal distance only;
                          arrival is well-defined regardless of alpha
                          since the horizontal motion is uniform)
    Returns t in seconds, or None.
    """
    a = max(float(alpha), 1e-3)
    p = tb.params
    if tb.kind == "freefall_vertical":
        start_z = float(p.get("start_z", 2.0))
        if distance > start_z:
            return None
        coef = 0.5 * _g_scale() * G
        if coef <= 0:
            return None
        return (distance / coef) ** (1.0 / a)
    if tb.kind == "ramp":
        length = float(p.get("length", 4.0))
        if distance > length:
            return None
        angle = np.radians(float(p.get("angle_deg", 20.0)))
        coef = 0.5 * G * _ramp_sin_coef(angle)
        if coef <= 0:
            return None
        return (distance / coef) ** (1.0 / a)
    if tb.kind == "horizontal_launch":
        v0 = float(p.get("v0", 1.0))
        if v0 <= 0:
            return None
        # ball lands when 0.5*G*t^a = launch_height -> t_land
        h = float(p.get("launch_height", 2.0))
        t_land = (h / (0.5 * _g_scale() * G)) ** (1.0 / a)
        x_land = v0 * t_land
        if distance > x_land:
            return None
        return distance / v0
    return None


def _draw_chime_overlays(
    draw: ImageDraw.ImageDraw,
    compiled: "CompiledScene",
    width: int, height: int,
    font,
    target: str,
    alpha: float,
    control_value: float,
    composite: Optional[Image.Image] = None,
) -> None:
    """Draw bell index labels next to each MJCF bell, plus a beat
    timeline on the right side of the image where each bell's arrival
    is highlighted.

    Time is shown in arbitrary BEAT units. Distance is shown in
    arbitrary GRID units (the numbers the user wrote in `positions`).
    No real-world seconds or meters appear anywhere on the image.
    """
    bells = compiled.metadata.get("chime_bells") or []
    chime_ents = compiled.metadata.get("chime_entities") or []
    if not bells or not chime_ents:
        return

    cam = compiled.metadata.get("camera") or {}
    cam_x = float(cam.get("x", 0.0))
    cam_y = float(cam.get("y", -8.0))
    cam_z = float(cam.get("z", 2.0))

    # group bells by ChimeTrack entity (matched via target_name); we
    # render one timeline per track, stacked vertically on the right.
    tracks: dict[str, list[dict]] = {}
    for b in bells:
        tracks.setdefault(b["target_name"], []).append(b)

    # ---- 1. label each bell with its index in image space ----
    # Bell labels go to the LEFT of the apparatus column so they don't
    # collide with strobe indices (which sit on the right of the ball).
    for b in bells:
        q = _project_point(cam_x, cam_y, cam_z, width, height,
                           b["x"], b["y"], b["z"])
        if q is None:
            continue
        u, v, _ = q
        label = f"B{b['index']}"
        _text_with_outline(draw, (u - 32, v - 8), label, font,
                           fill=(255, 220, 80))

    # ---- 2. compute arrival times per track, then build timelines ----
    track_arrivals: dict[str, list[tuple[int, Optional[float]]]] = {}
    for tname, tbells in tracks.items():
        tb = next(
            (t for t in compiled.tracked_bodies if t.name == tname), None
        )
        if tb is None:
            continue
        rows: list[tuple[int, Optional[float]]] = []
        for b in tbells:
            t = _arrival_time(tb, b["distance"], alpha)
            rows.append((b["index"], t))
        track_arrivals[tname] = rows

    # decide one shared time scale from the largest arrival across all
    # tracks so the metronome unit stays consistent within the image
    valid_t = [t for rows in track_arrivals.values()
               for _, t in rows if t is not None]
    if not valid_t:
        return
    t_max = max(valid_t) * 1.05  # padding so the last tick doesn't sit on edge

    # number of metronome subdivisions (default 40 unless any chime
    # entity overrides)
    n_ticks = 40
    for ent in chime_ents:
        try:
            n_ticks = int(ent.params.get("metronome_ticks", n_ticks))
        except Exception:
            pass
    n_ticks = int(np.clip(n_ticks, 5, 120))

    # ---- 3. draw a vertical metronome timeline on the right side ----
    # Layout (right-to-left):
    #   [bell-label gutter | timeline axis | tick gutter | next track ...]
    _sc_ch = max(height / 480.0, 1.0)
    label_gutter = int(round(36 * _sc_ch))
    tick_gutter = int(round(18 * _sc_ch))
    timeline_w_per_track = label_gutter + tick_gutter
    timeline_top = 30
    timeline_bottom = height - 30
    n_tracks = len(track_arrivals)

    panel_w = timeline_w_per_track * n_tracks + int(round(48 * _sc_ch))
    panel_x = width - panel_w
    if composite is not None:
        bg = Image.new("RGBA", (panel_w, height), (15, 15, 15, 150))
        composite.paste(bg, (panel_x, 0), bg)

    _text_with_outline(
        draw, (panel_x + int(round(6 * _sc_ch)),
               int(round(6 * _sc_ch))),
        "metronome", font,
        fill=(220, 220, 220),
    )

    for ti, (tname, rows) in enumerate(track_arrivals.items()):
        # rightmost track sits at the rightmost column
        col_right = width - 8 - ti * timeline_w_per_track
        col_x = col_right - label_gutter
        # vertical axis
        draw.line([(col_x, timeline_top), (col_x, timeline_bottom)],
                  fill=(180, 180, 180), width=2)
        # metronome ticks (uniform, no numeric labels). Major every 5.
        for k in range(n_ticks + 1):
            v = timeline_top + (timeline_bottom - timeline_top) * (k / n_ticks)
            major = (k % 5 == 0)
            tick_w = 6 if major else 3
            color = (210, 210, 210) if major else (130, 130, 130)
            draw.line([(col_x - tick_w, v), (col_x + tick_w, v)],
                      fill=color, width=1)
        # bell arrivals -> highlighted ticks with bell index label
        for idx, t in rows:
            if t is None:
                continue
            frac = t / t_max
            if frac < 0 or frac > 1:
                continue
            v = timeline_top + (timeline_bottom - timeline_top) * frac
            draw.line([(col_x - 11, v), (col_x + 11, v)],
                      fill=(255, 220, 80), width=3)
            _text_with_outline(
                draw, (col_x + 13, v - 8), f"B{idx}", font,
                fill=(255, 220, 80),
            )
        # column header: target name
        _text_with_outline(
            draw, (col_x - 24, timeline_bottom + 4), tname[:9], font,
            fill=(200, 200, 200),
        )


def _draw_scale_overlays(
    draw: ImageDraw.ImageDraw,
    compiled: "CompiledScene",
    width: int, height: int,
    font,
    times: np.ndarray,
    target: str,
    alpha: float,
    control_value: float,
    composite: Optional[Image.Image] = None,
) -> None:
    """Overlay numbered rulers + per-strobe guide lines in world units.

    For the TARGET tracked body we also draw guide lines from each
    strobe's (known) world position out to the ruler, labelling the
    exact meter reading at the intersection, plus a small legend
    listing (strobe_index, time) so the VLM can form (t, s) pairs.
    Non-target tracked bodies still get a ruler, without guide lines.
    """
    cam = compiled.metadata.get("camera") or {}
    cam_x = float(cam.get("x", 0.0))
    cam_y = float(cam.get("y", -8.0))
    cam_z = float(cam.get("z", 2.0))

    def project(X, Y, Z):
        return _project_point(cam_x, cam_y, cam_z, width, height, X, Y, Z)

    def _draw_line(p0, p1, color=(255, 220, 60), w=2):
        if p0 is None or p1 is None:
            return
        draw.line([(p0[0], p0[1]), (p1[0], p1[1])], fill=color, width=w)

    def _draw_ruler_vertical(rx, ry, z_min, z_max, d_at_z,
                             label_fmt="{d:.2f} m",
                             color=(210, 235, 255),
                             text_offset_x=-55,
                             ):
        """Vertical ruler along z axis at world (rx, ry).

        d_at_z(z) -> the "distance" value labelled on the ruler at
        world height z (may differ from z, e.g. for freefall we want
        distance-fallen, not world z).
        """
        step = _nice_tick(z_max - z_min)
        minor = step / 5.0
        p_top = project(rx, ry, z_max)
        p_bot = project(rx, ry, z_min)
        _draw_line(p_top, p_bot, color=color, w=3)
        # minor ticks
        z = z_min
        while z <= z_max + 1e-6:
            q = project(rx, ry, z)
            if q is not None:
                u, v, _ = q
                draw.line([(u - 4, v), (u + 4, v)], fill=color, width=1)
            z += minor
        # major ticks with labels
        z = z_min
        while z <= z_max + 1e-6:
            q = project(rx, ry, z)
            if q is not None:
                u, v, _ = q
                draw.line([(u - 10, v), (u + 10, v)], fill=color, width=3)
                _text_with_outline(
                    draw, (u + text_offset_x, v - 8),
                    label_fmt.format(d=d_at_z(z)), font,
                    fill=color,
                )
            z += step

    def _draw_ruler_horizontal(rx_a, rx_b, ry, rz, d_at_x,
                               label_fmt="{d:.2f}",
                               color=(255, 210, 140)):
        """Horizontal ruler along x from rx_a to rx_b at world y=ry, z=rz."""
        extent = abs(rx_b - rx_a)
        step = _nice_tick(extent)
        minor = step / 5.0
        p0 = project(rx_a, ry, rz)
        p1 = project(rx_b, ry, rz)
        _draw_line(p0, p1, color=color, w=3)
        x = rx_a
        while x <= rx_b + 1e-6:
            q = project(x, ry, rz)
            if q is not None:
                u, v, _ = q
                draw.line([(u, v - 4), (u, v + 4)], fill=color, width=1)
            x += minor
        x = rx_a
        while x <= rx_b + 1e-6:
            q = project(x, ry, rz)
            if q is not None:
                u, v, _ = q
                draw.line([(u, v - 10), (u, v + 10)], fill=color, width=3)
                _text_with_outline(
                    draw, (u - 12, v + 10),
                    label_fmt.format(d=d_at_x(x)), font,
                    fill=color,
                )
            x += step

    target_tb = next(
        (tb for tb in compiled.tracked_bodies if tb.name == target), None
    )
    target_ds = (
        _strobe_distances_for(target_tb, times, control_value, alpha)
        if target_tb is not None else None
    )

    for tb in compiled.tracked_bodies:
        p = tb.params
        is_target = tb.name == target

        if tb.kind == "freefall_vertical":
            start_z = float(p.get("start_z", 2.0))
            ox_w = float(p.get("ox", 0.0))
            oy_w = float(p.get("oy", 0.0))
            rx = ox_w + 0.7  # ruler offset to the side of the ball
            ry = oy_w
            _draw_ruler_vertical(
                rx, ry, 0.0, start_z,
                d_at_z=lambda z, s=start_z: s - z,
                label_fmt="{d:.1f} m",
            )
            # per-strobe guide lines + exact readings
            if is_target and target_ds is not None:
                for k, d in enumerate(target_ds):
                    d = float(d)
                    z_world = start_z - d
                    # line from ball world-X back to ruler (both at oy_w)
                    p_ball = project(ox_w, ry, z_world)
                    p_ruler = project(rx, ry, z_world)
                    if p_ball is None or p_ruler is None:
                        continue
                    # dashed-ish thin guide
                    draw.line([(p_ball[0], p_ball[1]),
                               (p_ruler[0], p_ruler[1])],
                              fill=(120, 180, 230), width=1)
                    # exact reading just above the ruler intersection
                    _text_with_outline(
                        draw,
                        (p_ruler[0] + 60, p_ruler[1] - 8),
                        f"[{k+1}] {d:.2f} m", font,
                        fill=(255, 235, 120),
                    )

        elif tb.kind == "horizontal_launch":
            launch_h = float(p.get("launch_height", 2.0))
            ox_w = float(p.get("ox", 0.0))
            v0 = float(p.get("v0", 1.0))
            # vertical ruler at launch x, labelled as "drop from launch"
            _draw_ruler_vertical(
                ox_w - 0.35, 0.0, 0.0, launch_h,
                d_at_z=lambda z, h=launch_h: h - z,
                label_fmt="{d:.1f} m",
                text_offset_x=-65,
            )
            # horizontal extent for floor ruler
            t_fall = (2.0 * max(launch_h - 0.05, 0.05) / G) ** 0.5
            extent = max(2.0, v0 * t_fall * 1.2)
            _draw_ruler_horizontal(
                ox_w, ox_w + extent, 0.0, 0.02,
                d_at_x=lambda x, o=ox_w: x - o,
                label_fmt="{d:.1f} m",
                color=(255, 210, 140),
            )
            if is_target and target_ds is not None and target_ds.ndim == 2:
                for k, (dx, dz) in enumerate(target_ds):
                    x_world = ox_w + float(dx)
                    z_world = launch_h - float(dz)
                    # horizontal guide from ball to vertical tower ruler
                    p_ball = project(x_world, 0.0, z_world)
                    p_tower = project(ox_w - 0.35, 0.0, z_world)
                    if p_ball is not None and p_tower is not None:
                        draw.line([(p_ball[0], p_ball[1]),
                                   (p_tower[0], p_tower[1])],
                                  fill=(120, 180, 230), width=1)
                    # vertical guide from ball down to floor ruler
                    p_floor = project(x_world, 0.0, 0.02)
                    if p_ball is not None and p_floor is not None:
                        draw.line([(p_ball[0], p_ball[1]),
                                   (p_floor[0], p_floor[1])],
                                  fill=(230, 180, 120), width=1)

        elif tb.kind == "ramp":
            length = float(p.get("length", 4.0))
            angle_deg = float(p.get("angle_deg", 20.0))
            ox_w = float(p.get("ox", 0.0))
            oy_w = float(p.get("oy", 0.0))
            oz_w = float(p.get("oz", 0.0))
            theta = np.radians(angle_deg)
            cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
            thickness = float(p.get("thickness", 0.08))
            surf_n_x = sin_t
            surf_n_z = cos_t
            y_side = oy_w + (float(p.get("width", 0.8)) * 0.5 + 0.12)
            lift = thickness * 0.5 + 0.02

            step = _nice_tick(length)
            minor = step / 5.0
            color = (255, 235, 120)
            p_start = project(
                ox_w + surf_n_x * lift, y_side,
                oz_w + surf_n_z * lift,
            )
            p_end = project(
                ox_w + length * cos_t + surf_n_x * lift, y_side,
                oz_w - length * sin_t + surf_n_z * lift,
            )
            _draw_line(p_start, p_end, color=color, w=3)
            # minor ticks (short)
            d = 0.0
            while d <= length + 1e-6:
                x = ox_w + d * cos_t + surf_n_x * lift
                z = oz_w - d * sin_t + surf_n_z * lift
                q = project(x, y_side, z)
                if q is not None:
                    u, v, _ = q
                    draw.line([(u, v - 4), (u, v + 4)],
                              fill=color, width=1)
                d += minor
            # major ticks + labels
            d = 0.0
            while d <= length + 1e-6:
                x = ox_w + d * cos_t + surf_n_x * lift
                z = oz_w - d * sin_t + surf_n_z * lift
                q = project(x, y_side, z)
                if q is not None:
                    u, v, _ = q
                    draw.line([(u, v - 10), (u, v + 10)],
                              fill=color, width=3)
                    _text_with_outline(
                        draw, (u - 10, v + 10), f"{d:.1f}", font,
                        fill=(255, 240, 180),
                    )
                d += step
            # per-strobe exact reading, drawn next to the ruler
            if is_target and target_ds is not None:
                for k, d in enumerate(target_ds):
                    d = float(d)
                    x = ox_w + d * cos_t + surf_n_x * lift
                    z = oz_w - d * sin_t + surf_n_z * lift
                    q = project(x, y_side, z)
                    if q is not None:
                        u, v, _ = q
                        _text_with_outline(
                            draw, (u - 16, v + 22),
                            f"[{k+1}] {d:.2f}", font,
                            fill=(255, 235, 120),
                        )

    # ---- timing legend in top-right corner ----
    if target_tb is not None and target_ds is not None and len(times):
        if target_ds.ndim == 1:
            rows = [f"k={k+1}  t={t:.2f}s  s={float(d):.2f}m"
                    for k, (t, d) in enumerate(zip(times, target_ds))]
        else:
            rows = [f"k={k+1}  t={t:.2f}s  dx={float(target_ds[k,0]):.2f}m"
                    f"  dy={float(target_ds[k,1]):.2f}m"
                    for k, t in enumerate(times)]
        # box background for readability
        pad = 6
        line_h = 15
        box_w = 300 if (target_ds.ndim == 2) else 230
        box_h = pad * 2 + line_h * len(rows) + 18
        # placement: for 1D vertical motion (freefall, ramp on the
        # left half) the far-left column is empty of strobes → top-
        # left is safe. For horizontal launch the parabola occupies
        # the center/right, so put the legend in the top-right.
        if target_tb.kind == "horizontal_launch":
            x0 = width - box_w - 8
            y0 = 8
        else:
            x0 = 8
            y0 = 8
        bg = Image.new("RGBA", (box_w, box_h), (20, 20, 20, 180))
        if composite is not None:
            composite.paste(bg, (x0, y0), bg)
        draw.rectangle(
            [(x0, y0), (x0 + box_w, y0 + box_h)],
            outline=(200, 200, 200),
        )
        _text_with_outline(
            draw, (x0 + pad, y0 + pad),
            "strobe timing (s) & distance (m)",
            font, fill=(240, 240, 240),
        )
        for i, row in enumerate(rows):
            _text_with_outline(
                draw, (x0 + pad, y0 + pad + 18 + i * line_h),
                row, font, fill=(220, 230, 255),
            )


def draw_background_grid_pil(img: Image.Image, cell_px: int = 64,
                             axis_labels: bool = False) -> Image.Image:
    """Standalone PIL primitive: draw a faint 2D pixel-space grid on `img`.

    Used by both the MuJoCo-strobe path and the ScenarioBackend path so
    `BackgroundGrid` produces consistent visuals across all 15 scenes.
    Mutates `img` in place; also returns it for chaining.
    """
    cell = max(8, int(cell_px))
    width, height = img.size
    line_color = (180, 180, 180)
    label_color = (130, 130, 130)
    draw = ImageDraw.Draw(img)
    for x in range(cell, width, cell):
        draw.line([(x, 0), (x, height)], fill=line_color, width=1)
    for y in range(cell, height, cell):
        draw.line([(0, y), (width, y)], fill=line_color, width=1)
    if axis_labels:
        font_small = None
        for _p in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            try:
                font_small = ImageFont.truetype(_p, 12)
                break
            except Exception:
                continue
        if font_small is None:
            font_small = ImageFont.load_default()
        for x in range(cell, width, cell):
            _text_with_outline(draw, (x + 2, 2), str(x), font_small,
                               fill=label_color)
        for y in range(cell, height, cell):
            _text_with_outline(draw, (2, y + 2), str(y), font_small,
                               fill=label_color)
    return img


def _draw_background_grid_overlay(
    draw: ImageDraw.ImageDraw, compiled: CompiledScene,
    width: int, height: int, font_small,
) -> None:
    """Wrapper for the MuJoCo-strobe pipeline: pulls the BackgroundGrid
    entity out of compiled metadata and applies it via the standalone
    `draw_background_grid_pil` primitive."""
    bg_ents = compiled.metadata.get("background_grid_entities", [])
    if not bg_ents:
        return
    ent = bg_ents[0]
    # The strobe path supplies its own draw + font_small for consistency
    # with the rest of the overlays, so we replicate the primitive's logic
    # without instantiating a new Draw object.
    cell = max(8, int(ent.params.get("cell_px", 64)))
    show_labels = bool(ent.params.get("axis_labels", False))
    line_color = (180, 180, 180)
    label_color = (130, 130, 130)
    for x in range(cell, width, cell):
        draw.line([(x, 0), (x, height)], fill=line_color, width=1)
    for y in range(cell, height, cell):
        draw.line([(0, y), (width, y)], fill=line_color, width=1)
    if show_labels and font_small is not None:
        for x in range(cell, width, cell):
            _text_with_outline(draw, (x + 2, 2), str(x), font_small,
                               fill=label_color)
        for y in range(cell, height, cell):
            _text_with_outline(draw, (2, y + 2), str(y), font_small,
                               fill=label_color)


def _luminance_of_target(frame: np.ndarray, base_np: np.ndarray) -> float:
    """Mean luminance of the tracked-ball pixels in this frame.

    Uses the same saturation+change mask trick as render_scene's
    `_ball_mask`, but factored out so LightStrip/SpectrumBand can reuse it.
    """
    r = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    saturated = (max_c - min_c) > 60
    diff = np.abs(frame.astype(np.int16) - base_np).sum(axis=2)
    changed = diff > 30
    mask = saturated & changed
    if not mask.any():
        return 0.0
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return float(lum[mask].mean()) / 255.0


def _mean_color_of_target(frame: np.ndarray, base_np: np.ndarray
                          ) -> tuple[int, int, int]:
    r = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    saturated = (max_c - min_c) > 60
    diff = np.abs(frame.astype(np.int16) - base_np).sum(axis=2)
    changed = diff > 30
    mask = saturated & changed
    if not mask.any():
        return (60, 60, 60)
    return (
        int(np.clip(r[mask].mean(), 0, 255)),
        int(np.clip(g[mask].mean(), 0, 255)),
        int(np.clip(b[mask].mean(), 0, 255)),
    )


def _thermal_palette(t: float) -> tuple[int, int, int]:
    """t in [0, 1] -> (r, g, b). Black -> dark red -> orange -> yellow -> white."""
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.25:
        f = t / 0.25
        return (int(40 + 160 * f), int(0 + 0 * f), int(0))
    if t < 0.50:
        f = (t - 0.25) / 0.25
        return (int(200 + 55 * f), int(40 + 110 * f), int(0))
    if t < 0.75:
        f = (t - 0.50) / 0.25
        return (255, int(150 + 80 * f), int(0 + 60 * f))
    f = (t - 0.75) / 0.25
    return (255, int(230 + 25 * f), int(60 + 195 * f))


def _viridis_palette(t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    stops = [
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    ]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            f = (t - t0) / max(t1 - t0, 1e-6)
            return (
                int(c0[0] + f * (c1[0] - c0[0])),
                int(c0[1] + f * (c1[1] - c0[1])),
                int(c0[2] + f * (c1[2] - c0[2])),
            )
    return stops[-1][1]


def _palette_color(name: str, t: float) -> tuple[int, int, int]:
    name = (name or "thermal").lower()
    if name == "viridis":
        return _viridis_palette(t)
    if name == "hue":
        # full hue sweep: red -> yellow -> green -> cyan -> blue -> magenta
        import colorsys
        rgb = colorsys.hsv_to_rgb(float(np.clip(t, 0.0, 1.0)), 0.85, 0.95)
        return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    return _thermal_palette(t)


def _draw_light_strip_overlay(
    composite: Image.Image, compiled: CompiledScene,
    frames: list[np.ndarray], base_np: np.ndarray,
    width: int, height: int,
) -> Image.Image:
    """Append a horizontal color/luminance strip below the composite.

    For each LightStrip entity, take n_samples evenly spaced from the
    rendered frames, extract either the target ball's mean color
    (channel='rgb') or a thermal-mapped luminance (channel='luminance'),
    and paint a row of solid-color cells. Returns a NEW image with the
    strip stitched below.
    """
    ls_ents = compiled.metadata.get("light_strip_entities", [])
    if not ls_ents:
        return composite
    if not frames:
        return composite
    ent = ls_ents[0]
    n = int(ent.params.get("n_samples", 12))
    n = int(np.clip(n, 4, len(frames)))
    channel = str(ent.params.get("channel", "luminance")).lower()

    idxs = np.linspace(0, len(frames) - 1, n).round().astype(int)
    swatches: list[tuple[int, int, int]] = []
    for i in idxs:
        f = frames[int(i)]
        if channel == "rgb":
            swatches.append(_mean_color_of_target(f, base_np))
        else:
            lum = _luminance_of_target(f, base_np)
            swatches.append(_thermal_palette(lum))

    pad = 6
    label_h = 18
    cell_w = max(int((width - pad * 2) / n), 16)
    strip_w = cell_w * n + pad * 2
    strip_h = 56 + label_h + pad * 2
    out_w = max(composite.width, strip_w)
    out_h = composite.height + strip_h
    out = Image.new("RGB", (out_w, out_h), color=(20, 20, 22))
    cx = (out_w - composite.width) // 2
    out.paste(composite.convert("RGB"), (cx, 0))
    d = ImageDraw.Draw(out)
    sx = (out_w - strip_w) // 2
    sy = composite.height + pad
    label_font = None
    for _p in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            label_font = ImageFont.truetype(_p, 12)
            break
        except Exception:
            continue
    if label_font is None:
        label_font = ImageFont.load_default()
    for i, col in enumerate(swatches):
        x0 = sx + pad + i * cell_w
        x1 = x0 + cell_w - 2
        y0 = sy
        y1 = sy + 56
        d.rectangle([(x0, y0), (x1, y1)], fill=col,
                    outline=(180, 180, 180))
        d.text((x0 + 2, y1 + 2), str(i + 1), fill=(220, 220, 220),
               font=label_font)
    cap_y = sy + 56 + label_h
    d.text((sx + pad, cap_y),
           f"LightStrip [{channel}]  t -> right",
           fill=(200, 200, 200), font=label_font)
    return out


def _draw_spectrum_band_overlay(
    draw: ImageDraw.ImageDraw, compiled: CompiledScene,
    last_frame: Optional[np.ndarray], base_np: Optional[np.ndarray],
    width: int, height: int, font_small,
) -> None:
    """Draw a vertical color-temperature scale on the right edge with a
    cursor pointing to the target's current luminance / value position.

    The scale itself is fixed (palette: thermal/viridis/hue) so a VLM
    has a stable reference; only the cursor moves with the observable.
    """
    sb_ents = compiled.metadata.get("spectrum_band_entities", [])
    if not sb_ents:
        return
    ent = sb_ents[0]
    palette = str(ent.params.get("palette", "thermal")).lower()
    vmin = float(ent.params.get("min", 0.0))
    vmax = float(ent.params.get("max", 1.0))
    anchor = str(ent.params.get("anchor", "right")).lower()

    bar_w = 28
    margin = 14
    if anchor == "left":
        bar_x0 = margin
    else:
        bar_x0 = width - margin - bar_w
    bar_x1 = bar_x0 + bar_w
    bar_y0 = 60
    bar_y1 = height - 60
    bar_h = bar_y1 - bar_y0

    # Fill gradient
    for j in range(bar_h):
        t = 1.0 - (j / max(bar_h - 1, 1))  # top = max, bottom = min
        col = _palette_color(palette, t)
        draw.line([(bar_x0, bar_y0 + j), (bar_x1, bar_y0 + j)],
                  fill=col, width=1)
    draw.rectangle([(bar_x0, bar_y0), (bar_x1, bar_y1)],
                   outline=(40, 40, 40))

    # Tick marks at 0, 0.25, 0.5, 0.75, 1.0 of the bar
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        ty = bar_y1 - int(frac * bar_h)
        draw.line([(bar_x0 - 6, ty), (bar_x0, ty)], fill=(40, 40, 40), width=2)
        val = vmin + frac * (vmax - vmin)
        if font_small is not None:
            _text_with_outline(draw, (bar_x0 - 46, ty - 7),
                               f"{val:.2f}", font_small, fill=(240, 240, 240))

    # Cursor at current observable
    if last_frame is not None and base_np is not None:
        lum = _luminance_of_target(last_frame, base_np)
        v = float(np.clip(lum, 0.0, 1.0))
    else:
        v = 0.5
    cy = bar_y1 - int(v * bar_h)
    # arrow on the bar's outer side
    if anchor == "left":
        ax0, ax1 = bar_x1 + 2, bar_x1 + 14
    else:
        ax0, ax1 = bar_x0 - 14, bar_x0 - 2
    draw.polygon([(ax0, cy - 6), (ax1, cy), (ax0, cy + 6)],
                 fill=(255, 240, 60), outline=(0, 0, 0))


def _draw_protractor_overlay(
    draw: ImageDraw.ImageDraw, compiled: CompiledScene,
    width: int, height: int, font_small,
) -> None:
    """Render every AngleProtractor as a semicircular angle scale.

    The arc lies in the camera's image plane at the protractor origin's
    projected depth, so the radius scales with how the camera projects
    a 1m world distance at that depth. Major ticks are drawn every
    `tick_step_deg` degrees from 0 to `max_angle_deg`, with degree labels.
    """
    cam = compiled.metadata["camera"]
    cam_x, cam_y, cam_z = cam["x"], cam["y"], cam["z"]
    for ent in compiled.metadata.get("protractor_entities", []):
        origin = ent.params.get("origin", [0.0, 0.0, 0.0])
        ox, oy, oz = (float(origin[0]), float(origin[1]), float(origin[2]))
        radius = float(ent.params.get("radius", 1.5))
        max_deg = float(ent.params.get("max_angle_deg", 90.0))
        tick_step = float(ent.params.get("tick_step_deg", 10.0))

        proj = _project_point(cam_x, cam_y, cam_z, width, height,
                              ox, oy, oz)
        if proj is None:
            continue
        cu, cv, depth = proj
        # Pixel radius for `radius` metres at this depth
        f = height / (2.0 * np.tan(np.radians(MUJOCO_DEFAULT_FOVY_DEG) * 0.5))
        rpx = float(f * radius / max(depth, 1e-3))
        if rpx < 6:
            continue

        # Outline arc (white over black for legibility against any bg).
        n_arc = max(int(rpx * 0.3), 24)
        thetas = np.linspace(0.0, np.radians(max_deg), n_arc)
        pts = [(cu + rpx * np.cos(t), cv - rpx * np.sin(t)) for t in thetas]
        # Black outer + white inner
        for w_, fill in ((4, (0, 0, 0)), (2, (255, 255, 255))):
            for p, q in zip(pts, pts[1:]):
                draw.line([p, q], fill=fill, width=w_)

        # Tick marks + degree labels
        n_ticks = int(max_deg / tick_step) + 1
        for k in range(n_ticks):
            deg = k * tick_step
            if deg > max_deg + 1e-6:
                break
            t = np.radians(deg)
            major = (k % 3 == 0) or (deg in (0, max_deg))
            inner = rpx * (0.85 if major else 0.92)
            outer = rpx * 1.0
            x1 = cu + inner * np.cos(t); y1 = cv - inner * np.sin(t)
            x2 = cu + outer * np.cos(t); y2 = cv - outer * np.sin(t)
            draw.line([(x1, y1), (x2, y2)],
                      fill=(0, 0, 0), width=3)
            draw.line([(x1, y1), (x2, y2)],
                      fill=(255, 255, 255), width=1)
            if major:
                lx = cu + (rpx + 14) * np.cos(t) - 10
                ly = cv - (rpx + 14) * np.sin(t) - 8
                _text_with_outline(draw, (lx, ly), f"{int(round(deg))}°",
                                   font_small, fill=(255, 235, 60))
        # Centre dot
        draw.ellipse((cu - 3, cv - 3, cu + 3, cv + 3),
                     fill=(255, 235, 60), outline=(0, 0, 0))


def _render_filmstrip_below(
    composite: Image.Image, scene: Scene, compiled: CompiledScene,
    target: str, control_value: float, alpha: float,
    width: int, height: int,
) -> Image.Image:
    """For each Filmstrip entity, render N evenly-spaced single frames
    of the target's motion and paste them in a horizontal strip below
    `composite`. Returns a NEW image with the strip appended; if the
    scene has no Filmstrip, returns `composite` unchanged."""
    fs_ents = compiled.metadata.get("filmstrip_entities", [])
    if not fs_ents:
        return composite

    # Use the first Filmstrip (combining multiple is rarely useful)
    ent = fs_ents[0]
    columns = int(ent.params.get("columns", 5))
    columns = max(2, min(columns, 12))
    frame_size = int(ent.params.get("frame_size", 200))
    frame_size = max(80, min(frame_size, 400))

    times, _ = _choose_times(compiled, target, control_value,
                             columns, alpha=float(alpha))

    target_tb = next(tb for tb in compiled.tracked_bodies if tb.name == target)
    non_targets = [tb for tb in compiled.tracked_bodies if tb.name != target]

    model = mujoco.MjModel.from_xml_string(compiled.mjcf)
    data = mujoco.MjData(model)
    rndr = mujoco.Renderer(model, height=height, width=width)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "scene_cam")
    for tb in non_targets:
        _apply_tracked_motion(model, data, tb, 0.0, control_value, alpha)

    thumbs: list[Image.Image] = []
    aspect = width / max(height, 1)
    th_h = frame_size
    th_w = max(int(frame_size * aspect), 80)
    for t in times:
        _apply_tracked_motion(model, data, target_tb, float(t),
                              float(control_value), float(alpha))
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        rndr.update_scene(data, camera=cam_id)
        arr = rndr.render().copy()
        img = Image.fromarray(arr).convert("RGB").resize(
            (th_w, th_h), Image.LANCZOS)
        thumbs.append(img)
    rndr.close()

    pad = 8
    label_h = 24
    strip_w = th_w * len(thumbs) + pad * (len(thumbs) + 1)
    strip_h = th_h + pad * 2 + label_h
    out_w = max(composite.width, strip_w)
    out_h = composite.height + strip_h
    out = Image.new("RGB", (out_w, out_h), color=(20, 20, 22))
    cx = (out_w - composite.width) // 2
    out.paste(composite, (cx, 0))

    # Center the strip horizontally
    sx = (out_w - strip_w) // 2
    sy = composite.height
    fs_draw = ImageDraw.Draw(out)
    fs_draw.rectangle([(0, sy), (out_w, sy + strip_h)],
                      fill=(28, 28, 32))
    label_font = font_small = None
    for _p in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            label_font = ImageFont.truetype(_p, max(14, int(label_h * 0.6)))
            break
        except Exception:
            continue
    if label_font is None:
        label_font = ImageFont.load_default()

    for i, (thumb, t) in enumerate(zip(thumbs, times)):
        x = sx + pad + i * (th_w + pad)
        y = sy + pad
        out.paste(thumb, (x, y))
        fs_draw.rectangle([(x, y), (x + th_w, y + th_h)],
                          outline=(180, 180, 180))
        cap = f"t={float(t):.2f}s"
        _text_with_outline(fs_draw, (x + 4, y + th_h + 4), cap,
                           label_font, fill=(255, 235, 60))
    return out


def _has_only_fading_trail(compiled: CompiledScene) -> bool:
    """True iff the scene has FadingTrail but no StrobeTrail. In that
    case render_scene replaces the discrete-strobe overlays with a
    finer-grained fading trail and skips numbered labels."""
    return (
        bool(compiled.metadata.get("fading_trail_entities"))
        and not compiled.metadata.get("strobe_entities")
    )


def render_scene(
    scene: Scene,
    control_value: float,
    alpha: float = 2.5,
    width: int = 640,
    height: int = 480,
    target: Optional[str] = None,
    clean: bool = False,
) -> Image.Image:
    """Render a strobe composite. `target` selects which tracked body's
    motion drives the time axis and is strobed; other tracked bodies
    are frozen at t=0.

    If the scene contains an AngleProtractor, a semicircular angle
    scale is drawn on top of the strobe composite. If the scene has
    a Filmstrip, a horizontal strip of evenly-spaced frames is appended
    below the main composite. If the scene has a FadingTrail (and no
    StrobeTrail), a finer-grained fading trail replaces the numbered
    strobe markers.
    """
    compiled = compile_scene(scene, width=width, height=height)

    strobe_ents = compiled.metadata["strobe_entities"]
    fade_ents = compiled.metadata.get("fading_trail_entities", [])
    use_fading = _has_only_fading_trail(compiled)
    if strobe_ents:
        n_strobes = int(strobe_ents[0].params.get("n_samples", 10))
    elif fade_ents:
        n_strobes = int(fade_ents[0].params.get("n_samples", 60))
    else:
        n_strobes = 10
    if use_fading:
        n_strobes = int(np.clip(n_strobes, 20, 80))
    else:
        n_strobes = int(np.clip(n_strobes, 3, 30))

    target = target or compiled.default_target
    if target not in {tb.name for tb in compiled.tracked_bodies}:
        raise ValueError(
            f"target={target!r} is not a tracked body. Available: "
            f"{[tb.name for tb in compiled.tracked_bodies]}"
        )

    model = mujoco.MjModel.from_xml_string(compiled.mjcf)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "scene_cam")

    times, n_strobes = _choose_times(compiled, target, control_value,
                                     n_strobes, alpha=float(alpha))

    # freeze non-targets at t=0
    non_targets = [tb for tb in compiled.tracked_bodies if tb.name != target]
    for tb in non_targets:
        _apply_tracked_motion(model, data, tb, 0.0, control_value, alpha)

    target_tb = next(tb for tb in compiled.tracked_bodies if tb.name == target)
    frames: list[np.ndarray] = []
    for t in times:
        _apply_tracked_motion(model, data, target_tb, float(t),
                              float(control_value), float(alpha))
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam_id)
        frames.append(renderer.render().copy())

    renderer.close()

    base = Image.fromarray(frames[0]).convert("RGBA")
    base_np = np.array(base.convert("RGB"), dtype=np.int16)
    composite = base.copy()

    def _ball_mask(frame_np: np.ndarray) -> np.ndarray:
        """Pixels that look like the (tracked) ball in this frame.

        We identify the ball by: a channel is much brighter than the
        other two (so any saturated RGBA ball color qualifies), AND
        the pixel is distinctly different from the base frame at the
        same location. This rejects camera lighting/shadow changes.
        """
        r = frame_np[..., 0]
        g = frame_np[..., 1]
        b = frame_np[..., 2]
        # "not gray": one channel dominates the others
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        saturated = (max_c - min_c) > 60
        # different from the starting frame
        diff = np.abs(frame_np - base_np).sum(axis=2)
        changed = diff > 30
        return (saturated & changed).astype(np.uint8)

    # collect (x, y) centroid for each strobe frame so we can label them
    centroids: list[tuple[int, int] | None] = [None] * len(frames)
    for i, frame in enumerate(frames):
        frame_np = np.array(Image.fromarray(frame).convert("RGB"),
                            dtype=np.int16)
        if i == 0:
            # Estimate frame-0 ball centroid directly: saturated pixels
            # in frame 0 that are NOT saturated in a later frame.
            # We'll revisit this after we have at least one later mask.
            continue
        m = _ball_mask(frame_np)
        mask = (m * 255).astype(np.uint8)
        if i == 0:
            # unreachable, but keep full composite behavior below
            pass

        if i == len(frames) - 1:
            a = 1.0
        else:
            frac = i / max(len(frames) - 1, 1)
            a = 0.30 + 0.55 * frac
        overlay = Image.fromarray(frame).convert("RGBA")
        mask_img = Image.fromarray(mask, mode="L")
        mask_scaled = mask_img.point(lambda v, a=a: int(v * a))
        composite.paste(overlay, (0, 0), mask_scaled)

        ys, xs = np.nonzero(m)
        if xs.size >= 5:
            centroids[i] = (int(xs.mean()), int(ys.mean()))

    # Estimate frame-0 centroid from saturated pixels in frame 0 that
    # are NOT saturated (or not present) in a late frame.
    base_rgb = np.array(base.convert("RGB"), dtype=np.int16)
    r0, g0, b0 = base_rgb[..., 0], base_rgb[..., 1], base_rgb[..., 2]
    max0 = np.maximum(np.maximum(r0, g0), b0)
    min0 = np.minimum(np.minimum(r0, g0), b0)
    sat0 = (max0 - min0) > 60
    ref_idx = None
    for j in range(len(frames) - 1, 0, -1):
        if centroids[j] is not None:
            ref_idx = j
            break
    if ref_idx is not None and sat0.any():
        ref_np = np.array(Image.fromarray(frames[ref_idx]).convert("RGB"),
                          dtype=np.int16)
        rr, gr, br = ref_np[..., 0], ref_np[..., 1], ref_np[..., 2]
        maxr = np.maximum(np.maximum(rr, gr), br)
        minr = np.minimum(np.minimum(rr, gr), br)
        satr = (maxr - minr) > 60
        only_in_base = sat0 & ~satr
        ys0, xs0 = np.nonzero(only_in_base)
        if xs0.size >= 5:
            centroids[0] = (int(xs0.mean()), int(ys0.mean()))

    draw = ImageDraw.Draw(composite)
    # Resolution-adaptive font sizes: scale by image height relative
    # to the legacy 480-px baseline.
    _sc = max(height / 480.0, 1.0)
    _fs_main = max(int(round(18 * _sc)), 16)
    _fs_small = max(int(round(13 * _sc)), 12)
    _font_paths = [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    font = font_small = None
    for _p in _font_paths:
        try:
            font = ImageFont.truetype(_p, _fs_main)
            font_small = ImageFont.truetype(_p, _fs_small)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
        font_small = font

    # ---- BackgroundGrid: faint pixel-space grid under everything else ----
    _draw_background_grid_overlay(draw, compiled, width, height, font_small)

    # ---- draw bell markers and the beat timeline for ChimeTrack ----
    _draw_chime_overlays(
        draw, compiled, width, height, font_small,
        target=target, alpha=float(alpha),
        control_value=float(control_value),
        composite=composite,
    )

    # ---- AngleProtractor overlay (drawn under strobe number labels) ----
    _draw_protractor_overlay(draw, compiled, width, height, font_small)

    # ---- SpectrumBand: vertical color-temp scale + cursor for last frame ----
    _last_frame_np = (
        np.array(Image.fromarray(frames[-1]).convert("RGB"), dtype=np.int16)
        if frames else None
    )
    _draw_spectrum_band_overlay(
        draw, compiled, _last_frame_np, base_np,
        width, height, font_small,
    )

    # If FadingTrail is the only trail-style entity, skip numbered
    # strobe labels: the continuous fade is the readout.
    if use_fading:
        composite = _render_filmstrip_below(
            composite, scene, compiled, target,
            float(control_value), float(alpha), width, height,
        )
        composite = _draw_light_strip_overlay(
            composite, compiled, frames, base_np, width, height,
        )
        return composite.convert("RGB")

    # Strobe labels: single-column layout on the RIGHT of each strobe
    # point, with a vertical "push-down" stacking so labels never
    # overlap in dense clusters. leader lines connect each label back
    # to its true strobe centroid.
    def _text_bbox(txt: str, at: tuple[float, float]):
        try:
            l, t_, r, b = draw.textbbox(at, txt, font=font)
        except Exception:
            w_ = len(txt) * _fs_main * 0.6
            h_ = _fs_main
            l, t_, r, b = at[0], at[1], at[0] + w_, at[1] + h_
        return (l - 2, t_ - 1, r + 2, b + 1)

    x_off = max(int(round(16 * _sc)), 16)
    line_h = _fs_main + max(int(round(4 * _sc)), 4)

    # First, determine per-strobe target label position (ly) for an
    # initial "ideal" vertical location (centered on the centroid).
    # Then sweep top-to-bottom and push any label that would collide
    # with the previous one downward by line_h.
    lys: list[float | None] = [None] * len(centroids)
    prev_bottom = -1e9
    half_th = _fs_main / 2
    for i, c in enumerate(centroids):
        if c is None:
            continue
        x, y = c
        ideal_ly = y - half_th
        ly = max(ideal_ly, prev_bottom + 2)
        # clip so label doesn't run off the bottom edge
        if ly + _fs_main > height - 4:
            ly = height - 4 - _fs_main
        lys[i] = ly
        prev_bottom = ly + _fs_main

    for i, c in enumerate(centroids):
        if c is None:
            continue
        x, y = c
        # In `clean` mode we drop the per-strobe numeric label (1, 2, ..., N)
        # and the leader line; only the bright centroid dot remains, so the
        # image carries no on-canvas text. Used by GCD demos and the GCD
        # runner where "PASS / FAIL is the only text" is the design rule.
        r = max(int(round(3 * _sc)), 3)
        if clean:
            draw.ellipse((x - r, y - r, x + r, y + r),
                         fill=(255, 235, 60), outline=(0, 0, 0))
            continue
        label = str(i + 1)
        ly = lys[i]
        lx = x + x_off
        # keep within right edge; if we'd fall off, pull back left
        tb = _text_bbox(label, (lx, ly))
        tw = tb[2] - tb[0]
        if lx + tw > width - 4:
            lx = width - 4 - tw
        tb = _text_bbox(label, (lx, ly))
        # leader line from strobe centroid to the middle-left of the
        # label rectangle
        lead_x = tb[0] - 2
        lead_y = (tb[1] + tb[3]) / 2
        draw.line([(x, y), (lead_x, lead_y)],
                  fill=(0, 0, 0), width=2)
        draw.line([(x, y), (lead_x, lead_y)],
                  fill=(255, 235, 60), width=1)
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                if ox == 0 and oy == 0:
                    continue
                draw.text((lx + ox, ly + oy), label, fill=(0, 0, 0),
                          font=font)
        draw.text((lx, ly), label, fill=(255, 235, 60), font=font)
        draw.ellipse((x - r, y - r, x + r, y + r),
                     fill=(255, 235, 60), outline=(0, 0, 0))

    if not clean:
        composite = _render_filmstrip_below(
            composite, scene, compiled, target,
            float(control_value), float(alpha), width, height,
        )
    composite = _draw_light_strip_overlay(
        composite, compiled, frames, base_np, width, height,
    )
    return composite.convert("RGB")


def render_scene_animation(
    scene: Scene,
    control_value: float,
    alpha: float = 2.5,
    width: int = 640,
    height: int = 480,
    target: Optional[str] = None,
) -> tuple[list[Image.Image], int]:
    """Render an Animation entity as a sequence of frames + fps.

    Use the scene's first `Animation` entity to set fps/n_frames; if
    none is present, defaults are used. Returns (frames, fps) where
    frames are RGB PIL images suitable for assembling into a GIF via
    `frames[0].save(path, format='GIF', append_images=frames[1:],
    duration=int(1000/fps), loop=0)`.
    """
    compiled = compile_scene(scene, width=width, height=height)

    anim_ents = compiled.metadata.get("animation_entities", [])
    if anim_ents:
        ent = anim_ents[0]
        fps = max(1, int(ent.params.get("fps", 8)))
        n_frames = int(ent.params.get("n_frames", 32))
    else:
        fps = 8
        n_frames = 32
    n_frames = int(np.clip(n_frames, 6, 64))

    target = target or compiled.default_target
    if target not in {tb.name for tb in compiled.tracked_bodies}:
        raise ValueError(
            f"target={target!r} is not a tracked body. Available: "
            f"{[tb.name for tb in compiled.tracked_bodies]}"
        )

    times, _ = _choose_times(compiled, target, control_value,
                             n_frames, alpha=float(alpha))

    target_tb = next(tb for tb in compiled.tracked_bodies if tb.name == target)
    non_targets = [tb for tb in compiled.tracked_bodies if tb.name != target]

    model = mujoco.MjModel.from_xml_string(compiled.mjcf)
    data = mujoco.MjData(model)
    rndr = mujoco.Renderer(model, height=height, width=width)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "scene_cam")
    for tb in non_targets:
        _apply_tracked_motion(model, data, tb, 0.0, control_value, alpha)

    out_frames: list[Image.Image] = []
    for t in times:
        _apply_tracked_motion(model, data, target_tb, float(t),
                              float(control_value), float(alpha))
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        rndr.update_scene(data, camera=cam_id)
        arr = rndr.render().copy()
        out_frames.append(Image.fromarray(arr).convert("RGB"))
    rndr.close()
    return out_frames, fps
