"""End-to-end demo for Galileo-DSL.

Reproduces the toy example discussed during the design phase:
  1. Basic apparatus: a single inclined ramp + freefall ball + strobe trail.
  2. Refined apparatus: the same ramp, but with a physical ruler along the
     surface and a reference cube placed near the release point, giving
     the VLM scale cues that are only expressible through geometry.

Both are rendered at the same control_value and alpha so the images can
be inspected side by side.

Run:
    cd /scratch/hlv8980/william/umm/draft/galileo
    python -m dsl.demo
"""
from __future__ import annotations

import sys
from pathlib import Path

from .api import compile_and_render, validate


BASIC_SCENE = """
scene:
  name: basic_ramp
  entities:
    - name: ramp1
      type: InclinedRamp
      params:
        ramp:
          name: ramp_body
          angle_deg: 25.0
          length: 6.0
    - name: ball1
      type: FreefallBall
      params:
        ball:
          name: ball_body
          radius: 0.18
          color: red
    - name: strobe
      type: StrobeTrail
      params:
        target_body: ball_body
        n_samples: 10
"""


REFINED_SCENE = """
scene:
  name: ramp_with_scale
  entities:
    - name: ramp1
      type: InclinedRamp
      params:
        ramp:
          name: ramp_body
          angle_deg: 25.0
          length: 6.0
        ruler_along:
          name: tick_posts
          length: 6.0
          tick_spacing: 1.0
    - name: ball1
      type: FreefallBall
      params:
        ball:
          name: ball_body
          radius: 0.18
          color: red
    - name: cube1
      type: ReferenceScale
      params:
        reference_cube:
          name: ref_cube
          edge_length: 0.5
        position: [-0.6, 0.8, 0.25]
    - name: strobe
      type: StrobeTrail
      params:
        target_body: ball_body
        n_samples: 10
"""


PENDULUM_SCENE = """
scene:
  name: simple_pendulum
  entities:
    - name: pend1
      type: Pendulum
      params:
        string:
          name: line
          length: 1.5
        ball:
          name: bob
          radius: 0.10
          color: blue
        pivot_height: 2.2
        theta_max_deg: 20.0
        n_samples: 12
    - name: strobe
      type: StrobeTrail
      params:
        target_body: bob
        n_samples: 12
"""


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "dsl_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (tag, yaml, control_value, alpha)
    cases = [
        ("basic", BASIC_SCENE, 0.9, 2.5),
        ("refined", REFINED_SCENE, 0.9, 2.5),
        ("pendulum", PENDULUM_SCENE, 1.5, 0.5),
    ]
    for tag, src, cv, alpha in cases:
        print(f"\n=== {tag} (cv={cv}, alpha={alpha}) ===")
        v = validate(src)
        print(f"validate -> ok={v.ok} gate={v.gate} reason={v.reason!r}")
        if not v.ok:
            return 1

        img = compile_and_render(src, control_value=cv, alpha=alpha)
        path = out_dir / f"{tag}.png"
        img.save(path)
        print(f"wrote {path} ({img.size[0]}x{img.size[1]})")

    print(f"\nAll outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
