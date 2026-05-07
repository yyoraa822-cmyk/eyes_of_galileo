"""Goal-Conditioned Discovery (GCD) layer.

This package sits *above* the DSL layer. The DSL handles "how to observe".
The goals layer handles "what task the agent must achieve" and "how to give
visual pass/fail feedback".

Public surface:
    Goal                dataclass
    GOAL_POOL           per-scene hold-out goals
    draw_goal_marker    PIL primitive — paints the green goal object
    overlay_position    PIL primitive — green target / red X / arrow
    overlay_scalar      PIL primitive — two bars + delta arrow
    overlay_phase       PIL primitive — polar dial with two markers
    materialize_goal_pool() -> populates GOAL_POOL via the oracle solver
    solve_for_control(scenario, target_y, ...) -> u*  (oracle, agents never call)
"""
from .config import Goal, GOAL_POOL, get_goals, VIEW_GOAL_META
from .marker import draw_goal_marker
from .overlay import overlay_position, overlay_scalar, overlay_phase, draw_pass_badge
from .projector import GoalProjection, project_goal, goal_kind_for
from .solver import materialize_goal_pool, solve_for_control

__all__ = [
    "Goal",
    "GOAL_POOL",
    "VIEW_GOAL_META",
    "get_goals",
    "draw_goal_marker",
    "overlay_position",
    "overlay_scalar",
    "overlay_phase",
    "draw_pass_badge",
    "GoalProjection",
    "project_goal",
    "goal_kind_for",
    "materialize_goal_pool",
    "solve_for_control",
]
