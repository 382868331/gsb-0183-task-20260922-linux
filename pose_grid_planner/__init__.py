"""带车体扫掠检查的姿态格路径规划（可复用库），含离散时间扩展。

公开接口：
    Grid, Heading, Pose, Action, StepEvidence, Plan,
    TimedState, Timetable, DEFAULT_WAIT_COST,
    SearchStatus, SearchResult, plan, plan_timed, replay, replay_timed,
    ReplayError, transition, successors, timed_transition, timed_successors,
    heuristic, MAX_GRID_SIDE
"""

from .model import (
    DEFAULT_WAIT_COST,
    MAX_GRID_SIDE,
    SPATIAL_ACTIONS,
    Action,
    Grid,
    Heading,
    Plan,
    Pose,
    StepEvidence,
    TimedState,
    Timetable,
    successors,
    timed_successors,
    timed_transition,
    transition,
)
from .planner import (
    ReplayError,
    SearchResult,
    SearchStatus,
    heuristic,
    plan,
    plan_timed,
    replay,
    replay_timed,
)

__all__ = [
    "DEFAULT_WAIT_COST",
    "MAX_GRID_SIDE",
    "SPATIAL_ACTIONS",
    "Action",
    "Grid",
    "Heading",
    "Plan",
    "Pose",
    "StepEvidence",
    "TimedState",
    "Timetable",
    "successors",
    "timed_successors",
    "timed_transition",
    "transition",
    "ReplayError",
    "SearchResult",
    "SearchStatus",
    "heuristic",
    "plan",
    "plan_timed",
    "replay",
    "replay_timed",
]

__version__ = "1.1.0"
