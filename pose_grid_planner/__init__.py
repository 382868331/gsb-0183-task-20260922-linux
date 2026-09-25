"""带车体扫掠检查的姿态格路径规划（可复用库）。

公开接口：
    Grid, Heading, Pose, Action, StepEvidence, Plan,
    SearchStatus, SearchResult, plan, replay, ReplayError,
    transition, successors, heuristic, MAX_GRID_SIDE
离散时间扩展：
    TimedGrid, TimedPlan, TimedStepEvidence, TimedSearchResult,
    timed_plan, timed_replay, timed_transition, timed_successors, MAX_HORIZON
"""

from .model import (
    MAX_GRID_SIDE,
    Action,
    Grid,
    Heading,
    Plan,
    Pose,
    StepEvidence,
    successors,
    transition,
)
from .planner import (
    ReplayError,
    SearchResult,
    SearchStatus,
    heuristic,
    plan,
    replay,
)
from .timed import (
    MAX_HORIZON,
    TimedGrid,
    TimedPlan,
    TimedSearchResult,
    TimedStepEvidence,
    timed_plan,
    timed_replay,
    timed_successors,
    timed_transition,
)

__all__ = [
    "MAX_GRID_SIDE",
    "Action",
    "Grid",
    "Heading",
    "Plan",
    "Pose",
    "StepEvidence",
    "successors",
    "transition",
    "ReplayError",
    "SearchResult",
    "SearchStatus",
    "heuristic",
    "plan",
    "replay",
    "MAX_HORIZON",
    "TimedGrid",
    "TimedPlan",
    "TimedSearchResult",
    "TimedStepEvidence",
    "timed_plan",
    "timed_replay",
    "timed_successors",
    "timed_transition",
]

__version__ = "1.1.0"
