"""带车体扫掠检查的姿态格路径规划（可复用库）。

公开接口：
    Grid, Heading, Pose, Action, StepEvidence, Plan,
    SearchStatus, SearchResult, plan, replay, ReplayError,
    transition, successors, heuristic, MAX_GRID_SIDE
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
from .timing import (
    SpeedLimitReason,
    TimedNode,
    TimedPath,
    TimedSegment,
    TimeParameterizationError,
    time_parameterize,
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
    "SpeedLimitReason",
    "TimedNode",
    "TimedPath",
    "TimedSegment",
    "TimeParameterizationError",
    "time_parameterize",
]

__version__ = "1.1.0"
