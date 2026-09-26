"""带车体扫掠检查的姿态格路径规划（可复用库）。

公开接口：
    Grid, Heading, Pose, ArticulatedPose, Action, StepEvidence,
    ArticulatedStepEvidence, Plan, ArticulatedPlan,
    SearchStatus, SearchResult, ArticulatedSearchResult,
    plan, plan_articulated, replay, replay_articulated, ReplayError,
    transition, successors, articulated_transition, articulated_successors,
    heuristic, articulated_heuristic, MAX_GRID_SIDE, MAX_ARTICULATION_TURNS
"""

from .model import (
    MAX_ARTICULATION_TURNS,
    MAX_GRID_SIDE,
    Action,
    ArticulatedPlan,
    ArticulatedPose,
    ArticulatedStepEvidence,
    Grid,
    Heading,
    Plan,
    Pose,
    StepEvidence,
    articulated_successors,
    articulated_transition,
    successors,
    transition,
)
from .planner import (
    ArticulatedSearchResult,
    ReplayError,
    SearchResult,
    SearchStatus,
    articulated_heuristic,
    heuristic,
    plan,
    plan_articulated,
    replay,
    replay_articulated,
)

__all__ = [
    "MAX_ARTICULATION_TURNS",
    "MAX_GRID_SIDE",
    "Action",
    "ArticulatedPlan",
    "ArticulatedPose",
    "ArticulatedStepEvidence",
    "Grid",
    "Heading",
    "Plan",
    "Pose",
    "StepEvidence",
    "successors",
    "transition",
    "articulated_successors",
    "articulated_transition",
    "ReplayError",
    "ArticulatedSearchResult",
    "SearchResult",
    "SearchStatus",
    "articulated_heuristic",
    "heuristic",
    "plan",
    "plan_articulated",
    "replay",
    "replay_articulated",
]

__version__ = "1.0.0"
