"""带车体扫掠检查的姿态格路径规划（pose-grid planner）。

公开 API：
- 数据结构：Heading / Action / Pose / GridWorld / PlanStatus /
  StepEvidence / PlanningResult
- 函数：plan()、heuristic()
- 错误：PlannerError / InvalidParameterError / InvalidGridError /
  InvalidPoseError

本包只依赖 Python 标准库。
"""

from .grid import (
    ACTION_COST,
    Action,
    GridWorld,
    Heading,
    InvalidGridError,
    InvalidParameterError,
    InvalidPoseError,
    PlannerError,
    Pose,
    Transition,
)
from .planner import (
    BUDGET_DEFAULT,
    MAX_GRID,
    PlanStatus,
    PlanningResult,
    StepEvidence,
    heuristic,
    plan,
)

__all__ = [
    "Heading",
    "Action",
    "Pose",
    "GridWorld",
    "Transition",
    "ACTION_COST",
    "PlanStatus",
    "StepEvidence",
    "PlanningResult",
    "plan",
    "heuristic",
    "PlannerError",
    "InvalidParameterError",
    "InvalidGridError",
    "InvalidPoseError",
    "MAX_GRID",
    "BUDGET_DEFAULT",
]
