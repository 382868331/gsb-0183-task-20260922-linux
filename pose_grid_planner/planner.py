"""A* 姿态格规划器。

- 状态 = Pose(x, y, heading)。
- 启发式 h = max(曼哈顿距离 * 10, 最少转向次数 * 8)。两者各自可采纳且一致，
  取 max 仍可采纳且一致，因此 A*（带 closed 集）返回最优解。
- 搜索预算（max_expansions）耗尽与“证明无解”严格区分：
  堆被取空才算证明无解（NO_PATH）；预算触顶返回 BUDGET_EXHAUSTED。
"""

from __future__ import annotations

import enum
import heapq
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from ._validation import require_int
from .model import Action, Grid, Plan, Pose, StepEvidence, successors

_FORWARD_COST = Action.FORWARD.cost  # 10，曼哈顿每格的最小平移代价
_TURN_COST = Action.TURN_LEFT.cost  # 8，单次 90 度转向代价


class SearchStatus(enum.Enum):
    OPTIMAL = "optimal"  # 找到最优路径
    NO_PATH = "no_path"  # 已证明无解（开集耗尽）
    BUDGET_EXHAUSTED = "budget_exhausted"  # 搜索预算耗尽，既未找到也未证明无解


@dataclass(frozen=True)
class SearchResult:
    """规划结果。status 为 OPTIMAL 时 plan 非空；其余情况 plan 为 None。"""

    status: SearchStatus
    plan: Optional[Plan]
    expansions: int
    start: Pose
    goal: Pose

    @property
    def found(self) -> bool:
        return self.status is SearchStatus.OPTIMAL


def heuristic(pose: Pose, goal: Pose) -> int:
    """h = max(曼哈顿*10, 最少转向代价)。只允许这两个分量（题面契约）。"""
    manhattan = abs(pose.x - goal.x) + abs(pose.y - goal.y)
    turn_diff = (int(goal.heading) - int(pose.heading)) % 4
    min_turns = min(turn_diff, 4 - turn_diff)
    return max(manhattan * _FORWARD_COST, min_turns * _TURN_COST)


def plan(
    grid: Grid,
    start: Pose,
    goal: Pose,
    *,
    max_expansions: Optional[int] = None,
) -> SearchResult:
    """在 grid 上从 start 规划到 goal（含目标朝向）。

    参数校验失败抛 TypeError/ValueError；max_expansions 为 None 表示不限预算。
    返回 SearchResult；无解与预算耗尽通过 status 区分，绝不静默返回次优路径。
    """
    if not isinstance(grid, Grid):
        raise TypeError(f"grid 必须是 Grid，收到 {type(grid).__name__}: {grid!r}")
    if not isinstance(start, Pose):
        raise TypeError(f"start 必须是 Pose，收到 {type(start).__name__}: {start!r}")
    if not isinstance(goal, Pose):
        raise TypeError(f"goal 必须是 Pose，收到 {type(goal).__name__}: {goal!r}")
    if max_expansions is not None:
        max_expansions = require_int("max_expansions", max_expansions)
        if max_expansions < 0:
            raise ValueError(f"max_expansions 必须 >= 0，收到 {max_expansions}")
    grid.check_pose(start, name="start")
    grid.check_pose(goal, name="goal")

    if start == goal:
        return SearchResult(
            status=SearchStatus.OPTIMAL,
            plan=Plan(actions=(), total_cost=0, evidence=()),
            expansions=0,
            start=start,
            goal=goal,
        )

    # 堆元素: (f, g, 序号, 姿态)。序号保证同 f 时按插入顺序弹出，结果确定。
    open_heap: list[tuple[int, int, int, Pose]] = [(heuristic(start, goal), 0, 0, start)]
    best_g: Dict[Pose, int] = {start: 0}
    # parent[pose] = (父姿态, 动作, 扫掠格, 动作代价)
    parent: Dict[Pose, Tuple[Pose, Action, Tuple[tuple[int, int], ...], int]] = {}
    counter = 1
    expansions = 0

    while open_heap:
        if max_expansions is not None and expansions >= max_expansions:
            return SearchResult(SearchStatus.BUDGET_EXHAUSTED, None, expansions, start, goal)
        _f, g, _seq, pose = heapq.heappop(open_heap)
        if g > best_g.get(pose, _INF):
            continue  # 过期堆项
        expansions += 1
        if pose == goal:
            return SearchResult(
                status=SearchStatus.OPTIMAL,
                plan=_reconstruct(parent, start, goal, g),
                expansions=expansions,
                start=start,
                goal=goal,
            )
        for action, nxt, swept, cost in successors(grid, pose):
            g2 = g + cost
            if g2 < best_g.get(nxt, _INF):
                best_g[nxt] = g2
                parent[nxt] = (pose, action, swept, cost)
                heapq.heappush(open_heap, (g2 + heuristic(nxt, goal), g2, counter, nxt))
                counter += 1
    return SearchResult(SearchStatus.NO_PATH, None, expansions, start, goal)


_INF = float("inf")


def _reconstruct(
    parent: Dict[Pose, Tuple[Pose, Action, Tuple[tuple[int, int], ...], int]],
    start: Pose,
    goal: Pose,
    total_cost: int,
) -> Plan:
    chain: list[StepEvidence] = []
    pose = goal
    while pose != start:
        prev, action, swept, cost = parent[pose]
        chain.append(StepEvidence(action=action, from_pose=prev, to_pose=pose, swept_cells=swept, cost=cost))
        pose = prev
    chain.reverse()
    return Plan(
        actions=tuple(e.action for e in chain),
        total_cost=total_cost,
        evidence=tuple(chain),
    )


class ReplayError(ValueError):
    """重放失败：动作序列在某一步扫掠不通过或类型非法。"""


def replay(grid: Grid, start: Pose, actions: Tuple[Action, ...]) -> Tuple[Pose, Tuple[StepEvidence, ...]]:
    """按动作序列从 start 逐步重放，返回 (最终姿态, 逐步证据)。

    任一步扫掠不通过即抛 ReplayError（携带步序号与姿态，可定位）。
    """
    from .model import transition

    if not isinstance(grid, Grid):
        raise TypeError(f"grid 必须是 Grid，收到 {type(grid).__name__}: {grid!r}")
    if not isinstance(start, Pose):
        raise TypeError(f"start 必须是 Pose，收到 {type(start).__name__}: {start!r}")
    grid.check_pose(start, name="start")
    pose = start
    evidence: list[StepEvidence] = []
    for i, action in enumerate(actions):
        if not isinstance(action, Action):
            raise ReplayError(f"actions[{i}] 必须是 Action，收到 {type(action).__name__}: {action!r}")
        result = transition(grid, pose, action)
        if result is None:
            raise ReplayError(f"第 {i} 步 {action.name} 在姿态 {pose!r} 处扫掠不通过（越界或碰障碍）")
        nxt, swept = result
        evidence.append(
            StepEvidence(
                action=action,
                from_pose=pose,
                to_pose=nxt,
                swept_cells=swept,
                cost=grid.step_cost(action, swept),
            )
        )
        pose = nxt
    return pose, tuple(evidence)
