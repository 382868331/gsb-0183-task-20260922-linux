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
from collections.abc import Iterable
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


@dataclass(frozen=True)
class ReplanResult:
    """一次“静态障碍更新后重规划”的结果。

    - prefix：已执行前缀（在**旧网格**上逐步重放验证过的 Plan，永不改变）；
    - current_pose：执行前缀后机器人实际所在姿态，即剩余段搜索起点；
    - updated_grid：并入新增障碍后的新网格（旧网格不受影响）；
    - search：在 updated_grid 上从 current_pose 到 goal 的完整 SearchResult；
    - status 为 OPTIMAL 时 remainder 非空（仅剩余段，不含前缀）。
    """

    status: SearchStatus
    prefix: Plan
    executed_actions: int
    current_pose: Pose
    updated_grid: Grid
    search: SearchResult
    start: Pose
    goal: Pose

    @property
    def found(self) -> bool:
        return self.status is SearchStatus.OPTIMAL

    @property
    def remainder(self) -> Optional[Plan]:
        return self.search.plan

    @property
    def total_cost(self) -> int:
        """前缀实际代价 + 剩余段最优代价；非 OPTIMAL 时不可用。"""
        if self.search.plan is None:
            raise ValueError(f"status={self.status.value} 时不存在 total_cost")
        return self.prefix.total_cost + self.search.plan.total_cost

    @property
    def full_actions(self) -> Tuple[Action, ...]:
        """前缀动作拼接剩余段动作；非 OPTIMAL 时不可用。

        注意：前缀证据按旧网格成立、剩余段证据按更新后网格成立，因此这里只
        拼接动作序列，不跨网格拼接扫掠证据。
        """
        if self.search.plan is None:
            raise ValueError(f"status={self.status.value} 时不存在 full_actions")
        return self.prefix.actions + self.search.plan.actions


def replan(
    grid: Grid,
    start: Pose,
    goal: Pose,
    actions: Iterable[Action],
    executed_actions: int,
    added_obstacles: Iterable[tuple[int, int]],
    *,
    max_expansions: Optional[int] = None,
) -> ReplanResult:
    """旧路径执行到指定步数后接收一次静态障碍更新，保留前缀并全量重新规划。

    流程（任一步失败即抛异常，不修改 grid、不返回半更新结果）：
    1. 在**旧网格**上从 start 完整重放 actions（复用 replay 的车体扫掠检查），
       旧路径本身必须无碰撞；取前 executed_actions 步作为已执行前缀；
    2. 把 added_obstacles 并入网格得到不可变的 updated_grid（只允许新增，
       重复/越界/非法类型均带索引报错）；
    3. 当前姿态在 updated_grid 上必须无碰撞——新增障碍压到机器人当前占用格
       属于自相矛盾的更新，抛 ValueError；
    4. 从当前姿态到原 goal 做全量 A*（允许全量搜索，预算语义与 plan 相同）。
       目标格被新障碍埋住或开集耗尽证明不可达均为 NO_PATH。

    参数校验失败抛 TypeError/ValueError（含 ReplayError）；旧网格与原 actions
    不会被修改。
    """
    if not isinstance(grid, Grid):
        raise TypeError(f"grid 必须是 Grid，收到 {type(grid).__name__}: {grid!r}")
    if not isinstance(start, Pose):
        raise TypeError(f"start 必须是 Pose，收到 {type(start).__name__}: {start!r}")
    if not isinstance(goal, Pose):
        raise TypeError(f"goal 必须是 Pose，收到 {type(goal).__name__}: {goal!r}")
    if isinstance(actions, (str, bytes)) or not isinstance(actions, Iterable):
        raise TypeError(f"actions 必须是 Action 的可迭代对象，收到 {type(actions).__name__}")
    executed_actions = require_int("executed_actions", executed_actions)
    if executed_actions < 0:
        raise ValueError(f"executed_actions 必须 >= 0，收到 {executed_actions}")
    actions = tuple(actions)
    if executed_actions > len(actions):
        raise ValueError(
            f"executed_actions={executed_actions} 超过旧路径动作数 {len(actions)}"
        )
    if max_expansions is not None:
        max_expansions = require_int("max_expansions", max_expansions)
        if max_expansions < 0:
            raise ValueError(f"max_expansions 必须 >= 0，收到 {max_expansions}")
    grid.check_pose(start, name="start")
    grid.check_pose(goal, name="goal")

    # 1) 旧路径在旧网格上必须完整可重放（任一步碰撞 → ReplayError，带步序号）
    _final, full_evidence = replay(grid, start, actions)
    prefix = Plan(
        actions=actions[:executed_actions],
        total_cost=sum(e.cost for e in full_evidence[:executed_actions]),
        evidence=full_evidence[:executed_actions],
    )
    current = start if executed_actions == 0 else full_evidence[executed_actions - 1].to_pose

    # 2) 静态更新：只新增；构造失败这里直接抛错，旧网格与前缀均已确定不受影响
    updated_grid = grid.with_added_obstacles(added_obstacles)

    # 3) 新障碍不能压在机器人当前占用格上（否则世界状态自相矛盾）
    for cell in updated_grid.pose_cells(current):
        if not updated_grid.is_free(cell):
            raise ValueError(
                f"新增障碍占据了当前姿态的占用格 {cell}（机器人当前位于 {current!r}），"
                f"该更新与已执行路径矛盾"
            )

    # 4) 目标被埋 = 平凡地不可达；按 NO_PATH 返回而非抛错（goal 在旧网格上合法）
    if not updated_grid.is_pose_free(goal):
        search = SearchResult(SearchStatus.NO_PATH, None, 0, current, goal)
        return ReplanResult(SearchStatus.NO_PATH, prefix, executed_actions, current,
                            updated_grid, search, start, goal)

    search = plan(updated_grid, current, goal, max_expansions=max_expansions)
    return ReplanResult(
        status=search.status,
        prefix=prefix,
        executed_actions=executed_actions,
        current_pose=current,
        updated_grid=updated_grid,
        search=search,
        start=start,
        goal=goal,
    )


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
        evidence.append(StepEvidence(action=action, from_pose=pose, to_pose=nxt, swept_cells=swept, cost=action.cost))
        pose = nxt
    return pose, tuple(evidence)
