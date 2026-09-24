"""静态障碍更新后的一次重规划（保留已执行前缀）。

场景契约（本次迭代）：
1. 已有旧规划 ``previous``（Plan）与旧网格 ``grid``、旧起点 ``start``；
2. 机器人沿旧规划执行了前 ``executed`` 个动作，停在第 ``executed`` 个姿态；
3. 此时接收**一次**静态障碍更新（新增若干障碍格，不清除旧障碍）；
4. 保留已执行前缀，从当前位置在更新后的网格上**全量**重新搜索到同一目标的剩余路径。

复用关系：旧规划在**旧网格**上经 model.transition 逐步复核（不盲信传入 Plan），
前缀仅作历史截取；当前安全性用 Grid.is_pose_free 判定，剩余路径搜索走
planner.plan（同一套 A* 与 NO_PATH/BUDGET 语义）。
本模块不做增量搜索、不处理连续移动障碍，也不做多轮承诺前缀。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

from ._validation import require_int
from .model import Grid, Plan, Pose, transition
from .planner import SearchResult, SearchStatus, plan


class ReplanningStatus(enum.Enum):
    """重规划结果状态。"""

    OPTIMAL = "optimal"  # 找到从当前位置到目标的最优剩余路径（前缀必然可行）
    NO_PATH = "no_path"  # 已证明当前位置无法到达目标（开集耗尽）
    BUDGET_EXHAUSTED = "budget_exhausted"  # 剩余路径搜索预算耗尽，未找到也未证明无解
    PREFIX_BLOCKED = "prefix_blocked"  # 新障碍压住机器人当前占用格，当前状态不安全


@dataclass(frozen=True)
class ReplanningResult:
    """一次重规划的完整结果。

    - ``status`` 为 OPTIMAL 时 ``plan`` 非空：它是“已执行前缀 + 新剩余段”的**完整** Plan。
      其中 ``remaining`` 保证能在 ``updated_grid`` 上从 ``current_pose`` 重放到达 goal；
      ``prefix`` 是在旧网格上已经发生的历史，若更新后旧起点仍空闲，整段 plan 也可在
      ``updated_grid`` 上从 start 重放（身后格被堵时只需也只应重放 remaining）；
    - ``prefix`` 始终给出已执行前缀（即使 PREFIX_BLOCKED 也保留，便于定位），其证据是
      在**旧网格**上执行当时的历史扫掠事实（身后格后被堵住不影响历史）；
    - ``remaining`` 仅 OPTIMAL 时非空，是当前位置到目标的新段；
    - ``updated_grid`` 是合并障碍后的新网格（旧网格不被修改）。
    """

    status: ReplanningStatus
    plan: Optional[Plan]
    prefix: Plan
    remaining: Optional[Plan]
    executed: int
    current_pose: Pose
    updated_grid: Grid
    expansions: int
    start: Pose
    goal: Pose

    @property
    def found(self) -> bool:
        return self.status is ReplanningStatus.OPTIMAL


def _prefix_plan(previous: Plan, executed: int) -> Plan:
    """截取已执行前缀。证据是“执行当时”（旧网格）的历史扫掠事实，原样保留：
    新障碍落在已离开的身后格不会让历史动作变成碰撞。"""
    evidence = previous.evidence[:executed]
    return Plan(
        actions=previous.actions[:executed],
        total_cost=sum(e.cost for e in evidence),
        evidence=evidence,
    )


def replan(
    grid: Grid,
    start: Pose,
    goal: Pose,
    previous: Plan,
    *,
    executed: int,
    added_obstacles=(),
    max_expansions: Optional[int] = None,
) -> ReplanningResult:
    """执行旧路径到指定姿态后，合并一次静态障碍更新并重规划剩余路径。

    参数：
        grid: 旧规划使用的（不可变）网格。
        start: 旧规划的起点姿态。
        goal: 目标姿态（重规划仍以同一姿态为终点，必须精确到达）。
        previous: 旧规划结果（Plan）。其证据将被独立复核而非盲信。
        executed: 已执行的动作个数（0..len(previous.actions)）；当前位置 = 前缀末姿态。
        added_obstacles: 本次新增的障碍坐标对可迭代对象（只增不删）。
        max_expansions: 剩余路径 A* 的扩展预算；None 不限。

    校验失败一律抛 TypeError/ValueError，且不修改任何传入对象（Grid 不可变，
    更新后的网格是全新对象），因此不会留下半更新状态。

    返回 ReplanningResult，状态四选一，见 ReplanningStatus。
    """
    if not isinstance(grid, Grid):
        raise TypeError(f"grid 必须是 Grid，收到 {type(grid).__name__}: {grid!r}")
    if not isinstance(start, Pose):
        raise TypeError(f"start 必须是 Pose，收到 {type(start).__name__}: {start!r}")
    if not isinstance(goal, Pose):
        raise TypeError(f"goal 必须是 Pose，收到 {type(goal).__name__}: {goal!r}")
    if not isinstance(previous, Plan):
        raise TypeError(f"previous 必须是 Plan，收到 {type(previous).__name__}: {previous!r}")
    executed = require_int("executed", executed)
    if not 0 <= executed <= len(previous.actions):
        raise ValueError(
            f"executed={executed} 越界，允许范围 [0, {len(previous.actions)}]"
            "（旧规划动作数）"
        )
    if max_expansions is not None:
        max_expansions = require_int("max_expansions", max_expansions)
        if max_expansions < 0:
            raise ValueError(f"max_expansions 必须 >= 0，收到 {max_expansions}")

    # 旧网格上先核对起点与旧规划自洽（证据链与旧网格必须一致），不盲信 previous。
    grid.check_pose(start, name="start")
    grid.check_pose(goal, name="goal")
    _check_previous_on_old_grid(grid, start, goal, previous)

    # 合并障碍（构造式校验：任何一个坐标非法都不会产生新网格）。
    updated_grid = grid.with_added_obstacles(added_obstacles)

    # 当前姿态：旧证据链上第 executed 步的终点（executed=0 即起点）。
    current_pose = _nominal_pose(start, previous, executed)
    prefix_plan = _prefix_plan(previous, executed)

    # 已执行是历史：身后格再被堵不算碰撞；但机器人此刻占用的两格必须仍安全，
    # 否则更新与当前状态矛盾（PREFIX_BLOCKED）。
    if not updated_grid.is_pose_free(current_pose):
        return ReplanningResult(
            status=ReplanningStatus.PREFIX_BLOCKED,
            plan=None,
            prefix=prefix_plan,
            remaining=None,
            executed=executed,
            current_pose=current_pose,
            updated_grid=updated_grid,
            expansions=0,
            start=start,
            goal=goal,
        )

    # 目标姿态被新障碍压住：可证明不可达（网格尺寸不变，故无越界可能）。
    if not updated_grid.is_pose_free(goal):
        return ReplanningResult(
            status=ReplanningStatus.NO_PATH,
            plan=None,
            prefix=prefix_plan,
            remaining=None,
            executed=executed,
            current_pose=current_pose,
            updated_grid=updated_grid,
            expansions=0,
            start=start,
            goal=goal,
        )

    # 边界：已经到达目标（前缀走完且终点即目标）——空剩余段即最优。
    if current_pose == goal:
        return ReplanningResult(
            status=ReplanningStatus.OPTIMAL,
            plan=prefix_plan,
            prefix=prefix_plan,
            remaining=Plan(actions=(), total_cost=0, evidence=()),
            executed=executed,
            current_pose=current_pose,
            updated_grid=updated_grid,
            expansions=0,
            start=start,
            goal=goal,
        )

    # 从当前位置全量重新搜索剩余路径。
    search: SearchResult = plan(
        updated_grid, current_pose, goal, max_expansions=max_expansions
    )
    if search.status is SearchStatus.NO_PATH:
        return ReplanningResult(
            status=ReplanningStatus.NO_PATH,
            plan=None,
            prefix=prefix_plan,
            remaining=None,
            executed=executed,
            current_pose=current_pose,
            updated_grid=updated_grid,
            expansions=search.expansions,
            start=start,
            goal=goal,
        )
    if search.status is SearchStatus.BUDGET_EXHAUSTED:
        return ReplanningResult(
            status=ReplanningStatus.BUDGET_EXHAUSTED,
            plan=None,
            prefix=prefix_plan,
            remaining=None,
            executed=executed,
            current_pose=current_pose,
            updated_grid=updated_grid,
            expansions=search.expansions,
            start=start,
            goal=goal,
        )

    remaining = search.plan
    full = Plan(
        actions=prefix_plan.actions + remaining.actions,
        total_cost=prefix_plan.total_cost + remaining.total_cost,
        evidence=prefix_plan.evidence + remaining.evidence,
    )
    return ReplanningResult(
        status=ReplanningStatus.OPTIMAL,
        plan=full,
        prefix=prefix_plan,
        remaining=remaining,
        executed=executed,
        current_pose=current_pose,
        updated_grid=updated_grid,
        expansions=search.expansions,
        start=start,
        goal=goal,
    )


def _nominal_pose(start: Pose, previous: Plan, executed: int) -> Pose:
    """按旧证据推导第 executed 个姿态（不校验碰撞），仅用于错误定位。"""
    if executed == 0:
        return start
    return previous.evidence[executed - 1].to_pose


def _check_previous_on_old_grid(grid: Grid, start: Pose, goal: Pose, previous: Plan) -> None:
    """复核旧规划在旧网格上自洽：动作数=证据数、逐步扫掠通过且终点为 goal。"""
    if len(previous.actions) != len(previous.evidence):
        raise ValueError("previous.actions 与 previous.evidence 长度不一致，不是合法的 Plan")
    pose = start
    total = 0
    for i, e in enumerate(previous.evidence):
        if e.action != previous.actions[i]:
            raise ValueError(f"previous.evidence[{i}] 的动作与 actions[{i}] 不一致")
        if e.from_pose != pose:
            raise ValueError(f"previous.evidence[{i}] 的 from_pose 与前缀推导姿态不一致")
        result = transition(grid, pose, e.action)
        if result is None:
            raise ValueError(f"previous 的第 {i} 步在旧网格上扫掠不通过，旧规划本身不合法")
        nxt, swept = result
        if e.to_pose != nxt:
            raise ValueError(f"previous.evidence[{i}] 的 to_pose 与旧网格重算结果不一致")
        if tuple(sorted(e.swept_cells)) != tuple(sorted(swept)):
            raise ValueError(f"previous.evidence[{i}] 的 swept_cells 与旧网格重算结果不一致")
        pose = nxt
        total += e.cost
    if pose != goal:
        raise ValueError("previous 的终点与 goal 不一致，不能作为到达该 goal 的旧规划")
    if total != previous.total_cost:
        raise ValueError("previous.total_cost 与逐步代价之和不一致")
