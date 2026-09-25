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
from .model import (
    DEFAULT_WAIT_COST,
    Action,
    Grid,
    Plan,
    Pose,
    StepEvidence,
    TimedState,
    Timetable,
    successors,
    timed_successors,
)

_FORWARD_COST = Action.FORWARD.cost  # 10，曼哈顿每格的最小平移代价
_TURN_COST = Action.TURN_LEFT.cost  # 8，单次 90 度转向代价


class SearchStatus(enum.Enum):
    OPTIMAL = "optimal"  # 找到最优路径
    NO_PATH = "no_path"  # 已证明无解（开集耗尽）
    BUDGET_EXHAUSTED = "budget_exhausted"  # 搜索预算耗尽，既未找到也未证明无解
    # 仅 plan_timed：在给定时域内证明无路。时域外可能存在路径，
    # 因此绝不等同于静态几何无解，必须与此区分开报告。
    NO_PATH_WITHIN_HORIZON = "no_path_within_horizon"


@dataclass(frozen=True)
class SearchResult:
    """规划结果。status 为 OPTIMAL 时 plan 非空；其余情况 plan 为 None。

    arrival_time 仅 plan_timed 使用：到达目标的时刻（静态 plan 为 None）。
    """

    status: SearchStatus
    plan: Optional[Plan]
    expansions: int
    start: Pose
    goal: Pose
    arrival_time: Optional[int] = None

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
        evidence.append(StepEvidence(action=action, from_pose=pose, to_pose=nxt, swept_cells=swept, cost=action.cost))
        pose = nxt
    return pose, tuple(evidence)


# ---------------------------------------------------------------------------
# 离散时间扩展：有限时域 + 每时刻禁占格 + WAIT 原语
# ---------------------------------------------------------------------------
#
# 启发式可采纳性（时序情形）证明：
#   任取一条时域内合法时序路径，删去其中全部 WAIT 后，剩下的空间动作序列
#   每一步都通过了与静态 transition 相同的空间扫掠检查（timed_transition
#   先做空间检查再查时刻表），因此它本身就是一条合法静态路径，其代价
#   C_spatial 满足 h(start) <= C_static_optimal <= C_spatial。
#   又因 wait_cost 为正整数，时序路径总代价 C = C_spatial + wait_cost*等待
#   步数 >= C_spatial >= h(start)。故 h 对时序问题可采纳。
#   一致性：空间边的一致性同静态证明；WAIT 边 h 不变而代价为正，
#   h(s) <= wait_cost + h(s) 显然成立。两个分量取 max 保持上述性质，
#   因此带 closed 集（best_g 按 (姿态, 时刻) 键控）的 A* 返回最优解。
#
# 状态 = (x, y, heading, time)：同位置同朝向、不同时刻是不同状态，不合并。
# 目标在任一时刻 t <= horizon 到达即停止，不要求之后永久占住终点。


def plan_timed(
    grid: Grid,
    timetable: Timetable,
    start: Pose,
    goal: Pose,
    *,
    horizon: int,
    wait_cost: int = DEFAULT_WAIT_COST,
    max_expansions: Optional[int] = None,
) -> SearchResult:
    """离散时间规划：在 [0, horizon] 时域内从 start 到 goal（含目标朝向）。

    - 每个原语（含 WAIT）耗时 1 步；WAIT 代价为 wait_cost（正整数）。
    - 从时刻 t 执行的动作，其扫掠格必须全部避开 blocked[t] ∪ blocked[t+1]。
    - 返回 OPTIMAL 时 plan 最优：先按**总代价最小**，并列时取**到达时刻最早**
      （不同结构的等代价路径可能有不同步数，例如 F+B 与 24 个 WAIT 代价相同）。
      完整搜索（max_expansions=None）下该准则被严格满足；独立时序 Dijkstra
      参考采用同一准则（堆并列按 (x,y,h,t) 字典序，等价于同代价最早时刻）。
    - 时域内证明无路返回 NO_PATH_WITHIN_HORIZON——只断言“该时域内无路”，
      不断言静态几何无解；预算耗尽返回 BUDGET_EXHAUSTED，三者严格区分。
      预算耗尽前若已弹出过目标，其总代价仍可由一致性启发式证明最优
      （返回 OPTIMAL），但“并列最早时刻”在预算截断下不作保证。
    """
    if not isinstance(grid, Grid):
        raise TypeError(f"grid 必须是 Grid，收到 {type(grid).__name__}: {grid!r}")
    if not isinstance(timetable, Timetable):
        raise TypeError(f"timetable 必须是 Timetable，收到 {type(timetable).__name__}: {timetable!r}")
    if not isinstance(start, Pose):
        raise TypeError(f"start 必须是 Pose，收到 {type(start).__name__}: {start!r}")
    if not isinstance(goal, Pose):
        raise TypeError(f"goal 必须是 Pose，收到 {type(goal).__name__}: {goal!r}")
    horizon = require_int("horizon", horizon)
    if horizon < 0:
        raise ValueError(f"horizon 必须 >= 0，收到 {horizon}")
    wait_cost = require_int("wait_cost", wait_cost)
    if wait_cost < 1:
        raise ValueError(f"wait_cost 必须是正整数，收到 {wait_cost}")
    if max_expansions is not None:
        max_expansions = require_int("max_expansions", max_expansions)
        if max_expansions < 0:
            raise ValueError(f"max_expansions 必须 >= 0，收到 {max_expansions}")
    grid.check_pose(start, name="start")
    grid.check_pose(goal, name="goal")
    if timetable.horizon > horizon:
        raise ValueError(
            f"timetable 给出了时刻 {timetable.horizon} 的禁占格，但 horizon={horizon}："
            f"搜索只覆盖时刻 0..{horizon}（含最后一步所需的 blocked[{horizon}]），"
            f"超出时刻的数据不会生效，请核对时域或时刻表"
        )
    for t in range(0, timetable.horizon + 1):
        for cell in timetable.blocked_at(t):
            if not grid.in_bounds(cell):
                raise ValueError(
                    f"timetable 时刻 {t} 的禁占格 {cell} 越界（网格 {grid.width}x{grid.height}）"
                )
    for cell in grid.pose_cells(start):
        if timetable.is_blocked(cell, 0):
            raise ValueError(f"start 的占用格 {cell} 在时刻 0 被禁占: {start!r}")

    start_state = TimedState.from_pose(start, 0)
    if start == goal:
        return SearchResult(
            status=SearchStatus.OPTIMAL,
            plan=Plan(actions=(), total_cost=0, evidence=()),
            expansions=0,
            start=start,
            goal=goal,
            arrival_time=0,
        )

    # 堆元素: (f, g, 序号, 状态)。状态含时刻，同姿态不同时刻不合并。
    open_heap: list[tuple[int, int, int, TimedState]] = [(heuristic(start, goal), 0, 0, start_state)]
    best_g: Dict[TimedState, int] = {start_state: 0}
    # parent[state] = (父状态, 动作, 扫掠格, 动作代价)
    parent: Dict[TimedState, Tuple[TimedState, Action, Tuple[tuple[int, int], ...], int]] = {}
    counter = 1
    expansions = 0
    # 优化准则：总代价最小；并列时到达时刻最早。目标在任一时刻到达即停止扩展
    # 其自身，但继续收集同 f=最优代价 的其他时刻目标，直到堆顶 f 更大。
    best_goal_cost: Optional[int] = None
    best_goal_time: Optional[int] = None
    best_goal_state: Optional[TimedState] = None

    while open_heap:
        f, g, _seq, state = heapq.heappop(open_heap)
        if g > best_g.get(state, _INF):
            continue  # 过期堆项（不耗预算）
        if best_goal_cost is not None and f > best_goal_cost:
            break  # 一致性 h 下，其余节点 f 更大，不可能再有同代价目标
        if max_expansions is not None and expansions >= max_expansions:
            if best_goal_state is not None:
                break  # 已有被证明最优的候选，预算耗尽也照常返回
            return SearchResult(SearchStatus.BUDGET_EXHAUSTED, None, expansions, start, goal)
        expansions += 1
        if state.pose == goal:
            if (
                best_goal_cost is None
                or g < best_goal_cost
                or (g == best_goal_cost and state.time < best_goal_time)
            ):
                best_goal_cost, best_goal_time, best_goal_state = g, state.time, state
            continue  # 到达即停止：不再扩展目标自身
        if state.time >= horizon:
            continue  # 时域边界：不再产生后继
        for action, nxt, swept, cost in timed_successors(grid, timetable, state, wait_cost=wait_cost):
            g2 = g + cost
            if g2 < best_g.get(nxt, _INF):
                best_g[nxt] = g2
                parent[nxt] = (state, action, swept, cost)
                heapq.heappush(open_heap, (g2 + heuristic(nxt.pose, goal), g2, counter, nxt))
                counter += 1

    if best_goal_state is not None:
        return SearchResult(
            status=SearchStatus.OPTIMAL,
            plan=_reconstruct_timed(parent, start_state, best_goal_state, best_goal_cost),
            expansions=expansions,
            start=start,
            goal=goal,
            arrival_time=best_goal_time,
        )
    # 开集耗尽：只证明 [0, horizon] 时域内无路，不断言静态几何无解。
    return SearchResult(SearchStatus.NO_PATH_WITHIN_HORIZON, None, expansions, start, goal)


def _reconstruct_timed(
    parent: Dict[TimedState, Tuple[TimedState, Action, Tuple[tuple[int, int], ...], int]],
    start: TimedState,
    goal: TimedState,
    total_cost: int,
) -> Plan:
    chain: list[StepEvidence] = []
    state = goal
    while state != start:
        prev, action, swept, cost = parent[state]
        chain.append(
            StepEvidence(
                action=action,
                from_pose=prev.pose,
                to_pose=state.pose,
                swept_cells=swept,
                cost=cost,
                time=prev.time,
            )
        )
        state = prev
    chain.reverse()
    return Plan(
        actions=tuple(e.action for e in chain),
        total_cost=total_cost,
        evidence=tuple(chain),
    )


def replay_timed(
    grid: Grid,
    timetable: Timetable,
    start: Pose,
    actions: Tuple[Action, ...],
    *,
    wait_cost: int = DEFAULT_WAIT_COST,
) -> Tuple[TimedState, Tuple[StepEvidence, ...]]:
    """按动作序列从 (start, 时刻 0) 逐步重放，返回 (最终时序状态, 逐步证据)。

    每步重新执行空间扫掠与 blocked[t] ∪ blocked[t+1] 禁占检查；
    任一步不通过即抛 ReplayError（携带步序号与时刻，可定位）。
    """
    from .model import timed_transition

    if not isinstance(grid, Grid):
        raise TypeError(f"grid 必须是 Grid，收到 {type(grid).__name__}: {grid!r}")
    if not isinstance(timetable, Timetable):
        raise TypeError(f"timetable 必须是 Timetable，收到 {type(timetable).__name__}: {timetable!r}")
    if not isinstance(start, Pose):
        raise TypeError(f"start 必须是 Pose，收到 {type(start).__name__}: {start!r}")
    wait_cost = require_int("wait_cost", wait_cost)
    if wait_cost < 1:
        raise ValueError(f"wait_cost 必须是正整数，收到 {wait_cost}")
    grid.check_pose(start, name="start")
    state = TimedState.from_pose(start, 0)
    evidence: list[StepEvidence] = []
    for i, action in enumerate(actions):
        if not isinstance(action, Action):
            raise ReplayError(f"actions[{i}] 必须是 Action，收到 {type(action).__name__}: {action!r}")
        result = timed_transition(grid, timetable, state, action)
        if result is None:
            raise ReplayError(
                f"第 {i} 步 {action.name} 在时刻 {state.time} 姿态 {state.pose!r} 处不通过"
                f"（越界、碰障碍或扫掠格落入禁占格）"
            )
        nxt, swept = result
        cost = wait_cost if action is Action.WAIT else action.cost
        evidence.append(
            StepEvidence(
                action=action,
                from_pose=state.pose,
                to_pose=nxt.pose,
                swept_cells=swept,
                cost=cost,
                time=state.time,
            )
        )
        state = nxt
    return state, tuple(evidence)
