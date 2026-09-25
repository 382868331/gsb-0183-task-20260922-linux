"""离散时间姿态格规划：有限时域 + 每时刻禁占格 + WAIT 原语。

契约（相对无时间静态接口的增量）：
- 输入给出有限时域 [0, horizon] 内每个时刻的禁占格集合（动态障碍）。
- 状态 = (Pose, t)：位置、朝向、时刻三元组；同地点不同时间是不同状态，绝不合并。
- 每个原动作耗时一步；新增 WAIT 也耗时一步，代价为正整数（wait_cost 参数）。
- 时刻 t 出发的动作，其扫掠格必须避开 forbidden(t) ∪ forbidden(t+1)（起止两时刻
  禁占格的并集），且仍须避开静态障碍与边界；WAIT 的扫掠 = 当前车体全部两格。
- 目标到达（弹出 goal 姿态）即停止，不要求之后永久占住终点；到达时刻可等于 horizon。
- 时域内开集耗尽只说明“该时域内无路”（NO_PATH_IN_HORIZON），不能据此断言
  静态几何无解——静态可行性须用无时间的 plan() 另行判定。

启发式可采纳性（证明见 README“离散时间扩展”一节）：
h = max(曼哈顿*10, 最少转向*8) 在加入 WAIT（正代价）与动态禁占（只删转移、
不降低任何路径代价）后仍可采纳且一致，因此带 closed 集的 A* 弹出目标即最优。
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

from ._validation import require_int, require_int_cell
from .model import _MOTION_ACTIONS, Action, Grid, Pose, _sweep
from .planner import ReplayError, SearchStatus, heuristic

# 时域上界的接口边界：状态数为 姿态数(<=1600) x (horizon+1)，超出即拒绝，
# 避免时间展开图无限膨胀（题面未给时域上限，此为库的设计取舍，见 README）。
MAX_HORIZON = 4096

_EMPTY: frozenset = frozenset()


class TimedGrid:
    """静态网格 + 有限时域内每时刻的禁占格集合。构造即完成全部校验。"""

    __slots__ = ("_grid", "_horizon", "_blocked")

    def __init__(
        self,
        grid: Grid,
        horizon: int,
        blocked: Optional[Mapping[int, Iterable[tuple[int, int]]]] = None,
    ) -> None:
        if not isinstance(grid, Grid):
            raise TypeError(f"grid 必须是 Grid，收到 {type(grid).__name__}: {grid!r}")
        horizon = require_int("horizon", horizon)
        if not 0 <= horizon <= MAX_HORIZON:
            raise ValueError(f"horizon={horizon} 超出允许范围 [0, {MAX_HORIZON}]")
        parsed: Dict[int, frozenset] = {}
        if blocked is not None:
            if isinstance(blocked, (str, bytes)) or not isinstance(blocked, Mapping):
                raise TypeError(
                    f"blocked 必须是 {{时刻: 禁占格可迭代对象}} 的映射，收到 {type(blocked).__name__}"
                )
            for raw_t, cells in blocked.items():
                t = require_int("blocked 的时刻键", raw_t)
                if not 0 <= t <= horizon:
                    raise ValueError(f"blocked 的时刻键 {t} 超出时域 [0, {horizon}]")
                if isinstance(cells, (str, bytes)) or not isinstance(cells, Iterable):
                    raise TypeError(
                        f"blocked[{t}] 必须是 (x, y) 坐标对的可迭代对象，收到 {type(cells).__name__}"
                    )
                seen: set[tuple[int, int]] = set()
                for i, raw in enumerate(cells):
                    cell = require_int_cell(f"blocked[{t}][{i}]", raw, width=grid.width, height=grid.height)
                    if cell in seen:
                        raise ValueError(f"blocked[{t}][{i}]={cell} 与同时刻之前的禁占格重复（不做静默去重）")
                    seen.add(cell)
                parsed[t] = frozenset(seen)
        self._grid = grid
        self._horizon = horizon
        self._blocked = parsed

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def blocked(self) -> Dict[int, frozenset]:
        """时刻 -> 禁占格集合 的副本（缺省时刻语义为空集）。"""
        return dict(self._blocked)

    def blocked_at(self, time: int) -> frozenset:
        time = require_int("time", time)
        if not 0 <= time <= self._horizon:
            raise ValueError(f"time={time} 超出时域 [0, {self._horizon}]")
        return self._blocked.get(time, _EMPTY)

    def __repr__(self) -> str:
        return f"TimedGrid(grid={self._grid!r}, horizon={self._horizon}, blocked={self._blocked!r})"


def _require_wait_cost(wait_cost: int) -> int:
    wait_cost = require_int("wait_cost", wait_cost)
    if wait_cost < 1:
        raise ValueError(f"wait_cost 必须是正整数，收到 {wait_cost}")
    return wait_cost


def _check_timed_args(tgrid: TimedGrid, pose: Pose, time: int, action: Action) -> int:
    if not isinstance(tgrid, TimedGrid):
        raise TypeError(f"tgrid 必须是 TimedGrid，收到 {type(tgrid).__name__}: {tgrid!r}")
    if not isinstance(pose, Pose):
        raise TypeError(f"pose 必须是 Pose，收到 {type(pose).__name__}: {pose!r}")
    if not isinstance(action, Action):
        raise TypeError(f"action 必须是 Action，收到 {type(action).__name__}: {action!r}")
    time = require_int("time", time)
    if not 0 <= time < tgrid.horizon:
        raise ValueError(
            f"出发时刻 time={time} 非法：动作耗时一步，出发时刻须在 [0, {tgrid.horizon - 1}]"
            f"（到达时刻不超过 horizon={tgrid.horizon}）"
        )
    return time


def timed_transition(
    tgrid: TimedGrid,
    pose: Pose,
    time: int,
    action: Action,
    *,
    wait_cost: int = 1,
) -> Optional[tuple[Pose, tuple[tuple[int, int], ...], int]]:
    """时刻 time 出发执行一个原语（耗时一步，到达 time+1）。

    返回 (下一姿态, 扫掠格, 该步代价)；扫掠不通过返回 None。
    通过条件：扫掠格全部在界内、非静态障碍，且不属于 forbidden(time) ∪ forbidden(time+1)。
    WAIT：姿态不变，扫掠 = 当前车体全部两格，代价 = wait_cost。
    """
    time = _check_timed_args(tgrid, pose, time, action)
    wait_cost = _require_wait_cost(wait_cost)
    grid = tgrid.grid
    nxt, swept = _sweep(grid, pose, action)
    forbidden = tgrid._blocked.get(time, _EMPTY) | tgrid._blocked.get(time + 1, _EMPTY)
    if all(grid.is_free(c) and c not in forbidden for c in swept):
        cost = wait_cost if action is Action.WAIT else action.cost
        return nxt, swept, cost
    return None


def timed_successors(
    tgrid: TimedGrid,
    pose: Pose,
    time: int,
    *,
    wait_cost: int = 1,
) -> Iterator[tuple[Action, Pose, Tuple[tuple[int, int], ...], int]]:
    """枚举 (pose, time) 的全部合法后继：(动作, 下一姿态, 扫掠格, 代价)。"""
    for action in (*_MOTION_ACTIONS, Action.WAIT):
        result = timed_transition(tgrid, pose, time, action, wait_cost=wait_cost)
        if result is not None:
            nxt, swept, cost = result
            yield action, nxt, swept, cost


@dataclass(frozen=True)
class TimedStepEvidence:
    """单步扫掠证据（含时刻）：动作、起止姿态与时刻、本步检查过的全部格子、该步代价。"""

    action: Action
    from_pose: Pose
    from_time: int
    to_pose: Pose
    to_time: int
    swept_cells: Tuple[tuple[int, int], ...]
    cost: int


@dataclass(frozen=True)
class TimedPlan:
    """一条完整时间路径：动作序列、总代价、到达时刻、逐步扫掠证据。"""

    actions: Tuple[Action, ...]
    total_cost: int
    arrival_time: int
    evidence: Tuple[TimedStepEvidence, ...]

    @property
    def final_pose(self) -> Pose:
        if not self.evidence:
            raise ValueError("空路径没有 final_pose（起终相同，请直接使用起点姿态）")
        return self.evidence[-1].to_pose


@dataclass(frozen=True)
class TimedSearchResult:
    """timed_plan 的结果。status 为 OPTIMAL 时 plan 非空；其余情况 plan 为 None。"""

    status: SearchStatus
    plan: Optional[TimedPlan]
    expansions: int
    start: Pose
    goal: Pose
    horizon: int
    wait_cost: int

    @property
    def found(self) -> bool:
        return self.status is SearchStatus.OPTIMAL


def timed_plan(
    tgrid: TimedGrid,
    start: Pose,
    goal: Pose,
    *,
    wait_cost: int = 1,
    max_expansions: Optional[int] = None,
) -> TimedSearchResult:
    """在 tgrid 上从 (start, t=0) 规划到 goal（含目标朝向，到达时刻 <= horizon）。

    - SearchStatus.OPTIMAL：plan 为最优 TimedPlan（总代价最小；到达即停）。
    - SearchStatus.NO_PATH_IN_HORIZON：时域内开集耗尽——只证明该时域内无路，
      不代表静态几何无解。
    - SearchStatus.BUDGET_EXHAUSTED：max_expansions 预算耗尽，与上两者严格区分。
    """
    if not isinstance(tgrid, TimedGrid):
        raise TypeError(f"tgrid 必须是 TimedGrid，收到 {type(tgrid).__name__}: {tgrid!r}")
    if not isinstance(start, Pose):
        raise TypeError(f"start 必须是 Pose，收到 {type(start).__name__}: {start!r}")
    if not isinstance(goal, Pose):
        raise TypeError(f"goal 必须是 Pose，收到 {type(goal).__name__}: {goal!r}")
    wait_cost = _require_wait_cost(wait_cost)
    if max_expansions is not None:
        max_expansions = require_int("max_expansions", max_expansions)
        if max_expansions < 0:
            raise ValueError(f"max_expansions 必须 >= 0，收到 {max_expansions}")
    grid = tgrid.grid
    grid.check_pose(start, name="start")
    grid.check_pose(goal, name="goal")
    blocked0 = tgrid._blocked.get(0, _EMPTY)
    for cell in grid.pose_cells(start):
        if cell in blocked0:
            raise ValueError(f"start 的占用格 {cell} 在时刻 0 被禁占: {start!r}")

    if start == goal:
        return TimedSearchResult(
            status=SearchStatus.OPTIMAL,
            plan=TimedPlan(actions=(), total_cost=0, arrival_time=0, evidence=()),
            expansions=0,
            start=start,
            goal=goal,
            horizon=tgrid.horizon,
            wait_cost=wait_cost,
        )

    # 堆元素: (f, g, 序号, 姿态, 时刻)。状态 = (姿态, 时刻)，同姿态不同时刻不合并。
    open_heap: list[tuple[int, int, int, Pose, int]] = [(heuristic(start, goal), 0, 0, start, 0)]
    best_g: Dict[tuple[Pose, int], int] = {(start, 0): 0}
    # parent[(pose, t)] = ((父姿态, 父时刻), 动作, 扫掠格, 动作代价)
    parent: Dict[tuple[Pose, int], Tuple[tuple[Pose, int], Action, Tuple[tuple[int, int], ...], int]] = {}
    counter = 1
    expansions = 0
    horizon = tgrid.horizon

    while open_heap:
        if max_expansions is not None and expansions >= max_expansions:
            return TimedSearchResult(
                SearchStatus.BUDGET_EXHAUSTED, None, expansions, start, goal, horizon, wait_cost
            )
        _f, g, _seq, pose, t = heapq.heappop(open_heap)
        if g > best_g.get((pose, t), _INF):
            continue  # 过期堆项
        expansions += 1
        if pose == goal:
            return TimedSearchResult(
                status=SearchStatus.OPTIMAL,
                plan=_reconstruct(parent, start, goal, t, g),
                expansions=expansions,
                start=start,
                goal=goal,
                horizon=horizon,
                wait_cost=wait_cost,
            )
        if t >= horizon:
            continue  # 到达时刻不得超过 horizon，时刻 horizon 不可再出发
        for action, nxt, swept, cost in timed_successors(tgrid, pose, t, wait_cost=wait_cost):
            g2 = g + cost
            key = (nxt, t + 1)
            if g2 < best_g.get(key, _INF):
                best_g[key] = g2
                parent[key] = ((pose, t), action, swept, cost)
                heapq.heappush(open_heap, (g2 + heuristic(nxt, goal), g2, counter, nxt, t + 1))
                counter += 1
    return TimedSearchResult(
        SearchStatus.NO_PATH_IN_HORIZON, None, expansions, start, goal, horizon, wait_cost
    )


_INF = float("inf")


def _reconstruct(
    parent: Dict[tuple[Pose, int], Tuple[tuple[Pose, int], Action, Tuple[tuple[int, int], ...], int]],
    start: Pose,
    goal: Pose,
    goal_time: int,
    total_cost: int,
) -> TimedPlan:
    chain: list[TimedStepEvidence] = []
    key = (goal, goal_time)
    while key != (start, 0):
        (prev_pose, prev_t), action, swept, cost = parent[key]
        pose, t = key
        chain.append(
            TimedStepEvidence(
                action=action,
                from_pose=prev_pose,
                from_time=prev_t,
                to_pose=pose,
                to_time=t,
                swept_cells=swept,
                cost=cost,
            )
        )
        key = (prev_pose, prev_t)
    chain.reverse()
    return TimedPlan(
        actions=tuple(e.action for e in chain),
        total_cost=total_cost,
        arrival_time=goal_time,
        evidence=tuple(chain),
    )


def timed_replay(
    tgrid: TimedGrid,
    start: Pose,
    actions: Tuple[Action, ...],
    *,
    wait_cost: int = 1,
) -> Tuple[Pose, int, Tuple[TimedStepEvidence, ...]]:
    """按动作序列从 (start, t=0) 逐步重放，返回 (最终姿态, 到达时刻, 逐步证据)。

    任一步扫掠不通过（含 WAIT）或出发时刻超出时域即抛 ReplayError（带步序号，可定位）。
    """
    if not isinstance(tgrid, TimedGrid):
        raise TypeError(f"tgrid 必须是 TimedGrid，收到 {type(tgrid).__name__}: {tgrid!r}")
    if not isinstance(start, Pose):
        raise TypeError(f"start 必须是 Pose，收到 {type(start).__name__}: {start!r}")
    wait_cost = _require_wait_cost(wait_cost)
    tgrid.grid.check_pose(start, name="start")
    blocked0 = tgrid._blocked.get(0, _EMPTY)
    for cell in tgrid.grid.pose_cells(start):
        if cell in blocked0:
            raise ReplayError(f"start 的占用格 {cell} 在时刻 0 被禁占: {start!r}")
    pose = start
    evidence: list[TimedStepEvidence] = []
    for i, action in enumerate(actions):
        if not isinstance(action, Action):
            raise ReplayError(f"actions[{i}] 必须是 Action，收到 {type(action).__name__}: {action!r}")
        if i >= tgrid.horizon:
            raise ReplayError(
                f"第 {i} 步 {action.name} 的出发时刻 {i} 超出时域（到达时刻 {i + 1} > horizon={tgrid.horizon}）"
            )
        result = timed_transition(tgrid, pose, i, action, wait_cost=wait_cost)
        if result is None:
            raise ReplayError(
                f"第 {i} 步 {action.name} 在姿态 {pose!r}、时刻 {i} 处扫掠不通过"
                f"（越界、碰静态障碍或触及时刻 {i}/{i + 1} 的禁占格）"
            )
        nxt, swept, cost = result
        evidence.append(
            TimedStepEvidence(
                action=action,
                from_pose=pose,
                from_time=i,
                to_pose=nxt,
                to_time=i + 1,
                swept_cells=swept,
                cost=cost,
            )
        )
        pose = nxt
    return pose, len(actions), tuple(evidence)
