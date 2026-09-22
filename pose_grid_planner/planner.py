"""姿态格 A* 规划器。

状态为 (x, y, heading)；边代价取自运动原语（10/14/8）。

启发式（严格按题面，仅此两项取 max）：
    h = max(曼哈顿距离(参考格) * 10, 最少转向代价)
其中“最少转向代价”= 8 *（当前朝向到目标朝向所需的最少 90 度转向次数）。
平移代价以最小单位代价 10/格低估；转向以 8/次低估，且二者本可叠加，
取 max 仍是可采纳启发式。

搜索预算（节点扩展次数）耗尽时返回 BUDGET_EXHAUSTED，与已证明无解
NO_PATH 严格区分；预算默认对 20x20x4 的状态空间足够完成完整搜索。
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import Enum

from .grid import (
    ACTION_COST,
    Action,
    GridWorld,
    Heading,
    InvalidParameterError,
    Pose,
    Transition,
    _strict_int,
)

MAX_GRID = GridWorld.MAX_SIZE
# 全状态上限 20*20*4 = 1600；留出冗余，默认预算允许把可达状态全部扩展完。
BUDGET_DEFAULT = 100_000
_TURN_COST = ACTION_COST[Action.TURN_LEFT]
_STEP_COST = ACTION_COST[Action.FORWARD]


class PlanStatus(str, Enum):
    """规划结论。"""

    SUCCESS = "success"
    NO_PATH = "no_path"                 # 已完整搜索，证明无解
    BUDGET_EXHAUSTED = "budget_exhausted"  # 预算耗尽，不能判定是否有解


@dataclass(frozen=True, slots=True)
class StepEvidence:
    """逐步动作与扫掠证据。

    :param action: 执行的原语
    :param cost: 该步代价
    :param src_pose: 执行前姿态 (x, y, heading)
    :param dst_pose: 执行后姿态
    :param swept_cells: 该步必须为空的全部检查格（有序、去重）
    :param swept_kind: "union"（平移：起终占用并集）或
                       "2x2"（转向：最小轴对齐 2×2 矩形）
    """

    action: Action
    cost: int
    src_pose: tuple[int, int, Heading]
    dst_pose: tuple[int, int, Heading]
    swept_cells: tuple[tuple[int, int], ...]
    swept_kind: str


@dataclass(frozen=True, slots=True)
class PlanningResult:
    """规划结果。

    status == SUCCESS 时 actions/evidence 非空（起终不同）或为空（起终相同）；
    其余状态下 actions/evidence 恒为空元组，total_cost 恒为 None。
    """

    status: PlanStatus
    actions: tuple[Action, ...] = ()
    evidence: tuple[StepEvidence, ...] = ()
    total_cost: int | None = None
    start: tuple[int, int, Heading] | None = None
    goal: tuple[int, int, Heading] | None = None
    expanded: int = 0

    @property
    def found(self) -> bool:
        return self.status is PlanStatus.SUCCESS


def _turn_count(a: Heading, b: Heading) -> int:
    """两个朝向之间最少的 90 度转向次数（0、1 或 2）。"""
    diff = abs(int(a) - int(b)) & 3
    return min(diff, 4 - diff)


def heuristic(pose: Pose, goal: Pose) -> int:
    """公开启发式：max(曼哈顿*10, 最少转向代价)。"""
    manhattan = abs(pose.x - goal.x) + abs(pose.y - goal.y)
    return max(
        manhattan * _STEP_COST,
        _turn_count(pose.heading, goal.heading) * _TURN_COST,
    )


def _as_pose(value: object, name: str) -> Pose:
    if isinstance(value, Pose):
        return value
    if isinstance(value, bool):
        raise InvalidParameterError(f"{name} 不能为 bool")
    if not isinstance(value, (tuple, list)):
        raise InvalidParameterError(
            f"{name} 必须为 Pose 或 (x, y, heading)，得到 {type(value).__name__}"
        )
    if len(value) != 3:
        raise InvalidParameterError(f"{name} 长度必须为 3，得到 {len(value)}")
    x, y, h = value
    return Pose(_strict_int(x, f"{name}[0]"), _strict_int(y, f"{name}[1]"), h)


def plan(
    world: GridWorld,
    start: Pose | tuple[int, int, Heading],
    goal: Pose | tuple[int, int, Heading],
    *,
    max_expansions: int = BUDGET_DEFAULT,
) -> PlanningResult:
    """在 ``world`` 中规划从 start 到 goal 的最小代价动作序列。

    :param world: 网格世界
    :param start: 起始姿态，Pose 或 (x, y, heading)
    :param goal: 目标姿态（朝向必须精确匹配）
    :param max_expansions: 搜索预算（A* 节点扩展次数上限，>=1）；
        耗尽时返回 ``BUDGET_EXHAUSTED``，与证明无解的 ``NO_PATH`` 区分。
    :returns: PlanningResult；无解时 ``status == NO_PATH`` 且不抛异常。
    :raises InvalidParameterError: 标量参数非法（含 NaN/Infinity、bool）
    :raises InvalidPoseError: 起/终姿态越界或与障碍重叠
    """
    if not isinstance(world, GridWorld):
        raise InvalidParameterError(
            f"world 必须为 GridWorld，得到 {type(world).__name__}"
        )
    budget = _strict_int(max_expansions, "max_expansions")
    if budget < 1:
        raise InvalidParameterError(f"max_expansions 必须 >= 1，得到 {budget}")

    s = _as_pose(start, "start")
    g = _as_pose(goal, "goal")
    world.check_pose(s, "start")
    world.check_pose(g, "goal")

    start_state = s.state
    goal_state = g.state
    if start_state == goal_state:
        return PlanningResult(
            status=PlanStatus.SUCCESS,
            actions=(),
            evidence=(),
            total_cost=0,
            start=start_state,
            goal=goal_state,
            expanded=0,
        )

    # A*：f 优先；f 相同时按 g 大的优先（更接近目标），再按状态元组定序，
    # 使输出在同代价路径间也保持确定性。
    open_heap: list[tuple[int, int, int, tuple[int, int, Heading]]] = []
    counter = 0
    g_score: dict[tuple[int, int, Heading], int] = {start_state: 0}
    came_from: dict[
        tuple[int, int, Heading],
        tuple[tuple[int, int, Heading], Transition],
    ] = {}
    closed: set[tuple[int, int, Heading]] = set()

    h0 = heuristic(s, g)
    heapq.heappush(open_heap, (h0, 0, counter, start_state))

    expanded = 0
    status = PlanStatus.NO_PATH

    while open_heap:
        _, neg_g, _, state = heapq.heappop(open_heap)
        g_cur = -neg_g
        if g_cur != g_score.get(state):
            continue  # 过期条目（已找到更便宜路径）
        if state in closed:
            continue
        if state == goal_state:
            status = PlanStatus.SUCCESS
            break
        if expanded >= budget:
            status = PlanStatus.BUDGET_EXHAUSTED
            break

        expanded += 1
        closed.add(state)
        cur_pose = Pose(state[0], state[1], state[2])
        for tr in world.neighbors(cur_pose):
            nxt = tr.dst.state
            if nxt in closed:
                continue
            tentative = g_cur + tr.cost
            if tentative < g_score.get(nxt, math.inf):
                g_score[nxt] = tentative
                came_from[nxt] = (state, tr)
                f = tentative + heuristic(tr.dst, g)
                counter += 1
                # -tentative 作为次级键：f 相同优先扩展 g 更大者。
                heapq.heappush(open_heap, (f, -tentative, counter, nxt))

    if status is not PlanStatus.SUCCESS:
        return PlanningResult(
            status=status,
            start=start_state,
            goal=goal_state,
            expanded=expanded,
        )

    # 回溯
    chain: list[Transition] = []
    cur = goal_state
    while cur != start_state:
        prev, tr = came_from[cur]
        chain.append(tr)
        cur = prev
    chain.reverse()

    actions = tuple(tr.action for tr in chain)
    evidence = tuple(
        StepEvidence(
            action=tr.action,
            cost=tr.cost,
            src_pose=tr.src.state,
            dst_pose=tr.dst.state,
            swept_cells=tr.swept,
            swept_kind=tr.swept_kind,
        )
        for tr in chain
    )
    total = sum(tr.cost for tr in chain)
    return PlanningResult(
        status=PlanStatus.SUCCESS,
        actions=actions,
        evidence=evidence,
        total_cost=total,
        start=start_state,
        goal=goal_state,
        expanded=expanded,
    )
