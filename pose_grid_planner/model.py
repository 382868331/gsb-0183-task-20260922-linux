"""公开数据结构：Grid / Heading / Pose / Action / 证据类型。

坐标约定：x 向右增，y 向下增（屏幕坐标）；Heading 顺序 NORTH→EAST→SOUTH→WEST
对应屏幕上的顺时针。机器人占用“参考格 (x,y)”与“沿 heading 的相邻一格”两格。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Optional, Tuple

from ._validation import require_int, require_int_cell, require_int_pair

MAX_GRID_SIDE = 20  # 题面上限：障碍格不超过 20×20

# WAIT（原地等待一个时刻）的默认代价。等待代价必须是正整数；plan_timed /
# replay_timed 可用 wait_cost 关键字指定其他正整数。
DEFAULT_WAIT_COST = 1


class Heading(enum.IntEnum):
    """四方向朝向。dx/dy 为沿朝向的位移。"""

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    @property
    def dx(self) -> int:
        return (0, 1, 0, -1)[int(self)]

    @property
    def dy(self) -> int:
        return (-1, 0, 1, 0)[int(self)]

    def turn_left(self) -> "Heading":
        return Heading((int(self) - 1) % 4)

    def turn_right(self) -> "Heading":
        return Heading((int(self) + 1) % 4)


class Action(enum.Enum):
    """运动原语。成员值仅作唯一标识；代价由 cost 属性给出（题面固定）。

    注意：不能用代价做枚举值——TURN_LEFT/TURN_RIGHT 代价相同会被 Enum
    当作别名而吞掉一个成员。
    """

    FORWARD = 0
    BACKWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    WAIT = 4  # 原地等待一个时刻：位置/朝向不变，耗时 1 步，代价为正整数

    @property
    def cost(self) -> int:
        return _ACTION_COSTS[self]


_ACTION_COSTS = {
    Action.FORWARD: 10,
    Action.BACKWARD: 14,
    Action.TURN_LEFT: 8,
    Action.TURN_RIGHT: 8,
    Action.WAIT: DEFAULT_WAIT_COST,
}


@dataclass(frozen=True)
class Pose:
    """A* 状态：参考格坐标 + 朝向。"""

    x: int
    y: int
    heading: Heading

    def __post_init__(self) -> None:
        require_int("Pose.x", self.x)
        require_int("Pose.y", self.y)
        if isinstance(self.heading, bool) or not isinstance(self.heading, Heading):
            raise TypeError(
                f"Pose.heading 必须是 Heading 枚举，收到 {type(self.heading).__name__}: {self.heading!r}"
            )

    def front_cell(self) -> tuple[int, int]:
        return (self.x + self.heading.dx, self.y + self.heading.dy)


@dataclass(frozen=True)
class StepEvidence:
    """单步扫掠证据：动作、起终姿态、本步检查过的全部格子、该步代价。

    time 为该步的起始时刻（动作从 time 执行到 time+1）；静态（无时刻表）
    接口下 time 为 None。
    """

    action: Action
    from_pose: Pose
    to_pose: Pose
    swept_cells: Tuple[tuple[int, int], ...]
    cost: int
    time: Optional[int] = None


@dataclass(frozen=True)
class Plan:
    """一条完整路径：动作序列、总代价、逐步扫掠证据。"""

    actions: Tuple[Action, ...]
    total_cost: int
    evidence: Tuple[StepEvidence, ...]

    @property
    def final_pose(self) -> Pose:
        if not self.evidence:
            raise ValueError("空路径没有 final_pose（起终相同，请直接使用起点姿态）")
        return self.evidence[-1].to_pose


@dataclass(frozen=True)
class TimedState:
    """时序搜索状态：位置 + 朝向 + 时刻。

    同一位置同一朝向、不同时刻是不同状态，不能合并——禁占格随时刻变化。
    """

    x: int
    y: int
    heading: Heading
    time: int

    def __post_init__(self) -> None:
        require_int("TimedState.x", self.x)
        require_int("TimedState.y", self.y)
        if isinstance(self.heading, bool) or not isinstance(self.heading, Heading):
            raise TypeError(
                f"TimedState.heading 必须是 Heading 枚举，收到 {type(self.heading).__name__}: {self.heading!r}"
            )
        require_int("TimedState.time", self.time)
        if self.time < 0:
            raise ValueError(f"TimedState.time 必须 >= 0，收到 {self.time}")

    @property
    def pose(self) -> Pose:
        return Pose(self.x, self.y, self.heading)

    @classmethod
    def from_pose(cls, pose: Pose, time: int) -> "TimedState":
        return cls(pose.x, pose.y, pose.heading, time)


class Timetable:
    """有限时域内各时刻的禁占格（临时障碍），不可变。

    blocked_by_time[t] 为时刻 t 禁占的格子集合；动作从 t 执行到 t+1 时，
    扫掠必须避开 blocked[t] ∪ blocked[t+1]。只给出显式列出的时刻，
    未列出的时刻没有禁占格（静态障碍仍由 Grid 负责）。
    """

    __slots__ = ("_blocked", "_horizon")

    def __init__(self, blocked_by_time: Mapping[int, Iterable[tuple[int, int]]]) -> None:
        if isinstance(blocked_by_time, (str, bytes)) or not isinstance(blocked_by_time, Mapping):
            raise TypeError(
                f"blocked_by_time 必须是 {{时刻: (x, y) 可迭代对象}} 的映射，"
                f"收到 {type(blocked_by_time).__name__}"
            )
        by_time: dict[int, frozenset[tuple[int, int]]] = {}
        horizon = -1
        for raw_t, cells in blocked_by_time.items():
            t = require_int("blocked_by_time 的时刻键", raw_t)
            if t < 0:
                raise ValueError(f"blocked_by_time 的时刻必须 >= 0，收到 {t}")
            if isinstance(cells, (str, bytes)) or not isinstance(cells, Iterable):
                raise TypeError(
                    f"blocked_by_time[{t}] 必须是 (x, y) 坐标对的可迭代对象，"
                    f"收到 {type(cells).__name__}"
                )
            bucket: set[tuple[int, int]] = set()
            for i, raw_cell in enumerate(cells):
                cell = require_int_pair(f"blocked_by_time[{t}][{i}]", raw_cell)
                if cell in bucket:
                    raise ValueError(
                        f"blocked_by_time[{t}][{i}]={cell} 在同一时刻重复（不做静默去重）"
                    )
                bucket.add(cell)
            if bucket:
                by_time[t] = frozenset(bucket)
                horizon = max(horizon, t)
        self._blocked = by_time
        self._horizon = horizon  # 出现过禁占格的最大时刻；无任何禁占格时为 -1

    @property
    def horizon(self) -> int:
        """出现过禁占格的最大时刻（空时刻表为 -1）。"""
        return self._horizon

    def blocked_at(self, time: int) -> frozenset[tuple[int, int]]:
        """某一时刻的禁占格集合（未列出的时刻为空集）。"""
        return self._blocked.get(time, frozenset())

    def is_blocked(self, cell: tuple[int, int], time: int) -> bool:
        return cell in self._blocked.get(time, frozenset())

    def sweep_blocked(
        self, swept: Iterable[tuple[int, int]], t: int
    ) -> Optional[tuple[tuple[int, int], int]]:
        """扫掠格与 blocked[t] ∪ blocked[t+1] 的第一个冲突 (格, 时刻)；无冲突返回 None。"""
        b0 = self._blocked.get(t, frozenset())
        b1 = self._blocked.get(t + 1, frozenset())
        for cell in swept:
            if cell in b0:
                return cell, t
            if cell in b1:
                return cell, t + 1
        return None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Timetable) and self._blocked == other._blocked

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._blocked.items())))

    def __repr__(self) -> str:
        return f"Timetable({ {t: sorted(c) for t, c in sorted(self._blocked.items())}!r})"


class Grid:
    """不可变障碍格。构造即完成全部校验；失败时不产生任何部分构造的对象。"""

    __slots__ = ("_width", "_height", "_obstacles")

    def __init__(self, width: int, height: int, obstacles: Iterable[tuple[int, int]] = ()) -> None:
        width = require_int("width", width)
        height = require_int("height", height)
        if not 1 <= width <= MAX_GRID_SIDE:
            raise ValueError(f"width={width} 超出允许范围 [1, {MAX_GRID_SIDE}]")
        if not 1 <= height <= MAX_GRID_SIDE:
            raise ValueError(f"height={height} 超出允许范围 [1, {MAX_GRID_SIDE}]")
        if isinstance(obstacles, (str, bytes)) or not isinstance(obstacles, Iterable):
            raise TypeError(f"obstacles 必须是 (x, y) 坐标对的可迭代对象，收到 {type(obstacles).__name__}")
        cells: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for i, raw in enumerate(obstacles):
            cell = require_int_cell(f"obstacles[{i}]", raw, width=width, height=height)
            if cell in seen:
                raise ValueError(f"obstacles[{i}]={cell} 与之前的障碍格重复（不做静默去重）")
            seen.add(cell)
            cells.append(cell)
        self._width = width
        self._height = height
        self._obstacles = frozenset(cells)

    @classmethod
    def from_rows(cls, rows: Iterable[str]) -> "Grid":
        """由字符串行构造：'#' 障碍，'.' 空地。行宽必须一致，字符仅限这两种。"""
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Iterable):
            raise TypeError(f"rows 必须是字符串的可迭代对象，收到 {type(rows).__name__}")
        rows = list(rows)
        if not rows:
            raise ValueError("rows 不能为空")
        for i, row in enumerate(rows):
            if not isinstance(row, str):
                raise TypeError(f"rows[{i}] 必须是 str，收到 {type(row).__name__}: {row!r}")
        height = len(rows)
        width = len(rows[0])
        if width == 0:
            raise ValueError("rows[0] 是空字符串，网格宽度至少为 1")
        obstacles: list[tuple[int, int]] = []
        for y, row in enumerate(rows):
            if len(row) != width:
                raise ValueError(f"rows[{y}] 宽度 {len(row)} 与 rows[0] 宽度 {width} 不一致")
            for x, ch in enumerate(row):
                if ch == "#":
                    obstacles.append((x, y))
                elif ch != ".":
                    raise ValueError(f"rows[{y}][{x}] 字符 {ch!r} 非法，只允许 '#' 和 '.'")
        return cls(width, height, obstacles)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def obstacles(self) -> frozenset[tuple[int, int]]:
        return self._obstacles

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self._width and 0 <= y < self._height

    def is_free(self, cell: tuple[int, int]) -> bool:
        return self.in_bounds(cell) and cell not in self._obstacles

    def pose_cells(self, pose: Pose) -> tuple[tuple[int, int], tuple[int, int]]:
        """姿态占用的两格：参考格 + 沿 heading 相邻格。"""
        if not isinstance(pose, Pose):
            raise TypeError(f"pose 必须是 Pose，收到 {type(pose).__name__}: {pose!r}")
        return ((pose.x, pose.y), pose.front_cell())

    def check_pose(self, pose: Pose, *, name: str = "pose") -> None:
        """姿态合法（占用格都在界内且无障碍）则返回，否则抛出可定位的 ValueError。"""
        for cell in self.pose_cells(pose):
            if not self.in_bounds(cell):
                raise ValueError(f"{name} 的占用格 {cell} 越界（网格 {self._width}x{self._height}）: {pose!r}")
            if cell in self._obstacles:
                raise ValueError(f"{name} 的占用格 {cell} 是障碍: {pose!r}")

    def is_pose_free(self, pose: Pose) -> bool:
        return all(self.is_free(c) for c in self.pose_cells(pose))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Grid)
            and self._width == other._width
            and self._height == other._height
            and self._obstacles == other._obstacles
        )

    def __hash__(self) -> int:
        return hash((self._width, self._height, self._obstacles))

    def __repr__(self) -> str:
        return f"Grid(width={self._width}, height={self._height}, obstacles={sorted(self._obstacles)!r})"


# 空间运动原语（不含 WAIT）：静态规划与静态 successors 只枚举这四个。
SPATIAL_ACTIONS: Tuple[Action, ...] = (
    Action.FORWARD,
    Action.BACKWARD,
    Action.TURN_LEFT,
    Action.TURN_RIGHT,
)


def transition(grid: Grid, pose: Pose, action: Action) -> Optional[tuple[Pose, tuple[tuple[int, int], ...]]]:
    """尝试执行一个原语（纯空间检查，不含时刻表）。

    返回 (下一姿态, 本步扫掠格) ；扫掠不通过（越界或碰障碍）返回 None。
    扫掠规则（题面契约）：
    - 平移（前进/后退）：起终占用格的并集必须全空；
    - 原地转向：包含起终全部占用格的最小轴对齐 2x2 矩形（4 格）必须全空；
    - 等待（WAIT）：姿态不变，车体当前占用的全部两格必须全空。
    """
    if not isinstance(grid, Grid):
        raise TypeError(f"grid 必须是 Grid，收到 {type(grid).__name__}: {grid!r}")
    if not isinstance(pose, Pose):
        raise TypeError(f"pose 必须是 Pose，收到 {type(pose).__name__}: {pose!r}")
    if not isinstance(action, Action):
        raise TypeError(f"action 必须是 Action，收到 {type(action).__name__}: {action!r}")

    h = pose.heading
    if action is Action.WAIT:
        nxt = pose
        swept = grid.pose_cells(pose)  # 等待也要检查车体全部格
    elif action is Action.FORWARD or action is Action.BACKWARD:
        sign = 1 if action is Action.FORWARD else -1
        nxt = Pose(pose.x + sign * h.dx, pose.y + sign * h.dy, h)
        swept = tuple(dict.fromkeys(grid.pose_cells(pose) + grid.pose_cells(nxt)))
    else:
        new_h = h.turn_left() if action is Action.TURN_LEFT else h.turn_right()
        nxt = Pose(pose.x, pose.y, new_h)
        # 起终占用格：ref, ref+d_old, ref+d_new；其最小轴对齐包围盒是 2x2，
        # 第 4 格为 ref+d_old+d_new。
        d_old = (h.dx, h.dy)
        d_new = (new_h.dx, new_h.dy)
        swept = (
            (pose.x, pose.y),
            (pose.x + d_old[0], pose.y + d_old[1]),
            (pose.x + d_new[0], pose.y + d_new[1]),
            (pose.x + d_old[0] + d_new[0], pose.y + d_old[1] + d_new[1]),
        )
    if all(grid.is_free(c) for c in swept):
        return nxt, swept
    return None


def successors(grid: Grid, pose: Pose) -> Iterator[tuple[Action, Pose, tuple[tuple[int, int], ...], int]]:
    """枚举姿态的全部合法空间后继：(动作, 下一姿态, 扫掠格, 代价)。

    只含四个空间原语；WAIT 只属于离散时间接口（见 timed_successors）。
    """
    for action in SPATIAL_ACTIONS:
        result = transition(grid, pose, action)
        if result is not None:
            nxt, swept = result
            yield action, nxt, swept, action.cost


def timed_transition(
    grid: Grid,
    timetable: Timetable,
    state: TimedState,
    action: Action,
) -> Optional[tuple[TimedState, tuple[tuple[int, int], ...]]]:
    """在时刻 state.time 尝试执行一个原语（耗时 1 步，含 WAIT）。

    先做与 transition 相同的空间扫掠检查，再要求扫掠格全部避开
    blocked[t] ∪ blocked[t+1]（t 为起始时刻）。任一不通过返回 None；
    通过则返回 (下一状态, 扫掠格)，下一状态时刻为 t+1。
    """
    if not isinstance(grid, Grid):
        raise TypeError(f"grid 必须是 Grid，收到 {type(grid).__name__}: {grid!r}")
    if not isinstance(timetable, Timetable):
        raise TypeError(f"timetable 必须是 Timetable，收到 {type(timetable).__name__}: {timetable!r}")
    if not isinstance(state, TimedState):
        raise TypeError(f"state 必须是 TimedState，收到 {type(state).__name__}: {state!r}")
    if not isinstance(action, Action):
        raise TypeError(f"action 必须是 Action，收到 {type(action).__name__}: {action!r}")

    result = transition(grid, state.pose, action)
    if result is None:
        return None
    nxt_pose, swept = result
    if timetable.sweep_blocked(swept, state.time) is not None:
        return None
    return TimedState.from_pose(nxt_pose, state.time + 1), swept


def timed_successors(
    grid: Grid,
    timetable: Timetable,
    state: TimedState,
    *,
    wait_cost: int = DEFAULT_WAIT_COST,
) -> Iterator[tuple[Action, TimedState, tuple[tuple[int, int], ...], int]]:
    """枚举时序状态的全部合法后继：(动作, 下一状态, 扫掠格, 代价)。

    四个空间原语之外增加 WAIT（代价 wait_cost，正整数）。
    """
    wait_cost = require_int("wait_cost", wait_cost)
    if wait_cost < 1:
        raise ValueError(f"wait_cost 必须是正整数，收到 {wait_cost}")
    for action in Action:
        result = timed_transition(grid, timetable, state, action)
        if result is not None:
            nxt, swept = result
            cost = wait_cost if action is Action.WAIT else action.cost
            yield action, nxt, swept, cost
