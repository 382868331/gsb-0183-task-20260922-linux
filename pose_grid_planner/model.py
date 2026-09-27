"""公开数据结构：Grid / Heading / Pose / Action / 证据类型。

坐标约定：x 向右增，y 向下增（屏幕坐标）；Heading 顺序 NORTH→EAST→SOUTH→WEST
对应屏幕上的顺时针。机器人占用“参考格 (x,y)”与“沿 heading 的相邻一格”两格。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Optional, Tuple

from ._validation import require_int, require_int_cell

MAX_GRID_SIDE = 20  # 题面上限：障碍格不超过 20×20


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

    @property
    def cost(self) -> int:
        return _ACTION_COSTS[self]


_ACTION_COSTS = {
    Action.FORWARD: 10,
    Action.BACKWARD: 14,
    Action.TURN_LEFT: 8,
    Action.TURN_RIGHT: 8,
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
    """单步扫掠证据：动作、起终姿态、本步检查过的全部格子、该步代价。"""

    action: Action
    from_pose: Pose
    to_pose: Pose
    swept_cells: Tuple[tuple[int, int], ...]
    cost: int


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


class Grid:
    """不可变障碍格与地面通行代价。

    terrain_costs 给出自由格的非负整数附加代价；未列出的格子附加代价为 0。
    构造即完成全部校验；失败时不产生任何部分构造的对象。
    """

    __slots__ = ("_width", "_height", "_obstacles", "_terrain_costs")

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: Iterable[tuple[int, int]] = (),
        terrain_costs: Mapping[tuple[int, int], int] | Iterable[tuple[tuple[int, int], int]] = (),
    ) -> None:
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
        if isinstance(terrain_costs, (str, bytes)):
            raise TypeError(
                f"terrain_costs 必须是 {{cell: cost}} 映射或 (cell, cost) 可迭代对象，收到 {type(terrain_costs).__name__}"
            )
        terrain_items = terrain_costs.items() if isinstance(terrain_costs, Mapping) else terrain_costs
        if not isinstance(terrain_items, Iterable):
            raise TypeError(
                f"terrain_costs 必须是 {{cell: cost}} 映射或 (cell, cost) 可迭代对象，收到 {type(terrain_costs).__name__}"
            )
        terrain: dict[tuple[int, int], int] = {}
        for i, raw in enumerate(terrain_items):
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
                raise TypeError(
                    f"terrain_costs[{i}] 必须是 (cell, cost) 二元组，收到 {type(raw).__name__}: {raw!r}"
                )
            item = list(raw)
            if len(item) != 2:
                raise ValueError(f"terrain_costs[{i}] 必须恰好包含 cell 和 cost 两部分，收到 {len(item)} 个: {raw!r}")
            raw_cell, raw_cost = item
            cell = require_int_cell(f"terrain_costs[{i}][0]", raw_cell, width=width, height=height)
            cost = require_int(f"terrain_costs[{i}][1]", raw_cost)
            if cost < 0:
                raise ValueError(f"terrain_costs[{i}][1] 必须 >= 0，格子 {cell} 收到 {cost}")
            if cell in terrain:
                raise ValueError(f"terrain_costs[{i}] 的格子 {cell} 重复（不做静默覆盖）")
            if cell in seen:
                raise ValueError(f"terrain_costs[{i}] 的格子 {cell} 同时是障碍格（障碍不可通行）")
            terrain[cell] = cost
        self._width = width
        self._height = height
        self._obstacles = frozenset(cells)
        self._terrain_costs = dict(sorted(terrain.items()))

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

    @property
    def terrain_costs(self) -> Mapping[tuple[int, int], int]:
        """返回已配置地面附加代价的副本；未包含的格子代价为 0。"""
        return dict(self._terrain_costs)

    def terrain_cost(self, cell: tuple[int, int]) -> int:
        """返回单格地面附加代价；越界格调用方应先通过扫掠检查。"""
        return self._terrain_costs.get(cell, 0)

    def step_cost(self, action: Action, swept_cells: Iterable[tuple[int, int]]) -> int:
        """计算一步代价：动作基础代价 + 扫掠过的不同格子的地面附加代价。

        同一个格子在单步扫掠集合中只计一次；在连续多步中重复占据则每步分别计费。
        """
        if not isinstance(action, Action):
            raise TypeError(f"action 必须是 Action，收到 {type(action).__name__}: {action!r}")
        if isinstance(swept_cells, (str, bytes)) or not isinstance(swept_cells, Iterable):
            raise TypeError(f"swept_cells 必须是格子可迭代对象，收到 {type(swept_cells).__name__}")
        return action.cost + sum(self._terrain_costs.get(cell, 0) for cell in dict.fromkeys(swept_cells))

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
            and self._terrain_costs == other._terrain_costs
        )

    def __hash__(self) -> int:
        return hash((self._width, self._height, self._obstacles, tuple(self._terrain_costs.items())))

    def __repr__(self) -> str:
        return (
            f"Grid(width={self._width}, height={self._height}, "
            f"obstacles={sorted(self._obstacles)!r}, terrain_costs={self._terrain_costs!r})"
        )


def transition(grid: Grid, pose: Pose, action: Action) -> Optional[tuple[Pose, tuple[tuple[int, int], ...]]]:
    """尝试执行一个原语。

    返回 (下一姿态, 本步扫掠格) ；扫掠不通过（越界或碰障碍）返回 None。
    扫掠规则（题面契约）：
    - 平移（前进/后退）：起终占用格的并集必须全空；
    - 原地转向：包含起终全部占用格的最小轴对齐 2x2 矩形（4 格）必须全空。
    """
    if not isinstance(grid, Grid):
        raise TypeError(f"grid 必须是 Grid，收到 {type(grid).__name__}: {grid!r}")
    if not isinstance(pose, Pose):
        raise TypeError(f"pose 必须是 Pose，收到 {type(pose).__name__}: {pose!r}")
    if not isinstance(action, Action):
        raise TypeError(f"action 必须是 Action，收到 {type(action).__name__}: {action!r}")

    h = pose.heading
    if action is Action.FORWARD or action is Action.BACKWARD:
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
    """枚举姿态的全部合法后继：(动作, 下一姿态, 扫掠格, 代价)。"""
    for action in Action:
        result = transition(grid, pose, action)
        if result is not None:
            nxt, swept = result
            yield action, nxt, swept, grid.step_cost(action, swept)
