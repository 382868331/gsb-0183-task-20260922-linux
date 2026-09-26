"""公开数据结构：Grid / Heading / Pose / Action / 证据类型。

坐标约定：x 向右增，y 向下增（屏幕坐标）；Heading 顺序 NORTH→EAST→SOUTH→WEST
对应屏幕上的顺时针。机器人占用“参考格 (x,y)”与“沿 heading 的相邻一格”两格。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Tuple

from ._validation import require_int, require_int_cell

MAX_GRID_SIDE = 20  # 题面上限：障碍格不超过 20×20
MAX_ARTICULATION_DEGREES = 90  # 牵引车与拖车朝向的最大铰接角（格点化四方向下即 0/90，180 禁止）


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
            yield action, nxt, swept, action.cost


@dataclass(frozen=True)
class RigState:
    """带拖车的全状态：牵引车参考格 (x,y)、牵引车朝向、拖车朝向。

    几何约定（题内无侧滑单车轴拖车，格点化）：
    - 铰接点（挂钩）= 牵引车参考格 (x, y)，连接杆长度恰为 1 格；
    - 拖车车轴中心 = 铰接点沿拖车朝向反方向 1 格，即
      (x - trailer_heading.dx, y - trailer_heading.dy)；
    - 拖车占用车轴格与沿拖车朝向的相邻格（即挂钩格），共两格，
      因此连接杆始终连接两个轴对齐相邻格，本身也必须不穿障碍；
    - 铰接角 = 牵引车朝向与拖车朝向的夹角，四方向离散下只允许 0°/90°，
      180°（折叠）超过 MAX_ARTICULATION_DEGREES，直接不可用。
    """

    x: int
    y: int
    tractor_heading: Heading
    trailer_heading: Heading

    def __post_init__(self) -> None:
        require_int("RigState.x", self.x)
        require_int("RigState.y", self.y)
        if isinstance(self.tractor_heading, bool) or not isinstance(self.tractor_heading, Heading):
            raise TypeError(
                "RigState.tractor_heading 必须是 Heading 枚举，收到 "
                f"{type(self.tractor_heading).__name__}: {self.tractor_heading!r}"
            )
        if isinstance(self.trailer_heading, bool) or not isinstance(self.trailer_heading, Heading):
            raise TypeError(
                "RigState.trailer_heading 必须是 Heading 枚举，收到 "
                f"{type(self.trailer_heading).__name__}: {self.trailer_heading!r}"
            )

    @property
    def tractor_pose(self) -> Pose:
        return Pose(self.x, self.y, self.tractor_heading)

    @property
    def hitch(self) -> tuple[int, int]:
        return (self.x, self.y)

    @property
    def trailer_axle(self) -> tuple[int, int]:
        return (self.x - self.trailer_heading.dx, self.y - self.trailer_heading.dy)

    def articulation_degrees(self) -> int:
        """牵引车与拖车朝向的最小夹角（0/90/180）。"""
        diff = abs(int(self.tractor_heading) - int(self.trailer_heading)) % 4
        if diff == 3:
            diff = 1
        return diff * 90

    def trailer_cells(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """拖车占用的两格：车轴格 + 挂钩格。"""
        return (self.trailer_axle, self.hitch)

    def drawbar_cells(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """连接杆扫过的格段：四方向下连接杆长 1 格，恰为（车轴格, 挂钩格）。"""
        return self.trailer_cells()


def _bbox_cells(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return tuple((x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))


def _trailer_sweep(
    old_axle: tuple[int, int],
    old_hitch: tuple[int, int],
    new_axle: tuple[int, int],
    new_hitch: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """拖车两段车体（车轴格/挂钩格）起终占用的保守扫掠。

    两格各自平移 1 格时取起终并集；车轴绕挂钩做 90° 折叠/展开时，
    以四个角点（旧车轴、旧挂钩=新车轴格之一、新车轴、新挂钩）的最小
    轴对齐包围盒作为保守扫掠，覆盖连续的扇形扫掠所经格。
    """
    corners = (old_axle, old_hitch, new_axle, new_hitch)
    return _bbox_cells(corners)


def _drawbar_sweep(
    old_axle: tuple[int, int],
    old_hitch: tuple[int, int],
    new_axle: tuple[int, int],
    new_hitch: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """连接杆扫掠：起终两姿态连接杆格段并集的保守包围盒。

    平移拉直时即两格；折叠/展开时连接杆绕挂钩转 90°，用 2×2 包围盒
    覆盖整根杆，保证不穿障碍。
    """
    return _bbox_cells((old_axle, old_hitch, new_axle, new_hitch))


def rig_transition(
    grid: Grid,
    state: RigState,
    action: Action,
) -> Optional[
    tuple[
        RigState,
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int], ...],
    ]
]:
    """在带拖车状态上执行一个牵引车原语。

    牵引车运动沿用既有原语（前进/后退/原地左右转，代价不变）；拖车角按
    无侧滑几何做格点化更新（见模块说明）。返回
    (新状态, 牵引车扫掠格, 拖车扫掠格, 连接杆扫掠格)；
    铰接超限、拖车/连接杆越界或碰障碍时返回 None。

    平移的无侧滑更新（挂钩位移 d 为 ±1 格）：
    设 s=+1 前进、s=-1 后退（相对牵引车朝向），θ 为拖车朝向，
    m 为拖车车轴沿 θ 的有向位移（格），由
    d = m·u(θ) - (m+s)·u(θ') 反解 θ' 与 m，只接受四方向离散解、|m|≤1，
    且 |m+s|≤1（新连接杆仍长 1 格），再施加铰接角上限。
    """
    if not isinstance(grid, Grid):
        raise TypeError(f"grid 必须是 Grid，收到 {type(grid).__name__}: {grid!r}")
    if not isinstance(state, RigState):
        raise TypeError(f"state 必须是 RigState，收到 {type(state).__name__}: {state!r}")
    if not isinstance(action, Action):
        raise TypeError(f"action 必须是 Action，收到 {type(action).__name__}: {action!r}")

    h = state.tractor_heading
    theta = state.trailer_heading
    tractor_result = transition(grid, state.tractor_pose, action)
    if tractor_result is None:
        return None
    new_tractor_pose, tractor_swept = tractor_result

    old_hitch = state.hitch
    old_axle = state.trailer_axle
    candidates: list[tuple[Heading, tuple[int, int]]] = []

    if action in (Action.FORWARD, Action.BACKWARD):
        s = 1 if action is Action.FORWARD else -1
        dx = new_tractor_pose.x - state.x
        dy = new_tractor_pose.y - state.y
        # 枚举新拖车朝向（四方向），按无侧滑方程反解车轴位移 m。
        for new_theta in Heading:
            ux, uy = new_theta.dx, new_theta.dy
            kx, ky = ux * theta.dx, uy * theta.dy
            # d = m·u(θ) - (m+s)·u(θ')；沿新 θ' 投影求 m
            dot = dx * ux + dy * uy
            # d·u(θ') = m·u(θ)·u(θ') - (m+s)
            proj_old = theta.dx * ux + theta.dy * uy  # ±1 或 0
            denom = proj_old - 1
            if denom == 0:
                continue
            if (dot + s) % (-denom) != 0:
                continue
            m = -(dot + s) // denom
            if abs(m) > 1 or abs(m + s) > 1:
                continue
            ax = state.x - theta.dx + m * theta.dx
            ay = state.y - theta.dy + m * theta.dy
            new_hitch = new_tractor_pose.x, new_tractor_pose.y
            new_axle = (new_hitch[0] - new_theta.dx, new_hitch[1] - new_theta.dy)
            if (ax, ay) != new_axle:
                continue
            candidates.append((new_theta, new_axle))
    else:
        # 原地转向：挂钩不动，车轴不滑移（最小位移解），拖车朝向不变。
        new_axle = old_axle
        candidates.append((theta, new_axle))

    new_hitch = (new_tractor_pose.x, new_tractor_pose.y)
    valid: list[
        tuple[
            RigState,
            tuple[tuple[int, int], ...],
            tuple[tuple[int, int], ...],
            tuple[tuple[int, int], ...],
        ]
    ] = []
    seen: set[RigState] = set()
    for new_theta, new_axle_cell in candidates:
        nxt = RigState(new_hitch[0], new_hitch[1], new_tractor_pose.heading, new_theta)
        if nxt in seen:
            continue
        if nxt.articulation_degrees() > MAX_ARTICULATION_DEGREES:
            continue
        trailer_swept = _trailer_sweep(old_axle, old_hitch, new_axle_cell, new_hitch)
        drawbar_swept = _drawbar_sweep(old_axle, old_hitch, new_axle_cell, new_hitch)
        # 牵引车扫掠沿用单车规则；拖车与连接杆各自保守检查。
        if all(grid.is_free(c) for c in tractor_swept) \
                and all(grid.is_free(c) for c in trailer_swept) \
                and all(grid.is_free(c) for c in drawbar_swept):
            seen.add(nxt)
            valid.append((nxt, tractor_swept, trailer_swept, drawbar_swept))
    # 一个原语在格点化无侧滑模型下至多一个合法离散后继；枚举仅用于方程求解。
    if not valid:
        return None
    return valid[0]


def rig_successors(
    grid: Grid,
    state: RigState,
) -> Iterator[
    tuple[
        Action,
        RigState,
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int], ...],
        int,
    ]
]:
    """枚举带拖车状态的全部合法后继。

    产出 (动作, 新状态, 牵引车扫掠, 拖车扫掠, 连接杆扫掠, 代价)。
    """
    for action in Action:
        result = rig_transition(grid, state, action)
        if result is not None:
            nxt, tractor_swept, trailer_swept, drawbar_swept = result
            yield action, nxt, tractor_swept, trailer_swept, drawbar_swept, action.cost
