"""网格世界、机器人姿态与运动原语模型。

坐标约定：格坐标 (x, y) 均为 0 基整数，x 向右增大、y 向上增大。
机器人占据“参考格”(x, y) 以及沿当前朝向相邻的一格，共 2 格。

运动原语：
- FORWARD  ：沿朝向前进一格，代价 10
- BACKWARD ：沿朝向反方向后退一格，代价 14
- TURN_LEFT / TURN_RIGHT：原地左转/右转 90 度，代价 8

碰撞/扫掠规则：
- 平移（前进/后退）合法 ⇔ 起点与终点占用格并集中的每一格都在界内且为空；
- 转向合法 ⇔ 恰好由“起点全部占用格 ∪ 终点全部占用格”构成的最小
  轴对齐 2×2 矩形中的 4 个格子全部在界内且为空（机器人占据的 2 格除外，
  它们本就必须为空——任何占用格都不允许落在障碍上）。
"""

from __future__ import annotations

import math
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from enum import IntEnum
from typing import AbstractSet


class Heading(IntEnum):
    """四方向；取值为公开协议的一部分（0..3）。"""

    EAST = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3

    @property
    def delta(self) -> tuple[int, int]:
        return _DELTA[self]  # type: ignore[index]

    def turned(self, action: "Action") -> "Heading":
        if action is Action.TURN_LEFT:
            return Heading((self + 1) & 3)
        if action is Action.TURN_RIGHT:
            return Heading((self - 1) & 3)
        return self


# EAST / NORTH / WEST / SOUTH 对应的 (dx, dy)
_DELTA: dict[Heading, tuple[int, int]] = {
    Heading.EAST: (1, 0),
    Heading.NORTH: (0, 1),
    Heading.WEST: (-1, 0),
    Heading.SOUTH: (0, -1),
}


class Action(IntEnum):
    """运动原语；取值为公开协议的一部分。"""

    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    BACKWARD = 3

    @property
    def cost(self) -> int:
        return ACTION_COST[self]


ACTION_COST: dict[Action, int] = {
    Action.FORWARD: 10,
    Action.BACKWARD: 14,
    Action.TURN_LEFT: 8,
    Action.TURN_RIGHT: 8,
}

_TURNS = (Action.TURN_LEFT, Action.TURN_RIGHT)
_MOVES = (Action.FORWARD, Action.BACKWARD)


class PlannerError(Exception):
    """本模块所有错误的基类。"""


class InvalidParameterError(PlannerError, ValueError):
    """单个标量参数非法（类型、取值、NaN/Infinity、bool 冒充 int 等）。"""


class InvalidGridError(PlannerError, ValueError):
    """障碍网格描述非法。"""


class InvalidPoseError(PlannerError, ValueError):
    """姿态非法（越界、落在障碍上、占用格非法）。

    :param where: "start" 或 "goal"，用于定位错误来源。
    """

    def __init__(self, message: str, *, where: str | None = None) -> None:
        super().__init__(message)
        self.where = where


@dataclass(frozen=True, slots=True)
class Pose:
    """机器人姿态：参考格 (x, y) + 朝向。"""

    x: int
    y: int
    heading: Heading

    def __post_init__(self) -> None:
        # 允许从 int 构造（IntEnum 值），但拒绝 bool 与非整数；x/y 同样严格。
        object.__setattr__(self, "x", _strict_int(self.x, "x"))
        object.__setattr__(self, "y", _strict_int(self.y, "y"))
        if isinstance(self.heading, bool) or not isinstance(self.heading, (int, Heading)):
            raise InvalidParameterError(
                f"heading 必须为 Heading/0..3 整数，得到 {self.heading!r}"
            )
        if int(self.heading) not in (0, 1, 2, 3):
            raise InvalidParameterError(f"heading 必须在 0..3，得到 {int(self.heading)}")
        object.__setattr__(self, "heading", Heading(int(self.heading)))

    @property
    def state(self) -> tuple[int, int, Heading]:
        return (self.x, self.y, self.heading)

    def occupied(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """返回 (参考格, 沿朝向相邻格)。"""
        dx, dy = self.heading.delta
        return ((self.x, self.y), (self.x + dx, self.y + dy))


@dataclass(frozen=True, slots=True)
class Transition:
    """一次成功的原语执行：起/终姿态、动作、代价与扫掠证据。"""

    src: Pose
    dst: Pose
    action: Action
    cost: int
    swept: tuple[tuple[int, int], ...]
    swept_kind: str  # "union"（平移）或 "2x2"（转向）


def _strict_int(value: object, name: str) -> int:
    """整数参数校验：拒绝 bool、拒绝 float（含 NaN/Inf）、拒绝非整数。"""
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float):
            kind = "bool" if isinstance(value, bool) else (
                "NaN/Infinity" if not math.isfinite(value) else "非整数 float"
            )
            raise InvalidParameterError(f"{name} 必须是整数，得到 {kind}: {value!r}")
        raise InvalidParameterError(f"{name} 必须是整数，得到 {type(value).__name__}: {value!r}")
    return value


def _normalize_obstacles(
    obstacles: IterableABC[object],
    width: int,
    height: int,
) -> frozenset[tuple[int, int]]:
    if isinstance(obstacles, (str, bytes)) or not isinstance(obstacles, IterableABC):
        raise InvalidGridError(
            f"obstacles 必须为 (x, y) 坐标的可迭代集合，得到 {type(obstacles).__name__}"
        )
    seen: set[tuple[int, int]] = set()
    # 字符串也是 Iterable，需要挡掉；逐元素严格校验。
    for index, cell in enumerate(obstacles):
        loc = f"obstacles[{index}]"
        if isinstance(cell, bool) or not isinstance(cell, (tuple, list)):
            raise InvalidGridError(f"{loc} 必须是 (x, y) 元组，得到 {cell!r}")
        if len(cell) != 2:
            raise InvalidGridError(f"{loc} 长度必须为 2，得到长度 {len(cell)}")
        ox, oy = cell
        try:
            ox = _strict_int(ox, f"{loc}[0]")
            oy = _strict_int(oy, f"{loc}[1]")
        except InvalidParameterError as exc:
            # 障碍描述内部的标量错误统一归为网格错误，信息保留定位。
            raise InvalidGridError(str(exc)) from None
        if not (0 <= ox < width and 0 <= oy < height):
            raise InvalidGridError(f"{loc}=({ox}, {oy}) 越出 {width}x{height} 网格")
        if (ox, oy) in seen:
            raise InvalidGridError(f"{loc}=({ox}, {oy}) 障碍重复给出")
        seen.add((ox, oy))
    return frozenset(seen)


class GridWorld:
    """固定尺寸与障碍集合的网格世界（不可变、可复用、可并发查询）。

    :param width:  x 方向格数，1..20
    :param height: y 方向格数，1..20
    :param obstacles: 障碍格坐标的可迭代集合，元素为 (x, y)，不得越界/重复
    """

    MAX_SIZE = 20

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: Iterable[tuple[int, int]] = (),
    ) -> None:
        w = _strict_int(width, "width")
        h = _strict_int(height, "height")
        if not (1 <= w <= self.MAX_SIZE):
            raise InvalidParameterError(f"width 必须在 1..{self.MAX_SIZE}，得到 {w}")
        if not (1 <= h <= self.MAX_SIZE):
            raise InvalidParameterError(f"height 必须在 1..{self.MAX_SIZE}，得到 {h}")
        self._width = w
        self._height = h
        self._obstacles = _normalize_obstacles(obstacles, w, h)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def obstacles(self) -> frozenset[tuple[int, int]]:
        return self._obstacles

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self._width and 0 <= y < self._height

    def is_free(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and (x, y) not in self._obstacles

    def valid_pose(self, pose: Pose) -> bool:
        """姿态的两个占用格都在界内且为空。"""
        for cx, cy in pose.occupied():
            if not self.is_free(cx, cy):
                return False
        return True

    def check_pose(self, pose: Pose, where: str) -> None:
        """校验起/终姿态，失败抛 InvalidPoseError（带 where 定位）。"""
        ref, nose = pose.occupied()
        for label, (cx, cy) in (("参考格", ref), ("相邻格", nose)):
            if not self.in_bounds(cx, cy):
                raise InvalidPoseError(
                    f"{where} 姿态 {pose.state} 的{label}({cx}, {cy}) 越出 "
                    f"{self._width}x{self._height} 网格",
                    where=where,
                )
            if (cx, cy) in self._obstacles:
                raise InvalidPoseError(
                    f"{where} 姿态 {pose.state} 的{label}({cx}, {cy}) 位于障碍格",
                    where=where,
                )

    def apply(self, pose: Pose, action: Action) -> Transition | None:
        """应用一个运动原语；非法（越界/碰撞/扫掠到障碍）返回 None。

        调用方需保证 ``pose`` 本身是合法姿态。
        """
        if isinstance(action, bool) or not isinstance(action, (int, Action)):
            raise InvalidParameterError(f"action 非法: {action!r}")
        try:
            action = Action(int(action))
        except ValueError:
            raise InvalidParameterError(f"action 取值必须在 0..3，得到 {int(action)}") from None

        if action in _MOVES:
            dx, dy = pose.heading.delta
            if action is Action.BACKWARD:
                dx, dy = -dx, -dy
            dst = Pose(pose.x + dx, pose.y + dy, pose.heading)
            swept = tuple(sorted(set(pose.occupied()) | set(dst.occupied())))
            for cx, cy in swept:
                if not self.is_free(cx, cy):
                    return None
            return Transition(pose, dst, action, action.cost, swept, "union")

        # 原地转向：2x2 扫掠矩形。
        new_heading = pose.heading.turned(action)
        dst = Pose(pose.x, pose.y, new_heading)
        cells = set(pose.occupied()) | set(dst.occupied())
        xs = {x for x, _ in cells}
        ys = {y for _, y in cells}
        # 契约上必为恰好 2 个 x 与 2 个 y（共 4 格）；若不符说明模型被破坏。
        if len(xs) != 2 or len(ys) != 2:  # pragma: no cover - 防御性
            return None
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if (x1 - x0) != 1 or (y1 - y0) != 1:  # pragma: no cover - 防御性
            return None
        swept = tuple(sorted([(x0, y0), (x0, y1), (x1, y0), (x1, y1)]))
        for cx, cy in swept:
            if not self.is_free(cx, cy):
                return None
        return Transition(pose, dst, action, action.cost, swept, "2x2")

    def neighbors(self, pose: Pose) -> list[Transition]:
        """生成姿态下全部合法原语（按动作枚举顺序，保证展开确定性）。"""
        result: list[Transition] = []
        for action in (Action.FORWARD, Action.BACKWARD, Action.TURN_LEFT, Action.TURN_RIGHT):
            tr = self.apply(pose, action)
            if tr is not None:
                result.append(tr)
        return result

    def all_states(self) -> AbstractSet[tuple[int, int, Heading]]:
        """全部可行 (x, y, heading) 状态，供独立参考实现使用。"""
        states: set[tuple[int, int, Heading]] = set()
        for x in range(self._width):
            for y in range(self._height):
                if (x, y) in self._obstacles:
                    continue
                for h in Heading:
                    p = Pose(x, y, h)
                    if self.valid_pose(p):
                        states.add(p.state)
        return states
