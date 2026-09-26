"""沿已验证离散路径做时间参数化。

模型边界：每一段使用一个恒定切向加速度；本模块不建模、也不宣称约束 jerk。
输入路径被视为已通过几何/碰撞验证，time_parameterize() 不重新搜索或碰撞检查。
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any, Sequence

from ._validation import require_finite_number

_REL_TOLERANCE = 1e-12


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=_REL_TOLERANCE, abs_tol=_REL_TOLERANCE)


class SpeedLimitReason(enum.Enum):
    """节点最终速度由哪类约束收紧。"""

    MAX_SPEED = "max_speed"
    CURVATURE = "curvature"
    START_SPEED = "start_speed"
    END_SPEED = "end_speed"
    FORWARD_ACCELERATION = "forward_acceleration"
    BACKWARD_DECELERATION = "backward_deceleration"


class TimeParameterizationError(ValueError):
    """给定边界速度或路径数据在当前分段恒加速度模型下不可行。"""


@dataclass(frozen=True)
class TimedNode:
    """原路径节点、最终速度、到达时间及生效限制。"""

    index: int
    node: Any
    speed: float
    arrival_time: float
    speed_limit_reason: SpeedLimitReason


@dataclass(frozen=True)
class TimedSegment:
    """一段恒定切向加速度运动的时间参数。"""

    index: int
    length: float
    curvature_bound: float
    start_speed: float
    end_speed: float
    duration: float
    tangential_acceleration: float


@dataclass(frozen=True)
class TimedPath:
    """完整时间参数化结果；zero_length_groups 保留原节点索引映射。"""

    nodes: tuple[TimedNode, ...]
    segments: tuple[TimedSegment, ...]
    zero_length_groups: tuple[tuple[int, ...], ...]
    total_time: float


def time_parameterize(
    path: Sequence[Any],
    segment_lengths: Sequence[float],
    curvature_bounds: Sequence[float],
    *,
    max_speed: float,
    max_tangential_acceleration: float,
    max_tangential_deceleration: float,
    max_lateral_acceleration: float,
    start_speed: float,
    end_speed: float,
) -> TimedPath:
    """计算节点速度、每段最短通过时间和累计到达时间。

    节点速度先取最大速度与相邻段横向加速度限制的交集，再沿正长段前向传播
    v_j² <= v_i²+2aL、反向传播 v_i² <= v_j²+2dL。零长段只合并速度相等的节点，
    不改变原节点数量或索引。
    """
    path = _require_sequence("path", path, minimum=1)
    n = len(path)
    if isinstance(segment_lengths, (str, bytes)) or not isinstance(segment_lengths, Sequence):
        raise TypeError(
            "segment_lengths 必须是数字序列，收到 "
            f"{type(segment_lengths).__name__}: {segment_lengths!r}"
        )
    if isinstance(curvature_bounds, (str, bytes)) or not isinstance(curvature_bounds, Sequence):
        raise TypeError(
            "curvature_bounds 必须是数字序列，收到 "
            f"{type(curvature_bounds).__name__}: {curvature_bounds!r}"
        )
    if len(segment_lengths) != n - 1:
        raise ValueError(
            f"segment_lengths 长度必须为 path 节点数-1={n - 1}，收到 {len(segment_lengths)}"
        )
    if len(curvature_bounds) != n - 1:
        raise ValueError(
            f"curvature_bounds 长度必须为 path 节点数-1={n - 1}，收到 {len(curvature_bounds)}"
        )

    v_max = require_finite_number("max_speed", max_speed, positive=True)
    a_max = require_finite_number(
        "max_tangential_acceleration", max_tangential_acceleration, positive=True
    )
    d_max = require_finite_number(
        "max_tangential_deceleration", max_tangential_deceleration, positive=True
    )
    lateral_max = require_finite_number(
        "max_lateral_acceleration", max_lateral_acceleration, positive=True
    )
    v_start = require_finite_number("start_speed", start_speed, nonnegative=True)
    v_end = require_finite_number("end_speed", end_speed, nonnegative=True)
    if v_start > v_max:
        raise TimeParameterizationError(f"start_speed={v_start} 超过 max_speed={v_max}")
    if v_end > v_max:
        raise TimeParameterizationError(f"end_speed={v_end} 超过 max_speed={v_max}")

    lengths = [
        require_finite_number(f"segment_lengths[{i}]", value, nonnegative=True)
        for i, value in enumerate(segment_lengths)
    ]
    curvatures = [
        require_finite_number(f"curvature_bounds[{i}]", value, nonnegative=True)
        for i, value in enumerate(curvature_bounds)
    ]

    caps = [v_max] * n
    reasons = [SpeedLimitReason.MAX_SPEED] * n
    for i, curvature in enumerate(curvatures):
        if curvature > 0.0:
            curvature_speed = math.sqrt(lateral_max / curvature)
            for node_index in (i, i + 1):
                if curvature_speed <= caps[node_index]:
                    caps[node_index] = curvature_speed
                    reasons[node_index] = SpeedLimitReason.CURVATURE

    if v_start > caps[0]:
        raise TimeParameterizationError(
            f"start_speed={v_start} 高于起点曲率/最大速度限制 {caps[0]:.12g}"
        )
    if v_end > caps[n - 1]:
        raise TimeParameterizationError(
            f"end_speed={v_end} 高于终点曲率/最大速度限制 {caps[n - 1]:.12g}"
        )

    parent = list(range(n))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i, length in enumerate(lengths):
        if length == 0.0:
            union(i, i + 1)

    roots = [find(i) for i in range(n)]
    root_set = set(roots)
    comp_cap: dict[int, float] = {}
    comp_reason: dict[int, SpeedLimitReason] = {}
    for i, root in enumerate(roots):
        if root not in comp_cap or caps[i] < comp_cap[root]:
            comp_cap[root] = caps[i]
            comp_reason[root] = reasons[i]

    boundary: dict[int, tuple[float, SpeedLimitReason, str]] = {}

    def set_boundary(root: int, speed: float, reason: SpeedLimitReason, name: str) -> None:
        if speed > comp_cap[root]:
            raise TimeParameterizationError(
                f"{name}={speed} 高于零长段合并后的速度限制 {comp_cap[root]:.12g}"
            )
        if root in boundary:
            old_speed, _old_reason, old_name = boundary[root]
            if speed != old_speed:
                raise TimeParameterizationError(
                    f"零长段合并要求 {old_name}={old_speed} 与 {name}={speed} 相等"
                )
        else:
            boundary[root] = (speed, reason, name)

    if n == 1:
        if v_start != v_end:
            raise TimeParameterizationError(
                f"单节点路径要求 start_speed=end_speed，收到 {v_start} 与 {v_end}"
            )
        set_boundary(roots[0], v_start, SpeedLimitReason.START_SPEED, "start_speed")

    set_boundary(roots[0], v_start, SpeedLimitReason.START_SPEED, "start_speed")
    set_boundary(roots[n - 1], v_end, SpeedLimitReason.END_SPEED, "end_speed")

    comp_speed = dict(comp_cap)
    comp_limiter = dict(comp_reason)
    for root, (speed, reason, _name) in boundary.items():
        comp_speed[root] = speed
        # 边界速度恰好压在上限上时，真正决定速度值的是该上限；显式边界标签保留给
        # 低于上限、因此由调用方边界收紧的情形。
        if speed < comp_cap[root]:
            comp_limiter[root] = reason

    positive_edges: list[tuple[int, int, int, float]] = []
    for i, length in enumerate(lengths):
        if length > 0.0:
            left_root = roots[i]
            right_root = roots[i + 1]
            if left_root == right_root:
                raise TimeParameterizationError(
                    f"segment_lengths[{i}]={length:g} 为正，但其端点经零长段合并后相同"
                )
            positive_edges.append((i, left_root, right_root, length))

    for _i, left_root, right_root, length in positive_edges:
        allowed_square = comp_speed[left_root] ** 2 + 2.0 * a_max * length
        allowed = math.sqrt(allowed_square)
        if allowed < comp_speed[right_root]:
            comp_speed[right_root] = allowed
            comp_limiter[right_root] = SpeedLimitReason.FORWARD_ACCELERATION

    for _i, left_root, right_root, length in reversed(positive_edges):
        allowed_square = comp_speed[right_root] ** 2 + 2.0 * d_max * length
        allowed = math.sqrt(allowed_square)
        if allowed < comp_speed[left_root]:
            comp_speed[left_root] = allowed
            comp_limiter[left_root] = SpeedLimitReason.BACKWARD_DECELERATION

    final_start = comp_speed[roots[0]]
    final_end = comp_speed[roots[n - 1]]
    if _close(final_start, v_start):
        comp_speed[roots[0]] = v_start
    else:
        raise TimeParameterizationError(
            f"start_speed={v_start:g} 不可行；受后续约束最多只能以 {final_start:.12g} 出发"
        )
    if _close(final_end, v_end):
        comp_speed[roots[n - 1]] = v_end
    else:
        raise TimeParameterizationError(
            f"end_speed={v_end:g} 不可行；受前文约束最多只能以 {final_end:.12g} 到达"
        )

    speeds = [comp_speed[roots[i]] for i in range(n)]
    limiters = [comp_limiter[roots[i]] for i in range(n)]
    segments: list[TimedSegment] = []
    arrival_times = [0.0]
    current_time = 0.0
    for i, length in enumerate(lengths):
        start = speeds[i]
        end = speeds[i + 1]
        if length == 0.0:
            if start != end:
                raise TimeParameterizationError(f"零长段 {i} 的端点速度不相等")
            duration = 0.0
            acceleration = 0.0
        else:
            if start == 0.0 and end == 0.0:
                raise TimeParameterizationError(
                    f"segment_lengths[{i}]={length:g} 为正，但两端速度均为 0，无法通过"
                )
            duration = 2.0 * length / (start + end)
            acceleration = (end - start) / duration
            acceleration_limit = a_max if acceleration >= 0.0 else d_max
            if abs(acceleration) > acceleration_limit * (1.0 + _REL_TOLERANCE) + _REL_TOLERANCE:
                required = abs(acceleration)
                raise TimeParameterizationError(
                    f"segment_lengths[{i}] 需要切向加速度 {required:.12g}，超过限制 {acceleration_limit:g}"
                )
            peak_speed = max(start, end)
            if peak_speed > v_max * (1.0 + _REL_TOLERANCE) + _REL_TOLERANCE:
                raise TimeParameterizationError(f"segment_lengths[{i}] 超过 max_speed")
            if (
                curvatures[i] > 0.0
                and peak_speed ** 2 * curvatures[i]
                > lateral_max * (1.0 + _REL_TOLERANCE) + _REL_TOLERANCE
            ):
                raise TimeParameterizationError(f"segment_lengths[{i}] 超过横向加速度限制")
            current_time += duration
        arrival_times.append(current_time)
        segments.append(
            TimedSegment(
                index=i,
                length=length,
                curvature_bound=curvatures[i],
                start_speed=start,
                end_speed=end,
                duration=duration,
                tangential_acceleration=acceleration,
            )
        )

    groups = _zero_length_groups(roots, root_set)
    nodes = tuple(
        TimedNode(
            index=i,
            node=path[i],
            speed=speeds[i],
            arrival_time=arrival_times[i],
            speed_limit_reason=limiters[i],
        )
        for i in range(n)
    )
    return TimedPath(
        nodes=nodes,
        segments=tuple(segments),
        zero_length_groups=groups,
        total_time=current_time,
    )


def _require_sequence(name: str, value: Any, *, minimum: int) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} 必须是节点序列，收到 {type(value).__name__}: {value!r}")
    if len(value) < minimum:
        raise ValueError(f"{name} 至少包含 {minimum} 个节点，收到 {len(value)} 个")
    return value


def _zero_length_groups(roots: Sequence[int], root_set: set[int]) -> tuple[tuple[int, ...], ...]:
    grouped: dict[int, list[int]] = {root: [] for root in root_set}
    for node_index, root in enumerate(roots):
        grouped[root].append(node_index)
    return tuple(tuple(indices) for indices in grouped.values() if len(indices) > 1)
