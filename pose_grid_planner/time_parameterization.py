"""离散路径的时间参数化（分段恒定切向加速度）。"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ._validation import require_finite_real, require_non_negative_real, require_positive_real


class SpeedConstraint(enum.Enum):
    """节点速度最终受到的限制来源。"""

    MAX_SPEED = "max_speed"
    LATERAL_ACCELERATION = "lateral_acceleration"
    TANGENTIAL_ACCELERATION = "tangential_acceleration"
    TANGENTIAL_DECELERATION = "tangential_deceleration"
    START_VELOCITY = "start_velocity"
    END_VELOCITY = "end_velocity"


class TimeParameterizationError(ValueError):
    """速度边界或加减速距离不可行。"""


@dataclass(frozen=True)
class TimeParameterization:
    """时间参数化结果，所有数组均保留调用方给出的原始节点/段索引。"""

    velocities: tuple[float, ...]
    segment_durations: tuple[float, ...]
    segment_accelerations: tuple[float, ...]
    arrival_times: tuple[float, ...]
    limiting_constraints: tuple[frozenset[SpeedConstraint], ...]
    node_group_indices: tuple[int, ...]


def time_parameterize(
    path: Sequence[Any],
    segment_lengths: Iterable[float],
    curvature_bounds: Iterable[float],
    *,
    max_speed: float,
    acceleration: float,
    deceleration: float,
    lateral_acceleration: float,
    start_speed: float,
    end_speed: float,
) -> TimeParameterization:
    """计算各节点最大可行速度、各段最短通过时间和节点到达时间。"""
    if isinstance(path, (str, bytes)) or not isinstance(path, Sequence):
        raise TypeError(f"path 必须是离散节点序列，收到 {type(path).__name__}")
    node_count = len(path)
    if node_count < 1:
        raise ValueError("path 至少包含一个节点")

    lengths = _require_number_tuple("segment_lengths", segment_lengths, node_count - 1)
    curvatures = _require_number_tuple("curvature_bounds", curvature_bounds, node_count - 1)
    max_speed = require_positive_real("max_speed", max_speed)
    acceleration = require_positive_real("acceleration", acceleration)
    deceleration = require_positive_real("deceleration", deceleration)
    lateral_acceleration = require_positive_real("lateral_acceleration", lateral_acceleration)
    start_speed = require_non_negative_real("start_speed", start_speed)
    end_speed = require_non_negative_real("end_speed", end_speed)

    if node_count == 1:
        if start_speed != end_speed:
            raise TimeParameterizationError(
                f"单节点路径的 start_speed={start_speed} 必须等于 end_speed={end_speed}"
            )
        constraints = frozenset(_endpoint_constraints(0, 1, start_speed))
        return TimeParameterization(
            velocities=(float(start_speed),),
            segment_durations=(),
            segment_accelerations=(),
            arrival_times=(0.0,),
            limiting_constraints=(constraints,),
            node_group_indices=(0,),
        )

    parent = list(range(node_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i, length in enumerate(lengths):
        if length == 0:
            union(i, i + 1)

    roots = [find(i) for i in range(node_count)]
    root_order: dict[int, int] = {}
    for root in roots:
        if root not in root_order:
            root_order[root] = len(root_order)
    node_groups = tuple(root_order[root] for root in roots)
    group_count = len(root_order)

    group_lengths: list[float] = [0.0] * (group_count - 1)
    group_curvatures: list[float] = [0.0] * (group_count - 1)
    edge_original_index: list[int] = [-1] * (group_count - 1)
    for i, length in enumerate(lengths):
        if length > 0:
            left_group, right_group = node_groups[i], node_groups[i + 1]
            if right_group != left_group + 1:
                raise TimeParameterizationError(f"segment_lengths[{i}] 零长段合并后结构异常")
            group_lengths[left_group] = length
            group_curvatures[left_group] = curvatures[i]
            edge_original_index[left_group] = i

    if group_count == 1:
        if start_speed != end_speed:
            raise TimeParameterizationError(
                f"全部为零长段时 start_speed={start_speed} 必须等于 end_speed={end_speed}"
            )
        velocity = float(start_speed)
        constraints = frozenset(_endpoint_constraints(0, 1, velocity))
        return TimeParameterization(
            velocities=(velocity,) * node_count,
            segment_durations=(0.0,) * (node_count - 1),
            segment_accelerations=(0.0,) * (node_count - 1),
            arrival_times=(0.0,) * node_count,
            limiting_constraints=(constraints,) * node_count,
            node_group_indices=node_groups,
        )

    curvature_speeds: list[set[float]] = [set() for _ in range(group_count)]
    for i, curvature in enumerate(group_curvatures):
        curvature_speed = math.inf if curvature == 0 else math.sqrt(lateral_acceleration / curvature)
        curvature_speeds[i].add(curvature_speed)
        curvature_speeds[i + 1].add(curvature_speed)

    velocities: list[float] = [max_speed] * group_count
    sources: list[set[SpeedConstraint]] = [set() for _ in range(group_count)]
    for i in range(group_count):
        for curvature_speed in curvature_speeds[i]:
            if _equal(curvature_speed, max_speed):
                sources[i].add(SpeedConstraint.LATERAL_ACCELERATION)
            elif curvature_speed < velocities[i]:
                velocities[i] = curvature_speed
                sources[i] = {SpeedConstraint.LATERAL_ACCELERATION}
        if velocities[i] == max_speed and not sources[i]:
            sources[i] = {SpeedConstraint.MAX_SPEED}
        elif _equal(velocities[i], max_speed):
            sources[i].add(SpeedConstraint.MAX_SPEED)

    if start_speed > max_speed or any(start_speed > value for value in curvature_speeds[0]):
        raise TimeParameterizationError(
            f"start_speed={start_speed} 超过起点速度上界 {velocities[0]}，不能静默降低"
        )
    if end_speed > max_speed or any(end_speed > value for value in curvature_speeds[-1]):
        raise TimeParameterizationError(
            f"end_speed={end_speed} 超过终点速度上界 {velocities[-1]}，不能静默降低"
        )

    velocities[0] = float(start_speed)
    sources[0] = {SpeedConstraint.START_VELOCITY}
    if _equal(start_speed, max_speed):
        sources[0].add(SpeedConstraint.MAX_SPEED)
    for value in curvature_speeds[0]:
        if _equal(start_speed, value):
            sources[0].add(SpeedConstraint.LATERAL_ACCELERATION)

    velocities[-1] = float(end_speed)
    sources[-1] = {SpeedConstraint.END_VELOCITY}
    if _equal(end_speed, max_speed):
        sources[-1].add(SpeedConstraint.MAX_SPEED)
    for value in curvature_speeds[-1]:
        if _equal(end_speed, value):
            sources[-1].add(SpeedConstraint.LATERAL_ACCELERATION)

    for i in range(group_count - 1):
        if i + 1 == group_count - 1:
            continue
        candidate = math.sqrt(velocities[i] ** 2 + 2.0 * acceleration * group_lengths[i])
        if _equal(candidate, velocities[i + 1]):
            sources[i + 1].add(SpeedConstraint.TANGENTIAL_ACCELERATION)
        elif candidate < velocities[i + 1]:
            velocities[i + 1] = candidate
            sources[i + 1] = {SpeedConstraint.TANGENTIAL_ACCELERATION}

    for i in range(group_count - 2, -1, -1):
        if i == 0:
            continue
        candidate = math.sqrt(velocities[i + 1] ** 2 + 2.0 * deceleration * group_lengths[i])
        if _equal(candidate, velocities[i]):
            sources[i].add(SpeedConstraint.TANGENTIAL_DECELERATION)
        elif candidate < velocities[i]:
            velocities[i] = candidate
            sources[i] = {SpeedConstraint.TANGENTIAL_DECELERATION}

    tolerance = 1e-10
    for i, length in enumerate(group_lengths):
        initial_speed, final_speed = velocities[i], velocities[i + 1]
        if initial_speed + final_speed <= tolerance:
            raise TimeParameterizationError(f"合并段 {i} 长度为正但两端速度均为零，通过时间无穷大")
        if final_speed**2 > initial_speed**2 + 2.0 * acceleration * length + tolerance:
            raise TimeParameterizationError(
                f"合并段 {i} 长度 {length} 不足以从 {initial_speed} 加速到 {final_speed}"
            )
        if initial_speed**2 > final_speed**2 + 2.0 * deceleration * length + tolerance:
            raise TimeParameterizationError(
                f"合并段 {i} 长度 {length} 不足以从 {initial_speed} 减速到 {final_speed}"
            )

    segment_durations = [0.0] * (node_count - 1)
    segment_accelerations = [0.0] * (node_count - 1)
    for group_index, length in enumerate(group_lengths):
        original_index = edge_original_index[group_index]
        initial_speed = velocities[group_index]
        final_speed = velocities[group_index + 1]
        duration = 2.0 * length / (initial_speed + final_speed)
        segment_accelerations[original_index] = (final_speed**2 - initial_speed**2) / (2.0 * length)
        segment_durations[original_index] = duration

    node_velocities = tuple(float(velocities[group]) for group in node_groups)
    node_constraints = tuple(frozenset(sources[group]) for group in node_groups)
    arrival_times = [0.0]
    elapsed = 0.0
    for duration in segment_durations:
        elapsed += duration
        arrival_times.append(elapsed)

    return TimeParameterization(
        velocities=node_velocities,
        segment_durations=tuple(segment_durations),
        segment_accelerations=tuple(segment_accelerations),
        arrival_times=tuple(arrival_times),
        limiting_constraints=node_constraints,
        node_group_indices=node_groups,
    )


def _require_number_tuple(name: str, values: Iterable[float], expected_length: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{name} 必须是实数可迭代对象，收到 {type(values).__name__}")
    result = tuple(float(require_finite_real(f"{name}[{i}]", value)) for i, value in enumerate(values))
    if len(result) != expected_length:
        raise ValueError(f"{name} 长度必须是 {expected_length}，收到 {len(result)}")
    for i, value in enumerate(result):
        if value < 0:
            raise ValueError(f"{name}[{i}] 不能为负数: {value}")
    return result


def _equal(left: float, right: float) -> bool:
    if not (math.isfinite(left) and math.isfinite(right)):
        return left == right
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= 1e-10 * scale


def _endpoint_constraints(index: int, endpoint_count: int, speed: float) -> set[SpeedConstraint]:
    if endpoint_count == 1:
        return {SpeedConstraint.START_VELOCITY, SpeedConstraint.END_VELOCITY}
    if index == 0:
        return {SpeedConstraint.START_VELOCITY}
    return {SpeedConstraint.END_VELOCITY}
