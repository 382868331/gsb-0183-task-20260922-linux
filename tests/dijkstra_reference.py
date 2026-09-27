"""独立的全状态 Dijkstra 参考实现。

验收用途：与被测核心 pose_grid_planner 完全独立——不导入被测包，
按题面契约重新推导占用/扫掠规则，用 Dijkstra 遍历全部 (x, y, heading)
状态求最优代价，作为 A* 结果的对照期望值。
"""

from __future__ import annotations

import heapq

# 朝向 0..3 = NORTH, EAST, SOUTH, WEST（y 向下增）
_DX = (0, 1, 0, -1)
_DY = (-1, 0, 1, 0)

COST_FORWARD = 10
COST_BACKWARD = 14
COST_TURN = 8


def _cells_of(x, y, h):
    return ((x, y), (x + _DX[h], y + _DY[h]))


def dijkstra_reference(width, height, obstacles, start, goal, terrain_costs=None):
    """返回 start→goal 的最优代价；无解返回 None。

    start/goal 为 (x, y, h)，h ∈ 0..3。obstacles 为 (x, y) 集合。
    terrain_costs 为 {cell: 非负附加代价}；每个动作对扫掠集合中的不同格子计一次。
    """
    terrain_costs = terrain_costs or {}

    def free(c):
        return 0 <= c[0] < width and 0 <= c[1] < height and c not in obstacles

    def pose_ok(x, y, h):
        return all(free(c) for c in _cells_of(x, y, h))

    def neighbors(x, y, h):
        # 平移：扫掠 = 起终占用并集
        for sign, cost in ((1, COST_FORWARD), (-1, COST_BACKWARD)):
            nx, ny = x + sign * _DX[h], y + sign * _DY[h]
            swept = set(_cells_of(x, y, h)) | set(_cells_of(nx, ny, h))
            if all(free(c) for c in swept):
                yield (nx, ny, h), cost + sum(terrain_costs.get(c, 0) for c in swept)
        # 原地转向：扫掠 = 包含起终占用格的最小轴对齐 2x2（4 格）
        for dh in (1, -1):
            nh = (h + dh) % 4
            swept = {
                (x, y),
                (x + _DX[h], y + _DY[h]),
                (x + _DX[nh], y + _DY[nh]),
                (x + _DX[h] + _DX[nh], y + _DY[h] + _DY[nh]),
            }
            if all(free(c) for c in swept):
                yield (x, y, nh), COST_TURN + sum(terrain_costs.get(c, 0) for c in swept)

    if not pose_ok(*start) or not pose_ok(*goal):
        raise ValueError("start/goal 姿态非法")
    if start == goal:
        return 0

    inf = float("inf")
    dist = {start: 0}
    pq = [(0, start)]
    while pq:
        d, s = heapq.heappop(pq)
        if d > dist.get(s, inf):
            continue
        if s == goal:
            return d
        for ns, c in neighbors(*s):
            nd = d + c
            if nd < dist.get(ns, inf):
                dist[ns] = nd
                heapq.heappush(pq, (nd, ns))
    return None
