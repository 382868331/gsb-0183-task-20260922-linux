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


def dijkstra_reference(width, height, obstacles, start, goal):
    """返回 start→goal 的最优代价；无解返回 None。

    start/goal 为 (x, y, h)，h ∈ 0..3。obstacles 为 (x, y) 集合。
    """

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
                yield (nx, ny, h), cost
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
                yield (x, y, nh), COST_TURN

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


COST_WAIT = 1  # 与被测库 DEFAULT_WAIT_COST 对应的等待代价（正整数）


def timed_dijkstra_reference(width, height, obstacles, blocked_by_time, horizon,
                             start, goal, wait_cost=COST_WAIT):
    """离散时间版独立参考：返回 (最优总代价, 到达时刻)；时域内无解返回 None。

    状态为 (x, y, h, t)，t ∈ [0, horizon]。blocked_by_time 为 {t: 格子集合}；
    从时刻 t 执行的动作，其扫掠格必须避开 blocked[t] ∪ blocked[t+1]。
    目标在任一 t <= horizon 到达即停止（不要求永久占住终点）。
    优化准则：总代价最小，等代价时到达时刻最早——堆元素为 (d, state)，
    heapq 在 d 并列时按 (x,y,h,t) 字典序弹出，goal 的 x,y,h 相同即等价于
    最早 t，与被测 plan_timed 的二级准则一致。
    与被测核心完全独立：不导入被测包，按题面契约重新推导规则。
    """

    def free(c):
        return 0 <= c[0] < width and 0 <= c[1] < height and c not in obstacles

    def blocked(c, t):
        return c in blocked_by_time.get(t, ())

    def pose_ok(x, y, h):
        return all(free(c) for c in _cells_of(x, y, h))

    def neighbors(x, y, h, t):
        if t >= horizon:
            return
        ban = set(blocked_by_time.get(t, ())) | set(blocked_by_time.get(t + 1, ()))

        def sweep_ok(swept):
            return all(free(c) and c not in ban for c in swept)

        # 等待：检查车体当前全部占用格
        if sweep_ok(_cells_of(x, y, h)):
            yield (x, y, h, t + 1), wait_cost
        # 平移：扫掠 = 起终占用并集
        for sign, cost in ((1, COST_FORWARD), (-1, COST_BACKWARD)):
            nx, ny = x + sign * _DX[h], y + sign * _DY[h]
            swept = set(_cells_of(x, y, h)) | set(_cells_of(nx, ny, h))
            if sweep_ok(swept):
                yield (nx, ny, h, t + 1), cost
        # 原地转向：扫掠 = 最小轴对齐 2x2（4 格）
        for dh in (1, -1):
            nh = (h + dh) % 4
            swept = {
                (x, y),
                (x + _DX[h], y + _DY[h]),
                (x + _DX[nh], y + _DY[nh]),
                (x + _DX[h] + _DX[nh], y + _DY[h] + _DY[nh]),
            }
            if sweep_ok(swept):
                yield (x, y, nh, t + 1), COST_TURN

    if not pose_ok(*start) or not pose_ok(*goal):
        raise ValueError("start/goal 姿态非法")
    if any(blocked(c, 0) for c in _cells_of(start[0], start[1], start[2])):
        raise ValueError("start 占用格在时刻 0 被禁占")
    if start == goal:
        return (0, 0)

    inf = float("inf")
    start4 = (start[0], start[1], start[2], 0)
    dist = {start4: 0}
    pq = [(0, start4)]
    while pq:
        d, s = heapq.heappop(pq)
        if d > dist.get(s, inf):
            continue
        if s[:3] == goal:
            return (d, s[3])
        for ns, c in neighbors(*s):
            nd = d + c
            if nd < dist.get(ns, inf):
                dist[ns] = nd
                heapq.heappush(pq, (nd, ns))
    return None
