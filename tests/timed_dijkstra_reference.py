"""独立的时间展开全状态 Dijkstra 参考实现。

验收用途：与被测核心 pose_grid_planner 完全独立——不导入被测包，
按任务契约重新推导占用/扫掠/时间规则，用 Dijkstra 遍历全部
(x, y, heading, t) 状态求最优总代价与到达时刻，作为 timed_plan 结果的对照期望值。

契约要点（与库文档一致，但独立推导）：
- 状态 = (x, y, h, t)；t ∈ [0, horizon]，到达时刻不得超过 horizon。
- 每个动作（含 WAIT）耗时一步；WAIT 代价为 wait_cost（正整数）。
- 时刻 t 出发的动作：扫掠格须全部在界内、非静态障碍，且不属于
  blocked(t) ∪ blocked(t+1)（起止两时刻禁占格的并集）。
- 平移扫掠 = 起终占用并集；转向扫掠 = 最小轴对齐 2x2；WAIT 扫掠 = 当前车体两格。
- 弹出目标姿态（任意时刻）即停，不要求之后占住终点。
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


def timed_dijkstra_reference(width, height, obstacles, blocked, horizon, wait_cost, start, goal):
    """返回 (最优总代价, 到达时刻)；时域内无解返回 None。

    width/height/obstacles：静态网格。blocked：dict[t] -> 禁占格集合（缺省为空）。
    horizon：时域上界（到达时刻 <= horizon）。wait_cost：正整数。
    start/goal 为 (x, y, h)，h ∈ 0..3。
    """

    def free(c):
        return 0 <= c[0] < width and 0 <= c[1] < height and c not in obstacles

    def blocked_at(t):
        return blocked.get(t, frozenset())

    def pose_ok(x, y, h):
        return all(free(c) for c in _cells_of(x, y, h))

    def sweep_ok(cells, t):
        forb = blocked_at(t) | blocked_at(t + 1)
        return all(free(c) and c not in forb for c in cells)

    def neighbors(x, y, h, t):
        if t >= horizon:
            return
        # 平移：扫掠 = 起终占用并集
        for sign, cost in ((1, COST_FORWARD), (-1, COST_BACKWARD)):
            nx, ny = x + sign * _DX[h], y + sign * _DY[h]
            swept = set(_cells_of(x, y, h)) | set(_cells_of(nx, ny, h))
            if sweep_ok(swept, t):
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
            if sweep_ok(swept, t):
                yield (x, y, nh), COST_TURN
        # WAIT：原地不动，扫掠 = 当前车体全部两格
        if sweep_ok(set(_cells_of(x, y, h)), t):
            yield (x, y, h), wait_cost

    if not pose_ok(*start):
        raise ValueError("start 姿态非法（静态）")
    if not pose_ok(*goal):
        raise ValueError("goal 姿态非法（静态）")
    if any(c in blocked_at(0) for c in _cells_of(*start)):
        raise ValueError("start 占用格在时刻 0 被禁占")
    if start == goal:
        return (0, 0)

    inf = float("inf")
    dist = {(start, 0): 0}
    pq = [(0, 0, start)]  # (代价, 时刻, 姿态)
    while pq:
        d, t, s = heapq.heappop(pq)
        if d > dist.get((s, t), inf):
            continue
        if s == goal:
            return (d, t)
        for ns, c in neighbors(*s, t):
            nd = d + c
            key = (ns, t + 1)
            if nd < dist.get(key, inf):
                dist[key] = nd
                heapq.heappush(pq, (nd, t + 1, ns))
    return None
