"""独立 Dijkstra 全状态参考实现（测试专用）。

本模块**不导入 pose_grid_planner**，用一份独立书写的模型（朝向、占用格、
扫掠规则、边代价、Dijkstra 全状态搜索）产生期望值，供被测 A* 对照。
动作编码与题面/公开枚举一致：前进=0、左转=1、右转=2、后退=3。
"""

from __future__ import annotations

import heapq
from collections import deque

FWD, LEFT, RIGHT, BACK = 0, 1, 2, 3
COSTS = {FWD: 10, BACK: 14, LEFT: 8, RIGHT: 8}
# 0=东 1=北 2=西 3=南
DX = {0: 1, 1: 0, 2: -1, 3: 0}
DY = {0: 0, 1: 1, 2: 0, 3: -1}


def _cells(x, y, h):
    """机器人占用格：参考格 + 沿朝向相邻格。"""
    return (x, y), (x + DX[h], y + DY[h])


class RefWorld:
    def __init__(self, width, height, obstacles):
        self.w = width
        self.h = height
        self.obs = set(obstacles)

    def free(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h and (x, y) not in self.obs

    def valid(self, state):
        x, y, h = state
        for cx, cy in _cells(x, y, h):
            if not self.free(cx, cy):
                return False
        return True

    def all_valid_states(self):
        out = []
        for x in range(self.w):
            for y in range(self.h):
                for h in range(4):
                    s = (x, y, h)
                    if self.valid(s):
                        out.append(s)
        return out

    def edges(self, state):
        """独立实现的合法原语展开：返回 (下一状态, 动作, 代价, 扫掠格元组, 类型)。"""
        x, y, h = state
        result = []
        # 平移：前进 / 后退，扫掠 = 起终占用并集
        for act in (FWD, BACK):
            sgn = 1 if act == FWD else -1
            nx, ny = x + sgn * DX[h], y + sgn * DY[h]
            nh = h
            swept = set(_cells(x, y, h)) | set(_cells(nx, ny, nh))
            if all(self.free(cx, cy) for cx, cy in swept):
                result.append(((nx, ny, nh), act, COSTS[act], tuple(sorted(swept)), "union"))
        # 原地转向：扫掠 = 覆盖起终全部占用格的最小轴对齐 2x2 矩形
        for act, nh in ((LEFT, (h + 1) % 4), (RIGHT, (h - 1) % 4)):
            cells = set(_cells(x, y, h)) | set(_cells(x, y, nh))
            xs = sorted({cx for cx, _ in cells})
            ys = sorted({cy for _, cy in cells})
            if len(xs) == 2 and len(ys) == 2 and xs[1] - xs[0] == 1 and ys[1] - ys[0] == 1:
                rect = {(a, b) for a in xs for b in ys}
                if all(self.free(cx, cy) for cx, cy in rect):
                    result.append(((x, y, nh), act, COSTS[act], tuple(sorted(rect)), "2x2"))
        return result


def dijkstra(world: RefWorld, start, goal):
    """全状态 Dijkstra。返回 (最短代价, 动作序列)；不可达返回 None。"""
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]
    done = set()
    while pq:
        d, s = heapq.heappop(pq)
        if s in done:
            continue
        if s == goal:
            break
        done.add(s)
        for nxt, act, cost, _sw, _kind in world.edges(s):
            nd = d + cost
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                prev[nxt] = (s, act)
                heapq.heappush(pq, (nd, nxt))
    if goal not in dist:
        return None
    path = []
    cur = goal
    while cur != start:
        p, act = prev[cur]
        path.append(act)
        cur = p
    path.reverse()
    return dist[goal], path


def reachable(world: RefWorld, start):
    seen = {start}
    q = deque([start])
    while q:
        s = q.popleft()
        for nxt, *_ in world.edges(s):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen
