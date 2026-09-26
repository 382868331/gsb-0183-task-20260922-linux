"""铰接车全状态 Dijkstra 独立参考；不导入 pose_grid_planner。"""

from __future__ import annotations

import heapq

_DX = (0, 1, 0, -1)
_DY = (-1, 0, 1, 0)
COSTS = {"forward": 10, "backward": 14, "turn": 8}


def dijkstra_articulated(width, height, obstacles, start, goal, max_articulation_turns=1):
    obstacles = set(obstacles)

    def free(cell):
        x, y = cell
        return 0 <= x < width and 0 <= y < height and cell not in obstacles

    def turn_distance(a, b):
        d = (b - a) % 4
        return min(d, 4 - d)

    def valid_state(state):
        x, y, h, t = state
        if turn_distance(h, t) > max_articulation_turns:
            return False
        hitch = (x, y)
        tractor_front = (x + _DX[h], y + _DY[h])
        trailer_axle = (x - _DX[t], y - _DY[t])
        return all(free(c) for c in (hitch, tractor_front, trailer_axle))

    def corner_sweep(a, b, c):
        cells = {a, b, c}
        min_x, max_x = min(p[0] for p in cells), max(p[0] for p in cells)
        min_y, max_y = min(p[1] for p in cells), max(p[1] for p in cells)
        if max_x - min_x == 1 and max_y - min_y == 1:
            return {(xx, yy) for xx in range(min_x, max_x + 1) for yy in range(min_y, max_y + 1)}
        return cells

    def neighbors(state):
        x, y, h, t = state
        hitch = (x, y)
        axle = (x - _DX[t], y - _DY[t])
        tractor_old = {hitch, (x + _DX[h], y + _DY[h])}

        nx, ny = x + _DX[h], y + _DY[h]
        forward_state = (nx, ny, h, h)
        new_hitch = (nx, ny)
        tractor = tractor_old | {new_hitch, (nx + _DX[h], ny + _DY[h])}
        bar = corner_sweep(axle, hitch, new_hitch)
        if valid_state(forward_state) and all(free(c) for c in tractor | bar):
            yield forward_state, COSTS["forward"]

        if h == t:
            bx, by = x - _DX[h], y - _DY[h]
            backward_state = (bx, by, h, t)
            new_axle = (bx - _DX[t], by - _DY[t])
            tractor = tractor_old | {(bx, by), (bx + _DX[h], by + _DY[h])}
            bar = {new_axle, axle, hitch}
            if valid_state(backward_state) and all(free(c) for c in tractor | bar):
                yield backward_state, COSTS["backward"]

        for dh in (-1, 1):
            nh = (h + dh) % 4
            turn_state = (x, y, nh, t)
            tractor = {
                hitch,
                (x + _DX[h], y + _DY[h]),
                (x + _DX[nh], y + _DY[nh]),
                (x + _DX[h] + _DX[nh], y + _DY[h] + _DY[nh]),
            }
            bar = {axle, hitch}
            if valid_state(turn_state) and all(free(c) for c in tractor | bar):
                yield turn_state, COSTS["turn"]

    if not valid_state(start) or not valid_state(goal):
        raise ValueError("start/goal articulated state invalid")
    if start == goal:
        return 0
    dist = {start: 0}
    pq = [(0, start)]
    inf = float("inf")
    while pq:
        cost, state = heapq.heappop(pq)
        if cost > dist.get(state, inf):
            continue
        if state == goal:
            return cost
        for nxt, edge_cost in neighbors(state):
            new_cost = cost + edge_cost
            if new_cost < dist.get(nxt, inf):
                dist[nxt] = new_cost
                heapq.heappush(pq, (new_cost, nxt))
    return None
