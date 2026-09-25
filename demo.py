"""固定输入演示：真实计算静态与时序路径并重放验证，再实际触发拒绝/失败边界。

运行：python demo.py   （纯标准库，离线，无 sleep、无预录结果）
"""

from __future__ import annotations

import time

from pose_grid_planner import (
    Action,
    Grid,
    Heading,
    Pose,
    SearchStatus,
    Timetable,
    plan,
    plan_timed,
    replay,
    replay_timed,
)

ARROW = {Heading.NORTH: "^", Heading.EAST: ">", Heading.SOUTH: "v", Heading.WEST: "<"}


def render(grid: Grid, poses: list[Pose]) -> str:
    """ASCII 地图：'#' 障碍，方向箭头为路径上各步姿态（后者覆盖前者），'.' 空地。"""
    canvas = [["#" if (x, y) in grid.obstacles else "." for x in range(grid.width)] for y in range(grid.height)]
    for p in poses:
        canvas[p.y][p.x] = ARROW[p.heading]
    return "\n".join("".join(row) for row in canvas)


def demo_normal() -> None:
    # 确定性固定场景：10x7，1 格宽竖走廊 + 右侧 4 格宽开阔区（转向需要 2x2 扫掠空间）。
    grid = Grid.from_rows([
        "..........",
        ".##.##....",
        ".##.##....",
        ".##.##....",
        ".##.##....",
        ".##.##....",
        "..........",
    ])
    start = Pose(0, 6, Heading.EAST)
    goal = Pose(8, 1, Heading.NORTH)
    print("=== 场景 1：正常规划（含转向与扫掠证据） ===")
    print(render(grid, [start, goal]))
    print(f"start={start}  goal={goal}")

    t0 = time.perf_counter()
    result = plan(grid, start, goal)
    elapsed = time.perf_counter() - t0
    assert result.status is SearchStatus.OPTIMAL, result.status
    p = result.plan
    print(f"status={result.status.value}  expansions={result.expansions}  "
          f"total_cost={p.total_cost}  steps={len(p.actions)}  用时={elapsed * 1000:.1f}ms")
    print("动作序列:", " ".join(a.name for a in p.actions))
    print("逐步扫掠证据（每步检查的格子，均须为空）:")
    for i, e in enumerate(p.evidence):
        print(f"  step {i:2d}: {e.action.name:<10} {e.from_pose} -> {e.to_pose}  "
              f"swept={sorted(e.swept_cells)} cost={e.cost}")

    # 重放验证：逐步重新执行，确认不碰撞且精确到达目标（含朝向）
    final_pose, evidence = replay(grid, start, p.actions)
    assert final_pose == goal, (final_pose, goal)
    assert sum(e.cost for e in evidence) == p.total_cost
    print(f"重放验证通过：到达 {final_pose}，{len(evidence)} 步无碰撞。")
    print("路径姿态序列：")
    poses = [start] + [e.to_pose for e in evidence]
    print(render(grid, poses))


def demo_rejection() -> None:
    print("\n=== 场景 2：实际触发的拒绝/异常边界 ===")

    # 2a. 非法数值输入：NaN 必须被可定位地拒绝（不静默纠正）
    try:
        Grid(float("nan"), 5)
    except ValueError as exc:
        print(f"[拒绝] Grid(width=nan) -> ValueError: {exc}")

    # 2b. bool 冒充整数参数：必须拒绝
    try:
        Grid(5, True)
    except TypeError as exc:
        print(f"[拒绝] Grid(height=True) -> TypeError: {exc}")

    # 2c. 搜索预算耗尽 vs 证明无解：两者必须可区分
    grid = Grid.from_rows([
        "..........",
        ".##.##....",
        ".##.##....",
        ".##.##....",
        ".##.##....",
        ".##.##....",
        "..........",
    ])
    start = Pose(0, 6, Heading.EAST)
    goal = Pose(8, 1, Heading.NORTH)
    tight = plan(grid, start, goal, max_expansions=3)
    print(f"[预算] max_expansions=3 -> status={tight.status.value}（预算耗尽，既未找到也未证明无解）")

    corridor = Grid(5, 1)
    no_path = plan(corridor, Pose(0, 0, Heading.EAST), Pose(3, 0, Heading.WEST))
    print(f"[无解] 1 格宽走廊内要求反向 -> status={no_path.status.value}（开集耗尽，已证明无解）")
    assert tight.status is SearchStatus.BUDGET_EXHAUSTED
    assert no_path.status is SearchStatus.NO_PATH


def demo_timed_normal() -> None:
    print("\n=== 场景 3：离散时间——窄路临时占用，等待让行 ===")
    grid = Grid(6, 1)  # 6x1 走廊，无法转向
    start = Pose(2, 0, Heading.EAST)  # 车身 (2,0),(3,0)
    goal = Pose(4, 0, Heading.EAST)  # 车身 (4,0),(5,0)
    # 宽障碍在 t=2,3 占据走廊东段；前进/原地等待都会被堵死，唯一活路是先倒车
    tt = Timetable({2: [(3, 0), (4, 0), (5, 0)], 3: [(3, 0), (4, 0), (5, 0)]})
    print(f"grid={grid.width}x{grid.height}（1 格宽走廊）  start={start}  goal={goal}")
    print("禁占格 t=2,3: (3,0),(4,0),(5,0)；其余时刻畅通。horizon=10, wait_cost=1")

    t0 = time.perf_counter()
    result = plan_timed(grid, tt, start, goal, horizon=10)
    elapsed = time.perf_counter() - t0
    assert result.status is SearchStatus.OPTIMAL, result.status
    p = result.plan
    print(f"status={result.status.value}  expansions={result.expansions}  "
          f"total_cost={p.total_cost}  到达时刻 t={result.arrival_time}  用时={elapsed * 1000:.1f}ms")
    print("动作序列:", " ".join(a.name for a in p.actions))
    print("逐步占用证据（每步扫掠格 vs blocked[t]∪blocked[t+1]，交集必须为空）:")
    for i, e in enumerate(p.evidence):
        ban = sorted(set(tt.blocked_at(i)) | set(tt.blocked_at(i + 1)))
        assert not (set(e.swept_cells) & set(ban)), (i, e.swept_cells, ban)
        body_now = sorted(grid.pose_cells(e.from_pose))
        print(f"  step {i}: t={e.time} {e.action.name:<11} {e.from_pose} -> {e.to_pose}  "
              f"cost={e.cost}  本步前车体={body_now}  扫掠={sorted(e.swept_cells)}  禁占并集={ban}")

    # 重放验证：逐步重新执行空间扫掠 + 起止时刻禁占检查
    final_state, evidence = replay_timed(grid, tt, start, p.actions)
    assert final_state.pose == goal and final_state.time == result.arrival_time
    assert sum(e.cost for e in evidence) == p.total_cost
    print(f"重放验证通过：{final_state.time} 步后到达 {final_state.pose}，无碰撞、无禁占冲突。")


def demo_timed_failure() -> None:
    print("\n=== 场景 4：离散时间失败示例——终点稍后被封 & 时域不足 ===")
    grid = Grid(4, 1)
    start = Pose(0, 0, Heading.EAST)
    goal = Pose(2, 0, Heading.EAST)

    # 4a. 终点从 t=1 起一直被封：时域内任何到达都撞禁占 → NO_PATH_WITHIN_HORIZON
    sealed = Timetable({t: [(2, 0), (3, 0)] for t in range(1, 8)})
    r = plan_timed(grid, sealed, start, goal, horizon=7)
    print(f"[时域无路] 终点 t>=1 被封 -> status={r.status.value}（expansions={r.expansions}）")
    # 同一几何静态规划有路：不能把时域无路报成静态几何无路
    r_static = plan(grid, start, goal)
    print(f"[几何有路] 忽略时刻表的静态 plan -> status={r_static.status.value} "
          f"cost={r_static.plan.total_cost}")
    assert r.status is SearchStatus.NO_PATH_WITHIN_HORIZON
    assert r.plan is None
    assert r_static.status is SearchStatus.OPTIMAL

    # 4b. 时域太短（到目标至少 2 步，只给 1 步）同样只报时域内无路
    r_short = plan_timed(grid, Timetable({}), start, goal, horizon=1)
    print(f"[时域不足] horizon=1 而至少需要 2 步 -> status={r_short.status.value}")
    assert r_short.status is SearchStatus.NO_PATH_WITHIN_HORIZON

    # 4c. 起刻车身被禁占：构造期不静默纠正，调用处给可定位 ValueError
    try:
        plan_timed(grid, Timetable({0: [(1, 0)]}), start, goal, horizon=4)
    except ValueError as exc:
        print(f"[拒绝] start 占用格在 t=0 被禁占 -> ValueError: {exc}")


def main() -> None:
    t0 = time.perf_counter()
    demo_normal()
    demo_rejection()
    demo_timed_normal()
    demo_timed_failure()
    print(f"\n总用时 {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
