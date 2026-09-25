"""固定输入演示：真实计算一条带扫掠证据的路径并重放验证，再实际触发一个拒绝边界。

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
    TimedGrid,
    plan,
    replay,
    timed_plan,
    timed_replay,
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


def demo_timed() -> None:
    # 离散时间场景：1 格宽走廊，移动障碍自右向左穿过，机器人必须倒车让行再等其通过。
    grid = Grid(5, 1)
    start = Pose(2, 0, Heading.EAST)  # 占用 (2,0),(3,0)
    goal = Pose(3, 0, Heading.EAST)
    blocked = {1: {(4, 0)}, 2: {(4, 0), (3, 0)}, 3: {(3, 0), (2, 0)}}  # 每时刻禁占格
    tgrid = TimedGrid(grid, 8, blocked=blocked)
    print("\n=== 场景 3：离散时间规划（移动障碍 + 倒车让行 + WAIT） ===")
    print(f"走廊 5x1，start={start}  goal={goal}  horizon=8  wait_cost=1")
    print("禁占格:", {t: sorted(c) for t, c in sorted(blocked.items())})

    t0 = time.perf_counter()
    result = timed_plan(tgrid, start, goal, wait_cost=1)
    elapsed = time.perf_counter() - t0
    assert result.status is SearchStatus.OPTIMAL, result.status
    p = result.plan
    print(f"status={result.status.value}  expansions={result.expansions}  "
          f"total_cost={p.total_cost}  arrival_time={p.arrival_time}  用时={elapsed * 1000:.1f}ms")
    print("动作序列:", " ".join(a.name for a in p.actions))
    print("逐步扫掠证据（每步检查的格子，须避开静态障碍与起止两时刻禁占格并集）:")
    for e in p.evidence:
        print(f"  t={e.from_time}->{e.to_time}: {e.action.name:<10} {e.from_pose} -> {e.to_pose}  "
              f"swept={sorted(e.swept_cells)} cost={e.cost}")

    # 重放验证：逐步重新执行，确认每步占用证据成立且精确到达（姿态与时刻）
    final_pose, final_time, evidence = timed_replay(tgrid, start, p.actions, wait_cost=1)
    assert (final_pose, final_time) == (goal, p.arrival_time)
    assert sum(e.cost for e in evidence) == p.total_cost
    print(f"重放验证通过：t={final_time} 到达 {final_pose}，{len(evidence)} 步无碰撞。")


def demo_timed_failure() -> None:
    # 失败示例：终点稍后被封——时域内无路，但静态几何有解，两者必须区分。
    grid = Grid(4, 1)
    start = Pose(0, 0, Heading.EAST)
    goal = Pose(2, 0, Heading.EAST)  # 占用 (2,0),(3,0)
    blocked = {t: {(3, 0)} for t in range(2, 7)}  # 时刻 2 起 (3,0) 被封
    tgrid = TimedGrid(grid, 6, blocked=blocked)
    print("\n=== 场景 4：失败示例（终点稍后被封） ===")
    r = timed_plan(tgrid, start, goal, wait_cost=1)
    print(f"[时域内无路] 时刻 2 起 (3,0) 被封 -> status={r.status.value}"
          "（只证明该时域内无路，不代表静态几何无解）")
    static = plan(grid, start, goal)
    print(f"[对照] 同一几何的静态规划 -> status={static.status.value} "
          f"total_cost={static.plan.total_cost}（静态几何其实有解）")
    assert r.status is SearchStatus.NO_PATH_IN_HORIZON
    assert static.status is SearchStatus.OPTIMAL

    # 非法输入：时域外的禁占格时刻键必须被可定位地拒绝
    try:
        TimedGrid(grid, 3, blocked={4: {(0, 0)}})
    except ValueError as exc:
        print(f"[拒绝] TimedGrid(blocked={{4: ...}}, horizon=3) -> ValueError: {exc}")


def main() -> None:
    t0 = time.perf_counter()
    demo_normal()
    demo_rejection()
    demo_timed()
    demo_timed_failure()
    print(f"\n总用时 {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
