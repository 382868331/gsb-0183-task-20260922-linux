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
    ReplanningStatus,
    SearchStatus,
    plan,
    replay,
    replan,
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


def demo_replanning() -> None:
    print("\n=== 场景 3：执行前缀后收到静态障碍更新，保留前缀重新规划 ===")
    # 9x5 空旷网格。旧路径 (1,2,E) -> (7,2,E)：6 次前进，代价 60。
    grid = Grid(9, 5)
    start = Pose(1, 2, Heading.EAST)
    goal = Pose(7, 2, Heading.EAST)
    first = plan(grid, start, goal)
    assert first.status is SearchStatus.OPTIMAL
    previous = first.plan
    print(f"旧规划：{len(previous.actions)} 步，总代价 {previous.total_cost}（一路向东直行）")

    # 执行了前 2 步，停在 (3,2,E)（占用 (3,2)(4,2)）；此时 (5,2) 新增静态障碍，
    # 正好堵住剩余直线（下一步前进的扫掠并集会撞到 (5,2)）。
    executed = 2
    added = [(5, 2)]
    print(f"已执行 {executed} 步 -> 当前 {Pose(3, 2, Heading.EAST)}；"
          f"收到障碍更新 {added}")
    r = replan(grid, start, goal, previous, executed=executed, added_obstacles=added)
    assert r.status is ReplanningStatus.OPTIMAL
    print(f"status={r.status.value}  剩余段 expansions={r.expansions}")
    print(f"保留前缀: {[a.name for a in r.prefix.actions]}  代价 {r.prefix.total_cost}")
    print(f"新剩余段: {' '.join(a.name for a in r.remaining.actions)}")
    print(f"剩余段代价 {r.remaining.total_cost}（手算：6 次前进 60 + 4 次转向 32 = 92），"
          f"完整路径代价 {r.plan.total_cost}（前缀 20 + 剩余 92）")

    # 剩余段从当前位置在更新后的网格上重放；完整路径同样可整体重放（前缀扫掠区未被堵）
    final, _ = replay(r.updated_grid, r.current_pose, r.remaining.actions)
    assert final == goal
    full_final, _ = replay(r.updated_grid, start, r.plan.actions)
    assert full_final == goal
    poses = [start] + [e.to_pose for e in r.plan.evidence]
    print("更新后网格上的完整路径姿态（'#' 为新障碍）：")
    print(render(r.updated_grid, poses))

    # 更新无影响的情形：障碍只落在身后，剩余段应保持旧后缀不变
    r2 = replan(grid, start, goal, previous, executed=2, added_obstacles=[(0, 2)])
    assert r2.status is ReplanningStatus.OPTIMAL
    assert r2.remaining.actions == previous.actions[2:]
    print(f"[无影响] 障碍只加在身后 (0,2)：剩余段仍是旧后缀，代价 {r2.remaining.total_cost}")


def demo_replan_failures() -> None:
    print("\n=== 场景 4：重规划的实际失败边界 ===")
    # 4a. 路径被堵死：9x1 直走廊，执行 2 步后前方 (6,0) 被堵，已证明无路可达
    corridor = Grid(9, 1)
    c_start, c_goal = Pose(0, 0, Heading.EAST), Pose(7, 0, Heading.EAST)
    c_prev = plan(corridor, c_start, c_goal).plan
    blocked = replan(corridor, c_start, c_goal, c_prev, executed=2, added_obstacles=[(6, 0)])
    print(f"[无路可达] 走廊 (6,0) 被堵 -> status={blocked.status.value}"
          f"（前缀 {len(blocked.prefix.actions)} 步保留，plan={blocked.plan}）")
    assert blocked.status is ReplanningStatus.NO_PATH

    # 4b. 更新直接压住机器人当前占用格：当前状态不安全，拒绝继续
    crushed = replan(corridor, c_start, c_goal, c_prev, executed=2, added_obstacles=[(3, 0)])
    print(f"[前缀受阻] 当前姿态占用 (2,0)(3,0)，(3,0) 被堵 -> status={crushed.status.value}")
    assert crushed.status is ReplanningStatus.PREFIX_BLOCKED

    # 4c. 非法更新批次：第 2 个坐标越界，整批失败，旧网格不发生任何改变
    before = corridor.obstacles
    try:
        replan(corridor, c_start, c_goal, c_prev,
               executed=2, added_obstacles=[(0, 0), (99, 0)])
    except ValueError as exc:
        print(f"[拒绝] 越界障碍批次 -> ValueError: {exc}")
    assert corridor.obstacles == before

    # 4d. 剩余搜索预算耗尽与无路可达严格区分
    open_grid = Grid(9, 5)
    d_start, d_goal = Pose(1, 2, Heading.EAST), Pose(7, 2, Heading.EAST)
    d_prev = plan(open_grid, d_start, d_goal).plan
    exhausted = replan(open_grid, d_start, d_goal, d_prev,
                       executed=1, added_obstacles=[(5, 2)], max_expansions=0)
    print(f"[预算] max_expansions=0 -> status={exhausted.status.value}"
          f"（未找到也未证明无解，可加大预算重试）")
    assert exhausted.status is ReplanningStatus.BUDGET_EXHAUSTED


def main() -> None:
    t0 = time.perf_counter()
    demo_normal()
    demo_rejection()
    demo_replanning()
    demo_replan_failures()
    print(f"\n总用时 {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
