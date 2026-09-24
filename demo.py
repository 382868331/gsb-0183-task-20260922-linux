"""固定输入演示：真实计算一条带扫掠证据的路径并重放验证，再实际触发一个拒绝边界。

还包含本次迭代：旧路径执行到指定姿态后接收一次静态障碍更新，保留已执行
前缀、在新网格上重新规划剩余路径的正常绕行、更新无影响、封死/被埋/参数
非法等边界。

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


def demo_replan() -> None:
    """本次迭代：执行 2 步后收到一次静态障碍更新，保留前缀重新规划。"""
    print("\n=== 场景 2：静态障碍更新后重规划（保留已执行前缀） ===")
    # 9x4 开阔区；y=0 是边界行。机器人朝东沿 y=0 直行去 (7,0,E)
    grid = Grid(9, 4)
    start = Pose(0, 0, Heading.EAST)
    goal = Pose(7, 0, Heading.EAST)
    old = plan(grid, start, goal)
    assert old.status is SearchStatus.OPTIMAL
    print(f"旧路径（{len(old.plan.actions)} 步, cost={old.plan.total_cost}）: "
          + " ".join(a.name for a in old.plan.actions))
    print(render(grid, [start, goal]))

    # 已执行 2 步，当前在 (2,0,E)；此时 (4,0) 新增静态障碍
    executed = 2
    print(f"\n-- 2a 正常绕行：执行前 {executed} 步后，正前方 (4,0) 出现障碍 --")
    r = replan(grid, start, goal, old.plan.actions, executed, [(4, 0)])
    print(f"保留前缀: {' '.join(a.name for a in r.prefix.actions)}  "
          f"(cost={r.prefix.total_cost})  当前姿态={r.current_pose}")
    print(f"status={r.status.value}  剩余 {len(r.remainder.actions)} 步, "
          f"剩余 cost={r.remainder.total_cost}, 全程 cost={r.total_cost}")
    print("剩余动作: " + " ".join(a.name for a in r.remainder.actions))
    # 剩余段在更新后网格上真实重放核对
    final, ev = replay(r.updated_grid, r.current_pose, r.remainder.actions)
    assert final == goal
    assert sum(e.cost for e in ev) == r.remainder.total_cost
    poses = [r.current_pose] + [e.to_pose for e in ev]
    print("剩余段路径（更新后网格，'#' 为新障碍）:")
    print(render(r.updated_grid, poses))
    print("重放核对通过：剩余段无碰撞精确到达目标（含朝向）。")

    # 3b 更新无影响：障碍在远处角落，剩余段仍是直线
    print("\n-- 2b 更新无影响：新增障碍在远处 (0,3) --")
    r2 = replan(grid, start, goal, old.plan.actions, executed, [(0, 3)])
    print(f"status={r2.status.value}  剩余动作: "
          + " ".join(a.name for a in r2.remainder.actions)
          + f"  cost={r2.remainder.total_cost}（仍为 5 次 FORWARD）")
    assert r2.remainder.actions == (Action.FORWARD,) * 5

    # 3c 无路可达：9x2 中 x=4 整列封死
    print("\n-- 2c 无路可达：x=4 整列在 9x2 内封死 --")
    narrow = Grid(9, 2)
    old3 = plan(narrow, start, goal)
    r3 = replan(narrow, start, goal, old3.plan.actions, executed, [(4, 0), (4, 1)])
    print(f"status={r3.status.value}（开集耗尽已证明无解） expansions={r3.search.expansions}；"
          f"已执行前缀仍保留：{' '.join(a.name for a in r3.prefix.actions)}")
    assert r3.status is SearchStatus.NO_PATH

    # 3d 实际错误：新障碍压在当前占用格上 / 参数非法
    print("\n-- 2d 实际触发的错误（不留半更新状态） --")
    try:
        replan(grid, start, goal, old.plan.actions, executed, [(3, 0)])  # (2,0,E) 的前端格
    except ValueError as exc:
        print(f"[矛盾更新] 障碍压在当前占用格 -> ValueError: {exc}")
    try:
        replan(grid, start, goal, old.plan.actions, 99, ())  # 超过旧路径长度
    except ValueError as exc:
        print(f"[非法步数] executed_actions=99 -> ValueError: {exc}")
    g_before = set(grid.obstacles)
    try:
        replan(grid, start, goal, old.plan.actions, executed, [(99, 0)])  # 越界
    except ValueError as exc:
        print(f"[越界障碍] (99,0) -> ValueError: {exc}")
    assert set(grid.obstacles) == g_before
    print("旧网格障碍集在全部失败调用后保持不变。")


def demo_rejection() -> None:
    print("\n=== 场景 3：实际触发的拒绝/异常边界 ===")
    # 3a. 非法数值输入：NaN 必须被可定位地拒绝（不静默纠正）
    try:
        Grid(float("nan"), 5)
    except ValueError as exc:
        print(f"[拒绝] Grid(width=nan) -> ValueError: {exc}")

    # 3b. bool 冒充整数参数：必须拒绝
    try:
        Grid(5, True)
    except TypeError as exc:
        print(f"[拒绝] Grid(height=True) -> TypeError: {exc}")

    # 3c. 搜索预算耗尽 vs 证明无解：两者必须可区分
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


def main() -> None:
    t0 = time.perf_counter()
    demo_normal()
    demo_replan()
    demo_rejection()
    print(f"\n总用时 {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
