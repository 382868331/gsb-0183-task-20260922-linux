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
    plan,
    replay,
)

ARROW = {Heading.NORTH: "^", Heading.EAST: ">", Heading.SOUTH: "v", Heading.WEST: "<"}


def render(grid: Grid, poses: list[Pose]) -> str:
    """ASCII 地图：'#' 障碍，数字为该格额外通行代价，方向箭头为路径姿态，'.' 空地。"""
    canvas = [
        [
            "#" if (x, y) in grid.obstacles
            else (str(grid.terrain_extra((x, y))) if grid.terrain_extra((x, y)) else ".")
            for x in range(grid.width)
        ]
        for y in range(grid.height)
    ]
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


def demo_terrain() -> None:
    # 地面代价场景：中间直道是高代价泥地（每格额外 9），上下两条土路免费。
    # 规划应绕行；同一路径重放代价必须与规划报告一致。
    print("\n=== 场景 2：不同地面通行代价（高代价地面可通过，但更贵） ===")
    grid = Grid.from_rows(
        [
            "..........",
            "..........",
            "..........",
        ],
        terrain_rows=[
            "..........",
            ".99999999.",
            "..........",
        ],
    )
    start = Pose(0, 1, Heading.EAST)
    goal = Pose(8, 1, Heading.EAST)
    print(render(grid, [start, goal]))
    print(f"start={start}  goal={goal}  （中间一行每格额外代价 9）")
    result = plan(grid, start, goal)
    assert result.status is SearchStatus.OPTIMAL, result.status
    p = result.plan
    print(f"status={result.status.value}  total_cost={p.total_cost}  steps={len(p.actions)}")
    print("动作序列:", " ".join(a.name for a in p.actions))
    detoured = any(e.to_pose.y != 1 for e in p.evidence)
    print(f"是否绕行高代价地面: {'是' if detoured else '否'}")
    assert detoured, "规划应选择更便宜的绕行路线"

    # 重放一致性：逐步重放的代价之和必须等于规划报告的总代价
    final_pose, evidence = replay(grid, start, p.actions)
    replay_cost = sum(e.cost for e in evidence)
    assert final_pose == goal
    assert replay_cost == p.total_cost, (replay_cost, p.total_cost)
    print(f"重放验证通过：到达 {final_pose}，重放代价 {replay_cost} == 规划代价 {p.total_cost}")
    poses = [start] + [e.to_pose for e in evidence]
    print(render(grid, poses))


def demo_rejection() -> None:
    print("\n=== 场景 3：实际触发的拒绝/异常边界 ===")

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


def main() -> None:
    t0 = time.perf_counter()
    demo_normal()
    demo_terrain()
    demo_rejection()
    print(f"\n总用时 {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
