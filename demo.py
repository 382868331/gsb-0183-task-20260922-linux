#!/usr/bin/env python3
"""固定输入演示：真实计算并打印一个正常规划结果与一个实际触发的异常边界。

运行：python demo.py    （约 8 秒内完成，无网络、无第三方依赖、无 sleep）
"""

from __future__ import annotations

import time

from pose_grid_planner import (
    Action,
    GridWorld,
    Heading,
    InvalidParameterError,
    PlanStatus,
    Pose,
    plan,
)

ARROW = {
    Heading.EAST: "→",
    Heading.NORTH: "↑",
    Heading.WEST: "←",
    Heading.SOUTH: "↓",
}
ACTION_NAME = {
    Action.FORWARD: "前进",
    Action.BACKWARD: "后退",
    Action.TURN_LEFT: "左转90°",
    Action.TURN_RIGHT: "右转90°",
}


def render(world: GridWorld, path_poses: list[Pose]) -> str:
    """把世界与规划路径渲染成字符图（路径占用格用 * 标注）。"""
    trail: set[tuple[int, int]] = set()
    for p in path_poses:
        trail.update(p.occupied())
    lines = []
    for y in range(world.height - 1, -1, -1):
        row = []
        for x in range(world.width):
            if (x, y) in world.obstacles:
                row.append("█")
            elif (x, y) in trail:
                row.append("*")
            else:
                row.append("·")
        lines.append("".join(row))
    return "\n".join(lines)


def replay(world: GridWorld, start: Pose, result) -> None:
    """独立重放校验：逐姿态、逐扫掠格检查碰撞与越界。"""
    cur = start
    for i, ev in enumerate(result.evidence, 1):
        for cx, cy in ev.swept_cells:
            assert world.is_free(cx, cy), f"第 {i} 步扫掠格 ({cx},{cy}) 碰撞/越界"
        assert set(cur.occupied()) <= set(ev.swept_cells)
        nx, ny, nh = ev.dst_pose
        nxt = Pose(nx, ny, nh)
        assert set(nxt.occupied()) <= set(ev.swept_cells)
        cur = nxt
    assert cur.state == result.goal


def section(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 场景一（正常结果）：带墙房间里的规划。
    # 9x7 世界，中部有墙并留出窄口；车从西南角出发，目标在东北角且要求
    # 朝向相反——必须绕路、前进、转向并用，路径中包含一段只能后退/绕行的
    # 窄口处理。全部数据在此确定性硬编码。
    # ------------------------------------------------------------------
    section("场景 1：正常规划（9x7 带墙房间，目标含朝向约束）")
    width, height = 9, 7
    obstacles = {
        # 竖墙 x=4，y=0..4；只在 y=5 留一个 1 格宽的口
        *( (4, y) for y in range(5) ),
        # 右侧一段矮墙，迫使车在房间内转向调头
        (6, 3), (7, 3),
    }
    world = GridWorld(width, height, obstacles)
    start = Pose(1, 1, Heading.EAST)
    goal = Pose(7, 5, Heading.WEST)
    print(f"网格 {width}x{height}，障碍 {len(obstacles)} 格")
    print(f"起点: 参考格({start.x},{start.y}) {ARROW[start.heading]}  占用 {start.occupied()}")
    print(f"终点: 参考格({goal.x},{goal.y}) {ARROW[goal.heading]}  占用 {goal.occupied()}")

    result = plan(world, start, goal)
    print(f"状态: {result.status.value}（扩展节点 {result.expanded}）")
    assert result.status is PlanStatus.SUCCESS, "固定输入场景必须有解"
    print(f"总代价: {result.total_cost}（前进10/后退14/转向8）")

    path_poses = [start] + [Pose(*ev.dst_pose) for ev in result.evidence]
    print("\n动作序列与逐步扫掠证据：")
    cur = start
    for i, ev in enumerate(result.evidence, 1):
        h0 = Heading(ev.src_pose[2])
        h1 = Heading(ev.dst_pose[2])
        print(
            f"  {i:2d}. {ACTION_NAME[ev.action]:>6}  代价{ev.cost:>2}  "
            f"({ev.src_pose[0]},{ev.src_pose[1]}){ARROW[h0]} -> "
            f"({ev.dst_pose[0]},{ev.dst_pose[1]}){ARROW[h1]}  "
            f"扫掠[{ev.swept_kind}]={list(ev.swept_cells)}"
        )
        cur = Pose(*ev.dst_pose)

    replay(world, start, result)
    print("\n重放校验通过：每步扫掠格均在界内且为空，终点姿态精确匹配。")
    print("\n路径图（█ 障碍，* 车体扫过的格）：")
    print(render(world, path_poses))

    # ------------------------------------------------------------------
    # 场景二（实际触发的拒绝边界）：整数参数拒绝 bool。
    # 用 True 冒充 width 会被立刻拒绝并给出可定位错误，而不是被当成 1。
    # ------------------------------------------------------------------
    section("场景 2：非法输入被拒绝（bool 冒充整数 width=True）")
    try:
        GridWorld(True, 7)  # type: ignore[arg-type]
    except InvalidParameterError as exc:
        print(f"按契约抛出 {type(exc).__name__}: {exc}")
        print("语义：bool 不被静默当作 1，调用方可据此定位参数 width。")
    else:
        raise AssertionError("应当拒绝 bool 输入")

    # 同时演示“搜索预算耗尽”与“证明无解”是两种不同结果
    section("附：预算耗尽 ≠ 证明无解（同一问题，两种预算）")
    tiny_budget = plan(world, start, goal, max_expansions=2)
    full = plan(world, start, goal)
    print(f"max_expansions=2 -> {tiny_budget.status.value}（不能判定是否有解）")
    print(f"默认预算         -> {full.status.value}，代价 {full.total_cost}")

    print(f"\n总耗时 {time.perf_counter() - t0:.2f} 秒。")


if __name__ == "__main__":
    main()
