"""铰接拖车姿态格迭代演示：真实搜索、逐步扫掠证据和预算边界。

运行：python demo_iteration.py
"""

from __future__ import annotations

import time

from pose_grid_planner import (
    ArticulatedPose,
    Grid,
    Heading,
    SearchStatus,
    plan_articulated,
    replay_articulated,
)


def print_evidence(plan) -> None:
    for index, step in enumerate(plan.evidence):
        print(
            f"  {index}: {step.action.name:<10} {step.from_pose} -> {step.to_pose}\n"
            f"     tractor={sorted(step.tractor_swept_cells)}\n"
            f"     trailer={sorted(step.trailer_swept_cells)}\n"
            f"     bar    ={sorted(step.hitch_bar_swept_cells)}  cost={step.cost}"
        )
        print(f"     articulation_turns={step.to_pose.articulation_turns()}")


def main() -> None:
    started = time.perf_counter()
    print("=== 铰接拖车：先拉直，再倒车进入 ===")
    rows = ["...", "...", "#.#", "#.#", "..."]
    grid = Grid.from_rows(rows)
    start = ArticulatedPose(1, 1, Heading.EAST, Heading.NORTH)
    goal = ArticulatedPose(1, 3, Heading.NORTH, Heading.NORTH)
    print("地图（# 障碍）:")
    for row in rows:
        print(f"  {row}")
    print(f"start={start}")
    print(f"goal ={goal}")

    result = plan_articulated(grid, start, goal)
    assert result.status is SearchStatus.OPTIMAL, result.status
    plan = result.plan
    print(f"status={result.status.value} expansions={result.expansions} cost={plan.total_cost}")
    print("actions:", " -> ".join(action.name for action in plan.actions))
    print_evidence(plan)
    final_pose, evidence = replay_articulated(grid, start, plan.actions)
    assert final_pose == goal
    assert all(grid.is_free(cell) for step in evidence for cell in step.swept_cells)
    print(f"重放通过：最终状态 {final_pose}")

    print("\n=== 边界：预算不足不同于证明无路 ===")
    corner_grid = Grid.from_rows(["####", "####", "####", "....", "#...", "#..."])
    corner_start = ArticulatedPose(1, 3, Heading.EAST, Heading.EAST)
    corner_goal = ArticulatedPose(1, 4, Heading.SOUTH, Heading.SOUTH)
    budgeted = plan_articulated(corner_grid, corner_start, corner_goal, max_expansions=0)
    exhausted = plan_articulated(corner_grid, corner_start, corner_goal)
    print(f"max_expansions=0 -> {budgeted.status.value}（不能据此宣称无解）")
    print(f"完整搜索       -> {exhausted.status.value}（开集耗尽，证明窄转角对拖车无路）")
    assert budgeted.status is SearchStatus.BUDGET_EXHAUSTED
    assert exhausted.status is SearchStatus.NO_PATH
    print(f"\n总用时 {time.perf_counter() - started:.3f}s")


if __name__ == "__main__":
    main()
