"""重规划迭代测试：执行前缀 + 一次静态障碍更新 + 全量重规划。

手算期望值（不调用被测实现生成）：
- 7x5 空旷网格绕单个障碍 (4,2) 的最优绕路代价 = 82
  （5 次前进 50 + 4 次转向 32，转向点受 2x2 扫掠限制只能下移到 x=5）；
- 其余有/无解判定同时与 tests/dijkstra_reference.py（独立全状态 Dijkstra）对照。
"""

from __future__ import annotations

import random
import unittest

from pose_grid_planner import (
    Action,
    Grid,
    Heading,
    Plan,
    Pose,
    ReplanningStatus,
    SearchStatus,
    plan,
    replay,
    replan,
)
from dijkstra_reference import dijkstra_reference

N, E, S, W = Heading.NORTH, Heading.EAST, Heading.SOUTH, Heading.WEST


def to_ref(pose: Pose) -> tuple[int, int, int]:
    return (pose.x, pose.y, int(pose.heading))


def make_previous(grid: Grid, start: Pose, goal: Pose) -> Plan:
    r = plan(grid, start, goal)
    assert r.status is SearchStatus.OPTIMAL
    return r.plan


class GridObstacleUpdateTest(unittest.TestCase):
    """Grid.with_added_obstacles：不可变、构造式校验、幂等。"""

    def test_returns_new_grid_and_keeps_old(self) -> None:
        g = Grid(3, 2, obstacles=[(0, 0)])
        g2 = g.with_added_obstacles([(2, 1)])
        self.assertEqual(g.obstacles, frozenset({(0, 0)}))  # 旧对象不变
        self.assertEqual(g2.obstacles, frozenset({(0, 0), (2, 1)}))
        self.assertEqual(g2.width, 3)
        self.assertEqual(g2.height, 2)
        self.assertIsNot(g, g2)

    def test_re_adding_existing_obstacle_is_idempotent(self) -> None:
        # 已是障碍的格子重复声明：集合并集，不报错（不算新信息）
        g = Grid(3, 3, obstacles=[(1, 1)])
        g2 = g.with_added_obstacles([(1, 1), (2, 2)])
        self.assertEqual(g2.obstacles, frozenset({(1, 1), (2, 2)}))

    def test_duplicate_within_batch_rejected(self) -> None:
        g = Grid(3, 3)
        with self.assertRaisesRegex(ValueError, r"added_obstacles\[1\].*重复"):
            g.with_added_obstacles([(0, 0), (0, 0)])

    def test_bad_cell_locatable_and_no_partial_update(self) -> None:
        g = Grid(3, 3)
        with self.assertRaisesRegex(ValueError, r"added_obstacles\[0\]"):
            g.with_added_obstacles([(9, 0)])  # 越界
        with self.assertRaises(TypeError):
            g.with_added_obstacles([(True, 0)])  # bool 不算整数
        with self.assertRaises(TypeError):
            g.with_added_obstacles("(0,0)")
        self.assertEqual(g.obstacles, frozenset())  # 失败后旧对象原样

    def test_batch_all_or_nothing(self) -> None:
        # 批次中第 3 个格子非法：整个批次失败，前两个也不会生效（新网格根本不构造）
        g = Grid(3, 3)
        with self.assertRaisesRegex(ValueError, r"added_obstacles\[2\]"):
            g.with_added_obstacles([(0, 0), (1, 1), (99, 99)])
        self.assertEqual(g.obstacles, frozenset())


class ReplanNormalTest(unittest.TestCase):
    """正常重规划：路径被堵后的绕路、更新无影响、前缀保留。"""

    def setUp(self) -> None:
        # 7x5 空旷网格。旧规划 (1,2,E) -> (5,2,E)：4 次前进，代价 40。
        self.grid = Grid(7, 5)
        self.start = Pose(1, 2, E)
        self.goal = Pose(5, 2, E)
        self.previous = make_previous(self.grid, self.start, self.goal)
        self.assertEqual(self.previous.total_cost, 40)

    def test_detour_hand_computed_cost_82(self) -> None:
        # 执行 1 步后当前姿态 (2,2,E)，占用 (2,2)(3,2)；新障碍 (4,2) 堵住直线路径，
        # 但不压当前姿态、也不压目标（目标占用 (5,2)(6,2)）。
        r = replan(self.grid, self.start, self.goal, self.previous,
                   executed=1, added_obstacles=[(4, 2)])
        self.assertEqual(r.status, ReplanningStatus.OPTIMAL)
        self.assertEqual(r.current_pose, Pose(2, 2, E))
        self.assertEqual(r.executed, 1)

        # 前缀保留：恰好 1 个动作，与旧规划第一步相同，证据为旧网格历史值
        self.assertEqual(r.prefix.actions, (Action.FORWARD,))
        self.assertEqual(r.prefix.total_cost, 10)
        self.assertEqual(r.plan.actions[:1], self.previous.actions[:1])

        # 手算：绕路需 5 次前进 + 4 次转向 = 50 + 32 = 82（上下两条对称最优）
        self.assertEqual(r.remaining.total_cost, 82)
        self.assertEqual(sum(a is Action.FORWARD or a is Action.BACKWARD
                             for a in r.remaining.actions), 5)
        self.assertEqual(sum(a in (Action.TURN_LEFT, Action.TURN_RIGHT)
                             for a in r.remaining.actions), 4)
        # 合并路径代价 = 前缀 + 剩余
        self.assertEqual(r.plan.total_cost, 10 + 82)

        # 与独立 Dijkstra 参考对照（不调用被测核心）
        expected = dijkstra_reference(
            7, 5, frozenset({(4, 2)}), to_ref(r.current_pose), to_ref(self.goal)
        )
        self.assertEqual(expected, 82)

        # 合并后的完整路径必须能在更新后的网格上逐步重放通过且精确到达
        final, evidence = replay(r.updated_grid, self.start, r.plan.actions)
        self.assertEqual(final, self.goal)
        self.assertEqual(len(evidence), len(r.plan.actions))
        # 更新后的网格确实包含新障碍；旧网格未被修改
        self.assertIn((4, 2), r.updated_grid.obstacles)
        self.assertNotIn((4, 2), self.grid.obstacles)

    def test_update_without_effect_keeps_suffix(self) -> None:
        # 障碍只加在身后：剩余最优解就是旧规划后缀（前进 x3，代价 30）
        r = replan(self.grid, self.start, self.goal, self.previous,
                   executed=1, added_obstacles=[(0, 2)])
        self.assertEqual(r.status, ReplanningStatus.OPTIMAL)
        self.assertEqual(r.remaining.actions, self.previous.actions[1:])
        self.assertEqual(r.remaining.total_cost, 30)
        self.assertEqual(r.plan.actions, self.previous.actions)
        self.assertEqual(r.plan.total_cost, 40)

    def test_executed_zero_replans_from_start(self) -> None:
        r = replan(self.grid, self.start, self.goal, self.previous,
                   executed=0, added_obstacles=[(4, 2)])
        self.assertEqual(r.status, ReplanningStatus.OPTIMAL)
        self.assertEqual(r.current_pose, self.start)
        self.assertEqual(r.prefix.actions, ())
        expected = dijkstra_reference(
            7, 5, frozenset({(4, 2)}), to_ref(self.start), to_ref(self.goal)
        )
        self.assertEqual(r.plan.total_cost, expected)

    def test_empty_previous_start_equals_goal(self) -> None:
        # 旧规划为空（起终相同、零代价）：无更新时空剩余段立即成功；
        # 更新压住当前/目标姿态时按 PREFIX_BLOCKED 处理。
        pose = Pose(1, 1, E)
        empty = Plan(actions=(), total_cost=0, evidence=())
        r = replan(self.grid, pose, pose, empty, executed=0, added_obstacles=[])
        self.assertEqual(r.status, ReplanningStatus.OPTIMAL)
        self.assertEqual(r.plan.total_cost, 0)
        self.assertEqual(r.remaining.actions, ())
        self.assertEqual(r.expansions, 0)
        r2 = replan(self.grid, pose, pose, empty, executed=0, added_obstacles=[(2, 1)])
        self.assertEqual(r2.status, ReplanningStatus.PREFIX_BLOCKED)

    def test_generator_obstacles_accepted(self) -> None:
        r = replan(self.grid, self.start, self.goal, self.previous,
                   executed=1, added_obstacles=(c for c in [(0, 0)]))
        self.assertEqual(r.status, ReplanningStatus.OPTIMAL)

    def test_executed_all_at_goal_empty_remaining(self) -> None:        # 全部执行完，当前即目标：即便障碍加在别处，剩余段为空，立即 OPTIMAL
        r = replan(self.grid, self.start, self.goal, self.previous,
                   executed=4, added_obstacles=[(0, 0)])
        self.assertEqual(r.status, ReplanningStatus.OPTIMAL)
        self.assertEqual(r.current_pose, self.goal)
        self.assertEqual(r.remaining.actions, ())
        self.assertEqual(r.remaining.total_cost, 0)
        self.assertEqual(r.plan.actions, self.previous.actions)
        self.assertEqual(r.expansions, 0)

    def test_obstacle_already_present_in_batch_is_fine(self) -> None:
        g = Grid(7, 5, obstacles=[(0, 0)])
        previous = make_previous(g, self.start, self.goal)
        # executed=1 时当前姿态占用 (2,2)(3,2)，(4,2) 尚未被占用
        r = replan(g, self.start, self.goal, previous,
                   executed=1, added_obstacles=[(0, 0), (4, 2)])
        self.assertEqual(r.status, ReplanningStatus.OPTIMAL)
        self.assertEqual(r.updated_grid.obstacles, frozenset({(0, 0), (4, 2)}))


class ReplanFailureTest(unittest.TestCase):
    """失败用例：路径堵死、目标被压、当前姿态被压、预算耗尽。"""

    def test_no_path_when_corridor_sealed(self) -> None:
        # 7x1 直走廊：执行 2 步后 (4,0) 被堵，前方区段与目标隔断，已证明无解
        g = Grid(7, 1)
        start, goal = Pose(0, 0, E), Pose(5, 0, E)
        previous = make_previous(g, start, goal)
        r = replan(g, start, goal, previous, executed=2, added_obstacles=[(4, 0)])
        self.assertEqual(r.status, ReplanningStatus.NO_PATH)
        self.assertIsNone(r.plan)
        self.assertIsNone(r.remaining)
        # 前缀仍保留用于定位
        self.assertEqual(r.prefix.actions, (Action.FORWARD, Action.FORWARD))
        self.assertEqual(r.current_pose, Pose(2, 0, E))
        # 独立参考同样判定无解
        self.assertIsNone(
            dijkstra_reference(7, 1, frozenset({(4, 0)}), (2, 0, int(E)), (5, 0, int(E)))
        )

    def test_no_path_when_goal_crushed(self) -> None:
        g = Grid(7, 5)
        start, goal = Pose(1, 2, E), Pose(5, 2, E)
        previous = make_previous(g, start, goal)
        # (6,2) 是目标朝向 E 的前伸格：目标姿态被压住
        r = replan(g, start, goal, previous, executed=2, added_obstacles=[(6, 2)])
        self.assertEqual(r.status, ReplanningStatus.NO_PATH)
        self.assertEqual(r.current_pose, Pose(3, 2, E))
        self.assertGreaterEqual(r.expansions, 0)

    def test_prefix_blocked_when_obstacle_on_current_pose(self) -> None:
        g = Grid(7, 1)
        start, goal = Pose(0, 0, E), Pose(5, 0, E)
        previous = make_previous(g, start, goal)
        # 执行 2 步后当前姿态占用 (2,0)(3,0)；(3,0) 被堵 -> 状态不安全
        r = replan(g, start, goal, previous, executed=2, added_obstacles=[(3, 0)])
        self.assertEqual(r.status, ReplanningStatus.PREFIX_BLOCKED)
        self.assertIsNone(r.plan)
        self.assertEqual(r.current_pose, Pose(2, 0, E))
        self.assertEqual(r.prefix.actions, (Action.FORWARD, Action.FORWARD))

    def test_prefix_blocked_on_reference_cell(self) -> None:
        g = Grid(7, 5)
        start, goal = Pose(1, 2, E), Pose(5, 2, E)
        previous = make_previous(g, start, goal)
        r = replan(g, start, goal, previous, executed=2, added_obstacles=[(3, 2)])
        self.assertEqual(r.status, ReplanningStatus.PREFIX_BLOCKED)

    def test_prefix_blocked_when_update_arrives_at_goal_cell(self) -> None:
        # 已全部执行到目标，但更新压住目标占用格：不能假装成功
        g = Grid(7, 5)
        start, goal = Pose(1, 2, E), Pose(5, 2, E)
        previous = make_previous(g, start, goal)
        r = replan(g, start, goal, previous, executed=4, added_obstacles=[(5, 2)])
        self.assertEqual(r.status, ReplanningStatus.PREFIX_BLOCKED)
        self.assertIsNone(r.plan)

    def test_obstacle_behind_does_not_block_prefix(self) -> None:
        # 身后格（已离开）被堵不算前缀碰撞，仍能正常重规划
        g = Grid(7, 5)
        start, goal = Pose(1, 2, E), Pose(5, 2, E)
        previous = make_previous(g, start, goal)
        r = replan(g, start, goal, previous, executed=2, added_obstacles=[(1, 2)])
        self.assertEqual(r.status, ReplanningStatus.OPTIMAL)

    def test_budget_exhausted_distinct(self) -> None:
        g = Grid(7, 5)
        start, goal = Pose(1, 2, E), Pose(5, 2, E)
        previous = make_previous(g, start, goal)
        r = replan(g, start, goal, previous, executed=1,
                   added_obstacles=[(4, 2)], max_expansions=0)
        self.assertEqual(r.status, ReplanningStatus.BUDGET_EXHAUSTED)
        self.assertIsNone(r.plan)
        self.assertEqual(r.expansions, 0)
        # 放开预算后同一调用应当成功（无状态残留）
        r2 = replan(g, start, goal, previous, executed=1, added_obstacles=[(4, 2)])
        self.assertEqual(r2.status, ReplanningStatus.OPTIMAL)


class ReplanValidationTest(unittest.TestCase):
    """非法输入：可定位错误，且无半更新状态。"""

    def setUp(self) -> None:
        self.g = Grid(4, 4)
        self.start = Pose(0, 1, E)
        self.goal = Pose(2, 1, E)
        self.previous = make_previous(self.g, self.start, self.goal)

    def test_executed_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "executed"):
            replan(self.g, self.start, self.goal, self.previous,
                   executed=-1, added_obstacles=[])
        with self.assertRaisesRegex(ValueError, "executed"):
            replan(self.g, self.start, self.goal, self.previous,
                   executed=len(self.previous.actions) + 1, added_obstacles=[])
        with self.assertRaises(TypeError):
            replan(self.g, self.start, self.goal, self.previous,
                   executed=True, added_obstacles=[])
        with self.assertRaises(TypeError):
            replan(self.g, self.start, self.goal, self.previous,
                   executed=1.0, added_obstacles=[])

    def test_bad_types(self) -> None:
        with self.assertRaises(TypeError):
            replan("grid", self.start, self.goal, self.previous, executed=0)
        with self.assertRaises(TypeError):
            replan(self.g, (0, 1, E), self.goal, self.previous, executed=0)
        with self.assertRaises(TypeError):
            replan(self.g, self.start, self.goal, ("F",), executed=0)

    def test_bad_added_obstacle_locatable(self) -> None:
        with self.assertRaisesRegex(ValueError, r"added_obstacles\[1\]"):
            replan(self.g, self.start, self.goal, self.previous,
                   executed=1, added_obstacles=[(0, 0), (99, 99)])
        with self.assertRaisesRegex(ValueError, r"重复"):
            replan(self.g, self.start, self.goal, self.previous,
                   executed=1, added_obstacles=[(0, 0), (0, 0)])

    def test_previous_must_reach_goal_on_old_grid(self) -> None:
        other_goal = Pose(2, 2, E)
        other = make_previous(self.g, self.start, other_goal)
        with self.assertRaisesRegex(ValueError, "终点与 goal 不一致"):
            replan(self.g, self.start, self.goal, other, executed=0)

    def test_tampered_previous_rejected(self) -> None:
        bad = Plan(actions=(Action.BACKWARD,), total_cost=10,
                   evidence=self.previous.evidence)
        with self.assertRaises(ValueError):
            replan(self.g, self.start, self.goal, bad, executed=0)
        # 动作与证据长度不一致
        bad2 = Plan(actions=self.previous.actions + (Action.FORWARD,),
                    total_cost=self.previous.total_cost, evidence=self.previous.evidence)
        with self.assertRaises(ValueError):
            replan(self.g, self.start, self.goal, bad2, executed=0)

    def test_start_goal_still_validated_on_old_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "start.*越界"):
            replan(self.g, Pose(3, 0, E), self.goal, self.previous, executed=0)

    def test_no_partial_state_after_failure(self) -> None:
        before = self.g.obstacles
        with self.assertRaises(ValueError):
            replan(self.g, self.start, self.goal, self.previous,
                   executed=1, added_obstacles=[(99, 0)])
        self.assertEqual(self.g.obstacles, before)  # 旧网格原样
        # 失败后用合法输入仍可正常调用
        r = replan(self.g, self.start, self.goal, self.previous,
                   executed=1, added_obstacles=[])
        self.assertEqual(r.status, ReplanningStatus.OPTIMAL)


class ReplanRandomCrossCheckTest(unittest.TestCase):
    """确定性随机场景：重规划状态与剩余代价全面对照独立 Dijkstra。"""

    def test_random_cases(self) -> None:
        rng = random.Random(20260924)
        checked = 0
        for _ in range(120):
            w, h = rng.randint(3, 8), rng.randint(3, 8)
            obstacles = {(x, y)
                         for x in range(w) for y in range(h)
                         if rng.random() < 0.18}
            g = Grid(w, h, obstacles=obstacles)
            free_poses = [
                Pose(x, y, Heading(hd))
                for x in range(w) for y in range(h) for hd in range(4)
                if g.is_pose_free(Pose(x, y, Heading(hd)))
            ]
            if len(free_poses) < 2:
                continue
            start, goal = rng.sample(free_poses, 2)
            first = plan(g, start, goal)
            if first.status is not SearchStatus.OPTIMAL:
                continue
            previous = first.plan
            executed = rng.randrange(len(previous.actions) + 1)
            current = start if executed == 0 else previous.evidence[executed - 1].to_pose
            added = [
                (rng.randrange(w), rng.randrange(h))
                for _ in range(rng.randint(0, 4))
            ]
            try:
                r = replan(g, start, goal, previous,
                           executed=executed, added_obstacles=added)
            except ValueError:
                # 批次内重复会抛错：与手算条件一致，跳过这类样本
                continue

            updated_obs = obstacles | set(added)
            current_free = all(
                0 <= c[0] < w and 0 <= c[1] < h and c not in updated_obs
                for c in g.pose_cells(current)
            )
            goal_free = all(
                0 <= c[0] < w and 0 <= c[1] < h and c not in updated_obs
                for c in g.pose_cells(goal)
            )

            if not current_free:
                self.assertEqual(r.status, ReplanningStatus.PREFIX_BLOCKED)
                checked += 1
                continue
            self.assertEqual(r.updated_grid.obstacles, updated_obs)
            self.assertEqual(r.current_pose, current)
            self.assertEqual(len(r.prefix.actions), executed)

            if not goal_free:
                self.assertEqual(r.status, ReplanningStatus.NO_PATH)
                checked += 1
                continue

            expected = dijkstra_reference(
                w, h, frozenset(updated_obs), to_ref(current), to_ref(goal)
            )
            if expected is None:
                self.assertEqual(r.status, ReplanningStatus.NO_PATH)
            else:
                self.assertEqual(r.status, ReplanningStatus.OPTIMAL)
                self.assertEqual(r.remaining.total_cost, expected)
                self.assertEqual(
                    r.plan.total_cost, r.prefix.total_cost + r.remaining.total_cost
                )
                # 剩余段在更新后网格上、从当前位置重放必须安全并精确到达
                final, rem_ev = replay(
                    r.updated_grid, current, r.remaining.actions
                )
                self.assertEqual(final, goal)
                # 前缀是旧世界的历史：在旧网格上、从旧起点重放必须仍然成立
                hist_final, pre_ev = replay(g, start, r.prefix.actions)
                self.assertEqual(hist_final, current)
                self.assertEqual(len(rem_ev) + len(pre_ev), len(r.plan.actions))
                # 若前缀每一步的扫掠格在更新后仍全空，完整路径可在新网格整体重放；
                # 否则只能也只应重放剩余段（身后扫掠区被堵不影响历史事实）。
                prefix_swept = {c for e in r.prefix.evidence for c in e.swept_cells}
                if all(r.updated_grid.is_free(c) for c in prefix_swept):
                    full_final, _ev = replay(r.updated_grid, start, r.plan.actions)
                    self.assertEqual(full_final, goal)
                else:
                    with self.assertRaises(Exception):
                        replay(r.updated_grid, start, r.plan.actions)
            checked += 1
        self.assertGreaterEqual(checked, 60)


if __name__ == "__main__":
    unittest.main()
