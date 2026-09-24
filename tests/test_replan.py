"""迭代验收：旧路径执行到指定姿态后接收一次静态障碍更新，保留前缀并重规划。

期望值来源（不调用被测实现自证）：
- 正常绕行 / 无影响 / 封死场景的代价与有/无解，先由手算推导，并用与被测包
  完全独立的 tests/dijkstra_reference.py 全状态 Dijkstra 核对；
- 前缀内容直接等于旧 plan 动作切片；剩余段逐格扫掠用 transition 独立复核。
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
    ReplayError,
    SearchStatus,
    plan,
    replay,
    replan,
)
from dijkstra_reference import dijkstra_reference

N, E, S, W = Heading.NORTH, Heading.EAST, Heading.SOUTH, Heading.WEST

F, B = Action.FORWARD, Action.BACKWARD
TL, TR = Action.TURN_LEFT, Action.TURN_RIGHT


def ref_pose(pose: Pose) -> tuple[int, int, int]:
    return (pose.x, pose.y, int(pose.heading))


def assert_evidence_valid(
    testcase: unittest.TestCase,
    grid: Grid,
    start: Pose,
    actions: tuple[Action, ...],
    evidence: tuple,
    goal: Pose,
) -> None:
    """证据链在给定网格上自洽：from/to 衔接、扫掠格与规则重算一致且全空。"""
    pose = start
    testcase.assertEqual(tuple(e.action for e in evidence), actions)
    for e in evidence:
        testcase.assertEqual(e.from_pose, pose)
        result = transition_result(grid, pose, e.action)
        testcase.assertIsNotNone(result)
        nxt, swept = result
        testcase.assertEqual(nxt, e.to_pose)
        testcase.assertEqual(tuple(sorted(e.swept_cells)), tuple(sorted(swept)))
        for cell in swept:
            testcase.assertTrue(grid.is_free(cell))
        pose = nxt
    testcase.assertEqual(pose, goal)


def transition_result(grid: Grid, pose: Pose, action: Action):
    # 局部 import 避免顶部循环噪音；transition 是模型层唯一扫掠入口
    from pose_grid_planner import transition

    return transition(grid, pose, action)


def make_open_grid() -> Grid:
    """9x4 空网格；y=0 边界行让 (4,0) 障碍的绕行必须借助后退（手算场景）。"""
    return Grid(9, 4)


def original_east_plan(grid: Grid) -> Plan:
    r = plan(grid, Pose(0, 0, E), Pose(7, 0, E))
    assert r.status is SearchStatus.OPTIMAL
    return r.plan


class ReplanDetourTest(unittest.TestCase):
    """正常场景：执行 2 步后正前方 (4,0) 出现静态障碍，需保留前缀并绕行。"""

    def setUp(self) -> None:
        self.grid = make_open_grid()
        self.start = Pose(0, 0, E)
        self.goal = Pose(7, 0, E)
        self.old = original_east_plan(self.grid)
        self.result = replan(
            self.grid, self.start, self.goal, self.old.actions, 2, [(4, 0)]
        )

    def test_prefix_preserved_exactly(self) -> None:
        r = self.result
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        # 已执行前缀原样保留：2 次 FORWARD，代价 20，当前姿态 (2,0,EAST)
        self.assertEqual(r.prefix.actions, (F, F))
        self.assertEqual(r.prefix.total_cost, 20)
        self.assertEqual(r.executed_actions, 2)
        self.assertEqual(r.current_pose, Pose(2, 0, E))
        assert_evidence_valid(
            self, self.grid, self.start, r.prefix.actions, r.prefix.evidence, Pose(2, 0, E)
        )

    def test_remainder_cost_hand_and_reference(self) -> None:
        r = self.result
        # 手算 + 独立 Dijkstra：从 (2,0,E) 绕 (4,0) 到 (7,0,E) 最优 = 106
        # = 4 次转向*8 + 6 次前进*10 + 1 次后退*14 = 32+60+14
        expected = dijkstra_reference(
            9, 4, frozenset({(4, 0)}), ref_pose(r.current_pose), ref_pose(self.goal)
        )
        self.assertEqual(expected, 106)
        self.assertEqual(r.remainder.total_cost, 106)
        self.assertEqual(r.total_cost, 20 + 106)
        # 剩余段必须真的用到 BACKWARD：y=0 朝北转向扫掠越界，只能借后退回边界行
        self.assertIn(B, r.remainder.actions)
        turns = sum(1 for a in r.remainder.actions if a in (TL, TR))
        forwards = r.remainder.actions.count(F)
        self.assertEqual(turns, 4)
        self.assertEqual(forwards, 6)

    def test_remainder_evidence_on_updated_grid(self) -> None:
        r = self.result
        self.assertEqual(r.updated_grid.obstacles, frozenset({(4, 0)}))
        assert_evidence_valid(
            self, r.updated_grid, r.current_pose,
            r.remainder.actions, r.remainder.evidence, self.goal,
        )
        # 拼接动作在更新后网格上从原起点重放也必须无碰撞精确到达
        final, _ = replay(r.updated_grid, self.start, r.full_actions)
        self.assertEqual(final, self.goal)
        self.assertEqual(r.full_actions, r.prefix.actions + r.remainder.actions)

    def test_original_grid_and_plan_untouched(self) -> None:
        # 更新产生新网格；旧网格障碍集与旧 plan 都不变
        self.assertEqual(self.grid.obstacles, frozenset())
        self.assertEqual(self.old.actions, (F,) * 7)
        self.assertIsNot(self.grid, self.result.updated_grid)


class ReplanNoImpactTest(unittest.TestCase):
    """更新无影响：新增障碍远离旧路径，剩余段仍是原直线。"""

    def test_far_obstacle_keeps_straight_tail(self) -> None:
        grid = make_open_grid()
        start, goal = Pose(0, 0, E), Pose(7, 0, E)
        old = original_east_plan(grid)
        r = replan(grid, start, goal, old.actions, 2, [(0, 3)])
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        # 独立参考确认 (0,3) 不改变最优代价
        expected = dijkstra_reference(
            9, 4, frozenset({(0, 3)}), (2, 0, int(E)), (7, 0, int(E))
        )
        self.assertEqual(expected, 50)
        self.assertEqual(r.remainder.actions, (F,) * 5)
        self.assertEqual(r.remainder.total_cost, 50)
        self.assertEqual(r.total_cost, 70)

    def test_empty_update_is_equivalent_to_tail_of_original(self) -> None:
        grid = make_open_grid()
        start, goal = Pose(0, 0, E), Pose(7, 0, E)
        old = original_east_plan(grid)
        r = replan(grid, start, goal, old.actions, 3, ())
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.remainder.actions, old.actions[3:])
        self.assertEqual(r.total_cost, old.total_cost)


class ReplanBlockedTest(unittest.TestCase):
    """失败场景：路径被堵死 / 目标被埋 / 当前位姿被压。"""

    def test_wall_across_corridor_no_path(self) -> None:
        # 9x2：x=4 整列成墙，边界行无法绕行（2x2 转向扫掠不出界也跨不过墙）
        grid = Grid(9, 2)
        start, goal = Pose(0, 0, E), Pose(7, 0, E)
        old = plan(grid, start, goal).plan
        expected = dijkstra_reference(
            9, 2, frozenset({(4, 0), (4, 1)}), (2, 0, int(E)), (7, 0, int(E))
        )
        self.assertIsNone(expected)
        r = replan(grid, start, goal, old.actions, 2, [(4, 0), (4, 1)])
        self.assertEqual(r.status, SearchStatus.NO_PATH)
        self.assertIsNone(r.remainder)
        self.assertGreater(r.search.expansions, 0)  # 是搜尽证明，不是预算耗尽
        # 前缀仍被保留并返回
        self.assertEqual(r.prefix.actions, (F, F))
        self.assertEqual(r.current_pose, Pose(2, 0, E))
        with self.assertRaisesRegex(ValueError, "no_path"):
            r.total_cost
        with self.assertRaisesRegex(ValueError, "no_path"):
            r.full_actions

    def test_goal_buried_returns_no_path_without_search(self) -> None:
        grid = make_open_grid()
        start, goal = Pose(0, 0, E), Pose(7, 0, E)
        old = original_east_plan(grid)
        # goal 的参考格 (7,0) 被埋
        r = replan(grid, start, goal, old.actions, 2, [(7, 0)])
        self.assertEqual(r.status, SearchStatus.NO_PATH)
        self.assertIsNone(r.remainder)
        self.assertEqual(r.search.expansions, 0)  # 目标非法直接判定，不浪费搜索
        # 埋的是 goal 的前向相邻格 (8,0) 同样不可达
        r2 = replan(grid, start, goal, old.actions, 2, [(8, 0)])
        self.assertEqual(r2.status, SearchStatus.NO_PATH)

    def test_new_obstacle_on_current_pose_raises(self) -> None:
        grid = make_open_grid()
        start, goal = Pose(0, 0, E), Pose(7, 0, E)
        old = original_east_plan(grid)
        # 当前姿态 (2,0,E) 占用 (2,0) 与 (3,0)；压到任一格都矛盾
        with self.assertRaisesRegex(ValueError, "当前姿态"):
            replan(grid, start, goal, old.actions, 2, [(2, 0)])
        with self.assertRaisesRegex(ValueError, "当前姿态"):
            replan(grid, start, goal, old.actions, 2, [(3, 0)])


class ReplanBoundaryTest(unittest.TestCase):
    """边界：executed_actions=0、执行完整条路径（已在目标）。"""

    def test_zero_executed_replans_from_start(self) -> None:
        grid = make_open_grid()
        start, goal = Pose(0, 0, E), Pose(7, 0, E)
        old = original_east_plan(grid)
        r = replan(grid, start, goal, old.actions, 0, [(4, 0)])
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.prefix.actions, ())
        self.assertEqual(r.prefix.total_cost, 0)
        self.assertEqual(r.current_pose, start)
        expected = dijkstra_reference(
            9, 4, frozenset({(4, 0)}), (0, 0, int(E)), (7, 0, int(E))
        )
        self.assertEqual(r.remainder.total_cost, expected)

    def test_all_executed_already_at_goal(self) -> None:
        grid = make_open_grid()
        start, goal = Pose(0, 0, E), Pose(7, 0, E)
        old = original_east_plan(grid)
        r = replan(grid, start, goal, old.actions, len(old.actions), [(0, 3)])
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.prefix.actions, old.actions)
        self.assertEqual(r.current_pose, goal)
        self.assertEqual(r.remainder.actions, ())
        self.assertEqual(r.remainder.total_cost, 0)
        self.assertEqual(r.total_cost, old.total_cost)
        self.assertEqual(r.full_actions, old.actions)

    def test_obstacle_on_executed_prefix_allowed(self) -> None:
        # 障碍落在“已经走过”的格子上：前缀事实不变，只影响剩余搜索
        grid = make_open_grid()
        start, goal = Pose(0, 0, E), Pose(7, 0, E)
        old = original_east_plan(grid)
        r = replan(grid, start, goal, old.actions, 4, [(1, 0)])
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.current_pose, Pose(4, 0, E))
        # 身后 (1,0) 被堵不影响继续东行
        self.assertEqual(r.remainder.actions, (F, F, F))


class ReplanValidationTest(unittest.TestCase):
    """非法输入给可定位错误，且不留下半更新状态。"""

    def setUp(self) -> None:
        self.grid = make_open_grid()
        self.start, self.goal = Pose(0, 0, E), Pose(7, 0, E)
        self.old = original_east_plan(self.grid)

    def test_old_path_must_be_collision_free(self) -> None:
        # 旧动作序列本身在旧网格上碰撞（第 8 步冲出东边界）→ ReplayError，带步序号
        with self.assertRaisesRegex(ReplayError, "第 7 步"):
            replan(self.grid, self.start, self.goal, (F,) * 8, 2, ())

    def test_executed_actions_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "executed_actions"):
            replan(self.grid, self.start, self.goal, self.old.actions, -1, ())
        with self.assertRaisesRegex(ValueError, "executed_actions.*超过"):
            replan(self.grid, self.start, self.goal, self.old.actions, 8, ())
        with self.assertRaises(TypeError):
            replan(self.grid, self.start, self.goal, self.old.actions, True, ())
        with self.assertRaises(TypeError):
            replan(self.grid, self.start, self.goal, self.old.actions, 2.0, ())

    def test_added_obstacles_validation_locatable(self) -> None:
        before = set(self.grid.obstacles)
        with self.assertRaisesRegex(ValueError, r"added_obstacles\[1\]"):
            replan(self.grid, self.start, self.goal, self.old.actions, 2,
                   [(0, 3), (9, 0)])  # 越界
        with self.assertRaisesRegex(ValueError, r"added_obstacles\[1\].*已是障碍"):
            replan(self.grid, self.start, self.goal, self.old.actions, 2,
                   [(0, 3), (0, 3)])  # 新增列表内部重复
        with self.assertRaises(TypeError):
            replan(self.grid, self.start, self.goal, self.old.actions, 2, "(0,3)")
        # 校验失败后旧网格仍是原状（无半更新）
        self.assertEqual(set(self.grid.obstacles), before)

    def test_wrong_types(self) -> None:
        with self.assertRaises(TypeError):
            replan("not grid", self.start, self.goal, self.old.actions, 2, ())
        with self.assertRaises(TypeError):
            replan(self.grid, (0, 0, E), self.goal, self.old.actions, 2, ())
        with self.assertRaises(TypeError):
            replan(self.grid, self.start, self.goal, "FF", 2, ())

    def test_max_expansions_passthrough(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_expansions"):
            replan(self.grid, self.start, self.goal, self.old.actions, 2, [(4, 0)],
                   max_expansions=-1)
        r = replan(self.grid, self.start, self.goal, self.old.actions, 2, [(4, 0)],
                   max_expansions=0)
        self.assertEqual(r.status, SearchStatus.BUDGET_EXHAUSTED)
        self.assertIsNone(r.remainder)
        # 放开预算后同一输入可达最优
        r2 = replan(self.grid, self.start, self.goal, self.old.actions, 2, [(4, 0)])
        self.assertEqual(r2.status, SearchStatus.OPTIMAL)


class WithAddedObstaclesTest(unittest.TestCase):
    """Grid.with_added_obstacles：纯新增、不可变、失败不留半成品。"""

    def test_returns_new_grid_and_keeps_old(self) -> None:
        grid = Grid(3, 2, obstacles=[(0, 0)])
        g2 = grid.with_added_obstacles([(2, 1)])
        self.assertEqual(grid.obstacles, frozenset({(0, 0)}))
        self.assertEqual(g2.obstacles, frozenset({(0, 0), (2, 1)}))
        self.assertEqual((g2.width, g2.height), (3, 2))

    def test_existing_obstacle_duplicate_rejected(self) -> None:
        grid = Grid(3, 2, obstacles=[(0, 0)])
        with self.assertRaisesRegex(ValueError, r"added_obstacles\[0\].*已是障碍"):
            grid.with_added_obstacles([(0, 0)])

    def test_bad_type(self) -> None:
        grid = Grid(3, 2)
        with self.assertRaises(TypeError):
            grid.with_added_obstacles("(0,0)")
        with self.assertRaisesRegex(ValueError, r"added_obstacles\[0\]"):
            grid.with_added_obstacles([(1, 1, 1)])


class ReplanRandomCrossCheckTest(unittest.TestCase):
    """小规模确定性枚举：剩余段与独立 Dijkstra 在更新后网格上逐场景对照。"""

    def test_random_updates_match_reference(self) -> None:
        rng = random.Random(20260924)
        checked = 0
        no_path_cases = 0
        collide_cases = 0
        for trial in range(200):
            w = rng.randint(3, 9)
            h = rng.randint(2, 6)
            obstacles = [
                (x, y)
                for x in range(w)
                for y in range(h)
                if rng.random() < rng.choice((0.0, 0.15, 0.25))
            ]
            grid = Grid(w, h, obstacles)
            free_poses = [
                Pose(x, y, Heading(hd))
                for x in range(w)
                for y in range(h)
                for hd in range(4)
                if grid.is_pose_free(Pose(x, y, Heading(hd)))
            ]
            if len(free_poses) < 2:
                continue
            start, goal = rng.sample(free_poses, 2)
            first = plan(grid, start, goal)
            if first.status is not SearchStatus.OPTIMAL:
                continue
            old_actions = first.plan.actions
            k = rng.randint(0, len(old_actions))
            # 独立推导当前姿态（不靠被测 replan）
            cur_ref = ref_pose(start)
            for a in old_actions[:k]:
                cur_ref = _ref_apply(cur_ref, a)
            # 随机选 0..3 个新障碍（界内、互不重复、不在原障碍集）
            candidates = [(x, y) for x in range(w) for y in range(h)
                          if (x, y) not in grid.obstacles]
            rng.shuffle(candidates)
            added = tuple(candidates[: rng.randint(0, 3)])
            new_obs = frozenset(grid.obstacles) | frozenset(added)

            cur_cells = {
                (cur_ref[0], cur_ref[1]),
                (cur_ref[0] + (0, 1, 0, -1)[cur_ref[2]],
                 cur_ref[1] + (-1, 0, 1, 0)[cur_ref[2]]),
            }
            if cur_cells & new_obs:
                with self.assertRaises(ValueError, msg=f"trial={trial}"):
                    replan(grid, start, goal, old_actions, k, added)
                collide_cases += 1
                continue

            r = replan(grid, start, goal, old_actions, k, added)
            # 前缀恒等于旧动作前 k 步
            self.assertEqual(r.prefix.actions, old_actions[:k], f"trial={trial}")
            self.assertEqual((r.current_pose.x, r.current_pose.y, int(r.current_pose.heading)),
                             cur_ref, f"trial={trial}")

            goal_cells = {
                (goal.x, goal.y),
                (goal.x + goal.heading.dx, goal.y + goal.heading.dy),
            }
            expected = None if goal_cells & new_obs else dijkstra_reference(
                w, h, new_obs, cur_ref, ref_pose(goal)
            )
            if expected is None:
                self.assertEqual(r.status, SearchStatus.NO_PATH, f"trial={trial}")
                self.assertIsNone(r.remainder, f"trial={trial}")
                no_path_cases += 1
            else:
                self.assertEqual(r.status, SearchStatus.OPTIMAL, f"trial={trial}")
                self.assertEqual(r.remainder.total_cost, expected, f"trial={trial}")
                assert_evidence_valid(
                    self, r.updated_grid, r.current_pose,
                    r.remainder.actions, r.remainder.evidence, goal,
                )
                checked += 1
        self.assertGreaterEqual(checked, 60)
        self.assertGreaterEqual(no_path_cases, 3)
        self.assertGreaterEqual(collide_cases, 3)


_DX = (0, 1, 0, -1)
_DY = (-1, 0, 1, 0)


def _ref_apply(state: tuple[int, int, int], action: Action) -> tuple[int, int, int]:
    """独立推导单个动作后的姿态（仅用于测试期望值，不做扫掠判定）。"""
    x, y, hd = state
    if action is F:
        return (x + _DX[hd], y + _DY[hd], hd)
    if action is B:
        return (x - _DX[hd], y - _DY[hd], hd)
    if action is TL:
        return (x, y, (hd - 1) % 4)
    return (x, y, (hd + 1) % 4)


if __name__ == "__main__":
    unittest.main()
