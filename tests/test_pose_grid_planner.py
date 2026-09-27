"""验收测试：窄道、目标朝向、后退必需、边界、Dijkstra 对照、输入校验、预算语义。"""

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
    heuristic,
    plan,
    replay,
    step_cost,
    successors,
    transition,
)
from dijkstra_reference import dijkstra_reference

N, E, S, W = Heading.NORTH, Heading.EAST, Heading.SOUTH, Heading.WEST


def to_ref(pose: Pose) -> tuple[int, int, int]:
    return (pose.x, pose.y, int(pose.heading))


def assert_plan_valid(testcase: unittest.TestCase, grid: Grid, start: Pose, goal: Pose, p: Plan) -> None:
    """路径自洽性：证据链完整、每步扫掠格全空、代价求和一致、终点等于目标。"""
    testcase.assertEqual(sum(e.cost for e in p.evidence), p.total_cost)
    testcase.assertEqual(tuple(e.action for e in p.evidence), p.actions)
    pose = start
    for e in p.evidence:
        testcase.assertEqual(e.from_pose, pose)
        for cell in e.swept_cells:
            testcase.assertTrue(grid.is_free(cell), f"扫掠格 {cell} 不空")
        # 证据中的扫掠格必须与模型规则独立重算的一致
        _, swept = transition(grid, pose, e.action)
        testcase.assertEqual(tuple(sorted(e.swept_cells)), tuple(sorted(swept)))
        pose = e.to_pose
    testcase.assertEqual(pose, goal)
    # 重放一致性（补充性质，不替代独立参考）
    final_pose, _ = replay(grid, start, p.actions)
    testcase.assertEqual(final_pose, goal)


class NarrowCorridorTest(unittest.TestCase):
    """1 格宽直道：可直行，不可转向（转向扫掠 2x2 必然越界）。"""

    def setUp(self) -> None:
        self.grid = Grid(5, 1)  # 5x1 空走廊

    def test_straight_ahead_ok(self) -> None:
        r = plan(self.grid, Pose(0, 0, E), Pose(3, 0, E))
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.plan.actions, (Action.FORWARD,) * 3)
        self.assertEqual(r.plan.total_cost, 30)
        assert_plan_valid(self, self.grid, Pose(0, 0, E), Pose(3, 0, E), r.plan)

    def test_turn_impossible_in_corridor(self) -> None:
        # 任何转向的 2x2 扫掠都越界：successors 中不应出现转向
        succ = list(successors(self.grid, Pose(1, 0, E)))
        self.assertTrue(all(a in (Action.FORWARD, Action.BACKWARD) for a, *_ in succ))

    def test_wrong_heading_goal_unreachable(self) -> None:
        # 目标姿态 (3,0,WEST) 合法但朝向不可达（走廊里转不了身）
        r = plan(self.grid, Pose(0, 0, E), Pose(3, 0, W))
        self.assertEqual(r.status, SearchStatus.NO_PATH)
        self.assertIsNone(r.plan)


class GoalHeadingTest(unittest.TestCase):
    """目标朝向是状态的一部分：同一格子不同朝向代价不同，且必须精确到达。"""

    def setUp(self) -> None:
        self.grid = Grid(4, 4)
        self.start = Pose(1, 1, E)

    def test_heading_changes_cost(self) -> None:
        r_north = plan(self.grid, self.start, Pose(2, 2, N))
        r_south = plan(self.grid, self.start, Pose(2, 2, S))
        self.assertEqual(r_north.status, SearchStatus.OPTIMAL)
        self.assertEqual(r_south.status, SearchStatus.OPTIMAL)
        self.assertNotEqual(r_north.plan.total_cost, r_south.plan.total_cost)
        # 与独立 Dijkstra 参考一致
        self.assertEqual(
            r_north.plan.total_cost,
            dijkstra_reference(4, 4, frozenset(), to_ref(self.start), (2, 2, int(N))),
        )
        self.assertEqual(
            r_south.plan.total_cost,
            dijkstra_reference(4, 4, frozenset(), to_ref(self.start), (2, 2, int(S))),
        )

    def test_final_heading_exact(self) -> None:
        for h in (N, E, S, W):
            r = plan(self.grid, self.start, Pose(2, 2, h))
            self.assertEqual(r.status, SearchStatus.OPTIMAL)
            self.assertEqual(r.plan.final_pose.heading, h)
            assert_plan_valid(self, self.grid, self.start, Pose(2, 2, h), r.plan)

    def test_same_cell_different_heading_needs_turn(self) -> None:
        # 起终同格、朝向相反：必须付出两次 90 度转向代价，不允许“零代价换向”
        r = plan(self.grid, self.start, Pose(1, 1, W))
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.plan.total_cost, 16)  # 两次 90 度转向
        self.assertEqual(len(r.plan.actions), 2)
        self.assertTrue(all(a in (Action.TURN_LEFT, Action.TURN_RIGHT) for a in r.plan.actions))


class BackwardRequiredTest(unittest.TestCase):
    """死胡同：前方出界、转向扫掠越界，唯一出路是后退。"""

    def setUp(self) -> None:
        self.grid = Grid(4, 1)
        self.start = Pose(2, 0, E)  # 占用 (2,0),(3,0)，前面就是边界

    def test_only_backward_works(self) -> None:
        r = plan(self.grid, self.start, Pose(0, 0, E))
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.plan.actions, (Action.BACKWARD, Action.BACKWARD))
        self.assertEqual(r.plan.total_cost, 28)
        assert_plan_valid(self, self.grid, self.start, Pose(0, 0, E), r.plan)

    def test_backward_primitive_cost_and_direction(self) -> None:
        # 后退原语：沿 heading 反方向移动一格，代价 14
        grid = Grid.from_rows([
            "....",
            "..#.",
        ])
        succ = {a: (nxt, cost) for a, nxt, _swept, cost in successors(grid, Pose(1, 0, E))}
        self.assertIn(Action.BACKWARD, succ)
        self.assertEqual(succ[Action.BACKWARD], (Pose(0, 0, E), 14))


class BoundaryTest(unittest.TestCase):
    """边界：占用格/扫掠格越界即非法；起终姿态非法要在调用处报可定位错误。"""

    def test_pose_sticking_out_rejected(self) -> None:
        grid = Grid(3, 3)
        with self.assertRaisesRegex(ValueError, "start.*越界"):
            plan(grid, Pose(2, 0, E), Pose(0, 0, E))  # 占用 (3,0) 越界
        with self.assertRaisesRegex(ValueError, "goal.*越界"):
            plan(grid, Pose(0, 0, E), Pose(0, 0, N))  # 占用 (0,-1) 越界

    def test_turn_sweep_respects_boundary(self) -> None:
        grid = Grid(2, 2)
        pose = Pose(0, 0, E)  # 占用 (0,0),(1,0)
        # 左转 → NORTH：扫掠含 y=-1，越界，必须被拒绝
        self.assertIsNone(transition(grid, pose, Action.TURN_LEFT))
        # 右转 → SOUTH：扫掠 {(0,0),(1,0),(0,1),(1,1)} 全空，允许
        result = transition(grid, pose, Action.TURN_RIGHT)
        self.assertIsNotNone(result)
        nxt, swept = result
        self.assertEqual(nxt, Pose(0, 0, S))
        self.assertEqual(tuple(sorted(swept)), ((0, 0), (0, 1), (1, 0), (1, 1)))

    def test_translation_sweep_is_union(self) -> None:
        # 前进扫掠 = 起终占用并集（3 格）；障碍挡在“新头部”即失败
        grid = Grid(4, 1, obstacles=[(3, 0)])
        self.assertIsNone(transition(grid, Pose(1, 0, E), Action.FORWARD))
        # 障碍在身后不影响前进
        grid2 = Grid(4, 1, obstacles=[(0, 0)])
        self.assertIsNotNone(transition(grid2, Pose(1, 0, E), Action.FORWARD))

    def test_max_grid_side_enforced(self) -> None:
        Grid(20, 20)  # 上限本身合法
        with self.assertRaisesRegex(ValueError, "width"):
            Grid(21, 1)
        with self.assertRaisesRegex(ValueError, "height"):
            Grid(1, 21)


class InputValidationTest(unittest.TestCase):
    """错误语义：NaN/Inf、bool 整数、错误类型都必须给可定位错误，不静默纠正。"""

    def test_nan_and_inf_rejected(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError, msg=f"应拒绝 {bad}"):
                Grid(bad, 3)
            with self.assertRaises(ValueError, msg=f"应拒绝 {bad}"):
                Pose(bad, 0, E)

    def test_bool_rejected_as_int(self) -> None:
        with self.assertRaises(TypeError):
            Grid(True, 3)
        with self.assertRaises(TypeError):
            Pose(0, True, E)
        with self.assertRaises(TypeError):
            Grid(2, 2, obstacles=[(True, 0)])
        with self.assertRaises(TypeError):
            plan(Grid(2, 2), Pose(0, 0, E), Pose(0, 0, W), max_expansions=False)

    def test_float_not_silently_truncated(self) -> None:
        with self.assertRaises(TypeError):
            Grid(2.9, 3)
        with self.assertRaises(TypeError):
            Pose(1.5, 0, E)

    def test_wrong_types(self) -> None:
        with self.assertRaises(TypeError):
            Pose(0, 0, 1)  # heading 必须是 Heading 枚举
        with self.assertRaises(TypeError):
            plan("not a grid", Pose(0, 0, E), Pose(0, 0, W))
        with self.assertRaises(TypeError):
            plan(Grid(2, 2), (0, 0, E), Pose(0, 0, W))
        with self.assertRaises(TypeError):
            Grid(3, 3, obstacles="(0,0)")

    def test_obstacle_errors_locatable(self) -> None:
        with self.assertRaisesRegex(ValueError, r"obstacles\[1\]"):
            Grid(3, 3, obstacles=[(0, 0), (9, 9)])  # 越界，报索引
        with self.assertRaisesRegex(ValueError, r"obstacles\[1\].*重复"):
            Grid(3, 3, obstacles=[(0, 0), (0, 0)])  # 重复不静默去重
        with self.assertRaisesRegex(ValueError, r"obstacles\[0\]"):
            Grid(3, 3, obstacles=[(0, 0, 0)])  # 长度不为 2

    def test_from_rows_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "rows\\[1\\]"):
            Grid.from_rows(["..", "."])
        with self.assertRaisesRegex(ValueError, "rows\\[0\\]\\[1\\]"):
            Grid.from_rows([".x"])
        with self.assertRaises(ValueError):
            Grid.from_rows([])
        g = Grid.from_rows(["#.", ".#"])
        self.assertEqual(g.obstacles, frozenset({(0, 0), (1, 1)}))

    def test_max_expansions_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_expansions"):
            plan(Grid(2, 2), Pose(0, 0, E), Pose(0, 0, W), max_expansions=-1)
        with self.assertRaises(TypeError):
            plan(Grid(2, 2), Pose(0, 0, E), Pose(0, 0, W), max_expansions=1.5)

    def test_replay_error_locatable(self) -> None:
        grid = Grid(4, 1)
        with self.assertRaisesRegex(ReplayError, "第 1 步"):
            replay(grid, Pose(0, 0, E), (Action.FORWARD, Action.TURN_LEFT))
        with self.assertRaisesRegex(ReplayError, r"actions\[0\]"):
            replay(grid, Pose(0, 0, E), ("FORWARD",))


class BudgetSemanticsTest(unittest.TestCase):
    """预算耗尽与证明无解必须分开。"""

    def test_budget_exhausted_distinct_from_no_path(self) -> None:
        grid = Grid(4, 4)
        start, goal = Pose(0, 0, E), Pose(3, 3, W)
        r0 = plan(grid, start, goal, max_expansions=0)
        self.assertEqual(r0.status, SearchStatus.BUDGET_EXHAUSTED)
        self.assertIsNone(r0.plan)
        r_full = plan(grid, start, goal)
        self.assertEqual(r_full.status, SearchStatus.OPTIMAL)

    def test_no_path_is_proven(self) -> None:
        # 1 格宽走廊里目标朝向不可达：开集耗尽 → NO_PATH（而非预算耗尽）
        corridor = Grid(5, 1)
        r = plan(corridor, Pose(0, 0, E), Pose(3, 0, W))
        self.assertEqual(r.status, SearchStatus.NO_PATH)
        self.assertGreater(r.expansions, 0)

    def test_start_equals_goal_zero_cost(self) -> None:
        r = plan(Grid(3, 3), Pose(1, 1, E), Pose(1, 1, E))
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.plan.total_cost, 0)
        self.assertEqual(r.plan.actions, ())


class HeuristicTest(unittest.TestCase):
    """启发式只能是 max(曼哈顿*10, 最少转向*8)，且可采纳。"""

    def test_exact_formula(self) -> None:
        goal = Pose(4, 4, S)
        # 曼哈顿 4+4=8 → 80；转向 |E-S|=1 → 8；max=80
        self.assertEqual(heuristic(Pose(0, 0, E), goal), 80)
        # 曼哈顿 0；转向 E→S 一次 → 8
        self.assertEqual(heuristic(Pose(4, 4, E), goal), 8)
        # 转向 E→W 两次 → 16
        self.assertEqual(heuristic(Pose(4, 4, W), Pose(4, 4, E)), 16)
        self.assertEqual(heuristic(goal, goal), 0)

    def test_admissible_against_dijkstra(self) -> None:
        rng = random.Random(20260922)
        grid = Grid(6, 6, obstacles=[(2, 2), (3, 2), (2, 3)])
        obs = grid.obstacles
        for _ in range(50):
            poses = []
            while len(poses) < 2:
                p = Pose(rng.randrange(6), rng.randrange(6), Heading(rng.randrange(4)))
                if grid.is_pose_free(p):
                    poses.append(p)
            start, goal = poses
            opt = dijkstra_reference(6, 6, obs, to_ref(start), to_ref(goal))
            if opt is not None:
                self.assertLessEqual(heuristic(start, goal), opt)


class DijkstraCrossCheckTest(unittest.TestCase):
    """A* 与独立全状态 Dijkstra 参考在确定性随机场景下对照。"""

    def test_random_grids_match_reference(self) -> None:
        rng = random.Random(12345)
        cases = 0
        for trial in range(40):
            w = rng.randint(2, 8)
            h = rng.randint(2, 8)
            density = rng.choice([0.0, 0.15, 0.3])
            obstacles = [
                (x, y)
                for x in range(w)
                for y in range(h)
                if rng.random() < density
            ]
            try:
                grid = Grid(w, h, obstacles=obstacles)
            except ValueError:
                continue
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
            expected = dijkstra_reference(w, h, grid.obstacles, to_ref(start), to_ref(goal))
            result = plan(grid, start, goal)
            if expected is None:
                self.assertEqual(
                    result.status, SearchStatus.NO_PATH,
                    f"trial={trial} 参考无解但 A* 给出 {result.status}",
                )
            else:
                self.assertEqual(
                    result.status, SearchStatus.OPTIMAL,
                    f"trial={trial} 参考有解({expected})但 A* 给出 {result.status}",
                )
                self.assertEqual(result.plan.total_cost, expected, f"trial={trial} 代价不一致")
                assert_plan_valid(self, grid, start, goal, result.plan)
            cases += 1
        self.assertGreaterEqual(cases, 30)  # 确保对照样本量

    def test_max_size_grid_matches_reference(self) -> None:
        # 上限规模 20x20 也对照一次（题面接口边界）
        rng = random.Random(777)
        obstacles = [(x, y) for x in range(20) for y in range(20) if rng.random() < 0.2]
        grid = Grid(20, 20, obstacles=obstacles)
        free_poses = [
            Pose(x, y, Heading(hd))
            for x in range(20)
            for y in range(20)
            for hd in range(4)
            if grid.is_pose_free(Pose(x, y, Heading(hd)))
        ]
        start, goal = rng.sample(free_poses, 2)
        expected = dijkstra_reference(20, 20, grid.obstacles, to_ref(start), to_ref(goal))
        result = plan(grid, start, goal)
        if expected is None:
            self.assertEqual(result.status, SearchStatus.NO_PATH)
        else:
            self.assertEqual(result.status, SearchStatus.OPTIMAL)
            self.assertEqual(result.plan.total_cost, expected)
            assert_plan_valid(self, grid, start, goal, result.plan)


class TerrainCostTest(unittest.TestCase):
    """地面通行代价：扫掠多格按集合计费、绕行选择、重放一致。"""

    def test_step_cost_sums_all_swept_cells(self) -> None:
        # 前进扫掠 3 格：起点占用 2 格 + 新头部 1 格，每格额外代价都计入
        grid = Grid(4, 1, terrain={(1, 0): 2, (2, 0): 3, (3, 0): 5})
        cost = step_cost(grid, Action.FORWARD, ((1, 0), (2, 0), (3, 0)))
        self.assertEqual(cost, 10 + 2 + 3 + 5)

    def test_non_destination_cells_are_charged(self) -> None:
        # 终点格免费、扫掠中间格昂贵：代价仍必须包含中间格
        grid = Grid(4, 1, terrain={(2, 0): 7})
        result = transition(grid, Pose(1, 0, E), Action.FORWARD)
        self.assertIsNotNone(result)
        _nxt, swept = result
        self.assertEqual(step_cost(grid, Action.FORWARD, swept), 10 + 7)
        # 转向扫掠 2x2 共 4 格，非终点格同样计费
        grid2 = Grid(2, 2, terrain={(1, 1): 4})
        result2 = transition(grid2, Pose(0, 0, E), Action.TURN_RIGHT)
        self.assertIsNotNone(result2)
        _nxt2, swept2 = result2
        self.assertEqual(tuple(sorted(swept2)), ((0, 0), (0, 1), (1, 0), (1, 1)))
        self.assertEqual(step_cost(grid2, Action.TURN_RIGHT, swept2), 8 + 4)

    def test_planner_prefers_cheaper_detour(self) -> None:
        # 直行走廊铺满高代价地面；上方绕行距离更长但更便宜
        terrain = {(x, 1): 50 for x in range(1, 6)}  # 中间一行极贵
        grid = Grid(7, 3, terrain=terrain)
        start, goal = Pose(0, 1, E), Pose(5, 1, E)
        r = plan(grid, start, goal)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        # 最优解应绕到 y=0 或 y=2，而不是硬闯 50 代价带
        self.assertTrue(any(e.to_pose.y != 1 for e in r.plan.evidence))
        # 与独立参考一致
        self.assertEqual(
            r.plan.total_cost,
            dijkstra_reference(7, 3, frozenset(), to_ref(start), to_ref(goal), terrain),
        )
        assert_plan_valid(self, grid, start, goal, r.plan)

    def test_high_cost_terrain_remains_passable(self) -> None:
        # 只有一条路且必须穿过高代价地面：仍应给出路径，只是更贵
        grid = Grid(5, 1, terrain={(1, 0): 25})
        r = plan(grid, Pose(0, 0, E), Pose(3, 0, E))
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        # 前进 3 步，第 1、2 步的扫掠并集都覆盖 (1,0)，各计一次 25
        self.assertEqual(r.plan.total_cost, 30 + 25 * 2)

    def test_replay_matches_plan_with_terrain(self) -> None:
        grid = Grid(6, 3, obstacles=[(2, 1)], terrain={(1, 0): 3, (3, 2): 6, (0, 2): 1})
        start, goal = Pose(0, 0, E), Pose(4, 1, S)
        r = plan(grid, start, goal)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        final_pose, evidence = replay(grid, start, r.plan.actions)
        self.assertEqual(final_pose, goal)
        self.assertEqual(sum(e.cost for e in evidence), r.plan.total_cost)
        for pe, re_ in zip(r.plan.evidence, evidence):
            self.assertEqual(pe.cost, re_.cost)

    def test_terrain_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "terrain"):
            Grid(3, 3, terrain={(1, 1): -1})  # 负代价
        with self.assertRaisesRegex(ValueError, "障碍"):
            Grid(3, 3, obstacles=[(1, 1)], terrain={(1, 1): 2})  # 障碍格带代价
        with self.assertRaises(ValueError):
            Grid(3, 3, terrain={(9, 9): 1})  # 越界
        with self.assertRaises(TypeError):
            Grid(3, 3, terrain={(1, 1): True})  # bool 拒绝
        with self.assertRaises(TypeError):
            Grid(3, 3, terrain=[((1, 1), 2)])  # 必须是映射
        with self.assertRaises(ValueError):
            Grid(3, 3, terrain={(1, 1): float("nan")})

    def test_from_rows_terrain_rows(self) -> None:
        grid = Grid.from_rows(["...", "..."], terrain_rows=[".2.", "1.3"])
        self.assertEqual(grid.terrain_extra((1, 0)), 2)
        self.assertEqual(grid.terrain_extra((0, 1)), 1)
        self.assertEqual(grid.terrain_extra((2, 1)), 3)
        self.assertEqual(grid.terrain_extra((0, 0)), 0)
        with self.assertRaisesRegex(ValueError, "terrain_rows"):
            Grid.from_rows([".."], terrain_rows=[".", "."])  # 行数不一致
        with self.assertRaisesRegex(ValueError, "障碍格"):
            Grid.from_rows(["#."], terrain_rows=["1."])  # 障碍格带代价
        with self.assertRaisesRegex(ValueError, "terrain_rows"):
            Grid.from_rows([".."], terrain_rows=[".x"])  # 非法字符

    def test_random_terrain_matches_reference(self) -> None:
        rng = random.Random(20260927)
        for trial in range(30):
            w = rng.randint(2, 7)
            h = rng.randint(2, 7)
            obstacles = {(x, y) for x in range(w) for y in range(h) if rng.random() < 0.2}
            terrain = {
                (x, y): rng.randint(0, 9)
                for x in range(w)
                for y in range(h)
                if (x, y) not in obstacles and rng.random() < 0.4
            }
            grid = Grid(w, h, obstacles=obstacles, terrain=terrain)
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
            expected = dijkstra_reference(w, h, grid.obstacles, to_ref(start), to_ref(goal), terrain)
            result = plan(grid, start, goal)
            if expected is None:
                self.assertEqual(result.status, SearchStatus.NO_PATH, f"trial={trial}")
            else:
                self.assertEqual(result.status, SearchStatus.OPTIMAL, f"trial={trial}")
                self.assertEqual(result.plan.total_cost, expected, f"trial={trial}")
                assert_plan_valid(self, grid, start, goal, result.plan)


if __name__ == "__main__":
    unittest.main()
