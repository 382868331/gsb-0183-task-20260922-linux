"""离散时间扩展验收测试：窄路临时占用、倒车让行、终点后被封、转向扫掠碰禁占、
时域语义（NO_PATH_WITHIN_HORIZON）、独立时序 Dijkstra 对照、重放逐步占用证据。"""

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
    TimedState,
    Timetable,
    heuristic,
    plan,
    plan_timed,
    replay_timed,
    timed_successors,
    timed_transition,
)
from dijkstra_reference import timed_dijkstra_reference

N, E, S, W = Heading.NORTH, Heading.EAST, Heading.SOUTH, Heading.WEST
F, B, TL, TR, WT = (
    Action.FORWARD,
    Action.BACKWARD,
    Action.TURN_LEFT,
    Action.TURN_RIGHT,
    Action.WAIT,
)


def to_ref(pose: Pose) -> tuple[int, int, int]:
    return (pose.x, pose.y, int(pose.heading))


def ref_call(grid: Grid, tt: Timetable, horizon: int, start: Pose, goal: Pose, wait_cost: int = 1):
    blocked = {t: set(tt.blocked_at(t)) for t in range(0, tt.horizon + 1)}
    return timed_dijkstra_reference(
        grid.width, grid.height, grid.obstacles, blocked, horizon,
        to_ref(start), to_ref(goal), wait_cost,
    )


def assert_timed_plan_valid(
    testcase: unittest.TestCase,
    grid: Grid,
    tt: Timetable,
    start: Pose,
    goal: Pose,
    p: Plan,
    wait_cost: int = 1,
) -> None:
    """时序路径自洽性：证据链完整、时刻递增、每步扫掠格避开静态障碍与
    blocked[t] ∪ blocked[t+1]、代价求和一致、终点等于目标。"""
    testcase.assertEqual(sum(e.cost for e in p.evidence), p.total_cost)
    testcase.assertEqual(tuple(e.action for e in p.evidence), p.actions)
    pose = start
    for i, e in enumerate(p.evidence):
        testcase.assertEqual(e.time, i, "每步耗时 1，起始时刻应等于步序号")
        testcase.assertEqual(e.from_pose, pose)
        for cell in e.swept_cells:
            testcase.assertTrue(grid.is_free(cell), f"第 {i} 步扫掠格 {cell} 不空")
            testcase.assertFalse(tt.is_blocked(cell, i), f"第 {i} 步扫掠格 {cell} 在 t={i} 被禁占")
            testcase.assertFalse(tt.is_blocked(cell, i + 1), f"第 {i} 步扫掠格 {cell} 在 t={i + 1} 被禁占")
        if e.action is WT:
            testcase.assertEqual(e.to_pose, e.from_pose, "WAIT 不得改变姿态")
            testcase.assertEqual(e.cost, wait_cost)
            # 等待必须检查车体全部格
            testcase.assertEqual(set(e.swept_cells), set(grid.pose_cells(pose)))
        # 证据中的扫掠格必须与模型规则独立重算的一致
        result = timed_transition(grid, tt, TimedState.from_pose(pose, i), e.action)
        testcase.assertIsNotNone(result)
        _, swept = result
        testcase.assertEqual(tuple(sorted(e.swept_cells)), tuple(sorted(swept)))
        pose = e.to_pose
    testcase.assertEqual(pose, goal)
    # 重放一致性（补充性质，不替代独立参考）
    final_state, ev2 = replay_timed(grid, tt, start, p.actions, wait_cost=wait_cost)
    testcase.assertEqual(final_state.pose, goal)
    testcase.assertEqual(final_state.time, len(p.actions))
    testcase.assertEqual(sum(e.cost for e in ev2), p.total_cost)


class TemporaryBlockageTest(unittest.TestCase):
    """窄路临时占用：移动障碍穿过 1 格宽走廊，机器人必须原地等待让其先通过。"""

    def setUp(self) -> None:
        self.grid = Grid(5, 1)  # 5x1 走廊
        self.start = Pose(0, 0, E)  # 占用 (0,0),(1,0)
        self.goal = Pose(3, 0, E)  # 占用 (3,0),(4,0)
        # 障碍在 t=1,2 占据走廊中点 (2,0)，其余时刻畅通
        self.tt = Timetable({1: [(2, 0)], 2: [(2, 0)]})

    def test_must_wait_for_obstacle_to_pass(self) -> None:
        r = plan_timed(self.grid, self.tt, self.start, self.goal, horizon=10)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        # 直接前进会在 t=0/1/2 撞上 (2,0) 的禁占：只能先等 3 步再走 3 步
        self.assertEqual(r.plan.actions, (WT, WT, WT, F, F, F))
        self.assertEqual(r.plan.total_cost, 3 * 1 + 3 * 10)
        self.assertEqual(r.arrival_time, 6)
        assert_timed_plan_valid(self, self.grid, self.tt, self.start, self.goal, r.plan)

    def test_matches_independent_reference(self) -> None:
        r = plan_timed(self.grid, self.tt, self.start, self.goal, horizon=10)
        self.assertEqual(
            (r.plan.total_cost, r.arrival_time),
            ref_call(self.grid, self.tt, 10, self.start, self.goal),
        )

    def test_without_timetable_no_wait_needed(self) -> None:
        # 同一几何、空时刻表：不需要等待，代价即静态最优
        r = plan_timed(self.grid, Timetable({}), self.start, self.goal, horizon=10)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.plan.actions, (F, F, F))
        self.assertEqual(r.plan.total_cost, 30)


class ReverseToYieldTest(unittest.TestCase):
    """只有倒车能让行：前进或原地等待都会被经过的障碍困死，唯一活路是先倒车。"""

    def setUp(self) -> None:
        self.grid = Grid(6, 1)  # 6x1 走廊，无法转向（2x2 扫掠必越界）
        self.start = Pose(2, 0, E)  # 占用 (2,0),(3,0)
        self.goal = Pose(4, 0, E)  # 占用 (4,0),(5,0)
        # 宽障碍在 t=2,3 占据走廊东段 (3,0),(4,0),(5,0)
        self.tt = Timetable({2: [(3, 0), (4, 0), (5, 0)], 3: [(3, 0), (4, 0), (5, 0)]})

    def test_forward_or_wait_is_trapped(self) -> None:
        s0 = TimedState.from_pose(self.start, 0)
        # t=0 前进：合法，但 t=1 时车身 (3,0),(4,0) 被禁占包围，进退等待全死
        r = timed_transition(self.grid, self.tt, s0, F)
        self.assertIsNotNone(r)
        s1, _ = r
        self.assertEqual([a for a, *_ in timed_successors(self.grid, self.tt, s1)], [])
        # t=0 等待：合法，但 t=1 同样无后继
        r = timed_transition(self.grid, self.tt, s0, WT)
        self.assertIsNotNone(r)
        s1w, _ = r
        self.assertEqual([a for a, *_ in timed_successors(self.grid, self.tt, s1w)], [])

    def test_only_reverse_survives(self) -> None:
        r = plan_timed(self.grid, self.tt, self.start, self.goal, horizon=10)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        # 唯一活路：先倒车退出障碍覆盖区，等其通过后前进
        self.assertEqual(r.plan.actions, (B, WT, WT, WT, F, F, F))
        self.assertEqual(r.plan.total_cost, 14 + 3 * 1 + 3 * 10)
        self.assertEqual(r.arrival_time, 7)
        assert_timed_plan_valid(self, self.grid, self.tt, self.start, self.goal, r.plan)
        self.assertEqual(
            (r.plan.total_cost, r.arrival_time),
            ref_call(self.grid, self.tt, 10, self.start, self.goal),
        )


class GoalSealedLaterTest(unittest.TestCase):
    """终点稍后被封：到达后停止，不要求永久占住终点；
    赶不上封死前到达则报时域内无路（而非静态无解）。"""

    def setUp(self) -> None:
        self.grid = Grid(4, 1)
        self.start = Pose(0, 0, E)  # 占用 (0,0),(1,0)
        self.goal = Pose(2, 0, E)  # 占用 (2,0),(3,0)

    def test_arrive_before_seal(self) -> None:
        # 终点两格在 t>=3 被封；机器人 t=2 到达即停止，之后封死不影响
        tt = Timetable({3: [(2, 0), (3, 0)], 4: [(2, 0), (3, 0)], 5: [(2, 0), (3, 0)]})
        r = plan_timed(self.grid, tt, self.start, self.goal, horizon=5)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.plan.actions, (F, F))
        self.assertEqual(r.plan.total_cost, 20)
        self.assertEqual(r.arrival_time, 2)
        assert_timed_plan_valid(self, self.grid, tt, self.start, self.goal, r.plan)

    def test_sealed_before_arrival_is_horizon_no_path(self) -> None:
        # 终点从 t=1 起就被封：任何到达时刻都撞禁占 → 时域内无路
        tt = Timetable({t: [(2, 0), (3, 0)] for t in range(1, 6)})
        r = plan_timed(self.grid, tt, self.start, self.goal, horizon=5)
        self.assertEqual(r.status, SearchStatus.NO_PATH_WITHIN_HORIZON)
        self.assertIsNone(r.plan)
        # 但同一几何的静态规划有路——两个结论必须区分，不能混报
        r_static = plan(self.grid, self.start, self.goal)
        self.assertEqual(r_static.status, SearchStatus.OPTIMAL)

    def test_horizon_too_short_is_not_static_no_path(self) -> None:
        # 时域太短（需要 2 步只给 1 步）：报时域内无路，不是静态无解
        r = plan_timed(self.grid, Timetable({}), self.start, self.goal, horizon=1)
        self.assertEqual(r.status, SearchStatus.NO_PATH_WITHIN_HORIZON)
        self.assertNotEqual(r.status, SearchStatus.NO_PATH)


class TurnSweepBlockedTest(unittest.TestCase):
    """转向扫掠碰禁占：2x2 扫掠格在特定时刻被禁占，必须等障碍离开再转。"""

    def setUp(self) -> None:
        self.grid = Grid(3, 3)
        self.start = Pose(1, 1, N)  # 占用 (1,1),(1,0)
        self.goal = Pose(1, 1, E)  # 占用 (1,1),(2,1)
        # 右转扫掠 {(1,1),(1,0),(2,1),(2,0)} 中的 (2,0) 在 t=0,1 被禁占
        self.tt = Timetable({0: [(2, 0)], 1: [(2, 0)]})

    def test_turn_blocked_at_t0_and_t1(self) -> None:
        for t in (0, 1):
            s = TimedState.from_pose(self.start, t)
            self.assertIsNone(timed_transition(self.grid, self.tt, s, TR))
        # t=2 起 (2,0) 解禁，可以右转
        s2 = TimedState.from_pose(self.start, 2)
        self.assertIsNotNone(timed_transition(self.grid, self.tt, s2, TR))

    def test_plan_waits_then_turns(self) -> None:
        r = plan_timed(self.grid, self.tt, self.start, self.goal, horizon=6)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.plan.actions, (WT, WT, TR))
        self.assertEqual(r.plan.total_cost, 2 * 1 + 8)
        self.assertEqual(r.arrival_time, 3)
        assert_timed_plan_valid(self, self.grid, self.tt, self.start, self.goal, r.plan)
        self.assertEqual(
            (r.plan.total_cost, r.arrival_time),
            ref_call(self.grid, self.tt, 6, self.start, self.goal),
        )


class TimedStateSemanticTest(unittest.TestCase):
    """状态 = (位置, 朝向, 时刻)：同姿态不同时刻不合并；等待也检查车体全部格。"""

    def test_same_pose_different_time_not_merged(self) -> None:
        # 若把 (姿态,时刻) 错误合并，等待会被当成原地自环而无法纠错；
        # 这里直接验证两个时刻的同姿态是不同状态、且 best_g 各自独立
        s1 = TimedState(0, 0, E, 1)
        s2 = TimedState(0, 0, E, 2)
        self.assertNotEqual(s1, s2)
        self.assertNotEqual(hash(s1), hash(s2))
        self.assertEqual(s1.pose, s2.pose)

    def test_wait_checks_both_body_cells(self) -> None:
        grid = Grid(4, 1)
        # 车身后格 (0,0) 在 t=0→1 被禁占：等待必须失败（只查头格会漏判）
        tt = Timetable({1: [(0, 0)]})
        s0 = TimedState.from_pose(Pose(0, 0, E), 0)  # 占用 (0,0),(1,0)
        self.assertIsNone(timed_transition(grid, tt, s0, WT))
        # 头格 (1,0) 被禁占同样失败
        tt2 = Timetable({1: [(1, 0)]})
        self.assertIsNone(timed_transition(grid, tt2, s0, WT))
        # 禁占格不在车身上：等待合法
        tt3 = Timetable({1: [(3, 0)]})
        self.assertIsNotNone(timed_transition(grid, tt3, s0, WT))

    def test_sweep_checks_both_endpoint_times(self) -> None:
        grid = Grid(4, 1)
        pose = Pose(0, 0, E)
        # 目标格 (2,0) 只在起始时刻 t=1 被禁占：仍然不许通过
        tt_start = Timetable({1: [(2, 0)]})
        self.assertIsNone(timed_transition(grid, tt_start, TimedState.from_pose(pose, 1), F))
        # 只在到达时刻 t=2 被禁占：同样不许通过
        tt_end = Timetable({2: [(2, 0)]})
        self.assertIsNone(timed_transition(grid, tt_end, TimedState.from_pose(pose, 1), F))
        # t=3 才禁占：t=1→2 的动作不受影响
        tt_later = Timetable({3: [(2, 0)]})
        self.assertIsNotNone(timed_transition(grid, tt_later, TimedState.from_pose(pose, 1), F))

    def test_start_blocked_at_t0_rejected(self) -> None:
        grid = Grid(4, 1)
        tt = Timetable({0: [(1, 0)]})
        with self.assertRaisesRegex(ValueError, "时刻 0"):
            plan_timed(grid, tt, Pose(0, 0, E), Pose(2, 0, E), horizon=4)


class TimedInputValidationTest(unittest.TestCase):
    """时序接口的错误语义：非法输入给可定位错误，不静默纠正。"""

    def test_timetable_validation(self) -> None:
        with self.assertRaises(TypeError):
            Timetable("not a mapping")
        with self.assertRaises(TypeError):
            Timetable({True: [(0, 0)]})  # bool 时刻键
        with self.assertRaisesRegex(ValueError, ">= 0"):
            Timetable({-1: [(0, 0)]})
        with self.assertRaisesRegex(ValueError, r"blocked_by_time\[0\]\[1\]"):
            Timetable({0: [(0, 0), (0, 0)]})  # 同时刻重复不静默去重
        with self.assertRaises(TypeError):
            Timetable({0: "(1,2)"})
        with self.assertRaises(TypeError):
            Timetable({0: [(1.5, 0)]})

    def test_plan_timed_parameter_validation(self) -> None:
        grid = Grid(4, 1)
        tt = Timetable({})
        s, g = Pose(0, 0, E), Pose(2, 0, E)
        with self.assertRaises(TypeError):
            plan_timed(grid, tt, s, g, horizon=True)
        with self.assertRaisesRegex(ValueError, "horizon"):
            plan_timed(grid, tt, s, g, horizon=-1)
        with self.assertRaisesRegex(ValueError, "wait_cost"):
            plan_timed(grid, tt, s, g, horizon=4, wait_cost=0)
        with self.assertRaises(TypeError):
            plan_timed(grid, tt, s, g, horizon=4, wait_cost=1.5)
        with self.assertRaises(TypeError):
            plan_timed("not a grid", tt, s, g, horizon=4)
        with self.assertRaises(TypeError):
            plan_timed(grid, "not a timetable", s, g, horizon=4)
        with self.assertRaisesRegex(ValueError, "max_expansions"):
            plan_timed(grid, tt, s, g, horizon=4, max_expansions=-1)

    def test_timetable_cell_out_of_bounds_rejected(self) -> None:
        grid = Grid(4, 1)
        tt = Timetable({2: [(9, 0)]})
        with self.assertRaisesRegex(ValueError, "时刻 2.*越界"):
            plan_timed(grid, tt, Pose(0, 0, E), Pose(2, 0, E), horizon=4)

    def test_timetable_time_beyond_horizon_rejected(self) -> None:
        # blocked[horizon] 是最后一步所必需的，合法（此处 t=2 即到达，不受其影响）；
        # blocked[horizon+1] 永不生效，属于笔误，报错而不静默忽略。
        grid = Grid(4, 1)
        s, g = Pose(0, 0, E), Pose(2, 0, E)
        r = plan_timed(grid, Timetable({4: [(0, 0)]}), s, g, horizon=4)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        with self.assertRaisesRegex(ValueError, "horizon"):
            plan_timed(grid, Timetable({5: [(0, 0)]}), s, g, horizon=4)

    def test_replay_timed_error_locatable(self) -> None:
        grid = Grid(4, 1)
        tt = Timetable({1: [(2, 0)]})  # t=0→1 前进会撞 (2,0) 在 t=1 的禁占
        with self.assertRaisesRegex(ReplayError, "第 0 步.*时刻 0"):
            replay_timed(grid, tt, Pose(0, 0, E), (F,))
        with self.assertRaisesRegex(ReplayError, r"actions\[0\]"):
            replay_timed(grid, tt, Pose(0, 0, E), ("WAIT",))
        with self.assertRaisesRegex(ValueError, "wait_cost"):
            replay_timed(grid, tt, Pose(0, 0, E), (WT,), wait_cost=0)

    def test_timed_state_validation(self) -> None:
        with self.assertRaises(TypeError):
            TimedState(0, True, E, 0)
        with self.assertRaises(TypeError):
            TimedState(0, 0, 1, 0)
        with self.assertRaisesRegex(ValueError, ">= 0"):
            TimedState(0, 0, E, -1)


class TimedBudgetTest(unittest.TestCase):
    """预算耗尽、时域内无路、找到解三者严格区分。"""

    def test_budget_exhausted_distinct(self) -> None:
        grid = Grid(4, 4)
        tt = Timetable({})
        start, goal = Pose(0, 0, E), Pose(3, 3, W)
        r0 = plan_timed(grid, tt, start, goal, horizon=20, max_expansions=0)
        self.assertEqual(r0.status, SearchStatus.BUDGET_EXHAUSTED)
        r_full = plan_timed(grid, tt, start, goal, horizon=20)
        self.assertEqual(r_full.status, SearchStatus.OPTIMAL)

    def test_start_equals_goal_zero_cost(self) -> None:
        r = plan_timed(Grid(3, 3), Timetable({}), Pose(1, 1, E), Pose(1, 1, E), horizon=5)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.plan.total_cost, 0)
        self.assertEqual(r.arrival_time, 0)

    def test_small_budget_that_still_pops_goal_returns_proven_optimal(self) -> None:
        # 4x1 走廊最优 2 步即达。预算恰好够把目标弹出（3 次扩展）时，
        # 目标总代价已由一致性启发式证明最优：返回 OPTIMAL，代价与全量搜索一致；
        # 预算更小（0..2）则严格返回 BUDGET_EXHAUSTED。
        grid = Grid(4, 1)
        s, g = Pose(0, 0, E), Pose(2, 0, E)
        full = plan_timed(grid, Timetable({}), s, g, horizon=8)
        self.assertEqual(full.status, SearchStatus.OPTIMAL)
        for budget in range(0, 3):
            self.assertEqual(
                plan_timed(grid, Timetable({}), s, g, horizon=8, max_expansions=budget).status,
                SearchStatus.BUDGET_EXHAUSTED,
            )
        r = plan_timed(grid, Timetable({}), s, g, horizon=8, max_expansions=3)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.plan.total_cost, full.plan.total_cost)


class TimedHeuristicTest(unittest.TestCase):
    """时序情形下空间启发式仍可采纳：对随机小地图与独立参考逐点核对。"""

    def test_admissible_against_timed_reference(self) -> None:
        rng = random.Random(20260925)
        grid = Grid(5, 5, obstacles=[(2, 2)])
        checked = 0
        for _ in range(60):
            poses = []
            while len(poses) < 2:
                p = Pose(rng.randrange(5), rng.randrange(5), Heading(rng.randrange(4)))
                if grid.is_pose_free(p):
                    poses.append(p)
            start, goal = poses
            horizon = rng.randint(4, 10)
            blocked = {}
            for t in range(horizon + 1):
                cell = (rng.randrange(5), rng.randrange(5))
                if cell not in grid.obstacles and rng.random() < 0.3:
                    blocked.setdefault(t, set()).add(cell)
            if any(c in blocked.get(0, ()) for c in grid.pose_cells(start)):
                continue
            tt = Timetable(blocked)
            opt = ref_call(grid, tt, horizon, start, goal)
            if opt is not None:
                self.assertLessEqual(heuristic(start, goal), opt[0])
                checked += 1
        self.assertGreater(checked, 20)


class TimedDijkstraCrossCheckTest(unittest.TestCase):
    """时序 A* 与独立全状态 Dijkstra 参考对照最优总代价与到达时刻。"""

    def test_random_timed_scenarios_match_reference(self) -> None:
        rng = random.Random(54321)
        cases = 0
        for trial in range(60):
            w = rng.randint(2, 6)
            h = rng.randint(2, 6)
            density = rng.choice([0.0, 0.15, 0.3])
            obstacles = [
                (x, y) for x in range(w) for y in range(h) if rng.random() < density
            ]
            try:
                grid = Grid(w, h, obstacles=obstacles)
            except ValueError:
                continue
            horizon = rng.randint(3, 12)
            blocked = {}
            for t in range(horizon + 1):
                for x in range(w):
                    for y in range(h):
                        if (x, y) not in grid.obstacles and rng.random() < 0.08:
                            blocked.setdefault(t, set()).add((x, y))
            tt = Timetable(blocked)
            free_poses = [
                Pose(x, y, Heading(hd))
                for x in range(w)
                for y in range(h)
                for hd in range(4)
                if grid.is_pose_free(Pose(x, y, Heading(hd)))
                and not any(c in blocked.get(0, ()) for c in grid.pose_cells(Pose(x, y, Heading(hd))))
            ]
            if len(free_poses) < 2:
                continue
            start, goal = rng.sample(free_poses, 2)
            wait_cost = rng.choice([1, 2, 3])
            expected = ref_call(grid, tt, horizon, start, goal, wait_cost)
            result = plan_timed(grid, tt, start, goal, horizon=horizon, wait_cost=wait_cost)
            if expected is None:
                self.assertEqual(
                    result.status, SearchStatus.NO_PATH_WITHIN_HORIZON,
                    f"trial={trial} 参考时域内无解但 A* 给出 {result.status}",
                )
            else:
                self.assertEqual(
                    result.status, SearchStatus.OPTIMAL,
                    f"trial={trial} 参考有解{expected}但 A* 给出 {result.status}",
                )
                self.assertEqual(
                    (result.plan.total_cost, result.arrival_time), expected,
                    f"trial={trial} 最优代价/到达时刻不一致",
                )
                assert_timed_plan_valid(self, grid, tt, start, goal, result.plan, wait_cost)
            cases += 1
        self.assertGreaterEqual(cases, 40)  # 确保对照样本量


if __name__ == "__main__":
    unittest.main()
