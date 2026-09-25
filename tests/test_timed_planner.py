"""离散时间扩展验收测试：窄路临时占用、倒车让行、终点稍后被封、转向扫掠碰动态障碍、
WAIT 车体检查、独立时间展开 Dijkstra 对照（最优总代价与到达时刻）、逐步占用证据重放、
输入校验与预算语义。"""

from __future__ import annotations

import random
import unittest

from pose_grid_planner import (
    MAX_HORIZON,
    Action,
    Grid,
    Heading,
    Pose,
    ReplayError,
    SearchStatus,
    TimedGrid,
    TimedPlan,
    heuristic,
    plan,
    replay,
    timed_plan,
    timed_replay,
    timed_transition,
    transition,
)
from timed_dijkstra_reference import timed_dijkstra_reference

N, E, S, W = Heading.NORTH, Heading.EAST, Heading.SOUTH, Heading.WEST


def to_ref(pose: Pose) -> tuple[int, int, int]:
    return (pose.x, pose.y, int(pose.heading))


def assert_timed_plan_valid(
    testcase: unittest.TestCase,
    tgrid: TimedGrid,
    start: Pose,
    goal: Pose,
    wait_cost: int,
    p: TimedPlan,
) -> None:
    """路径自洽性：证据链完整、每步扫掠格避开静态障碍与起止两时刻禁占格并集、
    代价求和一致、到达时刻与终点正确；并用 timed_replay 逐步重放占用证据。"""
    testcase.assertEqual(sum(e.cost for e in p.evidence), p.total_cost)
    testcase.assertEqual(tuple(e.action for e in p.evidence), p.actions)
    testcase.assertEqual(p.arrival_time, len(p.actions))  # 每动作（含 WAIT）耗时一步
    pose, t = start, 0
    for e in p.evidence:
        testcase.assertEqual((e.from_pose, e.from_time), (pose, t))
        testcase.assertEqual(e.to_time, t + 1)
        forb = tgrid.blocked_at(t) | tgrid.blocked_at(t + 1)
        for cell in e.swept_cells:
            testcase.assertTrue(tgrid.grid.is_free(cell), f"扫掠格 {cell} 不空（静态）")
            testcase.assertNotIn(cell, forb, f"扫掠格 {cell} 在时刻 {t}/{t + 1} 被禁占")
        if e.action is Action.WAIT:
            # WAIT 原地不动，且必须检查车体全部两格
            testcase.assertEqual(e.to_pose, pose)
            testcase.assertEqual(
                tuple(sorted(e.swept_cells)), tuple(sorted(tgrid.grid.pose_cells(pose)))
            )
            testcase.assertEqual(e.cost, wait_cost)
        else:
            testcase.assertEqual(e.cost, e.action.cost)
        pose, t = e.to_pose, e.to_time
    testcase.assertEqual(pose, goal)
    testcase.assertEqual(t, p.arrival_time)
    # 重放一致性（补充性质，不替代独立参考）
    final_pose, final_time, _ = timed_replay(tgrid, start, p.actions, wait_cost=wait_cost)
    testcase.assertEqual((final_pose, final_time), (goal, p.arrival_time))


class NarrowCorridorTemporaryBlockTest(unittest.TestCase):
    """窄路临时占用：走廊必经格在时刻 1 被占，最优解是原地等待让其先通过。"""

    def setUp(self) -> None:
        self.grid = Grid(5, 1)
        self.start = Pose(0, 0, E)
        self.goal = Pose(3, 0, E)
        self.tgrid = TimedGrid(self.grid, 6, blocked={1: {(2, 0)}})

    def test_wait_for_moving_obstacle(self) -> None:
        r = timed_plan(self.tgrid, self.start, self.goal, wait_cost=1)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        # 时刻 0/1 出发的前进都会扫到 (2,0) 在时刻 1 的禁占 → 必须等两步
        self.assertEqual(r.plan.actions, (Action.WAIT, Action.WAIT) + (Action.FORWARD,) * 3)
        self.assertEqual(r.plan.total_cost, 2 * 1 + 3 * 10)
        self.assertEqual(r.plan.arrival_time, 5)
        # 与独立时间展开 Dijkstra 对照最优总代价与到达时刻
        self.assertEqual(
            timed_dijkstra_reference(5, 1, frozenset(), {1: {(2, 0)}}, 6, 1,
                                     to_ref(self.start), to_ref(self.goal)),
            (r.plan.total_cost, r.plan.arrival_time),
        )
        assert_timed_plan_valid(self, self.tgrid, self.start, self.goal, 1, r.plan)

    def test_arrival_exactly_at_horizon_allowed(self) -> None:
        tgrid = TimedGrid(self.grid, 5, blocked={1: {(2, 0)}})
        r = timed_plan(tgrid, self.start, self.goal, wait_cost=1)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.plan.arrival_time, 5)

    def test_tight_horizon_reports_in_horizon_only(self) -> None:
        # 时域 4 不够（需要 5 步）：只报时域内无路，且静态几何其实有解
        tgrid = TimedGrid(self.grid, 4, blocked={1: {(2, 0)}})
        r = timed_plan(tgrid, self.start, self.goal, wait_cost=1)
        self.assertEqual(r.status, SearchStatus.NO_PATH_IN_HORIZON)
        self.assertIsNone(r.plan)
        static = plan(self.grid, self.start, self.goal)
        self.assertEqual(static.status, SearchStatus.OPTIMAL)  # 静态几何并非无解


class BackwardYieldTest(unittest.TestCase):
    """只有倒车能让行：移动障碍自右向左穿过 1 格宽走廊，机器人必须先后退腾位。"""

    def setUp(self) -> None:
        self.grid = Grid(5, 1)
        self.start = Pose(2, 0, E)  # 占用 (2,0),(3,0)
        self.goal = Pose(3, 0, E)
        # 障碍依次占 (4,0)@(t1, t2)、(3,0)@(t2, t3)、(2,0)@t3（按出发/到达时刻并集生效）
        self.blocked = {1: {(4, 0)}, 2: {(4, 0), (3, 0)}, 3: {(3, 0), (2, 0)}}
        self.tgrid = TimedGrid(self.grid, 8, blocked=self.blocked)

    def test_must_reverse_to_yield(self) -> None:
        r = timed_plan(self.tgrid, self.start, self.goal, wait_cost=1)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertIn(Action.BACKWARD, r.plan.actions)  # 前进/等待都无法让行
        expected = timed_dijkstra_reference(
            5, 1, frozenset(), self.blocked, 8, 1, to_ref(self.start), to_ref(self.goal)
        )
        self.assertEqual((r.plan.total_cost, r.plan.arrival_time), expected)
        assert_timed_plan_valid(self, self.tgrid, self.start, self.goal, 1, r.plan)

    def test_same_pose_different_time_not_merged(self) -> None:
        # 最优路径倒车让行后重新经过起点姿态：同地点不同时间是不同状态，不能合并
        r = timed_plan(self.tgrid, self.start, self.goal, wait_cost=1)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        all_visits = [(e.from_pose, e.from_time) for e in r.plan.evidence]
        all_visits.append((r.plan.evidence[-1].to_pose, r.plan.evidence[-1].to_time))
        times = [t for p, t in all_visits if p == self.start]
        self.assertGreaterEqual(len(times), 2, "同姿态应在不同时刻重复出现（状态按时间区分）")


class GoalSealedLaterTest(unittest.TestCase):
    """终点稍后被封：必须赶在封锁前到达；被封后只报时域内无路。"""

    def setUp(self) -> None:
        self.grid = Grid(4, 1)
        self.start = Pose(0, 0, E)
        self.goal = Pose(2, 0, E)  # 占用 (2,0),(3,0)

    def test_arrive_before_seal(self) -> None:
        blocked = {t: {(3, 0)} for t in range(3, 7)}  # 时刻 3 起 (3,0) 被封
        tgrid = TimedGrid(self.grid, 6, blocked=blocked)
        r = timed_plan(tgrid, self.start, self.goal, wait_cost=1)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.plan.actions, (Action.FORWARD, Action.FORWARD))
        self.assertEqual(r.plan.arrival_time, 2)  # 到达即停，不要求之后占住终点
        self.assertEqual(r.plan.total_cost, 20)
        assert_timed_plan_valid(self, tgrid, self.start, self.goal, 1, r.plan)

    def test_sealed_goal_is_not_static_no_path(self) -> None:
        blocked = {t: {(3, 0)} for t in range(2, 7)}  # 时刻 2 起封死，到达 (需 t=2) 已不可能
        tgrid = TimedGrid(self.grid, 6, blocked=blocked)
        r = timed_plan(tgrid, self.start, self.goal, wait_cost=1)
        self.assertEqual(r.status, SearchStatus.NO_PATH_IN_HORIZON)
        self.assertNotEqual(r.status, SearchStatus.NO_PATH)
        # 同一几何的静态规划有解：时域结论不能当成静态几何结论
        static = plan(self.grid, self.start, self.goal)
        self.assertEqual(static.status, SearchStatus.OPTIMAL)
        self.assertEqual(static.plan.total_cost, 20)


class TurnSweepDynamicObstacleTest(unittest.TestCase):
    """转向扫掠碰动态障碍：2x2 扫掠格在出发/到达时刻被占则该时刻不可转向。"""

    def setUp(self) -> None:
        self.grid = Grid(3, 3)
        self.start = Pose(1, 1, E)
        self.goal = Pose(1, 1, N)
        # 左转扫掠 {(1,1),(2,1),(1,0),(2,0)}；(2,0) 在时刻 0、1 被占
        self.tgrid = TimedGrid(self.grid, 4, blocked={0: {(2, 0)}, 1: {(2, 0)}})

    def test_turn_delayed_until_sweep_clear(self) -> None:
        # 静态语义下该转向本来合法
        self.assertIsNotNone(transition(self.grid, self.start, Action.TURN_LEFT))
        # 时刻 0、1 出发的左转都因 (2,0) 被禁占而失败
        self.assertIsNone(timed_transition(self.tgrid, self.start, 0, Action.TURN_LEFT))
        self.assertIsNone(timed_transition(self.tgrid, self.start, 1, Action.TURN_LEFT))
        self.assertIsNotNone(timed_transition(self.tgrid, self.start, 2, Action.TURN_LEFT))

        r = timed_plan(self.tgrid, self.start, self.goal, wait_cost=1)
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.plan.actions, (Action.WAIT, Action.WAIT, Action.TURN_LEFT))
        self.assertEqual(r.plan.total_cost, 2 * 1 + 8)
        self.assertEqual(r.plan.arrival_time, 3)
        self.assertIn((2, 0), r.plan.evidence[2].swept_cells)  # 转向扫掠确实覆盖该格
        assert_timed_plan_valid(self, self.tgrid, self.start, self.goal, 1, r.plan)


class WaitChecksAllBodyCellsTest(unittest.TestCase):
    """WAIT 也要检查车体全部两格（含沿朝向的前格），且对起止两时刻生效。"""

    def test_front_cell_blocked_at_arrival_time(self) -> None:
        grid = Grid(4, 1)
        pose = Pose(1, 0, E)  # 车体 {(1,0),(2,0)}
        tgrid = TimedGrid(grid, 3, blocked={1: {(2, 0)}})
        self.assertIsNone(timed_transition(tgrid, pose, 0, Action.WAIT))  # 前格在 t+1 被占

    def test_front_cell_blocked_at_departure_time(self) -> None:
        grid = Grid(4, 1)
        pose = Pose(1, 0, E)
        tgrid = TimedGrid(grid, 3, blocked={0: {(2, 0)}})
        self.assertIsNone(timed_transition(tgrid, pose, 0, Action.WAIT))  # 前格在 t 被占

    def test_unrelated_cell_does_not_block_wait(self) -> None:
        grid = Grid(4, 1)
        pose = Pose(1, 0, E)
        tgrid = TimedGrid(grid, 3, blocked={1: {(0, 0)}})
        result = timed_transition(tgrid, pose, 0, Action.WAIT, wait_cost=3)
        self.assertIsNotNone(result)
        nxt, swept, cost = result
        self.assertEqual(nxt, pose)
        self.assertEqual(tuple(sorted(swept)), ((1, 0), (2, 0)))
        self.assertEqual(cost, 3)

    def test_wait_respects_static_obstacle(self) -> None:
        grid = Grid(3, 1, obstacles=[(1, 0)])
        pose = Pose(0, 0, E)  # 车体 {(0,0),(1,0)} 含静态障碍
        tgrid = TimedGrid(grid, 2)
        self.assertIsNone(timed_transition(tgrid, pose, 0, Action.WAIT))


class TimedDijkstraCrossCheckTest(unittest.TestCase):
    """小地图独立 Dijkstra 对照：最优总代价与到达时刻必须一致；有解路径重放每步占用证据。"""

    def test_random_timed_grids_match_reference(self) -> None:
        rng = random.Random(20260925)
        cases = 0
        for trial in range(60):
            w = rng.randint(2, 5)
            h = rng.randint(2, 5)
            horizon = rng.randint(2, 8)
            wait_cost = rng.choice([1, 2, 5])
            density = rng.choice([0.0, 0.15, 0.25])
            obstacles = [
                (x, y) for x in range(w) for y in range(h) if rng.random() < density
            ]
            grid = Grid(w, h, obstacles=obstacles)
            blocked = {}
            for t in range(horizon + 1):
                cells = {(x, y) for x in range(w) for y in range(h) if rng.random() < 0.1}
                if cells:
                    blocked[t] = cells
            tgrid = TimedGrid(grid, horizon, blocked)
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
            if any(c in blocked.get(0, ()) for c in grid.pose_cells(start)):
                continue  # 起点在时刻 0 被禁占属于非法输入，不在本对照范围
            expected = timed_dijkstra_reference(
                w, h, grid.obstacles, blocked, horizon, wait_cost,
                to_ref(start), to_ref(goal),
            )
            result = timed_plan(tgrid, start, goal, wait_cost=wait_cost)
            if expected is None:
                self.assertEqual(
                    result.status, SearchStatus.NO_PATH_IN_HORIZON,
                    f"trial={trial} 参考时域内无解但 timed_plan 给出 {result.status}",
                )
            else:
                exp_cost, exp_arrival = expected
                self.assertEqual(
                    result.status, SearchStatus.OPTIMAL,
                    f"trial={trial} 参考有解{expected}但 timed_plan 给出 {result.status}",
                )
                self.assertEqual(result.plan.total_cost, exp_cost, f"trial={trial} 总代价不一致")
                self.assertEqual(result.plan.arrival_time, exp_arrival, f"trial={trial} 到达时刻不一致")
                assert_timed_plan_valid(self, tgrid, start, goal, wait_cost, result.plan)
            cases += 1
        self.assertGreaterEqual(cases, 40)  # 确保对照样本量


class TimedHeuristicAdmissibleTest(unittest.TestCase):
    """加入 WAIT 与动态禁占后，h=max(曼哈顿*10, 最少转向*8) 仍是可采纳下界。"""

    def test_heuristic_never_exceeds_timed_optimum(self) -> None:
        rng = random.Random(777)
        checked = 0
        for _ in range(120):
            w = rng.randint(2, 5)
            h = rng.randint(2, 5)
            horizon = rng.randint(3, 10)
            wait_cost = rng.choice([1, 3])
            obstacles = [(x, y) for x in range(w) for y in range(h) if rng.random() < 0.15]
            grid = Grid(w, h, obstacles=obstacles)
            blocked = {}
            for t in range(horizon + 1):
                cells = {(x, y) for x in range(w) for y in range(h) if rng.random() < 0.05}
                if cells:
                    blocked[t] = cells
            tgrid = TimedGrid(grid, horizon, blocked)
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
            if any(c in blocked.get(0, ()) for c in grid.pose_cells(start)):
                continue
            r = timed_plan(tgrid, start, goal, wait_cost=wait_cost)
            if r.status is SearchStatus.OPTIMAL:
                self.assertLessEqual(heuristic(start, goal), r.plan.total_cost)
                checked += 1
        self.assertGreaterEqual(checked, 10)


class TimedInputValidationTest(unittest.TestCase):
    """时间扩展的错误语义：可定位、不静默纠正。"""

    def test_horizon_validation(self) -> None:
        grid = Grid(3, 3)
        TimedGrid(grid, 0)  # 0 时域本身合法（只允许 start==goal）
        with self.assertRaises(TypeError):
            TimedGrid(grid, True)
        with self.assertRaises(TypeError):
            TimedGrid(grid, 1.5)
        with self.assertRaisesRegex(ValueError, "horizon"):
            TimedGrid(grid, -1)
        with self.assertRaisesRegex(ValueError, "horizon"):
            TimedGrid(grid, MAX_HORIZON + 1)
        with self.assertRaises(TypeError):
            TimedGrid("not a grid", 3)

    def test_blocked_validation(self) -> None:
        grid = Grid(3, 3)
        with self.assertRaises(TypeError):
            TimedGrid(grid, 3, blocked="not a mapping")
        with self.assertRaisesRegex(ValueError, "时刻键"):
            TimedGrid(grid, 3, blocked={4: {(0, 0)}})  # 超出时域
        with self.assertRaisesRegex(ValueError, "时刻键"):
            TimedGrid(grid, 3, blocked={-1: {(0, 0)}})
        with self.assertRaises(TypeError):
            TimedGrid(grid, 3, blocked={True: {(0, 0)}})  # bool 时刻键
        with self.assertRaisesRegex(ValueError, r"blocked\[1\]\[0\]"):
            TimedGrid(grid, 3, blocked={1: {(9, 9)}})  # 越界，报时刻与索引
        with self.assertRaisesRegex(ValueError, r"blocked\[0\]\[1\].*重复"):
            TimedGrid(grid, 3, blocked={0: [(1, 1), (1, 1)]})  # 重复不静默去重
        with self.assertRaises(TypeError):
            TimedGrid(grid, 3, blocked={1: (0, 0)})  # 值不是坐标对的可迭代对象
        with self.assertRaisesRegex(ValueError, "time"):
            TimedGrid(grid, 3).blocked_at(4)

    def test_wait_cost_validation(self) -> None:
        tgrid = TimedGrid(Grid(3, 3), 3)
        start, goal = Pose(0, 0, E), Pose(1, 0, E)
        with self.assertRaisesRegex(ValueError, "wait_cost"):
            timed_plan(tgrid, start, goal, wait_cost=0)
        with self.assertRaisesRegex(ValueError, "wait_cost"):
            timed_plan(tgrid, start, goal, wait_cost=-2)
        with self.assertRaises(TypeError):
            timed_plan(tgrid, start, goal, wait_cost=True)
        with self.assertRaises(TypeError):
            timed_plan(tgrid, start, goal, wait_cost=1.0)
        with self.assertRaises(ValueError):
            _ = Action.WAIT.cost  # WAIT 代价不是固定常量

    def test_start_blocked_at_time_zero(self) -> None:
        grid = Grid(4, 1)
        tgrid = TimedGrid(grid, 4, blocked={0: {(1, 0)}})
        with self.assertRaisesRegex(ValueError, "start.*时刻 0"):
            timed_plan(tgrid, Pose(0, 0, E), Pose(2, 0, E))
        with self.assertRaisesRegex(ReplayError, "start.*时刻 0"):
            timed_replay(tgrid, Pose(0, 0, E), ())

    def test_statically_invalid_goal_still_rejected(self) -> None:
        tgrid = TimedGrid(Grid(3, 3), 3)
        with self.assertRaisesRegex(ValueError, "goal.*越界"):
            timed_plan(tgrid, Pose(0, 0, E), Pose(0, 0, N))

    def test_wait_rejected_in_static_interfaces(self) -> None:
        grid = Grid(3, 3)
        with self.assertRaisesRegex(ValueError, "WAIT"):
            transition(grid, Pose(1, 1, E), Action.WAIT)
        with self.assertRaisesRegex(ReplayError, "WAIT"):
            replay(grid, Pose(1, 1, E), (Action.WAIT,))

    def test_timed_transition_time_range(self) -> None:
        tgrid = TimedGrid(Grid(3, 3), 2)
        with self.assertRaisesRegex(ValueError, "出发时刻"):
            timed_transition(tgrid, Pose(0, 0, E), 2, Action.WAIT)  # 到达时刻 3 > horizon
        with self.assertRaisesRegex(ValueError, "出发时刻"):
            timed_transition(tgrid, Pose(0, 0, E), -1, Action.WAIT)

    def test_timed_replay_errors_locatable(self) -> None:
        grid = Grid(4, 1)
        tgrid = TimedGrid(grid, 2, blocked={1: {(1, 0)}})
        with self.assertRaisesRegex(ReplayError, "第 0 步"):
            timed_replay(tgrid, Pose(0, 0, E), (Action.FORWARD,))  # 扫到 t1 禁占格
        tgrid_open = TimedGrid(grid, 2)
        with self.assertRaisesRegex(ReplayError, "超出时域"):
            timed_replay(tgrid_open, Pose(0, 0, E), (Action.WAIT,) * 3)  # 第 2 步出发时刻 2 >= horizon
        with self.assertRaisesRegex(ReplayError, r"actions\[0\]"):
            timed_replay(tgrid_open, Pose(0, 0, E), ("WAIT",))


class TimedBudgetAndTrivialTest(unittest.TestCase):
    """预算语义与平凡情形。"""

    def test_budget_exhausted_distinct(self) -> None:
        tgrid = TimedGrid(Grid(4, 4), 10)
        start, goal = Pose(0, 0, E), Pose(3, 3, W)
        r0 = timed_plan(tgrid, start, goal, max_expansions=0)
        self.assertEqual(r0.status, SearchStatus.BUDGET_EXHAUSTED)
        self.assertIsNone(r0.plan)
        r_full = timed_plan(tgrid, start, goal)
        self.assertEqual(r_full.status, SearchStatus.OPTIMAL)
        with self.assertRaisesRegex(ValueError, "max_expansions"):
            timed_plan(tgrid, start, goal, max_expansions=-1)

    def test_start_equals_goal_zero_cost(self) -> None:
        tgrid = TimedGrid(Grid(3, 3), 5, blocked={3: {(1, 1)}})  # 之后被封也无所谓：到达即停
        r = timed_plan(tgrid, Pose(1, 1, E), Pose(1, 1, E))
        self.assertEqual(r.status, SearchStatus.OPTIMAL)
        self.assertEqual(r.plan.total_cost, 0)
        self.assertEqual(r.plan.arrival_time, 0)
        self.assertEqual(r.plan.actions, ())

    def test_zero_horizon_only_trivial(self) -> None:
        grid = Grid(3, 1)
        tgrid = TimedGrid(grid, 0)
        r_ok = timed_plan(tgrid, Pose(0, 0, E), Pose(0, 0, E))
        self.assertEqual(r_ok.status, SearchStatus.OPTIMAL)
        r_no = timed_plan(tgrid, Pose(0, 0, E), Pose(1, 0, E))
        self.assertEqual(r_no.status, SearchStatus.NO_PATH_IN_HORIZON)


if __name__ == "__main__":
    unittest.main()
