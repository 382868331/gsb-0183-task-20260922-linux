"""A* 规划器测试：对照独立 Dijkstra 参考 + 题面场景。

独立参考 (tests/dijkstra_reference.py) 不导入被测包；被测 A* 的最优代价/
可达性必须与参考一致。动作序列期望在少数小场景上用硬编码断言。
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose_grid_planner import (  # noqa: E402
    Action,
    GridWorld,
    Heading,
    InvalidGridError,
    InvalidParameterError,
    InvalidPoseError,
    PlanStatus,
    Pose,
    heuristic,
    plan,
)
from dijkstra_reference import RefWorld, dijkstra, reachable  # noqa: E402

H = (Heading.EAST, Heading.NORTH, Heading.WEST, Heading.SOUTH)


def replay_clear(world, start, result):
    """用证据重放：每步扫掠格必须在界内且非障碍；姿态连续；占用格安全。"""
    cur = tuple(start.state)
    occupied_trace = []
    for ev in result.evidence:
        assert ev.src_pose == cur, "证据起点姿态与重放姿态不一致"
        for cx, cy in ev.swept_cells:
            assert world.is_free(cx, cy), f"扫掠格 ({cx},{cy}) 非空: {ev}"
        # 起终占用格必须是扫掠集合的子集
        def occ(s):
            x, y, h = s
            dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[h]
            return {(x, y), (x + dx, y + dy)}

        assert occ(ev.src_pose) <= set(ev.swept_cells)
        assert occ(ev.dst_pose) <= set(ev.swept_cells)
        # 动作语义
        x, y, h = ev.src_pose
        if ev.action is Action.FORWARD:
            dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[h]
            assert ev.dst_pose == (x + dx, y + dy, h)
            assert ev.swept_kind == "union"
        elif ev.action is Action.BACKWARD:
            dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[h]
            assert ev.dst_pose == (x - dx, y - dy, h)
            assert ev.swept_kind == "union"
        else:
            turn = 1 if ev.action is Action.TURN_LEFT else -1
            assert ev.dst_pose == (x, y, (h + turn) % 4)
            assert ev.swept_kind == "2x2"
            assert len(set(ev.swept_cells)) == 4
        cur = ev.dst_pose
        occupied_trace.append(occ(cur))
    assert cur == tuple(result.goal)
    return occupied_trace


class TestHeuristic(unittest.TestCase):
    def test_values(self):
        # 同朝向：只看曼哈顿
        self.assertEqual(heuristic(Pose(0, 0, Heading.EAST), Pose(3, 4, Heading.EAST)), 70)
        # 曼哈顿为 0 但朝向相反：2 次转向 * 8
        self.assertEqual(heuristic(Pose(2, 2, Heading.EAST), Pose(2, 2, Heading.WEST)), 16)
        # 侧向朝向 1 次转向 8，被曼哈顿 10 盖过
        self.assertEqual(heuristic(Pose(0, 0, Heading.EAST), Pose(1, 0, Heading.NORTH)), 10)
        # 原地 90 度：8
        self.assertEqual(heuristic(Pose(0, 0, Heading.EAST), Pose(0, 0, Heading.NORTH)), 8)

    def test_admissible_on_random_edges(self):
        rng = random.Random(1234)
        for _ in range(200):
            w = h = 6
            obs = {(rng.randrange(w), rng.randrange(h)) for _ in range(6)}
            world = GridWorld(w, h, obs)
            states = list(world.all_states())
            s = Pose(*states[rng.randrange(len(states))])
            for tr in world.neighbors(s):
                self.assertLessEqual(
                    heuristic(s, tr.dst),
                    tr.cost,
                    f"启发式超过单边代价: {s} -> {tr.dst}",
                )


class TestBasicPlans(unittest.TestCase):
    def test_straight_forward(self):
        world = GridWorld(5, 3, set())
        r = plan(world, Pose(0, 1, Heading.EAST), Pose(3, 1, Heading.EAST))
        self.assertIs(r.status, PlanStatus.SUCCESS)
        self.assertEqual(r.actions, (Action.FORWARD,) * 3)
        self.assertEqual(r.total_cost, 30)
        replay_clear(world, Pose(0, 1, Heading.EAST), r)

    def test_start_equals_goal(self):
        world = GridWorld(3, 3, set())
        r = plan(world, Pose(1, 1, Heading.NORTH), Pose(1, 1, Heading.NORTH))
        self.assertIs(r.status, PlanStatus.SUCCESS)
        self.assertEqual(r.actions, ())
        self.assertEqual(r.total_cost, 0)

    def test_goal_heading_requires_turn(self):
        # 同格到达但目标朝向相反：必须两次转向（16），不得报告 0
        world = GridWorld(4, 4, set())
        r = plan(world, Pose(1, 1, Heading.EAST), Pose(1, 1, Heading.WEST))
        self.assertIs(r.status, PlanStatus.SUCCESS)
        self.assertEqual(r.total_cost, 16)
        self.assertEqual(len(r.actions), 2)
        self.assertTrue(all(a in (Action.TURN_LEFT, Action.TURN_RIGHT) for a in r.actions))

    def test_turn_cheaper_than_drive_loop(self):
        # 目标在后方一格：后退 14 vs 两次转+前进+两次转 = 46，应选后退
        world = GridWorld(5, 5, set())
        r = plan(world, Pose(2, 2, Heading.EAST), Pose(1, 2, Heading.EAST))
        self.assertIs(r.status, PlanStatus.SUCCESS)
        self.assertEqual(r.actions, (Action.BACKWARD,))
        self.assertEqual(r.total_cost, 14)

    def test_backward_forced_in_dead_end(self):
        """后退必需：窄盲端内不能转向、不能前进，只有后退一条合法原语。"""
        # 7x3 世界。车位姿 (1,1)/EAST 占据 (1,1)、(2,1)：
        #   东侧 (3,*) 三格全封（前进撞 (3,1)）；
        #   转向扫掠的 2x2 矩形分别含 (2,2)（左转）与 (2,0)（右转），均被封。
        obs = {(2, 0), (2, 2), (3, 0), (3, 1), (3, 2)}
        world = GridWorld(7, 3, obs)
        start = Pose(1, 1, Heading.EAST)
        self.assertIsNone(world.apply(start, Action.FORWARD))
        self.assertIsNone(world.apply(start, Action.TURN_LEFT))
        self.assertIsNone(world.apply(start, Action.TURN_RIGHT))
        # 唯一合法原语：后退到 (0,1)/EAST
        self.assertIsNotNone(world.apply(start, Action.BACKWARD))

        r = plan(world, start, Pose(0, 1, Heading.EAST))
        self.assertIs(r.status, PlanStatus.SUCCESS)
        self.assertEqual(r.actions, (Action.BACKWARD,))
        self.assertEqual(r.total_cost, 14)
        replay_clear(world, start, r)


class TestNarrowCorridor(unittest.TestCase):
    def _corridor(self):
        # 水平 1 格宽走廊 y=1，x=0..4，上下全是墙；墙东端封住，西侧开口。
        obs = set()
        for x in range(6):
            obs.add((x, 0))
            obs.add((x, 2))
        obs.add((5, 1))
        return GridWorld(6, 3, obs)

    def test_can_drive_straight(self):
        world = self._corridor()
        r = plan(world, Pose(0, 1, Heading.EAST), Pose(3, 1, Heading.EAST))
        self.assertIs(r.status, PlanStatus.SUCCESS)
        self.assertEqual(r.actions, (Action.FORWARD, Action.FORWARD, Action.FORWARD))
        replay_clear(world, Pose(0, 1, Heading.EAST), r)

    def test_cannot_turn_inside(self):
        world = self._corridor()
        for x in range(0, 4):
            p = Pose(x, 1, Heading.EAST)
            self.assertIsNone(world.apply(p, Action.TURN_LEFT))
            self.assertIsNone(world.apply(p, Action.TURN_RIGHT))

    def test_goal_heading_unreachable_in_dead_end(self):
        # 车在封死的窄走廊里朝东，目标要求朝西：无处可转 -> 明确无解
        world = self._corridor()
        r = plan(world, Pose(2, 1, Heading.EAST), Pose(2, 1, Heading.WEST))
        self.assertIs(r.status, PlanStatus.NO_PATH)
        self.assertEqual(r.actions, ())
        self.assertIsNone(r.total_cost)


class TestSweepSemantics(unittest.TestCase):
    def test_turn_2x2_exact_cells(self):
        world = GridWorld(5, 5, set())
        tr = world.apply(Pose(2, 2, Heading.EAST), Action.TURN_LEFT)
        self.assertIsNotNone(tr)
        # EAST 占 (2,2)(3,2)；NORTH 占 (2,2)(2,3)；2x2 = 四格
        self.assertEqual(set(tr.swept), {(2, 2), (3, 2), (2, 3), (3, 3)})
        self.assertEqual(tr.swept_kind, "2x2")
        # 对角格 (3,3) 放障碍后转向必须失败，尽管它不属于起终占用格
        world2 = GridWorld(5, 5, {(3, 3)})
        self.assertIsNone(world2.apply(Pose(2, 2, Heading.EAST), Action.TURN_LEFT))
        # 但前进不受该障碍影响（并集不含 (3,3)）
        self.assertIsNotNone(world2.apply(Pose(2, 2, Heading.EAST), Action.FORWARD))

    def test_translation_union_cells(self):
        world = GridWorld(5, 5, set())
        tr = world.apply(Pose(2, 2, Heading.EAST), Action.FORWARD)
        self.assertEqual(set(tr.swept), {(2, 2), (3, 2), (4, 2)})
        self.assertEqual(tr.swept_kind, "union")
        # 远端新格 (4,2) 有障碍 -> 前进失败
        world2 = GridWorld(5, 5, {(4, 2)})
        self.assertIsNone(world2.apply(Pose(2, 2, Heading.EAST), Action.FORWARD))

    def test_boundary_turn_blocked(self):
        # 贴东北角：EAST 转向 NORTH 的 2x2 矩形出界
        world = GridWorld(4, 4, set())
        self.assertIsNone(world.apply(Pose(2, 3, Heading.EAST), Action.TURN_LEFT))
        # 反方向（矩形在界内）可行
        self.assertIsNotNone(world.apply(Pose(2, 3, Heading.EAST), Action.TURN_RIGHT))


class TestNoPathAndBudget(unittest.TestCase):
    def test_disconnected_no_path(self):
        # 竖墙隔断，且目标姿态本身合法但不可达
        obs = {(2, y) for y in range(5)}
        world = GridWorld(5, 5, obs)
        r = plan(world, Pose(0, 2, Heading.EAST), Pose(3, 2, Heading.EAST))
        self.assertIs(r.status, PlanStatus.NO_PATH)

    def test_budget_exhausted_distinct_from_no_path(self):
        world = GridWorld(20, 20, set())
        r = plan(
            world,
            Pose(0, 0, Heading.EAST),
            Pose(19, 19, Heading.WEST),
            max_expansions=1,
        )
        self.assertIs(r.status, PlanStatus.BUDGET_EXHAUSTED)
        self.assertIsNone(r.total_cost)
        # 预算充足时同问题有解，证明“预算耗尽 ≠ 无解”
        r2 = plan(world, Pose(0, 0, Heading.EAST), Pose(19, 19, Heading.WEST))
        self.assertIs(r2.status, PlanStatus.SUCCESS)

    def test_budget_zero_rejected(self):
        with self.assertRaises(InvalidParameterError):
            plan(GridWorld(3, 3, ()), (1, 1, 0), (1, 1, 0), max_expansions=0)


class TestReferenceAgreement(unittest.TestCase):
    """与独立 Dijkstra 全状态参考的批量一致性（最优代价 + 可达性 + 证据重放）。"""

    def _cmp_world(self, w, h, obs, seed):
        world = GridWorld(w, h, obs)
        ref = RefWorld(w, h, obs)
        states = list(world.all_states())
        self.assertEqual(set(states), set(ref.all_valid_states()))
        if len(states) < 2:
            return  # 无可配对姿态（如 1x1 世界放不下 2 格车身）
        rng = random.Random(seed)
        rng.shuffle(states)
        # 每个世界抽若干起终对
        pairs = []
        for _ in range(12):
            s = states[rng.randrange(len(states))]
            g = states[rng.randrange(len(states))]
            pairs.append((s, g))
        for s, g in pairs:
            res = plan(world, Pose(*s), Pose(*g))
            ref_res = dijkstra(ref, s, g)
            if ref_res is None:
                self.assertIs(res.status, PlanStatus.NO_PATH, f"{s}->{g} 参考无解")
                self.assertIsNone(res.total_cost)
            else:
                self.assertIs(res.status, PlanStatus.SUCCESS, f"{s}->{g}")
                self.assertEqual(res.total_cost, ref_res[0], f"{s}->{g} 代价不符")
                # A* 动作序列自身重放必须合法且到达
                replay_clear(world, Pose(*s), res)
                # 且动作序列累计代价一致
                self.assertEqual(sum(a.cost for a in res.actions), ref_res[0])

    def test_empty_worlds(self):
        self._cmp_world(1, 1, set(), 1)
        self._cmp_world(3, 3, set(), 2)
        self._cmp_world(5, 4, set(), 3)

    def test_random_worlds(self):
        rng = random.Random(99)
        for seed in range(20):
            w, h = rng.randint(2, 9), rng.randint(2, 9)
            n = rng.randint(0, w * h // 3)
            cells = [(x, y) for x in range(w) for y in range(h)]
            rng.shuffle(cells)
            obs = set(cells[:n])
            self._cmp_world(w, h, obs, 1000 + seed)

    def test_full_20x20_matches_reference(self):
        rng = random.Random(7)
        cells = [(x, y) for x in range(20) for y in range(20)]
        rng.shuffle(cells)
        obs = set(cells[:120])  # 约 30% 障碍
        self._cmp_world(20, 20, obs, 4242)

    def test_edge_legality_matches_reference(self):
        # 逐状态比对两边的合法边集合（动作 -> (dst, swept, kind)）
        rng = random.Random(55)
        for _ in range(10):
            w, h = rng.randint(2, 8), rng.randint(2, 8)
            cells = [(x, y) for x in range(w) for y in range(h)]
            rng.shuffle(cells)
            obs = set(cells[: rng.randint(0, w * h // 3)])
            world = GridWorld(w, h, obs)
            ref = RefWorld(w, h, obs)
            for s in world.all_states():
                got = {}
                for tr in world.neighbors(Pose(*s)):
                    got[int(tr.action)] = (
                        tr.dst.state,
                        tuple(sorted(tr.swept)),
                        tr.swept_kind,
                    )
                want = {}
                for nxt, act, _c, sw, kind in ref.edges(s):
                    want[act] = (nxt, sw, kind)
                self.assertEqual(got, want, f"边集合不一致 @ {s} obs={obs}")


class TestReachabilitySanity(unittest.TestCase):
    def test_replay_against_reachable_set(self):
        rng = random.Random(31337)
        for _ in range(15):
            w, h = rng.randint(2, 10), rng.randint(2, 10)
            cells = [(x, y) for x in range(w) for y in range(h)]
            rng.shuffle(cells)
            obs = set(cells[: rng.randint(0, w * h // 3)])
            world = GridWorld(w, h, obs)
            ref = RefWorld(w, h, obs)
            states = list(world.all_states())
            s = states[rng.randrange(len(states))]
            reach = reachable(ref, s)
            for _ in range(6):
                g = states[rng.randrange(len(states))]
                res = plan(world, Pose(*s), Pose(*g))
                if g in reach:
                    self.assertIs(res.status, PlanStatus.SUCCESS)
                else:
                    self.assertIs(res.status, PlanStatus.NO_PATH)


class TestInputValidation(unittest.TestCase):
    def test_bool_rejected_everywhere(self):
        with self.assertRaises(InvalidParameterError):
            GridWorld(True, 3)
        with self.assertRaises(InvalidParameterError):
            GridWorld(3, False)
        with self.assertRaises(InvalidParameterError):
            Pose(1, 1, True)
        with self.assertRaises(InvalidParameterError):
            Pose(True, 1, Heading.EAST)
        with self.assertRaises(InvalidParameterError):
            Pose(1, 1.0, Heading.EAST)

    def test_nan_infinity_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(InvalidParameterError):
                GridWorld(bad, 3)  # type: ignore[arg-type]
            with self.assertRaises(InvalidParameterError):
                Pose(1, bad, Heading.EAST)  # type: ignore[arg-type]

    def test_size_bounds(self):
        GridWorld(1, 1)
        GridWorld(20, 20)
        for bad in (0, 21, -1):
            with self.assertRaises(InvalidParameterError):
                GridWorld(bad, 5)
            with self.assertRaises(InvalidParameterError):
                GridWorld(5, bad)

    def test_bad_obstacles(self):
        with self.assertRaises(InvalidGridError):
            GridWorld(3, 3, [(3, 2)])  # 越界
        with self.assertRaises(InvalidGridError):
            GridWorld(3, 3, [(1, 1), (1, 1)])  # 重复
        with self.assertRaises(InvalidGridError):
            GridWorld(3, 3, [(1, True)])  # bool 坐标
        with self.assertRaises(InvalidGridError):
            GridWorld(3, 3, "ab")  # 非坐标集合
        with self.assertRaises(InvalidGridError):
            GridWorld(3, 3, [(1, 2, 3)])

    def test_pose_on_obstacle_and_out_of_bounds(self):
        world = GridWorld(3, 3, {(1, 1)})
        with self.assertRaises(InvalidPoseError) as cm:
            plan(world, Pose(1, 0, Heading.NORTH), Pose(2, 2, Heading.EAST))
        self.assertEqual(cm.exception.where, "start")
        world2 = GridWorld(3, 3, set())
        with self.assertRaises(InvalidPoseError) as cm:
            plan(world2, Pose(2, 2, Heading.EAST), Pose(0, 0, Heading.EAST))
        self.assertEqual(cm.exception.where, "start")  # 鼻端 (3,2) 出界
        with self.assertRaises(InvalidPoseError) as cm:
            plan(world2, Pose(0, 0, Heading.EAST), Pose(2, 2, Heading.EAST))
        self.assertEqual(cm.exception.where, "goal")

    def test_bad_world_and_heading(self):
        with self.assertRaises(InvalidParameterError):
            plan("not-a-world", (0, 0, 0), (1, 1, 0))  # type: ignore[arg-type]
        with self.assertRaises(InvalidParameterError):
            plan(GridWorld(3, 3, ()), (0, 0, 4), (1, 1, 0))
        with self.assertRaises(InvalidParameterError):
            plan(GridWorld(3, 3, ()), (0, 0, "E"), (1, 1, 0))
        with self.assertRaises(InvalidParameterError):
            plan(GridWorld(3, 3, ()), (0, 0, 0, 0), (1, 1, 0))

    def test_immutable_world_reuse(self):
        # 同一世界对象重复查询不应有任何状态残留
        world = GridWorld(4, 4, {(2, 2)})
        r1 = plan(world, Pose(0, 0, Heading.EAST), Pose(1, 1, Heading.NORTH))
        r2 = plan(world, Pose(0, 0, Heading.EAST), Pose(1, 1, Heading.NORTH))
        self.assertEqual(r1.total_cost, r2.total_cost)
        self.assertEqual(r1.actions, r2.actions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
