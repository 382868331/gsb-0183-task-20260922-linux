"""铰接拖车姿态格的验收测试。"""

from __future__ import annotations

import random
import unittest

from pose_grid_planner import (
    Action,
    ArticulatedPose,
    Grid,
    Heading,
    Pose,
    ReplayError,
    SearchStatus,
    articulated_heuristic,
    articulated_transition,
    plan,
    plan_articulated,
    replay_articulated,
)
from articulated_dijkstra_reference import dijkstra_articulated

E, N, S, W = Heading.EAST, Heading.NORTH, Heading.SOUTH, Heading.WEST


def state_tuple(p):
    return p.x, p.y, int(p.tractor_heading), int(p.trailer_heading)


def assert_articulated_plan_valid(testcase, grid, start, goal, result_plan):
    testcase.assertEqual(sum(e.cost for e in result_plan.evidence), result_plan.total_cost)
    pose = start
    for step in result_plan.evidence:
        testcase.assertEqual(step.from_pose, pose)
        testcase.assertLessEqual(step.to_pose.articulation_turns(), 1)
        testcase.assertTrue(step.tractor_swept_cells)
        testcase.assertTrue(step.trailer_swept_cells)
        testcase.assertTrue(step.hitch_bar_swept_cells)
        for cell in step.swept_cells:
            testcase.assertTrue(grid.is_free(cell), f"扫掠障碍/越界: {cell}")
        nxt, evidence = articulated_transition(grid, pose, step.action)
        testcase.assertEqual(nxt, step.to_pose)
        testcase.assertEqual(tuple(sorted(evidence.swept_cells)), tuple(sorted(step.swept_cells)))
        testcase.assertEqual(evidence.tractor_swept_cells, step.tractor_swept_cells)
        testcase.assertEqual(evidence.trailer_swept_cells, step.trailer_swept_cells)
        testcase.assertEqual(evidence.hitch_bar_swept_cells, step.hitch_bar_swept_cells)
        pose = nxt
    testcase.assertEqual(pose, goal)
    final, replayed = replay_articulated(grid, start, result_plan.actions)
    testcase.assertEqual(final, goal)
    testcase.assertEqual(len(replayed), len(result_plan.evidence))


class ArticulatedScenarioTest(unittest.TestCase):
    def test_single_tractor_passes_but_trailer_cannot_turn_corner(self):
        rows = ["####", "####", "####", "....", "#...", "#..."]
        grid = Grid.from_rows(rows)
        tractor_start = Pose(1, 3, E)
        tractor_goal = Pose(1, 4, S)
        tractor_result = plan(grid, tractor_start, tractor_goal)
        self.assertEqual(tractor_result.status, SearchStatus.OPTIMAL)
        self.assertIn(Action.TURN_RIGHT, tractor_result.plan.actions)

        trailer_start = ArticulatedPose(1, 3, E, E)
        trailer_goal = ArticulatedPose(1, 4, S, S)
        trailer_result = plan_articulated(grid, trailer_start, trailer_goal)
        self.assertEqual(trailer_result.status, SearchStatus.NO_PATH)
        self.assertIsNone(trailer_result.plan)

        turned = ArticulatedPose(1, 3, S, E)
        self.assertTrue(turned.within_articulation_limit())
        self.assertIsNone(articulated_transition(grid, turned, Action.FORWARD))
        nxt, evidence = articulated_transition(Grid.from_rows([*rows[:4], "....", "#..."]), turned, Action.FORWARD)
        self.assertEqual(nxt, ArticulatedPose(1, 4, S, S))
        self.assertIn((0, 4), evidence.swept_cells)

    def test_must_straighten_before_backing(self):
        grid = Grid.from_rows(["...", "...", "#.#", "#.#", "..."])
        start = ArticulatedPose(1, 1, E, N)
        goal = ArticulatedPose(1, 3, N, N)
        self.assertIsNone(articulated_transition(grid, start, Action.BACKWARD))
        result = plan_articulated(grid, start, goal)
        self.assertEqual(result.status, SearchStatus.OPTIMAL)
        self.assertEqual(result.plan.actions, (Action.TURN_LEFT, Action.BACKWARD, Action.BACKWARD))
        self.assertEqual(result.plan.total_cost, 36)
        assert_articulated_plan_valid(self, grid, start, goal, result.plan)


class ArticulatedRulesTest(unittest.TestCase):
    def test_articulation_limit_rejects_second_turn(self):
        grid = Grid(5, 5)
        perpendicular = ArticulatedPose(2, 2, E, N)
        self.assertIsNone(articulated_transition(grid, perpendicular, Action.TURN_RIGHT))
        self.assertIsNotNone(articulated_transition(grid, perpendicular, Action.TURN_LEFT))

    def test_invalid_start_articulation_is_value_error(self):
        with self.assertRaisesRegex(ValueError, "铰接角"):
            plan_articulated(Grid(6, 6), ArticulatedPose(2, 2, N, S), ArticulatedPose(3, 3, N, N))

    def test_replay_error_mentions_step(self):
        with self.assertRaisesRegex(ReplayError, "第 0 步"):
            replay_articulated(Grid(5, 5), ArticulatedPose(2, 2, E, N), (Action.BACKWARD,))

    def test_hitch_bar_corner_cells_are_checked(self):
        grid = Grid.from_rows([".....", ".....", ".....", ".....", ".#..."])
        self.assertIsNone(articulated_transition(grid, ArticulatedPose(2, 3, S, E), Action.FORWARD))

    def test_budget_distinct_from_no_path(self):
        grid = Grid.from_rows(["####", "####", "####", "....", "#...", "#..."])
        start = ArticulatedPose(1, 3, E, E)
        goal = ArticulatedPose(1, 4, S, S)
        self.assertEqual(plan_articulated(grid, start, goal, max_expansions=0).status, SearchStatus.BUDGET_EXHAUSTED)
        self.assertEqual(plan_articulated(grid, start, goal, max_expansions=1).status, SearchStatus.BUDGET_EXHAUSTED)
        self.assertEqual(plan_articulated(grid, start, goal).status, SearchStatus.NO_PATH)


class ArticulatedDijkstraCrossCheck(unittest.TestCase):
    def test_random_small_grid_matches_reference(self):
        rng = random.Random(20260927)
        checked = 0
        for trial in range(80):
            width = rng.randint(3, 7)
            height = rng.randint(3, 7)
            obstacles = {
                (x, y)
                for x in range(width)
                for y in range(height)
                if rng.random() < rng.choice((0.0, 0.12, 0.24))
            }
            grid = Grid(width, height, obstacles)
            states = []
            for x in range(width):
                for y in range(height):
                    for h in Heading:
                        for t in Heading:
                            p = ArticulatedPose(x, y, h, t)
                            try:
                                grid.check_articulated_pose(p)
                            except ValueError:
                                continue
                            states.append(p)
            if len(states) < 2:
                continue
            start, goal = rng.sample(states, 2)
            expected = dijkstra_articulated(width, height, obstacles, state_tuple(start), state_tuple(goal))
            result = plan_articulated(grid, start, goal)
            if expected is None:
                self.assertEqual(result.status, SearchStatus.NO_PATH, f"trial {trial}")
            else:
                self.assertEqual(result.status, SearchStatus.OPTIMAL, f"trial {trial}")
                self.assertEqual(result.plan.total_cost, expected, f"trial {trial}")
                assert_articulated_plan_valid(self, grid, start, goal, result.plan)
                self.assertLessEqual(articulated_heuristic(start, goal), expected)
            checked += 1
        self.assertGreaterEqual(checked, 40)


if __name__ == "__main__":
    unittest.main()
