"""时间参数化测试：手算直线/弯道、零长映射、不可行边界和数值积分复核。"""

from __future__ import annotations

import math
import unittest

from pose_grid_planner import (
    SpeedConstraint as C,
    TimeParameterizationError,
    time_parameterize,
)


def integrate_reference(
    testcase: unittest.TestCase,
    lengths: tuple[float, ...],
    curvatures: tuple[float, ...],
    velocities: tuple[float, ...],
    *,
    max_speed: float,
    acceleration: float,
    deceleration: float,
    lateral_acceleration: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """不调用核心算法：按分段恒定加速度做独立距离积分并复核全部约束。"""
    durations = []
    distances = []
    for i, length in enumerate(lengths):
        if length == 0:
            testcase.assertEqual(velocities[i], velocities[i + 1])
            durations.append(0.0)
            distances.append(0.0)
            continue

        initial, final = velocities[i], velocities[i + 1]
        segment_acceleration = (final * final - initial * initial) / (2.0 * length)
        testcase.assertGreaterEqual(acceleration + 1e-12, segment_acceleration)
        testcase.assertGreaterEqual(deceleration + 1e-12, -segment_acceleration)

        traveled = 0.0
        steps = 2000
        step_length = length / steps
        for _ in range(steps):
            next_traveled = traveled + step_length
            left_speed = math.sqrt(max(0.0, initial * initial + 2.0 * segment_acceleration * traveled))
            right_speed = math.sqrt(
                max(0.0, initial * initial + 2.0 * segment_acceleration * next_traveled)
            )
            for speed in (left_speed, right_speed):
                testcase.assertLessEqual(speed, max_speed + 1e-10)
                testcase.assertLessEqual(speed * speed * curvatures[i], lateral_acceleration + 1e-10)
            traveled += step_length

        if abs(segment_acceleration) > 1e-12:
            elapsed = (final - initial) / segment_acceleration
        else:
            elapsed = length / initial
        testcase.assertAlmostEqual(math.sqrt(max(0.0, initial * initial + 2.0 * segment_acceleration * length)), final, places=10)
        durations.append(elapsed)
        distances.append(traveled)
    return tuple(durations), tuple(distances)


class StraightLineReferenceTest(unittest.TestCase):
    def test_three_phase_straight_line(self) -> None:
        lengths = (25.0, 100.0, 25.0)
        curvatures = (0.0, 0.0, 0.0)
        result = time_parameterize(
            tuple(range(4)),
            lengths,
            curvatures,
            max_speed=10.0,
            acceleration=2.0,
            deceleration=2.0,
            lateral_acceleration=2.0,
            start_speed=0.0,
            end_speed=0.0,
        )

        for got, expected in zip(result.velocities, (0.0, 10.0, 10.0, 0.0), strict=True):
            self.assertAlmostEqual(got, expected, places=10)
        for got, expected in zip(result.segment_durations, (5.0, 10.0, 5.0), strict=True):
            self.assertAlmostEqual(got, expected, places=10)
        self.assertEqual(result.arrival_times, (0.0, 5.0, 15.0, 20.0))
        self.assertIn(C.START_VELOCITY, result.limiting_constraints[0])
        self.assertEqual(result.limiting_constraints[1], frozenset({C.MAX_SPEED, C.TANGENTIAL_ACCELERATION}))
        self.assertEqual(result.limiting_constraints[2], frozenset({C.MAX_SPEED, C.TANGENTIAL_DECELERATION}))
        self.assertIn(C.END_VELOCITY, result.limiting_constraints[3])

        durations, distances = integrate_reference(
            self,
            lengths,
            curvatures,
            result.velocities,
            max_speed=10.0,
            acceleration=2.0,
            deceleration=2.0,
            lateral_acceleration=2.0,
        )
        for got, expected in zip(durations, result.segment_durations, strict=True):
            self.assertAlmostEqual(got, expected, places=6)
        for got, expected in zip(distances, lengths, strict=True):
            self.assertAlmostEqual(got, expected, places=8)


class ConstantCurveReferenceTest(unittest.TestCase):
    def test_constant_curvature_speed_ceiling(self) -> None:
        lengths = (8.0, 12.0)
        curvatures = (0.5, 0.5)
        result = time_parameterize(
            ("a", "b", "c"),
            lengths,
            curvatures,
            max_speed=10.0,
            acceleration=2.0,
            deceleration=2.0,
            lateral_acceleration=2.0,
            start_speed=2.0,
            end_speed=2.0,
        )

        expected_speed = math.sqrt(4.0)
        for velocity in result.velocities:
            self.assertAlmostEqual(velocity, expected_speed, places=10)
        for got, expected in zip(result.segment_durations, (4.0, 6.0), strict=True):
            self.assertAlmostEqual(got, expected, places=10)
        for constraints in result.limiting_constraints:
            self.assertIn(C.LATERAL_ACCELERATION, constraints)

        durations, distances = integrate_reference(
            self,
            lengths,
            curvatures,
            result.velocities,
            max_speed=10.0,
            acceleration=2.0,
            deceleration=2.0,
            lateral_acceleration=2.0,
        )
        for got, expected in zip(durations, result.segment_durations, strict=True):
            self.assertAlmostEqual(got, expected, places=6)
        for got, expected in zip(distances, lengths, strict=True):
            self.assertAlmostEqual(got, expected, places=8)

    def test_internal_curvature_cap_propagates_both_ways(self) -> None:
        lengths = (100.0, 100.0, 100.0, 100.0)
        curvatures = (0.0, 0.0, 1.0, 0.0)
        result = time_parameterize(
            tuple(range(5)),
            lengths,
            curvatures,
            max_speed=10.0,
            acceleration=2.0,
            deceleration=2.0,
            lateral_acceleration=1.0,
            start_speed=0.0,
            end_speed=0.0,
        )
        for got, expected in zip(result.velocities, (0.0, 10.0, 1.0, 1.0, 0.0), strict=True):
            self.assertAlmostEqual(got, expected, places=10)
        self.assertIn(C.LATERAL_ACCELERATION, result.limiting_constraints[2])
        self.assertIn(C.LATERAL_ACCELERATION, result.limiting_constraints[3])
        durations, distances = integrate_reference(
            self,
            lengths,
            curvatures,
            result.velocities,
            max_speed=10.0,
            acceleration=2.0,
            deceleration=2.0,
            lateral_acceleration=1.0,
        )
        for got, expected in zip(durations, result.segment_durations, strict=True):
            self.assertAlmostEqual(got, expected, places=8)
        for got, expected in zip(distances, lengths, strict=True):
            self.assertAlmostEqual(got, expected, places=8)


class ZeroLengthMappingTest(unittest.TestCase):
    def test_zero_segments_share_velocity_and_preserve_indices(self) -> None:
        result = time_parameterize(
            tuple(range(4)),
            (10.0, 0.0, 10.0),
            (0.0, 100.0, 0.0),
            max_speed=10.0,
            acceleration=5.0,
            deceleration=5.0,
            lateral_acceleration=10.0,
            start_speed=0.0,
            end_speed=0.0,
        )
        self.assertEqual(result.node_group_indices, (0, 1, 1, 2))
        self.assertEqual(len(result.velocities), 4)
        self.assertEqual(len(result.segment_durations), 3)
        self.assertEqual(result.velocities[1], result.velocities[2])
        self.assertEqual(result.segment_durations[1], 0.0)
        self.assertEqual(result.arrival_times[1], result.arrival_times[2])
        self.assertAlmostEqual(result.segment_durations[0], 2.0, places=10)
        self.assertAlmostEqual(result.segment_durations[2], 2.0, places=10)


class InfeasibilityTest(unittest.TestCase):
    def test_start_speed_above_curvature_ceiling_rejected(self) -> None:
        with self.assertRaisesRegex(TimeParameterizationError, "start_speed"):
            time_parameterize(
                (0, 1),
                (5.0,),
                (1.0,),
                max_speed=10.0,
                acceleration=2.0,
                deceleration=2.0,
                lateral_acceleration=1.0,
                start_speed=2.0,
                end_speed=1.0,
            )

    def test_end_speed_above_curvature_ceiling_rejected(self) -> None:
        with self.assertRaisesRegex(TimeParameterizationError, "end_speed"):
            time_parameterize(
                (0, 1),
                (5.0,),
                (1.0,),
                max_speed=10.0,
                acceleration=2.0,
                deceleration=2.0,
                lateral_acceleration=1.0,
                start_speed=1.0,
                end_speed=2.0,
            )

    def test_too_short_to_reach_fixed_end_speed(self) -> None:
        with self.assertRaisesRegex(TimeParameterizationError, "不足以"):
            time_parameterize(
                (0, 1),
                (10.0,),
                (0.0,),
                max_speed=10.0,
                acceleration=2.0,
                deceleration=2.0,
                lateral_acceleration=10.0,
                start_speed=0.0,
                end_speed=10.0,
            )

    def test_too_short_to_brake_from_fixed_start_speed(self) -> None:
        with self.assertRaisesRegex(TimeParameterizationError, "不足以"):
            time_parameterize(
                (0, 1),
                (10.0,),
                (0.0,),
                max_speed=10.0,
                acceleration=2.0,
                deceleration=2.0,
                lateral_acceleration=10.0,
                start_speed=10.0,
                end_speed=0.0,
            )


class InputValidationTest(unittest.TestCase):
    def test_lengths_and_curvatures_must_match_segments(self) -> None:
        with self.assertRaisesRegex(ValueError, "segment_lengths 长度必须是 2"):
            time_parameterize(
                tuple(range(3)),
                (1.0,),
                (1.0, 1.0),
                max_speed=1,
                acceleration=1,
                deceleration=1,
                lateral_acceleration=1,
                start_speed=0,
                end_speed=0,
            )

    def test_non_finite_and_bool_values_rejected(self) -> None:
        kwargs = dict(
            max_speed=1.0,
            acceleration=1.0,
            deceleration=1.0,
            lateral_acceleration=1.0,
            start_speed=0.0,
            end_speed=0.0,
        )
        with self.assertRaises(ValueError):
            time_parameterize((0, 1), (1.0,), (float("nan"),), **kwargs)
        with self.assertRaises(TypeError):
            time_parameterize((0, 1), (True,), (0.0,), **kwargs)
        with self.assertRaises(ValueError):
            time_parameterize((0, 1), (1.0,), (-0.1,), **kwargs)


if __name__ == "__main__":
    unittest.main()
