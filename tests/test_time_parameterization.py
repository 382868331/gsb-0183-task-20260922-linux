"""时间参数化测试：直线/恒曲率手算剖面 + 独立数值积分复核。"""

from __future__ import annotations

import math
import unittest

from pose_grid_planner import (
    SpeedLimitReason as Reason,
    TimeParameterizationError,
    time_parameterize,
)


def integrate_profile(result, *, max_speed, max_a, max_d, max_lateral):
    """不使用实现中的代数结论，按 v(t)=v0+a*t 做独立距离积分和约束采样。"""
    distance = 0.0
    for segment in result.segments:
        steps = 200
        dt = segment.duration / steps if steps else 0.0
        for j in range(steps):
            t_mid = (j + 0.5) * dt
            speed = segment.start_speed + segment.tangential_acceleration * t_mid
            assert speed >= -1e-12
            assert speed <= max_speed * (1.0 + 1e-10) + 1e-10
            if segment.tangential_acceleration >= 0.0:
                assert segment.tangential_acceleration <= max_a + 1e-10
            else:
                assert -segment.tangential_acceleration <= max_d + 1e-10
            if segment.curvature_bound > 0.0:
                assert speed**2 * segment.curvature_bound <= max_lateral + 1e-10
            distance += speed * dt
        if segment.duration == 0.0:
            assert segment.length == 0.0
            assert segment.tangential_acceleration == 0.0
    return distance


class TimeParameterizationReferenceTest(unittest.TestCase):
    def test_straight_line_trapezoid_hand_calculation(self) -> None:
        # 总长 20，a=d=1，v_max=2，起终静止：加速 2m、匀速 16m、减速 2m。
        result = time_parameterize(
            ("A", "M", "B"),
            (10.0, 10.0),
            (0.0, 0.0),
            max_speed=2.0,
            max_tangential_acceleration=1.0,
            max_tangential_deceleration=1.0,
            max_lateral_acceleration=4.0,
            start_speed=0.0,
            end_speed=0.0,
        )

        self.assertEqual([node.node for node in result.nodes], ["A", "M", "B"])
        self.assertEqual([node.speed for node in result.nodes], [0.0, 2.0, 0.0])
        self.assertEqual([seg.duration for seg in result.segments], [10.0, 10.0])
        self.assertEqual([node.arrival_time for node in result.nodes], [0.0, 10.0, 20.0])
        self.assertEqual(result.total_time, 20.0)
        self.assertEqual(
            [node.speed_limit_reason for node in result.nodes],
            [Reason.START_SPEED, Reason.MAX_SPEED, Reason.END_SPEED],
        )
        self.assertAlmostEqual(result.segments[0].tangential_acceleration, 0.2)
        self.assertAlmostEqual(result.segments[1].tangential_acceleration, -0.2)

        integrated_distance = integrate_profile(
            result,
            max_speed=2.0,
            max_a=1.0,
            max_d=1.0,
            max_lateral=4.0,
        )
        self.assertAlmostEqual(integrated_distance, 20.0, places=10)

    def test_constant_curvature_arc_hand_calculation(self) -> None:
        # k=1、横向加速度上限=1 => 弯道速度上限 sqrt(1/1)=1；20m 弯道恒速 20s。
        result = time_parameterize(
            ("curve-start", "curve-end"),
            (20.0,),
            (1.0,),
            max_speed=3.0,
            max_tangential_acceleration=2.0,
            max_tangential_deceleration=2.0,
            max_lateral_acceleration=1.0,
            start_speed=1.0,
            end_speed=1.0,
        )

        self.assertEqual([node.speed for node in result.nodes], [1.0, 1.0])
        self.assertEqual(result.segments[0].duration, 20.0)
        self.assertEqual(result.segments[0].tangential_acceleration, 0.0)
        self.assertTrue(
            all(node.speed_limit_reason is Reason.CURVATURE for node in result.nodes)
        )
        integrated_distance = integrate_profile(
            result,
            max_speed=3.0,
            max_a=2.0,
            max_d=2.0,
            max_lateral=1.0,
        )
        self.assertAlmostEqual(integrated_distance, 20.0, places=10)

    def test_backward_propagation_is_not_per_segment_max_speed(self) -> None:
        # 两个 8m 直段、d=1、末端静止：中间必须先被反向传成 sqrt(16)=4。
        result = time_parameterize(
            (0, 1, 2),
            (8.0, 8.0),
            (0.0, 0.0),
            max_speed=10.0,
            max_tangential_acceleration=1.0,
            max_tangential_deceleration=1.0,
            max_lateral_acceleration=100.0,
            start_speed=0.0,
            end_speed=0.0,
        )
        self.assertAlmostEqual(result.nodes[1].speed, 4.0)
        self.assertAlmostEqual(result.total_time, 4.0 + 4.0)
        self.assertIs(result.nodes[1].speed_limit_reason, Reason.FORWARD_ACCELERATION)

    def test_end_speed_must_be_reachable_from_earlier_speed(self) -> None:
        with self.assertRaisesRegex(TimeParameterizationError, "end_speed=10"):
            time_parameterize(
                (0, 1, 2),
                (4.0, 4.0),
                (1.0, 0.0),
                max_speed=3.0,
                max_tangential_acceleration=1.0,
                max_tangential_deceleration=1.0,
                max_lateral_acceleration=100.0,
                start_speed=1.0,
                end_speed=10.0,
            )


class ZeroLengthMappingTest(unittest.TestCase):
    def test_single_node_has_zero_duration_and_matching_boundary(self) -> None:
        result = time_parameterize(
            ("only",),
            (),
            (),
            max_speed=2.0,
            max_tangential_acceleration=1.0,
            max_tangential_deceleration=1.0,
            max_lateral_acceleration=4.0,
            start_speed=0.5,
            end_speed=0.5,
        )
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(result.segments, ())
        self.assertEqual(result.total_time, 0.0)
        self.assertEqual(result.nodes[0].speed, 0.5)

        with self.assertRaisesRegex(TimeParameterizationError, "单节点路径"):
            time_parameterize(
                ("only",),
                (),
                (),
                max_speed=2.0,
                max_tangential_acceleration=1.0,
                max_tangential_deceleration=1.0,
                max_lateral_acceleration=4.0,
                start_speed=0.0,
                end_speed=0.5,
            )

    def test_zero_segments_preserve_original_nodes_and_share_times(self) -> None:
        result = time_parameterize(
            ("A", "A-duplicate", "B", "B-duplicate"),
            (0.0, 4.0, 0.0),
            (0.0, 0.0, 0.0),
            max_speed=2.0,
            max_tangential_acceleration=1.0,
            max_tangential_deceleration=1.0,
            max_lateral_acceleration=4.0,
            start_speed=2.0,
            end_speed=2.0,
        )
        self.assertEqual(len(result.nodes), 4)
        self.assertEqual(len(result.segments), 3)
        self.assertEqual(result.zero_length_groups, ((0, 1), (2, 3)))
        self.assertEqual([node.speed for node in result.nodes], [2.0, 2.0, 2.0, 2.0])
        self.assertEqual([seg.duration for seg in result.segments], [0.0, 2.0, 0.0])
        self.assertEqual([node.arrival_time for node in result.nodes], [0.0, 0.0, 2.0, 2.0])

    def test_zero_length_forces_conflicting_boundary_speeds_to_be_equal(self) -> None:
        with self.assertRaisesRegex(TimeParameterizationError, "start_speed.*end_speed"):
            time_parameterize(
                ("A", "B"),
                (0.0,),
                (0.0,),
                max_speed=2.0,
                max_tangential_acceleration=1.0,
                max_tangential_deceleration=1.0,
                max_lateral_acceleration=4.0,
                start_speed=1.0,
                end_speed=2.0,
            )

    def test_curvature_cap_propagates_through_zero_length_group(self) -> None:
        # 末节点经零长段与弯道终点相连，也必须接受弯道速度限制。
        result = time_parameterize(
            ("A", "B", "C"),
            (4.0, 0.0),
            (1.0, 0.0),
            max_speed=3.0,
            max_tangential_acceleration=2.0,
            max_tangential_deceleration=2.0,
            max_lateral_acceleration=1.0,
            start_speed=1.0,
            end_speed=1.0,
        )
        self.assertEqual([node.speed for node in result.nodes], [1.0, 1.0, 1.0])
        self.assertTrue(all(node.speed_limit_reason is Reason.CURVATURE for node in result.nodes))

    def test_chained_zero_length_segments_keep_one_transitive_group(self) -> None:
        result = time_parameterize(
            ("A", "B", "C", "D"),
            (0.0, 0.0, 2.0),
            (0.0, 0.0, 0.0),
            max_speed=2.0,
            max_tangential_acceleration=1.0,
            max_tangential_deceleration=1.0,
            max_lateral_acceleration=4.0,
            start_speed=2.0,
            end_speed=2.0,
        )
        self.assertEqual(result.zero_length_groups, ((0, 1, 2),))
        self.assertEqual([node.arrival_time for node in result.nodes], [0.0, 0.0, 0.0, 1.0])


class TimingValidationTest(unittest.TestCase):
    def test_boundary_above_global_max_is_rejected(self) -> None:
        kwargs = dict(
            max_speed=2.0,
            max_tangential_acceleration=1.0,
            max_tangential_deceleration=1.0,
            max_lateral_acceleration=4.0,
        )
        with self.assertRaisesRegex(TimeParameterizationError, "start_speed"):
            time_parameterize((0, 1), (1.0,), (0.0,), start_speed=2.5, end_speed=0.0, **kwargs)
        with self.assertRaisesRegex(TimeParameterizationError, "end_speed"):
            time_parameterize((0, 1), (1.0,), (0.0,), start_speed=0.0, end_speed=2.5, **kwargs)

    def test_curvature_boundary_is_rejected_without_silent_reduction(self) -> None:
        with self.assertRaisesRegex(TimeParameterizationError, "start_speed.*曲率"):
            time_parameterize(
                (0, 1),
                (4.0,),
                (1.0,),
                max_speed=3.0,
                max_tangential_acceleration=2.0,
                max_tangential_deceleration=2.0,
                max_lateral_acceleration=1.0,
                start_speed=2.0,
                end_speed=1.0,
            )

    def test_sequence_length_and_numeric_validation(self) -> None:
        kwargs = dict(
            max_speed=2.0,
            max_tangential_acceleration=1.0,
            max_tangential_deceleration=1.0,
            max_lateral_acceleration=4.0,
            start_speed=0.0,
            end_speed=0.0,
        )
        with self.assertRaisesRegex(ValueError, "segment_lengths 长度"):
            time_parameterize((0, 1, 2), (1.0,), (0.0, 0.0), **kwargs)
        with self.assertRaisesRegex(ValueError, "curvature_bounds 长度"):
            time_parameterize((0, 1, 2), (1.0, 1.0), (0.0,), **kwargs)
        with self.assertRaisesRegex(ValueError, r"segment_lengths\[0\]"):
            time_parameterize((0, 1), (-1.0,), (0.0,), **kwargs)
        with self.assertRaisesRegex(ValueError, r"curvature_bounds\[0\]"):
            time_parameterize((0, 1), (1.0,), (-0.1,), **kwargs)
        with self.assertRaises(TypeError):
            time_parameterize((0, 1), (True,), (0.0,), **kwargs)
        with self.assertRaises(ValueError):
            time_parameterize((0, 1), (float("nan"),), (0.0,), **kwargs)
        with self.assertRaisesRegex(ValueError, "path 至少"):
            time_parameterize((), (), (), **kwargs)

    def test_limits_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_speed"):
            time_parameterize(
                (0, 1),
                (1.0,),
                (0.0,),
                max_speed=0.0,
                max_tangential_acceleration=1.0,
                max_tangential_deceleration=1.0,
                max_lateral_acceleration=4.0,
                start_speed=0.0,
                end_speed=0.0,
            )


if __name__ == "__main__":
    unittest.main()
