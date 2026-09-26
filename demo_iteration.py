"""固定几何输入的时间参数化演示：现场计算，不重新运行碰撞规划。

运行：python demo_iteration.py
"""

from __future__ import annotations

import time

from pose_grid_planner import SpeedLimitReason, time_parameterize


def main() -> None:
    # 这里的 path/length/curvature 表示调用方已经完成几何与碰撞验证的离散路径。
    # 首尾零长重复节点用于演示：压缩传播速度，但保留原节点索引和到达时间映射。
    path = (
        "start-duplicate",
        "straight-start",
        "curve-entry",
        "curve-exit",
        "straight-end",
        "goal-duplicate",
    )
    lengths = (0.0, 9.0, 8.0, 9.0, 0.0)
    curvatures = (0.0, 0.0, 0.25, 0.0, 0.0)

    print("=== 时间参数化演示（路径已验证；不重新做碰撞搜索） ===")
    print(f"节点数={len(path)} 段数={len(lengths)} 长度={lengths} 曲率上界={curvatures}")
    print("限制：max_speed=3，a_+=a_-=1，横向加速度=1；起终速度=0")

    t0 = time.perf_counter()
    result = time_parameterize(
        path,
        lengths,
        curvatures,
        max_speed=3.0,
        max_tangential_acceleration=1.0,
        max_tangential_deceleration=1.0,
        max_lateral_acceleration=1.0,
        start_speed=0.0,
        end_speed=0.0,
    )
    elapsed = time.perf_counter() - t0

    print(f"计算用时={elapsed * 1000:.3f} ms；总最短到达时间={result.total_time:.6f} s")
    print(f"零长段合并组（原索引）={result.zero_length_groups}")
    print("\n节点结果：")
    for node in result.nodes:
        print(
            f"  #{node.index} {node.node:<17} v={node.speed:5.3f} m/s  "
            f"t={node.arrival_time:6.3f} s  limit={node.speed_limit_reason.value}"
        )

    print("\n分段结果（每段一个恒定切向加速度；不约束 jerk）：")
    for segment in result.segments:
        print(
            f"  seg {segment.index}: L={segment.length:4.1f} k<={segment.curvature_bound:4.2f}  "
            f"{segment.start_speed:4.2f}->{segment.end_speed:4.2f} m/s  "
            f"dt={segment.duration:6.3f} s  a_t={segment.tangential_acceleration: .3f} m/s^2"
        )

    assert result.nodes[2].speed_limit_reason is SpeedLimitReason.CURVATURE
    assert result.nodes[3].speed_limit_reason is SpeedLimitReason.CURVATURE
    print("\n弯道入口/出口速度由 sqrt(a_lateral/kappa)=sqrt(1/0.25)=2 m/s 限制。")


if __name__ == "__main__":
    main()
