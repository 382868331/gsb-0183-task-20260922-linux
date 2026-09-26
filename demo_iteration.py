#!/usr/bin/env python3
"""现场计算时间参数化：直线、恒曲率弯道和一个不可行边界。"""

from pose_grid_planner import SpeedConstraint, TimeParameterizationError, time_parameterize


def show(title: str, lengths: tuple[float, ...], curvatures: tuple[float, ...], **kwargs: float) -> None:
    print(f"\n{title}")
    try:
        result = time_parameterize(
            tuple(range(len(lengths) + 1)),
            lengths,
            curvatures,
            **kwargs,
        )
    except TimeParameterizationError as exc:
        print(f"拒绝: {exc}")
        return

    print("节点速度:", [round(value, 6) for value in result.velocities])
    print("段时间:  ", [round(value, 6) for value in result.segment_durations])
    print("到达时间:", [round(value, 6) for value in result.arrival_times])
    names = {
        SpeedConstraint.MAX_SPEED: "最大速度",
        SpeedConstraint.LATERAL_ACCELERATION: "横向加速度",
        SpeedConstraint.TANGENTIAL_ACCELERATION: "切向加速",
        SpeedConstraint.TANGENTIAL_DECELERATION: "切向减速",
        SpeedConstraint.START_VELOCITY: "起点速度",
        SpeedConstraint.END_VELOCITY: "终点速度",
    }
    limits = [[names[item] for item in constraints] for constraints in result.limiting_constraints]
    print("限制来源:", limits)


def main() -> None:
    common = dict(
        max_speed=10.0,
        acceleration=2.0,
        deceleration=2.0,
        lateral_acceleration=2.0,
        start_speed=0.0,
        end_speed=0.0,
    )
    show("直线：加速—匀速—减速", (25.0, 100.0, 25.0), (0.0, 0.0, 0.0), **common)
    show("恒曲率弯道：横向加速度限速", (8.0, 12.0), (0.5, 0.5), **{**common, "start_speed": 2.0, "end_speed": 2.0})
    show(
        "异常边界：起点速度超过弯道许可",
        (10.0,),
        (1.0,),
        max_speed=10.0,
        acceleration=2.0,
        deceleration=2.0,
        lateral_acceleration=1.0,
        start_speed=2.0,
        end_speed=1.0,
    )


if __name__ == "__main__":
    main()
