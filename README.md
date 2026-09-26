# pose-grid-planner：带车体扫掠检查的姿态格路径规划

离线移动机器人工程用可复用库：在障碍格上为“占两格、四朝向”的机器人做
A* 姿态格规划，每个运动原语都带车体扫掠检查，输出动作序列与逐步扫掠证据。

## 运行环境

- Python **3.14.7**（仅用标准库，**无需安装任何额外依赖**，`requirements.txt` 为空说明）
- Linux 原生运行；核心、测试、演示全部离线，不需要网络/账号/硬件/Docker

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
python --version                       # 应输出 Python 3.14.7；无需 pip install 第三方包
python -m unittest discover -s tests -v   # 运行全部测试（含独立 Dijkstra 对照）
python demo_iteration.py                  # 固定输入演示：时间参数化速度曲线
python demo.py                            # 可选：原碰撞规划演示
```

## 问题契约

- 网格不超过 **20×20**（`MAX_GRID_SIDE = 20`），坐标 `x` 向右、`y` 向下，均为 `0..n-1`。
- 机器人姿态 `Pose(x, y, heading)`：`heading ∈ {NORTH, EAST, SOUTH, WEST}`，
  占用**参考格 (x,y)** 与**沿 heading 的相邻一格**，共两格。
- 运动原语与代价：
  | 动作 | 代价 | 扫掠检查 |
  |---|---|---|
  | `FORWARD` 前进一格 | 10 | 起终占用格**并集**（3 格）必须全空 |
  | `BACKWARD` 后退一格 | 14 | 同上 |
  | `TURN_LEFT` / `TURN_RIGHT` 原地 90° | 8 | 包含起终全部占用格的**最小轴对齐 2×2 矩形**（4 格）必须全空 |
- “空” = 在网格界内且不是障碍；越界即不通过。
- A* 状态含朝向；目标必须**精确到达** (x, y, heading)，同格不同朝向不算到达。
- 启发式固定为 `max(曼哈顿距离×10, 最少 90° 转向次数×8)`（见 `heuristic()`），
  两个分量各自可采纳且一致，取 max 仍可采纳且一致，故 A* 返回**最优**解。

## 公开接口

```python
from pose_grid_planner import (
    Grid, Pose, Heading, Action,            # 数据结构
    plan, replay,                           # 规划 / 重放
    SearchResult, SearchStatus,             # 结果与状态枚举
    Plan, StepEvidence,                     # 路径与逐步证据
    transition, successors, heuristic,      # 单步原语 / 后继 / 启发式
    ReplayError, MAX_GRID_SIDE,
    time_parameterize,                      # 已验证路径的时间参数化
    TimedPath, TimedNode, TimedSegment,     # 时间参数化结果
    SpeedLimitReason, TimeParameterizationError,
)
```

- `Grid(width, height, obstacles=())` / `Grid.from_rows(["#.", ".#"])`
  构造不可变网格；构造期完成全部校验，失败不留下部分构造的对象。
- `plan(grid, start, goal, *, max_expansions=None) -> SearchResult`
  - `SearchStatus.OPTIMAL`：`result.plan` 为最优 `Plan`
    （`actions` 动作序列、`total_cost` 总代价、`evidence` 逐步 `StepEvidence`，
    每步含 `action / from_pose / to_pose / swept_cells / cost`）。
  - `SearchStatus.NO_PATH`：开集耗尽，**已证明无解**。
  - `SearchStatus.BUDGET_EXHAUSTED`：`max_expansions` 预算耗尽，
    既未找到也未证明无解——**与 NO_PATH 严格区分**，绝不静默返回次优路径。
- `replay(grid, start, actions) -> (final_pose, evidence)`：逐步重放校验，
  任一步扫掠不通过抛 `ReplayError`（带步序号，可定位）。
- `transition(grid, pose, action)`：单步执行，返回 `(下一姿态, 扫掠格)` 或 `None`。

### 时间参数化

`time_parameterize(path, segment_lengths, curvature_bounds, *, ...)` 的输入是
**已经验证过几何与碰撞可行性的离散路径**；本函数不会重新运行 A* 或碰撞检查，也不是实时控制器。

```python
timed = time_parameterize(
    path,                     # 任意 n 个已验证节点组成的序列
    segment_lengths,          # n-1 个非负段长
    curvature_bounds,         # n-1 个非负曲率上界 kappa_max；0 表示直线
    max_speed=3.0,
    max_tangential_acceleration=1.0,
    max_tangential_deceleration=1.0,
    max_lateral_acceleration=1.0,
    start_speed=0.0,
    end_speed=0.0,
)
```

- 每段先得到横向限制 `v_curve = sqrt(a_lateral / kappa_max)`，并与 `max_speed` 取交集；
  弯道段两端节点都受该段曲率限制。
- 再进行两次平方速度传播：前向 `v[j]² <= v[i]² + 2a_+L`，
  反向 `v[i]² <= v[j]² + 2a_-L`。因此不会只逐段套用最大速度，也不会遗漏前方弯道/终点的反向约束。
- 零长段的相邻节点被视为同一速度位置：约束会跨零长节点传播，但 `TimedPath.nodes`
  仍保留全部原节点与原索引；等价组放在 `zero_length_groups`，对应节点到达时间相同。
- 每段采用一个恒定切向加速度，时间由 `L = (v_start + v_end) * dt / 2` 得
  `dt = 2L / (v_start + v_end)`。**本模块不建模 jerk，也不做 jerk 限制。**
- `TimedNode.speed_limit_reason` 标明最终节点速度受 `max_speed`、曲率、起/终速度、
  前向加速传播或反向减速传播中哪条约束限制。
- 调用方给定的 `start_speed`/`end_speed` 若超过速度上限，或无法在给定距离/加减速度下满足，
  抛 `TimeParameterizationError`；不会悄悄降低边界速度。

## 错误语义

- 整数参数拒绝 `bool`（`TypeError`）；`float` 不隐式取整（`TypeError`）；
  `NaN`/`Infinity` 一律 `ValueError`。
- 非法输入抛带**参数名/索引**的 `TypeError`/`ValueError`（如 `obstacles[1] 与之前的障碍格重复`），
  不静默纠正、不静默去重。
- 库不持有全局/跨调用状态：`Grid` 不可变，`plan` 纯函数式，
  不存在“失败后留下部分变更”的批次操作。

## 误差口径

全部计算（代价、启发式、坐标）均为**精确整数运算**，无浮点结果，
因此无数值误差；路径代价可与独立参考逐点相等性比较（测试即如此断言）。

时间参数化使用浮点物理量。实现以闭式平方速度传播构造剖面，测试用独立中点数值积分重新积分
`v(t)=v0+a_t*t`，对照段长、节点到达时间、速度、切向加减速度与横向加速度；
实现仅用 `1e-12` 的相对容差吸收浮点舍入，不用宽容差放宽数学不可行边界。

## 设计取舍

- **A\* + 一致性启发式 + closed 集（best_g）**：状态数最多 20×20×4=1600，
  无需双向搜索等复杂手段；启发式按题面契约只用两个分量的 max。
- **扫掠证据随路径输出**：`transition()` 是唯一的扫掠判定入口，规划与重放共用，
  证据即每步实际检查过的格子集合，可独立复核。
- **预算语义**：`max_expansions` 限制“弹出扩展”的状态数；预算触顶单独返回
  `BUDGET_EXHAUSTED`，调用方可据此加大预算重试，而不会误读为无解。
- **时间参数化与碰撞规划解耦**：时间函数只消费已验证路径、段长与曲率上界；
  不重新搜索、不修改路径，也避免把离线剖面误认为闭环实时控制。
- **分段恒加速度**：每一段内部只允许一个切向加速度；平方速度前后传播保证节点速度同时满足
  最大速度、曲率横向加速度、加速、减速和起终边界。
- **测试独立参考**：`tests/dijkstra_reference.py` 不导入被测包，按题面契约
  重新推导占用/扫掠规则做全状态 Dijkstra，与 A* 在确定性随机场景（含 20×20 上限）
  对照最优代价与有/无解判定；往返一致性（重放）仅作补充检查。
  `tests/test_time_parameterization.py` 对手算直线梯形剖面与恒曲率弯道做闭式数值对照，
  并独立积分距离和采样检查全部运动约束。

## 已知限制

- 仅实现题面契约的四朝向、两格占用、四种原语；不是通用运动规划器，
  不宣称兼容任何行业标准（如完整 lattice planner / Hybrid A*）。
- 网格上限 20×20 为接口边界，超限构造即 `ValueError`。
- 代价固定为 10/14/8，不可配置（题面契约）；需要其他代价体系请改 `Action.cost` 表。
- 转向扫掠按“最小轴对齐 2×2”近似真实车体扫掠，不建模连续几何。
- 时间参数化要求调用方提供已验证离散路径、非负段长和每段曲率上界；库不重新计算几何或碰撞。
- 时间剖面是分段恒定切向加速度模型；不考虑 jerk、轮胎滑移、控制跟随误差、执行器延迟或障碍物动态变化。
- 曲率按“每段上界”保守处理，段内不推导连续曲率函数；`kappa=0` 表示直线。
- 正长段两端速度均为 0 无法在该模型下通过；零长段只表示同一位置的重复节点，不消耗时间。

## 目录结构

```
pose_grid_planner/     # 库：model.py（数据结构+原语） planner.py（A*+重放） _validation.py
tests/                 # unittest 测试 + Dijkstra/运动积分独立参考（不直接复用被测算法）
demo.py                # 固定输入演示（真实计算，无 sleep/预录）
demo_iteration.py      # 时间参数化增量演示（真实计算，无 sleep/预录）
```
