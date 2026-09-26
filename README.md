# pose-grid-planner：带车体扫掠检查的姿态格路径规划

离线移动机器人工程用可复用库：在障碍格上为“占两格、四朝向”的机器人做
A* 姿态格规划，每个运动原语都带车体扫掠检查，输出动作序列与逐步扫掠证据。
本次增量提供 `time_parameterize`：在几何路径已经验证的前提下，按速度、
切向加/减速度和横向加速度限制安排离散路径的时间曲线。

## 运行环境

- Python **3.14.7**（仅用标准库，**无需安装任何额外依赖**，`requirements.txt` 为空说明）
- Linux 原生运行；核心、测试、演示全部离线，不需要网络/账号/硬件/Docker

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
python -m unittest discover -s tests -v
python demo_iteration.py
```

`requirements.txt` 为空，venv 中不需要 `pip install`。原碰撞规划演示仍可用
`python demo.py` 运行；时间参数化演示 `demo_iteration.py` 现场计算，通常 1 秒内完成，
远低于 90 秒限制，不使用 sleep、网络或预存答案。

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
    time_parameterize,                      # 时间参数化
    TimeParameterization, SpeedConstraint,  # 时间结果与限制来源
    TimeParameterizationError,              # 不可行边界
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

```python
result = time_parameterize(
    path,                    # 已验证离散路径；本函数只使用节点数量，不重新做碰撞规划
    segment_lengths,         # 每段长度，长度必须为非负有限实数
    curvature_bounds,        # 每段曲率上界 kappa，0 表示直线段
    max_speed=10.0,
    acceleration=2.0,        # 最大切向加速度
    deceleration=2.0,        # 最大切向减速度（按正数输入）
    lateral_acceleration=2.0,
    start_speed=0.0,         # 调用方固定给定，不可被静默降低
    end_speed=0.0,           # 调用方固定给定，不可被静默抬升/降低
)
```

计算顺序是：

1. 节点初速上界取 `max_speed` 与所有相邻非零段的
   `sqrt(lateral_acceleration / curvature_bound)` 的最小值；
2. 正向传播平方速度约束 `v[i+1]^2 <= v[i]^2 + 2*a*s`；
3. 反向传播平方速度约束 `v[i]^2 <= v[i+1]^2 + 2*d*s`；
4. 校验固定起终速度在可用距离内可达，不可行则抛 `TimeParameterizationError`。

返回 `TimeParameterization`：

- `velocities`：每个原始节点的速度；
- `segment_durations`：每个原始段的最短通过时间；
- `segment_accelerations`：每个非零原始段使用的恒定切向加速度；
- `arrival_times`：每个原始节点的累计到达时间；
- `limiting_constraints`：每个节点最终速度的限制来源集合；并列生效时全部列出；
- `node_group_indices`：零长段合并后的节点组映射，原始节点数量和顺序不变。

零长段会把相邻节点合并为同一速度组，但返回数组仍按原始节点/段索引给出；
零长段自身时间为 0，其曲率不产生额外约束。段模型为**分段恒定切向加速度**，
段时间由 `t = 2s / (v0 + v1)` 精确定义；本模块不声称考虑 jerk。

## 错误语义

- 整数参数拒绝 `bool`（`TypeError`）；`float` 不隐式取整（`TypeError`）；
  `NaN`/`Infinity` 一律 `ValueError`。
- 非法输入抛带**参数名/索引**的 `TypeError`/`ValueError`（如 `obstacles[1] 与之前的障碍格重复`），
  不静默纠正、不静默去重。
- 库不持有全局/跨调用状态：`Grid` 不可变，`plan` 纯函数式，
  不存在“失败后留下部分变更”的批次操作。

## 误差口径

碰撞规划全部计算（代价、启发式、坐标）均为**精确整数运算**，无浮点结果，
因此无数值误差；路径代价可与独立参考逐点相等性比较（测试即如此断言）。

时间参数化使用 IEEE 754 双精度浮点。速度来自平方根，测试对手算值使用
`1e-10` 量级容差；独立参考按 2000 个子段复核距离和速度约束，并用分段恒定
加速度的闭式积分对照时间。约束复核另留 `1e-10` 的数值余量。该容差只用于
浮点表示误差，不用于把不可行的固定起终速度改成可行。

## 设计取舍

- **A\* + 一致性启发式 + closed 集（best_g）**：状态数最多 20×20×4=1600，
  无需双向搜索等复杂手段；启发式按题面契约只用两个分量的 max。
- **扫掠证据随路径输出**：`transition()` 是唯一的扫掠判定入口，规划与重放共用，
  证据即每步实际检查过的格子集合，可独立复核。
- **预算语义**：`max_expansions` 限制“弹出扩展”的状态数；预算触顶单独返回
  `BUDGET_EXHAUSTED`，调用方可据此加大预算重试，而不会误读为无解。
- **测试独立参考**：`tests/dijkstra_reference.py` 不导入被测包，按题面契约
  重新推导占用/扫掠规则做全状态 Dijkstra，与 A* 在确定性随机场景（含 20×20 上限）
  对照最优代价与有/无解判定；往返一致性（重放）仅作补充检查。
- **时间曲线独立复核**：直线和恒曲率弯道均与手算速度/时间对照；测试还按
  分段恒定加速度独立积分距离，并逐段检查最大速度、横向加速度、切向加/减速度。
  时间参数化不重新运行碰撞规划，也不加入实时控制器。

## 已知限制

- 仅实现题面契约的四朝向、两格占用、四种原语；不是通用运动规划器，
  不宣称兼容任何行业标准（如完整 lattice planner / Hybrid A*）。
- 网格上限 20×20 为接口边界，超限构造即 `ValueError`。
- 代价固定为 10/14/8，不可配置（题面契约）；需要其他代价体系请改 `Action.cost` 表。
- 转向扫掠按“最小轴对齐 2×2”近似真实车体扫掠，不建模连续几何。
- 时间参数化假定输入路径、段长和曲率上界已经由调用方验证；它不重算几何、
  不检查碰撞，也不处理实时控制跟踪。
- 时间模型只包含分段恒定切向加速度，不建模 jerk、转向速度变化、驱动器动态
  或曲率段内变化；曲率上界按该段两端节点施加横向限速。
- 起终速度是硬边界：超过曲率/最大速度上界，或距离不足以按规定加/减速度到达，
  都直接拒绝，不会自动改成“更接近”的速度。

## 目录结构

```
pose_grid_planner/     # 库：model.py、planner.py、time_parameterization.py、_validation.py
tests/                 # unittest 测试 + Dijkstra/数值积分独立参考
demo.py                # 原碰撞规划固定输入演示（真实计算，无 sleep/预录）
demo_iteration.py      # 时间参数化演示：直线、恒曲率弯道、实际拒绝边界
```
