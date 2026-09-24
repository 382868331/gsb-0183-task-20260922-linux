# pose-grid-planner：带车体扫掠检查的姿态格路径规划

离线移动机器人工程用可复用库：在障碍格上为“占两格、四朝向”的机器人做
A* 姿态格规划，每个运动原语都带车体扫掠检查，输出动作序列与逐步扫掠证据。

**本次迭代新增**：旧路径执行到指定姿态后接收一次**静态障碍更新**（只新增），
保留已执行前缀，在更新后的网格上从当前位置**全量重新规划**剩余路径
（`replan`，见下文“障碍更新后重规划”）。

## 运行环境

- Python **3.14.7**（仅用标准库，**无需安装任何额外依赖**，`requirements.txt` 为空说明）
- Linux 原生运行；核心、测试、演示全部离线，不需要网络/账号/硬件/Docker

## 快速开始

```bash
python -m unittest discover -s tests -v   # 运行全部测试（含独立 Dijkstra 对照）
python demo.py                            # 固定输入演示：正常规划 + 实际触发的拒绝边界
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
    replan, ReplanResult,                   # 障碍更新后重规划
    SearchResult, SearchStatus,             # 结果与状态枚举
    Plan, StepEvidence,                     # 路径与逐步证据
    transition, successors, heuristic,      # 单步原语 / 后继 / 启发式
    ReplayError, MAX_GRID_SIDE,
)
```

- `Grid(width, height, obstacles=())` / `Grid.from_rows(["#.", ".#"])`
  构造不可变网格；构造期完成全部校验，失败不留下部分构造的对象。
  `grid.with_added_obstacles([(x, y), ...])` 返回一份**只新增障碍**的新网格
  （原网格不变）；越界、与已有障碍重复、新增列表内部重复均带索引报错。
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

## 障碍更新后重规划

```python
result = replan(grid, start, goal, old_actions, executed_actions, added_obstacles,
                *, max_expansions=None) -> ReplanResult
```

语义（本次迭代的核心能力）：

1. 在**旧网格**上从 `start` 完整重放 `old_actions`（复用 `replay` 的车体扫掠
   检查）——旧路径本身必须无碰撞，否则抛 `ReplayError`（带步序号）；
2. 取前 `executed_actions` 步作为**已执行前缀**（事实不变、原样保留），
   其末姿态即当前位置 `current_pose`；
3. 把 `added_obstacles` 并入得到不可变的 `updated_grid`（只允许**新增**）；
4. 当前占用格被新障碍压住属于自相矛盾的更新，抛 `ValueError`；
5. 从当前姿态到**原 goal** 做全量 A*。`result.status` 复用 `SearchStatus`：
   - `OPTIMAL`：`result.remainder` 为剩余段 `Plan`（仅剩余段，不含前缀），
     `result.prefix` 为已执行前缀；`result.total_cost` = 前缀代价 + 剩余段代价，
     `result.full_actions` 为两段动作拼接；
   - `NO_PATH`：目标被新障碍埋住（`expansions=0`）或开集耗尽证明剩余段不可达；
   - `BUDGET_EXHAUSTED`：剩余段搜索预算耗尽（与无解严格区分）。
6. `grid`、`old_actions` 均不被修改；任何校验失败直接抛错，不返回半更新结果。

边界：`executed_actions=0` 表示尚未执行（前缀为空、从 start 重搜）；
`executed_actions=len(old_actions)` 表示已到目标（剩余段为空、代价 0）；
新障碍落在**已走过**的格子上允许（只影响剩余搜索，不改写历史）。
注意前缀证据按旧网格成立、剩余段证据按更新后网格成立，因此 `full_actions`
只拼接动作、不跨网格拼接扫掠证据；两段可分别用 `replay` 核对。

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

## 设计取舍

- **A\* + 一致性启发式 + closed 集（best_g）**：状态数最多 20×20×4=1600，
  无需双向搜索等复杂手段；启发式按题面契约只用两个分量的 max。
- **扫掠证据随路径输出**：`transition()` 是唯一的扫掠判定入口，规划与重放共用，
  证据即每步实际检查过的格子集合，可独立复核。
- **预算语义**：`max_expansions` 限制“弹出扩展”的状态数；预算触顶单独返回
  `BUDGET_EXHAUSTED`，调用方可据此加大预算重试，而不会误读为无解。
- **重规划复用同一套扫掠判定**：`replan` 不另写碰撞规则——旧路径用 `replay`
  在旧网格上验证，剩余段用同一个 `plan`/`transition` 在新网格上全量搜索；
  更新只允许新增静态障碍，前缀是不可变事实，失败一律抛错而非部分应用。
- **测试独立参考**：`tests/dijkstra_reference.py` 不导入被测包，按题面契约
  重新推导占用/扫掠规则做全状态 Dijkstra，与 A* 在确定性随机场景（含 20×20 上限）
  对照最优代价与有/无解判定；往返一致性（重放）仅作补充检查。

## 已知限制

- 仅实现题面契约的四朝向、两格占用、四种原语；不是通用运动规划器，
  不宣称兼容任何行业标准（如完整 lattice planner / Hybrid A*）。
- 网格上限 20×20 为接口边界，超限构造即 `ValueError`。
- 代价固定为 10/14/8，不可配置（题面契约）；需要其他代价体系请改 `Action.cost` 表。
- 转向扫掠按“最小轴对齐 2×2”近似真实车体扫掠，不建模连续几何。
- `replan` 按本次迭代范围实现：**一次**静态障碍更新、**全量**重新搜索、
  单轮已执行前缀。不支持连续移动障碍、增量搜索的性能保证、以及多轮更新的
  前缀提交机制；需要时应再次调用 `replan`（以上一次结果的剩余段为旧路径）。

## 目录结构

```
pose_grid_planner/     # 库：model.py（数据结构+原语） planner.py（A*+重放+replan） _validation.py
tests/                 # unittest 测试（原验收 + test_replan.py 重规划用例）
                       # dijkstra_reference.py：独立参考，不调用被测核心
demo.py                # 固定输入演示（真实计算，无 sleep/预录）
```
