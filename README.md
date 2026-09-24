# pose-grid-planner：带车体扫掠检查的姿态格路径规划

离线移动机器人工程用可复用库：在障碍格上为“占两格、四朝向”的机器人做
A* 姿态格规划，每个运动原语都带车体扫掠检查，输出动作序列与逐步扫掠证据；
并支持**执行旧路径到指定姿态后接收一次静态障碍更新、保留已执行前缀、
从当前位置全量重新规划剩余路径**。

## 运行环境

- Python **3.14.7**（仅用标准库，**无需安装任何额外依赖**，`requirements.txt` 为空说明）
- Linux 原生运行；核心、测试、演示全部离线，不需要网络/账号/硬件/Docker

## 快速开始

```bash
python -m unittest discover -s tests -v   # 全部测试（含独立 Dijkstra 对照与重规划随机对照）
python demo.py                            # 固定输入演示：正常规划、障碍更新后重规划、实际触发的失败边界
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
    replan,                                 # 障碍更新后的重规划
    ReplanningResult, ReplanningStatus,     # 重规划结果与状态枚举
    ReplayError, MAX_GRID_SIDE,
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

### 障碍更新后的重规划

执行旧路径到第 `executed` 个姿态后，接收**一次**静态障碍更新（只增不删），
保留已执行前缀，从当前位置在更新后的网格上**全量**重新搜索剩余路径：

```python
new_grid = grid.with_added_obstacles([(5, 2)])     # 返回新 Grid，旧网格不变
r = replan(grid, start, goal, previous,            # previous: 旧 Plan
           executed=2, added_obstacles=[(5, 2)],
           max_expansions=None)
```

- `Grid.with_added_obstacles(cells)`：不可变网格的“加障碍”操作，返回**新** `Grid`。
  每个坐标走与构造器相同的校验（带 `added_obstacles[i]` 索引）；批次内重复拒绝，
  但重复声明一个**本来就存在**的障碍是幂等空操作；任何一个非法则整批不生效。
- `replan(...) -> ReplanningResult`，字段：
  `status / plan / prefix / remaining / executed / current_pose / updated_grid /
  expansions / start / goal`。`previous` 会在旧网格上被独立复核（动作-证据链一致、
  逐步扫掠通过、终点等于 goal），不盲信调用方传入的 Plan。
- `status` 四种，失败与“未证明”严格区分：
  | status | 含义 |
  |---|---|
  | `OPTIMAL` | 找到当前位置到目标的最优剩余段；`plan` = 前缀 + 剩余段，代价为两段之和 |
  | `NO_PATH` | 开集耗尽，已证明当前位置不可达目标（目标姿态被新障碍压住时直接判定，不搜索） |
  | `BUDGET_EXHAUSTED` | 剩余段搜索预算耗尽，未找到也未证明无解（可加大预算重试） |
  | `PREFIX_BLOCKED` | 新障碍压住机器人**当前占用格**，当前状态不安全，拒绝继续 |
- **前缀语义**：已执行动作是旧世界里的历史事实，新障碍落在已离开的身后格
  不构成碰撞（仍返回 OPTIMAL）；只有当前占用的两格被压住才是 PREFIX_BLOCKED。
  `prefix` 的扫掠证据按旧网格原样保留；`remaining` 保证可在 `updated_grid` 上
  从 `current_pose` 重放。仅当更新未触及前缀任一步的扫掠格时，完整 `plan`
  才可在 `updated_grid` 上从 `start` 整体重放。
- 边界：`executed=0`（未出发，从起点重规划）、`executed=len(actions)` 且当前即
  目标（空剩余段，立即 OPTIMAL，0 次扩展）均已覆盖；`goal` 与 `start` 仍在旧
  网格上校验，非法姿态给可定位错误。
- 本迭代只做一次更新、全量重搜；**不**支持移动障碍、增量搜索、多轮承诺前缀。

## 错误语义

- 整数参数拒绝 `bool`（`TypeError`）；`float` 不隐式取整（`TypeError`）；
  `NaN`/`Infinity` 一律 `ValueError`。
- 非法输入抛带**参数名/索引**的 `TypeError`/`ValueError`（如 `obstacles[1] 与之前的障碍格重复`），
  不静默纠正、不静默去重。
- 库不持有全局/跨调用状态：`Grid` 不可变（障碍更新走 `with_added_obstacles`
  返回新对象），`plan`/`replan` 纯函数式，失败调用不会修改任何传入对象。

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
- **测试独立参考**：`tests/dijkstra_reference.py` 不导入被测包，按题面契约
  重新推导占用/扫掠规则做全状态 Dijkstra，与 A* 在确定性随机场景（含 20×20 上限）
  对照最优代价与有/无解判定；往返一致性（重放）仅作补充检查。
- **重规划复用既有扫掠与搜索**：前缀只做历史截取（证据原样保留），当前安全性
  用 `Grid.is_pose_free` 判定，剩余段直接调用同一个 `plan()`；旧 Plan 在旧网格上
  先独立复核，重规划结果的剩余代价同样由独立 Dijkstra 随机对照（见
  `tests/test_replanning.py`）。障碍合并是不可变 Grid 上的构造式新对象，
  从根上杜绝半更新状态。

## 已知限制

- 仅实现题面契约的四朝向、两格占用、四种原语；不是通用运动规划器，
  不宣称兼容任何行业标准（如完整 lattice planner / Hybrid A*）。
- 网格上限 20×20 为接口边界，超限构造即 `ValueError`。
- 代价固定为 10/14/8，不可配置（题面契约）；需要其他代价体系请改 `Action.cost` 表。
- 转向扫掠按“最小轴对齐 2×2”近似真实车体扫掠，不建模连续几何。
- 重规划只支持**一次静态障碍更新 + 全量重搜**：障碍只能新增不能撤销，
  不处理移动/临时障碍，不做增量搜索（无复用旧搜索树的性能保证），
  也不维护多轮承诺前缀；需要这些能力请在外层多次调用 `replan` 或另行扩展。

## 目录结构

```
pose_grid_planner/     # 库：model.py（数据结构+原语） planner.py（A*+重放）
                       #     replanner.py（障碍更新+前缀重规划） _validation.py
tests/                 # unittest 测试 + dijkstra_reference.py（独立参考，不调用被测核心）
demo.py                # 固定输入演示（真实计算，无 sleep/预录）
```
