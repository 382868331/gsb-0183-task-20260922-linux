# pose-grid-planner：带拖车姿态与扫掠检查的姿态格规划

离线移动机器人工程用可复用库：在障碍格上为“占两格、四朝向”的牵引车做 A*
姿态格规划，并在同一离散模型中加入一节拖车。搜索状态始终包含牵引车与拖车朝向；
每个原语分别保留牵引车、拖车和连接杆的保守扫掠证据。原有单车接口与测试继续保留。

## 运行环境

- Python **3.14.7**（仅用标准库，**无需安装任何额外依赖**，`requirements.txt` 为空说明）
- Linux 原生运行；核心、测试、演示全部离线，不需要网络/账号/硬件/Docker

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
python --version    # 期望 3.14.7；仅标准库，requirements.txt 无需安装第三方包
python -m unittest discover -s tests -v   # 运行全部测试（含独立 Dijkstra 对照）
python demo.py                            # 旧单车接口演示
python demo_iteration.py                  # 新拖车场景 + 预算/无路边界，90 秒内完成
```

也可不激活虚拟环境，直接使用系统的 Python 3.14.7 运行上述命令；不使用 Docker、数据库、在线服务或真实凭据。

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

## 铰接拖车模型

- 新状态为 `ArticulatedPose(x, y, tractor_heading, trailer_heading)`。
- `(x,y)` 是牵引车后方连接点 hitch：牵引车占用 hitch 与沿牵引车朝向前一格；拖车占用 hitch 与沿拖车朝向反方向的车轴格。
- 最大铰接角为一个 90° 档位：`MAX_ARTICULATION_TURNS = 1`；超过该角度的转向直接不可用。
- 动作仍只有四个旧原语，代价固定为非负整数：前进 10、倒车 14、左右转 8。
- 前进一格：hitch 沿牵引车朝向移动，离散后的新拖车朝向等于牵引车朝向；倒车只在车列已直时可用；原地转向只改变牵引车朝向。
- 前进拉直经过角点时，拖车与连接杆使用保守 2×2 包围盒，防止切墙；证据分别给出 `tractor_swept_cells`、`trailer_swept_cells`、`hitch_bar_swept_cells`。
- `plan_articulated()` 搜索完整状态 `(x,y,tractor_heading,trailer_heading)`，不是先找单车路径再事后检查拖车碰撞。

```python
from pose_grid_planner import (
    ArticulatedPose, ArticulatedPlan, ArticulatedStepEvidence,
    ArticulatedSearchResult, MAX_ARTICULATION_TURNS,
    plan_articulated, replay_articulated, articulated_heuristic,
    articulated_transition, articulated_successors,
)

result = plan_articulated(grid, start, goal, max_expansions=None)
```

`ArticulatedStepEvidence.swept_cells` 返回三类扫掠格的去重并集；`replay_articulated()`
逐步复核铰接角、越界、障碍和连接杆扫掠，失败抛带步序号的 `ReplayError`。

## 错误语义

- 整数参数拒绝 `bool`（`TypeError`）；`float` 不隐式取整（`TypeError`）；
  `NaN`/`Infinity` 一律 `ValueError`。
- 非法输入抛带**参数名/索引**的 `TypeError`/`ValueError`（如 `obstacles[1] 与之前的障碍格重复`），
  不静默纠正、不静默去重。
- 库不持有全局/跨调用状态：`Grid` 不可变，`plan` 纯函数式，
  不存在“失败后留下部分变更”的批次操作。
- 拖车起终若为 180° 铰接、越界或占用障碍，抛 `ValueError`；搜索中越过铰接界限不是异常，而是该动作不产生后继。
- `max_expansions=0` 在扩展前返回 `BUDGET_EXHAUSTED`；完整搜索开集耗尽才是 `NO_PATH`。

## 误差口径

全部计算（代价、启发式、坐标）均为**精确整数运算**，无浮点结果，
因此无数值误差；路径代价可与独立参考逐点相等性比较（测试即如此断言）。

## 设计取舍

- **A\* + closed 集（best_g）**：单车状态数最多 20×20×4=1600，铰接车状态数最多 20×20×4×4=6400。
- 单车启发式按题面契约使用曼哈顿与最少转向代价的 max；拖车启发式只使用 hitch 曼哈顿平移下界，以保证在前进/倒车/转向边下一致。
- **扫掠证据随路径输出**：`transition()` 是唯一的扫掠判定入口，规划与重放共用，
  证据即每步实际检查过的格子集合，可独立复核。
- **预算语义**：`max_expansions` 限制“弹出扩展”的状态数；预算触顶单独返回
  `BUDGET_EXHAUSTED`，调用方可据此加大预算重试，而不会误读为无解。
- **独立参考**：`tests/dijkstra_reference.py` 与 `tests/articulated_dijkstra_reference.py`
  均不导入被测包。拖车测试覆盖窄转角单车可过/挂车无路、必须先拉直再倒车、逐步铰接角与三类扫掠、预算/无路分离，并用随机小网格做全状态 Dijkstra 对照。

## 已知限制

- 仅实现 cardinal 姿态格、两格车体、一节一格杆长拖车和四种原语；不是连续角度通用运动规划器，不宣称兼容完整 lattice planner / Hybrid A*。
- 当前离散拖车前进会在一格动作结束时拉直为牵引车朝向；中间连续角只用保守 2×2 格子扫掠包络，不输出连续轨迹。倒车的有限 cardinal 后继要求进入动作前车列已直。
- 网格上限 20×20 为接口边界，超限构造即 `ValueError`。
- 代价固定为 10/14/8，不可配置（题面契约）；需要其他代价体系请改 `Action.cost` 表。
- 转向和铰接扫掠按轴对齐 2×2/起终并集做保守格子近似，不建模真实车辆尺寸、轮胎轨迹、速度或多节挂车。

## 目录结构

```
pose_grid_planner/     # 库：model.py（数据结构+原语） planner.py（A*+重放） _validation.py
tests/                 # unittest + 单车/拖车两份独立 Dijkstra 参考
demo.py                # 旧单车固定输入演示（真实计算，无 sleep/预录）
demo_iteration.py      # 新拖车固定输入演示（真实计算，无 sleep/预录）
```
