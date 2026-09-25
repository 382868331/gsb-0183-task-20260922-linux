# pose-grid-planner：带车体扫掠检查的姿态格路径规划

离线移动机器人工程用可复用库：在障碍格上为“占两格、四朝向”的机器人做
A* 姿态格规划，每个运动原语都带车体扫掠检查，输出动作序列与逐步扫掠证据。
v1.1 增加**离散时间扩展**：有限时域 + 每时刻禁占格（移动障碍）+ WAIT 原语。

## 运行环境

- Python **3.14.7**（仅用标准库，**无需安装任何额外依赖**，`requirements.txt` 为空说明）
- Linux 原生运行；核心、测试、演示全部离线，不需要网络/账号/硬件/Docker

## 快速开始（可复现路径）

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 运行全部测试（静态 + 离散时间，含独立 Dijkstra 对照）
./.venv/bin/python demo.py                            # 固定输入演示：静态规划 + 离散时间规划 + 实际触发的失败示例
```

（仓库外任意 Python 3.14.7 解释器亦可：`python -m unittest discover -s tests -v`。）

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
  | `WAIT` 原地等待一步（仅 timed_* 接口） | `wait_cost`（正整数参数） | 当前**车体全部两格**必须全空 |
- “空” = 在网格界内且不是障碍；越界即不通过。离散时间接口另要求避开起止两时刻禁占格（见下）。
- A* 状态含朝向；目标必须**精确到达** (x, y, heading)，同格不同朝向不算到达。
- 启发式固定为 `max(曼哈顿距离×10, 最少 90° 转向次数×8)`（见 `heuristic()`），
  两个分量各自可采纳且一致，取 max 仍可采纳且一致，故 A* 返回**最优**解
  （加入 WAIT 与动态禁占后的证明见“离散时间扩展”一节）。

## 公开接口

```python
from pose_grid_planner import (
    Grid, Pose, Heading, Action,            # 数据结构
    plan, replay,                           # 规划 / 重放
    SearchResult, SearchStatus,             # 结果与状态枚举
    Plan, StepEvidence,                     # 路径与逐步证据
    transition, successors, heuristic,      # 单步原语 / 后继 / 启发式
    ReplayError, MAX_GRID_SIDE,
    # 离散时间扩展：
    TimedGrid, timed_plan, timed_replay, timed_transition, timed_successors,
    TimedPlan, TimedStepEvidence, TimedSearchResult, MAX_HORIZON,
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

## 离散时间扩展（timed_*）

在静态契约上叠加离散时间：

- `TimedGrid(grid, horizon, blocked=None)`：`horizon` 为时域上界（允许时刻 `0..horizon`，
  上限 `MAX_HORIZON = 4096`）；`blocked` 为 `{时刻: 禁占格集合}` 映射，缺省时刻为空集。
  构造期完成全部校验（时刻键越界、禁占格越界/重复都会报可定位错误）。
- **状态 = (位置, 朝向, 时刻)**：同一姿态在不同时刻是不同状态，绝不合并
  （测试 `test_same_pose_different_time_not_merged` 专门验证）。
- 每个原动作耗时一步；`WAIT` 也耗时一步，代价为正整数 `wait_cost`（`timed_plan` 参数，默认 1）。
- 时刻 `t` 出发的动作：扫掠格必须避开 **forbidden(t) ∪ forbidden(t+1)**
  （起止两时刻禁占格的并集），同时仍须避开静态障碍与边界；WAIT 检查车体全部两格。
- **到达即停**：弹出目标姿态（任意时刻 `≤ horizon`）即返回，不要求之后永久占住终点；
  到达时刻可以恰好等于 `horizon`。
- `timed_plan(tgrid, start, goal, *, wait_cost=1, max_expansions=None) -> TimedSearchResult`
  - `OPTIMAL`：`plan` 为 `TimedPlan`（`actions / total_cost / arrival_time / evidence`，
    每步证据含 `from_time / to_time`）。
  - `NO_PATH_IN_HORIZON`：时域内开集耗尽——**只证明该时域内无路，不代表静态几何无解**
    （静态可行性请用无时间的 `plan()` 另行判定；测试与 demo 都有二者对照）。
  - `BUDGET_EXHAUSTED`：预算耗尽，与上两者严格区分。
- `timed_replay(tgrid, start, actions, *, wait_cost=1)`：逐步重放校验（含 WAIT 与
  每时刻禁占检查），返回 `(最终姿态, 到达时刻, 逐步证据)`；失败抛带步序号的 `ReplayError`。

**启发式仍是可采纳下界（证明）**：设 `d` 为当前姿态到目标的曼哈顿距离、`k` 为朝向到
目标朝向的最少 90° 转向次数。任意合法动作序列中：① 每步动作最多把曼哈顿距离减少 1
（平移 ±1 格或不动），平移代价 ≥ 10（前进 10 / 后退 14），WAIT 与转向不减少距离且
代价为正，故总代价 ≥ 10·d；② 只有转向改变朝向且每次恰转 90°，故至少需要 `k` 次转向、
代价 ≥ 8·k。于是总代价 ≥ max(10·d, 8·k) = h。动态禁占格只删除转移、不降低任何路径
代价，WAIT 代价为正也不会降低下界；一致性同理（WAIT 前后姿态相同、h 不变、步代价为正）。
因此带 closed 集的 A* 弹出目标即最优——这与静态情形共用同一个 `heuristic()`。

**失败示例**（demo.py 场景 4 实际触发）：4×1 走廊，目标占用格 (3,0) 从时刻 2 起被封，
`timed_plan` 返回 `NO_PATH_IN_HORIZON`，而同一几何的静态 `plan` 返回 `OPTIMAL`
（代价 20）——时域结论不能当成静态几何结论。

## 错误语义

- 整数参数拒绝 `bool`（`TypeError`）；`float` 不隐式取整（`TypeError`）；
  `NaN`/`Infinity` 一律 `ValueError`。
- 非法输入抛带**参数名/索引**的 `TypeError`/`ValueError`（如 `obstacles[1] 与之前的障碍格重复`、
  `blocked[1][0] 的 x=9 越界`），不静默纠正、不静默去重。
- `wait_cost` 必须是正整数；`Action.WAIT` 的代价不是固定常量，访问 `Action.WAIT.cost`
  会给出明确 `ValueError`；无时间接口（`transition`/`replay`/`plan`）明确拒绝 WAIT。
- 库不持有全局/跨调用状态：`Grid`/`TimedGrid` 不可变，`plan`/`timed_plan` 纯函数式，
  不存在“失败后留下部分变更”的批次操作。

## 误差口径

全部计算（代价、启发式、坐标、时刻）均为**精确整数运算**，无浮点结果，
因此无数值误差；路径代价与到达时刻可与独立参考逐点相等性比较（测试即如此断言）。

## 设计取舍

- **A\* + 一致性启发式 + closed 集（best_g）**：静态状态数最多 20×20×4=1600；
  时间展开后状态数为 姿态数 × (horizon+1)，故设 `MAX_HORIZON = 4096` 作为接口边界
  （题面未给时域上限，此为库的设计取舍，超限构造即 `ValueError`）。
- **扫掠证据随路径输出**：`_sweep()` 是唯一的扫掠几何入口，静态 `transition` 与
  `timed_transition` 共用，证据即每步实际检查过的格子集合，可独立复核。
- **预算语义**：`max_expansions` 限制“弹出扩展”的状态数；预算触顶单独返回
  `BUDGET_EXHAUSTED`，调用方可据此加大预算重试，而不会误读为无解。
- **测试独立参考**：`tests/dijkstra_reference.py`（静态）与
  `tests/timed_dijkstra_reference.py`（时间展开）不导入被测包，按契约重新推导
  占用/扫掠/时间规则做全状态 Dijkstra，与 A* 在确定性随机场景（含 20×20 上限）
  对照最优代价、到达时刻与有/无解判定；往返一致性（重放）仅作补充检查。

## 已知限制

- 仅实现题面契约的四朝向、两格占用、四种运动原语 + WAIT；不是通用运动规划器，
  不宣称兼容任何行业标准（如完整 lattice planner / Hybrid A*）。
- 不建模连续速度/加速度：每个动作（含 WAIT）恰好耗时一步，时间完全离散。
- 禁占格按“每时刻格子集合”整体输入，不做静态地图的增量修补/流式更新。
- 网格上限 20×20、时域上限 `MAX_HORIZON = 4096` 为接口边界，超限构造即 `ValueError`。
- 运动代价固定为 10/14/8，不可配置（题面契约）；WAIT 代价由 `wait_cost` 参数给出。
- 转向扫掠按“最小轴对齐 2×2”近似真实车体扫掠，不建模连续几何。

## 目录结构

```
pose_grid_planner/     # 库：model.py（数据结构+原语） planner.py（静态 A*+重放）
                       #     timed.py（离散时间扩展） _validation.py
tests/                 # unittest 测试 + dijkstra_reference.py / timed_dijkstra_reference.py
                       # （独立参考，不调用被测核心）
demo.py                # 固定输入演示（真实计算，无 sleep/预录）
```
