# pose-grid-planner：带车体扫掠检查的姿态格路径规划（含离散时间）

离线移动机器人工程用可复用库：在障碍格上为“占两格、四朝向”的机器人做
A\* 姿态格规划，每个运动原语都带车体扫掠检查，输出动作序列与逐步扫掠证据。
**离散时间扩展**：状态加入时刻，输入有限时域内每时刻的禁占格集合，
新增 `WAIT`（原地等待一个时刻），动作扫掠避开起止两时刻禁占格的并集。

## 运行环境

- Python **3.14.7**（仅用标准库，**无需安装任何额外依赖**，`requirements.txt` 为空说明）
- Linux 原生运行；核心、测试、演示全部离线，不需要网络/账号/硬件/Docker

## 快速开始

```bash
python -m unittest discover -s tests -v   # 运行全部测试（静态 + 离散时间，含独立 Dijkstra 对照）
python demo.py                            # 固定输入演示：正常规划 + 拒绝边界 + 时序让行 + 时序失败示例
```

推荐用仓库自带解释器：`./.venv/bin/python -m unittest discover -s tests -v`、`./.venv/bin/python demo.py`。

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
  | `WAIT` 原地等待（**仅时序接口**） | 正整数，默认 1 | 车体当前占用的**全部两格**必须全空 |
- “空” = 在网格界内且不是障碍；越界即不通过。
- A* 状态含朝向；目标必须**精确到达** (x, y, heading)，同格不同朝向不算到达。
- 启发式固定为 `max(曼哈顿距离×10, 最少 90° 转向次数×8)`（见 `heuristic()`），
  两个分量各自可采纳且一致，取 max 仍可采纳且一致，故 A* 返回**最优**解。

### 离散时间扩展（时刻表）

- 搜索状态变为 **(x, y, heading, time)**：同一位置同一朝向、不同时刻是
  **不同状态，不合并**（`TimedState`）。
- 输入 `Timetable({时刻: [(x, y), ...]})`：给出有限时域内**每时刻的禁占格**；
  未列出的时刻无禁占格。时刻表只管临时禁占，静态障碍仍由 `Grid` 负责。
- 每个原语（含新增 `WAIT`）**耗时恰好 1 步**（时刻 +1）；`WAIT` 位置/朝向
  不变，代价为**正整数**（默认 1，可经 `wait_cost` 改）。
- 从时刻 `t` 执行到 `t+1` 的动作，其扫掠格必须全部避开
  **blocked[t] ∪ blocked[t+1]**（起止两时刻禁占格的并集）；`WAIT` 同样检查
  车体当前占用的**全部两格**，两格中任一格在 `t` 或 `t+1` 被禁占即不合法。
- 只搜索时刻 `0..horizon`：**到达目标即停止**，不要求之后永久占住终点。
- 最优准则：总代价最小；等代价时到达时刻最早（结构不同的路径可能同代价
  但步数不同，例如一次前+后与若干 `WAIT` 代价相同）。
- 三种失败严格区分：`OPTIMAL` / `NO_PATH_WITHIN_HORIZON`（**仅**表示该时域内
  无路，时域外可能有路，绝不能说成静态几何无路）/ `BUDGET_EXHAUSTED`。
- **启发式可采纳性（时序情形，附证明）**：任取时域内合法时序路径，删去全部
  `WAIT` 后的空间动作序列每步都通过了与静态规则相同的空间扫掠检查，故是一条
  合法静态路径，其代价 ≥ 静态最优代价 ≥ `h`；又因 `wait_cost` 为正，时序总
  代价不小于删等待后的空间代价。故静态 `h` 在时序问题上仍是下界（可采纳）；
  `WAIT` 边上 `h` 不变而代价为正，一致性也成立。A*（`best_g` 以
  (姿态, 时刻) 键控）仍返回**总代价最优**解。


## 公开接口

```python
from pose_grid_planner import (
    Grid, Pose, Heading, Action,            # 数据结构
    TimedState, Timetable,                  # 时序状态 / 每时刻禁占格
    plan, replay,                           # 静态规划 / 重放
    plan_timed, replay_timed,               # 离散时间规划 / 重放
    SearchResult, SearchStatus,             # 结果与状态枚举
    Plan, StepEvidence,                     # 路径与逐步证据
    transition, successors,                 # 单步原语 / 后继 / 启发式
    timed_transition, timed_successors, heuristic,
    ReplayError, MAX_GRID_SIDE, DEFAULT_WAIT_COST,
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
- `plan_timed(grid, timetable, start, goal, *, horizon, wait_cost=1, max_expansions=None)
  -> SearchResult`：离散时间规划。
  - `OPTIMAL`：`result.plan` 为总代价最优 `Plan`，`result.arrival_time` 为到达时刻；
    每步证据的 `time` 字段为该步起始时刻。
  - `NO_PATH_WITHIN_HORIZON`：时刻 `0..horizon` 内**证明无路**（不等于静态无解）。
  - `BUDGET_EXHAUSTED`：预算耗尽，既未找到也未证明时域内无路。
  - 入口校验：`start` 占用格在时刻 0 被禁占、时刻表格子越界、禁占时刻超出
    `horizon`（`blocked[horizon]` 合法、`blocked[horizon+1]` 报错不静默忽略）、
    `horizon<0`、`wait_cost` 非正整数等，均给可定位的 `TypeError`/`ValueError`。
- `replay_timed(grid, timetable, start, actions, *, wait_cost=1) -> (TimedState, evidence)`：
  从时刻 0 逐步重放，逐步复核空间扫掠与 `blocked[t] ∪ blocked[t+1]`；
  任一步不通过抛带步序号与时刻的 `ReplayError`。
- `timed_transition(grid, timetable, state, action)` /
  `timed_successors(grid, timetable, state, *, wait_cost=1)`：时序单步原语与后继。

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

- **A\* + 一致性启发式 + closed 集（best_g）**：静态状态数最多 20×20×4=1600；
  时序状态键为 (姿态, 时刻)，数量为 1600×(horizon+1)，有限时域内仍可直接穷举，
  无需双向搜索等复杂手段；启发式按题面契约只用两个分量的 max（时序下仍可采纳，证明见上）。
- **扫掠证据随路径输出**：`transition()` 是唯一的扫掠判定入口，规划与重放共用，
  证据即每步实际检查过的格子集合，可独立复核。
- **预算语义**：`max_expansions` 限制“弹出扩展”的状态数；预算触顶单独返回
  `BUDGET_EXHAUSTED`，调用方可据此加大预算重试，而不会误读为无解。
- **测试独立参考**：`tests/dijkstra_reference.py` 不导入被测包，按题面契约
  重新推导占用/扫掠规则做全状态 Dijkstra，与 A* 在确定性随机场景（含 20×20 上限）
  对照最优代价与有/无解判定；离散时间场景用 `timed_dijkstra_reference`
  对照**最优总代价与到达时刻**（含随机 `wait_cost ∈ {1,2,3}`）；
  往返一致性（重放）仅作补充检查。
- **WAIT 不属于静态接口**：静态 `successors()` 仍只枚举四个空间原语，
  旧契约与旧测试不受影响；WAIT 只在 `timed_successors()` 中出现。
- **时域只剪时间、不改几何结论**：状态时间维最多到 `horizon`；
  开集耗尽时只断言时域内无路，绝不把“等不到”报成“静态几何无路”。

## 可复现使用路径（离散时间）

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 52 个测试（静态 27 + 时序 25，含独立 Dijkstra 对照）
./.venv/bin/python demo.py                            # 场景 3/4 即时序演示（现场计算，秒级）
```

最小可复现片段（窄路临时占用，必须等待让行）：

```python
from pose_grid_planner import Grid, Pose, Heading, Timetable, plan_timed, SearchStatus

grid = Grid(5, 1)                                   # 1 格宽走廊
start, goal = Pose(0, 0, Heading.EAST), Pose(3, 0, Heading.EAST)
tt = Timetable({1: [(2, 0)], 2: [(2, 0)]})          # (2,0) 在 t=1,2 被临时占用
r = plan_timed(grid, tt, start, goal, horizon=10)
assert r.status is SearchStatus.OPTIMAL
print(r.plan.actions)        # (WAIT, WAIT, WAIT, FORWARD, FORWARD, FORWARD)
print(r.plan.total_cost, r.arrival_time)   # 33 6
```

### 失败示例（会实际发生，非预录）

终点从 `t=1` 起被封：同一几何静态规划有路（代价 20），但 `plan_timed`
只能报告**该时域内无路**——这是预期的区分，不是静态几何无解：

```python
grid = Grid(4, 1)
start, goal = Pose(0, 0, Heading.EAST), Pose(2, 0, Heading.EAST)
sealed = Timetable({t: [(2, 0), (3, 0)] for t in range(1, 8)})
plan_timed(grid, sealed, start, goal, horizon=7).status
# -> SearchStatus.NO_PATH_WITHIN_HORIZON（plan 为 None）
plan(grid, start, goal).status                      # -> SearchStatus.OPTIMAL
plan_timed(grid, Timetable({0: [(1, 0)]}), start, goal, horizon=4)
# -> ValueError: start 的占用格 (1, 0) 在时刻 0 被禁占
```

以上两段即 `demo.py` 的场景 3、4，运行 `./.venv/bin/python demo.py` 可见完整输出。

## 已知限制

- 实现题面契约的四朝向、两格占用、四个空间原语 + 一个 `WAIT`；
  不是通用运动规划器，不宣称兼容任何行业标准（如完整 lattice planner / Hybrid A*）。
- 网格上限 20×20 为接口边界，超限构造即 `ValueError`；时刻数受 `horizon` 限制，
  状态规模最多 20×20×4×(horizon+1)，超大时域应配合 `max_expansions` 使用。
- 空间动作代价固定为 10/14/8，不可配置（题面契约）；`WAIT` 代价默认 1、
  可为任意正整数，但不建模连续速度/真实加速度。
- 转向扫掠按“最小轴对齐 2×2”近似真实车体扫掠，不建模连续几何；
  不做静态地图的增量在线修补，时刻表在构造后不可变。
- 禁占格语义为“该时刻不可被扫掠/占用”，不模拟移动障碍的连续运动学。

## 目录结构

```
pose_grid_planner/     # 库：model.py（数据结构+空间/时序原语） planner.py（A*+时序A*+重放） _validation.py
tests/                 # unittest 测试（静态 27 + 时序 25）+ dijkstra_reference.py（独立参考，不调用被测核心）
demo.py                # 固定输入演示（真实计算，无 sleep/预录；含时序让行与失败示例）
```
