# 带车体扫掠检查的姿态格路径规划（pose-grid-planner）

离线移动机器人的可复用路径规划库：机器人是**占两格**的车体（参考格 + 沿
朝向相邻一格），在 ≤20×20 的障碍网格上做含朝向状态的 A\* 搜索，并输出
**逐步扫掠证据**（每个动作实际要求为空的全部格子）。

- Python：**3.14.7**（仅标准库；3.10+ 亦可运行，未做更早版本兼容）
- 依赖：**无**。不联网、不需账号/密钥/数据库/中间件/Docker。
- 平台：Linux 原生。

## 运行命令

```bash
# 必要测试（含独立 Dijkstra 全状态参考对照）
python -m unittest discover -s tests -v

# 固定输入演示（约 0.1 秒，远低于 8 秒预算；真实计算，无 sleep/预录）
python demo.py
```

## 模型与代价

- 坐标：`(x, y)` 为 0 基整数格，x 向右、y 向上；`heading ∈ {0:东,1:北,2:西,3:南}`。
- 车体占用：参考格 `(x,y)` 与沿朝向相邻一格 `(x+dx, y+dy)`。
- 运动原语与代价：

| 动作 | 含义 | 代价 |
|---|---|---|
| `FORWARD` (0) | 沿朝向前进一格 | 10 |
| `BACKWARD` (3) | 沿朝向反方向后退一格 | 14 |
| `TURN_LEFT` (1) | 原地左转 90° | 8 |
| `TURN_RIGHT` (2) | 原地右转 90° | 8 |

- 扫掠检查（合法性 ⇔ 检查格全部在界内且非障碍）：
  - **平移**：起点与终点占用格的**并集**（3 格）；
  - **转向**：恰好包含“起点全部占用格 ∪ 终点全部占用格”的**最小轴对齐
    2×2 矩形**（4 格）。因此墙角、对角格障碍、贴边出界都会阻止原地转向，
    即使该格不在起终占用格内。

## 公开接口

```python
from pose_grid_planner import (
    GridWorld, Pose, Heading, Action, plan, heuristic,
    PlanStatus, StepEvidence, PlanningResult,
    PlannerError, InvalidParameterError, InvalidGridError, InvalidPoseError,
    BUDGET_DEFAULT,
)

world = GridWorld(width, height, obstacles={(x, y), ...})  # 1..20 × 1..20
result = plan(world,
              Pose(x, y, Heading.EAST),                     # 或 (x, y, 0)
              Pose(gx, gy, Heading.WEST))
if result.status is PlanStatus.SUCCESS:
    print(result.total_cost, result.actions)               # int；Action 元组
    for ev in result.evidence:                              # 逐步扫掠证据
        ev.action, ev.cost, ev.src_pose, ev.dst_pose,
        ev.swept_cells, ev.swept_kind                       # "union" / "2x2"
```

- `result.status`：`SUCCESS` / `NO_PATH`（完整搜索后**证明无解**）/
  `BUDGET_EXHAUSTED`（扩展预算耗尽，**不能**据此声称无解）。三者严格区分。
- 无路径不抛异常；`NO_PATH`/`BUDGET_EXHAUSTED` 时 `actions=()`、
  `total_cost=None`。起终姿态相同返回代价 0 的成功结果。
- `plan(..., max_expansions=N)`（默认 100000，对 20×20×4=1600 状态足够
  完成完整搜索）设置 A\* 节点扩展上限，耗尽即返回 `BUDGET_EXHAUSTED`。
- `heuristic(pose, goal)` 按题面固定为
  `max(曼哈顿距离 × 10, 最少转向次数 × 8)`，可采纳（平移以 10/格低估、
  转向以 8/次低估），测试中有逐边可采纳性校验。
- `GridWorld` 构造后不可变，可跨查询复用、并发只读使用。

### 输入上限

- `width, height ∈ [1, 20]`；障碍坐标必须在界内、不得重复。
- 注意：1×N（N<2）世界放不下两格车体，合法姿态集为空，此时无法给出
  起/终姿态（会抛 `InvalidPoseError`）——这是模型的固有边界。

### 错误语义（全部 fail-fast，不静默纠正）

- 继承关系：`PlannerError(ValueError)` →
  `InvalidParameterError` / `InvalidGridError` / `InvalidPoseError`。
- 所有整数参数**拒绝 bool**（`True` 不会被当作 1）；
  拒绝 float（含 `NaN`、`+Inf`、`-Inf` 及普通非整数 float）；
  heading 必须为 0..3；world 必须为 `GridWorld`。
- 障碍元素必须是长度 2 的整数坐标序列；字符串等非坐标集合直接拒绝。
- 起/终姿态越界或压障抛 `InvalidPoseError`，异常带 `where ∈ {"start","goal"}`
  与具体格子描述，便于定位。
- 纯函数 + 不可变世界：失败调用不产生任何部分变更或残留状态。

### 数值/误差口径

所有几何量、代价、启发式均为**精确整数运算**，无浮点参与，因此路径代价
与扫掠格集合**不存在舍入误差**，比较应使用精确相等，不使用容差。
（输入侧对 NaN/Infinity 的存在性判断仅用于拒绝，不产生任何浮点结果。）

## 设计取舍

1. **状态空间**：`(x, y, heading)`，至多 1600 个状态；A\* 用惰性邻接展开，
   不预建全图。`closed` + 惰性删除的优先队列；同代价路径用
   `(f, -g, 入队序号, 状态)` 保证结果确定、与字典哈希顺序无关。
2. **扫掠证据由模型层直接产出**：`GridWorld.apply` 同时负责合法性与
   `swept` 集合，规划器只转述，保证“打印的证据”就是“用于碰撞检查的
   同一组格子”，不存在两套判定。
3. **启发式严格按题面**，没有加入车身长度、障碍密度等额外项，宁可多扩展
   也不冒不可采纳风险。
4. **测试独立性**：`tests/dijkstra_reference.py` 是一份**不导入被测包**的
   独立 Dijkstra 全状态参考（独立重写占用格/扫掠/边代价），用于产出
   最优代价、可达性、合法边集合的期望值；往返重放只作补充校验。
5. **预算语义**：预算只约束“节点扩展次数”，与“开表为空 ⇒ 证明无解”
   分开报告，避免把超时误报成无解。

## 测试覆盖

`python -m unittest discover -s tests -v`（28 个用例）：

- 启发式取值与逐边可采纳性；
- 直走、同格起终、目标朝向必须转向、后退比绕行便宜；
- **窄道可直行不可转**（逐姿态断言两种转向都被 2×2 扫掠挡住）；
- **目标朝向**在封闭窄道中不可达 ⇒ 明确 `NO_PATH`；
- **后退必需**（盲端内前进/左转/右转全非法，仅后退合法）；
- **边界**：贴边转向扫掠出界被拒、2×2 对角格阻转但不阻平移；
- 预算耗尽 vs 证明无解分离、`max_expansions=0` 拒绝；
- 与独立 Dijkstra 参考在空世界、20 个随机世界、20×20 满规模世界上的
  最优代价/可达性/**逐状态合法边集合**一致；
- 全部成功路径证据独立重放：姿态连续、起终占用格 ⊆ 扫掠集、扫掠格全空；
- 输入校验：bool、NaN/Infinity、尺寸越界、障碍越界/重复/类型错、
  起终压障/越界（带 `where`）。

## 已知限制

- 只实现本题契约的离散子集：4 朝向、3 种原语、两格矩形车体、轴对齐
  2×2 转向扫掠；**不**宣称兼容任意角度、任意轮廓或连续位形空间规划。
- 不做时间维/动态障碍/多边车体会车；障碍在一次规划内静态。
- A\* 返回**一条**确定的最小代价路径；存在多条同代价路径时，选择由
  上述确定性堆规则决定，不枚举全部最优路径。
- 未内置地图文件解析或产品化 CLI——仅库 + `demo.py` 固定输入演示。

## 目录

```
pose_grid_planner/__init__.py   # 公开 API
pose_grid_planner/grid.py       # 网格/姿态/原语/扫掠合法性
pose_grid_planner/planner.py    # A*、启发式、结果与状态枚举
tests/dijkstra_reference.py     # 独立 Dijkstra 参考（不导入被测包）
tests/test_planner.py           # 必要测试
demo.py                         # 固定输入演示（正常结果 + 拒绝边界）
```
