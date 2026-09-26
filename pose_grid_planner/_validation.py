"""输入校验工具：错误必须可定位，禁止静默纠正。

规则（见 README“错误语义”）：
- 整数参数必须是真正的 int：bool 被拒绝（bool 是 int 子类）；
- 浮点若为 NaN/Infinity 给出明确 ValueError；其余 float 不隐式取整，按 TypeError 拒绝；
- 其他类型按 TypeError 拒绝。
所有错误信息都带参数名，保证可定位。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def require_finite_number(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    """要求物理量为有限实数；bool 不伪装成数字。"""
    if isinstance(value, bool):
        raise TypeError(f"{name} 必须是实数，不能是 bool: {value!r}")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是 int 或 float，收到 {type(value).__name__}: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 不能是 NaN 或 Infinity: {value!r}")
    if positive and number <= 0.0:
        raise ValueError(f"{name} 必须 > 0，收到 {value!r}")
    if nonnegative and number < 0.0:
        raise ValueError(f"{name} 必须 >= 0，收到 {value!r}")
    return number


def require_int(name: str, value: Any) -> int:
    """要求 value 为整数；拒绝 bool、NaN/Inf 浮点及其他非整数类型。"""
    if isinstance(value, bool):
        raise TypeError(f"{name} 必须是 int，不能是 bool: {value!r}")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} 不能是 NaN 或 Infinity: {value!r}")
        raise TypeError(f"{name} 必须是 int，收到 float（不做隐式取整）: {value!r}")
    if not isinstance(value, int):
        raise TypeError(f"{name} 必须是 int，收到 {type(value).__name__}: {value!r}")
    return value


def require_int_cell(name: str, cell: Any, *, width: int, height: int) -> tuple[int, int]:
    """要求 cell 是长度恰为 2 的整数坐标对，且落在 [0,width)×[0,height) 内。"""
    if isinstance(cell, (str, bytes)) or not isinstance(cell, Iterable):
        raise TypeError(f"{name} 必须是 (x, y) 坐标对，收到 {type(cell).__name__}: {cell!r}")
    items = list(cell)
    if len(items) != 2:
        raise ValueError(f"{name} 必须恰好包含 2 个整数坐标，收到 {len(items)} 个: {cell!r}")
    x = require_int(f"{name}[0]", items[0])
    y = require_int(f"{name}[1]", items[1])
    if not 0 <= x < width:
        raise ValueError(f"{name} 的 x={x} 越界，允许范围 [0, {width - 1}]")
    if not 0 <= y < height:
        raise ValueError(f"{name} 的 y={y} 越界，允许范围 [0, {height - 1}]")
    return x, y
