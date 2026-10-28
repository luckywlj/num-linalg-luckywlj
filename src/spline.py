# spline.py
# 自然三次样条插值（natural cubic spline）。
#
# 设计要点
# --------
# * 依赖：只用标准库；三对角方程组按约定交给 src.lu.solve（单向依赖，无循环引用）。
# * 只暴露两个函数：natural_cubic_spline 与 eval_spline（无模块级辅助函数）。
# * 入参：xs / ys 在入口处拷贝成 float list，调用者手里的序列不会被改动。
# * 校验：等长、≥2 点、xs 严格递增、数值有限 —— 非法一律 ValueError。
# * 数值：h[i] 只算一次并缓存复用；h[i] 必须有限且 > 0（防除零 / 溢出）；
#         求值用 Horner 形式 a + t*(b + t*(c + t*d))，乘法次数更少、舍入误差更小。
# * 复杂度：建段与求值都是 O(n)（区间定位用 bisect，实际 O(log n)），
#         额外内存 O(n)；三对角方程组按约定交给 lu.solve，其自身为 O(n^3) / O(n^2)。
#
# 记号：h[i] = xs[i+1] - xs[i]，M[i] = S''(xs[i])，自然边界 M[0] = M[n] = 0。

from __future__ import annotations

import os
import sys
from bisect import bisect_right
from collections.abc import Sequence
from math import inf, isfinite

# 保证从任意工作目录 import 本项目时都能找到 src 包
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.lu import solve  # noqa: E402

__all__ = ["natural_cubic_spline", "eval_spline"]

#: eval_spline 判断“是否落在 [xs[0], xs[-1]] 内”时的相对容差
_EDGE_TOL: float = 1e-12


def natural_cubic_spline(
    xs: Sequence[float], ys: Sequence[float]
) -> list[tuple[float, float, float, float]]:
    """
    自然三次样条插值：构造通过全部节点的三次分段多项式。

    参数
    ----
    xs : Sequence[float]
        节点横坐标，必须严格递增，至少两个点。
    ys : Sequence[float]
        节点纵坐标，与 xs 等长。

    返回
    ----
    list[tuple[float, float, float, float]]
        段数 = len(xs) - 1。第 i 段对应区间 [xs[i], xs[i+1]]，为四元组 (a, b, c, d)；
        令 t = x - xs[i]，则 S(x) = a + b*t + c*t^2 + d*t^3。

    方程组
    ------
    记 n 为段数。内点 i = 1..n-1 的三对角方程：

        h[i-1]*M[i-1] + 2*(h[i-1] + h[i])*M[i] + h[i]*M[i+1]
            = 6 * ((ys[i+1] - ys[i]) / h[i] - (ys[i] - ys[i-1]) / h[i-1])

    未知量只有 M[1..n-1]（两端 M[0] = M[n] = 0），交给 src.lu.solve 求解。
    只有两个节点时没有内点，不解 0 阶方程组，直接由 M[0] = M[1] = 0 算系数
    （此时退化为直线）。

    系数
    ----
        a = ys[i]
        b = (ys[i+1] - ys[i]) / h[i] - h[i] * (2*M[i] + M[i+1]) / 6
        c = M[i] / 2
        d = (M[i+1] - M[i]) / (6 * h[i])

    异常
    ----
    ValueError
        xs / ys 不是数值序列、长度不同、少于两个点、xs 非严格递增、
        含 nan / inf，或相邻节点之差溢出（h 非有限 / 非正）。
    """
    # ============ 1. 入参校验 + 拷贝成 float（不动调用者的数据） ============
    if isinstance(xs, (str, bytes)) or isinstance(ys, (str, bytes)):
        raise ValueError("xs 与 ys 必须是数值序列，不能是字符串")

    try:
        xs = [float(v) for v in xs]
        ys = [float(v) for v in ys]
    except (TypeError, ValueError):
        raise ValueError("xs 与 ys 必须是数值序列") from None

    npt = len(xs)
    if npt != len(ys):
        raise ValueError(
            "xs 与 ys 长度必须相同：len(xs)=%d, len(ys)=%d" % (npt, len(ys))
        )
    if npt < 2:
        raise ValueError("至少需要两个节点，当前只有 %d 个" % npt)

    # 防 nan / inf：它们会让 h、M 以及最终系数全部变成 nan 或 inf
    if not all(map(isfinite, xs)) or not all(map(isfinite, ys)):
        raise ValueError("xs 与 ys 必须全都是有限数值（不能是 nan 或 inf）")

    # ============ 2. 缓存各段长度 h[i]（只算一次，后面反复复用） ============
    nseg = npt - 1
    h = [xs[i + 1] - xs[i] for i in range(nseg)]
    for i in range(nseg):
        if xs[i + 1] <= xs[i]:
            raise ValueError(
                "xs 必须严格递增：xs[%d]=%g <= xs[%d]=%g" % (i + 1, xs[i + 1], i, xs[i])
            )
    # 严格递增已保证 h > 0；这里再挡掉极大节点相减溢出成 inf 的情形
    if not all(0.0 < v < inf for v in h):
        raise ValueError("相邻节点之差必须有限且为正：h=%r" % (h,))

    # ============ 3. 三对角方程组求内点二阶导数 M[1..n-1] ============
    M = [0.0] * npt                       # 自然边界：M[0] = M[-1] = 0
    if nseg >= 2:
        m = nseg - 1                      # 内点个数
        A = [[0.0] * m for _ in range(m)]
        rhs = [0.0] * m
        for idx in range(m):
            i = idx + 1                   # 该行对应的内点下标
            row = A[idx]
            if idx > 0:
                row[idx - 1] = h[i - 1]
            row[idx] = 2.0 * (h[i - 1] + h[i])
            if idx + 1 < m:
                row[idx + 1] = h[i]
            rhs[idx] = 6.0 * ((ys[i + 1] - ys[i]) / h[i]
                              - (ys[i] - ys[i - 1]) / h[i - 1])
        M[1:nseg] = solve(A, rhs)         # M[1..nseg-1]，共 m 个

    # ============ 4. 逐段算出 (a, b, c, d) ============
    segments: list[tuple[float, float, float, float]] = []
    for i in range(nseg):
        hi = h[i]
        mi = M[i]
        mj = M[i + 1]
        a = ys[i]
        b = (ys[i + 1] - ys[i]) / hi - hi * (2.0 * mi + mj) / 6.0
        c = mi / 2.0
        d = (mj - mi) / (6.0 * hi)
        segments.append((a, b, c, d))
    return segments


def eval_spline(
    segments: Sequence[tuple[float, float, float, float]],
    xs: Sequence[float],
    x: float,
) -> float:
    """
    用 natural_cubic_spline 得到的分段系数在 x 处求值。

        S(x) = a + b*t + c*t^2 + d*t^3,  t = x - xs[i]

    其中 i 为 x 所在的区间下标（xs[i] <= x <= xs[i+1]），
    区间用 bisect 定位（O(log n)），求值用 Horner 形式。

    参数
    ----
    segments : 由 natural_cubic_spline 返回的系数表，长度应为 len(xs) - 1
    xs : 节点横坐标，必须严格递增，至少两个点
    x : 要求值的点，必须落在 [xs[0], xs[-1]] 内

    异常
    ----
    ValueError
        segments 为空、段数与节点数不匹配、xs 非法、x 越界或不是有限数值。
    """
    # ============ 1. 校验系数表（None / 不可测长都会被挡在这里） ============
    try:
        if len(segments) == 0:
            raise ValueError("segments 不能为空")
    except TypeError:
        raise ValueError("segments 必须是系数四元组的序列") from None

    # ============ 2. 校验节点并拷贝成 float ============
    if isinstance(xs, (str, bytes)):
        raise ValueError("xs 必须是数值序列，不能是字符串")
    try:
        xs = [float(v) for v in xs]
    except (TypeError, ValueError):
        raise ValueError("xs 必须是数值序列") from None

    if len(xs) < 2:
        raise ValueError("至少需要两个节点，当前只有 %d 个" % len(xs))
    if not all(map(isfinite, xs)):
        raise ValueError("xs 必须全都是有限数值（不能是 nan 或 inf）")
    for i in range(len(xs) - 1):
        if xs[i + 1] <= xs[i]:
            raise ValueError(
                "xs 必须严格递增：xs[%d]=%g <= xs[%d]=%g" % (i + 1, xs[i + 1], i, xs[i])
            )

    if len(segments) != len(xs) - 1:
        raise ValueError(
            "段数必须比节点数少 1：len(segments)=%d, len(xs)=%d"
            % (len(segments), len(xs))
        )

    # ============ 3. 校验求值点 ============
    try:
        x = float(x)
    except (TypeError, ValueError):
        raise ValueError("x 必须是数值") from None
    if not isfinite(x):
        raise ValueError("x 必须是有限数值，当前是 %r" % (x,))

    # 端点带一点相对容差，避免节点自身的浮点误差把端点判成越界
    lo, hi = xs[0], xs[-1]
    tol = _EDGE_TOL * max(1.0, abs(lo), abs(hi))
    if x < lo - tol or x > hi + tol:
        raise ValueError("插值点 x=%g 超出节点范围 [%g, %g]" % (x, lo, hi))
    if x < lo:
        x = lo
    elif x > hi:
        x = hi

    # ============ 4. 定位区间并求值（Horner） ============
    i = bisect_right(xs, x) - 1
    if i < 0:
        i = 0
    elif i > len(xs) - 2:
        i = len(xs) - 2

    a, b, c, d = segments[i]      # 段不是四元组时这里抛 ValueError
    t = x - xs[i]
    return a + t * (b + t * (c + t * d))
