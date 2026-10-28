# rk4.py
# 经典四阶龙格-库塔法（RK4）求解一阶常微分方程初值问题 y' = f(x, y)。
#
# 设计要点
# --------
# * 文件中只有 rk4_step 与 rk4_solve 两个函数（无模块级辅助函数）。
# * 纯 Python float：不依赖 numpy，也不 import src.lu（无循环依赖）。
# * 步长：h 必须为正的有限数值，h <= 0 / nan / inf 一律 ValueError（信息含“步长”）；
#   因此积分只沿 x 增大的方向，x_end < x0 同样抛 ValueError。
# * 终点精度：
#     - 区间长度接近 h 的整数倍时，用「区间长度 / 步数」作实际步长，
#       且每一步的 x 都由 x0 + i*步长 直接算出（而不是反复累加），避免累积误差；
#     - 区间不能被 h 整除时，前面走完整步长，最后一步缩短为剩余距离；
#     - 无论哪种情况，xs 的最后一项都强制写成 x_end，严格相等。
# * 复杂度 O(n)，除返回的两条 list 外不额外占用内存。
# * 函数内不硬编码任何具体方程、常数或步长：f、x0、y0、x_end、h 全部由参数传入。

from __future__ import annotations

from collections.abc import Callable
from math import isfinite

__all__ = ["rk4_step", "rk4_solve"]

#: 判断「区间长度是否差不多是步长的整数倍」时使用的相对容差
_INT_TOL: float = 1e-9


def rk4_step(f: Callable[[float, float], float], x: float, y: float, h: float) -> float:
    """
    从 (x, y) 出发用步长 h 走一个 RK4 步，返回该步之后的 y。

        k1 = f(x, y)
        k2 = f(x + h/2, y + h*k1/2)
        k3 = f(x + h/2, y + h*k2/2)
        k4 = f(x + h,   y + h*k3)
        y_next = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6

    参数
    ----
    f : Callable[[float, float], float]
        右端函数 f(x, y)。
    x, y : float
        当前点。
    h : float
        步长，必须为正的有限数值。

    异常
    ----
    ValueError
        h 不是数值、h <= 0，或 h 为 nan / inf。
    """
    # ---------- 步长校验（热路径，只做必要的检查） ----------
    try:
        h = float(h)
    except (TypeError, ValueError):
        raise ValueError("步长 h 必须是数值，当前是 %r" % (h,)) from None
    if not (h > 0.0 and isfinite(h)):
        raise ValueError("步长 h 必须是正的有限数值（h <= 0 时无法推进），当前 h=%r" % (h,))

    x = float(x)
    y = float(y)

    # ---------- 四阶龙格-库塔 ----------
    k1 = f(x, y)
    k2 = f(x + h / 2.0, y + h * k1 / 2.0)
    k3 = f(x + h / 2.0, y + h * k2 / 2.0)
    k4 = f(x + h, y + h * k3)
    return float(y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)


def rk4_solve(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    x_end: float,
    h: float,
) -> tuple[list[float], list[float]]:
    """
    从 (x0, y0) 积分到 x_end，步长约为 h。

    返回 (xs, ys)：两条等长的 list，xs[0] = x0、ys[0] = y0，
    之后每走一步追加一个点，且 xs 的最后一项严格等于 x_end。

    步长处理
    --------
    * 区间长度差不多是 h 的整数倍时：用「区间长度 / 步数」作为实际步长，
      并且第 i 个点的 x 由 x0 + i*步长 直接算出（不反复累加，避免 0.1 加十次
      不等于 1 的累积误差）；
    * 区间不能被 h 整除时：前面走完整步长，最后一步缩短为剩余距离；
    * 最后一项统一写成 x_end，保证严格相等。

    参数
    ----
    f : Callable[[float, float], float]
        右端函数 f(x, y)。
    x0, y0 : float
        初值点。
    x_end : float
        积分终点，必须不小于 x0（步长按约定为正，只沿 x 增大的方向积分）。
    h : float
        步长，必须为正的有限数值。

    异常
    ----
    ValueError
        h 不是数值、h <= 0、h 为 nan / inf，x0 / y0 / x_end 不是有限数值，
        或 x_end < x0。
    """
    # ---------- 步长校验 ----------
    try:
        h = float(h)
    except (TypeError, ValueError):
        raise ValueError("步长 h 必须是数值，当前是 %r" % (h,)) from None
    if not (h > 0.0 and isfinite(h)):
        raise ValueError("步长 h 必须是正的有限数值（h <= 0 时无法推进），当前 h=%r" % (h,))

    # ---------- 端点校验 ----------
    try:
        x0 = float(x0)
        y0 = float(y0)
        x_end = float(x_end)
    except (TypeError, ValueError):
        raise ValueError("x0、y0、x_end 必须是数值") from None
    if not (isfinite(x0) and isfinite(y0) and isfinite(x_end)):
        raise ValueError("x0、y0、x_end 必须都是有限数值")

    # ---------- x_end == x0：直接返回单点 ----------
    span = x_end - x0
    if span == 0.0:
        return ([x0], [y0])
    if span < 0.0:
        # 步长 h 按约定必须为正，因此只能沿 x 增大的方向积分
        raise ValueError(
            "x_end=%g 不能小于 x0=%g：步长 h 必须为正，无法反向积分" % (x_end, x0)
        )

    # ---------- 规划步长 ----------
    span_abs = span
    q = span_abs / h                                  # 区间长度折合多少个步长
    n_near = int(round(q))
    if n_near >= 1 and abs(q - n_near) <= _INT_TOL * max(1.0, q):
        # 整数步：用 区间长度 / 步数 作实际步长
        n_full = n_near
        step_mag = span_abs / n_full
        remain = 0.0
    else:
        # 非整数步：前面走完整步长，最后缩短一步
        n_full = int(q)                               # q > 0，等价于 floor
        step_mag = h
        remain = span_abs - n_full * h
        if remain <= _INT_TOL * max(1.0, span_abs):   # 残余小到可忽略就并入
            remain = 0.0

    # ---------- 逐步推进：x 由 x0 + i*步长 直接算出，避免累加误差 ----------
    xs: list[float] = [x0]
    ys: list[float] = [y0]
    x = x0
    y = y0
    for i in range(1, n_full + 1):
        y = rk4_step(f, x, y, step_mag)
        x = x0 + i * step_mag
        xs.append(x)
        ys.append(y)

    if remain > 0.0:
        y = rk4_step(f, x, y, remain)
        x = x_end
        xs.append(x)
        ys.append(y)

    xs[-1] = x_end                                    # 终点严格等于 x_end
    return (xs, ys)
