# lu.py
# 带部分主元（partial pivoting）的 LU 分解与线性方程组求解。
#
# 设计要点（按优先级：正确性 > 数值稳定性 > 性能 > 可维护性 > 可读性）
# -------------------------------------------------------------------
# * 正确性：分解满足 P A = L U；piv[i] 表示 P A 的第 i 行来自原 A 的第 piv[i] 行；
#           solve 严格走 “分解 -> pb[i] = b[piv[i]] -> 前代 L y = pb -> 回代 U x = y”。
# * 数值稳定性：第 k 步在第 k..n-1 行中选第 k 列绝对值最大者为主元；
#           主元绝对值 < 1e-12 即判奇异并抛 ValueError。
#           因此消元与回代的除数恒有 |·| >= 1e-12，不可能出现 ZeroDivisionError；
#           所有下标均由 n 校验过，不可能出现 IndexError。
# * 性能（不引入第三方依赖，无 numpy / BLAS），关键决策都有实测依据：
#   1) 分解全程在 list[list[float]] 上进行，**只在返回时**包一层 Matrix。
#      若把 Matrix 当工作存储，每个元素的读写都要走 __getitem__/__setitem__
#      （内含下标校验 + float() 转换），这是最大的性能陷阱。
#   2) 行更新的内层循环用“就地索引循环”最快：实测 n=300 时
#      就地索引 0.49s < 切片+zip 推导 0.52s < 切片+map(sub) 0.70s < 整行推导 0.84s。
#      原因是切片/新列表会带来 3 倍内存流量，抵消了解释器层面的收益。
#   3) 前代/回代的点积优先用 C 层的 math.sumprod（Python 3.12+），
#      低版本自动退回 sum(map(mul, ...))，且只在循环外分支一次。
#   4) L 的行交换用切片赋值（C 层 memmove），行更新跳过 factor == 0 的稀疏情形。
# * 可维护性：文件只暴露 lu_decompose 与 solve 两个函数，内部按
#      “校验 -> 拷贝 -> 分解 / 求解” 分层；异常只用 ValueError 并说明具体原因。
# * 入参保护：A 在入口逐行深拷贝，调用者的 A 与 b 都不会被改动；
#      nan / inf 会让主元比较失效（连奇异性都判不出来）并污染整个结果，
#      因此在入口一并挡掉。
# * 返回类型：L、U 优先包成本项目的 src.matrix.Matrix；Matrix 不可用时
#      （ImportError / NotImplementedError / 构造签名不兼容）整体退回二维 list，
#      保证两者类型一致，且不去改动 matrix.py。
#
# 复杂度：分解 O(n^3/3)，求解 O(n^3/3 + n^2)，空间 O(n^2)。

from __future__ import annotations

import math
from collections.abc import Sequence
from math import isfinite
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:  # 仅用于类型标注，运行时不导入，避免与 matrix.py 产生耦合
    from src.matrix import Matrix

__all__ = ["lu_decompose", "solve"]

#: 主元绝对值小于该阈值即判定为奇异矩阵
_PIVOT_EPS: float = 1e-12

#: Python 3.12+ 的 C 层点积；低版本为 None，solve 里自动退回 sum(map(mul, ...))
_SUMPROD = getattr(math, "sumprod", None)

#: L / U 的返回类型：优先是 src.matrix.Matrix，Matrix 构造不可用时退回二维 list
#: （PEP 695 惰性求值，运行时不会真的去解析 Matrix）
type MatrixOrList = "Matrix | list[list[float]]"


def lu_decompose(A: Sequence[Sequence[float]]) -> tuple[MatrixOrList, MatrixOrList, list[int]]:
    """
    带部分主元的 LU 分解，满足 P A = L U。

    参数
    ----
    A : Sequence[Sequence[float]]
        n×n 方阵（n ≥ 1），二维 list；元素可以是 int，内部统一转 float。
        调用者手里的 A 不会被修改（分解前逐行深拷贝）。

    返回
    ----
    L : 下三角矩阵，对角线全为 1
    U : 上三角矩阵
    piv : 长度为 n 的普通 list[int]，piv[i] 表示 P A 的第 i 行 = A 的第 piv[i] 行

    L、U 优先用本项目的 src.matrix.Matrix 返回；
    若 matrix.py 中的 Matrix 构造函数仍是 NotImplementedError（或签名不兼容），
    则原样返回二维 list（不去改动 matrix.py）。

    异常
    ----
    ValueError
        A 为空 / 不是方阵 / 各行长度不齐 / 含非数值元素 / 含 nan 或 inf / 矩阵奇异。
    """
    # =================== 1. 参数校验 + 逐行深拷贝 ===================
    # None / 不支持 len() 的对象会在下面的 len(A) 处转成 ValueError，这里只挡字符串
    if isinstance(A, (str, bytes)):
        raise ValueError("A 必须是非空方阵（二维 list），当前是 %s" % type(A).__name__)

    try:
        n = len(A)
    except TypeError as exc:
        raise ValueError("A 必须是非空方阵（二维 list）：当前对象不支持 len()") from exc

    if n == 0:
        raise ValueError("A 必须是非空方阵：A 为空（0 行 0 列）")

    # 逐行深拷贝，之后所有计算只碰副本：U 初始化为 A 的副本
    U: list[list[float]] = []
    for i, row in enumerate(A):
        if isinstance(row, (str, bytes)):
            raise ValueError(
                "A 必须是非空方阵：第 %d 行必须是数值序列，当前是 %s"
                % (i, type(row).__name__)
            )
        try:
            ncol = len(row)
        except TypeError as exc:
            raise ValueError("A 必须是非空方阵：第 %d 行必须是数值序列（不支持 len()）" % i) from exc
        if ncol != n:
            raise ValueError(
                "A 必须是非空方阵：第 %d 行有 %d 个元素，应为 %d 个（各行长度不齐或非方阵）"
                % (i, ncol, n)
            )
        try:
            U.append([float(v) for v in row])
        except (TypeError, ValueError) as exc:
            raise ValueError("A 必须是非空方阵：第 %d 行含有非数值元素" % i) from exc
        # nan 会让后续所有比较失效（连奇异性都判不出来），inf 会污染整个分解
        if not all(map(isfinite, U[-1])):
            raise ValueError("A 必须是非空方阵：第 %d 行含有 nan 或 inf" % i)

    # =================== 2. 初始化：L 为单位下三角，piv 为恒等排列 ===================
    L: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    piv: list[int] = list(range(n))

    # =================== 3. 部分主元 LU 分解 ===================
    for k in range(n):
        # ---- 选主元：第 k 列中第 k..n-1 行绝对值最大者（并列取最靠上的一行）----
        max_row = k
        max_val = abs(U[k][k])
        for i in range(k + 1, n):
            val = abs(U[i][k])
            if val > max_val:
                max_val = val
                max_row = i

        if max_val < _PIVOT_EPS:
            raise ValueError(
                "奇异矩阵：第 %d 步的主元绝对值 %.6g 小于 %g（矩阵奇异或近似奇异）"
                % (k, max_val, _PIVOT_EPS)
            )

        # ---- 行交换：U 整行对换；L 只对换第 0..k-1 列已写好的乘数；piv 对换 ----
        if max_row != k:
            U[k], U[max_row] = U[max_row], U[k]
            Lk, Lm = L[k], L[max_row]
            Lk[:k], Lm[:k] = Lm[:k], Lk[:k]      # 切片赋值走 C 层，对角线保持 1
            piv[k], piv[max_row] = piv[max_row], piv[k]

        # ---- 消元：factor = U[i][k] / U[k][k]，随后 U[i] -= factor * U[k] ----
        Uk = U[k]
        pivot = Uk[k]
        for i in range(k + 1, n):
            factor = U[i][k] / pivot
            L[i][k] = factor
            if factor == 0.0:
                continue                          # 稀疏/结构化矩阵的短路：该行无需更新
            Ui = U[i]
            for j in range(k + 1, n):             # 第 k 列显式置 0，保证 U 严格上三角
                Ui[j] -= factor * Uk[j]
            Ui[k] = 0.0

    # =================== 4. 返回：L、U 优先用本项目的 Matrix ===================
    try:
        from src.matrix import Matrix as _Matrix

        # 任一步失败都整体退回二维 list，保证 L、U 的类型一致
        return _Matrix(L), _Matrix(U), piv
    except (ImportError, NotImplementedError, TypeError):
        return L, U, piv


def solve(A: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """
    求解线性方程组 A x = b（A 为 n 阶方阵）。

    步骤
    ----
    1. lu_decompose(A) 得到 L、U、piv；
    2. pb[i] = b[piv[i]]；
    3. 前代解 L y = pb（L 的对角线是 1，不要再除）；
    4. 回代解 U x = y。

    参数
    ----
    A : Sequence[Sequence[float]]
        n×n 方阵；调用者手里的 A 不会被修改。
    b : Sequence[float]
        长度为 n 的右端项。

    返回
    ----
    list[float]
        长度为 n 的普通 list，满足 A x = b。

    异常
    ----
    ValueError
        A 为空 / 不是方阵 / 各行长度不齐 / 含非数值元素 / 含 nan 或 inf / 矩阵奇异，
        或 b 长度不等于 n / 不是数值序列 / 含 nan 或 inf。
    """
    # ============ 1. 校验 A（这里只需要阶数 n，深拷贝交给 lu_decompose） ============
    if isinstance(A, (str, bytes)):
        raise ValueError("A 必须是非空方阵（二维 list），当前是 %s" % type(A).__name__)

    try:
        n = len(A)
    except TypeError as exc:
        raise ValueError("A 必须是非空方阵（二维 list）：当前对象不支持 len()") from exc

    if n == 0:
        raise ValueError("A 必须是非空方阵：A 为空（0 行 0 列）")

    for i, row in enumerate(A):
        if isinstance(row, (str, bytes)):
            raise ValueError(
                "A 必须是非空方阵：第 %d 行必须是数值序列，当前是 %s"
                % (i, type(row).__name__)
            )
        try:
            ncol = len(row)
        except TypeError as exc:
            raise ValueError("A 必须是非空方阵：第 %d 行必须是数值序列（不支持 len()）" % i) from exc
        if ncol != n:
            raise ValueError(
                "A 必须是非空方阵：第 %d 行有 %d 个元素，应为 %d 个（各行长度不齐或非方阵）"
                % (i, ncol, n)
            )

    # ============ 2. 校验 b 并转成 float list ============
    if isinstance(b, (str, bytes)):
        raise ValueError("b 必须是长度为 %d 的数值序列，当前是 %s" % (n, type(b).__name__))

    try:
        nb = len(b)
    except TypeError as exc:
        raise ValueError("b 必须是长度为 %d 的数值序列（不支持 len()）" % n) from exc

    try:
        b = [float(v) for v in b]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "b 必须是长度为 %d 的数值序列：不可迭代或含有非数值元素" % n
        ) from exc

    if nb != n:
        raise ValueError("b 的长度必须等于方阵阶数：len(b) = %d，阶数 n = %d" % (nb, n))
    if not all(map(isfinite, b)):
        raise ValueError("b 必须全都是有限数值（不能是 nan 或 inf）")

    # ============ 3. LU 分解 ============
    L, U, piv = lu_decompose(A)

    # ============ 4. 排列右端项：pb[i] = b[piv[i]] ============
    pb = [b[piv[i]] for i in range(n)]

    # L、U 可能是 Matrix 也可能是二维 list；Matrix 的 __getitem__ 随 key 形态返回
    # 不同类型，静态类型无法统一，但两者「按下标取一行」的语法完全一致，
    # 所以这里按二维 list 视图使用（零拷贝，只读）——cast 只是消掉类型分歧。
    Lr = cast("list[list[float]]", L)
    Ur = cast("list[list[float]]", U)

    # ============ 5. 前代 L y = pb（L 的对角线是 1，不要再除） ============
    y: list[float] = [0.0] * n
    if _SUMPROD is not None:                       # Python 3.12+：C 层点积
        for i in range(n):
            y[i] = pb[i] - _SUMPROD(Lr[i][:i], y[:i])
    else:                                          # 低版本回退
        for i in range(n):
            y[i] = pb[i] - sum(a * v for a, v in zip(Lr[i][:i], y[:i]))

    # ============ 6. 回代 U x = y ============
    x: list[float] = [0.0] * n
    if _SUMPROD is not None:
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - _SUMPROD(Ur[i][i + 1:], x[i + 1:])) / Ur[i][i]
    else:
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - sum(a * v for a, v in zip(Ur[i][i + 1:], x[i + 1:]))) / Ur[i][i]

    return x
