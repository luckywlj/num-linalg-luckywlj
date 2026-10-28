# matrix.py
# 纯 Python 的矩阵类：内部用 list[list[float]] 二维 list 存储，不依赖 numpy，
# 也不 import src.lu（避免循环依赖）。
#
# 结构约定
# --------
# * 文件中只有 Matrix 一个类：校验、乘法等逻辑全部作为（静态）方法收在类里，
#   没有任何模块级函数和模块级可执行语句。
# * 约定里的模块级 matmul(A, B) 是静态方法 Matrix._mul 的未绑定别名（见文件末尾），
#   所以 matmul(A, B) 与 A.matmul(B) 完全等价，A、B 也允许是普通二维序列。
#
# 设计要点
# --------
# * 拷贝语义：构造时把传入数据深拷贝进来；to_list() 返回深拷贝。
#   除这两处外不再复制——行操作（swap/scale/add_row_to）按约定全部原地修改。
# * 校验：非矩形 / 空矩阵 / 非数值元素一律 ValueError（信息含「矩形」）；
#   下标越界用 IndexError，下标类型不对用 TypeError（与内置 list 保持一致）。
# * 索引：M[i, j] 读/写元素；M[i] 返回第 i 行，因此 M[i][j] 也能读同一元素；
#   M[i] = row 把整行替换为新的 float 列表（长度必须等于列数，保持矩形）。
# * 行列式：在副本上做带部分主元的高斯消元，原矩阵不变；每交换一次行结果变号；
#   主元绝对值 < 1e-12 时返回 0.0；行列式 = 交换符号 × 消元后对角线乘积。
# * 复杂度：行操作 O(列数)，转置 O(行×列)，matmul O(n·k·m)，det O(n^3)；
#   除矩阵本身外不额外占用内存。

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, override

__all__ = ["Matrix", "matmul"]


class Matrix:
    """
    用二维 list 存储的矩阵，元素统一为 float。

        M = Matrix([[1, 2], [3, 4]])
        M.shape          -> (2, 2)
        M[0, 1] / M[0][1]-> 2.0
        M[0, 1] = 9.0    # 写元素
        M[0] = [7, 8]    # 整行替换
        M.T()            -> 新的转置矩阵（原矩阵不变）
        M.matmul(B)      -> 二维 list；M @ B -> 新的 Matrix
        M.det()          -> 行列式（仅方阵，原矩阵不变）
    """

    #: 行列式中主元绝对值小于该阈值即判定为 0
    _PIVOT_EPS: float = 1e-12

    # =================== 构造 / 拷贝 / 形状 ===================

    def __init__(self, data: Sequence[Sequence[float]]) -> None:
        """
        由二维序列构造矩阵。

        数据会被深拷贝成内部的 list[list[float]]：
        之后调用者再改原数据，矩阵内部不受影响。

        异常
        ----
        ValueError
            data 不是二维序列、是空矩阵、各行长度不同（非矩形）、
            或含有无法转成 float 的元素。
        """
        self._data: list[list[float]] = Matrix._as_rows(data, "data")

    @classmethod
    def _from_rows(cls, rows: list[list[float]]) -> "Matrix":
        """内部用：rows 已经是规整好的二维 float list，跳过校验直接包装。"""
        obj = cls.__new__(cls)
        obj._data = rows
        return obj

    @staticmethod
    def _as_rows(data: Any, name: str) -> list[list[float]]:
        """
        把二维序列规整成**新的** list[list[float]]（校验 + 深拷贝）。

        空矩阵和各行长度不同的数据都会抛 ValueError，信息里含「矩形」。
        """
        if isinstance(data, Matrix):
            return [list(row) for row in data._data]

        if isinstance(data, (str, bytes)):
            raise ValueError("%s 必须是矩形的二维序列，不能是字符串" % name)

        try:
            raw_rows = list(data)
        except TypeError as exc:
            raise ValueError(
                "%s 必须是矩形的二维序列（可迭代的行序列）" % name
            ) from exc

        if len(raw_rows) == 0:
            raise ValueError(
                "%s 是空矩阵：数据必须是矩形的二维序列，至少需要一行" % name
            )

        rows: list[list[float]] = []
        ncol: int | None = None
        for idx, raw_row in enumerate(raw_rows):
            if isinstance(raw_row, (str, bytes)):
                raise ValueError(
                    "%s 必须是矩形的二维序列：第 %d 行不能是字符串" % (name, idx)
                )
            try:
                row = list(raw_row)
            except TypeError as exc:
                raise ValueError(
                    "%s 必须是矩形的二维序列：第 %d 行必须是可迭代的数值序列"
                    % (name, idx)
                ) from exc

            if ncol is None:
                ncol = len(row)
                if ncol == 0:
                    raise ValueError(
                        "%s 必须是矩形的二维序列：第 %d 行没有任何元素（至少需要一列）"
                        % (name, idx)
                    )
            elif len(row) != ncol:
                raise ValueError(
                    "%s 必须是矩形的二维序列：第 %d 行有 %d 个元素，第 0 行有 %d 个"
                    % (name, idx, len(row), ncol)
                )

            try:
                rows.append([float(v) for v in row])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "%s 必须是矩形的二维序列：第 %d 行含有无法转成 float 的元素"
                    % (name, idx)
                ) from exc
        return rows

    @property
    def shape(self) -> tuple[int, int]:
        """返回 (行数, 列数)。"""
        nrow = len(self._data)
        ncol = len(self._data[0]) if nrow > 0 else 0
        return (nrow, ncol)

    @property
    def nrow(self) -> int:
        """行数。"""
        return len(self._data)

    @property
    def ncol(self) -> int:
        """列数。"""
        return len(self._data[0]) if self._data else 0

    def to_list(self) -> list[list[float]]:
        """返回二维 list 的深拷贝：改返回值不会影响矩阵内部。"""
        return [list(row) for row in self._data]

    def copy(self) -> "Matrix":
        """返回自身的副本。"""
        return Matrix(self.to_list())

    # =================== 索引访问 ===================

    @staticmethod
    def _fix_index(i: Any, length: int, what: str) -> int:
        """规整下标：支持负下标，越界抛 IndexError，类型不对抛 TypeError。"""
        if isinstance(i, bool) or not isinstance(i, int):
            raise TypeError("矩阵的%s下标必须是整数，当前是 %s" % (what, type(i).__name__))
        if i < 0:
            i += length
        if i < 0 or i >= length:
            raise IndexError("矩阵的%s下标 %d 越界（有效范围 0..%d）" % (what, i, length - 1))
        return i

    def __getitem__(self, key: Any) -> Any:
        """
        M[i, j] 读第 i 行第 j 列的元素；
        M[i] 返回第 i 行，因此 M[i][j] 也能读到同一个元素。
        """
        nrow, ncol = self.shape
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("只支持 M[i, j] 形式的二维下标，当前给了 %d 个" % len(key))
            i = Matrix._fix_index(key[0], nrow, "行")
            j = Matrix._fix_index(key[1], ncol, "列")
            return self._data[i][j]
        if isinstance(key, slice):
            return [self._data[i] for i in range(*key.indices(nrow))]
        i = Matrix._fix_index(key, nrow, "行")
        return self._data[i]

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        M[i, j] = v 写元素；
        M[i] = row 把第 i 行整体替换为新的 float 列表（长度必须等于列数）。
        """
        nrow, ncol = self.shape
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("只支持 M[i, j] 形式的二维下标，当前给了 %d 个" % len(key))
            i = Matrix._fix_index(key[0], nrow, "行")
            j = Matrix._fix_index(key[1], ncol, "列")
            try:
                v = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("写入的元素必须能转成 float，当前是 %r" % (value,)) from exc
            self._data[i][j] = v
            return

        i = Matrix._fix_index(key, nrow, "行")
        row = Matrix._as_rows([value], "赋值的一行")[0]
        if len(row) != ncol:
            raise ValueError(
                "赋值的一行有 %d 个元素，必须等于矩阵的列数 %d（保持矩形）"
                % (len(row), ncol)
            )
        self._data[i] = row

    # =================== 初等行变换（全部原地修改） ===================

    def swap_rows(self, i: int, j: int) -> None:
        """原地交换第 i 行和第 j 行。"""
        nrow = len(self._data)
        i = Matrix._fix_index(i, nrow, "行")
        j = Matrix._fix_index(j, nrow, "行")
        if i != j:
            self._data[i], self._data[j] = self._data[j], self._data[i]

    def scale_row(self, i: int, c: float) -> None:
        """原地把第 i 行的每个元素乘上 c。"""
        nrow = len(self._data)
        i = Matrix._fix_index(i, nrow, "行")
        c = float(c)
        row = self._data[i]
        for k in range(len(row)):
            row[k] *= c

    def add_row_to(self, i: int, j: int, c: float = 1.0) -> None:
        """
        原地把第 j 行的 c 倍加到第 i 行：第 i 行 += c * 第 j 行（方向不可反）。
        """
        nrow = len(self._data)
        i = Matrix._fix_index(i, nrow, "行")
        j = Matrix._fix_index(j, nrow, "行")
        c = float(c)
        ri, rj = self._data[i], self._data[j]
        for k in range(len(ri)):
            ri[k] += c * rj[k]

    # =================== 代数运算 ===================

    def T(self) -> "Matrix":
        """返回新的转置矩阵，原矩阵不变。"""
        nrow, ncol = self.shape
        rows = [[self._data[i][j] for i in range(nrow)] for j in range(ncol)]
        return Matrix._from_rows(rows)

    def matmul(self, other: Any) -> list[list[float]]:
        """
        矩阵乘法，返回二维 list（参与运算的两个矩阵都不被修改）。

        左矩阵的列数必须等于右矩阵的行数，否则抛 ValueError。
        """
        return Matrix._mul(self._data, other)

    def __matmul__(self, other: Any) -> "Matrix":
        """A @ B：调用 matmul，返回新的 Matrix。"""
        return Matrix(self.matmul(other))

    @staticmethod
    def _mul(a: Any, b: Any) -> list[list[float]]:
        """
        矩阵乘法的内部实现：a、b 可以是矩阵内部的二维 list，
        也可以是任意二维序列（文件末尾把它暴露成模块级 matmul）。
        两边都会先规整成副本，因此不会改动调用者的数据。
        """
        ra = Matrix._as_rows(a, "A")
        rb = Matrix._as_rows(b, "B")

        n = len(ra)
        k = len(ra[0])
        m = len(rb)
        p = len(rb[0])

        if k != m:
            raise ValueError(
                "矩阵乘法维度不匹配：左矩阵的列数 (%d) 必须等于右矩阵的行数 (%d)" % (k, m)
            )

        result: list[list[float]] = [[0.0] * p for _ in range(n)]
        for i in range(n):
            ai = ra[i]
            ri = result[i]
            for t in range(k):
                a_it = ai[t]
                if a_it == 0.0:      # 跳过 0，省掉一整轮乘加
                    continue
                bt = rb[t]
                for j in range(p):
                    ri[j] += a_it * bt[j]
        return result

    def det(self) -> float:
        """
        行列式：只对方阵有定义，否则抛 ValueError。

        在副本上用带部分主元的高斯消元计算，原矩阵不变：
          * 每发生一次行交换，结果变一次号；
          * 主元绝对值小于 1e-12 时行列式为 0.0；
          * 行列式 = 行交换符号 × 消元后对角线元素的乘积。
        """
        nrow, ncol = self.shape
        if nrow != ncol:
            raise ValueError(
                "det 只对 n 阶方阵有定义，当前 shape=(%d, %d)" % (nrow, ncol)
            )

        n = nrow
        a = [list(row) for row in self._data]     # 副本：消元不会改到原矩阵
        sign = 1.0

        for k in range(n):
            # ---- 选主元：第 k 列里绝对值最大者 ----
            piv = k
            maxv = abs(a[k][k])
            for i in range(k + 1, n):
                v = abs(a[i][k])
                if v > maxv:
                    maxv = v
                    piv = i

            if maxv < Matrix._PIVOT_EPS:
                return 0.0

            if piv != k:
                a[k], a[piv] = a[piv], a[k]
                sign = -sign

            ak = a[k]
            pivot = ak[k]
            for i in range(k + 1, n):
                ai = a[i]
                factor = ai[k] / pivot
                if factor == 0.0:
                    continue
                ai[k] = 0.0
                for j in range(k + 1, n):
                    ai[j] -= factor * ak[j]

        result = sign
        for i in range(n):
            result *= a[i][i]
        return result

    # =================== 显示与比较 ===================

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Matrix):
            return self._data == other._data
        try:
            return self._data == Matrix._as_rows(other, "other")
        except ValueError:
            return NotImplemented

    @override
    def __repr__(self) -> str:
        return "Matrix(%r)" % (self._data,)

    @override
    def __str__(self) -> str:
        return "\n".join("[" + ", ".join("%g" % v for v in row) + "]" for row in self._data)


# 约定里的模块级 matmul(A, B)：就是内部静态方法 Matrix._mul 的未绑定形式，
# 不额外定义模块级函数（保持「文件中仅一个类，无全局可执行逻辑」）。
# 因此 matmul(A, B) 与 A.matmul(B) 完全等价，A、B 也可以是普通二维序列，
# 例如 matmul([[1, 2]], [[1], [2], [3]]) 会抛维度不匹配的 ValueError。
matmul: Callable[..., list[list[float]]] = Matrix._mul
