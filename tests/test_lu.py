# test_lu.py
# 直接运行：python tests/test_lu.py
import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lu import lu_decompose, solve  # noqa: E402

try:
    from src.matrix import matmul
except Exception:  # pragma: no cover - matrix.py 不可用时用本地实现
    def matmul(A, B):
        n = len(A)
        k = len(A[0]) if n else 0
        m = len(B)
        p = len(B[0]) if m else 0
        if k != m:
            raise ValueError("维度不匹配")
        return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(p)]
                for i in range(n)]


_PASSED = 0
_FAILED = 0


def check(name, cond, detail=""):
    global _PASSED, _FAILED
    if cond:
        _PASSED += 1
        print("[PASS] %s %s" % (name, detail))
    else:
        _FAILED += 1
        print("[FAIL] %s %s" % (name, detail))


def to_lists(obj):
    """统一把 L/U 读成二维 list：有 to_list() 就用，没有就按二维 list 读。"""
    if hasattr(obj, "to_list"):
        return obj.to_list()
    return [list(row) for row in obj]


def approx(a, b, tol=1e-9):
    return len(a) == len(b) and all(abs(x - y) < tol for x, y in zip(a, b))


def approx_mat(A, B, tol=1e-9):
    if len(A) != len(B):
        return False
    return all(approx(p, q, tol) for p, q in zip(A, B))


# ======================================================================
# 用例 1：A = [[2,1,1],[4,3,3],[8,7,9]]，b = [7,19,49]，解为 [1,2,3]
# 第一列绝对值最大的是 8，必须发生行交换；L @ U 必须等于按 piv 重排后的 A
# ======================================================================
A1 = [[2, 1, 1], [4, 3, 3], [8, 7, 9]]
b1 = [7, 19, 49]
x1 = solve(A1, b1)
check("1. solve(A, b)", approx(x1, [1.0, 2.0, 3.0]), "-> x = %r" % (x1,))
check("1. solve 返回普通 list", type(x1) is list, "-> %s" % type(x1).__name__)

L1, U1, piv1 = lu_decompose(A1)
check("1. piv 发生行交换", piv1 != [0, 1, 2], "-> piv = %r" % (piv1,))
check("1. 第一列主元取到第 2 行", piv1[0] == 2, "-> piv[0] = %r" % piv1[0])

Ll, Ul = to_lists(L1), to_lists(U1)
check("1. L 对角线全为 1", all(abs(Ll[i][i] - 1.0) < 1e-12 for i in range(3)),
      "-> diag(L) = %r" % [Ll[i][i] for i in range(3)])
check("1. L 是下三角", all(Ll[i][j] == 0.0 for i in range(3) for j in range(i + 1, 3)))
check("1. U 是上三角", all(Ul[i][j] == 0.0 for i in range(3) for j in range(i)))

PA1 = [A1[piv1[i]] for i in range(3)]
LU1 = matmul(Ll, Ul)
check("1. L @ U == P A", approx_mat(LU1, PA1), "-> L@U = %r\n           P A = %r" % (LU1, PA1))

# ======================================================================
# 用例 2：任意阶方程组，残差很小，且原来的 A 不被修改
# ======================================================================
random.seed(20240607)
for n in (1, 2, 4, 7):
    A2 = [[random.uniform(-5.0, 5.0) for _ in range(n)] for _ in range(n)]
    x_true = [random.uniform(-3.0, 3.0) for _ in range(n)]
    b2 = [sum(A2[i][j] * x_true[j] for j in range(n)) for i in range(n)]
    A_orig = copy.deepcopy(A2)

    x2 = solve(A2, b2)
    residual = max(
        abs(sum(A2[i][j] * x2[j] for j in range(n)) - b2[i]) for i in range(n)
    )
    check("2. n=%d 残差 max|Ax-b| 很小" % n, residual < 1e-8, "-> %.3e" % residual)
    check("2. n=%d 原 A 未被修改" % n, A2 == A_orig)

# ======================================================================
# 用例 3：奇异矩阵必须抛 ValueError，信息含「奇异矩阵」
# ======================================================================
def _singular():
    solve([[1, 2], [2, 4]], [1, 2])


try:
    _singular()
except ValueError as e:
    check("3. 奇异矩阵抛 ValueError", "奇异矩阵" in str(e), "-> ValueError: %s" % e)
except Exception as e:  # noqa: BLE001
    check("3. 奇异矩阵抛 ValueError", False, "-> 抛出的是 %s: %s" % (type(e).__name__, e))
else:
    check("3. 奇异矩阵抛 ValueError", False, "-> 没有抛出异常")

try:
    solve([[1, 2], [2, 4]], [1])
except ValueError as e:
    check("3. len(b) 不等于阶数抛 ValueError", True, "-> ValueError: %s" % e)
else:
    check("3. len(b) 不等于阶数抛 ValueError", False)

try:
    lu_decompose([[1, 2, 3], [4, 5, 6]])
except ValueError as e:
    check("3. 非方阵抛 ValueError", True, "-> ValueError: %s" % e)
else:
    check("3. 非方阵抛 ValueError", False)

print("\n通过 %d 项，失败 %d 项" % (_PASSED, _FAILED))
sys.exit(1 if _FAILED else 0)
