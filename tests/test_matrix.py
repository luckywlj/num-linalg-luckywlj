# test_matrix.py
# 直接运行：python tests/test_matrix.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matrix import Matrix, matmul  # noqa: E402

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


def expect_value_error(name, fn):
    try:
        fn()
    except ValueError as e:
        check(name, "矩形" in str(e) or "方阵" in str(e) or "列数" in str(e),
              "-> ValueError: %s" % e)
    except Exception as e:  # noqa: BLE001
        check(name, False, "-> 抛出的是 %s: %s" % (type(e).__name__, e))
    else:
        check(name, False, "-> 没有抛出异常")


# 1. shape 与元素访问
M = Matrix([[1, 2], [3, 4]])
check("1. shape", M.shape == (2, 2), "-> %r" % (M.shape,))
check("1. M[0, 1]", M[0, 1] == 2, "-> %r" % M[0, 1])
check("1. M[0][1]", M[0][1] == 2, "-> %r" % M[0][1])

# 2. 初等行变换
M = Matrix([[1, 2], [3, 4]])
M.swap_rows(0, 1)
check("2. swap_rows(0, 1)", M.to_list() == [[3.0, 4.0], [1.0, 2.0]], "-> %r" % M.to_list())
M.scale_row(0, 2)
check("2. scale_row(0, 2)", M.to_list() == [[6.0, 8.0], [1.0, 2.0]], "-> %r" % M.to_list())
M.add_row_to(1, 0, 0.5)
check("2. add_row_to(1, 0, 0.5)", M.to_list() == [[6.0, 8.0], [4.0, 6.0]], "-> %r" % M.to_list())

# 3. 转置
M = Matrix([[1, 2], [3, 4]])
Mt = M.T()
check("3. T()", Mt.to_list() == [[1.0, 3.0], [2.0, 4.0]], "-> %r" % Mt.to_list())
check("3. T() 原矩阵不变", M.to_list() == [[1.0, 2.0], [3.0, 4.0]], "-> %r" % M.to_list())
check("3. T() 返回 Matrix", isinstance(Mt, Matrix))

# 4. 矩阵乘法
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])
check("4. matmul", matmul(A, B) == [[19.0, 22.0], [43.0, 50.0]], "-> %r" % matmul(A, B))
C = A @ B
check("4. @ 结果", C.to_list() == [[19.0, 22.0], [43.0, 50.0]], "-> %r" % C.to_list())
check("4. @ 返回 Matrix", isinstance(C, Matrix))
check("4. @ 不修改 A", A.to_list() == [[1.0, 2.0], [3.0, 4.0]], "-> %r" % A.to_list())
check("4. @ 不修改 B", B.to_list() == [[5.0, 6.0], [7.0, 8.0]], "-> %r" % B.to_list())

# 5. 非法情形
expect_value_error("5. Matrix 非矩形", lambda: Matrix([[1, 2, 3], [4, 5]]))
expect_value_error("5. matmul 维度不匹配", lambda: matmul([[1, 2]], [[1], [2], [3]]))
expect_value_error("5. 空矩阵", lambda: Matrix([]))
expect_value_error("5. det 非方阵", lambda: Matrix([[1, 2, 3], [4, 5, 6]]).det())

# 6. 约定相关的附加检查
src = [[1, 2], [3, 4]]
M = Matrix(src)
src[0][0] = 99.0
check("6. 构造时深拷贝", M[0, 0] == 1.0, "-> %r" % M[0, 0])

lst = M.to_list()
lst[0][0] = -7.0
check("6. to_list 返回副本", M[0, 0] == 1.0, "-> %r" % M[0, 0])

M = Matrix([[1, 2], [3, 4]])
M[1, 0] = 10
check("6. M[i, j] = v", M.to_list() == [[1.0, 2.0], [10.0, 4.0]], "-> %r" % M.to_list())
M[0] = [7, 8]
check("6. M[i] = row", M.to_list() == [[7.0, 8.0], [10.0, 4.0]], "-> %r" % M.to_list())

check("6. det 2x2", abs(Matrix([[1, 2], [3, 4]]).det() - (-2.0)) < 1e-9,
      "-> %r" % Matrix([[1, 2], [3, 4]]).det())
check("6. det 3x3", abs(Matrix([[6, 1, 1], [4, -2, 5], [2, 8, 7]]).det() - (-306.0)) < 1e-9,
      "-> %r" % Matrix([[6, 1, 1], [4, -2, 5], [2, 8, 7]]).det())
check("6. det 奇异矩阵", Matrix([[1, 2], [2, 4]]).det() == 0.0,
      "-> %r" % Matrix([[1, 2], [2, 4]]).det())
M = Matrix([[0, 1], [1, 0]])
before = M.to_list()
check("6. det 带行交换", abs(M.det() - (-1.0)) < 1e-9, "-> %r" % M.det())
check("6. det 不修改原矩阵", M.to_list() == before, "-> %r" % M.to_list())

print("\n通过 %d 项，失败 %d 项" % (_PASSED, _FAILED))
sys.exit(1 if _FAILED else 0)
