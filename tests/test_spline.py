# test_spline.py
# 直接运行：python tests/test_spline.py
import copy
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spline import eval_spline, natural_cubic_spline  # noqa: E402

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
        check(name, True, "-> ValueError: %s" % e)
    except Exception as e:  # noqa: BLE001
        check(name, False, "-> 抛出的是 %s: %s" % (type(e).__name__, e))
    else:
        check(name, False, "-> 没有抛出异常")


def deriv(seg, t):
    """一阶导数：b + 2*c*t + 3*d*t^2"""
    a, b, c, d = seg
    return b + 2.0 * c * t + 3.0 * d * t * t


def deriv2(seg, t):
    """二阶导数：2*c + 6*d*t"""
    a, b, c, d = seg
    return 2.0 * c + 6.0 * d * t


def eval_seg(seg, t):
    a, b, c, d = seg
    return a + b * t + c * t * t + d * t * t * t


# ======================================================================
# 用例 1：节点 (0,0)、(1,1)、(2,0)
# ======================================================================
xs1 = [0, 1, 2]
ys1 = [0, 1, 0]
seg1 = natural_cubic_spline(xs1, ys1)

check("1. 段数 = 节点数 - 1", len(seg1) == len(xs1) - 1, "-> %d" % len(seg1))
check("1. 每段是四元组 (a,b,c,d)",
      all(isinstance(s, tuple) and len(s) == 4 for s in seg1),
      "-> %r" % (seg1,))
check("1. 节点值 0", eval_spline(seg1, xs1, 0.0) == 0.0, "-> %r" % eval_spline(seg1, xs1, 0.0))
check("1. 节点值 1", eval_spline(seg1, xs1, 1.0) == 1.0, "-> %r" % eval_spline(seg1, xs1, 1.0))
check("1. 节点值 0", eval_spline(seg1, xs1, 2.0) == 0.0, "-> %r" % eval_spline(seg1, xs1, 2.0))

v05 = eval_spline(seg1, xs1, 0.5)
v15 = eval_spline(seg1, xs1, 1.5)
check("1. x=0.5 等于 0.6875", abs(v05 - 0.6875) < 1e-12, "-> %.15f" % v05)
check("1. x=1.5 等于 0.6875", abs(v15 - 0.6875) < 1e-12, "-> %.15f" % v15)
check("1. 对称性（关于 x=1）", abs(v05 - v15) < 1e-12)

# 手工核对第一段系数：a=0, b=1.5, c=0, d=-0.5
check("1. 第一段系数 (0, 1.5, 0, -0.5)",
      all(abs(seg1[0][k] - v) < 1e-12 for k, v in enumerate((0.0, 1.5, 0.0, -0.5))),
      "-> %r" % (seg1[0],))

# ======================================================================
# 用例 2：间距不相等的节点
# ======================================================================
xs2 = [0.0, 0.4, 1.1, 2.0, 3.2]
ys2 = [0.0, 1.2, -0.3, 0.5, 0.0]
seg2 = natural_cubic_spline(xs2, ys2)

check("2. 段数 = 节点数 - 1", len(seg2) == len(xs2) - 1, "-> %d" % len(seg2))
max_node_err = max(abs(eval_spline(seg2, xs2, xs2[i]) - ys2[i]) for i in range(len(xs2)))
check("2. 每个节点都取到给定的 y", max_node_err < 1e-12, "-> max err = %.3e" % max_node_err)

# ======================================================================
# 用例 3：内节点处一阶、二阶导数连续
# ======================================================================
max_gap1 = 0.0
max_gap2 = 0.0
for X, Y, S in ((xs1, ys1, seg1), (xs2, ys2, seg2)):
    for k in range(1, len(X) - 1):
        h_left = X[k] - X[k - 1]
        left, right = S[k - 1], S[k]
        # 值连续
        max_gap2 = max(max_gap2, abs(eval_seg(left, h_left) - Y[k]))
        # 一阶导数：左段在 t=h，右段在 t=0
        max_gap1 = max(max_gap1, abs(deriv(left, h_left) - deriv(right, 0.0)))
        # 二阶导数
        max_gap2 = max(max_gap2, abs(deriv2(left, h_left) - deriv2(right, 0.0)))

check("3. 内节点一阶导数连续", max_gap1 < 1e-9, "-> max gap = %.3e" % max_gap1)
check("3. 内节点二阶导数连续/值连续", max_gap2 < 1e-9, "-> max gap = %.3e" % max_gap2)

# 自然边界：两端二阶导数为 0
s2_left = deriv2(seg2[0], 0.0)
s2_right = deriv2(seg2[-1], xs2[-1] - xs2[-2])
check("4. 自然边界 M[0] = 0", abs(s2_left) < 1e-9, "-> %.3e" % s2_left)
check("4. 自然边界 M[n] = 0", abs(s2_right) < 1e-9, "-> %.3e" % s2_right)

# ======================================================================
# 用例 5：只有两个节点（无内点，退化成直线）
# ======================================================================
seg3 = natural_cubic_spline([0.0, 2.0], [1.0, 5.0])
check("5. 两个节点 -> 1 段", len(seg3) == 1, "-> %r" % (seg3,))
check("5. 退化为直线：中点", abs(eval_spline(seg3, [0.0, 2.0], 1.0) - 3.0) < 1e-12,
      "-> %r" % eval_spline(seg3, [0.0, 2.0], 1.0))
check("5. 端点值", eval_spline(seg3, [0.0, 2.0], 0.0) == 1.0
      and eval_spline(seg3, [0.0, 2.0], 2.0) == 5.0)

# ======================================================================
# 用例 6：非法输入一律 ValueError
# ======================================================================
expect_value_error("6. 长度不同",
                   lambda: natural_cubic_spline([0.0, 1.0, 2.0], [0.0, 1.0]))
expect_value_error("6. 少于两个点",
                   lambda: natural_cubic_spline([0.0], [1.0]))
expect_value_error("6. 空列表",
                   lambda: natural_cubic_spline([], []))
expect_value_error("6. xs 有相等项（非严格递增）",
                   lambda: natural_cubic_spline([0.0, 1.0, 1.0], [0.0, 1.0, 2.0]))
expect_value_error("6. xs 递减",
                   lambda: natural_cubic_spline([2.0, 1.0, 0.0], [0.0, 1.0, 2.0]))
expect_value_error("6. eval_spline 的 xs 非严格递增",
                   lambda: eval_spline(seg2, [0.0, 0.0, 1.0], 0.5))

# ======================================================================
# 用例 7：不能改调用者手里的 xs / ys
# ======================================================================
xs_c = [0, 1, 2]
ys_c = [0, 1, 0]
xs_snap, ys_snap = copy.deepcopy(xs_c), copy.deepcopy(ys_c)
natural_cubic_spline(xs_c, ys_c)
check("7. 未修改调用者的节点列表", xs_c == xs_snap and ys_c == ys_snap,
      "-> xs=%r, ys=%r" % (xs_c, ys_c))

# ======================================================================
# 用例 8：插值 sin(x)，非均匀节点，整体误差较小
# ======================================================================
xs4 = [0.0, 0.3, 0.8, 1.4, 2.2, 3.1]
ys4 = [math.sin(v) for v in xs4]
seg4 = natural_cubic_spline(xs4, ys4)
worst = 0.0
for k in range(len(xs4) - 1):
    for m in range(1, 10):
        x = xs4[k] + (xs4[k + 1] - xs4[k]) * m / 10.0
        worst = max(worst, abs(eval_spline(seg4, xs4, x) - math.sin(x)))
check("8. 插值 sin(x) 的最大误差 < 1e-2", worst < 1e-2, "-> max err = %.3e" % worst)

print("\n通过 %d 项，失败 %d 项" % (_PASSED, _FAILED))
sys.exit(1 if _FAILED else 0)
