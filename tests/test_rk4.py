# test_rk4.py
# 直接运行：python tests/test_rk4.py
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rk4 import rk4_solve, rk4_step  # noqa: E402

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


def expect_value_error(name, fn, keyword):
    try:
        fn()
    except ValueError as e:
        check(name, keyword in str(e), "-> ValueError: %s" % e)
    except Exception as e:  # noqa: BLE001
        check(name, False, "-> 抛出的是 %s: %s" % (type(e).__name__, e))
    else:
        check(name, False, "-> 没有抛出异常")


# ======================================================================
# 用例（题面指定）：y' = -y, y(0) = 1, h = 0.1, 积到 x = 1
# 精确解 y = e^(-x)，终点误差 < 1e-4
# ======================================================================
f_decay = lambda x, y: -y  # noqa: E731
xs, ys = rk4_solve(f_decay, 0, 1, 1, 0.1)
exact = math.exp(-1.0)
err = abs(ys[-1] - exact)
check("A. y'=-y 终点误差 < 1e-4", err < 1e-4, "-> y(1)=%.12f, e^-1=%.12f, err=%.3e" % (ys[-1], exact, err))
check("A. 两条 list 等长", len(xs) == len(ys), "-> len(xs)=%d, len(ys)=%d" % (len(xs), len(ys)))
check("A. xs[0], ys[0] 是起点", xs[0] == 0 and ys[0] == 1, "-> (%r, %r)" % (xs[0], ys[0]))
check("A. 步数", len(xs) == 11, "-> len(xs) = %d" % len(xs))
check("A. xs 最后一项严格等于 x_end", xs[-1] == 1, "-> xs[-1] = %r" % xs[-1])

# ======================================================================
# 单步 & 一步区间：rk4_step 只走一步；rk4_solve(..., 0.1, 0.1) 长度为 2
# ======================================================================
y_one = rk4_step(lambda x, y: y, 0, 1, 0.1)
xs2, ys2 = rk4_solve(lambda x, y: y, 0, 1, 0.1, 0.1)
check("B. rk4_solve 只走一步：长度 2", len(xs2) == 2 and len(ys2) == 2,
      "-> xs = %r" % (xs2,))
check("B. 终点 x 为 0.1", xs2[-1] == 0.1, "-> xs[-1] = %r" % xs2[-1])
check("B. rk4_step 与 rk4_solve 单步结果一致", abs(y_one - ys2[-1]) < 1e-15,
      "-> step=%.15f, solve=%.15f" % (y_one, ys2[-1]))
check("B. 单步 RK4 逼近 e^0.1", abs(y_one - math.exp(0.1)) < 1e-6,
      "-> err = %.3e" % abs(y_one - math.exp(0.1)))

# ======================================================================
# y' = y, y(0) = 1：h = 0.1 与 h = 0.05 积到 1，检查误差与收敛阶
# ======================================================================
f_grow = lambda x, y: y  # noqa: E731
xs10, ys10 = rk4_solve(f_grow, 0, 1, 1, 0.1)
xs05, ys05 = rk4_solve(f_grow, 0, 1, 1, 0.05)

err_x10 = abs(xs10[-1] - 1.0)
err_x05 = abs(xs05[-1] - 1.0)
err_y10 = abs(ys10[-1] - math.e)
err_y05 = abs(ys05[-1] - math.e)

check("C. h=0.1：|xs[-1] - 1| < 1e-9", err_x10 < 1e-9, "-> %.3e" % err_x10)
check("C. h=0.1：|ys[-1] - e| < 1e-4", err_y10 < 1e-4, "-> %.3e" % err_y10)
check("C. h=0.05：|xs[-1] - 1| < 1e-9", err_x05 < 1e-9, "-> %.3e" % err_x05)
check("C. h=0.05：|ys[-1] - e| < 1e-4", err_y05 < 1e-4, "-> %.3e" % err_y05)
check("C. h=0.05 的 x 误差更小", err_x05 <= err_x10, "-> %.3e -> %.3e" % (err_x10, err_x05))
check("C. h=0.05 的 y 误差更小", err_y05 < err_y10, "-> %.3e -> %.3e" % (err_y10, err_y05))

ratio = err_y10 / err_y05 if err_y05 > 0 else float("inf")
check("C. 误差比落在 8~32 之间（四阶收敛）", 8.0 < ratio < 32.0, "-> ratio = %.3f" % ratio)
check("C. h=0.1 步数为 10", len(xs10) == 11, "-> len = %d" % len(xs10))
check("C. h=0.05 步数为 20", len(xs05) == 21, "-> len = %d" % len(xs05))

# ======================================================================
# 约定：h <= 0 抛 ValueError（信息含「步长」）
# ======================================================================
expect_value_error("D. rk4_step h=0 抛 ValueError",
                   lambda: rk4_step(f_grow, 0, 1, 0), "步长")
expect_value_error("D. rk4_solve h=0 抛 ValueError",
                   lambda: rk4_solve(f_grow, 0, 1, 1, 0), "步长")
expect_value_error("D. rk4_step h<0 抛 ValueError",
                   lambda: rk4_step(f_grow, 0, 1, -0.1), "步长")
expect_value_error("D. rk4_solve h<0 抛 ValueError",
                   lambda: rk4_solve(f_grow, 0, 1, 1, -0.1), "步长")

# ======================================================================
# 约定：x_end == x0 时直接返回 ([x0], [y0])
# ======================================================================
xs0, ys0 = rk4_solve(f_grow, 2.0, 5.0, 2.0, 0.1)
check("E. x_end == x0", xs0 == [2.0] and ys0 == [5.0], "-> (%r, %r)" % (xs0, ys0))

# ======================================================================
# 约定：区间不能被 h 整除时，前面完整步长 + 最后一步缩短，终点仍是 x_end
# ======================================================================
xs3, ys3 = rk4_solve(f_grow, 0, 1, 1.05, 0.1)
check("F. 10 个完整步 + 1 个缩短步 = 12 个点", len(xs3) == 12, "-> len = %d" % len(xs3))
check("F. 终点严格等于 x_end", xs3[-1] == 1.05, "-> xs[-1] = %r" % xs3[-1])
check("F. 每步长度不超过 h",
      all(xs3[i + 1] - xs3[i] <= 0.1 + 1e-12 for i in range(len(xs3) - 1)),
      "-> 步长序列 = %s" % ["%.4f" % (xs3[i + 1] - xs3[i]) for i in range(len(xs3) - 1)])
check("F. 最后一段比 h 短", (xs3[-1] - xs3[-2]) < 0.1 - 1e-12,
      "-> 最后一段 = %.6f" % (xs3[-1] - xs3[-2]))
check("F. 终点 y 逼近 e^1.05",
      abs(ys3[-1] - math.exp(1.05)) < 1e-4, "-> err = %.3e" % abs(ys3[-1] - math.exp(1.05)))

# ======================================================================
# 附加：起点必须在列表首位；步长恒为正，因此 x_end < x0 也抛 ValueError
# （旧版本曾用 h<0 做反向积分，现已按「h 必须为正」的约定改为报错）
# ======================================================================
xs4, ys4 = rk4_solve(f_grow, 0.5, 2.0, 1.5, 0.1)
check("G. 起点是列表首位", xs4[0] == 0.5 and ys4[0] == 2.0, "-> (%r, %r)" % (xs4[0], ys4[0]))
check("G. 终点是列表末位且严格等于 x_end", xs4[-1] == 1.5, "-> xs[-1] = %r" % xs4[-1])
check("G. 单调向前推进", all(xs4[i + 1] > xs4[i] for i in range(len(xs4) - 1)),
      "-> xs = %s" % ["%.4f" % v for v in xs4])
expect_value_error("G. x_end < x0 抛 ValueError（步长为正，不反向）",
                   lambda: rk4_solve(f_grow, 1.0, math.e, 0.0, 0.1), "步长")

print("\n通过 %d 项，失败 %d 项" % (_PASSED, _FAILED))
sys.exit(1 if _FAILED else 0)
