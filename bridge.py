"""
bridge.py — الجسر بين الـ HTML وملف Numerical_Methods.py
-------------------------------------------------------
ملفك Numerical_Methods.py بيفضل زي ما هو من غير أي تعديل.
الملف ده بس بيعمل 3 حاجات:
  1) يعمل matplotlib وهمي (المتصفح مش محتاجه، الرسم بيتعمل بـ Plotly).
  2) يستدعي solve() بتاعة كل كلاس في ملفك.
  3) يحوّل النتيجة لشكل JSON تفهمه صفحة الـ HTML.
"""
import sys, types, json
import numpy as np

# matplotlib بيُستخدم بس في plot() وإحنا مش بنناديها، فنكتفي بنسخة فاضية
_mpl = types.ModuleType("matplotlib")
_plt = types.ModuleType("matplotlib.pyplot")
_mpl.pyplot = _plt
sys.modules["matplotlib"], sys.modules["matplotlib.pyplot"] = _mpl, _plt

import Numerical_Methods as NM

ERR = [("ea", "ea %"), ("et", "et %")]


def curve(text, pts):
    """نقط منحنى f(x) + نقط الـ iterations للرسم."""
    _, f = NM.parse_function(text)
    lo, hi = min(pts), max(pts)
    pad = (hi - lo) * .15 or 1
    cv = []
    for t in np.linspace(lo - pad, hi + pad, 201):
        try:
            y = float(f(t))
            if np.isfinite(y):
                cv.append([float(t), y])
        except Exception:
            pass
    return cv, [[float(t), float(f(t))] for t in pts]


def root_result(r, key, cols, pts, extra=None):
    """يحوّل نتيجة ميثود إيجاد الجذور (Bisection / Newton / Secant)."""
    rows = [dict(row, x=row[key]) for row in r["rows"]]
    cv, pp = curve(r["text"], pts)
    return dict(kind="root", rows=rows, cols=cols, msg=r.get("message"), curve=cv, pts=pp,
                best=r["best_x"], bi=r["best_iter"] or None, fb=r["f_best"],
                last=r["root"], fl=r["f_root"], **(extra or {}))


def bisection(d):
    r = NM.Bisection().solve(d["f"], d["xl"], d["xu"], d.get("es"), d["n"], d.get("xt"))
    cols = [("i", "Iter"), ("xl", "xl"), ("xu", "xu"), ("xr", "xr"), ("fxr", "f(xr)")] + ERR
    return root_result(r, "xr", cols, [r["xl0"], r["xu0"]] + r["points"])


def newton(d):
    r = NM.NewtonRaphson().solve(d["f"], d["x0"], d.get("es"), d["n"], d.get("xt"))
    cols = [("i", "Iter"), ("x0", "x(i)"), ("fx0", "f(x(i))"), ("dfx0", "f'(x(i))"), ("x1", "x(i+1)")] + ERR
    return root_result(r, "x1", cols, r["points"], dict(note="f'(x) = " + r["df_text"]))


def secant(d):
    m = d["m"]
    if m == "1" and d.get("xm") is None: raise ValueError("Enter x(-1)")
    if m == "2" and d.get("delta") is None: raise ValueError("Enter delta")
    r = NM.Secant().solve(m, d["f"], d["x0"], d.get("xm"), d.get("delta"), d.get("es"), d["n"], d.get("xt"))
    rows = [dict(row, xi=row["x"]) for row in r["rows"]]      # x(i) الحالية
    r["rows"] = rows
    if m == "1":
        cols = [("i", "Iter"), ("x_old", "x(i-1)"), ("xi", "x(i)"), ("fx_old", "f(x(i-1))"), ("fx", "f(x(i))")]
    else:
        cols = [("i", "Iter"), ("xi", "x(i)"), ("fx", "f(x(i))"), ("x_pert", "x(i)+δ·x(i)"), ("f_pert", "f(x(i)+δ·x(i))")]
    return root_result(r, "x_new", cols + [("x_new", "x(i+1)")] + ERR, [p for p in r["points"] if p is not None])


def jacobi(d):
    xt = np.array(d["xt"], dtype=float) if d.get("xt") else None
    n = len(d["b"])
    if any(len(row) != n for row in d["A"]) or len(d["x0"]) != n or (xt is not None and len(xt) != n):
        raise ValueError("Matrix must be n x (n+1) and guesses / true values must have n entries.")
    r = NM.JacobiSeidel().solve(d["m"], d["A"], d["b"], d["x0"], d.get("es"), d["n"], xt)
    rows = []
    for row in r["rows"]:
        out = dict(i=row["i"], ea=row["ea"], et=row["et"], res=float(np.max(np.abs(r["A"] @ row["x"] - r["b"]))))
        out.update({"x%d" % (k + 1): float(v) for k, v in enumerate(row["x"])})
        rows.append(out)
    cols = [("i", "Iter")] + [("x%d" % (k + 1),) * 2 for k in range(n)] + ERR + [("res", "max |Ax−b|")]
    return dict(kind="sys", rows=rows, cols=cols, init=list(d["x0"]), n=n, gs=d["m"] == "2", bi=r["best_iter"])


def regression(d):
    xs, ys = d["x"], d["y"]
    if len(xs) != len(ys) or len(xs) < 3:
        raise ValueError("Enter the same number of X and Y values (at least 3).")
    r = NM.LinearRegression().solve(xs, ys)
    rows = [dict(i=k + 1, x=a, y=b, x2=a * a, xy=a * b) for k, (a, b) in enumerate(zip(xs, ys))]
    rows.append(dict(i="Σ", x=r["sum_x"], y=r["sum_y"], x2=r["sum_x2"], xy=r["sum_xy"]))
    stats = dict(a0=r["a0"], a1=r["a1"], St=r["st"], Sr=r["sr"], sy=r["sy"], syx=r["sy_x"], r2=r["r2"], r=r["r"])
    return dict(kind="reg", rows=rows, stats=stats, xs=xs, ys=ys,
                cols=[("i", "No."), ("x", "X"), ("y", "Y"), ("x2", "X²"), ("xy", "X·Y")])


METHODS = dict(Bisection=bisection, Newton=newton, Secant=secant, Jacobi=jacobi, Regression=regression)


def _default(o):
    return o.tolist() if hasattr(o, "tolist") else float(o)


def run(js):
    d = json.loads(js)
    out = json.dumps(METHODS[d["method"]](d), default=_default)
    return out.replace("-Infinity", "null").replace("Infinity", "null").replace("NaN", "null")
