"""
bridge.py — الجسر بين الـ HTML وملف Numerical_Methods.py
-------------------------------------------------------
ملف Numerical_Methods.py فيه الحسابات بس، والجسر ده بيوصّلها بالصفحة.
الملف ده بس بيعمل 3 حاجات:
  1) يعمل matplotlib وهمي (المتصفح مش محتاجه، الرسم بيتعمل بـ Plotly).
  2) يستدعي solve() بتاعة كل كلاس في ملفك.
  3) يحوّل النتيجة لشكل JSON تفهمه صفحة الـ HTML.
"""
import sys, types, json, math
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


LABELS = dict(f="f(x)", xl="xl", xu="xu", x0="x(0)", xm="x(-1)", delta="delta", A="the matrix",
              b="the matrix", x="the X values", y="the Y values", m="the method")


def need(d, *keys):
    """Make sure the required inputs are filled in and are real numbers."""
    for k in keys:
        v = d.get(k)
        if v is None or v == "" or (isinstance(v, float) and not math.isfinite(v)):
            raise ValueError("Please fill in %s with a valid number." % LABELS.get(k, k))


def check_n(d):
    if d.get("n") is None:
        d["n"] = 100
    if d["n"] != int(d["n"]) or d["n"] < 1:
        raise ValueError("Max iterations must be a whole number of at least 1.")
    d["n"] = int(d["n"])
    if d.get("es") is not None and d["es"] <= 0:
        raise ValueError("es (%) must be a positive number, or leave it empty.")


def friendly_error(e):
    """Turn a Python exception into a message a student can act on."""
    name, text = type(e).__name__, str(e).strip()
    if isinstance(e, ValueError) and name == "ValueError" and not text.startswith(("could not convert", "math domain", "invalid literal")):
        return text
    if name in ("SympifyError", "SyntaxError", "TokenError", "TypeError") and ("sympif" in text.lower() or name != "TypeError"):
        return ("Couldn't read f(x). Write it like x^3 - x - 2, use * for multiplication "
                "(2*x, not 2x) and functions like sin(x), exp(x), log(x), sqrt(x).")
    if name == "NameError" or "is not defined" in text:
        return "f(x) uses a name I don't know. Use only x, numbers and functions like sin, cos, exp, log, sqrt."
    if isinstance(e, ZeroDivisionError):
        return "Division by zero happened during the calculation. Try different starting values."
    if isinstance(e, OverflowError):
        return "The numbers got too large to handle. Try different starting values."
    if isinstance(e, (ValueError, ArithmeticError)):
        return "The function can't be evaluated at one of the points (for example log or sqrt of a negative number). Try a different interval or starting value."
    if isinstance(e, KeyError):
        return "Some inputs are missing. Fill in every field and try again."
    return "Something went wrong with this input (%s). Check the values and try again." % (text or name)


def bisection(d):
    need(d, "f", "xl", "xu"); check_n(d)
    if d["xl"] >= d["xu"]:
        raise ValueError("xl must be smaller than xu.")
    r = NM.Bisection().solve(d["f"], d["xl"], d["xu"], d.get("es"), d["n"], d.get("xt"))
    cols = [("i", "Iter"), ("xl", "xl"), ("xu", "xu"), ("xr", "xr"), ("fxr", "f(xr)")] + ERR
    return root_result(r, "xr", cols, [r["xl0"], r["xu0"]] + r["points"])


def newton(d):
    need(d, "f", "x0"); check_n(d)
    r = NM.NewtonRaphson().solve(d["f"], d["x0"], d.get("es"), d["n"], d.get("xt"))
    cols = [("i", "Iter"), ("x0", "x(i)"), ("fx0", "f(x(i))"), ("dfx0", "f'(x(i))"), ("x1", "x(i+1)")] + ERR
    return root_result(r, "x1", cols, r["points"], dict(note="f'(x) = " + r["df_text"]))


def secant(d):
    m = d["m"]
    need(d, "f", "x0"); check_n(d)
    if m == "1": need(d, "xm")
    if m == "2": need(d, "delta")
    if m == "2" and d["delta"] == 0: raise ValueError("delta can't be zero.")
    r = NM.Secant().solve(m, d["f"], d["x0"], d.get("xm"), d.get("delta"), d.get("es"), d["n"], d.get("xt"))
    rows = [dict(row, xi=row["x"]) for row in r["rows"]]      # x(i) الحالية
    r["rows"] = rows
    if m == "1":
        cols = [("i", "Iter"), ("x_old", "x(i-1)"), ("xi", "x(i)"), ("fx_old", "f(x(i-1))"), ("fx", "f(x(i))")]
    else:
        cols = [("i", "Iter"), ("xi", "x(i)"), ("fx", "f(x(i))"), ("x_pert", "x(i)+δ·x(i)"), ("f_pert", "f(x(i)+δ·x(i))")]
    return root_result(r, "x_new", cols + [("x_new", "x(i+1)")] + ERR, [p for p in r["points"] if p is not None])


def jacobi(d):
    if not d.get("b") or not d.get("x0"):
        raise ValueError("Enter the augmented matrix [A | b] and the initial guesses.")
    check_n(d)
    flat = [v for row in d["A"] for v in row] + list(d["b"]) + list(d["x0"]) + list(d.get("xt") or [])
    if any(v is None or not math.isfinite(v) for v in flat):
        raise ValueError("The matrix and guesses must contain numbers only.")
    xt = np.array(d["xt"], dtype=float) if d.get("xt") else None
    n = len(d["b"])
    if any(len(row) != n for row in d["A"]) or len(d["x0"]) != n or (xt is not None and len(xt) != n):
        raise ValueError("The matrix must have n rows and n+1 columns (A and b), and the guesses / true values need n entries.")
    if any(row[k] == 0 for k, row in enumerate(d["A"])):
        raise ValueError("A diagonal element is zero. Reorder the equations so the diagonal has no zeros.")
    r = NM.JacobiSeidel().solve(d["m"], d["A"], d["b"], d["x0"], d.get("es"), d["n"], xt)
    rows = []
    for row in r["rows"]:
        out = dict(i=row["i"], ea=row["ea"], et=row["et"], res=float(np.max(np.abs(r["A"] @ row["x"] - r["b"]))))
        out.update({"x%d" % (k + 1): float(v) for k, v in enumerate(row["x"])})
        rows.append(out)
    cols = [("i", "Iter")] + [("x%d" % (k + 1),) * 2 for k in range(n)] + ERR + [("res", "max |Ax−b|")]
    return dict(kind="sys", rows=rows, cols=cols, init=list(d["x0"]), n=n, gs=d["m"] == "2", bi=r["best_iter"],
                warning=r["warning"], msg=r["message"], diverged=r["diverged"])


def regression(d):
    xs, ys = d.get("x") or [], d.get("y") or []
    if any(not math.isfinite(v) for v in xs + ys):
        raise ValueError("X and Y must contain numbers only.")
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
    try:
        d = json.loads(js)
        out = json.dumps(METHODS[d["method"]](d), default=_default)
    except Exception as e:  # never leak a raw traceback to the page
        return json.dumps(dict(kind="error", error=friendly_error(e)))
    return out.replace("-Infinity", "null").replace("Infinity", "null").replace("NaN", "null")
