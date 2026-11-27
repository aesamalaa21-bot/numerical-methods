"""
Numerical Methods Toolkit
=========================
كل ميثود في كلاس لوحدها، وكل كلاس فيه 3 حاجات منفصلة:
    - ask()     : بتسأل اليوزر وترجّع الـ inputs (dict)      <- هتتبدل بـ HTML form
    - solve()   : الحسابات بس، من غير print أو input        <- هتتحول لـ JS أو API
    - display() : بتطبع النتيجة                              <- هتتبدل بـ HTML table
    - plot()    : بترسم الجراف                               <- هتتبدل بـ Chart.js / Plotly
    - run()     : بتشغّل الخطوات دي ورا بعض

البرنامج الرئيسي بيفضل شغال لحد ما اليوزر يدوس 0.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

X = sp.symbols("x")


# ======================================================================
# Helpers مشتركة بين الميثودز
# ======================================================================
def parse_function(text):
    """يحوّل النص لـ (sympy expression, numpy function)."""
    text = text.replace("^", "**")
    expr = sp.sympify(text, locals={"e": sp.E})
    extra = expr.free_symbols - {X}
    if extra:
        raise ValueError("f(x) can only use the variable x, but it also contains: "
                         + ", ".join(sorted(str(v) for v in extra)) + ".")
    if expr.atoms(sp.core.function.AppliedUndef):
        raise ValueError("f(x) uses a function name I don't know. "
                         "Use sin, cos, tan, exp, log, sqrt, abs.")
    return expr, sp.lambdify(X, expr, "numpy")


# Default stopping tolerances, used only when the user leaves es empty.
# Without them the loop keeps running after the answer is already exact.
TOL_F = 1e-12      # stop when |f(x)| is below this
TOL_X = 1e-12      # stop when the iterate stops changing (relative)
DIVERGE_LIMIT = 1e10   # stop a linear solver when the residual explodes past this


def converged_by_default(x_old, x_new, f_new):
    """True when |f(x)| ~ 0 or the iterate no longer changes (used when es is empty)."""
    if not (math.isfinite(x_new) and math.isfinite(f_new)):
        return False
    return abs(f_new) < TOL_F or abs(x_new - x_old) <= TOL_X * max(1.0, abs(x_new))


def ask_stopping_criterion():
    """يسأل عن شرط التوقف ويرجّع (es, max_iter)."""
    print()
    print("Stopping criterion:")
    print("1) Stop by es (%)")
    print("2) Stop by number of iterations")
    print("3) Both (whichever comes first)")
    choice = input("Choose (1, 2 or 3): ")

    es = None
    max_iter = 100  # default safety limit

    if choice in ("1", "3"):
        es = float(input("es (%) = "))
    if choice in ("2", "3"):
        max_iter = int(input("Number of iterations = "))
        if max_iter <= 0:
            raise ValueError("Iterations must be positive")
    if choice not in ("1", "2", "3"):
        es = float(input("es (%) = "))
    return es, max_iter


def ask_true_value():
    text = input("True value (or press Enter to skip) = ")
    return float(text) if text.strip() != "" else None


def print_summary_root(result):
    print()
    print(f"Last estimate ≈ {result['root']:.8f}  after {result['iterations']} iterations")
    print(f"f(last estimate) = {result['f_root']:.8f}")
    if result["best_x"] is not None:
        print()
        print(f"Best root ≈ {result['best_x']:.8f}  (found at iteration {result['best_iter']})")
        print(f"f(best root) = {result['f_best']:.8f}")


# ======================================================================
# 1) Bisection (Mano)
# ======================================================================
class Bisection:
    name = "Bisection"

    def ask(self):
        text = input("f(x) = ")
        xl = float(input("xl = "))
        xu = float(input("xu = "))
        es, max_iter = ask_stopping_criterion()
        xt = ask_true_value()
        return dict(text=text, xl=xl, xu=xu, es=es, max_iter=max_iter, xt=xt)

    def solve(self, text, xl, xu, es=None, max_iter=100, xt=None):
        _, f = parse_function(text)

        if not (math.isfinite(float(f(xl))) and math.isfinite(float(f(xu)))):
            raise ValueError("f(x) can't be evaluated at xl or xu (check the domain, "
                             "e.g. log or sqrt of a negative number).")
        if f(xl) * f(xu) > 0:
            raise ValueError("No sign change between xl and xu, choose another interval.")

        xl0, xu0 = xl, xu
        rows, points = [], []
        xr_old = None
        root = xl
        best_x = best_f = None
        best_iter = 0
        message = None
        i = 0

        for i in range(1, max_iter + 1):
            xr = (xl + xu) / 2
            ea = abs((xr - xr_old) / xr) * 100 if xr_old is not None and xr != 0 else None
            et = abs((xt - xr) / xt) * 100 if xt not in (None, 0) else None

            rows.append(dict(i=i, xl=xl, xu=xu, xr=xr, fxr=float(f(xr)), ea=ea, et=et))
            points.append(xr)
            root = xr

            fabs = abs(f(xr))
            if best_f is None or fabs < best_f:
                best_x, best_f, best_iter = xr, fabs, i

            if f(xr) == 0:
                message = "Stopped: f(xr) = 0, exact root found."
                break
            if es is not None and ea is not None and ea < es:
                break
            if es is None and (xu - xl) / 2 <= TOL_X * max(1.0, abs(xr)):
                message = "Stopped: the interval is smaller than 1e-12, the root is found to machine precision."
                break

            if f(xl) * f(xr) < 0:
                xu = xr
            else:
                xl = xr
            xr_old = xr

        return dict(text=text, rows=rows, points=points, root=root, f_root=float(f(root)),
                    best_x=best_x, best_iter=best_iter, f_best=float(f(best_x)),
                    iterations=i, has_et=xt is not None, xl0=xl0, xu0=xu0, message=message)

    def display(self, r):
        print()
        header = f"{'Iter':>4} {'xl':>14} {'xu':>14} {'xr':>14} {'f(xr)':>14} {'ea (%)':>12}"
        if r["has_et"]:
            header += f" {'et (%)':>12}"
        print(header)
        print("-" * 90)
        for row in r["rows"]:
            ea = "-" if row["ea"] is None else f"{row['ea']:.6f}"
            line = (f"{row['i']:>4} {row['xl']:>14.8f} {row['xu']:>14.8f} "
                    f"{row['xr']:>14.8f} {row['fxr']:>14.8f} {ea:>12}")
            if r["has_et"]:
                line += f" {row['et']:>12.6f}"
            print(line)
        if r["message"]:
            print(r["message"])
        print_summary_root(r)

    def plot(self, r):
        _, f = parse_function(r["text"])
        xl0, xu0 = r["xl0"], r["xu0"]
        xs = np.linspace(xl0 - 0.1 * (xu0 - xl0), xu0 + 0.1 * (xu0 - xl0), 500)
        plt.figure(figsize=(8, 5))
        plt.plot(xs, f(xs) * np.ones_like(xs), label=f"f(x) = {r['text']}")
        plt.axhline(0, color="black", linewidth=1)
        plt.axvline(xl0, color="gray", linestyle="--", label="xl (initial)")
        plt.axvline(xu0, color="gray", linestyle="--", label="xu (initial)")
        plt.plot(r["points"], [f(p) for p in r["points"]], "g.", markersize=10, label="iterations")
        plt.plot(r["best_x"], f(r["best_x"]), "ro", label=f"Best root ≈ {r['best_x']:.6f}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Bisection Method")
        plt.grid(True)
        plt.legend()
        plt.show()

    def run(self):
        result = self.solve(**self.ask())
        self.display(result)
        self.plot(result)


# ======================================================================
# 2) Newton-Raphson (Sara)
# ======================================================================
class NewtonRaphson:
    name = "Newton-Raphson"

    def ask(self):
        text = input("f(x) = ")
        x0 = float(input("x(0) = "))
        es, max_iter = ask_stopping_criterion()
        xt = ask_true_value()
        return dict(text=text, x0=x0, es=es, max_iter=max_iter, xt=xt)

    def solve(self, text, x0, es=None, max_iter=100, xt=None):
        f_expr, f = parse_function(text)
        df_expr = sp.diff(f_expr, X)
        df = sp.lambdify(X, df_expr, "numpy")

        rows, points = [], [x0]
        root = x0
        best_x = best_f = None
        best_iter = 0
        message = None
        i = 0

        for i in range(1, max_iter + 1):
            if df(x0) == 0:
                message = "f'(x) = 0, can't divide. Try another x(0)."
                i -= 1
                break

            x1 = x0 - f(x0) / df(x0)
            ea = abs((x1 - x0) / x1) * 100 if x1 != 0 else float("inf")
            et = abs((xt - x1) / xt) * 100 if xt not in (None, 0) else None

            rows.append(dict(i=i, x0=x0, fx0=float(f(x0)), dfx0=float(df(x0)),
                             x1=x1, ea=ea, et=et))
            points.append(x1)
            root = x1

            fabs = abs(f(x1))
            if best_f is None or fabs < best_f:
                best_x, best_f, best_iter = x1, fabs, i

            if es is not None and ea < es:
                break
            if es is None and converged_by_default(x0, x1, float(f(x1))):
                message = "Stopped: converged (|f(x)| < 1e-12 or x stopped changing)."
                break
            if not math.isfinite(x1):
                message = "Stopped: the iteration blew up (x is not finite). Try another x(0)."
                break
            x0 = x1

        return dict(text=text, df_text=str(df_expr), rows=rows, points=points, root=root,
                    f_root=float(f(root)), best_x=best_x, best_iter=best_iter,
                    f_best=float(f(best_x)) if best_x is not None else None,
                    iterations=i, has_et=xt is not None, message=message)

    def display(self, r):
        print("f'(x) =", r["df_text"])
        print()
        header = f"{'Iter':>4} {'x(i)':>14} {'f(x(i))':>14} {'f_(x(i))':>14} {'x(i+1)':>14} {'ea (%)':>12}"
        if r["has_et"]:
            header += f" {'et (%)':>12}"
        print(header)
        print("-" * 90)
        for row in r["rows"]:
            line = (f"{row['i']:>4} {row['x0']:>14.8f} {row['fx0']:>14.8f} "
                    f"{row['dfx0']:>14.8f} {row['x1']:>14.8f} {row['ea']:>12.6f}")
            if r["has_et"]:
                line += f" {row['et']:>12.6f}"
            print(line)
        if r["message"]:
            print(r["message"])
        print_summary_root(r)

    def plot(self, r):
        _, f = parse_function(r["text"])
        pts = r["points"]
        best_x = r["best_x"] if r["best_x"] is not None else r["root"]
        xs = np.linspace(min(pts) - 1, max(pts) + 1, 500)
        plt.figure(figsize=(8, 5))
        plt.plot(xs, f(xs) * np.ones_like(xs), label=f"f(x) = {r['text']}")
        plt.axhline(0, color="black", linewidth=1)
        plt.plot(pts, [f(p) for p in pts], "g.", markersize=10, label="iterations")
        plt.plot(best_x, f(best_x), "ro", label=f"Best root ≈ {best_x:.6f}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Newton-Raphson Method")
        plt.grid(True)
        plt.legend()
        plt.show()

    def run(self):
        result = self.solve(**self.ask())
        self.display(result)
        self.plot(result)


# ======================================================================
# 3) Secant / Modified Secant (Essam)
# ======================================================================
class Secant:
    name = "Secant / Modified Secant"

    def ask(self):
        print("1) Secant")
        print("2) Modified Secant")
        mode = input("Choose method (1 or 2): ")
        text = input("f(x) = ")

        x_old = delta = None
        if mode == "1":
            x_old = float(input("x(-1) = "))
            x = float(input("x(0) = "))
        else:
            x = float(input("x(0) = "))
            delta = float(input("delta = "))

        es, max_iter = ask_stopping_criterion()
        xt = ask_true_value()
        return dict(mode=mode, text=text, x=x, x_old=x_old, delta=delta,
                    es=es, max_iter=max_iter, xt=xt)

    def solve(self, mode, text, x, x_old=None, delta=None, es=None, max_iter=100, xt=None):
        _, f = parse_function(text)

        points = [x_old, x] if mode == "1" else [x]
        rows = []
        best_x = best_f = None
        best_iter = 0
        message = None
        x_new = x
        i = 0

        for i in range(1, max_iter + 1):
            if mode == "1":
                denom = f(x_old) - f(x)
                if denom == 0:
                    message = "Division by zero: f(x(i-1)) = f(x(i)). Stopping."
                    i -= 1
                    break
                x_new = x - f(x) * (x_old - x) / denom
            else:
                denom = f(x + delta * x) - f(x)
                if denom == 0:
                    message = "Division by zero in denominator. Stopping."
                    i -= 1
                    break
                x_new = x - delta * x * f(x) / denom

            ea = abs((x_new - x) / x_new) * 100 if x_new != 0 else float("inf")
            et = abs((xt - x_new) / xt) * 100 if xt not in (None, 0) else None

            row = dict(i=i, x=x, fx=float(f(x)), x_new=x_new, ea=ea, et=et)
            if mode == "1":
                row.update(x_old=x_old, fx_old=float(f(x_old)))
            else:
                row.update(x_pert=x + delta * x, f_pert=float(f(x + delta * x)))
            rows.append(row)
            points.append(x_new)

            fabs = abs(f(x_new))
            if best_f is None or fabs < best_f:
                best_x, best_f, best_iter = x_new, fabs, i

            if es is not None and ea < es:
                break
            if es is None and converged_by_default(x, x_new, float(f(x_new))):
                message = "Stopped: converged (|f(x)| < 1e-12 or x stopped changing)."
                break
            if not math.isfinite(x_new):
                message = "Stopped: the iteration blew up (x is not finite). Try other starting values."
                break
            x_old = x
            x = x_new

        return dict(mode=mode, text=text, rows=rows, points=points, root=x_new,
                    f_root=float(f(x_new)), best_x=best_x, best_iter=best_iter,
                    f_best=float(f(best_x)) if best_x is not None else None,
                    iterations=i, has_et=xt is not None, message=message)

    def display(self, r):
        print()
        if r["mode"] == "1":
            header = (f"{'Iter':>4} {'x(i-1)':>14} {'x(i)':>14} {'f(x(i-1))':>14} "
                      f"{'f(x(i))':>14} {'x(i+1)':>14} {'ea (%)':>12}")
        else:
            header = (f"{'Iter':>4} {'x(i)':>14} {'f(x(i))':>14} {'x(i)+d*x(i)':>14} "
                      f"{'f(x(i)+d*x(i))':>16} {'x(i+1)':>14} {'ea (%)':>12}")
        if r["has_et"]:
            header += f" {'et (%)':>12}"
        print(header)
        print("-" * 105)

        for row in r["rows"]:
            if r["mode"] == "1":
                line = (f"{row['i']:>4} {row['x_old']:>14.8f} {row['x']:>14.8f} "
                        f"{row['fx_old']:>14.8f} {row['fx']:>14.8f} "
                        f"{row['x_new']:>14.8f} {row['ea']:>12.6f}")
            else:
                line = (f"{row['i']:>4} {row['x']:>14.8f} {row['fx']:>14.8f} "
                        f"{row['x_pert']:>14.8f} {row['f_pert']:>16.8f} "
                        f"{row['x_new']:>14.8f} {row['ea']:>12.6f}")
            if r["has_et"]:
                line += f" {row['et']:>12.6f}"
            print(line)
        if r["message"]:
            print(r["message"])
        print_summary_root(r)

    def plot(self, r):
        _, f = parse_function(r["text"])
        pts = r["points"]
        best_x = r["best_x"] if r["best_x"] is not None else r["root"]
        xs = np.linspace(min(pts) - 1, max(pts) + 1, 500)
        plt.figure(figsize=(8, 5))
        plt.plot(xs, f(xs) * np.ones_like(xs), label=f"f(x) = {r['text']}")
        plt.axhline(0, color="black", linewidth=1)
        plt.plot(pts, [f(p) for p in pts], "g.", markersize=10, label="iterations")
        plt.plot(best_x, f(best_x), "ro", label=f"Best root ≈ {best_x:.6f}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Secant Method" if r["mode"] == "1" else "Modified Secant Method")
        plt.grid(True)
        plt.legend()
        plt.show()

    def run(self):
        result = self.solve(**self.ask())
        self.display(result)
        self.plot(result)


# ======================================================================
# 4) Jacobi / Gauss-Seidel (Aya)
# ======================================================================
def is_diagonally_dominant(A):
    """|a_ii| >= sum of |a_ij| (j != i) for every row: sufficient for convergence."""
    A = np.abs(np.array(A, dtype=float))
    off = A.sum(axis=1) - np.diag(A)
    return bool(np.all(np.diag(A) >= off))


class JacobiSeidel:
    name = "Jacobi / Gauss-Seidel"

    def ask(self):
        print("1) Jacobi Method")
        print("2) Gauss-Seidel Method")
        mode = input("Choose method (1 or 2): ")

        n = int(input("Number of variables: "))
        A = np.zeros((n, n))
        b = np.zeros(n)

        print("\nEnter coefficients:")
        for i in range(n):
            for j in range(n):
                A[i][j] = float(input(f"a[{i+1}][{j+1}] = "))
            b[i] = float(input(f"b[{i+1}] = "))

        if np.any(np.diag(A) == 0):
            raise ValueError("Diagonal elements cannot be zero.")

        print("\nInitial guesses:")
        x0 = np.array([float(input(f"x{i+1} = ")) for i in range(n)])

        es, max_iter = ask_stopping_criterion()

        xt = None
        xt_text = input("True values (separated by spaces, Enter to skip) = ")
        if xt_text.strip() != "":
            xt = np.array([float(v) for v in xt_text.split()])
            if len(xt) != n:
                raise ValueError("Enter the correct number of true values.")

        return dict(mode=mode, A=A, b=b, x0=x0, es=es, max_iter=max_iter, xt=xt)

    def solve(self, mode, A, b, x0, es=None, max_iter=100, xt=None):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        x = np.array(x0, dtype=float)
        n = len(b)

        if np.any(np.diag(A) == 0):
            raise ValueError("Diagonal elements cannot be zero.")

        warning = None
        if not is_diagonally_dominant(A):
            warning = ("The matrix is not diagonally dominant, so Jacobi / Gauss-Seidel "
                       "may not converge. Try reordering the equations.")
        message = None

        rows, points = [], [x.copy()]
        best_x = best_res = None
        best_iter = 0
        i = 0

        for i in range(1, max_iter + 1):
            x_old = x.copy()

            for j in range(n):
                values = x_old if mode == "1" else x
                total = sum(A[j][k] * values[k] for k in range(n) if k != j)
                x[j] = (b[j] - total) / A[j][j]

            ea = max(abs((x - x_old) / np.where(x != 0, x, 1))) * 100
            et = max(abs((xt - x) / np.where(xt != 0, xt, 1))) * 100 if xt is not None else None

            rows.append(dict(i=i, x=x.copy(), ea=ea, et=et))
            points.append(x.copy())

            res = np.max(np.abs(A @ x - b))
            if best_res is None or res < best_res:
                best_x, best_res, best_iter = x.copy(), res, i

            if es is not None and ea < es:
                break
            if es is None and ea < 1e-10 and res < 1e-10 * max(1.0, float(np.max(np.abs(b)))):
                message = "Stopped: converged (x stopped changing and the residual is ~ 0)."
                break
            if not np.isfinite(res) or res > DIVERGE_LIMIT * max(1.0, float(np.max(np.abs(b)))):
                message = ("Diverged: the values are growing without bound, so there is no solution "
                           "to report. Check that the system is diagonally dominant.")
                break

        return dict(mode=mode, warning=warning, message=message, diverged=message is not None and message.startswith("Diverged"), n=n, A=A, b=b, rows=rows, points=np.array(points),
                    x=x, best_x=best_x, best_iter=best_iter, iterations=i,
                    residual=A @ x - b, best_residual=A @ best_x - b,
                    has_et=xt is not None)

    def display(self, r):
        n = r["n"]
        print()
        header = f"{'Iter':>4}" + "".join(f"{'x' + str(j + 1):>14}" for j in range(n))
        header += f"{'ea (%)':>14}"
        if r["has_et"]:
            header += f"{'et (%)':>14}"
        print(header)
        print("-" * (18 + 14 * n))

        for row in r["rows"]:
            line = f"{row['i']:>4}" + "".join(f"{v:>14.8f}" for v in row["x"])
            line += f"{row['ea']:>14.6f}"
            if r["has_et"]:
                line += f"{row['et']:>14.6f}"
            print(line)

        print()
        if r["warning"]:
            print("Warning:", r["warning"])
        if r["message"]:
            print(r["message"])
        print("Jacobi Method" if r["mode"] == "1" else "Gauss-Seidel Method")
        print("\nFinal Answers (last iteration):")
        for j in range(n):
            print(f"x{j+1} = {r['x'][j]:.8f}")
        print(f"\nNumber of iterations = {r['iterations']}")
        print("Residual (A*x - b):", r["residual"])

        print(f"\nBest solution (smallest residual) found at iteration {r['best_iter']}:")
        for j in range(n):
            print(f"x{j+1} = {r['best_x'][j]:.8f}")
        print("Residual (A*x - b):", r["best_residual"])

    def plot(self, r):
        pts = r["points"]
        plt.figure(figsize=(8, 5))
        for j in range(r["n"]):
            plt.plot(range(len(pts)), pts[:, j], marker="o", label=f"x{j+1}")
        plt.axvline(r["best_iter"], color="red", linestyle="--", label=f"Best (iter {r['best_iter']})")
        plt.xlabel("Iterations")
        plt.ylabel("Variable values")
        plt.title("Jacobi Method Convergence" if r["mode"] == "1" else "Gauss-Seidel Method Convergence")
        plt.grid(True)
        plt.legend()
        plt.show()

    def run(self):
        result = self.solve(**self.ask())
        self.display(result)
        self.plot(result)


# ======================================================================
# 5) Linear Regression (Raheeq)
# ======================================================================
class LinearRegression:
    name = "Linear Regression"

    @staticmethod
    def _get_float_list(prompt):
        while True:
            raw = input(prompt).replace(",", " ").split()
            try:
                vals = [float(item) for item in raw]
                if len(vals) < 3:
                    print("Please enter at least 3 values.")
                    continue
                return vals
            except ValueError:
                print("Invalid input. Enter numbers separated by spaces or commas.")

    def ask(self):
        while True:
            x_vals = self._get_float_list("Enter X values: ")
            y_vals = self._get_float_list("Enter Y values: ")
            if len(x_vals) != len(y_vals):
                print("Error: X and Y must have the same number of values.")
                continue
            return dict(x_values=x_vals, y_values=y_vals)

    def solve(self, x_values, y_values):
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_x2 = sum(v ** 2 for v in x_values)
        sum_y2 = sum(v ** 2 for v in y_values)
        sum_xy = sum(a * b for a, b in zip(x_values, y_values))

        denominator_x = n * sum_x2 - sum_x ** 2
        if denominator_x == 0:
            raise ValueError("All X values are identical; cannot compute a unique regression line.")

        a1 = (n * sum_xy - sum_x * sum_y) / denominator_x
        mean_x = sum_x / n
        mean_y = sum_y / n
        a0 = mean_y - a1 * mean_x

        st = sum((v - mean_y) ** 2 for v in y_values)
        sr = sum((v - (a0 + a1 * a)) ** 2 for a, v in zip(x_values, y_values))

        sy = math.sqrt(st / (n - 1)) if n > 1 else 0.0
        sy_x = math.sqrt(sr / (n - 2)) if n > 2 else 0.0

        r2 = (st - sr) / st if st != 0 else 0.0
        denominator_r = math.sqrt(denominator_x * (n * sum_y2 - sum_y ** 2))
        r = ((n * sum_xy - sum_x * sum_y) / denominator_r) if denominator_r != 0 else 0.0

        return dict(x_values=x_values, y_values=y_values, a0=a0, a1=a1,
                    sum_x=sum_x, sum_y=sum_y, sum_x2=sum_x2, sum_y2=sum_y2, sum_xy=sum_xy,
                    st=st, sr=sr, sy=sy, sy_x=sy_x, r2=r2, r=r)

    @staticmethod
    def predict(res, x_new):
        return res["a0"] + res["a1"] * x_new

    def display(self, r):
        print("\nData Table:")
        print(f"{'No.':<5} {'X':<10} {'Y':<10} {'X^2':<12} {'X*Y':<12}")
        print("-" * 52)
        for idx, (a, b) in enumerate(zip(r["x_values"], r["y_values"]), start=1):
            print(f"{idx:<5} {a:<10.4f} {b:<10.4f} {a**2:<12.4f} {a*b:<12.4f}")
        print("-" * 52)
        print(f"{'Sum':<5} {r['sum_x']:<10.4f} {r['sum_y']:<10.4f} {r['sum_x2']:<12.4f} {r['sum_xy']:<12.4f}")

        print(f"\nEquation: y = {r['a0']:.4f} + {r['a1']:.4f} * x")
        print(f"St = {r['st']:.4f}")
        print(f"Sr = {r['sr']:.4f}")
        print(f"Standard deviation (sy) = {r['sy']:.4f}")
        print(f"Standard error of estimate (sy/x) = {r['sy_x']:.4f}")
        print(f"Coefficient of determination (r^2) = {r['r2']:.4f}")
        print(f"Correlation coefficient (r) = {r['r']:.4f}")

    def plot(self, r):
        xs, ys = r["x_values"], r["y_values"]
        line_x = np.linspace(min(xs), max(xs), 100)
        plt.figure(figsize=(8, 5))
        plt.scatter(xs, ys, label="data")
        plt.plot(line_x, r["a0"] + r["a1"] * line_x, "r-",
                 label=f"y = {r['a0']:.4f} + {r['a1']:.4f}x")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Linear Regression")
        plt.grid(True)
        plt.legend()
        plt.show()

    def run(self):
        while True:
            try:
                res = self.solve(**self.ask())
                break
            except ValueError as e:
                print(f"Error: {e}")

        self.display(res)

        if input("\nPredict y for a new x? (y/n): ").strip().lower() == "y":
            while True:
                try:
                    x_new = float(input("Enter new x: "))
                    print(f"Predicted y = {self.predict(res, x_new):.4f}")
                    break
                except ValueError:
                    print("Please enter a valid number.")

        if input("Show plot? (y/n): ").strip().lower() == "y":
            self.plot(res)


# ======================================================================
# Main menu
# ======================================================================
METHODS = {
    "1": Bisection,
    "2": NewtonRaphson,
    "3": Secant,
    "4": JacobiSeidel,
    "5": LinearRegression,
}


def main():
    while True:
        print("\n" + "=" * 40)
        print("      Numerical Methods Toolkit")
        print("=" * 40)
        for key, cls in METHODS.items():
            print(f"{key}) {cls.name}")
        print("0) Exit")

        choice = input("Choose: ").strip()

        if choice == "0":
            print("Goodbye!")
            break

        if choice not in METHODS:
            print("Invalid choice, try again.")
            continue

        print(f"\n--- {METHODS[choice].name} ---")
        try:
            METHODS[choice]().run()
        except Exception as e:  # نمنع البرنامج إنه يقفل لو حصل error
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
