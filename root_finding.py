"""Root-finding methods for f(x) = 0 (linear and nonlinear equations).

Bracketing methods : bisection, false position (regula falsi)
Open methods       : Newton-Raphson, secant

Every solver returns a dict with the root, a convergence flag, a message and a
full iteration log (list of row dicts), mirroring the audit style of
gauss_jordan.py.
"""

import ast
import math
import operator
import re

# -----------------------------------------------------------------------------
# SAFE EXPRESSION PARSER  (never uses eval on raw user text)
# -----------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _cbrt(t):
    return math.copysign(abs(t) ** (1.0 / 3.0), t)


# Functions that do not depend on the angle mode
_COMMON_FUNCS = {
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "exp": math.exp,
    "log": math.log,  # natural log, same as ln
    "ln": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sqrt": math.sqrt,
    "cbrt": _cbrt,
    "abs": abs,
}

_FUNCS_RAD = dict(_COMMON_FUNCS)
_FUNCS_RAD.update({
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sec": lambda t: 1.0 / math.cos(t),
    "csc": lambda t: 1.0 / math.sin(t),
    "cot": lambda t: 1.0 / math.tan(t),
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
})

_FUNCS_DEG = dict(_COMMON_FUNCS)
_FUNCS_DEG.update({
    "sin": lambda t: math.sin(math.radians(t)),
    "cos": lambda t: math.cos(math.radians(t)),
    "tan": lambda t: math.tan(math.radians(t)),
    "sec": lambda t: 1.0 / math.cos(math.radians(t)),
    "csc": lambda t: 1.0 / math.sin(math.radians(t)),
    "cot": lambda t: 1.0 / math.tan(math.radians(t)),
    "asin": lambda t: math.degrees(math.asin(t)),
    "acos": lambda t: math.degrees(math.acos(t)),
    "atan": lambda t: math.degrees(math.atan(t)),
})

_FUNCS = _FUNCS_RAD  # kept for backwards compatibility
FUNCTION_NAMES = sorted(_FUNCS_RAD)
_CONSTS = {"pi": math.pi, "e": math.e}


def _validate_node(node, funcs):
    """Check that an AST only uses allowed syntax (does NOT evaluate it)."""
    if isinstance(node, ast.Expression):
        return _validate_node(node.body, funcs)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return
    if isinstance(node, ast.Name):
        if node.id == "x" or node.id in _CONSTS:
            return
        raise ValueError(f"Unknown name '{node.id}'. Use only x, pi, e.")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        _validate_node(node.left, funcs)
        _validate_node(node.right, funcs)
        return
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _validate_node(node.operand, funcs)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in funcs
        and len(node.args) == 1
        and not node.keywords
    ):
        return _validate_node(node.args[0], funcs)
    raise ValueError(
        "Unsupported expression. Use x, numbers, + - * / ^, parentheses and "
        "functions: " + ", ".join(FUNCTION_NAMES) + "."
    )


def _eval_node(node, x, funcs=_FUNCS_RAD):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, x, funcs)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id == "x":
            return x
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"Unknown name '{node.id}'. Use only x, pi, e.")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](
            _eval_node(node.left, x, funcs), _eval_node(node.right, x, funcs)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand, x, funcs))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in funcs
        and len(node.args) == 1
        and not node.keywords
    ):
        return funcs[node.func.id](_eval_node(node.args[0], x, funcs))
    raise ValueError("Unsupported expression.")


# ---- friendly typing: implicit multiplication, unicode symbols, equations ----
_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z][A-Za-z0-9]*)"
    r"|(?P<op>\*\*|[-+*/^])"
    r"|(?P<lp>\()|(?P<rp>\))|(?P<bad>.))"
)
_KNOWN_WORDS = sorted(set(FUNCTION_NAMES) | {"x", "pi", "e"}, key=len, reverse=True)


def _split_name(word):
    """Split e.g. 'xsin' -> ['x', 'sin'], 'pix' -> ['pi', 'x']. None if impossible."""
    if not word:
        return []
    for known in _KNOWN_WORDS:
        if word.startswith(known):
            rest = _split_name(word[len(known):])
            if rest is not None:
                return [known] + rest
    return None


def _tokenize(text):
    tokens = []  # (kind, text) with kind in NUM VAR FUNC OP LP RP
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None or m.end() == pos:
            break
        pos = m.end()
        if m.group("num"):
            tokens.append(("NUM", m.group("num")))
        elif m.group("name"):
            word = m.group("name").lower()
            parts = _split_name(word)
            if parts is None:
                raise ValueError(
                    f"Unknown name '{m.group('name')}'. Use only x, pi, e and "
                    "the functions listed in the tutorial."
                )
            for part in parts:
                tokens.append(("FUNC" if part in _FUNCS_RAD else "VAR", part))
        elif m.group("op"):
            op = m.group("op")
            tokens.append(("OP", "^" if op == "**" else op))
        elif m.group("lp"):
            tokens.append(("LP", "("))
        elif m.group("rp"):
            tokens.append(("RP", ")"))
        else:
            ch = m.group("bad")
            if ch is not None and ch.strip():
                raise ValueError(f"Unexpected character '{ch}'.")
    return tokens


def _normalize(expr):
    """Return the expression as a clean string with explicit * and ^."""
    if expr is None or not str(expr).strip():
        raise ValueError("Please enter a function.")
    text = str(expr).strip()
    for old, new in (("−", "-"), ("–", "-"), ("×", "*"), ("·", "*"), ("÷", "/"),
                     ("π", "pi"), ("**", "^")):
        text = text.replace(old, new)
    # equations: "f(x) = ...", "y = ...", or "left = right"
    if "=" in text:
        if text.count("=") > 1:
            raise ValueError("Use only one '=' sign.")
        left, right = (side.strip() for side in text.split("="))
        if not right or not left:
            raise ValueError("Both sides of the '=' need an expression.")
        if left.replace(" ", "").lower() in ("f(x)", "y"):
            text = right
        else:
            text = f"({left})-({right})"

    tokens = _tokenize(text)
    out = []
    for i, (kind, tok) in enumerate(tokens):
        if kind == "FUNC" and not (i + 1 < len(tokens) and tokens[i + 1][0] == "LP"):
            raise ValueError(
                f"Put the input of {tok} in parentheses, like {tok}(x), not {tok} x."
            )
        if out:
            prev_kind = out[-1][0]
            if prev_kind in ("NUM", "VAR", "RP") and kind in ("VAR", "FUNC", "LP"):
                out.append(("OP", "*"))  # implicit multiplication: 2x, 2sin(x), (x+1)(x-1)
            elif prev_kind == "RP" and kind == "NUM":
                out.append(("OP", "*"))
            elif prev_kind in ("NUM", "VAR") and kind == "NUM":
                raise ValueError(
                    f"Missing an operator between '{out[-1][1]}' and '{tok}'. "
                    "Use * for multiplication, e.g. 3*4."
                )
        out.append((kind, tok))

    # pretty string: spaces around binary + - * /
    pieces = []
    for i, (kind, tok) in enumerate(out):
        unary = kind == "OP" and tok in "+-" and (
            i == 0 or out[i - 1][0] in ("OP", "LP")
        )
        if kind == "OP" and tok in "+-*/" and not unary:
            pieces.append(f" {tok} ")
        else:
            pieces.append(tok)
    return "".join(pieces)


def normalize_expression(expr):
    """Public helper: how the app reads what the user typed (explicit * and ^)."""
    return _normalize(expr)


def _prepare_expression(expr):
    """Clean user text and return its AST."""
    cleaned = _normalize(expr).replace("^", "**")
    try:
        return ast.parse(cleaned, mode="eval")
    except SyntaxError:
        raise ValueError(
            "Could not read the expression. Check the parentheses and operators, "
            "for example: 2*x + sin(x)."
        )


def parse_function(expr, angle="rad"):
    """Turn text like 'x^3 - 2x - 5' or 'e^(-x) = x' into a callable f(x).

    angle: 'rad' (default) or 'deg' for sin, cos, tan, ... and their inverses.
    """
    funcs = _FUNCS_DEG if angle == "deg" else _FUNCS_RAD
    tree = _prepare_expression(expr)
    _validate_node(tree, funcs)

    def f(x):
        try:
            return float(_eval_node(tree, float(x), funcs))
        except ZeroDivisionError:
            raise ValueError(f"Division by zero while evaluating f({x:g}).")
        except OverflowError:
            raise ValueError(f"Overflow while evaluating f({x:g}).")
        except ValueError as e:
            raise ValueError(f"f({x:g}) is undefined ({e}).")

    return f


# -----------------------------------------------------------------------------
# POLYNOMIAL PARSER  (text -> coefficient list, highest power first)
# -----------------------------------------------------------------------------
def _p_trim(p):
    p = list(p)
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def _p_add(p, q, sign=1.0):
    n = max(len(p), len(q))
    p = p + [0.0] * (n - len(p))
    q = q + [0.0] * (n - len(q))
    return _p_trim([a + sign * b for a, b in zip(p, q)])


def _p_mul(p, q):
    out = [0.0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return _p_trim(out)


def _poly_node(node):
    """Evaluate an AST node to a coefficient list, LOWEST power first."""
    if isinstance(node, ast.Expression):
        return _poly_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return [float(node.value)]
    if isinstance(node, ast.Name):
        if node.id == "x":
            return [0.0, 1.0]
        if node.id in _CONSTS:
            return [_CONSTS[node.id]]
        raise ValueError(f"Unknown name '{node.id}'. Use only x, pi, e.")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        p = _poly_node(node.operand)
        return p if isinstance(node.op, ast.UAdd) else [-c for c in p]
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            sign = 1.0 if isinstance(node.op, ast.Add) else -1.0
            return _p_add(_poly_node(node.left), _poly_node(node.right), sign)
        if isinstance(node.op, ast.Mult):
            return _p_mul(_poly_node(node.left), _poly_node(node.right))
        if isinstance(node.op, ast.Div):
            d = _poly_node(node.right)
            if len(d) != 1 or d[0] == 0:
                raise ValueError("Divide only by a nonzero number in a polynomial.")
            return [c / d[0] for c in _poly_node(node.left)]
        if isinstance(node.op, ast.Pow):
            e = _poly_node(node.right)
            if len(e) != 1 or e[0] < 0 or e[0] != int(e[0]) or e[0] > 60:
                raise ValueError(
                    "Powers must be whole numbers from 0 to 60 for a polynomial."
                )
            result = [1.0]
            base = _poly_node(node.left)
            for _ in range(int(e[0])):
                result = _p_mul(result, base)
            return result
    raise ValueError(
        "This is not a polynomial. Bairstow's method needs only x, numbers, "
        "+ - * / ^ and parentheses (no sin, exp, ln, ...)."
    )


def parse_polynomial(expr):
    """'x^5 - 3*x^4 + ...' -> [1.0, -3.0, ...] (highest power first)."""
    low_first = _p_trim(_poly_node(_prepare_expression(expr)))
    return list(reversed(low_first))


def poly_eval(coeffs, x):
    """Horner evaluation; coeffs highest power first."""
    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result


def numerical_derivative(f, x, h=1e-6):
    """Central-difference approximation of f'(x)."""
    step = h * max(1.0, abs(x))
    return (f(x + step) - f(x - step)) / (2.0 * step)


def find_brackets(f, lo, hi, subintervals=200):
    """Scan [lo, hi] and return sub-intervals [a, b] where f changes sign."""
    if hi <= lo:
        raise ValueError("Upper bound must be greater than lower bound.")
    brackets = []
    step = (hi - lo) / subintervals
    prev_x, prev_f = lo, None
    try:
        prev_f = f(prev_x)
    except ValueError:
        prev_f = None
    for i in range(1, subintervals + 1):
        x = lo + i * step
        try:
            fx = f(x)
        except ValueError:
            prev_x, prev_f = x, None
            continue
        if prev_f is not None and prev_f * fx < 0:
            brackets.append((prev_x, x))
        elif prev_f is not None and fx == 0 and i < subintervals:
            brackets.append((x - step, x + step))
        prev_x, prev_f = x, fx
    return brackets


def _result(root, f, converged, message, rows, method):
    return {
        "method": method,
        "root": root,
        "f_root": f(root) if root is not None else None,
        "converged": converged,
        "message": message,
        "iterations": len(rows),
        "rows": rows,
    }


# -----------------------------------------------------------------------------
# BRACKETING METHODS
# -----------------------------------------------------------------------------
def bisection(f, a, b, tol=1e-6, max_iter=100):
    """Bisection: halve [a, b] each step, keeping the half with the sign change."""
    a, b = float(a), float(b)
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(
            f"f(a) = {fa:.4g} and f(b) = {fb:.4g} have the same sign, so [a, b] "
            "does not bracket a root. Choose a different interval."
        )
    rows, c_old = [], None
    for n in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        err = abs(c - c_old) if c_old is not None else float("nan")
        rows.append(
            {"n": n, "a": a, "b": b, "c": c, "f(a)": fa, "f(b)": fb,
             "f(c)": fc, "Error": err}
        )
        if fc == 0 or (b - a) / 2.0 < tol:
            return _result(c, f, True, "Converged.", rows, "Bisection")
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        c_old = c
    return _result(c, f, False, f"Did not converge within {max_iter} iterations.",
                   rows, "Bisection")


def false_position(f, a, b, tol=1e-6, max_iter=100):
    """False position (regula falsi): x-intercept of the chord through the bracket."""
    a, b = float(a), float(b)
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(
            f"f(a) = {fa:.4g} and f(b) = {fb:.4g} have the same sign, so [a, b] "
            "does not bracket a root. Choose a different interval."
        )
    if fa == fb:
        raise ValueError("f(a) equals f(b); the chord is horizontal.")
    rows, c_old = [], None
    for n in range(1, max_iter + 1):
        c = b - fb * (a - b) / (fa - fb)
        fc = f(c)
        err = abs(c - c_old) if c_old is not None else float("nan")
        rows.append(
            {"n": n, "a": a, "b": b, "c": c, "f(a)": fa, "f(b)": fb,
             "f(c)": fc, "Error": err}
        )
        if fc == 0 or (c_old is not None and err < tol):
            return _result(c, f, True, "Converged.", rows, "False Position")
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        c_old = c
    return _result(c, f, False, f"Did not converge within {max_iter} iterations.",
                   rows, "False Position")


# -----------------------------------------------------------------------------
# OPEN METHODS
# -----------------------------------------------------------------------------
def newton_raphson(f, x0, df=None, tol=1e-6, max_iter=100):
    """Newton-Raphson: x_{n+1} = x_n - f(x_n)/f'(x_n).

    `df` is an optional callable for f'(x); if omitted a central difference is used.
    """
    x = float(x0)
    deriv = df if df is not None else (lambda t: numerical_derivative(f, t))
    rows = []
    for n in range(1, max_iter + 1):
        fx, dfx = f(x), deriv(x)
        if abs(dfx) < 1e-14:
            rows.append({"n": n, "x_n": x, "f(x_n)": fx, "f'(x_n)": dfx,
                         "x_n+1": float("nan"), "Error": float("nan")})
            return _result(x, f, False,
                           f"Derivative is (nearly) zero at x = {x:.6g}; Newton "
                           "step is undefined. Try a different starting guess.",
                           rows, "Newton-Raphson")
        x_new = x - fx / dfx
        err = abs(x_new - x)
        rows.append({"n": n, "x_n": x, "f(x_n)": fx, "f'(x_n)": dfx,
                     "x_n+1": x_new, "Error": err})
        x = x_new
        if err < tol:
            return _result(x, f, True, "Converged.", rows, "Newton-Raphson")
    return _result(x, f, False, f"Did not converge within {max_iter} iterations.",
                   rows, "Newton-Raphson")


def secant(f, x0, x1, tol=1e-6, max_iter=100):
    """Secant: Newton's method with the derivative replaced by a finite slope."""
    x_prev, x = float(x0), float(x1)
    f_prev, fx = f(x_prev), f(x)
    rows = []
    for n in range(1, max_iter + 1):
        if fx == f_prev:
            rows.append({"n": n, "x_n-1": x_prev, "x_n": x, "f(x_n-1)": f_prev,
                         "f(x_n)": fx, "x_n+1": float("nan"),
                         "Error": float("nan")})
            return _result(x, f, False,
                           "f(x_n) equals f(x_n-1); the secant line is "
                           "horizontal. Try different starting guesses.",
                           rows, "Secant")
        x_new = x - fx * (x_prev - x) / (f_prev - fx)
        err = abs(x_new - x)
        rows.append({"n": n, "x_n-1": x_prev, "x_n": x, "f(x_n-1)": f_prev,
                     "f(x_n)": fx, "x_n+1": x_new, "Error": err})
        x_prev, f_prev = x, fx
        x, fx = x_new, f(x_new)
        if err < tol:
            return _result(x, f, True, "Converged.", rows, "Secant")
    return _result(x, f, False, f"Did not converge within {max_iter} iterations.",
                   rows, "Secant")



# -----------------------------------------------------------------------------
# BAIRSTOW'S METHOD  (all real and complex roots of a real polynomial)
# -----------------------------------------------------------------------------
def _quadratic_roots(u, v):
    """Roots of x^2 + u x + v = 0 by the quadratic formula."""
    disc = u * u - 4.0 * v
    if disc >= 0:
        r = math.sqrt(disc)
        return [(-u + r) / 2.0, (-u - r) / 2.0]
    r = math.sqrt(-disc)
    return [complex(-u / 2.0, r / 2.0), complex(-u / 2.0, -r / 2.0)]


def _bc_tables(a, u, v):
    """Synthetic division of a(x) by x^2 + ux + v, then of the b's again.

    b_0 = a_0, b_1 = a_1 - u b_0, b_k = a_k - u b_{k-1} - v b_{k-2}
    c_0 = b_0, c_1 = b_1 - u c_0, c_k = b_k - u c_{k-1} - v c_{k-2}
    Each *_u / *_v list is the row of products (-u)*prev and (-v)*prev2 that is
    added to the top row, laid out in the same columns as the hand method.
    """
    n = len(a) - 1
    b, b_u, b_v = [0.0] * (n + 1), [0.0] * (n + 1), [0.0] * (n + 1)
    for k in range(n + 1):
        b_u[k] = -u * b[k - 1] if k >= 1 else 0.0
        b_v[k] = -v * b[k - 2] if k >= 2 else 0.0
        b[k] = a[k] + b_u[k] + b_v[k]
    c, c_u, c_v = [0.0] * (n + 1), [0.0] * (n + 1), [0.0] * (n + 1)
    for k in range(n + 1):
        c_u[k] = -u * c[k - 1] if k >= 1 else 0.0
        c_v[k] = -v * c[k - 2] if k >= 2 else 0.0
        c[k] = b[k] + c_u[k] + c_v[k]
    return b, b_u, b_v, c, c_u, c_v


def bairstow_factor(a, u0, v0, tol=1e-6, max_iter=200):
    """Find ONE quadratic factor x^2 + ux + v of a(x) (a_0 = 1, a_n != 0).

    Follows the hand procedure:
      * stop when |b_n| + |b_n-1| < tol  OR  |du| + |dv| < tol
      * divergence  : |du|+|dv| at iteration 50 > the value at iteration 1
      * slow        : not converged after 100 iterations (keeps going)
      * hard stop   : after `max_iter` (200) iterations
    """
    n = len(a) - 1
    if n < 3:
        raise ValueError("Bairstow iteration needs a polynomial of degree 3 or more.")
    u, v = float(u0), float(v0)
    iters, first_step, slow = [], None, False
    status, message = "stopped", f"Iterating stopped after {max_iter} iterations."
    for i in range(1, max_iter + 1):
        b, b_u, b_v, c, c_u, c_v = _bc_tables(a, u, v)
        bn, bn1 = b[n], b[n - 1]
        row = {"iter": i, "u": u, "v": v, "a": list(a), "b": b, "b_u": b_u,
               "b_v": b_v, "c": c, "c_u": c_u, "c_v": c_v, "bn": bn,
               "bn1": bn1, "du": None, "dv": None, "det": None,
               "u_new": None, "v_new": None, "error": None}
        iters.append(row)
        if abs(bn) + abs(bn1) < tol:
            status, message = "converged", "Converged: |bₙ| + |bₙ₋₁| < tolerance."
            break
        det = c[n - 1] * c[n - 3] - c[n - 2] ** 2
        if abs(det) < 1e-14:
            status = "failed"
            message = ("The Cramer's-rule determinant is zero (singular). "
                       "Try different starting values u⁰, v⁰.")
            break
        du = (bn * c[n - 3] - c[n - 2] * bn1) / det
        dv = (c[n - 1] * bn1 - c[n - 2] * bn) / det
        err = abs(du) + abs(dv)
        row.update(du=du, dv=dv, det=det, u_new=u + du, v_new=v + dv, error=err)
        u, v = u + du, v + dv
        if i == 1:
            first_step = err
        if err < tol:
            status, message = "converged", "Converged: |Δu| + |Δv| < tolerance."
            break
        if i == 50 and first_step is not None and err > first_step:
            status = "diverged"
            message = ("Divergence occurring: |Δu|+|Δv| at iteration 50 is larger "
                       "than at iteration 1. Try different starting values.")
            break
        if i == 100:
            slow = True
    if status == "converged" and slow:
        message += " (Convergence was slow: it needed more than 100 iterations.)"
    if status == "stopped" and slow:
        message = ("Convergence is slow: the tolerance was not met after 100 "
                   f"iterations. Iterating stopped after {max_iter} iterations.")
    b = iters[-1]["b"]
    return {
        "u": u, "v": v, "status": status, "converged": status == "converged",
        "message": message, "iterations": iters,
        "quotient": b[: n - 1],  # b_0 ... b_{n-2}: the deflated polynomial
        "roots": _quadratic_roots(u, v),
    }


def bairstow(a, u0=1.8, v0=1.8, tol=1e-6, find_all=True, max_iter=200):
    """Bairstow's method driver.

    find_all=False -> return just the first quadratic factor.
    find_all=True  -> deflate and repeat until every root is found.
    """
    a = [float(c) for c in a]
    if not a or a[0] == 0:
        raise ValueError("The leading coefficient must be nonzero.")
    if len(a) < 2:
        raise ValueError("Enter a polynomial of degree 1 or higher.")
    notes = []
    if a[0] != 1.0:
        lead = a[0]
        a = [c / lead for c in a]
        notes.append(f"Divided every coefficient by a₀ = {lead:g} so that a₀ = 1.")
    roots, factors, ok = [], [], True
    poly = list(a)
    clean = lambda p: [0.0 if abs(c) < 1e-12 else c for c in p]  # noqa: E731

    if not find_all and len(poly) - 1 < 3:
        raise ValueError("Need degree 3 or higher to show an iterated quadratic "
                         "factor. Use 'All roots' for degree 1 or 2.")

    while len(poly) - 1 >= 1:
        poly = clean(poly)
        n = len(poly) - 1
        if poly[-1] == 0:  # a_n must be nonzero -> x = 0 is a root
            roots.append(0.0)
            notes.append(f"aₙ = 0, so x = 0 is a root; divided out a factor x.")
            poly = poly[:-1]
            continue
        if n == 1:
            roots.append(-poly[1] / poly[0])
            break
        if n == 2:
            roots.extend(_quadratic_roots(poly[1] / poly[0], poly[2] / poly[0]))
            break
        guesses = [(u0, v0)]
        if find_all:
            guesses += [(1.0, 1.0), (-1.0, -1.0), (0.5, -0.5), (2.0, 3.0),
                        (-2.0, 3.0), (0.1, 0.1)]
        result, tried = None, []
        for gu, gv in guesses:
            result = bairstow_factor(poly, gu, gv, tol, max_iter)
            tried.append((gu, gv))
            if result["converged"]:
                break
        if len(tried) > 1:
            result["message"] += (
                f" (Needed {len(tried)} starting guesses; last was "
                f"u⁰ = {tried[-1][0]:g}, v⁰ = {tried[-1][1]:g}.)"
            )
        result["degree"] = n
        result["poly"] = list(poly)
        factors.append(result)
        if not result["converged"]:
            ok = False
            break
        roots.extend(result["roots"])
        if not find_all:
            break
        poly = result["quotient"]
    return {"roots": roots, "factors": factors, "converged": ok, "notes": notes,
            "remaining": poly}


if __name__ == "__main__":
    g = parse_function("x^3 - 2*x - 5")
    for res in (
        bisection(g, 2, 3),
        false_position(g, 2, 3),
        newton_raphson(g, 2),
        secant(g, 2, 3),
    ):
        print(f"{res['method']:15s} root = {res['root']:.8f} "
              f"in {res['iterations']} iterations ({res['message']})")
