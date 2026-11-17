import pandas as pd
import streamlit as st
from gauss_jordan import solve_gauss_jordan
import root_finding as rf
from root_finding import (
    parse_function,
    parse_polynomial,
    poly_eval,
    normalize_expression,
    numerical_derivative,
)
from plot_utils import function_spec

# -----------------------------------------------------------------------------
# HELPER: UNICODE SUBSCRIPT GENERATOR (x₁ instead of x_1)
# -----------------------------------------------------------------------------
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

LIVE_APP_URL = "https://mlgaussjordan.streamlit.app"

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & METADATA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Numerical Methods Solver",
    layout="wide",
)

# -----------------------------------------------------------------------------
# 2. LAYOUT TWEAKS (colors come from .streamlit/config.toml)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; }
    /* let long metric values wrap instead of being cut off with "..." */
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere !important;
    }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION, INSTRUCTOR & GROUP MEMBERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Project Info")
    st.markdown("**Institution:** Davao Oriental State University")
    st.markdown("**Faculty:** Computing, Engineering and Technology")
    st.markdown("**Department:** Civil Engineering")
    st.markdown("**Linear Systems:** Gauss-Jordan Elimination")
    st.markdown(
        "**Nonlinear Equations:** Bisection, False Position, Newton-Raphson, "
        "Secant, Bairstow's Method"
    )
    st.markdown("**Stability:** Partial Pivoting + Scale-Aware Tolerance")

    st.divider()

    st.subheader("Course Instructor")
    st.markdown("• **Engr. Samuel Erespe**")

    st.subheader("Proponents & Contributors")
    st.markdown("• **John Joshua D. Ilisan**")
    st.markdown("• **Darius Lape**")
    st.markdown("• **Briel Jan M. Lacia**")

    st.divider()
    st.subheader("Roadmap")
    st.markdown(
        "**Next update:** Machine Learning module that predicts "
        "**concrete compressive strength**."
    )


# -----------------------------------------------------------------------------
# 4. TOP HERO BANNER & APPLICATION OVERVIEW
# -----------------------------------------------------------------------------
st.title("Numerical Methods Solver")
st.caption(
    "Davao Oriental State University · Department of Civil Engineering · "
    "Instructor: Engr. Samuel Erespe"
)
st.write(
    "Solve systems of linear equations with Gauss-Jordan elimination, and "
    "single equations f(x) = 0 with the bisection, false position, "
    "Newton-Raphson, secant and Bairstow methods."
)

with st.expander("About This Application & Engineering Methodology", expanded=False):
    st.markdown("### Executive Summary")
    st.markdown(
        "In engineering analysis and numerical computation, solving systems of linear "
        "equations ($Ax = b$) is a foundational requirement for structural analysis, "
        "network flow calculations, and frame modeling. This project provides a robust, "
        "transparent, first-principles implementation of **Gauss-Jordan Elimination** "
        "designed to solve general $n \\times n$ systems of linear equations without "
        "relying on external linear algebra solvers or commercial libraries."
    )

    col_desc1, col_desc2 = st.columns(2)
    with col_desc1:
        st.markdown("### Custom Gauss-Jordan Elimination Engine")
        st.markdown(
            "`gauss_jordan.py` solves general square systems using:\n\n"
            "• Scale-aware relative tolerance singularity detection\n\n"
            "• Partial pivoting for numerical stability\n\n"
            "• Full audit logging of row transformations"
        )
    with col_desc2:
        st.markdown("### Interactive Streamlit Dashboard")
        st.markdown(
            "`app.py` lets you:\n\n"
            "• Adjust matrix dimensions from $2 \\times 2$ up to $8 \\times 8$\n\n"
            "• Manually input or modify matrix coefficients\n\n"
            "• Compute exact solutions and trace every intermediate row operation step-by-step"
        )

    st.divider()
    st.markdown("### How to Use This Platform")
    st.markdown(
        "- **Linear Systems:** general $n \\times n$ solver with editable inputs "
        "and a step-by-step reduction log.\n"
        "- **Nonlinear Equations:** solve $f(x) = 0$ with Bisection, False "
        "Position, Newton-Raphson, Secant or Bairstow's Method, with a graph "
        "and full iteration tables.\n"
        "- **Mathematical Formulation:** the derivation behind the solver and "
        "its key technical features.\n"
        "- **Roadmap:** planned ML prediction of concrete compressive strength."
    )

tab1, tab_nl, tab2, tab3 = st.tabs(
    [
        "Linear Systems",
        "Nonlinear Equations",
        "Mathematical Formulation",
        "Roadmap",
    ]
)

# -----------------------------------------------------------------------------
# TAB 1: GAUSS-JORDAN MATRIX SOLVER
# -----------------------------------------------------------------------------
with tab1:
    st.header("Solve a System of Linear Equations ($Ax = b$)")
    st.caption(
        "General-purpose numerical solver powered by our custom Gauss-Jordan "
        "elimination engine."
    )

    with st.container(border=True):
        col_ctrl1, col_ctrl2 = st.columns([1, 3])
        with col_ctrl1:
            n = st.number_input(
                "Matrix Dimension (n x n):",
                min_value=2,
                max_value=8,
                value=4,
                step=1,
            )
        with col_ctrl2:
            st.info(
                "Click any cell in the table below to edit coefficients. "
                "Always press **Enter** after typing."
            )

        default_data = [
            [4.0, -1.0, 0.0, 2.0, 15.0],
            [-1.0, 5.0, -2.0, 0.0, 10.0],
            [0.0, -2.0, 6.0, -1.0, 8.0],
            [2.0, 0.0, -1.0, 4.0, 20.0],
        ]

        col_headers = [f"x{str(i+1).translate(sub)}" for i in range(n)] + [
            "Constants (b)"
        ]

        if n == 4:
            df_init = pd.DataFrame(default_data, columns=col_headers)
        else:
            df_init = pd.DataFrame(0.0, index=range(n), columns=col_headers)

        edited_df = st.data_editor(
            df_init,
            use_container_width=True,
            num_rows="fixed",
            key=f"matrix_editor_{n}",
        )

        solve_clicked = st.button(
            "Solve with Gauss-Jordan", type="primary", use_container_width=True
        )

    if solve_clicked:
        try:
            matrix_vals = edited_df.values
            A = matrix_vals[:, :n].tolist()
            b = matrix_vals[:, n].tolist()

            solution, steps = solve_gauss_jordan(A, b, return_steps=True)

            st.success("System successfully reduced to identity matrix!")

            cols = st.columns(n)
            for i in range(n):
                with cols[i]:
                    var_name = f"x{str(i+1).translate(sub)}"
                    st.metric(label=var_name, value=f"{solution[i]:.4f}")

            st.divider()
            st.subheader("Step-by-Step Row Operations")
            st.caption(f"Reduced in {len(steps)} systematic transformations.")

            for idx, step_info in enumerate(steps):
                with st.expander(
                    f"Step {idx+1}: {step_info['title']}", expanded=(idx == 0)
                ):
                    st.markdown(f"**Action:** `{step_info['operation']}`")
                    step_df = pd.DataFrame(
                        step_info["matrix"], columns=col_headers
                    )
                    st.dataframe(
                        step_df.style.format("{:.4f}"), use_container_width=True
                    )

        except Exception as e:
            st.error(f"Solver Error: {e}")

# -----------------------------------------------------------------------------
# TAB: NONLINEAR EQUATIONS / ROOT FINDING
# -----------------------------------------------------------------------------
def _vm(a, b, c, d):
    """LaTeX 2x2 determinant with numbers filled in."""
    return (
        r"\begin{vmatrix} " + f"{a:.4f} & {b:.4f} " + r"\\ " + f"{c:.4f} & {d:.4f} "
        + r"\end{vmatrix}"
    )


def _fmt_complex(z):
    z = complex(z)
    if abs(z.imag) < 1e-9 * max(1.0, abs(z.real)):
        return f"{z.real:.6f}"
    sign = "+" if z.imag >= 0 else "−"
    return f"{z.real:.6f} {sign} {abs(z.imag):.6f}i"


def _bairstow_table(it, n):
    """Build the hand-method Bairstow table (a's / b's / c's) for one iteration."""
    cols = [f"a{str(k).translate(sub)}" for k in range(n + 1)]
    rows = {
        "a's": it["a"],
        "(−u)·b  [k−1]": it["b_u"],
        "(−v)·b  [k−2]": it["b_v"],
        "b's": it["b"],
    }
    if it["du"] is not None:
        rows["(−u)·c  [k−1]"] = it["c_u"]
        rows["(−v)·c  [k−2]"] = it["c_v"]
        rows["c's"] = it["c"]
    return pd.DataFrame(rows, index=cols).T


def _auto_range(method, p, res):
    """Pick a sensible graph window from the inputs (and the result, if any)."""
    if method in ("bisection", "false_position"):
        lo, hi = min(p["a"], p["b"]), max(p["a"], p["b"])
        pad = max((hi - lo) * 0.15, 1e-9)
        return lo - pad, hi + pad
    if method == "bairstow":
        pts = []
        if res is not None:
            pts = [complex(r).real for r in res["roots"]]
        if not pts:
            return -5.0, 5.0
        lo, hi = min(pts), max(pts)
        half = max((hi - lo) * 0.6, 2.5)
        mid = (lo + hi) / 2
        return mid - half, mid + half
    pts = [p["x0"]] + ([p["x1"]] if method == "secant" else [])
    if res is not None:
        pts += [res["root"]] + [r["x_n"] for r in res["rows"]]
    lo, hi = min(pts), max(pts)
    half = max((hi - lo) / 2 * 1.6, 1.0)
    mid = (lo + hi) / 2
    return mid - half, mid + half


with tab_nl:
    st.header("Solve a Single Equation $f(x) = 0$")
    st.caption(
        "Works for linear and nonlinear equations. Choose a bracketing method "
        "(needs an interval with a sign change), an open method (needs starting "
        "guesses), or Bairstow's method (all real and complex roots of a polynomial)."
    )

    METHODS = {
        "Bisection (Bracketing)": "bisection",
        "False Position / Regula Falsi (Bracketing)": "false_position",
        "Newton-Raphson (Open)": "newton",
        "Secant (Open)": "secant",
        "Bairstow's Method (Polynomials: real & complex roots)": "bairstow",
    }

    # ---- widget defaults (set via session_state so example buttons can change them)
    for _k, _v in {
        "method_label": "Bisection (Bracketing)",
        "fx_general": "x^3 - 2x - 5",
        "fx_poly": "x^5 - 3x^4 - 10x^3 + 10x^2 + 44x + 48",
        "newton_df": "",
        "angle_mode": "Radians",
        "bair_mode": "Find one quadratic factor",
        "p_a": 2.0, "p_b": 3.0, "p_x0": 2.0, "p_x1": 3.0,
        "p_u0": 1.8, "p_v0": 1.8,
    }.items():
        st.session_state.setdefault(_k, _v)

    # Each example fills in the function, the best method and its inputs.
    EXAMPLES = [
        {"name": "e⁻ˣ = x  (exponential)", "fx": "e^(-x) = x",
         "method": "Bisection (Bracketing)", "p": {"p_a": 0.0, "p_b": 1.0}},
        {"name": "x·eˣ = 2  (exponential)", "fx": "x*e^x = 2",
         "method": "Secant (Open)", "p": {"p_x0": 0.0, "p_x1": 1.0}},
        {"name": "2ˣ = 10  (exponential)", "fx": "2^x = 10",
         "method": "Newton-Raphson (Open)", "p": {"p_x0": 3.0}},
        {"name": "cos x = x  (trigonometric)", "fx": "cos(x) = x",
         "method": "False Position / Regula Falsi (Bracketing)",
         "p": {"p_a": 0.0, "p_b": 1.0}},
        {"name": "2 sin x = x  (trigonometric)", "fx": "2sin(x) = x",
         "method": "Newton-Raphson (Open)", "p": {"p_x0": 2.0}},
        {"name": "sin x = e⁻ˣ  (mixed)", "fx": "sin(x) = e^(-x)",
         "method": "Bisection (Bracketing)", "p": {"p_a": 0.0, "p_b": 1.0}},
        {"name": "ln x + x = 2  (logarithmic)", "fx": "ln(x) + x = 2",
         "method": "Bisection (Bracketing)", "p": {"p_a": 1.0, "p_b": 2.0}},
        {"name": "sin x = 0.5  (in degrees → 30°)", "fx": "sin(x) = 0.5",
         "method": "Bisection (Bracketing)",
         "p": {"p_a": 0.0, "p_b": 90.0, "angle_mode": "Degrees"}},
    ]
    POLY_EXAMPLES = [
        {"name": "Degree 5 (sample problem)",
         "fx": "x^5 - 3x^4 - 10x^3 + 10x^2 + 44x + 48"},
        {"name": "x⁴ + 4  (all complex roots)", "fx": "x^4 + 4"},
        {"name": "Four real roots (1, 2, 3, 4)", "fx": "x^4 - 10x^3 + 35x^2 - 50x + 24"},
    ]

    def _use_example(ex):
        st.session_state["method_label"] = ex["method"]
        st.session_state["fx_general"] = ex["fx"]
        st.session_state["newton_df"] = ""
        st.session_state["angle_mode"] = "Radians"
        for k_, v_ in ex["p"].items():
            st.session_state[k_] = v_

    def _use_poly_example(ex):
        st.session_state["method_label"] = (
            "Bairstow's Method (Polynomials: real & complex roots)"
        )
        st.session_state["fx_poly"] = ex["fx"]
        st.session_state["p_u0"] = 1.8
        st.session_state["p_v0"] = 1.8

    with st.expander("Tutorial: how to type your function", expanded=False):
        st.markdown(
            "Type $f(x)$ the way you would on a calculator, using **x** as the "
            "variable. You can type just the expression (the app solves "
            "$f(x) = 0$) or a whole equation such as `cos(x) = x`."
        )
        t1, t2 = st.tabs(["What to type", "Try an example"])
        with t1:
            st.markdown(
                """
| What you mean | What to type |
| :--- | :--- |
| 3x² + 2 | `3x^2 + 2`  (or `3*x^2 + 2`) |
| x³ | `x^3` |
| eˣ | `e^x`  or  `exp(x)` |
| e⁻ˣ | `e^(-x)`  or  `exp(-x)` |
| e²ˣ | `e^(2x)`  (use brackets for the whole exponent) |
| 2ˣ | `2^x` |
| sin x | `sin(x)` |
| 2 sin x | `2sin(x)`  or  `2*sin(x)` |
| sin²x | `sin(x)^2` |
| cos 2x | `cos(2x)` |
| tan x | `tan(x)` |
| ln x (natural log) | `ln(x)` |
| log₁₀ x | `log10(x)` |
| log₂ x | `log2(x)` |
| √x | `sqrt(x)` |
| ³√x | `cbrt(x)` |
| \\|x\\| | `abs(x)` |
| π | `pi` |
| (x + 1)(x − 1) | `(x+1)(x-1)` |
| (x + 1) ÷ (x − 1) | `(x+1)/(x-1)` |
| 5 × 10⁻³ | `5e-3` |
| cos x = x | `cos(x) = x`  (solved as cos x − x = 0) |
"""
            )
            st.markdown(
                "**Available functions:** `sin cos tan sec csc cot asin acos atan "
                "sinh cosh tanh exp ln log log10 log2 sqrt cbrt abs`  ·  "
                "**Constants:** `pi`, `e`"
            )
            st.markdown("**Good to know**")
            st.markdown(
                "- Always put a function's input in brackets: `sin(x)`, not `sin x`.\n"
                "- `^` means \"to the power of\". Use brackets when the exponent has "
                "more than one part: `e^(2*x)` is e²ˣ, but `e^2*x` means "
                "e² · x.\n"
                "- `log(x)` is the **natural** log (same as `ln`). For base 10 use "
                "`log10(x)`.\n"
                "- sin, cos and tan use **radians** by default. Switch **Angle mode** "
                "to *Degrees* below if your problem is in degrees.\n"
                "- The line **Reads as** under the box shows exactly how the app "
                "understood what you typed. Check it if the answer looks wrong.\n"
                "- Exponential and trig functions can have **many roots**. Use the "
                "graph (or the sign-change scanner at the bottom) to pick a good "
                "interval or starting guess.\n"
                "- Bracketing methods need $f(a)$ and $f(b)$ to have opposite signs. "
                "Careful with `tan(x)`: it jumps from +∞ to −∞ at its asymptotes, "
                "which is not a root. Check that $f(x^*)$ is close to 0.\n"
                "- Bairstow's method accepts **polynomials only** (no sin, exp, ln)."
            )
            st.markdown("**Common mistakes**")
            st.markdown(
                """
| Wrong | Right | Why |
| :--- | :--- | :--- |
| `sin x` | `sin(x)` | functions need brackets |
| `e^2x` | `e^(2x)` | otherwise it means e² · x |
| `x²` | `x^2` | type ^ for powers |
| `√x` | `sqrt(x)` | write the function name |
| `3 4` | `3*4` | an operator is needed |
| `x2` | `x^2` or `2x` | `x2` is not understood |
"""
            )
        with t2:
            st.caption(
                "Click an example. It fills in the function, picks a suitable "
                "method and sets the inputs for you. Then press the solve button."
            )
            ec = st.columns(2)
            for i, ex in enumerate(EXAMPLES):
                with ec[i % 2]:
                    st.button(
                        ex["name"], key=f"ex_{i}", use_container_width=True,
                        on_click=_use_example, args=(ex,), help=f"f(x): {ex['fx']}",
                    )
            st.markdown("**Polynomials (Bairstow's method)**")
            pc_ = st.columns(3)
            for i, ex in enumerate(POLY_EXAMPLES):
                with pc_[i % 3]:
                    st.button(
                        ex["name"], key=f"pex_{i}", use_container_width=True,
                        on_click=_use_poly_example, args=(ex,),
                        help=f"f(x): {ex['fx']}",
                    )

    with st.container(border=True):
        method_label = st.selectbox("Method:", list(METHODS.keys()), key="method_label")
        method = METHODS[method_label]

        if method == "bairstow":
            fx_text = st.text_input(
                "Polynomial f(x) =",
                key="fx_poly",
                help="Any polynomial with real coefficients, e.g. "
                "x^5 - 3x^4 - 10x^3 + 10x^2 + 44x + 48. Open the tutorial above "
                "for typing tips.",
            )
        else:
            fx_text = st.text_input(
                "f(x) =",
                key="fx_general",
                help="Examples: x^3 - 2x - 5   ·   e^(-x) = x   ·   2sin(x) - x   ·   "
                "ln(x) + x = 2. Open the tutorial above for the full list.",
            )
            angle_choice = st.radio(
                "Angle mode for sin, cos, tan, …:",
                ["Radians", "Degrees"],
                horizontal=True,
                key="angle_mode",
            )

        try:
            st.caption(f"Reads as:  `f(x) = {normalize_expression(fx_text)}`")
        except ValueError as e_:
            st.caption(f"{e_}")

        p = {}
        p1, p2, p3 = st.columns(3)
        df_text = ""
        if method in ("bisection", "false_position"):
            with p1:
                p["a"] = st.number_input("Lower bound a:", format="%.6f", key="p_a")
            with p2:
                p["b"] = st.number_input("Upper bound b:", format="%.6f", key="p_b")
        elif method == "newton":
            with p1:
                p["x0"] = st.number_input("Initial guess x₀:", format="%.6f", key="p_x0")
            with p2:
                df_text = st.text_input(
                    "f'(x) (optional):",
                    key="newton_df",
                    help="Leave blank (recommended) to use a numerical "
                    "derivative. If you type f'(x) it must match your f(x).",
                )
        elif method == "secant":
            with p1:
                p["x0"] = st.number_input("First guess x₀:", format="%.6f", key="p_x0")
            with p2:
                p["x1"] = st.number_input("Second guess x₁:", format="%.6f", key="p_x1")
        else:  # bairstow
            with p1:
                p["u0"] = st.number_input("Initial u⁰:", format="%.4f", key="p_u0")
            with p2:
                p["v0"] = st.number_input("Initial v⁰:", format="%.4f", key="p_v0")

        with p3:
            tol_in = st.selectbox(
                "Tolerance:",
                [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12],
                index=5,
                format_func=lambda v: f"{v:.0e}",
            )

        if method == "bairstow":
            bair_mode = st.radio(
                "What to compute:",
                ["Find one quadratic factor", "Find all roots (deflate and repeat)"],
                horizontal=True,
                key="bair_mode",
            )
            st.caption(
                "Iteration limit follows the method notes: warns of divergence at "
                "iteration 50, of slow convergence at 100, and stops at 200."
            )
            max_iter_in = 200
        else:
            max_iter_in = st.number_input(
                "Maximum iterations:", min_value=1, max_value=500, value=100, step=1
            )

        with st.expander("Graph range (optional)"):
            auto_range = st.checkbox("Choose the range automatically", value=True)
            g1, g2 = st.columns(2)
            man_lo = g1.number_input("x min:", value=-5.0, key="g_lo")
            man_hi = g2.number_input("x max:", value=5.0, key="g_hi")

        solve_nl = st.button(
            "Find Root" if method != "bairstow" else "Solve with Bairstow's Method",
            type="primary",
            use_container_width=True,
            key="solve_nl",
        )

    angle = "deg" if (method != "bairstow" and angle_choice == "Degrees") else "rad"

    # ---- run the selected method (only when the button is pressed) ---------
    res, solver_error = None, None
    if solve_nl:
        try:
            if method == "bairstow":
                coeffs = parse_polynomial(fx_text)
                res = rf.bairstow(
                    coeffs, p["u0"], p["v0"], tol_in,
                    find_all=bair_mode.startswith("Find all"),
                )
            else:
                f_solve = parse_function(fx_text, angle)
                if method == "bisection":
                    res = rf.bisection(f_solve, p["a"], p["b"], tol_in, int(max_iter_in))
                elif method == "false_position":
                    res = rf.false_position(f_solve, p["a"], p["b"], tol_in, int(max_iter_in))
                elif method == "newton":
                    d = None
                    if df_text.strip():
                        if angle == "deg":
                            st.info(
                                "Degrees mode: your typed f'(x) is ignored and a "
                                "numerical derivative is used instead."
                            )
                        else:
                            d = parse_function(df_text)
                            expected = numerical_derivative(f_solve, p["x0"])
                            if abs(d(p["x0"]) - expected) > 1e-3 * max(1.0, abs(expected)):
                                st.warning(
                                    "Your f'(x) does not match f(x) at x₀ "
                                    f"(typed: {d(p['x0']):.6g}, numerical: "
                                    f"{expected:.6g}). Check it, or clear the box "
                                    "to use a numerical derivative."
                                )
                    res = rf.newton_raphson(f_solve, p["x0"], d, tol_in, int(max_iter_in))
                else:
                    res = rf.secant(f_solve, p["x0"], p["x1"], tol_in, int(max_iter_in))
        except Exception as e:
            solver_error = str(e)
            st.error(f"Solver Error: {e}")

    # ---- graph of f(x): live preview while typing, markers once solved -----
    st.subheader("Graph of f(x)")
    try:
        if method == "bairstow":
            poly_coeffs = parse_polynomial(fx_text)
            f_plot = lambda x: poly_eval(poly_coeffs, x)  # noqa: E731
        else:
            f_plot = parse_function(fx_text, angle)

        lo, hi = (man_lo, man_hi) if not auto_range else _auto_range(method, p, res)

        iterates, roots_mark, band = [], [], None
        if method in ("bisection", "false_position"):
            band = (min(p["a"], p["b"]), max(p["a"], p["b"]))
        if res is not None and method != "bairstow":
            key = "c" if method in ("bisection", "false_position") else "x_n"
            for r_ in res["rows"][:60]:
                try:
                    iterates.append((r_[key], f_plot(r_[key]), f"iteration {r_['n']}"))
                except ValueError:
                    pass
            if res["converged"]:
                roots_mark = [res["root"]]
        elif res is not None:  # bairstow: real roots cross the x-axis
            roots_mark = [
                complex(r_).real for r_ in res["roots"]
                if abs(complex(r_).imag) < 1e-9 * max(1.0, abs(complex(r_).real))
            ]

        st.vega_lite_chart(
            function_spec(f_plot, lo, hi, iterates, roots_mark, band),
            use_container_width=True,
        )
        legend = "Blue curve = f(x). Dashed line = the x-axis (f = 0)."
        if band:
            legend += " Shaded area = starting interval [a, b]."
        if res is not None:
            legend += " Orange dots = iterations; red diamonds = roots on the x-axis."
        else:
            legend += " Press the button to mark the root and the iterations."
        if method == "bairstow":
            legend += " Complex roots do not cross the x-axis, so they are listed below instead."
        st.caption(legend)
    except Exception as e:
        st.warning(f"Graph not shown yet: {e}")

    # ---- results ------------------------------------------------------------
    if res is not None and method != "bairstow":
        if res["converged"]:
            st.success(f"{res['method']} converged in {res['iterations']} iterations.")
        else:
            st.warning(f"{res['message']}")

        if res["converged"] and abs(res["f_root"]) > max(1e-3, 100 * tol_in):
            st.warning(
                "f(x*) is not close to 0, so this may be a jump (discontinuity) "
                "such as an asymptote of tan(x) or 1/x, not a true root. "
                "Check the graph."
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Approximate root x*", f"{res['root']:.8f}")
        m2.metric("f(x*)", f"{res['f_root']:.3e}")
        m3.metric("Iterations", res["iterations"])

        st.divider()
        st.subheader("Iteration Table")
        table = pd.DataFrame(res["rows"]).rename(columns={"n": "Iter"})
        st.dataframe(
            table.style.format(
                {c: "{:.8f}" for c in table.columns if c != "Iter"}, na_rep="—"
            ),
            use_container_width=True,
            hide_index=True,
        )

    if res is not None and method == "bairstow":
        for note in res["notes"]:
            st.info(note)

        all_roots = res["roots"]
        if res["converged"]:
            st.success(
                f"Bairstow's method finished: {len(res['factors'])} quadratic "
                f"factor(s) extracted, {len(all_roots)} root(s) found."
            )
        else:
            st.warning(res["factors"][-1]["message"])

        if all_roots:
            st.subheader("Roots")
            root_rows = []
            for i, z in enumerate(all_roots, start=1):
                z = complex(z)
                is_real = abs(z.imag) < 1e-9 * max(1.0, abs(z.real))
                root_rows.append(
                    {"Root": f"x{str(i).translate(sub)}", "Real part": z.real,
                     "Imaginary part": 0.0 if is_real else z.imag,
                     "Type": "Real" if is_real else "Complex"}
                )
            st.dataframe(
                pd.DataFrame(root_rows).style.format(
                    {"Real part": "{:.6f}", "Imaginary part": "{:.6f}"}
                ),
                use_container_width=True, hide_index=True,
            )

        for k, fac in enumerate(res["factors"], start=1):
            n = fac["degree"]
            st.divider()
            st.subheader(f"Quadratic Factor #{k}  (from a degree-{n} polynomial)")
            poly_txt = " + ".join(
                f"({c:.4g})x^{n - i}" for i, c in enumerate(fac["poly"])
            )
            st.caption(f"Working polynomial: {poly_txt}")

            if fac["converged"]:
                st.success(fac["message"])
            else:
                st.warning(fac["message"])

            q1, q2, q3, q4 = st.columns(4)
            q1.metric("u", f"{fac['u']:.6f}")
            q2.metric("v", f"{fac['v']:.6f}")
            q3.metric("Iterations", len(fac["iterations"]))
            q4.metric("Roots of factor", "  ,  ".join(_fmt_complex(z) for z in fac["roots"]))

            st.latex(
                rf"x^2 + ({fac['u']:.4f})x + ({fac['v']:.4f})"
                rf"\;\Rightarrow\; x_{{1,2}} = \frac{{-u \pm \sqrt{{u^2 - 4v}}}}{{2}}"
            )
            if fac["converged"] and len(fac["quotient"]) > 1:
                st.markdown(
                    "Remaining polynomial of degree "
                    f"**{n - 2}** (the b-coefficients): "
                    + ", ".join(f"`{c:.4f}`" for c in fac["quotient"])
                )

            # compact convergence summary
            summary = pd.DataFrame(
                [
                    {"Iter": it["iter"], "u": it["u"], "v": it["v"],
                     "bₙ": it["bn"], "bₙ₋₁": it["bn1"], "Δu": it["du"],
                     "Δv": it["dv"], "|Δu|+|Δv|": it["error"]}
                    for it in fac["iterations"]
                ]
            )
            st.dataframe(
                summary.style.format(
                    {c: "{:.6f}" for c in summary.columns if c != "Iter"}, na_rep="—"
                ),
                use_container_width=True, hide_index=True,
            )

            # step-by-step Bairstow tables, like the hand method
            st.markdown("**Step-by-step Bairstow tables**")
            its = fac["iterations"]
            shown = list(range(len(its)))
            if len(its) > 15:
                shown = list(range(10)) + list(range(len(its) - 5, len(its)))
                st.caption(
                    f"Showing the first 10 and last 5 of {len(its)} iterations "
                    "(the summary table above lists all of them)."
                )
            for idx in shown:
                it = its[idx]
                with st.expander(
                    f"Bairstow Table {it['iter']}   ·   u = {it['u']:.4f}, "
                    f"v = {it['v']:.4f}",
                    expanded=(idx == 0),
                ):
                    st.caption(
                        f"Rows use −u = {-it['u']:.4f} and −v = {-it['v']:.4f}:  "
                        "bₖ = aₖ − u·bₖ₋₁ − v·bₖ₋₂,   cₖ = bₖ − u·cₖ₋₁ − v·cₖ₋₂"
                    )
                    st.dataframe(
                        _bairstow_table(it, n).style.format("{:.4f}"),
                        use_container_width=True,
                    )
                    if it["du"] is None:
                        if abs(it["bn"]) + abs(it["bn1"]) < tol_in:
                            st.markdown(
                                f"**Stop:** |bₙ| + |bₙ₋₁| = "
                                f"{abs(it['bn']) + abs(it['bn1']):.2e} < ε = {tol_in:.0e}"
                            )
                        else:
                            st.markdown("**Stopped:** singular determinant.")
                    else:
                        bn, bn1 = it["bn"], it["bn1"]
                        c = it["c"]
                        c1, c2, c3 = c[n - 1], c[n - 2], c[n - 3]
                        num_u = bn * c3 - c2 * bn1
                        num_v = c1 * bn1 - c2 * bn
                        st.latex(
                            r"\Delta u = \frac{" + _vm(bn, c2, bn1, c3) + "}{"
                            + _vm(c1, c2, c2, c3) + "}"
                            + rf" = \frac{{{num_u:.4f}}}{{{it['det']:.4f}}} = {it['du']:.4f}"
                        )
                        st.latex(
                            r"\Delta v = \frac{" + _vm(c1, bn, c2, bn1) + "}{"
                            + _vm(c1, c2, c2, c3) + "}"
                            + rf" = \frac{{{num_v:.4f}}}{{{it['det']:.4f}}} = {it['dv']:.4f}"
                        )
                        st.markdown(
                            f"u' = {it['u']:.4f} + ({it['du']:.4f}) = **{it['u_new']:.4f}**"
                            f"  ·  v' = {it['v']:.4f} + ({it['dv']:.4f}) = "
                            f"**{it['v_new']:.4f}**"
                        )

    with st.expander("Not sure where the roots are? Scan for sign changes"):
        s1, s2 = st.columns(2)
        scan_lo = s1.number_input("Scan from:", value=-5.0, key="scan_lo")
        scan_hi = s2.number_input("Scan to:", value=5.0, key="scan_hi")
        if st.button("Scan interval", key="scan_btn"):
            try:
                if method == "bairstow":
                    pc = parse_polynomial(fx_text)
                    scan_f = lambda x: poly_eval(pc, x)  # noqa: E731
                else:
                    scan_f = parse_function(fx_text, angle)
                found = rf.find_brackets(scan_f, scan_lo, scan_hi)
                if found:
                    st.table(pd.DataFrame(found, columns=["a", "b"]).round(4))
                    st.caption("Each row is an interval containing at least one root.")
                else:
                    st.info("No sign change found in this range.")
            except Exception as e:
                st.error(f"Scan Error: {e}")

    st.divider()
    st.subheader("Method Formulas")
    fc1, fc2 = st.columns(2)
    with fc1:
        with st.container(border=True):
            st.markdown("#### Bisection")
            st.latex(r"c = \dfrac{a + b}{2}")
            st.markdown("Keep the half-interval where $f$ changes sign.")
        with st.container(border=True):
            st.markdown("#### False Position (Regula Falsi)")
            st.latex(r"c = b - \dfrac{f(b)\,(a - b)}{f(a) - f(b)}")
            st.markdown("The $x$-intercept of the chord through $(a, f(a))$ and $(b, f(b))$.")
    with fc2:
        with st.container(border=True):
            st.markdown("#### Newton-Raphson")
            st.latex(r"x_{n+1} = x_n - \dfrac{f(x_n)}{f'(x_n)}")
            st.markdown("Fast (quadratic) near a root, but fails if $f'(x_n) \\approx 0$.")
        with st.container(border=True):
            st.markdown("#### Secant")
            st.latex(
                r"x_{n+1} = x_n - \dfrac{f(x_n)\,(x_{n-1} - x_n)}{f(x_{n-1}) - f(x_n)}"
            )
            st.markdown("Like Newton, but replaces $f'(x_n)$ with a finite-difference slope.")

    with st.container(border=True):
        st.markdown("#### Bairstow's Method (quadratic factoring)")
        st.markdown(
            "Finds a quadratic factor $x^2 + ux + v$ of $f(x) = a_0x^n + \\dots + a_n$ "
            "($a_0 = 1$, $a_n \\neq 0$), starting from guesses $u^0, v^0$. Its two roots "
            "come from the quadratic formula; the remaining degree $n-2$ polynomial is "
            "factored the same way until all roots are found."
        )
        st.latex(
            r"b_0 = a_0,\quad b_1 = a_1 - u\,b_0,\quad "
            r"b_k = a_k - u\,b_{k-1} - v\,b_{k-2}"
        )
        st.latex(
            r"c_0 = b_0,\quad c_1 = b_1 - u\,c_0,\quad "
            r"c_k = b_k - u\,c_{k-1} - v\,c_{k-2}"
        )
        st.markdown(
            "Choose $\\Delta u, \\Delta v$ so the remainder terms vanish "
            "(Taylor series, with $\\partial b_k/\\partial u = -c_{k-1}$ and "
            "$\\partial b_k/\\partial v = -c_{k-2}$):"
        )
        st.latex(
            r"b_n = c_{n-1}\Delta u + c_{n-2}\Delta v,\qquad "
            r"b_{n-1} = c_{n-2}\Delta u + c_{n-3}\Delta v"
        )
        st.latex(
            r"\Delta u = \frac{\begin{vmatrix} b_n & c_{n-2} \\ b_{n-1} & c_{n-3} "
            r"\end{vmatrix}}{\begin{vmatrix} c_{n-1} & c_{n-2} \\ c_{n-2} & c_{n-3} "
            r"\end{vmatrix}},\qquad "
            r"\Delta v = \frac{\begin{vmatrix} c_{n-1} & b_n \\ c_{n-2} & b_{n-1} "
            r"\end{vmatrix}}{\begin{vmatrix} c_{n-1} & c_{n-2} \\ c_{n-2} & c_{n-3} "
            r"\end{vmatrix}}"
        )
        st.markdown(
            "Repeat with $u \\leftarrow u + \\Delta u$, $v \\leftarrow v + \\Delta v$ until "
            "$|b_n| + |b_{n-1}| < \\varepsilon$ or $|\\Delta u| + |\\Delta v| < \\varepsilon$. "
            "**Divergence** if $|\\Delta u_{50}| + |\\Delta v_{50}| > |\\Delta u_1| + |\\Delta v_1|$; "
            "**slow convergence** after 100 iterations; stop after 200."
        )


# -----------------------------------------------------------------------------
# TAB 2: MATHEMATICAL FORMULATION (from README)
# -----------------------------------------------------------------------------
with tab2:
    st.header("Mathematical Formulation")

    st.subheader("1. General System of Linear Equations")
    st.markdown(
        "A system of $n$ linear equations with $n$ unknown variables is expressed "
        "in matrix notation as:"
    )
    st.latex(r"Ax = b")
    st.latex(
        r"""
        \begin{bmatrix}
        a_{11} & a_{12} & \cdots & a_{1n} \\
        a_{21} & a_{22} & \cdots & a_{2n} \\
        \vdots & \vdots & \ddots & \vdots \\
        a_{n1} & a_{n2} & \cdots & a_{nn}
        \end{bmatrix}
        \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}
        =
        \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{bmatrix}
        """
    )
    st.markdown(
        "The goal is to determine the unknown vector "
        "$x = [x_1, x_2, \\dots, x_n]^T$ by reducing the augmented matrix "
        "$[A \\mid b]$ into reduced row echelon form (RREF) $[I \\mid x]$, where "
        "$I$ is the $n \\times n$ identity matrix."
    )

    st.divider()
    st.subheader("2. Scale-Aware Gauss-Jordan Elimination with Partial Pivoting")

    with st.container(border=True):
        st.markdown("#### Step 1: Augmented Matrix Setup")
        st.markdown(
            "The coefficient matrix $A$ and constant vector $b$ are combined into an "
            "augmented matrix $M$ of dimension $n \\times (n+1)$:"
        )
        st.latex(r"M = [A \mid b]")

    with st.container(border=True):
        st.markdown("#### Step 2: Partial Pivoting")
        st.markdown(
            "To maintain numerical stability and avoid division by zero (or small "
            "numbers that introduce roundoff errors), row swaps are performed at each "
            "column $k$:"
        )
        st.latex(
            r"\text{Find } r \text{ such that } \lvert M_{r,k} \rvert = "
            r"\max_{i \ge k} \lvert M_{i,k} \rvert"
        )
        st.markdown("If $r \\neq k$, Row $k$ and Row $r$ are swapped.")

    with st.container(border=True):
        st.markdown("#### Step 3: Scale-Aware Singularity Detection")
        st.markdown(
            "Rather than relying on a rigid fixed tolerance, the solver establishes a "
            "relative singularity threshold based on the overall magnitude of the "
            "working matrix:"
        )
        st.latex(
            r"\text{threshold} = \max\left(\text{tol} \times \max(\lvert M \rvert),"
            r"\ 10^{-14}\right), \quad \text{tol} = 10^{-10}"
        )
        st.markdown(
            "If the pivot value $\\lvert M_{k,k} \\rvert < \\text{threshold}$, the "
            "system is flagged as singular or ill-conditioned, and execution halts "
            "cleanly with an informative explanation."
        )

    with st.container(border=True):
        st.markdown("#### Step 4: Row Normalization & Elementary Transformations")
        st.markdown(
            "**Normalize Pivot Row:** divide the entire pivot row by the diagonal "
            "entry $M_{k,k}$ so that $M_{k,k} = 1$:"
        )
        st.latex(
            r"M_{k,j} \leftarrow \frac{M_{k,j}}{M_{k,k}}"
            r"\quad \text{for } j = k, \dots, n"
        )
        st.markdown(
            "**Eliminate Non-Zero Column Entries:** zero out all off-diagonal entries "
            "in column $k$ across all other rows $i \\neq k$:"
        )
        st.latex(
            r"M_{i,j} \leftarrow M_{i,j} - (M_{i,k} \times M_{k,j})"
            r"\quad \text{for all } i \neq k"
        )
        st.markdown(
            "Upon completing elimination across all $n$ columns, the solution vector "
            "is directly read from the final column:"
        )
        st.latex(r"x_i = M_{i,n+1} \quad \text{for } i = 1, 2, \dots, n")

    st.divider()
    st.subheader("Key Technical Features")
    features_df = pd.DataFrame(
        {
            "Feature": [
                "Partial Pivoting",
                "Scale-Aware Singularity Checking",
                "Auditable Transformation Logging",
                "Flexible System Sizing",
            ],
            "Description": [
                "Interchanges rows to place the absolute largest available pivot on the diagonal.",
                "Evaluates pivots against a dynamic threshold based on matrix magnitude (tol × max|M|).",
                "Records every intermediate matrix state and human-readable row operation formula.",
                "Dynamically scales from 2×2 up to 8×8 linear systems via interactive UI controls.",
            ],
            "Engineering Benefit": [
                "Prevents division by zero and minimizes floating-point precision degradation.",
                "Prevents false singular rejections on high-magnitude matrices while catching ill-conditioned systems.",
                "Serves as an educational step-by-step calculation trace for verification.",
                "Accommodates a wide variety of structural and linear algebra problems.",
            ],
        }
    )
    st.dataframe(features_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 3: ROADMAP - UPCOMING ML CONCRETE STRENGTH PREDICTION
# -----------------------------------------------------------------------------
with tab3:
    st.caption("Planned for the next update (not yet available)")
    st.header("Roadmap: Machine Learning for Concrete Strength Prediction")
    st.markdown(
        "In the **next update**, this application will include a **Machine Learning "
        "module that predicts the compressive strength of concrete**."
    )

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        with st.container(border=True):
            st.markdown("### The Civil Engineering Problem")
            st.markdown(
                "In structural design and construction quality assurance, the "
                "**28-day compressive strength** of concrete is the standard measure "
                "of mix adequacy. Conventionally, this requires casting cylinder "
                "specimens and curing them in water tanks for nearly a month before "
                "destructive testing on a **Universal Testing Machine (UTM)**."
            )
            st.markdown(
                "The upcoming model aims to give an early screening estimate of "
                "compressive strength directly from batch mix proportions "
                "(e.g., **Cement Content**, **Water-Cement Ratio**, **Curing Age**) "
                "before casting."
            )
    with col_r2:
        with st.container(border=True):
            st.markdown("### Planned Approach")
            st.markdown(
                "The ML module will be built on top of the same first-principles "
                "Gauss-Jordan engine used in this app:"
            )
            st.markdown(
                "• **Multiple Linear Regression** fitted by least squares\n\n"
                "• **Normal Equations** solved by our own Gauss-Jordan solver: "
                "$(X^T X)\\beta = X^T y$\n\n"
                "• **Model validation** and an interactive mix-design predictor"
            )

    st.info(
        "This feature is **not yet available**. For now, the app focuses on the "
        "general-purpose Gauss-Jordan linear system solver in Tab 1."
    )

# -----------------------------------------------------------------------------
# 5. FOOTER
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    f"Live app: [{LIVE_APP_URL.replace('https://', '')}]({LIVE_APP_URL}) · "
    "Davao Oriental State University · Faculty of Computing, Engineering and "
    "Technology · Department of Civil Engineering"
)
