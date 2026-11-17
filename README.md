# Numerical Methods Solver

A web application for solving systems of linear equations and single nonlinear equations, with every calculation step shown for verification.

**Davao Oriental State University**
Faculty of Computing, Engineering and Technology, Department of Civil Engineering

[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://mlgaussjordan.streamlit.app)
[![Python](https://img.shields.io/badge/Language-Python%203.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic%20Use-blue?style=for-the-badge)](#)

**Live application:** [mlgaussjordan.streamlit.app](https://mlgaussjordan.streamlit.app)

---

## Project Team

| Role | Name |
| :--- | :--- |
| Course Instructor | Engr. Samuel Erespe |
| Proponents | John Joshua D. Ilisan, Darius Lape, Briel Jan M. Lacia |

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Mathematical Background](#mathematical-background)
4. [Using the Application](#using-the-application)
5. [Installation](#installation)
6. [Project Structure](#project-structure)
7. [Verification](#verification)
8. [Limitations](#limitations)
9. [Roadmap](#roadmap)
10. [License](#license)

---

## Overview

Solving linear systems $Ax = b$ and nonlinear equations $f(x) = 0$ is a basic requirement in structural analysis, hydraulics, and general engineering computation. This project implements the underlying numerical methods from first principles, without calling an external linear algebra or root-finding library. The application records each intermediate matrix and iteration so that results can be checked against hand calculations.

---

## Features

| Module | Methods | Output |
| :--- | :--- | :--- |
| Linear systems | Gauss-Jordan elimination with partial pivoting | Solution vector and every row operation |
| Bracketing methods | Bisection, False Position (Regula Falsi) | Root, iteration table, graph with the starting interval |
| Open methods | Newton-Raphson, Secant | Root, iteration table, graph with the iterates |
| Polynomial roots | Bairstow's method | All real and complex roots, Bairstow tables for each iteration |

Additional capabilities:

- Functions are typed as text, including exponential, trigonometric, and logarithmic forms (for example `e^(-x) = x` or `2sin(x) - x`).
- A graph of f(x) is drawn as soon as the function is entered and is updated with the root and iterations after solving.
- Trigonometric functions can be evaluated in radians or degrees.
- Input is read by a restricted expression parser. User text is never passed to `eval`.
- Singular and ill-conditioned systems, intervals without a sign change, zero derivatives, and divergence are reported with explanatory messages.
- A built-in tutorial explains how to type functions and provides ready-made examples.

---

## Mathematical Background

### 1. Linear Systems: Gauss-Jordan Elimination

A system of $n$ equations in $n$ unknowns is written as

$$Ax = b$$

The augmented matrix $M = [A \mid b]$ of size $n \times (n+1)$ is reduced to reduced row echelon form $[I \mid x]$. For each column $k$:

1. **Partial pivoting.** Select the row $r \ge k$ with the largest $\lvert M_{r,k} \rvert$ and swap it into row $k$.
2. **Singularity check.** The system is rejected if
   $$\lvert M_{k,k} \rvert < \max\left(\text{tol} \cdot \max \lvert M \rvert,\ 10^{-14}\right), \qquad \text{tol} = 10^{-10}$$
   The threshold is relative to the size of the matrix entries, so large-valued systems are not rejected incorrectly.
3. **Normalization.** Divide row $k$ by the pivot so that $M_{k,k} = 1$.
4. **Elimination.** For every other row $i \ne k$:
   $$M_{i,j} \leftarrow M_{i,j} - M_{i,k}\, M_{k,j}$$

After the last column, the solution is read from the final column: $x_i = M_{i,n+1}$.

### 2. Bracketing Methods

These methods require an interval $[a, b]$ with $f(a)\,f(b) < 0$ and always keep a root inside the interval.

| Method | Update | Remarks |
| :--- | :--- | :--- |
| Bisection | $c = \dfrac{a + b}{2}$ | The interval halves at each step; convergence is guaranteed. |
| False Position | $c = b - \dfrac{f(b)\,(a - b)}{f(a) - f(b)}$ | The $x$-intercept of the chord through the endpoints. Often faster than bisection, but one endpoint may remain fixed. |

### 3. Open Methods

These methods need starting guesses but not a sign change. They are faster, but convergence is not guaranteed.

| Method | Update | Remarks |
| :--- | :--- | :--- |
| Newton-Raphson | $x_{n+1} = x_n - \dfrac{f(x_n)}{f'(x_n)}$ | Quadratic convergence near a root. Fails if $f'(x_n) \approx 0$. The derivative may be typed or approximated by a central difference. |
| Secant | $x_{n+1} = x_n - \dfrac{f(x_n)\,(x_{n-1} - x_n)}{f(x_{n-1}) - f(x_n)}$ | Replaces $f'(x_n)$ with a finite-difference slope; needs two starting values. |

**Stopping criteria.** Bisection stops when the half-width of the interval is below the tolerance. The other methods stop when $\lvert x_{n+1} - x_n \rvert$ is below the tolerance. All methods also stop at the maximum number of iterations (default 100) and report if the tolerance was not reached.

### 4. Bairstow's Method

Bairstow's method finds the quadratic factors $x^2 + ux + v$ of a polynomial

$$f(x) = a_0 x^n + a_1 x^{n-1} + \dots + a_n, \qquad a_0 = 1,\ a_n \neq 0$$

so that all real and complex roots can be obtained from the quadratic formula. Starting from guesses $u^0, v^0$:

1. **Synthetic division by the trial factor:**
   $$b_0 = a_0,\quad b_1 = a_1 - u\,b_0,\quad b_k = a_k - u\,b_{k-1} - v\,b_{k-2}$$
   The remainder is $(x + u)\,b_{n-1} + b_n$.
2. **Partial derivatives** (a second synthetic division of the $b$ values):
   $$c_0 = b_0,\quad c_1 = b_1 - u\,c_0,\quad c_k = b_k - u\,c_{k-1} - v\,c_{k-2}$$
   with $\partial b_k / \partial u = -c_{k-1}$ and $\partial b_k / \partial v = -c_{k-2}$.
3. **Corrections from a Taylor series expansion.** Setting the remainder terms to zero gives
   $$b_n = c_{n-1}\,\Delta u + c_{n-2}\,\Delta v, \qquad b_{n-1} = c_{n-2}\,\Delta u + c_{n-3}\,\Delta v$$
   solved by Cramer's rule:
   $$\Delta u = \frac{b_n\,c_{n-3} - c_{n-2}\,b_{n-1}}{c_{n-1}\,c_{n-3} - c_{n-2}^{2}}, \qquad \Delta v = \frac{c_{n-1}\,b_{n-1} - c_{n-2}\,b_n}{c_{n-1}\,c_{n-3} - c_{n-2}^{2}}$$
4. **Update** $u \leftarrow u + \Delta u$ and $v \leftarrow v + \Delta v$, and repeat until $\lvert b_n \rvert + \lvert b_{n-1} \rvert < \varepsilon$ or $\lvert \Delta u \rvert + \lvert \Delta v \rvert < \varepsilon$.
5. **Deflation.** The roots of $x^2 + ux + v$ are found with the quadratic formula. The $b$ coefficients form a polynomial of degree $n - 2$, and the process is repeated until all roots are found.

**Iteration safeguards.** Divergence is reported if $\lvert \Delta u_{50} \rvert + \lvert \Delta v_{50} \rvert > \lvert \Delta u_1 \rvert + \lvert \Delta v_1 \rvert$. Slow convergence is reported if the tolerance is not met after 100 iterations. Iteration stops after 200 iterations.

---

## Using the Application

### Typing a Function

Type $f(x)$ as on a calculator, using `x` as the variable. An expression is solved as $f(x) = 0$. A full equation such as `cos(x) = x` is also accepted and is solved as $\cos x - x = 0$. The line labeled "Reads as" under the input box shows how the entry was interpreted.

| Mathematical form | Entry |
| :--- | :--- |
| $3x^2 + 2$ | `3x^2 + 2` or `3*x^2 + 2` |
| $e^{x}$, $e^{-x}$, $e^{2x}$ | `e^x` (or `exp(x)`), `e^(-x)`, `e^(2x)` |
| $2^x$ | `2^x` |
| $2\sin x$, $\sin^2 x$, $\cos 2x$ | `2sin(x)`, `sin(x)^2`, `cos(2x)` |
| $\ln x$, $\log_{10} x$, $\log_2 x$ | `ln(x)`, `log10(x)`, `log2(x)` |
| $\sqrt{x}$, $\sqrt[3]{x}$, $\lvert x \rvert$, $\pi$ | `sqrt(x)`, `cbrt(x)`, `abs(x)`, `pi` |

**Available functions:** `sin`, `cos`, `tan`, `sec`, `csc`, `cot`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `ln`, `log`, `log10`, `log2`, `sqrt`, `cbrt`, `abs`. **Constants:** `pi`, `e`.

Notes:

- The input of a function must be in parentheses: `sin(x)`, not `sin x`.
- Use parentheses around an exponent with more than one term: `e^(2x)` is $e^{2x}$, whereas `e^2x` means $e^2 \cdot x$.
- `log(x)` is the natural logarithm, identical to `ln(x)`. Use `log10(x)` for base 10.
- Trigonometric functions use radians unless the angle mode is set to degrees.
- Bairstow's method accepts polynomials only.
- Exponential and trigonometric functions may have several roots. Use the graph and the sign-change scanner to choose an interval or starting guess.
- Near an asymptote (for example $\tan x$ or $1/x$), a sign change does not indicate a root. The application warns when $f(x^*)$ is not close to zero.

### Choosing a Method

| Situation | Suggested method |
| :--- | :--- |
| A sign change is known and reliability is the priority | Bisection or False Position |
| A good starting guess is available and speed matters | Newton-Raphson |
| The derivative is difficult to obtain | Secant |
| All roots of a polynomial are needed, including complex roots | Bairstow's method |

---

## Installation

**Requirements:** Python 3.9 or later.

```bash
# Obtain the project files and open a terminal in the project folder

# Optional: create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Start the application
streamlit run app.py
```

The application opens at `http://localhost:8501`.

If `pip` or `streamlit` is not recognized on Windows, use the Python launcher:

```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

### Deployment on Streamlit Community Cloud

1. Push all project files, including the `.streamlit` folder, to a GitHub repository.
2. Create a new app at [share.streamlit.io](https://share.streamlit.io) and set the main file path to `app.py`.
3. After the first deployment, the application redeploys automatically whenever the repository is updated.

### Using the Solvers Without the Interface

The solver modules can be imported directly:

```python
from gauss_jordan import solve_gauss_jordan
import root_finding as rf

# Linear system
x = solve_gauss_jordan([[2, 1], [1, 3]], [3, 5])

# Nonlinear equation
f = rf.parse_function("x^3 - 2x - 5")
result = rf.bisection(f, 2, 3, tol=1e-6)
print(result["root"], result["iterations"])

# Polynomial roots
coeffs = rf.parse_polynomial("x^5 - 3x^4 - 10x^3 + 10x^2 + 44x + 48")
print(rf.bairstow(coeffs, u0=1.8, v0=1.8)["roots"])
```

Running `python root_finding.py` prints a short comparison of the four root-finding methods.

---

## Project Structure

```text
.
|-- app.py                 Streamlit interface
|-- gauss_jordan.py        Gauss-Jordan solver with step logging
|-- root_finding.py        Bisection, False Position, Newton-Raphson, Secant,
|                          Bairstow, and the expression/polynomial parser
|-- plot_utils.py          Builds the graph of f(x) shown in the interface
|-- linear_regression.py   Multiple linear regression using the Gauss-Jordan
|                          solver (for the planned module; not yet in the interface)
|-- requirements.txt       Python dependencies
|-- .streamlit/
|   `-- config.toml        Interface theme
`-- README.md              Project documentation
```

---

## Verification

The following reference results were reproduced by the implementation and can be used to check an installation.

| Problem | Method | Result |
| :--- | :--- | :--- |
| $x^3 - 2x - 5 = 0$, tolerance $10^{-6}$ | Bisection on $[2, 3]$ | $x \approx 2.094552$ in 20 iterations |
| | False Position on $[2, 3]$ | $x \approx 2.094551$ in 13 iterations |
| | Newton-Raphson from $x_0 = 2$ | $x \approx 2.094551$ in 4 iterations |
| | Secant from $2$ and $3$ | $x \approx 2.094551$ in 6 iterations |
| $x^5 - 3x^4 - 10x^3 + 10x^2 + 44x + 48 = 0$ | Bairstow, $u^0 = v^0 = 1.8$ | Factor $x^2 + 2x + 2$, giving $x = -1 \pm i$; remaining factor $x^3 - 5x^2 - 2x + 24$ with roots $-2,\ 3,\ 4$ |
| Default $4 \times 4$ system in the interface | Gauss-Jordan | $x \approx (2.3791,\ 3.8303,\ 3.3863,\ 4.6570)$ |

The first Bairstow iteration of the sample problem gives $\Delta u \approx 0.204$ and $\Delta v \approx 0.153$, matching a hand calculation. Roots obtained for random polynomials of degree 3 to 8 were compared with NumPy's root finder and agreed.

---

## Limitations

- The linear solver handles square systems of size $2 \times 2$ to $8 \times 8$ with a unique solution.
- Root-finding methods find one root per run (Bairstow's method finds all roots of a polynomial). Several roots of a general function require separate runs with different intervals or starting values.
- Newton-Raphson and Secant may diverge or converge to an unintended root if the starting values are poor.
- Bairstow's method may fail for unfavorable starting values $u^0, v^0$. In the "all roots" mode, the application retries with alternative starting values before reporting failure.
- Results are limited by double-precision floating-point arithmetic.

---

## Roadmap

A machine learning module is planned for the next update. It will estimate the 28-day compressive strength of concrete from mix proportions (for example cement content, water-cement ratio, and curing age) using multiple linear regression. The model will be fitted by solving the normal equations $(X^T X)\beta = X^T y$ with the Gauss-Jordan solver in this project.

---

## License

This project is intended for academic use.
