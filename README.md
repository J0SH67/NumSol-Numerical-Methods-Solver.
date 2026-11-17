# 🏗️ Numerical Methods Solver
### *Linear Systems via Gauss-Jordan Elimination + Nonlinear Root Finding (Bisection, False Position, Newton-Raphson, Secant, Bairstow)*

**Davao Oriental State University**  
Faculty of Computing, Engineering and Technology — Department of Civil Engineering  

[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://mlgaussjordan.streamlit.app)
[![Python](https://img.shields.io/badge/Language-Python%203.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic%20Use-blue?style=for-the-badge)](#)

---

## 🌐 Live Web Application
You can try the platform directly in your browser without local installation:  
👉 **[mlgaussjordan.streamlit.app](https://mlgaussjordan.streamlit.app)**

---

## 👨‍🏫 Course Instructor
* **Engr. Samuel Erespe**

## 👥 Proponents & Contributors
* **John Joshua D. Ilisan**
* **Darius Lape**
* **Briel Jan M. Lacia**

---

## 📌 Executive Summary
In engineering analysis and numerical computation, solving systems of linear equations ($Ax = b$) is a foundational requirement for structural analysis, network flow calculations, and frame modeling. This project provides a robust, transparent, first-principles implementation of **Gauss-Jordan Elimination** designed to solve general $n \times n$ systems of linear equations without relying on external linear algebra solvers or commercial libraries.

### Core Architecture & Components
1. **Custom Gauss-Jordan Elimination Engine (`gauss_jordan.py`):** Solves general square systems using scale-aware relative tolerance singularity detection, partial pivoting for numerical stability, and full audit logging of row transformations.
2. **Root-Finding Engine (`root_finding.py`):** Solves single linear or nonlinear equations $f(x) = 0$ using two bracketing methods (Bisection, False Position), two open methods (Newton-Raphson, Secant) and Bairstow's method for all real and complex roots of a polynomial, with a safe expression parser and a full iteration log.
3. **Interactive Streamlit Web Dashboard (`app.py`):** Provides a visual, interactive interface allowing users to dynamically adjust matrix dimensions ($2 \times 2$ up to $8 \times 8$), manually input or modify matrix coefficients, compute exact solutions, and trace every intermediate row operation step-by-step.

---

## 📐 Mathematical Formulation

### 1. General System of Linear Equations
A system of $n$ linear equations with $n$ unknown variables is expressed in matrix notation as:

$$Ax = b$$

$$\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nn} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{bmatrix}$$

The goal is to determine the unknown vector $x = [x_1, x_2, \dots, x_n]^T$ by reducing the augmented matrix $[A \mid b]$ into reduced row echelon form (RREF) $[I \mid x]$, where $I$ is the $n \times n$ identity matrix.

---

### 2. Scale-Aware Gauss-Jordan Elimination with Partial Pivoting
The solver in `gauss_jordan.py` implements the following systematic numerical steps:

#### Step 1: Augmented Matrix Setup
The coefficient matrix $A$ and constant vector $b$ are combined into an augmented matrix $M$ of dimension $n \times (n+1)$:

$$M = [A \mid b]$$

#### Step 2: Partial Pivoting
To maintain numerical stability and avoid division by zero (or small numbers that introduce roundoff errors), row swaps are performed at each column $k$:

$$\text{Find } r \text{ such that } \lvert M_{r,k} \rvert = \max_{i \ge k} \lvert M_{i,k} \rvert$$

If $r \neq k$, Row $k$ and Row $r$ are swapped.

#### Step 3: Scale-Aware Singularity Detection
Rather than relying on a rigid fixed tolerance, the solver establishes a relative singularity threshold based on the overall magnitude of the working matrix:

$$\text{threshold} = \max(\text{tol} \times \max(\lvert M \rvert), 10^{-14}) \quad \text{where } \text{tol} = 10^{-10}$$

If the pivot value $\lvert M_{k,k} \rvert < \text{threshold}$, the system is flagged as singular or ill-conditioned, and execution halts cleanly with an informative explanation.

#### Step 4: Row Normalization & Elementary Transformations
1. **Normalize Pivot Row:** Divide the entire pivot row by the diagonal entry $M_{k,k}$ so that $M_{k,k} = 1$:
   $$M_{k, j} \leftarrow \frac{M_{k, j}}{M_{k, k}} \quad \text{for } j = k, \dots, n$$

2. **Eliminate Non-Zero Column Entries:** Zero out all off-diagonal entries in column $k$ across all other rows $i \neq k$:
   $$M_{i, j} \leftarrow M_{i, j} - (M_{i, k} \times M_{k, j}) \quad \text{for all } i \neq k$$

Upon completing elimination across all $n$ columns, the solution vector is directly read from the final column:

$$x_i = M_{i, n+1} \quad \text{for } i = 1, 2, \dots, n$$

---

## 📉 Nonlinear Equations: Root-Finding Methods

To solve $f(x) = 0$, `root_finding.py` provides four methods. Functions are typed as text (e.g. `x^3 - 2*x - 5` or `cos(x) = x`) and read by a safe parser (no `eval`).

### Bracketing Methods (need an interval $[a, b]$ with $f(a)\,f(b) < 0$)

| Method | Update formula | Notes |
| :--- | :--- | :--- |
| **Bisection** | $c = \dfrac{a + b}{2}$ | Always converges; error halves each step. |
| **False Position (Regula Falsi)** | $c = b - \dfrac{f(b)(a - b)}{f(a) - f(b)}$ | Often faster than bisection; one endpoint can get "stuck". |

### Open Methods (need starting guesses, no sign change required)

| Method | Update formula | Notes |
| :--- | :--- | :--- |
| **Newton-Raphson** | $x_{n+1} = x_n - \dfrac{f(x_n)}{f'(x_n)}$ | Quadratic convergence; fails if $f'(x_n) \approx 0$. $f'(x)$ may be typed in or approximated by central difference. |
| **Secant** | $x_{n+1} = x_n - \dfrac{f(x_n)(x_{n-1} - x_n)}{f(x_{n-1}) - f(x_n)}$ | No derivative needed; needs two starting guesses. |


### Bairstow's Method (polynomials: all real and complex roots)

Finds a quadratic factor $x^2 + ux + v$ of $f(x) = a_0x^n + a_1x^{n-1} + \dots + a_n$ (with $a_0 = 1$, $a_n \neq 0$), starting from guesses $u^0, v^0$. Its two roots come from the quadratic formula, and the remaining degree $n-2$ polynomial is factored the same way until every root is found.

1. **Synthetic division:** $b_0 = a_0,\; b_1 = a_1 - u b_0,\; b_k = a_k - u b_{k-1} - v b_{k-2}$; the remainder is $(x+u)b_{n-1} + b_n$.
2. **Partial derivatives:** the same recursion on the b's gives $c_0 = b_0,\; c_1 = b_1 - u c_0,\; c_k = b_k - u c_{k-1} - v c_{k-2}$, where $\partial b_k/\partial u = -c_{k-1}$ and $\partial b_k/\partial v = -c_{k-2}$.
3. **Taylor series + Cramer's rule:** solve $b_n = c_{n-1}\Delta u + c_{n-2}\Delta v$ and $b_{n-1} = c_{n-2}\Delta u + c_{n-3}\Delta v$ for $\Delta u, \Delta v$.
4. **Update and repeat** with $u \leftarrow u + \Delta u$, $v \leftarrow v + \Delta v$ until $|b_n| + |b_{n-1}| < \varepsilon$ or $|\Delta u| + |\Delta v| < \varepsilon$.
5. **Safeguards:** *divergence* if $|\Delta u_{50}| + |\Delta v_{50}| > |\Delta u_1| + |\Delta v_1|$; *slow convergence* after 100 iterations; stop after 200.

The app shows every iteration as a Bairstow table (a's, b's, c's), with the determinants for $\Delta u$ and $\Delta v$ filled in. Type the polynomial as text, e.g. `x^5 - 3*x^4 - 10*x^3 + 10*x^2 + 44*x + 48`; the default guesses $u^0 = v^0 = 1.8$ reproduce the sample problem (factor $x^2 + 2x + 2$, roots $-1 \pm i$).

### Graphs

Every method draws the graph of $f(x)$ as soon as the function is typed. After solving, the iterations (white dots), the root (red diamond) and, for the bracketing methods, the starting interval (shaded band) are added. For Bairstow's method the real roots are marked on the x-axis; complex roots are listed in the results table.

### Typing your function (exponential, trigonometric, logarithmic and more)

Type $f(x)$ like on a calculator, using `x` as the variable. You can type an expression (solved as $f(x) = 0$) or a full equation such as `cos(x) = x`. The app shows a **Reads as** line so you can check how it understood your input, and the **📘 Tutorial** in the Nonlinear Equations tab has a cheat-sheet, common mistakes, and one-click examples.

| What you mean | What to type |
| :--- | :--- |
| 3x² + 2 | `3x^2 + 2` or `3*x^2 + 2` |
| eˣ, e⁻ˣ, e²ˣ | `e^x` (or `exp(x)`), `e^(-x)`, `e^(2x)` |
| 2ˣ | `2^x` |
| 2 sin x, sin²x, cos 2x | `2sin(x)`, `sin(x)^2`, `cos(2x)` |
| ln x, log₁₀ x, log₂ x | `ln(x)`, `log10(x)`, `log2(x)` |
| √x, ³√x, \|x\|, π | `sqrt(x)`, `cbrt(x)`, `abs(x)`, `pi` |

Available functions: `sin cos tan sec csc cot asin acos atan sinh cosh tanh exp ln log log10 log2 sqrt cbrt abs` (`log` is the natural log). Constants: `pi`, `e`. Always put a function's input in brackets (`sin(x)`, not `sin x`) and use brackets around multi-part exponents (`e^(2x)`, since `e^2x` means e² · x). Trigonometric functions use **radians** by default; switch **Angle mode** to *Degrees* when needed. Bairstow's method accepts polynomials only.

Standalone test:

```bash
python root_finding.py
```

---

## 🔬 Key Technical Features

| Feature | Description | Engineering Benefit |
| :--- | :--- | :--- |
| **Partial Pivoting** | Interchanges rows to place the absolute largest available pivot on the diagonal. | Prevents division by zero and minimizes floating-point precision degradation. |
| **Scale-Aware Singularity Checking** | Evaluates pivots against a dynamic threshold based on matrix magnitude ($\text{tol} \times \max \lvert M \rvert$). | Prevents false singular rejections on high-magnitude matrices while catching ill-conditioned systems. |
| **Auditable Transformation Logging** | Records every intermediate matrix state and human-readable row operation formula. | Serves as an educational step-by-step calculation trace for verification. |
| **Flexible System Sizing** | Dynamically scales from $2 \times 2$ up to $8 \times 8$ linear systems via interactive UI controls. | Accommodates a wide variety of structural and linear algebra problems. |

---

## 🚀 Installation & How to Run

Copy and run the code block below in your terminal to set up the repository, virtual environment, dependencies, and launch the application:

```bash
# Clone the repository and enter directory
git clone <your-repository-url>
cd <repository-folder-name>

# Create virtual environment
python -m venv venv

# Activate virtual environment (Linux/macOS)
source venv/bin/activate
# Note for Windows users: run "venv\Scripts\activate" instead

# Install required dependencies
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

### 🧪 Standalone CLI Testing
To test the numerical solver directly in the command line without launching the UI:

```bash
python gauss_jordan.py
```

---

## 📂 Project Structure

```text
├── app.py           # Interactive Streamlit dashboard for matrix editing & visual solver logs
├── gauss_jordan.py  # Standalone Gauss-Jordan elimination algorithm with scale-aware pivoting
├── root_finding.py  # Bisection, False Position, Newton-Raphson, Secant, Bairstow + safe expression/polynomial parser
├── plot_utils.py    # Builds the f(x) graph (Vega-Lite spec) used by every method
├── .streamlit/
│   └── config.toml  # Light theme (colors used by the app)
├── requirements.txt # Project dependencies (streamlit, numpy, pandas)
└── README.md        # Technical project documentation
```
