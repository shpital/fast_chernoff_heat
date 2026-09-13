"""Study III, day 2: kill-test -- a POSITIVE, compactly supported, absolutely
continuous one-step operator of second order for the variable-coefficient
generator
        H f = a(x) f'' + b(x) f' + c(x) f,      a >= a_min > 0,
i.e. the question whether the signed, superdiffusive (tau^{1/4}) correction of
Remizov-Vedenin is *necessary* for quadratic convergence with variable
coefficients, or is a design choice.

Ansatz (two uniform windows, x- and t-dependent weights/centres/half-widths):

    S(t)f(x) = sum_{i=1,2} w_i(x,t) * (1/(2 r_i)) int_{-r_i}^{r_i} f(x + c_i + s) ds,

    w_i = w_i0 + w_i1 t + w_i2 t^2,   c_i = c_i1 t + c_i2 t^2,
    r_i^2 = rho_i1 t + rho_i2 t^2          (only r_i^2 enters the moments).

Part 1 (symbolic, exact): solve the moment equations so that

    S(t)f = f + t Hf + (t^2/2) H^2 f + O(t^3)        for every f in C^6,

for generic smooth a, b, c (sympy Functions), verify the identity term by term,
check positivity of weights and half-widths for small t, and check that the
construction reduces to the constant-heat mixture MIX2 when a = 1, b = c = 0.

Part 2 (numerical): periodic test problem u_t = a u_xx + b u_x + c u with
a = 1 + 0.5 sin x, b = 0.3 cos x, c = 0.2 sin 2x, smooth datum; the scheme is
applied exactly on the Fourier side (window average of e^{ikx} = e^{ik(x+c)}
sinc(k r)), reference = expm of the spectral collocation operator.  Expected:
sup-error ~ n^{-2}.  Also a rough datum (|sin x|^{1/2}).  CAVEAT: this is a
fixed-grid test -- the datum is replaced by its trigonometric interpolant and
the grid values are re-interpolated after every step -- so it checks a
finite-dimensional approximation, not uniformity over bounded data or over
resolution; see day2b_varcoef_norm.py for the N-dependence and the discrete
error-operator norm.

Outputs: data/day2_*.csv, figs/day2_varcoef.png; console = record.
"""
import os
import time

import numpy as np
import pandas as pd
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

# ---------------------------------------------------------------------------
# Part 1: symbolic construction
# ---------------------------------------------------------------------------
x, t = sp.symbols("x t", real=True)
a = sp.Function("a")(x)
b = sp.Function("b")(x)
c = sp.Function("c")(x)
f = sp.Function("f")(x)


def H(g):
    return a * sp.diff(g, x, 2) + b * sp.diff(g, x) + c * g


def target_moments(order=6):
    """m_j^*(t) = j! * [coefficient of f^{(j)}(x)] in f + t Hf + t^2/2 H^2 f."""
    T2 = sp.expand(f + t * H(f) + t ** 2 / 2 * H(H(f)))
    # replace derivatives of f by placeholders
    ders = [sp.Symbol(f"F{j}") for j in range(order + 1)]
    expr = T2
    for j in range(order, 0, -1):
        expr = expr.subs(sp.Derivative(f, (x, j)), ders[j])
    expr = expr.subs(f, ders[0])
    expr = sp.expand(expr)
    return {j: sp.expand(sp.factorial(j) * expr.coeff(ders[j])) for j in range(order + 1)}


def window_moment(j, cc, r2):
    """E[(cc + U)^j], U ~ Uniform[-r, r], r^2 = r2  (polynomial in r2)."""
    return sum(sp.binomial(j, l) * cc ** (j - l) * r2 ** (l // 2) / (l + 1)
               for l in range(0, j + 1, 2))


def solve_construction(weights0=(sp.Rational(1, 2), sp.Rational(1, 2)),
                       nodes=(6 - 2 * sp.sqrt(6), 6 + 2 * sp.sqrt(6)), verbose=True):
    """Return the parameter dictionary and the verified defect series."""
    import builtins
    print = builtins.print if verbose else (lambda *a, **k: None)
    print("== Part 1: symbolic construction ==")
    W0 = list(weights0)
    V = list(nodes)                       # rho_i1 = a * v_i (constant-heat mixture scaled by a)
    w1 = sp.symbols("w11 w21"); w2 = sp.symbols("w12 w22")
    c1 = sp.symbols("c11 c21"); c2 = sp.symbols("c12 c22")
    r2 = sp.symbols("rho12 rho22")
    ws = [W0[i] + w1[i] * t + w2[i] * t ** 2 for i in range(2)]
    cs = [c1[i] * t + c2[i] * t ** 2 for i in range(2)]
    rs = [a * V[i] * t + r2[i] * t ** 2 for i in range(2)]
    ms = {j: sp.expand(sum(ws[i] * window_moment(j, cs[i], rs[i]) for i in range(2)))
          for j in range(7)}
    tgt = target_moments()
    eqs = []
    for j in range(5):
        d = sp.expand(ms[j] - tgt[j])
        for p in range(3):
            e = sp.simplify(d.coeff(t, p))
            if e != 0:
                eqs.append(e)
    print(f"  moment equations (j<=4, t^0..t^2): {len(eqs)} nontrivial")
    unknowns = list(w1) + list(w2) + list(c1) + list(c2) + list(r2)
    # gauge: split the mass corrections equally, common second-order centre
    # and half-width corrections
    gauge = [w1[0] - w1[1], w2[0] - w2[1], c2[0] - c2[1], r2[0] - r2[1]]
    sol = sp.solve(eqs + gauge, unknowns, dict=True)
    assert len(sol) == 1, sol
    sol = sol[0]
    print("  solution (gauge: w11=w21, w12=w22, c12=c22, rho12=rho22):")
    for u in unknowns:
        print(f"    {u} = {sp.simplify(sol[u])}")
    # verification: full defect through order t^2, all j <= 6
    ok = True
    for j in range(7):
        d = sp.expand((ms[j] - tgt[j]).subs(sol))
        for p in range(3):
            e = sp.simplify(d.coeff(t, p))
            if e != 0:
                ok = False
                print(f"  RESIDUAL j={j} t^{p}: {e}")
    print(f"  defect S(t)f - [f + tHf + t^2/2 H^2 f] = O(t^3) for all f in C^6: "
          f"{'PASS' if ok else 'FAIL'}")
    # extra: the scheme's sixth moment at t^3 vs the semigroup's (constant a:
    # 6! a^3 t^3/3! = 120 a^3 t^3); the difference is the leading local defect
    m6_3 = sp.factor(sp.expand(ms[6].subs(sol)).coeff(t, 3))
    print(f"  m_6(S) at t^3 = {m6_3};  heat semigroup (constant a): 120 a^3;  "
          f"difference = {sp.simplify(m6_3 - 120 * a**3)} = -6!*(4/105) a^3 "
          f"(the MIX2 constant; the variable-coefficient t^3 remainder also "
          f"carries derivative terms in m_j, j <= 5)")
    # constant-coefficient reduction
    red = {u: sp.simplify(sol[u].subs({a: 1, b: 0, c: 0}).doit()) for u in unknowns}
    print(f"  a=1, b=c=0 reduction: {red}")
    # positivity for small t: weights and squared half-widths
    print("  positivity: w_i0 = 1/2 > 0, r_i^2 = a v_i t + O(t^2) with v_i > 0 "
          "=> positive for 0 < t < t_0(a_min, ||a,b,c||_{C^2}).")
    params = dict(W0=W0, V=V, sol=sol, w1=w1, w2=w2, c1=c1, c2=c2, r2=r2)
    pd.DataFrame([dict(unknown=str(u), value=str(sp.simplify(sol[u])))
                  for u in unknowns]).to_csv(
        os.path.join(DATA, "day2_construction.csv"), index=False)
    return params


# ---------------------------------------------------------------------------
# Part 2: numerical second-order check on a periodic variable-coefficient problem
# ---------------------------------------------------------------------------
TEST_A = 1 + sp.Rational(1, 2) * sp.sin(x)
TEST_B = sp.Rational(3, 10) * sp.cos(x)
TEST_C = sp.Rational(1, 5) * sp.sin(2 * x)


def make_numeric(params, A=TEST_A, B=TEST_B, Cc=TEST_C):
    """Lambdify w_i(x,t), c_i(x,t), r_i^2(x,t) for given coefficient expressions."""
    sub = {a: A, b: B, c: Cc}
    sol = params["sol"]
    W0, V = params["W0"], params["V"]
    def vec(fn, nargs):
        """lambdified constants come back as scalars; broadcast to the grid."""
        if nargs == 1:
            return lambda xg: np.broadcast_to(np.asarray(fn(xg), float), np.shape(xg)).copy()
        return lambda xg, tt: np.broadcast_to(np.asarray(fn(xg, tt), float), np.shape(xg)).copy()

    exprs = {}
    for i in range(2):
        w_i = W0[i] + sol[params["w1"][i]] * t + sol[params["w2"][i]] * t ** 2
        c_i = sol[params["c1"][i]] * t + sol[params["c2"][i]] * t ** 2
        r_i = A * V[i] * t + sol[params["r2"][i]] * t ** 2
        exprs[i] = [vec(sp.lambdify((x, t), sp.simplify(e.subs(sub).doit()), "numpy"), 2)
                    for e in (w_i, c_i, r_i)]
    coef = [vec(sp.lambdify(x, e, "numpy"), 1) for e in (A, B, Cc)]
    return exprs, coef


def spectral_reference(coef, u0_hat, N, T):
    """u(T) for u_t = a u_xx + b u_x + c u, periodic, spectral collocation + expm."""
    from scipy.linalg import expm
    xg = 2 * np.pi * np.arange(N) / N
    k = np.fft.fftfreq(N, d=1.0 / N)
    F = np.fft.fft(np.eye(N), axis=0)          # DFT matrix
    Finv = np.linalg.inv(F)
    D1 = np.real(Finv @ np.diag(1j * k) @ F)
    D2 = np.real(Finv @ np.diag(-(k ** 2)) @ F)
    A, B, C = (np.diag(cf(xg)) for cf in coef)
    L = A @ D2 + B @ D1 + C
    u0 = np.real(np.fft.ifft(u0_hat))
    return expm(T * L) @ u0, xg


def transfer_matrix(exprs, N, tau):
    """M[x_j, k] = sum_i w_i(x_j) e^{ik(x_j + c_i(x_j))} sinc(k r_i(x_j)): the
    exact window averages of the Fourier modes at the grid points.  Returns
    None if the step is not positive (some w_i <= 0 or r_i^2 <= 0) at tau."""
    xg = 2 * np.pi * np.arange(N) / N
    k = np.fft.fftfreq(N, d=1.0 / N)
    W = [ex[0](xg, tau) for ex in exprs.values()]
    Cc = [ex[1](xg, tau) for ex in exprs.values()]
    R2 = [ex[2](xg, tau) for ex in exprs.values()]
    if not (all(np.all(w > 0) for w in W) and all(np.all(r2 > 0) for r2 in R2)):
        return None
    M = np.zeros((N, N), dtype=complex)
    for w, cc, r2 in zip(W, Cc, R2):
        arg = np.outer(np.sqrt(r2), k)
        safe = np.where(np.abs(arg) < 1e-12, 1.0, arg)
        sc = np.where(np.abs(arg) < 1e-12, 1.0, np.sin(safe) / safe)
        M += w[:, None] * np.exp(1j * np.outer(xg + cc, k)) * sc
    return M


def chernoff_positive(exprs, u0_hat, N, T, n):
    """n steps of S(T/n) on the Fourier side (exact window averages of the
    modes; trigonometric re-interpolation of the grid values between steps)."""
    xg = 2 * np.pi * np.arange(N) / N
    M = transfer_matrix(exprs, N, T / n)
    if M is None:
        return None, xg                          # not a positive step at this tau
    u_hat = u0_hat.copy()
    for _ in range(n):
        u = (M @ u_hat) / N                      # values on the grid
        u_hat = np.fft.fft(u.real)
    return np.real(np.fft.ifft(u_hat)), xg


def numeric_test(params):
    print("== Part 2: numerical order check (periodic, variable coefficients) ==")
    exprs, coef = make_numeric(params)
    T = 0.5
    rows = []
    for tag, N in (("smooth", 128), ("rough", 512)):
        xg = 2 * np.pi * np.arange(N) / N
        u0 = (np.exp(np.cos(xg)) + 0.3 * np.sin(2 * xg) if tag == "smooth"
              else np.abs(np.sin(xg)) ** 0.5)
        u0_hat = np.fft.fft(u0)
        # scheme and reference act on the same band-limited grid datum
        uref, _ = spectral_reference(coef, u0_hat, N, T)
        prev = None
        for n in (2, 4, 8, 16, 32, 64, 128, 256, 512):
            un, _ = chernoff_positive(exprs, u0_hat, N, T, n)
            if un is None:
                rows.append(dict(datum=tag, n=n, sup_err=np.nan, rate=np.nan))
                continue
            err = np.max(np.abs(un - uref))
            rate = np.log2(prev / err) if prev else np.nan
            rows.append(dict(datum=tag, n=n, sup_err=err, rate=rate))
            prev = err
        sub = [r for r in rows if r["datum"] == tag]
        print(f"  {tag}: " + ", ".join(f"n={r['n']}: {r['sup_err']:.2e} "
                                        f"(rate {r['rate']:.2f})" for r in sub))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day2_order.csv"), index=False)
    return df


def figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for tag, mk in (("smooth", "o"), ("rough", "s")):
        sub = df[df.datum == tag]
        ax.loglog(sub.n, sub.sup_err, mk + "-", label=f"{tag} datum")
    nn = np.array([2, 512])
    ax.loglog(nn, df.sup_err.iloc[0] * (nn / 2.0) ** -2, "k:", label="n^-2")
    ax.set_xlabel("n"); ax.set_ylabel("sup |S(T/n)^n u0 - u(T)|")
    ax.set_title("positive two-window scheme, variable a, b, c", fontsize=10)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "day2_varcoef.png"), dpi=150)


if __name__ == "__main__":
    t0 = time.time()
    params = solve_construction()
    df = numeric_test(params)
    figure(df)
    print(f"== done in {time.time()-t0:.0f}s ==")
