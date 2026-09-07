"""Day 9, Study II: the explicit constants of Proposition 3 inside the G-model.

v0.4 of the note strengthens Proposition 3 to a statement with explicit
leading constants (rate remainder assumed o(rate)):

    S_n := sum_j a_j(xi) D_lam(2j) [1 - G(4 j^2/n)]
         ~ (A_xi/1300) I_xi n^{-xi/2},          0 < xi < 1,
           I_xi = int_0^inf w^{-1-xi} [1 - G(4 w^2)] dw,
         = c_G/(1300 pi) n^{-1/2} log n + O(n^{-1/2}),   xi = 1,

with A_xi = 2^{1-xi} Gamma(xi+1) sin(pi xi/2)/pi the leading Fourier
constant of |sin x|^xi and c_G = 1300 M_10 Gamma(9/2)/10! (< 0 sign
convention: 1 - G(rho) = -c_G sqrt(rho) ..., here c_G := |.| = 0.0380).
The leading constants are lambda-independent (lambda enters only through
finitely many low modes, an O(n^{-1/2}) correction).

This script evaluates S_n exactly inside the G-model (G from the day-7
series/quadrature evaluator, closed-form a_j, exact D_lam) for n = 4^8..4^14
and compares with the predicted leading terms:
  * xi < 1:  S_n / (C_xi n^{-xi/2}) -> 1 at the rate n^{-(1-xi)/2};
  * xi = 1:  (S_n - lead) sqrt(n) -> a finite constant (remainder is
             O(n^{-1/2}) exactly, no log).
Two lambda values are used to show the lambda-independence of the limit.

Output: data/day9_p3_constants.csv; console = record.
"""
import os
import sys
from math import factorial, pi

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day7_G import G_pred, M2P            # noqa: E402
from day6_xi import a_coeffs, D_coef, LAM_C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

C_G = -1300.0 * M2P[5] / factorial(10) * gamma_fn(4.5)   # = 0.03801218...
XIS = (0.25, 0.5, 1.0)
LAMS = (1.1 * LAM_C, 2.0 * LAM_C)
NS = [4 ** p for p in (8, 10, 12, 14)]

# G on a log grid once; interpolate in log rho (G is smooth in log rho).
# Below the table 1 - G = c_G sqrt(rho) + O(rho) is used, above it 520/rho;
# the table must reach far enough down that the integrand of I_xi
# (w^{-1-xi}[1-G(4w^2)], w down to 1e-9 => rho ~ 4e-18) is never clipped.
_RG = np.logspace(-40, 12, 10000)
_GG = G_pred(_RG)


def G(rho):
    rho = np.asarray(rho, dtype=float)
    inside = np.interp(np.log(np.clip(rho, _RG[0], _RG[-1])),
                       np.log(_RG), _GG)
    return np.where(rho < _RG[0], 1.0 - C_G * np.sqrt(rho),
                    np.where(rho > _RG[-1], 520.0 / rho, inside))


def A_xi(xi):
    return 2.0 ** (1 - xi) * gamma_fn(xi + 1) * np.sin(pi * xi / 2) / pi


def I_xi(xi, wmin=1e-12, wmax=1e8):
    """int_0^inf w^{-1-xi} [1-G(4w^2)] dw; log-spaced trapezoid on
    [wmin, wmax] plus the analytic head 2 c_G wmin^{1-xi}/(1-xi)
    (1-G(4w^2) ~ 2 c_G w) and the exact tail wmax^{-xi}/xi (1-G -> 1)."""
    lw = np.linspace(np.log(wmin), np.log(wmax), 800001)
    w = np.exp(lw)
    f = w ** (-xi) * (1.0 - G(4.0 * w ** 2))           # w^{-1-xi} * w (dw = w dlw)
    head = 2.0 * C_G * wmin ** (1 - xi) / (1 - xi)
    return head + np.trapezoid(f, lw) + wmax ** (-xi) / xi


def S_n(xi, lam, n):
    """sum_j a_j D_lam(2j) [1-G(4j^2/n)], summed to J=100 sqrt(n) with the
    exact asymptotic tail (a_j D -> A_xi j^{-1-xi}/1300, 1-G -> 1 up to
    the 520/rho correction, which is included to first order)."""
    J = int(100 * np.sqrt(n)) + 4000
    _, a = a_coeffs(xi, J)
    j = np.arange(1, J + 1, dtype=float)
    s = np.sum(a * D_coef(lam, 2.0 * j) * (1.0 - G(4.0 * j ** 2 / n)))
    # tail j > J:  (A/1300) sum j^{-1-xi} [1 - 520 n/(4 j^2)]
    A = A_xi(xi)
    tail = A / 1300.0 * (J ** (-xi) / xi
                         - 130.0 * n * J ** (-xi - 2) / (xi + 2))
    return s + tail


def main():
    print(f"c_G = {C_G:.10f}  (expected 0.0380121843)")
    rows = []
    for xi in XIS:
        A = A_xi(xi)
        if xi < 1:
            I = I_xi(xi)
            C = A / 1300.0 * I
            print(f"\nxi = {xi}: A_xi = {A:.6f}, I_xi = {I:.6f}, "
                  f"C_xi = A_xi I_xi/1300 = {C:.6e}")
        else:
            I, C = np.nan, C_G / (1300.0 * pi)
            print(f"\nxi = 1: leading coefficient c_G/(1300 pi) = {C:.6e}")
        for lam in LAMS:
            for n in NS:
                s = S_n(xi, lam, n)
                if xi < 1:
                    lead = C * n ** (-xi / 2)
                    ratio = s / lead
                    resid = np.nan
                else:
                    lead = C * n ** -0.5 * np.log(n)
                    ratio = s / lead
                    resid = (s - lead) * np.sqrt(n)
                rows.append(dict(xi=xi, lam=lam, n=n, S_n=s, lead=lead,
                                 ratio=ratio, resid_sqrt_n=resid,
                                 A_xi=A, I_xi=I, C_xi=C))
                extra = ("" if xi < 1
                         else f"  (S_n - lead) sqrt(n) = {resid:+.4e}")
                print(f"  lam = {lam:8.3f}  n = 4^{int(round(np.log(n)/np.log(4))):2d}"
                      f"  S_n/lead = {ratio:.5f}{extra}")
    df = pd.DataFrame(rows)
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "day9_p3_constants.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    # summary checks
    for xi in (0.25, 0.5):
        r = df[(df.xi == xi) & (df.n == NS[-1])].ratio.values
        print(f"xi = {xi}: S_n/(C_xi n^-xi/2) at n = 4^14 = "
              f"{r.min():.4f}..{r.max():.4f} (both lambda; -> 1)")
    r1 = df[(df.xi == 1.0)]
    for lam in LAMS:
        v = r1[r1.lam == lam].resid_sqrt_n.values
        print(f"xi = 1, lam = {lam:.2f}: (S_n - lead) sqrt(n) = "
              + ", ".join(f"{x:+.4e}" for x in v)
              + "  (converges -> remainder is O(n^-1/2), no log)")


if __name__ == "__main__":
    main()
