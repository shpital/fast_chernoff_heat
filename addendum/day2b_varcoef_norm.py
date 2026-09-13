"""Study III, day 2b: what the day-2 rough-datum curve did NOT test.

The day-2 numerics live on a fixed grid (N = 512): the datum is replaced by
its trigonometric interpolant and the grid values are re-interpolated after
every step, so only a finite-dimensional approximation was checked.  Two
diagnostics closer to the target statement

    sup_x int |p_n(T,x,y) - p(T,x,y)| dy  <=  C n^{-2}          (*)

for the positive two-window scheme S(t):

  (i)  resolution dependence: the rough datum |sin x|^{1/2} at N = 128 ... 1024,
       sup-error vs the same-N spectral reference, n^2 E(n) as a function of N;
  (ii) the discrete operator L_inf -> L_inf norm of the error,
           ||A_tau^n - exp(T L_N)||_{inf->inf} = max_i sum_j |(.)_{ij}|,
       where A_tau = Re(M F)/N is the one-step matrix acting on grid values and
       L_N the spectral collocation operator.  The row sum is the discrete
       analogue of the y-integral in (*), so n^2 times this norm, if bounded
       uniformly in N, is the numerical proxy of (*) on the grid scale.  Also
       recorded: max row sum of |A_tau| (the continuous S(t) is positive with
       S(t)1 = 1 when c = 0; the discretised kernel is not sign-definite because
       trigonometric interpolation of a window has Gibbs oscillations).

Cases: c = 0 (Markov, ||S(t)|| = 1 exactly) -- the first theorem target -- and
the day-2 coefficients with c = 0.2 sin 2x.  T = 0.5.
"""
import os
import time

import numpy as np
import pandas as pd
import sympy as sp
from scipy.linalg import expm

from day2_varcoef_positive import (solve_construction, make_numeric, transfer_matrix,
                                   spectral_reference, TEST_A, TEST_B, TEST_C, DATA, FIGS)

T = 0.5
NS_GRID = [128, 256, 512, 1024]
N_STEPS = [8, 16, 32, 64, 128, 256, 512]


def collocation_operator(coef, N):
    xg = 2 * np.pi * np.arange(N) / N
    k = np.fft.fftfreq(N, d=1.0 / N)
    F = np.fft.fft(np.eye(N), axis=0)
    Finv = np.linalg.inv(F)
    D1 = np.real(Finv @ np.diag(1j * k) @ F)
    D2 = np.real(Finv @ np.diag(-(k ** 2)) @ F)
    A, B, C = (np.diag(cf(xg)) for cf in coef)
    return A @ D2 + B @ D1 + C, F


def matrix_power(A, n):
    """A^n by binary powering (n = 2^m)."""
    P = A.copy()
    m = int(round(np.log2(n)))
    assert 2 ** m == n
    for _ in range(m):
        P = P @ P
    return P


def run_case(tag, Cc):
    print(f"== case {tag}: a = {TEST_A}, b = {TEST_B}, c = {Cc} ==", flush=True)
    params = solve_construction(verbose=False)
    exprs, coef = make_numeric(params, TEST_A, TEST_B, Cc)
    rows = []
    for N in NS_GRID:
        t0 = time.time()
        xg = 2 * np.pi * np.arange(N) / N
        L, F = collocation_operator(coef, N)
        E_T = expm(T * L)
        u0 = np.abs(np.sin(xg)) ** 0.5
        uref = E_T @ u0
        for n in N_STEPS:
            M = transfer_matrix(exprs, N, T / n)
            if M is None:
                rows.append(dict(case=tag, N=N, n=n, positive_step=False))
                continue
            A = np.real(M @ F) / N                      # one-step matrix on grid values
            An = matrix_power(A, n)
            Err = An - E_T
            op_norm = np.max(np.sum(np.abs(Err), axis=1))
            a_abs = np.max(np.sum(np.abs(A), axis=1))
            an_abs = np.max(np.sum(np.abs(An), axis=1))
            e_rough = np.max(np.abs(An @ u0 - uref))
            rows.append(dict(case=tag, N=N, n=n, positive_step=True,
                             rough_sup_err=e_rough, n2_rough=n ** 2 * e_rough,
                             op_norm=op_norm, n2_op_norm=n ** 2 * op_norm,
                             abs_rowsum_A=a_abs, abs_rowsum_An=an_abs,
                             rowsum_An_min=np.min(np.sum(An, axis=1)),
                             rowsum_An_max=np.max(np.sum(An, axis=1))))
        sub = [r for r in rows if r["N"] == N and r.get("positive_step")]
        print(f"  N={N:5d}: n^2 ||A^n - e^{{TL}}||_inf = "
              + ", ".join(f"n={r['n']}: {r['n2_op_norm']:.4f}" for r in sub)
              + f"   [{time.time()-t0:.0f}s]", flush=True)
        print(f"          n^2 sup-err(|sin|^1/2)     = "
              + ", ".join(f"n={r['n']}: {r['n2_rough']:.4f}" for r in sub), flush=True)
        print(f"          max row sum |A|: "
              + ", ".join(f"n={r['n']}: {r['abs_rowsum_A']:.4f}" for r in sub)
              + ";  max row sum |A^n| at n=512: "
              f"{sub[-1]['abs_rowsum_An']:.4f};  A^n 1 in "
              f"[{sub[-1]['rowsum_An_min']:.6f}, {sub[-1]['rowsum_An_max']:.6f}]", flush=True)
    return rows


def figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, tag in enumerate(("c=0", "c=0.2sin2x")):
        for N in NS_GRID:
            sub = df[(df.case == tag) & (df.N == N) & (df.positive_step == True)]
            ax[j].semilogx(sub.n, sub.n2_op_norm, "o-", label=f"N={N}: n^2 ||A^n-e^TL||")
            ax[j].semilogx(sub.n, sub.n2_rough, "s:", ms=3, label=f"N={N}: n^2 sup-err |sin|^1/2")
        ax[j].set_xlabel("n"); ax[j].set_title(f"discrete error-operator norm, {tag}")
        ax[j].legend(fontsize=6, ncol=2)
    ax[0].set_ylabel("n^2 x error")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "day2b_varcoef_norm.png"), dpi=150)


if __name__ == "__main__":
    t_all = time.time()
    rows = run_case("c=0", sp.Integer(0)) + run_case("c=0.2sin2x", TEST_C)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day2b_varcoef_norm.csv"), index=False)
    figure(df)
    print(f"== done in {time.time()-t_all:.0f}s ==")
