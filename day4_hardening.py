"""II.A4 hardening pass (day 4): cement the lambda_c result before freeze.

Three pre-registered tasks (no new datum classes, no new sweeps):

  1. lambda_c certification: global 2D scan in the natural coordinates
     (tau, z = k tau^{1/4}), followed by high-precision local optimization
     (Nelder-Mead on -log|M|/tau); range-extension check (larger tau, z
     windows must not move the maximum).  Also the datum-specific threshold
         lambda_c(f) = sup_{tau, k in supp fhat} log|M_tau(k)|/tau
     for |sin x| (even integer modes).
  2. Critical line lambda = lambda_c: prediction (nondegenerate interior
     maximum, Gamma = 0, Phi''(tau*) < 0)
         R_{lambda_c,n}(k*) ~ sqrt(2 pi n / |Phi''(tau*)|),
     i.e. R / sqrt(n) -> sqrt(2 pi / |Phi''|).  Three-regime phase picture:
     sqrt(n) e^{n Gamma} / sqrt(n) / R_lambda + C n^{-2}.
  3. True E_inf for |sin x| replacing the spectral upper bound: log-scaled
     partial series
         log E_inf = L + log max_x |F(x)|,
         F(x) = sum_j sgn(a_j dR_j) exp(log|a_j dR_j| - L) cos(2jx),
     with L = max_j log|a_j dR_j|; report also bound/true ratio.

Outputs: data/day4_*.csv; console log is the certification record.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day3_resolvent import M_tauvec, log_resolvent, gamma_and_curvature  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

NS = [4, 16, 64, 256, 1024, 4096]


def ratio_tz(tau, z):
    """log|M_tau(k)| / tau at k = z / tau^{1/4}."""
    k = z / tau**0.25
    M = abs(float(M_tauvec(np.array([tau]), k)[0]))
    if M <= 1.0:
        return 0.0
    return np.log(M) / tau


# ---------------------------------------------------------------------------
# task 1: lambda_c certification
# ---------------------------------------------------------------------------
def certify_lambda_c():
    print("== task 1: lambda_c certification ==")
    # global scan in (tau, z)
    from day1_symbol import M_num
    taus = np.geomspace(3e-4, 3.0, 400)
    zs = np.linspace(0.5, 40.0, 400)
    best = (-1.0, None, None)
    for tau in taus:
        kk = zs / tau**0.25
        vals = np.abs(M_num(tau, kk))
        with np.errstate(divide="ignore"):
            r = np.where(vals > 1, np.log(np.maximum(vals, 1e-300)) / tau, 0.0)
        i = int(np.argmax(r))
        if r[i] > best[0]:
            best = (r[i], tau, zs[i])
    print(f"  coarse (tau,z) scan: ratio={best[0]:.4f} at tau={best[1]:.5f}, "
          f"z={best[2]:.3f}")

    # range-extension check
    ext_best = best[0]
    for tau in np.geomspace(1.0, 10.0, 60):
        from day1_symbol import M_num
        kk = np.linspace(0.5, 100.0, 4000) / tau**0.25
        vals = np.abs(M_num(tau, kk))
        with np.errstate(divide="ignore"):
            r = np.where(vals > 1, np.log(np.maximum(vals, 1e-300)) / tau, 0.0)
        ext_best = max(ext_best, r.max())
    print(f"  range extension (tau up to 10, z up to 100): max ratio "
          f"{ext_best:.4f} -- {'unchanged' if abs(ext_best-best[0])<1e-9 else 'MOVED'}")

    # local high-precision optimization
    res = minimize(lambda p: -ratio_tz(np.exp(p[0]), p[1]),
                   x0=[np.log(best[1]), best[2]],
                   method="Nelder-Mead",
                   options=dict(xatol=1e-12, fatol=1e-12, maxiter=20000))
    lam_c = -res.fun
    tau_c = float(np.exp(res.x[0]))
    z_c = float(res.x[1])
    k_c = z_c / tau_c**0.25
    print(f"  refined: lambda_c = {lam_c:.6f} at tau* = {tau_c:.6f}, "
          f"z* = {z_c:.4f} (k* = {k_c:.4f})")

    # datum-specific threshold for |sin x| (even integer modes)
    rows = []
    for m in range(2, 61, 2):
        taus = np.geomspace(1e-4, 3.0, 3000)
        vals = np.abs(M_tauvec(taus, float(m)))
        with np.errstate(divide="ignore"):
            r = np.where(vals > 1, np.log(np.maximum(vals, 1e-300)) / taus, 0.0)
        i = int(np.argmax(r))
        if r[i] > 0:
            lo = np.log(taus[max(i - 2, 0)])
            hi = np.log(taus[min(i + 2, len(taus) - 1)])
            ref = minimize_scalar(
                lambda lt: -ratio_tz(np.exp(lt), float(m) * np.exp(lt)**0.25),
                bounds=(lo, hi), method="bounded",
                options=dict(xatol=1e-13))
            lck = max(-ref.fun, r[i])
        else:
            lck = 0.0
        rows.append(dict(mode=m, lambda_c_k=lck))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day4_lambdac_modes.csv"), index=False)
    top = df.sort_values("lambda_c_k", ascending=False).head(3)
    lam_c_f = top.lambda_c_k.iloc[0]
    print("  datum-specific lambda_c(|sin|) over even modes:")
    for _, r in top.iterrows():
        print(f"    mode {int(r['mode']):>2}: lambda_c(k) = {r.lambda_c_k:.6f}")
    print(f"  gap to continuous: lambda_c - lambda_c(|sin|) = "
          f"{lam_c - lam_c_f:.6f}")
    return lam_c, tau_c, k_c, lam_c_f, int(top['mode'].iloc[0])


# ---------------------------------------------------------------------------
# task 2: critical line
# ---------------------------------------------------------------------------
def critical_line(lam_c, k_c):
    print("== task 2: critical line lambda = lambda_c: R ~ sqrt(2 pi n/|Phi''|) ==")
    gam, tau_s, d2 = gamma_and_curvature(lam_c, k_c)
    C_pred = np.sqrt(2 * np.pi / abs(d2))
    print(f"  at (lambda_c, k*): Gamma = {gam:+.2e}, tau* = {tau_s:.6f}, "
          f"Phi'' = {d2:.4f}, predicted C = sqrt(2pi/|Phi''|) = {C_pred:.6f}")
    rows = []
    for n in NS:
        logR = log_resolvent(lam_c, k_c, n)
        R = np.exp(logR)
        rows.append(dict(n=n, R=R, R_over_sqrtn=R / np.sqrt(n),
                         ratio_to_pred=R / (np.sqrt(n) * C_pred)))
        print(f"    n={n:>5}: R = {R:12.6f}, R/sqrt(n) = {R/np.sqrt(n):.6f}, "
              f"ratio to prediction = {R/(np.sqrt(n)*C_pred):.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "day4_critical.csv"),
                              index=False)


# ---------------------------------------------------------------------------
# task 3: true E_inf for |sin x|
# ---------------------------------------------------------------------------
def true_Einf(lam_c, J=200):
    print("== task 3: true E_inf(|sin|) via log-scaled series ==")
    xg = np.linspace(-np.pi, np.pi, 8193)
    rows = []
    for frac in (0.5, 0.9, 1.1, 2.0):
        lam = frac * lam_c
        j = np.arange(1, J + 1)
        a = -4.0 / (np.pi * (4.0 * j**2 - 1.0))
        m = 2.0 * j
        for n in NS:
            logdev = np.empty(J)
            sgn = np.empty(J)
            for idx in range(J):
                k = m[idx]
                R_exact = 1.0 / (lam + k**2)
                logR = log_resolvent(lam, k, n)
                if logR > np.log(R_exact) + 50:
                    logdev[idx], sgn[idx] = logR, 1.0
                else:
                    dR = np.exp(logR) - R_exact
                    logdev[idx] = np.log(max(abs(dR), 1e-300))
                    sgn[idx] = np.sign(dR) if dR != 0 else 1.0
            logterm = np.log(np.abs(a)) + logdev
            s = np.sign(a) * sgn
            L = logterm.max()
            log_bound = L + np.log(np.sum(np.exp(logterm - L)))
            F = np.cos(np.outer(xg, m)) @ (s * np.exp(logterm - L))
            log_true = L + np.log(np.max(np.abs(F)))
            rows.append(dict(lam_frac=frac, lam=lam, n=n,
                             log10_E_true=log_true / np.log(10),
                             log10_E_bound=log_bound / np.log(10),
                             bound_over_true=np.exp(log_bound - log_true),
                             dom_mode=int(m[int(np.argmax(logterm))])))
        sub = [r for r in rows if r["lam_frac"] == frac]
        last = [r for r in sub if r["n"] == NS[-1]][0]
        wr = max(sub, key=lambda r: r["log10_E_true"])
        print(f"  lam = {frac}*lc: max log10 E_true = {wr['log10_E_true']:8.1f}"
              f" at n={wr['n']} (bound/true {wr['bound_over_true']:.3f});"
              f" n=4096: log10 E_true = {last['log10_E_true']:8.2f}"
              f" (bound/true {last['bound_over_true']:.3f})")
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "day4_abssin_true.csv"),
                              index=False)


if __name__ == "__main__":
    lam_c, tau_c, k_c, lam_c_f, m_worst = certify_lambda_c()
    print()
    critical_line(lam_c, k_c)
    print()
    true_Einf(lam_c)
