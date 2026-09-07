"""Day 10, Study II: where is the maximizer of the error profile? (v0.5)

The v0.4 text claimed that for |sin x|^xi the sup of |F_{xi,lambda,n}| is
attained at the cusp x = 0 "throughout".  day6_xi.py never recorded the
argmax (sup_series / C_pred_profile return only max|F|), so the claim was
unverified.  This script records, for all six safe configurations
(xi in {1/4, 1/2, 1}) x (lambda in {1.1, 2} lambda_c):

  (A) the limiting profile
        F(x) = sum_{j<=J0} a_j (D_lam(2j) + 1/1300) cos(2jx)
               - (|sin x|^xi - a_0/2)/1300           (exact tail completion)
      -- F(0), ||F||_inf, argmax (grid on [0, pi/2] + local refinement);

  (B) the finite-n profiles from the FROZEN day-6 deviation tables
      (modes m = 2..1024, n = 256, 1024, 4096), with the tail j > 512
      completed inside the G-model (n^2 dev -> D_lam G(4j^2/n), validated
      to a few 1e-4 relative for m >= 48):
        F_n(x) = sum_{j<=512} a_j n^2 dev_n(2j) cos(2jx)
                 + sum_{512<j<=J1} a_j D_lam(2j) G(4j^2/n) cos(2jx)
      -- F_n(0), ||F_n||_inf, argmax, and the same quantities WITHOUT the
      tail completion (= the day-6 definition of the measured n^2 E_inf),
      to quantify the truncation bias of the frozen plateau ratios.

Structural criterion.  Near x = 0 the limit profile behaves as
    F(x) = F(0) - |x|^xi/1300 + (smoother),
because the datum enters F with the coefficient -1/1300 (D_lam -> -1/1300)
and every other contribution is smoother at 0.  Hence |F| has a sharp local
maximum at the cusp iff F(0) > 0, and a local MINIMUM there iff F(0) < 0:
the cusp can be the global maximizer of |F| only when F(0) > 0.

No new resolvent quadratures are performed; all inputs are frozen CSVs
(day-6) and the day-7 evaluator of G.

Output: data/day10_maximizer.csv (one row per configuration and n, plus
n = inf for the limit); console = record.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day6_xi import a_coeffs, D_coef, LAM_C, XIS, J0  # noqa: E402
from day7_G import G_pred                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

FRACS = (1.1, 2.0)
NS_SAFE = (256, 1024, 4096)
J1 = 20_000                        # G-model tail completion of F_n (the
                                   # remainder j > J1 is O(n J1^{-2-xi}))
XG = np.linspace(0.0, np.pi / 2, 40_001)    # F is even and pi-periodic,
                                            # symmetric about pi/2

_RG = np.logspace(-16, 12, 8000)
_GG = G_pred(_RG)


def G(rho):
    rho = np.asarray(rho, dtype=float)
    return np.where(rho > _RG[-1], 520.0 / rho,
                    np.interp(np.log(np.clip(rho, _RG[0], _RG[-1])),
                              np.log(_RG), _GG))


def refine_argmax(fun, x0, h):
    """local maximization of |fun| around x0 (bracket +-h)."""
    lo, hi = max(0.0, x0 - h), min(np.pi / 2, x0 + h)
    res = minimize_scalar(lambda x: -abs(fun(np.array([x]))[0]),
                          bounds=(lo, hi), method="bounded",
                          options=dict(xatol=1e-12))
    return res.x, -res.fun


def limit_profile(xi, lam):
    a0, a = a_coeffs(xi, J0)
    ms = 2.0 * np.arange(1, J0 + 1)
    coef = a * (D_coef(lam, ms) + 1.0 / 1300.0)

    def F(x):
        s = np.abs(np.sin(x)); s = np.where(s < 1e-14, 0.0, s)
        return np.cos(np.outer(x, ms)) @ coef - (s**xi - a0 / 2) / 1300.0
    return F


def finite_profile(xi, lam, n, dev, with_tail):
    """dev: frozen day-6 table for this lambda (columns m, n, logdev, sign)."""
    sub = dev[dev.n == n].sort_values("m")
    ms = sub.m.to_numpy(dtype=float)
    n2dev = sub.sign.to_numpy() * np.exp(2 * np.log(n) + sub.logdev.to_numpy())
    _, a = a_coeffs(xi, len(ms))
    coef, modes = a * n2dev, ms
    if with_tail:
        _, a1 = a_coeffs(xi, J1)
        j = np.arange(J0 + 1, J1 + 1, dtype=float)
        mt = 2.0 * j
        ct = a1[J0:] * D_coef(lam, mt) * G(mt**2 / n)
        coef, modes = np.concatenate([coef, ct]), np.concatenate([modes, mt])

    def F(x):
        # chunked to bound memory for the long tail
        out = np.zeros_like(x)
        for s in range(0, len(modes), 4096):
            out += np.cos(np.outer(x, modes[s:s + 4096])) @ coef[s:s + 4096]
        return out
    return F


def analyse(F, label):
    vals = F(XG)
    i = int(np.argmax(np.abs(vals)))
    x_star, sup = refine_argmax(F, XG[i], XG[1] - XG[0])
    F0 = F(np.array([0.0]))[0]
    # the bounded minimizer never returns the endpoint itself; a cusp
    # maximum at x = 0 must be compared explicitly
    if abs(vals[i]) > sup:
        x_star, sup = XG[i], abs(vals[i])
    if abs(F0) >= sup:
        x_star, sup = 0.0, abs(F0)
    at_cusp = x_star < 1e-6
    print(f"    {label:<28s} F(0) = {F0:+.5e}  sup = {sup:.5e}  "
          f"argmax = {x_star:.5f}  {'CUSP' if at_cusp else 'off-cusp'}")
    return F0, sup, x_star, at_cusp


def main():
    sys.stdout.reconfigure(line_buffering=True)
    rows = []
    devs = {frac: pd.read_csv(os.path.join(DATA, f"day6_dev_lam{int(frac*10)}.csv"))
            for frac in FRACS}
    rec = pd.read_csv(os.path.join(DATA, "day6_recovery.csv"))
    for xi in XIS:
        for frac in FRACS:
            lam = frac * LAM_C
            print(f"\n== xi = {xi}, lambda = {frac} lambda_c = {lam:.3f} ==")
            F = limit_profile(xi, lam)
            F0, sup, xs, cusp = analyse(F, "limit F")
            Cp = rec[(rec.lam_frac == frac) & (rec.xi == xi)].C_pred.iloc[0]
            assert abs(sup / Cp - 1) < 1e-4, (sup, Cp)   # reproduces day 6 (8193-grid)
            rows.append(dict(xi=xi, lam_frac=frac, n=np.inf, tail_mode="exact",
                             F0=F0, sup=sup, x_star=xs, cusp_is_max=cusp,
                             ratio_to_Cpred=1.0, sign_F0=np.sign(F0)))
            for n in NS_SAFE:
                for tail in (False, True):
                    Fn = finite_profile(xi, lam, n, devs[frac], tail)
                    lab = f"n={n} ({'G-tail' if tail else 'J0-trunc'})"
                    F0n, supn, xsn, cuspn = analyse(Fn, lab)
                    if not tail:
                        meas = rec[(rec.lam_frac == frac) & (rec.xi == xi)
                                   & (rec.n == n)].n2E_over_pred.iloc[0]
                        assert abs(supn / Cp / meas - 1) < 2e-3, (supn / Cp, meas)
                    rows.append(dict(xi=xi, lam_frac=frac, n=n,
                                     tail_mode="G-model" if tail else "none",
                                     F0=F0n, sup=supn, x_star=xsn,
                                     cusp_is_max=cuspn, ratio_to_Cpred=supn / Cp,
                                     sign_F0=np.sign(F0n)))
    df = pd.DataFrame(rows)
    out = os.path.join(DATA, "day10_maximizer.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    print("\n== summary: limit profile ==")
    lim = df[df.n == np.inf]
    for _, r in lim.iterrows():
        print(f"  xi={r.xi:<4} lam={r.lam_frac}lc: F(0) {'>' if r.F0 > 0 else '<'} 0, "
              f"maximizer {'at cusp' if r.cusp_is_max else f'at x*={r.x_star:.5f}'}")
    print("  -> cusp is the limit maximizer exactly when F(0) > 0 "
          f"({int(lim.cusp_is_max.sum())}/6 configurations)")

    print("\n== G-model estimate of the n at which the cusp overtakes the "
          "off-cusp bump (limit maximizer at cusp, finite-n maximizer not) ==")
    from day9_p3_constants import S_n as S_model
    for _, r in lim[lim.cusp_is_max].iterrows():
        fin = df[(df.xi == r.xi) & (df.lam_frac == r.lam_frac)
                 & (df.n == 4096) & (df.tail_mode == "G-model")].iloc[0]
        if fin.cusp_is_max:
            print(f"  xi={r.xi:<4} lam={r.lam_frac}lc: cusp already the "
                  f"maximizer at n = 4096")
            continue
        lam = r.lam_frac * LAM_C
        # the bump height is essentially n-independent (smooth point);
        # solve F(0) - S_n = bump for n by bisection in log n
        target = r.F0 - fin.sup
        lo, hi = np.log(4096.0), np.log(1e12)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if S_model(r.xi, lam, np.exp(mid)) > target:
                lo = mid
            else:
                hi = mid
        n_cross = np.exp(0.5 * (lo + hi))
        print(f"  xi={r.xi:<4} lam={r.lam_frac}lc: bump {fin.sup:.4e} at "
              f"x*={fin.x_star:.4f}, F(0)={r.F0:.4e}; cusp overtakes at "
              f"n ~ {n_cross:.2e}")

    print("\n== summary: plateau ratio n^2 E_inf / C_pred at n = 4096, "
          "J0-truncated (frozen day 6) vs G-model tail completed ==")
    for xi in XIS:
        for frac in FRACS:
            a = df[(df.xi == xi) & (df.lam_frac == frac) & (df.n == 4096)]
            t0 = a[a.tail_mode == "none"].ratio_to_Cpred.iloc[0]
            t1 = a[a.tail_mode == "G-model"].ratio_to_Cpred.iloc[0]
            print(f"  xi={xi:<4} lam={frac}lc: {t0:.3f} -> {t1:.3f}")


if __name__ == "__main__":
    main()
