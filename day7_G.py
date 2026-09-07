"""II.A theory closure (day 7): the crossover function G(rho) in closed form.

Claim (parabolic boundary layer, k^2 = rho n, u = n k^2 tau):

    n^2 [ R_{lambda,n}(k_n) - 1/(lambda+k_n^2) ]  ->  H(rho),
    H(rho) = (1/rho) int_0^inf e^{-u} [ -u^2/5
             + (u/rho) Phat((rho u)^{1/4}) ] du,
    Phat(z) = int_{-1}^1 P(y) cos(zy) dy  (one-step correction kernel),

lambda drops out at leading order.  With D_lambda(k) -> -1/1300,

    G(rho) = -1300 H(rho),   G(0) = 1  (exact cancellation of the y^4
    moment against the sinc u^2-term),   G(rho) ~ 520/rho  (rho -> inf,
    from H ~ -2/(5 rho)).

Robust evaluation:
  * series in sqrt(rho) from the exact moments M_{2p} = int P y^{2p} dy
    (P is degree 6 => all moments are exact rationals):
        H(rho) = sum_{p>=3} (-1)^p M_{2p}/(2p)! Gamma(p/2+2) rho^{p/2-2}
    (the p=2 term cancels -u^2/5 exactly; M6 = 0).  Used for rho <= 4,
    where direct quadrature loses the cancellation in float64.
  * Gauss-Legendre quadrature on u in [0, 80] with Phat from the day-1
    m_j evaluators for rho > 2; overlap 2 <= rho <= 4 cross-checked.

Verification (existing data only, no new sweeps):
  1. G_pred vs the day-6 collapse CSV (both lambda), m >= 48, n = 256..4096;
  2. lambda-independence of the measured collapse;
  3. recovery law inside the G-model: plateau deficit ~ n^{-xi/2} with the
     constant int y^{-1-xi} (1 - G(4y^2)) dy; model extrapolated to n up to
     4^12 where the slow xi = 1/4 asymptotics becomes visible; model is
     validated against the measured day-6 recovery ratios at n = 256..4096.

Outputs: data/day7_Gcheck.csv, data/day7_recovery_model.csv,
figs/day7_G.png; console = record.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import gammaln, gamma as gamma_fn
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day1_symbol import m_num, BETA_F  # noqa: E402
from day6_xi import a_coeffs, D_coef, LAM_C, XIS, NS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")

CJ = {j: BETA_F[j] for j in (0, 2, 4, 6)}      # P(y) = sum c_j y^j


def Phat(z):
    """int_{-1}^1 P(y) cos(zy) dy via the validated day-1 m_j evaluators."""
    z = np.asarray(z, dtype=float)
    return sum(CJ[j] * m_num(j, z) for j in (0, 2, 4, 6))


def moments(pmax=80):
    """M_{2p} = int_{-1}^1 P(y) y^{2p} dy = 2 sum_j c_j/(2p+j+1), exact."""
    return {p: 2.0 * sum(c / (2 * p + j + 1) for j, c in CJ.items())
            for p in range(pmax + 1)}


M2P = moments()


def H_series(rho, pmax=80):
    """sum_{p>=3} (-1)^p M_{2p}/(2p)! Gamma(p/2+2) rho^{p/2-2} (rho <= ~4)."""
    rho = np.asarray(rho, dtype=float)
    out = np.zeros_like(rho)
    for p in range(3, pmax + 1):
        c = ((-1) ** p * M2P[p]
             * np.exp(gammaln(p / 2 + 2) - gammaln(2 * p + 1)))
        out += c * rho ** (p / 2 - 2)
    return out


GLX, GLW = np.polynomial.legendre.leggauss(128)
U_NODES = 40.0 * (GLX + 1.0)          # [0, 80]
U_WTS = 40.0 * GLW


def H_quad(rho):
    rho = np.asarray(rho, dtype=float)
    out = np.empty_like(rho)
    for i, r in enumerate(rho):
        z = (r * U_NODES) ** 0.25
        integ = np.exp(-U_NODES) * (-U_NODES**2 / 5.0
                                    + U_NODES / r * Phat(z))
        out[i] = integ @ U_WTS / r
    return out


def H_eval(rho):
    rho = np.asarray(rho, dtype=float)
    out = np.empty_like(rho)
    lo = rho <= 3.0
    if lo.any():
        out[lo] = H_series(rho[lo])
    if (~lo).any():
        out[~lo] = H_quad(rho[~lo])
    return out


def G_pred(rho):
    return -1300.0 * H_eval(rho)


def sanity():
    print("== sanity of H(rho) ==")
    print(f"  moments: M4 = {M2P[2]:.6f} (exact 24/5), "
          f"M6 = {M2P[3]:.2e} (exact 0), M8 = {M2P[4]:.6f} (exact -336/65), "
          f"M10 = {M2P[5]:.6f}")
    ov = np.array([2.0, 2.5, 3.0, 4.0])
    d = np.abs(H_series(ov) / H_quad(ov) - 1.0).max()
    print(f"  series vs quadrature on overlap rho in [2,4]: "
          f"max rel diff = {d:.2e}")
    assert d < 1e-9
    print(f"  G(0) limit: G(1e-8) = {G_pred(np.array([1e-8]))[0]:.8f} "
          "(predicted exactly 1)")
    big = np.array([1e4, 1e5, 1e6])
    print("  tail: G(rho)*rho/520 =",
          np.array2string(G_pred(big) * big / 520.0, precision=4),
          "(predicted -> 1)")


def check_collapse():
    """Two forms of the comparison:
    (a) factorized:  n^2 dev(m) =? D_lambda(m) * G(m^2/n)
        -- the day-6 collapse ordinate against G_pred;
    (b) pure limit:  n^2 dev(m) -> H(m^2/n)
        -- deviations at small m are the known finite-k factor
           D_lambda(m)/(-1/1300) = 1 + O((lambda+966)/m^2), not a failure
           of the boundary layer."""
    print("== G_pred vs measured day-6 collapse (no new sweeps) ==")
    cr = pd.read_csv(os.path.join(DATA, "day6_crossover.csv"))
    rows = []
    for frac in (1.1, 2.0):
        lam = frac * LAM_C
        for n in (256, 1024, 4096):
            sub = cr[(cr.lam_frac == frac) & (cr.n == n)
                     & (cr.m >= 48)].sort_values("m")
            ms = sub.m.to_numpy(dtype=float)
            pred_H = H_eval(ms**2 / n)
            pred_G = -1300.0 * pred_H
            rel_f = np.abs(sub.ratio.to_numpy() / pred_G - 1.0)
            meas_H = sub.ratio.to_numpy() * D_coef(lam, ms)
            rel_p = np.abs(meas_H / pred_H - 1.0)
            rows += [dict(lam_frac=frac, n=n, m=m_, G_meas=g, G_pred=gp,
                          rel_fact=rf, rel_pure=rp)
                     for m_, g, gp, rf, rp in
                     zip(ms, sub.ratio.to_numpy(), pred_G, rel_f, rel_p)]
            print(f"  lam={frac}lc n={n:>5}: factorized median "
                  f"{np.median(rel_f):.2e}, max {rel_f.max():.2e};  "
                  f"pure-limit median {np.median(rel_p):.2e}, "
                  f"max {rel_p.max():.2e}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day7_Gcheck.csv"), index=False)
    # the pure-limit max sits at m=48 and equals the finite-k D factor:
    lam = 2.0 * LAM_C
    fac = 1.0 - D_coef(lam, 48.0) * (-1300.0)
    mx = df[(df.lam_frac == 2.0) & (df.m == 48)].rel_pure.max()
    print(f"  pure-limit deviation at m=48: {mx:.4f} vs "
          f"1 - (-1300)D_lambda(48) = {fac:.4f} (finite-k factor, "
          "n-independent)")
    # lambda independence of the measured layer itself
    a = cr[(cr.lam_frac == 1.1) & (cr.n == 4096) & (cr.m >= 48)].ratio.to_numpy()
    b = cr[(cr.lam_frac == 2.0) & (cr.n == 4096) & (cr.m >= 48)].ratio.to_numpy()
    print(f"  measured lambda-independence at n=4096: "
          f"max |ratio_1.1/ratio_2.0 - 1| = {np.abs(a/b-1).max():.2e}")
    return df


def recovery_model():
    """Plateau deficit inside the G-model; validated against day-6 ratios."""
    print("== recovery law inside the G-model ==")
    lam = 2.0 * LAM_C
    # G on a log grid + spline (fast evaluation for huge mode counts)
    lg = np.linspace(-8, 10, 720)
    Gt = G_pred(10.0**lg)
    Gsp = CubicSpline(lg, Gt)

    def Gfast(rho):
        rho = np.clip(rho, 1e-8, 1e10)
        return np.where(rho >= 1e10, 0.0, Gsp(np.log10(rho)))

    xgrid = np.linspace(-np.pi, np.pi, 2049)
    rows = []
    # validation at measured n (sup over x, modes to J = 512 as in day 6)
    rec = pd.read_csv(os.path.join(DATA, "day6_recovery.csv"))
    for xi in XIS:
        a0, a = a_coeffs(xi, 512)
        ms = 2.0 * np.arange(1, 513)
        cosm = np.cos(np.outer(xgrid, ms))
        Cp = rec[(rec.lam_frac == 2.0) & (rec.xi == xi)].C_pred.iloc[0]
        line = []
        for n in NS[3:]:
            Fn = cosm @ (a * D_coef(lam, ms) * Gfast(ms**2 / n))
            model = np.max(np.abs(Fn)) / Cp
            meas = rec[(rec.lam_frac == 2.0) & (rec.xi == xi)
                       & (rec.n == n)].n2E_over_pred.iloc[0]
            line.append(f"n={n}: model {model:.3f} / meas {meas:.3f}")
        print(f"  validation xi={xi} (lam=2lc): " + ";  ".join(line))

    # asymptotic law from the cusp point x = 0 (coherent tail), large n.
    # Mode sums truncated at J with ANALYTIC tails (a_j ~ -C_xi j^{-1-xi},
    # D -> -1/1300, G ~ 520 n/(4 j^2)); without them the truncation bias of
    # F(0) is of the same order as the deficit at large n.
    print("  cusp-deficit law (model, x=0): local slope of deficit vs n")
    J = 4_000_000
    for xi in XIS:
        a0, a = a_coeffs(xi, J)
        j = np.arange(1, J + 1)
        D = D_coef(lam, 2.0 * j)
        Cxi = 2.0**(1 - xi) * gamma_fn(xi + 1) * np.sin(np.pi * xi / 2) / np.pi
        F0 = np.sum(a * D) + (Cxi / 1300.0) * J**(-xi) / xi
        defs = []
        ns = [4**k for k in range(4, 13)]
        for n in ns:
            tail_n = (Cxi / 1300.0) * 130.0 * n * J**(-2 - xi) / (2 + xi)
            F0n = np.sum(a * D * Gfast(4.0 * j**2 / n)) + tail_n
            defs.append(F0 - F0n)
            rows.append(dict(xi=xi, n=n, deficit0=defs[-1]))
        defs = np.array(defs)
        slopes = np.log(defs[1:] / defs[:-1]) / np.log(4.0)
        print(f"    xi={xi}: slopes {np.array2string(slopes, precision=3)} "
              f"(predicted -> {-xi/2})")
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "day7_recovery_model.csv"),
                              index=False)


def figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    rr = np.geomspace(1e-2, 1e5, 400)
    ax.semilogx(rr, G_pred(rr), "k-", lw=1.6, label=r"$G(\rho)=-1300H(\rho)$")
    cr = pd.read_csv(os.path.join(DATA, "day6_crossover.csv"))
    for n, c in zip((256, 1024, 4096), ("C0", "C1", "C2")):
        sub = cr[(cr.lam_frac == 2.0) & (cr.n == n) & (cr.m >= 48)]
        ax.semilogx(sub.m**2 / n, sub.ratio
                    * D_coef(2.0 * LAM_C, sub.m.to_numpy(dtype=float))
                    * (-1300.0), "o", ms=3, color=c, label=f"measured n={n}")
    ax.set_xlabel(r"$\rho = m^2/n$")
    ax.set_ylabel(r"$-1300\, n^2\,\mathrm{dev}(m)$")
    ax.set_title("crossover function: prediction vs day-6 data")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    dfm = pd.read_csv(os.path.join(DATA, "day7_recovery_model.csv"))
    for xi in XIS:
        sub = dfm[dfm.xi == xi]
        ax.loglog(sub.n, sub.deficit0, "o-", label=f"xi={xi}")
        ax.loglog(sub.n, sub.deficit0.iloc[-1]
                  * (sub.n / sub.n.iloc[-1])**(-xi / 2), "k:", lw=0.8)
    ax.set_xlabel("n")
    ax.set_ylabel("cusp deficit $F(0)-F_n(0)$ (model)")
    ax.set_title(r"recovery law: deficit $\sim n^{-\xi/2}$ (dotted)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.suptitle("II.A theory closure: parabolic boundary layer "
                 r"$k^2 = \rho n$")
    fig.tight_layout()
    out = os.path.join(FIGS, "day7_G.png")
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


if __name__ == "__main__":
    sanity()
    print()
    df = check_collapse()
    print()
    recovery_model()
    figure(df)
