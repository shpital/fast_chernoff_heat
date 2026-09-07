"""II.A3 (day 6): roughness family |sin x|^xi -- onset below lambda_c(f),
recovery above lambda_c.  Pre-registered design (no new branches):

  1. xi in {1/4, 1/2, 1}, datum f_xi = |sin x|^xi.
  2. Fourier coefficients in closed form,
         f_xi = a_0/2 + sum_j a_j(xi) cos(2jx),
         a_j(xi) = (-1)^j 2^{1-xi} Gamma(xi+1) /
                   [Gamma(xi/2-j+1) Gamma(xi/2+j+1)]
                 = -2^{1-xi} Gamma(xi+1) sin(pi xi/2)/pi
                   * Gamma(j-xi/2)/Gamma(j+xi/2+1)      (reflection, j>=1),
     one numerical cross-check (quadrature) per xi.
  3. lambda_c(f_xi): NOT searched again.  a_j(xi) != 0 for all j and all
     xi in (0,2) (sin(pi xi/2) != 0, Gammas finite), so supp fhat = all even
     modes and lambda_c(f_xi) = lambda_c(k=22) = 73.986718 (frozen day-4
     table) for every xi.  Roughness cannot move the threshold.
  4. lambda = 0.9 lambda_c: n_onset = min{n : k=22 is the largest modal
     error term}; prediction with no fitted exponents
         Delta n_onset ~ -Delta log|a_11(xi)| / Gamma_lambda(22);
     single-mode Laplace asymptotic ratio
         |dev(22,n)| / [sqrt(2 pi n/|Phi''|) e^{n Gamma}] -> 1.
     Onset scan: even n = 2..256 (measurement grid for n_onset only;
     primary error tables stay on the frozen NS).
  5. lambda = 1.1 lambda_c: recovery to the predicted limiting profile
         F_{xi,lambda}(x) = sum_j a_j(xi) D_lambda(2j) cos(2jx),
         D_lambda(k) = 6 (13k^6/105 - k^8/7800) / (lambda+k^2)^4,
         n^2 E_inf -> C_pred = ||F||_inf.
     Tail of F summed exactly through the datum: D_lambda(k) -> -1/1300 and
     D + 1/1300 = O(k^-2), so
         F = sum_{j<=J0} a_j (D(2j)+1/1300) cos(2jx)
             - (f_xi(x) - a_0/2 - partial_{J0}) / 1300.
  6. lambda = 2 lambda_c: clean-safe control (separates roughness recovery
     from near-critical slowing).
  7. n frozen: NS = {4,16,64,256,1024,4096}; modes m = 2..1024 even (J0=512)
     for measured E_inf (log-scaled series, day-4 method).

Soft secondary expectation (logged before the run, not a gate): the modal
n^{-2} asymptotic holds for k << sqrt(n) (z = k tau^{1/4} bounded in the
boundary layer), so the plateau deficit should scale roughly like the
fhat-mass beyond the crossover, i.e. n^{-xi/2}: rougher data reach the
plateau later.

Outputs: data/day6_*.csv, figs/day6_xi.png; console log = record.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import gammaln, gamma as gamma_fn
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day3_resolvent import log_resolvent, gamma_and_curvature  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")

LAM_C = 73.987479           # day-4 certified continuous threshold
LAM_C_F = 73.986718         # day-4 frozen lambda_c(k=22) = lambda_c(f) for all xi
XIS = (0.25, 0.5, 1.0)
NS = [4, 16, 64, 256, 1024, 4096]
J0 = 512                    # modes m = 2..1024 for measured E_inf
J_ONSET = 30                # modes m = 2..60 for the onset scan
N_SCAN = list(range(2, 257, 2))
XGRID = np.linspace(-np.pi, np.pi, 8193)


# ---------------------------------------------------------------------------
# coefficients
# ---------------------------------------------------------------------------
def a_coeffs(xi, J):
    j = np.arange(1, J + 1)
    pref = -2.0**(1 - xi) * gamma_fn(xi + 1) * np.sin(np.pi * xi / 2) / np.pi
    a = pref * np.exp(gammaln(j - xi / 2) - gammaln(j + xi / 2 + 1))
    a0 = 2.0**(1 - xi) * gamma_fn(xi + 1) / gamma_fn(xi / 2 + 1) ** 2
    return a0, a


def crosscheck_coeffs():
    print("== closed-form a_j(xi) vs quadrature ==")
    worst = 0.0
    for xi in XIS:
        a0, a = a_coeffs(xi, 60)
        for j in (1, 11, 50):
            num = (2 / np.pi) * quad(
                lambda x: np.sin(x)**xi * np.cos(2 * j * x),
                0.0, np.pi, limit=400)[0]
            rel = abs(num / a[j - 1] - 1.0)
            worst = max(worst, rel)
        num0 = (2 / np.pi) * quad(lambda x: np.sin(x)**xi, 0, np.pi)[0]
        worst = max(worst, abs(num0 / a0 - 1.0))
        print(f"  xi={xi}: a0={a0:+.8f}, a_11={a[10]:+.6e}  "
              f"(vs |sin| exact -4/(pi(4j^2-1)): "
              f"{'n/a' if xi != 1.0 else f'{abs(a[10]*np.pi*483/4+1):.1e}'})")
    print(f"  worst closed-form vs quadrature rel diff: {worst:.2e}")
    assert worst < 1e-9


# ---------------------------------------------------------------------------
# modal deviation tables (log domain, day-4 method)
# ---------------------------------------------------------------------------
def dev_table(lam, modes, ns, tag):
    """logdev[m][n], sign[m][n] for dev = R_{lam,n}(m) - 1/(lam+m^2)."""
    cachef = os.path.join(DATA, f"day6_dev_{tag}.csv")
    if os.path.exists(cachef):
        df = pd.read_csv(cachef)
        if set(df.m) >= set(modes) and set(df.n) >= set(ns):
            return df
    rows = []
    for m in modes:
        Rex = 1.0 / (lam + m * m)
        for n in ns:
            logR = log_resolvent(lam, float(m), n)
            if logR > np.log(Rex) + 50.0:
                rows.append(dict(m=m, n=n, logdev=logR, sign=1.0))
            else:
                dR = np.exp(logR) - Rex
                rows.append(dict(m=m, n=n,
                                 logdev=np.log(max(abs(dR), 1e-300)),
                                 sign=np.sign(dR) if dR != 0 else 1.0))
    df = pd.DataFrame(rows)
    df.to_csv(cachef, index=False)
    return df


def sup_series(logterms, signs, ms):
    """log sup_x |sum s_i e^{logterm_i} cos(m_i x)| by the day-4 scaling."""
    L = logterms.max()
    F = np.cos(np.outer(XGRID, ms)) @ (signs * np.exp(logterms - L))
    return L + np.log(np.max(np.abs(F)))


# ---------------------------------------------------------------------------
# part 1: unstable side, onset
# ---------------------------------------------------------------------------
def run_onset():
    lam = 0.9 * LAM_C
    print(f"== onset, lambda = 0.9 lambda_c = {lam:.3f} ==")
    gam22, tau22, d2_22 = gamma_and_curvature(lam, 22.0)
    print(f"  Gamma(22) = {gam22:+.6f}, tau* = {tau22:.6f}, "
          f"Phi'' = {d2_22:.1f}")
    modes = list(range(2, 2 * J_ONSET + 1, 2))
    df = dev_table(lam, modes, sorted(set(N_SCAN + NS)), "lam09")
    piv_ld = df.pivot(index="m", columns="n", values="logdev")

    onset_rows = []
    for xi in XIS:
        _, a = a_coeffs(xi, J_ONSET)
        loga = np.log(np.abs(a))
        n_onset, n_E1 = None, None
        for n in N_SCAN:
            terms = loga + piv_ld[n].loc[modes].to_numpy()
            if n_onset is None and modes[int(np.argmax(terms))] == 22:
                n_onset = n
            if n_E1 is None:
                logE = sup_series(terms, np.sign(a) *
                                  df[df.n == n].sort_values("m").sign.to_numpy(),
                                  np.array(modes, dtype=float))
                if logE > 0.0:
                    n_E1 = n
            if n_onset is not None and n_E1 is not None:
                break
        onset_rows.append(dict(xi=xi, a11=a[10], n_onset=n_onset, n_E1=n_E1))
        print(f"  xi={xi}: |a_11| = {abs(a[10]):.4e}, n_onset = {n_onset}, "
              f"first n with E_inf > 1: {n_E1}")

    print("  predicted vs measured onset shifts (naive single-competitor):")
    base = onset_rows[-1]  # xi = 1
    for r in onset_rows[:-1]:
        pred = -(np.log(abs(r["a11"])) - np.log(abs(base["a11"]))) / gam22
        meas = r["n_onset"] - base["n_onset"]
        print(f"    xi={r['xi']} vs 1: pred delta = {pred:+.1f}, "
              f"measured = {meas:+d}")
        r["pred_shift_vs_xi1"] = pred
        r["meas_shift_vs_xi1"] = meas
    pd.DataFrame(onset_rows).to_csv(os.path.join(DATA, "day6_onset.csv"),
                                    index=False)

    # single-mode Laplace asymptotic on the frozen NS
    print("  single-mode check at m=22 (datum-independent):")
    for n in NS:
        logpred = 0.5 * np.log(2 * np.pi * n / abs(d2_22)) + n * gam22
        ratio = np.exp(piv_ld[n].loc[22] - logpred)
        print(f"    n={n:>5}: dev(22)/[sqrt(2 pi n/|Phi''|) e^(n Gamma)] "
              f"= {ratio:.4f}")
    return piv_ld


# ---------------------------------------------------------------------------
# part 2: recovery side
# ---------------------------------------------------------------------------
def D_coef(lam, k):
    return 6.0 * (13.0 / 105.0 * k**6 - k**8 / 7800.0) / (lam + k**2) ** 4


def C_pred_profile(xi, lam):
    """||F_{xi,lambda}||_inf with the exact datum-tail completion:
    F = sum_{j<=J0} a_j (D(2j) + 1/1300) cos(2jx) - (f_xi - a0/2)/1300,
    identical to sum_{j<=J0} a_j D cos - (1/1300) sum_{j>J0} a_j cos.
    Also returns the J0-truncated prediction (no tail) for reference."""
    a0, a = a_coeffs(xi, J0)
    ms = 2.0 * np.arange(1, J0 + 1)
    cosm = np.cos(np.outer(XGRID, ms))
    s = np.abs(np.sin(XGRID))
    s[s < 1e-14] = 0.0                       # exact zeros at 0, +-pi
    F = cosm @ (a * (D_coef(lam, ms) + 1.0 / 1300.0)) - (s**xi - a0 / 2) / 1300.0
    F_head = cosm @ (a * D_coef(lam, ms))
    return np.max(np.abs(F)), np.max(np.abs(F_head))


def run_recovery():
    rows, cross_rows = [], []
    for frac in (1.1, 2.0):
        lam = frac * LAM_C
        print(f"== recovery, lambda = {frac} lambda_c = {lam:.3f} ==")
        modes = list(range(2, 2 * J0 + 1, 2))
        df = dev_table(lam, modes, NS, f"lam{int(frac*10)}")
        piv_ld = df.pivot(index="m", columns="n", values="logdev")
        piv_sg = df.pivot(index="m", columns="n", values="sign")

        # crossover profile: n^2 dev(m) / D_lambda(m) vs m (datum-free)
        ms_arr = np.array(modes, dtype=float)
        for n in NS:
            r = (piv_sg[n].loc[modes].to_numpy()
                 * np.exp(2 * np.log(n) + piv_ld[n].loc[modes].to_numpy())
                 / D_coef(lam, ms_arr))
            for m_, r_ in zip(modes, r):
                cross_rows.append(dict(lam_frac=frac, n=n, m=m_, ratio=r_))

        for xi in XIS:
            _, a = a_coeffs(xi, J0)
            Cp, Cp_head = C_pred_profile(xi, lam)
            loga = np.log(np.abs(a))
            line = []
            for n in NS:
                logE = sup_series(loga + piv_ld[n].loc[modes].to_numpy(),
                                  np.sign(a) * piv_sg[n].loc[modes].to_numpy(),
                                  ms_arr)
                ratio = np.exp(2 * np.log(n) + logE - np.log(Cp))
                rows.append(dict(lam_frac=frac, xi=xi, n=n, C_pred=Cp,
                                 C_pred_head=Cp_head,
                                 log10_Einf=logE / np.log(10),
                                 n2E_over_pred=ratio))
                line.append(f"{ratio:.3f}")
            print(f"  xi={xi}: C_pred = {Cp:.4e} (J0-truncated {Cp_head:.4e});"
                  f"  n^2 E_inf / C_pred = "
                  + "  ".join(f"{v}" for v in line) + f"   (n = {NS})")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day6_recovery.csv"), index=False)
    pd.DataFrame(cross_rows).to_csv(os.path.join(DATA, "day6_crossover.csv"),
                                    index=False)
    return df, pd.DataFrame(cross_rows)


def figure(df_rec, df_cross):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, frac in zip(axes[:2], (1.1, 2.0)):
        for xi in XIS:
            sub = df_rec[(df_rec.lam_frac == frac) & (df_rec.xi == xi)]
            ax.loglog(sub.n, sub.n2E_over_pred, "o-", label=f"xi={xi}")
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_title(f"lambda = {frac} lambda_c")
        ax.set_xlabel("n")
        ax.set_ylabel(r"$n^2 E_\infty / C^{pred}_{\xi,\lambda}$")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    ax = axes[2]
    for n in NS[1:]:
        sub = df_cross[(df_cross.lam_frac == 2.0) & (df_cross.n == n)
                       & (df_cross.m >= 40)]          # skip the D-zero region
        ax.semilogx(sub.m**2 / n, sub.ratio, lw=1.0, label=f"n={n}")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlabel(r"$m^2/n$")
    ax.set_ylabel(r"$n^2\,\mathrm{dev}(m)\,/\,D_\lambda(m)$")
    ax.set_title(r"crossover collapse (lambda = 2 lambda_c): "
                 r"$n^{-2}$ holds for $m \lesssim \sqrt{n}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.suptitle(r"II.A3: recovery of $n^{-2}$ toward the predicted "
                 r"limiting profile $F_{\xi,\lambda}$")
    fig.tight_layout()
    out = os.path.join(FIGS, "day6_xi.png")
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


if __name__ == "__main__":
    crosscheck_coeffs()
    print()
    print(f"lambda_c(f_xi) = lambda_c(k=22) = {LAM_C_F} for ALL xi "
          "(common even-mode support; frozen day-4 table, no new search)")
    print()
    run_onset()
    print()
    df_rec, df_cross = run_recovery()
    figure(df_rec, df_cross)
