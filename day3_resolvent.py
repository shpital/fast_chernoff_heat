"""II.A4, day 3: spectral resolvent kill-test for the RV amplification barrier.

Single-mode finite-n RV resolvent (substitution t = n tau):

    R_{lam,n}(k) = n * int_0^inf [ e^{-lam tau} M_tau(k) ]^n  d tau,

exact reference R_lam(k) = 1/(lam + k^2).  With even n the integrand is
e^{n Phi(tau)} >= 0, where

    Phi_{lam,k}(tau) = log|M_tau(k)| - lam*tau,      Phi(0) = 0,
    Gamma_lam(k)     = sup_{tau>0} Phi  (interior; boundary tau->0 gives 0).

Pre-registered protocol (fixed BEFORE the resolvent run):

  1. lambda_c2 = sup_{tau,k} log|M_tau(k)|/tau = sup_tau log g2(tau)/tau,
     computed first from the frozen Day-1 symbol.  Model prediction:
     g2 ~ 200 tau  =>  lambda_c2 ~ 200/e ~ 74.
  2. lambda / lambda_c2 in {0.5, 0.9, 1.1, 2}; fixed modes k in {1,2,4,8}
     plus the worst integer mode argmax_k Gamma_lam(k).
  3. For each (lam, k): Gamma_lam(k), tau*, Phi''(tau*), log-domain
     quadrature of R, and the Laplace-method check
         Gamma > 0:  R ~ n e^{n Gamma} sqrt(2 pi / (n |Phi''(tau*)|)),
         Gamma < 0:  R -> 1/(lam+k^2) with the n^{-2} law
                     n^2 (R - R_exact) -> 6 (13k^6/105 - k^8/7800)/(lam+k^2)^4.
  4. Only then |sin x| via its exact Fourier coefficients (no new datum).

n in {4, 16, 64, 256, 1024, 4096} (even => sign-free).
Outputs: data/day3_*.csv, figs/day3_resolvent.png.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day1_symbol import M_num, BETA_F, m_num  # noqa: E402


def M_tauvec(taus, k):
    """Symbol M_tau(k) vectorized over a tau array at fixed scalar k
    (same validated evaluator pieces as day1: series/closed-form m_j)."""
    taus = np.asarray(taus, dtype=float)
    u = np.sqrt(6.0 * taus) * k
    small = np.abs(u) < 1e-8
    first = np.where(small, 1.0 - u**2 / 6.0,
                     np.sin(u) / np.where(small, 1.0, u))
    z = k * taus**0.25
    second = sum(BETA_F[j] * taus * m_num(j, z) for j in (0, 2, 4, 6))
    return first + second

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")

NS = [4, 16, 64, 256, 1024, 4096]
LOG_CLIP = -700.0


# ---------------------------------------------------------------------------
# step 1: lambda_c2 from the frozen symbol (before any resolvent evaluation)
# ---------------------------------------------------------------------------
def lambda_c2():
    taus = np.geomspace(1e-4, 2.0, 400)
    best = (0.0, None, None)
    rows = []
    for tau in taus:
        kk = np.linspace(1e-6, 50.0 / tau**0.25, 200001)
        vals = np.abs(M_num(tau, kk))
        g2 = vals.max()
        lam = np.log(g2) / tau if g2 > 1 else 0.0
        rows.append(dict(tau=tau, g2=g2, k_at=kk[vals.argmax()], ratio=lam))
        if lam > best[0]:
            best = (lam, tau, kk[vals.argmax()])
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "day3_lambdac.csv"),
                              index=False)
    print(f"== step 1: lambda_c2 = {best[0]:.3f} at tau* = {best[1]:.5f}, "
          f"k = {best[2]:.2f}  (model 200/e = {200/np.e:.1f})")
    return best[0]


# ---------------------------------------------------------------------------
# Phi machinery
# ---------------------------------------------------------------------------
def phi_on_grid(lam, k, taus):
    vals = np.abs(M_tauvec(taus, k))
    with np.errstate(divide="ignore"):
        return np.log(np.maximum(vals, 1e-300)) - lam * taus


def gamma_and_curvature(lam, k):
    """Interior max of Phi (coarse grid + local refinement) and Phi''(tau*)."""
    taus = np.geomspace(1e-5, 3.0, 4000)
    ph = phi_on_grid(lam, k, taus)
    i = int(np.argmax(ph))
    if ph[i] <= 0.0 and i <= 1:          # boundary-dominated, no interior max
        return 0.0, None, None
    a = taus[max(i - 2, 0)]
    b = taus[min(i + 2, len(taus) - 1)]
    tt = np.linspace(a, b, 20001)
    pp = phi_on_grid(lam, k, tt)
    j = int(np.argmax(pp))
    tau_s, gam = tt[j], pp[j]
    h = (b - a) / 200
    d2 = (phi_on_grid(lam, k, np.array([tau_s - h]))[0]
          - 2 * gam + phi_on_grid(lam, k, np.array([tau_s + h]))[0]) / h**2
    return gam, tau_s, d2


def log_resolvent(lam, k, n):
    """log R_{lam,n}(k) for even n, via log-domain composite quadrature on a
    log-spaced tau grid (resolves both the tau->0 boundary layer of width
    ~1/(n(lam+k^2)) and the amplification band)."""
    from scipy.integrate import simpson
    tau_end = 3.0 + 20.0 / lam
    taus = np.geomspace(1e-14, tau_end, 300001)
    ph = phi_on_grid(lam, k, taus)
    m = ph.max()
    w = np.exp(np.maximum(n * (ph - max(m, 0.0)), LOG_CLIP))
    integral = simpson(w, x=taus)
    # boundary piece [0, 1e-14] contributes ~1e-14 * 1 -- add explicitly
    integral += 1e-14
    return np.log(n) + n * max(m, 0.0) + np.log(integral)


# ---------------------------------------------------------------------------
# steps 2-3: fixed modes + worst integer mode
# ---------------------------------------------------------------------------
def modes_experiment(lc2):
    print("== steps 2-3: Gamma, R, Laplace asymptotic ==")
    rows = []
    for frac in (0.5, 0.9, 1.1, 2.0):
        lam = frac * lc2
        # worst integer mode by Gamma
        gams = {m: gamma_and_curvature(lam, float(m))[0]
                for m in range(1, 61)}
        m_worst = max(gams, key=gams.get)
        ks = [1.0, 2.0, 4.0, 8.0] + [float(m_worst)]
        for k in ks:
            gam, tau_s, d2 = gamma_and_curvature(lam, k)
            R_exact = 1.0 / (lam + k**2)
            n2_pred = 6.0 * (13/105 * k**6 - k**8 / 7800) / (lam + k**2)**4
            for n in NS:
                logR = log_resolvent(lam, k, n)
                if gam > 0 and tau_s is not None and d2 is not None and d2 < 0:
                    log_asym = (np.log(n) + n * gam
                                + 0.5 * np.log(2 * np.pi / (n * abs(d2))))
                else:
                    log_asym = np.nan
                R = np.exp(logR) if logR < 700 else np.inf
                n2dev = n**2 * (R - R_exact) if np.isfinite(R) else np.nan
                rows.append(dict(
                    lam_frac=frac, lam=lam, k=k, worst=(k == float(m_worst)),
                    n=n, Gamma=gam, tau_star=tau_s, phi2=d2,
                    log10R=logR / np.log(10),
                    log10_asym=log_asym / np.log(10) if np.isfinite(log_asym)
                    else np.nan,
                    R_exact=R_exact, n2dev=n2dev, n2pred=n2_pred))
        sub = [r for r in rows if r["lam_frac"] == frac]
        wr = [r for r in sub if r["worst"] and r["n"] == NS[-1]][0]
        print(f"  lam = {frac:.1f}*lc2 = {lam:7.2f}: worst mode m = {m_worst}"
              f" Gamma = {gams[m_worst]:+.4f};"
              f" log10 R(n=4096) = {wr['log10R']:.1f}"
              f" (asym {wr['log10_asym']:.1f})" if gams[m_worst] > 0 else
              f"  lam = {frac:.1f}*lc2 = {lam:7.2f}: worst mode m = {m_worst}"
              f" Gamma = {gams[m_worst]:+.4f}; all modes safe")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day3_modes.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# step 4: |sin x| resolvent error
# ---------------------------------------------------------------------------
def abssin_experiment(lc2, J=200):
    print("== step 4: |sin x| resolvent error (exact Fourier route) ==")
    rows = []
    for frac in (0.5, 0.9, 1.1, 2.0):
        lam = frac * lc2
        j = np.arange(1, J + 1)
        a = -4.0 / (np.pi * (4.0 * j**2 - 1.0))
        m = 2.0 * j
        for n in NS:
            logdev = np.empty(J)
            for idx in range(J):
                k = m[idx]
                R_exact = 1.0 / (lam + k**2)
                logR = log_resolvent(lam, k, n)
                # |R - R_exact| in log domain
                R = np.exp(logR) if logR < 700 else np.inf
                dev = abs(R - R_exact) if np.isfinite(R) else np.inf
                logdev[idx] = np.log(max(dev, 1e-300)) if np.isfinite(dev) \
                    else logR
            logterm = np.log(np.abs(a)) + logdev
            lmax = logterm.max()
            # upper bound sum|a_j dev_j| via logsumexp; sup_x <= this bound
            lsum = lmax + np.log(np.sum(np.exp(logterm - lmax)))
            rows.append(dict(lam_frac=frac, lam=lam, n=n,
                             log10_E=lsum / np.log(10),
                             dom_mode=int(m[int(np.argmax(logterm))])))
        sub = [r for r in rows if r["lam_frac"] == frac]
        worst = max(sub, key=lambda r: r["log10_E"])
        last = [r for r in sub if r["n"] == NS[-1]][0]
        print(f"  lam = {frac:.1f}*lc2: max log10 E = {worst['log10_E']:6.1f} "
              f"at n = {worst['n']} (mode {worst['dom_mode']}); "
              f"log10 E(n=4096) = {last['log10_E']:6.1f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day3_abssin.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def figure(dfM, dfS, lc2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    colors = {0.5: "tab:red", 0.9: "tab:orange", 1.1: "tab:blue",
              2.0: "tab:green"}
    # worst-mode resolvent growth
    for frac, c in colors.items():
        sub = dfM[(dfM.lam_frac == frac) & dfM.worst]
        gam = sub.Gamma.iloc[0]
        axes[0].plot(sub.n, sub.log10R, "o-", color=c, ms=3,
                     label=f"$\\lambda={frac}\\lambda_c$, $\\Gamma="
                           f"{gam:+.3f}$")
        ok = sub.dropna(subset=["log10_asym"])
        if len(ok):
            axes[0].plot(ok.n, ok.log10_asym, "--", color=c, lw=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel(r"$\log_{10} R_{\lambda,n}(k_{\rm worst})$")
    axes[0].set_title("worst-mode resolvent (dashed: Laplace asymptotic)")
    # n^2 deviation for safe side, k=2
    for frac, c in colors.items():
        sub = dfM[(dfM.lam_frac == frac) & (dfM.k == 2.0)]
        if (sub.Gamma <= 0).all():
            axes[1].loglog(sub.n, np.abs(sub.n2dev), "o-", color=c, ms=3,
                           label=f"$\\lambda={frac}\\lambda_c$")
            axes[1].axhline(abs(sub.n2pred.iloc[0]), color=c, ls=":", lw=0.8)
    axes[1].set_xlabel("n")
    axes[1].set_ylabel(r"$n^2|R_{\lambda,n}-R_\lambda|$ at $k=2$")
    axes[1].set_title(r"safe modes: $n^{-2}$ law (dotted: predicted const)")
    # |sin| error
    for frac, c in colors.items():
        sub = dfS[dfS.lam_frac == frac]
        axes[2].plot(sub.n, sub.log10_E, "o-", color=c, ms=3,
                     label=f"$\\lambda={frac}\\lambda_c$")
    axes[2].axhline(0, color="k", lw=0.6)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("n")
    axes[2].set_ylabel(r"$\log_{10} E(|\sin x|)$")
    axes[2].set_title(r"$|\sin x|$ resolvent error")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"II.A4: spectral resolvent kill-test "
                 f"($\\lambda_c \\approx {lc2:.1f}$)")
    fig.tight_layout()
    out = os.path.join(FIGS, "day3_resolvent.png")
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


if __name__ == "__main__":
    lc2 = lambda_c2()
    print()
    dfM = modes_experiment(lc2)
    print()
    dfS = abssin_experiment(lc2)
    print()
    figure(dfM, dfS, lc2)
