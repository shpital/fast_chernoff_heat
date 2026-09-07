"""Paper figures for Study II v0.1 -- replots of FROZEN data only (no new
computations beyond re-evaluating the closed-form G(rho) curve).

fig1_band.png       : one-step symbol |M_tau(k)| on the correction scale
                      z = k tau^{1/4}; amplification band at z* ~ 7.6.
fig2_transition.png : (a) critical line R/sqrt(n) -> C_pred at lambda_c;
                      (b) measured sup-error of |sin x| across the threshold.
fig3_G.png          : factorized crossover n^2 dev/D_lambda vs G(rho) (MAIN),
                      pure-limit form -1300 n^2 dev as inset (finite-k hooks).
fig4_recovery.png   : (a) n^2 E_inf / C_pred plateaus; (b) model cusp deficit
                      with n^{-xi/2} guides.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.dirname(HERE)
sys.path.insert(0, UP)
DATA = os.path.join(UP, "data")
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

from day1_symbol import M_num                       # noqa: E402
from day6_xi import D_coef, LAM_C, XIS, NS          # noqa: E402
from day7_G import G_pred                           # noqa: E402


def fig1():
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for tau, c in zip((0.25, 0.05, 0.01, 0.005), ("C0", "C1", "C2", "C3")):
        kk = np.linspace(1e-4, 25.0 / tau**0.25, 6000)
        ax.semilogy(kk * tau**0.25, np.abs(M_num(tau, kk)), lw=1.0, color=c,
                    label=rf"$\tau={tau}$")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"$z = k\,\tau^{1/4}$")
    ax.set_ylabel(r"$|M_\tau(k)|$")
    ax.set_xlim(0, 25)
    ax.set_ylim(1e-4, 1e2)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_band.png"), dpi=170)


def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8))
    ax = axes[0]
    cr = pd.read_csv(os.path.join(DATA, "day4_critical.csv"))
    ax.semilogx(cr.n, cr.R_over_sqrtn, "o-", color="C0", lw=1.0,
                label=r"$R_{\lambda_c,n}(k_*)/\sqrt{n}$")
    Cpred = 0.0223532
    ax.axhline(Cpred, color="k", lw=0.9, ls=":",
               label=r"$\sqrt{2\pi/|\Phi''(\tau_*)|}$")
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$R/\sqrt{n}$")
    ax.set_title(r"(a) critical line $\lambda=\lambda_c$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ab = pd.read_csv(os.path.join(DATA, "day4_abssin_true.csv"))
    for frac, c in zip((0.5, 0.9, 1.1, 2.0), ("C3", "C1", "C0", "C2")):
        sub = ab[ab.lam_frac == frac]
        ax.plot(sub.n, sub.log10_E_true, "o-", lw=1.0, color=c,
                label=rf"$\lambda={frac}\lambda_c$")
    ax.set_xscale("log")
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$\log_{10} E_\infty(|\sin|)$")
    ax.set_title("(b) sup-error of $|\\sin x|$ across the threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_transition.png"), dpi=170)


def fig3():
    cr = pd.read_csv(os.path.join(DATA, "day6_crossover.csv"))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    rr = np.geomspace(1e-2, 3e5, 400)
    ax.semilogx(rr, G_pred(rr), "k-", lw=1.6,
                label=r"$G(\rho)=-1300\,H(\rho)$ (closed form)")
    lam = 2.0 * LAM_C
    for n, c in zip((256, 1024, 4096), ("C0", "C1", "C2")):
        sub = cr[(cr.lam_frac == 2.0) & (cr.n == n) & (cr.m >= 48)]
        ax.semilogx(sub.m**2 / n, sub.ratio, "o", ms=3.2, color=c,
                    label=rf"$n^2\mathrm{{dev}}/D_\lambda(m)$, $n={n}$")
    ax.set_xlabel(r"$\rho = m^2/n$")
    ax.set_ylabel(r"$n^2\,\mathrm{dev}_n(m)\,/\,D_\lambda(m)$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    # inset: pure-limit form with finite-k hooks
    axi = ax.inset_axes([0.09, 0.10, 0.36, 0.42])
    axi.semilogx(rr, G_pred(rr), "k-", lw=1.2)
    for n, c in zip((256, 1024, 4096), ("C0", "C1", "C2")):
        sub = cr[(cr.lam_frac == 2.0) & (cr.n == n) & (cr.m >= 48)]
        axi.semilogx(sub.m**2 / n, sub.ratio
                     * D_coef(lam, sub.m.to_numpy(dtype=float)) * (-1300.0),
                     "o", ms=2.0, color=c)
    axi.set_title(r"$-1300\,n^2\mathrm{dev}$ (pure limit)", fontsize=7)
    axi.tick_params(labelsize=6)
    axi.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_G.png"), dpi=170)


def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8))
    ax = axes[0]
    rec = pd.read_csv(os.path.join(DATA, "day6_recovery.csv"))
    for xi, c in zip(XIS, ("C0", "C1", "C2")):
        for frac, mk in ((1.1, "o-"), (2.0, "s--")):
            sub = rec[(rec.lam_frac == frac) & (rec.xi == xi) & (rec.n >= 256)]
            ax.semilogx(sub.n, sub.n2E_over_pred, mk, color=c, lw=1.0, ms=4,
                        label=rf"$\xi={xi}$, $\lambda={frac}\lambda_c$")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$n^2 E_\infty^{(1024)}\,/\,\|F_{\xi,\lambda}\|_\infty$")
    ax.set_title("(a) frozen approach to the predicted limiting profile")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax = axes[1]
    dfm = pd.read_csv(os.path.join(DATA, "day7_recovery_model.csv"))
    for xi, c in zip(XIS, ("C0", "C1", "C2")):
        sub = dfm[dfm.xi == xi]
        ax.loglog(sub.n, sub.deficit0, "o-", color=c, lw=1.0,
                  label=rf"$\xi={xi}$")
        ax.loglog(sub.n, sub.deficit0.iloc[-1]
                  * (sub.n / sub.n.iloc[-1])**(-xi / 2), "k:", lw=0.8)
    ax.set_xlabel("$n$")
    ax.set_ylabel("cusp deficit $F(0)-F_n(0)$")
    ax.set_title(r"(b) recovery law (model): guides $\propto n^{-\xi/2}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_recovery.png"), dpi=170)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("figures ->", OUT)
