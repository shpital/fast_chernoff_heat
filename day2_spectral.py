"""II.A2, day 2 (plan v2.4 section 4.2 + II.A1 revision): spectral route.

Pre-registered set (fixed BEFORE the run, no dataset extension until verdict):

  (a) fixed Fourier modes k in {1, 2, 4, 8}: smooth n^{-2} confirmation
      (E_freq = E_dom = 0 analytically);
  (b) moving mode k_n = z* (n/t)^{1/4}, z* = 7.5 (the amplification band
      tracks this frequency);
  (c) continuous amplification norm  A_n(t)   = g_2(t/n)^n = sup_k |M|^n;
      periodic amplification norm    A_n^Z(t) = max_{m in Z} |M(t/n, m)|^n;
  (d) smooth periodic datum sin(x): error = |M(t/n,1)^n - e^{-t}|;
      rough periodic datum |sin(x)| via exact Fourier series
        |sin x| = 2/pi - (4/pi) sum_j cos(2jx)/(4j^2-1):
      sup-norm error of the spectral composition against the exact
      semigroup reference (series evaluated to machine-negligible tails).

Prediction under test (from II.A1): pre-asymptotic amplification window
  n < n_amp(t) ~ t/tau_* ~ 200 t,   worst n ~ 200 t / e ~ 74 t,
with asymptotic second order restored beyond it.

t in {0.25, 1, 4} (theorem zone n >= t).  n = 4..4096 (powers of 2).
Outputs: data/day2_*.csv, figs/day2_spectral.png.
"""
import os

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day1_symbol import M_num  # noqa: E402  (validated float64 symbol, err <= 1.5e-13)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")

ZSTAR = 7.5
TS = (0.25, 1.0, 4.0)
NS = [2**p for p in range(2, 13)]                    # 4 .. 4096


# ---------------------------------------------------------------------------
# (a) fixed modes
# ---------------------------------------------------------------------------
def fixed_modes():
    print("== (a) fixed modes: n^2 * eps_{n,t}(k), reference law check ==")
    rows = []
    for t in TS:
        for k in (1.0, 2.0, 4.0, 8.0):
            pred = t**3 * np.exp(-t * k**2) * (13/105 * k**6 - k**8 / 7800)
            for n in NS:
                if n < t:
                    continue
                eps = float(M_num(t/n, np.array([k]))[0])**n - np.exp(-t*k**2)
                rows.append(dict(t=t, k=k, n=n, eps=eps, n2eps=n**2*eps,
                                 pred=pred))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day2_fixed_modes.csv"), index=False)
    for t in TS:
        sub = df[(df.t == t) & (df.n == 4096)]
        rat = sub.n2eps / sub.pred
        print(f"  t={t}: n=4096 ratio n2eps/pred by k: "
              + ", ".join(f"{r:.4f}" for r in rat))
    return df


# ---------------------------------------------------------------------------
# (b) moving mode + (c) amplification norms
# ---------------------------------------------------------------------------
def _log10_pow(vals_abs, n):
    """log10(|v|^n) without overflow."""
    with np.errstate(divide="ignore"):
        return n * np.log10(np.maximum(vals_abs, 1e-300))


def amplification():
    print("== (b,c) moving mode and amplification norms ==")
    rows = []
    for t in TS:
        for n in NS:
            if n < t:
                continue
            tau = t / n
            # moving mode
            kn = ZSTAR * (n / t)**0.25
            Mk = float(M_num(tau, np.array([kn]))[0])
            log10_eps_move = _log10_pow(np.array([abs(Mk)]), n)[0]
            # continuous sup over a dense k-grid through both scales
            kmax = 50.0 / tau**0.25
            kk = np.linspace(1e-6, kmax, 200001)
            vals = np.abs(M_num(tau, kk))
            g2 = vals.max()
            logA = _log10_pow(np.array([g2]), n)[0]
            # periodic: integer modes (datum period 2pi -> integer k;
            # |sin| uses even modes 2j, we track all integers for generality)
            mm = np.arange(1, int(kmax) + 1, dtype=float)
            valm = np.abs(M_num(tau, mm))
            iZ = valm.argmax()
            logAZ = _log10_pow(np.array([valm[iZ]]), n)[0]
            rows.append(dict(t=t, n=n, k_move=kn, M_move=Mk,
                             log10_eps_move=log10_eps_move,
                             g2=g2, log10_A=logA,
                             m_star=int(mm[iZ]), MZ=valm[iZ],
                             log10_AZ=logAZ,
                             log10_A_model=n*np.log10(max(200*t/n, 1.0))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day2_amplification.csv"), index=False)
    for t in TS:
        sub = df[df.t == t]
        i = sub.log10_A.idxmax()
        print(f"  t={t}: max log10 A_n = {sub.log10_A.max():.1f} at n={sub.n[i]}"
              f" (model {sub.log10_A_model[i]:.1f});"
              f" A_n^Z max log10 = {sub.log10_AZ.max():.1f};"
              f" first n with A_n=1: "
              f"{sub[sub.log10_A <= 1e-12].n.min()}")
    return df


# ---------------------------------------------------------------------------
# (d) periodic data: sin (smooth control) and |sin| (rough)
# ---------------------------------------------------------------------------
def periodic_data():
    print("== (d) periodic data: sin (smooth), |sin x| (rough, exact series) ==")
    xg = np.linspace(-np.pi, np.pi, 4097)
    rows = []
    for t in TS:
        for n in NS:
            if n < t:
                continue
            tau = t / n
            # smooth control: single mode k=1
            err_sin = abs(float(M_num(tau, np.array([1.0]))[0])**n
                          - np.exp(-t))
            # rough: |sin| even modes 2j, coefficients a_j = -4/(pi(4j^2-1))
            kmax = 50.0 / tau**0.25
            J = max(int(kmax / 2) + 50, 200)
            j = np.arange(1, J + 1, dtype=float)
            a = -4.0 / (np.pi * (4.0 * j**2 - 1.0))
            m = 2.0 * j
            Mm = M_num(tau, m)
            with np.errstate(over="ignore"):
                Mn = np.sign(Mm)**n * np.exp(
                    np.clip(n * np.log(np.maximum(np.abs(Mm), 1e-300)),
                            -700, 700))
            epsm = Mn - np.exp(-t * m**2)
            prof = (a[:, None] * np.cos(np.outer(m, xg))).T @ epsm \
                if False else np.cos(np.outer(xg, m)) @ (a * epsm)
            Einf = np.max(np.abs(prof))
            jd = int(j[np.abs(a * epsm).argmax()])
            rows.append(dict(t=t, n=n, err_sin=err_sin, Einf_abssin=Einf,
                             dom_mode=2 * jd,
                             band_mode=ZSTAR * (n / t)**0.25))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day2_periodic.csv"), index=False)
    for t in TS:
        sub = df[df.t == t].reset_index(drop=True)
        i = sub.Einf_abssin.idxmax()
        big = sub[sub.Einf_abssin > 1.0].n.tolist()
        print(f"  t={t}: |sin| max E_inf = {sub.Einf_abssin[i]:.3e} at "
              f"n={sub.n[i]} (dominant mode {sub.dom_mode[i]}, band mode "
              f"{sub.band_mode[i]:.1f}); E_inf>1 for n in {big}")
        tail = sub[sub.n >= 1024]
        if len(tail) >= 2:
            sl = np.polyfit(np.log(tail.n), np.log(tail.Einf_abssin), 1)[0]
            print(f"        post-barrier slope (n>=1024): {sl:.3f}")
    return df


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def figure(dfA, dfP):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    colors = {0.25: "tab:green", 1.0: "tab:blue", 4.0: "tab:red"}
    for t in TS:
        c = colors[t]
        A = dfA[dfA.t == t]
        axes[0].plot(A.n, A.log10_A, "o-", color=c, ms=3, label=f"t={t}")
        axes[0].plot(A.n, A.log10_A_model, "--", color=c, lw=0.8)
        axes[0].axvline(200 * t, color=c, ls=":", lw=0.8)
        P = dfP[dfP.t == t]
        axes[1].loglog(P.n, P.Einf_abssin, "o-", color=c, ms=3, label=f"t={t}")
        axes[1].axvline(200 * t, color=c, ls=":", lw=0.8)
        axes[2].loglog(P.n, P.err_sin, "o-", color=c, ms=3, label=f"t={t}")
    ref = np.array([64, 4096], dtype=float)
    axes[1].loglog(ref, 10.0 * ref**-2, "k--", lw=0.8, label=r"$n^{-2}$")
    axes[2].loglog(ref, 0.05 * ref**-2, "k--", lw=0.8, label=r"$n^{-2}$")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("n"); axes[0].set_ylabel(r"$\log_{10} A_n(t)$")
    axes[0].set_title(r"amplification norm $g_2(t/n)^n$ (dashed: $(200t/n)^n$)")
    axes[1].set_xlabel("n"); axes[1].set_ylabel(r"$E_\infty$")
    axes[1].set_title(r"$|\sin x|$: sup error (dotted: $n=200t$)")
    axes[2].set_xlabel("n"); axes[2].set_ylabel("error")
    axes[2].set_title(r"$\sin x$: smooth control")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("II.A2: pre-asymptotic spectral price of RV-2026 on rough periodic data")
    fig.tight_layout()
    out = os.path.join(FIGS, "day2_spectral.png")
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


if __name__ == "__main__":
    fixed_modes()
    print()
    dfA = amplification()
    print()
    dfP = periodic_data()
    print()
    figure(dfA, dfP)
