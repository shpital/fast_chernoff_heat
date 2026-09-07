"""II.A1, day 1 (plan v2.4, sections 4.1-4.2.1): RV-2026 Fourier symbol sanity.

Constant heat: a = 1, b = c = 0.  Locked object (plan 4.1): the reduced
even coefficients of the polynomial correction B(y) = sum_j beta_j(tau) y^j,

    beta_0 =  14553/64  * tau,
    beta_2 = -280665/64 * tau,
    beta_4 =  800415/64 * tau,
    beta_6 = -567567/64 * tau,      beta_1 = beta_3 = beta_5 = 0.

Pipeline (pre-registered):
    B_tau(y) -> moments int B y^{2j} dy -> M_tau(k) -> M_tau(k) - e^{-tau k^2}
             -> M_{t/n}(k)^n - e^{-t k^2}.

Unit tests (exact, plan 4.1 / 4.2.1):
    int B dy = 0,  int B y^2 dy = 0,  int B y^4 dy = (24/5) tau,
    int B y^6 dy = 0,  int B y^8 dy = -(336/65) tau;
    M_tau(k) = 1 - tau k^2 + (1/2) tau^2 k^4 + O(tau^3);
    M_tau(k) - e^{-tau k^2} = tau^3 (13/105 k^6 - k^8/7800) + O(tau^{7/2}).

Also logged: g_2(tau) = sup_k |M_tau(k)|, g_inf(tau) = 1 + int |B| dy,
float64 roundoff floor of the beta-cancellation (plan 4.1 safety note),
and the n^{-2} composition law at fixed k.

Outputs: data/day1_*.csv, figs/day1_symbol.png; console log = unit test log.
"""
import os

import numpy as np
import pandas as pd
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

# ---------------------------------------------------------------------------
# locked coefficients (exact rationals)
# ---------------------------------------------------------------------------
tau_s, k_s, y_s, z_s = sp.symbols("tau k y z", positive=True)

BETA = {
    0: sp.Rational(14553, 64),
    2: sp.Rational(-280665, 64),
    4: sp.Rational(800415, 64),
    6: sp.Rational(-567567, 64),
}
B_poly = sum(c * y_s**j for j, c in BETA.items()) * tau_s  # B_tau(y)

EXPECTED_MOMENTS = {
    0: sp.Integer(0),
    2: sp.Integer(0),
    4: sp.Rational(24, 5) * tau_s,
    6: sp.Integer(0),
    8: sp.Rational(-336, 65) * tau_s,
}


def check_moments():
    print("== moment identities (exact, sympy) ==")
    ok = True
    for p, want in EXPECTED_MOMENTS.items():
        got = sp.integrate(B_poly * y_s**p, (y_s, -1, 1))
        match = sp.simplify(got - want) == 0
        ok &= match
        print(f"  int B y^{p} dy = {got}   expected {want}   "
              f"{'PASS' if match else 'FAIL'}")
    assert ok, "moment identities FAILED -- fix the code, not the theory"
    return ok


# ---------------------------------------------------------------------------
# symbol: closed forms
# ---------------------------------------------------------------------------
# m_j(z) = 2 * int_0^1 y^j cos(z y) dy, closed forms generated symbolically
def m_closed(j):
    return sp.simplify(2 * sp.integrate(y_s**j * sp.cos(z_s * y_s), (y_s, 0, 1)))


M_CLOSED = {j: m_closed(j) for j in (0, 2, 4, 6)}


def symbol_expr():
    """M_tau(k), fully symbolic."""
    first = sp.sin(sp.sqrt(6 * tau_s) * k_s) / (sp.sqrt(6 * tau_s) * k_s)
    second = sum(BETA[j] * tau_s * M_CLOSED[j].subs(z_s, k_s * tau_s**sp.Rational(1, 4))
                 for j in (0, 2, 4, 6))
    return first + second


def check_expansion():
    """Series of M_tau(k) in tau at fixed k: tangency and leading defect."""
    print("== symbol expansion in tau (fixed k, sympy series) ==")
    M = symbol_expr()
    # expand in s = tau^{1/4} to keep the algebra polynomial
    s = sp.symbols("s", positive=True)
    Ms = M.subs(tau_s, s**4)
    ser = sp.series(Ms, s, 0, 15).removeO()          # up to s^14 = tau^{7/2}
    ser = sp.expand(ser)
    got = {p: sp.nsimplify(ser.coeff(s, 4 * p)) for p in range(4)}
    print(f"  coeff tau^0 : {got[0]}   expected 1")
    print(f"  coeff tau^1 : {got[1]}   expected -k^2")
    print(f"  coeff tau^2 : {got[2]}   expected k^4/2")
    print(f"  coeff tau^3 : {got[3]}")
    assert sp.simplify(got[0] - 1) == 0
    assert sp.simplify(got[1] + k_s**2) == 0
    assert sp.simplify(got[2] - k_s**4 / 2) == 0
    # fractional powers between integer orders must vanish
    for pw in range(1, 14):
        if pw % 4 == 0:
            continue
        c = ser.coeff(s, pw)
        assert sp.simplify(c) == 0, f"nonzero coeff at tau^{pw/4}: {c}"
    print("  fractional-order coefficients tau^{1/4..13/4} all zero: PASS")
    # leading defect vs e^{-tau k^2}
    heat = sp.exp(-s**4 * k_s**2)
    defect = sp.expand(ser - sp.series(heat, s, 0, 15).removeO())
    d3 = sp.simplify(defect.coeff(s, 12))            # tau^3 coefficient
    want = sp.Rational(13, 105) * k_s**6 - k_s**8 / 7800
    match = sp.simplify(d3 - want) == 0
    print(f"  defect tau^3 coeff: {sp.nsimplify(d3)}")
    print(f"  expected          : 13k^6/105 - k^8/7800   "
          f"{'PASS' if match else 'FAIL'}")
    assert match
    return True


# ---------------------------------------------------------------------------
# numeric symbol (float64) + roundoff floor
# ---------------------------------------------------------------------------
BETA_F = {j: float(c) for j, c in BETA.items()}
M_FUNCS = {j: sp.lambdify(z_s, M_CLOSED[j], "numpy") for j in (0, 2, 4, 6)}

# The closed forms carry z^{-(j+1)} prefactors with alternating O(720) terms:
# catastrophic cancellation for z <~ 1 (measured: garbage already at z ~ 0.01
# with a narrow series window).  Taylor series in even powers of z:
#   m_j(z) = 2 sum_{p>=0} (-1)^p z^{2p} / [ (2p)! (2p+j+1) ],
# entire function; with terms up to z^70 the truncation error at z = 4 is
# far below 1e-16, so we switch: series for |z| < 4, closed form beyond.
Z_SWITCH = 4.0
_PMAX = 36


def _series_coeffs(j):
    cs = []
    for p in range(_PMAX):
        cs.append(2.0 * (-1.0) ** p /
                  (float(sp.factorial(2 * p)) * (2 * p + j + 1)))
    return np.array(cs)


M_SERIES_C = {j: _series_coeffs(j) for j in (0, 2, 4, 6)}


def m_num(j, z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = np.abs(z) < Z_SWITCH
    if small.any():
        z2 = z[small] ** 2
        acc = np.zeros_like(z2)
        for c in M_SERIES_C[j][::-1]:                # Horner in z^2
            acc = acc * z2 + c
        out[small] = acc
    if (~small).any():
        out[~small] = np.asarray(M_FUNCS[j](z[~small]), dtype=float)
    return out


def M_num(tau, k):
    """Float64 production symbol."""
    tau = float(tau)
    k = np.asarray(k, dtype=float)
    u = np.sqrt(6.0 * tau) * k
    first = np.where(np.abs(u) < 1e-8, 1.0 - u**2 / 6.0, np.sin(u) / np.where(np.abs(u) < 1e-8, 1.0, u))
    z = k * tau**0.25
    second = sum(BETA_F[j] * tau * m_num(j, z) for j in (0, 2, 4, 6))
    return first + second


def M_mp(tau, k, dps=50):
    """High-precision reference via mpmath quadrature of the definition."""
    import mpmath as mp
    mp.mp.dps = dps
    tau = mp.mpf(tau)
    k = mp.mpf(k)
    first = mp.sin(mp.sqrt(6 * tau) * k) / (mp.sqrt(6 * tau) * k)
    q = tau**mp.mpf("0.25")
    B = lambda y: tau * (mp.mpf(14553) / 64 - mp.mpf(280665) / 64 * y**2
                         + mp.mpf(800415) / 64 * y**4 - mp.mpf(567567) / 64 * y**6)
    second = mp.quad(lambda y: B(y) * mp.cos(k * q * y), [-1, 0, 1])
    return first + second


def check_float64_floor():
    print("== float64 roundoff floor of the beta cancellation ==")
    rows = []
    for tau in (1e-1, 1e-2, 1e-3, 1e-4):
        for k in (0.5, 1.0, 2.0, 4.0, 8.0):
            ref = float(M_mp(tau, k))
            got = float(M_num(tau, np.array([k]))[0])
            rows.append(dict(tau=tau, k=k, M=ref, err=abs(got - ref)))
    df = pd.DataFrame(rows)
    print(f"  max |M_float64 - M_mp50| = {df.err.max():.3e} "
          f"(plan safety estimate ~1e4*eps ~ 2e-12 per step)")
    df.to_csv(os.path.join(DATA, "day1_float64_floor.csv"), index=False)
    return df.err.max()


# ---------------------------------------------------------------------------
# stability metrics
# ---------------------------------------------------------------------------
def exact_TV(tau):
    """Exact one-step TV norm ||mu_tau||_TV: the two density components share
    the shift space, so cancellation is possible and 1 + int|B| is only a
    triangle upper bound (g_inf_bar).  Density:
        mu(s) = 1_{|s|<=sqrt(6 tau)} / (2 sqrt(6 tau))
              + tau^{3/4} P(s / tau^{1/4}) 1_{|s|<=tau^{1/4}},
    P(y) = (14553 - 280665 y^2 + 800415 y^4 - 567567 y^6)/64."""
    su = np.sqrt(6.0 * tau)
    sb = tau**0.25

    def P(y):
        y2 = y * y
        return (14553.0 + y2 * (-280665.0 + y2 * (800415.0 - 567567.0 * y2))) / 64.0

    def dens(s):
        u = np.where(np.abs(s) <= su, 1.0 / (2.0 * su), 0.0)
        b = np.where(np.abs(s) <= sb, tau**0.75 * P(s / sb), 0.0)
        return u + b

    # integrate |mu| on segments split at all kinks/support edges
    edges = np.unique(np.concatenate([[-max(su, sb), max(su, sb)],
                                      [-su, su, -sb, sb]]))
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        ss = np.linspace(a, b, 20001)
        total += np.trapezoid(np.abs(dens(ss)), ss)
    return total


def stability(taus):
    print("== stability: g_2 = sup_k |M|, g_inf_bar = 1 + int|B| (TV upper "
          "bound), g_inf_exact = ||mu_tau||_TV ==")
    rows = []
    absB = sp.lambdify(y_s, sp.Abs(B_poly / tau_s), "numpy")
    yy = np.linspace(-1, 1, 200001)
    intabs = np.trapezoid(np.abs(absB(yy)), yy)      # per unit tau
    for tau in taus:
        # sup over k: scan through both scales k*tau^{1/4} ~ O(1..) and beyond
        kmax = 50.0 / tau**0.25
        kk = np.linspace(1e-6, kmax, 400001)
        vals = np.abs(M_num(tau, kk))
        g2 = vals.max()
        k2 = kk[vals.argmax()]
        ginf_bar = 1.0 + intabs * tau                # triangle/TV upper bound
        ginf_ex = exact_TV(tau)                      # exact one-step TV
        assert g2 <= ginf_ex * (1 + 1e-6), \
            f"sanity violated: g2={g2} > exact TV={ginf_ex} (numerical garbage)"
        rows.append(dict(tau=tau, g2=g2, k_at_g2=k2,
                         ginf_bar=ginf_bar, ginf_exact=ginf_ex))
        print(f"  tau={tau:8.1e}: g2={g2:.6f} (argmax k={k2:8.3f}), "
              f"g_inf_bar={ginf_bar:.4f}, g_inf_exact={ginf_ex:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day1_stability.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# composition: n^{-2} law at fixed k
# ---------------------------------------------------------------------------
def composition(t=1.0, ks=(0.5, 1.0, 2.0, 4.0), ns=(4, 8, 16, 32, 64, 128, 256)):
    """Law check in high precision (mpmath, separates the analytic law from
    the float64 floor), plus the float64 production value for comparison."""
    import mpmath as mp
    print(f"== composition M_(t/n)^n - e^(-t k^2), t={t} ==")
    print("   predicted: n^2*eps -> t^3 e^(-t k^2) (13k^6/105 - k^8/7800)")
    rows = []
    for k in ks:
        pred = t**3 * np.exp(-t * k**2) * (13.0 / 105.0 * k**6 - k**8 / 7800.0)
        for n in ns:
            tau = t / n
            eps_mp = float(M_mp(tau, k) ** n - mp.exp(-t * mp.mpf(k) ** 2))
            eps_f64 = float(M_num(tau, np.array([k]))[0]) ** n - np.exp(-t * k**2)
            rows.append(dict(k=k, n=n, eps_mp=eps_mp, eps_f64=eps_f64,
                             n2eps=n**2 * eps_mp, pred=pred,
                             ratio=n**2 * eps_mp / pred if pred != 0 else np.nan,
                             f64_noise=abs(eps_f64 - eps_mp)))
        r = rows[-1]
        print(f"  k={k}: n^2*eps(n={ns[-1]}) = {r['n2eps']:.6e}, "
              f"predicted {r['pred']:.6e}, ratio {r['ratio']:.4f}, "
              f"f64 noise {r['f64_noise']:.1e}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day1_composition.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# high-frequency structure (figure)
# ---------------------------------------------------------------------------
def figure(taus=(0.25, 0.0625, 0.015625)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for tau in taus:
        kk = np.linspace(1e-4, 40.0 / tau**0.25, 20000)
        M = M_num(tau, kk)
        axes[0].plot(kk * np.sqrt(tau), M, lw=0.9, label=f"tau={tau}")
        axes[1].plot(kk * tau**0.25, M, lw=0.9, label=f"tau={tau}")
        axes[2].semilogy(kk * tau**0.25, np.maximum(np.abs(M), 1e-18),
                         lw=0.9, label=f"tau={tau}")
    axes[0].set_xlabel(r"$k\sqrt{\tau}$"); axes[0].set_ylabel(r"$M_\tau(k)$")
    axes[0].set_title("shift-family scale"); axes[0].set_xlim(0, 12)
    axes[1].set_xlabel(r"$k\tau^{1/4}$")
    axes[1].set_title("correction-kernel scale"); axes[1].set_xlim(0, 40)
    axes[2].set_xlabel(r"$k\tau^{1/4}$"); axes[2].set_ylabel(r"$|M_\tau(k)|$")
    axes[2].set_title("high-frequency decay (log)")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("RV-2026 constant-heat symbol $M_\\tau(k)$ (II.A1)")
    fig.tight_layout()
    out = os.path.join(FIGS, "day1_symbol.png")
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


if __name__ == "__main__":
    check_moments()
    print()
    for j in (0, 2, 4, 6):
        print(f"m_{j}(z) = {M_CLOSED[j]}")
    print()
    check_expansion()
    print()
    check_float64_floor()
    print()
    stability([0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001])
    print()
    composition()
    print()
    figure()
