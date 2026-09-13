"""Study III, day 1: the constant-heat design set.

Four one-step kernels for u_t = u_xx, all written as proper Riemann
integrals of f over bounded intervals (same structural cost class):

  RV6   Remizov-Vedenin 2026: uniform on [-sqrt(6 tau), sqrt(6 tau)] plus the
        signed degree-6 correction tau^{3/4} P6(y/tau^{1/4}) (Study II).
  RV8   same ansatz, unique even degree-8 polynomial with
        M0 = M2 = M6 = M8 = 0, M4 = 24/5  (kills the k^8 tau^3 defect term,
        hence the leading boundary layer H(0) = 6 M8/8! = 0).
  MIX2  positive mixture  1/2 U[-a s, a s] + 1/2 U[-b s, b s],  s = sqrt(tau),
        a^2, b^2 = 6 -+ 2 sqrt 6   (moments 2, 4 of the Gaussian; order 2).
  GM2   Gauss-Maxwell q = 2: the Gaussian is a scale mixture of uniforms
        (Khintchine), N(0, 2 tau) = int U[-r, r] dnu(r), r^2 ~ Gamma(3/2, 4 tau).
        Two-point Gauss quadrature of nu in v = r^2:
        v = 10 -+ 2 sqrt 10, weights 1/2 +- sqrt(10)/10   (moments 2, 4, 6 of
        the Gaussian; order 3, positive, two integrals).

For each kernel (pre-registered list; Study-II conventions -- Fourier symbol,
even n, composite Simpson on a log grid in tau, |sin x|^xi mode truncation --
but with the cancellation-free deviation quadrature of `deviations()`, which
resolves n^{-3} deviations at n = 4096 where the Study-II log-domain route
hits the float64 floor; the RV6 numbers reproduce Study II):

  1. tangency order and leading defect coefficient (exact, sympy);
  2. lambda_c = sup_{tau,k} log|M_tau(k)|/tau  (0 for probability kernels);
  3. TV = sup_tau ||C(tau)||  (1 + tau int|P| for RV-type; 1 for positive);
  4. fixed-k limit D_lambda(k) of n^p [R_{lambda,n}(k) - 1/(lambda+k^2)],
     analytic vs measured at n = 4096;
  5. boundary layer at k^2 = rho n: n^2 dev_n(k_n) vs H(rho);
  6. rough-data resolvent error for f_xi = |sin x|^xi, xi in {1/4, 1/2, 1},
     modes m = 2j <= 1024, n in {4, ..., 4096}: n^p E_inf(n), the cusp value,
     the limit profile ||F||_inf and the plateau ratio;
  7. local-CLT check for the positive kernels: ||mu_{t/n}^{*n} - g_t||_{L^1}
     at t = 1 vs n (expected n^{-2} for MIX2, n^{-3} for GM2), which bounds
     the sup-error on *all* bounded data;
  8. the one-parameter family P_s = (1-s) P6 + s P8 (M8 scaled by 1-s):
     lambda_c(s), TV(s), H_s(0) = -(1-s)/1300.

Outputs: data/day1_*.csv, figs/day1_design_set.png; console = record.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import simpson
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

# Study II scripts: ../study_II in the working layout, the repository root when
# this file lives in fast_chernoff_heat/addendum/.
for _cand in (os.path.join(os.path.dirname(HERE), "study_II"), os.path.dirname(HERE)):
    if os.path.exists(os.path.join(_cand, "day6_xi.py")):
        STUDY_II = _cand
        break
else:
    raise ImportError("day6_xi.py of Study II not found next to this script")
sys.path.insert(0, STUDY_II)
from day6_xi import a_coeffs  # noqa: E402

LAM_C_RV6 = 73.987479          # Study II, numerically localized
NS = [4, 16, 64, 256, 1024, 4096]
XIS = (0.25, 0.5, 1.0)
J0 = 512
XGRID = np.linspace(0.0, np.pi / 2, 4001)      # f_xi even and pi-periodic
LOG_CLIP = -700.0

# ---------------------------------------------------------------------------
# polynomial corrections (exact rationals)
# ---------------------------------------------------------------------------
y_s = sp.symbols("y")
P6_C = {0: sp.Rational(14553, 64), 2: sp.Rational(-280665, 64),
        4: sp.Rational(800415, 64), 6: sp.Rational(-567567, 64)}


def solve_P8():
    c = sp.symbols("c0 c2 c4 c6 c8")
    P = sum(ci * y_s ** (2 * i) for i, ci in enumerate(c))
    mom = lambda m: sp.integrate(P * y_s ** m, (y_s, -1, 1))
    sol = sp.solve([mom(0), mom(2), mom(4) - sp.Rational(24, 5), mom(6),
                    mom(8)], c)
    return {2 * i: sol[ci] for i, ci in enumerate(c)}


P8_C = solve_P8()


def poly_moments(C, pmax=80):
    """M_{2p} = int_{-1}^1 P y^{2p} dy = 2 sum_j c_j/(2p+j+1)."""
    return {p: 2.0 * sum(float(cj) / (2 * p + j + 1) for j, cj in C.items())
            for p in range(pmax + 1)}


def poly_TV(C):
    P = sum(cj * y_s ** j for j, cj in C.items())
    roots = [complex(r) for r in sp.Poly(P, y_s).nroots(n=30)]
    pts = sorted({-1.0, 1.0} | {r.real for r in roots
                                 if abs(r.imag) < 1e-12 and abs(r.real) < 1})
    Pf = sp.lambdify(y_s, P, "numpy")
    tv = 0.0
    for lo, hi in zip(pts[:-1], pts[1:]):
        xs, ws = np.polynomial.legendre.leggauss(40)
        xm = 0.5 * (hi - lo) * xs + 0.5 * (hi + lo)
        tv += 0.5 * (hi - lo) * np.sum(ws * np.abs(Pf(xm)))
    return tv


# ---------------------------------------------------------------------------
# m_j(z) = 2 int_0^1 y^j cos(zy) dy, j even <= 8: series (|z|<4), recursion
# ---------------------------------------------------------------------------
_PMAX = 36
Z_SWITCH = 4.0
_SER = {j: np.array([2.0 * (-1.0) ** p / (float(sp.factorial(2 * p))
                                          * (2 * p + j + 1))
                     for p in range(_PMAX)]) for j in range(0, 9, 2)}


def m_all(z, jmax=8):
    """dict j -> m_j(z) for even j <= jmax, vectorized."""
    z = np.asarray(z, dtype=float)
    out = {}
    small = np.abs(z) < Z_SWITCH
    big = ~small
    for j in range(0, jmax + 1, 2):
        out[j] = np.empty_like(z)
    if small.any():
        z2 = z[small] ** 2
        for j in range(0, jmax + 1, 2):
            acc = np.zeros_like(z2)
            for c in _SER[j][::-1]:
                acc = acc * z2 + c
            out[j][small] = acc
    if big.any():
        zb = z[big]
        s, c = np.sin(zb), np.cos(zb)
        # I_j = int_0^1 y^j cos(zy) dy;  I_0 = s/z,
        # I_j = s/z + j c/z^2 - j(j-1) I_{j-2}/z^2   (j >= 2, by parts twice)
        I_prev2 = s / zb
        out[0][big] = 2.0 * I_prev2
        for j in range(2, jmax + 1, 2):
            I_j = s / zb + j * c / zb ** 2 - j * (j - 1) * I_prev2 / zb ** 2
            out[j][big] = 2.0 * I_j
            I_prev2 = I_j
    return out


def sinc(u):
    small = np.abs(u) < 1e-8
    return np.where(small, 1.0 - u ** 2 / 6.0,
                    np.sin(u) / np.where(small, 1.0, u))


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
A2, B2 = 6.0 - 2.0 * np.sqrt(6.0), 6.0 + 2.0 * np.sqrt(6.0)
V1, V2 = 10.0 - 2.0 * np.sqrt(10.0), 10.0 + 2.0 * np.sqrt(10.0)
W1, W2 = 0.5 + np.sqrt(10.0) / 10.0, 0.5 - np.sqrt(10.0) / 10.0

KERNELS = {
    "RV6": dict(name="RV6", kind="poly", C={j: float(c) for j, c in P6_C.items()},
                C_exact=P6_C, p=2),
    "RV8": dict(name="RV8", kind="poly", C={j: float(c) for j, c in P8_C.items()},
                C_exact=P8_C, p=2),
    "MIX2": dict(name="MIX2", kind="mix", w=(0.5, 0.5), v=(A2, B2), p=2),
    "GM2": dict(name="GM2", kind="mix", w=(W1, W2), v=(V1, V2), p=3),
}


def moment_series(C, pmax=36):
    """a_p = (-1)^p M_{2p}/(2p)!  with the exact moments M_{2p} = 2 sum_j
    c_j/(2p+j+1), so that Phat(z) = sum_p a_p z^{2p}: cancellation-free
    (the four/five huge c_j cancel symbolically inside M_0 = M_2 = 0)."""
    out = []
    for p in range(pmax + 1):
        M2p = 2 * sum(sp.nsimplify(cj) / (2 * p + j + 1) for j, cj in C.items())
        out.append(float((-1) ** p * M2p / sp.factorial(2 * p)))
    return np.array(out)


def Phat_poly(C, z, series=None):
    """int_{-1}^1 P(y) cos(zy) dy: moment series for |z| < 4, closed form beyond."""
    z = np.asarray(z, dtype=float)
    if series is None:
        series = moment_series(C)
    out = np.empty_like(z)
    small = np.abs(z) < Z_SWITCH
    if small.any():
        z2 = z[small] ** 2
        acc = np.zeros_like(z2)
        for a in series[::-1]:
            acc = acc * z2 + a
        out[small] = acc
    if (~small).any():
        mj = m_all(z[~small])
        out[~small] = sum(float(c) * mj[j] for j, c in C.items())
    return out


def symbol(kern, tau, k):
    """M_tau(k) broadcast over arrays tau, k."""
    tau = np.asarray(tau, dtype=float)
    k = np.asarray(k, dtype=float)
    if kern["kind"] == "mix":
        s = np.sqrt(tau) * k
        return sum(w * sinc(np.sqrt(v) * s) for w, v in zip(kern["w"], kern["v"]))
    if "mom_series" not in kern:
        kern["mom_series"] = moment_series(kern["C_exact"])
    first = sinc(np.sqrt(6.0 * tau) * k)
    z = k * tau ** 0.25
    zb = np.broadcast_to(z, np.broadcast(tau, k).shape)
    second = tau * Phat_poly(kern["C"], zb, kern["mom_series"])
    return first + second


# ---------------------------------------------------------------------------
# 1. tangency (sympy)
# ---------------------------------------------------------------------------
def _sinc_series(u2, pmax):
    """sin(u)/u = sum (-1)^p u^{2p}/(2p+1)!  as a polynomial in u2 = u^2."""
    return sum((-1) ** p * u2 ** p / sp.factorial(2 * p + 1)
               for p in range(pmax + 1))


def tangency():
    """Exact defect M_tau(k) - e^{-tau k^2} as a polynomial in s = tau^{1/4}
    (RV-type) or y^2 = tau k^2 (positive mixtures), truncated consistently."""
    print("== 1. tangency orders (exact, sympy) ==")
    s, k = sp.symbols("s k", positive=True)          # s = tau^{1/4}
    tau = s ** 4
    rows = []
    for name, kern in KERNELS.items():
        if kern["kind"] == "mix":
            if name == "MIX2":
                W = (sp.Rational(1, 2), sp.Rational(1, 2))
                V = (6 - 2 * sp.sqrt(6), 6 + 2 * sp.sqrt(6))
            else:
                W = (sp.Rational(1, 2) + sp.sqrt(10) / 10,
                     sp.Rational(1, 2) - sp.sqrt(10) / 10)
                V = (10 - 2 * sp.sqrt(10), 10 + 2 * sp.sqrt(10))
            M = sum(w * _sinc_series(v * tau * k ** 2, 6) for w, v in zip(W, V))
        else:
            C = P6_C if name == "RV6" else P8_C
            z2 = k ** 2 * s ** 2                       # z^2 = k^2 tau^{1/2}
            mj = {j: sum(2 * (-1) ** p * z2 ** p
                         / (sp.factorial(2 * p) * (2 * p + j + 1))
                         for p in range(8)) for j in C}
            M = (_sinc_series(6 * tau * k ** 2, 6)
                 + tau * sum(c * mj[j] for j, c in C.items()))
        E = sum((-tau * k ** 2) ** p / sp.factorial(p) for p in range(7))
        diff = sp.expand(sp.radsimp(sp.expand(M - E)))
        poly = sp.Poly(diff, s)
        terms = {}
        for (deg,), coef in zip(poly.monoms(), poly.coeffs()):
            if deg <= 16:                              # up to tau^4
                terms[sp.Rational(deg, 4)] = sp.simplify(coef)
        terms = {q: c for q, c in sorted(terms.items()) if c != 0}
        lead_q = min(terms)
        show = "  +  ".join(f"[{sp.factor(c)}] tau^{q}" for q, c in terms.items()
                            if q <= lead_q + 1)
        print(f"  {name}: order p = {lead_q - 1};  M - e^(-tau k^2) = {show} + ...")
        rows.append(dict(kernel=name, order=float(lead_q - 1),
                         terms=str({str(q): str(c) for q, c in terms.items()})))
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "day1_tangency.csv"),
                              index=False)


# ---------------------------------------------------------------------------
# 2. lambda_c and 3. TV
# ---------------------------------------------------------------------------
def lambda_c(kern, taus=None, verbose=True):
    if taus is None:
        taus = np.geomspace(1e-4, 2.0, 300)
    best = (0.0, None, None)
    supM = 0.0
    for tau in taus:
        kk = np.linspace(1e-6, 50.0 / tau ** 0.25, 100001)
        vals = np.abs(symbol(kern, tau, kk))
        g = vals.max()
        supM = max(supM, g)
        lam = np.log(g) / tau if g > 1 else 0.0
        if lam > best[0]:
            best = (lam, tau, kk[vals.argmax()])
    if best[1] is not None:
        # local refinement around (tau*, k*)
        t0, k0 = best[1], best[2]
        for it in range(3):
            tt = np.linspace(t0 * 0.9, t0 * 1.1, 401)
            kk = np.linspace(k0 * 0.95, k0 * 1.05, 401)
            vals = np.log(np.abs(symbol(kern, tt[:, None], kk[None, :]))) \
                / tt[:, None]
            i, j = np.unravel_index(np.argmax(vals), vals.shape)
            best = (vals[i, j], tt[i], kk[j])
            t0, k0 = tt[i], kk[j]
    return best, supM


def thresholds():
    print("== 2-3. lambda_c and TV ==")
    rows = []
    for name, kern in KERNELS.items():
        t0 = time.time()
        (lc, ts, ks), supM = lambda_c(kern)
        if kern["kind"] == "poly":
            tv = poly_TV(P6_C if name == "RV6" else P8_C)
            tvs = f"1 + {tv:.4f} tau"
        else:
            tv = 0.0
            tvs = "1 (probability kernel)"
        print(f"  {name}: sup|M| = {supM:.6f}, lambda_c = {lc:.4f}"
              f"{f' at tau* = {ts:.6f}, k* = {ks:.4f}' if ts else ''};"
              f"  ||C(tau)|| = {tvs}   [{time.time()-t0:.0f}s]")
        rows.append(dict(kernel=name, sup_abs_M=supM, lambda_c=lc,
                         tau_star=ts, k_star=ks, TV_slope=tv))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day1_thresholds.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# resolvent deviation quadrature (vectorized over modes, cancellation-free)
#
#   dev_n(k) = R_{lam,n}(k) - 1/(lam+k^2)
#            = n int_0^inf e^{-n tau (lam+k^2)} [ e^{n g} - 1 ] d tau,
#   g(tau,k) = log|M_tau(k)| + tau k^2    (even n => sign-free).
#
# g is evaluated from the exact defect series where the direct formula loses
# digits (small tau k^2 and small k tau^{1/4}); the integrand uses expm1 when
# |n g| < 1.  This resolves dev ~ n^{-3} (order-3 kernels) at n = 4096, which
# the log-domain route of Study II cannot (float64 floor ~1e-15 on R).
# ---------------------------------------------------------------------------
def _build_defect_poly(C, pmax=14):
    """Exact defect M - e^{-tau k^2} for the RV-type kernel with correction
    polynomial C, as {degree d: k-poly coeffs} in s = tau^{1/4}."""
    s, k = sp.symbols("s k", positive=True)
    tau = s ** 4
    z2 = k ** 2 * s ** 2
    mj = {j: sum(2 * (-1) ** p * z2 ** p / (sp.factorial(2 * p) * (2 * p + j + 1))
                 for p in range(pmax + 1)) for j in C}
    M = (_sinc_series(6 * tau * k ** 2, pmax)
         + tau * sum(c * mj[j] for j, c in C.items()))
    E = sum((-tau * k ** 2) ** p / sp.factorial(p) for p in range(pmax + 1))
    poly = sp.Poly(sp.expand(M - E), s)
    table = {}
    for (d,), coef in zip(poly.monoms(), poly.coeffs()):
        kp = sp.Poly(sp.expand(coef), k)
        table[d] = np.array([float(c) for c in kp.all_coeffs()])   # high->low
    return table


def _build_mix_gseries(W, V, pmax=16):
    """log F(y) + y^2 = sum_p g_p y^{2p} for F = sum w sinc(sqrt v y), by the
    power-series logarithm recurrence in 60-digit arithmetic (x = y^2):
        F = 1 + sum f_p x^p,  l_p = f_p - (1/p) sum_{m<p} m l_m f_{p-m}."""
    import mpmath as mp
    mp.mp.dps = 60
    Wm = [mp.mpf(sp.N(w, 70).__str__()) for w in W]
    Vm = [mp.mpf(sp.N(v, 70).__str__()) for v in V]
    f = [sum(w * (-1) ** p * v ** p / mp.factorial(2 * p + 1)
             for w, v in zip(Wm, Vm)) for p in range(pmax + 1)]
    assert abs(f[0] - 1) < mp.mpf(10) ** -50
    l = [mp.mpf(0)] * (pmax + 1)
    for p in range(1, pmax + 1):
        l[p] = f[p] - sum(m * l[m] * f[p - m] for m in range(1, p)) / p
    coeffs = np.array([float(l[p]) for p in range(pmax + 1)])
    coeffs[1] += 1.0                                 # + y^2
    return coeffs                                    # index p -> g_p


_MIX_EXACT = {
    "MIX2": ((sp.Rational(1, 2), sp.Rational(1, 2)),
             (6 - 2 * sp.sqrt(6), 6 + 2 * sp.sqrt(6))),
    "GM2": ((sp.Rational(1, 2) + sp.sqrt(10) / 10, sp.Rational(1, 2) - sp.sqrt(10) / 10),
            (10 - 2 * sp.sqrt(10), 10 + 2 * sp.sqrt(10))),
}
_SERIES_CACHE = {}


def _series_for(kern):
    key = id(kern)
    if key not in _SERIES_CACHE:
        if kern["kind"] == "mix":
            W, V = _MIX_EXACT[kern["name"]]
            _SERIES_CACHE[key] = _build_mix_gseries(W, V)
        else:
            _SERIES_CACHE[key] = _build_defect_poly(kern["C_exact"])
    return _SERIES_CACHE[key]


Y_SWITCH = 0.3          # mix: series for tau k^2 < Y_SWITCH^2
Z_SW, TK_SW = 1.5, 0.3   # poly: defect series for k tau^{1/4} < Z_SW, tau k^2 < TK_SW


def _log_sinc_series(pmax=20):
    """log(sin u/u) + u^2/6 = sum_{p>=2} l_p u^{2p}  (60-digit recurrence)."""
    import mpmath as mp
    mp.mp.dps = 60
    f = [(-1) ** p / mp.factorial(2 * p + 1) for p in range(pmax + 1)]
    l = [mp.mpf(0)] * (pmax + 1)
    for p in range(1, pmax + 1):
        l[p] = f[p] - sum(m * l[m] * f[p - m] for m in range(1, p)) / p
    c = np.array([float(v) for v in l])
    c[1] += 1.0 / 6.0
    return c


_LSINC = _log_sinc_series()


def log_sinc_plus(u):
    """log|sinc u| + u^2/6, cancellation-free (series for u < 1)."""
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    small = np.abs(u) < 1.0
    if small.any():
        u2 = u[small] ** 2
        acc = np.zeros_like(u2)
        for c in _LSINC[::-1]:
            acc = acc * u2 + c
        out[small] = acc
    if (~small).any():
        ub = u[~small]
        with np.errstate(divide="ignore"):
            out[~small] = np.log(np.abs(np.sin(ub) / ub)) + ub ** 2 / 6.0
    return out


def g_of(kern, tau, k):
    """g = log|M_tau(k)| + tau k^2, broadcast; accurate to ~1e-12 relative."""
    tau = np.asarray(tau, dtype=float)
    k = np.asarray(k, dtype=float)
    shape = np.broadcast(tau, k).shape
    tk2 = np.broadcast_to(tau * k ** 2, shape)
    if kern["kind"] == "mix":
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.log(np.abs(symbol(kern, tau, k))) + tk2
        g = np.broadcast_to(g, shape).copy()
    else:
        # direct: log|sinc(u) (1 + tau Phat/sinc)| + u^2/6,  u^2/6 = tau k^2
        if "mom_series" not in kern:
            kern["mom_series"] = moment_series(kern["C_exact"])
        u = np.broadcast_to(np.sqrt(6.0 * tau) * k, shape)
        z = np.broadcast_to(k * tau ** 0.25, shape)
        sc = sinc(u)
        corr = np.broadcast_to(tau, shape) * Phat_poly(kern["C"], z, kern["mom_series"])
        safe = np.abs(sc) > 1e-3
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(safe, corr / np.where(safe, sc, 1.0), 0.0)
            log_ratio = np.where(np.abs(r) < 0.5, np.log1p(np.where(np.abs(r) < 0.5, r, 0.0)),
                                 np.log(np.abs(1.0 + r)))
            g = np.where(safe, log_sinc_plus(u) + log_ratio,
                         np.log(np.abs(sc + corr)) + tk2)
    if kern["kind"] == "mix":
        gp = _series_for(kern)
        small = tk2 < Y_SWITCH ** 2
        if small.any():
            y2 = tk2[small]
            acc = np.zeros_like(y2)
            for c in gp[::-1]:
                acc = acc * y2 + c
            g[small] = acc
    else:
        table = _series_for(kern)
        z = np.broadcast_to(k * tau ** 0.25, g.shape)
        small = (z < Z_SW) & (tk2 < TK_SW)
        if small.any():
            ss = np.broadcast_to(tau ** 0.25, g.shape)[small]
            kk = np.broadcast_to(k, g.shape)[small]
            dmax = max(table)
            acc = np.zeros_like(ss)
            for d in range(dmax, -1, -1):
                acc = acc * ss
                if d in table:
                    acc = acc + np.polyval(table[d], kk)
            g[small] = np.log1p(acc * np.exp(tk2[small]))
    return g


def deviations(kern, lam, ks, n_list, NT=300001, chunk=32):
    """dev_n(k) = R_{lam,n}(k) - 1/(lam+k^2) for all k in ks, n in n_list.

    EVEN n ONLY: the integrand is built from |M_tau(k)|^n = exp(n log|M|);
    for odd n the sign factor sgn(M_tau(k))^n would be needed wherever the
    symbol is negative (a positive measure need not have a positive
    characteristic function), and it is not implemented."""
    if any(int(n) % 2 for n in n_list):
        raise ValueError("deviations(): even n only (|M|^n route); got %r" % (n_list,))
    ks = np.asarray(ks, dtype=float)
    tau_end = 3.0 + 20.0 / lam
    taus = np.geomspace(1e-14, tau_end, NT)
    out = np.empty((len(n_list), len(ks)))
    for c0 in range(0, len(ks), chunk):
        kc = ks[c0:c0 + chunk]
        g = g_of(kern, taus[:, None], kc[None, :])
        base = -taus[:, None] * (lam + kc[None, :] ** 2)      # -tau(lam+k^2)
        for i, n in enumerate(n_list):
            ng = n * g
            nb = n * base
            lin = np.abs(ng) < 1.0
            integ = np.where(
                lin,
                np.exp(nb) * np.expm1(np.where(lin, ng, 0.0)),
                np.exp(np.maximum(nb + ng, LOG_CLIP))
                - np.exp(np.maximum(nb, LOG_CLIP)))
            out[i, c0:c0 + chunk] = n * simpson(integ, x=taus, axis=0)
    return out


# ---------------------------------------------------------------------------
# 4. fixed-k constants
# ---------------------------------------------------------------------------
def D_analytic(name, lam, k):
    k = np.asarray(k, dtype=float)
    if name == "RV6":
        return 6.0 * (13.0 / 105.0 * k ** 6 - k ** 8 / 7800.0) / (lam + k ** 2) ** 4
    if name == "RV8":
        return 6.0 * (13.0 / 105.0) * k ** 6 / (lam + k ** 2) ** 4
    if name == "MIX2":
        return 6.0 * (4.0 / 105.0) * k ** 6 / (lam + k ** 2) ** 4
    if name == "GM2":
        c8 = -(15120.0 - 13200.0) / 9.0 / 40320.0    # y^8 coeff of log F
        return 24.0 * c8 * k ** 8 / (lam + k ** 2) ** 5
    raise KeyError(name)


def fixed_k(lams):
    print("== 4. fixed-k constants: n^p dev_n(k) at n = 4096 vs analytic ==")
    rows = []
    ks = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    for name, kern in KERNELS.items():
        p = kern["p"]
        for lam in lams:
            dev = deviations(kern, lam, ks, [1024, 4096])
            for i, k in enumerate(ks):
                Da = D_analytic(name, lam, k)
                rows.append(dict(kernel=name, lam=lam, k=k, p=p,
                                 np_dev_1024=1024.0 ** p * dev[0, i],
                                 np_dev_4096=4096.0 ** p * dev[1, i],
                                 D_analytic=Da,
                                 ratio_4096=4096.0 ** p * dev[1, i] / Da))
        sub = [r for r in rows if r["kernel"] == name and r["lam"] == lams[-1]]
        print(f"  {name} (p={p}, lam={lams[-1]:.2f}): ratio n^p dev/D at n=4096: "
              + ", ".join(f"k={r['k']:.0f}: {r['ratio_4096']:.4f}" for r in sub))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day1_fixedk.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 5. boundary layer
# ---------------------------------------------------------------------------
GLX, GLW = np.polynomial.legendre.leggauss(160)
U_NODES = 40.0 * (GLX + 1.0)
U_WTS = 40.0 * GLW


def H_poly(C, rho):
    """H(rho) = (1/rho) int e^{-u} [-u^2/5 + (u/rho) Phat((rho u)^{1/4})] du,
    series for rho <= 3 (exact moments), quadrature beyond."""
    M2P = poly_moments(C)
    rho = np.asarray(rho, dtype=float)
    out = np.empty_like(rho)
    lo = rho <= 3.0
    if lo.any():
        acc = np.zeros(lo.sum())
        for p in range(3, 81):
            c = ((-1) ** p * M2P[p]
                 * np.exp(gammaln(p / 2 + 2) - gammaln(2 * p + 1)))
            acc += c * rho[lo] ** (p / 2 - 2)
        out[lo] = acc
    if (~lo).any():
        for idx in np.where(~lo)[0]:
            r = rho[idx]
            z = (r * U_NODES) ** 0.25
            integ = np.exp(-U_NODES) * (-U_NODES ** 2 / 5.0
                                        + U_NODES / r * Phat_poly(C, z))
            out[idx] = integ @ U_WTS / r
    return out


def boundary_layer(lam_of):
    print("== 5. boundary layer k^2 = rho n (lam = 2 lam_c of each signed kernel) ==")
    rhos = np.geomspace(0.01, 100.0, 9)
    ns = [1024, 4096]
    rows = []
    for name, kern in KERNELS.items():
        lam = lam_of[name]
        for n in ns:
            ks = np.sqrt(rhos * n)
            dev = deviations(kern, lam, ks, [n])[0]
            if kern["kind"] == "poly":
                Hp = H_poly(P6_C if name == "RV6" else P8_C, rhos)
            else:
                Hp = np.zeros_like(rhos)
            Dk = D_analytic(name, lam, ks)
            p = kern["p"]
            for r, k, d, h, D in zip(rhos, ks, dev, Hp, Dk):
                rows.append(dict(kernel=name, lam=lam, n=n, rho=r, k=k,
                                 n2dev=n ** 2 * d, H_pred=h,
                                 D_fixedk=D, uniformity=n ** p * d / D))
        sub = pd.DataFrame([r for r in rows if r["kernel"] == name
                            and r["n"] == 4096])
        if kern["kind"] == "poly":
            print(f"  {name} (lam={lam:.1f}): n^2 dev vs H(rho) at n=4096: "
                  + ", ".join(f"rho={r.rho:.2g}: {r.n2dev:+.3e}/{r.H_pred:+.3e}"
                              for r in sub.itertuples()))
        print(f"  {name} (lam={lam:.1f}): uniformity n^p dev_n(k_n)/D(k_n) at "
              f"n=4096 (1 = no boundary layer): "
              + ", ".join(f"rho={r.rho:.2g}: {r.uniformity:.4f}"
                          for r in sub.itertuples()), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day1_boundary_layer.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 6. rough data
# ---------------------------------------------------------------------------
J_LIM = 8192


def limit_profile(name, lam, xi):
    """n^p-scaled limiting error profile F(x) = sum_j a_j D(2j) cos(2jx) on
    XGRID, J_LIM modes; for RV6 the D -> -1/1300 tail is added in closed form
    (Study II convention), for the others the tail is O(J_LIM^{-2-xi})."""
    a0, a = a_coeffs(xi, J_LIM)
    ks = 2.0 * np.arange(1, J_LIM + 1)
    D = D_analytic(name, lam, ks)
    coef = a * (D + 1.0 / 1300.0) if name == "RV6" else a * D
    F = np.zeros_like(XGRID)
    for c0 in range(0, J_LIM, 1024):
        F += np.cos(np.outer(XGRID, ks[c0:c0 + 1024])) @ coef[c0:c0 + 1024]
    if name == "RV6":
        F -= (np.sin(XGRID) ** xi - a0 / 2.0) / 1300.0
    return F


def rough_data(lam_sets):
    print("== 6. rough data f_xi = |sin x|^xi, finite-n modes m = 2j <= 1024 ==")
    rows = []
    j = np.arange(1, J0 + 1)
    ks = 2.0 * j
    cosmat = np.cos(np.outer(XGRID, ks))
    for name, kern in KERNELS.items():
        p = kern["p"]
        for lam in lam_sets[name]:
            t0 = time.time()
            dev = deviations(kern, lam, ks, NS)          # (len(NS), J0)
            for xi in XIS:
                a0, a = a_coeffs(xi, J0)
                F_lim = limit_profile(name, lam, xi)
                F_lim_sup = np.max(np.abs(F_lim))
                F_lim_0 = F_lim[0]
                for i, n in enumerate(NS):
                    if not np.all(np.isfinite(dev[i])):
                        E, Fn0, xarg = np.inf, np.inf, np.nan
                    else:
                        Fn = cosmat @ (a * dev[i])
                        E = np.max(np.abs(Fn))
                        Fn0 = Fn[0]
                        xarg = XGRID[int(np.argmax(np.abs(Fn)))]
                    rows.append(dict(kernel=name, p=p, lam=lam, xi=xi, n=n,
                                     E_inf=E, np_E=n ** p * E,
                                     Fn0_scaled=n ** p * Fn0, x_argmax=xarg,
                                     F_lim_sup=F_lim_sup, F_lim_0=F_lim_0,
                                     plateau_ratio=n ** p * E / F_lim_sup))
            last = {r["xi"]: r for r in rows if r["kernel"] == name
                    and r["lam"] == lam and r["n"] == 4096}
            print(f"  {name} lam={lam:8.3f}: " + "; ".join(
                f"xi={xi}: ||F||={last[xi]['F_lim_sup']:.3e}, "
                f"n^{p}E(4096)/||F||={last[xi]['plateau_ratio']:.4f}"
                for xi in XIS) + f"   [{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day1_rough.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 7. local CLT in L^1 (positive kernels)
# ---------------------------------------------------------------------------
def local_clt():
    print("== 7. ||mu^{*n} - g||_L1 at t = 1 (FFT) ==")
    N = 2 ** 18
    dx = 0.002
    x = (np.arange(N) - N // 2) * dx
    dk = 2 * np.pi / (N * dx)
    kk = (np.arange(N) - N // 2) * dk
    rows = []
    ns = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for name in ("MIX2", "GM2"):
        kern = KERNELS[name]
        for n in ns:
            Fn = symbol(kern, 1.0 / n, kk) ** n
            diff_hat = Fn - np.exp(-kk ** 2)
            # inverse FT: p(x) = (1/2pi) int diff_hat e^{ikx} dk
            d = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(diff_hat))) \
                * N * dk / (2 * np.pi)
            L1 = np.sum(np.abs(d.real)) * dx
            rows.append(dict(kernel=name, n=n, L1=L1, n2L1=n ** 2 * L1,
                             n3L1=n ** 3 * L1))
        sub = pd.DataFrame([r for r in rows if r["kernel"] == name])
        ok = sub.L1.values > 1e-11                  # above the FFT/float floor
        slope = np.polyfit(np.log(sub.n.values[ok][-4:]),
                           np.log(sub.L1.values[ok][-4:]), 1)[0]
        print(f"  {name}: L1 at n=16..4096: "
              + ", ".join(f"{v:.2e}" for v in sub.L1.values)
              + f";  slope (last 4 above 1e-11) = {slope:.3f}", flush=True)
    # Edgeworth constants: leading term n^{-p} |kappa_{2p+2}|/(2p+2)! ||He_{2p+2} phi||_1
    #   MIX2: kappa_6 (standardised) = -24/7;  GM2: kappa_8 = (m8 - 105 m2^4)/m2^4
    from numpy.polynomial.hermite_e import hermeval
    from scipy.integrate import trapezoid
    xs = np.linspace(-14, 14, 400001)
    phi = np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi)
    he6 = hermeval(xs, [0] * 6 + [1])
    he8 = hermeval(xs, [0] * 8 + [1])
    c_edge = (24.0 / 7.0) / 720.0 * trapezoid(np.abs(he6 * phi), xs)
    # GM2 8th moment: E s^8 = E v^4 / 9 with E v^4 = 13200 (Gauss nodes), sigma^2 = 2
    kappa8 = (13200.0 / 9.0 - 105.0 * 16.0) / 16.0
    c_edge8 = abs(kappa8) / 40320.0 * trapezoid(np.abs(he8 * phi), xs)
    gm_1024 = [r["L1"] for r in rows if r["kernel"] == "GM2" and r["n"] == 1024][0]
    mix_4096 = [r["L1"] for r in rows if r["kernel"] == "MIX2" and r["n"] == 4096][0]
    print(f"  Edgeworth predictions: MIX2 n^2 L1 -> {c_edge:.5f} (measured n=4096: "
          f"{4096**2 * mix_4096:.5f});  GM2 n^3 L1 -> {c_edge8:.5f} (measured "
          f"n=1024: {1024**3 * gm_1024:.5f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day1_clt.csv"), index=False)
    return df, c_edge


# ---------------------------------------------------------------------------
# 8. the M8 family
# ---------------------------------------------------------------------------
def m8_family():
    print("== 8. family P_s = (1-s) P6 + s P8 ==")
    rows = []
    for s in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        C = {j: (1 - s) * float(P6_C.get(j, 0)) + s * float(P8_C.get(j, 0))
             for j in (0, 2, 4, 6, 8)}
        Cs = {j: sp.nsimplify((1 - s)) * P6_C.get(j, 0) + sp.nsimplify(s) * P8_C.get(j, 0)
              for j in (0, 2, 4, 6, 8)}
        kern = dict(name=f"P_{s}", kind="poly", C=C, C_exact=Cs, p=2)
        (lc, ts, ks), supM = lambda_c(kern, taus=np.geomspace(1e-4, 2.0, 150))
        tv = poly_TV(Cs)
        M8 = poly_moments(C)[4]
        rows.append(dict(s=s, lambda_c=lc, tau_star=ts, k_star=ks, TV_slope=tv,
                         M8=M8, H0=6.0 * M8 / 40320.0))
        print(f"  s={s:+.2f}: lambda_c = {lc:8.3f}, TV slope = {tv:8.2f}, "
              f"M8 = {M8:+.5f}, H(0) = {6*M8/40320:+.3e}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "day1_m8_family.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def figure(df_fix, df_bl, df_rough, df_clt, df_fam):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    lam = df_rough.lam.max()
    kk = np.geomspace(0.5, 200, 400)
    for name in KERNELS:
        p = KERNELS[name]["p"]
        ax[0, 0].loglog(kk, np.abs(D_analytic(name, 2 * LAM_C_RV6, kk)),
                        label=f"{name} (p={p})")
    ax[0, 0].set_xlabel("k"); ax[0, 0].set_ylabel("|D(k)| at lam = 2 lam_c(RV6)")
    ax[0, 0].set_title("fixed-k limit constants"); ax[0, 0].legend()
    for name in KERNELS:
        sub = df_bl[(df_bl.kernel == name) & (df_bl.n == 4096)]
        ax[0, 1].semilogx(sub.rho, sub.n2dev, "o-",
                          label=f"{name} n=4096 lam={sub.lam.iloc[0]:.0f}")
        if KERNELS[name]["kind"] == "poly":
            ax[0, 1].semilogx(sub.rho, sub.H_pred, "k--", lw=0.8)
    ax[0, 1].set_xlabel("rho = k^2/n"); ax[0, 1].set_ylabel("n^2 dev_n")
    ax[0, 1].set_title("boundary layer (dashed: H(rho))"); ax[0, 1].legend()
    for name in KERNELS:
        for xi, mk in zip(XIS, ("o", "s", "^")):
            sub = df_rough[(df_rough.kernel == name) & (df_rough.xi == xi)]
            lam_use = sub.lam.max()
            sub = sub[(sub.lam == lam_use) & (sub.n >= 16)]
            ax[1, 0].semilogx(sub.n, sub.plateau_ratio, mk + "-",
                              label=f"{name} xi={xi} lam={lam_use:.0f}")
    ax[1, 0].axhline(1.0, color="k", lw=0.6)
    ax[1, 0].set_ylim(0.0, 1.3)
    ax[1, 0].set_xlabel("n"); ax[1, 0].set_ylabel("n^p E_inf / ||F||_inf")
    ax[1, 0].set_title("rough-data plateau approach (n >= 16)")
    ax[1, 0].legend(fontsize=6, ncol=2)
    for name in ("MIX2", "GM2"):
        sub = df_clt[df_clt.kernel == name]
        ax[1, 1].loglog(sub.n, sub.L1, "o-", label=name)
    nn = np.array([16, 4096])
    ax[1, 1].loglog(nn, 0.03 * (nn / 16.0) ** -2, "k:", label="n^-2")
    ax[1, 1].loglog(nn, 0.003 * (nn / 16.0) ** -3, "k--", label="n^-3")
    ax[1, 1].set_xlabel("n"); ax[1, 1].set_ylabel("||mu^{*n} - g||_L1")
    ax[1, 1].set_title("local CLT (bounds sup-error on all bounded f)")
    ax[1, 1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "day1_design_set.png"), dpi=150)


if __name__ == "__main__":
    t_all = time.time()
    tangency()
    df_thr = thresholds()
    lc8 = float(df_thr[df_thr.kernel == "RV8"].lambda_c.iloc[0])
    lams_common = [1.1 * LAM_C_RV6, 2.0 * LAM_C_RV6]
    df_fix = fixed_k(lams_common + [2.0 * lc8])
    df_bl = boundary_layer({"RV6": 2.0 * LAM_C_RV6, "RV8": 2.0 * lc8,
                            "MIX2": 2.0 * LAM_C_RV6, "GM2": 2.0 * LAM_C_RV6})
    lam_sets = {
        "RV6": lams_common,
        "RV8": sorted(set(lams_common + [1.1 * lc8, 2.0 * lc8])),
        "MIX2": [1.0] + lams_common,
        "GM2": [1.0] + lams_common,
    }
    df_rough = rough_data(lam_sets)
    df_clt, c_edge = local_clt()
    df_fam = m8_family()
    figure(df_fix, df_bl, df_rough, df_clt, df_fam)
    print(f"== done in {time.time()-t_all:.0f}s ==")
