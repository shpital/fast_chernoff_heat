"""II.A grid-bridge smoke-test (day 5): real-space grid vs spectral reference.

Scope (pre-registered, deliberately narrow -- this is a bridge, not a
benchmark): datum |sin x| on the periodic uniform grid, TWO configurations
of the threshold:

    safe:      lambda = 2.0 * lambda_c   (boundary-dominated, n^{-2} law),
    unstable:  lambda = 0.9 * lambda_c   (interior maximum, e^{n Gamma}),

n in {4, 16, 64} (small n on the unstable side keeps numbers non-astronomic).

Real-space implementation (independent of the day-3 spectral route):
  * one step = circulant convolution on the grid; the weights are the exact
    integrals of the one-step density
        mu_tau(s) = 1_{|s|<=sqrt(6 tau)} / (2 sqrt(6 tau))
                  + tau^{3/4} P(s/tau^{1/4}) 1_{|s|<=tau^{1/4}}
    against periodic linear hat functions (P1 interpolation of f), with
    integration split at the support edges (uniform part in closed form,
    polynomial part by Gauss-Legendre, exact for the degree);
  * n-fold composition and the outer Laplace quadrature
        R = n * int_0^infty [e^{-lambda tau} C(tau)]^n dtau
    on geometric Gauss-Legendre panels; the circulant is applied via its
    DFT eigenvalues (FFT only diagonalizes the *discretized* operator).

Comparisons:
  * E_inf_grid  = max_x |R_grid f - R_exact f|   (R_exact per DFT mode),
  * E_inf_spec  = same with day-3 log-domain modal resolvent (J <= 200),
  * bridge gap  = max_x |R_grid f - R_spec f|  -- must be << E_inf,
  * modal table R_grid(m)/R_spectral(m) for m in {2, 18, 22},
  * N-doubling (8192 -> 16384) to estimate E_repr.

Outputs: data/day5_bridge.csv, data/day5_modal.csv, figs/day5_bridge.png.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day3_resolvent import log_resolvent  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")

LAM_C = 73.987479          # day-4 certified value
CONFIGS = [("safe", 2.0 * LAM_C), ("unstable", 0.9 * LAM_C)]
NS = [4, 16, 64]
JMAX = 200                 # spectral route mode cutoff (as in days 3-4)

GLX8, GLW8 = np.polynomial.legendre.leggauss(8)
GLX16, GLW16 = np.polynomial.legendre.leggauss(16)


def P_poly(y):
    y2 = y * y
    return (14553.0 + y2 * (-280665.0 + y2 * (800415.0 - 567567.0 * y2))) / 64.0


# ---------------------------------------------------------------------------
# one-step circulant weights: mu_tau integrated against a local interpolation
# kernel (P1 hat, order 2, or Keys cubic convolution, order ~4 in the symbol)
# ---------------------------------------------------------------------------
# kernel pieces: (v_lo/h, v_hi/h, K(u)) with v = s - c, u = v/h (signed)
def _keys1(u):
    au = np.abs(u)
    return 1.5 * au**3 - 2.5 * au**2 + 1.0


def _keys2(u):
    au = np.abs(u)
    return -0.5 * au**3 + 2.5 * au**2 - 4.0 * au + 2.0


KERNELS = {
    "p1": [(-1.0, 0.0, lambda u: 1.0 + u), (0.0, 1.0, lambda u: 1.0 - u)],
    "keys": [(-2.0, -1.0, _keys2), (-1.0, 0.0, _keys1),
             (0.0, 1.0, _keys1), (1.0, 2.0, _keys2)],
}


def kernel_weights(tau, h, interp="p1"):
    """Circulant taps w_d = int mu_tau(s) K((s - d h)/h) ds, d = -D..D.

    Every (tap, kernel-piece) subinterval is clipped at the density support
    edges; GL-8 is then exact (uniform part x cubic, and the degree-6
    polynomial correction x cubic <= degree 9 < 16)."""
    a = np.sqrt(6.0 * tau)
    b = tau**0.25
    pieces = KERNELS[interp]
    half_supp = max(abs(p[0]) for p in pieces)
    D = int(np.ceil((max(a, b) + half_supp * h) / h))
    d = np.arange(-D, D + 1)
    c = d * h
    w = np.zeros_like(c, dtype=float)
    for ulo, uhi, K in pieces:
        for edge, dens in ((a, "unif"), (b, "corr")):
            lo = np.clip(c + ulo * h, -edge, edge)
            hi = np.clip(c + uhi * h, -edge, edge)
            hi = np.maximum(hi, lo)
            mid = 0.5 * (lo + hi)
            hw = 0.5 * (hi - lo)
            s = mid[:, None] + hw[:, None] * GLX8[None, :]
            kv = K((s - c[:, None]) / h)
            if dens == "unif":
                vals = kv / (2.0 * a)
            else:
                vals = tau**0.75 * P_poly(s / b) * kv
            w += hw * (vals @ GLW8)
    return d, w


# ---------------------------------------------------------------------------
# outer Laplace quadrature nodes (shared eigenvalue table)
# ---------------------------------------------------------------------------
def tau_nodes(tau_max=0.15, n_panels=48):
    edges = np.concatenate([[0.0], np.geomspace(1e-6, tau_max, n_panels)])
    taus, wts = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        taus.append(mid + half * GLX16)
        wts.append(half * GLW16)
    return np.concatenate(taus), np.concatenate(wts)


def eigen_table(taus, N, interp):
    """DFT eigenvalues of the discretized one-step operator, per tau node."""
    h = 2.0 * np.pi / N
    eig = np.empty((len(taus), N // 2 + 1))
    for i, tau in enumerate(taus):
        d, w = kernel_weights(tau, h, interp)
        mass = w.sum()
        assert abs(mass - 1.0) < 1e-9, f"kernel mass {mass} at tau={tau}"
        arr = np.zeros(N)
        np.add.at(arr, d % N, w)
        eig[i] = np.fft.rfft(arr).real     # symmetric kernel => real symbol
    return eig


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
def run_bridge(N=8192, interp="p1", label=""):
    print(f"== grid bridge, N = {N}, interp = {interp} {label}==")
    h = 2.0 * np.pi / N
    x = -np.pi + h * np.arange(N)
    f = np.abs(np.sin(x))
    fh = np.fft.rfft(f)
    m = np.arange(N // 2 + 1, dtype=float)

    taus, wts = tau_nodes()
    eig = eigen_table(taus, N, interp)
    print(f"  tau-quadrature: {len(taus)} nodes; eigenvalue table ready")

    Rref_h = fh / (LAM_C + m**2)  # placeholder, rebuilt per lambda below

    active = np.where(np.abs(fh) > 1e-9 * np.abs(fh).max())[0]
    rows, modal_rows, profiles = [], [], {}
    cache = {}
    for name, lam in CONFIGS:
        Rref_h = fh / (lam + m**2)
        R_ref = np.fft.irfft(Rref_h)
        for n in NS:
            # grid route
            base = np.exp(-lam * taus)[:, None] * eig
            Rh_grid = (wts[:, None] * n * base**n * fh[None, :]).sum(axis=0)
            R_grid = np.fft.irfft(Rh_grid)

            # spectral route (day-3 modal resolvent on the same amplitudes)
            Rh_spec = Rref_h.copy()
            for mi in active:
                if mi == 0:
                    Rh_spec[mi] = fh[mi] / lam
                elif mi <= JMAX:
                    key = (lam, mi, n)
                    if key not in cache:
                        cache[key] = np.exp(log_resolvent(lam, float(mi), n))
                    Rh_spec[mi] = fh[mi] * cache[key]
            R_spec = np.fft.irfft(Rh_spec)

            E_grid = np.max(np.abs(R_grid - R_ref))
            E_spec = np.max(np.abs(R_spec - R_ref))
            gap = np.max(np.abs(R_grid - R_spec))
            rows.append(dict(N=N, interp=interp, regime=name, lam=lam, n=n,
                             E_grid=E_grid, E_spec=E_spec, gap=gap,
                             gap_over_E=gap / E_spec))
            print(f"  {name:>8} lam={lam:8.3f} n={n:>3}: "
                  f"E_grid={E_grid:.4e}  E_spec={E_spec:.4e}  "
                  f"gap={gap:.2e}  gap/E={gap/E_spec:.2e}")
            if n == 16:
                profiles[name] = (x, R_grid - R_ref, R_spec - R_ref)

            # modal table
            for mi in (2, 18, 22):
                key = (lam, mi, n)
                if key not in cache:
                    cache[key] = np.exp(log_resolvent(lam, float(mi), n))
                Rg = Rh_grid[mi].real / fh[mi].real
                modal_rows.append(dict(N=N, interp=interp, regime=name,
                                       lam=lam, n=n, mode=mi,
                                       R_grid=Rg, R_spec=cache[key],
                                       rel=abs(Rg / cache[key] - 1.0)))
    return rows, modal_rows, profiles


if __name__ == "__main__":
    all_rows, all_modal = [], []
    # P1 route: documents the O(n h^2) representation floor and its halving
    rows, modal, _ = run_bridge(N=8192, interp="p1")
    all_rows += rows; all_modal += modal
    rows, modal, _ = run_bridge(N=16384, interp="p1", label="(doubling) ")
    all_rows += rows; all_modal += modal
    # Keys cubic route: floor pushed below every Chernoff error in the table
    rows, modal, profiles = run_bridge(N=8192, interp="keys")
    all_rows += rows; all_modal += modal
    rows, modal, _ = run_bridge(N=16384, interp="keys", label="(doubling) ")
    all_rows += rows; all_modal += modal

    print()
    print("== modal check R_grid(m) vs day-3 log-domain R(m), keys N=16384 ==")
    dfm = pd.DataFrame(all_modal)
    sel = dfm[(dfm.interp == "keys") & (dfm.N == 16384)]
    for _, r in sel.iterrows():
        print(f"  {r['regime']:>8} n={int(r['n']):>3} m={int(r['mode']):>2}: "
              f"R_grid={r['R_grid']:.8e}  R_spec={r['R_spec']:.8e}  "
              f"rel={r['rel']:.2e}")

    pd.DataFrame(all_rows).to_csv(os.path.join(DATA, "day5_bridge.csv"),
                                  index=False)
    dfm.to_csv(os.path.join(DATA, "day5_modal.csv"), index=False)

    # figure: error profiles, n = 16
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, name in zip(axes, ("safe", "unstable")):
        x, eg, es = profiles[name]
        ax.plot(x, eg, lw=1.0, label="grid route")
        ax.plot(x, es, lw=0.8, ls="--", label="spectral route")
        ax.set_title(f"{name}: error profile, n=16")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Grid bridge: $R_{\\lambda,n}f - R_\\lambda f$ for $|\\sin x|$"
                 " (real-space vs spectral)")
    fig.tight_layout()
    out = os.path.join(FIGS, "day5_bridge.png")
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")
