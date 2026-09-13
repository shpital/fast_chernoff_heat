"""Study III, day 1b: the beta-family of signed corrections (numerical
confirmation of the analytic fixed-profile scaling).

    M_tau^{(beta)}(k) = sinc(sqrt(6 tau) k) + tau^{2-4beta} Phat(k tau^beta),
    lambda_c(beta) = sup_{tau,k} log|M| / tau,

with the *same* polynomial profile P (RV6: admissible for beta in [1/4, 1/2);
RV8, M8 = 0: admissible for beta in [1/6, 1/2)).  Analytic prediction (note
2026-09-07, sec. 2): with x = tau^{1-2beta}, k = z tau^{-beta} the symbol
sinc(sqrt6 z sqrt x) + x^2 Phat(z) is beta-independent, hence

    lambda_c(beta) >= log|M(x*,z*)| / x*^{1/(1-2beta)}  -> infinity  (beta -> 1/2-)

from the single RV amplification point (x*, z*).  Also the endpoint beta = 1/2
(self-similar symbol F(k sqrt tau)): lambda_c = infinity iff sup|F| > 1.

STATUS OF THE NUMBERS.  A grid search over (tau, k) only produces values of
log|M|/tau at grid points, i.e. LOWER BOUNDS for the supremum.  Each row
reports: the search value, whether its maximiser sits on the tau-grid boundary
(then the supremum is certainly not localised), the analytic single-point
bound, and the certified lower bound max(search, bound).  Values are quoted as
lambda_c only when the maximiser is interior and the two agree.
"""
import os
import time

import numpy as np
import pandas as pd

from day1_design_set import KERNELS, Phat_poly, sinc, DATA, moment_series

BETAS = {"RV6": [0.25, 0.275, 0.30, 0.35, 0.40, 0.45, 0.475],
         "RV8": [1 / 6, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.475]}
# amplification points of the beta = 1/4 members (day1 thresholds)
AMP = {"RV6": (0.014479, 21.9642, 73.9875), "RV8": (0.004839, 35.8658, 226.1093)}
TAU_MIN, TAU_MAX, N_TAU = 1e-10, 3.0, 900
K_MIN, K_MAX, N_K = 0.5, 3.0e5, 120000


def lambda_c_beta(kern, beta, taus, ks):
    ser = moment_series(kern["C_exact"])
    best = (-np.inf, None, None)
    for tau in taus:
        z = ks * tau ** beta
        M = sinc(np.sqrt(6.0 * tau) * ks) + tau ** (2 - 4 * beta) * Phat_poly(kern["C"], z, ser)
        with np.errstate(divide="ignore"):
            phi = np.log(np.abs(M)) / tau
        i = int(np.argmax(phi))
        if phi[i] > best[0]:
            best = (phi[i], tau, ks[i])
    return best


def main():
    t0 = time.time()
    taus = np.geomspace(TAU_MIN, TAU_MAX, N_TAU)
    ks = np.geomspace(K_MIN, K_MAX, N_K)
    rows = []
    print("== beta-family: lambda_c(beta) for the fixed RV profiles ==")
    print(f"   grid: tau in [{TAU_MIN:g}, {TAU_MAX:g}] x {N_TAU}, k in [{K_MIN:g}, {K_MAX:g}] x {N_K}")
    for name, betas in BETAS.items():
        kern = KERNELS[name]
        tau_s, k_s, lc0 = AMP[name]
        logM_s = lc0 * tau_s                    # log|M| at the beta=1/4 amplification point
        x_s = np.sqrt(tau_s)                    # x* = tau*^{1-2*1/4}
        z_s = k_s * tau_s ** 0.25
        for beta in betas:
            eps = 1 - 2 * beta
            lb = logM_s / x_s ** (1 / eps)
            lc, tau_c, k_c = lambda_c_beta(kern, beta, taus, ks)
            on_edge = (tau_c <= taus[0] * 1.0001) or (tau_c >= taus[-1] * 0.9999) \
                or (k_c <= ks[0] * 1.0001) or (k_c >= ks[-1] * 0.9999)
            cert = max(lc, lb)
            status = ("lower bound (grid edge)" if on_edge else
                      ("interior maximiser" if lc >= 0.98 * lb else
                       "lower bound (search below single-point bound)"))
            rows.append(dict(kernel=name, beta=beta, search_value=lc, tau_star=tau_c,
                             k_star=k_c, z_star=k_c * tau_c ** beta,
                             x_star=tau_c ** eps, on_grid_edge=on_edge,
                             single_point_bound=lb, certified_lower_bound=cert,
                             status=status))
            print(f"  {name} beta={beta:.4f}: search = {lc:12.4g}  (tau*={tau_c:.3e}, "
                  f"k*={k_c:.4g}, x*={tau_c**eps:.4f}, z*={k_c*tau_c**beta:.3f});  "
                  f"single-point bound = {lb:12.4g};  certified >= {cert:12.4g}  [{status}]",
                  flush=True)
        print(f"  ({name}: beta=1/4 amplification point x*={x_s:.4f}, z*={z_s:.3f}, "
              f"log|M*|={logM_s:.4f})")
    # endpoint beta = 1/2: self-similar symbol F(y) = sinc(sqrt6 y) + Phat(y), y = k sqrt(tau)
    for name in BETAS:
        kern = KERNELS[name]
        y = np.geomspace(1e-3, 200.0, 200000)
        F = sinc(np.sqrt(6.0) * y) + Phat_poly(kern["C"], y, moment_series(kern["C_exact"]))
        print(f"  {name} beta=1/2 endpoint: sup|F| >= {np.max(np.abs(F)):.4f} at y = "
              f"{y[int(np.argmax(np.abs(F)))]:.3f}  => lambda_c = "
              f"{'infinity' if np.max(np.abs(F)) > 1 else 'undetermined by this grid'}")
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "day1b_beta_sweep.csv"), index=False)
    print(f"== done in {time.time()-t0:.0f}s ==")


if __name__ == "__main__":
    main()
