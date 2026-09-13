# Addendum to Study II: scripts

Scripts behind the addendum

> S. Shpital, *Addendum to "Spectral Phase Transition and Rough-Data Boundary
> Layers in a Fast Chernoff Approximation": Two Prices of Sign, a Positive
> Second-Order Kernel, and Prior Art*, 2026 (same Zenodo record as the paper,
> [doi:10.5281/zenodo.22643037](https://doi.org/10.5281/zenodo.22643037)).

They run from this folder and depend only on the Study II scripts in the
repository root (`day6_xi.py` is imported by `day1_design_set.py`) and on the
packages of `requirements.txt` (`numpy`, `scipy`, `sympy`, `mpmath`,
`pandas`, `matplotlib`). Outputs (`data/*.csv`, `figs/*.png`) are written
next to the scripts and are not versioned; the console output is the record.

| Script | Addendum | What it prints | Runtime |
|---|---|---|---|
| `addendum_check.py` | Sections 2, 3.1; Table 1 | `omega = int|P|` and the zeros of `P` (25 digits), exact `||mu_tau||_TV` and its linear regime `tau <= tau_0`, `g(tau) = log||mu_tau||_TV / tau` and its minimum on the contraction regime, FFT norms `||mu_tau^{*n}||_TV` vs `sup_k|M|^n` vs `||mu_tau||_TV^{2n}`, tangency expansions of RV6, MIX2, GM2, `omega_8` for the degree-8 kernel | 1 min |
| `day1_design_set.py` | Section 2 (Remark 2), Section 3.1 | design-set table for RV6, RV8, MIX2, GM2: tangency, `lambda_c`, `omega`, boundary-layer uniformity ratio, rough data, `L^1` rates and Edgeworth constants; segment `P_s = (1-s)P + sP_8` | 10 min |
| `day1b_beta_sweep.py` | not cited (auxiliary) | `lambda_c` and `omega` along a one-parameter deformation of the RV kernel; imports `day1_design_set.py` | 2 min |
| `day2_varcoef_positive.py` | Section 3.2, eq. (3) | symbolic solution of the six moment equations for the two-window scheme (`w_i, c_i, r_i^2`), symbolic second-order check for generic `a, b, c`, numerical order test | 3 min |
| `day2b_varcoef_norm.py` | Section 3.2 | discrete operator norm `||A_tau^n - e^{T L_N}||_{inf->inf}` for `a = 1 + sin(x)/2`, `b = 0.3 cos x`, `T = 0.5`, `N = 128..1024` (`n^2 ||.|| = 0.1465` at `n = 512`); imports `day2_varcoef_positive.py` | 5 min |
| `day3_adjoint_check.py` | Section 3.2 (Corollary 1, hypothesis `E*`) | dual expansion `S(h)^* g = g + h H^* g + h^2/2 (H^*)^2 g + O(h^3)` in exact rational arithmetic, two random trials; `--c` adds `c != 0` | 5 min |

```bash
cd addendum
python addendum_check.py
python day1_design_set.py
python day2_varcoef_positive.py
python day2b_varcoef_norm.py
python day3_adjoint_check.py
```

Release tag of the addendum: `study2_addendum_v1`.
