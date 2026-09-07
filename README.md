# Spectral Phase Transition and Rough-Data Boundary Layers in a Fast Chernoff Approximation

Code for reproducing the computations of the preprint

> S. Shpital, *Spectral Phase Transition and Rough-Data Boundary Layers in a
> Fast Chernoff Approximation*, 2026.
> Published version (Zenodo): [doi:10.5281/zenodo.22643037](https://doi.org/10.5281/zenodo.22643037)

The fast (second-order) Chernoff function of Remizov and Vedenin for the heat
semigroup replaces the positive shift lattice by a signed, absolutely
continuous one-step kernel. The scripts here quantify the price of that
signed correction on the constant-coefficient heat equation:

- the exact fixed-mode defect and the `n^{-2}` tangency of the one-step
  Fourier symbol `M_tau(k)`;
- the high-frequency amplification band of the symbol and its transient
  role in the evolution problem `C(t/n)^n`;
- the resolvent phase transition: under Laplace transformation the band is
  sharpened into a computable threshold `lambda_c = 73.987...`, below which
  `R_{lambda,n}` diverges exponentially in `n` and above which second order
  holds (Proposition 1, three-regime modal asymptotics);
- an independent real-space grid bridge that excludes implementation
  artefacts;
- the parabolic boundary layer `k^2 = rho n` above the threshold: the
  `lambda`-free limit `H(rho)` in closed form from the moments of the
  correction kernel, and the crossover function `G = -1300 H` with
  `G(0) = 1`, `G(rho) = 1 - 0.0380 sqrt(rho) + O(rho)`, `G ~ 520/rho`
  (Proposition 2);
- the Hölder family `|sin x|^xi`: `xi`-independent threshold, onset below
  it, recovery above it; the conditional cusp-point recovery law with
  explicit constants (Proposition 3) and the audit of where the maximum of
  the error profile sits (Lemma 2, Corollary 1, Table 1).

Everything is constant-coefficient, spectral (Fourier on the torus) unless
stated otherwise; no external fixture is needed.

## Requirements

Python 3.10+ with `numpy`, `scipy`, `pandas`, `matplotlib`, `sympy`,
`mpmath` (`requirements.txt`). All computations are deterministic
`float64` except the pre-registered high-precision references in
`day1_symbol.py` (mpmath, 50 digits), which are used only to certify the
float64 floor.

```bash
git clone https://github.com/shpital/fast_chernoff_heat
cd fast_chernoff_heat
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Layout

The scripts form a pipeline; each reads the CSVs written by the previous
ones into `data/` (created on first run). Scripts 1–8 are the
pre-registered study; scripts 9–10 were added during the proof audit of
the paper (v0.4–v0.5) and use frozen data only.

| Script | Produces | Content |
|---|---|---|
| `day1_symbol.py` | `data/day1_*.csv`, `figs/day1_symbol.png` | one-step symbol `M_tau(k)`: exact moment identities (sympy), tangency and leading defect, float64 floor `1.5e-13` against 50-digit references, stability/composition tables |
| `day2_spectral.py` | `data/day2_*.csv`, `figs/day2_spectral.png` | evolution sweep: fixed modes, amplification norms, rough-datum composition, periodic checks |
| `day3_resolvent.py` | `data/day3_*.csv`, `figs/day3_resolvent.png` | resolvent quadratures in the log domain (values up to `10^1257` exact in the exponent); `lambda_c(k)`; the `|sin x|` kill-test across the threshold |
| `day4_hardening.py` | `data/day4_*.csv` | `lambda_c` localization: global scan in `(tau, z)`, local optimization, range-extension check, critical-line verification, true `E_inf` of `|sin x|` |
| `day5_gridbridge.py` | `data/day5_*.csv`, `figs/day5_bridge.png` | independent real-space bridge (`P_1` and cubic representations, `h`-refinement) |
| `day6_xi.py` | `data/day6_*.csv`, `figs/day6_xi.png` | Hölder family `|sin x|^xi`: closed-form Fourier coefficients, onset below `lambda_c`, recovery above it, crossover tables (frozen deviation tables `day6_dev_lam*.csv`) |
| `day7_G.py` | `data/day7_*.csv`, `figs/day7_G.png` | crossover function `G(rho)`: series/quadrature evaluator (`3e-10` overlap agreement), factorized comparison with day-6 data, recovery model |
| `day9_p3_constants.py` | `data/day9_p3_constants.csv` | explicit constants of Proposition 3 inside the `G`-model: `I_xi`, `S_n` ratios, `lambda`-independence of the leading constants |
| `day10_maximizer.py` | `data/day10_maximizer.csv` | maximizer audit (Table 1): limit and finite-`n` error profiles for the six safe configurations, `G`-model tail completion, truncation bias of the frozen plateau ratios |
| `note/make_figs.py` | `note/figs/fig*.png` | the four figures of the paper; replots frozen CSV data only |

`data/`, `figs/` and `note/figs/` are not tracked: they are regenerated in
full by running the pipeline below. No frozen data is shipped or needed ---
every CSV, every working figure and the four figures of the paper are
outputs of the scripts in this repository.

## Reproducing the study

```bash
PY=.venv/bin/python
$PY day1_symbol.py           #   4 s
$PY day2_spectral.py         #   3 s
$PY day3_resolvent.py        # ~ 5 min  (resolvent quadratures, log domain)
$PY day4_hardening.py        # ~ 4 min  (lambda_c localization, critical line)
$PY day5_gridbridge.py       # ~ 2 min  (real-space bridge, N = 16384)
$PY day6_xi.py               # ~ 9 min  (modal deviation tables, m <= 1024, 6 n, 3 lambda)
$PY day7_G.py                #   5 s
$PY day9_p3_constants.py     #   2 s
$PY day10_maximizer.py       # ~ 1.5 min
$PY note/make_figs.py        #   2 s
```

Total wall time about 21 minutes on one core of a 2024 laptop (times measured
on the clean re-run described below).

Every script prints its record to the console; the numbers quoted in the
paper are those console values and the CSVs. The scripts of days 3, 4 and 6
cache their expensive tables in `data/` and reuse them on subsequent runs;
delete `data/` for a clean re-run. Before the release the whole pipeline was
re-executed from a clean copy of this repository (no `data/`, `figs/`,
`note/figs/`) and the 24 regenerated CSVs were compared column by column
with the frozen tables behind the paper: 21 files are identical, the other
three (`day6_crossover`, `day6_recovery`, `day7_Gcheck`) agree to relative
`3e-14`, `4e-15` and `3e-9` (the last on the series/quadrature overlap
diagnostic, whose own level is `3e-10`). All computations are
deterministic; differences of this size are last-digit rounding.

## Related records

- Paper (this study):
  [doi:10.5281/zenodo.22643037](https://doi.org/10.5281/zenodo.22643037)
- Companion study of positive shift families under Laplace transformation:
  doi:10.5281/zenodo.22550189,
  <https://github.com/shpital/chernoff_laplace_heat>
- Gauss–Hermite hierarchy of positive Chernoff approximations:
  doi:10.5281/zenodo.22224038, <https://github.com/shpital/gauss_hermite_heat>
- Chebyshev-controlled Chernoff approximations (heat note):
  doi:10.5281/zenodo.21708475, <https://github.com/shpital/cheb_chernoff_heat>

## License

CC BY 4.0, matching the paper's Zenodo record.
