"""Numerical record for the Addendum to Study II ("two prices of sign").

Everything printed here is quoted in addendum.tex.  Inputs are the exact
Remizov--Vedenin correction polynomial P (Study II, eq. (2)) and the unique
degree-8 variant P8 with M_8 = 0 (study_III/day1_design_set.py); no data files
are read.

Checks:
  1. omega_bar = int_{-1}^{1} |P| dy  (the TV growth constant of Study II), its
     positive/negative halves, and the same constant for P8;
  2. the exact total-variation norm ||mu_tau||_TV of the signed one-step measure
     as a function of tau: 1 + omega_bar tau is the triangle bound, the exact
     value is 1 + 2 m^-(tau) with m^- the negative mass, and
     (||mu_tau||_TV - 1)/tau -> omega_bar as tau -> 0;
  3. the second-moment factor ||mu_tau||_TV^{2n}, n = T/tau, of the optimal
     importance-sampling estimator (T = 1), against e^{2 omega_bar T};
  4. the general inequality lambda_c <= omega_bar (sup_k |M_tau(k)| <= ||mu_tau||_TV)
     evaluated at the Study II critical point;
  5. tangency of the two positive kernels MIX2 (order 2) and GM2 (order 3).

Run:  python addendum_check.py        (about 10 s; needs sympy, mpmath, numpy)
"""
import mpmath as mp
import numpy as np
import sympy as sp

mp.mp.dps = 30
y = sp.symbols("y")

P6 = {0: sp.Rational(14553, 64), 2: sp.Rational(-280665, 64),
      4: sp.Rational(800415, 64), 6: sp.Rational(-567567, 64)}


def solve_P8():
    c = sp.symbols("c0 c2 c4 c6 c8")
    P = sum(ci * y ** (2 * i) for i, ci in enumerate(c))
    mom = lambda m: sp.integrate(P * y ** m, (y, -1, 1))
    sol = sp.solve([mom(0), mom(2), mom(4) - sp.Rational(24, 5), mom(6), mom(8)], c)
    return {2 * i: sol[ci] for i, ci in enumerate(c)}


def poly(C):
    return sum(cj * y ** j for j, cj in C.items())


def abs_integral(C):
    """int_{-1}^1 |P| dy to 25 digits: split at the real roots in (-1, 1)."""
    P = poly(C)
    roots = sorted({-1.0, 1.0} | {float(sp.re(r)) for r in sp.Poly(P, y).nroots(n=40)
                                   if abs(sp.im(r)) < 1e-30 and abs(sp.re(r)) < 1})
    Pf = sp.lambdify(y, P, "mpmath")
    pos = neg = mp.mpf(0)
    for lo, hi in zip(roots[:-1], roots[1:]):
        v = mp.quad(Pf, [lo, hi])
        if v > 0:
            pos += v
        else:
            neg -= v
    return pos + neg, pos, neg, roots


def tv_exact(C, tau):
    """Exact TV norm of mu_tau = U[-sqrt(6 tau), sqrt(6 tau)] + tau^{3/4} P(s/tau^{1/4}).

    In the variable y = s / tau^{1/4}:  mu_tau(s) ds = [u(y) + tau P(y)] dy with
    u = tau^{1/4} / (2 sqrt(6 tau)) on |y| <= sqrt(6) tau^{1/4}, 0 outside."""
    tau = mp.mpf(tau)
    P = poly(C)
    Pf = sp.lambdify(y, P, "mpmath")
    yw = mp.sqrt(6) * tau ** mp.mpf("0.25")          # half-width of the uniform window in y
    u = tau ** mp.mpf("0.25") / (2 * mp.sqrt(6 * tau))
    dens = lambda t: (u if abs(t) <= yw else 0) + (tau * Pf(t) if abs(t) <= 1 else 0)
    pts = sorted({-1, 1, -yw, yw} | {float(sp.re(r)) for r in sp.Poly(P, y).nroots(n=40)
                                     if abs(sp.im(r)) < 1e-30 and abs(sp.re(r)) < 1})
    pts = [mp.mpf(p) for p in pts if -1 <= p <= 1]
    if yw > 1:                                          # window wider than the correction support
        pts = [-yw] + pts + [yw]
    tv = mp.mpf(0)
    for lo, hi in zip(pts[:-1], pts[1:]):
        if hi > lo:
            tv += mp.quad(lambda t: abs(dens(t)), [lo, hi])
    return tv


def first_root(C):
    P = poly(C)
    return min(float(sp.re(r)) for r in sp.Poly(P, y).nroots(n=40)
               if abs(sp.im(r)) < 1e-30 and sp.re(r) > 0)


def tangency(name, sym, order_max=5):
    tau, k = sp.symbols("tau k", positive=True)
    s = sp.series(sym - sp.exp(-tau * k ** 2), tau, 0, order_max).removeO()
    s = sp.expand(s)
    lead = min(t.as_coeff_exponent(tau)[1] for t in sp.Add.make_args(s) if t != 0)
    coef = sp.factor(s.coeff(tau, lead))
    print(f"  {name}: M - e^(-tau k^2) = [{coef}] tau^{lead} + ...   (order p = {lead - 1})")


if __name__ == "__main__":
    print("== 1. omega_bar = int |P| dy ==")
    P8 = solve_P8()
    print("  P8 coefficients:", {j: str(c) for j, c in P8.items()})
    for name, C in (("RV6", P6), ("RV8", P8)):
        tot, pos, neg, roots = abs_integral(C)
        print(f"  {name}: omega_bar = {mp.nstr(tot, 12)},  int P^+ = {mp.nstr(pos, 10)},"
              f"  int P^- = {mp.nstr(neg, 10)},  roots in (-1,1): {[round(r, 6) for r in roots[1:-1]]}")
        print(f"        P(0) = {float(C[0]):.4f},  P(1) = {float(sum(C.values())):.4f}")
    omega6 = abs_integral(P6)[0]

    print("== 2. exact TV norm of mu_tau (RV6) ==")
    y1 = first_root(P6)
    tau0 = y1 ** 4 / 36
    print(f"  smallest positive root of P: y_1 = {y1:.6f};  the uniform window |y| <= sqrt(6) tau^(1/4)"
          f" lies inside the central positive lobe iff tau <= tau_0 = (y_1/sqrt 6)^4 = {tau0:.4e}")
    print(f"  {'tau':>10s} {'||mu||_TV - 1':>16s} {'(TV-1)/tau':>12s} {'omega tau (bound)':>18s} {'neg. mass':>12s}")
    taus = [1e-1, 3e-2, 0.014482, 1e-2, 5e-3, 3e-3, 2e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6]
    for t in taus:
        tv = tv_exact(P6, t)
        print(f"  {t:10.3e} {mp.nstr(tv - 1, 10):>16s} {mp.nstr((tv - 1) / t, 10):>12s}"
              f" {mp.nstr(omega6 * t, 10):>18s} {mp.nstr((tv - 1) / 2, 8):>12s}")

    print("== 3. second-moment factor ||mu_tau||_TV^(2n), n = T/tau, T = 1 ==")
    print(f"  asymptote 2 omega_bar T / ln 10 = {mp.nstr(2 * omega6 / mp.log(10), 8)} decades;"
          f"  exact (triangle bound attained) for n >= T/tau_0 = {1 / tau0:.0f} T")
    # g(tau) = log||mu_tau||_TV / tau: the factor is exp(2 T g(T/n)); g -> omega_bar as tau -> 0
    # restricted to tau <= 5e-3, the band-submersion threshold of Study II (sup_k |M_tau(k)| <= 1)
    grid = np.geomspace(1e-6, 5e-3, 241)
    g = np.array([float(mp.log(tv_exact(P6, t)) / t) for t in grid])
    i = int(np.argmin(g))
    print(f"  g(tau) = log||mu_tau||_TV/tau on [1e-6, 5e-3]: min = {g[i]:.3f} at tau = {grid[i]:.4e}"
          f" (n = {1 / grid[i]:.0f} T);  g(1e-6) = {g[0]:.4f};  g <= omega_bar everywhere: {bool(np.all(g <= float(omega6) + 1e-9))}")
    for n in (16, 64, 128, 256, 512, 1024, 4096, 16384, 65536):
        tv = tv_exact(P6, 1.0 / n)
        dec = 2 * n * mp.log10(tv)
        print(f"  n = {n:6d}: log10 ||mu||_TV^(2n) = {mp.nstr(dec, 8)} decades,"
              f"   ||mu||_TV^n = 10^{mp.nstr(dec / 2, 6)}")
    for T in (0.25, 1.0, 4.0):
        print(f"  T = {T}: asymptotic factor e^(2 omega_bar T) = 10^{mp.nstr(2 * omega6 * T / mp.log(10), 6)};"
              f"  lower bound over all steps e^(2 g_min T) = 10^{2 * g[i] * T / np.log(10):.2f}")

    print("== 3b. deterministic norm ||C(tau)^n||_(sup->sup) = ||mu_tau^{*n}||_TV by FFT (T = 1) ==")
    # symbol on a k-grid: M = sinc(k sqrt(6 tau)) + tau * sum_j c_j m_j(k tau^{1/4}), m_j(z) = 2 int_0^1 y^j cos(zy) dy
    xg, wg = np.polynomial.legendre.leggauss(96)
    yg, wy = 0.5 * (xg + 1.0), 0.5 * wg
    def symbol(k, tau):
        z = k * tau ** 0.25
        cos = np.cos(np.outer(z, yg))
        Phat = sum(2.0 * float(cj) * cos @ (wy * yg ** j) for j, cj in P6.items())
        arg = k * np.sqrt(6.0 * tau)
        return np.sinc(arg / np.pi) + tau * Phat
    K, N = 600.0, 2 ** 17
    dk = 2 * K / N
    k = (np.arange(N) - N // 2) * dk
    for n in (64, 128, 256, 1024, 4096, 16384):
        tau = 1.0 / n
        Mn = symbol(k, tau) ** n
        ds = 2 * np.pi / (N * dk)
        dens = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Mn))) * dk / (2 * np.pi)
        dens = dens.real
        tv_n = np.sum(np.abs(dens)) * ds
        mass = np.sum(dens) * ds        # = M(0)^n = 1 exactly; meaningful only when the density is O(1)
        print(f"  n = {n:5d} (tau = {tau:.4e}): sup_k|M|^n = {np.max(np.abs(Mn)):.3e},"
              f"  ||mu^(*n)||_TV = {tv_n:.4e}" + (f",  mass = {mass:.6f}" if tv_n < 10 else "") +
              f",  IS second moment ||mu||_TV^(2n) = 10^{float(2 * n * mp.log10(tv_exact(P6, tau))):.1f}")

    print("== 4. lambda_c <= omega_bar at the Study II critical point ==")
    lam_c, tau_s, k_s = 73.987479, 0.014482, 21.9646
    Pf = sp.lambdify(y, poly(P6), "mpmath")
    Phat = mp.quad(lambda t: Pf(t) * mp.cos(k_s * tau_s ** 0.25 * t), [-1, 1])
    M = mp.sin(k_s * mp.sqrt(6 * tau_s)) / (k_s * mp.sqrt(6 * tau_s)) + tau_s * Phat
    print(f"  |M_tau*(k*)| = {mp.nstr(abs(M), 8)},  log|M|/tau* = {mp.nstr(mp.log(abs(M)) / tau_s, 8)}"
          f"  (Study II: lambda_c = {lam_c})")
    print(f"  ||mu_tau*||_TV = {mp.nstr(tv_exact(P6, tau_s), 8)} >= |M|;"
          f"  omega_bar / lambda_c = {mp.nstr(omega6 / lam_c, 5)}")

    print("== 5. tangency of the positive kernels ==")
    tau, k = sp.symbols("tau k", positive=True)
    sinc = lambda z: sp.sin(z) / z
    a, b = 6 - 2 * sp.sqrt(6), 6 + 2 * sp.sqrt(6)
    tangency("MIX2 (1/2 U[-sqrt(a tau), sqrt(a tau)] + 1/2 U[-sqrt(b tau), sqrt(b tau)], a,b = 6 -/+ 2 sqrt 6)",
             (sinc(k * sp.sqrt(a * tau)) + sinc(k * sp.sqrt(b * tau))) / 2)
    v1, v2 = 10 - 2 * sp.sqrt(10), 10 + 2 * sp.sqrt(10)
    w1, w2 = sp.Rational(1, 2) + sp.sqrt(10) / 10, sp.Rational(1, 2) - sp.sqrt(10) / 10
    tangency("GM2  (w1 U[.., sqrt(v1 tau)] + w2 U[.., sqrt(v2 tau)], v = 10 -/+ 2 sqrt 10, w = 1/2 +/- sqrt10/10)",
             w1 * sinc(k * sp.sqrt(v1 * tau)) + w2 * sinc(k * sp.sqrt(v2 * tau)))
    tangency("RV6  (Study II symbol)",
             sinc(k * sp.sqrt(6 * tau)) + tau * sp.integrate(poly(P6) * sp.cos(k * tau ** sp.Rational(1, 4) * y), (y, -1, 1)))
    print("== done ==")
