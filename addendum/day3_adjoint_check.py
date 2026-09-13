"""
Study III / literature gate (2026-09-07), day 3.
Check of the DUAL short-time expansion for the two-window scheme of
note_positive_kernels_2026-09-07.md §4 (Bally–Rey hypothesis E*_n(h,q),
eq. (2.6) of EJP 21 (2016) no. 12; see t1/literature_gate_bally_rey_2026-09-07.md §5):

    S(h)^* g = g + h H^* g + (h^2/2) (H^*)^2 g + O(h^3),
    H^* g = (a g)'' - (b g)' + c g   (c = 0 unless --c).

S(h)^* g(y) = sum_i int_{-1}^{1} (du/2) w_i(X_i) g(X_i) dX_i/dy,  X_i(y,u) solving
X + c_i(X) + r_i(X) u = y,   dX/dy = 1/(1 + c_i'(X) + r_i'(X) u).

The identity is polynomial in the Taylor data (a^{(k)}, b^{(k)}, c^{(k)}, g^{(k)})(y),
so it is checked with EXACT rational random Taylor data at several points
(Schwartz–Zippel); s = sqrt(h), u and sv = sqrt(v_i) are kept symbolic and all
products are truncated at s^ORDER incrementally.  Odd powers of s must vanish.
"""
import sympy as sp
import sys, time, random

ORDER = 6                      # keep s^0..s^6  (h^3 remainder)
WITH_C = "--c" in sys.argv
def _arg(name, default):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
NTRIALS = _arg("--ntrials", 3)
MAXK = _arg("--maxk", 10**6)    # zero out Taylor data of order > MAXK (probe of remainder's regularity demand)
s, u, sv = sp.symbols("s u sv", positive=True)
K = ORDER + 4

def trunc(expr, order=ORDER):
    expr = sp.expand(expr)
    if expr == 0:
        return sp.Integer(0)
    P = sp.Poly(expr, s)
    return sp.expand(sum(coef * s**k for (k,), coef in P.terms() if k <= order))

def tmul(x, y):
    return trunc(x * y)

def tpowers(x, kmax):
    """[1, x, x^2, ..., x^kmax] with truncation after each multiplication."""
    out = [sp.Integer(1)]
    for _ in range(kmax):
        out.append(tmul(out[-1], x))
    return out

def taylor(coeffs, epow, shift=0):
    return trunc(sum(coeffs[k + shift] * epow[k] / sp.factorial(k) for k in range(ORDER + 1)))

def series_1p(wpow, exponent):
    """(1+w)^exponent truncated, given truncated powers of w."""
    return trunc(sum(sp.binomial(exponent, k) * wpow[k] for k in range(ORDER + 1)))

def window(sign, a, b, c, g, alpha):
    """u-averaged  w_i(X) g(X) dX/dy  for window i (sign=-1: i=1, +1: i=2); a[0] = alpha^2."""
    def coeff_funcs(epow):
        a_, a1, a2, a3 = (taylor(a, epow, k) for k in range(4))
        b_, b1, b2, b3 = (taylor(b, epow, k) for k in range(4))
        c1_, c2_ = ((taylor(c, epow, k) if WITH_C else sp.Integer(0)) for k in (1, 2))
        ci1 = trunc(b_ + sign * sp.sqrt(6) / 2 * a1)
        ci2 = trunc(tmul(a_, b2) / 2 + tmul(b_, b1) / 2 + tmul(a_, c1_))
        rho = trunc(3 * tmul(a_, a2) + 6 * tmul(a_, b1) + 3 * tmul(b_, a1) - sp.Rational(9, 2) * tmul(a1, a1))
        dci1 = trunc(b1 + sign * sp.sqrt(6) / 2 * a2)
        dci2 = trunc((tmul(a1, b2) + tmul(a_, b3)) / 2 + (tmul(b1, b1) + tmul(b_, b2)) / 2
                     + tmul(a1, c1_) + tmul(a_, c2_))
        drho = trunc(3 * (tmul(a1, a2) + tmul(a_, a3)) + 6 * (tmul(a1, b1) + tmul(a_, b2))
                     + 3 * (tmul(b1, a1) + tmul(b_, a2)) - 9 * tmul(a1, a2))
        return a_, a1, b_, ci1, ci2, rho, dci1, dci2, drho
    eps = sp.Integer(0)
    for _ in range(ORDER + 2):
        epow = tpowers(eps, ORDER)
        a_, a1, b_, ci1, ci2, rho, *_ = coeff_funcs(epow)
        w = trunc((a_ - a[0]) / alpha**2 + rho * s**2 / (alpha**2 * sv**2))
        wpow = tpowers(w, ORDER)
        r_over_s = alpha * sv * series_1p(wpow, sp.Rational(1, 2))
        eps_new = trunc(-(s**2 * ci1 + s**4 * ci2 + s * u * r_over_s))
        if sp.expand(eps_new - eps) == 0:
            break
        eps = eps_new
    epow = tpowers(eps, ORDER)
    a_, a1, b_, ci1, ci2, rho, dci1, dci2, drho = coeff_funcs(epow)
    w = trunc((a_ - a[0]) / alpha**2 + rho * s**2 / (alpha**2 * sv**2))
    wpow = tpowers(w, ORDER)
    inv_sqrt = series_1p(wpow, -sp.Rational(1, 2))
    # r_i' = s (a' v + rho' s^2) / (2 alpha sv sqrt(1+w))
    dr = trunc(s * tmul(a1 * sv**2 + drho * s**2, inv_sqrt) / (2 * alpha * sv))
    z = trunc(s**2 * dci1 + s**4 * dci2 + u * dr)       # c_i'(X) + r_i'(X) u = O(s)
    zpow = tpowers(z, ORDER)
    dXdy = trunc(sum((-1)**k * zpow[k] for k in range(ORDER + 1)))
    if WITH_C:
        c_, c1_, c2_ = (taylor(c, epow, k) for k in range(3))
        wi = trunc(sp.Rational(1, 2) + c_ / 2 * s**2
                   + sp.Rational(1, 4) * (tmul(c_, c_) + tmul(a_, c2_) + tmul(b_, c1_)) * s**4)
    else:
        wi = sp.Rational(1, 2)
    integrand = tmul(tmul(wi, taylor(g, epow, 0)), dXdy)
    P = sp.Poly(integrand, u)
    avg = sum(coef * (sp.Rational(1, k + 1) if k % 2 == 0 else 0) for (k,), coef in P.terms())
    return sp.expand(avg)

def target_coeffs(a, b, c, g):
    y = sp.Symbol("y")
    af, bf, cf, gf = (sp.Function(n)(y) for n in "abcg")
    Hs = lambda G: sp.diff(af * G, y, 2) - sp.diff(bf * G, y) + (cf * G if WITH_C else 0)
    def ev(expr):
        expr = expr.doit()
        for k in range(K - 1, 0, -1):
            for fam, f in ((a, af), (b, bf), (c, cf), (g, gf)):
                expr = expr.subs(sp.Derivative(f, (y, k)), fam[k])
        for fam, f in ((a, af), (b, bf), (c, cf), (g, gf)):
            expr = expr.subs(f, fam[0])
        return sp.expand(expr)
    return {0: ev(gf), 2: ev(Hs(gf)), 4: ev(Hs(Hs(gf)) / 2)}

random.seed(20260907)
def rnd():
    return sp.Rational(random.randint(-9, 9), random.randint(1, 5))

all_ok = True
t0 = time.time()
for trial in range(NTRIALS):
    alpha = sp.Rational(random.randint(1, 4), random.randint(1, 3))       # a(y) = alpha^2 > 0
    a = [alpha**2] + [rnd() for _ in range(K - 1)]
    b = [rnd() for _ in range(K)]
    c = [rnd() for _ in range(K)] if WITH_C else [sp.Integer(0)] * K
    g = [rnd() for _ in range(K)]
    if MAXK < K:
        for fam in (a, b, c, g):
            for k in range(MAXK + 1, K):
                fam[k] = sp.Integer(0)
    tot = sp.Integer(0)
    for sign, v in ((-1, 6 - 2 * sp.sqrt(6)), (+1, 6 + 2 * sp.sqrt(6))):
        W = window(sign, a, b, c, g, alpha)
        # after u-averaging only even powers of sv survive -> substitute sv^2 = v
        Wp = sp.Poly(W, sv)
        assert all(k % 2 == 0 for (k,) in Wp.monoms()), "odd power of sqrt(v) survived"
        tot += sum(coef * v**(k // 2) for (k,), coef in Wp.terms())
    tot = sp.expand(tot)
    P = sp.Poly(tot, s)
    coeffs = {k: sp.radsimp(sp.expand(coef)) for (k,), coef in P.terms()}
    tgt = target_coeffs(a, b, c, g)
    ok, report = True, []
    for k in range(ORDER + 1):
        ck = sp.expand(coeffs.get(k, 0))
        if k in tgt:
            d = sp.simplify(ck - tgt[k]); ok &= (d == 0)
            report.append(f"s^{k}:{'OK' if d == 0 else 'MISMATCH '+str(d)}")
        elif k % 2 == 1:
            d = sp.simplify(ck); ok &= (d == 0)
            report.append(f"s^{k}:{'0' if d == 0 else 'NONZERO '+str(d)}")
        else:
            report.append(f"s^{k}:rem={sp.N(ck, 6)}")
    all_ok &= ok
    print(f"trial {trial}: alpha={alpha}  " + "  ".join(report) + f"  [{time.time()-t0:.1f}s]", flush=True)
print("WITH_C =", WITH_C, "| PASS" if all_ok else "| FAIL")
