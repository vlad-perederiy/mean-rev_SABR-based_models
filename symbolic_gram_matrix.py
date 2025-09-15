"""
Efficient symbolic integration for squared polynomials of x and exp(x)
"""

import sympy as sp
from functools import lru_cache


def antiderivative_square_polyexp(f, x):
    """
    indefinite integral of f wrt x
    """
    # !!! AI/GPT-generated then manually reviewed

    #print(" step1...", end="")
    f = sp.expand(f)
    #print (x, "", f)
    parts = decompose_polyexp_cancel_den_exps(f, x)   # the fixed decomposition you’re using
    #print("\b" * 9, end="")

    groups = {}  # slope S -> polynomial in x (EX coeffs)
    items = list(parts.items())
    #print(" STEP2...")

    for i, (S, P) in enumerate(items):
        #print("entered", len(items))
        #prg = ".D." + str(i + 1) + "/" + str(len(items)) + "..."
        #print(f"{prg:<15}", end="")

        for T, Q in items:
            ST = sp.simplify(S + T)
            for p, ap in P.items():
                for q, bq in Q.items():
                    groups[ST] = groups.get(ST, 0) + sp.simplify(ap*bq) * I_polyexp_antideriv(p+q, ST, x)
        #print("\b" * 15, end="")

    # Optional: group & Horner only polynomials in x inside exp(ST*x)
    res = 0
    #print(" step3...", end="")
    for ST, expr in groups.items():
        # expr is already e^{ST x} * poly(x) (or pure poly if ST==0); just tidy:
        if ST == 0:
            res += sp.horner(sp.together(expr), x)
        else:
            # factor out the exponential for readability
            res += sp.exp(ST*x) * sp.horner(sp.simplify(sp.together(expr/ sp.exp(ST*x))), x)
    #print("\b" * 9, end="")
    # Final light cleanup
    #print ("cleanup", res)
    res = sp.together(res)
    #print("cleanup", res)
    res = sp.powsimp(res)  # no force
    #print("cleanup", res)
    return res  # (+ arbitrary constant if you like)



def integrate_square_polyexp_grouped(f, x, T):
    """
    definite integral of f wrt x over 0 - T
    """
    # !!! AI/GPT-generated then manually reviewed

    #print (" step1...", end="")
    f = sp.expand(f)
    parts = decompose_polyexp_cancel_den_exps(f, x)
    #print ("\b"*9, end="")

    # Accumulate by slope: res = C0 + sum_S exp(S*T) * PS(T)
    groups = {}     # S -> poly in T (EX coefficients)
    C0 = 0
    #print (" step2...", end="")
    items = list(parts.items())
    for S, P in items:
        for U, Q in items:
            SU = sp.simplify(S + U)
            for p, ap in P.items():
                for q, bq in Q.items():
                    polyT, const = J_parts(p + q, SU, T)
                    coeff = sp.simplify(ap*bq)
                    if polyT != 0:
                        groups[SU] = groups.get(SU, 0) + coeff*polyT
                    if const != 0:
                        C0 += coeff*const
    #print ("\b"*9, end="")

    #print (" step3...", end="")
    # Build expression without ever calling Poly on something non-polynomial in T
    res = sp.simplify(C0)
    for S, poly in groups.items():
        # This 'poly' is guaranteed polynomial in T -> horner is safe here
        poly_h = sp.horner(sp.together(poly), T)
        if S == 0:
            res += poly_h
        else:
            res += sp.exp(S*T) * poly_h
    #print ("\b"*9, end="")

    # Final light cleanup (avoid moving exp(...) into denominators)
    res = sp.together(res)
    # IMPORTANT: avoid powsimp(..., force=True) which can push exp(...) into denom
    res = sp.powsimp(res)         # no force
    return res


######################################
#  helpers
# !!! AI/GPT-generated then manually reviewed

def I_polyexp_antideriv(n, S, x):
    S = sp.simplify(S)
    if S == 0:
        return x**(n+1)/(n+1)
    k = sp.symbols('k', integer=True, nonnegative=True)
    polyX = sp.summation(
        (-1)**k * sp.factorial(n) / (sp.factorial(n-k) * S**(k+1)) * x**(n-k),
        (k, 0, n)
    )
    return sp.exp(S*x) * sp.simplify(polyX)


@lru_cache(maxsize=None)
def J_parts(n, S, T):
    """Return the decomposition of ∫_0^T x^n e^{Sx} dx as (polyT, const)
       where polyT is polynomial in T, and const is T-free."""
    S = sp.simplify(S)
    if S == 0:
        return (T**(n+1)/(n+1), 0)
    k = sp.symbols('k', integer=True, nonnegative=True)
    polyT = sp.summation(
        (-1)**k * sp.factorial(n)/sp.factorial(n-k) * T**(n-k) / S**(k+1),
        (k, 0, n)
    )
    const = -(-1)**n * sp.factorial(n) / S**(n+1)
    return (sp.simplify(polyT), sp.simplify(const))


def decompose_polyexp_cancel_den_exps(f, x):
    """
    Decompose f(x) as sum_S ( sum_p coeff[S][p] * x**p ) * exp(S*x),
    allowing arbitrary (x-free) coefficients (domain=EX).
    """

    parts = {}
    adds = sp.Add.make_args(f)

    for i, term in enumerate(adds):
        prg = ".D." + str(i + 1) + "/" + str(len(adds)) + "..."
        #print(f"{prg:<15}", end="")

        # 1) normalize a single term
        num, den = sp.fraction(sp.together(term))
        num_res, sN = _remove_linear_exp_factors(num, x)
        den_res, sD = _remove_linear_exp_factors(den, x)
        S = sp.simplify(sN - sD)

        # 2) after removing exp(s*x), the core must have x-free denominator
        core = sp.simplify(num_res/den_res)
        num2, den2 = sp.fraction(sp.together(core))

        # last-ditch: if a leftover exp(s*x) got buried by simplify, try to pull it
        if x in den2.free_symbols:
            den2_fact = sp.factor(den2)
            den2_noexp, s_extra = _remove_linear_exp_factors(den2_fact, x)
            if s_extra != 0:
                den2 = sp.simplify(den2_noexp)
                S += s_extra

        if x in den2.free_symbols:
            raise ValueError("Denominator still depends on x (not only via exp(s*x)).")

        # 3) polynomial in x with EX coefficients
        P = sp.Poly(num2, x, domain=sp.EX)
        for (deg,), coeff in P.terms():
            parts.setdefault(S, {})
            parts[S][deg] = parts[S].get(deg, 0) + coeff/den2

        #print("\b" * 15, end="")
    return parts



def _remove_linear_exp_factors(expr, x):
    """Return (residual_without_exp, slope_sum) where we strip factors exp(s*x) with g(x)=s*x+b."""
    slope = sp.Integer(0)
    residual = sp.Integer(1)
    for fac in sp.Mul.make_args(expr):
        if fac.func is sp.exp and fac.args:
            s = _linear_slope_in_x(fac.args[0], x)
            if s is not None:
                slope += s
                continue
        residual *= fac
    return sp.simplify(residual), sp.simplify(slope)



def _linear_slope_in_x(expr, x):
    """
    If expr == a*x + b with no x in denominators, return a; else None.
    """
    g = sp.simplify(expr)
    num, den = sp.fraction(sp.together(g))
    if x in den.free_symbols:
        return None
    if x not in num.free_symbols:
        return sp.Integer(0)
    s = sp.simplify(sp.diff(g, x))
    # g - s*x must be x-free
    if x in sp.simplify(g - s*x).free_symbols:
        return None
    return s
