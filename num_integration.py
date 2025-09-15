from functools import lru_cache
import sympy as sp
import scipy.integrate as spi  # keep the alias to avoid shadowing
import mpmath
import math

LIB_MATH = "math"

 
def num_integration_single (expr):
    """
    simple numerical integration for a single integral
    expr is here sympy Integral object (no free parameters !!!) with specified numeric limits (definite)
    """
    integrand = expr.function
    (u, lower_val, upper_val) = expr.limits[0]
    # Convert the integrand into a Python function using mpmath
    f = sp.lambdify(u, integrand, LIB_MATH)
    # Perform the numerical integration
    return mpmath.quad (f, [lower_val, upper_val])




def num_integration_double (nested_integral_expr):
    """
     numerical integration of  a double/nested integral (nested_integral_expr) of the sympy form:
        Integral(
            f(x,y),
            (y, e1_expr(x), e2_expr(x)),  # inner integral (w.r.t y)
            (x, x1_val, x2_val)           # outer integral (w.r.t x)
        )
    where f(x,y) has no extra symbolic parameters,
    outer limits x1_val and x2_val are numeric,
    and inner limits e1_expr(x), e2_expr(x) may be expressions/functions in the outer variable x.
    no free parameters !!!
    """

    # We expect exactly 2 sets of limits in the order: (y, e1, e2), (x, x1, x2).
    (y_sym, y1_expr, y2_expr) = nested_integral_expr.limits[0] # inner
    (x_sym, x1_val, x2_val)   = nested_integral_expr.limits[1] # outer

    # The integrand f(x, y)
    f_expr = nested_integral_expr.function

    # Convert sympy expressions to Python callables via lambdify
    # e1_func(x) and e2_func(x) are the y-limits, possibly depending on x
    y1_func = sp.lambdify(x_sym, y1_expr, LIB_MATH)
    y2_func = sp.lambdify(x_sym, y2_expr, LIB_MATH)

    # f_func(x, y) is the integrand
    f_func = sp.lambdify((x_sym, y_sym), f_expr, LIB_MATH)

    # Define the inner integral as a function of x
    def inner_integral(x_val):
        # Evaluate numeric limits for y
        lower_y = y1_func(x_val)
        upper_y = y2_func(x_val)
        # Integrate w.r.t y
        return mpmath.quad(lambda y_val: f_func(x_val, y_val),
                           [lower_y, upper_y])

    # Now integrate the inner integral w.r.t x over [x1_val, x2_val]
    result = mpmath.quad(inner_integral, [x1_val, x2_val])
    return result


def num_integration_double2 (nested_integral_expr):
    """
    version of num_integration_double using scipy  (somewhat faster)
    """

    # Unpack limits (we expect exactly two, inner then outer, as stated)
    (y_sym, y1_expr, y2_expr) = nested_integral_expr.limits[0]  # inner
    (x_sym, x1_val, x2_val)   = nested_integral_expr.limits[1]  # outer

    # Integrand
    f_expr = nested_integral_expr.function

    # Lambdify expressions
    y1_func = sp.lambdify(x_sym, y1_expr, LIB_MATH)                 # y lower bound as function of x
    y2_func = sp.lambdify(x_sym, y2_expr, LIB_MATH)                 # y upper bound as function of x
    f_func  = sp.lambdify((x_sym, y_sym), f_expr, LIB_MATH)         # f(x, y)

    # nquad expects the integrand with the innermost variable first: here y then x.
    def integrand(y, x):
        return f_func(x, y)

    # Bounds for y depend on x; bounds for x are constants
    def limits_y(x):
        return (y1_func(x), y2_func(x))

    x1 = float(sp.N(x1_val))
    x2 = float(sp.N(x2_val))
    ranges = [limits_y, [x1, x2]]

    res, err = spi.nquad(integrand, ranges)
    return res




def num_integration_recursive_single_double (expr, useScipy=False):
    """
    Recursively evaluates (numerically) a sympy expression which may contain integrals
    Can go up to double (incl nested) level integrals
    Cannot handle free parameters !!!
    """
    if isinstance(expr, sp.Integral):
        # Found an integral, evaluate it with the custom function
        lim = expr.limits
        assert (len(lim) <= 2)
        if len(lim) == 1:
            return num_integration_single(expr)
        if len(lim) == 2:
            if useScipy:
                return num_integration_double2(expr) # use scipy
            else:
                return num_integration_double(expr)
    elif expr.args:
        # Recursively process subexpressions
        new_args = tuple(num_integration_recursive_single_double (arg) for arg in expr.args)
        return expr.func(*new_args)
    else:
        # Base case: atomic expression
        return expr



def num_integration_simpson (expr, useScipy=False, n=20):
    """
    expects one single Integral object in the expression at the top level
    converts it to a function and evaluates via Simpson' rule
    Cannot handle free parameters !!!
    """
    integrand = expr.function
    (u, lower_val, upper_val) = expr.limits[0]

    # define a function which evaluates the expression depending on x
    def outer_function (x):
        I_this = integrand.subs(u, x)
        return num_integration_recursive_single_double (I_this, useScipy)  # can handle nested, double integrals etc

    # standard Simpson for the function
    val = _integrate_function_simpson (outer_function, lower_val, upper_val, n)
    return val

def _integrate_function_simpson (f, a, b, n=100):
    """
    numerical Simpons's integration for a function f over interval [a, b] via n subintervals
    """
    if n % 2 == 1:
        raise ValueError("n must be even")
    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    result = f(x[0]) + f(x[-1])
    for i in range(1, n):
        if i % 2 == 0:
            result += 2 * f(x[i])
        else:
            result += 4 * f(x[i])
    return result * h / 3

def _integrate_function_simpson2  (f, a: float, b: float, n: int):
    """
    alternative version
    numerical Simpons's integration for a function f over interval [a, b] via n subintervals
    """
    # alternative version
    # numerical Simpons's integration for a function f over interval [a, b] via n subintervals
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule.")

    h = (b - a) / n
    # Odd and even interior sums
    s_odd = sum(f(a + (2*k - 1) * h) for k in range(1, n // 2 + 1))
    s_even = sum(f(a + 2*k * h) for k in range(1, n // 2))
    return (h / 3) * (f(a) + f(b) + 4 * s_odd + 2 * s_even)





################################################################################################################################


def num_integration_recursive_unlimited (expr, opts=None, _fixed=None):
    """
    AI/GPT-generated function for numerical evaluation of sympy expressions which might contain multiple definite Integral objects
    Here,  expressions !!! should contain no parameters !!! except for integration variables
    Handles nested integrals of unlimited depth (via recursive calls) incl where inner-integral limits depend on outer-integral integration variables
    Seems to work fine for 3-4 levels, but sometimes takes longer to evaluate (up to about 10 secs)

    An extension to free parameters would be nice !!! also, unclear if lambbdifications can be optimized
    possibly, an approach similar to build_nested_quad_from_Raw below would be possible :
    call expects just sympy expression and returns a lambdified function
    parameters are then fed in via calls to the lambdified function
    """
    if _fixed is None:
        _fixed = {}
    return _eval_expr_recursive(expr, _fixed, LIB_MATH, opts)


def _eval_expr_recursive(e, fixed, modules, opts):
    # Case 1: handle a single Integral node directly (build nquad, recurse into its function)
    if isinstance(e, sp.Integral):
        limits = list(e.limits)  # e.g. [(v,u,T), (u,0,T), (T,0,3)]  inner -> outer
        vars_inner_to_outer = [lim[0] for lim in limits]

        bound_funcs = []
        for i, (v, a_expr, b_expr) in enumerate(limits):
            # bounds for variable i may depend on outer vars i+1...
            outer_syms = [lim[0] for lim in limits[i + 1:]]
            lower_fun = _lambdify_with_fixed(a_expr, outer_syms, fixed, modules)
            upper_fun = _lambdify_with_fixed(b_expr, outer_syms, fixed, modules)

            def make_bounds(lf=lower_fun, uf=upper_fun):
                return lambda *outers: (float(lf(*outers)), float(uf(*outers)))

            bound_funcs.append(make_bounds())

        # nquad integrand: args come inner-first
        def integrand(*vals_inner_first):
            local_fixed = dict(fixed)
            for s, val in zip(vars_inner_to_outer, vals_inner_first):
                local_fixed[s] = float(val)
            # the integrand may itself contain Integrals (powers, sums, etc.)
            return _eval_expr_recursive(e.function, local_fixed, modules, opts)

        res, _err = spi.nquad(integrand, bound_funcs, opts=opts)
        return float(res)

    # Case 2: not an Integral — if it *contains* integrals, only evaluate the OUTERMOST ones
    outers = _outermost_integrals(e)
    if outers:
        replaced = e
        for subI in outers:
            val = _eval_expr_recursive(subI, fixed, modules, opts)
            replaced = replaced.xreplace({subI: sp.Float(val)})
        return _eval_numeric_expr(replaced, fixed, modules)

    # Case 3: plain numeric expression
    return _eval_numeric_expr(e, fixed, modules)


def _outermost_integrals(expr):
    """Return Integrals in expr that are not inside another Integral."""
    atoms = list(expr.atoms(sp.Integral))
    if not atoms:
        return []
    inner_of_any = set()
    for I in atoms:
        inner_of_any |= I.function.atoms(sp.Integral)
    outers = [I for I in atoms if I not in inner_of_any]
    return sorted(outers, key=sp.default_sort_key)


@lru_cache(maxsize=2048)
def _cached_lambdify(expr, arg_syms, modules):
    return sp.lambdify(arg_syms, expr, modules)


def _lambdify_with_fixed(expr, arg_syms, fixed, modules):
    # substitute any already-fixed outer vars (e.g., T) before lambdify
    if fixed:
        expr = expr.xreplace({s: sp.Float(v) for s, v in fixed.items() if s in expr.free_symbols})
    arg_syms = tuple(arg_syms)  # hashable for cache
    return _cached_lambdify(expr, arg_syms, modules)


def _eval_numeric_expr(expr, fixed, modules):
    # evaluate a non-Integral expression numerically with given fixed values
    free_syms_in_fixed = tuple(s for s in sorted(expr.free_symbols, key=sp.default_sort_key) if s in fixed)
    unresolved = expr.free_symbols.difference(free_syms_in_fixed)
    if unresolved:
        maybe_num = expr.xreplace({s: sp.Float(fixed[s]) for s in free_syms_in_fixed})
        if maybe_num.free_symbols:
            raise ValueError(
                f"Unresolved symbols {sorted(map(str, maybe_num.free_symbols))} in '{expr}'. "
                "Make sure all outer variables are bound when this part is evaluated."
            )
        return float(maybe_num)
    fn = _cached_lambdify(expr, free_syms_in_fixed, modules)
    vals = [float(fixed[s]) for s in free_syms_in_fixed]
    return float(fn(*vals) if free_syms_in_fixed else fn())


###############################################################################################################################
# versions with parameters
################################################################################################################################


def build_nested_quad_from_Raw (rawInt, params, modules="math"):
    """
    Returns a callable eval(*param_vals, epsabs=..., epsrel=..., limit=...)
    that numerically evaluates the nested integral using scipy.integrate.quad.

    AI/GPT-generated
    works for double / nested, !!! allows free parameters !!!, quite fast due to lambdifications
    expects one nested Integral, with function under the  inner

    returns a function expecting free parameters
    """
    F_expr=rawInt.function
    U, Ulo_expr, Uhi_expr = rawInt.limits[0]
    V, Vlo_expr, Vhi_expr  = rawInt.limits[1]

    return build_nested_quad_from_exprs(F_expr,
    U, Ulo_expr, Uhi_expr,
    V, Vlo_expr, Vhi_expr,
    params,
    modules,
    )


def build_nested_quad_from_exprs(
    F_expr,                 # SymPy expr in (u, v, *params)
    u, Ulo_expr, Uhi_expr,  # inner var 'u' and its SymPy limits Ulo(v,params), Uhi(v,params)
    v, Vlo_expr, Vhi_expr,  # outer var 'v' and its SymPy limits Vlo(params), Vhi(params)
    params,                 # tuple/list of SymPy symbols for parameters
    modules,         # backend for lambdify; "math" is good for scalars
):

    # Lambdify everything ONCE (scalar-friendly)
    F_num   = sp.lambdify((u, v, *params), F_expr, modules)
    Ulo_num = sp.lambdify((v, *params), Ulo_expr, modules)
    Uhi_num = sp.lambdify((v, *params), Uhi_expr, modules)
    Vlo_num = sp.lambdify((*params,),   Vlo_expr, modules)
    Vhi_num = sp.lambdify((*params,),   Vhi_expr, modules)

    def _as_float(x):
        if x is None:
            raise RuntimeError("A limit function returned None.")
        # Accept Python/NumPy scalars
        if isinstance(x, complex):
            if abs(x.imag) > 1e-12:
                raise RuntimeError(f"Complex bound not supported: {x}")
            x = x.real
        return float(x)

    def _wrap_integrand(uu, vv, *pv):
        y = F_num(uu, vv, *pv)
        if y is None:
            raise RuntimeError("Integrand returned None (unmapped function?).")
        if isinstance(y, complex):
            if abs(y.imag) > 1e-12:
                raise RuntimeError(f"Complex integrand value: {y}")
            y = y.real
        return float(y)

    def eval_nested(*param_vals, epsabs=1e-10, epsrel=1e-10, limit=200):
        # Outer bounds (depend only on params)
        Vlo = _as_float(Vlo_num(*param_vals))
        Vhi = _as_float(Vhi_num(*param_vals))
        # Allow reversed finite bounds by flipping sign
        outer_sign = 1.0
        if (math.isfinite(Vlo) and math.isfinite(Vhi)) and Vlo > Vhi:
            Vlo, Vhi = Vhi, Vlo
            outer_sign = -1.0

        def _inner_at_v(vv, *pv):
            # Inner bounds for this vv
            Ulo = _as_float(Ulo_num(vv, *pv))
            Uhi = _as_float(Uhi_num(vv, *pv))
            inner_sign = 1.0
            if (math.isfinite(Ulo) and math.isfinite(Uhi)) and Ulo > Uhi:
                Ulo, Uhi = Uhi, Ulo
                inner_sign = -1.0

            # Integrate in u for this vv
            g = lambda uu, Vfixed, *Pfixed: _wrap_integrand(uu, Vfixed, *Pfixed)
            val_u, _err_u = spi.quad(
                g, Ulo, Uhi,
                args=(vv, *pv),
                epsabs=epsabs, epsrel=epsrel, limit=limit
            )
            return inner_sign * val_u

        # Now integrate over v, passing params via args
        val_v, err_v = spi.quad(
            lambda vv, *pv: _inner_at_v(vv, *pv),
            Vlo, Vhi,
            args=param_vals,
            epsabs=epsabs, epsrel=epsrel, limit=limit
        )
        return outer_sign * val_v, err_v, (Vlo, Vhi)

    return eval_nested


###############################################################################
# simplified / cleansed version of  the above
def build_nested_quad_from_Raw2 (rawInt, params, modules="math",  epsabs=1e-10, epsrel=1e-10, limit=200, fn_inner= lambda x: x ):
    # params is the tuple of free paremeters as sympy symbols
    F_expr=rawInt.function
    u, Ulo_expr, Uhi_expr = rawInt.limits[0]
    v, Vlo_expr, Vhi_expr  = rawInt.limits[1]
    # Lambdify everything ONCE (scalar-friendly)
    F_num   = sp.lambdify((u, v, *params), F_expr, modules)
    Ulo_num = sp.lambdify((v, *params), Ulo_expr, modules)
    Uhi_num = sp.lambdify((v, *params), Uhi_expr, modules)
    Vlo_num = sp.lambdify((*params,),   Vlo_expr, modules)
    Vhi_num = sp.lambdify((*params,),   Vhi_expr, modules)
    def eval_nested(*param_vals): # outer over v
        # Outer bounds (depend only on params)
        Vlo = Vlo_num(*param_vals)
        Vhi = Vhi_num(*param_vals)

        def _inner_at_v(vv, *pv): # inner over u
            # Inner bounds for this vv
            Ulo = Ulo_num(vv, *pv)
            Uhi = Uhi_num(vv, *pv)
            g= F_num
            val_u, _err_u = spi.quad(
                g, Ulo, Uhi,
                args=(vv, *pv),
                epsabs=epsabs, epsrel=epsrel, limit=limit
            )
            return val_u
        # Now integrate over v, passing params via args
        val_v, err_v = spi.quad(
            lambda vv, *pv: fn_inner(_inner_at_v(vv, *pv)),
            Vlo, Vhi,
            args=param_vals,
            epsabs=epsabs, epsrel=epsrel, limit=limit
        )
        return  val_v, err_v
    return eval_nested


# dito extended to three levels
def build_nested_quad_from_Raw3(rawInt, params, modules="math", epsabs=1e-10, epsrel=1e-10, limit=200):
    # params is a tuple of free parameters as sympy symbols
    F_expr = rawInt.function
    u, Ulo_expr, Uhi_expr = rawInt.limits[0]  # innermost
    v, Vlo_expr, Vhi_expr = rawInt.limits[1]  # middle
    w, Wlo_expr, Whi_expr = rawInt.limits[2]  # outermost

    # Lambdify once
    F_num   = sp.lambdify((u, v, w, *params), F_expr, modules)
    Ulo_num = sp.lambdify((v, w, *params),    Ulo_expr, modules)
    Uhi_num = sp.lambdify((v, w, *params),    Uhi_expr, modules)
    Vlo_num = sp.lambdify((w, *params),       Vlo_expr, modules)
    Vhi_num = sp.lambdify((w, *params),       Vhi_expr, modules)
    Wlo_num = sp.lambdify((*params,),         Wlo_expr, modules)
    Whi_num = sp.lambdify((*params,),         Whi_expr, modules)

    def eval_nested(*param_vals):
        Wlo = Wlo_num(*param_vals)
        Whi = Whi_num(*param_vals)

        def _inner_at_vw(vv, ww, *pv):
            Ulo = Ulo_num(vv, ww, *pv)
            Uhi = Uhi_num(vv, ww, *pv)
            val_u, _ = spi.quad(
                F_num, Ulo, Uhi,
                args=(vv, ww, *pv),
                epsabs=epsabs, epsrel=epsrel, limit=limit
            )
            return val_u

        def _middle_at_w(ww, *pv):
            Vlo = Vlo_num(ww, *pv)
            Vhi = Vhi_num(ww, *pv)
            val_v, _ = spi.quad(
                lambda vv, ww2, *pv2: _inner_at_vw(vv, ww2, *pv2),
                Vlo, Vhi,
                args=(ww, *pv),
                epsabs=epsabs, epsrel=epsrel, limit=limit
            )
            return val_v

        val_w, err_w = spi.quad(
            lambda ww, *pv: _middle_at_w(ww, *pv),
            Wlo, Whi,
            args=param_vals,
            epsabs=epsabs, epsrel=epsrel, limit=limit
        )
        return val_w, err_w

    return eval_nested

# dito just one-level, for consistency
def build_nested_quad_from_Raw1 (rawInt, params, modules="math", epsabs=1e-10, epsrel=1e-10, limit=200):

    integrand = rawInt.limits[0][0]
    f_int = sp.lambdify([integrand] + params, rawInt.function, modules)
    f_lim1 = sp.lambdify(params, rawInt.limits[0][1], modules)
    f_lim2 = sp.lambdify(params, rawInt.limits[0][2], modules)

    def integral_value_simple(*args):
        val, err = spi.quad(f_int, f_lim1(*args), f_lim2(*args), args=args,    epsabs=epsabs, epsrel=epsrel, limit=limit)
        return val, err

    return integral_value_simple



##################################################################################################################
# tests
##################################################################################################################



if __name__ == "__main__":
    from sympy import *

    T = symbols("T", positive=True)
    u = symbols("u", positive=True)
    v = symbols("v", positive=True)
    z = symbols("z", positive=True)

    INT = Integral(0.05*Integral(0.00972710787278743*(-0.1*exp(8.82716378683761*T) + 0.1*exp(17.6543275736752*T - 8.82716378683761*u))*exp(8.82716378683761*u)*exp(-26.4814913605128*T + 8.82716378683761*u), (u, 0, T)) + Integral(-0.010412211621565*exp(-8.82716378683761*T + 8.82716378683761*u), (u, 0, T))**2 + 0.1*Integral(0.00325242452556761*exp(-8.82716378683761*T + 8.82716378683761*u), (v, u, T), (u, 0, T)), (T, 0, 3))
    INT2 = Integral(0.06*Integral(0.00972710783*(-0.11*exp(8.8271633761*T) + 0.13*exp(17.6545736752*T - 8.8271683761*u))*exp(8.8278683761*u)*exp(-26.481405128*T + 8.8271683761*u), (u, 0, T)) + Integral(-0.01041221565*exp(-8.8271633761*T + 8.8271637861*u), (u, 0, T))**2 + 0.1*Integral(0.05242452556761*exp(-8.82716378761*T + 8.8271683761*u), (v, u, T), (u, 0, T)), (T, 0, 3))

    res = num_integration_recursive_unlimited (INT)
    print (res)


    res = num_integration_simpson (INT, useScipy=True, n=100)
    print (res)


    u, v, w, x = sp.symbols('u v w x', real=True)
    a, b = sp.symbols('a b', positive=True)
    INT = sp.Integral(    sp.exp(-a*(u + v)) * sp.sin(b*u*v),    (u, sp.Integer(0), v / (1 + a * v)), (v, sp.Integer(0), 2 / a))
    J = build_nested_quad_from_Raw (INT , (a, b))

    val, err, (vlo, vhi) = J (1.5, 0.75, epsabs=1e-10, epsrel=1e-10)
    print(f"Integral over v∈[{vlo}, {vhi}]  ->  {val:.12g}  (±{err:.2e})")


    INT = sp.Integral(sp.exp(-a * (u + v)) * sp.sin(b * u * v), (u, sp.Integer(0), v / (1 + a * v)), (v, sp.Integer(0), 2 / a))
    J = build_nested_quad_from_Raw2(INT, (a, b), epsabs=1e-8, epsrel=1e-8, limit=50)
    val, err = J(1.5, 0.75)
    print(f"Integral over v∈[{vlo}, {vhi}]  ->  {val:.12g}  (±{err:.2e})")


    F = sp.exp(-a*(u+v+w)) * sp.sin(b*u*v*w)

    INT = sp.Integral(
        F,
        (u, 0, v/(1 + a*v + w)),       # may depend on v,w
        (v, 0, w/(1 + a*w)),           # may depend on w
        (w, 0, 2/a)                    # depends only on params
    )

    J  = build_nested_quad_from_Raw3(INT, (a, b), epsabs=1e-8, epsrel=1e-8, limit=50)
    val, err = J(1.5, 0.75)
    print(f"{val:.12g}  (±{err:.2e})")