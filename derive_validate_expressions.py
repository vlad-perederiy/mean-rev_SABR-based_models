"""
The code derives closed-form solutions in the form of effective coefficients
(from which the standardized SABR parameters can be easily calculated).
For one code execution, *hSABR*, *mrSABR* and *mrZABR* coefficients are generated
(the latter - for the particular *mrZABR* constellation as set in the code beginning/settings).
The coefficients are saved as *sympy* symbolic expressions in the *pickle* files.
The code also rigorously verifies the results by matching to numeric integrations,
and by comparing the resulting expressions to the closed-formed expressions provided in the previous research
for the simpler special case of constant expected instantaneous volatility ("flat expected vola").
"""


################################################
# !!! tested and best run in PyCharm !!!
# !!! see SETTINGS here below !!!
################################################

import sympy as sp
from   sympy import *
import math
import time
import random
import pickle



# sympy symbols for model parameters
lam = symbols("lam", positive=True) # lambda
alp = symbols("alp", positive=True) # ampha
sig = symbols("sig", positive=True) # sigma
alps = symbols("alps", positive=True) # alpha squared
rho = symbols("rho", real=True) # rho
nu = symbols("nu", positive=True) # nu
the = symbols("the", positive=True) # theta
thes = symbols("thes", positive=True) # theta squared
Tex = symbols("T_ex", nonegative=True) # T_ex (option expiry)
gam = symbols ("gam", positive=True) # gamma (ZABR)
G0 = symbols("G0", real=True) # Γ₀ from backbone function: Γ₀ = -C'(F), for lognormal forward: C(F) = F, Γ₀ = -1
T = symbols("T", nonegative=True) # argument in hSABR/mrSABR interim integral functions
t = symbols ("t", positive=True) # argument in mrZABR interim integral functions




####################################################################################################################
## SETTINGS ##############################################################################################



#######################################
# mrZABR settings:
#######################################

# gamma-parameter to set the mrZABR model specification (CEV-parameter in vola process):

#mrZABR_GAMMA  = Rational (3, 2)   # gamma as a fixed CEV parameter, expressed as a fraction from interval (0, 2), e.g. Rational (3, 4) for 0.75 (float values lead to numeric problems)
#mrZABR_GAMMA  = Rational (4, 3)
#mrZABR_GAMMA  = Integer (1)       # 1: effectively reduces mrZABR to mrSABR, can be used to chcek mrSABR and mrZABR consistency (with mrZABR_EXACT set to True)
#mrZABR_GAMMA  = Rational (3, 4)
#mrZABR_GAMMA  = Rational (2, 3)
mrZABR_GAMMA  = Rational (1, 2)   # Rational (1, 2)  implements CIR-ZABR, with CIR process for the vola
#mrZABR_GAMMA  = Rational (1, 3)
#mrZABR_GAMMA  = Rational (1, 4)
#mrZABR_GAMMA  = gam              # gam: handling gamma as yet another parameter (symbolic variable) making it a 6-parameter model. Very slow/lengthy for mrZABR_ORDER>2

mrZABR_APPRX  = True  # ig True: approximate mrZABR integrands via Taylor / binomial expansion

# Expansion settings if mrZABR_APPRX=True:
# Integrands contain terms of the type:  [ theta + (alpha - theta) * exp(-lambda*T) ] ** power
# which need to be expanded/approximated to be integrated to closed form
#mrZABR_EXPNS  = 1  # treats alpha as the variable x and expands it around x0 = theta
#mrZABR_EXPNS  = 2  # treats theta as the variable x and expands it around x0 = alpha
mrZABR_EXPNS  = 3 # operates with (and returns) substituted variables: mu = (alpha + theta)/2 and delta = (alpha - theta) / (alpha + theta). Treats delta as the variable x and expands around x0 = 0
# 1 and 2 might result in  shorter expressions, but fail (expansion diverges) in marginal cases:
# 1 fails if alpha is higher than 2x theta (frequently observed for equities)
# 2 fails  if theta is higher than 2x alpha (rare but possible for equities)
# 3 works always (series converge with rising order)


# Expansion order, which sets approximation precision if mrZABR_APPRX=True
mrZABR_ORDER = 6
# 4, 5, 6 are reasonable options:
# 4 seems to produce minimally acceptable approximation quality, with expression lengths comparable to mrSABR
# 6 seems to result in quasi-exact approximations (but considerably lengthier expressions)
# precision tends to decrease with increasing gamma, especially gamma>1 (although still convergent with increasing order)


# exact inegration for mrZABR:
mrZABR_EXACT = False
# If True: try exact integration (without integrand expansions) for mrZABR
# If that succeeds, results will overwrite expanded/approximated results  (if mrZABR_APPRX  = True)
# But that succeeds actually only with mrZABR_GAMMA = 1 (for all integrals) which reduces ZABR to SABR
# Note: with mrZABR_GAMMA=1/2, variable substitution would help, in pirnciple, to integrate some of the ZABR integrals to closed form
# In particular via switching of integration variable x to  sqrt (alpT).subs (T, x)
# This was not implemented in this code, however:
# - the resulting expressions are numerically complicated involving logs, branches, degenerating at alpha=theta etc
# - some of the needed integrals do not integrate even with that substitution

#######################################
# end of mrZABR settings
#######################################



###########################################################################
DO_CHECKS = True # perform and output various consistency / performance checks


###########################################################################
TRY_CASES = {sig: [1], G0:[-1], t: [0.02, 0.25, 1, 3], T: [0.02, 0.25, 1, 3], Tex: [0.02, 0.25, 1, 3], gam: (0.5, 1.5), alp: [0.05, 0.20, 0.50], alps: [0.05**2, 0.20**2, 0.50**2], the: [0.05, 0.20, 0.50], thes: [0.05**2, 0.20**2, 0.50**2], lam: (1, 30), nu: (0.5, 5), rho: (-0.99, -0.10)}
# Dictionary for simulation of parameters for consistency/performance checks
# Format: symbol: [possible values] to randomly chose from, OR symbol: (min, max) to randomly draw from the interval
# Note: the values/min/max should cover extremes (to stress-test the evaluations) but still remain realistic
# Note: this is a universal / redundant list - only parameters contained in expressions will get their values set to the random values
# also, "nu" will be restricted to enforce non-degeneracy conditions


###########################################################################
# lambdification library to be  used  for consistency / performance checks of symbolic expressions (no influence on proper expressions):
LIB_MATH_symb = "mpmath" # slower but higher precision. also, precision here can be steered via: mpmath.mp.dps = 50
#LIB_MATH_symb = "math" # simple fast, but can result in problems in symb. evaluation due to overflows, in case of extreme parameter values (especially large T with large expansion orders)

LIB_MATH_num = "math" # dito for numeric integration lambdifications (here no such numeric issues)


###########################################################################
# COLORs for output
CLR_text    = "\033[0m"   # ordinary text: black
CLR_check   = "\033[93m"  # checks text and checks results: neutral / yellow
CLR_alarm   = "\033[91m"  # checks results: alarm / red
CLR_ok      = "\033[92m"  # checks results: as expected / green


########################################################################################################################
## END OPTIONS & SETTINGS ##############################################################################################
########################################################################################################################







##### check settings
assert mrZABR_APPRX or mrZABR_EXACT #mrZABR_APPRX and/or mrZABR_EXACT should be set to True"
assert mrZABR_EXPNS in (1,2,3) #mrZABR_EXPNS: only 1, 2, 3 allowed"
assert int(mrZABR_ORDER) ==mrZABR_ORDER and mrZABR_ORDER>=1 #mrZABR_ORDER: only integer 1 or above allowed"
assert mrZABR_GAMMA == gam or (mrZABR_GAMMA >=0 and mrZABR_GAMMA<=2) #mrZABR_GAMMA: should be from interval (0, 2)"
#####



# additional / helper sympy symbols
mu = symbols("mu",   pistive=True)
dlt = symbols("dlt", real=True)
x = symbols("x")
Ts = symbols("T'", nonegative=True)
T1 = symbols("T1", nonegative=True)
T2 = symbols("T2", nonegative=True)

a0 = symbols ("a0", positive=True)
af = symbols ("af", positive=True)
tf = symbols ("tf", positive=True)
t0 = symbols ("t0", positive=True)
u = symbols ("u", positive=True)
v = symbols ("v", positive=True)
Z = symbols ("Z", positive=True)
zzz = symbols ("zzz", positive=True)






# INTEGRATION ROUTINES ####################

# import numeric integration routines
import num_integration
num_integration.LIB_MATH = LIB_MATH_num
from num_integration import num_integration_recursive_single_double, num_integration_simpson, num_integration_recursive_unlimited,  build_nested_quad_from_Raw1, build_nested_quad_from_Raw2, build_nested_quad_from_Raw3

# import symbolic integration routines
from symbolic_gram_matrix import antiderivative_square_polyexp, integrate_square_polyexp_grouped # for some alternative derivations



def integrate_sum (expr, var_limits, simplify_terms=False):
    """
    exact/symbolic integration which first enforces dealing an integral of a sum as a sum of integrals
    (as sympy integrate method sometimes fails to do that)
    """
    if str(expr.func) == "<class 'sympy.core.add.Add'>":
        exa = expr.args
        res = 0
        for i, arg in enumerate(exa):
            prg = "..." + str(i + 1) + "/" + str(len(exa)) + "..."
            print(f"{prg:<15}", end="")
            if simplify_terms:
                ex_this = integrate(arg.simplify(), var_limits)
            else:
                ex_this = integrate(arg, var_limits)
            res = res + ex_this
            print("\b" * 15, end="")
        return res
    else:
        return integrate(expr, var_limits)




def integrate_expansion (expr, var_limits, appr_mode, appr_order, series_exp = False, simplify_terms=False):
    """
    approximative symbolic integration:
    first approximate the integrand via series
    then apply exact symbolic integration

    appr_mode:  sets which variable is expanded around which value/variable
    appr_order: degree of series expansion
    series_exp: expand (develop) the terms of the expanded series prior to symbolic integration (helps sympy in certain situations)
    """
    res = None
    assert appr_mode in (1, 2, 3)
    if appr_mode == 1:
        # expand alpha around theta
        expr_exp=expr.series(alp, the, appr_order).removeO()
        if series_exp:
            expr_exp=expr_exp.expand()
        res = integrate_sum (expr_exp, var_limits, simplify_terms=simplify_terms) # .expand()
    elif appr_mode == 2:
        # expand theta around alpha
        expr_exp =expr.series(the, alp, appr_order).removeO()
        if series_exp:
            expr_exp=expr_exp.expand()
        res = integrate_sum (expr_exp, var_limits, simplify_terms=simplify_terms) # .expand()
    elif appr_mode == 3:
        if expr.has(dlt) and expr.has(mu):
            # !!! more efficient: for re-parametrization to mu/delta in the very beginning (and substituting back in the very end)
            # with delta = (alpha - theta) / (alpha + theta) and  mu = (alpha + theta) / 2

            expr_exp = expr.series(dlt, 0, appr_order).removeO()
            if series_exp:
                expr_exp=expr_exp.expand()
            res = integrate_sum (expr_exp, var_limits, simplify_terms=simplify_terms) # series expansion in mu-delta space

        else:
            # same as above but use original variables (alpha, theta)
            expr = expr.subs (the, mu * (1 - dlt)).subs(alp, mu * (1 + dlt)).simplify() # substitute alpha and theta thru delta and mu
            expr_exp = expr.series(dlt, 0, appr_order).removeO()
            if series_exp:
                expr_exp=expr_exp.expand()
            res = integrate_sum (expr_exp, var_limits, simplify_terms=simplify_terms) # series expansion in mu-delta space
            res = res.subs(dlt, (alp - the) / (alp + the)).subs(mu, (alp + the) / 2).simplify() # substitute back
    return res


# transformations from mu-delta to alpha-theta and back

def trf_mu_dlt (expr):
    if  mrZABR_EXPNS==3:
        return expr.subs({the: mu * (1 - dlt), alp:  mu * (1 + dlt)})
    else:
        return expr

def trf_back_mu_dlt (expr):
    if  mrZABR_EXPNS==3:
        return expr.subs({dlt: (alp - the) / (alp + the), mu: (alp + the) / 2})
    else:
        return expr

def trf_make_flat (expr, flatVola): # flat vola in  mu-delta space, flatVola is symbol to set for the flat vola
    if  mrZABR_EXPNS==3:
        return expr.subs(dlt,0).subs(mu, flatVola)
    else:
        return expr.subs(alp,flatVola).subs(the, flatVola)



# OTHER FUNCTIONS ###############################################################################################

def get_random_parameters (heston = False):
    """
    returns a dictionary with random parameters

    if TRY_CASES value is array [possible values] - randomly chose from it
    if tuple (min, max) - randomly draw from the interval
    """

    rnd = {symbol: random.uniform(logic[0], logic[1]) if isinstance(logic, tuple) else random.choice(logic) for symbol, logic in TRY_CASES.items()}

    # apply non-degeneracy conditions
    if heston:
        max_nu = math.sqrt ( 4 * rnd[lam] * rnd[thes] ) # this is condition for no critical degeneracy in Heston
    else:
        max_nu = math.sqrt ( 4  * rnd[lam] * rnd [the]) # strictly speaking this is the condition for no critical degeneracy in CIR-ZABR (mrZABR with gamma=1/2) only

    if rnd[nu]> max_nu:
        rnd[nu] =  random.uniform(0, max_nu)

    rnd[mu]   = float( (rnd[alp] + rnd[the])/2  )
    rnd[dlt]  = float( (rnd[alp] - rnd[the])  /  (rnd[alp] + rnd[the])   )

    return rnd


# relative deviation
def get_rel_deviation (val1, val2):
    return abs (val1 - val2) / (0.5*abs (val1) + 0.5*abs (val2))

def print_eval_stats (expr_symbolic, expr_raw=None,  prefix="", lambdify_Symb=True, lambdify_Num=True, fn_inner=None):
    """
    Uses simulated parameter inputs
    to estimate the speed ans robustness of symbolic integration (upon lambdification)
    and (optionally) to cross-check  symbolic vs numerical integrations

    expr_symbolic: symbolic closed-form sympy expression with all integrals evaluated
    OPTIONAL:
    expr_raw: sympy expression with raw integrals not evaluated, to be evaluated numerically
    """

    NUM_PRECISION_abs = 1e-9 # for numeric integration
    NUM_PRECISION_rel = 1e-9 # for numeric integration

    max_time=10        # max time (seconds) for  evaluations
    min_samples = 30   # min sample size for  evaluations
    max_samples = 1000 # max sample size

    # determine if hSABR model (as it has a different non-degeneracy condition)
    heston = False
    if expr_symbolic.has(alps) or expr_symbolic.has(thes):
        heston = True

    # generate/simulate random parameter samples
    sample_cases=[]
    for i in range(0, max_samples):
        sample_cases.append (get_random_parameters(heston))

    print(f"{CLR_check}{prefix}Evaluation: SYMBOLIC: ", "...", end="")

    #all_vars = [k for k, v in TRY_CASES.items()]
    vrs = list(expr_symbolic.free_symbols)
    values_symbolic  = []

    if expr_symbolic.has(Integral):
        print(f"\b\b\b{CLR_alarm} uneval.Integrals {CLR_text}")

        return
    if lambdify_Symb:
        function_symbolic = sp.lambdify(vrs, expr_symbolic, LIB_MATH_symb)

    # do symbolic integrations on random parameters, up to time limit
    start_lim = time.time()  # process_time()
    start = time.perf_counter()# process_time()  # time()
    for sc in sample_cases:
        prm_values = [sc[vr0] for vr0 in vrs]
        try:
            if lambdify_Symb:# call symbolic evaluation via lambdified function
                values_symbolic.append(  function_symbolic(*prm_values)  )
            else:# call symbolic evaluation via simple replace
                values_symbolic.append(float(expr_symbolic.xreplace(sc).evalf()))
        except:
            print (f"\b\b\b{CLR_alarm}problems in symb. evaluation with parameters: ", sc, f"{CLR_text}")
            #assert  False
            return
        time_lim = time.time() - start_lim  # process_time() #perf_counter()

        if time_lim > max_time and len(values_symbolic) >= min_samples:
            break

    time_symbolic = - start + time.perf_counter() #process_time()   #time()

    # avg time in microseconds
    print (f"\b\b\b{CLR_check}Avg time {'(lambd.)'if lambdify_Symb else ''} {time_symbolic / len(values_symbolic) * 1e6:.2f} µs. ", end="")



    # do numeric integrations on random parameters, up to time limit
    if  expr_raw is not None:
        print(f"{CLR_check} NUMERIC: ...", end="")
        vrs = list(expr_raw.free_symbols)
        values_numeric = []


        if lambdify_Num:

            # lambdification makes numeric integrals fast functions of parameters
            # which is needed to e.g. fit parameters etc

            if len(expr_raw.limits) == 1: # simple integrals
                expr_raw_lmb = build_nested_quad_from_Raw1 (expr_raw, vrs, modules= LIB_MATH_num, epsabs=NUM_PRECISION_abs,  epsrel=NUM_PRECISION_rel, limit=1000)
            elif len(expr_raw.limits) == 2: # double nested integrals
                if fn_inner is None:# to cover I1^2 in cTex
                    expr_raw_lmb = build_nested_quad_from_Raw2 (expr_raw, vrs, modules=LIB_MATH_num, epsabs=NUM_PRECISION_abs, epsrel=NUM_PRECISION_rel, limit=1000)
                else:
                    expr_raw_lmb = build_nested_quad_from_Raw2(expr_raw, vrs, fn_inner=fn_inner, modules=LIB_MATH_num, epsabs=NUM_PRECISION_abs, epsrel=NUM_PRECISION_rel, limit=1000)
            elif len(expr_raw.limits) == 3:# triple nested integrals
                expr_raw_lmb = build_nested_quad_from_Raw3(expr_raw, vrs, modules=LIB_MATH_num, epsabs=NUM_PRECISION_abs,  epsrel=NUM_PRECISION_rel, limit=1000)
            else:
                assert False # in principle, can be extended, but max needed depth is here 3

            start_lim = time.time()  # process_time()
            start = time.perf_counter()
            for sc in sample_cases:
                prm_values = [sc[vr0] for vr0 in vrs]
                val, err = expr_raw_lmb (*prm_values)
                values_numeric.append(     val     )
                time_lim = time.time() - start_lim  # process_time() #perf_counter()

                if time_lim > max_time and len(values_numeric) >= min_samples:
                    break
            time_numeric = - start + time.perf_counter()


        else:
            # here, lambdification is also internally used for nesting, but not for the parameters

            start_lim = time.time()# process_time()
            start = time.perf_counter()

            single_double_ok=True
            val=None
            for sc in sample_cases:
                if single_double_ok:
                    try:
                        val = num_integration_recursive_single_double(expr_raw.xreplace(sc), useScipy=True)
                        # works faster but can handle only single / double
                    except:
                        single_double_ok = False
                if single_double_ok==False:
                    val = num_integration_recursive_unlimited (expr_raw.xreplace(sc))
                values_numeric.append (val)

                time_lim = time.time () - start_lim #process_time() #perf_counter()

                if time_lim > max_time and len(values_numeric)>= min_samples:
                    break
            time_numeric = - start + time.perf_counter()  # process_time()   #time()


        n_samples = min (len(values_symbolic), len(values_numeric))
        values_symbolic2 = values_symbolic[0: n_samples]
        values_numeric2  = values_numeric[0: n_samples]

        # invsestigate relative deviations between numeric and symbolic
        devs  = [ float(  get_rel_deviation (symb_value, num_value) )   for symb_value, num_value in  zip (values_numeric2, values_symbolic2) ]
        dev_max = max(devs)
        index_of_max = devs.index(dev_max)
        dev_avg = sum(devs) /len(devs)

        # only relevant parameters
        vars_relevant=list(set(expr_symbolic.free_symbols) | set(expr_raw.free_symbols))
        #[k for k, v in TRY_CASES.items() if expr_symbolic.has(k) or expr_raw.has(k)]

        # format deviation metrics
        dev_avg_s= f"{dev_avg*100:.2f}" if dev_avg*100 >=0.01 else f"{dev_avg*100:.0e}"
        if dev_avg < 0.01:
            dev_avg_s = f"{CLR_ok}" + dev_avg_s + f"{CLR_check}" # green
        elif dev_avg > 0.1:
            dev_avg_s = f"{CLR_alarm}" + dev_avg_s + f"{CLR_check}" # red

        dev_max_s= f"{dev_max*100:.2f}" if dev_max*100 >=0.01 else f"{dev_max*100:.0e}"
        if dev_max < 0.10:
            dev_max_s = f"{CLR_ok}" + dev_avg_s + f"{CLR_check}" # green
        elif dev_max> 0.5:
            dev_max_s = f"{CLR_alarm}" + dev_max_s + f"{CLR_check}" # red

        print (f"\b\b\bAvg time {'(lambd.)'if lambdify_Num else ''} {time_numeric / len(values_numeric)  * 1e6:.2f} µs, Deviat. vs symb.: avg {dev_avg_s} %, max {dev_max_s} % on {n_samples} input sets of {vars_relevant}, max dev ({values_symbolic2[index_of_max]} vs {values_numeric2[index_of_max]}) at: {[  sample_cases[index_of_max] [vr0] for vr0 in vars_relevant]}", end="")
    print(f"{CLR_text}")
    return


# two functions used when running sympy symbolic integration
def print_expression_title(expr_name, width):
    expr_name = expr_name + " ="
    print (f"{expr_name:<{width}}...", end="") # print out with fixed width + ... to indicate a workload in progress

def print_expression_value(expr_value):
    print ("\b\b\b", end="") # delete the ... previously printed
    if isinstance(expr_value, sp.Basic):
        if expr_value.has(Integral): # if the result still contains unevaluated integrals
            print (f" {CLR_alarm}(!!! still has integrals !!!){CLR_text} ", expr_value) # clearly indicate the unevaluated integrals
        else:
            print (expr_value) # just print out the result
    else:
        print(expr_value)

    # combined
def print_expression (expr_name, expr_value, width):
    print_expression_title(expr_name, width)
    print_expression_value(expr_value)
    return

# two functions used to check if two sympy symbolic expressions are mathematically identical
def print_identity_check_title (descr):
    print(f"{CLR_check}" + descr + "   ", end="...") # description of the check in yellow
    return

def print_identity_check_result (expr1, expr2, numeric_check=False):
    if expr1.free_symbols != expr2.free_symbols:
        print(f"\b\b\b{CLR_alarm}FALSE: diff. parameters {CLR_text}")  # red
        return
    if expr1.has(Integral) or expr2.has(Integral):
        print(f"\b\b\b{CLR_alarm}FALSE: uneval. Integrals {CLR_text}")  # red
        return
    if numeric_check:
        ck=-1
    else:
        ck = ((expr1) - (expr2)).simplify()
    if ck==0:
        print(f"\b\b\b{CLR_ok}TRUE{CLR_text}")
        return# green
    else: # if sympy fails to prove the identity analytically, check numerically
        difnum=[]
        #prms=[]
        for i in range(0, 20):
            prm = get_random_parameters()
            expr1num = expr1.subs (prm)
            expr2num = expr2.subs (prm)
            if im (expr1num)>0 or im(expr2num)>0 or expr1num.is_infinite or expr2num.is_infinite: # if complex or infinite numebrs
                difnum.append(3) # max deviation, even higher than otherwise possible (2) to mark
            else:
                difnum.append( get_rel_deviation(expr2num,expr1num ))
            #prms.append(prm)
        difnum_max = max(difnum)
        if difnum_max>1e-6: # max rel deviation
            print(f"\b\b\b{CLR_alarm}FALSE, num dev: " + str(difnum_max * 100) + f" %{CLR_text}")  # red
        else:
            print(f"\b\b\b{CLR_check}??? (symbolic),{CLR_ok}TRUE (numeric), num dev: " + str(difnum_max * 100), f" %{CLR_text}")  # neutral
    return

# combined
def print_identity_check (descr, expr1, expr2, numeric_check=False):
    print_identity_check_title (descr)
    print_identity_check_result(expr1, expr2, numeric_check)
    return




####################################################################################################################
####################################################################################################################

if __name__ == "__main__":

    print ("")
    print("* key expressions **************************")
    print ("")
    #  expected instantaneous vola for a time point T (mrSABR)
    alpT = alp * sp.exp(- (T * lam) ) + integrate( the * lam *  sp.exp( - lam * (T - x) ), (x, 0, T)   )
    #alpT = alpT.simplify()
    print_expression (" mrSABR α(T)", alpT, 23)
    # squared  for inner integrals:
    alpT1sq = (alpT.subs(T, T1))**2
    #  expected  instantaneous variance for a time point T (hSABR)
    varT = alps  * sp.exp(- (T * lam) ) + integrate( thes * lam *  sp.exp( - lam * (T - x) ), (x, 0, T)   ) # Heston
    print_expression (" hSABR V(T)", varT, 23)

    # instantaneous variance integrated until a time point T
    tauT_mrSABR = (sig ** 2) * integrate((alpT.subs(T, Ts)) ** 2, (Ts, 0, T))
    tauT_mrSABR = tauT_mrSABR.simplify()
    print_expression (" mrSABR τ(T)", tauT_mrSABR, 23)
    # here until option expiry Tex
    tauex_mrSABR =   tauT_mrSABR.subs(T, Tex)
    #print (" mrSABR τ_ex = τ(T_ex) ", tauex_mrSABR)


    tauT_hSABR =    (sig**2) * integrate(  varT.subs(T, Ts), (Ts, 0, T) )
    tauT_hSABR = tauT_hSABR.simplify()
    print_expression (" hSABR τ(T)", tauT_hSABR, 23)
    DT1T_hSABR =    (sig**2) * integrate(  sp.exp (- lam * (T2-T1) ) , (T2, T1, T) )
    #print (" hSABR D(T₁,T)         ", DT1T_hSABR)
    tauex_hSABR =   tauT_hSABR.subs(T, Tex)
    #print (" hSABR τ_ex = τ(T_ex)  ", tauex_hSABR)



    # mrZBAR helper variables f
    print ("")
    print (" mrZABR:")

    if DO_CHECKS:
        # mrZABR PAPER function Y (t=t_f, t_0, a_0): expected vola in t_f given vola a_0 in t_0
        Y_tft0a0 = a0 * sp.exp (-lam*(tf-t0)) + the * (     1 - sp.exp(-lam*(tf-t0))   )

        # mrZABR PAPER function y (t_0, t=t_f, a_f): inverted Y: vola in t_0 which generates  expected vola a_f in t_f
        y_t0tfaf = (af - the * (     1 - sp.exp(-lam*(tf-t0)) )    )    / sp.exp (-lam*(tf-t0))

        # mrZABR PAPER function X (t=t_f, t_0, a_0): derivative of Y (t=t_f, t_0, a_0) wrt a_0 (does not depend here on a_0)
        X_tft0_1 = sp.exp(-lam*(tf-t0))

        # just rename t_f to t and a0 to alpha
        Y_tft0a0_2=Y_tft0a0.subs([(tf, t), (t0, t0), (a0, alp)])

        # mrZABR PAPER version for Z (t, u): expected vola in u given vola=alpha in t0, caluclated via expected value at t
        Z_tu1 = y_t0tfaf.subs([(t0, u), (tf, t)]).subs(af, Y_tft0a0_2).simplify()

    # simplification: Z (t, u) must be just the expected vola in u given vola=alpha in t0
    Z_tu2 = the + (alp-the)*sp.exp(-lam*(u-t0))
    print_expression (" Z (t,u)", Z_tu2, 23)

    # check that simplification vs mrZABR PAPER is correct
    X_tft0_2= Z_tu2.diff(alp).subs(u, tf)   # idetical to X_tft0 above (mrZABR PAPER version), just for consistency
    print_expression (" X(t=t_f,t0,alpha)", X_tft0_2, 23)

    if DO_CHECKS:
        print_identity_check ("CHECK: Z (t,u) vs mrZABR PAPER is correct?", Z_tu1, Z_tu1)
        print_identity_check ("CHECK: X (t=t_f, t0, alpha) vs mrZABR PAPER is correct?", X_tft0_1, X_tft0_2)



    # use below simplified versions
    Z_tu = Z_tu2
    X_tft0=X_tft0_2




    # apply variable changes to mu-delta  (only for interim mrZABR expressions which are used later)

    Z_tu = trf_mu_dlt (Z_tu).simplify()
    X_tft0 = trf_mu_dlt (X_tft0).simplify()
    alpT_mrZABR = trf_mu_dlt (alpT).simplify()
    tauex_mrZABR  = trf_mu_dlt(tauex_mrSABR.subs(sig, 1)).simplify()

    #####################################################################
    # ν(Z) function which defines the mrZABR model type #
    if mrZABR_GAMMA ==gam: # full-fledged mr-ZABR
        nu_Z  = nu * Z**gam # mrZABR, works too but generates very long expressions
    elif mrZABR_GAMMA == 1:
        nu_Z   = nu * Z  # reduces to mrSABR
    #elif mrZABR_GAMMA==1/2 :
    #    nu_Z  = nu * sp.sqrt (Z) # CIR-ZABR
    elif mrZABR_GAMMA>=0 and mrZABR_GAMMA<=2:
        nu_Z = nu * Z ** mrZABR_GAMMA
    else:
        print ("wrong mrZABR_GAMMA")
        assert (False)


    print_expression (" ν(Z)", nu_Z, 23)
    ######################################################################
    psi_tuZ   = nu_Z.subs(Z, Z_tu)  *  Z_tu  * X_tft0.subs(tf, t).subs(t0, u) # as used in mrZABR PAPER
    print_expression (" Ψ (t, u, Z)", psi_tuZ, 23)
    psi_Z_alt   = nu_Z  *  Z # alternative definition without exponential term as used in this PAPER


    #derivative of Ψ (t, u, Z) wrt Z
    psi_tuZ_deriv = diff( nu_Z * Z , Z).subs(Z, Z_tu)  * X_tft0.subs(tf, t).subs(t0, u)
    psi_tuZ_deriv = psi_tuZ_deriv.simplify()
    print_expression (" d Ψ(t, u, Z) / dZ", psi_tuZ_deriv, 23)
    psi_Z_alt_deriv = diff (psi_Z_alt, Z) # alternative in this PAPER



    #############################################################################################################
    print ("")
    print ("")
    print ("*****************************************************************************")
    print( "* INTERIM INTEGRALS ********************************************************")
    print ("*****************************************************************************")


    # the integrations below follow hSABR PAPER, mrSABR PAPER and mrZABR PAPER
    # after the simplifications resulting from the time-independent parameters
    # besides, for mrZABR, an approximation of the integrands is used to enable symbolic integration


    print ("")
    print ("")
    print("* I1 interim integral **************************")

    print("")
    print_expression_title ("hSABR: I₁(T)", 30)
    I1T_hSABR =   rho * nu * sig * integrate ( varT.subs(T, T1)  *  sp.exp( -lam * (T-T1) ) , (T1, 0, T) )
    I1T_hSABR = I1T_hSABR.simplify()
    print_expression_value (I1T_hSABR)

    print ("")
    print_expression_title ("mrSABR: I₁(T)", 30)
    I1T_mrSABR =   rho * nu * sig * integrate (  (alpT1sq)  *  sp.exp( -lam * (T-T1) ) , (T1, 0, T) )
    I1T_mrSABR = I1T_mrSABR.simplify()
    print_expression_value (I1T_mrSABR)


    I1t_mrZABR_intgd = psi_tuZ # integrand
    I1t_mrZABR_intgd =I1t_mrZABR_intgd.subs(t0,0)#.simplify()

    I1t_mrZABR_raw = Integral( rho * I1t_mrZABR_intgd, (u, 0, t))


    if mrZABR_APPRX:
        print("")
        print_expression_title("mrZABR: I₁(t) approx", 30)
        I1t_mrZABR = rho * integrate_expansion (I1t_mrZABR_intgd, (u, 0, t), appr_mode = mrZABR_EXPNS, appr_order = mrZABR_ORDER, series_exp=True)
        I1t_mrZABR = I1t_mrZABR.simplify()
        print_expression_value(I1t_mrZABR)

        if DO_CHECKS:
            # above is integral specification  as in mrZABR PAPER
            # this PAPER uses slightly simplified / rearranged specification, just crosscheck both are identical
            print_identity_check_title ("Integral in mrZABR PAPER is identical to the alternative one in this PAPER?")
            I_int = psi_Z_alt.subs(Z, alpT_mrZABR.subs(T, T1)) * sp.exp (-lam* (T-T1))
            I_this_paper = rho  * integrate_expansion (I_int, (T1, 0, T), appr_mode = mrZABR_EXPNS, appr_order = mrZABR_ORDER, series_exp=True)
            print_identity_check_result (  I1t_mrZABR.subs(t, T),  I_this_paper )

            # just to show once advantages lambdification
            print_eval_stats(I1t_mrZABR, I1t_mrZABR_raw, lambdify_Symb=False, lambdify_Num=False, prefix= "NO LAMBDIFY: ")
            print_eval_stats(I1t_mrZABR, I1t_mrZABR_raw)


    if mrZABR_EXACT:
        print ("")
        print_expression_title("mrZABR: I₁(t) exact", 30)
        I1t_mrZABR = rho * integrate_sum (I1t_mrZABR_intgd.expand(), (u, 0, t))
        I1t_mrZABR = I1t_mrZABR.simplify()
        print_expression_value(I1t_mrZABR)

        if DO_CHECKS:

            if mrZABR_GAMMA == 1:
                # check indetity vs mrSABR when  gamma=1
                print_identity_check("Identity vs mrSABR (expected only if gamma=1):", I1T_mrSABR.subs(sig, 1), trf_back_mu_dlt(I1t_mrZABR.subs(t, T)))

            print_eval_stats(I1t_mrZABR, I1t_mrZABR_raw)



    print("")
    #######################################

    print ("")
    print("* I2 interim integral **************************")

    print("")
    print_expression_title  ("hSABR: I₂(T)", 30)
    I2T_hSABR =  rho * nu * sig * integrate ( varT.subs(T, T1)  * DT1T_hSABR, (T1, 0, T) )
    I2T_hSABR = I2T_hSABR.simplify()
    print_expression_value  (I2T_hSABR)


    print("")
    print_expression_title ("mrSABR: I₂(T)", 30)
    I2int_mrSABR = (sig**2) * integrate ( (alpT.subs(T, T2)) * sp.exp(- lam * (T2-T1)) , (T2, T1, T) ) # inner integral
    I2T_mrSABR =  (nu**2) *  integrate ( (alpT1sq) * sp.exp(- lam * (T-T1) ) *  I2int_mrSABR, (T1, 0, T) )
    I2T_mrSABR = I2T_mrSABR.simplify()
    print_expression_value (I2T_mrSABR)


    print("")
    print_expression_title ("mrZABR: inner of I₂(T)", 30)
    I2_mrZABRint_integrand = Z_tu.subs(u, v) * (X_tft0.subs(tf, t).subs(t0, v))**(-1)
    I2_mrZABRint = integrate( I2_mrZABRint_integrand  , (v, u, t)) # inner integral
    I2_mrZABRint=I2_mrZABRint.simplify()
    print_expression_value (I2_mrZABRint)


    # integrand
    I2t_mrZABR_intgd = nu_Z.subs(Z,Z_tu)**2  *  X_tft0.subs(tf, t).subs(t0, u) **2  * I2_mrZABRint
    I2t_mrZABR_intgd = I2t_mrZABR_intgd.subs(t0, 0).simplify()


    #I2t_mrZABR_raw  =  Integral( 2 * I2t_mrZABR_intgd, (u, 0, t))
    # the inner integral does have closed-form solution here,
    # but for consistency (performance comparison) do not use it for raw and leave it double nested
    I2t_mrZABR_raw = Integral(Integral (2 * nu_Z.subs(Z,Z_tu)**2  *  X_tft0.subs(tf, t).subs(t0, u) **2
                   *  I2_mrZABRint_integrand, (v, u, t)), (u, 0, t)).subs(t0, 0)



    if mrZABR_APPRX:
        print("")
        print_expression_title("mrZABR: I₂(t) approx", 30)
        I2t_mrZABR = 2 * integrate_expansion(I2t_mrZABR_intgd, (u, 0, t), appr_mode=mrZABR_EXPNS, appr_order=mrZABR_ORDER, series_exp=True)
        I2t_mrZABR = I2t_mrZABR.simplify()
        print_expression_value(I2t_mrZABR)

        if DO_CHECKS:
            print_identity_check_title("Integral in mrZABR PAPER is identical to the alternative one in this PAPER?")
            I_int = integrate ( alpT_mrZABR.subs(T, T2) * sp.exp (lam*(T-T2)), (T2, T1, T))
            I_int = I_int * nu_Z.subs(Z, alpT_mrZABR.subs(T, T1)) **2 * sp.exp (-2 * lam * (T-T1))
            #psi_Z_alt.subs(Z, alpT_mrZABR.subs(T, T1)) * sp.exp (-lam* (T-T1))
            I_this_paper = integrate_expansion (I_int, (T1, 0, T), appr_mode = mrZABR_EXPNS, appr_order = mrZABR_ORDER, series_exp=True)
            print_identity_check_result ( I2t_mrZABR.subs(t, T),  I_this_paper * 2)

            print_eval_stats(I2t_mrZABR, I2t_mrZABR_raw, lambdify_Symb=False, lambdify_Num=False, prefix= "NO LAMBDIFY: ")
            print_eval_stats(I2t_mrZABR, I2t_mrZABR_raw)


    if mrZABR_EXACT:
        print("")
        print_expression_title("mrZABR: I₂(t) exact", 30)
        I2t_mrZABR = 2 * integrate_sum (I2t_mrZABR_intgd.expand(), (u, 0, t))
        I2t_mrZABR = I2t_mrZABR.simplify()
        print_expression_value(I2t_mrZABR)

        if DO_CHECKS:
            if mrZABR_GAMMA == 1:
                #check reduction to mrSABR (expected for   gamma=1)
                print_identity_check("Identity vs mrSABR (expected only if gamma=1):", 2 * I2T_mrSABR.subs(sig, 1), trf_back_mu_dlt (I2t_mrZABR.subs(t, T)))
                # I2 in mrZABR PAPER = 2 x I2 in mrSABR PAPER

            print_eval_stats(I2t_mrZABR, I2t_mrZABR_raw)


    print ("")
    print ("")
    print("* I3 interim integral **************************")


    print("")
    print_expression_title ("hSABR: I₃(t)", 30)
    I3T_hSABR =  nu **2  * integrate( varT.subs(T, T1)  * DT1T_hSABR * sp.exp (- lam*(T-T1) ), (T1, 0, T))
    I3T_hSABR = I3T_hSABR.simplify()
    print_expression_value (I3T_hSABR)

    print("")
    print_expression_title ("mrSABR: I₃(t)", 30)
    I3int_mrSABR = I2int_mrSABR # inner inregral - same as in I2
    I3T_mrSABR=  rho * nu * sig * integrate ( (alpT1sq) * I3int_mrSABR, (T1, 0, T) )
    I3T_mrSABR= I3T_mrSABR.simplify()
    print_expression_value(I3T_mrSABR)

    if DO_CHECKS:
        print_identity_check_title("Relashioship I₁ vs I₃ as implied in mrSABR PAPER?")
        ck = sig**2 *  integrate ((I1T_mrSABR * alpT).subs(T, x), (x, 0 , T) )
        print_identity_check_result(I3T_mrSABR, ck)


    I3_mrZABRint_integrand = I2_mrZABRint_integrand
    I3_mrZABRint = I2_mrZABRint # inner inregral - same as in I2
    #integrand of outer integral without constants
    I3t_mrZABR_intgd = psi_tuZ * I3_mrZABRint
    I3t_mrZABR_intgd = I3t_mrZABR_intgd.subs(t0, 0).simplify()


    #I3t_mrZABR_raw = Integral(rho * I3t_mrZABR_intgd, (u, 0, t))
    # the inner integral does have closed-form solution here,
    # but for consistency (performance comparison) do not use it for raw and leave it double nested
    I3t_mrZABR_raw = Integral(rho *  psi_tuZ * I3_mrZABRint_integrand, (v, u, t), (u, 0, t)).subs(t0, 0)



    if mrZABR_APPRX:
        print("")
        print_expression_title("mrZABR: I₃(t) approx", 30)
        I3t_mrZABR =  rho * integrate_expansion(I3t_mrZABR_intgd, (u, 0, t), appr_mode=mrZABR_EXPNS, appr_order=mrZABR_ORDER, series_exp=True)
        I3t_mrZABR = I3t_mrZABR.simplify()
        print_expression_value(I3t_mrZABR)

        if DO_CHECKS:
            print_identity_check_title ("Integral in mrZABR PAPER is identical to the alternative one in this PAPER?")
            I_int = integrate ( alpT_mrZABR.subs(T, T2) * sp.exp (lam*(T-T2)), (T2, T1, T))
            I_int = I_int * psi_Z_alt.subs(Z, alpT_mrZABR.subs(T, T1)) * sp.exp ( - lam * (T-T1))
            #psi_Z_alt.subs(Z, alpT_mrZABR.subs(T, T1)) * sp.exp (-lam* (T-T1))
            I_this_paper = rho * integrate_expansion (I_int, (T1, 0, T), appr_mode = mrZABR_EXPNS, appr_order = mrZABR_ORDER, series_exp=True)
            print_identity_check_result (  I3t_mrZABR.subs(t, T),   I_this_paper)

            print_eval_stats(I3t_mrZABR, I3t_mrZABR_raw)


    if mrZABR_EXACT:
        print("")
        print_expression_title("mrZABR: I₃(t) exact", 30)
        I3t_mrZABR = rho * integrate_sum (I3t_mrZABR_intgd.expand(), (u, 0, t))
        I3t_mrZABR = I3t_mrZABR.simplify()
        print_expression_value (I3t_mrZABR)

        if DO_CHECKS:
            if mrZABR_GAMMA == 1:
                #check reduction to mrSABR (expected for   gamma=1)
                print_identity_check("Identity vs mrSABR (expected only if gamma=1):", I3T_mrSABR.subs(sig, 1),  trf_back_mu_dlt ( I3t_mrZABR.subs(t, T) ))

            print_eval_stats(I3t_mrZABR, I3t_mrZABR_raw)


    print ("")
    print ("")
    print("* I4 interim integral **************************")

    print ("")
    print_expression_title("hSABR: I₄(T)", 30)
    I4int_hSABR = rho * nu * sig * integrate (1, (T2, T1, T))
    I4T_hSABR =  rho * nu * sig * integrate (varT.subs(T, T1)  * sp.exp(-lam * (T- T1)) * I4int_hSABR,  (T1, 0, T))
    I4T_hSABR = I4T_hSABR.simplify()
    print_expression_value(I4T_hSABR)


    print ("")
    print("mrSABR: I₄(T):")
    print_expression_title("inner integral", 30)
    I4int_mrSABR = rho * nu * sig * integrate ( alpT.subs(T, T2), (T2, T1, T)) # inner integral
    print_expression_value (I4int_mrSABR)
    print_expression_title("I₄(T)", 30)
    I4T_mrSABR =  rho * nu * sig* integrate ( (alpT1sq) *  sp.exp(- lam * (T-T1) ) *  I4int_mrSABR, (T1, 0, T) )
    I4T_mrSABR = I4T_mrSABR.simplify()
    print_expression_value (I4T_mrSABR)

    ex_int=psi_tuZ_deriv.subs(u, v) * (X_tft0.subs(tf, t).subs(t0, v))**(-1) # WAS PSI_tuZd_deriv

    I4t_mrZABR_raw = Integral(Integral(rho ** 2 *   psi_tuZ *  ex_int, (v, u, t)) , (u, 0, t)).subs(t0, 0)

    if mrZABR_APPRX:
        print("")
        print("mrZABR: I₄(t) approx:")
        print_expression_title("inner integral", 30)
        # here not even inner has closed-form solution
        I4_mrZABRint =  integrate_expansion(ex_int, (v, u, t), appr_mode=mrZABR_EXPNS, appr_order=mrZABR_ORDER, series_exp=True)
        I4_mrZABRint=I4_mrZABRint.simplify()
        print_expression_value(I4_mrZABRint)

        # integrand of outer integral
        I4t_mrZABR_intgd = psi_tuZ * I4_mrZABRint
        I4t_mrZABR_intgd = I4t_mrZABR_intgd.subs(t0, 0).simplify()


        print_expression_title("I₄(t) approx", 30)
        I4t_mrZABR = rho ** 2 *  integrate_expansion(I4t_mrZABR_intgd, (u, 0, t), appr_mode=mrZABR_EXPNS, appr_order=mrZABR_ORDER, series_exp=True)
        I4t_mrZABR = I4t_mrZABR.simplify()
        print_expression_value (I4t_mrZABR)

        if DO_CHECKS:
            print_identity_check_title("Integral in mrZABR PAPER is identical to the alternative one in this PAPER?")
            I_int = integrate_expansion (psi_Z_alt_deriv.subs(Z, alpT_mrZABR.subs(T, T2)), (T2, T1, T), appr_mode = mrZABR_EXPNS, appr_order = mrZABR_ORDER, series_exp=True)
            I_int = I_int * psi_Z_alt.subs(Z, alpT_mrZABR.subs(T, T1)) * sp.exp ( - lam * (T-T1))
            #psi_Z_alt.subs(Z, alpT_mrZABR.subs(T, T1)) * sp.exp (-lam* (T-T1))
            I_this_paper = 1/2*rho**2 * integrate_expansion (I_int, (T1, 0, T), appr_mode = mrZABR_EXPNS, appr_order = mrZABR_ORDER, series_exp=True)
            print_identity_check_result (  I4t_mrZABR.subs(t, T), I_this_paper * 2)

            print_eval_stats(I4t_mrZABR,  I4t_mrZABR_raw)



    if mrZABR_EXACT:
        print("")
        print("mrZABR: I₄(t) exact:")
        print_expression_title("inner integral", 30)
        I4_mrZABRint = integrate(ex_int, (v, u, t))
        print_expression_value (I4_mrZABRint)

        # integrand of outer integral
        I4t_mrZABR_intgd = psi_tuZ * I4_mrZABRint
        I4t_mrZABR_intgd = I4t_mrZABR_intgd.subs(t0, 0).simplify()



        print_expression_title("I₄(t)", 30)
        I4t_mrZABR = rho ** 2 * integrate_sum (I4t_mrZABR_intgd.expand(), (u, 0, t))
        I4t_mrZABR = I4t_mrZABR.simplify()
        print_expression_value(I4t_mrZABR)

        if DO_CHECKS:
            if mrZABR_GAMMA == 1:
                #check reduction to mrSABR (expected for   gamma=1)
                print_identity_check("Identity vs mrSABR (expected only if gamma=1):", trf_back_mu_dlt( I4t_mrZABR.subs(t, T) ), 2 * I4T_mrSABR.subs(sig, 1))
                # I4 mrZABR PAPER  =  2 x I4 in mrSABR PAPER

            print_eval_stats(I4t_mrZABR,  I4t_mrZABR_raw)


    print ("")
    print ("")
    print("* I5 interim integral **************************")

    print("")
    print_expression_title("mrZABR: I₅(T)", 30)
    I5T_mrSABR =  (nu**2) * integrate ( (alpT1sq) *  sp.exp(- 2 * lam * (T-T1) ), (T1, 0, T) )
    I5T_mrSABR = I5T_mrSABR.simplify()
    print_expression_value (I5T_mrSABR)


    #integrand
    I5t_mrZABR_intgd =  nu_Z.subs(Z,Z_tu)**2    *   X_tft0.subs(tf, t).subs(t0, u) **2
    I5t_mrZABR_intgd = I5t_mrZABR_intgd.subs(t0, 0).simplify()

    I5t_mrZABR_raw = Integral(I5t_mrZABR_intgd, (u, 0, t) )


    if mrZABR_APPRX:
        print("")
        print_expression_title("mrZABR: I₅(t) approx", 30)
        I5t_mrZABR = integrate_expansion(I5t_mrZABR_intgd, (u, 0, t), appr_mode=mrZABR_EXPNS, appr_order=mrZABR_ORDER, series_exp=True)
        I5t_mrZABR = I5t_mrZABR.simplify()
        print_expression_value (I5t_mrZABR)

        if DO_CHECKS:
            print_identity_check_title("Integral in mrZABR PAPER is identical to the alternative one in this PAPER?")
            I_int = nu_Z.subs(Z, alpT_mrZABR.subs(T, T1)) **2 * sp.exp (-2 * lam * (T-T1))
            #psi_Z_alt.subs(Z, alpT_mrZABR.subs(T, T1)) * sp.exp (-lam* (T-T1))
            I_this_paper = integrate_expansion (I_int, (T1, 0, T), appr_mode = mrZABR_EXPNS, appr_order = mrZABR_ORDER, series_exp=True)
            print_identity_check_result (   I5t_mrZABR.subs(t, T) , I_this_paper)

            print_eval_stats(I5t_mrZABR, I5t_mrZABR_raw)



    if mrZABR_EXACT:
        print("")
        print_expression_title("mrZABR: I₅(t) exact", 30)

        I5t_mrZABR =  integrate_sum (I5t_mrZABR_intgd.expand(), (u, 0, t))
        I5t_mrZABR = I5t_mrZABR.simplify()
        print_expression_value(I5t_mrZABR)

        if DO_CHECKS:
            if mrZABR_GAMMA == 1:
                #check reduction to mrSABR (expected for   gamma=1)
                print_identity_check("Identity vs mrSABR (expected only if gamma=1):", trf_back_mu_dlt ( I5t_mrZABR.subs(t, T) ), I5T_mrSABR.subs(sig, 1))

            print_eval_stats(I5t_mrZABR, I5t_mrZABR_raw)




    ################################################################

    # transform mrZABR  integrals to mrSABR notations
    I1T_mrZABR = I1t_mrZABR.subs(t, T)
    I2T_mrZABR = I2t_mrZABR.subs(t, T) / 2
    I3T_mrZABR = I3t_mrZABR.subs(t, T)
    I4T_mrZABR = I4t_mrZABR.subs(t, T) / 2
    I5T_mrZABR = I5t_mrZABR.subs(t, T)

    # note that the mrZABR expressions (as in mrZABR PAPER) do not contain the sig parameter
    # unlike the mrSABR / hSABR expressions (as in hSABR PAPER amd mrSABR PAPER) which do contain it
    # applying sig =1 easily removes this parameter later

    ###############################################################


    print ("")
    print ("")
    print ("")
    print("*++++++++++++++++**************************")
    print("* b coefficients **************************")
    print("*++++++++++++++++**************************")


    ######### bTex for hSABR

    bTex_hSABR = I2T_hSABR.subs(T, Tex) / (tauex_hSABR**2)
    bTex_hSABR = bTex_hSABR.simplify()
    print            ("       _")
    print_expression ("hSABR: b(T_ex)",  bTex_hSABR, 50)


    if DO_CHECKS:
        print_eval_stats (bTex_hSABR)

        #derive independently from b(T) using the logic in the hSABR PAPER
        print_identity_check_title("consistent derivation?")
        bT_hSABR = I1T_hSABR / (2 * tauT_hSABR *varT)
        bT_hSABR = bT_hSABR.simplify()
        bTex2_hSABR = 2 / tauex_hSABR**2 * sig**2 * integrate ( varT  * bT_hSABR * tauT_hSABR, (T, 0, Tex))
        print_identity_check_result(bTex_hSABR, bTex2_hSABR)
        # yet another interim formula in hSABR PAPER
        print_identity_check_title("consistent derivation?")
        bTex3_hSABR = 1 / (tauex_hSABR**2)  * sig**2 * integrate (I1T_hSABR, (T, 0, Tex))
        bTex3_hSABR = bTex3_hSABR.simplify()
        print_identity_check_result (bTex_hSABR, bTex3_hSABR)
        # no checks vs flat-vola formula from hSABR PAPER here, as done for cTex later, and b is used in c

    #print_excel(bTex_hSABR)


    ######### bTex for mrSABR
    print("")
    bTex_mrSABR = 2 / tauex_mrSABR ** 2 * I3T_mrSABR.subs(T, Tex)
    bTex_mrSABR = bTex_mrSABR.simplify()
    print            ("        _")
    print_expression ("mrSABR: b(T_ex)", bTex_mrSABR, 50)


    if DO_CHECKS:
        print_eval_stats(bTex_mrSABR)

        #derive independently from b(T) using the logic in the mrSABR PAPER
        print_identity_check_title ("consistent derivation?")
        bT_mrSABR = I1T_mrSABR / alpT / tauT_mrSABR
        bT_mrSABR = bT_mrSABR.simplify()
        # bT_at1=bT.subs(alp, 1).subs(the, 1).simplify()
        # print ("bT_at1", bT_at1)
        bTex2_mrSABR = 2 / tauex_mrSABR ** 2 * sig ** 2 * integrate (alpT ** 2 * bT_mrSABR * tauT_mrSABR, (T, 0, Tex))
        bTex2_mrSABR = bTex2_mrSABR.simplify()
        #expr_excel ("mrSABR: bTex2", bTex2_mrSABR)
        print_identity_check_result(bTex_mrSABR, bTex2_mrSABR)

        # no checks vs flat-vola formula from mrSABR PAPER here, as done for c later, and b is used in c

    #print_excel (bTex_mrSABR)


    ######### bTex for mrZABR

    print("")
    # same logic as in mrSABR (but sigma parameter was not used)
    print                  ("        _")
    print_expression_title ("mrZABR: b(T_ex)", 50)
    bTex_mrZABR = 2 / tauex_mrZABR ** 2 * I3T_mrZABR.subs(T, Tex)
    bTex_mrZABR = bTex_mrZABR.simplify()
    print_expression_value (bTex_mrZABR)
    # no comparison to numeric here, as only I3T involved (already evaluated above)

    if DO_CHECKS:
        print_eval_stats(bTex_mrZABR)

        print_identity_check_title ("consistent vs flat-expected-vola formula?")
        # formula from mrZABR PAPER for special case of flat expected vola
        bTex_mrZABR_flatVola_paper = 2 * rho * nu * alp ** (gam - 1) / lam ** 2 / Tex ** 2 * (lam * Tex - 1 + sp.exp(-lam * Tex))

        cmp1 = trf_make_flat (bTex_mrZABR * sp.sqrt(tauex_mrZABR / Tex), alp)  .simplify().expand()
        # b(T_ex) in mrZABR PAPER is already after multiplication with the term sqrt(tauex/Tex) which is applied here later
        cmp2 = bTex_mrZABR_flatVola_paper.subs (gam, mrZABR_GAMMA).simplify().expand()
        print_identity_check_result (cmp1, cmp2)

    #print_excel (bTex_mrZABR)


    print ("")
    print ("")
    print("*++++++++++++++++**************************")
    print("* c coefficients **************************")
    print("*++++++++++++++++**************************")


    ######### cTex for hSABR
    print                  ("       _")
    print_expression_title ("hSABR: c(T_ex)", 50)
    DTTex_hSABR = sig**2 * integrate (sp.exp (-lam * (T1-T)),  (T1, T, Tex) )

    # as in this PAPER:
    cTex_int_hSABR = integrate (nu**2 * varT * DTTex_hSABR **2 / 4      +  sig**2 * I4T_hSABR,  (T, 0, Tex))
    cTex_int_hSABR =cTex_int_hSABR.simplify()
    cTex_hSABR = 3 / (  tauex_hSABR**3) * cTex_int_hSABR - 3* bTex_hSABR **2
    cTex_hSABR=cTex_hSABR.simplify()
    print_expression_value  (cTex_hSABR)

    if DO_CHECKS:
        print_eval_stats(cTex_hSABR)

        print_identity_check_title("equivalent to mrZABR paper?")
        cTex_hSABR_alt = 3 / (4 * tauex_hSABR ** 3) * nu ** 2 * integrate(varT * DTTex_hSABR ** 2, (T, 0, Tex))
        cTex_hSABR_alt = cTex_hSABR_alt + 3 / (tauex_hSABR ** 3) * sig ** 2 * integrate(I4T_hSABR, (T, 0, Tex))
        cTex_hSABR_alt = cTex_hSABR_alt - 3 * bTex_hSABR **2
        print_identity_check_result(cTex_hSABR, cTex_hSABR_alt)

        # derive independently from c(T) using the logic in the hSABR PAPER
        print_identity_check_title("consistent derivation?")
        cT_hSABR = I3T_hSABR / (2 * tauT_hSABR**2 * varT) - 3*bT_hSABR  * I2T_hSABR  / tauT_hSABR**2 + I4T_hSABR /  (tauT_hSABR**2 * varT)
        cT_hSABR = cT_hSABR.simplify()
        cTex2_hSABR = 3 / tauex_hSABR**3 * sig**2 * ( integrate (I3T_hSABR /2, (T, 0, Tex)) +integrate (I4T_hSABR, (T, 0, Tex)) ) - 3*bTex_hSABR**2
        cTex2_hSABR=cTex2_hSABR.simplify()
        #expr_excel ("hSABR: cTex2", cTex2_hSABR)
        print_identity_check_result(cTex_hSABR, cTex2_hSABR)

        # formula from hSABR PAPER for special case of flat expected vola
        cTex_hSABR_flatVola_paper = (3*nu**2)/sig**2  * (1+2*lam*Tex-(2-sp.exp(-lam*Tex ) )**2)/(8*lam**3 * Tex**3 )+3* (rho**2 * nu**2)/sig**2 *  (lam**2 * Tex**2 *sp.exp(-lam*Tex )-(1-sp.exp(-lam * Tex ) )**2)/( lam**4 * Tex**4 )
        # same approach as in hSABR PAPER: set alpha/theta to 1, sigma defines both
        cTex_hSABR_flatVola = cTex_hSABR.subs(alps, 1).subs(thes, 1).simplify()
        print_identity_check("consistent vs flat-expected-vola formula?", cTex_hSABR_flatVola, cTex_hSABR_flatVola_paper)
        # alterantive approach: set sigma to 1 and alpha=theta to a (new) sigma
        cTex_hSABR_flatVola = cTex_hSABR.subs(sig, 1).subs(alps, sig**2).subs(thes, sig**2).simplify()
        print_identity_check("consistent vs flat-expected-vola formula?", cTex_hSABR_flatVola, cTex_hSABR_flatVola_paper.subs(sig, sig ** 2))
        # !!! The  constant-vola simplification in this PAPER is alpha=theta=sigma (with no extra sigma parameter)
        # but in hSABR PAPER it is alpha=theta and the additional sigma parameter defines the flat vola
        # both approaches are mathematically  identical if sigma in hSABR PAPER formula is squared

    #print_excel (cTex_hSABR)


    ######### cTex for mrSABR
    print("")
    print                  ("        _")
    print                  ("mrSABR: c(T_ex):")

    print_expression_title ("  part 1 (c_int)", 50)
    cTex_int_mrSABR = integrate_sum ( (2*alpT * I2T_mrSABR + I1T_mrSABR**2 + 4 * alpT * I4T_mrSABR).expand() , (T, 0, Tex) )
    cTex_int_mrSABR = cTex_int_mrSABR.simplify()
    print_expression_value (cTex_int_mrSABR)

    if DO_CHECKS:
        print_eval_stats(cTex_int_mrSABR)

    print_expression_title ("  full", 50)
    cTex_mrSABR =  (3 / tauex_mrSABR ** 3 * sig ** 2 * cTex_int_mrSABR - 3 * bTex_mrSABR ** 2) # simplification takes actually most processing time
    print_expression_value (cTex_mrSABR)
    if DO_CHECKS:
        print_eval_stats(cTex_mrSABR)
    #print ("length before simplification:", len (str(cTex_mrSABR)))

    print_expression_title ("  full simplified", 50)
    cTex_mrSABR = cTex_mrSABR.simplify()
    print_expression_value  (cTex_mrSABR)
    #print ("length after simplification:", len (str(cTex_mrSABR)))


    if DO_CHECKS:
        print_eval_stats(cTex_mrSABR)

        # warning overflow (probably due to lambdification) !!!!

        # try to derive from cT (without conversion I1->I3)
        # cT= 2*I2T/alpT/tauT**2 + bT**2 - 6*bT*I3T/tauT**2 + 4*I4T/alpT/tauT**2
        #cTex2_1 = sig**2* integrate ((alpT**2 * tauT**2 *cT).simplify(),  (T, 0, Tex) )
        # not working: python does not reduce the terms, so reduce mannualy (bt= I1T  / alpT / tauT):
        # cTex2_1 = sig**2* integrate ((2*I2T*alpT + I1T**2 - 6*I1T/tauT*I3T*alpT  + 4*I4T*alpT).simplify(),  (T, 0, Tex) )
        # not working either, try simpler special case with alp=the=1
        # cTex2_1 = sig**2* integrate ((2*I2T*alpT + I1T**2 - 6*I1T/tauT*I3T*alpT  + 4*I4T*alpT).subs(alp, 1).subs(the, 1).simplify(),  (T, 0, Tex) )
        # seems to work but results in special functions
        # thus no cross check here
        # but analytical derivation checked and correct (see Word),
        # if identity I3T = sig**2 *  intergral over I1T is correct (and this is checked above)


        # formula from mrSABR PAPER for special case of flat expected vola
        cTex_mrSABR_flatVola_paper = (3*nu**2)/sig**2 *(1+rho**2 ) *(1+2*lam*Tex-(2-sp.exp(-lam*Tex ) )**2)/(2*lam**3*Tex**3 )+12*(rho**2* nu**2)/sig**2  *(lam**2 *Tex**2*sp.exp(-lam*Tex )-(1-sp.exp(-lam*Tex ) )**2)/( lam**4 *Tex**4 )
        # same approach as in mrSABR PAPER: set alpha/theta to 1, sigma defines both
        cTex_mrSABR_flatVola = cTex_mrSABR.subs(alp, 1).subs(the, 1).simplify()
        print_identity_check("consistent vs flat-expected-vola formula?", cTex_mrSABR_flatVola, cTex_mrSABR_flatVola_paper)

        # alterantive approach: set sigma to 1 and alpha=theta to a (new) sigma
        cTex_mrSABR_flatVola = cTex_mrSABR.subs(sig, 1).subs(alp, sig).subs(the, sig).simplify()
        print_identity_check("consistent vs flat-expected-vola formula?", cTex_mrSABR_flatVola, cTex_mrSABR_flatVola_paper)
        # !!! The  constant-vola simplification in mrSABR PAPER identical to alpha=theta=Sigma

    #print_excel (cTex_mrSABR, simplify_result=True) # simplify again (might help a bit coz of substitutions)


    ######### cTex for mrZABR
    print("")
    print ("        _")
    print ("mrZABR: c(T_ex):")

   # full integrand
    cTex_int_mrZABR_intgd = 2 * alpT_mrZABR * I2T_mrZABR + I1T_mrZABR ** 2 + 4 * alpT_mrZABR * I4T_mrZABR

    # sometimes formulas get too lengthy so evaluate splitted versions
    print_expression_title ("  part 1 (c_int) subpart 1", 50)
    cTex_int1_mrZABR = integrate_sum(  (2 * alpT_mrZABR * I2T_mrZABR).expand(), (T, 0, Tex), simplify_terms=True)
    cTex_int1_mrZABR = cTex_int1_mrZABR.simplify()
    print_expression_value (cTex_int1_mrZABR)
    #print_excel (cTex_int1_mrZABR, simplify_result=False)
    if DO_CHECKS:
        # this is the most difficult subpart, as it involves triple nested
        #4 * alpT_mrZABR * I4t_mrZABR_raw.subs(t, T) / 2
        # make triple nested object
        INT0 = I2t_mrZABR_raw.subs(t, T)
        INT1 = Integral ( 2 * alpT_mrZABR *  INT0.function / 2,  INT0.limits[0],  INT0.limits[1] )
        INT2 = Integral (INT1, (T, 0, Tex))
        print_eval_stats(cTex_int1_mrZABR, INT2)



    print_expression_title ("  part 1 (c_int) subpart 2", 50)
    cTex_int2_mrZABR =integrate_sum((I1T_mrZABR ** 2 ).expand(), (T, 0, Tex), simplify_terms=True)
    cTex_int2_mrZABR = cTex_int2_mrZABR.simplify()
    print_expression_value (cTex_int2_mrZABR)
    #print_excel (cTex_int2_mrZABR, simplify_result=False)
    if DO_CHECKS:
        INT0 = I1t_mrZABR_raw.subs(t, T)
        INT2 = Integral(INT0, (T, 0, Tex))
        print_eval_stats(cTex_int2_mrZABR, INT2, fn_inner= lambda x: x**2)
        # here the advantage symbolic is just 10x (numeric basically only double nested with square)




    # sympy integrate + simplify not optimal fot the squared polynomial I1T_mrZABR to be integrated
    # some 50% economy in length / time when using alternative symbolic derivation via the below methods

    print_expression_title("    alternative derivation (matrix sq pol)", 50)
    try:
        I1T_mrZABR_indef = antiderivative_square_polyexp(I1T_mrZABR, T)
        cTex_int2_mrZABR_alt1 =  I1T_mrZABR_indef.subs(T, Tex) - I1T_mrZABR_indef.subs(T, 0)

        cTex_int2_mrZABR_alt2 = integrate_square_polyexp_grouped(I1T_mrZABR, T, Tex)

        # prefer 1st if 2nd is not considerably shorter:
        if len(str(cTex_int2_mrZABR_alt1)) > len(str(cTex_int2_mrZABR_alt2)) + 10:
            cTex_int2_mrZABR_alt =  cTex_int2_mrZABR_alt2
        else:
            cTex_int2_mrZABR_alt = cTex_int2_mrZABR_alt1

        cTex_int2_mrZABR_alt=cTex_int2_mrZABR_alt.simplify()

        print_expression_value (cTex_int2_mrZABR_alt)
        if DO_CHECKS:
            print_identity_check("Is alternative identical? :", cTex_int2_mrZABR_alt, cTex_int2_mrZABR)
            print_eval_stats(cTex_int2_mrZABR_alt, INT2, fn_inner=lambda x: x ** 2)
    except:
        print_expression_value("failed")
        cTex_int2_mrZABR_alt = None



    print_expression_title ("  part 1 (c_int) subpart 3", 50)
    cTex_int3_mrZABR =integrate_sum((4 * alpT_mrZABR * I4T_mrZABR).expand(), (T, 0, Tex), simplify_terms=True)
    cTex_int3_mrZABR = cTex_int3_mrZABR.simplify()
    print_expression_value (cTex_int3_mrZABR)
    #print_excel (cTex_int3_mrZABR, simplify_result=False)
    if DO_CHECKS:
        # this is the most difficult subpart, as it involves triple nested
        # 4 * alpT_mrZABR * I4t_mrZABR_raw.subs(t, T) / 2
        # make triple nested object
        INT0 = I4t_mrZABR_raw.subs(t, T)
        INT1 = Integral(4 * alpT_mrZABR * INT0.function / 2, INT0.limits[0], INT0.limits[1])
        INT2 = Integral(INT1, (T, 0, Tex))
        print_eval_stats(cTex_int3_mrZABR, INT2)


    print_expression_title ("  part 1 (c_int) all 3 subparts", 50)
    cTex_int_mrZABR_sum = cTex_int1_mrZABR + cTex_int2_mrZABR + cTex_int3_mrZABR
    try:
        # try if integration of all 3 subparts together results in shorter expression
        cTex_int_mrZABR_try = integrate_sum(cTex_int_mrZABR_intgd.expand(), (T, 0, Tex))
        #print_excel (cTex_int_mrZABR, simplify_result=False)
        cTex_int_mrZABR_try_smp = None

        #cTex_int_mrZABR_try_smp = simplify_timeout (cTex_int_mrZABR_try, minutes = 5) # with timeout
        #if cTex_int_mrZABR_try_smp is not None:
        #    cTex_int_mrZABR_try = cTex_int_mrZABR_try_smp
        cTex_int_mrZABR_try = cTex_int_mrZABR_try.simplify()

        # prefer simple sum if not considerably shorter

        if  len(str (cTex_int_mrZABR_sum)) > len(str(cTex_int_mrZABR_try)) + 10:
            cTex_int_mrZABR = cTex_int_mrZABR_try
        else:
            cTex_int_mrZABR = cTex_int_mrZABR_sum
    except:
        cTex_int_mrZABR = cTex_int_mrZABR_sum

    print_expression_value (cTex_int_mrZABR)


    if DO_CHECKS:
        print_identity_check("subparts sum up?", cTex_int_mrZABR_sum, cTex_int_mrZABR)
        print_eval_stats(cTex_int_mrZABR)
        #print_excel(cTex_int_mrZABR, simplify_result=False)
        pass


    print_expression_title ("    part 1 (c_int) with alternative derivation", 50)

    if  cTex_int2_mrZABR_alt is None :
        print_expression_value("failed")
        cTex_int_mrZABR_alt = None
    else:
        cTex_int_mrZABR_alt = cTex_int1_mrZABR + cTex_int2_mrZABR_alt + cTex_int3_mrZABR
        cTex_int_mrZABR_alt=cTex_int_mrZABR_alt.simplify()

        print_expression_value (cTex_int_mrZABR_alt)
        if DO_CHECKS:
            print_eval_stats(cTex_int_mrZABR_alt)



    print_expression_title ("c(T_ex) full", 50)
    cTex_mrZABR = (3 / tauex_mrZABR ** 3  * cTex_int_mrZABR - 3 * bTex_mrZABR ** 2)


    # try to simplify within limited time
    #try:
    #    cTex_mrZABR_try=simplify_timeout(cTex_mrZABR, minutes=5)
    #    if cTex_mrZABR_try is not None:
    #        cTex_mrZABR = cTex_mrZABR_try
    #except:
    #    pass

    print_expression_value (cTex_mrZABR)


    if DO_CHECKS:
        print_eval_stats(cTex_mrZABR)

        print_identity_check_title("consistent vs flat-expected-vola formula?")

        # formula from mrZABR PAPER for special case of flat expected vola
        cTex_mrZABR_flatVola_paper =                3*(1 + rho**2) * nu**2 * alp**(2*(gam-1)) / (2*(lam*Tex)**3) * ( 2*lam*Tex + 4*sp.exp(-lam*Tex) - 3 - sp.exp(-2*lam*Tex) )
        cTex_mrZABR_flatVola_paper = cTex_mrZABR_flatVola_paper  + 6*(1 + gam) * rho**2 * nu**2 * alp**(2*(gam-1)) / (lam*Tex)**3  * (lam*Tex + 2*sp.exp(-lam*Tex) - 2 + lam*Tex*sp.exp (-lam*Tex) )
        # original term in mrZABR PAPER:
        #cTex_mrZABR_flatVola_paper= cTex_mrZABR_flatVola_paper  - 12* rho**2 * nu**2 * alp**(2*(gam-1))  * (lam*Tex - 1 + sp.exp (-lam*Tex) ) / (lam*Tex)**2
        # corrected term: teh paper was missing power of 2
        cTex_mrZABR_flatVola_paper = cTex_mrZABR_flatVola_paper  - 12* rho**2 * nu**2 * alp**(2*(gam-1))  * ((lam*Tex - 1 + sp.exp (-lam*Tex) ) / (lam*Tex)**2)**2

        # c(T_ex) in mrZABR PAPER is already after multiplication with the term (tauex/Tex) which is applied here later
        cmp1 = trf_make_flat (cTex_mrZABR * (tauex_mrZABR / Tex) , alp).simplify().expand()
        cmp2 = cTex_mrZABR_flatVola_paper.subs(gam, mrZABR_GAMMA).simplify().expand()
        print_identity_check_result (cmp1, cmp2)

        cmp1 = (cTex_mrSABR_flatVola_paper *sig**2).expand() # multiply by  (tauex_mrSABR / Tex) , as it is contained in cTex in mrZABR PAPER
        cmp2 = (cTex_mrZABR_flatVola_paper.subs(alp, sig).subs(gam,1).simplify()).expand()
        print_identity_check ("mrZABR PAPER formula for gamma=1 reduces to mrSABR PAPER formula (flat-expected-vola)?", cmp1, cmp2)

        # this is the deepest integral in the model, so the performance here dominates the general performance
        # define the integrand using unevaluated / raw expressions:
        cTex_int_mrZABR_RAW = 2 * alpT_mrZABR * I2t_mrZABR_raw.subs(t, T) / 2 + I1t_mrZABR_raw.subs(t, T) ** 2 + 4 * alpT_mrZABR * I4t_mrZABR_raw.subs( t, T) / 2




        print(f"{CLR_check}\nTry a few cases with full numeric integration for cTex_int (no optimal lambdification):")
        for cs in range(0, 5):
            prm_rnd = get_random_parameters()

            prm_rnd = {p: v for p, v in prm_rnd.items() if cTex_int_mrZABR.has(p) or (p != T and cTex_int_mrZABR_RAW.has(p)) } # T gets integrated away below
            print("\nrandom parameters:", prm_rnd)

            cTex_int_mrZABR_value = cTex_int_mrZABR.subs(prm_rnd).evalf()
            # analytical evaluation for fixed values (here without lambdification, as was already evaluated with lambdification above)
            val0 = float(cTex_int_mrZABR_value)
            print("mrZABR: c(T_ex) part 1 symbolic:                                  ", f"{val0:.2e}")

            expiry = prm_rnd [Tex] # extract expiry from random parameters
            # input all random parameters except T which will be the integration variable
            I_RAW = cTex_int_mrZABR_RAW.subs(T, zzz)
            I_RAW = I_RAW.subs (prm_rnd)
            I_RAW = I_RAW.subs(zzz, T)
            I_RAW_INT = Integral (I_RAW, (T, 0, expiry)) # define final integral
            #I_RAW_INT = I_RAW_INT.simplify()

            print("mrZABR: c(T_ex) part 1 numeric via umlimied recursive (not lambd.)", "...", end="")
            start=time.time()
            val = num_integration_recursive_unlimited ( I_RAW_INT)
            dev =  get_rel_deviation (val, val0)

            print("\b\b\b", f"{CLR_ok if dev < 0.01 else CLR_alarm if dev > 0.1 else CLR_check}{float(val):.2e}{CLR_check}, eval time: {time.time() - start:.2f} secs")

            print("mrZABR: c(T_ex) part 1 numeric via Simpson's rule:                ", "...", end="")
            start = time.time()
            val = num_integration_simpson (I_RAW_INT, useScipy=True, n=30)
            dev =  get_rel_deviation (val, val0)

            print("\b\b\b", f"{CLR_ok if dev < 0.01 else CLR_alarm if dev > 0.1 else CLR_check}{float(val):.2e}{CLR_check}, eval time: {time.time() - start:.2f} secs")

            # numeric takes around 1 sec !!!

        print (f"{CLR_text}")

    #print_excel (cTex_mrZABR, simplify_result=False)





    print("*++++++++++++++++**************************")
    print("* other coefficients / checks**************")
    print("*++++++++++++++++**************************")

    print_expression_title ("mrSABR: G_int, integral over I5(T)", 50)
    Gint_mrSABR = integrate (I5T_mrSABR, (T, 0, Tex) ).simplify()
    print_expression_value (Gint_mrSABR)

    if DO_CHECKS:
        Gstar_mrSABR = -1 / 2 * cTex_mrSABR * tauex_mrSABR + sig ** 2 / tauex_mrSABR * Gint_mrSABR
        # Note  that G* is defied without G0 term, only G~ with G0 term

        print_identity_check_title ("mrSABR GStar vs formula in mrSABR PAPER (flat vola)")
        GStar_mrSABR_flatVola_paper = (- cTex_mrSABR_flatVola_paper /2 + nu**2/sig**2 * (2*lam*Tex -1 + sp.exp(-2*lam*Tex)) / (4*lam**2*Tex**2))* sig**2*Tex
        GStar_mrSABR_flatVola_paper = GStar_mrSABR_flatVola_paper.simplify()
        Gstar_flatVola1 = Gstar_mrSABR.subs(sig, 1).subs(alp, sig).subs(the, sig).simplify()
        Gstar_flatVola2 = Gstar_mrSABR.subs(alp, 1).subs(the, 1).simplify()

        print_identity_check_result (GStar_mrSABR_flatVola_paper, Gstar_flatVola1)
        print_identity_check("mrSABR GStar vs formula in mrSABR PAPER (flat vola)", GStar_mrSABR_flatVola_paper, Gstar_flatVola2)
        # GStar_mrSABR_flatVola_paper.free_symbols
        # in case of flat vola and constant sigma, Gstar does not depend on vola


    print_expression_title ("mrZABR: G_int, integral over I5(T)", 50)
    Gint_mrZABR = integrate_sum(I5T_mrZABR.expand(), (T, 0, Tex)).simplify()
    print_expression_value (Gint_mrZABR)

    # α_std assuming same logic as in mrSABR and sig=1
    Gstar_mrZABR = -1 / 2 * cTex_mrZABR * tauex_mrZABR + 1/ tauex_mrZABR * Gint_mrZABR
    alphaStd_mrZABR = sp.sqrt(tauex_mrZABR / Tex) * sp.exp(1 / 2 * Gstar_mrZABR)

    if DO_CHECKS:

        # in mrZABR PAPER: nuStd = sqrt (cTex)
        # i.e. the definition of cTex is there already upon multiplcation with delsq
        # in mrZABR PAPER: rhoStd unchanged
        # i.e. the definition of bTex is there already upon multiplcation with sqrt ( delsq)
        # delsq is average expected variance, from 0 until expiry

        # derived independently:
        Gbar_mrZABR = -1 / 2 * (G0 * bTex_mrZABR + cTex_mrZABR) * tauex_mrZABR + 1 / tauex_mrZABR * Gint_mrZABR
        # Gbar_mrZABR = Gbar_mrZABR.simplify()

        # check flat vola Gbar from mrZABR PAPER
        Gbar_mrZABR_flatVola_paper =  nu**2 * alp**(2*(gam-1))  /  (4*lam**2 *Tex)  * (2*lam*Tex + sp.exp(-2*lam*Tex)  -1)   \
                          - 1/2* cTex_mrZABR_flatVola_paper * Tex  - rho * nu * alp**gam / (lam**2*Tex) * (lam*Tex  - 1 +sp.exp(-lam*Tex))*G0

        print_identity_check_title ("Identity Gbar vs mrZABR PAPER")
        cmp1 = trf_make_flat (Gbar_mrZABR, the).simplify()
        cmp2 = Gbar_mrZABR_flatVola_paper.subs(gam, mrZABR_GAMMA).subs(alp, the).simplify()
        print_identity_check_result (cmp1, cmp2)


        print("")
        print("")
        print (f"{CLR_check}Difference α_std: mrSABR PAPER logic (as also used in this PAPER) vs mrZABR PAPER")
        # in mrZABR PAPER: formula 6: obvious difference vs mrSABR PAPER: standardized alpha in mrZABR PAPER is based on inst. vola, not on avg vola as in mrSABR PAPER
        # note also that it uses bTex term instead of cTex term in mrSABR PAPER
        # mrZABR PAPER approach appears questionable as it does not seem to average btw alpha and theta to arrive at the standardized alpha

        # investigate differences for random parameters:

        #alphaStd_mrZABR_flatVola_paper = alp * (1 + 1/2*Gbar_mrZABR_flatVola_paper + 1/4 * alp * bTex_mrZABR_flatVola_paper * G0 * Tex)
        #  mrSABR PAPER version:
        alphaStd_mrZABR_paper = alp * (1 + 1 / 2 * Gbar_mrZABR + 1 / 4 * alp * bTex_mrZABR* G0 * Tex)

        # check values for a number of random parameter sets
        for i in range (0, 20):
            prm=get_random_parameters()
            cmp1 = alphaStd_mrZABR
            #cmp1 = alphaStd_mrZABR.subs(alp, the) # reduce mrSABR-analogous solution to  flat vola at theta

            cmp2 = alphaStd_mrZABR_paper.subs(gam, mrZABR_GAMMA).subs(G0, -1)
            #cmp2 = alphaStd_mrZABR_flatVola_paper.subs(gam, mrZABR_GAMMA).subs(G0, -1)
            #cmp2=cmp2.subs(alp, the) # same for mrZABR solution

            prm = {p:v  for p, v in prm.items() if cmp1.has(p) or cmp2.has(p)}

            cmp1 = cmp1.subs(prm)
            cmp2 = cmp2.subs(prm)
            dif= get_rel_deviation(cmp1, cmp2)
            print(f"{CLR_check}parameters:", prm)
            if dif <0.1:
                print(f"{CLR_check}   alphaStd as in mrSABR PAPER:{CLR_ok}", f"{cmp1:.2f}", f"{CLR_check}   alphaStd mrZABR PAPER:{CLR_ok}", f"{cmp2:.2f}")
            else:
                print(f"{CLR_check}   alphaStd as in mrSABR PAPER:{CLR_ok}", f"{cmp1:.2f}", f"{CLR_check}   alphaStd mrZABR PAPER:{CLR_alarm}", f"{cmp2:.2f}")

        print("\n", f"{CLR_check}!!! This PAPER proceeds with mrSABR PAPER definition of standardized alpha")

        print(f"{CLR_text}")

        # small differences ONLY if alpa and theta are the same (flat vola)
        # mrZABR PAPER seems to focus on this special case and does not embed theta at all
        # mrSABR PAPER definition was used in this PAPER as it appears more natural (interpolates btw alpha and theta)






    print("")
    print("")

    print ("*****************************************************************************")
    print( "* exporting closed-form solutions (effective coefficients) ++++++++++++++++++")
    print ("*****************************************************************************")

    # strictly speaking, cTex output is not required for further calculations of standardized coefficients,
    # as cTex can be derived from tauex, bTex and cTex_int
    # Nonetheless, output it here, as it might be beneficial to minimize the number of calc steps e.g. in Excel




    hSABR_results ={"tauex": tauex_hSABR, "bTex": bTex_hSABR,  "cTex": cTex_hSABR, "cTex_int": cTex_int_hSABR, "Gint" : sp.Integer(0)} # Gint empty in hSABR
    with open("results/hSABR_expressions.pkl", "wb") as f:   # "wb" = write binary
        pickle.dump(hSABR_results, f)


    mrSABR_results ={"tauex": tauex_mrSABR, "bTex": bTex_mrSABR, "cTex": cTex_mrSABR, "cTex_int": cTex_int_mrSABR, "Gint" : Gint_mrSABR}
    with open("results/mrSABR_expressions.pkl", "wb") as f:   # "wb" = write binary
        pickle.dump(mrSABR_results, f)


    # ZABR setttings
    settings={"gamma": mrZABR_GAMMA, "expansion_mode": mrZABR_EXPNS, "expansion_order": mrZABR_ORDER}
    mrZABR_results_this={"tauex": tauex_mrZABR, "bTex": bTex_mrZABR, "cTex": cTex_mrZABR,  "cTex_int": cTex_int_mrZABR, "cTex_int_alt": cTex_int_mrZABR_alt, "Gint" : Gint_mrZABR}

    if mrZABR_APPRX and not mrZABR_EXACT:
        mrZABR_results={}
        # load ZABR solutions (can store different versions depending on settings)
        try:
            with open("results/mrZABR_expressions.pkl", "rb") as f:   # "rb" = read binary
                mrZABR_results = pickle.load(f)
        except:
            mrZABR_results={}
        # update the current version
        mrZABR_results [ frozenset(settings.items())  ] = mrZABR_results_this
        # save
        with open("results/mrZABR_expressions.pkl", "wb") as f:   # "wb" = write binary
            pickle.dump(mrZABR_results, f)

    print("")
    print ("*****************************************************************************")
    print( "* END ********************************************************+++++++++++++++")
    print ("*****************************************************************************")

