"""
Loads a chosen set of result expressions (effective coefficients) as generated/saved in *derive_validate_expressions.py*.
From these, the code derives standardized SABR coefficients,
and applies standard SABR model to derive implied Black & Scholes volatilities.
The code then optimizes original mean-reverting SABR-based parameter values
(*alpha*, *theta*, *lambda*, *nu*, *rho*) to best fit an empirical dataset of implied volatility quotes.
Finally, the code generates an interactive 3D surface, which interpolates and extrapolates the empirical quotes provided.
"""


import mpmath
import math
import numpy
import numpy as np

import sympy as sp
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt
import pickle
import time
from   scipy.optimize import minimize
import sys

gam = sp.symbols("gam", positive=True)  # alpha



#####################################################################################################################
# SETTINGS    #######################################################################################################
#####################################################################################################################

##################################################################################################
# DATA for empirical implied volas, to fit the surface to
##################################################################################################



####################################
# example with low current vola
####################################
DATA_IV = np.array([
# first colulmn:
# Experies (in years)
[np.nan,    0.8,    0.9,    1.0,    1.1,    1.2],# first row: MONEYNESS = Strike / Forward

[9/12,      0.22,   0.18,   0.15,   0.13,   0.12],
[3/12,      0.25,   0.18,   0.13,   0.11,   0.13],
[1/12,      0.32,   0.20,   0.11,   0.13,   0.18]
])# np.nan if data missing / unavailable



#####################################
# example with mid-level current vola
#####################################
DATA_IV = np.array([
# first colulmn:
# Experies (in years)
[np.nan,    0.8,    0.9,    1.0,    1.1,    1.2],# first row: MONEYNESS = Strike / Forward

[9/12,      0.24,   0.21,   0.17,   0.14,   0.13],
[3/12,      0.29,   0.22,   0.17,   0.13,   0.14],
[1/12,      0.36,   0.25,   0.17,   0.15,   0.20]
])# np.nan if data missing / unavailable



#####################################
# example with high current vola
#####################################
DATA_IV = np.array([
# first colulmn:
# Experies (in years)
[np.nan,    0.8,    0.9,    1.0,    1.1,    1.2],# first row: MONEYNESS = Strike / Forward

[9/12,      0.35,   0.30,   0.26,   0.22,   0.20],
[3/12,      0.44,   0.37,   0.30,   0.24,   0.22],
[1/12,      0.54,   0.46,   0.37,   0.28,   0.27]
])# np.nan if data missing / unavailable




##################################################################################################
# end of DATA for empirical implied volas
##################################################################################################



##################################################################################################
# bounds for fitting mean-reverting parameters: alpha, theta, lambda, nu, rho
##################################################################################################
BOUNDS = [(0.02, 2), (0.02, 2), (0.1, None), (0.1, None), (-1, 1) ]


##################################################################################################
# Definition of moneynesss, as depicted on X-axis of 3D surface plots
##################################################################################################
MONEYNESS_SCALE = 0.0
# Moneyness is defined as  log (Strike / Forward) / ( Expiry ** MONEYNESS_SCALE )
# typical values for MONEYNESS_SCALE are 0.5 (log & expiry-root scaling) and 0 (just log scaling)
# 0.5 often produces surfaces whose constituent smiles are more similar across expiries


###################################################################################################
#  mean-reverting SABR-based MODEL for which the cloased-form expressions shoudl be loaded
###################################################################################################

MODEL = "hSABR"
MODEL = "mrSABR"

MODEL = "mrZABR"

# SETTINGS_mrZABR: used to select/load desired mrZABR model version (ignored for hSABR and mrSABR)
# if the desired model version is not found, available versions will be displayed as a list

# CIR-type porcess for vola, Rational (1, 2) = 1/2
SETTINGS_mrZABR = {"gamma" : sp.Rational (1, 2),            "expansion_mode": 3, "expansion_order": 6}
#SETTINGS_mrZABR = {"gamma" : sp.Rational (1, 2),            "expansion_mode": 3, "expansion_order": 5}
#SETTINGS_mrZABR = {"gamma" : sp.Rational (1, 2),            "expansion_mode": 3, "expansion_order": 4}
#SETTINGS_mrZABR = {"gamma" : sp.Rational (1, 2),            "expansion_mode": 1, "expansion_order": 6} # expansion_mode 1 fails for high current vola
#SETTINGS_mrZABR = {"gamma" : sp.Rational (1, 2),            "expansion_mode": 2, "expansion_order": 6} # expansion_mode 2 fails for low current vola

# mean-reverting CEV-type process for vola with fixed CEV parameter.e.g. 3/4 for Rational (3, 4)
#SETTINGS_mrZABR = {"gamma" : sp.Rational (1, 4),            "expansion_mode": 3, "expansion_order": 6}
#SETTINGS_mrZABR = {"gamma" : sp.Rational (1, 3),            "expansion_mode": 3, "expansion_order": 6}
#SETTINGS_mrZABR = {"gamma" : sp.Rational (2, 3),            "expansion_mode": 3, "expansion_order": 6}
#SETTINGS_mrZABR = {"gamma" : sp.Rational (3, 4),            "expansion_mode": 3, "expansion_order": 6}
#SETTINGS_mrZABR = {"gamma" : sp.Integer (1),                "expansion_mode": 3, "expansion_order": 6}  # effectively reduces mrZABR to mrSABR
#SETTINGS_mrZABR = {"gamma" : sp.Rational (4, 3),            "expansion_mode": 3, "expansion_order": 6}
#SETTINGS_mrZABR = {"gamma" : sp.Rational (3, 2),            "expansion_mode": 3, "expansion_order": 6}

# mean-reverting CEV-type process for vola with  CEV parameter as additional "gam" parameter
#SETTINGS_mrZABR = {"gamma" : gam, "gamma_value": 0.75 , "expansion_mode": 3, "expansion_order": 6}
#SETTINGS_mrZABR = {"gamma" : gam, "gamma_value": 0.50 , "expansion_mode": 3, "expansion_order": 5}
#SETTINGS_mrZABR = {"gamma" : gam, "gamma_value": 1.25 , "expansion_mode": 3, "expansion_order": 4}

# !!! Expansion mode 1 might severely fail if instantaneous vola (alpha) is ABOVE average
# But it will be good enough (compact and precise) for high orders if current vola is under-average
# Mode 2 might severely fail if the instantaneous vola (alpha) is significantly UNDER average
# But it will be good enough  (compact and precise) for high orders  if current vola is above-average
# Mode 3 can be used always (the closed-form expressions are longer, though)

# gamma set to gam loads expressions for general case where gamma is the additional, CEV-type parameter
# For these models, gamma_value needs to be preset additionally
# In principle, gamma, if parametrized in the expressions, can be treated as other parameters
# and then be optimized / fitted to empirical vola data
# For a single surface, however, there would be  a very high correlation / redundancy vs nu parameter
# which might result in estimation difficulty / instability
# Instead, to estimate gamma, several empirical surfaces from multiple observations times (üreferably with both high and low current volas) should be fitted simultaneously
# In the code below, for simplicity, gamma parameter is just replaced  with the preset value

#########################################################
# set library for lambdification of the model expressions
#########################################################
#LIB_MATH = "mpmath" # slower but higher precision. eaxct precision can be steered via e.g.: mpmath.mp.dps = 50
LIB_MATH = "math"    # lower precision, faster on scalars, slower on vectors, for extreme parameter values might lead to overflows
#LIB_MATH = "numpy"  # same precision as numpy but for vectors

#####################################################################################################################
# END OF SETTINGS    ################################################################################################
#####################################################################################################################


alp = sp.symbols("alp", positive=True)  # alpha
alps = sp.symbols("alps", positive=True)  # alpha
the = sp.symbols("the", positive=True)  # theta
thes = sp.symbols("thes", positive=True)  # theta
lam = sp.symbols("lam", positive=True)  # lambda
rho = sp.symbols("rho", real=True)  # rho
nu = sp.symbols("nu", positive=True)  # nu
Tex = sp.symbols("T_ex", positive=True)  # nu
sig = sp.symbols("sig", positive=True)  # nu
dlt = sp.symbols("dlt", real=True)  # alpha
mu = sp.symbols("mu", positive=True)  # alpha


def calc_std (alp, the, lam, nu, rho, Tex, action_rho="restrict"):
    """
    calculates 3 standard-SABR parameters from 5 mean-reverting parameters + expiry
    """
    if MU_DELTA:
        prm1 = (alp+the)/2
        prm2 = (alp-the)/(alp+the)
    else:
        prm1 = alp
        prm2 = the

    if LIB_MATH == "math":
        inp_vars = [float(prm1), float(prm2), float(lam), float(nu), float(rho), float(Tex)]
    elif LIB_MATH == "mpmath":
        inp_vars = [mpmath.mpf(prm1), mpmath.mpf(prm2), mpmath.mpf(lam), mpmath.mpf(nu), mpmath.mpf(rho), mpmath.mpf(Tex)]
    elif LIB_MATH == "numpy":
        inp_vars = [np.float64(prm1), np.float64(prm2), np.float64(lam), np.float64(nu), np.float64(rho), np.float64(Tex)]

    Tex0 = Tex

    # use simple math precision from this point (no more lengthy expressions where numerical precision might matter)
    tauex0 = float ( get_tauex       (*inp_vars))
    bTex0  = float (get_bTex         (*inp_vars))
    Gint0  = float (get_Gint         (*inp_vars))
    cTex_int0 = float (get_cTex_int  (*inp_vars))


    # calculate standard-SABR parameters from the expressions
    cTex0     = 3 / tauex0 ** 3 * cTex_int0 - 3 * bTex0 ** 2

    # at extreme parameter values (particularly for lower-precision math lib), cTex might get negative
    # this collapses the solutions because of the square roots, intercept that here:
    if cTex0 <= 0:
        print("negative cTex for inputs: ", [alp, the, lam, nu, rho, Tex])
        if action_rho=="raise":
            assert False
        else:
            return (float('nan'), float('nan'), float('nan'))


    alphaStd0 = math.sqrt(tauex0 / Tex0) * math.exp (- cTex0 * tauex0 / 4 + Gint0 / 2 / tauex0)
    rhoStd0  = ( bTex0 ) / math.sqrt (cTex0)
    nuStd0 = math.sqrt(tauex0 / Tex0)  * math.sqrt (cTex0 )


    # in case of fitting to data, gradient-search algorithm sometimes needs to pass via a region with std rho outside [-1,1]
    # restrict std rho in such cases
    # the final/optimal solution will then be with std rho within [-1,1] as expected

    if rhoStd0 > 0.999:
        if action_rho=="restrict":
            rhoStd0 = 0.999
        elif action_rho=="raise":
            print("rho >=1 for inputs:", inp_vars)
            assert False
        else:
            print("rho >=1 for inputs:", inp_vars)
            return (float('nan'), float('nan'), float('nan'))
    if rhoStd0 < -0.999:
        if action_rho=="restrict":
            rhoStd0 = -0.999
        elif action_rho=="raise":
            print("rho <=-1 for inputs:", inp_vars)
            assert False
        else:
            print("rho <=-1 for inputs:", inp_vars)
            return (float('nan'), float('nan'), float('nan'))

    # until now, sympy expressions
    return (float(alphaStd0), float(nuStd0), float(rhoStd0))


def get_IV (stdParams, moneyness, expiry):
    """
    standard SABR model: implied vola from 3 standard-SABR parameters
    """
    alpha_std = stdParams[0]
    nu_std    = stdParams[1]
    rho_std   = stdParams[2]
    try:
        z =  nu_std / alpha_std * math.log (1 /moneyness)
        x = math.log (    (math.sqrt (1-2*rho_std*z + z**2)   + z -rho_std)  / (1-rho_std)  )
        if moneyness ==1:
            IV =  alpha_std * ( 1   +     (1/4*rho_std* nu_std * alpha_std   +    (2 - 3*rho_std**2)/24 * nu_std**2)    *expiry)
        else:
            IV = z / x * alpha_std * ( 1   +     (1/4*rho_std* nu_std * alpha_std   +    (2 - 3*rho_std**2)/24 * nu_std**2)    *expiry)
    except:
        print ("numeric error resulting from std parameters:", alpha_std, nu_std, rho_std)
        assert False

    return IV


def get_residual_sum (params, data):

    alp, the, lam, nu, rho = params
    rows = data.shape[0]
    cols = data.shape[1]
    expiry = None
    std = None
    resid = 0
    for r in range(rows):
        for c in range(cols):
            el = data[r, c]
            if r == 0:
                continue
            if c == 0:
                expiry = el # expiry in left column
                std = calc_std(alp, the, lam, nu, rho, expiry, action_rho="restrict")
                continue
            if np.isnan(el)==False:
                mns = data[0, c] # moneyness in top row
                iv = float (get_IV(std, mns, expiry))
                resid = resid + (iv-el)**2
    return resid

def fit_parameters (data):
    print ("")
    print ("fitting to data...")
    print("*" * 50)
    start = time.time()

    #init = [data[3, 3], 2* data[1,3] - data[3, 3] , 3, 1, -0.7] # set alpha to short-term, theta - to longer term
    vola_long = data[1, 3]
    vola_short = data[3, 3]
    #if MU_DELTA:
        #init = [  (vola_short + vola_long) /2, (vola_short - vola_long) / (vola_short + vola_long),   3, 1, -0.7  ]  # set alpha to short-term, theta - to longer term
    #    init = [  (vola_short + vola_long) /2, 0,   3, 1, -0.7  ]  # set alpha to short-term, theta - to longer term
    #else:
    #    #init = [ vola_short, vola_long , 3, 1, -0.7  ]  # set alpha to short-term, theta - to longer term
    #    init = [(vola_short + vola_long) /2, (vola_short + vola_long) /2, 3, 1, -0.7]  # set alpha to short-term, theta - to longer term
    init = [(vola_short + vola_long) /2, (vola_short + vola_long) /2, 1, 1, -0.5]  # set alpha to short-term, theta - to longer term
    bounds = BOUNDS
    print ("Initial:", init)
    result = minimize(get_residual_sum, init, bounds=bounds, args=(data,))
    print (result)
    print ("")
    print ("*"*50)
    #if MU_DELTA:
    #    print("RESULTS", [mu, dlt, lam, nu, rho], ":  ", result.x)
    #else:
    print ("RESULTS", [alp, the, lam, nu, rho], ":  ", result.x)
    el_cnt = np.count_nonzero(~np.isnan( DATA_IV[1:, 1:]))
    print (f"RMSE: {math.sqrt (result.fun / el_cnt  )*100:.2f} % on {el_cnt} data points")
    print (f"fitting took {(time.time() - start)*1000:.2f}", "milliseconds")
    return  result.x



def calc_XYZ (alp, the, lam, nu, rho, Moneyness_Range=(0.8, 1.2), Expiry_Range=(0.1, 2), Moneyness_N=50, Expiry_N=50):
    """
    prepares/generates data for 3D surfaces
    """

    # Create grid of scaled moneyness to dsiplay on X
    x_trf = np.linspace(*Moneyness_Range, Moneyness_N)

    # non-scaled expiry
    y = np.linspace(*Expiry_Range, Expiry_N)

    X_trf, Y = np.meshgrid(x_trf, y)


    # infer unscaled moneyness ( = Strike / Forward)
    X= X_trf.copy()
    for sc in range (len(y)):
        X[sc] =  np.exp(  X[sc]  *   ( y[sc] ** MONEYNESS_SCALE ) )


    print("")
    print("*" * 50)
    print ("Precompute "+str(y.size)+" standard-SABR parameter sets for each expiry (y values) via mean-rev. SABR-based model")
    start = time.perf_counter()
    std_values = {yi: calc_std (alp, the, lam, nu, rho, yi) for yi in y}
    # here nan values possible if calculations failed
    print ("avg", f"{(time.perf_counter()  - start)/y.size *1e3:.3f}", "milliseconds / expiry")
    print ("\nMIN / MAX values for std parameters:")
    alphas = [v[0] for v in std_values.values()]
    nus = [v[1] for v in std_values.values()]
    ros = [v[2] for v in std_values.values()]
    print ("alpha:", min(alphas), max (alphas))
    print ("nu:   ", min(nus), max (nus))
    print ("rho:  ", min(ros), max (ros))

    print("")
    print("*" * 50)
    print("Calculate implied volas using precomputed standard SABR parameters + standard SABR model for " , X.size  ,"moneyness/expiry grid points")
    start = time.time()
    Z = np.zeros_like(X)
    try:
        for i in range(Expiry_N):
            for j in range(Moneyness_N):
                std = std_values [Y[i, j]]
                Z[i, j] = get_IV(std, X[i, j], Y[i, j])
                assert math.isnan(Z[i, j])==False
    except:
        Z[i, j] = 0 # in case of any troubles set implied vola to 0 to clearly mark it
    print (f"{(time.time() - start)*1000:.2f}", "milliseconds")
    print ("")
    print("*" * 50)

    return X_trf, Y, Z


def extract_dataXYZ (empData):
    """
    prepares/generates data for 3D surfaces
    """
    xyz =[]
    xs = empData[0, 1:]
    ys = empData[1:, 0]
    Z  = empData[1:, 1:]

    for y0 in range(ys.size):
        xss = np.log (xs) /  (ys[y0]) ** MONEYNESS_SCALE
        yss= xs.copy()
        yss[:] =ys[y0]

        # array of smiles
        xyz.append ( (xss, yss, Z[y0, :]) )

    return xyz



def disp_plotly (X, Y, Z, empData=None):

    #  fitted (and extrapolated surface)
    surface = go.Surface(z=Z, x=X, y=Y, colorscale="Viridis",  contours = {
                "x": {"show": True, "color": "grey", "width": 1},  # vertical slices
                "y": {"show": True, "color": "grey", "width": 1},  # horizontal slices
                "z": {"show": False}  # optional: hide z-contours
            })

    # empirical data provided
    scatters =[]
    if empData is not None:
        for xyz in empData:
            scatter = go.Scatter3d(
            x=xyz[0], y=xyz[1], z=xyz[2],
            mode="lines+markers",
            line=dict(color="blue", width=4),
            marker=dict(size=6, color="blue"),
            name="Line 1"
            )
            scatters.append (scatter)

    fig = go.Figure(data=[surface] + scatters)

    fig.update_layout(
        title="Fitted " + MODEL + " model" + (", settings: " + str(SETTINGS_mrZABR) if MODEL == "mrZABR" else "")  ,
        scene=dict(
            xaxis=dict(
                title="Moneyness = ln (Strike/Fwd)" + ((" / T^" + str(MONEYNESS_SCALE)) if  MONEYNESS_SCALE >0 else ""),
                showgrid=True,  # enable grid
                gridcolor="lightgray",
                zeroline=True,
                zerolinecolor="black"
            ),
            yaxis=dict(
                title="Expiry (T), years",
                showgrid=True,
                gridcolor="lightgray",
                zeroline=True,
                zerolinecolor="black"
            ),
            zaxis=dict(
                title="Implied Volatility (B&S)",
                showgrid=True,
                gridcolor="lightgray",
                zeroline=True,
                zerolinecolor="black"
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1.0),
            camera=dict(
                eye=dict(x=1.0, y=-2.0, z=0.5)
                # x<0 → look from negative X (X goes left)
                # y>0 → look from positive Y (Y goes right)
                # z sets height
            )

        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    #fig.show()  # Or wrap in Dash via dcc.Graph
    pio.write_html(fig, file="results/surface.html", auto_open=True)


def disp_matplotLib (X, Y, Z, empData=None):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    #  fitted (and extrapolated surface)
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # empirical data provided
    if empData is not None:
        for xyz in empData:
            ax.plot( xyz[0], xyz[1], xyz[2], color='blue', marker="o")

    ax.set_xlabel("Moneyness = ln (Strike/Fwd)" + ((" / T^" + str(MONEYNESS_SCALE)) if  MONEYNESS_SCALE >0 else ""))
    ax.set_ylabel('Expiry (T), years')
    ax.set_zlabel('Implied Volatility (B&S)')

    plt.title ("Fitted " + MODEL + " model" + (", settings: " + str(SETTINGS_mrZABR) if MODEL == "mrZABR" else ""))

    plt.show()

#############################################################################################################
#############################################################################################################





if __name__ == "__main__":

    ###########################################################################
    # load effective coefficients

    expressions = None
    if MODEL == "hSABR":
        # load dataset with mrSABR symbolic solutions
        with open("results/hSABR_expressions.pkl", "rb") as f:
            results = pickle.load(f)
        expressions = {k: v.subs(sig, 1).subs(alps, alp ** 2).subs(thes, the ** 2) for k, v in results.items()}

    elif MODEL == "mrSABR":
        # load dataset with SABR symbolic solutions
        with open("results/mrSABR_expressions.pkl", "rb") as f:
            results = pickle.load(f)
        expressions = {k: v.subs(sig, 1) for k, v in results.items()}

    elif MODEL == "mrZABR":
        # load dataset with ZABR symbolic solutions
        with open("results/mrZABR_expressions.pkl", "rb") as f:
            results = pickle.load(f)

        # choose solution with desired precision etc
        settings_target = {"gamma": SETTINGS_mrZABR["gamma"], "expansion_mode": SETTINGS_mrZABR["expansion_mode"],
                           "expansion_order": SETTINGS_mrZABR["expansion_order"]}
        try:
            expressions = results[frozenset(settings_target.items())]
        except:
            print("ONLY FOLLOWING mrZABR VARIANTS HAVE BEEN STORED:")
            for k in results.keys():
                print(sorted(k))
            sys.exit()
    else:
        assert False, "incorrect model"

    print("")
    print("**************************************************")
    print("Symbolic expressions loaded, with following lengths (in characters):")

    # load formulas from the solution
    tauexF = expressions["tauex"]
    print("tauex:        ", len(str(tauexF)))
    GintF = expressions["Gint"]
    print("Gint:         ", len(str(GintF)))
    bTexF = expressions["bTex"]
    print("bTex:         ", len(str(bTexF)))

    cTex_intF = expressions["cTex_int"]  # major variant

    # alternative cTex_intF variant
    # should be mathematically equivalent but more succinct in many cases (optimized integration in cTex integral)
    # supposed to speed up in these cases
    # not yet thoroughly tested,though
    try:
        cTex_intF_alt = expressions["cTex_int_alt"]
        # use alternative if considerably shorter
        if len(str(cTex_intF)) > len(str(cTex_intF_alt)) + 10:
            cTex_intF = cTex_intF_alt
            print("cTex_int:     ", len(str(cTex_intF)), " (alternative variant used)")
        else:
            print("cTex_int:     ", len(str(cTex_intF)))
    except:
        print("cTex_int:     ", len(str(cTex_intF)))
        pass




    ###########################################################################
    # lambdification

    MU_DELTA = False
    if cTex_intF.has(dlt):
        print("MU-DELTA re-parametrization is used in expressions")
        inp_vars = [mu, dlt, lam, nu, rho, Tex]
        MU_DELTA = True
    else:
        inp_vars = [alp, the, lam, nu, rho, Tex]

    if cTex_intF.has(gam):
        print("GAMMA set to", SETTINGS_mrZABR["gamma_value"])
        tauexF = tauexF.subs(gam, SETTINGS_mrZABR["gamma_value"])
        bTexF = bTexF.subs(gam, SETTINGS_mrZABR["gamma_value"])
        cTex_intF = cTex_intF.subs(gam, SETTINGS_mrZABR["gamma_value"])
        # cTexF =cTexF.subs(gam, SETTINGS_mrZABR ["gamma_value"])
        GintF = GintF.subs(gam, SETTINGS_mrZABR["gamma_value"])

    get_tauex = sp.lambdify(inp_vars, tauexF, LIB_MATH)
    get_bTex = sp.lambdify(inp_vars, bTexF, LIB_MATH)
    get_cTex_int = sp.lambdify(inp_vars, cTex_intF, LIB_MATH)
    # get_cTex=   sp.lambdify(inp_vars, cTexF, LIB_MATH)
    get_Gint = sp.lambdify(inp_vars, GintF, LIB_MATH)



    ###########################################################################
    # fit  parameters from data

    alpha_, theta_, lambda_, nu_, rho_ = fit_parameters (DATA_IV)

    ###########################################################################
    # generate 3D coordinates for surface
    X, Y, Z = calc_XYZ (alpha_, theta_, lambda_, nu_, rho_, Moneyness_Range = (-1, +1), Expiry_Range = (0.01, 3.0), Moneyness_N = 100, Expiry_N = 100)
    # extact 3D coordinates for empirical smiles
    xyz_emp = extract_dataXYZ(DATA_IV)


    ###########################################################################
    #  generate interactive 3D plots

    # faster / better quality via plotly, requires viewing in browser
    disp_plotly (X, Y, Z, xyz_emp)

    # basic via matplotLib (pure Python)
    disp_matplotLib (X, Y, Z, xyz_emp)


    ###########################################################################
    #  example for trying out arbitrary parameters (with no empirical data)
    #alpha_   = 0.50
    #theta_   = 0.05
    #lambda_  = 3
    #nu_      = math.sqrt ( 4  * lambda_ * theta_) # non-generation mrZABR
    #rho_     = -0.6

    #X, Y, Z = calc_XYZ (alpha_, theta_, lambda_, nu_, rho_, Moneyness_Range=(-1, +1), Expiry_Range=(0.01, 3.0), Moneyness_N=100, Expiry_N=100)
    #disp_matplotLib (X, Y, Z)
    #disp_plotly (X, Y, Z)
