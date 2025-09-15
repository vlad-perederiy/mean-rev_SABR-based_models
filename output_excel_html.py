from  sympy import symbols
import  sympy as sp
import pickle
from sympy.printing.mathml import mathml

#################################################
## SETTINGS 
CLR_excel   = "\033[34m"  # Excel formulas: blue
CLR_text    = "\033[0m"    # black
#################################################


def gen_mathML (expr, doSubs=True):
    """
    converts a sympy expression into mathml
    """
    if doSubs:
        expr = expr.subs(sp.exp(Tex * lam), zzz).subs (mathML_sbs)# using "zzz" moves exp. term to the end of terms
    mathml_expr= mathml (expr, printer='presentation')
    # change parentheses handling
    mathml_expr = mathml_expr.replace('<mfenced>', '<mo>(</mo><mrow>')
    mathml_expr = mathml_expr.replace( '</mfenced>', '</mrow><mo>)</mo>')
    mathml_str = '<math xmlns = "http://www.w3.org/1998/Math/MathML">' + mathml_expr + '</math>'
    mathml_str = mathml_str.replace ("zzz", expTerm) # replace with intended term
    return mathml_str


def print_excel (expr, simplify_result=True):
    """
    converts a sympy expression into Excel cell formula
    """
    print(f"{CLR_excel}Excel format, with E≡exp(T_ex*lam) & sig≡1,   ...", end="")
    if expr is None:
        print ("failed")
        return
    # remove sigma paramater, substitute exponential terms, first with zzz (to avoid confusing with scientific E)
    expr= expr.subs(sig, 1).subs(sp.exp(Tex*lam), zzz)
    if simplify_result:
        expr=expr.simplify() # try to simplify (again)
    xls = str(expr)
    xls=xls.replace("**", "^")
    xls = xls.replace(" ", "")

    # replace simple minus/negation (except before numbers) with "-1*"
    # as simple negation is processed before power ^ in Excel terms
    # e.g. Excel formula "=-(1+2)^2" results in +9
    # so make it multiplication instead "=-1*(1+2)^2" which results in -9 as expected
    for d in "0123456789": # substitute minus with numbers with $ with numbers to exclude from substitutions (as these work fine in Excel. e.g. -1^2 = 1)
        xls=xls.replace("-"+d, "$"+d)
    if xls[0]=="-":xls= "-1*"+xls[1:] # replace other negations, in the very beginning
    xls = xls.replace("(-", "(-1*") # replace other negations, after opening brackets
    xls = xls.replace("^-", "^?-") # mark negations in powers (so that, if present, would result in error in EXcel)
    xls = xls.replace("$", "-") # substitute back minus with numbers

    xls = xls.replace("-", " - ") # insert spaces around minus/plus for readability of terms
    xls = xls.replace("+", " + ")
    xls = xls.replace("zzz", "E") # replace exp terms with E
    xls= xls.strip() # remove trailing spaces
    print ("\b\b\b" + str(len(xls)) +" chars:           ", xls, f"{CLR_text}")





def process_expressions (expressions, intro_text):
    """
    goes through all expressions (effective coefficients)
    generates Excel and formatted html
    """


    global html_results

    print (intro_text)
    html_results+= "<h2> " + intro_text + "</h2>\n"


    print("")
    print("*"*80)
    print(intro_text)
    print("*"*80)

    print ("")
    print("Basic Inputs (effective coefficients):\n")


    tauex = expressions["tauex"]
    print("τ_ex = ", tauex)
    print_excel(tauex, simplify_result=True)
    html_results += "<b>" + gen_mathML(symbols("tau_ex"))  + ":" + "</b><BR>" + gen_mathML(tauex) + "<BR><BR>\n\n"

    bTex = expressions["bTex"]
    print("b_Tex = ", bTex)
    print_excel(bTex, simplify_result=True)
    html_results += "<b>" + gen_mathML(symbols("b_Tex")) + ":" + "</b><BR>" + gen_mathML(bTex) + "<BR><BR>\n\n"
    print("")

    cTex_int = expressions["cTex_int"]
    print("c_int within c_Tex = ", cTex_int)
    print_excel(cTex_int, simplify_result=True)
    html_results += "<b> " +gen_mathML(symbols("cTex_int")) + "</b>"+ f" (length: {len(str(cTex_int))} chars):</b><BR>"
    html_results += gen_mathML(cTex_int) + "<BR><BR>\n\n"

    try:
        cTex_int_alt  = expressions["cTex_int_alt"]
        print("c_int alternative within c_Tex = ", cTex_int_alt)
        print_excel(cTex_int_alt, simplify_result=True)
        print ("Equal? ",  end="")
        print((cTex_int_alt - cTex_int).simplify() ==0)
        html_results += "<b> alternative  "+ gen_mathML(symbols("cTex_int")) + "</b>" + f" (length: {len(str(cTex_int_alt))} chars):</b><BR>"
        html_results += gen_mathML(cTex_int_alt) + "<BR><BR>\n\n"
    except:
        pass

    print("")
    Gint = expressions["Gint"]
    print("G_int, integral over I5(T) = ", Gint)
    print_excel(Gint, simplify_result=True)
    html_results += "<b>" + gen_mathML(symbols("G_int")) + ":" + "</b><BR>" + gen_mathML(Gint) + "<BR><BR>\n\n"

    print("")
    print("Standardized Coefficients:\n")
    cTex= (3 / tauex** 3 * cTex_int- 3 * bTex** 2)


    alphaStd = sp.sqrt(tauex/ Tex) * sp.exp(
        - cTex* tauex/ 4 + Gint/ 2 / tauex)
    print("α_std = ", alphaStd)
    print_excel(alphaStd, simplify_result=False)

    rhoStd= (bTex) / sp.sqrt(cTex)
    print("ρ_std = ", rhoStd)
    print_excel(rhoStd, simplify_result=False)
    # print_excel (rhoStd, simplify_result=True)

    nuStd= sp.sqrt(tauex/ Tex) * sp.sqrt(cTex)
    print("ν_std = ", nuStd)
    print_excel(nuStd, simplify_result=False)



def write_html (html_results):
    # output the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body>
        <h1>Closed-Form  Output for Mean-Reversion SABR-based models</h1>
        {html_results}
    </body>
    </html>
    """

    # Write the content to an HTML file for manual inspection in a browser
    file_path = 'results/expressions.html'
    with open(file_path, 'w') as f:
        f.write(html_content)



if __name__ == '__main__':

    print ("")
    print ("")
    print("*++++++++++++++++**************************")
    print("* Excel and Html Output  *****+++++********")
    print("*++++++++++++++++**************************")



    # sympy symbols for model parameters
    lam = symbols("lam", positive=True) # lambda
    alp = symbols("alp", positive=True) # ampha
    sig = symbols("sig", positive=True) # sigma
    alps = symbols("alps", positive=True) # alpha squared
    mu = symbols("mu", positive=True) #
    dlt = symbols("dlt", real=True) #
    rho = symbols("rho", real=True) # rho
    nu = symbols("nu", positive=True) # nu
    the = symbols("the", positive=True) # theta
    thes = symbols("thes", positive=True) # theta squared
    Tex = symbols("T_ex", nonegative=True) # T_ex (option expiry)
    gam = symbols ("gam", positive=True) # gamma (ZABR)
    #G0 = symbols("G0", real=True) # Γ₀ from backbone function: Γ₀ = -C'(F), for lognormal forward: C(F) = F, Γ₀ = -1
    #T = symbols("T", nonegative=True) # argument in hSABR/mrSABR interim integral functions
    zzz = symbols ("zzz", positive=True) # argument in mrZABR interim integral functions



    #############################################################



    # for html output: full Greeks so that browsers can display as such
    lam2  = symbols("lambda", positive=True)
    alp2  = symbols("alpha", positive=True)
    sig2  = symbols("sigma", positive=True)
    alps2 = (symbols("alpha", positive=True))**2
    rho2  = symbols("rho", real=True)
    nu2   = symbols("nu", positive=True)
    the2  = symbols("theta", positive=True)
    thes2 = (symbols("theta", positive=True))**2
    Tex2  = symbols("T_ex", positive=True)
    gam2  = symbols ("gamma", positive=True)
    dlt2  = symbols ("delta", positive=True)
    mu2  = symbols ("mu", positive=True)

    #expTerm  = symbols ("𝔼", positive=True)
    expTerm = "&Euml;" #for exponential term  exp(Tex * lam), to shorten formulas in html/Excel

    mathML_sbs={lam:lam2, alp: alp2, sig: sig2,alps: alps2,rho: rho2, nu: nu2, the: the2, thes: thes2, Tex: Tex2, gam: gam2, dlt: dlt2, mu: mu2}


    # global variable for html content
    html_results =f"{expTerm}={gen_mathML(sp.exp(lam2 * Tex), doSubs=False)} <BR>"
    html_results+=f"{gen_mathML(dlt2, doSubs=False)}={gen_mathML( (alp2 -the2) / (alp2+the2), doSubs=False)} <BR>"
    html_results+=f"{gen_mathML(mu2, doSubs=False)}={gen_mathML(  ( (alp2 + the2) / 2), doSubs=False)} <BR>"


    # load dataset with hSABR symbolic solutions
    with open("results/hSABR_expressions.pkl", "rb") as f:
        results = pickle.load(f)
    process_expressions(results, "hSABR") #process

    # load dataset with mrSABR symbolic solutions
    with open("results/mrSABR_expressions.pkl", "rb") as f:
        results = pickle.load(f)
    process_expressions(results, "mrSABR") #process

    # load dataset with mrZABR symbolic solutions
    with open("results/mrZABR_expressions.pkl", "rb") as f:
        results = pickle.load(f)
    results_keys = [dict(sorted (k, reverse=True)) for k in  results.keys()]

    results_keys = sorted(# sort mrZABR variants
        results_keys,
        key=lambda d: ( str(d['gamma']),  d['expansion_order'], d['expansion_mode'],)
    )

    for k in results_keys:
        settings = {"gamma": k["gamma"], "expansion_mode": k["expansion_mode"],
                    "expansion_order": k["expansion_order"]}
        txt=f'mrZABR (gamma: {k["gamma"]}, expansion: order {k["expansion_order"]} / mode {k["expansion_mode"]})'
        expressions = results[frozenset(settings.items())]
        process_expressions (expressions, txt) # process mrZABR variants in a loop

        write_html (html_results) # write after each variant (to save in case sth goes wrong)




