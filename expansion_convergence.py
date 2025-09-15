import sympy
import numpy as np
import matplotlib.pyplot as plt

########################################################
# Settings
########################################################
# Parameters:
alp_value = 0.10   # alpha
the_value = 0.40   # theta
lam_value = 2      # lambda
pwr_value = 1.5    # power, corresponds to gamma, 2*gamma or gamma + 1 in integrand terms

appr_mode = 3 # expansion mode:
# 1: expand alpha around theta. Effect: fails/diverges for alp_value > 2 * the_value
# 2: expand theta around alpha. Effect: fails/diverges for the_value > 2 * alp_value
# 3: works fine for all alpha and theta values
#
########################################################
# End of Settings
########################################################




# Symbolic variables
alp = sympy.Symbol('alp', real=True)
the = sympy.Symbol('the', real=True)
lam = sympy.Symbol('lam', real=True)
T   = sympy.Symbol('T', real=True)
pwr = sympy.Symbol('pwr', real=True)

mu = sympy.Symbol('mu', real=True)
dlt = sympy.Symbol('dlt', real=True)


subs ={alp: alp_value, the:the_value, lam: lam_value, pwr: pwr_value}

# target expression
s_exact =  ( the + (alp - the) * sympy.exp(-lam * T)) ** (pwr)


def get_expansion (expr, appr_order):
    print ("calc approximation order", appr_order, "...")
    if appr_mode == 1:
        # expand alpha around theta
        expr = expr.series(alp, the, appr_order).removeO()
    elif appr_mode == 2:
        # expand theta around alpha
        expr = expr.series(the, alp, appr_order).removeO()
    elif appr_mode == 3:
        # expand delta = (alpha - theta) / (alpha + theta) around 0 via mu = (alpha + theta) / 2
        expr = expr.subs(the, mu * (1 - dlt)).subs(alp, mu * (
                    1 + dlt))#.simplify()  # substitute alpha and theta thru delta and mu
        expr = expr.series(dlt, 0, appr_order).removeO()
        #print (appr_order, expr)
        expr = expr.subs(dlt, (alp - the) / (alp + the)).subs(mu, (alp + the) / 2)#.simplify()  # substitute back
    return expr

orders_to_show = [2, 3, 4, 5, 6]
colors = ['r', 'g', 'b', 'm', 'c', 'p']

s_apprx = [get_expansion(s_exact, o)  for o in orders_to_show]


# Range of x-values for plotting
T_vals = np.linspace(0, 3, 109)  # years

# exact function
f_exact = sympy.lambdify  (T, s_exact.subs(subs), 'numpy')
# approximate functions
f_apprx = [sympy.lambdify (T, s.subs(subs), 'numpy')   for s in s_apprx]

# plot
plt.figure(figsize=(8, 6))

for N, color, f_apprx0 in zip(orders_to_show, colors, f_apprx):
    plt.plot(T_vals, f_apprx0(T_vals), color,
             label=f'Taylor order {N}')


plt.plot(T_vals, f_exact(T_vals), 'k', label='Exact function', linewidth=2)

plt.xlabel("Time (T)")
plt.ylabel("Exact / Approximation")
plt.ylim((0, 0.1+max (alp_value, the_value)**pwr_value))  # Adjust if needed
plt.legend()

plt.title(r"Taylor Expansion (mode "+str(appr_mode)+") for: " + str(s_exact.xreplace(subs)) , fontsize=11)

plt.grid(True)

plt.show()


