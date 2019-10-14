#### This is a small function to fit the Sh vs Phi curve, using blah blah!!

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model_func(phi, lamb):
    return 3 + phi*np.tanh(lamb*phi)

phi = np.logspace(-2, 3, 100)

Sh = (1/(phi*np.tanh(phi)) - 1/phi**2)**-1

fig, ax = plt.subplots()
ax.loglog(phi, Sh, label='original')

popt, pconv = curve_fit(model_func, phi, Sh)
Sh_lamb = model_func(phi, *popt)

ax.loglog(phi, Sh_lamb, color='r', label='fitted')
ax.legend()
print(popt)



