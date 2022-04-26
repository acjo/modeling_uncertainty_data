import numpy as np
from scipy.integrate import quad

def compute_cdf(x0, xf):
    pdf = lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi)
    return quad(pdf, x0, xf)

cdf = 1 - compute_cdf(-3, 3)[0]
chernoff = np.exp(-9/2)

print(cdf < chernoff)

