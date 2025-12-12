import matplotlib.pyplot as plt
import numpy as np
from random import randint

from hermite import hermite_poly


def psi_func(x,n):
    """Evaluate wavefunction psi_n(x) at a given point in x

    Args:
        x (float): independent horizontal location
        n (int): Index begins at 0
    """
    _hermite = 0.
    a_n = hermite_poly(n+2)

    for k, a_k in enumerate(a_n[n][::-1]):
        _hermite += a_k * x**k

    return _hermite * np.exp(-0.5*x**2)

def df_dx(f,x):
    f1 = np.zeros(len(x)-1)
    for n in range(len(x)-1):
        f1[n] = (f[n+1]-f[n])/(x[n+1]-x[n])
    return f1

def d2f_dx2(f,x):
    f1 = df_dx(f,x)
    return df_dx(f1,x[:-1])


def E_l(x,n):
    _x = x[2:]
    return 0.5*_x**2 - 0.5*d2f_dx2(psi_func(x,n),x) / psi_func(_x,n)



NO=1000

v = np.linspace(-5,5,NO)

def rho_dist(x,n,dom,N=1000):
    _rho_integrate = 0.
    _range = np.linspace(dom[0],dom[1],N)
    _psi_sq = psi_func(x,n)**2
    for r in _range:
        _rho_integrate += (1/N)*psi_func(r,n)**2
    return _psi_sq / _rho_integrate

def rho_cum(p,N=1000):
    _cdf = [ ]
    _acc = 0.
    for val in p:
        _acc += val/N
        _cdf.append(_acc)
    return np.array(_cdf)

rho = rho_dist(v,4,[-5,5],NO)
plt.plot(v, rho)
plt.show()

rhoc = rho_cum(rho,NO)

print(rhoc)
plt.plot(v, rhoc)
plt.show()