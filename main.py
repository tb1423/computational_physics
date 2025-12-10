import matplotlib.pyplot as plt
import numpy as np

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
    _x = x[:-2]
    return 0.5*_x**2 - 0.5*d2f_dx2(psi_func(x,n),x) / psi_func(_x,n)

v = np.linspace(0,3,100000)

plt.plot(v[:-2], E_l(v,5))
plt.show()
