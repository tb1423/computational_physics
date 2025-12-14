

import numpy as np
from hermite import hermite_poly
from finite_differential import d2f_dx2

def psi_func(x,n):
    """Evaluate wavefunction psi_n(x) at a given point in x

    Args:
        x (float): independent horizontal location
        n (int): Index begins at 0
    """
    _hermite = 0.
    a_n = hermite_poly(n+5)
    

    for k, a_k in enumerate(a_n[n]):
        _hermite += a_k * x**k

    return _hermite * np.exp(-0.5*x**2)

def local_energy(x,n):
    """Local energy E_l of wavefunction

    Args:
        x (np.ndarray): indept variable
        n (int): index for given eigenfunction

    Returns:
        array of values
    """
    psi = psi_func(x, n)
    psi2 = d2f_dx2(psi, x)
    xi = x[2:]
    return 0.5*xi**2 - 0.5*psi2/psi[:-2]

def rho_particular(x,n,N):
    _rho_integrate = 0.
    _range = np.linspace(np.min(x),np.max(x),N)
    _psi_sq = psi_func(x,n)**2
    _rho_integrate = np.trapezoid(_psi_sq, _range)
    return _psi_sq / _rho_integrate

def rho_cumulative(p,N):
    _cdf = [ ]
    _acc = 0.
    for i, _ in enumerate(p):
        if i == 0:
            continue
        _acc += .5 * ( p[i]+p[i-1] ) / N
        _cdf.append(_acc)
    return np.array(_cdf)
