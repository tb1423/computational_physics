
import numpy as np
from finite_differential import d2f_dx2
from ansatz import psi_func_1d


def local_energy(x,n):
    """Local energy E_l of wavefunction

    Args:
        x (np.ndarray): indept variable
        n (int): index for given eigenfunction

    Returns:
        array of values
    """
    psi = psi_func_1d(x, n)
    psi2 = d2f_dx2(psi, x)
    xi = x[2:]
    return 0.5*xi**2 - 0.5*psi2/psi[:-2]


def rho_particular(x,n,N):
    """ pdf for a 1D harmonic oscillator """
    _rho_integrate = 0.
    _range = np.linspace(np.min(x),np.max(x),N)
    _psi_sq = psi_func_1d(x,n)**2
    _rho_integrate = np.trapezoid(_psi_sq, _range)
    return _psi_sq / _rho_integrate


def rho_cumulative(p,N):
    """ numerically calculated cdf for a 
    1D harmonic oscillator"""
    _cdf = [ ]
    _acc = 0.
    for i, _ in enumerate(p):
        if i == 0:
            continue
        _acc += .5 * ( p[i]+p[i-1] ) / N
        _cdf.append(_acc)
    return np.array(_cdf)
