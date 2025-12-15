
import numpy as np
from finite_differential import d2f_dx2
from ansatz import psi_func_3d

def rho_particular(x,y,z,t,N):
    _rho_integrate = 0.
    _x_range = np.linspace(np.min(x),np.max(x),N)
    _y_range = np.linspace(np.min(y),np.max(y),N)
    _z_range = np.linspace(np.min(z),np.max(z),N)
    _psi_sq = psi_func_3d(x,y,z,t)**2
    _rho_integrate = np.trapezoid(_psi_sq, _x_range)
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

def local_energy(x,y,z,t):
    """Local energy E_l of wavefunction

    Args:
        x (np.ndarray): indept variable
        n (int): index for given eigenfunction

    Returns:
        Scalar value
    """
    psi = psi_func_3d(x,y,z,t)
    psi_dx = d2f_dx2(psi, x)
    psi_dy = d2f_dx2(psi, y)
    psi_dz = d2f_dx2(psi, z)
    xi = x[2:]
    yi = y[2:]
    zi = z[2:]
    return - 0.5*(psi_dx + psi_dy + psi_dz)/psi[:-2] - (xi**2 + yi**2 + zi**2)**(-0.5)
