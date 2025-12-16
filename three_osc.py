
import numpy as np
from finite_differential import d2f_dx2
from ansatz import psi_func_3d


def rho_particular_3d_num(x,y,z,t):

    X, Y, Z = np.meshgrid(x, y, z)
    psi2 = np.abs(psi_func_3d(X, Y, Z, t))**2
    Znorm = np.trapezoid(np.trapezoid(np.trapezoid(psi2, z, axis=2), y, axis=1), x, axis=0)
    return psi2 / Znorm


def rho_particular_3d(x,y,z,t):

    X, Y, Z = np.meshgrid(x, y, z)
    r = np.sqrt(X**2+Y**2+Z**2)
    return (1 / np.pi) * t**3 * np.exp(-2*r*t)



def local_energy(x,y,z,t):
    """Local energy E_l of wavefunction

    Args:
        x (np.ndarray): indept variable
        n (int): index for given eigenfunction

    Returns:
        Scalar value
    """
    X, Y, Z = np.meshgrid(x, y, z)
    
    psi = psi_func_3d(X,Y,Z,t)
    psi_dx = d2f_dx2(psi, x)
    psi_dy = d2f_dx2(psi, y)
    psi_dz = d2f_dx2(psi, z)

    print(psi_dx.shape)
    xi = x[2:]
    yi = y[2:]
    zi = z[2:]
    return - 0.5*(psi_dx + psi_dy + psi_dz)/psi[:-2] - (xi**2 + yi**2 + zi**2)**(-0.5)
