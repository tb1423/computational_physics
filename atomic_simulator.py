"""- atomic_simulator.py -"""

import numpy as np
from finite_differential import d2f_dx2
from ansatz import psi_func_3d
from interpolate import interplt_3d_vec


def rho_particular_3d_num(x,y,z,t):

    X, Y, Z = np.meshgrid(x, y, z)
    psi2 = np.abs(psi_func_3d(X, Y, Z, t))**2
    Znorm = np.trapezoid(np.trapezoid(np.trapezoid(psi2, z, axis=2), y, axis=1), x, axis=0)
    return psi2 / Znorm


def rho_particular_3d(x,y,z,t):

    X, Y, Z = np.meshgrid(x, y, z)
    r = np.sqrt(X**2+Y**2+Z**2)
    return (1 / np.pi) * t**3 * np.exp(-2*r*t)



def local_energy_3d(x, y, z, t):

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    psi = psi_func_3d(X, Y, Z, t).astype(float)


    psi_xx = np.apply_along_axis(lambda a: d2f_dx2(a, x), axis=0, arr=psi)
    psi_yy = np.apply_along_axis(lambda a: d2f_dx2(a, y), axis=1, arr=psi)
    psi_zz = np.apply_along_axis(lambda a: d2f_dx2(a, z), axis=2, arr=psi)

    # common interior grids
    xi = x[:-2]; yi = y[:-2]; zi = z[:-2]
    psi_int = psi[:-2, :-2, :-2]
    lap_int = psi_xx[:, :-2, :-2] + psi_yy[:-2, :, :-2] + psi_zz[:-2, :-2, :]

    return psi_int, lap_int, xi, yi, zi

def local_nerg(x0,y0,z0,psi_int,lap_int, xi, yi, zi, eps=1e-12):

    # interpolate ψ and ∇²ψ at Ri
    psi_val = float(interplt_3d_vec(xi, yi, zi, [x0], [y0], [z0], psi_int)[0])
    lap_val = float(interplt_3d_vec(xi, yi, zi, [x0], [y0], [z0], lap_int)[0])

    # potential term and safe division
    r = float(np.sqrt(x0**2 + y0**2 + z0**2))
    V = -1.0 / max(r, eps)
    if abs(psi_val) < eps:
        return np.nan
    return -0.5 * (lap_val / psi_val) + V
