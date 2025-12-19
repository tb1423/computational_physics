"""- atomic_simulator.py -"""

import numpy as np
from finite_differential import df_dx, d2f_dx2
from ansatz import psi_func_3d
from interpolate import interplt_3d_vec, lin_interplt


def rho_particular_3d_num(x,y,z,t):
    """ Numerically calculated pdf """

    X, Y, Z = np.meshgrid(x, y, z)
    psi2 = np.abs(psi_func_3d(X, Y, Z, t))**2
    Znorm = np.trapezoid(np.trapezoid(np.trapezoid(psi2, z, axis=2), y, axis=1), x, axis=0)
    return psi2 / Znorm


def rho_particular_3d(x,y,z,t):
    """ Analytical pdf function """

    X, Y, Z = np.meshgrid(x, y, z)
    r = np.sqrt(X**2+Y**2+Z**2)
    return (1 / np.pi) * t**3 * np.exp(-2*r*t)


def local_energy_3d(x, y, z, t):
    """ Generates local energy function in 3 dimensions """

    ## Generate meshgrids
    X, Y, Z = np.meshgrid(x, y, z)
    psi = psi_func_3d(X, Y, Z, t)

    ## Laplacians
    psi_xx = np.apply_along_axis(lambda a: d2f_dx2(a, x), axis=0, arr=psi)
    psi_yy = np.apply_along_axis(lambda a: d2f_dx2(a, y), axis=1, arr=psi)
    psi_zz = np.apply_along_axis(lambda a: d2f_dx2(a, z), axis=2, arr=psi)

    # Truncate (artifact from 2nd derr)
    xi = x[:-2]; yi = y[:-2]; zi = z[:-2]
    psi_int = psi[:-2, :-2, :-2]
    lap_int = psi_xx[:, :-2, :-2] + psi_yy[:-2, :, :-2] + psi_zz[:-2, :-2, :]

    return psi_int, lap_int, xi, yi, zi


def get_local_energy(x0,y0,z0,psi_int,lap_int, xi, yi, zi, eps=1e-12):
    """ Returns <H> with specific position and ansatz parameters"""

    psi_val = float(interplt_3d_vec(xi, yi, zi, [x0], [y0], [z0], psi_int)[0])
    lap_val = float(interplt_3d_vec(xi, yi, zi, [x0], [y0], [z0], lap_int)[0])

    r = float(np.sqrt(x0**2 + y0**2 + z0**2))
    V = -1.0 / max(r, eps)
    if abs(psi_val) < eps:
        return np.nan
    return -0.5 * (lap_val / psi_val) + V

def del_th_H(r_i,e_l_i,th,th_space):
    """ Returns the gradient of <H> wrt theta"""

    exp_e = np.sum(np.array(e_l_i)) / np.size(e_l_i)

    vd_e = 0.
    for i, _ in enumerate(r_i):
        psi_i = psi_func_3d(r_i[i][0],r_i[i][1],r_i[i][2],th_space)

        dpsi_dth = df_dx(psi_i,th_space)

        d_th_psi = lin_interplt(th, th_space[1:], dpsi_dth)

        vd_e += ( e_l_i[i] - exp_e ) * d_th_psi

    return 2 / len(e_l_i) * vd_e
