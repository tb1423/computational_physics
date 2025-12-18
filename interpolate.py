"""- interpolate.py -"""

import numpy as np


def inv_interplt(f_val_un, x_i_di, f_i_di):
    """Perform linear interpolation on a function.
        This must be a bijective function where df/dx > 0 for all x in domain.

    Args:
        f_val_un (_type_): function value from unniform distribution
        x_i_di (_type_): x value from distribution
        f_i_di (_type_): f value from distribution

    Returns:
        float: interpolated x value
    """
    for i, val in enumerate(f_i_di):
        if val > f_val_un:

            ## Return with linear interpolate of x wrt cdf
            x_i   = x_i_di[i-1]
            x_ip1 = x_i_di[i]
            f_i   = f_i_di[i-1]
            f_ip1 = f_i_di[i]

            return x_i + (f_val_un - f_i) * (x_ip1 - x_i) / (f_ip1 - f_i) # + (1/NO)

    ## Return the same inputted if interpolation does not work
    return f_val_un

def lin_interplt(x_val, x_grid, f_grid):
    """
    Linear interpolation of f at x_val given samples (x_grid, f_grid).
    Clamps to endpoints if x_val lies outside the grid.
    """
    xg = np.asarray(x_grid, dtype=float)
    fg = np.asarray(f_grid, dtype=float)
    i = np.searchsorted(xg, x_val, side='left')

    if i <= 0:
        return float(fg[0])
    if i >= len(xg):
        return float(fg[-1])

    x0, x1 = xg[i-1], xg[i]
    f0, f1 = fg[i-1], fg[i]
    t = (x_val - x0) / (x1 - x0)
    return float(f0 + t * (f1 - f0))

def intpt_df(x_i, f_i, r_i):
    """Interpolate arbitrary value x_i from a specified empirical function (r_i, f_i).

    Args:
        x_i (_type_): _description_
        f_i (_type_): _description_
        r_i (_type_): _description_

    Returns:
        _type_: _description_
    """

    v_i = []

    for _f_val in f_i:
        v_i.append(inv_interplt(_f_val, x_i, r_i))

    return np.array(v_i)


def interplt_3d_vec(xg, yg, zg, X, Y, Z, rho):
    """_summary_

    Args:
        xg (_type_): _description_
        yg (_type_): _description_
        zg (_type_): _description_
        X (_type_): _description_
        Y (_type_): _description_
        Z (_type_): _description_
        rho (_type_): _description_

    Returns:
        _type_: _description_
    """
    Nx, Ny, Nz = rho.shape

    i = np.searchsorted(xg, X, side='right'); i0 = np.clip(i-1, 0, Nx-2)
    j = np.searchsorted(yg, Y, side='right'); j0 = np.clip(j-1, 0, Ny-2)
    k = np.searchsorted(zg, Z, side='right'); k0 = np.clip(k-1, 0, Nz-2)

    tx = (X - xg[i0]) / (xg[i0+1] - xg[i0])
    ty = (Y - yg[j0]) / (yg[j0+1] - yg[j0])
    tz = (Z - zg[k0]) / (zg[k0+1] - zg[k0])

    f00 = rho[i0, j0, k0]*(1-tx) + rho[i0+1, j0, k0]*tx
    fx0 = rho[i0, j0+1, k0]*(1-tx) + rho[i0+1, j0+1, k0]*tx
    f0y = rho[i0, j0, k0+1]*(1-tx) + rho[i0+1, j0, k0+1]*tx
    fxy = rho[i0, j0+1, k0+1]*(1-tx) + rho[i0+1, j0+1, k0+1]*tx

    f0 = f00*(1-ty) + fx0*ty
    f1 = f0y*(1-ty) + fxy*ty
    return f0*(1-tz) + f1*tz

def rejection_df_3d(rho, x, y, z, n, no_pts, scalar=1.05, rng=None):
    """Utilises the rejection method to generate a distribution function
        with the same topology as rho. See 7.3 in notes for theory.

    Args:
        rho (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_
        z (_type_): _description_
        N (_type_): _description_
        scalar (float, optional): _description_. Defaults to 1.05.
        batch (int, optional): _description_. Defaults to 10000.
        rng (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if rng is None:
        rng = np.random.default_rng()

    cmpsn_fnc = scalar * np.max(rho)

    out = []
    while len(out) < n:
        x_c = rng.uniform(x[0], x[-1], no_pts)
        y_c = rng.uniform(y[0], y[-1], no_pts)
        z_c = rng.uniform(z[0], z[-1], no_pts)
        f_c = interplt_3d_vec(x, y, z, x_c, y_c, z_c, rho)
        ufm = rng.random(no_pts)
        keep = ufm <= f_c / cmpsn_fnc

        if np.any(keep):
            out.append(np.stack([x_c[keep], y_c[keep], z_c[keep]], axis=1))

    return np.concatenate(out, axis=0)
