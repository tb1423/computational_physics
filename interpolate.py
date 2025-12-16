import numpy as np


def interplt(f_val_un, x_i_di, f_i_di):
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
        v_i.append(interplt(_f_val, x_i, r_i))

    return np.array(v_i)


def retn_df(rho,x,y,z,N,scalar=1.01):

    cmpsn_func = scalar*np.max(rho)

    p_i = np.random.uniform(x[0],cmpsn_func,N)
    q_i = np.random.uniform(y[0],cmpsn_func,N)
    r_i = np.random.uniform(z[0],cmpsn_func,N)

    for i, _ in enumerate(x):
        if rho(x,y,z) > p_i[i] and rho(x,y,z) < q_i[i] and rho(x,y,z) < r_i[i]:
            yield [x,y,z]
