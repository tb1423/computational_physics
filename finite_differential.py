

import numpy as np


def df_dx(f,x):
    """Take the first derivative of a function f wrtx.

    Args:
        f: f(x) numerical value
        x : indept variable

    Returns:
        array of values
    """
    f1 = np.zeros(len(x)-1)
    for n in range(len(x)-1):
        f1[n] = (f[n+1]-f[n])/(x[n+1]-x[n])
    return f1


def d2f_dx2(f,x):
    """Take the second derivative of a function f wrtx.

    Args:
        f: f(x) numerical value
        x : indept variable

    Returns:
        array of values
    """
    f1 = df_dx(f,x)
    return df_dx(f1,x[:-1])
