"""  """

import numpy as np


def hermite_poly(n):
    """
    e.g., H_3 = np.poly1d(a[3][::-1])

    Args:
        n (int): maximum degree of polynomial

    Returns:
        ndarray: 2d matrix with polynomials. Ensure that coefficients are reversed before using
    """
    __a=np.zeros((n,n))
    __a[0,0] = 1
    for _np1 in range(1,n):
        for _k in range(n-1):
            if _k==0:
                __a[_np1,_k]=-__a[_np1-1,_k+1]
            else:
                __a[_np1,_k]=2*__a[_np1-1,_k-1]-(_k+1)*__a[_np1-1,_k+1]
    return __a
