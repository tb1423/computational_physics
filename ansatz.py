"""- ansatz.py -"""

import numpy as np
from hermite import hermite_poly


def psi_func_1d(x,n):
    """Evaluate wavefunction psi_n(x) at a given point in x.

    Args:
        x (float): independent horizontal location
        n (int): Index begins at 0
    """
    _hermite = 0.
    a_n = hermite_poly(n+5)

    for k, a_k in enumerate(a_n[n]):
        _hermite += a_k * x**k

    return _hermite * np.exp(-0.5*x**2)


def psi_func_3d(x,y,z,theta):
    """Evaluate wavefunction for 3d hydrogen atom with given parameter T.
        As defined by eq. 15 in project notes.

    Args:
        x (float): x position
        y (float): y position
        z (float): z position
        theta (float): parameter value - to be minimised in Q.3
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    return np.exp(-theta * r)

def psi_func_ml(r1,r2,t1,t2,t3,q1,q2):
    """Evaluate wavefunction for 3d H2 molecule system.
        As defined by eq. 18 in project notes.

    Args:
        x (float): x position
        y (float): y position
        z (float): z position
        theta (float): parameter value - to be minimised in Q.3
    """
    e_1 = np.exp(-t1 * ( np.abs(r1 - q1) + np.abs(r2 - q2)))
    e_2 = np.exp(-t1 * ( np.abs(r1 - q2) + np.abs(r2 - q1)))

    return (e_1 + e_2) * np.exp2(-t2, 1+t3*np.abs(r1-r2))
