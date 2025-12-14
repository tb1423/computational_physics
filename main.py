import matplotlib.pyplot as plt
import numpy as np
from random import randint

from hermite import hermite_poly


def psi_func(x,n):
    """Evaluate wavefunction psi_n(x) at a given point in x

    Args:
        x (float): independent horizontal location
        n (int): Index begins at 0
    """
    _hermite = 0.
    a_n = hermite_poly(n+2)

    for k, a_k in enumerate(a_n[n][::-1]):
        _hermite += a_k * x**k

    return _hermite * np.exp(-0.5*x**2)

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


def local_energy(x,n):
    """Local energy E_l of wavefunction

    Args:
        x (np.ndarray): indept variable
        n (int): index for given eigenfunction

    Returns:
        array of values
    """
    _x = x[2:]
    return 0.5*_x**2 - 0.5*d2f_dx2(psi_func(x,n),x) / psi_func(_x,n)


NO=10000

def rho_dist(x,n,dom,N=1000):
    _rho_integrate = 0.
    _range = np.linspace(dom[0],dom[1],N)
    _psi_sq = psi_func(x,n)**2
    for r in _range:
        _rho_integrate += (1/N)*psi_func(r,n)**2
    return _psi_sq / _rho_integrate

def rho_cum(p,N=1000):
    _cdf = [ ]
    _acc = 0.
    for val in p:
        _acc += val/N
        _cdf.append(_acc)
    return np.array(_cdf)

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
            x_i = x_i_di[i-1]
            x_ip1 = x_i_di[i]
            f_i   = f_i_di[i-1]
            f_ip1 = f_i_di[i]

            return x_i + (f_val_un - f_i) * (x_ip1 - x_i) / (f_ip1 - f_i) # + (1/NO)

        ## Return the same inputted if interpolation does not work
    print("Interpolation error")
    return f_val_un

def intpt_cdf(x_i, f_i, cdf):
    """Interpolate individual x values from an inverted cdf.

    Args:
        r (_type_): _description_
        f_i (_type_): _description_
        cdf (_type_): _description_

    Returns:
        _type_: _description_
    """

    v_i = []

    for _f_val in f_i:
        v_i.append(interplt(_f_val, x_i, cdf))

    return np.array(v_i)


##  
x_space = np.linspace(-5,5,NO)
r = np.linspace(-5,5,NO)

##  Generate uniform distribution
Fi = np.random.uniform(0,1,NO)

##  Calculate energy distribution using eq.3
rho_pdf = rho_dist(x_space,4,[-5,5],NO)

##  Generate cdf
rho_cdf = rho_cum(rho_pdf,NO)

##  Interpolate cdf inverse
v_i = intpt_cdf(r, Fi, rho_cdf)

plt.hist(np.array(v_i),bins=50)
plt.show()
