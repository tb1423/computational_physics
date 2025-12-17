""" """

import numpy as np
import matplotlib.pyplot as plt
from interpolate import intpt_df
import oscillator as so


## 1D Harmonic Oscillator Parameters
N_S     =   100
X_MIN   =  -3
X_MAX   =   3
N_LIM   =   20
#N_MODE  =   3

def energy_eigenvalues(n):
    """ I didn't need to write this, I just wanted to be complete :/ """
    return 0.5 + n

clc_err = []
erg_egn = np.array([ energy_eigenvalues(_n) for _n in range(N_LIM) ])


for N_MODE in range(0,N_LIM):

    ##  Linspaces
    x_space = np.linspace(X_MIN,X_MAX,N_S)

    ##  Generate uniform distribution
    Fi = np.random.uniform(0,1,N_S)

    ##  Calculate energy distribution using eq.3
    rho_pdf = (X_MAX - X_MIN)*so.rho_particular(x_space,N_MODE,N_S)
    plt.plot(x_space, rho_pdf)
    plt.close()

    ##  Generate cdf
    rho_cdf = so.rho_cumulative(rho_pdf,N_S)
    plt.plot(x_space[1:], rho_cdf)
    plt.close()

    ## Generagte local energy distribution
    E_l = so.local_energy(x_space, N_MODE)
    plt.plot(x_space[2:], E_l)
    plt.close()

    ##  Interpolate cdf inverse
    R_i = intpt_df(x_space, Fi, rho_cdf)

    plt.hist(R_i,bins=50)
    plt.close()

    e_l_ri = intpt_df(E_l, R_i, x_space[2:])

    H_mean = np.sum(e_l_ri) / N_S

    clc_err.append(H_mean)

    print(f'{N_MODE}: {H_mean}')

plt.scatter(np.arange(0,N_LIM), erg_egn - np.array(clc_err), marker='x')
plt.show()
