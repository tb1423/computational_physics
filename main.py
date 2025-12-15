""" """

import numpy as np
import matplotlib.pyplot as plt
from interpolate import intpt_df
import three_osc as to


## 1D Harmonic Oscillator Parameters
N_S     =   1000
X_MIN   =  -25
X_MAX   =   25
N_MODE  =   3

#for N_MODE in range(0,10):

##  Linspaces
x_space = np.linspace(X_MIN,X_MAX,N_S)
y_space = np.linspace(X_MIN,X_MAX,N_S)
z_space = np.linspace(X_MIN,X_MAX,N_S)

t = 1.

##  Generate uniform distribution
Fri = np.random.uniform(0,1,(N_S,N_S,N_S))

##  Calculate energy distribution using eq.3
rho_pdf = (X_MAX - X_MIN)*to.rho_particular(x_space,y_space,z_space,t,N_S)

##  Generate cdf
rho_cdf = to.rho_cumulative(rho_pdf,N_S)

## Generagte local energy distribution
E_l = to.local_energy(x_space,y_space,z_space,t)

###  Interpolate cdf inverse
#R_i = intpt_df(x_space, Fi, rho_cdf)
#plt.hist(R_i,bins=50)
#plt.close()
#
#e_l_ri = intpt_df(E_l, R_i, x_space[2:])
#
#H_mean = np.sum(e_l_ri) / N_S
#
#print(f'{N_MODE}: {H_mean}')