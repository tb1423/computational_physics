"""  """

import numpy as np
import matplotlib.pyplot as plt
from interpolate import retn_df
import three_osc as to


## 3D hydrogen atom parameters
N_S     =   100
NO_PTS  =   10000
X_MIN   =  -3
X_MAX   =   3
C       =   1.1


## Linspaces
x_space = np.linspace(X_MIN,X_MAX,N_S)
y_space = np.linspace(X_MIN,X_MAX,N_S)
z_space = np.linspace(X_MIN,X_MAX,N_S)


## Uniform distributions for 3 DoF
x_i = np.random.uniform(X_MIN,X_MAX,NO_PTS)
y_i = np.random.uniform(X_MIN,X_MAX,NO_PTS)
z_i = np.random.uniform(X_MIN,X_MAX,NO_PTS)


## Temporary parameter 'theta'
T = 1.
#E_l = to.local_energy(x_space,y_space,z_space,T)

##  Calculate energy distribution using eq.3
rho_pdf = to.rho_particular_3d(x_space,y_space,z_space,T)
plt.imshow(rho_pdf[:,:,5])
plt.colorbar()
plt.show()


## rejection pts uniform distribution



accepted = list(retn_df(rho_pdf,x_i,y_i,z_i,NO_PTS))

print(accepted)

## Generagte local energy distribution

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
