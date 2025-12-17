"""- main.py -"""

import numpy as np
import matplotlib.pyplot as plt
from interpolate import rejection_df_3d
import atomic_simulator as to


## 3D hydrogen atom parameters
N_S     =   int(1E2)
NO_PTS  =   int(2E2)
X_MIN   =  -3
X_MAX   =   3
## Temporary parameter 'theta'
T = 1.

## Linspaces
x_space = np.linspace(X_MIN,X_MAX,N_S)
y_space = np.linspace(X_MIN,X_MAX,N_S)
z_space = np.linspace(X_MIN,X_MAX,N_S)


## Uniform distributions for 3 DoF
#x_i = np.random.uniform(X_MIN,X_MAX,NO_PTS)
#y_i = np.random.uniform(X_MIN,X_MAX,NO_PTS)
#z_i = np.random.uniform(X_MIN,X_MAX,NO_PTS)



#E_l = to.local_energy(x_space,y_space,z_space,T)

##  Calculate energy distribution using eq.3
rho_pdf = to.rho_particular_3d(x_space,y_space,z_space,T)
plt.imshow(rho_pdf[:,:,5])
plt.colorbar()
plt.close()


## rejection pts uniform distribution
accepted = rejection_df_3d(rho_pdf, x_space, y_space, z_space, NO_PTS, NO_PTS)

plt.hist2d(accepted[:,0],accepted[:,1],bins=20)

plt.close()

loc_enrg = []

ints = to.local_energy_3d(x_space,y_space,z_space,T)

for i in accepted:
    loc_enrg.append(to.local_nerg(i[0],i[1],i[2],ints[0],ints[1],ints[2],ints[3],ints[4]))

H_mean = np.sum(np.array(loc_enrg)) / np.size(loc_enrg)

print(H_mean)
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
