"""- hydrogen_atom.py -"""

import numpy as np
import matplotlib.pyplot as plt
from interpolate import rejection_df_3d
import atomic_simulator as to


## 3D hydrogen atom parameters
N_S      =   100
NO_PTS   =   100
X_MIN    =  -3
X_MAX    =   3
T_MIN    =   0.1
T_MAX    =   2.
T_RES    =   1000
LOOP_MAX =   1000
ALPHA    =  -1e1

## Temporary parameter 'theta'
T = 1.

## Linspaces
x_space = np.linspace(X_MIN,X_MAX,N_S)
y_space = np.linspace(X_MIN,X_MAX,N_S)
z_space = np.linspace(X_MIN,X_MAX,N_S)

## theta range
t_space = np.linspace(T_MIN,T_MAX,T_RES)

##  Calculate energy distribution using eq.3
rho_pdf = to.rho_particular_3d_num(x_space,y_space,z_space,T)


## Generate local energy function
E_l = to.local_energy_3d(x_space,y_space,z_space,T)


## rejection pts uniform distribution
R_i = rejection_df_3d(rho_pdf, x_space, y_space, z_space, NO_PTS, NO_PTS)

## Plot distribution on x-y plane
plt.hist2d(R_i[:,0],R_i[:,1],bins=20)
plt.close()


## Locate local energy values by interpolating R_i
loc_enrg = []
for r in R_i:
    loc_enrg.append(to.get_local_energy(r[0],r[1],r[2],E_l[0],E_l[1],E_l[2],E_l[3],E_l[4]))
loc_enrg = np.array(loc_enrg)

for _ in range(LOOP_MAX):
    H_mean = np.sum(np.array(loc_enrg)) / np.size(loc_enrg)
    d_th_H = to.del_th_H(R_i, loc_enrg,T,t_space)
    T -= ALPHA * d_th_H

    #print(T)
    print(d_th_H)
    if np.abs(ALPHA * d_th_H) < 1e-4:
        break

print(f'<H> = {H_mean}, T = {T}')
print(d_th_H)
