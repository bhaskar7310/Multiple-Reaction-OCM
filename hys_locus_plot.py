#### A script to join all the data of hysteresis locus for OCM_two_reaction.txt
# =============================================================================
# USER INPUT SECTION FOR HYSTERESIS LOCUS PLOT
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ourstyle')

# Reaction System Identification
react_sys = 'OCM_two_reaction.txt'
catalyst = 'la_ce'
system = 'coup'                         


#### Fixed_inputs
inlet_species = 'CH4'
inlet_ratio = 2
P = 1
tau = 1e-02                              
T_f_in = 300                        

R_omega = 1e-03                          
R_omega_wc = 100e-06
particle_density = 3600

#### Bifurcation parameter
fixed_var = ['pressure', 'R_omega', 'R_omega_wc', 'particle_density']
fixed_val = [P, R_omega, R_omega_wc, particle_density]
fixed_dict = dict(zip(fixed_var, fixed_val))

bif_par_var = 'tau'
bif_par_val = tau
bif_par_dict = {bif_par_var : bif_par_val}

#### First file (catalytic case, left)
data_vals = [value for (key, value) in sorted(fixed_dict.items())]
n = len(data_vals)

filename = 'hysteresis_locus_left_first'
for i in range(n):
    filename += '_{}'.format(data_vals[i])
print(filename)
filename += '.npz'

react_filename, ext = os.path.splitext(react_sys)

Target_folder = os.path.join(os.getcwd(), react_filename, 
                             catalyst.lower(), 'Data', system)
FullfileName = os.path.join(Target_folder, filename)
npzfile = np.load(FullfileName)
T_hl_arr = npzfile.files
T_hl = npzfile[T_hl_arr[0]]

T_hl_left = np.flip(T_hl, 1)

#### Second file (catalytic case, right)
filename = 'hysteresis_locus_right_first'
for i in range(n):
    filename += '_{}'.format(data_vals[i])
print(filename)
filename += '.npz'

FullfileName = os.path.join(Target_folder, filename)
npzfile = np.load(FullfileName)
T_hl_arr = npzfile.files
T_hl_right = npzfile[T_hl_arr[0]]

T_hl_cat = np.c_[T_hl_left, T_hl_right]

#### Third file (Homogeneous case, left)
filename = 'hysteresis_locus_left_second'
for i in range(n):
    filename += '_{}'.format(data_vals[i])
print(filename)
filename += '.npz'

FullfileName = os.path.join(Target_folder, filename)
npzfile = np.load(FullfileName)
T_hl_arr = npzfile.files
T_hl = npzfile[T_hl_arr[0]]
T_hl_left = np.flip(T_hl, 1)

#### Fourth file (Homogeneous case, right)
filename = 'hysteresis_locus_right_second'
for i in range(n):
    filename += '_{}'.format(data_vals[i])
print(filename)
filename += '.npz'

FullfileName = os.path.join(Target_folder, filename)
npzfile = np.load(FullfileName)
T_hl_arr = npzfile.files
T_hl_right = npzfile[T_hl_arr[0]]

T_hl_homo = np.c_[T_hl_left, T_hl_right]

#### Fifth file (Homogeneous case, second file in left)
filename = 'hysteresis_locus_left_second_second'
for i in range(n):
    filename += '_{}'.format(data_vals[i])
print(filename)
filename += '.npz'

FullfileName = os.path.join(Target_folder, filename)
npzfile = np.load(FullfileName)
T_hl_arr = npzfile.files
T_hl = npzfile[T_hl_arr[0]]
T_hl_correct = T_hl[:, :201]
T_hl_left_left = np.flip(T_hl_correct, 1)
T_hl_homo = np.c_[T_hl_left_left, T_hl_homo]


#### Plotting
fig, ax = plt.subplots()
ax.semilogx(T_hl_cat[-1, :], 1/T_hl_cat[-2, :], label='Catalytic')
ax.semilogx(T_hl_homo[-1, :], 1/T_hl_homo[-2, :], label='Thermally coupled')
ax.set_xlim([5e-03, 1])
ax.set_ylim([0, 0.35])
ax.set_xlabel(r'$\mathbf{\tau}$ (s)')
ax.set_ylabel(r'Inlet $\mathbf{O_2/CH_4}$ ratio')
ax.legend(loc='best')
plt.show()


