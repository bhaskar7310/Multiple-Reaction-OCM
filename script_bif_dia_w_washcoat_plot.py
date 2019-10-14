# USER INPUT SECTION FOR BIFURCATION DIAGRAM CALCULATION
# ============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ourstyle')

# Reaction System Identification
#react_sys = 'OCM_eleven_reaction.txt'
#catalyst = 'model'

react_sys = 'OCM_eleven_reaction.txt'
catalyst = 'model'                        
system = 'coup'                           

homo_basis = 'mole_fraction'              
cat_basis = 'pressure'               
rate_basis = [homo_basis, cat_basis]

homo_rate_units = 'second'                
cat_rate_units = 'gm_sec'                  
rate_units = [homo_rate_units, cat_rate_units]

#### Fixed Inputs
inlet_species = 'CH4'                       
inlet_ratio = 6
P = 1             # Pressure is in atm (for the time being)            
tau = 1e-02     
T_f_in = 300

R_omega = 0.3e-03                  
R_omega_wc = 15e-06
R_omega_w = 100e-06
particle_density = 3600

##### Bifurcation Parameter
fixed_var = ['inlet_ratio', 'pressure', 
             'tau', 'R_omega', 'R_omega_wc', 
             'R_omega_w', 'particle_density']
fixed_val = [inlet_ratio, P, tau, R_omega, 
             R_omega_wc, R_omega_w, particle_density]
fixed_dict = dict(zip(fixed_var, fixed_val))

bif_par_var = 'T_f_in'
bif_par_val = T_f_in
bif_par_dict = {bif_par_var : bif_par_val}

#### Loading the file
data_vals = [value for (key, value) in sorted(fixed_dict.items())]
n = len(data_vals)
filename = 'bif_dia_sh_phi'

for i in range(n):
    filename += '_{}'.format(data_vals[i])
filename += '.npz'
react_filename, ext = os.path.splitext(react_sys)
        
Target_folder = os.path.join(os.getcwd(), 'Washcoat', react_filename, 
                             catalyst.lower(), 'Data', system)
FullfileName = os.path.join(Target_folder, filename)

print(FullfileName)
npzfile = np.load(FullfileName)
T_bd_arr, F_in_arr, species_ID_arr = npzfile.files
T_bd = npzfile[T_bd_arr]
F_in = npzfile[F_in_arr]
species_ID = npzfile[species_ID_arr].item()

#### Conversion Calculations
hc_index = species_ID[inlet_species]
conv_hc = 1 - T_bd[hc_index, :]/F_in[hc_index]

O2_index = species_ID['O2']
conv_O2 = 1 - T_bd[O2_index, :]/F_in[O2_index]

#### Plotting the Fluid temperaturs
fig, ax = plt.subplots()
ax.plot(T_bd[-1, :], T_bd[-2, :], label=r'$\mathbf{Sh_\phi}$ model')
ax.set_xlabel('Inlet Fluid Temperature (K)')
ax.set_ylabel('Exit Fluid Temperature (K)')

#### Plotting the conversion
fig, ax1 = plt.subplots()
ax1.plot(T_bd[-1, :], conv_O2, label=r'$\mathbf{Sh_\phi}$ model')
ax1.set_xlabel('Inlet Fluid Temperature (K)')
ax1.set_ylabel(r'Conversion of $\mathbf{O_2}$')

##### Loading the file for other model
#data_vals = [value for (key, value) in sorted(fixed_dict.items())]
#n = len(data_vals)
#filename = 'bif_dia_sh_inf'
#
#for i in range(n):
#    filename += '_{}'.format(data_vals[i])
#filename += '.npz'
#react_filename, ext = os.path.splitext(react_sys)
#        
#Target_folder = os.path.join(os.getcwd(), 'Washcoat', react_filename, 
#                             catalyst.lower(), 'Data', system)
#FullfileName = os.path.join(Target_folder, filename)
#
#npzfile = np.load(FullfileName)
#T_bd_arr, F_in_arr, species_ID_arr = npzfile.files
#T_bd = npzfile[T_bd_arr]
#F_in = npzfile[F_in_arr]
#species_ID = npzfile[species_ID_arr].item()
#
##### Conversion Calculations
#hc_index = species_ID[inlet_species]
#conv_hc = 1 - T_bd[hc_index, :]/F_in[hc_index]
#
#O2_index = species_ID['O2']
#conv_O2 = 1 - T_bd[O2_index, :]/F_in[O2_index]
#
##### Plotting the Fluid temperaturs
##fig, ax = plt.subplots()
#ax.plot(T_bd[-1, :], T_bd[-2, :], label=r'$\mathbf{Sh_\infty}$ model')
#ax.set_xlabel('Inlet Fluid Temperature (K)')
#ax.set_ylabel('Exit Fluid Temperature (K)')
#ax.set_xlim(200, 800)
#ax.set_ylim(300, 1800)
#ax.legend(loc='best')
#
##### Plotting the conversion
##fig, ax1 = plt.subplots()
#ax1.plot(T_bd[-1, :], conv_O2, label=r'$\mathbf{Sh_\infty}$ model')
#ax1.set_xlabel('Inlet Fluid Temperature (K)')
#ax1.set_ylabel(r'Conversion of $\mathbf{O_2}$')
#ax1.set_xlim(200, 800)
#ax1.set_ylim([0, 1])
#ax1.legend(loc='best')
