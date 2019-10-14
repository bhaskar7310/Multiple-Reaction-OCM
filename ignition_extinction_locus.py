#### This is a script file to solve for the bifurcation set at the
#### User-specified values

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ourstyle')

# =============================================================================
# USER INPUT SECTION FOR BIFURCATION SET CALCULATION
# =============================================================================

# Reaction System Identification
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
P = 1
tau = 1
T_f_in = 400

R_omega = 0.25e-03                        
R_omega_wc = R_omega/0.25e-03 * 100e-06                      
particle_density = 3600                   

##### Bifurcation Parameter
fixed_var = ['pressure', 'inlet_ratio', 'R_omega', 'R_omega_wc', 'particle_density']
fixed_val = [P, inlet_ratio, R_omega, R_omega_wc, particle_density]
fixed_dict = dict(zip(fixed_var, fixed_val))


#### File Specifications 1
data_vals = [value for (key, value) in sorted(fixed_dict.items())]
n = len(data_vals)
filename = 'bif_set_ignition'
    
for i in range(n):
    filename += '_{}'.format(data_vals[i])
        
filename += '.npz'

react_filename, ext = os.path.splitext(react_sys)

FullfileName = os.path.join(os.getcwd(), react_filename, catalyst.lower(),
                            'Data', system, filename)

if os.path.exists(FullfileName):
    npzfile = np.load(FullfileName)
else:
    raise FileNotFoundError('File with the path: {}'
                            'does not exist.'.format(FullfileName))

T_bs_ig_arr, = npzfile.files
T_bs_ig = npzfile[T_bs_ig_arr]

 
#### File Specifications 2
data_vals = [value for (key, value) in sorted(fixed_dict.items())]
n = len(data_vals)
filename = 'bif_set_extinction'
    
for i in range(n):
    filename += '_{}'.format(data_vals[i])
        
filename += '.npz'

react_filename, ext = os.path.splitext(react_sys)

FullfileName = os.path.join(os.getcwd(), react_filename, catalyst.lower(),
                            'Data', system, filename)

if os.path.exists(FullfileName):
    npzfile = np.load(FullfileName)
else:
    raise FileNotFoundError('File with the path: {}'
                            'does not exist.'.format(FullfileName))
    
T_bs_ext_arr, = npzfile.files
T_bs_ext = npzfile[T_bs_ext_arr]    
    
#### Plotting
fig, ax = plt.subplots()
ax.semilogx(T_bs_ig[-1, :], T_bs_ig[-2, :], label='Ignition')
ax.semilogx(T_bs_ext[-1, :], T_bs_ext[-2, :], label='Extinction')
ax.axvline(1e-06)
ax.set_xlabel('Space times (s)')
ax.set_ylabel('Inlet temperatures (K)')
ax.legend(loc='best')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    