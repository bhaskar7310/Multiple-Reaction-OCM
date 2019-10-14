#### This is a script file to plot bifurcation sets
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
inlet_ratio = 10                           
P = 1
tau = 1
T_f_in = 300

R_omega = 1e-03                        
#R_omega_wc = R_omega/0.25e-03 * 100e-06                      
R_omega_wc = 100e-06
particle_density = 3600                   

##### Bifurcation Parameter
bif_par_var = 'inlet_ratio'

if bif_par_var == 'inlet_ratio':
    xlabel = 'Inlet ' + r'$\mathbf{CH_{4}/O_{2}}$' + ' ratio'
    ylim = [50, 1000]
    plottype = 'linear'
elif bif_par_var == 'tau':
    xlabel = 'Space times (s)'
    plottype = 'log'
    
#### Fixed parameters
fixed_var = ['pressure', 'tau', 'R_omega', 'R_omega_wc', 'particle_density']
fixed_val = [P, tau, R_omega, R_omega_wc, particle_density]
fixed_dict = dict(zip(fixed_var, fixed_val))

#### File Specifications
data_vals = [value for (key, value) in sorted(fixed_dict.items())]
n = len(data_vals)
filename = 'bif_set_ignition_first'
    
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
print(FullfileName)
T_bs_arr, species_ID_arr= npzfile.files
T_bs = npzfile[T_bs_arr]

#### Plotting
fig, ax = plt.subplots()

if plottype is  'linear':
    ax.plot(T_bs[-1, :], T_bs[-2, :], label='Catalytic')
    ax.set_ylim(ylim)
    ax.set_xlim([4, 28])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Inlet temperatures (K)')

    
##### File Specifications
#data_vals = [value for (key, value) in sorted(fixed_dict.items())]
#n = len(data_vals)
#filename = 'bif_set_ignition_second'
#    
#for i in range(n):
#    filename += '_{}'.format(data_vals[i])
#        
#filename += '.npz'
#
#react_filename, ext = os.path.splitext(react_sys)
#
#FullfileName = os.path.join(os.getcwd(), react_filename, catalyst.lower(),
#                            'Data', system, filename)
#
#if os.path.exists(FullfileName):
#    npzfile = np.load(FullfileName)
#else:
#    raise FileNotFoundError('File with the path: {}'
#                            'does not exist.'.format(FullfileName))
#
#T_bs_arr, = npzfile.files
#T_bs = npzfile[T_bs_arr]
#
##### Plotting
##fig, ax = plt.subplots()
#
#if plottype is  'linear':
#    ax.plot(T_bs[-1, :], T_bs[-2, :], label='Thermally coupled')
#    ax.set_ylim(ylim)
#    ax.set_xlabel(xlabel)
#    ax.set_ylabel('Inlet temperatures (K)')
  
#ax.legend(loc='best')