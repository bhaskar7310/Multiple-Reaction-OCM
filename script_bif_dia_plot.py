#### This is a script file to plot different bifurcation diagrams

import lineplots as lplt

# =============================================================================
# USER INPUT SECTION FOR BIFURCATION DIAGRAM PLOTTING
# =============================================================================

#Reaction System Identification
react_sys = 'OCM_eleven_reaction.txt'
#react_sys = 'OCM_Zhe_Sun_homo.txt'
#react_sys = 'OCM_Stansch.txt'
#catalyst = 'La-Ca'                    # Name of catalyst, default: model   
catalyst = 'model'                    # Name of catalyst, default: model   
system = 'coup'                       # homo/cat/coup system spec      

homo_basis = 'mole_fraction'              
cat_basis = 'pressure'               
rate_basis = [homo_basis, cat_basis]

homo_rate_units = 'second'                
cat_rate_units = 'gm_sec'                  
rate_units = [homo_rate_units, cat_rate_units]

#### Fixed Inputs
inlet_species = 'CH4'                 # Name of the hydrocarbon only
inlet_ratio = 6                 # Inlet mole ration of HC to O2
pressure = 1                        # Total pressure, in atm
tau = 5e-06                           # Space time, in secs
T_f_in = 400                          # Inlet temperature, in K

R_omega = 0.25e-03              # Channel hydraulic radius, in m
R_omega_wc = 100e-06
particle_density = 3600               # Catalyst particle density in kg/m3
    
#### Basic Plot Specifications
basic_plot = True

fluid_temp = True
solid_temp = True
conversion = True
yield_all = False
yield_comb = False
select_all = False
select_comb = False

basic_plot_names = ['fluid_temp', 'solid_temp', 'conversion', 'yield_all', 
                    'yield_comb', 'select_all', 'select_comb']
basic_plot_vals = [fluid_temp, solid_temp, conversion, yield_all, yield_comb, 
                   select_all, select_comb]
basic_plot_dict = dict(zip(basic_plot_names, basic_plot_vals))

user_def_products = ['CO', 'CO2', 'C2H6', 'C2H4']
 
#### Analysis Plot specifications
analysis_plot = True

reaction_rates = True
thiele_modulus = False
ratio_surface = False
ratio_bulk = False
surface_to_bulk = False
C2H4_C2H6_ratio = False

analysis_plot_names = ['reaction_rates', 'thiele_modulus', 
                       'ratio_surface', 'ratio_bulk',
                       'surface_to_bulk', 'C2H4_C2H6_ratio']
analysis_plot_vals = [reaction_rates, thiele_modulus, 
                      ratio_surface, ratio_bulk, 
                      surface_to_bulk, C2H4_C2H6_ratio]
analysis_plot_dict = dict(zip(analysis_plot_names, analysis_plot_vals))

#### Bifurcation Parameter
fixed_var = ['inlet_ratio', 'pressure', 'tau', 'R_omega', 'R_omega_wc', 'particle_density']
fixed_val = [inlet_ratio, pressure, tau, R_omega, R_omega_wc, particle_density]
fixed_dict = dict(zip(fixed_var, fixed_val))

bif_par_var = 'T_f_in'

if bif_par_var == 'tau':
    x_limits = [1e-08, 1]
    y_temp_limits = [T_f_in, 2000]
    
else:
    x_limits = [10, 1200]
    y_temp_limits = [400, 1800]    
        
basic_plot_options = {'basic_plot': basic_plot,
                      'xaxis_lim' : x_limits, 
                      'yaxis_lim' : y_temp_limits, 
                      'products' : user_def_products, 
                      'plots' : basic_plot_dict}

analysis_plot_options = {'analysis_plot': analysis_plot,
                         'plots': analysis_plot_dict,
                         'rate_basis': rate_basis,
                         'rate_units': rate_units}

#### Other Accessory parameter values (Can be Kept to DEFAULT Values)
act_var = ['T_f_in', 'pressure', 'tau', 'inlet_ratio', 'R_omega']

break_val_list = [10, 0.5, 1e-010, 0, 0]
step_change_val_list = [0.05, 0.01, tau, 0.01, 1e-07]
max_val_list = [2000, 50, 1, 20, 1e-02]

break_dict = dict(zip(act_var, break_val_list))
step_change_dict = dict(zip(act_var, step_change_val_list))
max_dict = dict(zip(act_var, max_val_list))

#### Solving the bif_dia_solver using the timer decorator function
T_bd = lplt.bif_dia_plot(fixed_dict, bif_par_var, react_sys, system, 
                        catalyst, inlet_species, basic_plot_options,
                        analysis_plot_options)
# =============================================================================
