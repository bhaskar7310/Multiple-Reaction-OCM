#### This is a script file to solve for the bifurcation set at the
#### User-specified values

import bifurcation_arc_length as bif

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
fixed_var = ['pressure', 'tau', 'R_omega', 'R_omega_wc', 'particle_density']
fixed_val = [P, tau, R_omega, R_omega_wc, particle_density]
fixed_dict = dict(zip(fixed_var, fixed_val))

bif_par_var = 'inlet_ratio'
bif_par_val = inlet_ratio
bif_par_dict = {bif_par_var : bif_par_val}

#### Other Accessory parameter values (Can be Kept to DEFAULT Values)
act_var = ['T_f_in', 'pressure', 'tau', 'inlet_ratio', 'R_omega']

break_val_list = [10, 0.5, 1e-06, 9.5, 0]
step_change_val_list = [10, 0.01, 1, 1, 1e-07]
max_val_list = [10, 50, 1.1, 60, 1e-02]

break_dict = dict(zip(act_var, break_val_list))
step_change_dict = dict(zip(act_var, step_change_val_list))
max_dict = dict(zip(act_var, max_val_list))

break_val = break_dict[bif_par_var]
step_change = step_change_dict[bif_par_var]
max_val = max_dict[bif_par_var]

#### Solver Options
continuation = 'arc_length'
solver = 'python'
jac_eps = 1e-04
Testing = False
save = False

options = {'limit_point' : 'ignition',
           'occurence' : 'first',
           'continuation': continuation,
           'solver' : solver, 
           'jac_eps': jac_eps,
           'Testing' : Testing, 
           'save' : save}

#### Solving the bif_dia_solver using the timer decorator function
T_bs = bif.bif_set_solver(fixed_dict, bif_par_dict, react_sys, system, 
                          catalyst, rate_basis, rate_units, inlet_species, 
                          break_dict, step_change, max_val, options)
if Testing:
    print('\nThe function values at the testing point')
    print(T_bs)
# =============================================================================
