#### This is a script file to solve for the bifurcation diagram at the
#### User-specified values
#### Right now let's focus on the catalytic partial oxidation reactions only 
#### with only hydrocarbon and O2 in feed.

#import bifurcation as bif
import bifurcation_arc_length_copy as bif

# ============================================================================
# USER INPUT SECTION FOR BIFURCATION DIAGRAM CALCULATION
# ============================================================================

# Reaction System Identification
react_sys = 'OCM_Zhe_Sun_homo.txt'
#react_sys = 'OCM_Stansch.txt'
#react_sys = 'OCM_two_reaction.txt'
catalyst = 'model'                           
#catalyst = 'La_Ce'
system = 'homo'                           

homo_basis = 'mole_fraction'              
cat_basis = 'mole_fraction'               
rate_basis = [homo_basis, cat_basis]

homo_rate_units = 'second'                
cat_rate_units = 'second'                  
rate_units = [homo_rate_units, cat_rate_units]

#### Fixed Inputs
inlet_species = 'CH4'                       
inlet_ratio = 4             
tau = 10            
T_f_in = 400

R_omega = 0.25e-03                  
R_omega_wc = 100e-06
particle_density = 3600

##### Bifurcation Parameter
fixed_var = ['inlet_ratio', 'tau', 'R_omega', 'R_omega_wc', 'particle_density']
fixed_val = [inlet_ratio, tau, R_omega, R_omega_wc, particle_density]
fixed_dict = dict(zip(fixed_var, fixed_val))

bif_par_var = 'T_f_in'
bif_par_val = T_f_in
bif_par_dict = {bif_par_var : bif_par_val}

#### Other Accessory parameter values (Can be Kept to DEFAULT Values)
act_var = ['T_f_in', 'tau', 'inlet_ratio', 'R_omega']

break_val_list = [10, 1e-010, 0, 0]
step_change_val_list = [0.001, tau, 0.01, 1e-07]
max_val_list = [1300, 1, 20, 1e-02]

break_dict = dict(zip(act_var, break_val_list))
step_change_dict = dict(zip(act_var, step_change_val_list))
max_dict = dict(zip(act_var, max_val_list))

break_val = break_dict[bif_par_var]
step_change = step_change_dict[bif_par_var]
max_val = max_dict[bif_par_var]

#### Solver Options
continuation = 'arc_length'  
solver = 'fortran'
Testing = False
save = False

options = {'continuation': continuation,
           'solver' : solver, 
           'Testing' : False,
           'save' : True}

#### Solving the bif_dia_solver using the timer decorator function
T_bd = bif.bif_dia_solver(fixed_dict, bif_par_dict, react_sys, system, 
                          catalyst, rate_basis, rate_units, inlet_species, 
                          break_val, step_change, max_val, options)
#if not Testing:
#    bif.ig_ext_point_calculator(T_bd, system, bif_par_var, break_dict, 
#                                 fulloutput=True)
# =============================================================================
