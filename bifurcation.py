#### This module named as 'bifurcation' has all the functions and solvers to
#### calculate all kinds of bifurcation diagrams and sets of Propane Oxidation
#### case. Refer Imran Alam's first paper.

#### Author: Bhaskar Sarkar (2018)

import os
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

import constants
import reaction as react

def bif_dia_func(Y, Y0, inlet_species, fixed, const, thermo_object, 
                 react_const, flag,  rate_basis, rate_units, del_s=0):
    '''
    This function is used to generate bifurcation diagrams. It can identify 
    what are the fixed parameters and what is the bifurcation parameter.
    Write a story about this later.
    '''
    #### Assert statements
    assert (flag == 'cat') or (flag == 'coup') or (flag == 'homo'), ('This '
                            'program can only handle catalytic ("cat") or '
                            'coupled ("coup") system.')

    #### Unpacking the inputs
    
    #### Species identifier
    species = const['species']
    no_of_species = len(species)
    ID = np.arange(no_of_species)
    species_ID = dict(zip(species, ID))

    #### Unpacking the Y vector
    n = Y.size
    try:
        m = Y0.size
    except AttributeError:
        m = 1

    F_j = Y[:no_of_species]                 # Molar flow rate of species j in bulk (mol/s)
    C_s = Y[no_of_species:2*no_of_species]  # Surface concentrations (mole/m3-s)
    T_s = Y[2*no_of_species]                # Solid Temperature
    T_f = Y[2*no_of_species + 1]            # Fluid Temperature

    if (n == m):
        bif_par = Y[-1]
    else:
        bif_par = Y0

    #### Identifying the fixed inputs and the bifurcation parameter
    fixed_var = ['inlet_ratio', 'tau', 'R_omega', 'T_f_in', 'R_omega_wc', 
                 'particle_density']
    fixed_val = []
    for elem in fixed_var:
        if elem in fixed:
            fixed_val.append(fixed[elem])
        else:
            fixed_val.append(bif_par)
    
    fixed_dict = dict(zip(fixed_var, fixed_val))

    inlet_ratio = fixed_dict['inlet_ratio']
    tau = fixed_dict['tau']
    R_omega = fixed_dict['R_omega']
    T_f_in = fixed_dict['T_f_in']
    R_omega_wc = fixed_dict['R_omega_wc']
    particle_density = fixed_dict['particle_density']

    #### Calculating the inlet mole fractions
    N2 = 0      # For the time being, we use pure O2
    F_A_in = inlet_ratio/(N2 + inlet_ratio + 1)
    F_B_in = 1/(N2 + inlet_ratio + 1)

    F_in = 1e-08 * np.ones(no_of_species, dtype=float)
    
    A_index = species_ID[inlet_species]
    B_index = species_ID['O2']
    
    F_in[A_index] = F_A_in
    F_in[B_index] = F_B_in
    F_T0 = np.sum(F_in) # This will always be 1, as per out calculations 

    #### Defining mole fractions and concentrations
    C_total = 101325 /(8.314 * T_f)
    C_in = 101325 /(8.314 * T_f_in)
    Y_j = F_j/np.sum(F_j)
    C_f = Y_j * C_total
    V = tau * F_T0/C_in    

    #### Unpacking the constants
    alpha_f_const = const['alpha_f']
    eps_f = const['eps_f']
    D_AB = const['D_AB']
    nu = const['nu']

    #### Calculating the bulk diffusivity of the species by 
    #### Wilke and Fairbanks Equation
    D_all = D_AB[0] * T_f ** D_AB[1]
    D_m = np.zeros(no_of_species)
    for i in range(no_of_species):
        D_m[i] = (1 - Y_j[i])/np.sum(np.r_[Y_j[:i], Y_j[i+1:]]/D_all[:, i])

    alpha_f = alpha_f_const * T_f ** 1.75

    #### Calculating the dependent parameters
    a_v = eps_f/R_omega
    C_p_j = thermo_object.Cp_at_T(T_f)           
    C_p_avg = thermo_object.avg_Cp([300, 2000], 
                                    output='average')
    C_p_integ = thermo_object.avg_Cp([T_f_in, T_f], 
                                    output='integrate')
    del_H_rxn = thermo_object.del_H_reaction(T_f)
    
    C_pf_hat = np.dot(C_p_avg.T, Y_j)
    C_pf_tilde = C_total * C_pf_hat              # Volumetric Cp, in J/m3-K

    #### Calculation of Heat and mass transfer coefficients
    k_c_m = 2*(D_m/R_omega) + (D_m/tau) ** 0.5
    h_hat = C_pf_tilde * (2*(alpha_f/R_omega) + (alpha_f/tau) ** 0.5)
    
    #### Calculation of reaction rates
    homo, cat, homo_index, cat_index = react_const
    homo_basis, cat_basis = rate_basis
    homo_units, cat_units = rate_units

    if (flag == 'cat'):
        homo_rate = np.zeros(len(homo_index))    
        cat_rate = cat.act_rate(species, cat_basis, C_s, T_s)
    elif (flag == 'homo'):
        cat_rate = np.zeros(len(cat_index))
        homo_rate = homo.act_rate(species, homo_basis, C_f, T_f)
    else:
        homo_rate = homo.act_rate(species, homo_basis, C_f, T_f)
        cat_rate = cat.act_rate(species, cat_basis, C_s, T_s)

    #### Unit correction
    homo_units, cat_units = rate_units

    #### This is not required for the time being
    # We got to check the units and perform subsequent calculations 
    if homo_basis == 'mole_fraction':
        if homo_units != 'second':
            raise Exception ('There is a discrepancy '
                             'in the homogeneous reaction rate')

    if (cat_units == 'kg_sec'):
        cat_rate *= particle_density
    elif (cat_units == 'gm_sec'):
        cat_rate *= particle_density * 1000

    #### Dealing with stoichiometric coefficients matrix and Heat of reactions
    nu_homo = nu.T[homo_index]
    nu_cat = nu.T[cat_index]

    del_H_homo = -del_H_rxn[homo_index]
    del_H_cat = -del_H_rxn[cat_index]

    #### Pre-allocation of return variable
    F = np.zeros(n)

    #### Setting up the equations
    F[:no_of_species] = (F_in - F_j)/ V \
                      + eps_f * np.dot(nu_homo.T, homo_rate) \
                      - k_c_m * a_v * (C_f - C_s)
    
    F[no_of_species : 2*no_of_species] = k_c_m * (C_f - C_s) \
                                       + R_omega_wc \
                                       * np.dot(nu_cat.T, cat_rate)
                                                           
    F[2*no_of_species] = -1/V * np.dot(F_in.T, C_p_integ) \
                       + eps_f * np.dot(del_H_homo.T, homo_rate) \
                       - h_hat * a_v * (T_f - T_s)
    
    F[2*no_of_species + 1] = h_hat * (T_f - T_s) \
                           + R_omega_wc * np.dot(del_H_cat.T, cat_rate)

    #### Conitnuation equation using Pseudo-Arc Length method
    if (n == m):
        F[-1] = np.sum((Y - Y0)**2) - del_s ** 2

    return F


def bif_dia_jacobian(Y, Y0, inlet_species, fixed, const, react_const, flag,  
                     rate_basis, rate_units, del_s= 0):
    '''
    This function calculates the Jacobian of the bif_dia_func by second
    order derivative approximation. It can be used while solving for the
    bifurcation diagram and bifurcation sets.
    '''
    #### Checking the inputs
    n = Y.size
    
    eps = 1e-06
    J = np.zeros([n, n])

    for i in range(n):
        Y_pos = Y.copy()
        Y_neg = Y.copy()
        h = eps * Y[i]

        if (h == 0):
            h = eps
        Y_pos[i] = Y[i] + h
        Y_neg[i] = Y[i] - h

        if Y_neg[i] < 0:
            Y_pos_2h = Y.copy()
            Y_pos_2h[i] = Y[i] + 2*h
            J[:, i] = 1/(2*h) * (4 * bif_dia_func(Y_pos, Y0, inlet_species, 
                                fixed, const, react_const, flag,  rate_basis, 
                                rate_units, del_s)
                        - 3 * bif_dia_func(Y, Y0, inlet_species, fixed, const, 
                                           react_const, flag,  rate_basis, 
                                           rate_units, del_s)
                        - bif_dia_func(Y_pos_2h, Y0, inlet_species, fixed, const, 
                                       react_const, flag,  rate_basis, rate_units, del_s))
        else:
            J[:, i] = (bif_dia_func(Y_pos, Y0, inlet_species, fixed, const, react_const, 
                                    flag,  rate_basis, rate_units, del_s) 
                         - bif_dia_func(Y_neg, Y0, inlet_species, fixed, const, react_const, 
                                        flag,  rate_basis, rate_units, del_s))/(2*h)
        
    return J



    
def bif_dia_solver(fixed_dict, bif_par_dict, react_system, system, catalyst, 
                   rate_basis, rate_units, inlet_species, break_val, 
                   step_change, max_val, options):
    '''
    This function solves for the bifurcation diagram.
    
    fixed_dict= Dictionary of fixed variables and values.
    bif_par_dict= Dictionary of Bifurcation variable and value.
    react_system= Describes the reaction system chosen, e.g: 'OCM_three_reaction.txt'.

    system= Whether the system is catalytic only, homogeneous only, or homogeneous-heterogeneous coupled.
    inlet_species= The hydrocarbon species, as of now just hydrocarbon and O2 is taken as inlet.
    break_val= break value of the bifurcation parameter.
    
    step_change= step_change value of the bifurcation parameter, necessary in Pseudo-Arc Length Continuation.
    max_val= maximum value of the bifurcation parameter.
    options= Solver Options.
    '''

    #### Constant parameters involved with the system
    const, thermo_object = constants.fixed_parameters(react_system)
    species = const['species']
    no_of_species = len(species)
    ID = np.arange(no_of_species)
    species_ID = dict(zip(species, ID))
    print(species_ID)

    #### Manipulation of input variables
    all_dict = dict(fixed_dict, **bif_par_dict)

    bif_par_var = list(bif_par_dict.keys())[0]
    bif_par_val = bif_par_dict.get(bif_par_var)

    #### Initial guess
    inlet_ratio = all_dict['inlet_ratio']
    N2 = 0  # For the time being, we use pure O2
    F_A_in = inlet_ratio/(N2 + inlet_ratio + 1)
    F_B_in = 1/(N2 + inlet_ratio + 1)

    F_in = 1e-08 * np.ones(no_of_species)
     
    A_index = species_ID[inlet_species]
    B_index = species_ID['O2']
    
    F_in[A_index] = F_A_in
    F_in[B_index] = F_B_in
    state_var_val = np.r_[F_in, F_in, all_dict['T_f_in'], 
                          all_dict['T_f_in'], bif_par_val]
   
    #### Generating the reaction data
    homo, cat, homo_index, cat_index = react.instantiate(react_system, 
                                       const, catalyst)
    react_const = [homo, cat, homo_index, cat_index]
    
    #### Function name
    func = bif_dia_func
    jac = bif_dia_jacobian

    if options['Testing']:
        #### Just testing the function(s)
        Ysol = func(state_var_val, bif_par_val, inlet_species, fixed_dict, 
                    const, thermo_object, react_const, system,  rate_basis, 
                    rate_units, 0)
        return Ysol

    else:
        if options['continuation'] == 'arc_length':
            raise Exception ('This is not the correct solver to solve the problem')
        else:
            #### Solving for the bifurcation diagram
            iter_count = 0
            _run = True
            rand_count = 0
            
            while(_run):
                
                _run = False
                
                #### First Intial Guess
                while(True):

                    if (iter_count != 0):
                        rand_count += 1
                        print(rand_count)
                        if (rand_count == 1):
                            print('Working on generating the next initial guess')
                        ### Guessing the initial point through random number generation
                        T_s = np.random.randint(T_bd[-3, -1], 3000)
                        T_f = np.random.randint(T_bd[-2,-1], T_s)
                            
                        state_var_val = np.r_[1e-01*np.ones(2*no_of_species), T_s, T_s]
                        bif_par_val = break_val
                        
#                        while(True):        
#                            #### Generating the new initial guess after the break point
#                            print('The last solution point is:\n{}'.format(T_bd[:,-1]))
#                            
#                            state_var_str = (input('Enter an initial guess close to this first point '
#                                                        '[comma-separeted]: ')).split(',')
#                            state_var_list = []
#                            for elem in state_var_str:
#                                try:
#                                    elem_float = float(elem)
#                                except ValueError:
#                                    print('Enter a valid float-type data input')
#                                else:
#                                    state_var_list.append(elem_float)
#                                    
#                            if len(state_var_list) == (2*no_of_species + 2):
#                                state_var_val = np.array(state_var_list)
#                                #### Checking the validity of the input
#                                if (any(state_var_val < 0)) or (state_var_val[-1] < T_bd[-2, -1]):
#                                    print('Enter a valid input')
#                                else:
#                                    bif_par_val -= step_change
#                                    break

                    ###  Solving with first initial guess
                    sol0 = root(func, state_var_val, args= (bif_par_val, 
                                inlet_species, fixed_dict, const,
                                thermo_object, react_const, system,  
                                rate_basis, rate_units))
                    act_sol = sol0.x
                    
                    #### Checking the validity of the solution
                    if (iter_count != 0):
                        
                        delta = func(act_sol, bif_par_val, inlet_species, fixed_dict, const, 
                                     thermo_object, react_const, system,  rate_basis, rate_units)
                        
                        if any(act_sol < 0) or any(act_sol.imag != 0) or any(abs(delta) > 1e-03):
                            print('Didn\'t get a perfect solution, trying again!!!')
                        
                        elif (abs(act_sol[-1] - bif_par_val) <= 1):
                            print('Oops!!! Got the first point, trying again.')
                        
#                        elif (abs(act_sol[-1] - T_bd[-2,-1]) <= 5):
#                            print('Got the previous point, trying again.')
                        
                        else:
                            print('The last solution point is:\n{}'.format(T_bd[:,-1]))
                            sol_check = input('Do you think this is the solution: \n{}'.format(act_sol))

                    else:
                        sol_check = 'y'
                        
                    if (sol_check == 'y'):
                        iter_count += 1
                        sol_check = 'n'
                        Y0 = np.r_[act_sol, bif_par_val]
                        try:
                            T_bd = np.c_[T_bd, Y0[:]]
                        except NameError:
                            T_bd = Y0[:]
                        finally:
                            break
                print('We got the first point, moving on now!')
                
                #### Second initial guess
                bif_par_val += step_change
                
                ###  Solving with second initial guess
                sol0 = root(func, state_var_val, args= (bif_par_val, 
                            inlet_species, fixed_dict, const,
                            thermo_object, react_const, system,
                            rate_basis, rate_units))
                act_sol = sol0.x
                
                print('We got the second point too')
                iter_count += 1
                Y1 = np.r_[act_sol, bif_par_val]
                T_bd = np.c_[T_bd, Y1[:]]
#                print(T_bd)
                
                #### Calculation of the suitable delta_s value
                if bif_par_var == 'tau':
                    plot_flag = 'log'
                    delta_s = np.log10(Y1[-1]/Y0[-1])
                else:
                    plot_flag = 'norm'
                    delta_s = np.linalg.norm(Y1 - Y0)
                    
                #### Continuation method
                while(Y1[-1] <= max_val):
                    print(Y1[-1])
                    iter_count += 1
                    Y_guess = 2*Y1 - Y0
                    delta_s = np.linalg.norm(Y1-Y0)
                    sol2 = root(func, Y_guess, args= (Y1, inlet_species, 
                                fixed_dict, const, 
                                thermo_object, react_const, system,  
                                rate_basis, rate_units, delta_s))
                    Y2 = sol2.x
                    T_bd = np.c_[T_bd, Y2]
                    Y0[:] = Y1[:]
                    Y1[:] = Y2[:]
                    
                    if (Y1[-1] < break_val):
                        print('Extinction point is lower than the '
                              'break point ({}).'.format(break_val))
                        
                        while(True):
                            A = input('Do you want to start the calculation '
                                      'with a different initial guess [y/n]?: ')
                            
                            if (A.lower() == 'y') or (A.lower() == 'n'):
                                break
                            else:
                                print('Enter \'y\' for a yes and \'n\' for a '
                                    'no, no other input will be considered')
                        
                        if (A.lower() == 'y'):
                            _run = True
                        break
                                            
        #### Plotting the figure
        fig = plt.figure()
        if plot_flag == 'log':
            plt.semilogx(T_bd[-1, :], T_bd[-2, :])
        else:
            plt.plot(T_bd[-1, :], T_bd[-2, :])
        plt.show
        
        #### Storing data
        data_vals = [value for (key, value) in sorted(fixed_dict.items())]
        n = len(data_vals)
        filename = 'bif_dia'
        
        for i in range(n):
            filename += '_{}'.format(data_vals[i])
            
        print(filename)
        
        if options['save']:
            #### Saving Data
            react_filename, ext = os.path.splitext(react_system)
            
            Target_folder = os.path.join(os.getcwd(), react_filename, 
                                    catalyst.lower(), 'Data', system)
    
            if os.path.isdir(Target_folder) == False:
                New_Folder = react_filename + '/' + catalyst.lower() + '/' \
                           + '/Data/' + system
                os.makedirs(New_Folder)
    
            FullfileName = os.path.join(Target_folder, filename)
            np.savez(FullfileName, T_bd, Y_in, species_ID)
            
            #### Saving Diagrams (Why I don't know)
            Target_folder = os.path.join(os.getcwd(), react_filename, 
                                        catalyst.lower(), 'Diagram', system)
    
            if os.path.isdir(Target_folder) == False:
                New_Folder = react_filename + '/' + catalyst.lower() + '/' \
                           + '/Diagram/' + system
                os.makedirs(New_Folder)
    
            dia_filename = filename + '.png'
            FullfileName = os.path.join(Target_folder, dia_filename)
            fig.savefig(FullfileName)
        
        #### Returning the final result
        return T_bd
    
def bif_set_func(Y, Y0, inlet_species, fixed, const, react_const, flag, 
                rate_basis, rate_units, del_s= 0, get_jacobian=False):
    '''
    This function is used to generate the bifurcation sets.
    If get_jacobian is set to True, it returns the numerically
    calculated jacobian only, else it returns the values
    of the functions to be solved.

    #### Note: Here bifurcation set is actually the ignition-
    extinction locus, for that reason 'T_f_in' is taken
    out of the fixed_var (a local variable).
    '''

    #### Assert statements
    assert (flag == 'cat') or (flag == 'coup') or (flag == 'homo'), ('This '
           'program can only handle catalytic ("cat") or coupled ("coup") system.')

    #### Unpacking the Y vector

    #### Species identifier
    species = const['species']
    no_of_species = len(species)
    ID = np.arange(no_of_species)
    species_ID = dict(zip(species, ID))
    
    #### Unpacking the Y vector
    n = Y.size         
    try:
        m = Y0.size
    except AttributeError:
        m = 1
    
    if n == m:
        bif_par = Y[-1]
    else:
        bif_par = Y0

    no_of_func = 2*(no_of_species + 1)
    y0 = Y[:no_of_func]                     # Nullspace vector
    X = Y[no_of_func : 2*no_of_func]        # State Variables

    #### Identifying the fixed inputs and the bifurcation parameter
    fixed_var = ['inlet_ratio', 'tau', 'R_omega', 'R_omega_wc', 'particle_density']
    fixed_val = []
    for elem in fixed_var:
        if elem in fixed:
            fixed_val.append(fixed[elem])
        else:
            fixed_val.append(bif_par)
    
    fixed_dict = dict(zip(fixed_var, fixed_val))
    
    #### Calculation of the Jacobian of the actual functions
    J = bif_dia_jacobian(X, Y[no_of_func+1], inlet_species, fixed_dict, const, react_const, 
                         flag, rate_basis, rate_units)

    if get_jacobian:
        return J
    else:
        #### Pre-allocation of return variable
        F = np.zeros(n)

        #### Setting up the equations to solve for bifurcation set
        F[:no_of_func] = bif_dia_func(X, Y[no_of_func+1], inlet_species, fixed_dict,
                                      const, react_const, flag, rate_basis, rate_units)

        F[no_of_func : 2*no_of_func] = np.dot(J, y0)

        F[2*no_of_func] = np.dot(y0.T, y0) - 1

        if n == m:
            F[-1] = np.sum((Y - Y0)**2) - del_s ** 2

        return F

def null_space(y, J):
    '''This function is required to get the initial guess of y0.'''
    f1 = np.dot(J, y)
    f2 = np.dot(y.T, y) - 1
    f = np.r_[f1, f2]

    return f

def bif_set_solver(fixed_dict, bif_par_dict, react_system, system, catalyst, rate_basis,
                   rate_units, inlet_species, break_dict, step_change, max_val, options):

    '''
    This function solves for the bifurcation set (or ignition-extinction locus)
    The significance of different arguments are same as that of the bif_dia_solver.
    '''
    # Constant parameters involved with the system
    const = constants.fixed_parameters(react_system)
    species = const['species']
    no_of_species = len(species)

    # Manipulation of input variables
    all_dict = dict(fixed_dict, **bif_par_dict)

    bif_par_var = list(bif_par_dict.keys())[0]
    bif_par_val = bif_par_dict.get(bif_par_var)

    # Generating the reaction data
    homo, cat, homo_index, cat_index = react.instantiate(react_system, const, catalyst)
    react_const = [homo, cat, homo_index, cat_index]

    # First Dataset
    npzfile = _load_file(react_system, system, catalyst, all_dict)
    T_bd_arr, Y_in_arr, species_ID_arr = npzfile.files
    T_bd_0 = npzfile[T_bd_arr]
    
    index_0 = ig_ext_point_calculator(T_bd_0, system, 'T_f_in', break_dict, fulloutput=True)

    # Second Dataset
    all_dict[bif_par_var] += step_change

    npzfile = _load_file(react_system, system, catalyst, all_dict)
    T_bd_arr, Y_in_arr, species_ID_arr = npzfile.files
    T_bd_1 = npzfile[T_bd_arr]
    
    index_1 = ig_ext_point_calculator(T_bd_1, system, 'T_f_in', break_dict, fulloutput=True)
    
    # Function names
    func = bif_set_func
#    jac = bif_set_jacobian

    # Solving for ignition-extinction locus

    # First Ignition
    y0 = np.zeros(2*(no_of_species + 1))
    y0[1] = 1
    x0 = T_bd_0[:, index_0[0]]
    x_init = np.r_[y0, x0]

    J_0 = func(x_init, bif_par_val, inlet_species, fixed_dict, const, 
                react_const, system, rate_basis, rate_units, get_jacobian=True)
    y_sol_0 = root(null_space, y0, args= J_0, method= 'lm')
    X_init_0 = np.r_[y_sol_0.x, x0]

    sol0 = root(func, X_init_0, args=(bif_par_val, inlet_species, fixed_dict, const,
                react_const, system, rate_basis, rate_units))
    
    Y0 = np.r_[sol0.x, bif_par_val]
    T_bs = Y0[:]

    # Second solution
    bif_par_val += step_change

    x1 = T_bd_1[:, index_1[0]]
    x_init = np.r_[y0, x1]

    J_1 = func(x_init, bif_par_val, inlet_species, fixed_dict, const, 
                react_const, system, rate_basis, rate_units, get_jacobian=True)
    y_sol_1 = root(null_space, y0, args= J_1, method= 'lm')
    X_init_1 = np.r_[y_sol_1.x, x1]

    sol1 = root(func, X_init_1, args=(bif_par_val, inlet_species, fixed_dict, const,
                react_const, system, rate_basis, rate_units))
    
    Y1 = np.r_[sol1.x, bif_par_val]
    T_bs = np.c_[T_bs, Y1[:]]

    plot_flag = 'norm'
    delta_s = np.linalg.norm(Y1 - Y0)
    
    # Continuation method
    while(Y1[-1] <= max_val 
          and Y1[-1] >= break_dict[bif_par_var]
          and Y1[-2] >= break_dict['T_f_in']):
        print(Y1[-1])
        Y_guess = 2*Y1 - Y0
        
        sol2 = root(func, Y_guess, args=(Y1, inlet_species, fixed_dict, const, react_const,
                    system, rate_basis, rate_units, delta_s))
        Y2 = sol2.x

        T_bs = np.c_[T_bs, Y2]
        Y0[:] = Y1[:]
        Y1[:] = Y2[:]
    
    #### Plotting the figure
    fig = plt.figure()
    if plot_flag == 'log':
        plt.semilogx(T_bs[-1, :], T_bs[-2, :])
    else:
        plt.plot(T_bs[-1, :], T_bs[-2, :])
    plt.show
    
    #### Storing data
    data_vals = [value for (key, value) in sorted(fixed_dict.items())]
    n = len(data_vals)
    filename = 'bif_set'
    
    for i in range(n):
        filename += '_{}'.format(data_vals[i])
        
    print(filename)
    
    if options['save']:
        #### Saving Data
        react_filename, ext = os.path.splitext(react_system)
        
        Target_folder = os.path.join(os.getcwd(), react_filename, catalyst.lower(), 'Data', system)
    
        if os.path.isdir(Target_folder) == False:
            New_Folder = react_filename + '/' + catalyst.lower() + '/' + '/Data/' + system
            os.makedirs(New_Folder)
    
        FullfileName = os.path.join(Target_folder, filename)
        np.savez(FullfileName, T_bs, Y_in, species_ID)
        
        #### Saving Diagrams (Why I don't know)
        Target_folder = os.path.join(os.getcwd(), react_filename, catalyst.lower(), 'Diagram', system)
    
        if os.path.isdir(Target_folder) == False:
            New_Folder = react_filename + '/' + catalyst.lower() + '/' + '/Diagram/' + system
            os.makedirs(New_Folder)
    
        dia_filename = filename + '.png'
        FullfileName = os.path.join(Target_folder, dia_filename)
        fig.savefig(FullfileName)
    
    #### Returning the final result
    return T_bs

def ig_ext_point_calculator(T_bd, system, bif_par_var, break_dict, \
                            fulloutput= True):
    '''
    This function calculates the indices of ignition and extinction points and
    returns them in form of ndarray. It takes in one array (T_f) in which it
    searches for ignition-extinction points and the type of the system, (whether
    the system is 'homo'-geneous or 'cat'-alytic only, or thermally 'coup'-led.
    They keyword arguments are the bifurcation parameter (bif_par) and
    fulloutput. If fulloutput is set to True the ignition and extinction points
    will be printed out. 
    And finally, based on the system specified it returns the indices from which
    the entire dataset at ignition and extinction points can be calculated.
    '''
    #### Manipulating the inputs
    T = T_bd[-1, :]
    T_break = break_dict[bif_par_var]
    
    #### Initialization of certain values
    n = T.shape[0]  
    index = 0
    i =  0
    
    if system == 'coup':
        i_max = 4
        T_pt = np.zeros(i_max, dtype = int)
    else:
        i_max = 2
        T_pt = np.zeros(i_max, dtype = int)

    #### Search and find for ignition-extincion points
    while(index < n-1) and (i < i_max-1):
        
        #### Ignition_point
        while (index < n-1):
            if (T[index] <= T[index + 1]):
                index += 1
            else:
                ig_point = index
                break
        
        #### Index checking and storing data
        if (index == n-1):
            break
        else:
            T_pt[i] = ig_point
            i += 1
        
        index += 1
        
        #### Extinction point
        ext_first = -1
        while(index <= n-1):
            if T[index] < T[index-1]:
                if T[index] > T_break:
                    index += 1
                else:
                    break
            else:
                ext_first = index
                break
        
        T_pt[i] = ext_first
        i += 1
        index += 1

    #### Packing and shipping
    if fulloutput == True:
        
        act_var = ['T_f_in', 'tau', 'y_A_in', 'R_omega']
        units = ['K', 'secs', 'mole fraction', 'm']
        units_dict = dict(zip(act_var, units))
        unit = units_dict[bif_par_var]

        if system == 'homo':
            if T_pt[0] == 0:
                print('Oops!!! No ignition or extinction behavior is observed.')
            elif T_pt[0] != 0 and T_pt[1] == -1:
                print('Homogeneous ignition is observed at {0} {2} and '
                        'extinction is below {1} {2}.'.format(T[T_pt[0]], \
                        T_break, unit))
            else:
                print('Homogeneous ignition is observed at {0} {2} and '
                        'extinction is observed at {1} {2}.'.format(T[T_pt[0]],\
                        T[T_pt[1]], unit))
    
        elif system == 'cat':
            if T_pt[0] == 0:
                print('Oops!!! No ignition or extinction behavior is observed.')
            elif T_pt[0] != 0 and T_pt[1] == -1:
                print('Catalytic ignition is observed at {0} {2} and extinction'
                        ' is below {1} {2}.'.format(T[T_pt[0]], T_break, unit))
            else:
                print('Catalytic ignition is observed at {0} {2} and extinction'
                        ' is observed at {1} {2}.'.format(T[T_pt[0]], \
                        T[T_pt[1]], unit))

        else:
            #### The system is coupled
            if T_pt[0] == 0:
                print('Oops!!! No ignition or extinction behavior is observed.')
            elif T_pt[1] == -1 and T_pt[2] == 0:
                print('Only one ignition is observed at {0} {2} and the '
                        'extinction temperature is below {1} '
                        '{2}.'.format(T[T_pt[0]], T_break, unit))
            elif T_pt[1] == -1 and T_pt[2] != 0 and T_pt[3] == -1:
                print('The first ignition is observed at {0} {2} and the second'
                        ' ignition is at {1} {2}.'.format(T[T_pt[0]],\
                        T[T_pt[2]], unit))
                print('The extinction temperature in both cases are below {0} '
                        '{1}.'.format(T_break, unit))
            elif T_pt[3] == -1:
                print('The first ignition is observed at {0} {2} and the second'
                        ' ignition is at {0} {2}.'.format(T[T_pt[0]],\
                        T[T_pt[2]], unit))
                print('First extinction is at {0} {2} and the second extinction'
                        ' is below {1} {2}.'.format(T[T_pt[1]], T_break, unit))
            elif T_pt[2] == 0 and T_pt[3] == 0:
                print('The only ignition point is at {0} {1}.'.format(T[T_pt[0]], unit))
                print('And the only extinction pont is at {0} {1}.'.format(T[T_pt[1]], unit))
            else:
                print('The first ignition is observed at {0} {2} and the second'
                        ' is at {1} {2}.'.format(T[T_pt[0]], T[T_pt[2]], unit))
                print('The first extinction is observed at {0} {2} and the' 
                ' second is at {1} {2}'.format(T[T_pt[1]], T[T_pt[3]], unit))
        
    return T_pt


def bif_dia_plot(fixed_dict, bif_par_var, react_system, system, catalyst, inlet_species, options):
    '''This function will load the correct file in order to plot the different bifurcation diagrams.'''

    #### Retreiving the data from .npz file
    npzfile = _load_file(react_system, system, catalyst, fixed_dict)
    T_bd_arr, Y_in_arr, species_ID_arr = npzfile.files
    T_bd = npzfile[T_bd_arr]
    Y_in = npzfile[Y_in_arr]
    species_ID = npzfile[species_ID_arr].item()            # npzfile[species_ID_arr] is an object
    
    #### Retrieving the species_specific data from T_bd
    hc_index = species_ID[inlet_species]
    conv_hc = 1 - T_bd[hc_index, :]/Y_in[hc_index]

    O2_index = species_ID['O2']
    conv_O2 = 1 - T_bd[O2_index, :]/Y_in[O2_index]
    
    #### Plotting Area
    plot_dict = options['plots']
    
    if bif_par_var == 'tau':
        xplottype = 'log'
        x_axis = 'Residence Time (s)'
    else:
        xplottype = 'normal'
        x_axis = 'Inlet Fluid Temperature ' + r'$\mathbf{T_{f,in}}$' + ' (K)'
        
    
    #### Exit Fluid Temperature vs Inlet Fluid Temperature
    if plot_dict['fluid_temp']:

        fig, ax1 = plt.subplots()
        
        if xplottype == 'log':
            ax1.semilogx(T_bd[-1, :], T_bd[-2, :], color= 'b', linewidth= 2.0)
        else:
            ax1.plot(T_bd[-1, :], T_bd[-2, :], color= 'b', linewidth= 2.0)
        ax1.set_xlabel(x_axis, fontsize= 14, fontweight= 'bold')
        ax1.set_ylabel(('Exit Fluid Temperature ' + r'$\mathbf{T_s}$' + ' (K)'), 
                         fontsize= 14, fontweight= 'bold')
        axis_limits = options['xaxis_lim'] + options['yaxis_lim']
        ax1.axis(axis_limits)
    
    #### Solid Temperature vs Inlet Fluid Temperature
    if plot_dict['solid_temp']:

        fig, ax2 = plt.subplots()
        
        if xplottype == 'log':
            ax2.semilogx(T_bd[-1, :], T_bd[-3, :], color= 'k', linewidth= 2.0)
        else:
            ax2.plot(T_bd[-1, :], T_bd[-3, :], color= 'k', linewidth= 2.0)
        ax2.set_xlabel(x_axis, fontsize= 14)
        ax2.set_ylabel(('Catalyst Surface Temperature ' + r'$T_s$' + ' (K)'), fontsize= 14)

        axis_limits = options['xaxis_lim'] + options['yaxis_lim']
        ax2.axis(axis_limits)
    
    #### Conversion of Hydrocarbon and O2
    if plot_dict['conversion']:

        fig2, (ax1, ax2) = plt.subplots(1, 2)
        fig2.tight_layout()
        fig2.subplots_adjust(wspace= 0.4)

        if xplottype == 'log':
            ax1.semilogx(T_bd[-1, :], conv_hc, color= 'b', linewidth= 2.0)
            ax2.semilogx(T_bd[-1, :], conv_O2, color= 'r', linewidth= 2.0)
        else:    
            ax1.plot(T_bd[-1, :], conv_hc, color= 'b', linewidth= 2.0)
            ax2.plot(T_bd[-1, :], conv_O2, color= 'r', linewidth= 2.0)
            
        ax1.set_xlabel(x_axis, fontsize= 14)
        ax1.set_ylabel(('Conversion of ' + r'$CH_4$'), fontsize= 14)
        axis_limits = options['xaxis_lim'] + [0, 1]
        ax1.axis(axis_limits)
        
        ax2.set_xlabel(x_axis, fontsize= 14)
        ax2.set_ylabel(('Conversion of ' + r'$O_2$'), fontsize= 14)
        ax2.axis(axis_limits)
        
#        ax1m.plot(T_bd[-1, :], no_of_moles_fluid, color= 'b')
#        ax1m.plot(T_bd[-1, :], no_of_moles_solid, color= 'k')
        
        
    #### Yields of all 'products', (Identifying the compound and then identifying limiting reactant is
    #### important, we will do it later)
    if plot_dict['yield_all']:

        products = options['products']
        if products:
            fig, ax_y = plt.subplots()
            label = []        
            not_calc_elem = []

            for elem in products:
                elem_index = species_ID.get(elem, None)
                carbon_no = species_identifier(elem)
                
                if (elem_index != None) and (carbon_no != 0):
                    yield_elem = T_bd[elem_index, :] * carbon_no/Y_in[hc_index]
                    
                    if xplottype == 'log':
                        ax_y.semilogx(T_bd[-1, :], yield_elem, linewidth= 2.0)
                    else:
                        ax_y.plot(T_bd[-1, :], yield_elem, linewidth= 2.0)
                    label.append(elem)
                else:
                    not_calc_elem.append(elem)

        ax_y.set_xlabel(x_axis, fontsize= 14)
        ax_y.set_ylabel(('Yield of Products'), fontsize= 14)
        ax_y.legend(tuple(label))
        axis_limits = options['xaxis_lim'] + [0, 1]
        ax_y.axis(axis_limits)

        if not_calc_elem:
            print('The yields of {} are not calculated, the program thinks they are '
                    'unimportant'.format(not_calc_elem))
        
    #### Selectivity of all products
    if plot_dict['select_all']:

        products = options['products']
        if products:
            fig, ax_s = plt.subplots()
            label = []        
            not_calc_elem = []

            for elem in products:
                elem_index = species_ID.get(elem, None)
                carbon_no = species_identifier(elem)
                if (elem_index != None) and (carbon_no != 0):
                    yield_elem = T_bd[elem_index, :] * carbon_no/Y_in[hc_index]
                    selectivity_elem = yield_elem/conv_hc
                    
                    if xplottype == 'log':
                        ax_s.semilogx(T_bd[-1, :], selectivity_elem, linewidth= 2.0)
                    else:
                        ax_s.plot(T_bd[-1, :], selectivity_elem, linewidth= 2.0)
                    label.append(elem)
                else:
                    not_calc_elem.append(elem)
        
        ax_s.set_xlabel(x_axis, fontsize= 14)
        ax_s.set_ylabel(('Selectivity of Products'), fontsize= 14)
        ax_s.legend(tuple(label))
        axis_limits = options['xaxis_lim'] + [0, 1]
        ax_s.axis(axis_limits)    

        if not_calc_elem:
            print('The yields of {} are not calculated, the program thinks they are '
                    'unimportant'.format(not_calc_elem))

    if plot_dict['yield_comb']:
        
        #### Hardcoded yields of products
        CO_index = species_ID.get('CO', None)
        if CO_index >= 0:
            yield_CO = T_bd[CO_index, :]/Y_in[hc_index]
        else:
            yield_CO = 0
    
        CO2_index = species_ID.get('CO2', None)
        if CO2_index >= 0:
            yield_CO2 = T_bd[CO2_index, :]/Y_in[hc_index]
        else:
            yield_CO2 = 0

        yield_COx = (yield_CO + yield_CO2)
        selectivity_COx = yield_COx/conv_hc
    
        C2H6_index = species_ID.get('C2H6', None)
        if C2H6_index >= 0:
            yield_C2H6 = T_bd[C2H6_index, :]*2/Y_in[hc_index]
        else:
            yield_C2H6 = 0

        C2H4_index = species_ID.get('C2H4', None)
        if C2H4_index >= 0:
            yield_C2H4 = T_bd[C2H4_index, :]*2/Y_in[hc_index]
        else:
            yield_C2H4 = 0    
#        print(yield_CO)
#        print(yield_CO2)
#        print(yield_C2H6)
#        print(yield_C2H4)
#        print(hc_index)
        yield_C2 = (yield_C2H6 + yield_C2H4)
        selectivity_C2 = yield_C2/conv_hc
    
        fig, ax_yc = plt.subplots()
        if xplottype == 'log':
            ax_yc.semilogx(T_bd[-1, :], yield_COx, color= 'r', linewidth= 2.0)
            ax_yc.semilogx(T_bd[-1, :], yield_C2, color= 'b', linewidth= 2.0)
        else:
            ax_yc.plot(T_bd[-1, :], yield_COx, color= 'r', linewidth= 2.0)
            ax_yc.plot(T_bd[-1, :], yield_C2, color= 'b', linewidth= 2.0)
        
        label1 = r'$CO_x$' + ' yield'
        label2 = r'$C_{2}$' + ' yield'
    
        ax_yc.set_xlabel(x_axis, fontsize= 14, fontweight= 'bold')
        ax_yc.set_ylabel(('Yield of Products'), fontsize= 14, fontweight= 'bold')
        
        axis_limits = options['xaxis_lim'] + [0, 1]
        ax_yc.axis(axis_limits)
        ax_yc.legend((label1, label2))

        if plot_dict['select_comb']:
            fig, ax_sc = plt.subplots()
            
            if xplottype == 'log':
                ax_sc.semilogx(T_bd[-1, :], selectivity_COx, color= 'r', linewidth= 2.0) 
                ax_sc.semilogx(T_bd[-1, :], selectivity_C2, color= 'b', linewidth= 2.0)
            else:
                ax_sc.plot(T_bd[-1, :], selectivity_COx, color= 'r', linewidth= 2.0)
                ax_sc.plot(T_bd[-1, :], selectivity_C2, color= 'b', linewidth= 2.0)
            
            label1 = r'$CO_x$' + ' selectivity'
            label2 = r'$C_{2}$' + ' selectivity'
        
            ax_sc.set_xlabel(x_axis, fontsize= 14)
            ax_sc.set_ylabel(('Selectivity of Products'), fontsize= 14)
            ax_sc.legend((label1, label2))
        
            ax_sc.axis(axis_limits)
    
    return T_bd

def _load_file(react_system, system, catalyst, fixed_dict):
    '''
    Loads the specific file if it exists and returns
    the file contents.
    '''
    
    #### File Specifications
    data_vals = [value for (key, value) in sorted(fixed_dict.items())]
    n = len(data_vals)
    filename = 'bif_dia'
        
    for i in range(n):
        filename += '_{}'.format(data_vals[i])
            
    filename += '.npz'

    react_filename, ext = os.path.splitext(react_system)

    FullfileName = os.path.join(os.getcwd(), react_filename, catalyst.lower(), 'Data', system, filename)

    if os.path.exists(FullfileName):
        npzfile = np.load(FullfileName)
        return npzfile
    else:
        raise FileNotFoundError('File does not exist.')


def species_identifier(species):
    
    elements = []
    elements_count = []
    index = -1
    
    #### Identification of the species
    for e1 in species:
        try:
            float_e1 = float(e1)
            try:
                elements_count[i] = elements_count[i] - 1 + float_e1
                del i
            except NameError:    
                elements_count[index] = float_e1
        except ValueError:
            index += 1
            if (e1 not in elements):
                elements.append(e1)
                elements_count.append(1)
            else:
                i = elements.index(e1)
                elements_count[i] += 1
   
    #### Final return dictionary
    species_identity = dict(zip(elements, elements_count))
    carbonNo = species_identity.get('C', 0)
    return carbonNo





