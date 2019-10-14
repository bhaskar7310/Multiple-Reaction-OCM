#### This module named as 'bifurcation' has all the functions and solvers to
#### calculate all kinds of bifurcation diagrams and sets of Short-Monolith
#### Reactor models

#### Author: Bhaskar Sarkar (2018)
import os
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

import constants
import reaction as react
import DERPAR_code as der
from lineplots import _load_file, _rearrange_species

def bif_dia_func(Y, inlet_species, fixed, const, thermo_object, 
                 react_const, flag, rate_basis, rate_units):
    '''
    This function is used to generate bifurcation diagrams. 
    It can identify what are the fixed parameters and what is the 
    bifurcation parameter.
    Write a story about this later.
    '''
    #### Assert statements
    assert (flag == 'cat') or (flag == 'coup') or (flag == 'homo'), ('This '
                            'program can only handle catalytic ("cat") or '
                            'coupled ("coup") system.')

    #### Unpacking the inputs
    
    #### Species identification
    species = const['species']
    no_of_species = len(species)
    ID = np.arange(no_of_species)
    species_ID = dict(zip(species, ID))

    #### Unpacking the Y vector
    n = Y.size
    F_j = Y[:no_of_species]                         
    C_s = Y[no_of_species:2*no_of_species]
    T_s = Y[2*no_of_species]
    T_f = Y[2*no_of_species + 1]                          
    bif_par = Y[-1]

    #### Identifying the fixed inputs and the bifurcation parameter
    fixed_var = ['inlet_ratio', 'pressure', 
                 'tau', 'R_omega', 'T_f_in', 'R_omega_wc', 'particle_density']
    fixed_val = []
    for elem in fixed_var:
        if elem in fixed:
            fixed_val.append(fixed[elem])
        else:
            fixed_val.append(bif_par)
    
    fixed_dict = dict(zip(fixed_var, fixed_val))

    inlet_ratio = fixed_dict['inlet_ratio']
    P = fixed_dict['pressure']
    tau = fixed_dict['tau']
    R_omega = fixed_dict['R_omega']
    T_f_in = fixed_dict['T_f_in']
    R_omega_wc = fixed_dict['R_omega_wc']
    particle_density = fixed_dict['particle_density']
    
    #### Calculating the inlet mole fractions
    N2 = 0  # For the time being, we use pure O2
    F_A_in = inlet_ratio/(N2 + inlet_ratio + 1)
    F_B_in = 1/(N2 + inlet_ratio + 1)
    
    F_in = 1e-08 * np.ones(no_of_species, dtype=float)
    
    A_index = species_ID[inlet_species]
    B_index = species_ID['O2']
    
    F_in[A_index] = F_A_in
    F_in[B_index] = F_B_in
    
    F_T0 = np.sum(F_in) # This will always be 1, as per our calculations

    #### Defining mole fractions and concentrations
    C_total = 101325 * P/(8.314 * T_f)
    C_in = 101325 * P/(8.314 * T_f_in)
    Y_j = F_j/np.sum(F_j)
    C_f = Y_j * C_total
    V = tau * F_T0/C_in

    #### Unpacking the constants
    alpha_f_const = const['alpha_f']
    R_omega_w = const['R_omega_w']
    D_AB = const['D_AB']
    nu = const['nu']

    eps_f = 4*R_omega**2/(2*R_omega + R_omega_wc + R_omega_w)**2

    #### Calculating the bulk diffusivity of the species by 
    #### Wilke and Fairbanks Equation
    D_all = D_AB[0] * T_f ** D_AB[1] / P
    D_m = np.zeros(no_of_species)
    for i in range(no_of_species):
        D_m[i] = (1 - Y_j[i])/np.sum(np.r_[Y_j[:i], Y_j[i+1:]]/D_all[:, i])

    alpha_f = alpha_f_const * T_f ** 1.75 / P

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
    Y_homo_react = np.r_[C_f, T_f]
    Y_cat_react = np.r_[C_s, T_s]

    if (flag == 'cat'):
        homo_rate = np.zeros(len(homo_index))    
        cat_rate = cat.act_rate(Y_cat_react, species, cat_basis, P)
    elif (flag == 'homo'):
        cat_rate = np.zeros(len(cat_index))
        homo_rate = homo.act_rate(Y_homo_react, species, homo_basis, P)
    else:
        homo_rate = homo.act_rate(Y_homo_react, species, homo_basis, P)
        cat_rate = cat.act_rate(Y_cat_react, species, cat_basis, P)

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

    #### Stoichiometric coefficients matrix and Heat of reactions
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
#    F[2*no_of_species] = np.dot(F_in.T, C_p_avg) / V * (T_f_in - T_f) \
#                       + eps_f * np.dot(del_H_homo.T, homo_rate) \
#                       - h_hat * a_v * (T_f - T_s)
    
    F[2*no_of_species + 1] = h_hat * (T_f - T_s) \
                           + R_omega_wc * np.dot(del_H_cat.T, cat_rate)

    return F


def jacobian(func_name, Y, fvec, eps, no_of_rows, no_of_cols, 
             start_from, end_at, *args):
    '''
    This function calculates the Jacobian of the function given by 
    'func_name', using forward difference approximation.
    '''
    #### Checking the inputs
    column = 0.
    J = np.zeros([no_of_rows, no_of_cols])
    Y_pos = Y.copy()
    
    for i in range(start_from, end_at):
        h = eps * Y[i]
        if (h == 0):
            h = eps
        Y_pos[i] = Y[i] + h
        column = (func_name(Y_pos, *args) - fvec)/h
        J[:, i-start_from] = column[:-1]
        Y_pos[i] = Y_pos[i] - h
        
    return J

    
def bif_dia_solver(fixed_dict, bif_par_dict, react_system, system, catalyst, 
                   rate_basis, rate_units, inlet_species, break_val, 
                   step_change, max_val, options):
    '''
    This function solves for the bifurcation diagram.
    
    fixed_dict= Dictionary of fixed variables and values.
    bif_par_dict= Dictionary of Bifurcation variable and value.
    react_system= Describes the reaction system chosen, 
                  e.g: 'OCM_three_reaction.txt'.

    system= Whether the system is catalytic only, homogeneous only, or 
            homogeneous-heterogeneous coupled.
    inlet_species= The hydrocarbon species, as of now just hydrocarbon and 
                   O2 is taken as inlet.
    break_val= break value of the bifurcation parameter.
    
    step_change= step_change value of the bifurcation parameter, necessary in 
                 Pseudo-Arc Length Continuation.
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

    #### Generating the reaction objects
    homo, cat, homo_index, cat_index = react.instantiate(react_system,
                                       const, catalyst)
    react_const = [homo, cat, homo_index, cat_index]
    
    #### Function name
    func = bif_dia_func
    jac = jacobian
    plot_flag = 'norm'
    
    if options['Testing']:
        #### Just testing the function(s)
        print('\nAll the varibles going into the function')
        print(state_var_val)
        print('\nAnd the fixed valriables going into the function')
        print(fixed_dict)
        Ysol = func(state_var_val, inlet_species, fixed_dict, const, 
               thermo_object, react_const, system,  rate_basis, rate_units)
        return Ysol

    elif options['solver'] == 'python': 
        #### Solving the set of equations using Arc-Length method in Python
        no_of_var = len(state_var_val)
        pref = np.ones(no_of_var)
        weights = np.ones(no_of_var)
        weights[-2:] = 1e-03
        jac_eps = options['jac_eps']

        T_bd = der.derpar(func, jac, state_var_val, pref, max_val, break_val,
               weights, jac_eps, initial= False, hh = step_change, 
               maxout = 200000, hhmax = 10*step_change,
               args=(inlet_species, fixed_dict, const, 
               thermo_object, react_const, system, rate_basis, rate_units)) 

    else:
        #### Solving the set of equations using Arc-Length method in Fortran
        eps = 1e-08
        w = np.ones(no_of_var, dtype=float, order='F')
        initial = 0
        itin = 50
        
        hh = 0.05
        hmax = 0.1 * np.ones(no_of_var, dtype=float, order='F')
        ndir = np.ones(no_of_var, dtype=np.int64, order='F')
        
        e = 1e-06
        mxadms = 4
        ncorr = 4
        ncrad = 0
        maxout = 4

        nout, out  = arc_length.derpar(func, jac, state_var_val, 
                     break_val, max_val, eps, w, initial, itin, hh, 
                     hmax, pref, ndir, e, mxadms, ncorr, ncrad, maxout, 
                     func_extra_args=(inlet_species, fixed_dict, const, 
                     thermo_object, react_const, system, rate_basis,
                     rate_units),
                     jac_extra_args=(inlet_species, fixed_dict, const,
                     thermo_object, react_const, system, rate_basis, 
                     rate_units))
        
        out_act = out[:nout,:-1]
        T_bd = out_act.T
        
    #### Plotting the figure
    fig = plt.figure()
    if plot_flag == 'log':
        plt.semilogx(T_bd[-1, :], T_bd[-2, :])
    else:
        plt.plot(T_bd[-1, :], T_bd[-2, :])
    plt.show()
    
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
        np.savez(FullfileName, T_bd, F_in, species_ID)
        
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


def null_space(y, J, method='root'):
    '''This function is required to get the initial guess of y0.'''
    f1 = np.dot(J, y)
    f2 = np.dot(y.T, y) - 1
    f = np.r_[f1, f2]

    return f


def bif_set_func(Y, inlet_species, fixed, const, thermo_object, 
                 react_const, flag, rate_basis, rate_units, options,
                 get_function=False, get_jacobian=False,
                 get_second_deriv=False):
    '''
    This part will be taken from our pilot bifurcation set code.

    #### Note: Here bifurcation set is actually the ignition-
    extinction locus, for that reason 'T_f_in' is taken
    out of the fixed_var (a local variable).

    Here values of jac_eps and second_deriv_eps are hardcoded.
    '''
    #### Assert statements
    assert (flag == 'cat') or (flag == 'coup') or (flag == 'homo'), ('This '
                            'program can only handle catalytic ("cat") or '
                            'coupled ("coup") system.')

    #### Unpacking the inputs

    #### Species identifier
    species = const['species']
    no_of_species = len(species)

    #### Unpacking the Y vector
    n = Y.size   

    no_of_func = 2*(no_of_species + 1)
    y0 = Y[:no_of_func]                     # Nullspace vector
    X = Y[no_of_func : -1]                  # State Variables
    bif_par = Y[-1]                         # Bifurcation parameter

    #### Identifying the fixed inputs and the bifurcation parameter
    fixed_var = ['inlet_ratio', 'pressure', 'tau', 'R_omega', 'R_omega_wc', 
                 'particle_density']
    fixed_val = []
    for elem in fixed_var:
        if elem in fixed:
            fixed_val.append(fixed[elem])
        else:
            fixed_val.append(bif_par)
    
    fixed_dict = dict(zip(fixed_var, fixed_val))
    
    #### Calculation of the Jacobian of the original functions
    bif_dia_func_val = bif_dia_func(X, inlet_species, fixed_dict, const, 
                                    thermo_object, react_const, flag, 
                                    rate_basis, rate_units)
    #### Here we have to play with it
    jac_eps = options['jac_eps']
    J = jacobian(bif_dia_func, X, bif_dia_func_val, jac_eps,
                      no_of_func, no_of_func, 0, no_of_func, 
                      inlet_species, 
                      fixed_dict, const, thermo_object, react_const, flag, 
                      rate_basis, rate_units)

    fvec = bif_dia_func_val[:-1] 

    #### Shipping the variables requested for
    if get_jacobian and get_function and get_second_deriv:
        second_deriv_eps = options['second_deriv_eps']
        second_deriv = second_derivatives(bif_dia_func, X, J, second_deriv_eps,
                                          jac_eps, inlet_species,
                                          fixed_dict, const, thermo_object,
                                          react_const, flag, rate_basis,
                                          rate_units)
        return fvec, J, second_deriv
    if get_function and get_jacobian:
        return fvec, J

    elif get_function:
        return fvec

    elif get_jacobian:
        return J

    else:
        #### Setting up the equations to solve for bifurcation set
        F = np.zeros(n)

        F[:no_of_func] = fvec[:]

        F[no_of_func : 2*no_of_func] = np.dot(J, y0)

        F[2*no_of_func] = np.dot(y0.T, y0) - 1
        
        return F


def bif_set_solver(fixed_dict, bif_par_dict, react_system, system, catalyst, 
                   rate_basis, rate_units, inlet_species, break_dict, 
                   step_change, max_val, options):

    '''
    This function solves for the bifurcation set 
    (ignition-extinction locus).

    we will write other details later

    Just to point out one important difference between bif_set_solver 
    and bif_dia_solver is that, here we take in the entire break_dict
    '''
    # Constant parameters involved with the system
    const, thermo_object = constants.fixed_parameters(react_system)
    species = const['species']
    no_of_species = len(species)
    ID = np.arange(no_of_species)
    species_ID = dict(zip(species, ID))
    print('This is the new species ID',species_ID)

    # Manipulation of input variables
    all_dict = dict(fixed_dict, **bif_par_dict)

    bif_par_var = list(bif_par_dict.keys())[0]
    bif_par_val = bif_par_dict.get(bif_par_var)
    break_val = break_dict[bif_par_var]
    temp_break_val = break_dict['T_f_in']

    # Generating the reaction objects
    homo, cat, homo_index, cat_index = react.instantiate(react_system, 
                                       const, catalyst)
    react_const = [homo, cat, homo_index, cat_index]

    # First Dataset
    npzfile = _load_file(react_system, system, catalyst, all_dict)
    T_bd_arr, Y_in_arr, species_ID_arr = npzfile.files
    T_bd = npzfile[T_bd_arr]
    species_ID_old = npzfile[species_ID_arr]

    T_bd_wo_noise = T_bd[:, 1000:]
    index = ig_ext_point_calculator(T_bd_wo_noise, system, 
                                      'T_f_in', break_dict, 
                                      fulloutput=True)

    #### Fixing the starting point (ignition/extinction, first/second)
    if options['limit_point'] == 'ignition':
        if options['occurence'] == 'first':
            limit_index = 0
        elif options['occurence'] == 'second':
            limit_index = 2
        else:
            raise Exception('Wrong value of occurence')
    elif options['limit_point'] == 'extinction':
        if options['occurence'] == 'first':
            limit_index = 1
        elif options['occurence'] == 'second':
            limit_index = 3
        else:
            raise Exception('Wrong value of occurence')
    else:
        raise Exception('Wrong value of limit_point')

    #### Function names
    func = bif_set_func
    jac = jacobian

    #### Initial guess
    no_of_func = 2*(no_of_species + 1)
    y0 = np.zeros(no_of_func)
    y0[1] = 1
    col_index = index[limit_index]
    if col_index == -1 or col_index == 0:
        raise Exception('Limit point does not exist at this configuration')

    x_old = T_bd_wo_noise[:, col_index]
    
    #### Rearranging the species data (Because of different species IDs)
    fluid_f_old = x_old[:no_of_species]
    wc_f_old = x_old[no_of_species : 2*no_of_species]
    invariant = x_old[2*no_of_species : ]

    fluid_f_new = _rearrange_species(species_ID_old, species_ID, fluid_f_old)
    wc_f_new = _rearrange_species(species_ID_old, species_ID, wc_f_old)
    x = np.r_[fluid_f_new, wc_f_new, invariant]
    x_init = np.r_[y0, x, bif_par_val]

    J = func(x_init, inlet_species, fixed_dict, const, thermo_object, 
            react_const, system, rate_basis, rate_units, options, 
            get_jacobian=True)
    y0_sol = root(null_space, y0, args= J, method= 'lm')
    state_var_val = np.r_[y0_sol.x, x, bif_par_val]

    if options['Testing']:
        print('\nThe eigen vectors are calculated under the following status')
        print('Status: {0}, msg: {1}'.format(y0_sol.status, y0_sol.message))
        print('\nAll the varibles going into the function')
        print(state_var_val)
        print('\nAnd the fixed variables are:')
        print(fixed_dict)
        Ysol = func(state_var_val, inlet_species, fixed_dict, const, 
                    thermo_object, react_const, system, rate_basis, 
                    rate_units, options)
        return Ysol
    else:
        #### Solving for ignition-extinction locus
        no_of_var = len(state_var_val)
        pref = np.ones(no_of_var)
        weights = np.ones(no_of_var)
#        weights[:2*(no_of_species + 1)] = 1e-03
#        weights[-4:-1] = 1e-03
#        weights[-1] = 1e-01
        jac_eps = options['jac_eps']

        T_bs = der.derpar(func, jac, state_var_val, pref, max_val, 
                          break_val, weights, jac_eps, initial=False, 
                          hh=step_change, maxout=100000, 
                          hhmax= 10*step_change, ncorr=5,
                          args=(inlet_species, fixed_dict, const, 
                          thermo_object, react_const, system, rate_basis,
                          rate_units, options), kwargs=(temp_break_val))

    #### Plotting the figure
    if bif_par_var == 'T_f_in' or bif_par_var == 'inlet_ratio':
        plot_flag = 'norm'
    else:
        plot_flag = 'log'

    fig = plt.figure()
    if plot_flag == 'log':
        plt.semilogx(T_bs[-1, :], T_bs[-2, :])
    else:
        plt.plot(T_bs[-1, :], T_bs[-2, :])
    plt.show()
    
    #### Storing data
    data_vals = [value for (key, value) in sorted(fixed_dict.items())]
    n = len(data_vals)
    filename = 'bif_set_{}_{}'.format(options['limit_point'], 
                                      options['occurence'])
    
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
        np.savez(FullfileName, T_bs[no_of_func:, :], species_ID)
        
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
    return T_bs
   

def ig_ext_point_calculator(T_bd, system, bif_par_var, break_dict, \
                            fulloutput= True, tol=1e-03):
    '''
    This function calculates the indices of ignition and extinction points 
    and returns them in form of ndarray. It takes in one array (T_f) in 
    which it searches for ignition-extinction points and the type of the 
    system, (whether the system is 'homo'-geneous or 'cat'-alytic only, or 
    thermally 'coup'-led. They keyword arguments are the bifurcation 
    parameter (bif_par) and fulloutput. If fulloutput is set to True the 
    ignition and extinction points will be printed out. 
    And finally, based on the system specified it returns the indices 
    from which the entire dataset at ignition and extinction points 
    can be calculated.
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
            elif ((T[index] > T[index + 1]) 
                    and (abs(T[index] - T[index-1]) < tol)):
                ig_point = index
                break
            else:
                index += 1
        
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
            elif ((T[index] > T[index-1])
                    and (abs(T[index-1] - T[index-2]) < tol)):
                ext_first = index
                break
            else:
                index += 1
        
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
                print('Homogeneous ignition is observed around {0} {2} and '
                        'extinction is below {1} {2}.'.format(T[T_pt[0]], \
                        T_break, unit))
            else:
                print('Homogeneous ignition is observed around {0} {2} and '
                      'extinction is observed around {1} {2}.'
                      .format(T[T_pt[0]], T[T_pt[1]], unit))
    
        elif system == 'cat':
            if T_pt[0] == 0:
                print('Oops!!! No ignition or extinction behavior is observed.')
            elif T_pt[0] != 0 and T_pt[1] == -1:
                print('Catalytic ignition is observed around {0} {2} and extinction'
                      ' is below {1} {2}.'.format(T[T_pt[0]], T_break, unit))
            else:
                print('Catalytic ignition is observed around {0} {2} and '
                      'extinction is observed around {1} {2}.'
                      .format(T[T_pt[0]], T[T_pt[1]], unit))

        else:
            #### The system is coupled
            if T_pt[0] == 0:
                print('Oops!!! No ignition or extinction behavior is observed.')
            elif T_pt[1] == -1 and T_pt[2] == 0:
                print('Only one ignition is observed around {0} {2} and the '
                        'extinction temperature is below {1} '
                        '{2}.'.format(T[T_pt[0]], T_break, unit))
            elif T_pt[1] == -1 and T_pt[2] != 0 and T_pt[3] == -1:
                print('The first ignition is observed around {0} {2} and the second'
                        ' ignition is around {1} {2}.'.format(T[T_pt[0]],\
                        T[T_pt[2]], unit))
                print('The extinction temperature in both cases are below {0} '
                        '{1}.'.format(T_break, unit))
            elif T_pt[3] == -1:
                print('The first ignition is observed around {0} {2} and the second'
                        ' ignition is around {0} {2}.'.format(T[T_pt[0]],\
                        T[T_pt[2]], unit))
                print('First extinction is around {0} {2} and the second extinction'
                        ' is below {1} {2}.'.format(T[T_pt[1]], T_break, unit))
            elif T_pt[2] == 0 and T_pt[3] == 0:
                print('The only ignition point is around {0} {1}.'.format(T[T_pt[0]], unit))
                print('And the only extinction pont is around {0} {1}.'.format(T[T_pt[1]], unit))
            else:
                print('The first ignition is observed around {0} {2} and the second'
                        ' is around {1} {2}.'.format(T[T_pt[0]], T[T_pt[2]], unit))
                print('The first extinction is observed around {0} {2} and the' 
                ' second is around {1} {2}'.format(T[T_pt[1]], T[T_pt[3]], unit))
        
    return T_pt


def left_right_eigen_vector(Y0, J, d2f, method='root'):
    '''
    This function calculates the right and left eigen vector
    corresponding to zero eigen value of J. 
    '''
    #### Checking the inputs
    N = Y0.size
    n = N//2
    y = Y0[:n]
    v = Y0[n:]
    
    #### Preparing the second-frechet derivative
    outer_pdt = np.outer(y, y)
    flattened_outer = np.ndarray.flatten(outer_pdt)
    second_frechet_deriv = np.dot(d2f, flattened_outer.T)

    #### Setting up the equations
    f1 = np.dot(J, y)
    f2 = np.dot(J.T, v)
    f3 = np.dot(y.T, v) - 1
    f4 = np.dot(second_frechet_deriv.T, v)

    f = np.r_[f1, f2, f3]
    return f


def second_derivatives(func_name, Y, jac, eps, jac_eps, *args):
    '''
    This function calculates the second order derivatives of the
    original functions using forward difference approximation.
    We can either evaluate the original functions at different points
    and then can directly calculate the second derivates or we can 
    calculate them using our jacobian function.
    So in our first attempt we try to calculate them using Jacobian.
    '''
    #### Checking the inputs
    n = Y.size

    for i in range(n-1):
        Y_pos = Y.copy()
        h = eps * Y[i]
        if h == 0:
            h = eps
        Y_pos[i] = Y[i] + h
        fvec_new = func_name(Y_pos, *args)
        jac_new = jacobian(func_name, Y_pos, fvec_new, 
                           jac_eps, n-1, n-1, 0, n-1, *args)

        deriv = (jac_new - jac)/h
        try:
            second_deriv = np.c_[second_deriv, deriv]
        except NameError:
            second_deriv = deriv[:]
        
        
    return second_deriv


def hys_locus_func(Y, inlet_species, fixed, const, thermo_object, 
                   react_const, flag, rate_basis, rate_units, options,
                   get_function=False, get_jacobian=False, 
                   get_second_deriv=False):
    '''
    This function solves for the hysteresis locus of the
    system of equations.

    Y : initial guess, ndarray consisting of the eigen vector
        and right eigen vector corresponding to zero eigen value 
        of the first Frechet derivative of the original functions, 
        the state variables and three parameters. 
        The last parameter i.e. Y[-1] is the bifurcation parameter.

    fixed : Dictionary of fixed parameters

    const : Dictionary of fixed constants

    react_const : Objects defining the reaction constants

    flag :  String, specifying the nature of the system ('cat'-alytic, 
            'homo'-geneous or 'coup'-led)

    get_function : Boolean, if True, only returns the function values
                   of the original functions

    get_jacobian : Boolean, if True, only returns the jacobian of the
                  original functions (First Frechet derivative)
    
    get_second_deriv : Boolean, if True, only returns the second 
                       derivatives of the original function.
    '''
    #### Assert statements
    assert (flag == 'cat') or (flag == 'coup') or (flag == 'homo'), ('This '
                            'program can only handle catalytic ("cat") or '
                            'coupled ("coup") system.')

    #### Unpacking the inputs

    #### Species identifier
    species = const['species']
    no_of_species = len(species)
    
    #### Unpacking the inputs
    n = Y.size

    no_of_func = 2*(no_of_species + 1)
    y0 = Y[:no_of_func]                      # Eigen vector, Ly0 = 0
    v0 = Y[no_of_func : 2*no_of_func]        # Left Eigen vector, L*v0 = 0
    X = Y[2*no_of_func:-1]                   # State variables
    bif_par = Y[-1]                          # Bifurcation parameter

    #### Identification of the fixed inputs and the bifurcation parameter
    fixed_var = ['pressure', 'tau', 'R_omega', 'R_omega_wc', 'particle_density']
    fixed_val = []
    for elem in fixed_var:
        if elem in fixed:
            fixed_val.append(fixed[elem])
        else:
            fixed_val.append(bif_par)

    fixed_dict = dict(zip(fixed_var, fixed_val))
    
    #### Obtaining the function values and jacobian at the point X
    X_bif_set = np.r_[y0, X]
    fvec, jac, second_deriv = bif_set_func(X_bif_set, inlet_species, 
                                           fixed_dict, const, thermo_object, 
                                           react_const, flag,
                                           rate_basis, rate_units, options,
                                           get_function=True, 
                                           get_jacobian=True,
                                           get_second_deriv=True)

    #### Shipping the variables requested for
    if get_jacobian and get_function and second_deriv:
        return fvec, jac, second_deriv

    elif get_jacobian and get_second_deriv:
        return jac, second_deriv

    elif get_jacobian and get_function:
        return fvec, jac

    elif get_jacobian:
        return jac
    
    elif get_function:
        return fvec

    else:
        #### Evaluating the second Frechet Derivative at (y0, y0)
        outer_pdt = np.outer(y0, y0)
        flattened_outer = np.ndarray.flatten(outer_pdt)
        second_frechet_deriv = np.dot(second_deriv, flattened_outer.T)
            
        #### Setting up the equations to calculate the hysteresis locus
        F = np.zeros(n)

        F[:no_of_func] = fvec[:]

        F[no_of_func : 2*no_of_func] = np.dot(jac, y0)

        F[2*no_of_func:3*no_of_func] = np.dot(jac.T, v0)

        F[3*no_of_func] = np.dot(second_frechet_deriv.T, v0)

        F[3*no_of_func + 1] = np.dot(y0.T, v0) - 1

        return F


def hys_locus_solver(fixed_dict, bif_par_dict, react_system, system, 
                     catalyst, rate_basis, rate_units, inlet_species,
                     break_dict, step_change, max_val, options):
    '''
    This function solves for the hysteresis locus.
    (Region of multiplicities)

    We will write other details later.

    Just to point out one important difference between hys_locus_solver
    and bif_dia_solver is that, here we take in the entire break_dict
    It is similar to bif_set_solver in that respect.
    '''
    #### Generating the constant parameters
    const, thermo_object = constants.fixed_parameters(react_system)
    species = const['species']
    no_of_species = len(species)
    ID = np.arange(no_of_species)
    species_ID = dict(zip(species, ID))
    print('This is the new species ID',species_ID)
     
    #### Manipulation of input variables 
    all_dict = dict(bif_par_dict, **fixed_dict)

    bif_par_var = list(bif_par_dict.keys())[0]  
    bif_par_val = bif_par_dict.get(bif_par_var)
    break_val = break_dict[bif_par_var]
    #y_A_in_max_val = max_dict['y_A_in']
    
    # Generating the reaction objects
    homo, cat, homo_index, cat_index = react.instantiate(react_system, 
                                       const, catalyst)
    react_const = [homo, cat, homo_index, cat_index]

    ##### First dataset
    if options['occurence']:
        filename = 'bif_set_{0}_{1}'.format(options['limit_point'],
                                                    options['occurence'])
    else:
        filename = 'bif_set_{0}'.fomat(options['limit_point'])
    npzfile = _load_file(react_system, system, catalyst, all_dict, 
                         filename=filename)
    T_bs_arr, species_ID_arr = npzfile.files
    T_bs = npzfile[T_bs_arr]
    species_ID_old = npzfile[species_ID_arr]

    index = ig_ext_point_calculator(T_bs, system, 'inlet_ratio', 
                                    break_dict, fulloutput= False,
                                    tol=1e-02)
    limit_index = 0
    #data_vals = [value for (key, value) in sorted(fixed_dict.items())]
    #n = len(data_vals)
    #
    #filename = 'hysteresis_locus_left_{}'.format(options['occurence'])
    #for i in range(n):
    #    filename += '_{}'.format(data_vals[i])
    #filename += '.npz'
    #print(filename)
    #
    #react_filename, ext = os.path.splitext(react_system)
    #
    #Target_folder = os.path.join(os.getcwd(), react_filename, 
    #                                 catalyst.lower(), 'Data', system)
    #FullfileName = os.path.join(Target_folder, filename)
    #
    #print(FullfileName)

    #npzfile = np.load(FullfileName)
    #T_hl_arr = npzfile.files
    #T_hl = npzfile[T_hl_arr[0]]
    #x = T_hl[4*(no_of_species+1):-1, -1]
    #bif_par_val = T_hl[-1, -1]

    #### Function name
    func = hys_locus_func
    jac = jacobian
    
    #### Initial guess of the left and right eigen vectors
    no_of_func = 2*(no_of_species + 1)
    y0 = np.zeros(no_of_func, dtype=float)
    v0 = np.zeros_like(y0)
    y0[1] = 1
    v0[1] = 1

    col_index = index[limit_index]
    if col_index == -1 or col_index == 0:
        raise Exception('Cusp point does not exist at this configuration')

    x_old = T_bs[:, col_index]

    #### Rearranging the species data (Because of different species IDs)
    fluid_f_old = x_old[:no_of_species]
    wc_f_old = x_old[no_of_species : 2*no_of_species]
    invariant = x_old[2*no_of_species : ]

    fluid_f_new = _rearrange_species(species_ID_old, species_ID, fluid_f_old)
    wc_f_new = _rearrange_species(species_ID_old, species_ID, wc_f_old)
    x = np.r_[fluid_f_new, wc_f_new, invariant]
    x_init = np.r_[y0, v0, x, bif_par_val]

    J, d2f = func(x_init, inlet_species, fixed_dict, const, thermo_object, 
                  react_const, system, rate_basis, rate_units, options, 
                  get_jacobian=True, get_second_deriv=True)
    Y_init = np.r_[y0, v0]

    Y0 = root(left_right_eigen_vector, Y_init, args=(J, d2f), method='lm')
    state_var_val = np.r_[Y0.x, x, bif_par_val]

    if options['Testing']:
        print('\nThe eigen vectors are calculated under the following status')
        print('Status: {0}, msg: {1}'.format(Y0.status, Y0.message))
        print('\nAll the varibles going into the function')
        print(state_var_val)
        F = func(state_var_val, inlet_species, fixed_dict, const, 
                 thermo_object, react_const, system, rate_basis, rate_units,
                 options)
        return F
    else:
        #### Solving for hysteresis locus
        no_of_var = len(state_var_val)
        pref = np.ones(no_of_var)
        no_of_func = 2*(no_of_species + 1)
        weights = np.ones(no_of_var)
        weights[:2*no_of_func] = 1e-04
        weights[-5:-2] = 1e-03
        weights[-2] = 1e-01
        jac_eps = options['jac_eps']

        T_hl = der.derpar(func, jac, state_var_val, pref, max_val, 
                          break_val, weights, jac_eps, initial=False, 
                          hh=step_change, maxout=5000, hhmax=10*step_change, 
                          ncorr=5, args=(inlet_species, fixed_dict, const, 
                          thermo_object, react_const, system, rate_basis, 
                          rate_units, options), kwargs=())

    ##### Plotting
    if bif_par_var == 'T_f_in' or bif_par_var == 'inlet_species':
        plot_flag = 'norm'
    else:
        plot_flag = 'log'

    fig = plt.figure()
    if plot_flag == 'log':
        plt.semilogx(T_hl[-1, :], 1/T_hl[-2, :])
    else:
        plt.plot(T_hl[-1, :], 1/T_hl[-2, :])
    plt.show()
    
    #### Storing data
    data_vals = [value for (key, value) in sorted(fixed_dict.items())]
    n = len(data_vals)
    if options['occurence']:
        filename = 'hysteresis_locus_left_{0}'.format(options['occurence'])
    else:
        filename = 'hysteresis_locus_left'

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
        np.savez(FullfileName, T_hl[2*no_of_func:, :], species_ID)
        
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
    return T_hl
