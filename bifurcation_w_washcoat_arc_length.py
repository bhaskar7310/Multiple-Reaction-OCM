#### This module named as 'bifurcation_w_washcoat_arc_length' has all the
#### functions and solvers to calculate all kinds of bifurcation diagrams
#### and sets of 'Short-Monolith' Reactor model in presence of finite
#### washcoat

#### Ref: Ratnakar (2019), Mingjie, Bala, Ratnakar (2019),
####      Pankaj Kumar (2012), Imran Alam (2015)

#### Author: Bhaskar Sarkar (2019)
import os
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

import constants
import reaction as react
import DERPAR_code as der
import transfer_coefficients as tc
import caley_hamilton_theorem as cht

def bif_dia_func(Y, inlet_species, fixed, const, thermo_object, 
                 react_const, flag, rate_basis, rate_units, options):
    '''
    Write this later.
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
    
    #### Options specification
    model = options['model']
    eig_val_filename = options.get('eig_val_filename', None)
    
    #### Unpacking the Y vector
    n = Y.size
    F_j = Y[:no_of_species]                         
    C_s = Y[no_of_species:2*no_of_species]
    T_s = Y[2*no_of_species]
    T_f = Y[2*no_of_species + 1]                          
    bif_par = Y[-1]

    #### Identifying the fixed inputs and the bifurcation parameter
    fixed_var = ['inlet_ratio', 'pressure', 
                 'tau', 'R_omega', 'T_f_in', 'R_omega_wc', 
                 'R_omega_w', 'particle_density']
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
    R_omega_w = fixed_dict['R_omega_w']
    particle_density = fixed_dict['particle_density']
    
    #### Printing the Temperature in eig_val_file
    if eig_val_filename:
        with open(eig_val_filename, 'a') as infile:
            infile.write(str(T_f_in) + '\n')
            

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
    D_AB = const['D_AB']
    nu = const['nu']
    eps_wc = const['eps_wc']
    tort = const['tortuosity']
    Cp_w = const['Cp_w']
    rho_w = const['rho_w']
    k_w = const['k_w']

    eps_f = 4*R_omega**2/(2*R_omega + R_omega_w)**2

    #### Calculating the bulk diffusivity of the species by 
    #### Wilke and Fairbanks Equation
    D_all = D_AB[0] * T_f ** D_AB[1] / P
    D_f = np.zeros(no_of_species)
    for i in range(no_of_species):
        D_f[i] = (1 - Y_j[i])/np.sum(np.r_[Y_j[:i], Y_j[i+1:]]/D_all[:, i])

    D_w = eps_wc/tort * D_f
    D_f_recipro = 1/D_f
    D_w_recipro = 1/D_w

    ident = np.eye(no_of_species)
    D_f_inv = D_f_recipro * ident
    D_w_inv = D_w_recipro * ident
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
    Cp_f_tilde = C_total * C_pf_hat              # Volumetric Cp, in J/m3-K

    #### Calculation of reaction rates
    homo, cat, homo_index, cat_index = react_const
    homo_basis, cat_basis = rate_basis
    homo_units, cat_units = rate_units
    
    Y_homo_react = np.r_[C_f, T_f]
    Y_cat_react = np.r_[C_s, T_s]

    homo_rate = np.zeros(len(homo_index))     
    cat_rate = np.zeros(len(cat_index))
    
    jac_homo_rate_conc = np.zeros([len(homo_index), no_of_species])
    jac_cat_rate_conc = np.zeros([len(cat_index), no_of_species])

    jac_homo_rate_T = np.zeros(len(homo_index))
    jac_cat_rate_T = np.zeros(len(cat_index))
    
    if (flag == 'cat'):
        cat_rate = cat.act_rate(Y_cat_react, species, cat_basis, P)
        
        if model is 'sh_phi':
            jac_cat_rate_conc, \
            jac_cat_rate_T = cat.reaction_jacobian(Y_cat_react, cat_rate, 
                                                    species, cat_basis, P)
    
    elif (flag == 'homo'):
        homo_rate = homo.act_rate(Y_homo_react, species, homo_basis, P)

        if model is 'sh_phi':
            jac_homo_rate_conc, \
            jac_homo_rate_T = homo.reaction_jacobian(Y_homo_react, homo_rate, 
                                                      species, homo_basis, P)
    else:
        homo_rate = homo.act_rate(Y_homo_react, species, homo_basis, P)
        cat_rate = cat.act_rate(Y_cat_react, species, cat_basis, P)

        if model is 'sh_phi':
            jac_cat_rate_conc, \
            jac_cat_rate_T = cat.reaction_jacobian(Y_cat_react, cat_rate, 
                                                    species, cat_basis, P)
            jac_homo_rate_conc, \
            jac_homo_rate_T = homo.reaction_jacobian(Y_homo_react, homo_rate, 
                                                      species, homo_basis, P)
    
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
        if model is 'sh_phi':
            jac_cat_rate_conc *= particle_density

    elif (cat_units == 'gm_sec'):
        cat_rate *= particle_density * 1000
        if model is 'sh_phi':
            jac_cat_rate_conc *= particle_density * 1000

    #### Stoichiometric coefficients matrix and Heat of reactions
    nu_homo = nu.T[homo_index]
    nu_cat = nu.T[cat_index]

    del_H_homo = -del_H_rxn[homo_index]
    del_H_cat = -del_H_rxn[cat_index]

    #### Calculation of Sherwood numbers and Nusselt numbers
    #### (based on the model)
    lambd = 0.2
    
    if model is 'sh_phi':

        #### Calculation of Thiele Modulus (Mass)
        DR_DC_homo = np.matmul(nu_homo.T, -jac_homo_rate_conc)
        DR_DC_cat = np.matmul(nu_cat.T, -jac_cat_rate_conc)
        
        Pe_trans = R_omega**2 * 1/tau * D_f_inv
        phi_sq_f = R_omega**2 * np.matmul(D_f_inv, DR_DC_homo)
        phi_sq_w = R_omega_wc**2 * np.matmul(D_w_inv, DR_DC_cat)
        
        phi_sq_f_hat = phi_sq_f + Pe_trans

        if eig_val_filename:
            with open(eig_val_filename, 'a') as infile:
                infile.write('External' + ' ')
        Sh_ext = cht.mat_func(tc.sherwood_lambda_func,
                              tc.sherwood_lambda_derivative_at_zero,
                              phi_sq_f_hat, lambd, filename=eig_val_filename)

        if eig_val_filename:
            with open(eig_val_filename, 'a') as infile:
                infile.write('Internal' + ' ')
        Sh_int = cht.mat_func(tc.sherwood_lambda_func,
                              tc.sherwood_lambda_derivative_at_zero,
                              phi_sq_w, lambd, filename=eig_val_filename)

        ##### Calculation of Thiele Modulus (Heat)
        Pe_trans_heat = R_omega**2/(alpha_f * tau)     
        phi_sq_f_heat = R_omega**2 * np.dot(del_H_homo.T, jac_homo_rate_T) \
                      / (alpha_f * Cp_f_tilde) 
        phi_sq_f_hat_heat = Pe_trans_heat + phi_sq_f_heat
        
        phi_sq_w_heat = R_omega_w**2 * np.dot(del_H_cat.T, jac_cat_rate_T)/k_w

        ##### Calculation of Nusselt numbers
        Nu_f = tc.sherwood_lambda_func(phi_sq_f_hat_heat.flatten(), lambd).real
        Nu_w = tc.sherwood_lambda_func(phi_sq_w_heat.flatten(), lambd).real

    elif model is 'sh_inf':
        #### Asymptotic values of Sherwood numbers
        Sh_ext = 3.0 * np.eye(no_of_species)
        Sh_int = 3.0 * np.eye(no_of_species)
        
        #### Asymptotic values of Nusselt numbers
        Nu_f = 3
        Nu_w = 1/3
    else:
        raise Exception('No such model available.')
        
    #### Calculation of Mass transfer coefficients
    Sh_f_inv = LA.inv(Sh_ext)
    Sh_w_inv = LA.inv(Sh_int)
    
    K_ext_inv = 4 * R_omega * np.matmul(Sh_f_inv, D_f_inv)
    K_int_inv = R_omega_wc * np.matmul(Sh_w_inv, D_w_inv)
    K_overall_inv = K_ext_inv + K_int_inv

    #### Calculation of Heat transfer coefficients
    h_ext_inv = 4 * R_omega /(Nu_f * Cp_f_tilde * alpha_f)
    h_int_inv = R_omega_w/(Nu_w * k_w)
    h_overall_inv = h_ext_inv + h_int_inv
    h = 1/h_overall_inv

    #### Calculation of Mass flux
    b = C_f - C_s
    J = np.matmul(LA.inv(K_overall_inv), b)

    #### Pre-allocation of return variable
    F = np.zeros(n)

    #### Setting up the equations
    F[:no_of_species] = (F_in - F_j)/ V \
                      + eps_f * np.dot(nu_homo.T, homo_rate) \
                      - a_v * J
    
    F[no_of_species : 2*no_of_species] = J \
                                       + R_omega_wc \
                                       * np.dot(nu_cat.T, cat_rate)
                                                           
    F[2*no_of_species] = -1/V * np.dot(F_in.T, C_p_integ) \
                       + eps_f * np.dot(del_H_homo.T, homo_rate) \
                       - h * a_v * (T_f - T_s)
#    F[2*no_of_species] = np.dot(F_in.T, C_p_avg) / V * (T_f_in - T_f) \
#                       + eps_f * np.dot(del_H_homo.T, homo_rate) \
#                       - h_hat * a_v * (T_f - T_s)
    
    F[2*no_of_species + 1] = h * (T_f - T_s) \
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
    We will write this thing later
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
    T_f_in = all_dict['T_f_in']
    inlet_ratio = all_dict['inlet_ratio']
    
    N2 = 0  # For the time being, we use pure O2
    F_A_in = inlet_ratio/(N2 + inlet_ratio + 1)
    F_B_in = 1/(N2 + inlet_ratio + 1)
    
    F_in = 1e-08 * np.ones(no_of_species)
     
    A_index = species_ID[inlet_species]
    B_index = species_ID['O2']

    F_in[A_index] = F_A_in
    F_in[B_index] = F_B_in

    state_var_val = np.r_[F_in, F_in, T_f_in, 
                          T_f_in, bif_par_val]

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
               thermo_object, react_const, system,  rate_basis, rate_units,
               options)
        return Ysol

    else:
        #### Solving the set of equations using Arc-Length method in Python
        no_of_var = len(state_var_val)
        pref = np.ones(no_of_var)
        weights = np.ones(no_of_var)
        weights[-2:] = 1e-03
        jac_eps = options['jac_eps']
        max_iter = options['max_iter']

        data_vals = [value for (key, value) in sorted(fixed_dict.items())]
        n = len(data_vals)
        react_filename, ext = os.path.splitext(react_system)
        
        if options['model'] is 'sh_phi' and options['write_eig_val']:
            #### Filename
            file = react_filename + '_' + catalyst.lower() + '_' + system
            for i in range(n):
                file += '_{}'.format(data_vals[i])
            file +='.txt'
            
            #### Directory
            folder = os.path.join(os.getcwd(), 'EigenValues')
            if not os.path.isdir(folder):
                os.mkdir('EigenValues')
            fullfilename = os.path.join(folder, file)

            print(fullfilename)
            #### Creating the file
            with open(fullfilename, "w") as fh:
                fh.write(react_system)
                fh.write('\n')
            options.update({'eig_val_filename': fullfilename})
            
        T_bd = der.derpar(func, jac, state_var_val, pref, max_val, break_val,
               weights, jac_eps, eps=1e-04, initial= False, maxIter=50, 
               hh = step_change, maxout=max_iter, 
               hhmax = 10*step_change, ncorr=6,
               args=(inlet_species, fixed_dict, const, 
               thermo_object, react_const, system, rate_basis, rate_units,
               options)) 

    #### Plotting the figure
    fig = plt.figure()
    if plot_flag == 'log':
        plt.semilogx(T_bd[-1, :], T_bd[-2, :])
    else:
        plt.plot(T_bd[-1, :], T_bd[-2, :])
    plt.show()
    
    #### Storing data
    filename = 'bif_dia_{}'.format(options['model'].lower())
    
    for i in range(n):
        filename += '_{}'.format(data_vals[i])
        
    print(filename)
    
    if options['save']:
        #### Saving Data
        Target_folder = os.path.join(os.getcwd(), 'Washcoat', react_filename, 
                                     catalyst.lower(), 'Data', system)
    
        
        if os.path.isdir(Target_folder) == False:
            New_Folder = 'Washcoat/' + react_filename + '/' + catalyst.lower() + '/' \
                       + '/Data/' + system
            os.makedirs(New_Folder)
    
        FullfileName = os.path.join(Target_folder, filename)
        np.savez(FullfileName, T_bd, F_in, species_ID)
        
        #### Saving Diagrams (Why I don't know)
        Target_folder = os.path.join(os.getcwd(), 'Washcoat', react_filename, 
                                     catalyst.lower(), 'Diagram', system)

        if os.path.isdir(Target_folder) == False:
            New_Folder = 'Washcoat/' + react_filename + '/' + catalyst.lower() + '/' \
                       + '/Diagram/' + system
            os.makedirs(New_Folder)
    
        dia_filename = filename + '.png'
        FullfileName = os.path.join(Target_folder, dia_filename)
        fig.savefig(FullfileName)
    
    #### Returning the final result
    return T_bd

