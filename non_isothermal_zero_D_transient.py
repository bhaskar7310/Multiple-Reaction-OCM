#### This module named as 'non_isothermal_zero_dimensional' has all the 
#### functions and solvers to integrate the conservation equations of 
#### 0D reactor models.

#### Ref: Pankaj Kumar

#### Author: Bhaskar Sarkar (2019)
import time
import cmath
import numpy as np
import numpy.linalg as LA
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ourstyle')

import constants
import reaction as react
import caley_hamilton_theorem as cht

def zero_D_non_iso_model(Y, t, inlet_species, fixed_dict, const,
                         thermo_object, react_const, flag, rate_basis, 
                         rate_units, model):
    '''
    This function describes the ODEs required to solve for the
    species and energy balance equations in fluid phase and 
    washcoat at non-isothermal conditions using the OD model. 
    The interfacial flux are calculated using aysmptotic Sherwood 
    and Nusselt numbers.
    
    Y : Vector of concentrations and temperature, 
        in fluid and washcoat, concentration in mol/m3, T in K
    t  : time, in s
    inlet_species : the name of the fuel species, e.g. 'CH4'
    fixed_dict    : dictionary of the fixed parameters
    const         : Constant parameters, from constant module
    thermo_object : Themodynamic parameters
    react_const   : reaction parameters
    flag          : Whether the system is 'homo'-geneous, 'cat'-alytic,
                    or 'coup'-led
    rate_basis    : the basis of rate expressions, whether pressure,
                    concentration or mole fraction
    rate_units    : units of the rate expressions
    model         : Name of the model, whether sh_inf, or sh_phi

    This function returns a vector, which can be integrated to get the
    species concentration and Temperature in fluid and washcoat over 
    time.
    '''
    #### Assert statements
    assert (flag == 'cat') or (flag == 'coup') or (flag == 'homo'), ('This '
                            'program can only handle catalytic ("cat") or '
                            'coupled ("coup") system.')
    
    print(t)
    #### Species identification
    species = const['species']
    no_of_species = len(species)
    ID = np.arange(no_of_species)
    species_ID = dict(zip(species, ID))

    #### Unpacking the Y vector
    N = Y.size
    n = N//2
    if n != no_of_species + 1:
        raise Exception('There is discrepancy in the no. of species and the '
                        'dimension of the C0 vector')

    C_f = Y[:no_of_species]
    C_w = Y[no_of_species : 2*no_of_species] 
    T_f = Y[-2]    
    T_s = Y[-1]

    #### Unpacking the fixed inputs
    P = fixed_dict['pressure']
    u0 = fixed_dict['velocity']
    L = fixed_dict['length']
    R_omega = fixed_dict['R_omega']
    R_omega_wc = fixed_dict['R_omega_wc']
    R_omega_w = fixed_dict['R_omega_w']
    inlet_ratio = fixed_dict['inlet_ratio']

    #### Unpacking the constants
    nu = const['nu']
    eps_wc = const['eps_wc']
    tort = const['tortuosity']
    D_AB = const['D_AB']
    alpha_f_const = const['alpha_f']

    #### Calculating Inlet concentrations and Temperature
    T_f_in = temp_ramp_func(t)
    C_in_total = 101325 * P/(8.314 * T_f_in)
    C_f_in = 1e-03 * np.ones(no_of_species)

    y_A_in = inlet_ratio/(inlet_ratio + 1)
    y_B_in = 1/(inlet_ratio + 1)
    c_A_in = y_A_in * C_in_total
    c_B_in = y_B_in * C_in_total
    
    A_index = species_ID[inlet_species]
    B_index = species_ID['O2']
    
    C_f_in[A_index] = c_A_in
    C_f_in[B_index] = c_B_in 

    #### This following physical properties are taken from Dadi
    rho_w = 2000            # Kg/m3
    Cp_w = 1000             # J/Kg-K
    k_w = 1.5               # W/m-k

    #### Calculation of Diffusivities (Bulk and washcoat)
    conc_sum = np.sum(C_f)
    Y_f = C_f/conc_sum
    D_all = D_AB[0] * T_f ** D_AB[1] / P
    D_f = np.zeros(no_of_species)
 
    for i in range(no_of_species):
        D_f[i] = (1 - Y_f[i])/np.sum(np.r_[Y_f[:i], Y_f[i+1:]]/D_all[:, i])
    
    D_w = eps_wc/tort * D_f
    D_f_recipro = 1/D_f
    D_w_recipro = 1/D_w
    
    ident = np.eye(no_of_species)
    D_f_inv = D_f_recipro * ident
    D_w_inv = D_w_recipro * ident

    alpha_f = alpha_f_const * T_f ** 1.75 / P

    #### Calculation of Cp values
    Cp_species_T = thermo_object.Cp_at_T(T_f)
    Cp_f = np.dot(Cp_species_T.T, Y_f)
    Cp_tilde = Cp_f * conc_sum
    k_f = alpha_f * Cp_tilde

    #### Calculation of reaction rates and its derivatives w.r.t C (Jacobian)
    homo, cat, homo_index, cat_index = react_const
    homo_basis, cat_basis = rate_basis
    homo_units, cat_units = rate_units

    if (flag == 'cat'):
        homo_rate = np.zeros(len(homo_index))
        cat_rate = cat.act_rate(C_w, species, cat_basis, T_s, P)

        if model is 'Sh_phi':
            jac_homo_rate = np.zeros([len(homo_index), n])
            jac_cat_rate = -jacobian(cat.act_rate, C_w, cat_rate, species, 
                                    cat_basis, T_s, P)
    elif (flag == 'homo'):
        cat_rate = np.zeros(len(cat_index))
        homo_rate = homo.act_rate(C_f, species, homo_basis, T_f, P)
        
        if model is 'Sh_phi':
            jac_cat_rate = np.zeros([len(cat_index), n])
            jac_homo_rate = -jacobian(homo.act_rate, C_f, homo_rate, species, 
                                     homo_basis, T_f, P)
    else:
        cat_rate = cat.act_rate(C_w, species, cat_basis, T_s, P)
        homo_rate = homo.act_rate(C_f, species, homo_basis, T_f, P)
        
        if model is 'Sh_phi':
            jac_cat_rate = -jacobian(cat.act_rate, C_w, cat_rate, species, 
                                    cat_basis, T_s, P)
            jac_homo_rate = -jacobian(homo.act_rate, C_f, homo_rate, species, 
                                     homo_basis, T_f, P)
    #### Unit correction
    homo_units, cat_units = rate_units

    if homo_basis == 'mole_fraction':
        if homo_units != 'second':
            raise Exception ('There is a discrepancy '
                             'in the homogeneous reaction rate')

    if (cat_units == 'kg_sec'):
        cat_rate *= particle_density
        if model is 'Sh_phi':
            jac_cat_rate *= particle_density

    elif (cat_units == 'gm_sec'):
        cat_rate *= particle_density * 1000
        if model is 'Sh_phi':
            jac_cat_rate *= particle_density * 1000

    #### Stoichiometric coefficients and reaction rate matrices
    nu_homo = nu.T[homo_index]
    nu_cat = nu.T[cat_index]
    
    rate_homo = np.dot(nu_homo.T, homo_rate)
    rate_cat = np.dot(nu_cat.T, cat_rate)

    #### Calculating the heat of reactions
    del_H_rxn = thermo_object.del_H_reaction(T_f)
    del_H_homo = -del_H_rxn[homo_index]
    del_H_cat = -del_H_rxn[cat_index]

    source_homo = 1/(Cp_tilde) * np.dot(del_H_homo.T, homo_rate).flatten()
    source_cat = 1/(rho_w * Cp_w) * np.dot(del_H_cat.T, cat_rate).flatten()
    
    #### Heat transfer and generation times
    heat_diff_time_f = R_omega** 2/alpha_f
    heat_diff_time_w = R_omega_w**2 * rho_w * Cp_w/k_w

    heat_gen_time_homo = T_f/source_homo
    heat_gen_time_cat = T_s/source_cat

    #### Calculation of Sherwood numbers based on the model
#    lambd=0.10210414

    if model is 'Sh_phi':

        #### Calculation of Thiele Modulus (Mass)
        DR_DC_homo = np.matmul(nu_homo.T, jac_homo_rate)
        DR_DC_cat = np.matmul(nu_cat.T, jac_cat_rate)
        
        Pe_trans = R_omega**2 * u0/L * D_f_inv
        phi_sq_f = R_omega**2 * np.matmul(D_f_inv, DR_DC_homo)
        phi_sq_w = R_omega_wc**2 * np.matmul(D_w_inv, DR_DC_cat)

        phi_sq_f_hat = phi_sq_f + Pe_trans

#        #### Calculation of Thiele Modulus (Heat)
#        Pe_trans_heat = u0/L * heat_diff_time_f
#        phi_sq_heat_f = heat_diff_time_f/heat_gen_time_homo
#        phi_sq_heat_w = heat_diff_time_w/heat_gen_time_cat
#
#        phi_sq_heat_f_hat = phi_sq_heat_f + Pe_trans_heat
#        
#        #### Calculation of Nusselt numbers
#        Nu_f_inv = sherwood_inv_func(phi_sq_heat_f_hat).real
#        Nu_w_inv = sherwood_inv_func(phi_sq_heat_w).real

        #### Calculation of Sherwood numbers
        #Sh_w_at_x = 3.0*np.eye(no_of_species) \
        #          + cht.mat_func(sherwood_lambda_func, 
        #                         sherwood_lambda_derivative_at_zero,
        #                         phi_sq_w, lambd)
        #Sh_w_inv = LA.inv(Sh_w_at_x)

        Sh_w_inv = cht.mat_func(sherwood_inv_func,
                                sherwood_inv_derivative_at_zero,
                                phi_sq_w)

        #Sh_f_at_x = 3.0*np.eye(no_of_species) \
        #          + cht.mat_func(sherwood_lambda_func, 
        #                         sherwood_lambda_derivative_at_zero,
        #                         phi_sq_f_hat, lambd)
        #Sh_f_inv = LA.inv(Sh_f_at_x)
        Sh_f_inv = cht.mat_func(sherwood_inv_func,
                                sherwood_inv_derivative_at_zero,
                                phi_sq_f_hat)

    elif model is 'Sh_inf':
        #### Asymptotic values of Nusselt numbers
        Nu_f_inv = 1/3
#        Nu_w_inv = 1/3

        #### Asymptotic values of Sherwood numbers
        Sh_w_at_x = 3.0 * np.eye(no_of_species)
        Sh_w_inv = LA.inv(Sh_w_at_x)

        Sh_f_at_x = 3.0 * np.eye(no_of_species)
        Sh_f_inv = LA.inv(Sh_f_at_x)
    else:
        raise Exception('No such model available.')
    
    #### Calculation of Mass Transfer coefficients
    K_int_inv = R_omega_wc * np.matmul(Sh_w_inv, D_w_inv)
    K_ext_inv = 4 * R_omega * np.matmul(Sh_f_inv, D_f_inv)
    K_overall_inv = K_int_inv + K_ext_inv

    #### Calculation of Heat Transfer coefficients
    Nu_f_inv = 1/3
    h_ext_inv = 4 * R_omega/k_f * Nu_f_inv
#    h_int_inv = R_omega_w/k_w * Nu_w_inv
    h_overall_inv = h_ext_inv #+ h_int_inv

    #### Calculation of interfacial flux

    #### Mass flux
    b = C_f - C_w
    J = np.zeros_like(C_f)

#   J_dummy[:, i] = LA.solve(K_overall_inv, b) 
    J = np.matmul(LA.inv(K_overall_inv), b)
    
    #### Heat Flux
    delta_T = T_f - T_s
    q0 = 1/h_overall_inv * delta_T

    #### Setting up the equations

    # Fluid Phase species balance
    bulk_f = -u0 * (C_f - C_f_in)/L + rate_homo - 1/R_omega * J

    # Washcoat species balance
    wc_f = 1/eps_wc * (rate_cat + 1/R_omega_wc *  J)

    # Fluid phase energy balance
    energy_f = -u0 * (T_f - T_f_in)/L + source_homo \
             - 1/(R_omega * Cp_tilde) * q0

    # Washcoat energy balance
    energy_wc = source_cat * R_omega_wc/R_omega_w \
              + 1/(R_omega_w * rho_w * Cp_w) * q0

    #### Packaging and shipping
    F = np.r_[bulk_f, wc_f, energy_f, energy_wc]

    return F


def jacobian(func_name, Y, fvec, *args):
    '''
    This function calculates the Jacobian of the function given by 'func_name',
    using forward difference approximation.
    '''
    #### Checking the inputs
    no_of_species = Y.shape[0]
    no_of_reaction = fvec.shape[0]

    #### Specifying variables
    column = 0.
    eps = 1e-04
    J = np.zeros([no_of_reaction, no_of_species], dtype=float)
    Y_pos = Y.copy()
    
    for i in range(no_of_species):
        h = eps * Y[i]
        if h == 0:
            h = eps
        Y_pos[i] = Y[i] + h
            
        column = (func_name(Y_pos, *args) - fvec)/h
        J[:, i] = column[:]
        Y_pos[i] = Y_pos[i] - h
        
    return J


def sherwood_inv_func(x):
    '''
    x : an input vector
    f(x) = 1/(np.sqrt(x) * np.tanh(x)) - 1/(np.sqrt(x))**2
    Now this will give error if x goes to 0.
    '''
    #### Checking the inputs
    try:
        n = len(x)
    except TypeError:
        x = np.array([x])
        n = 1
    f = np.zeros(n, dtype=complex)
    
    #### Evaluating the function
    for i in range(n):
        if abs(x[i]) < 1e-04:
            f[i] = 1/3 + 0j
        else:
            f[i] = 1/(cmath.sqrt(x[i]) * cmath.tanh(cmath.sqrt(x[i]))) \
                 - 1/x[i] 

    return f


def sherwood_inv_derivative_at_zero(k):
    '''
    Returns analytically solved derivatives of 
    sherwood_inv_func at x = 0.
    '''
    if k == 0:
        return 1/3
    elif k == 1:
        return -1/45
    elif k == 2:
        return 4/945
    elif k == 3:
        return -2/1575
    elif k == 4:
        return 16/31185
    elif k == 5:
        return -11056/42567525
    elif k == 6:
        return 64/405405
    elif k == 7:
        return 57872/516891375
    else:
        raise Exception ('Higher order derivatives at with limit x-> 0 is not '
                         'provided, requested k = {}.'.format(k))
    

def sherwood_lambda_func(x, alpha):
    '''
    x : a vector
    f(x) = np.sqrt(x) * np.tanh(alpha*np.sqrt(x))
    '''
    #### Checking the inputs
    try:
        n = len(x)
    except TypeError:
        x = np.array([x])
        n = 1
    f = np.zeros(n, dtype=complex)

    #### Evaluating the function
    for i in range(n):
        f[i] = cmath.sqrt(x[i]) * cmath.tanh(alpha*cmath.sqrt(x[i])) 

    return f
 

def sherwood_lambda_derivative_at_zero(k, alpha):
    '''
    Returns analytically solved derivatives of 
    sherwood_lambda_func at x = 0.
    '''
    if k == 0:
        return 0
    elif k == 1:
        return alpha
    elif k == 2:
        return -2*alpha**3/3
    elif k == 3:
        return 4*alpha**5/5
    elif k == 4:
        return -136*alpha**7/105
    elif k == 5:
        return 496*alpha**9/189
    elif k == 6:
        return -22112 * alpha**11/3465
    elif k == 7:
        return 349504 * alpha**13/19305
    else:
        raise Exception ('Higher order derivatives at with limit x-> 0 is not '
                         'provided, requested k = {}.'.format(k))


def low_D_non_iso_model_solver(fixed_dict, react_system, system, catalyst,
                               rate_basis, rate_units, inlet_species, 
                               options):

    '''
    This function solves the low_D_Sh_phi function to generate the 
    conversion vs Temperature curve.

    fixed_dict : Dictionary of fixed variables and values.
    react_system : Name of the reaction system
    system : Whether the system is 'cat'-alytic, 'coup'-led, or 
             'homo'-geneous
    catalyst : Name of the catalyst used
    rate_basis : Basis of the rate expressions, whether pressure,
                 concentration, mole fraction
    rate_units : Units of the rate expressions
    inlet_species : Name of the fuel, e.g. 'CH4'
    options : Some options, like whether to save the result, or just 
              to test the model.
    '''

    #### Constant parameters involved with the system
    const, thermo_object = constants.fixed_parameters(react_system)
    species = const['species']
    no_of_species = len(species)
    ID = np.arange(no_of_species)
    species_ID = dict(zip(species, ID))
    print(species_ID)
    
    #### Generating the reaction objects
    homo, cat, homo_index, cat_index = react.instantiate(react_system,
                                       const, catalyst)
    react_const = [homo, cat, homo_index, cat_index]
    
    inlet_ratio = fixed_dict['inlet_ratio']
    y_A_in = inlet_ratio/(inlet_ratio + 1)
    y_B_in = 1/(inlet_ratio + 1)
    T_f_in = 500

    #### Function names
    func = zero_D_non_iso_model
    tspan = np.linspace(0, 3000, 10000)

    #### Inlet values
    c_A_in = y_A_in * 101325 * fixed_dict['pressure']/(8.314 * T_f_in)
    c_B_in = y_B_in * 101325 * fixed_dict['pressure']/(8.314 * T_f_in)
    
    C_f_in = 1e-03 * np.ones(no_of_species)
    A_index = species_ID[inlet_species]
    B_index = species_ID['O2']
    C_f_in[A_index] = c_A_in
    C_f_in[B_index] = c_B_in
    
    C_0 = np.r_[C_f_in, C_f_in, T_f_in, T_f_in]
    
    #### Testing
    if options['Testing']:

        Y = func(C_0, 4200, inlet_species, fixed_dict, const,
                 thermo_object, react_const, 
                 system, rate_basis, 
                 rate_units, options['model'])
        return Y
    else:
        #### Integration
        Y = odeint(func, C_0, tspan, args=(inlet_species, fixed_dict, const,
                                           thermo_object, react_const, 
                                           system, rate_basis, rate_units, 
                                           options['model']))
        
        ### Conversion calculations
        T_f_in = np.array(list(map(temp_ramp_func, tspan)))
        C_in = 101325 * fixed_dict['pressure']/(8.314*T_f_in) 

        CH4_in = y_A_in * C_in
        O2_in = y_B_in * C_in
        
        C_f_exit = Y[:, :no_of_species]
        CH4_exit = C_f_exit[:, A_index]
        O2_exit = C_f_exit[:, B_index]
    
#        conv_A = (1 - CH4_exit/CH4_in)
        conv_B = (1 - O2_exit/O2_in)
        
        T_f_exit = Y[:, no_of_species*2]
        
        #### Plotting
        fig, ax = plt.subplots()
        ax.plot(T_f_in, conv_B)
        ax.set_xlabel('Inlet Fluid Temperature (K)')
        ax.set_ylabel(r'Conversion of $\mathbf{O_2}$')
        ax.set_ylim([0, 1])
        
        fig, ax1 = plt.subplots()
        ax1.plot(T_f_in, T_f_exit)
        ax1.set_xlabel('Inlet fluid Temperature (K)')
        ax1.set_ylabel('Exit fluid Temperature (K)')
        
        plt.show()
        
        return Y
        

def temp_ramp_func(t):
    if t <= 4200:
        return 500 + t/6
    else:
        return 1900 - t/6

if __name__ == '__main__':
    
    #### Time starts NOW
    t0 = time.time() 
    
    #### Reaction system Identification
#    react_sys = 'OCM_two_reaction.txt'
    react_sys = 'OCM_eleven_reaction.txt'
#    catalyst = 'la_ce'
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
    pressure = 1
    velocity = 6
    length = 6.25e-02
    R_omega = 0.262e-03                  
    R_omega_wc = 50e-06
    R_omega_w = 142.5e-06
    particle_density = 3600

    #### Fixed inputs
    fixed_var = ['inlet_ratio', 'pressure', 'velocity',
                 'length', 'R_omega', 'R_omega_wc', 
                 'R_omega_w', 'particle_density']
    fixed_val = [inlet_ratio, pressure, velocity, 
                 length, R_omega, R_omega_wc, 
                 R_omega, particle_density]
    fixed_dict = dict(zip(fixed_var, fixed_val))
    
    #### Solver Options
    Testing = False
    save = True
    model = 'Sh_phi'

    options = {'Testing' : Testing,
               'save' : save,
               'model': model}

    #### Solving the ODEs
    Y = low_D_non_iso_model_solver(fixed_dict, react_sys, system, catalyst,
                                   rate_basis, rate_units, inlet_species,
                                   options)
    
    t1 = time.time()
    print('Time taken: ',(t1 - t0))


