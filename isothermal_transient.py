#### This module named as 'dynamic_evolution' has all the functions and solvers to 
#### integrate the conservation equations of different types of reactor models
#### (1+1D, Low_D_Sh_asymptotic, Low_D_Sh_Phi) of monolith reactors

#### Author: Bhaskar Sarkar (2019)
#import os
import time
import cmath
import numpy as np
import numpy.linalg as LA
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ourstyle')
import warnings
warnings.filterwarnings("error")


import constants
import reaction as react
import caley_hamilton_theorem as cht

def low_D_Sh_phi(C0, t, inlet_species, fixed_dict, C_f_in, T_f_in, const,
                 thermo_object, react_const, flag, rate_basis, rate_units,
                 model):
    '''
    This function describes the ODEs required to solve for the
    species balance equations in fluid phase and 
    washcoat at isothermal conditions. The interfacial flux are
    calculated using position dependent Sherwood numbers.
    
    C0 : Vector of concentrations, in fluid and washcoat, in mol/m3
    t  : time, in s
    inlet_species : the name of the fuel species, e.g. 'CH4'
    fixed_dict    : dictionary of the fixed parameters
    C_f_in        : Inlet concentration vector, in mol/m3
    T_f_in        : Inlet temperature, in K
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
    species concentration in fluid and washcoat over time.
    '''

    #### Assert statements
    assert (flag == 'cat') or (flag == 'coup') or (flag == 'homo'), ('This '
                            'program can only handle catalytic ("cat") or '
                            'coupled ("coup") system.')
    
    #### Species identification
    species = const['species']
    no_of_species = len(species)
    
    #### Unpacking the C0 vector
    N = C0.size
    n = N//(2*no_of_species)     # No. of points in X

    C_f =  C0[:n*no_of_species].reshape((no_of_species, n), order='F')
    C_w = C0[n*no_of_species:].reshape((no_of_species, n), order='F')

#    C_f[:, 0] = C_f_in
    #### Unpacking the fixed inputs
    P = fixed_dict['pressure']
    u0 = fixed_dict['velocity']
    L = fixed_dict['length']
    R_omega = fixed_dict['R_omega']
    R_omega_wc = fixed_dict['R_omega_wc']

    hx = L/(n-1)                       
    x = np.linspace(0, L, n)
    x[0] = 1e-05
    x_inv_wo_shape = 1/x
    x_inv = x_inv_wo_shape.reshape(n, 1, 1)
    
    #### Unpacking the constants
    nu = const['nu']
    eps_wc = const['eps_wc']
    tort = const['tortuosity']
    D_AB = const['D_AB']

    #### Calculation of Diffusivities (Bulk and Washcoat)
    conc_sum = np.sum(C_f, axis=0)
    Y_f = C_f/conc_sum
    D_all = D_AB[0] * T_f_in ** D_AB[1] / P
    D_f = np.zeros([no_of_species, n])

    for i in range(no_of_species):
        D_f[i, :] = (1 - Y_f[i, :]) \
                  / np.sum(np.r_[Y_f[:i, :], Y_f[i+1:, :]]/D_all[:, [i]], 
                           axis=0)
    
    D_w = eps_wc/tort * D_f
    D_f_recipro = 1/D_f
    D_w_recipro = 1/D_w

    ident = np.eye(no_of_species)
    D_f_inv = D_f_recipro.T.reshape(n, no_of_species, 1) * ident
    D_w_inv = D_w_recipro.T.reshape(n, no_of_species, 1) * ident
    X_inv = x_inv * ident
  
    #### Calculation of reaction rates and its derivative w.r.t C (Jacobian)
    homo, cat, homo_index, cat_index = react_const
    homo_basis, cat_basis = rate_basis
    homo_units, cat_units = rate_units

    if (flag == 'cat'):
        homo_rate = np.zeros([len(homo_index), n])    
        cat_rate = cat.act_rate(C_w, species, cat_basis, T_f_in, P)
        
        if model is 'Sh_phi':
            jac_homo_rate = np.zeros([n, len(homo_index), no_of_species])
            jac_cat_rate = jacobian(cat.act_rate, C_w, cat_rate, species, 
                                    cat_basis, T_f_in, P)

    elif (flag == 'homo'):
        cat_rate = np.zeros([len(cat_index), n])
        homo_rate = homo.act_rate(C_f, species, homo_basis, T_f_in, P)
        
        if model is 'Sh_phi':
            jac_cat_rate = np.zeros([n, len(cat_index), no_of_species])
            jac_homo_rate = jacobian(homo.act_rate, C_f, homo_rate, species, 
                                     homo_basis, T_f_in, P)
    else:
       

        cat_rate = cat.act_rate(C_w, species, cat_basis, T_f_in, P)
        homo_rate = homo.act_rate(C_f, species, homo_basis, T_f_in, P)
        
        if model is 'Sh_phi':
            jac_cat_rate = -jacobian(cat.act_rate, C_w, cat_rate, species, 
                                    cat_basis, T_f_in, P)
            jac_homo_rate = -jacobian(homo.act_rate, C_f, homo_rate, species, 
                                     homo_basis, T_f_in, P)
    
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
    
    #### Calculating Sherwood numbers based on the model
    Sh_w_inv = np.zeros([n, no_of_species, no_of_species], dtype=float)
    Sh_f_inv = np.zeros([n, no_of_species, no_of_species], dtype=float)
#    lambd=0.10210414
    
    if model is 'Sh_phi':

        #### Calculation of Thiele Modulus
        DR_DC_homo = np.matmul(nu_homo.T, jac_homo_rate)
        DR_DC_cat = np.matmul(nu_cat.T, jac_cat_rate)
        
        Pe_trans = R_omega**2 * u0 * np.matmul(D_f_inv, X_inv)
        phi_sq_f = R_omega**2 * np.matmul(D_f_inv, DR_DC_homo)
        phi_sq_w = R_omega_wc**2 * np.matmul(D_w_inv, DR_DC_cat)

        phi_sq_f_hat = phi_sq_f + Pe_trans

        #### Calculation of Sherwood numbers
        for i in range(n):
#            Sh_w_at_x = 3.0*np.eye(no_of_species) \
#                      + cht.mat_func(sherwood_lambda_func, 
#                                     sherwood_lambda_derivative_at_zero,
#                                     phi_sq_w[i, :, :], lambd)
#            Sh_w_inv[i, :, :] = LA.inv(Sh_w_at_x)
            Sh_w_inv[i, :, :] = cht.mat_func(sherwood_inv_func,
                                             sherwood_inv_derivative_at_zero,
                                             phi_sq_w[i, :, :])
            
#
#            Sh_f_at_x = 3.0*np.eye(no_of_species) \
#                      + cht.mat_func(sherwood_lambda_func, 
#                                     sherwood_lambda_derivative_at_zero,
#                                     phi_sq_f_hat[i, :, :], lambd)
#            Sh_f_inv[i, :, :] = LA.inv(Sh_f_at_x)
            Sh_f_inv[i, :, :] = cht.mat_func(sherwood_inv_func,
                                             sherwood_inv_derivative_at_zero,
                                             phi_sq_f_hat[i, :, :])
            
    elif model is 'Sh_inf':
        for i in range(n):
            Sh_w_at_x = 3.0 * np.eye(no_of_species)
            Sh_w_inv[i, :, :] = LA.inv(Sh_w_at_x)

            Sh_f_at_x = 3.0 * np.eye(no_of_species)
            Sh_f_inv[i, :, :] = LA.inv(Sh_f_at_x)
    else:
        raise Exception('No such model available.')
    
    #### Calculation of Mass Transfer coefficients
    K_int_inv = R_omega_wc * np.matmul(Sh_w_inv, D_w_inv)
    K_ext_inv = R_omega * np.matmul(Sh_f_inv, D_f_inv)
    K_overall_inv = K_int_inv + K_ext_inv
     
    #### Calculation of interfacial flux
    b = C_f - C_w
    J = np.zeros_like(C_f)

    for i in range(n):
#        J_dummy[:, i] = LA.solve(K_overall_inv[i, :, :], b[:, i]) 
        J[:, i] = np.matmul(LA.inv(K_overall_inv[i, :, :]), b[:, i])

    #### Setting up the equations

    # Fluid Phase species balance
    bulk_f = np.zeros_like(C_f)

    bulk_f[:, 0] = -u0**2 * (C_f[:, 0] - C_f_in)/D_f[:, 0] + rate_homo[:, 0] \
                - 1/R_omega * J[:, 0]
                
    bulk_f[:, 1:] = -u0 * (C_f[:, 1:] - C_f[:, 0:-1])/hx + rate_homo[:, 1:] \
               - 1/R_omega * J[:, 1:]

    # Washcoat species balance
    wc_f = 1/eps_wc * (rate_cat + 1/R_omega_wc *  J)
    
    #### Packaging and shipping
    F = np.r_[bulk_f.reshape(no_of_species*n, order='F'), 
              wc_f.reshape(no_of_species*n, order='F')]

    return F
    

def jacobian(func_name, Y, fvec, *args):
    '''
    This function calculates the Jacobian of the function given by 'func_name',
    using forward difference approximation.
    '''
    #### Checking the inputs
    no_of_species, n1 = Y.shape
    no_of_reaction, n2 = fvec.shape
    assert (n1 == n2), ('There is some discrepancy between the dimensions of '
                        'concentration matrix and reaction rate matrix.')

    #### Specifying variables
    n = n1
    column = 0.
    eps = 1e-04
    J = np.zeros([n, no_of_reaction, no_of_species], dtype=float)
    Y_pos = Y.copy()
    
    for i in range(no_of_species):
        hh = eps * Y[i, :]
        h = np.where(hh==0, eps, hh)  
        Y_pos[i, :] = Y[i, :] + h
            
        column = (func_name(Y_pos, *args) - fvec)/h
        column_3d = column.T.reshape(n, no_of_reaction, 1)
        J[:, :, i] = column_3d[:, :, 0]
        Y_pos[i, :] = Y_pos[i, :] - h
        
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


def low_D_Sh_phi_solver(fixed_dict, T_f_in_span, tspan,
                        react_system, system, catalyst,
                        rate_basis, rate_units, inlet_species, n, nT, options):

    '''
    This function solves the low_D_Sh_phi function to generate the 
    conversion vs Temperature curve.

    fixed_dict : Dictionary of fixed variables and values.
    T_f_in_span : Span of inlet temperature
    react_system : Name of the reaction system
    system : Whether the system is 'cat'-alytic, 'coup'-led, or 
             'homo'-geneous
    catalyst : Name of the catalyst used
    rate_basis : Basis of the rate expressions, whether pressure,
                 concentration, mole fraction
    rate_units : Units of the rate expressions
    inlet_species : Name of the fuel, e.g. 'CH4'
    n : No of discretized points in x
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
    
    #### Function names
    func = low_D_Sh_phi
    
    T_f_in = np.linspace(T_f_in_span[0], T_f_in_span[1], nT)
    conv_A = np.zeros_like(T_f_in)
    conv_B = np.zeros_like(T_f_in)

    if options['model'] is 'both':
        conv_A_inf = np.zeros_like(T_f_in)
        conv_B_inf = np.zeros_like(T_f_in)

    for i in range(len(T_f_in)):
        
        #### Inlet values
        c_A_in = y_A_in * 101325 * fixed_dict['pressure']/(8.314 * T_f_in[i])
        c_B_in = y_B_in * 101325 * fixed_dict['pressure']/(8.314 * T_f_in[i])
    
        C_f_in = 1e-03 * np.ones(no_of_species)
        A_index = species_ID[inlet_species]
        B_index = species_ID['O2']
        C_f_in[A_index] = c_A_in
        C_f_in[B_index] = c_B_in
        
        C_f_0 = np.tile(C_f_in, n)
        C_w_0 = np.tile(C_f_in, n) 
        C_0 = np.r_[C_f_0, C_w_0]
        
        if options['model'] is not 'both':
            #### Calculating for only one model
            try:
                count = i + 1
                print('Solving for T_f_in = {0} with {1} model'
                      .format(T_f_in[i], options['model']))
                Y = odeint(func, C_0, tspan, args=(inlet_species, fixed_dict, 
                                                   C_f_in, T_f_in[i], 
                                                   const, thermo_object, 
                                                   react_const, system, 
                                                   rate_basis, rate_units,
                                                   options['model']))
            except RuntimeWarning as e:
                msg = ('The concentration of one of the reactants is going '
                       'close to zero, hence further calculations can result '
                       'in inaccurate results.')
                temp = msg + 'Calculated till {}'.format(T_f_in[i-1])
                print(temp)
                count = i
                break

        elif options['model'] is 'both':
            #### Comparing two models
            print('Solving for T_f_in = {0} with {1} model'
                  .format(T_f_in[i], 'Sh_phi'))
            Y = odeint(func, C_0, tspan, args=(inlet_species, fixed_dict, 
                                               C_f_in, T_f_in[i], 
                                               const, thermo_object, 
                                               react_const, system, 
                                               rate_basis, rate_units,
                                               'Sh_phi'))

            print('Solving for T_f_in = {0} with {1} model'
                  .format(T_f_in[i], 'Sh_inf'))
            Y_inf = odeint(func, C_0, tspan, args=(inlet_species, fixed_dict, 
                                               C_f_in, T_f_in[i], 
                                               const, thermo_object, 
                                               react_const, system, 
                                               rate_basis, rate_units,
                                               'Sh_inf'))
    
            #### Calculating the conversion for Sh_inf model
            C_f_ss_inf = Y_inf[-1, :no_of_species*n]

            C_f_ss_mat_inf = C_f_ss_inf.reshape((no_of_species, n), order='F')
            CH4_exit_inf = C_f_ss_mat_inf[A_index, -1]
            O2_exit_inf = C_f_ss_mat_inf[B_index, -1]
    
            conv_A_inf[i] = (1 - CH4_exit_inf/c_A_in)
            conv_B_inf[i] = (1 - O2_exit_inf/c_B_in)

        else:
            raise Exception ('No such model exists!!!')

        #### Conversion calculations
        C_f_ss = Y[-1, :no_of_species*n]

        C_f_ss_mat = C_f_ss.reshape((no_of_species, n), order='F')
        CH4_exit = C_f_ss_mat[A_index, -1]
        O2_exit = C_f_ss_mat[B_index, -1]
    
        conv_A[i] = (1 - CH4_exit/c_A_in)
        conv_B[i] = (1 - O2_exit/c_B_in)
        
    #### Some random plotting
    if options['model'] is not 'both': 
        fig, ax = plt.subplots()

        ax.plot(T_f_in[:count], conv_A[:count], color='b', label='CH4 conv')
        ax.plot(T_f_in[:count], conv_B[:count], color='r', label='O2 conv')
        ax.legend(loc='best')
        ax.set_xlim([0, 1])
        ax.set_ylim(T_f_in_span)
        ax.set_xlabel('Inlet Temperature, in K')
        ax.set_ylabel('Conversions')
    
    else:
        fig, ax = plt.subplots()

        ax.plot(T_f_in, conv_A, color='b', label=r'$\mathbf{Sh_\phi}$ model')
        ax.plot(T_f_in, conv_A_inf, color='r', label=r'$\mathbf{Sh_\infty}$ model')
        ax.legend(loc='best')
        ax.set_ylim([0, 1])
        ax.set_xlim(T_f_in_span)
        ax.set_xlabel('Inlet Temperature, in K')
        ax.set_ylabel(r'Conversion of $\mathbf{CH_4}$')
        
        fig, ax1 = plt.subplots()
        ax1.plot(T_f_in, conv_B, color='b', label= r'$\mathbf{Sh_\phi}$ model')
        ax1.plot(T_f_in, conv_B_inf, color='r', label= r'$\mathbf{Sh_\infty}$ model')
        ax1.legend(loc='best')
        ax1.set_ylim([0, 1])
        ax1.set_xlim(T_f_in_span)
        ax1.set_xlabel('Inlet Temperature, in K')
        ax1.set_ylabel(r'Conversion of $\mathbf{O_2}$')
    plt.show()
    return Y
        
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
    R_omega = 0.264e-03                  
    R_omega_wc = 50e-06
    particle_density = 3600

    n = 2
    nT = 30
    tspan = np.linspace(0, 100, 1000000)
    T_f_in_span = [400, 1160]

    #### Fixed inputs
    fixed_var = ['inlet_ratio', 'pressure', 'velocity',
                 'length', 'R_omega', 'R_omega_wc', 'particle_density']
    fixed_val = [inlet_ratio, pressure, velocity, 
                 length, R_omega, R_omega_wc, particle_density]
    fixed_dict = dict(zip(fixed_var, fixed_val))
    
    #### Solver Options
    Testing = False
    save = True
    model = 'both'

    options = {'Testing' : Testing,
               'save' : save,
               'model': model}

    #### Solving the ODEs
    Y = low_D_Sh_phi_solver(fixed_dict, T_f_in_span, tspan,
                            react_sys, system, catalyst,
                            rate_basis, rate_units, inlet_species, n, nT, 
                            options)
    
    t1 = time.time()
    print('Time taken: ',(t1 - t0))
