'''
This module does the reverse calculations and analyzes the stored data 
obtained from the detailed calculations. In terms of analysis,
it calculates the reaction rates (we will do this in an interactive plot),
ratio of HC to O2 on bulk and surface and others and then plots them as 
requested by the user.

We will do this in an object oriented pattern.
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from itertools import cycle

import constants
import reaction as react
style.use('ourstyle')

def bif_dia_plot(fixed_dict, bif_par_var, react_system, system, 
                 catalyst, inlet_species, b_options, a_options):
    '''
    This function will load the correct file in order to plot the 
    different bifurcation diagrams.'''
    
    #### Specifying linestyles and colors
    lines = [':', '-', '--', ':', '-.']
    no_of_lines = len(lines)

    units = {'inlet_ratio' : '',
             'pressure'    : ' atm',
             'tau'         : 's',
             'R_omega'     : 'mm',
             'R_omega_wc'  : r'$\mathbf{\mu m}$',
             'particle_density' : 'kg/m3'}
    
    symbol = {'inlet_ratio' : r'$\mathbf{CH_4/O_2}$',
              'pressure'    : r'$\mathbf{P}$',
              'tau'         : r'$\mathbf{\tau}$',
              'R_omega'     : r'$\mathbf{R_\Omega}$',
              'R_omega_wc'  : r'$\mathbf{R_{\Omega, wc}}$',
              'particle_density' : r'$\mathbf{\rho}$'}
    
    #### Identifying the variable with multiple inputs
    count = 0
    multiple = False
    multiple_variable = 'inlet_ratio'
    for key, val in fixed_dict.items():
        try:
            len(val)
        except TypeError:
            continue
        else:
            multiple = True
            multiple_variable = key
            break

    if multiple:
        values = fixed_dict[multiple_variable]
    else:
        values = []
        values.append(fixed_dict[multiple_variable])

    for val in values:
        count += 1
        fixed_dict[multiple_variable] = val
        if multiple_variable == 'R_omega':
            fixed_dict['R_omega_wc'] = fixed_dict['R_omega']/0.25e-03 * 100e-06
        style_index = count % no_of_lines

        #### Retreiving the data from .npz file
        npzfile = _load_file(react_system, system, catalyst, fixed_dict)
        T_bd_arr, F_in_arr, species_ID_arr = npzfile.files
        T_bd = npzfile[T_bd_arr]
        F_in = npzfile[F_in_arr]
        species_ID = npzfile[species_ID_arr].item()
        n, m = T_bd.shape
         
        if multiple_variable == 'R_omega':
            label = (symbol[multiple_variable] + ' = ' + str(val*1e03) 
              + units[multiple_variable])    
        else:
            label = (symbol[multiple_variable] + ' = ' + str(val) 
              + units[multiple_variable])
        
        if b_options['basic_plot']:

            #### Retrieving the species_specific data from T_bd
            hc_index = species_ID[inlet_species]
            conv_hc = 1 - T_bd[hc_index, :]/F_in[hc_index]

            O2_index = species_ID['O2']
            conv_O2 = 1 - T_bd[O2_index, :]/F_in[O2_index]
            
            #### Plotting Area
            plot_dict = b_options['plots']
            
            if bif_par_var == 'tau':
                xplottype = 'log'
                x_axis = 'Residence Time (s)'
            else:
                xplottype = 'normal'
                x_axis = ('Inlet Fluid Temperature, ' 
                       + r'$\mathbf{T_{f,in}}$' 
                       + ' (K)')
            
            #### Exit Fluid Temperature vs Inlet Fluid Temperature
            if plot_dict['fluid_temp']:

                if count == 1:
                    fig, ax1 = plt.subplots()
                    fig.subplots_adjust(left=0.145, bottom=0.11)
                    ax1.set_xlabel(x_axis)
                    ax1.set_ylabel(('Exit Fluid Temperature, ' 
                                    + r'$\mathbf{T_f}$' + ' (K)'))
                    axis_limits = (b_options['xaxis_lim'] 
                                + b_options['yaxis_lim'])
                    ax1.axis(axis_limits)
                    
                if xplottype == 'log':
                    ax1.semilogx(T_bd[-1, :], T_bd[-2, :], 
                                 linestyle=lines[style_index], 
                                 label= label)
                else:
                    ax1.plot(T_bd[-1, :], T_bd[-2, :], 
                             linestyle=lines[style_index], 
                             label= label)
                ax1.legend(loc='lower right')
                plt.show()
            
            #### Solid Temperature vs Inlet Fluid Temperature
            if plot_dict['solid_temp']:

                if count == 1:
                    fig, ax2 = plt.subplots()
                    ax2.set_xlabel(x_axis, fontsize= 14, fontweight= 'bold')
                    ax2.set_ylabel(('Catalyst Surface Temperature ' 
                                     + r'$\mathbf{T_s}$' + ' (K)'))

                    axis_limits = (b_options['xaxis_lim'] 
                                + b_options['yaxis_lim'])
                    ax2.axis(axis_limits)
   
                if xplottype == 'log':
                    ax2.semilogx(T_bd[-1, :], T_bd[-3, :], 
                                 linestyle=lines[style_index], 
                                 label= label)
                else:
                    ax2.plot(T_bd[-1, :], T_bd[-3, :], 
                             linestyle=lines[style_index], 
                             label= label)
                ax2.legend(loc='best')

            #### Conversion of Hydrocarbon and O2
            if plot_dict['conversion']:

                if count == 1:
                    fig31, ax31 = plt.subplots()
                    fig32, ax32 = plt.subplots()

                    axis_limits_CH4 = b_options['xaxis_lim'] + [0, 0.5]
                    ax31.set_xlabel(x_axis)
                    ax31.set_ylabel(('Conversion of ' + r'$\mathbf{CH_4}$'))
                    ax31.axis(axis_limits_CH4)
                    
                    axis_limits = b_options['xaxis_lim'] + [0, 1]
                    ax32.set_xlabel(x_axis)
                    ax32.set_ylabel(('Conversion of ' + r'$\mathbf{O_2}$'))
                    ax32.axis(axis_limits)
 
                if xplottype == 'log':
                    ax31.semilogx(T_bd[-1, :], conv_hc, 
                                  linestyle=lines[style_index], 
                                  label= label)
                    ax32.semilogx(T_bd[-1, :], conv_O2, 
                                  linestyle=lines[style_index], 
                                  label= label)
                else:    
                    ax31.plot(T_bd[-1, :], conv_hc, 
                              linestyle=lines[style_index], 
                              label= label)
                    ax32.plot(T_bd[-1, :], conv_O2, 
                              linestyle=lines[style_index], 
                              label= label)
                ax31.legend(loc='best')
                ax32.legend(loc='best')
                
           
            #### Combined Selectivities and Yields
            if (plot_dict['select_comb'] or plot_dict['yield_comb']):
                
                elem_comb = ['CO', 'CO2', 'C2H6', 'C2H4', 'C2H2']
                selectivity = [np.zeros(m)] * 5
                select_dict = dict(zip(elem_comb, selectivity))
                yield_dict = dict(zip(elem_comb, selectivity))

                for elem in elem_comb:
                    elem_index = species_ID.get(elem, None)
                    carbon_no = species_identifier(elem)
                    if (elem_index != None) and (carbon_no != 0):
                        yield_elem = (T_bd[elem_index, :] - F_in[elem_index]) \
                                   * carbon_no/F_in[hc_index]
                                
                        selectivity_elem = (T_bd[elem_index, :] 
                                         - F_in[elem_index]) \
                                         * carbon_no \
                                         / (F_in[hc_index] - T_bd[hc_index, :])
                        
                        select_dict[elem] = selectivity_elem
                        yield_dict[elem] = yield_elem

                selectivity_COx = select_dict['CO'] + select_dict['CO2']
                selectivity_C2 = select_dict['C2H6'] + select_dict['C2H4'] \
                               + select_dict['C2H2']

                #selectivity_C2 = np.where(np.isnan(selectivity_C2), 0.0, selectivity_C2)
                #selectivity_C2 = np.where(np.isinf(selectivity_C2), 0.0, selectivity_C2)
                #selectivity_C2 = np.where(selectivity_C2 < 0.0, 0.0, selectivity_C2)
                #selectivity_C2 = np.where(selectivity_C2 > 1.0, 0.0, selectivity_C2)
                if fixed_dict['inlet_ratio'] == 4:
                     selectivity_C2[:1500] = 0.0 # Didn't find any better method, 
                     selectivity_COx[:1500] = 1.0 
                else:
                    selectivity_C2[:1000] = 0.0 # Didn't find any better method, 
                    selectivity_COx[:1000] = 1.0 
                if plot_dict['select_comb']:
                    if (count == 1):
                        fig61, ax6_sc1 = plt.subplots()
                        fig62, ax6_sc2 = plt.subplots()
                        axis_limits = b_options['xaxis_lim'] + [0, 1]
                        
                        ax6_sc1.set_xlabel(x_axis)
                        ax6_sc1.set_ylabel(('Selectivity of (' 
                                        +  r'$\mathbf{CO + CO_{2}}$' + ')'))
                        ax6_sc1.axis(axis_limits)
                        
                        ax6_sc2.set_xlabel(x_axis)
                        ax6_sc2.set_ylabel(('Selectivity of (' 
                            +  r'$\mathbf{C_{2}H_{6} + C_{2}H_{4}}$' + ')'))
                        ax6_sc2.axis(axis_limits)
                   
                    if xplottype == 'log':
                        ax6_sc1.semilogx(T_bd[-1, :], selectivity_COx, 
                                    linestyle=lines[style_index], label=label) 
                        ax6_sc2.semilogx(T_bd[-1, :], selectivity_C2, 
                                    linestyle=lines[style_index], label=label)
                    else:
                        ax6_sc1.plot(T_bd[-1, :], selectivity_COx, 
                                     linestyle=lines[style_index], label=label)
                        ax6_sc2.plot(T_bd[-1, :], selectivity_C2, 
                                     linestyle=lines[style_index], label=label)
                
                    ax6_sc1.legend(loc='best')
                    ax6_sc2.legend(loc='best')
                
                if plot_dict['yield_comb']:
                    yield_COx = yield_dict['CO'] + yield_dict['CO2']
                    yield_C2 = yield_dict['C2H6'] + yield_dict['C2H4']\
                             + yield_dict['C2H2']
                    
                    
                    if (count == 1):
                        fig71, ax7_yc1 = plt.subplots()
                        fig71.subplots_adjust(left=0.145, bottom=0.11)
                        
                        fig72, ax7_yc2 = plt.subplots()
                        fig72.subplots_adjust(left=0.145, bottom=0.11)
                        axis_limits = b_options['xaxis_lim'] + [0, 0.3]
                        
                        ax7_yc1.set_xlabel(x_axis)
                        ax7_yc1.set_ylabel(('Yields of (' 
                                        +  r'$\mathbf{CO + CO_{2}}$' + ')'))
                        ax7_yc1.axis(axis_limits)
    
                        ax7_yc2.set_xlabel(x_axis)
                        ax7_yc2.set_ylabel(('Yield of (' 
                            +  r'$\mathbf{C_{2}H_{6} + C_{2}H_{4}}$' + ')'))
                        ax7_yc2.axis(axis_limits)
                        
                    if xplottype == 'log':
                        ax7_yc1.semilogx(T_bd[-1, :], yield_COx, 
                                    linestyle=lines[style_index], label=label)
                        ax7_yc2.semilogx(T_bd[-1, :], yield_C2,
                                    linestyle=lines[style_index], label=label)
                    else:
                        ax7_yc1.plot(T_bd[-1, :], yield_COx,
                                    linestyle=lines[style_index], label=label)
                        ax7_yc2.plot(T_bd[-1, :], yield_C2, 
                                    linestyle=lines[style_index], label=label)
    
                    ax7_yc1.legend(loc='best')
                    ax7_yc2.legend(loc='best')
            
            #### Yields of all 'products', (Identifying the compound and then 
            #### identifying limiting reactant is
            #### important, we will do it later)
            if plot_dict['yield_all']:

                products = b_options['products']
                if products:
                    fig4, ax4_y = plt.subplots()
                    label = []        
                    not_calc_elem = []

                    for elem in products:
                        elem_index = species_ID.get(elem, None)
                        carbon_no = species_identifier(elem)
                        
                        if (elem_index != None) and (carbon_no != 0):
                            yield_elem = T_bd[elem_index, :] \
                                       * carbon_no/F_in[hc_index]
                            
                            if xplottype == 'log':
                                ax4_y.semilogx(T_bd[-1, :], yield_elem)
                            else:
                                ax4_y.plot(T_bd[-1, :], yield_elem)
                            label.append(elem)
                        else:
                            not_calc_elem.append(elem)

                ax4_y.set_xlabel(x_axis)
                ax4_y.set_ylabel(('Yield of Products'))
                ax4_y.legend(tuple(label), loc='best')
                
                axis_limits = b_options['xaxis_lim'] + [0, 0.5]
                ax4_y.axis(axis_limits)
                title = ('Inlet ratio = ' + str(fixed_dict['inlet_ratio'])
                          + ', tau = ' + str(fixed_dict['tau']) 
                          + 's, Radius = ' + str(fixed_dict['R_omega']*1e03)
                          + 'mm.')
                ax4_y.set_title(title)

                if not_calc_elem:
                    print('The yields of {} are not calculated, '
                          'the program thinks they are '
                          'unimportant'.format(not_calc_elem))
                
            #### Selectivity of all products
            if plot_dict['select_all']:

                products = b_options['products']
                if products:
                    fig5, ax5_s = plt.subplots()
                    label = []        
                    not_calc_elem = []

                    for elem in products:
                        elem_index = species_ID.get(elem, None)
                        carbon_no = species_identifier(elem)
                        if (elem_index != None) and (carbon_no != 0):
                            yield_elem = (T_bd[elem_index, :] - F_in[elem_index]) \
                                       * carbon_no/F_in[hc_index]
                            selectivity_elem = (T_bd[elem_index, :] 
                                             - F_in[elem_index]) \
                                             * carbon_no \
                                             / (F_in[hc_index] - T_bd[hc_index, :])
                            
                            if xplottype == 'log':
                                ax5_s.semilogx(T_bd[-1, :], selectivity_elem, 
                                              linewidth= 2.0)
                            else:
                                ax5_s.plot(T_bd[-1, :], selectivity_elem, 
                                          linewidth= 2.0)
                            label.append(elem)
                        else:
                            not_calc_elem.append(elem)
                
                ax5_s.set_xlabel(x_axis)
                ax5_s.set_ylabel(('Selectivity of Products'))
                ax5_s.legend(tuple(label), loc='best')
                axis_limits = b_options['xaxis_lim'] + [0, 1]
                ax5_s.axis(axis_limits)    
                title = ('Inlet ratio = ' + str(fixed_dict['inlet_ratio'])
                          + ', tau = ' + str(fixed_dict['tau'])
                          + 's, Radius = ' + str(fixed_dict['R_omega']*1e03)
                          + 'mm.')
                ax5_s.set_title(title)
                if not_calc_elem:
                    print('The yields of {} are not calculated, '
                          'the program thinks they are '
                          'unimportant'.format(not_calc_elem))

        if a_options['analysis_plot']:

            #### Retrieving the species_specific data from T_bd
            const, thermo_object = constants.fixed_parameters(react_system)
            homo, cat, homo_index, cat_index = react.instantiate(react_system, 
                                                              const, catalyst)
            
            #### The local species ID is useful in Thiele Modulus calculations
            local_species = const['species']
            no_of_local_species = len(local_species)
            local_ID = np.arange(no_of_local_species)
            local_species_ID = dict(zip(local_species, local_ID))
            hc_local_index = local_species_ID[inlet_species]
            O2_local_index = local_species_ID['O2']

            #### This is the species_ID of the actual calculation coming from
            #### the calling function   
            no_of_species = len(species_ID)
            species = list(species_ID.keys())
            hc_index = species_ID[inlet_species]
            O2_index = species_ID['O2']

            #### User-requested plots
            plot_dict = a_options['plots']
            
            if bif_par_var == 'tau':
                xplottype = 'log'
                x_axis = 'Residence Time (s)'
            else:
                xplottype = 'normal'
                x_axis = ('Inlet Fluid Temperature ' 
                       + r'$\mathbf{T_{f,in}}$' 
                       + ' (K)')
            
            #### Calculation of reaction rates

            #### Unpacking the matrix T_bd
            F_j = T_bd[:no_of_species, :]                         
            C_s = T_bd[no_of_species:2*no_of_species, :]
            T_s = T_bd[2*no_of_species, :]
            T_f = T_bd[2*no_of_species + 1, :]                          
            bif_par = T_bd[-1, :]
            no_of_iter = T_bd.shape[1]
            
            T_f_in = T_bd[-1, :]
            
            #### Unpacking the constants
            P = fixed_dict['pressure']
            R_omega = fixed_dict['R_omega']
            R_omega_wc = fixed_dict['R_omega_wc']
            R_omega_w = const['R_omega_w']
            nu = const['nu']

            nu_cat= nu.T[cat_index]
            nu_cat_inlet_species = nu_cat[:, hc_local_index]
            nu_cat_O2 = nu_cat[:, O2_local_index]
            
            eps_f = 4*R_omega**2/(2*R_omega + R_omega_wc + R_omega_w)**2

            #### Defining mole fractions and concentrations
            C_total = 101325 * P/(8.314 * T_f)
            Y_j = F_j/np.sum(F_j, axis=0)
            C_f = Y_j * C_total
            C_in = 101325 * P/(8.314 * T_f_in)

            #### Calculation of reaction rates [Fig. 1]
            if plot_dict['reaction_rates'] or plot_dict['thiele_modulus']:
                homo_basis, cat_basis = a_options['rate_basis']
                homo_units, cat_units = a_options['rate_units']
                all_homo_rate = np.zeros([len(homo_index), no_of_iter], dtype=float)
                all_cat_rate = np.zeros([len(cat_index), no_of_iter], dtype=float)
                
                for i in range(no_of_iter):
                    if (system == 'cat'):
                        homo_rate = np.zeros(len(homo_index))    
                        cat_rate = cat.act_rate(C_s[:, i], species, cat_basis, T_s[i], P)
                    elif (system == 'homo'):
                        cat_rate = np.zeros(len(cat_index))
                        homo_rate = homo.act_rate( C_f[:, i], species, homo_basis, T_f[i], P)
                    else:
                        homo_rate = homo.act_rate( C_f[:, i], species, homo_basis, T_f[i], P)
                        cat_rate = cat.act_rate(C_s[:, i], species, cat_basis, T_s[i], P)

                    all_homo_rate[:, i] = homo_rate[:]
                    all_cat_rate[:, i] = cat_rate[:]

                #### This is not required for the time being
                # We got to check the units and perform subsequent calculations 
                if homo_basis == 'mole_fraction':
                    if homo_units != 'second':
                        raise Exception ('There is a discrepancy '
                                         'in the homogeneous reaction rate')
                if (cat_units == 'kg_sec'):
                    all_cat_rate *= fixed_dict['particle_density']
                elif (cat_units == 'gm_sec'):
                    all_cat_rate *= fixed_dict['particle_density'] * 1000

                # Here we make two separate variables for catalytic reactions
                #homo_rate_C0 = all_homo_rate/C_in
                cat_rate_C0 = all_cat_rate.copy()
                cat_rate_C0_surf = all_cat_rate.copy()
                all_cat_rate *= R_omega_wc * eps_f / R_omega
               
                #return all_cat_rate - cat_rate_C0
                #### Plotting the reaction rates
                if plot_dict['reaction_rates'] and not multiple:

                    #### Catalytic reaction rates
                    fig, ax1 = plt.subplots()
                    label_cat = ()
                    for i in range(len(cat_index)):
                        if xplottype == 'log':
                            ax1.loglog(bif_par, R_omega_wc * all_cat_rate[i, :])
                        else:
                            ax1.plot(bif_par, R_omega_wc * all_cat_rate[i, :])
                        label = 'Reaction No.= ' + str(cat_index[i] + 1)
                        label_cat += (label,)
                    
                    ax1.set_xlabel(x_axis)
                    ax1.set_ylabel(('Catalytic reaction rates (' 
                                    + r'$\mathbf{mol/m^2 s}$' + ')'))
                    ax1.axes.set_xlim(10, 1200) # Hardcoded
                    ax1.legend(label_cat, loc='best')

                    #### Homogeneous reaction rates
                    fig, ax11 = plt.subplots()
                    label_homo = ()
                    for i in range(len(homo_index)):
                        if xplottype == 'log':
                            ax11.loglog(bif_par, all_homo_rate[i, :])
                        else:
                            ax11.plot(bif_par, all_homo_rate[i, :])
                        label = 'Reaction No.= ' + str(homo_index[i] + 1) 
                        label_homo += (label,)
                    
                    ax11.set_xlabel(x_axis)
                    ax11.set_ylabel(('Homogeneous reaction rates (' 
                                    + r'$\mathbf{mol/m^3 s}$' + ')'))
                    ax11.axes.set_xlim(10, 1200) # Hardcoded
                    ax11.legend(label_homo, loc='best')
                
                #### Plotting Thiele Modulus
                if plot_dict['thiele_modulus']:

                    #### Normal Thiele Modulus based on vol. cat. reac. rate
                    inlet_species_cat_rate = np.dot(-nu_cat_inlet_species, 
                                                    cat_rate_C0)
                    D_f = 9.8e-10 * T_f**1.75
                    D_e = 0.01 * D_f
                    thiele_mod_inlet_species = (inlet_species_cat_rate
                                             / C_s[hc_index, :]
                                             * R_omega_wc**2/D_e)
                                 
                    #### Surface Thiele Modulus based on surf. cat. reac. rate
                    cat_rate_C0_surf *= R_omega_wc
                    inlet_species_surf_cat_rate = np.dot(-nu_cat_O2, 
                                                         cat_rate_C0_surf)
                    surf_thiele_modulus = (inlet_species_surf_cat_rate
                                        / C_s[O2_index, :]
                                        * R_omega/D_f)

                    #print(fixed_dict['tau'])
                    #damkohler_second_type = 1/ (inlet_species_surf_cat_rate
                    #                      / C_s[hc_index, :])
                    if multiple_variable == 'R_omega':
                        label1 = (symbol['R_omega_wc'] + ' = ' 
                               + str(int(100*val/0.25e-03)) 
                               + units['R_omega_wc'])    
                        label2 = (symbol['R_omega'] + ' = ' + str(val*1e03)
                               + units['R_omega'])
                    else:
                        label1 = (symbol[multiple_variable] + ' = ' + str(val) 
                          + units[multiple_variable])
                        label2 = label1

                    if count == 1:
                        fig, ax111 = plt.subplots()
                        ax111.set_xlabel(x_axis)
                        ax111.set_ylabel(('Thiele Modulus (' 
                                          + r'$\mathbf{\phi_{wc}^2}$' + ')'))
                        ax111.axes.set_xlim(10, 1200) # Hardcoded

                        fig, ax112 = plt.subplots()
                        ax112.set_xlabel(x_axis)
                        ax112.set_ylabel(('External Damkohler No. (' 
                                         + r'$\mathbf{Da_{ext}}$' + ')'))
                        ax112.axes.set_xlim(10, 1200) # Hardcoded
                        ax112.axes.set_ylim(1e-06, 1e06)
                    ax111.plot(bif_par, thiele_mod_inlet_species, 
                               linestyle=lines[style_index],
                               label=label1)
                    ax112.semilogy(bif_par, surf_thiele_modulus,
                               linestyle=lines[style_index],
                               label=label2)

                    ax111.legend(loc='best')
                    ax112.legend(loc='best')
                    #ax113.legend(loc='best')
    return T_bd

def _analysis(fixed_dict, bif_par_var, react_system, system, catalyst,
             inlet_species, T_bd, F_in, species_ID, multiple, count, a_options):
    
    #### LineStyles
    lines = ['-', '--', '-.', ':']
    no_of_lines = len(lines)
    style_index = count % no_of_lines
    #linecycler = cycle(lines)
    
    #### Retrieving the species_specific data from T_bd
    const, thermo_object = constants.fixed_parameters(react_system)
    homo, cat, homo_index, cat_index = react.instantiate(react_system, 
                                                         const, catalyst)
    
    #### The local species ID is useful in Thiele Modulus calculations
    local_species = const['species']
    no_of_local_species = len(local_species)
    local_ID = np.arange(no_of_local_species)
    local_species_ID = dict(zip(local_species, local_ID))
    hc_local_index = local_species_ID[inlet_species]
    O2_local_index = local_species_ID['O2']

    #### This is the species_ID of the actual calculation coming from
    #### the calling function   
    no_of_species = len(species_ID)
    species = list(species_ID.keys())
    hc_index = species_ID[inlet_species]
    O2_index = species_ID['O2']

    #### User-requested plots
    plot_dict = a_options['plots']
    
    if bif_par_var == 'tau':
        xplottype = 'log'
        x_axis = 'Residence Time (s)'
    else:
        xplottype = 'normal'
        x_axis = ('Inlet Fluid Temperature ' 
               + r'$\mathbf{T_{f,in}}$' 
               + ' (K)')
    
    #### Calculation of reaction rates

    #### Unpacking the matrix T_bd
    F_j = T_bd[:no_of_species, :]                         
    C_s = T_bd[no_of_species:2*no_of_species, :]
    T_s = T_bd[2*no_of_species, :]
    T_f = T_bd[2*no_of_species + 1, :]                          
    bif_par = T_bd[-1, :]
    no_of_iter = T_bd.shape[1]
    
    T_f_in = T_bd[-1, :]
    
    #### Unpacking the constants
    R_omega = fixed_dict['R_omega']
    R_omega_wc = fixed_dict['R_omega_wc']
    R_omega_w = const['R_omega_w']
    nu = const['nu']
    
    nu_cat= nu.T[cat_index]
    nu_cat_inlet_species = nu_cat[:, hc_local_index]
    nu_cat_O2 = nu_cat[:, O2_local_index]
    
    eps_f = 4*R_omega**2/(2*R_omega + R_omega_wc + R_omega_w)**2
    
    #### Defining mole fractions and concentrations
    C_total = 101325 /(8.314 * T_f)
    Y_j = F_j/np.sum(F_j, axis=0)
    C_f = Y_j * C_total
    C_in = 101325 /(8.314 * T_f_in)

    #### Calculation of reaction rates [Fig. 1]
    if plot_dict['reaction_rates'] or plot_dict['thiele_modulus']:
        homo_basis, cat_basis = a_options['rate_basis']
        homo_units, cat_units = a_options['rate_units']
        all_homo_rate = np.zeros([len(homo_index), no_of_iter], dtype=float)
        all_cat_rate = np.zeros([len(cat_index), no_of_iter], dtype=float)
        
        for i in range(no_of_iter):
            if (system == 'cat'):
                homo_rate = np.zeros(len(homo_index))    
                cat_rate = cat.act_rate(species, cat_basis, C_s[:, i], T_s[i])
            elif (system == 'homo'):
                cat_rate = np.zeros(len(cat_index))
                homo_rate = homo.act_rate(species, homo_basis, C_f[:, i], T_f[i])
            else:
                homo_rate = homo.act_rate(species, homo_basis, C_f[:, i], T_f[i])
                cat_rate = cat.act_rate(species, cat_basis, C_s[:, i], T_s[i])

            all_homo_rate[:, i] = homo_rate[:]
            all_cat_rate[:, i] = cat_rate[:]

        #### This is not required for the time being
        # We got to check the units and perform subsequent calculations 
        if homo_basis == 'mole_fraction':
            if homo_units != 'second':
                raise Exception ('There is a discrepancy '
                                 'in the homogeneous reaction rate')
        if (cat_units == 'kg_sec'):
            all_cat_rate *= fixed_dict['particle_density']
        elif (cat_units == 'gm_sec'):
            all_cat_rate *= fixed_dict['particle_density'] * 1000

        # Here we make two separate variables for catalytic reactions
        #homo_rate_C0 = all_homo_rate/C_in
        cat_rate_C0 = all_cat_rate.copy()
        all_cat_rate *= R_omega_wc * eps_f / R_omega
       
        #return all_cat_rate - cat_rate_C0
        #### Plotting the reaction rates
        if plot_dict['reaction_rates'] and not multiple:

            #### Catalytic reaction rates
            fig, ax1 = plt.subplots()
            label_cat = ()
            for i in range(len(cat_index)):
                if xplottype == 'log':
                    ax1.loglog(bif_par, all_cat_rate[i, :], 
                               linestyle=next(linecycler))
                else:
                    ax1.plot(bif_par, all_cat_rate[i, :], 
                             linestyle=next(linecycler))
                label = 'Reaction No.= ' + str(cat_index[i] + 1)
                label_cat += (label,)
            
            ax1.set_xlabel(x_axis)
            ax1.set_ylabel(('Catalytic reaction rates (' 
                            + r'$\mathbf{mol/m^3 s}$' + ')'))
            ax1.axes.set_xlim(10, 1200) # Hardcoded
            ax1.legend(label_cat, loc='best')

            #### Homogeneous reaction rates
            fig, ax11 = plt.subplots()
            label_homo = ()
            for i in range(len(homo_index)):
                if xplottype == 'log':
                    ax11.loglog(bif_par, all_homo_rate[i, :], 
                                linestyle=next(linecycler))
                else:
                    ax11.plot(bif_par, all_homo_rate[i, :], 
                              linestyle=next(linecycler))
                label = 'Reaction No.= ' + str(homo_index[i] + 1) 
                label_homo += (label,)
            
            ax11.set_xlabel(x_axis)
            ax11.set_ylabel(('Homogeneous reaction rates (' 
                            + r'$\mathbf{mol/m^3 s}$' + ')'))
            ax11.axes.set_xlim(10, 1200) # Hardcoded
            ax11.legend(label_homo, loc='best')
        
        #### Plotting Thiele Modulus based on volumetric cat. reaction rates
        if plot_dict['thiele_modulus']:
            inlet_species_cat_rate = np.dot(-nu_cat_inlet_species, cat_rate_C0)
            thiele_mod_inlet_species = (inlet_species_cat_rate
                                     / C_s[hc_index, :] 
                                     * R_omega_wc**2/1e-06)

            inlet_species_surf_cat_rate = np.dot(-nu_cat_inlet_species, 
                                                 all_cat_rate)
            surf_thiele_modulus = (inlet_species_surf_cat_rate
                                / C_s[hc_index, :]
                                * R_omega**2/1e-08)
            if count == 1:
                fig, ax111 = plt.subplots()
                ax111.set_xlabel(x_axis)
                ax111.set_ylabel(('Thiele Modulus (' 
                                  + r'$\mathbf{\phi^2}$' + ')'))
                ax111.axes.set_xlim(10, 1200) # Hardcoded

                fig, ax112 = plt.subplots()
                ax112.set_xlabel(x_axis)
                ax112.set_ylabel(('Surface Thiele Modulus (' 
                                 + r'$\mathbf{\phi_{s}^2}$' + ')'))
                ax112.axes.set_xlim(10, 1200) # Hardcoded

            ax111.plot(bif_par, thiele_mod_inlet_species, 
                           linestyle=lines[style_index])
            ax112.plot(bif_par, surf_thiele_modulus,
                       linestyle=lines[style_index])


    
    #### Calculation of ratio at the surface [Fig. 2]
    if plot_dict['ratio_surface'] or plot_dict['surface_to_bulk']:
        conc_surf_hc = C_s[hc_index, :]
        conc_surf_O2 = C_s[O2_index, :]
        surface_ratio = conc_surf_hc/conc_surf_O2

        if plot_dict['ratio_surface']:
            fig, ax2 = plt.subplots()
            
            if xplottype == 'log':
                ax2.loglog(bif_par, surface_ratio)
            else:
                ax2.semilogy(bif_par, surface_ratio)
            ax2.set_xlabel(x_axis)
            ax2.set_ylabel(('HC to O2 ratio at surface'))
            #axis_limits = b_options['xaxis_lim'] + b_options['yaxis_lim']
            #ax1.axis(axis_limits)

    #### Calculation of ratio in bulk [Fig. 3]
    if plot_dict['ratio_bulk'] or plot_dict['surface_to_bulk']:
        molar_rate_hc = F_j[hc_index, :]
        molar_rate_O2 = F_j[O2_index, :]
        bulk_ratio = molar_rate_hc/molar_rate_O2

        if plot_dict['ratio_bulk']:
            fig, ax3 = plt.subplots()
            
            if xplottype == 'log':
                ax3.loglog(bif_par, bulk_ratio)
            else:
                ax3.semilogy(bif_par, bulk_ratio)
            ax3.set_xlabel(x_axis)
            ax3.set_ylabel('HC to O2 ratio at bulk')
            #axis_limits = b_options['xaxis_lim'] + b_options['yaxis_lim']
            #ax1.axis(axis_limits)
    
    #### Calculation of ratio of HC to O2 in surface to that in bulk [Fig. 4]
    if plot_dict['surface_to_bulk']:
        surface_to_bulk = surface_ratio/bulk_ratio
        
        fig, ax4 = plt.subplots()
        if xplottype == 'log':
            ax4.loglog(bif_par, surface_to_bulk, color= 'b', 
                         linewidth= 2.0)
        else:
            ax4.semilogy(bif_par, surface_to_bulk, color= 'b', 
                         linewidth= 2.0)
        ax4.set_xlabel(x_axis)
        ax4.set_ylabel('HC to O2 ratio at surface to that in bulk')
        #axis_limits = b_options['xaxis_lim'] + b_options['yaxis_lim']
        #ax1.axis(axis_limits)
        
    if plot_dict['C2H4_C2H6_ratio']:
        ethylene_index = species_ID['C2H4']
        ethane_index = species_ID['C2H6']
        
        ethylene_ethane_ratio = F_j[ethylene_index, :]/F_j[ethane_index, :]
        
        fig, ax5 = plt.subplots()
        if xplottype == 'log':
            ax5.loglog(bif_par, ethylene_ethane_ratio)
        else:
            ax5.semilogy(bif_par, ethylene_ethane_ratio)
        ax5.set_xlabel(x_axis)
        ax5.set_ylabel('Ethylene to Ethane ratio in bulk')


def _load_file(react_system, system, catalyst, fixed_dict, filename='bif_dia'):
    '''
    Loads the specific file if it exists and returns
    the file contents.
    '''
    
    #### File Specifications
    data_vals = [value for (key, value) in sorted(fixed_dict.items())]
    n = len(data_vals)
#    filename = 'bif_dia'
        
    for i in range(n):
        filename += '_{}'.format(data_vals[i])
            
    filename += '.npz'

    react_filename, ext = os.path.splitext(react_system)

    FullfileName = os.path.join(os.getcwd(), react_filename, catalyst.lower(),
                                'Data', system, filename)
    if os.path.exists(FullfileName):
        npzfile = np.load(FullfileName)
        return npzfile
    else:
        raise FileNotFoundError('File with the path: {}'
                                'does not exist.'.format(FullfileName))


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

def _rearrange_species(species_ID_old, species_ID_new, data):
    '''
    This function rearranges the old saved data into the new
    index.
    '''
    data_new = np.zeros_like(data)
    species_ID_old_dict = species_ID_old.tolist()

    for key, old_index in species_ID_old_dict.items():
        new_index = species_ID_new[key]
        data_new[new_index] = data[old_index]
    
    
    #print('Old data', data)
    #print('Old index', species_ID_old)

    #print('\nNew data', data_new)
    #print('New index', species_ID_new)
    return data_new
