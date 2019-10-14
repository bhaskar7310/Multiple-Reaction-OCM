'''
This module does the reverse calculations and analyzes the stored data 
obtained from the detailed calculations. In terms of analysis,
it calculates the reaction rates (we will do this in an interactive plot),
ratio of HC to O2 on bulk and surface and others and then plots them as 
requested by the user.

We will do this in an object oriented pattern.
'''

import numpy as np
import matplotlib.pyplot as plt



def _analysis(fixed_dict, bif_par_var, react_system, system, catalyst,
             inlet_species, T_bd, F_in, species_ID, a_options):
    
    #### Retrieving the species_specific data from T_bd
    const, thermo_object = constants.fixed_parameters(react_system)
    homo, cat, homo_index, cat_index = react.instantiate(react_system, 
                                                         const, catalyst)
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
    
    #### Defining mole fractions and concentrations
    C_total = 101325 /(8.314 * T_f)
    Y_j = F_j/np.sum(F_j, axis=0)
    C_f = Y_j * C_total
    C_in = 101325 /(8.314 * T_f_in)
    
    #### Calculation of reaction rates [Fig. 1]
    if plot_dict['reaction_rates']:
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

        all_cat_rate *= (fixed_dict['R_omega_wc'] * const['eps_f']
                      / fixed_dict['R_omega'])
        
        #### Plotting all the catalytic reaction rates
        fig, ax1 = plt.subplots()

        label_cat = ()
        for i in range(len(cat_index)):
            if xplottype == 'log':
                ax1.loglog(bif_par, all_cat_rate[i, :])
            else:
                ax1.plot(bif_par, all_cat_rate[i, :])
                
            label = 'Rxn No.= ' + str(cat_index[i])
            label_cat += (label,)
        
        ax1.set_xlabel(x_axis)
        ax1.set_ylabel(('Catalytic reaction rates (' + r'$\mathbf{mol/m^3 s}$' + ')'))
        ax1.legend(label_cat)

        #### *****************************************************************
        #### Experimenting with instantaneous selectivities w.r.t rxn rates
        #C2H6_rate = all_cat_rate[0, :] + all_cat_rate[3, :]
        #CO_rate = all_cat_rate[2, :] + all_cat_rate[4, :] + all_cat_rate[1, :]
        #selectivity_C2H6_CO = C2H6_rate/CO_rate
        #fig, ax101 = plt.subplots()
        #if xplottype == 'log':
        #    ax101.loglog(bif_par, selectivity_C2H6_CO, linewidth= 2.0)
        #else:
        #    ax101.semilogy(bif_par, selectivity_C2H6_CO, linewidth= 2.0)
           
        #### *****************************************************************

        #### Plotting all the homogeneous reaction rates
        fig, ax11 = plt.subplots()

        label_homo = ()
        for i in range(len(homo_index)):
            if xplottype == 'log':
                ax11.loglog(bif_par, all_homo_rate[i, :])
            else:
                ax11.plot(bif_par, all_homo_rate[i, :])

            label = 'Rxn No.= ' + str(homo_index[i])
            label_homo += (label,)
        
        ax11.set_xlabel(x_axis)
        ax11.set_ylabel(('Homogeneous reaction rates (' + r'$\mathbf{mol/m^3 s}$' + ')'))
        ax11.legend(label_homo)
#        axis_limits = b_options['xaxis_lim'] + b_options['yaxis_lim']
#        ax1.axis(axis_limits)

    
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
        
