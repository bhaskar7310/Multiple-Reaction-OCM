#### This module calculates the delta G and K_eq values of reactions provided in the reaction profile. If the 
#### rates of this reaction profile is available it will also calculate the K_eq from the provided reaction rate 
#### file and thereby compare. (One important thing to implement is to calculate K_eq values for reversible 
#### reactions. That makes more sense)


#### Importing built-in modules
import numpy as np
import os

#### Importing user-defined modules
import reaction_model as react
import stoichio_coeff_calculator as stoich
import GRI_data_calculator as gri
import matplotlib.pyplot as plt

def thermo_chem_calc(filename, catalyst = None):
    
    species, nu, rev_bool = stoich.main_func(filename)    # nu is a ndarray of shape (no. of species, no_of_react)
    coeff_data = gri.gri_data(species)
    
    #### Fixed dimension
    no_of_react = nu.shape[1] 
    R_g = 8.314
    
    coeff_data = gri.gri_data(species)    
   
    #### Segregating the coefficients and making a big fat matrix out of it
    no_of_species = len(species)
    no_of_react = nu.shape[1]
    index = np.arange(no_of_species)
    
    Cp = np.zeros([2, no_of_species, 5])       # Pre-allocation of C_p matrix
    H = np.zeros([2, no_of_species, 6])        # Pre-allocation of H matrix
    S = np.zeros([2, no_of_species, 6])        # Pre-allocation of S matrix
    
    for e1, elem in zip(index, species):
        coeff = np.array(coeff_data[elem])
        Cp[0, e1, :] = R_g * coeff[:5]
        H[0, e1, :] = R_g * coeff[:6]
        S[0, e1, :] = R_g * np.r_[coeff[:5], coeff[6]]

        Cp[1, e1, :] = R_g * coeff[7:12]
        H[1, e1, :] = R_g * coeff[7:13]
        S[1, e1, :] = R_g * np.r_[coeff[7:12], coeff[13]]
    
    Temp = np.linspace(300, 1600, 10)
    n = len(Temp)

    del_G_T = np.zeros([no_of_react, n])
    K_eq_all = np.zeros_like(del_G_T)
    
    del_H1 = np.dot(nu.T, H[0, :, :])
    del_H2 = np.dot(nu.T, H[1, :, :])
    del_H_all = np.r_[del_H1, del_H2]
    del_H = del_H_all.reshape(2, no_of_react, 6)
    
    del_S1 = np.dot(nu.T, S[0, :, :])
    del_S2 = np.dot(nu.T, S[1, :, :])
    del_S_all = np.r_[del_S1, del_S2]
    del_S = del_S_all.reshape(2, no_of_react, 6)

    for i in range(n):
        T = Temp[i]
        H_T_dependence = np.array([T, T**2/2, T**3/3, T**4/4, T**5/5, 1]).reshape(6, 1)
        S_T_dependence = np.array([np.log(T), T, T**2/2, T**3/3, T**4/4, 1]).reshape(6, 1)
        
        if T > 1000:
            del_H_T = np.dot(del_H[0, :, :], H_T_dependence)
            del_S_T = np.dot(del_S[0, :, :], S_T_dependence)
        else:    
            del_H_T = np.dot(del_H[1, :, :], H_T_dependence)
            del_S_T = np.dot(del_S[1, :, :], S_T_dependence)
        print(del_S_T)
        del_G_T[:, [i]] = del_H_T - T * del_S_T
#        print(del_G_T[:, [i]])
        K_eq_all[:, [i]] = np.exp(-del_G_T[:, [i]]/(R_g * T)) 
    
    #### Calculating the Equilibrium constansts only for reversible reactions
    K_eq = rev_bool.reshape(len(rev_bool), 1) * K_eq_all 
    rev_react_id = np.arange(no_of_react) * rev_bool
    
#    #### Plotting Shotting
#    
#    #### Plotting Del_G values for all reactions
#    fig, ax = plt.subplots()
#    
#    del_G_zeroLine = np.zeros(n)
#    label = [] 
#    
#    for i in range(no_of_react):
#        ax.plot(Temp, del_G_T[i, :])
#        label.append('Reaction: {}'.format(i+1))
#    
#    reactName, ext = os.path.splitext(filename)
#    ax.set_title(('Del_G(T) values of ' + reactName), fontsize= 14)
#    ax.set_xlabel('Temperature (K)', fontsize= 14)
#    ax.set_ylabel('Del_G(T)', fontsize= 14)
#    ax.legend(tuple(label))
#    ax.plot(Temp, del_G_zeroLine, linestyle= '--')
#    ax.set_xlim(Temp[0], Temp[-1])
    
#    #### Plotting Thermodynamically retrieved K_eq values for reversible reactions only
#    if any(rev_react_id):
#        fig, ax1 = plt.subplots()
#        label_k_eq = [] 
#        rev_react_no = []
#    
#        for elem in rev_react_id:
#            if elem:
#                rev_react_no.append(elem)
#                ax1.semilogy(Temp, K_eq[elem, :])
#                label_k_eq.append('Reaction: {}'.format(elem+1))
#            
#            ax1.set_title(('Keq values of ' + reactName), fontsize= 14) 
#            ax1.set_xlabel('Temperature (K)', fontsize= 14)
#            ax1.set_ylabel('Keq', fontsize= 14)
#            ax1.legend(tuple(label_k_eq))
#            ax1.set_xlim(Temp[0], Temp[-1])
#    
#
#        #### Comparison of Keq values calculated from thermodynamics and that from literature
#        homo, cat, homo_index, cat_index = react.instantiate(filename, catalyst)
#        K_eq_list, react_id = homo.equilibrium_const(Temp) 
#
#        rev_check = np.array(homo_index) * np.array(react_id)
#
#        rev_react_rate_no = []
#        for elem in rev_check:
#            if elem:
#                rev_react_rate_no.append(elem)
#
#        rev_react_no = np.array(rev_react_no)
#        rev_react_rate_no = np.array(rev_react_rate_no)
#        
#        if all(rev_react_no != rev_react_rate_no):
#            raise Exception ('There are some discrepancies in identifying the reversible reactions. Check the code and reaction files.')
#
#        #### Comparing the plots derived from thermodynamics and literature
#        no_of_plots = len(rev_react_rate_no)
#        fig, ax2 = plt.subplots(no_of_plots, sharex= True)
#        
#        if no_of_plots == 1:
#            ax2.semilogy(Temp, K_eq[rev_react_no[0], :])
#            ax2.semilogy(Temp, K_eq_list, linestyle= '--')
#            ax2.set_title('Reaction: {}'.format(i))
#            
#            ax2.set_xlabel('Temperature (K)', fontsize= 14)
#        
#        else:   
#            for i in range(no_of_plots):
#                ax2[i].semilogy(Temp, K_eq[rev_react_no[i], :])
#                ax2[i].semilogy(Temp, K_eq_list[:, i], linestyle= '--')
#                ax2[i].set_title('Reaction: {}'.format(i))
#        
#            ax2[-1].set_xlabel('Temperature (K)', fontsize= 14)
#
#    else:
#        print('No Reversible reaction is present in the system.')
    
    #### Shipping
    return K_eq


if __name__ == '__main__':
    react_sys = 'OCM_Arun_Kota.txt'
    catalyst = 'Model'
    K_eq = thermo_chem_calc(react_sys, catalyst)
    

    
