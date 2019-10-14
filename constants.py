import numpy as np
import MoleculeProp as mp
import stoichio_coeff_calculator as stoich
import GRI_data_calculator as gri

class Constants:
    
    R_g = 8.314                 # Universal gas constant, in J/mol-K
    T0 = 298.15                 # Reference Temperature, in K 
    
    def __init__(self, rxn_system, species, nu, thermo_coeff, rev_bool):
        self.rxn_system = rxn_system
        self.species = species
        self.nu = nu
        self.thermo_coeff = thermo_coeff
        self.rev_bool = rev_bool

    def diffusion_coeff(self):
        '''
        This function calculates all the possible binary diffusion coefficients
        between the species. If there are n species, nC2 combinations are possible. 
        All these nC2 combinations are calculated and then stored in a (n-1) X (n-1)
        matrix. These (n-1) X (n-1) matrix is returned, finally the bulk diffusivity 
        will be calculated using Wilke's method in the 'bif_dia_func' function.
        Ref: Diffusion Coefficients in Multicomponent Gas Mixtures, D.F. Fairbanks, 
             C.R. Wilke, Ind. Eng. Chem., 1952, 42(3), 471-475.
        '''
        #### Checking the inputs
        no_of_species = len(self.species)
        n = no_of_species - 1
        D_A = np.zeros([n, no_of_species])                  #### Stores the constant a in a*T**b
        D_B = np.zeros([n, no_of_species])                  #### Stores the constant b in a*T**b
        
        #### Calculating the binary diffusion coefficients
        for i in range(n):
            for j in range(i+1,n+1):
                coeff = mp.diffusivity(self.species[i], self.species[j])
                D_A[j-1, i] = coeff['A']
                D_B[j-1, i] = coeff['B']
            D_A[:i+1, i+1] = D_A[i, :i+1]
            D_B[:i+1, i+1] = D_B[i, :i+1]
        
        return D_A, D_B

    def get_all_coeff(self, return_flag= 'all'):
        '''
        This function segregates the coefficient and make a big fat matrix for Cp, H and S.
        If nothing is mentioned in return_flag, it will return all the matrices, 
        if return_flag == 'Cp', it will return only Cp values
        if return_flag == 'H', it will return only H values
        if return_flag == 'S', it will return only S values.
        '''

        #### Segregating the coefficients and making a big fat matrix out of it
        no_of_species = len(self.species)
        index = np.arange(no_of_species)
        
        Cp = np.zeros([2, no_of_species, 5])
        H = np.zeros([2, no_of_species, 6])
        S = np.zeros([2, no_of_species, 6])

        for e1, elem in zip(index, self.species):
            coeff = np.array(self.thermo_coeff[elem])
            Cp[0, e1, :] = self.R_g * coeff[:5]
            H[0, e1, :] = self.R_g * coeff[:6]
            S[0, e1, :] = self.R_g * np.r_[coeff[:5], coeff[6]]

            Cp[1, e1, :] = self.R_g * coeff[7:12]
            H[1, e1, :] = self.R_g * coeff[7:13]
            S[1, e1, :] = self.R_g * np.r_[coeff[7:12], coeff[13]]

        #### Shipping
        if return_flag == 'all':
            return Cp, H, S
        elif return_flag == 'Cp':
            return Cp
        elif return_flag == 'H':
            return H
        elif return_flag == 'S':
            return S
        else:
            raise Exception('Only fundamental thermodynamic coefficients are available. Try again!!!')

    def Cp_at_T(self, T):
        '''
        This function calculates the actual Cp of all the species at a specific
        temperature (T).
        '''
        
        #### Retrieving the compound specific Cp values
        Cp_all = self.get_all_coeff(return_flag= 'Cp')
        #index = self.species.index(compound)

        #Cp = np.r_['0, 2', Cp_all[0, index, :], Cp_all[1, index, :]]
        T_array = np.array([1, T, T**2, T**3, T**4])
        
        if (T >=300) and (T <= 1000):
            act_Cp = np.dot(Cp_all[1, :, :], T_array)
            return act_Cp
        elif (T > 1000) and (T <= 3000):
            act_Cp = np.dot(Cp_all[0, :, :], T_array)
            return act_Cp
        else:
#           print('warning: Temperature is beyond the physical range.')
           act_Cp = np.dot(Cp_all[0, :, :], T_array)
           return act_Cp
    
    def avg_Cp(self, T, output='average'):
        '''
        This function calculates the average Cp of all the  compounds in a given 
        range of temperature (T). It will take in a lower bound and an upper 
        bound in T and can calculate only for a single compound.
        '''
        #### Checking the inputs
        assert (type(T) != float) and (type(T) != int), ('A lower and upper '
                                           'bound of temperature is expected.')
        assert (len(T) == 2), 'Only the lower bound and upper bound will work!!'
        
        #### Unpacking the inputs
        if T[0] < T[1]:
            T_low = T[0]    
            T_upp = T[1]
        else:
            T_low = T[1]
            T_upp = T[0]
        
#        if (T_low < 200) or (T_upp > 3000):
#            print(T)
#            raise Exception('Temperature is beyond the physical range.')
        
        #### Retrieving the compound specific Cp values
        Cp = self.get_all_coeff(return_flag= 'Cp')
        
        #### Calculating the average Cp
        if (T_upp > 1000) and (T_low > 1000):
            T_full_arr = self.T_arr(T_upp, T_low)
            Cp_avg = np.dot(Cp[0, :, :], T_full_arr)
    
        elif (T_upp > 1000) and (T_low <= 1000):
            T_upp_arr = self.T_arr(T_upp, 1000)
            T_low_arr = self.T_arr(1000, T_low)
            Cp_avg = np.dot(Cp[0, :, :], T_upp_arr) + np.dot(Cp[1, :, :], T_low_arr)

        else:
            T_full_arr = self.T_arr(T_upp, T_low)
            Cp_avg = np.dot(Cp[1, :, :], T_full_arr)
        
        #### Shipping
        if output == 'average':
            return Cp_avg/(T_upp - T_low)
        elif output == 'integrate':
            return Cp_avg
        else:
            raise Exception ('Not a valid output ({}) requested!'.format(output))
            
    def del_H_reaction(self, T):
        '''
        This function calculates the heat of reaction of all the reactions
        involved in the process.
        '''
        #### Retrieving the coefficients
        H_all = self.get_all_coeff(return_flag= 'H')
        if T >= 1000:
            H = H_all[0, :, :]
        else:
            H = H_all[1, :, :]
        
        #### Standard heat of reaction @ T K
        std_T = np.r_[self.T_arr(T, 0), [[1]]]
        del_H_std = np.dot(H, std_T)
        del_H_rxn = np.dot(self.nu.T, del_H_std)
        
        return del_H_rxn

    def K_eq_mat_data(self):
        '''
        This function calculates the reaction specific matrix of del_H and del_S,
        so that Temperature dependent K_eq values can be easily calculated.
        '''

        #### All thermodynamic coefficients
        Cp, H, S = self.get_all_coeff()
        no_of_react = self.nu.shape[1]

        del_H1 = np.dot(self.nu.T, H[0, :, :])
        del_H2 = np.dot(self.nu.T, H[1, :, :])
        del_H_all = np.r_[del_H1, del_H2]
        del_H = del_H_all.reshape(2, no_of_react, 6)

        del_S1 = np.dot(self.nu.T, S[0, :, :])
        del_S2 = np.dot(self.nu.T, S[1, :, :])
        del_S_all = np.r_[del_S1, del_S2]
        del_S = del_S_all.reshape(2, no_of_react, 6)

        return del_H, del_S
    
    @staticmethod
    def T_arr(T, T0):
    
        T_list = [T-T0, 1/2*(pow(T,2) - pow(T0,2)), 1/3*(pow(T,3) - pow(T0,3)), 1/4*(pow(T,4) - pow(T0,4)),\
              1/5*(pow(T,5) - pow(T0,5))]
        T_mat = np.array(T_list).reshape(5, 1)
    
        return T_mat
        
    
        
def fixed_parameters(rxn_system, param= None):
    '''
    This function defines the constant parameters like hydraulic radius,
    C_p values etc. Once called it returns a dictionary with all the 
    names of the constants as keys and their corresponding values.
    All values are in SI units. 
    '''
    'A is the fuel and B is Oxygen'
    
    #### Identifying the system
    species, nu, rev_bool = stoich.main_func(rxn_system)
    thermo_coeff = gri.gri_data(species)
    
    #### Instantiating a variable from Constants class
    react_sys = Constants(rxn_system, species, nu, thermo_coeff, rev_bool)

    #### Calculating the binary diffusion coeffcients
    D_bin_a, D_bin_b = react_sys.diffusion_coeff()

    #### Storing del_H and del_S matrix
    all_del_H, all_del_S = react_sys.K_eq_mat_data()

    #### Calculating the thermal Diffusivities
    #### alpha_f is taken as that of air for the time being, we will incorporate
    #### the correct alpha_f along with the mixture formula later
    #### Ref : Imran Alam
    alpha_f = 9.8e-10
    R_omega_w = 90e-06 # Monolith wall thickness, in m   
    
    #### Washcoat and wall physical parameters (LATER)
    #### Ref : Dadi, Pritpal, Pankaj Kumar, Tian Gu
    eps_wc = 0.4
    tort = 4
    Cp_w = 1000
    rho_w = 2000
    k_w = 1.5

    #### Creating the final dictionary
    names = ['species', 'alpha_f',  'D_AB', 'nu', 'all_del_H', 'all_del_S', 
             'rev_bool', 'R_omega_w', 'eps_wc', 'tortuosity',
             'Cp_w', 'rho_w', 'k_w']
    
    values = [species, alpha_f, [D_bin_a, D_bin_b], nu, all_del_H, all_del_S, 
              rev_bool, R_omega_w, eps_wc, tort,
              Cp_w, rho_w, k_w]
    
    name_dict = dict(zip(names, values))
    
    if param == None:
        return name_dict, react_sys
    else:
        # Try to remember this way of constructing dictionaries from another 
        # dictionary
        param_name = {names : name_dict[names] for names in param}
        return param_name, react_sys

if __name__ == '__main__':
    names, react_sys = fixed_parameters('OCM_Zhe_Sun_homo.txt')
    print(names['species'])
    print(react_sys.del_H_reaction(1300))
#    print(names['D_AB'])
#    print(names['nu'])]
#    print(names['all_del_S'])
#    print(names['del_H_rxn'])
#    D_AB = names['D_AB']
#    A = D_AB[0]
#    B = D_AB[1]
#    
##    D = A * 300 ** B
##    print(D)
#    print(A)
#    print(B)
    
