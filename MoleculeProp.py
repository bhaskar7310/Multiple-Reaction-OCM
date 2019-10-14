#### This module calculates and stores all the important properties of a particular molecule
#### like its molecular weight, mass diffusivity and thermal diffusivity constants.
#### Particulary it stores the transport data of each molecule.

import numpy as np
from math import pow
import GRI_data_calculator as gri
from scipy.optimize import curve_fit


####----------------------------------------------------------------------------------------
####                     Diffusivity Calculation using Chapman-Enskog
####----------------------------------------------------------------------------------------
def get_data():
    '''Lennard-Jones potential parameters for important components.'''
    
    species_name = ['H2', 'Air', 'N2', 'O2', 'H2O',\
                     'CO', 'CO2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'C3H8']
    
    mol_weight = [2.016,  28.964, 28.013, 31.999, 18.0155,\
                  28.010, 44.010, 16.04, 26.04, 28.05, 30.07, 44.10]
    
    sigma = [2.915, 3.617, 3.667, 3.433, 2.641,\
             3.590, 3.996, 3.780, 4.114, 4.228, 4.328, 4.934]
    
    epsilon_k = [38.0, 97.0, 99.8, 113, 110, 809.1,\
                 190, 154, 212, 216, 232, 273]
#    sigma = [2.827, 3.771, 3.798, 3.467, 2.641,\
#             3.690, 3.941, 3.758, 4.033, 4.163, 4.443, 5.118]
#    
#    epsilon_k = [59.7, 78.6, 71.4, 106.7, 809.1,\
#                 91.7, 195.2, 148.6, 231.8, 224.7, 215.7, 237.1]
    
    mol_weight_dict = dict(zip(species_name, mol_weight))
    
    sigma_dict = dict(zip(species_name, sigma))
    
    epsilon_k_dict = dict(zip(species_name, epsilon_k))
    
    return mol_weight_dict, sigma_dict, epsilon_k_dict
    

def get_K_eps_T():
    '''Pre-defined points to calculate the Collision Integrals.'''

    #### K_eps/T values
    arr1 = np.arange(1, 5, 0.5)
    arr2 = np.arange(5, 10, 1.0)
    arr3 = np.arange(10, 20, 2)
    arr4 = np.arange(20, 41, 10)
    
    return np.r_[arr1, arr2, arr3, arr4]


def sigmaDiffFormula(T_star):
    '''
    Curve Fitted formula for deriving the collision integrals. 
    Ref: P.D. Neufeld, A.R. Janzen and R.A.Aziz, J.Chem.Phys., 57, 1100-1102 (1972).
    '''
    return 1.06036/T_star ** 0.15160 + 0.19300/np.exp(0.47635 * T_star) + \
            1.03587/np.exp(1.52996 * T_star) + 1.76474/np.exp(3.89411 * T_star)


def sigmaDiffTable(T_star):
    '''
    Collision Integrals for use with Lennard-Jones (6-12) Potential for the prediction of
    diffusivities of Gases at low densities.
    '''
    #### Retrieving the pre-defined points  
    k_eps_T = get_K_eps_T()
    
    #### Sigma Diff values
    sigmaArr1 = [1.440, 1.199, 1.075, 1.0006, 0.95, 0.9131, 0.8845, 0.8617]
    sigmaArr2 = [0.8428, 0.8129, 0.7898, 0.7711, 0.7555]
    sigmaArr3 = [0.7422, 0.7202, 0.7025, 0.6878, 0.6751]
    sigmaArr4 = [0.6640, 0.6414, 0.6235, 0.6088, 0.5964]
    
    sigma = np.r_[sigmaArr1, sigmaArr2, sigmaArr3, sigmaArr4]
    
    #### Creating the main dictionary
    sigma_Dict = {key : value for key, value in zip(k_eps_T, sigma)}

    sigma_actual = []
    
    for elem in T_star:
        sigma_actual.append(sigma_Dict.get(elem))
        
    sigma_array = np.array(sigma_actual)

    return sigma_array


def func(x, a, b):
    '''Function to be curve fitted if Exponent== None.'''
    return a * x ** b


def func_17(x, a):
    '''Function to be curve fitted if Exponent == 1.7.'''
    return a * x**1.7


def func_175(x, a):
    '''Function to be curve fitted if Exponent == 1.75.'''
    return a * x**1.75         


def diffusivity_calc_Chapman(A, B, data= 'table', exponent= None):
    '''
    This function calculates the binary diffusion coefficient of A and B 
    using the Chapman-Enskog formulation. It calculates the coefficient and
    exponents in SI units
    '''
    
    M_w, sigma, eps_k = get_data()
    
    M_A = M_w.get(A, None)
    
    if M_A == None:
        raise Exception ('The species is not in our dictionary.')
    
    sigma_A = sigma.get(A)
    eps_k_A = eps_k.get(A)
    
    if A == B:
        sigma_AB = sigma_A
        eps_k_AB = eps_k_A
        M_B = M_A
    
    else:
        sigma_B = sigma.get(B)
        eps_k_B = eps_k.get(B)
        M_B = M_w.get(B)

        sigma_AB = (sigma_A + sigma_B)/2
        eps_k_AB = np.sqrt(eps_k_A * eps_k_B)

    preDiff = 0.0018583 * np.sqrt(1/M_A + 1/M_B) / (sigma_AB ** 2) * 1e-04
    
    #### Retrieving the pre-defined points    
    k_eps_T = get_K_eps_T()
    
    T_points = eps_k_AB * k_eps_T
    
    #### Actual Data points and the corresponding k_eps_T values
    k_eps_actual = k_eps_T[(T_points >= 300)*(T_points <= 2000)]
    T_actual = T_points[(T_points >= 300)*(T_points <= 2000)]
    
    
    #### Retrieving Collision Integral data from table or formula (based on input)  
    if data == 'formula':
        sigma_diff = sigmaDiffFormula(k_eps_actual)
    else:
        sigma_diff = sigmaDiffTable(k_eps_actual) 
    
    #### Getting the Y values
    D_AB = preDiff * T_actual ** 1.5 / sigma_diff
    
    #### Mathematically fitting the data
    if exponent == 1.7:
        popt, pcov = curve_fit(func_17, T_actual, D_AB)
        return {'A': popt[0], 'B' : 1.7} 
    
    elif exponent == 1.75:
        popt, pcov = curve_fit(func_175, T_actual, D_AB)
        return {'A': popt[0], 'B' : 1.75}
    else:
        popt, pcov = curve_fit(func, T_actual, D_AB)
        return {'A': popt[0], 'B' : popt[1]} 


####----------------------------------------------------------------------------------------
####            Diffusivity calculation using Fuller-Schettler-Giddings formula
####----------------------------------------------------------------------------------------
def atom_volume_calc(species):
    '''
    This function takes in the name of the species (elem) as input, calculates its molecular 
    weight and atomic volume and returns these two values. If elem == 'Air' it just takes in 
    the values from the pre-defined dictionary and returns those values.
    '''
    #### Dictionaries containg the respective molecular weight of elements
    molecular_weights = {'C': 12, 'O': 16, 'H': 1, 'Air' : 28.964}
    
    #### Dictionaries containing the respective atomic volumes
    atomic_vol = {'C' : 16.5, 'H' : 1.98, 'O' : 5.48, 'H2' : 7.07, \
              'O2' : 16.6, 'Air' : 20.1, \
              'CO' : 18.9, 'CO2' : 26.9, 'H2O' : 12.7}
    
    if species == 'Air':
        M_B = molecular_weights['Air']
        nu_B = atomic_vol['Air']
        return M_B, nu_B
    else:

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
    
        mol_weight = 0
        atom_vol = 0
    
        #### Calculation of molecular weight and volume
        for e1, e2 in zip(elements, elements_count):
            mol_weight += molecular_weights[e1]*e2
            atom_vol += atomic_vol[e1]*e2
                
        if species in atomic_vol:
            atom_vol_species = atomic_vol[species]
        else:
            atom_vol_species = atom_vol
    
        #### Shipping
        return mol_weight, atom_vol_species


def diffusivity_calc_Fuller(elemA, elemB):
    '''
    This function calculates the binary diffusion coefficient of A and B 
    using the 'Fuller-Schettler-Giddings' formulation. It calculates the 
    coefficient and exponents in SI units
    '''
    #### Calculating the diffusion coefficient
    Mol_A, nu_A = atom_volume_calc(elemA)
    Mol_B, nu_B = atom_volume_calc(elemB)
    diff = 1e-07 * ((1/Mol_A + 1/Mol_B) ** 0.5)/ \
            (nu_A**(1/3) + nu_B**(1/3))**2
    
    #### Shipping
    return diff

####----------------------------------------------------------------------------------------
####                        Diffusivity Calculation, Main Function
####----------------------------------------------------------------------------------------
def diffusivity(elemA, elemB, formula= 'Fuller-Schettler-Giddings', data= 'table', \
                    exponent= None):
    '''
    The binary mass_diffusivites of low density gases are calculated using this function.
    There are two ways to calculate it, one is the 'Fuller-Schettler-Giddings' formula 
    which is pretty straightforward, and another is the 'Chapman-Enskog' formula. 
    The user decides the way the diffusivities are to be calculated. If 'Chapman-Enskog' 
    is chosen the collison integrals and epsilons can be either calculated from 'table' or 
    'formula'. That is specified by the data variable. And 'Chapman-Enskog' process also 
    needs the user to specify the exponent of T. exponent can take values like 'None', 1.7
    or 1.75. 
    '''
    #### Diffusivity calculation using Chapman-Enskog formula
    if formula == 'Chapman-Enskog':
        diff_const = diffusivity_calc_Chapman(elemA, elemB, data, exponent) 
        return diff_const
    elif formula == 'Fuller-Schettler-Giddings':
        diff_const_a = diffusivity_calc_Fuller(elemA, elemB)
        return {'A' : diff_const_a, 'B' : 1.75}


####----------------------------------------------------------------------------------------
####                        Thermal Diffusivity Calculations
####----------------------------------------------------------------------------------------
def sigmaMuFormula(T_star):
    '''
    Curve Fitted formula for deriving the collision integrals. 
    Ref: P.D. Neufeld, A.R. Janzen and R.A.Aziz, J.Chem.Phys., 57, 1100-1102 (1972).
    '''
    return 1.16145/T_star ** 0.14874 + 0.52487/np.exp(0.77320*T_star) + \
            2.16178/np.exp(2.43787*T_star)


def sigmaMuTable(T_star):
    '''
    Collision Integrals for use with Lennard-Jones (6-12) Potential for the prediction of
    thermal diffusivities or viscosities of Gases at low densities.
    '''
    #### Retrieving the pre-defined points 
    k_eps_T = get_K_eps_T()
    
    #### Sigma Mu Values
    sigmaArr1 = [1.593, 1.315, 1.176, 1.0933, 1.0388, 0.9996, 0.9699, 0.9462]
    sigmaArr2 = [0.9268, 0.8962, 0.8727, 0.8538, 0.8380]
    sigmaArr3 = [0.8244, 0.8018, 0.7836, 0.7683, 0.7552]
    sigmaArr4 = [0.7436, 0.7198, 0.7010, 0.6854, 0.6723]
    
    sigma_mu = np.r_[sigmaArr1, sigmaArr2, sigmaArr3, sigmaArr4]
    
    #### Creating the main dictionary
    sigma_Dict = {key : value for key, value in zip(k_eps_T, sigma_mu)}

    sigma_actual = []
    
    for elem in T_star:
        sigma_actual.append(sigma_Dict.get(elem))
        
    sigma_array = np.array(sigma_actual)

    return sigma_array


def viscosity(A, data= 'table', fitdata= False, exponent= None):
    '''
    This function will just calculate the viscosity at different temperatures.
    The unit of return value of viscosity is gm/cm-s
    '''
    
    M_w, sigma, eps_k = get_data()

    M_A = M_w.get(A, None)
    
    if M_A == None:
        raise Exception ('The species is not in our dictionary.')
    
    sigma_A = sigma.get(A)
    eps_k_A = eps_k.get(A)     
    
    preMu = 2.6693e-05 * np.sqrt(M_A)/sigma_A ** 2
    
    #### Pre-defined points
    arr1 = np.arange(1, 5, 0.5)
    arr2 = np.arange(5, 10, 1.0)
    arr3 = np.arange(10, 20, 2)
    arr4 = np.arange(20, 41, 10)
    
    k_eps_T = np.r_[arr1, arr2, arr3, arr4]
    
    T_points = eps_k_A * k_eps_T
    
    #### Actual Data points and the corresponding k_eps_T values
    k_eps_actual = k_eps_T[(T_points >= 300)*(T_points <= 2000)]
    T_actual = T_points[(T_points >= 300)*(T_points <= 2000)]
    
    #### Retrieving Collision Integral data from table or formula (based on input)  
    if data == 'formula':
        sigma_mu = sigmaMuFormula(k_eps_actual)
    else:
        sigma_mu = sigmaMuTable(k_eps_actual)
   
    #### Getting the Y values
    mu = preMu * T_actual ** 0.5 / sigma_mu
    
    #### Mathematically fitting the data
    if fitdata == False:
        return M_A, T_actual, mu
    
    else:       
        if exponent == 1.7:
            popt, pcov = curve_fit(func_17, T_actual, mu)
            return {'A': popt[0], 'B' : 1.7} 
        
        elif exponent == 1.75:
            popt, pcov = curve_fit(func_175, T_actual, mu)
            return {'A': popt[0], 'B' : 1.75}
        else:
            popt, pcov = curve_fit(func, T_actual, mu)
            return {'A': popt[0], 'B' : popt[1]}


def Cp_at_T(Coeff, T):
    
    T_array = np.array([1, T, pow(T, 2), pow(T, 3), pow(T, 4)])

    if (T >=300) and (T <= 1000):
        act_Cp = np.dot(Coeff[1, :], T_array)
        return act_Cp
    elif (T > 1000) and (T <= 3000):
        act_Cp = np.dot(Coeff[0, :], T_array)
        return act_Cp
    else:
        raise Exception('Temperature is beyond the physical range.')


def conductivity(A, data= 'table', unit= 'CGS'):
    '''
    Conductivity of species A is calculated using the Chapman-Enskog Kinetic
    theory of gasses and modified Eucken Models.
    The basic calculations are done in CGS units, so the return value is by
    default in CGS units, (cal/gm-s-K). If SI units are specifically requested 
    by the user then only K will be returned in terms of W/m-K.
    '''
    
    R_g = 1.987 #### Unit = Cal/mol-K
    gamma = 1.4 ### Valid only for dry air, Cp/Cv = gamma = 1.4
    
    M_A, T_actual, mu = viscosity(A, data)

    #### Cp Calculation of A, if A == Air, Dry Air composition is taken as
    #### 0.78084 N2, 0.20946 O2, 0.0094 Ar and 0.0004 CO2
    if A == 'Air':
        species = ['N2', 'O2']    
    else:
        species = [A]
        
    n = len(species)
    thermo_coeff = gri.gri_data(species)
    Cp_species = np.zeros([n, len(T_actual)])

    index = 0
    for elem in species:    
        coeff_values = np.array(thermo_coeff[elem])
        Cp_coeff = np.zeros([2, 5])    # Pre-allocation of C_p matrix
    
        #### Unpacking the coeff data
        Cp_coeff[0, :] = R_g * coeff_values[:5]    # Above 1000 K data
        Cp_coeff[1, :] = R_g * coeff_values[7:12]  # Below 1000 K data

        #### Calculating Cp at the required temperature points
        Cp = []
        
        for Temp in T_actual:
            Cp.append(Cp_at_T(Cp_coeff, Temp))

        for i in range(len(Cp)):
            Cp_species[index, i] = Cp[i]
        index += 1

    if A == 'Air':
        Cp_arr = Cp_species[0, :] * 0.79 + Cp_species[1, :] * 0.21
    else:
        Cp_arr = Cp_species[0, :]
 
    #### The final K values
    K = (4.47 + Cp_arr/gamma) * mu/M_A
    
    if unit == 'SI':
        return K * 4.184e02, Cp * 4.184, T_actual
    else:
        return K, Cp, T_actual
    
def thermal_diffusivity(A, data= 'table', exponent= None):    
    ''' 
    Thermal diffusivity, alpha = K/(rho * Cp), unit = cm**2/s
    The return value though will be converted to m**2/s
    Considering idea gas conditions rho = P/RT, unit = mol/gm-cm3
    Cp will be in CGS units, ie. Cal/mol-K
    K will also be in CGS units, ie. Cal/gm-cm-s
    Pressure P is taken as 1 atm
    '''
    R_g = 82.057 #### Unit = cm**3 atm/mol-K
    K, Cp, T = conductivity(A, data)
    rho = 1/(R_g * T)
    
    alpha = K/(rho * Cp)

    #### Mathematically Fitting the data
    if exponent == 1.7:
        popt, pcov = curve_fit(func_17, T, alpha)
        return {'A' : popt[0] * 1e-04, 'B' : 1.7}

    elif exponent == 1.75:
        popt, pcov = curve_fit(func_175, T, alpha)
        return {'A' : popt[0] * 1e-04, 'B' : 1.75}

    else:
        popt, pcov = curve_fit(func, T, alpha)
        return {'A': popt[0] * 1e-04, 'B' : popt[1]} 
        

if __name__ == '__main__':
    
    #### Mass Diffusivities
    elemA = 'H2'
    elemB = 'Air'
    formula1 = 'Fuller-Schettler-Giddings'
    formula2 = 'Chapman-Enskog'
    const = diffusivity(elemA, elemB, formula1, data= 'table', exponent= 1.75)
    print(const)
#    T = 352.3
#    D_AB = const['A'] * T ** const['B']
#    print(D_AB)
    
#    #### Thermal Diffusivities
#    elem = 'CO'
#    alpha_f = thermal_diffusivity(elem, data= 'table', exponent= 1.75)
#    print(alpha_f)
    
