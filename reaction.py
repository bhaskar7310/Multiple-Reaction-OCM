'''
This module will read the rate equation provided by the user in 
the [reaction_system]_rates.txt file and then convert it to a 
mathematical form, which will be called for different values of 
Temperature and mole fractions to calculate the actual rate.

Now one important point to note is that this calculation file is 
entirely dependent on the validity of the [name]_rates.txt file. 
If there is any manual input error, rest of the calculations will then 
be effected. So one drawback to be dealt with in future is handling 
user input errors.

We also need some getters and setters here, we need to implement 
the principle of 'INFORMATION HIDING'.

There can be different ways of writing the rate expression based on 
the mechanism. It can be Langmuir-Hinshelwood type or Power-Law type. 
Even in Power Law model, the reverse rate expressions
may not be available and can be calculated from thermodynamics. 
Our code can tackle all these different types of rate expressions. 

For LNHW or ER reaction mechanisms, this code will only work if the 
reaction mechanism is single site and 
the rate-determining step is irreversible.
'''

import os
import numpy as np
import warnings
warnings.filterwarnings("error")

class Reaction:
    
    R_g = 8.314
    eps = 1e-06
    
    def __init__(self, rate_type, k_f, Ea_f, species_f, n_f, k_r, 
                 Ea_r, species_r, n_r, power, del_H, del_S):
        self.rate_type = rate_type   # Type of the reaction, PL/LNHW
        
        self.k_f = k_f               # Fwd reaction pre-exp factors
        self.Ea_f = Ea_f             # Fwd reaction activation energies
        self.species_f = species_f   # Fwd reaction species
        self.n_f = n_f               # Fwd reaction exponents

        self.k_r = k_r               # Rev reaction pre-exp factors
        self.Ea_r = Ea_r             # Rev reaction activation (or ads) enegies 
        self.species_r = species_r   # So and So
        self.n_r = n_r               # So and So
        self.power = power

        self.del_H = del_H           # Polynomial coefficients 
                                     # for calculating heat of reactions
        self.del_S = del_S           # Polynomial coefficient 
                                     # for calculating change in entropy

    def calc_PL_rate(self, conc_dict, basis, k, Ea, spec, n, T, P):
        '''
        This function calculates PL type rates which are specifically 
        in the form of
        rate = k * exp(-Ea/RT) * spec['A']**n['A'] * spec['B']**n['B']
        '''
        
        rate_const = k * np.exp(-Ea/(self.R_g * T))  # Raw reaction rate
        
        #### Correction based on the basis of the reaction rates
        if basis == 'mole_fraction':
            rate_const *= (self.R_g * T/(101325 * P)) ** (sum(n) - 1)
        elif basis == 'pressure' or basis == 'partial_pressure':
            rate_const *= (self.R_g * T) ** (sum(n))
        else:
            if basis != 'concentration':
                raise Exception ('Basis of the reaction rates (whether '
                                 'concentration, pressure, or '
                                 'mole fraction) is not understood')
        rate = rate_const
        
        if spec[0] != None:
            for e1, e2 in zip(spec, n):
                if e2 >= 1:
                    rate *= conc_dict[e1] ** e2
                else:
                    
                    rate *= (conc_dict[e1] ** (1+e2))/(conc_dict[e1] + self.eps)
        

        return rate_const, rate
        
    def act_rate(self, Y, species, basis, P=1):
        '''
        This function calculates the actual rate calculated at a 
        specific temperature and mole fractions.
        '''
        #### Unpacking the input list Y
        Ylist = Y[:-1]
        T = Y[-1]

        #### Initialization
        rate_fwd_const = []    # List of the forward rate constants 
        rate_fwd_list = []     # List of the actual forward reaction rates
       
        rate_rev_const = []    # List of the reverse rate constants
        rate_rev_list = []     # List ofof the actual reverse reaction rates
        
        act_rate_list = []     # List of actual rates, 
                               # (fwd - rev) for PL and (num/den) for LNHW
        
        #### Manipulating the inputs
        conc_dict = dict(zip(species, Ylist))

        #### Calculating the forward rate (for PL type) 
        ### or the numerator (for LNHW type)
        for k, Ea, spec, n in zip(self.k_f, self.Ea_f, 
                                  self.species_f, self.n_f):
            
            fwd_const, fwd_rate = self.calc_PL_rate(conc_dict, basis, k, Ea, 
                                                    spec, n, T, P)

            rate_fwd_const.append(fwd_const)
            rate_fwd_list.append(fwd_rate)


        #### Calculating the reverse rate (for PL type) 
        #### or the denominator (for LNHW type)
        index = 0                   # Keeps a count of the reactions

        for rate_type, k, Ea, spec, n in zip(self.rate_type, self.k_r, 
                                             self.Ea_r, self.species_r, 
                                             self.n_r):
            
            #### Power Law type reverse reaction rates
            if rate_type == 'PL':
                
                #### The reaction rates are provided
                if k > 0:
                    rev_const, rev_rate = self.calc_PL_rate(conc_dict, basis, 
                                                            k, Ea, spec, n, 
                                                            T, P)
                    rate_rev_const.append(rev_const)
                    rate_rev_list.append(rev_rate)
                
                #### Irreversible reaction rates    
                elif k == 0:
                    rate_rev_list.append(0)
                    rate_rev_const.append(0)

                #### Reaction rate constant is calculated from thermodynamics
                elif k == -1:
                    del_H = self.del_H[:, index, :]
                    del_S = self.del_S[:, index, :]
                
                    H_T_dependence = np.array([T, T**2/2, T**3/3, 
                                               T**4/4, T**5/5, 1]).reshape(6, 1)
                    S_T_dependence = np.array([np.log(T), T, T**2/2, T**3/3, 
                                               T**4/4, 1]).reshape(6, 1)

                    if T > 1000:
                        del_H_T = np.dot(del_H[0, :], H_T_dependence)
                        del_S_T = np.dot(del_S[0, :], S_T_dependence)

                    else:
                        del_H_T = np.dot(del_H[1, :], H_T_dependence)
                        del_S_T = np.dot(del_S[1, :], S_T_dependence)

                    del_G_T = del_H_T - T * del_S_T
                    K_eq_T = np.exp(-del_G_T/(self.R_g * T))

                    rate_rev = rate_fwd_const[index]/K_eq_T[0]
                    rate_rev_const.append(rate_rev)
                
                    if spec != None:
                        for e1, e2 in zip(spec, n):
                            if e2 >= 1:
                                rate_rev *= conc_dict[e1] ** e2
                            else:
                                rate_rev *= (conc_dict[e1] ** (1+e2))\
                                          / (conc_dict[e1] + self.eps)
                    rate_rev_list.append(rate_rev)
                
                act_rate_list.append(rate_fwd_list[index] - rate_rev_list[index])
                index += 1    
            
            elif rate_type == 'LNHW':
                deno_sum = 1
                for k_1, Ea_1, spec_1, n_1 in zip(k, Ea, spec, n):
                    deno_const, deno_rate = self.calc_PL_rate(conc_dict, 
                                            basis, k_1, Ea_1, spec_1, n_1, T, P)
                    if basis == 'mole_fraction':
                        deno_rate *= (self.R_g * T)/(101325 * P) 
                    deno_sum += deno_rate
                rate_rev_list.append(deno_sum**self.power[index])
                act_rate_list.append(rate_fwd_list[index]/rate_rev_list[index])
                index += 1

            #### Other options
            else: 
                raise Exception('We are stuck here, work is in progress, '
                                'we will calculate this soon!!!')

        #### Shipping
        actual_rate = np.array(act_rate_list)
      
        return actual_rate

    def reaction_jacobian(self, Y, fvec, *args):
        '''
        This function calculates the derivatives of reaction 
        rates with respect to concentration and temperature and 
        returns the jacobian matrix.
        '''
        #### Checking the inputs
        no_of_cols = Y.shape[0]
        no_of_reaction = fvec.shape[0]
    
        #### Specifying variables
        column = 0.
        eps = 1e-06
        J = np.zeros([no_of_reaction, no_of_cols], dtype=float)
        Y_pos = Y.copy()
        
        for i in range(no_of_cols):
            h = eps * Y[i]
            if h == 0:
                h = eps
            Y_pos[i] = Y[i] + h
                
            column = (self.act_rate(Y_pos, *args) - fvec)/h
            J[:, i] = column[:]
            Y_pos[i] = Y_pos[i] - h
            
        #### Packaging and Shipping
        J_conc = J[:, :-1]
        J_T = J[:, -1]
        return J_conc, J_T;


def get_rate_type(react_ID):
    '''
    This function takes in the reaction ID (which is a string) 
    and looks for consecutive upper case letters and then returns 
    joining all those upper case consecutive letters 
    (which will represent the reaction rate type).
    
    Possible types of rate expressions are:
    LNHW : Langmuir-HinshelWood Hougen Watson
    ER : Eley-Riedel
    MVK : Mars van Krevelen
    PL : Power Law (Arrhenius type)
    '''
    #### A not-so-required assertion statement
    assert isinstance(react_ID, str), 'Error!!! The input has to be a string!!!'

    index = 0
    new_index = 0
    possible_rate_type = ['LNHW', 'ER', 'MVK', 'PL']

    #### Iterating over the react_ID string
    for elem in react_ID:
        if elem.isupper():
            new_react_ID = react_ID[index:]
        else:
            index += 1

    for new_elem in new_react_ID:
        if new_elem.isupper():
            new_index += 1
        else:
            break
    
    rate_type = new_react_ID[:new_index]
    
    if rate_type not in possible_rate_type:
        raise Exception('Reaction rate type is not understood.')
    else:
        return rate_type


def get_rxn_no(react_ID):
    '''
    This function takes in the reaction ID (which is string) 
    looks for consecutive numbers and then returns
    that number. That number represents the reaction number.
    '''
    index = 0                  
    num_str_list = []         
    new_index = 0              
    
    #### Iterating through the react_ID
    for elem in react_ID:
        try:
            int(elem)
        except ValueError:
            index += 1
        else:
            num_str_list.append(elem)
            index += 1
            new_ID = react_ID[index:]
            break
    
    for new_elem in new_ID:
        try:
            int(new_elem)
        except ValueError:
            break
        else:
            num_str_list.append(new_elem)
            new_index += 1

    rxn_no_str = ''.join(num_str_list)
    rxn_no = int(rxn_no_str)

    rate_type = get_rate_type(new_ID[new_index:])

    return rxn_no, rate_type


def segregate_PL(rate_exp):
    '''
    This function identifies the different components in the 
    Power Law (Arrhenius) type rate expression.
    
    By different components we mean:
    rate_const : Pre-exponential
    activation_energy: Activation energy
    n_react : Exponents of species
    species_reactive : Active species involved in the rate expression
    '''

    #### Variable Initialization
    activation_energy = 0
    n_react = 0
    species_react = None
    
    terms = rate_exp.split('*')

    try:
        #### Pre-exponential factor
        rate_const = float(terms[0])
    except ValueError:
        rate_const = -1
    else:
        #### Activation Energy
        if (rate_const !=0):
            second_word = terms[1]
            
            #### Checking for the sign inside the exponential
            if second_word[4] == '-':
                index = 5
            else:
                index = 4
            
            num_start = index
            for elem in second_word[num_start+1:]:
                index += 1
                if elem == '/':
                    break
            
            activation_energy = float(second_word[num_start:index])
            if (num_start == 4):
                activation_energy *= -1
    finally:
        #### Active Species in rate equation
        if (rate_const != 0):
            species = terms[2:]
            species_react = []
            n_react = []
            for elem in species:
                if elem != '':
                    try:
                        num = float(elem)
                        n_react.append(num)
                    except:
                        species_react.append(elem[1:-1])

    return rate_const, activation_energy, n_react, species_react


def segregate_LNHW(deno_exp):
    '''
    This function identifies the different expressions in the 
    denominator of a LNHW (or Eley Ridel) type rate 
    expression with a single site mechanism. 
    If the mechanism is dual site, this code will fail. 
    Neither can it handle rate expressions where the rate-
    determining step is reversible. For all other single site reaction 
    mechanisms with irreversible rate 
    determining step, this code will work fine.

    General expression of the denominator:
    deno_exp = (1 + K1*exp(-del_H1_/RT) * [A]**n_A * [B]**n_B 
             + K_2*exp(-del_H2/RT) * [A]**n2_A * [B]**n2)**2
    '''
    
    #### List Initialization
    K_ads = []
    del_H_ads = []
    n_ads = []
    species_ads = []
    power_list = []

    #### Getting the power to which the denominator 
    #### of LNHW type reaction rate is raised
    n = len(deno_exp)
    for i in range(-1, -(n+1), -1):
        try:
            int(deno_exp[i])
        except ValueError:
            if deno_exp[i] == '.':
                power_list.append(deno_exp[i])
            elif deno_exp[i:i-2:-1] == '**':
                i -= 2
                break
            elif deno_exp[i] == ')':
                break
            else:
                raise Exception ('The expression of denominator given by {} '
                                 'cannot be recognised'.format(deno_exp))
        else:
            power_list.append(deno_exp[i])

    if power_list:
        power_str = ''.join(power_list) # Making a single string from the list
        power = float(power_str[::-1])  # Making a floating point number 
                                        # from the reversed list
    else:
        power = 1

    #### Iterating through the true expression 
    #### (devoid of the power it is raised to)
    expression = deno_exp[1:i]
    terms = expression.split('+')

    for elem in terms:
        try:
            float(elem)
        except ValueError:
            rate_const, \
            adsorption_enthalpy, \
            n_react, species_react = segregate_PL(elem)
            
            K_ads.append(rate_const)
            del_H_ads.append(adsorption_enthalpy)
            n_ads.append(n_react)
            species_ads.append(species_react)
            
        else:
            term_num = float(elem)
    
    return term_num, K_ads, del_H_ads, n_ads, species_ads, power


def instantiate(react_system, const, catalyst= None):
    '''
    This funtion reads the kinetic data file and then stores the 
    rate expressions of homogeneous and catalytic rate expressions.
    '''
    #### Finding the correct file:
    filename, ext = os.path.splitext(react_system)
    
    if catalyst:
        newFileName = filename + '_' + catalyst.lower() + '_rates' + ext
    else:
        newFileName = filename + '_rates' + ext
        
    if not os.path.isfile(newFileName):
        raise FileNotFoundError ('The rate expression of {} system is '
                                 'not provided.'.format(filename))

    #### Initialization of lists
    k_f = []
    Ea_f = []
    species_f = []
    n_f = []
    
    k_r = []
    Ea_r = []
    species_r = []
    n_r = []
    deno_raised_power = []

    cat_index = []
    homo_index = []
    
    nu_homo_index = []
    nu_cat_index = []
    
    rate_type_list = []
    
    #### Reading the rate expression file and storing the required data
    with open(newFileName, 'r') as infile:
        infile.readline()           # Reading customary first line
        count = 0                   # Counting the number of lines
        react_count = 0             # Reaction counting no.
        
        for line in infile.readlines():
            count += 1
            if (count % 2) != 0:
                react_count += 1
                forward = True
                reverse = False
            else:
                forward = False
                reverse = True
                
            line_new = line.replace(' ', '').replace('\n', '')
            id_expression = line_new.split('=')
            
            #### Reaction identifier (whether Catalytic or Homogeneous)
            react_ID = id_expression[0]
            rate_exp = id_expression[1] 

            if (count % 2) != 0:
                rxn_no, rate_type = get_rxn_no(react_ID)
                rate_type_list.append(rate_type)
                
                if react_ID[2] == 'c':
                    nu_cat_index.append(rxn_no - 1)
                    cat_index.append(react_count - 1)
                
                elif react_ID[2] == 'h':
                    nu_homo_index.append(rxn_no - 1)
                    homo_index.append(react_count - 1)
        
            #### Power-law (PL) type reactions
            if (rate_type == 'PL') or (forward == True):              
                rate_const, \
                activation_energy, \
                n_react, species_react = segregate_PL(rate_exp)
            
            #### Langmuir-Hinshelwood (LNHW) type reactions
            elif (rate_type == 'LNHW') and (reverse == True):
                term_num, \
                K_ads, \
                del_H_ads, \
                n_ads, species_ads, power = segregate_LNHW(rate_exp)

            #### Any other type of reaction rate expressions
            else:
                raise Exception('We are still working on it. Come back later.')

            #### Storing all the rate expression information    
            if (forward == True):
                k_f.append(rate_const)
                Ea_f.append(activation_energy)
                species_f.append(species_react)
                n_f.append(n_react)
                
            elif (reverse == True) and (rate_type == 'PL'):
                deno_raised_power.append(0)

                if (rate_const > 0):
                    k_r.append(rate_const)
                    Ea_r.append(activation_energy)
                    species_r.append(species_react)
                    n_r.append(n_react)

                elif (rate_const == 0):
                    k_r.append(0)
                    Ea_r.append(0)
                    species_r.append(None)
                    n_r.append([0])

                elif (rate_const == -1):
                    k_r.append(rate_const)
                    Ea_r.append(0)
                    species_r.append(species_react)
                    n_r.append(n_react)
                    
                else:
                    raise Exception ('Error in the reverse expression '
                                     'of Reaction : {}, in '
                                     'Power Law (PL) '
                                     'type rate expression'.format(rxn_no))

            elif (reverse == True) and (rate_type == 'LNHW'):
                k_r.append(K_ads)
                Ea_r.append(del_H_ads)
                species_r.append(species_ads)
                n_r.append(n_ads)
                deno_raised_power.append(power)

            else:
                raise Exception ('Something unusual spotted in Reaction : {}, '
                                 'in reaction type.'.format(rxn_no))

    #### Converting all the lists to np.ndarray
    k_f = np.array(k_f)
    Ea_f = np.array(Ea_f)
    species_f = np.array(species_f)
    n_f = np.array(n_f)
    
    k_r = np.array(k_r, dtype= object)
    Ea_r = np.array(Ea_r, dtype= object)
    species_r = np.array(species_r, dtype= object)
    n_r = np.array(n_r, dtype= object)
    deno_raised_power = np.array(deno_raised_power, dtype= object)

    rate_type_arr = np.array(rate_type_list)
    
    #### Segregating the homogeneous and catalytic reaction parameters
    k_f_homo = list(k_f[homo_index])
    Ea_f_homo = list(Ea_f[homo_index])
    species_f_homo = list(species_f[homo_index])
    n_f_homo = list(n_f[homo_index])

    k_r_homo = list(k_r[homo_index])
    Ea_r_homo = list(Ea_r[homo_index])
    species_r_homo = list(species_r[homo_index])
    n_r_homo = list(n_r[homo_index])
    power_homo = list(deno_raised_power[homo_index])

    homo_rate_type = list(rate_type_arr[homo_index])

    k_f_cat = list(k_f[cat_index])
    Ea_f_cat = list(Ea_f[cat_index])
    species_f_cat = list(species_f[cat_index])
    n_f_cat = list(n_f[cat_index])

    k_r_cat = list(k_r[cat_index])
    Ea_r_cat = list(Ea_r[cat_index])
    species_r_cat = list(species_r[cat_index])
    n_r_cat = list(n_r[cat_index])
    power_cat = list(deno_raised_power[cat_index])
    
    cat_rate_type = list(rate_type_arr[cat_index])

    #### Retrieving the Thermodynamic datas
    all_del_H = const['all_del_H']
    all_del_S = const['all_del_S']
    
    del_H_homo = all_del_H[:, nu_homo_index, :]
    del_H_cat = all_del_H[:, nu_cat_index, :]

    del_S_homo = all_del_S[:, nu_homo_index, :]
    del_S_cat = all_del_S[:, nu_cat_index, :]

    #### Instantiating the reaction parameters
    homo = Reaction(homo_rate_type, k_f_homo, Ea_f_homo, 
                    species_f_homo, n_f_homo, k_r_homo, Ea_r_homo, 
                    species_r_homo, n_r_homo, power_homo, 
                    del_H_homo, del_S_homo)
    cat = Reaction(cat_rate_type, k_f_cat, Ea_f_cat, 
                   species_f_cat, n_f_cat, k_r_cat, Ea_r_cat, 
                   species_r_cat, n_r_cat, power_cat, 
                   del_H_cat, del_S_cat)

    #### Shipping
    return homo, cat, nu_homo_index, nu_cat_index

if __name__ == '__main__':

    import constants
    react_system = 'OCM_two_reaction.txt'
    catalyst = 'La_Ce'
    
    homo_basis = 'mole_fraction'
    cat_basis = 'mole_fraction'
    
    const, thermo_object = constants.fixed_parameters(react_system)
    homo, cat, \
    homo_index, cat_index = instantiate(react_system, const, catalyst)

    
    print(homo.n_r)
    species = const['species']
    no_of_species = len(species)
    ID = np.arange(no_of_species)
    species_ID = dict(zip(species, ID))
    print(species_ID)
    
    inlet_species = 'CH4'
    inlet_ratio = 4
    
    F_A_in = inlet_ratio/(inlet_ratio+1)
    F_B_in = 1/(inlet_ratio+1)
    
    F_in = 1e-08 * np.ones(no_of_species)
    A_index = species_ID[inlet_species]
    B_index = species_ID['O2']
    
    F_in[A_index] = F_A_in
    F_in[B_index] = F_B_in
    
    T_f = 300

    conc_dict = dict(zip(species, F_in))
    fwd_rate = homo.calc_PL_rate(conc_dict, homo_basis, homo.k_f[0], 
                                 homo.Ea_f[0], homo.species_f[0], 
                                 homo.n_f[0], T_f, 1)
    print(fwd_rate)
    print(homo.k_r)
