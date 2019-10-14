'''
This is a module that reads the reactions written in the form of a textfile and 
then returns the number of reactions, species and the stoichiometric
coefficients of the reaction system.

Now this is the very beginning phase, we haven't included all the docstrings
and the comments in this codes yet

'''


import numpy as np


def reversibility_check(fullline_str):
    '''
    This function takes a list of string and looks for '<->' or '->' element 
    in that. Based on the result it returns '<->' or '->'. '<->' signifies 
    reversible reaction and '->' irreversible reactions.
    '''
    space_split = fullline_str.split(' ')
    if '<->' in space_split:
        return '<->', 1
    else:
        return '->', 0
    

def stoich_coeff(compounds, flag):
    '''
    This function takes in the different compounds as lists and then seperates 
    the stoichiometric coefficient involved with that particular compound. 
    The value of flag determines whether the compounds are products or reactants.
    If flag == 1 then the compounds are products, if flag == -1, then they are 
    reactants. At the end this function returns one list containg the coefficients
    stripped molecules/compounds, and another list containg the corresponding 
    coefficients.
    '''
    comp_list = []
    nu_list = []
    for elem in compounds:
        try:
            float(elem[0])
            index = 0
            for e1 in elem[1:]:
                if (e1 == '/'):
                    index += 1
                    count = index
                else:
                    try:
                        float(e1)
                        index += 1
                    except:
                        index += 1
                        break
        
            comp_list.append(elem[index:])
            try:
                nu_act = float(elem[:count])/float(elem[count+1:index])
                nu_list.append(flag * nu_act)
            except NameError:
                nu_list.append(flag * float(elem[:index]))
            
        except ValueError:
            comp_list.append(elem)
            nu_list.append(flag)
            
    return comp_list, nu_list


def seperator(str_list, flag):
    '''
    This is a small function that splits the str_list against the '+' sign. 
    Basically it separates the different compounds involved in the reactions and
    asks the stoich_coeff function to seperate the stoichiometric coefficients.
    And at last it returns the coefficient stripped compounds and the corresponding
    coefficients.
    '''
    loc_comp = []
    elem = str_list.split('+')
    for e1 in elem:
        loc_comp.append(e1.strip())
    comp, nu = stoich_coeff(loc_comp, flag)
    return comp, nu

def micro_kinetic_check(spcs):
    '''
    This function takes in the the list of identified species and then checks 
    whether the reaction mechanism specified is a micro-kinetic model. If it
    finds that the model is micro-kinetic it classifies the species and make two 
    seperate lists of species, one containing the compounds and other containing
    the surface adsorped species.
    '''
    species = []
    surf_species = []
    surf = []
    
    #### Segregating the normal species and the surface species
    for elem in spcs:
        micro_check = elem.split('-')
        if (len(micro_check) == 1):
            species.append(elem)
        else:
            surf_species.append(elem)
            surf.append(micro_check[-1])
    
    #### Checking for any vacant species in normal species    
    for e1 in surf:
        if e1 in species:
            surf_species.append(e1)
            species.remove(e1)
            
    #### Finding the number of sites avaiable
    surf_site = list(set(surf))
    
    return species, surf_species, surf_site       
    

def main_func(filename):
    '''
    This is the main function of this module (so far). It open the Reaction.txt
    file, reads the reaction, understands them (by some other functions) and
    then returns the total no. of functions, total no. of species and their
    corressponding stoichiometric coefficients.
    '''
    
    R_comps = []
    R_nus = []
    R_rev = []
    
    with open(filename, 'r') as infile:
        first_line = infile.readline()
        for line in infile.readlines():
            limiter, rev_flag = reversibility_check(line)
            first_seg = line.split(limiter)
            R_rev.append(rev_flag)
            
            reactants = first_seg[0].strip()
            products = first_seg[-1].strip()
     
            comp1, nu1 = seperator(reactants, flag = -1)
            comp2, nu2 = seperator(products, flag = 1)
            
            comp_all = comp1 + comp2
            nu_all = nu1 + nu2
            
            R_comps.append(comp_all)
            R_nus.append(nu_all)
    
    
    no_of_react = len(R_comps)              # No. of reactions
    all_comps = sum(R_comps, [])                   
    final_comp = list(set(all_comps))       # The names of species involved 
    
    no_of_species = len(final_comp)         # No. of species
    nu = np.zeros([no_of_species, no_of_react])
    n = np.arange(no_of_species)
    
    #### Stoichiometric coefficient matrix calculation
    for num, elem in zip(n, final_comp):
    
        for i in range(no_of_react):
            if elem in R_comps[i]:
                pos = R_comps[i].index(elem)
                nu[num, i] = R_nus[i][pos]
                
    #### We do need to deal this through classes. for the time being we are just
    #### returning a set of values
    
    #### Checking whether the reaction mechanism is a micro-kinetic one
    species, surf_species, surf_site = micro_kinetic_check(final_comp)
    no_of_surf = len(surf_species) 
    
    if (no_of_surf == 0):
        print('This is a global kinetic model')
        rev_bool = np.array(R_rev)
        return final_comp, nu, rev_bool
    
    #### This is not required as of now
    else:
        #### Checking the sizes 
        no_of_species = len(species)
        n = len(surf_site)
        print('This is a micro-kinetic model with %d surface sites' %n)
        
        #### Pre-allocation
        nu_gas = np.zeros([no_of_species, no_of_react])     # nu for gas phase species
        nu_surf = np.zeros([no_of_surf, no_of_react])       # nu for surface species
        
        spcs_index = 0
        surf_index = 0
        
        #### Defining the stoichiometric coefficient matrix for gas phase species
        for e1 in species:
            i = final_comp.index(e1)
            nu_gas[spcs_index, :] = nu[i, :]
            spcs_index += 1
        
        #### Defining the stoichiometric coefficient matrix for surface species
        for e2 in surf_species:
            j = final_comp.index(e2)
            nu_surf[surf_index, :] = nu[j, :]
            surf_index += 1
        
        return species, nu_gas, surf_species, nu_surf


if __name__ == '__main__':
    '''
    Executing the module as a funtion.
    '''
#    filename = 'OCM_Arun_Kota_homo.txt'
    filename = 'OCM_ten_reaction.txt'
    all_val = main_func(filename)
    
    if len(all_val) == 3:
        print('The number of species involved:\n {}'.format(all_val[0]))
        print('The stoichiometric coefficient matrix:\n {}'.format(all_val[1]))
        print('The reversible reactions can be identified by this boolean list:\n {}'.format(all_val[2]))
    else:
        print('The number of gas phase species involved:\n {}'.format(all_val[0]))
        print('The stoichiometric coefficient matrix of gas phase reactions:\n {}'.format(all_val[1]))
        print('The number of surface species involved:\n {}'.format(all_val[2]))
        print('The stoichiometric coefficient matrix of surface reactions:\n {}'.format(all_val[3]))
        
