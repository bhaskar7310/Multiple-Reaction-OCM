#### This module extracts the essential data (basically coefficients) from the
#### GRI-thermo datafile for a required set of compounds. It can also be used to 
#### calculate the average C_p data for different elements or compounds.

#### Author : Bhaskar Sarkar (2018)

import os
import numpy as np
from math import pow

def T_arr(T, T0):
    
    T_list = [T-T0, 1/2*(pow(T,2) - pow(T0,2)), 1/3*(pow(T,3) - pow(T0,3)), 1/4*(pow(T,4) - pow(T0,4)),\
              1/5*(pow(T,5) - pow(T0,5))]
    T_mat = np.array(T_list).reshape(5, 1)
    
    return T_mat


def post_process(elem, return_list):
    
    try:
        float_elem = float(elem)
        return_list.append(float_elem)
    
        return return_list
    
    except ValueError:
                
            n = len(elem)
            index = 0
            while (index < n):
                if elem[index] == 'E':
                    break
                index += 1
                
            first_elem = elem[:index+4]
            second_elem = elem[index+4:]
            
            return_list.append(float(first_elem))
            post_process(second_elem, return_list)
     
    return return_list


def gri_data(set_of_compounds):
    '''
    This is the main function that takes the set of compounds for which GRI-Data 
    is required and then calculates the coefficients from the table.
    '''
    filename = 'thermo_gri.txt'
    FullFileName = os.path.join(os.getcwd(), filename)

    if os.path.isfile(FullFileName) == False:
        raise Exception('The data file named {} does not exist'.format(filename))
        
    with open(filename, 'r') as infile:
        
        first_line = infile.readline()
        flag = 0
        count = 0
        
        compound_count = 0
        output_compounds = []
        
        filestr = []
        fileno_list = []
    
        for lines in infile.readlines():
    
            if (flag == 0):
                lines_mod = lines.strip()
                lines_act = lines_mod.split(" ")
                compound = lines_act[0]
            
            #### Processing of the three lines attached to our specific compound
            if (flag == 1):
                line_mod = lines.strip()
                line_no_spaces = line_mod.replace("   ", "")
                filestr[compound_count-1].append(line_no_spaces.split())
                count += 1
        
            #### We found our sweet element/radical/compound or whatever fuck 
            #### it is, in the following next three step we
            #### will start our processing
            if compound in set_of_compounds:
                count = 0
                flag = 1
                output_compounds.append(compound)
                compound_count += 1
                filestr.append([])
                compound = 'None'
        
            #### As soon as we are done with that specific compound, 
            #### we will start searching for the next compound 
            if (count == 3):
                flag = 0
    
                
    #### Simple check to ensure that we got data for all the required compounds
    no_inputs = len(set_of_compounds)
    no_outputs = len(output_compounds)
    
    if (no_inputs != no_outputs):
        for elem in set_of_compounds:
            if elem not in output_compounds:
                raise ValueError('Data for the {} compound is not found'.format(elem))
    
    
    #### This is where we process the three lines we read from the file for 
    #### the specific compounds.
    
    for elem in filestr:
        fileno = []
        for e1 in elem:
            del e1[-1]
            for e2 in e1:
                fileno = post_process(e2, return_list = fileno)
        fileno_list.append(fileno)
    
        
    #### The final step, creating a dictionary
    compounds_coeff = dict(zip(output_compounds, fileno_list))

    return compounds_coeff

if __name__ == '__main__':
    import stoichio_coeff_calculator as stoich
    filename = 'ODH.txt'
    all_val = stoich.main_func(filename)
    coeff = gri_data(all_val[0])
    print(coeff)