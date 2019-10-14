#### This function/module will read the eigen values written in the
#### eigen_value.txt to make some sense out of the system
#### The important question is how to make sense of all those eigen values
import os
import numpy as np
import matplotlib.pyplot as plt


def reader(file):
    '''
    This function reads the entire file and stores
    all the values into a giant matrix (since there are
    lots and lots of eigen values)
    '''
    eig_val_list = []
    temp = []
    count = 0
    
    with open(file, "r") as infile:
        system = infile.readline()
        for line in infile.readlines():
            count += 1
            vals = line.strip().split(" ")
            m = len(vals)
            for i in range(m):
                if i == 0 and count % 2 != 0:
                    temp.append(float(vals[i]))
                else:
                    try:
                        eig_val_list.append(float(vals[i]))
                    except ValueError:
                        eig_val_list.append(complex(vals[i]))
            print(count)
                
    n = len(eig_val_list)
    no_of_rows = count
    no_of_cols = int(n/count)
    
    eig_val_mat = np.array(eig_val_list).reshape(no_of_rows, no_of_cols)
    
    return np.array(temp), eig_val_mat


if __name__ == "__main__":
    
    #### Reaction system
    react_sys = 'OCM_eleven_reaction.txt'
    catalyst = 'model'                        
    system = 'coup'   

    #### Values of parameters                      
    inlet_ratio = 6
    P = 1             # Pressure is in atm (for the time being)            
    tau = 1e-02      
    T_f_in = 300
    
    R_omega = 0.25e-03                  
    R_omega_wc = 50e-06
    R_omega_w = 140e-06
    particle_density = 3600
    
    ##### Bifurcation Parameter
    fixed_var = ['inlet_ratio', 'pressure', 
                 'tau', 'R_omega', 'R_omega_wc', 
                 'R_omega_w', 'particle_density']
    fixed_val = [inlet_ratio, P, tau, R_omega, 
                 R_omega_wc, R_omega_w, particle_density]
    fixed_dict = dict(zip(fixed_var, fixed_val))                        


    #### File and folder specifications
    data_vals = [value for (key, value) in sorted(fixed_dict.items())]
    n = len(data_vals)
    react_filename, ext = os.path.splitext(react_sys)

    #### Filename
    file = react_filename + '_' + catalyst.lower() + '_' + system
    for i in range(n):
        file += '_{}'.format(data_vals[i])
    file += '.txt'
    
    folder = os.path.join(os.getcwd(), 'EigenValues')
    fullfilename = os.path.join(folder, file)    
    print(fullfilename)
    
    #### Reading the files
    temp, eig_val_mat = reader(fullfilename)
    cat_eig_val_mat = eig_val_mat[::2]
    homo_eig_val_mat = eig_val_mat[1::2]

    #### Post-processing the read eigen values
    homo_real_val = homo_eig_val_mat.real
    homo_imag_val = homo_eig_val_mat.imag
    homo_abs = np.absolute(homo_eig_val_mat)

    cat_real_val = cat_eig_val_mat.real
    cat_imag_val = cat_eig_val_mat.imag
    cat_abs = np.absolute(cat_eig_val_mat)

    #### Visualizing the eigen values
    fig, ax = plt.subplots()
    ax.scatter(temp, cat_real_val[:, 3], s=1)