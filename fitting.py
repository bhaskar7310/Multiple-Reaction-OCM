'''
Sample file to solve for the diffusivity coefficients using the Chapman-Enskog 
theory. The coefficients are derived using the Least Square method.
'''

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def least_sq_fit(X, Y, no_of_depen= 2):
    '''
    This function takes in the X and Y vectors (arrays) as inputs. Using the 
    linear least squares algorithm it calculates the vector b which fits the 
    data in the form of Y = A*b. And then it returns the b vector i.e in terms 
    of function y = b0 + b1*x is the linear fit of the data.
    '''

    #### Checking the inputs
    n = X.shape[0]
    m = Y.shape[0]

    if (n!= m):
        raise Exception ('The dimensions of X and Y data points do not match, ' 
                            'check the discrepancy.')

    #### Forming the rectangular matrix A
    A = np.c_[np.ones(n).reshape(n, 1), X]
    
    #### Solving for b
    left_matrix = np.dot(A.T, A)
    right_vector = np.dot(A.T, Y)

    b = LA.solve(left_matrix, right_vector)

    return b


####------------------------------------------------------------------------------------------------
####                Our Method (Little bit of error will be associated with this method)
####------------------------------------------------------------------------------------------------
#### Diffusivity of Methane in air
#Dab = np.array([2.155e-05, 2.9389e-05, 5.1721e-05, 7.58122e-05, 1.0328e-04, 1.678e-04, 2.4349e-04])
#T = np.array([300, 354, 489, 611, 733, 978, 1222])

##### Diffusivity of Oxygen in air
#Dab = np.array([2.2457e-05, 2.945e-05, 3.715e-05, 4.5798e-05, 5.44885e-05, 7.42616e-05, 9.6318e-05, \
#                12.0531e-05, 14.05286e-05, 17.5006e-05])
#T = np.array([314, 366.4, 418.8, 471.1, 523.475, 628.17, 732.87, 837.56, 915.255, 1046.95])

##### Diffusivity of Propane in air
#Dab = np.array([7.733e-06, 1.3278e-05, 1.9937e-05, 2.7605e-05, 3.61907e-05, 4.5647e-05, 5.59106e-05, \
#                6.69497e-05, 9.1245e-05, 11.834e-05, 14.8096e-05, 18.0363e-05, 21.5029e-05])
#T = np.array([244.1, 325.46, 406.825, 488.2, 569.55, 650.92, 732.3, 813.65, 976.38, 1139.11, 1301.84, 1464.57, 1627.3])
    
##### Thermal Diffusivity of Air
#alpha = 1e-05 * np.array([1.5672, 1.6896, 1.9448, 2.2156, 2.5003, 2.7967, 3.1080, 3.7610, 4.4537, 5.1836, 5.9421, 7.1297, 9.6632, 11.9136, 16.7583])
#T = np.array([250, 260, 280, 300, 320, 340, 360, 400, 440, 480, 520, 580, 700, 800, 1000])
#alpha = 1e-05 * np.array([2.2156, 2.5003, 2.7967, 3.1080, 3.7610, 4.4537, 5.1836, 5.9421, 7.1297, 9.6632, 11.9136, 16.7583])
#T = np.array([300, 320, 340, 360, 400, 440, 480, 520, 580, 700, 800, 1000])
    
##### Thermal Diffusivity of Oxygen
#alpha = 1e-05 * np.array([1.5803, 2.2365, 2.9639, 3.7705, 4.6216, 5.5056, 6.4502, 7.4192])
#T = np.arange(250, 601, 50)
    
##### Thermal Diffusivity of Carbon dioxide
#alpha = 1e-05 * np.array([0.7394, 1.0818, 1.4808, 1.9454, 2.4756, 3.0763, 3.7406, 4.4793])
#T = np.arange(250, 601, 50)
    
#### Thermal Diffusivity of Carbon Monoxide
alpha = 1e-05 * np.array([1.5040, 2.1277, 2.8323, 3.6057, 4.4408, 5.3205, 6.2350, 7.1894])
T = np.arange(250, 601, 50)
#X = np.log(T)
#Y = np.log(alpha)
#
#B = least_sq_fit(X, Y)
#
##### Calculating a and b in y = aT**b
#a = np.exp(B[0])
#b = B[1]
#print(a)
#print(b)

####------------------------------------------------------------------------------------------------
####                                    Scipy Curve Fitting Method
####------------------------------------------------------------------------------------------------

def func(x, a):
    return a*x**1.75

popt, pcov = curve_fit(func, T, alpha)
print(popt)

