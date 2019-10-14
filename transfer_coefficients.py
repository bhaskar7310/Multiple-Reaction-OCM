import cmath
import numpy as np

def sherwood_inv_func(x):
    '''
    x : an input vector
    f(x) = 1/(np.sqrt(x) * np.tanh(x)) - 1/(np.sqrt(x))**2
    Now this will give error if x goes to 0.
    '''
    #### Checking the inputs
    try:
        n = len(x)
    except TypeError:
        x = np.array([x])
        n = 1
    f = np.zeros(n, dtype=complex)
    
    #### Evaluating the function
    for i in range(n):
        if abs(x[i]) < 1e-04:
            f[i] = 1/3 + 0j
        else:
            f[i] = 1/(cmath.sqrt(x[i]) * cmath.tanh(cmath.sqrt(x[i]))) \
                 - 1/x[i] 

    return f


def sherwood_inv_derivative_at_zero(k):
    '''
    Returns analytically solved derivatives of 
    sherwood_inv_func at x = 0.
    '''
    if k == 0:
        return 1/3
    elif k == 1:
        return -1/45
    elif k == 2:
        return 4/945
    elif k == 3:
        return -2/1575
    elif k == 4:
        return 16/31185
    elif k == 5:
        return -11056/42567525
    elif k == 6:
        return 64/405405
    elif k == 7:
        return 57872/516891375
    else:
        raise Exception ('Higher order derivatives at with limit x-> 0 is not '
                         'provided, requested k = {}.'.format(k))
    

def sherwood_lambda_func(x, alpha):
    '''
    x: a vector
    f(x) = 3.0 + cmath.sqrt(x) * cmath.tanh(alpha * cmath.sqrt(x))
    '''
    #### Checking the inputs
    try:
        n = len(x)
    except TypeError:
        x = np.array([x])
        n = 1
    f = np.zeros(n, dtype=complex)

    #### Evaluating the function
    for i in range(n):
        f[i] = 3.0 + cmath.sqrt(x[i]) * cmath.tanh(alpha * cmath.sqrt(x[i]))

    return f

def sherwood_lambda_derivative_at_zero(k, alpha):
    '''
    Returns analytically solved derivatives of 
    sherwood_lambda_func at x = 0.
    '''
    if k == 0:
        return 3
    elif k == 1:
        return alpha
    elif k == 2:
        return -2*alpha**3/3
    elif k == 3:
        return 4*alpha**5/5
    elif k == 4:
        return -136*alpha**7/105
    elif k == 5:
        return 496*alpha**9/189
    elif k == 6:
        return -22112 * alpha**11/3465
    elif k == 7:
        return 349504 * alpha**13/19305
    else:
        raise Exception ('Higher order derivatives at with limit x-> 0 is not '
                         'provided, requested k = {}.'.format(k))


