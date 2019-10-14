import cmath
from math import pow
import numpy as np
import numpy.linalg as LA

def mat_func(func_name, jac_name, A, *args, **kwargs):
    '''
    Matrices with distinct eigen values.
    And just 2D matrix
    '''
    #### Checking the inputs
    N = A.shape
    assert (len(N) > 1), ('1D Arrays cannot be dealt with this function')
    assert (N[-1] == N[-2]),('You can calculate functions of a non-square '
                       'or rectangular matrix using Caley-Hamilton Theorem')
    filename = kwargs.get('filename', None) 
    
    #### Getting the eigenvalues
    n = A.shape[0]
    lamb, X = LA.eig(A)
    #print("The incoming matrix:\n", A)
    #print("The eigen values:\n", lamb)
    #print("The eigen vectors:\n", X)
    #max_lamb = np.max(lamb)
    
    #lamb = np.where(abs(lamb)/max_lamb < 1e-03, 0, lamb)
    
    
    if lamb.dtype == float or lamb.dtype == int:
        lamb = np.where(abs(lamb)<1e-2, 0, lamb)
    if lamb.dtype == complex:
        lamb = np.where(abs(lamb)<1e-2, 0, lamb)

    if filename:
        with open(filename, 'a') as fh:
            for i in range(n):
                fh.write(str(lamb[i]) + ' ')
            fh.write('\n')
            

    ## Checking if any eigen value is zero
    if 0 in lamb:
        C = _mat_func_zero_eigen_value(func_name, jac_name, A, lamb, n, *args)
        
    else: 
        #### Gettting the right hand side vector b of lhs * C = rhs
        rhs = func_name(lamb, *args)
    
        #### Getting the left hand side matrix, lhs
        lhs = np.zeros([n, n], dtype=complex)
        for i in range(n-1):
            lhs[:, i] = lamb**(n-1-i)
        lhs[:, -1] = 1.0
    
        #### Vector of coefficients
        try:
            C = LA.solve(lhs, rhs)
            
        except LA.LinAlgError:
            raise LA.LinAlgError
    
    #### Calculating the function of the matrix
    func_A = np.zeros([n, n], dtype=complex)
    for i in range(n-1):
        func_A += C[i] * LA.matrix_power(A, n-1-i)

    func_A += C[-1] * np.eye(n)
    
    #### Shipping
    return func_A.real


def _mat_func_zero_eigen_value(func_name, jac_name, A, lamb, n, *args):
    '''
    This is a helper function which calculates the function of a
    matrix A which has atleast 1 eigen value (lamb) equal to zero.

    func_name : Name of the function, for which f(A) has to be calculated.

    A : Matrix A

    lamb : eigen values of A

    n : No. of rows/cols of matrix
    
    returns func_name(A) to the calling function.
    '''
    #### The unique set of elements
    unique_lamb, index, inv_index, count = np.unique(lamb, return_index=True,
                                                     return_inverse=True,
                                                     return_counts=True)

    #### Now iterating through the unique set of eigen values
    n_unique = len(unique_lamb)
    C = np.zeros(n, dtype=complex)

    for i in range(n_unique):
        if unique_lamb[i] == 0.0:
            zero_repeat = count[i]
            for k in range(zero_repeat):

                C[-1-k] = 1/factorial(k)\
                        * jac_name(k, *args)
            break

    unique_wo_zero = np.delete(unique_lamb, i)
    count_wo_zero = np.delete(count, i)
    m = n - zero_repeat

    for j in range(m):
        try:
            if count_wo_zero[j] != 1:
                raise Exception ('This process will not work in case of multiple '
                                 'non-zero eigen values')
        except IndexError:
            if count_wo_zero != 1:
                raise Exception ('This process will not work in case of multiple '
                                 'non-zero eigen values')
                
    #### Getting the right hand side vector b of lhs * c = RHS - known_consts
    RHS = func_name(unique_wo_zero, *args)
    
    #### Getting the known constants
    known_const = np.zeros(m, dtype=complex)

    for i in range(zero_repeat, 0, -1):
        known_const +=  C[n-i] * unique_wo_zero**(i-1)
    rhs = RHS - known_const

    #### Getting the left hand side matrix lhs
    lhs = np.zeros([m, m], dtype=complex)
    for i in range(m):
        lhs[:, i] = unique_wo_zero**(n-1-i)

    #### Solving for the rest of the coeffcients
    c = LA.solve(lhs, rhs)

    #### Packing and Shipping 
    C[0:m] = c
    return C


def factorial(x):
    '''
    This is a function that calculates the factorial of a
    positive integer using recursion.
    '''
    assert (x == int(x)),('Factorial of only integer values can be calculated.')
    assert (x >= 0), ('Factorial of only positive integer values can be '
                      'calculated.')

    if x == 0 or x == 1:
        return 1
    else:
        return x * factorial(x-1)


####--------------------------------------------------------------------------
####                SOME USEFUL COMMON MATHEMATICAL FUNCTIONS         
####--------------------------------------------------------------------------
def square(x):
    try:
        n = len(x)
    except TypeError:
        x = np.array([x])
        n = 1
    f = np.zeros(n, dtype=complex)
    for i in range(n):
        f[i] = x[i]**2
    return f

def cube(x):
    try:
        n = len(x)
    except TypeError:
        x = np.array([x])
        n = 1
    f = np.zeros(n, dtype=complex)
    for i in range(n):
        f[i] = x[i]**3
    return f

def neg_square(x):
    return x**-2

def square_root(x):
    try:
        n = len(x)
    except TypeError:
        x = np.array([x])
        n = 1
    f = np.zeros(n, dtype=complex)
    for i in range(n):
        f[i] = cmath.sqrt(x[i])
    return f
    
def inverse(x):
    return 1/x

def tanh(x):
    try:
        n = len(x)
    except TypeError:
        x = np.array([x])
        n = 1
    f = np.zeros(n, dtype=complex)
    for i in range(n):
        f[i] = cmath.tanh(x[i])
    return f


def sherwood_tanh(x):
    try:
        n = len(x)
    except TypeError:
        x = np.array([x])
        n = 1
    f = np.zeros(n, dtype=complex)
    for i in range(n):
        f[i] = cmath.sqrt(x[i]) * cmath.tanh(cmath.sqrt(x[i]))
    return f


if __name__ == '__main__':
    
    k = 4
    eps = 0.01*pow(10, -6/(k))
    h_func_deriv = func_derivative(tanh, 0, k, eps)
#    print(h_func_deriv)
#    print(eps)
    print(h_func_deriv/pow(eps, k))
    
#    A = np.array([[3, 6, -8],
#                  [0, 0, 6],
#                  [0, 0, 0]])
##    print(LA.eig(A))
#    A_sq_our_method = mat_func(tanh, A)
#    A_sq_builtin = funm(A, tanh)
#    A_sq_matrix_power = LA.matrix_power(A, 3)
#    print('Builtin method:')
#    print(A_sq_builtin)
#    print(A_sq_matrix_power)
#    print('\nOur method:')
#    print(A_sq_our_method)
#    
    
#    
#    A = np.random.rand(10, 10)
#    A[0, :] = 0
#    A[2, :] = 0
#    A[3, :] = 0
#    A[5, :] = 0
#    A_tanh_builtin= funm(A, sherwood_tanh)
#    A_tanh_our_methd = mat_func(sherwood_tanh, A)
#    print('Builtin method:')
#    print(A_tanh_builtin)
#    print('\nOur method:')
#    print(A_tanh_our_methd)
#    
    
    #A = np.array([[1, 2], [3, 4]])
    #A_sq_builtin = LA.matrix_power(A, -2)
    #A_sq_our_method = mat_func(neg_square, A)
    #print('Test 1\n')
    #print('Builtin method:')
    #print(A_sq_builtin)
    #print('\nOur method:')
#    #print(A_sq_our_method)
#
#    A = np.random.randint(1, 10, (5,5))
#    A_sq_builtin = LA.matrix_power(A, -2)
#    A_sq_our_method = mat_func(neg_square, A)
#    print('Test 2\n')
#    A_check = A_sq_builtin - A_sq_our_method
#    print(A_check < 1e-04)
#    print(A_sq_builtin)
#    print(A_sq_our_method)
    
#    A = np.random.rand(10, 10)
#    A_sq_builtin = LA.matrix_power(A, -1)
#    A_sq_our_method = mat_func(inverse, A)
#    print('Test 3\n')
#    print('Builtin method:')
#    print(A_sq_builtin)
#    print('\nOur method:')
#    print(A_sq_our_method)
#    A_check = A_sq_builtin - A_sq_our_method
#    print(A_check < 1e-04)
#    print(A_sq_builtin)
#    print(A_sq_our_method)
