'''
This is Python version of the derpar subroutine for arc_length continuation 
method originally written in Fortran.
'''
from math import copysign
import numpy as np    

def gauss_elimination(A, B, pref):
    '''
    This is the equivalent of GAUSE subroutine of DERPAR code. 
    It takes in the full jacobian matrix, A (N x N+1) and the 
    function values, B (N X 1) evaluated at a certain X (N+1, 1). 
    pref (N+1, 1) is the preference number for X (N+1, 1). 
    0 < pref[i] < 1, the lower the pref[i], the higher is preference of X[i].
    '''

    #### Checking the inputs
    n, n1 = A.shape

    #### Initializing variables
    irk = -1 * np.ones(n1, dtype= int)
    irr = -1 * np.ones(n1, dtype= int)

    #### Line no. 20 loop starts here
    for k in range(n):
        ir = 0
        ic = 0
        amax = 0

        for i in range(n):
            if irr[i] == -1:
                for j in range(n1):
                    p = pref[j] * abs(A[i, j])
                    if (p - amax) > 0:
                        ir = i
                        ic = j
                        amax = p
        
        #### Singularity check
        if amax == 0:
            raise Exception ('Singular matrix is encountered')
        
        irr[ir] = ic
        
        for i in range(n):
            if (i != ir) and (A[i, ic] != 0):
                factor = A[i, ic]/A[ir, ic]

                #### Row elimination
                for j in range(n1):
                    A[i, j] = A[i, j] - factor * A[ir, j]
                A[i, ic] = 0
                
                #### Dealing with the function values
                B[i] -= factor * B[ir]
    
    #### Back substitution stuffs
  
    #### Pre-allocation
    F = np.zeros(n1)                # X = Function/Jacobian
    beta = np.zeros(n1)             # These are supposedly beta values

    for i in range(n):
        ir = irr[i]
        F[ir] = B[i]/A[i, ir]
        irk[ir] = 1
    
    for k in range(n1):
        if (irk[k] == -1):
            for i in range(n):
                ir = irr[i]
                beta[ir] = -A[i, k]/A[i, ir]
            break

    beta[k] = 0
    F[k] = 0

    return beta, F, k

def adams(dxds, X, madms, h, der):
    ''' 
    ODE integrator by Adams-Bashforth formula.
    '''
    
    mxadms = 4
    madms = madms + 1
    if (madms > mxadms):
        madms = mxadms
    if (madms > 4):
        madms = 4
        
    for i in range(mxadms-1, 0, -1):
        der[i, :] = der[i-1, :]
  
    #### Adams-Bashforth Integration formula
    der[0,:] = dxds[:]
    if (madms == 1):
        X += h*der[0,:]
    elif (madms == 2):
        X += 0.5 * h * (3.0*der[0,:] - der[1, :])
    elif (madms == 3):
        X += h*(23*der[0, :] - 16*der[1,:] + 5*der[3,:])/12
    elif (madms == 4):
        X += h*(55*der[0,:] - 59*der[1,:] + 37*der[2,:] - 9*der[3,:])/24
    else:
        raise Exception('madms out of range!')
        
    return X, der, madms


def derpar(original_func, jacobian, X, pref, max_val, min_val, 
           weights, jac_eps, eps= 1e-04, initial= True, maxIter= 50,
           hh = 0.05, hhmax = 0.5, ncorr= 4, ncrad= 0, 
           maxout=0, args= (), kwargs=()):

    #### Checking inputs
    n1 = X.shape[0]
    
    #### Declaration of variables
    nout = 0        
    kout = 0
    madms = 0
    nc = 1
    k1 = -1

    hhmax = copysign(hhmax, hh)
    hmax = hhmax*np.ones(n1)

    ndir = np.ones(n1)
    der = np.zeros((4, n1))
    
    #### Initial Newton Iterations
    if not initial:
        for count in range(maxIter):
            func = original_func(X, *args)
            jac = jacobian(original_func, X, func, jac_eps, n1-1, n1, 
                           0, n1, *args)
            beta, fx_dx, k = gauss_elimination(jac, func, pref)

            X -= fx_dx
#            print('At iteration {0}, the abs_err is {1}'.format(count, abs(fx_dx)))
            abs_err = np.dot(weights.T, abs(fx_dx))
            print('At iteration {0}, the abs_err is {1}'.format(count, abs_err))
            if (abs_err <= eps):
                initial= True
                break
        
        if initial == False:
            raise Exception ('The solution cannot be converged in the ' 
                              'initial point.')
        else:
            print('Solution converged at the initial point after ' 
                  '{} iterations'.format(count + 1))

    print(X)
    #### After initial Newton Iterations
    while(True):
        jac_eps_corr_loop = 1
        while (True):

            func = original_func(X, *args)
            jac = jacobian(original_func, X, func, jac_eps, n1-1, n1, 
                           0, n1, *args)
            beta, fx_dx, k = gauss_elimination(jac, func, pref)
#            print(beta)
            # Change of independent variable (its index = k now)
            if (k1 != k):
#                print('Order changed because of k')
                madms = 0
                k1 = k
                
            abs_err = np.dot(weights.T, abs(fx_dx))

            #### Breaking criterion of the inner loop
            if (abs_err > eps) and (nc < ncorr):
                X -= fx_dx
                nc += 1
            elif (abs_err > eps) and (nc >= ncorr):
                nc = 1
                print('Warning! Cannot converge in {} iterations.'
                      .format(ncorr))
                if jac_eps > 1e-03:
                    jac_eps = 1e-04
                else:
                    jac_eps *= 10.0
                print('Trying with jac_eps = ', jac_eps)
                jac_eps_corr_loop += 1
                if jac_eps_corr_loop > 3:
                    return Xsol
            else:
                nc = 1
                break

        #### Total Iteration counts
        nout += 1
        if (nout > maxout):
            print('Maximum nuber of output reached')
            if nout == 1:
#                print(X)
                return X
            else:
#                print(Xsol)
                return Xsol
        if nout == 1:
            Xsol = X.copy()
        else:
            Xsol = np.c_[Xsol, X]
        
        print('At {0} iteration, value : {1}'.format(nout, X[-1]))
    
        #### Last point check
        if abs(X[-1]) > max_val:
            print('Maximum value of bifurcation parameter exceeded')
            break
        
        if abs(X[-1]) < min_val:
            print('Minimum value of bifurcation parameter reached')
            break

        #### Checking for closed curve
        if (nout >= 3):
            p = np.dot(weights.T, abs(X - Xsol[:, 0]))
            
            if (p <= 1e-08):
                print('Execution stopped because of closed curves')
                return Xsol
        
        #### Integration starts here
        dxdt = np.zeros(n1) 
        denom = 1 + np.sum(beta ** 2)  
        dxdt[k] = 1/np.sqrt(denom) * ndir[k]
       
        h = hh
        #### Defining the integrals and manipulating the step size
        for i in range(n1):
            ndir[i] = 1
            if (i != k):
                dxdt[i] = beta[i] * dxdt[k]
            
            if (dxdt[i] < 0):
                ndir[i] = -1
            
            if (abs(h)*abs(dxdt[i]) > abs(hmax[i])):
                madms = 0
                h = hmax[i]/abs(dxdt[i])

        #### Calling the Adams-Bashforth integrator
        while(True):      
            if ((nout > kout+3)
                    and (abs(h)*abs(dxdt[k]) > 0.8*abs(X[k] - Xsol[k, 0]))
                    and ((Xsol[k, 0] - X[k])*ndir[k] > 0)):
                madms = 0
            else:
                X, der, madms = adams(dxdt, X, madms, h, der)
                break
            
            if (abs(h)*abs(dxdt[k]) > abs(X[k] - Xsol[k, 0])):
                h_no_sign = abs(X[k] - Xsol[k, 0])/abs(dxdt[k])
                h = copysign(h_no_sign, hh)
                kout = nout
   
            X, der, madms = adams(dxdt, X, madms, h, der)
            break
    
    return Xsol
