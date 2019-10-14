import numpy as np

def gauss_elimination(A, B, pref):
    '''
    This is the equivalent of GAUSE subroutine of DERPAR code. 
    It takes in the full jacobian matrix, A (N x N+1) and the function values, 
    B (N X 1) evaluated at a certain X (N+1, 1). pref (N+1, 1) is the 
    preference number for X (N+1, 1). 
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


def derpar(original_func, jacobian, X, pref, maxval, eps= 1e-04, initial= True, 
           maxIter= 50, hh = 0.05, ncorr= 4, ncrad= 0, 
           maxout= 0, args=()):
 
    #### Checking inputs
    n1 = X.shape[0]
    
    #### Declaration of variables
    nout = 0
    kout = 0
    madms = 0
    nc = 1
    hmax = 0.1 * np.ones(n1)
    weights = np.ones(n1)
    ndir = np.ones(n1)
    der = np.zeros((4, n1))
    print(original_func)
    #### Initial Newton Iterations
    if not initial:
        for count in range(maxIter):
            func = original_func(X, *args)
            print('func values in initial iterations: ', func)
            jac = jacobian(X, *args)
            
            beta, fx_dx, k = gauss_elimination(jac, func, pref)

            X -= fx_dx
            
            abs_err = np.dot(weights.T, abs(fx_dx))
            if (abs_err <= eps):
                initial= True
                break
        
        if initial == False:
            raise Exception ('The solution cannot be converged in the initial point.')
        else:
            print('Solution is converged at the initial point after {} iterations'.format(count + 1))

    while(True):

        while (True):
            
            func = original_func(X, *args)
            jac = jacobian(X, *args)
            beta, fx_dx, k = gauss_elimination(jac, func, pref)
            #### We didn't understand that check with K1 and K
          
            abs_err = np.dot(weights.T, abs(fx_dx))
            
            #### Breaking criterion of the inner loop
            if (abs_err > eps) and (nc <= ncorr):
              
                X -= fx_dx
                nc += 1
            elif (abs_err > eps) and (nc > ncorr):
                nc = 1
                break
            else:
                nc = 1
                break
    
        #### Total Iteration counts
        nout += 1
        if nout > maxout:
            print('Maximum no. of iteration reached!')
            if nout == 1:
                return X
            else:
                return Xsol
        
        if nout == 1:
            Xsol = X.copy()
        else:
            Xsol = np.c_[Xsol, X]
        print('At {0} iteration, value : {1}'.format(nout, X[-1]))
        #### Last point check
        if X[-1] > maxval:
            print('Maximum value of bifurcation parameter exceeded!')
            return Xsol
    
        #### Integration starts here
        dxdt = np.zeros(n1)             # Pre-allocation
        denom = 1/np.sqrt(1 + np.sum(beta ** 2))        
        
        dxdt[k] = 1/denom * float(ndir[k])
        
        h = hh
        
        #### Defining the integrals and manipulating the step size
        #### (We are not interested in the step-size control part)
        for i in range(n1):
            ndir[i] = 1
            if (i != k):
                dxdt[i] = beta[i] * dxdt[k]
            
            
            if (dxdt[i] < 0):
                ndir[i] = -1
            
            if (h*abs(dxdt[i]) > hmax[i]):
                madms = 0
                h = hmax[i]/abs(dxdt[i])

        while(True):      
            if ((nout > kout+3)
                    and (h*abs(dxdt[k]) > 0.8*abs(X[k] - Xsol[k, 0]))
                    and ((Xsol[k, 0] - X[k])*ndir[k] > 0)):
                madms = 0
            else:
                X, der, madms = adams(dxdt, X, madms, h, der)
                break
            
            if (h*abs(dxdt[k]) > abs(X[k] - Xsol[k, 0])):
                h = abs(X[k] - Xsol[k, 0])/abs(dxdt[k])
                kout = nout
                
            print('Otherwise: ',madms)
            X, der, madms = adams(dxdt, X, madms, h, der)
            break
    return Xsol

        
            
    
