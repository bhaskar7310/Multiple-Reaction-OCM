subroutine derpar(func, jac, n, x, xlow, xupp, eps, w, initial, itin,    &
                  hh, hmax, pref, ndir, e, mxadms, ncorr, ncrad, nout,   & 
                  out, maxout)                            
! Obtaining of dependence of solution x(alpha) of equation
! f(x, alpha) = 0 on parameter alpha by modified method of
! differentiation with respect to parameter

! n- Number of unknowns x(i)
! x(1),....., x(n) - Initial values of x(i), after return final values 
!                    of x(i)
! x(n+1) - Initial value of parameter alpha, after return final values
!          of alpha
! xlow - Lower bound for alpha, x(n+1)
!        If xlow or xupp is exceeded, then end of 
!        derpar and maxout= -2 after return
! eps - Accuracy desired in newton iteration for sum of 
!       (w(i)*abs(xnew(i) - xold(i))), i= 1,....,n+1
! w(1),......,W(n+1) - Weights used in termination criterion of Newton
!                      process
! initial - If (initial .ne. 0) then several steps in Newton iteration
!           are made before computation in order to increase accuracy
!           of initial point.
!           If (initial. eq. 1 .and. eps-accuracy is not fulfilled in 
!           itin iterations) then return
!           if (initial. eq. 2) then always return after initial Newton
!           iteration, results are in x
!           if (initial. eq. 3) then continue in derpar after initial
!           Newton iterations
!           if (initial. eq. 0) then no initial newton iteration reqd.
! itin - Maximum number of initial newton iterations. If eps-accuracy
!        is not fulfilled in itin iterations then itin= -1 after return
! hh - Integration step along arc-length of solution locus.
! hmax(1),......,hmax(n+1) - Upper bounds for increments of x(i) in 
!                            one integration step (approximation only)
! pref(1),......,pref(n+1) - Preference numbers (Explanation see in
!                            subr. gause)
! ndir(1),......,ndir(n+1) - Initial change of x(i) is positive along
!                            solution locus (curve) if ndir(i) = 1 and
!                            negative if ndir(i) = -1
! e - criterion for test on closed curve, if (
!     (sum of (w(i) * abs(x(i) - xinitial(i))), i=1,....,n+1) . le. e)
!     then closed curved may be expected
! mxadms - maximal order of Adams-Bashforth formula,
!          1 .le. mxadms .le. 4
! ncorr - maximal number of newton corrections after prediction by
!         Adams-Bashforth method.
! ncrad - If (ncrad. ne. 0) then additional Newton correction without
!         new computation of Jacobian matrix is used.
! nout - After return number of calculated points on the curve x(alpha),
!        nout. le. maxout
! out(j,1), out(j,2),....., out(j,n+1) - j-th point x(1),...,x(n), alpha
!                                        on curve x(alpha)
! out(j,n+2) - Value of sqrt(sum of squares of f).
!              If (out(j,n+2) .lt. 0.0) then abs(out(j, n+2))
!              corresponds to x and alpha not exactly, because
!              additional Newton correction was used (ncrad. ne. 0).
!              Values f(i) are not computed for x and alpha printed and/
!              or stored and therefore last time computed value of 
!              sqrt(sum of squares of f) is on our disposal only.
! maxout - Maximal number of calculated points on curve, x(alpha).
!          If maxout after return equals to-
!              -1 - then closed curve x(alpha) may be expected
!              -2 - then bound xlow or xupp may be exceeded
!              -3 - then singular jacobian matrix occured, its rank is
!                   .lt. N.

   implicit none
   
   ! Incoming and outgoing variable type declarations
   integer, intent(in) :: n, initial, mxadms, ncorr
   integer, intent(in) :: ncrad
   double precision, intent(in) :: eps, hh, e 
   
   integer, intent(inout) :: maxout, itin
   integer, intent(out) :: nout
   double precision, intent(in) :: xlow, xupp
   double precision, dimension(n+1), intent(in) :: w, hmax, pref
   double precision, dimension(n+1), intent(inout) :: x
   integer, dimension(n+1), intent(inout) :: ndir
   double precision, dimension(maxout, n+2), intent(out) :: out 
   external :: func
   external :: jac

   ! Local variable type declarations
   integer :: n1, l, ll, fileunit, i, j
   integer :: kout, madms, nc, k1
   double precision :: squar, p, dxk2, h
   double precision, dimension(n+1) :: f
   double precision, dimension(n, n+1) :: g
   double precision, dimension(n+1) :: dxdt
   double precision, dimension(mxadms, n+1) :: der
   logical :: initial_conv

   ! Variables required for the Gausse Elimination subroutine
   integer :: m, k
   double precision, dimension(n+1) :: beta

   ! Assigning local variables
   n1 = n + 1
   fileunit = 3
   dxdt = 0.0
   der = 0.0
   ! Initial Newton iterations
   initialif: if (initial .ne. 0) then
      
      initial_conv = .False.
      newtoniter: do l = 1, itin
         !call fctn(n, x, f, g)
         call func(n, x, f)
         call jac(n, x, g)

         squar = 0
         do i = 1, n
            squar = squar + f(i)**2
         end do
         
         ll = l-1
         squar = sqrt(squar)
         
         call gause(n, g, f, m, pref, beta, k)
         
         ! If Jacobian is singular then abort calculation
         singularjac: if (m == 0) then
            maxout = -3
            write(*,*) 'Singular Jacobian encountered'
            return
         end if singularjac
         
         ! Calculating error
         p = 0.0
         newerror: do j = 1, n1
            x(j) = x(j) - f(j)
            p = p + abs(f(j))*w(j)
         end do newerror

         ! If solution converges
         if (p .le. eps) then
            initial_conv = .True.
            if (initial == 2) return
            exit
         end if
      end do newtoniter
         
      ! Maximum number of initial iterations reached
      notconv: if (.not. initial_conv) then   
         itin = -1
         
         ! Return since the first iteration didn't converge
         if (initial == 1) return

         ! initial = 2 always return after initial iterations
         if (initial == 2) return
      end if notconv

   end if initialif
   
   ! After Initial newton iterations
   ! Assigning values to variables (type integer)
   kout = 0
   nout = 0
   madms = 0
   nc = 1
   k1 = 0

   integratorloop: do while (.True.) 
      newton: do while (.True.) 
         
         call func(n, x, f)
         call jac(n, x, g)
         squar = 0.0
         do i = 1, n
            squar = squar + f(i)**2
         end do
         
         call gause(n, g, f, m, pref, beta, k)
         ! If jacobian is singular
         if (m == 0) then
            write(*,*) 'Singular Jacobian encountered'
            maxout = -3
            return
         end if

         ! Change of independent variables (its index = k now)
         if (k1 /= k) then
            madms = 0
            k1 = k
         end if

         ! Something to deal with Newton corrections 
         ! without Jacobian calculation
         squar = sqrt(squar)
         if (ncrad .eq. 1) then
            squar = -squar
         end if

         ! Calculating error
         p = 0.0
         do j = 1, n1
            p = p + abs(f(j))*w(j)
         end do

         ! If Error is greater than tolerance and iter count is less 
         ! than maximum allowed iterative corrections
         if ((p .gt. eps) .and. (nc .lt. ncorr)) then
               ! One iteration in Newton method
               do i = 1, n1
                  x(i) = x(i) - f(i)
               end do
               nc = nc + 1
         else
            exit
         end if
      end do newton
      write(*,*) 'K = ', k

      ! Maximum allowed iterative correction is not equal to 0 and is
      ! exceeded, or 
      !if ((ncorr /= 0) .and. (p .gt. eps)) write(*, 99995) ncorr, p
      nc = 1
      
      ! Additional Newton correction with out jacobian calculation
      if (ncrad .ne. 0) then
         do i = 1, n1
            x(i) = x(i) - f(i)
         end do
      end if
      
      ! Done with Newton corrections, now storing and displaying the
      ! calculated new point
      nout = nout + 1
      
      ! If array out is requested
      if (nout .gt. maxout) then
         write(*,*) 'Maximum number of output reached'
         return
      end if
      do i = 1, n1
         out(nout, i) = x(i)
      end do
      out(nout, n+2) = squar
      write(*,*) 'At', nout, '-th iteration, value: ', out(nout, n+1)

      ! Ensuring that the calculated values are within bounds
      if ((x(n+1) .lt. xlow) .or. (x(n+1) .gt. xupp)) then
         write(*,*) 'Lower or upper limits are exceeded'
         maxout = -2 ! Statement label: 300
         return
      end if

      ! Checking for closed curves
      if (nout .gt. 3) then
         p = 0.0
         do i = 1, n1
            p = p + w(i) * abs(x(i) - out(1,i))
         end do
         
         ! Closed curve may be expected (Statement label: 290)
         if (p .le. e) then
            maxout = -1
            write(*,*) 'Execution stopped because of closed curve'
            return
         end if
      end if

      ! Preparing the functions for integration
      dxk2 = 1.0 ! The denominator
      do i = 1, n1
         dxk2 = dxk2  + beta(i) ** 2
      end do
      ! Derivative of independent variable x(k) with respect to 
      ! arc length of solution is computed here
      dxdt(k) = 1.0/sqrt(dxk2) * float(ndir(k))
      
      h = hh
      do i = 1, n1
         ndir(i) = 1
         ! Calculating the derivative 
         if (i /= k) then
            dxdt(i) = beta(i) * dxdt(k)
         end if
         ! Checking the direction of change
         if (dxdt(i) .lt. 0) ndir(i) = -1
         ! Step size control (probably)
         if (h*abs(dxdt(i)) .gt. hmax(i)) then
            madms = 0
            h = hmax(i)/abs(dxdt(i))
         end if
      end do

      ! Calling the Adams-Bashforth integrator
      do while(.True.)
         if ((nout .gt. kout+3) &
            .and. (h*abs(dxdt(k)) .gt. 0.8*abs(x(k) - out(1,k))) &
            .and. ((out(1,k) - x(k))*float(ndir(k)) .gt. 0.0) &
            ) then
            madms = 0
         else
            write(*,*) h
            write(*,*) madms
            call adams(n, dxdt, der, madms, h, x, mxadms)
            exit
         end if

         if (h*abs(dxdt(k)) .gt. abs(x(k) - out(1,k))) then
            h = abs(x(k) - out(1,k))/abs(dxdt(k))
            kout = nout
         end if
         write(*,*) h
         write(*,*) madms
         call adams(n, dxdt, der, madms, h, x, mxadms)
         exit
      end do

      
   end do integratorloop

end subroutine derpar
                
         
subroutine adams(n, d, der, madms, h, x, mxadms)
   implicit none

   ! Variable type declaration
   integer, intent(in) :: n
   double precision, dimension(n+1), intent(in) :: d
   integer, intent(in) :: mxadms
   double precision, intent(in) :: h
   
   double precision, dimension(mxadms, n+1), intent(inout) :: der
   integer, intent(inout) :: madms
   double precision, dimension(n+1), intent(inout) :: x

   ! Local variable type declaration
   !double precision, dimension(4, n+1) :: der
   integer :: n1, i, j
   
   ! Adams-bashforth methods
   n1 = n + 1
   do i = 3,1,-1
      do j = 1,n1
         der(i+1, j) = der(i,j)
      end do
   end do

   madms = madms + 1
   if (madms .gt. mxadms) madms = mxadms
   if (madms .gt. 4) madms = 4

   do i = 1, n1
      der(1, i) = d(i)
      
      select case (madms)
        
         case (1)
           x(i) = x(i) + h*der(1,i)
         case (2)
            x(i) = x(i) + 0.5*h*(3.0*der(1,i) - der(2,i))
         case (3)
            x(i) = x(i) + h*(23.0*der(1,i) - 16.0*der(2,i) &
                   + 5.0*der(3,i))/12.0
         case(4)
            x(i) = x(i) + h*(55.0*der(1,i) - 59.0*der(2,i) &
                   + 37.0*der(3,i) - 9.0*der(4,i))/24.0
         case default
            write(*,*)'Invalid order of integration!'
            stop
            return
      end select
   end do
end subroutine adams

subroutine gause(n, A, B, m, pref, beta, k)

! Solution of n linear equations for n+1 unknowns
! Based on gaussian elimination with pivoting
! n - Number of equations
! A - n x (n+1) matrix system
! B - Right hand sides
! m - If (m .eq. 0) after return then rank(A) .lt. n
! pref(i) - Preference number for x(i) to be independent variable
!           0.0 .le. pref(i) .le. 1.0, the lower is pref(i), the higher
!           is preference of x(i).
! beta(i) - Coefficients in explicit dependences obtained in form
!           x(i) = b(i) + beta(i)*x(k), i .ne. k
! k- Resulting index of independent variable
   
   implicit none

   ! Variable type declaration
   integer, intent(in) :: n
   double precision, dimension(n, n+1), intent(inout) :: A
   double precision, dimension(n+1), intent(inout) :: B
   double precision, dimension(n+1), intent(in) :: pref
   integer, intent(out) :: m
   double precision, dimension(n+1), intent(out) :: beta
   integer, intent(out) :: k

   ! Local variable type declaration
   integer :: n1, id, i, ir, is, j
   integer, dimension(n+1) :: irk, irr
   double precision, dimension(n+1) :: y, x
   double precision :: p, amax
   
   ! Initializing variables
   n1 = n + 1
   id = 1
   m = 1
   do i = 1, n1
      irk(i) = 0
      irr(i) = 0
   end do
   x = 0.0
   y = 0.0

   ! Gausse Elimination section
   do while (id .le. n) 
      ir = 1
      is = 1
      amax = 0.0
      
      ! Finding the maximum element based on the preference numbers
      do i = 1, n
         if (irr(i) .eq. 0) then
            do j = 1,n1
               p = pref(j) * abs(A(i,j))
               if ((p-amax) .gt. 0.0) then
                  ir  = i
                  is = j
                  amax = p
               end if
            end do
         end if
      end do
      
      ! Checking for singularities
      if (amax .eq. 0.0) then
         m = 0
         return
      end if

      ! Row and column operations on the non-singular matrix A
      irr(ir) = is
      do i = 1, n
         if ((i .ne. ir) .and. (A(i, is) .ne. 0.0)) then
            p = A(i, is)/A(ir, is)
            do j = 1, n1
               A(i, j) = A(i, j) - p*A(ir, j)
            end do
            A(i, is) = 0.0
            B(i) = B(i) - p*B(ir)
         end if
      end do
      
      id = id + 1
   end do

   ! Back substitution
   do i = 1, n
      ir = irr(i)
      x(ir) = B(i)/A(i, ir)
      irk(ir) = 1
   end do

   do k = 1, n1
      if (irk(k) .eq. 0) then
         do i = 1,n
            ir = irr(i)
            y(ir) = -A(i, k) / A(i, ir)
         end do
         exit
      end if
   end do

   do i = 1, n1
      B(i) = x(i)
      beta(i) = y(i)
   end do
   B(k) = 0.0
   beta(k) = 0.0

end subroutine gause
