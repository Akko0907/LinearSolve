import numpy as np
from . import Converge as cvg


#============================================================================
#============================================================================


def GaussE(A: np.ndarray, bvec: np.ndarray) -> np.ndarray:
    ''' Linear equation solve with Gauss-Elimination method.
    Receives a matrix with the equations coefficients and
    a vector of the independent terms '''

    # Gen the augmented matrix
    A = np.column_stack((A,bvec))
    dim1,dim2 = np.shape(A)

    # Triangulization of the matrix
    for i in range(dim1-1):
        pivot = A[i][i]
        count = 0
        while (pivot==0) and (count<=dim1-i):
            B = A[i]
            A[i] = A[i+1]
            A[i+1] = B
            pivot = A[i][i]
        if pivot==0:
            break

        m_ij = (A.T[i]/A[i][i])
        A[i+1:] = np.outer( m_ij, A[i] )[i+1:] - A[i+1:]

    # Solve for the variables
    s = np.zeros(dim1)
    for i in range(0,dim1):
        if np.any(A[dim1-1-i]!=0):
            b = A.T[dim2-1][dim1-1-i]
            a = A[dim1-1-i][dim2-2-i]
            k = np.dot(A[dim1-1-i][:-1],s[::-1])

            s[i] = (b-k)/a

    s = s[::-1]
    return s


#============================================================================
#============================================================================


def GaussJ(A: np.ndarray, bvec: np.ndarray, 
            x0: np.ndarray, error: float=0.001) -> np.ndarray:
    ''' Linear equation solve with Gauss-Jacobi method.
    Receives a matrix with the equations coefficients,
    a vector of the independent terms and a first kick value
    for the solution '''

    # Check convergence
    s =cvg.Converge(A)
    if s:
        counter = 0
        loop = True
        while loop:
            counter+=1
            #Iterate using GJ method
            xn = (bvec-(np.dot(A,x0)-np.diag(A)*x0))/np.diag(A)
            x0 = xn

            # Check condition to keep looping
            check = abs(np.dot(A,xn)-bvec)
            if np.all(check<error):
                loop = False

        xf = xn
        print(f'\n{counter} steps taken to reach an error of {error}')
        print(f'x = {xf}\n')
        return xf
   
    else:
        print("sorry, but it doesn't seems to converge :c ")


#============================================================================
#============================================================================


def GaussS(A: np.ndarray, bvec: np.ndarray, 
            x0: np.ndarray, error: float=0.001) -> np.ndarray:
    ''' Linear equation solve with Gauss-Seidel method.
    Receives a matrix with the equations coefficients,
    a vector of the independent terms and a first kick value
    for the solution '''

    # Check convergence
    s =cvg.Converge(A)
    if s:
        # Initiate variables
        loop = True
        counter = 0
        N = len(A)
        xn = np.zeros_like(x0,dtype='float')

        #Iterate using GS method
        while loop:
            counter+=1
            for i in range(N):
                xn[i] = (bvec[i]-(np.dot(A,x0)[i]-np.diag(A)[i]*x0[i]))/np.diag(A)[i]
                x0[i] = xn[i] 

                # Check condition to keep looping
                check = abs(np.dot(A,xn)-bvec)
                if np.all(check<error):
                    loop = False   

        xf = xn
        print(f'\n{counter} steps taken to reach an error of {error}')
        print(f'x = {xf}\n')
        return xf


#============================================================================
#============================================================================


def SOR_method(A: np.ndarray,bvec: np.ndarray,
               x0: np.ndarray, omega: float,
               error: float=0.01) -> np.ndarray:
               
    # Initiate variables
    loop = True
    counter = 0
    N = len(A)
    xn = np.zeros_like(x0,dtype='float')

    #Iterate using SOR method
    while loop:
        counter+=1
        for i in range(N):
            xn[i] = (1-omega)*x0[i] + omega*(bvec[i]-(np.dot(A,x0)[i]-np.diag(A)[i]*x0[i]))/np.diag(A)[i]
            x0[i] = xn[i] 

            # Check condition to keep looping
            check = abs(np.dot(A,xn)-bvec)
            if np.all(check<error):
                loop = False   
        if counter==1000:
            break

    xf = xn
    print(f'\n{counter} steps taken to reach an error of {error}')
    print(f'U = {xf}\n')
    return xf
