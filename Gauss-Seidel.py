import numpy as np
import Converge as cvg

def Gauss_S(A,bvec,x0,error=0.001):
    # Check convergence
    s =cvg.Converge(A)
    if s:
        # Initiate variables
        k = True
        counter = 0
        N = len(A)
        xn = np.zeros_like(x0,dtype='float')

        #Iterate using GS method
        while k:
            counter+=1
            for i in range(N):
                xn[i] = (bvec[i]-(np.dot(A,x0)[i]-np.diag(A)[i]*x0[i]))/np.diag(A)[i]
                x0[i] = xn[i] 

                # Check condition to keep looping
                check = abs(np.dot(A,xn)-bvec)
                if np.all(check<error):
                    k = False   

        xf = xn
        print(f'\n{counter} steps taken to reach an error of {error}')
        print(f'x = {xf}\n')
        return xf



A = np.array([[9,4],[7,14]])
bvec = np.array([2,3])
x0 = np.array([0.,0.])


x = Gauss_S(A,bvec,x0)
