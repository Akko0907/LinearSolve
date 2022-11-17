import numpy as np
import Converge as cvg

def GaussJ(A,bvec,x0,error=0.01):
    # Check convergence
    s =cvg.Converge(A)
    if s:
        counter = 0
        k = True
        while k:
            counter+=1
            #Iterate using GJ method
            xn = (bvec-(np.dot(A,x0)-np.diag(A)*x0))/np.diag(A)
            x0 = xn

            # Check condition to keep looping
            check = abs(np.dot(A,xn)-bvec)
            if np.all(check<error):
                k = False

        xf = xn
        print(f'\n{counter} steps taken to reach an error of {error}')
        print(f'x = {xf}\n')
        return xf
   
    else:
        print("sorry, but it doesn't seems to converge :c ")


A = np.array([[9,4],[7,14]])
bvec = np.array([2,3])
x0 = np.array([0,0])

x = GaussJ(A,bvec,x0)

