import numpy as np

# Example of initial problem.
def Problem_init(N):
    M = np.diag(np.full(N,2))+np.diag(np.full(N-1,-1),1)+np.diag(np.full(N-1,-1),-1)
    f = lambda x: np.sin(np.pi*x)
    u = lambda x: -np.sin(np.pi*x)/(np.pi**2)
    h = 1/(N+1)

    omega = 2/(1+np.sin(np.pi/(N+1)))
    f_vec = -(h**2)*f(np.linspace(h,1-h,N))
    u_vec = u(np.linspace(h,1-h,N))
    U0 = np.zeros(N)

    return M,omega,f_vec,u_vec,U0

def SOR_method(M,f_vec,U0,u_vec,omega,error=0.01):
    # Initiate variables
    k = True
    counter = 0
    N = len(M)
    U_new = np.zeros_like(U0,dtype='float')

    #Iterate using SOR method
    while k:
        counter+=1
        for i in range(N):
            U_new[i] = (1-omega)*U0[i] + omega*(f_vec[i]-(np.dot(M,U0)[i]-np.diag(M)[i]*U0[i]))/np.diag(M)[i]
            U0[i] = U_new[i] 

            # Check condition to keep looping
            check = abs(U_new-u_vec)
            if np.all(check<error):
                k = False   
        if counter==1000:
            break

    U_final = U_new
    print(f'\n{counter} steps taken to reach an error of {error}')
    print(f'U = {U_final}\n')
    return U_final
