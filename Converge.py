import numpy as np

# We need a Diagonal dominant matrix for the Gauss-Jacobi method
def Converge(A: np.ndarray) -> bool :
    k = np.sum(A,axis=1)-np.diag(A)
    converge = np.all(k<=np.diag(A))

    return converge

