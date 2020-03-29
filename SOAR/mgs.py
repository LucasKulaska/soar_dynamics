import numpy as np
from numpy.linalg import norm

class MGS:
    def __init__(self):
        pass

    @staticmethod
    def basis(X):
        Y = np.zeros_like(X)
        for i, x in enumerate(X.T):
            coef = Y.T @ x
            x -= Y @ coef
            Y[:,i] = x / norm(x)
        return Y
    
    @staticmethod
    def projection(X,v):

        coef = np.zeros(X.shape[1])
        for i, x in enumerate(X.T):
            coef[i] = x.T @ v / np.dot(x,x)
        
        v -= X @ coef
        return np.c_[X,v]

if __name__ == "__main__":
    N = 100
    A = np.random.rand(N,N)
    u = np.random.rand(N)

    mgs = MGS(A, u)

    Q = mgs.projection()

    norm ( Q.T @ Q - np.eye(N) )
