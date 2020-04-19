import numpy as np
from numpy.linalg import norm

class Mgs:
    def __init__(self):
        pass

    @staticmethod
    def basis(X):
        Y = np.zeros_like(X)
        for i, x in enumerate(X.T):
            # Initiate projection loop
            norm_init = norm(x)
            basis = Y[:,:i].T

            # first orthogonalization
            for y in basis:
                coef = y.T @ x
                x -= coef * y
            
            # reorthogonalization if need.
            if norm(x) < 0.7*norm_init:
                for y in basis:
                    coef = y.T @ x
                    x -= coef * y
            
            # Orthogonal condition
            if norm(x) < 1e-12:
                print('The', str(i)+'th', 'vector is orthogonal to the basis.')
                Y[:,i] = np.zeros_like(x)
            
            # New basis' element normalized.
            Y[:,i] = x / norm(x)
        return Y
    
    @staticmethod
    def new_element(Y,v):
        norm_init = norm(v)

        # first orthogonalization
        for y in Y.T:
            coef = y.T @ v
            v -= coef * y
        
        # reorthogonalization if need.
        if norm(v) < 0.7*norm_init:
            for y in Y.T:
                coef = y.T @ v
                v -= coef * y
        
        if norm(v) < 1e-12:
            print('The given vector is orthogonal to the given basis.')
            return Y
        return np.c_[Y,v]

if __name__ == "__main__":
    N = 100
    n = 30
    A = np.random.rand(N,N)
    B = np.random.rand(N,N)
    u = np.random.rand(N)

    # Krylov space
    V = np.zeros([N,n])
    V[:,0] = u
    V[:,1] = A @ u

    for i in range(2,n):
        V[:,i] = A @ V[:,i-1] + B @ V[:,i-2]

    Q = Mgs.basis(V)

    print(f'The program took {end - start} seconds to execute the orthonormalization.')
    print(norm ( Q.T @ Q - np.eye(N) ))
