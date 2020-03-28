import numpy as np
from numpy.linalg import solve, norm

class SOAR:
    def __init__(self, A, B, u):
        self.A = A
        self.B = B
        self.u = u

    @staticmethod
    def GramSchmidt_basis(X):
        Y = np.zeros_like(X)
        for i, x in enumerate(X.T):
            coef = Y.T @ v
            x -= Y @ coef
            Y[:,i] = x / np.linalg.norm(x)
        return Y
    
    @staticmethod
    def GramSchmidt_projection(X, v, normalized = True):
        coef = X.T @ v
        if not normalized:
            inv_norm = np.zeros(Y.shape[1])
            for i, x in enumerate(X.T):
                inv_norm[i] = 1 / np.dot(x,x)
            coef = np.multiply(coef, inv_norm)
        v -= X @ coef
        return np.c_[X,v]

    def procedure(self, n):
        q = self.u / norm(self.u)
        Q = np.zeros([len(q),n])
        Q[:,0] = q
        T = np.zeros([n,n])
        f = np.zeros_like(q)
        F = None
        A = self.A
        B = self.B
        deflation = []
        for j in range(n-1):
            r = A @ q + B @ f
            coef = Q[:,:j+1].T @ r
            r -= Q[:,:j+1] @ coef
            T[:len(coef), j] = coef
            r_norm = norm(r)
            if r_norm > 1e-10:
                T[j+1,j] = r_norm
                Q[:,j+1] = r / r_norm
                eye = np.zeros(j+1)
                eye[j] = 1
                v_aux = solve(T[1:j+2,0:j+1], eye )
                f = Q[:,:j+1] @ v_aux
            else:
                deflation.append(j)
                T[j+1,j] = 1
                Q[:,j+1] = np.zeros(len(r))
                eye = np.zeros(j+1)
                eye[j] = 1
                v_aux = solve(T[1:j+2,0:j+1], eye )
                f = Q[:,:j+1] @ v_aux
                if F == None:
                    F = f
                else:
                    coef_f = F.T @ f
                    inv_norm = np.zeros(Y.shape[1])
                    for i, x in enumerate(F.T):
                        inv_norm[i] = 1 / np.dot(x,x)
                    coef_f = np.multiply(coef, inv_norm)
                    f_0 = f - F @ coef_f
                    if np.linalg.norm(f_0) < 1e-10:
                        break
                    else:
                        F = np.c_[F,f]
        j = n
        r = A @ q + B @ f
        coef = Q[:,:j+1].T @ r
        r -= Q[:,:j+1] @ coef
        r_norm = norm(r)
        q = r
        return Q, T, deflation, q, r_norm

if __name__ == "__main__":
    import seaborn as sns
    N = 100
    n = 30
    A = np.random.rand(N,N)
    B = np.random.rand(N,N)
    u = np.random.rand(N)

    soar = SOAR(A, B, u)
    Q, T, deflation, q, r_norm  = soar.procedure(n = n)

    ax = sns.heatmap(( Q.T @ Q - np.eye(n)), center = 0)

    ay = sns.heatmap(T, center = 0)

    eye = np.zeros(n)
    eye[n-1] = 1

    T_inv = np.linalg.inv(T)

    S = [   [np.zeros([n-1 , 1]), T_inv[1:n, 0:n-1],
            [0                  , np.zeros([ 1, n-1])]]]

    norm ( A @ Q + B @ Q @ S - Q @ T - r_norm * q @ eye.T )

    deflation
