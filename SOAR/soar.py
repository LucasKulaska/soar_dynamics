import numpy as np
from numpy.linalg import solve, norm

class SOAR:
    def __init__(self, A, B, u):
        self.A = A
        self.B = B
        self.u = u

    def procedure(self, n):
        ''' Q = SOAR(A, B, q1, n)
        computes an orthonormal basis Q of the second-order Krylov subspace:
            span{ Q }   = G_n(A,B,v_0).
                        = {v_0, v_1, v_2, ..., v_{n-1}}
            with    v_0 = q1 / norm(q1)
                    v_1 = A @ v_0
                    v_j = A @ v_{j-1} + B @ v_{j-2}
        using space-efficient SOAR procedure.

        Parameters:
        A, B :   N-dimensional square matrices
        q1   :   starting vector, of size N-by-1
        n    :   dimension of second-order Krylov subspace.

        Returns:
        Q   :   N-by-n matrix whose the vector collumns form an orthonormal basis
                of the second-order Krylov subspace G_n(A,B,v_0).
        P   :   N-by-n matrix.
        T   :   n-by-n upper Hessenberg matrix.

        The following compact Krylov decomposition holds:
            A @ Q + B @ P = Q @ T + t_{n+1,n} * q_{n+1} @ e_n.T,
                        Q = P @ T + t_{n+1,n} * P_{n+1} @ e_n.T

        where, e_n is the nth collumn of n-dimensional identity matrix.

        This python code is adapted from [2].

        References:
        [1] Yangfeng Su, and Zhaojun Bai, SOAR: a second-order Arnoldi 
            method for the solution of the quadratic eigenvalue problem,
            SIAM J. Matrix Anal. Appl., 2005, 26(3): 640-659.
        
        [2] Ding Lu, Fudan Univ. TOAR: A Two-level Orthogonal ARnoldi procedure.
            http://www.unige.ch/~dlu/toar.html

        Author:
        Lucas Kulakauskas, UFSC Brazil, 2020/03/28. '''
        # Initialize
        tol = 1e-12 # breakdown threshold
        N = len(self.u)
        q = self.u / norm(self.u)
        f = np.zeros_like(q)
        A = self.A
        B = self.B

        # Prealocate memory
        Q = np.zeros([N,n])
        P = np.zeros([N,n])
        T = np.zeros([n,n])

        Q[:,0] = q
        P[:,0] = f
        deflation = [] # Deflation index list.

        for j in range(n-1):
            # Recurrence role
            r = A @ Q[:,j] + B @ P[:,j]
            aux = norm(r)

            # Gram Schmidt orthogonalization
            coef = Q[:,:j+1].T @ r
            r -= Q[:,:j+1] @ coef
            T[:j+1, j] = coef
            

            # Reorthogonalization, if needed.
            if norm(r) < 0.7 * aux:
                # Second Gram Schmidt orthogonalization
                coef = Q[:,:j+1].T @ r
                r -= Q[:,:j+1] @ coef
                T[:j+1, j] += coef
            
            r_norm = norm(r)
            T[j+1,j] = r_norm

            # Check for breackdown
            if r_norm > tol:
                Q[:,j+1] = r / r_norm
                e_j = np.zeros(j+1)
                e_j[j] = 1
                v_aux = solve( T[1:j+2,0:j+1], e_j )
                f = Q[:,:j+1] @ v_aux
            else:
                # Deflation reset
                T[j+1,j] = 1
                Q[:,j+1] = np.zeros(N)
                e_j = np.zeros(j+1)
                e_j[j] = 1
                v_aux = solve( T[1:j+2,0:j+1], e_j )
                f = Q[:,:j+1] @ v_aux

                # Deflation verification
                coef_f = np.zeros(P[:,:j+1].shape[1])
                for i, p in enumerate(P[:,:j+1].T):
                    coef_f[i] = P[:,i].T @ f / np.dot(p,p)
                f_proj = f - P[:,:j+1] @ coef_f
                if norm(f_proj) > tol:
                    deflation.append(j)
                else:
                    print('SOAR lucky breackdown.')
                    break
            P[:,j+1] = f
        return Q, T, P, deflation

if __name__ == "__main__":
    import seaborn as sns
    N = 100
    n = 30
    A = np.random.rand(N,N)
    B = np.random.rand(N,N)
    u = np.random.rand(N)

    soar = SOAR(A, B, u)
    Q, T, P, deflation = soar.procedure(n = n)

    ax = sns.heatmap(( Q.T @ Q - np.eye(n)), center = 0)

    ay = sns.heatmap(T, center = 0)

    e_n = np.zeros([n,1])
    e_n[n-1] = 1

    r = A @ Q[:,n-1] + B @ P[:,n-1]

    # Gram Schmidt orthogonalization
    coef = Q.T @ r
    r -= Q @ coef
    r = r.reshape([N,1]) 

    norm ( A @ Q + B @ P - Q @ T - r @ e_n.T )

    deflation
