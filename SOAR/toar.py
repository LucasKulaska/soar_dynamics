import numpy as np
from numpy.linalg import solve, norm
from math import sqrt

class TOAR:
    def __init__(self, A, B, u):
        self.A = A
        self.B = B
        self.u = u

    def procedure(self, k):
        '''
        Q = TOAR(A, B, q1, n)
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

        This python code is adapted from [3].

        References:
        [1] Yangfeng Su, and Zhaojun Bai, SOAR: a second-order Arnoldi 
            method for the solution of the quadratic eigenvalue problem,
            SIAM J. Matrix Anal. Appl., 2005, 26(3): 640-659.

        [2] Ding Lu, Yangfeng Su, and Zhaojun Bai, Stability Analysis 
            of the two-level orthogonal Arnoldi procedure,
            SIAM J. Matrix Anal. Appl., 2016. 37(1): 195--214.
        
        [3] Ding Lu, Fudan Univ. TOAR: A Two-level Orthogonal ARnoldi procedure.
            http://www.unige.ch/~dlu/toar.html

        Author:
        Lucas Kulakauskas, UFSC Brazil, 2020/03/28. '''
        # Deflation tolerences
        tol = 1e-12
        n = len(self.u)

        # Prealocate memory
        Q  = np.zeros([n, k+1])
        U1 = np.zeros([k+1, k+1])
        U2 = np.zeros([k+1, k+1])
        H  = np.zeros([k+1, k])

        # Initilization 
        u = self.u/norm(self.u)
        Q[:, 0] = u
        U1[0, 0] = 1
        rk = 1
        
        p = u
        q = np.zeros(n)
        
        # Main loop
        for i in range(k-1):
            # matrix vector product
            p = A @ p + B @ q
            alpha = norm(p)
            
            # level-one orthogonalization: MGS
            x = np.zeros(rk)
            coef = Q[:, :rk+1].T @ p
            p -= Q[:, :rk+1] @ coef
            
            # Reorthogonalization to p
            aux = 0
            if norm(p) < 0.717 * alpha:
                aux = Q[:, :rk+1].T @ p
                p -= Q[:, :rk+1] @ aux
            
            coef += aux            
            alpha = norm(p)
            
            if alpha > tol :
                nrk = rk + 1
                Q[:,nrk-1] = p / alpha
            else:
                nrk = rk
                print('BREAK')
                
            # level-two orthogonalization: MGS
            sp = coef
            sq = U1[0:rk+1,i]
            alpha = norm([sp, sq])

            h = U1[0:rk+1, :i+1].T @ sp + U2[0:rk+1, :i+1].T @ sq 
            sp -= U1[0:rk+1, :i+1] @ h
            sq -= U2[0:rk+1, :i+1] @ h
            
            # reorthogonalization to sp and sq
            aux_h = 0
            if norm([sp,sq]) < 0.717 * alpha:
                aux_h = U1[0:rk+1, :i+1].T @ sp + U2[0:rk+1, :i+1].T @ sq 
                sp -= U1[0:rk+1, :i+1] @ aux_h
                sq -= U2[0:rk+1, :i+1] @ aux_h
            
            h += aux_h
            
            beta = sqrt( norm([sp, sq])**2 + alpha**2 )
            if beta > tol:
                sp = sp / beta
                sq = sq / beta
                alpha = alpha / beta
            else:
                print('break')
            
            U1[0:rk+1, i+1] = sp
            U2[0:rk+1, i+1] = sq
            U1[rk+1, i+1] = alpha
            H[0:i+1, i] = h
            H[i+1, i] = beta
            p = Q[:, 0:rk+1] @ sp + alpha * Q[:, rk+1]
            q = Q[:, 0:rk+1] @ sq
            rk = nrk
        return Q[:,:k], U1[:k, :k], U2[:k, :k], H, rk

if __name__ == "__main__":
    import seaborn as sns
    N = 100
    k = 30
    A = np.random.rand(N,N)
    B = np.random.rand(N,N)
    u = np.random.rand(N)

    soar = TOAR(A, B, u)
    Q, U1, U2, H, rk = soar.procedure(k = k)

    ax = sns.heatmap(( Q.T @ Q - np.eye(k)), center = 0)

    ay = sns.heatmap(H, center = 0)

    Aux = np.r_[ np.c_[A,B], np.c_[np.eye(N), np.zeros_like(A)] ]
    qu = np.r_[Q @ U1, Q @ U2]