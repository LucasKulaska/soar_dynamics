import numpy as np
from numpy.linalg import solve, norm
from math import sqrt

class Toar:
    def __init__(self, A, B, u):
        self.A = A
        self.B = B
        self.u = u

    def procedure(self, n):
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
        N = len(self.u)

        # Prealocate memory
        Q  = np.zeros([N, n+1])
        U1 = np.zeros([n+1, n+1])
        U2 = np.zeros([n+1, n+1])
        H  = np.zeros([n+1, n])

        # Initilization 
        p = self.u/norm(self.u)
        Q[:, 0] = p
        U1[0, 0] = 1
        
        q = np.zeros(N)
        
        # Main loop
        for j in range(n-1):
            # Recurrence role
            p = self.A @ p + self.B @ q

            norm_init = norm(p)
            basis = Q[:,:j+1].T 
            ## Modified Gram Schmidt procedure
            # first orthogonalization
            coef = np.zeros(j+1)
            for index, v in enumerate(basis):
                # Projection coeficients and projection subtraction
                coef[index] = v.T @ p
                p -= coef[index] * v
            
            ## Reorthogonalization if needed
            if norm(p) < 0.7 * norm_init:
                sum_coef  = np.zeros_like(coef)
                for index, v in enumerate(basis):
                    sum_coef[index] = v.T @ p
                    p -= sum_coef[index] * v
                coef += sum_coef
            
            norm_proj = norm(p)
            if norm_proj > tol:
                Q[:,j+1] = p / norm_proj
            else:
                print('Break at Level-one orthogonalization')
                
            # level-two orthogonalization: MGS
            s = coef
            u = U1[:j+1,j]

            coef = np.zeros(j+1)
            for index in range(j+1):
                coef[index] = U1[ :j+1, index ].T @ s + U2[ :j+1, index ].T @ u
                s -= coef * U1[ :j+1, index ]
                u -= coef * U2[ :j+1, index ]
            
            beta = sqrt( norm(s)**2 + norm(u)**2 + norm_proj**2 )
            if beta > tol:
                s = s / beta
                u = u / beta
                alpha = norm_proj / beta
            else:
                print('Break at Level-two orthogonalization')

            U1[:j+1, j+1] = s
            U1[ j+1, j+1] = alpha

            U2[:j+1, j+1] = u

            H[:j+1, j] = coef
            H[ j+1, j] = beta

            p = Q[:, :j+1] @ s + alpha * Q[:, j+1]
            q = Q[:, :j+1] @ u
        return Q[:,:n], U1[:n, :n], U2[:n, :n], H[:n, :n-1]

if __name__ == "__main__":
    # Example and test setup
    N = 100
    n = 30
    A = np.random.rand(N,N)
    B = np.random.rand(N,N)
    u = np.random.rand(N)

    toar = Toar(A, B, u)
    Q, U1, U2, H = toar.procedure(n = n)

    print( norm( Q.T @ Q - np.eye(n) ) ) # must be zero!

    # Error matrix based onde equation (3.1) and (3.2) from [2]
    Vn = np.r_[Q @ U1, Q @ U2]
    L = np.r_[ np.c_[A, B], np.c_[np.eye(N), np.zeros_like(B)]]

    print( norm( L @ Vn[:, :n-1] - Vn @ H) ) # must be zero!