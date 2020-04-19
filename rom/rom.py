import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import spsolve, splu
from numpy.linalg import svd, norm
from math import pi, floor
from statistics import median_high

from arnoldi.soar import Soar

class ROM:
    # Reduced Order Model configuration
    moment_matching = 10
    tol_svd = 1e-10
    tol_proj = 1e-5
    num_freqs = 5
    freq_step = 100

    def __init__(self, stiffness, damping, mass, force, freq):
        self.K = stiffness
        self.C = damping
        self.M = mass
        self.b = force
        self.freq = freq
    
    @property
    def norm_b(self):
        return norm(self.b)

    @property
    def number_dof(self):
        return self.K.shape[0]

    @property
    def M_array(self):
        return self.M.toarray()
    
    @property
    def C_array(self):
        return self.C.toarray()

    def full_order(self):
        solution = np.empty([self.number_dof, len(self.freq)], dtype = 'complex')
        for i, freq in enumerate(self.freq):
            omega = 2 * pi * freq
            K_dynamic = - omega**2 * self.M + 1j * omega * self.C + self.K
            solution[:, i] = spsolve(K_dynamic, self.b)
        return solution
    
    def expansion_freqs(self):
        num_freqs = self.num_freqs
        freq_step = self.freq_step

        # num_freqs must be a odd number
        if num_freqs%2 == 0:
            num_freqs =+ 1
        
        aux = floor(num_freqs / 2)
        freq_central = median_high(self.freq)
        lower_freq = freq_central - aux * freq_step

        expansion_freqs = np.zeros(num_freqs)

        for i in range(num_freqs):
            freq = i * lower_freq
            index = np.abs(self.freq - freq).argmin()
            expansion_freqs[i] = self.freq[index]
        return expansion_freqs

    def expansion_basis(self, expansion_freq):
        ## Init: defining the dynamic stiffness matrix and dynamic damping matrix
        omega = 2 * pi * expansion_freq

        # Keep K_dynamic as a sparse matrix
        K_dynamic = - omega**2 * self.M + 1j * omega * self.C + self.K

        # Turn C_dynamic into NpArray used below
        C_dynamic = 2 * 1j * omega * self.M_array + self.C_array

        ## Defining the second order Krylov subspace, apply Arnoldi procedure
        K_inv = splu(-K_dynamic) # Sparse complete LU decomposition solver

        A = K_inv.solve( C_dynamic )
        B = K_inv.solve( self.M_array )
        u = K_inv.solve( self.b )
        Q, _, _, _ = Soar.procedure(A, B, u, n = self.moment_matching)

        return Q, u

    def basis(self):
        # Init
        expansion_freqs = self.expansion_freqs()

        for i, freq in enumerate(expansion_freqs):

            Q, _ = self.expansion_basis(freq)
            if i == 0:
                basis = Q
            else:
                basis = np.c_[basis, Q]
        
        # Basis reduction via Singular value Decomposition
        u, s, vh = svd(basis)
        
        # Take the singular values that are relevant
        index = s > self.tol_svd

        index_u = np.zeros(u.shape[1], dtype = 'bool')
        index_u[:len(index)] = index

        U, Sigma, V_hermit = u[:, index_u], np.diag( s[index] ), vh[index,:]        
        return U, Sigma, V_hermit

    def projection(self):
        U, Sigma, V_hermit = self.basis()
        W  = U @ Sigma @ V_hermit
        W_hermit = W.conj().T

        K_rom = W_hermit @ self.K @ W
        C_rom = W_hermit @ self.C @ W
        M_rom = W_hermit @ self.M @ W
        b_rom = W_hermit @ self.b

        error = np.zeros_like(self.freq)
        solution = np.zeros([self.number_dof, len(self.freq)], dtype = 'complex')

        for i, freq in enumerate(self.freq):
            omega = 2 * pi * freq
            K_dynamic_rom = - omega**2 * M_rom + 1j*omega * C_rom + K_rom
            x_rom = spsolve(K_dynamic_rom, b_rom)

            x_proj = W @ x_rom

            error[i] = self.error(freq, x_proj)
            solution[:, i] = x_proj
        return solution, error

    def error(self, freq, x_proj):
        omega = 2 * pi * freq
        K_dynamic = - omega**2 * self.M + 1j * omega * self.C + self.K
        error = norm(self.b - K_dynamic @ x_proj) / self.norm_b
        return error

if __name__ == "__main__":
    num_dof = 10000
    K = np.random.rand(num_dof,num_dof)
    C = np.random.rand(num_dof,num_dof)
    M = np.random.rand(num_dof,num_dof)
    F = np.random.rand(num_dof)
    freq = np.linspace(1,2001)

    rom = ROM(K, C, M, F, freq)