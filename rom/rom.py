import numpy as np
from scipy.linalg import lu_factor, lu_solve, solve
from scipy.sparse.linalg import spsolve, splu
from numpy.linalg import svd, norm, pinv, lstsq
from math import pi, floor
from statistics import median_low

from arnoldi.soar import Soar

class ROM:
    # Reduced Order Model configuration


    def __init__(self, stiffness, damping, mass, force, freq, **kwargs):
        self.K = stiffness
        self.C = damping
        self.M = mass
        self.b = force
        self.freq = freq

        # Config
        self.moment_matching = kwargs.get('moment_matching', 10)
        self.tol_svd = kwargs.get('tol_svd', 1e-10)
        self.tol_proj = kwargs.get('tol_proj', 1e-3)
        self.num_freqs = kwargs.get('num_freqs', 5)
        self.freq_step = kwargs.get('freq_step', 100)

        if self.num_freqs == 1:
            self.exp_freqs  = self.freq_central
        else:
            steps = self.freq_step * np.arange(-self.num_freqs+1,1)

            self.exp_freqs  = self.freq_central + steps

            for i, freq in enumerate(self.exp_freqs):
                index = np.abs(self.freq - freq).argmin()
                self.exp_freqs[i] = self.freq[index]

        self.exp_basis = {}
        self.solution = {}
        self.proj_error = {}
    
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

    @property
    def freq_central(self):
        return median_low(self.freq)

    def full_order(self):
        solution = np.empty([self.number_dof, len(self.freq)], dtype = 'complex')
        for i, freq in enumerate(self.freq):
            omega = 2 * pi * freq
            K_dynamic = - omega**2 * self.M + 1j * omega * self.C + self.K
            solution[:, i] = spsolve(K_dynamic, self.b)
        return solution
    
    def strategy(self, freq):

        if self.num_freqs == 1:
            self.exp_freqs = freq
        else:
            freq_central = self.freq_central
            freq_step = self.freq_step

            # add frequency where tol wasn't reached to expansion frequencies 
            self.exp_freqs = np.r_[self.exp_freqs, freq]

            # Take out the further freq from the actual solving frequency
            index = np.abs(self.exp_freqs - freq).argmax()
            self.exp_freqs = np.delete(self.exp_freqs, index)


            if freq <= freq_central:
                index = self.exp_freqs.argmax()
                self.exp_freqs = np.delete(self.exp_freqs, index)
                new_freq = freq - freq_step
                index = np.abs(self.freq - new_freq).argmin()

                if self.freq[index] in self.exp_freqs:
                    mid_freq = self.freq[index] + freq
                    index = np.abs(self.freq - mid_freq).argmin()
                    self.exp_freqs = np.r_[self.exp_freqs, self.freq[index]]
                else:
                    self.exp_freqs = np.r_[self.exp_freqs, self.freq[index]]
            else:
                index = self.exp_freqs.argmin()
                self.exp_freqs = np.delete(self.exp_freqs, index)
                new_freq = freq + freq_step
                index = np.abs(self.freq - new_freq).argmin()

                if self.freq[index] in self.exp_freqs:
                    mid_freq = self.freq[index] + freq
                    index = np.abs(self.freq - mid_freq).argmin()
                    self.exp_freqs = np.r_[self.exp_freqs, self.freq[index]]
                else:
                    self.exp_freqs = np.r_[self.exp_freqs, self.freq[index]]

    def expansion_basis(self, expansion_freq):
        ## Init: defining the dynamic stiffness matrix and dynamic damping matrix
        omega = 2 * pi * expansion_freq

        # Keep K_dynamic as a sparse matrix
        K_dynamic = - omega**2 * self.M + 1j * omega * self.C + self.K

        # Turn C_dynamic into NpArray used below
        C_dynamic = 2 * 1j * omega * self.M + self.C

        ## Defining the second order Krylov subspace, apply Arnoldi procedure
        K_inv = splu(-K_dynamic) # Sparse complete LU decomposition solver

        A = lambda v: K_inv.solve( C_dynamic @ v )
        B = lambda v: K_inv.solve( self.M @ v )
        u = - K_inv.solve( self.b )
        Q, _, _, _ = Soar.procedure(A, B, u, n = self.moment_matching)

        return Q, u

    def basis(self):
        # Init

        if self.num_freqs == 1:
            f = self.exp_freqs
            Q, u = self.expansion_basis(f)
            # save expansion basis already calculated
            self.exp_basis.update({f : Q})
            self.solution.update({f : u})
            self.proj_error.update({f : 0})
            return Q
        else:
            for i, f in enumerate(self.exp_freqs):
                if f in self.exp_basis:
                    Q = self.exp_basis[f]
                
                else:
                    Q, u = self.expansion_basis(f)
                    # save expansion basis already calculated
                    self.exp_basis.update({f : Q})
                    self.solution.update({f : u})
                    self.proj_error.update({f : 0})
                    
                if i == 0:
                    basis = Q
                else:
                    basis = np.c_[basis, Q]
            
            # Basis reduction via Singular value Decomposition
            u, s, vh = svd(basis)
            print(s)
            
            # Take the singular values that are relevant
            index = s > self.tol_svd

            index_u = np.zeros(u.shape[1], dtype = 'bool')
            index_u[:len(index)] = index

            U, Sigma, V_hermit = u[:, index_u], np.diag( s[index] ), vh[index,:]
            W  = U @ Sigma @ V_hermit 
            return W
    
    def projection_step(self, freq, M_rom, C_rom, K_rom, b_rom, W ):
        omega = 2 * pi * freq
        K_dynamic_rom = - omega**2 * M_rom + 1j*omega * C_rom + K_rom

        x_rom = solve(K_dynamic_rom, b_rom)

        x_proj = W @ x_rom

        error = self.error(freq, x_proj)
        if error < self.tol_proj:
            self.proj_error.update({freq : error})
            self.solution.update({freq : x_proj})
        return error

    def projection(self):

        error = 0
        W = self.basis()
        W_hermit = W.conj().T

        K_rom = W_hermit @ self.K @ W
        C_rom = W_hermit @ self.C @ W
        M_rom = W_hermit @ self.M @ W
        b_rom = W_hermit @ self.b

        frequencies = self.freq[self.freq_central-1::-1]
        for freq in frequencies:
            if freq in self.solution:
                pass
            else:
                error = self.projection_step(freq, M_rom, C_rom, K_rom, b_rom, W)
                if error > self.tol_proj:
                    self.strategy(freq)
                    W = self.basis()
                    W_hermit = W.conj().T

                    K_rom = W_hermit @ self.K @ W
                    C_rom = W_hermit @ self.C @ W
                    M_rom = W_hermit @ self.M @ W
                    b_rom = W_hermit @ self.b
        
        error = 0
        W = self.exp_basis[self.freq_central]
        W_hermit = W.conj().T

        K_rom = W_hermit @ self.K @ W
        C_rom = W_hermit @ self.C @ W
        M_rom = W_hermit @ self.M @ W
        b_rom = W_hermit @ self.b
        frequencies = self.freq[self.freq_central:]
        for freq in frequencies:
            if freq in self.solution:
                pass
            else:
                error = self.projection_step(freq, M_rom, C_rom, K_rom, b_rom, W)
                if error > self.tol_proj:
                    self.strategy(freq)
                    W = self.basis()
                    W_hermit = W.conj().T

                    K_rom = W_hermit @ self.K @ W
                    C_rom = W_hermit @ self.C @ W
                    M_rom = W_hermit @ self.M @ W
                    b_rom = W_hermit @ self.b
        
        
        solution = np.array([value for (key, value) in sorted(self.solution.items())]).T
        error = np.array([value for (key, value) in sorted(self.proj_error.items())])
        return solution, error, list(self.exp_basis.keys())

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