import numpy as np
import seaborn as sns

from arnoldi.soar import Soar
from arnoldi.toar import Toar
from arnoldi.mgs import Mgs

# Test setup
N = 100
n = 30
A = np.random.rand(N,N)
B = np.random.rand(N,N)
u = np.random.rand(N)

# SOAR
soar = Soar(A, B, u)
Q_soar, _, _, _ = soar.procedure(n = n)

# TOAR
toar = Toar(A, B, u)
Q_toar, _, _, _= toar.procedure(n = n)

# MGS
# Krylov space
V = np.zeros([N,n])
V[:,0] = u
V[:,1] = A @ u

for i in range(2,n):
    V[:,i] = A @ V[:,i-1] + B @ V[:,i-2]

Q_mgs = Mgs.basis(V)

print(np.linalg.norm( Q_soar.T @ Q_soar - np.eye(n) ))
print(np.linalg.norm( Q_toar.T @ Q_toar - np.eye(n) ))
print(np.linalg.norm( Q_mgs.T @ Q_mgs - np.eye(n) ))