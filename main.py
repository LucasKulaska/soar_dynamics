import numpy as np
import seaborn as sns

from SOAR.soar import Soar
from SOAR.toar import Toar
from SOAR.mgs import Mgs

# Test setup
N = 100
k = 30
A = np.random.rand(N,N)
B = np.random.rand(N,N)
u = np.random.rand(N)

# SOAR
soar = Soar(A, B, u)
Q_soar, _, _, _ = soar.procedure(n = k)

# TOAR
toar = Toar(A, B, u)
Q_toar, _, _, _ = toar.procedure(k = k)

# MGS
V = np.zeros([N,k])
V[:,0] = u
V[:,1] = A @ u

for i in range(2,k):
    V[:,i] = A @ V[:,i-1] + B @ V[:,i-2]

Q_mgs = Mgs.basis(V)