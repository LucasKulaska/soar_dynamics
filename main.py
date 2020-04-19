import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from scipy.sparse import csc_matrix


from arnoldi.soar import Soar
from arnoldi.toar import Toar
from arnoldi.mgs import Mgs

from rom.rom import ROM
# Profilers
import cProfile, pstats, io
from pstats import SortKey

#%% Test setup
import h5py


f = h5py.File('matrices.hdf5', 'r')
global_matrices = f.get('global_matrices').items()

data = list(f.get('global_matrices').items())

I = data[0][1][()]
J = data[1][1][()]
data_K = data[2][1][()]
data_M = data[3][1][()]
dofs_free = data[4][1][()]
total_dofs = data[5][1][()]

f.close()

total_dofs = total_dofs.astype('int')
dofs_free = dofs_free.astype('int')

K = csc_matrix( (data_K, (I, J)), shape = [total_dofs, total_dofs] )
K = K[ dofs_free, : ][ :, dofs_free ]
M = csc_matrix( (data_M, (I, J)), shape = [total_dofs, total_dofs] )
M = M[ dofs_free, : ][ :, dofs_free ]

F = np.loadtxt(open("C:\\Users\\Kula\\Trabalho de Conclusão\\soar_dynamics\\examples\\F.csv", "rb"), delimiter=",", skiprows=0)
F = F[ dofs_free]

C = csc_matrix( np.zeros_like(K.toarray()) )

freq = np.arange(1,501)

#%% Reduced Order Model (ROM)
# init:  profile
pr = cProfile.Profile()
pr.enable()


rom = ROM(K, C, M, F, freq)

rom.moment_matching = 7
rom.tol_svd = 1e-10
rom.tol_proj = 1e-5
rom.num_freqs = 7
rom.freq_step = 50


start = time()
solution_rom, error = rom.projection()
end = time()


solution_fom = rom.full_order()

end2 = time()

print('Time took to perform order reduction:', end - start, 'seconds')
print('Time took to perform full order:', end2 - end, 'seconds')

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())

#%% plot

fig = plt.figure(figsize=[12,8])
ax = fig.add_subplot(1,1,1)
plt.semilogy(freq, np.abs(solution_fom[15, :]), color = [0,0,0], linewidth=3)
plt.semilogy(freq, np.abs(solution_rom[15, :]), color = [1,0,0], linewidth=1.5)
ax.set_title(('Full order vs Reduced order Model'), fontsize = 18, fontweight = 'bold')
ax.set_xlabel(('Frequency [Hz]'), fontsize = 16, fontweight = 'bold')
ax.set_ylabel(("FRF's magnitude"), fontsize = 16, fontweight = 'bold')
ax.legend(['Full Order Model','Reduced Order Model'])
plt.show()

fig = plt.figure(figsize=[12,8])
ax = fig.add_subplot(1,1,1)
plt.semilogy(freq, error, color = [0,0,0], linewidth=3)
ax.set_title(('Full order vs Reduced order Model'), fontsize = 18, fontweight = 'bold')
ax.set_xlabel(('Frequency [Hz]'), fontsize = 16, fontweight = 'bold')
ax.set_ylabel(("Error"), fontsize = 16, fontweight = 'bold')
ax.legend(['Error'])
plt.show()