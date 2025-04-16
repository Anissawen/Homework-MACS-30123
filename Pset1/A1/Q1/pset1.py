import numpy as np
import scipy.stats as sts
import time

# Set model parameters
rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu

# Set simulation parameters
S = 1000  # Number of lives to simulate
T = 4160  # Number of periods for each simulation
np.random.seed(25)

# Draw all idiosyncratic random shocks and create containers
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
z_mat = np.zeros((T, S))

start_time = time.time()
for s_ind in range(S):
    z_tm1 = z_0
    for t_ind in range(T):
        e_t = eps_mat[t_ind, s_ind]
        z_t = rho * z_tm1 + (1 - rho) * mu + e_t
        z_mat[t_ind, s_ind] = z_t
        z_tm1 = z_t
end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.4f} seconds")


import numpy as np
import scipy.stats as sts
import time
import q1a  # the precompiled module

# Parameters
rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu
S = 1000  # Number of lifetimes
T = 4160  # Number of weeks (80 years × 52)

np.random.seed(25)
eps_mat = np.random.normal(loc=0, scale=sigma, size=(T, S))
z_mat = np.zeros((T, S))

# Time performance
start = time.time()
q1a.simulate_lifetimes(z_mat, eps_mat, rho, mu, sigma, S, T)
end = time.time()

print(f"AOT-compiled simulation took: {end - start:.4f} seconds")


import numpy as np
import scipy.stats as sts
import time
import q1a  # the precompiled module

# Parameters
rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu
S = 1000  # Number of lifetimes
T = 4160  # Number of weeks (80 years × 52)

np.random.seed(25)
eps_mat = np.random.normal(loc=0, scale=sigma, size=(T, S))
z_mat = np.zeros((T, S))

# Time performance
start = time.time()
q1a.simulate_lifetimes(z_mat, eps_mat, rho, mu, sigma, S, T)
end = time.time()

print(f"AOT-compiled simulation took: {end - start:.4f} seconds")


import numpy as np
import scipy.stats as sts
import time
import q1a  # the precompiled module
from mpi4py import MPI


# Parameters
rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu
S_total = 1000
T = 4160


# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Divide S_total lifetimes across processes
base = S_total // size
extra = 1 if rank < S_total % size else 0
S_local = base + extra

# Seed and simulate only this chunk
np.random.seed(rank)
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S_local)).astype(np.float64)
z_mat = np.zeros((T, S_local), dtype=np.float64)


# Time and run
start_time = time.time()
q1a.simulate_lifetimes(z_mat, eps_mat, rho, mu, sigma, S_local, T)
end_time = time.time()
elapsed = end_time - start_time

# Collect timing
all_elapsed = comm.gather(elapsed, root=0)
if rank == 0:
    max_time = max(all_elapsed)
    print(f"{size} cores, {S_total} lifetimes: {max_time:.4f} seconds")




