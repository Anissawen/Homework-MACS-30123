from mpi4py import MPI
import numpy as np
import scipy.stats as sts
import time
import q1a

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
