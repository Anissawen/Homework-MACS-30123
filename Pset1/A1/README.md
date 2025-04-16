# Assignment 1
**Name:** Siqi Wen

**ID:** anissa6

---
## Question 1: Clocking CPU Parallelism
### (a) Precompiling with Numba
For this question I refactor the computationally intensive nested for loops into a separate function and pre-compile it using Numba. Running the simulation using the original Python loops vesus the precompiled version (as shown in [q1a_time.py](https://github.com/Anissawen/Homework-MACS-30123/blob/main/Pset1/A1/Q1/pset1.py)) produced the following timing data ([q1a_time.out](https://github.com/Anissawen/Homework-MACS-30123/blob/main/Pset1/A1/Q1/q1a_time.out)) on the Midway Cluster:

- **Without AOT compilation:** 3.0968 seconds
- **With AOT compilation:** 0.0270 seconds

This roughly **114x speedup** illustrates the significant benefit of leveraging Numba for computationally intensive tasks. 
The following code from [q1a.py](https://github.com/Anissawen/Homework-MACS-30123/blob/main/Pset1/A1/Q1/q1a.py) demonstrates the precompiled function:

```python
from numba.pycc import CC
import numpy as np

cc = CC("q1a")

# Export function: x is 2D float64 (f8) arrays, and scalar inputs are int32 (i4) or float64 (f8)
@cc.export('simulate_lifetimes', 'void(f8[:,:], f8[:,:], f8, f8, f8, i4, i4)')
def simulate_lifetimes(z_mat, eps_mat, rho, mu, sigma, S, T):
    for s_ind in range(S):
        z_tm1 = mu
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t

cc.compile()
```

### (b) Parallel Simulation with MPI
For increased throughput, simulations were parallelized using the `mpi4py` library. 20 different simulation runs were executed, each with a varing number of cores from 1 to 20. In every run, 1000 simulations were equally distributed among the available cores. The simulations were run via a SBATCH script ([q1b.sh](https://github.com/Anissawen/Homework-MACS-30123/blob/main/Pset1/A1/Q1/run_q1b_loop_variable_cores.sh)). The output recorded in [q1b.out](https://github.com/Anissawen/Homework-MACS-30123/blob/main/Pset1/A1/Q1/q1b.out) reflect the execution time across various core counts. 
``` python
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

```

A plot summarizing the timing data was generated using code from [q1b_plot.py](https://github.com/Anissawen/Homework-MACS-30123/blob/main/Pset1/A1/Q1/q1b_plot.py). The plot demonstrates reduced execution times as the number of cores increases. 

### (c) Discussion of Speedup
(Your answer here)

## Question 2
### (a)
