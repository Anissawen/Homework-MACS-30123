# Assignment 1
**Name:** Siqi Wen

**ID:** anissa6

---
## Question 1: Clocking CPU Parallelism
### (a) Precompiling with Numba
For this question I rewrite the computationally intensive nested for loops as a separate function, pre-compile it using numba, and compare the execution time between the original version and the numba-accelerated version using a single CPU core. The following code from [q1a.py](https://github.com/Anissawen/Homework-MACS-30123/blob/main/Pset1/A1/Q1/q1a.py) demonstrates the precompiled function. 

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

The following code from [pset1.py](https://github.com/Anissawen/Homework-MACS-30123/blob/main/Pset1/A1/Q1/pset1.py) demonstrates the simulation using the original python loops and the ahead-of-time precompiled function. Running both versions on the Midway cluster produced the following timing measurement (as shown in [q1a.out](https://github.com/Anissawen/Homework-MACS-30123/blob/main/Pset1/A1/Q1/q1a.out)):

- **Without AOT compilation:** (Your result)
- **With AOT compilation:** (Your result)

The significant speedup shows the benefit of precompiling the computationally intensive function with Numba. 

``` python
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
```

```python
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
```
### (b) Parallel Simulation with MPI
Time 20 different simulation runs using varying numbers of cores (from 1 to 20). Use mpi4py to run 1,000 simulations per run, equally distributing them among cores. I use [run_q1b_loop_variable_cores.sh](https://github.com/Anissawen/Homework-MACS-30123/blob/main/Pset1/A1/Q1/run_q1b_loop_variable_cores.sh) to realize this. Recording the running time, and we plot the running time.

``` python
import matplotlib.pyplot as plt

cores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
         11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Corresponding total elapsed times (in seconds) for each run
elapsed_times = [
    0.0236, 0.0137, 0.0096, 0.0059, 0.0048,
    0.0044, 0.0038, 0.0034, 0.0031, 0.0027,
    0.0026, 0.0024, 0.0023, 0.0021, 0.0021,
    0.0019, 0.0019, 0.0019, 0.0018, 0.0016
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(cores, elapsed_times, marker='o', linestyle='-', color='b')

# Labeling axes and title
plt.xlabel("Number of MPI Processes (Cores)")
plt.ylabel("Total Elapsed Time (seconds)")
plt.title("Scaling Study: Computation Time vs. Number of MPI Processes")
plt.xticks(cores)  # Show integer ticks for each core count
plt.grid(True)

# Save the plot to the current directory as a PNG file
plt.savefig("q1b_plot.png", dpi=300, bbox_inches="tight")
```


