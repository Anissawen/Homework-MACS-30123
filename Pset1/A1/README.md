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

![q1b_plot](https://github.com/PaulWang-Uchicago/Homework-MACS-30123/blob/main/Pset1/A1/Q1/q1b_plot.png)

### (c) Discussion of Speedup
(Your answer here)

---

## Question 2: Embarrassingly Parallel Processing – Grid Search

### (a) Grid Search Implementation

In this section, we perform a grid search to determine the optimal persistence parameter (ρ) that maximizes the average number of periods until an individual's health index (zₜ) drops to zero or below. We simulate 1,000 lifetimes (each with T = 4160 periods) for each of 200 ρ values evenly spaced between -0.95 and 0.95. To efficiently explore this parameter space, we use mpi4py to run the simulations in parallel on 10 CPU cores, with each process handling 20 ρ values.

The $\epsilon$ matrix—representing the random shocks affecting the health index—is generated on rank 0 using a fixed random seed and then broadcasted to all processes to ensure consistency across simulations. The simulation itself is performed using an AOT-compiled module ([q1a.py](https://github.com/Anissawen/Homework-MACS-30123/blob/main/Pset1/A1/Q1/q1a.py)), which significantly accelerates the calculation of the lifetimes.

The following code ([q2a.py](https://github.com/Anissawen/Homework-MACS-30123/blob/main/Pset1/A1/Q2/q2a.py)) implements the grid search:

``` python
from mpi4py import MPI
import numpy as np
import q1a  # Precompiled AOT module with simulate_lifetimes
import time

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Constants
mu = 3.0
sigma = 1.0
z_0 = mu - 3 * sigma
T = 4160
S = 1000

# Grid of ρ values (200 points)
rho_values = np.linspace(-0.95, 0.95, 200)

# Split ρ values across cores
chunk_size = len(rho_values) // size
start = rank * chunk_size
end = (rank + 1) * chunk_size if rank != size - 1 else len(rho_values)
local_rhos = rho_values[start:end]

# Initialize shared ε matrix on rank 0
eps_mat = None
if rank == 0:
    np.random.seed(0)
    eps_mat = np.random.normal(loc=0, scale=sigma, size=(T, S)).astype(np.float64)

# Broadcast shared ε matrix to all processes
eps_mat = comm.bcast(eps_mat, root=0)

# Function to compute first time when z_t ≤ 0 for each individual
def time_to_zero(z_matrix):
    zero_crossings = np.argmax(z_matrix <= 0, axis=0)
    mask = np.any(z_matrix <= 0, axis=0)
    return np.where(mask, zero_crossings, T).mean()

# Run local grid search
results = []
z_mat = np.zeros((T, S), dtype=np.float64)

start_time = time.time()
for rho in local_rhos:
    q1a.simulate_lifetimes(z_mat, eps_mat, rho, mu, sigma, S, T)
    avg_time = time_to_zero(z_mat)
    results.append((rho, avg_time))
elapsed = time.time() - start_time

# Gather results at root
all_results = comm.gather(results, root=0)

if rank == 0:
    all_results = [item for sublist in all_results for item in sublist]
    best_rho, best_avg = max(all_results, key=lambda x: x[1])
    print(f"✅ Optimal ρ: {best_rho:.5f}, Avg. time to z_t ≤ 0: {best_avg:.2f} periods")
    print(f"⏱️ Total grid search time: {elapsed:.4f} seconds on {size} cores")

    import matplotlib.pyplot as plt

    # Extract ρ values and average times directly
    rho_vals = [item[0] for item in all_results]
    avg_vals = [item[1] for item in all_results]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(rho_vals, avg_vals, marker='o', markersize=3, linewidth=1, color='orange')
    plt.xlabel("ρ (Persistence Parameter)")
    plt.ylabel("Average Time to zₜ ≤ 0")
    plt.title("Average Time Until First Negative Health Index vs ρ")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rho_vs_avg_fall_time.png")
    plt.show()
```

The simulations were executed in parallel on the Midway cluster using the SBATCH file: [q2a.sh](https://github.com/Anissawen/Homework-MACS-30123/blob/main/Pset1/A1/Q2/q2a.sh). 

### (b) Visualization of Simulation Outcomes
After collecting the simulation results from the grid search, we visualize the relationship between the persistence parameter ($\rho$) and the average number of periods until the health index first falls to zero or below. In this plot, the x-axis represents the ρ values and the y-axis shows the corresponding average time to failure. This visualization provides an overview of the behavior of the system under different health persistence scenarios.

![grid_search_result](https://github.com/PaulWang-Uchicago/Homework-MACS-30123/blob/main/Pset1/A1/Q2/grid_search_results.png)

### (c) Optimal Parameter and Performance Metrics
The gird search identified the following optimal parameter and performance metrics: 
- **Optimal $\rho$:** -0.03342
- **Average time to $z_t \leq 0$:** 754.25 periods

These results are also available in the output file [q2b.out](https://github.com/Anissawen/Homework-MACS-30123/blob/main/Pset1/A1/Q2/q2b.out). 

