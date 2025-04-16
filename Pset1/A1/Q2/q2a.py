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
