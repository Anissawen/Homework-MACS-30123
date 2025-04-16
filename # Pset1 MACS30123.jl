# Pset1 MACS30123
### Student Name: Siqi Wen 
### Student ID: 12451279
# P1
### (a)
### codes:

```python
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
```















**Bold**, *Italic*, `Code`