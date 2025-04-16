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