from source.numerical_methods import *
import numpy as np
from source.eoms import *
def ivp_ab3_error(u_0, T, delta_t, delta_t_baseline):
    
    """Final-time relative error: ||u(T)_dt - u(T)_baseline|| / ||u(T)_baseline||."""
    u_dt, _   = ab3_solve(skycrane_damping, u_0, delta_t, T)
    u_base, _ = ab3_solve(skycrane_damping, u_0, delta_t_baseline, T)
    uT_dt, uT_base = u_dt[-1], u_base[-1]
    return np.linalg.norm(uT_dt - uT_base) / max(np.linalg.norm(uT_base), 1e-16)