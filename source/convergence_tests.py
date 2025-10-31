import numpy as np
from source.numerical_methods import *
from source.eoms import *

# Damped
def ivp_ab3_error(u_0, T, delta_t, delta_t_baseline):
    
    """Final-time relative error: ||u(T)_dt - u(T)_baseline|| / ||u(T)_baseline||."""
    u_dt, _   = ab3_solve(skycrane_damping, u_0, delta_t, T) #Broader time step less accurate results
    u_base, _ = ab3_solve(skycrane_damping, u_0, delta_t_baseline, T) #Finer time step more accurate results
    uT_dt, uT_base = u_dt[-1], u_base[-1] #final state at time T for both simulations
    return np.linalg.norm(uT_dt - uT_base) / max(np.linalg.norm(uT_base), 1e-16) #relative error calculation between both results then divides by the norm

# Undamped
def ivp_ab3_errorundamped(u_0, T, delta_t, delta_t_baseline):
    """Final-time relative error: ||u(T)_dt - u(T)_baseline|| / ||u(T)_baseline||."""
    u_dt, _   = ab3_solve(skycrane, u_0, delta_t, T)
    u_base, _ = ab3_solve(skycrane, u_0, delta_t_baseline, T)
    uT_dt, uT_base = u_dt[-1], u_base[-1]
    return np.linalg.norm(uT_dt - uT_base) / max(np.linalg.norm(uT_base), 1e-16)