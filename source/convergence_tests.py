
def ivp_ab3_error(u_0, T, delta_t, delta_t_baseline):
    import numpy as np
    """Final-time relative error: ||u(T)_dt - u(T)_baseline|| / ||u(T)_baseline||."""
    u_dt, _   = ivp_ab3(u_0, T, delta_t)
    u_base, _ = ivp_ab3(u_0, T, delta_t_baseline)
    uT_dt, uT_base = u_dt[-1], u_base[-1]
    return np.linalg.norm(uT_dt - uT_base) / max(np.linalg.norm(uT_base), 1e-16)