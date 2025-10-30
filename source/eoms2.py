import numpy as np
from sympy import *

from data.initial_conditions import constants

def skycrane(u):
    x, xdot, theta, thetadot = u 
    m1, m2, l, k, b, g = constants

    A = Matrix([[m1+m2, m2*l*cos(theta)],
                [cos(theta)/l, 1]])
    B = Matrix([[m2*l*thetadot**2*sin(theta)-k*x],
                [-g/l * sin(theta)]])

    O = A.inv() @ B
    xddot = O[0]
    thetaddot = O[1]
    udot = np.array([xdot, xddot, thetadot, thetaddot])
    return udot

def skycrane_damping(u):
    x, xdot, theta, thetadot = u 
    m1, m2, l, k, b, g = constants

    A = Matrix([[m1+m2, m2*l*cos(theta)],
                [cos(theta)/l, 1]])
    B = Matrix([[m2*l*thetadot**2*sin(theta)-k*x-b*xdot],
                [-g/l * sin(theta)]])

    O = A.inv() @ B
    xddot = O[0]
    thetaddot = O[1]
    udot = np.array([xdot, xddot, thetadot, thetaddot])
    return udot

# ============================================================
# Q1: Cable Length Sweep (Damped vs Undamped) — ab3/euler friendly
# ============================================================

import numpy as np

def _set_constants(new_constants):
    """
    Update the module-level 'constants' tuple so skycrane() and
    skycrane_damping() pick up the current (m1, m2, L, k, b, g).
    """
    global constants
    constants = new_constants

def _measure_frequency(theta, t):
    """
    Estimate dominant oscillation frequency from zero-crossings of theta(t).
    Uses sign-bit change to be robust near zero.
    """
    theta = np.asarray(theta)
    t = np.asarray(t)
    zc_idx = np.where(np.diff(np.signbit(theta)))[0]
    if zc_idx.size < 2:
        return np.nan
    # zero-crossings give half-periods when measured between alternating indices
    # convert to period estimate: average of 2*Δt between every other crossing
    half_periods = np.diff(t[zc_idx])
    if half_periods.size == 0:
        return np.nan
    periods = 2.0 * half_periods
    mean_T = np.mean(periods)
    if not np.isfinite(mean_T) or mean_T <= 0:
        return np.nan
    return 1.0 / mean_T

def _theta_max(theta):
    return float(np.max(np.abs(theta)))

def sweep_length_effect(L_values, integrator, u0, dt, t_final, constants_base=None):
    """
    Sweep across cable lengths (L) for both undamped and damped systems.

    Parameters
    ----------
    L_values : array-like
        Cable lengths [m] to simulate.
    integrator : callable
        e.g., ab3_solve or euler_forward; must have signature
        integrator(f, u0, dt, t_final) or integrator(f, u0, dt=..., t_final=...).
    u0 : array-like
        Initial state [x, xdot, theta, thetadot].
    dt : float
        Time step [s].
    t_final : float
        Final time [s].
    constants_base : tuple or None
        (m1, m2, L, k, b, g). If None, use current module 'constants'.

    Returns
    -------
    results : dict
        {
          'L': np.ndarray,
          'freq_damped': np.ndarray,
          'freq_undamped': np.ndarray,
          'theta_max_damped': np.ndarray,
          'theta_max_undamped': np.ndarray,
          'time_series': {
              L_value: {'t': t, 'undamped': U_und, 'damped': U_dmp}, ...
          }
        }
    """
    # pull current constants as baseline if not provided
    if constants_base is None:
        constants_base = constants
    m1, m2, L0, k, b, g = constants_base

    freq_dmp, freq_und = [], []
    amp_dmp, amp_und = [], []
    time_series = {}

    # Helper to call integrator with either positional or keyword dt/t_final
    def _run_integrator(f, u0, dt, t_final):
        try:
            return integrator(f, u0, dt, t_final)
        except TypeError:
            # fall back to keyword args if needed
            return integrator(f, u0, dt=dt, t_final=t_final)

    for L in L_values:
        # update constants so EOM functions use this L
        _set_constants((m1, m2, float(L), k, b, g))

                # Detect output order (ab3_solve returns U, T; Euler returns T, U)
        out_und = _run_integrator(skycrane, u0, dt, t_final)
        if isinstance(out_und, tuple) and out_und[0].ndim > 1:
            U_und, t_und = out_und  # ab3_solve-style
        else:
            t_und, U_und = out_und  # euler_forward-style

        freq_u = _measure_frequency(U_und[:, 2], t_und)
        amp_u = _theta_max(U_und[:, 2])

        # Damped
        out_dmp = _run_integrator(skycrane_damping, u0, dt, t_final)
        if isinstance(out_dmp, tuple) and out_dmp[0].ndim > 1:
            U_dmp, t_dmp = out_dmp
        else:
            t_dmp, U_dmp = out_dmp

        freq_d = _measure_frequency(U_dmp[:, 2], t_dmp)
        amp_d = _theta_max(U_dmp[:, 2])


        freq_und.append(freq_u)
        freq_dmp.append(freq_d)
        amp_und.append(amp_u)
        amp_dmp.append(amp_d)

        time_series[float(L)] = {
            't': t_dmp if len(t_dmp) >= len(t_und) else t_und,
            'undamped': U_und,
            'damped': U_dmp
        }

    results = {
        'L': np.asarray(L_values, dtype=float),
        'freq_undamped': np.asarray(freq_und, dtype=float),
        'freq_damped': np.asarray(freq_dmp, dtype=float),
        'theta_max_undamped': np.asarray(amp_und, dtype=float),
        'theta_max_damped': np.asarray(amp_dmp, dtype=float),
        'time_series': time_series
    }

    # restore original constants
    _set_constants((m1, m2, L0, k, b, g))
    return results
