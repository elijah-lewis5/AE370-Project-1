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


# ============================================================
# Q2: Damping Sweep (Stability & Settling Time)
# ============================================================

import numpy as np

# Reuse helpers if they already exist; otherwise define.
try:
    _set_constants
except NameError:
    def _set_constants(new_constants):
        global constants
        constants = new_constants

def _run_integrator(f, u0, dt, t_final, integrator=None):
    """
    Run either ab3_solve (returns U, T) or euler_forward (returns T, U).
    """
    if integrator is None:
        from source.numerical_methods import euler_forward as integrator  # fallback
    try:
        out = integrator(f, u0, dt, t_final)
    except TypeError:
        out = integrator(f, u0, dt=dt, t_final=t_final)
    # Detect which element is the state array (N,4).
    if isinstance(out, tuple) and getattr(out[0], "ndim", 1) > 1:
        U, T = out[0], out[1]
    else:
        T, U = out[0], out[1]
    return np.asarray(T), np.asarray(U)

def _theta_max(theta):
    return float(np.max(np.abs(theta)))

def _x_max(x):
    return float(np.max(np.abs(x)))

def _measure_frequency(theta, t):
    theta = np.asarray(theta)
    t = np.asarray(t)
    zc = np.where(np.diff(np.signbit(theta)))[0]
    if zc.size < 2:
        return np.nan
    half_periods = np.diff(t[zc])
    if half_periods.size == 0:
        return np.nan
    Tmean = 2.0 * np.mean(half_periods)
    return 1.0 / Tmean if np.isfinite(Tmean) and Tmean > 0 else np.nan

def _settling_time(t, x, theta, eps_x=0.01, eps_th=0.01, dwell=2.0):
    """
    First time where |x|<eps_x and |theta|<eps_th continuously for 'dwell' seconds.
    Returns np.nan if never settles.
    """
    t = np.asarray(t)
    x = np.asarray(x)
    th = np.asarray(theta)
    cond = (np.abs(x) < eps_x) & (np.abs(th) < eps_th)
    if t.size < 2:
        return np.nan
    dt_local = np.median(np.diff(t))
    need = max(1, int(np.ceil(dwell / max(dt_local, 1e-12))))
    # Find first index where the next 'need' samples all satisfy cond
    idx = 0
    while idx + need <= cond.size:
        if np.all(cond[idx:idx+need]):
            return float(t[idx])
        idx += 1
    return np.nan

def _total_energy(u, constants_tuple):
    """
    Total mechanical energy (cart + pendulum + spring + gravity).
    Kinetic:
      cart: 0.5*m1*xdot^2
      pendulum bob: v^2 = (xdot + l*thetadot*cos)^2 + (l*thetadot*sin)^2
    Potential:
      spring: 0.5*k*x^2
      gravity: m2*g*l*(1-cos(theta))
    """
    m1, m2, l, k, b, g = constants_tuple
    x = u[:, 0]; xdot = u[:, 1]; th = u[:, 2]; thdot = u[:, 3]
    v_bob_sq = (xdot + l*thdot*np.cos(th))**2 + (l*thdot*np.sin(th))**2
    KE = 0.5*m1*xdot**2 + 0.5*m2*v_bob_sq
    PE = 0.5*k*x**2 + m2*g*l*(1 - np.cos(th))
    return KE + PE

def sweep_damping_effect(
    b_values, integrator, u0, dt, t_final,
    constants_base=None, eps_x=0.01, eps_th=0.01, dwell=2.0
):
    """
    Sweep damping coefficient b and compute settling time, overshoot, energy, frequency.

    Returns
    -------
    results : dict
        {
          'b': array,
          't_settle': array,
          'overshoot_theta': array,
          'overshoot_x': array,
          'freq_damped': array,
          'time_series': {
              b_val: {'t': t, 'u': U, 'energy': E}
          }
        }
    """
    if constants_base is None:
        constants_base = constants
    m1, m2, L, k, b0, g = constants_base

    t_settle = []
    Mp_theta = []
    Mp_x = []
    f_d = []
    series = {}

    for b_val in b_values:
        _set_constants((m1, m2, L, k, float(b_val), g))
        # Run damped model
        t, U = _run_integrator(skycrane_damping, u0, dt, t_final, integrator=integrator)

        # Metrics
        ts = _settling_time(t, U[:, 0], U[:, 2], eps_x=eps_x, eps_th=eps_th, dwell=dwell)
        Mp_theta.append(_theta_max(U[:, 2]))
        Mp_x.append(_x_max(U[:, 0]))
        t_settle.append(ts)
        f_d.append(_measure_frequency(U[:, 2], t))

        E = _total_energy(U, (m1, m2, L, k, b_val, g))
        series[float(b_val)] = {'t': t, 'u': U, 'energy': E}

    # restore
    _set_constants((m1, m2, L, k, b0, g))

    return {
        'b': np.asarray(b_values, dtype=float),
        't_settle': np.asarray(t_settle, dtype=float),
        'overshoot_theta': np.asarray(Mp_theta, dtype=float),
        'overshoot_x': np.asarray(Mp_x, dtype=float),
        'freq_damped': np.asarray(f_d, dtype=float),
        'time_series': series
    }
