import numpy as np
from sympy import *
from data.initial_conditions import constants

# EOMs for the dynamical system
def skycrane(u):
    """
    Returns the derivative of the state function using the
    equations of motion for the system without damping.
    """
    x, xdot, theta, thetadot = u 
    m1, m2, L, k, b, g = constants

    A = Matrix([[m1+m2, m2*L*cos(theta)],
                [cos(theta)/L, 1]])
    B = Matrix([[m2*L*thetadot**2*sin(theta)-k*x],
                [-g/L * sin(theta)]])

    O = A.inv() @ B
    xddot = O[0]
    thetaddot = O[1]
    udot = np.array([xdot, xddot, thetadot, thetaddot])
    return udot

def skycrane_damping(u):
    """
    Returns the derivative of the state function using the
    equations of motion for the system with damping.
    """
    x, xdot, theta, thetadot = u 
    m1, m2, L, k, b, g = constants

    A = Matrix([[m1+m2, m2*L*cos(theta)],
                [cos(theta)/L, 1]])
    B = Matrix([[m2*L*thetadot**2*sin(theta)-k*x-b*xdot],
                [-g/L * sin(theta)]])

    O = A.inv() @ B
    xddot = O[0]
    thetaddot = O[1]
    udot = np.array([xdot, xddot, thetadot, thetaddot])
    return udot

# Analysis and helper functions to answer the questions

# ============================================================
# Q1: Cable Length Sweep (Damped vs Undamped) — ab3/euler friendly
# ============================================================

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


# ============================================================
# Q3: Stiffness Sweep (Oscillations & Overall Response)
# ============================================================

def _run_integrator_q3(integrator, f, u0, dt, t_final):
    """
    Handle ab3_solve (returns U, T) vs euler_forward (returns T, U).
    """
    try:
        out = integrator(f, u0, dt, t_final)
    except TypeError:
        out = integrator(f, u0, dt=dt, t_final=t_final)
    if isinstance(out, tuple) and getattr(out[0], "ndim", 1) > 1:
        U, T = out
    else:
        T, U = out
    return np.asarray(T), np.asarray(U)

def _rms_over_window(y, frac_window=0.5):
    """
    RMS over the last 'frac_window' portion of the signal (default: last 50%).
    """
    y = np.asarray(y)
    n = len(y)
    i0 = int(max(0, np.floor((1.0 - frac_window) * n)))
    seg = y[i0:]
    if seg.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(seg**2)))

# Fallbacks if earlier helpers are not present (safe no-ops if already defined)
try:
    _set_constants
except NameError:
    def _set_constants(new_constants):
        global constants
        constants = new_constants

try:
    _measure_frequency
except NameError:
    def _measure_frequency(theta, t):
        theta = np.asarray(theta); t = np.asarray(t)
        zc = np.where(np.diff(np.signbit(theta)))[0]
        if zc.size < 2: return np.nan
        Tmean = 2.0 * np.mean(np.diff(t[zc]))
        return 1.0 / Tmean if np.isfinite(Tmean) and Tmean > 0 else np.nan

# energy helpers (use whichever exists)
def _total_energy_q3(u, constants_tuple):
    """
    Total mechanical energy (cart + pendulum + spring + gravity).
    """
    m1, m2, l, k, b, g = constants_tuple
    x = u[:, 0]; xdot = u[:, 1]; th = u[:, 2]; thdot = u[:, 3]
    v_bob_sq = (xdot + l*thdot*np.cos(th))**2 + (l*thdot*np.sin(th))**2
    KE = 0.5*m1*xdot**2 + 0.5*m2*v_bob_sq
    PE = 0.5*k*x**2 + m2*g*l*(1 - np.cos(th))
    return KE + PE

def _spring_energy_q3(u, k):
    x = u[:, 0]
    return 0.5 * k * x**2

def sweep_stiffness_effect(
    k_values, integrator, u0, dt, t_final, constants_base=None,
    frac_window=0.5
):
    """
    Sweep spring stiffness k and compute frequency, amplitudes, RMS, and energy partition.

    Parameters
    ----------
    k_values : array-like
        Stiffness values to simulate.
    integrator : callable
        ab3_solve or euler_forward.
    u0 : array-like
        Initial state [x, xdot, theta, thetadot].
    dt, t_final : float
        Time step and final time.
    constants_base : tuple or None
        (m1, m2, L, k, b, g); if None, use current 'constants'.
    frac_window : float
        Fraction of the tail window used for RMS and energy averaging.

    Returns
    -------
    results : dict with keys:
        'k', 'freq_theta', 'freq_x', 'theta_max', 'x_max',
        'theta_rms', 'x_rms', 'energy_frac_spring', 'time_series'
    """
    if constants_base is None:
        constants_base = constants
    m1, m2, L, k0, b, g = constants_base

    freq_th, freq_x = [], []
    th_max, x_max = [], []
    th_rms, x_rms = [], []
    efrac_spring = []
    series = {}

    for kval in k_values:
        _set_constants((m1, m2, L, float(kval), b, g))

        # Use DAMPED model for overall response behavior
        t, U = _run_integrator_q3(integrator, skycrane_damping, u0, dt, t_final)

        # Frequencies from theta and x
        fth = _measure_frequency(U[:, 2], t)
        fx  = _measure_frequency(U[:, 0], t)
        freq_th.append(fth); freq_x.append(fx)

        # Peak amplitudes (over full horizon)
        th_max.append(float(np.max(np.abs(U[:, 2]))))
        x_max.append(float(np.max(np.abs(U[:, 0]))))

        # RMS over tail window (robust to transient spikes)
        th_rms.append(_rms_over_window(U[:, 2], frac_window=frac_window))
        x_rms.append(_rms_over_window(U[:, 0], frac_window=frac_window))

        # Energy partition (average over tail window)
        E_tot = _total_energy_q3(U, (m1, m2, L, float(kval), b, g))
        E_spr = _spring_energy_q3(U, float(kval))
        n = len(t)
        i0 = int(max(0, np.floor((1.0 - frac_window) * n)))
        Etail = E_tot[i0:]; Es_tail = E_spr[i0:]
        if Etail.size == 0:
            efrac = np.nan
        else:
            Etail_mean = np.mean(Etail)
            efrac = float(np.mean(Es_tail) / Etail_mean) if Etail_mean > 0 else np.nan
        efrac_spring.append(efrac)

        series[float(kval)] = {'t': t, 'u': U}

    # restore baseline constants
    _set_constants((m1, m2, L, k0, b, g))

    return {
        'k': np.asarray(k_values, dtype=float),
        'freq_theta': np.asarray(freq_th, dtype=float),
        'freq_x': np.asarray(freq_x, dtype=float),
        'theta_max': np.asarray(th_max, dtype=float),
        'x_max': np.asarray(x_max, dtype=float),
        'theta_rms': np.asarray(th_rms, dtype=float),
        'x_rms': np.asarray(x_rms, dtype=float),
        'energy_frac_spring': np.asarray(efrac_spring, dtype=float),
        'time_series': series
    }



# ============================================================
# Q4: Mass Sweeps (m1, m2) — Stability & Transient Behavior
# ============================================================

# ---- helpers (Q4-specific names to avoid conflicts) ----
def _run_integrator_q4(integrator, f, u0, dt, t_final):
    """Handle ab3_solve (U, T) vs euler_forward (T, U)."""
    try:
        out = integrator(f, u0, dt, t_final)
    except TypeError:
        out = integrator(f, u0, dt=dt, t_final=t_final)
    if isinstance(out, tuple) and getattr(out[0], "ndim", 1) > 1:
        U, T = out
    else:
        T, U = out
    return np.asarray(T), np.asarray(U)

def _rms_tail_q4(y, frac_window=0.5):
    y = np.asarray(y)
    n = len(y); i0 = int(max(0, np.floor((1.0 - frac_window) * n)))
    seg = y[i0:]
    return float(np.sqrt(np.mean(seg**2))) if seg.size else np.nan

def _first_peak_after_q4(t, y, t_skip=1.0):
    t = np.asarray(t); y = np.asarray(y)
    mask = t >= (t[0] + t_skip)
    if not np.any(mask):
        return np.nan
    return float(np.max(np.abs(y[mask])))

def _measure_frequency_q4(y, t):
    y = np.asarray(y); t = np.asarray(t)
    zc = np.where(np.diff(np.signbit(y)))[0]
    if zc.size < 2:
        return np.nan
    Tmean = 2.0 * np.mean(np.diff(t[zc]))
    return 1.0 / Tmean if np.isfinite(Tmean) and Tmean > 0 else np.nan

def _total_energy_q4(U, constants_tuple):
    m1, m2, L, k, b, g = constants_tuple
    x = U[:, 0]; xdot = U[:, 1]; th = U[:, 2]; thdot = U[:, 3]
    v_bob_sq = (xdot + L*thdot*np.cos(th))**2 + (L*thdot*np.sin(th))**2
    KE = 0.5*m1*xdot**2 + 0.5*m2*v_bob_sq
    PE = 0.5*k*x**2 + m2*g*L*(1 - np.cos(th))
    return KE + PE

def _zeta_eff_q4(k, b, m1, m2):
    M = m1 + m2
    wn = np.sqrt(k / M)
    return b / (2.0 * M * wn)  # = b / (2*sqrt(k*M))

# Reuse _set_constants if present; else define
try:
    _set_constants
except NameError:
    def _set_constants(new_constants):
        global constants
        constants = new_constants

# ---- main sweep ----
def sweep_mass_effects(
    mode, mass_values, integrator, u0, dt, t_final,
    constants_base=None, frac_window=0.5, t_skip_for_peaks=1.0
):
    """
    Sweep mass parameter and compute stability/transient metrics.

    Parameters
    ----------
    mode : {'m1','m2'}
        Which mass to vary.
    mass_values : array-like
        Values for the chosen mass.
    integrator : callable
        ab3_solve or euler_forward.
    u0 : array-like
        [x, xdot, theta, thetadot].
    dt, t_final : float
        Time step and horizon.
    constants_base : tuple or None
        (m1, m2, L, k, b, g) baseline; if None, use module 'constants'.
    frac_window : float
        Tail window fraction for RMS.
    t_skip_for_peaks : float
        Seconds to ignore before measuring the first peak.

    Returns
    -------
    results : dict with keys:
        'param' ('m1' or 'm2'), 'values',
        'freq_theta','freq_x',
        'overshoot_theta','overshoot_x',
        'theta_rms','x_rms',
        'zeta_eff',
        'time_series' : { value: {'t','u','energy'} }
    """
    if constants_base is None:
        constants_base = constants
    m1_0, m2_0, L, k, b, g = constants_base

    freq_th, freq_x = [], []
    os_th, os_x = [], []
    rms_th, rms_x = [], []
    zeta_list = []
    series = {}

    for val in mass_values:
        if mode == 'm2':
            m1, m2 = m1_0, float(val)
        elif mode == 'm1':
            m1, m2 = float(val), m2_0
        else:
            raise ValueError("mode must be 'm1' or 'm2'")

        _set_constants((m1, m2, L, k, b, g))
        t, U = _run_integrator_q4(integrator, skycrane_damping, u0, dt, t_final)

        # Frequencies
        fth = _measure_frequency_q4(U[:, 2], t)
        fx  = _measure_frequency_q4(U[:, 0], t)
        freq_th.append(fth); freq_x.append(fx)

        # Overshoot (percent, relative to initial amplitude if nonzero)
        th_peak = _first_peak_after_q4(t, U[:, 2], t_skip=t_skip_for_peaks)
        x_peak  = _first_peak_after_q4(t, U[:, 0], t_skip=t_skip_for_peaks)
        th0 = abs(u0[2]); x0 = abs(u0[0])
        os_th.append(((th_peak - th0)/th0 if th0 > 1e-12 else th_peak))
        os_x.append(((x_peak  - x0)/x0  if x0  > 1e-12 else x_peak))

        # RMS (tail)
        rms_th.append(_rms_tail_q4(U[:, 2], frac_window=frac_window))
        rms_x.append(_rms_tail_q4(U[:, 0], frac_window=frac_window))

        # Energy & zeta
        E = _total_energy_q4(U, (m1, m2, L, k, b, g))
        zeta_list.append(_zeta_eff_q4(k, b, m1, m2))

        series[float(val)] = {'t': t, 'u': U, 'energy': E}

    # restore baseline
    _set_constants((m1_0, m2_0, L, k, b, g))

    return {
        'param': mode,
        'values': np.asarray(mass_values, dtype=float),
        'freq_theta': np.asarray(freq_th, dtype=float),
        'freq_x':     np.asarray(freq_x, dtype=float),
        'overshoot_theta': np.asarray(os_th, dtype=float),
        'overshoot_x':     np.asarray(os_x, dtype=float),
        'theta_rms':  np.asarray(rms_th, dtype=float),
        'x_rms':      np.asarray(rms_x, dtype=float),
        'zeta_eff':   np.asarray(zeta_list, dtype=float),
        'time_series': series
    }
