# convergence_tests.py
# AE 370 — Part 3: Error & convergence driver for the Skycrane system (with damping)
# Matches class notation: IVP u̇ = f(u, t); final-time relative error vs baseline; log–log slope.

from __future__ import annotations

import numpy as np
from typing import Tuple, Callable, Sequence, Optional

from eoms import skycrane_damping
from error_tools import run_convergence, plot_convergence, estimate_order

# ---------------------------------------------------------------------
# RK4 (single step) and AB3 (multistep) — clean version for convergence
#   • RK4 bootstrap supplies u1, u2
#   • AB3: u_{k+1} = u_k + (Δt/12) [ 23 f_k − 16 f_{k-1} + 5 f_{k-2} ]
#   • No event/early-exit logic (fixed-horizon per class error analysis)
# ---------------------------------------------------------------------

def rk4_step(f: Callable[[np.ndarray, float], np.ndarray],
             u: np.ndarray, t: float, dt: float) -> np.ndarray:
    k1 = f(u, t)
    k2 = f(u + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(u + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(u + dt * k3, t + dt)
    return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def ab3_solve_clean(f: Callable[[np.ndarray, float], np.ndarray],
                    u0: np.ndarray, t_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    AB3 march on the provided time grid.
    Returns (U, t_grid) with U.shape = (len(t_grid), len(u0)).
    """
    n = len(t_grid)
    m = len(u0)
    U = np.zeros((n, m), dtype=float)
    U[0] = np.asarray(u0, dtype=float)

    if n == 1:
        return U, t_grid

    # Bootstrap with RK4 for u1 and u2
    dt0 = t_grid[1] - t_grid[0]
    U[1] = rk4_step(f, U[0], t_grid[0], dt0)
    if n == 2:
        return U, t_grid

    dt1 = t_grid[2] - t_grid[1]
    U[2] = rk4_step(f, U[1], t_grid[1], dt1)

    # Precompute f at k-2, k-1, k
    f_km2 = f(U[0], t_grid[0])
    f_km1 = f(U[1], t_grid[1])
    f_k   = f(U[2], t_grid[2])

    for k in range(2, n - 1):
        dt_k = t_grid[k+1] - t_grid[k]
        U[k+1] = U[k] + (dt_k / 12.0) * (23.0 * f_k - 16.0 * f_km1 + 5.0 * f_km2)
        f_km2, f_km1, f_k = f_km1, f_k, f(U[k+1], t_grid[k+1])

    return U, t_grid

# ---------------------------------------------------------------------
# Adapter: expected solver signature for error_tools
#   solver(u0, T, dt, f, **kwargs) -> (U, t)
# ---------------------------------------------------------------------

def _make_time_grid(T: float, dt: float) -> np.ndarray:
    steps = int(np.floor(T / float(dt)))
    return (np.arange(steps + 1, dtype=float) * float(dt))

def solver_ab3_adapter(u0: np.ndarray, T: float, dt: float,
                       f: Callable[[np.ndarray, float], np.ndarray],
                       **_: dict) -> Tuple[np.ndarray, np.ndarray]:
    t_grid = _make_time_grid(T, dt)
    U, t = ab3_solve_clean(f, u0, t_grid)
    return U, t

# ---------------------------------------------------------------------
# RHS wrapper: curry constants so f has class-style signature f(u, t)
# State order: u = [x, xdot, theta, thetadot]
# constants = (m1, m2, l, k, b, g)
# ---------------------------------------------------------------------

def make_rhs_with_damping(constants: Tuple[float, float, float, float, float, float]
                          ) -> Callable[[np.ndarray, float], np.ndarray]:
    def f(u: np.ndarray, t: float) -> np.ndarray:
        return skycrane_damping(u, constants)
    return f

# ---------------------------------------------------------------------
# Default configuration (edit as needed for your study)
# ---------------------------------------------------------------------

DEFAULT_CONSTANTS = (5.0, 2.0, 1.0, 50.0, 1.0, 9.81)  # (m1, m2, l, k, b, g)
DEFAULT_U0 = np.array([0.2, 0.0, 0.2, 0.0], dtype=float)  # [x, xdot, theta, thetadot]
DEFAULT_T  = 5.0  # s — fixed horizon for final-time error
DEFAULT_DT_LIST = [8e-3, 4e-3, 2e-3, 1e-3, 5e-4]  # finest is baseline

# Optional: compare only a subset of the state (e.g., positions)
# Set PROJECT=None to compare the full state vector.
def project_positions(u: np.ndarray) -> np.ndarray:
    # indices 0 -> x, 2 -> theta (skip velocities)
    return u[[0, 2]]

PROJECT = None  # or set to project_positions

# ---------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------

def main(constants=DEFAULT_CONSTANTS, u0=DEFAULT_U0, T=DEFAULT_T,
         dt_list: Sequence[float] = DEFAULT_DT_LIST,
         expected_order: Optional[float] = 3.0,
         project=PROJECT):
    f = make_rhs_with_damping(constants)

    # Compute convergence data (Δt vs relative error)
    hs, errs = run_convergence(
        solver=solver_ab3_adapter,
        f=f,
        u0=u0,
        T=T,
        dt_list=dt_list,
        solver_kwargs={},   # placeholder for future flags; kept for API symmetry
        project=project
    )

    # Report observed order from least-squares slope on log–log data
    p_obs = estimate_order(hs, errs)
    print("Convergence data:")
    for h, e in zip(hs, errs):
        print(f"  Δt = {h:.5g}  ->  e = {e:.6e}")
    print(f"Observed slope (order) ≈ {p_obs:.3f}")

    # Plot (matplotlib only; class style)
    plot_convergence(
        hs, errs,
        expected_order=expected_order,
        guide_at="largest",
        title="Skycrane (damped) — AB3 convergence: final-time error vs Δt",
        xlabel=r"$\Delta t$ (s)",
        ylabel=r"relative error $e$",
        show=True,
    )

if __name__ == "__main__":
    main()
