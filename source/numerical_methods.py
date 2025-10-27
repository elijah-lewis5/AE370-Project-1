import numpy as np

def euler_forward(f, u0, dt, t_final):
    """
    Integrate u_dot = f(u) from t=0 to t_final using Euler Forward.

    Parameters
    ----------
    f : callable
        Function that computes u_dot = f(u)
    u0 : array_like
        Initial condition vector
    dt : float
        Time step
    t_final : float
        Final simulation time

    Returns
    -------
    t : ndarray
        Time array
    u : ndarray
        Array of solution values at each time step (len(t) x len(u0))
    """
    u0 = np.array(u0, dtype=float)
    n_steps = int(t_final / dt) + 1
    u = np.zeros((n_steps, len(u0)))
    t = np.linspace(0, t_final, n_steps)

    # Set initial condition
    u[0] = u0

    # Time stepping loop
    for n in range(n_steps - 1):
        u_dot = f(u[n])
        u[n + 1] = u[n] + dt * u_dot

    return t, u