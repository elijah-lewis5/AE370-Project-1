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


#RK4 --> AB3 needs three previous points, only have initial state, use RK4 to generate first two time steps (u1, u2)
def rk4_step(f, u, dt):
  k1 = f(u) #compute estimated slope at different points inside the step
  k2 = f(u + 0.5 * dt * k1)
  k3 = f(u + 0.5 * dt * k2)
  k4 = f(u + dt * k3)
  return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4) #return the weighted average of the slopes


#AB3 Method
# u_{k+1} = u_k + dt * [ 23/12 f_k - 16/12 f_{k-1} + 5/12 f_{k-2}
def ab3_solve(f, u0, dt, T):
  #initialize arrays
  n = int(T/dt)
  m = len(u0)
  U = np.zeros((n+1,m)) #array to store every time steps solution (row --> timestep, column --> comp of u)
  U[0] = u0
  t = np.linspace(0,T,n+1)

  #use RK4 to get U1 and U2
  U[1] = rk4_step(f, U[0], dt) #use RK4 twice to get u1 and u2
  U[2] = rk4_step(f, U[1], dt)

  #precomute initial derivative values
  f_vals = np.zeros((n+1,m)) #f_vals[k] stores derivative f_k = f(uk, tk)
  f_vals[0] = f(U[0])
  f_vals[1] = f(U[1])
  f_vals[2] = f(U[2]) #precompute firsat three so AB3 can use them straight away

  #main AB3 timestep loop
  for k in range(2,n):
    U[k+1] = U[k] + dt * ((23/12)*f_vals[k] - (16/12)*f_vals[k-1] + (5/12)*f_vals[k-2])
    f_vals[k+1] = f(U[k+1])

  return U, t

