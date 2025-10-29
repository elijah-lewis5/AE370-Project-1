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
def rk4_step(f, u, t, dt):
  k1 = f(u,t) #compute estimated slope at different points inside the step
  k2 = f(u + 0.5 * dt * k1, t + 0.5 * dt)
  k3 = f(u + 0.5 * dt * k2, t + 0.5 * dt)
  k4 = f(u + dt * k3, t + dt)
  return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4) #return the weighted average of the slopes


#AB3 Method
# u_{k+1} = u_k + dt * [ 23/12 f_k - 16/12 f_{k-1} + 5/12 f_{k-2}
def ab3_solve(f, u0, t_grid):
  #initialize arrays
  n = len(t_grid)
  m = len(u0)
  U = np.zeros((n,m)) #array to store every time steps solution (row --> timestep, column --> comp of u)
  U[0] = u0

  #use RK4 to get U1 and U2
  U[1] = rk4_step(f, U[0], t_grid[0], t_grid[1] - t_grid[0]) #use RK4 twice to get u1 and u2
  U[2] = rk4_step(f, U[1], t_grid[1], t_grid[2] - t_grid[1])

  #precomute initial derivative values
  f_vals = np.zeros((n,m)) #f_vals[k] stores derivative f_k = f(uk, tk)
  f_vals[0] = f(U[0], t_grid[0])
  f_vals[1] = f(U[1], t_grid[1])
  f_vals[2] = f(U[2], t_grid[2]) #precompute firsat three so AB3 can use them straight away

  #main AB3 timestep loop
  for k in range(2, n-1): #start at 2 since RK4 did previous
    dt_k = t_grid[k+1] - t_grid[k] #store current time step size
    #apply AB3 update
    U[k+1] = U[k] + dt_k * ((23/12)*f_vals[k] - (16/12)*f_vals[k-1] + (5/12)*f_vals[k-2]) #applying AB3 formula derived earlier
    #compute new derivative
    f_vals[k+1] = f(U[k+1], t_grid[k+1])

    
  return U, t_grid
