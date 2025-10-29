import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_skycrane(t, u, constants, skip=10, save_path=None):
    """
    Animate the Sky Crane (cart-pendulum with spring).

    Parameters
    ----------
    t : ndarray
        Time array [s]
    u : ndarray
        State history [x, xdot, theta, thetadot] over time, shape (N, 4)
    constants : tuple
        (m1, m2, l, k, b, g)
    skip : int, optional
        Frame skipping for faster playback (default 10)
    save_path : str, optional
        If provided, saves the animation as MP4 or GIF
    """
    if u.ndim != 2 or u.shape[1] < 4:
        raise ValueError("u must be shape (N,4): [x, xdot, theta, thetadot]")

    x, theta = u[:, 0], u[:, 2]
    m1, m2, l, k, b, g = constants

    px = x + l * np.sin(theta)
    py = -l * np.cos(theta)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_aspect('equal')
    ax.set_xlim(min(x) - 1.0, max(x) + 1.0)
    ax.set_ylim(-1.5 * l, 0.8 * l)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Sky Crane Simulation")

    wall, = ax.plot([-2, -1.8], [0, 0], 'k', lw=8)
    spring_line, = ax.plot([], [], 'gray', lw=2)
    cart, = ax.plot([], [], 'b', lw=8)
    rod, = ax.plot([], [], 'k-', lw=2)
    mass, = ax.plot([], [], 'ro', ms=8)

    def init():
        for line in (spring_line, cart, rod, mass):
            line.set_data([], [])
        return spring_line, cart, rod, mass

    def update(i):
        xi, pxi, pyi = x[i], px[i], py[i]

        # Cart
        cart.set_data([xi - 0.2, xi + 0.2], [0, 0])

        # Spring (decorative)
        s = np.linspace(-1.8, xi - 0.2, 20)
        spring_line.set_data(s, 0.05 * np.sin(25 * np.linspace(0, 1, len(s))))

        # Pendulum
        rod.set_data([xi, pxi], [0, pyi])
        mass.set_data([pxi], [pyi]) 

        return spring_line, cart, rod, mass

    ani = FuncAnimation(
        fig,
        update,
        frames=range(0, len(t), skip),
        init_func=init,
        interval=30,
        blit=True,
    )

    # --- Save or show ---
    if save_path:
        try:
            if save_path.endswith(".mp4"):
                ani.save(save_path, writer="ffmpeg")
            elif save_path.endswith(".gif"):
                ani.save(save_path, writer="pillow")
            else:
                raise ValueError("save_path must end with .mp4 or .gif")
        except Exception as e:
            print("Animation saving failed:", e)
    else:
        plt.show()

    return ani

def plot_x_theta(t, u):
    plt.plot(t, u[:,0], label="x(t)")
    plt.plot(t, u[:,2], label="theta(t)")
    plt.legend()
    plt.show()
    return