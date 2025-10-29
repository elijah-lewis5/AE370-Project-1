import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from data.initial_conditions import constants

def animate_skycrane(t, u, dt, skip = 10, save_path=None, title = ''):
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

    # visual scaling: make the rod a bit shorter for clarity (render only)
    rod_scale = 1
    px = x + l * np.sin(theta)
    py = -l * np.cos(theta)
    px_plot = x + l * rod_scale * np.sin(theta)
    py_plot = -l * rod_scale * np.cos(theta)

    # Make the figure wider and more rectangular
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_aspect('equal')
    ax.set_xlim(-1.2 * l * rod_scale, 1.2 * l * rod_scale)
    # Y limits based on the (physical) pendulum length but tightened a bit
    ax.set_ylim(-1.2 * l * rod_scale, 0.6 * l * rod_scale)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f'Sky Crane Simulation ({title})')

    # Current time
    time_text = ax.text(
    0.02, 0.95, "", transform=ax.transAxes,
    fontsize=12, color="black", ha="left", va="top"
    )

    # Cart as a rectangle patch
    cart_width = 2
    cart_height = 1
    cart_y = 0.05  # slight offset above y=0 so rod can attach nicely
    cart_patch = patches.Rectangle((0, cart_y), cart_width, cart_height,
                                   facecolor='C0', edgecolor='k', zorder=3)
    ax.add_patch(cart_patch)

    wall_x = -rod_scale * l / 2
    wall, = ax.plot([wall_x, wall_x], [-0.5, cart_height + 0.5], 'k', lw=6)
    spring_line, = ax.plot([], [], 'gray', lw=2)

    # Rod and mass (mass rendered slightly larger)
    rod, = ax.plot([], [], 'k-', lw=2, zorder=4)
    mass_patch = patches.Circle((0, 0), radius=0.15, color='r', zorder=5)
    ax.add_patch(mass_patch)

    def init():
        # initialize visual elements at the first timestep
        spring_line.set_data([], [])
        rod.set_data([], [])
        cart_patch.set_x(x[0] - cart_width / 2)
        cart_patch.set_y(cart_y)
        mass_patch.center = (px_plot[0], py_plot[0])
        return spring_line, cart_patch, rod, mass_patch

    def update(i):
        xi = x[i]
        pxi = px[i]
        pyi = py[i]
        pxi_plot = px_plot[i]
        pyi_plot = py_plot[i]

        # Update cart rectangle position (centered horizontally at xi)
        cart_patch.set_x(xi - cart_width / 2)
        cart_patch.set_y(cart_y)

        # Spring (decorative) - keep it anchored to the left wall and the left
        # side of the cart visual
        spring_height = 0.05
        s = np.linspace(wall_x + 0.05, xi - cart_width / 2, 20)
        spring_line.set_data(s, cart_y + cart_height/2 + spring_height * np.sin(25 * np.linspace(0, 1, len(s))))

        # Pendulum: attach to bottom of cart visually
        anchor_y = cart_y 
        rod.set_data([xi, pxi_plot], [anchor_y, pyi_plot])
        mass_patch.center = (pxi_plot, pyi_plot)
        
        # Current time
        current_time = t[i]
        time_text.set_text(f"t = {current_time:.2f} s")

        return spring_line, cart_patch, rod, mass_patch, time_text

    ani = FuncAnimation(
        fig,
        update,
        frames=range(0, len(t), skip),
        init_func=init,
        interval=dt * skip * 1000,
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

def plot_x_theta(t, u, title="Cart Position and Pendulum Angle vs Time"):
    """
    Plot cart position x(t) and pendulum angle θ(t) vs time.

    Parameters
    ----------
    t : ndarray
        Time array [s]
    u : ndarray
        State array [x, xdot, theta, thetadot] over time
    title : str, optional
        Plot title
    """
    plt.figure(figsize=(8, 4))
    plt.plot(t, u[:, 0], label="x(t) [m]", linewidth=2)
    plt.plot(t, u[:, 2], label="θ(t) [rad]", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("State Variables")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_x_vs_theta(u, t=None, title=''):
    """
    Compare simple and time-colored x vs θ plots.
    
    Parameters
    ----------
    u : ndarray
        Array of shape (N, 4): [x, x_dot, theta, theta_dot].
    t : ndarray, optional
        Time array of shape (N,). Used for coloring trajectory by time.
    """
    x = u[:, 0]
    theta = u[:, 2]

    plt.figure(figsize=(6, 5))

    plt.subplot(1, 1, 1)
    plt.plot(x, theta, 'b-', linewidth=2)
    plt.title(f"x vs θ ({title})")
    plt.xlabel("x (position)")
    plt.ylabel("θ (angle)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()