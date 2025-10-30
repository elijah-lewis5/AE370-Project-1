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



 # ============================================================
# Q1: Cable Length Sweep Plots (Damped vs Undamped)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

def plot_frequency_vs_length(results):
    L = results['L']
    plt.figure(figsize=(6, 4))
    plt.plot(L, results['freq_undamped'], 'o-', label='Undamped', linewidth=2)
    plt.plot(L, results['freq_damped'], 's--', label='Damped', linewidth=2)
    plt.xlabel('Cable Length L [m]')
    plt.ylabel('Dominant Frequency [Hz]')
    plt.title('Effect of Cable Length on Oscillation Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_amplitude_vs_length(results):
    L = results['L']
    plt.figure(figsize=(6, 4))
    plt.plot(L, results['theta_max_undamped'], 'o-', label='Undamped', linewidth=2)
    plt.plot(L, results['theta_max_damped'], 's--', label='Damped', linewidth=2)
    plt.xlabel('Cable Length L [m]')
    plt.ylabel('Maximum Angle |θ|max [rad]')
    plt.title('Effect of Cable Length on Maximum Swing Amplitude')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_phase_comparison(results, lengths_to_show):
    """
    Plot representative x vs θ phase portraits for chosen L values.
    Accepts floats in lengths_to_show that match the sweep values.
    """
    plt.figure(figsize=(7, 5))
    for L in lengths_to_show:
        key = float(L)
        data = results['time_series'][key]
        Uu = data['undamped']
        Ud = data['damped']
        plt.plot(Uu[:, 0], Uu[:, 2], label=f'L={key:g} m (undamped)')
        plt.plot(Ud[:, 0], Ud[:, 2], '--', label=f'L={key:g} m (damped)')
    plt.xlabel('Cart Position x [m]')
    plt.ylabel('Pendulum Angle θ [rad]')
    plt.title('Phase Portraits for Selected Cable Lengths')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Q2: Damping Sweep Plots (Stability & Settling Time)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

def plot_settling_time_vs_b(results):
    b = results['b']
    ts = results['t_settle']
    plt.figure(figsize=(6, 4))
    plt.plot(b, ts, 'o-', linewidth=2)
    plt.xlabel('Damping Coefficient b')
    plt.ylabel('Settling Time t_s [s]')
    plt.title('Settling Time vs. Damping')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_overshoot_vs_b(results):
    b = results['b']
    mp_th = results['overshoot_theta']
    mp_x = results['overshoot_x']
    plt.figure(figsize=(6, 4))
    plt.plot(b, mp_th, 'o-', label='|θ|_{max}', linewidth=2)
    plt.plot(b, mp_x, 's--', label='|x|_{max}', linewidth=2)
    plt.xlabel('Damping Coefficient b')
    plt.ylabel('Peak Overshoot')
    plt.title('Peak Overshoot vs. Damping')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_energy_decay_examples(results, b_list):
    """
    Plot normalized total energy E(t)/E(0) for selected b values.
    """
    plt.figure(figsize=(7, 4))
    for b_val in b_list:
        data = results['time_series'][float(b_val)]
        t = data['t']; E = data['energy']
        E0 = E[0] if E[0] != 0 else 1.0
        plt.plot(t, E / E0, label=f'b={float(b_val):g}')
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Energy E(t)/E(0)')
    plt.title('Energy Decay for Selected Damping Values')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_phase_by_b(results, b_list):
    """
    x vs θ phase portraits for selected b values.
    """
    plt.figure(figsize=(7, 5))
    for b_val in b_list:
        data = results['time_series'][float(b_val)]
        U = data['u']
        plt.plot(U[:, 0], U[:, 2], label=f'b={float(b_val):g}')
    plt.xlabel('Cart Position x [m]')
    plt.ylabel('Pendulum Angle θ [rad]')
    plt.title('Phase Portraits for Selected Damping Values')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



# ============================================================
# Q3: Stiffness Sweep Plots
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

def plot_frequency_vs_k(results):
    k = results['k']
    plt.figure(figsize=(6,4))
    plt.plot(k, results['freq_theta'], 'o-', label='θ dominant freq', linewidth=2)
    plt.plot(k, results['freq_x'], 's--', label='x dominant freq', linewidth=2)
    plt.xlabel('Spring stiffness k')
    plt.ylabel('Dominant Frequency [Hz]')
    plt.title('Dominant Frequency vs. Spring Stiffness')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_amplitude_vs_k(results):
    k = results['k']
    plt.figure(figsize=(6,4))
    plt.plot(k, results['theta_max'], 'o-', label='|θ|_{max}', linewidth=2)
    plt.plot(k, results['x_max'], 's--', label='|x|_{max}', linewidth=2)
    plt.xlabel('Spring stiffness k')
    plt.ylabel('Peak Amplitude')
    plt.title('Peak Amplitude vs. Spring Stiffness')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_rms_vs_k(results):
    k = results['k']
    plt.figure(figsize=(6,4))
    plt.plot(k, results['theta_rms'], 'o-', label='RMS(θ)', linewidth=2)
    plt.plot(k, results['x_rms'], 's--', label='RMS(x)', linewidth=2)
    plt.xlabel('Spring stiffness k')
    plt.ylabel('RMS over tail window')
    plt.title('RMS Response vs. Spring Stiffness')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_phase_by_k(results, k_list):
    plt.figure(figsize=(7,5))
    for kval in k_list:
        data = results['time_series'][float(kval)]
        U = data['u']
        plt.plot(U[:, 0], U[:, 2], label=f'k={float(kval):g}')
    plt.xlabel('Cart Position x [m]')
    plt.ylabel('Pendulum Angle θ [rad]')
    plt.title('Phase Portraits for Selected Stiffness Values')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_energy_partition_vs_k(results):
    k = results['k']
    plt.figure(figsize=(6,4))
    plt.plot(k, results['energy_frac_spring'], 'o-', linewidth=2)
    plt.xlabel('Spring stiffness k')
    plt.ylabel(r'$\langle E_{\mathrm{spring}}\rangle / \langle E_{\mathrm{total}}\rangle$')
    plt.title('Energy Partition vs. Spring Stiffness')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



# ============================================================
# Q4: Mass Sweep Plots (m1, m2)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

def _x_label_for_param(param):
    return r'Rover mass $m_2$' if param == 'm2' else r'Crane mass $m_1$'

def plot_frequency_vs_mass(results):
    x = results['values']; param = results['param']
    plt.figure(figsize=(6,4))
    plt.plot(x, results['freq_theta'], 'o-', label='θ dominant freq', linewidth=2)
    plt.plot(x, results['freq_x'], 's--', label='x dominant freq', linewidth=2)
    plt.xlabel(_x_label_for_param(param))
    plt.ylabel('Dominant Frequency [Hz]')
    plt.title(f'Dominant Frequency vs. {param}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_overshoot_vs_mass(results):
    x = results['values']; param = results['param']
    plt.figure(figsize=(6,4))
    plt.plot(x, results['overshoot_theta'], 'o-', label='θ percent overshoot', linewidth=2)
    plt.plot(x, results['overshoot_x'], 's--', label='x percent overshoot', linewidth=2)
    plt.xlabel(_x_label_for_param(param))
    plt.ylabel('Percent Overshoot (relative to initial)')
    plt.title(f'Peak Overshoot vs. {param}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_rms_vs_mass(results):
    x = results['values']; param = results['param']
    plt.figure(figsize=(6,4))
    plt.plot(x, results['theta_rms'], 'o-', label='RMS(θ)', linewidth=2)
    plt.plot(x, results['x_rms'], 's--', label='RMS(x)', linewidth=2)
    plt.xlabel(_x_label_for_param(param))
    plt.ylabel('RMS over tail window')
    plt.title(f'RMS Response vs. {param}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_phase_by_mass(results, values_list):
    param = results['param']
    plt.figure(figsize=(7,5))
    for val in values_list:
        data = results['time_series'][float(val)]
        U = data['u']
        plt.plot(U[:, 0], U[:, 2], label=f'{param}={float(val):g}')
    plt.xlabel('Cart Position x [m]')
    plt.ylabel('Pendulum Angle θ [rad]')
    plt.title(f'Phase Portraits for Selected {param} values')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_energy_decay_examples_mass(results, values_list):
    param = results['param']
    plt.figure(figsize=(7,4))
    for val in values_list:
        data = results['time_series'][float(val)]
        t = data['t']; E = data['energy']
        E0 = E[0] if E[0] != 0 else 1.0
        plt.plot(t, E/E0, label=f'{param}={float(val):g}')
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Energy E(t)/E(0)')
    plt.title(f'Energy Decay for Selected {param} values')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_zeta_vs_mass(results):
    x = results['values']; param = results['param']
    plt.figure(figsize=(6,4))
    plt.plot(x, results['zeta_eff'], 'o-', linewidth=2)
    plt.xlabel(_x_label_for_param(param))
    plt.ylabel(r'Effective damping ratio $\zeta_{\mathrm{eff}} = \dfrac{b}{2\sqrt{k(m_1+m_2)}}$')
    plt.title(f'Effective Damping Ratio vs. {param}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
