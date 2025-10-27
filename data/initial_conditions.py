"""
initial_condtions.py
------------
Defines initial conditions for the chemical reactions.
"""
import numpy as np


class methane_conditions:
    """Initial conditions and constants for methane combustion."""
    u0 = np.array([1.0, 2.0, 0.0, 0.0])
    species = ["CH4", "O2", "CO2", "H2O"]
    kf = 5.0
    kr = 0.1