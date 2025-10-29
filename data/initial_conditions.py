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


m1 = 1
m2 = 1
l = 5
k = 0.5
b = 0.5
g = 3.73
constants = [m1, m2, l, k, b, g]