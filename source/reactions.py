"""
reactions.py
------------
Defines reaction kinetics in the form u_dot = f(u).
"""

import numpy as np

def methane_combustion(u, constants, kf=1.0, kr=0.1):
    """
    Simplified reversible methane-oxygen combustion reaction:
        CH4 + 2 O2  <=>  CO2 + 2 H2O

    State vector:
        u = [CH4, O2, CO2, H2O]

    Parameters
    ----------
    u : array_like
        Current concentrations [CH4, O2, CO2, H2O]
    kf : float
        Forward rate constant
    kr : float
        Reverse rate constant

    Returns
    -------
    u_dot : ndarray
        Time derivatives [d(CH4)/dt, d(O2)/dt, d(CO2)/dt, d(H2O)/dt]
    """

    CH4, O2, CO2, H2O = u

    # Reaction rate (mass-action law)
    r_forward = kf * CH4 * O2**2
    r_reverse = kr * CO2 * H2O**2

    # Net reaction rate
    r_net = r_forward - r_reverse

    # Differential equations
    dCH4_dt = -r_net
    dO2_dt  = -2 * r_net
    dCO2_dt =  r_net
    dH2O_dt =  2 * r_net

    return np.array([dCH4_dt, dO2_dt, dCO2_dt, dH2O_dt])

def methane_combustion_CSTR(u, constants, kf=1.0, kr=0.1, D=0.1):
    CH4, O2, CO2, H2O = u
    CH4_in, O2_in, CO2_in, H2O_in = constants

    r_forward = kf * CH4 * O2**2
    r_reverse = kr * CO2 * H2O**2
    r_net = r_forward - r_reverse

    dCH4_dt = -r_net - D*(CH4 - CH4_in)
    dO2_dt  = -2*r_net - D*(O2 - O2_in)
    dCO2_dt =  r_net - D*(CO2 - CO2_in)
    dH2O_dt =  2*r_net - D*(H2O - H2O_in)

    return np.array([dCH4_dt, dO2_dt, dCO2_dt, dH2O_dt])
