import numpy as np
from sympy import *

def skycrane(u, constants):
    x, xdot, theta, thetadot = u 
    m1 = 5
    m2 = 2
    l = 1
    g = 9.81
    m1, m2, l, g = constants

    A = Matrix([[m1+m2, m2*l*cos(theta)],
                [cos(theta)/l, 1]])
    B = Matrix([[m2*l*thetadot**2*sin(theta)],
                [-g/l * sin(theta)]])

    O = A.inv() @ B
    xddot = O[0]
    thetaddot = O[1]
    udot = np.array([xdot, xddot, thetadot, thetaddot])
    return udot

