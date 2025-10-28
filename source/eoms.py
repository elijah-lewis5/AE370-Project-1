import numpy as np
from sympy import *

#def skycrane(u):
 #   return None

m1 = 5
m2 = 2
l = 1
g = 9.81

theta_0 = 0
x_0 = 0

theta, x, thetadot = symbols('theta x thetadot')

A = Matrix([[m1+m2, m2*l*cos(theta)],
            [cos(theta)/l, 1]])
B = Matrix([[m2*l*thetadot**2*sin(theta)],
            [-g/l * sin(theta)]])

O = A.inv() @ B
xdd = O[0]
thetadd = O[1]

