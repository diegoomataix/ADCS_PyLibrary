# A program to find the Euler angles that relate the final attitude
# to the original attitude of a spacecraft that performs a 45 deg 
# single principle Euler eigenaxis rotation.

#%% Import libraries
import numpy as np
from math import sin, cos, pi

#%% Data and functions
e = 1/np.sqrt(3) * np.array([1, 1, 1])
phi = pi/4
#%% Calculations
# Skew symmetric matrix
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
# Vector product between a vector and its transpose
def vp_trans(v):
    return np.array([ [v[0]*v[0], v[0]*v[1], v[0]*v[2]],
                      [v[1]*v[0], v[1]*v[1], v[1]*v[2]],
                      [v[2]*v[0], v[1]*v[2], v[2]*v[2]]])

# The rotation matrix for a given principle Euler eigenaxis rotation is given by:
def C(e,phi):
    C = cos(phi) * np.eye(3) + np.dot((1-cos(phi)), vp_trans(e)) - sin(phi)*skew(e)
    return C

C = C(e, phi)

#%% Euler angles from DCM
def Eul_ang(DCM):
    return np.array([np.rad2deg(np.arctan(DCM[1, 2] / DCM[2, 2])),
                     np.rad2deg(-np.arcsin(DCM[0, 2])),
                     np.rad2deg(np.arctan(DCM[0, 1] / DCM[0, 0]))])

theta = Eul_ang(C)
