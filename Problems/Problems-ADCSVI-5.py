# A program to find the relative orientation of 2 spacecrafts

#%% Import libraries
import numpy as np
from math import sin, cos

#%% Data and functions
# Let the orientations of two spacecraft A and B relative to an inertial frame I 
# be given through the 3-2-1 Euler angles rotation sequences:
theta_A = np.deg2rad(np.array([60, -45, 30]).T)
theta_B = np.deg2rad(np.array([-15, 25, 10]).T)

# Direction cosine matrix for 3-2-1 Euler angles
def DCM_321(Eul_ang): 
    return np.array([[cos(Eul_ang[1])*cos(Eul_ang[2]), cos(Eul_ang[1])*sin(Eul_ang[2]), -sin(Eul_ang[1])],
                [sin(Eul_ang[0])*sin(Eul_ang[1])*cos(Eul_ang[2]) - cos(Eul_ang[0])*sin(Eul_ang[2]), sin(Eul_ang[0])*sin(Eul_ang[1])*sin(Eul_ang[2]) + cos(Eul_ang[0])*cos(Eul_ang[2]), sin(Eul_ang[0])*cos(Eul_ang[1])],
                [cos(Eul_ang[0])*sin(Eul_ang[1])*cos(Eul_ang[2]) + sin(Eul_ang[0])*sin(Eul_ang[2]), cos(Eul_ang[0])*sin(Eul_ang[1])*sin(Eul_ang[2]) - sin(Eul_ang[0])*cos(Eul_ang[2]), cos(Eul_ang[0])*cos(Eul_ang[1])]])
#%% Orientation matrices CAI and CBI
C_AI = DCM_321(theta_A)
C_BI = DCM_321(theta_B)

#%% Direction cosine matrix CAB
C_AB = np.dot(C_AI, np.linalg.inv(C_BI))

#%% Euler angles from DCM
def Eul_ang(DCM):
    return np.array([np.rad2deg(np.arctan(DCM[1,2] / DCM[2,2])),
                     np.rad2deg(-np.arcsin(DCM[0,2])),
                     np.rad2deg(np.arctan(DCM[0,1] / DCM[0,0]))])

Eul_ang_AB = Eul_ang(C_AB)
print('Euler angles of the orientation of the spacecraft A relative to spacecraft B:', (Eul_ang_AB))