# A program to find the direction cosine Euler rotation matrix, the principle Euler eigenaxis, the principal Euler rotation eigenaxis,
# and the Euler parameters or quaternions

#%% Import libraries
import numpy as np
from math import sin, cos, acos

# The orientation of an object is given in terms of the 3-2-1 Euler angles (-15, 25, 10 ).
Eul_ang = np.array([-15, 25, 10])
Eul_ang = np.radians(Eul_ang)

#%% Write the direction cosine Euler rotation matrix C21.

# Use the direction cosine matrix for the  3-2-1 Euler angles
def DCM_321(Eul_ang): 
    return np.array([[cos(Eul_ang[1])*cos(Eul_ang[2]), cos(Eul_ang[1])*sin(Eul_ang[2]), -sin(Eul_ang[1])],
                [sin(Eul_ang[0])*sin(Eul_ang[1])*cos(Eul_ang[2]) - cos(Eul_ang[0])*sin(Eul_ang[2]), sin(Eul_ang[0])*sin(Eul_ang[1])*sin(Eul_ang[2]) + cos(Eul_ang[0])*cos(Eul_ang[2]), sin(Eul_ang[0])*cos(Eul_ang[1])],
                [cos(Eul_ang[0])*sin(Eul_ang[1])*cos(Eul_ang[2]) + sin(Eul_ang[0])*sin(Eul_ang[2]), cos(Eul_ang[0])*sin(Eul_ang[1])*sin(Eul_ang[2]) - sin(Eul_ang[0])*cos(Eul_ang[2]), cos(Eul_ang[0])*cos(Eul_ang[1])]])

C21 = DCM_321(Eul_ang)

#%% Find the principle Euler eigenaxis rotation angle phi

phi = acos(1/2*(C21[0,0] + C21[1,1] + C21[2,2] - 1))
print('The principle Euler eigenaxis rotation angle is: ', np.rad2deg(phi))

#%% Find the principle Euler eigenaxis e

e = (1/(2*sin(phi))) * np.array([C21[1,2] - C21[2,1], C21[2,0] - C21[0,2], C21[0,1] - C21[1,0]])
print(e)

#%% Verify that C21*e=e
print(np.allclose(C21@e, e))

#%% Find the corresponding Euler parameters = Quaternions
q = np.zeros(4)
q[3] = 1/2 * (1 + C21[0,0] + C21[1,1] + C21[2,2])**0.5
q[0:3] = (1/(4*q[3])) * np.array([C21[1, 2] - C21[2, 1],
                                  C21[2, 0] - C21[0, 2], 
                                  C21[0, 1] - C21[1, 0]])
print(q)

#%% Is the quaternion q a unit quaternion?
print(np.allclose(q, q/np.linalg.norm(q)))