# A program to find the direction cosine Euler rotation matrix, the principle Euler eigenaxis, the principal Euler rotation eigenaxis,
# and the Euler parameters or quaternions

#%% Import libraries
import numpy as np
from math import sin, cos, acos

import sys
sys.path.append(
   "c:\\Users\\diego\\Dropbox\\Academic\\MEng Space Systems\\3. DOCA\\ADCS functions")
import ADCS_Functions as adcs


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
print(C21)

# Note! can also use: THIS WORKS FOR ALL TYPES OF SEQUENCES
# from ADCS_Functions_sym import *
# x, y, z = symbols('x y z')
# DCM321 = C1(x) * C2(y) * C3(z)
# DCM321 = DCM321.subs(x, Eul_ang[0])
# DCM321 = DCM321.subs(y, Eul_ang[1])
# DCM321 = DCM321.subs(z, Eul_ang[2])
# print(DCM321)

#%% Find the principle Euler eigenaxis rotation angle phi
phi = adcs.Eigenaxis_rot(C21) 
print('The principle Euler eigenaxis rotation angle is: ', np.rad2deg(phi))

#%% Find the principle Euler eigenaxis e
e = adcs.Eigenaxis_e(C21)
print(e)

#%% Verify that C21*e=e
print(np.allclose(C21@e, e))

#%% Find the corresponding Euler parameters = Quaternions
q = adcs.DCM_to_Quaternion(C21)
print(q)

#%% Is the quaternion q a unit quaternion?
print(np.allclose(q, q/np.linalg.norm(q)))
