""" 
A spacecraft is orbiting the Earth as shown in figure below. 
As shown in the figure, at this particular location in the orbit, 
the Earth-pointing and Sun-pointing vectors are given in the ECI frame as:

    ne = [-1, 0, 0]; ns = [0, 1, 0]

Also the spacecraft attitude is obtained by rotating 45 degrees about vector zG. 
The ECI coordinate have sub-indices G and the body coordinates sub-indices b.
"""

#%% Import libraries
import numpy as np
from math import sin, cos, pi

#%% Data
# Define vectors
ne_eci = np.array([-1, 0, 0]).T
ns_eci = np.array([0, 1, 0]).T

#%% 1) Determine rotation matrix CbG from body coordinates to ECI coordinates.

# Define rotation matrix
CbG = np.array([[cos(pi/4), sin(pi/4), 0],
                [-sin(pi/4), cos(pi/4), 0],
                [0, 0, 1]])

print('rotation matrix CbG = ', CbG)

#%% 2) Determine the coordinates of Earth and Sun vectors ne and ns respectively, in the spacecraft body frame.

# Calculate body coordinates
ne_body = np.dot(CbG, ne_eci)
ns_body = np.dot(CbG, ns_eci)

print('ne_body = ', ne_body)
print('ns_body = ', ns_body)


#%%  3) Using the TRIAD method, construct the body frame triad with vectors t1b, t2b and t3b with the spacecraft coordinates of the unit vectors. Construct the reference frame triad with vectors t1i, t2i and t3i with the ECI coordinates of the unit vectors

def triad(b1, b2, r1, r2):
    """
    Input: 
     - body frame vectors: b1 and b2  
     - reference frame vectors: r1 and r2
    Output: 
     - body frame triad: t1b, t2b and t3b  
     - reference frame triad: t1i, t2i and t3i
     - rotation matrix given by triad method
    """
    # Normalize the vectors
    b1 = b1 / np.linalg.norm(b1)
    b2 = b2 / np.linalg.norm(b2)
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 / np.linalg.norm(r2)

   	# Calculate body coordinates
    t1b = b1
    t2b = np.cross(b1,b2)/np.linalg.norm(np.cross(b1,b2))
    t3b = np.cross(t1b, t2b)
    
    rot_tb = np.array([t1b, t2b, t3b]).T # Rotational matrix 

    # Calculate inertial coordinates
    t1r = r1
    t2r = np.cross(r1,r2)/np.linalg.norm(np.cross(r1,r2))
    t3r = np.cross(t1r, t2r)
    
    rot_tr = np.array([t1r, t2r, t3r]).T # Rotational matrix
    
    # Calculate rotation matrix
    Cbr = np.dot(rot_tb, rot_tr.T)
    
    return rot_tb, rot_tr, Cbr

rot_triad_b, rot_triad_i, CbG_triad = triad(ne_body, ns_body, ne_eci, ns_eci)


t1b = rot_triad_b[:,0]
t2b = rot_triad_b[:,1]
t3b = rot_triad_b[:,2]

t1i = rot_triad_i[:,0]
t2i = rot_triad_i[:,1]
t3i = rot_triad_i[:,2]

print('t1b = ', t1b), print('t2b = ', t2b), print('t3b = ', t3b)
print('t1i = ', t1i), print('t2i = ', t2i), print('t3i = ', t3i)
#%%  4) Obtain the rotation matrices of [t1b, t2b, t3b] to [t1i, t2i, t3i]

print('triad_b = ', rot_triad_b)
print('triad_i = ', rot_triad_i)

#%%  5) Using the solution to part 4), compute rotation matric CbG using TRIAD method. Compare this with the result in part 1).
print('CbG = ', CbG_triad)
print('Does the rot matrix coincide with the one from part 1?',
np.allclose(CbG, CbG_triad))

#%%  6) Using the measured vectors obtained in part 2), compute CbG using the q-method and QUEST method. 
# Verify that you obtain the same result as in part 4). Note that it should be exactly the same, 
# since no measurement noise has been added
