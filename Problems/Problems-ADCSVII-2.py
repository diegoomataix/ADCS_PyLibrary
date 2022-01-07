import numpy as np

#%% Data
v1b = np.array([0.8273, 0.5541, -0.0920]).T
v2b = np.array([-0.8285, 0.5522, -0.0955]).T

v1i = np.array([-0.1517, -0.9669, 0.2050]).T
v2i = np.array([-0.8393, 0.4494, -0.3044]).T

#%% Functions

# TRIAD Method
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
    t2b = np.cross(b1, b2)/np.linalg.norm(np.cross(b1, b2))
    t3b = np.cross(t1b, t2b)

    rot_tb = np.array([t1b, t2b, t3b]).T  # Rotational matrix

    # Calculate inertial coordinates
    t1r = r1
    t2r = np.cross(r1, r2)/np.linalg.norm(np.cross(r1, r2))
    t3r = np.cross(t1r, t2r)

    rot_tr = np.array([t1r, t2r, t3r]).T  # Rotational matrix

    # Calculate rotation matrix
    Cbr = np.dot(rot_tb, rot_tr.T)

    return rot_tb, rot_tr, Cbr

#%% Calculate
rot_triad_b, rot_triad_i, Cbi_triad = triad(v1b, v2b, v1i, v2i)


t1b = rot_triad_b[:, 0]
t2b = rot_triad_b[:, 1]
t3b = rot_triad_b[:, 2]

t1i = rot_triad_i[:, 0]
t2i = rot_triad_i[:, 1]
t3i = rot_triad_i[:, 2]

print('t1b = ', t1b), print('t2b = ', t2b), print('t3b = ', t3b)
print('t1i = ', t1i), print('t2i = ', t2i), print('t3i = ', t3i)

print('Cbi =', Cbi_triad)

# Check if correct
print(np.allclose(Cbi_triad@v1i, v1b))
