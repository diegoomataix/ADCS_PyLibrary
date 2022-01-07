#%% This file contains a series of functions to be used for attitude determination and control
import numpy as np
from math import sin, cos, acos
import matplotlib.pyplot as plt
from sympy.physics.mechanics import ReferenceFrame, dot, cross

#%% Reference frame
class IJKReferenceFrame(ReferenceFrame):
    def __init__(self, name):
        super().__init__(name, latexs=['\mathbf{%s}_{%s}' % (
            idx, name) for idx in ("i", "j", "k")])
        self.i = self.x
        self.j = self.y
        self.k = self.z

################################################################################
################################################################################
#%% Basic functions
################################################################################
def skew(vector):
    """
    Function to calculate a 3x3 skew-symmetric matrix
    """
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])

################################################################################
################################################################################
#%% DCM and Euler angles
################################################################################
def DCM_A_B(A, B):
    """
    DCM between two reference frames, A with respect to B
    """
    return np.dot(A.T, B)
    
################################################################################
def DCM_321(Eul_ang):
    """
    Direction cosine matrix for the  3-2-1 Euler angles
    """
    return np.array([[cos(Eul_ang[1])*cos(Eul_ang[2]), cos(Eul_ang[1])*sin(Eul_ang[2]), -sin(Eul_ang[1])],
                     [sin(Eul_ang[0])*sin(Eul_ang[1])*cos(Eul_ang[2]) - cos(Eul_ang[0])*sin(Eul_ang[2]), sin(Eul_ang[0])
                      * sin(Eul_ang[1])*sin(Eul_ang[2]) + cos(Eul_ang[0])*cos(Eul_ang[2]), sin(Eul_ang[0])*cos(Eul_ang[1])],
                     [cos(Eul_ang[0])*sin(Eul_ang[1])*cos(Eul_ang[2]) + sin(Eul_ang[0])*sin(Eul_ang[2]), cos(Eul_ang[0])*sin(Eul_ang[1])*sin(Eul_ang[2]) - sin(Eul_ang[0])*cos(Eul_ang[2]), cos(Eul_ang[0])*cos(Eul_ang[1])]])

################################################################################
def Eul_ang(DCM):
    """
    Euler angles from DCM
    """
    return np.array([np.rad2deg(np.arctan(DCM[1, 2] / DCM[2, 2])),
                     np.rad2deg(-np.arcsin(DCM[0, 2])),
                     np.rad2deg(np.arctan(DCM[0, 1] / DCM[0, 0]))])

################################################################################
def diff_kinem_321_Euler(time, theta):
    """
    Differential kinematics for a 3-2-1 Euler angle sequence.
    """
    w = np.array([np.sin(0.1*time), 0.01, np.cos(0.1*time)]) * np.deg2rad(5)

    dot_angles = np.dot((1/cos(theta[1]) * np.array([
        [cos(theta[1]), sin(theta[0])*sin(theta[1]),
         cos(theta[0])*sin(theta[1])],
        [0,           cos(theta[0])*cos(theta[1]), -
         sin(theta[0])*cos(theta[1])],
        [0,           sin(theta[0]),             cos(theta[0])]
    ])), w)
    return dot_angles
################################################################################
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

################################################################################
################################################################################
#%% Quaternions
################################################################################
def Eigenaxis_rot(C):
    """
    Input: 
     - direction cosine Euler rotation matrix
    Output: 
     - principle Euler eigenaxis rotation angle phi
    """
    return acos(1/2*(C[0, 0] + C[1, 1] + C[2, 2] - 1))

################################################################################
def Eigenaxis_e(C):
    """
    Input: 
     - direction cosine Euler rotation matrix
    Output: 
     - Find the principle Euler eigenaxis e
    """
    phi = Eigenaxis_rot(C)
    return (1/(2*sin(phi))) * np.array([C[1, 2] - C[2, 1], C[2, 0] - C[0, 2], C[0, 1] - C[1, 0]])

################################################################################
def Euler_eigenaxis(C):
    """
    Find the principle Euler eigenaxis rotation angle phi and principle Euler eigenaxis e
    """
    return Eigenaxis_rot(C), Eigenaxis_e(C)

################################################################################
def DCM_to_Quaternion(C):
    """
    DCM in Euler angles to Quaternion
    """
    q = np.zeros(4)
    q[3] = 1/2 * (1 + C[0, 0] + C[1, 1] + C[2, 2])**0.5
    q[0:3] = (1/(4*q[3])) * np.array([C[1, 2] - C[2, 1],
                                  C[2, 0] - C[0, 2],
                                  C[0, 1] - C[1, 0]])
    return q#/np.linalg.norm(q)

################################################################################
def diff_kinem_Quaternion(time, q):
    """
    Differential kinematics for quaternion.
    """
    w = np.array([np.sin(0.1*time), 0.01, np.cos(0.1*time), 0]) * np.deg2rad(50)

    dot_q = np.dot( (1/2) * np.array([
                [0,     w[2], -w[1], w[0]],
                [-w[2],   0,   w[0], w[1]],
                [w[1],  -w[0],   0,  w[2]],
                [-w[0], -w[1], -w[2],   0]
                ]), q)
    return dot_q

################################################################################
def plot_quaternion(t, q):
    """
    Plot the quaternion.
    """
    with plt.style.context('seaborn-notebook'):
        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(10, 10))
        fig.suptitle('Quaternions over time')
        ax[0, 0].plot(t, q[0, :], label='$q_0$')
        ax[0, 1].plot(t, q[1, :], label='$q_1$', color='r')
        ax[1, 0].plot(t, q[2, :], label='$q_2$', color='g')
        ax[1, 1].plot(t, q[3, :], label='$q_3$', color='y')
        for ax in ax.flat:
            ax.set(xlabel='Time (s)', ylabel='Quaternion')
            ax.legend()
            ax.label_outer()
            ax.grid()
            ax.set_xlim(0, 60)
        fig.tight_layout()
        plt.show()
        
################################################################################
def plot_quaternion_constraint(t, q):
    """
    Plot the quaternion constraint | q | = 1.
    """
    q_mag = q[0, :]**2 + q[1, :]**2 + q[2, :]**2 + q[3, :]**2
    with plt.style.context('seaborn-notebook'):
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 10))
        fig.suptitle('Quaternion constraint $|q|= 1$')
        ax.plot(t, q_mag, label='$|q|$')
        ax.set(xlabel='Time (s)', ylabel='$|q|$')
        ax.legend()
        ax.grid()
        ax.set_xlim(0, 60)
        fig.tight_layout()
        plt.show()
