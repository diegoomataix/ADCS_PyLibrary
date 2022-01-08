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
def print_matrix(array, decimals=3):
    """
    A function to just print a matrix in Latex form. It just looks nicer.
    """
    from IPython.display import display, Latex, Math
    matrix = ''
    for row in array:
        try:
            for number in row:
                matrix += f'{round(number,decimals)}&'
        except TypeError:
            matrix += f'{round(row,decimals)}&'
        matrix = matrix[:-1] + r'\\'
    display(Math(r'\begin{bmatrix}'+matrix+r'\end{bmatrix}'))

################################################################################
def find_eig_3x3(J):
    """
    A function to find the eigenvalues and eigenvectors of a 3x3 matrix.
    """
    Eigen = np.linalg.eigh(
        J)  # Other methods include la.eig (import scipy.linalg as la) or np.linalg.eig, but give a different order for eigenvectors
    Eigenvalues = Eigen[0]
    # The column v[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i]. Will return a matrix object if a is a matrix object.
    Eigenvectors = Eigen[1]
    return Eigenvalues, Eigenvectors

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
def Eigenaxis_rotMAT(e, phi):
    """
    Input:
     - principle Euler eigenaxis e
     - principle Euler eigenaxis rotation angle phi
    Output:
     - The rotation matrix for a given principle Euler eigenaxis rotation
    """
    C = cos(phi) * np.eye(3) + np.dot((1-cos(phi)), (e.T*e)) - sin(phi)*skew(e)
    return C

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
def Quaternion_to_DCM(q):
    """
    Quaternion to DCM
    """
    q1, q2, q3, q4 = q[0], q[1], q[2], q[3]
    dcm = np.zeros((3, 3))

    dcm[0, 0] = 1 - 2*(q2**2 + q3**2)
    dcm[0, 1] = 2*(q1*q2 + q3*q4)
    dcm[0, 2] = 2*(q1*q3 - q2*q4)
    dcm[1, 0] = 2*(q2*q1 - q3*q4)
    dcm[1, 1] = 1 - 2*(q3**2 + q1**2)
    dcm[1, 2] = 2*(q2*q3 + q1*q4)
    dcm[2, 0] = 2*(q3*q1 + q2*q4)
    dcm[2, 1] = 2*(q3*q2 - q1*q4)
    dcm[2, 2] = 1 - 2*(q1**2 + q2**2)
    
    return dcm

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

################################################################################
################################################################################
#%% Attitude Determination
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
# q-METHOD
def q_method(b, RF, weights=None):
    """
    Input: 
     - body frame vectors: vb, ub, ..., un
     - reference frame vectors: vi, ui, ..., un
     - weights: weights for each vector. Input as an array
    Output: 
     - B matrix (3x3 matrix)
     - K matrix (4x4 matrix) with components: K11, k12, k22
     - Max Eigenvalue and corresponding Eigenvector
     - C Rotation matrix
    """
    if weights == None:
        weights = np.ones(len(b))

    B = np.zeros((3, 3))
    for i in range(len(b)):
        # Normalize the vectors
        b[i] = b[i] / np.linalg.norm(b[i])
        RF[i] = RF[i] / np.linalg.norm(RF[i])
        # Find B matrix
        Bb = weights[i]*np.outer(b[i], RF[i])
        B += Bb

    k22 = np.trace(B)
    K11 = B + B.T - k22*np.eye(3, 3)
    k12 = np.array(
        [(B[1, 2] - B[2, 1]), (B[2, 0] - B[0, 2]), (B[0, 1] - B[1, 0])]).T

    K = np.zeros((4, 4))
    K[0:3, 0:3] = K11
    K[0:3, 3] = k12.T
    K[3, 0:3] = k12.T
    K[3, 3] = k22

    Eigenvalues, Eigenvectors = find_eig_3x3(K)
    max_Eigenvalue = np.max(Eigenvalues)
    max_Eigenvector = Eigenvectors[:, np.where(
        Eigenvalues == np.max(Eigenvalues))]
    C = Quaternion_to_DCM(max_Eigenvector)

    return B, k22, K11, k12, K, max_Eigenvalue, max_Eigenvector, C

################################################################################
# QUEST-METHOD
def QUEST(b, RF, weights=None):
    """
    Inputs:
        b_vec: Body vectors
        RF_vec: Reference Frame vectors
        weights: Weights for the different components
    Outputs:
        S: S matrix (3x3 matrix)
        K: K matrix (4x4 matrix) with components: K11, k12, k22
        p: p vector (3x1 vector)
        q: quaternion
        C: Rotation matrix
    """
    # Weights
    if weights == None:
        weights = np.ones(len(b))

    B = np.zeros((3, 3))
    for i in range(len(b)):
        # Normalize the vectors
        b[i] = b[i] / np.linalg.norm(b[i])
        RF[i] = RF[i] / np.linalg.norm(RF[i])
        # Find B matrix
        Bb = weights[i]*np.outer(b[i], RF[i])
        B += Bb

    S = B + B.T

    k22 = np.trace(B)
    K11 = B + B.T - k22*np.eye(3, 3)
    k12 = np.array(
        [(B[1, 2] - B[2, 1]), (B[2, 0] - B[0, 2]), (B[0, 1] - B[1, 0])]).T

    K = np.zeros((4, 4))
    K[0:3, 0:3] = K11
    K[0:3, 3] = k12.T
    K[3, 0:3] = k12.T
    K[3, 3] = k22

    Eigenvalues, _ = find_eig_3x3(K)
    max_Eigenvalue = np.max(Eigenvalues)

    p = np.dot(np.linalg.inv(
        np.array([(max_Eigenvalue + k22)*np.eye(3, 3) - S])), k12).T
    p4 = np.array([p[0], p[1], p[2], 1], dtype=object).T

    q = 1 / (np.sqrt(1 + p.T @ p)) * p4
    q = np.array(q[0])

    C = Quaternion_to_DCM(q)
    return S, K, p, q, C

################################################################################
# Solve Kinematic Differential Equation
def solve_KDE(input, time_range=[0, 60], time_array=np.linspace(0, 60, 244), solver='E'):
    """
    Solve the Kinematic Differential Equation
    Input:
        input: Input data (quaternion or Euler angles)
        time_range: Time range i.e. [0,60]
        time_array: Time array i.e. time = np.linspace(0, 60, 244)
        solver: Solver: either "q" for quaternion or "E" for Euler angles
    Output:
        output: Output data (solution time, solution_data)
    """
    from scipy.integrate import solve_ivp

    if solver=='E':
        sol = solve_ivp(diff_kinem_321_Euler, [0, 60], input, t_eval=time_array)
    elif solver=='q':
        sol = solve_ivp(diff_kinem_Quaternion, [
                        0, 60], input, t_eval=time_array)
    else:
        print('Solver not found')
    
    return sol.t, sol.y