""" 
The initial yaw, pitch, and roll angles (of a 3-2-1 Euler angle sequence) 
of a vehicle are (theta1, theta2, theta3) = (80, 30, 40) at time t=0.
Assume that the angular velocity vector of the vehicle is given in body
frame components as: (w1, w2, w3) = (sin(0.1*t), 0.01, cos(0.1*t)) * 5 deg/s.
The time t is given in seconds.

A program to numerically integrate the quaternions over a 
simulation time of 1 minute.
"""

#%% Import libraries
import numpy as np
from math import sin, cos
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#%% Data
theta = np.deg2rad([80, 30, 40])
time = np.linspace(0, 60, 244)
#%% Functions
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

def DCM_321(Eul_ang):
    """
    Direction cosine matrix for the  3-2-1 Euler angles
    """
    return np.array([[cos(Eul_ang[1])*cos(Eul_ang[2]), cos(Eul_ang[1])*sin(Eul_ang[2]), -sin(Eul_ang[1])],
                     [sin(Eul_ang[0])*sin(Eul_ang[1])*cos(Eul_ang[2]) - cos(Eul_ang[0])*sin(Eul_ang[2]), sin(Eul_ang[0])
                      * sin(Eul_ang[1])*sin(Eul_ang[2]) + cos(Eul_ang[0])*cos(Eul_ang[2]), sin(Eul_ang[0])*cos(Eul_ang[1])],
                     [cos(Eul_ang[0])*sin(Eul_ang[1])*cos(Eul_ang[2]) + sin(Eul_ang[0])*sin(Eul_ang[2]), cos(Eul_ang[0])*sin(Eul_ang[1])*sin(Eul_ang[2]) - sin(Eul_ang[0])*cos(Eul_ang[2]), cos(Eul_ang[0])*cos(Eul_ang[1])]])

#%% Solve ODE
## 1) Translate from Euler angles, theta, to quaternions, q
# Compute the corresponding rotation matrix from the 3-2-1 Euler angles
C = DCM_321(theta)
print("The corresponding rotation matrix is =", C)

# Find the corresponding Euler parameters = Quaternions
q = np.zeros(4)
q[3] = 1/2 * (1 + C[0, 0] + C[1, 1] + C[2, 2])**0.5
q[0:3] = (1/(4*q[3])) * np.array([C[1, 2] - C[2, 1],
                                  C[2, 0] - C[0, 2],
                                  C[0, 1] - C[1, 0]])
print("The quaternion is = ", q)

## 2) Solve the ODE
sol = solve_ivp(diff_kinem_Quaternion, [0, 60], q, t_eval=time)
time_sol = sol.t
q_sol = sol.y

#%% Plot the results
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

plot_quaternion(time_sol, q_sol)

#%% Given the result of the numerical integration, plot the quaternion (or
# Euler parameters) constraint |q|= 1 and comment on this constraint


def plot_quaternion_constraint(t, q):
    """
    Plot the quaternion constraint |q| = 1.
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

plot_quaternion_constraint(time_sol, q_sol)
