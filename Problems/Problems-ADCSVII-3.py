""" 
The initial yaw, pitch, and roll angles (of a 3-2-1 Euler angle sequence) 
of a vehicle are (theta1, theta2, theta3) = (80, 30, 40) at time t=0.
Assume that the angular velocity vector of the vehicle is given in body
frame components as: (w1, w2, w3) = (sin(0.1*t), 0.01, cos(0.1*t)) * 5 deg/s.
The time t is given in seconds.

A program to numerically integrate the yaw, pitch and roll angles over a 
simulation time of 1 minute. Integrate using radians but show the results in 
degrees.
"""

#%% Import libraries
import numpy as np
from math import sin, cos
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#%% Data
theta = np.deg2rad([80, 30, 40])
time = np.linspace(0, 60, 61)
#%% Functions
def diff_kinem_321_Euler(time, theta):
    """
    Differential kinematics for a 3-2-1 Euler angle sequence.
    """
    w = np.array([np.sin(0.1*time), 0.01, np.cos(0.1*time)]) * np.deg2rad(5)

    dot_angles = np.dot( (1/cos(theta[1]) * np.array([
                [cos(theta[1]), sin(theta[0])*sin(theta[1]), cos(theta[0])*sin(theta[1])],
                [0,           cos(theta[0])*cos(theta[1]), -sin(theta[0])*cos(theta[1])],
                [0,           sin(theta[0]),             cos(theta[0])] 
                ])), w)
    return dot_angles

#%% Solve kinematic differential equation in matrix form
sol = solve_ivp(diff_kinem_321_Euler, [0,60], theta, t_eval=time)

time_sol = sol.t
dot_angles = np.rad2deg(sol.y)

#%% Plot the results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(time_sol, dot_angles[0,:], label='yaw')
plt.plot(time_sol, dot_angles[1,:], label='pitch')
plt.plot(time_sol, dot_angles[2,:], label='roll')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
plt.grid()
plt.show()

with plt.style.context('seaborn-notebook'):
    fig = plt.figure(figsize=(10,5))
    plt.subplot(311)
    plt.plot(time_sol, dot_angles[0,:], label='yaw')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw Angle (deg)')
    plt.xlim(0,60)
    plt.ylim(50,200)
    
    plt.subplot(312)
    plt.plot(time_sol, dot_angles[1,:], label='pitch', color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Angle (deg)')
    plt.xlim(0,60)
    plt.ylim(-40,40)
    
    plt.subplot(313)
    plt.plot(time_sol, dot_angles[2,:], label='roll', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Roll Angle (deg)')
    plt.xlim(0,60)
    plt.ylim(0,200)
    
    fig.tight_layout()
    plt.show()