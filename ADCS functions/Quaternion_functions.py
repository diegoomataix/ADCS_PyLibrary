"""
Create a system of quaternions that represents the attitude of a spacecraft.
"""
#%% Import libraries
import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt

#%% Functions
def quat_mult(q1, q2):
    """
    Multiply two quaternions.
    """
    q3 = np.zeros(4)
    q3[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    q3[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    q3[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    q3[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    return q3

def quat_inv(q):
    """
    Invert a quaternion.
    """
    q_inv = np.zeros(4)
    q_inv[0] = q[0]
    q_inv[1] = -q[1]
    q_inv[2] = -q[2]
    q_inv[3] = -q[3]
    return q_inv

def quat_norm(q):
    """
    Normalize a quaternion.
    """
    q_norm = q/norm(q)
    return q_norm

def quat_conj(q):
    """
    Conjugate a quaternion.
    """
    q_conj = np.zeros(4)
    q_conj[0] = q[0]
    q_conj[1] = -q[1]
    q_conj[2] = -q[2]
    q_conj[3] = -q[3]
    return q_conj

def quat_to_dcm(q):
    """
    Convert a quaternion to a direction cosine matrix.
    """
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    dcm = np.zeros((3,3))
    dcm[0,0] = q0**2 + q1**2 - q2**2 - q3**2
    dcm[0,1] = 2*(q1*q2 + q0*q3)
    dcm[0,2] = 2*(q1*q3 - q0*q2)
    dcm[1,0] = 2*(q1*q2 - q0*q3)
    dcm[1,1] = q0**2 - q1**2 + q2**2 - q3**2
    dcm[1,2] = 2*(q2*q3 + q0*q1)
    dcm[2,0] = 2*(q1*q3 + q0*q2)
    dcm[2,1] = 2*(q2*q3 - q0*q1)
    dcm[2,2] = q0**2 - q1**2 - q2**2 + q3**2
    return dcm

def dcm_to_quat(dcm):
    """
    Convert a direction cosine matrix to a quaternion.
    """
    q = np.zeros(4)
    q[0] = 0.5*np.sqrt(dcm[0,0] + dcm[1,1] + dcm[2,2] + 1)
    q[1] = (dcm[1,2] - dcm[2,1])/(4*q[0])
    q[2] = (dcm[2,0] - dcm[0,2])/(4*q[0])
    q[3] = (dcm[0,1] - dcm[1,0])/(4*q[0])
    return q

def quat_to_euler(q):
    """
    Convert a quaternion to Euler angles.
    """
    euler = np.zeros(3)
    euler[0] = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))
    euler[1] = np.arcsin(2*(q[0]*q[2] - q[3]*q[1]))
    euler[2] = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
    return euler

def euler_to_quat(euler):
    """
    Convert Euler angles to a quaternion.
    """
    q = np.zeros(4)
    q[0] = np.cos(euler[0]/2)*np.cos(euler[1]/2)*np.cos(euler[2]/2) + np.sin(euler[0]/2)*np.sin(euler[1]/2)*np.sin(euler[2]/2)
    q[1] = np.sin(euler[0]/2)*np.cos(euler[1]/2)*np.cos(euler[2]/2) - np.cos(euler[0]/2)*np.sin(euler[1]/2)*np.sin(euler[2]/2)
    q[2] = np.cos(euler[0]/2)*np.sin(euler[1]/2)*np.cos(euler[2]/2) + np.sin(euler[0]/2)*np.cos(euler[1]/2)*np.sin(euler[2]/2)
    q[3] = np.cos(euler[0]/2)*np.cos(euler[1]/2)*np.sin(euler[2]/2) - np.sin(euler[0]/2)*np.sin(euler[1]/2)*np.cos(euler[2]/2)
    return q

def quat_derivative(q, w):
    """
    Compute the quaternion derivative for a given angular velocity and quaternion.
    """
    q_dot = 0.5*quat_mult(q, np.append(0, w))
    return q_dot
    

#%% Example
""" Now represent the attitude of a spacecraft in LEO using these functions"""

# Define the initial attitude quaternion
q_init = np.array([1, 0, 0, 0])

# Define the initial angular velocity
w_init = np.array([0, .1, 0])

# Define the time step
dt = 0.01

# Define the time vector
time = np.arange(0, 200, dt)

# Define the quaternion container
q = np.zeros((len(time), 4))

# Define the angular velocity container
w = np.zeros((len(time), 3))

# Set the initial conditions
q[0,:] = q_init
w[0,:] = w_init

# Loop through the time vector
for i in range(1, len(time)):
    # Compute the derivative of the quaternion
    q_dot = quat_derivative(q[i-1,:], w[i-1,:])
    
    # Integrate to get the quaternion at the next time step
    q[i,:] = q[i-1,:] + q_dot*dt
    
    # Compute the angular velocity for the next time step
    w[i,:] = w[i-1,:] + np.random.randn(3)*0.001

# Convert the quaternions to Euler angles
euler = np.zeros((len(time), 3))
for i in range(0, len(time)):
    euler[i,:] = quat_to_euler(q[i,:])

# Plot the Euler angles
import matplotlib.pyplot as plt
plt.plot(time, euler[:,0], label='phi')
plt.plot(time, euler[:,1], label='theta')
plt.plot(time, euler[:,2], label='psi')
plt.xlabel('Time (sec)')
plt.ylabel('Euler Angle (rad)')
plt.legend()
plt.show()

# Plot the quaternions
plt.plot(time, q[:,0], label='q0')
plt.plot(time, q[:,1], label='q1')
plt.plot(time, q[:,2], label='q2')
plt.plot(time, q[:,3], label='q3')
plt.xlabel('Time (sec)')
plt.ylabel('Quaternion')
plt.legend()
plt.show()


#%% Example 2: include random perturbation

""" Now do the same but include the effects of perturbations."""

# Define the initial attitude quaternion
q_init = np.array([1, 0, 0, 0])

# Define the initial angular velocity
w_init = np.array([0, .1, 0])

# Define the time step
dt = 0.01

# Define the time vector
time = np.arange(0, 200, dt)

# Define the quaternion container
q = np.zeros((len(time), 4))

# Define the angular velocity container
w = np.zeros((len(time), 3))

# Set the initial conditions
q[0,:] = q_init
w[0,:] = w_init

# Loop through the time vector
for i in range(1, len(time)):
    # Compute the derivative of the quaternion
    q_dot = quat_derivative(q[i-1,:], w[i-1,:])
    
    # Integrate to get the quaternion at the next time step
    q[i,:] = q[i-1,:] + q_dot*dt
    
    # Compute the angular velocity for the next time step
    w[i,:] = w[i-1,:] + np.random.randn(3)*0.001
    
    # Compute the perturbation
    pert = np.random.randn(3)*0.001
    
    # Compute the derivative of the quaternion
    q_dot = quat_derivative(q[i-1,:], w[i-1,:] + pert)
    
    # Integrate to get the quaternion at the next time step
    q[i,:] = q[i-1,:] + q_dot*dt

# Convert the quaternions to Euler angles
euler = np.zeros((len(time), 3))
for i in range(0, len(time)):
    euler[i,:] = quat_to_euler(q[i,:])

# Plot the Euler angles
plt.plot(time, euler[:,0], label='phi')
plt.plot(time, euler[:,1], label='theta')
plt.plot(time, euler[:,2], label='psi')
plt.xlabel('Time (sec)')
plt.ylabel('Euler Angle (rad)')
plt.legend()
plt.show()

# Plot the quaternions
plt.plot(time, q[:,0], label='q0')
plt.plot(time, q[:,1], label='q1')
plt.plot(time, q[:,2], label='q2')
plt.plot(time, q[:,3], label='q3')
plt.xlabel('Time (sec)')
plt.ylabel('Quaternion')
plt.legend()
plt.show()

#%% Example 3: Include SOLAR PRESSURE RADIATION
""" Make a function that calculates the torque of the Solar Pressure Radiation """

def solar_pressure_torque(q, w, sun_vec, sun_vec_body):
    """
    Compute the torque due to solar pressure.
    """
    # Compute the unit vector from the spacecraft to the Sun
    sun_vec_sc = sun_vec/norm(sun_vec)
    
    # Compute the unit vector in the Sun direction in the body frame
    sun_vec_body = sun_vec_body/norm(sun_vec_body)
    
    # Compute the vector from the Sun to the spacecraft
    sun_vec_sc_hat = np.zeros(3)
    sun_vec_sc_hat[0] = sun_vec_sc[1]
    sun_vec_sc_hat[1] = -sun_vec_sc[0]
    sun_vec_sc_hat[2] = 0
    
    # Compute the vector from the Sun to the spacecraft in the body frame
    sun_vec_body_hat = np.zeros(3)
    sun_vec_body_hat[0] = sun_vec_body[1]
    sun_vec_body_hat[1] = -sun_vec_body[0]
    sun_vec_body_hat[2] = 0
    
    # Compute the magnitude of the force due to the Sun
    F_sun = 3.5e-6*(1.496e11)**2/((norm(sun_vec) + 6.378e6)**2)
    
    # Compute the torque due to the Sun
    torque_sun = np.cross(sun_vec_sc_hat, F_sun*sun_vec_body_hat)
    
    # Compute the torque due to the angular velocity
    torque_w = np.cross(w, np.array([0, 0, 0.1]))
    
    # Compute the total torque
    torque = torque_sun + torque_w
    
    return torque

""" Now compute the torque due to solar pressure for the entire time series """

# Define the initial attitude quaternion
q_init = np.array([1, 0, 0, 0])

# Define the initial angular velocity
w_init = np.array([0, .1, 0])

# Define the time step
dt = 0.01

# Define the time vector
time = np.arange(0, 200, dt)

# Define the quaternion container
q = np.zeros((len(time), 4))

# Define the angular velocity container
w = np.zeros((len(time), 3))

# Set the initial conditions
q[0,:] = q_init
w[0,:] = w_init

# Define the Sun vector in the inertial frame
sun_vec = np.array([1, 0, 0])

# Define the Sun vector in the body frame
sun_vec_body = np.array([0, 0, 1])

# Loop through the time vector
for i in range(1, len(time)):
    # Compute the derivative of the quaternion
    q_dot = quat_derivative(q[i-1,:], w[i-1,:])
    
    # Integrate to get the quaternion at the next time step
    q[i,:] = q[i-1,:] + q_dot*dt
    
    # Compute the angular velocity for the next time step
    w[i,:] = w[i-1,:] + np.random.randn(3)*0.001
    
    # Compute the torque due to solar pressure
    torque = solar_pressure_torque(q[i,:], w[i,:], sun_vec, sun_vec_body)
    
    # Compute the angular acceleration
    w_dot = torque/0.1
    
    # Integrate to get the angular velocity at the next time step
    w[i,:] = w[i-1,:] + w_dot*dt

# Convert the quaternions to Euler angles
euler = np.zeros((len(time), 3))
for i in range(0, len(time)):
    euler[i,:] = quat_to_euler(q[i,:])

# Plot the Euler angles
plt.plot(time, euler[:,0], label='phi')
plt.plot(time, euler[:,1], label='theta')
plt.plot(time, euler[:,2], label='psi')
plt.xlabel('Time (sec)')
plt.ylabel('Euler Angle (rad)')
plt.legend()
plt.show()

# Plot the quaternions
plt.plot(time, q[:,0], label='q0')
plt.plot(time, q[:,1], label='q1')
plt.plot(time, q[:,2], label='q2')
plt.plot(time, q[:,3], label='q3')
plt.xlabel('Time (sec)')
plt.ylabel('Quaternion')
plt.legend()
plt.show()
