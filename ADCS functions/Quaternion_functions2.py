# A program to carry out conversions between Euler Angles, DCM Matrices, and Quaternions.

import numpy as np
import math
import matplotlib.pyplot as plt

def EulerAngles_from_DMC(DCM):
    '''
    This function takes a DCM matrix and returns the Euler angles.
    '''
    phi = math.atan2(DCM[2,1],DCM[2,2])
    theta = math.atan2(-DCM[2,0],math.sqrt(DCM[2,1]**2+DCM[2,2]**2))
    psi = math.atan2(DCM[1,0],DCM[0,0])
    return phi, theta, psi

def quaternions_from_DMC(DCM):
    '''
    This function takes a DCM matrix and returns the quaternions.
    '''
    q0 = 0.5*math.sqrt(1+DCM[0,0]+DCM[1,1]+DCM[2,2])
    q1 = 0.5*math.sqrt(1+DCM[0,0]-DCM[1,1]-DCM[2,2])
    q2 = 0.5*math.sqrt(1-DCM[0,0]+DCM[1,1]-DCM[2,2])
    q3 = 0.5*math.sqrt(1-DCM[0,0]-DCM[1,1]+DCM[2,2])
    return q0, q1, q2, q3

def DMC_from_EulerAngles(phi, theta, psi):
    '''
    This function takes Euler angles and returns the DCM matrix.
    '''
    DCM = np.zeros((3,3))
    DCM[0,0] = math.cos(psi)*math.cos(theta)
    DCM[0,1] = math.cos(psi)*math.sin(theta)*math.sin(phi)-math.sin(psi)*math.cos(phi)
    DCM[0,2] = math.cos(psi)*math.sin(theta)*math.cos(phi)+math.sin(psi)*math.sin(phi)
    DCM[1,0] = math.sin(psi)*math.cos(theta)
    DCM[1,1] = math.sin(psi)*math.sin(theta)*math.sin(phi)+math.cos(psi)*math.cos(phi)
    DCM[1,2] = math.sin(psi)*math.sin(theta)*math.cos(phi)-math.cos(psi)*math.sin(phi)
    DCM[2,0] = -math.sin(theta)
    DCM[2,1] = math.cos(theta)*math.sin(phi)
    DCM[2,2] = math.cos(theta)*math.cos(phi)
    return DCM

def DMC_from_quaternions(q0, q1, q2, q3):
    '''
    This function takes quaternions and returns the DCM matrix.
    '''
    DCM = np.zeros((3,3))
    DCM[0,0] = q0**2+q1**2-q2**2-q3**2
    DCM[0,1] = 2*(q1*q2-q0*q3)
    DCM[0,2] = 2*(q1*q3+q0*q2)
    DCM[1,0] = 2*(q1*q2+q0*q3)
    DCM[1,1] = q0**2-q1**2+q2**2-q3**2
    DCM[1,2] = 2*(q2*q3-q0*q1)
    DCM[2,0] = 2*(q1*q3-q0*q2)
    DCM[2,1] = 2*(q2*q3+q0*q1)
    DCM[2,2] = q0**2-q1**2-q2**2+q3**2
    return DCM

# Test the functions
DCM = np.array([[1,0,0],[0,1,0],[0,0,1]])
phi, theta, psi = EulerAngles_from_DMC(DCM)
print('Euler Angles from DCM:', phi, theta, psi)
q0, q1, q2, q3 = quaternions_from_DMC(DCM)
print('Quaternions from DCM:', q0, q1, q2, q3)
