#%% This file contains a series of functions to be used for attitude determination and control

#%% Import libraries
import numpy as np
from sympy.physics.mechanics import ReferenceFrame, dot, cross
from sympy import *
init_printing(use_latex='mathjax')

#%% Reference frame
class IJKReferenceFrame(ReferenceFrame):
    def __init__(self, name):
        super().__init__(name, latexs=['\mathbf{%s}_{%s}' % (
            idx, name) for idx in ("i", "j", "k")])
        self.i = self.x
        self.j = self.y
        self.k = self.z

################################################################################
#%% Basic functions
################################################################################

def vector_sym(sym1=symbols('a1'), sym2=symbols('a2'), sym3=symbols('a3')):
    """
    Function to symbolically display a 3x1 vector
    """
    a1, a2, a3 = symbols('a1 a2 a3')
    Vector = Matrix([a1, a2, a3])
    return Vector.subs(a1, sym1).subs(a2, sym2).subs(a3, sym3)

################################################################################
def crossProduct(vect_A, vect_B):
    """
    Function to find cross product symbolically of 2 3x1 vectors. 
    Made out of desperation because cross(a,b) did not find with vector_sym function
    """
    cross_P = []
    cross_P.append(vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1])
    cross_P.append(vect_A[2] * vect_B[0] - vect_A[0] * vect_B[2])
    cross_P.append(vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0])
    return Matrix(cross_P)

################################################################################
def dotProduct(vect_A, vect_B, n=3):
    """
    Function to find dot product symbolically. 
    Made out of desperation because dot(a,b) did not find with vector_sym function
    """
    product = 0
    # Loop for calculate dot product
    for i in range(0, n):
        product = product + vect_A[i] * vect_B[i]

    return product

################################################################################
def skew_sym(sym1=symbols('a1'), sym2=symbols('a2'), sym3=symbols('a3')):
    """
    Function to symbolically calculate a 3x3 skew-symmetric matrix
    """
    a1, a2, a3 = symbols('a1 a2 a3')
    Skew = Matrix([
        [0, -a3, a2],
        [a3, 0, -a1],
        [-a2, a1, 0]
    ])
    return Skew.subs(a1, sym1).subs(a2, sym2).subs(a3, sym3)

# Test the skew function
#b1, b2, b3 = symbols('b1 b2 b3')
#skew_sym(b1, b2, b3)
#skew_sym()

################################################################################
def get_matrix_terms(matrix, dotx, doty, dotz):
    """
    Expand the terms of a 3x1 matrix into a 3x3 matrix(with the coefficients).
    Example use: When deriving the kinematic differential equation get the coefficients of the 
                 3x1 matrix containing the angular velocity components (dotx, doty, dotz)
    """
    return Matrix([
        [matrix[0].coeff(dotx), matrix[0].coeff(doty), matrix[0].coeff(dotz)],
        [matrix[1].coeff(dotx), matrix[1].coeff(doty), matrix[1].coeff(dotz)],
        [matrix[2].coeff(dotx), matrix[2].coeff(doty), matrix[2].coeff(dotz)]])

################################################################################
#%% Elementary rotation matrices Functions:
################################################################################

def C1(angle=symbols("theta_1")):
    x = symbols('x')
    Rx = Matrix([
        [1, 0, 0],
        [0, cos(x), sin(x)],
        [0, -sin(x), cos(x)]])
    return Rx.subs(x, angle)


def C2(angle=symbols("theta_2")):
    y = symbols('y')
    Ry = Matrix([
        [cos(y), 0, -sin(y)],
        [0,  1, 0],
        [sin(y), 0, cos(y)]])
    return Ry.subs(y, angle)


def C3(angle=symbols("theta_3")):
    z = symbols('z')
    Rz = Matrix([
        [cos(z), sin(z), 0],
        [-sin(z),  cos(z), 0],
        [0,    0, 1]])
    return Rz.subs(z, angle)

################################################################################
def DCM(mode, rot3, rot2=None, rot1=None, Eul_ang=None):
    """
    Function to calculate the rotation matrix from the 3 angles of the Euler angles.
    Input:
        mode: 'sim' or 'num' <- sim for symbolic, num for numerical
        rot1, rot2, rot3: Rotation angles
        Eul_ang: Euler angles. 
            ! Note the angles should in be in the order: theta3, theta2, theta1, where:
            theta3 -> rotation along the z axis
            theta2 -> rotation along the y axis
            theta1 -> rotation along the x axis
    Output:
        R: Rotation matrix

    Example use: get_R_matrix('num', 3, 2, 1, np.array([-15, 25, 10]))
    """
    x, y, z = symbols('x y z')
    if rot3 == 1:
        R3 = C1(x)
    elif rot3 == 2:
        R3 = C2(x)
    elif rot3 == 3:
        R3 = C3(x)

    if rot2 == 1:
        R2 = C1(y)
    elif rot2 == 2:
        R2 = C2(y)
    elif rot2 == 3:
        R2 = C3(y)

    if rot1 == 1:
        R1 = C1(z)
    elif rot1 == 2:
        R1 = C2(z)
    elif rot1 == 3:
        R1 = C3(z)

    if rot2 == None:
        R2 = 1
    if rot1 == None:
        R1 = 1

    R = R1 * R2 * R3
   
    if mode == 'num':
        if rot2 != None and rot1 == None:
            R = R.subs(z, Eul_ang[0])
            R = R.subs(y, Eul_ang[1])
        if rot2 == None and rot1 == None:
            R = R.subs(z, Eul_ang[0])

        R = R.subs(x, Eul_ang[0])
        if rot2 != None:
            R = R.subs(y, Eul_ang[1])
        if rot1 != None:
            R = R.subs(z, Eul_ang[2])
        R = np.array(R).astype(np.float64)
    return R
