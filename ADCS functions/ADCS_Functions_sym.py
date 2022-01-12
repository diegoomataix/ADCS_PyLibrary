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
def DCM(mode, rot3, rot2=None, rot1=None, Eul_ang=None, invorder=False):
    """
    Function to calculate the rotation matrix from the 3 angles of the Euler angles.
    Input:
        mode: 'sim' or 'num' <- sim for symbolic, num for numerical
        rot1, rot2, rot3: Rotation angles
        Eul_ang: Euler angles. 
        invorder: If True, the angles are in the inverse order.
    
        (!) Note the angles are in the order: theta3, theta2, theta1, where:
            theta1 -> rotation along the 1st rotation axis 
            theta2 -> rotation along the 2nd rotation axis 
            theta3 -> rotation along the 3rd rotation axis 

        if invorder == True: the angles are in the order: theta1, theta2, theta3, where:
            theta1 -> rotation along the 3rd rotation axis 
            theta2 -> rotation along the 2nd rotation axis 
            theta3 -> rotation along the 1st rotation axis 
    Output:
        R: Rotation matrix

    Example use: get_R_matrix('num', 3, 2, 1, np.array([-15, 25, 10]))
    """
    x, y, z = symbols('x y z')

    if invorder==False:
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

    if invorder == True:
        if rot3 == 1:
            R3 = C1(z)
        elif rot3 == 2:
            R3 = C2(z)
        elif rot3 == 3:
            R3 = C3(z)

        if rot2 == 1:
            R2 = C1(y)
        elif rot2 == 2:
            R2 = C2(y)
        elif rot2 == 3:
            R2 = C3(y)

        if rot1 == 1:
            R1 = C1(x)
        elif rot1 == 2:
            R1 = C2(x)
        elif rot1 == 3:
            R1 = C3(x)

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

################################################################################
def diff_kinem_Euler_theta_matrix(rot1, rot2, rot3, invorder=False):
    """
    Differential kinematics matrix for any Euler angle sequence.

    If the Euler kinematics differential equation is:
        d(theta) = R(theta) * d(omega) 
    [3x1 matrix] = [3x3 matrix] * [3x1 matrix]
    
    The function returns the 3x3 R(theta) matrix of the differential equation in
    symbollical form.

    This can be then used together with the diff_kinem_Euler function from the 
    numerical library to solve for any Euler kinematic differential equation.

    Input:
        rot1, rot2, rot3: Rotation angles.
        invorder: If True, the angles are in the inverse order.
    
        (!) Note the angles are in the order: theta3, theta2, theta1, where:
            theta1 -> rotation along the 1st rotation axis 
            theta2 -> rotation along the 2nd rotation axis 
            theta3 -> rotation along the 3rd rotation axis 

        if invorder == True: the angles are in the order: theta1, theta2, theta3, where:
            theta1 -> rotation along the 3rd rotation axis 
            theta2 -> rotation along the 2nd rotation axis 
            theta3 -> rotation along the 1st rotation axis 
    """

    x, y, z = symbols("theta_1 theta_2 theta_3")
    dotx, doty, dotz = symbols(r"\dot{\theta_1} \dot{\theta_2} \dot{\theta_3}")

    if invorder == False:
        if rot3 == 1:
            R3 = C1(z)
            column1 = Matrix([dotz, 0, 0])
        elif rot3 == 2:
            R3 = C2(z)
            column1 = Matrix([0, dotz, 0])
        elif rot3 == 3:
            R3 = C3(z)
            column1 = Matrix([0, 0, dotz])

        if rot2 == 1:
            R2 = C1(y)
            column2 = Matrix([doty, 0, 0])
        elif rot2 == 2:
            R2 = C2(y)
            column2 = Matrix([0, doty, 0])
        elif rot2 == 3:
            R2 = C3(y)
            column1 = Matrix([0, 0, doty])

        if rot1 == 1:
            R1 = C1(x)
            column3 = Matrix([dotx, 0, 0])
        elif rot1 == 2:
            R1 = C2(x)
            column3 = Matrix([0, dotx, 0])
        elif rot1 == 3:
            R1 = C3(x)
            column3 = Matrix([0, 0, dotx])

    if invorder==True:
        if rot3 == 1:
            R3 = C1(x)
            column1 = Matrix([dotx, 0, 0])
        elif rot3 == 2:
            R3 = C2(x)
            column1 = Matrix([0, dotx, 0])
        elif rot3 == 3:
            R3 = C3(x)
            column1 = Matrix([0, 0, dotx])

        if rot2 == 1:
            R2 = C1(y)
            column2 = Matrix([doty, 0, 0])
        elif rot2 == 2:
            R2 = C2(y)
            column2 = Matrix([0, doty, 0])
        elif rot2 == 3:
            R2 = C3(y)
            column1 = Matrix([0, 0, doty])

        if rot1 == 1:
            R1 = C1(z)
            column3 = Matrix([dotz, 0, 0])
        elif rot1 == 2:
            R1 = C2(z)
            column3 = Matrix([0, dotz, 0])
        elif rot1 == 3:
            R1 = C3(z)
            column3 = Matrix([0, 0, dotz])

    Rmatrix = simplify(column1 + R3*column2 + R3*R2*column3)
    Rmatrix = get_matrix_terms(Rmatrix, dotx, doty, dotz)
    Rmatrix = simplify(Rmatrix.inv())

    return Rmatrix

################################################################################
def diff_kinem_Euler(rot1, rot2, rot3, w=None, invorder=True):
    """
    Differential kinematics for any Euler angle sequence.

    Input:
        rot1, rot2, rot3: Rotation angles.
        w: Angular velocity vector as a sympy Matrix. i.e. Matrix([w1, w2, w3])

    (!) Note: Need to review the solver. It is not working properly.
    """
    ## Required library
    from sympy.solvers.ode.systems import dsolve_system

    ## Angular velocity vector
    time = Symbol('t')
    if w ==None:
        w = Matrix([sin(0.1*time), 0.01, cos(0.1*time)]) * np.deg2rad(5)

    ## Thetas matrix
    thetas = diff_kinem_Euler_theta_matrix(rot1, rot2, rot3, invorder=invorder)

    ## Euler angle rates
    dotx, doty, dotz = symbols(r"\dot{\theta_1} \dot{\theta_2} \dot{\theta_3}")
    dot_angles = Matrix([dotx, doty, dotz])

    ## Differential Equation
    from sympy.physics.vector import dynamicsymbols
    xt, yt, zt = dynamicsymbols("theta_1 theta_2 theta_3")
    thetanum = np.deg2rad([80, 30, 40])

    eq1 = Eq(diff(xt, time), (thetas @ w)[0])
    eq2 = Eq(diff(yt, time), (thetas @ w)[1])
    eq3 = Eq(diff(zt, time), (thetas @ w)[2])

    eqs = Matrix([eq1, eq2, eq3])

    ## Solve the differential equation
    sol = dsolve_system(eqs, t=time, ics={xt: thetanum[0], yt: thetanum[1], zt: thetanum[2]})
    sol = Matrix(sol[0])
    return sol
################################################################################
#%% Rigid Body Dynamics Functions:
################################################################################

def char_poly(J):
    """
    Inputs a matrix in sympy form.
    Finds the characteristic polynomial of a matrix in sympy form.
    Takes the coefficients of the polynomial as a numpy array and Outputs the roots of the polynomial.
    NOTE: TBH I could also just use the find_eigen function or just numpy... But this way I get the characteristic polynomial.
    """
    # J.charpoly() gives the characteristic polynomial of J. Can also write as J.charpoly().as_expr() to just get the poly equation
    char_eq = J.charpoly()
    coef = np.array(char_eq.all_coeffs())
    return J.charpoly().as_expr(), np.roots(coef)

################################################################################
def find_eigen(J):
    """
    Input: a matrix in sympy form.
    Output: the eigenvalues and eigenvectors of a matrix in sympy form as numpy arrays
    """
    # J.eigenvects() gives the eigenvectors of J. Can also write as J.eigenvects().as_expr() to just get the eigenvectors
    Eigen = np.linalg.eigh(np.array(J, dtype='float'))
    Eigenvalues = Eigen[0]
    Eigenvectors = Eigen[1]
    return Eigenvalues, Eigenvectors

################################################################################
def Inertia_cylinder():
    m = Symbol('m', positive=True, real=True)
    r = Symbol('r', positive=True, real=True)
    h = Symbol('h', positive=True, real=True)
    I1 = 1/12 * m * (3*r**2 + h**2)
    I2 = I1
    I3 = 1/2 * m * r**2
    return I1, I2, I3
