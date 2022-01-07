#%% This file contains a series of functions to be used for attitude determination and control

#%% Import libraries
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
