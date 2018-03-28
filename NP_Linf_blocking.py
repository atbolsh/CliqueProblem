import math
from copy import deepcopy

import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv
from numpy.linalg import svd
from numpy.linalg import norm

import matplotlib.pyplot as plt


from blocking import *

def grad(x, E, tol=1e-4):
    """Only in this form because of the loop construction;
    otherwise, one ought to show where E comes from.

    This notation only works because E has a zero diagonal,
    because H2 has diagonals all along the axis.

    Assumes norm n is an int, > 2."""
    m = np.max(E)*(1 - tol)
    r = E - m
    r = np.sign(ReLu(r))
    return np.matmul(r*E, x) 



def seek(x, H2, ren=renormL2, beta = 0.001, cutoff = 1e-4, maxIter = 100000, recordStep=100):
    """If recordStep = 0, then there is no recording done.
    Undefined behavior if not integer."""
    errors = []
    i = 0
    y = ren(x)
    M = getM(H2)
    R = []
    while i < maxIter:
        #### First, the deep math.
        # Error matrix
        E = errorMat(y, H2)
        # Gradient. CRUCIAL STEP IS HERE. THIS IS DIFFERENT.
        g = grad(x, E)
        #  Grad descent.
        y = y - beta*g
        #  ReLu
        y = ReLu(y)
        #  Renorm
        y = ren(y)
        #### Next the logistics; counters, etc.
        r = np.max(E)
        if recordStep != 0 and i % recordStep == 0:
            print y
            print E
            print r
            print '#####'
            errors.append(r)
        ### Here is the blocking:
        if recordStep != 0 and i %(10*recordStep) == 0:
            w = aboveCompetition(y, H2)
            if len(w) > 0:
               R = R + w
               print R
               print len(R)
               #np.sign to avoid local minima
               y = np.sign(pruneWhales(y, w, H2))
        if r < cutoff:
            break
        i += 1
    #If we did not dip below the cutoff, we did not converge.
    if r > cutoff:
        print "Warning: Error greater than cutoff, did not converge."
    #Return vector, the clique it represents, and the error trace.
    R += aboveCompetition(y, H2)
    return y, R, errors

def solve(G, ren=renormL2, beta=1e-3, cutoff=1e-4, maxIter=100000, recordStep=100):
    H2 = graphToH(G)
    print H2
    x = np.ones(G.shape[0])
    print x
    return seek(x, H2, ren, beta, cutoff, maxIter, recordStep)


