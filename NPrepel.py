import math
from copy import deepcopy

import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv
from numpy.linalg import svd
from numpy.linalg import norm

import matplotlib.pyplot as plt

from cliqueProject import extractClique


def graphToH(G):
    """For a connectivity matrix G, returns the corresponding H2."""
    H2 = deepcopy(G)
    np.fill_diagonal(H2, 1)
    return H2

def R1(H2):
    """Returns the projection of H2 to N-dimensional rank-1 space

    As written, this only works for symmetric matrices, but that's OK,
    becaurse H2 is a symmetric matrix."""
    u, s, v = svd(H2)
    x = v[0]
    return s[0]*np.outer(v[0], v[0])

def errorMat(X, H2):
    return X - R1(X)

def grad(E, H2):
    return E*H2

def renormL2(X):
    return X/norm(X)

def renormCube(X):
    return X/np.max(X)

def repulsiveGrad(X):
    """ Punish going into the middle."""
    # Element-wise multiplication, notice.
    # Discontinuous, but I don't want a norm that's too soft at first.
    return np.fabs(2*X - np.ones(X.shape))



def seek(X, H2, ren=renormL2, beta = 0.001, alpha=0.1, cutoff = 1e-4, maxIter = 100000, recordStep=100):
    """If recordStep = 0, then there is no recording done.
    Undefined behavior if not integer."""
    errors = []
    i = 0
    y = ren(X)
    while i < maxIter:
        #### First, the deep math.
        E = errorMat(y, H2)
        # Gradient.
        g = grad(E, H2) + alpha*repulsiveGrad(X)
        #  Grad descent.
        y = y - beta*g
        #  Renorm
        y = ren(y)
        #### Next the logistics; counters, etc.
        r = norm(E)
        if recordStep != 0 and i % recordStep == 0:
            print y
            print E
            print r
            print '#####'
            errors.append(r)
        if r < cutoff:
            break
        i += 1
    #If we did not dip below the cutoff, we did not converge.
    if r > cutoff:
        print "Warning: Error greater than cutoff, did not converge."
    #Return vector, the clique it represents, and the error trace.
    return y, extractClique(np.fabs(np.sign(np.round(y, 4)))), errors


def solve(G, ren=renormL2, beta=1e-3, alpha=0.1, cutoff=1e-4, maxIter=100000, recordStep=100):
    H2 = graphToH(G)
    print H2
    x = deepcopy(H2)
    print x
    return seek(x, H2, ren, beta, alpha, cutoff, maxIter, recordStep)


