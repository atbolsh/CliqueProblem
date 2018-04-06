"""
Same as blocking, but more aggressive pruning.
"""

import math
from copy import deepcopy

import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv
from numpy.linalg import svd
from numpy.linalg import norm

import matplotlib.pyplot as plt


from NPLn import *

def aboveCompetition(y, H2, n=2):
    """Prints a list of indeces of y which are guaranteed not to die out."""
    R = []
    #Sum of all rivals.
    adv = np.matmul(np.ones(H2.shape) - H2, y**(n-1))
    for i in range(len(y)):
        if y[i]**(n-1) > adv[i]:
            R.append(i)
    return R

def pruneWhales(y, whales, H2):
    """Returns a version of y without the biggest, guaranteed entries, 
    and without their competitors."""
    z = deepcopy(y)
    for w in whales:
        z[w] = 0
        z = z*H2[w]
    return z


def seek(x, H2, n=8, ren=renormL2, beta = 0.001, cutoff = 1e-4, maxIter = 100000, recordStep=100):
    """If recordStep = 0, then there is no recording done.
    Undefined behavior if not integer."""
    errors = []
    N = x.shape[0]
    i = 0
    y = ren(x)
    M = getM(H2)
    R = []
    while i < maxIter:
        #### First, the deep math.
        # Error matrix
        E = errorMat(y, H2)
        # Gradient. CRUCIAL STEP IS HERE. THIS IS DIFFERENT.
        g = grad(x, E, n)
        #  Grad descent.
        y = y - beta*g
        #  ReLu
        y = ReLu(y)
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
        ### Here is the blocking:
        if recordStep != 0 and i %(10*recordStep) == 0:
            w = aboveCompetition(y, H2, n)
            if len(w) > 0:
               R = R + w
               print R
               print len(R)
               #np.sign to avoid local minima
               #Randomness to avoid getting trapped by symmetries.
               y = np.sign(pruneWhales(y, w, H2))*(0.999*np.ones(N) + 0.001*np.random.rand(N))
        if r < cutoff or np.sum(y)<1e-8:
            break
        i += 1
    #If we did not dip below the cutoff, we did not converge.
    if r > cutoff:
        print "Warning: Error greater than cutoff, did not converge."
    #Return vector, the clique it represents, and the error trace.
    R += aboveCompetition(y, H2)
    return y, R, errors

def solve(G, n = 8, ren=renormL2, beta=1e-3, cutoff=1e-4, maxIter=100000, recordStep=100):
    H2 = graphToH(G)
    print H2
    x = np.ones(G.shape[0])
    print x
    return seek(x, H2, n, ren, beta, cutoff, maxIter, recordStep)


