import math
from copy import deepcopy

import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv
from numpy.linalg import svd
from numpy.linalg import norm

import matplotlib.pyplot as plt


#### For N-space

def keepTop(x, k):
    y = deepcopy(x)
    ind = np.argsort(np.fabs(y))
    rev = np.argsort(ind)
    y = y[ind]
    y = y/y[-k]
    y[-k:] = 1
    y = y[rev]
    return y


