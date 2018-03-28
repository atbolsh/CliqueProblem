import numpy as np


def renLn(y, n):
    a = np.sum(y**n)**(1.0/n)
    return y/a


