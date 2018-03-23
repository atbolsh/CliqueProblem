import numpy as np
import math

fname = "C125.9.clq"
N = 125

def getGraph(s = fname, n = N):
    f = open(s, 'r')
    l = f.readlines()
    f.close()
    edges = []
    for line in l:
        ll = line.split()
        if ll[0] == 'e':
            edges.append((int(ll[1])-1, int(ll[2])-1))
    H = np.zeros((n, n))
    for e in edges:
        H[e[0], e[1]] = 1
        H[e[1], e[0]] = 1
    return H





