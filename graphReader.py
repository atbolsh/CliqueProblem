import numpy as np
import math

dN     = "DIMACS_all_ascii/"

c125_9 = dN + "C125.9.clq"       #34
c250_9 = dN + "C250.9.clq"       #44
b200_2 = dN + "brock200_2.clq"   #12
b200_4 = dN + "brock200_4.clq"   #17
h8_4   = dN + "hamming8-4.clq"   #16
p15_3  = dN + "p_hat1500-3.clq"  #94

N125_9 = 125
N250_9 = 250
N200_2 = 200
N200_4 = 200
N8_4   = 256
N15_3  = 1500

def getGraph(s = c125_9, n = N125_9):
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





