from NP import *
from graphReader import getGraph
from cliqueProject import extractClique
from cliqueProject import verifyClique
from coolRenorms import *

G = getGraph()
y, c2, errors2 = solve(G, lambda x: keepTop(x, 34), 1e-3, 1e-8, 100000000, 1000)

R = extractClique(np.outer(c2, c2))
print R
print len(R)
print verifyClique(R, G)


