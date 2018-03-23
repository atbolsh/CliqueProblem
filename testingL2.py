from NP import *
from graphReader import getGraph
from cliqueProject import extractClique
from cliqueProject import verifyClique


G = np.array(
[[0, 1, 1, 0, 0],
[1, 0, 1, 0, 1],
[1, 1, 0, 1, 0],
[0, 0, 1, 0, 0],
[0, 1, 0, 0, 0]])

x, c, errors = solve(G, renormL2, 1e-3, 1e-8, 10000, 0)


G = getGraph()
y, c2, errors2 = solve(G, renormL2, 1e-3, 1e-8, 100000000, 1000)

R = extractClique(np.outer(c2, c2))
print R
print len(R)
print verifyClique(R, G)


