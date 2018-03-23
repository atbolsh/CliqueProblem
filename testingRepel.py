from NPrepel import *
from graphReader import getGraph
from cliqueProject import extractClique
from cliqueProject import verifyClique

G = np.array(
[[0, 1, 1, 0, 0],
[1, 0, 1, 0, 1],
[1, 1, 0, 1, 0],
[0, 0, 1, 0, 0],
[0, 1, 0, 0, 0]])

x, c, errors = solve(G, renormL2, 1e-3, 0, 1e-8, 10000, 100)
print c


G = getGraph()
y, R, errors2 = solve(G, renormCube, 1e-2, 0.3, 1e-4, 100000000000, 1000)

print R
print len(R)
print verifyClique(R, G)




