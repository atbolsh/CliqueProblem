from NPLn import *
from graphReader import getGraph
from BigC import biggestClique
from BigC import verifyClique

#First, a simplistic test.
G = np.array(
[[0, 1, 1, 0, 0],
[1, 0, 1, 0, 1],
[1, 1, 0, 1, 0],
[0, 0, 1, 0, 0],
[0, 1, 0, 0, 0]])

x, c, errors = solve(G, 8, renormL2, 1e-2, 1e-6, 1000000, 0)
print c



#Now, we load the full problem.
G = getGraph()
#We add the main diagonal (that's really it).
H2 = graphToH(G)
#We want the clique that has the most vertices, so we start the gradient descent with all of them
x = np.ones(125)

#Returns -- exact floating point numbers; only the ones that are part of the clique; and an error trace.
#Takes: initial guess; H matrix; norm to be used; renormalization for x; 
#stepsize; error cutoff; maxIter; how often to print.
y, c2, errors2 = seek(x, H2, 3, renormL2, 1e-2, 1e-8, 100000000000000, 1000)


#Gets the biggest k entries of y, under the condition that they form a clique.
R = biggestClique(y, G)

print R
print len(R)
print verifyClique(R, G)

