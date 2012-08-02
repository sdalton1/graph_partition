import numpy
import scipy
from scipy.sparse.linalg import lobpcg
import pylab
from pyamg import smoothed_aggregation_solver
from scipy.io import loadmat
from scipy.sparse import spdiags
mesh=loadmat('usroads-48.mat')
V=mesh['Problem'][0][0][8][0][0][0]
Emod=mesh['Problem'][0][0][2].tocsr()
d = numpy.array(Emod.sum(axis=0)).ravel()
A = -Emod + spdiags(d,[0],Emod.shape[0],Emod.shape[1])

ml = smoothed_aggregation_solver(A, coarse_solver='pinv2',max_coarse=100)
M = ml.aspreconditioner()

X = scipy.rand(A.shape[0], 2) 
(eval,evec,res) = lobpcg(A, X, M=M, tol=1e-12, largest=False, \
        verbosityLevel=0, retResidualNormsHistory=True, maxiter=200)

print ml
pylab.semilogy(res)

pylab.figure()
fiedler = evec[:,1]
vmed = numpy.median(fiedler)
v = numpy.zeros((A.shape[0],))
K = numpy.where(fiedler<=vmed)[0]
v[K]=-1
K = numpy.where(fiedler>vmed)[0]
v[K]=1

# plot the mesh and partition
#trimesh(V,E)
#sub = pylab.gca()
#sub.hold(True)
#sub.scatter(V[:,0],V[:,1],marker='o',s=50,c=v)
from pylab import *
scatter(V[:,0],V[:,1],marker='o',s=50,c=v)
show()
#pylab.show()
