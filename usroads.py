import numpy
import scipy
from scipy.sparse.linalg import lobpcg
import pylab
from pyamg import smoothed_aggregation_solver
from scipy.io import loadmat
from scipy.sparse import spdiags
def dist(a,b):
    return numpy.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
mesh=loadmat('usroads-48.mat')
V=mesh['Problem'][0][0][8][0][0][0]
A=mesh['Problem'][0][0][2].tocoo()
if 1:
    for i in range(A.nnz):
        A.data[i] = dist(V[A.col[i],:],V[A.row[i],:])

d = numpy.array(A.sum(axis=0)).ravel()
A = -A + spdiags(d,[0],A.shape[0],A.shape[1])


ml = smoothed_aggregation_solver(A, coarse_solver='pinv2',max_coarse=100)
M = ml.aspreconditioner()

X = scipy.rand(A.shape[0], 2) 
(eval,evec,res) = lobpcg(A, X, M=M, tol=1e-12, largest=False, \
        verbosityLevel=0, retResidualNormsHistory=True,maxiter=100)

fiedler = evec[:,1]
vmed = numpy.median(fiedler)
v = numpy.zeros((A.shape[0],))
K = numpy.where(fiedler<=vmed)[0]
v[K]=-1
K = numpy.where(fiedler>vmed)[0]
v[K]=1

from pylab import *
scatter(V[:,0],V[:,1],marker='o',s=10,c=v)
show()
