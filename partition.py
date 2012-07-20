import numpy
import scipy

from random import randint, seed

from scipy.sparse.linalg import lobpcg

from pyamg import smoothed_aggregation_solver
from pyamg.krylov import cg

def isoperimetric(A, ground=None, residuals=None) :

  #from pyamg.graph import pseudo_peripheral_node
  #ground = pseudo_peripheral_node(A)[0]

  # select random ground 'dead' node
  if ground is None :
    seed()
    ground = randint(0,A.shape[0]-1)

  coarse = numpy.arange(0,A.shape[0])
  coarse = numpy.delete(coarse,ground,0)

  # remove ground node row and column
  L = A[coarse,:][:,coarse]
  r = numpy.ones((L.shape[0],))

  if residuals is None :
    res = []

  # construct preconditioner
  ml = smoothed_aggregation_solver(L,coarse_solver='pinv2')
  M = ml.aspreconditioner()

  # solve system using cg
  (x,flag) = cg(L,r,residuals=res,tol=1e-12,M=M)

  # use the median of solution, x, as the separator
  vmed = numpy.median(x)
  P1 = coarse[numpy.where(x<=vmed)[0]]
  P2 = coarse[numpy.where(x>vmed)[0]]
  numpy.append(P1,ground)

  return P1,P2,ground

def spectral(A,eval=None,evec=None) :

  # solve for lowest two modes: constant vector and Fiedler vector
  X = scipy.rand(A.shape[0], 4) 
  # specify lowest eigenvector and orthonormalize fiedler against it
  X[:,0] = numpy.ones((A.shape[0],))
  X = numpy.linalg.qr(X, mode='full')[0]

  # construct preconditioner
  ml = smoothed_aggregation_solver(A,coarse_solver='pinv2')
  M = ml.aspreconditioner()

  (eval,evec,res) = lobpcg(A, X, M=M, tol=1e-8, largest=False, \
        verbosityLevel=0, retResidualNormsHistory=True, maxiter=200)
  
  # use the median of fiedler, as the separator
  fiedler = evec[:,1]
  vmed = numpy.median(fiedler)
  P1 = numpy.where(fiedler<=vmed)[0]
  P2 = numpy.where(fiedler>vmed)[0]

  return P1,P2

