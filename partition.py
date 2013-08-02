import numpy
import scipy

from random import randint, seed

from scipy.sparse.linalg import lobpcg

from pyamg import smoothed_aggregation_solver
from pyamg.krylov import cg
from tracemin_fiedler import tracemin_fiedler 

from improve import edge_cuts

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
  vmin = numpy.min(x)
  P1 = coarse[numpy.where(x<=vmed)[0]]
  P2 = coarse[numpy.where(x>vmed)[0]]

  weights = numpy.zeros((A.shape[0],))
  weights[P1] = x[numpy.where(x<=vmed)[0]]
  weights[P2] = x[numpy.where(x>vmed)[0]]
  weights[ground] = vmin-1

  P1 = numpy.append(P1,ground)

  return P1,P2,weights

def rqi(A=None, x=None, k=None):
    from numpy.linalg import norm, solve
    from numpy import dot, eye
    from tracemin_fiedler import cg
    from scipy.sparse.linalg import minres

    for j in range(k):    
        u = x/norm(x)                            # normalize
        lam = dot(u,A*u) 	                 # Rayleigh quotient
	#B = A - lam * eye(A.shape[0], A.shape[1])
        #x = solve(B,u)  			 # inverse power iteration
    	#D = scipy.sparse.dia_matrix((1.0/(A.diagonal()-lam), 0), shape=A.shape)

	x,flag = minres(A,u,tol=1e-5,maxiter=30,shift=lam)
    x = x/norm(x)
    lam = dot(x,A*x)                        	 # Rayleigh quotient
    return [lam,x]

def spectral(A,eval=None,evec=None,plot=False,method='lobpcg') :

  # solve for lowest two modes: constant vector and Fiedler vector
  X = scipy.rand(A.shape[0], 2) 

  if method == 'lobpcg' :
  	# specify lowest eigenvector and orthonormalize fiedler against it
  	X[:,0] = numpy.ones((A.shape[0],))
  	X = numpy.linalg.qr(X, mode='full')[0]

  	# construct preconditioner
  	ml = smoothed_aggregation_solver(A,coarse_solver='pinv2')
  	M = ml.aspreconditioner()

  	(eval,evec,res) = lobpcg(A, X, M=M, tol=1e-5, largest=False, \
        	verbosityLevel=0, retResidualNormsHistory=True, maxiter=200)
  elif method == 'tracemin':
	res = []
	evec = tracemin_fiedler(A, residuals=res, tol=1e-5)
	evec[:,1] = rqi(A, evec[:,1], k=3)[1]
  else :
	raise InputError('Unknown method')
  
  # use the median of fiedler, as the separator
  fiedler = evec[:,1]
  vmed = numpy.median(fiedler)
  P1 = numpy.where(fiedler<=vmed)[0]
  P2 = numpy.where(fiedler>vmed)[0]

  if plot is True :
     from matplotlib.pyplot import semilogy,figure,show,title,xlabel,ylabel
     figure()
     semilogy(res)
     xlabel('Iteration')
     ylabel('Residual norm')
     title('Spectral convergence history')
     show()

  return P1,P2,fiedler

def spectral_sweep(A, fiedler) :
  m,n = A.shape

  order = fiedler[numpy.argsort(fiedler)] # sort 
  cut_location = numpy.ceil(m/2)
  vmed = order[cut_location]
  P1 = numpy.where(fiedler<=vmed)[0]
  P2 = numpy.where(fiedler>vmed)[0]

  sweep_length = numpy.ceil(A.shape[0] * 0.05)
  sweep_start = int(cut_location - sweep_length)
  sweep_end   = int(cut_location + sweep_length)

  quotient = edge_cuts(A,P1) / min(len(P1),len(P2))
  cut_value = vmed

  for i in range(sweep_start,sweep_end+1) :
    P1 = numpy.where(fiedler<=order[i])[0]
    P2 = numpy.where(fiedler>order[i])[0]
    proposed_cuts = edge_cuts(A,P1)
    proposed_quotient = proposed_cuts / min(len(P1),len(P2))
    if proposed_quotient < quotient :
	print 'original_cuts : %d, proposed cuts : %d'%(cuts,proposed_cuts)
	cut_value = order[i]
	cuts = proposed_cuts
	quotient = proposed_quotient
     
    P1 = numpy.where(fiedler<=cut_value)[0]
    P2 = numpy.where(fiedler>cut_value)[0]

    return P1,P2,cut_value

def metis(A, parts) :
   from collections import defaultdict
   from pymetis import part_graph

   adj = defaultdict(list)
   for i in range(A.shape[0]):
     adj[i] = list(A.indices[A.indptr[i]:A.indptr[i+1]])

   return part_graph(parts, adj)
