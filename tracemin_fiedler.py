import numpy as np
import scipy as sp
import pyamg
from helper import *

import numpy
from numpy import inner, conjugate, asarray, mod, ravel, sqrt
from scipy.sparse.linalg.isolve.utils import make_system
from scipy.sparse.sputils import upcast
from pyamg.util.linalg import norm
from warnings import warn

def cg(A, b, x0=None, tol=1e-5, maxiter=None, xtype=None, M=None, 
       callback=None, residuals=None):
    '''Conjugate Gradient algorithm
    
    Solves the linear system Ax = b. Left preconditioning is supported.

    Parameters
    ----------
    A : {array, matrix, sparse matrix, LinearOperator}
        n x n, linear system to solve
    b : {array, matrix}
        right hand side, shape is (n,) or (n,1)
    x0 : {array, matrix}
        initial guess, default is a vector of zeros
    tol : float
        relative convergence tolerance, i.e. tol is scaled by the
        preconditioner norm of r_0, or ||r_0||_M.
    maxiter : int
        maximum number of allowed iterations
    xtype : type
        dtype for the solution, default is automatic type detection
    M : {array, matrix, sparse matrix, LinearOperator}
        n x n, inverted preconditioner, i.e. solve M A x = b.
    callback : function
        User-supplied function is called after each iteration as
        callback(xk), where xk is the current solution vector
    residuals : list
        residuals contains the residual norm history,
        including the initial residual.  The preconditioner norm
        is used, instead of the Euclidean norm.
     
    Returns
    -------    
    (xNew, info)
    xNew : an updated guess to the solution of Ax = b
    info : halting status of cg

            ==  ======================================= 
            0   successful exit
            >0  convergence to tolerance not achieved,
                return iteration count instead.  
            <0  numerical breakdown, or illegal input
            ==  ======================================= 

    Notes
    -----
    The LinearOperator class is in scipy.sparse.linalg.interface.
    Use this class if you prefer to define A or M as a mat-vec routine
    as opposed to explicitly constructing the matrix.  A.psolve(..) is
    still supported as a legacy.

    The residual in the preconditioner norm is both used for halting and
    returned in the residuals list. 

    Examples
    --------
    >>> from pyamg.krylov.cg import cg
    >>> from pyamg.util.linalg import norm
    >>> import numpy 
    >>> from pyamg.gallery import poisson
    >>> A = poisson((10,10))
    >>> b = numpy.ones((A.shape[0],))
    >>> (x,flag) = cg(A,b, maxiter=2, tol=1e-8)
    >>> print norm(b - A*x)
    10.9370700187

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems, 
       Second Edition", SIAM, pp. 262-67, 2003
       http://www-users.cs.umn.edu/~saad/books.html

    '''
    A,M,x,b,postprocess = make_system(A,M,x0,b,xtype=None)
    n = len(b)
    
    ##
    # Ensure that warnings are always reissued from this function
    import warnings
    warnings.filterwarnings('always', module='pyamg\.krylov\._cg')

    # determine maxiter
    if maxiter is None:
        maxiter = int(1.3*len(b)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')
    
    # choose tolerance for numerically zero values
    t = A.dtype.char
    eps = numpy.finfo(numpy.float).eps
    feps = numpy.finfo(numpy.single).eps
    geps = numpy.finfo(numpy.longfloat).eps
    _array_precision = {'f': 0, 'd': 1, 'g': 2, 'F': 0, 'D': 1, 'G':2}
    numerically_zero = {0: feps*1e3, 1: eps*1e6, 2: geps*1e6}[_array_precision[t]]

    # setup method
    r  = b - A*x
    z  = M*r
    p  = z.copy()
    rz = inner(r.conjugate(), z)
    
    # use preconditioner norm
    normr = sqrt(rz)

    if residuals is not None:
        residuals[:] = [normr] #initial residual 

    # Check initial guess ( scaling by b, if b != 0, 
    #   must account for case when norm(b) is very small)
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0
    if normr < tol*normb:
        return (postprocess(x), 0)

    # Scale tol by ||r_0||_M
    if normr != 0.0:
        tol = tol*normr
   
    # How often should r be recomputed
    recompute_r = 8

    iter = 0

    while True:
        Ap = A*p

        rz_old = rz
                                                  # Step # in Saad's pseudocode
        pAp = inner(Ap.conjugate(), p)             # check curvature of A
        #if pAp < 0.0:
        #    warn("\nIndefinite matrix detected in CG, aborting\n")
        #    return (postprocess(x), -1)

        alpha = rz/pAp                            # 3  
        x    += alpha * p                         # 4

        if mod(iter, recompute_r) and iter > 0:   # 5
            r-= alpha * Ap                  
        else:
            r = b - A*x

        z     = M*r                               # 6
        rz    = inner(r.conjugate(), z)

        #if rz < 0.0:                              # check curvature of M
        #    warn("\nIndefinite preconditioner detected in CG, aborting\n")
        #    return (postprocess(x), -1)

        beta  = rz/rz_old                         # 7
        p    *= beta                              # 8
        p    += z

        iter += 1
        
        normr = sqrt(rz)                          # use preconditioner norm

        if residuals is not None:
            residuals.append(normr)
        
        if callback is not None:
            callback(x)

        if normr < tol:
            return (postprocess(x), 0)
        elif rz == 0.0:
            # important to test after testing normr < tol. rz == 0.0 is an
            # indicator of convergence when r = 0.0
            warn("\nSingular preconditioner detected in CG, ceasing iterations\n")
            return (postprocess(x), -1)
        
        if iter == maxiter:
            return (postprocess(x), iter)

def tracemin_fiedler(L, tol=1e-5, max_iter=100, p=2, residuals=None) :
	# initial variables
	M = L.shape[0]
	N = L.shape[1]
	q = 2*p	
	n_conv = 0

	X_conv = np.zeros((L.shape[0],2))

	# preturb diagonal to ensure PCG doesn't fail
	L_norm = pyamg.util.linalg.infinity_norm(L)
	L_infy = L_norm * 10e-12
	L_hat = L + L_infy * sp.sparse.eye(M, N)

	# extract diagonal entries
	D = sp.sparse.dia_matrix((1/L.diagonal(), 0), shape=L.shape)
	D_hat = sp.sparse.dia_matrix((1/L_hat.diagonal(), 0), shape=L.shape)

	# initialize initial random vectors
	X = sp.rand(L.shape[0],q)

	#ml = pyamg.aggregation.smoothed_aggregation_solver(L_hat)
	#M = ml.aspreconditioner(cycle='V') 

	for k in np.arange(max_iter) :
		V,R = np.linalg.qr(X)
		LV = L * V
		H = sp.dot(V.T, LV)
		E,Y = sp.linalg.eigh(H)
		X = sp.dot(V, Y)
		LX = L * X
		XE = X * sp.sparse.dia_matrix((E, 0), shape=(q, q))
		res = sp.linalg.norm(LX - XE, sp.inf) / L_norm

        	if residuals is not None:
            		residuals.append(res)
		
		if res < tol :
			X_conv[:,n_conv] = X[:,0] 
			#X_conv = np.append(X_conv, X[:,0], axis=1)
			#X = np.delete(X, 0, axis=1)
			n_conv += 1
			if n_conv >= p : 
				break

		# deflate
		if n_conv > 0 :
			X_conv_ = X_conv[:,:n_conv]
			T = sp.dot(X_conv_, (sp.dot(X_conv_.T, X)))
			X = X - T

		W = np.zeros((L.shape[0],q))
		if k == 0 :
			for i in np.arange(q) :
				w,flag = cg(L_hat,X[:,i],tol=1e-5,M=D_hat)
				W[:,i] = w
		else :
			for i in np.arange(q) :
				w,flag = cg(L,X[:,i],tol=1e-5,M=D)
				#w,flag = cg(L,X[:,i],tol=1e-5,M=M)
				W[:,i] = w

		S = sp.dot(X.T, W)
		N = np.linalg.solve(S, sp.dot(X.T, X))
		#N = np.linalg.lstsq(S, sp.dot(X.T, X))[0]
		X = sp.dot(W,N)

	return X[:,:p]
