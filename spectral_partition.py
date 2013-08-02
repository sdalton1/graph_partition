import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pyamg import smoothed_aggregation_solver
from scipy.sparse.linalg import lobpcg

def spectral_partition(A):
    ml = smoothed_aggregation_solver(A,
            coarse_solver='pinv2',max_coarse=100,smooth=None, strength=None)
    print ml

    M = ml.aspreconditioner()

    X = sp.rand(A.shape[0], 2) 
    (evals,evecs,res) = lobpcg(A, X, M=M, tol=1e-12, largest=False, \
        verbosityLevel=0, retResidualNormsHistory=True, maxiter=200)

    fiedler = evecs[:,1]
    vmed = np.median(fiedler)
    v = np.zeros((A.shape[0],))
    K = np.where(fiedler<=vmed)[0]
    v[K]=-1
    K = np.where(fiedler>vmed)[0]
    v[K]=1
    return v, res

def graph_laplacian_from_florida(G):
    from scipy.sparse import spdiags
    d = np.array(G.sum(axis=1)).ravel()
    D = spdiags(d,0,G.shape[0],G.shape[1]).tocsr()
    A = -1 * G + D
    return A

if __name__ == '__main__':
    from scipy.io import loadmat
    # from Florida Sparse Matrix Collection
    d = loadmat('bcsstk30.mat')
    G = d['Problem'][0][0][1].tocsr()
    x = np.loadtxt('bcsstk30.x')
    x = x[1:,:]

    import networkx
    GG=networkx.from_scipy_sparse_matrix(G)
    conn=networkx.is_connected(GG)
    print "Connected?  %s"%conn
    if not conn:
        c = networkx.connected_components(GG)
        ccid = np.argmin([len(cc) for cc in c])
        cc = np.array(c[ccid])
        cc.sort()

        print G.shape
        I=np.setdiff1d(np.arange(0,G.shape[0]),cc)
        # collapse columns
        G = G[:,I]
        # collapse rows
        G = G[I,:]
        print G.shape

    A = graph_laplacian_from_florida(G)
    v, res = spectral_partition(A)

    vminus = np.where(v<0)[0]
    vplus = np.where(v>0)[0]
    plt.figure()
    plt.hold(True)
    plt.plot(x[vminus,0],x[vminus,1],'bo')
    plt.plot(x[vplus,0],x[vplus,1],'mo')
    plt.show()
    plt.figure()
    plt.semilogy(res)
    plt.show()
