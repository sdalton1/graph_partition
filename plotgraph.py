from scipy.sparse import csr_matrix, coo_matrix, isspmatrix_coo
from matplotlib.collections import LineCollection
from warnings import warn

import numpy
import networkx as nx
import pylab as pl
import numpy as np

from helper import *
from spy import *

def make_graph(A):
    #if not (isspmatrix_coo(A)):
    #    try:
    #        A_coo = coo_matrix(A)
    #       warn("Implicit conversion of A to COO", scipy.sparse.SparseEfficiencyWarning)
    #    except:
    #        raise TypeError('Argument A must have type coo_matrix,\
    #                         or be convertible to coo_matrix')
    A = A.tocoo()
    G = nx.Graph()
    G.add_edges_from([(i,j) for (i,j) in zip(A.row,A.col) if (i != j)])
    G.add_nodes_from(range(A.shape[0]))

    return G

def plotgraph(xy,edges,edgecolor='b'):
    lcol = xy[edges]
    lc = LineCollection(xy[edges])
    lc.set_linewidth(0.1)
    lc.set_color(edgecolor)
    pl.gca().add_collection(lc)
    #pl.plot(xy[:,0], xy[:,1], 'ro')
    pl.xlim(xy[:,0].min(), xy[:,0].max())
    pl.ylim(xy[:,1].min(), xy[:,1].max())
    pl.show()

def plotsplitting(xy,edges,splitting):
    u = np.where(splitting==1)[0]
    v = np.where(splitting==-1)[0]
    edgeflag = splitting[edges].sum(axis=1)
    uedges = np.where(edgeflag==2)[0]
    vedges = np.where(edgeflag==-2)[0]
    wedges = np.where(edgeflag==0)[0]
    pl.figure()
    pl.hold(True)
    plotgraph(xy,edges[uedges,:],edgecolor='b')
    plotgraph(xy,edges[vedges,:],edgecolor='r')
    plotgraph(xy,edges[wedges,:],edgecolor='g')


def networkx_draw_graph(A, P1, pos=None, with_labels=False, node_size=50):
   G = make_graph(A)
   pos = nx.spring_layout(G, pos=pos, iterations=200, scale=100)
   P2 = numpy.setdiff1d(numpy.arange(A.shape[0]), P1)

   pl.figure()
   pl.hold(True)
   nx.draw(G, pos, node_size=node_size, node_color='red', with_labels=with_labels, nodelist=list(P1), width=0.1)
   nx.draw(G, pos, node_size=node_size, node_color='blue', with_labels=with_labels, nodelist=list(P2), width=0.1)
   pl.show()

def draw_graph(V,E,P,title,subplot=None,c=None) :
   show_colors = True
   if c is None :
     show_colors = False
     c = numpy.ones(V.shape[0])
     c[P] = -0.1

   if subplot is None :
     pl.figure()
     sub = pl.gca()
     trimesh(V,E)
     sub.hold(True)
     cax = sub.scatter(V[:,0],V[:,1],marker='o',s=50,c=c)
     sub.title(title)
     pl.show()
   else :
     pl.subplot(subplot)
     trimesh(V,E)
     pl.hold(True)
     cax = pl.scatter(V[:,0],V[:,1],marker='o',s=50,c=c)
     pl.title(title)

   if show_colors :
     minc = numpy.min(c)
     maxc = numpy.max(c)
     medc = numpy.median(c)
     fig = pl.gcf()
     cbar = fig.colorbar(cax, ticks=[minc,medc,maxc], shrink=0.75)
     cbar.ax.set_yticklabels(['%4.2f'%minc, '%4.2f'%medc, '%4.2f'%maxc])

def plotperms(A, P1, title=None, subplot=None) :
    if subplot is None:
	subplot = 111
	pl.figure()

    #P2 = numpy.setdiff1d(numpy.arange(A.shape[0]), P1)
    #v = numpy.zeros((A.shape[0],), dtype=numpy.int32)
    #v[P1] = 1
    #v[P2] = -1
    #perm = v.argsort()
    #A_perm = A[perm,:][:,perm]

    ax = pl.subplot(subplot)
    #pl.title(title)
    #pl.spy(A_perm, marker='.', markersize=5)

    Spy(A, P1, '.', title, ax) 

if __name__=="__main__":
    from scipy.io import loadmat
    d = loadmat('random_disk_graph_1000.mat')
    splitting = loadmat('random_disk_graph_1000_splitting.mat')['splitting'].ravel()
    #plotgraph(d['V'],d['E'])
    plotsplitting(d['V'],d['E'],splitting)
