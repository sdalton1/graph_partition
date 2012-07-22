from scipy.sparse import csr_matrix, coo_matrix, isspmatrix_coo
from matplotlib.collections import LineCollection
from warnings import warn

import numpy
import networkx as nx
import pylab as pl

from helper import *

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

def plotgraph(xy,edges):
    lcol = xy[edges]
    pl.figure()
    pl.hold(True)
    lc = LineCollection(xy[edges])
    pl.gca().add_collection(lc)
    pl.plot(xy[:,0], xy[:,1], 'ro')
    pl.show()

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
     c[P] = -1

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
