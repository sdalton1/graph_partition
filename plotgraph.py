from scipy.sparse import csr_matrix, coo_matrix, isspmatrix_coo
from matplotlib.collections import LineCollection
from warnings import warn

import networkx as nx
import pylab as pl

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

def networkx_draw_graph(A, parts, pos=None, with_labels=False, node_size=50):
   G = make_graph(A)
   pos = nx.spring_layout(G, pos=pos, iterations=500, scale=100)

   pl.figure()
   pl.hold(True)
   nx.draw(G, pos, node_size=node_size, node_color='red', with_labels=with_labels, nodelist=list(parts[0]), width=0.1)
   nx.draw(G, pos, node_size=node_size, node_color='blue', with_labels=with_labels, nodelist=list(parts[1]), width=0.1)
   pl.show()

