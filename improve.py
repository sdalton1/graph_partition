import scipy
import numpy
import networkx as nx
import matplotlib.pyplot as plt
from warnings import warn

from scipy.sparse import csr_matrix, coo_matrix, isspmatrix_coo, isspmatrix_csr

def make_graph(A):
    if not (isspmatrix_coo(A)):
        try:
            A = coo_matrix(A)
            warn("Implicit conversion of A to COO", scipy.sparse.SparseEfficiencyWarning)
        except:
            raise TypeError('Argument A must have type coo_matrix,\
                             or be convertible to coo_matrix')
    G = nx.DiGraph()
    G.add_edges_from([(i,j) for (i,j) in zip(A.row,A.col) if (i != j)], capacity=1)
    G.add_nodes_from(range(A.shape[0]))

    return G

def edge_cuts(G, P1) :
    if not (isspmatrix_csr(G)):
        try:
            G = csr_matrix(G)
            warn("Implicit conversion of G to CSR", scipy.sparse.SparseEfficiencyWarning)
        except:
            raise TypeError('Argument G must have type csr_matrix or bsr_matrix,\
                             or be convertible to csr_matrix')

    part = numpy.zeros(G.shape[0])
    part[P1] = 1

    edge_count = 0
    for i in range(G.shape[0]) :
	p1 = part[i]
	for pos in range(G.indptr[i], G.indptr[i+1]) :
		j = G.indices[pos]
		if(i > j) :
			p2 = part[j]
			edge_count += (p1 != p2)		

    return edge_count

def quotient_score(G,P,weights=None) :
    if not (isspmatrix_csr(G)):
        try:
            G = csr_matrix(G)
            warn("Implicit conversion of G to CSR", scipy.sparse.SparseEfficiencyWarning)
        except:
            raise TypeError('Argument G must have type csr_matrix or bsr_matrix,\
                             or be convertible to csr_matrix')

    if G.shape[0] != G.shape[1]:
        raise ValueError('expected square matrix')

    if weights is None :
	weights = numpy.ones(G.shape[0])

    part_weight = numpy.sum(weights[P])
    edge_count = edge_cuts(G, P)

    return edge_count / part_weight

def rel_quotient_score(G,A,S,weights=None) :
    if not (isspmatrix_csr(G)):
        try:
            G = csr_matrix(G)
            warn("Implicit conversion of G to CSR", scipy.sparse.SparseEfficiencyWarning)
        except:
            raise TypeError('Argument G must have type csr_matrix or bsr_matrix,\
                             or be convertible to csr_matrix')

    if G.shape[0] != G.shape[1]:
        raise ValueError('expected square matrix')

    if weights is None :
	weights = numpy.ones((G.shape[0],))

    A_bar = numpy.setdiff1d(numpy.arange(G.shape[0]), A)

    A_weight = numpy.sum(weights[A])
    A_bar_weight = numpy.sum(weights[A_bar])

    intersect = numpy.intersect1d(S, A)
    inter_weights_A = numpy.sum(weights[intersect]);

    intersect = numpy.intersect1d(S, A_bar)
    inter_weights_A_bar = numpy.sum(weights[intersect]);

    fA = A_weight/A_bar_weight
    D = inter_weights_A - inter_weights_A_bar*fA

    edge_count = edge_cuts(G, S)

    if D == 0 or edge_count == 0 :
	return numpy.inf
    else :
	return edge_count / D

def augmented_graph(G,A,alpha,weights=None) :
    if not (isspmatrix_csr(G)):
        try:
            G = csr_matrix(G)
            warn("Implicit conversion of G to CSR", scipy.sparse.SparseEfficiencyWarning)
        except:
            raise TypeError('Argument G must have type csr_matrix or bsr_matrix,\
                             or be convertible to csr_matrix')

    if G.shape[0] != G.shape[1]:
        raise ValueError('expected square matrix')

    if weights is None :
	weights = numpy.ones((G.shape[0],))

    A_bar = numpy.setdiff1d(numpy.arange(G.shape[0]), A)
    A_weight = numpy.sum(weights[A])
    A_bar_weight = numpy.sum(weights[A_bar])
    fA = A_weight/A_bar_weight
    aug_G = make_graph(G.tocoo())

    aug_G.add_node('s')
    aug_G.add_node('t')

    for n,nbrs in aug_G.adjacency_iter():
	if n in A :
	  aug_G.add_edge('s',n, capacity=weights[n]*alpha)	
	elif n in A_bar :
	  aug_G.add_edge(n,'t', capacity=weights[n]*fA*alpha)

    return aug_G

def min_cut(G, F, source='s') :
    nodes = [source]
    visited = set()

    for start in nodes:
	if start in visited :
	  continue
	visited.add(start)
	stack = [(start,iter(G[start]))]
	while stack:
	  parent,children = stack[-1]
	  try:
	    child = next(children)
	    residual = G[parent][child]['capacity'] - F[parent][child]
	    if child not in visited and residual > 0 :
		yield parent,child
		visited.add(child)
		stack.append((child,iter(G[child])))
	  except StopIteration:
	     stack.pop() 

def improve(G, A, score) :
   part_num = -numpy.ones((G.shape[0],))
   aug_G = augmented_graph(G,A,score)

   flow,F = nx.ford_fulkerson(aug_G, 's', 't')
   for u,v in min_cut(aug_G,F) :
     if not isinstance(u,str) :
       part_num[u] = 1
     if not isinstance(v,str) :
       part_num[v] = 1

   P1 = numpy.where(part_num==1)[0]
   P2 = numpy.where(part_num==-1)[0]

   return P1,P2

