import scipy
import numpy
import networkx as nx
import matplotlib.pyplot as plt
from warnings import warn

from scipy.sparse import csr_matrix, isspmatrix_csr

def make_graph(A):
    if not (isspmatrix_csr(A)):
        try:
            A = coo_matrix(A)
            warn("Implicit conversion of A to COO", scipy.sparse.SparseEfficiencyWarning)
        except:
            raise TypeError('Argument A must have type coo_matrix,\
                             or be convertible to coo_matrix')
    A = A.tocoo()
    G = nx.DiGraph()
    G.add_edges_from([(i,j) for (i,j) in zip(A.row,A.col) if (i != j)], capacity=1)
    G.add_nodes_from(range(A.shape[0]))

    return G

def quotient_score(G,S0,weights=None) :
    if not (isspmatrix_csr(G)):
        try:
            G = csr_matrix(G)
            warn("Implicit conversion of G to CSR", scipy.sparse.SparseEfficiencyWarning)
        except:
            raise TypeError('Argument G must have type csr_matrix or bsr_matrix,\
                             or be convertible to csr_matrix')

    if G.shape[0] != G.shape[1]:
        raise ValueError('expected square matrix')

    S = numpy.zeros_like(S0)
    S[:] = S0 # deep copy

    if S.dtype != numpy.int32:
        S = numpy.array(S, numpy.int32)
        warn("Implicit conversion of S to int datatype", scipy.sparse.SparseEfficiencyWarning)

    if weights is None :
	weights = numpy.ones_like(S)

    part_ids = numpy.unique(S)
    min_id = numpy.min(S)
    if min_id < 0 :
	S -= min_id

    part_weights = numpy.bincount(S, weights)
    part_weights = numpy.delete(part_weights, numpy.where(part_weights == 0))

    edge_count = 0
    for i in range(G.shape[0]) :
	p1 = S[i]
	for pos in range(G.indptr[i], G.indptr[i+1]) :
		j = G.indices[pos]
		if(i > j ) :
			p2 = S[j]
			edge_count += (p1 != p2)		

    return edge_count / min(part_weights)

def rel_quotient_score(G,A,S,parts,weights=None) :
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

    edge_count = 0
    for i in range(G.shape[0]) :
	p1 = parts[i]
	for pos in range(G.indptr[i], G.indptr[i+1]) :
		j = G.indices[pos]
		if(i > j) :
			p2 = parts[j]
			edge_count += (p1 != p2)		

    score = edge_count / D

    if score <= 0 :
	score = numpy.inf
    
    return score 

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
    aug_G = make_graph(G)

    aug_G.add_node('s')
    aug_G.add_node('t')

    for n,nbrs in aug_G.adjacency_iter():
	if n in A :
	  aug_G.add_edge('s',n, capacity=weights[n]*alpha)	
	elif n in A_bar :
	  aug_G.add_edge(n,'t', capacity=weights[n]*fA*alpha)

    #for edge in nx.edge_boundary(aug_G, A) :
	#if aug_G.has_edge('s', edge[0]) :
	#	aug_G.remove_edge(edge[1],edge[0])
	#elif aug_G.has_edge('s', edge[1]) :
	#	aug_G.remove_edge(edge[0],edge[1])

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

def draw_graph(G, pos, parts, labels=None, node_size=10):
   plt.figure()
   plt.hold(True)
   nx.draw(G, pos, node_size=node_size, node_color='red', with_labels=True, nodelist=list(parts[0]))
   nx.draw(G, pos, node_size=node_size, node_color='green', with_labels=True, nodelist=list(parts[1]))
   plt.show()

def improve(G, A, score) :
   v_new = -numpy.ones_like(v)
   aug_G = augmented_graph(A,B1,rel_score)

   flow,F = nx.ford_fulkerson(aug_G, 's', 't')
   for u,v in min_cut(aug_G,F) :
     if not isinstance(u,str) :
       v_new[u] = 1
     if not isinstance(v,str) :
       v_new[v] = 1

   B1_new = numpy.where(v_new==1)[0]
   B2_new = numpy.where(v_new==-1)[0]
   parts_new = [B1_new,B2_new]

   rel_score2 = rel_quotient_score(A,B1,B1_new,v_new)
   print 'relative quotient_score : %4.2f' % rel_score2

