import numpy
import scipy
import pylab
import networkx as nx

from random import randint, seed
from scipy.sparse.linalg import lobpcg

from pyamg import smoothed_aggregation_solver
from pyamg.krylov import cg

from helper import trimesh, graph_laplacian
from improve import *

meshnum = 3
nodes = 16

if meshnum==1:
    from pyamg.gallery import mesh
    V,E = mesh.regular_triangle_mesh(nodes,nodes)
if meshnum==2:
    from scipy.io import loadmat
    mesh = loadmat('crack_mesh.mat')
    V=mesh['V']
    E=mesh['E']
if meshnum==3:
    from pyamg.gallery import poisson
    mesh = poisson((nodes,nodes),format='coo')
    N=mesh.shape[0]
    grid = numpy.meshgrid(range(nodes),range(nodes))
    V=numpy.vstack(map(numpy.ravel,grid)).T
    E=numpy.vstack((mesh.row,mesh.col)).T

A = graph_laplacian(V,E)

# select ground 'dead' node
#if meshnum==1 or meshnum==3 :
#  ground = A.shape[0] / 2
#else :
seed()
ground = randint(0,A.shape[0]-1)
#from pyamg.graph import pseudo_peripheral_node
#ground = pseudo_peripheral_node(A)[0]

coarse = numpy.arange(0,A.shape[0])
coarse = numpy.delete(coarse,ground,0)

# remove ground node row and column
L = A[coarse,:][:,coarse]
r = numpy.ones((L.shape[0],))
#A.setdiag(numpy.zeros((A.shape[0],),dtype=float))
#r = -1*numpy.array(A.sum(axis=1)).ravel()
#r = numpy.delete(r,ground,0)
res = []

# construct preconditioner
ml = smoothed_aggregation_solver(L,coarse_solver='pinv2',max_coarse=10)
M = ml.aspreconditioner()

# solve system using cg
(fiedler,flag) = cg(L,r,residuals=res,tol=1e-12,M=M)

# use the median of the Fiedler vector as the separator
vmed = numpy.median(fiedler)
v = numpy.zeros((A.shape[0],),dtype=numpy.int32)
K = coarse[numpy.where(fiedler<=vmed)[0]]
v[K]=-1
K = coarse[numpy.where(fiedler>vmed)[0]]
v[K]=1

# plot the mesh and partition
pylab.interactive(True)

pylab.figure()
trimesh(V,E)
sub = pylab.gca()
sub.hold(True)
sub.scatter(V[:,0],V[:,1],marker='o',s=50,c=v)
#sub.scatter(V[:,0],V[:,1],marker='o',s=50,c=fiedler)
pylab.show()

#pylab.figure()
#pylab.semilogy(res)
#pylab.title('CG residual history')
#pylab.xlabel('iteration')
#pylab.ylabel('residual norm')
#pylab.show()

v[ground] = -1
score = quotient_score(A,v)
print 'quotient_score : %4.2f' % score

v_orig = numpy.zeros_like(v)
v_orig[:] = v

B1 = numpy.where(v==1)[0]
B2 = numpy.where(v==-1)[0]
parts = [B1,B2]
rel_score = rel_quotient_score(A,B1,B1,v)
print 'relative quotient_score : %4.2f' % rel_score

if numpy.isfinite(rel_score) :
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

	#aug_G.remove_node('s')
	#aug_G.remove_node('t')
    	#pos=nx.spring_layout(aug_G, iterations=400)
	#draw_graph(aug_G, pos, parts, node_size=700)

	pylab.figure()
	trimesh(V,E)
	sub = pylab.gca()
	sub.hold(True)
	sub.scatter(V[:,0],V[:,1],marker='o',s=50,c=v_new)
	pylab.show()

	if rel_score2 < rel_score :
	  v_new2 = -numpy.ones_like(v_orig)
	  aug_G = augmented_graph(A,B1_new,rel_score2)

	  flow,F = nx.ford_fulkerson(aug_G, 's', 't')
	  for u,v in min_cut(aug_G,F) :
	    if not isinstance(u,str) :
	      v_new2[u] = 1
	    if not isinstance(v,str) :
	      v_new2[v] = 1

	  B1_new2 = numpy.where(v_new2==1)[0]
	  B2_new2 = numpy.where(v_new2==-1)[0]
	  parts_new2 = [B1_new2,B2_new2]

	  rel_score = rel_quotient_score(A,B1_new,B1_new2,v_new2)
	  print 'relative quotient_score : %4.2f' % rel_score

	  pylab.figure()
	  trimesh(V,E)
	  sub = pylab.gca()
	  sub.hold(True)
	  sub.scatter(V[:,0],V[:,1],marker='o',s=50,c=v_new2)
	  pylab.show()

	raw_input("Press Enter to continue...")
else :
	print 'Infinite score'

