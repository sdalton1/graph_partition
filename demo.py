import numpy
import scipy
import pylab
import networkx as nx

from helper import *
from improve import *
from partition import *
from plotgraph import *

method = 2 # partiton method : 1=isopermetric, 2=spectral
meshnum = 2
nodes = 16

if meshnum==1:
    from pyamg.gallery import mesh
    V,E = mesh.regular_triangle_mesh(nodes,nodes)
if meshnum==2:
    from scipy.io import loadmat
    graph_names = ['crack_mesh','random_disk_graph']
    mesh = loadmat(graph_names[1])
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

if method==1 :
  P1,P2 = isoperimetric(A)
elif method == 2:
  P1,P2 = spectral(A)

v = numpy.zeros((A.shape[0],), dtype=numpy.int32)
v[P1] = -1
v[P2] =  1

# plot the mesh and partition
pylab.interactive(True)

#pylab.figure()
#trimesh(V,E)
#sub = pylab.gca()
#sub.hold(True)
#sub.scatter(V[:,0],V[:,1],marker='o',s=50,c=v)
#pylab.show()

rel_score = quotient_score(A,v)
print 'quotient_score : %4.2f' % rel_score

parts = [P1,P2]

X = scipy.rand(A.shape[0],2)
initial_pos_x = 3*(v+1) + X[:,0]
initial_pos_y = 3*(v+1) + X[:,1]
pos = dict(zip(range(A.shape[0]), zip(initial_pos_x, initial_pos_y))) # use partition ids as initial coordinates
networkx_draw_graph(A, parts, pos=pos) 

v_orig = numpy.zeros_like(v)
v_orig[:] = v

if numpy.isfinite(rel_score) :
	v_new = -numpy.ones_like(v)
	aug_G = augmented_graph(A,P1,rel_score)

	flow,F = nx.ford_fulkerson(aug_G, 's', 't')
	for u,v in min_cut(aug_G,F) :
	  if not isinstance(u,str) :
	    v_new[u] = 1
	  if not isinstance(v,str) :
	    v_new[v] = 1

	P1_new = numpy.where(v_new==1)[0]
	P2_new = numpy.where(v_new==-1)[0]
	parts_new = [P1_new,P2_new]

	rel_score2 = rel_quotient_score(A,P1,P1_new,v_new)
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

	#if rel_score2 < rel_score :
	#  v_new2 = -numpy.ones_like(v_orig)
	#  aug_G = augmented_graph(A,B1_new,rel_score2)

	#  flow,F = nx.ford_fulkerson(aug_G, 's', 't')
	#  for u,v in min_cut(aug_G,F) :
	#    if not isinstance(u,str) :
	#      v_new2[u] = 1
	#    if not isinstance(v,str) :
	#      v_new2[v] = 1

	#  B1_new2 = numpy.where(v_new2==1)[0]
	#  B2_new2 = numpy.where(v_new2==-1)[0]
	#  parts_new2 = [B1_new2,B2_new2]

	#  rel_score = rel_quotient_score(A,B1_new,B1_new2,v_new2)
	#  print 'relative quotient_score : %4.2f' % rel_score

	#  pylab.figure()
	#  trimesh(V,E)
	#  sub = pylab.gca()
	#  sub.hold(True)
	#  sub.scatter(V[:,0],V[:,1],marker='o',s=50,c=v_new2)
	#  pylab.show()

	if run_from_ipython() is False :
	  raw_input("Press Enter to continue...")
else :
	print 'Infinite score'

