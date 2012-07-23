import numpy
import scipy
import pylab
import networkx as nx

from helper import *
from improve import *
from partition import *
from plotgraph import *
from test_graphs import *

method = 2 # partiton method : 1=isopermetric, 2=spectral
meshnum = 4
nodes = 20

dir = 'data/'
if meshnum==1:
    from pyamg.gallery import mesh
    V,E = mesh.regular_triangle_mesh(nodes,nodes)
if meshnum==2:
    from scipy.io import loadmat
    graph_names = ['crack_mesh','random_disk_graph','random_disk_graph_1000']
    mesh = loadmat(dir+graph_names[1])
    V=mesh['V']
    E=mesh['E']
if meshnum==3:
    from pyamg.gallery import poisson
    mesh = poisson((nodes,nodes),format='coo')
    N=mesh.shape[0]
    grid = numpy.meshgrid(range(nodes),range(nodes))
    V=numpy.vstack(map(numpy.ravel,grid)).T
    E=numpy.vstack((mesh.row,mesh.col)).T
if meshnum==4:
    mesh = load_graph(dir+'wing') 
    V=mesh['V']
    E=mesh['E']

A = graph_laplacian(V,E)

if method==1 :
  P1,P2,weights = isoperimetric(A)
elif method == 2:
  P1,P2,weights = spectral(A,plot=True)

parts = [P1,P2]
list_sizes = numpy.array([len(P1),len(P2)])

min_size = min(list_sizes)
max_size = max(list_sizes)
min_pos = numpy.where(list_sizes==min_size)[0][0]
max_pos = numpy.where(list_sizes==max_size)[0][0]

P1 = parts[min_pos]
P2 = parts[max_pos]

part1 = [P1]
cuts = [edge_cuts(A,P1)]
imbalance = [(len(P2)-len(P1)+0.0)/len(P2)]

rel_scores = [A.shape[0]]
rel_scores.append(quotient_score(A,P1))
print 'quotient_score : %4.5f' % rel_scores[-1]

while (rel_scores[-1] > 0) and (rel_scores[-1] < rel_scores[-2]) :
   P1_new,P2_new = improve(A, P1, rel_scores[-1])

   score = rel_quotient_score(A,P1,P1_new)

   if not numpy.isfinite(score) : break

   print 'relative quotient_score : %4.5f' % score
   rel_scores.append(score)
   cuts.append(edge_cuts(A, P1_new))

   P1_weight = len(P1_new) + 0.0
   P2_weight = len(P2_new) + 0.0
   min_weight = min(P1_weight, P2_weight)
   max_weight = max(P1_weight, P2_weight)
   imbalance.append((max_weight-min_weight)/max_weight)

   part1.append(P1_new)
   P1,P2 = P1_new,P2_new

# plot the mesh and partition
pylab.interactive(True)
x_range = range(len(rel_scores)-1)

pylab.figure()
pylab.subplot(221)
pylab.plot(x_range, rel_scores[1:], '-ob')
pylab.xlabel('Iteration')
pylab.ylabel('Quotient Ratio')

pylab.subplot(222)
pylab.plot(x_range, cuts, '-or')
pylab.xlabel('Iteration')
pylab.ylabel('Edge Cuts')

pylab.subplot(223)
imbalance = numpy.array(imbalance) * 100
pylab.plot(x_range, imbalance, '-og')
pylab.xlabel('Iteration')
pylab.ylabel('% Imbalance')

pylab.figure()
draw_graph(V, E, part1[0], 'Weights', subplot=221, c=weights)
draw_graph(V, E, part1[0], 'Before', subplot=222)
draw_graph(V, E, part1[-1],'After',  subplot=223)

pylab.figure()
plotperms(A, range(A.shape[0]), title='Original', subplot=221)
plotperms(A, part1[0], title='Spectral', subplot=222)
plotperms(A, part1[-1], title='Spectral+Improve', subplot=223)

#X = scipy.rand(A.shape[0],2)
#initial_pos_x = 4*(v+1) + X[:,0]
#initial_pos_y = 4*(v+1) + X[:,1]
#pos = dict(zip(range(A.shape[0]), zip(initial_pos_x, initial_pos_y))) # use partition ids as initial coordinates
#networkx_draw_graph(A, part1[0], pos=pos) 

if run_from_ipython() is False :
	raw_input("Press Enter to continue...")

