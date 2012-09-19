import numpy
import scipy
import pylab
import networkx as nx

from helper import *
from improve import *
from partition import *
from plotgraph import *
from test_graphs import *

from pyamg.util.utils import get_diagonal

names = ['lobpcg', 'tracemin']
method = 1 # partiton method : 1=isopermetric, 2=spectral
meshnum = 3
nodes = 100

if meshnum==1:
    from pyamg.gallery import mesh
    V,E = mesh.regular_triangle_mesh(nodes,nodes)
if meshnum==2:
    from scipy.io import loadmat
    graph_names = ['crack_mesh','random_disk_graph','random_disk_graph_1000']
    mesh = loadmat('data/'+graph_names[0])
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
    mesh = load_graph(0) 
    V=mesh['V']
    E=mesh['E']

A = graph_laplacian(V,E)

P1,P2,weights = spectral(A, method=names[method], plot=True)
cuts = edge_cuts(A,P1)

print 'cuts : ',cuts
