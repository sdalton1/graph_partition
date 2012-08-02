import numpy 
import pyvtk
import networkx as nx

from dolfin import Mesh
from scipy.io import mmread
from pymetis import part_graph

def make_graph(A):
    A = A.tocoo()
    G = nx.Graph()
    G.add_edges_from([(i,j) for (i,j) in zip(A.row,A.col) if (i != j)])
    G.add_nodes_from(range(A.shape[0]))
    return G

# import data
filename = 'crack.mtx'
A = mmread(filename).tocsr()

# node positions
M = Mesh(filename + '.xml.gz') 
coor = M.coordinates()
        
G = make_graph(A)

adjacency = {}
for e1,e2 in G.edges():
  adjacency.setdefault(e1, []).append(e2)
  adjacency.setdefault(e2, []).append(e1)

num_parts = 10
cuts, part_vert = part_graph(num_parts, adjacency=adjacency)

coor = numpy.append( numpy.zeros((A.shape[0],1)), coor, axis=1 )
cells = M.cells().astype(int)

vtkelements = pyvtk.VtkData(
    pyvtk.UnstructuredGrid(coor, triangle=cells),
    "Mesh",
    pyvtk.PointData(pyvtk.Scalars(part_vert, name="partition")))
vtkelements.tofile('{0}_{1}.vtk'.format(filename,num_parts))
