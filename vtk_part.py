import numpy 
import scipy
import pyvtk
import itertools

from scipy.io import mmread
from partition import *
from helper import *

def plot_vtk(A, V, partition) :
   V = numpy.append( V, numpy.zeros((A.shape[0],1)), axis=1 )

   triples = []
   A = scipy.sparse.triu(A,k=1).tocsr()
   for i in range(A.shape[0]-1):
      row = A.indices[A.indptr[i]:A.indptr[i+1]].tolist()
      for t1,t2 in itertools.combinations(row, 2) :
	if A[t1,t2] : 
	   triples.append((i,t1,t2))

   vtkelements = pyvtk.VtkData(
       pyvtk.UnstructuredGrid(V, triangle=triples),
       "Mesh",
       pyvtk.PointData(pyvtk.Scalars(partition, name="partition")))
   vtkelements.tofile('{0}_{1}.vtk'.format(graph_name,num_parts))

meshnum = 2

# node positions
from scipy.io import loadmat
graph_name = 'crack_mesh'
mesh = loadmat('data/'+graph_name)
V=mesh['V']
E=mesh['E']

num_parts = 2
A = graph_laplacian(V,E)
#cuts, partition = metis(A, num_parts)
P1,P2,weights = spectral(A,plot=True)
plot_vtk(A,V,weights)
