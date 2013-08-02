import urllib2
import itertools
import numpy
import networkx as nx

from scipy.sparse import coo_matrix
from scipy.io import mmread, mmwrite, savemat, loadmat
from helper import *
from plotgraph import *

names = ['add20', 'data', '3elt', 'uk', 'add32', 'bcsstk33', 'whitaker3', \
	 'crack', 'wing_nodal', 'fe_4elt2', 'vibrobox', 'bcsstk29', '4elt', \
	 'fe_sphere', 'cti', 'memplus', 'cs4', 'bcsstk30', 'bcsstk31', 'fe_pwt', \
	 'bcsstk32', 'fe_body', 't60k', 'wing', 'brack2', 'finan512', 'fe_tooth', \
	 'fe_rotor', '598a', 'fe_ocean', '144', 'wave', 'm14b', 'auto']

base = 'http://staffweb.cms.gre.ac.uk/~c.walshaw/partition/archive/%s/%s.graph'

def load_graph(name) :

  dir = 'data/'
  try :
    if isinstance(name, str) :
	meshname = dir+name
    else :
	meshname = dir+names[name]

    mesh = loadmat(meshname)
  except IOError as e:
    print 'Matrix market file : %s.mtx not available...downloading'%meshname
    url = base%(meshname,meshname)
    response = urllib2.urlopen(url)
    graph = response.read()
    adj_lists = [map(int,a.split()) for a in graph.splitlines() if a]

    num_nodes,num_edges = adj_lists[0]
    vertex_degrees = [len(edges) for edges in adj_lists[1:]]
    node_lists = [itertools.repeat(i,n) for i,n in zip(range(num_nodes), vertex_degrees)] 
    I = numpy.array(list(itertools.chain(*node_lists)))
    J = numpy.array(list(itertools.chain(*adj_lists[1:]))) - 1
    V = numpy.ones(2*num_edges)

    G = coo_matrix((V,(I,J)), shape=(num_nodes,num_nodes))
    mmwrite(dir+meshname, G) 

    G_nx = make_graph(G)
    pos = nx.spring_layout(G_nx, iterations=200)
    x = numpy.array([pos[i][0] for i in range(G.shape[0])])
    y = numpy.array([pos[i][1] for i in range(G.shape[0])])
    V=numpy.vstack((x,y)).T
    E=numpy.vstack((G.row,G.col)).T
    mesh = {'V':V, 'E':E}
    savemat(dir+meshname,mesh)

  return mesh

###########################################################################
if __name__ == '__main__':
   
   # generate all test graphs
   for name in names :
	load_graph(name)
