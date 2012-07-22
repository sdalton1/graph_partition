import urllib2
import itertools
import numpy

from scipy.sparse import coo_matrix
from scipy.io import mmread, mmwrite

names = ['add20', 'data', '3elt', 'uk', 'add32', 'bcsstk33', 'whitaker3', \
	 'crack', 'wing_nodal', 'fe_4elt2', 'vibrobox', 'bcsstk29', '4elt', \
	 'fe_sphere', 'cti', 'memplus', 'cs4', 'bcsstk30', 'bcsstk31', 'fe_pwt', \
	 'bcsstk32', 'fe_body', 't60k', 'wing', 'brack2', 'finan512', 'fe_tooth', \
	 'fe_rotor', '598a', 'fe_ocean', '144', 'wave', 'm14b', 'auto']

base = 'http://staffweb.cms.gre.ac.uk/~c.walshaw/partition/archive/%s/%s.graph'

def load_graph(name) :

  try :
    if name is str :
    	G = mmread(name)
    else :
	G = mmread(names[name])
  except IOError as e:
    print 'Matrix market file : %s.mtx not available...downloading'%name
    url = base%(name,name)
    response = urllib2.urlopen(url)
    graph = response.read()
    adj_lists = [map(int,a.split()) for a in graph.splitlines() if a]

    num_nodes,num_edges = adj_lists[0]
    vertex_degrees = [len(edges) for edges in adj_lists[1:]]
    node_lists = [itertools.repeat(i,n) for i,n in zip(range(num_nodes), vertex_degrees)] 
    I = numpy.array(list(itertools.chain(*node_lists)))
    J = numpy.array(list(itertools.chain(*adj_lists[1:]))) - 1
    V = numpy.ones(2*num_edges)

    print len(I), len(J), len(V)
    G = coo_matrix((V,(I,J)), shape=(num_nodes,num_nodes))
    mmwrite(name, G) 

  return G
