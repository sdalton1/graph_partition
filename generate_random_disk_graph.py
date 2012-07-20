import numpy as np
import pylab as pl
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.csgraph as csg

"""
Described in "An algorithm for improving graph partitions"
                by Reid, E. and Lang, K.J.
                http://research.yahoo.com/pub/2267

(From section 3.2, page 657, numbering added)

[1] First we choose ten thousand points from a uniform distribution on the unit
disk (not the unit square). 

[2] We generate a set of candidate edges consisting of each point and its 50 nearest
neighbors. 

[3] These candidate edges are then added to the graph in order from shortest to
longest, stopping when the graph becomes connected. 

[4] Finally, we add 2000 additional completely random edges. 

[5] The resulting graphs contain 10000 nodes and an average of 58913 edges, for
an average degree of 11.7826.
"""

##################################
#[1]
##################################
nv = 1000
xy = np.zeros((nv,2))
np.random.seed(625) # anything (not in paper)
i=0
while i<nv:
    newxy = np.random.rand(1,2) * 2 - 1.0
    if np.sqrt(newxy[:,0]**2 + newxy[:,1]**2)<=1.0:
        xy[i,:]=newxy
        i+=1

#pl.figure()
#pl.scatter(xy[:,0],xy[:,1])
#pl.show()

##################################
#[2]
##################################
nn = 50
from scipy.spatial import cKDTree
tree = cKDTree(xy)

I = np.zeros((nv*nn,),dtype=int)
J = np.zeros((nv*nn,),dtype=int)
dist = np.zeros((nv*nn,))
for i in range(nv):
    d = tree.query(xy[i,:],nn+1) # first edges is self edge, so use nn+1
    assert len(d[0])==nn+1, "vertex %d does not have %d neighbors" % (i,nn)
    I[i*nn:((i+1)*nn)] = i*np.ones((nn,))
    J[i*nn:((i+1)*nn)] = d[1][1:]
    dist[i*nn:((i+1)*nn)] = d[0][1:]

##################################
#[3] note: this is not efficient
##################################
order = np.argsort(dist) # sort 
A = sparse.lil_matrix((nv, nv))

connected=False
nextedge=0
while not connected:
    i=I[order[nextedge]]
    j=J[order[nextedge]]
    d=dist[order[nextedge]]
    if np.mod(nextedge,100)==0:
        print "edge %d of %d"%(nextedge,len(order))
    if not A[i,j]:
        A[i,j] = d
        A[j,i] = d
        
        # check connectivity
        minedges = A.nnz/2.0 > nv-1
        everyrow = A.sum(axis=1).min()>0.0
        if minedges and everyrow:
            cstree = csg.cs_graph_components(A)
            print "   ...checking connectivity (%d components)"%cstree[0]
            if cstree[0]==1 and len(np.where(cstree[1]==-2)[0])==0:
                connected=True
    nextedge+=1

##################################
#[4]
##################################
randne=2000
randI = np.random.randint(nv,size=randne)
randJ = np.random.randint(nv,size=randne)

while len(np.where(randI==randJ)[0])>0:
    loc = np.where(randI==randJ)[0]
    print "fixing %d random edges..."%len(loc)
    randJ[loc]=np.random.randint(nv,size=len(loc))
    randI[loc]=np.random.randint(nv,size=len(loc))

for nextedge in range(randne):
    i = randI[nextedge]
    j = randJ[nextedge]
    d = np.sqrt((xy[i,0]-xy[j,0])**2 + np.sqrt((xy[i,0]-xy[j,0])**2))
    A[i,j]=d
    A[j,i]=d

A = A.tocsr()
B = sparse.triu(A.tocoo())

edges = np.vstack((B.row,B.col)).T

##################################
#[5]
##################################
refnv = 10000
refne = 58913
refdeg = 11.7826

C = A.copy()
C.data = np.ones(C.data.shape)
print "The target number of nodes is %d"%refnv
print "The target number of edges is %d"%refne
print "The target average degree is %g"%refdeg
print "The number of nodes is %d"%nv
print "The number of edges is %d"%(A.nnz/2.0)
print "The average degree is %g"%C.sum(axis=1).mean()

from scipy.io import savemat
savemat('random_disk_graph',{'V':xy, 'E':edges})
