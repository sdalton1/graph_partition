from matplotlib.collections import LineCollection
import pylab as pl
import numpy as np

def plotgraph(xy,edges,edgecolor='b'):
    lcol = xy[edges]
    lc = LineCollection(xy[edges])
    lc.set_linewidth(0.1)
    lc.set_color(edgecolor)
    pl.gca().add_collection(lc)
    #pl.plot(xy[:,0], xy[:,1], 'ro')
    pl.xlim(xy[:,0].min(), xy[:,0].max())
    pl.ylim(xy[:,1].min(), xy[:,1].max())
    pl.show()

def plotsplitting(xy,edges,splitting):
    u = np.where(splitting==1)[0]
    v = np.where(splitting==-1)[0]
    edgeflag = splitting[edges].sum(axis=1)
    uedges = np.where(edgeflag==2)[0]
    vedges = np.where(edgeflag==-2)[0]
    wedges = np.where(edgeflag==0)[0]
    pl.figure()
    pl.hold(True)
    plotgraph(xy,edges[uedges,:],edgecolor='b')
    plotgraph(xy,edges[vedges,:],edgecolor='r')
    plotgraph(xy,edges[wedges,:],edgecolor='g')

if __name__=="__main__":
    from scipy.io import loadmat
    d = loadmat('random_disk_graph_1000.mat')
    splitting = loadmat('random_disk_graph_1000_splitting.mat')['splitting'].ravel()
    #plotgraph(d['V'],d['E'])
    plotsplitting(d['V'],d['E'],splitting)
