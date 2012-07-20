from matplotlib.collections import LineCollection
import pylab as pl

def plotgraph(xy,edges):
    lcol = xy[edges]
    pl.figure()
    pl.hold(True)
    lc = LineCollection(xy[edges])
    pl.gca().add_collection(lc)
    pl.plot(xy[:,0], xy[:,1], 'ro')
    pl.show()
