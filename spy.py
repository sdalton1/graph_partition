#from numpy import *
#from matplotlib.pyplot import *
import numpy
import matplotlib.pyplot as plt

def plotcolors(cindex) :
	cmap = numpy.zeros((16,3))
	cmap[0 ,:] = [    1,0.125, 0.09]
	cmap[1 ,:] = [    0,0.976,0.298]
	cmap[2 ,:] = [ 0.11,0.224,0.965]
	cmap[3 ,:] = [    1,0.263,0.969]
	cmap[4 ,:] = [    0,0.992,0.996]
	cmap[5 ,:] = [    0,    0,    0]
	cmap[6 ,:] = [    1,0.718,0.231]
	cmap[7 ,:] = [0.663,0.663,0.663]
	cmap[8 ,:] = [0.792, 0.78,0.243]
	cmap[9 ,:] = [0.596,0.051,0.031]
	cmap[10,:] = [    0,  0.4,    0]
	cmap[11,:] = [    0,    0,  0.4]
	cmap[12,:] = [ 0.59,    1, 0.83]
	cmap[13,:] = [    1,    1,    0]
	cmap[14,:] = [  0.5,    0,  0.5]
	cmap[15,:] = [    0,  0.5,  0.5]

	num = len(cmap[:,1])

	return cmap[numpy.mod(cindex,num), :]

def plotsymbols(cindex) :
	symlist[0] = 's'
	symlist[1] = '^'
	symlist[2] = 'o'
	symlist[3] = 'd'
	symlist[4] = '.'
	symlist[5] = 'x'
	symlist[6] = 'h'
	symlist[7] = 'p'
	symlist[8] = 'v'

	num = len(symlist[:,1])

	return symlist[numpy.mod(cindex-1,num), :]

def Spy(A, P1, symb, ttl, ax, parts=2, printperm=0) :

	m,n = A.shape

	colors = numpy.zeros((parts+1,3))
	
	for cnt in numpy.arange(parts+1) :
		colors[cnt,:] = plotcolors(cnt)

	part_ids = numpy.zeros(m, dtype=numpy.int32)
	part_ids[P1] = 1
	perm = part_ids.argsort()
	A_perm = A[perm,:][:,perm]
	A_perm = A_perm.tocoo()
	part_ids = part_ids[perm]

	nnz = A.nnz
	xlab = 'nnz = %d\n' % nnz
	for i in range(nnz) :
	  if part_ids[A_perm.row[i]] == part_ids[A_perm.col[i]] :
	    A_perm.data[i] = part_ids[A_perm.col[i]]
	  else :
	    A_perm.data[i] = parts

	plt.hold(True)
	
	plt.title(ttl, fontsize=14)
	bbox = plt.get(plt.gca(), 'position')
	pos = [bbox.x1, bbox.y1]
	markersize = max(6,min(14,round(6*min(pos[:])/max(m+1,n+1))));
	sy = symb

	for cnt in numpy.arange(parts+1) :
		ii, = numpy.where(A_perm.data==cnt)
		#print 'part %d has %d entries'%(cnt, len(ii))
		c = colors[cnt,:]
		plt.plot(A_perm.col[ii], A_perm.row[ii], marker=sy, markersize=markersize, color=c, linestyle='.')

	plt.axis('tight')
	plt.xlim(xmin=-2)
	plt.ylim(ymin=-2)
	plt.xlim(xmax=m+2)
	plt.ylim(ymax=n+2)
	ax.set_ylim(ax.get_ylim()[::-1])
	ax.set_aspect((m+1)/(n+1),adjustable='box')
	plt.hold(False)
	plt.show()

