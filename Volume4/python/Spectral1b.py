
import numpy as np

def cheb(N):
	x =  np.cos((np.pi/N)*np.linspace(0,N,N+1))
	x.shape = (N+1,1)
	lin = np.linspace(0,N,N+1)
	lin.shape = (N+1,1)
	
	c = np.ones((N+1,1))
	c[0], c[-1] = 2., 2.
	c = c*(-1.)**lin
	X = x*np.ones(N+1) # broadcast along 2nd dimension (columns)
	
	dX = X - X.T
	
	D = (c*(1./c).T)/(dX + np.eye(N+1))
	D  = D - np.diag(np.sum(D.T,axis=0))
	x.shape = (N+1,)
	# Here we return the differentiation matrix and the Chebyshev points,
	# numbered from x_0 = 1 to x_N = -1
	return D, x


#The following code will force U[0] = U[N] = 0
D, x = cheb(N)    #for some N
D2 = np.dot(D, D)
D2[0,:], D2[-1,:] = 0, 0
D2[0,0], D2[-1,-1] = 1, 1
F[0], F[-1] = 0, 0

from scipy.optimize import root

N = 20
D, x = cheb(20)

def F(U):
	out = None	#Set up the equation you want the root of.
	#Make sure to set the boundaries correctly
	
	return out	#Newtons Method will update U until the output is all 0's.

guess = None    #Make your guess, same size as the cheb(N) output
solution = root(F, guess).x 

from mpl_toolkits.mplot3d import Axes3D

barycentric = None	#This is the output of barycentric_interpolate() on 100 points

lin = np.linspace(-1, 1, 100)
theta = np.linspace(0,2*np.pi,401)
X, T = np.meshgrid(lin, theta)
Y, Z = barycentric*np.cos(T), barycentric*np.sin(T)

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
plt.show()
