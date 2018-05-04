
# Dependencies for this lab's code:
from __future__ import division
from math import sqrt, pi
import numpy as np					
from scipy.fftpack import fft, ifft		
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# Array of wave numbers.  This array is reordered in Python to 
# accomodate the ordering inside the fft function in scipy.
k = np.concatenate(( np.arange(0,N/2) ,
					 np.array([0])	,
					 np.arange(-N/2+1,0,1)	)).reshape(N,)

# Defines the left hand side of the ODE y' = G(t,y)
# defined above.
ik3 = 1j*k**3.
def G_unscaled(t,y):
	out = -.5*1j*k*fft(ifft(y,axis=0)**2.,axis=0)  + ik3*y        
	return out

N = 256
x = (2.*np.pi/N)*np.arange(-N/2,N/2).reshape(N,1)   # Space discretization
s, shift = 25.**2., 2.  							# Initial data is a soliton
y0 = (3.*s*np.cosh(.5*(sqrt(s)*(x+shift)))**(-2.)).reshape(N,) 

# Solves the ODE.
max_t = 
dt = # constant*N**(-2.)
max_tsteps = int(round(max_t/dt))
y0 = fft(y0,axis=0)
T,Y = RK4(G_unscaled, y0, t0=0, t1=max_t, n=max_tsteps)

# Using the variable stride, we step through the data, 
# applying the inverse fourier transform to obtain u.
# These values will be plotted.
stride = int(np.floor((max_t/25.)/dt))
uvalues, tvalues = np.real(ifft(y0,axis=0)).reshape(N,1), np.array(0.).reshape(1,1)
for n in range(1,max_tsteps+1):
	if np.mod(n,stride) == 0:
		t = n*dt
		u = np.real( ifft(Y[n], axis=0) ).reshape(N,1)
		uvalues = np.concatenate((uvalues,np.nan_to_num(u)),axis=1)
		tvalues = np.concatenate((tvalues,np.array(t).reshape(1,1)),axis=1)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=45., azim=150)
tv, xv = np.meshgrid(tvalues,x,indexing='ij')
surf = ax.plot_surface(tv,xv, uvalues.T, rstride=1, cstride=1, cmap=cm.coolwarm,
						linewidth=0, antialiased=False)
tvalues = tvalues[0]; ax.set_xlim(tvalues[0], tvalues[-1])
ax.set_ylim(-pi, pi); ax.invert_yaxis()
ax.set_zlim(0., 4000.)
ax.set_xlabel('T'); ax.set_ylabel('X'); ax.set_zlabel('Z')
plt.show()

def initialize_all(y0, t0, t1, n):
	""" An initialization routine for the different ODE solving
	methods in the lab. This initializes Y, T, and h. """
	
	if isinstance(y0, np.ndarray):
		Y = np.empty((n, y0.size),dtype=complex).squeeze()
	else:
		Y = np.empty(n,dtype=complex)
	Y[0] = y0
	T = np.linspace(t0, t1, n)
	h = float(t1 - t0) / (n - 1)
	return Y, T, h


def RK4(f, y0, t0, t1, n):
	""" Use the RK4 method to compute an approximate solution
	to the ODE y' = f(t, y) at n equispaced parameter values from t0 to t
	with initial conditions y(t0) = y0.
	
	'y0' is assumed to be either a constant or a one-dimensional numpy array.
	't0' and 't1' are assumed to be constants.
	'f' is assumed to accept two arguments.
	The first is a constant giving the current value of t.
	The second is a one-dimensional numpy array of the same size as y.
	
	This function returns an array Y of shape (n,) if
	y is a constant or an array of size 1.
	It returns an array of shape (n, y.size) otherwise.
	In either case, Y[i] is the approximate value of y at
	the i'th value of np.linspace(t0, t, n).
	"""
	Y, T, h = initialize_all(y0, t0, t1, n)
	for i in xrange(1, n):
		K1 = f(T[i-1], Y[i-1])
		tplus = (T[i] + T[i-1]) * .5
		K2 = f(tplus, Y[i-1] + .5 * h * K1)
		K3 = f(tplus, Y[i-1] + .5 * h * K2)
		K4 = f(T[i], Y[i-1] + h * K3)
		Y[i] = Y[i-1] + (h / 6.) * (K1 + 2 * K2 + 2 * K3 + K4)
	return T, Y

% 
% 
% 
% 
% 
% 
% import numpy as np
% from scipy.fftpack import fft, ifft
% import matplotlib.pyplot as plt
% 
% N=24
% x1 = (2.*np.pi/N)*np.arange(1,N+1)
% f = np.sin(x1)**2.*np.cos(x1) + np.exp(2.*np.sin(x1+1))
% 
% 
% k = np.concatenate(( np.arange(0,N/2) ,
% 					 np.array([0])	, # Because hat{f}'(k) at k = N/2 is zero.
% 					 np.arange(-N/2+1,0,1)	))
% 
% # Approximates the derivative using the pseudospectral method
% f_hat = fft(f)
% fp_hat = ((1j*k)*f_hat)
% fp = np.real(ifft(fp_hat))
% 
% # Calculates the derivative analytically
% x2 = np.linspace(0,2*np.pi,200)
% derivative = (2.*np.sin(x2)*np.cos(x2)**2. - 
% 				np.sin(x2)**3. + 
% 				2*np.cos(x2+1)*np.exp(2*np.sin(x2+1))
% 				)
% 
% plt.plot(x2,derivative,'-k',linewidth=2.)
% plt.plot(x1,fp,'*b')
% plt.savefig('spectral2_derivative.pdf')
% plt.show()
% 
% 
% import numpy as np
% import matplotlib.pyplot as plt
% 
% # Solve the ODE $u_{xx} = e^u,$ with boundary conditions
% #  $u(0) = u(2\pi) = 0$
% 
% 
% 