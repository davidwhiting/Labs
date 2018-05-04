
import numpy as np

a, b = -1, 1.
alpha, beta = 1., 7.
####  Define variables x_steps, final_T, time_steps  ####
delta_t, delta_x = final_T/time_steps, (b-a)/x_steps
x0 = np.linspace(a,b,x_steps+1)

# Check a stability condition for this numerical method
if delta_t/delta_x**2. > .5:
	print "stability condition fails"
	
u = np.empty((2,x_steps+1))
u[0]  = (beta - alpha)/(b-a)*(x0-a)  + alpha
u[1] = (beta - alpha)/(b-a)*(x0-a)  + alpha

def rhs(y):
	# Approximate first and second derivatives to second order accuracy.
	yp = (np.roll(y,-1) - np.roll(y,1))/(2.*delta_x)
	ypp = (np.roll(y,-1) - 2.*y + np.roll(y,1))/delta_x**2.
	# Find approximation for the next time step, using a first order Euler step
	y[1:-1] -= delta_t*(1. + yp[1:-1]**2. - 1.*y[1:-1]*ypp[1:-1])


# Time step until successive iterations are close
iteration = 0
while iteration < time_steps:
	rhs(u[1])
	if norm(np.abs((u[0] - u[1]))) < 1e-5: break
	u[0] = u[1]
	iteration+=1

print "Difference in iterations is ", norm(np.abs((u[0] - u[1])))
print "Final time = ", iteration*delta_t

from numpy.random import random_integers, uniform, randn
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.misc import imread, imsave

imagename = 'baloons_resized_bw.jpg'
changed_pixels=40000
# Read the image file imagename into an array of numbers, IM
# Multiply by 1. / 255 to change the values so that they are floating point
# numbers ranging from 0 to 1.
IM = imread(imagename, flatten=True) * (1. / 255)
IM_x, IM_y = IM.shape
	
for lost in xrange(changed_pixels):
	x_,y_ = random_integers(1,IM_x-2), random_integers(1,IM_y-2)
	val =  .1*randn() + .5
	IM[x_,y_] = max( min(val,1.), 0.)
imsave(name=("noised_"+imagename),arr=IM)	

u_xx = np.roll(u,-1,axis=1) - 2*u + np.roll(u,1,axis=1)	

% delta_t = 1e-3
% lmbda = 40
% u = np.empty((2,IM_x,IM_y))
% u[1] = IM
%
% def laplace(z):
% 	# Approximate first and second derivatives to second order accuracy.
% 	z_xx = (np.roll(z,-1,axis=0) - 2.*z + np.roll(z,1,axis=0))#/delta_x**2.
% 	z_yy = (np.roll(z,-1,axis=1) - 2.*z + np.roll(z,1,axis=1))#/delta_y**2.
% 	# Find approximation for the next time step, using a first order Euler step
% 	z[1:-1,1:-1] -= delta_t*(   (z[1:-1,1:-1]-IM[1:-1,1:-1])
% 									-lmbda*(z_xx[1:-1,1:-1] + z_yy[1:-1,1:-1]))
%
% # Iterate towards a steady state solution of the gradient descent flow.
% iteration = 0
% while iteration < time_steps:
% 	laplace(u[1])
% 	if norm(np.abs((u[0] - u[1]))) < 1e-4: break
% 	u[0] = u[1]
% 	iteration+=1
%
% 
u_x = (np.roll(u,-1,axis=1) -  np.roll(u,1,axis=1))/2	
u_xx = np.roll(u,-1,axis=1) - 2*u + np.roll(u,1,axis=1)	
u_xy = (np.roll(u_x,-1,axis=0) - np.roll(u_x,1,axis=0))/2.
