
from matplotlib import rcParams, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
rcParams['figure.figsize'] = (16,10)     #Affects output size of graphs.
'''
Code up your X, Y, Z values
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot( X, Y, Z )    #Make sure X, Y, Z are same length.
                      #Connect points (X[i], Y[i], Z[i]) for i in len(X)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim3d([min(X), max(X)])    #Bounds the axes nicely
ax.set_ylim3d([min(Y), max(Y)])
ax.set_zlim3d([min(Z), max(Z)])

plt.show()

plt.switch_backend('qt5agg') # This backend opens the graph in a new window

import numpy as np
from scipy.integrate import odeint

def lorenz_ode(inputs, T):
	'''
	Code up the sytem of equations given
	'''
	return Xprime, Yprime, Zprime

def solve_lorenx(init_cond, time=10):
	T = np.linspace(0, time, time*100)	#initialize time interval for ode
	'''
	Use odeint in conjuction with lorenz_ode and the time interval T
	To get the X, Y, and Z values for this system.
	You will need to transpose the output of odeint to graph it correctly.
	'''
 	return X, Y, Z

sigma = 'value'
rho = 'value'
beta = 'value'
init_cond = [x0, y0, z0]

X, Y, Z = solve_lorenz(init_cond, 50)
'''
Code to graph
'''

from matplotlib.animation import FuncAnimation

from matplotlib.animation import FuncAnimation

def sine_cos_animation():
	#Calculate the data to be animated
	x = np.linspace(0, 2*np.pi, 200)[:-1]
	y1, y2 = np.sin(x), np.cos(x)
	
	#Create a figure and set the window boundaries
	fig = plt.figure()
	plt.xlim(0, 2*np.pi)
	plt.ylim(-1.2, 1.2)
	
	#Initiate empty lines of the correct dimension
	sin_drawing, = plt.plot([], [])
	cos_drawing, = plt.plot([], [])	#note the comma after the variable name
	
	#Define a function that updates each line
	def update(index):
		sin_drawing.set_data(x[:index], y1[:index])
		cos_drawing.set_data(x[:index], y2[:index])
		return sin_drawing, cos_drawing,
	
	a = FuncAnimation(fig, update, frames=len(x), interval=10)
	plt.show()

from scipy import linalg as la
from scipy.stats import linregress
