
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def f(x):
	out = -np.ones(x.shape)
	m = np.where(x<.5)
	out[m] = -6*x[m]**2. + 3.*x[m] - 1.
	return out

def u(x):
	return (x+1./4)**2. + 1./4

def integral_of_f(x):
	# out =  \int_0^x f(s) ds
	return out

def derivative_of_u(x):
	# out = u'(x)
	return out

x = np.linspace(0,1,11)
F, u_p = integral_of_f(x), derivative_of_u(x)

def sum_of_squares(alpha):
	pass

guess = (1./4)*(3-x)
sol = minimize(sum_of_squares,guess)

plt.plot(x,sol.x,'-ob',linewidth=2)
plt.show()

% 