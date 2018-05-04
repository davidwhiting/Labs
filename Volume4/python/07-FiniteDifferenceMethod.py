
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def bvp(func, epsilon, alpha, beta, N):
	a,b = 0., 1. 	# Interval for the BVP
	h = (b-a)/N 	# The length of each subinterval
	
	# Initialize and define the vector F on the right
	F = np.empty(N-1.)			
	F[0] = func(a+1.*h)-alpha*(epsilon+h/2.)*h**(-2.)
	F[N-2] = func(a+(N-1)*h)-beta*(epsilon-h/2.)*h**(-2.)
	for j in xrange(1,N-2):
		F[j] = func(a + (j+1)*h)
		
	# Here we define the arrays that will go on the diagonals of A
	data = np.empty((3,N-1))
	data[0,:] = -2.*epsilon*np.ones((1,N-1)) # main diagonal
	data[1,:]  = (epsilon+h/2.)*np.ones((1,N-1))  	 # off-diagonals
	data[2,:] = (epsilon-h/2.)*np.ones((1,N-1))
	# Next we specify on which diagonals they will be placed, and create A
	diags = np.array([0,-1,1])
	A=h**(-2.)*spdiags(data,diags,N-1,N-1).asformat('csr')
	
	U = np.empty(N+1)
	U[1:-1] = spsolve(A,F)
	U[0], U[-1] = alpha, beta
	return np.linspace(a,b,N+1), U

x, y = bvp(lambda x:-1., epsilon=.05,alpha=1, beta=3, N=400)
plt.plot(x,y,'-k',linewidth=2.0)
plt.show()


num_approx = 10 # Number of Approximations
N = 5*np.array([2**j for j in range(num_approx)])
h, max_error = (1.-0)/N[:-1], np.ones(num_approx-1)

# Best numerical solution, used to approximate the true solution.
# bvp returns the grid, and the grid function, approximating the solution
# with N subintervals of equal length.
num_sol_best = bvp(lambda x:-1, epsilon=.1, alpha=1, beta=3, N=N[-1])
for j in range(len(N)-1):
    num_sol = bvp(lambda x:-1, epsilon=.1, alpha=1, beta=3, N=N[j])
    max_error[j] = np.max(np.abs( num_sol- num_sol_best[::2**(num_approx-j-1)] ) )
plt.loglog(h,max_error,'.-r',label="$E(h)$")
plt.loglog(h,h**(2.),'-k',label="$h^{\, 2}$")
plt.xlabel("$h$")
plt.legend(loc='best')
plt.show()
print "The order of the finite difference approximation is about ", ( (np.log(max_error[0]) -
    np.log(max_error[-1]) )/( np.log(h[0]) - np.log(h[-1]) ) ), "."
