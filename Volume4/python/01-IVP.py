
def initialize_all(a,b,y0,h):
    """Given an initial and final time a and b, with y(a)=y0, and step size h,
    return several things.
    
    X: an aray from a to b with n elements, where n is the number of steps from a to b.
    Y: an empty array of size (n, y.size), Y[0]=y0.
    h: the step size.
    n: the number of steps to be taken.
    
    """
    n = int((b-a)/h+1)
    X = np.linspace(a, b, n)
    if isinstance(y0, np.ndarray):
        Y = np.empty((n, y0.size))
    else:
        Y = np.empty(n)
    Y[0] = y0
    return X, Y, h, int(n)

def euler(f,X,Y,h,n):
    """Use the Euler method to compute an approximate solution
    to the ODE y' = f(t, y) over X.

    Y[0] = y0
    f is assumed to accept two arguments.
    The first is a constant giving the value of t.
    The second is a one-dimensional numpy array of the same size as y.

    This function returns an array Y of shape (n,) if
    y is a constant or an array of size 1.
    It returns an array of shape (n, y.size) otherwise.
    In either case, Y[i] is the approximate value of y at
    the i'th value of X.
    """
    
    return None

import matplotlib.pyplot as plt

a, b, ya = 0., 2., 0.

def ode_f(x,y):
	return np.array([y - 2*x + 4.])
	
best_grid = 320					#  number of subintervals in most refined grid
h = 2./best_grid
X, Y, h, n = initialize_all(a, b, ya, h)
# Requires an implementation of the euler method
best_val = euler(ode_f, X, Y, h, n)[-1]  

smaller_grids = [10, 20, 40, 80]  # number of subintervals in smaller grids
h = [2./N for N in smaller_grids]

Euler_sol = [euler(ode_f, initialize_all(a, b, ya, h[i])[0],
			initialize_all(a, b, ya, h[i])[1], h[i], N+1)[-1]
			for i, N in enumerate(smaller_grids)]
Euler_error = [abs((val - best_val)/best_val) for val in Euler_sol]
	
plt.loglog(h, Euler_error, '-b', label="Euler method", linewidth=2.)
plt.show()

