
% from scipy.integrate import ode
% import numpy as np
% import matplotlib.pyplot as plt
%
% a, ya, b = 0., 2., 1.6
% def ode_f(t,y): return np.array([-1.*y+6.+2.*t])
%
% ode_object = ode(ode_f)
% ode_object.set_integrator('dopri5',atol=1e-5)
% ode_object.set_initial_value(ya,a)
% print ode_object.integrate(b)
% 
% ode_object = ode(ode_f).set_integrator('dopri5',atol=1e-5)
% ode_object.set_initial_value(ya,a)
%
% dim, t = 1, np.linspace(a,b,51)
% Y = np.zeros((len(t),dim))
% Y[0,:] = ya
% for j in range(1,len(t)): Y[j,:] = ode_object.integrate(t[j])
%
% plt.plot(t,Y[:,0],'-k')
% plt.show()
% 
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

epsilon, lbc, rbc = .1, 1, - 1/3

# The ode function takes the independent variable first
# It has return shape (n,)
def ode(x , y):
    return np.array([y[1] , (1/epsilon) * (y[0] - y[0] * y[1])])

# The BVP solver expects you to pass it the boundary
# conditions as a callable function that computes the difference
# between a guess at the boundary conditions
# and the desired boundary conditions.
# When we use the BVP solver, we will tell it how many constraints
# there should be on each side of the domain so that it knows
# how many entries to expect in the tuples BCa and BCb.
# In this case, we have one boundary condition on either side.
# These constraints are expected to evaluate to 0 when the
# boundary condition is satisfied.

# The return shape of bcs() is (n,)
def bcs(ya, yb):
    BCa = np.array([ya[0] - lbc])   # 1 Boundary condition on the left
    BCb = np.array([yb[0] - rbc])   # 1 Boundary condition on the right
    # The return values will be 0s when the boundary conditions are met exactly
    return np.hstack([BCa, BCb])

# The independent variable has size (m,) and goes from a to b with some step size
X = np.linspace(-1, 1, 200)
# The y input must have shape (n,m) and includes our initial guess for the boundaries
y = np.array([-1/3, -4/3]).reshape((-1,1))*np.ones((2, len(X)))

# There are multiple returns from solve_bvp(). We are interested in the y values which can be found in the sol field.
solution = solve_bvp(ode, bcs, X, y)
# We are interested in only y, not y', which is found in the first row of sol.
y_plot = solution.sol(X)[0]

plt.plot(X, y_plot)
plt.show()
