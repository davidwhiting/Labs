
>>> from scipy import optimize as opt
>>> import numpy as np

>>> f = lambda x : np.exp(x) - 4*x
>>> result = opt.golden(f, brack=(0,3), tol=.001)

>>> df = lambda x : 2*x + 5*np.cos(5*x)
>>> d2f = lambda x : 2 - 25*np.sin(5*x)
>>> result = opt.newton(df, x0 = 0, fprime = d2f, tol = 1e-10, maxiter = 500)
# If fprime is not provided, the Secant method will be used.

# Recall that autograd takes the derivatives of functions.
>>> from autograd import grad

>>> f = lambda x: x[0]**2 + x[1]**2 + x[2]**2
>>> x = np.array([150., .03, 40.])
>>> p = np.array([-.5, -100., -4.5])
>>> phi = lambda alpha: f(x + alpha*p)
>>> derphi = grad(phi)
>>> alpha, _ = opt.linesearch.scalar_search_armijo(phi, phi(0.), derphi(0.))

