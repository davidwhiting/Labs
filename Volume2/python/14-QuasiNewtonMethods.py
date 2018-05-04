
>>> import scipy.optimize as opt
>>> f = opt.rosen
>>> df = opt.rosen_der
>>> d2f = opt.rosen_hess
>>> minx = opt.fmin_bfgs(f=f,x0=np.array([-2,2]),fprime=df,maxiter=1000,avextol=10**-2)

>>> import scipy.optimize as opt
>>> f = lambda x : np.exp(x[0]-1) + np.exp(1 - x[1]) + (x[0] - x[1])**2
>>> df = lambda x : np.array([np.exp(x[0]-1) + 2*(x[0]-x[1]), -1*np.exp(1-x[1]) - 2*(x[0]-x[1])])
>>> minx = opt.fmin_bfgs(f=f,fprime=df,x0=[2,3],gtol=10**-2,maxiter=1000)

>>> t = np.arange(10)
>>> y = 3*np.sin(0.5*t)+ 0.5*np.random.randn(10)

>>> def model(x, t):
>>>     return x[0]*np.sin(x[1]*t)
>>> def residual(x):
>>>     return model(x, t) - y
>>> def jac(x):
>>>     ans = np.empty((10,2))
>>>     ans[:,0] = np.sin(x[1]*t)
>>>     ans[:,1] = x[0]*t*np.cos(x[1]*t)
>>>     return ans

>>> x0 = np.array([2.5,.6])
>>> x,niters,conv = gaussNewton(jac, residual, x0, maxiter=10, tol=10**-3)

dom = np.linspace(0,10,100)
plt.plot(t, y, '*')
plt.plot(dom, 3*np.sin(.5*dom), '--')
plt.plot(dom, x[0]*np.sin(x[1]*dom))
plt.show()

>>> import scipy.optimize as opt
>>> r = lambda x:
>>> J = lambda x:
>>> minx = opt.leastsq(fun=r, x0=, Dfun=J,xtol=,maxfev=maxiter)

>>> import numpy as np
>>> pop_sample1 = np.load('pop_sample1.npy')
>>> print(pop_sample1)
<<[[  0.      1.      2.      3.      4.      5.      6.      7.   ]>>
<<[  3.929   5.308   7.24    9.638  12.866  17.069  23.192  31.443]]>>

>>> import scipy.optimize as opt
>>> f = lambda x: np.array([])
>>> df = lambda x: np.array([np.exp(x[0] - 1) + 2*(x[0] - x[1]), np.exp(1-x
    ...: [1]) - 2*(x[0]-x[1])])
>>> A0 = np.array([])
>>> minx = opt.broyden1(F=df, xin=[2,3], x_tol=10**2, maxiter=1000)

>>> def model(x, t):
>>>     return x[0]*np.sin(x[1]*t)
>>> def residual(x):
>>>     return model(x, t) - y
>>> def jac(x):
>>>     ans = np.empty((10,2))
>>>     ans[:,0] = np.sin(x[1]*t)
>>>     ans[:,1] = x[0]*t*np.cos(x[1]*t)
>>>     return ans
>>> def objective(x):
>>>     return .5*(residual(x)**2).sum()
>>> def grad(x):
>>>     return jac(x).T.dot(residual(x))

>>> x0 = np.array([2.5,.6])
>>> x = gaussNewton(jac, residual, x0, niter=10)

dom = np.linspace(0,10,100)
plt.plot(t, y, '*')
plt.plot(dom, 3*np.sin(.5*dom), '--')
plt.plot(dom, x[0]*np.sin(x[1]*dom))
plt.show()

>>> from scipy.optimize import leastsq
>>> x2 = leastsq(residual, x0)[0]
