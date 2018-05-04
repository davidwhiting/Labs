
>>> from cvxopt import matrix
>>> c = matrix([-4., -5.])
>>> G = matrix([[1., 2., -1., 0.],[2., 1., 0., -1.]])
>>> h = matrix([ 3., 3., 0., 0.])

>>> import numpy as np
>>> c = np.array([-4., -5.])
>>> G = np.array([[1., 2.],[2., 1.],[-1., 0.],[0., -1]])
>>> h = np.array([3., 3., 0., 0.])

>>> #Now convert to CVXOPT matrix type
>>> c = matrix(c)
>>> G = matrix(G)
>>> h = matrix(h)

>>> from cvxopt import solvers
>>> sol = solvers.lp(c, G, h)
     pcost       dcost       gap    pres   dres   k/t
 0: -8.1000e+00 -1.8300e+01  4e+00  0e+00  8e-01  1e+00
 1: -8.8055e+00 -9.4357e+00  2e-01  1e-16  4e-02  3e-02
 2: -8.9981e+00 -9.0049e+00  2e-03  1e-16  5e-04  4e-04
 3: -9.0000e+00 -9.0000e+00  2e-05  1e-16  5e-06  4e-06
 4: -9.0000e+00 -9.0000e+00  2e-07  1e-16  5e-08  4e-08
Optimal solution found.
>>> print sol['x']
[ 1.00e+00]
[ 1.00e+00]
>>> print sol['primal objective']
-8.99999981141
>>> print type(sol['x'])
<type 'cvxopt.base.matrix'>

solvers.options['show_progress'] = False

>>> c = matrix([4., 7., 6., 8., 8., 9])
>>> G = matrix(-1*np.eye(6))
>>> h = matrix(np.zeros(6))
>>> A = matrix(np.array([[1.,1.,0.,0.,0.,0.],
                         [0.,0.,1.,1.,0.,0.],
                         [0.,0.,0.,0.,1.,1.],
                         [1.,0.,1.,0.,1.,0.],
                         [0.,1.,0.,1.,0.,1.]]))
>>> b = matrix([7., 2., 4., 5., 8.])
>>> sol = solvers.lp(c, G, h, A, b)
     pcost       dcost       gap    pres   dres   k/t
 0:  8.9500e+01  8.9500e+01  2e+01  4e-17  2e-01  1e+00
Terminated (singular KKT matrix).
>>> print sol['x']
[ 3.00e+00]
[ 4.00e+00]
[ 5.00e-01]
[ 1.50e+00]
[ 1.50e+00]
[ 2.50e+00]
>>> print sol['primal objective']
89.5

>>> from cvxopt import matrix, solvers
>>> G = matrix([ [-1., 0., 0., -1., 0.,  -1., 0., 0., 0., 0., 0.],
             [-1., 0., 0., 0., -1.,  0., -1., 0., 0., 0., 0.],
             [0., -1., 0., -1., 0.,  0., 0., -1., 0., 0., 0.],
             [0., -1., 0., 0., -1.,  0., 0., 0., -1., 0., 0.],
             [0., 0., -1., -1., 0.,  0., 0., 0., 0., -1., 0.],
             [0., 0., -1., 0., -1.,  0., 0., 0., 0., 0., -1.] ])

>>> h = matrix([-7., -2., -4., -5., -8.,  0., 0., 0., 0., 0., 0.,])
>>> c = matrix([4., 7., 6., 8., 8., 9])
>>> sol = solvers.lp(c,G,h)
>>> print sol['x']
>>> print sol['primal objective']

>>> from cvxopt import matrix, solvers, glpk
>>> G = matrix([ [-1., 0., 0., -1., 0.,  -1., 0., 0., 0., 0., 0.],
             [-1., 0., 0., 0., -1.,  0., -1., 0., 0., 0., 0.],
             [0., -1., 0., -1., 0.,  0., 0., -1., 0., 0., 0.],
             [0., -1., 0., 0., -1.,  0., 0., 0., -1., 0., 0.],
             [0., 0., -1., -1., 0.,  0., 0., 0., 0., -1., 0.],
             [0., 0., -1., 0., -1.,  0., 0., 0., 0., 0., -1.] ])

>>> h = matrix([-7., -2., -4., -5., -8.,  0., 0., 0., 0., 0., 0.,])
>>> o = matrix([4., 7., 6., 8., 8., 9])
>>> sol = glpk.ilp(o,G,h)
>>> print sol[1]

>>> from cvxopt import matrix, solvers, glpk
>>> G = matrix([ [-1., 0., 0., 0., 0., 0.],
             [0., -1., 0., 0., 0., 0.],
             [0., 0., -1., 0., 0., 0.],
             [0., 0., 0., -1., 0., 0.],
             [0., 0., 0., 0., -1., 0.],
             [0., 0., 0., 0., 0., -1.] ])

>>> h = matrix([ 0., 0., 0., 0., 0., 0.,])
>>> o = matrix([4., 7., 6., 8., 8., 9])
>>> A = matrix([ [1., 0., 0., 1., 0.],
             [1., 0., 0., 0., 1.],
             [0., 1., 0., 1., 0.],
             [0., 1., 0., 0., 1.],
             [0., 0., 1., 1., 0.],
             [0., 0., 1., 0., 1.] ])
>>> b = matrix([7., 2., 4., 5., 8])
>>> sol = glpk.ilp(o,G,h,A,b)
>>> print sol[1]

>>> P = matrix(np.array([[4., 2.], [2., 2.]]))
>>> q = matrix([1., -1.])
>>> sol=solvers.qp(P, q)
>>> print(sol['x'])
[-1.00e+00]
[ 1.50e+00]
>>> print sol['primal objective']
-1.25
