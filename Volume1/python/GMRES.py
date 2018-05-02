A = np.array([[1,0,0],[0,2,0],[0,0,3]])
b = np.array([1, 4, 6])
x0 = np.zeros(b.size)
gmres(A, b, x0, k=100, tol=1e-8)

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

A = np.random.rand(300, 300)
b = np.random(300)
x, info = spla.gmres(A, b)
print(info)

la.norm((A @ x) - b)

# Restart after 1000 iterations.
x, info = spla.gmres(A, b, restart=1000)
info
la.norm((A @ x) - b)
