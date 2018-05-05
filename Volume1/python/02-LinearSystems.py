
import numpy as np

A = np.array([[1, 1, 1, 1],
              [1, 4, 2, 3],
              [4, 7, 8, 9]], dtype=np.<<float>>)

# Reduce the 0th column to zeros below the diagonal.
A[1,0:] -= (A[1,0] / A[0,0]) * A[0]
A[2,0:] -= (A[2,0] / A[0,0]) * A[0]

# Reduce the 1st column to zeros below the diagonal.
A[2,1:] -= (A[2,1] / A[1,1]) * A[1,1:]
print(A)

import scipy as sp
hasattr(sp, "stats")            # The stats module isn't loaded yet.
from scipy import stats         # Import stats explicitly. Access it
hasattr(sp, "stats")            # with 'stats' or 'sp.stats'.

from scipy import linalg as la

# Make a random matrix and a random vector.
A = np.random.random((1000,1000))
b = np.random.random(1000)

# Compute the LU decomposition of A, including pivots.
L, P = la.lu_factor(A)

# Use the LU decomposition to solve Ax = b.
x = la.lu_solve((L,P), b)

# Check that the solution is legitimate.
np.allclose(A @ x, b)

from scipy import sparse

# Define the rows, columns, and values separately.
rows = np.array([0, 1, 0])
cols = np.array([0, 1, 1])
vals = np.array([3, 5, 2])
A = sparse.coo_matrix((vals, (rows,cols)), shape=(3,3))
print(A)

# The toarray() method casts the sparse matrix as a NumPy array.
print(A.toarray())              # Note that this method forfeits all sparsity-related optimizations.

B = sparse.lil_matrix((2,6))
B[0,2] = 4
B[1,3:] = 9

print(B.toarray())

# Use sparse.diags() to create a matrix with diagonal entries.
diagonals = [[1,2],[3,4,5],[6]]     # List the diagonal entries.
offsets = [-1,0,3]                  # Specify the diagonal they go on.
print(sparse.diags(diagonals, offsets, shape=(3,4)).toarray())

# If all of the diagonals have the same entry, specify the entry alone.
A = sparse.diags([1,3,6], offsets, shape=(3,4))
print(A.toarray())

# Modify a diagonal with the setdiag() method.
A.setdiag([4,4,4], 0)
print(A.toarray())

# Use sparse.bmat() to create a block matrix. Use 'None' for zero blocks.
A = sparse.coo_matrix(np.ones((2,2)))
B = sparse.coo_matrix(np.full((2,2), 2.))
print(sparse.bmat([[  A , None,  A  ],
                       [None,  B  , None]], <<format>>='bsr').toarray())

 # Use sparse.block_diag() to construct a block diagonal matrix.
 print(sparse.block_diag((A,B)).toarray())
 
from matplotlib import pyplot as plt

# Construct and show a matrix with 50 2x3 diagonal blocks.
B = sparse.coo_matrix([[1,3,5],[7,9,11]])
A = sparse.block_diag([B]*50)
plt.spy(A, markersize=1)
plt.show()

# Initialize a sparse matrix incrementally as a lil_matrix.
A = sparse.lil_matrix((10000,10000))
for k in range(10000):
    A[np.random.randint(0,9999), np.random.randint(0,9999)] = k

A

# Convert A to CSR and CSC formats to compute the matrix product AA.
Acsr = A.tocsr()
Acsc = A.tocsc()
Acsr.dot(Acsc)

from scipy.sparse import linalg as spla

A = np.zeros(3) + np.vstack(np.arange(3))
P = np.arange(3)
print(A)

# Swap rows 1 and 2.
A[1], A[2] = np.copy(A[2]), np.copy(A[1])
P[1], P[2] = P[2], P[1]
print(A)                        # A with the new row arrangement.
print(P)                        # The permutation of the rows.
print(A[P])                     # A with the original row arrangement.
