from scipy import linalg as la

# The standard Schur decomposition.
A = np.array([[0,0,2],[-3,2,6],[0,0,1]])
T,Z = la.schur(A)
T                       # The eigenvalues (2, 0, and 1) are not sorted.

# Specify a sorting function to get the desired result.
f = lambda x: abs(x) > 0
T1,Z1,k = la.schur(A, sort=f)
T1
k                       # k is the number of columns satisfying the sort,
                           # which is the number of nonzero eigenvalues.

A = np.random.randint(-9,9,(3,3))
A

# Find the minimum value in the array.
minval = np.<<min>>(A)
minval

# Find the location of the minimum value.
loc = np.where(A==minval)
loc
