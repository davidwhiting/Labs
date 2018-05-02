import numpy as np

D = np.array([[2,0],[0,16]])    # Let D be a diagonal matrix.
d = np.diag(D)                  # Extract the diagonal as a 1-D array.
x = np.random.random(2)
np.allclose(D.dot(x), d*x)

\begin{lstlisting}
from scipy import linalg as la

x = np.random.random(10)
la.norm(x, <<ord>>=np.inf)          # Use la.norm() for ||x||.
np.<<max>>(np.<<abs>>(x))               # The equivalent in NumPy for ||x||.

def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant (n, n) matrix.
    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.
    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in range(n):
        A[i,i] = np.<<sum>>(np.<<abs>>(A[i])) + 1
    return A

x0 = np.random.random(5)        # Generate a random vector.
x1 = x0                         # Attempt to make a copy.
x1[3] = 1000                    # Modify the "copy" in place.
np.allclose(x0, x1)             # But x0 was also changed!

# Instead, make a copy of x0 when creating x1.
x0 = np.copy(x1)                # Make a copy.
x1[3] = -1000
np.allclose(x0, x1)

# Get the indices of where the i-th row of A starts and ends if the
# nonzero entries of A were flattened.
rowstart = A.indptr[i]
rowend = A.indptr[i+1]

# Multiply only the nonzero elements of the i-th row of A with the
# corresponding elements of x.
Aix = A.data[rowstart:rowend] @ x[A.indices[rowstart:rowend]]


from scipy import sparse

A = sparse.csr_matrix(diag_dom(50000))
b = np.random.random(50000)

from matplotlib import pyplot as plt
import numpy as np

n = 100
A,b = finite_difference(n)
x = sparse_sor(A,b,1.9,maxiters=10000,tol=10**-2)
U = x.reshape((n,n))
x,y = np.linspace(0,10,n), np.linspace(0,10,n)
X,Y = np.meshgrid(x,y)

plt.pcolormesh(X,Y,U,cmap='coolwarm')
plt.show()

from matplotlib import pyplot as plt

def jacobi(n=100, tol=1e-8):

    # Perform the algorithm, storing the result in the array 'U'.

    # Visualize the results.
    plt.imshow(U)
    plt.show()

