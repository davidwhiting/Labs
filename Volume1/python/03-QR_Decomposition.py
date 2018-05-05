import numpy as np
from scipy import linalg as la

# Generate a random matrix and get its reduced QR decomposition via SciPy.
A = np.random.random((6,4))
Q,R = la.qr(A, mode="economic") # Use mode="economic" for reduced QR.
print(A.shape, Q.shape, R.shape)

# Verify that R is upper triangular, Q is orthonormal, and QR = A.
np.allclose(np.triu(R), R)
np.allclose(Q.T @ Q, np.identity(4))
np.allclose(Q @ R, A)

sign = lambda x: 1 if x >= 0 else -1

A = np.random.random((5, 3))
Q,R = la.qr(A)                  # Get the full QR decomposition.
print(A.shape, Q.shape, R.shape)
np.allclose(Q @ R, A)

# Generate a random matrix and get its upper Hessenberg form via SciPy.
A = np.random.random((8,8))
H, Q = la.hessenberg(A, calc_q=True)

# Verify that H has all zeros below the first subdiagonal and QHQ^T = A.
np.allclose(np.triu(H, -1), H)
np.allclose(Q @ H @ Q.T, A)

A = np.reshape(np.arange(4) + 1j*np.arange(4), (2,2))
print(A)
print(A.T)                      # Regular transpose.
print(A.conj().T)               # Hermitian conjugate.

x = np.arange(2) + 1j*np.arange(2)
print(x)
np.dot(x, x)                    # Standard real inner product.
np.dot(x.conj(), y)             # Standard complex inner product.

sign = lambda x: 1 if np.real(x) >= 0 else -1

# Get the decomposition AP = QR for a random matrix A.
A = np.random.random((8,10))
Q,R,P = la.qr(A, pivoting=True)

# P is returned as a 1-D array that encodes column ordering,
# so A can be reconstructed with fancy indexing.
np.allclose(Q @ R, A[:,P])

# Generate a random orthonormal matrix and a random upper-triangular matrix.
Q, _ = la.qr(np.random.normal(size=(500,500)))
R  = np.triu(np.random.normal(size=(500,500)))

# Calculate A = QR, noting that Q and R are the EXACT QR decomposition of A.
A = Q @ R

# Use SciPy to rediscover the QR decomposition of A.
Q1, R1 = la.qr(A)

# Compare the true Q and R to the computed Q1 and R1.
print(la.norm(Q1-Q, <<ord>>=np.inf) / la.norm(Q, <<ord>>=np.inf))
print(la.norm(R1-R, <<ord>>=np.inf) / la.norm(R, <<ord>>=np.inf))

A1 = Q1 @ R1
la.norm(A1 - A, <<ord>>=np.inf) / la.norm(A, <<ord>>=np.inf)
