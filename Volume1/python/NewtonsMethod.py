lambda x: x / 2
tol, maxiters = 10, 1e-9, 8
k in range(maxiters):           # Iterate at most N times.
print(x0, end='  ')
x1 = F(x0)                      # Compute the next iteration.
if abs(x1 - x0) < tol:          # Check for convergence.
   break                       # Upon convergence, stop iterating.
x0 = x1                         # Otherwise, continue iterating.

import numpy as np
f = lambda x: np.sign(x) * np.power(np.<<abs>>(x), 1./3)

x_real = np.linspace(-1.5, 1.5, 500)    # Real parts.
x_imag = np.linspace(-1.5, 1.5, 500)    # Imaginary parts.
X_real, X_imag = np.meshgrid(x_real, x_imag)
X_0 = X_real + 1j*X_imag                # Combine real and imaginary parts.

f = lambda x: x**3 - 1
Df = lambda x: 3*x**2
X_1 = X_0 - f(X_0)/Df(X_0)
