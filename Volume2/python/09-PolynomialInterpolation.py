
# Evaluate the polynomial (x-2)(x+1) at 10 points without expanding the expression.
>>> pts = np.arange(10)
>>> (pts - 2) * (pts + 1)
array([ 2,  0,  0,  2,  6, 12, 20, 30, 42, 56])

# Given a Numpy array xint of interpolating x-values, calculate the weights.
>>> n = len(xint)                   # Number of interpolating points.
>>> w = np.ones(n)                  # Array for storing barycentric weights.
# Calculate the capacity of the interval.
>>> C = (np.max(xint) - np.min(xint)) / 4
    
>>> shuffle = np.random.permutation(n-1)
>>> for j in range(n):
>>>     temp = (xint[j] - np.delete(xint, j)) / C
>>>     temp = temp[shuffle]        # Randomize order of product.
>>>     w[j] /= np.product(temp)

>>> from scipy.interpolate import BarycentricInterpolator

>>> f = lambda x: 1/(1+25 * x**2)   # Function to be interpolated.
# Obtain the Chebyshev extremal points on [-1,1].
>>> n = 11
>>> pts = np.linspace(-1, 1, n)
>>> domain = np.linspace(-1, 1, 200)

>>> poly = BarycentricInterpolator(pts[:-1])
>>> poly.add_xi(pts[-1])          # Oops, forgot one of the points.
>>> poly.set_yi(f(pts))           # Set the y values.

>>> plt.plot(domain, f(domain))
>>> plt.plot(domain, poly.eval(domain))

from numpy.polynomial.chebyshev import chebval

>>> domain = np.linspace(-1, 1, 5)
>>> f = lambda x: x**4              # Function to interpolate.
>>> coeffs = chebyshv_coeffs(f, 4)  # Function from Problem 6.
>>> print(coeffs)
[  3.75000000e-01  -5.88784672e-17   5.00000000e-01   5.88784672e-17
   1.25000000e-01]

>>> chebval(domain, coeffs)         # Evaluate at the points in domain.
[ 1.      0.0625  0.      0.0625  1.    ]

>>> fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n))
>>> a, b = 0, 366 - 1/24
>>> domain = np.linspace(0, b, 8784)
>>> points = fx(a, b, n)
>>> temp = np.abs(points - domain.reshape(8784, 1))
>>> temp2 = np.argmin(temp, axis=0)

>>> poly = barycentric(domain[temp2], data[temp2])

[9.52973125e-08,  -1.89451973e-09,  -7.42182166e-08,
 1.89319137e-09,   5.26564839e-08,  -1.89451836e-09,
 -3.13050802e-08,   1.89319005e-09,   1.03700608e-08,
 -9.47258778e-10]
