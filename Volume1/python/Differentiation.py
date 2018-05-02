import sympy as sy

x = sy.symbols('x')
sy.diff(x**3 + x, x)     # Differentiate x^3 + x with respect to x.

from matplotlib import pyplot as plt

ax = plt.gca()
ax.spines["bottom"].set_position("zero")


import numpy as np
np.logspace(-3, 0, 4)           # Get 4 values from 1e-3 to 1e0.

# f accepts a length-2 NumPy array
f = lambda x: np.array([x[0]**2, x[0]+x[1]])

from autograd import numpy as anp       # Use autograd's version of NumPy.
from autograd import grad

g = lambda x: anp.exp(anp.sin(anp.cos(x)))
dg = grad(g)                            # dg() is a callable function.
dg(1.)                                  # Use floats as input, not ints.

f = lambda x: anp.sin(x) + 3**anp.cos(x)

# Calculate the first derivative.
df = grad(f)

# Calculate the second derivative and so forth.
ddf = grad(df)
dddf = grad(ddf)
dddf(1.)

from autograd import elementwise_grad

pts = anp.array([1, 2, 3], dtype=anp.<<float>>)
dg = elementwise_grad(g)        # Calculate g'(x) with array support.
dg(pts)                         # Evaluate g'(x) at each of the points.


from sympy import factorial

def taylor_exp(x, tol=.0001):
    """Compute the Taylor series of e^x with terms greater than tol."""
    result, i, term = 0, 0, x
    while anp.abs(term) > tol:
        term = x**i / int(factorial(i))
        result, i = result + term, i + 1
    return result

d_exp = grad(taylor_exp)
print(d_exp(2., .1), d_exp(2., .0001))

f = lambda x,y: 3*x*y + 2*y - x

# Take the derivative of f with respect to the first variable, x.
dfdx = grad(f, argnum=0)            # Should be dfdx(x,y) = 3y - 1,
dfdx(5., 1.)                        # so dfdx(5,1) = 3 - 1 = 2.

# Take the gradient with respect to the second variable, y.
dfdy = grad(f, argnum=1)            # Should be dfdy(x,y) = 3x + 2,
dfdy(5., 1.)                        # so dfdy(5,1) = 15 + 2 = 17.

# Get the full gradient.
grad_f = grad(f, argnum=[0,1])
anp.array(grad_f(5., 1.))

from autograd import jacobian

f = lambda x: anp.array([x[0]**2, x[0]+x[1]])
f_jac = jacobian(f)
f_jac(anp.array([1., 1.]))

import tangent              # Install with 'pip install tangent'.

def f(x):                   # Tangent does not support lambda functions,
    return x**2 - x + 3

df = tangent.grad(f)
df(10)                      # ...but the functions do accept integers.

image = plt.imread('cameraman.jpg')
plt.imshow(image, cmap = 'gray')
plt.show()

#1. def Filter(image, F):
#2.     m, n = image.shape
#3.     h, k = F.shape

#5.    image_pad = # Create an array of zeros of the appropriate size
#6.   # Make the interior of image_pad equal to image
#7.    C = np.zeros(image.shape)
#8.    for i in range(m):
#9.        for j in range(n):
#10.            C[i,j] = # Compute C[i, j]

 # Create a larger matrix of zeros
image_pad = np.zeros((m+2, n+2))
# Make the interior of image_pad equal to the original image
image_pad[1:1+m, 1:1+n] = image
