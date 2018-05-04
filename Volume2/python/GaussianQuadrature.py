
>>> from scipy.integrate import quad

>>> f = lambda x: np.cos(x) * np.sin(x)**2 # Function to integrate.
>>> g = lambda x: np.sin(x)**3 / 3         # Indefinite integral.

# quad returns an array, the first entry is the computed value.
>>> calc = quad(f, -2, 3)[0]
>>> exact = g(3) - g(-2)                   # Exact value of the integral.
>>> np.<<abs>>(exact - calc)                   # Error of the approximation.
0.0

from scipy.stats import norm
normal = norm()                  # Make a standard normal random variable.
exact = normal.cdf(1)            # Integrate the pdf from -infinity to 1.
