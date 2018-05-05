
>>> from scipy import stats
>>> import numpy as np

# Create an object for the standard normal distribution.
>>> F = stats.norm()
# loc is the mean and scale is the standard deviation.
>>> G = stats.norm(loc=3, scale=2)

# Calculate the probability of drawing a 1 from the normal distribution.
>>> F.pdf(1)
0.24197072451914337

# Draw a number at random from the normal distribution.
>>> F.rvs()
0.95779975

# Specifying a size returns a numpy.ndarray.
>>> F.rvs(size=2)
array([-0.40375954, 1.10956538])


>>> from matplotlib import pyplot as plt
# Create a linspace for our graph.
>>> X = np.linspace(-4, 4, 100)
# Use the normal distribution created previously.
>>> plt.plot(X, F.pdf(X))
>>> plt.show()

# Choose the importance distribution with mean 4 and std dev 1
>>> G = stats.norm(loc=4, scale=1)
>>> g = G.pdf                   # Equation for importance distribution
>>> sampler = G.rvs             # Samples from importance distribution

>>> 1 - stats.norm.cdf(3)

# Create the gamma distribution object with a = 9, theta = .5
>>> F = stats.gamma(a=9, scale=.5)

# Create a 2-dim multivariate normal object with a zero vector mean and cov matrix I
>>> F = stats.multivariate_normal(mean=np.zeros(2), cov=np.eye(2))
>>> F.pdf(np.array([1,1]))
0.058549831524319168
>>> F.rvs(size=3)
array([[ 0.03429396,  0.13618787],
       [-0.12011818,  0,88691591],
       [-0.16356289,  0.53757853]])