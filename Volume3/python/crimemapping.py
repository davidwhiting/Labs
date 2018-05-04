
>>> import scipy as sp
>>> from scipy import stats
>>> kernel = stats.gaussian_kde(data)
>>> print kernel.evaluate(sp.zeros(data.shape[0]))
