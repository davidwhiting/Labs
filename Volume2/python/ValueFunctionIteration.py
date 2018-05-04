
# The policy functions used in the Figure below.
>>> pol1 = np.array([1, 0, 0, 0, 0])
>>> pol2 = np.array([0, 0, 0, 0, 1])
>>> pol3 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
>>> pol4 = np.array([.4, .3, .2, .1, 0])

%b = [[1, 2, 3]]
%np.repeat(b, 3, axis = 0)
%array([[1, 2, 3],
%[1, 2, 3],
%[1, 2, 3]])
%
>>> from matplotlib import pyplot as plt
>>> from matplotlib import cm
>>> from mpl_toolkits.mplot3d import Axes3D

>>> W = np.linspace(0, Wmax, N)
>>> x = np.arange(0, N)
>>> y = np.arange(0, T+2)
>>> X, Y = np.meshgrid(x, y)
>>> fig1 = plt.figure()
>>> ax1 = Axes3D(fig1)
>>> ax1.plot_surface(W[X], Y, np.transpose(V), cmap=cm.coolwarm)
>>> plt.show()

>>> fig2 = plt.figure()
>>> ax2 = Axes3D(fig2)
>>> y = np.arange(0,T+1)
>>> X, Y = np.meshgrid(x, y)
>>> ax2.plot_surface(W[X], Y, np.transpose(psi), cmap=cm.coolwarm)
>>> plt.show()

>>> from scipy import stats as st
>>> mu = 0
>>> sigma = 1
>>> eps = st.norm.cdf(1,loc=mu,scale=sigma) - st.norm.cdf(0,loc=mu,scale=sigma)

>>> e, gamma = discretenorm(*e_params)

>>> import numpy as np
>>> u_hat = np.repeat(u, K).reshape((N,N,K))*e

>>> E = (v*gamma).sum(axis=1)

>>> c = np.swapaxes(np.swapaxes(u_hat, 1, 2) + beta*E, 1, 2)

>>> c[np.triu_indices(N, k=1)] = -1e10

>>> v_new = np.max(c, axis=1)

>>> max_indices = np.argmax(c, axis=1)
>>> p = w[max_indices]

>>> x = np.arange(0,N)
>>> y = np.arange(0,K)
>>> X,Y = np.meshgrid(x,y)
>>> fig1 = plt.figure()
>>> ax1 = Axes3D(fig1)
>>> ax1.plot_surface(w[X], Y, v.T, cmap=cm.coolwarm)
>>> plt.show()

>>> e_params = (7, 4*.5, .5)
>>> stuff = stochEatCake(.9, 100, e_params, plot=True)

>>> from tauchenhussey import tauchenhussey
>>> e, gamma = tauchenhussey(*e_params)

>>> E = v.dot(gamma.T)

>>> c = u_hat + beta*E

>>> if iid:
>>>     # compute E as outlined in the previous problem
>>> else:
>>>     # compute E as outlined in the current problem
