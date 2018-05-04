
>>> # toy HMM example to be used to check answers
>>> A = np.array([[.7, .4],[.3, .6]])
>>> B = np.array([[.1,.7],[.4, .2],[.5, .1]])
>>> pi = np.array([.6, .4])
>>> obs = np.array([0, 1, 0, 2])

class hmm(object):
    """
    Finite state space hidden markov model.
    """
    def __init__(self):
        """
        Initialize model parameters.
        """
        self.A = None
        self.B = None
        self.pi = None

def _forward(self, obs):
    """
    Compute the scaled forward probability matrix and scaling factors.

    Parameters
    ----------
    obs : ndarray of shape (T,)
        The observation sequence

    Returns
    -------
    alpha : ndarray of shape (T,N)
        The scaled forward probability matrix
    c : ndarray of shape (T,)
        The scaling factors c = [c_1,c_2,...,c_T]
    """
    pass

>>> h = hmm()
>>> h.A = A
>>> h.B = B
>>> h.pi = pi
>>> alpha, c = h._forward(obs)
>>> print -(np.log(c)).sum() # the log prob of observation
-4.6429135909

def _backward(self, obs, c):
    """
    Compute the scaled backward probability matrix.

     Parameters
    ----------
    obs : ndarray of shape (T,)
        The observation sequence
    c : ndarray of shape (T,)
        The scaling factors from the forward pass

    Returns
    -------
    beta : ndarray of shape (T,N)
        The scaled backward probability matrix
    """
    pass

>>> beta = h._backward(obs, c)
>>> print beta
[[ 3.1361635   2.89939354]
 [ 2.86699344  4.39229044]
 [ 3.898812    2.66760821]
 [ 3.56816483  3.56816483]]

def _delta(self, obs, alpha, beta):
    """
    Compute the delta probabilities.

    Parameters
    ----------
    obs : ndarray of shape (T,)
        The observation sequence
    alpha : ndarray of shape (T,N)
        The scaled forward probability matrix from the forward pass
    beta : ndarray of shape (T,N)
        The scaled backward probability matrix from the backward pass

    Returns
    -------
    delta : ndarray of shape (T-1,N,N)
        The delta probability array
    gamma : ndarray of shape (T,N)
        The gamma probability array
    """
    pass

>>> delta, gamma = h._delta(obs, alpha, beta)
>>> print delta
[[[ 0.14166321  0.0465066 ]
  [ 0.37776855  0.43406164]]

 [[ 0.17015868  0.34927307]
  [ 0.05871895  0.4218493 ]]

 [[ 0.21080834  0.01806929]
  [ 0.59317106  0.17795132]]]
>>> print gamma
[[ 0.18816981  0.81183019]
 [ 0.51943175  0.48056825]
 [ 0.22887763  0.77112237]
 [ 0.8039794   0.1960206 ]]

def _estimate(self, obs, delta, gamma):
    """
    Estimate better parameter values.

    Parameters
    ----------
    obs : ndarray of shape (T,)
        The observation sequence
    delta : ndarray of shape (T-1,N,N)
        The delta probability array
    gamma : ndarray of shape (T,N)
        The gamma probability array
    """
    # update self.A, self.B, self.pi in place
    pass

h._estimate(obs, delta)
>>> print h.A
[[ 0.55807991  0.49898142]
 [ 0.44192009  0.50101858]]
>>> print h.B
[[ 0.23961928  0.70056364]
 [ 0.29844534  0.21268397]
 [ 0.46193538  0.08675238]]
>>> print h.pi
[ 0.18816981  0.81183019]

>>> # assume N is defined
>>> # define M to be the number of distinct observed states
>>> M = len(set(obs))
>>> A = np.random.dirichlet(np.ones(N), size=N).T
>>> B = np.random.dirichlet(np.ones(M), size=N).T
>>> pi = np.random.dirichlet(np.ones(N))

def fit(self, obs, A, B, pi, max_iter=100, tol=1e-3):
    """
    Fit the model parameters to a given observation sequence.

    Parameters
    ----------
    obs : ndarray of shape (T,)
        Observation sequence on which to train the model.
    A : stochastic ndarray of shape (N,N)
        Initialization of state transition matrix
    B : stochastic ndarray of shape (M,N)
        Initialization of state observation matrix
    pi : stochastic ndarray of shape (N,)
        Initialization of initial state distribution
    max_iter : integer
        The maximum number of iterations to take
    tol : float
        The convergence threshold for change in log-probability
    """
    # initialize self.A, self.B, self.pi
    # run the iteration
    pass

>>> import numpy as np
>>> import string
>>> import codecs


>>> def vec_translate(a, my_dict):    

>>>     # translate numpy array from symbols to state numbers or vice versa
>>>     return np.vectorize(my_dict.__getitem__)(a)


>>> def prep_data(filename):
    
>>>     # Get the data as a single string
>>>     with codecs.open(filename, encoding='utf-8') as f:
>>>         data=f.read().lower()  #and convert to all lower case
    
>>>     # remove punctuation and newlines
>>>     remove_punct_map = {ord(char): None for char in string.punctuation+"\n\r"}
>>>     data = data.translate(remove_punct_map)

>>>     # make a list of the symbols in the data 
>>>     symbols = sorted(list(set(data)))

>>>     # convert the data to a NumPy array of symbols
>>>     a = np.array(list(data))
    
>>>     #make a conversion dictionary from symbols to state numbers
>>>     symbols_to_obsstates = {x:i for i,x in enumerate(symbols)}

>>>     #convert the symbols in a to state numbers
>>>     obs_sequence = vec_translate(a,symbols_to_obsstates)    
    
>>>     return symbols, obs_sequence

>>> symbols, obs = prep_data('declaration.txt')

>>> for i in xrange(len(h.B)):
>>>     print  u"{0}, {1:0.4f}, {2:0.4f}".format(symbols[i], h.B[i,0], h.B[i,1])
 , 0.0051, 0.3324
a, 0.0000, 0.1247
c, 0.0460, 0.0000
b, 0.0237, 0.0000
e, 0.0000, 0.2245
d, 0.0630, 0.0000
g, 0.0325, 0.0000
f, 0.0450, 0.0000
i, 0.0000, 0.1174
h, 0.0806, 0.0070
k, 0.0031, 0.0005
j, 0.0040, 0.0000
m, 0.0360, 0.0000
l, 0.0569, 0.0001
o, 0.0009, 0.1331
n, 0.1207, 0.0000
q, 0.0015, 0.0000
p, 0.0345, 0.0000
s, 0.1195, 0.0000
r, 0.1062, 0.0000
u, 0.0000, 0.0546
t, 0.1600, 0.0000
w, 0.0242, 0.0000
v, 0.0185, 0.0000
y, 0.0147, 0.0058
x, 0.0022, 0.0000
z, 0.0010, 0.0000

>>> symbols, obs = prep_data('WarAndPeace.txt')
