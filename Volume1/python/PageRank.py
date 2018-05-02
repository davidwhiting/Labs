A = np.array([[ 0,  0,  0,  0,  0,  0,  0,  1],
              [ 1,  0,  0,  0,  0,  0,  0,  0],
              [ 0,  0,  0,  0,  0,  0,  0,  0],
              [ 1,  0,  1,  0,  0,  0,  1,  0],
              [ 1,  0,  0,  0,  0,  1,  1,  0],
              [ 1,  0,  0,  0,  0,  0,  1,  0],
              [ 1,  0,  0,  0,  0,  0,  0,  0],
              [ 1,  0,  0,  0,  0,  0,  0,  0]])

def to_matrix( filename, n ):
    ''' Return the nxn adjacency matrix described by the file.

    INPUTS:
    filename - Name of a .txt file describing a directed graph. Lines
            describing edges should have the form
        '<from node>\t<to node>'.
        The file may also include comments.
    n   - The number of nodes in the graph described by datafile

    RETURN:
    Return a NumPy array.
    '''

# Open `matrix.txt' for read-only
with open('./matrix.txt', 'r') as myfile:
    for line in myfile:
        print line

line = '0\t4\n'
# strip() removes trailing whitespace from a line.
# split() returns a list of the space-separated pieces of the line.
line.strip().split()

Am = np.array([[ 0,  0,  0,  0,  0,  0,  0,  1],
               [ 1,  0,  0,  0,  0,  0,  0,  0],
               [ 1,  1,  1,  1,  1,  1,  1,  1],
               [ 1,  0,  1,  0,  0,  0,  1,  0],
               [ 1,  0,  0,  0,  0,  1,  1,  0],
               [ 1,  0,  0,  0,  0,  0,  1,  0],
               [ 1,  0,  0,  0,  0,  0,  0,  0],
               [ 1,  0,  0,  0,  0,  0,  0,  0]])

K = np.array([[ 0   ,  1   ,  1./8,  1./3,  1./3,  1./2,  1   ,  1   ],
              [ 0   ,  0   ,  1./8,  0   ,  0   ,  0   ,  0   ,  0   ],
              [ 0   ,  0   ,  1./8,  1./3,  0   ,  0   ,  0   ,  0   ],
              [ 0   ,  0   ,  1./8,  0   ,  0   ,  0   ,  0   ,  0   ],
              [ 0   ,  0   ,  1./8,  0   ,  0   ,  0   ,  0   ,  0   ],
              [ 0   ,  0   ,  1./8,  0   ,  1./3,  0   ,  0   ,  0   ],
              [ 0   ,  0   ,  1./8,  1./3,  1./3,  1./2,  0   ,  0   ],
              [ 1   ,  0   ,  1./8,  0   ,  0   ,  0   ,  0   ,  0   ]])

from scipy import linalg as la
I = np.eye(8)
d = .85
la.solve(I-d*K, ((1-d)/8)*np.ones(8))

def iter_solve( adj, N=None, d=.85, tol=1E-5):
    '''
    Return the page ranks of the network described by 'adj' using the iterative method.

    INPUTS:
    adj - A NumPy array representing the adjacency matrix of a directed
            graph
    N     - Restrict the computation to the first `N` nodes of the graph.
            Defaults to N=None; in this case, the entire matrix is used.
    d     - The damping factor, a float between 0 and 1.
            Defaults to .85.
    tol  - Stop iterating when the change in approximations to the
            solution is less than `tol'. Defaults to 1E-5.

    OUTPUTS:
    Return the approximation to the steady state of p.
    '''

def eig_solve( adj, N=None, d=.85):
    '''
    Return the page ranks of the network described by `adj`.

    INPUTS:
    adj - A NumPy array representing the adjacency matrix of a directed
            graph
    N     - Restrict the computation to the first `N` nodes of the graph.
            Defaults to N=None; in this case, the entire matrix is used.
    d     - The damping factor, a float between 0 and 1.
            Defaults to .85.

    OUTPUTS:
    Return the approximation to the steady state of p.
    '''

open('./ncaa2013.csv', 'r') as ncaafile:
ncaafile.readline() #reads and ignores the header line
for line in ncaafile:
   teams = line.strip().split(',') #split on commas
   print teams


s = array([[ 0,  7],
          [ 1,  0],
          [ 3,  0],
          [ 3,  6],
          [ 4,  0],
          [ 4,  5],
          [ 4,  6],
          [ 5,  0],
          [ 5,  6],
          [ 6,  0],
          [ 7,  0]])

import networkx as nx
G = nx.from_edgelist(edges, create_using=nx.DiGraph())
G.in_degree()
G.out_degree()
G.in_edges(0)
G.out_edges(0)

nx.pagerank(G, alpha=0.85) # alpha is the dampening factor.