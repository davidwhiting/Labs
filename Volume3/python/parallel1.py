[style=ShellInput]
$ ipcontroller start
[style=ShellInput]
$ ipengine start
[style=ShellInput]
$ ipcluster start # By default assigns an engine to each processor core.
$ ipcluster start --n 4 # Starts a cluster with 4 engines.
[style=ShellInput]
$ ipcluster nbextension enable

>>> from ipyparallel import Client
>>> client = Client()
>>> client.ids # If you had four processors, the output would be as follows.
<< [0, 1, 2, 3] >>

>>> dview = client[:] # Group all engines into a Direct View.
>>> dview2 = client[:2] # Group engines 0,1, and 2 into a Direct View.

# Target only engines 0 and 2 until changed.
>>> dview.targets = [0,2]
# To revert to all engines,
>>> dview.targets = None

>>> with dview.sync_imports():
...	import numpy
>>> dview.execute('''np = numpy''')
# Or simply,
>>> dview.execute('''import numpy as np''')

# The following is a blocking implementation of a for loop
>>> dview.execute('''import numpy as np''')
>>> results = []
>>> for i in range(1000):
...	results.append(dview.apply_sync(lambda x: np.<<sum>>(np.random.random(x)),i))
''' 
The for loop waits until each answer is gathered, then appends them. This 
blocking method takes 16.8495s with 4 engines.
'''

# The following is a non-blocking implementation
>>> results2 = []
>>> for j in range(1000):
...	results2.append(dview.apply_async(lambda x: np.sum(np.random.random(x)),j))
>>> results2 = [x.get() for x in results2]
''' 
In this example, the for loop appends an ASyncResult object to the list
and continues the loop. The answers are later retrieved after they have
finished processing. Though this method has two for loops, it takes only
12.9706s with 4 engines.
'''

# Make blocking default
>>> dview.block = True

# To share the variables 'a' and 'b' across all engines
>>> dview['a'] = 10
>>> dview['b'] = 5
# These two commands are shorthand for
>>> dview.push({'a':10, 'b':5}, block=True)

# To ensure the variables are on engine 0
>>> client[0]['a']
<< 10 >>

# On all engines
>>> dview['a']
<< [10, 10, 10, 10] >>
# Which is shorthand for
>>> dview.pull('a', block=True)

>>> res = dview.pull(['a', 'b'],block=False)
>>> res.ready()
<< True >>
>>> res.get()
<< [[10, 5], [10, 5], [10, 5], [10, 5]] >>

>>> import numpy as np
>>> one = [1, 2, 3, 4]
>>> two = np.array([1, 2, 3, 4])
>>> dview.scatter("one_part", one, block=True)
>>> dview["one_part"]
<< [[1], [2], [3], [4]] >>
>>> dview.scatter("two_part", one, block=True, targets=[0,2])
>>> dview["two_part"]
<< [array([1, 2]), array([3, 4])] >>

>>> print(dview.gather("one_part", block=True))
<< [1, 2, 3, 4] >>
>>> print(dview.gather("two_part", block=True,targets=[0,2]))
<< array([1, 2, 3, 4]) >>

>>> dview.execute('''
... import numpy as np
... rand = np.random.random
... c = np.sum(rand(a+b))
... ''')
>>> dview['c']
<< [7.9619371, 7.3431609, 7.4818468, 8.2783728] >>

# More involved example
>>> dview.scatter("p1", list(range(4)), block=True)
>>> dview.scatter("p2", list(range(10,14)), block=True)
>>> dview["p1"]
<< [[0], [1], [2], [3]] >>
>>> dview["p2"]
<< [[10], [11], [12], [13]] >>
>>> dview.execute('''
... def adding(a,b):
...     return a+b
... result = adding(p1[0],p2[0])
... ''')
<< <AsyncResult: execute:finished> >>
>>> dview['result']
<< [10, 12, 14, 16] >>

# apply_sync always blocks
>>> dview.apply_sync(lambda x: a+b+x, 20)
<< [35, 35, 35, 35] >>

# apply_async never blocks
>>> def double_add(x,y):
...     return 2*x + 2*y
>>> answer = dview.apply_async(double_add,5,7)
>>> answer.ready()
<< True >>
>>> answer.get()
<< [24, 24, 24, 24] >>

<< means = [0.0031776784, -0.0058112042, 0.0012574772, -0.0059655951] >>
<< maxs = [4.0388107, 4.3664958, 4.2060184, 4.3391623] >>
<< mins = [-4.1508589, -4.3848019, -4.1313324, -4.2826519] >>

# Single input function
>>> num_list = [1,2,3,4,5,6,7,8]
>>> def triple(x):
...     return 3*x
>>> answer = dview.<<map>>(triple, num_list, block=True)
<< [3, 6, 9, 12, 15, 18, 21, 24] >>

# Multiple input function
>>> def add_three(x,y,z):
...     return x+y+z
>>> x_list = [1,2,3,4]
>>> y_list = [2,3,4,5]
>>> z_list = [3,4,5,6]
>>> dview.map_sync(add_three, x_list, y_list, z_list)
<< [6, 9, 12, 15] >>

# Engine variable function
>>> def mult(x):
...		return a*x
>>> answer = dview.map_async(mult, x_list)
>>> answer.get()
<< [10, 20, 30, 40] >>

>>> from scipy.spatial import KDTree
# Columns should be latitude then longitude
>>> lat_long_array = np.array([ [1,2], [2,3], [3,4] ])
>>> tree = KDTree(lat_long_array)
>>> sample_point = np.array([2,5])
# Queries can be made with
>>> q = tree.query(sample_point)
>>> q
<< (1.4142135623730951, 2) >>

%>>> from nltk.corpus import stopwords
%>>> set_stopwords = set(stopwords.words('english'))
%
%      'blue' 'car' 'red' 'rose' 'violet'
%doc1     0     0     1      1       0
%doc2     1     0     0      0       1
%doc3     0     1     1      0       0
%doc4     0     1     2      1       0
%[style=ShellInput]
$ conda update conda
$ conda update anaconda
$ conda install ipyparallel
[style=ShellInput]
$ ipcontroller --ip=<controller IP> --user=<user of controller> --enginessh=<user of controller>@<controller IP>
[style=ShellInput]
$ ipengine --location=<controller IP> --ssh=<user of controller>@<controller IP>

# %px
In [4]: with dview.sync_imports():
   ...:     import numpy
   ...:     
importing numpy on engine(s)
In [5]: \%px a = numpy.random.random(2)

In [6]: dview['a']
Out[6]: 
[array([ 0.30390162,  0.14667075]),
 array([ 0.95797678,  0.59487915]),
 array([ 0.20123566,  0.57919846]),
 array([ 0.87991814,  0.31579495])]
 
 # %autopx
In [7]: %autopx
%autopx enabled
In [8]: max_draw = numpy.max(a)

In [9]: print('Max_Draw: {}'.format(max_draw))
[stdout:0] Max_Draw: 0.30390161663280246
[stdout:1] Max_Draw: 0.957976784975849
[stdout:2] Max_Draw: 0.5791984571339429
[stdout:3] Max_Draw: 0.8799181411958089

In [10]: %autopx
%autopx disabled

# Remote decorator
>>> @dview.remote(block=True)
>>> def plusone():
...	return a+1
>>> dview['a'] = 5
>>> plusone()
<< [6, 6, 6, 6,] >>

# Parallel decorator
>>> import numpy as np

>>> @dview.parallel(block=True)
>>> def combine(A,B):
...	return A+B
>>> ex1 = np.random.random((3,3))
>>> ex2 = np.random.random((3,3))
>>> print(ex1+ex2)
<< [[ 0.87361929  1.41110357  0.77616724]
 [ 1.32206426  1.48864976  1.07324298]
 [ 0.6510846   0.45323311  0.71139272]] >>
>>> print(combine(ex1,ex2))
<< [[ 0.87361929  1.41110357  0.77616724]
 [ 1.32206426  1.48864976  1.07324298]
 [ 0.6510846   0.45323311  0.71139272]] >>
 
import numpy as np
for i in range(10000):
    np.random.random(100000)

from ipyparallel import Client
client = Client()
dview = client[:]

dview.execute("""
import numpy as np
for i in range(10000):
    np.random.random(100000)
""")
