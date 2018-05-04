
def recursive_sum(n):
	"""Calculate the sum of all positive integers in [1, n] recursively."""
    # Base Case: the sum of all positive integers in [1, 1] is 1.
	if n == 1:
		return 1

	# If the base case hasn't been reached, the function recurses by calling
    # itself on the next smallest integer. The result of that call, plus the
    # particular 'n' from this call, gives the result.
	else:
		return n + recursive_sum(n-1)

# To find recursive_sum(5), calculate recursive_sum(4).
# To find recursive_sum(4), calculate recursive_sum(3).
# This continues until the base case is reached.

recursive_sum(5)		# return 5 + recursive_sum(4)
	recursive_sum(4)		# return 4 + recursive_sum(3)
		recursive_sum(3)		# return 3 + recursive_sum(2)
			recursive_sum(2)		# return 2 + recursive_sum(1)
				recursive_sum(1)		# Base case: return 1.

recursive_sum(5)		# return 5 + 10
	recursive_sum(4)		# return 4 + 6
		recursive_sum(3)		# return 3 + 3
			recursive_sum(2)		# return 2 + 1
				recursive_sum(1)		# Base case: return 1.

def iterative_fib(n):
	"""Calculate the nth Fibonacci number iteratively."""
	fibonacci = []          # Initialize an empty list.
	fibonacci.append(0)		# Append 0 (the 0th Fibonacci number).
	fibonacci.append(1)		# Append 1 (the 1st Fibonacci number).
	for i in range(1, n):
		# Starting at the third entry, calculate the next number
		# by adding the last two entries in the list.
		fibonacci.append(fibonacci[-1] + fibonacci[-2])
	# When the entire list has been loaded, return the nth entry.
	return fibonacci[n]

def recursive_fib(n):
	"""Calculate the nth Fibonacci number recursively."""
	# The base cases are the first two Fibonacci numbers.
	if n == 0:				# Base case 1: the 0th Fibonacci number is 0.
		return 0
	elif n == 1:			# Base case 2: the 1st Fibonacci number is 1.
		return 1
	# If this call isn't a base case, the function recurses by calling
	# itself to calculate the previous two Fibonacci numbers.
	else:
		return recursive_fib(n-1) + recursive_fib(n-2)

recursive_fib(5)		# The original call makes two additional calls:
	recursive_fib(4)		# this one...
		recursive_fib(3)
			recursive_fib(2)
				recursive_fib(1)		# Base case 2: return 1
				recursive_fib(0)		# Base case 1: return 0
			recursive_fib(1)		# Base case 2: return 1
		recursive_fib(2)
			recursive_fib(1)		# Base case 2: return 1
			recursive_fib(0)		# Base case 1: return 0
	recursive_fib(3)		# ...and this one.
		recursive_fib(2)
			recursive_fib(1)		# Base case 2: return 1
			recursive_fib(0)		# Base case 1: return 0
		recursive_fib(1)		# Base case 2: return 1

class SinglyLinkedListNode:
    """Simple singly linked list node."""
    def __init__(self, data):
        self.value, self.<<next>> = data, None

class SinglyLinkedList:
    """A very simple singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a Node containing 'data' to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.<<next>> = n
            self.tail = n

def iterative_search(linkedlist, data):
    """Search 'linkedlist' iteratively for a node containing 'data'."""
	current = linkedlist.head
	while current is not None:
		if current.value == data:
			return current
		current = current.<<next>>
	raise ValueError(str(data) + " is not in the list.")

class BSTNode:
    """A Node class for Binary Search Trees. Contains some data, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the data attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None      # A reference to this node's parent node.
        self.left = None      # This node's value will be less than self.value
        self.right = None     # This node's value will be greater than self.value

class BST:
    """Binary Search Tree data structure class.
    The 'root' attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

class BST:
    # ...
    def find(self, data):
        """Return the node containing 'data'. If there is no such node
        in the tree, or if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            'data' is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

# Sequentially adding ordered integers destroys the efficiency of a BST.
>>> unbalanced_tree = BST()
>>> for i in range(10):
...     unbalanced_tree.insert(i)
...
# The tree is perfectly flat, so it loses its search efficiency.
>>> print(unbalanced_tree)
[0]
[1]
[2]
[3]
[4]
[5]
[6]
[7]
[8]
[9]

>>> balanced_tree = AVL()
>>> for i in range(10):
...     balanced_tree.insert(i)
...
# The AVL tree is balanced, so it retains (and optimizes) its search efficiency.
>>> print(balanced_tree)
[3]
[1, 7]
[0, 2, 5, 8]
[4, 6, 9]
