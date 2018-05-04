

def linearized_init(M, m, l, q1, q2, q3, q4, r): 
	'''
	Parameters:
	----------
	M, m: floats
          masses of the rickshaw and the present
	l 	: float
          length of the rod
	q1, q2, q3, q4, r : floats
         relative weights of the position and velocity of the rickshaw, the 
		 angular displacement theta and the change in theta, and the control
	
	Return
	-------
	A : ndarray of shape (4,4) 
	B : ndarray of shape (4,1) 
	Q : ndarray of shape (4,4) 
	R : ndarray of shape (1,1) 
	'''
	pass	

def find_P(A, B, Q, R):
	'''
	Parameters:
	----------
	A, Q 	: ndarrays of shape (4,4)
	B		: ndarray of shape (4,1)
	R		: ndarray of shape (1,1)
	
	Returns
	-------
	P		: the matrix solution of the Riccati equation
	'''
	pass



M, m = 23., 5.
l = 4.
q1, q2, q3, q4 = 1., 1., 1., 1.
r = 10.

def rickshaw(tv, X0, A, B, Q, R, P):
	'''
	Parameters:
	----------
	tv 	: ndarray of time values, with shape (n+1,)
	X0 	: Initial conditions on state variables
	A, Q: ndarrays of shape (4,4)
	B	: ndarray of shape (4,1)
	R	: ndarray of shape (1,1)
	P	: ndarray of shape (4,4)
	
	Returns
	-------
	Z : ndarray of shape (n+1,4), the state vector at each time
	U : ndarray of shape (n+1,), the control values
	'''
	pass

M, m = 23., 5.
l = 4.
q1, q2, q3, q4 = 1., 1., 1., 1.
r = 10.
tf = None
X0 = np.array([-1, -1, .1, -.2])

M, m = 23., 5.
l = 4.
q1, q2, q3, q4 = 1., 1., 1., 1.
r = 10.
tf = 60
X0 = np.array([-1, -1, .1, -.2])
