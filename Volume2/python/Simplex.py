
>>> L = [2, 3, 4, 0, 1]

>>> L[2], L[3] = L[3], L[2]
>>> L
[2, 3, 0, 4, 1]

>>> import SimplexSolver

# Initialize objective function and constraints.
>>> c = np.array([3., 2])
>>> b = np.array([2., 5, 7])
>>> A = np.array([[1., -1], [3, 1], [4, 3]])

# Instantiate the simplex solver, then solve the problem.
>>> solver = SimplexSolver(c, A, b)
>>> sol = solver.solve()
>>> print(sol)
(5.200,
 {0: 1.600, 1: 0.200, 2: 0.600},
 {3: 0, 4: 0})

>>> N = zeros((s+1,s+1))
>>> N[0:s,0:s] = T
>>> N[n:s+1,s] = 1
>>> N[0,s] = -1

T[1:s,0:s] = N[1:s,0:s]

T[0,0:s] = np.dot(T[0,0:s], T)
