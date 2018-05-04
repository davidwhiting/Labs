
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

r0 = 5 # Initial rabbit population
w0 = 3 # Initial wolf population

# Define rabbit growth paramters
a = 1.0
alpha = 0.5

# Define wolf growth parameters
c = 0.75
gamma = 0.25

t_f = 20 # How long we want to run the model
y0 = [r0, w0]

# Initialize time and output arrays needed for the ode solver
t = np.linspace(0, t_f, 5*t_f)
y = np.zeros((len(t), len(y0)))
y[0,:] = y0

def predator_prey(t, y, a, alpha, c, gamma):
	'''
	Parameters:
	--------------
	t:	time variable.
	y:	an array of length len(y0) representing current wolf and rabbit populations at time t.
	a, alpha, c, gamma:	growth parameters. These are keyword arguments and can be of any length.
	
	Return:
	--------
	Return a list corresponding to the Predator-Prey model.
	'''
	pass

predator_prey_ode = lambda t, y:predator_prey(t, y, a, alpha, c, gamma)
p_p_solver = ode(predator_prey_ode).set_integrator('dopri5') # set the numerical integrator
p_p_solver.set_initial_value(y0, 0) # Set the initial values. The second argument is the initial time, which we set to 0

for j in range(1, len(t)):
	y[j,:] = p_p_solver.integrate((t[j]))

plt.plot(t, y[:,0], label='rabbit')
plt.plot(t, y[:,1], label='wolf')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.show()

from math import log
# Fixed Constants:
rho_F = 9400.
rho_L = 1800.
gamma_F = 3.2
gamma_L = 22.
eta_F = 180.
eta_L = 230.
C = 10.4  # Forbes constant
beta_AT = 0.14  # Adaptive Thermogenesis
beta_TEF = 0.1   # Thermic Effect of Feeding
K = 0

def forbes(F):
    C1 = C * rho_L / rho_F
    return C1 / (C1 + F)

def energy_balance(F, L, EI, PAL):
    p = forbes(F)
    a1 = (1. / PAL - beta_AT) * EI - K - gamma_F * F - gamma_L * L
    a2 = (1 - p) * eta_F / rho_F + p * eta_L / rho_L + 1. / PAL
    return a1 / a2

def weight_odesystem(t, y, EI, PAL):
    F, L = y[0], y[1]
    p, EB = forbes(F), energy_balance(F, L, EI, PAL)
    return np.array([(1 - p) * EB / rho_F , p * EB / rho_L])

def fat_mass(BW, age, H, sex):
    BMI = BW / H**2.
    if sex == 'male':
        return BW * (-103.91 + 37.31 * log(BMI) + 0.14 * age) / 100
    else:
        return BW * (-102.01 + 39.96 * log(BMI) + 0.14 * age) / 100


from scipy.integrate import odeint
a, b = 0., 13.                    # (Nondimensional) Time interval for one 'period'
alpha = 1. / 3                    # Nondimensional parameter
dim = 2                           # dimension of the system
y0 = np.array([1 / 2., 1 / 3.])   # initial conditions

# Note: swapping order of arguments to match the calling convention
# used in the built in IVP solver.
def Lotka_Volterra(y, x):
    return np.array([y[0] * (1. - y[1]), alpha * y[1] * (y[0] - 1.)])

subintervals = 200
# Using the built in ode solver
Y = odeint(Lotka_Volterra, y0, np.linspace(a, b, subintervals))

# Plot the direction field
Y1, Y2 = np.meshgrid(np.arange(0, 4.5, .2), np.arange(0, 4.5, .2), sparse=True, copy=False)
U, V = Lotka_Volterra((Y1, Y2), 0)
Q = plt.quiver(Y1[::3, ::3], Y2[::3, ::3],  U[::3, ::3],  V[::3, ::3], pivot='mid', color='b', units='dots',width=3.)
# Plot the 2 Equilibrium points
plt.plot(1, 1, 'ok', markersize=8)
plt.plot(0, 0, 'ok', markersize=8)
# Plot the solution in phase space
plt.plot(Y[:,0], Y[:,1], '-k', linewidth=2.0)
plt.plot(Y[::10,0], Y[::10,1], '*b')

plt.axis([-.5, 4.5, -.5, 4.5])
plt.title("Phase Portrait of the Lotka-Volterra Predator-Prey Model")
plt.xlabel('Prey',fontsize=15)
plt.ylabel('Predators',fontsize=15)
plt.show()
