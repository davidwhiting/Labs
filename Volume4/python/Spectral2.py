
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

N=24
x1 = (2.*np.pi/N)*np.arange(1,N+1)
f = np.sin(x1)**2.*np.cos(x1) + np.exp(2.*np.sin(x1+1))

# This array is reordered in Python to
# accomodate the ordering inside the fft function in scipy.
k = np.concatenate(( np.arange(0,N/2) ,
					 np.array([0])	, # Because hat{f}'(k) at k = N/2 is zero.
					 np.arange(-N/2+1,0,1)	))

# Approximates the derivative using the pseudospectral method
f_hat = fft(f)
fp_hat = ((1j*k)*f_hat)
fp = np.real(ifft(fp_hat))

# Calculates the derivative analytically
x2 = np.linspace(0,2*np.pi,200)
derivative = (2.*np.sin(x2)*np.cos(x2)**2. - 
				np.sin(x2)**3. + 
				2*np.cos(x2+1)*np.exp(2*np.sin(x2+1))
				)

plt.plot(x2,derivative,'-k',linewidth=2.)
plt.plot(x1,fp,'*b')
plt.savefig('spectral2_derivative.pdf')
plt.show()


% import numpy as np
% import matplotlib.pyplot as plt
% 
% # Solve the ODE $u_{xx} = e^u,$ with boundary conditions
% #  $u(0) = u(2\pi) = 0$
% 
% 
% 
t_steps = 150    # Time steps
x_steps = 100     # x steps

'''
Your code here to set things up
'''

sol = # RK4 method. Should return a t_steps by x_steps array

X,Y = np.meshgrid(x_domain, t_domain)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_wireframe(X,Y,sol)
plt.show()

