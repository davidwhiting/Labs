
from __future__ import division
from math import pi, sqrt, sin, cos, exp
from numpy import linspace, array, tanh, cosh, ones, arctan
import numpy as np
from scipy.special import erf
from scipy.optimize import root

from bvp6c import bvp6c, bvpinit, deval
from structure_variable import struct

R = 209
beta = 4.26
rho0 = 2.704e-3
g = 3.2172e-4
s = 26600

def C_d(u):
	return 1.174 - 0.9*cos(u)

def C_l(u):
	return 0.6*sin(u)

def ode(x,y):
	# Parameters:
	# x: independent variable (unused in our ODEs)
	# y: vector-valued dependent variable; it is an ndarray 
	# 	 with shape (7,)
	
	# Returns: 
	# ndarray of length (7,) that evalutes the RHS of the ODES
	u =	 arctan((6*y[4])/(9*y[0]*y[3] ))
	rho = rho0*exp(-beta*R*y[2])
	out = y[6]*array([
				 # G_0
				 -s*rho*y[0]**2*C_d(u) - g*sin(y[1])/(1+y[2])**2,	
				 # G_1	 
				( s*rho*y[0]*C_l(u) + y[0]*cos(y[1])/(R*(1 + y[2])) - 
				  g*cos(y[1])/(y[0]*(1+y[2])**2) ),						 
				 # G_2
				y[0]*sin(y[1])/R,		
				 # G_3								 
				-( 30*y[0]**2.*sqrt(rho)+ y[3]*(-2*s*rho*y[0]*C_d(u)) + 
				   y[4]*( s*rho*C_l(u) +cos(y[1])/(R*(1 + y[2])) + 
						  g*cos(y[1])/( y[0]**2*(1+y[2])**2 ) 
							) + 
				   y[5]*(sin(y[1])/R)	   ),	
				  # G_4						 
				-( y[3]*( -g*cos(y[1])/(1+y[2])**2	) + 
				   y[4]*( -y[0]*sin(y[1])/(R*(1+y[2])) + 
						  g*sin(y[1])/(y[0]*(1+y[2])**2 ) 
							) + 
				   y[5]*(y[0]*cos(y[1])/R )	   ),
				  # G_5 -- This line needs to be completed.						 
				  ,			
				  # G_6	
					0 									 
			   ])
	return out

T0 = 230	
	
def ode_auxiliary(t,y):
	u = y[3]*erf( y[4]*(y[5]-(1.*t)/T0) )
	rho = rho0*exp(-beta*R*y[2])
	out = array([-s*rho*y[0]**2*C_d(u) - g*sin(y[1])/(1+y[2])**2,
				  ( s*rho*y[0]*C_l(u) + y[0]*cos(y[1])/(R*(1 + y[2])) -
				  g*cos(y[1])/(y[0]*(1+y[2])**2) ),
				  y[0]*sin(y[1])/R,
				  0,
				  0,
				  0		])
	return out

def bcs_auxiliary(ya,yb):
	out1 = array([ ya[0]-.36,
				  ya[1]+8.1*pi/180,
				  ya[2]-4/R
				  ])
	out2 = array([ yb[0]-.27,
				  yb[1],
				  yb[2]-2.5/R
				  ])
	return out1, out2

%
% options = struct()
% # options include abstol, reltol, singularterm, stats, vectorized, maxnewpts,slopeout,xint
% options.abstol, options.reltol = 1e-8, 1e-7
% options.fjacobian = ode_auxiliary_jacobian
% options.bcjacobian = bcs_auxiliary_jacobian
% options.nmax = 2000
%
% solinit = bvpinit(np.linspace(0,1,100),initial_guess)
% sol = bvp6c(ode,bcs,solinit,options)
%
% N = 240
% xint = linspace(0,T0,N+1)
% num_sol_auxiliary, _ = deval(sol,xint)
%
%
% 
problem_auxiliary = bvp_solver.ProblemDefinition(num_ODE = 6,
										  num_parameters = 0,
										  num_left_boundary_conditions = 3,
										  boundary_points = (0, T0),
										  function = ode_auxiliary,
										  boundary_conditions = bcs_auxiliary)

solution_auxiliary = bvp_solver.solve(problem_auxiliary,
								solution_guess = guess_auxiliary)

N = 240
t_guess = linspace(0,T0,N+1)
guess = solution_auxiliary(t_guess)

def guess_auxiliary(t):
	out = array([ .5*(.36+.27)-.5*(.36-.27)*tanh(.025*(t-.45*T_init)),
			# Finish this line, 
			# And this one, 
			p1*ones(t.shape),
			p2*ones(t.shape),
			p3*ones(t.shape)   ])
	return out
