
% def ode_f(x,y): return numpy.array([y[1] , -4.*y[0] - 9.*numpy.sin(x)])
%
% a, b = 0., 3*numpy.pi/4.
% alpha, beta =  1., -(1.+3*numpy.sqrt(2))/2.
%
% 
% reltol, abstol = 1e-9,1e-8
% example = ode(ode_f).set_integrator('dopri5',atol=abstol,rtol=reltol)
% example.set_initial_value(np.array([alpha,t0]),a)
% y0 = example.integrate(b)[0]
% 