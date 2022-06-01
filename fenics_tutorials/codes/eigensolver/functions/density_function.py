from fenics import *
import matplotlib.pyplot as plt
import numpy as np

mesh = UnitIntervalMesh(100)
V = FunctionSpace(mesh, 'CG', 1)

u = TrialFunction(V)
v = TestFunction(V)


# Parameters

gamma = 1.4 # ratio of specific heats (non-dim)
p_amb = 1e5 # ambient pressure [Pa]

rho_in = 1.400  # Upstream density (kg/m3)
rho_out = 0.975 # Downstream Density (kg/m3)

x_f = 0.25 # Position of the heat release (metres)
a_f = 0.05 # Length of the heat release (metres)

c_in  = 1.0
c_out = 1.1980376111153852

rho = Expression("rho_u+0.5*(rho_d-rho_u)*(1+tanh((x[0]-x_f)/a_f))", degree=1,
                 rho_u = rho_in,
                 rho_d = rho_out,
                 x_f = x_f,
                 a_f = a_f)

c = Expression("sqrt(gamma*p_amb/rho)", degree = 1,
               gamma = gamma,
               p_amb = p_amb,
               rho = rho) # Variable Speed of sound (m/s)

c_ = Expression("c_u+0.5*(c_d-c_u)*(1+tanh((x[0]-x_f)/a_f))", degree=1,
                 c_u = c_in,
                 c_d = c_out,
                 x_f = x_f,
                 a_f = a_f)

c_stefano = Expression('x[0] <= x_f ? c_in : c_out', degree=0, x_f=x_f, c_in=c_in, c_out=c_out)

f = interpolate(c_, V)
sos = interpolate(c_stefano, V)

pl, ax = plt.subplots(); fig = plt.gcf(); fig.set_size_inches(16, 4)
plt.subplot(1, 2, 1); p1 = plot(f)
plt.subplot(1, 2, 2); p2 = plot(sos)

plt.show()
