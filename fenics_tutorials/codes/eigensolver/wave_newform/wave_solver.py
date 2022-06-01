from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print( "DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_slepc():
    print ("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()


# Parameters

gamma = 1.4 # ratio of specific heats (non-dim)
p_amb = 1e5 # ambient pressure [Pa]

rho_in_dim = 1.400  # Upstream density (kg/m3)
rho_out_dim = 0.975 # Downstream Density (kg/m3)

x_f = 0.25 # Position of the heat release (metres)
a_f = 0.05 # Length of the heat release (metres)



rho = Expression("rho_u+0.5*(rho_d-rho_u)*(1+tanh((x[0]-x_f)/a_f))", degree=1,
                 rho_u = rho_in_dim,
                 rho_d = rho_out_dim,
                 x_f = x_f,
                 a_f = a_f)


c = Expression("sqrt(gamma*p_amb/rho)", degree = 1,
               gamma = gamma,
               p_amb = p_amb,
               rho = rho) # Variable Speed of sound (m/s)


"""
----------    FINITE ELEMENT MODEL    ------------

"""
# Define mesh, function space

N=400
x_left  =  0.0
x_right = +1.0

mesh = IntervalMesh ( N, x_left, x_right )
V = FunctionSpace(mesh, "Lagrange", 1)

# #define boundary
def boundary(x, on_boundary):
    return on_boundary

#apply essential boundary conditions
bc = DirichletBC(V, 0, boundary)


#define functions
u = TrialFunction(V)
v = TestFunction(V)

func = Expression("1.0", degree = 1)
# c_test = Constant(1.4)
c_test = Expression("0.0*x[0]+1.0",degree=1)
#define problem
a_ =  c_test * inner(grad(u), grad(v)) * dx
c_ = u * v * dx

#assemble stiffness matrix
A = PETScMatrix()
assemble(a_, tensor=A)
C = PETScMatrix()
assemble(c_, tensor=C)
bc.apply(A)          # apply the boundary conditions



#create eigensolver
eigensolver = SLEPcEigenSolver(A,C)
eigensolver.parameters['spectrum'] = 'smallest magnitude'
eigensolver.parameters['solver']   = 'lapack'
eigensolver.parameters['tolerance'] = 1.e-15

#solve for eigenvalues
eigensolver.solve()

u = Function(V)
for i in range(0,3):
    #extract next eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(i)
    print ('eigenvalue:', r)

    #assign eigenvector to function
    u.vector()[:] = rx

    #plot eigenfunction
    label="%.2f" % round(r, 2)
    # plt.legend(label)
    plot(u,label=label)
    plt.legend()

# plt.autoscale()
    
plt.ylim(-0.15, 0.15)
    