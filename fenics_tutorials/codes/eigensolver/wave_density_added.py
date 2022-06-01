from dolfin import *
import matplotlib.pyplot as plt

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

f = Expression('0.0',degree=2)

# Define density function

rho = Expression("rho_d+0.5*(rho_d-rho_u)*(1+tanh((x[0]-x_f)/a_f))", degree=1,
                 rho_d = 0.975,
                 rho_u = 1.400,
                 x_f = 0.2,
                 a_f = 1)

# Other parameters
gamma = 1.4
p_0 = 100000

c_square = gamma * p_0 / rho
w_square = 1
lamda = c_square / w_square
#define problem
a = (inner(grad(u), grad(v)) +f*u*v )*dx
m = -lamda*u*v*dx

#assemble stiffness matrix
A = PETScMatrix()
assemble(a, tensor=A)
M = PETScMatrix()
assemble(m, tensor=M)
bc.apply(A)          # apply the boundary conditions



#create eigensolver
eigensolver = SLEPcEigenSolver(A,M)
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
    