from dolfin import *
import matplotlib.pyplot as plt

# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print( "DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_slepc():
    print ("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

# Define mesh, function space
N=40
x_left  =  0.0
x_right = +1.0
mesh = IntervalMesh ( N, x_left, x_right )
V = FunctionSpace(mesh, "Lagrange", 1)

# #define boundary
# def boundary(x, on_boundary):
#     return on_boundary

# #apply essential boundary conditions
# bc = DirichletBC(V, 0, boundary)

u_left = Constant("-0.0010")
def on_left ( x, on_boundary ):
  return ( x[0] <= x_left + DOLFIN_EPS )
bc_left = DirichletBC ( V, u_left, on_left )

u_right = Constant("0.0010")
def on_right ( x, on_boundary ):
  return ( x_right - DOLFIN_EPS <= x[0] )
bc_right = DirichletBC ( V, u_right, on_right )

bc = [ bc_left, bc_right ]

#define functions
u = TrialFunction(V)
v = TestFunction(V)

Pot = Expression('0.0',degree=2)

#define problem
a = (inner(grad(u), grad(v)) \
     + Pot*u*v)*dx
m = u*v*dx

#assemble stiffness matrix
A = PETScMatrix()
assemble(a, tensor=A)
M = PETScMatrix()
assemble(m, tensor=M)
# bc.apply(A)          # apply the boundary conditions
# bc.apply(M)
bc_left.apply(A)          # apply the boundary conditions
bc_right.apply(M)

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
    plot(u)
plt.autoscale()
