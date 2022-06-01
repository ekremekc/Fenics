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
def boundary(x, on_boundary):
    return on_boundary

#apply essential boundary conditions
bc = DirichletBC(V, 0, boundary)


#define functions
u = TrialFunction(V)
v = TestFunction(V)

f = Expression('0.0',degree=2)

#define problem
a = (inner(grad(u), grad(v)) +f*u*v )*dx
m = u*v*dx

#assemble stiffness matrix
A = PETScMatrix()
assemble(a, tensor=A)
M = PETScMatrix()
assemble(m, tensor=M)
bc.apply(A)          # apply the boundary conditions
# bc.apply(M)


#create eigensolver
eigensolver = SLEPcEigenSolver(A, M)
eigensolver.parameters['problem_type'] = 'gen_hermitian'
eigensolver.parameters['spectrum'] = 'target real'
eigensolver.parameters['tolerance'] = 1e-6

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
    