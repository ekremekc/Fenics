from dolfin import *

# Set up some simple variational form for illustration:
mesh = UnitIntervalMesh(100)
V = FunctionSpace(mesh,"CG",1)
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u),grad(v))*dx + u*v*dx

# Must assemble form to a PETScMatrix:
A = PETScMatrix()
assemble(a,tensor=A)

# Create eigenvalue solver:
eigenSolver = SLEPcEigenSolver(A)

# Configure solver to get the smallest eigenvalues first:
eigenSolver.parameters["spectrum"]="smallest magnitude"

# Solve for the N smallest eigenvalues:
N = 5
eigenSolver.solve(N)

from matplotlib import pyplot as plt
for i in range(0,N):

    # The smallest eigenpair is the first (0-th) one:
    r, c, rx, cx = eigenSolver.get_eigenpair(i)
    
    # Turn the eigenvector into a Function:
    rx_func = Function(V)
    rx_func.vector()[:] = rx

    # Function can be plotted as usual:
    plot(rx_func)

plt.autoscale()
plt.show()