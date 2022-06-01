from dolfin import *
import matplotlib.pyplot as plt

N=40
x_left  =  0.0
x_right = +1.0
mesh = IntervalMesh ( N, x_left, x_right )

boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

V = FunctionSpace ( mesh, "Lagrange", 1 )


u = TrialFunction ( V )
v = TestFunction ( V )

#ROBIN BC DATA
tol = 1e-14

class BoundaryX0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0, tol)

class BoundaryX1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1, tol)

bx0 = BoundaryX0()
bx1 = BoundaryX1()
bx0.mark(boundary_markers, 0)
bx1.mark(boundary_markers, 1)

k0 = 10
k1 = 10

s = 0

boundary_conditions = {0: {'Robin': (k0,s)}, # x = 0
                       1: {'Robin': (k1,s)}}  # x = 1 

integrals_R_a = []
integrals_R_L = []
for i in boundary_conditions:
    if 'Robin' in boundary_conditions[i]:
        r, s = boundary_conditions[i]['Robin']
        integrals_R_a.append(r*u*v*ds(i))
        integrals_R_L.append(r*s*v*ds(i))

f = Expression('0.0',degree=2)

# Define density function

rho = Expression("rho_d+0.5*(rho_d-rho_u)*(1+tanh((x[0]-x_f)/a_f))", degree=1,
                 rho_d = 1.400,
                 rho_u = 1.400,
                 x_f = 0.5,
                 a_f = 0.1)

# Other parameters
gamma = 1.4
p_0 = 100000

c_square = gamma * p_0 / rho
w_square = 1
lamda = c_square / w_square
#define problem
a = (inner(grad(u), grad(v)) +f*u*v )*dx + sum(integrals_R_a)
m = -lamda*u*v*dx + sum(integrals_R_L)

#assemble stiffness matrix
A = PETScMatrix()
assemble(a, tensor=A)
M = PETScMatrix()
assemble(m, tensor=M)

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
    
plt.autoscale()
    