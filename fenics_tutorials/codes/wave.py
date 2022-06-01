from fenics import *

#parameters

c=1
T = T = 0.0004
num_steps = 200
dt = T/num_steps

# Generate simple mesh
nx = 40

mesh = UnitIntervalMesh(nx)
   
# Defining FE function space

V = FunctionSpace(mesh, 'Lagrange', 1)

def boundary(x, on_boundary):
    return on_boundary

# Define boundary condition
bc = DirichletBC(V, 0, boundary)

# Define initial value

p1= interpolate(Constant(0.0), V)
p0= interpolate(Constant(0.0), V)

# Defining trial and test functions

p = TrialFunction(V)
v = TestFunction(V)

# Define variational problem

# F= p*v*dx + dt*dt*c**2*dot(grad(u)*grad(v))*dx - p_n*v*dx

a = p*v*dx - dt*dt*c*c*inner(grad(p), grad(v))*dx

L= 2*p1*v*dx-p0*v*dx

# Compute solution
p = Function(V)
t = 0
for n in range(num_steps):
    # Update current time
    t += dt
    # p_D.t = t
    # Compute solution
    solve(a == L, p, bc)
    plot(p)
    # Compute error at vertices
    # u_e = interpolate(u_D, V)
    # error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
    # print('t = %.2f: error = %.3g' % (t, error))
    # Update previous solution
    p0.assign(p1)
    p1.assign(p)
