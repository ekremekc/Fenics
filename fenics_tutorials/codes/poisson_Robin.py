from fenics import *


f = Expression("pow((x[0]-0.5),2)", degree = 2)

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


a = dot(grad(u), grad(v))*dx + sum(integrals_R_a)
L = f*v*dx + sum(integrals_R_L)

u = Function ( V )

solve ( a == L, u )

plot(u)
