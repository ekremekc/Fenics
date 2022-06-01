from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# def f():
#     # forcing = 'uniform';
#     forcing = 'parabola';
    
#     if forcing=='parabola':
#         f = Expression('(x[0]-0.5)**2', degree = 2)
#     elif forcing=='uniform':
#         f = Expression('1 + 0*x[0]', degree = 1)
#     return f   
         
f = Expression("pow((x[0]-0.5),2)", degree = 2)
# f = Expression('1 + 0*x[0]', degree = 1)

N=40
x_left  =  0.0
x_right = +1.0
mesh = IntervalMesh ( N, x_left, x_right )

V = FunctionSpace ( mesh, "Lagrange", 1 )

u_left = -0.0010
def on_left ( x, on_boundary ):
  return ( x[0] <= x_left + DOLFIN_EPS )
bc_left = DirichletBC ( V, u_left, on_left )

u_right = 0.0010
def on_right ( x, on_boundary ):
  return ( x_right - DOLFIN_EPS <= x[0] )
bc_right = DirichletBC ( V, u_right, on_right )

bc = [ bc_left, bc_right ]
# bc = bc_right

#  Define the trial functions (u) and test functions (v).
#
u = TrialFunction ( V )
v = TestFunction ( V )

a = dot ( grad ( u ), grad ( v ) ) * dx
L = f * v * dx

u = Function ( V )

solve ( a == L, u, bc )

plot(u)

