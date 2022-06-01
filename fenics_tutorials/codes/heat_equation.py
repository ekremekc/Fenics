# import dolfin as dolfin
# import numpy as np

# mesh = dolf.Rectangle(dolf.Point(0,0),dolf.Point(0,0))

#! /usr/bin/env python3
#
"""
from fenics import *

def heat_steady_01 ( ):

#*****************************************************************************80
#
## heat_steady_01, 2D steady heat equation on a rectangle, constant K.
#
#  Discussion:
#
#    Del K Del U = 0 in Omega
#    U = 10 on dOmegaTop
#    U = 100 on dOmega - dOmegaTop
#
#    Omega = rectangle with corners (0.0, 0.0 ) and ( 5.0, 1.0 ).
#    K = 1 (thermal diffusivity)
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    19 October 2018
#
#  Author:
#
#    John Burkardt
#
  import matplotlib.pyplot as plt
#
#  MESH:
#    The region is a rectangle with corners (0.0, 0.0 ) and ( 5.0, 1.0 ).
#    The mesh uses 50 horizontal and 10 vertical divisions.
#
  sw = Point ( 0.0, 0.0 )
  ne = Point ( 5.0, 1.0 )
  mesh = RectangleMesh ( sw, ne, 50, 10 )
#
#  FUNCTION SPACE:
#    piecewise linear Lagrange functions.
#
  V = FunctionSpace ( mesh, "Lagrange", 1 )
#
#  BOUNDARY CONDITIONS:
#    BC_TOP: Along the top, we impose U=10.
#    BC_SIDE: Along the bottom and sides, U=100.
#    Combine these conditions as "BC".
#
  y_top = 1.0
  u_top = 10.0
  def on_top ( x, on_boundary ):
    return ( y_top - DOLFIN_EPS <= x[1] )
  bc_top = DirichletBC ( V, u_top, on_top )

  u_side = 100.0
  def on_side ( x, on_boundary ):
    return ( on_boundary and x[1] < y_top - DOLFIN_EPS )
  bc_side = DirichletBC ( V, u_side, on_side )

  bc = [ bc_top, bc_side ]
#
#  TRIAL and TEST FUNCTIONS:
#
  u = TrialFunction ( V )
  v = TestFunction ( V )
#
#  BILINEAR and LINEAR FORMS:
#
  k = Constant ( 1.0 )
  Auv = k * inner ( grad ( u ), grad ( v ) ) * dx

  f = Constant ( 0.0 )
  Lv = f * v * dx
#
#  SOLVE:
#    Solve the variational problem a(u,v)=l(v) with boundary conditions.
#
  w = Function ( V )
  solve ( Auv == Lv, w, bc )
#feni
#  Plot the solution W.
#
  plot( w, title = 'heat_steady_01' )
  filename = 'heat_steady_01.png'
  plt.savefig ( filename )
  print ( "Saving graphics in file '%s'" % ( filename ) )
  plt.close ( )
#
#  Terminate.
#
  return

def heat_steady_01_test ( ):

#*****************************************************************************80
#
## heat_steady_01_test tests heat_steady_01.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    21 October 2018
#
#  Author:
#
#    John Burkardt
#
  import time

  print ( time.ctime ( time.time() ) )
#
#  Report level = only warnings or higher.
#
  level = 30
  set_log_level ( level )

  print ( '' )
  print ( 'heat_steady_01_test:' )
  print ( '  FENICS/Python version' )
  print ( '  2D steady heat equation in a rectangle.' )

  heat_steady_01 ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'heat_steady_01_test:' )
  print ( '  Normal end of execution.' )
  print ( '' )
  print ( time.ctime ( time.time() ) )
  return

if ( __name__ == '__main__' ):

  heat_steady_01_test ( )
  
"""
from dolfin import *
import numpy as np 
import matplotlib.pyplot as plt
mesh = UnitSquareMesh(16,8)
V = FunctionSpace(mesh,'CG',1)

C1 = Function(V)

class InitialCondition(UserExpression):
    def eval_cell(self, value, x, ufc_cell):
        if x[0] <= 0.5:
            value[0] = 1.0
        else:
            value[0] = 0.5

C1.interpolate(InitialCondition())
p = plot(C1)
# c = plt.plot(interpolate(C1,V))
plt.colorbar(p)
plt.show()










