from fenics import *

mesh = UnitIntervalMesh(5)

CG = FiniteElement('CG', mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, CG * CG)

u_r, u_im = TrialFunctions (W)  
v_r, v_im = TestFunctions (W )



