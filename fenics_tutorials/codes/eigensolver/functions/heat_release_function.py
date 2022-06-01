from fenics import *
import matplotlib.pyplot as plt

mesh = UnitIntervalMesh(100)
V = FunctionSpace(mesh, 'CG', 1)

u = TrialFunction(V)
v = TestFunction(V)


h = Expression("exp(-pow((x[0]-X_h), 2)/pow(L_h,2))/sqrt(pi)/L_h", degree=1,
                 X_h = 0.25, # position of heat release region (metres)
                 L_h = 0.05) # length of heat release region (metres)

heat = interpolate(h, V)

plot(heat)
plt.xlabel('x')
plt.ylabel('Heat Release')
plt.show()