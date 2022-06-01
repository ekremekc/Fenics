from fenics import *
import matplotlib.pyplot as plt

mesh = UnitIntervalMesh(100)
V = FunctionSpace(mesh, 'CG', 1)

u = TrialFunction(V)
v = TestFunction(V)


w = Expression("exp(-pow((x[0]-X_w), 2)/pow(L_w,2))/sqrt(pi)/L_w", degree=1,
                 X_w = 0.20, # position of heat release region (metres)
                 L_w = 0.05) # length of heat release region (metres)

measurement_f = interpolate(w, V)

# print('Function w is:\n', measurement_f.vector()[:])

plot(measurement_f)
plt.xlabel('x')
plt.ylabel('Distibution of Measurement')
plt.show()