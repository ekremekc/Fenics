import dolfin as dolf
from JoukowskyAirfoil import JoukowskyAirfoil
from CircleAirfoil import CircleAirfoil
from GivenAirfoil import GivenAirfoil

from FlowAroundAirfoil import FlowAroundAirfoil
import methods
import math
import numpy as np

import time
t0 = time.clock()

if __name__ == "__main__":
    # Setting the parameters
    v_in = 1.0  # inflow velocity
    mu = 0.01
    rho = 1.
    resolution = 200
    outerGeometry = {"xFrontLength": 6.0, \
                     "xBackLength": 25.0, \
                     "yHalfLength":  6.0}
    joukowskyParameters = [-0.09, -0.16, -0.16108]
    joukowskyParameters[2] = -0.
    
    
    outputString = "\n\nResults Joukowsky Base Change:"
    outputString += "\nResolution: "+str(resolution)
    
    outputString += "\np0: "+str(joukowskyParameters[0])
    outputString += "\np1: "+str(joukowskyParameters[1])
    outputString += "\np2: "+str(joukowskyParameters[2])
    
    
    time_1 = 1.5
    dt = 1./15.
    amp = 20.
    textFileName = "results/joukowsky_base_change_"+str(resolution)+".dat"
    
    flowFilep = dolf.File("solution/joukowsky_base_change_"+str(resolution)+"/pressure.pvd")
    flowFileu = dolf.File("solution/joukowsky_base_change_"+str(resolution)+"/velocity.pvd")
    flowFile = [flowFileu, flowFilep]
    
    
    flow = FlowAroundAirfoil(JoukowskyAirfoil(resolution, outerGeometry, joukowskyParameters), \
                                 v_in, mu, rho)
    for t in np.arange(0, time_1, dt):
        flow.saveSolution(flowFile, t)
    
    u = flow.u
    p = flow.p
    
    (sReal, sImag), sIndex = flow.getMaxEigenvalue(nEigenvalues=120)
    
    outputString += "\neigenvalue: "+str(sReal)+" "+str(sImag)
    
    outputFile = open(textFileName,'a')
    outputFile.write(outputString)
    outputFile.close()
    
    wRealFull, wImagFull = flow.eigenflows[sIndex]
    P1 = dolf.FiniteElement("CG",flow.airfoil.mesh.ufl_cell(),1)  # for pressure
    P2 = dolf.FiniteElement("CG",flow.airfoil.mesh.ufl_cell(),2)  # for a velocity component
    W = dolf.FunctionSpace(flow.airfoil.mesh, dolf.MixedElement([P2,P2,P1]))
    
    uSpace = dolf.FunctionSpace(flow.airfoil.mesh, dolf.MixedElement([P2,P2]))
    pSpace = dolf.FunctionSpace(flow.airfoil.mesh, P1)
    
    wRealFull = dolf.project(wRealFull, W)
    wImagFull = dolf.project(wImagFull, W)
    
    wReal_x,wReal_y,rReal = dolf.split(wRealFull)
    wImag_x,wImag_y,rImag = dolf.split(wImagFull)
    wReal = dolf.as_vector([wReal_x,wReal_y])
    wImag = dolf.as_vector([wImag_x,wImag_y])
    
    
    time_2 = time_1 + 10.
    
    for t in np.arange(time_1+dt, time_2, dt):
        cRe = math.cos(sImag*t)
        cIm = math.sin(sImag*t)
        
        uSave = u + amp*(cRe*wReal - cIm*wImag)
        pSave = p + amp*(cRe*rReal - cIm*rImag)
        
        uSave = dolf.project(uSave,uSpace)
        pSave = dolf.project(pSave,pSpace)
        uSave.rename('velocity', 'uSave')
        pSave.rename('pressure', 'pSave')
        
        flowFileu << (uSave, t)
        flowFilep << (pSave, t)
    
    """
    # Change the angle to 5 deg
    joukowskyParameters[2] = -0.0872664625997
    
    flow = FlowAroundAirfoil(JoukowskyAirfoil(resolution, outerGeometry, joukowskyParameters), \
                                 v_in, mu, rho)
    u = flow.u
    p = flow.p
    
    (sReal, sImag), sIndex = flow.getMaxEigenvalue(nEigenvalues=120)
    
    outputString = "\nangle: "+str(joukowskyParameters[2]) 
    outputString += "\neigenvalue: "+str(sReal)+" "+str(sImag)
    
    outputFile = open(textFileName,'a')
    outputFile.write(outputString)
    outputFile.close()
    
    wRealFull, wImagFull = flow.eigenflows[sIndex]
    P1 = dolf.FiniteElement("CG",flow.airfoil.mesh.ufl_cell(),1)  # for pressure
    P2 = dolf.FiniteElement("CG",flow.airfoil.mesh.ufl_cell(),2)  # for a velocity component
    W = dolf.FunctionSpace(flow.airfoil.mesh, dolf.MixedElement([P2,P2,P1]))
    
    uSpace = dolf.FunctionSpace(flow.airfoil.mesh, dolf.MixedElement([P2,P2]))
    pSpace = dolf.FunctionSpace(flow.airfoil.mesh, P1)
    
    wRealFull = dolf.project(wRealFull, W)
    wImagFull = dolf.project(wImagFull, W)
    
    wReal_x,wReal_y,rReal = dolf.split(wRealFull)
    wImag_x,wImag_y,rImag = dolf.split(wImagFull)
    wReal = dolf.as_vector([wReal_x,wReal_y])
    wImag = dolf.as_vector([wImag_x,wImag_y])
    
    time_3 = time_2 + 30.
    
    for t in np.arange(time_2+dt, time_3, dt):
        cRe = math.cos(sImag*t)*math.exp(sReal*(t-time_2))
        cIm = math.sin(sImag*t)*math.exp(sReal*(t-time_2))
        
        uSave = u + amp*(cRe*wReal - cIm*wImag)
        pSave = p + amp*(cRe*rReal - cIm*rImag)
        
        uSave = dolf.project(uSave,uSpace)
        pSave = dolf.project(pSave,pSpace)
        uSave.rename('velocity', 'uSave')
        pSave.rename('pressure', 'pSave')
        
        flowFileu << (uSave, t)
        flowFilep << (pSave, t)
    """
    
    
    t1 = time.clock()
    dTime = t1-t0
    dHour = int(dTime//3600)
    dTime -= 3600*dHour
    dMin = int(dTime//60)
    dTime -= 60*dMin
    dSec = dTime
    outputString = "\nDone in: "+str(dHour)+":"+str(dMin)+":"+str(dSec)
    
    
    outputFile = open(textFileName,'a')
    outputFile.write(outputString)
    outputFile.close()
    print("Done!")
    