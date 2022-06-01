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
    
    textFileName = "results/joukowsky_eigenvalues_"+str(resolution)+".dat"
    
    outputString = "\n\nResults Joukowsky Eigenvalue search:"
    outputString += "\nResolution: "+str(resolution)
    
    outputString += "\np0: "+str(joukowskyParameters[0])
    outputString += "\np1: "+str(joukowskyParameters[1])
    
    outputString += "\nList of eigenvalues:"
    
    outputFile = open(textFileName,'a')
    outputFile.write(outputString)
    outputFile.close()
    
    maxSIndex = float('-inf')
    maxSList = []
    angleArray = np.linspace(-0.*math.pi/180., -90.*math.pi/180., 37)
    for angle in angleArray:
        joukowskyParameters[2] = angle
        flow = FlowAroundAirfoil(JoukowskyAirfoil(resolution, outerGeometry, joukowskyParameters), \
                                 v_in, mu, rho)
        
        (sReal, sImag), sIndex = flow.getMaxEigenvalue(nEigenvalues=100)
        if(sIndex > maxSIndex):
            maxSIndex = sIndex
        maxSList.append([angle, sReal, sImag])
        
        eigenvalueList = flow.eigenvalues
        
        outputString = "\n\n======================================"
        outputString += "\nangle: "+str(angle)+" rad"
        outputString += "\nangle: "+str(-180.*angle/math.pi)+" deg"
        outputString += "\nsReal sImag"
        
        for (sr, si) in eigenvalueList:
            outputString += "\n"+str(sr)+" "+str(si)
        
        outputFile = open(textFileName,'a')
        outputFile.write(outputString)
        outputFile.close()
    
    print("maxSIndex = %i"%(maxSIndex))
    
    outputString = "\n\n======================================"
    outputString +=  "\n======================================"
    outputString += "\nMax eigenvalues:"
    outputString += "\nangle(rad) angle(deg) sReal sImag"
    for (angle, sReal, sImag) in maxSList:
        outputString += "\n"+str(angle)+" "+str(-180.*angle/math.pi)+" "+str(sReal)+" "+str(sImag)
    
    outputFile = open(textFileName,'a')
    outputFile.write(outputString)
    outputFile.close()
        
    
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
    