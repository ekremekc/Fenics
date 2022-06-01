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
    mu = 0.0445
    mu = 0.01
    rho = 1.
    resolution = 250
    outerGeometry = {"xFrontLength": 6.0, \
                     "xBackLength": 16.0, \
                     "yHalfLength":  6.0}
    joukowskyParameters = [-0.0881949142413, -0.234053642152, -0.329185036682]
    joukowskyParameters = [-0.1, -0.05, -0.2]
    area0 = 1.25
    sLim = -0.1
    #freq0 = 0.2*v_in/math.sqrt(area0)
    #rCircle = 1.
    
    textFileName = "results/joukowsky_op1_"+str(resolution)+".dat"
    
    flowFilep = dolf.File("solution/joukowsky_op1_"+str(resolution)+"/pressure.pvd")
    flowFileu = dolf.File("solution/joukowsky_op1_"+str(resolution)+"/velocity.pvd")
    
    outputString = "\n\nResults Joukowsky Optimisation:"
    outputString += "\nResolution: "+str(resolution)
    
    outputString += "\n\ncycleNum \t L2Der \t  drag \t p0 \t p1 \t p2"
    
    outputFile = open(textFileName,'a')
    outputFile.write(outputString)
    outputFile.close()
    
    
    L2tol = 1E-4
    L2Der = 1.+L2tol
    stepSize = 0.01
    cycleNum = -1
    # Reduce drag
    # joukowskyParameters = [-0.0928977508606, -0.000173478857969, -0.00152121082928]
    while(L2Der > L2tol):
        cycleNum += 1
        
        flow = FlowAroundAirfoil(JoukowskyAirfoil(resolution, outerGeometry, joukowskyParameters), \
                                 v_in, mu, rho)
        flow.saveSolution([flowFileu, flowFilep], cycleNum)
        
        area = flow.airfoil.area
        areaDer = flow.airfoil.areaDerivative()
        
        drag = flow.dragForceGauss((1,0))
        dragDer = flow.forceDerivative((1,0))
        
        delta = methods.constrainedStep(dragDer, areaDer, area-area0, stepSize, maximise=False)
        
        
        L2Der = 0.
        for i in range(len(joukowskyParameters)):
            L2Der += pow(delta[i],2)
        L2Der = math.sqrt(L2Der)
        
        outputString = "\n"+str(cycleNum)+" "+str(L2Der)+" "+str(drag)+" "
        outputString += str(joukowskyParameters[0])+" "+str(joukowskyParameters[1])+" "+str(joukowskyParameters[2])
        outputFile = open(textFileName,'a')
        outputFile.write(outputString)
        outputFile.close()
        
        # Take the step
        for i in range(len(joukowskyParameters)):
            joukowskyParameters[i] += delta[i]
    
    # Max Lift-Drag
    #joukowskyParameters = [-0.0897421861573, -0.180264504885, -0.247031555091]
    outputString = "\n\ncycleNum \t L2Der \t LD \t p0 \t p1 \t p2"
    
    outputFile = open(textFileName,'a')
    outputFile.write(outputString)
    outputFile.close()
    
    L2Der = 1.+L2tol
    while(L2Der > L2tol):
        cycleNum += 1
        
        flow = FlowAroundAirfoil(JoukowskyAirfoil(resolution, outerGeometry, joukowskyParameters), \
                                 v_in, mu, rho)
        flow.saveSolution([flowFileu, flowFilep], cycleNum)
        
        area = flow.airfoil.area
        areaDer = flow.airfoil.areaDerivative()
        
        LD = flow.costLD()
        LDDer = flow.costLDDerivative()
        
        delta = methods.constrainedStep(LDDer, areaDer, area-area0, stepSize)
        
        
        L2Der = 0.
        for i in range(len(joukowskyParameters)):
            L2Der += pow(delta[i],2)
        L2Der = math.sqrt(L2Der)
        
        outputString = "\n"+str(cycleNum)+" "+str(L2Der)+" "+str(LD)+" "
        outputString += str(joukowskyParameters[0])+" "+str(joukowskyParameters[1])+" "+str(joukowskyParameters[2])
        outputFile = open(textFileName,'a')
        outputFile.write(outputString)
        outputFile.close()
        
        # Take the step
        for i in range(len(joukowskyParameters)):
            joukowskyParameters[i] += delta[i]
    
    # Stabilising the flow
    # Changing only the angle
    # This keeps the area constant as well
    #joukowskyParameters = [-0.0892603409084, -0.196574229289, -0.188263051184]
    outputString = "\n\ncycleNum \t L2Der \t sReal \t p0 \t p1 \t p2"
    
    outputFile = open(textFileName,'a')
    outputFile.write(outputString)
    outputFile.close()
    
    L2Der = 1.+L2tol
    while(L2Der > L2tol):
        cycleNum += 1
        
        flow = FlowAroundAirfoil(JoukowskyAirfoil(resolution, outerGeometry, joukowskyParameters), \
                                 v_in, mu, rho)
        flow.saveSolution([flowFileu, flowFilep], cycleNum)
        
        sReal = flow.getMaxEigenvalue()[0][0]
        sRealDer = flow.eigenvalueDerivative()[0]
        
        delta = [0. for x in joukowskyParameters]
        
        # Changing the angle
        delta[2] = -(sReal-sLim)/sRealDer[2]
        
        
        L2Der = 0.
        for i in range(len(joukowskyParameters)):
            L2Der += pow(delta[i],2)
        L2Der = math.sqrt(L2Der)
        
        outputString = "\n"+str(cycleNum)+" "+str(L2Der)+" "+str(sReal)+" "
        outputString += str(joukowskyParameters[0])+" "+str(joukowskyParameters[1])+" "+str(joukowskyParameters[2])
        outputFile = open(textFileName,'a')
        outputFile.write(outputString)
        outputFile.close()
        
        # Take the step
        for i in range(len(joukowskyParameters)):
            joukowskyParameters[i] += delta[i]
    
    # Maximising Lift/Drag
    # while keeping the area and sReal constant
    outputString = "\n\ncycleNum \t L2Der \t sReal \t L/D \t p0 \t p1 \t p2"
    
    outputFile = open(textFileName,'a')
    outputFile.write(outputString)
    outputFile.close()
    
    L2Der = 1.+L2tol
    while(L2Der > L2tol):
        cycleNum += 1
        
        flow = FlowAroundAirfoil(JoukowskyAirfoil(resolution, outerGeometry, joukowskyParameters), \
                                 v_in, mu, rho)
        flow.saveSolution([flowFileu, flowFilep], cycleNum)
        
        sReal = flow.getMaxEigenvalue()[0][0]
        sRealDer = flow.eigenvalueDerivative()[0]
        
        area = flow.airfoil.area
        areaDer = flow.airfoil.areaDerivative()
        
        LD = flow.costLD()
        LDDer = flow.costLDDerivative()
        
        delta = methods.twoConstrainedStep(LDDer, areaDer, area-area0, sRealDer, sReal-sLim, stepSize)
        
        L2Der = 0.
        for i in range(len(joukowskyParameters)):
            L2Der += pow(delta[i],2)
        L2Der = math.sqrt(L2Der)
        
        outputString = "\n"+str(cycleNum)+" "+str(L2Der)+" "+str(sReal)+" "+str(LD)+" "
        outputString += str(joukowskyParameters[0])+" "+str(joukowskyParameters[1])+" "+str(joukowskyParameters[2])
        outputFile = open(textFileName,'a')
        outputFile.write(outputString)
        outputFile.close()
        
        # Take the step
        for i in range(len(joukowskyParameters)):
            joukowskyParameters[i] += delta[i]
    
    
    
    
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
    
    
    """
    while(L2Der > L2tol):
        cycleNum += 1
        
        flow = FlowAroundAirfoil(JoukowskyAirfoil(resolution, outerGeometry, joukowskyParameters), \
                                 v_in, mu, rho)
        flow.saveSolution([flowFileu, flowFilep], cycleNum)
        
        area = flow.airfoil.area
        areaDer = flow.airfoil.areaDerivative()
        
        sReal, sImag = flow.getMaxEigenvalue(nonZeroFreq=True)[0]
        sRealDer, sImagDer = flow.eigenvalueDerivative()
        
        if(sImag < 0):
            sImag *= -1
            sImagDer = [-x for x in sImagDer]
        print(areaDer)
        print(sImagDer)
        
        # preserve area and frequency
        delta = methods.twoConstrainedStep(sRealDer, areaDer, area-area0, sImagDer, sImag-freq0, stepSize)
        
        
        L2Der = 0.
        for i in range(len(joukowskyParameters)):
            L2Der += pow(delta[i],2)
        L2Der = math.sqrt(L2Der)
        
        outputString = "\n"+str(cycleNum)+" "+str(L2Der)+" "+str(sReal)+" "+str(sImag)+" "
        outputString += str(joukowskyParameters[0])+" "+str(joukowskyParameters[1])+" "+str(joukowskyParameters[2])
        outputFile = open(textFileName,'a')
        outputFile.write(outputString)
        outputFile.close()
        
        # Take the step
        for i in range(len(joukowskyParameters)):
            joukowskyParameters[i] += delta[i]
    """
    