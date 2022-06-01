
import numpy as np


def shapeTransform(originalShape, finalShape, nFrames=150):
    
    # ShapeList[i][j][k] = frame[i] point[j] coordinate[k]
    shapeList = np.zeros((nFrames, len(originalShape), len(originalShape[0])))
    
    for j in range(len(originalShape)):
        for k in range(len(originalShape[0])):
            tempList = np.linspace(originalShape[j][k], finalShape[j][k], nFrames)
            for i in range(nFrames):
                shapeList[i][j][k] = tempList[i]
    
    return shapeList

def constrainedStep(costDer, constraintDer, constraintVal, stepSize, maximise=True):
    
    c = [stepSize*x for x in costDer]
    if(maximise == False):
        for i in range(len(costDer)):
            c[i] *= -1
    
    fDer = constraintDer
    fVal = constraintVal
    
    lamF = (-fVal-np.dot(c,fDer))/np.dot(fDer,fDer)
    
    delta = np.zeros(len(costDer))
    
    for i in range(len(costDer)):
        delta[i] = c[i] + lamF*fDer[i]
    
    return delta

def twoConstrainedStep(costDer, constraintDer_1, constraintVal_1, constraintDer_2, constraintVal_2, stepSize, maximise=True):
    
    c = [stepSize*x for x in costDer]
    if(maximise == False):
        for i in range(len(costDer)):
            c[i] *= -1
    
    fVal = constraintVal_1
    gVal = constraintVal_2
    
    fDer = constraintDer_1
    gDer = constraintDer_2
    
    denom = np.dot(fDer,fDer)*np.dot(gDer,gDer) - \
            pow(np.dot(fDer,gDer),2)
    
    lamF = -fVal*np.dot(gDer,gDer) + gVal*np.dot(fDer,gDer) + \
           np.dot(gDer,c)*np.dot(fDer,gDer) - np.dot(fDer,c)*np.dot(gDer,gDer)
    lamF = lamF/denom
    
    lamG = fVal*np.dot(fDer,gDer) - gVal*np.dot(fDer,fDer) + \
           np.dot(fDer,c)*np.dot(fDer,gDer) - np.dot(gDer,c)*np.dot(fDer,fDer)
    lamG = lamG/denom
    
    if(denom == 0.):
        lamF = 0.
        lamG = 0.
    
    delta = np.zeros(len(costDer))
    
    for i in range(len(costDer)):
        delta[i] =  c[i] + lamF*fDer[i] + lamG*gDer[i]
    
    return delta


