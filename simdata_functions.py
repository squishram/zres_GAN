import numpy as np
import math
import random as r
import imageio

def lineq(p1, p2):
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    c = (p2[1] - (m * p2[0]))
    return m, c


def v_img(structA, rads=np.pi/10, edgSpace=1/7, scaleFact=10):

    # Here the coordinates of the V is determined by rads:
    v = np.zeros([3,2])
    v[0, 1] = (1 - edgSpace)*len(structA[0])
    v[1, 1] = len(structA[0]) - v[0, 1]
    v[2, 1] = len(structA[0]) - v[0, 1]
    v[0, 0] = (1/2)*len(structA)
    v[1, 0] = v[0, 0] + (v[0, 1] - v[1, 1])*np.tan(rads/2)
    v[2, 0] = v[0, 0] - (v[0, 1] - v[1, 1])*np.tan(rads/2)
    
    x = [i[1] for i in v]
    x = np.array(range(round(min(x)), round(max(x))))
    
    # Obtain the coefficients for the 1st order lines that join the points into,# a V shape & define the lines in terms of those coefficients:
    m1, c1 = lineq([v[0, 1], v[0, 0]],[v[1, 1], v[1, 0]])
    m2, c2 = lineq([v[0, 1], v[0, 0]],[v[2, 1], v[2, 0]])
    y1 = m1*x + c1
    y2 = m2*x + c2

    y1 = [round(i) for i in y1]
    y2 = [round(i) for i in y2]
    
    for i in range(len(x)):
        structA[y1[i], x[i]] = 1
        structA[y2[i], x[i]] = 1
        
    return structA



#function for stage 1 generate blank frames with noise floor
def NoisePic(pixelsY, pixelsX, NumFrames, cameraBase=400, cameraSTD=25, backCount=25):
    cameraBase = 2*200*1.0                    # EMCCD base level in ADC
    cameraSTD = 0.5*50.0                      # per pixel standard deviation in base level 
    NumFrames = round(NumFrames)
    backCount = 25.0*1.0/10                  # background light level in photons per frame per pixel

    #create noise background
    noisep = np.ones((pixelsY, pixelsX, NumFrames))*cameraBase
    for icount in range(0, NumFrames):
        for jcount in range(0, pixelsX):
            for kcount in range(0, pixelsY):
                noise = r.gauss(cameraBase, cameraSTD)
                noisep[kcount, jcount, icount] = round(noise)
    return noisep


# function to generate a structure matrix
def two_lines(PixelsY, PixelsX, scaleFact=10):
    strboarder = 10
    linegap = 600/100
    xlower = (PixelsX*5) - (round(linegap*5))
    xupper = (PixelsX*5) + (round(linegap*5))
    Astruct = np.zeros((PixelsY*scaleFact,PixelsX*scaleFact))
    for icount in range(strboarder*scaleFact,(PixelsY - strboarder)*scaleFact):
        Astruct[icount, xlower] = 1
        Astruct[icount, xupper] = 1
    return Astruct


# function to generate a gausian spot
def GaussSquare(SigmaP, SizeP, OffSY, OffSX):
    GPatch = np.zeros((SizeP, SizeP))
    NormC = 0
    for icount in range(0, SizeP):
        for jcount in range(0, SizeP):
            GvalX = OffSX + (SizeP/2) - (jcount + 0.5)
            GvalY = OffSY + (SizeP/2) - (icount + 0.5)
            GvalX = (GvalX**2)/(2*SigmaP*SigmaP)
            GvalY = (GvalY**2)/(2*SigmaP*SigmaP)
            Gval = math.exp(-GvalX)*math.exp(-GvalY)
            GPatch[icount, jcount] = Gval
            NormC = NormC + Gval
    GPatch = GPatch/NormC
    return GPatch