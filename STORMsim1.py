#test program to produce som STORM output, converted from MATLAB
import math
import random as r
import numpy as np
import imageio
import simdata_functions as f
import matplotlib.pyplot as plt


#Frame details
# scale factor
scaleFact = 10
# number of frames to simulate
numFrames = 256*1
# frame size No. of X pixels
pixelsX = 32*5
# frame size No. of Y pixels
pixelsY = 32*5
# size of a pixel in nm
pixSize = 100
# number of photoelectrons per ADC level
cameraADC = 6.5
# EMCCD multiplier gain
cameraGain = 150*1.0/3.0
sizePix = 7
sigmaPix = 1.6

noisepic = f.NoisePic(pixelsY, pixelsX, numFrames)
print("Done noise background")

structA = np.zeros((pixelsX*scaleFact, pixelsY*scaleFact))
#structA = f.two_lines(pixelsX, pixelsY)
#print("Done Generating line structure")

# This one makes a converging 'V'
structA = f.v_img(structA)

ONtime = 5
OFFtime = 50*100
molint = 1000

AsizeX = np.size(structA, 1)
AsizeY = np.size(structA, 0)
nummols = 0
simpic = np.zeros((pixelsY,pixelsX,numFrames))

for ycount in range(0, AsizeY):
    for xcount in range (0, AsizeX):
        if structA[ycount, xcount]==0:
            continue
        
        OnOffFlag = round(r.random()*ONtime/(OFFtime+ONtime))
        moltraj = np.zeros((numFrames))
        for fcount in range(0,numFrames):
            if OnOffFlag == 1 and r.random()<(1/ONtime):
                OnOffFlag = 0
            if OnOffFlag == 0 and r.random()<(1/OFFtime):
                OnOffFlag = 1
            
            moltraj[fcount] = OnOffFlag
        
        for fcount in range(0,numFrames):
            if moltraj[fcount] == 0:
                continue
            
            offsetX = (xcount/10) - math.floor(xcount/10)
            offsetY = (ycount/10) - math.floor(ycount/10)
            patch = f.GaussSquare(sigmaPix,sizePix,offsetY,offsetX)
            startX = math.floor(xcount/10)
            startY = math.floor(ycount/10)
            for icount in range(0, sizePix):
                for jcount in range(0, sizePix):
                    rawval = simpic[icount + startY, jcount + startX, fcount] + (molint*patch[icount, jcount])
        
        
        nummols += 1
        if nummols % 10 == 0:
            print("Num Molecules = ", nummols) 

# gaussian because poisson will take forever
print("adding noise")
for ycount in range(0, pixelsY):
    for xcount in range (0, pixelsX):
        for fcount in range(0, numFrames):
            rawval = simpic[ycount, xcount, fcount]
            if rawval == 0:
                continue
            newval = r.gauss(rawval, math.sqrt(rawval))
            simpic[ycount, xcount, fcount] = newval
            



print("Writing tiff")
simpic = simpic*cameraGain/cameraADC
simpic = simpic + noisepic
simpic = np.rot90(simpic, 1, [0,2])
simpic = np.rot90(simpic, 1, [1,2])
imageio.volwrite("Data.tif", simpic.astype(np.int32))
print("Done")
