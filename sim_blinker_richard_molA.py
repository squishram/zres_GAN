import numpy as np
from random import random
from scipy.stats import norm

# def molA(pixelsX, pixelsY, numFrames, factorA=10, structA=0):

#  add molecules from structure type A
factorA = 10
structA = 0
pixelsX = 64
pixelsY = 64
numFrames = 256*factorA
# is data 3d? (>0 =yes):
elip3d = 0
# 3D calibration data for ThunderSTORM:
zcalib = [[1E-5, 1.1466, 100], [1E-5, 1.1466,  -100]]
# mean ON time [frames] (non integer):
ONtime = 5*factorA
# mean OFF time [frames] (non integer):
# (<0 values in structA override this value)
OFFtime = 50*1
# Molecular brightness [photons/ frame]:
molInt = 2500*1/2.5
# mean time to photobleach  or long dark state [frames] (non integer):
bleachtime = 256*200000000
# how many time steps to devide each frame by when calculating ON OFF
# trajectory:
tempresol = 10
# FWHM in pixels
fwhm = 2.7*1
# 0 for initialy off, 1 for initialy on, 2 for initialy at equilibrium:
startON = 2

if structA == 0:
    structA = np.zeros([10*(pixelsY - 9), 10*(pixelsX - 9)])
    structA[(5*pixelsY) - 55, (5*pixelsX) - 55] = 1


simpic = np.zeros(pixelsY + 0, pixelsX + 0, numFrames)
frametraject = np.zeros(1, numFrames)

# Elliptical Gaussian:
squ = lambda n: np.power(n, 2)
eg2 = lambda z, nx: z[1]*np.exp(-(((squ(nx - z[2])/(2*z[4]*z[4])) + (((squ(np.rot90(nx, 3) - z[3]))/(2*z[5]*z[5])))))) + z[6]


# Begin sum over cromophore grid:
r = np.zeros([7, 7])
molNum = 0
for icount in range(len(structA)):
    for jcount in range(len(structA[0])):
        # if no molecule present, skip to next location in structure:
        if structA[icount, jcount] == 0:
            continue

        # if negative value in structA scale emitter density by value:
        # if (structA(icount,jcount)<0)
        #    OFFtime=abs(structA(icount,jcount));
        # end

        molNum += 1
        molONOFF = startON

        # random initial state weighted by on off times:
        if startON == 2:
            if random() < (ONtime/(ONtime + OFFtime)):
                molONOFF = 1
            else:
                molONOFF = 0

        molBleached = 0
        moltraject = np.zeros([1, numFrames*tempresol])

        # simulate one on off traject at tempresol * frame resolution:
        for kcount in range(1, numFrames*tempresol + 1):
            if molBleached == 1:
                continue

            if molONOFF == 1 & (random() < (1 / (ONtime*tempresol))):
                molONOFF = 0
            elif molONOFF == 0 & (random() < (1/(OFFtime*tempresol))):
                molONOFF = 1

            moltraject(kcount) = molONOFF
            if random() < (1 / bleachtime):
                molBleached = 1

        # rebin to frame size:
        frametraject[molNum, :] = np.zeros([1, numFrames])
        for kcount in range(1, numFrames + 1):
            frametraject(molNum, kcount) = sum(moltraject(((kcount - 1)*tempresol) + range(1, kcount*tempresol)))/10

        # calculate PSF for position:
        if elip3d == 0:
            for lcount in range(-3, 4):
                for mcount in range(-3, 4):
                    r[lcount + 3, mcount + 3] = np.sqrt((squ(((icount%tempresol)/tempresol) - lcount - 0.5)) + (squ((jcount%tempresol)/tempresol) - mcount - 0.5))
            nx = np.meshgrid(range(1, 8), range(1, 8))
        else:
            # calculate larger PSF for ellipsoid position:
            for lcount in range(-5, 6):
                for mcount in range(-5, 6):
                    r[lcount + 5, mcount + 5] = np.sqrt((squ(((icount%tempresol)/tempresol) - lcount - 0.5)) + (squ(((jcount%tempresol)/tempresol) - mcount - 0.5)))
                nx = np.meshgrid(range(1, 12),  range(1, 12))

        if elip3d == 0:
            points = norm.pdf(r, 0, fwhm*0.5 / np.sqrt(np.log(4)))
        else:
            z[1] = 1
            z[2] = ((jcount % tempresol)/tempresol) + 5.5
            z[3] = ((icount % tempresol)/tempresol) + 5.5
            z[4] = ((squ(structA[icount, jcount] - zcalib(1, 3)))*zcalib(1, 1)) + zcalib(1, 2)
            z[5] = ((squ(structA(icount,jcount) - zcalib(2, 3)))*zcalib(2, 1)) + zcalib(2, 2)
            z[6] = 0
            points = eg2(z, nx)

        points = points/sum(sum(points))

        # for neg Astructure(variable density) single label for intensity.
        # otherwise, structA = num labels
        # if (structA(icount,jcount) < 1)
        numlabels = 1
        # else
        #    numlabels=structA(icount,jcount);
        # end

        # add photon intensity-relevent pixel PSF to cumulative image
        # for each frame molecule is ON
        for kcount in range(1, numFrames + 1):
            if (frametraject[molNum, kcount] > 0) & elip3d == 0:
                simpic(np.ceil((icount + 0.0001)/10) + range(2, np.ceil((icount + 0.0001)/10) + 9, np.ceil((jcount + 0.0001)/10) +
                    range(2, np.ceil((jcount + 0.0001)/10) + 8, kcount) = simpic(ceil((icount + 0.0001)/10)...
                    + 2:ceil((icount + 0.0001)/10) + 8, ...
                    ceil((jcount + 0.0001)/10)...
                    + 2:ceil((jcount + 0.0001)/10) + 8, kcount)...
                    + (molInt*numlabels*frametraject(molNum, kcount)*points)
            elseif ((frametraject(molNum, kcount) > 0) && elip3d > 0)
                simpic(ceil((icount + 0.0001)/10)...
                    + 0:ceil((icount + 0.0001)/10) + 10,...
                    ceil((jcount + 0.0001)/10)...
                    + 0:ceil((jcount + 0.0001)/10) + 10, kcount)...
                    = simpic(ceil((icount + 0.0001)/10)...
                    + 0:ceil((icount + 0.0001)/10) + 10,...
                    ceil((jcount + 0.0001)/10) +...
                    0:ceil((jcount + 0.0001)/10) + 10, kcount)...
                    + (molInt*frametraject(molNum, kcount)*points);
            end
        end

    waitbar(icount/size(structA, 1), bar, 'Generating structure A...');

close(bar)
disp 'Structure A: complete'

# Add  Poisson/shot noise:
# for icount = 1:numFrames         
#        simpic(:,:,icount)=poissrnd(simpic(:,:,icount));
#        if (rem(100*icount/numFrames,1)==0)
#            100*(icount/numFrames)
#        end
# end
