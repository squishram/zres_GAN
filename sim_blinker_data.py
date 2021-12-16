"""
Simulation of STORM data for an arbitry structure for two types of molecule
Includes option for astigmatic 3D
Calibration parameters from real system can be used
Currently set for NIC system
Run one section at a time

First section:
- creates arrays from camera info and produces a Gaussian
- includes camera noise packground.
- creates an appropriate size image grid for structure if none already exists
  with a single molecule at center.
 - If you want to change the structure do it after this section
View it with 'imagesc(structA)'

Second grid of different molecules can be % used with Bstructurepic.
Run next section after entering fluorophore parameters for molecule A.
Run next section for molecule B. This also adds the noise. If not using
two types of label just make intensity ' molInt' for B = 0
The 'Export video' section writes a multi image tiff of the results
into the MATLAB directory.

Due to an unvesolved bug the number of structure subpixels per camera
pixel must equal the number of timesteps per frame, I've always kept this
at 10 which gives me 10nm label spacing for my 100nm sized pixels.

the remaining sections generate example structures for resolution testing.
"""

# PACKAGES
import numpy as np
from matplotlib import pyplot as plt
# from numpy.linalg import lstsq
from sim_blinker_richard_functions import *


# frame size [# X pixels]:
pixelsX = 64
# frame size [# Y pixels]:
pixelsY = 64
# Side length of a square pixel [nm]:
pixSize = 100
# SR pixel sizing factor:
SRpix = 10
# Do you need structure B? (1 = yes, 0 = no)
BB = 0

factorA = 1

# Allocate xy size of final stack in SR pixels:
structA = np.zeros([SRpix*(pixelsY - 9), SRpix*(pixelsX - 9)])
structA[(5*pixelsY) - 55, (5*pixelsX) - 55] = 1

# This one makes a converging 'V'
structA = v_img(structA)


# Display your ground truth (to make sure they are how you want them)
plt.imshow(structA, interpolation='nearest')
plt.show()
