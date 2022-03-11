import numpy as np
import os
from pathlib import Path
from tifffile import imsave

###########
# STORAGE #
###########

# path to data
path_data = os.path.join(os.getcwd(), "images/sims/")
# path to training samples (low-z-resolution, high-z-resolution)
path_data = os.path.join(path_data, Path("noise/cuboidal_noise"))
# make directories if they don't already exist
os.makedirs(path_data, exist_ok=True)
# file name root:
filename = "noise3d_"


###############
# IMAGE SPECS #
###############

# how many images do you want?
nimg = 200
# size of final image in pixels:
size_img = np.array([96, 96, 96])
# size_img = np.array([96, 48])
# lambda for the poisson noise
lamb = 5

for i in range(nimg):
    # make an image of random points
    noise_img = np.random.poisson(lamb, size_img)

    # normalise all the brightness values
    # then scale them up so that the brightest value is 255:
    # noise_img = (noise_img - np.mean(noise_img)) / np.std(noise_img)
    # noise_img = (noise_img / np.max(noise_img)) * 255

    # write to file
    filename_ind = filename + str(i + 1) + ".tif"
    print("Writing to tiff: " + filename_ind)
    file_path = os.path.join(path_data, filename_ind)

    imsave(file_path, noise_img.astype(np.uint16))
