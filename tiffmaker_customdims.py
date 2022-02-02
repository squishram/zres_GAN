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
path_data = os.path.join(path_data, Path("noise/projections"))
# make directories if they don't already exist
os.makedirs(path_data, exist_ok=True)
# file name root:
filename = "flat_noise_"


###############
# IMAGE SPECS #
###############

# how many images do you want?
nimg = 10
# bittage of final image - 8, 16, 32, or 64?
img_bit = 32
# size of final image in pixels:
size_img = np.array([96, 64, 32])
# size_img = np.array([96, 48])
# lambda for the poisson noise
lamb = 5

for i in range(nimg):
    # make an image of random points
    noise_img = np.random.poisson(lamb, size_img)

    # normalise all the brightness values
    # then scale them up so that the brightest value is 255:
    noise_img = (noise_img - np.mean(noise_img)) / np.std(noise_img)
    noise_img = (noise_img / np.max(noise_img)) * 255

    # normalise and scale according to z-projection values
    z_projection =

    # tiff writing in python gets the axes wrong
    # rotate the image before writing so it doesn't!
    # noise_img = np.rot90(noise_img, 1, [0, 2])
    # noise_img = np.rot90(noise_img, 1, [1, 2])

    # write to file
    filename_ind = filename + str(i + 1) + ".tif"
    print("Writing to tiff: " + filename_ind)
    file_path = os.path.join(path_data, filename_ind)

    if img_bit == 8:
        imsave(file_path, noise_img.astype(np.uint8))
    elif img_bit == 16:
        imsave(file_path, noise_img.astype(np.uint16))
    elif img_bit == 32:
        imsave(file_path, noise_img.astype(np.uint32))
