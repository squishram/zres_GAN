"""
This code takes as input a 'real' and 'fake' image from a GAN
And calculates the difference between them as a pixel sum
It can be used to give a rough comparison of quality between different parameters
"""

import os
from pathlib import Path
from skimage import io
import numpy as np
# from iso_fgan_wdisc_newdataloader import something


def sum_absolute_differences(real_img, fake_img):

    images = []

    for image in (real_img, fake_img):
        image = os.path.join(path_data, image)
        # import image (scikit-image.io imports tiffs as np.array(z, x, y))
        image = io.imread(image)
        # now: img.shape = (1, z, x, y)
        # choose datatype using numpy
        image = np.asarray(image, dtype=np.float32)
        # normalise to range (0, 1)
        image = image / np.max(image)
        # now: imgs.shape = (z, x, y)
        image = np.swapaxes(image, -2, -1)
        # now: imgs.shape = (z, y, x)
        images.append(image)

    # numpy array for faster calculations
    images = np.array(images)
    sum_abs_diff = 0
    pixel_count = 0
    for i in range(images.shape[1]):
        for j in range(images.shape[2]):
            for k in range(images.shape[3]):
                # difference between the two images as sum of pixel differences
                sum_abs_diff += np.absolute(images[0, i, j, k] - images[1, i, j, k])
                pixel_count += 1

    # return the mean pixel difference
    return sum_abs_diff / pixel_count


# folder containing images
# "20220311_blackmanharris"
# "20220315_hann"
# "20220315_hamming"
# "20220314_nowindow"
dir = "20220315_hamming"
# fake image
fake_img = "generated_images_epoch10.tif"
# real image
real_img = "hires_img.tif"

# path to data
path_data = os.path.join(os.getcwd(), Path("images/sims/microtubules/generated/"), Path(dir))
# print name of directory from which images were taken, alongside mean pixel difference
print(dir, sum_absolute_differences(real_img, fake_img))
