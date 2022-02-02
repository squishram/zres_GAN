from pathlib import Path
from PIL import Image
from astropy.nddata import CCDData
# from torch.utils.data import DataLoader
import os
# from datetime import date
# import math
from typing import Union
# import matplotlib.pyplot as plt
import torch
import numpy as np
# from tifffile import imsave
# from iso_fgan_dataloader import Custom_Dataset

# define the 'Number' type as being either an integer or a float
Number = Union[int, float]


def load_cube_suco(dir_data) -> torch.Tensor:
    """
    Function for loading some data, assumes res is in the filename
    """

    # Path('X').glob('Y')
    # = list of all of the image files in dir X with filename structure Y
    files = sorted(Path(dir_data).glob("flat_noise_*.fits"))
    # data will contain a list of all the images
    data = []

    for i in files:
        # print the file name
        print(i)
        # add the image with filename i to the list of images
        # CCDData.read(filename, pixel brightness unit) reads in an image file
        # adu = arbitrary data unit(?)
        # data.append(np.asarray(CCDData.read(i, unit="adu")))
        data.append(np.asarray(Image.open(i)))

    print(len(data))

    # data index is [y, x], i.e. row, col, so add a z dimension and stack
    # Expand and contatenate to make [z, y, x]
    return torch.tensor(
        np.concatenate([np.expand_dims(n, 0) for n in data], 0), dtype=torch.float
    )


def load_cube_test(dir_data) -> torch.Tensor:
    """
    Function for loading some data, assumes res is in the filename
    """

    # Path('X').glob('Y')
    # = list of all of the image files in dir X with filename structure Y
    files = Path(dir_data).glob("flat_noise_*.fits")
    # data will contain a list of all the images
    data = []

    for i in files:
        # print the file name
        print(i)
        # add the image with filename i to the list of images
        # CCDData.read(filename, pixel brightness unit) reads in an image file
        # adu = arbitrary data unit
        # data.append(np.asarray(CCDData.read(i, unit="adu")))
        data.append(np.asarray(Image.open(i)))

    # data index is [y, x], i.e. row, col, so add a z dimension and stack
    # Expand and contatenate to make [z, y, x]
    return torch.tensor(np.dstack(data), dtype=torch.float)


###########
# STORAGE #
###########

# path to data
dir_data = os.path.join(os.getcwd(), "images/sims/")
# path to training samples (low-z-resolution, high-z-resolution)
dir_flatimgs = os.path.join(dir_data, Path("noise/flat_noise"))

##############
# CODE START #
##############

img1 = load_cube_suco(dir_flatimgs)
img2 = load_cube_test(dir_flatimgs)

print(img1.shape)
print(img2.shape)
