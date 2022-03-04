from pathlib import Path
import os
from skimage import io
# from datetime import date
import numpy as np
import torch
import math


# full path to the image
dir_path = os.path.join(os.getcwd(), Path("images/sims/microtubules/hires_test/"))
# filename of the image, including extension
img_name = "mtubs_sim_1_hires.tif"
img_path = Path(os.path.join(dir_path, img_name))
image = torch.from_numpy(np.array(io.imread(img_path), dtype="int16"))

print(f"image shape is {image.shape}")

# projections for original data
# we project down two axes as a 1D signal feed is easier to fourier transform
x_projection = image.sum(1).sum(0)
y_projection = image.sum(2).sum(0)
z_projection = image.sum(2).sum(1)

for image in [x_projection, y_projection, z_projection]:

    # this is the side length of the projection
    image_size = image.shape
    image_size = image_size[0]

    # cosine window arguments
    cos_args = (2 * math.pi / image_size) * torch.tensor(range(image_size))

    # generate the sampling window
    sampling_window = torch.zeros(image_size)

    # window coefficients (for BH window, [0.35875, 0.48829, 0.14128, 0.01168])
    coeffs = [0.35875, 0.48829, 0.14128, 0.01168]

    for idx, coefficient in enumerate(coeffs):
        sampling_window += ((-1) ** idx) * coefficient * torch.cos(cos_args * idx)

    print(f"sampling_window is {sampling_window}")

    image = image.to(float)
    image *= sampling_window
