"""
current use of test.misc:
testing Susan's function
"""

import os
from pathlib import Path
from typing import Union
from affirm3d_noGT_functions import Custom_Dataset_Pairs
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as tf
import numpy as np
import math


Number = Union[int, float]


def gaussian_kernel(sigma: float, sigmas: float = 3.0) -> torch.Tensor:
    """
    Make a normalized 1D Gaussian kernel
    """

    radius = math.ceil(sigma * sigmas)
    xs = np.array(range(-radius, radius + 1))
    kernel = np.exp(-(xs**2) / (2 * sigma**2))

    return torch.tensor(kernel / sum(kernel), dtype=torch.float)


def conv_1D_z_axis(
    data: torch.Tensor, kernel: torch.Tensor, stride: int, pad: str = "zeros"
) -> torch.Tensor:

    """
    Perform a 1-D convolution along the z-axis to downsample along that dimension
    """

    assert len(data.size()) == 5, "Data must have dimensions [batch, channels, z, y, x]"
    assert len(kernel.size()) == 1, "Kernel must be 1D"
    assert len(kernel) % 2 == 1, "Kernel must be have odd side length"

    #
    radius = int(len(kernel - 1) / 2)
    # kernel is [z], we need [channels out, channels in, z, y, x] = [1, 1, z, 1, 1]
    kernel = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(4)

    # add z-padding of zeros
    if pad == "zeros":
        padding = (radius, 0, 0)

    # replicate the top and bottom planes of the image, then add them on as padding
    elif pad == "edge":
        padding = (0, 0, 0)
        bottom = data[:, :, 0, :, :].expand(
            data.size(0), data.size(1), radius, data.size(-2), data.size(-1)
        )
        top = data[:, :, -1, :, :].expand(
            data.size(0), data.size(1), radius, data.size(-2), data.size(-1)
        )
        data = torch.cat([bottom, data, top], dim=0)

    # punish users who do not use padding for their insolence
    else:
        padding = (0, 0, 0)
        print("WARNING: you are not using any padding!")
        print(f"{pad} is not a valid padding setting")

    # perform convolution
    return tf.conv3d(data, kernel, bias=None, stride=(stride, 1, 1), padding=padding)


# put it on the gpu!
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# stride of the downsampler kernel in z
kernel_stride_ds = 3
# batch size
batch_size = 5

##################
# WITH REAL DATA #
##################

# # path to data
# path_data = os.path.join(os.getcwd(), Path("images/sims/microtubules/"))
# path_hires = os.path.join(path_data, "hires_test_batch")
# path_lores = os.path.join(path_data, "lores_test_batch")
# # glob of filnames
# filename = "mtubs_sim_*.tif"
#
# # dataset:
# dataset = Custom_Dataset_Pairs(
#     dir_data=path_data,
#     subdirs=(path_lores, path_hires),
#     filename=filename,
#     transform=None,
# )
# # image dataloaders when loading in hires and lores together
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#
# # iterable from the dataloader
# data_iterator = iter(dataloader)
# # pull out a single batch of data
# data_batch = next(data_iterator)
# lores_batch = data_batch[:, 0, :, :, :, :]
# hires_batch = data_batch[:, 1, :, :, :, :]
# # from the batch, pull out a single image each of hires and lores data
# lowimg = lores_batch[0, 0, :, :, :]
# higimg = hires_batch[0, 0, :, :, :]

##################
# WITH FAKE DATA #
##################

# generate batched data of random numbers
hires_batch = torch.rand(5, 1, 96, 96, 96)

# camera pixel size
size_pix_nm = 100.0
# z-resolution (output of generator)
zres_hi = 240.0
# z-resolution (input to generator)
zres_lo = 600.0

# sigma for real (lores) and isomorphic (hires) data
sig_lores = (zres_lo / size_pix_nm) / (2 * math.sqrt(2 * math.log(2)))
sig_hires = (zres_hi / size_pix_nm) / (2 * math.sqrt(2 * math.log(2)))
# this is derived from the formula of convolving 2 gaussians, where sig_extra is defined as:
# gaussian(sigma=sig_lores) *convolve* gaussian(sigma=sig_extra) = gaussian(sigma=sig_extra)
sig_extra = math.sqrt(sig_lores**2 - sig_hires**2)

# generate gaussian kernel for downsampling
kernel = gaussian_kernel(sig_extra, 6.0)
# apply downsampling convolution
downsampled_batch = conv_1D_z_axis(hires_batch, kernel, kernel_stride_ds)

# printouts:
print(f"the hi-res  batch has dims {hires_batch.shape}")
print(f"downsampled batch has dims {downsampled_batch.shape}")
print("Finished")
