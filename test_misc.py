import numpy as np
import tifffile
from datetime import date
import os
from pathlib import Path
import math
import torch
import torch.nn.functional as tf


def gaussian_kernel(sigma: float, sigmas: float = 3.0) -> torch.Tensor:
    """
    Make a normalized 1D Gaussian kernel
    """

    radius = math.ceil(sigma * sigmas)
    xs = torch.tensor(range(-radius, radius + 1), dtype=torch.float)
    kernel = torch.exp(-(xs**2) / (2 * sigma**2))

    return kernel / sum(kernel)


def conv_1D_z_axis(
    data: torch.Tensor, kernel: torch.Tensor, stride: int, pad: str = "zeros"
) -> torch.Tensor:

    """
    Perform a 1-D convolution along the z-axis to downsample along that dimension
    """

    # make sure calculations can be done on the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert len(data.size()) == 5, "Data must have dimensions [batch, channels, z, y, x]"
    assert len(kernel.size()) == 1, "Kernel must be 0D"
    assert len(kernel) % 2 == 1, "Kernel must be have odd side length"

    #
    radius = int(len(kernel - 1) / 2)
    # kernel is [z], we need [channels out, channels in, z, y, x] = [1, 1, z, 1, 1]
    kernel = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(4).to(device)

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
        print("WARNING: you are not using any padding!")
        assert False, f"{pad} is not a valid padding setting"

    # perform convolution
    return tf.conv3d(data, kernel, bias=None, stride=(stride, 1, 1), padding=padding)


if __name__ == "__main__":

    print(np.array([2, 4, 6]) / np.array([1, 2, 3]))

    # sigma for real (lores) and isomorphic (hires) data
    sig_lores = (zres_lo / size_pix_nm) / (2 * sqrt(2 * log(2)))
    sig_hires = (zres_hi / size_pix_nm) / (2 * sqrt(2 * log(2)))
    # sig_extra is derived from the formula of convolving 2 gaussians, where it is defined as:
    # gaussian(sigma=sig_hires) *convolve* gaussian(sigma=sig_extra) = gaussian(sigma=sig_lores)
    sig_extra = sqrt(sig_lores**2 - sig_hires**2)

    stride_downsampler = 3

    path_hires = os.path.join(
        os.getcwd(), Path("images/sims/microtubules/hires_v2/mtubs_sim_1_hires.tif")
    )
    hires_img = tifffile.imread(path_hires)
    conv_1D_z_axis(gaussian_kernel(sig_extra, 6.0), stride_downsampler, "zeros")
