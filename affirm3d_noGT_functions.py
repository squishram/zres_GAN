"""
functions for use in affirm3D and networks
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tf


def conv(
    in_channels: int,
    out_channels: int,
    maxpool: int = 0,
    batchnorm: bool = True,
    relu: bool = True,
) -> nn.ModuleList:
    """
    Inputs
        in_channels (int): number of channels in the input
        out_channels (int): number of channels in the output
        maxpool (int): rate of downsampling (adds MaxPool3d layer if != 0)
        batchnorm (bool): whether to include a BatchNorm3d layer
        relu (bool): whether to include a LeakyReLU(0.2) layer

    Returns:
        convolutional layer as list of torch.nn modules
    """

    # define the convolutional layer
    conv = nn.ModuleList(
        [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        ]
    )

    # add batch normalisation layer
    if batchnorm:
        conv.append(nn.BatchNorm3d(out_channels))

    # add ReLu layer
    if relu:
        conv.append(nn.LeakyReLU(0.2, inplace=True))

    # add maxpool layer (to downsample for discriminators)
    if maxpool != 0:
        # conv = [nn.MaxPool3d(downsample)] + conv
        conv.append(nn.MaxPool3d(maxpool))

    return conv


def initialise_weights(model):
    """
    Weight Initiliaser
    input: the generator instance
    output: the generator instance, with initalised weights
            (for the Conv3d and BatchNorm3d layers)
            i.e. they are normally distributed with normal 0 and sigma 0.02
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.BatchNorm3d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


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


def fourier_loss(x_proj: torch.Tensor, y_proj: torch.Tensor, z_proj: torch.Tensor):

    """
    obtains the loss from the x, y, and z power spectra
    defined as the mean difference between the z-spectrum and the x-and-y-spectra

    Args:
        x_proj (torch tensor of dims (batch, 1, length of spectrum)): A single batch of power spectra generated from the x-projection
        y_proj (torch tensor of dims (batch, 1, length of spectrum)): A single batch of power spectra generated from the y-projection
        z_proj (torch tensor of dims (batch, 1, length of spectrum)): A single batch of power spectra generated from the z-projection

    Returns:
        freq_domain_loss, the mean deviation of the z-spectrum from the x-and-y-spectra (calcualted as a simple difference)
    """

    assert (
        len(x_proj.size()) == 3
    ), "x-projection must have dimensions [batch, channels, spectrum_length]"
    assert (
        len(y_proj.size()) == 3
    ), "y-projection must have dimensions [batch, channels, spectrum_length]"
    assert (
        len(z_proj.size()) == 3
    ), "z-projection must have dimensions [batch, channels, spectrum_length]"

    batch_size = torch.tensor(x_proj.size(0)).item()

    # this is the x and y projections (in a single tensor)
    xy_proj = torch.stack([x_proj, y_proj], dim=0)
    # to calculate the loss compared to the z projection, we need to double it up
    zz_proj = torch.stack([z_proj, z_proj], dim=0)

    # the loss is the difference between the log of the projections
    # + 1e-4 to ensure there is no log(0)
    freq_domain_loss = torch.log(xy_proj + 1e-4) - torch.log(zz_proj + 1e-4)
    # take the absolute value to remove imaginary components, square them, and sum
    freq_domain_loss = torch.sum(torch.pow(torch.abs(freq_domain_loss), 2), dim=-1)
    # channels not needed here - remove the channels dimension
    freq_domain_loss = freq_domain_loss.squeeze()

    # for batches of multiple images, take the mean as the loss
    if batch_size > 1:
        # this is the mean loss for the batch when compared with the x axis
        freq_domain_loss_x = torch.mean(freq_domain_loss[0, :])
        # this is the mean loss for the batch when compared with the y axis
        freq_domain_loss_y = torch.mean(freq_domain_loss[1, :])
        # both means as a single tensor
        freq_domain_loss = torch.tensor((freq_domain_loss_x, freq_domain_loss_y))

    return freq_domain_loss


if __name__ == "__main__":
    print(conv(4, 8))
