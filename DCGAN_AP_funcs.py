"""
Discriminator and Generator class implementation from DCGAN paper
"""

import torch
import torch.nn as nn


# A factory that churns out discriminators:
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_discriminator):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_discriminator, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # nn_block(in_channels, out_channels, kernel_size, stride, padding)
            self.nn_block(features_discriminator, features_discriminator * 2, 4, 2, 1),
            self.nn_block(features_discriminator * 2, features_discriminator * 4, 4, 2, 1),
            self.nn_block(features_discriminator * 4, features_discriminator * 8, 4, 2, 1),
            # After all nn_block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_discriminator * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def nn_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.Conv2d(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride,
                                       padding,
                                       bias=False,),
                             nn.BatchNorm2d(out_channels),
                             nn.LeakyReLU(0.2, inplace=True),)

    def forward(self, x):
        return self.disc(x)


# A factory that churns out generators:
class Generator(nn.Module):
    def __init__(self, size_noise, n_colour_channels, features_generator):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self.nn_block(size_noise, features_generator * 8, 4, 1, 0, 0.2),  # img: 4x4
            self.nn_block(features_generator * 8, features_generator * 4, 4, 2, 1, 0.2),  # img: 8x8
            self.nn_block(features_generator * 4, features_generator * 2, 4, 2, 1, 0.2),  # img: 16x16
            self.nn_block(features_generator * 2, features_generator * 1, 4, 2, 1, 0.2),  # img: 32x32
            nn.ConvTranspose2d(features_generator * 1, n_colour_channels, kernel_size=4, stride=2, padding=1),
            # Output: N x n_colour_channels x 64 x 64
            nn.Tanh(),)

    def nn_block(self, in_channels, out_channels, kernel_size, stride, padding, leak_grad):
        # NOTE: change to nn.Transpose2d (to increase channels)
        # then add interpolation layer
        # better to go faster up and slower down
        return nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                out_channels,
                                                kernel_size,
                                                stride,
                                                padding,
                                                bias=False,),
                             nn.BatchNorm2d(out_channels),
                             nn.LeakyReLU(leak_grad),)

    def forward(self, x):
        return self.net(x)

### THIS IS WHAT I NEED TO UNDERSTAND ###
def initialise_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)