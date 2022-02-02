"""
Generator for increasing the resolution of a 3D image
"""

import torch
import torch.nn as nn


# A factory that churns out generators:
class Generator(nn.Module):
    def __init__(self, n_features, kernel_size, padding):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self.nn_block(1, n_features * 1, kernel_size, 1, padding),
            self.nn_block(n_features * 1, n_features * 2, kernel_size, 1, padding),
            self.nn_block(n_features * 2, n_features * 4, kernel_size, 1, padding),
            self.nn_block(n_features * 4, n_features * 8, kernel_size, 1, padding),
            nn.Conv3d(
                n_features * 8, 1, kernel_size=kernel_size, stride=1, padding=padding
            ),
            # nn.Tanh(),
        )

    def nn_block(self, in_channels, out_channels, kernel_size, stride, padding):
        # better to go faster up and slower down
        return nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


# TODO THIS IS WHAT I NEED TO UNDERSTAND #
def initialise_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.BatchNorm3d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
