"""
Discriminator and Generator class implementation from DCGAN paper
"""

import torch
import torch.nn as nn


# A factory that churns out generators:
class Generator(nn.Module):
    def __init__(self, n_colour_channels, features_generator, kernel_size, padding):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self.nn_block(n_colour_channels * 1,  features_generator * 1, kernel_size, 1, padding),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            self.nn_block(features_generator * 1, features_generator * 2, kernel_size, 1, padding),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            self.nn_block(features_generator * 2, features_generator * 4, kernel_size, 1, padding),
            self.nn_block(features_generator * 4, features_generator * 8, kernel_size, 1, padding),
            nn.Conv2d(features_generator * 8, n_colour_channels, kernel_size=5, stride=1, padding=2),
            # Output: N x n_colour_channels x 64 x 64
            nn.Tanh(),)

    # remove padding as input, calculate from kernel_size
    def nn_block(self, in_channels, out_channels, kernel_size, stride, padding):
        # NOTE: change to nn.Transpose2d (to increase channels)
        # then add interpolation layer
        # better to go faster up and slower down
        return nn.Sequential(nn.Conv2d(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride,
                                       padding,
                                       bias=False,),
                             nn.BatchNorm2d(out_channels),
                             nn.LeakyReLU(0.2),)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_discriminator, kernel_size, padding):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_discriminator, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
            # nn_block(in_channels, out_channels, kernel_size, stride, padding)
            self.nn_block(features_discriminator * 1, features_discriminator * 2, kernel_size, 2, padding=0),
            self.nn_block(features_discriminator * 2, features_discriminator * 4, kernel_size, 2, padding=0),
            self.nn_block(features_discriminator * 4, features_discriminator * 8, kernel_size, 3, padding=0),
            # After all nn_block img output is 2x2 (Conv2d below makes into 1x1)
            nn.Conv2d(features_discriminator * 8, 1, kernel_size=4, stride=2, padding=0),
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


# THIS IS WHAT I NEED TO UNDERSTAND #
def initialise_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def check_success(loader, model, device):
    n_samples = 0
    n_correct = 0
    model.eval()

    with torch.no_grad():
        print("obtaining accuracy on test data")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max()
            n_correct += (predictions == y).sum()
            n_samples += predictions.size(0)

        print(f'Got {n_correct}/{n_samples} with accuracy {(n_correct/n_samples) * 100}')



