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
    def __init__(self, channels_img, features_discriminator):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # each conv2d halves the size of the image
            # img_dimensions = 64 x 64 x colour_channels
            nn.Conv2d(channels_img, features_discriminator, kernel_size=3, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            # img_dimensions = 32 x 32 x features_discriminator
            self.nn_block(features_discriminator * 1, features_discriminator * 2, 3, 2, 0),
            # img_dimensions = 16 x 16 x features_discriminator * 2
            self.nn_block(features_discriminator * 2, features_discriminator * 4, 3, 2, 0),
            # img_dimensions = 8 x 8 x features_discriminator * 4
            self.nn_block(features_discriminator * 4, features_discriminator * 8, 3, 2, 0),
            # img_dimensions = 4 x 4 x features_discriminator * 8
            nn.Conv2d(features_discriminator * 8, 1, kernel_size=4, stride=1, padding=0),
            # img_dimensions = 1 x 1 x 1
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


