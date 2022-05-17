import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage import io
from pathlib import Path


# make sure calculations in these classes can be done on the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FourierProjection(object):
    """
    Transform Class

    fields: sigma, the standard deviation of the psf in z
            the coefficients for the cosine window (default to blackman-harris window)
    args: image, the image being projected
          dim, the dimension we wish to project down {0:x, 1:y, 2:z}
    output: a projection of the input (in x, y, or z), but in the frequency domain
            which has been cosine-windowed and normalised
    """

    def __init__(self, sigma, window=None):

        # standard deviation of PSF in z
        self.sigma = sigma
        # sampling window
        self.window = window

    def __call__(self, image, dim):

        # batch size - .item() to convert from tensor -> int
        batch_size = torch.tensor(image.size(0)).item()

        # projections for original data
        # we project down two axes as a 1D signal feed is easier to fourier transform
        if dim == 0:
            projection = image.sum(3).sum(2)
        elif dim == 1:
            projection = image.sum(4).sum(2)
        elif dim == 2:
            projection = image.sum(4).sum(3)

        # this is the side length of the projection
        image_size = torch.tensor(projection.size(2)).item()

        ######################################################################
        # MAKING AND APPLYING A SAMPLING WINDOW BEFORE THE FOURIER TRANSFORM #
        ######################################################################
        """
        Signals are assumed periodic but are processed as discontinuous where
        one period ends and another begins. Sampling windows bring the amplitude
        of a signal towards 0 at the lobes where there is period change, forcing continuity and
        reducing noise in the fourier transform
        """

        # cosine arguments for sampling windows
        cos_args = (2 * math.pi / image_size) * torch.tensor(range(image_size))

        # HANN WINDOW #
        if self.window == "hann":
            sampling_window = 0.5 * (1 - torch.cos(cos_args))

            sampling_window = sampling_window.to(device)
            projection *= sampling_window

        # HAMMING WINDOW #
        elif self.window == "hamming":
            sampling_window = 0.56 - 0.46 * (1 - torch.cos(cos_args))

            sampling_window = sampling_window.to(device)
            projection *= sampling_window

        # 4-TERM BLACKMAN-HARRIS WINDOW #
        elif self.window == "bharris":
            sampling_window = torch.zeros(image_size)
            coeffs = [0.35875, 0.48829, 0.14128, 0.01168]
            for idx, coefficient in enumerate(coeffs):
                sampling_window += ((-1) ** idx) * coefficient * torch.cos(cos_args * idx)

            sampling_window = sampling_window.to(device)
            projection *= sampling_window

        # fourier transform of the projection
        projection = torch.abs(torch.fft.rfft(projection, dim=2)) ** 2

        #########################################
        # APPLYING THE HIGHPASS GAUSSIAN FILTER #
        #########################################
        """
        Filter ensures that projections are compared on the basis of
        high-frequency information only
        """

        # Highpass Gaussian Kernel Filter
        # centre of the image (halfway point)
        filter = math.floor(image_size / 2)
        # centre filter on origin (this is just a list of pixel indexes)
        filter = torch.tensor(range(image_size), dtype=torch.float) - filter
        # convert to gaussian distribution
        filter = torch.exp(-(filter**2) / (2 * self.sigma**2))
        # normalise (to normal distribution)
        filter /= sum(filter)
        # compute the fourier transform of the filter
        filter = 1 - torch.abs(torch.fft.rfft(filter, dim=0))
        # must be on the gpu
        filter = filter.to(device)

        # apply highpass gaussian kernel filter to transformed image
        for idx in range(batch_size):
            projection[idx, 0, :] *= filter

        return projection


class FourierProjectionLoss(nn.Module):
    """
    Loss Function Class
    finds the loss of a z-projection with respect to an x-and-y-projection

    all projections must have the same dimensions, but need not come from the same image
    output: loss, as float
    fields: none
    args: highpass_filter(x,y,z-projections(fourier-transformed images))
    """

    def __init__(self):

        # super() to inherit from parent class (standard for pytorch transforms)
        super().__init__()

    def forward(self, x_proj, y_proj, z_proj):

        batch_size = torch.tensor(x_proj.size(0)).item()

        # this is the x and y projections (in a single tensor)
        xy_proj = torch.stack([x_proj, y_proj], dim=0)
        # to calculate the loss compared to the z projection, we need to double it up
        zz_proj = torch.stack([z_proj, z_proj], dim=0)

        # the loss is the difference between the log of the projections
        # + 1e-4
        freq_domain_loss = torch.log(xy_proj + 1e-4) - torch.log(zz_proj + 1e-4)
        # take the absolute value to remove imaginary components, square them, and sum
        freq_domain_loss = torch.sum(torch.pow(torch.abs(freq_domain_loss), 2), dim=-1)
        # channels not needed here - remove the channels dimension
        freq_domain_loss = freq_domain_loss.squeeze()

        if batch_size > 1:
            # this is the mean loss for the batch when compared with the x axis
            freq_domain_loss_x = torch.mean(freq_domain_loss[0, :])
            # this is the mean loss for the batch when compared with the y axis
            freq_domain_loss_y = torch.mean(freq_domain_loss[1, :])
            # both means as a single tensor
            freq_domain_loss = torch.tensor((freq_domain_loss_x, freq_domain_loss_y))

        return freq_domain_loss


class Custom_Dataset_Pairs(Dataset):
    """
    makes a pytorch dataset with torch.utils.data.Dataset as its only argument
    this means it automatically includes a set of methods for loading datasets
    of these, the following need to be overwritten on a 'per-case' basis:
        __init__(self, *args, **kwargs),
        which obtains data like filepath (and csv for supervised learning)

        __len__(self),
        which obtains the size of the dataset (dunno why this isn't standard)

        __getitem__(self, idx)
        which reads and transforms the images themselves
        (this is more memory-efficient than doing this in __init__
        because this way, not all the images are not stored in memory at once -
        instead, they are read as required)
    """

    def __init__(self, dir_data: str, subdirs: tuple, filename: str, transform=None):
        """
        defines the directory & filenames of the images to be loaded
        dir_data e.g. "~/Pictures/"
        subdirs e.g. ("lores", "hires")
        filename e.g. "image_*.tif"
        """

        # the directory containing the images
        self.dir_data1 = Path(os.path.join(dir_data, subdirs[0]))
        self.dir_data2 = Path(os.path.join(dir_data, subdirs[1]))
        # the structure of the filename (e.g. "image_*.tif")
        self.filename = filename
        # list of filenames
        self.files1 = sorted(self.dir_data1.glob(self.filename))
        self.files2 = sorted(self.dir_data2.glob(self.filename))
        # transform for the images, if defined
        self.transform = transform

    def __len__(self):
        """
        method for obtaining the total number of samples
        """

        return min(len(self.files1), len(self.files2))

    def __getitem__(self, file):
        """
        method for obtaining one sample of data
        the argument 'file':
        is simply the index of the filenames listed in self.files
        it is included in classes that call torch.utils.data.Dataset
        and does not need to be defined anywhere else
        """

        # select image
        img_path1 = os.path.join(self.dir_data1, self.files1[file])
        img_path2 = os.path.join(self.dir_data2, self.files2[file])
        # import image (scikit-image.io imports tiffs as np.array(z, x, y))
        img1 = io.imread(img_path1)
        img2 = io.imread(img_path2)
        imgs = np.stack((img1, img2))
        # now: img.shape = (2, z, x, y)
        # choose datatype using numpy
        imgs = np.asarray(imgs, dtype=np.float32)
        # normalise to range (0, 1)
        imgs = imgs / np.max(imgs)
        # convert to tensor
        imgs = torch.tensor(imgs)
        # add channels dimensions
        imgs = imgs.unsqueeze(1)
        # now: imgs.shape = (2, 1, z, x, y)
        imgs = torch.swapaxes(imgs, -2, -1)
        # now: imgs.shape = (2, 1, z, y, x)

        # apply transform from torchio library if defined
        if self.transform:
            imgs = self.transform(imgs)

        return imgs


class Custom_Dataset(Dataset):
    """
    makes a pytorch dataset with torch.utils.data.Dataset as its only argument
    this means it automatically includes a set of methods for loading datasets
    of these, the following need to be overwritten on a 'per-case' basis:
        __init__(self, *args, **kwargs),
        which obtains data like filepath (and csv for supervised learning)

        __len__(self),
        which obtains the size of the dataset (dunno why this isn't standard)

        __getitem__(self, idx)
        which reads and transforms the images themselves
        (this is more memory-efficient than doing this in __init__
        because this way, not all the images are not stored in memory at once -
        instead, they are read as required)
    """

    def __init__(self, dir_data: str, filename: str, transform=None):
        """
        defines the directory & filenames of the images to be loaded
        dir_data e.g. "~/Pictures/"
        filename e.g. "image_*.tif"
        """

        # the directory containing the images
        self.dir_data = Path(dir_data)
        # the structure of the filename (e.g. "image_*.tif")
        self.filename = filename
        # list of filenames
        self.files = sorted(self.dir_data.glob(self.filename))
        # transform for the images, if defined
        self.transform = transform

    def __len__(self):
        """
        method for obtaining the total number of samples
        """

        return len(self.files)

    def __getitem__(self, file):
        """
        method for obtaining one sample of data
        the argument 'file':
        is simply the index of the filenames listed in self.files
        it is included in classes that call torch.utils.data.Dataset
        and does not need to be defined anywhere else
        """

        # select image
        img_path = os.path.join(self.dir_data, self.files[file])
        # import image (scikit-image.io imports tiffs as np.array(z, x, y))
        img = io.imread(img_path)
        # now: img.shape = (z, x, y)
        # choose datatype using numpy
        img = np.asarray(img, dtype=np.float32)
        # normalise to range (0, 1)
        img = img / np.max(img)
        # convert to tensor
        img = torch.tensor(img)
        # add channels dimension
        img = img.unsqueeze(0)
        # now: img.shape = (1, z, x, y)
        img = torch.swapaxes(img, 2, 3)
        # now: img.shape = (1, z, y, x)

        # apply transform from torchio library if defined
        if self.transform:
            img = self.transform(img)

        return img


class Generator(nn.Module):
    """
    Convolutional Generational Network Class
    Takes in 3D images and outputs 3D images of the same dimension

    fields:
    n_features, channel depth after convolution
    kernel_size (int), side length of cubic kernel
    padding (int), count of padding blocks
    """

    def __init__(self, n_features, kernel_size=False, padding=False):
        """
        formula for calculating dimensions after convolution
        output = 1 + (input + 2*padding - kernel) / stride
        """

        # class inheritance
        super(Generator, self).__init__()

        if not kernel_size:
            kernel_size = 3
        if not padding:
            padding = int(kernel_size / 2)

        self.gen = nn.Sequential(
            self.nn_block(1, n_features * 4, kernel_size, 1, padding),
            self.nn_block(n_features * 4, n_features * 2, kernel_size, 1, padding),
            self.nn_block(n_features * 2, n_features * 1, kernel_size, 1, padding),
            # self.nn_block(n_features * 4, n_features * 8, kernel_size, 1, padding),
            nn.Conv3d(
                n_features * 1, 1, kernel_size=kernel_size, stride=1, padding=padding
            ),
            # image values need to be between 0 and 1
            nn.Sigmoid(),
        )

    def nn_block(self, in_channels, out_channels, kernel_size, stride, padding):
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

    def forward(self, batch):
        return self.gen(batch)


class Discriminator(nn.Module):
    """
    Convolutional Generational Network Class
    Takes in 3D images and outputs a number between 1 and 0

    fields:
    n_features, channel depth after convolution
    kernel_size (int), side length of cubic kernel
    padding (int), count of padding blocks
    """

    def __init__(self, n_features):

        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # 128*128*32
            nn.Conv3d(1, n_features * 8, kernel_size=[3, 4, 4], stride=[1, 2, 2], padding=[1, 1, 1]),
            nn.LeakyReLU(0.2, inplace=True),
            # 64*64*32
            self.nn_block(n_features * 8, n_features * 4, [3, 4, 4], [1, 2, 2], [1, 1, 1]),
            # 32*32*32
            self.nn_block(n_features * 4, n_features * 2, 2, 2, 0),
            # 16*16*16
            self.nn_block(n_features * 2, n_features * 1, 2, 2, 0),
            # 8*8*8
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            # 4*4*4
            nn.Conv3d(n_features * 1, 1, kernel_size=4, stride=1, padding=0),
            # 1*1*1
            nn.Sigmoid(),
        )
        # self.disc = nn.Sequential(
        #     # 128*128*32
        #     nn.Conv3d(1, n_features * 8, kernel_size=[3, 4, 4], stride=[1, 2, 2], padding=[1, 1, 1]),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # 64*64*32
        #     self.nn_block(n_features * 8, n_features * 4, [3, 4, 4], [1, 2, 2], [1, 1, 1]),
        #     # 32*32*32
        #     self.nn_block(n_features * 4, n_features * 2, 8, 4, 2),
        #     # 16*16*16
        #     self.nn_block(n_features * 4, n_features * 2, 8, 4, 2),
        #     # 8*8*8
        #     self.nn_block(n_features * 2, n_features * 1, 4, 2, 1),
        #     # 4*4*4
        #     nn.Conv3d(n_features * 2, 1, kernel_size=4, stride=1, padding=0),
        #     # 1*1*1
        #     nn.Sigmoid(),
        # )
        # self.disc = nn.Sequential(
        #     # 128*128*32
        #     nn.Conv3d(1, n_features * 8, kernel_size=[8, 8, 4], stride=[4, 4, 2], padding=[2, 2, 1]),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # 64*64*32
        #     # self.nn_block(n_features * 8, n_features * 4, [8, 8, 4], [4, 4, 2], [1, 1, 1]),
        #     # 32*32*32
        #     self.nn_block(n_features * 8, n_features * 4, [4, 4, 3], [2, 2, 1], [1, 1, 1]),
        #     # self.nn_block(n_features * 8, n_features * 4, [8, 8, 4], [4, 4, 2], [2, 2, 1]),
        #     # 16*16*16
        #     self.nn_block(n_features * 4, n_features * 2, 8, 4, 2),
        #     # 8*8*8
        #     # self.nn_block(n_features * 2, n_features * 1, 4, 2, 1),
        #     # 4*4*4
        #     # nn.Conv3d(n_features * 2, n_features * 1, kernel_size=4, stride=1, padding=0),
        #     nn.Conv3d(n_features * 2, 1, kernel_size=4, stride=1, padding=0),
        #     # 1*1*1
        #     nn.Sigmoid(),
        # )

    def nn_block(self, in_channels, out_channels, kernel_size, stride, padding):
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
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.disc(x)


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


def test():
    """
    check that Custom_Dataset outputs datasets correctly
    the 'noisetest' dataset has dimensions {x:96, y:64, z:32}
    and can thus be used to check that the batches are the right shape
    each batch should have dimensions {batch_size, 1, z, y, x}
    """

    # Image filepaths
    # lores_filepath = os.path.join(os.getcwd(), Path("images/sims/microtubules/lores"))
    # hires_filepath = os.path.join(os.getcwd(), Path("images/sims/microtubules/hires"))
    # noisetest_filepath = os.path.join(
    #     os.getcwd(), Path("images/sims/noise/cuboidal_noise")
    # )
    dualtest_filepath = path_data = os.path.join(
        os.getcwd(), Path("images/sims/microtubules")
    )
    # subdirectories with lores and hires data
    lores_subdir = "lores"
    hires_subdir = "hires"

    # image datasets
    # lores_dataset = Custom_Dataset(
    #     dir_data=lores_filepath, filename="mtubs_sim_*_lores.tif"
    # )
    # hires_dataset = Custom_Dataset(
    #     dir_data=hires_filepath, filename="mtubs_sim_*_hires.tif"
    # )
    # noisetest_dataset = Custom_Dataset(
    #     dir_data=noisetest_filepath, filename="test_*.tif", transform=transform
    # )
    dualtest_dataset = Custom_Dataset_Pairs(
        dir_data=dualtest_filepath,
        subdirs=(lores_subdir, hires_subdir),
        filename="mtubs_sim_*.tif",
    )

    # image dataloaders
    # lores_dataloader = DataLoader(
    #     lores_dataset, batch_size=5, shuffle=True, num_workers=2
    # )
    # hires_dataloader = DataLoader(
    #     hires_dataset, batch_size=5, shuffle=True, num_workers=2
    # )
    # noisetest_dataloader = DataLoader(
    #     noisetest_dataset, batch_size=5, shuffle=True, num_workers=2
    # )
    dualtest_dataloader = DataLoader(dualtest_dataset, batch_size=5, shuffle=True)

    # iterator objects from dataloaders
    # lores_iterator = iter(lores_dataloader)
    # hires_iterator = iter(hires_dataloader)
    # noisetest_iterator = iter(noisetest_dataloader)
    dualtest_iterator = iter(dualtest_dataloader)

    # pull out a batch!
    # sometimes the iterators 'run out', this stops that from happening
    try:
        # lores_batch = next(lores_iterator)
        # hires_batch = next(hires_iterator)
        # noisetest_batch = next(noisetest_iterator)
        dualtest_batch = next(dualtest_iterator)
    except StopIteration:
        # lores_iterator = iter(lores_dataloader)
        # lores_batch = next(lores_iterator)
        # hires_batch = next(hires_iterator)
        # hires_iterator = iter(hires_dataloader)
        # noisetest_iterator = iter(noisetest_dataloader)
        # noisetest_batch = next(noisetest_iterator)
        dualtest_iterator = iter(dualtest_dataloader)
        dualtest_batch = next(dualtest_iterator)

    # pull out the lores and hires images
    lores_batch = dualtest_batch[:, 0, :, :, :, :]
    hires_batch = dualtest_batch[:, 1, :, :, :, :]

    # print the batch shape
    print(f"lores batch shape is {lores_batch.shape}")
    print(f"hires batch shape is {hires_batch.shape}")
    # max and min values should be 1 and 0
    # print(lores_batch.max())
    # print(lores_batch.min())
    # print(hires_batch.max())
    # print(hires_batch.min())


if __name__ == "__main__":
    test()
