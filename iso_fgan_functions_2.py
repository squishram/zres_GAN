import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage import io
from pathlib import Path
import torchio as tio
import torchio.transforms as transforms


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

    def __init__(self, sigma, coeffs=[0.35875, 0.48829, 0.14128, 0.01168]):

        # standard deviation of PSF in z
        self.sigma = sigma
        # the defaults are the coefficients for a blackman-harris window
        self.coeffs = coeffs

    def hipass_gauss_kernel_fourier(self, image_size):
        """
        Make an unshifted Gaussian highpass filter in Fourier space. All real
        """

        filter = math.floor(image_size / 2)
        # centre filter on origin
        filter = torch.tensor(range(0, image_size), dtype=torch.float) - filter
        # convert to gaussian distribution
        filter = torch.exp(-(filter**2) / (2 * self.sigma**2))
        # normalise (to normal distribution)
        filter /= sum(filter)
        # compute the fourier transform of the distribution
        filter = 1 - torch.abs(torch.fft.rfft(filter, dim=0))
        # must be on the gpu
        filter = filter.to(device)

        return filter

    def __call__(self, image, dim):

        # batch size - .item() to convert from tensor -> int
        batch_size = torch.tensor(image.size(0)).item()

        # projections for original data
        # why are we projecting down two axes surely it should just be the one?
        # information density okay godditt
        if dim == 0:
            image = image.sum(3).sum(2)
        elif dim == 1:
            image = image.sum(4).sum(2)
        elif dim == 2:
            image = image.sum(4).sum(3)

        # this is the side length of the projection
        image_size = torch.tensor(image.size(2)).item()

        # these are the arguments to make a cosine window
        cos_args = torch.tensor(range(0, image_size)) * 2 * math.pi / image_size
        # generate the sampling window
        sampling_window = torch.zeros(image_size)
        for idx, coefficient in enumerate(self.coeffs):
            sampling_window += ((-1) ** idx) * coefficient * torch.cos(cos_args * idx)

        # sampling_window must be on the same device as the image
        sampling_window = sampling_window.to(device)
        # apply the window to the projection
        image *= sampling_window.expand((1, image_size))

        # fourier transform
        image = torch.abs(torch.fft.rfft(image, dim=2)) ** 2

        # apply highpass gaussian kernel filter to transformed image
        for idx in range(batch_size):
            image[idx, 0, :] *= self.hipass_gauss_kernel_fourier(image_size)

        return image


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
        freq_domain_loss = torch.log(xy_proj) - torch.log(zz_proj)
        # take the absolute value to remove imaginary components, square them, and sum
        freq_domain_loss = torch.sum(torch.pow(torch.abs(freq_domain_loss), 2), dim=-1)
        # channels not needed here - remove the channels dimension!
        freq_domain_loss = freq_domain_loss.squeeze()

        if batch_size > 1:
            # this is the mean loss for the batch when compared with the x axis
            freq_domain_loss_x = torch.mean(freq_domain_loss[0, :])
            # this is the mean loss for the batch when compared with the y axis
            freq_domain_loss_y = torch.mean(freq_domain_loss[1, :])
            # both means as a single tensor
            freq_domain_loss = torch.tensor((freq_domain_loss_x, freq_domain_loss_y))

        return freq_domain_loss


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

    def __init__(self, n_features, kernel_size, padding=None):
        """
        formula for calculating dimensions after convolution
        output = 1 + (input + 2*padding - kernel) / stride
        """

        super(Generator, self).__init__()

        if not padding:
            padding = int(kernel_size / 2)

        self.net = nn.Sequential(
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
        return self.net(batch)


def initialise_weights(model):
    """
    Weight Initiliaser
    input: the generator instance
    output: the generator instance, with initalised weights
            (for the Conv3d and BatchNorm3d layers)

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

    transform = tio.Compose(
        [
            transforms.RescaleIntensity((0, 1)),
            # transforms.ZNormalization(),
            # transforms.RescaleIntensity((0, 1)),
        ]
    )
    # Image filepaths
    # lores_filepath = os.path.join(os.getcwd(), Path("images/sims/microtubules/lores"))
    # hires_filepath = os.path.join(os.getcwd(), Path("images/sims/microtubules/hires"))
    noisetest_filepath = os.path.join(
        os.getcwd(), Path("images/sims/noise/cuboidal_noise")
    )

    # image datasets
    # lores_dataset = Custom_Dataset(
    #     dir_data=lores_filepath, filename="mtubs_sim_*_lores.tif"
    # )
    # hires_dataset = Custom_Dataset(
    #     dir_data=hires_filepath, filename="mtubs_sim_*_hires.tif"
    # )
    noisetest_dataset = Custom_Dataset(
        dir_data=noisetest_filepath, filename="test_*.tif", transform=transform
    )

    # image dataloaders
    # lores_dataloader = DataLoader(
    #     lores_dataset, batch_size=5, shuffle=True, num_workers=2
    # )
    # hires_dataloader = DataLoader(
    #     hires_dataset, batch_size=5, shuffle=True, num_workers=2
    # )
    noisetest_dataloader = DataLoader(
        noisetest_dataset, batch_size=5, shuffle=True, num_workers=2
    )

    # iterator objects from dataloaders
    # lores_iterator = iter(lores_dataloader)
    # hires_iterator = iter(hires_dataloader)
    noisetest_iterator = iter(noisetest_dataloader)

    # pull out a batch!
    # sometimes the iterators 'run out', this stops that from happening
    try:
        # lores_batch = next(lores_iterator)
        # hires_batch = next(hires_iterator)
        noisetest_batch = next(noisetest_iterator)
    except StopIteration:
        # lores_iterator = iter(lores_dataloader)
        # hires_iterator = iter(hires_dataloader)
        noisetest_iterator = iter(noisetest_dataloader)
        # lores_batch = next(lores_iterator)
        # hires_batch = next(hires_iterator)
        noisetest_batch = next(noisetest_iterator)

    # print the batch shape
    # print(lores_batch.shape)
    # print(hires_batch.shape)
    print(noisetest_batch.shape)
    # max and min values should be 1 and 0
    print(noisetest_batch.max())
    print(noisetest_batch.min())


if __name__ == "__main__":
    test()
