import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from skimage import io
from pathlib import Path


# make sure calculations in these classes can be done on the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FourierProjection(object):
    """
    Transform

    input: a symmetrically-sized 3D image
    output: a projection of the input (in x, y, or z), but in the frequency domain
    args: the dimension you want (x, y, or z) defined as 0, 1, or 2 respectively
          sigma, the standard deviation of the psf in z
    """

    def __init__(self, dim, sigma):

        # 0, 1, 2 == x, y, z
        self.dim = dim
        # standard deviation of PSF in z
        self.sigma = sigma
        # these are the coefficients for a blackman-harris window
        self.coeffs = [0.35875, 0.48829, 0.14128, 0.01168]

    def __call__(self, sample):

        # torch transform essential syntax
        image, landmarks = sample['image'], sample['landmarks']

        # projections for original data
        if self.dim == 0:
            image = image.sum(1).sum(0)
        elif self.dim == 1:
            image = image.sum(2).sum(0)
        elif self.dim == 2:
            image = image.sum(2).sum(1)

        # define image dimensions
        image_size = torch(max(image.shape))

        # these are the cosine arguments to make a cosine window
        cos_args = torch.tensor(range(0, image_size)) * 2 * math.pi / image_size
        # generate the sampling window
        sampling_window = torch.zeros(image_size)
        for i, c in enumerate(self.coeffs):
            sampling_window += ((-1) ** i) * c * torch.cos(cos_args * i)

        # apply a window to each projection
        image *= sampling_window.expand((1, image_size.shape(0)))

        # power spectrum for each projection
        image = torch.abs(torch.fft.rfft(image, dim=1)) ** 2

        # create highpass gaussian kernel filter
        # centre of the image (halfway point)
        filter = math.floor(image_size / 2)
        # centre filter on 0
        filter = torch.tensor(range(0, image_size), dtype=torch.float) - filter
        # convert to gaussian distribution
        filter = torch.exp(-(filter ** 2) / (2 * self.sigma ** 2))
        # normalise (to normal distribution)
        filter /= sum(filter)
        # compute the fourier transform of the distribution
        filter = 1 - torch.abs(torch.fft.rfft(filter, dim=0))

        # apply highpass gaussian kernel filter to transformed image
        image *= filter

        # torch transform essential syntax
        return {'image': image, 'landmarks': landmarks}


class FourierProjectionLoss(nn.Module):
    """
    Loss Function Class

    input: highpass_filter(x,y,z-projections(fourier-transformed images))
    all projections must have the same dimensions, but need not come from the same image
    output: loss, as float
    args: none for creating instance
    """

    def __init__(self, x_proj, y_proj, z_proj):

        # super() to inherit from parent class (standard for pytorch transforms)
        super().__init__()
        # this is the x and y projections (in a single tensor)
        self.xy_proj = torch.stack([x_proj, y_proj], dim=0)
        # to calculate the loss compared to the z projection, we need to double it up
        self.zz_proj = z_proj.expand((2, x_proj.size(1)))

    def forward(self):

        # the loss is the difference between the log of the projections
        freq_domain_loss = torch.log(self.xy_proj) - torch.log(self.zz_proj)
        # take the absolute value to remove imaginary components
        freq_domain_loss = torch.abs(freq_domain_loss)
        # square the values
        freq_domain_loss = torch.pow(freq_domain_loss, 2)
        # add them together
        freq_domain_loss = torch.sum(freq_domain_loss, dim=1)
        # make it autograd-sensitive so it is receptive to .backward()
        freq_domain_loss = Variable(freq_domain_loss, requires_grad=True).to(device)

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
        # import image (scikit-image.io imports tiffs as (z, x, y))
        img = io.imread(img_path)
        # convert to numpy array for faster calculations
        img = np.asarray(img)
        # normalise pixel values
        img = img / np.max(img)

        # convert to tensor
        img = torch.tensor(img)
        # now: img.shape = (z, x, y)
        img = torch.swapaxes(img, 1, 2)
        # now: img.shape = (z, y, x)
        img = img.unsqueeze(0)
        # now: img.shape = (1, z, y, x)
        return img


def test():
    """
    check that the class outputs datasets correctly
    the 'noisetest' dataset has dimensions (96, 64, 32)
    and can thus be used to check that the batches are the right shape
    """

    # Image filepaths
    lores_filepath = os.path.join(os.getcwd(),
                                  Path("images/sims/microtubules/lores"))
    hires_filepath = os.path.join(os.getcwd(),
                                  Path("images/sims/microtubules/hires"))
    # noisetest_filepath = os.path.join(os.getcwd(),
    #                               Path("images/sims/noise"))

    # image datasets
    lores_dataset = Custom_Dataset(dir_data=lores_filepath,
                                   filename="mtubs_sim_*_lores.tif")
    hires_dataset = Custom_Dataset(dir_data=hires_filepath,
                                   filename="mtubs_sim_*_hires.tif")
    # noisetest_dataset = Custom_Dataset(dir_data=lores_filepath,
    #                                filename="test_*.tif")

    # image dataloaders
    lores_dataloader = DataLoader(lores_dataset,
                                  batch_size=5,
                                  shuffle=True,
                                  num_workers=2)
    hires_dataloader = DataLoader(hires_dataset,
                                  batch_size=5,
                                  shuffle=True,
                                  num_workers=2)
    # noisetest_dataloader = DataLoader(noisetest_dataset,
    #                                   batch_size=5,
    #                                   shuffle=True,
    #                                   num_workers=2)

    # iterator objects from dataloaders
    lores_iterator = iter(lores_dataloader)
    hires_iterator = iter(hires_dataloader)
    # noisetest_iterator = iter(noisetest_dataloader)

    # pull out a batch!
    # sometimes the iterators 'run out', this stops that from happening
    try:
        lores_batch = next(lores_iterator)
        hires_batch = next(hires_iterator)
        # noisetest_batch = next(noisetest_iterator)
    except StopIteration:
        lores_iterator = iter(lores_dataloader)
        hires_iterator = iter(hires_dataloader)
        # noisetest_iterator = iter(noisetest_dataloader)
        lores_batch = next(lores_iterator)
        hires_batch = next(hires_iterator)
        # noisetest_batch = next(noisetest_iterator)

    # print the batch shape
    print(lores_batch.shape)
    print(hires_batch.shape)
    # print(noisetest_batch.shape)


if __name__ == "__main__":
    test()
