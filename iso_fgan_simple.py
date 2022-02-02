"""
this code represents a trainig methodology for a transductive GAN
the generator will take an image of low-z-resolution and make it isotropic

experimental ("original") image as input
fourier loss is F(z-proj) - [F(x-proj), F(y-proj)]
direct loss is (G(original_image) * gaussian kernel) - original_image

it combines a pixel-to-pixel comparison of:
    a low-z-resolution image and its isotropic equivalent
    a z-projection and an x/y projection of the same image, having undergone:
        a fourier tranform
        lobe removal
        a high-pass filter

there is no discriminator in this version of the network
it does not use a pytorch dataloader
"""

from pathlib import Path
# from astropy.nddata import CCDData
# from torch.utils.data import DataLoader
import os
from skimage import io
from datetime import date
import math
from typing import List
import matplotlib.pyplot as plt
import torch
import numpy as np
from tifffile import imsave


def fwhm_to_sigma(fwhm: float):
    """
    Convert FWHM to standard deviation
    """

    return fwhm / (2 * math.sqrt(2 * math.log(2)))


def load_data(filename):
    # this is a list of all the filenames
    lores_glob = sorted(path_lores.glob("mtubs_sim_*_lores.tif"))
    hires_glob = sorted(path_hires.glob("mtubs_sim_*_hires.tif"))
    # this tensor will contain all the images as float tensors
    lores_data = torch.empty((1, size_img, size_img, size_img))
    hires_data = torch.empty((1, size_img, size_img, size_img))

    for img_path in (lores_glob, hires_glob):
        for file in img_path:

            img = io.imread(file)
            # convert to numpy array for faster calculations
            img = np.asarray(img)
            # normalise pixel values
            img = img - np.mean(img) / np.std(img)
            # scale pixel values to 1
            img = img / np.max(img)

            # convert to tensor
            img = torch.tensor(img)
            # now: img.shape = (z, x, y)
            img = torch.swapaxes(img, 1, 2)
            # now: img.shape = (z, y, x)
            img = img.unsqueeze(0)
            # now: img.shape = (1, z, y, x)

            if img_path == lores_glob:
                lores_data = torch.cat((lores_data, img), 0)
            elif img_path == hires_glob:
                hires_data = torch.cat((hires_data, img), 0)

    # this is all the data
    lores_cpu = lores_data[1:]
    hires_cpu = hires_data[1:]
    # on the gpu
    lores_data = lores_cpu.to(device)
    hires_data = hires_cpu.to(device)

    # this is an optimisation loop on a single image, so we pull out a single sample
    # cpu samples
    # lores_sample_cpu = lores_data[0]
    # hires_sample_cpu = hires_data[0]

    # gpu samples
    # lores_sample = lores_cpu.to(device)
    # hires_sample = hires_cpu.to(device)

    return lores_data, hires_data


def cosine_window(N: int, coefficients: List[float]):
    """
    General cosine window: a sum of increasing cosines with alternating signs
    - N + 1 is the number of samples in the range [0 to 2pi]
    so N represents a discrete approximation of [0, 2pi)
    """

    x = torch.tensor(range(0, N)) * 2 * math.pi / N

    result = torch.zeros(N)

    for i, c in enumerate(coefficients):
        result += c * torch.cos(x * i) * ((-1) ** i)

    return result


def blackman_harris_window(N: int):
    """
    Create a Blackman-Harris cosine window,
    https://en.wikipedia.org/wiki/Window_function#Blackman%E2%80%93Harris_window
    """

    return cosine_window(N, [0.35875, 0.48829, 0.14128, 0.01168])


def complex_abs_sq(data: torch.Tensor):
    # Old style FFTs before complex:
    """
    Compute squared magnitude, assuming the last dimension is
    [re, im], and therefore of size 2
    """
    # assert data.size(-1) == 2,
    # "Last dimension size must be 2, representing [re, im]"
    # return torch.sqrt(torch.sum(data**2, data.ndim-1))

    # print(data.dtype)
    # print(torch.complex64)

    # this assertion will need to be changed if incoming data is of a different type
    # 32 bit images become complex64, etc
    assert data.dtype == torch.complex128
    return torch.abs(data) ** 2


def gaussian_kernel(sigma: float, sigmas: float = 3.0) -> torch.Tensor:
    """
    Make a normalized 1D Gaussian Kernel
    """

    radius = math.ceil(sigma * sigmas)
    xs = np.array(range(-radius, radius + 1))
    kernel = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    return torch.tensor(kernel / sum(kernel), dtype=torch.float)


def conv_1D_z_axis(
    data: torch.Tensor, kernel: torch.Tensor, pad: bool = False
) -> torch.Tensor:
    """
    Assuming data is a cube [z, y, x],
    then perform a 1D convolution along the z axis
    """

    assert len(kernel.size()) == 1, "Kernel is not 1D"
    assert len(data.size()) == 3, "Data is not a cuboid"
    assert len(kernel) % 2 == 1, "Kernel must be odd-sized"

    # Data is [z, y, x]. We need [batch, channels, z, y, x] = [1, 1, z, y, z]
    d = data.unsqueeze(0).unsqueeze(0)

    # Kernel is [z]. We need [channels out, channels in, z, y, x] = [1, 1, z, 1, 1]
    kz = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(4)
    radius = int(len(kernel - 1) / 2)

    if pad == "zero":
        padding = (radius, 0, 0)

    elif pad == "edge":
        padding = (0, 0, 0)
        bottom = data[0, :, :].expand(radius, data.size(1), data.size(2))
        top = data[-1, :, :].expand(radius, data.size(1), data.size(2))
        d = torch.cat([bottom, data, top], dim=0).unsqueeze(0).unsqueeze(0)

    else:
        assert False, "bad pad: " + pad

    # d seems to be a double type, which doesn't work with torch.nn.functional.conv3d()
    d = d.float()

    result = torch.nn.functional.conv3d(d, kz, bias=None, padding=padding)

    return result.squeeze(0).squeeze(0)


def xyz_projections(data: torch.Tensor) -> torch.Tensor:
    """
    Project the cube on to the 3 1D axes. 0th index goes as x, y, z
    """

    assert (data.size(0) == data.size(1)) and (data.size(0) == data.size(2))
    "Data is not a cube"

    x_projection = data.sum(1).sum(0)
    y_projection = data.sum(2).sum(0)
    z_projection = data.sum(2).sum(1)

    return torch.stack([x_projection, y_projection, z_projection], dim=0)


def windowed_projected_PSD(data: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """
    Return the windowed power spectral density (onesided) of input data cube
    0th index goes as x, y, z
    """

    assert (data.size(0) == data.size(1)) and (data.size(0) == data.size(2))
    "Data is not a cube"
    assert len(window.size()) == 1
    "Window must be 1D"
    assert window.size(0) == data.size(0)
    "Window must match data size"

    windowed_projections = xyz_projections(data) * window.expand((3, window.size(0)))

    return complex_abs_sq(torch.fft.rfft(windowed_projections, dim=1))


def fourier_anisotropy_loss_pnorm(
    data: torch.Tensor, window: torch.Tensor, filter_ft: torch.Tensor, p: float = 2
) -> torch.Tensor:
    """
    Compute the XZ and YZ fourier losses
    (error between PSD on X compared to Z axis)
    with the provided windowing function and fourier domain filter
    """

    assert data.size(0) == data.size(1) and data.size(0) == data.size(2)
    "Data is not a cube"
    assert len(window.size()) == 1
    "Window must be 1D"
    assert window.size(0) == data.size(0)
    "Window must match data size"
    assert len(filter_ft.size()) == 1
    "Filter must be real 1D"
    assert len(filter_ft) == math.floor(data.size(0) / 2) + 1
    "Filter must match size of 1 sided real FT of the data"

    filtered_psd = windowed_projected_PSD(data, window) * filter_ft.expand(
        (3, filter_ft.size(0))
    )

    xy = filtered_psd[0:2, :]
    zz = filtered_psd[2, :].expand(0, filtered_psd.size(1))
    zz = filtered_psd[2, :].expand((2, filtered_psd.size(1)))

    return torch.sum(torch.pow(torch.abs(xy - zz), p), dim=1)


def hipass_gauss_kernel_fourier(sigma: float, N: int) -> torch.Tensor:
    """
    Make an unshifted Gaussian highpass filter in Fourier space. All real
    """

    centre = math.floor(N / 2)
    xs = torch.tensor(range(0, N), dtype=torch.float) - centre

    space_domain = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    space_domain /= sum(space_domain)

    fourier = torch.fft.rfft(space_domain, dim=0)

    return 1 - torch.abs(fourier)


###########
# STORAGE #
###########

# get the date
today = str(date.today())
# remove dashes
today = today.replace("-", "")
# path to data
path_data = os.path.join(os.getcwd(), "images/sims/")
# path to training samples (low-z-resolution, high-z-resolution)
path_lores = Path(os.path.join(path_data, Path("microtubules/lores")))
path_hires = Path(os.path.join(path_data, Path("microtubules/hires")))
# path to gneerated images - will make directory if there isn't one already
path_gens = os.path.join(path_data, Path("microtubules/generated"), today)
os.makedirs(path_gens, exist_ok=True)


#####################
# (HYPER)PARAMETERS #
#####################

# use gpu if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# learning rate
# for DCGAN, anecdotal recommendation = 3e-4
learning_rate = 3e-4
# batch size, i.e. #forward passes per backward propagation
size_batch = 10
# side length of (cubic) images
size_img = 96
# number of epochs i.e. number of times you re-use the same training images
n_epochs = 5
# channel depth of generator hidden layers in integers of this number
features_generator = 16
# channel depth of discriminator hidden layers in integers of this number
features_discriminator = 16
# the side length of the convolutional kernel in the network
kernel_size = 3
# pixel size in nm
size_pix_nm = 100.0
# z-resolution in the isotropic case
zres_hi = 240.0
# z-resolution in the anisotropic case
zres_lo = 600.0


##################
# LOAD IN IMAGES #
##################

# if loading in a whole dataset (not just one image)
# lores_data, hires_data = load_data("mtubs_sim_*_")

lores_file = Path(os.path.join(path_lores, "mtubs_sim_1_lores.tif"))
hires_file = Path(os.path.join(path_hires, "mtubs_sim_1_hires.tif"))

for file in lores_file, hires_file:

    img = io.imread(file)
    # convert to numpy array for faster calculations
    img = np.asarray(img)
    # normalise pixel values
    img = img / np.max(img)

    # convert to tensor
    img = torch.tensor(img)
    # now: img.shape = (z, x, y)
    img = torch.swapaxes(img, 1, 2)
    # now: img.shape = (z, y, x)

    if file == lores_file:
        lores_cpu = img
    elif file == hires_file:
        hires_cpu = img

lores_gpu = lores_cpu.to(device)
hires_gpu = hires_cpu.to(device)

##################
# IMPLEMENTATION #
##################

# sigma for real (lores) and isometric (hires) data
sig_lores = fwhm_to_sigma(zres_lo / size_pix_nm)
sig_hires = fwhm_to_sigma(zres_hi / size_pix_nm)
# this is how much worse the resolution is in the anisotropic data
sig_extra = math.sqrt(sig_lores ** 2 - sig_hires ** 2)

# they are arrays of dims [z, y, x]
hpf_ft = hipass_gauss_kernel_fourier(sig_lores, N=lores_gpu.size(-1)).to(device)
print(hpf_ft.shape)
sampling_window = blackman_harris_window(size_img).to(device)

z_kernel = gaussian_kernel(sig_extra, 3.0).to(device)

# starting point for the reconstruction is the square root of the data we want
reconstruction = torch.sqrt(lores_gpu.clone()).to(
    device
)  # TODO lores_gpu - every element is the same negative number??
reconstruction.requires_grad = True

optimiser = torch.optim.LBFGS([reconstruction], lr=1)

# there are only two for loops
# (because we are going to plot some things every 50 iterations)
for _ in range(1000):
    for i in range(50):
        print("------------------------------", i, "------------------------------")

        def closure():
            optimiser.zero_grad()
            # reco is the reconstruction
            # (squared so guaranteed to be positive)
            reco = reconstruction ** 2

            ###################
            # REAL SPACE LOSS #
            ###################
            """
            the real component of the loss is =0 when:
            img_original == z_blur(img_optimised)
            """

            # blur the optimised image in z
            rendering = conv_1D_z_axis(reco, z_kernel, pad="edge")

            # rendering loss is the direct loss on the image as compared to the blurred reconstruction
            rendering_loss = torch.sum((rendering - lores_gpu) ** 2)

            ######################
            # FOURIER SPACE LOSS #
            ######################
            """
            the fourier component of the loss is =0 (note that this is oversimplified) when:
            x_projection(fourier(img_original)) = y_projection(fourier(img_original)) = z_projection(fourier(img_optimised))
            z_projection(fourier(img_optimised)) gets a little closer to this outcome with each step
            """

            # fourier domain x/y projections for original data
            data_x_projection = lores_gpu.sum(1).sum(0)
            data_y_projection = lores_gpu.sum(2).sum(0)
            # fourier domain z projection for optimised image
            reco_z_projection = reco.sum(2).sum(1)
            # put them into a single object
            projections = torch.stack(
                [data_x_projection, data_y_projection, reco_z_projection], dim=0
            )

            # apply a window
            windowed_projections = projections * sampling_window.expand(
                (3, sampling_window.size(0))
            )

            power_spectra = complex_abs_sq(torch.fft.rfft(windowed_projections, dim=1))
            filtered_psd = power_spectra * hpf_ft.expand((3, hpf_ft.size(0)))
            xy = filtered_psd[:2, :]
            zz = filtered_psd[2, :].expand((2, filtered_psd.size(1)))

            fourier_loss = (
                torch.sum(torch.pow(torch.abs(torch.log(xy) - torch.log(zz)), 2), dim=1)
                * 4e5
            )

            # the loss is the difference between the log of the projections
            fourier_loss = torch.log(xy) - torch.log(zz)
            # take the absolute value to remove imaginary components, square, & sum to get the loss
            fourier_loss = torch.sum(torch.pow(torch.abs(fourier_loss), 2), dim=1)

            # rendering_loss.item()
            # == hires - lores; .item() method comverts from a torch to a number
            # fourier_loss.cpu.attach.numpy()
            # == [real part of fourier loss, imaginary part of fourier loss]
            print(rendering_loss.item(), fourier_loss.cpu().detach().numpy())

            # total loss
            loss = rendering_loss + sum(fourier_loss)
            # backpropagation to get the gradient
            loss.backward()

            return loss

        # gradient descent step
        optimiser.step(closure)

    reco = reconstruction ** 2
    recodata = reco.clone().detach()

    plt.clf()
    plt.subplot(2, 4, 1)
    proj_data = xyz_projections(lores_gpu).to("cpu")
    proj_recon = xyz_projections(recodata).to("cpu")
    plt.plot(proj_data[0, :], label="Data x")
    plt.plot(proj_data[1, :], label="Data y")
    plt.plot(proj_data[2, :], label="Data z")
    plt.plot(proj_recon[0, :], label="Recon x")
    plt.plot(proj_recon[1, :], label="Recon y")
    plt.plot(proj_recon[2, :], label="Recon z")
    plt.title("XYZ projections")

    plt.subplot(2, 4, 2)
    psd_data = windowed_projected_PSD(lores_gpu, sampling_window).to("cpu")
    psd_recon = windowed_projected_PSD(recodata.clone().detach(), sampling_window).to(
        "cpu"
    )
    plt.semilogy(psd_data[0, :], label="Data x")
    plt.semilogy(psd_data[1, :], label="Data y")
    plt.semilogy(psd_data[2, :], label="Data z")
    plt.semilogy(psd_recon[0, :], label="Recon x")
    plt.semilogy(psd_recon[1, :], label="Recon y")
    plt.semilogy(psd_recon[2, :], label="Recon z")
    plt.title("Fourier spectra")

    plt.subplot(2, 4, 3)
    plt.plot(0, label="Data x")
    plt.plot(0, label="Data y")
    plt.plot(0, label="Data z")
    plt.plot(0, label="Recon x")
    plt.plot(0, label="Recon y")
    plt.plot(0, label="Recon z")
    plt.legend()
    plt.axis(False)

    plt.subplot(2, 4, 5)
    plt.imshow(lores_cpu[int(size_img / 2), :, :])
    plt.subplot(2, 4, 6)
    plt.imshow(recodata[int(size_img / 2), :, :].to("cpu"))

    plt.subplot(2, 4, 7)
    plt.imshow(hires_gpu[int(size_img / 2), :, :])

    plt.subplot(2, 4, 8)
    plt.imshow(
        recodata[int(size_img / 2), :, :].to("cpu") - hires_gpu[int(size_img / 2), :, :]
    )

    plt.show(block=False)
    print(loss)

    for z in range(1, lores_gpu.size(0)):

        hires_orig = hires_gpu[z, :, :]
        lores_orig = lores_cpu[z, :, :]
        hires_new = recodata[z, :, :].cpu()
        res = torch.cat(
            (hires_orig, lores_orig, hires_new, (hires_orig - hires_new)), 1
        )
        imsave(Path("out") / "catted-{0:03d}.tiff".format(z), res.numpy())

    plt.waitforbuttonpress()


# plt.clf()
# x_only = data_normal.sum(1).sum(0) * blackman_harris_window(128)
# y_only = data_normal.sum(2).sum(0) * blackman_harris_window(128)
# z_only = data_normal.sum(2).sum(1) * blackman_harris_window(128)
# plt.subplot(1, 2, 1)
# plt.plot(x_only, label='x')
# plt.plot(y_only, label='y')
# plt.plot(z_only, label='z')
# plt.legend()


# What's a good fourier measure?

# Only care about high frequency stuff
# fta = complex_abs(torch.rfft(x_only,
#                              signal_ndim=1,
#                              normalized=False,
#                              onesided=True))
# plt.subplot(1, 2, 2)
# plt.semilogy((abs(np.fft.fft(x_only))), label='x')
# plt.semilogy(psd[0, :], label='xx')
# plt.semilogy((abs(np.fft.fft(y_only))), label='y')
# plt.semilogy(psd[1, :], label='yy')
# plt.semilogy((abs(np.fft.fft(z_only))), label='z')
# plt.semilogy(psd[2, :], label='zz')
# plt.semilogy(filter_ft, label='kern')
# plt.semilogy((fta), label='zz', LineWidth=1)
# plt.legend()
# plt.show(block=False)


# plt.clf();
# x = blackman_harris_window(128)
# plt.plot(x)
# plt.show(block=False)
