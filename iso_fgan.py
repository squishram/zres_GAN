"""
this code represents a trainig methodology for a transductive GAN
the generator will take an image of low-z-resolution and make it isotropic

it combines a pixel-to-pixel comparison of:
    a low-z-resolution image and its isotropic equivalent
    a z-projection and an x/y projection of the same image, having undergone:
        a fourier tranform
        lobe removal
        a high-pass filter

there is no discriminator in this version of the network
it will use a pytorch dataloader
"""

import os
from pathlib import Path
from datetime import date
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from iso_fgan_dataloader import (
    FourierProjection,
    FourierProjectionLoss,
    Custom_Dataset,
    initialise_weights,
)
from iso_fgan_networks import Generator
from torch.utils.data import DataLoader

# from tifffile import imsave


def fwhm_to_sigma(fwhm: float):
    """Convert FWHM to standard deviation"""
    return fwhm / (2 * math.sqrt(2 * math.log(2)))


def blackman_harris_window(N: int):
    """
    Create a Blackman-Harris cosine window, a specific case of a cosine window
    https://en.wikipedia.org/wiki/Window_function#Blackman%E2%80%93Harris_window

    General cosine window: a sum of increasing cosines with alternating signs
    - N + 1 is the number of samples in the range [0 to 2pi]
    so N represents a discrete approximation of [0, 2pi)
    """

    # these are the coefficients that make it a blackman-harris window
    coeffs = [0.35875, 0.48829, 0.14128, 0.01168]

    x = torch.tensor(range(0, N)) * 2 * math.pi / N

    result = torch.zeros(N)

    for i, c in enumerate(coeffs):
        result += c * torch.cos(x * i) * ((-1) ** i)

    return result


def gaussian_kernel(sigma: float, sigmas: float = 3.0) -> torch.Tensor:
    """
    Make a normalized 1D Gaussian Kernel
    """

    radius = math.ceil(sigma * sigmas)
    xs = np.array(range(-radius, radius + 1))
    kernel = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    return torch.tensor(kernel / sum(kernel), dtype=torch.float)


def hipass_gauss_kernel_fourier(sigma: float, N: int) -> torch.Tensor:
    """
    Make an unshifted Gaussian highpass filter in Fourier space
    All real (i.e. not imaginary)
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
path_lores = os.path.join(path_data, Path("microtubules/lores"))
path_hires = os.path.join(path_data, Path("microtubules/hires"))
# path to gneerated images - will make directory if there isn't one already
path_gens = os.path.join(path_data, Path("microtubules/generated"), today)
os.makedirs(path_gens, exist_ok=True)


#####################
# (HYPER)PARAMETERS #
#####################

# use gpu if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# learning rate
learning_rate = 3e-4
# relative scaling of the loss components (use 0 and 1 to see how well they do alone)
freq_domain_loss_scaler = 400000
space_domain_loss_scaler = 1
# batch size, i.e. #forward passes per backward propagation
size_batch = 20
# side length of (cubic) images
size_img = 96
# number of epochs i.e. number of times you re-use the same training images
n_epochs = 5
# channel depth of generator hidden layers in integers of this number
features_gen = 16
# channel depth of discriminator hidden layers in integers of this number
# features_discriminator = 16
# the side length of the convolutional kernel in the network
kernel_size = 3
# padding when doing convolutions to ensure no change in image size
padding = int(kernel_size / 2)
# pixel size in nm
size_pix_nm = 100.0
# z-resolution in the isotropic case
zres_hi = 240.0
# z-resolution in the anisotropic case
zres_lo = 600.0

############################
# DATASETS AND DATALOADERS #
############################

# image datasets
lores_dataset = Custom_Dataset(dir_data=path_lores, filename="mtubs_sim_*_lores.tif")
hires_dataset = Custom_Dataset(dir_data=path_hires, filename="mtubs_sim_*_hires.tif")

# image dataloaders
lores_dataloader = DataLoader(lores_dataset, batch_size=5, shuffle=False, num_workers=2)
hires_dataloader = DataLoader(hires_dataset, batch_size=5, shuffle=False, num_workers=2)

# iterator objects from dataloaders
lores_iterator = iter(lores_dataloader)
hires_iterator = iter(hires_dataloader)

# pull out a single batch
lores_batch = lores_iterator.next()
hires_batch = hires_iterator.next()

# print the batch shape
print("low-z-res minibatch has dimensions: {}".format(lores_batch.shape))
print("high-z-res minibatch has dimensions: {}".format(hires_batch.shape))


###########################
# NETWORKS AND OPTIMISERS #
###########################

gen = Generator(features_gen, kernel_size, padding).to(device)
initialise_weights(gen)
gen.train()

# the optimiser uses Adam to calculate the steps
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
# MSELoss == (target - output)
criterion_mse = nn.MSELoss()
# FourierProjectionLoss ==
# log(filtered x,y-projection(fourier(image))) - log(filtered z-projection(fourier(image)))
criterion_ftp = FourierProjectionLoss()

##################
# IMPLEMENTATION #
##################

# sigma for real (lores) and isometric (hires) data
sig_lores = fwhm_to_sigma(zres_lo / size_pix_nm)
sig_hires = fwhm_to_sigma(zres_hi / size_pix_nm)
# this is how much worse the resolution is in the anisotropic data
sig_extra = math.sqrt(sig_lores ** 2 - sig_hires ** 2)

# NOTE:
# data_normal is the data made by load_cube(), a single low-z-res tiff stack
# data_hires is the data made by load_cube(), a single high-z-res tiff stack
# they are arrays of dims [z, y, x]
hpf_ft = hipass_gauss_kernel_fourier(sig_lores, N=hires_batch.size(-1)).to(device)
sampling_window = blackman_harris_window(size_img).to(device)
z_kernel = gaussian_kernel(sig_extra, 3.0).to(device)

# starting point for the reconstruction is the square root of the data we want
reconstruction = torch.sqrt(data_normal.clone()).to(device)
reconstruction.requires_grad = True

optimiser = torch.optim.LBFGS([reconstruction], lr=1)


# step += 1 for every forward pass
step = 0
# this list contains the losses
# [step, loss_dis_real, loss_dis_fake, loss_dis, loss_gen]
loss_list = [[] for i in range(7)]

for epoch in range(n_epochs):

    for batch_idx, (lores, _) in enumerate(lores_dataloader):

        hires, _ = next(iter(hires_dataloader))

        # send the batches of images to the gpu
        lores = lores.to(device)
        hires = hires.to(device)
        # pass the low-z-res images through the generator to make improved z-res images
        spres = gen(lores)

        #####################
        # SPACE DOMAIN LOSS #
        #####################
        """
        the real component of the loss is =0 when:
        img_hires == optimised(img_lores)
        """

        # space domain loss is simply (hires - spres)**2
        space_domain_loss = criterion_mse(spres, hires)
        space_domain_loss *= space_domain_loss_scaler

        #########################
        # FREQUENCY DOMAIN LOSS #
        #########################
        """
        the frequency, or fourier component of the loss is =0 (note that this is oversimplified) when:
        x_projection(fourier(img_original)) = y_projection(fourier(img_original)) = z_projection(fourier(img_optimised))
        z_projection(fourier(img_optimised)) gets a little closer to this outcome with each step
        """

        # fourier domain x/y projections for original data
        lores_xproj = lores.sum(1).sum(0)
        lores_yproj = lores.sum(2).sum(0)
        # fourier domain z projection for super-z-res image
        spres_zproj = spres.sum(2).sum(1)
        # put them into a single object
        projections = torch.stack([lores_xproj, lores_yproj, spres_zproj], dim=0)

        # apply a window
        windowed_projections = projections * sampling_window.expand(
            (3, sampling_window.size(0))
        )

        power_spectra = complex_abs_sq(torch.fft.rfft(windowed_projections, dim=1))

        filtered_psd = power_spectra * hpf_ft.expand((3, hpf_ft.size(0)))

        xy = filtered_psd[0:2, :]
        zz = filtered_psd[2, :].expand(0, filtered_psd.size(1))
        zz = filtered_psd[2, :].expand((2, filtered_psd.size(1)))

        freq_domain_loss = (
            torch.sum(torch.pow(torch.abs(torch.log(xy) - torch.log(zz)), 2), dim=1)
            * freq_domain_loss_scaler
        )

        # rendering_loss.item()
        # = hires - lores; .item() converts from 0-dimensional tensor to a number
        # fourier_loss.cpu.attach.numpy()
        # = [real part of fourier loss, imaginary part of fourier loss]
        print(space_domain_loss.item(), freq_domain_loss.cpu().detach().numpy())

        # scale the loss appropriately
        space_domain_loss *= space_domain_loss_scaler
        freq_domain_loss *= freq_domain_loss_scaler
        # total loss
        loss = space_domain_loss + freq_domain_loss

        # backpropagation to get the gradient
        loss.backward()
        # gradient descent step
        optimiser.step()

        if step % 100 == 0:
            loss_list[0].append(int(step))
            loss_list[4].append(float(freq_domain_loss))
            loss_list[5].append(float(space_domain_loss))
            loss_list[6].append(float(loss))

        # count the number of backpropagations
        step += 1

    # using the 'with' method in conjunction with no_grad() simply
    # disables grad calculations for the duration of the statement
    # Thus, we can use it to generate a sample set of images without initiating
    # a backpropagation calculation
    # with torch.no_grad():
    #     downsampled = pool(first_images)
    #     spres = gen(downsampled)
    #     # denormalise the images so they look nice n crisp
    #     spres *= 0.5
    #     spres += 0.5
    #     # name your image grid according to which training iteration it came from
    #     fake_fname = "generated_images_epoch-{0:0=2d}.png".format(epoch + 1)
    #     # make a grid i.e. a sample of generated images to look at
    #     img_grid_fake = utils.make_grid(spres[:32], normalize=True)
    #     utils.save_image(spres, os.path.join(path_gens, fake_fname), nrow=8)
    #     # Print losses
    #     print(f"Epoch [{epoch + 1}/{n_epochs}] - saving {fake_fname}")


# make a metadata file
# metadata = today + "_metadata.txt"
# metadata = os.path.join(path_gens, metadata)
# # make sure to remove any other metadata files in the subdirectory
# if os.path.exists(metadata):
#     os.remove(metadata)
# # metadata = open(metadata, "a")
# with open(metadata, "a") as file:
#     file.writelines(
#         [
#             os.path.basename(__file__),
#             "\nlearning_rate = " + str(learning_rate),
#             "\nsize_batch = " + str(size_batch),
#             "\nsize_img = " + str(size_img),
#             "\nn_epochs = " + str(n_epochs),
#             "\nfeatures_generator = " + str(features_generator),
#             "\nfeatures_discriminator = " + str(features_discriminator),
#         ]
#     )
# make sure to add more about the network structures!


# plot out all the losses for examination!
for i in range(len(loss_list) - 1):
    plt.plot(loss_list[0], loss_list[i])

plt.xlabel("Backpropagation Count")
plt.ylabel("Total Loss")
plt.legend(
    [
        "loss_dis_real",
        "loss_dis_fake",
        "loss_dis",
        "loss_gen_bce",
        "loss_gen_L1",
        "loss_gen",
    ],
    loc="upper left",
)

print("Saving loss graph...")
plt.savefig(os.path.join(path_gens, "losses"), format="pdf")

print("Done!")
