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

CURRENT ISSUES
1. the projections are a single dimension large
shouldn't a projection of a 3D image have two dimensions?
2. the loss function doesn't take the batch or channel dimensions into account

"""

import os
from pathlib import Path
from datetime import date
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from iso_fgan_functions import (
    FourierProjection,
    FourierProjectionLoss,
    Custom_Dataset,
    Generator,
    initialise_weights,
)
from torch.utils.data import DataLoader
# import torchio.transforms as transforms
# import torchio as tio
from tifffile import imwrite


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
# path_lores = os.path.join(path_data, Path("microtubules/lores"))
# path_hires = os.path.join(path_data, Path("microtubules/hires"))
path_lores = os.path.join(path_data, Path("microtubules/lores_test"))
path_hires = os.path.join(path_data, Path("microtubules/hires_test"))
# path to gneerated images - will make directory if there isn't one already
path_gens = os.path.join(path_data, Path("microtubules/generated"), today)
os.makedirs(path_gens, exist_ok=True)


#####################
# (HYPER)PARAMETERS #
#####################

# use gpu if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# learning rate
learning_rate = 1e-3
# relative scaling of the loss components (use 0 and 1 to see how well they do alone)
freq_domain_loss_scaler = 0
space_domain_loss_scaler = 1
# batch size, i.e. #forward passes per backward propagation
batch_size = 1
# side length of (cubic) images
size_img = 96
# number of epochs i.e. number of times you re-use the same training images
n_epochs = 100
# channel depth of generator hidden layers in integers of this number
features_gen = 16
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
# TODO change the dataloader so these are loaded in pairs of images!
# lores_dataset = Custom_Dataset(dir_data=path_lores, filename="mtubs_sim_*_lores.tif")
# hires_dataset = Custom_Dataset(dir_data=path_hires, filename="mtubs_sim_*_hires.tif")
lores_dataset = Custom_Dataset(dir_data=path_lores, filename="mtubs_sim_*_lores.tif")
hires_dataset = Custom_Dataset(dir_data=path_hires, filename="mtubs_sim_*_hires.tif")

# image dataloaders
lores_dataloader = DataLoader(
    lores_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)
hires_dataloader = DataLoader(
    hires_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

# pull out a single batch
lores_iterator = iter(lores_dataloader)
lores_batch = lores_iterator.next().to(device)

########################################
# NETWORKS, LOSS FUNCTIONS, OPTIMISERS #
########################################

# sigma for real (lores) and isometric (hires) data
sig_lores = (zres_lo / size_pix_nm) / (2 * math.sqrt(2 * math.log(2)))
sig_hires = (zres_hi / size_pix_nm) / (2 * math.sqrt(2 * math.log(2)))

# this function can create filtered fourier projections
projector = FourierProjection(sig_lores)

# Generator Setup
gen = Generator(features_gen, kernel_size, padding).to(device)
initialise_weights(gen)
gen.train()

# Loss and Optimisation
# mean squared error loss
criterion_l1 = nn.L1Loss()
# fourier-transformed projection loss
criterion_ftp = FourierProjectionLoss()
# Adam optimiser is supposed to be 'the shit for generators'
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))


##################
# IMPLEMENTATION #
##################

# step += 1 for every forward pass
step = 0
# this list contains the losses
loss_list = [[] for i in range(4)]

for epoch in range(n_epochs):

    for batch_idx, lores in enumerate(lores_dataloader):

        # iterate through the hi-res versions of the image as well!
        hires = next(iter(hires_dataloader))

        # send the batches of images to the gpu
        lores = lores.to(device=device, dtype=torch.float)
        hires = hires.to(device=device, dtype=torch.float)

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
        # space_domain_loss = criterion_mse(spres, hires).to(device)
        space_domain_loss = criterion_l1(spres, hires)

        #########################
        # FREQUENCY DOMAIN LOSS #
        #########################
        """
        the frequency, or fourier component of the loss is =0 (note that this is oversimplified) when:
        x_projection(fourier(img_original)) = y_projection(fourier(img_original)) = z_projection(fourier(img_optimised))
        z_projection(fourier(img_optimised)) gets a little closer to this outcome with each step
        """

        # fourier transform, projection, window, hipass filter
        # ... for x dimension of original image
        lores_xproj = projector(lores, 0)
        # ... for y dimension of original image
        lores_yproj = projector(lores, 1)
        # ... for z dimension of generated image
        spres_zproj = projector(spres, 2)
        # dims are [batch, 1, 49] for 96**3 shape images

        # the z-projections comes from the generated image so must be backpropagation-sensitive
        # not the case with the x/y-projections
        # lores_xproj = lores_xproj.no_grad().to(device)
        # lores_yproj = lores_yproj.no_grad().to(device)
        # spres_zproj = spres_zproj.requires_grad_(True).to(device)

        # loss calculation
        # freq_domain_loss = criterion_ftp(lores_xproj, lores_yproj, spres_zproj)
        freq_domain_loss = 0

        ####################################
        # LOSS AGGREGATION, BACKPRPAGATION #
        ####################################

        # add the x and y components of the frequency domain loss
        # freq_domain_loss = sum(freq_domain_loss)
        # scale the loss appropriately
        space_domain_loss *= space_domain_loss_scaler
        # freq_domain_loss *= freq_domain_loss_scaler
        # total loss
        loss = space_domain_loss + freq_domain_loss

        # the zero grad thingy is come
        gen.zero_grad()
        # backpropagation to get the gradient
        loss.backward()
        # gradient descent step
        opt_gen.step()

        # aggregate loss data
        if step % 50 == 0:
            loss_list[0].append(int(step))
            loss_list[1].append(float(freq_domain_loss))
            loss_list[2].append(float(space_domain_loss))
            loss_list[3].append(float(loss))
            # using the 'with' method in conjunction with no_grad() simply
            # disables grad calculations for the duration of the statement
            # Thus, we can use it to generate a sample set of images without initiating
            # a backpropagation calculation
            with torch.no_grad():
                # pass low-z-res image through the generator
                genimg = gen(lores_batch)
                # pull out a single image
                genimg = genimg[0, 0, :, :, :].cpu().numpy()
                # genimg = np.flipud(genimg)
                # genimg = np.rot90(genimg)
                # name your image grid according to which training iteration it came from
                genimg_name = "generated_images_epoch{0:0=2d}.tif".format(epoch + 1)
                print(f"Epoch [{epoch + 1}/{n_epochs}] - saving {genimg_name}")
                # save the sample image
                imwrite(os.path.join(path_gens, genimg_name), genimg)

        # count the number of backpropagations
        step += 1

    # print the loss after each epoch
    # space_domain_loss.item()
    # = hires - lores; .item() converts from 0-dimensional tensor to a number
    # space_domain_loss.cpu.attach.numpy()
    # = [x part of fourier loss, y part of fourier loss]
    print("backpropagation count:", step)
    print(f"Weighted Spatial Loss: {space_domain_loss.item()}")
    # print(f"Weighted Fourier Loss: {freq_domain_loss.cpu().detach().numpy()}")


# make a metadata file
metadata = today + "_metadata.txt"
metadata = os.path.join(path_gens, metadata)
# make sure to remove any other metadata files in the subdirectory
if os.path.exists(metadata):
    os.remove(metadata)
# metadata = open(metadata, "a")
with open(metadata, "a") as file:
    file.writelines(
        [
            os.path.basename(__file__),
            "\nlearning_rate = " + str(learning_rate),
            "\nsize_batch = " + str(batch_size),
            "\nsize_img = " + str(size_img),
            "\nn_epochs = " + str(n_epochs),
            "\nfeatures_gen= " + str(features_gen),
        ]
    )
# TODO make sure to add more about the network structures!

# plot out all the losses:
for i in range(1, len(loss_list)):
    plt.plot(loss_list[0], loss_list[i])
# legend
plt.xlabel("Backpropagation Count")
plt.ylabel("Total Loss")
plt.legend(
    [
        "frequency domain loss",
        "space domain loss",
        "total loss",
    ],
    loc="upper left",
)

print("Saving loss graph...")
plt.savefig(os.path.join(path_gens, "losses"), format="pdf")

print("Done!")
