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
3. remember that edge effects are due to convolutional layers (with padding)
   just remove the outer frames before displaying the images

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
    Custom_Dataset_Pairs,
    Generator,
    Discriminator,
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
path_data = os.path.join(os.getcwd(), Path("images/sims/microtubules"))
# subdirectories with lores and hires data
lores_subdir = "lores_test"
hires_subdir = "hires_test"
# path to training samples (low-z-resolution, high-z-resolution)
# path_lores = os.path.join(path_data, Path("microtubules/lores"))
# path_hires = os.path.join(path_data, Path("microtubules/hires"))
# path_lores = os.path.join(path_data, Path("microtubules/lores_test"))
# path_hires = os.path.join(path_data, Path("microtubules/hires_test"))
# path to gneerated images - will make directory if there isn't one already
path_gens = os.path.join(path_data, Path("generated"), today)
os.makedirs(path_gens, exist_ok=True)


#####################
# (HYPER)PARAMETERS #
#####################

# use gpu if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# learning rate
learning_rate = 1e-3
# relative scaling of the loss components (use 0 and 1 to see how well they do alone)
# combos that seem to kind of 'work': 1e-3 & 1e2
# for the generator:
freq_domain_loss_scaler = 1e-3
space_domain_loss_scaler = 1e2
adversary_gen_loss_scaler = 1
# for the discriminator:
loss_dis_real_scaler = 1e-4
loss_dis_fake_scaler = 1e-4
# batch size, i.e. #forward passes per backward propagation
batch_size = 1
# side length of (cubic) images
size_img = 96
# number of epochs i.e. number of times you re-use the same training images
n_epochs = 500
# after how many backpropagations do you generate a new image?
save_increment = 50
# channel depth of generator hidden layers in integers of this number
features_gen = 16
# channel depth of generator hidden layers in integers of this number
features_dis = 16
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
# lores_dataset = Custom_Dataset(dir_data=path_lores, filename="mtubs_sim_*.tif")
# hires_dataset = Custom_Dataset(dir_data=path_hires, filename="mtubs_sim_*.tif")
dataset = Custom_Dataset_Pairs(
    dir_data=path_data,
    subdirs=(lores_subdir, hires_subdir),
    filename="mtubs_sim_*.tif",
)

# image dataloaders when loading in hires and lores together
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# image dataloaders when loading in hires and lores separately
# lores_dataloader = DataLoader(
#     lores_dataset, batch_size=batch_size, shuffle=False, num_workers=2
# )
# hires_dataloader = DataLoader(
#     hires_dataset, batch_size=batch_size, shuffle=False, num_workers=2
# )

# pull out a single batch
data_iterator = iter(dataloader)
data_batch = next(data_iterator)
lores_batch = data_batch[:, 0, :, :, :, :].to(device)

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

# Discriminator Setup
dis = Discriminator(features_dis).to(device)
initialise_weights(dis)
dis.train()

# Loss and Optimisation
# bce loss for the adversarial battle
criterion_bce = nn.BCELoss()
# mean squared error loss
criterion_l1 = nn.L1Loss()
# fourier-transformed projection loss
criterion_ftp = FourierProjectionLoss()
# Adam optimiser is supposed to be 'the shit for GANs'
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
opt_dis = optim.Adam(dis.parameters(), lr=learning_rate, betas=(0.5, 0.999))


##################
# IMPLEMENTATION #
##################

# step += 1 for every forward pass
step = 0
# this list contains the losses
loss_list = [[] for i in range(8)]

for epoch in range(n_epochs):

    for batch_idx, data in enumerate(dataloader):

        # pull out the lores and hires images
        lores = data[:, 0, :, :, :, :].to(device=device, dtype=torch.float)
        hires = data[:, 1, :, :, :, :].to(device=device, dtype=torch.float)

        # pass the low-z-res images through the generator to make improved z-res images
        spres = gen(lores)

        ######################
        # DISCRIMINATOR LOSS #
        ######################
        """
        how good is the discriminator at not getting fooled by generator fakes?
        """

        # pass the real images through the discriminator i.e. calculate D(x)
        dis_real = dis(hires).reshape(-1)
        # how badly was it fooled?
        loss_dis_real = criterion_bce(dis_real, torch.ones_like(dis_real))

        # pass the generated image through the discriminator, D(G(z))
        dis_fake = dis(spres.detach()).reshape(-1)
        # "how well did the discriminator discern the generator's fakes"-loss
        loss_dis_fake = criterion_bce(dis_fake, torch.zeros_like(dis_fake))

        # add the two components of the Discriminator loss
        loss_dis = loss_dis_real + loss_dis_fake
        # do the zero grad thing
        dis.zero_grad()
        # backpropagation to get the gradient
        loss_dis.backward()
        # take an appropriately sized step (gradient descent)
        opt_dis.step()

        ###########################################################################################

        ##############################
        # GENERATOR ADVERSARIAL LOSS #
        ##############################
        """
        how well does the generator trick the discriminator?
        """

        # pass the generated image through the discriminator, D(G(z))
        output = dis(spres).reshape(-1)
        # "how well did the generator trick the discriminator?"-loss
        adversary_gen_loss = criterion_bce(output, torch.ones_like(output))

        ################################
        # GENERATOR SPACE DOMAIN LOSS #
        ################################
        """
        the real component of the loss is =0 when:
        img_hires == optimised(img_lores)
        """

        # space domain loss is simply hires - spres
        space_domain_loss = criterion_l1(spres, hires)

        ###################################
        # GENERATOR FREQUENCY DOMAIN LOSS #
        ###################################
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
        freq_domain_loss = criterion_ftp(lores_xproj, lores_yproj, spres_zproj)
        # freq_domain_loss = 0

        # add the x and y components of the frequency domain loss
        freq_domain_loss = sum(freq_domain_loss)

        ####################################
        # LOSS AGGREGATION, BACKPRPAGATION #
        ####################################

        # scale the loss appropriately
        adversary_gen_loss *= adversary_gen_loss_scaler
        space_domain_loss *= space_domain_loss_scaler
        freq_domain_loss *= freq_domain_loss_scaler
        # total loss
        loss_gen = space_domain_loss + freq_domain_loss + adversary_gen_loss

        # the zero grad thingy is come
        gen.zero_grad()
        # backpropagation to get the gradient
        loss_gen.backward()
        # gradient descent step
        opt_gen.step()

        # aggregate loss data
        if step % save_increment == 0:
            loss_list[0].append(int(step))
            loss_list[1].append(float(freq_domain_loss))
            loss_list[2].append(float(space_domain_loss))
            loss_list[3].append(float(adversary_gen_loss))
            loss_list[4].append(float(loss_gen))
            loss_list[5].append(float(loss_dis_real))
            loss_list[6].append(float(loss_dis_fake))
            loss_list[7].append(float(loss_dis))
            # using the 'with' method in conjunction with no_grad() simply
            # disables grad calculations for the duration of the statement
            # Thus, we can use it to generate a sample set of images without initiating
            # a backpropagation calculation
            with torch.no_grad():
                # pass low-z-res image through the generator
                genimg = gen(lores_batch)
                # pull out a single image
                genimg = genimg[0, 0, :, :, :].cpu().numpy()
                # TODO fix the images so you don't have to flip them and rotate 90 clockwise in imagej
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
    print(f"backpropagation count: {step}")
    print(f"Weighted Spatial Loss: {space_domain_loss.item()}")
    print(f"Weighted Fourier Loss: {freq_domain_loss.cpu().detach().numpy()}")
    print(f"Weighted 'GAN'-y Loss: {adversary_gen_loss.cpu().detach().numpy()}")


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
        "adversary generator loss",
        "total generator loss",
        "real discriminator loss",
        "fake discriminator loss",
        "total discriminator loss",
    ],
    loc="upper left",
)

print("Saving loss graph...")
plt.savefig(os.path.join(path_gens, "losses"), format="pdf")

print("Done!")
