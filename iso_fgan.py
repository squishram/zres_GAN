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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from iso_fgan_dataloader import (
    FourierProjection,
    FourierProjectionLoss,
    Custom_Dataset,
)
from iso_fgan_networks import (
    Generator,
    initialise_weights,
)
from torch.utils.data import DataLoader
# from tifffile import imsave


def fwhm_to_sigma(fwhm: float):
    """Convert FWHM to standard deviation"""
    return fwhm / (2 * math.sqrt(2 * math.log(2)))


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
freq_domain_loss_scaler = 1000
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


########################################
# NETWORKS, LOSS FUNCTIONS, OPTIMISERS #
########################################

# sigma for real (lores) and isometric (hires) data
sig_lores = fwhm_to_sigma(zres_lo / size_pix_nm)
sig_hires = fwhm_to_sigma(zres_hi / size_pix_nm)
# this is how much worse the resolution is in the anisotropic data
sig_extra = math.sqrt(sig_lores ** 2 - sig_hires ** 2)

# this function can create filtered fourier projections
projector = FourierProjection(sig_lores)

# this is the generator - make sure it's ready for training
gen = Generator(features_gen, kernel_size, padding).to(device)
initialise_weights(gen)
gen.train()

# Adam optimiser is supposed to be the shit for generators
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
# MSELoss == (target - output) ** 2
criterion_mse = nn.MSELoss()
# FourierProjectionLoss ==
# log(filtered x,y-projection(fourier(image))) - log(filtered z-projection(fourier(image)))
criterion_ftp = FourierProjectionLoss()


##################
# IMPLEMENTATION #
##################

# step += 1 for every forward pass
step = 0
# this list contains the losses
loss_list = [[] for i in range(4)]

for epoch in range(n_epochs):

    for batch_idx, lores in enumerate(lores_dataloader):

        if step % 50 == 0:
            print("backpropagation count:", step)

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
        space_domain_loss = criterion_mse(spres, hires)

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

        # loss calculation
        freq_domain_loss = criterion_ftp(lores_xproj, lores_yproj, spres_zproj)

        ####################################
        # LOSS AGGREGATION, BACKPRPAGATION #
        ####################################

        # space_domain_loss.item()
        # = hires - lores; .item() converts from 0-dimensional tensor to a number
        # space_domain_loss.cpu.attach.numpy()
        # = [x part of fourier loss, y part of fourier loss]
        print(space_domain_loss.item(), freq_domain_loss.cpu().detach().numpy())

        # add the x and y components of the frequency domain loss
        freq_domain_loss = sum(freq_domain_loss)
        # scale the loss appropriately
        space_domain_loss *= space_domain_loss_scaler
        freq_domain_loss *= freq_domain_loss_scaler
        # total loss
        loss = space_domain_loss + freq_domain_loss

        # the zero grad thingy is come
        gen.zero_grad()
        # backpropagation to get the gradient
        loss.backward()
        # gradient descent step
        opt_gen.step()

        if step % 100 == 0:
            loss_list[0].append(int(step))
            loss_list[1].append(float(freq_domain_loss))
            loss_list[2].append(float(space_domain_loss))
            loss_list[3].append(float(loss))

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
            "\nsize_batch = " + str(size_batch),
            "\nsize_img = " + str(size_img),
            "\nn_epochs = " + str(n_epochs),
            "\nfeatures_gen= " + str(features_gen),
        ]
    )
# make sure to add more about the network structures!


# plot out all the losses for examination!
for i in range(len(loss_list) - 1):
    plt.plot(loss_list[0], loss_list[i])

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
