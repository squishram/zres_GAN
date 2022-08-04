"""
pseudo-code for new affirm3d, which does not use the ground truth to calculate the real-space loss:

Generated image flow:
    I: input: a batch of images that are undersampled, and have reduced resolution, in z (w.r.t. x & y)
    --> put into upsampler to equalise sampling across all 3 dimensions
    U: upsampled: a batch of images that have reduced resolution in z (w.r.t. x & y)
    --> put into generator network to improve resolution in z
    G: generated: a batch of images that (need to, with training) have isometric resolution in x, y, & z
    --> put into downsampler to match sampling of the input image
    O: output: a batch of images that are undersampled in z (w.r.t. x & y) but have isometric resolution in x, y, & z

Loss for generator training:
    1. l1 real-space loss - the sum of the signal difference, pixel-to-pixel, between O and I
    2. fourier-space loss - the difference in the power spectra of the z-projection of G and the x & y-projections of I
    3. adversarial loss - calculated from the ability of the generator to ensure that small cross-sectional volumes of O are mistaken by the discriminator for small cross-sectional volumes of I

Loss for discriminator training:
    1. discriminator positive loss - calculated from the ability of the network to recognise that small cross-sectional volumes of O do not belong to I
    2. discriminator negative loss - calculated from the ability of the network to recognise that small cross-sectional volumes of I do not belong to O
"""

import os
from pathlib import Path
from datetime import date
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as tf
import torch.optim as optim
from affirm3d_noGT_functions import (
    FourierProjection,
    Custom_Dataset,
    Generator,
    Discriminator,
    gaussian_kernel,
    conv_1D_z_axis,
    fourier_loss,
    initialise_weights,
)
from torch.utils.data import DataLoader
from tifffile import imwrite
from time import perf_counter


start_time = perf_counter()

###########
# STORAGE #
###########

# path to data
path_data = os.path.join(os.getcwd(), Path("images/sims/microtubules/noGT/"))
# glob of filnames
filename = "mtubs_sim_noGT_*.tif"

# path to generated images - will make directory if there isn't one already
# get the date
today = str(date.today())
# remove dashes
today = today.replace("-", "")
path_gens = os.path.join(os.getcwd(), Path("images/sims/microtubules/generated/"), today)
os.makedirs(path_gens, exist_ok=True)

# path to saved networks (for retrieval/ testing)
path_models = os.path.join(path_gens, Path("models"))
os.makedirs(path_gens, exist_ok=True)


#####################
# (HYPER)PARAMETERS #
#####################

# use gpu if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# learning rate
learning_rate = 1e-3
# relative scaling of the loss components (use 0 and 1 to see how well they do alone)
# combos that seem to kind of 'work': 1, 1, 1 & 1, 1
# for the generator:
freq_domain_loss_scaler = 1
space_domain_loss_scaler = 1
adversary_gen_loss_scaler = 1
# for the discriminator:
loss_dis_real_scaler = 1
loss_dis_fake_scaler = 1
# batch size, i.e. #forward passes per backpropagation
batch_size = 5
# number of epochs i.e. number of times you re-use the same training images
n_epochs = 10
# after how many backpropagations do you generate a new image?
save_increment = 50
# channel depth of generator hidden layers in integers of this number
features_gen = 16
# channel depth of discriminator hidden layers in integers of this number
features_dis = 16
# the side length of the convolutional kernel in the network
kernel_size = 3
# the stride of the kernel responsible for downsampling the generated image
stride_downsampler = 3
# padding when doing convolutions to ensure no change in image size
padding = int(kernel_size / 2)
# NOTE: These nm bits used to be 10x higher, but I have changed them to the values they have in the simulated data generator
# pixel size in nm
size_pix_nm = 10.0
# z-resolution in the isotropic case
zres_hi = 24.0
# z-resolution in the anisotropic case
zres_lo = 60.0
# windowing function: can be hann, hamming, bharris
window = "hamming"

##########################
# DATASET AND DATALOADER #
##########################

# image datasets
dataset = Custom_Dataset(
    dir_data=path_data,
    filename=filename,
    transform=None,
)

# image dataloaders when loading in hires and lores together
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

########################################
# NETWORKS, LOSS FUNCTIONS, OPTIMISERS #
########################################

# sigma for real (lores) and isomorphic (hires) data
sig_lores = (zres_lo / size_pix_nm) / (2 * math.sqrt(2 * math.log(2)))
sig_hires = (zres_hi / size_pix_nm) / (2 * math.sqrt(2 * math.log(2)))
# sig_extra is derived from the formula of convolving 2 gaussians, where it is defined as:
# gaussian(sigma=sig_lores) *convolve* gaussian(sigma=sig_extra) = gaussian(sigma=sig_extra)
sig_extra = math.sqrt(sig_lores**2 - sig_hires**2)

# this function can create filtered fourier projections
# fields: z_sigma, window type
projector = FourierProjection(sig_lores, window)

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
# criterion_ftp = FourierProjectionLoss()
# Adam optimiser is supposed to be 'the shit for GANs'
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
opt_dis = optim.Adam(dis.parameters(), lr=learning_rate, betas=(0.5, 0.999))

##############
# TEST IMAGE #
##############

# iterable from the dataloader
data_iterator = iter(dataloader)
# pull out a single batch of data
data_batch = next(data_iterator).to(device)
# from the batch, pull out a single imag
img_sample = data_batch[0, 0, :, :, :].cpu().numpy()
# how big are these images?
size_img = img_sample.shape

# save the inputs (hires & lores) that make the generated image so we can compare fairly
img_name = "input_img.tif"
imwrite(os.path.join(path_gens, img_name), img_sample)


##################
# IMPLEMENTATION #
##################

# step += 1 for every forward pass
step = 0
# this list contains the losses (to be plotted)
loss_list = [[] for _ in range(8)]
# this list contains the fourier spectra (to be plotted)
fourier_list = [[] for _ in range(3)]

for epoch in range(n_epochs):

    for batch_idx, data in enumerate(dataloader):

        # pull out the input image batch (I)
        input_batch = data.to(device=device, dtype=torch.float)
        # upsample the input image batch (U)
        upsampled_batch = tf.interpolate(
            input_batch,
            size=(input_batch.shape[-1], input_batch.shape[-1], input_batch.shape[-1]),
            mode="trilinear",
            # align_corners=True,
            # recompute_scale_factor=None,
            # antialias=False,
        )
        # pass U through the generator to get a generated image batch (G)
        gen_batch = gen(upsampled_batch)
        # pass G though the covolutional layer to get the output image batch (O) which should look the same as I
        output_batch = conv_1D_z_axis(
            upsampled_batch, gaussian_kernel(sig_extra, 6.0), stride_downsampler
        )

        ######################
        # DISCRIMINATOR LOSS #
        ######################
        """
        how good is the discriminator at recognising real and fake images?
        """

        # pass the I through the discriminator i.e. calculate D(I)
        dis_real = dis(input_batch).reshape(-1)
        # how badly was it fooled?
        loss_dis_real = criterion_bce(dis_real, torch.ones_like(dis_real))

        # pass the generated image through the discriminator, D(O)
        dis_fake = dis(output_batch.detach()).reshape(-1)
        # "how well did the discriminator discern the generator's fakes"-loss
        loss_dis_fake = criterion_bce(dis_fake, torch.zeros_like(dis_fake))

        # add the two components of the Discriminator loss
        loss_dis = (loss_dis_real * loss_dis_real_scaler) + (
            loss_dis_fake * loss_dis_fake_scaler
        )
        # do the zero grad thing
        opt_dis.zero_grad()
        # backpropagation to get the gradient
        loss_dis.backward()
        # take an appropriately sized step (gradient descent)
        opt_dis.step()

        ##############################
        # GENERATOR ADVERSARIAL LOSS #
        ##############################
        """
        how well does the generator trick the discriminator?
        """

        # pass the generated image through the discriminator, D(G(z))
        output = dis(output_batch).reshape(-1)
        # "how well did the generator trick the discriminator?"-loss
        adversary_gen_loss = criterion_bce(output, torch.ones_like(output))

        ################################
        # GENERATOR SPACE DOMAIN LOSS #
        ################################
        """
        the real component of the loss is =0 when:
        img_hires == optimised(img_lores)
        """

        # space domain loss is simply I - O
        space_domain_loss = criterion_l1(output_batch, input_batch)

        ###################################
        # GENERATOR FREQUENCY DOMAIN LOSS #
        ###################################
        """
        the frequency, or fourier component of the loss is =0 (note that this is oversimplified) when:
        fourier(x_proj(img_original)) = fourier(y_proj(img_original)) = fourier(z_pro(img_optimised))
        z_projection(fourier(img_optimised)) gets a little closer to this outcome with each step
        """

        # fourier transform, projection, window, hipass filter
        # ... for x dimension of original image
        input_xpojection = projector(input_batch, 0)
        # ... for y dimension of original image
        input_yprojection = projector(input_batch, 1)
        # ... for z dimension of generated image
        output_zprojection = projector(gen_batch, 2)

        # calculate the power spectral loss ('fourier loss') from the projections
        freq_domain_loss = fourier_loss(
            input_xpojection, input_yprojection, output_zprojection
        )
        # freq_domain_loss = 0

        # add the x and y components of the frequency domain loss
        freq_domain_loss = sum(freq_domain_loss)

        ####################################
        # LOSS AGGREGATION, BACKPRPAGATION #
        ####################################

        # scale the loss appropriately
        # adversary_gen_loss *= adversary_gen_loss_scaler
        space_domain_loss *= space_domain_loss_scaler
        freq_domain_loss *= freq_domain_loss_scaler
        # total loss
        loss_gen = space_domain_loss + freq_domain_loss
        # loss_gen = space_domain_loss + freq_domain_loss + adversary_gen_loss

        # the zero grad thingy is come
        opt_gen.zero_grad()
        # backpropagation to get the gradient
        loss_gen.backward()
        # gradient descent step
        opt_gen.step()

        if step % save_increment == 0:
            # aggregate loss data
            loss_list[0].append(int(step))
            # generator loss: frequency domain
            loss_list[1].append(float(freq_domain_loss))
            # generator loss: signal domain
            loss_list[2].append(float(space_domain_loss))
            # generator loss: adversarial
            # loss_list[3].append(float(adversary_gen_loss))
            # generator loss: total
            loss_list[4].append(float(loss_gen))
            # discriminator loss: detecting real isomorphic images
            # loss_list[5].append(float(loss_dis_real))
            # discriminator loss: detecting fake (generated) isomorphic images
            # loss_list[6].append(float(loss_dis_fake))
            # discriminator loss: total
            # loss_list[7].append(float(loss_dis))

        # count the number of backpropagations
        step += 1

    # using the 'with' method in conjunction with no_grad() simply
    # disables grad calculations for the duration of the statement
    # Thus, we can use it to generate a sample set of images without initiating
    # a backpropagation calculation
    with torch.no_grad():
        # pass low-z-res batch through the generator
        gen_sample = gen(input_batch)
        # pull out a single image
        gen_sample = gen_sample[0, 0, :, :, :].cpu().numpy()
        # name your image grid according to which training iteration it came from
        gen_sample_name = "generated_images_epoch{0:0=2d}.tif".format(epoch + 1)
        print(f"Epoch [{epoch + 1}/{n_epochs}] - saving {gen_sample_name}")
        # save the sample image
        imwrite(os.path.join(path_gens, gen_sample_name), gen_sample)

    # get fourier power spectra...
    # for the x projection
    fourier_list[0].append(input_xpojection[0, 0].cpu().detach().numpy())
    # for the y projection
    fourier_list[1].append(input_yprojection[0, 0].cpu().detach().numpy())
    # for the z projection
    fourier_list[2].append(output_zprojection[0, 0].cpu().detach().numpy())

    # print(np.array(fourier_list).shape)
    # print(fourier_list)

    # print the loss after each epoch
    print(f"backpropagation count: {step}")
    print(f"Weighted Spatial Loss: {space_domain_loss.item()}")
    print(f"Weighted Fourier Loss: {freq_domain_loss}")
    # print(f"Weighted 'GAN'-y Loss: {adversary_gen_loss.cpu().detach().numpy()}")
    print(f"Generator Total  Loss: {loss_gen.cpu().detach().numpy()}")
    # print(f"Discriminator fk Loss: {loss_dis_fake}")
    # print(f"Discriminator rl Loss: {loss_dis_real}")
    # print(f"Discriminator tt Loss: {loss_dis}")


############
# METADATA #
############

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
            "\nfreq_domain_loss_scaler= " + str(freq_domain_loss_scaler),
            "\nspace_domain_loss_scaler= " + str(space_domain_loss_scaler),
            "\nadversary_gen_loss_scaler= " + str(adversary_gen_loss_scaler),
            "\nloss_dis_real_scaler= " + str(loss_dis_real_scaler),
            "\nloss_dis_fake_scaler = " + str(loss_dis_fake_scaler),
        ]
    )
# TODO make sure to add more about the network structures!

###########################
# PLOT THE GENERATOR LOSS #
###########################

plt.figure(0)
# plot out all the losses:
for i in range(1, 5):
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
    ],
    loc="upper right",
)

print("Saving generator loss graph...")
plt.savefig(os.path.join(path_gens, "generator losses"), format="pdf")

###############################
# PLOT THE DISCRIMINATOR LOSS #
###############################

plt.figure(1)
# plot out all the losses:
for i in range(5, len(loss_list)):
    # plt.plot(loss_list[0], loss_list[i])
    plt.semilogy(loss_list[0], loss_list[i])
# legend
plt.xlabel("Backpropagation Count")
plt.ylabel("Total Loss")
plt.legend(
    [
        "real discriminator loss",
        "fake discriminator loss",
        "total discriminator loss",
    ],
    loc="upper right",
)

print("Saving discriminator loss graph...")
plt.savefig(os.path.join(path_gens, "discriminator losses"), format="pdf")

###########################################
# PLOT THE PROFILE OF THE FOURIER SPECTRA #
###########################################

# convert to numpy array for faster calculations
fourier_list = np.array(fourier_list)
mean_fourier_spectra = np.mean(fourier_list, axis=1)
error_bars = np.std(fourier_list, axis=1)

for i in range(fourier_list.shape[1]):
    plt.figure(i + 2)
    for j in range(fourier_list.shape[0]):
        plt.plot(range(fourier_list.shape[2]), fourier_list[j, i])

    plt.xlabel("Frequency")
    plt.ylabel("Signal (AU)")
    plt.legend(
        [
            "low-res x power spectrum",
            "low-res y power spectrum",
            "super-res z power spectrum",
        ],
        loc="upper right",
    )

    plt.savefig(
        os.path.join(path_gens, f"fourier_projections_{i}"),
        format="pdf",
    )


# save the network parameters
torch.save(gen.state_dict(), os.path.join(path_models))
# torch.save(dis.state_dict(), os.path.join(path_models))

end_time = perf_counter()

print(f"Training took {start_time - end_time} seconds")
print(f"Training took {(start_time - end_time) / 60} minutes")
print(f"Training took {(start_time - end_time) / 3600} hours")

print("Done!")
