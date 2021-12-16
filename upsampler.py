"""
A GAN that increases image resolution
Networks imported as classes from upsampler_funcs
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tt
import torchvision.utils as utils
from upsampler_funcs import Generator, Discriminator, initialise_weights
from datetime import date


# STORAGE #
# get the date
today = str(date.today())
today = today.replace('-', '')
# path_data - the path to the root of the dataset folder
path_data = os.path.join(os.getcwd(), "images/")
# this is the full path for the sample images
path_real = os.path.join(path_data, Path("catface/real/"))
path_gens = os.path.join(path_data, Path("catface/generated/"), today)
os.makedirs(path_gens, exist_ok=True)

# (HYPER)PARAMETERS #
# use gpu if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# learning rate for DCGAN
# paper recommendation - 2e-4
# anecdotal recommendation - 3e-4
learning_rate = 3e-4
# batch size, i.e. forward passes per backward propagation
size_batch = 128
# side length of (square) images
size_img = 64
# number of colour channels (1=grayscale, 3=RGB)
n_colour_channels = 3
# number of epochs i.e. number of times you re-use the same training images in total
n_epochs = 5
# the channel depth of the hidden layers of the generator will be in integers of this number
features_generator = 16
# the channel depth of the hidden layers of the discriminator will be in integers of this number
features_discriminator = 16
# the side length of the convolutional kernel in the network
kernel_size = 3
# the amount of padding needed to leave the image dimensions unchanged is given by the kernel size
# NOTE: only works if kernel_size % 2 == 1
if kernel_size % 2 == 1:
    padding = int((kernel_size - 1) / 2)
else:
    padding = int(kernel_size / 2)
# how much of the total dataset will be used for training?
# the 'test dataset' will be = 1 - train_portion
train_portion = 0.9


if n_colour_channels == 1:
    transform = tt.Compose([tt.Grayscale(),
                            tt.Resize(size_img),
                            tt.CenterCrop(size_img),
                            tt.ToTensor(),
                            tt.Normalize((0.5,), (0.5,)), ])
else:
    transform = tt.Compose([tt.Resize(size_img),
                            tt.CenterCrop(size_img),
                            tt.ToTensor(),
                            tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])


############################
# DATASETS AND DATALOADERS #
############################

# first pull out the whole dataset
dataset = dset.ImageFolder(path_real, transform=transform)
# split into training and testing datasets according to fraction train_portion
train_size = int(train_portion * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
# stick them into dataloaders for training and testing
trainloader = torch.utils.data.DataLoader(trainset, size_batch, shuffle=True, num_workers=2, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, size_batch, shuffle=True, num_workers=2, pin_memory=True)


#################################
# INITIALISE THE ACTUAL NETWORK #
#################################

# A class object describes a format for an 'instance'
# e.g. we may have the class 'T-shirt(size)'
# and make an XXL instance of that object, extra_large_tee = T-shirt(XXL)
# Thus to make use of a class, we use it to create instances
# In this case, the instances are the generator and discriminator networks
gen = Generator(n_colour_channels, features_generator, kernel_size, padding).to(device)
initialise_weights(gen)
dis = Discriminator(n_colour_channels, features_discriminator).to(device)
initialise_weights(dis)
# make the pooling function - this downsamples the original image
# nn.AvgPool2d(side length of pooling kernel, stride)
pool = nn.AvgPool2d(4, stride=4).to(device)

# the optimiser uses Adam to calculate the steps
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
opt_dis = optim.Adam(dis.parameters(), lr=learning_rate, betas=(0.5, 0.999))
# BCELoss() == Binary Cross Entropy Loss; L1Loss == target - output
criterion_bce = nn.BCELoss()
criterion_L1 = nn.L1Loss()

# this sets the networks to training mode
# they should be by default, but it doesn't hurt to have assurance
gen.train()
dis.train()

# pull out some test images to check then
data_iter = iter(testloader)
first_images, labels = data_iter.next()
first_images = first_images.to(device)

# step += 1 for every forward pass
step = 0
# this list contains the losses, [step, loss_dis_real, loss_dis_fake, loss_dis, loss_gen]
loss_list = [[] for i in range(7)]
for epoch in range(n_epochs):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(trainloader):
        # 'real' are images from the dataset; send them to the GPU
        real = real.to(device)
        # a batch of latent noise from which generator makes the image
        # torch.randn(size_batch, size_noise, 1, 1) if latent vector
        # torch.randn(size_batch, size_noise, size_noise, 1) if latent square matrix
        downsampled = pool(real)
        # send the noise through the generator to make fake images
        # i.e. calculate G(z)
        upsampled = gen(downsampled)

        ########################################################
        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))#
        ########################################################

        # pass the real images through the discriminator i.e. calculate D(x)
        # reshape(-1) ensures the output is in a single column
        dis_real = dis(real).reshape(-1)
        # obtain the loss by passing dis_real through the BCELoss function
        # the inclusion of the torch.ones_like() argument sets nn.BCELoss to be equal to: min(-log(D(x)))
        # (the first term of the 'Train Discriminator' equation above)
        loss_dis_real = criterion_bce(dis_real, torch.ones_like(dis_real))

        # pass the fake images through the discriminator i.e. calculate D(G(z))
        # reshape(-1) ensures the output is in a single column
        # tensor.detach() is used because loss.backward() clears the gradients from the fakes (G(z)) to save memory
        # but we want to reuse the fakes to calculate the generator loss, so .detach() preserves them
        dis_fake = dis(upsampled.detach()).reshape(-1)
        # obtain the loss by passing dis_real through the BCELoss function
        # the inclustion of the torch.zeros_like() argument sets nn.BCELoss to be equal to: min(-log(D(G(z)) - 1))
        # (the second term of the 'Train Discriminator' equation above)
        loss_dis_fake = criterion_bce(dis_fake, torch.zeros_like(dis_fake))

        # add the two components of the Discriminator loss
        loss_dis = loss_dis_real + loss_dis_fake
        # do the zero grad thing
        dis.zero_grad()
        # backpropagation to get the gradient
        loss_dis.backward()
        # take an appropriately sized step (gradient descent)
        opt_dis.step()

        ####################################
        # Train Generator: max log(D(G(z)) #
        ####################################

        # pass the fake images through the discriminator i.e. calculate D(G(z))
        # reshape(-1) ensures the output is in a single column
        output = dis(upsampled).reshape(-1)
        # obtain the loss by passing disc_real through the BCELoss function
        # the inclustion of the torch.ones_like() argument sets nn.BCELoss to be equal to: max(log(D(G(z)))
        # (the 'Train Generator' equation above)
        # torch.zeros_like(input) <-> torch.ones(input.size())
        loss_gen_bce = criterion_bce(output, torch.ones_like(output))
        # obtain the L1 loss as the difference between the real and upsampled
        loss_gen_L1 = criterion_L1(upsampled, real)
        # the total generator loss is the sum of these
        loss_gen = loss_gen_bce + loss_gen_L1
        # do the zero grad thing
        gen.zero_grad()
        # backpropagation to get the gradient
        loss_gen.backward()
        # take an appropriately sized step (gradient descent)
        opt_gen.step()

        if step % 100 == 0:
            loss_list[0].append(int(step))
            loss_list[1].append(float(loss_dis_fake))
            loss_list[2].append(float(loss_dis_real))
            loss_list[3].append(float(loss_dis))
            loss_list[4].append(float(loss_gen_bce))
            loss_list[5].append(float(loss_gen_L1))
            loss_list[6].append(float(loss_gen))

        # count the number of backpropagations
        step += 1

    # using the 'with' method in conjunction with no_grad() simply
    # disables grad calculations for the duration of the statement
    # Thus, we can use it to generate a sample set of images without initiating
    # a backpropagation calculation
    with torch.no_grad():
        downsampled = pool(first_images)
        upsampled = gen(downsampled)
        # denormalise the images so they look nice n crisp
        upsampled *= 0.5
        upsampled += 0.5
        # name your image grid according to which training iteration it came from
        fake_fname = 'generated_images_epoch-{0:0=2d}.png'.format(epoch + 1)
        # make a grid i.e. a sample of generated images to look at
        img_grid_fake = utils.make_grid(upsampled[:32], normalize=True)
        utils.save_image(upsampled, os.path.join(path_gens, fake_fname), nrow=8)
        # Print losses
        print(f"Epoch [{epoch + 1}/{n_epochs}] - saving {fake_fname}")


# make a metadata file
metadata = today + "_metadata.txt"
metadata = os.path.join(path_gens, metadata)
# make sure to remove any other metadata files in the subdirectory
if os.path.exists(metadata):
  os.remove(metadata)
# metadata = open(metadata, "a")
with open(metadata, "a") as file:
        file.writelines([os.path.basename(__file__),
                         "\nlearning_rate = " + str(learning_rate),
                         "\nsize_batch = " + str(size_batch),
                         "\nsize_img = " + str(size_img),
                         "\nn_colour_channels = " + str(n_colour_channels),
                         "\nn_epochs = " + str(n_epochs),
                         "\nfeatures_generator = " + str(features_generator),
                         "\nfeatures_discriminator = " + str(features_discriminator)])
# make sure to add more about the network structures!


# plot out all the losses for examination!
for i in range(len(loss_list) - 1):
    plt.plot(loss_list[0], loss_list[i])

plt.xlabel('Backpropagation Count')
plt.ylabel('Total Loss')
plt.legend(['loss_dis_real',
            'loss_dis_fake',
            'loss_dis',
            'loss_gen_bce',
            'loss_gen_L1',
            'loss_gen'],
            loc='upper left')

print("Saving loss graph...")
plt.savefig(os.path.join(path_gens, 'losses'), format='pdf')

print("Done!")
