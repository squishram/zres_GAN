"""
Training of DCGAN network on MNIST dataset with Discriminator
and Generator imported from DCGAN_AP_funcs.py
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision
import torchvision.datasets as dset
import torchvision.transforms as tt
import torchvision.utils as utils
from upsampler_funcs import Generator, Discriminator, initialise_weights
from datetime import date


# STORAGE #
# get the date
today = str(date.today())
# path_data - the path to the root of the dataset folder
path_data = os.path.join(os.getcwd(), "images/")
# make a directory for the generated images
dir_samples = Path("catface/generated/")
# this is the full path for the sample images
path_samples = os.path.join(path_data, dir_samples, today)
os.makedirs(path_samples, exist_ok=True)

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
features_generator = 32
# the channel depth of the hidden layers of the disriminator will be in integers of this number
features_disriminator = 32
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
dataset = dset.ImageFolder(path_data, transform=transform)
# split into training and testing datasets according to fraction train_portion
train_size = int(train_portion * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
# stick them into dataloaders for training and testing
trainloader = torch.utils.data.DataLoader(trainset, size_batch, shuffle=True, num_workers=2, pin_memory=True, train=True)
testloader = torch.utils.data.DataLoader(testset, size_batch, shuffle=True, num_workers=2, pin_memory=True, train=False)


#################################
# INITIALISE THE ACTUAL NETWORK #
#################################

# A class object describes a format for an 'instance'
# e.g. we may have the class 'Tshirt(size)'
# and use it to make an XXL instance of that object, xtra_large_tee = Tshirt(XL)
# Thus to make use of a class, we use it to create instances
# In this case, the instances are the generator and disriminator networks
# this is the instance of the generator
gen = Generator(n_colour_channels, features_generator, kernel_size, padding).to(device)
initialise_weights(gen)
dis = Discriminator(n_colour_channels, features_disriminator, kernel_size).to(device)
initialise_weights(dis)

# make the pooling function - this downsamples the original image
# nn.AvgPool2d(side length of pooling kernel, stride)
pool = nn.AvgPool2d(4, stride=4).to(device)

# the optimiser uses Adam to calculate the steps
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
opt_dis = optim.Adam(dis.parameters(), lr=learning_rate, betas=(0.5, 0.999))
# BCELoss() == Binary Cross Entropy Loss
criterion = nn.BCELoss()

# this sets the networks to training mode
# they should be by default, but it doesn't hurt to have assurance
gen.train()
dis.train()

# pull out some test images to check then
data_iter = iter(testloader)
first_images, labels = data_iter.next()
first_images = first_images.to(device)

step = 0
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

        # pass the real images through the disriminator i.e. calculate D(x)
        # reshape(-1) ensures the output is in a single column
        dis_real = dis(real).reshape(-1)
        # obtain the loss by passing dis_real through the BCELoss function
        # the inclusion of the torch.ones_like() argument sets nn.BCELoss to be equal to: min(-log(D(x)))
        # (the first term of the 'Train Discriminator' equation above)
        loss_dis_real = criterion(dis_real, torch.ones_like(dis_real))

        # pass the fake images through the disriminator i.e. calculate D(G(z))
        # reshape(-1) ensures the output is in a single column
        # tensor.detach() is used because loss.backward() clears the gradients from the fakes (G(z)) to save memory
        # but we want to reuse the fakes to calculate the generator loss, so .detach() preserves them
        dis_fake = dis(upsampled.detach()).reshape(-1)
        # obtain the loss by passing dis_real through the BCELoss function
        # the inclustion of the torch.zeros_like() argument sets nn.BCELoss to be equal to: min(-log(D(G(z)) - 1))
        # (the second term of the 'Train Discriminator' equation above)
        loss_dis_fake = criterion(dis_fake, torch.zeros_like(dis_fake))

        # add the two components of the Discriminator loss
        loss_dis = loss_dis_real + loss_dis_fake
        # do the zero grad thing
        dis.zero_grad()
        # backpropagation to get the gradient
        loss_dis.backward()
        # take an appropriately sized step (gradient descent)
        opt_dis.step()

        #############################################################
        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)) #
        #############################################################

        # pass the fake images through the discriminator i.e. calculate D(G(z))
        # reshape(-1) ensures the output is in a single column
        output = dis(upsampled).reshape(-1)
        # obtain the loss by passing disc_real through the BCELoss function
        # the inclustion of the torch.ones_like() argument sets nn.BCELoss to be equal to: max(log(D(G(z)))
        # (the 'Train Generator' equation above)
        # torch.zeros_like(input) <-> torch.ones(input.size())
        loss_gen = criterion(output, torch.ones_like(output))
        # do the zero grad thing
        gen.zero_grad()
        # backpropagation to get the gradient
        loss_gen.backward()
        # take an appropriately sized step (gradient descent)
        opt_gen.step()

        # Print losses
        if batch_idx % (len(trainloader) / 2) == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}] Batch {batch_idx}/{len(trainloader)} \
                  Loss D: {loss_dis:.4f}, loss G: {loss_gen:.4f}")

        if (batch_idx % 500 == 0) or ((epoch == n_epochs - 1) and (batch_idx == len(trainloader) - 1)):
            # using the 'with' method in conjunction with no_grad() simply
            # disables grad calculations for the duration of the statement
            # Thus, we can use it to generate a sample set of images without initiating a backpropagation step
            with torch.no_grad():
                downsampled = pool(first_images)
                upsampled = gen(downsampled)
                # denormalise the images so they look nice n crisp
                upsampled *= 0.5
                upsampled += 0.5
                # name your image grid according to which training iteration it came from
                fake_fname = 'generated_images-{0:0=4d}.png'.format(epoch + step + 1)
                # make a grid i.e. a sample of generated images to look at
                img_grid_fake = utils.make_grid(upsampled[:32], normalize=True)
                utils.save_image(upsampled, os.path.join(path_samples, fake_fname), nrow=8)
                print('Saving', fake_fname)

            step += 1

# END OF THE CODE #
# THANKS FOR WATCHING #
