"""
Training of DCGAN network on MNIST dataset with Discriminator
and Generator imported from DCGAN_AP_funcs.py
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as tt
import torchvision.utils as utils
from upsampler_funcs import Generator, initialise_weights


# DATA #
# path_data - the path to the root of the dataset folder
path_data = os.path.join(os.getcwd(), "images/")
# make a directory for the generated images
dir_samples = Path("catface/generated/")
# this is the full path for the sample images
path_samples = os.path.join(path_data, dir_samples)
os.makedirs(path_samples, exist_ok=True)

# Hyperparameters etc.
# use gpu if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# learning rate for DCGAN
# paper recommendation - 2e-4
# anecdotal recommendation - 3e-4
learning_rate = 3e-4
# batch size, i.e. forward passes per backward propagation
size_batch = 128
# side length of images
size_img = 64
# number of colour channels (1=grayscale, 3=RGB)
n_colour_channels = 3
# number of epochs i.e. number of times you re-use the same training images in total
n_epochs = 5
# the channel depth of the hidden layers of the generator will be in integers of this number
features_generator = 64
n_workers = 2

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

# Create the dataset
dataset = dset.ImageFolder(path_data, transform=transform)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, size_batch, shuffle=True, num_workers=n_workers, pin_memory=True)

# A class object describes a format for an 'instance'
# e.g. we may have the class 'Tshirt(size)'
# and use it to make an XXL instance of that object, xtra_large_tee = Tshirt(XL)
# Thus to make use of a class, we use it to create instances
# In this case, the instances are the generator and discriminator networks
# this is the instance of the generator
gen = Generator(n_colour_channels, features_generator).to(device)
initialise_weights(gen)

# the optimiser uses
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
criterion = nn.L1Loss()

step = 0

# this sets the networks to training mode
# they should be by default, but it doesn't hurt to have assurance
gen.train()


data_iter = iter(dataloader)
first_images, labels = data_iter.next()
first_images = torch.tensor(first_images).to(device)

pool = nn.AvgPool2d(4, stride=2).to(device)

for epoch in range(n_epochs):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(dataloader):
        # 'real' are images from the dataset; send them to the GPU
        real = real.to(device)
        # a batch of latent noise from which generator makes the image
        # torch.randn(size_batch, size_noise, 1, 1) if latent vector
        # torch.randn(size_batch, size_noise, size_noise, 1) if latent square matrix
        downsampled = pool(real)
        # send the noise through the generator to make fake images
        # i.e. calculate G(z)
        upsampled = gen(downsampled)

        #############################################################
        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)) #
        #############################################################
        # the loss is the difference between the real and the upsampled image
        loss_gen = criterion(upsampled, real)
        # do the zero grad thing
        gen.zero_grad()
        # backpropagation to get the gradient
        loss_gen.backward()
        # take an appropriately sized step (gradient descent)
        opt_gen.step()

        # Print losses
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}] Batch {batch_idx}/{len(dataloader)} \
                  loss G: {loss_gen:.4f}")

        if (batch_idx % 500 == 0) or ((epoch == n_epochs - 1) and (batch_idx == len(dataloader) - 1)):
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



