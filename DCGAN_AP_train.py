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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from DCGAN_AP_funcs import Discriminator, Generator, initialise_weights, test


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
# side length of input and output images
size_img = 64
# number of colour channels (1=grayscale, 3=RGB)
n_colour_channels = 1
# length of the noise vector
size_noise = 100
# number of epochs i.e. number of times you re-use the same training images in total
n_epochs = 5
#
features_discriminator = 64
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

gen = Generator(size_noise, n_colour_channels, features_generator).to(device)
disc = Discriminator(n_colour_channels, features_discriminator).to(device)
initialise_weights(gen)
initialise_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, size_noise, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

# this sets the networks to training mode
# they should be by default, but it doesn't hurt to have assurance
gen.train()
disc.train()

for epoch in range(n_epochs):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(dataloader):
        # 'real' are images from the dataset; send them to the GPU
        real = real.to(device)
        # a batch of latent noise from which generator makes the image
        noise = torch.randn(size_batch, size_noise, 1, 1).to(device)
        # send the noise through the generator to make fake images
        # i.e. calculate G(z)
        fake = gen(noise)

        ########################################################
        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))#
        ########################################################

        # pass the real images through the discriminator i.e. calculate D(x)
        # reshape(-1) ensures the output is in a single column
        disc_real = disc(real).reshape(-1)
        # obtain the loss by passing disc_real through the BCELoss function
        # the inclustion of the torch.ones_like() argument sets nn.BCELoss to be equal to: min(-log(D(x)))
        # (the first term of the 'Train Discriminator' equation above)
        # torch.ones_like(input) <-> torch.ones(input.size())
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

        # pass the fake images through the discriminator i.e. calculate D(G(z))
        # reshape(-1) ensures the output is in a single column
        # tensor.detach() is used because loss.backward() clears the gradients from the fakes (G(z)) to save memory
        # but we want to reuse the fakes to calculate the generator loss, so .detach() preserves them
        disc_fake = disc(fake.detach()).reshape(-1)
        # obtain the loss by passing disc_real through the BCELoss function
        # the inclustion of the torch.zeros_like() argument sets nn.BCELoss to be equal to: min(-log(D(G(z)) - 1))
        # (the second term of the 'Train Discriminator' equation above)
        # torch.zeros_like(input) <-> torch.zeros(input.size())
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        # add the two components of the Discriminator loss together
        # dividing by 2 (i.e getting a mean rather than a sum) apparently get better results
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        # do the zero grad thing
        disc.zero_grad()
        # backpropagation to get the gradient
        loss_disc.backward()
        # take an appropriately sized step (gradient descent)
        opt_disc.step()

        #############################################################
        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)) #
        #############################################################

        # pass the fake images through the discriminator i.e. calculate D(G(z))
        # reshape(-1) ensures the output is in a single column
        output = disc(fake).reshape(-1)
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

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{n_epochs}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
