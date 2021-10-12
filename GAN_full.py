from __future__ import print_function
import os
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as tt
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from time import time
from pathlib import Path

# begin timer
t1 = time()

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# path_data - the path to the root of the dataset folder
path_data = os.path.join(os.getcwd(), "catface/")
# make a directory for the generated images
dir_samples = Path("images/generated/")
# this is the full path for the sample images
path_samples = os.path.join(path_data, dir_samples)
os.makedirs(path_samples, exist_ok=True)
# workers - the number of worker threads for loading the data with the DataLoader
# must be 0 for non-multi-threaded CPU/GPU
n_workers = 2
# the batch size used in training
size_batch = 128

# The spatial size of the images used for training. This implementation defaults to 64x64
# NOTE: If another size is desired, the structures of D and G must be changed
size_img = 64
# nc - number of color channels in the input images.
# For color images this is =3, for BW images it's =1
nc = 3
# nz - length of latent vector (this is the random noise from which the fake image is generated)
n_latent = 100
# ngf - relates to the depth of feature maps carried through the generator
ngf = 64
# ndf - sets the depth of feature maps propagated through the discriminator
ndf = 64
# number of training epochs to run
n_epochs = 60
# lr - learning rate for training. As described in the DCGAN paper, this number should be 0.0002
lr = 0.0002
# beta1 - beta1 hyperparameter for Adam optimizers. As described in paper, this number should be 0.5
beta1 = 0.5


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(n_latent, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5


# The function to save all the generated images:
def save_samples(index, latent_tensors, path):
    fake_images = netG(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(path, fake_fname), nrow=8)
    print('Saving', fake_fname)


# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(path_data,
                           transform=tt.Compose([
                               tt.Resize(size_img),
                               tt.CenterCrop(size_img),
                               tt.ToTensor(),
                               tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, size_batch, shuffle=True, num_workers=n_workers, pin_memory=True)


# find the number of GPUs available
n_gpu = torch.cuda.device_count()
# check for a GPU; use as device if available
device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))


########################
# Create the generator #
########################
netG = Generator(n_gpu).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (n_gpu > 1):
    netG = nn.DataParallel(netG, list(range(n_gpu)))
# Apply the weights_init function to randomly initialize all weights
# to mean=0, stdev=0.2.
netG.apply(weights_init)
# Print the model
print(netG)


############################
# Create the Discriminator #
############################
netD = Discriminator(n_gpu).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (n_gpu > 1):
    netD = nn.DataParallel(netD, list(range(n_gpu)))
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)
# Print the model
print(netD)


# Initialise BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
# the progression of the generator
fixed_latent = torch.randn(64, n_latent, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# TRAINING LOOP TIME!

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(n_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ###################################################################
        # (1) Update Discriminator: maximise log(D(x)) + log(1 - D(G(z))) #
        ###################################################################

        # Train with all-real batch #
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch #
        # Generate batch of latent vectors
        noise = torch.randn(b_size, n_latent, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ###############################################
        # (2) Update Generator: maximise log(D(G(z))) #
        ###############################################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch + 1, n_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == n_epochs - 1) and (i == len(dataloader) - 1)):
            # with torch.no_grad():
            #    fake = netG(fixed_latent).detach().cpu()
            # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            save_samples(epoch + iters + 1, fixed_latent, path_samples)

        iters += 1

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


t2 = time()

print("Congrats! Training completed. It took " + str((t2-t1)/60) + " minutes to complete.")
print("You used a batch size of " + str(size_batch) + " and trained for" + str(n_epochs) + " epochs.")