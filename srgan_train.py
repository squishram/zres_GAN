import albumentations
from albumentations.pytorch import ToTensorV2
import os
from pathlib import Path
# import numpy as np
import matplotlib.pylab as plt
import torchvision.datasets as dset
import torchvision.utils as utils
import torch
from torch import nn
from torch import optim
# from srgan_utils import load_checkpoint, save_checkpoint, plot_examples
# from torch.utils.data import DataLoader
from srgan_networks import Generator, Discriminator, VGGLoss
from tqdm import tqdm
# from dataset import MyImageFolder
from datetime import date


# This enables 'benchmark mode', which usually leads to faster runtime
# but this only works when the input sizes for your network do not vary
# cudnn will look for the optimal set of algorithms for that configuration
# (which takes some time)
# if your input sizes changes at each iteration it will be slower
torch.backends.cudnn.benchmark = True


###########
# STORAGE #
###########

# get the date
today = str(date.today())
today = today.replace('-', '')
# path_data - the path to the root of the dataset folder
path_data = os.path.join(os.getcwd(), "images/")
# this is the full path for the sample images
path_real = os.path.join(path_data, Path("catface/real/"))
path_gens = os.path.join(path_data, Path("catface/generated/"), today)
os.makedirs(path_gens, exist_ok=True)

#####################
# (HYPER)PARAMETERS #
#####################

# load_model = True
# save_model = True
# checkpoint_gen = "gen.pth.tar"
# checkpoint_disc = "disc.pth.tar"
device = "cuda" if torch.cuda.is_available() else "cpu"
# learning rate - srgan paper says 1e-4
learning_rate = 1e-4
# number of epochs - srgan paper says 100
n_epochs = 10
# batch size - srgan paper says 16
size_batch = 128
# this is to do with cpu usage when loading in the dataset - just set as 2
n_workers = 2
# side length of real images
size_img = 64
# pooling kernel i.e. resolution restoration factor
pool_kernel = 4
# size_low_res_img = size_img // pool_kernel
# number of colour channels of images
# TODO - if =1, add Grayscale layer to image transform composition (below)
n_colour_channels = 3
transform = albumentations.Compose(
    [albumentations.Resize(width=size_img, height=size_img),
     albumentations.CenterCrop(size_img, size_img),
     albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
     ToTensorV2(), ])

############################
# DATASETS AND DATALOADERS #
############################

# how much of the total dataset will be used for training?
# the 'test dataset' will be = 1 - train_portion
train_portion = 0.9
# first pull out the whole dataset
dataset = dset.ImageFolder(path_real, transform=transform)
# split into training and testing datasets according to fraction train_portion
train_size = int(train_portion * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(dataset,
                                                  [train_size, test_size])
# stick them into dataloaders for training and testing
trainloader = torch.utils.data.DataLoader(trainset,
                                          size_batch,
                                          shuffle=True,
                                          num_workers=2,
                                          pin_memory=True)
testloader = torch.utils.data.DataLoader(testset,
                                         size_batch,
                                         shuffle=True,
                                         num_workers=2,
                                         pin_memory=True)

# TODO: below is an alternative dataloader
# that can be imported from a custom image batch
# dataset = MyImageFolder(root_dir="new_data/")
# loader = MyDataLoader(dataset,
#                     size_batch=size_batch,
#                     shuffle=True,
#                     pin_memory=True,
#                     n_workers=n_workers,)


##########################
# NETWORK INITIALISATION #
##########################

# create instances of the generator and discriminator from their classes
gen = Generator(in_channels=3).to(device)
dis = Discriminator(in_channels=3).to(device)
# make the pooling function - this downsamples the original image
# nn.AvgPool2d(side length of pooling kernel, stride)
pool = nn.AvgPool2d(4, stride=4).to(device)

# create step-takers for each one (Adam is best for GANs)
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.9, 0.999))
opt_dis = optim.Adam(dis.parameters(), lr=learning_rate, betas=(0.9, 0.999))
# create loss functions from their classes
criterion_mse = nn.MSELoss()
criterion_bce = nn.BCEWithLogitsLoss()
criterion_vgg = VGGLoss()

# this sets the networks to training mode
# they should be by default, but it doesn't hurt to have assurance
gen.train()
dis.train()

# pull out some test images to check them
data_iter = iter(testloader)
first_images, labels = data_iter.next()
first_images = first_images.to(device)

# TODO: it could be the case that you want to pretrain a model before training
# it here, in which case you will need to save and import the network weights
# and biases, these conditions can be set at the beginning
# if load_model:
#     load_checkpoint(checkpoint_gen, gen, opt_gen, learning_rate)
#     load_checkpoint(checkpoint_dis, dis, opt_dis, learning_rate)


# step += 1 for every forward pass
step = 0
# this list will contain the losses
loss_list = [[] for i in range(7)]

for epoch in range(n_epochs):
    # tqdm wrapper enables progress bar
    loop = tqdm(trainloader, leave=True)

    for idx, (real, _) in enumerate(loop):

        # put the real images on the GPU
        real = real.to(device)
        # pass them through the pooling function to get low res images
        blur = pool(real).to(device)
        # low res images --generator--> fake super res images
        # i.e. calculate G(z)
        fake = gen(blur)

        ########################################################
        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))#
        ########################################################

        # real images ---discriminator---> D(x)
        dis_real = dis(real)
        # fake images ---discriminator---> D(G(z))
        dis_fake = dis(fake.detach())
        # dis_real ---criterion_bce function---> first part of loss
        # adding rand_like() acts as a smoothing function
        loss_dis_real = criterion_bce(dis_real, torch.ones_like(dis_real)
                                      - 0.1*torch.rand_like(dis_real))
        # dis_fake ---criterion_bce function---> second part of loss
        loss_dis_fake = criterion_bce(dis_fake, torch.zeros_like(dis_fake))
        # sum two discriminator loss components
        loss_dis = loss_dis_fake + loss_dis_real

        # take step
        opt_dis.zero_grad()
        loss_dis.backward()
        opt_dis.step()

        ####################################
        # Train Generator: max log(D(G(z)) #
        ####################################

        # pass the fake super-res images through the discriminator
        # i.e. calculate D(G(z))
        dis_fake = dis(fake)
        # l2_loss = criterion_mse(fake, real)
        loss_gen_adv = 1e-3 * criterion_bce(dis_fake,
                                            torch.ones_like(dis_fake))
        # vgg loss is an alternative pixel-wise loss (as opposed to e.g. L1)
        loss_gen_vgg = 0.006 * criterion_vgg(fake, real)
        # sum two generator loss components
        loss_gen = loss_gen_vgg + loss_gen_adv

        # take step
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if step % 100 == 0:
            loss_list[0].append(int(step))
            loss_list[1].append(float(loss_dis_fake))
            loss_list[2].append(float(loss_dis_real))
            loss_list[3].append(float(loss_dis))
            loss_list[4].append(float(loss_gen_adv))
            loss_list[5].append(float(loss_gen_vgg))
            loss_list[6].append(float(loss_gen))

        # count the number of backpropagations
        step += 1

    # using the 'with' method in conjunction with no_grad() simply
    # disables grad calculations for the duration of the statement
    # Thus, we can use it to generate a sample set of images without initiating
    # a backpropagation calculation
    with torch.no_grad():
        blur = pool(first_images)
        fake = gen(blur)
        # denormalise the images so they look nice n crisp
        fake *= 0.5
        fake += 0.5
        # name your image grid according to its epoch
        fake_fname = 'generated_images_epoch-{0:0=2d}.png'.format(epoch + 1)
        # make a grid i.e. a sample of generated images to look at
        img_grid_fake = utils.make_grid(fake[:32], normalize=True)
        utils.save_image(fake,
                         os.path.join(path_gens, fake_fname), nrow=8)
        # Print losses
        print(f"Epoch [{epoch + 1}/{n_epochs}] - saving {fake_fname}")

    # TODO if you want to save the weights and biases, set this to go at the
    # beginning
    # if save_model:
    #     save_checkpoint(gen, opt_gen, filename=checkpoint_gen)
    #     save_checkpoint(dis, opt_dis, filename=checkpoint_dis)


########################
# make a metadata file #
########################
# give it a nice name
metadata = today + "_metadata.txt"
# place it with the generated images
metadata = os.path.join(path_gens, metadata)
# make sure to remove any other metadata files in the subdirectory
if os.path.exists(metadata):
    os.remove(metadata)

# loop to write metadata to .txt file
# TODO make sure to add more about the network structures!
with open(metadata, "a") as file:
    file.writelines([os.path.basename(__file__),
                     "\nlearning_rate = " + str(learning_rate),
                     "\nsize_batch = " + str(size_batch),
                     "\nsize_img = " + str(size_img),
                     "\nn_colour_channels = " + str(n_colour_channels),
                     "\nn_epochs = " + str(n_epochs), ])

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
            'loss_gen'], loc='upper left')

print("Saving loss graph...")
plt.savefig(os.path.join(path_gens, 'losses'), format='pdf')

print("Done!")
