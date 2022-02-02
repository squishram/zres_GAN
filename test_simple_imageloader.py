from pathlib import Path
import os
from skimage import io
from datetime import date
import torch
import numpy as np

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
path_lores = Path(os.path.join(path_data, Path("microtubules/lores")))
path_hires = Path(os.path.join(path_data, Path("microtubules/hires")))
# path to gneerated images - will make directory if there isn't one already
path_gens = os.path.join(path_data, Path("microtubules/generated"), today)
os.makedirs(path_gens, exist_ok=True)


#####################
# (HYPER)PARAMETERS #
#####################

# use gpu if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# learning rate
# for DCGAN, anecdotal recommendation = 3e-4
learning_rate = 3e-4
# batch size, i.e. #forward passes per backward propagation
size_batch = 10
# side length of (cubic) images
size_img = 96
# number of epochs i.e. number of times you re-use the same training images
n_epochs = 5
# channel depth of generator hidden layers in integers of this number
features_generator = 16
# channel depth of discriminator hidden layers in integers of this number
features_discriminator = 16
# the side length of the convolutional kernel in the network
kernel_size = 3
# pixel size in nm
size_pix_nm = 100.0
# z-resolution in the isotropic case
zres_hi = 240.0
# z-resolution in the anisotropic case
zres_lo = 600.0


##################
# LOAD IN IMAGES #
##################

lores_glob = sorted(path_lores.glob("mtubs_sim_*_lores.tif"))
hires_glob = sorted(path_hires.glob("mtubs_sim_*_hires.tif"))
# lores_all = torch.tensor([[[[]]]])
# hires_all = torch.tensor([[[[]]]])
lores_all = torch.empty((1, size_img, size_img, size_img))
hires_all = torch.empty((1, size_img, size_img, size_img))

for img_path in (lores_glob, hires_glob):
    for file in img_path:

        img = io.imread(file)
        # convert to numpy array for faster calculations
        img = np.asarray(img)
        # normalise pixel values
        img = img - np.mean(img) / np.std(img)
        # scale pixel values to 1
        img = img / np.max(img)

        # convert to tensor
        img = torch.tensor(img)
        # now: img.shape = (z, x, y)
        img = torch.swapaxes(img, 1, 2)
        # now: img.shape = (z, y, x)
        img = img.unsqueeze(0)
        # now: img.shape = (1, z, y, x)

        if img_path == lores_glob:
            lores_all = torch.cat((lores_all, img), 0)
        elif img_path == hires_glob:
            hires_all = torch.cat((hires_all, img), 0)

lores_all = lores_all[1:]
hires_all = hires_all[1:]
