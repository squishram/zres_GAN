import numpy as np
import os
# from PIL import Image
import iso_fgan_config
from pathlib import Path
# import matplotlib.pylab as plt
# import torch
# import torch.nn as nn
# import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tt
# import torchvision.utils as utils
from datetime import date

# STORAGE #
# get the date
today = str(date.today())
today = today.replace('-', '')
# path to the image directory
dir_data = os.path.join(os.getcwd(), Path("images/sims/microtubules/"))
# this is the full path for the sample images
path_hires = os.path.join(dir_data, "hires")
path_lores = os.path.join(dir_data, "lores")
path_gens = os.path.join(dir_data, "generated", today)
# make a folder for the generated images if there isn't one
os.makedirs(path_gens, exist_ok=True)

# setting image shape to 32x32
img_shape = (96, 96, 96)

# listing out all file names
img_names = np.sort(os.listdir(dir_data))

# how much of the total dataset will be used for training?
# the 'test dataset' will be = 1 - train_portion
train_portion = 0.9


transform = tt.Compose([  # tt.Resize(img_shape),
                        tt.CenterCrop(img_shape),
                        tt.ToTensor(),
                        tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])


############################
# DATASETS AND DATALOADERS #
############################

# first pull out the whole dataset
dataset_hires = dset.ImageFolder(path_hires, transform=transform)
dataset_lores = dset.ImageFolder(path_lores, transform=transform)
