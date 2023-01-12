import os
from pathlib import Path
from datetime import date
import torch


###########
# STORAGE #
###########

# path to data
path_data = os.path.join(
    os.getcwd(), Path("images/sims/microtubules/noGT_LD_zres5xWorse")
)
# glob of filenames
filename = "mtubs_sim_*.tif"

# path to generated images - will make directory if there isn't one already
# get the date
today = str(date.today())
# remove dashes
today = today.replace("-", "")
path_gens = os.path.join(
    os.getcwd(), Path("images/sims/microtubules/generated/"), today
)
# path_gens = Path(f"{os.getcwd()}images/sims/microtubules/generated/{today}")
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
freq_domain_loss_scaler = 0
space_domain_loss_scaler = 100
adversary_gen_loss_scaler = 0.1
# for the discriminator:
loss_dis_real_scaler = 0.1
loss_dis_fake_scaler = 0.1
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
window = "bharris"
# type of discriminator: can be patchgan or normal
type_disc = "patchgan"
