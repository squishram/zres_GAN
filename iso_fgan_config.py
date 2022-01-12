import torch
import albumentations
from albumentations.pytorch import ToTensorV2

# (HYPER)PARAMETERS #
load_model = True
save_model = True
checkpoint_gen = "gen.pth.tar"
checkpoint_disc = "disc.pth.tar"
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-4
n_epochs = 10
size_batch = 128
n_workers = 2
sample_dims = 32
n_colour_channels = 3

# transform to squeeze more epochs out of the datasets
transform = albumentations.Compose([
    # uses a random crop of the image as it appears in the dataset
    albumentations.RandomCrop(width=sample_dims, height=sample_dims),
    # uses a random flip of the image as it appears in the dataset
    albumentations.HorizontalFlip(p=0.5),
    # includes a random 90 degree rotation of the picture
    albumentations.RandomRotate90(p=0.5),
    albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])
