import torch
from PIL import Image
import albumentations
from albumentations.pytorch import ToTensorV2


# (HYPER)PARAMETERS #
load_model = True
save_model = True
checkpoint_gen = "gen.pth.tar"
checkpoint_disc = "disc.pth.tar"
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-4
n_epochs = 100
size_batch = 16
n_workers = 2
high_res = 64
pool_kernel = 4
low_res = high_res // pool_kernel
n_colour_channels = 3

highres_transform = albumentations.Compose(
    [albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
     ToTensorV2(), ])

lowres_transform = albumentations.Compose(
    [albumentations.Resize(width=low_res, height=low_res, interpolation=Image.BICUBIC),
     albumentations.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
     ToTensorV2(), ])

both_transforms = albumentations.Compose(
    [albumentations.RandomCrop(width=high_res, height=high_res),
     albumentations.HorizontalFlip(p=0.5),
     albumentations.RandomRotate90(p=0.5), ])

test_transform = albumentations.Compose(
    [albumentations.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
     ToTensorV2(), ])
