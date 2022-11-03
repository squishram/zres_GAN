import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import albumentations
from albumentations.pytorch import ToTensorV2
from affirm3d_noGT_classes import Custom_Dataset


def mean_and_std(dataloader):

    signal_sum = 0.
    signal_squared_sum = 0.
    n_batches = 0.
    for data in dataloader:
        signal_sum += torch.mean(data, dim=[0, -3, -2, -1])
        signal_squared_sum += torch.mean(data ** 2, dim=[0, -3, -2, -1])
        n_batches += 1

    mean = signal_sum / n_batches
    std = ((signal_squared_sum / n_batches) - (mean ** 2)) ** 0.5

    return mean, std


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_workers = 2
img_size = [32, 96, 96]
batch_size = 12

# path to data
path_data = os.path.join(
    os.getcwd(), Path("images/sims/microtubules/noGT_LD_zres5xWorse")
)
# glob of filenames
filename = "mtubs_sim_*.tif"

transformoid = albumentations.Compose(
    [
        # albumentations.Resize(img_size[-1], width=img_size[-2]),
        albumentations.HorizontalFlip(),
        albumentations.Normalize(mean=0.0309, std=0.0740),
        ToTensorV2(),
    ]
)

# image datasets
dataset = Custom_Dataset(
    dir_data=path_data,
    filename=filename,
    transform=None,
)

# image dataloaders when loading in hires and lores together
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

print(mean_and_std(dataloader))
