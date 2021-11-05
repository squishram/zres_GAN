import torch.nn as nn
import torch
import os
import torchvision.transforms as tt
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import tqdm

# path_data - the path to the root of the dataset folder
path_data = os.path.join(os.getcwd(), "images/")
# side length of input and output images
size_img = 64
# size batch
size_batch = 1
# colour channels
n_colour_channels = 3

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
# dataloader = torch.utils.data.DataLoader(dataset, size_batch, shuffle=True, num_workers=1, pin_memory=True)
# take out one image

cat_list = []
for img, imtype in dataset:
    cat_list.append(img)

pool = nn.AvgPool2d(2, stride=2)

for inp in cat_list[:3]:
    out = pool(inp)
    plt.subplot(1, 2, 1)
    plt.imshow(inp.permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(out.permute(1, 2, 0))
    plt.show()
