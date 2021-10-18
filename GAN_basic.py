import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter as SW


class Discriminator(nn.Module):

    def __init__(self, dim_img, l1, nl):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(dim_img, l1),
            nn.LeakyReLU(0.1),
            nn.Linear(l1, nl),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):

    def __init__(self, nz, l1, dim_img):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(nz, l1),
            nn.LeakyReLU(0.1),
            nn.Linear(l1, dim_img),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


#
device = "cuda" if torch.cuda.is_available() else "cpu"
#
lr = 3e-4
#
nz = 64
#
nl = 1
#
dim_img = 64
#
l1 = 256
#
n_batch = 64
#
n_epochs = 50

disc = Discriminator(dim_img=dim_img, l1=l1, nl=nl).to(device)
gen = Generator(nz=nz, l1=l1, dim_img=dim_img).to(device)

z_fix = torch.randn((n_batch, nz, nz)).to(device)
transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
)

