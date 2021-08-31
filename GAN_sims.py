from torch import nn

# the batch size used in training
size_batch =  128

# The spatial size of the images used for training. This implementation defaults to 64x64
# NOTE: If another size is desired, the structures of D and G must be changed
size_img = 96
# nc - number of color channels in the input images. For color images this is 3
nc = 3
# nz - length of latent vector (this is the random noise from which the fake image is generated)
n_latent = 100
# ngf - relates to the depth of feature maps carried through the generator
ngf = 64
# ndf - sets the depth of feature maps propagated through the discriminator
ndf = 64
# number of training epochs to run
n_epochs = 5
# lr - learning rate for training. As described in the DCGAN paper, this number should be 0.0002
lr = 0.0002
# beta1 - beta1 hyperparameter for Adam optimizers. As described in paper, this number should be 0.5
beta1 = 0.5
# ngpu - number of GPUs available. If this is 0, code will run in CPU mode. If this number is greater than 0 it will run on that number of GPUs
n_gpu = 0


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 1st conv layer
            nn.ConvTranspose2d(in_channels=size_img, out_channels=size_img, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(size_img, size_img, 5, 1, 2, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.01, True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.01, True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.01, True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)