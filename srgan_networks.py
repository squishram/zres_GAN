"""
classes and functions for SRGAN
"""

import torch
from torch import nn
from torchvision.models import vgg19
import srgan_config

# A factory that churns out generators:
class ConvBlock(nn.Module):
    # kwargs == key work arguments eg kernel size etc
    def __init__(self, in_channels, out_channels, is_disc=False, use_act=True, use_bn=True, **kwargs):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True) if is_disc
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))



class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c*scale_factor ** 2, 3, 1, 1)
        # PixelShuffle distributes channels into the height and width
        # so in this case the dimensions change like this:
        # C * W * H --> C/4 * 2W * 2H
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False,
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, n_channels=64, num_blocks=16):
        super().__init__()
        self.initial = ConvBlock(in_channels, n_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(n_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(n_channels, n_channels, kernel_size=3,stride=1, padding=1, use_act=False)
        self.upsamples = nn.Sequential(UpsampleBlock(n_channels, 2), UpsampleBlock(n_channels, 2))
        self.final = nn.Conv2d(n_channels, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        return torch.tanh(self.final(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx%2,
                    padding=1,
                    is_disc=True,
                    use_act=True,
                    use_bn=False if idx==0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)



# phi_5,4 5th conv layer before maxpooling but after activation
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(srgan_config.device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)


#TODO THIS IS WHAT I NEED TO UNDERSTAND #
def initialise_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# This isn't actually being used yet
# Ultimately will be checking the hit rate of the network
def check_success(loader, model, device):
    n_samples = 0
    n_correct = 0
    model.eval()

    with torch.no_grad():
        print("obtaining accuracy on test data")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max()
            n_correct += (predictions == y).sum()
            n_samples += predictions.size(0)

        print(f'Got {n_correct}/{n_samples} with accuracy {(n_correct/n_samples) * 100}')

def test_structure():
    low_res = 24
    with torch.cuda.amp.autocast():
        x = torch.randn((5, 3, low_res, low_res))
        gen = Generator()
        gen_out = gen(x)
        disc = Discriminator()
        disc_out = disc(gen_out)

        print(gen_out.shape)
        print(disc_out.shape)

if __name__ == "__main__":
    test_structure()
