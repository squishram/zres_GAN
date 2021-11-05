import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import tqdm

# download dataset (code taken from website)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create dataset of list of images of type frog
froglist = []
for image, imtype in trainset:
    if imtype==6:
        froglist.append(image)


trainloader = torch.utils.data.DataLoader(froglist, batch_size=batch_size, shuffle=True, num_workers=2)


# this is the upsampling network
class upsample(nn.Module):
    """Simple fully convolutional network"""
    # pylint: disable=too-many-instance-attributes
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,4,3,padding=1)
        self.conv2 = nn.Conv2d(4,8,3,padding=1)
        self.conv3 = nn.Conv2d(8,16,3,padding=1)
        self.conv4 = nn.Conv2d(16,3,5,padding=0)
    
    def forward(self, x):
        assert(x.size()[1]==25)
        n = x.size()[0]
        xnew = torch.reshape(x,(n,5,5))
        xnew = xnew.unsqueeze(1)
        xnew = self.conv1(xnew)
        xnew = F.relu(xnew)
        xnew = F.interpolate(xnew,scale_factor=2,align_corners=False,mode='bilinear')
        xnew = self.conv2(xnew)
        xnew = F.relu(xnew)
        xnew = F.interpolate(xnew,scale_factor=2,align_corners=False,mode='bilinear')
        xnew = self.conv3(xnew)
        xnew = F.relu(xnew)
        xnew = F.interpolate(xnew,scale_factor=2,align_corners=False,mode='bilinear')
        xnew = self.conv4(xnew)
        xnew = xnew[:,:,0:32,0:32]
        xnew = torch.sigmoid(xnew)
        return xnew


# this is the downsampling network
# it increases the number of channels and decreases the size
class downsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3,stride=2,padding=0)
        self.conv2 = nn.Conv2d(6,12,3,stride=2,padding=0)
        self.conv3 = nn.Conv2d(12,24,3,stride=2,padding=0)
        self.conv4 = nn.Conv2d(24,1,3,padding=0)


    def forward(self,x):
        xnew = self.conv1(x)
        xnew = F.relu(xnew)
        xnew = self.conv2(xnew)
        xnew = F.relu(xnew)
        xnew = self.conv3(xnew)
        xnew = F.relu(xnew)
        xnew = self.conv4(xnew)
        xnew = torch.sigmoid(xnew)
        xnew = torch.squeeze(xnew,3)
        xnew = torch.squeeze(xnew,2)
        return xnew


# takes in a module and a bool on/off
# if on, allows training for that module
# if off, turns training off
def set_training_mode(module: torch.nn.Module, train: bool) -> None:
    """Sets both batchnorms etc and gradients"""
    for param in module.parameters():
        param.requires_grad = train

    if train: 
        module.train()
    else:
        module.eval()


# set up generative network
gan = upsample().to(device)

# set up adversary network
adversaryout = downsample().to(device)

# set up optimisers
generator_optimizer = torch.optim.Adam(gan.parameters(), lr=0.001)
adversary_optimizer = torch.optim.Adam(adversaryout.parameters(), lr=0.0001)


epochs = 10

for i in range(epochs):
    for frogs_cpu in tqdm.tqdm(trainloader):
        
        frogs = frogs_cpu.to(device)

        # set training to be on for gan and off for adversary
        set_training_mode(gan, True)
        set_training_mode(adversaryout, False)

        # initialise a short vector of noise
        batch = 4
        init = torch.normal(0, 1, (batch, 25), device=device)
        # run generative network
        genim = gan.forward(init)

        # run adversary on generated images
        adout = adversaryout.forward(genim)
        
        # 0 is fake images and 1 is real images
        # calculate the binary loss between output of adversary and list of ones
        loss = F.binary_cross_entropy(adout,torch.ones(batch,1, device=device))

        # take a step of the generator
        generator_optimizer.zero_grad()
        loss.backward()
        generator_optimizer.step()

        # set training to be off for gan and on for adversary
        set_training_mode(gan, False)
        set_training_mode(adversaryout, True)

        # run adversary on data and compute loss
        adout = adversaryout.forward(frogs)
        lossreal = F.binary_cross_entropy(adout,torch.ones(batch, 1, device=device))

        # generate images
        init = torch.normal(0, 1, (batch, 25), device=device)
        
        # train adversary on generated images
        genim = gan.forward(init)
        adout = adversaryout.forward(genim)
        lossgen = F.binary_cross_entropy(adout, torch.zeros(batch, 1, device=device))

        # optimise adversary on both generated and real losses
        totalloss = lossgen + lossreal

        adversary_optimizer.zero_grad()
        totalloss.backward()
        adversary_optimizer.step()

    # write images at the end of each epoch
    # initialise a short vector of noise
    batch = 4
    init = torch.normal(0, 1, (batch, 25), device=device)
    # run generative network
    genim = gan.forward(init)
    torchvision.utils.save_image(genim, "testimage{:02}.png".format(i))







