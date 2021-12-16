# import numpy as np
import torch
# import torchvision
# import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn.optim
from nn_functions import *

# -----------------------------------------------------------------------------
# INPUTS
# -----------------------------------------------------------------------------

# How many possible outputs?
output_layer = 10
# How many nodes in each HIDDEN layer?
layers = [128, 64]
# How many in a batch?
batch = 64
# Give me 2 numbers in a tuple whose product is <batch
# (this is the grid we will display that part of the batch in, for inspection)
disp_count = (6, 10)
# How many epochs?
epochs = 15
# learning rate?
learnRate = 0.003
# momentum?
p = 0.9


# -----------------------------------------------------------------------------
# DATA IMPORT, FORMATTING, INSPECTION
# -----------------------------------------------------------------------------

# torchvision.transforms.Compose(): defines transform that can be applied to MINST dataset to convert images into tensors
# transforms.ToTensor(): separates the image into three colour channels, from these obtains 8-bit brightness, scales these down to [0, 1] interval...
# ... now, type(dataset) == torch.tensor()
# transforms.Normalize(mu, sigma): transform the value using newval = (oldval - mu)/ sigma
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, ), (0.5,)), ])

# Download the training and test datasets
trainset = datasets.MNIST(root=r'C:\\Users\\User\\Documents\\Current_Work\\code_Python\\SMLM', download=True, train=True,  transform=transform)
valset = datasets.MNIST(root=r'C:\\Users\\User\\Documents\\Current_Work\\code_Python\\SMLM', download=True, train=False, transform=transform)
# Load the training and test datasets
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch,
                                          shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch, shuffle=True)

# create iterator object with attribute "next" for the training data
dataiter = iter(trainloader)
# .next() retrieves the next item in dataiter
# (first batch the first time, second batch second time)
images, labels = dataiter.next()

# inpect the dataset retrieve dimensions of one batch
# print(images.squeeze().shape)
# [64, 1, 28, 28] - so, there are 64x 28*28 pixel images
# print(labels.shape)
# [64] - 64 labels for 64 pictures - checks out

# look at one of the images!
# .numpy() converts to a numpy array
# .squeeze() removes all dimensions of unit length from a numpy array (notice that images.shape == [64, 1, 28, 28])
# figure = plt.figure()
# for i in range(1, disp_count[0]*disp_count[1] + 1):
#    plt.subplot(disp_count[0], disp_count[1], i)
#    plt.axis('off')
#    plt.imshow(images[i].numpy().squeeze(), cmap='gray_r')
# plt.show()


# -----------------------------------------------------------------------------
# THE NEURAL NETWORK
# -----------------------------------------------------------------------------

# The layers list contains the number of nodes in each layer -
# add the first and last layers to the beginning and end respectively
layers.append(output_layer)
layers.insert(0, images.shape[2]*images.shape[3])


# model = ANN(layers)
model = CNN()
print(model)

"""
nn.NLLLoss() is The 'negative log likelihood loss' - it is a cost function
it calculates 'minimising the loss' as 'minimising the entropy'
where entropy is basically the 'uncertainty' of the network's output
it's often better than the sum of squared residuals/ mean squared error,
as it computes a much larger loss at outputs that are very wrong
meaning the slope is sharper and the gradient descent is steeper
it is used in conjunction with a nn.LogSoftmax() transform on the final layer
"""
cost = nn.NLLLoss()
# cost = nn.MSELoss()

"""
nn.torch.optim.SGD() is the method by which the size of that step is taken
- stochastic gradient descent
SGD just means you use a random subset of the full dataset
when calculating each step, instead of the whole thing
This makes each step less accurate, but less computationally expensive,
and therefore faster

model.parameters() are the weights and biases obtained from the model
lr is the learning rate (how big are the steps down the hill)

momentum is like: think of rolling a ball down the hill instead of walking down
it might roll up the next hill and into a lower local minimum!

Momentum is calculated as a weighted average as the gradient descent step sizes

If momentum=B where B<1 and n is the last step before the bottom of the hill is reached
then the momentum 'step up the other side of the hill' is sum[B^(i+1)*(n-i) for i in range(0, n)]

opt = optim.SGD(model.parameters(), lr=learnRate, momentum=p)
"""

# start a timer:
time0 = time()

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:

        # Reset the optimiser to start with all elements == None
        opt.zero_grad()

        # Do a forward pass
        output = model(images)

        # The output is what the network thinks,
        # the labels are the actual correct answer.
        # Apply the cost function to these two, and you get the loss!
        loss = cost(output, labels)

        # backward() computes the gradient, so we know the step size.
        # this is 'backpropagation':
        loss.backward()

        # The weights are optimised here - "1 step down the hill" is applied
        opt.step()

        running_loss += loss.item()
    # for/ else just applies the else statement once the for loop is done
    else:
        print(running_loss)
        print("Epoch {} - Training loss: {}".format(e + 1, running_loss/len(trainloader)))

print("\nTraining Time (in minutes) =", (time()-time0)/60)
