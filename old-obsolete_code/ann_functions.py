from torch import nn
"""
Build the model with nn.Sequential()
nn.Linear() performs a linear transform from one layer to the next
(applies weights and biases)
nn.ReLU() maps node values to max(value, 0)
nn.LogSoftmax()
maps node values to exp(val)
expresses them as a fraction of the sum of all exp(val) in the layer
then takes the log() of this result,
as it reduces the instances of over & underflow (i.e. too many digits)
"""


def ANN(layers):
    model = nn.Sequential(nn.Flatten(),
                          nn.Linear(layers[0], layers[1]),
                          nn.ReLU(),
                          nn.Linear(layers[1], layers[2]),
                          nn.ReLU(),
                          nn.Linear(layers[2], layers[3]),
                          nn.LogSoftmax(dim=1))
    return model


"""
nn.Conv2d(layer1, layer2, kernel_size, padding) creates a 2d convolution layer
nn.AvgPool2d creates a pooling layer for the convolved layer, and extracts the mean
the kernels are 5x5 for the convolutional layers and 2x2 for the pooling layers
the 'stride' for the pooling layer is the amount it moves across the convolved layers
padding='same' adds zero-padding to the outside of the layer before convolution
such that the size of the layer is preserved
padding='valid' for some reason this is the term used to mean 'no padding'
so that the layers shrink on each convolution
alternatively, you can use an integer or tuple to add custom-sized padding
"""


def CNN(cnnlayers=[1, 6, 16], kernels_conv=[5, 5], kernels_pool=[2, 2], padding='same', annlayers=[400, 120, 84, 10]):
    model = nn.Sequential(
        nn.Conv2d(cnnlayers[0], cnnlayers[1], kernel_size=kernels_conv[0], padding=padding),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=kernels_pool[0], stride=2),
        nn.Conv2d(cnnlayers[1], cnnlayers[2], kernel_size=kernels_conv[1], padding=padding),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=kernels_pool[1], stride=2),
        nn.Flatten(),
        nn.Linear(annlayers[0], annlayers[1]),
        nn.ReLU(),
        nn.Linear(annlayers[1], annlayers[2]),
        nn.ReLU(),
        nn.Linear(annlayers[2], annlayers[3]),
        nn.LogSoftmax(dim=1)
    )
    return model

