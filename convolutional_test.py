"""
Testing input/output dimensions of convolutional layers
"""

import torch
import torch.nn.functional as ff


# USER INPUT
input = 24
padding = 1
kernel = 3
stride = 1
# USER INPUT END

input = torch.rand([1, 1, input])
kernel = torch.rand([1, 1, kernel])

output = ff.conv1d(input, kernel, None, stride, padding).squeeze(0).squeeze(0).numpy()
print(len(output))

# input = 96
# padding = 1
# kernel = 3
# stride = 2
# output = 48

# input = 48
# padding = 1
# kernel = 3
# stride = 2
# output = 24

# input = 24
# padding = 1
# kernel = 3
# stride = 1
# output = 24

# input = 24
# padding = 1
# kernel = 3
# stride = 3
# output = 8
