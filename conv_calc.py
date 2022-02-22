"""
Convolutional Network Calculator
"""

dims = 31
"""
options:
6, 3, 0 -> 31
3, 2, 0 -> 15
3, 2, 0 -> 7
3, 2, 0 -> 3
3, 2, 0 -> 1
"""
kernel = 3
stride = 2
padding = 0

while dims > 1:
    dims = 1 + ((dims + (2 * padding) - kernel) / stride)
    print(dims)
