"""
For calculating how to convolve each layer to the amount you want
"""

padding = [2, 2, 1]
kernel = [8, 8, 4]
stride = [4, 4, 2]
inp = [64, 64, 32]
out = []

for i in range(3):
    out.append(((inp[i] + (2 * padding[i]) - kernel[i]) / stride[i]) + 1)

print(out)

# padding = [1, 1, 1]
# kernel = [4, 4, 3]
# stride = [2, 2, 1]
# inp = [128, 128, 32]

# padding = [1, 1, 1]
# kernel = [4, 4, 3]
# stride = [2, 2, 1]
# inp = [64, 64, 32]

# padding = [1, 1, 1]
# kernel = [4, 4, 4]
# stride = [2, 2, 2]
# inp = [32, 32, 32]
