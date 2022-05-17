from sympy import symbols, Eq, solve

padding = [1, 1, 1]
kernel = [3, 3, 3]
stride = [2, 2, 1]
inp = [128, 128, 32]
out = []

for i in range(3):
    out.append(((inp[i] + (2 * padding[i]) - kernel[i]) / stride[i]) + 1)

print(out)
