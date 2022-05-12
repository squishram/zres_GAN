from sympy import symbols, Eq, solve

stride = 1
input = 128
output = 64

# defining symbols used in equations
# or unknown variables
padding, kernel = symbols('p,k')

# defining equations
equation = Eq(2 * padding - kernel, output - 1 - (input / stride))
print(solve(equation, (padding, kernel)))

padding = [1, 1, 1]
kernel = [3, 3, 3]
stride = [2, 2, 1]
inp = [128, 128, 32]
out = []

for i in range(3):
    out.append(((inp[i] + (2 * padding[i]) - kernel[i]) / stride[i]) + 1)

print(out)
