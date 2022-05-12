import numpy as np

x_len = 3
y_len = 4
z_len = 5

# arrays = []
# for idx, length in enumerate([x_len, y_len, z_len]):
#     arr = np.array(range(length))
#     arr = np.broadcast_to(arr, (z_len, y_len, x_len))
#     if idx != 0:
#         arr = arr.transpose(idx - 1, 2)
#     print(arr)
#     arrays.append(arr)

arr = np.indices((x_len, y_len, z_len))
for i in range(3):
    print(arr[i])
