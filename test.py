import torch
import torch.nn as nn

w = torch.empty(3, 5)
print(w)
print(nn.init.normal_(w))
nn.init.normal_(w)
print(w)

