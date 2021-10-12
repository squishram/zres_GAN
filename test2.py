import torch

print(torch.version.cuda)
print(torch.__version__)
torch.zeros(1).cuda()

