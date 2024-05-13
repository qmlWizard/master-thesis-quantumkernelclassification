import torch

torch.device('cuda')

print(torch.cuda.get_device_name(0))